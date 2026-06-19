---
title: "Why Concurrency Is Hard (and Why You Can't Avoid It)"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The one bug behind almost every concurrency failure, why your CPU stopped getting faster on its own, and the map you need to navigate the rest of this series."
tags:
  [
    "concurrency",
    "parallelism",
    "multithreading",
    "race-condition",
    "systems-programming",
    "fundamentals",
    "scalability",
    "mental-model",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-1.png"
---

A payments team I worked with once shipped a feature that let two devices push money into the same account at the same time — a phone tap and a watch tap, near-simultaneous. The code was three lines. Read the balance, add the deposit, write the balance back. In testing it was flawless. In production, once in maybe every fifty thousand double-deposits, the account ended up richer by exactly one deposit instead of two. A customer deposited two payments of forty dollars each; the balance went up by forty, not eighty. The forty dollars did not vanish into thin air — the money moved — but the ledger said otherwise, and a ledger that disagrees with reality is the worst kind of bug a financial system can have.

Nobody had written a bug, exactly. Each of the three lines was correct. The function was correct *when run alone*. It became wrong only when two copies of it ran at the same time and their steps interleaved in one specific, rare order. That is the whole subject of this series in one paragraph: **code that is correct in isolation can be catastrophically wrong when it runs concurrently**, and the reason is almost never the line you are staring at. The reason lives in the spaces *between* the lines — the moments where the scheduler can step away from one thread and let another run.

This post is the entry point to a 34-part series, "Concurrency & Parallelism, From the Ground Up." It is the map and the spine. By the end of it you will be able to say, precisely and without hand-waving, what concurrency *is* (and how it differs from parallelism); why you can no longer avoid it even if you wanted to; what the actual enemy is (it is nondeterminism, not threads); why shared mutable state is the root hazard behind nearly every concurrency bug; the three guarantees — atomicity, visibility, ordering — you must establish over every shared access; and the difference between latency and throughput, which are two different goals demanding two different tools. The figure below is the map of the entire series: one hazard at the root, and the four families of techniques that tame it. We will return to this map at the end of every post.

![the concurrency series spine map showing one shared state hazard branching into mutual exclusion non-blocking and message-passing then structured concurrency](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-1.png)

The promise of the series is not that you will memorize APIs. It is that you will be able to look at any piece of concurrent code and answer three questions: *What is shared and mutable here? What establishes the order of accesses to it? What does that ordering mechanism cost?* Those three questions are the discipline. Everything else — locks, atomics, channels, actors, async, structured concurrency — is just a menu of ways to answer the middle question cheaply. Let us begin by getting the vocabulary exactly right, because the single most common source of confusion in this field is treating two different words as if they meant the same thing.

## Concurrency Is Not Parallelism

Here is the one-sentence distinction, and it is worth committing to memory because almost everything downstream depends on getting it right:

> **Concurrency is about *structure* — dealing with many things at once. Parallelism is about *execution* — doing many things at once.**

Rob Pike, one of the designers of Go, put it as "concurrency is not parallelism," and the slogan is precise rather than cute. Concurrency is a property of how your program is *written and composed*: it has multiple independent logical flows of control — tasks, goroutines, coroutines, threads — that can make progress out of order, that can be paused and resumed, that interleave. Parallelism is a property of how your program *runs on physical hardware*: at one instant in real time, two or more of those flows are literally executing simultaneously on two or more processing units.

You can have one without the other, and recognizing all four quadrants kills most of the confusion in this field at once. A single-core machine running an operating system is *concurrent* — dozens of processes make progress, the scheduler slices the one core between them so finely that they appear simultaneous — but it is not *parallel*, because at any given nanosecond exactly one instruction stream is on that core. Conversely, a tight numerical loop spread across eight cores with no coordination, no shared mutable state, nothing to interleave, is *parallel* but barely *concurrent* in the interesting sense: there is no scheduling drama, no question of order, just eight cores grinding eight independent chunks. The hard, interesting, paged-at-3-AM territory is where you have both: many logical flows *and* real simultaneity, sharing state.

![sequential and concurrent execution on one core compared with parallel execution on two cores](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-2.png)

The figure contrasts the two ideas with the same pair of tasks. On the left, concurrency on a single core: task A runs for a slice, the runtime pays a small cost to switch, task B runs, and they finish one after the other in wall-clock time — interleaved, not simultaneous. On the right, parallelism on two cores: A and B run at the same instant and finish in roughly half the wall-clock time. Concurrency *organized* the work into two flows; parallelism *executed* them at once. The crucial insight is that concurrency is the enabling structure — once your program is decomposed into independent flows, the runtime is *free* to run them in parallel if hardware allows, but the correctness questions exist the moment you have concurrency, with or without parallelism.

Why does the distinction matter so much in practice? Because it tells you which problem you actually have, and therefore which tool to reach for.

- If your bottleneck is **waiting** — for the network, the disk, a database, another service — you have an I/O-bound problem, and the answer is *concurrency*: structure the program so that while one task waits, another runs. You may need exactly one core for ten thousand simultaneous network connections, because the cores are idle most of the time anyway. This is the world of async, event loops, and lightweight tasks.
- If your bottleneck is **computing** — hashing, encoding, matrix math, parsing gigabytes — you have a CPU-bound problem, and the answer is *parallelism*: split the work across cores so more arithmetic happens per second. Here you genuinely need multiple cores, and adding more "tasks" than you have cores buys you nothing but scheduling overhead.

Reaching for the wrong one is a classic and expensive mistake. Adding threads to an I/O-bound workload often makes it slower (more context switches, more lock contention, the same idle cores). Adding async to a CPU-bound workload does nothing at all (a single event loop still computes on one core; the work doesn't get faster by being interleaved). We will spend a whole post (the next one) on this split and the scaling laws that govern it. For now, hold the one sentence: **concurrency is structure; parallelism is execution.** And here is the matrix that pins down every cell of the comparison.

![a comparison grid of concurrency versus parallelism across definition goal needs and example rows](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-4.png)

Two definitions before we go on, because they recur constantly. A **process** is a running program with its own private address space — its own memory, isolated from other processes by hardware. A **thread** is a flow of execution *inside* a process; multiple threads in one process *share* that process's address space, which is exactly why they can corrupt each other's data and exactly why threads are powerful and dangerous. A process talks to another process only through explicit channels (pipes, sockets, shared-memory segments you opt into); threads share everything by default. That default — *shared by default* — is the seed of the entire hazard we are about to dissect. Hold that thought; we will earn it in the section on shared mutable state.

## The End of the Free Lunch

For about forty years, programmers got faster code for free, and most of them never noticed because they never had to do anything. You wrote your single-threaded program, you waited eighteen months, you bought a new machine, and the same binary ran roughly twice as fast. That was the free lunch, and it had a name behind it: **Dennard scaling**, the observation (Robert Dennard, 1974) that as transistors got smaller, their power density stayed constant, so you could clock them faster without melting the chip. Smaller transistors, higher clock speeds, more instructions per second, faster software — automatically.

That ended around 2005. Not because Moore's Law stopped — transistor counts kept doubling for years after — but because Dennard scaling broke. Below a certain feature size, current leakage and power density stopped cooperating; cranking the clock higher meant the chip ran hotter than you could cool. Clock speeds, which had climbed from megahertz to nearly four gigahertz over two decades, simply *flattened*. A desktop CPU in 2005 ran at roughly three gigahertz. A desktop CPU today runs at roughly three to five gigahertz. Twenty years, and single-thread clock speed barely moved.

![single thread clock speed flat after 2005 while core counts keep rising](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-6.png)

So what did the chip designers do with all those extra transistors Moore's Law kept handing them, if they couldn't spend them on clock speed? They spent them on *more cores*. The figure tells the story: before 2005, clock frequency rose and you got speed for free on one core; after 2005, frequency went flat and core count took off — dual-core, quad-core, then eight, sixteen, sixty-four, and server parts north of a hundred and twenty-eight cores today. The free lunch did not just end; it was replaced by a buffet you have to *cook yourself*. The extra performance is still there, sitting in those cores — but a single-threaded program uses exactly one of them and leaves the rest idle. On a 64-core machine, sequential code captures less than two percent of the silicon's throughput.

This is why you cannot avoid concurrency. It is not a fashionable technique or a resume keyword; it is the *only remaining way to make compute-bound code faster on modern hardware*. The hardware industry made a deal on your behalf around 2005: you get exponentially more transistors, but you have to learn to split your work across cores to use them. Herb Sutter wrote the canonical essay on this in 2005 — "The Free Lunch Is Over" — and two decades later the bill has fully come due. Every performance-sensitive system, every database, every game engine, every ML training run, every high-throughput service, is concurrent because it has no choice.

And there is a second, independent reason you can't avoid it, having nothing to do with raw speed: **responsiveness**. The instant your program has to do two things where one of them might block — serve a user *and* wait on a slow database, render a frame *and* download an asset, handle ten thousand network connections that each spend most of their life idle — you need concurrency to keep one task from freezing the others. A web server that handled one request at a time, fully, before touching the next would be unusable at any real load. So the pressure comes from both directions: parallelism for throughput on many cores, and concurrency for responsiveness under blocking I/O. There is no modern, performant, responsive software that is purely sequential. The lunch is over for everyone.

#### Worked example: the idle cores

Suppose you have a CPU-bound batch job that takes 80 seconds on a single core, and you run it on a 32-core server. Sequentially, it takes 80 seconds and pins exactly one core — the other 31 sit at zero percent the entire time. If the job is perfectly splittable into 32 independent chunks, parallel execution finishes in 80 / 32 = 2.5 seconds, a 32× speedup, and the machine you already paid for finally earns its keep. The difference between 80 seconds and 2.5 seconds is not better hardware, a faster language, or a cleverer algorithm. It is the same hardware, the same algorithm, the same arithmetic — merely *arranged* to run concurrently and *executed* in parallel. That 32× is the free lunch, except now you have to write the code to claim it. (Real workloads are never perfectly splittable; the next post derives exactly how much of that 32× you actually get, and it is humbling.)

## The Real Enemy Is Nondeterminism

Most engineers, when they first hit a concurrency bug, blame threads. Threads are not the enemy. The enemy is **nondeterminism**: the fact that a concurrent program can produce *different results on different runs of the identical code with the identical input*, depending on the precise, uncontrollable timing of how the scheduler interleaved the threads. A sequential program is a function — same input, same output, every time, forever. A naively concurrent program is a *relation* — same input, a *set* of possible outputs, and which one you get this run depends on cosmic-ray-level details: cache state, interrupt timing, what else the OS happened to be doing, the phase of the moon.

This is what makes concurrency bugs uniquely vicious, and it is worth being precise about *why*. Three properties compound:

1. **They are timing-dependent, so they are rare and load-dependent.** The bad interleaving requires the scheduler to switch threads at one specific instruction boundary. On a quiet test machine that window almost never opens, so the bug "doesn't reproduce." Under production load — more threads, more contention, more context switches — the window opens far more often. The bug that fired once in fifty thousand double-deposits in our opening story fired *zero* times in a thousand-iteration test suite. The test suite was not wrong; the bug's probability per run was simply below the test's resolution.

2. **They resist debugging by observation.** Attach a debugger, add a `print`, single-step — and you change the timing. The act of observing slows one thread relative to another, which closes the bad window. This is the famous "Heisenbug": the bug vanishes the moment you look at it. You cannot reliably reproduce on demand what only happens at one timing you cannot control.

3. **They violate the reasoning habit of every engineer trained on sequential code.** When you read sequential code, you trace it line by line, top to bottom, and that trace is the truth. When you read concurrent code, *that trace is one of thousands of possible traces*, and the bug lives in a trace you did not think to enumerate. Correctness is no longer "does this line do the right thing" but "is there *any* interleaving of all these lines, across all these threads, that does the wrong thing." That is a combinatorial explosion, and your brain is not built for it.

The number of possible interleavings is genuinely astronomical. Two threads of just ten instructions each have $\binom{20}{10} = 184{,}756$ distinct interleavings. Three threads of ten instructions each have $20!/(10!)^3 \approx 5.6 \times 10^{11}$ interleavings. A real service with dozens of threads and thousands of instructions per critical path has more possible interleavings than there are atoms in the galaxy. You cannot test them all. You cannot even enumerate them. **The only viable strategy is to *constrain* the set of allowed interleavings to those you can prove are all correct** — which is precisely what every synchronization mechanism in this series does. A lock does not make your code faster; it makes the set of possible interleavings *smaller and provably safe*. So does an atomic, a channel, an actor, an immutable value. The whole game is interleaving reduction.

This reframing is the single most important idea in the series, so let me state it as a principle you can carry into every later post: **you do not fix concurrency bugs by being careful; you fix them by removing interleavings.** "Being careful" is a strategy that scales to zero — there is always an interleaving you didn't foresee. Removing interleavings — by serializing access, by establishing an order, by not sharing in the first place — is a strategy that scales, because it shrinks the space of behaviors down to a set you can actually reason about. Keep that distinction sharp. Every tool we study is an interleaving-reduction device with a particular cost and a particular failure mode.

## Shared Mutable State Is the Root Hazard

Now we can name the villain exactly. Nondeterministic interleaving is only dangerous when threads *share mutable state* — memory that more than one thread can both read and write. Decompose that phrase, because all three words are load-bearing:

- **Shared**: more than one thread can reach it. A local variable on one thread's stack is not shared; no other thread has a pointer to it. A global counter, a field of a heap object two threads both hold a reference to, a slot in a shared array — those are shared.
- **Mutable**: it can change after creation. A value that is written once and only ever read afterward is effectively a constant; no interleaving of reads can corrupt it. The danger is *writes* racing with reads or other writes.
- **State**: it is data whose value matters to correctness — a balance, a count, a flag, a pointer, the length field of a list.

Remove any one of the three and the hazard disappears. **Not shared** (thread-local data, or one thread that owns the data exclusively) — safe. **Not mutable** (immutable values, functional data structures, copy-on-write) — safe; you can share an immutable value across a thousand threads and never need a lock, because there are no writes to race. **No state** (pure stateless computation) — safe. This is not a footnote; it is the deepest lever in the entire field. The cheapest synchronization is the synchronization you don't need, and you don't need it when there is nothing shared *and* mutable. Whole families of concurrency models — actors, channels, persistent immutable data structures, Rust's ownership system — are built around *attacking one of these three words* so the hazard cannot arise. We will meet all of them.

But when you *do* have shared mutable state, here is exactly how it bites. Let's walk the canonical example — incrementing a shared counter — at the instruction level, because this single mechanism is the seed of nearly every race you will ever debug. Consider this innocent line, in any language:

```go
// Two goroutines both run this on the same shared variable.
count = count + 1
```

It looks like one indivisible operation. It is not. The CPU cannot add to a memory location in one step on most architectures; it must bring the value into a register, modify it, and write it back. So that one line compiles to (at least) three machine operations — a **read-modify-write** sequence:

```c
/* What `count = count + 1` actually becomes, roughly. */
r1 = LOAD count;     /* step 1: read the current value into a register   */
r1 = r1 + 1;         /* step 2: compute the new value in the register    */
STORE count = r1;    /* step 3: write the register back to memory         */
```

Three steps. And between any two of them, the scheduler can suspend this thread and run another. That gap is the entire bug. Suppose `count` starts at 0 and two threads, T1 and T2, both execute the three steps. If they run one fully before the other, you get 2, as intended. But the scheduler is under no obligation to do that. Watch the lethal interleaving.

![the lost update interleaving where two threads load zero each add one and the second store overwrites the first leaving the counter at one](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-3.png)

The timeline above is the lost-update race, step by step. T1 loads `count`, reading 0. *Before T1 can store its result*, the scheduler switches to T2, which also loads `count` — and reads the same stale 0, because T1 hasn't written back yet. Now both threads are holding the value 0 in their private registers. T1 computes 0 + 1 = 1 and stores 1. T2 computes 0 + 1 = 1 and stores 1, *overwriting* T1's store with the identical value. The counter ends at 1. Two increments happened; one was lost. The +1 from T1 was silently erased by T2's store because T2 was working from a snapshot taken before T1's write landed. That is the exact bug from our opening story, with "balance" in place of "count" and "+= deposit" in place of "+= 1." Money did not vanish — an *update to the ledger* vanished, which is just as bad.

This is a **race condition**: the outcome depends on the relative timing (the "race") between threads. More specifically it is a **data race** — two threads access the same memory location concurrently, at least one of them writes, and there is no synchronization ordering the accesses. (Data race and race condition are not synonyms — a precise post later in this series pulls them apart — but every data race is a hazard, and this one is the prototype.) The fix is to make the read-modify-write *atomic* — indivisible, so no other thread can observe or interpose between its steps. We will see two ways to do that (a lock that serializes the whole sequence, and a single hardware atomic instruction) in the code section. But first, two extensions, because the counter is the simplest instance of a far broader pattern.

The first extension is the **check-then-act race**, which shows that the hazard does not require an obvious read-modify-write — it appears any time you *check* a shared condition and then *act* on it, with a gap in between. Consider lazy initialization, one of the most-written and most-broken patterns in all of software:

```java
class Lazy {
    private Resource instance = null;

    Resource get() {
        if (instance == null) {      // CHECK: is it built yet?
            instance = new Resource(); // ACT: build it
        }
        return instance;
    }
}
```

Sequentially flawless; concurrently, a textbook bug. Two threads call `get()` at the same time. Both run the `if (instance == null)` check, both see `null` (because neither has built it yet), and both proceed to `new Resource()` — so the resource is constructed *twice*, the second assignment clobbers the first, and any thread that grabbed the first instance now holds an object the system thinks doesn't exist. If `Resource` is something with side effects (opening a file, registering a callback, allocating a connection), you've now done it twice and leaked one. The gap between the check and the act is the same gap that destroyed the counter — the scheduler can switch threads inside it. The same shape appears everywhere: "if the file doesn't exist, create it," "if the cache is empty, populate it," "if the user has balance, deduct it." Every one of those is a check-then-act, and every one is a race unless the check and the act are made atomic together (a lock, or a built-for-it primitive like a double-checked `volatile` with proper ordering, or a concurrent map's `computeIfAbsent`). The lost-update race teaches that *increments* aren't atomic; the check-then-act race teaches the more general truth that *any decision based on shared state, followed by an action, is a race unless the decision and the action happen as one*.

The second extension is the one our opening story exemplified: the counter is *every* aggregate piece of shared mutable state. A bank balance is a counter. A reference count (how garbage collectors and smart pointers track liveness) is a counter — and a racy reference count means use-after-free or a memory leak, depending on which way it loses. A queue's length, a connection pool's available-permits count, a rate limiter's token bucket, a metrics hit counter, an inventory's units-in-stock — all counters, all subject to the exact lost-update interleaving you just watched. When you understand the three-instruction race on one integer, you understand the failure mode of a vast swath of production systems. Now name what was actually missing, because it generalizes to all of them.

## The Three Things You Actually Need

When you reason about a single shared variable accessed by multiple threads, you need *three distinct guarantees*, and they are genuinely independent — you can have one without the others, and dropping each one produces a *different* bug. Naming them precisely is what separates engineers who guess at `volatile` and `synchronized` from engineers who know exactly which guarantee they're buying.

![a grid showing atomicity visibility and ordering with the distinct bug each one prevents](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-7.png)

**1. Atomicity.** A multi-step operation must run as an indivisible whole — all of it, or none of it, with no other thread observing a partial state. The counter race is an *atomicity* failure: the read-modify-write was three steps, and another thread interposed in the middle. Atomicity is about *indivisibility of a compound action*. A lock buys it by making the critical section run with mutual exclusion; an atomic CPU instruction buys it for one operation by doing the read-modify-write in hardware as a single uninterruptible step.

**2. Visibility.** When one thread writes a value, *when* (and whether) do other threads see that write? On a modern CPU, the answer is *not necessarily ever*, absent synchronization. Each core has store buffers and private caches; a write may sit in the writing core's store buffer, visible to that core but invisible to others, for an unbounded time. The classic visibility bug is a flag: one thread sets `done = true`, another loops `while (!done)`, and the loop *never exits* because the second thread keeps reading a stale cached `false` — the write is "done," but it never became *visible*. Visibility is about *propagation of writes between cores*, and it is governed by the memory model, not by whether your code "ran." This is why `volatile` exists in Java and C++, why `count++` even when made atomic still needs care, and why "it set the flag, I saw it in the debugger" can be a lie on the other thread.

**3. Ordering.** Different threads can disagree about the *order* in which writes happened, because compilers and CPUs reorder memory operations for speed — as long as each thread's *own* view stays consistent with its program order, the hardware is free to let *other* threads observe writes in a different order. So thread A writes X then Y; thread B may legally see Y before X. The infamous bug here is publishing an object: thread A allocates an object, fills its fields, then stores the pointer; thread B reads the pointer and then reads the fields — and on a weakly-ordered CPU, B can see the new *pointer* but the *old, garbage fields*, because the field writes and the pointer write got reordered relative to B's view. Ordering is about *the relative order of distinct writes as seen by another thread*, and it is the subtlest of the three because it depends on the hardware's memory model — x86 (Total Store Order) is relatively strong, ARM and POWER are weak and will surface reorderings that x86 hides.

The unifying concept that gives you all three at once is **happens-before**: a partial order over memory operations such that if operation A *happens-before* operation B, then A's effects are atomic, visible, and ordered with respect to B. Every correct synchronization mechanism — a lock acquire/release, an atomic with the right memory ordering, a channel send/receive, a thread join — establishes happens-before edges. The entire discipline of concurrency reduces to: **for every pair of conflicting accesses to shared mutable state, establish a happens-before order between them.** If you can't draw that edge, you have a bug, full stop. We will define happens-before formally in the memory-model post; for now, internalize that "make it correct" means "establish happens-before," and the three guarantees above are what happens-before delivers.

| Guarantee | One-line meaning | Bug when missing | Bought by |
| --- | --- | --- | --- |
| Atomicity | Compound op is indivisible | Lost update (counter off by one) | Lock, atomic instruction |
| Visibility | My write is seen by others | Stale read (flag never observed) | `volatile`, lock release/acquire, memory fence |
| Ordering | Writes seen in a sane order | Reorder bug (see half an object) | Acquire/release, fences, lock |

The table above is one you will reach for again and again. Notice that a plain lock, used correctly, buys *all three* — which is why locks are the workhorse and why beginners reach for them first. The trickier mechanisms (atomics with relaxed ordering, lock-free structures) deliberately buy *less* — only the guarantee you actually need — in exchange for speed, which is exactly why they are dangerous in unpracticed hands.

#### Worked example: the visibility bug that runs forever

The counter race is an atomicity failure; here is a *visibility* failure, which is sneakier because the code looks obviously correct and the bug never even prints a wrong answer — it just hangs. One thread does work and then sets a flag; another thread spins until it sees the flag. In Java:

```java
class Stopper {
    boolean done = false; // NOT volatile — this is the bug

    void worker() {
        while (!done) {
            // spin, doing nothing, waiting for done to flip
        }
        // we expect to reach here once the other thread sets done = true
    }

    void stopper() {
        done = true; // set the flag from another thread
    }
}
```

Read it sequentially and it's plainly right: one thread sets `done = true`, the other's `while (!done)` exits. Run it on a real multi-core JVM, with the worker spinning hot, and the worker can loop *forever*. Why? Because `done` is not `volatile`, the JIT is permitted to *hoist* the read out of the loop — it reads `done` once, sees `false`, and rewrites the loop as `while (true)` since nothing *in this thread* changes `done`. Even without that optimization, the writing thread's `done = true` can sit in its store buffer or local cache and never propagate to the spinning core. The write happened; it never became *visible*. The fix is one keyword — `volatile boolean done` — which forces every read to go to memory and every write to be published, establishing the happens-before edge. This is the canonical proof that *visibility is a separate guarantee from atomicity*: nothing here is a compound operation, no update is lost, and yet the program is broken. (The same trap in Go is fixed with `sync/atomic` or a channel; in C++ with `std::atomic<bool>`; in Rust the type system again won't let you share a plain `bool` mutably across threads at all.) The deeper, formal treatment — why the JIT is *allowed* to do this, and what the memory model actually promises — is the subject of the memory-model post; this is the symptom you'll recognize in the wild: a spin loop that never exits because a flag write went unseen.

## Latency vs Throughput: Two Goals, Two Tools

Before we lay out the arc of the series, one more pair of words people constantly conflate, because conflating them leads you to optimize the wrong thing and build the wrong system. **Latency** is how long a *single* operation takes from start to finish — the time one request waits. **Throughput** is how *many* operations complete per unit time — requests per second. They are related but distinct, and improving one can *harm* the other.

The mechanical relation between them is **Little's Law**, which holds for any stable system: $L = \lambda W$, where $L$ is the average number of in-flight operations (concurrency), $\lambda$ is the throughput (arrival/completion rate), and $W$ is the average latency (time in system). Read it as: to push throughput $\lambda$ up while latency $W$ is fixed, you must increase $L$ — keep more operations in flight at once — which is exactly what concurrency lets you do. A server handling requests that each take 100 ms ($W = 0.1$ s) at one-at-a-time concurrency ($L = 1$) tops out at $\lambda = L/W = 10$ requests per second. Run 100 in flight ($L = 100$) and the *same* per-request latency yields $\lambda = 1000$ requests per second — a 100× throughput gain with *zero* improvement to any individual request's latency. Concurrency bought throughput by overlapping the waiting, not by making any single request faster.

This is why latency and throughput need different tools:

- **To reduce latency** of a single compute-heavy operation, you parallelize *within* the operation — split one request's work across cores so it finishes sooner. This makes one thing faster.
- **To increase throughput**, you process more operations concurrently — overlap their waiting (for I/O-bound work) or pack more onto more cores (for CPU-bound work). This makes more things finish per second, and may not touch the latency of any one of them.

These can actively trade off. Batching requests together raises throughput (amortized overhead, better cache use, fewer syscalls) but *raises* the latency of the first request in the batch (it waits for the batch to fill). Adding threads beyond your core count raises context-switch overhead, which can raise tail latency even as it nudges throughput. A reader who knows which one they're optimizing makes the right call; a reader who conflates them tunes a throughput system for latency and wonders why p99 got worse. Whenever a later post measures a "before → after," ask: *is this number latency or throughput?* The honest answer is usually "it improved one and you should check it didn't wreck the other."

## The Running Example: One Counter, Then the Whole System

This series uses a single example as its spine, deliberately, because seeing the *same* problem grow in difficulty teaches far more than a parade of unrelated toys. The example is the one you've already met: **a shared counter under concurrent updates**. It starts trivial — increment a number from many threads — and that triviality is the point: if you can't get *a single integer* right under concurrency, you have no business with anything harder. Almost every reader's first concurrency bug is exactly this counter, dressed up as a balance, a request count, a reference count, a queue length, or a hit counter.

But the counter is only the seed. Across the series it grows in three stages, and each stage introduces the family of techniques that stage demands:

- **Stage 1 — the shared counter (this post and the locks track).** One mutable integer, many writers. This is the purest form of the hazard, and it exercises *mutual exclusion* and *atomics*. Get the read-modify-write right and you understand atomicity, visibility, and ordering in their simplest setting. Everything later is this, scaled.
- **Stage 2 — the producer–consumer pipeline (the condition-variable and channel tracks).** Now the shared state is a *bounded queue*: producer threads put items in, consumer threads take them out, and the structure must block a producer when the queue is full and block a consumer when it's empty — without busy-waiting, without losing items, without deadlocking. This is where *condition variables*, *semaphores*, *channels*, and *backpressure* enter. The counter became a buffer, and "increment" became "enqueue/dequeue with flow control." The cross-process version of this exact problem — bounded queues, backpressure, who slows down when the consumer can't keep up — is owned by the message-queue series; see [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control), and we link out rather than re-derive it.
- **Stage 3 — the connection server (the async, event-loop, and structured-concurrency tracks).** Finally the pipeline becomes a *server*: thousands of clients connect, each connection a small state machine that mostly waits on the network, and the server must serve them all on a handful of cores without a thread per connection melting the box. This is the C10k problem (ten thousand concurrent connections), and it forces *event loops*, *async/await*, *coroutines*, and *structured concurrency* — the non-blocking and avoid-sharing families in their natural habitat. The counter, by now, is a per-connection or per-shard piece of state, and the discipline you learned on one integer is what keeps ten thousand connections correct.

Notice the through-line: at every stage the question is identical — *what is shared and mutable, what orders the accesses, what does that cost?* — and only the answer's complexity grows. The shared integer becomes a shared queue becomes a shared server. The mechanism escalates from a single lock to condition variables to channels to event loops. But the discipline never changes. That is why one example carries the whole series: master the counter and you have the template; everything else is the template under more pressure. Keep this staircase in mind — counter, pipeline, server — as we lay out the map.

## The Arc of This Series

Now the map. Everything in concurrency is a strategy for taming the same root hazard — shared mutable state under nondeterministic scheduling — and the strategies fall into four families, in roughly increasing order of how aggressively they *avoid* the hazard rather than merely *manage* it. The series follows this arc, and the spine figure from the intro is exactly this progression. Here is the layered view of what sits under your code, because every layer in it can reorder or interleave your supposedly-simple line.

![the layers from your code down through the language runtime os threads cpu cores and main memory](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-5.png)

That stack is *why* concurrency is hard: your one line of source passes through a runtime that may JIT and reschedule it, an OS that may preempt it mid-instruction-sequence, CPU cores with private store buffers and caches that may delay and reorder its memory effects, all converging on one shared address in memory. Each layer is optimizing for *its own* speed, and those optimizations are invisible and individually correct — but together they mean your code does not execute the way you read it.

The OS-threads layer deserves a closer look, because it is where the nondeterminism is born. The operating-system scheduler runs threads in time slices — typically a few milliseconds each — and at the end of a slice, or whenever a thread blocks (on I/O, a lock, a sleep), it performs a **context switch**: it saves the running thread's registers and program counter to memory, picks the next runnable thread from a run queue, and restores *its* saved state. That switch is not free. The direct cost — saving and restoring registers, updating kernel bookkeeping — is on the order of one to a few microseconds. The *indirect* cost is often larger: the new thread's working set isn't in the cache or the TLB, so it suffers a flurry of cache and TLB misses as it warms back up, which can cost many more microseconds of stalled execution. A useful order-of-magnitude figure (and it is approximate — it depends heavily on the CPU, the working-set size, and whether the switch crosses to a different core) is *single-digit microseconds direct, with indirect cache effects sometimes pushing the effective cost well past ten*. This is precisely why "just spawn more threads" stops helping past a point: when you have far more runnable threads than cores, the cores spend an increasing share of their time *switching between threads* instead of *running* them, and throughput bends down. It is also why lightweight tasks — goroutines, coroutines, green threads, virtual threads — exist: they multiplex many logical flows onto a few OS threads, switching in user space at a fraction of the cost (tens of nanoseconds rather than microseconds), because no kernel transition or full register/TLB churn is required. The scheduler is the source of the interleaving *and* a major cost center, and several later posts are about working *with* it rather than against it.

The crucial consequence of a preemptive scheduler is this: **a context switch can land between any two machine instructions** — including between the load and the store of our counter's read-modify-write. The scheduler does not know or care that those three instructions "belong together"; nothing told it they form an atomic unit, because nothing did. That is the mechanical root of the lost-update race: preemption at an inconvenient instruction boundary, on state that two threads share. Every synchronization primitive is, at bottom, a way to tell some layer of this stack "these accesses must be ordered" — a lock tells the OS to block other threads, an atomic tells the CPU to do the read-modify-write as one uninterruptible step, a memory fence tells the core to stop hiding its reordering. With that mechanism named, the four families of the series each impose order on this chaos differently:

**Family 1 — Mutual exclusion (locks).** The most direct answer: if only one thread can touch the shared state at a time, no bad interleaving exists. We build the mutex from hardware (`test-and-set`, compare-and-swap), study critical sections, then meet the bugs locks *cause* — deadlock (the four Coffman conditions and how to break the circular wait), livelock, priority inversion, lock contention — and the memory model underneath that makes "the lock worked" actually mean something. Tracks B, C, and D of the series. Locks buy all three guarantees, simply, at the cost of serializing and the risk of deadlock.

**Family 2 — Non-blocking (atomics, async, lock-free).** Locks make threads *wait*; sometimes you can't afford that. So we study mechanisms that make progress without blocking: atomic instructions and the memory orderings (`relaxed`, `acquire`, `release`, `seq_cst`); compare-and-swap loops; lock-free and wait-free data structures and the brutal subtleties (the ABA problem, hazard pointers, memory reclamation); and on the I/O side, the event loop, async/await, coroutines, and how a single thread juggles ten thousand connections. These buy *exactly the guarantee you need and no more*, for speed, at the cost of being far harder to get right.

**Family 3 — Avoid sharing (message-passing, actors, CSP, STM, immutability).** The deepest fix: if threads don't share mutable state, the hazard can't arise. We study channels and CSP (Go's `select`, the "share memory by communicating" philosophy), the actor model (Erlang/Elixir, each actor a private state behind a mailbox), software transactional memory (Clojure, Haskell — optimistic concurrency that retries on conflict), and immutability/persistent data structures. These attack the *shared* and *mutable* words directly, trading some overhead and a different failure mode (mailbox overflow, transaction retry storms) for the elimination of data races by construction.

**Family 4 — Structured concurrency.** Finally, making concurrency *composable and sane*: task scopes with bounded lifetimes, structured cancellation, so that "spawn a task" has the same disciplined nesting as a function call — no leaked goroutines, no orphaned tasks, propagated errors, deterministic shutdown. This is the youngest family (Trio, Kotlin's `coroutineScope`, Java's structured concurrency, Swift), and it ties the whole series together.

![a taxonomy of concurrency models grouped under guard shared state avoid sharing and never block](/imgs/blogs/why-concurrency-is-hard-and-why-you-cant-avoid-it-8.png)

The taxonomy figure groups the models you'll meet by *which of the three words they attack*. Guard the shared state (locks, atomics) — keep sharing and mutation but serialize access. Avoid sharing (actors, channels, STM, immutability) — remove the *shared* or the *mutable*. Never block (async, event loops) — keep one thread busy across many waiting tasks. The capstone post, [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model), turns this taxonomy into a decision procedure: given your problem, which model do you reach for? Every intermediate post links back to this map so you always know where you are. And throughout, three sibling series own the adjacent territory — the Python-specific concurrency story lives in [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) and [threading done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits), the architecture-scale view of flow control lives in [consistency models for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects), and cross-process backpressure lives in [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control). We link out rather than re-derive.

## From Bug to Fix: The Counter, in Code

Theory is cheap; let's make the hazard and its fixes concrete, in real code, in more than one language, because the *idioms diverge sharply* and seeing the divergence teaches the concept better than any single language can. We will take the racy counter and fix it three ways: a lock, an atomic, and (the deepest fix) by not sharing at all.

First, the **bug**, in Go. Launch a thousand goroutines, each incrementing a shared counter a thousand times. The correct answer is one million.

```go
package main

import (
	"fmt"
	"sync"
)

func main() {
	var count int
	var wg sync.WaitGroup
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				count = count + 1 // DATA RACE: read-modify-write, unsynchronized
			}
		}()
	}
	wg.Wait()
	fmt.Println(count) // not 1000000; some lower, nondeterministic number
}
```

Run this and you will almost never see 1000000. You'll see 743912, then 821004, then 998877 — a different wrong number every run, all below a million, because increments collide and get lost exactly as the timeline figure showed. Go ships a race detector for precisely this: run `go run -race main.go` and it prints a stack trace pinpointing the unsynchronized read and write on `count`. (Use it. It is the cheapest concurrency-bug insurance in any ecosystem, and we devote a later post to dynamic race detection.)

The **lock fix** — serialize the read-modify-write so it's atomic:

```go
var (
	count int
	mu    sync.Mutex
)
// inside the goroutine loop:
mu.Lock()
count = count + 1 // critical section: only one goroutine here at a time
mu.Unlock()
```

`mu.Lock()` blocks until this goroutine holds the lock; while it holds it, no other goroutine can enter; `mu.Unlock()` releases. The three-step read-modify-write now runs with no other thread interposing — atomicity restored. The lock release also establishes happens-before, so the write is visible and ordered for the next thread that acquires. All three guarantees, from one primitive. It prints 1000000 every time. The cost: threads serialize on `mu`, so under heavy contention this counter does not scale — every increment waits for the lock. (When the lock *is* your bottleneck, you go lock-free; that's a whole later post, and the measurement section below shows the contention curve.)

The **atomic fix** — buy atomicity from the hardware, no lock:

```go
import "sync/atomic"

var count int64
// inside the goroutine loop:
atomic.AddInt64(&count, 1) // one indivisible read-modify-write instruction
```

`atomic.AddInt64` maps to a single atomic add instruction (on x86, a `lock xadd`); the CPU performs the read-modify-write as one uninterruptible step, and the cache-coherence protocol makes the result visible. No software lock, no blocking — just hardware atomicity. It prints 1000000 and is typically faster under contention than the mutex, because there's no lock to acquire and release.

Now the same fix in **Rust**, where the type system makes the bug *impossible to compile*. This is "fearless concurrency": Rust's ownership and its `Send`/`Sync` marker traits mean you literally cannot share a plain mutable integer across threads — the borrow checker rejects it before runtime. To share, you must wrap it in a thread-safe container, and the container *is* the synchronization:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::thread;

fn main() {
    // Arc = atomically reference-counted shared pointer; AtomicI64 = lock-free counter.
    let count = Arc::new(AtomicI64::new(0));
    let mut handles = vec![];
    for _ in 0..1000 {
        let c = Arc::clone(&count);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                c.fetch_add(1, Ordering::Relaxed); // atomic add; Relaxed is enough for a counter
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    println!("{}", count.load(Ordering::SeqCst)); // 1000000, always
}
```

Try to write the *bug* version in Rust — share a plain `i64` and `c += 1` across threads — and the compiler refuses: `i64` is not `Sync`, the closure can't capture a mutable reference into multiple threads, the borrow checker stops you. The race that *runs* and silently corrupts in Go is a *compile error* in Rust. That is a fundamentally different bargain: Go trusts you and gives you a detector; Rust distrusts you and gives you a proof. Note also the `Ordering::Relaxed` — Rust forces you to *name* the memory ordering you want, surfacing the visibility/ordering question that Go's `atomic.AddInt64` hides behind a sequentially-consistent default. For a pure counter, relaxed ordering is correct and cheapest (we don't depend on the order relative to *other* variables); choosing it is the kind of decision the atomics post teaches.

And in **Java**, the divergence again — three idioms, escalating:

```java
import java.util.concurrent.atomic.AtomicLong;

class Counter {
    // BUG: plain field, count++ is read-modify-write, races across threads.
    long racy = 0;

    // FIX 1: a lock via synchronized — serializes the whole method.
    long guarded = 0;
    synchronized void incGuarded() { guarded++; }

    // FIX 2: an atomic — lock-free hardware increment.
    final AtomicLong atomic = new AtomicLong(0);
    void incAtomic() { atomic.incrementAndGet(); }
}
```

`synchronized` is Java's built-in mutual exclusion (every object has an intrinsic lock); `AtomicLong.incrementAndGet()` is the lock-free atomic. Note what `volatile` would *not* do here: marking `racy` as `volatile` fixes *visibility* (other threads see the latest value) but **not atomicity** — `count++` is still three steps, still races, still loses updates. That trap (reaching for `volatile` to fix a counter) is one of the most common Java concurrency mistakes, and it exists precisely because atomicity and visibility are *different guarantees*, as the three-things section insisted.

The deepest fix of all is **don't share** — give each thread its own counter and sum at the end. This is the data-parallel reduction pattern, and it scales perfectly because there is no shared mutable state during the hot loop:

```rust
use std::thread;

fn main() {
    let mut handles = vec![];
    for _ in 0..8 {
        // Each thread owns a private local counter — nothing shared, no synchronization.
        handles.push(thread::spawn(|| {
            let mut local: i64 = 0;
            for _ in 0..1_000_000 {
                local += 1; // plain increment, no atomics, no lock — it's thread-private
            }
            local
        }));
    }
    let total: i64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    println!("{}", total); // 8000000, and it scales linearly with cores
}
```

No lock, no atomic, no contention — because there is no shared mutable state *until* the final sum, which happens once, after all threads finish. This is the lesson the whole "avoid sharing" family is built on: the fastest synchronization is the one you removed by not sharing. The `thread::join` at the end establishes happens-before (the spawned thread's writes are visible to the joiner), so reading each `local` is safe. Keep this pattern in your pocket; it is the answer surprisingly often.

## Case Studies / Real-World

These are not academic puzzles. The most consequential failures in computing history include concurrency bugs that were exactly the hazards above, scaled up to where they killed people, grounded spacecraft, or blacked out a continent.

**The Therac-25 radiation overdoses (1985–1987).** The Therac-25 was a radiation therapy machine. Between 1985 and 1987 it delivered massive radiation overdoses to at least six patients, several of whom died, in what is the canonical software-safety case study (Leveson & Turner, *IEEE Computer*, 1993). One of the two distinct failure modes was a **race condition**: a fast-typing operator could change the treatment parameters during a narrow window in which a shared variable was being read by another concurrent task, so the machine fired a high-power beam configured for a low-power mode. The bug was timing-dependent — it only manifested when the operator edited the screen faster than the setup task expected, which is exactly why it survived testing and only surfaced with experienced operators in the field. It is the lost-update hazard with lethal stakes: two flows of control, one shared mutable configuration, no synchronization, an interleaving nobody anticipated. (The failure is well-documented in the public investigation; the precise interleaving is reconstructed in Leveson's account.)

**The Mars Pathfinder priority inversion (1997).** Days after landing on Mars, the Pathfinder lander began experiencing total system resets, threatening the mission. The cause was **priority inversion** — a low-priority task held a lock (a mutex on a shared information bus) that a high-priority task needed; a medium-priority task, needing no lock, preempted the low-priority one and kept it from running, so the high-priority task stalled waiting on a lock that couldn't be released; a watchdog timer noticed the high-priority task missing its deadline and reset the system. The fix, uploaded to Mars, was to enable **priority inheritance** on that mutex (temporarily boost the lock-holder's priority to that of the highest waiter). This is a *liveness* failure — not a corrupted value but a thread that can't make progress — and it's a vivid reminder that locks introduce their own failure modes. We devote a full post to deadlock and its cousins (priority inversion, livelock) for exactly this reason. (Account by Glenn Reeves, JPL, who led the team that diagnosed it.)

**The Northeast blackout race condition (2003).** The August 2003 blackout that darkened much of the northeastern United States and Ontario — roughly 50 million people — was triggered in part by a software fault: a **race condition** in the alarm-and-event-processing system of FirstEnergy's control room (an XA/21 system from GE). A race in the multi-threaded alarm software caused it to stall silently; the operators were left blind to the cascading grid failures they should have been alerted to, and a manageable local fault cascaded into a continent-scale outage. The bug, per the subsequent investigation, was a latent race that had survived years of operation and surfaced under a specific load of simultaneous events — once again, a rare interleaving that testing never hit. (U.S.–Canada Power System Outage Task Force final report, 2004.)

| Incident | Year | Hazard class | One-line cause |
| --- | --- | --- | --- |
| Therac-25 | 1985–87 | Race condition | Operator edit raced a shared config variable |
| Mars Pathfinder | 1997 | Priority inversion (liveness) | Low-priority task held a lock the high-priority task needed |
| Northeast blackout | 2003 | Race condition | Alarm-system race stalled silently, blinded operators |

The pattern across all three is identical to our counter: **shared mutable state, concurrent access, an unsynchronized interleaving that testing never hit, manifesting only under real-world timing.** Different domains, decades apart, same root hazard. That is why this series spends so long on the mechanism — because the mechanism is universal, and the stakes are sometimes enormous.

## A Measured Demonstration: How to Know It's Real

The series' third mandate is *measured behavior* — never assert a number you haven't justified. So let's measure the counter race honestly and lay down the rules for honest measurement, because half of all concurrency benchmarks you'll read on the internet are wrong in one of a few predictable ways.

#### Worked example: the race failure rate and the contention curve

Take the Go racy counter (1000 goroutines × 1000 increments, target 1000000) and run it many times. On a multi-core machine you'll observe something like the table below — these are *representative orders of magnitude*, not measurements from your machine, and the exact numbers depend on cores, load, and the scheduler, so treat them as the *shape* of the result, not gospel:

| Config | Target | Typical result | Lost updates | Notes |
| --- | --- | --- | --- | --- |
| Racy, 2 goroutines | 2000 | ~1990–2000 | 0–10 | Few collisions; bug rarely fires |
| Racy, 1000 goroutines | 1000000 | ~700k–999k | 1k–300k | Heavy contention; many lost updates |
| Mutex, 1000 goroutines | 1000000 | 1000000 | 0 | Correct, but serialized |
| Atomic, 1000 goroutines | 1000000 | 1000000 | 0 | Correct, faster under contention |
| Per-thread sum, 8 threads | 8000000 | 8000000 | 0 | Correct, scales linearly |

Two lessons jump out. First, **the bug's visibility scales with contention** — with two goroutines it barely fires, with a thousand it's blatant. This is *exactly* why concurrency bugs hide in testing (low contention) and explode in production (high contention). Second, **correctness and speed are different axes**: the mutex and the atomic are both correct, but they have different throughput, and the per-thread version is both correct *and* fastest because it removed the sharing. A measurement that only checked correctness would miss that the mutex doesn't scale; a measurement that only checked speed would miss that the racy version is *wrong*. You must check both.

Now, the throughput-vs-threads picture, which every concurrency engineer must internalize: adding threads does **not** monotonically increase throughput. A shared-mutex counter's throughput *rises* as you add threads up to a point, then **bends downward** as lock contention dominates — threads spend more time fighting for the lock than doing work, and cache-line ping-pong between cores adds cost. The atomic version bends down later and more gently. The per-thread version scales near-linearly until you run out of cores. The single most important shape in applied concurrency is that **the throughput curve goes up, peaks, and comes back down** — there is an optimal thread count, and more is worse past it. We derive and measure that curve in the locks and lock-free posts; for now, know it exists and never assume "more threads = more speed."

There is a hard ceiling on parallel speedup that no amount of cores can beat, and it has a name: **Amdahl's Law**. If a fraction $p$ of your program's work is parallelizable and $(1-p)$ is inherently serial, then on $N$ processors the speedup is

$$S(N) = \frac{1}{(1-p) + \dfrac{p}{N}}$$

As $N \to \infty$, the parallel term $p/N \to 0$ and the speedup ceiling becomes $S_{\max} = \frac{1}{1-p}$ — set *entirely* by the serial fraction.

#### Worked example: the Amdahl ceiling

Suppose 95% of your work parallelizes ($p = 0.95$, serial fraction $1-p = 0.05$). On 8 cores, $S(8) = 1 / (0.05 + 0.95/8) = 1/0.16875 \approx 5.9\times$ — already well short of 8×. On 32 cores, $S(32) = 1/(0.05 + 0.95/32) = 1/0.0797 \approx 12.6\times$ — you quadrupled the cores from 8 to 32 and barely doubled the speedup. And the absolute ceiling, on infinite cores, is $1/0.05 = 20\times$. That last 5% of serial work caps you at 20× *no matter how many cores you buy*. This is the most sobering number in parallel computing: a tiny serial fraction strangles your scaling, and it's why "just add cores" so often disappoints. (Gustafson's Law offers a more optimistic counterpoint for problems that grow with the hardware — the next post covers both.)

How to measure all of this *honestly* — the rules that separate real benchmarks from folklore:

```bash
set -euo pipefail
  # Honest concurrency measurement, the non-negotiable checklist.
  # 1. WARM UP: discard the first runs (JIT, cache, page faults skew them).
  # 2. MANY RUNS: report a distribution, not one number — concurrency is nondeterministic.
  # 3. REPORT THE SPREAD: median and p99, not just the mean; tails matter.
  # 4. PIN THE PLATFORM: x86 (TSO) and ARM (weak) give DIFFERENT answers on ordering bugs.
  # 5. ISOLATE: a noisy machine (other load, thermal throttling) corrupts the numbers.
  # 6. SWEEP THE VARIABLE: thread count 1,2,4,8,16,32 — find where the curve BENDS DOWN.
for threads in 1 2 4 8 16 32; do
  for run in $(seq 1 30); do
    THREADS="$threads" ./counter_bench   # discard warm-up, keep the rest
  done | sort -n | awk 'NR>5'            # drop the first five as warm-up
done
```

The cardinal sin is the single-run benchmark: "I ran it once, threads were faster." On a nondeterministic system, one run is noise. The second sin is ignoring the platform: an ordering bug that *never* reproduces on your x86 laptop will *reliably* corrupt data on an ARM server, because x86's Total Store Order hides reorderings that ARM's weak model exposes. If your test fleet is x86 and your production is ARM (increasingly common), you are testing on the easy hardware and shipping to the hard hardware. The memory-model post lives and dies on this distinction; the takeaway for now is that **"it works on my machine" is, for concurrency, a statement about your machine's memory model, not about your code's correctness.**

## When to Reach for Concurrency (and When Not To)

Concurrency is not free and not always worth it. The decisive engineering judgment is knowing when the complexity pays for itself. Here is the honest accounting.

**Reach for concurrency when:**

- **You are I/O-bound and need responsiveness.** A server handling many connections that each mostly wait — use async/event-loop concurrency. One core, ten thousand connections, no blocking. This is the clearest win, and it's *concurrency without parallelism*.
- **You are CPU-bound, the work splits cleanly, and the serial fraction is small.** A batch job, an encoder, a simulation that decomposes into independent chunks — parallelize across cores. Check Amdahl first: if your serial fraction is 20%, your ceiling is 5×, and you should ask whether 5× justifies the bug surface.
- **You must overlap independent latencies.** Fetch from three services that don't depend on each other — do them concurrently and wait once for all three, turning 3 × 100 ms serial into ~100 ms overlapped.

**Do NOT reach for concurrency when:**

- **The sequential version is fast enough.** This is the most ignored rule. Concurrency multiplies your bug surface by the number of interleavings — a combinatorial explosion. If a single thread meets your latency and throughput targets, *stay single-threaded* and spend your complexity budget elsewhere. The fastest, most correct concurrent program is often the sequential one you didn't write.
- **The work is inherently serial.** If each step depends on the previous (a strict state machine, a sequential parse, a cryptographic hash chain), there's nothing to overlap; concurrency adds overhead and bugs for zero speedup. Amdahl with $p \approx 0$ gives $S \approx 1$.
- **You'd add threads to an I/O-bound task.** Threads have real memory and context-switch cost; for I/O-bound work, async concurrency on a single thread usually beats thread-per-task. (For the Python-specific version of this argument — why threads help I/O but not CPU, because of the GIL — see [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) and [threading done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits); this series gives the language-agnostic view and lets that series own the Python story.)
- **You'd go lock-free before measuring.** Lock-free programming is brutally hard to get right and only pays off when a lock is provably your bottleneck. Reach for a plain mutex first; measure; only then consider atomics or lock-free structures. Premature lock-freedom is the concurrency equivalent of premature optimization, with worse failure modes.

The meta-rule: **the cost of concurrency is the explosion of possible behaviors, and you pay it in debugging time forever.** Add concurrency only when the throughput or responsiveness gain is real, measured, and necessary — and when you do, use the *highest-level* mechanism that solves your problem (a channel before a lock, a lock before an atomic, an atomic before lock-free), because higher-level mechanisms remove more interleavings and leave fewer ways to be wrong. The capstone post, [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model), formalizes this into a decision tree.

## Key Takeaways

1. **Concurrency is structure (dealing with many tasks); parallelism is execution (doing many at once).** Decide which problem you have — I/O-bound wants concurrency, CPU-bound wants parallelism — before reaching for a tool.
2. **You can't avoid concurrency.** Dennard scaling ended around 2005; clock speeds went flat and cores multiplied. Sequential code captures a single core and leaves the rest idle. Extra performance now requires concurrent code.
3. **The enemy is nondeterminism, not threads.** A concurrent program is a relation, not a function — same input, a *set* of possible outputs depending on uncontrollable scheduling. You fix bugs by *removing interleavings*, not by being careful.
4. **Shared mutable state is the root hazard.** Remove any of *shared*, *mutable*, or *state* and the hazard disappears. The cheapest synchronization is the one you don't need because nothing is shared and mutable.
5. **`count++` is a read-modify-write, not one step.** The lost-update interleaving — both threads load the stale value, both store, one update erased — is the prototype of nearly every race you'll debug.
6. **You need three independent guarantees: atomicity, visibility, ordering.** Dropping each produces a different bug (lost update, stale read, reorder bug). A correct lock buys all three; atomics buy exactly the one you ask for. The unifying concept is *happens-before*.
7. **Latency and throughput are different goals** (Little's Law: $L = \lambda W$). Concurrency buys throughput by overlapping waits, often without improving any single operation's latency. Know which one you're optimizing.
8. **Amdahl's Law caps your speedup at $1/(1-p)$.** A 5% serial fraction limits you to 20× on infinite cores. Throughput-vs-threads rises, peaks, and *bends down* under contention — more threads is not more speed.
9. **Measure honestly:** warm up, run many times, report the spread, pin the platform (x86 TSO vs ARM weak ordering give different answers), sweep the thread count to find where the curve bends. One run is noise.
10. **Use the highest-level mechanism that works**, and don't add concurrency the sequential version doesn't need — the cost is a combinatorial explosion of behaviors you pay for in debugging time forever.

## Further Reading

- **Herb Sutter, "The Free Lunch Is Over" (2005)** — the canonical essay on why clock speeds stalled and concurrency became mandatory. The origin of this post's middle section.
- **Rob Pike, "Concurrency Is Not Parallelism" (2012 talk)** — the precise distinction, from a designer of Go, with the worker-gopher example.
- **Maurice Herlihy & Nir Shavit, *The Art of Multiprocessor Programming*** — the rigorous text on synchronization, linearizability, and lock-free structures. The reference for the later non-blocking posts.
- **Brian Goetz et al., *Java Concurrency in Practice*** — the clearest practical treatment of the three guarantees (atomicity, visibility, ordering) and the Java Memory Model; the `volatile`-doesn't-fix-a-counter trap is from here.
- **Anthony Williams, *C++ Concurrency in Action*** — atomics, memory orderings, and lock-free programming done carefully in C++.
- **Nancy Leveson & Clark Turner, "An Investigation of the Therac-25 Accidents" (IEEE Computer, 1993)** — the definitive account of the race condition that killed.
- **Within this series:** the [concurrency playbook capstone](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) (the decision tree this map becomes), and the Python-specific story in [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs). For architecture-scale flow control, see [consistency models for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects); for cross-process backpressure, [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control).

You now have the map. The next post sharpens the concurrency-vs-parallelism distinction into CPU-bound vs I/O-bound and derives the scaling laws — Amdahl and Gustafson — that tell you exactly how much speedup your cores can actually deliver. From there we descend into the four families: mutual exclusion, non-blocking, message-passing, and structured. Every one of them is a different answer to the same three questions you now know to ask of any concurrent code: *what is shared and mutable, what orders the accesses, and what does that cost?*

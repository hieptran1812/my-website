---
title: "The Concurrency Playbook: Choosing the Right Model"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A decision framework for picking the right concurrency model — locks, atomics, async, actors, channels, STM, or data-parallel — by asking what is shared, what orders it, and what it costs, then proving the choice with a measurement."
tags:
  [
    "concurrency",
    "parallelism",
    "decision-framework",
    "playbook",
    "architecture",
    "async",
    "actors",
    "lock-free",
    "channels",
    "systems-programming",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-1.png"
---

A small team I worked with once shipped the same feature three times, and each time it broke in a new and educational way. The feature was unremarkable: a counter of how many times each item in a catalog had been viewed, updated on every request, read on every page. The first version used a global mutex around the whole request handler, because that was the obvious way to "make it thread-safe." Under load, the eight-core box did the work of barely one core: every request waited its turn for the one lock, and most of the time inside that lock was spent waiting on a network call to the cache. The second version tried to fix the throughput by going lock-free with a fancy concurrent map and a CAS loop on each counter — which was genuinely faster on the developer's laptop and genuinely buggy in production, because nobody had reasoned about the memory model and a stale read occasionally served a count from before an update was visible. The third version threw the whole thing onto a thread-per-request model with a thread pool sized to four hundred, which fell over the first time traffic spiked, because four hundred threads blocking on a slow downstream dependency is four hundred threads' worth of stack and scheduler overhead doing nothing.

The fourth version worked, and it was the simplest of the four. The counters lived behind a single-threaded event loop that batched updates and flushed them asynchronously; the network calls were non-blocking; the per-item state was never shared across threads because it was never touched by more than one. It scaled to the traffic, it was correct, and a new engineer could read it. The difference between the three failures and the one success was not cleverness. It was a decision made on purpose instead of by reflex: a deliberate answer to "is this work CPU-bound or IO-bound, and is this state actually shared?" — followed by the cheapest mechanism that bought correctness, and a measurement that proved it.

This post is the capstone of the series. It is the hub that ties together everything the other thirty-three posts taught — [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), the anatomy of a race, the memory model, locks and their bugs, async and event loops, actors and channels, lock-free progress, structured lifetimes — and turns it into a decision framework you can actually use under pressure. The figure below is the spine of that framework: two branch questions that route any workload to one of seven models. We will spend the rest of the post earning every node in it.

![A decision tree that branches first on whether work is CPU-bound or IO-bound and then routes to data-parallel, lock-free, async, actors, or channels](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-1.png)

By the end you will be able to look at a new system — an API gateway, a market-data fan-out, a batch job — and say, with reasons, "this wants an async event loop with a bounded worker pool, not threads-per-request," and then back that claim up with numbers. You will know the cost and the signature failure of every model, the anti-patterns that quietly destroy throughput, and how to test concurrent code that lies to you every time you run it.

## The series spine, restated rigorously

Before the decision tree, the one idea the whole series rests on, stated precisely so the rest of the post can lean on it.

Concurrency is about **correctness under nondeterminism**. Parallelism is about **throughput under finite hardware**. They are different problems that happen to use the same tools. You can have concurrency with one core (an event loop juggling thousands of sockets) and parallelism with no concurrency hazard at all (a pure data-parallel sum over disjoint slices of an array). Conflating them is the root of most bad model choices: people add threads "to go faster" when the work was IO-bound and threads buy nothing, or they reach for an event loop when the work was a CPU-bound number-crunch that a single loop can only serialize.

The hazard at the center of all of it is one sentence: **shared mutable state plus nondeterministic scheduling equals a bug waiting to manifest.** Pull either leg out and the hazard collapses. If the state is not shared — each task owns its own — there is nothing to race over. If the state is not mutable — it is immutable, copied, or handed off so only one party touches it at a time — there is nothing to race over. If the scheduling were deterministic, you could reason about the one interleaving that happens; but real schedulers preempt at instruction boundaries you do not control, on cores that reorder your memory operations, so the set of possible interleavings is enormous and a test that passes a million times can fail the million-and-first. The [anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) walks the canonical `count++` interleaving instruction by instruction; the short version is that `count++` is a load, a modify, and a store, and two threads can both load the same value, both increment it, and both store the same result, losing one update entirely.

So the discipline is a four-step loop, and it is the same loop no matter which model you end up picking:

1. **Name what is shared and mutable.** Not "the system is concurrent" — *which exact piece of state is touched by more than one thread, and is it written by any of them?* Read-only shared state is free. The bug lives only in the cells that are both shared and written.
2. **Establish a happens-before order over every access to that state.** [Happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) is the formal relation a memory model gives you: if access A happens-before access B, then B is guaranteed to see A's effect. Without a happens-before edge between two conflicting accesses, you have a **data race** — undefined behavior in the C++, Java, Go, and Rust models — and the compiler and CPU are free to make your program do anything at all.
3. **Pick the cheapest mechanism that buys that order.** A lock, an atomic, a channel send, an actor message, an immutable handoff — each one establishes happens-before edges, at a different price. The whole point of the playbook is matching the mechanism to the shape of the problem so you pay the least.
4. **Prove it with a measurement.** Concurrency intuition is unreliable; the only honest answer to "is this faster / correct enough" is a number from a warmed-up, many-times-repeated benchmark on representative hardware. Measure first, measure again after.

That loop is the engine. The seven models are just seven different answers to step three. The decision tree in figure 1 is just a fast way to narrow the seven down to one or two candidates. Everything below is detail.

## Step one: is the work CPU-bound or IO-bound?

This is the first branch, and it is the one people get wrong most often, so it is worth being precise.

A task is **CPU-bound** if its wall-clock time is dominated by the CPU executing instructions — matrix multiplies, JSON parsing of huge payloads, image resizing, cryptographic hashing, a tight numerical loop. The bottleneck is the cores. The only way to make CPU-bound work faster is to use more cores (parallelism) or to do less work (a better algorithm). Adding "concurrency" in the sense of more threads than you have cores does nothing but add context-switch overhead, because the cores are already saturated.

A task is **IO-bound** if its wall-clock time is dominated by *waiting* — for a network round-trip, a disk read, a database query, a lock held by someone else. The CPU is idle most of the time. Here, more cores buy you almost nothing because the cores were not the bottleneck; what you want is the ability to have many operations *in flight* at once so the waiting overlaps. One thread can wait on ten thousand sockets at once; you do not need ten thousand threads.

The scaling laws make the CPU-bound ceiling concrete. **Amdahl's law** says that if a fraction $p$ of your work is parallelizable and $(1-p)$ is serial, then with $N$ cores your speedup is

$$S = \frac{1}{(1-p) + \frac{p}{N}}.$$

If 95% of the work parallelizes, the best speedup you can ever reach, even with infinite cores, is $1/0.05 = 20\times$. The serial 5% — argument parsing, the final reduction, a global lock — caps you. The [scaling-laws post](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws) derives this and Gustafson's complement in full. The practical takeaway for the playbook: for CPU-bound work, your enemy is the serial fraction, so the right models are the ones that minimize coordination — data-parallel fork/join over disjoint data, and lock-free structures only on the genuinely hot shared cell.

The IO-bound side has its own law. **Little's law** says the average number of concurrent requests in a system is

$$L = \lambda W,$$

the arrival rate $\lambda$ times the average time in system $W$. If 5,000 requests per second each take 200 ms (because they wait on a slow backend), you have $L = 5000 \times 0.2 = 1000$ requests in flight at any instant. With a thread-per-request model that is a thousand threads, most of them blocked, each costing a megabyte of stack and a slot in the scheduler's run queue. With an event loop it is one thread tracking a thousand outstanding operations. This is the [C10k problem](/blog/software-development/concurrency/blocking-vs-non-blocking-io-and-the-c10k-problem) and it is why async exists.

#### Worked example: classifying three real workloads

Take three concrete tasks and classify each, because the classification is the whole game.

- **Resizing 50,000 uploaded images.** Each resize is a few hundred milliseconds of pure pixel math, no waiting. This is **CPU-bound**. Eight cores can give close to $8\times$ if the work splits cleanly (it does — images are independent). Model: data-parallel fork/join. Throwing async at it is pointless; the loop would just run resizes one at a time on one core.
- **An API gateway proxying 8,000 requests/second to upstream services.** Each request spends 1 ms of CPU (parse, route, serialize) and 50 ms waiting on the upstream. By Little's law that is $8000 \times 0.051 \approx 408$ concurrent requests, almost all of them *waiting*. This is overwhelmingly **IO-bound**. Model: async event loop. Thread-per-request would need ~400 blocked threads; an event loop needs a handful.
- **A risk engine recomputing portfolio value-at-risk on every market tick.** Each recompute is a 30 ms Monte Carlo simulation — pure CPU — but ticks arrive 2,000 times a second from a socket — pure IO. This is **mixed**, and mixed is the interesting case: you want an async front end to absorb the ticks without blocking, handing the CPU-heavy simulations to a bounded pool of worker threads sized to the core count. Async for the IO, parallel workers for the CPU. The two halves use different models on purpose.

The mixed case is not a cop-out; it is the common case in real services, and the playbook's answer is "decompose the system into IO-bound and CPU-bound stages and apply the right model to each stage," which is exactly what the worked example at the end of this post does for an API gateway.

## Step two: is the state shared, or can you avoid sharing?

The second branch is the more powerful one, because the cheapest way to win the concurrency game is to not play it. If two tasks never touch the same mutable state, no synchronization is needed, no happens-before edge has to be established, no lock can be held wrong, no atomic can be ordered wrong. The most reliable concurrent code is code that has nothing to coordinate.

So before you reach for any mechanism, spend real effort trying to make the answer "no, not shared":

- **Partition the data.** Give each thread a disjoint slice. A parallel sum over an array splits the array into N chunks, sums each chunk on its own core with zero coordination, and only combines N partial sums at the end. The only shared write is the final reduction, which is tiny. This is the entire philosophy of [data-parallel fork/join](/blog/software-development/concurrency/data-parallelism-fork-join-and-work-stealing).
- **Make state immutable.** If a value never changes after construction, any number of threads can read it with no synchronization at all (you only need to publish it safely once — see the memory model). Functional-style code that returns new values instead of mutating old ones gets concurrency almost for free.
- **Confine state to one owner.** The [actor model](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision) takes this to its logical end: each actor owns its state exclusively and processes one message at a time, so inside an actor there are *no locks and no races* — the isolation does the work. [CSP channels](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement) do it by *transferring ownership*: "don't communicate by sharing memory; share memory by communicating." When you send a value down a channel, you hand off ownership; the sender promises not to touch it again, so only one goroutine ever has it.

Only when you genuinely cannot avoid sharing — there is one piece of state that multiple workers legitimately must read and write, and partitioning or confining it would be more complex than it is worth — do you reach into the shared-memory toolbox: locks, then atomics, then lock-free. And even then, the [precise distinction](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction) matters: a **data race** (unsynchronized conflicting access to one location) is always a bug, but a **race condition** (an ordering bug at the logical level) can exist even in fully synchronized code, so synchronization is necessary but not sufficient — you still have to get the protocol right.

Here is the same shared-counter problem solved three ways, to make the "which mechanism" choice concrete. First, the shared-memory-plus-lock version, in Java:

```java
// Shared state guarded by a lock. Correct, simple, fine at low contention.
final class CounterMutex {
    private long value = 0;
    private final Object lock = new Object();

    void increment() {
        synchronized (lock) {   // happens-before via the monitor
            value++;            // load-modify-store, now atomic w.r.t. the lock
        }
    }

    long get() {
        synchronized (lock) { return value; }
    }
}
```

Second, the atomic version — no lock, a single hardware compare-and-swap loop hidden inside `getAndAdd`, which the [atomics post](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) shows maps to one `lock xadd` on x86:

```java
import java.util.concurrent.atomic.AtomicLong;

// One hot counter, no lock. Faster under contention than a mutex,
// because there is no parking/wakeup, just a CAS retry loop.
final class CounterAtomic {
    private final AtomicLong value = new AtomicLong();

    void increment() { value.getAndIncrement(); }  // atomic add, hardware ordered
    long get()       { return value.get(); }       // acquire load
}
```

Third, the model that avoids sharing entirely — an actor in Elixir, where the count is private to one process and every update is a message, so there is no lock because there is nothing concurrent inside:

```elixir
defmodule Counter do
  # The counter's state is owned by one process. Updates are messages.
  # No lock: only this process ever touches count, one message at a time.
  use GenServer

  def start_link(_), do: GenServer.start_link(__MODULE__, 0, name: __MODULE__)
  def increment,     do: GenServer.cast(__MODULE__, :inc)
  def get,           do: GenServer.call(__MODULE__, :get)

  def init(n),                 do: {:ok, n}
  def handle_cast(:inc, n),    do: {:noreply, n + 1}
  def handle_call(:get, _, n), do: {:reply, n, n}
end
```

All three are correct. Which one is *right* depends entirely on the rest of the system. If this counter is one of millions of low-contention counters scattered through a service, the mutex is fine and the simplest. If it is the one hot counter that every request bumps, the atomic wins because it never parks a thread. If the counter is part of a larger stateful entity that needs supervision and fault isolation — a per-user session, a per-connection state machine — the actor wins because the isolation buys you more than just the counter. The playbook is the art of knowing which sentence applies.

## The seven models, each with its cost and failure mode

Now the heart of it. Seven models, each a different answer to "what establishes the happens-before order." For each I give when it wins, what it costs, how it dies, and the deep-dive that covers it. The figure summarizes the trade-offs; the prose gives you the reasoning.

![A matrix listing the seven concurrency models against the situation each fits best and the main cost each one imposes](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-2.png)

### 1. Shared memory plus locks

**When it wins.** This is the default, the familiar one, and there is no shame in it. When state is genuinely shared, contention is low to moderate, and the critical sections are short, a [mutex](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) is the right tool. It is easy to reason about: the lock protects *data*, and inside the critical section you can assume no one else is touching it. Most application code that touches shared state correctly uses a mutex and should keep using one.

**What it costs.** An uncontended lock is cheap — roughly 20 to 50 nanoseconds for the acquire/release on modern hardware, since it is often just a CAS that succeeds. The cost shows up under *contention*: when threads pile up on a hot lock they form a **convoy**, each waiting for the one before, and throughput collapses — the eight-core box does the work of one. There is also the visibility subtlety: a lock gives you mutual exclusion *and* the happens-before edge that makes the previous holder's writes visible, but only if every access uses the same lock. Get the [granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity) wrong — one coarse lock for everything — and you serialize unrelated work.

**How it dies.** [Deadlock](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them): two threads each hold a lock the other needs, and both wait forever. The fix is a global lock-ordering discipline (always acquire locks in the same order) or `tryLock` with backoff. The four Coffman conditions tell you exactly what to break. The secondary failure is the convoy above, and [priority inversion](/blog/software-development/concurrency/livelock-starvation-and-priority-inversion) — a low-priority thread holding a lock a high-priority thread needs, the bug that nearly killed the Mars Pathfinder mission.

### 2. Atomics and lock-free structures

**When it wins.** When there is exactly one hot, contended cell — a counter, a flag, a stack/queue head pointer — and a mutex around it is provably your bottleneck. An [atomic](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) operation maps to a single hardware instruction (`lock xadd`, `cmpxchg`, or LL/SC), so it never parks a thread; under heavy contention a CAS-retry loop can sustain far more throughput than a lock that puts threads to sleep and wakes them. [Lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) — a Treiber stack, a Michael–Scott queue — give *system-wide progress*: even if one thread is preempted mid-operation, others keep going.

**What it costs.** It is the hardest model to get right, because you are programming directly against the [memory model](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before). You must choose [memory orderings](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences) — relaxed, acquire, release, seq_cst — and a wrong choice is a bug that only appears on weakly-ordered hardware (ARM, POWER) and never on your x86 laptop. And the truly hard part nobody warns you about is [memory reclamation](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu): in a manual-memory language, when *can* you free a node another thread might still be reading? Hazard pointers, epochs, and RCU exist precisely because the obvious answer ("free it after I pop it") is a use-after-free.

**How it dies.** The [ABA problem](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads): a CAS sees the value it expected and succeeds, but the value cycled $A \to B \to A$ underneath it and the structure is now corrupt. The fix is a tagged pointer (version counter) or a reclamation scheme. The meta-failure is **premature lock-free** — reaching for it before measuring, getting code that is subtly wrong *and*, at low contention, often *slower* than the mutex you replaced because of the retry overhead. The [progress hierarchy](/blog/software-development/concurrency/the-progress-hierarchy-blocking-lock-free-and-wait-free) is clear that lock-free is about *progress guarantees*, not speed.

### 3. Async and the event loop

**When it wins.** Massive IO concurrency on few threads. When you have thousands of connections that are mostly waiting — a web server, a proxy, a chat backend, a websocket fan-out — one [event loop](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) multiplexes them all with `epoll`/`kqueue`/IOCP, handling whichever socket is ready. [async/await](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) is just syntactic sugar over a state machine that suspends at each `await` point and resumes when the IO completes; [futures and promises](/blog/software-development/concurrency/futures-promises-and-composing-asynchronous-code) compose those suspensions. One thread, thousands of in-flight operations, no per-connection stack.

**What it costs.** The [function-coloring tax](/blog/software-development/concurrency/function-coloring-and-bridging-sync-and-async): async functions can only be called from async functions, so "going async" tends to color your whole call graph. And the model assumes you never hold the CPU for long — it is cooperative, not preemptive, so a single CPU-heavy task starves every other connection on that loop.

**How it dies.** **Blocking the event loop.** A synchronous database call, a CPU-heavy loop, a `Thread.sleep` — anything that does not yield — freezes *every* connection on that loop, not just its own. This is the canonical async incident: latency for thousands of users spikes because one handler did a blocking call. The fix is to offload blocking and CPU work to a thread pool and keep the loop doing only IO orchestration. (For the Python-specific version of this story — the GIL, `asyncio`, and blocking calls — see [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) and [async in practice](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code); this series stays language-agnostic and links out.)

### 4. Actors

**When it wins.** Isolated stateful entities that need fault tolerance: per-user sessions, per-device state machines, per-connection protocol handlers, anything where the state naturally belongs to one owner and you want crashes contained. Each actor owns its state, processes one message at a time, and communicates only by sending messages — so inside an actor there are no locks and no data races by construction. [Supervision trees](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision) and "let it crash" turn a crashed actor into a restart, not a corrupted shared heap. The model also scales across machines because message-passing does not care whether the recipient is local.

**What it costs.** Message-passing overhead (copying or serializing), and the loss of synchronous shared invariants. There is also the unbounded-mailbox trap (below).

**How it dies.** You cannot express a **cross-actor invariant** atomically — "transfer money from actor A to actor B such that the total is always conserved" is not a single actor's local decision, and naive message-passing makes it a distributed-transaction problem. If your domain has invariants that span entities, the actor boundary is in the wrong place. The other failure is the **unbounded mailbox**: a fast producer floods a slow actor's queue, memory grows without bound, and the process OOMs. The fix is a bounded mailbox with backpressure.

### 5. CSP and channels

**When it wins.** Pipelines and ownership handoff. When work flows through stages — read, parse, transform, write — [channels](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement) connect lightweight tasks (goroutines) and the value's ownership moves down the pipe, so only one stage touches it at a time. The `select` statement waits on multiple channels, which makes timeouts, cancellation (the done-channel idiom), and fan-in/fan-out natural. It is [message-passing without the actor's identity](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy) — the channel, not the recipient, is the unit of synchronization.

**What it costs.** A buffered channel is a queue, and a queue has a depth you must choose; the wrong depth either blocks too eagerly or buffers too much. Reasoning about a graph of channels is harder than it looks.

**How it dies.** **Deadlock by channel**: a goroutine waiting to send on a full channel while the only reader is waiting to send to *it*. Closed-channel panics and goroutine leaks (a goroutine blocked forever on a channel nobody will write to) are the everyday version. Bounded channels plus a clear ownership/cancellation protocol are the fix.

### 6. Software transactional memory

**When it wins.** When you need **composable** atomic operations over multiple pieces of shared state — the one thing locks famously cannot do, because composing two correctly-locked operations can deadlock. [STM](/blog/software-development/concurrency/software-transactional-memory-and-optimistic-concurrency) treats memory like a database: you wrap a block in a transaction, it runs optimistically, validates at commit, and retries automatically if another transaction conflicted. Two STM operations compose into one larger atomic operation for free. Clojure refs and Haskell's `STM` monad are the canonical implementations. (The isolation/MVCC machinery is the same family as a database; the [database consistency models](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) post covers the architecture-scale version.)

**What it costs.** Retries under contention — a hot, frequently-conflicting transaction can retry many times, and side effects (IO) cannot live inside a transaction because they cannot be rolled back. Hardware TM (Intel TSX) promised to make it cheap and largely stalled.

**How it dies.** Livelock-style retry storms under high contention, and the seductive mistake of putting an irreversible side effect inside a transaction that then retries and performs it twice.

### 7. Data-parallel fork/join

**When it wins.** CPU-bound divisible work: parallel map/reduce/scan, divide-and-conquer (sort, search, numerical kernels), anything where the data partitions into independent slices. [Fork/join with work-stealing](/blog/software-development/concurrency/data-parallelism-fork-join-and-work-stealing) — Rayon, Java's `ForkJoinPool`, TBB — splits the work, runs the pieces on a pool sized to the cores, and steals from busy queues to keep all cores fed. It is the cleanest path to Amdahl-limited speedup because the coordination is minimal: split, compute disjoint, combine.

**What it costs.** An overhead floor: splitting and combining cost something, so on tiny tasks the overhead dominates and you go *slower* than sequential. You must pick a sensible grain size (stop splitting below ~10k elements, say). And it only helps CPU-bound work — forking IO-bound tasks just creates blocked threads.

**How it dies.** Over-decomposition (more overhead than work) and hidden shared state in what you thought was a pure map — a shared accumulator, a non-thread-safe library call — which turns the "no coordination" win back into a race.

It is worth seeing how *little* code the right model needs when the model fits the work. Compare the shared-counter dance above — locks, atomics, orderings — with a CPU-bound parallel sum, where the data partitions cleanly and there is no shared mutable state to guard at all. In Rust with Rayon, the parallel version is a one-word change from the sequential one, and it is correct by construction because each slice is summed independently and the partial sums are combined by the framework:

```rust
use rayon::prelude::*;

// Sequential: one core, no parallelism.
fn sum_seq(data: &[f64]) -> f64 {
    data.iter().sum()
}

// Data-parallel: par_iter splits the slice across the thread pool,
// sums disjoint chunks on each core, then combines. No lock, no atomic,
// no shared mutable state — the partition removes the hazard entirely.
fn sum_par(data: &[f64]) -> f64 {
    data.par_iter().sum()
}
```

There is no mutex, no `Arc`, no memory ordering to reason about, because there is nothing shared and mutable: `par_iter` hands each core a disjoint slice, and Rust's `Send`/`Sync` type system will not even let you accidentally share a non-thread-safe accumulator across the slices. The lesson for the playbook is that the *right* model makes the dangerous parts unrepresentable — you are not being careful about a race, you have arranged for there to be no race to be careful about. That is the entire payoff of step two: avoid sharing, and the synchronization problem disappears instead of being managed.

A second matrix makes the *failure modes* of the four most error-prone models explicit, because knowing how a model dies is most of how you avoid it:

![A matrix pairing four concurrency models with their classic bug and the structural fix for each one](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-3.png)

## The decision flowchart, in words

The figure at the top of the post is the flowchart; here is how to actually run it, because a tree is only useful if you know how to walk it.

**First question — CPU-bound or IO-bound?** Profile if you do not know; do not guess. Look at whether the threads are *running* or *waiting*. If they are pegged at 100% CPU, you are CPU-bound. If they are mostly in `epoll_wait`, `recv`, `futex`, or `read`, you are IO-bound. If it is genuinely a mix, decompose it into stages and run the flowchart per stage.

**If CPU-bound:** the default is **data-parallel fork/join**. Partition the data, run disjoint slices on a pool sized to the cores, combine. Reach for **atomics/lock-free** *only* if profiling shows a single hot shared cell (a shared counter, a shared accumulator, a work-queue head) that the partitioning could not eliminate — and even then, measure that the lock is actually your bottleneck before going lock-free.

**If IO-bound:** ask the second question — **shared state, or can you avoid it?**
- If the state can be confined to one owner and you want fault isolation, reach for **actors**.
- If the work flows through stages with ownership handoff, reach for **CSP/channels**.
- If it is mostly stateless IO orchestration (a proxy, a fan-out), reach for the **async event loop** directly.
- If you truly cannot avoid shared mutable state across the IO workers, fall back to **shared memory + locks** (and atomics only on a measured hot cell).

**The STM branch** is orthogonal: reach for it specifically when you need *composable* multi-location atomicity and a lock-based version would deadlock or be too coarse — it is a specialist tool, not a default.

The single most common right answer for a network service is "async event loop for the IO, plus a bounded worker pool for the CPU-heavy parts, with as little shared mutable state as you can manage." That is not a cop-out; it is the mixed case, and it is the common case.

## What the models actually cost: one workload, measured

The flowchart tells you *which* model. The only way to know whether the model paid off is to measure it on your workload. To make the trade-offs concrete, here is a comparison of how the seven models behave on a single representative job — incrementing a set of shared counters from many threads — and what each one buys and costs. The numbers below are order-of-magnitude figures from the kind of microbenchmark you should run yourself; treat them as the *shape* of the answer, not exact values, because the precise numbers depend on your hardware, your contention level, and your memory model. The point is the *relative* behavior and the cost columns, which are stable across machines.

| Model | Shares state? | Comms / sync | Scales to | Failure mode | Relative cost at low contention |
| --- | --- | --- | --- | --- | --- |
| Locks (mutex) | yes | the lock + happens-before | moderate threads | deadlock, convoy | ~25–50 ns acquire/release, cheap |
| Atomics / lock-free | yes (one cell) | hardware CAS / fences | many cores on one cell | ABA, reclamation, reordering | ~10–30 ns, but retry overhead under contention |
| Async / event loop | no (per task) | the loop, await points | 10k+ IO connections | blocking the loop | near-zero per IO op, one thread |
| Actors | no (isolated) | messages / mailboxes | millions of entities | unbounded mailbox, no cross-actor invariant | message copy/serialize per op |
| Channels / CSP | no (ownership moves) | channel send/recv | pipelines of tasks | deadlock-by-channel, leaks | a queue op per handoff |
| STM | yes | optimistic txn + retry | composable txns | retry storms under contention | log + validate per txn |
| Data-parallel fork/join | no (partitioned) | split + combine only | N cores on divisible work | over-decomposition | split/combine overhead floor |

Reading the table top to bottom is the whole argument of this post in one view. The mechanisms that *avoid* sharing — async, actors, channels, fork/join — sit in the "no" column and have the most forgiving failure modes, because they sidestep the hazard instead of guarding it. The mechanisms that *share* — locks, atomics, STM — are not wrong, but they carry the heavier failure modes (deadlock, ABA, retry storms) and demand more care. That is exactly why step two of the discipline is "can you avoid sharing?": the "no" column is where the cheap, robust choices live.

The measured-behavior lesson that surprises people most is the **low-contention column**. At low contention a plain mutex is genuinely cheap — tens of nanoseconds, often a single CAS that succeeds with no kernel involvement — and a lock-free structure is frequently *no faster and sometimes slower*, because the CAS-retry loop and the memory fences cost real cycles even when nobody is contending. Lock-free only pulls ahead when contention is high enough that a mutex starts parking and waking threads (a syscall, microseconds, plus the cache-line bounce of the wakeup). So the honest benchmark to run is not "lock-free vs mutex on my laptop with one thread" — that test will mislead you into thinking lock-free is pointless or, worse, that your buggy lock-free code is correct. The honest benchmark sweeps the thread count from 1 to well past the core count and plots throughput at each point, because the *crossover* — the contention level where the lines swap — is the only number that decides the model. If your real workload never reaches that crossover, the mutex wins on every axis that matters: speed, simplicity, and correctness.

#### Worked example: reading a contention sweep

Suppose you benchmark a shared counter under a mutex and under an atomic, sweeping threads from 1 to 32 on an 8-core box, warmed up, 50 trials each, reporting the median. A plausible shape: at 1 thread the mutex does ~40 million ops/sec and the atomic ~45 million — basically a tie, the atomic marginally ahead. At 4 threads the mutex starts to show contention (~25 million) while the atomic holds (~40 million). At 16 threads, past the core count, the mutex collapses to a convoy (~8 million, threads parking and waking) while the atomic degrades more gracefully (~22 million, CAS retries but no parking). The crossover where the atomic clearly wins is somewhere around 4–8 threads. The decision falls straight out of the curve: if your production contention sits at 2–3 threads on this counter, ship the mutex — it is simpler and the atomic's edge is noise. If it sits at 16+, the atomic is worth its complexity. You cannot make that call without the sweep, and reading one point (1 thread, where they tie) would have told you the opposite of the truth.

## The anti-patterns gallery

These are the mistakes I have watched destroy throughput and correctness, over and over, across teams and languages. Each one has a clear symptom and a structural fix. Memorize the symptoms; they are how you diagnose a sick system from a dashboard.

![A matrix of five concurrency anti-patterns each shown with the production symptom it causes and the structural fix](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-6.png)

**Locks held across I/O.** This is the throughput killer from the opening story. You acquire a lock, then make a network call or a disk read *while holding it*. Now the lock is held for the duration of the slowest dependency in your system, and every other thread that needs that lock waits behind it. Throughput caps at one operation in flight regardless of how many cores you have. The symptom on a dashboard is high lock-wait time and CPU sitting idle while latency climbs. The fix is to do the IO *outside* the lock and hold the lock only for the in-memory state update — microseconds, not milliseconds. The before-and-after figure makes the shape of the fix explicit:

![A before-and-after diagram showing a lock held across a network call shrinking to a tiny in-memory critical section](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-4.png)

**Unbounded queues.** Every queue — a channel, an actor mailbox, an executor's work queue, an async task list — has a depth, and if you leave it unbounded you have built a memory bomb with a delay fuse. Under normal load it looks fine. Under a spike, the producer outruns the consumer, the queue grows, latency for the items already in the queue climbs (Little's law again: $W = L/\lambda$ grows as $L$ grows), and eventually the process OOMs and takes everything with it. The symptom is latency that climbs steadily under load and then a sudden crash. The fix is a **bounded** queue plus **backpressure** — when the queue is full, the producer is forced to slow down (block, drop, or shed load) rather than pile on. This is so important to system health that [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) at the architecture level and [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) at the message-queue level both get dedicated treatments; the concurrency-level version is the bounded queue in your [thread-pool design](/blog/software-development/concurrency/structured-concurrency-cancellation-and-thread-pool-design).

**Blocking the event loop.** Covered above as the async failure mode, but it earns a spot in the gallery because it is so common. Any synchronous, non-yielding call inside an async handler — a blocking DB driver, a `time.sleep`, a tight CPU loop, a synchronous file read — freezes every other connection on that loop. The symptom is that *all* request latencies spike together, not just the slow one's. The fix is to offload anything that blocks or burns CPU to a thread pool and keep the loop doing only non-blocking IO.

**Premature lock-free.** Reaching for atomics and CAS loops before you have measured that a lock is your bottleneck. The result is code that is far harder to reason about, often *slower* at low contention (the retry loop and the memory fences cost more than an uncontended mutex), and prone to memory-model and ABA bugs that only show up under load or on weak hardware. The symptom is subtle, intermittent corruption plus a benchmark that is no faster than the mutex it replaced. The fix is discipline: use a lock until you have measured that it is the problem.

**Double-checked locking done wrong.** The classic broken lazy-initialization idiom: check a flag without a lock, and if unset, take the lock and initialize. Without the right memory ordering, another thread can see the flag set but read a *half-constructed* object, because the write that publishes the object and the write that sets the flag can be reordered — the [torn read / reordering](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads) failure. The symptom is a rare crash or a field that is mysteriously null on a freshly-"initialized" object. The fix is an acquire/release pair (a `volatile` field in Java, an `Acquire` load in C++/Rust), or just use the language's init-once primitive (`std::call_once`, `sync.Once`, a static holder).

The sixth sin, not in the figure but worth naming: **ignoring the memory model.** Assuming that because your code is correct on x86 (which has a strong TSO model and forgives a lot) it is correct everywhere. The same code can break on ARM or POWER, where stores can be reordered far more aggressively. If you write any lock-free or hand-rolled synchronization, you are programming against the memory model whether you acknowledge it or not — the only question is whether you do it deliberately. [Why your code does not run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering) is the post on this; the rule is to use the language's atomics with explicit orderings and never rely on incidental ordering.

## A testing and verification strategy

Concurrent code lies to you. A race can pass a test ten million times and fail the ten-million-and-first because the scheduler happened to preempt one microsecond earlier. You cannot test a race by running it and hoping; you need tools that are built for nondeterminism. Match the tool to the bug:

![A matrix matching four classes of concurrency bug to the verification tool that catches it and why the tool fits](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-7.png)

**Data races → a dynamic race detector.** ThreadSanitizer (in C/C++/Go via `-race`, and others) instruments every memory access and tracks the happens-before relation plus a lockset; when it sees two conflicting accesses with no happens-before edge between them, it reports the race *even if the bad interleaving did not actually happen this run*. This is the single highest-value tool in the box — run your test suite under the race detector in CI. Go's `-race` has caught bugs in the standard library; it will catch yours. The [bug-finding post](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing) covers the happens-before and lockset algorithms in detail.

**Race conditions (ordering bugs that are not data races) → stress testing with randomized delays.** A race condition can exist in fully synchronized code — it is a logical ordering bug, not an unsynchronized access — so a race detector will not flag it. The tool here is stress: run the operation from many threads, thousands of times, with randomized sleeps and yields injected at suspension points to shake out rare interleavings. Tools like Go's stress harness or injected jitter make the rare interleaving likely.

**Lock-free data structures → a model checker like `loom`.** When you write a lock-free structure, the space of possible interleavings and memory-reordering outcomes is too large to stress-test with confidence. Rust's `loom` (and similar tools) *exhaustively* explores all the orderings the memory model permits for a small test, so it can prove the absence of a race for that test rather than just failing to find one. This is how you gain real confidence in a CAS loop.

**Protocols and designs → a specification checker like TLA+.** For a distributed protocol, a lock-ordering scheme, or a consensus algorithm, the bug is often in the *design*, before any code exists. TLA+ lets you specify the protocol and model-check it against invariants and liveness properties, finding the deadlock or the lost-update at the whiteboard stage. Amazon famously used TLA+ to find subtle bugs in production systems' designs.

The meta-rule: **test for the bug class you actually have.** A race detector finds data races and misses ordering bugs; stress testing finds ordering bugs probabilistically; model checking proves small cases exhaustively; spec checking proves the design. Use the one that matches your failure shape, and run the cheap one (the race detector) on everything, all the time, in CI.

The four tools differ sharply in cost and in what guarantee they give, which is why you layer them rather than pick one:

| Tool | What it catches | Guarantee | Runtime cost | When to run it |
| --- | --- | --- | --- | --- |
| Race detector (TSan / `-race`) | data races | finds a real race if any access pattern this run touched it | 2–20x slowdown, ~5–10x memory | every CI run, always |
| Stress test + jitter | race conditions, rare interleavings | probabilistic — more runs, more confidence | many repeats, hours of CI | nightly / before a release |
| Model checker (`loom`, CDSChecker) | lock-free correctness | exhaustive for a small bounded test | minutes per tiny test | when you write lock-free code |
| Spec checker (TLA+) | design-level deadlock / lost-update | exhaustive over the model's state space | engineer-days to write the spec | for a new protocol, before coding |

The cheapest tool — the race detector — has the weakest guarantee but the highest leverage, because it costs you nothing but CPU and catches the most common bug (the data race) automatically. The most expensive tool — a TLA+ spec — has the strongest guarantee but demands you write the model by hand, so you reserve it for designs where a bug is catastrophic and expensive to discover late. The mistake is to skip the cheap one ("our tests pass") and reach for the expensive one only after an incident; the discipline is the opposite — race-detector on by default, escalate to stress and model checking as the risk of the code rises.

#### Worked example: catching the counter race in CI

Take the broken counter — `count++` with no synchronization, from the very first section — and show how each tool reacts. Run a test that spawns 8 goroutines, each incrementing a shared `int` one million times, then asserts the total is 8 million.

- **Plain test run:** passes *sometimes*. On a lightly loaded machine the increments may happen to not collide; you see 8,000,000 and ship the bug.
- **Under `go test -race`:** fails *immediately and deterministically*, printing the exact two stack traces — "Read at ... by goroutine 7 / Previous write at ... by goroutine 4" — with no happens-before edge between them. The detector does not need the bad total to occur; it sees the unsynchronized conflicting access. This is the difference between hoping and knowing.
- **After the fix (`atomic.AddInt64`):** the race detector is silent, the total is always 8,000,000, and a `loom`-style exhaustive check on a 2-thread, 2-increment reduction confirms there is no ordering the memory model permits that loses an update.

The lesson the team learned the hard way: a green test suite means nothing for concurrency unless it ran under a race detector. We made `-race` mandatory in CI after that, and it paid for itself within a week by catching a second, unrelated race nobody had noticed.

## The cost and measurement discipline

Every mechanism in this playbook is a cost. A lock is a cost. An atomic is a cost. An async runtime is a cost. The discipline that separates engineering from cargo-culting is refusing to pay a cost you have not measured a need for. The loop is simple and it is the same loop every time:

![A timeline of the optimization discipline from naming shared state through measuring and repeating on the next bottleneck](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-5.png)

**Name the bottleneck.** Before you change anything, find out where the time actually goes. Profile. Is the service CPU-bound or waiting on a lock or waiting on the network? The opening story's first version *looked* like a lock-contention problem and *was* really an IO-under-lock problem — the lock was the symptom, the network call inside it was the cause. You cannot fix what you have not located.

**Quantify it.** Put a number on it. "The hot lock has 40% wait time at 8 threads." "Each request spends 50 ms in the upstream and 1 ms on CPU." "The unpadded counters lose 60% of throughput to false sharing." A number turns an argument into a measurement.

**Fix the named bottleneck — and only it.** Apply the cheapest mechanism that addresses the specific bottleneck you measured. If it was IO-under-lock, shrink the critical section. If it was a genuinely hot counter, switch it to an atomic. If it was thread-per-connection memory pressure, switch to an event loop. Resist the urge to also rewrite the parts that were fine.

**Re-measure.** Did the fix help? By how much? Did it move the bottleneck somewhere else (it usually does)? Concurrency optimization is whack-a-mole: fixing the lock often exposes that the network is now the limit, which is fine — that is the next loop iteration. Stop when the system meets its target, not when you have run out of clever ideas.

How to measure *honestly*, because it is easy to lie to yourself with a benchmark:

- **Warm up.** The first runs hit cold caches, an unwarmed JIT, an empty connection pool. Discard them.
- **Run many times and report the distribution,** not one number. Concurrency is nondeterministic; a single run tells you almost nothing. Report the median and a tail percentile (p99), because tail latency is what users feel.
- **Name the hardware and the memory model.** "$2.4\times$ on an 8-core x86" is a real claim; "faster" is not. And the *same* lock-free code can behave differently on ARM's weak model than on x86's TSO — say which you measured on.
- **Acknowledge the confounds.** The OS scheduler, other processes, turbo-boost thermal throttling, the GC — all add noise. Pin threads, isolate the machine, or at least run enough trials that the noise averages out.
- **Never fabricate a precise figure.** If you do not have a measurement, give a defensible order of magnitude and *say it is approximate*. "An uncontended mutex is on the order of tens of nanoseconds; a context switch is on the order of a microsecond" is honest. A made-up "exactly 2.7×" is not.

The single highest-leverage habit: measure *before* you optimize, so you know what the bottleneck is, and *after*, so you know whether you fixed it. Most concurrency "optimizations" I have reviewed made things slower or more fragile, and the author never measured to find out.

## Worked examples: choosing a model for real systems

The playbook is only worth anything if it survives contact with a real system. Two worked examples, reasoned end to end.

#### Worked example: a high-throughput API gateway

The system: an API gateway handling 10,000 requests/second. Each request authenticates (a 2 ms call to an auth cache), routes to one of a dozen upstream services (a 40 ms call, the dominant cost), applies a rate limit (an in-memory token-bucket check, microseconds), and streams the response back. We need it to hold 10k rps on a single 8-core box.

*Step one — CPU-bound or IO-bound?* Per request, CPU work is parse + route + serialize, call it 1.5 ms; waiting is auth (2 ms) + upstream (40 ms) = 42 ms. Overwhelmingly **IO-bound** — the box spends 96% of its time waiting. By Little's law, $L = 10000 \times 0.0435 \approx 435$ requests in flight. Thread-per-request would mean ~435 blocked threads, ~435 MB of stacks, and a scheduler thrashing through context switches. Verdict: **async event loop**, not threads.

*Step two — shared state?* The rate limiter's token buckets are shared mutable state — every request reads and decrements one. But they partition cleanly *per route key*, and each bucket is a tiny counter. So most of the system has *no* shared mutable state (each request's parse/route/serialize is independent), and the one shared thing — the buckets — is a hot-counter problem. Verdict: confine where possible (per-request state is local to the request's task), and for the buckets, an **atomic** decrement on each bucket, not a global lock.

*The shape:* an async event loop drives all the IO (auth call, upstream call, response stream) with non-blocking sockets; per-request state lives in the request's own async task (not shared); the rate-limiter buckets are a sharded map of atomic counters (sharded so different routes never touch the same cache line — avoid [false sharing](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing)). Here is the rate-limiter core in Go, the language whose goroutine-plus-channel model fits a gateway well:

```go
// One token bucket per route, refilled by a background ticker.
// take() is a single atomic compare-and-swap loop: no lock, no parking.
type Bucket struct {
    tokens int64 // atomic; on its own cache line via padding below
    _      [56]byte
}

func (b *Bucket) take() bool {
    for {
        cur := atomic.LoadInt64(&b.tokens)
        if cur <= 0 {
            return false // rate limited, shed load — bounded by construction
        }
        if atomic.CompareAndSwapInt64(&b.tokens, cur, cur-1) {
            return true
        }
        // CAS lost the race; another goroutine took a token. Retry.
    }
}
```

The padding (`_ [56]byte`) keeps each bucket on its own 64-byte cache line so two busy routes do not bounce one line between cores — a measured 2–8× difference under contention. The CAS loop is the right call *here* because the bucket is genuinely hot and a mutex would park goroutines; we measured that before choosing it.

*The anti-patterns we deliberately avoid:* we never hold a lock across the 40 ms upstream call (there is no lock on that path at all — the only shared state is the atomic bucket); the request task queue feeding the event loop is **bounded**, so a traffic spike sheds load instead of OOMing; and the event loop never makes a blocking call — the auth and upstream calls are non-blocking, and the only CPU work (TLS, parsing) is bounded and small. We *measured* it: at 10k rps the box ran at ~55% CPU with p99 latency dominated by the upstream's own p99, exactly as Little's law predicted.

#### Worked example: a market-data fan-out

The system: a market-data service receives a firehose of price ticks (~50,000/second) on a single socket and must fan each tick out to ~2,000 subscribed clients, each of which wants a filtered subset (their watchlist). The constraint is latency: a tick must reach interested clients within a millisecond, and order must be preserved per symbol.

*Step one — CPU-bound or IO-bound?* Receiving is one socket (IO, but trivial). Filtering and serializing per subscriber is CPU (50k ticks × matching against 2k watchlists is real work). Sending to 2k client sockets is IO. This is genuinely **mixed**, with a CPU-heavy middle.

*Step two — shared state?* The tick stream is read-only once received (immutable per tick). Each subscriber's connection state is owned by that subscriber. The only shared thing is the routing table (symbol → list of interested subscribers), which is read-mostly — updated only when someone subscribes/unsubscribes, read on every tick.

*The shape:* this is a **pipeline**, and the right model is **CSP channels** (or actors per subscriber). Stage 1: a single goroutine reads the socket and pushes ticks onto a channel — preserving order. Stage 2: a fan-out of worker goroutines reads ticks, consults the read-mostly routing table (guarded by an [RWMutex](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity) — many readers, rare writer, exactly its sweet spot), and pushes each tick to the channels of the interested subscribers. Stage 3: each subscriber is a goroutine owning its socket, reading from its own bounded channel and writing to the client. Per-symbol order is preserved because each symbol's ticks flow through in order and each subscriber processes its channel in order.

```go
// Stage 3: one goroutine per subscriber owns its socket and its channel.
// Bounded channel = built-in backpressure. A slow client cannot OOM us;
// when its buffer fills we drop the slow client instead of buffering forever.
func (s *Subscriber) run() {
    for tick := range s.inbox { // inbox is a bounded channel, cap 256
        if err := s.conn.Write(encode(tick)); err != nil {
            return // client gone; goroutine exits, channel GC'd
        }
    }
}

func (s *Subscriber) deliver(t Tick) {
    select {
    case s.inbox <- t: // fast path: enqueued
    default:
        s.markSlow() // buffer full: this client can't keep up.
        // Drop or disconnect — bounded by design, never unbounded growth.
    }
}
```

The `select` with a `default` is the crucial detail: it makes the channel *non-blocking on send*, so a single slow client cannot back up the whole fan-out. That is backpressure expressed at the channel level — the bounded inbox plus the drop-on-full policy means a misbehaving subscriber degrades only itself. We chose channels over a shared subscriber map with locks because the ownership-handoff model removes the contention by construction: only one goroutine ever touches a given subscriber's socket. And we chose an RWMutex (not a plain mutex, not lock-free) for the routing table because we *measured* the read/write ratio — thousands of reads per write — which is the textbook case for a readers-writer lock and not worth the complexity of a lock-free map.

Two systems, two completely different model choices, both arrived at by running the same flowchart: classify the work, decide whether sharing is avoidable, pick the cheapest mechanism that buys the order, then measure.

## Case studies / real-world

Three real systems, each a load-bearing example of one model done right (with sources, and order-of-magnitude figures flagged as approximate).

**Redis: the single-threaded event loop.** Redis famously served its core data operations from a *single thread* for most of its history, using an event loop (`epoll`/`kqueue`) to multiplex tens of thousands of client connections. The design choice is instructive: Redis's operations are tiny and CPU-cheap (a hash lookup, a list push), so the bottleneck is never the CPU — it is the network and the sheer number of connections, an IO-bound profile. A single-threaded loop avoids *all* locking on the data structures (there is no shared mutable state across threads because there is one thread), which both simplifies the code and removes lock contention as a possibility. Redis routinely handles on the order of 100,000+ operations per second per core this way. Antirez (Salvatore Sanfilippo) documented the reasoning extensively; later Redis versions added threaded IO for the network read/write while keeping command execution single-threaded — a deliberate split of the IO-bound and CPU-bound parts, exactly the mixed-case pattern.

**nginx: the event-driven worker model.** nginx solved the C10k problem with a small number of single-threaded worker processes (typically one per core), each running an event loop over `epoll`. Where Apache's older model spawned a thread or process per connection — and hit a memory and scheduler wall at thousands of connections — nginx's event loop lets one worker handle tens of thousands of mostly-idle connections with a flat memory footprint. This is the async-event-loop model at production scale, and it is *why* nginx displaced thread-per-connection servers for high-concurrency reverse-proxy workloads. The lesson for the playbook: thread-per-connection is the wrong model for IO-bound connection serving, and the industry relearned this expensively.

**WhatsApp / Erlang: actors at massive scale.** WhatsApp famously ran on Erlang and reportedly handled on the order of two million concurrent connections per server (a figure they published in engineering talks around 2012; treat the exact number as approximate and version-dependent). Erlang's actor model — millions of cheap, isolated processes, each owning its state, communicating by messages, supervised so a crash restarts one process rather than corrupting shared state — is purpose-built for exactly this: enormous numbers of isolated stateful entities (connections, sessions) with fault tolerance. No locks, no shared heap to corrupt, crashes contained by supervision trees. This is the actor model's home turf, and it is why telecom and messaging systems reach for it.

**A counter-example worth naming — the LMAX Disruptor and lock-free done right.** The LMAX trading platform needed to process millions of orders per second with very low latency, and they found that conventional queues with locks were the bottleneck. Their answer was the Disruptor: a lock-free ring buffer with careful cache-line padding (to avoid false sharing) and memory-barrier-based publication, achieving on the order of millions of messages per second on a single thread. The instructive part is that they got there *by measuring* — they profiled, found that lock contention and cache-line bouncing dominated, and engineered against the specific measured bottleneck. It is the rare case where lock-free was the right call, and it was right *because* it was measured, not because it was clever. That is the discipline this whole post is about.

These four span the playbook: an event loop (Redis, nginx), actors (WhatsApp), and lock-free (LMAX) — each chosen because it matched the shape of the problem, each validated by measurement, none of them a default reflex.

## When to reach for this (and when not to)

The decisive section. For each model, the one-line "use it when" and the blunt "do not use it when."

The figure pins the defaults to the work type; the prose below adds the "and when not."

![A matrix giving the first-choice and fallback concurrency model for CPU-bound, IO-bound, and mixed work](/imgs/blogs/the-concurrency-playbook-choosing-the-right-model-8.png)

- **Locks.** Reach for them when state is shared, contention is low to moderate, and critical sections are short — which is *most* application code, so this is the right default for "I have some shared state." Do **not** reach for them when you find yourself holding one across IO (shrink the section), when a single lock serializes unrelated work (split it or use finer granularity), or when the lock is provably your bottleneck under contention (then, and only then, consider atomics).

- **Atomics / lock-free.** Reach for them for one hot, contended cell that profiling proved is your bottleneck, where a mutex parks too many threads. Do **not** reach for them before measuring (premature lock-free is slower and buggier), for anything more than a counter/flag/pointer unless you are prepared to reason about the memory model and reclamation, or "because it sounds fast." A mutex you understand beats a lock-free structure you do not.

- **Async / event loop.** Reach for it for massive IO concurrency on few threads — connection servers, proxies, anything where threads would mostly be waiting. Do **not** reach for it for CPU-bound work (a single loop serializes CPU work — you want parallelism, not concurrency), and never block it (offload blocking and CPU-heavy work to a pool).

- **Actors.** Reach for them for isolated stateful entities that need fault tolerance — sessions, devices, connection state machines. Do **not** reach for them when you need an invariant that spans entities atomically (the actor boundary is then in the wrong place), or when the messaging overhead dominates tiny operations.

- **Channels / CSP.** Reach for them for pipelines and ownership handoff, where the value flows through stages and only one stage touches it at a time. Do **not** reach for them when the "pipeline" is really just shared state in disguise (you will fight deadlock-by-channel), and always bound your channels.

- **STM.** Reach for it when you need *composable* multi-location atomicity that locks cannot give you without deadlock. Do **not** reach for it for single-location updates (an atomic or lock is simpler), for high-contention hot spots (retry storms), or anything with irreversible side effects inside the transaction.

- **Data-parallel fork/join.** Reach for it for CPU-bound divisible work — parallel map/reduce, divide-and-conquer. Do **not** reach for it for IO-bound work (you would just create blocked threads), or for tasks so small the split/combine overhead dominates (pick a sensible grain size).

The meta-rule that subsumes all of these: **the cheapest correct mechanism wins, and "cheapest" is decided by measurement, not by which model is most interesting.** A boring mutex that meets your throughput target is a better engineering decision than a thrilling lock-free structure that does not — and you only know which is which because you measured.

## Key takeaways

The closing rules of the entire series, distilled.

1. **Name the shared mutable state first.** The bug lives only where state is both shared and written. Read-only shared state and confined state are free. Most of the work is reducing what is actually shared.

2. **Every access needs a happens-before order.** Without a happens-before edge between two conflicting accesses, you have a data race and undefined behavior. The mechanism you pick — lock, atomic, channel, actor — is just the thing that establishes that edge.

3. **Classify the work before you pick a tool.** CPU-bound wants parallelism (more cores busy: fork/join, lock-free on hot cells). IO-bound wants concurrency (more in flight on few threads: async, actors, channels). Adding threads to IO-bound work, or an event loop to CPU-bound work, is the most common mistake.

4. **The cheapest mechanism that buys the order wins.** A mutex is the right default for shared state at low contention. Reach past it — to atomics, lock-free, STM — only when you have measured that it is your bottleneck. Avoiding sharing entirely (partition, immutability, confinement) beats every locking scheme.

5. **Know how each model dies.** Locks deadlock; lock-free hits ABA and reclamation; async stalls on a blocking call; actors OOM on unbounded mailboxes; channels deadlock; STM retry-storms; fork/join over-decomposes. Knowing the signature failure is most of how you avoid it.

6. **Bound your queues and respect backpressure.** An unbounded queue is a memory bomb on a delay fuse. Bound every channel, mailbox, and work queue, and have a policy (block, drop, shed) for when it fills.

7. **Never hold a lock across IO, never block the event loop.** These two anti-patterns destroy throughput more than any other, and both have the same fix: keep the slow thing out of the part that must stay fast.

8. **Test for the bug class you have.** Run everything under a race detector in CI. Stress-test for ordering bugs. Model-check lock-free structures. Spec-check protocols. A green test suite that never ran under `-race` proves nothing about concurrency.

9. **Measure honestly, before and after.** Warm up, run many times, report the distribution and the tail, name the hardware and memory model, and never fabricate a precise figure. Most "optimizations" make things worse; measurement is how you find out.

10. **Make the decision on purpose.** The difference between the three broken versions of the view counter and the one that worked was not skill — it was deciding deliberately instead of reaching by reflex. Run the flowchart, pick the model, prove it with a number.

## Further reading

The seminal sources, then the map back across the series.

- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming*.** The definitive treatment of locks, lock-free structures, linearizability, and the progress hierarchy. If you want the formal foundations of everything in the lock-free track, this is the book.
- **Brian Goetz et al., *Java Concurrency in Practice*.** Still the clearest practical guide to the happens-before model, safe publication, and building correct concurrent code, even if you do not write Java.
- **Anthony Williams, *C++ Concurrency in Action*.** The reference for `std::atomic`, the memory orderings, and lock-free programming in C++, with the memory model treated carefully.
- **C. A. R. Hoare, *Communicating Sequential Processes* (1978).** The original CSP paper — the intellectual root of channels, goroutines, and "share memory by communicating."
- **Carl Hewitt et al., the Actor model papers; and Joe Armstrong's thesis on Erlang/OTP.** The foundations of actors, isolation, and let-it-crash supervision.
- **"Notes on Structured Concurrency, or: Go statement considered harmful"** (Nathaniel J. Smith) — the argument for nurseries/scopes that the structured-concurrency post builds on.
- **Jeff Preshing's blog (preshing.com)** — the most readable explanations of memory ordering, acquire/release, and lock-free patterns anywhere online.
- **Leslie Lamport's TLA+ materials** — for specifying and model-checking concurrent and distributed designs before you write the code.

Within this series, this capstone is the hub; here is where to go deep on each idea:

- The why and the spine: [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [concurrency vs parallelism and the scaling laws](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws), [processes, threads, and the OS scheduler](/blog/software-development/concurrency/processes-threads-and-how-the-os-scheduler-runs-them), [the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition).
- Locks and their bugs: [mutexes and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections), [how a lock is built](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks), [condition variables and waiting correctly](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly), [semaphores, barriers, and latches](/blog/software-development/concurrency/semaphores-barriers-and-latches), [readers-writer locks and granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity), [deadlock and the four conditions](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them), [livelock, starvation, and priority inversion](/blog/software-development/concurrency/livelock-starvation-and-priority-inversion).
- The memory model: [why your code does not run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering), [sequential consistency and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before), [memory barriers, acquire/release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences), [atomics and memory orderings](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), [cache coherence, MESI, and false sharing](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing).
- The hazards: [data races vs race conditions](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction), [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads), [finding concurrency bugs with race detectors and stress testing](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing).
- Async and non-blocking IO: [blocking vs non-blocking IO and C10k](/blog/software-development/concurrency/blocking-vs-non-blocking-io-and-the-c10k-problem), [the event loop and the reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern), [async/await and how coroutines work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work), [futures, promises, and composition](/blog/software-development/concurrency/futures-promises-and-composing-asynchronous-code), [function coloring and bridging sync and async](/blog/software-development/concurrency/function-coloring-and-bridging-sync-and-async).
- The avoid-sharing models: [message passing vs shared memory](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy), [the actor model](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision), [CSP channels, goroutines, and select](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement), [software transactional memory](/blog/software-development/concurrency/software-transactional-memory-and-optimistic-concurrency), [data parallelism, fork/join, and work stealing](/blog/software-development/concurrency/data-parallelism-fork-join-and-work-stealing).
- Lock-free and putting it together: [the progress hierarchy](/blog/software-development/concurrency/the-progress-hierarchy-blocking-lock-free-and-wait-free), [compare-and-swap and lock-free structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures), [memory reclamation with hazard pointers, epochs, and RCU](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu), [structured concurrency, cancellation, and thread-pool design](/blog/software-development/concurrency/structured-concurrency-cancellation-and-thread-pool-design).
- Beyond this series: the Python-specific concurrency story in [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) and [threading done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits); architecture-scale flow control in [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) and [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects); cross-process delivery in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once); and the GPU/cluster view of collective communication in [NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch).

That is the playbook. Name what is shared, order every access, pick the cheapest mechanism that buys the order, and prove it with a measurement. Do that on purpose, every time, and the 3 AM page never comes.

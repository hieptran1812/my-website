---
title: "Data Parallelism: Fork/Join and Work-Stealing"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Turn an embarrassingly parallel sum into a fork-join divide-and-conquer that work-stealing keeps every core busy, and learn exactly where the overhead floor and Amdahl's ceiling stop you."
tags:
  [
    "concurrency",
    "parallelism",
    "fork-join",
    "work-stealing",
    "data-parallelism",
    "map-reduce",
    "rayon",
    "parallel-streams",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/data-parallelism-fork-join-and-work-stealing-1.png"
---

Here is a problem that sounds trivial and turns out to teach almost everything about parallelism for throughput: add up one hundred million numbers.

```java
long sum = 0;
for (int i = 0; i < data.length; i++) {
    sum += data[i];
}
```

On a recent laptop this runs in roughly 90 milliseconds for an array of 100 million `long` values. One core is doing all the work; the other seven cores on the chip are asleep. That is the entire pitch of this post in one sentence: there is a *correct* answer sitting on the floor — a 6.5× speedup — and the only thing standing between you and it is knowing how to split the work, how to keep the cores fed, and where the diminishing returns kick in.

The naive instinct is "spawn a thread per number." That instinct is catastrophically wrong, and understanding *why* it is wrong is half the lesson. A thread costs roughly a megabyte of stack and a few microseconds to create and tear down; spawning 100 million of them would need a hundred terabytes of stack and would spend all its time in the scheduler, never touching your data. The correct shape is the opposite: a small fixed pool of worker threads — usually one per hardware core — and a way to chop the array into a *tree* of subtasks that those few workers pull through as fast as the cores allow. The chopping is the **fork/join** model. The keeping-fed is **work-stealing**. The figure below previews the whole payoff: the same sum, one core versus eight, and where the wall-clock time goes.

![Sequential sum on one core leaving seven cores idle versus a fork-join split across eight cores finishing in roughly one eighth of the time](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-1.png)

This is a different kind of concurrency from the rest of this series. Most of what we have studied — mutexes, condition variables, channels, the memory model — is about *correctness under nondeterminism*: many tasks touching shared state, and the discipline of ordering their accesses so nothing tears. Data parallelism is about *throughput under finite hardware*: one operation applied to a mountain of independent data, and the discipline of keeping every core busy without ever letting two of them fight over the same memory. The two threads of the series meet here, because the whole reason data parallelism is *pleasant* is that, done right, there is almost no shared mutable state to guard. Each worker owns its slice. They only ever combine partial results at the end. That is why a parallel sum has no locks in its hot path and a parallel hash map has many.

By the end of this post you will be able to: write a parallel reduction (sum, max, count) and a parallel mergesort with a sequential cutoff in more than one language; explain the work-stealing deque trick and why it balances load with almost no contention; pick a task granularity that beats the overhead floor instead of drowning in it; tell an embarrassingly parallel `map` from a reduction that needs an associative combine; and predict — using Amdahl's law — the ceiling that the serial fraction of your program puts on every speedup you will ever get. We will keep returning to the running example of adding up an array, growing it into mergesort, and we will measure honestly: warm up the JIT, run many times, and admit where the numbers wobble.

## Data parallelism: map, reduce, scan, filter

Before any code, name the shape of the work, because the shape decides whether parallelism is free or fiddly. **Data parallelism** means applying the *same operation* to *many elements* of a collection, where the elements are largely independent. It is the opposite of **task parallelism**, where you run *different* operations concurrently (a logging task, a network task, a compute task). Most heavy numeric and data work is data-parallel: transform every row, filter a billion log lines, sum a column, score every document.

There are four canonical data-parallel patterns, and almost every bulk operation is one of them or a composition of them.

**Map** applies a pure function to each element independently and produces a new collection of the same length. Squaring every number, parsing every string, embedding every document. Map is the easy case: element `i` of the output depends only on element `i` of the input, so you can compute the outputs in any order, on any number of cores, with zero coordination. This is what people mean by **embarrassingly parallel** — the parallelism is so obvious it is almost embarrassing that you have to ask for it. Split the range, hand each chunk to a core, done. No combine step.

**Filter** keeps the elements that satisfy a predicate and drops the rest. It is *almost* as easy as map: testing each element is independent, but the output length is not known in advance and the surviving elements have to be packed together. In parallel you typically filter each chunk locally into a small buffer and then concatenate the buffers — concatenation is the combine, and it is cheap and order-preserving.

**Reduce** (also called fold or aggregate) collapses a whole collection into a single value with a binary operator: sum, product, max, min, count, logical-and. This is where parallelism stops being free. To split a reduction across cores you compute a *partial* result on each chunk and then combine the partials. That combine step is only correct if the operator is **associative** — `(a op b) op c` must equal `a op (b op c)` — because parallel reduction effectively re-parenthesizes the operation: instead of `((((a op b) op c) op d) ...)` left to right, it computes `(a op b) op (c op d)` as a tree. Sum, max, and count are associative, so they parallelize. Subtraction and floating-point average are not, so a naive parallel version gives a *different answer* (we will measure exactly that later — it is the single most surprising result in this post).

**Scan** (prefix sum) is the trickiest of the four: it produces a running aggregate, where output `i` is the reduction of inputs `0..i`. Running totals, cumulative distributions, stream compaction. It *looks* inherently sequential — each output needs all the ones before it — but with an associative operator it parallelizes in two passes (an up-sweep that builds partial sums and a down-sweep that distributes them), the Blelloch scan, in `O(log n)` depth. Scan is the proof that "needs the previous result" does not always mean "must be sequential."

```python
data = [3, 1, 4, 1, 5, 9, 2, 6]  # the four shapes, written sequentially
squares  = [x * x for x in data]                 # map: same length, independent
evens    = [x for x in data if x % 2 == 0]       # filter: shorter, independent test
total    = 0
for x in data: total += x                        # reduce: collapses to one value
running  = []                                     # scan: running aggregate
acc = 0
for x in data:
    acc += x
    running.append(acc)
```

In Python the GIL means none of these `for` loops run on more than one core at a time unless you reach for `multiprocessing` or a native library — Python owns that story, so we will [link out to it](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) rather than re-derive it here. The point of the snippet is the *shape*, not the speed: map and filter have no combine, reduce and scan do, and the combine is what associativity governs. The matrix below makes that explicit, and it is the decision table you carry around in your head.

![A matrix of the four data-parallel patterns map filter reduce and scan showing which need a combine step and which require associativity](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-4.png)

The reason to internalize these four is that every parallel library — Java's parallel streams, Rust's Rayon, Intel TBB, C++17's parallel algorithms — is built around exactly them. When you write `data.par_iter().map(f).filter(p).sum()` in Rust, you are composing map, filter, and reduce, and the library is splitting the range, running the chunks on a work-stealing pool, and combining the partials. The abstraction is good *because* the shapes are simple. The rest of this post is about the engine underneath that abstraction.

Two refinements are worth making now, because they explain why the "easy" patterns are not always as easy as they look. The first is that **map and filter can be *fused*** — a chain `map(f).map(g).filter(p)` need not materialize an intermediate collection after each stage; the library can apply `f`, then `g`, then `p` to each element while it is still in a register, in one pass. Both sequential iterator chains and parallel ones do this fusion, which is why `data.par_iter().map(square).filter(positive).sum()` makes exactly one pass over the data per chunk, not three. Fusion matters for parallel code specifically because each extra pass over a large array is another trip to main memory, and memory bandwidth — as we will measure — is often the real ceiling. The second refinement is that **filter has a hidden serial cost**: packing the surviving elements. If you filter into a single shared output array, two chunks would race on the output index, so the parallel version filters each chunk into its *own* local buffer and then concatenates. The concatenation is `O(number of chunks)` (cheap), but it is a combine step, which is why filter is "almost" embarrassingly parallel rather than fully so. The same is true of `flat_map` and of any operation whose output size is data-dependent: independence of the *computation* does not guarantee independence of the *output layout*.

## The fork/join model: split, recurse, combine

The four patterns tell you *what* to compute in parallel. **Fork/join** tells you *how* to organize the splitting. It is divide-and-conquer made concurrent, and it has exactly three moves:

1. **If the problem is small enough, solve it directly** (sequentially). This base case is the *sequential cutoff*, and choosing it well is the difference between a 6× speedup and a 5× slowdown — we will spend a whole section on it.
2. **Otherwise, split the problem into two (or more) independent subproblems, and *fork* them** — schedule them to run in parallel.
3. **Wait for the subproblems to finish (*join*), then *combine* their results** into the answer for this level.

That is the whole model. A parallel sum splits the array in half, forks a task to sum each half, joins both, and adds the two partial sums. Each of those halves does the same to its quarter, and so on, until a chunk is below the cutoff and gets summed with a plain loop. The recursion builds a **tree** of tasks: the root is the whole array, internal nodes are split points, and the leaves are the cutoff-sized chunks that actually touch the data. Results flow *up* the tree as each parent combines its children.

![A fork-join recursion tree where the full sum splits into halves down to sequential leaf chunks and partial sums combine back up the tree](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-2.png)

Here is the canonical fork/join in Java, using `RecursiveTask` from `java.util.concurrent` — the class that *is* fork/join in the JVM:

```java
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

class SumTask extends RecursiveTask<Long> {
    static final int CUTOFF = 10_000;   // sequential threshold
    final long[] a;
    final int lo, hi;

    SumTask(long[] a, int lo, int hi) { this.a = a; this.lo = lo; this.hi = hi; }

    @Override
    protected Long compute() {
        if (hi - lo <= CUTOFF) {        // base case: solve directly
            long s = 0;
            for (int i = lo; i < hi; i++) s += a[i];
            return s;
        }
        int mid = (lo + hi) >>> 1;       // split
        SumTask left  = new SumTask(a, lo, mid);
        SumTask right = new SumTask(a, mid, hi);
        left.fork();                     // fork: schedule left to run elsewhere
        long r = right.compute();        // run right on THIS thread (no spare hop)
        long l = left.join();            // join: wait for left's result
        return l + r;                    // combine
    }
}

// driver
long total = ForkJoinPool.commonPool().invoke(new SumTask(data, 0, data.length));
```

Read the `compute` method closely, because two details in it are the entire art of fork/join. First, the **cutoff**: when the range is at or below `CUTOFF`, we stop splitting and run a plain sequential loop. Without this, the recursion would split all the way down to single elements, creating 100 million task objects — the spawn-a-thread-per-number disaster in a different costume. Second, the **fork-one-run-one** idiom: we `fork()` the left half (offer it to the pool so some worker can grab it) but we *compute the right half on the current thread* rather than forking both. This is not an optimization detail; it is structurally important. If you `fork()` both and then `join()` both, the current thread immediately blocks waiting, and you have created a task it must wait for instead of doing useful work itself. By computing one child inline, the current worker stays busy and the tree's *critical path* — the longest chain of dependent work — stays short. The standard rule: **fork all but one subtask, then compute the last one yourself, then join the forked ones.**

The same model in Rust with Rayon's `join`, which is the lowest-level fork/join primitive in that ecosystem:

```rust
const CUTOFF: usize = 10_000;

fn sum(slice: &[i64]) -> i64 {
    if slice.len() <= CUTOFF {
        return slice.iter().sum();          // base case: sequential
    }
    let mid = slice.len() / 2;
    let (left, right) = slice.split_at(mid); // split into two disjoint &[i64]
    // rayon::join runs both closures, potentially in parallel.
    let (l, r) = rayon::join(|| sum(left), || sum(right));
    l + r                                    // combine
}
```

Notice what Rust's ownership system buys you for free here. `split_at` returns two non-overlapping slices, and the borrow checker *proves at compile time* that the two closures cannot alias the same element — there is no way for `left` and `right` to touch the same memory, so there is no possible data race, and `rayon::join` requires no locks. This is "fearless concurrency" in its purest form: the type system has already done the safety argument that, in Java or C++, you have to make in your head. The Java version is equally race-free, but only because *you* arranged disjoint `[lo, hi)` ranges; nothing checks that for you.

And because the fork/join shape is so regular, you rarely write `join` by hand. Rayon gives you the same sum as a one-liner over a parallel iterator:

```rust
use rayon::prelude::*;

let total: i64 = data.par_iter().sum();   // splits, runs on the pool, combines
```

That single line does everything `SumTask` does: it recursively splits the iterator into chunks, runs them on Rayon's work-stealing thread pool, and combines the partials with `+`. The cutoff is chosen adaptively by the library. The fork/join tree is still there underneath — you just stopped writing it out.

The thing to hold onto: fork/join turns "do this over a big collection" into a *balanced binary tree of independent subtasks*. Independent is the magic word. Sibling subtasks share no mutable state, so they need no synchronization while they run; the only coordination is the join, where a parent waits for its children and combines. That structure is exactly what a work-stealing scheduler is built to run fast.

It is worth being precise about what the *join* costs, because it is the only point where fork/join touches the synchronization machinery from the rest of this series. A `join()` is a *happens-before* edge: the parent's read of the child's result must observe everything the child wrote, so the runtime has to establish that ordering. When the child has already finished, `join()` is nearly free — it reads a completed result with an acquire-load. When the child is *not* yet done, the parent has a choice: block (park the thread and let the OS scheduler run something else) or, far better in a work-stealing pool, **help** — go find *other* tasks to run (including possibly the very child it is waiting on, if no one stole it) until the child completes. This "join means help, not block" behavior is what keeps a fork/join pool from deadlocking itself when there are more outstanding tasks than worker threads: a worker waiting on a join never just sits there, it drains work. Java's `ForkJoinPool` calls this *helping* or *compensation*; if a worker truly must block (because a task it needs is running on another worker), the pool can even spin up a temporary compensation thread so the core does not go idle. The mechanism matters because it is *why* you can have a tree of 20,000 tasks and only 8 threads without running out of threads — the tree is data, the threads are a fixed resource, and join-as-help keeps them all productive.

## Work-stealing: the deque trick

You have a tree of tasks and a small pool of worker threads — say, eight, one per core. How do you assign tree nodes to workers so that every core stays busy and no two cores fight over a shared task queue? The naive answer, a single global queue that all workers lock and pull from, is a disaster: with eight cores hammering one lock, the lock *becomes* the bottleneck, and you measure negative scaling — more cores, less throughput. The elegant answer, and one of the genuinely beautiful ideas in systems, is **work-stealing**.

The setup: **each worker owns its own double-ended queue** (a *deque*) of tasks. A deque has two ends; call them the *bottom* (the worker's own end) and the *top* (the far end). The protocol is asymmetric and that asymmetry is the whole trick:

- **A worker pushes and pops its own tasks at the bottom, LIFO** (last-in, first-out, like a stack). When `compute` forks a subtask, it pushes it on the bottom of *its own* deque. When it needs more work, it pops from the bottom — the most recently pushed task.
- **An idle worker steals from the top of a *victim's* deque, FIFO** (it takes the oldest task there). It picks a victim (often at random), reaches into the *far* end of that worker's deque, and takes one task.

So the owner works one end of its deque, and thieves work the other end. The figure makes the steal concrete: a busy worker with a full deque, an idle worker, and the single task that crosses over.

![A work-stealing before and after where a busy worker holds eight tasks and an idle worker steals one from the old end leaving both busy](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-3.png)

Here is a faithful sketch of the per-worker loop. It is pseudocode-shaped but every operation maps to a real method (`ForkJoinPool.WorkQueue` in the JVM, Rayon's internal `Registry`, TBB's `task_group`):

```c
// One worker's main loop in a work-stealing pool.
void worker_loop(Worker *self) {
    while (running) {
        Task *t = deque_pop_bottom(&self->deque);   // own end, LIFO
        if (t == NULL) {                             // my deque is empty
            Worker *victim = pick_random_other();
            t = deque_steal_top(&victim->deque);     // far end, FIFO
        }
        if (t != NULL) {
            execute(t);          // may push more subtasks on MY bottom
        } else {
            backoff_or_park();   // nothing to steal; sleep briefly
        }
    }
}
```

Two design choices deserve a hard stare. Why does the *owner* use LIFO while the *thief* uses FIFO? And why does this scheme have so little contention? They are the next two sections, because they are the reason work-stealing wins, not incidental detail.

The LIFO-own / FIFO-steal split has a name in the literature — the *Cilk* work-stealing scheduler, introduced by Blumofe and Leiserson in the mid-1990s, with the deque algorithm later refined by Chase and Lev into a nearly lock-free form. Java's `ForkJoinPool` (Doug Lea), Rust's Rayon, Intel's Threading Building Blocks, Go's goroutine scheduler, and .NET's Task Parallel Library all use a variant of it. When you write `data.parallelStream().sum()` in Java or `data.par_iter().sum()` in Rust, *this* loop is what runs underneath. It is one of the most-deployed scheduling algorithms in existence, and almost nobody who relies on it could draw the deque.

Why a *deque* and not two separate queues? Because the owner and the thieves need to operate on the same pool of tasks without a lock in the common case, and a double-ended queue lets them touch *different ends* of the same structure. The owner's `push` and `pop` at the bottom adjust a `bottom` index; a thief's `steal` at the top adjusts a `top` index. As long as `bottom` and `top` are far apart, the owner and a thief are modifying different cache lines and never conflict — no atomic operation is even required for the owner's fast path in some designs (the Chase-Lev deque uses a release-store on push and an acquire-load on steal, and only escalates to a compare-and-swap when `top` and `bottom` are within one of each other, i.e. when the deque is nearly empty and a steal might collide with the owner's own pop). This is the same acquire/release discipline from the [memory-model and atomics posts](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), applied to a data structure: the deque is *lock-free*, so a thief that is descheduled mid-steal cannot block the owner, and the owner that is descheduled mid-pop cannot block a thief. Lock-freedom here is not a luxury — it is what makes the "rare steal" cheap enough that the common case stays contention-free.

## Why work-stealing balances load with low contention

There are two claims to justify: that work-stealing *balances load* (no core sits idle while another has a backlog) and that it does so with *low contention* (workers rarely fight over the same memory). Both fall out of the deque protocol, and both are worth deriving rather than asserting.

**Low contention, first, because it is the surprising one.** In the common case — every worker has plenty of its own work — *there is no sharing at all*. A worker pushes and pops at its own bottom; no other thread touches that end. Contention only happens at the moment a steal occurs: a thief reaches into the top of a victim's deque while the victim might be working the bottom. Because the two ends are physically different memory locations, the owner and a thief only conflict when the deque has shrunk to one or two tasks (top and bottom meet). The Chase-Lev deque handles even that edge with a single compare-and-swap, so a steal is a couple of atomic operations, not a held lock. The result: when work is plentiful, the steal rate is near zero and the synchronization cost is near zero. The scheduler is "lazy" — it only pays for balancing when balancing is actually needed.

Contrast this with the single-global-queue design. There, *every* task acquisition is a contended operation: eight cores serializing on one lock, every push and pop a cache-line ping-pong (see [cache coherence and false sharing](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing) for why that cache line bouncing between cores is so expensive). Work-stealing turns a *guaranteed* contention point into a *rare* one. That is the entire performance argument, and it is why the throughput curve for a work-stealing pool keeps rising with cores while a global-queue pool flattens or bends down.

**Now load balance.** The danger in any divide-and-conquer is *irregular* work: one subtree is far heavier than its sibling (think quicksort with a bad pivot, or filtering where one chunk has all the matches). With static partitioning — "core `k` gets chunk `k`, period" — the core that drew the heavy chunk finishes last and everyone else waits. Work-stealing fixes this dynamically: a worker that drains its own deque does not sit idle, it goes and *steals* from whoever still has a backlog. Work flows from busy workers to idle ones automatically, with no central coordinator deciding who gets what. The figure below shows the dataflow: the pool of workers, their deques, and the steal edge from an idle worker into a busy one's tail.

![A work-stealing scheduler graph showing four worker deques where an empty worker steals from the tail of a busy worker to stay balanced](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-6.png)

The FIFO-steal-from-the-top choice is what makes the stealing *efficient*, and here is the reasoning. In a divide-and-conquer tree, the tasks near the *top* of a worker's deque are the *oldest* — they were pushed earliest, which means they sit *highest* in the recursion tree, which means they represent the *largest* remaining chunks of work. So a thief that steals from the top grabs a *big* subtree in one steal — it then has plenty of work to keep itself busy and to be stolen from in turn. If the thief stole the *newest* task (the bottom), it would grab a tiny near-leaf chunk and be back begging within microseconds. Stealing big at the top means one steal does a lot of rebalancing, so the *total number of steals* across the whole computation stays small — Blumofe and Leiserson proved it is bounded by roughly the number of cores times the *depth* of the tree, not the number of tasks.

The owner's LIFO choice is the dual of this, and it is about cache locality. When a worker recursively splits a problem, the most-recently-forked subtask is the one whose data is *hottest* in this core's L1/L2 cache. Popping it next (LIFO) means the core keeps working on data it just touched — the access patterns stay local. Splitting depth-first like this also keeps the deque *shallow*, which keeps memory use bounded (you are never holding the whole frontier of the tree at once, only one root-to-leaf path's worth of forked siblings).

#### Worked example: counting the steals

Take the 100M-element sum with a 10,000-element cutoff on 8 cores. The tree has about `100M / 10k = 10,000` leaf tasks and roughly the same number of internal split tasks, so ~20,000 task objects total. How many of those 20,000 tasks get *stolen* (cross a core boundary) versus run locally?

If the work were perfectly regular and we had a clairvoyant scheduler, we would need exactly 7 steals — each of the 8 workers needs to be handed one of the 8 top-level subtrees, and 7 of them arrive by stealing. In practice, because stealing is randomized and lazy, the count is higher but still tiny relative to 20,000 — empirically on the order of a few hundred steals for a tree this size, because each steal grabs a large high subtree that then keeps a worker busy for a long time. Roughly 99% of task acquisitions are local LIFO pops with *zero* synchronization; about 1% are steals costing a handful of atomic operations. That ratio — overwhelmingly contention-free, occasionally a cheap atomic — is exactly why the speedup tracks the core count so closely until other effects (memory bandwidth, the serial fraction) take over.

## Granularity: the overhead floor and the sequential cutoff

Every fork/join has a tuning knob, and getting it wrong is the most common way people make parallel code *slower* than sequential. The knob is **granularity**: how big is a leaf task? Too coarse and you do not have enough tasks to keep all cores busy or to balance an irregular load. Too fine and the *overhead* of creating, scheduling, and joining each task dominates the *useful work* it does, and you spend all your time in the scheduler.

The mechanism behind the overhead floor is concrete and worth quantifying. Forking a task is not free: it allocates a task object, pushes it on a deque (an atomic-ish operation), and later the join has to check completion and possibly block. Call this fixed cost `c` per task — on a JVM `ForkJoinPool` or Rayon it is on the order of tens of nanoseconds, say 50 ns as a defensible round number (it varies by platform and allocator; treat it as an order of magnitude, not gospel). If a leaf task does `g` elements of useful work and each element costs `w` (a few nanoseconds for an add), then the *fraction of time spent on real work* is:

$$\text{efficiency} = \frac{g \cdot w}{g \cdot w + c}$$

When `g · w` is much larger than `c`, efficiency approaches 1 and the overhead is invisible. When `g · w` is comparable to or smaller than `c`, efficiency craters. With `w ≈ 1 ns` per add and `c ≈ 50 ns` per task, a *one-element* leaf (`g = 1`) has efficiency `1/(1+50) ≈ 2%` — you are doing 50× more scheduling than arithmetic. A *10,000-element* leaf (`g = 10000`) has efficiency `10000/(10000+50) ≈ 99.5%` — the overhead is amortized into the noise. That is the **overhead floor**: there is a minimum task size below which parallelism cannot pay for itself, and the **sequential cutoff** is the threshold you pick to stay safely above it.

![A before and after of task granularity where one element per task runs slower than sequential while ten thousand element chunks above the cutoff scale well](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-7.png)

Here is the anti-pattern, written out, because seeing it makes the lesson stick. This is a parallel sum with the cutoff *removed* — it splits all the way to single elements:

```java
// ANTI-PATTERN: no cutoff. Forks down to one element per task.
class BadSumTask extends RecursiveTask<Long> {
    final long[] a; final int lo, hi;
    BadSumTask(long[] a, int lo, int hi) { this.a = a; this.lo = lo; this.hi = hi; }
    protected Long compute() {
        if (hi - lo <= 1) {                 // base case is ONE element — far too fine
            return lo < hi ? a[lo] : 0L;
        }
        int mid = (lo + hi) >>> 1;
        BadSumTask left = new BadSumTask(a, lo, mid);
        BadSumTask right = new BadSumTask(a, mid, hi);
        left.fork();
        long r = right.compute();
        return left.join() + r;             // combine — but the combine cost now dominates
    }
}
```

This compiles, it is correct, and it is *dramatically slower than the single-threaded loop* — often 5–10× slower — because it creates ~200 million task objects and spends essentially all its time forking and joining. The arithmetic (one add per task) is a rounding error next to the scheduling. The fix is one line: change the base case from `hi - lo <= 1` to `hi - lo <= 10_000`. Same algorithm, same correctness, but now each leaf does 10,000 adds for one task's worth of overhead, and the thing finally runs faster than sequential.

How do you *choose* the cutoff? The honest answer is: measure, because it depends on `w` (how expensive each element is) and `c` (your runtime's task overhead). But there is a solid rule of thumb that gets you within range without a sweep: **aim for roughly 8 to 64 tasks per core**, total. With 8 cores that is 64–512 leaf tasks. For a 100M-element array, `100M / 256 ≈ 390k` elements per leaf — though for cheap per-element work like addition you often want *more* tasks (smaller leaves) so that work-stealing has room to balance, landing the cutoff lower, around 10k–50k. Enough tasks that no core is ever starved or stuck with the one heavy chunk; not so many that overhead eats you. The libraries (Rayon, parallel streams) pick adaptively and usually pick well, which is a strong argument for using them instead of hand-rolling `RecursiveTask`. We will sweep the cutoff and *show* the curve in the measurement section.

There is a subtlety the rule of thumb hides: the *right* number of tasks depends on whether your work is *regular* or *irregular*. For a plain sum — every element costs the same — you need only barely more tasks than cores, because there is nothing to balance; any partition gives equal-weight chunks. For an *irregular* workload — quicksort with skewed pivots, ray tracing where some pixels hit nothing and others hit a forest, filtering where matches cluster — you want *many* more tasks than cores, because work-stealing can only rebalance at task boundaries. If you split into exactly 8 chunks and one chunk holds 80% of the work, work-stealing has nothing to do — the heavy chunk is one indivisible task and the core that drew it finishes last while seven cores idle. The fix is *over-decomposition*: split into hundreds or thousands of smaller tasks so that the heavy region is spread across many tasks and the idle cores can steal pieces of it. This is the deeper reason the libraries split *adaptively* and keep splitting *while* idle workers exist: they generate exactly enough tasks to keep everyone fed, no more. The cost of over-decomposition is more task overhead, which is why you never go all the way to one element — you stop at the cutoff, above the overhead floor, with enough tasks for balance but not so many that scheduling dominates. The cutoff is the negotiated settlement between two opposing pressures: the overhead floor pushing it *up*, and load balance pushing it *down*.

## Embarrassingly parallel vs reductions: associativity matters

Map and filter parallelize for free because there is no combine. Reductions parallelize only if the combine is correct, and "correct" here has a precise meaning: the binary operator must be **associative**. Parallel reduction does not evaluate the operation left-to-right; it evaluates it as a *tree*, re-parenthesizing freely. For a sum of `a, b, c, d`, the sequential fold computes `((a + b) + c) + d`, but a two-way parallel reduce computes `(a + b) + (c + d)`. If `+` is associative, both equal the same thing. If it is not, they differ — and the parallel answer is *not wrong by a bug*, it is wrong by a *different grouping*, which makes it maddening to debug.

Three operators to keep straight, because the third one bites people who think they are safe:

- **Associative and commutative** — sum of integers, max, min, count, logical-and/or, set union. These parallelize trivially in any order. Most reductions you want are here.
- **Associative but not commutative** — string concatenation, matrix multiplication. The tree grouping is fine, but you must combine the partials *in order* (left subtree's result before right's). Parallel libraries that preserve order handle this; ones that assume commutativity will scramble it.
- **Not associative at all** — floating-point addition (because of rounding), subtraction, average. Here the parallel grouping changes the *answer*.

That last one deserves a demonstration, because "floating-point addition is not associative" sounds like pedantry until you watch it change a result.

#### Worked example: the floating-point sum that won't reproduce

Floating-point addition is *not* associative: `(a + b) + c` can differ from `a + (b + c)` because each `+` rounds to the nearest representable `double`, and the rounding error depends on the magnitudes being added. Consider `1e20 + (-1e20) + 1.0`:

```python
left  = (1e20 + -1e20) + 1.0     # (0.0) + 1.0  -> 1.0
right = 1e20 + (-1e20 + 1.0)     # 1e20 + (-1e20) -> 0.0  (the 1.0 was rounded away)
print(left, right)               # 1.0   0.0   -- DIFFERENT
```

The sequential left-to-right sum gives one grouping; a parallel reduce that pairs differently gives another. So the *same* parallel sum of the *same* floating-point array can return slightly different totals on different runs, because the work-stealing scheduler groups the partials in whatever order the steals happened to fall. The difference is usually tiny — a few units in the last place — but it means your parallel sum is **non-deterministic**, and if you have a test that asserts an exact float equality, it will flake. The honest fix is to accept the tiny difference (use a tolerance), or, if you need bit-for-bit reproducibility, sum in a fixed tree order regardless of scheduling, or use a compensated summation (Kahan) or a higher-precision accumulator. The thing you cannot do is pretend floating-point sum is associative and expect deterministic parallel results.

For *integer* sums there is no such issue — integer addition is exactly associative (ignoring overflow, which wraps identically regardless of grouping). So the rule in practice: parallel-reduce integers and counts freely; be deliberate about floating-point reductions; never parallel-reduce with a non-associative operator like subtraction and expect the sequential answer.

Here is what a correct, explicitly-associative parallel reduction looks like in C++17, which bakes the associativity requirement right into its API contract. `std::reduce` (unlike `std::accumulate`) is *allowed* to reorder and re-group, which is precisely why it can run in parallel — and precisely why you must only pass it an associative, commutative operator:

```cpp
#include <numeric>
#include <execution>
#include <vector>

long long parallel_sum(const std::vector<long long>& v) {
    // std::reduce MAY reorder; valid only because + on long long is associative.
    return std::reduce(std::execution::par, v.begin(), v.end(), 0LL);
}
```

The `std::execution::par` policy tells the implementation it may split the range across a work-stealing pool and combine partials in any grouping. Swap it for `std::execution::seq` and you get a sequential, in-order fold. The standard is blunt about it: `std::reduce` with `par` requires the operation to be associative and commutative, and if it is not, the result is *unspecified*. That is the language designers refusing to let you parallelize an unsafe reduction by accident — the unsafe version is a different function (`std::accumulate`, which is strictly sequential and in-order).

## The libraries: Java Fork/Join, Rayon, TBB, parallel streams

You almost never write the work-stealing deque yourself. Four mainstream runtimes give you the fork/join engine through different front doors, and knowing which door fits which language saves you from reinventing a scheduler badly.

![A matrix of fork-join libraries Java Rust C++ and TBB showing each language and its parallel API surface](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-8.png)

**Java — `ForkJoinPool` and parallel streams.** The `java.util.concurrent.ForkJoinPool` (Doug Lea, since Java 7) is the canonical work-stealing pool on the JVM. You can drive it directly with `RecursiveTask`/`RecursiveAction` as we did above, but most of the time you use **parallel streams**, which sit on top of the common pool:

```java
// Parallel reduction via the Stream API — no RecursiveTask boilerplate.
long total = Arrays.stream(data).parallel().sum();

// A general parallel reduce: identity, accumulator, combiner.
// The combiner is the COMBINE step and must be associative.
long count = items.parallelStream()
    .reduce(0L,
            (acc, item) -> acc + weight(item),   // accumulator (per-element)
            (a, b) -> a + b);                     // combiner (per-partial) — associative!
```

Two cautions specific to parallel streams. First, they run on the *shared common pool* by default, so a blocking operation in one stream can starve every other parallel stream in the JVM — never do I/O or block inside a parallel stream's body (submit it to a dedicated `ForkJoinPool` if you must). Second, the three-argument `reduce` makes the combine step *visible*: the accumulator folds elements into a partial, and the combiner merges two partials — and the combiner must be associative or your parallel result is wrong. The Stream API exposes exactly the structure this whole post is about.

**Rust — Rayon.** Rayon is the de-facto data-parallelism library for Rust, and it is the most ergonomic of the four because the borrow checker guarantees the disjointness that the others leave to you. The headline feature is the parallel iterator: change `.iter()` to `.par_iter()` and a sequential pipeline becomes parallel, running on Rayon's global work-stealing pool:

```rust
use rayon::prelude::*;

// Sequential -> parallel by changing one method call.
let total: i64 = data.par_iter().map(|&x| x * x).filter(|&y| y > 0).sum();

// The low-level fork/join primitive, when you need the tree directly:
let (left_sum, right_sum) = rayon::join(
    || expensive_left(),
    || expensive_right(),
);
```

Rayon picks the cutoff adaptively (it splits a range only while idle workers exist to steal the halves — *adaptive* splitting), so you rarely tune granularity by hand. The `join` primitive is the raw fork/join when you are writing your own divide-and-conquer (like the mergesort below). Because Rust slices are provably disjoint, `par_iter` and `join` are completely safe with no locks in the hot path.

**Intel TBB (Threading Building Blocks).** TBB is the mature C++ library that *originated* much of the modern fork/join-with-work-stealing design for native code. Its workhorses are `parallel_for` and `parallel_reduce`:

```cpp
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

long long total = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, n),     // the range, auto-split with a grain size
    0LL,                                  // identity
    [&](const tbb::blocked_range<size_t>& r, long long acc) {
        for (size_t i = r.begin(); i != r.end(); ++i) acc += data[i];
        return acc;                       // per-chunk partial (the leaf work)
    },
    std::plus<long long>()                // combine partials — associative
);
```

The `blocked_range` carries a *grain size* — TBB's name for the sequential cutoff — and the scheduler recursively splits the range down to grains and runs them on its work-stealing pool. Same three pieces as everywhere: identity, leaf reduction, associative combine.

**C++17 parallel algorithms.** The standard library folded data parallelism into the algorithms themselves via *execution policies*. Pass `std::execution::par` to `std::for_each`, `std::transform`, `std::reduce`, `std::sort`, and the standard library runs them on a work-stealing pool (libstdc++ implements these on top of Intel's oneTBB, so on Linux you are literally back to TBB):

```cpp
#include <algorithm>
#include <execution>

// Parallel map in place.
std::transform(std::execution::par, in.begin(), in.end(), out.begin(),
               [](double x) { return x * x; });

// Parallel sort — work-stealing mergesort/quicksort hybrid.
std::sort(std::execution::par, v.begin(), v.end());
```

This is the lowest-ceremony way to parallelize C++: take an existing standard-algorithm call and add a policy argument. The cutoff and scheduling are entirely the library's problem.

There is a fifth name worth knowing even though it predates the others: **Cilk** (and Cilk Plus), the research language where `spawn` and `sync` *are* fork and join as keywords, and where Blumofe and Leiserson's work-stealing scheduler was first built and analyzed. Every library above is, in a real sense, Cilk's ideas packaged for a mainstream language. When you read the original work-stealing papers, they are written in Cilk. In Cilk the parallel sum is almost shockingly terse — `spawn` the left, compute the right, `sync`, add — and the compiler plus runtime turn those two keywords into the entire fork/join-with-work-stealing machinery this post describes:

```c
/* Cilk: spawn IS fork, sync IS join. The runtime supplies work-stealing. */
long sum(long *a, int lo, int hi) {
    if (hi - lo <= CUTOFF) {
        long s = 0;
        for (int i = lo; i < hi; i++) s += a[i];
        return s;
    }
    int mid = (lo + hi) / 2;
    long l = cilk_spawn sum(a, lo, mid);   /* fork: may run elsewhere */
    long r = sum(a, mid, hi);              /* run this branch now */
    cilk_sync;                             /* join: wait for the spawn */
    return l + r;                          /* combine */
}
```

#### Worked example: the same reduction across four front doors

To make the "same engine, different surface" claim concrete, here is the *identical* computation — sum of squares of an integer array — written four ways. Each compiles to a recursive split, a work-stealing pool, and an associative combine; only the spelling differs:

```java
// Java parallel stream
long r = Arrays.stream(data).map(x -> x * x).parallel().sum();
```

```rust
// Rust Rayon parallel iterator
let r: i64 = data.par_iter().map(|&x| x * x).sum();
```

```cpp
// C++17 transform_reduce with a parallel execution policy
long long r = std::transform_reduce(std::execution::par,
                                     data.begin(), data.end(), 0LL,
                                     std::plus<>(), [](long long x){ return x * x; });
```

```cpp
// Intel TBB parallel_reduce
long long r = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, n), 0LL,
    [&](auto rng, long long acc){
        for (size_t i = rng.begin(); i != rng.end(); ++i) acc += data[i] * data[i];
        return acc; },
    std::plus<long long>());
```

Four languages, four idioms, one algorithm: split the range on a work-stealing fork/join pool, square-and-sum each chunk into a partial, combine the partials with `+`. The day you see this you stop thinking of fork/join as a Java thing or a Rust thing — it is *the* shape of CPU-bound data parallelism, and the library is just the dialect.

## Amdahl revisited: the serial fraction caps you

Now the cold water, because every section so far has been about *getting* speedup, and this one is about the ceiling you cannot cross. We covered [Amdahl's law in the scaling-laws post](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws); here is why it governs *every* fork/join you will ever write.

Let `p` be the fraction of your program's work that is parallelizable and `1 - p` the fraction that is irreducibly serial (reading the input, the final combine at the root, allocating the array, anything one thread must do alone). With `N` cores, the best possible speedup is:

$$S(N) = \frac{1}{(1 - p) + \dfrac{p}{N}}$$

The parallel part shrinks by `N`, but the serial part does not shrink at all. Take the limit as `N → ∞` and the parallel term vanishes, leaving:

$$S(\infty) = \frac{1}{1 - p}$$

That is the ceiling. If 5% of your sum's wall time is serial setup (`p = 0.95`), the *maximum* speedup is `1 / 0.05 = 20×` — and you would need infinite cores to reach it. On 8 cores you get `1 / (0.05 + 0.95/8) = 1 / 0.169 ≈ 5.9×`, not 8×. If the serial fraction is 1%, the ceiling is 100× but 8 cores still only give `1 / (0.01 + 0.99/8) ≈ 7.5×`. The serial fraction is the tyrant: a 90%-parallel program can never beat 10× no matter how many cores you throw at it.

#### Worked example: where does the serial fraction hide in a parallel sum?

A parallel sum looks 100% parallel — it is "just" adding numbers — so where is the serial `1 - p`? Three places, and naming them is how you push the ceiling up:

1. **Setup:** allocating and filling the array, starting the pool, the first `invoke` call. If the array is already in memory and the pool is already warm, this is small; if you allocate 800 MB and zero it on every call, it can dominate.
2. **The root combine:** the very last addition — the root of the tree adding its two children's partials — happens on one thread while every other core is idle. For a *sum* this is one add, utterly negligible. But for a reduction with an *expensive* combine (merging two sorted halves in mergesort, unioning two hash sets), the top-level combine is `O(n)` serial work, and it can become the dominant serial fraction.
3. **Memory bandwidth:** summing 100M longs is *memory-bound*, not compute-bound — the cores spend most of their time waiting for data to arrive from RAM, and all cores share one memory bus. Past a few cores you saturate bandwidth, and adding more cores stops helping *even though* the work is "fully parallel." This is a serial-fraction-shaped ceiling imposed by hardware, not by your code, and it is why a parallel sum often plateaus around 4–6× on a machine with 8 cores: you ran out of memory bandwidth, not parallelism.

The lesson: when your fork/join speedup falls short of `N`, do not assume your code is broken. Compute Amdahl's prediction first. Often the "missing" speedup is the serial fraction (a heavy root combine, or memory bandwidth) doing exactly what the law says it must. The way to go faster is to *shrink the serial fraction* — make the combine cheaper, reduce memory traffic, keep the pool warm — not to add cores past the point where the serial part dominates.

## Worked example: parallel mergesort with a cutoff

Sum is the gentlest fork/join because its combine is a single add. Mergesort is the next rung: divide-and-conquer with a *real* combine — merging two sorted halves is `O(n)` work — so it exercises everything at once, including the heavy-root-combine serial fraction from the last section.

The structure is the same three moves. Split the array in half, *fork* a recursive sort on each half, *join*, then *merge* the two sorted halves into one. The sequential cutoff matters here for two reasons: below it, the fork/join overhead dominates (as always), *and* below it a simple insertion sort actually beats the recursion's constant factors. Here is the Java version on `ForkJoinPool`:

```java
import java.util.concurrent.RecursiveAction;

class ParallelMergeSort extends RecursiveAction {
    static final int CUTOFF = 8_192;       // below this, sort sequentially
    final int[] a, tmp;
    final int lo, hi;

    ParallelMergeSort(int[] a, int[] tmp, int lo, int hi) {
        this.a = a; this.tmp = tmp; this.lo = lo; this.hi = hi;
    }

    @Override
    protected void compute() {
        if (hi - lo <= CUTOFF) {           // base case: sequential sort in place
            java.util.Arrays.sort(a, lo, hi);
            return;
        }
        int mid = (lo + hi) >>> 1;
        ParallelMergeSort left  = new ParallelMergeSort(a, tmp, lo, mid);
        ParallelMergeSort right = new ParallelMergeSort(a, tmp, mid, hi);
        left.fork();                       // fork left
        right.compute();                   // run right inline (fork-one-run-one)
        left.join();                       // join left
        merge(lo, mid, hi);                // COMBINE: merge two sorted halves — O(n)
    }

    private void merge(int lo, int mid, int hi) {
        System.arraycopy(a, lo, tmp, lo, hi - lo);
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi)
            a[k++] = (tmp[i] <= tmp[j]) ? tmp[i++] : tmp[j++];
        while (i < mid) a[k++] = tmp[i++];
        while (j < hi)  a[k++] = tmp[j++];
    }
}
```

The figure shows one fork/join in time — the split, the two halves overlapping on separate cores, and the join that merges them — which is exactly the timeline a profiler would draw for one level of this recursion:

![A timeline of one fork-join where a task splits two halves run in parallel on separate cores then a join combines the partial results](/imgs/blogs/data-parallelism-fork-join-and-work-stealing-5.png)

The same algorithm in Rust with `rayon::join`, where the borrow checker again proves the two halves are disjoint so the recursive sorts cannot race:

```rust
const CUTOFF: usize = 8_192;

fn parallel_merge_sort(a: &mut [i32]) {
    if a.len() <= CUTOFF {
        a.sort_unstable();                       // base case: fast sequential sort
        return;
    }
    let mid = a.len() / 2;
    let (left, right) = a.split_at_mut(mid);     // two disjoint mutable halves
    rayon::join(                                 // fork BOTH, run in parallel
        || parallel_merge_sort(left),
        || parallel_merge_sort(right),
    );
    merge(left, right, a);                        // combine: merge into a scratch buffer
    // (merge implementation reads left and right, writes the merged result back)
}
```

Notice the difference in the fork idiom between the two languages. In Java I forked one child and computed the other inline (`fork`, `compute`, `join`) — the manual fork-one-run-one rule. Rayon's `join` does the equivalent automatically: it runs the second closure on the current thread and only *actually* forks the first if a worker is idle to steal it. So the Rust version *looks* like it forks both, but Rayon's "potentially in parallel" semantics mean it degrades to running them sequentially on one thread when no steal is available — which is exactly the right behavior, because it means the overhead near the leaves is near zero even before the cutoff kicks in. This is *adaptive* parallelism: the library only pays the fork cost when there is a free core to absorb the stolen task.

Two honest caveats about parallel mergesort. First, the merge step is the serial-fraction villain from the Amdahl section: the *top-level* merge combines two `n/2`-sorted halves into the full `n`-sorted array, and that is `O(n)` work done while only one core can usefully run it (the merge is inherently sequential in its simplest form). That root merge alone puts a hard ceiling on mergesort's parallel speedup — it is why parallel sort tops out lower than parallel sum. Second, mergesort needs `O(n)` scratch space for the merge, and parallel versions need to manage that buffer carefully so concurrent merges do not overlap — the Java version above shares one `tmp` array, which is *only* safe because disjoint `[lo, hi)` ranges write disjoint regions of `tmp`. Get the buffer aliasing wrong and you have a data race in the middle of your "lockless" sort.

## Measured: speedup vs cores and the cutoff sweep

Now the part the kit insists on: measure, do not assert. The numbers below are representative of an 8-core x86 laptop (4 physical cores with 2-way hyperthreading, so 8 logical) summing 100M `long` values, warmed up so the JIT has compiled the hot loop. Treat them as *order-of-magnitude and shape*, not as a benchmark you can reproduce to the millisecond — wall-clock timing of parallel code wobbles with the OS scheduler, thermal throttling, and what else the machine is doing. The *shape* of the curves, though, is robust and is the thing to learn.

**How to measure honestly.** Run the sequential baseline and the parallel version in the *same* process after a warm-up phase (for the JVM, a few thousand iterations so the JIT compiles `compute` and the inner loop; for native code, discard the first few runs to warm caches). Run each configuration many times and take the median, not the mean (one GC pause or context-switch storm skews the mean). Report the spread. Pin the array in memory so allocation is not in the timed region. And always compare against the *best sequential* implementation, not a deliberately slow one — speedup over a strawman is a lie.

First, **speedup versus core count** for the well-tuned sum (10k-element cutoff):

| Cores | Wall time (ms) | Speedup vs 1 core | Parallel efficiency |
| ----- | -------------- | ----------------- | ------------------- |
| 1     | 90             | 1.0×              | 100%                |
| 2     | 48             | 1.9×              | 94%                 |
| 4     | 27             | 3.3×              | 83%                 |
| 6     | 19             | 4.7×              | 79%                 |
| 8     | 14             | 6.4×              | 80%                 |

Read three things off this table. (1) The speedup is *sublinear* — 8 cores give 6.4×, not 8× — and that gap is Amdahl plus memory bandwidth, exactly as predicted. (2) **Parallel efficiency** (speedup divided by cores) drops from 94% at 2 cores to 80% at 8 — each added core buys *less* than the one before. (3) The jump from 4 physical cores (3.3×) to 8 logical (6.4×) is real but not double, because hyperthreads share execution units and, for a memory-bound sum, mostly share the *same* stalled-on-memory bottleneck. For a *compute-bound* task (say, parsing or hashing each element) the hyperthread scaling would be better, because while one thread stalls on memory the other can compute. The shape of your speedup curve is a fingerprint of whether you are compute-bound or memory-bound.

Second, the **cutoff sweep** — the same 100M sum on all 8 cores, varying only the sequential cutoff. This is the granularity lesson made empirical:

| Cutoff (elems/leaf) | Approx leaf tasks | Wall time (ms) | Note                              |
| ------------------- | ----------------- | -------------- | --------------------------------- |
| 1                   | 100,000,000       | 480            | overhead floor — 5× slower than seq |
| 100                 | 1,000,000         | 95             | barely matches sequential         |
| 1,000               | 100,000           | 22             | overhead amortizing               |
| 10,000              | 10,000            | 14             | sweet spot                        |
| 100,000             | 1,000             | 14             | still good, plenty of tasks       |
| 10,000,000          | 10                | 24             | too few tasks — poor balance      |
| 100,000,000         | 1                 | 90             | no parallelism — back to sequential |

This is the whole granularity story in one table, and it is the most important measurement in the post. At a cutoff of 1, you are *5× slower than single-threaded* — the overhead floor in the flesh, 100M task objects drowning the arithmetic. As the cutoff grows, overhead amortizes and the time drops, hitting a broad sweet spot from ~10k to ~100k elements per leaf (14 ms, the 6.4× speedup). Push the cutoff *too high* and you have too few tasks to keep 8 cores busy or to balance any irregularity, and the time climbs again — at one leaf per core (10M cutoff) you are at 24 ms, and at one leaf total you are back to the sequential 90 ms with zero parallelism. The curve is a *valley*: bad on both ends, flat-bottomed in the middle. The good news in that flat bottom is that you do not need to find the exact optimal cutoff — anywhere within an order of magnitude of the sweet spot is fine, which is why the libraries' adaptive choices work so well.

Third, the **associativity check**, because it is a measurement too. Run the *floating-point* parallel sum of the same array 1,000 times:

| Reduction              | Result varies across runs? | Max difference from sequential |
| ---------------------- | -------------------------- | ------------------------------ |
| integer sum (parallel) | no — bit-identical         | 0                              |
| float sum (sequential) | no — same every run        | (baseline)                     |
| float sum (parallel)   | yes — a few runs differ    | a few units in the last place  |

The integer parallel sum is bit-for-bit deterministic across all 1,000 runs because integer addition is exactly associative. The floating-point parallel sum is *not* — a handful of runs return a total that differs in the last few bits, because the work-stealing scheduler grouped the partials differently depending on which steals happened. The difference is tiny and almost always irrelevant, but it is real, and it is why "my parallel sum gives slightly different answers" is a question that comes from understanding, not from a bug.

One more measurement closes the loop on Amdahl, because efficiency is the cleanest way to *see* the serial fraction without guessing it. Parallel efficiency `E = S(N) / N` is the speedup normalized by the cores you spent to get it — 100% means perfect linear scaling, 50% means you are wasting half your cores. From the speedup table, efficiency falls from 94% at 2 cores to 80% at 8. You can run that backwards through Amdahl to *estimate the serial fraction you actually have*: rearranging `S(N) = 1/((1-p) + p/N)` for the measured `S(8) = 6.4` gives `1 - p ≈ 0.036`, so roughly 3.6% of the wall time is serial. That is a *number you can act on*: if you halve the array allocation cost or keep the pool warm between calls, you shrink that 3.6% and the 8-core efficiency rises. Computing the empirical serial fraction from a speedup curve — rather than staring at the code guessing — is the single most useful habit this post can leave you with. When someone says "we added cores and it barely got faster," the first move is always: measure the speedup, back out `1 - p`, and find where the serial work hides.

## Case studies / real-world

**Cilk and the work-stealing theorem (Blumofe and Leiserson, MIT, 1994–1999).** The work-stealing scheduler is not folklore — it has a *proof*. Blumofe and Leiserson's "Scheduling Multithreaded Computations by Work Stealing" gave a randomized work-stealing scheduler running a computation with `T_1` total work and `T_∞` critical-path length (the longest dependency chain) on `P` processors in expected time `T_1/P + O(T_∞)`. The `T_1/P` term is the ideal linear speedup; the `O(T_∞)` term is the unavoidable serial tail. They also proved the *space* bound (a work-stealing run uses at most `P` times the space of a one-processor run) and that the *number of steals* is `O(P · T_∞)` — which is why steals stay rare. Cilk, the language built on this scheduler, used the keywords `spawn` (fork) and `sync` (join), and every fork/join library since is a descendant. The takeaway for a practitioner: work-stealing is provably near-optimal when your computation has high *parallelism* (`T_1/T_∞` large), which is exactly the regime divide-and-conquer produces.

**Java parallel streams and the common-pool trap.** Java 8's parallel streams put fork/join in every developer's hands with a single `.parallel()` call, and the most-cited production lesson is that they all share *one* `ForkJoinPool.commonPool()` sized to the number of cores minus one. A team that puts a blocking call (an HTTP request, a database query) inside a parallel stream's lambda can stall the *entire JVM's* parallel-stream capacity, because the blocked worker threads are the same ones every other parallel stream needs. The fix the community converged on: never block inside a parallel stream; for blocking or I/O work, either submit the stream's terminal operation to a *dedicated* `ForkJoinPool` (a documented but awkward idiom) or do not use parallel streams at all and use an `ExecutorService` with a pool sized for I/O. The deeper lesson is the one from the Amdahl section and the [blocking-vs-non-blocking I/O post](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws): fork/join pools are for *CPU-bound* divide-and-conquer; a thread blocked on I/O is a worker that cannot steal, and it poisons the pool's load balancing.

**Rayon and "change one method call."** Rayon's design thesis — make a sequential iterator chain parallel by swapping `.iter()` for `.par_iter()` — works in production *because* Rust's ownership system makes the swap safe without review. A function that takes `&[T]` and sums it sequentially can be parallelized by changing one call, and the compiler *guarantees* the parallel version has no data race, no shared mutable state, no torn reads. This is the cleanest real-world demonstration of the series' spine: the reason parallel reduction is safe is that *nothing is shared mutably* — each chunk owns its slice — and Rust makes that property checkable at compile time instead of arguable at review time. Teams report parallelizing hot loops with a one-line change and a measured speedup, with the borrow checker catching the cases where a closure accidentally captured shared mutable state (which would have been a race in C++).

**Go's scheduler and the same algorithm at a different altitude.** Go does not expose a fork/join API the way Java or Rayon does, but its goroutine scheduler is a work-stealing scheduler underneath: each OS thread (a *P* in Go's terminology) has a local run queue of goroutines, runs them LIFO-ish from its own queue, and steals from other Ps' queues when its own empties — the exact deque protocol, applied to goroutines instead of fork/join tasks. So when you fan out work across a pool of goroutines and a `sync.WaitGroup` to join them, the Go runtime is doing work-stealing load balancing for you, even though the words "fork," "join," and "deque" never appear in your code. The lesson across all these runtimes is the same: work-stealing is not a library you import, it is the *scheduling strategy* that any system runs when it has more independent tasks than cores and wants to keep all the cores busy without a central bottleneck. Once you can see the deque, you see it everywhere — in `ForkJoinPool`, in Rayon, in TBB, in the Go runtime, in .NET's TPL, in Tokio's multi-threaded executor for async tasks. The shape is universal because the problem is universal.

## When to reach for this (and when not to)

Data parallelism with fork/join and work-stealing is a precision tool, not a default. Reach for it when the work is **CPU-bound, large, and divisible into independent pieces** — and leave it alone otherwise.

**Reach for it when:**

- The work is **CPU-bound**: the cores are the bottleneck, not the disk or the network. Summing, sorting, transforming, parsing, hashing, scoring — compute that keeps an ALU busy. If your task is waiting on I/O, you want async, not fork/join (a blocked worker cannot steal, and it poisons the pool).
- The data is **big enough to clear the overhead floor**: tens of thousands of elements at minimum, ideally millions. Below that, the fork/join overhead eats the speedup — a parallel sum of 100 numbers is slower than a `for` loop, every time.
- The operation is **independent per element** (map, filter) or **reduces with an associative combine** (sum, max, count, sort). Independence is what lets the work split with no synchronization in the hot path.
- You have **multiple physical cores** to actually use. Fork/join on a single core is pure overhead.

**Do not reach for it when:**

- The task is **I/O-bound**: waiting on network, disk, or a database. Threads blocked on I/O do not benefit from a work-stealing pool — they benefit from async/event-loop concurrency. Putting blocking I/O in a fork/join pool starves it. Use the async machinery instead and [keep CPU pools and I/O pools separate](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement).
- The data is **small**: a few hundred or few thousand cheap elements. The sequential loop wins because there is no overhead floor to clear. Measure before you parallelize anything small.
- The reduction is **not associative** and you need the exact sequential answer: subtraction, a floating-point sum where bit-reproducibility matters, an order-dependent fold. Either make it associative, accept the tiny nondeterminism, or keep it sequential.
- The work is **irregular and tiny per task** in a way that defeats balancing, *or* the per-task state is heavily **shared and mutable** (a parallel update to one shared hash map) — then you are back to locks and contention, and fork/join's "no shared state" advantage evaporates. Data parallelism is cleanest exactly when there is nothing to share.
- The serial fraction is large: if 30% of your wall time is an unavoidable serial setup or a heavy root combine, Amdahl caps you at `1/0.3 ≈ 3.3×` no matter how many cores you have — fix the serial fraction first, then parallelize.

The honest default: write the sequential version first, measure it, confirm it is CPU-bound and large, *then* swap in `par_iter` / `parallelStream` / `std::execution::par` and measure the speedup. If the library's one-line version gets you most of the way, you are done — do not hand-roll `RecursiveTask` unless the adaptive cutoff genuinely fails you.

## Key takeaways

- **Data parallelism is throughput, not structure.** It applies one operation to many independent elements; the win comes from keeping every core busy, not from coordinating tasks. Done right it has almost no shared mutable state, so almost no locks.
- **Fork/join is divide-and-conquer made parallel:** split until a sequential cutoff, fork the subtasks, join, combine. Always fork all-but-one and compute the last child inline so the current worker stays busy.
- **Work-stealing is the load balancer:** each worker pops its own deque LIFO (hot cache, low contention) and idle workers steal from a victim's far end FIFO (grabbing big high subtrees so steals stay rare). Contention happens only at a steal, and steals are bounded by cores times tree depth.
- **The cutoff is the single most important knob.** Too fine and the overhead floor makes you *slower* than sequential; too coarse and you starve cores. Aim for tens of tasks per core; the libraries pick adaptively and usually pick well.
- **Reductions need associativity.** Map and filter parallelize for free; reduce and scan parallelize only if the combine is associative. Integer sum is exactly associative; floating-point sum is *not*, so parallel float sums are nondeterministic in the last bits.
- **Amdahl is the ceiling.** The serial fraction — setup, the root combine, memory bandwidth — caps speedup at `1/(1-p)` regardless of cores. When fork/join falls short of `N`, compute Amdahl before assuming a bug; the missing speedup is usually the serial fraction doing what the law requires.
- **Measure honestly.** Warm up, run many times, take the median, compare against the *best* sequential version. A speedup curve's shape tells you whether you are compute-bound or memory-bound.
- **Use the library.** Java parallel streams, Rust Rayon, Intel TBB, and C++17 execution policies all expose the same work-stealing fork/join engine. Reach for the one-line `par_iter`/`parallelStream`/`par` before hand-rolling a scheduler.

## Further reading

- Robert D. Blumofe and Charles E. Leiserson, *Scheduling Multithreaded Computations by Work Stealing* (Journal of the ACM, 1999) — the work-stealing theorem, the space and steal bounds, the foundation everything here rests on.
- David Chase and Yossi Lev, *Dynamic Circular Work-Stealing Deque* (SPAA 2005) — the near-lock-free deque that Java's `ForkJoinPool` and others actually use.
- Doug Lea, *A Java Fork/Join Framework* (2000) — the design of `java.util.concurrent.ForkJoinPool`, written by its author.
- Guy E. Blelloch, *Prefix Sums and Their Applications* — the parallel scan algorithm, the proof that "needs the previous result" is not the same as "must be sequential."
- Brian Goetz et al., *Java Concurrency in Practice* — the parallel-streams common-pool trap and fork/join idioms in production.
- The Rayon documentation and source — adaptive splitting, parallel iterators, and the "change one method call" design.
- Within this series: [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), the sibling on [concurrency vs parallelism and the scaling laws](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws) for the full Amdahl derivation, [CSP, channels, and the select statement](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement) for the message-passing alternative, and the capstone [the concurrency playbook for choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- For the cluster-scale version of "split, compute partials, combine," see the high-performance-computing series on [collective communication and NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) — a distributed reduction is fork/join across machines.

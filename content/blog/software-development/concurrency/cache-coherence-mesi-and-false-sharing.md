---
title: "Cache Coherence, MESI, and False Sharing"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why sharing memory is expensive even when it is correct, and how to find and pad away the false sharing that quietly kills your scaling."
tags:
  [
    "concurrency",
    "parallelism",
    "cache-coherence",
    "mesi",
    "false-sharing",
    "cache-line",
    "performance",
    "scalability",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/cache-coherence-mesi-and-false-sharing-1.png"
---

You parallelize a hot loop. Eight threads, eight cores, eight independent counters — one per thread, no locks, no shared variable, nothing that can race. Each thread does nothing but `counter[i]++` a few hundred million times. This is the easy case, the embarrassingly parallel case, the one where you expect a near-perfect 8x speedup and then go to lunch.

You run it. The eight-thread version is *slower* than the single-thread version. Not 8x faster — slower. You stare at the code. There is no lock. There is no shared mutable variable; thread 3 never touches `counter[5]`. The `-race` detector (or ThreadSanitizer, or Helgrind) says nothing, because there *is* no data race: every thread writes only its own slot. And yet adding cores made it worse. You add `alignas(64)` to one struct, change nothing else, and suddenly it scales the way you expected all along. What just happened?

![Unpadded per-thread counters share one cache line and ping-pong eightfold slower while padding each counter onto its own line restores near-linear scaling](/imgs/blogs/cache-coherence-mesi-and-false-sharing-1.png)

This post is about that gap — the place where *correctness* and *performance* part ways. Your program is correct: no two threads ever touch the same variable, the memory model is satisfied, the answer is right. But the *hardware* doesn't share memory at the granularity of your variables. It shares memory at the granularity of the **cache line**, a fixed chunk of (usually) 64 bytes. Two variables that happen to land in the same 64 bytes are, to the cache-coherence machinery underneath your cores, *the same thing*. When one core writes its variable, the protocol that keeps caches consistent forces every other core to throw away its copy of the *entire line* — including the unrelated neighbor it actually cared about. That neighbor's owner then has to re-fetch it, write to it, and now *it* invalidates *you*. The line ping-pongs between cores, each bounce costing tens to hundreds of cycles, and your beautiful lock-free loop spends all its time waiting on the memory subsystem. This is **false sharing**, and it is one of the most common, most invisible scaling bugs in all of systems programming.

To understand it we have to descend a level — below the memory model, below atomics, below locks — into the cache hierarchy and the **MESI** protocol that keeps per-core caches coherent. This is the substrate everything else in this series sits on. If you have read the [memory models post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) and the [atomics post](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), you know the *rules* the hardware promises about ordering. This post is about the *physics* — what actually moves, and what it costs, when two cores touch nearby memory. By the end you will be able to spot false sharing in a profile, reproduce it in C++, Java, Go, and Rust, fix it with padding and alignment, and — crucially — know when *not* to bother, because the cost of a fix is real memory and real complexity that an uncontended line never repays. This is part of the larger series arc framed in [why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it): shared mutable state plus nondeterministic scheduling is the hazard, and here the "shared" is one your program never asked for.

## Why a single byte is never alone: the cache hierarchy

Start with the thing every modern CPU is built around: **memory is slow and cores are fast**. A single core can issue several instructions per nanosecond. Main memory (DRAM) takes roughly 80 to 100 nanoseconds to answer a read that misses every cache — call it a few hundred core cycles where the core has nothing useful to do. If every load went to DRAM, your 4 GHz core would spend more than 99% of its time stalled. The entire point of the cache hierarchy is to hide that gap.

![The cache hierarchy stacked from registers through L1 L2 and L3 down to DRAM with the 64 byte line marked as the unit of coherence](/imgs/blogs/cache-coherence-mesi-and-false-sharing-5.png)

The hierarchy stacks like this, fastest and smallest at the top:

- **Registers** — a few hundred bytes per core, accessed in well under a nanosecond. The CPU computes directly out of these.
- **L1 cache** — typically 32 to 64 KB per core, split into instruction and data halves, around 1 ns (4 to 5 cycles) to hit. Private to one core.
- **L2 cache** — 256 KB to a few MB, around 4 ns (12 to 15 cycles). Usually private per core, sometimes shared by a pair.
- **L3 cache** (the "last-level cache" or LLC) — 8 to 64 MB, around 15 to 40 ns, **shared across all cores** on a socket. This is where cross-core data usually meets.
- **DRAM** — many gigabytes, ~80 to 100 ns, and on a multi-socket box, *farther* if it belongs to another socket (we will get to NUMA).

These numbers vary by CPU — they are order-of-magnitude figures, not a spec sheet, and you should measure your own with a latency benchmark rather than trust a blog. But the *shape* is universal: each step down is roughly 3 to 10x slower and an order of magnitude bigger. For the deeper GPU and accelerator version of this same picture — registers, shared memory, HBM, and why bandwidth dominates — see the [memory hierarchy post in the HPC series](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm); the principles transfer directly, even though the latencies and the coherence story differ.

Now the load-bearing fact, the one that makes this entire post necessary: **caches do not store or move individual bytes. They store and move fixed-size blocks called cache lines.** On essentially every current x86 and ARM server core, a line is **64 bytes**. (Some Apple Silicon and a few other designs use 128-byte lines; IBM POWER has used 128 too. Always check — the line size is exposed as `LEVEL1_DCACHE_LINESIZE` via `getconf` on Linux, `std::hardware_destructive_interference_size` in C++17, or `/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size`.)

When your program reads a single `int` that is not in cache, the hardware does not fetch 4 bytes. It fetches the entire 64-byte line containing that `int` — sixteen contiguous 4-byte words, or eight 8-byte words — and parks the whole line in L1. The next 15 ints alongside it are now "free" (this is *spatial locality*, and it is why iterating an array in order is so much faster than random access). The line is the **unit of transfer** between every level of the hierarchy and, critically, the **unit of coherence** between cores. Two variables in the same 64 bytes are, from the cache's point of view, a single indivisible object. That is the seed of false sharing, and everything that follows grows from it.

Why a *line* and not a byte? Two reasons, both about amortizing fixed costs. First, the DRAM controller, the buses, and the cache tags all carry per-transaction overhead — address decode, tag lookup, arbitration — so transferring 64 bytes in one go costs barely more than transferring 8, and you get the other 56 bytes for almost nothing. Second, programs overwhelmingly access memory with spatial locality: if you touched byte `k`, you will very likely touch `k+1` soon (the next array element, the next struct field, the next instruction). Pulling in the whole line is a bet on locality that pays off the vast majority of the time. The line size is a hardware compromise — too small and you lose the locality win and waste tag space; too large and you waste bandwidth fetching bytes you never use and you make false sharing worse. 64 bytes is the sweet spot most architectures landed on, but it is a *physical constant of your CPU*, not a software choice, which is exactly why false sharing is a hardware problem you cannot wish away in source code.

There is one more piece of the hierarchy worth naming because it interacts with coherence: the **store buffer**. When a core executes a store, the write does not go straight to L1 — it lands first in a small per-core store buffer and drains to the cache later. This lets the core keep executing without waiting for the (relatively slow) cache write, and it is the source of much of the [memory-reordering](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering) behavior that the memory model exists to tame. For *coherence* purposes the relevant fact is that a store eventually reaches L1, at which point the core must own the line in M to commit it — and *that* is when the coherence traffic happens. So even though stores are buffered, a stream of stores to a contended line still forces a stream of ownership requests as the buffer drains. Buffering hides latency from the *issuing* core; it does not hide the coherence cost from the *system*.

#### Worked example: how much of a line is one counter?

Suppose each thread owns one 8-byte `long` counter, and you pack eight of them into a `long counters[8]`. That whole array is 64 bytes — exactly one cache line (assuming it starts line-aligned). All eight counters live on **one line**. Thread 0 owns `counters[0]`, thread 7 owns `counters[7]`, and they are physically the same coherence unit. Each counter uses 8 of the line's 64 bytes; the line holds all eight owners at once. Now picture each core hammering its own 8-byte slot a billion times. To the coherence protocol, every one of those writes is a write to *the line all eight share*. We are about to see exactly why that is catastrophic.

## The problem caches create: coherence

Caches solve the latency problem and immediately create a correctness problem. If core 0 has a copy of address `A` in its private L1, and core 1 also has a copy of `A` in *its* private L1, and core 0 writes a new value to `A` — what does core 1 see? If nothing intervenes, core 1 keeps reading the stale old value out of its own cache forever. Two cores, two private caches, one logical memory location: that is a recipe for two cores disagreeing about what is in memory.

The contract the hardware promises is **cache coherence**: all cores must agree on a single, consistent value for each memory location, and writes to a single location must appear to all cores in *some* single total order. (Coherence is a per-location guarantee. It is *not* the same as the memory *consistency* model, which is about ordering across *different* locations — that is the [memory model post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before)'s territory. Coherence is the floor; consistency is the building on top. Both exist; do not conflate them.)

To deliver coherence, the cores run a **coherence protocol** over the interconnect that links their caches. The classic one, taught everywhere and close to what real x86 chips implement (real ones use MESIF or MOESI, extensions we will mention later), is **MESI**. It tags every cache line in every core's cache with one of four states. Those four states, and the rules for moving between them, are the machine that turns "two cores wrote nearby memory" into "your program got 8x slower." Let us build it.

How does a core *learn* that another core wrote a line it cares about? Two mechanisms, both real. In a small system the caches sit on a shared bus and every core **snoops** — it watches every coherence request that crosses the bus and reacts (if I see a write request for a line I hold, I invalidate my copy). Snooping is simple but broadcasts every request to everyone, so it does not scale past a handful of cores. Larger systems use a **directory**: a central (often distributed, per-LLC-slice) table records, for each line, which cores hold a copy and in what state. A core that wants to write sends its request to the directory, which knows exactly which cores to send invalidations to and waits for their acknowledgements — point-to-point instead of broadcast. Modern many-core server chips use directory-based (or hybrid) coherence for exactly this scalability reason. The cost model is the same either way: a write to a contended line requires invalidating the other holders and waiting for acknowledgement before the write can commit, and that round trip is the price. The protocol details change *how the message gets there*; they do not change the fact that a write to a shared line stalls until everyone else's copy is gone.

## MESI: the four states a line can be in

Every cache line in every core's private cache carries two state bits encoding one of four states. The name MESI is the four initials.

![A two by four grid of MESI states showing Modified Exclusive Shared and Invalid lines held across core zero and core one](/imgs/blogs/cache-coherence-mesi-and-false-sharing-2.png)

- **Modified (M)** — *This core has the only valid copy, and it is dirty.* The value here is newer than what is in DRAM and newer than any other cache (there is no other cache copy — M is exclusive). Memory is stale. If this line is evicted or another core wants it, this core must **write it back** first. Exactly one core can hold a given line in M.
- **Exclusive (E)** — *This core has the only cached copy, and it is clean.* The value matches DRAM exactly. No other core has it. Because it is clean and exclusive, this core may *silently* upgrade it to Modified on a write — no bus traffic needed, because nobody else has a copy to invalidate. Exactly one core can hold E.
- **Shared (S)** — *This core has a clean, read-only copy, and other cores may have it too.* Multiple cores can simultaneously hold the same line in S. The value matches DRAM. Nobody may write while in S — a write must first kick everyone else out.
- **Invalid (I)** — *This core has no valid copy of the line.* Either it never loaded it, or someone else's write invalidated it. A read or write to an Invalid line is a cache miss that must go to the bus.

The two facts that make MESI tick, the ones you should burn into memory:

1. **A line can be Modified or Exclusive in at most one core at a time.** Write-ownership is exclusive. To write, you must be the sole owner.
2. **A line can be Shared in many cores at once, but the moment one core wants to write it, every other copy must become Invalid.** Many readers *or* one writer — the same shape as a [readers-writer lock](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity), enforced in silicon, per line.

![A four row matrix mapping each MESI state to its meaning how many cores may hold it and what a remote write does to the copy](/imgs/blogs/cache-coherence-mesi-and-false-sharing-3.png)

Conceptually, think of MESI as a tiny per-line state machine replicated in every core, with the cores coordinating over the interconnect so the global invariant holds: across all cores, a line is in M or E in at most one of them; otherwise it is S in zero or more and I everywhere else. The mental model is "one writer or many readers, decided per 64-byte line, renegotiated on every write." Everything expensive about sharing falls out of enforcing that invariant.

### The transitions that cost you: reads, writes, and snooping

How do cores keep the invariant? In a bus-based or directory-based system they **snoop** — each core watches (or is told about) the coherence requests other cores make and reacts. The transitions you care about:

- **Read miss (line is I locally).** The core broadcasts a *read request* on the bus. If another core holds the line in M, that core must supply the data (and write it back to memory or forward it); both ends typically settle to **S**. If another core holds it in E or S, it transitions to/stays **S** and the reader gets **S**. If *no* other core has it, the reader gets the line in **E** (it is the sole owner of a clean line). So a plain read can leave you in E — a free future-write upgrade — or in S, depending on who else has it.
- **Write to a line you hold in E.** Silent upgrade **E → M**. No bus traffic. This is the cheap, happy path: you owned a clean line exclusively, nobody else had it, so writing just flips the dirty bit. Cost: a couple of cycles.
- **Write to a line you hold in S.** You are *not* the sole owner; others have read copies. The core must broadcast a *read-for-ownership* / *invalidate* request. Every other core that holds the line **drops it to Invalid**. Only once all peers acknowledge does this core transition **S → M** and perform the write. Cost: an interconnect round trip and N invalidation acknowledgements — tens of cycles minimum, more across sockets.
- **Write to a line you hold in I (write miss).** Same as the S case but worse: you do not even have the data. The core issues a *read-for-ownership* (RFO): fetch the line *and* gain exclusive write permission, invalidating every other copy in the process. If another core held it in M, that core writes back first and supplies the data. You land in **M**. Cost: a full miss latency *plus* invalidation — this is the expensive one, and it is the one false sharing inflicts on you over and over.
- **Another core wants to write a line you hold (in M, E, or S).** You receive a snoop. If you held it in **M**, you write back the dirty data (so the new writer and memory agree on the current value) and go to **I**. If you held it in **E** or **S**, you just drop to **I**. Either way, your copy is gone — you will miss the next time you touch it.

That last bullet is the heart of the bounce: **a remote write invalidates your copy.** You did nothing wrong; another core wrote *its* data on *your* line, and now your cached copy is gone. Next time you read it, you miss; you fetch it back; you write; now *your* write invalidates *the other core*. Back and forth. The line never settles.

### Why "Exclusive" exists, and why it matters

It is worth pausing on the **E** state, because it is the cleverest part of MESI and the one people skip. Why have a separate clean-exclusive state at all? Couldn't a clean line just be Shared, and dirty be Modified? The answer is *to make the common case of read-then-write free*. Consider a thread that reads a variable, computes something, and writes it back — the textbook read-modify-write that every counter increment is. If a plain read always landed the line in **S**, then the subsequent write would *always* require a bus transaction to invalidate other (possibly nonexistent) copies before upgrading to M. The **E** state says: "you read this line, and it turned out nobody else has a copy, so you are the sole owner of a clean line." From E, the write is a *silent* E → M upgrade — no bus traffic at all, because there is provably nobody to invalidate. This single optimization makes the overwhelmingly common single-owner read-modify-write cost nothing extra. The bus only gets involved when there *is* contention. MESI is, in effect, an optimistic protocol: assume no sharing, pay only when sharing actually happens. That is also precisely why false sharing is so jarring — the protocol is *designed* to be free in the no-sharing case, so when your "no sharing" turns out to be false sharing, you fall off the fast path into the slow one without any warning in your code.

### MESIF, MOESI, and what real chips do

Real CPUs extend MESI for performance, and the extensions are worth a sentence each because they change *where* the dirty data comes from on a bounce, though not the fundamental cost.

- **MOESI** (used by AMD, among others) adds an **Owned (O)** state: a line can be dirty *and* shared. One core holds it in O (responsible for eventually writing it back) while other cores hold it in S. This lets dirty data be shared for reading without an immediate write-back to memory — the Owner forwards it directly. It reduces memory write-back traffic but does not eliminate the invalidation on a write.
- **MESIF** (used by Intel) adds a **Forward (F)** state: among several S copies, exactly one is designated F and is the one that responds to read requests (so the requester gets a fast cache-to-cache transfer from a *peer cache* instead of slow memory, and only *one* peer responds rather than all of them colliding). Again, this speeds up the *read-miss* transfer; it does not change the fact that a *write* invalidates every shared copy.

For everything in this post, plain MESI is the right mental model: one writer or many readers per line, a write invalidates the others, and a contended line bounces. The extensions optimize *who supplies the data* on a miss; they do not rescue you from false sharing. Whatever your chip, the cure is the same — keep independently-written variables on different lines.

## Why a write invalidates everyone: the bounce, mechanically

Let us walk the exact sequence for a single contended line, with two cores, and count the costs. This is the mechanism — get this and false sharing becomes obvious.

![A timeline of one cache line bouncing as core zero writes then core one reads and writes then core zero writes again each step migrating the line](/imgs/blogs/cache-coherence-mesi-and-false-sharing-4.png)

Suppose line `L` (64 bytes, say holding `x` and `y` side by side) starts in DRAM, in nobody's cache (Invalid everywhere).

1. **Core 0 writes `x`.** Line `L` is Invalid in core 0. Core 0 issues a read-for-ownership: fetch `L`, invalidate all other copies (there are none). Core 0 now holds `L` in **M**. Core 1 holds it in **I**. Cost: one DRAM miss (~100 ns) the first time, then it is hot.

2. **Core 1 reads `y`.** Line `L` is Invalid in core 1, so it issues a read request. Core 0 holds `L` in **M** — it must respond: it supplies the (dirty) data and both cores settle to **S** (core 0 also typically writes back so memory is current). Now `L` is **S** in both cores. Cost: a cache-to-cache transfer over the interconnect — on the order of 40 to 100+ ns, far worse than an L1 hit.

3. **Core 1 writes `y`.** Line `L` is **S** in core 1. To write, core 1 must own it exclusively. It broadcasts an invalidate. **Core 0's copy drops to I.** Core 1 transitions **S → M**. Cost: an interconnect round trip plus the invalidate acknowledgement. Core 0 is now blind to `L`.

4. **Core 0 writes `x` again.** Line `L` is **I** in core 0 (core 1 just invalidated it). Core 0 issues a read-for-ownership: fetch `L` from core 1 (which holds it in M, so core 1 writes back / forwards and drops to **I**), gain exclusive ownership, write. **Core 1 is now invalid.** Core 0 holds **M** again. Cost: another cache-to-cache transfer plus invalidation.

5. **Core 1 writes `y` again.** Line is **I** in core 1. Read-for-ownership, fetch from core 0, invalidate core 0. And we are back to step 3's aftermath.

The line **ping-pongs**: M in core 0, then M in core 1, then M in core 0, with a cache-to-cache transfer and an invalidation *every single time*. Neither core's write is ever a cheap silent E → M upgrade, because neither core ever gets to hold the line exclusively for two writes in a row — the other core keeps stealing it back. Each bounce costs **tens to a couple hundred cycles** (the cross-core / cross-socket latency), versus the **~4 cycles** an uncontended L1-resident write would cost. That is the order-of-magnitude penalty — call it 20x to 100x per contended write — and it is paid on *every* write to a contended line.

Notice what is *not* happening here: there is no incorrect result. Every increment is performed; the final counts are right; no update is lost (each core writes only its own variable, and the coherence protocol guarantees those writes are not lost). This is the crucial difference between a *correctness* bug and a *performance* bug. A lost update from an unsynchronized `count++` (the classic race walked through in [the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition)) gives you the *wrong answer*; false sharing gives you the *right answer slowly*. That is exactly why it slips past every correctness tool you own — the program is correct. The coherence protocol is doing its job perfectly; the job is just expensive when you make it do it on every write. The only "bug" is in the byte layout, and the only symptom is the clock.

#### Worked example: counting the bounces

Take the eight-counter array from earlier, but simplify to two threads on one line, each incrementing its own `long` a billion times, perfectly interleaved. In the worst case, *every* increment by core 0 is followed by an increment by core 1 before core 0 goes again. Then *every one* of the 2 billion total increments is a write to a line the other core just invalidated — a full read-for-ownership bounce. If a bounce costs ~60 ns and an uncontended increment costs ~1 ns, the contended version runs roughly 60x slower than it should. In practice the interleaving is not perfectly adversarial — sometimes a core squeezes in a few writes before the other steals the line — so the real slowdown is "only" 3x to 10x. But that is the gap between the speedup you expected and the slowdown you got. Nothing in your *source* changed; the *line* is the contended resource, and you never declared it.

## Measuring a cache-line bounce honestly

You should never take a cycle count on faith, including mine. Here is how to *measure* the bounce so the numbers are yours, on your CPU, with its line size and its interconnect.

The honest-measurement checklist for anything in this post:

- **Pin threads to specific cores** (`taskset`/`sched_setaffinity` on Linux, `thread_affinity` APIs elsewhere). If the OS scheduler migrates a thread mid-run, the line follows it and your numbers turn to noise. We are measuring *cross-core* traffic, so the cores must be fixed and distinct.
- **Warm up.** The first iterations pay one-time DRAM misses and page faults. Run a few hundred million iterations of warm-up, then time the steady state.
- **Run many times** and report the distribution (median and spread), not one number. Coherence latency varies with what else is on the bus.
- **Know your line size** (`getconf LEVEL1_DCACHE_LINESIZE`) and your topology (`lscpu`, `lstopo`). Two cores on the same physical core via SMT/hyperthreading share L1/L2 and will *not* show a bounce — you need two distinct physical cores, ideally pinned far apart, and the cross-socket case is a separate, larger number.
- **Disable turbo / fix the frequency** if you want stable absolute numbers, or report relative speedups (padded vs unpadded), which are robust to frequency wobble.

Here is a minimal C++ microbenchmark that contrasts two threads writing the *same* line versus *separate* lines. The only difference between the two runs is whether the two `long`s are adjacent (same line) or 64 bytes apart (different lines).

```cpp
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdint>
#include <cstdio>

// Two counters. Layout decides whether they share a line.
struct Shared  { volatile int64_t a; volatile int64_t b; };        // a,b adjacent: same 64B line
struct Padded  { volatile int64_t a; char pad[56]; volatile int64_t b; }; // a,b 64B apart: separate lines

template <class T>
double run(T* s, long iters) {
    auto worker = [iters](volatile int64_t* slot) {
        for (long i = 0; i < iters; ++i) (*slot)++;   // load-modify-store on one slot
    };
    auto t0 = std::chrono::steady_clock::now();
    std::thread t1(worker, &s->a);
    std::thread t2(worker, &s->b);
    t1.join(); t2.join();
    auto t1e = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1e - t0).count();
}

int main() {
    const long iters = 500'000'000;
    Shared sh{}; Padded pd{};
    run(&sh, 10'000'000);                       // warm up, ignore
    double tShared = run(&sh, iters);
    double tPadded = run(&pd, iters);
    printf("shared line:   %.3f s\n", tShared);
    printf("padded lines:  %.3f s\n", tPadded);
    printf("false-sharing slowdown: %.2fx\n", tShared / tPadded);
    return 0;
}
```

Pin it with `taskset -c 0,1 ./a.out` and run it a dozen times. On a typical recent x86 server (think a Xeon or EPYC core, two distinct physical cores), you will see the `shared line` run several times slower than `padded lines` — commonly 3x to 8x. (`volatile` here is a crude way to stop the compiler from hoisting the increment out of the loop into a register, which would erase the memory traffic we are trying to measure; in real code you would use `std::atomic` with relaxed ordering, or just real work that touches memory. Do *not* read `volatile` as "thread-safe" — see the [atomics post](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst).)

That single ratio — `tShared / tPadded` — *is* false sharing, measured. Now let us name it.

A note on the methodology, because microbenchmarking coherence is famously easy to get wrong. The single biggest mistake is letting the compiler delete the work: an optimizer that proves the loop's result is unused will simply remove the loop, and you will measure nothing (and conclude false sharing "doesn't matter," which is a worse error than measuring it badly). The `volatile` qualifier above is a blunt instrument to prevent that; in production-quality benchmarks you would use a framework that provides a `DoNotOptimize`/`benchmark::ClobberMemory` barrier (Google Benchmark) or `std::hint::black_box` (Rust) or the JMH `Blackhole` (Java) — these consume the result so the compiler must actually perform the writes. The second mistake is **thread placement**: if your two worker threads land on two *hyperthreads of the same physical core*, they share the same L1 and L2, so there is no cross-core bounce at all and the "false sharing" vanishes — you measured the wrong thing. Confirm with `lstopo` which logical CPUs are siblings, and pin to two distinct *physical* cores. The third is forgetting that the *first* touch of each line pays a cold-miss and possibly a page fault; warm up. Do all three and the ratio is trustworthy.

## False sharing: the mechanism

**False sharing** is when two cores contend on the *same cache line* despite touching *different variables* on that line. It is "false" because there is no logical sharing — thread A never reads or writes thread B's data — yet the cores fight over the line as if there were. The coherence protocol cannot tell the difference between "you wrote the variable I care about" (true sharing) and "you wrote a variable that merely *lives next to* the variable I care about" (false sharing). The unit of coherence is the line, not the variable. The protocol invalidates the whole line.

![Two unrelated variables on one 64 byte line cross invalidate each other while padding moves them to two lines so the writes stay independent](/imgs/blogs/cache-coherence-mesi-and-false-sharing-6.png)

Concretely, here is the full causal chain for the false-sharing case, which is the bounce from earlier but with the punchline that the two cores *never share data*:

1. `x` and `y` are two unrelated variables that the compiler/allocator happened to place 8 bytes apart — same 64-byte line `L`.
2. Core 0 only ever writes `x`. Core 1 only ever writes `y`. By the program's logic, these are independent; no lock is needed; no race exists.
3. Core 0 writes `x`: it needs `L` in M, so it invalidates core 1's copy of `L`. But core 1's copy of `L` *contained `y`* — the value core 1 actually cares about. Core 1 just lost `y` for no logical reason.
4. Core 1 writes `y`: `L` is now Invalid in core 1 (core 0 just stole it). Core 1 issues a read-for-ownership, fetches `L` back from core 0, invalidates core 0 — and now core 0 has lost `x`.
5. Forever. The line bounces on *every* write even though the two cores share *nothing*.

This is why the eight-counter array collapsed. Eight threads, eight counters, all on one (or two) lines. Every increment by any thread invalidated the line in all the other threads. The cores spent their lives shuttling one 64-byte line around the interconnect instead of incrementing in their private L1. Worse, false sharing **scales the wrong way**: with more threads on the same line, there are more cores to invalidate and more writers stealing the line, so the contention *grows* with core count. That is why adding cores made it slower — you added more participants to the ping-pong.

The insidious part is *invisibility*. False sharing leaves no trace in your source: no shared variable, no lock, no `volatile`, nothing a race detector flags (because, again, there is no data race — every variable has exactly one writer). It only shows up as a performance cliff and, if you go looking, as a high count of one specific hardware event: the cross-core cache-line transfer / HITM ("hit-modified": a load that hit a line dirty in *another* core's cache). On Linux you find it with `perf c2c` (cache-to-cache), which is purpose-built to attribute HITM events to the exact cache line and the exact pair of variables fighting over it. That tool is how you go from "it's mysteriously slow" to "these two fields are on the same line." We will return to it in the case studies.

### Finding false sharing in a real profile

Because false sharing is invisible in source, you find it by *measurement*, and the right tool is hardware performance counters that attribute coherence misses to cache lines. The workflow on Linux:

```bash
perf c2c record -F 60000 -a -- ./my_program   # record HITM (cache-to-cache, hit-in-modified) events
perf c2c report -NN --stdio                    # report hottest lines + per-offset CPU read/write + symbols
```

What you are looking for in the report is a single cache line with a high HITM count where the *offsets* within that 64-byte line are written by *different* CPUs — that is the signature of false sharing: two distinct variables, at two distinct offsets, on one line, bouncing between two cores. If instead the *same* offset is hammered by many CPUs, that is *true* sharing (one variable, real contention), and padding will not help. The tool literally shows you the byte offsets, so you can map them back to struct fields and decide which case you are in. A complementary signal: `perf stat -e cache-misses,LLC-load-misses,mem_load_l3_hit_retired.xsnp_hitm ./my_program` — a high `xsnp_hitm` (cross-snoop hit-modified) count relative to instructions retired is the fingerprint of a bouncing line. On platforms without `perf c2c`, the cruder but portable approach is the A/B microbenchmark: pad the suspect struct, re-run, and see if throughput jumps; a 2x+ swing from padding alone is false sharing by definition.

The reason this matters as a *workflow* and not just a tool tip: false sharing does not announce itself, and you can stare at correct, lock-free, race-free code for hours without seeing it. The discipline is to *suspect the line, not the variable* whenever a parallel, CPU-bound workload scales sub-linearly or backward, and to let the HITM counter point you at the offending 64 bytes.

#### Worked example: the allocator put them there

You write a thread-pool where each worker has a small `struct Worker { long tasks_done; State* state; };` and you allocate a contiguous `Worker pool[NUM_THREADS]`. `sizeof(Worker)` is 16 bytes. Four `Worker`s fit in one 64-byte line. So workers 0 through 3 share a line, 4 through 7 share the next, and so on. Each worker increments its own `tasks_done` in its hot loop — independent counters, you reasoned. But workers 0–3 cross-invalidate each other's line on every increment, and so do 4–7. You never wrote a shared variable. The *array layout* created the sharing. The fix is not in your logic; it is in the *bytes*. That is the next section.

## The fix: padding, alignment, and `@Contended`

The cure for false sharing is to guarantee that each independently-written variable sits on its **own** cache line, so no two cores ever fight over a line. You do this by **padding** (inserting filler bytes so the next hot variable is pushed past the 64-byte boundary) and/or **aligning** (forcing a variable to start at a line boundary). The cost is memory — up to ~64 bytes of waste per padded variable, sometimes 128 to also dodge hardware prefetchers that pull in *pairs* of lines. The benefit is killing the bounce. For hot, write-heavy, per-thread data, this trade is overwhelmingly worth it.

![A matrix of four false sharing fixes alignas padding the Contended annotation per thread accumulation and separate allocation with how cost and when columns](/imgs/blogs/cache-coherence-mesi-and-false-sharing-7.png)

Here is the same fix expressed idiomatically across four languages, because the *concept* is universal but the *spelling* diverges sharply.

**C++ — `alignas(64)` and the standard interference constants.** The cleanest modern way is to align each per-thread struct to a cache line. C++17 added `std::hardware_destructive_interference_size` precisely for this — the minimum offset to *avoid* false sharing (and `..._constructive_...` for the offset to *encourage* sharing, for data you *want* together).

```cpp
#include <new>      // std::hardware_destructive_interference_size
#include <atomic>
#include <cstdint>

constexpr std::size_t CL = std::hardware_destructive_interference_size; // usually 64 (or 128)

// BUG: all counters share lines — false sharing across threads.
struct CountersBad { std::atomic<int64_t> c[8]; };  // 8 * 8 = 64 bytes = ONE line

// FIX: each counter aligned to its own line. alignas pads the struct to CL.
struct alignas(CL) PaddedCounter {
    std::atomic<int64_t> value{0};
    // alignas(CL) makes sizeof(PaddedCounter) a multiple of CL, so an
    // array of these places each `value` on its own line.
};
struct CountersGood { PaddedCounter c[8]; };  // 8 * 64 = 512 bytes, one line each
```

Now `&good.c[i].value` and `&good.c[j].value` are at least 64 bytes apart for `i != j`, so two cores writing different counters never invalidate each other. `sizeof(CountersGood)` jumped from 64 bytes to 512 — that is the memory cost, and it is fine for eight counters.

**Java — `@Contended` and manual padding.** The JVM moves and packs fields, so you cannot rely on source order to control layout. JDK 8+ provides `@jdk.internal.vm.annotation.Contended` (or the older `sun.misc.Contended`), which tells the JVM to pad a field (or a whole class) onto its own line. It is gated behind the flag `-XX:-RestrictContended` for application code (the JDK uses it internally without the flag).

```java
import jdk.internal.vm.annotation.Contended;

// BUG: two hot fields, written by two threads, likely on one line.
class CounterBad {
    volatile long a;   // thread 1 writes this
    volatile long b;   // thread 2 writes this — false sharing with a
}

// FIX: @Contended tells the JVM to isolate the field on its own cache line.
class CounterGood {
    @Contended volatile long a;   // padded onto its own line by the JVM
    @Contended volatile long b;
}
// Run with:  java -XX:-RestrictContended ...   (app-level @Contended needs this flag)
```

Before `@Contended` existed (and still, when you cannot use it), the manual idiom is to surround the hot field with long padding fields — six `long`s on each side is the classic "pad to a line." You will see exactly this in the JDK source and in high-performance libraries; we look at the LMAX Disruptor's version in the case studies.

**Go — explicit struct padding.** Go has no alignment attribute for this, so you pad the struct manually with a filler array sized to fill out the line. The idiom in the Go runtime and standard library is a `_ [N]byte` pad field.

```go
const cacheLine = 64

// BUG: a slice of small structs packs several per line.
type counterBad struct {
	n int64 // 8 bytes; 8 of these per 64B line -> false sharing
}

// FIX: pad each struct out to a full cache line.
type counterGood struct {
	n   int64
	_   [cacheLine - 8]byte // filler pushes the next element onto a new line
}

// A slice []counterGood now places each .n on its own line:
counters := make([]counterGood, numThreads)
// counters[i].n and counters[j].n are >= 64 bytes apart for i != j.
```

Go's runtime does exactly this internally — search the runtime source for `cpu.CacheLinePad`, a struct whose only job is to be a line-sized pad embedded in hot per-P (per-processor) structures.

**Rust — `#[repr(align(64))]` and `crossbeam::CachePadded`.** Rust lets you force alignment on a type with `#[repr(align(N))]`, and the ecosystem-standard `crossbeam_utils::CachePadded<T>` wraps any value to sit alone on a line (it even uses 128-byte alignment on some targets to dodge prefetcher pairing).

```rust
use std::sync::atomic::{AtomicI64, Ordering};

// FIX via repr(align): force each counter to start on a 64-byte boundary.
#[repr(align(64))]
struct PaddedCounter {
    value: AtomicI64,
}

// Or, idiomatically, use crossbeam's wrapper which handles the alignment for you:
// use crossbeam_utils::CachePadded;
// let counters: Vec<CachePadded<AtomicI64>> =
//     (0..n).map(|_| CachePadded::new(AtomicI64::new(0))).collect();

let counters: Vec<PaddedCounter> =
    (0..num_threads).map(|_| PaddedCounter { value: AtomicI64::new(0) }).collect();
// Each counters[i].value is on its own line; no cross-invalidation.

fn bump(c: &PaddedCounter) {
    c.value.fetch_add(1, Ordering::Relaxed); // relaxed is fine: independent counters
}
```

Across all four: the concept is *one hot variable per line*. C++ spells it `alignas`/`hardware_destructive_interference_size`, Java spells it `@Contended`, Go spells it a `_ [56]byte` pad, Rust spells it `#[repr(align(64))]`/`CachePadded`. Python, notably, mostly *cannot* express this — its objects are heap-boxed and the GIL serializes bytecode-level access anyway, so per-line layout is not something you control; if you are fighting false sharing in Python you are almost certainly in a native extension, and the story belongs to the [Python performance series on the GIL](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) and multiprocessing, not here.

### The measured before → after

The whole point is the number. Here is the shape of results from the microbenchmark above (two threads, 500M increments each, two distinct physical cores, pinned; representative ratios — *measure your own*):

| Layout | Two `long`s, where? | Median time | Relative |
| --- | --- | --- | --- |
| Same line (false sharing) | adjacent, one 64B line | ~3.1 s | 1.0x (baseline) |
| Padded to separate lines | 64 bytes apart | ~0.6 s | **~5x faster** |
| Single thread, one counter | n/a, no contention | ~0.5 s | (reference) |

The padded two-thread version nearly matches the single-thread time *per counter* — exactly what you want, because the two counters are now genuinely independent. The unpadded version is ~5x slower despite doing identical arithmetic. The only difference is 56 bytes of padding. (The exact multiplier depends heavily on your CPU and how adversarial the interleaving is — I have seen anywhere from 2x to 8x on x86 servers, and *larger* across NUMA sockets. The direction is never in doubt; the magnitude is yours to measure.)

## True sharing and the cost of a contended atomic

Padding fixes *false* sharing. It does nothing for *true* sharing — when the contention is real because multiple cores genuinely read and write the *same* variable. A global counter that every thread increments, a shared work-queue head, a single atomic flag everyone polls: these are *true* sharing, and the line bounces for a real reason, not an accidental layout one.

![A two row matrix contrasting true sharing and false sharing across their cause symptom and fix columns](/imgs/blogs/cache-coherence-mesi-and-false-sharing-8.png)

Here is the trap people fall into: they replace a `Mutex<int>` counter with an `AtomicInt` and `fetch_add`, expecting it to "scale better because it's lock-free." Under contention it often does *not* scale, because the cost was never the lock instruction — it was the **cache line the counter lives on**. An atomic increment is, at the hardware level, a locked read-modify-write: on x86 a `lock xadd` (or a CAS loop), which requires the core to own the line exclusively (in M) for the duration. If 16 cores all `fetch_add` the same counter, all 16 fight for exclusive ownership of that one line. The line bounces among all 16, each acquiring M, incrementing, losing it to the next. Throughput collapses — it can be *worse* than a mutex, because at least a mutex lets one thread make a burst of progress while others sleep, whereas atomic contention has everyone spinning the bus.

To see *why* an atomic on a contended line is so expensive, walk the hardware. An `AtomicLong.fetch_add` compiles, on x86, to a single `lock xadd` instruction (exchange-and-add with the `lock` prefix). The `lock` prefix does not, on modern CPUs, lock the whole memory bus — that would be catastrophic — instead it makes the operation atomic by *holding the cache line in M for the duration of the read-modify-write*, refusing snoops until the operation commits. So to execute `lock xadd` on a line, the core must (1) own the line exclusively in M, which means (2) invalidating every other core's copy first, then (3) performing the indivisible read-add-write while no other core can steal the line. If 16 cores all want to `lock xadd` the same line, they form a queue: only one can hold M at a time, each must invalidate the previous owner, and the line marches around all 16 caches once per *batch* of operations — except there is no batching, so it marches around once per *operation*. The throughput ceiling is therefore roughly "how fast can one cache line visit 16 cores," which is set by interconnect latency, not by how fast a core can add. That is why throughput is *flat or declining* in core count for a single contended atomic: adding cores adds contenders for the one line, not adders. On ARM, the same logic holds with LL/SC (load-linked / store-conditional) or the newer `LDADD` atomics — the store-conditional fails and retries if another core touched the line, which under contention is a retry storm with the same root cause. Atomics are the right tool for *low-contention* coordination; on a *hot* line they are a bounce with extra steps. See the [atomics post](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) and [how a lock is built](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) for the instruction-level story.

The mechanism is the same MESI bounce, but now there is no padding fix, because the sharing is *intended*. The fixes are algorithmic:

- **Shard the counter.** Give each thread (or each core) its own padded counter and sum them when you need the total. This converts true sharing into *no* sharing for the hot path, paying a merge cost only on read. This is exactly what Java's `LongAdder` does (case study below) and what you should reach for instead of a single contended `AtomicLong` under heavy write load.
- **Batch.** Accumulate locally, flush to the shared counter occasionally. One contended write per 1000 local increments cuts the bounce rate 1000x.
- **Back off / combine.** Combining trees and flat-combining let one thread apply many threads' updates in a batch, trading latency for throughput.

The comparison that matters:

| | True sharing | False sharing |
| --- | --- | --- |
| **Cause** | Many cores read/write the *same* variable | Two variables coincidentally on one 64B line |
| **Logical sharing?** | Yes — the data is genuinely shared | No — each variable has one writer |
| **Symptom** | Line bounces; atomic/lock contention; throughput collapses with cores | Line bounces; "no shared state" yet it scales backward |
| **Detector flags it?** | Yes (it's real contention; profilers show the hot line) | No race; only `perf c2c`/HITM counts reveal it |
| **Fix** | Algorithm: shard, batch, combine, or rethink the data structure | Layout: pad/align each variable onto its own line |

The decision tree is short: if two cores touch the *same* variable, you have true sharing and need an algorithmic fix; if they touch *different* variables that share a line, you have false sharing and need padding. Both look identical in a flat profile ("this line is hot"); `perf c2c` distinguishes them by showing you the *offsets* within the line that the different cores hit.

#### Worked example: the atomic that didn't scale

A team replaced a mutex-guarded request counter (`mutex.lock(); count++; mutex.unlock();`) with `std::atomic<long> count; count.fetch_add(1, relaxed);`, expecting better scaling at 32 threads. Throughput *dropped*. The mutex version, under heavy contention, had each thread grab the lock, do a burst, release — the line holding `count` moved in chunks. The atomic version had all 32 cores issuing `lock xadd` against the single line as fast as they could, so the line bounced on *every* increment across all 32 cores: maximal coherence traffic, minimal forward progress. The fix was not "go back to the mutex" — it was a sharded counter (one padded `atomic<long>` per thread, summed on read), which made the hot-path write hit a *private* line that never bounced. Throughput went from worse-than-mutex to near-linear. The lesson: *atomic does not mean cheap; an atomic on a contended line is a bounce.*

## NUMA: when the line lives on another socket

So far every core has been equidistant from memory. On a multi-socket server (and on some large single-socket chips with multiple memory controllers), that is false. Memory is **Non-Uniform**: each socket has its own DRAM and its own portion of last-level cache, and accessing *your* socket's memory is fast while accessing the *other* socket's memory crosses the inter-socket link (Intel UPI, AMD Infinity Fabric) and costs noticeably more — often 1.5x to 2x the latency, with lower bandwidth. This is **NUMA**, Non-Uniform Memory Access.

NUMA makes every cost in this post worse and more variable:

- A **cache-line bounce across sockets** is far more expensive than within a socket — the cross-socket interconnect latency dwarfs the on-die one. False sharing that is merely painful within a socket becomes brutal across sockets.
- **Where memory is allocated matters.** Linux defaults to a *first-touch* policy: a physical page is placed on the NUMA node of the core that first *writes* to it. So if thread 0 (on socket 0) initializes a big array and then threads on socket 1 work on it, every access from socket 1 is a remote access. The fix is to have each thread initialize the memory it will use (so first-touch places it locally), or to use explicit NUMA allocation (`numactl`, `libnuma`'s `numa_alloc_onnode`, `mbind`).
- **A contended atomic or lock crossing sockets** pays the remote penalty on every bounce. A global lock that is fine on one socket can fall off a cliff when the box is two sockets, purely from the coherence traffic crossing the link.

The practical NUMA rules: pin threads to cores *and* their data to the same node; prefer per-socket (or per-core) sharded data structures so the hot path stays node-local; let each thread first-touch its own memory; and when you measure, *report the topology* — a number from a single-socket laptop does not predict a dual-socket server, and a thread migrated to the wrong socket mid-benchmark will produce garbage. The GPU and multi-node analog of "data locality dominates" shows up again in the [HPC interconnect and collective-communication work](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm); the same instinct — keep the data near the compute — governs both.

#### Worked example: the first-touch trap

A team parallelized a numerical kernel over a 4 GB array on a dual-socket box (two NUMA nodes). The driver thread, running on socket 0, allocated the array with one big `malloc` and zero-initialized it in a simple `memset` loop — all on socket 0. Then they spawned 32 worker threads, 16 pinned to each socket, and split the array in half. The socket-0 workers were fast; the socket-1 workers were ~1.8x slower for *identical* work. Why? First-touch placed *every* physical page on socket 0 (because socket 0's `memset` touched them first), so the 16 socket-1 workers made *every* access across the inter-socket link — remote DRAM on every cache miss. The fix changed nothing about the algorithm: they parallelized the *initialization* too, so each worker first-touched the slice of the array it would later process. Now first-touch placed each half on the socket that would use it, both halves of the array went node-local, and the asymmetry disappeared — the kernel sped up by close to the full remote/local ratio on the previously-remote half. The bytes never moved; *where they were born* changed. NUMA placement is a layout problem, just like false sharing — a different axis of the same lesson that the hardware, not your source, decides where data physically lives.

## Measured scaling: padded versus unpadded counters

Let us put the whole thing together with the experiment from the intro: N threads, each incrementing its own counter in a shared array, padded versus unpadded. This is the canonical false-sharing demonstration and the one to keep in your pocket.

The setup, measured honestly (pin each thread to its own physical core, warm up, 500M increments per thread, median of many runs, named CPU class, line size confirmed at 64B). Representative *shape* of results on a recent multi-core x86 server — the trend is robust; your absolute numbers will differ:

| Threads | Unpadded (one array, 8B slots) | Padded (each on own 64B line) | Padded speedup vs unpadded |
| --- | --- | --- | --- |
| 1 | 0.50 s | 0.50 s | 1.0x (no contention either way) |
| 2 | 1.7 s | 0.50 s | ~3.4x |
| 4 | 3.6 s | 0.51 s | ~7x |
| 8 | 6.5 s | 0.53 s | ~12x |

Read the two columns. The **padded** column is flat: each thread works on its own line, so 8 threads finish in about the same wall time as 1 thread (each is independent, near-perfect parallelism — total work scales but per-thread time does not). The **unpadded** column gets *worse* as you add threads: more cores on the shared line means more invalidations per write, so wall time *grows*. The speedup of padding over no-padding therefore widens with core count — from ~3x at two threads toward an order of magnitude at eight. That divergence *is* false sharing, and it is why the eight-thread unpadded version in the intro was slower than single-threaded: at high core counts the unpadded line spends essentially all its time in flight on the interconnect.

Two honesty caveats. First, "padded is flat" assumes the threads truly do independent work; if they also touch genuinely shared data, *that* sharing reappears and padding does not save you. Second, the exact unpadded numbers depend on the interleaving the scheduler happens to produce and on whether the eight counters fit on one line or two — pack them differently and the curve shifts. The qualitative result — *padding turns a backward-scaling loop into a flat, near-linear one* — is what you can rely on.

Here is the N-thread version of the benchmark, the one that produced the table's shape, with per-thread core pinning so the measurement is clean. This is the code to keep — run it on your own box and read the divergence of the two columns:

```cpp
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <pthread.h>   // for pinning on Linux

constexpr std::size_t CL = 64;

struct alignas(CL) Padded   { std::atomic<int64_t> v{0}; };  // each on its own line
struct           Unpadded { std::atomic<int64_t> v{0}; };    // packed: 8 per line

void pin(std::thread& t, int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
}

template <class Slot>
double run(int n, long iters) {
    std::vector<Slot> slots(n);                 // n counters, layout per Slot type
    auto worker = [&](int i) {
        for (long k = 0; k < iters; ++k)
            slots[i].v.fetch_add(1, std::memory_order_relaxed); // own slot only
    };
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < n; ++i) { ts.emplace_back(worker, i); pin(ts.back(), i); }
    for (auto& t : ts) t.join();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    const long iters = 500'000'000;
    for (int n : {1, 2, 4, 8}) {
        double u = run<Unpadded>(n, iters);
        double p = run<Padded>(n, iters);
        printf("threads=%d  unpadded=%.2fs  padded=%.2fs  speedup=%.1fx\n",
               n, u, p, u / p);
    }
}
```

Note the only difference between the two runs is the `alignas(CL)` on the slot type — identical arithmetic, identical thread count, identical pinning. The `Unpadded` slots are 8 bytes each, so eight of them land on one or two lines and all the threads cross-invalidate; the `Padded` slots are 64 bytes each, so every counter owns a line and the threads never collide. Run it a dozen times and report medians; the unpadded column climbing while the padded column stays flat *is* the phenomenon, reproduced from first principles. If you want this concrete in another language, the same scaffold transcribes directly to the Java, Go, and Rust padding idioms shown earlier — the layout trick is the same, only the spelling of "put it on its own line" changes.

## Case studies / real-world

False sharing is not a textbook curiosity — it has driven the design of widely-used production code. Three concrete, documented cases.

**Java's `LongAdder` and `@Contended` (JDK).** When many threads increment one counter, a single `AtomicLong` becomes a true-sharing bottleneck (one contended line, as we saw). Doug Lea's `java.util.concurrent.atomic.LongAdder` (JDK 8+) solves this by keeping an *array of cells*, each thread hashing to a cell and incrementing it, with the total computed by summing cells on `sum()`. Crucially, the `Cell` class is annotated `@Contended` so each cell sits on its own cache line — without that annotation, the cells would false-share and the sharding would be pointless. This is the canonical "shard the counter, then pad the shards" pattern, shipped in the standard library. `ConcurrentHashMap`'s internal counter cells use the same `@Contended` striping. You can read the `Striped64` superclass source in the JDK to see both the sharding and the padding spelled out. (Reference: the OpenJDK `Striped64`/`LongAdder` source and the `@Contended` JEP discussion.)

**The LMAX Disruptor's padded sequences.** The LMAX Disruptor is a high-throughput inter-thread messaging ring buffer (famous for processing millions of messages per second on a single thread). Its `Sequence` counters — the producer cursor and the consumer cursors — are the hottest, most-contended longs in the system, and several threads watch them. The Disruptor's authors found that without padding, these cursors false-shared with each other and with adjacent fields, capping throughput. Their fix, documented in Martin Thompson's "Mechanical Sympathy" writing and in the Disruptor source, was to pad each sequence with extra `long` fields on both sides so each cursor occupies a line alone (the classic `p1..p7` padding-field idiom, later replaced by `@Contended` on newer JVMs). The padding was load-bearing for their headline throughput number. (Reference: the LMAX Disruptor technical paper and Martin Thompson's blog on false sharing and mechanical sympathy.)

**The Linux kernel's per-CPU and `____cacheline_aligned` data.** The kernel is saturated with cache-line alignment to prevent false sharing on hot structures. The `____cacheline_aligned` and `____cacheline_aligned_in_smp` macros force a structure (or field) onto its own line; per-CPU variables exist precisely so each CPU writes its *own* line and never bounces. Hot fields in the scheduler, the memory allocator, and the networking stack are deliberately laid out — sometimes with explicit padding between a read-mostly group and a write-heavy group — so that frequently-written fields do not share a line with frequently-read ones. The kernel community has repeatedly found and fixed false-sharing regressions where an innocent struct-field reordering put two hot fields on one line and tanked throughput on big NUMA machines; `perf c2c` was built largely for exactly this hunt. (Reference: `include/linux/cache.h` in the Linux source and the `perf c2c` documentation.)

**A concurrent data structure that regressed on a field reorder.** A recurring class of incident — seen in the JDK, the Go runtime, and countless application codebases — is a "harmless" refactor that reorders struct fields and accidentally moves a hot, frequently-written field next to another hot field, putting two writers on one line where they were previously apart. The change passes every test (it is correct), passes code review (it looks like cosmetic reordering), and then throughput drops 20 to 40% on the multi-core production box while staying fine on the reviewer's laptop (fewer cores, less contention, smaller bounce). The fix is to restore the separation — group read-mostly fields together, isolate write-hot fields onto their own lines, and in performance-critical structs, *comment the layout* so the next person does not undo it. The general defense is a microbenchmark in CI that would catch the regression, plus a `perf c2c` pass on the hot path before shipping a layout change. The lesson is uncomfortable: in concurrent, CPU-bound code, *field order is a performance contract*, and reordering it can cost you cores' worth of throughput with zero change in behavior.

The through-line: in every one of these, smart people who *knew* about cache lines still got bitten, found it with HITM/`c2c` measurement, and fixed it with alignment and sharding. False sharing is not a beginner-only mistake; it is a layout property that is invisible until you measure the line. The recurring pattern across all four — `LongAdder`, the Disruptor, the kernel, the field reorder — is identical: a hot, write-heavy variable shared a line with something else, the line bounced, and the fix was to give it a line of its own. Once you internalize "the line is the unit," you start seeing the pattern everywhere, and you start laying out hot data deliberately instead of by accident.

## When to reach for this (and when not to)

Padding is not free and not always warranted. Every padded variable costs up to 64 (or 128) bytes of memory and can hurt *spatial* locality — if you pad apart things that are read together, you turn one cache miss into several. So be decisive about when it pays.

**Reach for padding/alignment when *all* of these hold:**

- The data is **per-thread (or per-core) and written frequently** — counters, accumulators, ring-buffer cursors, per-thread statistics. Writes are what trigger invalidation; read-mostly data rarely false-shares painfully.
- The variables are **physically close** — in the same struct, same array, or allocated together — so they plausibly land on one line. Check with `perf c2c` or by reasoning about `sizeof` and array stride.
- You have **measured a scaling problem** — throughput flat or declining with cores, a hot line in `perf c2c`/HITM, or the padded-vs-unpadded microbenchmark showing a real swing on *your* data.
- The work is genuinely **parallel and CPU-bound** — false sharing only bites when multiple cores are actually hammering the line concurrently. (See [concurrency vs parallelism](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws) for why an IO-bound workload won't show this.)

**Do *not* reach for padding when:**

- The data is **read-mostly or rarely written** — a config value read by all threads sits happily in S in every cache, no bounce. Padding it just wastes memory and hurts locality.
- The contention is **true sharing** — padding does nothing; you need an algorithmic fix (shard, batch, combine), and *then* pad the shards.
- You **haven't measured** — do not sprinkle `@Contended`/`alignas(64)` preemptively across every struct. It bloats memory, evicts more useful lines, and most fields never false-share. Measure first; pad the lines that actually bounce.
- You are **memory-constrained** — on a structure you allocate millions of, 64 bytes of pad each is gigabytes of waste. Pad only the genuinely-hot few.
- The workload is **single-threaded or IO-bound** — no concurrent writers, no bounce, no benefit.

The meta-rule, consistent with the rest of this series and the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model): name what is shared (here, the *line*, not the variable), establish what order/ownership the hardware must enforce, choose the cheapest mechanism — and *prove it with a measurement*. Padding is a tool for a measured problem, not a reflex.

## Key takeaways

- **The cache line (~64 bytes) is the unit of coherence, not the variable.** Two variables in the same 64 bytes are one object to the hardware; one core's write invalidates the *whole line* in every other core.
- **MESI enforces "one writer or many readers" per line.** A line is Modified or Exclusive in at most one core; Shared in many; Invalid where there is no valid copy. Writing requires exclusive ownership, so a write to a Shared/Invalid line invalidates all other copies.
- **A write makes a contended line bounce.** Each alternating write drags the line into the writer's cache as Modified and invalidates the peer, costing tens to hundreds of cycles per bounce versus ~4 for an uncontended L1 write — an order-of-magnitude tax paid on every write.
- **False sharing is contention with no logical sharing.** Two cores writing *different* variables that share a line still cross-invalidate; there is no data race, no detector flags it, and scaling silently runs backward.
- **The fix is layout: one hot variable per line.** `alignas(64)`/`hardware_destructive_interference_size` in C++, `@Contended` in Java, a `_ [56]byte` pad in Go, `#[repr(align(64))]`/`CachePadded` in Rust. Measured before→after is commonly a 2x to 8x swing.
- **True sharing needs an algorithm, not padding.** A contended atomic is a bounce, not a free lunch; shard (`LongAdder`), batch, or combine to make the hot path hit a private line.
- **NUMA multiplies the cost.** Cross-socket bounces and remote first-touch memory are far more expensive; pin threads and data to the same node and shard per socket.
- **Measure honestly and never preemptively pad.** Pin cores, warm up, run many times, confirm your line size, find the bouncing line with `perf c2c`/HITM — and pad only the hot, write-heavy, per-thread lines that a measurement proves are bouncing.

## Further reading

- **Ulrich Drepper, *What Every Programmer Should Know About Memory*** — the definitive long-form treatment of caches, lines, coherence, and NUMA; section 6 covers false sharing and alignment directly.
- **Maurice Herlihy & Nir Shavit, *The Art of Multiprocessor Programming*** — the coherence and contention chapters, plus the algorithmic side (combining trees, counting networks) for true sharing.
- **Anthony Williams, *C++ Concurrency in Action*** — practical false sharing, `std::hardware_destructive_interference_size`, and atomics under contention.
- **Brian Goetz et al., *Java Concurrency in Practice*** and the OpenJDK `Striped64`/`LongAdder` source — the sharded-counter-plus-`@Contended` pattern in production.
- **The LMAX Disruptor technical paper** and **Martin Thompson's "Mechanical Sympathy" blog** — false sharing, padding, and why mechanical sympathy matters for throughput.
- **The `perf c2c` documentation** and the **Linux kernel `include/linux/cache.h`** — how to *find* a bouncing line by HITM event, and how a large codebase aligns hot data.
- **Within this series:** [why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), the [memory models post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before), the [atomics and memory orderings post](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), and the [concurrency playbook capstone](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- **For the GPU/accelerator analog of the memory hierarchy and data locality:** the [HPC memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).

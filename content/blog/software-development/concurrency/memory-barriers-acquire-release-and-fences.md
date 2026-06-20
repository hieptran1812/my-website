---
title: "Memory Barriers: Acquire, Release, and Fences"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The tools that restore order to reordered code: the four barrier types, one-way acquire and release fences, the publish and subscribe pattern, full fences, and the volatile trap."
tags:
  [
    "concurrency",
    "parallelism",
    "memory-barrier",
    "acquire-release",
    "fence",
    "memory-model",
    "volatile",
    "atomics",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/memory-barriers-acquire-release-and-fences-1.png"
---

A config service ships a new release. Once a minute, a background thread builds a fresh `Config` object — reads the feature flags, parses the rate limits, fills in a dozen fields — and then publishes it by storing a pointer into a shared variable that every request thread reads. The build is single-threaded and obviously correct. The publish is a single pointer write, which is atomic on every CPU you care about. The reads are plain loads of a pointer. Nothing is `null`, nothing is partially written in the source code. And yet, a few times a day, on the ARM-based fleet only, a request thread crashes dereferencing a `Config` whose `rateLimit` field is still zero — a value it never held after construction finished. The x86 fleet never sees it.

This is not a torn pointer, not a missing `null` check, not a use-after-free. It is a publication bug, and it is one of the most common and most baffling failures in all of concurrent programming. The writer wrote the fields, *then* wrote the pointer. But "then" is a lie the source code tells. The compiler and the CPU are both free to make the pointer store globally visible *before* the field stores reach memory, because to a single thread the order of two independent writes is invisible. Another thread reads the published pointer, follows it, and finds an object that — from its point of view — is half-built. The fix is not a lock and not a bigger struct. It is a single *release-store* on the writer and a matching *acquire-load* on the reader: a one-way memory barrier that forces the fields to be visible before the pointer, and forces the reader who sees the pointer to also see the fields. The figure below shows the broken view and the fixed one.

![Unsafe publication where a reader sees the pointer but stale fields versus a release store and acquire load that reveal a fully built object](/imgs/blogs/memory-barriers-acquire-release-and-fences-1.png)

By the end of this post you will be able to name the four hardware reorderings, place an acquire or a release fence exactly where it forms a [happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) edge, recognize the publish and subscribe pattern that underpins every safe-publication and lock-free algorithm, explain why a full `seq_cst` fence is the expensive one, map all of this onto `mfence`, `dmb ish`, and `lwsync`, and — critically — never again confuse Java's `volatile` (a real barrier) with C and C++'s `volatile` (not a barrier at all). This is the post in the series where the abstract memory model from [why your code does not run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering) becomes a set of concrete instructions you can place by hand. If you have not read the [memory model post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before), skim it first — barriers are how you *buy* the happens-before edges that post defines.

## Why reordering needs a tool to fix it

Recall the core hazard of this whole series: shared mutable state plus nondeterministic scheduling is the danger, and the cure is to establish an order — a happens-before relation — over every conflicting access. The previous two posts established that the order you *wrote* is not the order that *runs*. Compilers reorder independent operations to keep the pipeline full and registers busy. CPUs reorder too: a store sits in a per-core **store buffer** and becomes visible to other cores later than the program-order successor loads on the same core. On a weakly-ordered machine like ARM or POWER, loads and stores to independent addresses can be observed in almost any order by another core.

For a single thread this is invisible and harmless — the hardware and compiler both preserve the illusion of sequential execution *as seen by that one thread*. The trouble is purely cross-thread: thread B can observe thread A's memory operations in an order that thread A's own source code never permitted. A memory barrier (also called a memory fence) is the instruction that revokes a specific reordering. It is the *only* tool the hardware gives you to say "this operation must be visible before that one, to every other core." Locks, atomics, `volatile` flags in Java, channels with proper synchronization — every one of them is built on barriers underneath. Understanding the barrier is understanding the floor that everything else stands on.

There is a subtle point worth stating plainly before we go deeper: a barrier orders *memory operations*, not *time*. It does not make a write happen "sooner" in wall-clock terms. It constrains the *order* in which writes become visible relative to other writes and reads. Two threads can still race to the same release-store; the barrier only guarantees that *if* you observe the store, you also observe everything ordered before it. Keep that framing — "ordering, not timing" — and the rest of this post follows.

To make the mechanism less hand-wavy, look at the store buffer in detail, because it is the single piece of hardware that motivates almost every barrier you will ever place. A modern core does not write straight to the L1 cache on every store. Doing so would couple the core's pipeline to the latency of cache-coherence arbitration — the protocol (MESI and its cousins) by which cores negotiate exclusive ownership of a cache line before writing it. Acquiring that ownership can take tens to hundreds of cycles when another core holds the line. If the pipeline stalled on every store waiting for ownership, throughput would collapse. So the core writes into a small per-core FIFO of pending stores — the store buffer — marks the store as retired, and moves on. The store drains into the coherent cache later, asynchronously, when ownership arrives. From the writing core's own perspective nothing is lost: a subsequent load to the *same* address is satisfied by *store-to-load forwarding* directly out of the buffer, so the core always reads its own latest write. But from *another* core's perspective, the store does not exist until it drains. That single asymmetry — "I see my own store immediately; you see it whenever it drains" — is the root cause of the StoreLoad reordering and the reason a full fence is the expensive one. Keep that store-buffer model in mind; we return to it three times.

The compiler is the *other* half of the reordering story, and it is easy to forget because it leaves no instruction behind. An optimizing compiler is a reordering engine by design: it hoists loads out of loops (loop-invariant code motion), sinks stores past unrelated work, keeps a value in a register instead of re-reading memory (so a flag another thread flips is never re-read), and merges or eliminates "redundant" accesses. Every one of those transformations is correct under the as-if rule for a *single* thread and catastrophic across threads. A spin loop `while (!ready) {}` on a plain non-atomic `ready` can be compiled to `if (!ready) { for(;;); }` — the compiler reads `ready` once into a register, sees nothing in the loop changes it, and spins forever even after another thread sets it. That is not a CPU reordering; it is a *compiler* reordering, and no hardware fence fixes it. This is why every real barrier has two halves: a compiler barrier that constrains the optimizer, and (where the ISA needs it) a CPU fence that constrains the hardware. A correct synchronization tool always supplies both; `volatile` in C and C++ supplies only the first, which is exactly the trap we dissect later.

## The four barrier types

Every reordering the hardware can do is one of exactly four kinds, named by the pair of operations whose order is at risk. A barrier is classified by which of these pairs it forbids from swapping.

- **LoadLoad** — forbids a later load from being reordered before an earlier load. If you read flag, then read data, a LoadLoad barrier between them guarantees the data read sees memory at least as fresh as the flag read implied.
- **LoadStore** — forbids a later store from being reordered before an earlier load. Rare to need alone; it shows up bundled into acquire and release.
- **StoreStore** — forbids a later store from being reordered before an earlier store. This is the one that fixes our config bug on the writer side: it forces the field writes to be visible before the pointer write.
- **StoreLoad** — forbids a later load from being reordered before an earlier store. This is the expensive one, and the reason is mechanical: it requires draining the store buffer so the store is globally visible before the subsequent load executes. The other three never require a buffer drain.

It is worth being painfully precise about what "forbids a reordering" buys you, because the name of a barrier describes the *pair* it constrains, and the constraint is always relative to *some other core's observation*. Take **LoadLoad** concretely. You read `flag` (load 1) and then read `data` (load 2). Without a LoadLoad barrier between them, the core is free to issue load 2 *before* load 1 retires — modern cores execute loads out of order and speculatively, so `data` may be fetched from cache before `flag` is. If another core is concurrently writing `data` then setting `flag`, your core can observe the new `flag` but the *old* `data`, because your `data` load happened earlier in real time even though it appears later in your source. The LoadLoad barrier pins load 2 to execute no earlier than load 1, so once you have seen the fresh `flag`, the subsequent `data` load is guaranteed to see memory at least as new as the moment the flag read observed. That is the reader half of every publish/subscribe handshake.

**StoreStore** is the dual on the writer. You write `data` (store 1) then write `flag` (store 2). Without a StoreStore barrier, the two stores can drain from the store buffer to coherent memory in either order — they target independent addresses, so the core sees no dependency. Another core can therefore observe the new `flag` while `data` is still the old value. The StoreStore barrier forces store 1 to drain (become globally visible) before store 2 does. Crucially, this is *exactly* the bug in our config publication: the field write is store 1, the pointer write is store 2, and the missing StoreStore barrier is what let the pointer overtake the fields.

**LoadStore** is the quietest of the four. It forbids a *later store* from being hoisted above an *earlier load*. You rarely reach for it alone, but it is the glue inside both acquire and release: an acquire-load must not let stores below it climb above it, and a release-store must not let loads-turned-stores from the critical section leak past it. It is the barrier that keeps a critical section's *contents* from escaping its boundaries. The remaining pair, **StoreLoad**, is the one the other three never cover, and the rest of this post keeps circling back to why.

![A matrix of the four barrier types LoadLoad LoadStore StoreStore and StoreLoad with the reordering each forbids what it is used for and its cost](/imgs/blogs/memory-barriers-acquire-release-and-fences-2.png)

The asymmetry of cost is the single most important fact about hardware barriers, so let me make the mechanism precise. On x86, the memory model is **Total Store Order** (TSO): the only reordering the hardware permits is StoreLoad. Loads are never reordered with earlier loads, stores never with earlier stores, and a load is never reordered before an earlier store *except* the StoreLoad case where a store is still in the buffer and a later load to a *different* address can complete first. That means on x86, LoadLoad, LoadStore, and StoreStore barriers are *free* — the hardware already enforces them; the compiler just has to not reorder across them, which costs nothing at run time. Only a StoreLoad barrier costs a real instruction (`mfence`, or a `lock`-prefixed op), because only StoreLoad is something x86 actually does.

On ARM and POWER the model is weak: all four reorderings are permitted, so all four barriers cost real fence instructions (though ARM has cheaper targeted forms like `dmb ishld` and the `ldar`/`stlr` load-acquire/store-release instruction pair). This is exactly why our config bug appeared *only* on the ARM fleet: x86 happens to forbid StoreStore for free, so a sloppy publication often works by accident on x86 and then explodes the day you deploy to ARM servers or Apple silicon.

#### Worked example: the config bug as a reordering

Walk the writer's two stores. In source order: `cfg.rateLimit = 5000` (store 1), then `published = cfg` (store 2). On ARM, the core is free to make store 2 globally visible before store 1, because they target independent addresses and to this single core the order is invisible. Now interleave a reader on another core:

```c
// interleaving illustration, NOT runnable source
// T1 (writer)               T2 (reader)
store 2: published = cfg;
                             load: p = published;    // sees cfg, non-null
                             load: r = p.rateLimit;  // reads 0 — store 1 not visible yet
store 1: cfg.rateLimit = 5000;
```

The reader followed a perfectly valid pointer and read a field that, in program order, was written *before* the pointer — but the StoreStore reordering let the pointer overtake the field. A StoreStore barrier between store 1 and store 2 (which is precisely what a release-store inserts) forbids this interleaving. That is the whole fix.

Two caveats so I am not lying to you by omission. First, that code block is an interleaving illustration, not runnable source — I labeled it as such; the runnable fix follows below. Second, on x86 this specific interleaving cannot happen because x86 forbids StoreStore for free, which is exactly why the x86 fleet never reproduced the crash. Same source code, different memory model, different outcome. Never trust that "it works on my x86 laptop" means the code is ordering-correct.

## Acquire and release: the one-way fences

A *full* barrier forbids reordering in both directions across a point. That is more than you usually need, and "more than you need" means "slower than you need." The two workhorse barriers of practical concurrency are **one-way fences**: they let operations move freely in one direction across the fence and forbid movement in the other. There are exactly two, and they are duals.

**Acquire** semantics attach to a load (or a lock acquisition). The rule: *no memory operation that appears after the acquire in program order may be reordered to before it.* Operations above the acquire can sink down past it; operations below the acquire cannot rise up past it. An acquire is a one-way roof. In barrier terms, an acquire-load is a LoadLoad barrier plus a LoadStore barrier — it pins everything *after* it from floating above. It does *not* stop earlier operations from sinking below, which is why it is cheaper than a full fence.

**Release** semantics attach to a store (or a lock release). The rule: *no memory operation that appears before the release in program order may be reordered to after it.* Operations below the release can rise up past it; operations above it cannot sink below it. A release is a one-way floor. In barrier terms, a release-store is a StoreStore barrier plus a LoadStore barrier — it pins everything *before* it from sinking past, so all your prior writes are committed before the release store becomes visible.

![Acquire as a one-way roof where nothing later sinks above it versus release as a one-way floor where nothing earlier rises below it](/imgs/blogs/memory-barriers-acquire-release-and-fences-3.png)

The directionality is the entire point, so let me state it as a mnemonic you can carry. **Acquire keeps later work down; release keeps earlier work up.** A reader *acquires* — it reads a flag and then everything that follows must see at least as much memory as the flag implied, so the later reads cannot be hoisted above the flag read. A writer *releases* — it does all its work and then sets a flag, and all that prior work must be committed before the flag becomes visible, so the earlier writes cannot be pushed below the flag write. Notice that an acquire alone does not order *stores* before it, and a release alone does not order *loads* after it. Neither one, by itself, gives you StoreLoad ordering. That gap is exactly why a full fence is a separate, more expensive thing — and why two threads each using only acquire/release can still observe a store-then-load surprise (the Dekker / Peterson litmus that needs `seq_cst`).

Why are they *one-way* and not full barriers? Because making them two-way would forbid reorderings that are not just harmless but actively desirable. Consider the acquire-load at the top of a critical section. It must stop the section's body from leaking *above* the lock — that direction is correctness-critical. But there is no reason to stop work that was already in flight *before* the lock from sinking down *into* the section as the pipeline catches up; that motion is invisible to other threads and lets the core keep its execution units busy. Forbidding it would be pure cost for zero safety. Symmetrically, the release-store at the bottom of a critical section must stop the body from leaking *below* the unlock, but there is no reason to stop later, post-unlock work from being started early and rising up *toward* the unlock. So each one-way fence forbids exactly the direction that would let a critical section's contents escape, and permits the direction that merely lets independent work overlap. That is not a compromise; it is the *minimal* constraint that preserves the publish/subscribe contract, and minimality is why acquire/release is cheap. A full fence is more expensive precisely because it forbids *both* directions, including the StoreLoad direction that neither one-way fence touches.

There is a second way to see the one-wayness that some engineers find clearer: think of the acquire and release as the two ends of a *region*, and the region as a one-way valve for visibility. Everything the writer did before its release flows *out* through the release and *in* through a matching acquire — but nothing flows the other way. A reader that acquires cannot push its later reads back in time before the acquire (they would miss the publication), and a writer that releases cannot pull its earlier writes forward in time after the release (they would be published before they exist). The valve only lets memory effects cross the boundary in the publish-to-subscribe direction. When you place an acquire with no matching release, or a release with no matching acquire, you have built half a valve: it leaks. That is why the iron rule is *always pair them*.

Here is acquire/release made concrete in C++ `std::atomic`, the most explicit memory-ordering API in wide use:

```cpp
#include <atomic>
#include <thread>

struct Config { int rate_limit; int burst; const char* region; };

Config* g_cfg = nullptr;            // the actual object storage (heap)
std::atomic<Config*> g_published{nullptr};  // the publication slot

void writer() {
    Config* c = new Config{5000, 200, "us-east"};  // fill fields (plain stores)
    // release-store: every store above is committed before this becomes visible
    g_published.store(c, std::memory_order_release);
}

int read_rate_limit() {
    // acquire-load: if we see a non-null pointer, we see all the writer's fields
    Config* c = g_published.load(std::memory_order_acquire);
    if (c == nullptr) return -1;        // not published yet
    return c->rate_limit;               // guaranteed to read 5000, never 0
}
```

The `release` on the store and the `acquire` on the load are the two halves of one barrier. Drop either to `memory_order_relaxed` and the guarantee evaporates: a relaxed store is still atomic (no torn pointer) but carries no StoreStore barrier, so the fields can lag; a relaxed load carries no LoadLoad barrier, so the reader's field reads can be hoisted above the pointer read. Atomicity and ordering are different properties, and `std::atomic` lets you pay for each independently — that is the whole point of [memory orderings from relaxed to seq_cst](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst).

#### Worked example: safe publication of a config object, step by step

Let me walk the safe version above one operation at a time on a weakly-ordered machine, because seeing exactly which barrier catches which reorder is what makes this stick. The writer thread runs three plain stores followed by one release-store:

```cpp
c->rate_limit = 5000;        // plain store P1
c->burst      = 200;         // plain store P2
c->region     = "us-east";   // plain store P3
g_published.store(c, std::memory_order_release);   // release-store R
```

Step 1. P1, P2, P3 land in the writer's store buffer in some order — the hardware does not promise they drain to coherent memory in program order, and an aggressive core might drain P3 first. Left alone, any of them could become globally visible *after* the pointer.

Step 2. The release-store R executes. Its StoreStore + LoadStore semantics impose a hard rule: every store sequenced before R — that is P1, P2, P3 — must become globally visible *before* R itself becomes visible. On x86 this costs nothing because TSO already drains stores in order; on ARM the compiler emits `stlr` (store-release) or a preceding `dmb`, which holds the pointer store until the three field stores have drained. Either way, the moment any other core can observe the published pointer, it is *guaranteed* that all three fields are already visible to that core.

Step 3. The reader runs the mirror image:

```cpp
Config* c = g_published.load(std::memory_order_acquire);   // acquire-load A
if (c == nullptr) return -1;
int rl = c->rate_limit;   // plain load L1
int b  = c->burst;        // plain load L2
```

The acquire-load A has LoadLoad + LoadStore semantics: no load sequenced after A — that is L1, L2 — may be reordered to execute before A. So the reader cannot speculatively prefetch `c->rate_limit` *before* it has actually observed the published pointer. Combined with step 2's guarantee, the logic closes: if A reads the non-null pointer that R published, then R was globally visible, so P1–P3 were globally visible, so L1 and L2 are guaranteed to read 5000 and 200, never the pre-construction zeros.

Step 4. The case that does *not* form an edge. If the reader's acquire-load A runs *before* the writer's release-store R drains, A reads `nullptr`, the `return -1` fires, and no field is read — exactly correct, because nothing was published to this reader yet. The synchronizes-with edge is conditional on A observing R's value; when it does not, there is simply no edge, and there is no race because there was no read of the protected data. The contract never over-promises: you inherit the writer's memory if and only if you saw the writer's flag.

## Publish and subscribe: the bedrock pattern

Acquire and release are duals for a reason: they are designed to be used *together*, as a matched pair, to form a synchronizes-with edge. The pattern has a name worth internalizing because you will use it for the rest of your career: **publish and subscribe**.

- A **release-store publishes** everything the writer did before it. "Here is a flag; if you see it set, I promise everything I wrote before setting it is visible to you."
- An **acquire-load subscribes** to all of it. "I read the flag; because I read it with acquire, I am guaranteed to see everything the publisher wrote before its release-store."

The contract is one-directional and conditional. The reader only inherits the writer's memory *if* the acquire-load actually observes the value the release-store wrote. If the reader's acquire-load reads the *old* value (publication has not happened yet), no edge forms and it gets no guarantee — which is correct, because nothing has been published to it. The edge exists between a *specific* release and the *specific* acquire that reads its value.

![A timeline where the writer fills two fields then release stores a ready flag and the reader acquire loads the flag then reads both fields with all writes visible](/imgs/blogs/memory-barriers-acquire-release-and-fences-4.png)

This is the timeline above: writes `a = 7`, `b = 9`, then `release-store ready = true`; the reader does `acquire-load ready`, sees `true`, then reads `a` and `b` and is guaranteed to see `7` and `9`. The release barrier holds the two field writes *above* the flag write; the acquire barrier holds the two field reads *below* the flag read; the matched value carries the synchronizes-with edge across the gap. Every safe-publication idiom, every lock (a lock release is a release-store on the lock word, a lock acquire is an acquire-load), every lock-free queue's "make the node visible" step, every double-checked-locking singleton done right — all of them are this one pattern. Learn it once.

Here is the *same* pattern in Java, where it is even terser because `volatile` *is* an acquire/release pair by definition of the Java Memory Model:

```java
public final class ConfigHolder {
    // plain fields, written before the volatile publish
    private int rateLimit;
    private int burst;

    // the publication flag: a volatile store is a release, a volatile load is an acquire
    private volatile boolean ready = false;

    // called by the single writer thread
    public void publish(int rl, int b) {
        this.rateLimit = rl;       // plain store
        this.burst = b;            // plain store
        this.ready = true;         // volatile store == release: publishes the two fields
    }

    // called by many reader threads
    public int readRateLimit() {
        if (!this.ready) return -1;     // volatile load == acquire: subscribes
        return this.rateLimit;          // guaranteed to see the writer's value
    }
}
```

In the Java Memory Model, a write to a `volatile` field *happens-before* every subsequent read of that same field. Because the two plain field writes are ordered before the `volatile` write (release semantics), and the two plain field reads are ordered after the `volatile` read (acquire semantics), the JMM gives you the full chain: the plain field writes happen-before the plain field reads, *transitively*, through the volatile flag. No `synchronized`, no lock, no `mfence` you can see — the JVM emits the right barrier for the target CPU. This is the canonical Java safe-publication idiom, and it is exactly the C++ release/acquire pair wearing different syntax.

A note on the `final` field special rule, since people reach for it here: Java *also* guarantees that `final` fields set in a constructor are visible to any thread that sees the object through a *properly published* reference, even without a volatile — that is the immutable-object publication guarantee. But that only covers `final` fields and only if the reference does not escape during construction. The general, mutable-friendly tool is the release/acquire publish above.

Modern Java (9+) exposes the *exact* C++-style fine-grained control through `VarHandle`, which lets you choose `setRelease` / `getAcquire` instead of paying full `volatile` everywhere. This matters when a field is published once and then read in a hot loop: a plain `volatile` read is an acquire on *every* read, but `getAcquire` lets you express intent precisely, and `setOpaque`/`getOpaque` give you atomicity without any ordering when you genuinely do not need it. Here is the same publication written with `VarHandle`, showing the *bug* (a plain store that the JIT may reorder or hoist) and the *fix* (an explicit release/acquire pair):

```java
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public final class FlagHolder {
    private int payload;          // plain field, published via the flag
    private boolean ready;        // NOT volatile — ordering comes from the VarHandle

    private static final VarHandle READY;
    static {
        try {
            READY = MethodHandles.lookup()
                .findVarHandle(FlagHolder.class, "ready", boolean.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    // BUG: a plain store gives no release; the payload write may sink below it,
    // or the JIT may keep payload in a register so a reader never sees it.
    public void publishBuggy(int v) {
        this.payload = v;
        this.ready = true;                    // plain store — NO barrier
    }

    // FIX: setRelease publishes payload; getAcquire subscribes to it.
    public void publish(int v) {
        this.payload = v;                     // plain store
        READY.setRelease(this, true);         // release: payload write is committed first
    }

    public int read() {
        if (!(boolean) READY.getAcquire(this)) return -1;  // acquire: subscribes
        return this.payload;                  // guaranteed to see v
    }
}
```

The `setRelease`/`getAcquire` pair is the JMM's release/acquire, identical in meaning to the C++ `memory_order_release`/`memory_order_acquire` store and load — the JIT lowers them to the same hardware fences. The buggy `publishBuggy` compiles to a plain store with no fence; on x86 it may *appear* to work (TSO forbids the StoreStore reorder for free), and then misbehave on an ARM-based JVM. Same trap, same fix, one language up the stack.

## Why this is exactly a happens-before edge

The [memory model post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) defines happens-before as the partial order that, if it covers every pair of conflicting accesses, makes a data race impossible. It is built from two ingredients: **program order** within a thread, and **synchronizes-with** edges across threads. A release-store that is read by an acquire-load is *the* canonical synchronizes-with edge. That is not a coincidence of this post and that one; it is the same fact stated at two altitudes.

Make the derivation explicit. Let $W_1, W_2$ be the writer's field stores and $R$ its release-store. Let $A$ be the reader's acquire-load and $L_1, L_2$ its field loads. We have, by program order on the writer, $W_1 \rightarrow W_2 \rightarrow R$ (release forbids $W_1, W_2$ from sinking below $R$). We have, by program order on the reader, $A \rightarrow L_1 \rightarrow L_2$ (acquire forbids $L_1, L_2$ from rising above $A$). And *if* $A$ reads the value $R$ wrote, the synchronizes-with rule gives $R \rightarrow A$. Compose by transitivity:

$$W_1 \rightarrow W_2 \rightarrow R \rightarrow A \rightarrow L_1 \rightarrow L_2$$

So $W_1$ happens-before $L_1$ and $W_2$ happens-before $L_2$. There is no race on the fields, and the reads are guaranteed to observe the writes. Remove the release barrier and the first arrow $W_2 \rightarrow R$ can break (a field write sinks below the flag); remove the acquire barrier and the last arrow $A \rightarrow L_1$ can break (a field read rises above the flag). Either break severs the chain and the field reads become a data race with undefined results. The barrier is *literally* the thing that buys the happens-before edge. Everything else — locks, channels, atomics — is sugar over placing these two barriers correctly.

This is also why "just use a lock" works: `unlock()` ends with a release-store on the lock word and `lock()` begins with an acquire-load on it, so the critical-section body is sandwiched in release/acquire and inherits the same happens-before chain. A mutex is, at bottom, an acquire on entry and a release on exit wrapped around mutual exclusion. When you write `synchronized` or `lock_guard`, you are placing these two barriers without naming them.

One property of this chain deserves emphasis because it is what makes complex lock-free code tractable: the synchronizes-with edge is *transitive through the release/acquire pair*, which means it carries memory that the flag itself never touched. The release-store wrote a *boolean* `ready`, but the happens-before edge it forms publishes the *entire prior write set* of the writer thread — the three config fields, a heap allocation, anything sequenced before the release. The reader's acquire on that one boolean subscribes to all of it. This is the deep reason a single one-bit flag can safely hand off a megabyte of data: the flag is not the payload, it is the *carrier* of the ordering edge, and the edge transports every write that came before it. Beginners often try to make the *data itself* atomic; that is both unnecessary and slower. You make the *flag* atomic with release/acquire and leave the data as plain memory, because the plain data is protected by the happens-before edge the flag creates, not by any property of the data's own type.

There is a precise boundary to this guarantee that trips people up: the edge only covers writes that *precede* the release in program order, and reads that *follow* the acquire. A write the writer performs *after* its release-store is not published by that release; a read the reader performs *before* its acquire-load is not covered by it. So the safe pattern is strict: writer does *all* its writes, *then* releases; reader acquires, *then* does *all* its reads. Interleave a stray write after the release or a stray read before the acquire and you have punched a hole in the chain. The compiler and CPU will happily exploit that hole, and the resulting bug — a field that is usually correct but occasionally stale — is among the hardest to reproduce in the entire discipline, because it depends on store-buffer timing you cannot control or observe directly.

## The full fence: StoreLoad and why it costs

Acquire and release between them cover three of the four barrier types: a release gives StoreStore + LoadStore, an acquire gives LoadLoad + LoadStore. Add them up across a release/acquire pair and you have LoadLoad, LoadStore, StoreStore — but *not* StoreLoad. No combination of one-way acquire/release barriers gives you StoreLoad ordering. For that you need a **full fence**, which in the C++ model is `std::memory_order_seq_cst` (sequential consistency) and in hardware is the StoreLoad barrier.

![A matrix comparing acquire release and full seq cst by what each pairs with what it blocks and its cost](/imgs/blogs/memory-barriers-acquire-release-and-fences-5.png)

Why is StoreLoad the one that costs real cycles? Because of the store buffer. When a core executes a store, it does not wait for the value to reach the cache coherency fabric and become visible to other cores — that would stall the pipeline on every write. Instead it drops the store into a small per-core FIFO (the store buffer) and moves on, retiring the instruction immediately. The store becomes globally visible later, when it drains. A *subsequent load* on the same core can execute and complete *before* the buffered store has drained — that is the StoreLoad reordering, and it is the *only* one x86 allows. To forbid it, the hardware must *drain the store buffer* before the load is allowed to read: that is what `mfence` does, and draining the buffer is a genuine pipeline stall measured in tens of cycles. The other three barriers never require a drain — they only constrain ordering among operations that were going to commit anyway.

The canonical place you *cannot* avoid a full fence is the **Dekker / Peterson** mutual-exclusion litmus: two threads, each writes its own flag then reads the other's flag, and the algorithm's correctness depends on at least one thread seeing the other's write. With only release/acquire, both threads can read the *stale* value of the other's flag — because each thread's flag-read (a load) is allowed to execute before its own flag-write (a store) drains, the StoreLoad reordering. Both enter the critical section. Only a StoreLoad fence (a `seq_cst` store-then-load, or an explicit full fence between the store and the load) forbids it. This is the textbook example of "acquire/release is not enough; you need sequential consistency." If your algorithm has a thread that *writes a shared location and then reads a different shared location and the relationship between those two matters*, suspect you need a full fence.

It is worth understanding *why* no stacking of one-way fences can manufacture a StoreLoad barrier, because the impossibility is structural, not an oversight. A release-store forbids earlier operations from sinking below it (StoreStore + LoadStore). An acquire-load forbids later operations from rising above it (LoadLoad + LoadStore). Now place a release-store followed by an acquire-load: `store(x, release); load(y, acquire)`. The release does not constrain anything *after* it — the later `load(y)` is free to climb. The acquire does not constrain anything *before* it — the earlier `store(x)` is free to sink. So the store and the load on opposite sides of the boundary can *still* swap: the load executes while the store sits in the buffer. The two one-way fences each guard their own side and neither guards the seam between a prior store and a later load. That seam is precisely StoreLoad, and closing it requires draining the buffer — which is a separate, heavier operation that neither one-way fence performs. This is why `seq_cst` is a distinct memory order and not just "acquire plus release": the difference is the buffer drain.

There is a second, subtler property of `seq_cst` beyond StoreLoad: it provides a *single total order* over all `seq_cst` operations that every thread agrees on. Release/acquire only gives you pairwise synchronizes-with edges; different threads can disagree about the relative order of *independent* release/acquire operations. With `seq_cst`, there is one global sequence into which every `seq_cst` operation slots, and no thread observes a contradiction. This total order is what certain algorithms — Peterson's lock, some hazard-pointer reclamation schemes, the independent-reads-of-independent-writes (IRIW) litmus — actually depend on, and it is strictly stronger than "no StoreLoad reorder on one thread." You pay for both properties together; you cannot buy only the cheaper half through the standard `seq_cst` order.

#### Worked example: store-then-load needs StoreLoad

Two threads, both flags start `false`:

```c
// litmus illustration, NOT runnable source
// T1                        T2
x = true;   // store         y = true;   // store
r1 = y;     // load          r2 = x;     // load
```

Question: can `r1 == false && r2 == false`? Under sequential consistency, no — at least one store must precede the other thread's load in the single global order. But under TSO (x86) and under release/acquire, *yes*: each thread's store sits in its store buffer while its load reads the other variable's still-`false` value from cache. Both read `false`. This is not a theoretical curiosity; it is the exact reason Peterson's lock needs a fence on real hardware, and why `std::atomic` defaults to `seq_cst` (the safe-but-not-free default). Insert `std::atomic_thread_fence(std::memory_order_seq_cst)` between the store and the load in each thread, or make the operations `seq_cst`, and `r1 == false && r2 == false` becomes impossible. The cost is the buffer drain on each thread.

#### Worked example: the cost of a full fence versus acquire and release

Now make the cost difference quantitative with a back-of-the-envelope. Suppose a publish operation does a tiny payload write and then sets a flag, and you run it in a tight loop. With `memory_order_release` on the flag store, on x86 the flag store is a plain `mov` and there is *no* added latency — the publish costs essentially the few cycles of the store itself, call it on the order of a handful of cycles. Switch that flag store to `memory_order_seq_cst` and on x86 the compiler must add the StoreLoad barrier, typically by emitting the store as a `lock`-prefixed exchange or by following it with `mfence`. That barrier drains the store buffer, and a store-buffer drain on a mainstream x86 core is on the order of *tens of cycles* — call it roughly 20 to 100 depending on microarchitecture and how full the buffer is. So in this idealized publish loop, the per-operation cost goes from a few cycles to a few-cycles-plus-tens-of-cycles: the full fence can dominate the operation, easily making the `seq_cst` publish on the order of a few times slower per iteration than the `release` publish.

The number that matters is not the absolute cycle count — it is the *ratio* and *when it bites*. If your publish loop is doing real work between publishes (allocating, copying a struct, touching cache lines), the buffer drain is amortized into that work and may be invisible. If your publish loop is essentially *just* the publish — a hot single-producer handoff — the drain is the whole cost and `seq_cst` shows up as a clean multiple. This is exactly why high-performance lock-free queues are religious about `release`/`acquire` and reserve `seq_cst` for the one place a store-then-load truly demands it. The honest way to confirm the ratio on your hardware is below in the measurement section; never trust the cycle figures here as exact — they are order-of-magnitude, and the only number you should act on is one you measured on the chip you ship to.

## Mapping to hardware fences

The barrier you *request* in source is compiled down to whatever the target ISA actually needs. The point of the language-level model (`std::memory_order_*`, the JMM) is precisely that you write your intent *once* and the compiler emits a *different* real fence per architecture. Here is the map for the three common ISAs.

![A stack showing a release store descending from your intent through the language model and a compiler barrier into a plain x86 move an ARM dmb ish and a POWER lwsync](/imgs/blogs/memory-barriers-acquire-release-and-fences-7.png)

On **x86 / x86-64** (TSO), acquire and release are *free at run time* — the hardware already forbids LoadLoad, LoadStore, and StoreStore. An `acquire` load or a `release` store compiles to a plain `mov`; the only thing the compiler must do is suppress its *own* reordering across the barrier (a compiler-only fence, zero instructions). A `seq_cst` store, however, needs the StoreLoad barrier, so it compiles to either an `xchg` (which has an implicit `lock` and so a full barrier) or a plain `mov` followed by `mfence`. So on x86 the *only* memory order you pay real cycles for is `seq_cst`.

On **ARM / AArch64** (weak), all four reorderings are allowed, so acquire and release each cost a real fence — but ARM gives you cheap *targeted* instructions: `ldar` (load-acquire) and `stlr` (store-release) bake the one-way barrier into the load/store itself, and the standalone `dmb ish` (data memory barrier, inner-shareable domain) is the general full fence. A `seq_cst` operation typically becomes a `dmb ish` or the `ldar`/`stlr` pair plus the StoreLoad-implying fence. This is why ARM exposed the config bug: a sloppy plain-store publication has *no* `stlr` and *no* `dmb`, so nothing forbids the StoreStore reorder.

On **POWER / PowerPC** (weak), the general barrier is `sync` (a heavyweight full fence) and the cheaper `lwsync` (lightweight sync) provides LoadLoad + LoadStore + StoreStore — exactly the cumulative ordering that acquire and release need, *without* the StoreLoad that `sync` adds. So a release/acquire pair on POWER is an `lwsync`, and a `seq_cst` operation is the full `sync`. The Linux kernel's `smp_mb()` is `sync` on POWER, `dmb ish` on ARM, and `mfence` (or a locked op) on x86; `smp_wmb()` (write barrier ≈ StoreStore) is `lwsync` on POWER and free on x86. The kernel's barrier macros are this table made into code.

There is a third weak-memory mechanism worth naming because it is how real ARM and POWER code avoids barriers entirely in the common case: **dependency ordering**. If the address of a load *depends* on the value of a prior load — you read a pointer and then dereference it — the hardware will not let the dependent load execute before the load it depends on, because it physically cannot: it does not yet know the address. This data dependency provides LoadLoad-like ordering *for free*, with no fence at all, on every architecture except the famously quirky DEC Alpha (which could reorder even dependent loads and is the reason the kernel once carried `smp_read_barrier_depends()`). This is exactly what `rcu_dereference()` and `std::memory_order_consume` were meant to exploit: a *consume* load is cheaper than an *acquire* because it relies on the dependency chain rather than a fence. In practice compilers found `consume` extremely hard to implement correctly — tracking which subsequent operations carry a real dependency through arbitrary code is brittle — so most of them silently promote `consume` to `acquire`, and the C++ committee has discussed redesigning it. The lesson for you is narrow but real: when you publish a *pointer* and the reader's only access is *through* that pointer, the dependency already orders the dereference after the pointer load on real hardware, and the acquire is conservatively covering a case the dependency mostly handles. The kernel's RCU exploits this to make its read side nearly free.

To see that acquire and release truly compile to *nothing* on x86 but to real instructions on ARM, look at the same release-store on both ISAs side by side. On x86-64 a release-store is a bare `mov`; on AArch64 it is the dedicated `stlr` (store-release register), which the hardware treats as a store with built-in release ordering:

```c
/* x86-64 vs AArch64: the SAME release-store, lowered differently.
   x86-64:   movq  %rsi, (%rdi)        ; plain store IS a release on TSO
   AArch64:  stlr  x1, [x0]            ; store-release: built-in one-way barrier

   And the matching acquire-load:
   x86-64:   movq  (%rdi), %rax        ; plain load IS an acquire on TSO
   AArch64:  ldar  x0, [x0]            ; load-acquire: built-in one-way barrier

   The full StoreLoad fence, which neither side gets for free:
   x86-64:   mfence                    ; or a lock-prefixed RMW
   AArch64:  dmb   ish                 ; inner-shareable full barrier */
```

That single side-by-side is the entire portability argument for a language memory model: you write `store(c, memory_order_release)` once, and the compiler emits a no-op-plus-compiler-barrier on x86 and a real `stlr` on ARM, so your code is *both* correct on the weak machine *and* free on the strong one. Hand-rolling the right fence per architecture is exactly the job you should let the model do.

Here is the asm made literal — a hand-written acquire/release publication and a full fence on x86-64, so you can see there is genuinely *no instruction* for the one-way barriers and a real one for the full fence:

```c
/* x86-64 GCC inline asm. Release-store: a plain mov is enough on TSO;
   the "memory" clobber is the COMPILER barrier that stops reordering. */
static inline void release_store(long *p, long v) {
    __asm__ __volatile__(
        "movq %1, %0\n\t"        /* plain store; x86 makes it a release for free */
        : "=m"(*p)
        : "r"(v)
        : "memory");             /* compiler barrier — no CPU fence emitted */
}

/* Full StoreLoad fence: this one is a real instruction. */
static inline void full_fence(void) {
    __asm__ __volatile__("mfence" ::: "memory");   /* drains the store buffer */
}
```

The `"memory"` clobber is doing real work even when no CPU instruction is emitted: it tells the compiler it may not move loads or stores across this point and must not cache memory in registers around it. That is the *compiler* half of the barrier; `mfence` is the *CPU* half. Both halves matter, and forgetting the compiler half is the next trap.

## The volatile trap: Java vs C and C++

Now the single most expensive misconception in this whole area, the one that turns up in interviews and 3 AM pages alike: **`volatile` means completely different things in Java and in C/C++, and only one of them is a memory barrier.**

![A matrix of volatile across languages comparing Java volatile C and C plus plus volatile and std atomic by whether each is a memory barrier provides visibility and what to use it for](/imgs/blogs/memory-barriers-acquire-release-and-fences-6.png)

**Java `volatile` is a real barrier.** Since Java 5 (JSR-133), a `volatile` write has release semantics and a `volatile` read has acquire semantics, and a write to a volatile field happens-before every subsequent read of it. It also guarantees *visibility* — a volatile read always sees the most recent volatile write, never a stale cached copy — and atomicity for `long`/`double` (which can otherwise tear). That is exactly the publication tool from the Java example above. When a Java engineer says "make the flag `volatile`," they are correctly reaching for an acquire/release barrier with cross-thread visibility.

**C and C++ `volatile` is NOT a barrier.** It was designed for memory-mapped I/O and signal handlers, not for threads. `volatile` in C/C++ guarantees exactly three things: the compiler will not *elide* a read or write (each access in source becomes a real access — useful for a hardware register that changes underneath you), it will not *reorder* `volatile` accesses *relative to each other*, and it will not cache the value in a register across accesses. It guarantees *none* of: ordering relative to *non-volatile* accesses, a CPU memory fence, cross-thread visibility, or atomicity. A `volatile int` in C++ shared between threads is a **data race** — undefined behavior — full stop. There is no StoreStore barrier, so the config bug is *not* fixed by making the pointer `volatile`. The compiler will not optimize the access away, which is why the bug *sometimes* hides on x86 (where the hardware happens to forbid the reorder anyway), but the language gives you nothing.

Here is the trap in code — the *broken* C++ that looks like it should work and does not:

```cpp
// BROKEN: volatile is NOT a thread-synchronization tool in C++.
struct Config { int rate_limit; };
Config* cfg = nullptr;
volatile bool ready = false;        // WRONG: not a barrier, not atomic

void writer() {
    cfg = new Config{5000};
    ready = true;                   // no release! the cfg store can sink below this
}

int reader() {
    while (!ready) { /* spin */ }   // no acquire! cfg read can rise above this
    return cfg->rate_limit;         // DATA RACE: may dereference garbage / read 0
}
```

The compiler will faithfully emit the `ready` load and store (because `volatile`), and on x86 the hardware's TSO may even make it *appear* to work — which is the cruelest part, because the bug ships and then surfaces on ARM. The standards-compliant fix is `std::atomic`, which *is* the portable barrier:

```cpp
// CORRECT: std::atomic gives the barrier and the cross-thread visibility.
#include <atomic>
struct Config { int rate_limit; };
Config* cfg = nullptr;
std::atomic<bool> ready{false};

void writer() {
    cfg = new Config{5000};
    ready.store(true, std::memory_order_release);   // release: publishes cfg
}

int reader() {
    while (!ready.load(std::memory_order_acquire)) { /* spin */ }  // acquire: subscribes
    return cfg->rate_limit;          // guaranteed to see 5000
}
```

The rule to carry: in C and C++, **`volatile` is for hardware you do not control (MMIO, `sig_atomic_t` in a signal handler), never for threads. Use `std::atomic` for thread communication.** In Java, `volatile` *is* the lightweight thread tool. Same keyword, opposite meaning. Rust sidesteps the trap entirely — there is no `volatile` keyword for sharing; you use `std::sync::atomic::AtomicBool` with an explicit `Ordering`, and `std::ptr::read_volatile`/`write_volatile` exist *only* for the MMIO case, clearly separated from the synchronization tools. Go has no `volatile` either; you use `sync/atomic` or a channel, both of which carry the barrier.

To be fair to C and C++ `volatile`, it is *not* useless — it is just misnamed for what most people reach for. Its three real jobs are genuine. First, **memory-mapped I/O**: a device register at a fixed address changes under you, so `volatile uint32_t* status = (uint32_t*)0xFEE00000;` forces the compiler to actually re-read the register each time instead of caching it, and not to elide a write that "looks dead" but actually pokes the hardware. Second, **`sig_atomic_t` in signal handlers**: a `volatile sig_atomic_t stop_flag` set in a handler and polled in the main loop, where the only concern is that the compiler not optimize the poll into a single cached read — single-core, single-address, no cross-thread coherence in play. Third, **`setjmp`/`longjmp`**: locals that must survive a `longjmp` are declared `volatile` so the compiler keeps them in memory, not a register that the jump clobbers. None of these is thread synchronization, and that is the whole point: `volatile` keeps the *compiler* honest about an access; it does nothing about *CPU* memory ordering or *cross-core* visibility. The instant a second core is reading what a first core wrote, you have left `volatile`'s job description and entered `std::atomic`'s.

Here is the same publication done correctly in Rust and in Go, to show that the languages designed after the threading lesson was learned simply do not offer the wrong tool:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

static READY: AtomicBool = AtomicBool::new(false);
static mut PAYLOAD: i32 = 0;   // protected by READY's happens-before edge

fn writer(v: i32) {
    unsafe { PAYLOAD = v; }                       // plain store
    READY.store(true, Ordering::Release);         // release: publishes PAYLOAD
}

fn reader() -> i32 {
    if !READY.load(Ordering::Acquire) { return -1; }  // acquire: subscribes
    unsafe { PAYLOAD }                            // guaranteed to see v
}
```

```go
import "sync/atomic"

var ready atomic.Bool
var payload int32   // published via ready's happens-before edge

func writer(v int32) {
    payload = v                  // plain store
    ready.Store(true)            // Go's atomic Store/Load carry release/acquire
}

func reader() int32 {
    if !ready.Load() {           // acquire-equivalent under the Go memory model
        return -1
    }
    return payload               // guaranteed to see v
}
```

Notice Go's `sync/atomic` does not even expose a memory-order parameter — the Go memory model specifies that an atomic `Store` synchronizes-with an atomic `Load` that observes it, which is release/acquire by definition, and Go deliberately does not let you ask for anything weaker because the language values "obviously correct" over "maximally tunable." Rust takes the opposite design stance and *forces* you to name the `Ordering`, so the choice is explicit and reviewable, and the type system's `Send`/`Sync` rules make sharing a non-atomic across threads a *compile error* rather than the undefined behavior C++ silently permits. Two philosophies, same underlying barrier, neither with a `volatile` foot-gun.

| Property | Java `volatile` | C/C++ `volatile` | C++ `std::atomic` |
| --- | --- | --- | --- |
| Memory barrier (ordering) | Yes — acquire/release | No | Yes — you choose the order |
| Cross-thread visibility | Yes | No (data race) | Yes |
| Atomic read/write | Yes (incl. `long`/`double`) | Not guaranteed | Yes |
| Intended use | thread flags, publication | MMIO, signal handlers | all lock-free / sharing |
| Safe to share between threads | Yes | No — undefined behavior | Yes |

## Standalone fences vs fences on atomics

There are two syntactic ways to place a barrier, and knowing when to use each saves cycles. The first is a **fence carried on the atomic operation itself** — `ready.store(true, std::memory_order_release)`. The barrier and the memory operation are fused; the release applies to *that* store. This is the common, preferred form: it is precise (you order around the exact operation that matters) and it lets the compiler pick the cheapest encoding.

The second is a **standalone fence** — `std::atomic_thread_fence(std::memory_order_release)` — a barrier not attached to any particular load or store. A standalone release fence orders *all* prior stores before *all* subsequent atomic stores; a standalone acquire fence orders all subsequent loads after all prior atomic loads. You reach for these when you want to publish *several* relaxed writes with a *single* barrier, instead of paying a release on each:

```cpp
#include <atomic>
std::atomic<int> a{0}, b{0}, flag{0};

void writer_batch() {
    a.store(1, std::memory_order_relaxed);   // cheap, unordered
    b.store(2, std::memory_order_relaxed);   // cheap, unordered
    std::atomic_thread_fence(std::memory_order_release);  // one barrier...
    flag.store(1, std::memory_order_relaxed);            // ...publishes a and b
}

void reader_batch() {
    while (flag.load(std::memory_order_relaxed) == 0) { }
    std::atomic_thread_fence(std::memory_order_acquire);  // one barrier subscribes
    int x = a.load(std::memory_order_relaxed);  // sees 1
    int y = b.load(std::memory_order_relaxed);  // sees 2
}
```

The standalone-fence form is what the Linux kernel uses (`smp_wmb()`, `smp_rmb()`, `smp_mb()` are standalone fences), and what you want when a barrier guards a *region* rather than a single variable. The fused form is what you want for a single flag. Both express the same hardware fence; pick the one that names your intent and costs the least. A common micro-optimization in lock-free ring buffers is exactly this: do all the slot writes relaxed, then one release fence, then publish the index — one barrier instead of N.

## Case studies / real-world

**The Linux kernel's `smp_*` barrier family.** The kernel cannot rely on a language memory model (it long predates the C11 one and supports exotic ISAs), so it defines its own barrier macros and a famous documentation file, `Documentation/memory-barriers.txt`, that is required reading. `smp_mb()` is a full barrier (StoreLoad included), `smp_rmb()` is a read barrier (LoadLoad), `smp_wmb()` is a write barrier (StoreStore), and `smp_load_acquire()`/`smp_store_release()` are the one-way pair. On x86 `smp_wmb()` and the acquire/release helpers compile to *nothing* (just a compiler barrier); on ARM they become `dmb` variants; on POWER `smp_wmb()` is `lwsync` and `smp_mb()` is `sync`. The kernel's RCU (read-copy-update) mechanism is built directly on `smp_store_release()` to publish a new node and `rcu_dereference()` (a dependency-ordered acquire) to subscribe — the exact publish/subscribe pattern from this post, at the heart of the most-used lock-free technique in the kernel.

**The Java `volatile` publication idiom and double-checked locking.** Before Java 5, the double-checked-locking singleton was *broken* — the famous "Double-Checked Locking is Broken" declaration signed by many JVM experts — precisely because the old memory model did not give the volatile read acquire semantics, so a thread could see a non-null reference to a not-yet-constructed object (our config bug, in singleton clothing). JSR-133 (Java 5) redefined `volatile` to have acquire/release semantics, which *fixed* double-checked locking: declaring the instance field `volatile` makes the publish a release and the check an acquire, and the idiom became correct. This is a real, documented language-level memory-model change driven by exactly the barrier semantics this post covers.

**The seqlock (sequence lock).** A seqlock is a lock-free reader pattern used heavily in the Linux kernel for things like reading the system time (`gettimeofday` fast path) — data that is read far more often than written. The writer increments a sequence counter to an *odd* value (with a release/write barrier) before modifying the data, then increments it to the next *even* value (with another barrier) after. A reader snapshots the counter (acquire/read barrier), reads the data, then re-reads the counter: if it changed or was odd, a write was in progress and the reader retries. The whole correctness argument rests on the StoreStore barrier between bumping the counter and writing the data, and the LoadLoad barrier between reading the data and re-checking the counter — barriers, not locks, are what make the reader see a *consistent* snapshot without ever blocking the writer.

#### Worked example: a seqlock reader and writer

```c
#include <stdatomic.h>
atomic_uint seq = 0;
int data_x = 0, data_y = 0;   // protected pair that must be read consistently

void writer(int nx, int ny) {
    unsigned s = atomic_load_explicit(&seq, memory_order_relaxed);
    atomic_store_explicit(&seq, s + 1, memory_order_relaxed);   // now ODD: write in progress
    atomic_thread_fence(memory_order_release);                  // StoreStore: bump before data
    data_x = nx;
    data_y = ny;
    atomic_thread_fence(memory_order_release);                  // data before final bump
    atomic_store_explicit(&seq, s + 2, memory_order_relaxed);   // now EVEN: done
}

int reader(int *out_x, int *out_y) {
    unsigned s0, s1;
    do {
        s0 = atomic_load_explicit(&seq, memory_order_acquire);  // snapshot
        if (s0 & 1u) continue;                                  // odd: writer mid-update, retry
        *out_x = data_x;
        *out_y = data_y;
        atomic_thread_fence(memory_order_acquire);              // data reads before recheck
        s1 = atomic_load_explicit(&seq, memory_order_relaxed);
    } while (s0 != s1);                                         // changed: retry
    return 0;
}
```

The reader never takes a lock and never blocks the writer; it just retries if it caught the writer mid-update. The price is that readers must be side-effect-free and retryable, and the data must be copyable in the read window. Notice there is *not a single mutex here* — the entire mutual visibility is bought with four barriers. This is the kind of code where understanding acquire/release versus full fences is not academic: get the barrier wrong and the reader silently returns a torn `(x, y)` pair a few times a day. The Linux kernel uses exactly this structure for `seqlock_t` (see `include/linux/seqlock.h`); the time-of-day fast path reads the kernel's clock without taking a spinlock, which is why a million `gettimeofday`-style calls do not serialize on a single lock.

**The LMAX Disruptor.** The LMAX exchange's open-source Disruptor is a single-writer, multi-reader ring buffer that famously processed on the order of millions of messages per second on a single thread, and a large part of its speed comes from getting the memory barriers exactly right instead of using locks. Producers claim a slot, write the event into pre-allocated ring storage, and then publish by advancing a *sequence* with a release-store; consumers spin on the published sequence with an acquire-load and only then read the event. It is the publish/subscribe pattern of this post, scaled into a high-throughput pipeline, and its authors documented (in the Disruptor technical paper) that replacing lock-based queues with a barrier-ordered ring was the core of the win. The Disruptor also pads its sequence counters to a full cache line to avoid false sharing — a reminder that barriers fix ordering while padding fixes contention, and a serious lock-free structure needs both.

**The x86 IRIW surprise and why even `seq_cst` matters.** A subtle litmus called *independent reads of independent writes* (IRIW) shows two writer threads each setting a different variable, and two reader threads each reading both variables in opposite order. On a sufficiently weak model, the two readers can disagree about the *order* of the two independent writes — reader A thinks `x` was set first, reader B thinks `y` was first — even though each reader used acquire loads. Plain acquire/release does not forbid this, because it only forms pairwise edges and does not impose a single global order. Only `seq_cst` on the operations restores a total order all four threads agree on. x86's TSO happens to forbid IRIW by accident (its store buffers are per-core but there is a single point of coherence), but ARM and POWER historically allowed it, which is the practical reason the C++ and Java memory models bother to define a `seq_cst` total order at all. It is rare to *need* IRIW agreement, but when you do, no amount of acquire/release substitutes for the full order.

## Measured: what a fence actually costs

Now the honest numbers, with the loud caveat that fence cost is *deeply* platform-, microarchitecture-, and contention-dependent, and that microbenchmarking fences is notoriously hard because the CPU is reordering and speculating around your measurement. Treat these as order-of-magnitude, not gospel; measure on *your* hardware with a warmed-up loop, many iterations, and the surrounding work that actually stresses the store buffer.

The robust qualitative findings, which hold across a wide range of x86 and ARM parts:

| Operation | x86-64 (TSO) | ARM (weak) | Why |
| --- | --- | --- | --- |
| acquire-load / release-store | ~0 extra cycles (plain mov) | a few cycles (`ldar`/`stlr`) | x86 forbids the relevant reorders for free |
| full fence (`mfence` / `dmb ish`) | tens of cycles (often ~20–100) | tens of cycles | must drain the store buffer |
| `seq_cst` atomic store | tens of cycles (it is a full fence) | tens of cycles | StoreLoad needs the drain |
| relaxed atomic op | ~0 extra (just the op) | ~0 extra | no barrier at all |
| `lock`-prefixed RMW (CAS) | tens of cycles, more under contention | similar | full barrier + cache-line ownership |

The headline, measured many times by many people: on x86, **acquire/release is essentially free and `seq_cst` is the only memory order that costs you** — the difference between an acquire-load and a `seq_cst` load in a tight publication loop is the difference between a `mov` and a `mov` plus the StoreLoad drain, which is why high-performance lock-free code is so careful to use `acquire`/`release` and reach for `seq_cst` only where a Dekker-style store-then-load demands it. On ARM, *all* the barriers cost something, so the gap between weak and strong ordering is smaller in relative terms but the absolute baseline is higher — and the *correctness* stakes are higher because the weak model actually performs the reorderings x86 hides.

A word on *why microbenchmarking fences is treacherous*, because this is where well-meaning engineers publish wrong numbers. The CPU is actively working against your measurement. If your benchmark loop has no real dependency between iterations, the out-of-order engine overlaps many iterations and hides the fence's latency, so you measure something far cheaper than the fence costs in a realistic dependent chain. If your loop is too short, the timer resolution and loop overhead swamp the signal. If you forget to prevent the compiler from optimizing the whole loop into a constant (it can, if it proves the result is unused), you measure nothing at all. And the store buffer's depth matters: a fence that drains an *empty* buffer is far cheaper than one that drains a *full* one, so the apparent cost of `mfence` depends on how many pending stores precede it — meaning the number you get is sensitive to the exact instruction mix around it. The defensible methodology is: insert a real data dependency so iterations cannot overlap, use a `volatile`-sink or a compiler barrier to keep the work live, run for whole seconds across millions of iterations, pin the thread to a core to avoid migration noise, repeat the whole run a dozen times, and report the median with the spread — never a single "X nanoseconds" point estimate. If after all that you cannot show a difference between `release` and `seq_cst`, the honest conclusion is that the fence is not your bottleneck, which is a result, not a failure.

A second honest caveat: the *contended* cost dwarfs the *uncontended* cost, and it is the one that actually shows up in production. The cycle figures above are for a barrier hitting a cache line the core already owns. The moment another core is fighting for that line, a `lock`-prefixed read-modify-write (a CAS) does not just drain the buffer — it must win exclusive ownership of the cache line through the coherence protocol, which under heavy contention can cost *hundreds* of cycles and grows worse with core count. So the relevant comparison in a real system is rarely "release versus seq_cst on an idle line"; it is "how often is this line contended," and that is a false-sharing-and-contention question, not a memory-order question. Get the memory order right for correctness with the cheapest fence that is correct, then profile for contention as a separate axis.

#### Worked example: acquire/release vs seq_cst in a publish loop

Picture a single-producer, single-consumer flag handoff repeated 100 million times, the producer doing a tiny amount of work then publishing, the consumer spinning on acquire. Switching the publish from `memory_order_release` to `memory_order_seq_cst` on an x86 box typically *adds the StoreLoad drain to every publish* — in a tight loop that can show up as a measurable throughput drop (often on the order of 1.5×–3× slower per-op for the publish itself, swamped or revealed depending on how much real work surrounds it). On ARM the relative gap is smaller because release was not free to begin with. The lesson is not "seq_cst is always slow" — it is "seq_cst buys you StoreLoad ordering you usually do not need, so do not pay for it by default." How to measure it honestly: warm up the caches and the branch predictor, run for seconds not milliseconds, pin threads to cores, repeat the run a dozen times and report the median and spread, and *vary the surrounding work* so you are not measuring a degenerate loop the compiler optimized into nonsense. If you cannot show the difference, you may not have a bottleneck — which is itself the answer.

## When to reach for this (and when not to)

Memory barriers are a sharp, low-level tool. Most application code should *never* place a raw barrier — it should use a lock, a `volatile` (Java), a channel, an `Arc<Mutex<T>>`, a `ConcurrentHashMap`, or a thread-safe queue, all of which place the barriers correctly for you. Reach for explicit barriers only when you are *building* such a primitive or chasing a measured bottleneck in one. Here is the decision, made plainly.

![A full seq cst fence that drains the store buffer at tens of cycles versus a cheaper acquire release one way gate and when each is needed](/imgs/blogs/memory-barriers-acquire-release-and-fences-8.png)

**Reach for acquire/release (publish/subscribe) when:** you are safely publishing an object built on one thread to others (the config holder, an immutable snapshot, a newly-allocated lock-free node); you are building a single-producer/single-consumer queue or a flag handoff; you are implementing a lock or a one-shot latch. This is the 95% case, and it is cheap — near-free on x86, modest on ARM. Always pair them: a lone release with no matching acquire publishes to nobody, a lone acquire that reads a relaxed store subscribes to nothing.

**Reach for a full fence (`seq_cst`) when, and only when:** your algorithm has a thread that *stores* one shared location and then *loads* a different shared location, and correctness depends on at least one thread observing the other's store — the Dekker/Peterson store-then-load, a Petersen-style mutual exclusion, certain wait-free constructions, or a sequence barrier where total ordering across all threads is required. If you cannot point at a store-then-load that must not reorder, you almost certainly do not need a full fence, and `seq_cst` is just a tax.

**Do not reach for raw barriers when:** a mutex is not your bottleneck (measure first — an uncontended mutex is ~20–40 ns and correct by construction); you are tempted to use C/C++ `volatile` for thread communication (use `std::atomic`); you think `relaxed` is "faster so why not" without a proof that the ordering does not matter (relaxed is for counters and statistics where you genuinely do not care about ordering, like a hit-counter you read approximately); or you are on a deadline and not building infrastructure. The failure mode of getting a barrier wrong is the worst kind: a bug that passes every test on your x86 dev box and corrupts data once a day on the ARM production fleet. If you are not certain, use the higher-level primitive — that is not a cop-out, it is the senior move.

A last sizing note: barriers fix *ordering*, not *contention*. If your problem is that 64 threads are all hammering one cache line, no choice of memory order will save you — that is false sharing or a hot lock, a different post's problem. Barriers make a *correct* algorithm; they do not make a *scalable* one. Get the ordering right with the cheapest barrier that is correct, then measure for contention separately.

## Key takeaways

- There are exactly four hardware reorderings — **LoadLoad, LoadStore, StoreStore, StoreLoad** — and a barrier is named by the pair it forbids. StoreLoad is the expensive one because only it requires draining the store buffer.
- **Acquire** is a one-way roof on a load (nothing later sinks above it); **release** is a one-way floor on a store (nothing earlier rises below it). They are duals and are meant to be used as a matched pair.
- A **release-store publishes** everything written before it; an **acquire-load that reads it subscribes** to all of it. This publish/subscribe pair *is* the synchronizes-with edge of happens-before — the bedrock of safe publication, locks, and lock-free code.
- A release/acquire pair gives you LoadLoad + LoadStore + StoreStore but **not StoreLoad**. For StoreLoad you need a **full fence** (`seq_cst`), which is the only memory order that costs real cycles on x86.
- These compile to *different* real fences per ISA: free `mov` on x86 TSO for acquire/release, `dmb ish` / `ldar` / `stlr` on ARM, `lwsync` (one-way) and `sync` (full) on POWER. The Linux `smp_*` macros are this map in code.
- **Java `volatile` is a real acquire/release barrier with visibility; C and C++ `volatile` is NOT a barrier** — it is for MMIO and signal handlers, and sharing it between threads is undefined behavior. Use `std::atomic` (C++), `AtomicBool` (Rust), or `sync/atomic` (Go) instead.
- A bug here often *works by accident on x86* (TSO hides StoreStore for free) and *only fails on ARM/POWER* — never trust "it works on my x86 laptop" as evidence of ordering correctness.
- Use the cheapest barrier that is correct: acquire/release for publication (the 95% case), a full fence only for a store-then-load that must not reorder. When in doubt, use a higher-level primitive — it places the barriers for you.

## Further reading

- *The Art of Multiprocessor Programming*, Herlihy & Shavit — the rigorous treatment of memory consistency, linearizability, and the synchronization primitives barriers underpin.
- *C++ Concurrency in Action*, Anthony Williams — chapters 5 and 7 are the definitive practical guide to `std::memory_order`, fences, and lock-free data structures.
- *Java Concurrency in Practice*, Goetz et al. — the JMM, `volatile`, safe publication, and why double-checked locking was broken and then fixed.
- Jeff Preshing's blog ("Acquire and Release Semantics", "Memory Barriers Are Like Source Control Operations", "This Is Why They Call It a Weakly-Ordered CPU") — the clearest intuition-building writing on barriers anywhere.
- The Linux kernel `Documentation/memory-barriers.txt` — the canonical, exhaustive treatment of `smp_mb`, `smp_rmb`, `smp_wmb`, and dependency ordering, with real ISA mappings.
- "The JSR-133 Cookbook for Compiler Writers" by Doug Lea — the barrier-placement reference behind the Java Memory Model.
- Within this series: [why your code does not run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering), [memory models, sequential consistency, and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before), [atomics and memory orderings from relaxed to seq_cst](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), and the capstone [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model). For where these barriers sit in the broader memory system, see [the memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).

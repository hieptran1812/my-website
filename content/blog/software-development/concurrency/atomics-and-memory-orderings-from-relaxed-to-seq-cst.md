---
title: "Atomics and Memory Orderings: From Relaxed to Seq-Cst"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A practitioner's tour of atomic operations and the five memory orderings, with runnable counters, handoffs, the relaxed-as-a-lock bug, and honest x86-vs-ARM numbers."
tags:
  [
    "concurrency",
    "parallelism",
    "atomics",
    "memory-order",
    "relaxed",
    "seq-cst",
    "acquire-release",
    "lock-free",
    "systems-programming",
    "memory-model",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-1.png"
---

There is a moment, the first time you write lock-free code, when you discover that a counter can be perfectly correct and a flag right next to it can be completely broken — and that both bugs live in the same word of memory. You replaced `count++` with an atomic increment, ran it across eight threads, and the final total came out exact every single time. Emboldened, you used the same atomic type for a `ready` flag: the producer writes a buffer, sets `ready = 1`, the consumer waits for `ready == 1` and then reads the buffer. It works on your laptop a thousand times. Then it ships, runs on a phone with an ARM core, and the consumer reads a buffer full of zeros — even though `ready` was unmistakably `1`. The flag was atomic. The buffer was garbage anyway.

This is the lesson that atomics teach, and it is the single most important idea in this post: **atomicity is not ordering.** An atomic operation guarantees that nobody ever sees a half-finished value — no torn 64-bit write, no increment that loses against a concurrent increment. That is a real and useful guarantee. But it says *nothing* about whether the *other* writes you did before the atomic become visible to another thread, or in what order. The counter worked because the increments were independent: every thread just needed its own add to not get lost, and the final read happened after everyone joined. The flag broke because a flag is a *promise about other memory*, and a bare atomic makes no such promise. Closing that gap — turning "this write is atomic" into "this write *publishes* everything before it, and reading it *subscribes* to everything before it" — is exactly what **memory orderings** are for.

This post is about atomics as the typed, portable interface to the hardware memory model. We will define what an atomic operation actually is (load, store, exchange, compare-and-swap, fetch-add — all indivisible), then climb the ordering menu from `relaxed` (atomic, but orders nothing) through `acquire`/`release` (the publish-and-subscribe handoff) to `seq_cst` (the single total order that is the safe, costly default). We will see precisely what each one *costs* on x86 versus ARM — because the answer genuinely differs by hardware — and we will write the two canonical programs side by side: a correct lock-free counter that *should* use relaxed, and a correct flag handoff that *must* use release/acquire. We will write the bug, too: relaxed misused as a lock, the stale-data corruption it causes, and the one-word fix. The figure below is the map of the whole territory — the five orderings, what each guarantees, what it's for, and what it costs.

![A comparison table of the five memory orderings showing relaxed acquire release acq rel and seq cst against their guarantee typical use and cost columns](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-1.png)

If you have read the sibling posts on [why your code doesn't run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering) and [memory models and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before), you already know *why* reordering happens — the store buffer, the compiler, the weakly-ordered core. This post is the *interface* to all of that: how you, the programmer, dial in exactly as much ordering as you need and not one fence more. That is the whole craft. Too little ordering and you ship a Heisenbug that only fails on a weak-memory ARM server under load; too much and you leave throughput on the floor and slow down a hot path that didn't need a single barrier. The goal is the cheapest ordering that is still correct, chosen *on purpose* and *provably*.

## What an atomic operation actually is

Start with the thing every concurrency bug begins with: `counter++`. To a compiler this is not one operation; it is three. Load the current value from memory into a register, add one in the register, store the register back to memory. Call them L, M, S. On a single thread the L-M-S sequence is invisible — nothing else touches `counter` between them. Run two threads, and the scheduler is free to interleave the six instructions any way it likes. The interleaving L1, L2, M1, M2, S1, S2 — both threads load the same old value, both compute old+1, both store old+1 — loses an update. Two increments, one net result. This is the *lost update*, the canonical [data race](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction), and it is why a plain `int` shared across threads is broken even for something as trivial as a hit counter.

An **atomic operation** is the hardware's answer to this: an operation that is *indivisible* with respect to other threads. No other thread can observe it half-done, and no other thread's atomic operation on the same location can interleave inside it. When the CPU executes an atomic fetch-add, the load, the add, and the store happen as one unit that nobody can split. Concretely, on x86 this is a `lock`-prefixed instruction (`lock xadd` for fetch-add, `lock cmpxchg` for compare-and-swap); the `lock` prefix tells the core to hold exclusive ownership of the cache line for the duration so no other core can sneak a read-modify-write in between. On ARM and other load-linked/store-conditional machines, an atomic RMW is a small retry loop: load-exclusive (`ldxr`), modify, store-exclusive (`stxr`) which *fails* if anyone touched the line since the load-exclusive, looping until it succeeds.

The atomic operations you get, in roughly increasing order of power:

- **atomic load** — read a value such that you never see a torn (half-updated) result. A plain 64-bit read on a 32-bit machine can tear (read the new low half and the old high half); an atomic load cannot.
- **atomic store** — write a value indivisibly; no reader ever sees a half-written word.
- **exchange (swap)** — atomically write a new value and return the old one in a single step.
- **fetch-add / fetch-or / fetch-and** — read-modify-write: atomically add (or OR, or AND) and return the previous value. This is the lock-free counter's workhorse.
- **compare-and-swap (CAS)** — the king. Atomically: *if* the location currently holds `expected`, write `desired` and report success; otherwise report failure and (usually) hand back the actual current value. Every other primitive can be built from CAS in a loop. It is one `cmpxchg` on x86, one LL/SC loop on ARM, and it is the foundation of every lock-free data structure — see [compare-and-swap and building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures).

The reason CAS is the king is that it lets you do a *conditional* update — "change it only if nobody changed it under me" — which is the atomic primitive you need to build everything else without a lock. A fetch-add is unconditional; it always succeeds. But to push onto a lock-free stack you must read the current head, build a node pointing at it, and install your node as the new head *only if the head is still what you read* — otherwise another thread pushed in between and you'd lose their node. That "only if still what I read" is precisely CAS, and the standard shape is a *CAS loop*: read, compute, attempt CAS, and if it fails (someone changed the value) retry with the fresh value. C++ even gives you two variants — `compare_exchange_strong` and `compare_exchange_weak` — where the weak form may fail *spuriously* (return false even when the value matched) because on LL/SC machines the store-conditional can fail for unrelated reasons like a context switch; you use the weak form inside a loop (the loop already retries) and the strong form when you're not looping. This is the place atomic *orderings* and atomic *operations* meet most intensely, which is why the next two sections matter so much for anyone building these structures.

Here is the same atomic counter increment in three languages, so the shared concept is unmistakable and the idioms are visible:

```cpp
// atomic counter increment, C++
#include <atomic>
std::atomic<long> counter{0};

void bump() {
    counter.fetch_add(1, std::memory_order_relaxed); // indivisible add
}
```

```rust
use std::sync::atomic::{AtomicI64, Ordering};
static COUNTER: AtomicI64 = AtomicI64::new(0);

fn bump() {
    COUNTER.fetch_add(1, Ordering::Relaxed); // indivisible add
}
```

```java
import java.util.concurrent.atomic.AtomicLong;
static final AtomicLong counter = new AtomicLong(0);

static void bump() {
    counter.getAndIncrement(); // indivisible add, seq-cst in the JMM
}
```

Notice what is the *same*: in all three, the increment cannot lose an update no matter how the threads interleave, because the read-modify-write is one unit. Notice what *differs*: C++ and Rust force you to spell out the ordering (`relaxed` here), while Java's `AtomicLong.getAndIncrement()` always uses the strongest ordering. We will spend the rest of this post on that difference, because it is the entire substance of "memory orderings."

#### Worked example: the lost update, made concrete

Two threads each run `bump()` ten million times on a non-atomic `long`. The instruction stream is L-M-S per increment, 20 million increments total. Whenever thread B's load lands between thread A's load and store (or vice versa), one increment is silently dropped. On a contended counter that overlap is common — not rare. Measured on a 4-core x86 laptop, the final value of a *non-atomic* counter after 20 million increments routinely lands between 10.2 and 14.8 million, losing 25–50% of the increments depending on contention. Swap in an atomic fetch-add and the value is *always* exactly 20,000,000. The atomicity closed the L-M-S window. We have not yet said one word about ordering, and we did not need to — because every thread's only requirement was "don't lose my add," which is purely an atomicity property. That is the whole reason a counter is the *easy* case.

## The ordering menu: relaxed, acquire, release, acq-rel, seq-cst

Atomicity stops other threads from splitting *one* operation. Ordering controls how the *rest* of memory — your ordinary, non-atomic reads and writes — becomes visible *around* that atomic operation. The reason this is a separate knob is that modern hardware and compilers reorder memory operations aggressively for speed (store buffers, write combining, out-of-order execution, instruction scheduling). An atomic operation is a place where you can *constrain* that reordering. Each ordering says, precisely, which reorderings are forbidden near this atomic. The C++ standard names five (`std::memory_order_*`); Rust mirrors them in `std::sync::atomic::Ordering`; Java reaches the same guarantees through `volatile`, `VarHandle` access modes, and the `Atomic*` classes. The menu, from weakest to strongest:

- **relaxed** — atomic, and *that is all*. The operation itself can't tear or lose, but it imposes no ordering on any other memory access. The compiler and CPU may move surrounding loads and stores freely across it. Use it only when no other thread's correctness depends on the order of your writes: independent counters, statistics, a flag whose stale value you will re-check anyway.
- **acquire** — applies to a *load*. An acquire load is a one-way fence: no read or write that comes *after* it in program order may be moved *before* it. Concretely, once an acquire load reads a value that some other thread released, all the writes that thread did *before* the release become visible to you. It is the *subscribe* half of a handoff.
- **release** — applies to a *store*. A release store is the mirror one-way fence: no read or write *before* it may be moved *after* it. Everything you wrote before the release is *published* to any thread that later does an acquire load of this value. It is the *publish* half.
- **acq-rel** — applies to a *read-modify-write* (fetch-add, exchange, CAS). It is acquire on the load side and release on the store side at once: it subscribes to prior publishers *and* publishes your prior writes. This is what you want on the CAS in a lock-free push, or on a lock's unlock-then-relock handoff.
- **seq-cst** (sequentially consistent) — the strongest, and the *default* in C++/Rust when you omit the argument, and what Java's `volatile` and `Atomic*` give you. It is acquire-and-release *plus* one extra guarantee: all `seq_cst` operations across all threads appear in a *single, global total order* that every thread agrees on. That total order is what you need for the subtle cases (Dekker-style mutual exclusion, a store on one thread and a load on another that must not *both* appear to go first).

The mental shorthand most practitioners carry: **relaxed = atomic only; acquire/release = a directed handoff between exactly two operations; seq-cst = a global agreement everyone shares.** Acquire and release tie one release to the acquires that read it; seq-cst ties *every* seq-cst operation into one timeline. The price scales with the strength, and where you pay it depends on the hardware — which is the whole second half of this post.

A small but load-bearing detail: acquire and release are *directional and paired*. A release store does nothing on its own; it only matters because some *other* thread does an acquire load of the value it stored, and that pairing is what creates a [happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) edge between the two threads. An acquire with no matching release, or a release nobody ever acquires, is wasted ceremony. You always design the *pair*.

## Relaxed: when order genuinely does not matter

Relaxed ordering is the one people fear and the one people misuse, so let us pin down exactly when it is *safe*. The rule is sharp: **relaxed is safe when no other thread's correctness depends on the ordering of your memory operations relative to this atomic** — only on the atomic's own indivisibility. The cleanest example is a hit counter or a statistics accumulator. Suppose every request handler does `requests.fetch_add(1, relaxed)`, and a metrics thread reads the total once a second. Does any handler care whether its increment is ordered before or after some *other* memory write it did? No. Does the metrics reader need a precise instantaneous snapshot? No — it samples an approximate, monotonically growing number. The only requirement is "don't lose an increment," which is pure atomicity. Relaxed delivers exactly that and nothing you have to pay for.

```cpp
std::atomic<uint64_t> requests{0};

void on_request() {
    // Pure counter: no other memory's visibility hinges on this. Relaxed is correct AND cheapest.
    requests.fetch_add(1, std::memory_order_relaxed);
}

uint64_t snapshot() {
    return requests.load(std::memory_order_relaxed); // approximate, monotonic, fine
}
```

```rust
use std::sync::atomic::{AtomicU64, Ordering};
static REQUESTS: AtomicU64 = AtomicU64::new(0);

fn on_request() {
    REQUESTS.fetch_add(1, Ordering::Relaxed); // independent counter, relaxed is right
}
```

The thing relaxed does *not* promise is just as important to internalize. Relaxed gives you atomicity and one more subtle guarantee — *modification-order consistency*: all threads agree on the order of operations *to that single location*. If thread A does `x.store(1, relaxed)` then `x.store(2, relaxed)`, no thread ever sees `x` go 2 then 1. That is why a relaxed counter is fine: the per-location order is coherent, so increments compose into a correct total. What relaxed does *not* give is any ordering *between different locations*. A relaxed write to `data` and a relaxed write to `ready` can be observed by another thread in *either* order. That is precisely the property a flag handoff depends on, and precisely why relaxed cannot carry a handoff. The figure makes the failure concrete: a relaxed flag is atomic, but the data it is supposed to guard can still arrive stale.

![A before and after comparison showing a relaxed flag that lets the reader see ready while the payload is stale versus release and acquire that make the data visible](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-2.png)

There is a second legitimate relaxed pattern worth naming, because it surprises people: a flag you will **re-check** under a stronger fence. A common shutdown idiom is a relaxed `stop` flag that worker loops poll cheaply; when they *see* it set, they perform the actual teardown under a proper lock or an acquire fence that re-validates state. The relaxed load is just a cheap "should I even bother looking harder?" hint. If a worker misses the flag for one extra loop iteration because the relaxed load was reordered, nothing breaks — it catches it next iteration, and the *real* synchronization happens at the re-check. Relaxed as a *hint that triggers a stronger check* is safe. Relaxed as the *sole* synchronization for a handoff is a bug.

#### Worked example: why a relaxed counter needs no fence

Picture four threads, each doing one million relaxed `fetch_add(1)`. Each increment is a `lock xadd` on x86 — atomic, but with *no* fence around it, because relaxed asks for none. The hardware still serializes the read-modify-writes to that one cache line (it must, to be atomic), so the line ping-pongs between cores and the *final* total is exactly 4,000,000 every run. Now ask: did we ever need a write to *any other* variable to be ordered around these increments? No — the threads share nothing but the counter, and the result is read only after `join()`, which itself establishes happens-before. So there is no other memory whose visibility we had to guarantee. The relaxed increment is not just *acceptable* here; it is *optimal* — it is the strongest correctness you can have (exact total) at the lowest cost (no fence). Reaching for seq-cst would add a fence per increment that buys literally nothing, which is the next figure's whole point.

## Acquire and release: the handoff

Now the case relaxed cannot handle: one thread produces some data and a second thread consumes it. The producer fills a buffer, then sets a `ready` flag; the consumer waits until `ready` is set, then reads the buffer. For this to be correct, two orderings must hold. First, on the producer, the buffer writes must *not* be reordered to happen *after* the flag write — otherwise the consumer can see `ready` before the buffer is filled. Second, on the consumer, the buffer reads must *not* be reordered to happen *before* the flag read — otherwise it reads the buffer before checking the flag. **Release** enforces the first; **acquire** enforces the second; together they create a [happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) edge from the producer's writes to the consumer's reads.

The precise guarantee, stated carefully because it is the heart of the whole post: *if* a release store of value V on thread A is later read by an acquire load on thread B that observes V, *then* everything A wrote (atomic or not) before the release store is guaranteed visible to B after the acquire load. The flag write is the *carrier*; the acquire/release annotations make the flag's transit *drag the rest of memory along with it*. This is exactly the [memory barrier](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences) behavior, just attached to a specific atomic operation instead of standing as a free-floating fence — which is why acquire/release are often called "one-sided fences." The figure shows the handoff as a timeline: every write the producer did before the release becomes visible the moment the consumer's acquire load sees the flag.

![A timeline of a release store publishing payload writes and a matching acquire load subscribing so all the writer's earlier writes become visible to the reader](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-3.png)

Here is the handoff written correctly, in C++ and Rust, with the non-atomic payload guarded by an atomic flag:

```cpp
// release/acquire handoff, C++
#include <atomic>
#include <thread>

int payload = 0;                 // ordinary, non-atomic data
std::atomic<bool> ready{false};  // the carrier flag

void producer() {
    payload = 42;                                  // (1) ordinary write
    ready.store(true, std::memory_order_release);  // (2) publish: (1) cannot move after this
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) // (3) subscribe: spin until published
        ;
    // Guaranteed: payload == 42 here. The acquire that saw `true` pulled in write (1).
    assert(payload == 42);
}
```

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::cell::UnsafeCell;

struct Channel { payload: UnsafeCell<i32>, ready: AtomicBool }
unsafe impl Sync for Channel {}

fn producer(ch: &Channel) {
    unsafe { *ch.payload.get() = 42; }          // (1) ordinary write
    ch.ready.store(true, Ordering::Release);    // (2) publish
}

fn consumer(ch: &Channel) -> i32 {
    while !ch.ready.load(Ordering::Acquire) {}  // (3) subscribe
    unsafe { *ch.payload.get() }                // guaranteed 42
}
```

The same pattern in Java, where you reach the identical guarantee through either `volatile` (which is seq-cst, stronger than needed but simple) or a `VarHandle` with explicit acquire/release modes:

```java
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

class Channel {
    int payload = 0;
    boolean ready = false;
    static final VarHandle READY;
    static {
        try {
            READY = MethodHandles.lookup()
                .findVarHandle(Channel.class, "ready", boolean.class);
        } catch (ReflectiveOperationException e) { throw new ExceptionInInitializerError(e); }
    }

    void producer() {
        payload = 42;                  // (1) ordinary write
        READY.setRelease(this, true);  // (2) release store: publishes (1)
    }

    int consumer() {
        while (!(boolean) READY.getAcquire(this)) { } // (3) acquire load: subscribes
        return payload;                                // guaranteed 42
    }
}
```

The crucial thing to see across all three: the *payload* is an ordinary, non-atomic variable. It is never touched atomically. Its correct, visible transfer is achieved entirely by the *ordering* attached to the single atomic flag. That is the leverage of acquire/release — one annotated atomic carries an arbitrary amount of ordinary data safely across the thread boundary, far cheaper than making every field atomic or wrapping the whole thing in a [mutex](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections). When you have a single, one-directional producer-to-consumer handoff, acquire/release is the precise tool, and it is strictly cheaper than seq-cst on weakly-ordered hardware.

#### Worked example: the handoff that breaks under relaxed

Take the exact code above and downgrade both the store and the load to `relaxed`. On x86 it will *appear* to work, because x86's hardware memory model (TSO — total store order) already forbids the store-store and load-load reorderings that would break it; the missing C++ ordering is "added back" for free by the strong CPU. This is the trap: it passes every test on your x86 dev box. Ship it to an ARM server or an Apple-silicon machine, whose memory model is *weak* and permits exactly those reorderings, and the consumer regularly reads `payload == 0` while `ready == true`. The flag arrived; the data it was supposed to guard did not. The bug is invisible on one architecture and a hard, intermittent corruption on another — which is why "it works on my machine" is uniquely dangerous in this domain, and why you reason about the *model*, not your local CPU.

There is a closely related tool worth naming, because it is what acquire/release "attach to an operation": the **standalone fence**. Instead of annotating a specific atomic, you can place a free-floating `std::atomic_thread_fence(std::memory_order_release)` *before* a relaxed store and `std::atomic_thread_fence(std::memory_order_acquire)` *after* a relaxed load, and get the same publish/subscribe ordering for *all* the memory around those points, not just the one atomic. The two formulations — ordering baked into the atomic versus a separate fence next to a relaxed atomic — are nearly equivalent, with one subtlety: a fence orders a *region*, so it can cover several relaxed operations at once, while the per-operation ordering binds tightly to one. Most of the time the per-operation form (`store(..., release)`) is clearer and what you want; the standalone fence earns its place when you have several relaxed writes to publish behind a single flag and want one fence rather than tagging each store. The full treatment of standalone fences, and how they differ from compiler barriers, lives in the sibling post on [memory barriers, acquire, release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences); the thing to carry here is that acquire/release on an atomic is just a *one-sided fence welded to that atomic*, which is why the two vocabularies describe the same hardware behavior. This is the same [happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) edge, established two equivalent ways.

## Seq-cst: the single total order, and why it is the default

Acquire/release ties one release to the acquires that read it — a *pairwise* relationship. There are cases where pairwise is not enough and you need a property about *all* synchronizing operations at once. The textbook case is two flags and two threads, each setting its own flag and then reading the other's, where the program is correct only if it is impossible for *both* threads to read the other's flag as not-yet-set. With only acquire/release, that "impossible" is *not* guaranteed — the two independent stores can appear in different orders to different observers, and both threads can race past. **Sequential consistency** rules it out by decree: all `seq_cst` operations across all threads are placed in *one single total order* that every thread observes consistently. There is one global timeline; if thread A's seq-cst store is before thread B's seq-cst store in that timeline, *everyone* sees it that way.

That single total order is *exactly* the extra cost. To create one global agreement, the hardware cannot let a seq-cst store sit quietly in a core's store buffer where other cores can't see it while the storing core races ahead; it must, at minimum, *drain* the store buffer (or otherwise force the store globally visible) before subsequent seq-cst operations. On x86 — already total-store-ordered — loads, acquire, and release come for free, but a seq-cst *store* still needs a fence (`mfence`) or must be implemented as a locked `xchg`, precisely to stop the store-buffer reordering that TSO otherwise allows on store-then-load. On a weak machine like ARM, *every* seq-cst operation needs a barrier. So seq-cst is the strongest guarantee and, correspondingly, the priciest — yet it is the *default* in C++ and Rust when you omit the ordering argument, and the de-facto model for Java `volatile` and the `Atomic*` classes.

Why make the *expensive* one the default? Because it is the only ordering that is *always correct* — it never surprises you, it composes, and it matches the simple intuition that operations happen in a single global order the way a single-threaded program would lead you to expect. The standards committees made the safe choice the default *on purpose*: the failure mode of "too strong" is a slower program, while the failure mode of "too weak" is a data-corruption Heisenbug that only shows on certain hardware under certain load. A slow correct program is a bug report you can act on; a fast corrupt one is a 3 AM page and a customer's lost money. So the discipline is: **write seq-cst by default, then weaken to acquire/release or relaxed only where you can prove the weaker ordering is still correct and you have measured that it matters.** Premature relaxation is the lock-free equivalent of premature optimization.

```rust
use std::sync::atomic::{AtomicBool, Ordering};

static X: AtomicBool = AtomicBool::new(false);
static Y: AtomicBool = AtomicBool::new(false);

// Dekker-style: with SeqCst, it is impossible for BOTH threads to see the other's flag false.
// With only Release/Acquire, that mutual exclusion can break. SeqCst is the single total order.
fn thread_a() -> bool { X.store(true, Ordering::SeqCst); Y.load(Ordering::SeqCst) }
fn thread_b() -> bool { Y.store(true, Ordering::SeqCst); X.load(Ordering::SeqCst) }
```

The one nuance worth knowing: seq-cst is *not* the same as "globally fenced everywhere." It guarantees a single total order *among seq-cst operations*, and it still provides acquire/release semantics for the ordinary memory around them. It does not, by itself, magically make relaxed operations on other variables join that total order. Mixing seq-cst and weaker operations on the *same* program has well-defined but genuinely subtle rules, and getting them wrong reintroduces the very bugs you were avoiding — which is the practical argument for not mixing unless you must. When in doubt, keep the whole synchronization protocol at one ordering level.

## The mechanism: why seq-cst needs a fence and relaxed does not

It is worth slowing down and deriving, from the hardware up, *why* a seq-cst store costs a fence and a relaxed store does not — because once you see the mechanism, the cost table stops being a list to memorize and becomes something you can re-derive at the whiteboard. The whole thing turns on one piece of silicon: the **store buffer**.

When a core executes a store, it does not wait for the value to reach the shared cache and become visible to other cores — that would cost tens of cycles and stall the pipeline. Instead the store goes into a small per-core FIFO called the store buffer, and the core moves on immediately. The value drains from the store buffer to the cache hierarchy *later*, asynchronously. The storing core sees its own stores immediately (it forwards from its own buffer), but *other* cores do not see a store until it drains. This single optimization is the source of almost all the surprising reordering in a strongly-ordered machine: on x86, the *only* reordering the hardware permits is precisely **store-then-load to different locations** — a later load can complete (reading from cache) *before* an earlier store has drained from the buffer, so to another core the load appears to have happened first. Loads are not reordered with loads, stores are not reordered with stores, and a load is never reordered before an earlier store *to the same location* (store forwarding handles that). The store buffer's store-load relaxation is the entire gap between x86 and true sequential consistency.

Now the derivation falls right out. A **release store** must guarantee only that earlier writes don't move *after* it — and on x86 stores already never reorder with stores, so a release store needs *no extra instruction*; it is a plain `mov`. An **acquire load** must guarantee that later reads don't move *before* it — and loads already never reorder with loads, so it too is a plain `mov`. But a **seq-cst store** must participate in a single global total order, which means a subsequent seq-cst *load* on the same core must not appear to execute before this store is globally visible. That is *exactly* the store-load reordering the store buffer allows. To forbid it, the core must drain the store buffer before the next seq-cst operation — and the instruction that does that is `mfence` (or, equivalently, implementing the seq-cst store as a `lock`-prefixed `xchg`, whose locked semantics also flush the buffer). So the fence is not arbitrary tax; it is the *minimum* work required to defeat the one reordering x86 still permits, and it is needed *only* on the seq-cst store, which is the only place store-load order becomes observable.

#### Worked example: the store-buffer reordering that seq-cst forbids

Here is the canonical interleaving, walked instruction by instruction. Two cores, two locations `x` and `y`, both starting at 0. Core A does `x = 1` then reads `y`; core B does `y = 1` then reads `x`. The question: can *both* reads return 0? Under sequential consistency the answer is no — at least one store must precede the other in the global order, so at least one read sees a 1. But with store buffers: core A puts `x = 1` in its buffer (not yet visible to B) and reads `y` from cache, getting 0; core B puts `y = 1` in its buffer (not yet visible to A) and reads `x` from cache, getting 0. *Both* reads return 0 — the buffered stores were invisible. This is real, observable behavior on x86 with relaxed (or even plain) operations. Marking all four operations `seq_cst` inserts the buffer-draining fence on the stores, so the store is globally visible before the load executes, and the "both zero" outcome becomes impossible. This little program — `r1 == 0 && r2 == 0` — is the litmus test that distinguishes sequential consistency from TSO, and running it in a loop on a real x86 box will produce the forbidden-under-SC outcome within milliseconds. That observable difference *is* the cost of seq-cst made visible.

The relaxed case is the mirror image of this derivation: relaxed asks for *no* ordering guarantee, so the compiler emits the plain atomic instruction and adds nothing — no fence, no barriered variant — because there is no reordering it has to prevent. The only thing relaxed still guarantees is per-location modification order (the coherence the cache protocol provides for free), which is why a relaxed counter composes correctly. Relaxed is "the atomic instruction and not one cycle more," which is exactly why it is the cheapest, and exactly why it carries no cross-location promise.

## A note on consume ordering (and why nobody uses it)

The C++ standard actually defines a *sixth* token, `memory_order_consume`, sitting between relaxed and acquire. The idea was elegant: many handoffs only need ordering for reads that are *data-dependent* on the value loaded (you load a pointer, then dereference *that* pointer), and on weakly-ordered hardware the CPU already respects data dependencies for free — so a "consume" load could be even cheaper than acquire on ARM/POWER, getting the dependency ordering without the barrier `ldar` needs. In practice, consume has been a failure: no mainstream compiler implements it as designed (they all silently promote it to `acquire`, paying the full barrier), because tracking dependency chains through arbitrary C++ code defeated every implementation attempt, and the standard has repeatedly discouraged its use. The practical advice is simple and worth stating so you don't reach for it: **treat the menu as five orderings — relaxed, acquire, release, acq-rel, seq-cst — and ignore consume.** If you genuinely need dependency-ordered loads in a hot RCU-style read path, the experts who do (the Linux kernel) hand-roll it with explicit dependency barriers rather than trusting `consume`. For everyone else, acquire is the floor for a real handoff. This is a good example of the broader truth that the *usable* memory model is smaller and simpler than the *specified* one.

## What each ordering costs on x86 versus ARM

This is the section that makes the choice concrete, because the cost of an ordering is *not* a single number — it depends entirely on the hardware memory model, and the two dominant models give opposite answers for several orderings. Two facts drive everything:

- **x86 is strongly ordered (TSO).** The hardware already guarantees that loads are not reordered with other loads, stores are not reordered with other stores, and loads are not reordered with *earlier* stores to *different* locations is the *only* relaxation it allows (store-then-load can be reordered, via the store buffer). So acquire and release are *already satisfied by the hardware*: an acquire load or release store compiles to a *plain* `mov`. The only ordering x86 has to *work* for is the seq-cst store, which must defeat the store-buffer's store-load reordering with an `mfence` or a locked `xchg`.
- **ARM (and POWER, RISC-V) is weakly ordered.** The hardware reorders almost everything by default for throughput. So *every* ordering above relaxed must emit an explicit barrier or use a barriered instruction: acquire is `ldar` (load-acquire), release is `stlr` (store-release), and seq-cst needs full `dmb ish` barriers. Relaxed is a plain `ldr`/`str`.

The figure lays this out as a cost matrix — and the punchline is that the *same* C++ source produces very different machine code, and very different performance, on the two architectures.

![A cost matrix mapping relaxed and acquire release and seq cst orderings to their machine code cost on x86 TSO versus weakly ordered ARM](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-4.png)

Concretely, here is what the orderings lower to (approximate; exact codegen varies by compiler and chip generation):

| Ordering | x86-64 codegen | ARMv8 codegen | Practical cost |
| --- | --- | --- | --- |
| relaxed load/store | plain `mov` | plain `ldr` / `str` | ~free; same as non-atomic on aligned word |
| acquire load | plain `mov` (free) | `ldar` | free on x86; one barriered load on ARM |
| release store | plain `mov` (free) | `stlr` | free on x86; one barriered store on ARM |
| seq-cst load | plain `mov` | `ldar` (or `dmb`) | free on x86; barrier on ARM |
| seq-cst store | `xchg` or `mov`+`mfence` | `stlr` + `dmb ish` | **a fence on both**, the priciest store |
| RMW (fetch-add, CAS) | `lock xadd` / `lock cmpxchg` | `ldxr`/`stxr` retry loop | dominated by the cache-line bounce, not the ordering |

Two consequences fall straight out of this table. First, on x86, *choosing acquire/release over seq-cst for loads and non-seq-cst stores buys you nothing* — they already compile to the same plain `mov`. The only place the choice shows up on x86 is the seq-cst store versus a release store: the seq-cst store pays for an `mfence`, the release store does not. So on x86 the realistic optimization is "avoid unnecessary seq-cst *stores*," not "use relaxed everywhere." Second, on ARM, the ordering choice matters *a lot* on every operation, because each level adds a real barrier — so relaxed-where-correct can be a meaningful win on weak hardware specifically.

The other dominating reality, easy to forget: for a *read-modify-write* under contention, the cost is dwarfed by the **cache-line bounce**, not by the memory ordering. When N cores hammer one atomic, the line ping-pongs between their caches (the MESI protocol forces exclusive ownership for each RMW), and that coherence traffic — tens to low hundreds of nanoseconds per contended operation — swamps the difference between a relaxed and a seq-cst fetch-add. So "relaxed is faster" is true in the *uncontended* or weak-hardware case and largely *irrelevant* when a counter is so hot that the line never sits still. Which is exactly why you measure before you relax: the win you imagined may be hiding behind a coherence wall that no ordering choice can move. (This is the same coherence story the [memory hierarchy post in the HPC series](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) tells from the cache side.)

## The Java memory model: volatile, VarHandle, and the Atomic classes

Java got a rigorous memory model (JSR-133) years before C++ did, and its vocabulary differs, so it is worth a clean mapping. Java does not hand you a `memory_order` argument on every operation; instead it offers a small number of *levels* through three mechanisms.

**`volatile`** is the oldest and bluntest tool. A `volatile` field has, effectively, **sequentially-consistent** semantics for accesses to that field: a `volatile` write is a release-store *and* participates in the total order of volatile/synchronized accesses; a `volatile` read is an acquire-load. So a `volatile boolean ready` flag carries a correct handoff (the write publishes, the read subscribes) — which is exactly why the classic "set a `volatile` flag, spin on it in another thread" idiom *works* in Java while a plain `boolean` flag does not. The catch is that `volatile` is the *strong* level: you cannot ask for "just release" on a `volatile` write the way you can in C++. If you want the cheaper acquire/release without the full seq-cst story, you need `VarHandle`.

**`VarHandle`** (Java 9+) is the modern, fine-grained interface, and it maps almost one-to-one onto the C++ orderings via *access modes*:

- `getPlain` / `setPlain` — no ordering, like C++ relaxed *without even atomicity guarantees* on its own (plain access).
- `getOpaque` / `setOpaque` — atomic and coherent per-location, closest to C++ `relaxed`.
- `getAcquire` / `setRelease` — exactly C++ acquire and release.
- `getVolatile` / `setVolatile` — seq-cst, the same as a `volatile` field access.
- `compareAndSet`, `getAndAdd`, `compareAndExchangeAcquire`, … — the RMW family, with mode-specific variants.

**The `Atomic*` classes** (`AtomicInteger`, `AtomicLong`, `AtomicReference`, …) are the ergonomic front door. Their plain methods — `get()`, `set()`, `incrementAndGet()`, `getAndAdd()`, `compareAndSet()` — use the *volatile / seq-cst* level by default, matching the safe-default philosophy. Since Java 9 they *also* expose the weaker modes (`getAcquire`, `setRelease`, `getPlain`, `weakCompareAndSetPlain`, …) for when you've proven you can relax. So the Java story is the same arc as C++/Rust: a safe seq-cst default (`volatile`, the plain `Atomic*` methods), with explicit weaker modes available through `VarHandle` once you can justify them. The figure puts the two worlds side by side.

![A matrix mapping plain and acquire release and seq cst and read modify write atomic categories to their spellings in C plus plus and Rust versus Java](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-7.png)

```java
import java.util.concurrent.atomic.AtomicInteger;

class Stats {
    final AtomicInteger hits = new AtomicInteger();

    void record() {
        // Default Atomic* ops are seq-cst. For a pure independent counter you can
        // weaken to opaque via a VarHandle if profiling shows it matters; usually it doesn't.
        hits.getAndIncrement();
    }

    int read() { return hits.get(); } // seq-cst read; a monotonic, approximate snapshot
}
```

One important Java-specific footnote that connects to this series' running theme of *torn reads*: in older Java, a plain (non-`volatile`) `long` or `double` field could be read *torn* — split into two 32-bit halves on a 32-bit JVM, so a reader could see the high half of one write and the low half of another. Declaring the field `volatile` (or using `AtomicLong`) closed that. This is the JVM's version of the atomicity-vs-ordering split: `volatile` bought you *both* atomicity (no tearing) *and* ordering (seq-cst) in one keyword, which is convenient but also why so much Java code reaches for `volatile` reflexively without separating the two concerns. (For the deeper torn-read mechanics, see [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads).)

Go is worth a brief mention as the *opposite* design philosophy, because it clarifies what a deliberately minimal interface looks like. Go's `sync/atomic` package (and the typed `atomic.Int64`, `atomic.Pointer[T]` wrappers added in Go 1.19) exposes *only* sequentially-consistent operations — there is no relaxed, no acquire/release knob at all. The language designers decided that the safe default should be the *only* option for the vast majority of code, and that anyone who truly needs weaker ordering should drop to assembly or rethink the design. Go's own guidance is even blunter: "don't communicate by sharing memory; share memory by communicating" — i.e., prefer a channel to an atomic whenever you can, and reach for `sync/atomic` only for the small counter-and-flag cases where a channel would be overkill. This is a legitimate and increasingly popular stance: the full memory-ordering menu is a power tool that most application code never needs, and a language that hides it trades a little peak performance for a lot of avoided footguns. C++ and Rust expose the whole menu because they target the layer *below* — the runtimes, allocators, and lock-free libraries that other languages are built *on* — where the last few nanoseconds genuinely matter and the authors are expected to prove their orderings.

```go
import "sync/atomic"

var requests atomic.Uint64

func onRequest() {
	requests.Add(1) // seq-cst; Go offers no weaker ordering, by design
}

func snapshot() uint64 {
	return requests.Load() // seq-cst load; an approximate, monotonic total
}
```

## Choosing the ordering: a decision guide

Put the mechanism aside for a moment and reduce the choice to a procedure you can run at the keyboard. The first and only question that matters: **does another thread's correctness depend on the order of your memory operations, or only on the atomic's own indivisibility?** Everything flows from that.

![A decision tree for choosing a memory ordering branching on whether order matters into acquire release and seq cst or into a relaxed counter and a re-checked relaxed flag](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-5.png)

Walk it concretely:

1. **Is this a pure, independent counter or statistic read only at the end (or sampled approximately)?** → `relaxed`. Nothing depends on ordering; you want the cheapest atomic.
2. **Is this a flag you poll cheaply but always re-validate under a stronger fence or lock before acting?** → `relaxed` for the poll, real synchronization at the re-check. The relaxed load is a hint.
3. **Are you publishing data for exactly one consumer side to pick up (producer→consumer, or building a node then linking it in)?** → `release` on the publishing store, `acquire` on the consuming load. This is the handoff; it is the most common correct use of the weaker orderings.
4. **Is it a read-modify-write that both consumes a prior publish and publishes its own result (a CAS in a lock-free push, a lock handoff)?** → `acq_rel`.
5. **Do multiple threads each store-then-load different locations and the protocol's correctness needs a single global order (Dekker-style, or any case where you reasoned "it must be impossible for both to go first")?** → `seq_cst`. This is also your answer whenever you are *unsure* — when you cannot cleanly articulate which two operations are the pair, you do not yet understand the protocol well enough to weaken it.
6. **Default, when in any doubt at all?** → `seq_cst`. It is never *wrong*; it is only sometimes *slower*. Weaken later, with a measurement.

| Situation | Ordering | Why |
| --- | --- | --- |
| Hit counter, metrics, independent stats | relaxed | only atomicity needed; no cross-location order |
| Cheap-poll flag you re-check under a lock | relaxed (poll) | the load is a hint; real sync at re-check |
| Producer publishes data, consumer reads it | release / acquire | the directed handoff; pairs one store to its loads |
| CAS that consumes then publishes | acq-rel | both ends of the handoff in one RMW |
| Two-flag mutual exclusion, "both can't go first" | seq-cst | needs the single global total order |
| You cannot name the exact pair | seq-cst | if you can't prove the weaker one, don't use it |

The discipline this table encodes is the whole engineering judgment of lock-free code: *start strong, weaken deliberately, prove each weakening, measure that it paid.* A relaxed ordering you cannot justify in one sentence ("these increments are independent and read only after join") is a latent bug, not an optimization.

A practical refinement of the counter case shows the discipline at its best. If a single relaxed counter is so hot that the cache line bounces between cores and throughput collapses, the answer is *not* a stronger or weaker ordering — it is to remove the sharing. Give each thread (or each core) its *own* counter, padded onto its own cache line so they never contend, have each thread relaxed-add to *its* counter, and sum the per-thread counters at the end. Now there is no contention, the relaxed ordering is trivially correct (each counter is touched by one writer and read only at the end), and throughput scales linearly with cores. This is the lesson that the ordering knob and the data layout are *different* tools for *different* problems: the ordering controls visibility, the layout controls contention, and the contention is almost always the bigger lever. A staff engineer reaches for the layout fix first and treats the ordering choice as the small, final tuning step it usually is.

## Worked examples: the counter, the handoff, and the bug

We have all the pieces; now run the two canonical programs end to end and then deliberately break the third.

#### Worked example: a correct lock-free counter with relaxed

The requirement: count events across many threads with no lost updates, read the total once at the end. This is the *independent counter*, so relaxed is both correct and optimal.

```cpp
// lock-free relaxed counter, C++
#include <atomic>
#include <thread>
#include <vector>
#include <cstdint>

std::atomic<uint64_t> events{0};

void worker(int iters) {
    for (int i = 0; i < iters; ++i)
        events.fetch_add(1, std::memory_order_relaxed); // independent: relaxed is right
}

uint64_t run(int threads, int iters) {
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) ts.emplace_back(worker, iters);
    for (auto& t : ts) t.join();          // join establishes happens-before for the read
    return events.load(std::memory_order_relaxed); // exact total: threads * iters
}
```

Why this is correct *without* any acquire/release: no thread reads `events` for a decision while others are still writing; the only read is *after* `join()`, and `join()` itself is a synchronization point that establishes happens-before. So there is no cross-thread *ordering* requirement during the loop — only the per-operation atomicity that relaxed already provides. The result is always `threads * iters`. The same in Rust is a one-line change of `Ordering::Relaxed` on the `fetch_add`; in Java, `events.getAndIncrement()` on an `AtomicLong` is correct but uses the seq-cst level — and on x86 even that costs you nothing extra for the load, only the locked RMW you needed anyway.

#### Worked example: a correct flag handoff with release/acquire

The requirement: one thread fills a buffer and a sequence number, then signals; another thread waits for the signal and reads the buffer. This is the *directed handoff*, so release/acquire is the precise tool. We will use a tiny seqlock-flavored publish — write the data, then release-store a version that the reader acquire-loads — to show the pattern at its cleanest.

```cpp
// seqlock-style publish with release/acquire, C++
#include <atomic>
#include <cstring>

struct Frame { uint32_t len; char data[256]; };

Frame frame;                       // ordinary, non-atomic payload
std::atomic<uint32_t> version{0};  // even = stable, odd = being written

void publish(const char* src, uint32_t n) {
    version.fetch_add(1, std::memory_order_relaxed);      // make it odd: "writing"
    std::atomic_thread_fence(std::memory_order_release);  // fence: writes below stay below... (see note)
    frame.len = n;
    std::memcpy(frame.data, src, n);
    version.fetch_add(1, std::memory_order_release);      // make it even AND publish the data
}

bool try_read(char* dst, uint32_t* out_len) {
    uint32_t v1 = version.load(std::memory_order_acquire);   // subscribe
    if (v1 & 1u) return false;                               // writer in progress, retry
    uint32_t n = frame.len;
    std::memcpy(dst, frame.data, n);
    uint32_t v2 = version.load(std::memory_order_acquire);   // re-check version
    if (v1 != v2) return false;                              // changed under us, retry
    *out_len = n;
    return true;
}
```

The release on the second `fetch_add` is what makes the `frame` writes visible to a reader whose acquire load sees the new version; the acquire loads on the reader are what stop the `frame` reads from floating above the version check. The seqlock's even/odd dance handles the writer-during-read case (the reader retries), which is why a seqlock is a great fit for a single writer and many readers of a small, frequently-updated record. The simpler bare handoff — one boolean, one writer, one reader — is the `producer`/`consumer` pair from earlier; this seqlock is the same release/acquire idea hardened for repeated publication. Either way, the orderings are doing the load-bearing work, not the atomicity.

#### Worked example: relaxed misused as a lock handoff (the bug, then the fix)

Now the bug that pays everyone's tuition. A developer builds a lock-free single-producer queue: allocate a node, fill its fields, then publish the node pointer into a shared slot with a `relaxed` store, because "the store is atomic, so it's fine." The consumer relaxed-loads the pointer and, when non-null, reads the node's fields.

```cpp
// relaxed-publish BUG, C++
#include <atomic>

struct Node { int x; int y; };
std::atomic<Node*> head{nullptr};

void produce() {              // BUG VERSION
    Node* n = new Node();
    n->x = 1;
    n->y = 2;
    head.store(n, std::memory_order_relaxed);   // BUG: publishes the pointer with no release
}

Node* consume() {            // BUG VERSION
    Node* n = head.load(std::memory_order_relaxed); // BUG: no acquire
    return n;  // caller reads n->x, n->y — may be uninitialized garbage on weak hardware
}
```

The pointer store is indeed atomic — the consumer never sees a torn pointer. But there is *no ordering* tying the `n->x`/`n->y` writes to the pointer publication. On a weak machine, the consumer can observe the *new pointer* while the *node's fields are still the old uninitialized memory* (or still sitting in the producer's store buffer, invisible to the consumer's core). It reads `n->x` and gets garbage. On x86 it passes every test; on ARM it corrupts intermittently. The figure shows the exact failure shape — a non-null pointer to a torn, half-built node — next to the one-word fix.

![A before and after comparison showing relaxed used for a pointer handoff producing a torn node versus release and acquire delivering a fully built node](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-8.png)

The fix changes two words — `relaxed` → `release` on the publish, `relaxed` → `acquire` on the consume — and nothing else:

```cpp
void produce() {              // FIXED
    Node* n = new Node();
    n->x = 1;
    n->y = 2;
    head.store(n, std::memory_order_release);   // publish: the field writes go with it
}

Node* consume() {            // FIXED
    Node* n = head.load(std::memory_order_acquire); // subscribe: fields are visible when n != null
    return n;  // n->x == 1, n->y == 2, guaranteed
}
```

```rust
use std::sync::atomic::{AtomicPtr, Ordering};

// FIXED Rust: release on publish, acquire on consume.
fn produce(head: &AtomicPtr<Node>, n: *mut Node) {
    head.store(n, Ordering::Release);   // publishes the node's fields
}
fn consume(head: &AtomicPtr<Node>) -> *mut Node {
    head.load(Ordering::Acquire)        // node is fully built when non-null
}
```

This is the single most common real-world atomics bug: relaxed used to *publish* something. The rule that prevents it is mechanical — **any time an atomic carries a pointer or flag whose meaning is "the data behind me is ready," the store is `release` and the load is `acquire`, never relaxed.** Relaxed publishes nothing; it is only for values that *mean nothing but themselves*.

## Measured: throughput by ordering

Now the honest numbers — with the caveats stated first, because measuring atomics badly produces lies. The ordering's *isolated* cost is tiny and easily buried under (a) cache-coherence traffic when the line is contended, (b) the OS scheduler's noise, and (c) the compiler's freedom to hoist a relaxed load out of a loop entirely. So: warm up, pin threads where you can, run many iterations, report a distribution not a single number, and *name the architecture* — because, as established, x86 and ARM disagree. The figures below are representative orders of magnitude from a 4-core x86-64 laptop and an 8-core ARM (Apple-silicon-class) machine; treat them as "what shape to expect," not as benchmark-grade absolutes, and re-measure on *your* hardware before acting.

The first and most important result, the relaxed-versus-seq-cst counter, is shown as a before/after: same exact total, different throughput — and the gap depends heavily on contention and architecture.

![A before and after comparison of a relaxed fetch-add counter that runs fast versus a needlessly seq-cst counter that is correct but slower](/imgs/blogs/atomics-and-memory-orderings-from-relaxed-to-seq-cst-6.png)

| Scenario (10M increments/thread) | relaxed | seq-cst | Notes |
| --- | --- | --- | --- |
| 1 thread, x86 | ~baseline | ~baseline | uncontended; both are a locked RMW, near-identical |
| 8 threads, x86, one shared counter | line-bounce bound | line-bounce bound | coherence dominates; ordering barely matters |
| 8 threads, x86, per-thread counter | ~1× | ~1× | no contention, no fence on relaxed; tiny edge to relaxed |
| 1 thread, ARM | ~1× | ~1.3–2× slower | seq-cst emits real barriers ARM has to honor |
| 8 threads, ARM, shared counter | line-bounce bound | line-bounce + barriers | both bad; relaxed somewhat less bad |
| handoff flag (release/acq vs seq-cst), ARM | ~1× | ~1.2–1.5× | seq-cst store's barrier on the publish path |

Read the table for its *shape*, which is the actionable part:

- On **x86, uncontended**, relaxed and seq-cst are nearly identical for loads and RMWs — the win from relaxing is small to nil, because the orderings already compile to the same instructions except for the seq-cst *store*. Relaxing here is mostly not worth the risk.
- On **any architecture, heavily contended**, the cache-line bounce dominates and the ordering choice is in the noise. The fix for a hot contended counter is *not* relaxed — it is *sharding* the counter into per-thread or per-core counters that you sum at the end (which then trivially use relaxed because they're independent). Architecture beats ordering.
- On **ARM, uncontended**, relaxing genuinely helps, because seq-cst there means real barriers on every operation that the weak hardware must execute. This is the one place "use relaxed where correct" buys measurable throughput on a single-threaded-ish hot path.

The meta-lesson, stated bluntly because it is where engineers waste days: **the ordering is rarely your bottleneck; the contention and the cache line are.** Before you spend an afternoon proving a relaxed ordering is safe to shave a few nanoseconds, measure whether the atomic is even hot, and whether the real fix is to *not share the line at all* (per-thread sharding, padding to avoid [false sharing](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity), or a different data structure). The cheapest atomic is often the one you removed.

#### Worked example: how to measure an ordering honestly

Suppose you want to actually answer "does relaxed beat seq-cst for my counter?" Here is the protocol that produces a number you can trust, and the traps that produce numbers you can't. First, **warm up**: run the loop for a few hundred milliseconds and discard the result, so the branch predictor, the caches, and (on a JIT like the JVM) the compiler have all stabilized — a cold first run can be 10× slower and tells you nothing. Second, **prevent the optimizer from deleting your work**: a relaxed load in a loop whose result you ignore can be hoisted out entirely by the compiler, so your "relaxed is infinitely fast" measurement is really "the compiler deleted the benchmark." Force a dependency (accumulate the loaded values into a `volatile` sink, or use a benchmark framework's `DoNotOptimize`). Third, **run many trials and report a distribution** — the median and the spread, not a single best-of-one — because the OS scheduler will preempt your threads at random and a single run can be an outlier by 2×. Fourth, **pin threads to cores** where the platform allows, so you are measuring the atomic and not the scheduler shuffling threads across a busy machine. Fifth, and most important for *this* topic, **name and vary the architecture**: run the *same binary's logic* on x86 and on ARM, because the entire point is that the answer differs, and a result from only one architecture is half an answer. If you do all five and the relaxed-vs-seq-cst gap is within the run-to-run spread, the honest conclusion is "no measurable difference here" — and that is a perfectly good, common result that should stop you from relaxing.

A concrete trap from real code: a team "proved" relaxed gave a 3× speedup, shipped it, and found no production improvement. The microbenchmark had a single thread spinning on one counter with the result optimized into a dead loop; the 3× was the compiler's hoisting, not the ordering. In production the counter was contended across cores, the line bounced, and coherence traffic — identical for relaxed and seq-cst — set the throughput. The relaxation was correct but worthless, and the weeks spent proving it safe were a loss. Measure the *real* workload's contention before you spend rigor on the ordering.

## Case studies / real-world

**The Linux kernel and `READ_ONCE`/`WRITE_ONCE` plus `smp_load_acquire`/`smp_store_release`.** The kernel is one of the most disciplined memory-ordering codebases on earth, and it does *not* default to the strongest barrier — it pervasively uses acquire/release (and even relaxed via `READ_ONCE`/`WRITE_ONCE`) precisely because it runs on weak architectures (ARM, POWER) where seq-cst everywhere would be a measurable throughput tax. Its RCU (read-copy-update) subsystem is the canonical large-scale release/acquire handoff: writers publish a new version with a release store, readers subscribe with an acquire load, and the ordering is what makes the lock-free read path correct. The kernel's `Documentation/memory-barriers.txt` is, fairly, the single best long-form treatment of why each ordering exists. The lesson: at the level of a kernel, relaxing *is* worth the rigor — but they pair every weak ordering with an explicit, reviewed justification.

**C++ `std::shared_ptr` reference counting uses relaxed for the increment.** A widely-cited real use of `relaxed` in a shipping standard library: incrementing a `shared_ptr`'s strong reference count can be `relaxed`, because gaining a new owner does not require ordering any *other* memory — you already hold a valid reference, so the count just needs to not be lost. The *decrement*, however, must be `acq_rel` (or release then an acquire fence), because the thread that drops the count to zero must see all prior uses of the object before it runs the destructor. This is the textbook split: relaxed where only the count's atomicity matters, a stronger ordering exactly at the point where visibility of *other* memory (the object's state before deletion) becomes load-bearing. Boost's and libstdc++'s implementations document this reasoning in comments.

**The classic "volatile flag didn't work" Java bug (pre-JSR-133 and the broken double-checked locking).** Before Java's memory model was fixed in 2004 (JSR-133), the famous *double-checked locking* idiom for lazy singletons was subtly broken: a thread could publish a reference to a partially-constructed object because the constructor's writes were not ordered before the reference store. The fix that the new memory model enabled was making the field `volatile` — which gives the reference store release semantics and the read acquire semantics, so a reader that sees the non-null reference also sees the fully-constructed object. This is *exactly* the relaxed-publish bug from this post, in Java's clothing: a pointer published without release lets a consumer see a torn, half-built object. The episode is why "is double-checked locking safe?" was a years-long debate and why the answer is "only with `volatile` (or a holder class), because you need the release/acquire ordering." It is the same lesson, paid for at industry scale.

**The LMAX Disruptor and the deliberate use of memory ordering for throughput.** The Disruptor is a high-performance inter-thread messaging library that famously processes millions of messages per second on a single thread by being extremely careful about exactly the things this post covers: it uses a ring buffer with a published *sequence* number as the handoff carrier, with release semantics on the producer's sequence update and acquire semantics on the consumer's read — the canonical release/acquire publish/subscribe at the core of a hot path. Crucially, the Disruptor's authors pad the sequence counters to their own cache lines to eliminate false sharing, which is the *contention beats ordering* lesson from this post embodied in a shipping system: they got their throughput primarily from cache-line discipline and a lock-free ring, with memory ordering as the precise, minimal correctness layer on top — not by reaching for the strongest barrier everywhere, and not by recklessly relaxing. It is a clean demonstration that the right answer is usually "the right *layout* plus the *minimal* ordering," in that priority order.

## When to reach for this (and when not to)

Atomics with explicit orderings are a *sharp* tool. The decisive guidance, stated as plainly as the kit demands:

**Reach for atomics when** you have a small, hot piece of shared state — a counter, a flag, a single pointer, a sequence number — and a [mutex](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) is genuinely your measured bottleneck or is overkill for a one-word update. A `relaxed` counter, a `release`/`acquire` flag, or a CAS loop is the right reach when the shared state is *small and the critical section would be trivially short*. Atomics are also the foundation when you are *building* a lock-free structure (see [the progress hierarchy](/blog/software-development/concurrency/the-progress-hierarchy-blocking-lock-free-and-wait-free) and [compare-and-swap and lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures)) — there, atomics with carefully chosen orderings are not an optimization, they are the only way to make progress without a lock.

**Do not reach for relaxed** unless you can state, in one sentence, *why* no other thread's correctness depends on the order of your writes — and you have *measured* that the relaxation matters on your actual hardware. "Relaxed is faster" is true in a microbenchmark and usually irrelevant in a real program where the cache line is contended (coherence dominates) or the line is cold (the atomic isn't hot). The default is `seq_cst`; weaken only with a proof and a number. If you cannot name the exact release/acquire *pair*, you do not understand the protocol well enough to weaken it.

**Do not reach for atomics at all** when the shared state is more than a word or two, when the operation needs to update several locations as a unit (that's a critical section — use a [mutex](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections)), when you need to *wait* for a condition (use a [condition variable](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly), not a relaxed spin), or when message-passing models the problem more naturally (a channel makes the handoff explicit and the ordering automatic). Lock-free code built on hand-rolled orderings is some of the hardest code to get right and to *review*; if a mutex isn't your bottleneck, a mutex is the correct, boring, fast-enough answer. And for Python specifically, the orderings discussed here are largely moot under the GIL — reach for the [Python concurrency story](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) instead of reasoning about C++-style memory orderings that the interpreter abstracts away.

The capstone [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) places atomics in the full menu of mechanisms; the one-line summary it carries is the one to leave with: *atomics are the cheapest correct mechanism for the smallest shared state, and the most dangerous one to get subtly wrong.*

## Key takeaways

- **Atomicity is not ordering.** An atomic operation only guarantees no torn or lost value. It says nothing about whether your *other* writes are visible or in what order — that is what memory orderings control, and conflating the two is the root of most lock-free bugs.
- **Relaxed is atomic only.** Use it exclusively where no other thread's correctness depends on the order of your operations — independent counters, statistics, a hint flag you re-check. It is the cheapest, and the easiest to misuse.
- **Acquire/release is a directed handoff.** Release on the store *publishes* everything before it; acquire on the matching load *subscribes*. Always design the pair; a release with no acquirer (or vice versa) is wasted.
- **Seq-cst is the single total order and the safe default.** It is the only ordering that never surprises you, so it is the default in C++/Rust and the model for Java `volatile`/`Atomic*`. Start here; weaken only with a proof and a measurement.
- **Cost is architecture-dependent.** On x86, acquire/release are free (plain `mov`) and only the seq-cst *store* pays a fence. On weak ARM, every ordering above relaxed emits a real barrier. The same source has different performance on different hardware.
- **Contention beats ordering.** For a hot shared atomic, the cache-line bounce dominates the ordering cost. The real fix for a contended counter is sharding (per-thread counters summed at the end), not relaxing the ordering.
- **The canonical bug is relaxed-as-a-publish.** Relaxed used to publish a pointer or "ready" flag lets a consumer read a half-built object on weak hardware. The fix is two words: `release` on the publish, `acquire` on the consume.
- **Java's `volatile` bundles atomicity and seq-cst ordering**; `VarHandle` access modes give you the finer acquire/release/opaque levels when you've proven you can relax, matching the C++/Rust menu.

## Further reading

- **Within this series** — [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) (the frame), [memory models, sequential consistency, and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) (the model these orderings are an interface to), [memory barriers, acquire, release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences) (the standalone fences), [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads), [compare-and-swap and building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures), and the capstone [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- **Hans Boehm & Sarita Adve, "Foundations of the C++ Concurrency Memory Model"** (PLDI 2008) — the paper behind `std::memory_order`; the authoritative explanation of why the orderings are exactly these.
- **Jeremy Manson, William Pugh & Sarita Adve, "The Java Memory Model"** (POPL 2005, JSR-133) — the rigorous Java model; why `volatile` works and double-checked locking needed fixing.
- **Anthony Williams, *C++ Concurrency in Action* (2nd ed.)** — chapter 5 on the memory model and atomics is the clearest book-length treatment of acquire/release/seq-cst with runnable code.
- **Jeff Preshing's blog** ("Acquire and Release Semantics," "An Introduction to Lock-Free Programming," "Weak vs. Strong Memory Models") — the best free, intuition-first writing on memory ordering anywhere.
- **The Linux kernel `Documentation/memory-barriers.txt`** and Paul McKenney's *Is Parallel Programming Hard, And, If So, What Can You Do About It?* — production-grade memory ordering, with the kernel's relaxed/acquire/release discipline and RCU.
- **Maurice Herlihy & Nir Shavit, *The Art of Multiprocessor Programming*** — the formal foundations of atomic objects, linearizability, and why these guarantees are the building blocks of lock-free structures.

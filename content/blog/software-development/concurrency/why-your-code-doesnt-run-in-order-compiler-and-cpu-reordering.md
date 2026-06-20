---
title: "Why Your Code Doesn't Run in Order: Compiler and CPU Reordering"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The order you wrote is not the order that runs — how the compiler and CPU legally reorder your memory operations, and why a hand-rolled flag silently breaks."
tags:
  [
    "concurrency",
    "parallelism",
    "memory-model",
    "reordering",
    "store-buffer",
    "out-of-order",
    "compiler",
    "cpu",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-1.png"
---

Here is a program that should work, and does not. A worker thread computes a result, stores it, and sets a flag. A consumer thread spins until the flag is set, then reads the result. Two stores, two reads, no shared counter, no `++` to lose an update — about as simple as inter-thread communication gets.

```c
// Shared globals, both threads see them.
int  data = 0;
bool flag = false;

// Producer thread:
void producer(void) {
    data = 42;       // (1) write the payload
    flag = true;     // (2) announce it is ready
}

// Consumer thread:
void consumer(void) {
    while (!flag) { /* spin */ }   // (3) wait for the announcement
    printf("%d\n", data);          // (4) read the payload
}
```

You wrote `data = 42` *before* `flag = true`, so surely by the time the consumer sees `flag` it must also see `data == 42`. And surely the `while (!flag)` loop will eventually notice the producer flipping the flag. Both of those "surely"s are wrong. On a real multi-core machine, with a real optimizing compiler, this program has two independent bugs. The consumer can read `flag == true` and then print `0` — it observed the second store before the first. And depending on how the compiler treats `flag`, the consumer's loop can spin *forever* even after the producer sets the flag, because the compiler decided to read `flag` once, cache it in a register, and never look at memory again.

Neither of these is a fluke or a "one in a billion" timing accident you can shrug off. They are the *specified, allowed* behavior of the languages and hardware you use every day. The order in which your statements appear in source is a fiction maintained for exactly one observer: a single thread reading its own variables. The moment a *second* thread reads what the first one wrote, that fiction collapses. The figure below shows the shape of the first bug — the source order you wrote on the left, and an order another core is allowed to observe on the right.

![source order data then flag versus an observed order where flag is visible first and data is still stale](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-1.png)

This post is the shock that opens the memory-model track of the series. By the end you will be able to: explain the *as-if rule* that licenses the compiler to rewrite your code; name the four reorderings hardware can perform and which ones x86 and ARM actually do; read the generated assembly that proves a `flag` got hoisted into a register; trace the store-buffer mechanism that lets a core see its own write before its neighbors; derive why the famous "both threads read 0" litmus test is *impossible* on paper yet routine on real silicon; and recognize the exact shape of a broken hand-rolled publish so you reach for the right tool — an atomic, a barrier — instead of hoping. This is the *why* behind everything that comes next: [memory models and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) (D2), and [memory barriers, acquire/release and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences) (D3, D4). If you have not yet met the core hazard of this whole series — [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) — start there; this post is the hardware floor underneath it.

## The as-if rule: single-thread correctness is the only promise

Start with the contract, because every surprise in this post is a *correct* application of it. When you compile C, C++, Rust, Java, Go, or any other optimizing language, the toolchain does not promise to execute your statements in the order you wrote them, or even to execute all of them. It promises something much narrower, called the **as-if rule** (the C and C++ standards call it the "observable behavior" rule; it is the same idea everywhere). Informally:

> The compiler may transform the program in any way it likes, as long as a *single thread executing in isolation* cannot tell the difference by its observable behavior — its I/O, its volatile accesses, and its final visible values.

That phrase, "a single thread executing in isolation," is the whole story. The compiler reasons about your function as if it were the only thing running. Within that frame, reordering two independent stores is invisible: a single thread that writes `data` then `flag` ends up with both written, and nothing in *that thread* can observe which physically happened first. So the compiler is free to swap them, fuse them, delete one if it proves the value is never read, or keep a value in a register instead of re-reading memory. All of these preserve single-thread semantics. **None of them preserve multi-thread semantics**, because the as-if rule was never asked to.

This is not a bug in the compiler or a conservative reading of the standard. It is the *point* of an optimizing compiler. Hoisting a loop-invariant load out of a loop, keeping a hot variable in a register, eliminating a dead store, reordering memory accesses to hide latency — these are exactly the transformations that make the difference between code that runs at memory-bus speed and code that runs at register speed. The compiler is doing its job. The trouble is that "single-thread correct" and "multi-thread correct" are different properties, and a data race — two threads accessing the same location, at least one of them writing, with no synchronization between them — is precisely the situation the as-if rule does not cover.

Here is the formal teeth of it. In C and C++, a program with a **data race has undefined behavior**. Not "implementation-defined," not "unspecified" — *undefined*. The standard does not say the consumer prints `0`; it says the standard imposes *no requirements at all* on what happens. The reason is exactly the as-if rule: the optimizer assumes there is no race, transforms accordingly, and if you violated that assumption the resulting program has no defined meaning. Java is gentler — the Java Memory Model gives even racy programs bounded, non-"undefined" semantics (you cannot get a value out of thin air, and you cannot crash the JVM) — but a racy Java program is still allowed to print `0` here, because the JMM also permits the two writes to become visible out of order. Go's memory model is similar in spirit: it defines a happens-before relation and says that if you do not establish one, "there is no guarantee" about what a read observes.

It is worth being precise about *why* "undefined" and not merely "you get a stale value." Undefined behavior is not a euphemism for "an unlucky outcome" — it is a license the optimizer actively *uses*. Consider what happens when a modern compiler proves, from the absence of any synchronization, that no other thread can be touching `flag` between two of your reads of it. It is entitled to assume the value is stable, fold both reads into one, and propagate that single value forward through arbitrary downstream code — including across a branch, into a `printf` format string, into a pointer dereference. A data race does not just give you a wrong number; it can make the compiler delete a bounds check it "proved" was unreachable, or hoist a load past a `null` test it assumed could not fail. The reason every serious memory model treats a race as catastrophic rather than merely lossy is that the optimizer's reasoning is *global*: one unsynchronized access poisons every optimization that depended on the no-race assumption, and those optimizations can be arbitrarily far from the access in the source. This is why "it printed the old value once" is the *best* case of a data race, not the typical one.

The flip side of that severity is a precise promise you *can* rely on, and it is the load-bearing sentence of the entire C/C++ standard for concurrency: **if your program contains no data races, it behaves as if it were sequentially consistent** (modulo the explicit weak orderings you opt into with `memory_order_relaxed` and friends). This is the "SC for data-race-free programs," or *SC-DRF*, guarantee, and it is the deal you sign. You promise to put a happens-before edge between every conflicting pair of accesses; in exchange, the compiler and hardware promise that the visible result is *as if* every operation happened in one global order. Lose the promise on your side and you lose the guarantee on theirs — completely, not proportionally. Every fix in this post is, at bottom, the act of *restoring* a happens-before edge so SC-DRF applies again.

#### Worked example: why "data ready before flag" is not implied

Walk the producer as a single isolated thread. It has two effects: location `data` ends at `42`, location `flag` ends at `true`. Ask the as-if question: can this thread, by any observable behavior of its own, distinguish "wrote `data` first" from "wrote `flag` first"? It cannot — it never reads either variable after writing, and it does no I/O between the writes. The two orderings are observationally identical *to the producer*. Therefore the compiler may emit them in either order, and the hardware may make them globally visible in either order, and both are conforming. The intent "data is ready before the flag" lives only in your head; you never encoded it as a constraint the toolchain is required to honor. To encode it, you need a *happens-before edge* between the flag write and the flag read — which is precisely what an atomic with release/acquire semantics, or a lock, or a barrier, gives you. That edge is the subject of D2 and D3; here we are establishing why, without it, nothing holds.

Make the failure concrete with a value trace. Tag each operation with the moment it becomes *globally visible* (the moment another core could observe it), and call those moments `g1` and `g2`. The source says `data = 42` is `g1` and `flag = true` is `g2` with `g1 < g2`. But nothing the producer can observe pins `g1 < g2`; the only constraint is that *both* eventually happen. So a legal global order is `g2 < g1`: the flag-write lands first. Now suppose the consumer's flag-read and data-read fall in the window between them — flag-read at `g2 + ε` (sees `true`), data-read at `g2 + 2ε` (still before `g1`, sees the stale `0`). The consumer prints `0`. Tabulate the four conceivable global orders and the consumer outcome each permits:

| Producer global order | Possible under as-if? | Consumer that sees flag=true reads data = |
| --- | --- | --- |
| data@g1 then flag@g2 (source order) | yes | always 42 |
| flag@g2 then data@g1 (swapped) | yes (no edge forbids it) | 0 **or** 42, depending on read timing |

Two of the rows can produce the wrong answer, and *every one of them is a conforming execution*. The bug is not a rare race condition that "sometimes" fires; it is that you authorized both columns by declining to write down which one you needed. The whole job of a release/acquire pair is to delete the second row from the table of legal executions.

## Compiler reorderings: hoist, sink, and reuse — with the asm

Let us make the compiler's freedom concrete, because the single most common "the memory model bit me" bug is not exotic hardware reordering at all — it is the compiler caching a value in a register. Take the consumer's spin loop in isolation:

```c
// The spin, as the compiler sees it (no synchronization in sight).
while (!flag) { }
```

To the compiler, `flag` is an ordinary `bool` in memory, and within this thread *nothing in the loop body modifies it*. The loop is `flag`-invariant. So the textbook optimization called **loop-invariant code motion** applies: read `flag` once, before the loop, into a register, and test the register. If the register holds `false`, the loop can never change it, so the optimizer is fully entitled to compile the loop into an unconditional infinite loop. Here is what GCC at `-O2` actually does with a plain global `bool flag`:

```asm
; x86-64, gcc -O2, the plain `while (!flag) {}` version
        movzx   eax, BYTE PTR flag[rip]   ; read flag ONCE into eax
        test    al, al
        jne     .Ldone                    ; if already true, skip
.Lspin:
        jmp     .Lspin                    ; <-- infinite loop, flag never re-read
.Ldone:
        ...
```

Read that `.Lspin: jmp .Lspin` carefully. There is no load inside the loop. The compiler read `flag` exactly once, decided it was `false`, and emitted a loop that *can never observe the producer's write* — not because of any hardware subtlety, but because the generated code never touches memory again. This is **load hoisting** (also called load reuse: the compiler reuses the register copy instead of re-reading). The figure contrasts the two outcomes — the plain load cached in a register on one side, an atomic load that re-reads memory each spin on the other.

![a plain flag read hoisted once into a register so the loop never sees the update versus an atomic load that re-reads memory every spin](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-4.png)

The fix at the compiler level is to tell it the variable can change underneath it. In C/C++ the *minimum* tool is `volatile`, which forbids the compiler from caching the access — every read of a `volatile` is a real memory load, every write a real store:

```c
volatile bool flag = false;   // compiler must re-read on every access
// while (!flag) {}  now reloads flag from memory each iteration
```

```asm
; x86-64, gcc -O2, the `volatile bool flag` version
.Lspin:
        movzx   eax, BYTE PTR flag[rip]   ; reload flag EVERY iteration
        test    al, al
        je      .Lspin                    ; keep spinning while still false
        ; ... falls through when flag becomes true
```

Now there is a `movzx` *inside* the loop: a fresh load each spin. The loop will terminate when the producer's store becomes visible. But — and this is the trap that ruined a decade of C++ code — `volatile` fixes only the compiler half of the problem, and only the *visibility* half at that. `volatile` does **not** establish a happens-before edge, does **not** stop the hardware from reordering, and does **not** make the `data` payload safe. It stops the compiler from caching `flag`; it does nothing about the store buffer or the out-of-order core. In Java, confusingly, `volatile` means something *stronger* — it is a full memory-model construct with acquire/release and visibility guarantees — which is the source of endless cross-language confusion. We will untangle that in the case studies.

The hoist is not limited to a do-nothing spin loop, which is what makes it dangerous. The compiler hoists the load whenever the variable is *loop-invariant from this thread's view*, even when the loop does real work. Here is a loop that polls a shared `stop` flag while processing a buffer:

```c
extern bool stop;            // another thread sets this to ask us to halt
void worker(int *buf, int n) {
    for (int i = 0; i < n && !stop; i++)   // intent: bail out when stop flips
        buf[i] = expensive(buf[i]);
}
```

You wrote `!stop` in the loop condition *on purpose*, expecting it to be re-checked every iteration. But `expensive()` is a pure function the compiler can see does not touch `stop`, and nothing else in the loop writes `stop`, so to this thread `stop` is invariant. GCC at `-O2` hoists it:

```asm
; x86-64, gcc -O2: stop is read ONCE before the loop
        movzx   eax, BYTE PTR stop[rip]   ; load stop a single time
        test    al, al
        jne     .Lreturn                  ; if already set, do nothing
.Lloop:
        ; ... call expensive, store buf[i] ...   (NO reload of stop here)
        cmp     ebx, esi
        jl      .Lloop                    ; only the i < n test remains
.Lreturn:
```

The `!stop` check has *evaporated from the loop body*. The worker now runs to completion regardless of who sets `stop`, because the generated code committed to the value it read at entry. This is the same load-hoist as the spin loop, but it hides inside a loop that visibly does work, so it is far easier to miss in review. The fix is identical — make `stop` an atomic (or in C, at minimum a `volatile` or a `READ_ONCE`/`atomic_load`), which tells the compiler the value is *not* invariant and forces a reload each iteration.

The other two compiler reorderings round out the family. **Store sinking** is the mirror image: if a thread writes a variable several times and the intermediate values are never read within the thread, the compiler can sink the writes, keeping the value in a register and only writing memory once at the end (or never, if it proves the final value is dead). **Store/load reordering** between two *different* independent locations — exactly our `data = 42; flag = true;` pair — is permitted because, again, no single-thread observation distinguishes the orders. The compiler may emit `flag = true` before `data = 42` in the machine code if its instruction scheduler decides that is better, and you will never see it by reading the producer's own behavior.

There is a subtle reason all three of these — hoist, sink, reorder — are *the same phenomenon*. The compiler builds a dependency graph over your accesses and is free to execute any schedule that respects the *intra-thread* dependencies it can see. A read of `flag` that no intra-thread write feeds has no incoming dependency edge, so it floats freely to wherever the scheduler likes — out of a loop, before a barrier, fused with a sibling read. A write whose value is overwritten before any intra-thread read sees it is a dead node and gets pruned. Inter-thread dependencies are *invisible* to this graph by construction, because they live in another thread the compiler is not looking at. An atomic access is the one node the compiler is forbidden to treat as floatable or dead: it is marked as having effects the single-thread analysis must not assume away. That single property — "do not assume nobody else cares about this access" — is the whole of what `volatile`-for-the-compiler and `atomic` provide at the compiler layer.

#### Worked example: a hoist that turns a 200 ns wait into an infinite loop

Suppose the producer takes 200 ns to compute `data` and set `flag`, and the consumer starts spinning a few ns earlier. With `volatile bool flag`, the consumer reloads `flag` perhaps a few hundred million times during those 200 ns, sees the store land, and exits — total wait ≈ 200 ns. With a *plain* `bool flag` at `-O2`, the consumer reads `flag` once (`false`), enters `jmp .Lspin`, and spins until the process is killed — wait time = ∞. Same source loop, same optimizer, one keyword of difference. The lesson is not "always use `volatile`" (it is the wrong tool for the payload), but: **a plain read in a loop is not a poll of memory; it is a poll of a register the compiler chose for you.** To poll memory you must say so, and the right way to say so — `std::atomic` in C++, `volatile` in Java, `sync/atomic` in Go, an `Ordering` in Rust — also fixes the hardware half. That is the whole reason those types exist.

## The out-of-order core and the store buffer

Now turn off the compiler entirely — imagine you wrote the program in hand-tuned assembly so *no* compiler reordering can happen, every load and store is exactly where you put it. The bug does not go away. The hardware reorders too, and the central mechanism is the **store buffer**.

A modern core does not write to the cache (and thus to the coherent, shared view of memory) the instant it executes a store instruction. Writes are slow relative to the core's clock: the target cache line may not be in this core's L1, may be owned by another core, may need a coherence transaction over the interconnect. If the core *stalled* on every store waiting for the line, it would crawl. So instead, the core has a small private FIFO queue called the store buffer (a handful to a few dozen entries). A store instruction *retires* by dropping its (address, value) pair into the store buffer; the core moves on immediately. Sometime later — tens of cycles, sometimes more — the buffer *drains* the entry into the cache, at which point the write becomes visible to other cores via the coherence protocol.

Here is the consequence that breaks everything. The store buffer is **private to its core** and is consulted by that core's *own* loads via *store-to-load forwarding*: if this core reads an address it recently wrote, it sees its own buffered value, even though no other core can. So a thread always observes its own writes in program order — single-thread semantics are preserved, exactly as the as-if rule requires. But a thread's *load of a different address* can complete — reading the current value from cache — while an earlier *store to some other address* is still sitting in the buffer, invisible to everyone else. That is a **store→load reordering**, and it is visible to other cores even though it is invisible to the thread doing it.

It pays to state the x86-TSO model precisely, because it is small enough to hold in your head and it explains *exactly* which reorderings you will and will not see on an Intel or AMD box. The Sewell et al. "x86-TSO" formalization (CACM 2010) models the machine as: shared memory, plus one FIFO store buffer per core, plus a global lock for fenced/locked instructions. The operational rules are just these. (1) A store appends `(address, value)` to *this core's* buffer. (2) The buffer may, at any time, *dequeue its oldest entry* and write it to memory — this is the only path by which a store becomes visible to other cores, and because the buffer is FIFO, **stores from one core reach memory in program order** (that is why Store→Store is forbidden on x86). (3) A load to address `a` first checks *this core's* buffer for the most recent buffered store to `a` and forwards it if present; otherwise it reads memory. (4) An `mfence` (or any `lock`-prefixed op) blocks until *this core's* buffer is empty. From these four rules, every x86 ordering property follows mechanically. Loads happen "now" against the current memory/buffer state, in program order. Stores to a *given* address are seen by everyone in program order. The *only* slack is a load to address `b` reading memory while an earlier store to address `a` (`a ≠ b`) still sits unflushed in the buffer — pure Store→Load reordering, and nothing else. The model is so tight that you can prove a synchronization algorithm correct or buggy on x86 by hand-simulating these four rules.

![a core writes x into its private store buffer then reads y from memory before x drains so another core still sees the old x](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-2.png)

The figure traces it in time: core 1 posts `x = 1` to its store buffer (t1), the write is still buffered (t2), core 1 reads `y` from memory and gets the current value (t3), but another core still reads `x == 0` because the buffer has not drained (t4); only later (t5–t6) does `x = 1` become globally visible. The store happened *first in program order*, but the load became *globally visible first*. To the writing core nothing is wrong — its own loads would forward from the buffer — but to the rest of the system the two operations appear swapped.

This is not the only hardware reordering source, just the most universal one (every mainstream multiprocessor has store buffers). On weakly ordered machines there is more: the core executes instructions **out of order** — it issues independent loads early to hide cache-miss latency, speculates past branches, and lets a younger load complete before an older one. Caches propagate writes between cores with their own delays.

The out-of-order engine deserves a closer look, because it is the source of the *load*-side reorderings that x86's store-buffer-only model does not have. A modern core does not execute one instruction at a time; it has a wide front-end that decodes several instructions per cycle into an out-of-order back-end with dozens of in-flight operations. Loads are the latency-critical ones — a single L3 miss to DRAM costs on the order of a hundred-plus cycles — so the core issues loads *as early as their addresses are known*, often well ahead of older instructions, and lets the results arrive whenever the cache delivers them. Two independent loads `L1; L2` in program order can therefore *complete* in the order `L2, L1` if `L1` missed cache and `L2` hit. On a weakly ordered machine that completion order is *observable to other cores*, which is Load→Load reordering. The core also speculates: it predicts a branch, runs ahead, and issues loads on the predicted path before the branch resolves. If the data those loads touch is being written by another core, the speculatively-early load can grab a value from "before" a point your source code says it should be "after." x86 plugs this leak with extra machinery — it speculates too, but it *snoops* the coherence traffic and squashes-and-reissues any load whose line was invalidated before the load retired, so the *visible* result still looks in-order for loads. ARM and POWER simply do not pay for that machinery on ordinary loads; they let the reordering be visible and give you barrier instructions (`dmb`, `ldar`/`stlr`, `lwsync`) to suppress it where you actually need order. That is the deal weak ordering offers: cheaper loads in the common case, an explicit barrier in the rare case you need ordering — versus x86's "always pay for in-order-looking loads."

The figure below lays out the full set of layers that sit between the line you wrote and the moment another core reads it.

![the layered set of reorderers between source and another core covering the compiler the out-of-order core the store buffer and cache coherence](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-7.png)

Source order is the top; below it the compiler can hoist, sink, and reuse; below that the out-of-order core can issue and complete instructions in a different order; below that the store buffer delays your writes by tens of cycles; below that cache coherence (MESI) carries the write to other cores with its own latency; and only at the bottom does another core actually observe a *final* order — which may differ from the top on every one of those layers. Any single layer can reorder; in the worst case several of them do, and the gap between "what you wrote" and "what another core sees" is the sum of all of them.

## The store-buffer litmus test: both threads read zero

The store buffer's signature is a tiny program that *cannot* produce a certain outcome on paper and *routinely* produces it on real hardware. It is the heart of Dekker's and Peterson's mutual-exclusion algorithms, and it is the cleanest possible demonstration that hardware reorders. Two shared variables start at 0:

```c
int x = 0, y = 0;
int r1, r2;

// Thread 1:                // Thread 2:
x = 1;                      y = 1;
r1 = y;                     r2 = x;
```

Each thread writes its own variable, then reads the other thread's variable. Ask: **can both `r1` and `r2` end up 0?** Reason it out under the naive assumption that all memory operations happen in a single global order (this assumption has a name — *sequential consistency*, SC — and it is what most people unconsciously assume). Under SC, take whichever of the four operations happens *first* in the global order. If it is `x = 1`, then by the time thread 2 does `r2 = x` it must read 1, so `r2 = 1`. If it is `y = 1`, then `r1 = y` reads 1, so `r1 = 1`. Either way, **at least one of `r1`, `r2` is 1**. The outcome `r1 == 0 && r2 == 0` is *impossible* under sequential consistency. There is no interleaving of these four operations, in any single global order, that leaves both reads at 0.

Let me make that impossibility argument airtight, because its precision is the whole point. Under SC, the execution is *some* total order over the four operations consistent with each thread's program order. Program order forces `x=1` before `r1=y` (thread 1) and `y=1` before `r2=x` (thread 2). Consider the operation that is *first* in the global total order. It is one of the four. It cannot be a load that reads 0 *and* be globally first while the matching store comes later in a way that still yields both-zero — let us check exhaustively. Whatever is first, by definition no other operation precedes it. If `x=1` is first, then `r2=x` (which comes after it in the order, since `x=1` is first) reads the current value of `x`, which is now 1 — so `r2=1`, contradiction. If `y=1` is first, then `r1=y` reads 1 — so `r1=1`, contradiction. If `r1=y` is first, then `r1` reads the *initial* `y=0`, fine — but now `x=1` must come before `r1=y` in thread 1's program order, contradicting `r1=y` being globally first. Symmetrically `r2=x` cannot be first. So the globally-first operation must be a store, and either store being first forces one read to 1. **No SC execution yields `r1==r2==0`.** The result is not merely improbable under SC; it is excluded by the structure of any total order.

And yet on every x86, ARM, and POWER machine you can buy, `r1 == r2 == 0` happens — rarely on x86, often on ARM. The reason is the store buffer. Both stores (`x = 1` and `y = 1`) land in their respective cores' store buffers and have not drained. Then both loads execute, reading from coherent cache/memory, which still shows 0 for both. Each core's own load of the *other* variable is unaffected by its own buffered store (different address — store-to-load forwarding does not apply). So `r1 = y` reads 0, `r2 = x` reads 0, and only *afterward* do the buffers drain. Both reads slipped past both buffered writes. Re-express this in the x86-TSO operational rules from earlier: thread 1 appends `x=1` to buffer 1 (not yet drained), thread 2 appends `y=1` to buffer 2 (not yet drained), thread 1's `r1=y` checks buffer 1 (no entry for `y`) and reads memory's `y=0`, thread 2's `r2=x` checks buffer 2 (no entry for `x`) and reads memory's `x=0`. Every step obeys the four rules; the outcome is `0/0`. The model that forbids it under SC *predicts* it under TSO — which is exactly why TSO is the right model and naive SC is the wrong one for reasoning about real x86.

![both threads store their own flag into a buffer then read the other and both reads return zero which sequential consistency forbids](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-5.png)

The figure walks the interleaving: T1 buffers `x = 1` (s1), T2 buffers `y = 1` (s2), T1 reads `y` and gets 0 (s3), T2 reads `x` and gets 0 (s4), the buffers drain afterward (s5), and the result `r1 = r2 = 0` is recorded (s6) — an outcome no sequentially-consistent machine could produce. This is *the* canonical store→load reordering. It is exactly why Dekker's and Peterson's algorithms, which are provably correct under SC, *break* on real hardware unless you insert a memory fence between each store and the following load: the fence forces the store buffer to drain before the load proceeds, restoring the SC outcome at the cost of stalling for the drain.

#### Worked example: the litmus harness and what it measures

You can observe this yourself. The harness runs both threads in a tight loop over many iterations, resetting `x` and `y` to 0 each round, and counts how often both reads come back 0:

```c
#include <stdatomic.h>
#include <pthread.h>
#include <stdio.h>

// relaxed atomics so the COMPILER won't reorder/cache, leaving only
// the HARDWARE store->load reordering to observe.
atomic_int x, y;
int r1, r2;
volatile int go;          // barrier flag for the two threads

void *t1(void *_) {
    while (!go) {}
    atomic_store_explicit(&x, 1, memory_order_relaxed);
    r1 = atomic_load_explicit(&y, memory_order_relaxed);
    return 0;
}
void *t2(void *_) {
    while (!go) {}
    atomic_store_explicit(&y, 1, memory_order_relaxed);
    r2 = atomic_load_explicit(&x, memory_order_relaxed);
    return 0;
}

int main(void) {
    long both_zero = 0;
    for (long i = 0; i < 1000000; i++) {
        atomic_store_explicit(&x, 0, memory_order_relaxed);
        atomic_store_explicit(&y, 0, memory_order_relaxed);
        go = 0;
        pthread_t a, b;
        pthread_create(&a, 0, t1, 0);
        pthread_create(&b, 0, t2, 0);
        go = 1;                       // release both threads
        pthread_join(a, 0);
        pthread_join(b, 0);
        if (r1 == 0 && r2 == 0) both_zero++;
    }
    printf("both-zero: %ld / 1000000\n", both_zero);
    return 0;
}
```

Two things matter here. First, we use `memory_order_relaxed` atomics rather than plain `int` so the *compiler* is forbidden from caching or reordering the accesses — relaxed atomics give *no* ordering guarantees between different locations but *do* force a real load/store of each one. That isolates the experiment to pure *hardware* reordering. Second, this naive thread-per-iteration version is dominated by `pthread_create`/`join` overhead; a real measurement pins two long-lived threads to two cores and synchronizes them per iteration with a sense-reversing barrier so the two stores happen close in time. The honest result, and how it differs by platform, is the next section.

Here is the *proper* harness in C++ — two long-lived threads pinned to distinct cores, synchronized each round by a shared `epoch` counter so both stores fire within a few nanoseconds of each other. This is the structure `litmus7` generates for you; writing it once by hand is the best way to understand why the naive version misses the window:

```cpp
#include <atomic>
#include <thread>
#include <cstdio>

std::atomic<int> X{0}, Y{0};
std::atomic<int> epoch{0};        // round synchronizer
int r1, r2;
long both_zero = 0;

// Each thread waits for the round to start, runs its store+load, then
// waits for the round to end. Both stores fire nearly simultaneously.
void run(std::atomic<int>& mine, std::atomic<int>& other, int& r) {
    for (int round = 1; ; ++round) {
        while (epoch.load(std::memory_order_acquire) != round) {}  // wait for start
        mine.store(1, std::memory_order_relaxed);                  // my store
        r = other.load(std::memory_order_relaxed);                // read the other
        epoch.fetch_add(1, std::memory_order_acq_rel);            // signal done
        if (round == 1'000'000) return;
    }
}
```

The controller thread resets `X` and `Y` to 0, bumps `epoch` to start a round, waits for both workers to bump it back, then checks `r1 == 0 && r2 == 0`. On a quiet two-core x86 box this typically reports a handful of `0/0` outcomes per million rounds; on Apple silicon or a Graviton core it reports hundreds to thousands. The exact number is not the lesson — the lesson is that the *same source* yields a count that is near-zero on one architecture and three orders of magnitude larger on another, with no code change. That gap *is* the difference between TSO and weak ordering, made into a number you can print.

#### Worked example: counting outcomes, SC versus observed

Suppose you run the harness for `N = 1,000,000` rounds and tabulate the four possible `(r1, r2)` outcomes on two machines. A representative result looks like this:

| Outcome `(r1, r2)` | Allowed under SC? | x86-64 desktop (count / 1e6) | Apple silicon (count / 1e6) |
| --- | --- | --- | --- |
| `(1, 0)` — T1 saw T2's store | yes | ~331,000 | ~300,000 |
| `(0, 1)` — T2 saw T1's store | yes | ~331,000 | ~298,000 |
| `(1, 1)` — both saw the other | yes | ~338,000 | ~399,000 |
| `(0, 0)` — both missed | **no (SC forbids)** | ~12 | ~3,100 |

The first three rows are the SC-legal interleavings and dominate on both machines. The fourth row is the one SC declared *impossible* — and it has a nonzero count on both, ~12 on x86 versus ~3,100 on the weakly ordered core, roughly a 250× difference. (Treat the exact integers as approximate; they swing with frequency scaling, which physical cores you land on, and machine load.) Read the table the way a physicist reads a forbidden spectral line that nonetheless appears: the line you were told could not exist is faint on x86 and bright on ARM, and its mere presence falsifies the SC model for real hardware. If you insert an `mfence` (x86) or `dmb ish` (ARM) between the store and the load and rerun, the fourth row drops to a hard `0` across all million rounds — the experimental confirmation that the fence does exactly what the model says it does.

## x86-TSO versus ARM and POWER: how strong is your model?

Not all hardware reorders equally. A **memory consistency model** is the formal contract a CPU offers about which reorderings are visible. They form a spectrum from *strong* (few reorderings, closer to SC, easier to reason about, more expensive to enforce) to *weak* (many reorderings, faster, brutal to reason about). The two ends you will actually meet are **x86-TSO** and **ARM/POWER**.

x86 (and SPARC's default mode) implements **Total Store Order**, TSO. The mental model is precise: it is sequential consistency *plus a store buffer*, and nothing else. Concretely, of the four possible reordering pairs, x86-TSO allows **exactly one**: a store followed by a load to a *different* address can be reordered (the store buffer effect). Load→load, load→store, and store→store are all kept in program order. This is why x86 is forgiving: the only litmus that fails on x86 is the store-buffer one we just walked, and a single `mfence` (or any `lock`-prefixed instruction) between the store and the load fixes it. Most accidentally-racy code *appears* to work on x86 precisely because TSO is so close to SC — which is a trap, because the same code ported to ARM falls apart.

ARM and POWER are **weakly ordered**. Almost any pair of independent memory operations can be reordered: load→load, load→store, store→store, and store→load are all fair game, and on POWER even the *visibility* of a single write can reach different cores at different times (non-multi-copy-atomic stores, in the jargon). The figure puts the two models side by side across the four pairs.

![a matrix of the four reordering pairs showing x86-TSO allows only store then load while ARM and POWER allow all four](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-3.png)

Read across the rows: under x86-TSO, LoadLoad, LoadStore, and StoreStore are kept in order, only StoreLoad can reorder (the store-buffer pair). Under ARM/POWER, every row is "can reorder." This single table explains a whole genre of bug reports: *"it worked on my laptop and broke on the phone / the Graviton instance / the Mac with Apple silicon."* x86 was hiding three of the four reorderings; the weakly ordered chip hides none of them. Here is the same comparison as a reference table you can keep:

| Reordering pair | x86-TSO | ARM / POWER | Triggering mechanism |
| --- | --- | --- | --- |
| Store → Load (different addrs) | allowed | allowed | store buffer drains after the load |
| Store → Store | forbidden | allowed | write combining / coalescing buffers |
| Load → Load | forbidden | allowed | out-of-order issue, speculation |
| Load → Store | forbidden | allowed | out-of-order issue |
| Single store, seen by all at once | yes (multi-copy atomic) | x86 yes; POWER no | coherence propagation order |

The practical upshot: **x86 is the easy mode and a liar.** It will let buggy synchronization code pass your tests for years, then detonate the day someone runs it on ARM. The defensive posture is to never rely on the model's strength — write code that is correct under the *weakest* model you will ever target, by establishing happens-before edges explicitly, and let the strong models simply do less work to honor edges you already declared.

This "let the strong model do less work" is not a slogan — it is visible in the generated code, and it is the best argument for *always* writing the atomic even when you only ship on x86. Compile the same `flag.store(true, std::memory_order_release)` for both architectures. On x86-TSO, a release store needs *no extra instruction*: TSO already forbids Store→Store reordering, so a release store is just a plain `mov`. On ARM, the same release store compiles to a special store-release instruction, `stlr`, which the hardware orders against all prior writes:

```asm
; x86-64: flag.store(true, release)  -- release is free under TSO
        mov     BYTE PTR flag[rip], 1     ; a plain store; TSO gives the ordering

; AArch64: flag.store(true, release)  -- needs the release-store instruction
        mov     w8, #1
        stlrb   w8, [x0]                  ; store-release: ordered after prior writes
```

The acquire load is symmetric: a plain `mov`/`ldr` on x86 (TSO forbids the Load→Load reordering for free), a `ldarb` (load-acquire) on ARM. The crucial point: **you wrote one program with `memory_order_release`/`acquire`, and the compiler emitted the minimal fence each architecture needs — nothing on x86, a `stlr`/`ldar` pair on ARM.** Had you written plain stores "because it works on x86," you would have shipped code that is correct on x86 by accident and wrong on ARM by omission, and the binary would look *identical* on x86 to the correct version — which is precisely why the bug is invisible until the ARM port. The atomic costs you nothing on x86 and saves you on ARM. There is no scenario where writing the plain version was the better engineering choice.

The full-fence case (`mfence` / `dmb ish`) is the expensive one, and it is worth knowing what you pay. An `mfence` on x86 drains the entire store buffer and blocks subsequent loads until it is empty — on the order of tens of cycles when the buffer has dirty entries, sometimes more if those entries miss cache. A `dmb ish` on ARM is comparable. That cost is why `seq_cst` atomics (which insert a full fence on the Store→Load boundary to recover SC) are measurably slower than `release`/`acquire` in tight loops, and why the memory-model track spends so much effort teaching you to use the *weakest correct* order rather than reaching for `seq_cst` reflexively. You only pay for the StoreLoad fence when you genuinely need to forbid the one reordering even x86 allows — the Dekker/Peterson case below.

#### Worked example: a `data; flag` publish that "works" on x86 and breaks on ARM

Take the broken publish from the intro and compile it with plain (non-atomic) stores on each architecture. On **x86-TSO**, the producer's two stores are `data = 42` then `flag = true` — a Store→Store pair, which TSO *forbids* from reordering. So the writes become visible in program order: any consumer that sees `flag == true` also sees `data == 42`. The broken code *appears correct on x86* — not because it is correct, but because the architecture happens to forbid the specific reordering that would expose it. On **ARM**, Store→Store *is* allowed, so the core can make `flag = true` visible before `data = 42`. A consumer reads `flag == true`, then reads `data == 0`, prints `0`. Same source, same compiler family, one architecture exposes the latent bug and the other masks it. The code was *always* wrong — it never established that the data write happens-before the flag read — but only the weak model reveals it. This is the single most common way teams discover their flag protocol was never actually synchronized.

## The broken publish: flag visible, data stale

Now we can fully dissect the intro's first bug, because both the compiler half and the hardware half are on the table. The producer does two stores; the consumer does a load (the flag) and then a dependent load (the data). For the protocol to be correct, the consumer reading `flag == true` must imply the consumer also reads `data == 42`. That requires an *ordering guarantee on the producer's stores* (data before flag must be visible in that order) **and** an *ordering guarantee on the consumer's loads* (read flag before read data, with no caching of either). Plain code gives you neither.

On the producer side, three things can break it. The **compiler** may reorder the two independent stores (Store→Store reordering is legal at the source level even though x86 forbids it at the hardware level — the *compiler* is a separate reorderer). The **store buffer / coherence** may make `flag` visible before `data` on a weakly ordered machine. Either one lets the flag overtake the data.

On the consumer side, two things can break it. The compiler may hoist `flag` into a register (the infinite-spin bug) so it never sees the update. And even if it does see the flag, on a weakly ordered machine the consumer's *load of data* may have been issued *before* the load of flag — speculative, out-of-order execution can read `data` early, get the stale `0`, and then read `flag` and get `true`, combining a fresh flag with a stale payload. ARM allows Load→Load reordering precisely this way.

It is worth seeing the bug in Java specifically, because Java's `volatile` is a trap for the unwary: it is *stronger* than C's, which makes the non-`volatile` version look superficially fine and lulls you. Here is the broken Java — no `volatile` anywhere — which the JMM explicitly permits to misbehave:

```java
class BrokenPublisher {
    int data = 0;           // plain field
    boolean flag = false;   // plain field -- NO volatile (the bug)

    void producer() {
        data = 42;          // (1)
        flag = true;        // (2) JIT/hardware may make this visible before (1)
    }
    void consumer() {
        while (!flag) { }   // (3) JIT may hoist 'flag' -> spin forever
        use(data);          // (4) may read 0 even after seeing flag == true
    }
}
```

This compiles, runs, and on a lightly loaded x86 JVM usually *appears* to work — which is exactly the danger. Two independent failures lurk. The HotSpot JIT may hoist the plain `flag` read out of the spin loop (the infinite-spin bug, identical to the C case), so `consumer()` never terminates. And even if it does see the flag, the JMM permits the producer's two plain writes to become visible out of order, so `use(data)` can read `0`. The JMM does *not* establish a happens-before edge between a plain write and a plain read of a different variable — that edge is exactly what `volatile` (or `synchronized`, or a `java.util.concurrent` construct) creates. The fix is a one-word change, shown just below, and it is the same structural fix as every other language: turn the flag into something that carries release/acquire.

So the broken publish is not one bug; it is up to four independent reorderings that all have to be suppressed. The figure contrasts the broken version against the fixed one.

![a plain flag publish where stores reorder and the reader sees a stale payload versus a release store and acquire load that pins the order](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-8.png)

The fix, previewed here and developed fully in D3, is to make the flag an **atomic with release/acquire semantics**. A *release store* on `flag` carries a guarantee: every memory write the producer did *before* the release (including `data = 42`) is visible to any thread that performs an *acquire load* of the same flag and sees the released value. The acquire load on the consumer side carries the dual guarantee: every read the consumer does *after* the acquire sees at least everything the releasing thread published. Together they create the **happens-before edge** the plain code lacked — `data = 42` happens-before the flag write happens-before the flag read happens-before reading `data`, transitively forcing the consumer to read `42`. Here it is in C++:

```cpp
#include <atomic>
#include <thread>

std::atomic<bool> flag{false};
int data = 0;              // plain int is fine; the atomic carries the order

void producer() {
    data = 42;                                          // (1) payload
    flag.store(true, std::memory_order_release);        // (2) release: publishes (1)
}

void consumer() {
    while (!flag.load(std::memory_order_acquire)) { }   // (3) acquire: re-reads + syncs
    // Guaranteed: data == 42 here. The acquire pairs with the release.
    use(data);                                          // (4) sees 42, never 0
}
```

Three things changed and all three matter. The `flag` is now `std::atomic<bool>`, so the compiler cannot cache it in a register — the load in the spin loop re-reads memory every iteration (kills the infinite-spin bug). The store is `memory_order_release`, which forbids the producer's earlier writes (the `data = 42`) from being reordered *after* it, at both compiler and hardware level (kills the producer-side reordering). The load is `memory_order_acquire`, which forbids the consumer's later reads (the `use(data)`) from being reordered *before* it (kills the consumer-side load-load reordering). The plain `int data` does not need to be atomic, because the release/acquire pair *transports* its visibility — that is the elegance of the construct: you pay for ordering on the single flag and get the whole payload published for free.

Java expresses the same fix with `volatile` (which, unlike C's `volatile`, is a full release/acquire construct in the JMM since Java 5):

```java
class Publisher {
    int data = 0;                 // plain field
    volatile boolean flag = false; // JMM volatile = release on write, acquire on read

    void producer() {
        data = 42;                 // (1)
        flag = true;               // (2) volatile write = release; publishes (1)
    }
    void consumer() {
        while (!flag) { }          // (3) volatile read = acquire; re-reads + syncs
        use(data);                 // (4) guaranteed to see 42
    }
}
```

Go uses a channel or `sync/atomic` to the same end; the idiomatic Go answer is *don't hand-roll a flag at all* — send the data over a channel, which establishes happens-before by construction:

```go
// The Go way: the channel send/receive IS the happens-before edge.
func main() {
    ch := make(chan int)
    go func() {
        data := 42        // any prep work
        ch <- data        // send: happens-before the receive below
    }()
    got := <-ch           // receive: sees everything the sender did before the send
    use(got)              // guaranteed to see 42
}
```

Rust makes the requirement unavoidable: you *cannot* share a plain `bool` across threads — the type system rejects it (`bool` is `Sync`, but a mutable shared reference is not). You reach for `AtomicBool` with an explicit `Ordering`, and the compiler will not let you forget:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

let flag = Arc::new(AtomicBool::new(false));
let data = Arc::new(std::sync::atomic::AtomicUsize::new(0));
// producer:
data.store(42, Ordering::Relaxed);
flag.store(true, Ordering::Release);   // release: publishes the data store
// consumer:
while !flag.load(Ordering::Acquire) {} // acquire: pairs with the release
let v = data.load(Ordering::Relaxed);  // guaranteed 42
```

The common thread across all four languages: you must *name the ordering*, and once you do, the compiler and the hardware both honor it. The plain version named nothing, so nothing was honored.

## What actually fixes it: a preview of barriers and atomics

Step back and name the tools, because the rest of the memory-model track is about choosing among them. There are three levels of fix, from coarse to surgical:

1. **A lock** (`mutex`, `synchronized`). Acquiring a lock is an acquire operation; releasing it is a release operation. Everything one thread did before releasing the lock is visible to the next thread that acquires it. Locks solve the memory-model problem *as a side effect* of mutual exclusion — which is why most application code never has to think about reordering at all: the lock's release/acquire edges do it for you. This is the default and usually the right one. ([Mutexes and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) build on exactly this.)
2. **An atomic with an explicit memory order** (`std::atomic` + `memory_order_*`, Java `volatile`, Go `sync/atomic`, Rust `Ordering`). This is the release/acquire pair we just used: cheaper than a lock for a single flag or counter, and it is how you build lock-free structures. It is also where you can get it *wrong* by choosing too weak an order (`relaxed`), so it demands you actually understand happens-before — the subject of D2.
3. **A bare memory barrier / fence** (`std::atomic_thread_fence`, `__sync_synchronize`, `smp_mb()` in the kernel, `mfence`/`dmb`/`lwsync` at the ISA level). A fence is the raw hardware instruction that forces ordering — e.g., a full fence drains the store buffer and blocks later loads until earlier stores are visible. You rarely write these by hand outside a kernel or a lock-free library; atomics generate them for you. D4 covers fences directly.

The figure maps each reorderer to the brake that stops it.

![a matrix listing the compiler the out-of-order core the store buffer and cache coherence against what each does and what brake stops it](/imgs/blogs/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering-6.png)

The compiler's hoist/sink/reuse is stopped by an atomic or `volatile` access. The out-of-order core is stopped by a fence instruction. The store buffer is drained by a store-load fence. Cache coherence (MESI) is not something you stop — it is the mechanism that *guarantees* a write eventually reaches every core; it is the floor of correctness, not a reorderer you fight. The key insight the figure encodes: **different layers need different brakes, and an atomic's memory order is a request for the right combination of them.** A release store emits a compiler barrier plus whatever hardware barrier the target needs (nothing extra on x86 for a release store, a `dmb ish` on ARM). The atomic abstracts the per-architecture detail so you write one correct program that compiles to the right fences everywhere.

## Measured: what reordering actually looks like on real hardware

The honest part. How often does this happen? The answer is *deeply* microarchitecture-dependent, and anyone who quotes you a single precise rate is selling something. What I can give you is the shape of the result, the methodology, and order-of-magnitude bands from the well-known litmus-test literature and reproducible tools — with the caveat that your numbers will differ by chip, core count, frequency scaling, and how tightly you synchronize the two threads.

The store-buffer litmus (`SB`, "both read 0") is the standard probe. The right way to measure it is **not** the naive thread-per-iteration harness above — that is dominated by thread-creation cost and the two stores almost never line up in time. The proper tool is `litmus7` from the **diy / herd7 toolsuite** (Alglave, Maranget, et al.), which pins two long-lived threads to two specified cores, synchronizes them per iteration with a tuned sense-reversing barrier, randomizes affinity and timing to maximize the chance the two stores overlap, and runs millions of iterations. It reports how many times each outcome was *observed*. Run the `SB` test and you get a table like this (representative magnitudes, not a guarantee for your box):

| Platform (representative) | `SB` "0/0" observed | Notes |
| --- | --- | --- |
| x86-64 desktop, 2 cores, `litmus7` | a few per 1e5 to 1e6 iters | rare but real; needs tight sync to hit |
| x86-64 with `mfence` between store and load | 0 / 1e9 | fence drains the buffer; outcome forbidden |
| ARMv8 / Apple silicon, 2 cores | a few per 1e3 to 1e4 iters | orders of magnitude more frequent than x86 |
| ARMv8 with `dmb ish` barrier | 0 | barrier restores SC for this test |
| Same code, single core / pinned to 1 core | 0 | no second core to observe the reorder |

Three things this table teaches. First, **x86 reorders too** — it is not "strongly ordered = never reorders," it is "reorders only Store→Load, rarely, when the stars align." You can observe it; it is just hard to hit because the window (store buffered AND the other thread loads in that window) is narrow. Second, **ARM reorders far more readily** — the same `SB` test fires orders of magnitude more often, which is the empirical face of the "weakly ordered" label. Third, **the barrier makes it disappear entirely** — inserting the fence between the store and the load drops the count to zero across billions of iterations, which is the experimental proof that the fence does what the model says. (And on a single core there is no reordering to observe at all — the whole phenomenon is inter-core.)

```bash
# Reproduce the canonical store-buffer litmus with the herd7 toolsuite.
# (opam install herdtools7  — then write the SB.litmus test and run it)
litmus7 -a 2 -i 5 SB.litmus       # -a 2: use 2 cores; -i 5: 5 affinity modes
# ... prints e.g.:
#   Observation SB Sometimes 412 999588   <- "0/0" seen 412 times in ~1e6
# Add a fence to the test and rerun:
litmus7 -a 2 SB+mfences.litmus
#   Observation SB+mfences Never 0 1000000 <- fence forbids it
```

For the *compiler* half — the hoisted-flag infinite spin — measurement is even simpler and fully deterministic: compile the plain-`bool` spin at `-O2`, run it, and it hangs 100% of the time; compile the `volatile`/atomic version and it terminates 100% of the time. There is no nondeterminism because the reordering is a *compile-time* decision baked into the binary. You can confirm by disassembly (`objdump -d`) and *see* the missing load inside the loop, exactly the `.Lspin: jmp .Lspin` shown earlier. That is the most convincing measurement of all: the bug is visible in the machine code before you ever run it.

#### Worked example: measuring honestly, and the confounds

If you build the `SB` harness yourself and get *zero* "0/0" outcomes on x86, do not conclude "x86 never reorders." More likely you hit one of the confounds: (a) the two threads were not pinned to *different physical cores* (on the same core there is no inter-core reorder to see; on two hyperthreads of one core, the store buffer may be shared); (b) the two stores never overlapped in time because your barrier was loose, so one buffer always drained before the other thread loaded; (c) the compiler reordered or fused your plain accesses, changing the test (use relaxed atomics to pin the *compiler* behavior so you measure only *hardware*); (d) frequency scaling / a busy machine widened the timing so the window closed. Honest measurement here means: pin threads to distinct cores, use a tight per-iteration barrier, use relaxed atomics so only hardware is under test, run ≥1e6 iterations, and report a *rate with a platform name*, never a bare "X%." The reordering is real; observing it reliably is an experimental skill, which is exactly why dedicated tools like `litmus7` exist.

## When you can ignore all of this (and when you absolutely cannot)

This is a long post about a deep problem, so it is worth being blunt about how often you actually have to think about it. The answer for most code is: **almost never — because you are already synchronized.** The reordering hazard only bites *unsynchronized* shared access. If every shared access is inside a lock, behind a channel, through a properly-ordered atomic, or through a concurrent data structure someone else got right, the happens-before edges are already there and the memory model is invisible to you. That is the entire point of those higher-level constructs: they let application programmers ignore store buffers.

You can ignore reordering entirely when:

- **The code is single-threaded.** The as-if rule guarantees single-thread correctness, full stop. No second observer, no problem. The store buffer forwards your own writes to your own reads; the compiler preserves your observable behavior. Reordering is undetectable.
- **All shared state is behind a lock or `synchronized` block.** The lock's release/acquire edges order everything for you. You are paying for it (locks have cost), but you bought correctness, including memory ordering, as part of the deal.
- **Communication is via channels / message passing / actors.** A channel send happens-before the matching receive; an actor processes one message at a time. The model establishes the order — there is nothing to hand-order. (This is why Go and Erlang programmers rarely touch atomics.)
- **You use the language's concurrent collections** (`ConcurrentHashMap`, `sync.Map`, `crossbeam` structures). The library authors handled the orderings; you inherit them.

You **cannot** ignore reordering — and must establish happens-before explicitly — when:

- **You hand-roll any flag-based publish or wait protocol** (`done = true; ... if (done) use(data)`), a double-checked-locking lazy init, a hand-spun spinlock, a sequence counter, or a status word. This is the exact pattern that breaks; it needs a release store and an acquire load.
- **You write lock-free or wait-free code** — a CAS-based stack/queue, a ring buffer, an RCU read side. The whole field *is* memory ordering; getting the orders wrong is the bug. (See [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads) and [how a lock is built from test-and-set, CAS, and spinlocks](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) for where this gets real.)
- **You are porting from x86 to ARM/POWER** (a phone, Graviton, Apple silicon, a Power server). Code that "worked" on x86's strong model can expose latent reordering bugs on the weak model. Re-audit every hand-rolled synchronization.
- **You reach for `relaxed` ordering to shave cost.** `relaxed` gives *no* inter-location ordering; it is correct only when you can *prove* you do not need an edge (e.g., a statistics counter nobody reads to make a decision). Reaching for it "to be fast" without that proof is how you ship a Heisenbug.

The decisive rule: **default to a lock or a channel; reach for explicit atomics/orderings only when you have measured a real contention or latency problem that a lock cannot solve; reach for `relaxed`/bare fences only with a proof.** The memory model is a tax you pay for *removing* the higher-level synchronization, and most code should not be removing it. The [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) frames this choice across the whole series.

## Case studies / real-world

**The double-checked locking saga (Java, pre-5).** The most famous victim of reordering in the wild is the "double-checked locking" idiom for lazy singleton init: check `if (instance == null)`, and only if so take a lock and check again before constructing. It looks like it avoids locking on the common path. Before Java 5 it was *broken*, and the reason is exactly this post's mechanism. The construction `instance = new Singleton()` is, at the bytecode level, three steps: allocate memory, run the constructor, publish the reference. The JMM (and the JVM's reordering) allowed the *publish* to become visible *before* the constructor finished — a Store→Store reordering. So a second thread could see a non-null `instance` and use a half-constructed object: flag visible, data stale, exactly our pattern. The fix that finally worked (Java 5+) was to declare the field `volatile`, whose release/acquire semantics forbid the reordering — *the same fix* as our broken publish. The episode produced the famous paper "The 'Double-Checked Locking is Broken' Declaration," signed by much of the concurrency community, and it directly motivated the rewrite of the JMM in JSR-133. The takeaway: a reordering bug fooled experts for years because it *happened to work* most of the time and on most JVMs.

**The Linux kernel and `smp_mb()` / `READ_ONCE`.** The Linux kernel runs on x86, ARM, POWER, RISC-V, and more, so it cannot lean on any one architecture's strength. It encodes the *weakest* model in its own memory-barrier API (`smp_mb()`, `smp_rmb()`, `smp_wmb()`, `smp_load_acquire()`, `smp_store_release()`) and its `READ_ONCE`/`WRITE_ONCE` macros, which are exactly C's "stop the compiler from caching/reordering this access" tool (a `volatile` cast internally). Kernel developers maintain a formal model of these (the LKMM, the Linux Kernel Memory Model, by Alglave, Maranget, McKenney, et al., built on the same herd7 tooling). The lesson for application code is the kernel's discipline: never touch shared memory without an explicit ordering construct, and *test it on the weak architectures*, because x86 will lie to you about correctness.

**The x86→ARM porting surprise (industry-wide, 2020s).** When Apple shipped Apple silicon and AWS shipped Graviton, a wave of subtle concurrency bugs surfaced in software that had run "correctly" on x86 for years. The cause was always the same: hand-rolled synchronization that relied, unknowingly, on x86-TSO forbidding Store→Store or Load→Load reordering. On ARM those reorderings are allowed, and the latent bug became a real one. The fix was never "add a sleep" or "retry" — it was to find the missing happens-before edge and insert the right atomic or barrier. The industry-scale version of our worked example: the code was always wrong; the strong model had been hiding it. Reproducing such bugs is exactly what `litmus7` and ARM's own model documentation are for.

**The C++11 `std::memory_order_consume` retreat (approximate, mid-2010s).** A subtler case study, and a cautionary one about *how hard this is even for experts*: when C++11 standardized atomics, it included a fourth ordering, `memory_order_consume`, intended to capture *dependency-ordered* loads — the cheap, fence-free ordering that POWER and ARM give you "for free" when a later load address-depends on an earlier load (the basis of RCU's read side in the Linux kernel). The idea was sound and the hardware really does provide it. But the *specification* of which source-level dependencies a compiler must preserve turned out to be so intricate that, in practice, every major compiler implemented `consume` by silently promoting it to the stronger, more expensive `acquire` — meaning the feature delivered none of its intended performance and the standard committee has spent multiple revisions trying to re-specify it. (Treat the dates and details as approximate; the high-level arc is well documented in the C++ standards papers and Paul McKenney's writing on RCU.) The lesson is humbling and on-theme: the gap between "what the hardware allows," "what the compiler can prove it must preserve," and "what a programmer can correctly write down" is wide enough that even a careful standardization effort by the world's experts can fall into it. When the committee that *wrote* the memory model finds an ordering too subtle to specify, you should be very sure before hand-rolling your own.

## When to reach for this (and when not to)

- **Reach for explicit memory ordering** (atomics with `acquire`/`release`, or a fence) when you are building the synchronization primitive itself — a lock-free queue, a sequence lock, a one-shot publish, a hand-spun spinlock — *and* a plain lock is genuinely your measured bottleneck. The order is the whole correctness argument; write it down and pair every release with an acquire.
- **Do not reach for atomics/orderings** when a `mutex`, a channel, or a concurrent collection already solves your problem. Those carry the right edges for free and are far harder to get wrong. Hand-rolled lock-free code is a maintenance liability that should earn its place with a benchmark, not a hunch.
- **Reach for `volatile` (C/C++)** only to stop the *compiler* from caching a memory-mapped I/O register or a flag a signal handler touches — and know it does *nothing* for inter-thread ordering. For threads, use `std::atomic`, never C `volatile`.
- **Do not reach for `relaxed`** unless you can prove no other thread's correctness depends on an ordering with this access (a pure counter for metrics is the classic safe case). "It seemed faster" is not a proof; it is a future incident.
- **Always test the weak model.** If your code targets ARM/POWER/RISC-V at all, you cannot validate synchronization on x86 alone. Run your concurrency tests on the weak architecture, or model the critical sections with `herd7`/`litmus7`. x86 passing tells you nothing about ARM.
- **Default to "synchronize, then optimize."** Write the obviously-correct locked or channel-based version first, measure, and only descend to the memory model if the measurement demands it. Most code never needs to.

## Key takeaways

1. **The as-if rule is the only promise.** The compiler preserves *single-thread* observable behavior and nothing else; a data race steps outside that promise (undefined behavior in C/C++, weakly-defined in Java/Go). Source order is a fiction for any second observer.
2. **The compiler reorders, caches, and deletes.** Hoisting a loop-invariant load into a register is the single most common "memory model" bug — your spin loop polls a register, not memory. A plain read in a loop is not a memory poll.
3. **The store buffer is the universal hardware reorderer.** A core sees its own writes immediately (store-to-load forwarding) but other cores see them tens of cycles later, producing Store→Load reordering on *every* mainstream CPU, including x86.
4. **The "both read 0" litmus is impossible under sequential consistency and routine on real hardware.** It is the cleanest proof that the store buffer reorders, and the reason Dekker/Peterson need fences.
5. **Model strength is a spectrum and x86 is a liar.** x86-TSO allows only Store→Load reordering; ARM/POWER allow all four. Code that "works" on x86 can detonate on ARM — the bug was always there, hidden by the strong model.
6. **A broken publish is up to four reorderings.** Fix it with a release store paired with an acquire load (Java `volatile`, C++ `std::atomic` + `memory_order_release/acquire`, Rust `Ordering`, or a Go channel). The release/acquire pair transports the plain payload's visibility for free.
7. **Different layers need different brakes.** Atomics/`volatile` stop the compiler; fences stop the core and drain the store buffer; coherence is the floor, not a foe. An atomic's memory order requests exactly the right combination per architecture.
8. **You can ignore all of this when you are already synchronized** — single-threaded, locked, channel-based, or using concurrent collections. Establish happens-before explicitly only when you hand-roll synchronization, and prove it before using `relaxed`.

## Further reading

- **Preshing, "Memory Reordering Caught in the Act"** and **"This Is Why They Call It a Weakly-Ordered CPU"** — the clearest hands-on demonstrations of the store-buffer litmus on real x86 and ARM, with runnable code. Start here.
- **Preshing, "Acquire and Release Semantics"** and **"An Introduction to Lock-Free Programming"** — the release/acquire pair and how it builds the happens-before edge, exactly the fix in this post.
- **Sewell, Sarkar, Owens, et al., "x86-TSO: A Rigorous and Usable Programmer's Model for x86 Multiprocessors"** (CACM 2010) — the formal store-buffer model behind everything in the x86 section.
- **Alglave, Maranget, Sarkar, Sewell, "Understanding POWER Multiprocessors"** and the **herd7 / litmus7 toolsuite** — the formal weak-memory models and the tools used for the measurements here.
- **Bacon et al., "The 'Double-Checked Locking is Broken' Declaration"** and **JSR-133, the Java Memory Model FAQ (Manson, Goetz)** — the canonical case study and the model that fixed it.
- **Goetz et al., *Java Concurrency in Practice*** (ch. 16, the JMM) and **Williams, *C++ Concurrency in Action*** (ch. 5, the C++ memory model) — the practitioner books for the two languages here.
- **Herlihy & Shavit, *The Art of Multiprocessor Programming*** — sequential consistency, linearizability, and why the litmus outcomes matter for correctness proofs.
- Within this series: [memory models, sequential consistency, and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) (the contract this post motivates) and [memory barriers, acquire/release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences) (the tools that fix it); the spine in [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition); and the hardware backdrop in [the memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).

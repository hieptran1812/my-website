---
title: "Memory Models: Sequential Consistency and Happens-Before"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The contract between you, the compiler, and the hardware about which writes a read can observe, and why a data-race-free program runs as if perfectly ordered."
tags:
  [
    "concurrency",
    "parallelism",
    "memory-model",
    "sequential-consistency",
    "happens-before",
    "data-race-free",
    "java-memory-model",
    "atomics",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/memory-models-sequential-consistency-and-happens-before-1.png"
---

Two threads. One has a piece of data and a flag. It writes the data, then sets the flag to say "the data is ready." The other thread spins until it sees the flag, then reads the data. This is the most ordinary pattern in all of concurrent programming — publish-then-consume, the handshake at the heart of every queue, every future, every lazy initializer, every double-checked lock. You have written it a hundred times without thinking.

Here is the uncomfortable question that this entire post exists to answer: **when the reader sees the flag set, is the data guaranteed to be there?** Your intuition screams yes — of course it is, the writer set the data *first*. But your intuition is running a model of the machine that the machine does not implement. The compiler is allowed to reorder those two writes. The CPU is allowed to make them visible to other cores in a different order than you issued them. The reader's CPU is allowed to speculate and read the data before it reads the flag. On a real multicore machine, with a real optimizing compiler, the reader can see the flag set and the data *still stale* — zero, garbage, the value from before. The handshake silently breaks, maybe one time in ten million, and you get a corrupted balance, a half-initialized object, a crash in a place that makes no sense.

A **memory model** is the contract that decides whether that question has a safe answer. It is the single most important abstraction in concurrent programming, and almost nobody is taught it explicitly. It is the rulebook — written jointly by the language designers, the compiler writers, and the hardware architects — that says: *given a program with these threads and these reads and writes, here is exactly the set of values each read is allowed to return.* Get the model right and your handshakes hold. Get it wrong and you are debugging a "one-in-a-billion" ghost that no debugger will ever stop on.

![Sequential consistency reads in program order while a weak model reorders the writes so the reader sees a stale value](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-1.png)

In this post we build the model from the ground up. We start with **sequential consistency** — the beautifully simple ideal where everything happens in one global order, the model your intuition already runs — and we show why no mainstream CPU gives it to you for free. Then we build the model you actually program against: the **happens-before** relation, defined formally as program order stitched together by **synchronizes-with** edges across threads. We prove the one theorem that makes concurrency tractable for working engineers — **DRF-SC**: if your program has no data races, it behaves as if it were sequentially consistent, full stop. That theorem is the lens for everything that follows in this series — [memory barriers](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences), [atomics and memory orderings](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), and lock-free data structures all reduce to *which happens-before edges do I have, and do they cover every access to my shared state?* If you have read [why your code doesn't run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering), you already know the machine reorders; this post tells you the rules of that game and how to win.

## What a memory model is, and why you cannot avoid one

Let us define the object precisely, because the word "model" is doing a lot of work. A **memory model** is a specification that answers exactly one question: *for a multithreaded program, what values is each read allowed to return?* That is it. It does not tell you how the hardware is built. It does not tell you which optimization the compiler applied. It draws a fence around the set of legal outcomes and says "anything inside this fence is a correct execution; your program must tolerate every point inside it."

You might think you could avoid this. You write `x = 1` and then `y = x`, and surely `y` is `1`. In a single thread, yes — every language guarantees that a thread sees its own operations in program order. This is the **as-if-serial** rule, and it is the only thing that lets you reason about sequential code at all. The compiler and CPU may reorder, eliminate, and fuse your operations however they like, as long as a single thread cannot tell the difference. That last clause is the catch. *A single thread* cannot tell. **Another thread can.** The moment a second thread observes your memory, all those invisible reorderings become visible, and "obvious" facts about ordering evaporate.

So a memory model exists at two levels, and they compose. There is the **hardware memory model** — the rules the CPU promises about how one core's stores become visible to other cores. There is the **language memory model** — the rules your programming language promises, which the compiler must preserve when it translates your code down to instructions. Your code talks to the language model; the language model is implemented on top of the hardware model. When the two disagree, the compiler inserts fences and special instructions to bridge the gap, so that the guarantees you were promised at the language level actually hold on the silicon underneath.

Why can't the hardware just keep everything in order? Performance, by orders of magnitude. A load that misses cache costs roughly two hundred cycles — hundreds of times slower than an arithmetic instruction. If every core had to wait for every store to become globally visible before doing anything else, multicore machines would crawl. So CPUs use **store buffers**: a store goes into a small per-core queue and the core moves on immediately, draining the buffer to cache in the background. The core sees its own store right away (it checks its own buffer), but other cores do not see it until it drains. That single optimization — present on every modern CPU including x86 — is enough to break the flag-and-data handshake. The compiler adds a second layer: it freely reorders independent memory operations, hoists loads out of loops, sinks stores, and caches values in registers, because the as-if-serial rule says it may. We covered those mechanisms in depth in [why your code doesn't run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering); here we care about the *rules*, not the mechanism.

One sharp distinction before we go further, because the whole series turns on it. A **data race** is a precise, technical thing: two threads access the same memory location, at least one access is a write, and there is no happens-before edge ordering them. A **race condition** is a looser, design-level bug: your program's correctness depends on timing in a way you did not intend. They are not the same — a program can have a race condition with no data race, and the memory model only speaks about data races. We draw the full line in [data races versus race conditions](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction); for now, hold onto this: *a data race is the thing the memory model declares either undefined or merely weakly-ordered, and avoiding it is the price of getting the simple model back.*

## Sequential consistency: the model your intuition already runs

Leslie Lamport defined **sequential consistency** in a 1979 paper with a sentence so precise it is worth quoting in spirit. An execution is sequentially consistent if the result is the same as if all operations of all processors were executed in *some single sequential order*, and the operations of each individual processor appear in this sequence in the order specified by its program.

Unpack that into two clauses, because both matter:

1. **There is one global order.** Every read and every write from every thread can be laid out on a single timeline, one operation after another, with nothing happening "at the same time." Pick up all the operations, shuffle them into one line.
2. **Each thread's program order is preserved within that line.** When you shuffle, you may interleave threads however you like, but you may never reorder two operations *from the same thread*. If thread T1 does A before B in its source code, A comes before B in the global line.

That is the whole definition. A read returns the value of the *most recent* write to that location in the global order. There is no store buffer, no caching, no reordering visible to anyone — it is as if all threads share one memory and take turns touching it, one operation at a time.

Let us make this completely precise, because the formal statement is the thing every later definition is measured against. Let an execution consist of a set of memory operations $O$, where each operation is a read or a write to some location. Each thread imposes a total **program order** $<_{p}$ on its own operations. Sequential consistency asks for a single total order $<_{s}$ over *all* of $O$ — a linearization — satisfying two constraints. First, **consistency with program order**: for any two operations $a, b$ from the same thread, $a <_{p} b \implies a <_{s} b$. Second, the **value rule**: every read $r$ of location $\ell$ returns the value written by the write $w$ to $\ell$ that is the latest in $<_{s}$ among all writes to $\ell$ that precede $r$ in $<_{s}$ (or the initial value if there is none). If at least one such total order $<_{s}$ exists that produces the observed results, the execution is sequentially consistent. If no linearization can produce them, the execution is not SC — and that "no linearization exists" is exactly the cycle argument we will use to rule out $(0,0)$ below.

Two things about this definition trip people up, so name them now. It does **not** say there is real-time agreement on a global clock — SC is about the *existence* of some consistent order, not about wall-clock simultaneity, which is why it is strictly weaker than **linearizability** (the distributed-systems cousin that additionally pins each operation to its real-time interval). And it does **not** forbid the hardware from reordering internally; it forbids any reordering that another thread can *observe*. A store buffer that no other thread ever queries would be invisible and therefore SC-legal. The trouble is that on a multicore machine, another thread always can query, and that is where the model bites.

![A single global sequence interleaves both threads while preserving each thread's program order](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-2.png)

This is exactly the model you have been running in your head. When you reason "T1 writes data, then sets flag; T2 sees flag, then reads data — so T2 reads the data T1 wrote," you are reasoning under sequential consistency. The global order is `data=42, flag=1, (T2) read flag=1, (T2) read data=42`, and because the data write precedes everything T2 does, T2 must read 42. Under SC, the handshake is airtight, by definition.

#### Worked example: enumerating the legal outcomes under SC

Take the classic two-thread program. Two shared variables `x` and `y`, both starting at 0. T1 runs `x = 1; r1 = y`. T2 runs `y = 1; r2 = x`. After both finish, what are the possible `(r1, r2)` pairs under sequential consistency?

We enumerate the interleavings. There are four operations: call them `Wx` (T1 writes x), `Ry` (T1 reads y), `Wy` (T2 writes y), `Rx` (T2 reads x). T1's order is `Wx` before `Ry`; T2's order is `Wy` before `Rx`. Any global order that respects both gives a legal SC outcome:

- `Wx, Ry, Wy, Rx` → T1 reads y before T2 wrote it: `r1 = 0`. T2 reads x after T1 wrote it: `r2 = 1`. Result `(0, 1)`.
- `Wx, Wy, Ry, Rx` → both writes first: `r1 = 1`, `r2 = 1`. Result `(1, 1)`.
- `Wy, Rx, Wx, Ry` → symmetric to the first: `(1, 0)`.
- `Wy, Wx, Rx, Ry` → both writes first: `(1, 1)`.

Three outcomes are possible under SC: `(0, 1)`, `(1, 0)`, `(1, 1)`. The one outcome that is **impossible** under SC is `(0, 0)` — both threads reading 0. The proof is the cycle argument from the formal definition, written out. Suppose, for contradiction, that some SC linearization $<_{s}$ produced $(0,0)$. For `r1 = 0`, the read `Ry` saw the initial value, so `Ry` must precede `Wy` in $<_{s}$: `Ry <_s Wy`. For `r2 = 0`, symmetrically `Rx <_s Wx`. Program order forces `Wx <_s Ry` (T1) and `Wy <_s Rx` (T2). Chain them: `Wx <_s Ry <_s Wy <_s Rx <_s Wx`. That is a cycle — `Wx` precedes itself — and a total order admits no cycle. Contradiction. So no SC linearization yields $(0,0)$; it is forbidden.

Hold that result. `(0, 0)` is the **store-buffer litmus test**, and the punchline of the next section is that real hardware *does* produce `(0, 0)`, which is the most direct proof you will ever see that the machine is not sequentially consistent.

#### Worked example: the message-passing litmus under SC

The other litmus you must know is **message passing (MP)**, because it is the publish-then-consume handshake from the intro stripped to four operations. Shared `data` and `flag`, both 0. T1 (the writer) does `data = 42` then `flag = 1`. T2 (the reader) does `r1 = flag` then `r2 = data`. The dangerous outcome is `r1 = 1, r2 = 0` — the reader saw the flag but read stale data. Is it possible under SC?

Label the operations `Wd` (write data), `Wf` (write flag), `Rf` (read flag), `Rd` (read data). Program order: `Wd <_s Wf` and `Rf <_s Rd`. For `r1 = 1`, the flag read saw the flag write, so `Wf <_s Rf`. Now chain: `Wd <_s Wf <_s Rf <_s Rd`. By transitivity `Wd <_s Rd`, and `Wd` is the only write to `data`, so `Rd` must return 42, never 0. Under SC, `r1 = 1` **implies** `r2 = 42` — the handshake is airtight, exactly as the intro promised. The forbidden outcome `(1, 0)` requires `Rd` to precede `Wd`, but we just derived `Wd <_s Rd`; contradiction. This is the litmus we will *break* on real hardware: MP failing — flag seen, data stale — is the most common shape of a real publication bug, and it is the one the release-acquire edge exists to prevent.

## Why sequential consistency is too expensive to ship

Sequential consistency is so intuitive that you would expect every machine to implement it. None of the mainstream ones do — not x86, not ARM, not POWER, not RISC-V — and the reason is cost. To give you SC, the hardware would have to make *every single memory access* a synchronization point. Every store would have to become globally visible before the next operation could proceed. Every load would have to be ordered against every other access. In practice that means a memory fence — an instruction that stalls the pipeline until the store buffer drains and outstanding memory traffic settles — on essentially every load and store.

Put a number on it. A store that hits L1 cache is effectively free in steady state because it goes into the store buffer and the core keeps running. Force that store to be globally visible before the core continues — which is what SC demands — and you pay the drain. Depending on the microarchitecture and the state of the memory system, a full barrier costs on the order of tens of nanoseconds; an uncontended atomic read-modify-write on x86 sits in the same neighborhood. That is not a precise figure for your machine — it varies with the chip, the cache state, and the surrounding code, so treat it as an order of magnitude — but the shape is unambiguous: tens of nanoseconds is **tens of times** the cost of an ordinary store. Programs touch memory billions of times a second. Multiply a tens-of-times penalty across every access and your fast multicore machine performs like a slow single-core one. The optimizations the hardware uses — store buffers, out-of-order execution, speculative loads, write combining — are exactly the things that make a modern CPU fast, and every one of them is a violation of sequential consistency that some other core can, in principle, observe.

![Sequential consistency orders every access at a barrier cost while a weak model stays cheap and recovers SC only for race-free code](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-5.png)

The compiler makes the same trade for the same reason. Sequential consistency at the language level would mean the compiler could *never* reorder two memory operations that another thread might observe, never cache a shared variable in a register across a possible observation point, never hoist a load out of a loop if a concurrent writer exists. Those are the bread-and-butter optimizations that make compiled code fast. Forbidding them globally — on every variable, just in case it is shared — would cripple single-threaded performance, which is the overwhelming majority of all code. No compiler vendor will pay that tax on the 99% to make the 1% of genuinely-shared accesses safe.

So both layers made the same engineering decision: **default to fast and weak, and provide explicit, opt-in tools to recover ordering exactly where the programmer needs it.** Plain reads and writes are unordered across threads — cheap. When you need ordering, you ask for it with a lock, an atomic with a memory ordering, or a fence, and you pay the cost only at those specific points. The rest of this post is about the rules of that opt-in: what "asking for ordering" formally means, and what you get when you do.

This is the same shape of trade you see one level down in [how a lock is built from test-and-set and CAS](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks): the atomic instruction that implements the lock is also the fence that gives you the ordering. The lock and the memory ordering are two faces of the same hardware primitive.

## The happens-before relation, defined formally

If the machine will not give you a single global order for free, what does it give you? It gives you a **partial order** — a relation that orders *some* pairs of operations and leaves the rest unordered. That relation is **happens-before**, written `A ⟶ B` ("A happens-before B"). It is the central definition of every modern memory model — the C++11 model, the Java Memory Model, the Go model, the Rust model — and once you have it, everything else is bookkeeping.

Happens-before is built from two ingredients, combined and closed under transitivity:

1. **Program order.** Within a single thread, if operation A comes before operation B in the source, then `A ⟶ B`. This is the within-thread spine — it is what the as-if-serial rule guarantees, lifted into the formal relation.
2. **Synchronizes-with.** Across threads, certain paired operations create an edge. A release operation in one thread **synchronizes-with** the acquire operation in another thread that reads the value the release wrote. The canonical pair: an `unlock` of a mutex synchronizes-with the next `lock` of that same mutex; an atomic **store-release** synchronizes-with the atomic **load-acquire** that reads its value.

Then the master rule: **happens-before is the transitive closure of program order and synchronizes-with.** If `A ⟶ B` and `B ⟶ C`, then `A ⟶ C`. That transitivity is the engine. A synchronizes-with edge connects one operation in T1 to one operation in T2; transitivity then propagates the ordering to *everything before the release in T1 and everything after the acquire in T2.*

State the formal object precisely. Write program order as $\xrightarrow{po}$, synchronizes-with as $\xrightarrow{sw}$, and happens-before as $\xrightarrow{hb}$. The C++11 definition is the clean one: $\xrightarrow{hb}$ is the smallest relation containing $\xrightarrow{po}$ and $\xrightarrow{sw}$ that is **transitively closed**. Formally, $a \xrightarrow{hb} b$ iff $a \xrightarrow{po} b$, or $a \xrightarrow{sw} b$, or there exists some $c$ with $a \xrightarrow{hb} c$ and $c \xrightarrow{hb} b$. (The real standard threads in a third ingredient, *dependency-ordered-before*, to support the `consume` ordering — but `consume` is the ordering almost nobody should use, so the program-order-plus-synchronizes-with picture is the one to carry.) This relation has exactly the three properties of a **strict partial order**: it is **irreflexive** (no operation happens-before itself — guaranteed because a real execution has no causal cycle, the same acyclicity we leaned on for SC), **transitive** by construction, and therefore **antisymmetric** (you cannot have both $a \xrightarrow{hb} b$ and $b \xrightarrow{hb} a$). "Partial" is the load-bearing word: unlike SC's *total* order, happens-before deliberately leaves most cross-thread pairs **unrelated** — and an unrelated conflicting pair is precisely a data race.

![Program order plus one release to acquire edge makes the writer's data write happen-before the reader's data read by transitivity](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-3.png)

Read that figure as a directed acyclic graph, because that is literally what happens-before is — a DAG over the operations of an execution. T1's two operations are chained by program order: `write data ⟶ store-release flag`. T2's two operations are chained by program order: `load-acquire flag ⟶ read data`. The one cross-thread edge is the synchronizes-with: `store-release flag ⟶ load-acquire flag`, which exists precisely *because the acquire read the value the release wrote*. Now follow the transitive path: `write data ⟶ store-release ⟶ load-acquire ⟶ read data`. By transitivity, `write data ⟶ read data`. The data write happens-before the data read. The handshake holds — not by intuition, but by a chain of edges you can point at.

This is the whole game. To make a write visible to a reader, you do not need a global order. You need a **path** in the happens-before graph from the write to the read. Program order gives you the within-thread segments for free; you supply the one cross-thread synchronizes-with edge with a lock release-acquire or an atomic release-acquire. Transitivity stitches them into a path.

And note what is *not* ordered. If two operations have no happens-before path between them — neither `A ⟶ B` nor `B ⟶ A` — they are **concurrent**. Concurrent operations are unordered, and if they touch the same location with at least one write, that is a data race. Happens-before is a *partial* order precisely because most pairs of operations across threads are concurrent. You order only the pairs you must.

## The edge makers: where synchronizes-with comes from

Program order you get automatically. The interesting question is: which concrete operations create a synchronizes-with edge across threads? The list is short and worth memorizing, because every correct concurrent program is built from these edges and nothing else.

![Happens-before edges come from program order within a thread plus three synchronizes-with sources across threads](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-4.png)

**Lock release synchronizes-with the next lock acquire of the same lock.** When thread T1 unlocks a mutex and thread T2 later locks the same mutex, `unlock ⟶ lock`. Everything T1 did before releasing the lock happens-before everything T2 does after acquiring it. This is why a mutex gives you visibility, not just mutual exclusion — the critical sections are totally ordered, and each one sees the previous one's writes. People assume a lock is only about keeping threads out of each other's way; the deeper truth is that the lock's release-acquire pair is what makes the protected data *visible* across the boundary.

**An atomic store-release synchronizes-with an atomic load-acquire that reads its value.** This is the lock-free version of the same edge, and it is the one you reach for when you want the ordering without the mutual exclusion. We go deep on the orderings in [atomics and memory orderings](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) and on the fences themselves in [memory barriers, acquire, release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences); the one fact to carry now is that a release "pushes" everything before it and an acquire "pulls" everything after it, and the two snap together into an edge only when the acquire actually observes the released value.

**Thread start and thread join.** When a thread spawns another, the spawn happens-before the first operation of the new thread — the child sees everything the parent did before launching it. When a thread joins another, the last operation of the joined thread happens-before the join returns — the parent sees everything the child did. These two edges make `start` and `join` natural synchronization points, which is why handing data to a freshly-spawned thread or collecting a result after `join` needs no extra locking.

**A `volatile` write synchronizes-with a `volatile` read in Java** (and the equivalent in other models). In the Java Memory Model, a write to a `volatile` field synchronizes-with every subsequent read of that field — `volatile` in Java is precisely a release-store / acquire-load pair, which is why the famous broken double-checked-locking pattern is *fixed* by making the field `volatile`. C and C++ `volatile` is a different and weaker beast — it controls device-register access, not inter-thread ordering — so do not transfer the Java intuition to C++.

There are a few more (the relationship between a thread's interruption status, final-field freezes in Java's constructor semantics, the consume ordering in C++ that almost nobody should use), but those five cover essentially all correct concurrent code you will write. Every one is a *paired* operation: a release-side and an acquire-side that must rendezvous on the same lock, the same atomic, or the same thread boundary. There is no such thing as a one-sided synchronizes-with edge — if only one side is synchronized, you have a data race.

#### Worked example: spotting the data race in a three-thread trace

Reasoning about edges is a skill, so drill it on a trace that mixes a real edge with a missing one. Shared state: a plain `int balance = 100`, a plain `int log = 0`, and a mutex `m`. Three threads run concurrently.

- **T1**: `lock(m); balance = balance - 30; unlock(m);`
- **T2**: `lock(m); balance = balance - 50; unlock(m);`
- **T3**: `int snapshot = balance; log = snapshot;` — *no lock at all.*

Walk the edges. Between T1 and T2, the mutex does its job: whichever runs first, its `unlock(m)` synchronizes-with the other's `lock(m)`, so the two `balance` writes are happens-before-ordered. That pair is **not** a race — `balance` ends at `100 - 30 - 50 = 20` regardless of order, and each critical section sees the other's write. Good.

Now T3. It reads `balance` and writes `log` with no lock. Pair T3's `read balance` against T1's `write balance`: are they ordered by happens-before? There is no `unlock ⟶ lock` edge connecting them, no atomic, no thread start/join between T3 and T1 — they are **concurrent**. They conflict (same location `balance`, one is a write). Concurrent plus conflicting equals a **data race**. The `snapshot` T3 reads is undefined: it might be 100, 70, 50, or 20 depending on when its unsynchronized read landed, and in C++ it is undefined behavior outright. The fix is to make T3 take the lock too — `lock(m); snapshot = balance; unlock(m);` — which threads it into the same total order as T1 and T2, at which point `snapshot` is guaranteed to be one of the *committed* values (100, 70, or 20). The lesson the trace teaches: **a single unsynchronized accessor poisons an otherwise-correct design.** It is not enough for *most* accesses to take the lock; happens-before is required across *every* conflicting pair, and the one reader that skipped the lock is the whole bug. This is why "the mutex protects `balance`" must mean *every* touch of `balance`, with no exceptions, or DRF-SC does not apply.

## Transitivity and the visibility rule

The reason happens-before is *useful* and not merely a definition is the **visibility guarantee** that rides on top of it:

> If `A ⟶ B`, then the effects of A are visible to B. Concretely: if A writes a value to a location and `A ⟶ B`, then B (and anything after B in happens-before) is guaranteed to see that write, not an older value — unless some other write to that location is also ordered after A and before B.

That is the payoff. Happens-before is not just an abstract ordering; it is a *visibility* contract. An edge from a write to a read means the read sees the write. No edge means no guarantee — the read may see the write, or it may see a stale value, and which one you get is up to the scheduler, the cache state, and the phase of the moon. Unordered means undefined-in-practice.

Transitivity is what makes this powerful with a single edge. You do not have to synchronize every variable. You synchronize *one* thing — the flag, the lock, the atomic — and transitivity carries the visibility of *everything before the release* across to *everything after the acquire*. This is sometimes called the "release sequence" or the "publishing" pattern: pack all your data writes before a single release, and a single acquire on the other side makes all of them visible at once. One edge, many writes published.

#### Worked example: tracing visibility through transitivity

Concrete trace. Shared state: an integer `data`, an integer `aux`, and an atomic boolean `ready`, all initially 0/false. T1 does, in order: `(1) data = 42`, `(2) aux = 7`, `(3) ready.store(true, release)`. T2 does, in order: `(4) while(!ready.load(acquire)) {}`, `(5) read data`, `(6) read aux`.

Build the edges. Program order in T1: `1 ⟶ 2 ⟶ 3`. Program order in T2: `4 ⟶ 5 ⟶ 6`. The synchronizes-with edge: operation 3 is a store-release and operation 4 is the load-acquire that finally reads `true` — so `3 ⟶ 4`. Now run transitivity. From `1 ⟶ 2 ⟶ 3 ⟶ 4 ⟶ 5` we get `1 ⟶ 5`: the `data = 42` write happens-before the `read data`, so operation 5 reads 42. From `2 ⟶ 3 ⟶ 4 ⟶ 6` we get `2 ⟶ 6`: the `aux = 7` write happens-before the `read aux`, so operation 6 reads 7. **Both** writes are visible, through the single edge at the flag, even though `data` and `aux` are plain non-atomic variables that were never themselves synchronized.

That is the engineering lesson in one trace: you synchronize the *handshake variable*, and everything you wrote before publishing it comes along for free. The data does not need to be atomic; the *publication* does.

One subtlety the C++ model adds here, because it generalizes the single-edge picture, is the **release sequence**. When a thread does a store-release on an atomic, and then *other* atomic read-modify-writes (or relaxed stores by the same thread) chain off that value, the synchronizes-with edge attaches to the *acquire that reads any link in that chain*, not only the original release. Concretely: a producer does `flag.store(1, release)`, several worker threads each do `flag.fetch_add(1, acq_rel)` to claim a slot, and a consumer does `flag.load(acquire)` — the consumer still synchronizes-with the *original* release through the chain of read-modify-writes, so everything written before the producer's release is visible to the consumer even though it read a value several increments later. You rarely construct release sequences by hand, but they are why reference-counting and ticket-lock patterns built on `fetch_add` publish correctly. The takeaway is the same one, generalized: it is not the *value* that carries visibility, it is the *acquire observing a write in the release's chain* — the edge is about the synchronization event, not the bits.

## DRF-SC: race-free programs behave as if sequentially consistent

Now the theorem that the whole field rests on. We lost sequential consistency for performance. The happens-before machinery gives us a way to recover it — not globally, but exactly where we establish edges. The **Data-Race-Free / Sequential-Consistency** guarantee, formalized by Sarita Adve and Mark Hill in 1990 and made the foundation of the C++ and Java models by Hans Boehm, Sarita Adve, and the JSR-133 group, states it crisply:

> **DRF-SC.** If a program has no data races — that is, every pair of conflicting accesses (same location, at least one a write) is ordered by happens-before — then every execution of that program is sequentially consistent.

Read what that buys you. If you are disciplined enough to put a happens-before edge between every pair of conflicting accesses, the memory model *promises* you that the program behaves as if it ran on the simple, intuitive, single-global-order machine. The weak hardware, the store buffers, the compiler reorderings — all of it disappears from your reasoning. You get to think in interleavings again. **You get your intuition back, on the sole condition that you have no data races.**

This is the load-bearing wall of practical concurrency. It is why the advice "just use a mutex" works: a mutex orders every access to the data it protects with `unlock ⟶ lock` edges, so a correctly-locked program is data-race-free, so by DRF-SC it is sequentially consistent, so you can reason about it as plain interleavings of critical sections without ever thinking about the hardware memory model. The mutex is not magic; it is a happens-before edge factory, and DRF-SC is the theorem that turns those edges into the simple mental model.

#### Worked example: why race-free implies sequentially consistent

The theorem deserves a proof sketch, because seeing *why* it holds turns it from a slogan into a tool you trust. The argument is by contradiction and it is short. Suppose a program is data-race-free under *every* SC execution, yet some real (weakly-ordered) execution $E$ produces a result that no SC execution could. We want to show this is impossible.

Take that real execution $E$ and try to build an SC linearization for it. Order the operations by happens-before, breaking ties arbitrarily but consistently, into a candidate total order $<_{s}$. Two operations can fail to extend cleanly only if they **conflict** (same location, at least one a write) and are **unordered** by happens-before — because for non-conflicting or already-ordered pairs, any topological order of the happens-before DAG gives the same observable values. But "conflicting and unordered by happens-before" is the *definition of a data race*. By assumption the program has none. So every conflicting pair is happens-before-ordered, every read in $E$ sees the most-recent-write according to happens-before, and that order *is* a valid SC linearization producing exactly $E$'s results. Hence $E$ was sequentially consistent after all — contradicting that it produced a non-SC result. The single hinge of the whole proof is that **the only thing that can make a weak execution diverge from SC is a conflicting, unordered access** — kill those (be race-free) and there is nothing left to diverge. That is DRF-SC, and the contrapositive is its teeth: the moment you have one unordered conflicting pair, the proof collapses and the model owes you nothing.

The compiler and hardware uphold their half by construction: each is allowed to reorder, buffer, and speculate freely *as long as* it preserves the happens-before edges the program established (the fences a `store-release` lowers to are exactly what stop a weak CPU from reordering across the edge). Race-free code only depends on happens-before order, so every legal reordering is invisible to it — which is the same statement as DRF-SC, viewed from the implementer's side rather than the prover's.

The contrapositive is the warning. If your program *has* a data race, DRF-SC says nothing. In C++, Go, and Rust, a data race on ordinary memory is **undefined behavior** — the compiler is permitted to assume it never happens and may "optimize" your program into nonsense. In Java, a data race is not undefined behavior (the JMM was deliberately built to keep the JVM memory-safe even for racy code), but the values a racy read can return are governed by weak, surprising rules that no human reasons about correctly. Either way: a data race takes you outside the simple model. The entire discipline of concurrent programming reduces to **stay inside DRF-SC** — name your shared mutable state, put a happens-before edge across every conflicting access, and then reason as if the world were sequentially consistent.

![Without a happens-before edge the reader may observe a stale value but adding a release-acquire pair forces the published write to be visible](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-6.png)

## Establishing the edge in code, in two languages

Enough theory. Here is the flag-and-data handshake done wrong and then right, in the languages where the idioms diverge. We will show the bug, name why it is a data race, and fix it with each language's edge-maker.

### The bug, in C++

This is the canonical broken publisher. `data` is a plain `int`, `ready` is a plain `bool`. The producer writes the data then sets the flag; the consumer spins on the flag then reads the data.

```cpp
#include <thread>
#include <cassert>

int data = 0;       // plain, shared
bool ready = false; // plain, shared — this is the bug

void producer() {
    data = 42;      // (1)
    ready = true;   // (2)
}

void consumer() {
    while (!ready) { /* spin */ } // (3)
    assert(data == 42);           // (4) can FAIL
}
```

Two data races here: `data` is written by the producer and read by the consumer with no ordering, and `ready` is written and read with no ordering. Because there is no happens-before edge between operation (2) and operation (3), there is no edge from (1) to (4), so the consumer is not guaranteed to see `data == 42`. Worse, in C++ this is undefined behavior — the optimizer may hoist `ready` into a register and spin forever, or assume the race cannot happen and delete the loop. The assert can fail, or the program can hang, or anything.

The fix is to make `ready` an atomic with release-acquire ordering. That converts operation (2) into a store-release and operation (3) into a load-acquire, creating the synchronizes-with edge — and `data` can stay a plain `int`, because transitivity carries it across.

```cpp
#include <atomic>
#include <thread>
#include <cassert>

int data = 0;                       // still plain — that's fine
std::atomic<bool> ready{false};     // the synchronizer

void producer() {
    data = 42;                                   // (1)
    ready.store(true, std::memory_order_release); // (2) release
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) { } // (3) acquire
    assert(data == 42);                                // (4) GUARANTEED
}
```

Now `(2) store-release ⟶ (3) load-acquire`, and program order gives `(1) ⟶ (2)` and `(3) ⟶ (4)`. Transitivity: `(1) ⟶ (4)`. The assert cannot fail. The release-acquire pair is the edge; `data` rides across it for free.

### The same edge, in Java

In Java the idiomatic edge-maker for this exact pattern is `volatile`. A `volatile` write is a release; a `volatile` read is an acquire. So marking `ready` as `volatile` does the same job as the C++ atomic release-acquire — and again, `data` can stay a plain field, because the happens-before edge through `ready` carries it.

```java
class Handshake {
    int data = 0;              // plain field
    volatile boolean ready = false; // volatile: the synchronizer

    void producer() {
        data = 42;     // (1)
        ready = true;  // (2) volatile write = release
    }

    void consumer() {
        while (!ready) { }       // (3) volatile read = acquire
        assert data == 42;       // (4) GUARANTEED by the JMM
    }
}
```

The Java Memory Model (JSR-133) specifies exactly this: a write to a `volatile` field happens-before every subsequent read of that field, and happens-before is transitive, so the `data = 42` write happens-before the `assert`. Remove the `volatile` keyword and you have a data race; the consumer may spin forever (the JIT can cache `ready` in a register) or read a stale `data`. This is the same bug that made naive double-checked locking infamous before JSR-133, and `volatile` is its fix.

### And with a lock, in Go

If you do not want to think about acquire and release at all, use a lock — the `unlock ⟶ lock` edge does the same work, and in Go the race detector will catch you if you forget. Here the shared state is a classic counter, the running example of this whole series.

```go
package main

import "sync"

type Counter struct {
    mu sync.Mutex
    n  int // shared, protected by mu
}

func (c *Counter) Add() {
    c.mu.Lock()   // acquire: synchronizes-with the previous Unlock
    c.n++         // load-modify-store, now race-free
    c.mu.Unlock() // release: synchronizes-with the next Lock
}

func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.n // sees every Add before this Lock
}
```

Every `Unlock` synchronizes-with the next `Lock` of the same mutex, so all the critical sections on `c.n` form a total order with a happens-before edge between consecutive ones. The program is data-race-free; by DRF-SC it is sequentially consistent; you can reason about `n` as if the increments happened one at a time in some order. Run it under `go test -race` and the detector — which works by tracking the happens-before relation at runtime — stays silent. Strip the lock and `c.n++` becomes the textbook lost-update race we dissect in [the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition), and `-race` lights up immediately.

### No edge, stale read — and why `relaxed` is not enough

It is worth seeing the *failure* spelled out as code, because the bug is what you will actually meet. Here is the handshake with an atomic flag but the **wrong ordering** — `relaxed` on both ends — in Rust. `relaxed` makes each individual access atomic (no torn reads, no undefined behavior) but creates **no synchronizes-with edge**, so it does not order the `data` write against the flag.

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

static mut DATA: i32 = 0; // plain, shared — unsynchronized

fn main() {
    let ready = Arc::new(AtomicBool::new(false));
    let r2 = Arc::clone(&ready);

    let producer = thread::spawn(move || {
        unsafe { DATA = 42; }                 // (1) plain write
        ready.store(true, Ordering::Relaxed);  // (2) relaxed: NO release
    });

    let consumer = thread::spawn(move || {
        while !r2.load(Ordering::Relaxed) {}   // (3) relaxed: NO acquire
        unsafe { println!("{}", DATA); }       // (4) may print 0 — stale!
    });

    producer.join().unwrap();
    consumer.join().unwrap();
}
```

There is no happens-before path from `(1)` to `(4)`. The `relaxed` store and load order *themselves* (the loop is guaranteed to eventually see `true`, by the atomic's own coherence) but they carry nothing else across — `DATA` is left behind. On a weak machine, or under a compiler that hoisted the plain write past the relaxed store, the consumer can observe `ready == true` while `DATA` is still 0. This is the message-passing litmus failing in practice, and it is also why `static mut` plus `unsafe` is required to even *write* this in Rust: safe Rust would not let you share `DATA` mutably across threads without a real synchronizer, so the type system has already steered you away from the bug. Promote both atomics to `Release`/`Acquire` and the edge appears: `(2) ⟶ (3)` synchronizes-with, transitivity gives `(1) ⟶ (4)`, and the read is guaranteed to be 42.

For the Python version of this story — the GIL, why `count += 1` is still not atomic at the bytecode level, and how `threading.Lock` and `asyncio` change the picture — see [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) in the python-performance series, which owns that language's memory story. Here we keep the lens language-agnostic.

## The model hierarchy: from your code down to coherence

It helps to see where the memory model sits in the stack, because the guarantee you actually program against is assembled from several layers, each with its own rules.

![Your code rests on a language memory model promising DRF-SC compiled onto a weaker hardware model that rests on cache coherence](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-7.png)

At the top is **your code** — plain reads, writes, locks, and atomics. Directly below is the **language memory model** (C++11, the JMM, Go's model, Rust's), whose central promise is DRF-SC: race-free code runs as if sequentially consistent. The compiler is responsible for honoring that promise. When it lowers your code to instructions for a weaker machine, it inserts the fences and special instructions needed to manufacture the happens-before edges you asked for. Your `store-release` might compile to nothing extra on x86 (whose TSO model is already strong enough) but to a `dmb` barrier on ARM (whose model is weak). The language model is the portable contract; the compiler bridges it to each target.

Below the language model is the **hardware memory model** — the rules the specific CPU promises. x86 implements **Total Store Order (TSO)**: stores from one core reach other cores in program order, but a store can be reordered after a later load from a different address (the store buffer again — this is exactly the relaxation that lets the `(0, 0)` outcome appear). ARM and POWER are **weakly ordered**: even stores can be observed out of order, and the compiler must emit more barriers to recover the same guarantees. This is why a concurrency bug can be invisible on your x86 laptop and reproduce instantly on an ARM server — the *hardware model changes the set of legal outcomes*, and code that was accidentally-correct on TSO breaks on a weak machine. Always name the platform when you report a memory-ordering measurement.

At the bottom is **cache coherence** — the MESI-family protocol that keeps the multiple cached copies of a single cache line consistent across cores. Coherence is often confused with the memory model, so be precise: **coherence is per-location, the memory model is across locations.** Coherence guarantees that all cores agree on the order of writes to *one* location (you will never see a single variable flicker backward in time). The memory model governs how writes to *different* locations are ordered relative to each other — and that is the hard part, the part coherence does not solve, the part that needs happens-before. A machine can be perfectly coherent and still produce the `(0, 0)` litmus outcome, because that outcome is about the relative order of accesses to two *different* variables.

Make the distinction airtight by stating what coherence actually guarantees, in the standard four axioms (from Sorin, Hill, and Wood, *A Primer on Memory Consistency and Cache Coherence*). For a *single* location: (1) a read by a core returns the value of the last write *by that same core* if no other core wrote in between — the as-if-serial rule, per location; (2) a read returns the value of *another* core's write eventually, once enough time has passed (writes propagate); (3) writes to the *same* location are **serialized** — all cores observe them in one agreed order (this is "write serialization," the no-flicker-backward guarantee); (4) that single per-location order is consistent with program order for each writer. Notice every clause says "the same location." Coherence is a stack of single-variable contracts; it says **nothing** about how a write to `x` is ordered against a write to `y`. The `(0,0)` litmus needs exactly that cross-variable ordering — `Wx` versus `Ry`, where `x` and `y` are different — so a fully coherent machine is free to produce it. The memory consistency model is the layer that picks up where coherence stops: it is the contract *across* locations, and happens-before is the tool it gives you to impose order where you need it. Confusing the two is the single most common conceptual error in this area; the crisp slogan is **coherence is per-address, consistency is the whole address space.**

The lesson of the stack: you write to the language model, the compiler and hardware honor it underneath, and the only contract that crosses all four layers cleanly is "race-free ⇒ sequentially consistent." Stay inside it and you never have to descend the stack. Break it and you are debugging at whichever layer happened to expose the reordering on whichever machine happened to run it.

## Observing an SC violation under a weak model

The cleanest way to *believe* that the machine is not sequentially consistent is to watch the impossible `(0, 0)` outcome happen. Recall the litmus test: two threads, `x` and `y` both 0, T1 does `x=1; r1=y`, T2 does `y=1; r2=x`. Under SC, `(0, 0)` is impossible. On real hardware with plain accesses, it occurs.

Here is a harness you can actually run. The key is to repeat the test millions of times, because the reordering only manifests on the rare interleaving where both store buffers are still holding their store when the other thread's load fires.

```cpp
#include <atomic>
#include <thread>
#include <cstdio>

int x, y, r1, r2;

void run_once() {
    x = y = r1 = r2 = 0;
    std::atomic<bool> go{false};
    auto t1 = std::thread([&]{ while(!go); x = 1; r1 = y; });
    auto t2 = std::thread([&]{ while(!go); y = 1; r2 = x; });
    go = true;            // release both at once to maximize overlap
    t1.join(); t2.join();
}

int main() {
    long long both_zero = 0, iters = 100'000'000;
    for (long long i = 0; i < iters; ++i) {
        run_once();
        if (r1 == 0 && r2 == 0) ++both_zero; // SC says: never
    }
    printf("(0,0) seen %lld times in %lld runs\n", both_zero, iters);
}
```

On an x86 machine, `(0, 0)` shows up — not often, but it shows up, perhaps a handful of times to a few hundred times across a hundred million runs depending on the chip, the thread placement, and how aggressively the store buffers overlap. The exact rate is wildly machine-dependent and timing-dependent; do not trust a precise number, only the *qualitative* fact that the count is greater than zero. That count being nonzero is the proof: the store buffer let `x=1` sit unseen while `r1=y` read the old `y`, and symmetrically on the other thread. Both reads got the stale value. The machine produced an outcome that no sequentially-consistent execution can. (In practice you would spawn the threads once and pin them and loop the body inside, rather than re-spawning per iteration; the spawn overhead in this simplified version makes the window smaller. The point stands either way.)

Now make the two stores **sequentially consistent atomics** — `std::atomic<int>` with the default `memory_order_seq_cst` — and the `(0, 0)` count drops to exactly zero, forever, because `seq_cst` inserts the full barrier that drains the store buffer before the load. You have bought back sequential consistency for those two variables, at the cost of a barrier on each access. That is the entire trade, made visible in one program: weak-and-fast by default, strong-and-slow when you pay for it.

A word on **measuring this honestly**, because litmus tests are notoriously easy to get wrong and report a fabricated number from. Three confounds dominate. First, **thread placement**: if the OS scheduler parks both threads on the *same* physical core (or on two hyperthreads of one core that share a store buffer), the window closes and you see zero — not because the machine is SC, but because the two stores never raced. Pin the threads to distinct physical cores (`taskset`/`pthread_setaffinity_np` on Linux) before you trust a zero. Second, **compiler optimization**: build the litmus at `-O2` or the compiler may keep `x` and `y` in registers and never expose the store-buffer window at all; conversely it may reorder your source in ways that change which litmus you are actually running. Always inspect the emitted assembly for a litmus result you intend to publish. Third, **the spawn-per-iteration confound** in the simplified harness above: thread creation costs microseconds and serializes the threads, shrinking the overlap window by orders of magnitude — a proper harness (the style used by the `herd7`/`litmus7` tools from the Cambridge/INRIA weak-memory group) spawns the worker threads once, synchronizes them on a barrier, and loops the four-operation body millions of times inside. With those three fixed, on a typical x86 box you will see a $(0,0)$ rate somewhere in the range of one-in-thousands to one-in-millions depending on the microarchitecture; the honest report is "nonzero and reproducible," with the *qualitative* claim — TSO is not SC — being the only thing that is platform-independent. Never quote a litmus rate without naming the exact chip, the core pinning, and the compiler flags, because all three move the number by orders of magnitude.

The mirror-image honesty applies to the `seq_cst` run: do not report "zero in a hundred million" as *proof* of correctness. Absence of a litmus failure is weak evidence — the window may simply not have opened on your machine in that run. The strong evidence for the `seq_cst` version is the *argument* (the full barrier orders the store before the load, closing the cycle), confirmed by reading the assembly for the `mfence` or `lock`-prefixed instruction the compiler emitted. Measurement shows you the *weak* model misbehaving; reasoning, not measurement, is what proves the *strong* model correct. That asymmetry — you can observe a bug but you cannot observe its absence — is the deepest practical lesson of the whole memory-model story, and it is why this series keeps insisting on naming the happens-before edge rather than trusting a green test run.

#### Worked example: which outcomes each ordering allows

Put the litmus results in a table so the cost-versus-guarantee trade is concrete. "Reorder window" is the qualitative chance the harness above hits the rare overlap; the numbers are illustrative, not measured constants.

| Configuration | `(0,0)` possible | Cost per access | Why |
| --- | --- | --- | --- |
| Plain `int` on x86 TSO | yes, rare | ~free store | store buffer reorders store-then-load |
| Plain `int` on ARM weak | yes, more often | ~free store | even stores reorder; bigger window |
| `atomic` relaxed | yes | ~free | atomicity only, no ordering across vars |
| `atomic` release/acquire | yes | cheap-ish | release-acquire does not order store-then-load |
| `atomic` seq_cst | no | full barrier | total order over all seq_cst ops |

The subtle row is release/acquire: it still permits `(0, 0)`. Release-acquire orders a write-then-publish handshake (store-release before, load-acquire after), but it does *not* impose a total order strong enough to forbid the store-buffer reordering of a store followed by a load to a different address. Only `seq_cst` forbids `(0, 0)`. This is the single most common misconception about acquire-release, and it is why the [atomics post](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) spends real time on the difference. For the publish-then-consume handshake — which is what you almost always have — release-acquire is exactly right and `seq_cst` is overkill; the `(0, 0)` litmus is one of the few patterns that genuinely needs the stronger ordering.

## How the major languages specify it

Every mainstream systems language adopted the same load-bearing promise — DRF-SC — and they differ mostly in what happens when you *break* it. That difference is not academic; it determines how a memory bug manifests, and therefore how you debug it.

![Each language promises sequential consistency for race-free programs and exposes atomics, differing in what a data race does](/imgs/blogs/memory-models-sequential-consistency-and-happens-before-8.png)

| Language | Race-free guarantee | What a data race does | Atomics / synchronizers |
| --- | --- | --- | --- |
| C++11 and later | DRF-SC | undefined behavior — compiler may do anything | `std::atomic<T>` with `std::memory_order_*` |
| Java (JMM, JSR-133) | DRF-SC | defined but weak — no SC, but memory-safe | `volatile`, `VarHandle`, `java.util.concurrent.atomic` |
| Go | DRF-SC | undefined behavior (but `-race` detects it) | `sync/atomic`, channels, `sync.Mutex` |
| Rust | DRF-SC | cannot compile a data race on safe code | `std::sync::atomic` with `Ordering`, `Arc<Mutex<T>>` |

The C++11 model (formalized by Hans Boehm and Sarita Adve, "Foundations of the C++ Concurrency Memory Model," PLDI 2008) is the reference design. It defines happens-before, the synchronizes-with edges, and the full menu of memory orderings — `relaxed`, `acquire`, `release`, `acq_rel`, `seq_cst`, plus the ill-fated `consume` — and declares any data race undefined behavior. Undefined behavior is brutal but principled: it lets the compiler optimize aggressively on the assumption that races never happen, which is what makes plain accesses free.

The **Java Memory Model**, rebuilt by JSR-133 in 2004 after the original 1995 model was found to be unimplementable and broken, made a different choice. A data race in Java is *not* undefined behavior — the JMM was designed so that even racy programs stay memory-safe and type-safe (you cannot forge a pointer through a race), because Java runs untrusted code and a single race must not corrupt the JVM. So a racy read returns *some* legally-written value, just not necessarily the one you wanted, governed by the model's "causality" rules. JSR-133 is also where `volatile` was given its modern release-acquire semantics and where the broken double-checked-locking idiom was finally fixed.

**Go's** model is the most pragmatic: it specifies happens-before in terms of channel sends/receives, mutex operations, and `sync/atomic`, declares racing programs to have undefined behavior, and then ships a *first-class race detector* (`-race`) that tracks the happens-before relation at runtime and prints the exact two stacks that raced. The cultural message — "if you race, the tool will find it" — has probably prevented more memory bugs than any specification clause.

**Rust** is the outlier and the most interesting. Its memory model for atomics is borrowed directly from C++11 (same orderings, same happens-before). But Rust's ownership system and the `Send`/`Sync` marker traits make a data race on safe code a **compile error** — you cannot share a mutable reference across threads without going through a synchronizer (`Arc<Mutex<T>>`, an atomic, a channel) that the type system has verified. "Fearless concurrency" is precisely this: the borrow checker enforces DRF-SC's precondition (no data races) statically, so the DRF-SC guarantee applies to *all* safe Rust by construction. You opt into the risk only in `unsafe` blocks, where you re-take responsibility for the edges.

**Go's** happens-before specification is worth one extra note because it is unusually readable and because channels are its signature edge-maker. The Go memory model states it directly: a send on a channel happens-before the corresponding receive from that channel completes; and for an *unbuffered* channel, a receive happens-before the send completes (the rendezvous synchronizes in both directions). So `ch <- v` in one goroutine and `x := <-ch` in another is a synchronizes-with edge exactly like a release-acquire — everything the sender did before the send is visible to the receiver after the receive. This is why the Go idiom "don't communicate by sharing memory; share memory by communicating" is not a slogan but a *memory-model strategy*: route your data through a channel and you get the happens-before edge for free, with no atomic or lock in sight. The closing of a channel likewise happens-before a receive that observes the close, which is the standard way to broadcast a cancellation edge to many goroutines at once. Every one of these is the same partial-order machinery wearing a friendlier API.

## Case studies / real-world

Three concrete moments where the memory model stopped being theory.

**JSR-133 and the rebuilt Java Memory Model (2004).** Java shipped in 1995 with a memory model that, by the late 1990s, was understood to be both too weak to support common idioms (the original `volatile` did not order non-volatile accesses around it, so it could not safely publish data) and too strong to permit standard compiler optimizations — it was, in the words of the people who repaired it, "broken." JSR-133, led by Jeremy Manson, Bill Pugh, and Sarita Adve and finalized for Java 5, redefined the model around happens-before, gave `volatile` its release-acquire meaning, defined final-field semantics so a properly-constructed immutable object is safe to publish via a race, and pinned down DRF-SC. The most cited practical consequence: the **double-checked locking** pattern, used everywhere for lazy singletons, was genuinely broken under the old model — a thread could see a non-null reference to a *partially constructed* object — and was fixed by JSR-133's `volatile` semantics, where the `volatile` write of the reference happens-before the `volatile` read, publishing the fully-built object. This is documented in Goetz et al., *Java Concurrency in Practice*, and in the JSR-133 FAQ.

**The C++11 memory model (Boehm and Adve, 2008).** Before C++11, the C++ standard simply did not acknowledge that threads existed; the *de facto* memory model was "whatever your compiler and pthreads implementation happened to do," which meant portable concurrent C++ was technically impossible — you were relying on undocumented behavior. Hans Boehm's paper "Threads Cannot Be Implemented As a Library" (2005) showed *why*: a threads library bolted onto a thread-unaware compiler cannot be made correct, because the compiler is free to introduce races the library cannot see (for example, by speculatively writing to a variable the source never wrote). The fix had to be in the language. C++11 added `std::atomic`, the memory orderings, and the formal happens-before model, finally making portable lock-free C++ well-defined. Every `std::memory_order_acquire` you write is a direct descendant of that work.

**Go's race detector catching real bugs in production code.** Go shipped `-race` (built on Google's ThreadSanitizer, a dynamic happens-before plus lockset analyzer) in 2012, and it immediately found data races in the standard library and in widely-deployed services — races that had been "working" for years because they only manifested under rare schedules or on weakly-ordered hardware. The detector instruments every memory access and maintains a vector clock per goroutine; when it sees two conflicting accesses with no happens-before edge between them, it reports both stacks. The practical lesson, echoed by the [finding concurrency bugs](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing) post, is that you do not find memory-model bugs by reading code or by testing on one machine — you find them by running a happens-before tracker under load, because the human eye cannot see a missing edge.

A fourth, briefer one for flavor: a recurring class of bugs is the **non-`volatile` shutdown flag**. A worker loops `while (!stopRequested) { ... }`; a controller sets `stopRequested = true`. If the flag is a plain field, the JIT (or the C compiler) is entitled to hoist it into a register because, within that thread, nothing visibly changes it — so the loop never sees the update and the worker never stops. The fix is one keyword (`volatile`) or one atomic, establishing the happens-before edge that forces the loop to re-read memory. This bug has shipped in countless codebases and is the smallest possible demonstration that *visibility is not free*.

**The C11 `volatile`-is-not-atomic trap, and the Linux kernel's answer.** A persistent source of production bugs in C and C++ is the assumption — imported from Java — that `volatile` provides inter-thread ordering. It does not. In C and C++, `volatile` was designed for memory-mapped device registers and `setjmp`/signal interaction: it forbids the compiler from eliding or coalescing accesses, but it imposes **no** cross-thread happens-before edge and emits **no** fence, so a `volatile int` shared between threads is still a data race and still reorderable by the CPU. The Linux kernel ran into exactly this and codified the lesson in its memory-ordering documentation (`Documentation/memory-barriers.txt`, largely the work of Paul McKenney and David Howells): the kernel deliberately does *not* trust bare `volatile` for concurrency and instead provides `READ_ONCE`/`WRITE_ONCE` (which prevent the compiler from tearing or inventing accesses, the legitimate job of `volatile`) layered with explicit `smp_rmb`/`smp_wmb`/`smp_mb` barriers and `smp_store_release`/`smp_load_acquire` for the ordering. The kernel even maintains a formal, executable memory model — the **Linux Kernel Memory Consistency Model (LKMM)**, by Alglave, Maranget, McKenney, Parri, and Stern — that the `herd7` tool can run litmus tests against, so a kernel developer can *mechanically check* whether a proposed ordering is sound rather than guess. The portable lesson for application code: never reach for C/C++ `volatile` to synchronize threads; reach for `std::atomic` with the ordering you actually need, and let the compiler emit the right fence for the target.

## When to reach for this (and when not to)

This post is foundational, so the "when to reach for this" question is really "when do I have to think at the memory-model level at all, versus letting a higher-level construct handle it for me?"

**Reach for explicit happens-before reasoning when** you are building a synchronization primitive or a lock-free data structure — a queue, a ring buffer, a sequence lock, a hazard-pointer scheme. There you are constructing the edges by hand with raw atomics, and you must be able to point at the happens-before path for every shared access or you have a bug. Reach for it when you are debugging a "impossible" visibility bug — a flag that "didn't work," a value that's stale on ARM but fine on x86, a shutdown that hangs — because the diagnosis is always "find the missing edge." And reach for it when you are deliberately weakening ordering for performance (using `relaxed` or `acquire-release` instead of the default `seq_cst`), because then you are *proving* a weaker model is still safe, which requires the formal relation, not vibes.

**Do not reach for it — use a higher-level construct instead — when** an ordinary mutex or a channel will do, which is the overwhelming majority of application code. A mutex manufactures the edges for you; DRF-SC then hands you back the simple interleaving model, and you never touch a memory ordering. If you find yourself reaching for `std::atomic` with a hand-picked `memory_order` in business logic, stop and ask whether a lock is actually your bottleneck — measure first; an uncontended mutex is around twenty-five nanoseconds and almost never the thing to optimize away. Do not reason at this level to "go faster" without a measurement showing the lock is the cost. And **never** assume sequential consistency for plain accesses just because your test passed on your x86 laptop — x86's TSO is strong enough to hide many bugs that ARM and POWER expose. "It works on my machine" is, for memory-model bugs, almost a guarantee that it *doesn't* work on someone else's.

The decision rule that subsumes all of the above: **name your shared mutable state, and for every pair of conflicting accesses, identify the happens-before edge that orders them.** If you can name the edge (this lock, this atomic, this channel send), you are inside DRF-SC and safe. If you cannot, you have a data race — fix it before you optimize anything. The capstone, [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model), turns this into a model-selection flowchart; this post is the rule the flowchart is built on.

One contrast worth drawing, because the names collide. This post is about **memory consistency** — the order in which writes to memory by threads *on one machine, sharing one address space* become visible. There is a separate, larger world of **distributed consistency** — linearizability, sequential consistency, causal consistency, eventual consistency — about how replicas *across machines* agree on order, covered in [consistency models for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects). The vocabulary overlaps (sequential consistency means the analogous thing in both: a single order respecting each participant's program order) and the *intuition transfers* — a happens-before edge in shared memory is the same idea as a causal dependency in a distributed log — but the mechanisms are different. Shared-memory ordering is enforced by cache coherence and fences in hardware; distributed ordering is enforced by consensus and replication protocols over a network. Do not let the shared word make you think a CPU fence and a Raft log are the same tool; they solve the same shape of problem at wildly different scales.

## Key takeaways

- A **memory model** is the contract that says which values a read may return — the fence around the legal outcomes of a multithreaded program. You are always programming against one, whether you know it or not.
- **Sequential consistency** — one global order of all operations, each thread's program order preserved — is the intuitive ideal, but no mainstream CPU provides it for free because it would require a barrier on every access. Default hardware and compilers are fast-and-weak.
- **Happens-before** is the partial order you actually program against: program order within a thread, plus **synchronizes-with** edges across threads (unlock→lock, store-release→load-acquire, thread start/join, `volatile` write→read in Java), closed under transitivity.
- The **visibility rule**: if `A ⟶ B`, then A's writes are visible to B. To publish a value, you need a happens-before *path* from the write to the read — one synchronizes-with edge plus transitivity carries every earlier write across at once. The data need not be atomic; the publication must be.
- **DRF-SC** is the theorem that makes concurrency tractable: a data-race-free program behaves as if sequentially consistent. Put an edge between every pair of conflicting accesses and you get your simple interleaving model back.
- A **data race** is the model's red line — undefined behavior in C++, Go, and Rust; defined-but-weak in Java; a compile error in safe Rust. Avoiding races is the entire price of admission to the simple model.
- A **lock or channel manufactures the edges for you**, so most code never touches a memory ordering. Reach for raw atomics and hand-built happens-before only when building primitives or proving a deliberate weakening is safe — and measure before you assume the lock is your cost.
- The **platform changes the answer**: x86 TSO hides bugs that ARM and POWER expose. Always name the hardware when you report a memory-ordering measurement, and never trust an x86-only pass.

## Further reading

- Leslie Lamport, "How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs" (IEEE Transactions on Computers, 1979) — the original definition of sequential consistency.
- Sarita V. Adve and Mark D. Hill, "Weak Ordering — A New Definition" (ISCA 1990) — the data-race-free framework and the DRF-SC guarantee.
- Hans-J. Boehm and Sarita V. Adve, "Foundations of the C++ Concurrency Memory Model" (PLDI 2008) — the reference design every modern language model descends from. See also Boehm's "Threads Cannot Be Implemented As a Library" (PLDI 2005).
- Jeremy Manson, William Pugh, and Sarita V. Adve, "The Java Memory Model" (POPL 2005), and the JSR-133 specification and FAQ — how Java rebuilt its model around happens-before.
- Brian Goetz et al., *Java Concurrency in Practice* — chapter 16 on the memory model, the clearest practitioner treatment of `volatile`, publication, and safe construction.
- Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming* — linearizability, the formal cousin of consistency, and the theory of concurrent objects.
- Jeff Preshing's blog (preshing.com) — "Memory Ordering at Compile Time," "Acquire and Release Semantics," and the lock-free series; the best free, hands-on explanations of these ideas.
- Within this series: [why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [why your code doesn't run in order](/blog/software-development/concurrency/why-your-code-doesnt-run-in-order-compiler-and-cpu-reordering), [memory barriers, acquire, release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences), [atomics and memory orderings](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), and the capstone [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

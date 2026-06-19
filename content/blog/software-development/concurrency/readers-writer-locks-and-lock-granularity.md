---
title: "Readers-Writer Locks and Lock Granularity"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Squeeze more concurrency out of locking by letting readers run in parallel, splitting one hot lock into many, and knowing exactly when the extra bookkeeping pays for itself and when it quietly costs you."
tags:
  [
    "concurrency",
    "parallelism",
    "readers-writer-lock",
    "lock-striping",
    "granularity",
    "contention",
    "scalability",
    "synchronization",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/readers-writer-locks-and-lock-granularity-1.png"
---

Here is a service that looks correct and runs slow for a reason that is almost invisible in a code review. You have an in-memory configuration cache: a map from feature-flag name to value, loaded at startup and refreshed maybe once a minute when an operator pushes a change. Every request handler reads from it — checks three or four flags, looks up a routing rule, reads a rate limit. So this map is read tens of thousands of times per second and written a handful of times per hour. It is the most read-mostly object in the whole process. And because it is shared mutable state touched by every request thread, someone — correctly — wrapped it in a [mutex](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) to keep a refresh from tearing a read. The cache is thread-safe. It is also a bottleneck, and it gets worse the more cores you add.

The problem is that a plain mutex does not know the difference between a read and a write. It enforces *mutual* exclusion: one thread inside the critical section at a time, full stop. So when forty request threads all want to read three flags out of the cache, they line up single file and take turns, each one waiting for the lock the previous one is holding — even though none of them is changing anything, even though forty concurrent reads of an immutable snapshot could not possibly conflict with each other. The lock is serializing operations that did not need to be serialized. Add a 64th core and the line just gets longer; the cache reads do not get faster, because they were never running in parallel in the first place. You bought a 64-core machine and the config cache is running on one.

![a single mutex forcing three readers to take turns one at a time contrasted with a readers writer lock letting all three read in parallel](/imgs/blogs/readers-writer-locks-and-lock-granularity-1.png)

The figure above shows the shape of the fix and the shape of this entire post. On the left, the mutex: reader 1 holds the lock, readers 2 and 3 block and wait, and the reads happen in series. On the right, a **readers-writer lock** (RW lock, also written `RWLock`, `shared_mutex`, or `RWMutex`): all three readers hold the lock in *shared* mode simultaneously and run in parallel, because the lock has learned the one fact the mutex never knew — that readers do not conflict with other readers. Only a *writer* needs to be alone. That single distinction is the whole idea, and it is worth real money in a read-mostly system. But — and this is the part that separates the engineer who reaches for an RW lock by reflex from the one who reaches for it by measurement — the RW lock is not free, it is not always faster, and there are entirely ordinary workloads where it is *slower* than the dumb mutex it replaced. By the end of this post you will be able to look at a contended lock and decide, with a number to back it, whether the right move is an RW lock, or splitting one lock into many (**lock granularity** and **lock striping**), or threading a sequence of locks through a data structure (**hand-over-hand** locking), or — and this is a real and frequent answer — leaving the single boring mutex exactly where it is.

The throughline of this whole series is that concurrency is about naming what is shared and mutable, establishing a happens-before order over every access to it, and then buying that order with the *cheapest* mechanism that works. A mutex is one such mechanism; it is correct, and it is often the right call. This post is about the cases where it is correct but *too coarse* — where it serializes work that could safely overlap — and the family of techniques that recover that lost parallelism without giving up correctness. Every one of those techniques trades simplicity for throughput, and every one of them introduces a new way to be wrong. That trade is the real subject here.

## The readers-writer lock: shared read, exclusive write

Start with the precise contract, because everything else is a consequence of it. A readers-writer lock has two modes of acquisition instead of the mutex's one:

- **Shared mode** (the "read lock"): any number of threads may hold the lock in shared mode at the same time. While even one thread holds it shared, no thread may hold it exclusively.
- **Exclusive mode** (the "write lock"): at most one thread may hold the lock in exclusive mode, and while it does, no other thread may hold it in either mode.

That is the entire specification. Restate it as an invariant the way we stated mutual exclusion as $n(t) \le 1$ in the mutex post. Let $r(t)$ be the number of threads holding the lock in shared mode at time $t$ and $w(t)$ be the number holding it exclusively. The RW lock guarantees, for all $t$:

$$w(t) \le 1 \quad \text{and} \quad w(t) = 1 \implies r(t) = 0.$$

In words: never more than one writer, and a writer is never simultaneous with any reader. Readers are unbounded; the only constraint among them is that they must all stand aside for a writer. Notice what this buys: the writer still gets the full mutual-exclusion guarantee it had under a plain mutex — when it is in, it is alone, so it can mutate the structure freely without anyone observing a half-updated state. The readers get something new: they get to run *concurrently with each other*. A plain mutex is the special case of an RW lock where you simply never use shared mode — every acquisition is exclusive — which is why "use a mutex" is always a safe, if sometimes slow, answer.

The reason this is *safe* rests on a fact about data races that the [race-condition post](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) established: a data race requires two accesses to the same location, on different threads, where *at least one is a write*, with no happens-before ordering between them. Two reads do not race. Reading a value cannot corrupt it, cannot tear it, cannot lose an update, because reading does not change anything. So letting a hundred threads read the same immutable snapshot in parallel is not just a performance trick — it is *correct by construction*, as long as no writer is mutating underneath them. The RW lock's job is precisely to enforce that last clause: keep writers out while readers are in, and keep readers out while a writer is in. It is the mutual-exclusion guarantee, refined to apply only between operations that actually conflict.

### What the lock has to track, and why that costs something

A plain mutex can be a single machine word: locked or unlocked, flipped with one atomic [compare-and-swap](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks). An RW lock cannot be that simple, and the reason it is more expensive is worth understanding precisely, because that cost is exactly what makes the RW lock lose on the wrong workload.

To honor the contract, the lock must maintain at minimum a **reader count** and a **writer flag** (or, more commonly, pack both into a single integer word). When a reader acquires, it must atomically increment the reader count — but only if no writer holds or, depending on policy, is waiting. When a reader releases, it must atomically decrement the count, and *the reader that drops the count to zero* must wake any waiting writer. When a writer acquires, it must wait until the reader count is zero and no other writer holds, then set the writer flag. None of these operations is a single unconditional atomic; each is a *conditional* atomic, typically a compare-and-swap loop or a guarded update behind an internal lock plus a condition variable. The reader fast path — "increment the count, return" — is two or three atomic operations where the mutex needed one, and on a contended cache line every one of those atomics is a coherence transaction that ping-pongs the line between cores.

#### Worked example: the cost of the reader count cache line

Picture the reader-count word living in one cache line. Thread on core 0 wants to acquire shared: it must read the current count, check no writer is set, and compare-and-swap the count up by one. Thread on core 1 wants the same thing at the same instant. Both cores need the line in a modified/exclusive coherence state to do the CAS, so the line bounces: core 0 grabs it, increments, core 1's CAS fails because the value changed, core 1 re-reads (pulling the line over), retries, succeeds. Every reader acquisition is a write to a *shared* word, so even though the readers are logically only reading the *data*, they are all *writing* the lock's bookkeeping. On a 32-core box hammering one RW lock, that single counter word can become the hottest cache line in the machine — and now your "parallel" readers are serialized again, not on the data, but on the lock's own counter. This is the central irony of the RW lock and we will measure it: the bookkeeping that lets readers overlap can itself become the thing that stops them from overlapping. Modern RW locks fight this with per-core or per-thread reader slots (so readers touch *different* cache lines), but that makes the writer's job — which must now scan or drain all those slots — more expensive. There is no free lunch; you are moving the cost around.

The matrix below states the admission rules the lock enforces — exactly the conditional logic the reader count and writer flag drive. Read each row as "the lock is currently in this state," and each cell as "what a newly arriving caller of this type is allowed to do."

![a matrix of readers writer lock states showing whether a new reader or a new writer can enter when the lock is idle has active readers or has an active writer](/imgs/blogs/readers-writer-locks-and-lock-granularity-2.png)

The interesting cell is the middle row, right column: when N readers are active, a new *writer* must wait for the reader count to drain to zero. That single cell is the source of the nastiest behavior in RW locks, and it deserves its own section.

## Reader and writer starvation, and fairness policies

The contract — $w \le 1$, and a writer excludes all readers — says *nothing about who goes first or how long anyone waits*. It is a safety property, not a liveness or fairness property. And exactly as with the plain mutex, that silence is where the bugs live. The signature failure mode of a naive RW lock is **writer starvation**, and it is not a rare edge case; it is the *default* behavior of the most obvious implementation.

Here is how it happens. Suppose the lock uses the simplest possible policy: a new reader may enter as long as no writer currently *holds* the lock. This is called **reader preference** (or read-preferring). It maximizes read throughput, which sounds great. Now run it on our config cache: reads arrive continuously, tens of thousands per second, and they overlap — at any instant the reader count is some positive number. A writer (the once-a-minute config refresh) shows up and asks for exclusive access. It must wait for the reader count to hit zero. But under reader preference, *new readers keep being admitted* while the writer waits, because no writer is *holding* (the writer is only waiting). So the reader count never reaches zero. The writer waits. And waits. The config refresh that was supposed to take a microsecond hangs indefinitely behind an unbroken stream of readers, and your "once a minute" update silently stops happening. Nobody sees an error. The cache just goes stale forever.

![a timeline showing a stream of overlapping readers keeping the reader count above zero so a waiting writer is starved until a writer preference fix parks new readers and lets the count drain](/imgs/blogs/readers-writer-locks-and-lock-granularity-3.png)

The timeline traces it: reader A enters (count 1), the writer arrives and waits because the count is positive, reader B enters before A leaves (count 2), and the overlap never breaks — the writer is starved. The fix, shown in the back half of the timeline, is a different fairness policy.

There are three standard policies, and choosing among them is a real design decision with real consequences:

| Policy | New reader admitted while a writer waits? | Optimizes for | Failure mode |
| --- | --- | --- | --- |
| **Reader preference** | Yes, always | Max read throughput | Writers starve under continuous reads |
| **Writer preference** | No — new readers park if any writer is waiting | Writer freshness, bounded writer latency | Readers can stall in write-heavy bursts |
| **Fair (FIFO-ish)** | Depends on arrival order | Bounded waiting for both | Slightly lower peak throughput; more bookkeeping |

**Writer preference** breaks the starvation cycle by changing one rule: once a writer is *waiting* (not just holding), new readers are no longer admitted — they park behind the writer. The readers already inside finish and drop the count; because no new readers join, the count reaches zero; the writer gets in, does its update, and releases; then the parked readers proceed. The writer's wait is now bounded by the duration of the in-flight reads, not by the infinite future stream of them. The cost is symmetric and you should see it coming: in a write-heavy burst, a steady stream of writers can now starve *readers*, because every arriving writer keeps parking the readers. You traded one starvation for the possibility of the other, and which one you can tolerate depends entirely on your workload. For a read-mostly config cache, writer preference is almost always right: writes are rare, so readers rarely park, and when they do it is for the microsecond a refresh takes.

**Fair** policies enforce a rough first-come-first-served order, often by queueing waiters: a writer that arrives is placed in line, and readers that arrive *after* it wait behind it, but readers that arrived *before* it are let through. This bounds the waiting time for everyone at the cost of a little throughput (you cannot greedily batch all available readers) and more internal bookkeeping (you need an actual queue, not just two counters).

### What the real APIs give you, and the trap of not knowing

This is where "language-agnostic concept, idiomatic implementation" matters most, because the defaults differ and the defaults bite.

- **Go**'s `sync.RWMutex` is documented to give writer preference in spirit: "if a goroutine holds a RWMutex for reading and another goroutine might call Lock, no goroutine should expect to be able to acquire a read lock until the initial read lock is released" — a blocked writer prevents new readers from acquiring, precisely to avoid writer starvation. Good default for most servers.
- **Java**'s `ReentrantReadWriteLock` lets you *choose* at construction: `new ReentrantReadWriteLock(true)` is the fair variant; the default `new ReentrantReadWriteLock()` is non-fair (higher throughput, weaker ordering guarantees, and it can starve). It also supports lock *downgrading* (hold write, acquire read, release write) but **not** upgrading (acquiring write while holding read deadlocks — two readers each trying to upgrade wait for each other forever).
- **C++**'s `std::shared_mutex` (since C++17) deliberately leaves the policy *unspecified* — the standard does not promise reader or writer preference, so it varies by implementation (libstdc++ and libc++ differ, and have changed across versions). If you depend on a specific fairness behavior in portable C++, you are depending on something the standard does not give you.
- **Rust**'s `std::sync::RwLock` also makes no priority guarantee in the standard library and explicitly documents that it "may" be either preference depending on the OS; the `parking_lot` crate's `RwLock` is a popular alternative with documented, tunable behavior and a smaller footprint.

The lesson is blunt: **an RW lock without a known fairness policy is a starvation bug waiting for a workload to trigger it.** Before you ship one, know — from the docs, not from a guess — what happens to a waiting writer under continuous reads, and what happens to waiting readers under continuous writes. If the docs say "unspecified," treat that as "will starve someone, on some platform, eventually."

## Building an RW lock from a mutex and a condition variable

The fairness discussion stays abstract until you see the mechanism, so let us build a writer-preferring RW lock out of primitives you already understand — a plain mutex and a [condition variable](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections). This is not how a production library does it (they pack the state into a single atomic word and use futexes to avoid the inner mutex), but it makes every rule in the contract visible, and it shows exactly where the bookkeeping cost lives. The state is three integers, all protected by one inner mutex: the count of active readers, a flag for whether a writer is active, and a count of *waiting* writers — that last one is the entire difference between writer preference and writer starvation.

```cpp
#include <mutex>
#include <condition_variable>

// A writer-preferring RW lock built from a mutex + condition variable.
// State: how many readers are inside, whether a writer is inside, how many
// writers are waiting. The waiting-writer count is what blocks new readers.
class RWLock {
    std::mutex mu_;
    std::condition_variable cv_;
    int readers_ = 0;          // readers currently inside (shared mode)
    bool writer_ = false;      // is a writer currently inside (exclusive)
    int waitingWriters_ = 0;   // writers parked, waiting to get in

public:
    void lockShared() {                 // acquire read (shared) mode
        std::unique_lock<std::mutex> lk(mu_);
        // Writer preference: a reader waits if a writer is in OR waiting.
        cv_.wait(lk, [&] { return !writer_ && waitingWriters_ == 0; });
        ++readers_;
    }
    void unlockShared() {
        std::unique_lock<std::mutex> lk(mu_);
        if (--readers_ == 0) cv_.notify_all();   // last reader wakes a writer
    }
    void lockExclusive() {              // acquire write (exclusive) mode
        std::unique_lock<std::mutex> lk(mu_);
        ++waitingWriters_;              // announce intent BEFORE waiting
        cv_.wait(lk, [&] { return !writer_ && readers_ == 0; });
        --waitingWriters_;
        writer_ = true;
    }
    void unlockExclusive() {
        std::unique_lock<std::mutex> lk(mu_);
        writer_ = false;
        cv_.notify_all();               // wake parked readers and writers
    }
};
```

Walk the writer-preference rule through it. A reader, in `lockShared`, waits not just while a writer holds the lock but while any writer is *waiting* (`waitingWriters_ == 0` is part of the predicate). So the moment a writer increments `waitingWriters_` in `lockExclusive`, every subsequently arriving reader parks — that is precisely the "new readers no longer admitted" rule that breaks the starvation cycle from the timeline. The writer announces its intent *before* it waits (the `++waitingWriters_` comes before `cv_.wait`), which is the subtle, load-bearing ordering: if it waited first and announced after, a flood of readers could squeak in during the gap. The condition variable's predicate-in-a-loop form (`cv_.wait(lk, pred)`) handles spurious wakeups correctly — the thread re-checks the predicate every time it wakes and goes back to sleep if it is not satisfied, which is the one non-negotiable rule of condition-variable use.

Now you can *see* the cost the earlier section asserted. Every single read acquisition takes the inner mutex `mu_`, checks a predicate, increments a counter, and releases — that is the bookkeeping $b$, and it is a *write* to shared state (`mu_` and `readers_`) on the read path. A plain mutex's read path was one CAS; this is a lock, a branch, an increment, and an unlock. Multiply that by tens of thousands of reads per second across many cores all hammering the same `mu_` and `readers_`, and the inner mutex becomes the very serialization point the RW lock was supposed to eliminate. Production locks fight this by replacing the inner mutex with a single atomic word and lock-free CAS on the fast path, and by sharding the reader count across cache lines — but the fundamental shape remains: **the RW lock turns every read into a write of the lock's bookkeeping, and that write is shared.** That sentence is the whole reason the RW lock can lose. The Go and Java standard-library locks are far more clever than this teaching version, but they are managing the same tension.

#### Worked example: tracing one writer past three readers

Concretely, run the writer-preferring lock through one cycle. Three readers (R1, R2, R3) are inside: `readers_ = 3`, `writer_ = false`, `waitingWriters_ = 0`. A writer W calls `lockExclusive`: it takes `mu_`, sets `waitingWriters_ = 1`, evaluates the predicate `!writer_ && readers_ == 0` — false, because `readers_ == 3` — and waits, releasing `mu_`. Now reader R4 arrives and calls `lockShared`: it takes `mu_`, evaluates `!writer_ && waitingWriters_ == 0` — false, because `waitingWriters_ == 1` — so R4 *parks too*, behind the writer. This is writer preference doing its job: R4 did not jump the queue. R1, R2, R3 finish and call `unlockShared`; on the third, `--readers_` hits 0 and it calls `notify_all`. W wakes, re-checks its predicate — now `readers_ == 0` and `!writer_` — succeeds, sets `writer_ = true`, and proceeds exclusively. When W finishes, `unlockExclusive` clears `writer_` and notifies; R4 wakes, sees `!writer_ && waitingWriters_ == 0`, and finally enters. The writer's wait was bounded by the three in-flight reads — *not* by the infinite stream of future readers — which is exactly the starvation fix made mechanical.

## When an RW lock beats a mutex — and the surprising cases where it loses

Now the question that actually decides whether you should use one. The RW lock's whole value proposition is *reader parallelism*: it lets N readers run at once instead of one at a time. So the obvious model is "if I have lots of readers, use an RW lock." That model is wrong often enough to be dangerous. Let me give you the real one, derived rather than asserted.

Model a critical section that takes time $C$ to execute. Suppose a fraction $f$ of operations are reads and $1-f$ are writes, and operations arrive at total rate $\lambda$. Under a **plain mutex**, every operation — read or write — serializes, so the lock is busy a fraction $\lambda C$ of the time and you saturate (the lock becomes the bottleneck) when $\lambda C \to 1$. Reads and writes are indistinguishable to the mutex; throughput is capped at $1/C$ operations per second regardless of how many cores you have, the moment the lock is the bottleneck.

Under an **RW lock**, the reads can overlap. In the ideal case — perfect reader preference, no contention on the lock's own counter, infinite cores — the reads cost effectively nothing in serialization (they all run at once), and only the writes serialize. The writes occupy the lock exclusively for a fraction $(1-f)\lambda C$ of the time, plus each write must wait for in-flight readers to drain. So the *serialization ceiling* drops from $\lambda C$ to roughly $(1-f)\lambda C$. If $f = 0.99$ (99% reads), the RW lock removes 99% of the serialization — that is the win, and on a read-heavy long-section workload it is enormous.

But that derivation hid three costs that are exactly the cases where the RW lock loses:

1. **The bookkeeping overhead per acquisition.** Call it $b$ — the extra atomics for the reader count versus the mutex's single CAS. Every read now costs $C + b$ instead of $C$ under the mutex. If $C$ is *tiny* — a single map lookup, a few nanoseconds — then $b$ is a large *fraction* of $C$, and the RW lock's reads are meaningfully slower than the mutex's reads even when they run in parallel. The parallelism win is real but it is paid for in per-operation tax, and for short sections the tax can exceed the win.
2. **Contention on the reader-count cache line.** That $b$ is not constant — under heavy read concurrency it *grows*, because the reader count word is a shared cache line that every reader writes, and as we saw it can ping-pong across cores until the readers are serialized on the counter. So the "infinite cores, reads cost nothing" assumption fails in practice exactly when you most wanted it to hold.
3. **Writes still fully exclude.** If $f$ is small — a write-heavy workload — then $(1-f)\lambda C \approx \lambda C$ and the RW lock's serialization ceiling is barely better than the mutex's, while you are *still* paying the bookkeeping overhead $b$ on every operation. The RW lock has all the cost and almost none of the benefit. It loses outright.

Put those together and the rule sharpens to something you can actually apply:

![a matrix comparing an RW lock against a plain mutex across read heavy long sections read heavy short sections write heavy and balanced workloads](/imgs/blogs/readers-writer-locks-and-lock-granularity-4.png)

The matrix is the decision in one glance. An RW lock is a clear win in exactly one quadrant — **read-heavy workloads with long critical sections** (the config cache that does real parsing on read; an in-memory index that does a multi-key range scan under the lock; a routing table walked deeply per request). For **read-heavy but very short** sections, the bookkeeping often eats the gain and a plain mutex (or, better, a lock-free read path or an immutable snapshot you swap atomically) wins. For **write-heavy** or **balanced** workloads, the RW lock loses to the mutex because writes serialize either way and you are paying extra for nothing.

#### Worked example: the section that is too short to benefit

Concretely: a thread-safe counter map where a "read" is `map.get(key)` — a single hash lookup, call it 30 nanoseconds. The RW lock's reader bookkeeping (two atomics on a contended counter, say) adds maybe 20-40 nanoseconds under load. So your 30 ns read becomes a 50-70 ns read, *and* under high reader concurrency the counter cache line contention can push that higher still. Meanwhile the mutex version's read was 30 ns of work plus one CAS, roughly 40-50 ns, and under contention the readers serialize — but each read is so short that the queue drains fast. The crossover is real and not obvious: for sections this short, the mutex is often *faster* in wall-clock throughput than the RW lock, because the RW lock's per-operation overhead and counter contention outweigh the benefit of overlapping 30-nanosecond reads. The RW lock pays off when the thing you are overlapping is *long* enough that overlapping it actually saves meaningful time. Overlapping nanoseconds is not worth the bookkeeping; overlapping microseconds or more is. **Measure the section length before you reach for the RW lock.**

## Lock granularity: one coarse lock versus many fine locks

The RW lock attacks contention along one axis — read versus write. **Granularity** attacks it along a different and often more powerful axis: *which data does the lock cover?* This is, in my experience, where the bigger wins usually live, and it applies whether your locks are plain mutexes or RW locks.

A **coarse-grained** lock protects a large amount of data — in the limit, one lock for the entire data structure (or, the nightmare case, one "global lock" for the whole program — the original Linux "Big Kernel Lock," CPython's [GIL](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs), the giant `synchronized` block someone wrapped around a whole service). It is simple: one lock, impossible to forget, no ordering questions, no deadlock between locks because there is only one. And it is a scalability disaster, because *every* operation that touches *any* part of the protected data contends on that one lock. Two threads updating completely unrelated entries — different keys, different rows, different accounts — still serialize, because they share the lock even though they do not share the data.

A **fine-grained** lock protects a small amount of data — in the limit, one lock per item (per map entry, per list node, per row). Now two threads touching different items do not contend at all; they grab different locks and run fully in parallel. The contention drops in proportion to how finely you split, and the scalability improves correspondingly. But you have bought that with three new costs, and they are the costs that define the rest of this post:

1. **Memory and overhead.** A lock is not free to store. One lock per million map entries is a million lock objects — that can dwarf the data itself, and every operation now does lock acquisition bookkeeping per item.
2. **Complexity.** The code is harder to get right. Which lock protects which datum? What if an operation must touch *two* items — does it take two locks? In what order? Operations that span multiple fine-grained locks are where correctness gets genuinely hard.
3. **Deadlock risk.** The instant an operation must hold more than one lock at a time, you can deadlock. Thread 1 holds lock A and wants B; thread 2 holds B and wants A; both wait forever. With a single coarse lock this is *impossible* (there is only one lock to hold). Fine-grained locking *introduces* the possibility of deadlock that coarse locking did not have. We will come back to this hard.

So granularity is a dial, not a binary. All the way coarse: simple, deadlock-free, doesn't scale. All the way fine: scales, complex, deadlock-prone, and memory-hungry. The engineering question is *where on the dial to sit*, and the beautiful answer — the one that gives most of the scaling benefit with a fraction of the cost — is usually not at either extreme. It is **striping**.

It helps to make the contention concrete with a quick model. Suppose $T$ threads each repeatedly perform an operation that holds a lock for time $C$, and the operations are spread uniformly over $L$ independent items. Under one coarse lock ($L$ items, 1 lock), every operation contends with every other, so the effective serialization is total: throughput is capped at $1/C$ no matter how many threads or items you have. Under fully fine locking ($L$ items, $L$ locks), two operations collide only when they hit the *same* item; with $L$ much larger than $T$ that is rare, so throughput scales nearly linearly with threads until you saturate the cores — at the cost of $L$ lock objects and the deadlock and complexity baggage. The key realization is that you almost never need $L$ locks to get most of that benefit: with $N$ locks where $N$ is a small multiple of the core count, collisions are already rare, and you have paid for $N$ locks instead of $L$. That gap between "$N$ locks gets you 95% of the scaling" and "$L$ locks gets you 100% at huge cost" is exactly the space striping lives in, and it is why striping is the technique you reach for far more often than true per-item locking.

## Lock striping: a striped hash map

Lock striping is the pragmatic middle of the granularity dial, and it is one of the most useful concurrency patterns there is. The idea is dead simple once you see it: instead of one lock for the whole structure (too coarse) or one lock per element (too fine, too many locks), use a *fixed, modest number* of locks — call it $N$ **stripes** — and deterministically map each element to one stripe by hashing. Element $e$ is protected by stripe `hash(e) mod N`. Two operations on elements that happen to land on different stripes proceed fully in parallel; two that land on the same stripe contend, exactly as they would under a single lock. With $N$ stripes and uniformly hashed keys, the expected contention per lock drops to about $1/N$ of the single-lock case.

![a single coarse lock funneling every operation through one contention point contrasted with sixteen stripe locks spreading the same operations so each lock sees about a sixteenth of the traffic](/imgs/blogs/readers-writer-locks-and-lock-granularity-5.png)

The before-after captures the whole point: on the left, one coarse lock, every operation funneling through a single contended point, throughput flat no matter how many cores you add. On the right, sixteen stripes, operations hashed across them, each lock seeing roughly a sixteenth of the traffic, throughput that actually scales with cores until you saturate the stripes. Sixteen is a typical choice; the original `java.util.concurrent.ConcurrentHashMap` (Java 5-7) used exactly 16 segments by default, each an independently locked sub-map, and that one design choice is most of why it scaled where the old fully-synchronized `Hashtable` (a single coarse lock) did not.

The grid below shows the mapping concretely for a smaller map: eight buckets, four stripes, each stripe owning two buckets and its own lock.

![a grid showing eight hash buckets mapped onto four stripe locks where each stripe owns two buckets so writes under different stripes run in parallel](/imgs/blogs/readers-writer-locks-and-lock-granularity-6.png)

A write to bucket 0 takes stripe 0's lock; a write to bucket 1 takes stripe 1's lock; those two writes run in parallel because they contend on nothing. A write to bucket 0 and a write to bucket 4 *both* take stripe 0's lock and serialize — that is the residual contention you accept in exchange for using a bounded number of locks. The stripe count $N$ is the dial: bigger $N$ means less collision and more parallelism, but more locks to store and (if you ever need to lock multiple stripes) more ordering complexity.

### Building it: the striped map in two languages

Let me show the bug first — the coarse version that works but does not scale — and then the striped fix. Here is the coarse, single-lock concurrent map in Go. It is correct. It is also the bottleneck.

```go
// Coarse-grained: ONE mutex guards the whole map. Correct but does not scale —
// every Get and Put on any key contends on the single lock m.mu.
type CoarseMap struct {
	mu sync.RWMutex
	m  map[string]int
}

func (c *CoarseMap) Get(k string) (int, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.m[k]
	return v, ok
}

func (c *CoarseMap) Put(k string, v int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[k] = v
}
```

Now the striped fix. We carve the key space into `N` independent shards, each with its own lock and its own sub-map. A key is routed to a shard by hashing. Two puts on keys that hash to different shards never touch the same lock.

```go
// Fine-grained via striping: N independent shards, each with its own lock and
// sub-map. Puts/Gets on keys in different shards run fully in parallel.
type StripedMap struct {
	shards []shard
	n      uint32
}

type shard struct {
	mu sync.RWMutex
	m  map[string]int
}

func NewStripedMap(n int) *StripedMap {
	s := &StripedMap{shards: make([]shard, n), n: uint32(n)}
	for i := range s.shards {
		s.shards[i].m = make(map[string]int)
	}
	return s
}

// fnv-1a hash, then mod the shard count to pick the stripe for this key.
func (s *StripedMap) shardFor(k string) *shard {
	h := uint32(2166136261)
	for i := 0; i < len(k); i++ {
		h ^= uint32(k[i])
		h *= 16777619
	}
	return &s.shards[h%s.n]
}

func (s *StripedMap) Get(k string) (int, bool) {
	sh := s.shardFor(k)
	sh.mu.RLock()
	defer sh.mu.RUnlock()
	v, ok := sh.m[k]
	return v, ok
}

func (s *StripedMap) Put(k string, v int) {
	sh := s.shardFor(k)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	sh.m[k] = v
}
```

Notice that each shard combines *both* techniques in this post: it is one stripe of a fine-grained split (granularity), and each stripe uses an RW lock (`sync.RWMutex`) so that reads within a shard still overlap. That composition — striping for cross-key parallelism, RW locks for within-stripe read parallelism — is exactly the design of a high-throughput concurrent map.

The Java idiom is the same shape but you mostly do not write it yourself, because the standard library already did: `ConcurrentHashMap` *is* a striped map. In Java 5-7 it was literally an array of `Segment` objects, each a `ReentrantLock`-guarded sub-table; in Java 8+ the design changed to per-bucket (per-node) locking with CAS for the common case, which is striping taken to its fine limit plus lock-free reads. If you want to *see* the explicit striping idiom, Guava's `Striped<Lock>` exposes it directly:

```java
import com.google.common.util.concurrent.Striped;
import java.util.concurrent.locks.Lock;

// A pool of, say, 16 locks. Striped.lock(key) deterministically maps a key to
// one of the 16 — same key always maps to the same lock, so concurrent ops on
// the SAME key serialize, but ops on different keys usually hit different locks.
Striped<Lock> stripes = Striped.lock(16);

void update(String key, int value) {
    Lock lock = stripes.get(key);   // hash(key) -> one of 16 locks
    lock.lock();
    try {
        backingMap.put(key, value); // backingMap is a plain HashMap per region
    } finally {
        lock.unlock();
    }
}
```

The same pattern in C++ with `std::shared_mutex` per stripe makes the read-parallelism explicit:

```cpp
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <string>

// Striped map: N stripes, each a shared_mutex + sub-map. shared_lock for reads
// (many concurrent), unique_lock for writes (exclusive within the stripe only).
class StripedMap {
    struct Stripe {
        mutable std::shared_mutex mu;
        std::unordered_map<std::string, int> m;
    };
    std::vector<Stripe> stripes_;
    std::hash<std::string> hasher_;

    Stripe& stripeFor(const std::string& k) {
        return stripes_[hasher_(k) % stripes_.size()];
    }

public:
    explicit StripedMap(std::size_t n) : stripes_(n) {}

    bool get(const std::string& k, int& out) {
        Stripe& s = stripeFor(k);
        std::shared_lock<std::shared_mutex> lock(s.mu);  // shared: readers overlap
        auto it = s.m.find(k);
        if (it == s.m.end()) return false;
        out = it->second;
        return true;
    }

    void put(const std::string& k, int v) {
        Stripe& s = stripeFor(k);
        std::unique_lock<std::shared_mutex> lock(s.mu);  // exclusive within stripe
        s.m[k] = v;
    }
};
```

(One detail that bites people: `std::vector<Stripe>` of a struct containing a non-movable `std::shared_mutex` must be sized in the constructor as above — you cannot `push_back` stripes after the fact, because the mutex is neither copyable nor movable. Size it once and leave it.)

A Python note, since it is the clearest *cautionary* illustration: you *can* write a striped dict in Python, but the [GIL](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) means CPU-bound threads do not run Python bytecode in parallel anyway, so striping buys you far less than it does in Go, Rust, Java, or C++. For the Python concurrency story — when threads help (I/O-bound) and when they cannot (CPU-bound under the GIL), and what `free-threaded` builds change — the [python-performance series owns that ground](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits); I will not re-derive it here. The pattern is universal; the *payoff* depends on whether your runtime gives you real parallelism.

### How many stripes? The sizing question

The stripe count $N$ is a genuine tuning parameter and the math is approachable. With $T$ threads each repeatedly hitting a uniformly random key, the chance that two given threads collide on the same stripe in a given instant is about $1/N$. The expected number of threads piled on the *busiest* stripe (a balls-into-bins problem) grows slowly once $N \gtrsim T$, so a good rule of thumb is **$N$ at least equal to the number of cores, often $2\times$ to $4\times$**, rounded to a power of two so the modulo becomes a cheap mask. Past that, more stripes mostly waste memory: you cannot have more genuine parallelism than you have cores, so once each core can usually find an uncontended stripe, adding stripes only adds lock objects. Sixteen is a fine default for up to ~16 cores; 64 or 128 for a big many-core box. And measure — a skewed key distribution (one hot key getting 90% of traffic) defeats striping entirely, because that one key always lands on the same stripe and that stripe becomes the bottleneck no matter how many other stripes sit idle.

#### Worked example: sizing stripes for a 32-core server

You are running a concurrent counter map on a 32-core box, peak load roughly 32 busy threads, keys reasonably uniform. Start at $N = 32$: the chance two threads collide on a stripe at a given instant is about $1/32 \approx 3\%$, so the vast majority of operations find an uncontended lock — already a 32-fold contention reduction over the single coarse lock. Bump to $N = 64$ (the $2\times$ rule): collision probability halves to ~1.5%, and the busiest stripe almost never has more than two threads. Bump to $N = 256$ and the collision rate keeps dropping, but you now hold 256 lock objects and 256 sub-maps, and you cannot run more than 32 operations truly in parallel anyway — so the extra 192 locks buy almost nothing but memory. The honest sweet spot here is 64: most of the contention relief, modest memory, and a power of two so `hash & 63` replaces a division. Then *check your key distribution* — if profiling shows one tenant's key takes 40% of traffic, no stripe count saves you, because that key pins one stripe; the fix is a different decomposition (e.g., shard that hot key's value into per-thread sub-counters and sum on read), not more stripes.

One more wrinkle the standard libraries handle and a hand-rolled striped map usually does not: **resizing**. A growing hash map eventually needs to rehash into more buckets. With a single coarse lock that is one stop-the-world resize under the lock. With striping it is genuinely hard — you must either resize one stripe's sub-table at a time (so different stripes can be different sizes mid-resize) or coordinate a global resize that briefly locks *all* stripes in a fixed order (and now you are holding many locks at once, which reintroduces deadlock risk if anything else ever locks multiple stripes). This is a big part of why you should reach for `ConcurrentHashMap` or a library striped structure rather than rolling your own past a toy: the resize-under-striping problem is subtle and they have solved it. A fixed-size striped map (no resize) is easy and safe to hand-roll; a growable one is not.

## Hand-over-hand locking for a linked structure

Striping works because a hash map's elements are *independent* — any key can map to any stripe and operations on different keys do not interact. But many structures are *not* independent: a linked list, a tree, a skip list — their nodes are connected, and an operation traverses a *path* through them. You cannot stripe a linked list by hashing, because to get to node 5 you must walk through nodes 1 through 4. For these, the fine-grained technique is **hand-over-hand locking**, also called **lock coupling**.

The idea is a moving window of locks that walks the structure. To traverse safely with fine-grained locks, you never let go of where you are until you have a grip on where you are going next. Concretely, to move from node A to node B: you are holding A's lock; you acquire B's lock; *then* you release A's lock. At every instant you hold at most two adjacent locks, and there is never a gap — you are always holding the lock on the node you currently occupy, so no other thread can splice a node out from under you or change the link you are about to follow.

![a timeline of hand over hand traversal acquiring the next node lock before releasing the current one so at most two adjacent locks are held and no gap ever opens](/imgs/blogs/readers-writer-locks-and-lock-granularity-7.png)

The timeline is the algorithm: lock A, lock B (now holding A and B), unlock A (now holding B), lock C (now holding B and C), unlock B (now holding C). The two-lock window slides forward. The crucial invariant is the overlap — you grab the *next* lock *before* releasing the *current* one. If you released A before grabbing B, there would be an instant where you hold nothing, and in that instant another thread could delete B, or insert between A and B, and your traversal would step into freed or rearranged memory. The overlap is what makes it correct.

Here is a hand-over-hand sorted-list insert in Java, fine-grained with a per-node lock:

```java
// Each node has its own lock. Insert walks the list holding at most two adjacent
// locks at a time (pred and curr), never letting go of pred until it holds curr.
class Node {
    int key;
    Node next;
    final ReentrantLock lock = new ReentrantLock();
    Node(int key) { this.key = key; }
}

class HandOverHandList {
    private final Node head = new Node(Integer.MIN_VALUE); // sentinel

    void insert(int key) {
        Node pred = head;
        pred.lock.lock();                 // grab the head first
        try {
            Node curr = pred.next;
            if (curr != null) curr.lock.lock();
            try {
                // Slide the two-lock window forward until we find the slot.
                while (curr != null && curr.key < key) {
                    pred.lock.unlock();   // release pred AFTER we hold curr
                    pred = curr;          // window moves forward by one node
                    curr = curr.next;
                    if (curr != null) curr.lock.lock(); // grab the new curr
                }
                Node node = new Node(key); // splice in between pred and curr
                node.next = curr;
                pred.next = node;
            } finally {
                if (curr != null) curr.lock.unlock();
            }
        } finally {
            pred.lock.unlock();
        }
    }
}
```

Trace the locking discipline: we always hold `pred`'s lock; before advancing we acquire `curr`'s lock; only *then*, inside the loop, do we release the *old* `pred` after `pred` has been reassigned to the locked `curr`. At no point is the path we are walking unlocked. The same algorithm in C++ with `std::mutex` per node and `std::unique_lock` for scoped release looks structurally identical; the *idiom* differs (RAII lock guards vs. try/finally) but the *invariant* — acquire-next-before-release-current — is the universal part.

#### Worked example: why the overlap is not optional

Suppose two threads, T1 inserting key 5 and T2 deleting key 4, both walking a list `head -> 3 -> 4 -> 6`. Without the overlap (release current before acquiring next), here is a losing interleaving. T1 is positioned with `pred = node3`, about to move to `node4`; it releases node3's lock, then *pauses* (the scheduler preempts it) before locking node4. In that gap, T2 acquires node3 and node4, unlinks node4 (`node3.next = node6`), and frees it. T1 resumes, locks `node4` — which is now freed memory — reads `node4.next`, and follows a dangling pointer. Corruption, or a crash, or worse, a silent wrong answer. With hand-over-hand's overlap, T1 *never released node3* until it held node4, so T2 could not have unlinked node4 (it could not acquire node4's lock while T1 held it, and it could not unlink node4 without holding node3, which T1 held). The overlap closes the exact window the bug needs. This is the same lesson as the whole series in miniature: establish an unbroken happens-before chain over every access to shared mutable state, and the race has nowhere to live.

Hand-over-hand is powerful but it is also a vivid illustration of the granularity cost: it is *far* more complex than a single lock around the whole list, and it is correct only if every operation that walks the list uses the *same* discipline. One method that traverses the list without lock coupling — or that takes the locks in a different order — and you have a deadlock or a corruption. The fine-grained payoff (two operations on distant parts of the list run concurrently) is real, but the code that buys it is unforgiving. In practice, for most lists, the honest answer is "use a coarse lock unless you have measured that the list traversal is your bottleneck," and for the cases where it *is* the bottleneck, a lock-free list or a concurrent skip list (which the standard libraries provide — `ConcurrentSkipListMap` in Java) is usually a better destination than hand-rolled hand-over-hand.

## The trade: contention versus complexity versus deadlock risk

Step back and look at what every technique in this post is actually trading. They all buy the same thing — less contention, more parallelism — and they all pay in the same two currencies: **code complexity** and **deadlock risk**. The whole art is reading that trade correctly.

![a matrix comparing a coarse single lock fine per item locks and striped locking across contention code complexity and deadlock risk](/imgs/blogs/readers-writer-locks-and-lock-granularity-8.png)

The matrix lays it out. A **coarse single lock** has high contention but trivial complexity and zero deadlock risk (one lock cannot deadlock against itself, ignoring re-entrancy). **Fine per-item locks** have low contention but high complexity and high deadlock risk. **Striping** sits in the middle of all three — low contention (about $1/N$), medium complexity (a fixed lock pool and a hash function), medium deadlock risk (only if an operation must lock multiple stripes, and then you must order them). Striping is so popular precisely because it occupies that sweet spot: most of the contention reduction of fine-grained locking, with a fraction of the complexity and deadlock exposure.

The deadlock risk is the one that turns a performance optimization into a 3 AM page, so it deserves a hard look. **The moment any operation must hold two or more locks at once, deadlock becomes possible.** The classic cycle: a `transfer(A, B)` that locks account A then account B, run concurrently with `transfer(B, A)` that locks B then A — thread 1 holds A waiting for B, thread 2 holds B waiting for A, neither ever proceeds. Coarse locking is immune (one lock); fine locking is exposed the instant operations span items; striping is exposed only if an operation needs multiple stripes (e.g., a `move(key1, key2)` that touches two shards). The universal defense is **lock ordering**: define a total order over locks (by stripe index, by memory address, by account id) and *always acquire multiple locks in that order*. If both `transfer(A,B)` and `transfer(B,A)` lock the lower-id account first, the cycle cannot form. This is the single most important rule when you go fine-grained, and it has a whole post to itself — the [deadlock post](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them) covers the four Coffman conditions and how lock ordering breaks the circular-wait one. For now, internalize the trigger: **fine-grained locking that holds multiple locks needs a lock order, or it will deadlock.** Striped maps that only ever touch one stripe per operation sidestep this entirely, which is another reason striping is the pragmatic default — it keeps you in the one-lock-at-a-time regime where deadlock is impossible.

There is a database analogy worth naming here, because the same trade plays out at a larger scale. Databases face exactly this granularity question — should a transaction lock a whole table, a page, or a single row? — and they resolve it with the same dial plus a clever middle: row-level locks for concurrency, with *lock escalation* up to coarser locks when too many fine locks accumulate, and rigorous lock ordering to avoid the deadlocks fine locking invites. If you want to see how a production system manages this trade at scale — including how it *detects* the deadlocks it cannot prevent and aborts a victim transaction to break the cycle — the [database locks and deadlocks post](/blog/software-development/database/database-locks-and-deadlocks-in-production) is the same lesson at the storage-engine layer.

## Measured behavior: throughput as readers scale

Now the part the kit demands and the part that actually settles arguments: numbers, and how to get honest ones. The claim under test is "RW lock beats mutex for reads, striping beats both for mixed traffic." Let us see how to measure it without lying to ourselves, and what the curve actually looks like.

The benchmark is a shared map, fixed key set, $T$ worker threads each doing operations in a tight loop for a fixed wall-clock window, with a tunable read fraction $f$. We measure total operations per second. The honest-measurement checklist, which matters more than the numbers:

- **Warm up.** Run the loop untimed for a second or two first, so the JIT has compiled (on the JVM), caches are warm, and the allocator has settled. Timing cold code measures the compiler, not the lock.
- **Run many times and report a distribution, not one number.** Concurrency benchmarks are noisy — the OS scheduler, other processes, turbo-boost thermal throttling all add variance. Run 10-30 trials; report median and spread. A single run is a rumor.
- **Pin the read/write ratio explicitly and report it**, because it is *the* variable that flips the answer. "RW lock is faster" is meaningless without "...at 99% reads, 1 µs sections." Sweep $f$ from 50% to 99.9% and watch the curves cross.
- **Name the platform.** Core count, whether hyperthreading is on, x86 vs ARM (the memory model differs and affects atomic costs), the language runtime and version. An RW lock benchmark on a 4-core laptop tells you almost nothing about a 64-core server, because contention effects are super-linear in core count.
- **Watch for the bend-*down*.** The signature of lock contention is throughput that rises with threads, peaks, and then *falls* as you add more threads — because past the peak you are adding contention faster than parallelism. Where the curve bends down is the most informative point in the whole experiment.

Here is the shape of a representative result on a read-heavy mixed workload (95% reads, ~1 µs critical sections) as the thread count scales — these are illustrative orders of magnitude on a many-core x86 box, *not* a precise benchmark of your hardware, and you should reproduce them on yours:

| Threads | Single mutex (ops/s) | RW lock (ops/s) | Striped (16, ops/s) |
| --- | --- | --- | --- |
| 1 | ~5.0 M | ~4.6 M | ~4.8 M |
| 4 | ~4.8 M | ~9 M | ~17 M |
| 16 | ~4.5 M | ~14 M | ~55 M |
| 32 | ~4.0 M | ~12 M | ~80 M |
| 64 | ~3.5 M | ~9 M | ~95 M |

Read the story in the columns. The **single mutex** never scales — it is roughly flat and even *bends down* as threads rise, because every operation serializes on one lock and adding threads only adds contention and context-switch overhead. At 1 thread it is actually the *fastest* (no contention, cheapest bookkeeping); that is the crossover the short-section worked example warned about. The **RW lock** scales for a while — readers overlap, so 4 and 16 threads see real gains — but then it *bends down* past ~16 threads, because the reader-count cache line becomes the bottleneck and the readers re-serialize on the counter. This is the central irony made measurable: the RW lock helps until its own bookkeeping becomes the contended thing. The **striped** map scales the furthest and the best, because cross-key operations contend on *different* locks (different cache lines), so there is no single hot line to bottleneck on — until you saturate the stripes or the memory system.

#### Worked example: reading the crossover points

Three crossovers are worth naming precisely from that table. (1) At **1 thread**, the mutex wins — it has the least bookkeeping, and with no contention there is nothing for the fancier locks to improve; their overhead is pure loss. *Lesson: never use an RW lock or striping for a structure only one thread touches.* (2) Between **1 and 4 threads**, the RW lock and striped versions overtake the mutex, as soon as there is enough read concurrency to overlap. *Lesson: the benefit appears only under real concurrency.* (3) Around **16-32 threads**, the RW lock peaks and starts falling while striping keeps climbing. *Lesson: at high core counts, granularity (different cache lines) beats reader-sharing (one hot counter) — striping is the technique that scales to many cores, the RW lock is the technique that helps at modest concurrency.* If you only benchmarked at 4 threads you would conclude the RW lock and striping are comparable; the truth only shows up at 32+, which is exactly why "name the core count" is on the checklist.

Now flip the read fraction to test the losing case. At **50% reads** (write-heavy-ish), re-run and you will see the RW lock collapse to roughly mutex performance or *worse* — because half the operations are writes that exclude everyone, so the reader-parallelism advantage evaporates while the bookkeeping overhead remains. The striped map still wins at 50% reads (writes to different stripes still parallelize), which is a key insight: **striping helps write-heavy workloads, the RW lock does not.** The RW lock's win is specifically read parallelism; striping's win is *cross-key* parallelism, which covers writes too. That is why, for a write-heavy concurrent map, striping is the answer and an RW lock is not.

## Case studies / real-world

These techniques are not academic — they are load-bearing in systems you use every day, and the design choices are documented.

**`java.util.concurrent.ConcurrentHashMap` (lock striping in production).** The canonical real-world striped structure. In Java 5 through 7 it was explicitly an array of `Segment` objects (default 16, the `concurrencyLevel`), each segment a small `ReentrantLock`-guarded hash table — textbook striping. A `put` locked only the one segment its key hashed to, so up to 16 writers could proceed in parallel, while reads were largely lock-free (using `volatile` reads of the table). This is why `ConcurrentHashMap` scaled where the older `Hashtable` and `Collections.synchronizedMap` (both a single coarse lock around the whole map) flat-lined under concurrent load. In Java 8 the design was reworked to per-bucket locking (synchronize on the first node of each bin) plus CAS for empty-bin inserts, pushing the granularity even finer and making the common-case write lock-free — but the core idea is the same lineage: split the one hot lock into many. The `concurrencyLevel` constructor argument still exists (now mostly a sizing hint), a direct fossil of the stripe-count tuning knob.

**Read-mostly caches and the immutable-snapshot alternative.** Our opening config cache is the archetypal RW-lock candidate, and many production caches do use an RW lock for it. But the very-read-heavy, rarely-written case has an even better answer that this post should name honestly: **don't lock the readers at all** — store the whole config in one immutable object behind a single atomic reference (`AtomicReference` in Java, `atomic.Value` in Go, `arc-swap` in Rust, an `atomic<shared_ptr>` in C++), have readers do a single atomic load of the pointer and read freely from the immutable snapshot, and have the rare writer build a *new* snapshot and atomically swap the pointer. Readers never block, never touch a shared counter, and never contend — they just deref a pointer. This is read-copy-update (RCU) in spirit, and for a 99.999%-read config cache it beats an RW lock outright because it removes the reader-count cache line entirely. The lesson: the RW lock is the right tool when readers must see a *consistent mutable* structure under the lock; when the data can be a swappable immutable snapshot, atomic-pointer-swap is even better. Measure which one your access pattern is.

**The Big Kernel Lock and CPython's GIL (coarse locking's long tail).** The cautionary case studies. Early Linux used one "Big Kernel Lock" (BKL) — a single coarse lock serializing huge swaths of the kernel; the multi-year effort to remove it, replacing it with fine-grained per-subsystem locks, was one of the defining scalability projects of the 2.x/3.x era, and it is the granularity dial being turned the hard way under a running operating system. CPython's [Global Interpreter Lock](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) is the same shape: one coarse lock around the interpreter that makes the implementation simple and single-threaded code fast, at the cost of CPU-bound multithreading, and the free-threaded ("no-GIL") work is, again, the granularity dial being turned toward fine-grained locking — with all the complexity and subtle-bug cost that this post says fine-grained locking carries. Both stories are the same trade at OS and runtime scale: coarse is simple and serial, fine is complex and parallel, and moving from one to the other is expensive and bug-prone.

## When to reach for this (and when not to)

Decisive recommendations, because every technique here is a cost you should refuse to pay unless you have measured the need.

**Reach for an RW lock when** the workload is genuinely read-heavy (reads strongly outnumber writes — think 10:1 or more) *and* the critical section is long enough that overlapping reads saves meaningful time (microseconds, not nanoseconds), *and* you know your library's fairness policy and it does not starve your writers. The config cache that parses on read, the in-memory index walked deeply per query, the routing table with expensive per-request lookups — these are the wins.

**Do not reach for an RW lock when** the section is tiny (a single field read or map lookup — the bookkeeping eats the gain), or the workload is write-heavy or balanced (writes serialize either way; you pay overhead for nothing), or the structure is touched by only one thread (no contention to relieve), or the data can instead live behind an atomic-pointer-swapped immutable snapshot (then readers need no lock at all). And never reach for an RW lock you cannot characterize for fairness — an unspecified-policy `shared_mutex` is a latent starvation bug.

**Reach for striping when** you have a structure with *independent* elements (a map, a set, a counter table) under multi-core write or mixed contention, and one coarse lock is your measured bottleneck. Striping is the highest-leverage, lowest-risk granularity move there is: bounded lock count, $1/N$ contention, and — as long as each operation touches one stripe — no deadlock exposure. It is the right default for a contended concurrent map and it scales further than an RW lock.

**Reach for hand-over-hand (or, better, an off-the-shelf concurrent skip list / lock-free structure) when** you have a *linked* structure whose traversal is the measured bottleneck and coarse locking it is serializing too much. But treat hand-rolled lock coupling as a last resort: it is unforgiving, every operation must obey the same discipline, and a concurrent skip list from the standard library is usually the better destination.

**Leave the single mutex exactly where it is when** — and this is the most common correct answer, so say it loudly — the lock is *not your bottleneck*. If the structure is touched by one thread, or the contention is low, or the section is short and the throughput is already fine, then the RW lock, the striping, and the hand-over-hand are all pure added complexity and added bug surface for no measured gain. The right sequence is always: measure first, find that the lock is actually the bottleneck, then reach for the cheapest technique that relieves it. A plain mutex is correct, simple, deadlock-resistant, and fast enough astonishingly often. The whole [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) is "use the simplest mechanism that buys the order you need" — and for a great many shared structures, that mechanism is one boring mutex.

## Key takeaways

1. **A readers-writer lock encodes the one fact a mutex ignores: readers do not conflict with readers.** Its contract is $w(t) \le 1$ and a writer excludes all readers — so any number of readers run in parallel, only writers are exclusive. A mutex is the special case where every acquisition is exclusive.
2. **The RW lock's reader count and writer flag are not free.** Each acquisition is a conditional atomic on a shared word, and under heavy read concurrency that counter cache line can become the bottleneck — the bookkeeping that enables reader overlap can itself prevent it.
3. **Fairness policy is a correctness decision, not a tuning knob.** Reader preference starves writers under continuous reads; writer preference can starve readers under continuous writes; fair bounds both at a throughput cost. An RW lock with an unspecified policy is a latent starvation bug — know your library's behavior from the docs.
4. **An RW lock wins only for read-heavy, long-section workloads.** For short sections the bookkeeping eats the gain; for write-heavy or balanced workloads writes serialize anyway and you pay overhead for nothing. Measure the read fraction and the section length before reaching for one.
5. **Granularity is a dial.** Coarse: simple, deadlock-free, does not scale. Fine: scales, complex, deadlock-prone, memory-hungry. The sweet spot is rarely at either extreme.
6. **Lock striping is the pragmatic middle and the right default for contended independent-element structures.** A fixed pool of $N$ locks, keys hashed to stripes, gives about $1/N$ contention with bounded lock count — and as long as each operation touches one stripe, zero deadlock risk. It scales further than an RW lock and helps write-heavy workloads too.
7. **Hand-over-hand locking traverses linked structures by always grabbing the next lock before releasing the current one** — a sliding two-lock window with no gap. The overlap is what prevents another thread from splicing the path out from under you. It is correct but unforgiving; prefer a standard concurrent skip list when you can.
8. **The instant an operation holds two or more locks, deadlock is possible — impose a global lock order.** Coarse locking is immune; fine and multi-stripe locking are exposed; lock ordering breaks the circular-wait condition. Striping that touches one stripe per op sidesteps this entirely.
9. **For very-read-heavy data that can be an immutable snapshot, beat the RW lock entirely with atomic-pointer-swap** (RCU-style): readers deref a pointer and never block; the rare writer builds a new snapshot and swaps it in.
10. **Measure honestly or do not claim a win.** Warm up, run many trials, pin and report the read/write ratio, name the core count and platform, and watch for the throughput curve that bends *down* — the bend is where contention overtakes parallelism, and it moves with every variable.

## Further reading

- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming*** — Chapter 8 on monitors and readers-writers locks, and Chapter 9 on linked lists, derives hand-over-hand (lock coupling), optimistic, and lazy synchronization rigorously. The definitive treatment.
- **Brian Goetz et al., *Java Concurrency in Practice*** — the chapters on building blocks and on `ConcurrentHashMap`'s striped design; the canonical explanation of why fine-grained and lock-striped structures scale where coarse-locked ones do not.
- **Anthony Williams, *C++ Concurrency in Action*** — `std::shared_mutex`, `std::shared_lock`, and the design of fine-grained and lock-free concurrent data structures, with the C++ memory model underneath.
- **Doug Lea, "The java.util.concurrent Synchronizer Framework"** — the AQS paper behind `ReentrantReadWriteLock`; how the reader count and writer state are packed into one atomic word and how fairness is implemented.
- **Paul McKenney, "What is RCU, Fundamentally?"** (LWN) — read-copy-update, the read-mostly technique that beats the RW lock for the very-read-heavy snapshot case; the kernel's answer to reader scalability.
- Within this series: start at [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), build the foundation with [mutual exclusion: mutexes and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections), see what fine-grained locking risks in [deadlock: the four conditions and how to break them](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them), and zoom out to the decision framework in [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

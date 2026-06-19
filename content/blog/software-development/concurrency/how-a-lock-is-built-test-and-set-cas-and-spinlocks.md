---
title: "How a Lock Is Built: Test-and-Set, CAS, and Spinlocks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Build a mutex from raw hardware: the atomic instructions, a spinlock from test-and-set, the cache thrash and the fixes, fairness, and the futex underneath every blocking lock."
tags:
  [
    "concurrency",
    "parallelism",
    "spinlock",
    "compare-and-swap",
    "atomics",
    "locks",
    "cas",
    "low-level",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-1.png"
---

The first time you write `mutex.lock()`, it feels like magic. You call a function, and somehow — across cores, across caches, across a scheduler that can preempt you at any instruction — exactly one thread gets to run the next few lines. The balance update lands. The linked-list pointer swings cleanly. No torn writes, no lost updates. It just works. And because it just works, most engineers never ask the question that turns a working programmer into a systems programmer: *what, physically, is a lock?*

The answer is humbling and clarifying at once. A lock is not magic. A lock is a single integer in memory — usually one machine word — and a hardware instruction that can read it, decide, and write it back, all in one indivisible step that no other core can interleave with. That's it. Everything else — fairness, blocking, the kernel, the elaborate machinery of `pthread_mutex_t` or `std::sync::Mutex` — is software built on top of that one hardware guarantee. If you understand the instruction and the cache-coherence protocol it rides on, the mutex stops being magic and becomes something you can reason about, profit from, and occasionally beat with something simpler.

This post builds a lock from the metal up. We start with the atomic read-modify-write instructions every modern CPU exposes — test-and-set, compare-and-swap (CAS), and fetch-and-add — and how each one maps to a real opcode like x86 `lock cmpxchg` or ARM's load-linked / store-conditional pair. We build a spinlock from test-and-set in a dozen lines, then watch it melt under contention because of the cache-coherence protocol, and fix it with the test-and-test-and-set trick. We derive the spin-versus-block crossover against the cost of a context switch (roughly 1–5 microseconds), make our spinlock *fair* with ticket locks and MCS locks, and finally see how a production mutex avoids the kernel entirely on the happy path using a futex — one userspace CAS when uncontended, a syscall only when threads actually collide. Along the way we measure: spinlock versus mutex throughput and fairness under contention, honestly, with the confounds named.

This is the third move in this series' recurring discipline. We've named [shared mutable state and the race it creates](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition), and we've named the [mutex and the critical section it protects](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) as the tool that imposes a happens-before order. Now we open the mutex up and find the hardware. By the end you'll know exactly what costs a lock acquire pays, when a spinlock is faster than a mutex and when it's a catastrophe, and why "just use a lock" is excellent advice precisely *because* someone already did this hard engineering for you.

![A broken load-then-store lock lets two threads enter while a single atomic compare and swap flips the flag from free to held and only one thread wins](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-1.png)

## The problem a lock solves, restated as one flag

Strip a lock down to its job. We have a shared resource — a bank balance, a free-list head, a hash-table bucket — and we want a *critical section*: a stretch of code that runs for one thread at a time. The classic way to enforce that is a single shared variable, call it `flag`, with two states: `0` means *free*, `1` means *held*. To acquire the lock, a thread changes the flag from `0` to `1`. To release, it sets it back to `0`. Mutual exclusion holds if and only if no two threads can both observe `0` and both set it to `1`.

Here's the trap, and it's the same trap that defines almost every concurrency bug: the naive acquire is *two* operations, not one.

```c
// BROKEN acquire — a textbook race, do not ship this
while (flag != 0) { /* wait */ }   // step 1: read, see 0 (free)
flag = 1;                          // step 2: write 1 (held)
// ... critical section ...
```

Walk the interleaving. Thread T1 executes step 1 and sees `0`. Before it runs step 2, the scheduler preempts it — or on another core, T2 runs concurrently — and T2 also executes step 1 and *also* sees `0`. Now both threads believe the lock is free. Both run step 2, both set `flag = 1`, and both enter the critical section. Mutual exclusion is violated, and the bug surfaces as a corrupted balance or a freed-twice pointer long after the actual race. There is a window — the gap between the read and the write — where the decision was made on stale information. That window is the entire problem.

The fix is hardware that closes the window. We need an instruction that reads the flag, decides, and writes it back as one *atomic* unit — atomic meaning no other core can observe or modify the flag in the middle. The hardware vendors give us exactly this family of instructions, called *read-modify-write* (RMW) operations, and a lock is nothing more than a clever use of one of them. The figure above contrasts the broken load-then-store with the correct one-instruction flip: the broken version has a gap two threads can slip through; the correct version is a single atomic compare-and-swap that exactly one thread can win.

That single idea — collapse "read, decide, write" into one indivisible step — is the foundation of every lock, every atomic counter, and every lock-free data structure in existence. So let's meet the instructions.

## The atomic read-modify-write instructions

A read-modify-write instruction does three things — reads a memory location, computes a new value, writes it back — and guarantees that the whole sequence appears instantaneous to every other core. No other thread can read or write that location between the read and the write. The CPU enforces this by locking the cache line (or the memory bus on older hardware) for the duration. Three RMW primitives matter for building locks, and they form a small hierarchy of power.

![A matrix of the three atomic read-modify-write primitives showing what each does its return value and the matching x86 and ARM instruction](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-2.png)

**Test-and-set (TAS)** is the simplest. It atomically writes `1` into a memory location and returns the *old* value. If the old value was `0`, you just claimed something that was free — you won. If it was `1`, it was already taken — you lost, and you didn't change anything meaningful (it was `1`, you set it to `1`). In one atomic step you both test the previous state and set the new one. On x86 the natural mapping is `xchg` (exchange), which swaps a register with memory atomically — `xchg` carries an *implicit* lock prefix, so it's always atomic without writing `lock` explicitly. ARM has no single TAS instruction; it builds one from its load-linked / store-conditional pair, which we'll see in a moment.

**Compare-and-swap (CAS)** is the workhorse and the most general of the three. It takes three arguments: a memory location, an *expected* value, and a *new* value. Atomically, it checks whether the location currently holds the expected value; if so, it writes the new value and reports success; if not, it leaves memory unchanged and reports failure (often returning the actual current value so you can retry). In C-like pseudocode the semantics are:

```c
// Semantics of CAS — executed ATOMICALLY by hardware, shown as if-block
bool cas(int *addr, int expected, int new_val) {
    if (*addr == expected) {   // compare
        *addr = new_val;       // swap (only if compare held)
        return true;           // success
    }
    return false;              // someone changed it first
}
```

That whole if-block runs as one indivisible instruction. On x86 it's `lock cmpxchg` — the `cmpxchg` instruction compares the accumulator `EAX`/`RAX` against the destination and conditionally swaps, and the `lock` prefix makes the read-modify-write atomic across cores. CAS is strictly more powerful than test-and-set: with TAS you can only flip to `1`, but with CAS you can conditionally change *any* value based on *any* expected value, which is what makes it the universal primitive for lock-free algorithms. (Maurice Herlihy proved CAS has unbounded *consensus number*, the formal sense in which it's the most powerful primitive — but you don't need the theory to feel its reach: nearly everything non-trivial in concurrent data structures is a CAS loop.)

**Fetch-and-add (FAA)** atomically adds a number to a memory location and returns the value *before* the add. It's how you build a correct concurrent counter without a lock (`fetch_add(&counter, 1)` is the entire body), and — crucially for this post — it's how you hand out monotonically increasing ticket numbers in a fair lock. On x86 it's `lock xadd` (exchange-and-add): `xadd` swaps the register and memory then adds, and the `lock` prefix makes it atomic. The figure above lays out all three side by side — what each computes, what it returns, and the ISA instruction it compiles to.

### Two ISA shapes: x86 locked instructions versus ARM load-linked / store-conditional

There are two architectural philosophies for atomics, and they change how you reason about the cost. **x86 takes the "locked instruction" approach**: dedicated opcodes (`xchg`, `cmpxchg`, `xadd`) that, with the `lock` prefix, perform the whole read-modify-write atomically. The hardware acquires exclusive ownership of the cache line, does the operation, and releases — the programmer sees a single instruction that cannot fail spuriously. Disassemble an atomic increment and you'll find one `lock incl` or `lock xadd`.

```asm
; x86-64: atomic compare-and-swap of a lock word, GNU/AT&T syntax
;   try to swap *lockptr from 0 (free) to 1 (held)
        mov     $1, %ecx          ; new value = 1 (held)
        xor     %eax, %eax        ; expected = 0 (free) in EAX
        lock cmpxchg %ecx, (%rdi) ; if *rdi==EAX: *rdi=ECX, ZF=1; else EAX=*rdi, ZF=0
        jne     .Lcontended       ; ZF=0 means CAS failed -> someone holds it
        ; ZF=1: we own the lock, fall through into the critical section
```

**ARM (and other RISC machines like RISC-V and POWER) takes the load-linked / store-conditional approach**, abbreviated LL/SC. Instead of one fused instruction, you do two: a special load that *links* — `ldxr` on AAr64, "load exclusive register" — marks the address as monitored, and a conditional store — `stxr`, "store exclusive register" — succeeds *only if* nothing wrote the monitored address since the link. If another core touched the line, `stxr` fails and reports it, and you loop. CAS, TAS, and FAA are all *built* from this pair in software:

```asm
; AArch64: atomic compare-and-swap built from LL/SC
;   swap *x0 from w1 (expected) to w2 (new); loop on contention
.Lretry:
        ldxr    w3, [x0]          ; load-exclusive: read flag, start monitoring
        cmp     w3, w1            ; does it equal expected?
        b.ne    .Lfail            ; no -> someone else owns it, bail out
        stxr    w4, w2, [x0]      ; store-exclusive: write new; w4=0 if it stuck
        cbnz    w4, .Lretry       ; store failed (line was touched) -> retry
        ; success: we performed the swap atomically
.Lfail:
```

The practical difference: LL/SC can **fail spuriously** — `stxr` may report failure even with no real contention (a cache-line eviction, an interrupt, a nearby store to the same line). That's why ARM code loops on `stxr`. It also means a single LL/SC sequence is naturally a *retry loop*, which is the right way to reason about CAS everywhere: a CAS is something you do *in a loop*, expecting occasional failure, not a one-shot you can assume succeeds. (Modern ARMv8.1+ adds true single-instruction atomics — `CAS`, `LDADD`, `SWP` — under the Large System Extensions, so the newest chips look more like x86; but the LL/SC model is what shaped the APIs and the retry-loop habit.) Either way, the language wrappers hide the difference: when you write `compare_exchange` in Rust or `compareAndSet` in Java, the compiler emits `lock cmpxchg` on x86 and an `ldxr`/`stxr` loop (or a native `CAS`) on ARM. You program against the *concept*; the ISA fills in the shape.

#### Worked example: why `count++` needs an atomic and what the instruction looks like

Take the running example of this series — a shared counter incremented by many threads. The C statement `count++` compiles, on x86, to roughly three instructions: a load of `count` into a register, an increment of the register, and a store back to memory. Three instructions means two preemption windows. If T1 loads `count=41`, T2 loads `count=41`, both increment to `42`, both store `42` — you ran two increments and the counter advanced by one. That's the *lost update*, the canonical race.

The atomic fix replaces the three instructions with one. Instead of load/inc/store, the compiler emits a single `lock xadd` (or `lock incl` when you don't need the old value):

```c
// BUG: three instructions, a lost-update race under concurrency
count++;                       // load count; inc; store count

// FIX (C11): one atomic fetch-and-add, compiles to `lock xadd` on x86
atomic_fetch_add(&count, 1);   // read-modify-write, indivisible
```

The fix isn't "add a fence" or "be careful" — it's a structurally different instruction that the hardware refuses to interleave. There is no window because there is no gap between the read and the write. That is the whole game, and a lock is the same trick applied to a flag instead of a counter.

It's worth being precise about *why* the hardware can promise this, because it's the foundation everything else rests on. When a core executes `lock xadd`, it asserts exclusive ownership of the affected cache line for the duration of the operation — on modern CPUs this is done at the cache-coherence level (the core holds the line in the Modified state and refuses coherence requests for the line until the RMW completes), not by literally locking the whole memory bus as the oldest x86 chips did. During that brief exclusive window, no other core can read or write the line, so no other core can observe a half-finished increment. The cost is real but small: an uncontended atomic RMW on a line you already own is on the order of a few nanoseconds to low tens of nanoseconds (a handful of cycles plus the ordering effects), versus sub-nanosecond for a plain non-atomic increment. That cost — exclusive ownership plus the memory fence — is the *price of atomicity*, and it's the same price a lock acquire pays, because a lock acquire *is* an atomic RMW. Keep this number in mind: a lock isn't slow because of "the lock"; it's "slow" by exactly the cost of one atomic instruction, and only when contended does it cost more.

## Building a spinlock from test-and-set

Now we have the primitive; let's build the lock. A *spinlock* is the simplest possible lock: to acquire, atomically try to flip the flag from free to held, and if it's already held, *spin* — loop, retrying — until it frees. No blocking, no kernel, no queue. Just a tight loop hammering an atomic instruction. Here it is in C using GCC/Clang's `__atomic` builtins (`test_and_set` semantics via an exchange):

```c
#include <stdatomic.h>

typedef struct { atomic_flag held; } spinlock_t;
// atomic_flag initializes to clear (0) with ATOMIC_FLAG_INIT

void spin_lock(spinlock_t *l) {
    // atomic_flag_test_and_set: set to true, return PREVIOUS value
    // returns true while the lock is held -> keep spinning
    while (atomic_flag_test_and_set_explicit(&l->held, memory_order_acquire)) {
        // busy-wait: the lock is held, try again
    }
}

void spin_unlock(spinlock_t *l) {
    // clear to false (0) with release ordering so the critical section's
    // writes are visible to the next acquirer before it sees the unlock
    atomic_flag_clear_explicit(&l->held, memory_order_release);
}
```

Two details carry the correctness and they are worth dwelling on. First, `atomic_flag_test_and_set` *is* the test-and-set primitive: it sets the flag to `true` and returns the old value, atomically. While the lock is held the old value is `true`, so the loop keeps spinning; the moment a release sets it to `false`, our test-and-set returns `false`, we stop spinning, and — because we already set it to `true` in the same atomic step — we hold the lock. The test and the set are one instruction, so there's no window for two threads to both succeed. Exactly one test-and-set sees the `0`-to-`1` transition.

Second, the *memory orderings*. The acquire on lock and release on unlock are not decoration — they're what make the critical section actually exclusive of *data*, not just of the flag. `memory_order_acquire` on the successful test-and-set prevents the CPU and compiler from hoisting reads of the protected data *above* the lock acquisition; `memory_order_release` on unlock prevents writes inside the critical section from sinking *below* the unlock. Together they establish the happens-before edge from "thread A's unlock" to "thread B's subsequent acquire," which is what guarantees B sees everything A did. This is the typed interface to the memory model, and it's the whole subject of [atomics and memory orderings from relaxed to seq-cst](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) — here we just use acquire/release correctly and move on.

The same lock in **Rust** uses `AtomicBool` and `compare_exchange`, which makes the acquire/release explicit in the API and lets the type system enforce that you can't forget to unlock (an `Acquire` returns a guard whose `Drop` releases):

```rust
use std::sync::atomic::{AtomicBool, Ordering};

pub struct SpinLock {
    held: AtomicBool, // false = free, true = held
}

impl SpinLock {
    pub fn lock(&self) {
        // compare_exchange_weak: try to swap false -> true.
        // _weak may fail spuriously (cheap on LL/SC like ARM), so we loop.
        while self
            .held
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // spin until it frees
            std::hint::spin_loop(); // emits PAUSE on x86, YIELD on ARM
        }
    }

    pub fn unlock(&self) {
        self.held.store(false, Ordering::Release);
    }
}
```

Notice `compare_exchange_weak` — the "weak" variant is allowed to fail spuriously, which is *exactly* the LL/SC behavior on ARM, so on those architectures it compiles to a bare `ldxr`/`stxr` loop with no extra retry wrapper. On x86 it's a `lock cmpxchg`. The `std::hint::spin_loop()` call emits the architecture's spin-wait hint (`PAUSE` on x86, `YIELD` on ARM) — a hint to the CPU that we're in a busy-wait, which reduces power draw and, on hyperthreaded cores, lets the sibling thread make progress. We'll come back to why that hint matters.

The **Go** and **Java** standard libraries don't expose a raw spinlock as a public type — they steer you to `sync.Mutex` and `synchronized`, which block — but both give you the atomic primitive to build one, and seeing the CAS spelled out in each cements that it's the same instruction underneath:

```go
import "sync/atomic"

type SpinLock struct{ held atomic.Bool }

func (l *SpinLock) Lock() {
    // CompareAndSwap(old, new) returns true if it swapped.
    // Loop while we fail to swap false -> true.
    for !l.held.CompareAndSwap(false, true) {
        // runtime.Gosched() here would yield to the scheduler;
        // a pure spin just retries.
    }
}

func (l *SpinLock) Unlock() { l.held.Store(false) }
```

```java
import java.util.concurrent.atomic.AtomicBoolean;

final class SpinLock {
    private final AtomicBoolean held = new AtomicBoolean(false);

    void lock() {
        // compareAndSet(expect, update) returns true on success.
        while (!held.compareAndSet(false, true)) {
            Thread.onSpinWait(); // JDK 9+: emits the CPU spin hint (PAUSE)
        }
    }

    void unlock() { held.set(false); }
}
```

Four languages, one idea: a loop around a single atomic instruction that flips a flag from free to held and tells you whether you won. That is a spinlock. It is correct — it never lets two threads into the critical section. But "correct" is not "good," and the next section is about why this exact code, under real contention, can make your 32-core server slower than one core.

Before we leave the code, one detail that trips people up: notice the C and Rust versions both pass an *explicit* memory ordering, while the Go and Java snippets don't. That's not because Go and Java skip the ordering — it's because their atomics are *sequentially consistent by default*. Go's `atomic.Bool.CompareAndSwap` and Java's `AtomicBoolean.compareAndSet` both imply full acquire-and-release semantics (effectively sequential consistency) with no way to relax them in the basic API; you trade the ability to pick a cheaper ordering for the safety of never accidentally picking too weak a one. C/C++ and Rust expose the full `memory_order` / `Ordering` menu, which lets an expert shave the fence cost on a hot path — `Acquire` on the lock, `Release` on the unlock, and `Relaxed` for the spurious-failure path — but also lets a non-expert write a subtle bug. The lesson generalizes across the series: the *languages that let you go faster* are the ones that let you be wrong, and the right default in all of them is acquire on lock, release on unlock, nothing weaker.

## Why a naive test-and-set spinlock thrashes the cache

The spinlock above has a performance bug so severe that it's worth a section on its own, because the *reason* is the cache-coherence protocol — the same machinery behind false sharing, and the thing that separates engineers who guess about performance from those who can predict it.

![A naive test and set spinlock writes the cache line on every spin causing the line to ping pong between cores while test and test and set spins on a shared read and only writes once when the lock frees](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-4.png)

Recall that a CPU doesn't read and write individual bytes from main memory; it moves *cache lines* (typically 64 bytes) between RAM and per-core caches, and a coherence protocol (MESI: Modified, Exclusive, Shared, Invalid) keeps the copies consistent. The key rule: **to write a cache line, a core must hold it in the Modified or Exclusive state, which means it must have exclusive ownership — every other core's copy is Invalidated.** A read can be shared across many cores (Shared state); a write cannot.

Now look at what the naive spinlock does while waiting. `atomic_flag_test_and_set` is a *write* — it stores `true` into the flag *every single iteration*, even when the lock is already held and the store changes nothing (`true` to `true`). Every write demands exclusive ownership of the lock's cache line. So when eight cores are all spinning on a held lock, here's the dance: core 1 grabs the line exclusive to do its `xchg`, which invalidates cores 2 through 8. Core 2 then grabs it exclusive for *its* `xchg`, invalidating 1 and 3 through 8. Core 3 grabs it next, and so on. The single cache line containing the lock *ping-pongs* between cores at the speed of the coherence interconnect — tens to hundreds of nanoseconds per transfer — and it does so *continuously*, even though none of these cores can make progress because the lock is still held by someone else entirely. The waiters are generating a storm of coherence traffic that competes with, and slows down, the very thread holding the lock — the one core whose progress would actually free everyone.

This is why a naive TAS spinlock doesn't just waste the spinning cores' cycles; it can make the whole system *slower* as you add cores, because contention on the coherence fabric grows super-linearly. Throughput goes *down* under load. The figure above shows the before state: every spin is a write, the line bounces between cores, and bus traffic climbs with the core count.

### The fix: test-and-test-and-set

The fix is elegant and is the canonical "test-and-test-and-set" (TTAS) optimization. The insight: **a read can be shared; only a write must be exclusive.** So instead of hammering an atomic write while waiting, *spin on a plain read* and only attempt the expensive atomic write when the read suggests the lock might be free:

```c
void spin_lock_ttas(spinlock_t *l) {
    for (;;) {
        // INNER: spin on a plain (non-atomic) READ — line stays Shared,
        // cached locally, no coherence traffic while it's held.
        while (atomic_load_explicit(&l->held_int, memory_order_relaxed) != 0) {
            __builtin_ia32_pause(); // x86 PAUSE hint while spinning
        }
        // The read says it MIGHT be free; now do the one expensive
        // atomic test-and-set to actually claim it.
        if (atomic_exchange_explicit(&l->held_int, 1, memory_order_acquire) == 0)
            return; // we won
        // else: lost the race between the read and the exchange -> back to spinning
    }
}
```

The structure is two nested loops. The *inner* loop spins on `atomic_load` — a read — so the cache line sits in the Shared state, cached locally on every waiting core, generating **zero coherence traffic** while the lock stays held. Each waiter reads its own local copy at full L1 speed and burns nothing on the interconnect. Only when the holder releases (writing `0`, which invalidates the shared copies *once*) do the waiters notice the change, fall out of the inner loop, and race to do the *single* atomic exchange. Yes, there's now a thundering herd at release time — all waiters wake and one wins — but the steady-state waiting cost drops from "continuous ping-pong" to "nearly free local reads." That's the difference between a spinlock that scales acceptably to a handful of cores and one that collapses. This is the "test, then test-and-set" name: test cheaply with a read, and only then do the expensive test-and-set.

#### Worked example: counting the coherence transfers

Make it concrete. Suppose a lock is held for a window during which 7 other cores are spinning, and the coherence fabric needs about 100 ns to transfer an exclusive cache line between cores. With the **naive TAS** spinlock, all 7 waiters write every iteration; if each iteration is ~20 ns of CPU work, each waiter forces a line transfer roughly every iteration, so the lock's cache line is being yanked between cores on the order of *millions* of times per second of contention — and the holder, who needs that same line to release the lock, is fighting the herd for it the entire time, stretching the critical section. With **TTAS**, the waiters read their local Shared copies and produce *no* transfers until release; the line moves exclusively only twice: once when the holder writes `0` to release (invalidating the 7 shared copies in one shot), and once when the winning waiter writes `1` to acquire. Two transfers per handoff versus a continuous storm. On a real machine this is routinely a 5–20× throughput difference under heavy contention — not a micro-optimization, a different asymptotic behavior. (The exact factor depends on core count, interconnect, and hold time; treat these as order-of-magnitude, the kind you confirm by measuring, not memorizing.)

## Spin versus block: the crossover math

A spinlock, even a well-behaved TTAS one, makes a stark bet: *I will burn a CPU core doing nothing useful, betting the lock frees up soon enough that spinning is cheaper than the alternative.* The alternative is to *block* — tell the OS scheduler "park me; wake me when the lock is free" — which frees the core for other work but pays the cost of two context switches (out and back in). Whether spinning or blocking wins is not a matter of taste; it's arithmetic, and you can derive the crossover.

![A matrix comparing spinning against blocking across short holds long holds oversubscribed cores and a uniprocessor showing spin wins only for short holds on spare cores](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-5.png)

Let $C$ be the cost of a context switch — roughly the time to save one thread's registers, run the scheduler, and restore another thread's state. On modern hardware the *direct* cost is about 1–5 microseconds, and the *indirect* cost (cold caches and TLB after the switch) can be several times that, but use $C \approx 1\text{–}5\,\mu s$ as the working figure. Let $H$ be the expected time you'd wait for the lock — roughly the lock's hold time, since you wait for the current holder to finish. The decision is:

- **If you spin**, you waste your own core for the wait $H$. Cost to you: about $H$ of burned CPU.
- **If you block**, you pay $\approx 2C$ — one switch to park, one to wake — but you free the core for other threads during $H$.

Spinning is the cheaper bet when the wasted spin time is less than the switch overhead you'd otherwise pay:

$$H < 2C \approx 2\text{–}10\,\mu s.$$

In words: **spin only if you expect the lock to be free in less time than a context switch round-trip; otherwise block.** If the critical section is a handful of instructions — incrementing a counter, swapping a pointer — the hold time is tens of nanoseconds, far below $2C$, so spinning wins easily; paying a 1–5 µs context switch to wait 50 ns would be absurd. But if the critical section does I/O, allocates memory, or holds the lock across a system call, the hold time can be milliseconds, and spinning would burn a core for thousands of context-switch-equivalents while other threads starve. The figure above tabulates the four cases: short hold favors spin, long hold favors block, and two situations (oversubscription and uniprocessor) force blocking regardless — for reasons worth their own subsection.

There's a sharper way to state the same trade-off that exposes a hidden subtlety: you don't actually know $H$ in advance, so the real decision is over a *distribution* of hold times. The optimal strategy when the hold-time distribution is unknown is a competitive one — spin for up to $C$ (one context switch worth of time), and if the lock still hasn't freed, give up and block. This bounds your worst case: you never waste more than $C$ spinning before paying the $\approx 2C$ to block, so you're within a factor of about 3 of the offline-optimal choice no matter what the holder does. That "spin for one context-switch's worth, then park" rule is provably near-optimal (it's the classic competitive-analysis result for the spin-block problem), and it is *exactly* what adaptive mutexes implement. The number to internalize: a few thousand spin iterations or roughly a microsecond of spinning is the sweet spot before you cut your losses and block. Spinning longer than that is gambling that the holder is about to finish, and the math says stop gambling.

### Two cases where you must never spin

The crossover assumes a *spare core* to spin on. Two situations break that assumption and turn spinning from a tuning decision into a correctness-adjacent disaster.

**The uniprocessor (or one runnable core).** On a single-core machine, spinning on a lock held by another thread is a deadlock-in-slow-motion. You're spinning, which means *you* are using the only core. The thread holding the lock can't run — there's no core for it — so it can never release. You spin for your entire scheduling quantum doing nothing, the scheduler eventually preempts you, the holder runs and releases, and you got lucky. On a uniprocessor a spinlock degrades to "waste a full time slice, then block anyway." The right move is always to block (yield the CPU) so the holder can run. This is why the Linux kernel's spinlocks become no-ops that just disable preemption on a `CONFIG_SMP=n` (uniprocessor) build — there's literally no one to spin against.

**Oversubscription (more runnable threads than cores).** Even with many cores, if you have more threads than cores — say 64 threads on 8 cores — the thread holding the lock may not be currently scheduled on *any* core. It's been preempted, sitting in the run queue, while *your* core spins waiting for it. You're burning a core that the holder could have used to finish and release. This is the classic spinlock pitfall on a busy server or, catastrophically, on a hypervisor: a virtual CPU spinning on a lock held by another vCPU that the hypervisor has descheduled. The "Lock-Holder Preemption" problem can inflate a critical section by an entire scheduling quantum (tens of milliseconds), turning a 50 ns spin into a 30 ms stall and torpedoing throughput. It's why naive spinlocks are dangerous in VMs and why hypervisors and guest kernels added *paravirtualized* spinlocks that detect a preempted holder and yield instead of spinning blindly.

#### Worked example: the hold-time decision in microseconds

Suppose you measure two critical sections. Section A increments a shared histogram bucket: ~30 ns of work, lock held ~40 ns including acquire/release. Section B looks up a key in a memory-mapped file, occasionally touching disk: usually ~2 µs, but a page fault makes it ~5 ms. For Section A, $H \approx 40\,\text{ns} \ll 2C \approx 4\,\mu s$, so a spinlock is the right call — paying a context switch to wait 40 ns is 100× overkill. For Section B, the *median* $H \approx 2\,\mu s$ is already near the crossover and the *tail* $H \approx 5\,\text{ms}$ is catastrophic for spinning (you'd burn a core for 5 ms). So Section B must block. The production answer for "I don't know the hold time" is an **adaptive mutex**: spin briefly (a bounded number of iterations, ~tens of nanoseconds to a microsecond), and if the lock hasn't freed, fall back to blocking. That's exactly what `pthread_mutex` adaptive locks, the JVM's biased/adaptive locking, and Go's `sync.Mutex` do — spin a little for the common short-hold case, block for the long-hold case, getting the best of both. Measure your hold time; let the number choose.

## Fairness: the basic spinlock starves, ticket locks and MCS locks fix it

Our TTAS spinlock has another flaw, subtler than cache thrash and just as real in production: it's **unfair**. When the lock releases and the thundering herd of waiters all race to do the atomic exchange, *which* one wins is whatever the cache-coherence protocol decides — typically the core that's physically closest on the interconnect or happened to have the line. There's no notion of "who's been waiting longest." A thread can lose the race over and over, starving indefinitely while threads that arrived later sail through. Under sustained contention this shows up as a brutal latency distribution: median acquire time looks fine, but the 99.9th percentile is a thread that's been spinning for a hundred milliseconds because it keeps losing the coherence lottery. Fairness is not a luxury for a tail-latency-sensitive service; it's a requirement.

### Ticket locks: fetch-and-add for FIFO order

The *ticket lock* makes acquisition first-come-first-served using the deli-counter idea: take a number, wait until it's called. It uses two counters — `next_ticket` (the next number to hand out) and `now_serving` (the number currently allowed in). To acquire, atomically *fetch-and-add* `next_ticket` to grab your unique ticket, then spin until `now_serving` equals your ticket. To release, increment `now_serving`, which lets in exactly the thread holding the next number.

![A timeline showing a thread take ticket five then spin while now serving climbs from three to four to five at which point it enters the critical section in strict arrival order](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-6.png)

```c
typedef struct {
    atomic_uint next_ticket;  // hand these out, monotonically
    atomic_uint now_serving;  // whose turn it is right now
} ticket_lock_t;

void ticket_lock(ticket_lock_t *l) {
    // fetch-and-add: atomically grab a unique ticket, get the old value
    unsigned my = atomic_fetch_add_explicit(&l->next_ticket, 1,
                                            memory_order_relaxed);
    // spin (on a READ — TTAS-style, no write traffic) until it's our turn
    while (atomic_load_explicit(&l->now_serving, memory_order_acquire) != my)
        __builtin_ia32_pause();
    // now_serving == my: we hold the lock, in strict FIFO order
}

void ticket_unlock(ticket_lock_t *l) {
    unsigned next = atomic_load_explicit(&l->now_serving, memory_order_relaxed) + 1;
    atomic_store_explicit(&l->now_serving, next, memory_order_release);
}
```

The fairness is exact: tickets are handed out by an atomic fetch-and-add, so they're strictly increasing and unique, and `now_serving` climbs by one each release, so threads enter in precisely the order they arrived. No starvation is possible — every waiter's turn comes after a bounded number of releases. The figure above traces a single thread: it takes ticket 5, watches `now_serving` climb 3 → 4 → 5, and enters exactly when its number is called, then bumps the counter to 6 on the way out. This is the lock the Linux kernel used for years (replacing the old unfair test-and-set spinlock in 2008) precisely because the unfair version was causing unbounded latency on contended locks.

But the ticket lock has a scaling weakness that motivates the next design. *Every* waiter spins on the *same* `now_serving` variable. When it changes (a release), it invalidates that one cache line on *all* waiting cores — every waiter's cached copy goes stale, and they all reload it, even though only one of them (the next ticket) actually gets to proceed. So a release triggers $O(N)$ cache-line invalidations for $N$ waiters. Better than naive TAS (the spinning itself is read-only and quiet), but the *handoff* still costs $O(N)$ coherence traffic per release. On a 64-core box under heavy contention, that's the bottleneck.

### MCS locks: a queue where each thread spins on its own line

The *MCS lock* (named for its inventors Mellor-Crummey and Scott, 1991) fixes the ticket lock's handoff cost with a beautiful idea: instead of all waiters spinning on one shared variable, give each waiter its **own** local flag to spin on, and link the waiters into an explicit FIFO queue. A release signals *only the next* waiter's private flag — invalidating *one* cache line, not $N$. The result is the holy grail for a spinlock: FIFO fairness *and* $O(1)$ coherence traffic per handoff, so it scales to hundreds of cores.

Each thread brings a small queue-node (typically on its own stack) holding a `locked` flag and a `next` pointer. The lock itself is just a `tail` pointer to the last node in the queue:

```c
typedef struct mcs_node {
    atomic_bool      locked;  // spin on MY OWN flag (local cache line)
    _Atomic(struct mcs_node *) next; // link to the waiter behind me
} mcs_node_t;

typedef struct { _Atomic(mcs_node_t *) tail; } mcs_lock_t;

void mcs_lock(mcs_lock_t *l, mcs_node_t *my) {
    atomic_store_explicit(&my->next, NULL, memory_order_relaxed);
    // atomically append myself to the queue, get the previous tail
    mcs_node_t *pred = atomic_exchange_explicit(&l->tail, my,
                                                memory_order_acq_rel);
    if (pred == NULL) return;             // queue was empty -> lock is ours
    // there's a predecessor: I will spin on MY OWN locked flag
    atomic_store_explicit(&my->locked, true, memory_order_relaxed);
    atomic_store_explicit(&pred->next, my, memory_order_release); // link in
    while (atomic_load_explicit(&my->locked, memory_order_acquire))
        __builtin_ia32_pause();            // spin on local line — quiet!
}

void mcs_unlock(mcs_lock_t *l, mcs_node_t *my) {
    mcs_node_t *succ = atomic_load_explicit(&my->next, memory_order_acquire);
    if (succ == NULL) {
        // no known successor; try to clear tail if I'm still last
        mcs_node_t *expected = my;
        if (atomic_compare_exchange_strong_explicit(
                &l->tail, &expected, NULL,
                memory_order_release, memory_order_relaxed))
            return; // queue now empty, done
        // someone is mid-enqueue; wait for them to link, then signal
        while ((succ = atomic_load_explicit(&my->next, memory_order_acquire)) == NULL)
            __builtin_ia32_pause();
    }
    // wake exactly ONE successor by clearing its private flag
    atomic_store_explicit(&succ->locked, false, memory_order_release);
}
```

Trace it. To acquire, a thread atomically swaps itself into `tail` (one fetch-and-store, getting its predecessor). If there was no predecessor, the queue was empty and the lock is instantly its. Otherwise it sets its *own* `locked` flag and links itself behind its predecessor, then spins on that private flag — which lives in *its own* cache line, so the spin is purely local and generates no shared traffic. To release, the holder clears the `locked` flag of *its specific successor* (the next node in the queue), invalidating exactly one cache line and waking exactly one thread, in FIFO order. The handoff is $O(1)$ regardless of how many threads are queued. The cost is the extra per-thread node and the more intricate code — and a subtle edge case (a release racing an in-progress enqueue, handled by the CAS-and-wait in `mcs_unlock`). This is the structure behind Linux's modern `qspinlock` (queued spinlock), which uses an MCS-style queue and replaced the ticket lock in 2014/2015 precisely because the ticket lock's $O(N)$ handoff didn't scale to large NUMA machines.

We'll compare all four lock designs head-to-head in the measurements section. The progression — TAS → TTAS → ticket → MCS — is a clean staircase: each step fixes the previous one's worst flaw (no exclusion → cache thrash → unfairness → $O(N)$ handoff), trading a little more code for a lot more scalability.

## Blocking locks and the futex: a userspace fast path with a kernel slow path

So far everything spins, which is only ever the right answer for *short* holds on spare cores. For the general-purpose mutex — the `pthread_mutex_t`, the `std::sync::Mutex`, the Java `synchronized` — you need to *block*: when the lock is contended, park the waiting thread so its core can do other work, and wake it when the lock frees. Blocking requires the kernel, because only the kernel can deschedule a thread and put it back on the run queue. But entering the kernel costs a system call (~100 ns to a microsecond), and you absolutely do *not* want to pay that on the common case where the lock is *uncontended*. The resolution is the **futex** — "fast userspace mutex" — the Linux primitive (with equivalents on every modern OS) that nearly every blocking lock is built on.

![A stack showing the futex lock path where an uncontended lock takes a userspace compare and swap with no syscall while a contended lock parks with a futex wait syscall and is woken by futex wake](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-7.png)

The futex idea is a clean split of responsibility. The lock state — that same one-word flag — lives in ordinary userspace memory. The kernel provides just two operations on an address:

- `futex_wait(addr, expected)` — atomically: *if* the value at `addr` still equals `expected`, put the calling thread to sleep on a wait queue keyed by that address; otherwise return immediately. The "if still equals" check, done by the kernel atomically against wakers, closes the race where the lock frees between your userspace check and your decision to sleep.
- `futex_wake(addr, n)` — wake up to `n` threads sleeping on `addr`.

With those two kernel calls, a mutex needs the kernel **only when threads actually contend**:

```c
// A minimal futex mutex (Linux). State: 0=free, 1=held, 2=held+waiters.
// (Drepper's three-state design; the classic reference implementation.)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdatomic.h>

static int futex(_Atomic int *uaddr, int op, int val) {
    return syscall(SYS_futex, uaddr, op, val, NULL, NULL, 0);
}

void mutex_lock(_Atomic int *m) {
    int c;
    // FAST PATH: try to grab a free lock with one CAS, no syscall.
    if ((c = 0, atomic_compare_exchange_strong(m, &c, 1)))
        return;                       // was 0 (free) -> now 1 (held), done
    // SLOW PATH: contended. Mark "held + waiters" (state 2) and sleep.
    if (c != 2)
        c = atomic_exchange(m, 2);    // announce we are waiting
    while (c != 0) {
        futex(m, FUTEX_WAIT, 2);      // sleep IF *m still == 2 (in kernel)
        c = atomic_exchange(m, 2);    // re-acquire attempt on wake
    }
}

void mutex_unlock(_Atomic int *m) {
    // FAST PATH: if there were no waiters (state was 1), just clear it.
    if (atomic_fetch_sub(m, 1) != 1) {   // was 2 (waiters present)
        atomic_store(m, 0);
        futex(m, FUTEX_WAKE, 1);         // wake one parked waiter
    }
    // else: was 1 (no waiters), fetch_sub left it 0, no syscall needed
}
```

Read the fast paths. `mutex_lock` on an uncontended lock is a **single CAS** that flips `0` to `1` and returns — no syscall, no kernel, a handful of nanoseconds. `mutex_unlock` on a lock with no waiters is a single decrement and return — again no syscall. The kernel (`futex_wait`/`futex_wake`) is touched *only* when the CAS fails because another thread holds the lock (contention) or when releasing a lock that has parked waiters. This is the whole reason a `std::mutex` lock/unlock pair costs ~25 ns uncontended but jumps to microseconds under contention: you're measuring the userspace CAS versus the kernel round-trip. The figure above shows the two paths — the green uncontended CAS that never leaves userspace, and the contended path that parks via `futex_wait` and is later woken by `futex_wake`.

The three-state design (`0` free, `1` held-no-waiters, `2` held-with-waiters) is the trick that keeps the unlock fast path syscall-free: a releaser only calls `futex_wake` if the state says waiters exist. Get this wrong and you either wake no one (lost wakeup, a hang) or wake on every unlock (a syscall storm). Real production mutexes layer more on top — a brief *adaptive spin* before the `futex_wait` (to win the short-hold case without a syscall, exactly the crossover from earlier), priority inheritance to dodge priority inversion, and robust-futex cleanup if a holder dies — but the skeleton above is genuinely how glibc's `pthread_mutex` and Rust's `parking_lot`/`std` mutex work at the core. In **Rust**, `std::sync::Mutex` on Linux is a thin wrapper over a futex with this exact fast-path CAS; in **Go**, `sync.Mutex` implements its own spin-then-park state machine over the runtime's goroutine parking (`runtime_SemacquireMutex`) rather than the OS futex, because Go parks goroutines, not OS threads — same idea, different scheduler.

The subtle race the kernel closes for you is the *lost wakeup*, and it's worth seeing why the futex API is shaped the way it is. Suppose you do this naively in pure userspace: a waiter checks the flag, sees it's held, and decides to sleep. But between "sees it's held" and "actually goes to sleep," the holder releases the lock and tries to wake waiters — except the waiter isn't asleep *yet*, so the wake finds no one and does nothing, and then the waiter goes to sleep forever. That's a lost wakeup, and it's a classic hang. The futex fixes it by making the sleep *conditional on the value*: `futex_wait(addr, expected)` checks, atomically with respect to wakers and *inside the kernel*, that `*addr` still equals `expected` before parking — if a releaser changed the value in the gap, the wait returns immediately instead of sleeping. That single atomic "compare-then-sleep" is the irreducible kernel primitive you can't build correctly in userspace alone, which is the entire reason the futex syscall exists. Everything else about a mutex — the spinning, the state machine, the wakeup-one-vs-all policy — is userspace policy layered over that one kernel guarantee.

A related production refinement, *wait morphing*: when you signal a condition variable whose waiters will immediately try to re-lock a mutex, a smart implementation moves the woken thread directly from the condvar's wait queue onto the mutex's wait queue rather than waking it (so it races for the lock, loses, and parks again) — avoiding a thundering-herd of wake-then-immediately-block syscalls. It's a small detail, but it's the kind of thing that separates a textbook futex mutex from glibc's, and it underscores the theme: the hard part of a blocking lock isn't the fast path (one CAS), it's making the *contended* path not melt down under load.

## Backoff: smoothing the thundering herd

There's one more tool that sits between "spin tightly" and "block," and it shows up in spinlocks, CAS loops, and adaptive mutexes alike: **backoff**. When many threads contend, having all of them retry as fast as possible is maximally destructive — it's the cache-thrash storm again, now at the algorithm level. *Exponential backoff* says: each time you fail to acquire (or your CAS loses the race), wait a little longer before retrying, doubling the wait up to a cap, ideally with randomized jitter so threads don't re-synchronize into lockstep.

```c
void spin_lock_backoff(spinlock_t *l) {
    unsigned backoff = 1;              // start at 1 pause
    while (atomic_exchange_explicit(&l->held_int, 1, memory_order_acquire) != 0) {
        // failed to acquire — back off before retrying
        for (unsigned i = 0; i < backoff; i++)
            __builtin_ia32_pause();    // burn a bounded, growing delay
        if (backoff < 1024) backoff <<= 1; // double, capped
        // (best: TTAS read-spin INSIDE here, plus randomized jitter)
    }
}
```

The reasoning mirrors Ethernet's collision backoff and the retry strategy of any distributed client hammering a contended server: spreading retries out in time reduces the collision rate, so the *aggregate* throughput goes *up* even though each individual thread waits longer per attempt. The cost is latency — a thread that backed off to a 1024-cycle delay might miss a release window and wait longer than necessary — so backoff is a throughput-for-latency trade. The cap matters (unbounded backoff means a thread that's been unlucky waits forever), and the jitter matters (without randomization, threads that collided once tend to collide again on the same schedule). In CAS loops for lock-free structures the same applies: a `compare_exchange` loop under heavy contention should back off between failed attempts, not retry at full speed. Production locks combine all of it — TTAS read-spinning, bounded exponential backoff with jitter, then a futex park — each layer handling the regime where it wins.

#### Worked example: a CAS loop that pushes to a lock-free stack

The CAS loop is the pattern you'll write far more often than a from-scratch lock, so let's see it racing. To push a node onto a lock-free stack, you read the current head, point your new node at it, and CAS the head from the old value to your node — retrying if someone else changed the head in between.

```c
void push(_Atomic(node_t *) *head, node_t *n) {
    node_t *old = atomic_load_explicit(head, memory_order_relaxed);
    do {
        n->next = old;     // point new node at the current head
        // try to swing head from `old` to `n`. compare_exchange UPDATES
        // `old` with the actual current value on failure -> built-in retry.
    } while (!atomic_compare_exchange_weak_explicit(
                 head, &old, n,
                 memory_order_release, memory_order_relaxed));
}
```

Now the race, the same shape as the spinlock acquire. T1 and T2 both read `head = A`. Both set their node's `next = A`. T1's CAS succeeds: `head` is now T1's node, with `next = A`. T2's CAS *fails*, because `head` is no longer `A` — but the `weak` variant has already reloaded `old` with the new head (T1's node), so T2's loop body runs again with the correct head and succeeds on the second try. No lock, no blocking, and *progress is guaranteed*: every failed CAS means *some other* thread succeeded, so the system as a whole always advances. That's the formal property called *lock-freedom*, and it's why CAS is the foundation of non-blocking data structures. The figure below traces exactly this two-thread CAS race step by step.

![A timeline of two threads racing a single compare and swap where both read a free flag T1 swaps successfully T2 sees the new value its swap fails and it re-reads then retries](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-3.png)

There's a famous hazard lurking here — the **ABA problem**: if the head changes from A to B and back to A between T2's read and its CAS, T2's CAS succeeds even though the world changed underneath it (a node may have been freed and reused). Solving ABA needs version tags or hazard pointers, which is a whole topic for the lock-free post in this series. For now, the takeaway is just that the CAS loop is the universal non-blocking primitive, with the same race-and-retry shape as a lock acquire — and that the timeline above (read, read, win, fail, re-read, retry) is the canonical picture you should carry for *any* CAS.

## Measured behavior: spinlock versus mutex throughput and fairness

Theory says spinlocks win for short holds on spare cores and lose badly otherwise; mutexes are the safe general choice; fairness costs throughput. Let's put numbers on it — *honestly*, which in concurrency benchmarking means: **pin threads to cores** so the OS scheduler doesn't randomize placement; **warm up** before timing so the JIT/caches/branch-predictor are hot; **run many iterations** and report a distribution, not one number; vary the **critical-section length** because it changes the answer entirely; and **name the platform**, because x86's strong memory model and ARM's weak one give different constants. The numbers below are representative orders of magnitude from this class of microbenchmark on a typical multi-core x86 server — treat them as the *shape* of the result you should reproduce on your own hardware, not as gospel constants.

First, **uncontended cost** — one thread, no competition, measuring the raw lock/unlock overhead:

| Lock type | Uncontended lock+unlock | Why |
| --- | --- | --- |
| TTAS spinlock | ~15–25 ns | one atomic exchange + one store |
| Ticket lock | ~20–30 ns | fetch-and-add + a load + a store |
| Futex mutex (e.g. `std::mutex`) | ~20–30 ns | one CAS, no syscall on the fast path |
| MCS lock | ~25–40 ns | exchange + node setup, more bookkeeping |

The headline: **uncontended, they're all roughly the same** — tens of nanoseconds, because every one is dominated by a single atomic instruction and its memory-ordering fences, and *none* of them touches the kernel on the happy path. This is the single most important measurement for everyday code: an uncontended mutex is *not* slow. The "locks are expensive, go lock-free" instinct is usually wrong, because the lock you're worried about is uncontended, and uncontended it costs about what a single CAS costs — which is what your lock-free alternative costs too.

The interesting story is **under contention**, where threads actually collide. Here the critical-section length and the lock design dominate. With many threads hammering a *short* critical section (a counter bump, ~tens of ns held):

| Lock type | Throughput under heavy contention | Fairness (tail latency) |
| --- | --- | --- |
| Naive TAS spinlock | collapses — can go *below* single-thread as cores rise | terrible; unbounded starvation |
| TTAS spinlock | much better than TAS; flat-to-degrading past a few cores | poor; coherence lottery picks winners |
| Ticket lock | good; FIFO; degrades on big NUMA from $O(N)$ handoff | excellent; strict FIFO |
| MCS / qspinlock | best spin design at high core counts; $O(1)$ handoff | excellent; strict FIFO |
| Futex mutex | spins briefly then parks; frees cores; lower raw throughput, no waste | good; kernel wake order, roughly fair |

#### Worked example: the contention curve and where it bends down

The measurement that teaches the most is a *throughput-versus-threads* curve. Run $N$ threads, each repeatedly acquiring a lock, incrementing a shared counter, releasing — and plot total increments per second against $N$, for $N$ from 1 up past your physical core count. The shape is the lesson:

- **Naive TAS spinlock**: throughput *peaks around 1–2 threads and then falls*, often dropping *below* the single-thread number by the time you reach 8 cores. The cache-line ping-pong (Section 4) means adding cores adds coherence traffic faster than it adds useful work. This is the most counterintuitive and important result in the whole post: a "correct" lock can make parallelism *negative*. Plot it once and you'll never write a naive TAS spinlock again.
- **TTAS spinlock**: throughput rises then plateaus and slowly degrades — the read-spinning kills the steady-state thrash, but the thundering herd at each release still costs, so it doesn't *scale*, it just doesn't *collapse*.
- **Ticket / MCS**: throughput holds up far better as cores rise; MCS's $O(1)$ handoff keeps it nearly flat where the ticket lock's $O(N)$ handoff starts to sag on large machines.
- **Futex mutex with adaptive spin**: similar plateau to TTAS at low contention (the spin wins the short holds), and — critically — when you *oversubscribe* (more threads than cores), the mutex *holds throughput* while the pure spinlocks *fall off a cliff*, because the mutex parks waiters and frees cores for the holder while the spinlocks burn cores spinning against a descheduled holder (Section 5's lock-holder preemption).

Now lengthen the critical section to ~10 µs of held work and rerun: the spinlocks become a disaster (every waiter burns a full core for 10 µs per acquire, so total useful work craters as you add threads), while the mutex barely notices (waiters park and the cores do other work). Same code, opposite conclusion, decided entirely by the hold time relative to $2C$ — exactly the crossover we derived. **The benchmark only tells the truth if you sweep the critical-section length**; a single hold time gives you a single, misleading data point.

A word on honesty: every number here jiggles run to run because the OS scheduler, frequency scaling, and other processes are confounds you can reduce but not eliminate. Pin cores (`taskset`/`pthread_setaffinity_np`), disable turbo for stable numbers, run for seconds not milliseconds, take the median of many runs, and report the spread. And remember the platform: on ARM's weaker memory model the acquire/release fences are real instructions (`dmb`/`ldar`/`stlr`) with non-trivial cost, where on x86 the same orderings are often free because the hardware is already strongly ordered — so the *uncontended* constants shift between architectures even for identical source.

## Case studies / real-world

These designs aren't academic exercises; the exact staircase from TAS to MCS played out in production systems, and the pitfalls bit real people.

**The Linux kernel: ticket locks (2008) then qspinlock (2014).** For years the Linux kernel's spinlock was a simple unfair test-and-set, and on large SMP machines it caused pathological, unbounded acquisition latency under contention — a thread could be starved while others streamed past. In 2008 (kernel 2.6.25) Nick Piggin replaced it with a **ticket spinlock** to guarantee FIFO fairness, fixing the starvation at the cost of the $O(N)$-per-release coherence traffic described above. That handoff cost then became the bottleneck on big NUMA boxes, so in 2014–2015 (kernel 3.15+) the kernel moved to the **queued spinlock** (`qspinlock`), an MCS-derived design where each waiter spins on its own cache line and the handoff is $O(1)$. The progression in the mainline kernel is *exactly* the TAS → ticket → MCS staircase of this post, driven each time by a measured scalability wall on real hardware. (Sources: the LWN.net articles "Ticket spinlocks" (2008) and "MCS locks and qspinlocks" (2014) document both transitions.)

**The MCS lock paper (1991).** John Mellor-Crummey and Michael Scott's "Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors" (ACM TOCS, 1991) introduced the queue-based lock where each thread spins on a local variable, proving you could have FIFO fairness *and* constant coherence traffic per handoff. It's one of the most-cited systems papers ever, and three decades later its core idea is in the Linux kernel, in Java's lock implementations, and in `parking_lot`. The lesson it crystallized — *spin on memory that's local to you, not shared with every other waiter* — is the single most important principle for scalable locking.

**Spinlocks on a hypervisor: lock-holder preemption.** When spinlocks meet virtualization, the spin-versus-block assumption ("the holder is running on some core") breaks, and the result is the lock-holder preemption / lock-waiter preemption problem: a guest's virtual CPU spins on a lock whose holder is *another* vCPU that the hypervisor has descheduled, so the spinner burns its entire timeslice (tens of ms) against a holder that isn't even running. VMware, Xen, and KVM all documented severe throughput collapse from this, and the fix — *paravirtualized spinlocks* that detect a long spin and issue a hypercall to yield the vCPU so the hypervisor can schedule the holder — shipped in all major hypervisors and guest kernels. It's the most expensive real-world demonstration of "never spin when the holder might not be running" (Section 5). (Sources: the Xen/KVM "paravirtualized spinlock" / "pv-qspinlock" kernel documentation and the VMware "co-scheduling" guidance.)

**Java's `synchronized`: biased and adaptive locking.** The HotSpot JVM spent years making `synchronized` cheap for the overwhelmingly common uncontended case. *Biased locking* (default for ~15 years) let an uncontended lock skip the atomic CAS entirely by "biasing" the lock to the first thread that took it — a measured win when locks are reentered by one thread, which is most of them. Under contention it falls back to a *thin lock* (a spin-then-inflate CAS scheme) and finally a *fat lock* (a full OS monitor that blocks). It's the adaptive spin-then-block strategy of Section 5 baked into a language runtime. (Biased locking was eventually deprecated and removed in JDK 15+ because the bookkeeping cost outweighed the benefit on modern hardware — a reminder that even the "right" optimization has a shelf life as hardware changes.)

**The LMAX Disruptor: when the lock itself was the bottleneck.** The LMAX exchange needed to process millions of trades per second on a single thread's worth of throughput, and profiling showed their queue's *locks* — not the business logic — were the wall: even an uncontended `ReentrantLock` lock/unlock pair, multiplied across millions of operations per second, plus the occasional contention stall, dominated. Their published 2011 analysis measured a contended lock costing on the order of a thousand-plus nanoseconds versus a lock-free CAS on the order of tens, and an uncontended path where the CAS still won by avoiding the lock's bookkeeping. Their fix — the Disruptor ring buffer — replaced the lock-based queue with a pre-allocated ring and a single CAS (or, for the single-producer case, a plain sequence counter with memory barriers) to claim a slot, plus deliberate cache-line padding to kill false sharing on the sequence counters. It's a clean real-world instance of the whole post: they measured the lock was the bottleneck (not assumed it), understood the cost as "an atomic operation plus contention," and went lock-free *only because the measurement justified it*. (Source: Thompson, Farley, Barker, Gee, Stewart, "Disruptor: High performance alternative to bounded queues for exchanging data between concurrent threads," LMAX technical paper, 2011.)

## When to reach for this (and when not to)

The honest summary: **you should understand how a lock is built, and then in 95% of your code you should use the standard library mutex and not build one.** Here is the decision, made plainly.

![A matrix comparing TAS spinlock ticket lock MCS lock and futex mutex across fairness scaling under contention and cost showing the trade between simplicity and scalable fairness](/imgs/blogs/how-a-lock-is-built-test-and-set-cas-and-spinlocks-8.png)

**Reach for the standard blocking mutex (the default).** `std::sync::Mutex`, `sync.Mutex`, `synchronized`/`ReentrantLock`, `std::mutex`, `pthread_mutex_t` — these are adaptive (brief spin, then park), correct on uniprocessors and under oversubscription, syscall-free when uncontended, and battle-tested. Unless you've *measured* a lock to be your bottleneck, this is the answer. The futex fast path means it costs about a CAS when it's not contended, which is almost always.

**Reach for a spinlock only when you've proven all of: (1) the critical section is genuinely short** — tens to low hundreds of nanoseconds, well under $2C$; **(2) you control scheduling** so the holder is never preempted while you spin — kernel/driver code with preemption disabled, a real-time thread pinned to a dedicated core, or a tight HPC loop on isolated cores; **(3) you are not oversubscribed and not on a hypervisor** that can deschedule the holder; and **(4) you actually measured** a mutex being too slow here. Outside the kernel and a few latency-critical user-space niches, those conditions rarely all hold. The matrix above is the quick reference: a TAS spinlock is tiny and unfair and doesn't scale; a ticket lock buys fairness cheaply but sags on big machines; MCS buys fairness *and* scaling at the cost of per-thread node state; a futex mutex blocks and is the right general default.

**Do not roll your own naive TAS spinlock.** If you've decided you truly need a spinlock, at minimum make it test-and-test-and-set with the CPU spin hint (`PAUSE`/`YIELD`) and bounded backoff — a naive write-every-spin TAS lock is a cache-coherence weapon that can make your multi-core machine slower than one core (Section 4). Better, use a vetted implementation (`parking_lot`'s primitives, the kernel's `qspinlock`, a known-good MCS) rather than hand-rolling the subtle release/enqueue races.

**Do not reach for lock-free just because locks "feel slow."** The CAS loop at the heart of a lock-free structure costs about the same as an uncontended mutex (it's the same atomic instruction), and lock-free code is dramatically harder to get right (ABA, memory reclamation, the memory orderings). Go lock-free when you have a *measured* contention bottleneck that a better lock can't fix, not on instinct. Measure first, every time.

**Never spin on a uniprocessor or against a possibly-preempted holder.** This is the one rule with no exceptions (Section 5): if there's a real chance the lock holder isn't currently running on some core, spinning is a way to burn a timeslice accomplishing nothing while *preventing* the holder from running. Block. This is why the right default lock blocks: it's correct in the cases where a spinlock is catastrophic, and only modestly slower in the cases where a spinlock would win.

## Key takeaways

- **A lock is one atomic flag plus one read-modify-write instruction.** Acquire is "atomically flip free → held"; the magic is entirely in the hardware's promise that the read-decide-write is indivisible. Once you see the `lock cmpxchg`, the mutex stops being magic.
- **CAS is the universal primitive; test-and-set and fetch-and-add are its specialized cousins.** CAS conditionally swaps any value (maps to `lock cmpxchg` / LL-SC and is the basis of lock-free code); test-and-set flips a flag (`xchg`); fetch-and-add gives a fair ticket (`lock xadd`). A CAS is something you do *in a loop*, expecting occasional failure.
- **A naive test-and-set spinlock thrashes the cache line and can make more cores mean less throughput.** Every spin is a *write* demanding exclusive ownership, so the lock's line ping-pongs between cores. Spin on a *read* (test-and-test-and-set); only the holder's release and the winner's acquire need a write.
- **Spin only if the expected hold is shorter than a context switch.** The crossover is $H < 2C \approx 2\text{–}10\,\mu s$. Short critical sections favor spinning; long ones (I/O, allocation, syscalls) favor blocking. When unsure, spin briefly then block — the adaptive mutex.
- **Never spin on a uniprocessor or under oversubscription/virtualization.** If the holder might not be running, spinning prevents it from running and burns a timeslice for nothing. This is the spinlock's one unforgivable failure mode.
- **Basic spinlocks starve; ticket locks and MCS locks restore fairness.** A fetch-and-add ticket lock grants FIFO order; an MCS lock adds per-thread local spinning for $O(1)$ handoff that scales to hundreds of cores. The Linux kernel walked exactly this TAS → ticket → qspinlock staircase.
- **A production mutex is a futex: a userspace CAS fast path, a kernel syscall only on contention.** Uncontended lock/unlock never touches the kernel and costs about one CAS (~25 ns); `futex_wait`/`futex_wake` park and wake threads only when they actually collide. An uncontended mutex is *not* slow.
- **Measure before you optimize, and sweep the critical-section length.** Pin cores, warm up, run many times, name the platform. Uncontended, all locks cost about a CAS; under contention the hold time and the lock design decide everything — and a single hold time gives a single misleading number.

## Further reading

- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming* (2nd ed., 2020)** — chapters 7 and 8 build TAS, TTAS, ticket, and MCS/CLH locks from primitives with exactly this progression, plus the consensus-number theory that explains why CAS is universal. The definitive treatment.
- **Mellor-Crummey and Scott, "Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors" (ACM TOCS, 1991)** — the original MCS lock paper; the source of "spin on a local variable, queue the waiters."
- **Ulrich Drepper, "Futexes Are Tricky" (2011)** — the canonical walkthrough of building a correct mutex from a futex, including the three-state design and the subtle lost-wakeup races. Essential before you ever touch `SYS_futex`.
- **LWN.net: "Ticket spinlocks" (2008) and "MCS locks and qspinlocks" (2014)** — how and why the Linux kernel moved from unfair TAS to ticket locks to queued spinlocks; production scalability engineering documented in real time.
- **Paul McKenney, *Is Parallel Programming Hard, And, If So, What Can You Do About It?*** — the deferred-processing and locking chapters cover spinlock variants, the cache-coherence cost model, and the measurement discipline, from a kernel practitioner.
- **Jeff Preshing, "Locks Aren't Slow; Lock Contention Is" and the acquire/release posts** — clear, measured intuition for why an uncontended lock is cheap and where the cost actually comes from.
- Within this series: start from [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), see the lock used as a tool in [mutual exclusion, mutexes, and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections), go deeper on the orderings in [atomics and memory orderings from relaxed to seq-cst](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst), connect the cache-coherence cost to the hardware in [the memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), and fit it all into the decision framework in [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

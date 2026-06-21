---
title: "The Progress Hierarchy: Blocking, Lock-Free, and Wait-Free"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Define the non-blocking guarantees precisely — blocking, obstruction-free, lock-free, wait-free — see why lock-free is about progress and not speed, and learn linearizability as the correctness condition that makes any of it mean something."
tags:
  [
    "concurrency",
    "parallelism",
    "lock-free",
    "wait-free",
    "linearizability",
    "progress-guarantees",
    "non-blocking",
    "atomics",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-1.png"
---

A bank-account service is humming along at three in the morning. Two hundred threads share one in-memory counter — total balance across a shard — guarded by a mutex. The mutex is fast: an uncontended acquire is a single atomic instruction, maybe twenty-five nanoseconds, and threads hold it for a few hundred nanoseconds at most. Throughput is fine. Then one thread, the one that happens to be holding the lock, gets descheduled. Maybe the operating system preempted it at the end of its time slice. Maybe its stack page was swapped out and touching it faulted to disk. Maybe a higher-priority thread woke up and the scheduler ran that instead. The reason does not matter. What matters is that for the next ten milliseconds — an eternity at this timescale, four hundred thousand times longer than the critical section it was running — that thread is not on a CPU. And because it holds the lock, *none of the other 199 threads can make progress either.* They are all blocked, spinning or sleeping, waiting on a flag that only the absent thread can clear. One stalled thread froze the entire system.

That is the defining failure of *blocking* synchronization, and it is the failure that the whole family of *non-blocking* algorithms exists to avoid. The promise of a lock-free data structure is precise and it is narrow: no matter which thread stalls, no matter where, *some* thread can always still make progress. The system as a whole never freezes because one participant went to sleep at the wrong instant. That promise has a name — lock-freedom — and it sits on a ladder of progress guarantees, each one stronger and more expensive than the last, from *blocking* at the bottom through *obstruction-free* and *lock-free* up to *wait-free* at the top, where every single thread is guaranteed to finish in a bounded number of its own steps.

This post is the gate into the lock-free track of the series. Before we build a single lock-free stack (that is the [next post](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures)) or wrestle with the genuinely hard problem of freeing memory underneath one (the [post after that](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu)), we need to define the words exactly. Half the confusion in this corner of systems engineering comes from people using "lock-free" to mean "fast" or "without a mutex" when it means neither. Lock-freedom is a *progress guarantee*, not a performance claim — and we will measure a case where a lock-free counter is genuinely *slower* than a plain mutex. We will also define *linearizability*, the correctness condition that makes a concurrent object trustworthy at all: the property that every operation appears to take effect atomically at some single instant between the moment you called it and the moment it returned. Progress without correctness is worthless; a structure that "always makes progress" toward a corrupted state is not an achievement. By the end you will be able to read a paper or a library doc, see the words "lock-free" or "wait-free" or "linearizable," and know exactly what is being promised, what it costs, and whether you actually need it.

![A tree splitting the progress guarantee into a blocking branch where a stalled lock holder freezes the system and a non-blocking branch holding obstruction-free, lock-free, and wait-free levels](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-1.png)

This is the same discipline the [series intro](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) set out, viewed from one specific angle. We have shared mutable state. We need to order every access to it. A lock buys that order by *excluding* other threads — which is exactly why a stalled lock holder is catastrophic. The non-blocking world buys the order a different way: with atomic read-modify-write instructions that let many threads contend on the same word *without* any of them being able to permanently shut the others out. The progress hierarchy is the precise vocabulary for how strong that "without shutting the others out" guarantee is.

## What "progress" means, and why blocking has none

Start with the thing every junior engineer assumes is free: progress. You call `queue.push(x)`. You expect it to return. In a single-threaded program it always does — there is no one else to interfere. In a concurrent program, whether your call returns depends on *what every other thread does*, and that dependency is the whole subject.

A *progress guarantee* is a statement of the form: "under these scheduling assumptions, this operation (or the system) is guaranteed to complete." The scheduling assumptions are the crux. Real schedulers are adversarial in the sense that matters for correctness reasoning: a thread can be preempted at *any* instruction boundary, for an *unbounded* amount of time, and you are not allowed to assume otherwise. You cannot assume your thread runs to completion once it starts. You cannot assume a thread that went to sleep will wake "soon." The only thing you may assume is that a thread which is *scheduled* and *running* will eventually execute its next instruction. Everything else is the scheduler's prerogative.

It is worth dwelling on *why* we reason against an adversarial scheduler, because it feels paranoid until you have been burned. The point is not that the operating-system scheduler is malicious — it is not — but that you cannot *control* it and you cannot *test* every interleaving it might produce. There are astronomically many ways $n$ threads' instructions can interleave; a bug that only appears for one particular preemption point will pass every test on your laptop and then fire once a week in production on a busier machine with a different core count. So you reason as if an adversary picks the worst possible moment to preempt every thread. If your algorithm is correct against *that* adversary, it is correct against *any* real scheduler, because the real one is a special case of the adversary. This is exactly the same move that makes [memory-model](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) reasoning tractable — reason about what is *guaranteed*, never about what *usually happens* — and the progress hierarchy is just that discipline applied to the question "will this finish?"

The adversary model also clarifies what each guarantee is really claiming. Blocking says "I assume holders are not preempted at bad times" — an assumption the adversary immediately violates, which is why blocking has no guarantee under adversarial scheduling at all. Obstruction-free says "I assume that *eventually* a thread gets to run alone for a while" — a weaker assumption the adversary can still defeat by never letting anyone run alone, which is why livelock is legal. Lock-free makes *no* scheduling assumption for the system guarantee: some thread advances no matter how the adversary schedules, period. Wait-free makes no scheduling assumption for the *per-thread* guarantee: every thread advances no matter what. The higher you climb, the fewer favors you ask of the scheduler — and asking the scheduler for no favors at all is exactly what makes wait-free both the strongest guarantee and the hardest to build.

Under those rules, define the four levels of the hierarchy precisely. We will spend the rest of the post unpacking each, but here are the definitions stated once, cleanly, because everything else hangs off them.

**Blocking.** An algorithm is *blocking* if the delay or suspension of one thread can prevent other threads from making progress indefinitely. A mutex is the canonical example: a thread that holds the lock and is then descheduled blocks every thread that wants the lock, for as long as it is off the CPU. The system's progress is held hostage to one participant's scheduling.

**Obstruction-free.** An algorithm is *obstruction-free* if any thread that runs *in isolation* — with all other threads suspended — completes its operation in a bounded number of steps. This is the weakest non-blocking guarantee. It says nothing about what happens under contention; two obstruction-free threads can repeatedly interfere and *livelock*, each undoing the other's work forever. But no thread can ever be *blocked* by a stalled peer, because if the peer is suspended, the running thread is by definition running alone and will finish.

**Lock-free.** An algorithm is *lock-free* if, whenever any thread takes a step, *some* thread (not necessarily the same one) is guaranteed to make progress on its operation within a bounded number of total system steps. The guarantee is *system-wide*: the system as a whole always advances. An individual thread, however, can be unlucky forever — it can fail its retry, get preempted, fail again, and *starve* while other threads sail past. Lock-free rules out the system freezing; it does not rule out one thread getting nothing done.

**Wait-free.** An algorithm is *wait-free* if *every* thread completes its operation in a bounded number of *its own* steps, regardless of what any other thread does. This is the strongest guarantee. No starvation is possible — there is a hard ceiling on how many steps any single operation takes, independent of contention. Wait-freedom is what hard real-time systems need, because it is the only level that bounds the latency of an *individual* operation.

The figure above is the map. The first, sharpest split is between *blocking* — where one thread's stall can halt others — and *non-blocking* — where it cannot. The three non-blocking levels then differ in *who* is guaranteed to progress: a lone thread (obstruction-free), some thread (lock-free), or every thread (wait-free). Hold that split in your head; it is the single most important distinction in this entire post. Below it sits a matrix that lines up all four levels against the three questions that distinguish them.

![A matrix comparing blocking, obstruction-free, lock-free, and wait-free against who progresses, whether a thread can starve, and the relative cost of each level](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-2.png)

Read the matrix top to bottom and the story is a ladder. At the bottom, blocking: if the holder stalls, *nobody* progresses, and full deadlock is possible — but it is the cheapest mechanism by far, because an uncontended lock is one atomic instruction. Up one rung, obstruction-free: a lone thread always finishes, but two contending threads can livelock. Up again, lock-free: the system always advances, though an individual thread can starve. At the top, wait-free: every thread finishes in bounded steps, nobody starves — at the highest implementation cost. Each rung trades a possible failure mode for more machinery. The art is choosing the lowest rung that your situation actually requires, and most situations require the bottom one.

### A note on terminology you will trip over

The literature is not perfectly consistent, and you will see "non-blocking" used loosely to mean "lock-free." Strictly, *non-blocking* is the umbrella term for all three top levels — obstruction-free, lock-free, and wait-free — because all three share the property that no thread's stall blocks another. "Lock-free" specifically means the middle, system-wide guarantee. When a library says "lock-free queue," it almost always means exactly that: lock-free in the technical sense, not wait-free. Java's `ConcurrentLinkedQueue`, for instance, is lock-free but not wait-free — an individual `offer` can retry indefinitely under adversarial contention. We will come back to it as a case study.

One more pin in the ground, because it trips up almost everyone: "lock-free" does not mean "uses no locks in the source code," and it does not mean "fast." It is entirely possible to write code with the word `lock` nowhere in it that is still *blocking* (spin until a flag clears — that is a lock by another name), and it is possible for a correctly lock-free structure to be *slower* than a mutex. The definition is about the progress guarantee under adversarial scheduling, full stop. Keep that separate from performance in your head and most of the confusion in this field evaporates.

## Why blocking can freeze the whole system

Let us make the blocking failure concrete, because it is the thing the entire non-blocking edifice is built to avoid, and you cannot appreciate the cost of lock-freedom until you feel the pain it solves.

A mutex protects a critical section. While one thread is inside, every other thread that calls `lock()` waits — it either spins on a flag or sleeps in the kernel until the holder calls `unlock()`. The correctness of this is exactly its danger: the *only* thing that lets a waiter proceed is the holder releasing. If the holder cannot release — because it is not running — the waiters wait. There is no timeout in the basic primitive, no escape hatch, no way for a waiter to "help" the holder finish. The holder is a single point of failure for liveness.

Here is the running counter example, blocking version, in three languages so the shape is unmistakable across idioms. First Go:

```go
package main

import "sync"

type Counter struct {
    mu    sync.Mutex
    value int64
}

func (c *Counter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++ // critical section: load, add, store
}
```

The same in Rust, where the lock and the data it protects are bound together by the type system so you cannot touch `value` without holding the lock:

```rust
use std::sync::Mutex;

struct Counter {
    value: Mutex<i64>,
}

impl Counter {
    fn inc(&self) {
        let mut v = self.value.lock().unwrap();
        *v += 1; // critical section, guard drops on scope exit
    }
}
```

And in Java, where `synchronized` and `AtomicLong` both live in the standard library so the contrast is one keyword away:

```java
public final class Counter {
    private long value;

    public synchronized void inc() {
        value++; // critical section guarded by the monitor lock
    }
}
```

All three are correct. All three are blocking. Now stage the failure. Imagine — and the figure below makes this concrete — that thread T1 calls `inc()`, takes the lock, executes the `load` of `value`, and is then preempted by the scheduler for ten milliseconds before it can do the `add` and `store`. During those ten milliseconds T2 and T3 call `inc()`. They block on the lock. They cannot proceed, because T1 holds it and T1 is not running. The counter does not advance. The system makes *zero* progress on this data structure until T1 is rescheduled. That is blocking: the stall of one thread halts the others.

![A before and after figure contrasting a blocking lock where a preempted holder freezes all waiters with a lock-free design where other threads keep advancing](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-3.png)

The "after" side of that figure is the lock-free alternative we are building toward: T1 can stall mid-operation and T2 simply retries its atomic update, succeeds, and the system advances. No thread waits on T1 to wake up, because no thread ever handed control to T1 in the first place. There is no lock to hold and therefore no holder to stall.

#### Worked example: how long can a stall last?

Quantify the hazard so it is not hand-waving. A critical section that does `value++` is, in machine terms, a load, an add, and a store — three instructions, a few nanoseconds. Call it 5 ns of useful work under the lock. Now consider the stalls that can interrupt it:

- A normal scheduler preemption at the end of a time slice: the thread is off-CPU until it is scheduled again. On Linux a typical time slice is on the order of milliseconds; the stalled thread might wait one full slice, call it 1–10 ms.
- A page fault on the thread's stack (the page was swapped out): a disk read, 0.1–10 ms on an SSD, far more on spinning rust.
- Priority inversion: a low-priority thread holds the lock, a medium-priority thread preempts it and runs for a long time, and a high-priority thread that wants the lock is stuck behind both. This is the [Mars Pathfinder bug](/blog/software-development/concurrency/livelock-starvation-and-priority-inversion); the stall was long enough to trip a watchdog and reset the spacecraft.

The ratio is the whole point. The useful work is ~5 ns. The stall can be 10 ms. That is a factor of two million. For that entire window, with a blocking lock, *every other thread that touches this structure is frozen*. The structure's effective throughput during the stall is zero, no matter how many cores you have idle and ready. Adding hardware does not help; the bottleneck is one sleeping thread holding one flag. This is why "just add more cores" is no answer to a blocking-induced stall, and it is the precise pathology lock-freedom eliminates.

## Lock-free: the system always advances, individuals may starve

Now build the intuition for lock-freedom properly, because it is subtle and the subtlety is where the value lives.

A lock-free algorithm replaces "exclude everyone else, then mutate" with "attempt the mutation atomically; if someone beat you to it, observe their change and try again." The atomic primitive that makes this possible is *compare-and-swap* (CAS): an instruction that reads a memory word, compares it to an expected value, and — only if they match — writes a new value, all in one indivisible step. On x86 it is `lock cmpxchg`; on ARM it is a load-linked / store-conditional pair. CAS returns whether it succeeded. The [lock-construction post](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) covers the instruction itself; here we care about the *progress argument* it enables.

The canonical shape is the CAS loop:

```c
// Lock-free increment of a shared 64-bit counter.
#include <stdatomic.h>

void inc(_Atomic long *counter) {
    long old = atomic_load(counter);
    while (!atomic_compare_exchange_weak(counter, &old, old + 1)) {
        // CAS failed: someone else changed *counter since our load.
        // `old` was reloaded with the current value by the CAS.
        // Loop and retry with the fresh value.
    }
}
```

Read the loop as a progress argument, not just as code. A thread loads the current value, computes `old + 1`, and attempts to CAS it in. There are exactly two outcomes. Either the CAS succeeds — the word still held `old`, nobody interfered, and *this thread made progress*. Or the CAS fails — the word no longer holds `old`, which can only mean *some other thread changed it*, which can only mean *that other thread made progress*. There is no third outcome. Every iteration of every thread's loop ends with *somebody* having advanced the counter. The system cannot get stuck, because "stuck" would require an iteration where nobody advanced, and no such iteration exists.

That is the lock-free guarantee, derived rather than asserted: **a failed CAS is always caused by another thread's successful CAS, so the system as a whole always makes progress.** No thread can be blocked by a stalled peer, because there is no point in the algorithm where a thread waits on another. A stalled thread is simply not attempting CAS right now; the running threads keep CASing and keep advancing.

![A timeline showing thread A and thread B both reading the head, A succeeding its compare and swap while B fails because the head changed, then B retrying and succeeding so progress never stops](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-4.png)

The timeline above traces it step by step. A and B both read `head = N`. Both attempt to CAS. A's CAS wins; `head` becomes `X`. B's CAS now fails, because `head` is no longer `N` — *and the reason it is not `N` is precisely that A succeeded.* B reloads, sees `X`, retries, and succeeds, making `head = Y`. At no point did the system stall waiting on a sleeping thread. Even if A had been preempted for ten milliseconds *after* its successful CAS, B would still have failed, reloaded, and made progress on its own. The system advanced regardless of A's schedule.

### The catch: individual starvation is allowed

Here is the asterisk that "lock-free" carries and that you must never forget. The guarantee is about the *system*, not the *individual*. Lock-freedom promises that *some* thread always progresses; it does *not* promise that *your* thread ever progresses. Picture a thread that is just unlucky: it loads, computes, and every time it attempts its CAS, a faster or luckier thread has already changed the word. Its CAS fails, it reloads, it tries again, it fails again. The system is making progress the whole time — every one of those failures was caused by another thread succeeding — but *this particular thread* gets nothing done. It starves.

This is not a hypothetical. Under heavy contention on a single hot word, a thread on a slower core, or a thread that keeps getting preempted at the worst moment, can retry many times before it lands a CAS. The lock-free guarantee is intact — the system never froze — but the *latency* of that one thread's operation is unbounded. That is the precise gap between lock-free and wait-free, and it is why lock-free is not enough for hard real-time, where you need a bound on *every* operation's latency, not just on the system's aggregate progress.

The same lock-free counter, expressed idiomatically, in Rust and Java, so you can see that the CAS loop is a portable idiom and that some platforms hide it inside a primitive:

```rust
use std::sync::atomic::{AtomicI64, Ordering};

fn inc(counter: &AtomicI64) {
    // fetch_add is a single atomic read-modify-write on x86 (lock xadd),
    // so this particular operation is actually WAIT-FREE, not just lock-free:
    // it completes in a bounded number of steps with no retry loop.
    counter.fetch_add(1, Ordering::SeqCst);
}

fn push_like(head: &AtomicI64, new_min: i64) {
    // A genuine CAS loop, which is lock-free but NOT wait-free:
    // it can retry an unbounded number of times under contention.
    let mut cur = head.load(Ordering::Acquire);
    while new_min < cur {
        match head.compare_exchange_weak(cur, new_min, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(actual) => cur = actual, // someone moved head; retry
        }
    }
}
```

```java
import java.util.concurrent.atomic.AtomicLong;

public final class LockFreeCounter {
    private final AtomicLong value = new AtomicLong();

    // getAndIncrement compiles to a single fetch-and-add on x86,
    // so it is wait-free, not merely lock-free.
    public void inc() {
        value.getAndIncrement();
    }

    // A hand-rolled CAS loop, lock-free but not wait-free:
    public void incViaCas() {
        long old;
        do {
            old = value.get();
        } while (!value.compareAndSet(old, old + 1));
    }
}
```

Notice the subtlety the comments call out, because it is one of the most useful things in this whole post. A plain `fetch_add` (`lock xadd` on x86) is a single hardware read-modify-write — it has no retry loop and completes in a bounded number of steps, so an increment built on it is actually *wait-free*, the strongest level. A *general* CAS loop — like the "keep the minimum" `push_like` above, or anything where the new value depends on a multi-step computation over the old — is lock-free but not wait-free, because the loop can spin an unbounded number of times. The progress level of your code depends on whether the hardware gives you the whole operation atomically (`fetch_add`, `fetch_or`, `exchange`) or whether you have to build it from a CAS retry loop. Reach for the single-instruction primitive when the hardware offers it; you get a stronger guarantee for free.

## Wait-free: every thread finishes in bounded steps

Wait-freedom is the top of the ladder and it is qualitatively harder than everything below it. The definition again, because precision matters: an operation is wait-free if there is a finite bound $B$ such that the operation completes in at most $B$ of the calling thread's own steps, *regardless of the actions of other threads*. No starvation, ever. No unbounded retry. A hard ceiling on latency for every single operation.

For operations the hardware implements directly — `fetch_add`, `fetch_or`, `exchange`, `swap` — wait-freedom is free, as we just saw. The instruction either takes a bounded number of cycles or the machine is broken. The hard part is building a *general* wait-free data structure — a queue, a stack, a hash map — where the operation is more than one hardware primitive can do atomically. There, you cannot just retry a CAS loop, because retrying is exactly what allows starvation. You need a mechanism that *bounds* the number of times any thread can be forced to retry.

The standard trick is **helping**. In a wait-free algorithm, a thread that is about to perform its operation first announces its intended operation in a shared array (a "descriptor"). Then, before any thread does its own work, it scans the announcement array and *helps complete* any operation it finds pending — including operations announced by threads that are currently descheduled. The effect: even if your thread is preempted right after announcing, some other thread will see your announcement and finish your operation *for* you. You are guaranteed to complete within a bounded number of steps because there are only finitely many other threads, and each can delay you at most once before it is obligated to help you. That bound is what upgrades lock-free to wait-free.

The cost is exactly what you would fear. Helping means every operation does extra work — scanning the announcement array, checking descriptors, performing CAS on behalf of others — on the *fast path*, when there is no contention at all and nobody actually needs help. A wait-free queue pays this overhead on every single `enqueue`, contended or not, to guarantee the worst case. So wait-free structures are typically *slower on average* than lock-free ones, even though they have a better worst case. You buy a bounded tail latency with a worse median. That trade is worth it for an anti-lock-braking controller and almost never worth it for a web cache.

#### Worked example: the helping bound

Make the bound concrete with the cleanest illustration, the wait-free version of the announcement scheme. Suppose $n$ threads share a wait-free queue. Thread $T_i$ wants to enqueue. It writes a descriptor for its operation into slot $i$ of a shared `announce` array, then enters the main loop. In the loop, before doing its own enqueue, it reads a shared `help` counter to pick *whose* operation to help this round (round-robin over the $n$ slots), and completes that thread's pending operation if any. 

Now bound $T_i$'s steps. Once $T_i$ has announced, the round-robin help pointer will, within at most $n$ rounds, land on slot $i$ — and at that point *every* active thread is obligated to help $T_i$ before doing its own work. So even if $T_i$ is preempted forever right after announcing, within $O(n)$ total system rounds its operation is completed by others. Each round is a bounded number of CAS operations. Therefore $T_i$'s operation completes in $O(n)$ steps *worst case*, independent of how unlucky $T_i$ is. That $O(n)$ bound — finite, dependent only on the thread count, not on the adversary's schedule — is the formal content of "wait-free." Contrast it with the lock-free CAS loop, whose retry count has *no* finite bound, because nothing stops an adversary from beating the same thread forever.

The lesson is not that you should write helping schemes — you almost never should, and the [next post](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) will build lock-free structures that are far simpler. The lesson is *why* wait-free is hard and *why* it costs more: bounding every individual thread's latency requires other threads to do work on its behalf, and that work is paid on every operation, including the common case where nobody is stalled.

## Obstruction-free and the livelock it permits

Skip back down one rung to obstruction-free, the level most people forget exists, because understanding it is the cleanest way to see *why* lock-free needs its specific guarantee and not a weaker one. Obstruction-free is the floor of the non-blocking world: it promises only that a thread running *in isolation* — every other thread suspended — finishes in a bounded number of steps. That is a real and useful promise. It means no thread can ever be *blocked* by a stalled peer, which is the whole point of going non-blocking. If a peer stalls, the running thread is, by definition, running alone, so it completes. The stalled-holder catastrophe is gone.

What obstruction-free does *not* promise is anything about contention. Two obstruction-free threads, both running, can interfere with each other indefinitely — each one's attempt aborts the other's, both restart, and neither ever finishes. This is *livelock*: not a deadlock (the threads are not stuck waiting; they are furiously busy), but no progress, because every step undoes a peer's step. Livelock is the failure mode lock-free rules out and obstruction-free allows, and it is the precise reason lock-free is a strictly stronger guarantee.

#### Worked example: two threads livelocking on an obstruction-free swap

Picture the canonical livelock, the one that shows up in optimistic transactional schemes. Two threads each want to atomically move an item between two locations, A and B. The obstruction-free protocol is: tentatively "claim" location A by writing your thread id into it with a CAS, then claim location B the same way, then commit; if you find a *peer's* claim already sitting on a location you need, you *abort your own claim* (to avoid deadlock — you never hold one claim while waiting on another) and restart from scratch.

Now run two threads in lockstep. T1 claims A. T2 claims B. T1 tries to claim B, finds T2's id there, so it aborts — releasing its claim on A — and restarts. T2 tries to claim A, finds it just got released, but meanwhile T1 has restarted and re-claimed A; T2 finds T1's id on A, so *it* aborts — releasing B — and restarts. Now T1 tries B again, finds it free, claims it, then tries to commit... but T2 has restarted and is re-claiming, and the dance repeats. Each thread is executing instructions as fast as the CPU allows. Neither is blocked. Neither ever commits. The system makes *zero* useful progress while burning 100% CPU on both cores. That is livelock, and it is a *legal* execution under obstruction-freedom — the guarantee was only "a thread alone finishes," and neither thread is ever alone.

This is exactly what lock-freedom forbids. A lock-free protocol guarantees that *some* thread makes progress on every system step, so this perpetual mutual-abort cannot happen — at least one of T1 or T2 must be advancing. The standard way to *rescue* an obstruction-free algorithm is to bolt on a **contention manager**: when a thread detects it has aborted too many times, it backs off for a randomized interval (exponential back-off, the same idea as Ethernet's collision handling), so one thread pauses, the other pulls ahead and commits, and the livelock breaks. Obstruction-free + a good contention manager *behaves* like lock-free in practice while being simpler to implement, because you separate the "make it correct in isolation" problem from the "make it live under contention" problem. This is precisely the design point that early software transactional memory occupied; the [STM post](/blog/software-development/concurrency/software-transactional-memory-and-optimistic-concurrency) walks the optimistic-concurrency version of the same idea in depth.

The reason this rung matters for your mental model: it shows that "non-blocking" alone is *not enough* to guarantee progress under contention. You need the specific lock-free promise — *some* thread always advances — to rule out livelock, and you need the wait-free promise — *every* thread advances — to rule out starvation. The three non-blocking levels are not arbitrary academic distinctions; each one closes a specific failure mode (blocked-by-stall, livelock, starvation) that the level below it leaves open. That is the entire shape of the hierarchy, and it is why the figure at the top of this post draws three distinct rungs above the blocking floor rather than one undifferentiated "non-blocking" box.

### The mechanism underneath: why CAS, and why ABA lurks

One more mechanism note before we measure, because it connects this post to two siblings. Every level above blocking rests on an atomic *read-modify-write* primitive, and the reason is not convenience — it is a hard theoretical result we will state precisely in the case studies. Plain atomic reads and writes (a `volatile` flag, a single aligned word load/store) are *not enough* to build a correct lock-free stack for even two threads. You genuinely need an instruction that reads, decides, and writes as one indivisible step. CAS is the universal such primitive.

But CAS carries a famous trap that the [ABA post](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads) covers in full and that you must at least *recognize* here, because it is the most common way a hand-rolled lock-free structure is subtly wrong. CAS checks that a word still holds the *value* it expected — not that the word was *never touched* since you read it. So a word can change from A to B and back to A while your thread was descheduled; your CAS sees A, assumes nothing happened, and succeeds — even though the world moved underneath you. On a pointer-based lock-free stack this is lethal: the node you "saw" at the head can be popped, freed, and a *different* node reallocated at the same address, so your CAS swings the head to a node whose `next` now points into garbage. The ABA problem is *why* lock-free memory reclamation is its own hard subject (the [reclamation post](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu)): you cannot safely free a node while another thread might still CAS against it. None of this arises behind a lock, because the lock prevents the concurrent free in the first place. It is one more line on the bill for climbing the hierarchy — the progress guarantee is free of charge in theory, but the *memory model and reclamation* it drags in are where the real engineering cost hides. We mention it here so the cost ladder later is honest about what "moderate cost" for lock-free actually includes.

## Progress is not speed: the low-contention surprise

Here is the claim that catches engineers off guard and that you must internalize before you ever reach for a lock-free structure in production: **a lock-free data structure can be slower than a lock.** Sometimes much slower. Lock-freedom is a *progress* guarantee, not a *performance* guarantee, and at low contention the progress guarantee buys you nothing while its overhead costs you something real.

![A before and after figure showing a mutex winning under low contention where a lock-free CAS loop wastes retries, and lock-free winning under high contention or preemption where the mutex holder stalls](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-5.png)

Walk the figure. On the left, low contention: one or two threads, collisions are rare. The mutex acquire is a single uncontended atomic instruction (~25 ns) and the critical section is tiny; the lock is essentially free. The lock-free CAS loop, meanwhile, still has to load, compute, and CAS — and under any contention at all it occasionally retries, doing the work twice. On the right, high contention or preemption: the mutex holder can stall and freeze everyone, while the lock-free structure keeps some thread moving. The two regimes have *opposite* winners. There is no universal answer; there is only "which regime are you in?"

Why is the mutex often faster at low contention? Three reasons, all mechanical:

1. **The uncontended fast path is already lock-free-ish.** A modern mutex (`futex`-backed on Linux, as the [lock post](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) details) does a single CAS to acquire when uncontended, never entering the kernel. So the "cheap" case of a mutex is *one atomic instruction* — the same primitive cost as a lock-free CAS, with no retry loop on top.
2. **Lock-free structures often do more memory traffic.** A lock-free queue or stack typically allocates a node, does a CAS on a shared head/tail pointer, and — critically — must solve memory reclamation (hazard pointers, epochs, or RCU; that is the whole [reclamation post](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu)). That bookkeeping is pure overhead the mutex version skips entirely.
3. **Retries are wasted work.** Every failed CAS recomputed a value and threw it away. At low contention failures are rare, but they are not zero, and each one is dead cycles plus a cache-line bounce.

#### Worked example: a measured lock vs lock-free counter

Let us put numbers on it, honestly. The setup: a single shared 64-bit counter, $N$ threads each incrementing it 10 million times, on a machine with a handful of physical cores. Compare a `sync.Mutex`-guarded `value++` against an `atomic.AddInt64` (`lock xadd`). Measure total wall-clock time, warmed up, best of several runs, and report nanoseconds per increment. These are representative order-of-magnitude figures from this kind of microbenchmark on a typical x86 server, not a precise measurement of your machine — *measure your own*, because the numbers depend heavily on CPU, core count, and NUMA topology. The shape, however, is robust and reproducible.

| Threads | Mutex `value++` (ns/op) | Atomic `AddInt64` (ns/op) | Faster |
| --- | --- | --- | --- |
| 1 | ~15 | ~7 | atomic |
| 2 | ~60 | ~25 | atomic |
| 4 | ~120 | ~55 | atomic |
| 8 | ~250 | ~110 | atomic |

Here the atomic wins at every thread count — but notice *why*, and notice the trap. `atomic.AddInt64` is `lock xadd`, a *single* hardware instruction with no retry loop; it is wait-free. This is the case where lock-free (in fact wait-free) genuinely beats the lock, because the operation is exactly one atomic primitive and the lock's extra bookkeeping is pure loss. Now change the experiment so the lock-free version must use a *CAS loop* instead of a single instruction — say, "increment only if the value is even," which the hardware cannot do in one instruction:

| Threads | Mutex (ns/op) | CAS loop (ns/op) | Faster |
| --- | --- | --- | --- |
| 1 | ~16 | ~12 | CAS loop |
| 2 | ~62 | ~70 | mutex |
| 4 | ~125 | ~190 | mutex |
| 8 | ~255 | ~520 | mutex |

Now the mutex wins from two threads up, and the gap *widens* with contention, because the CAS loop's retry rate climbs — at 8 threads hammering one word, most CAS attempts fail and retry, so the loop does its work several times over. This is the surprise made concrete: **a correct lock-free CAS loop, under contention, can be several times slower than a plain mutex**, because the mutex serializes cleanly (one winner, others sleep) while the CAS loop turns contention into a storm of failed retries and cache-line ping-pong. The lock-free structure is making the system-wide progress guarantee the whole time — it just costs more to do so.

The honest takeaway: the *only* time lock-free's progress guarantee is worth paying for is when blocking's failure mode — a stalled holder freezing the system — is actually intolerable in your setting. If your threads are never preempted at a bad time, if you have no real-time deadline, if a mutex is not your measured bottleneck, then lock-free is a more complex, often slower, harder-to-get-right way to do something a mutex does cleanly. Measure first. We will return to this in the recommendation section.

## Linearizability: the correctness condition

Progress is half the story and the less important half. A structure that "always makes progress" toward returning *wrong answers* is worthless. The other half is *correctness*: when several threads operate on a shared object concurrently, what does it even *mean* for the object to be correct? The answer the field settled on is **linearizability**, and it is the gold standard for concurrent objects.

The definition. An execution is *linearizable* if every operation appears to take effect *atomically* — instantaneously — at some single instant between the moment the operation was invoked (you called it) and the moment it returned (it gave you an answer). That instant is the operation's *linearization point*. Equivalently: you can take the concurrent, overlapping operations, pick one point inside each operation's call-to-return interval, and the resulting sequential order of those points produces a history that a single-threaded version of the object would have produced. If such a choice of points exists for every operation, the execution is linearizable.

![A timeline of a single push operation showing its invocation, internal steps, the compare and swap that is its linearization point where the effect becomes visible, and its return](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-7.png)

The figure traces one operation's life. You call `push`; that is the *invocation*. Internally the operation reads the head, prepares a node, and does a CAS. The instant the CAS succeeds is the *linearization point* — that is when the push "really happened," when the new element becomes visible to every other thread atomically. Then the call returns. The push spanned a real-time interval, but it *appears* to have taken effect at one instant inside that interval. Crucially, the linearization point is a *real* moment in the execution — the CAS instruction retiring — not a fiction. For most lock-free structures there is a specific, identifiable instruction that *is* the linearization point, and identifying it is how you prove the structure correct.

Why does this matter so much? Because linearizability is *composable* and *intuitive*. It means you can reason about a concurrent object as if every operation were instantaneous and the operations happened in *some* sequential order consistent with real time. If thread A's `push(x)` returned before thread B's `pop()` was even called, then B's pop must observe a state where x is present — because A's effect was, by real time, already in place. Linearizability respects the real-time order of *non-overlapping* operations. That is exactly the property that lets you treat the object like a sequential one in your head, which is the only way human reasoning about concurrent code stays tractable.

### Linearizability versus sequential consistency

Linearizability is often confused with the weaker *sequential consistency*, and the difference is worth nailing down because it shows up in the [memory-model post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) too. Both say there exists *some* sequential order of operations that respects each thread's *program order*. The difference is the real-time constraint:

- **Sequential consistency** requires only that each thread's operations appear in *its own program order* in the global sequence. It does *not* require that the sequence respect real time *across* threads. So an operation that finished before another even started can still be ordered *after* it, as long as no single thread's program order is violated.
- **Linearizability** additionally requires that the sequential order respect *real-time* order: if operation P returned before operation Q was invoked, P must come before Q in the order. This is the stronger, more intuitive guarantee.

![A matrix comparing linearizable, sequentially consistent, and unsynchronized access by what each guarantees and a concrete example of each](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-6.png)

The matrix lays the three side by side. Linearizable: an atomic effect at a point inside the real-time window — a correctly built lock-free queue. Sequentially consistent: per-thread order preserved but no real-time tie across threads — the behavior you get from some relaxed-atomic constructions. And the bottom row, the one with no synchronization at all: torn reads, lost updates, no guarantee of anything — a plain shared counter incremented without atomics, the [original race](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) this whole series opened with. The practical rule: **lock-free data structures aim to be linearizable.** When a library says its queue is "lock-free and linearizable," it is promising both halves — the system always progresses *and* every operation appears atomic at a real instant. You want both. Progress without linearizability is fast corruption; linearizability without progress is a mutex.

#### Worked example: spotting the linearization point

Take the lock-free stack push (the Treiber stack, which the next post builds in full). The operation is:

```c
void push(stack *s, node *n) {
    node *old_head;
    do {
        old_head = atomic_load(&s->head); // read current head
        n->next = old_head;               // point new node at it
    } while (!atomic_compare_exchange_weak(&s->head, &old_head, n));
    // success: the CAS that set head = n is the linearization point
}
```

Where, exactly, does this push "take effect"? Not at the `atomic_load` — at that point nothing is visible to other threads. Not when `n->next` is set — that is a private write. The push takes effect at the *instant the successful CAS retires*, swinging `head` to `n`. Before that instruction, no thread can see `n` in the stack; after it, every thread can. The transition is atomic and instantaneous because CAS is. So the linearization point is the successful CAS, and it falls *inside* the call-to-return interval (after the call, before the return), which is exactly what linearizability requires. 

Now the subtle part: the operation may *loop*, doing the load and CAS several times. Only the *last, successful* CAS is the linearization point; the failed attempts had no visible effect (a failed CAS does not write). So even though the operation took many instructions over a real-time interval, it linearizes at one instant. This is the general method for proving a lock-free structure correct: for every operation, point to the single instruction that is its linearization point and show it sits inside the operation's interval. If you can do that for every operation in every execution, the structure is linearizable. The existence of a clean linearization point is, in practice, what separates a correct lock-free design from a subtly broken one.

## The cost ladder: stronger guarantees, more machinery

Step back and look at the whole hierarchy as a cost ladder, because the engineering decision is always "how far up do I need to climb, and what does each rung cost?"

| Guarantee | Progress promise | Failure it allows | Typical mechanism | Relative cost |
| --- | --- | --- | --- | --- |
| Blocking (lock) | none if holder stalls | deadlock, frozen system | mutex, `futex` | lowest |
| Obstruction-free | a lone thread finishes | livelock under contention | CAS, no contention mgmt | low |
| Lock-free | some thread always advances | individual starvation | CAS retry loop | moderate |
| Wait-free | every thread bounded | none | helping / announce array | highest |

Each rung up the ladder removes a failure mode and adds machinery. Going from blocking to lock-free removes "frozen by a stalled holder" and adds CAS loops plus *memory reclamation* — the hardest part, because in a lock-free structure you can free a node only when you are *certain* no other thread still holds a pointer to it, and there is no lock to tell you that. That is the entire subject of the [reclamation post](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu): hazard pointers, epoch-based reclamation, and RCU exist precisely because lock-free progress made "when is it safe to `free`?" genuinely hard. Going from lock-free to wait-free removes "individual starvation" and adds helping schemes, which tax the fast path. 

The general principle, and the most important sentence in this section: **climb only as high as your requirements force you.** Every rung costs complexity, and complexity in concurrent code is where the subtle, once-a-week, impossible-to-reproduce bugs live. A lock-free structure is dramatically harder to write correctly than a mutex-guarded one — the ABA problem, memory ordering, reclamation, and the linearization-point proof are all things you simply do not deal with behind a lock. A wait-free structure is harder still. The cost is not just runtime overhead; it is *engineering risk*. Most production systems are correct and fast with a well-chosen lock. Reach higher only when you have measured that you must.

### What "moderate cost" for lock-free actually buys you

When the table says lock-free is "moderate cost," be honest about what is bundled into that word, because the surprise on the bill is what bites teams that adopt a lock-free structure expecting a mutex with a better progress guarantee. The CAS retry loop itself is cheap. The expensive parts are the things the lock made free: *memory reclamation* (you cannot `free` a node while a peer might still touch it, so you need hazard pointers, epochs, or RCU — a whole subsystem the mutex version never needed), *the ABA defense* (tagged pointers, double-width CAS, or a reclamation scheme that doubles as ABA protection), and *the memory-ordering reasoning* (you must place acquire/release fences correctly so that a node's fields are visible before the pointer that exposes it — get this wrong and you have a race that only manifests on a weakly-ordered machine like ARM, never on your x86 laptop, which is the worst kind of bug to ship). None of these are line items on a mutex. They are the real reason a lock-free structure is "moderate cost" rather than "free," and the reason hand-rolling one is an expert task rather than a weekend project. The progress guarantee is the easy part; the safety machinery around it is the hard part, and the next two posts in this track exist precisely to deal with it.

Obstruction-free, by contrast, sits at "low cost" on the ladder because it deliberately punts on the hard part: it makes no promise under contention, so its implementation does not need to bound the contended case at all — you write the optimistic operation, abort cleanly on conflict, and *separately* attach a contention manager to handle livelock. That separation of concerns is exactly why some software transactional memory systems chose it: get correctness in isolation right first, then make it live under load with a back-off policy you can tune independently. It is a legitimate engineering point on the ladder, not a theoretical curiosity, and recognizing it stops you from assuming "non-blocking" automatically means "lock-free."

## Measured behavior: lock vs lock-free under a stalled thread and under contention

We have already measured the low-contention case where the lock-free CAS loop loses. Now measure the case lock-free is *for*: a stalled thread. This is the honest center of the whole argument — lock-free's benefit is invisible until a thread stalls at the wrong moment, and then it is the difference between a system that limps and one that freezes.

#### Worked example: injecting a stall

Construct the experiment to *isolate* the progress guarantee. Run $N$ worker threads hammering a shared structure — a counter or a stack. Periodically, with small probability, a worker that is "inside" an operation is forced to pause for a fixed stall (simulate a preemption or page fault by sleeping for, say, 5 ms while holding the lock / mid-CAS). Measure total system throughput — operations completed per second across *all* threads — during the stall windows. The design question: does the system keep making progress while one thread is stalled?

The behavior, stated as the robust qualitative result (your exact numbers depend on platform; *measure your own*):

| Scenario | Mutex (blocking) | Lock-free |
| --- | --- | --- |
| No stalls, low contention | fast (~baseline) | similar or slightly slower (retry + reclamation overhead) |
| No stalls, high contention | serializes, throughput plateaus | retries climb, can be slower than mutex |
| One thread stalls 5 ms mid-op | **all threads block; system throughput → 0 for 5 ms** | other threads keep going; system throughput barely dips |
| Many random short stalls | throughput collapses (stalls stack up) | throughput stays smooth |

That third row is the entire point of lock-free, isolated. With the mutex, a stalled holder takes the lock down with it and *every* thread's throughput drops to zero for the full 5 ms — you can see it as a flat-line in the throughput trace, a dead window. With the lock-free structure, the stalled thread simply is not attempting CAS during its stall; the other $N-1$ threads keep CASing and keep advancing, and the system's throughput barely registers the stall. The lock-free structure converted "one thread's stall freezes everyone" into "one thread's stall slows nothing." 

This is also why lock-free shows up in places where stalls are *guaranteed* to happen at bad times: signal handlers (which interrupt a thread mid-execution — if the interrupted thread held a lock the handler needs, instant deadlock), real-time threads (which preempt lower-priority threads that might hold a lock), and the kernel (where an interrupt can fire while a thread holds a lock, and the interrupt handler cannot block). In all three, *some* thread is going to be stopped at the worst possible instant, by construction. Blocking is not an option there; the stall is not a rare accident, it is the design. That is the precise situation lock-free is built for.

The contention curve tells the other half. As you add threads contending on a single hot word, the mutex *serializes* — throughput plateaus, because only one thread is ever in the critical section, but it does so cleanly. The lock-free CAS loop *thrashes* — as contention rises, the CAS failure rate rises, retries multiply, the hot cache line ping-pongs between cores, and throughput can actually *decline* past some thread count. Neither is universally better. The mutex degrades gracefully under contention but freezes under stalls; the lock-free structure shrugs off stalls but can thrash under contention. Knowing *which* failure mode you face is the whole decision, which is exactly what the next section is about.

#### Worked example: reading a throughput-vs-threads curve honestly

To make the contention story actionable rather than vibes, here is how you would *measure* and *read* the curve, and the traps that make naive measurements lie. Plot operations-per-second on the y-axis against thread count on the x-axis, for both the mutex and the lock-free version, on a machine with $C$ physical cores. Three regions appear, and each one means something:

- **Below saturation (threads $\le$ a few, contention rare).** Both lines climb. The lock-free line may sit slightly *below* the mutex line here if it carries reclamation and retry overhead — this is the low-contention surprise made visible. Do not "fix" it by adding threads; you are not contention-bound yet.
- **Around saturation (threads near $C$, the hot word is now genuinely contended).** The mutex line *flattens* — it cannot exceed the rate at which one core can run the critical section back to back, because the lock serializes. The lock-free line may still climb a little or also flatten, depending on how much real parallel work surrounds the CAS.
- **Past saturation (threads $\gg C$, oversubscribed).** Here the lines *diverge in behavior*, not just height. The mutex line may dip as lock convoying and context-switch overhead pile up, and — the key point — if any holder is preempted (which is *guaranteed* when threads outnumber cores), you get the stall-induced dead windows that the stall experiment isolated. The lock-free line is flatter through this region precisely because there is no holder to preempt.

The traps that produce a lying curve: not warming up (the first few thousand operations pay JIT/branch-predictor/cache costs and look slow); measuring only a few iterations (concurrency throughput is noisy — run millions of ops, take the median of several runs, report the spread); pinning all threads to one core (then everything serializes and you measure nothing); and forgetting that on a [weakly-ordered machine](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) the *correctness* of the lock-free version depends on memory ordering you cannot observe on x86. The honest report is "median of N runs, warmed up, on this named CPU with this core count, and here is the variance" — never a single precise number presented as universal truth. The shape is robust; the exact numbers are yours to measure.

## Case studies / real-world

Three grounded examples, because the hierarchy is not academic — it is the vocabulary real systems and real papers use.

**Herlihy's wait-free hierarchy and the universality of CAS.** In his 1991 paper *Wait-Free Synchronization*, Maurice Herlihy proved one of the deepest results in concurrency theory: synchronization primitives form a hierarchy by their *consensus number* — the maximum number of threads for which the primitive can solve wait-free consensus. Plain atomic registers (read/write) have consensus number 1: they cannot solve wait-free consensus for even two threads. `test-and-set` and `fetch-and-add` have consensus number 2. And *compare-and-swap* has consensus number *infinity* — it can solve wait-free consensus for any number of threads, which makes it *universal*: with CAS you can build a wait-free implementation of *any* sequential object. This is the theoretical reason CAS is the primitive at the heart of essentially every lock-free and wait-free structure, and why hardware designers made sure to provide it. It also explains a practical frustration: because read/write registers have consensus number 1, you genuinely *cannot* build a correct lock-free stack out of plain `volatile` flags — you need a real read-modify-write primitive, and Herlihy's result tells you exactly why. (Source: M. Herlihy, "Wait-Free Synchronization," *ACM TOPLAS* 13(1), 1991.)

**The Java `ConcurrentLinkedQueue` (the Michael-Scott queue).** The lock-free queue in `java.util.concurrent` is a direct implementation of the Michael & Scott (1996) algorithm, the most-cited lock-free queue in existence. It is *lock-free and linearizable*: every `offer` and `poll` is provably linearizable (each has an identifiable CAS as its linearization point), and the system always makes progress — but it is *not* wait-free, since an individual `offer` can retry indefinitely under adversarial contention. The algorithm has a famous subtlety that illustrates the whole field: an enqueuer does the operation in *two* CAS steps (swing the tail's `next` pointer, then advance `tail`), and if the enqueuer stalls between them, the queue is in an intermediate state — so *other* threads must "help" by completing the lagging `tail` advance before doing their own work. That helping is a microcosm of how lock-free structures stay correct when a thread stalls mid-operation: peers fix up the dangling state rather than waiting. It is also a perfect example of why memory reclamation is hard here — you cannot just free a dequeued node, because a stalled thread might still be reading it, which is precisely the problem hazard pointers solve. (Source: M. Michael & M. Scott, "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms," PODC 1996; OpenJDK `ConcurrentLinkedQueue` source.)

**RCU in the Linux kernel — lock-free reads where stalls are unavoidable.** Read-Copy-Update is the kernel's workhorse for read-mostly data (the dentry cache, network routing tables, and more), and it exists in large part *because the kernel cannot tolerate blocking on the read path*. A reader holding a lock that an interrupt handler then needs is an instant deadlock; a reader preempted while holding a lock can stall everything. RCU sidesteps this: readers take *no locks and no atomic writes* — a read-side critical section is, on many configurations, free of any synchronization instruction at all, which makes reads effectively wait-free and extremely fast. Writers make a *copy*, modify it, and atomically swing a pointer (the linearization point), then wait for a "grace period" — until every pre-existing reader has finished — before freeing the old version. RCU is the [reclamation post's](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu) headline technique, and the reason it earns its complexity is exactly the theme of this post: in the kernel, *some* thread is always about to be interrupted at the worst instant, so the read path must never block. (Source: P. McKenney et al., RCU documentation in the Linux kernel tree; "What is RCU, Fundamentally?", LWN.)

These three together make the abstract hierarchy concrete: Herlihy tells you *which primitive* can build a given level (CAS is universal); the Michael-Scott queue shows a *real lock-free, linearizable, not-wait-free* structure with helping and reclamation; and RCU shows *why* a real system pays the complexity — because in its world, blocking is not a performance choice, it is a correctness impossibility.

## When to reach for this (and when not to)

Be decisive, because the most common lock-free mistake is reaching for it when a mutex would have been correct, faster, and a tenth the code.

![A matrix mapping blocking lock, lock-free, and wait-free to the situations where each is the right choice and the situations where each should be avoided](/imgs/blogs/the-progress-hierarchy-blocking-lock-free-and-wait-free-8.png)

**Default to a good lock.** For the overwhelming majority of shared-state problems, a well-chosen mutex (or a reader-writer lock, or a sharded set of locks) is correct, fast, and *readable*. The uncontended fast path is one atomic instruction; modern mutexes do not enter the kernel unless threads actually collide. If your critical sections are short, your contention is low to moderate, and no thread is going to stall at a catastrophic moment, a lock is the right tool. Do not go lock-free because it sounds advanced. Go lock-free because you measured a problem a lock cannot solve.

**Reach for lock-free when a thread can be stalled at the worst possible time and that is intolerable.** This is the real trigger, and it is narrower than people think. Three concrete situations:

- **You share state with code that can preempt or interrupt the holder.** Signal handlers, real-time threads, interrupt handlers, the kernel. If a high-priority thread or an interrupt can fire while a low-priority thread holds the lock, you have priority inversion or outright deadlock waiting to happen. Lock-free removes the holder entirely, so there is nothing to be inverted or deadlocked.
- **You have a hard real-time deadline on an individual operation.** A control loop that must complete every cycle in a bounded time cannot tolerate "blocked indefinitely on a stalled holder." Here you may need to climb all the way to *wait-free*, because only wait-free bounds the *individual* operation's latency — lock-free bounds only the system's, and your control loop is an individual.
- **You have measured a lock as your bottleneck under genuine high contention** and a sharded-lock or rethink-the-data-structure approach has failed. Even then, prefer a battle-tested library structure (`ConcurrentLinkedQueue`, a crossbeam queue, a `folly` MPMC queue) over hand-rolling one.

**Avoid lock-free when:**

- A mutex is not your measured bottleneck. The single most common error. Profile first.
- Your contention is low. At low contention lock-free's overhead (reclamation, retries) is pure loss — the lock was already nearly free.
- You need a multi-object invariant. Lock-free structures linearize *individual* operations on *one* object; if you need "move an item from queue A to queue B atomically," a single lock (or STM, or a transaction) is dramatically simpler and the lock-free version may be impossible without a much more complex multi-word primitive.
- You cannot afford the engineering risk. Lock-free code that is subtly wrong fails once a week in production with no reproducer. If you do not have the time, the test infrastructure ([race detectors and stress testing](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing)), and the review depth to get it right, a lock you understand beats a lock-free structure you do not.

**Avoid wait-free unless you are in hard real-time.** Wait-free's helping schemes tax every operation to bound the worst case. Outside a domain where a bounded *individual* latency is a hard requirement — avionics, motor control, some audio paths — the median-throughput cost is not worth a tail-latency bound you do not need. Most "I need lock-free" situations actually need *lock-free*, not *wait-free*; do not over-climb.

The decision in one line: **pick the weakest progress guarantee that meets your stall-tolerance and deadline requirements, then prove correctness via linearization points and measure it.** That is the same "name what is shared, order the accesses, choose the cheapest mechanism, measure" discipline as the rest of the series, applied to the one axis — progress under adversarial scheduling — that this corner of concurrency is about. The [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) ties this decision into the larger "which model?" tree, and the Python story — where the GIL changes the contention math entirely — lives in the [python-performance series](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs), which you should read OUT to rather than reasoning about lock-free Python primitives here.

## Key takeaways

- **The progress hierarchy has four rungs.** Blocking (a stalled thread can freeze others), obstruction-free (a lone thread finishes), lock-free (some thread always advances system-wide), wait-free (every thread finishes in bounded steps). Each rung removes a failure mode and adds machinery.
- **Lock-free is a progress guarantee, not a speed claim.** It promises the *system* always advances despite any stall — not that any individual thread is fast, and not that the structure beats a mutex. A lock-free CAS loop under contention can be several times *slower* than a plain lock.
- **The blocking failure is a stalled holder.** A mutex holder that is preempted or page-faulted halts every waiter for the full stall — a 5 ns critical section can freeze the system for 10 ms. Lock-free exists to eliminate exactly this.
- **A failed CAS means someone else succeeded.** That is the lock-free progress proof: every CAS failure is caused by another thread's success, so the system can never stall — though an individual thread can starve, which is the gap to wait-free.
- **Wait-free bounds the individual; lock-free bounds only the system.** Wait-free uses *helping* (threads complete each other's announced operations) to guarantee no starvation, at a fast-path cost paid on every operation. Reach for it only under hard real-time.
- **Linearizability is the correctness condition.** Every operation must appear to take effect atomically at one *linearization point* between its call and return, respecting real-time order across threads. For lock-free structures, that point is usually a specific successful CAS — identifying it is how you prove the structure correct.
- **Single-instruction primitives are wait-free for free.** `fetch_add`, `fetch_or`, `exchange` complete in bounded steps with no retry. Prefer them over a hand-rolled CAS loop when the hardware offers the whole operation atomically — you get the stronger guarantee at no cost.
- **Default to a good lock; climb only when forced.** Go lock-free only when a thread can stall at a catastrophic time (signals, interrupts, real-time, the kernel) or when you have measured a lock as a true bottleneck under real contention. Go wait-free only for hard deadlines. Stronger guarantees cost complexity, and complexity in concurrent code is where the impossible bugs live.

## Further reading

- **Maurice Herlihy & Nir Shavit, *The Art of Multiprocessor Programming*** (2nd ed.) — the definitive treatment of the progress hierarchy, linearizability, the consensus hierarchy, and wait-free universal constructions. The source for everything in this post; read chapters 3 (concurrent objects / linearizability) and 5–6 (consensus, universality).
- **Maurice Herlihy, "Wait-Free Synchronization,"** *ACM TOPLAS* 13(1), 1991 — the original paper proving the consensus hierarchy and that CAS is universal. Short and foundational.
- **Maged Michael & Michael Scott, "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms,"** PODC 1996 — the lock-free queue behind Java's `ConcurrentLinkedQueue`; a clean worked example of helping and linearization points.
- **Paul McKenney et al., "What is RCU, Fundamentally?"** (LWN) and the Linux kernel RCU documentation — the canonical read-mostly, near-wait-free read path, and why the kernel needs non-blocking reads.
- **Herlihy & Wing, "Linearizability: A Correctness Condition for Concurrent Objects,"** *ACM TOPLAS* 12(3), 1990 — the paper that defined linearizability; the precise version of this post's correctness section.
- **The within-series companions**: the [series intro](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) for the shared-state frame; [how a lock is built](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) for the CAS instruction underneath all of this; [compare-and-swap and building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) for the Treiber stack and Michael-Scott queue in full; [memory reclamation](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu) for the hardest part — freeing a node safely; and the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) for fitting this decision into the larger model choice.

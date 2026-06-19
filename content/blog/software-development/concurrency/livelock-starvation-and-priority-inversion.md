---
title: "Livelock, Starvation, and Priority Inversion"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a system can run at 100% CPU and still make zero progress — and how to fix the liveness bugs that deadlock detectors never catch."
tags:
  [
    "concurrency",
    "parallelism",
    "livelock",
    "starvation",
    "priority-inversion",
    "fairness",
    "liveness",
    "scheduling",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/livelock-starvation-and-priority-inversion-1.png"
---

Picture the page you actually get at 2 AM. The dashboard says every core is pinned at 100%. Load average is climbing. The service is up — health checks even pass, sometimes — but the request queue is growing without bound and p99 latency has gone to infinity. You SSH in, run `top`, and see your worker threads burning CPU like they're doing real work. You attach a profiler and they *are* doing work: grabbing a lock, checking a flag, backing off, retrying, grabbing the lock again. Round and round. The machine is screaming and the business is getting exactly nothing for it.

This is not a deadlock. A deadlock is quiet — threads parked on a wait queue, CPU near zero, everything frozen. What you have is worse in one specific way: it *looks* healthy. Every thread is runnable, the scheduler is happily time-slicing them, and yet the count of completed transactions is flat. The threads are alive but the system is dead. That gap — running but not progressing — is the subject of this post, and it has three classic faces: **livelock** (threads endlessly react to each other and retry), **starvation** (one thread is perpetually denied a resource others keep taking), and **priority inversion** (a high-priority task is blocked behind a low-priority one that a medium task keeps shoving aside). They are the *liveness* failures, the bugs your deadlock detector will never catch because, technically, nothing is deadlocked.

![A comparison table of deadlock, livelock, and starvation across whether threads are running, whether progress happens, and the root cause](/imgs/blogs/livelock-starvation-and-priority-inversion-1.png)

By the end of this post you will be able to: tell these three apart on sight from a profiler trace; reproduce a livelock from naive deadlock-avoidance and break it with asymmetry or randomized back-off; recognize the priority-inversion chain that famously froze a spacecraft on Mars and apply the priority-inheritance fix; reason about when a *fair* lock is worth its throughput tax and when it isn't; and write a dining-philosophers solution that is provably free of deadlock, livelock, and starvation all at once. If you've read [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), you already know the core hazard of this whole series: shared mutable state plus nondeterministic scheduling. Liveness bugs are what happens when you tame the *correctness* hazard (no data races, no lost updates) but forget that "no two threads corrupt each other" is not the same as "every thread eventually gets its turn."

## Liveness versus safety: the property we keep forgetting

Concurrency correctness has two halves, and most engineers internalize only one of them. The first half is **safety**: nothing bad ever happens. No two threads write the same variable without a happens-before edge; no invariant is ever observed broken; the balance is never off by a cent. Safety is what a mutex, an atomic, or a lock-ordering discipline buys you. It is the half everyone learns first because it's the half that corrupts data.

The second half is **liveness**: something good eventually happens. The request eventually completes. The lock is eventually acquired. The thread eventually gets to run. Liveness is a *progress* guarantee, and it is exactly what livelock, starvation, and priority inversion violate. A formally minded way to say it: safety properties are violated by a *finite* prefix of execution (you can point at the exact moment the invariant broke), while liveness properties can only be violated by an *infinite* execution (you can never point at a single moment and say "here is where it failed to make progress forever" — you can only observe that it keeps not progressing). That asymmetry is why liveness bugs are so maddening to debug: there is no smoking gun, no corrupted byte, no assertion that fires. There is only the absence of forward motion, and absence is hard to put a breakpoint on.

It helps to name a hierarchy of progress guarantees, because "liveness" is not one thing. From weakest to strongest:

- **Deadlock-freedom (system-wide progress):** *some* thread always eventually makes progress. The system as a whole never freezes — but a *particular* thread might starve forever while others charge ahead. This is the weakest useful guarantee.
- **Starvation-freedom (per-thread progress):** *every* thread that tries eventually makes progress. No individual is left behind. This is what fair locks and fair schedulers aim for.
- **Lock-freedom** and **wait-freedom** are the non-blocking analogues (lock-free = some thread completes in a bounded number of system steps; wait-free = *every* thread completes in a bounded number of its *own* steps), which we cover in the lock-free posts of this series. The same ladder, just without locks.

Livelock violates even deadlock-freedom: *no* thread progresses though all are running. Starvation satisfies deadlock-freedom but violates starvation-freedom: the system progresses, one victim doesn't. Priority inversion is a starvation of a *specific* high-priority thread caused by the scheduling policy itself. Keep that ladder in your head — it's the spine of everything below.

There's a sharp formal reason these bugs resist the tools that catch races and deadlocks. A *safety* property has the form "this bad state is never reached," and it can be *monitored*: you instrument the code with an assertion, and the first time the bad state appears, the assertion fires and you have your culprit, with a stack trace and a core dump. A *liveness* property has the form "this good state is *eventually* reached," and there is no finite observation that proves it was violated — at any moment you can only say "it hasn't happened *yet*," never "it will never happen." This is why a deadlock detector works (a wait-for cycle is a finite, observable bad state) but a "livelock detector" is fundamentally harder (it has to *infer* that no progress will *ever* happen from the fact that none has happened recently — a heuristic, never a proof). The practical upshot: you catch liveness bugs by watching a *progress metric over time* and alerting when it flatlines, not by asserting on a state. Build that metric in from day one; you cannot bolt it on during the incident.

Worth a brief word on terminology, because the literature is inconsistent and it trips people up. "Lockout" is an older synonym for starvation. "Indefinite postponement" is the same idea. "Bounded waiting" is the *property* a fair lock provides (there's a finite bound on how many times another thread can acquire the lock ahead of you). And "fairness" itself comes in flavors — *weak* fairness (a continuously-enabled action eventually runs) and *strong* fairness (an action enabled infinitely often eventually runs) — distinctions that matter in model checking but collapse, for our purposes, into "every waiter eventually gets served." When a textbook says a scheduler is "starvation-free," it means it satisfies one of these fairness conditions; when it says "deadlock-free but not starvation-free," it means the system as a whole always moves but an individual might not.

#### Worked example: reading a profiler trace

Suppose you sample 10,000 stack traces from a stuck service over ten seconds and bucket them. Three signatures tell three different stories:

- **Deadlock:** ~100% of samples sit in `futex_wait` / `park` / `pthread_cond_wait` — kernel wait states. CPU is near 0%. Threads are *off* the run queue. The wait-for graph has a cycle. (That's the subject of the sibling post, [deadlock: the four conditions and how to break them](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them).)
- **Livelock:** ~100% of samples sit in *userspace* — a CAS loop, a `tryLock`-then-`unlock`, a back-off sleep, a retry. CPU is pinned near 100%. Threads are *on* the run queue, constantly. The transaction counter is flat.
- **Starvation:** the *aggregate* counter climbs fine, throughput looks healthy — but if you tag completions by thread or by request class, one bucket is empty. One writer hasn't run in 40 seconds while readers complete millions of ops.

Same symptom to a naive alert ("requests aren't completing"), three completely different root causes and three completely different fixes. The trace tells you which.

## Livelock: busy, polite, and stuck forever

The canonical mental model is two people meeting in a narrow hallway. You step left to let them pass; they step left too. You step right; they step right. You step left; so do they. Neither of you is frozen — you're both moving, energetically, cooperatively — and yet neither gets past. That is livelock: threads that keep *responding to each other* and changing state in lockstep, doing work the whole time, never reaching a configuration where one of them just goes.

![A timeline of two threads repeatedly backing off and retrying in perfect lockstep so neither ever proceeds](/imgs/blogs/livelock-starvation-and-priority-inversion-2.png)

The cruel irony is that livelock is usually born from a *good-faith attempt to avoid deadlock*. Here's the trap. You have the classic two-lock deadlock: thread 1 holds lock A and wants B, thread 2 holds B and wants A — circular wait, frozen. Someone reads about deadlock and learns the wisdom "don't hold a lock while blocking on another." So they rewrite it: instead of blocking, *try* to acquire the second lock; if it fails, release the first lock, back off, and retry the whole thing. No thread ever holds one lock while waiting on another, so the circular wait is broken, so — surely — no deadlock. And they're right: there's no deadlock. They've traded it for a livelock.

The mechanism is symmetry. If both threads run the *exact same* avoidance routine and the scheduler keeps them in step, they will both grab their first lock, both fail to get the second, both release, both back off the same amount, both retry — and arrive back at the identical starting configuration. Forever. The retry that was supposed to dissolve the conflict instead *reproduces* it on every iteration. Nothing is held across a block, so no detector flags a deadlock, but no philosopher ever eats.

Let me make it concrete. Here is a livelock in Go — a deliberately symmetric "polite" account-transfer that tries to avoid the two-lock deadlock and instead spins:

```go
package main

import (
	"sync"
	"time"
)

type Account struct {
	mu      sync.Mutex
	balance int64
}

// transferPolite tries to be deadlock-safe by never holding one lock
// while blocking on another. It tryLocks the second account, and if
// that fails it releases the first and retries. Symmetric -> livelock.
func transferPolite(from, to *Account, amt int64, done chan<- bool) {
	for {
		from.mu.Lock()
		if to.mu.TryLock() { // Go 1.18+: non-blocking acquire
			from.balance -= amt
			to.balance += amt
			to.mu.Unlock()
			from.mu.Unlock()
			done <- true
			return
		}
		// Couldn't get 'to'. Be polite: release and retry.
		from.mu.Unlock()
		// No back-off, or a fixed back-off -> both threads stay in lockstep.
		time.Sleep(0)
	}
}

func main() {
	a, b := &Account{balance: 100}, &Account{balance: 100}
	done := make(chan bool)
	// Two transfers in OPPOSITE directions -> the classic conflict.
	go transferPolite(a, b, 10, done)
	go transferPolite(b, a, 10, done)
	<-done
	<-done
}
```

Run this and you'll frequently watch it spin: both goroutines repeatedly grab their first lock, `TryLock` the second, fail, release, and try again — both at once, neither winning. CPU pegged, `done` never sent. There's no deadlock the runtime can detect (the `-race` detector won't help; nothing is racing), and Go's own deadlock detector (which fires only when *all* goroutines are blocked) stays silent because the goroutines are runnable the whole time.

The fix is to **break the symmetry**. There are three honest ways:

1. **Randomized back-off.** Each thread sleeps a *random* amount before retrying. The randomness makes it overwhelmingly likely that on some iteration one thread acquires both locks while the other is still sleeping. This is exactly how Ethernet's CSMA/CD and TCP handle collisions, and why exponential back-off has a *random* jitter term — deterministic back-off would just resynchronize the collision.
2. **Asymmetry / a tie-breaker.** Give the threads a fixed total order and have them always contend in that order, so they can't both be "first." This is lock ordering, the real deadlock fix, and it makes the retry loop unnecessary.
3. **An arbitrator.** Route all acquisition through a single coordinator that hands out the right to proceed, so two threads physically cannot both be in the contended region.

Here is the randomized-back-off fix, the smallest change that turns the livelock into reliable progress:

```go
import "math/rand"

func transferPoliteFixed(from, to *Account, amt int64, done chan<- bool) {
	backoff := time.Microsecond
	for {
		from.mu.Lock()
		if to.mu.TryLock() {
			from.balance -= amt
			to.balance += amt
			to.mu.Unlock()
			from.mu.Unlock()
			done <- true
			return
		}
		from.mu.Unlock()
		// RANDOMIZED back-off breaks lockstep. Jitter is the whole point.
		jitter := time.Duration(rand.Int63n(int64(backoff)))
		time.Sleep(backoff + jitter)
		if backoff < 256*time.Microsecond {
			backoff *= 2 // exponential ceiling caps the wasted time
		}
	}
}
```

The same livelock and the same cure show up in Java, where the idiom is `ReentrantLock.tryLock` with a timeout. The bug looks like this — note that without jitter, two threads on a lightly loaded box will happily ping-pong:

```java
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.ThreadLocalRandom;

class Account {
    final ReentrantLock lock = new ReentrantLock();
    long balance;
    Account(long b) { balance = b; }
}

static void transfer(Account from, Account to, long amt) throws InterruptedException {
    while (true) {
        if (from.lock.tryLock()) {
            try {
                if (to.lock.tryLock()) {
                    try {
                        from.balance -= amt;
                        to.balance += amt;
                        return; // success
                    } finally {
                        to.lock.unlock();
                    }
                }
            } finally {
                from.lock.unlock();
            }
        }
        // RANDOMIZED sleep is what makes this terminate. Remove it -> livelock.
        Thread.sleep(ThreadLocalRandom.current().nextInt(1, 5));
    }
}
```

Two things to notice. First, the `finally` blocks matter for *correctness* (a lock must be released on every path), but they're orthogonal to the livelock — the livelock is about the *retry policy*, not the release. Second, the randomized sleep is not a "make it work" hack; it is the actual mechanism that decorrelates the two threads' schedules. If you removed it and instead trusted "the scheduler will surely interleave them differently," you'd be betting against the scheduler, which on a quiet machine loves to run two identical threads in near-lockstep.

Rust shows the *asymmetry* fix particularly clearly, because Rust's ownership model pushes you toward expressing the tie-breaker structurally. The idiomatic deadlock-and-livelock-free pattern is to lock the two mutexes in a fixed order derived from a stable identity — here, the memory address of the lock, which gives a total order for free:

```rust
use std::sync::{Arc, Mutex};

struct Account { balance: Mutex<i64> }

// Deterministic asymmetry: always lock the lower-addressed mutex first.
// A fixed total order means two transfers can never form a cycle AND
// never enter a symmetric release-retry dance -- no deadlock, no livelock.
fn transfer(from: &Arc<Account>, to: &Arc<Account>, amt: i64) {
    let (first, second) = if Arc::as_ptr(from) < Arc::as_ptr(to) {
        (from, to)
    } else {
        (to, from)
    };
    let mut a = first.balance.lock().unwrap();   // ordered acquire
    let mut b = second.balance.lock().unwrap();  // never blocks while
    // ... apply the transfer using a and b ...   // holding out of order
    if Arc::ptr_eq(first, from) {
        *a -= amt; *b += amt;
    } else {
        *b -= amt; *a += amt;
    }
}                                                 // both released on scope exit
```

This version has *no retry loop at all* — and therefore no possibility of livelock — precisely because the ordering removes the conflict instead of managing it. That's the deeper lesson: a retry loop is a sign you couldn't establish an order; if you *can* establish one (an address, an ID, a fixed rank), prefer it, because it converts a probabilistic-progress design into a guaranteed-progress one. Randomized back-off is the fallback for when no cheap total order exists.

It is worth being precise about *why* the un-jittered version actually loops rather than "usually working out." Two identical threads released onto two cores at nearly the same instant execute the same instruction stream. The lock acquisition, the failing `TryLock`, the release, and the back-off all take very similar wall-clock time on identical code, so the two threads stay phase-locked: each iteration ends with both threads in the same relative state they started in. This is a *resonance* — the system has a stable oscillation it can fall into. Jitter detunes the resonance: by making the back-off durations differ run-to-run, it guarantees the phases drift apart, and once they drift, one thread reaches the "grab both locks" window while the other is asleep, and the conflict resolves. The mechanism is identical to why two metronomes on a rigid table synchronize while two on a soft surface don't — coupling plus symmetry produces lockstep, and breaking either one breaks the lock.

The honest engineering lesson: **retry-based deadlock avoidance trades a *certain* freeze for a *probabilistic* one.** Randomized back-off makes the probability of an infinite livelock zero in the limit, but the *expected* number of retries — and thus wasted CPU — is nonzero and grows with contention. If you find yourself in this pattern under heavy contention, that's a signal the right fix is lock ordering or a coarser-grained lock, not a cleverer back-off curve. We'll see lock ordering done properly in the dining-philosophers section.

#### Worked example: measuring livelock progress

Take the Go program above and instrument it: count completed transfers per 100 ms. With the un-jittered "polite" version on a 2-core VM, I've watched runs where the completion count stays at **0 for seconds at a time** while both cores read ~100% — the textbook livelock signature. Swap in the randomized-back-off version and completions start immediately; the cost shows up only as a handful of wasted `TryLock` attempts per success (typically 1–3 retries under light contention, more under heavy). The point isn't the exact numbers — they're nondeterministic and platform-dependent — it's the *shape*: livelock is a flat line of zero progress at full CPU, and the fix turns it into a rising line at slightly-less-than-full efficiency. Measure the *progress counter*, not CPU; CPU lies.

## Starvation: when one thread never gets a turn

Starvation is the quieter cousin. The system *is* making progress — the aggregate throughput counter climbs, dashboards look fine — but one particular thread, or one class of request, is perpetually denied the resource it needs because others keep grabbing it first. There's no lockstep, no symmetry, no busy-spin necessarily. Just an unlucky thread that the allocation policy never gets around to.

The simplest source is an **unfair lock**. Most production mutexes are, by default, *not* FIFO — they're "barging" locks. When a lock is released, the runtime doesn't necessarily hand it to the thread that's been waiting longest; it often just wakes a waiter and *also* lets any thread that happens to be running right now try to grab it. On a hot lock, the thread that just released it is still on-CPU, cache-warm, and frequently re-acquires it before the woken waiter is even scheduled. That's great for throughput (no context switch, no cache miss) and terrible for the waiter, which can sit on the queue for a very long time while a tight loop in another thread monopolizes the lock. Barging is the default precisely because it's faster; fairness is something you opt into.

![A comparison table of four starvation sources showing which thread starves and the fairness fix for each](/imgs/blogs/livelock-starvation-and-priority-inversion-5.png)

The second classic source — and one you'll hit constantly in real systems — is **reader starvation of a writer** in a readers-writer lock. (The sibling post [readers-writer locks and lock granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity) goes deep on the lock itself; here we care about its *liveness*.) The whole point of a read-write lock is to let many readers hold the lock simultaneously, since concurrent reads don't conflict. But a *reader-preferring* implementation has a fatal liveness flaw: as long as *any* reader holds the lock, new readers are admitted immediately — and if reads arrive faster than they drain, there is never a moment with zero readers. A writer that asks for exclusive access waits for that moment, which never comes. The writer starves, indefinitely, while the data it wants to update never gets updated. Under a steady read flood the writer's wait is *unbounded*.

The fix is a **writer-preferring** (or at least fair) RW lock: once a writer is waiting, *block new readers from entering*, let the in-flight readers drain, then let the writer in. That bounds the writer's wait to the duration of the currently-active reads. Go's `sync.RWMutex` documents exactly this behavior — a blocked `Lock` (writer) call prevents new `RLock` (reader) calls from succeeding, specifically "to ensure that the lock eventually becomes available; a blocked Lock call excludes new readers from acquiring the lock." Java's `ReentrantReadWriteLock` gives you the choice: the default is nonfair (higher throughput, reader-starvation-of-writer possible), and `new ReentrantReadWriteLock(true)` is fair (FIFO-ish, bounded waits, lower throughput).

Here's the writer-starvation bug and the one-flag fix in Java:

```java
import java.util.concurrent.locks.ReentrantReadWriteLock;

// BUG-PRONE: default (nonfair). Under a heavy read flood, a writer can
// wait a very long time -- new readers keep barging in ahead of it.
ReentrantReadWriteLock unfair = new ReentrantReadWriteLock();

// FIX: fair mode. Once a writer is queued, new readers wait behind it,
// so the writer's wait is bounded by the in-flight readers' duration.
ReentrantReadWriteLock fair = new ReentrantReadWriteLock(true);

// reader hot loop (many threads):
unfair.readLock().lock();
try { /* read shared state */ }
finally { unfair.readLock().unlock(); }

// the writer that may starve under 'unfair':
fair.writeLock().lock();          // bounded wait under 'fair'
try { /* update shared state */ }
finally { fair.writeLock().unlock(); }
```

The third source is the most fundamental: **priority-based scheduling**. If your scheduler always runs the highest-priority runnable thread, then a low-priority thread on a busy machine may *never* run — there's always something more important. A pure priority scheduler offers deadlock-freedom (the top priority always progresses) but flagrantly violates starvation-freedom. Real operating systems patch this with **aging**: a thread's effective priority is boosted the longer it waits, so a long-ignored low-priority thread eventually floats up high enough to be scheduled. Aging is the scheduler's built-in anti-starvation valve. (It is *also*, as we'll see, half of the priority-inheritance idea.)

A fourth, subtler source is the **lock convoy**. When a popular lock is held just long enough that arriving threads pile up behind it, every thread ends up doing a full sleep/wake cycle for the lock instead of acquiring it cheaply — the queue "convoys" forward in slow lockstep, throughput collapses, and threads that join the back of the convoy late can wait disproportionately long. The cure there isn't fairness per se; it's *shorter critical sections* (hold the lock for less time) or *less contention* (shard the lock), so the convoy never forms.

It's worth slowing down on the *mechanism* of barging, because it's the part most engineers find surprising. When a thread calls `unlock()`, two things could happen to the lock's owner field: the runtime could atomically transfer ownership to the longest-waiting parked thread (a *handoff*), or it could simply mark the lock free and let whoever wins the next race take it (a *release*). The release path is dramatically cheaper — no thread needs to be woken, no scheduler involvement, the data the lock protects is still hot in the releasing core's cache — so high-performance mutexes default to it. But "whoever wins the next race" is biased: the thread that just released the lock is still running on its core, cache-warm, with the lock variable in its L1; a parked waiter has to be woken by the scheduler (microseconds away) before it can even attempt the acquire. So in a tight loop, the releasing thread re-acquires its own lock again and again before any waiter gets a look in. That bias *is* the throughput win of barging, and it is *exactly* the starvation risk — the same mechanism produces both. You cannot get the throughput without accepting the bias, which is why fairness is opt-in: the runtime won't pay the handoff cost unless you tell it that bounded waiting matters more than raw speed.

A related and important nuance: a *spin*-then-park lock (most modern adaptive mutexes spin briefly before parking, on the bet that the holder will release soon) amplifies the barging bias, because a spinning waiter is *also* on-CPU and racing — but it's racing from a cold cache against a holder that owns the line. Adaptive spinning is great for throughput on short critical sections and makes starvation *more* likely, not less, which is one more reason the fairness decision is independent of (and often opposed to) the throughput-tuning decisions you make elsewhere in the lock.

#### Worked example: the writer that never writes

Set up a `ReentrantReadWriteLock` in nonfair mode. Spawn 16 reader threads, each grabbing the read lock, sleeping 1 ms (simulating a read), releasing, and looping with no pause. Spawn 1 writer that tries to grab the write lock once and time how long it waits. With 16 readers cycling every ~1 ms, there is essentially never a gap with zero readers, so the writer can wait *seconds* — I've measured waits that grow without an obvious ceiling as you add readers. Flip the constructor argument to `true` (fair), rerun: the writer now acquires within roughly one read-duration of asking (a handful of ms), because once it's queued, new readers stop barging ahead. Same code, one boolean, the difference between "writes happen" and "writes never happen." That boolean costs you read throughput — which is the trade-off we quantify later.

## Priority inversion: the bug that froze a Mars rover

Priority inversion is starvation with a twist that makes it genuinely counterintuitive: a *high*-priority task is blocked, indirectly, by a *medium*-priority task that has nothing to do with the resource in question. It happens because of a transitive dependency through a lock.

The chain has three actors and one shared lock. A **low**-priority task L acquires lock M and enters its critical section. Before L finishes, a **high**-priority task H wakes up, preempts L (it's higher priority), runs, and then tries to acquire M — which L still holds. So H blocks on M, correctly waiting for L to release it. Now L should just finish quickly and release M, except: a **medium**-priority task X — which doesn't touch M at all — becomes runnable. The scheduler sees X (medium) is higher priority than L (low), so it preempts L and runs X. L is now frozen mid-critical-section, holding M, with no CPU to make progress. H is blocked on M waiting for L. And X can run as long as it likes. The result: H, the *highest*-priority task in the system, makes no progress because a *medium*-priority task is hogging the CPU that the *low*-priority lock-holder needs. The priorities are, effectively, inverted — H now waits on X, the exact opposite of what the priority scheme promised.

![A timeline showing low holding a lock, high blocking on it, then medium preempting low so high stays stuck](/imgs/blogs/livelock-starvation-and-priority-inversion-3.png)

The reason this is so dangerous is that the inversion can be *unbounded*: as long as medium-priority work keeps arriving, L never gets the CPU back, so it never releases M, so H is stuck arbitrarily long. In a real-time system with a watchdog timer expecting H to do something within a deadline, the watchdog fires. And that is, almost exactly, what happened to NASA's Mars Pathfinder in July 1997 — the most famous concurrency bug ever shipped to another planet. We'll tell that story in full in the case studies; the mechanism above *is* the bug.

The mechanism, stated as the precise failure condition: priority inversion occurs whenever a high-priority task must wait on a resource held by a lower-priority task, and *that* lower-priority task can itself be preempted by an unrelated intermediate-priority task. The first part (high waits on low) is *normal* and unavoidable — it's just blocking. The second part (an unrelated medium preempts the holder) is what makes the wait unbounded and turns a brief block into a deadline-blowing inversion.

### The fix: priority inheritance and priority ceilings

The cure is beautifully direct. The problem is that L holds a lock H needs but runs at L's low priority, so it gets preempted. So: **temporarily lend L the high priority while it holds the lock.** This is **priority inheritance**. When H blocks on M, the system *boosts* the priority of M's current holder (L) up to H's priority. Now L is no longer preemptible by medium-priority X — L runs, finishes its critical section, releases M, and *immediately drops back* to its original low priority. H then acquires M and proceeds. The medium task X never gets to interpose, so the inversion is bounded by the length of L's critical section, not by the arrival rate of medium work.

![A before-and-after view contrasting plain locks where the high task starves with priority inheritance where the boosted low task finishes and the high task proceeds](/imgs/blogs/livelock-starvation-and-priority-inversion-4.png)

The mechanism, made rigorous: with priority inheritance, the maximum time H can be blocked by lower-priority tasks is bounded by the sum of the critical-section durations of the locks H can transitively need — a *bounded* blocking term you can actually put in a schedulability analysis. Without inheritance, that term is unbounded (it depends on medium-priority arrivals), so the schedule is simply not analyzable. Inheritance turns "could be forever" into "at most the time the holder needs to finish," which is the entire reason real-time kernels implement it.

A close relative is the **priority ceiling protocol**. Each lock is statically assigned a *ceiling* equal to the highest priority of any task that will ever acquire it. When a task acquires the lock, it is immediately raised to that ceiling (or, in the "immediate ceiling" variant, even before contention occurs). This has two extra benefits over plain inheritance: it prevents *chained* blocking (a task can be blocked by at most one lower-priority critical section, not a stack of them) and it prevents *deadlock* among the ceiling-protected locks entirely, because a task can only acquire a lock if its priority is strictly higher than the ceilings of all locks currently held by *other* tasks. Ceilings need the static priority map up front; inheritance is dynamic and needs nothing declared. Most general-purpose OSes (Linux's `PTHREAD_PRIO_INHERIT` mutexes, the real-time patches, VxWorks, QNX) offer inheritance; the ceiling protocol shows up more in hard-real-time settings where you can enumerate every task and lock.

Here's how you actually *ask* for inheritance — it's a one-line mutex attribute in POSIX/C, and it's the fix NASA uploaded to Pathfinder:

```c
#include <pthread.h>

pthread_mutex_t lock;
pthread_mutexattr_t attr;

void init_inheriting_mutex(void) {
    pthread_mutexattr_init(&attr);
    /* THE FIX: the holder inherits the priority of any higher-priority
       waiter, so a medium task can't preempt it mid-critical-section. */
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    pthread_mutex_init(&lock, &attr);
}

/* Or, for the priority-ceiling protocol instead of inheritance: */
void init_ceiling_mutex(int ceiling_prio) {
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_PROTECT);
    pthread_mutexattr_setprioceiling(&attr, ceiling_prio);
    pthread_mutex_init(&lock, &attr);
}
```

In Rust on a real-time target you'd reach for the same OS primitive (RTIC's resource model uses the *immediate ceiling* protocol at compile time, so it's enforced statically and inversions are impossible by construction — a particularly elegant version of the idea). The language differs; the mechanism — boost the holder so medium can't interpose — is identical everywhere.

One trap worth naming: **priority inversion can hide inside language runtimes you don't think of as real-time.** Garbage-collected runtimes, async executors, and even plain thread pools can manufacture inversions when a low-priority task holds a lock or a worker slot that a latency-critical task needs. The defense in those worlds is usually structural — don't share a lock across priority bands, give latency-critical work its own dedicated executor, keep critical sections tiny — rather than reaching for `PTHREAD_PRIO_INHERIT`. The inversion mechanism is universal even when the inheritance fix isn't available.

There's a second, subtler hazard with inheritance worth understanding: **inheritance must be transitive, and the boost must follow the dependency chain.** Suppose high task H blocks on lock M held by low task L, and L is *itself* blocked on lock N held by an even-lower task K. Boosting only L doesn't help — L can't run, it's waiting on K. A correct inheritance implementation propagates the boost *transitively*: K inherits H's priority too, so K runs, releases N, which unblocks L (still boosted), which releases M, which unblocks H. If the OS only boosts the immediate holder and not the chain, you get a *transitive* inversion that inheritance fails to cure. This is one reason the priority-ceiling protocol is attractive in hard-real-time work: by raising a task to a lock's ceiling the *instant* it acquires the lock — before any contention — it bounds the blocking to a single critical section and sidesteps the whole transitive-chain problem. Inheritance reacts to contention; ceilings prevent the configuration that would require a long chain of reactions.

A worked sketch of the bound makes the value concrete. Say H can transitively need locks $M_1$ and $M_2$, with worst-case critical-section durations $c_1 = 50\,\mu s$ and $c_2 = 30\,\mu s$. Under priority inheritance, the worst-case time H is blocked by *lower*-priority tasks is bounded by $c_1 + c_2 = 80\,\mu s$ — a finite number you can plug into a schedulability test. Under the priority-ceiling protocol, the bound tightens to a *single* critical section, $\max(c_1, c_2) = 50\,\mu s$, because ceilings prevent the chained blocking that lets a task be held up by more than one lower-priority section. Without either protocol, the bound is $\infty$ — it depends on how much medium-priority work arrives, which the schedule has no control over. That step from $\infty$ to a finite, computable number is the entire reason real-time systems mandate one of these protocols: an unbounded blocking term makes the system simply not analyzable, and "not analyzable" is "not certifiable" for anything safety-critical.

## Fairness: bounding the wait, and what it costs

Everything above points at one lever: **fairness**. A fair lock or a fair scheduler guarantees that every waiter eventually gets served, usually by serving them in arrival order (FIFO). Fairness is the direct antidote to starvation — it converts an *unbounded* worst-case wait into a *bounded* one. So why isn't everything fair by default? Because fairness has a real, measurable throughput cost, and on the hot path that cost often dwarfs the benefit.

![A comparison table of fair FIFO locks, barging locks, and fair schedulers across fairness, throughput, and when to use each](/imgs/blogs/livelock-starvation-and-priority-inversion-8.png)

The cost has a precise mechanism. A barging (unfair) lock can perform a **lock handoff for free**: when thread A releases and thread A *also* immediately wants the lock again (a tight loop), it just re-acquires it — no context switch, no cache miss, the lock and the data it protects are already hot in A's L1. A *fair* lock forbids this: on release it must hand the lock to the head of the wait queue, thread B, which means waking B (a context switch, ~1–5 µs), and B then touches the protected data cold (a cache miss, tens to hundreds of ns, possibly a cross-core cache-line transfer). So fairness *forces* a context switch and a cache miss on every contended handoff that barging would have avoided. Under high contention with short critical sections — exactly the workload where the lock is hot — that overhead per handoff can be larger than the critical section itself, and fair-lock throughput can fall to a *fraction* of unfair-lock throughput. Java's documentation is blunt about it: fair locks have "a much lower throughput" than nonfair ones, with the gain being only "a reduced variance in times to obtain locks and guaranteed lack of starvation." That's the trade in one sentence: fairness buys you bounded tail latency at the price of mean throughput.

The same trade-off lives in *schedulers*. A fair scheduler (Linux's CFS, the "Completely Fair Scheduler," or its EEVDF successor) tries to give every runnable thread an equal share of CPU over time, tracking each thread's accumulated runtime and always running the one that's furthest behind. This prevents the pure-priority starvation problem entirely — no thread is ignored forever — but it pays a context-switching tax: to keep shares balanced it preempts more often than a "run the hottest thread to completion" policy would, and every preemption is a cache-eviction and TLB-refill event. The fairness/throughput dial is the *same dial* at the lock level and the scheduler level: equalize who-gets-served and you pay in handoffs and switches; let the hot party keep going and you pay in tail latency and starvation risk.

Here is how you actually select fairness at the lock level. In Java it's a constructor flag; in most other runtimes the default mutex is unfair and you either accept that or build FIFO ordering yourself with a ticket/queue:

```java
import java.util.concurrent.locks.ReentrantLock;

// Unfair (default): barging allowed, highest throughput, starvation possible.
ReentrantLock fast = new ReentrantLock();          // == new ReentrantLock(false)

// Fair: FIFO, bounded waits, no starvation, LOWER throughput.
ReentrantLock fairLock = new ReentrantLock(true);

void critical(ReentrantLock lock) {
    lock.lock();
    try {
        // ... short critical section ...
    } finally {
        lock.unlock();
    }
}
```

A useful middle path that many systems land on: **mostly unfair, occasionally fair**. The lock barges for throughput most of the time, but tracks how long each waiter has been queued and, past a threshold, forces a fair handoff to drain the oldest waiter. The Linux kernel's queued spinlocks and Go's `sync.Mutex` both do versions of this — Go's mutex switches a waiter into a "starvation mode" if it's been waiting longer than 1 ms, handing the lock directly to that waiter (FIFO) until the backlog clears, then reverting to the fast barging path. You get barging throughput in the common case and a *bounded* worst case in the tail. It's the pragmatic answer to "fair or fast?": be fast until someone is about to starve, then be fair just long enough to rescue them.

#### Worked example: the fairness throughput tax

Take a single hot `ReentrantLock`, 8 threads, each acquiring it, incrementing a counter (a ~10 ns critical section), releasing, in a tight loop. Measure aggregate increments per second, unfair vs fair. In this regime — many threads, tiny critical section, maximal contention — the unfair lock wins big: the barging thread keeps re-acquiring its own hot lock, so most acquisitions cost ~25 ns with no context switch, and you might see millions of ops/sec. The fair lock forces a handoff to the queue head on essentially every release, so each acquisition eats a wakeup (~1–5 µs) plus a cold cache touch; aggregate throughput can drop by an order of magnitude. *But* — and this is the point — measure the *distribution*, not just the mean. Under the unfair lock, some threads' max wait is enormous (one thread might hog the lock for stretches while another barely runs); the variance is huge and one thread can effectively starve. Under the fair lock, every thread's wait is tightly bounded around the same value — low variance, no starvation. So the honest summary: fair locks trade ~2–10× mean throughput (workload-dependent; the gap shrinks as critical sections grow) for dramatically lower wait *variance* and a hard starvation guarantee. Whether that's a good trade depends entirely on whether your SLO is about *mean throughput* or *tail latency*.

## Dining philosophers: the canonical study that ties it all together

Dijkstra's dining philosophers (1965, as a classroom problem; later dressed up by Hoare) is the single best vehicle for seeing deadlock, livelock, and starvation in one picture — and seeing how to defeat all three at once. The setup: five philosophers sit around a table, alternating between thinking and eating. Between each pair of neighbors is a single fork. To eat, a philosopher needs *both* the fork on their left and the fork on their right. There are five philosophers and only five forks, and adjacent philosophers share a fork, so they contend.

![An acyclic graph of three philosophers each reaching for two shared forks with the contention noted in the node labels](/imgs/blogs/livelock-starvation-and-priority-inversion-6.png)

The naive solution is the trap that contains every failure mode we've discussed — here it is in Go, the version every introductory course writes and every introductory course's program eventually hangs on:

```go
func dineNaive(id int, forks []*sync.Mutex, n int) {
	left, right := id, (id+1)%n
	for {
		// think ...
		forks[left].Lock()   // pick up LEFT fork
		forks[right].Lock()  // pick up RIGHT fork -- may wait forever
		// eat ...
		forks[right].Unlock()
		forks[left].Unlock()
	}
}
```

Walk the failure modes:

- **Deadlock.** Suppose every philosopher gets hungry at once and each picks up their left fork. Now all five forks are held as "left" forks, every philosopher is waiting for their right fork, and *no one* will ever put a fork down because no one is eating. That's a textbook circular wait — philosopher 1 waits on philosopher 2's fork, who waits on 3's, ... who waits on 1's. The four Coffman conditions all hold; the table is frozen. (We never draw that wait-for cycle as a cyclic figure — it's stated in the node labels — because a wait-for cycle isn't a valid acyclic dataflow graph.)
- **Livelock.** Try to fix the deadlock the naive way: "if I can't get my right fork in a moment, I'll put my left fork back down and try again." If all five do this in lockstep — pick up left, fail to get right, put down left, wait, repeat — you get the hallway dance at table scale. Forks clatter up and down forever; no one eats. Busy, polite, stuck.
- **Starvation.** Even a solution that avoids deadlock and livelock can starve an individual. If philosophers 2 and 4 keep alternating their eating so that philosopher 3's two forks are *never both free at the same time*, philosopher 3 can sit hungry forever while the table as a whole makes progress. The system is live; philosopher 3 is not.

So a *correct* dining-philosophers solution must defeat all three: no circular wait (no deadlock), no symmetric retry (no livelock), and a guarantee that *every* philosopher eats (no starvation). There are three classic families, and they map cleanly onto the fixes we've already built.

![A tree of dining-philosopher solutions grouping resource ordering and limited diners under breaking the circular wait, and arbitrator and chandy-misra under coordinating access](/imgs/blogs/livelock-starvation-and-priority-inversion-7.png)

### Solution 1: resource ordering (break the circular wait)

Number the forks 0 through 4. Require every philosopher to pick up their *lower-numbered* fork first and their higher-numbered fork second. This breaks the circular wait directly: a cycle requires that some philosopher acquires a high fork before a low one, but the rule forbids that, so no cycle can form. The asymmetry is concentrated in *one* philosopher — the one whose two forks happen to be in the "wrong" order relative to the others (for example, philosopher 4 sitting between fork 4 and fork 0 must take fork 0 first while everyone else takes their left first). That single inversion is what makes the whole thing deadlock-free. It's the same lock-ordering discipline that fixes the two-lock deadlock, applied around a ring.

```go
package main

import "sync"

// Each fork is a mutex. Lock ordering: always lock the lower-indexed
// fork first, then the higher. This breaks the circular wait, so no
// deadlock; and because we never release-and-retry, no livelock.
func dine(id int, forks []*sync.Mutex, n int) {
	left, right := id, (id+1)%n
	// Enforce a global order: acquire the lower-numbered fork first.
	first, second := left, right
	if first > second {
		first, second = second, first
	}
	for { // think/eat rounds
		forks[first].Lock()
		forks[second].Lock()
		// --- eat ---
		forks[second].Unlock()
		forks[first].Unlock()
		// --- think ---
	}
}

func main() {
	const n = 5
	forks := make([]*sync.Mutex, n)
	for i := range forks {
		forks[i] = &sync.Mutex{}
	}
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(id int) { defer wg.Done(); dine(id, forks, n) }(i)
	}
	wg.Wait()
}
```

This is deadlock-free and livelock-free (no retry loop at all). Its weakness is *fairness*: lock ordering alone doesn't promise every philosopher eats equally often — a fast neighbor pair could still starve the philosopher between them. If the underlying mutexes are fair (or use Go's starvation-mode rescue), that weakness is bounded; with purely barging mutexes you'd want to layer on aging.

The same lock-ordering solution in C++ is even cleaner because `std::scoped_lock` acquires *multiple* mutexes at once using a deadlock-avoidance algorithm, so you don't even have to sort by hand for the deadlock-freedom part:

```cpp
#include <mutex>
#include <vector>

void dine(int id, std::vector<std::mutex>& forks) {
    int n = forks.size();
    std::mutex& left  = forks[id];
    std::mutex& right = forks[(id + 1) % n];
    while (true) {
        // scoped_lock locks BOTH with a built-in deadlock-avoidance
        // ordering algorithm -- no circular wait, no manual sort needed.
        std::scoped_lock both(left, right);
        // --- eat (both forks held) ---
    }                                  // both released on scope exit
}
```

`std::scoped_lock` (and the older `std::lock`) internally uses a try-and-back-off algorithm to grab all the mutexes without deadlocking, which is the standard library doing the lock-ordering reasoning for you. It defeats deadlock; like solution 1 it doesn't by itself guarantee per-philosopher fairness.

### Solution 2: limit the diners (break hold-and-wait, structurally)

Allow at most four philosophers to *sit down* (attempt to eat) at once, enforced by a counting semaphore initialized to 4. With at most four reaching for forks among five forks, by the pigeonhole principle at least one philosopher always has both forks available, so the table can never deadlock. This attacks the deadlock at the "hold and wait" condition: it caps the number of partial acquisitions so a full circular wait is impossible.

```go
import "golang.org/x/sync/semaphore" // or a buffered channel as a semaphore

// seats permits at most n-1 philosophers to attempt to eat at once.
// With 4 of 5 seated, someone always has two free forks -> no deadlock.
var seats = make(chan struct{}, 4) // buffered channel as a counting semaphore

func dineLimited(id int, forks []*sync.Mutex, n int) {
	left, right := id, (id+1)%n
	for {
		seats <- struct{}{}      // acquire a seat (blocks if 4 are seated)
		forks[left].Lock()
		forks[right].Lock()
		// --- eat ---
		forks[right].Unlock()
		forks[left].Unlock()
		<-seats                  // release the seat
		// --- think ---
	}
}
```

Notice this version even lets everyone grab "left then right" safely, because the seat limit — not lock ordering — is what kills the deadlock. (This connects to the broader [semaphores, barriers, and latches](/blog/software-development/concurrency/semaphores-barriers-and-latches) toolkit.) It's deadlock- and livelock-free; fairness again depends on the semaphore's and mutexes' own queuing discipline.

### Solution 3: an arbitrator (coordinate access)

Introduce a waiter (an arbitrator) at the table. A philosopher must ask the waiter's permission before picking up forks; the waiter only grants permission when *both* of that philosopher's forks are free, and never grants two neighbors at once. Because the waiter serializes the *decision*, two adjacent philosophers can never both be mid-grab, so there's no circular wait and no clatter-up-clatter-down livelock. And crucially, the waiter can grant permission in a *fair order* (FIFO, or longest-waiting-first), which gives you the starvation-freedom the previous two solutions only get for free if their underlying primitives happen to be fair. The cost is throughput: the waiter is a central serialization point, so the arbitrator solution trades some concurrency for a clean fairness guarantee — the exact fairness/throughput dial from the last section, now visible at the algorithm level.

A fourth, more sophisticated family is the **Chandy-Misra** solution, which assigns each fork a "clean/dirty" state and a direction, letting philosophers exchange forks according to a set of rules that is provably deadlock-free *and* starvation-free *and* fully decentralized (no central waiter, so no single bottleneck). It's the most scalable answer and the one closest to how real distributed mutual-exclusion protocols work, at the cost of being the most intricate to implement. For most practical code, resource ordering or the seat limit is plenty; reach for Chandy-Misra when you genuinely need a decentralized, starvation-free protocol.

The Chandy-Misra mechanism is worth a sentence of detail because it shows how to get *fairness without a central serializer*. Each fork is either "clean" or "dirty," and it starts dirty. A philosopher who finishes eating marks both forks dirty. When a hungry philosopher needs a fork a neighbor holds, it sends a request; the neighbor gives up the fork *only if it's dirty*, and cleans it on the way out. A philosopher keeps a clean fork (it has "priority" on it) but always yields a dirty one. This single rule — yield-if-dirty, keep-if-clean — provably prevents both deadlock (the clean/dirty asymmetry breaks the symmetry that a circular wait needs) and starvation (after you eat, your forks become dirty, so a waiting neighbor is guaranteed to be able to take them next, which means no one can monopolize). It's a beautiful illustration that you can encode "whose turn is it next" into the *state of the shared resource itself* rather than into a referee — which is exactly the trick distributed systems use when there's no central coordinator to appeal to.

#### Worked example: the fully safe philosopher, all three properties

Let's nail down *why* solution 1 (resource ordering) plus *fair* forks gives you all three guarantees at once, by checking each property against its definition:

- **Deadlock-free?** A deadlock needs a circular wait. With strict lock ordering (lower fork first), suppose a cycle existed: philosopher A waits for a fork held by B, B for one held by C, ... back to A. Each "waits-for" edge means the waiter already holds its *lower* fork and wants its *higher* fork, which the next philosopher holds as *their* lower fork. Follow the fork numbers around the cycle and they must strictly increase at every step — but a cycle returns to its start, so the numbers would have to both strictly increase and return to where they began. Contradiction. No cycle can exist. Deadlock-free. ✓
- **Livelock-free?** There is no release-and-retry anywhere — a philosopher that holds its lower fork *blocks* (waits) on its higher fork rather than putting the lower one back. No symmetric back-off means no lockstep dance. Livelock-free. ✓
- **Starvation-free?** This is the part lock ordering alone doesn't give you — but if each fork-mutex is *fair* (FIFO), then a philosopher waiting for a fork is guaranteed to get it after at most the finite number of philosophers ahead of it in that fork's queue have been served. Bounded wait per fork ⇒ bounded wait to eat ⇒ no starvation. Starvation-free. ✓ (With *unfair* forks you'd add aging or fall back to the arbitrator to recover this property.)

That's the whole point of the dining philosophers as a teaching device: it forces you to verify all three liveness properties separately, because a fix for one (retry, to dodge deadlock) can manufacture another (livelock), and a fix for both (lock ordering) can still leave the third (starvation) on the table unless you choose fair primitives.

## Measured behavior: progress and fairness under load

Theory says fair primitives bound the wait and pay throughput; the only way to *trust* that is to measure it on your own hardware, because the magnitudes swing wildly with critical-section length, thread count, and the platform's memory model. Here is how I measure these honestly, and the shapes you should expect.

**Measure progress, not utilization.** The single most important instrument for liveness bugs is a *completed-work counter* sampled over time — transactions per 100 ms, not CPU%. Livelock is invisible to CPU monitoring (it shows 100%, which looks like health) and obvious to a progress counter (flat at zero). Starvation is invisible to aggregate throughput and obvious to a *per-class* progress counter (one bucket flat while others climb). If your only metric is utilization, you are blind to exactly the bugs in this post.

**For fairness, measure the distribution, not the mean.** A fair lock and an unfair lock can have *similar mean* wait times while having wildly different *tails*. The whole value of fairness is in the tail — the p99 and the max wait — so a mean-only benchmark will systematically *undersell* fairness and *oversell* barging. Always report the wait-time distribution per thread: min, median, p99, max, and the spread across threads. The unfair lock's signature is a huge max and high cross-thread variance (some threads barely starved, one badly); the fair lock's is a tight band.

**Warm up, run many times, and name the platform.** Concurrency benchmarks are dominated by JIT warm-up (JVM), cache state, the OS scheduler's mood, and frequency scaling. Discard the first few seconds, run for tens of seconds, repeat the whole thing several times, and report the spread, not a single number. State the machine: core count, x86 (TSO, stronger ordering) versus ARM (weaker ordering, where uncontended atomics and the scheduler behave differently). A fairness result measured on a quiet 2-core laptop will not transfer to a loaded 64-core server — contention is superlinear, and the fair-lock handoff tax grows with the number of waiters.

**Beware the observer effect, and confounds from the scheduler.** Adding instrumentation to measure liveness can *change* the liveness. A `printf` or a metric increment inside the retry loop adds latency that perturbs the phase relationship between two livelocked threads — sometimes "fixing" the livelock by accident, which sends you chasing a ghost. Prefer sampling profilers (which perturb little) over heavy in-loop logging, and when you must instrument, instrument *outside* the hot path (a per-thread counter read by a separate sampler thread). Be equally wary of scheduler confounds: pinning threads to cores (`taskset`, `sched_setaffinity`) versus letting them float changes contention dramatically; running under a container with a CPU quota introduces *throttling* that looks exactly like starvation on the progress counter but is really cgroup CPU accounting. Before you blame a lock, confirm the machine isn't simply being CPU-throttled or migrated mid-critical-section by the scheduler — those produce the same flat-progress symptom for an entirely different reason, and no fairness flag will help them.

**Reproduce the failure deterministically before you trust the fix.** Liveness bugs are nondeterministic, so a single passing run proves nothing. The disciplined approach is to *increase the probability* of the failure until it's reliable — crank the thread count up to many times the core count (so the scheduler is forced to interleave), shrink the critical section to maximize handoff frequency, and add a controlled symmetry (start all threads from a barrier so they're phase-aligned). Once you can make the livelock or starvation appear in, say, 9 of 10 runs, *then* apply the fix and confirm it drops to 0 of 1,000 runs. A fix you can't demonstrate against a reliably-failing reproduction is a fix you don't actually understand.

Here is the behavior table I'd expect for a single hot lock under high contention with a tiny critical section, stated as *shapes and orders of magnitude* (your exact numbers will differ — measure):

| Lock / policy | Mean throughput (hot, tiny CS) | Worst-case wait | Starvation possible? | When it wins |
| --- | --- | --- | --- | --- |
| Unfair / barging mutex | Highest (barging reuses hot lock) | Unbounded (one thread can hog) | Yes | Throughput-bound, no tail SLO |
| Go-style mostly-unfair + rescue | Near-unfair in common case | Bounded (~1 ms rescue threshold) | No (rescued) | Default: fast but safe tail |
| Fair / FIFO mutex | Lowest (forced handoff each release) | Bounded (FIFO position) | No | Hard tail-latency SLO |
| Lock-free / CAS loop | High, scales better | Bounded per op (lock-free) | No (system-wide) | High core counts, short ops |

And the liveness-failure signatures, as a diagnostic table:

| Symptom on the box | Likely failure | Key metric to confirm | First fix to try |
| --- | --- | --- | --- |
| ~0% CPU, threads parked, frozen | Deadlock | Wait-for graph has a cycle | Lock ordering / `tryLock` + back-off |
| ~100% CPU, zero completions | Livelock | Progress counter flat, userspace stacks | Randomized back-off / asymmetry |
| Throughput fine, one class never completes | Starvation | Per-class progress counter | Fair lock / writer-preference / aging |
| High task misses deadline, holder preempted | Priority inversion | Holder runs below an unrelated medium task | Priority inheritance / ceiling |

The discipline is the same every time: name the *shared, contended* resource; pick the *progress* metric that exposes the specific failure; change one variable (the back-off, the fairness flag, the inheritance attribute); and confirm the metric moved. Don't trust CPU%. Don't trust the mean.

## Case studies / real-world

**Mars Pathfinder, July 1997 — priority inversion on another planet.** The textbook case, and a true story. The Pathfinder lander used the VxWorks real-time OS, and its tasks communicated through a shared `information bus` protected by a mutex. A low-priority meteorological data task (`ASI/MET`) would acquire the bus mutex to publish data. A high-priority `bus management` task also needed that mutex. The classic chain followed: the low-priority task took the lock, a high-priority task blocked on it — and meanwhile a *medium*-priority, long-running communications task (which didn't touch the bus at all) would preempt the low-priority lock-holder, keeping it off the CPU so it couldn't release the lock. The high-priority bus task stayed blocked past its deadline. A watchdog timer, noticing the high-priority task hadn't run, concluded the system was wedged and triggered a total system reset. On Mars. Repeatedly — the spacecraft kept rebooting. Engineers at JPL reproduced it on the ground replica, traced it to priority inversion, and — because VxWorks mutexes supported it — *uploaded a patch that turned on priority inheritance* for that mutex (flipping the `pthread`-style protocol flag to inherit). With inheritance, the low-priority holder was boosted whenever the high-priority task waited, so the medium task could no longer interpose, the lock was released promptly, and the resets stopped. The fix was, almost literally, one boolean flag — and it had to be radioed to another planet. (Glenn Reeves' first-hand account from the JPL team is the authoritative source, widely cited; Mike Jones' write-up popularized it.) Every line of this post's priority-inversion section is that story's mechanism.

**Reader-starved writers in production read-write locks.** Less famous but vastly more common: services that use a default (reader-preferring or nonfair) readers-writer lock around a hot configuration or cache structure, then find that under sustained read load a periodic *writer* — the config-reload, the cache-invalidation, the metrics-flush — stalls for seconds or never runs. The aggregate dashboard looks perfectly healthy (millions of reads/sec) while the one writer starves. This is why Go's `sync.RWMutex` was deliberately specified so a blocked writer *excludes new readers* (documented as ensuring "the lock eventually becomes available"), and why Java exposes a `fair` constructor on `ReentrantReadWriteLock`. The lesson engineers learn the hard way: a readers-writer lock is a *liveness* decision, not just a throughput optimization — choosing reader-preference silently signs you up for possible writer starvation under load.

**Lock convoys and the "fast" lock that wasn't.** A third recurring pattern (reported across Windows NT critical sections and various server runtimes over the years) is the lock convoy: a frequently-taken lock with a critical section just long enough that, once a queue forms, every thread pays a full sleep/wake cycle and throughput collapses to a crawl even though no single critical section is slow. The fix is rarely "make the lock fair" — it's "make the critical section shorter" or "shard the lock so the convoy never forms." It's the counterexample that keeps you honest: not every liveness problem is solved by adding fairness; sometimes the answer is *less* time under the lock, full stop.

## When to reach for this (and when not to)

These are progress *guarantees*, and each one costs something. Be decisive about when the cost is worth it.

**Reach for randomized back-off** when you genuinely have a retry-based protocol (optimistic concurrency, a `tryLock` loop, a CAS loop, network collision avoidance) and you've confirmed the threads can resynchronize. Don't reach for it as a band-aid over a deadlock you could fix with lock ordering — a deterministic fix (an ordering, a coarser lock) beats a probabilistic one (back-off) whenever you can establish it. Back-off is for when you *can't* impose an order cheaply.

**Reach for a fair lock** only when you have a real tail-latency or starvation requirement — an SLO on p99/p999, a request class that must not be starved, a writer that must eventually run. Do *not* make a hot lock fair "to be safe": you'll pay a large throughput tax (a forced context switch and cache miss per handoff) to solve a problem you may not have. Measure first; if no thread is actually starving and your tail latency is fine, the unfair lock is the right default. The "mostly unfair with a starvation-mode rescue" design (Go's mutex) is the best default for most code — fast in the common case, bounded in the tail — and you rarely need to override it.

**Reach for priority inheritance** whenever you mix priorities and share a lock across priority bands in a latency-sensitive system — real-time control, audio, a scheduler, anything with a deadline and a watchdog. If you can enumerate every task and lock statically, the priority-ceiling protocol is even stronger (it also prevents deadlock and chained blocking). But if you *can't* turn on inheritance (you're in a runtime that doesn't expose it), the better fix is structural: don't share a lock between a latency-critical task and a background one. Give them separate resources so the inversion can't form in the first place. Inheritance is the cure; *not sharing the lock across priorities* is the prevention.

**Reach for an arbitrator / limited-diners structure** when you need a *guaranteed* fair allocation and can tolerate a serialization point — a connection admission controller, a resource pool with bounded waits, a fair scheduler. When throughput is paramount and starvation is merely unlikely (not catastrophic), prefer the decentralized lock-ordering approach and accept that fairness is best-effort.

**Don't reach for any of this prematurely.** The most common mistake is solving a liveness problem you don't have: making locks fair, adding back-off, enabling inheritance, all on a system that was never starving anyone. Each of those has a throughput cost, and concurrency throughput is hard-won. The right sequence is: measure progress per class; if something is starving, identify *which* of the four sources it is (unfair lock, reader flood, priority, convoy); apply the *matching* fix; re-measure. Fairness is a tool, not a virtue — spend it where the tail actually hurts. For the broader "which model and which guarantees do I even need" decision, see the capstone, [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model). And when the contention is really *flow control* between producers and consumers rather than mutual exclusion, the right lever is often [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure), not a fairer lock.

## Key takeaways

1. **Liveness ≠ safety.** Safety (no corruption) and liveness (eventual progress) are independent. You can be perfectly race-free and still make zero progress. Deadlock detectors only catch one of the four failure modes here.
2. **CPU% lies; measure progress.** Livelock pins the CPU at 100% while completing nothing. The only metric that exposes it is a completed-work counter sampled over time. Starvation needs a *per-class* progress counter — aggregate throughput hides it.
3. **Livelock is usually a failed deadlock fix.** Symmetric retry-to-avoid-deadlock turns a certain freeze into an endless dance. Break the symmetry with randomized back-off, a fixed ordering, or an arbitrator — and prefer the deterministic fix when you can establish one.
4. **Starvation has four classic sources** — unfair (barging) locks, reader floods drowning a writer, pure priority scheduling, and lock convoys — each with a matching fix (FIFO/fair lock, writer-preference, aging, shorter critical sections). Diagnose the source before applying a fix.
5. **Priority inversion is starvation through a lock.** A high task blocked on a lock held by a low task that a medium task keeps preempting — unbounded until you boost the holder. Priority inheritance (dynamic) and the priority-ceiling protocol (static) are the cures; not sharing a lock across priority bands is the prevention.
6. **Fairness bounds the wait but taxes throughput.** A fair lock forces a context switch and a cache miss on every contended handoff that a barging lock would skip. Spend fairness only where you have a real tail-latency or starvation requirement; the "mostly unfair with a starvation rescue" design is the best default.
7. **Dining philosophers tests all three at once.** A correct solution must avoid deadlock (no circular wait), livelock (no symmetric retry), *and* starvation (every philosopher eats). Lock ordering plus fair forks gets all three; verify each property against its definition, separately.
8. **Apply the matching fix, then re-measure.** Name the contended resource, pick the metric that exposes the specific failure, change one variable, confirm it moved. Don't make everything fair "to be safe" — you'll pay throughput for a problem you may not have.

## Further reading

- E. W. Dijkstra, "Hierarchical Ordering of Sequential Processes" (1971) and the original dining-philosophers formulation — the source of resource-ordering as a deadlock fix.
- C. A. R. Hoare, *Communicating Sequential Processes* — the dressed-up dining-philosophers story and the CSP foundations behind the arbitrator approach.
- Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming* — chapters on progress conditions (deadlock-, starvation-, lock-, and wait-freedom) make the liveness hierarchy rigorous.
- Brian Goetz et al., *Java Concurrency in Practice* — the practical treatment of fairness, `ReentrantLock(true)`, and read-write lock starvation, with the throughput trade-off spelled out.
- Glenn Reeves (JPL), "What Really Happened on Mars Rover Pathfinder" and Mike Jones' write-up — the first-hand account of the 1997 priority-inversion bug and the priority-inheritance fix.
- L. Sha, R. Rajkumar, J. Lehoczky, "Priority Inheritance Protocols: An Approach to Real-Time Synchronization" (IEEE TC, 1990) — the formal treatment of inheritance and the priority-ceiling protocol with the bounded-blocking proof.
- K. M. Chandy and J. Misra, "The Drinking Philosophers Problem" (1984) — the decentralized, starvation-free fork-exchange solution.
- Within this series: [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [deadlock: the four conditions and how to break them](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them), [readers-writer locks and lock granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity), and the capstone [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

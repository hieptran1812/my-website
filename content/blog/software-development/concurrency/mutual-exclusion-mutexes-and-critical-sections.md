---
title: "Mutual Exclusion: Mutexes and Critical Sections"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make the racy counter correct with a lock, understand exactly what mutual exclusion buys you and what it costs, and learn to protect an invariant over data rather than lines of code."
tags:
  [
    "concurrency",
    "parallelism",
    "mutex",
    "locks",
    "critical-section",
    "thread-safety",
    "mutual-exclusion",
    "synchronization",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-1.png"
---

In the [previous post](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) we put `count++` under a microscope and watched it lose updates. Two threads, one shared counter, each incrementing two million times, and the final number came out short — `1,983,402` one run, `1,991,067` the next — because `count++` is not one instruction but three: load the old value into a register, add one, store the new value back. When two threads interleave those three steps, one of them reads a stale value, computes from it, and stomps the other's write. The increment that vanished is gone forever. That was the diagnosis. This post is the first real cure.

The cure has a name that is almost a definition: **mutual exclusion.** If we can guarantee that *at most one thread at a time* executes the load-modify-store sequence, then no thread can ever read a value another thread is in the middle of changing. The three steps become, in effect, indivisible — not because the hardware made them atomic, but because we drew a fence around them and posted a guard who lets exactly one thread through at a time. The fenced region is a **critical section**. The guard is a **mutex** — short for *mutual exclusion lock*. By the end of this post your counter will print exactly `2,000,000` every single time, and you will understand precisely why, what else the lock quietly fixed along the way, and what it pointedly did *not* fix.

![a racy unlocked counter that loses updates contrasted with a mutex guarded counter that reaches the exact total](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-1.png)

But "wrap it in a lock" is the kind of advice that is true and useless at the same time, like "just don't have bugs." The figure above shows the shape of the fix, and the hard part is everything the shape hides. A lock does not protect *lines of code* — it protects an **invariant over data**, and confusing the two is the source of a large fraction of real concurrency bugs. A lock guarantees mutual exclusion, and bundled with that, it also fixes **visibility** and **ordering** — the two memory-model hazards that the race post warned about — but it guarantees *nothing* about deadlock-freedom or fairness, and the moment you forget that, you ship a service that freezes at peak load. A lock has a cost: it **serializes**, and serialization is the enemy of throughput, so a lock held a microsecond too long, or held across a network call, can turn an eight-core machine into a one-core machine with extra steps. And there is a subtle trap waiting in the lock itself — try to acquire a lock your own thread already holds, with the wrong kind of lock, and you deadlock against yourself instantly.

So this post is not "here is `sync.Mutex`, go forth." It is the full mental model of the lock: what a critical section *is*, what mutual exclusion *means* formally, why the lock also buys you the memory-model guarantees for free, where to draw the boundary of the critical section, why you must never do I/O while holding one, and what serialization costs you in measured throughput. We will fix the counter in Go, Rust, Java, and C++ — because the *concept* is universal but the *idioms* diverge sharply, and seeing the same fix four ways is how you learn to recognize it in any language. We will reproduce a self-deadlock and explain reentrancy. And we will measure, honestly, what a lock costs. This is the foundation the rest of the mutual-exclusion track stands on: the next posts take the lock apart to show [how it is built from hardware](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) and how to [wait on a condition correctly](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly). Get the lock right here and those become refinements; get it wrong and they become a hall of mirrors.

## What a critical section actually is

Start with the precise definition, because the whole edifice rests on it. A **critical section** is a region of code that accesses shared mutable state and that must not be executed by more than one thread *at the same time*. That is the entire idea. It is defined not by syntax but by a safety requirement: "if two threads run this region concurrently, the program can produce a wrong result." Any such region is critical. Identifying your critical sections — finding every place where shared mutable state is read-then-written, or read in a way that assumes it will not change underneath you — is fully half the work of writing correct concurrent code. The mechanism that enforces "not at the same time" is secondary; the diagnosis comes first.

The canonical critical section is the read-modify-write we have been chasing:

```go
count = count + 1   // load count, add 1, store count
```

Three machine steps, and the bug lives in the gap between them. But critical sections are everywhere once you learn to see them. `balance -= amount` is one. `if key not in cache: cache[key] = compute(key)` is one — the check and the insert must be one indivisible decision, or two threads both find the key missing and both compute and both insert. `node.next = head; head = node` — pushing onto a linked list — is one. `total += item.price; count += 1` — updating two related fields that must stay consistent with each other — is one critical section spanning *both* writes, because the invariant "total equals the sum of `count` prices" must hold whenever another thread looks. That last example is the crucial one and we will return to it: a critical section can be larger than a single variable. It is defined by the **invariant** you need to hold, not by the number of lines.

Once you have identified a critical section, mutual exclusion is the property you need to enforce over it. Formally, label the section $C$. Mutual exclusion is the guarantee that at any instant in real time, the number of threads currently executing inside $C$ is at most one. Write $n(t)$ for the count of threads inside $C$ at time $t$; mutual exclusion is the invariant $n(t) \le 1$ for all $t$. That is it. It says nothing about *which* thread goes first, nothing about *how long* a thread waits, nothing about whether progress is even made — only that the inside of the section is never shared. Everything good about a lock, and everything frustrating about it, follows from how narrow that guarantee is.

### A critical section is a region in time, not a place in the file

Here is a subtlety that trips up beginners and occasionally seniors. A critical section is not a static block of source code that is "special." The same line of code might be a critical section in one program and perfectly safe in another, depending entirely on whether the data it touches is shared and mutable. `count = count + 1` on a thread-local variable that no other thread can reach is not a critical section at all — it needs no lock, ever. The very same line on a shared global *is* a critical section. The criticality is a property of the data's sharing, not of the syntax. This is why "thread-safe" is never a property of a function in isolation; it is a property of a function *together with the data it touches and who else touches that data*. We will make this concrete at the end with a taxonomy of what does and does not need a lock.

### Why the section is critical: the lost-update interleaving, one more time

Before we install the lock, fix firmly in mind *exactly* what goes wrong without it, because the lock's job is defined entirely by the failure it prevents. The line `count = count + 1` compiles to roughly three machine steps against a register `r`:

```c
load  r <- count   // step 1: read the current value into a register
add   r <- r + 1   // step 2: compute the new value
store count <- r   // step 3: write it back to memory
```

Now suppose `count` is `41` and two threads, T1 and T2, each run those three steps. If they run one after the other — T1's three steps, then T2's three steps — the result is `43`, correct. But the scheduler is free to interleave them, and one interleaving in particular loses an update. Trace it instant by instant:

#### Worked example: the exact interleaving that loses an update

| Step | Thread | Action | T1 register | T2 register | `count` in memory |
| ---- | ------ | ------ | ----------- | ----------- | ----------------- |
| 1    | T1     | load   | 41          | —           | 41                |
| 2    | T2     | load   | 41          | 41          | 41                |
| 3    | T1     | add    | 42          | 41          | 41                |
| 4    | T2     | add    | 42          | 42          | 41                |
| 5    | T1     | store  | 42          | 42          | **42**            |
| 6    | T2     | store  | 42          | 42          | **42**            |

Two increments ran. The final value is `42`, not `43`. One increment *vanished* — not delayed, not reordered, *gone*, because at step 2 T2 read `count` while T1 was mid-update, so T2 computed its new value from the *stale* `41` and at step 6 overwrote T1's `42` with its own `42`. The write-after-write clobbered an update. This is the lost update, and it requires *all three* of T2's steps to straddle T1's three steps. The fix is now stateable with total precision: **make the three steps indivisible with respect to other threads** — guarantee that once T1 begins step 1, no other thread touches `count` until T1 finishes step 3. That is mutual exclusion over the section spanning load-through-store. The lock is the mechanism; the requirement is "T2's load may not happen between T1's load and T1's store." When you read the rest of this post, hold this table in mind: every guarantee the lock provides exists to forbid the row where T2's `load` slips inside T1's load-modify-store.

## The mutex API across languages

A **mutex** is the object that enforces mutual exclusion over a critical section. Its interface is almost laughably small: two operations. `acquire` (also spelled `lock`) and `release` (also spelled `unlock`). The contract is: when a thread calls `acquire`, if the lock is free, the thread takes it and proceeds; if the lock is already held by another thread, the calling thread *blocks* — it is suspended by the runtime, consuming no CPU, parked on a wait queue — until the holder calls `release`, at which point one waiting thread is woken to take the lock. Between a thread's successful `acquire` and its `release`, that thread is said to *hold* the lock, and no other thread can be holding it. That single invariant — at most one holder — is what gives you $n(t) \le 1$ over the section bracketed by the calls.

The *concept* is identical everywhere. The *idiom* — specifically, how you guarantee that every `acquire` is paired with a `release` even when an exception or early return fires in the middle — is where languages diverge sharply, and the divergence is worth studying because it encodes decades of hard-won lessons about how locks get misused.

In **Go**, the lock is a plain value of type `sync.Mutex`, and the canonical idiom uses `defer` to schedule the unlock at function exit the instant you take the lock:

```go
package main

import (
	"fmt"
	"sync"
)

type Counter struct {
	mu    sync.Mutex
	value int
}

func (c *Counter) Inc() {
	c.mu.Lock()
	defer c.mu.Unlock() // runs when Inc returns, even on panic
	c.value++           // the critical section
}

func main() {
	c := &Counter{}
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1_000_000; j++ {
				c.Inc()
			}
		}()
	}
	wg.Wait()
	fmt.Println(c.value) // always 2000000
}
```

Run this with `go run -race` and the race detector is silent; run the unlocked version from the previous post under `-race` and it screams. The `defer c.mu.Unlock()` is the key idiom: pairing the unlock with the lock *at the point of acquisition* means you cannot forget it, and it fires on every exit path including a panic. Bundling the mutex and the data it guards into one struct — `Counter` holds both `mu` and `value` — is the Go convention that keeps the lock physically next to the thing it protects. (A caution Go's own documentation states: a `sync.Mutex` must not be copied after first use; copy the struct and you copy the lock's internal state, which corrupts it. Pass pointers.)

In **Rust**, the lock is not a separate object you remember to pair with the data — it *wraps* the data, and the type system makes it impossible to touch the data without holding the lock:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0u64));
    let mut handles = Vec::new();

    for _ in 0..2 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1_000_000 {
                let mut guard = counter.lock().unwrap(); // acquire
                *guard += 1;                              // critical section
                // guard drops here -> release, automatically
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
    println!("{}", *counter.lock().unwrap()); // 2000000
}
```

This is `Arc<Mutex<T>>`, the most important pattern in Rust concurrency, and it is worth slowing down on. `Mutex<T>` is a lock that *owns* the value of type `T` — here a `u64`. The only way to read or write that `u64` is to call `.lock()`, which returns a `MutexGuard`, a smart pointer through which you access the data. While you hold the guard, you hold the lock; when the guard goes out of scope, its destructor releases the lock. There is no `unlock` method to forget, because unlock is tied to scope exit. And because the data is *inside* the mutex, you literally cannot access it without going through `.lock()` — the compiler rejects any attempt. `Arc` is the atomic reference count that lets multiple threads share ownership of the same mutex. This is what Rust calls "fearless concurrency": the data-race bug from the last post is not a runtime failure here, it is a *compile error*. You cannot write the racy version. The price is that you must spell out the sharing explicitly, but the payoff is that the most common concurrency bug class is eliminated at compile time.

In **Java**, the original idiom is the `synchronized` keyword, which acquires a lock associated with an object (its "monitor") for the duration of a block, releasing it automatically when the block exits by any means:

```java
public class Counter {
    private long value = 0;
    private final Object lock = new Object();

    public void inc() {
        synchronized (lock) {   // acquire monitor of lock
            value++;            // critical section
        }                       // release on block exit, even on exception
    }

    public long get() {
        synchronized (lock) {   // reads need the lock too — see visibility
            return value;
        }
    }
}
```

The `synchronized` block is exception-safe by construction: the monitor is released whether the block exits normally, via a return, or via a thrown exception. There is no explicit unlock to forget. For more control, Java also offers the explicit `java.util.concurrent.locks.ReentrantLock`, where you call `lock()` and `unlock()` yourself — and where the idiom mandates `try`/`finally` so the unlock runs even if the body throws:

```java
import java.util.concurrent.locks.ReentrantLock;

public class Counter {
    private long value = 0;
    private final ReentrantLock lock = new ReentrantLock();

    public void inc() {
        lock.lock();
        try {
            value++;        // critical section
        } finally {
            lock.unlock();  // MUST be in finally, or an exception leaks the lock
        }
    }
}
```

That `try`/`finally` is non-negotiable. If you put `lock.unlock()` after the body without a `finally` and the body throws, the lock is never released and every other thread blocks forever — a self-inflicted deadlock. The `synchronized` keyword and `defer` and the Rust guard all exist precisely to make this failure impossible by tying release to scope.

In **C++**, the lock is `std::mutex`, and you almost never call `.lock()`/`.unlock()` directly — you use a RAII guard, `std::lock_guard` or `std::scoped_lock`, that acquires in its constructor and releases in its destructor:

```cpp
#include <mutex>
#include <thread>
#include <vector>
#include <iostream>

struct Counter {
    std::mutex mu;
    long value = 0;

    void inc() {
        std::lock_guard<std::mutex> guard(mu); // acquire
        ++value;                               // critical section
    }                                          // guard destructor -> release
};

int main() {
    Counter c;
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&c] {
            for (int j = 0; j < 1'000'000; ++j) c.inc();
        });
    }
    for (auto& t : threads) t.join();
    std::cout << c.value << "\n"; // 2000000
}
```

RAII — Resource Acquisition Is Initialization — is the same idea as Rust's guard and Go's `defer`: tie the release to the destruction of a stack object, and the compiler emits the unlock on every exit path automatically, including stack unwinding from an exception. Notice the pattern across all four languages: **none of the idiomatic forms make you remember the unlock.** Go's `defer`, Rust's drop, Java's `synchronized` block (and `try`/`finally` for the explicit lock), and C++'s RAII guard all exist for one reason — a forgotten unlock is a deadlock, and a deadlock on an exception path is the kind of bug that ships because it only fires when something else already went wrong. Learn the guard idiom in your language and use it by default.

Python's threading lock follows the same shape — `with lock:` is the context-manager idiom that acquires and releases — but Python's concurrency story is dominated by the Global Interpreter Lock, which changes the calculus entirely; that is its own subject, covered in the [GIL deep-dive](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs). We will use Python here only as illustration and link out for the details.

## A lock protects data and invariants, not lines of code

This is the single most important idea in the post, and it is the one most often gotten wrong. A lock does not protect a *block of code*. It protects an **invariant over data**. The mutex is just a token; what matters is the discipline that *every access to the shared data — every read and every write — happens while holding the same lock.* If even one access slips through without the lock, the protection is void, no matter how carefully every other access is guarded.

Think about what "the same lock" means. A mutex is a piece of memory. Two threads exclude each other *only if they contend for the same mutex object.* If thread A guards `balance` with `lockA` and thread B guards the same `balance` with a different `lockB`, then A and B do not exclude each other at all — they each take a lock no one else wants, and they both walk straight into the critical section together. The race is fully intact. The lock did nothing. This is a real and common bug: two code paths that both touch the same data but acquire different locks (or one path forgets the lock entirely). The fix is a *convention you must hold in your head and document*: "field `value` is guarded by `mu`; you may not touch `value` without holding `mu`." The compiler will not enforce this for you in most languages — Rust is the exception, because the data lives inside the mutex so there is no way to reach it without the lock. In Go, Java, and C++ it is a discipline, and disciplines are broken by the new hire who adds a fast-path read "just this once."

There is a useful discipline that makes the "same lock for every access" rule enforceable even in languages that do not check it for you: **document the lock-to-data binding next to the data, and never deviate.** Write a comment on the field — `// guarded by mu` — and treat any access that does not hold `mu` as a bug, full stop, including read-only accesses and including the "fast path" someone wants to add. Some toolchains can check this for you: C and C++ under Clang support `__attribute__((guarded_by(mu)))` (Clang's thread-safety analysis), which makes the compiler warn when you touch a guarded field without the lock; Go's `vet` and the `-race` detector catch many violations at test time; Rust enforces it structurally because the data lives inside the `Mutex<T>`. The point is that the binding "this lock guards this data" is the actual contract — the mutex object is meaningless without it — and the more of that contract you can make the machine check, the fewer 3 AM pages you earn.

The "invariant over data, not lines of code" framing also explains why a critical section can be *bigger* than one variable, and why drawing it too small is a bug. Recall the two-field example:

```go
func (s *Stats) Record(price int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.total += price // both writes must be
	s.count += 1     // inside ONE critical section
}

func (s *Stats) Average() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.count == 0 {
		return 0
	}
	return s.total / s.count // reads both under the SAME lock
}
```

The invariant here is "`total` is the sum of exactly `count` prices." If you locked each field separately — a tiny critical section around `total +=` and another around `count +=` — then another thread calling `Average` could squeeze in *between* the two updates, read the new `total` but the old `count`, and compute a wrong average from a state that should never have been visible. The two fields are *coupled by an invariant*, so they must be updated and read inside *one* critical section that brackets the whole transition from one consistent state to the next. The lock's job is to make sure no thread ever observes the half-updated state where the invariant is temporarily false. **Identify the invariant, then make the critical section exactly as large as the span over which the invariant is violated and restored — no smaller, and no larger.** Too small and you expose the broken intermediate state; too large and you needlessly serialize and risk holding the lock across something slow.

#### Worked example: the bank transfer invariant

Consider a transfer that debits one account and credits another. The invariant is *conservation*: the total money across all accounts is constant; every dollar debited from A must appear in B, and at no point should an observer see money that has left A but not yet arrived in B (or, worse, see it in neither).

```go
func Transfer(from, to *Account, amount int) {
	// Both mutations must be one critical section, or an observer
	// summing all balances mid-transfer sees the money vanish.
	// (Lock ordering matters here — covered in the deadlock post.)
	from.mu.Lock()
	to.mu.Lock()
	defer to.mu.Unlock()
	defer from.mu.Unlock()
	from.balance -= amount
	to.balance += amount
}
```

If you split this into two independent critical sections — one locking `from` to debit, then releasing, then locking `to` to credit — there is a window in which the money has left `from` but not arrived in `to`. An auditor thread that sums all balances during that window sees a total that is `amount` short. The money is not *lost* — it reappears when the credit lands — but the invariant "total is conserved at every observable instant" is violated, and any code that reads the total during the window makes a decision on a false state. The critical section must span the *entire* transition so the broken intermediate is never observable. (Locking two mutexes introduces a *new* hazard — deadlock from inconsistent lock ordering — which is exactly the subject of the next track; here the point is only the boundary of the critical section.)

## Mutual exclusion, safety, and liveness

Now we can be precise about *what kind* of property mutual exclusion is, because concurrency correctness splits into two families and a lock sits squarely on one side of the divide.

A **safety** property says "nothing bad ever happens." Mutual exclusion is a safety property: the bad thing is "two threads inside the critical section simultaneously," and mutual exclusion forbids it for all time. Safety properties are about what the program *never* does. The lost update is a safety violation — money evaporated, which should never happen. Mutual exclusion *restores* the relevant safety property.

A **liveness** property says "something good eventually happens." Examples: "a thread that wants the lock eventually gets it" (no starvation), "the program eventually makes progress" (no deadlock), "every request is eventually served." Liveness is about what the program *eventually* does. And here is the uncomfortable truth: **a plain mutex gives you a safety guarantee but makes no liveness guarantee at all.** It promises $n(t) \le 1$ forever — but it does *not* promise that a waiting thread ever gets in, does not promise the program ever makes progress, does not promise freedom from deadlock. Those are *your* responsibility, layered on top.

This split is why "I added a lock and now my service hangs" is such a common story. The lock did its job — it enforced safety — and in doing so it introduced the *possibility* of a liveness failure that did not exist before. Two threads each holding one lock and waiting for the other's lock will wait forever: a **deadlock**, a liveness violation, and a brand-new bug that the lock *created*. The race went away; a freeze took its place. This is not a reason to avoid locks — it is a reason to understand that a lock trades a safety problem for a *potential* liveness problem, and you must then go solve the liveness problem too (with lock ordering, timeouts, and the careful design that the deadlock post covers).

![two threads serialized through one critical section as the second blocks at acquire until the first releases](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-2.png)

The figure above traces the serialization in time. Thread T1 acquires the lock and works inside the section. T2 reaches its `acquire`, finds the lock held, and *blocks* — it is parked, off the CPU, until T1 releases. Only after T1's release does T2 wake, take the lock, and run its own critical section. At no instant are both inside. That is mutual exclusion as a picture: the critical sections are *interleaved in time but never overlapping*, no matter how the scheduler tries to braid the threads. Notice what the picture also shows about liveness — T2 *waited*. If T1 never released (say it threw an exception on a path that skipped the unlock, or hit an infinite loop, or blocked on I/O that never returns), T2 would wait *forever*. The lock guarantees T2 never enters while T1 is inside; it does *not* guarantee T2 ever enters at all.

## Reentrancy and recursive locks

Here is a trap that catches people the first time, and it follows directly from "a plain mutex makes no liveness guarantee." Suppose a thread holds a lock and then, before releasing it, tries to acquire *the same lock again*. What happens?

With a **non-reentrant** (plain) mutex, the answer is brutal: the thread *blocks*, waiting for the lock to be released — but the only thread that can release it is *itself*, and it is the one blocked. The thread deadlocks against itself, instantly and permanently. This is **self-deadlock**, and it is easy to trigger by accident. You write a method `withdraw` that locks and does its work; later you write a method `transfer` that locks and then calls `withdraw`; now `transfer` holds the lock and `withdraw` tries to take it again. Freeze.

```go
type Account struct {
	mu      sync.Mutex
	balance int
}

func (a *Account) Withdraw(amount int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.balance -= amount
}

func (a *Account) Transfer(to *Account, amount int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Withdraw(amount) // BUG: Withdraw tries to Lock the SAME mutex -> self-deadlock
	// ... credit `to` ...
}
```

Go's `sync.Mutex` is deliberately **not reentrant**. The call `a.Withdraw(amount)` inside `Transfer` calls `a.mu.Lock()` while the current goroutine already holds `a.mu`, and the goroutine blocks forever waiting for itself. Run this and your program hangs; Go's runtime may even detect that all goroutines are asleep and report `fatal error: all goroutines are asleep - deadlock!`.

A **reentrant** (or **recursive**) mutex fixes this specific problem by tracking *which thread owns the lock and how many times it has acquired it*. When the owning thread acquires again, the lock just increments an internal counter and lets it through; each release decrements the counter, and the lock is only truly freed when the count returns to zero. Java's `synchronized` is reentrant — a synchronized method can call another synchronized method on the same object without deadlocking — and so is `java.util.concurrent.locks.ReentrantLock` (the name advertises it). In C++, the reentrant variant is `std::recursive_mutex`.

```java
public class Account {
    private long balance;
    // synchronized methods use the object's monitor, which is REENTRANT

    public synchronized void withdraw(long amount) {
        balance -= amount;
    }

    public synchronized void transfer(Account to, long amount) {
        withdraw(amount);     // re-acquires THIS object's monitor — fine, reentrant
        to.deposit(amount);   // acquires `to`'s monitor (a different lock)
    }

    public synchronized void deposit(long amount) {
        balance += amount;
    }
}
```

The same shape that self-deadlocks in Go is harmless in Java, purely because the JVM's intrinsic lock counts re-entries by the owning thread. So is a reentrant lock simply better? No — and this is the principled position. Reentrancy is a *convenience* that papers over a *design smell*. The fact that you needed to re-acquire a lock you already hold usually means your locking structure is muddled: a public, locking method is calling another public, locking method, and the boundaries of your critical sections have become tangled with the boundaries of your function calls. The cleaner design separates the *locked* logic from the *public* entry: a private helper that assumes the lock is already held, called by a public method that takes the lock once.

```go
// Clean design: lock once at the boundary, call a lock-free internal helper.
func (a *Account) Transfer(to *Account, amount int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.withdrawLocked(amount) // helper ASSUMES the lock is held; takes no lock
	// ...
}

// withdrawLocked must be called with a.mu held. Name encodes the contract.
func (a *Account) withdrawLocked(amount int) {
	a.balance -= amount
}
```

This is why a number of senior engineers regard non-reentrant locks as the *safer* default: the self-deadlock is loud and immediate, it surfaces the muddled design during development, and it pushes you toward the cleaner "lock at the boundary, helpers assume the lock" structure. A reentrant lock silently accepts the tangled call graph, which means a reentrancy *bug* — where you *thought* you were entering fresh but were actually nested, and your invariant was mid-update — goes undetected. Both positions are defensible; what is not defensible is using reentrancy without knowing you are relying on it.

#### Worked example: the reentrancy that hid a broken invariant

Picture a reentrant lock guarding a list and its cached length, with the invariant "`len == size of list`." A public `add` locks, appends to the list, and — before updating `len` — fires a callback to notify listeners. A listener, on the same thread, calls the public `size()` method, which re-acquires the (reentrant) lock and reads `len`. Because the lock is reentrant, the re-acquire succeeds instantly — and the listener reads `len` *while the invariant is broken*: the list already has the new element but `len` has not been bumped yet. The reentrant lock let the thread back into a critical section whose invariant was mid-repair. A non-reentrant lock would have *self-deadlocked* at the callback, which is a far better failure: a hang in development that points straight at the bug, instead of a silently wrong length read in production. Reentrancy did not cause the broken invariant, but it *hid* it by waving the thread through.

![a matrix of what a lock guarantees showing mutual exclusion visibility and ordering as provided while deadlock-freedom and fairness are your responsibility](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-3.png)

## Visibility and ordering come bundled with the lock

Now for the part that surprises people, and the reason a lock fixes *more* than the lost update. Recall the [race condition post](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition): shared mutable state has *three* hazards, not one. **Atomicity** — the read-modify-write must be indivisible. **Visibility** — a write by one thread must become visible to others (without it, a thread can spin forever on a stale cached copy of a flag). **Ordering** — operations must not be reordered across threads in ways that break your reasoning (compilers and CPUs reorder instructions aggressively for speed). Mutual exclusion, on its own, only obviously addresses *atomicity*: it makes the section indivisible by excluding other threads. But here is the gift: **a correct mutex also guarantees visibility and ordering, bundled in, for free.**

The mechanism is the **acquire/release** semantics of the lock operations, and it is worth understanding because it is the bridge from this post to the memory-model track. A lock's `acquire` carries an *acquire barrier*; its `release` carries a *release barrier*. The guarantee these barriers establish is a **happens-before** relationship: *everything a thread did before it released the lock is guaranteed to be visible to the next thread that acquires the same lock.* Formally, the release of a lock *happens-before* the subsequent acquire of that same lock, and happens-before is transitive, so all the writes before the release are ordered before — and visible to — all the reads after the acquire. The lock does not just keep threads out of the section; it *publishes* the section's writes to the next thread that enters.

Concretely: when T1 releases the lock, the release barrier flushes T1's pending writes (out of store buffers, with the right ordering) so they are globally visible. When T2 acquires the same lock, the acquire barrier ensures T2's subsequent reads see those writes rather than stale cached values. Without this, even with perfect atomicity, T2 might read a value T1 wrote *minutes ago* from its own stale cache line. The lock prevents that.

Make the *happens-before* relation precise, because it is the spine of every memory-model argument you will ever make. Happens-before is a partial order over the operations of a program, built from two kinds of edge. First, **program order**: within a single thread, each operation happens-before the next one in source order — sequential reasoning holds *inside* one thread. Second, **synchronization order**: certain pairs of operations across threads are ordered by the runtime, and *the release of a lock happens-before every subsequent acquire of that same lock*. Happens-before is **transitive**: if `A` happens-before `B` and `B` happens-before `C`, then `A` happens-before `C`. Chain the two edge kinds together and you get the guarantee that matters: T1's write (program-order before T1's release) happens-before T1's release, which happens-before T2's acquire (synchronization edge), which happens-before T2's read (program order). By transitivity, *T1's write happens-before T2's read* — so T2 is guaranteed to observe it. If two operations are *not* ordered by happens-before — two unsynchronized accesses to the same location, at least one a write — that is the formal definition of a **data race**, and a data race is undefined behavior in C++ and produces no guarantees in Java and Go. The lock's entire visibility contribution is that it manufactures the happens-before edge that turns a would-be data race into an ordered, well-defined hand-off. That is the mechanism, stated rigorously. This is why, in the Java `Counter` above, the `get()` method is `synchronized` *even though it only reads*: the lock on the read side is what establishes happens-before with the writers, guaranteeing the reader sees the latest value rather than a stale one. A common bug is to lock the writes but not the reads "because reads are atomic anyway" — and then a reader sees a stale value forever because no happens-before edge was ever established to publish the writer's update to it.

![a stack showing the anatomy of a critical section from acquire through read modify write to release as a publish barrier](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-5.png)

The figure above is the anatomy of a correct critical section as a vertical stack: `acquire` (a barrier on entry that synchronizes with the previous holder's release), then the three steps — read shared, modify, write back — running with no other thread inside, then `release` (a barrier on exit that publishes everything to the next holder). The two barriers are not decoration; they are the visibility and ordering guarantees. This is the precise sense in which the lock fixes all three hazards at once: atomicity from exclusion, visibility and ordering from the acquire/release barriers. You do not have to reason about store buffers and cache coherence and instruction reordering *as long as every access to the data goes through the same lock.* That is an enormous simplification, and it is the practical reason locks remain the default tool: they let you think sequentially inside the critical section. The full machinery of barriers, happens-before, and why a `volatile` flag is *not* a substitute for a lock is the subject of the memory-model track; for now, hold onto the headline: **the lock gives you mutual exclusion AND, bundled via acquire/release, visibility and ordering.**

#### Worked example: the flag that "didn't work" without a lock

A classic. One thread loops `while (!done) { /* work */ }`; another thread sets `done = true` to stop it. Without any synchronization, the worker thread may *never see* `done` become true — it caches `done` in a register or a stale cache line, and the writer's update never propagates, so the loop spins forever even though the flag was "obviously" set. The bug is a *visibility* failure, not an atomicity failure; a single boolean write is already atomic on every real platform, yet the loop hangs. Reading and writing `done` under the same lock fixes it, because the release-then-acquire establishes happens-before and publishes the write. (In Java the lighter-weight fix is to declare `done` as `volatile`, which gives visibility without mutual exclusion; in Go you would use `sync/atomic` or a channel. The point for *this* post is that a lock would also fix it, because visibility comes bundled.) This is the memory model peeking through, and it is exactly why a lock buys you more than it appears to.

## Lock scope: short critical sections, and never I/O under a lock

If the lock buys you so much, why not lock generously — take a big lock at the top of every request and hold it through the whole thing? Because the lock *serializes*, and serialization is the direct enemy of the throughput you bought the extra cores for. Every nanosecond a thread holds the lock is a nanosecond no other thread can be in the critical section. The total time the lock is held, summed across all acquisitions, is time your program is effectively *single-threaded*. This is the practical heart of [Amdahl's law](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws): the serial fraction caps your speedup, and a lock held too long *is* serial fraction. So the governing rule of lock scope is: **make the critical section as short as the invariant allows, and not one instruction longer.**

The single worst violation of this rule — the one that turns a healthy service into a frozen one under load — is doing **I/O while holding a lock.** Network calls, disk reads, RPCs, even a slow log write: these take milliseconds to seconds, which is *millions* of times longer than the nanoseconds a memory update takes. Hold the lock across one and every other thread that wants the lock is stuck behind a network round-trip. The throughput of the entire critical-section path collapses to the rate of the slowest I/O. Worse, the queue of blocked threads grows, latency spikes, and you get a **lock convoy** — threads piling up behind the held lock, the system spending its time context-switching threads on and off the wait queue instead of doing work. The fix is almost always to do the I/O *outside* the lock and only take the lock for the brief in-memory update that incorporates the result.

```go
// WRONG: holds the lock across a slow network call.
func (c *Cache) GetSlow(key string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.data[key]; ok {
		return v
	}
	v := fetchFromNetwork(key) // ~50 ms with the lock HELD — convoy city
	c.data[key] = v
	return v
}

// RIGHT: do the slow I/O OUTSIDE the lock; lock only the in-memory updates.
func (c *Cache) GetFast(key string) string {
	c.mu.Lock()
	if v, ok := c.data[key]; ok {
		c.mu.Unlock()
		return v
	}
	c.mu.Unlock() // release BEFORE the slow call

	v := fetchFromNetwork(key) // ~50 ms with NO lock held; others run freely

	c.mu.Lock()
	c.data[key] = v // re-check or just overwrite; brief in-memory write
	c.mu.Unlock()
	return v
}
```

![a coarse lock held across slow I/O causing a convoy contrasted with a minimal critical section that releases the lock before the slow call](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-7.png)

The figure contrasts the two. On the left, `acquire` then a ~50 ms network call then update then `release`: the lock is held for ~50 ms, every other thread queues, throughput dies. On the right, the slow call happens with no lock held, and the lock is taken only for the ~100 ns map write: the hold time drops by a factor of roughly half a million, other threads flow through, throughput stays high. The right-hand version has a subtlety — between the release and the re-acquire, *two* threads might both miss the cache and both fetch the same key (a redundant fetch), and you must decide whether to re-check on re-acquire or accept the duplicate work. That is a real trade-off, but it is *vastly* better than the convoy: a little duplicated I/O beats serializing every request behind one network round-trip. The general principle generalizes beyond I/O: **never hold a lock across anything slow or anything you do not control** — not I/O, not a callback into user code, not a memory allocation that might fault, not another lock acquisition if you can help it (each is a deadlock or convoy risk). Compute and prepare *outside* the lock; take the lock only to commit the result.

## Contention: the cost of serializing

Even a perfectly short critical section is not free, and under load it is far from free. There are two distinct costs, and conflating them leads to bad optimization decisions.

The first is the **uncontended cost**: the raw overhead of `acquire` + `release` when no other thread wants the lock. On modern hardware an uncontended mutex acquire-release is on the order of **tens of nanoseconds** — roughly the cost of a couple of atomic instructions and the associated cache-coherence traffic. (The exact number depends on the platform, the lock implementation, and whether the cache line is hot; treat "tens of nanoseconds" as an order of magnitude, not a precise figure. Modern lock implementations begin with a fast user-space path — typically a compare-and-swap — and only fall into a slow kernel-mediated wait when actually contended; the [how-a-lock-is-built post](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks) shows exactly how.) For a critical section that does meaningful work, tens of nanoseconds of lock overhead is negligible. For a critical section that does almost nothing — like a single `count++`, which is a few nanoseconds of actual work — the lock overhead *dominates*, and you are paying more for the lock than for the operation it protects. That is the first sign you might want an atomic instead of a lock for that particular case.

The second cost is the **contended cost**, and it is qualitatively worse. When multiple threads pile onto one lock, they cannot all proceed; all but one are blocked at any instant. Adding more threads does not add throughput — past the point where the lock is saturated, it *reduces* it, because the threads spend their time contending: cache lines bounce between cores (the lock's memory ping-pongs from cache to cache), the OS context-switches blocked threads on and off wait queues (each context switch costing on the order of a microsecond), and the lock's own internal book-keeping gets hammered. This is why the throughput-versus-threads curve for a contended lock *rises, flattens, and then bends down* — the famous shape every concurrency engineer learns to recognize. The serial fraction (the locked region) imposes an Amdahl ceiling; the coordination overhead drags you *below* even that ceiling.

#### Worked example: throughput versus threads on a shared counter

Take the shared counter and measure throughput (millions of increments per second, total across all threads) as you scale the thread count, for three implementations: no synchronization (racy, *wrong*, but a speed baseline), a mutex around `count++`, and a hardware atomic increment. The shape of the result — measured on a multi-core machine, warmed up, averaged over many runs, with the usual caveat that exact numbers depend heavily on the CPU, the OS scheduler, and the lock implementation — looks like this:

| Threads | Racy (wrong, baseline) | Mutex `count++` | Atomic `count++` |
| ------- | ---------------------- | --------------- | ---------------- |
| 1       | ~250 M/s               | ~70 M/s         | ~180 M/s         |
| 2       | ~480 M/s               | ~22 M/s         | ~90 M/s          |
| 4       | ~900 M/s               | ~12 M/s         | ~55 M/s          |
| 8       | ~1500 M/s              | ~8 M/s          | ~40 M/s          |

Read the columns, not the absolute numbers (which are illustrative order-of-magnitude figures, not a benchmark you should quote). The racy version scales up with threads — and is *wrong*, losing updates, so its throughput is the throughput of garbage. The mutex version *gets slower as you add threads*: at 8 threads it does roughly a third the work of 1 thread, because the lock is a single point of serialization and the eight threads spend their time fighting over it and context-switching. The atomic version also degrades under contention (it too funnels every thread through one cache line) but far less steeply, because it never blocks a thread or enters the kernel — it just retries. The lesson is stark: **a lock on a tiny, hot critical section does not just fail to scale, it scales *negatively*.** Adding cores makes it slower. If your hot path is a single increment, a mutex is the wrong tool and an atomic is right; if your critical section does real work, the lock overhead vanishes into the work and the mutex is fine. *Measure which regime you are in before you optimize.*

The honest measurement method matters as much as the numbers, and concurrency benchmarks are unusually easy to get wrong. A few rules that separate a trustworthy measurement from a misleading one. **Warm up first**: the first runs pay one-time costs — the JIT compiling hot methods, cold caches filling, the lock's slow path being exercised once — so discard them and measure steady state. **Run many times and report the distribution**, not a single number: concurrency results are nondeterministic because the OS scheduler braids the threads differently each run, so report a median and a spread, and be suspicious of any single figure. **Pin the conditions**: fix the thread count, and if you can, pin threads to cores, because the scheduler migrating a thread to a cold core mid-run can swing the result. **Name the confounds**: the OS scheduler is a confound (a busy machine inflates wait times), the memory model is a confound (x86's strong TSO ordering hides reordering bugs that ARM's weaker model exposes, so a benchmark that "passes" on your laptop may fail on a different CPU), and the lock implementation is a confound (a `std::mutex` and a `pthread_mutex` and a Go `sync.Mutex` have different fast paths). **Measure the right metric**: total throughput across all threads, not per-thread time, because per-thread time can *improve* while total throughput collapses. The numbers in the table above are illustrative order-of-magnitude figures gathered under these caveats — treat the *shape* (mutex bends down, atomic bends down less, racy is fast and wrong) as the durable lesson and the absolute values as approximate. Never trust a concurrency benchmark you ran once, on one machine, without a warm-up.

![a graph of the acquire path branching into a fast uncontended path and a contended blocking path that both merge at unlock to wake the next waiter](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-4.png)

The figure traces the two paths through an acquire and why contention is expensive. A request to take the lock branches: if the lock is free (uncontended), the thread enters the critical section immediately — the cheap path, tens of nanoseconds. If the lock is held (contended), the thread blocks and waits — parked off the CPU, a context switch out now and another back later, microseconds of overhead plus the cache-coherence traffic of the lock's memory bouncing between cores. Both paths merge at the critical section (only one thread is ever inside) and again at `unlock`, which wakes the next waiter. The expensive path is the *contended* one, and the more threads pile up, the more of them take it — which is the mechanism behind the throughput curve bending down.

## Lock granularity: coarse versus fine, and the dial between them

The throughput problem points straight at the most consequential design decision in any locked system: **granularity** — how much data each lock protects, and therefore how much of your program is serialized. It is a dial, and neither end is free.

A **coarse-grained** lock protects a lot of data with one mutex — one lock for the entire cache, one lock for the whole connection pool, one lock for the kernel. The virtue is simplicity: one lock means one ordering, no possibility of acquiring two locks in the wrong order, no deadlock from lock interleaving, and a trivial correctness argument ("everything that touches this subsystem holds *the* lock"). The vice is scalability: one lock means one point of serialization, so all the threads that touch *any* of that data contend for the *same* mutex even when they are working on completely unrelated parts of it. Two threads updating two different keys in the same map both queue on the one map lock though they never actually conflict. The coarse lock turns *independent* work into *serialized* work.

A **fine-grained** lock protects a small slice with its own mutex — a lock per hash-bucket, a lock per row, a lock per connection. The virtue is parallelism: threads working on different slices take different locks and proceed simultaneously, so throughput scales with cores instead of collapsing onto one mutex. The vice is complexity and a new hazard: with many locks, a thread sometimes needs *two at once* (to move an item from bucket A to bucket B, or to transfer between two accounts), and the moment two threads acquire two locks in opposite orders, you have a **deadlock** — the exact liveness failure the next track is about. Fine-grained locking buys scalability and pays for it in deadlock surface, lock-ordering discipline, and the per-operation cost of taking more locks. A sharded map with 64 bucket-locks scales beautifully for single-key operations and becomes a minefield the instant you need an operation that spans two buckets.

The principled middle is **lock striping**: instead of one lock for the whole structure or one lock per element, use a fixed array of $N$ locks and map each element to a lock by a hash, so element `k` is guarded by `locks[hash(k) % N]`. With $N=16$ or $64$, sixteen or sixty-four threads working on different elements rarely collide, yet you have only a small, bounded number of locks to reason about — and crucially, a *consistent* order over a small fixed set, which makes multi-lock operations tractable. Java's older `ConcurrentHashMap` used exactly this (a fixed array of segment locks) before moving to per-bucket locking with CAS. The lesson generalizes: **start coarse for correctness, profile to find the lock that is actually your bottleneck, and split only that one finer** — do not pre-emptively shard every lock, because each split adds deadlock surface and complexity for a scalability win you may not need.

#### Worked example: when one lock is enough and when it is not

Suppose your service has an in-memory metrics map updated on every request. With *one* coarse lock around the whole map, at low request rates the lock is uncontended and costs tens of nanoseconds — invisible. You ship it; it is correct and simple. Then traffic grows ten-fold, you profile, and the metrics-map lock is now the top contention point: threads spend a measurable fraction of their time waiting on it, and the throughput-versus-cores curve has bent down. *Now* you stripe it — sixteen locks keyed by a hash of the metric name — and the contention disappears because sixteen unrelated metrics update in parallel. You did not stripe it on day one, because on day one the coarse lock was not your bottleneck and the extra complexity would have bought nothing. Granularity is a response to a *measured* bottleneck, not a default. This is the whole moral of the Big Kernel Lock story in the case-studies section, in miniature: coarse first, then split the *measured* hot lock finer, and only that one.

## Mutex variants: pick the simplest that does the job

The plain mutex is the default, but several variants exist, each adding one capability at one cost. Knowing them keeps you from reaching for a heavier tool than you need — or from missing the lighter tool that solves a specific liveness problem.

![a matrix of mutex variants showing plain reentrant timed and error-checking locks with what each adds and its cost](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-6.png)

| Variant            | What it adds over plain                          | Cost / when to use                                                       |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------------------------ |
| **Plain mutex**    | Mutual exclusion, nothing else                   | Cheapest; the default. Use unless you have a concrete reason not to.     |
| **Reentrant**      | Same thread may re-acquire without self-deadlock | Tracks owner + recursion count. Convenience that can hide a design smell. |
| **Timed / try**    | `try_lock` returns instead of blocking; `lock` with a timeout | Lets you avoid an unbounded wait; you MUST handle the failure return.    |
| **Error-checking** | Detects double-unlock, unlock-by-non-owner, etc. | Extra checks; great in debug builds, often disabled in release.          |
| **Read-write lock**| Many readers OR one writer (not in this table's scope) | Helps read-heavy workloads; heavier than a plain mutex; its own pitfalls. |

The `try_lock` / timed variant deserves a note because it is the principled escape hatch from the "lock makes no liveness guarantee" problem. `try_lock` attempts to take the lock and *returns immediately* with success or failure rather than blocking — so you can do something else (retry later, back off, take a different path) instead of waiting forever. A timed `lock` waits but gives up after a deadline. Both are tools for *bounding* the liveness exposure a lock creates. In Go, `sync.Mutex.TryLock` exists (added in Go 1.18, with a documentation note that its use is rare and usually indicates a design problem); in Java, `ReentrantLock.tryLock()` with an optional timeout; in C++, `std::mutex::try_lock` and `std::timed_mutex::try_lock_for`. Use them when a thread genuinely has useful alternative work or when you need to break a potential deadlock by refusing to wait — but reach for them deliberately, because code littered with `try_lock`-and-retry is usually a sign the locking design needs rethinking.

## What needs a lock, and what does not

We close the conceptual arc where it began: a lock protects *shared mutable state*, and the most reliable way to avoid lock bugs is to have *less shared mutable state to protect.* Not every variable needs a lock; reaching for one reflexively adds serialization and bug surface for nothing. Here is the taxonomy.

![a tree dividing data into things that need a lock such as a shared counter and shared map versus things that do not such as immutable confined and thread-local data](/imgs/blogs/mutual-exclusion-mutexes-and-critical-sections-8.png)

The figure splits your data into two families. **Guard it** — needs a lock — covers anything *shared and mutable*: a counter many threads increment, a map many threads put into and get from, any field whose value changes and is reachable from more than one thread. **No lock needed** covers three escape routes that each remove one of the ingredients the hazard requires:

- **Immutable data.** If a value never changes after construction, any number of threads can read it concurrently with no lock — there is no write to race. Removing *mutability* removes the hazard. This is why functional-style code and "build a new value instead of mutating the old one" patterns are so concurrency-friendly: an immutable snapshot is freely shareable.
- **Thread-confined data.** If data is only ever reachable from one thread — a local variable, an object owned by one thread and never handed out — it is not *shared*, so no lock is needed. Removing *sharing* removes the hazard. Much of the cleanest concurrent code works by giving each thread its own private data and only coordinating at well-defined hand-off points.
- **Thread-local storage.** A formalized version of confinement: each thread gets its own private copy of a variable (`thread_local` in C++, `ThreadLocal` in Java, goroutine-local patterns in Go). No sharing, no lock.

The strategic point: the four genuine ways to make shared state safe are mutual exclusion (this post), atomics (for single-word operations), *immutability* (remove the write), and *confinement / message passing* (remove the sharing). A lock is the right tool when you have genuinely shared, genuinely mutable, multi-word state with an invariant to maintain. When you can instead make the data immutable, or confine it to one thread, or replace it with a single atomic word, those are often *better* than a lock because they sidestep both the serialization cost and the liveness hazards a lock introduces. The whole rest of this series is, in a sense, the catalog of these alternatives — atomics, lock-free structures, channels, actors, STM — each a different answer to "how do I coordinate without a lock, or with a cheaper one." The lock is the foundation you measure them against.

## Case studies / real-world

**The lock convoy and the database connection pool.** A recurring production pattern: a connection pool guarded by a single mutex, where the critical section was made too large — the pool's lock was held not just while picking a connection from the free list (nanoseconds) but across *validating* the connection with a round-trip ping to the database (milliseconds). Under load, every thread requesting a connection queued behind the one thread doing a ping, the pool's throughput collapsed to the ping rate, and threads convoyed on the lock — the classic symptom of I/O held under a lock. The fix is exactly the pattern from the lock-scope section: take the lock only to remove a connection from the free list, release it, *then* validate outside the lock, and re-acquire briefly only to return the connection. The general lesson — never hold a contended lock across a network round-trip — is one of the most reliably learned-the-hard-way rules in server engineering, and it shows up in connection pools, cache layers, and rate limiters alike. (For the architecture-scale version of bounding such queues, see [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure).)

**The Big Kernel Lock.** Early multiprocessor support in operating systems often started with one giant lock around the whole kernel — Linux's "Big Kernel Lock" (the BKL), introduced when SMP support arrived, is the canonical example, and the giant-lock approach also appeared in the early BSD and other Unix lineages. The BKL guaranteed correctness trivially: only one CPU could be inside the kernel at a time, so there were no kernel data races. But it serialized *all* kernel activity — a system call on one core blocked system calls on every other core — so an N-core machine got nowhere near N times the kernel throughput. The multi-year project to *remove* the BKL was precisely the project of replacing one coarse lock with many fine-grained locks, each protecting a small, specific invariant, so that unrelated kernel work could proceed in parallel. The Linux BKL was finally removed entirely in 2011 (kernel 2.6.39), the end of a long campaign. The story is the canonical illustration of the central trade-off of this post: a coarse lock is *correct and simple but does not scale*; fine-grained locking *scales but multiplies the surface for deadlock and complexity*. Granularity is the dial, and there is no free setting.

**The double-checked locking pattern and the memory model.** A famous bug class — important because it shows visibility is not optional. The "double-checked locking" idiom tried to make lazy singleton initialization fast by checking an `instance` field *without* a lock, and only locking if it was null. For years it was written without the proper memory-model annotations, and it was *broken*: due to instruction reordering and visibility (a thread could see a non-null `instance` pointer before the object's constructor writes were visible), a second thread could get a reference to a half-constructed object. The fix required understanding exactly the acquire/release publishing semantics this post described — in Java, marking the field `volatile`; in C++, using atomics with the right memory ordering. The lesson is that the visibility guarantee a lock provides is *load-bearing*, and the moment you try to skip the lock on the read side for speed, you are back in memory-model territory and must reason about happens-before explicitly. That full story is the subject of the memory-model track; here it stands as proof that "the lock also fixes visibility" is not a footnote.

## When to reach for this (and when not to)

A lock is the right tool when you have **genuinely shared, genuinely mutable, multi-word state with an invariant that spans several operations**, and the critical section does enough real work that the lock's overhead disappears into it. The counter-with-two-coupled-fields, the bank transfer, the in-memory index that must stay consistent — these are textbook lock territory. Reach for a plain, non-reentrant mutex first; it is the simplest tool that establishes mutual exclusion plus the bundled visibility and ordering. Keep the critical section tight, keep I/O out of it, and put the lock physically next to the data it guards (or, in Rust, *inside* the type that owns the data).

Do **not** reach for a lock when a cheaper tool removes the hazard entirely. If the data can be **immutable**, make it immutable — there is no faster, safer concurrency than no shared writes. If the data can be **confined to one thread** or replaced with **message passing**, do that — no sharing, no lock, no deadlock. If the shared state is a **single word** updated atomically — a counter, a flag, a pointer swap — an **atomic** instruction is faster and never blocks or deadlocks; a mutex around a bare `count++` is the wrong tool, and the throughput table showed it scaling *negatively*. If your workload is **read-heavy with rare writes**, a read-write lock (many readers or one writer) may beat a plain mutex — but measure, because RWLocks are heavier per-operation and have their own writer-starvation pitfalls. And if you are tempted to go **lock-free** for speed, do not, *unless you have measured that the lock is your bottleneck and proven the lock-free version correct under the memory model* — lock-free code is dramatically harder to get right (the next track shows why), and a mutex that is not your bottleneck is not worth replacing.

The decisive rule: **identify the invariant first, choose the cheapest mechanism that maintains it, then measure under contention.** A lock is rarely the wrong answer for protecting real multi-word invariants, but it is frequently the *lazy* answer for things that did not need to be shared and mutable at all. The best lock is the one you never had to take because you removed the sharing or the mutability instead.

## Key takeaways

1. **A critical section is defined by a safety requirement, not by syntax** — it is any region accessing shared mutable state that must not run concurrently. A mutex enforces $n(t) \le 1$ over it: at most one thread inside at any instant.
2. **A lock protects an invariant over data, not lines of code.** Every access — read and write — must use the *same* lock, and the critical section must span exactly the region where the invariant is temporarily broken. Different locks on the same data exclude no one.
3. **Mutual exclusion is a safety property; a plain lock gives you no liveness guarantee.** It promises the bad thing never happens, but not that a waiting thread ever progresses — deadlock and starvation are new problems the lock can create.
4. **A correct lock fixes all three race hazards at once:** atomicity from exclusion, and *visibility plus ordering bundled in for free* via the acquire/release barriers, which establish happens-before from one holder's release to the next holder's acquire. Lock your reads too, or they may see stale values.
5. **A non-reentrant lock self-deadlocks if a thread re-acquires it;** a reentrant lock allows re-entry by counting owner re-acquisitions, but reentrancy often hides a muddled design. Prefer "lock at the boundary, helpers assume the lock is held."
6. **Keep critical sections short, and never do I/O or call uncontrolled code while holding a lock** — the lock convoy that results collapses throughput to the rate of the slowest operation under the lock. Compute outside, commit inside.
7. **A lock serializes, so it has an Amdahl ceiling and, under contention, scales *negatively*** — a mutex on a tiny hot critical section can be slower at 8 threads than at 1. Measure which regime you are in; an atomic may be the right tool for single-word state.
8. **The cheapest concurrency is no shared mutable state.** Immutability removes the write, confinement and message passing remove the sharing, an atomic replaces a single-word lock. Reach for a mutex when you have a real multi-word invariant — and prefer to remove the hazard when you can.

## Further reading

- **Within this series:** the [intro on why concurrency is unavoidable](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it); the [anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) (the bug this post fixes); the sibling [how a lock is built from test-and-set and CAS](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks); [condition variables and waiting correctly](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly); and the capstone [concurrency playbook for choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- **The Python angle:** [the GIL explained — what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs), for why the locking calculus is different when the runtime already holds a giant lock.
- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming*** — the rigorous treatment of mutual exclusion, from Dekker's and Peterson's algorithms up through locks and lock-free structures.
- **Brian Goetz et al., *Java Concurrency in Practice*** — the definitive practical guide to locking, visibility, the Java Memory Model, and how `synchronized`/`volatile`/`java.util.concurrent` actually behave.
- **Anthony Williams, *C++ Concurrency in Action*** — `std::mutex`, RAII guards, the C++ memory model, and the cost of contention, with measured examples.
- **Edsger W. Dijkstra, "Cooperating Sequential Processes"** — the origin of the critical-section problem, mutual exclusion, and semaphores; the foundational source.
- **Leslie Lamport, "A New Solution of Dijkstra's Concurrent Programming Problem"** (the Bakery algorithm) — mutual exclusion without lower-level atomic primitives, and the seed of the happens-before relation.

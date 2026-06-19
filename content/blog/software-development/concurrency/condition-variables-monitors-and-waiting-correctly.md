---
title: "Condition Variables, Monitors, and Waiting Correctly"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a thread waits for a condition without pinning a core, why wait must atomically release the lock and sleep, and why you always loop on the predicate."
tags:
  [
    "concurrency",
    "parallelism",
    "condition-variable",
    "monitor",
    "producer-consumer",
    "synchronization",
    "wait-notify",
    "thread-safety",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/condition-variables-monitors-and-waiting-correctly-1.png"
---

A queue sits between a producer thread and a consumer thread. The producer drops work in; the consumer takes work out. The first version anyone writes looks reasonable and runs correctly — and then you open a system monitor and see one CPU core pinned at 100% with the program doing, as far as you can tell, nothing at all. No work is flowing. The consumer is just *looking*. It checks the queue, finds it empty, checks again, finds it empty, checks again, a hundred million times a second, melting a core to ask the same question that keeps getting the same answer: nothing's here yet. Your laptop fan spins up. Your cloud bill goes up. The battery drains. And the moment a real item finally arrives, the consumer is already busy spinning, so it grabs the item *eventually* — but it paid for that responsiveness by never once letting the core idle.

That spinning loop is called **busy-waiting**, and it is the wrong way to wait. The right way is to tell the operating system *"wake me when something changes, and until then, take me off the run queue entirely so I cost nothing."* The primitive that does exactly this is the **condition variable** — a place a thread can go to sleep on a logical condition (`the queue is non-empty`) and be woken by another thread that just made that condition possibly-true. With it, the same producer-consumer queue runs at essentially 0% CPU while idle, wakes within microseconds when work arrives, and scales to thousands of waiters without melting anything.

![a busy-wait spin loop that pins one core at full utilization contrasted with a condition variable wait that parks the thread and uses almost no CPU](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-1.png)

But condition variables are also one of the most consistently *misused* primitives in all of concurrency. There is a subtle race — the **lost wakeup** — that will make your consumer sleep forever if you get the ordering of two operations wrong by a single instruction. There is a rule — *always loop on the predicate, never use an `if`* — that every textbook states and that an astonishing amount of production code still violates, with consequences that show up as "once a month, this thread is mysteriously stuck." There are **spurious wakeups**, where a thread wakes up for no reason at all and a naive `if` check waltzes straight past a false condition. And there's a whole vocabulary — `signal` versus `broadcast`, **Mesa** versus **Hoare** semantics, the **monitor** that bundles a lock with its conditions — that you need in order to reason about *which* thread runs *when* after a wakeup.

This post builds all of it from the ground up. We'll start from the busy-wait and earn the condition variable. We'll walk the lost-wakeup race instruction by instruction and watch the atomic `release-and-sleep` close it. We'll write the canonical bug — an `if`-guarded wait — in two languages, watch it break, and fix it with a `while`. We'll build a complete, correct **bounded buffer** (the producer-consumer queue done right) and then measure it: busy-wait versus condition variable, side by side, CPU percentage on the table. By the end you'll know not just the API but the *reasoning* — and you'll never write a bare `if` around a `wait()` again. This is the third post in the locks track, following [mutual exclusion and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections); a condition variable is the tool you reach for once a plain mutex isn't enough because a thread needs to *wait for a state*, not just *take a turn*.

## Waiting is a first-class problem

Step back from the API for a moment, because the thing we're solving is more fundamental than any one function call. A thread frequently needs to wait until *some other thread* establishes a condition. The consumer waits until the producer has put an item in the queue. A worker waits until the connection pool has a free slot. A request handler waits until a downstream response arrives. The recurring frame of this whole series is **shared mutable state plus nondeterministic scheduling**, and here the shared state is the queue and the *condition* over it ("is it non-empty?"), while the nondeterminism is that you have no idea *when* the producer will run relative to the consumer. The waiting thread can't predict the future; it can only react to a change made by someone else.

There are exactly two ways to wait. You can **poll** — loop, check, loop, check — which keeps the CPU busy and is therefore called busy-waiting or spinning. Or you can **block** — surrender the CPU, ask the OS to set you aside, and be woken by an event. The whole reason condition variables exist is that the OS scheduler already knows how to put a thread to sleep and wake it later, but it needs to know *what to wake it for*. A condition variable is the rendezvous point: waiters register their interest in a condition by going to sleep on the variable, and the thread that changes the underlying state *notifies* the variable, which wakes the sleepers.

There is one crucial subtlety that distinguishes a condition variable from a simple event flag, and it's the source of every bug in this post: **a condition variable does not remember anything.** It is *stateless*. If you signal a condition variable and nobody is currently waiting on it, the signal is *lost* — there is no flag set, no counter incremented, nothing. This is unlike a semaphore, which *does* count and remember. The condition variable's amnesia is by design (it lets you decouple the *condition* — which lives in your own shared variables — from the *waiting mechanism*), but it means the actual truth about whether the queue has items must live in *your* state, protected by *your* lock, and the condition variable is purely the wakeup channel. Forget this and you get lost wakeups. We'll come back to this repeatedly.

#### Worked example: the cost of polling

Make the cost concrete. A tight polling loop `while (queue.isEmpty()) {}` executes, conservatively, on the order of $10^8$ to $10^9$ iterations per second on a modern core, because each iteration is just a load and a compare-and-branch. That's an entire core — call it 100% of one hardware thread — spent producing the answer "still empty." If the item you're waiting for arrives once per second, you have spent roughly $10^8$ wasted iterations per useful wakeup. On a laptop that's a few watts of heat and a spun-up fan; on a 64-core server with one polling thread per core you've turned the whole machine into a space heater that does no work. The condition-variable version, by contrast, executes *zero* instructions while waiting — the thread is off the run queue entirely — and the kernel's wakeup path costs a few microseconds when the item finally arrives. We measure this exactly later; the order of magnitude is "100% of a core" versus "unmeasurably close to zero."

So the goal is clear: wait *correctly* means wait at near-zero CPU, wake promptly, and — this is the hard part — never miss a wakeup and never proceed on a false condition. Let's build the tool.

## The condition-variable API

Every condition-variable implementation, across every language, exposes the same three operations. The names differ; the contract is identical.

- **`wait()`** — called by a thread that wants to wait for the condition. It atomically releases the associated lock and puts the calling thread to sleep. When the thread is later woken, `wait()` reacquires the lock before returning. (We'll dissect that "atomically" in the next section; it is the whole ballgame.)
- **`signal()`** / **`notify()`** — called by a thread that just changed the shared state. It wakes *one* thread currently waiting on the condition variable (if any). If no thread is waiting, it does nothing — the signal evaporates.
- **`broadcast()`** / **`notifyAll()`** — wakes *all* threads currently waiting. Again, if none are waiting, nothing happens.

The mapping across the languages we use in this series:

| Concept            | C++ (`std::condition_variable`) | Java (`Object` / `Condition`) | Go (idiom)                | Rust (`std::sync::Condvar`)     |
| ------------------ | ------------------------------- | ----------------------------- | ------------------------- | ------------------------------- |
| the lock           | `std::unique_lock<std::mutex>`  | `synchronized` / `Lock`       | `sync.Mutex`              | `Mutex<T>` guard                |
| wait for condition | `cv.wait(lock, pred)`           | `obj.wait()` / `cond.await()` | `cond.Wait()` or channels | `cv.wait_while(guard, cond)`    |
| wake one           | `cv.notify_one()`               | `obj.notify()` / `signal()`   | `cond.Signal()`           | `cv.notify_one()`               |
| wake all           | `cv.notify_all()`               | `obj.notifyAll()` / `signalAll()` | `cond.Broadcast()`    | `cv.notify_all()`               |

A condition variable is **always paired with a mutex.** This is not optional and not a stylistic choice — the pairing is what makes correctness possible. The mutex protects the shared state that the condition is *about* (the queue, the count, the flag). You hold the lock, you inspect the state, and if the condition isn't satisfied you call `wait()`, which releases the lock so other threads (the producer!) can make progress and possibly establish the condition. When woken, `wait()` reacquires the lock so you can safely re-inspect the state. The lock and the condition variable are two halves of one mechanism; this bundle has a name — a **monitor** — which we'll formalize shortly.

Here is the shape of *correct* usage. Memorize this skeleton; everything else in the post is a justification of one of its lines.

```cpp
std::mutex m;
std::condition_variable cv;
std::queue<Item> q;          // the shared state the condition is about

// Consumer: wait until the queue is non-empty, then take an item.
Item take() {
    std::unique_lock<std::mutex> lock(m);   // 1. acquire the lock
    while (q.empty()) {                      // 2. LOOP on the predicate
        cv.wait(lock);                       // 3. atomically release+sleep; reacquire on wake
    }                                        // 4. loop back and RE-CHECK the predicate
    Item it = q.front();                     // 5. condition holds: safe to proceed
    q.pop();
    return it;
}                                            // 6. lock released here (RAII)

// Producer: add an item, then wake a waiter.
void put(Item it) {
    {
        std::unique_lock<std::mutex> lock(m);
        q.push(it);                          // change the state
    }                                        // release the lock (see pitfalls re: ordering)
    cv.notify_one();                          // wake one waiter
}
```

Two lines in there are doing almost all the work and are almost universally gotten wrong by beginners: line 2, the `while` (not an `if`), and line 3, the `wait` that *atomically* releases and sleeps. We spend the next two sections on exactly those two lines, because they are where the dragons live.

## Why wait must atomically release-and-sleep

Here is the most important sentence in this entire post: **`wait()` must release the lock and put the thread to sleep as a single, indivisible, atomic operation.** If those two steps — release the lock, go to sleep — could be separated by even a single instruction during which neither has happened or both halfway have, a wakeup can be lost and the thread can sleep forever. Let me prove it by building the broken version and watching it fail.

Suppose, hypothetically, that `wait()` were *not* atomic. Suppose it were implemented in two visible steps: first `unlock(m)`, then `sleep()`. The consumer's code would effectively be:

```c
// BROKEN: a non-atomic "wait" — DO NOT DO THIS. This is the bug.
lock(m);
if (queue_is_empty()) {   // we hold the lock here; the predicate is true
    unlock(m);            // step A: release the lock
    // ----- DANGER WINDOW: we are awake, hold no lock, have not yet slept -----
    sleep();              // step B: go to sleep, waiting to be signaled
}
// ... woken, reacquire, proceed ...
```

Now run two threads against it and pick the worst interleaving the scheduler can hand you. This is the **lost-wakeup race**, and the danger window is the gap between step A and step B.

![a step by step timeline of the lost wakeup race where the producer signal lands in the gap between releasing the lock and sleeping so the consumer never wakes](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-2.png)

1. **Consumer** acquires the lock, checks the queue, finds it empty. The predicate (`queue_is_empty`) is true, so the consumer is committed to going to sleep.
2. **Consumer** executes step A: `unlock(m)`. It now holds no lock. It is *awake*. It has *not yet* called `sleep()`. It is sitting in the danger window.
3. The scheduler preempts the consumer here — at the worst possible instant — and runs the **producer**.
4. **Producer** acquires the lock (it's free — the consumer released it in step A), pushes an item into the queue, releases the lock, and calls `signal()`. But *no thread is currently waiting* — the consumer hasn't called `sleep()` yet! The condition variable is stateless; it remembers nothing. **The signal evaporates.** It wakes zero threads because zero threads are asleep.
5. The scheduler now resumes the **consumer**, which executes step B: `sleep()`. It goes to sleep, waiting for a signal that *already came and went*.
6. The consumer sleeps **forever.** The queue has an item in it. There is no future signal coming (the producer already did its one signal). The consumer will never wake. This is a hang — and it's load-dependent, timing-dependent, and "works on my machine" right up until production traffic widens that danger window at exactly the wrong moment.

That is the lost wakeup, and it is a *deadlock by amnesia*: the consumer is permanently blocked waiting for an event that has already occurred. Notice what made it possible — the gap between releasing the lock and sleeping, during which the producer could observe a free lock, change the state, signal, and find nobody home.

**The fix is to make release-and-sleep atomic.** A real `wait()` is implemented so that no thread can sneak in between the unlock and the sleep. The kernel enqueues the thread onto the condition variable's wait queue *before or as part of* releasing the lock, so the moment the lock becomes available to the producer, the consumer is *already registered as a waiter*. Now when the producer signals, there is a waiter to wake. The danger window is gone.

Mechanistically, here is how the atomicity is actually achieved (the details vary by platform, but the principle is universal). On Linux, condition variables are built on **futexes** (fast userspace mutexes). `wait()` records the futex's current value, then makes a single `futex(FUTEX_WAIT, ...)` syscall that says "sleep me, *but only if the futex still has this value*." The kernel checks the value atomically under its own internal lock. If a signaler bumped the value in the meantime, the `FUTEX_WAIT` returns immediately instead of sleeping — so the wakeup is not lost, it just turns into an "already changed, don't sleep" return. The userspace condition-variable code combines this with the mutex hand-off so that the enqueue-as-waiter and the unlock are, from any other thread's perspective, a single step. POSIX `pthread_cond_wait` specifies this contract directly: *"These functions atomically release the mutex and cause the calling thread to block on the condition variable."* The word "atomically" in that sentence is the entire reason the primitive is safe.

![a five step stack showing what the wait call does internally check the predicate then atomically release the lock and sleep then reacquire and re-check](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-4.png)

The practical upshot for *you*, the application programmer: you must never roll your own "unlock then sleep." Always use the `wait()` provided by your standard library, which has the atomicity baked in. Your job is to hold the lock when you call `wait()`, and `wait()` will hand it off correctly. In C++ this is why `cv.wait(lock)` *takes the lock as an argument* — it needs to release it as part of the atomic step. In Java, `obj.wait()` requires you to already hold the monitor (you must be `synchronized` on `obj`), and it releases that monitor as it sleeps. The API forces the pairing on you precisely so the atomicity can be guaranteed.

#### Worked example: the danger window in instruction count

How wide is the danger window in the broken version? It's the number of instructions between `unlock(m)` and `sleep()` — typically a handful: the return from `unlock`, maybe a function-call setup for `sleep`, the syscall trap. Call it tens of nanoseconds of wall-clock time. So the bug fires only if the scheduler preempts the consumer *and* runs the producer to completion *and* the producer's whole put-and-signal happens *within those tens of nanoseconds*. That's why it's rare — a "one in ten million" hang. But "rare" is not "never," and at a million requests a second a one-in-ten-million race fires several times a day. The atomic `wait()` makes the window *zero instructions wide*, which is the only window width that's actually safe. You cannot make a rare race acceptable by making it rarer; you make it impossible by closing the window entirely.

## Always loop on the predicate — never use `if`

You may have noticed that the *correct* skeleton used `while (q.empty()) cv.wait(lock);` but the *broken* lost-wakeup example used `if (queue_is_empty())`. The `if`-versus-`while` distinction is a second, independent correctness rule, and it trips people even after they've internalized the atomic-wait rule. The rule is absolute: **after `wait()` returns, you must re-check the predicate, because the predicate may be false even though you were woken.** The only way to re-check is to loop.

Why might the predicate be false after a wakeup? Three independent reasons, any one of which is sufficient to mandate the loop.

![a matrix of three situations spurious wakeup multiple waiters and a stolen item showing how an if check breaks and a while loop stays correct](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-3.png)

**Reason 1: spurious wakeups.** A `wait()` is permitted, by specification, to return *without any signal having been sent at all.* This is not a bug in your code or the library — it is a deliberate concession in the POSIX and C++ standards (and the JMM) that lets the implementation be faster and simpler. The futex layer, signal delivery, and certain kernel paths can all cause a `wait()` to return early. POSIX `pthread_cond_wait` explicitly warns: *"Spurious wakeups ... may occur."* So a thread can wake up, find the queue still empty (because nobody put anything in it), and — with an `if` — sail right past into `q.front()` on an empty queue, which is undefined behavior, a crash, or a corrupted read. With a `while`, the thread re-checks, finds the queue empty, and goes right back to `wait()`. No harm done.

**Reason 2: multiple waiters and a single item.** Suppose three consumers are all waiting on a non-empty condition and the producer adds *one* item and calls `broadcast()` (wakes all three). All three wake. All three race to reacquire the lock; one wins, takes the single item, and the queue is empty again. The other two then acquire the lock in turn — and the queue is empty. With an `if`, they proceed anyway and try to take a non-existent item. With a `while`, they re-check, see empty, and wait again. The loop is what makes a `broadcast` to many waiters safe even when only some of them can actually proceed.

**Reason 3: a stolen item (the woken thread isn't the next to run).** Even with `signal()` (wake exactly one), under **Mesa semantics** — which every mainstream language uses, and which we'll define formally below — the woken thread does not run *immediately*. It is merely made *runnable*. Between the signal and the moment the woken thread actually reacquires the lock and runs, some *other* thread (a newly-arrived consumer that wasn't waiting at all) can swoop in, acquire the lock first, and take the item. Now the woken thread finally runs, reacquires the lock — and the item it was woken for is gone. With an `if`, it proceeds on an empty queue. With a `while`, it re-checks and waits. This is the most common real-world cause, and it has nothing to do with spurious wakeups or broadcasts — it's the fundamental nature of Mesa-style condition variables.

Here is the bug and the fix, side by side, in **Java** (where `Object.wait`/`notifyAll` is the classic monitor API):

```java
// BUG: an `if` guard. A spurious wakeup, a stolen item, or a broadcast to many
// waiters lets this thread proceed when the queue is actually empty.
synchronized (queue) {
    if (queue.isEmpty()) {   // checked ONCE
        queue.wait();        // released monitor, slept, reacquired, returned
    }
    return queue.remove();   // BUG: may run on an empty queue -> exception
}
```

```java
// FIX: a `while` loop. After every wakeup we re-check the predicate; if it's
// still false we go right back to wait(). This is the only correct shape.
synchronized (queue) {
    while (queue.isEmpty()) {   // RE-CHECK on every wakeup
        queue.wait();
    }
    return queue.remove();      // guaranteed non-empty: we just checked under the lock
}
```

And the same bug and fix in **C++**, where the two-argument `wait(lock, predicate)` overload exists precisely to encode the loop *for* you:

```cpp
// BUG: bare wait() with no re-check. Equivalent to an `if` — proceeds blindly
// after a single wakeup even if the predicate is false.
std::unique_lock<std::mutex> lock(m);
if (q.empty()) {
    cv.wait(lock);       // wakes for ANY reason; no re-check
}
Item it = q.front();     // BUG: q may be empty here
q.pop();
```

```cpp
// FIX, explicit loop — the canonical form, shows the mechanism plainly:
std::unique_lock<std::mutex> lock(m);
while (q.empty()) {
    cv.wait(lock);
}
Item it = q.front();     // safe
q.pop();

// FIX, idiomatic — the predicate overload IS the while-loop, written once:
std::unique_lock<std::mutex> lock(m);
cv.wait(lock, [&]{ return !q.empty(); });   // loops internally until predicate true
Item it = q.front();     // safe; also immune to spurious wakeups
q.pop();
```

The `cv.wait(lock, pred)` overload is *exactly* `while (!pred()) cv.wait(lock);` — the standard library wrote the loop so you can't forget it. There's a deeper reason this matters than mere convenience. The predicate you pass and the predicate you'd write in the `while` must be *identical* — they must test the same shared state, read under the same lock. A common subtle bug is a predicate that tests a *different* or *stale* variable than the one the signaler actually changed: you wake, the closure checks the wrong field, finds it satisfied, and proceeds while the thing you actually needed is still not ready. Keeping the predicate as a small closure right next to the `wait` call (rather than a flag set somewhere far away) is a discipline that keeps the check and the wakeup talking about the same state. **Rust** goes a step further and bakes the loop into the API name with `wait_while`, so the type system practically pushes you toward correctness:

```rust
// Rust: wait_while loops until the closure returns false. The std API name
// itself reminds you that you are looping on a predicate, not checking once.
let mut q = cv.wait_while(mutex.lock().unwrap(), |q| q.is_empty()).unwrap();
let item = q.pop_front().unwrap(); // q is guaranteed non-empty
```

The takeaway is a single mechanical rule that you apply without thinking: **a `wait()` always lives inside a `while` loop that tests the exact predicate you're waiting for.** Never an `if`. The C++ and Rust predicate overloads are the loop in disguise; the Java and Go forms make you write the loop yourself. If you remember nothing else from this post, remember `while (!predicate) wait();`.

There's a question worth answering directly, because it bothers careful readers: *isn't the `while`-loop just re-introducing a busy-wait?* No, and the distinction is the whole point. A busy-wait loops *without sleeping* — every iteration runs at full speed, burning CPU, never yielding the core. The `while`-around-`wait` loop sleeps *inside* the loop body: each iteration that finds the predicate false calls `wait()`, which parks the thread at ~0% CPU until the next signal. The loop iterates only *once per wakeup*, not billions of times per second. So in the common case the loop runs its body exactly once (check, false, sleep), then once more after the real wakeup (check, true, proceed) — two iterations total, with a long zero-CPU sleep in between. Only a pathological storm of spurious wakeups would make it iterate more, and even then each iteration is gated by a real sleep. The `while`-loop is correctness scaffolding around a genuinely blocking wait, not a spin. That's why it costs you a re-check, not a core.

## Spurious wakeups, in detail

Spurious wakeups deserve their own moment because they sound like folklore and are actually specified behavior. A *spurious wakeup* is a return from `wait()` that was not caused by any matching `signal()` or `broadcast()`. The thread simply wakes up "for no reason." Newcomers often assume this is a defect — surely a correct implementation wakes a thread *only* when signaled? — but the standards explicitly *permit* it, and they permit it for good engineering reasons.

The mechanism: on many systems, condition variables are layered over lower-level wait primitives (futexes on Linux, similar on other OSes), and several events can cause that lower primitive to return early — the delivery of a POSIX signal to the thread, certain kernel scheduling paths, and an optimization where the implementation prefers to wake a thread and let it re-check rather than maintain exact bookkeeping about who must wake. Crucially, there's also a subtler "spurious-like" effect from the Mesa hand-off we already saw: a genuine signal wakes you, but by the time you run, the condition is false again. From your code's perspective, "I woke and the condition is false" looks identical whether the cause was a truly spurious kernel wakeup or a stolen item. **You don't need to distinguish them — the `while`-loop handles both identically.** That's the beauty of the loop-on-predicate discipline: it makes spurious wakeups a non-event. You re-check, it's false, you wait again, costing one extra predicate evaluation and one extra trip back to sleep. No correctness impact whatsoever.

It's worth dispelling one piece of folklore: spurious wakeups are *not* common in practice — on a quiet system you may go millions of waits without seeing one. That rarity is exactly the trap. If you write an `if` instead of a `while`, your code will pass every test, run clean in staging, and survive code review, because the spurious wakeup that exposes the bug almost never fires under light load. Then production traffic, an unlucky POSIX signal, or simply more waiters arrive, the wakeup fires, the `if` lets the thread through on a false predicate, and you get a once-a-month crash that nobody can reproduce. The cost of the `while` is one extra predicate evaluation in the rare case; the cost of the `if` is a Heisenbug. Write the `while` even though you'll "never" see the wakeup it guards against — *especially* because you'll never see it, since that's what makes the alternative bug so insidious.

This is why the C++ standard's single-argument `cv.wait(lock)` documentation says, in so many words, that it "may block ... or may unblock spuriously," and why the predicate-taking `cv.wait(lock, pred)` is preferred — it absorbs spurious wakeups silently. Java's `Object.wait` documentation contains the same warning and recommends the canonical `while` loop. The Linux man page for `pthread_cond_wait` and `pthread_cond_timedwait` states plainly that "spurious wakeups ... may occur" and that "applications ... should not assume that [a return] implies that the condition is true." Across every platform, the same rule falls out: *loop on the predicate.* You've now seen three independent justifications for that loop — spurious wakeups, multiple waiters, and stolen items — converge on one line of code.

## Signal versus broadcast — and the thundering herd

When you've changed the shared state and want to wake a waiter, you choose between `signal()` (wake one) and `broadcast()` (wake all). The choice is a real trade-off with correctness *and* performance consequences, and getting it wrong gives you either a hang or a stampede.

![a contrast between signal which wakes exactly one parked waiter and broadcast which wakes all of them and produces a thundering herd that re-contends](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-5.png)

**`signal()` wakes one waiter.** It's cheaper — one thread becomes runnable, one trip through the scheduler, one lock acquisition. Use it when *any single waiter can handle the event and you only created enough work for one.* The classic case: a producer adds *one* item to the queue. Exactly one consumer can take that item, so waking one is correct and efficient. Waking all of them would be wasteful — they'd all wake, all contend for the lock, one would take the item, and the rest would re-check, find the queue empty, and go back to sleep, having accomplished nothing but burning CPU and lock traffic.

**`broadcast()` wakes all waiters.** It's the safe-but-expensive choice. Use it when *you don't know which waiter (or how many) can now proceed*, or when *multiple waiters might be able to proceed and they're waiting on different conditions sharing one variable.* The classic case: a writer releases a readers-writer lock, and *all* the readers can now proceed — so you broadcast. Another classic: you've added several items at once, or you've changed state that several differently-predicated waiters might care about. When in doubt about correctness, `broadcast()` is never *wrong* (the `while`-loop guarantees the spuriously-woken ones just go back to sleep) — it's only ever *inefficient*.

That inefficiency has a name: the **thundering herd.** Broadcast wakes $N$ threads; they all become runnable; they all stampede toward the *one* lock that `wait()` must reacquire; the lock is a serialization point, so they acquire it one at a time, each re-checking the predicate, and all but a few discover the condition is no longer satisfiable and go right back to sleep. You paid $N$ context switches and $N$ lock acquisitions to let perhaps one thread make progress. On a queue with thousands of idle waiters, a careless `broadcast()` on every single `put()` turns each enqueue into a thread storm. The fix is to `signal()` when one item means one consumer, and reserve `broadcast()` for the genuinely-many-can-proceed cases.

There's a subtle correctness trap with `signal()` that's worth flagging now and revisiting in the bounded-buffer section: if you have **waiters with different predicates sharing one condition variable**, `signal()` can wake the *wrong* one. Imagine producers waiting on `not-full` and consumers waiting on `not-empty`, but both using *the same* condition variable. A consumer takes an item (making the buffer not-full) and calls `signal()` to wake a producer — but `signal()` picks an arbitrary waiter, and it might wake *another consumer*, who re-checks `not-empty`, finds nothing actionable for it, and goes back to sleep. The producer that should have woken never does. Now you can deadlock with the buffer half-full and everyone asleep. The two robust fixes: use a **separate condition variable per predicate** (a `not-full` CV and a `not-empty` CV — the approach our bounded buffer will take), or use `broadcast()` so the right waiter is guaranteed to be among those woken. This is a real bug class; "use one CV and `signal`" is the seductive, broken shortcut.

| | `signal()` / `notify()` | `broadcast()` / `notifyAll()` |
| --- | --- | --- |
| wakes | exactly one waiter | every waiter |
| cost | one wakeup, one lock acquire | $N$ wakeups, $N$ lock acquires |
| use when | one event, any single waiter can handle it | many can proceed, or mixed predicates on one CV |
| failure if misused | wakes the wrong waiter when predicates differ — possible hang | thundering herd — wasted CPU and lock contention |
| safe default | only when one CV serves one predicate | always correct (with a `while`-loop), sometimes slow |

## Monitors: bundling the lock with its conditions

We've been pairing a mutex with a condition variable by hand. The idea of bundling *the lock that protects some shared state* together with *one or more condition variables for waiting on that state* is old and important enough to have a name: a **monitor.** The term comes from Per Brinch Hansen and Tony Hoare in the early 1970s, and it's the conceptual ancestor of Java's `synchronized` and every "thread-safe object" you've ever used.

A monitor is an object (or module) with three ingredients: (1) some encapsulated shared state, (2) a single mutual-exclusion lock that *every* method of the object acquires on entry and releases on exit — so only one thread is ever "inside" the monitor at a time — and (3) one or more condition variables on which threads inside the monitor can wait (temporarily leaving, by releasing the lock) and be signaled. The monitor's promise is that all access to the protected state happens under the lock, so the *invariant* over that state holds whenever no thread is mid-method. Waiting is the only way to leave the monitor without finishing your method, and `wait()` carefully restores the lock before your method continues.

Java makes this nearly literal: *every object is a monitor.* The `synchronized` keyword acquires the object's intrinsic lock; `wait()`, `notify()`, and `notifyAll()` are methods on `Object` itself and operate on that same intrinsic lock's condition. So `synchronized (obj) { while (!ready) obj.wait(); }` is a textbook monitor in three keywords. The newer `java.util.concurrent.locks` package separates the pieces — a `ReentrantLock` plus one or more `Condition` objects from `lock.newCondition()` — which buys you exactly the "multiple condition variables on one lock" capability the bounded buffer wants. Go's `sync.Cond` wraps a `sync.Locker` (usually a `sync.Mutex`) to form the same bundle, though Go programmers more often reach for channels (more on that shortly). C++ keeps them separate (`std::mutex` + `std::condition_variable`) but the discipline is identical: one mutex guards the state, the condition variable(s) coordinate the waiting. Rust's `Condvar` is likewise paired with a `Mutex<T>`, and because the data lives *inside* the `Mutex`, the type system won't even let you touch the state without holding the lock — the monitor's "all access under the lock" rule becomes a compile error to violate.

The reason "monitor" is worth knowing as a concept and not just a keyword: it tells you *where the condition variable's lock comes from* and *why the predicate must be checked under that lock.* The predicate is a statement about the monitor's protected state; to evaluate it soundly you must hold the monitor lock (otherwise another thread could change the state between your check and your `wait()`); and `wait()` releasing-and-reacquiring *that same lock* is what lets other threads enter the monitor, change the state, and signal you. The lock, the state, the predicate, and the condition variable are one coherent unit. That's the monitor.

A monitor also clarifies a property that's easy to lose track of: the **invariant**. A well-designed monitor maintains some invariant over its protected state that is *true whenever no thread is executing inside it* — for the bounded buffer, the invariant is roughly "the queue holds between 0 and `cap` items and `size` matches the actual count." Each method may *temporarily* break the invariant while it's mutating state (mid-`push`, the count and the contents are briefly inconsistent), but it must *restore* it before releasing the lock. Critically, `wait()` is a point where you *leave* the monitor — it releases the lock — so the invariant must hold *before* you call `wait()`, just as it must before any normal method return. If you call `wait()` with the invariant broken, another thread enters the monitor and observes a corrupt state. This is why the canonical pattern checks the predicate (a fact about a *consistent* state), waits if needed, and only mutates *after* the predicate holds — the mutation is the one place the invariant is allowed to wobble, bracketed by a consistent before and a consistent after. Thinking in invariants is what turns "I sprinkled some locks and signals around" into "I can state precisely what this monitor guarantees."

## Mesa versus Hoare semantics

When a thread inside a monitor calls `signal()` to wake a waiter, *who runs next* — the signaler or the woken waiter? There are two classic answers, and which one your system uses changes whether you *must* loop on the predicate.

![a matrix comparing Mesa and Hoare monitor semantics across who runs after a signal whether the predicate must be re-checked and whether the signaler keeps the lock](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-6.png)

**Hoare semantics** (named for Tony Hoare's original monitor proposal): `signal()` *immediately* transfers control — and the lock — to the woken waiter. The signaler is suspended; the waiter runs *right now*, before any other thread can touch the state. Because control passes directly and atomically, the waiter is *guaranteed* that the condition it was woken for is still true — nothing could have changed it in between. Under pure Hoare semantics you could, in principle, use an `if` instead of a `while`, because the predicate is provably true on wakeup. The cost is implementation complexity: the runtime must do an immediate, precise context switch and lock hand-off on every signal, and it must track suspended signalers to resume them later. Almost no mainstream system implements true Hoare semantics for general condition variables because of this overhead.

**Mesa semantics** (named for the Mesa language at Xerox PARC, and described by Lampson and Redell): `signal()` merely makes the waiter *runnable* — it moves the waiter from the "waiting" set to the "ready" set — and the signaler *keeps running and keeps the lock.* The woken waiter doesn't run until it's scheduled and can reacquire the lock, which may be much later. In the interim, the signaler can keep changing the state, *other* threads can enter the monitor and change the state, and by the time the waiter actually runs, the condition it was woken for may be *false again.* This is precisely the "stolen item" scenario. **Therefore, under Mesa semantics, you MUST loop on the predicate** — the wakeup is only a *hint* that the condition *might* be true, never a guarantee.

Why did Mesa win so completely? Because the immediate-hand-off that Hoare semantics requires is expensive and awkward to implement on a real preemptive multiprocessor. To hand control *directly* from signaler to waiter, the runtime must perform a precise context switch at the exact instant of `signal()`, transfer lock ownership atomically, and then remember to resume the suspended signaler later — and it must do all this even when the signaler still had useful work to do after the signal. On a multicore machine where threads run truly in parallel, "the waiter runs immediately and nobody else touches the state in between" is a guarantee that fights the hardware: you'd have to stall other cores to honor it. Mesa's signal-and-continue, by contrast, is almost free — just move a thread from the wait set to the ready set and let the scheduler do its normal thing. The signaler doesn't even have to stop. The price is that the waiter's wakeup becomes a *hint*, and the application must re-check. Lampson and Redell judged that price trivial (you were going to handle spurious wakeups anyway) against the implementation savings, and every system since has agreed.

**Every mainstream language and OS uses Mesa semantics.** Java's monitors are Mesa. POSIX condition variables are Mesa. C++ `std::condition_variable` is Mesa. Go's `sync.Cond` is Mesa. Rust's `Condvar` is Mesa. This is *the* reason the `while`-loop rule is universal and non-negotiable: a Mesa `signal()` is an advisory nudge, not a contract. The Lampson-Redell paper that introduced Mesa monitors made this design choice deliberately — simpler implementation, more flexible signaling (you can signal before or after changing state, signal-and-continue, etc.) — at the explicit cost of requiring waiters to re-check. They accepted spurious wakeups in the bargain, noting that since you must loop anyway, an occasional unnecessary wakeup is harmless. So the `while`-loop isn't defensive paranoia; it's the *specified contract* of the Mesa monitors that every system you'll ever use actually implements.

| | Mesa (everyone uses this) | Hoare (textbook ideal) |
| --- | --- | --- |
| who runs after `signal()` | signaler continues; waiter becomes *runnable* | waiter runs *immediately*; signaler suspended |
| lock after `signal()` | signaler keeps it | handed directly to the waiter |
| is the predicate true on wake? | not guaranteed — may be false again | guaranteed true |
| `if` or `while`? | **`while` required** | `if` would suffice (but `while` is harmless) |
| spurious wakeups | permitted | not really applicable |
| implementations | Java, POSIX, C++, Go, Rust, Win32 | rare; mostly academic / formal models |

#### Worked example: a stolen item under Mesa semantics

Three threads: consumer C1 waiting on a `not-empty` condition, consumer C2 *not* waiting (busy elsewhere), producer P. The queue is empty. Walk the Mesa interleaving: (1) P acquires the lock, pushes one item, calls `signal()`, releases the lock. Under Mesa, C1 is now *runnable* but not running; P keeps going and finishes. (2) Before the scheduler runs C1, consumer C2 finishes its other work and calls `take()`. C2 acquires the lock, checks the predicate — queue has one item, not empty! — takes the item, releases the lock. (3) *Now* the scheduler runs C1. C1 reacquires the lock inside `wait()` and returns. If C1 used an `if`, it now does `q.front()` on an empty queue — crash. If C1 used a `while`, it re-checks, sees empty, calls `wait()` again, and correctly goes back to sleep. The item it was "woken for" was legitimately consumed by C2, and that's *fine* — C1 just waits for the next one. Under Hoare semantics step (2) couldn't happen, because C1 would have run *immediately* at the `signal()` in step (1), before C2 could intervene. Since you're on a Mesa system, you loop. Always.

## The bounded buffer, end to end

Now we assemble everything into the canonical worked example of condition variables: the **bounded buffer**, also called the **producer-consumer queue**. It's a fixed-capacity FIFO shared between producer threads (which `put` items) and consumer threads (which `take` items). "Bounded" means it has a maximum size, which gives us *two* conditions to wait on, not one:

- A **consumer** that calls `take()` on an *empty* buffer must wait until it's **not empty** (a producer added something).
- A **producer** that calls `put()` on a *full* buffer must wait until it's **not full** (a consumer removed something). This is **backpressure** — the buffer's fixed size naturally throttles a fast producer to the speed of the consumers, which is exactly the cross-process flow-control story told in [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control); here we build the in-process primitive underneath it.

Two predicates means we'll use **two condition variables** on one shared lock — `not_full` and `not_empty` — which sidesteps the "signal the wrong waiter" hazard from the signal-versus-broadcast section. The producer waits on `not_full` and signals `not_empty`; the consumer waits on `not_empty` and signals `not_full`. They meet at the shared buffer.

![a graph showing a producer and a consumer meeting at one shared bounded buffer coordinated by two condition variables not-full and not-empty with no busy waiting](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-7.png)

Here it is complete and correct in **C++**, using two `std::condition_variable`s and the predicate-overload `wait` so the loop is built in:

```cpp
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class BoundedBuffer {
    std::mutex m_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::queue<T> q_;
    const size_t cap_;

public:
    explicit BoundedBuffer(size_t cap) : cap_(cap) {}

    void put(T item) {
        std::unique_lock<std::mutex> lock(m_);
        // Wait until there is room. The predicate overload loops for us,
        // absorbing spurious wakeups and re-checking after every wakeup.
        not_full_.wait(lock, [&]{ return q_.size() < cap_; });
        q_.push(std::move(item));
        lock.unlock();           // release before signaling to avoid the
        not_empty_.notify_one(); // woken consumer immediately blocking on us
    }

    T take() {
        std::unique_lock<std::mutex> lock(m_);
        not_empty_.wait(lock, [&]{ return !q_.empty(); });
        T item = std::move(q_.front());
        q_.pop();
        lock.unlock();
        not_full_.notify_one();  // we made a slot; wake one waiting producer
        return item;
    }
};
```

The same buffer in **Java**, using a `ReentrantLock` with two `Condition`s — the closest Java analog to "two condition variables on one lock," and the reason `java.util.concurrent.locks` exists alongside the older intrinsic-monitor `wait`/`notify`:

```java
import java.util.ArrayDeque;
import java.util.Queue;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class BoundedBuffer<T> {
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition notFull  = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();
    private final Queue<T> queue = new ArrayDeque<>();
    private final int cap;

    public BoundedBuffer(int cap) { this.cap = cap; }

    public void put(T item) throws InterruptedException {
        lock.lock();
        try {
            while (queue.size() == cap) {   // WHILE, not if
                notFull.await();            // releases lock, sleeps, reacquires
            }
            queue.add(item);
            notEmpty.signal();              // wake one waiting consumer
        } finally {
            lock.unlock();
        }
    }

    public T take() throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {        // WHILE, not if
                notEmpty.await();
            }
            T item = queue.remove();
            notFull.signal();                // wake one waiting producer
            return item;
        } finally {
            lock.unlock();
        }
    }
}
```

Notice the structure is identical across languages: lock, `while`-loop on the predicate calling the wait, mutate the state, signal the *other* condition, unlock. That symmetry — wait on one condition, signal the other — is what makes a bounded buffer balanced and deadlock-free.

**Go's idiom is different and instructive**, because Go gives you a better tool than a raw condition variable for this exact problem: the **buffered channel.** A channel of capacity `N` *is* a bounded buffer with the condition-variable logic already implemented inside the runtime. Sending to a full channel blocks (waits on `not-full`); receiving from an empty channel blocks (waits on `not-empty`); the runtime parks and wakes goroutines for you, at near-zero CPU, with no `while`-loop for you to get wrong:

```go
// A buffered channel IS a bounded buffer. The runtime handles the waiting,
// the wakeups, and the predicate re-checking internally. No condvar needed.
buf := make(chan Item, 8) // capacity-8 bounded buffer

// Producer: blocks here when the channel (buffer) is full.
func produce(buf chan<- Item, items []Item) {
    for _, it := range items {
        buf <- it // waits on "not full" automatically
    }
    close(buf)
}

// Consumer: blocks here when the channel is empty; ranges until closed.
func consume(buf <-chan Item) {
    for it := range buf { // waits on "not empty" automatically
        process(it)
    }
}
```

Go *does* expose `sync.Cond` with `Wait`/`Signal`/`Broadcast` for the cases channels don't cover, and `sync.Cond.Wait` is Mesa-semantics so you still loop on the predicate. But the Go community's strong default — *"share memory by communicating, don't communicate by sharing memory"* — means you usually reach for a channel and let the runtime be the monitor. That's a legitimate design answer to "how do I wait correctly": pick a higher-level primitive that already encodes the discipline. We compare the channel model head-to-head with condition variables in the [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model); the short version is that channels are condition variables with the bug-prone parts hidden inside a vetted runtime.

The buffer's behavior across its fill levels is worth seeing as a state table — it's the whole contract on one card:

![a matrix of bounded buffer fill states empty partial and full mapped to what the producer and consumer each do at that state](/imgs/blogs/condition-variables-monitors-and-waiting-correctly-8.png)

#### Worked example: backpressure in action

Run the C++ buffer with capacity 8, one producer that puts 1,000,000 items as fast as it can, and one consumer that does ~10 microseconds of work per item. The producer is far faster than the consumer can drain, so within a few microseconds the buffer fills to 8 and the producer *blocks* on `not_full`. From then on, the producer runs at *exactly* the consumer's pace — one `put` unblocks each time the consumer does one `take` and signals `not_full`. The buffer hovers at or near full; memory usage is bounded at 8 items regardless of how many million the producer wants to push. That's backpressure: the bounded buffer converts an unbounded producer into a flow-controlled one, with no explicit rate-limiting code — the blocking *is* the rate limiter. If you'd used an *unbounded* queue (no `not_full` wait), the same workload would balloon memory to a million items and likely OOM the process. The bound is a feature, and the `not_full` condition variable is what enforces it.

## Measured behavior: busy-wait versus condition variable

Time for honest measurement, the third leg of this series' contract. The claim is that busy-waiting pins a core and a condition variable doesn't. Let's quantify, and let's be careful about *how* we measure so the numbers mean something.

**The experiment.** One producer, one consumer, the consumer waiting for items. We build the consumer two ways. Version A busy-waits: `while (queue.empty()) { /* spin */ }` with no sleep, no yield. Version B uses a condition variable: `while (queue.empty()) cv.wait(lock);`. The producer puts one item every 10 milliseconds — a deliberately *slow* producer, so the consumer spends almost all its time *waiting*, which is exactly where the two strategies differ. We let each version run for 60 seconds and read CPU utilization from the OS (e.g. `/proc/<pid>/stat` on Linux, or a per-thread monitor). We warm up first, run several times, and report the steady state, because a single sample of a scheduler-driven workload is noise.

**How to read CPU%.** "100% CPU" means one hardware thread fully occupied. A pure busy-wait on one core shows ~100% for that thread the entire time it's waiting. A blocked thread shows ~0% — it's not on any run queue, so it accrues no CPU time. The numbers below are representative of a modern x86 laptop core; *your* exact figures will differ with CPU, OS scheduler, and load, so treat the magnitudes — "a full core" versus "a rounding error" — as the durable result, not the third decimal place.

| Strategy | CPU while idle (1 waiter) | wakeup latency | scales to 1000 waiters? |
| --- | --- | --- | --- |
| busy-wait (`while(empty){}`) | ~100% of one core | ~immediate (already running) | catastrophic — 1000 cores' worth of spin |
| busy-wait + `yield`/`pause` | ~30–100% (scheduler-dependent) | near-immediate | still burns cores, less aggressively |
| condition variable (`cv.wait`) | ~0% (often < 1%) | ~1–20 µs (kernel wakeup path) | fine — 1000 parked threads cost ~nothing idle |

The headline: the busy-wait holds a core at ~100% to do *nothing*, while the condition variable holds the thread at ~0% and wakes it in single-digit-to-low-double-digit microseconds. That microsecond wakeup latency is the *only* thing the condition variable gives up versus the spin, and for a producer that fires every 10 milliseconds, paying ~10 µs to wake means the consumer reacts in ~0.1% of the inter-arrival time — utterly negligible. You burned a whole core to shave 10 µs off a 10,000 µs wait. That is a catastrophic trade, and it's why busy-waiting is almost always wrong.

**Measuring this honestly is harder than it looks**, and it's worth saying how, because a careless benchmark will mislead you. First, *warm up*: the JVM JIT-compiles hot code, the OS faults in pages, caches fill — your first few iterations are not representative, so discard them. Second, *run many times and look at the distribution*, not one number: a condition-variable wakeup latency has a long tail (the scheduler might not run your thread immediately if the core is busy), so report a median and a high percentile, not a single mean. Third, *name the confound*: the OS scheduler is a giant source of nondeterminism here — if another process is hammering the same cores, your wakeup latency balloons through no fault of the condition variable. Fourth, *measure CPU at the thread granularity if you can* — a process-level reading lumps producer and consumer together and hides which thread is actually spinning. And fifth, *don't trust a precise figure you can't reproduce*: if you measure "7.3 µs wakeup latency" once, the honest claim is "single-digit microseconds, with a tail into the tens," because that 7.3 will move with kernel version, CPU, and load. The durable, reproducible result is the *order of magnitude gap* — a full core versus essentially nothing — which holds across every machine you'll run it on, and which is the only number worth putting in a design decision.

**When does the spin actually win?** Only when the wait is *expected to be extremely short* — shorter than the cost of a context switch (a few microseconds). If you'll almost certainly be unblocked within, say, 100 nanoseconds, then parking and unparking the thread (a couple of microseconds of syscall and scheduler overhead, plus cache effects) costs *more* than just spinning through those 100 ns. That's exactly the regime where a **spinlock** beats a blocking lock, which we covered in [how a lock is built](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks). The sophisticated real-world answer is the **adaptive** or **hybrid** wait: spin for a bounded handful of microseconds (in case the condition flips almost immediately), and only *then* fall back to a true `cv.wait()` / futex sleep. Production mutexes and condition-variable implementations (glibc's, the JVM's) do exactly this internally. But for waiting on a *condition* — "is the queue non-empty?" — where the wait time is unknown and frequently long, the blocking condition variable is the right default, and the naive spin is the trap.

#### Worked example: the cost of getting it wrong at scale

A service runs a thread pool of 64 worker threads, each waiting for jobs from a shared queue. Suppose someone wrote the wait as a busy-loop. With 64 threads each spinning on an empty queue, all 64 cores of the box sit at ~100%, the load average reads 64, the fans roar, and *no work is being done* — the box is maxed out polling. Monitoring fires a CPU alert; an engineer logs in, sees 100% across the board, and reasonably assumes the service is overloaded and *adds more instances*, which also sit at 100% doing nothing. The actual fix is one word: change the busy-loop to a `cv.wait()` (or, in practice, use a `BlockingQueue` / channel that does it for you). After the fix, the same idle service sits at ~0% CPU, the alert clears, and the autoscaler scales *back down*. The lesson: a busy-wait doesn't just waste a core, it *lies to your operational signals* — it makes an idle service look saturated. Wait correctly and your CPU graph tells the truth.

## Pitfalls

Even with the atomic-wait and `while`-loop rules internalized, there are a handful of recurring mistakes that bite real code. Here are the ones worth tattooing on the inside of your eyelids.

**Lost wakeup from forgetting to hold the lock when changing state.** The condition variable is stateless; the *real* condition lives in your shared state under your lock. If you change the state *without* holding the lock and then signal, a waiter can observe stale state and miss the wakeup — the exact lost-wakeup race from the non-atomic `wait`, reintroduced through the back door. **Rule: always hold the lock while you change the state that the predicate reads.** Then signal. The signal itself can sometimes be done after releasing the lock (see below), but the *state change* must be under the lock.

**Calling `signal()` while not holding the lock — and the ordering subtlety.** Whether you should hold the lock when you *signal* is genuinely nuanced and language-dependent. POSIX *permits* signaling without holding the lock and notes it can be a performance win (the woken thread doesn't immediately block trying to reacquire a lock the signaler still holds — this is the **wait-morphing** concern). C++ similarly allows `notify_one()` outside the lock. *But* signaling outside the lock opens a window for a different lost wakeup: if you do `unlock(); signal();` and a brand-new waiter checks the predicate and calls `wait()` in between your unlock and your signal, sequencing gets subtle. The safe, portable rule for application code: **hold the lock while you change the state; you may release the lock immediately before `signal()` for performance, but if you're unsure, signal while still holding the lock — it's always correct, just possibly slightly slower.** Java's intrinsic monitors *require* you to hold the lock to call `notify()` (it throws `IllegalMonitorStateException` otherwise), which removes the choice and the footgun.

**Using an `if` instead of a `while`.** We've belabored this, but it's the single most common condition-variable bug in real codebases, so it earns a repeat: an `if` is correct *only* under pure Hoare semantics, which your system does not implement. On Mesa systems (all of them), an `if` is a latent bug that fires on spurious wakeups, broadcasts, and stolen items. Use `while`. Or use the predicate-overload (`cv.wait(lock, pred)`, `wait_while`) that bakes the loop in.

**Signaling the wrong waiter with one shared CV and mixed predicates.** Covered earlier: if producers and consumers wait on the *same* condition variable with *different* predicates, `signal()` can wake the wrong kind of waiter, who re-checks, finds nothing for it, and sleeps again — while the waiter who *could* proceed never gets woken. Use one condition variable per predicate (the bounded buffer's `not_full` + `not_empty`), or `broadcast()`.

**Forgetting that `wait()` can be interrupted or time out.** In Java, `wait()` throws `InterruptedException`; in many languages there's a `wait_for` / `wait_until` with a timeout that returns a status. A timed wait that returns must *still* re-check the predicate (it may have returned due to timeout *or* a real signal *or* spuriously). The `while`-loop with a deadline handles all three; an `if` does not.

**Holding the lock across slow work, including across `wait`.** A monitor serializes every method behind one lock. If you do expensive work (I/O, a network call) while holding the monitor lock, every other thread is blocked out of the monitor for that whole duration — a self-inflicted serialization bottleneck. Do the slow work *outside* the critical section; hold the lock only for the brief moment you inspect or mutate the shared state. (`wait()` itself is fine — it *releases* the lock while sleeping; that's the point.)

## Case studies / real-world

**The JDK's own `notify` versus `notifyAll` guidance.** Java's core library documentation and *Java Concurrency in Practice* (Goetz et al.) both spend real ink warning that `notify()` (wake one) is an optimization you must *earn*: it's only safe when all waiters wait on the same condition with interchangeable predicates and one notification enables at most one waiter to proceed. The book's standing advice is "prefer `notifyAll()` unless you can prove `notify()` is safe," precisely because a misplaced `notify()` that wakes the wrong waiter produces a hang that's nearly impossible to reproduce. This is the signal-versus-broadcast trade-off, codified as a hard-won library-design lesson: correctness first (`notifyAll`), optimize to `notify` only with proof.

**Spurious wakeups as a *specified* behavior, not a bug report.** The POSIX threads standard and the Linux `pthread_cond_wait`/`pthread_cond_timedwait` man pages explicitly document that "spurious wakeups ... may occur" and instruct that the predicate must be re-tested in a loop. This is unusual — a standard *promising* that a function may do something seemingly wrong — and it exists because the implementers found that *forbidding* spurious wakeups would force slower, more complex code for a guarantee that's cheap to provide at the call site (just loop). The C++ standard inherited the same language for `std::condition_variable::wait`. So the `while`-loop isn't a workaround for buggy libraries; it's the contract the libraries deliberately wrote, trading a tiny re-check cost for implementation freedom.

**The LMAX Disruptor and the cost of the wait strategy.** The LMAX Disruptor — a high-performance inter-thread messaging library famous in low-latency finance — treats the *wait strategy* as a first-class, pluggable choice, and the menu is exactly the trade-off this post is about. Its `BlockingWaitStrategy` uses a lock and condition variable (low CPU, higher latency — the right default). Its `BusySpinWaitStrategy` spins (lowest latency, burns a core — only for dedicated cores where you'd rather melt a core than ever pay a microsecond of wakeup). Its `YieldingWaitStrategy` and `SleepingWaitStrategy` sit in between. The Disruptor's authors made the spin-versus-block decision *configurable* precisely because there is no universal right answer — it depends on whether you have a spare core to burn and how allergic you are to wakeup latency. For the overwhelming majority of systems, the blocking condition-variable strategy is correct; the busy-spin is a specialized tool for pinned cores in latency-critical paths. That's the same conclusion our measurement section reached, validated by a production system whose entire reason for existing is low latency.

## When to reach for this (and when not to)

A condition variable is the right tool when **a thread must wait for a state established by another thread, the wait may be of unknown or non-trivial duration, and you need near-zero CPU while waiting.** Producer-consumer queues, thread pools waiting for jobs, a worker waiting for a connection-pool slot, any "block until ready" handoff — these are the home turf of the condition variable. It's the cheapest correct way to say "wake me when the world changes."

**Don't reach for a raw condition variable when a higher-level primitive already encodes the discipline.** If your language gives you a `BlockingQueue` (Java), a buffered channel (Go, Rust's `crossbeam`/`tokio` channels), or a thread-safe bounded queue, *use it* — it's a vetted monitor with the `while`-loop, the two-condition-variable dance, and the signal/broadcast choice already correct inside it. You should write a raw condition variable only when you're building such a primitive, or when your waiting condition is genuinely custom (a complex predicate over several variables that no off-the-shelf queue captures).

**Don't busy-wait** unless you can prove the wait is reliably *shorter than a context switch* (sub-microsecond) and you have a dedicated core to burn — the spinlock / pinned-core regime. For waiting on a *condition* whose timing you don't control, busy-waiting is the wrong answer; it pins a core and lies to your monitoring.

**Don't use a condition variable when a one-shot or counting primitive fits better.** If you're waiting for a *count* of events (N tasks to finish) reach for a [semaphore, barrier, or latch](/blog/software-development/concurrency/semaphores-barriers-and-latches) — a `CountDownLatch` for "wait until N things are done" is clearer and remembers its count, which a stateless condition variable does not. If you're coordinating a fixed number of threads at a rendezvous point, a barrier is the right primitive. Condition variables are for *conditions over shared state*; counting and one-shot signaling have purpose-built tools that are harder to get wrong.

**Don't reach for a condition variable to fix a CPU-bound throughput problem** — that's not what it's for. It's a *waiting* primitive, not a parallelism primitive; it makes idle threads cheap, not busy threads faster. If your problem is "use all my cores," that's the parallelism story, not the condition-variable story.

This whole decision — lock, atomic, channel, semaphore, condition variable — is the subject of the series capstone, [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model); the condition variable is one well-defined tool in that kit, the one for "block efficiently until a condition holds."

## Key takeaways

- **Busy-waiting pins a core at ~100% to do nothing.** A condition variable parks the thread at ~0% CPU and wakes it in single-digit microseconds. For any wait longer than a context switch, the condition variable wins decisively — and an idle busy-wait also lies to your CPU monitoring by making an idle service look saturated.
- **`wait()` must atomically release the lock and sleep.** If those two steps could separate, a signal can land in the gap and be lost forever — the lost-wakeup hang. Never roll your own "unlock then sleep"; always use the library's `wait()`, which fuses them (futex `FUTEX_WAIT` on Linux).
- **Always loop on the predicate: `while (!predicate) wait();` — never `if`.** Three independent reasons force this: spurious wakeups, multiple waiters sharing one item, and a stolen item under Mesa hand-off. The C++ `wait(lock, pred)` and Rust `wait_while` overloads are the loop in disguise.
- **Condition variables are stateless and have amnesia.** A signal with no current waiter evaporates. The real condition must live in *your* shared state, under *your* lock; the condition variable is only the wakeup channel.
- **`signal()` wakes one, `broadcast()` wakes all.** Use `signal()` when one event suits any single waiter; use `broadcast()` when many can proceed or predicates differ on one CV. A careless `broadcast` on every event causes a thundering herd; a careless `signal` with mixed predicates wakes the wrong waiter and hangs.
- **A monitor bundles a lock with its condition variables** and protects an invariant over shared state. Java's `synchronized`/`wait`/`notify`, C++'s `mutex`+`condition_variable`, Rust's `Mutex<T>`+`Condvar`, and Java's `ReentrantLock`+`Condition` are all monitors.
- **Every real system uses Mesa semantics**, where `signal()` only makes a waiter *runnable* and the predicate may be false by the time it runs. That — not just spurious wakeups — is why the `while`-loop is non-negotiable. Hoare semantics (immediate hand-off, guaranteed-true predicate) is a textbook ideal nobody ships.
- **The bounded buffer is the canonical example**: one lock, two condition variables (`not_full`, `not_empty`), wait on one and signal the other. It gives you backpressure for free — the bound throttles a fast producer to consumer speed and caps memory.
- **Prefer a higher-level primitive when one fits** — a `BlockingQueue`, a buffered channel, a `CountDownLatch`. They're vetted monitors with the loop, the two-CV dance, and the signal choice already correct inside.

## Further reading

- **Butenhof, *Programming with POSIX Threads*** — the definitive treatment of `pthread_cond_wait`, the atomic release-and-sleep contract, spurious wakeups, and the predicate loop, with the C-level mechanism spelled out.
- **Goetz, Peierls, Bloch, Bowbeer, Holmes, Lea, *Java Concurrency in Practice*** — chapter 14 ("Building Custom Synchronizers") is the canonical treatment of condition queues, the `notify` versus `notifyAll` decision, and the bounded-buffer monitor in Java.
- **Lampson & Redell, "Experience with Processes and Monitors in Mesa" (1980)** — the original paper that introduced Mesa semantics and deliberately chose signal-and-continue plus the re-check loop. The source of the rule you now know by heart.
- **Hoare, "Monitors: An Operating System Structuring Concept" (1974)** — the original monitor proposal with Hoare (signal-and-urgent-wait) semantics, for the contrast that explains why Mesa won in practice.
- **Williams, *C++ Concurrency in Action*** — `std::condition_variable`, the predicate-overload `wait`, `notify_one`/`notify_all`, and a thread-safe queue built from them.
- **The Linux `pthread_cond_wait(3)` and `futex(2)` man pages** — the spurious-wakeup wording and the kernel mechanism (`FUTEX_WAIT`/`FUTEX_WAKE`) underneath every condition variable on Linux.
- **Within this series**: start at [why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), build up through [mutual exclusion and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) and [how a lock is built](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks), continue to [semaphores, barriers, and latches](/blog/software-development/concurrency/semaphores-barriers-and-latches), and see the whole decision framework in [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

---
title: "Data Races vs Race Conditions: A Precise Distinction"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a data race is undefined behavior the compiler exploits, why a race condition can exist with zero data races, and why works on my machine is a lie."
tags:
  [
    "concurrency",
    "parallelism",
    "data-race",
    "race-condition",
    "undefined-behavior",
    "memory-model",
    "thread-safety",
    "rust",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-1.png"
---

A junior engineer once handed me a bug report that read, in full: "the cache sometimes returns two different objects for the same key." The cache was a `ConcurrentHashMap`. The engineer had read the Javadoc — thread-safe, every method synchronized, the gold standard — and concluded the bug had to be somewhere else. It was not somewhere else. The code did this: check whether the key is present, and if it is not, compute the value and put it. Two threads checked at the same instant, both saw "absent," both computed, both put. The map did exactly what it promised on every single method call, and the program was still wrong.

That bug has no data race. Every access to the map was synchronized by the map itself. There was no torn read, no undefined behavior, no memory-model violation — and the tools that hunt for data races would run clean over it forever. It was a *race condition*: a correctness bug whose outcome depends on the order in which independently-correct operations interleave. Meanwhile, two doors down, another team had a counter that lost increments under load. Their bug was the opposite kind — a genuine data race, two threads doing `count++` with no lock, which is not merely "a bug" but *literally undefined behavior* in C, C++, Java, Go, and Rust, the kind the compiler is allowed to miscompile into something that loses updates, tears bytes, or invents a value out of thin air.

These two phenomena get used interchangeably in standups, code reviews, and postmortems, and the conflation costs real money, because the *fix* for each is different and the *detection* for each is different. The figure below is the whole post in one frame: a two-by-two where "has a data race" and "is a race condition" are independent axes. There is a cell for each, including the one most people swear is impossible — a data race that is benign-looking, and a race condition with no data race at all.

![A two by two matrix with rows has a data race and no data race and columns is a race condition and not a race condition, each cell holding a concrete example](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-1.png)

By the end of this post you will be able to look at a concurrency bug and say, precisely, which kind it is — and therefore which tool will find it and which fix will close it. We will define a data race down to its four formal clauses, define a race condition as an ordering bug, walk a data race that the compiler is allowed to turn into garbage, walk a race condition where every individual operation is perfectly synchronized and the program is still wrong, and finish on why Rust's type system can ban one of these at compile time and *cannot* ban the other. This is the post in the series that the others lean on. If you have not yet read [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) or the sibling on [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition), this is a good place to anchor the vocabulary you will use for the rest of the series.

## The shared-state frame, and why two words are not one word

The spine of this whole series is a single sentence: shared mutable state plus nondeterministic scheduling is the hazard. Every concurrency bug is some thread reading or writing a piece of memory while another thread reads or writes it, with the scheduler free to interleave them in an order you did not anticipate. The discipline that tames it has three moves — name what is shared and mutable, establish a happens-before order over every access to it, then buy that order with the cheapest mechanism that works.

The reason "data race" and "race condition" are different words is that they sit at *different layers of that frame*. A data race is a violation at the **memory-model layer**: you failed to establish a happens-before edge between two conflicting accesses, so the language standard withdraws all guarantees about what your program means. A race condition is a violation at the **logic layer**: you established happens-before on each individual access — every read and write is properly synchronized — but the *sequence* of those individually-safe operations can interleave into a state your invariant forbids. One is a contract you broke with the compiler. The other is a contract you broke with yourself.

Here is the trap that makes them feel like one thing: many bugs are *both*. The `count++` example is both a data race (the load and store are unsynchronized) and a race condition (the lost-update interleaving is an ordering bug). When the only races you have ever debugged live in that overlap cell, it is natural to assume the two words are synonyms. They are not, and the cells *off* the diagonal — data race that is not a race condition, race condition that is not a data race — are where the expensive, baffling, "but the library is thread-safe!" bugs live.

Let me give you the precise definitions first, because precision is the entire point of this post, and then we will spend the rest of it building intuition around the edges.

## Defining a data race precisely

A **data race** occurs when **two accesses to the same memory location** happen **concurrently**, **at least one of them is a write**, and there is **no happens-before relationship** ordering one before the other. All four clauses must hold simultaneously. Drop any one and it is not a data race. Let me take them one at a time, because every word is load-bearing.

![A matrix listing the four clauses same location concurrent at least one write and no happens before with whether each is required and an example](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-4.png)

**Same memory location.** Two threads touching the *same* object field, the same array element, the same global. If they touch different locations there is no conflict — but beware that "different" is at the byte level, which is why false sharing and adjacent bit-fields can surprise you. Two `bool` flags packed into the same machine word can race even though they look independent in the source.

**Concurrently.** The two accesses are not ordered with respect to each other by the program's synchronization. Concurrency here is a *formal* property, not a wall-clock one: even on a single core where the accesses never truly overlap in time, they are "concurrent" in the memory-model sense if nothing in the program orders them.

**At least one is a write.** Two reads of the same location never conflict — reading shared immutable data from a thousand threads is perfectly safe, which is the entire reason immutability is such a powerful concurrency tool. The hazard requires a writer, because a write is what makes the value depend on *when* you look.

**No happens-before relationship.** This is the clause people skip, and it is the one that actually matters. Happens-before is the partial order the memory model defines over all memory operations in an execution. If operation A happens-before operation B, then B is guaranteed to see A's effect. You create happens-before edges by synchronizing: unlocking a mutex happens-before the next lock of the same mutex; a `release` store happens-before the `acquire` load that reads it; `thread.start()` happens-before everything the new thread does; everything a thread does happens-before another thread's `join` on it; a write to a `volatile` (Java) / `atomic` (C++) variable happens-before the read that observes it. If *any* such edge orders your two conflicting accesses, there is no data race. If there is *no* such edge, there is.

So the formal statement is compact: a data race is a pair of conflicting accesses (same location, at least one write) that are *unordered* by happens-before. And the consequence — this is the part engineers underestimate — is not "you might read a stale value." The consequence is that the program has **undefined behavior**. The C++ standard says it outright (`[intro.races]`): "Any such data race results in undefined behavior." The C standard says the same. Go's memory model says a program with a data race "may exhibit any behavior." Java is the careful exception we will get to. Undefined behavior means the standard imposes *no requirements at all* — the compiler may assume data races do not happen and optimize on that assumption, which is how a racy read gets hoisted out of a loop, or a racy write gets reordered past a lock, or a value you never stored materializes.

#### Worked example: walking the happens-before clause

Two threads, one shared `int x` initialized to 0. Thread 1 does `x = 1`. Thread 2 does `print(x)`. Is this a data race? It depends entirely on the fourth clause.

- If there is no synchronization between them — both started, then left to the scheduler — then the write `x = 1` and the read of `x` are unordered by happens-before. **Data race. Undefined behavior.** The print might show 0, might show 1, and on a weak-memory machine with an aggressive compiler might show something you cannot explain.
- If Thread 1 does `x = 1` *before* `start(Thread 2)`, then thread-start gives you a happens-before edge: the write happens-before the read. **No data race.** The print is guaranteed to show 1.
- If both go through a mutex — Thread 1 locks, writes, unlocks; Thread 2 locks, reads, unlocks — then the unlock-before-lock edge orders them (in whichever direction the lock is actually acquired). **No data race.** The read sees either the value before or the value after the write, but it is always a real, complete value, never garbage.

Same two lines of code. The presence or absence of a single happens-before edge is the entire difference between a defined program and one the compiler may turn into nonsense. *That* is why the fourth clause is the one that matters.

It is worth being concrete about *what creates a happens-before edge*, because "synchronize the access" is too vague to act on. Happens-before is built from a small, fixed set of edge-makers, and every correct concurrent program is, at bottom, a careful arrangement of these edges so that every conflicting access pair is covered by at least one of them. The table catalogs the ones you will actually use, with the cost of buying that edge, because the cost is what drives the engineering choice:

| Edge-maker | Establishes | Mechanism | Rough cost |
| --- | --- | --- | --- |
| Mutex unlock then lock | Everything before the unlock happens-before everything after the next lock | Memory barrier in the lock/unlock | ~20–40 ns uncontended, far more contended |
| `release` store then `acquire` load | The release-side writes happen-before the acquire-side reads | A single barrier on each side, no lock | A few ns over a plain access |
| `volatile` write then read (Java) / `atomic` (C++) | The write happens-before the read that observes it | Compiler + CPU ordering barriers | Cheap read, slightly costlier write |
| `thread.start()` | Everything before start happens-before everything the new thread does | Thread creation already synchronizes | Free relative to spawning a thread |
| `thread.join()` | Everything the joined thread did happens-before the join returning | The join already synchronizes | Free relative to the join |
| Channel send then receive | The send-side state happens-before the receiver observing it | Channel internals synchronize | Channel overhead, usually a lock or CAS |

Notice that the *cheapest* correct edge for a given access depends on the access shape — a single shared word wants an atomic (a few nanoseconds), a compound action wants a lock (tens of nanoseconds), a hand-off between stages wants a channel. The series' third discipline — buy the order with the *cheapest* mechanism that works — is precisely the act of picking the right row of this table for each conflicting access pair. The happens-before relation is also *transitive*: if A happens-before B and B happens-before C, then A happens-before C, which is how a chain of these edges propagates an ordering all the way across a program. The formal model is developed in [memory models, sequential consistency, and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before); here the point is only that "no happens-before edge" — the fourth clause of the data-race definition — means *none of these edge-makers covers your access pair*.

## Defining a race condition precisely

A **race condition** is a flaw where the correctness of a program depends on the *relative timing or interleaving* of operations by multiple threads. Put differently: there exists at least one interleaving of the threads' operations that violates an invariant the program is supposed to maintain. The defining feature is that the bug lives in the *sequence of operations*, not in any single operation's memory access.

Notice what this definition does *not* mention: memory locations, writes, happens-before, undefined behavior. None of it. A race condition is a property of the *logic*, and it is entirely possible — this is the crux — for every individual operation in that logic to be perfectly synchronized, properly ordered, free of any data race, and for the *composition* of those operations to still be wrong.

The canonical shape is **check-then-act**: you check a condition, then act on the result of the check, and between the two steps another thread changes the thing you checked. Your action is now based on a fact that is no longer true. The `ConcurrentHashMap` bug from the intro is exactly this — `if (!map.containsKey(k)) map.put(k, compute())`. The `containsKey` and the `put` are each atomic and each safe. But there is a *gap* between them, and in that gap another thread can slip in and do its own `put`, so your `containsKey` returns a stale answer and your subsequent `put` clobbers their value.

Other shapes of race condition: **lost update** (read a value, modify it, write it back, while another thread did the same in between, so one modification vanishes — even with atomic reads and atomic writes, if they are not atomic *together*); **ordering dependence** (the program is correct only if A's write lands before B's read, but nothing enforces that); **two-step invariant violation** (you transfer money by debiting one account and crediting another, and a reader between the two steps sees money that has temporarily vanished). In every case, the individual operations can be flawless and the trouble is in the *between*.

The key intuition to hold onto here: a race condition is about the *granularity of atomicity*. You made each operation atomic, but the unit of correctness was the *combination* of operations, and you left that combination non-atomic. The fix is never "make the operations more thread-safe" — they already are. The fix is to make the *compound action* atomic: hold a lock across the whole check-and-act, or use a primitive that does the compound action in one indivisible step (`putIfAbsent`, `compareAndSet`, a database transaction).

There is a useful formal way to see why this is unavoidable without explicit atomicity. Model an execution as an interleaving — a single global sequence that merges the operations of all threads in *some* order consistent with each thread's program order. If your program has `n` threads each doing `k` operations, the number of distinct interleavings is astronomically large (it grows roughly like the multinomial coefficient $\binom{nk}{k, k, \ldots, k}$, which for even small `n` and `k` is in the thousands or millions). A race condition exists precisely when *at least one* of those interleavings drives the program into a state your invariant forbids. Your tests exercise a vanishing fraction of that space — usually the handful of "benign" orderings that a lightly-loaded scheduler happens to produce. The malign interleaving is still in the space; it just has not fired yet. This is why a race condition is a property of the *set of all allowed executions*, not of any one run, and why "I ran it and it worked" is such weak evidence: you sampled one point from a space of millions.

The structural cure is to *shrink the space*. Making the compound action atomic does not change the per-operation correctness — it removes the malign interleavings from the set entirely, because no other thread can observe or modify the intermediate state. A lock held across the check-and-act collapses all the interleavings where another thread slips into the gap into a single ordering where it cannot. A `compareAndSet` collapses the read-then-write into one indivisible step that either fully succeeds against the value it read or retries. Either way, the fix is *eliminating interleavings*, which is a fundamentally different operation from *establishing a happens-before edge on an access* (the data-race fix). One prunes the schedule space; the other orders a memory access. They are not the same move, which is the whole reason the two bug families have different fixes.

## The Venn diagram: four cells, not one

Now we can populate the two-by-two from the intro precisely. Two independent axes — "has a data race" (a memory-model question) and "is a race condition" (a logic question) — give four cells.

**Both (the overlap, the famous one).** `count++` with no synchronization. It is a data race: the load of `count`, the increment, and the store are three unsynchronized operations on a shared location, at least one a write, no happens-before edge. It is *also* a race condition: even if each were atomic, the read-modify-write as a whole is non-atomic, so two threads can both read 5, both write 6, and you lose an increment. This is the cell everyone has debugged, and the reason the two terms feel identical.

**Data race only (UB, but not a logic ordering bug in the usual sense).** A status flag that one thread writes and another polls in a loop, with no atomics and no lock — `bool done = false; while (!done) {}` in one thread, `done = true` in another. There is no invariant about *ordering* being violated in the check-then-act sense; the program's *intent* is simply "stop when done is set." But the access is a data race (plain read and plain write of a shared `bool`, no happens-before), so it is undefined behavior, and the compiler is allowed to hoist `done` out of the loop and spin forever, or tear the read on a platform where `bool` writes are not atomic. The *logic* is fine; the *memory model* is violated. We will dwell on this cell because it is the one people insist cannot hurt them.

**Race condition only (no data race at all).** The `ConcurrentHashMap` check-then-act. Every access goes through a thread-safe structure that establishes happens-before internally, so there is no data race anywhere — a data-race detector finds *nothing*. Yet the compound check-then-act is non-atomic, so the program is wrong. This cell is the one that breaks people's mental model, because they were taught "use a thread-safe collection and you are fine," and here they used one and are *not* fine.

**Neither.** A correctly written program: the compound action is atomic (one `putIfAbsent`, or a lock held across the whole check-and-act), every access is synchronized. No data race, no race condition. This is the goal, and it requires getting *both* layers right — establishing happens-before on every access *and* choosing the right atomicity granularity for your invariants.

The off-diagonal cells are the lesson. "Data race only" means you can have UB with logically-correct intent. "Race condition only" means you can have a correctness bug with zero UB and perfectly thread-safe building blocks. Conflate the two words and you will reach for the wrong tool every time you land off the diagonal.

It helps to line the four cells up against the concrete examples and the fix each demands, because the *fix* is the thing the distinction is actually for:

| Cell | Example | Data race? | Race condition? | The fix |
| --- | --- | --- | --- | --- |
| Both | `count++` with no lock | Yes — unsynchronized load/store | Yes — non-atomic read-modify-write | A lock or an atomic increment (closes both) |
| Data race only | `bool done` flag polled, plainly written | Yes — plain read/write, no edge | No — intent has no ordering invariant | Make the flag atomic / `volatile` |
| Race condition only | Check-then-act on a thread-safe map | No — every access synchronized | Yes — compound action non-atomic | Widen the atomic unit (`computeIfAbsent`) |
| Neither | One atomic `putIfAbsent` under a lock | No | No | Nothing — already correct |

Read the table top to bottom and the structure jumps out: the two "only" rows have *opposite* answers in the two columns and *opposite* fixes. The "data race only" fix touches the *access* (make it atomic); the "race condition only" fix touches the *granularity* (make the compound action atomic). Apply the wrong row's fix and you either over-engineer a benign flag into a mutex or, worse, sprinkle `volatile` on a check-then-act and ship a bug you have convinced yourself you fixed.

## A data race that is not (obviously) a race condition

Let me make the "data race only" cell concrete, because it is the most counterintuitive, and because it is where the phrase "works on my machine" turns into an outright lie.

Here is the classic stop flag in C++, written the way a lot of production code was written before people internalized the memory model:

```cpp
#include <thread>
#include <chrono>
#include <cstdio>

bool done = false;   // plain bool, shared, no atomic

void worker() {
    while (!done) {           // read done every iteration... or so we hope
        // do a unit of work
    }
    printf("worker stopped\n");
}

int main() {
    std::thread t(worker);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    done = true;              // plain write, no synchronization
    t.join();
    return 0;
}
```

The *intent* is innocent: the main thread sets a flag, the worker notices and stops. There is no check-then-act, no lost update, no invariant about ordering being subtly violated. If you reason about it as a sequence of source-level steps, it looks correct. But the read of `done` in the loop and the write of `done` in `main` are conflicting accesses (same location, one is a write) with **no happens-before edge** between them. That is a data race. That is undefined behavior. And undefined behavior is not a polite suggestion — it changes what the compiler is *allowed* to produce.

What the optimizer sees is a loop that reads `done` but never writes it, with no synchronization that could let another thread change it. Under the as-if rule, with the legal assumption that data races do not occur, the compiler may *hoist the load out of the loop*: read `done` once, into a register, and spin on the register forever. The transformation looks like this:

```cpp
// What the compiler may legally generate from the racy loop:
void worker_optimized() {
    bool cached = done;       // load ONCE
    while (!cached) {         // never reload; pure register spin
        // ...
    }
}
```

Now the main thread's `done = true` writes to memory that the worker never reads again. The worker spins forever. This is not a theoretical worry — it is a well-documented consequence of compiling a racy spin loop at `-O2`, and it is exactly why `volatile` (in C/C++, which forces a reload but does *not* establish happens-before) and then `std::atomic` (which does both) exist. The fix is to make the flag atomic so there is no data race and so the load cannot be hoisted:

```cpp
#include <atomic>

std::atomic<bool> done{false};

void worker() {
    while (!done.load(std::memory_order_acquire)) {
        // do a unit of work
    }
}
// main: done.store(true, std::memory_order_release);
```

Now is this "data race only" cell *also* a race condition? You can argue the original spin loop has an ordering bug — it depends on the write becoming visible — but the crucial point is that the *symptom* (infinite spin) is produced purely by the data-race UB, by the compiler exploiting the absence of a happens-before edge. You did not write a check-then-act. You did not write a lost update. You wrote a correct-looking flag, and the *memory model* alone turned it into a hang. A race-condition lens would never find this; only a data-race lens does. That is the separation made visible.

#### Worked example: the value you never stored

Here is a sharper version of "the compiler can do anything." Consider a shared `int x` read racily, where the compiler decides — legally — to re-read it:

```c
int x;  // shared, racily written by another thread

int compute(void) {
    if (x > 10) {
        // ... lots of code ...
        return x * 2;   // x re-read here; UB lets it differ from the check
    }
    return 0;
}
```

You checked `x > 10`, so you "know" `x` is at least 11, so `x * 2` is at least 22. But `x` is read twice — once in the condition, once in the multiply — and because the access is racy, the compiler is under no obligation to give you the same value both times, and another thread may have written `x = 3` in between. Your "guaranteed at least 22" returns 6. Worse, real compilers sometimes optimize on the *first* read's assumption (`x > 10`) while a later re-read produces a smaller value, so you get a return value that is *inconsistent with the branch you took*. In a security-sensitive context, that is the source of real vulnerabilities — a bounds check that passes, followed by an access using a value that no longer satisfies the bound. The memory model gave you no guarantee that a single source-level variable corresponds to a single, stable value, because you raced on it.

## A race condition with no data race

Now the other off-diagonal cell, the one from the intro. We will write it in two languages where the idioms diverge, because the whole point is that the building blocks are *correct* and the composition is *wrong* — and that looks different in Java (a thread-safe collection) than in Go (a map plus an explicit mutex).

First, Java, with `ConcurrentHashMap`:

```java
import java.util.concurrent.ConcurrentHashMap;

class Cache {
    private final ConcurrentHashMap<String, Object> map = new ConcurrentHashMap<>();

    // BUG: check-then-act is not atomic, even though each call is.
    Object getOrCompute(String key) {
        if (!map.containsKey(key)) {     // call 1: atomic, safe
            Object value = expensiveCompute(key);
            map.put(key, value);          // call 2: atomic, safe
            return value;
        }
        return map.get(key);
    }

    Object expensiveCompute(String key) { /* ... */ return new Object(); }
}
```

Run a race detector on this and it finds nothing — every access to `map` is synchronized internally by the map, there is no unordered conflicting access, no data race exists. Yet two threads calling `getOrCompute("k")` at once can both see `containsKey` return false, both call `expensiveCompute` (so you do the expensive work twice — a waste at best), and both `put` (so one value silently overwrites the other — a correctness bug if callers hold a reference to "the" cached object and now hold two different ones). The bug is the *gap* between call 1 and call 2. The timeline makes the interleaving concrete:

![A timeline where T1 contains returns false then T2 contains returns false then both put producing a lost insert](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-3.png)

The fix is to make the compound action atomic. `ConcurrentHashMap` gives you exactly the right primitive — `computeIfAbsent`, which performs the check and the act as one indivisible operation (and, as a bonus, computes the value at most once even under contention):

```java
Object getOrCompute(String key) {
    return map.computeIfAbsent(key, this::expensiveCompute);  // atomic compound action
}
```

The before-and-after of the two structures — the racy two-call window versus the single atomic call — is worth seeing side by side:

![A before and after comparison showing two separate calls with a racy gap versus a single atomic putIfAbsent call that inserts exactly once](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-2.png)

Now Go, where there is no built-in concurrent map with a `computeIfAbsent`, so the idiom is a plain `map` guarded by a `sync.Mutex` — and the *same* race condition appears, except now it is even clearer that thread-safety of each access does not save you:

```go
package main

import "sync"

type Cache struct {
    mu sync.Mutex
    m  map[string]any
}

// BUG: each lock/unlock is correct, but the compound action is split across two of them.
func (c *Cache) GetOrCompute(key string, compute func() any) any {
    c.mu.Lock()
    _, ok := c.m[key]
    c.mu.Unlock()           // <-- released here

    if !ok {
        v := compute()      // another goroutine can run the whole thing here
        c.mu.Lock()
        c.m[key] = v        // clobbers a concurrent insert
        c.mu.Unlock()
        return v
    }

    c.mu.Lock()
    v := c.m[key]
    c.mu.Unlock()
    return v
}
```

There is genuinely *no data race* here — every map access happens under the mutex, so every access pair is ordered by the unlock-before-lock happens-before edge. `go build -race` will run this all day and report nothing. And it is still wrong, for exactly the same reason as the Java version: the lock is released between the check and the act, so the compound action is not atomic. The fix is to hold the lock across the whole check-and-act (or, idiomatically in Go, to call `compute` outside the lock with a `singleflight` pattern, but the minimal correctness fix is a single critical section):

```go
func (c *Cache) GetOrCompute(key string, compute func() any) any {
    c.mu.Lock()
    defer c.mu.Unlock()
    if v, ok := c.m[key]; ok {
        return v
    }
    v := compute()          // note: holds the lock across compute — a real trade-off
    c.m[key] = v
    return v
}
```

(Holding the lock across `compute` serializes the expensive work, which may be the wrong trade if `compute` is slow — that is a separate engineering decision about lock granularity, covered in [readers-writer locks and lock granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity). The point here is only that the *correctness* fix is to make the compound action atomic.)

So: two languages, two idioms — a thread-safe collection in Java, a mutex-guarded plain map in Go — and the *identical* race condition, with *zero* data races in either. The thread-safety of the parts did nothing to make the whole correct. This is the cell that ruins people's day, because their entire defense was "I used the concurrent thing."

#### Worked example: a check-then-act on atomics, still racy

Even atomics do not save you if your invariant spans two of them. Suppose you maintain a bounded counter with a hard cap, using one atomic integer:

```java
import java.util.concurrent.atomic.AtomicInteger;

AtomicInteger count = new AtomicInteger(0);
final int CAP = 100;

// BUG: get-then-incrementAndGet is a check-then-act over two atomic ops.
boolean tryAdd() {
    if (count.get() < CAP) {              // atomic read
        count.incrementAndGet();          // atomic write
        return true;
    }
    return false;
}
```

`count.get()` is atomic. `count.incrementAndGet()` is atomic. There is no data race — every access to `count` is a synchronized atomic operation with happens-before edges between them. But the *two* operations are not atomic *together*: a hundred threads can all read `count == 99 < 100`, all pass the check, and all increment, leaving `count == 199`, blowing the cap by 99. The fix is a single atomic compound operation — a CAS loop that checks and increments indivisibly:

```java
boolean tryAdd() {
    int cur;
    do {
        cur = count.get();
        if (cur >= CAP) return false;
    } while (!count.compareAndSet(cur, cur + 1));  // atomic check-and-set
    return true;
}
```

The `compareAndSet` makes the check-and-act one indivisible step: it succeeds only if `count` is still `cur` at the moment of the write, and retries otherwise. No data race existed before, and none exists after — the difference is purely *atomicity granularity*. This is the heart of the race-condition family: the bug is always "I made the parts atomic but the unit of correctness was bigger than a part."

## Why undefined behavior matters: the compiler is your adversary

It is tempting to treat "undefined behavior" as standards-committee pedantry — surely a racy read just gives you a stale value, and a stale value is at worst a logic bug you can reason about. That intuition is wrong, and the gap between "stale value" and "the compiler may do anything" is where the genuinely terrifying bugs live. The figure below lays out the three concrete things UB licenses the compiler to do to a racy access that it could never do to a synchronized one.

![A before and after figure contrasting a racy plain access that may tear hoist or invent a value with a synchronized access that is reloaded and always a real value](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-5.png)

**Tearing.** A write or read of a value wider than the platform's atomic-access granularity can be split into two machine operations. On a 32-bit platform, storing a 64-bit `long` is two 32-bit stores; a concurrent reader can observe the low half of the new value and the high half of the old. Java explicitly permits non-atomic `long` and `double` reads/writes precisely for this reason — a torn `double` read on a 32-bit JVM was a real, specified hazard (the fix being `volatile long`/`volatile double`, which the JMM guarantees atomic). The reader sees a number that *was never written by anyone* — a Frankenstein of two values.

**Hoisting and reordering.** As we saw with the spin loop, the compiler may cache a racy load in a register and never reload it, or sink a store, or reorder accesses around each other, because it is allowed to assume no other thread interferes. Synchronization is precisely the signal that says "you may *not* assume that here." Remove it and the compiler optimizes as if the variable were thread-local.

**Invented values and "impossible" states.** The standard says UB imposes *no* constraints, so a sufficiently aggressive optimizer can produce code that takes a branch based on one read of a racy variable and uses a different value in the branch body, leaving the program in a state the source code says is unreachable — like our `x > 10` example returning 6, or a `switch` on a racy enum jumping to a case the value can no longer be. There are documented cases of compilers turning racy code into out-of-bounds accesses and security holes because the UB let them assume something false.

The mechanism behind all three is the same: the **as-if rule** plus the **assumption that data races do not occur**. The compiler is allowed to transform your program in any way that preserves the observable behavior *of a well-defined program*. A data race makes your program not-well-defined, so "preserve the observable behavior" has no meaning to preserve, and the transformations are unconstrained. This is why "it's just a stale read" is a dangerous way to think: you are reasoning about the *source*, but the compiler is allowed to ship something that does not correspond to your source at all.

Contrast this with a race condition that has no data race. There, every access is well-defined; the compiler preserves your operations faithfully; the only thing that varies is the *interleaving*, which is a property of the scheduler, not the optimizer. A race condition gives you a *real, explainable* (if undesired) execution — some legal order of well-defined operations. A data race gives you an execution that may not correspond to *any* legal order of your source. That is the deep reason the two are different: one is a scheduling problem, the other is a *meaning* problem.

To make the tearing hazard concrete rather than abstract, walk a 64-bit value on a 32-bit machine. A `long` is stored as two 32-bit words, the low half and the high half, written by two separate machine stores. Thread A writes `0x0000_0001_0000_0000` (high word 1, low word 0) over a previous value of `0x0000_0000_FFFF_FFFF` (high word 0, low word `0xFFFFFFFF`). The store is *not* one operation; it is "write high word = 1" and "write low word = 0," and the scheduler can interrupt between them. If thread B reads the `long` in that window, it can observe high word 1 *and* low word `0xFFFFFFFF` — the value `0x0000_0001_FFFF_FFFF`, which **neither thread ever wrote and which no sequential execution could ever produce**. That is the precise sense in which a data race "invents" a value: not metaphysically, but because a wide access decomposes into multiple machine operations and a racy reader can splice halves from different writes. Java's specification calls this out and *permits* it for non-`volatile` `long` and `double` exactly because forcing 64-bit atomicity on 32-bit hardware would cost every access; the documented fix is to mark the field `volatile`, which the JMM promises is atomic even for 64-bit types. C++ gives you `std::atomic<int64_t>`, which the compiler lowers to a locked 8-byte operation (or a lock-prefixed `cmpxchg8b` on older x86) so the read and write are indivisible. The mechanism is the same across languages: an access wider than the hardware's atomic granularity tears unless you ask for atomicity explicitly.

## Java, the careful exception: data race without UB

Java is the language that took the hardest look at this and made a deliberately different choice, and it is worth understanding because it sharpens the definitions. The Java Memory Model (JLS Chapter 17, the result of JSR-133) does *not* say a data race is undefined behavior. It instead guarantees **out-of-thin-air safety**: a racy read in Java will return *some* value that was actually written to that variable by *some* thread at *some* point (or the default zero/null) — never a torn `int`/reference, never an invented value, never a security-violating "impossible" state. The JVM is memory-safe even in the presence of data races, by design, because Java cannot afford to let a data race corrupt the heap in a way that breaks the sandbox.

But — and this is the subtlety that trips people — "no UB" does *not* mean "no bug." A Java data race still gives you **no happens-before guarantee**, so a racy read may return a *stale* value indefinitely, and you get no ordering guarantees about *other* writes that the racy thread made. The classic Java footgun is exactly our spin flag:

```java
class StopFlag {
    private boolean done = false;   // NOT volatile — data race

    void runWorker() {
        new Thread(() -> {
            while (!done) {          // may never observe the write; can spin forever
                // work
            }
        }).start();
    }

    void stop() { done = true; }     // racy write
}
```

This is a data race in the JMM (conflicting accesses, no happens-before). It is *not* UB — the JVM will not invent a value or crash. But the JIT compiler is still allowed to hoist the non-`volatile` `done` into a register, so the worker can spin forever, and there is no guarantee of *when* (or whether) the write becomes visible. The fix is one keyword — `volatile boolean done` — which makes every write happens-before every subsequent read, killing the data race and forbidding the hoist. Same bug as the C++ spin loop, same fix in spirit, but the consequence is bounded (a hang, never heap corruption) because Java spent enormous design effort to bound it. The lesson: even the language that *refused* to make data races UB still cannot make them *correct*. A data race is at minimum a missing-visibility bug everywhere.

This is also the reason Java's `double`/`long` torn-read carve-out exists at all. Java guarantees atomicity for all types *except* non-`volatile` `long` and `double`, where a 32-bit JVM may tear. So Java has UB-free data races that still produce a torn value for those two types — the one place where the "no out-of-thin-air" promise is relaxed, and a `volatile` is the documented fix.

## How Rust bans data races at compile time

Here is where the type-system story gets genuinely beautiful, and where the precise distinction pays off, because Rust can eliminate one of these two bug families *entirely* at compile time and is *structurally incapable* of eliminating the other.

Rust's claim is "fearless concurrency," and the specific guarantee is: **safe Rust code cannot have a data race.** Not "is unlikely to," not "warns about" — *cannot*, as a compile-time property, enforced by the borrow checker and two marker traits, `Send` and `Sync`. The mechanism is the same ownership-and-borrowing system Rust uses for memory safety, turned on aliasing across threads.

The core rule of the borrow checker is "aliasing XOR mutation": at any time you may have *either* any number of shared `&T` references *or* exactly one exclusive `&mut T` reference to a value, never both. A data race requires two threads accessing the same location with at least one writing — that is precisely *aliased mutation across threads*, which the aliasing-XOR-mutation rule forbids. So the very thing that prevents use-after-free in single-threaded Rust prevents data races in multi-threaded Rust, for free, by construction.

Two marker traits extend this across thread boundaries. **`Send`** means a value can be *moved* to another thread (transferring ownership; safe because ownership is exclusive). **`Sync`** means a value can be *shared* (`&T` accessed) from multiple threads (safe because `&T` is read-only, or because the type does its own internal synchronization). The compiler *automatically* derives these traits structurally — a struct is `Send` if all its fields are `Send` — and the thread-spawning APIs *require* them. `thread::spawn` demands its closure be `Send`; sharing across threads demands `Sync`. A type that is not safe to send or share (like `Rc<T>`, the non-atomic reference count, or a raw `Cell<T>`) simply does *not* implement the trait, and the program *does not compile*. Here is the data race the compiler refuses:

```rust
use std::thread;

fn main() {
    let mut count = 0;
    let mut handles = vec![];

    for _ in 0..10 {
        // ERROR: closure may outlive `count`, and you'd be aliasing a mutable value
        // across threads. The borrow checker rejects this; there is no data race
        // because there is no compiled program.
        let h = thread::spawn(|| {
            count += 1;          // E0373 / E0499: cannot borrow `count` mutably here
        });
        handles.push(h);
    }
    for h in handles { h.join().unwrap(); }
    println!("{}", count);
}
```

The compiler does not let you write the `count++` data race. The fix — the one Rust *forces* you toward — is shared ownership (`Arc`, the atomic reference count, which *is* `Send + Sync`) plus a `Mutex` for the exclusive access:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let count = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let count = Arc::clone(&count);          // shared ownership, refcount is atomic
        let h = thread::spawn(move || {
            let mut guard = count.lock().unwrap(); // exclusive access via the lock
            *guard += 1;                           // no data race: lock gives happens-before
        });
        handles.push(h);
    }
    for h in handles { h.join().unwrap(); }
    println!("{}", *count.lock().unwrap());        // always 10
}
```

This compiles, and it is *guaranteed* free of data races, because `Arc<Mutex<i32>>` is `Send + Sync` and the `Mutex` establishes happens-before on every access. The before-and-after — the rejected plain counter versus the accepted `Arc<Mutex>` — is the whole pitch of fearless concurrency in one frame:

![A before and after figure showing a plain shared counter that the Rust borrow checker rejects at compile time versus an Arc Mutex version that compiles with no data race](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-8.png)

Now the punchline that the whole post has been building toward. **Rust cannot ban race conditions.** It bans data races — the memory-model layer — because aliased mutation is visible to the type system. But a race condition lives at the *logic* layer, in the *sequence* of well-typed, well-synchronized operations, and the type system has no idea what your *invariant* is. The `Arc<Mutex>` version above is data-race-free, but if you wrote the check-then-act over it — lock, read, unlock, decide, lock, write, unlock — Rust would compile it happily and you would have the exact same race condition as the Go example. The Rust documentation says this explicitly: Rust's guarantees prevent data races but "do not prevent all race conditions in general." A deadlock is also a perfectly compilable Rust program. The type system can prove "no two threads touch this memory unsynchronized." It cannot prove "your sequence of synchronized operations preserves your invariant," because it does not know your invariant. That is the structural reason the two bug families are different: one is a property of *memory access* (mechanizable in a type system) and the other is a property of *application logic* (not).

## A taxonomy to keep the families straight

Putting it together, the bugs split into two families with distinct sub-shapes, and keeping the tree in your head is what lets you reach for the right tool. The data-race family is memory-model UB (or, in Java, missing-visibility); the race-condition family is logic-ordering. The figure makes the split and the sub-shapes explicit.

![A taxonomy tree splitting race bugs into a data race family with torn read and stale read and a race condition family with check then act and lost update and ordering](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-6.png)

Under **data race (UB / missing-visibility)**: a **torn read/write** (a wide value split across machine operations, observed half-and-half) and a **stale read** (a load hoisted or never refreshed, observing an old value forever, or a write never made visible). These are *physical* symptoms — they are about how memory and the compiler actually behave when you skip synchronization.

Under **race condition (logic)**: **check-then-act** (the TOCTOU gap — time-of-check to time-of-use), **lost update** (read-modify-write that drops a concurrent modification because the compound op is non-atomic), and **ordering** (the program is correct only under a particular interleaving that nothing enforces). These are *logical* symptoms — they are about the granularity of atomicity relative to your invariant.

The reason the taxonomy is worth memorizing: the *family* tells you the *fix*. A data-race symptom is fixed by establishing happens-before on the access — make it atomic, put it under a lock, mark it `volatile`/`Sync`. A race-condition symptom is fixed by widening the atomic unit — make the *compound* action atomic with a single lock held across it, a CAS loop, a `computeIfAbsent`, or a transaction. Diagnose into the wrong family and you will, for instance, sprinkle `volatile` on a check-then-act (which fixes nothing, because there was no data race) or add a heavyweight lock to a benign double-read (which fixes the symptom but for the wrong reason and at the wrong cost).

## Detecting each kind: different tools entirely

This is the most practical payoff of the distinction: **the two families require different detection strategies, because a data-race detector finds data races and is structurally blind to race conditions.**

**Detecting data races.** This is the *more* tractable problem, surprisingly, because a data race has a precise definition — conflicting accesses unordered by happens-before — that a tool can check mechanically by tracking the happens-before relation at runtime. The premier tools:

- **Go's `-race`** (`go build -race`, `go test -race`): a ThreadSanitizer-based dynamic detector. It instruments every memory access and synchronization operation, builds the happens-before graph, and reports any pair of conflicting accesses with no edge between them. It found real data races in the Go standard library itself.
- **ThreadSanitizer (TSan)** for C/C++/Rust (`-fsanitize=thread`): the same happens-before machinery. Catches the `count++` race, the spin-flag race, the torn access — anything that is a true data race in an *executed* interleaving.
- **Helgrind / DRD** (Valgrind tools) for C/C++: older, slower, but happens-before-based detectors.
- The key honest caveat: these are **dynamic** detectors. They only flag races on code paths and interleavings that *actually execute during the run*. A race on a rare path is missed unless you exercise that path. They have near-zero false positives (a flagged race is real) but real false negatives (an unflagged run does not prove race-freedom). This is why you pair them with stress testing — covered in [finding concurrency bugs with race detectors and stress testing](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing).

**Detecting race conditions.** This is *harder*, because a race condition has no syntactic signature — it is a violation of *your* invariant, which the tool does not know. A data-race detector will run clean over the `ConcurrentHashMap` check-then-act forever, because there is no data race to find. Your tools here are:

- **Invariant checks and assertions** that encode your correctness condition (`assert count <= CAP`), run under stress.
- **Stress testing** with many threads, many iterations, randomized timing, and a *checker* that validates the invariant after — this is what surfaces the rare interleaving.
- **Deterministic schedulers / model checkers** that systematically explore interleavings (Go's `GOMAXPROCS` tweaking, `rr` for record-replay, Loom for Rust which exhaustively explores the possible orderings of atomic operations, the Java `jcstress` harness which runs adversarial interleavings and tallies the observed outcomes against the allowed ones).
- **Code review with the granularity question front of mind**: "is the *unit of correctness* atomic, or just the parts?" Every check-then-act is a candidate.

The table below is the one to keep:

| Aspect | Data race | Race condition |
| --- | --- | --- |
| What it is | Conflicting memory accesses unordered by happens-before | Correctness depends on interleaving of operations |
| Layer | Memory model | Application logic |
| Consequence | Undefined behavior (Java: missing visibility, no UB) | A real but invalid execution |
| Needs a shared write | Yes, at least one access must be a write | Not necessarily a raw write — can be over atomics/locks |
| Has a syntactic signature | Yes — a checkable property of accesses | No — it is a violation of your invariant |
| Found by | `-race`, TSan, Helgrind (happens-before detectors) | Stress tests, model checkers, invariant assertions, review |
| False positives | Near zero (a flag is a real race) | The tool has no notion of "the bug" without your invariant |
| Fixed by | Establish happens-before: atomic, lock, volatile, Send/Sync | Widen the atomic unit: lock across the compound action, CAS, transaction |
| Bannable by a type system | Yes (Rust Send/Sync) | No — logic is opaque to the type system |

#### Worked example: measuring the failure rate honestly

A race's *failure rate* is itself a measurement, and measuring it teaches the lesson better than any argument. Take the unsynchronized `count++` from two threads, each incrementing a shared counter one million times, then compare the final value against the expected two million. The shortfall is the number of lost updates — every collision where two threads read the same value and one increment vanished. Run it many times and you observe something instructive: the *magnitude* of the loss is wildly nondeterministic, swinging from a handful of lost updates to hundreds of thousands depending on how the scheduler interleaved the two threads and how much the cores actually contended. The honest way to report this is not "it loses about X updates" — it does not, the number is not stable — but "across 100 runs the final value was wrong every time, with the loss ranging over several orders of magnitude." The qualitative result (*always* wrong) is robust; any precise loss figure is not, and quoting one would be fabricating stability that does not exist.

The same experiment exposes the platform dependence that makes "works on my machine" so treacherous. On a single-core machine, or with both threads pinned to one core, the loss can drop to *zero* on many runs, because the scheduler rarely interleaves the load-modify-store across a context switch and the cache never bounces — so the racy program *passes its test* and looks correct. Move it to a multi-core box where the two threads run truly in parallel and contend for the cache line, and the loss appears immediately and reliably. Same code, same source, opposite test outcome, governed entirely by the hardware and the scheduler. The failure-mode table below summarizes how each bug family behaves under measurement — and why a single run is the weakest evidence you can collect:

| Property under measurement | Data race | Race condition |
| --- | --- | --- |
| Reproducibility | Often rare; sensitive to `-O` level and CPU | Rare; sensitive to thread count and load |
| Single-core behavior | May vanish (no true parallelism) | May vanish (benign interleaving dominates) |
| Effect of more cores / load | More likely to manifest | More likely to manifest |
| Effect of compiler optimization | Can appear or worsen at `-O2` | Unaffected — it is a logic bug |
| What a clean run proves | Nothing — UB can look correct | Nothing — the malign interleaving may not have fired |
| Best detection | Happens-before detector in CI on multi-core | Stress test plus invariant check, many iterations |

#### Worked example: the detector that finds one bug and not the other

Take a single program with *both* bugs: a `count++` data race (no lock on the increment) *and* a check-then-act race condition (a non-atomic get-then-set on a separate, properly-locked structure). Run `-race` / TSan. It flags the `count++` immediately — a precise file-and-line report of two unordered conflicting accesses. It says *nothing* about the check-then-act, because every access there is synchronized; there is no data race for it to see. You fix the `count++`, the detector goes green, you ship — and the check-then-act bug is still there, corrupting state in production, because the *clean detector report was never evidence of correctness*. It was evidence of *data-race-freedom*, which is a strictly weaker property. This is the single most important operational consequence of the distinction: **a green race detector does not mean your program is correct.** It means it has no data races. The race conditions are still your job.

## Case studies / real-world

**Dirty COW (CVE-2016-5195) — a race condition with catastrophic reach.** The Linux kernel's copy-on-write handling for private memory mappings had a time-of-check-to-time-of-use race condition. An attacker could race the `madvise(MADV_DONTNEED)` path against a write to a copy-on-write page so that the write landed on the *original*, read-only mapping (e.g. a `setuid` binary or `/etc/passwd`) instead of the private copy. The individual operations were each correctly synchronized kernel primitives; the bug was the *interleaving* — a classic check-then-act where the state checked (the page was a private copy) was no longer true at the moment of use. It was a race *condition*, not a data race in the memory-model sense: no torn read, no UB, just an interleaving the code did not account for, present in the kernel for roughly a decade and exploited in the wild to gain root. The fix changed the logic so the COW decision and the write could not be split. This is the race-condition family at its most consequential: a logic-ordering bug that a data-race detector would never have flagged.

**The Java Memory Model rewrite (JSR-133) — why the distinction needed a formal model.** Before Java 5, the original JMM was subtly broken: it permitted compilers and processors to reorder writes in ways that broke `final` fields and double-checked locking, and the semantics of a data race were so loose that "safe" idioms were silently wrong. JSR-133 (2004) rebuilt the model around happens-before, specified out-of-thin-air safety, and fixed `volatile` and `final` semantics — precisely so that engineers could reason about the boundary between "this access is ordered" and "this is a data race." That a major language needed a multi-year formal effort to nail down what a data race *means*, and to bound its consequences short of arbitrary UB, is the strongest evidence that this is a real, hard distinction and not pedantry. Goetz's *Java Concurrency in Practice* and the JLS Chapter 17 are the durable references.

**Go's `-race` finding real stdlib data races.** When Go shipped its ThreadSanitizer-based race detector, running it across the standard library and major codebases surfaced genuine data races that had survived code review and testing — unsynchronized map writes, racy lazy initialization, shared-slice mutation. These were *data races* (the detector's whole job), and many had never manifested as a visible failure because the racy interleaving was rare and the compiler had not yet chosen to exploit the UB. They are the textbook case for "works on my machine": code that was *always* undefined behavior, ran correctly for years by luck, and was provably racy the moment a happens-before-aware tool looked at it. The lesson the Go team drew — run `-race` in CI, not just locally — is the right operational response to a dynamic detector's false negatives.

The thread through all three: the catastrophic, decade-living, exploited-in-the-wild bug (Dirty COW) was a *race condition* that no data-race tool would catch; the bugs the *tools* catch are *data races* that often never visibly fail until the compiler or scheduler happens to expose them. Two different failure profiles, two different defenses.

## When to reach for this distinction (and when not)

The point of a precise vocabulary is to make better decisions faster. Here is when the data-race-vs-race-condition distinction actively changes what you do, and when it does not.

**Reach for "is this a data race?" first when:** you have UB symptoms — a value that was never written, a loop that spins forever despite a flag being set, behavior that changes between `-O0` and `-O2` or between x86 and ARM, a crash inside a "thread-safe" structure. These scream memory model. Run the race detector (`-race`, TSan). If it flags something, you have a data race; fix it by establishing happens-before (atomic, lock, `volatile`, the right `Send`/`Sync`). A clean detector run *rules this family in or out cheaply* — that is its highest value.

**Reach for "is this a race condition?" when:** the detector is green but the program is still intermittently wrong, especially when you are composing thread-safe building blocks (a concurrent collection, atomics, locked structures) and the symptom is a *logical* corruption — a duplicate, a lost update, a violated cap, an invariant that spans two updates. The tell is "every individual operation is synchronized." When you hear yourself say "but the map is thread-safe," you are almost certainly in the race-condition cell. The fix is to widen the atomic unit.

**Do not reach for `volatile`/atomics to fix a race condition.** This is the most common misapplication of a half-understood distinction: a check-then-act bug gets "fixed" by making the checked variable `volatile`, which closes a data race that was never there and leaves the ordering bug fully intact. If the operations were already synchronized, more synchronization on the *same granularity* does nothing — you need a *wider* atomic action.

**Do not reach for a heavyweight lock to fix a benign data race when an atomic is the right tool.** A single shared flag does not need a mutex; an `atomic<bool>` / `volatile boolean` / `AtomicBoolean` is cheaper and exactly sufficient. The distinction tells you the *minimum* mechanism: a data race on a single word wants an atomic; a race condition over a compound action wants the action made atomic, which usually means a lock or a CAS loop.

**Do not treat a green race detector as a correctness proof.** Worth repeating because it is the single most expensive mistake: `-race`/TSan finding nothing means "no data races on the executed interleavings," which is necessary but nowhere near sufficient. The race conditions, the rare-path data races, and the logic bugs are all still yours. Pair the detector with stress tests and invariant checks.

**When the distinction does not matter:** in the overlap cell — an unsynchronized `count++` — it is both, and any correct fix (lock or atomic) closes both, so you need not agonize over the taxonomy. The distinction earns its keep specifically off the diagonal, where the wrong diagnosis leads to the wrong fix. The four-cell stance is summarized per language below.

![A matrix of four languages C and C plus plus Java Go and Rust showing how each treats a data race and what tooling detects it](/imgs/blogs/data-races-vs-race-conditions-a-precise-distinction-7.png)

The per-language stance is itself a useful decision aid: in C/C++ and Go a data race is UB and your defense is the dynamic detector plus discipline; in Java it is bounded (no UB) but still a visibility bug, defended with `volatile`/`j.u.c` and `jcstress`; in Rust safe code *cannot* have one, so your remaining concurrency budget goes entirely to race conditions and deadlocks, which the compiler does not catch.

## Why "works on my machine" is a lie

We can now state the title's claim precisely, because both bug families make it a lie for different reasons.

A **data race** is undefined behavior. "Works on my machine" means "on my machine, with my compiler, my optimization level, my CPU's memory model, and the particular interleaving the scheduler happened to pick, the UB happened to do what I wanted." Change the compiler version (it adds an optimization that hoists your racy load), change `-O0` to `-O2` (it starts exploiting the UB), move from x86's strong TSO memory ordering to ARM's weak ordering (reorderings your x86 never exhibited become visible), change the thread count or the machine's load (a different interleaving), and the program does something different. The code was *never* correct; it was *undefined*, and undefined behavior is permitted to look correct on Tuesday and corrupt your heap on Wednesday. "Works on my machine" is not evidence of correctness for a racy program; it is the *symptom* of UB — code whose meaning depends on facts outside the language.

A **race condition** makes it a lie for a subtler reason: it is correct on most interleavings and wrong on a rare one. "Works on my machine" means "the rare interleaving that violates the invariant did not happen during my testing." It is governed by the scheduler, which on your laptop under light load almost always serializes things in the benign order, and in production under contention on many cores eventually hits the malign order — the "one-in-a-million" race that fires a thousand times a day at scale. Here the code is well-defined on every interleaving; it is just that *some* legal interleaving is wrong, and your machine did not happen to produce it.

Both reduce to the same engineering truth: concurrency correctness is a property of *all* allowed executions, and "it worked when I ran it" tests *one* execution out of an astronomical space — one compiler decision, one memory-ordering outcome, one scheduler interleaving. A single passing run is the weakest possible evidence. The strong evidence is a *proof* of happens-before coverage (no data races, by construction or by detector), plus a *systematic* exploration of interleavings (stress tests, model checkers) against an *explicit invariant*. That is the discipline the whole series is about, and it begins with knowing which of these two things you are even looking at.

## Key takeaways

- A **data race** is a memory-model violation: two accesses to the same location, concurrent, at least one a write, with **no happens-before edge** between them. It is **undefined behavior** in C, C++, Go, and Rust — not "a bug," but UB the compiler may exploit.
- A **race condition** is a logic violation: correctness depends on the **interleaving** of operations. It can exist with **zero data races** — every individual operation perfectly synchronized, the *compound* action non-atomic.
- The two are **independent axes**. A bug can be a data race only (a racy flag the compiler hoists into an infinite spin), a race condition only (check-then-act on a thread-safe map), both (`count++`), or neither.
- **Undefined behavior is not "a stale value."** It licenses the compiler to tear, hoist, reorder, and invent values, producing executions that correspond to no legal ordering of your source. Java is the exception that bounds this to missing-visibility with no UB — but a Java data race is still always at least a visibility bug.
- The **family dictates the fix.** Data race → establish happens-before (atomic, lock, `volatile`, `Send`/`Sync`). Race condition → widen the atomic unit (lock across the compound action, a CAS loop, `computeIfAbsent`, a transaction). Putting `volatile` on a check-then-act fixes nothing.
- **Different tools find each.** Happens-before detectors (`-race`, TSan, Helgrind) find data races precisely and are structurally blind to race conditions. Race conditions need stress tests, model checkers, and invariant assertions. **A green race detector is not a correctness proof.**
- **Rust bans data races at compile time** via the borrow checker plus `Send`/`Sync` — aliased mutation across threads is a type error. It **cannot** ban race conditions, because logic invariants are opaque to the type system. Deadlocks and check-then-act bugs compile fine.
- **"Works on my machine" is a lie** for both: a data race is UB that happened to look right under one compiler/CPU/interleaving; a race condition is correct on the common interleaving and wrong on the rare one your test never hit.

## Further reading

- **Within this series:** [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) for the foundational frame; [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) for the lost-update mechanics; [memory models, sequential consistency, and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) for the formal happens-before relation this post leans on; [finding concurrency bugs with race detectors and stress testing](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing) for the detection toolchain; and the capstone, [the concurrency playbook for choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- **The Python angle:** for how the GIL changes (and does not eliminate) this picture in Python, see [the GIL explained: what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) — the GIL prevents many data races on Python objects but does *not* prevent race conditions, which is the same distinction in a different runtime.
- **Brian Goetz et al., *Java Concurrency in Practice*** (2006) — Chapter 2 and 3 are the clearest treatment of compound actions, check-then-act, and the JMM happens-before rules anywhere.
- **The Java Language Specification, Chapter 17 (Threads and Locks)** and the JSR-133 cookbook — the formal happens-before model and the out-of-thin-air guarantee.
- **The C++ standard, `[intro.races]`**, and Hans Boehm's papers (notably "Threads Cannot Be Implemented as a Library") — why a data race is UB and why the memory model belongs in the language, not a library.
- **The Rust Book, "Fearless Concurrency" chapter**, and the Nomicon's treatment of `Send`/`Sync` — how the type system mechanizes data-race freedom and why it stops at race conditions.
- **The Dirty COW disclosure (CVE-2016-5195)** and the Linux commit that fixed it — a canonical, consequential race condition (TOCTOU) to study end to end.
- **Jeff Preshing's blog** (preshing.com), especially the posts on acquire/release semantics and the difference between memory ordering and atomicity — the best free intuition-building resource for the memory-model layer.

---
title: "Race Conditions: The Hardest Bugs to Catch"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to define, detect, reproduce, and fix the bugs that pass 999 times and fail on the run that matters, using ThreadSanitizer, the Go race flag, stress, and atomics."
tags:
  [
    "debugging",
    "software-engineering",
    "race-conditions",
    "concurrency",
    "thread-safety",
    "threadsanitizer",
    "memory-model",
    "data-race",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-1.png"
---

The ticket said "counter is occasionally low." Not crashes. Not errors. Just: the daily total of processed jobs, which should equal the number of jobs we actually processed, was sometimes off by a handful. Forty-one thousand jobs ran; the counter read 40,996. The next day it read 41,000 exactly. The day after, 40,988. There was no error in the logs, no exception, no stack trace, nothing to grep for. The code was four lines long and looked obviously correct. I ran it locally ten thousand times in a loop and it was right every single time. I shipped a "fix" that did nothing, because there was nothing to fix in the code-as-written. The bug was not in the code. The bug was in the *timing*.

This is the defining property of a race condition, and it is what makes it the hardest class of bug to catch: the program is correct for almost every interleaving of its threads, and wrong for a few. The wrong ones are rare, schedule-dependent, and — this is the cruel part — they get rarer the harder you look, because attaching a debugger, adding a log line, or running on your quiet laptop all change the timing in a way that hides the bug. A test that fails 6 times in 2,000 runs on a loaded CI box will fail 0 times in 2,000 runs on your machine. You can stare at the source for an hour and see nothing, because the source *is* correct — for the schedule you imagine. The schedule the machine actually picked is the one you never thought about. The figure below is the whole problem in one picture: the exact same four-line source, run twice, producing 2 in the lucky case and 1 in the unlucky one.

![Two columns showing the same increment code producing a count of two under one thread schedule and a count of one under another schedule that loses an update](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-1.png)

By the end of this post you will be able to do five concrete things. First, *name* the bug precisely — distinguish a **data race** (two threads touch the same memory, at least one writes, with no synchronization between them, which is undefined behavior in C, C++, Java, and Go) from a **race condition** (correctness depends on the interleaving even when every access is properly locked). Second, explain the *mechanism* — why the compiler and the CPU make this worse through reordering, store buffers, and cache coherence, and why "no happens-before edge" means another thread may *never* see your write. Third, *detect* races on purpose with ThreadSanitizer and the Go race detector, reading their reports as the two conflicting stacks they are. Fourth, *reproduce* a race on demand — turn a 0.3% flake into a 90% failure with scheduling pressure, stress, and thousands of runs, so you actually have a bug to fix. Fifth, *fix* it correctly: a mutex around the whole invariant, an atomic with the right memory order, immutability, message passing, or making check-and-act atomic. This is the same loop the rest of this series runs on — observe, reproduce, hypothesize, bisect, fix, prevent — applied to the one bug class that fights you at every step. If you have not read the series intro, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) sets up the loop; this post is what happens when the symptom refuses to reproduce.

## 1. Two bugs wearing one name

The word "race condition" gets used for two genuinely different bugs, and conflating them is the first reason people fix the wrong thing. Let me define both precisely, because the fix you need depends entirely on which one you have.

A **data race** is a mechanical, low-level property of your program: two threads access the same memory location, at least one of those accesses is a write, and there is no *happens-before* relationship between them — no lock, no atomic, no thread join, nothing that orders one access before the other. That is the exact definition the language memory models use, and when it holds, the C, C++, Java, and Go memory models all say the same thing: the behavior is **undefined**. Not "you get one of the two values." Undefined. The compiler is allowed to assume data races do not happen and optimize accordingly, which is why a data race can produce results that no interleaving of the source code could ever produce — a torn read that is half of one value and half of another, a loop that never terminates because the compiler hoisted a read out of it, a `1` where only `0` and `2` were ever stored.

A **race condition** is a higher-level logic bug: the correctness of your program depends on the *order* in which threads execute, and some valid orders give wrong answers — *even if every individual memory access is perfectly synchronized*. The canonical example is **check-then-act**: you check whether a file exists, and if it doesn't, you create it. Each step can be individually thread-safe. The bug is that another thread can create the file in the gap *between* your check and your act, so your "it doesn't exist" conclusion is stale by the time you act on it. There is no data race here — every access might go through a lock — and yet the program is wrong.

The reason this distinction matters so much in practice: **a data race is usually fixable by adding synchronization, but a race condition often is not.** Wrapping the check and the act in a mutex fixes nothing if they are in two separate critical sections; you have to make the *whole* check-and-act a single atomic operation. The figure below lays out the taxonomy with the concrete sub-cases of each that you will actually meet.

![A tree dividing a concurrency bug into a data race branch with torn read and unsafe publication leaves and a race condition branch with check-then-act and order-dependent leaves](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-2.png)

Here is the litmus test I use when triaging. Ask: *"If every shared access were already perfectly synchronized, would the program still be wrong?"* If yes, it's a race condition (a logic bug about ordering) and a detector like ThreadSanitizer will probably *not* find it, because there's no unsynchronized access to flag. If the answer is "no, it would be correct," then it's a data race, and a detector will find it for you in seconds. Most real concurrency bugs are one of these two; some are both stacked on top of each other, which is why your first fix sometimes only moves the failure rate from 0.3% to 0.1% instead of to zero.

One more piece of vocabulary, because the whole post leans on it. **Happens-before** is the partial order the memory model gives you. If action A happens-before action B, then B is guaranteed to see the effects of A. Within a single thread, program order gives you happens-before for free. *Across* threads, you only get a happens-before edge through a synchronization action: unlocking a mutex happens-before the next lock of that same mutex; a write to an atomic with release semantics happens-before a read of that atomic with acquire semantics; the end of a thread happens-before a successful join of it. If there is no chain of happens-before edges connecting your write to another thread's read, the memory model gives you *no guarantee whatsoever* that the read sees the write. Not "it'll probably see it eventually." No guarantee. That is the single most important sentence in this entire post.

## 2. The mechanism: why your write may never arrive

Newcomers to concurrency carry an intuition that goes like this: threads run on real hardware, memory is real, so when thread 1 writes `x = 1`, the value `1` is now in memory, and thread 2 will read it. Maybe thread 2 reads it a few nanoseconds before the write lands and gets the old value — a timing thing — but eventually the write is visible. This intuition is wrong in a way that matters, and understanding *why* it's wrong is what separates engineers who fix races from engineers who sprinkle `volatile` until the flake rate drops and then move on.

There is no single "memory" that all cores agree on instantly. Between your `x = 1` statement and another thread's `read x`, there are at least four layers, and every one of them is allowed to reorder, delay, or hide your write. The figure below stacks them.

![A vertical stack from source order down through compiler reordering, the store buffer, cache coherence, and finally the other thread which never sees the write](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-3.png)

**Layer one: the compiler reorders.** The compiler's job is to make single-threaded code fast, and it assumes no other thread is watching. If you write `x = 1; done = true;` and the compiler can prove that, *for this thread*, the order doesn't matter, it may emit them in the other order, or keep `x` in a register and never write it to memory at all inside a loop. The classic bug is a spin-wait: `while (!done) {}` compiled to read `done` once into a register and loop forever on the register, because nothing in *this* thread changes `done`. The compiler is correct under the single-threaded "as-if" rule. Your code is wrong for assuming the compiler knew about the other thread.

**Layer two: the store buffer.** Modern CPUs don't write to cache synchronously. A store goes into a per-core *store buffer* first and drains to cache later. Your core sees its own stores immediately (it snoops its own buffer), but other cores don't see them until the buffer drains. So thread 1 can write `x = 1`, then write `done = true`, and another core can observe `done == true` while still reading `x == 0`, because the two stores drained in a different order or one hasn't drained yet. This is real, it is observable on x86 (store-load reordering) and far more aggressively on ARM and POWER, and it is exactly why double-checked locking written naively is broken.

**Layer three: cache coherence — but lazily.** Caches *are* kept coherent by protocols like MESI, which is why people think writes propagate. But coherence guarantees that there's a single global order of writes *to one location*; it does not guarantee *when* a core sees another core's write, nor does it order writes to *different* locations as seen by a third core. Coherence is necessary but nowhere near sufficient for your multi-variable invariants.

**Layer four: no happens-before edge means no promise at all.** Stack the three above and the conclusion is stark — without a synchronization action establishing happens-before, the language standard makes *no promise* that thread 2 ever observes thread 1's write. In practice it usually does, eventually, which is exactly what makes the bug so insidious: it works 99.9% of the time, so you ship it.

Here is a torn-read mechanism made concrete. On a 32-bit platform, a 64-bit `long` is written as two 32-bit stores. If thread 1 is writing `0x00000001_FFFFFFFF` (low half then high half) and thread 2 reads in between, thread 2 can read `0x00000000_FFFFFFFF` — a value that was never stored by anyone. That's a **torn read**: a value that is half of the old and half of the new. No interleaving of the *source* produces it; it only exists because the store and load aren't atomic at the hardware level. The same thing happens to misaligned values, to wide vector writes, and — the version you'll actually hit — to any object reference published without synchronization, where another thread can see the reference before it sees the writes to the object's fields.

This is why the fix for a data race is never "add a sleep" or "hope it's fine." The fix is to **establish a happens-before edge** between the write and the read — a lock, an atomic with the right ordering, a channel send/receive — so the memory model is forced to make the write visible. Everything else is luck. For a deeper, distributed-systems treatment of the same ordering ideas at a larger scale, see the [consistency models guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) — happens-before is the same concept whether it's two cores or two data centers.

Let me make the spin-wait failure utterly concrete, because it's the single best demonstration that "the value is in memory, so the other thread will read it" is false. Consider this innocent-looking handoff:

```c
int done = 0;
int data = 0;

// Producer thread:
data = 42;
done = 1;          // signal "data is ready"

// Consumer thread:
while (!done) { }  // spin until ready
printf("%d\n", data);
```

You expect this to print `42`. It may print `0`, or it may loop forever. Two independent failures hide here. First, the consumer's compiler sees that nothing *in the consumer thread* changes `done`, so it is free to load `done` into a register once and spin on the register — an infinite loop even after the producer sets `done = 1` in memory. Second, even if the consumer does re-read `done` from memory, the producer's two writes (`data = 42` then `done = 1`) can become visible to the other core *in the opposite order* because of store-buffer and compiler reordering — so the consumer can see `done == 1` while still reading `data == 0`. Both failures are real; neither is a bug in the source as you read it; both vanish the instant you make `done` an atomic (or `volatile` in Java / `std::atomic` in C++) with acquire-release semantics, because that forces both the re-read and the ordering. This one tiny example contains the entire mechanism: compiler reordering, store-buffer delay, and the absence of a happens-before edge.

There's a subtler memory-model trap worth naming because it produces *performance* bugs that look like races: **false sharing.** Two threads update two *different* variables that happen to live on the same 64-byte cache line. There's no data race — the variables are distinct and each is properly synchronized — but every write by one thread invalidates the cache line in the other thread's core, so the line ping-pongs between cores and throughput collapses. The symptom is "I added a per-thread counter and the program got *slower* with more threads." The fix is to pad each thread's data to its own cache line (`alignas(64)` in C++, padding fields in Go structs). False sharing isn't a correctness race, but it lives in the same memory-coherence machinery, and I mention it so you recognize it when a "race fix" mysteriously tanks your performance — you may have moved two hot variables onto one line.

## 3. The classic patterns you will actually meet

Races are not infinitely varied. In fifteen years I have met essentially the same five patterns over and over, and once you can name them on sight, you find them faster. Here they are, each with the smallest reproducer I can write.

**Lost update (read-modify-write without a lock).** This is the counter from the intro. `count++` is not one operation; it is *read* `count`, *add one*, *write* `count`. Two threads can both read the same value, both add one, and both write the same value — two increments collapse into one. The lost-update pattern is the single most common data race in the wild, and it hides anywhere a shared accumulator lives: a metrics counter, a reference count, a "jobs remaining" tally, a cache hit/miss ratio.

```go
// LOST UPDATE: count++ is read-modify-write, not atomic.
var count int
var wg sync.WaitGroup
for g := 0; g < 8; g++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 1_000_000; i++ {
            count++ // <- two goroutines can read the same value
        }
    }()
}
wg.Wait()
fmt.Println(count) // want 8_000_000; got 7_312_004 on one run
```

**Check-then-act / TOCTOU.** TOCTOU stands for "time of check to time of use." You test a condition, then act on the assumption that the condition still holds — but another thread can invalidate it in the gap. The "create file if it doesn't exist" pattern is the textbook case; so is "if user not in DB, insert user," "if balance >= amount, withdraw," and "if the connection pool has a free slot, take it." Each is a window where stale truth leads to a wrong act. We'll dissect this one in depth in §6.

**Double-checked locking done wrong.** The idiom: check a flag without a lock (fast path), and only if it's unset, take a lock and check again before initializing. The bug is that the first, unlocked read can see a *non-null* reference to a *partially constructed* object — because the publication of the reference raced ahead of the writes to the object's fields (the store-buffer reordering from §2). This bug ate the Java community alive in the late 1990s and is the reason `volatile` got fixed semantics in the Java 5 memory model. Done right, the field must be `volatile` (Java) / `atomic` with acquire-release (C++) so the reference publication carries a happens-before edge. Here is the broken version and the fix side by side, because the difference is one keyword and it's the most consequential keyword in the file:

```java
// BROKEN double-checked locking: a non-volatile field can be
// published before the constructor's writes are visible.
class Holder {
    private static Config instance;          // <- NOT volatile: the bug

    static Config get() {
        if (instance == null) {              // 1st check, no lock
            synchronized (Holder.class) {
                if (instance == null) {      // 2nd check, locked
                    instance = new Config(); // <- reference may publish
                }                            //    BEFORE the fields are written
            }
        }
        return instance; // another thread can see a half-built Config
    }
}
```

```java
// FIXED: volatile makes the publication carry a happens-before edge,
// so a thread that sees a non-null reference also sees all the fields.
class Holder {
    private static volatile Config instance; // <- one word fixes it

    static Config get() {
        Config local = instance;             // read volatile once
        if (local == null) {
            synchronized (Holder.class) {
                local = instance;
                if (local == null) {
                    local = new Config();
                    instance = local;        // volatile store: full publication
                }
            }
        }
        return local;
    }
}
```

The reason `volatile` fixes it: a volatile *write* in Java has release semantics and a volatile *read* has acquire semantics, which means everything the writing thread did *before* the volatile store is guaranteed visible to any thread that *reads* a non-null value from the volatile field. That's the happens-before edge that the non-volatile version was missing. You verify this kind of publication guarantee not by staring at it — staring is what failed the whole community for years — but with **jcstress**, the JVM concurrency stress harness, which runs the experiment across thread configurations millions of times and reports whether the "half-built object" result was ever actually observed. We'll return to jcstress when we compare detectors in §4.

**Publication via a non-volatile field.** The general form of the double-checked-locking bug: you build an object fully, then store a reference to it in a plain field, expecting other threads to see a fully-built object. They might see the reference but not the fields. The fix is a *safe publication* mechanism: a `volatile`/atomic field, a `final` field set in the constructor, storing it inside a synchronized block that readers also synchronize on, or handing it through a concurrent collection or channel that provides the edge for you.

**Order-dependent logic even under locks.** The pure race condition: every access is locked, but the *outcome* depends on which thread won. "Transfer \$100 from A to B" and "transfer \$50 from B to A" each lock both accounts correctly, but if they grab locks in opposite orders you get a deadlock (a sibling bug — see [deadlocks, livelocks, and starvation](/blog/software-development/debugging/deadlocks-livelocks-and-starvation), which covers the lock-ordering discipline that prevents it). Even without deadlock, "send the welcome email after the account is committed" can fire the email first if two callbacks race. No detector flags this; only reasoning does.

These five share a single root: **a sequence of operations that must be atomic with respect to other threads is not.** The cure is always to make the *invariant-preserving unit* atomic — never just the individual reads and writes inside it.

## 4. The method: detect it with ThreadSanitizer

You cannot reliably find a data race by reading code, and you cannot find it by running the program once. You find it by instrumenting *every* memory access and *every* synchronization action and letting the tool build the happens-before graph for you. That is exactly what **ThreadSanitizer** (TSan) does. In Go it's the `-race` flag; in C, C++, and Rust it's `-fsanitize=thread`. It is the single highest-leverage tool in this entire post, and if you take one practical thing away, make it "run my concurrent tests under the race detector in CI."

Here's how TSan works, because understanding it tells you what it can and cannot find. At runtime, TSan maintains *shadow memory*: for each application byte, it stores a set of recent accesses tagged with a vector clock — a logical timestamp per thread. Every load and store you do gets recorded into shadow memory. Every synchronization action (lock, unlock, atomic, thread create/join, channel op) updates the vector clocks so TSan knows the happens-before edges. When a new access lands on a byte that already has a recent conflicting access (same byte, at least one write) and the two accesses' vector clocks are *not* ordered — meaning no happens-before edge connects them — TSan reports a data race, with the stack trace of *both* accesses and the synchronization that's missing. The figure below traces that decision.

![A flow where shadow memory feeds two writes to the same address into a happens-before check that branches to no report when an edge exists and to a data race report with both stacks when no edge exists](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-4.png)

Let's run it on the lost-update counter. In Go this is one flag:

```bash
$ go test -race ./...
# or to run a program directly:
$ go run -race ./cmd/counter
```

The race detector prints something like this — and notice it gives you *both* sides of the race, which is the whole point:

```bash
==================
WARNING: DATA RACE
Read at 0x00c0000b4010 by goroutine 9:
  main.main.func1()
      /app/cmd/counter/main.go:14 +0x44

Previous write at 0x00c0000b4010 by goroutine 8:
  main.main.func1()
      /app/cmd/counter/main.go:14 +0x5a

Goroutine 9 (running) created at:
  main.main()
      /app/cmd/counter/main.go:11 +0xd8
Goroutine 8 (running) created at:
  main.main()
      /app/cmd/counter/main.go:11 +0xd8
==================
Found 1 data race(s)
exit status 66
```

Read that report like a debugger trained you to read a stack trace. The *address* (`0x00c0000b4010`) is the contended memory. The *read* and *previous write* are the two conflicting accesses, each with its own stack — both land on `main.go:14`, which is the `count++` line. The two goroutines are 8 and 9, both created on line 11 (the loop). There is no lock in either stack, so there is no happens-before edge: data race. TSan didn't guess. It observed two unordered accesses to the same byte and one of them was a write. That is a *proof*, not a heuristic.

The C and C++ story is identical in spirit. Compile with the thread sanitizer:

```c
// race.c — two threads, one unsynchronized counter
#include <pthread.h>
#include <stdio.h>

static long count = 0;

void *worker(void *arg) {
    for (long i = 0; i < 1000000; i++) count++; // data race
    return NULL;
}

int main(void) {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, worker, NULL);
    pthread_create(&t2, NULL, worker, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("count = %ld\n", count); // want 2000000
    return 0;
}
```

```bash
$ clang -fsanitize=thread -g -O1 race.c -o race
$ ./race
==================
WARNING: ThreadSanitizer: data race (pid=4127)
  Write of size 8 at 0x000000604070 by thread T2:
    #0 worker race.c:8 (race+0x... )
  Previous write of size 8 at 0x000000604070 by thread T1:
    #0 worker race.c:8 (race+0x... )
  Location is global 'count' at 0x000000604070 (race+0x...)
SUMMARY: ThreadSanitizer: data race race.c:8 in worker
==================
```

Same structure: two writes, same address, named global `count`, two threads, no synchronization. The fix is one line, and TSan goes silent — we'll do that in §8.

The crucial honesty here: **TSan only sees races that actually execute during the run.** It is *dynamic*. If a code path with a race never runs in your test, TSan reports nothing. This is why TSan is most powerful when paired with the reproduction techniques in §5 — you must *exercise* the racy interleaving for TSan to catch it. It also has overhead (typically 5–15x slower, 5–10x more memory), so you run it in CI and stress, not in production. The comparison table below puts it next to the other detectors so you reach for the right one.

One reading tip that saves real time: **a TSan report names two stacks, and the bug is almost never in either one — it's in the missing edge between them.** Beginners read the first stack, "fix" that line by adding a lock there, and reintroduce the race because the *other* access is still unsynchronized. Read both stacks, identify the shared address (TSan names the variable when it can), and ask "what synchronization, held across *both* accesses, would order them?" The fix goes on the *invariant*, not on whichever line TSan happened to print first. And when a data race is a *regression* — the test was clean last month and flaky now — TSan composes beautifully with `git bisect run`: write a wrapper that runs the test under `-race` in a loop and exits nonzero on the first race report, hand that script to `bisect run`, and git binary-searches the exact commit that introduced the unsynchronized access. A race that hid for 400 commits falls out in about $\log_2(400) \approx 9$ steps. See [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) for the full `bisect run` pattern.

![A matrix comparing TSan, Helgrind and DRD, jcstress, and stress with rr chaos across what each finds, its overhead, and when to reach for it](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-6.png)

A few notes on the alternatives in that table. **Helgrind** and **DRD** (both Valgrind tools) find data races and lock-ordering problems without recompiling — useful when you can't rebuild with `-fsanitize=thread` — but they're slower (20–50x) and noisier. **jcstress** is the JVM's concurrency stress harness: it runs tiny concurrency "experiments" millions of times across many thread configurations and reports which *results* were actually observed, which is the right tool for verifying Java Memory Model publication and ordering guarantees. Running Java with `-Xint` (interpreter only, no JIT) sometimes exposes ordering bugs the JIT was hiding and sometimes hides ones it created — it's a perturbation, not a detector. And `rr chaos` plus stress is your fallback for the *logic* races that TSan can't see by construction, which we get to next.

## 5. The method: reproduce it on demand

Here is the brutal truth about race conditions that nothing else in debugging prepares you for: **a bug you can't reproduce, you can't fix — and you can't even confirm you fixed it.** If a race fails 6 times in 2,000 runs, and you "fix" it and see it pass 2,000 times, you have learned almost nothing, because 2,000 clean runs is fully consistent with a 0.3% failure rate by chance. (The probability of seeing zero failures in 2,000 runs at a true 0.3% rate is about $0.997^{2000} \approx 0.0025$ — so a clean 2,000-run isn't even strong evidence.) The discipline of the whole series applies double here: [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging). For races, "reproduce" means *make the rare interleaving common*. There are five levers, and you stack them.

**Lever one: widen the window.** The race needs a specific thread to be preempted at a specific instruction. You can make that preemption far more likely by inserting a yield or a tiny sleep *inside the window* between the two operations that must be atomic. This is a diagnostic, not a fix — you're deliberately *provoking* the bug.

```go
// Deliberately widen the check-then-act window to force the race.
if !exists(key) {
    runtime.Gosched()              // yield: invite the scheduler to switch here
    // time.Sleep(time.Microsecond) // even more aggressive
    create(key)                    // now two goroutines reliably collide
}
```

In C the equivalent is `sched_yield()` or a `nanosleep` between the check and the act. With the window widened, a race that fired 6/2000 will fire 1800/2000. Now you have a reliable reproducer — and now, and *only* now, you can trust that making it pass means you fixed it.

**Lever two: scheduling pressure.** Run many more threads than you have cores, so the scheduler is forced to preempt constantly. Pin the process to a small number of CPUs so the OS context-switches aggressively. On Linux, `taskset -c 0,1` pins to two cores; combine with hundreds of threads to maximize interleaving churn. Run the whole thing under `stress-ng --cpu 8 --io 4` in another terminal so the box is loaded and time slices are short and unpredictable.

```bash
# Maximize preemption: pin to 2 cores, hammer the box, run the racy test.
$ stress-ng --cpu 8 --vm 2 --timeout 60s &
$ taskset -c 0,1 ./race_test
```

**Lever three: run it thousands of times.** A 0.3% race needs ~230 runs to have a 50/50 chance of catching it once, and ~1,500 runs to be 99% sure of catching it at least once. So loop. The single most useful shell snippet in this whole post is the repeat-until-fail loop:

```bash
# Run until it fails, counting iterations. Ctrl-C when you're convinced.
i=0
while ./race_test; do
  i=$((i+1))
  printf "\rpassed %d times" "$i"
done
echo
echo "FAILED on run $((i+1))"
```

For Go specifically, the test runner has this built in — and combined with `-race` it's the gold-standard reproducer:

```bash
# 5000 runs, race detector on, stop at first failure.
$ go test -race -count=5000 -failfast -run TestCreateIfAbsent ./...
```

**Lever four: deterministic and chaos schedulers.** `rr` (Mozilla's record-replay debugger) has a **chaos mode** (`rr record --chaos`) that deliberately perturbs thread scheduling to surface races, and once it records a failing run, you can replay it *deterministically* — the same interleaving, every time — and even step *backwards* through it. That's the holy grail for a race: a recording of the exact bad schedule you can replay under a debugger as many times as you like. There are also deterministic schedulers (e.g. `rr`'s replay, or language-level tools that serialize the scheduler) that remove timing nondeterminism entirely so a failing seed always fails.

**Lever five: stress the real system, not the unit.** Some races only appear under production-shaped concurrency — two real requests interleaving through a connection pool, a cache, and a database. Reproduce those with a load generator (`wrk`, `vegeta`, `k6`) pointed at a local instance, turned up until requests overlap. For those, the message-queue and database analogues matter: an "exactly once" guarantee that's really "at least once" creates the same duplicate-action race at the system level, and the cure is the same — see [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe).

The timeline below is the workflow these five levers compose into — the series loop, instantiated for races.

![A left-to-right timeline from observing a six in two thousand flake through widening the window, reproducing at high rate, detecting with TSan, fixing, and preventing with a clean five thousand run in CI](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-8.png)

#### Worked example: the counter that was wrong under load

Back to the intro bug. The symptom: a daily job counter that read 40,996 instead of 41,000 — about a 0.01% shortfall, invisible most days. Here is the full investigation, step by step, with the numbers.

**Observe.** I had production telemetry: `jobs_processed` (the counter) vs `jobs_emitted` (an independent downstream count). Over 30 days, the counter was short on 11 days, by 1 to 21. Never *over*. "Never over, sometimes under" is the fingerprint of a lost update — increments vanish, they don't duplicate.

**Reproduce.** Locally, the counter was *always* right, because my laptop ran the workers near-serially. I forced concurrency: 8 goroutines, 1,000,000 increments each, expected total 8,000,000. First run: 7,998,433. Short by 1,567. There it was — reproduced in one run, because I'd cranked the contention way past production levels. Lesson: production's "rare" is an artifact of low contention; crank contention and "rare" becomes "always."

**Detect.** I ran it under the race detector: `go run -race ./cmd/counter`. Instant `WARNING: DATA RACE`, both stacks on the `count++` line, no synchronization between them (the report in §4). Confirmed: data race, lost update.

**Quantify the loss.** With 8 goroutines × 1M increments and observed total 7,998,433, I lost 1,567 increments — about 0.02% — under heavy contention. The fingerprint matched production exactly: small, never-over, contention-dependent.

**Fix.** Replace `count++` with `atomic.AddInt64(&count, 1)`. (We'll show the diff and the proof in §8.) Re-run under `-race`: clean. Re-run 5,000 times: 8,000,000 every single time, zero shortfall. The flake rate went from "0.02% of increments lost under load" to *exactly zero*, provably, because an atomic add carries its own happens-before edge.

The thing I want you to take from this worked example is the *order of operations*: I reproduced before I theorized, I cranked contention to make the rare common, I let the tool name the exact line, and I confirmed the fix with thousands of runs — not three. That order is the difference between fixing a race and pretending to.

### Stress-testing your reproducer: what if it only reproduces under X?

Reproduction is rarely as clean as "crank the threads and it fails." Real races have escape hatches, and a good debugger thinks through them before giving up. Here are the ones I run into, and what each one tells you.

*What if it only reproduces under load?* Then your unit test, which runs the racy code in isolation on a quiet box, will never catch it — the time slices are too long and preemption too rare. This is the §5 lever-five situation: you need production-shaped concurrency. Point a load generator at a local instance and turn the request rate up until requests demonstrably overlap, then watch for the corrupted invariant. The 2003 blackout race was exactly this — invisible until the cascading-alarm load profile.

*What if it only reproduces in release builds?* Then you have a compiler-reordering or optimization-dependent race (§2). The optimizer at `-O2` reordered or hoisted something that `-O0` left in source order. Do *not* "fix" it by shipping `-O0` — that just hides it. Reproduce it at `-O2` under TSan (TSan works fine on optimized builds and is in fact recommended at `-O1`/`-O2`), and fix the missing happens-before edge so the optimizer is *allowed* to reorder and your code is still correct.

*What if it only reproduces on one host?* Suspect a hardware memory-model difference. x86 has a relatively strong memory model (it forbids many reorderings); ARM and POWER are far weaker and will surface store-load and store-store reorderings that x86 silently masked. A race that "only happens on the ARM build" is usually a real race that x86's strong model was papering over. The portable fix is the same — correct synchronization — and now it's correct on *every* architecture.

*What if it only reproduces after six hours?* Then the window is genuinely tiny and you're waiting for the dice to land. Don't wait — *widen the window* (lever one). A `sched_yield` or microsecond sleep inserted in the suspected gap can turn "once per six hours" into "once per second." If you don't know *where* the window is, that's a hypothesis to test: instrument the candidate critical sections, widen each in turn, and see which one makes the failure rate jump.

*What if it only reproduces when two specific requests interleave?* Then you have a logic race (a race condition), not a memory race, and TSan is blind to it. Reproduce it by firing exactly those two request types concurrently in a tight loop with the window widened — the §6 signup investigation is precisely this case.

*What if you can't attach a debugger in prod?* This is the common and frightening one. You cannot pause the payments process. So you don't — you capture *evidence* instead: enable the race detector in a canary or staging replica that takes mirrored traffic; add cheap, lock-free, sequence-numbered trace points around the suspect invariant and reconstruct the interleaving from logs after a failure; or arm an `assert` on the invariant that dumps a core on violation, then do post-mortem analysis on the corpse (§7). The discipline is to gather enough to reproduce *offline*, then move the whole fight to a box where you *can* attach a recorder.

The meta-point: every one of these "only reproduces under X" conditions is a *clue about the mechanism*, not a dead end. "Only in release" points at the optimizer. "Only on ARM" points at memory ordering. "Only under load" points at scheduling pressure. "Only when two requests interleave" points at a logic race. Read the escape hatch as a hypothesis about the cause.

## 6. The check-then-act trap, dissected

The lost-update counter is a *data race* — TSan finds it for you. The next pattern, **check-then-act**, is often a *race condition* with no data race at all, which means TSan may be completely silent while your program still double-creates rows, double-charges cards, or double-sends emails. You catch this one with reasoning, stress, and a load generator — not a sanitizer. Let me dissect the canonical case: "create the record if it doesn't already exist."

```python
# TOCTOU: check then act. Looks innocent. Double-creates under concurrency.
def ensure_user(db, email):
    if db.query("SELECT 1 FROM users WHERE email = ?", email) is None:  # CHECK
        # --- another request can run the same CHECK here and also see None ---
        db.execute("INSERT INTO users (email) VALUES (?)", email)        # ACT
        return "created"
    return "exists"
```

Every line here can be perfectly thread-safe. The database connection is fine. The query is fine. The insert is fine. And yet, run two requests for the same new email *at the same time*, and both execute the `SELECT`, both see "no such user," both proceed to `INSERT`, and now you have two users with the same email — or, worse, a crash on the second insert if you have a constraint, or a silent duplicate if you don't. The window is the gap between CHECK and ACT, and under concurrency that gap is wide open. The grid below freezes one doomed interleaving in time.

![A grid with two thread columns and time rows showing thread A checking and seeing no record, then thread B checking and seeing no record, then both creating, with the second create marked as a duplicate](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-5.png)

Notice what's missing from that picture: there is no *unsynchronized memory access*. Each thread's database operations might go through a fully serialized connection. So a memory-race detector has nothing to flag. This is the deep reason the kit and I keep insisting on the data-race-vs-race-condition distinction: **the most dangerous races are the ones your best tool cannot see.** You find them by asking, at every check-then-act, "what if another thread runs the same check in my gap?"

The TOCTOU pattern is everywhere once you have the eyes for it:

- **Filesystem:** `if not os.path.exists(path): open(path, "w")` — two processes both see "not exists," both create, one clobbers the other. The fix is `open(path, "x")` (Python) / `O_CREAT | O_EXCL` (C), which makes "create only if it doesn't exist" a *single atomic syscall* the kernel arbitrates. TOCTOU file races are also a classic *security* vulnerability — a privileged program checks a file's permissions, then opens it, and an attacker swaps the file for a symlink in the gap.
- **Database:** the `ensure_user` above. The fix is to push the atomicity into the database: a `UNIQUE` constraint on `email` plus `INSERT ... ON CONFLICT DO NOTHING` (Postgres) / `INSERT IGNORE` (MySQL), or `MERGE`. The database engine, holding the right locks, makes check-and-act atomic for you. This is the same family of guarantees as transaction isolation — see [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), where "lost update" and "write skew" are exactly these races named formally.
- **Connection pool:** `if pool.has_free(): conn = pool.take()` — two threads see a free slot, both take, one gets nothing or a slot that's already gone. The fix is an atomic `try_take()` that checks-and-removes under one lock.
- **Money:** `if balance >= amount: balance -= amount` — two withdrawals both see sufficient balance, both deduct, the account goes negative. The fix is to make the read-check-decrement atomic (a single locked critical section or a conditional `UPDATE ... WHERE balance >= amount`).

The unifying fix is one sentence: **make the check and the act a single atomic operation that no other thread can interleave into.** Sometimes that's a mutex spanning both. Sometimes it's a compare-and-swap. Sometimes it's pushing the atomicity down to a layer that already arbitrates it — the kernel (`O_EXCL`), the database (`UNIQUE` + upsert), the message broker (idempotency key). What never works is wrapping the check in one lock and the act in another lock; that just makes each half thread-safe while leaving the *combination* racy.

The kernel-level fix for the filesystem version is worth seeing in C, because it's the canonical example of replacing a racy two-step with one atomic syscall:

```c
// RACY: stat-then-open is a two-step TOCTOU window.
if (access(path, F_OK) != 0) {     // CHECK: does it exist?
    int fd = open(path, O_CREAT | O_WRONLY, 0644); // ACT: create it
    // two processes can both pass the access() check and both create
}

// ATOMIC: O_EXCL makes "create only if absent" a single kernel-arbitrated op.
int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0644);
if (fd < 0 && errno == EEXIST) {
    // someone else created it first — handle the "already exists" case
} else if (fd >= 0) {
    // we created it; exactly one process reaches here
}
```

With `O_CREAT | O_EXCL`, the kernel guarantees that of any number of racing processes, *exactly one* gets a fresh `fd` and the rest get `EEXIST`. There is no window because the check and the act are one syscall the kernel serializes. This is the same shape as the database `INSERT ... ON CONFLICT` fix — let the arbitrating layer make it atomic.

#### Worked example: the signup that double-created users

A real-shaped incident. A signup endpoint occasionally created two accounts for one email — support saw it maybe once a week out of ~50,000 signups, a rate around 0.002%. No errors logged. The `ensure_user`-style code looked correct, passed every unit test, and TSan (well, this was Python, but the equivalent reasoning) had nothing to say because there was no memory race.

**Hypothesize.** "Never over" would be a lost update; here we had *duplicates* — "sometimes too many." Too-many is the fingerprint of check-then-act: two actors both passing the same check. Hypothesis: two concurrent requests for the same email both pass the existence check before either inserts.

**Reproduce.** I couldn't trigger it by hand — one request at a time always worked. So I widened the window and stressed it. I added a `time.sleep(0.05)` between the check and the insert (lever one), then fired 50 concurrent requests for the same brand-new email with a load generator (lever five):

```bash
# 50 concurrent identical signups; count how many accounts got created.
$ seq 50 | xargs -P 50 -I{} curl -s -X POST localhost:8080/signup \
    -d 'email=race@example.com' > /dev/null
$ psql -c "SELECT count(*) FROM users WHERE email='race@example.com';"
 count
-------
    37
```

Thirty-seven accounts for one email. The bug went from "once a week, can't reproduce" to "37 duplicates in one command" — because I widened the window and cranked the concurrency. *Now* I had a reproducer I could trust.

**Fix.** I added a `UNIQUE` constraint on `email` and changed the insert to `INSERT ... ON CONFLICT DO NOTHING RETURNING id`, then treated "no row returned" as "already existed." The database now arbitrates the check-and-act atomically: of 50 concurrent inserts, exactly one wins and 49 are no-ops.

**Prove it.** I re-ran the same 50-way concurrent command with the `sleep` still in place (the worst case): `count` = 1. I removed the sleep and ran it 1,000 times in a loop: 1,000 times, `count` = 1. The duplicate rate went from ~0.002% to provably zero, and — this matters — I proved it *with the window still artificially widened*, so I know it's the atomicity that fixed it, not luck. The illustrative numbers (37 duplicates, then a clean 1,000-run) are the kind you should demand of yourself before you close a race ticket.

## 7. Why a debugger is nearly useless here

Everywhere else in this series, the interactive debugger is the microscope — see [the debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it). For races, it's nearly the worst tool you can pick, and the reason is fundamental, not incidental. **A breakpoint changes the timing.** The moment you pause thread 1 at a breakpoint, the scheduler hands the CPU to the other threads, which now run *unobstructed* through the window you were trying to observe — so by the time you step thread 1 forward, the race has either already happened (and the state looks fine) or can no longer happen (because the other thread sailed through). You perturb the very thing you're measuring. This is the **heisenbug**: a bug that changes or vanishes when you observe it. Races are the purest heisenbugs there are, and they're important enough that they get their own post — [heisenbugs that vanish when you look](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look) — which goes deeper on the observer effect across all of debugging.

It's not just breakpoints. *Everything* you do to observe a race perturbs it:

- A `print` or log line adds I/O, which is slow and synchronizing — it can serialize threads that were racing and hide the bug, or it can change the timing just enough to make it appear.
- Compiling with `-O0` (no optimization) to get clean debug info changes the instruction scheduling and removes the compiler reordering — so a race that only manifests at `-O2` *vanishes* at `-O0`. (This is the inverse heisenbug: the debug build is the one that works.)
- Running under `gdb` slows the process and changes scheduling. Single-stepping serializes everything.
- Even adding an unrelated variable can shift memory layout and cache-line sharing, changing false-sharing behavior and the race window.

So what *do* you use? Tools that observe *without* perturbing the timing of a live run:

- **TSan / `-race`**: instruments at compile time and observes happens-before relationships, not wall-clock timing, so it finds the race regardless of whether *this particular run's* timing triggered the visible symptom. It does have overhead, but it's not trying to catch the bug "in the act" — it's reasoning about ordering.
- **`rr` record-replay**: record the failing run *once* (chaos mode helps you get one), then replay it deterministically as many times as you want, stepping forward *and backward*, with zero further nondeterminism. This is the one way to use a debugger on a race that actually works — you debug a *recording* of the bad schedule, not a live process whose schedule you're disturbing.
- **Post-mortem analysis**: a core dump captures the state at the instant of failure (a corrupted invariant, a torn value), and you inspect the corpse without perturbing anything because it's already dead. This pairs beautifully with `assert`s on invariants that dump core when violated.
- **Logging that records *order*, not just events**: instead of trying to catch the race interactively, add cheap, lock-free, timestamped (or sequence-numbered) trace points and reconstruct the interleaving *after* the fact. A per-thread ring buffer flushed on failure tells you the order without the heavy synchronization of a `print`.

The rule I follow: **never try to catch a race by stepping through it.** Reproduce it under a detector or a recorder, let the tool name the conflicting accesses, and reason about the interleaving from the report. If you find yourself setting a breakpoint inside a critical section hoping to "watch the race happen," stop — you're about to spend three hours watching the bug refuse to show up.

## 8. The fix: establish the ordering you need

Detecting and reproducing are most of the battle; the fix, once you know *which* of the two bugs you have, is usually short. But "short" is not "obvious," because there are five genuinely different fix strategies and choosing the wrong one either doesn't work or kills performance. Here they are, in rough order of how often I reach for them.

**1. A mutex around the whole invariant.** The default, and the one you should reach for unless you have a measured reason not to. The discipline that makes mutexes *correct* is: identify the invariant (the set of variables that must change together to stay consistent), and hold one lock for the *entire* duration of every read-modify-write of that invariant. The classic mistake is locking too *little* — locking each individual access but not the combination — which fixes the data race and leaves the race condition. Lock the *unit of consistency*, not the individual variables.

```go
// FIX 1: mutex around the whole check-then-act invariant.
type Registry struct {
    mu    sync.Mutex
    items map[string]bool
}

func (r *Registry) CreateIfAbsent(key string) bool {
    r.mu.Lock()
    defer r.mu.Unlock()
    if r.items[key] {        // CHECK and
        return false         //   ACT are now
    }                        //   one atomic unit:
    r.items[key] = true      //   no thread can
    return true              //   interleave between them
}
```

**2. Atomics with the right memory order.** For a single shared scalar (a counter, a flag, a reference), an atomic is faster than a mutex and carries its own happens-before edge. The counter fix is one line:

```go
// FIX 2: atomic add. No lock, no lost update, no data race.
var count int64
// ... in each goroutine:
atomic.AddInt64(&count, 1)   // read-modify-write as ONE atomic op
```

```c
// C11 atomics version of the counter.
#include <stdatomic.h>
static atomic_long count = 0;
// ... in worker:
atomic_fetch_add(&count, 1); // atomic, with sequential consistency by default
```

The before/after of this fix is the cleanest measured result in the post, and worth seeing as a picture: same 8 goroutines, same million increments each, but the race flag goes silent and the total is exact on every run.

![Two columns contrasting the unsynchronized counter which the race flag warns on and totals seven point three million against the atomic add version which the race flag passes and totals exactly eight million every run](/imgs/blogs/race-conditions-the-hardest-bugs-to-catch-7.png)

A word on *memory order*, because it's where atomics get subtle. The default in Go's `sync/atomic` and C++'s `std::atomic` (without arguments) is **sequentially consistent** — the strongest, simplest, and slightly slower ordering, and the right default. You only reach for weaker orderings (`memory_order_acquire`/`release`/`relaxed` in C++) when you've measured that the atomic is a hot spot *and* you can prove the weaker ordering still gives you the happens-before edges you need. For 99% of code, sequential consistency is correct and the performance difference is noise. The danger of `relaxed` is that it gives you atomicity (no torn value) but *no ordering guarantee with respect to other variables* — so you can fix the torn-read and reintroduce a publication race. When in doubt, use the default.

**3. Immutability.** A value that never changes after construction cannot be raced on — there's no write to conflict with the reads. This is the cheapest correctness guarantee in concurrency: build the object fully, publish it safely *once* (via a `final` field in Java, an atomic store, or a channel), and then share the reference freely with zero further synchronization. Functional-style "copy-on-write" — produce a new immutable value and atomically swap the pointer — turns a thorny shared-mutable-state problem into a single atomic pointer store.

**4. Message passing / channels.** Don't share memory; communicate. If only *one* goroutine ever touches a piece of state, and other threads send it messages over a channel, there's no shared mutable state to race on. Go's mantra — "don't communicate by sharing memory; share memory by communicating" — is exactly this. The channel send/receive *is* the happens-before edge, so handing data through a channel publishes it safely for free.

```go
// FIX 4: serialize all mutation through one owner goroutine.
type incr struct{}
func counter(in <-chan incr, out chan<- int64) {
    var count int64           // owned by THIS goroutine only
    for range in {
        count++               // no lock needed: single owner
    }
    out <- count
}
```

**5. Compare-and-swap for lock-free updates.** When you need a lock-free read-modify-write of a shared value (a hot counter, a lock-free stack, an optimistic update), the primitive is **compare-and-swap** (CAS): atomically, "if the value is still what I read, set it to my new value; otherwise tell me it changed and I'll retry." CAS is how lock-free data structures and optimistic concurrency control work — and it's the same idea as a database's optimistic locking (`UPDATE ... WHERE version = ?`). It's powerful but easy to get wrong (the ABA problem, livelock under heavy contention), so reach for it only when a mutex is a measured bottleneck.

```go
// FIX 5: lock-free max via compare-and-swap retry loop.
func atomicMax(addr *int64, val int64) {
    for {
        old := atomic.LoadInt64(addr)
        if val <= old {
            return                                  // already >= val
        }
        if atomic.CompareAndSwapInt64(addr, old, val) {
            return                                  // we won the swap
        }
        // someone else changed it; loop and retry
    }
}
```

Here's the decision table I keep in my head for choosing among these:

| Situation | Reach for | Why |
| --- | --- | --- |
| Multi-variable invariant must stay consistent | Mutex around the whole unit | Only a held lock makes a *sequence* atomic |
| Single shared scalar counter or flag | Atomic add / store | Faster than a lock, carries happens-before |
| Read-heavy, rarely-written config | Immutability + atomic pointer swap | Reads need zero synchronization |
| One logical owner of the state | Channel / message passing | The send is the happens-before edge |
| Hot value, mutex is a measured bottleneck | Compare-and-swap loop | Lock-free, but watch ABA and retry storms |
| Check-then-act across a process boundary | Push atomicity down (DB UNIQUE, `O_EXCL`) | Let the layer that arbitrates do it |

And here's the trade-off table for the cost side, because every one of these has a price:

| Technique | Catches | Cost / risk | When NOT to use it |
| --- | --- | --- | --- |
| Coarse mutex | Both data races and race conditions | Serializes; can become the bottleneck | Hot path where the lock dominates |
| Fine-grained locks | Same, with more parallelism | Lock-ordering deadlocks; hard to reason about | When a coarse lock is fast enough |
| Atomics (seq-cst) | Single-scalar data races | Only one variable at a time | A multi-variable invariant |
| Atomics (relaxed/acq-rel) | Torn reads, with care | Easy to reintroduce ordering bugs | Anytime you're not 100% sure of the order |
| Immutability | All races on that value | Allocation churn; redesign cost | Genuinely hot mutable state |
| Channels | Shared-state races | Throughput overhead; can deadlock | Ultra-low-latency single-variable updates |

## 9. War story: the races that made history

Races aren't an academic curiosity; some of the most expensive and dangerous software failures in history were race conditions, and walking through a few sharpens your instinct for where they hide.

**Therac-25 (1985–1987).** The Therac-25 was a radiation therapy machine, and a race condition in its control software contributed to several patients receiving massive radiation overdoses, some fatal. The relevant flaw: an operator who edited treatment parameters *quickly* — within a roughly eight-second window — could trigger an interleaving where a one-byte shared flag was set inconsistently between two concurrent tasks, leaving the machine in high-energy mode without the beam-spreading hardware in place. It was a check-then-act and a shared-flag publication race, and it only manifested for operators *fast enough* to hit the window — which is exactly why it survived testing (testers were slow and careful) and emerged only after operators got fluent. The lesson that still applies: a race that depends on a *narrow timing window* is invisible to anyone who doesn't hit the window, and your testers almost never do. You must *deliberately widen the window* (lever one from §5) to find it — the machine's own designers never did.

**The 2003 Northeast blackout.** A race condition in the alarm system of a power grid management application (GE XA/21) contributed to the largest blackout in North American history, affecting ~55 million people. Multiple processes contended over a shared data structure without adequate locking; under the load of cascading alarms, the race stalled the alarm subsystem, so operators didn't see the developing failure for over an hour. The race was latent for years and only triggered under the specific high-concurrency load of a real cascading event — the textbook "only reproduces under production-scale load" stress case from §5. It took the vendor weeks of analysis to find it, precisely because it couldn't be reproduced without the production load profile.

**The "double-checked locking is broken" reckoning (late 1990s, Java).** Not a single incident but a community-wide one: a wildly popular singleton-initialization idiom — check a field for null without a lock, lock and check again, then construct — was *published as correct* in books and used everywhere, and it was *broken* under the Java Memory Model because the reference could be published before the object's fields were written (the store-buffer reordering from §2). The fix required the field to be `volatile`, and the whole episode is why Java 5 rewrote the memory model to give `volatile` proper acquire-release semantics. The lesson: an idiom can be wrong *even when thousands of experienced engineers use it daily and it "works,"* because "works in practice on this hardware" is not "correct per the memory model." This is the deepest reason to run a detector instead of trusting your eyes — your eyes and your test runs both share the same blind spot.

I'm presenting these as accurately as I can from the public record; the Therac-25 and 2003 blackout are extensively documented incidents, and the double-checked-locking saga is well-known Java history. The common thread across all three is the property we opened with: **the bug was in the timing, was latent for a long time because the triggering interleaving was rare, and only emerged under conditions (operator speed, production load, specific hardware) that ordinary testing didn't reproduce.** That is the signature of the entire bug class.

There's a modern, lower-stakes version of this you've probably lived through: the metrics dashboard that's "a little off." A request counter that's read-modify-written without synchronization undercounts under load — exactly the lost-update race from the intro — and because the error is small and never *over*, nobody notices for months. Then someone reconciles two independent counts, finds a 0.02% discrepancy, and chases it for a day before realizing the counter itself is racy. The damage isn't a radiation overdose; it's a slow erosion of trust in your own telemetry, which is its own kind of expensive. The fix is the one-line atomic from §8, and the prevention is the CI race detector — which would have flagged that `count++` on the first PR. Every one of these stories, from the fatal to the merely annoying, ends the same way: the race was findable by a tool the team didn't run, and reproducible by a stress the team didn't apply. The whole craft of this post is making sure you're not the next entry in that list.

## 10. How to reach for this (and when not to)

Concurrency tools have real costs, and applying them reflexively wastes time and sometimes makes things worse. Here is my decisive guidance on what to reach for and, just as important, what to skip.

**Always run the race detector in CI.** If you write Go, `go test -race` in CI is non-negotiable — it's a few flags and it turns "data races we'll find in production" into "data races we find on the PR." For C and C++, a `-fsanitize=thread` build of your test suite in CI is the same leverage. The overhead (5–15x) is irrelevant for a test run and priceless for a production incident avoided. This single practice catches more real concurrency bugs than any amount of code review.

**Reproduce before you "fix."** Never close a race ticket on the strength of "it passed a few times now." A race that fails 0.3% of the time passes a few times *by definition*. Widen the window, crank the concurrency, run it thousands of times — get the failure rate up to where a fix visibly drives it to zero. If you can't reproduce it at an elevated rate, you cannot prove you fixed it, and you'll be back.

**Don't attach a debugger to a live race, and especially not in prod.** Stepping through a race perturbs the timing into hiding (§7). And attaching `gdb` to a production process — payments, anything latency-sensitive — pauses it and can cascade into timeouts and a worse incident than the bug. Use a detector, a recorder (`rr`), or post-mortem core analysis instead.

**Don't reach for lock-free and weak memory orderings prematurely.** A coarse mutex is correct, simple, and fast enough almost everywhere. Compare-and-swap loops and `memory_order_relaxed` are for *measured* hot spots where you've proven the lock is the bottleneck and you can prove the weaker ordering preserves the happens-before edges you need. Premature lock-free code is how you turn one race into three.

**Don't add a `sleep` and call it a fix.** Widening a *window* is a diagnostic. *Narrowing* a window with a sleep to make the bug rarer is not a fix — it's hiding the bug under a thinner blanket. The race is still there; you've just made it fire at 3am under unusual load instead of in your test. The only real fix is to establish the ordering with a synchronization primitive.

**Don't fix a race condition with more locking when the real fix is atomicity-at-the-right-layer.** If two processes check-then-act on a database row, no amount of in-process locking helps — the contention is *across* processes. Push the atomicity down to the layer that arbitrates it: a `UNIQUE` constraint and upsert in the database, `O_EXCL` in the kernel, an idempotency key in the message broker. The right fix is often *removing* your clever in-process coordination and letting a layer that already serializes do the job.

**When the data is read-mostly, prefer immutability over locking.** A config or lookup table that's written rarely and read constantly should be an immutable value swapped atomically, not a mutex-guarded mutable structure that every reader contends on. You eliminate the race *and* the read contention.

## Key takeaways

- **A data race and a race condition are different bugs.** A data race is two unsynchronized accesses to the same memory with at least one write — undefined behavior, and a detector finds it. A race condition is order-dependent logic that can be wrong even with perfect locking — and a detector usually can't see it.
- **"No happens-before edge" means no guarantee at all.** Without a synchronization action ordering your write before another thread's read, the memory model promises *nothing* about visibility — not "eventually," nothing. The compiler reorders, the store buffer delays, caches are lazily coherent.
- **Run the race detector in CI.** `go test -race` and `-fsanitize=thread` turn "data races we ship" into "data races on the PR." It's the highest-leverage concurrency practice there is. But it only finds races that actually *execute*, so pair it with stress.
- **Reproduce before you fix, and prove the fix.** Widen the window (`Gosched`/`sched_yield`), crank concurrency (`taskset`, `stress-ng`, more threads than cores), and run thousands of times. A race that fails 0.3% of the time passes a "quick check" by definition; only an elevated, repeatable failure rate lets you prove you fixed it.
- **A debugger is nearly useless on a race.** Breakpoints, logging, and `-O0` all change the timing and hide the bug — that's the heisenbug. Use a detector, `rr` record-replay (chaos + deterministic replay), or post-mortem cores instead of stepping through it live.
- **Lock the whole invariant, not the individual accesses.** The most common failed fix locks each read and write but leaves the *combination* racy. Make the unit of consistency atomic.
- **Match the fix to the bug:** atomic for a single scalar, mutex for a multi-variable invariant, immutability for read-mostly data, channels when one goroutine owns the state, CAS for measured lock-free hot spots — and push atomicity down to the database or kernel for cross-process check-then-act.
- **Default to sequential consistency.** Reach for relaxed/acquire-release memory orderings only on a *measured* hot path where you can prove the weaker ordering still gives the edges you need.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post instantiates for races.
- [Reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — the discipline of turning a flake into a reliable failure, which races demand more than any other bug.
- [Heisenbugs that vanish when you look](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look) — the observer effect across all of debugging; races are the purest heisenbugs.
- [Deadlocks, livelocks, and starvation](/blog/software-development/debugging/deadlocks-livelocks-and-starvation) — the sibling concurrency failures and the lock-ordering discipline that prevents them.
- [Isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) — lost update and write skew are these same races, named formally at the database layer.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the cross-process version of "make check-and-act atomic."
- The **ThreadSanitizer** documentation (Clang/LLVM) and the **Go race detector** guide (`golang.org/doc/articles/race_detector`) — the canonical references for the detectors used throughout this post.
- *Java Concurrency in Practice* by Brian Goetz — still the clearest book on safe publication, happens-before, and the memory model; and the **C++ memory model** references for `std::atomic` orderings.

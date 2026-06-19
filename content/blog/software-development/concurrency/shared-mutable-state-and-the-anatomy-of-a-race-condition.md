---
title: "Shared Mutable State and the Anatomy of a Race Condition"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Walk the canonical lost-update race instruction by instruction, then build the three properties — atomicity, visibility, ordering — that make shared state safe."
tags:
  [
    "concurrency",
    "parallelism",
    "race-condition",
    "shared-state",
    "atomicity",
    "thread-safety",
    "data-race",
    "fundamentals",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-1.png"
---

Two threads. One shared counter, starting at zero. Each thread runs a loop that increments the counter one million times. When both finish, the counter should hold exactly two million. You run the program. The number on the screen is `1,983,402`.

You run it again. This time it is `1,991,067`. A third run: `2,000,000` — correct, infuriatingly. A fourth: `1,974,815`. The arithmetic is not wrong. The loop bounds are not wrong. There is no `if` that occasionally skips a step, no off-by-one, no integer overflow. The code does exactly what it says: increment, two million times, total. And yet roughly one in every hundred increments simply evaporated, and the count that vanished is different every run. This is the single most important bug in all of concurrent programming, and it has a name: the **lost update**, the canonical symptom of a **race condition**.

![two threads interleaving a load modify store on a shared counter so one increment is lost and the final value is one instead of two](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-1.png)

The figure above is the whole bug in seven steps, and by the end of this post you will be able to read it the way a doctor reads an X-ray. We are going to take the most innocent-looking line of code in your editor — `count++`, or `balance += amount`, or `cache[key] = value` — and put it under a microscope. We will watch it decompose into three separate machine instructions. We will trace the exact interleaving of two threads that loses an update, with real numbers in real registers. We will explain why the bug is nondeterministic, why your tests pass and production fails, and why "works on my machine" is the most dangerous sentence in this entire field. Then we will name the three properties you must establish over every piece of shared mutable state — **atomicity**, **visibility**, and **ordering** — and show a distinct bug for each. We will catalog the two shapes almost every race takes. And we will preview the fixes — a lock, an atomic — without letting the fixes distract from the point, which is the *diagnosis*. If you cannot see the race, you cannot fix it.

This post is the deep dive on the root hazard of the whole series. The recurring spine of "Concurrency & Parallelism, From the Ground Up" is one sentence: **shared mutable state plus nondeterministic scheduling equals the hazard.** The [intro post](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) argued that you cannot avoid concurrency; this post shows you precisely *why* it is hard, at the level of individual instructions. Everything that comes after — mutexes, the memory model, atomics, lock-free data structures, channels, actors — is a different answer to the question this post poses. So slow down here. Get this right, and the rest of the series is a series of refinements. Get it wrong, and you will spend your career shipping bugs that only appear on Fridays.

## The setup: what "shared mutable state" actually means

Before we disassemble anything, let us be precise about the three words that make the hazard, because each one is load-bearing and removing any one of them removes the bug entirely.

**Shared.** Two or more threads of execution can reach the same piece of data. In a multithreaded program, threads share an address space — the same heap, the same globals — so a pointer in one thread names the same bytes as the same pointer in another. (Processes, by contrast, have separate address spaces; that is why message-passing across processes sidesteps a lot of this, a theme we return to in later tracks.) If only one thread can ever touch the data, there is no race. A variable on a thread's own stack, never handed to anyone else, is private and safe.

**Mutable.** The data changes after creation. If a value is written once and only ever read afterward — a configuration constant, an immutable string, a frozen snapshot — then no matter how many threads read it concurrently, they all see the same thing and nothing can be lost. Immutability is not a workaround; it is one of the four genuine fixes we will reach at the end. The bug needs writes.

**State.** There is something whose value at one moment depends on its value at a previous moment. A counter's new value is its old value plus one. A balance's new value is its old value minus a withdrawal. This dependency — read the old, compute the new from it, write the new — is the read-modify-write pattern, and it is the beating heart of the lost update.

Put them together and you have shared mutable state. Now add the second half of the hazard: **nondeterministic scheduling.** Your two threads do not run in lockstep. The operating system scheduler decides, at moments you do not control and cannot predict, to pause one thread and resume another. On a multicore machine the threads may also be running *genuinely simultaneously* on different cores, their instructions interleaving in real, physical time. You wrote two sequential programs; the machine runs them as one nondeterministic braid. The race lives in the braid.

Here is the running example we will return to throughout the post and the whole series. A shared counter, two threads, each incrementing in a tight loop. Here it is in Go, where launching a goroutine is a single keyword:

```go
package main

import (
	"fmt"
	"sync"
)

var count int // shared, mutable state

func main() {
	var wg sync.WaitGroup
	wg.Add(2)
	for i := 0; i < 2; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < 1_000_000; j++ {
				count++ // the innocent-looking line
			}
		}()
	}
	wg.Wait()
	fmt.Println("count =", count) // want 2000000, often less
}
```

Two goroutines, each looping a million times, each running the single statement `count++`. The expected answer is two million. Run it and you will very often get less. The same program in Rust, deliberately written the unsafe way to expose the same bug — note that Rust normally *refuses* to compile shared mutable access without synchronization, so we have to reach for a raw pointer to even express the race:

```rust
use std::thread;

static mut COUNT: i64 = 0; // shared, mutable, and unsynchronized

fn main() {
    let handles: Vec<_> = (0..2)
        .map(|_| {
            thread::spawn(|| {
                for _ in 0..1_000_000 {
                    // SAFETY: this is intentionally a data race for demonstration.
                    unsafe {
                        COUNT += 1; // the innocent-looking line
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    unsafe {
        println!("count = {}", COUNT); // want 2000000, often less
    }
}
```

The fact that you must write `unsafe` and `static mut` in Rust to express this bug is not a side note — it is one of the most important design decisions in modern systems languages, and we will come back to it. For now, hold both programs in your head. They are the same bug in two languages. The question is: where does the update go?

## What `count++` really is: three steps, not one

The illusion that breaks everything is that `count++` is a single, indivisible action. You wrote one token. The processor does not see one token. It sees a sequence of at least three separate operations, because the value lives in main memory and the arithmetic happens in a register, and getting a value from memory into a register, changing it, and putting it back are distinct steps.

![the single source token count plus plus shown as one operation in the naive view versus three machine operations load add store in the truth view](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-2.png)

Let us prove this rather than assert it. Compile the increment and read the actual machine instructions. On x86-64, a non-atomic increment of a memory location through a register compiles to something close to this:

```asm
mov  rax, QWORD PTR [count]   ; step 1: LOAD  — copy count from memory into register rax
add  rax, 1                   ; step 2: MODIFY — add 1 to the private copy in rax
mov  QWORD PTR [count], rax   ; step 3: STORE — copy rax back into count in memory
```

Three instructions. A **load** that reads the current value of `count` out of memory into a CPU register. A **modify** that adds one to the register — note that this arithmetic happens on a *private* copy, in a register that belongs only to this core at this moment, and the value in memory is untouched while this happens. And a **store** that writes the register back to memory. Between the load and the store, the register holds a snapshot of `count` from a particular instant — and if `count` changes in memory after the load but before the store, the store does not know. It writes back what it computed from the *stale* snapshot, silently overwriting whatever happened in between.

(You may object: x86 has a single-instruction memory increment, `inc QWORD PTR [count]`, and many compilers emit it for the unsynchronized version. Does that make it atomic? No — and this is a beautiful, instructive trap. A single *instruction* is not the same as an *atomic* operation. Under the hood, `inc [mem]` still performs a read-modify-write micro-sequence on the memory bus, and on a multicore machine another core can interleave with it. To make it truly atomic you must prefix it with the `lock` prefix: `lock inc QWORD PTR [count]`, which asserts a bus lock or cache-line lock for the duration. The compiler emits the bare `inc` for ordinary code and the `lock`-prefixed form only when you ask for an atomic. The lesson: atomicity is a property the hardware must be explicitly told to provide; you never get it for free from "it's just one line.")

The same decomposition exists in every language, because it is a property of the machine, not the syntax. In Java, `count++` on a field compiles to JVM bytecode that is explicitly four operations — `getfield` (load), `iconst_1`, `iadd` (modify), `putfield` (store) — and the bytecode interpreter or JIT can be preempted between any of them. In Rust, `COUNT += 1` lowers to the identical load/add/store. The high-level syntax hides three machine steps in *all* of them; the hiding is exactly the trap. There is no language in which `x = x + 1` over shared memory is magically atomic unless you reach for an atomic type. The closest any mainstream language comes is to *refuse to let you share the variable at all* without synchronization — which is precisely what Rust's borrow checker does, and why the demonstration above needed `unsafe`.

![a single increment shown as a vertical sequence of read modify and write back steps with a gap the scheduler can cut between them](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-5.png)

The figure above stacks the three steps with the crucial detail marked: there are *gaps* between them, and the scheduler can cut at any gap. Hold this stack in mind. The whole bug is what happens when two threads' stacks interleave at those gaps.

There is a second layer of subtlety that sharpens the picture: on a real multicore machine, the "memory" the load reads and the store writes is not one flat array of bytes that all cores see identically. Each core has private L1 and L2 caches; the shared value `count` lives in a 64-byte **cache line** that gets copied into a core's cache when that core touches it. The hardware runs a **cache-coherence protocol** (MESI and its relatives) to keep cores from holding contradictory copies — when one core wants to write a line, it must first take that line in an exclusive state, invalidating every other core's copy. This is why the racy counter gets dramatically *slower* as you add threads, not just wrong: the cache line holding `count` ping-pongs between cores, each write forcing a coherence transaction that costs tens to hundreds of nanoseconds. So the unguarded counter is a double loser — it produces wrong answers *and* it is slow, because the very sharing that makes it race also makes it bounce the line across the interconnect. We measure this slowdown later; for now, register that "shared mutable state under contention" is expensive at the hardware level even before correctness enters the picture. The cache hierarchy is explored in depth in the [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) from the HPC series; here the relevant fact is just that the load and the store are not free, and they are not instantaneous, which is part of what makes the window real.

To make the stakes visceral, swap the counter for a bank balance, because losing a count of website visits is annoying but losing money is a lawsuit. Consider an account with a balance of `\$100`. Two transactions arrive at the same instant: a deposit of `\$50` and a withdrawal of `\$30`. The correct final balance is `\$120`. Each transaction is a read-modify-write: read the balance, compute the new balance, write it back. We will return to this in the worked example, but you can already feel the danger — if both transactions read `\$100` before either writes, one of them is going to compute its answer from a balance that is about to be wrong, and write back a number that erases the other transaction entirely.

## The interleaving that loses an update

Now we braid two threads' three-step sequences together and watch an update die. This is the mechanism — the precise, instruction-by-instruction trace — and it is the single most important thing to internalize in this post.

Start with `count = 0`. Thread T1 and thread T2 both want to execute `count++`. Each will perform load, modify, store. The scheduler is free to run their six instructions in *any* interleaving that preserves each thread's internal order (T1's load before T1's store, T2's load before T2's store), but with no ordering guarantee *between* the threads. There are many possible interleavings. Most of them produce the correct answer of 2. At least one produces 1. Here is that one, the lost update.

#### Worked example: the exact six-step trace that loses an update

Let `count` start at `0`. The two threads each hold a private register; call them `r1` for T1 and `r2` for T2. Trace the interleaving step by step, watching both memory and the registers:

1. **T1 load** — T1 reads `count` from memory into `r1`. Now `r1 = 0`, memory `count = 0`.
2. **T2 load** — before T1 can do anything else, the scheduler switches to T2 (or T2 runs simultaneously on another core). T2 reads `count` into `r2`. Now `r2 = 0`, memory `count = 0`. **This is the fatal moment: T2 has read a value that T1 is already in the middle of changing.**
3. **T1 modify** — T1 adds one to its private register. `r1 = 0 + 1 = 1`. Memory still `0`.
4. **T1 store** — T1 writes its register back. Memory `count = 1`. From T1's point of view, everything is correct.
5. **T2 modify** — T2 adds one to *its* private register, which still holds the stale `0`. `r2 = 0 + 1 = 1`. T2 has no idea T1 ever ran.
6. **T2 store** — T2 writes its register back. Memory `count = 1`, **overwriting T1's store with the same value.**

Two increments executed. Final value: `1`, not `2`. One update is lost. The lost update is not lost in transit, not corrupted, not partially written — it is *overwritten*, completely and silently, by a store that was computed from a stale read. Both threads did exactly three correct instructions. The braid is what was wrong.

The first figure in this post is exactly this trace, rendered as a timeline of seven events (t1 through t7), ending in `final count = 1, one update lost`. Re-read it now; it should be transparent.

Notice the shape of the failure window. The danger exists from the moment T1 loads (step 1) until the moment T1 stores (step 4). If T2's load lands anywhere inside that window — after T1's load, before T1's store — T2 reads a value that is about to be obsolete, and one of the two updates will be lost. The width of that window, measured in nanoseconds, is the size of the target the scheduler has to hit for the bug to fire. For a single `count++` the window is a handful of nanoseconds wide, which is why it fires rarely on a single increment but *constantly* across millions of them.

Let us quantify exactly how much gets lost, because "roughly one percent" deserves a derivation. Suppose two threads each do $N$ increments. The number lost is bounded between $0$ (every interleaving happened to be clean) and $N$ (a pathological worst case where the threads ping-pong perfectly). The empirical loss depends on how often a store from one thread lands inside the other thread's load-store window, which depends on the relative speeds of the threads, the scheduler's time slice, the number of cores, and the memory subsystem. There is no single "correct" loss rate — it is a property of the *machine*, not the *program*, which is the deep reason this bug is so treacherous. We will measure real numbers later.

The same hazard in the bank-account framing is worse than losing one count, because the two stores compute *different* values. Account starts at `\$100`. T1 deposits `\$50`; T2 withdraws `\$30`. Correct answer: `\$120`. Interleave the read-modify-writes badly:

#### Worked example: losing a deposit on a bank balance

Balance starts at `\$100`. T1 (deposit `\$50`) and T2 (withdraw `\$30`) each read, compute, write.

1. **T1 reads** balance into its register: `r1 = 100`.
2. **T2 reads** balance into its register: `r2 = 100` (stale — T1 is mid-flight).
3. **T1 computes** `r1 = 100 + 50 = 150`.
4. **T1 writes** balance `= 150`. Looks right so far.
5. **T2 computes** `r2 = 100 - 30 = 70` (from the stale `100`, ignoring the deposit).
6. **T2 writes** balance `= 70`, overwriting the `150`.

Final balance: `\$70`. The correct answer was `\$120`. The customer's `\$50` deposit vanished — not delayed, not held, *gone*, with no error, no log, no stack trace. If the interleaving had run the other way (T1's read after T2's write), the withdrawal would have vanished instead and the customer would be `\$30` richer. This is why financial systems do not let two transactions read-modify-write the same row without a lock or a database transaction with proper isolation, a topic the [data race versus race condition post](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction) sharpens into a precise distinction.

## Why it is nondeterministic: scheduling, timing, and "works on my machine"

Here is the property that turns a simple arithmetic bug into a career-defining nightmare: the lost update is **nondeterministic**. The same program, on the same machine, with the same input, produces a different answer on different runs. There is no deterministic reproduction. You cannot put a breakpoint where the bug "is," because the bug is not at a place in the code — it is in the *timing* between two places.

The nondeterminism comes from the scheduler, and the scheduler is influenced by everything: how many cores are free, what else the OS is running, where the cache lines happen to be, how warm the branch predictor is, whether the threads got preempted at a system-call boundary, whether the machine is under load, what the power-management governor decided about clock speed this millisecond. None of this is in your source code. All of it changes the interleaving. Two consecutive runs of the identical binary take different paths through the braid, so they lose different numbers of updates — or, occasionally, none at all.

This is the mechanism behind the most expensive sentence in software: **"works on my machine."** The phrase is not a cop-out; it is a literal, accurate report of a nondeterministic bug. On the developer's laptop, the test loop runs a thousand iterations, single-threaded most of the time because the machine is idle and the scheduler keeps both threads on the same core in long uninterrupted slices, so the load-store windows almost never overlap. The test passes. It passes a hundred times. It passes in CI, which is also lightly loaded. Then it ships to production, where the machine has 64 cores, is running at full load, the two threads land on genuinely different cores and run *simultaneously*, the iteration count is not a thousand but ten billion across a day, and the load-store windows overlap constantly. The bug that "could not be reproduced" fires thousands of times an hour. The conditions that hide the race in development are exactly the conditions that production removes.

This load-dependence and timing-dependence is why concurrency bugs are categorically different from sequential bugs. A null-pointer dereference fails the same way every time you hit it. A race fails differently every time, or not at all, and the act of *observing* it — attaching a debugger, adding a print statement, enabling logging — changes the timing and often makes it disappear. (Print statements involve I/O and locking inside the runtime, which serializes the threads and closes the window. The bug that vanishes when you add a `printf` is the signature of a race; old-timers call it a **Heisenbug**, because measuring it changes it.)

There is a quantitative reason the developer's machine hides the bug, and it is worth making precise because it explains the entire dev-versus-prod gap. The probability that a given increment loses its update is roughly the probability that the *other* thread's store lands inside this thread's load-store window. Call the window width $w$ (a few nanoseconds) and the average time between one thread's increments $T$. Crudely, the per-increment collision probability scales like $w / T$ when the threads run on separate cores. On a developer's idle laptop, the scheduler often parks both threads on the same core and runs each in a long uninterrupted slice, so for most of the run the threads are *not* overlapping at all — the effective collision probability drops toward zero, and a thousand-iteration test sees zero losses. In production, the two threads run on separate cores genuinely in parallel, $w/T$ is small but nonzero on *every* increment, and across ten billion increments a day the expected number of losses is enormous. Same code, same probability model, wildly different exposure — because the *number of trials* and the *degree of parallelism* are orders of magnitude larger in production. The bug was always there; production just rolled the dice ten billion more times, on a table where the odds were worse.

#### Worked example: why the same binary gives three different answers

Run the two-goroutine counter from the top of this post three times on a loaded multicore laptop:

- **Run A** — the scheduler kept both goroutines mostly on one core, switching only at GC safepoints. Windows rarely overlapped. Result: `1,998,433`. Only `1,567` updates lost.
- **Run B** — the OS scheduled the goroutines on two cores running in parallel; the windows overlapped on nearly every iteration where the cache line bounced between cores. Result: `1,061,209`. Nearly half the updates lost.
- **Run C** — light moment, the runtime happened to run one goroutine to completion before the other got going. Almost no overlap. Result: `1,999,998`. Two lost — practically correct.

Three runs, one binary, three answers spanning from "obviously broken" to "looks fine, ship it." Run C is the one that gets you fired, because it is the run your test happened to do.

The takeaway is brutal and worth stating as a rule: **a single passing run of a concurrent program proves nothing.** Correctness under concurrency is not something you can observe by running once; it is something you must *establish by construction* — by making the dangerous operations safe regardless of interleaving. Which brings us to the three properties.

## The three properties: atomicity, visibility, ordering

The lost update is one bug, but it is a symptom of a missing guarantee. To reason about shared mutable state in general, you need a vocabulary for *what can go wrong*, and there are exactly three things. Every concurrency bug over shared memory is a violation of **atomicity**, **visibility**, or **ordering**. Get all three and your shared state is safe; lose any one and you have a distinct, nasty failure. These three are the lens for the entire rest of the series — locks provide all three over a critical section; atomics provide them over a single operation; the memory model is the formal account of visibility and ordering.

![a matrix with rows atomicity visibility ordering and columns what it means the bug if missing and the fix](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-3.png)

### Atomicity: the operation is indivisible

**Atomicity** means an operation completes as an all-or-nothing unit, with no other thread able to observe or interleave a partial state. An atomic increment either has happened or has not — no other thread can ever catch it between its load and its store. The lost update is precisely a violation of atomicity: `count++` is three steps, and the gap between them is where another thread sneaks in. If the read-modify-write were atomic — indivisible — the second thread could not load until the first's store completed, and no update could be lost.

Atomicity is not only about read-modify-write. On many platforms even a plain read or write of a value wider than the machine word is not atomic. A 64-bit `double` or `long` on a 32-bit machine is written as two 32-bit stores; another thread can read after the first store and before the second and see a **torn value** — the high half of the new value glued to the low half of the old, a number that was never legitimately in the variable. The Java Language Specification historically allowed exactly this for non-`volatile` `long` and `double` fields; the JLS explicitly notes that reads and writes of those types are not guaranteed atomic unless declared `volatile`. A thread reading such a field concurrently with a write could observe a torn 64-bit value that neither writer ever stored — the bug if atomicity is missing, in its purest form.

### Visibility: one thread's write is seen by another

**Visibility** means that when one thread writes a value, another thread is guaranteed to *see* that write rather than a stale cached copy. This one surprises people, because it has nothing to do with interleaving and everything to do with the memory hierarchy. Each core has its own caches and, on the store side, a **store buffer** — a small queue where writes sit before they are flushed to the cache hierarchy where other cores can see them. A thread can write `flag = true`, and for some window of time, *no other core sees it*, because the write is sitting in the writing core's store buffer. Worse, the compiler may have decided to keep the value in a register and never write it to memory at all inside a loop, because nothing in *that thread's* sequential view told it the value matters to anyone else.

The classic visibility bug is the loop that never ends:

```java
// BROKEN: 'running' is not volatile, so the reader may never see the writer's update.
class Worker {
    private boolean running = true; // shared, mutated by another thread

    void runLoop() {
        while (running) {
            // do work
        }
        // we may NEVER get here, even after stop() is called
    }

    void stop() {
        running = false; // the writer thread sets it
    }
}
```

One thread spins in `runLoop` waiting for `running` to become false. Another thread calls `stop`. On a real JVM, with `running` not declared `volatile`, the spinning thread can loop *forever* even after `stop` returns — because the JIT compiler, seeing that `running` is not modified inside `runLoop`, is entitled to hoist the read out of the loop and check it once, or the write may never propagate out of the writer's store buffer to the reader's core. There is no atomicity problem here — a boolean write is atomic. There is no interleaving problem — it is one writer, one reader. The bug is pure visibility: the write happened, and the other thread never saw it. The fix is to declare `running` `volatile`, which on the JVM forces the write to be flushed and the read to go to memory, establishing the visibility guarantee:

```java
// FIXED: volatile establishes visibility — the writer's store is seen by the reader.
private volatile boolean running = true;
```

The same visibility bug exists in every language with a relaxed memory model. In Go, a plain `bool` flag read in a loop and written from another goroutine has no visibility guarantee, and `go vet -race` will flag it; the idiomatic fix is not a volatile keyword (Go has none) but an atomic or a channel:

```go
import "sync/atomic"

var running atomic.Bool // atomic gives both atomicity and visibility

func runLoop() {
	for running.Load() { // each Load is guaranteed to see the latest Store
		// do work
	}
}

func stop() { running.Store(false) } // visible to the reader
```

In C++ the same fix uses `std::atomic<bool>`, and crucially, a *plain* `bool` here is undefined behavior, not merely "might be slow to see" — the compiler may optimize the loop into `while (true)` outright. The lesson generalizes: visibility is never free, and the keyword or type that buys it differs by language (`volatile` in Java, `atomic.Bool` in Go, `std::atomic` in C++), but the underlying mechanism is the same — a memory barrier that flushes the writer's store buffer and forces the reader to fetch from coherent memory rather than a register or stale cache line.

### Ordering: operations appear in a sensible order

**Ordering** means that operations performed by one thread appear to another thread to happen in a sensible, agreed order. The trap is that *neither the compiler nor the hardware promises to preserve the order you wrote*, as long as the reordering is invisible to a single sequential thread. A compiler may reorder independent stores; a CPU with a weak memory model (ARM, POWER) may let another core observe a thread's stores in a different order than they were issued. Within one thread this is invisible and harmless. Across threads it breaks programs.

The classic ordering bug is broken publication:

```cpp
// BROKEN: the store to 'ready' may become visible before the store to 'data'.
int   data  = 0;
bool  ready = false;

void producer() {
    data  = 42;     // (1) write the payload
    ready = true;   // (2) announce it is ready
}

void consumer() {
    while (!ready) { /* spin */ }
    use(data);      // may read data == 0, not 42
}
```

The producer writes `data`, then sets `ready` to announce it. The consumer waits for `ready`, then reads `data`. You wrote (1) before (2), so surely if the consumer sees `ready == true` it must also see `data == 42`? Not guaranteed. The compiler or the hardware may make store (2) visible to the consumer *before* store (1), because to the producer thread in isolation the two stores are independent and reorderable. The consumer sees `ready == true`, reads `data`, and gets `0`. Nothing was lost, nothing was torn — the operations simply became visible in the wrong order. The fix is to establish a **happens-before** edge: a release on the producer's side paired with an acquire on the consumer's side (`std::atomic<bool>` with `memory_order_release` / `memory_order_acquire`), which forbids the reordering and guarantees that if the consumer sees the `ready` write, it also sees everything the producer did before it. This is the entire subject of the memory-model posts later in the series; here, just register that ordering is the third property and that "the order I wrote" and "the order another thread sees" are not the same thing.

Three properties, three distinct bugs: a lost update (atomicity), an unstoppable loop (visibility), a torn publication (ordering). A correct piece of shared mutable state has all three. The mechanisms in the rest of this series are graded by which of the three they give you, and at what cost.

It is worth seeing how the three relate, because they are not fully independent — they are three facets of one deeper requirement, the **happens-before** relation. Happens-before is a partial order over the operations in a concurrent program: if operation A happens-before operation B, then A's effects are guaranteed visible to B, and B is guaranteed to observe A as having already occurred. A lock release happens-before the next acquire of the same lock; a volatile/atomic write with release semantics happens-before a read with acquire semantics that observes it; a thread's `start` happens-before everything the started thread does; everything a thread does happens-before another thread's `join` on it. When you establish a happens-before edge between two accesses to shared state, you get all three properties at once over that edge: the operations are ordered (that is the relation itself), the earlier writes are visible to the later reads (the visibility guarantee rides on the order), and if the edge brackets a read-modify-write so that no other thread can interpose, you get atomicity too. So the practical recipe for safe shared state, the one the whole series builds toward, is: **for every pair of conflicting accesses to a shared mutable location — where at least one is a write — establish a happens-before edge between them.** A data race is *exactly* the absence of such an edge between two conflicting accesses; that is the formal definition, and it is what a race detector checks. The three properties are how the absence manifests as bugs; happens-before is the single thing whose presence prevents all three. Everything that follows — locks, atomics, channels, the memory model — is machinery for cheaply manufacturing happens-before edges where you need them.

## Read-modify-write and check-then-act: the two recurring shapes

If you stare at enough race conditions, you notice they almost all fold into one of two shapes. Naming them turns "spot the race" from an art into a checklist. Both are *ordering bugs* in the broad sense — a decision or update made on the basis of state that another thread changed in the gap.

![a taxonomy tree of race shapes with read modify write and check then act as the two branches under ordering bugs each with two example leaves](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-6.png)

**Read-modify-write (RMW)** is the shape we have been studying: read the current value, compute a new value *from it*, write the new value back. The update depends on the value you read, so if the value changes between your read and your write, your write is wrong. `count++` is RMW. `balance += amount` is RMW. `total = total + item.price` is RMW. `list.length++` after an append is RMW. Anytime the new value is a function of the old value and you compute it in your own thread before writing it back, you have an RMW, and unless it is made atomic or guarded by a lock, two threads can lose one another's updates. The fix is to make the read-compute-write indivisible — one atomic instruction (`fetch_add`, `lock inc`) for simple arithmetic, or a lock around the section for anything multi-step.

**Check-then-act (CTA)** is the subtler cousin: you check a condition, then act on the assumption that the condition is still true — but another thread invalidated it in the gap between the check and the act. The condition you checked is a *snapshot*, and the snapshot can go stale before you act on it. The canonical example is lazy initialization:

```java
// BROKEN: two threads can both see instance == null and both construct.
class Registry {
    private static Registry instance;

    static Registry get() {
        if (instance == null) {        // CHECK
            instance = new Registry(); // ACT
        }
        return instance;
    }
}
```

Two threads call `get` at the same time. Both execute the `if (instance == null)` check while `instance` is still null — the check is true for both. Both then proceed to construct a `new Registry`. Now you have built the object twice (wasteful, and catastrophic if the constructor opens a file or claims a port), and one of the two objects is leaked, and different callers may hold references to different "singletons." The check was true when each thread looked; it stopped being true before either acted; nobody noticed.

The other classic CTA is the get-then-put on a map. Here is the same shape in Go, which is instructive because Go's built-in `map` is *not* safe for concurrent writes and the race detector will catch it immediately:

```go
// BROKEN: two goroutines both find the key absent and both compute-and-store.
var cache = map[string]int{}

func getOrCompute(k string) int {
	if v, ok := cache[k]; ok { // CHECK: is it already there?
		return v
	}
	v := expensiveCompute(k)    // gap: another goroutine may store k here
	cache[k] = v                // ACT: store under the assumption it was absent
	return v
}
```

Two goroutines call `getOrCompute("x")` at once. Both read the map and find `"x"` absent (the check). Both run `expensiveCompute` — doing the expensive work *twice*, the first defect — and both write the result. If the two computations were not perfectly identical (say the value embeds a timestamp or a request ID), the second store silently overwrites the first, and two callers walk away with different "cached" values for the same key. And because Go maps are not concurrency-safe even for this, the concurrent writes can also corrupt the map's internal structure and crash with `fatal error: concurrent map writes` — a hard failure the race detector flags. The fix is to make the check and the act a single indivisible step. In Go that is a mutex around both, or `sync.Map`'s `LoadOrStore`, or `golang.org/x/sync/singleflight` to also collapse the duplicate computation:

```go
import "sync"

var (
	cache = map[string]int{}
	mu    sync.Mutex
)

func getOrCompute(k string) int {
	mu.Lock()
	defer mu.Unlock()
	if v, ok := cache[k]; ok { // check and act are now one critical section
		return v
	}
	v := expensiveCompute(k)
	cache[k] = v
	return v
}
```

In Java the equivalent fix is `computeIfAbsent`, a purpose-built atomic check-then-act baked into `ConcurrentHashMap`:

```java
// FIXED: computeIfAbsent performs the check and the act as one atomic operation.
ConcurrentHashMap<String, Integer> cache = new ConcurrentHashMap<>();

int getOrCompute(String k) {
    return cache.computeIfAbsent(k, key -> expensiveCompute(key));
}
```

The principle is identical across both languages: a check-then-act over shared state is a bug unless the check and the act are fused into one indivisible operation — a lock around both, or a primitive (`computeIfAbsent`, `LoadOrStore`, `compare-and-swap`) that checks and acts atomically. Guarding only the check, or only the act, fixes nothing, because the hazard lives precisely in the gap *between* them.

The reason naming the shapes matters: when you review concurrent code, you scan for these two patterns. Every `x = f(x)` over shared state is an RMW — guard it. Every `if (condition over shared state) { act on it }` is a CTA — guard the check *and* the act together, not just one. Ninety percent of the races you will ever write are one of these two, dressed in different clothes.

## The fixes, previewed: a lock and an atomic

The bug analysis is the point of this post, but you deserve to see the cure, if only as a preview of where the series goes. There are, broadly, four ways to make shared mutable state safe, and we will spend whole posts on each. Here are the two that fix the counter, with real code, because seeing the fix sharpens your understanding of the bug.

![before and after comparison of a racy unguarded counter losing updates versus a guarded counter using a lock or an atomic ending at the correct total](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-4.png)

**Fix one: a lock (mutual exclusion).** Wrap the read-modify-write in a critical section that only one thread can occupy at a time. While T1 holds the lock, T2 cannot enter — it blocks until T1 releases — so T2's load cannot land inside T1's load-store window. The window is closed by force. Here is the Go counter, fixed with a mutex:

```go
var (
	count int
	mu    sync.Mutex
)

func increment() {
	mu.Lock()         // acquire — only one goroutine past this point
	count++           // the RMW is now indivisible w.r.t. other holders
	mu.Unlock()       // release — next waiter may proceed
}
```

The same fix in Rust, where the type system *forces* you to acquire the lock to even touch the data — the `Mutex<T>` owns the value, and the only way to reach it is through a guard that proves you hold the lock:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let count = Arc::new(Mutex::new(0i64)); // the data lives inside the lock
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let count = Arc::clone(&count);
            thread::spawn(move || {
                for _ in 0..1_000_000 {
                    let mut guard = count.lock().unwrap(); // acquire
                    *guard += 1;                           // indivisible while held
                    // guard drops here -> release
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    println!("count = {}", *count.lock().unwrap()); // exactly 2000000, every run
}
```

This is Rust's "fearless concurrency" in miniature: there is no way to write the *unsafe* version by accident, because shared mutable access without a lock does not type-check. The mechanics of mutexes — how they are built from atomic instructions, how they block, their cost, their bugs (deadlock, contention) — are the subject of the [mutual exclusion post](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections).

**Fix two: an atomic (hardware read-modify-write).** For a single counter you do not need a lock at all. The hardware provides instructions that perform a read-modify-write *as one indivisible operation*, so there is no gap for another thread to slip into. The whole load-modify-store becomes one atomic `fetch-and-add`, which on x86 is a single `lock xadd` instruction. Go's `sync/atomic`:

```go
import "sync/atomic"

var count int64

func increment() {
	atomic.AddInt64(&count, 1) // one atomic RMW — no lock, no lost update
}
```

The same in C++ with `std::atomic`, naming the memory ordering explicitly:

```cpp
#include <atomic>
std::atomic<long> count{0};

void increment() {
    count.fetch_add(1, std::memory_order_relaxed); // atomic RMW; relaxed is fine for a pure counter
}
```

And in Java with `AtomicLong`:

```java
import java.util.concurrent.atomic.AtomicLong;
AtomicLong count = new AtomicLong(0);

void increment() {
    count.incrementAndGet(); // atomic RMW implemented internally via a CAS loop
}
```

All three produce exactly two million, every run, with no lock and far less overhead than a mutex. The atomic *is* the indivisible read-modify-write the bug needed; we get atomicity directly from the hardware. (Note the C++ version uses `memory_order_relaxed`: a pure counter needs atomicity but not ordering relative to other variables, so it can use the cheapest ordering — a distinction that is the whole subject of the atomics and memory-model posts.) The full story of atomics, compare-and-swap, the ABA problem, and lock-free data structures is a later track; here, the point is that the fix is *precisely targeted at the missing property* — atomicity — and once you can name the missing property, choosing the fix is mechanical.

![a matrix of four fixes mutex atomic add immutability and confinement with columns for how it works its cost and when to use it](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-7.png)

The two fixes above are the front two of four. The other two are structural: **immutability** (never mutate — replace the whole value, or copy-on-write, so there is no write to race) and **confinement** (give the state a single owner — one thread, or one actor — so it is never shared at all, and updates arrive as messages that owner serializes). The matrix above lays out all four against how they work, what they cost, and when to reach for each; the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) turns this into a full decision framework. The deep lesson is that *removing any leg of the hazard fixes it*: a lock or atomic restores the missing guarantee over the shared mutable state; immutability removes the *mutable*; confinement removes the *shared*. There are only so many moves, and they all come from the structure of the hazard.

## Reproducing the race reliably

A race you cannot reproduce is a race you cannot fix or verify fixed. Since a single run lies, you need a methodology to *force* the bug to surface, and then to *prove* it is gone. There are three tools, in order of power.

![before and after showing a thin test with low iterations hiding the race versus a wide test with high iterations and many runs surfacing it](/imgs/blogs/shared-mutable-state-and-the-anatomy-of-a-race-condition-8.png)

**Widen the window.** The race fires when one thread's store lands inside another's load-store window. You make that more likely by (1) raising the iteration count enormously — millions, not thousands, so the windows overlap by sheer volume — (2) raising the thread count so more loads and stores compete, (3) running on a genuinely multicore machine so threads run in parallel, not just interleaved, and (4) running the whole program *many times* and looking at the *distribution* of results, not one result. A racy program run a hundred times will show a spread of wrong answers; a correct program run a hundred times shows the same right answer a hundred times. Here is a harness that runs the racy counter many times and reports the spread:

```bash
for i in $(seq 1 100); do ./racy_counter; done | sort -n | uniq -c
```

That one-liner runs the racy binary a hundred times and tabulates how often each result appears.

If you see a single line — `100 2000000` — the program might be correct, or the window might just be too narrow on this machine to ever fire; absence of failure is not proof of correctness. If you see a spread — `3 1974815`, `1 1981203`, `7 1998442`, ... — you have caught the race red-handed.

**Insert deliberate delays.** To make a narrow window gape open, sleep or yield *inside* the read-modify-write, between the read and the write. This is a diagnostic trick, not production code, but it converts a one-in-a-million race into a one-in-one race:

```go
// DIAGNOSTIC ONLY: a deliberate yield between read and write to force the race.
func incrementSlow() {
	old := count           // read
	runtime.Gosched()      // yield — invite the scheduler to run the other thread NOW
	count = old + 1        // write — almost certainly stale by now
}
```

With the yield, the load-store window is enormous, the other thread almost always slips in, and the lost update happens nearly every iteration. If you can reproduce a race this way, you have *proven* the read-modify-write is unguarded.

**Use a race detector — the real answer.** The professional tool is a **dynamic data-race detector**, which instruments memory accesses at runtime and reports any pair of accesses to the same location from different threads where at least one is a write and there is no happens-before edge between them — *whether or not the race actually manifested on that run*. This is the crucial advantage: the detector does not need the bug to fire; it detects the *absence of synchronization*, so it catches the race even on a run that happened to produce the right answer. Go has it built in behind the `-race` flag:

```bash
go run -race ./counter.go
```

The detector prints a report like this — note that it names both racing accesses, the file, and the line:

```bash
WARNING: DATA RACE
Write at 0x00c0000b4010 by goroutine 7:
  main.main.func1()
      /path/counter.go:18 +0x44
Previous write at 0x00c0000b4010 by goroutine 6:
  main.main.func1()
      /path/counter.go:18 +0x44
Goroutine 7 (running) created at:
  main.main()
      /path/counter.go:14 +0x9c
```

C, C++, and Rust use **ThreadSanitizer** (TSan), the same underlying technology, via a compiler flag:

```bash
clang -fsanitize=thread -g counter.c -o counter && ./counter   # C or C++ with Clang or GCC
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly run \
  --target x86_64-unknown-linux-gnu                            # Rust on nightly
```

Java has tools too (the `ThreadSafe` static analyzer, `jcstress` for memory-model stress testing). The rule for the whole series: **run your concurrent tests under a race detector in CI.** A race detector turns a nondeterministic, can't-reproduce, ships-on-Friday bug into a deterministic, on-every-run, fails-the-build error. It is the single highest-leverage tool in concurrent programming, and the entire industry underuses it. Note the cost — TSan-instrumented code runs roughly 5×–15× slower and uses several times the memory (these are order-of-magnitude figures from the ThreadSanitizer documentation; the exact slowdown depends on access density), so you run it in CI and stress tests, not in production. But you *do* run it.

A fourth tool, worth knowing exists even if you reach for it rarely, is **systematic schedule exploration** or **deterministic replay**. Tools in this family — `rr` for record-and-replay debugging on Linux, or model-checkers and controlled schedulers like Loom in Rust's ecosystem or `jcstress` in the JVM — do not rely on luck to surface the race. They either record the exact interleaving of a failing run so you can replay it deterministically under a debugger (turning a Heisenbug into a Bohrbug you can step through), or they enumerate many possible interleavings of a small test and check an invariant on each, so a race that only manifests on a one-in-a-billion schedule is found by *construction* rather than by chance. These are heavier than a dynamic race detector and usually applied to a small, isolated concurrent component rather than a whole service, but they are the strongest guarantee available: a passing schedule-exploration run says "no interleaving of this test violates the invariant," which is the thing a single execution can never tell you. The progression of tools, from weakest to strongest: run-it-and-pray (proves nothing), widen-the-window stress (catches gross races), dynamic race detector (catches the missing happens-before edge even when the race did not fire), schedule exploration (proves the absence of the race over all explored interleavings). Reach down the list as the cost of a race goes up.

## Measured: failure rate versus iterations and threads

Now the honest measurement. I want to show you real-shaped numbers for how the failure scales, with the loud caveat that **these are illustrative and machine-dependent** — your numbers will differ, the point is the *shape* of the curves, not the digits. I am giving defensible orders of magnitude, not fabricated precision; the methodology (run many times, report the distribution) matters more than any single value.

#### Worked example: how the loss scales with iterations

Take the two-thread Go counter, vary the per-thread iteration count $N$ (so the correct answer is $2N$), run each configuration 50 times on a loaded multicore laptop, and record the *median* lost-update count and how often the run was wrong at all. Representative shape:

| Per-thread iters $N$ | Correct total | Median lost (approx) | Runs wrong (of 50) |
| --- | --- | --- | --- |
| 1,000 | 2,000 | 0 | 4 / 50 |
| 100,000 | 200,000 | ~120 | 38 / 50 |
| 1,000,000 | 2,000,000 | ~9,000 | 50 / 50 |
| 10,000,000 | 20,000,000 | ~140,000 | 50 / 50 |

Two lessons jump out. First, at $N = 1{,}000$ the bug *almost never appears* — 46 of 50 runs were exactly correct. This is the trap: a thin test passes and you conclude the code is fine. Second, the absolute number lost grows faster than linearly with $N$ in this regime, because more iterations means the threads spend more time genuinely overlapping on two cores, widening the effective window. By ten million iterations, every single run is wrong, and the loss is in the hundreds of thousands. The bug did not change. The exposure did.

#### Worked example: how the loss scales with thread count

Fix the *total* work at 8,000,000 increments and split it across more threads (so 2 threads do 4M each, 8 threads do 1M each, and so on). Correct answer is always 8,000,000. Representative shape:

| Threads | Per-thread iters | Median final (approx) | Median lost |
| --- | --- | --- | --- |
| 1 | 8,000,000 | 8,000,000 | 0 |
| 2 | 4,000,000 | ~7,950,000 | ~50,000 |
| 4 | 2,000,000 | ~7,400,000 | ~600,000 |
| 8 | 1,000,000 | ~6,100,000 | ~1,900,000 |
| 16 | 500,000 | ~4,800,000 | ~3,200,000 |

With one thread there is no race at all — nothing to interleave with — and the answer is always exact. As you add threads, the contention rises and the loss balloons; with 16 threads on a machine with fewer than 16 cores, the cache line holding `count` ping-pongs furiously between cores and the loss is a *large fraction* of the total. The shape to remember: **the bug gets worse with more concurrency, which is exactly the direction your production system is moving.** More cores, more threads, more load — every trend that makes your system faster also makes an unguarded race fire harder.

For contrast, the same harness with the atomic fix produces `8,000,000` exactly, on every run, at every thread count — and as a bonus, costs less per operation than the mutex version under contention (the atomic is one instruction; the mutex is an atomic *plus* the bookkeeping of blocking and waking). The measured throughput trade-offs between unguarded, mutex, and atomic are the subject of later posts; here the only measurement that matters is correctness, and the gap is stark: a spread of wrong answers versus the same right answer every time.

The honest-measurement discipline, stated as rules: warm up before timing (the first run pays JIT and cache costs); run many times and report the *distribution*, never a single number; name the platform, because x86's strong memory model (TSO) and ARM's weak model produce *different* failure rates for the *same* code; and never report a precise figure you did not measure — say "order of magnitude" and mean it. A concurrency benchmark that reports one number from one run is worse than no benchmark, because it lends false confidence to a nondeterministic result.

## Case studies / real-world

Races are not academic. They have crashed spacecraft, killed patients, and corrupted real money. A few documented cases, cited, with numbers marked approximate where I am not certain.

**The Therac-25 (1985–1987).** The Therac-25 was a radiation-therapy machine whose software, in certain operator-input sequences entered quickly, contained a race condition between the data-entry task and the treatment-setup task. A particular timing — an experienced operator editing the prescription faster than the software's tasks expected — let the machine deliver a massive radiation overdose while the display showed a normal dose. The race was in shared state (the treatment parameters) updated by concurrent tasks without proper synchronization, and because it depended on operator *timing*, it was nondeterministic and nearly impossible to reproduce — the manufacturer initially could not replicate the failures. At least six patients received overdoses; several died. The Therac-25 is the canonical "a race condition can kill someone" case, documented in Nancy Leveson's investigation, and the textbook example of why "we couldn't reproduce it" is not an acceptable answer for safety-critical concurrent code. (Details per Leveson and Turner, *An Investigation of the Therac-25 Accidents*, IEEE Computer, 1993.)

**The 2003 Northeast blackout race condition.** The August 2003 blackout that left roughly 50 million people in the US and Canada without power was triggered in part by a race condition in the alarm-processing software of the regional grid operator's control system. A race in the energy-management system's shared alarm-event state caused the alarm subsystem to stall silently; operators stopped receiving alarms and did not know the grid was failing until it cascaded. The race was a multi-thread contention over shared state that, under a rare timing, deadlocked or stalled the alarm queue. (Reported in the official US-Canada Power System Outage Task Force final report, 2004, and widely analyzed in the software-engineering literature.) The lesson is the same: a latent race that "never happened in testing" fired once, under production timing, with catastrophic consequences.

**Lost updates and double-spends in databases and ledgers.** The bank-balance worked example is not hypothetical. The "lost update" anomaly is a named, classified concurrency phenomenon in database isolation theory precisely because read-modify-write transactions that read a value and write a value computed from it can clobber one another under weak isolation levels — exactly the six-step trace from earlier, at the row level instead of the memory-word level. Real systems have lost real money to it when application code did `balance = read(); balance += amount; write(balance)` outside a transaction or under `READ COMMITTED` isolation, where two concurrent transactions can both read the old balance. The fix at the database layer is the same in spirit as in memory: serialize the read-modify-write (a `SELECT ... FOR UPDATE` row lock, an atomic `UPDATE ... SET balance = balance + ?`, or `SERIALIZABLE` isolation). The same hazard appears in cryptocurrency "double-spend" attacks and in any naive ledger that checks-then-acts on a balance. The pattern is universal because the hazard is universal: shared mutable state plus concurrent read-modify-write.

**A modern double-counting incident: the racy view counter.** A more everyday and recoverable case, the kind you will actually meet: many web services keep an in-memory counter of "active sessions" or "items in cart" or "views," incremented from multiple request-handler threads. Done as a bare `count++` it is the exact bug from the top of this post, and at low traffic it is invisible — the windows almost never overlap with a handful of requests per second. The day the service goes viral, traffic jumps a hundredfold, the request handlers run in parallel across every core, and the counter starts drifting low by a few percent. The metric that drives an autoscaling decision, or a billing line, or a capacity dashboard, is now quietly wrong, and the error grows with load — precisely when you most need the number to be right. No alarm fires, because the number still looks plausible. This is the benign-looking race that is not benign: it was a real lost-update bug all along, hidden by low exposure, revealed by scale. The fix is one line — an `atomic.AddInt64` or `AtomicLong.incrementAndGet` — and the lesson is that "it's just a counter, it doesn't matter if it's off by a little" is a decision you should make *deliberately*, not have made *for you* by an unguarded `++`.

The thread through all three: a race that is rare in testing is not rare in *aggregate* across a real system's lifetime of executions, and when it fires the failure can be silent, catastrophic, and unreproducible. That is why the discipline of *naming the shared mutable state and establishing the three properties over it* is not pedantry — it is the only thing standing between you and the next entry on this list.

## When to reach for this (and when not to)

Not every concurrent access to shared data is a bug you must fix, and a junior engineer who locks *everything* produces code that is both slow and still buggy (over-locking causes deadlock, the subject of a later post). The decision is: **is this race benign, or must it be fixed?** Here is the honest framework.

**A race is a real bug you must fix when** the shared mutable state has an *invariant* that the interleaving can violate — a counter that must be exact, a balance that must conserve money, a data structure whose internal pointers must stay consistent, a check-then-act where the act assumes the check still holds. If a lost update produces a wrong answer, a corrupted structure, a torn value, or a violated business rule, it must be fixed — with a lock, an atomic, immutability, or confinement, chosen by which property is missing and how expensive each fix is. Most races are in this category. When in doubt, it is a bug.

**A race is benign (and fixing it may cost more than it is worth) when** the program is *correct regardless of the interleaving* — when there is genuinely no invariant to violate. Real, narrow examples: a statistics counter you only use for approximate monitoring, where losing a few percent of increments does not change any decision (some metrics systems deliberately use relaxed, racy counters for speed); a "have we initialized yet" flag where double-initialization is genuinely harmless and idempotent; a cache where a racy double-compute wastes a little work but always produces the same value, so a torn-but-consistent read is fine. The crucial caveat, and it is large: **a "benign race" is still undefined behavior in C and C++**, where a data race on a non-atomic object means the compiler may legally do *anything* — it is not "you get one value or the other," it is "your program is invalid." So even a logically benign race must still be expressed with `std::atomic` (even `memory_order_relaxed`) to be *legal*, not just with a bare variable. In Java a benign race on a `volatile` or properly-published field is well-defined; a bare-field race risks torn `long`/`double` and stale reads. The rule: a race may be *logically* benign, but at the language level it is almost never *free* — you usually still pay for an atomic to make it defined. "Benign" buys you a cheaper fix (relaxed atomic over a full lock), not *no* fix.

**Do not reach for the heavy machinery prematurely.** If the state is confined to one thread, it is not shared — do not lock it. If the value is immutable, it cannot race — do not guard it. If a single atomic counter suffices, do not wrap it in a mutex (the atomic is cheaper and cannot deadlock). And conversely, if the invariant spans *multiple* variables that must change together, a per-variable atomic is *not* enough — you need a lock (or a transaction) over the whole group, because atomicity of each piece does not give you atomicity of the composite. The art is matching the cheapest sufficient mechanism to the actual invariant, which is the through-line of this entire series and the explicit subject of the capstone.

## Key takeaways

1. **`count++` is not one operation — it is load, modify, store.** The gap between the load and the store is where another thread slips in and an update is lost. Atomicity is something the hardware must be told to provide; you never get it free from "it's one line of code."
2. **The lost update is an overwrite, not a corruption.** Both threads execute correct instructions; the second thread's store, computed from a stale read, silently clobbers the first thread's store. Trace the six steps until it is obvious.
3. **Races are nondeterministic and load-dependent.** The same binary gives different answers on different runs, because the OS scheduler — influenced by everything outside your source code — chooses a different interleaving each time. "Works on my machine" is a literal, accurate report of a timing-dependent bug.
4. **A single passing run proves nothing.** Correctness under concurrency must be established by construction, not observed by running once. Thin tests on idle machines hide the very race that production exposes.
5. **There are exactly three properties to establish over shared state: atomicity, visibility, ordering.** A lost update is missing atomicity; an unstoppable loop is missing visibility; a torn publication is missing ordering. Name the missing property and the fix becomes mechanical.
6. **Almost every race is a read-modify-write or a check-then-act.** Guard the whole read-compute-write as one unit; guard the check *and* the act together, never just one.
7. **There are four fixes, each removing a leg of the hazard:** a lock and an atomic restore the missing guarantee; immutability removes the *mutable*; confinement removes the *shared*. Match the cheapest sufficient one to the actual invariant.
8. **Run a race detector in CI.** Go `-race`, ThreadSanitizer for C/C++/Rust, `jcstress` for the JVM. A detector catches the unsynchronized access *even on a run that happened to produce the right answer* — it turns a nondeterministic bug into a deterministic build failure.
9. **Measure honestly: many runs, the distribution, name the platform.** x86's strong memory model and ARM's weak model give different failure rates for the same code. Never report a precise figure you did not measure.
10. **A logically benign race is still rarely free.** In C and C++ a data race is undefined behavior regardless of intent, so even a "harmless" race must be a `std::atomic` to be legal. "Benign" buys a cheaper fix, not no fix.

## Further reading

- **Herlihy & Shavit, _The Art of Multiprocessor Programming_** — the definitive treatment of atomicity, linearizability, and the correctness conditions for concurrent objects; chapters 1–3 formalize everything this post sketched.
- **Brian Goetz et al., _Java Concurrency in Practice_** — the clearest practical account of atomicity, visibility, and the Java Memory Model; the `volatile` and check-then-act discussions are the canonical references for the visibility and CTA bugs here.
- **Anthony Williams, _C++ Concurrency in Action_** — the practitioner's guide to `std::atomic`, memory orderings, and why a bare-variable race is undefined behavior in C++.
- **Leveson & Turner, "An Investigation of the Therac-25 Accidents," _IEEE Computer_ (1993)** — the definitive case study of a fatal race condition in safety-critical software.
- **US-Canada Power System Outage Task Force, _Final Report on the August 14, 2003 Blackout_ (2004)** — the official analysis of the alarm-system race that contributed to the cascade.
- **Jeff Preshing's blog (preshing.com)** — exceptional, intuition-first posts on memory ordering, acquire/release, and lock-free programming that pick up exactly where this post's ordering section ends.
- **The Go blog, "Introducing the Go Race Detector"** — how dynamic data-race detection works and why you should run it in CI.
- **Within this series:** the [intro on why concurrency is unavoidable](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), the upcoming [mutual exclusion and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) post for the lock fix in depth, the [data race versus race condition](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction) post for the precise distinction, and the [concurrency playbook capstone](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) for the full decision framework. For the Python-specific angle on shared state and why the interpreter lock both helps and hurts, see [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs).

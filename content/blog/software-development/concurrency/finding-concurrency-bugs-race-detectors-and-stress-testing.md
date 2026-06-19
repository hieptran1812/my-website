---
title: "Finding Concurrency Bugs: Race Detectors and Stress Testing"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to stop hoping a race shows up and start forcing it into the open with happens-before detectors, scheduler fuzzing, exhaustive interleaving exploration, and model checking."
tags:
  [
    "concurrency",
    "parallelism",
    "race-detector",
    "thread-sanitizer",
    "stress-testing",
    "testing",
    "loom",
    "model-checking",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-1.png"
---

A team I worked with shipped a payment service that passed every test. Unit tests: green. Integration tests: green. The flaky-test bot re-ran the suite ten thousand times overnight, the way it always did, and reported nothing. Two weeks after deploy, a customer's wallet balance went negative by exactly one transaction's worth, then a second customer's, then a handful more, all under the heaviest traffic of the day. The bug was a textbook lost update: two goroutines read a balance, both decremented their local copy, both wrote it back, and one write clobbered the other. It had been there since the first commit. The tests had run that code path millions of times. It had simply never lost the race during a test run — because the window in which the interleaving goes wrong was a few nanoseconds wide, and the only way to widen it reliably was the exact production scheduling pressure the tests never reproduced.

This is the central, demoralizing fact about concurrency bugs: **you cannot find a race by hoping it shows up.** A sequential bug is deterministic — the same input takes the same path and fails the same way every time, so a single failing test pins it forever. A race depends on the *interleaving* of threads, which the operating-system scheduler chooses nondeterministically and which your test harness has almost no control over. Re-running the same test is not sampling new interleavings in any useful way; it is mostly re-rolling the same heavily-biased die. A green test suite over thousands of runs tells you almost nothing about whether a one-in-a-million interleaving is lurking. The figure below contrasts the two postures: passively re-running tests and watching the race hide, versus actively using a detector or a stress harness that forces the bad interleaving into the open on essentially every run.

![two postures for finding a race showing passive test reruns hiding the bug on the left and a detector or stress harness surfacing it on the right](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-1.png)

This post is about the real toolbox for catching these bugs — the one that replaces hoping with engineering. We will cover dynamic race detectors (ThreadSanitizer and Go's `-race`, how the happens-before and lockset algorithms actually work, what shadow memory is, why you pay a 5–10x slowdown, and crucially what they catch and what they cannot see); stress testing (injecting delays and yields to widen the window, fuzzing the scheduler, running millions of iterations and doing the probability math on how many runs you need); deterministic replay and schedule exploration (controlled schedulers like Rust's `loom`, Java's `jcstress`, and Go's `GODEBUG` knobs that enumerate interleavings instead of sampling them); model checking the design with TLA+ and litmus tests; and fault injection. The discipline that ties them together is simple to state and hard to practice: **make the bug reproducible, then shrink it.** A race you can reproduce on demand is a race you can fix and, more importantly, regression-guard so it never comes back. By the end you will know which tool to reach for given a symptom, how each one works underneath, and how to read the report when ThreadSanitizer finally catches your counter race red-handed.

If the vocabulary here — data race, race condition, happens-before, shared mutable state — is not yet second nature, read [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) and [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) first. This post assumes you know that a data race is two unsynchronized accesses to the same location with at least one write, and builds the tooling on top of that.

## Why concurrency bugs evade normal tests

Start with the mechanism, because it explains every tool that follows. A test exercises a *path* through your code. A concurrency bug lives not in a path but in an *interleaving* — the specific order in which two or more threads' instructions are merged onto the timeline. For a sequential program, the path is determined by the input, so one passing test means that path is correct forever. For a concurrent program, the same input can run under astronomically many interleavings, and the test exercises whichever one the scheduler happened to pick. You are not testing the program; you are testing one sample from a distribution over interleavings, and the distribution is brutally skewed.

Quantify it. Suppose the bug requires a precise interleaving: thread T2 must read a shared variable in the few-nanosecond window between thread T1's read and T1's write. On a machine that is not under scheduling pressure, both threads run their critical section without preemption almost every time, so the bad window is rarely open. Say the probability that any single execution hits the bad interleaving is $p = 10^{-6}$. The probability that $N$ independent runs *all* miss it is $(1-p)^N$. To have even a 63% chance of catching it at least once, you need roughly $N \approx 1/p = 10^6$ runs (this is the standard $1 - e^{-Np}$ approximation: $1 - (1-p)^N \approx 1 - e^{-Np}$, which crosses 0.63 at $Np = 1$). Ten thousand CI runs at $p=10^{-6}$ give you a catch probability of about $1 - e^{-0.01} \approx 1\%$. You will almost certainly see green, and green will be a lie. That is exactly what happened to the payment team.

Three properties conspire to make this worse than the raw math suggests:

1. **The scheduler is not a fair coin.** Real schedulers are deeply biased toward running a thread to completion or to its next blocking call, because context switches are expensive (a switch costs on the order of 1–5 µs of direct cost plus cache-pollution aftershocks). So the "natural" preemption points cluster around system calls and lock acquisitions, and a race whose window sits in the middle of a tight non-blocking sequence may have $p$ far below $10^{-6}$ — sometimes effectively zero on a quiescent test box. The bug only "wakes up" under the memory pressure, the cache thrashing, and the dozens of competing runnable threads of a loaded production host.
2. **Test environments are quiescent by design.** We run tests on idle machines with few cores busy, precisely the condition that minimizes preemption and therefore minimizes the chance of an adversarial interleaving. The environment that makes tests fast and reproducible is the environment that hides races.
3. **The bug may not even be a data race.** Some of the nastiest concurrency bugs are *race conditions* without any data race — every individual access is properly locked, but the *sequence* of locked operations has a check-then-act gap (a TOCTOU bug: time-of-check to time-of-use). A detector that only looks for unsynchronized memory access will sail right past these. We will be precise about this split, because choosing the wrong tool for the wrong bug class is the most common mistake. The distinction is the whole subject of [data races vs race conditions](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction); here we care which tools cover which.

There is a fourth conspirator worth naming: **the bug's probability is not even constant across machines.** A race whose window is a single un-fenced store may be effectively impossible to hit on x86 (whose Total Store Order memory model forbids the reordering that the bug needs) and routine on ARM or POWER (whose weak models permit it). So a test suite that is green on your x86 CI fleet can be a minefield on the ARM servers you deploy to — the bug's manifestation probability jumped from zero to nonzero purely because the hardware memory model changed. This is why "works on my machine" is not just an excuse; for memory-ordering bugs it is a literal, mechanical truth about which reorderings the silicon permits, and it means the *platform* is part of your test matrix, not a footnote.

So the strategy cannot be "run the tests more." It has to be one of three fundamentally different moves: **instrument** the program so a detector reports the bug from a *single* run regardless of whether the bug actually manifested as wrong output (dynamic race detection); **bias the scheduler adversarially** so the rare interleaving becomes common, then run enough iterations to hit it (stress testing and scheduler fuzzing); or **enumerate the interleavings** so you check all of them instead of sampling (schedule exploration and model checking). The next sections take each in turn.

#### Worked example: how many runs do you actually need

A team has a flaky integration test that fails about once every 2,000 runs in CI. They want to "make sure it's fixed" by re-running. How many green runs prove the fix? The empirical failure rate is $p \approx 1/2000 = 5\times10^{-4}$. After a fix, to be 95% confident the true rate dropped below, say, $10^{-4}$, you'd need $N$ such that $(1-10^{-4})^N \le 0.05$, i.e. $N \ge \ln(0.05)/\ln(1-10^{-4}) \approx 3.0/10^{-4} \approx 30{,}000$ green runs — and that only bounds the rate, it does not prove correctness. This is the trap: passing runs are weak evidence, and the number needed to make them strong evidence is enormous. A race detector that flags the *cause* from one run, or an exhaustive checker that proves *no* bad interleaving exists, is categorically more powerful than counting green runs. Reach for those instead.

## Dynamic race detectors: happens-before and lockset

A dynamic race detector instruments every memory access and every synchronization operation at runtime and decides, *as the program runs*, whether any pair of conflicting accesses lacks an ordering between them. The brilliance of the approach is that it does **not** require the bug to actually corrupt anything on this run. If T1 writes `x` and T2 reads `x` and there is no happens-before edge connecting them, the detector reports a data race *even if the particular timing this run produced the correct answer*. It catches the latent bug, not just the manifest symptom. That is what lets a single run replace a million.

There are two classic algorithms underneath, and modern detectors blend them. Both are described in the matrix figure later in this section; first the mechanism.

### The happens-before algorithm (vector clocks)

The **happens-before relation**, written $a \to b$, is a partial order over the events in a concurrent execution. It is defined by two rules: (1) within a single thread, events are ordered by program order; (2) a release operation on a synchronization object (unlocking a mutex, sending on a channel, signaling a condition variable, a `release`-ordered atomic store) happens-before the matching acquire operation (locking the same mutex, receiving from the channel, the `acquire`-ordered load that reads the value). Transitivity closes the relation. The key theorem: **two conflicting accesses (same address, at least one a write) constitute a data race if and only if neither happens-before the other** — that is, they are *concurrent* under $\to$.

To compute this online, a happens-before detector gives each thread a **vector clock**: an array `VC[t]` of logical timestamps, one entry per thread. The rules:

- Each thread increments its own entry on each relevant event.
- On a release of synchronization object `m`, the thread copies its current vector clock into a clock `L_m` stored with `m` (element-wise max into whatever was there).
- On an acquire of `m`, the thread takes the element-wise max of its own clock and `L_m`. This is exactly the moment the acquiring thread "learns" everything the releasing thread had done.
- For each memory location, the detector stores the vector-clock timestamp of the last write and (a summary of) the last reads.

When thread T accesses address `a`, the detector compares T's current clock against the stored access clock for `a`. If the stored conflicting access's clock is *not* dominated by T's clock (i.e. there is some thread whose stamp on the stored access exceeds T's knowledge of it), then no happens-before edge connects them — **data race**. Two vector clocks are comparable iff one dominates the other element-wise; concurrency is precisely the incomparable case.

The happens-before approach is *precise*: it produces **no false positives**. If it reports a race, two accesses really were unordered on this execution. Its weakness is the flip side: it only sees the ordering that *this run* produced. If on this run a lock happened to serialize the two accesses, the detector sees an edge and stays silent — even though a different schedule could have removed that edge and raced. So happens-before detection's coverage is bounded by the interleavings you actually execute. Run it on a path where the bad ordering never occurs and it finds nothing. This is the single most important limitation to internalize, and the reason detectors must be paired with stress.

### The lockset algorithm (Eraser)

The older **lockset** algorithm, introduced by the Eraser tool, takes a different bet. Instead of checking the *actual* ordering, it checks a *discipline*: the rule that "every shared variable should be protected by some consistent lock held on every access." For each shared variable `v`, the detector maintains a **candidate lock set** `C(v)` — initially the set of all locks. On each access to `v` by a thread holding lock set `H`, it refines: `C(v) := C(v) ∩ H`. If `C(v)` ever becomes empty, the variable was accessed under no common lock — a likely race, reported.

Lockset's strength is that it can flag a race **even on a run where the bad interleaving did not happen**: it does not need the threads to actually collide, only to violate the locking discipline. That makes it less sensitive to the exact schedule. Its weakness is **false positives**: plenty of correct code intentionally uses no lock — initialization before threads start, ownership handoff via a channel, read-only-after-publish data, benign races on statistics counters, custom synchronization the detector doesn't recognize. Eraser had to special-case all of these (initialization phases, read-shared states) and still produced noise. That false-positive problem is why production detectors lean primarily on happens-before and use lockset-style heuristics only as a secondary signal.

The figure below lays the two algorithms side by side with shadow memory, the storage layer both depend on.

![comparison of happens-before lockset and shadow memory detector internals showing each one core idea its strength and its weakness](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-6.png)

### Shadow memory and the cost

Both algorithms need somewhere to store per-location metadata: the last-access vector clock, the candidate lock set, the thread id and access type. That store is **shadow memory** — a parallel region of memory that maps each application word (or each byte, depending on granularity) to a few words of metadata. ThreadSanitizer, for instance, keeps multiple "shadow cells" per application word, each recording one recent access (thread id, a compressed clock, offset, and is-write bit), and on every load and store it inserts instrumentation that updates and checks these cells. The mapping is a direct address transform — a shift and an offset — so a shadow lookup is O(1), but the program now touches several times more memory than before.

This is where the famous **5–10x slowdown** comes from. Every memory access becomes a memory access plus a handful of shadow-memory reads, comparisons, and conditional updates; every lock/unlock/atomic becomes a vector-clock operation. Memory footprint inflates too — TSan can use several times the application's memory for shadow state (it reserves a large virtual mapping). The slowdown is real and it is the price of seeing every access. Crucially, it is a *constant-factor* cost, not an algorithmic blowup, which is why you can run a detector on a realistic workload (a load test, a fuzz run, a CI integration suite) rather than just a toy. The next figure traces, as a timeline, exactly how a happens-before detector decides to fire on a pair of unordered accesses.

![timeline of a happens-before detector recording two thread accesses to the same address finding no ordering edge and reporting a data race](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-3.png)

### Running a detector: the counter race, caught

Now the practical flow — the bug, and the detector catching it. Here is the canonical lost-update race in Go: two goroutines incrementing a shared counter with no synchronization.

```go
package main

import (
	"fmt"
	"sync"
)

func main() {
	var counter int
	var wg sync.WaitGroup
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counter++ // load, add, store — three steps, not atomic
		}()
	}
	wg.Wait()
	fmt.Println(counter) // often < 1000, nondeterministically
}
```

Run it normally and you will usually see a number a little under 1000, occasionally exactly 1000 if you got lucky, and it changes every run. The wrong output is itself nondeterministic — useless as a test signal. Now run it with the race detector:

```bash
go run -race counter.go
```

Go's detector (which is built on TSan internally) prints a report like this — note that it pinpoints *both* accesses, the goroutine that did each, and where each goroutine was spawned:

```bash
==================
WARNING: DATA RACE
Read at 0x00c0000b4010 by goroutine 8:
  main.main.func1()
      /tmp/counter.go:15 +0x2e

Previous write at 0x00c0000b4010 by goroutine 7:
  main.main.func1()
      /tmp/counter.go:15 +0x44

Goroutine 8 (running) created at:
  main.main()
      /tmp/counter.go:13 +0x...

Goroutine 7 (finished) created at:
  main.main()
      /tmp/counter.go:13 +0x...
==================
Found 1 data race(s)
exit status 66
```

That report is the entire game. It does not say "the output was wrong" — the output might have been right on this run. It says two accesses to address `0x...b4010` were not ordered by happens-before, one was a read and one a write, and here are the exact source lines and goroutine origins. The fix is to establish an ordering, either with a mutex or, since this is a simple counter, with an atomic:

```go
import "sync/atomic"

var counter atomic.Int64
// ...
go func() {
	defer wg.Done()
	counter.Add(1) // single atomic read-modify-write; happens-before respected
}()
// ...
fmt.Println(counter.Load()) // always 1000
```

Re-run with `-race`: silent, and the output is now deterministically 1000. The same bug in C or C++ uses ThreadSanitizer directly via a compiler flag:

```c
// race.c — compile with: clang -fsanitize=thread -g -O1 race.c -o race -lpthread
#include <pthread.h>
#include <stdio.h>

static long counter = 0;

static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 100000; i++)
        counter++;          // unsynchronized read-modify-write — data race
    return NULL;
}

int main(void) {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, worker, NULL);
    pthread_create(&t2, NULL, worker, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("%ld\n", counter); // expected 200000, observed less
    return 0;
}
```

```bash
clang -fsanitize=thread -g -O1 race.c -o race -lpthread
./race
# ==================
# WARNING: ThreadSanitizer: data race (pid=...)
#   Write of size 8 at 0x... by thread T2:
#     #0 worker race.c:9 ...
#   Previous write of size 8 at 0x... by thread T1:
#     #0 worker race.c:9 ...
# ==================
```

The fix in C is `_Atomic long counter;` with `atomic_fetch_add(&counter, 1)`, or a `pthread_mutex_t` around the increment, or in C++ a `std::atomic<long>` and `counter.fetch_add(1, std::memory_order_relaxed)`. The detector goes quiet because the atomic establishes the happens-before edge it was looking for. In Java the JVM lacks a built-in equivalent at the same granularity (you reach for `jcstress`, covered later, or third-party tools), and in Rust the type system *prevents* the un-`Sync` version from compiling at all — but if you reach for `unsafe` and create the race anyway, you can run the program under TSan via the nightly `-Zsanitizer=thread` flag and get the same report. Two languages, one mechanism, identical underlying detector.

### Reading the report like a detective

The report is not a stack trace to skim — it is a precise statement of a happens-before *violation*, and every field is load-bearing. Read it in this order. First, the **address**: both accesses name the same address (`0x...b4010`); if they don't, it's two different races stacked, fix them separately. Second, the **access types**: at least one must be a write (a read-read pair is never a race), and the pairing tells you the failure mode — write-write is a lost update, read-write is a torn or stale read. Third, the **two stacks**: these are the two unordered accesses, and the fix is to insert a synchronization edge *between* them — find the lowest common point in the call graph where a lock, channel, or atomic can serialize them. Fourth, the **goroutine/thread origins**: where each was spawned, which tells you the lifetimes involved and whether one is a background worker, a request handler, or a finalizer.

The most common misreading is treating the two stacks as "the bug is in one of these functions." The bug is almost never *in* either access — both accesses are individually fine. The bug is the *missing edge between them*, which lives at the synchronization design level: someone forgot the lock, used two different locks, published a pointer without a release fence, or shared a field they assumed was thread-confined. Train yourself to read the report as "these two correct-looking lines lack an ordering" rather than "one of these two lines is wrong," and the fix becomes obvious almost every time.

One more practical note: detectors report the *first* race they find and may stop, or may continue and flood you. Set `GORACE="halt_on_error=1"` (Go) or `TSAN_OPTIONS="halt_on_error=1"` (C/C++) to stop at the first race when you're fixing them one at a time, and unset it for a survey run that lists everything. And `TSAN_OPTIONS="history_size=7"` enlarges the per-thread access history so the "previous access" stack is more likely to be captured for races with a longer gap between the two accesses — invaluable when the default history has already been overwritten by the time the second access fires.

## What detectors catch and what they miss

This is the section that prevents the most wasted effort, so be precise. A dynamic race detector catches a **data race** — two conflicting memory accesses with no happens-before edge — and it catches it *on the interleavings it executes*. That is genuinely powerful: data races are undefined behavior in C, C++, and Rust, and a source of torn reads, lost updates, and compiler-miscompilation in every language. Catching them is the highest-value, lowest-effort win in concurrency testing, and you should turn the detector on in CI today.

But the coverage has two hard edges, and ignoring either one burns teams.

**Miss #1: it only sees executed interleavings.** Because the happens-before engine reasons about the *actual* ordering this run produced, a race on a code path you never ran, or under an interleaving you never hit, is invisible. If your test never schedules T2's read into T1's window — even though a different schedule could — the detector reports nothing. This is why a detector is necessary but not sufficient: you must drive it across *many* interleavings, which means combining it with stress testing or schedule exploration. A detector under a single quiescent run is barely better than no detector for a rare race. Under a fuzzed, high-contention stress harness it becomes devastatingly effective, because now the bad interleavings *do* occur and the detector flags them from the first occurrence.

**Miss #2: it does not catch race conditions that aren't data races.** Consider a perfectly locked check-then-act:

```go
// Each operation is individually locked. There is NO data race.
// But the check-then-act across two locked sections is a race CONDITION.
func (b *Bank) Transfer(amount int) error {
	if b.Balance() >= amount { // locked read — atomic by itself
		// ... another thread can withdraw here ...
		b.Withdraw(amount)     // locked write — atomic by itself
		return nil
	}
	return errors.New("insufficient funds")
}
```

`Balance()` and `Withdraw()` each take the lock, so there is no unsynchronized access — `-race` is silent. Yet two threads can both pass the `>=` check on the same balance and both withdraw, overdrawing the account. The bug is a **race condition** in the higher-level logic: the invariant "never overdraw" spans two atomic operations with a gap between them. No memory-access detector will ever find this; it is not a memory-access bug. You find it by stress testing the *behavior* (assert the invariant after a flood of concurrent transfers and watch it break), by exhaustive exploration (jcstress/loom enumerate the interleaving where both checks pass), or by model checking the *design* (TLA+ proves the invariant is violable). The fix is to widen the atomic boundary — hold one lock across check-and-act, or use an atomic compare-and-swap on the whole operation.

There is also a quieter miss: detectors generally don't model **memory-order** subtleties beyond what the runtime exposes. A bug that only manifests on ARM's weak memory model because of a missing barrier may not reproduce on your x86 test box at all, and the detector reasons about the ordering it *observes*, not every ordering the hardware *permits*. For those you want litmus tests (later). The matrix below summarizes the whole landscape — what each technique catches, misses, and costs — so you can pick deliberately.

It is also worth being honest that detectors have a third, softer miss: **anything not executed.** A race detector instruments the code that runs; a code path your test never enters is invisible to it, exactly as it is to a coverage tool. So a detector's effective coverage is the intersection of "code paths executed" and "interleavings executed" — two limits multiplied together. The remedy is the same as for any coverage gap: drive more paths (more test scenarios, fuzzed inputs) *and* more interleavings (stress, fuzzing), both under the detector. A detector run over a fuzzer's corpus, under high contention, is far more than the sum of its parts: the fuzzer drives diverse paths, the contention drives diverse interleavings, and the detector flags the data race in any path-interleaving pair that races, from its first occurrence. This combination — coverage-guided fuzzing under a sanitizer — is the state of the art for finding deep bugs in systems code and is exactly how projects like Chromium and the Linux kernel run their detectors continuously.

![matrix comparing race detector stress test schedule exploration and model checking on what each catches misses and costs](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-2.png)

#### Worked example: the detector that stayed silent

A service used a `sync.Map` (Go's concurrent map) for a cache and a plain `int` field for a hit counter, incremented on every cache hit without synchronization. Code review missed it because the map was obviously concurrent-safe; the eye skipped the adjacent `c.hits++`. CI ran `-race` on every PR — and stayed green for months, because the unit tests used a single goroutine to exercise the cache, so the counter increment never ran concurrently with itself. The race was real but the *executed interleavings* never included two concurrent increments. It surfaced only when someone added a load test that hit the cache from 50 goroutines, *also* under `-race`: the detector fired on the first concurrent increment. Lesson: the detector is only as good as the concurrency you drive through it. Run it under load, not just unit tests.

## Stress testing: widening the window and fuzzing the scheduler

If a detector's reach is bounded by the interleavings you execute, the obvious move is to make the dangerous interleavings *common*. Stress testing does exactly that, with two complementary tactics: **widen the race window** so a normally-tiny gap becomes huge, and **fuzz the scheduler** so different runs explore genuinely different orderings. Then run a lot of iterations.

The intuition for window-widening is concrete. Recall the lost-update race: T2 must read between T1's read and T1's write, a gap of a few nanoseconds. If you *deliberately* insert a yield or a short sleep into that gap, you stretch it from nanoseconds to milliseconds — a factor of a million — and a thread that would almost never land there now lands there nearly every time. The before/after figure makes the asymmetry vivid.

![two timing diagrams contrasting a nanosecond race window that almost never fires with an injected delay that widens it into a reliable repro](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-5.png)

Here is the technique applied to reproduce the counter race deterministically in C, by injecting a `sched_yield()` (or a tiny `nanosleep`) into the load-modify-store so the preemption window is wide open:

```c
// repro.c — deliberately widen the window to make the race near-certain.
#include <pthread.h>
#include <sched.h>
#include <stdio.h>

static long counter = 0;

static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 1000; i++) {
        long tmp = counter;   // 1. load
        sched_yield();        // 2. hand the CPU away — window is now huge
        counter = tmp + 1;    // 3. store the stale value back
    }
    return NULL;
}

int main(void) {
    pthread_t t[8];
    for (int i = 0; i < 8; i++) pthread_create(&t[i], NULL, worker, NULL);
    for (int i = 0; i < 8; i++) pthread_join(t[i], NULL);
    printf("%ld (expected 8000)\n", counter); // reliably far below 8000
    return 0;
}
```

Without the `sched_yield()`, eight threads doing 1000 increments each will *usually* total 8000 on a fast machine because each increment finishes before preemption. *With* it, you will see numbers like 1200 — most updates are lost — on essentially every run. The yield did not create the bug; it exposed a bug that was always there. This is the heart of stress testing: you are not changing the program's correctness, you are changing the *probability* that its existing defect manifests, from near-zero to near-one.

The second tactic, **scheduler fuzzing**, randomizes *where* the delays go so you don't just probe one window. The pattern is a stress loop that, between operations, inserts a randomized yield/sleep/spin, and runs the whole scenario thousands or millions of times. In Go:

```go
// stress_test.go — run with: go test -race -run TestTransfer -count=1
package bank

import (
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestTransferStress(t *testing.T) {
	for iter := 0; iter < 200000; iter++ { // many iterations
		acct := &Account{balance: 100}
		var wg sync.WaitGroup
		for g := 0; g < 4; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				jitter()              // randomized scheduling perturbation
				acct.Withdraw(40)     // four threads racing to overdraw
			}()
		}
		wg.Wait()
		if acct.balance < 0 {        // invariant the detector can't check
			t.Fatalf("overdrawn: balance=%d at iter %d", acct.balance, iter)
		}
	}
}

func jitter() {
	switch rand.Intn(3) {
	case 0:
		runtime.Gosched()                 // yield to the scheduler
	case 1:
		time.Sleep(time.Microsecond)      // brief sleep widens the window
	case 2:                               // case 2: no perturbation (baseline)
	}
}
```

This catches the *race condition* from the earlier section — the one `-race` is blind to — because it asserts the *invariant* (`balance >= 0`) after each concurrent flood, and the randomized jitter makes the bad interleaving (two threads both passing the check) common. Run it under `-race` as well and you get both: the detector flags any data race, the assertion flags any invariant violation. Belt and suspenders.

A subtle but important refinement: you can **inject the scheduler perturbation systematically** rather than randomly. ThreadSanitizer ships exactly this — set `TSAN_OPTIONS="atexit_sleep_ms=0 flush_memory_ms=... "` and, more usefully, the historical `TSAN_OPTIONS=... `-style schedule perturbation, plus there are dedicated tools (the `pctest`/PCT algorithm, "Probabilistic Concurrency Testing") that insert a small number of carefully-placed priority-change points to *provably* hit any bug of "depth d" with probability at least $1/(n k^{d-1})$ for $n$ threads, $k$ steps, depth $d$ (number of ordering constraints the bug needs). PCT is the rigorous version of "fuzz the scheduler": instead of uniform random jitter, it picks interleavings from a distribution with a *guaranteed lower bound* on catching shallow bugs, which most real concurrency bugs are (depth 1–3). You don't need to implement PCT, but knowing it exists tells you that scheduler fuzzing can be principled, not just flailing.

How many iterations? Back to the probability math. If your stress harness raises the per-iteration manifestation probability to $p = 0.1$ (one in ten — realistic once the window is widened), then $N = 100$ iterations give catch probability $1 - 0.9^{100} \approx 1 - 2.7\times10^{-5}$, effectively certain. If the harness only reaches $p = 10^{-3}$, you need $N \approx 3000$ for a 95% catch ($1 - (1-10^{-3})^{3000} \approx 0.95$). The whole point of window-widening and fuzzing is to push $p$ up by orders of magnitude so the iteration count you need drops from millions to thousands — which fits comfortably in a CI run. The table below shows the relationship concretely.

| Per-run manifest prob. $p$ | Runs for 50% catch | Runs for 95% catch | Runs for 99.9% catch |
|---|---|---|---|
| $10^{-6}$ (native, quiescent) | ~693,000 | ~3.0 million | ~6.9 million |
| $10^{-3}$ (light fuzzing) | ~693 | ~2,995 | ~6,905 |
| $10^{-1}$ (widened window) | ~7 | ~29 | ~66 |
| $0.5$ (yield in the gap) | 1 | ~5 | ~10 |

The numbers come from $N = \ln(1 - \text{target}) / \ln(1 - p)$. Read it as the core economic argument for stress testing: a native race needs millions of runs you cannot afford; a *widened* race needs a few dozen you run in a second. Engineering the window is what makes the search tractable.

### Stress in other ecosystems: the same move, different knobs

The window-widening tactic is language-agnostic; only the yield primitive changes. In Java, `Thread.yield()` is the cooperative hint, but the more reliable widener inside a stress loop is `Thread.onSpinWait()` in a short busy loop or a `LockSupport.parkNanos(1)` between the read and the write of a critical section; the JMM-correct way to *prove* the fix is `jcstress` (next section). In Rust, `std::thread::yield_now()` plays the role of `sched_yield`, and `loom` plays the role of exhaustive exploration. In C++, `std::this_thread::yield()`. The pattern is identical across all of them: find the gap, widen it, assert the invariant, iterate. Here is the Rust stress shape for the same counter, deliberately racing a non-atomic field through `unsafe` to demonstrate the widened window (you would normally never write this — it exists to be caught):

```rust
use std::thread;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

fn main() {
    let counter = Arc::new(AtomicI64::new(0));
    let mut handles = vec![];
    for _ in 0..8 {
        let c = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                let v = c.load(Ordering::Relaxed); // 1. load
                thread::yield_now();               // 2. widen the window
                c.store(v + 1, Ordering::Relaxed); // 3. store stale value back
            }
        }));
    }
    for h in handles { h.join().unwrap(); }
    // Even though these are "atomic" ops, the load-yield-store is NOT atomic,
    // so updates are lost — prints far below 8000, reliably, thanks to yield.
    println!("{} (expected 8000)", counter.load(Ordering::Relaxed));
}
```

The lesson generalizes: a load-modify-store split into separate atomic loads and stores is *not* atomic, and the yield exposes it exactly as `sched_yield` did in C. The fix is a single `fetch_add(1, Ordering::Relaxed)` — one atomic read-modify-write, not three operations with a gap. Atomic *operations* are not the same as atomic *sequences*, and stress testing is how you feel the difference.

## Deterministic replay and schedule exploration

Stress testing samples interleavings, cleverly biased, but it still samples — it can never *prove* a structure is correct, only fail to find a bug. For small, critical, hand-rolled concurrency primitives (a lock-free queue, a custom mutex, a sequence-lock, a wait-free counter), sampling is not good enough; you want **exhaustive** coverage. Schedule exploration tools take control of the scheduler away from the OS, run your code under a *model* scheduler, and systematically enumerate every distinct interleaving that the memory model permits. If none of them violates your assertion, you have a proof — for the bounded scenario you specified.

Three tools dominate, one per ecosystem, and they share the core idea.

### Rust: loom

[`loom`](https://github.com/tokio-rs/loom) is a model checker for concurrent Rust. You write your data structure against loom's drop-in replacements for `std::sync` and `std::sync::atomic`, write a small test that exercises it from a few threads, and loom runs that test *under every interleaving and every permitted memory-model reordering*. It uses an efficient exploration strategy (partial-order reduction plus a bounded form of the C11 memory model) so it doesn't redundantly explore equivalent orderings. Crucially, loom models *weak* memory: it will explore the relaxed-ordering reorderings that a real ARM chip could produce but your x86 dev box never would, catching missing-`Acquire`/`Release` bugs that no amount of x86 stress testing finds.

```rust
// Run with: RUSTFLAGS="--cfg loom" cargo test --release
#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::thread;

#[test]
#[cfg(loom)]
fn concurrent_increment_is_exhaustively_checked() {
    loom::model(|| {
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let t = thread::spawn(move || {
            n2.fetch_add(1, Ordering::Relaxed);
        });
        n.fetch_add(1, Ordering::Relaxed);
        t.join().unwrap();
        // loom replays EVERY interleaving; assertion must hold in all of them
        assert_eq!(2, n.load(Ordering::Relaxed));
    });
}
```

`loom::model` runs the closure repeatedly, once per distinct schedule, and if any schedule trips the `assert_eq!` it reports the exact thread-step sequence that did it — a deterministic, replayable trace, not a flaky failure. The catch is the **state-space explosion**: the number of interleavings grows combinatorially with thread count and step count, so loom is for *small* harnesses — two or three threads doing a handful of operations each. That is exactly the size of a lock-free primitive's core invariant, which is why loom is the right tool there and the wrong tool for an integration test.

### Java: jcstress

The [Java Concurrency Stress tool (`jcstress`)](https://github.com/openjdk/jcstress) is an OpenJDK harness built by the people who wrote the Java Memory Model. You annotate a tiny "actor" method per thread and declare the *acceptable* and *forbidden* result tuples; jcstress runs the actors concurrently billions of times under aggressive JIT optimization and memory-model stress, records the frequency of every observed outcome, and flags any *forbidden* outcome that occurred — including ones that only appear once in a billion runs because of a specific JIT reordering. It is the gold standard for proving (empirically, at massive scale) what a piece of JMM-sensitive code can and cannot observe.

```java
// IntCounterTest.java — run via the jcstress harness
import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.I_Result;

@JCStressTest
@Outcome(id = "2", expect = Expect.ACCEPTABLE,           desc = "Both increments seen")
@Outcome(id = "1", expect = Expect.ACCEPTABLE_INTERESTING, desc = "Lost update — the race")
@State
public class IntCounterTest {
    int x; // plain int, no synchronization

    @Actor public void actor1() { x++; } // thread 1: load-add-store
    @Actor public void actor2() { x++; } // thread 2: load-add-store

    @Arbiter public void check(I_Result r) { r.r1 = x; } // observe final value
}
```

jcstress will run this an enormous number of times and report that the outcome `1` (a lost update) occurs with some nonzero frequency — empirically demonstrating the race at a scale no hand-written loop reaches, and on the actual JIT-compiled code with all its reorderings. Switch `int x` to `AtomicInteger` with `getAndIncrement()` and rerun: the `1` outcome vanishes, proving the fix across billions of trials. jcstress has been used inside the JDK itself to validate `VarHandle` semantics and concurrent classes — it is not a toy.

### Go: controlled scheduling and GODEBUG

Go does not ship a full model checker, but it gives you scheduler control hooks that make exploration tractable. `GOMAXPROCS=1` forces a single OS thread so goroutines are cooperatively scheduled, which combined with `runtime.Gosched()` at chosen points lets you *script* interleavings deterministically. The `GODEBUG` environment variable exposes scheduler tracing (`GODEBUG=schedtrace=1000` prints scheduler state) and, important for replay, `GODEBUG=randomizeallocs=...`-style knobs and the `asyncpreemptoff` flag let you control preemption to make a run reproducible. The community tool [`rrace`/`go-deadlock`] and, more powerfully, the research tool **GoChecker / GFuzz** systematically mutate channel-operation orderings and `select` choices to explore Go-specific concurrency bugs (channel races, `select` nondeterminism). The pragmatic Go workflow is: `-race` for data races, a stress test with `runtime.Gosched()` jitter for race conditions, and `GOMAXPROCS=1` + scripted yields when you need a *deterministic replay* of a specific interleaving to debug it.

The unifying idea across all three: **stop letting the OS pick the schedule; pick it yourself, exhaustively or scriptedly, so the search is a proof (small scope) or a replay (debugging), not a gamble.** The figure later in the toolbox section places these "dynamic, run-it" tools next to the "static, design" tools so you can see where exploration sits.

### Deterministic replay: turning a flaky failure into a movie you can pause

There is a deeper capability lurking in scheduler control: **deterministic replay**. Once you can dictate the interleaving, you can *record* the exact schedule that produced a failure and *replay* it as many times as you want, with a debugger attached, stepping through the precise instruction order that broke the invariant. This is the difference between a Heisenbug that vanishes when you attach `gdb` (because the debugger perturbs timing) and a bug you can single-step at will.

Three mechanisms deliver replay at different levels. At the **runtime** level, forcing `GOMAXPROCS=1` plus scripted `runtime.Gosched()` calls (Go) or running a single OS thread with cooperative yields gives you a deterministic goroutine order you control. At the **process** level, record-and-replay tools — Mozilla's `rr` for Linux being the standard — record all sources of nondeterminism (including the scheduler's choices and signal delivery) and replay the execution bit-for-bit, so a race that failed once during recording fails identically on every replay, debugger attached, reverse-stepping supported. `rr` is the single most powerful tool for a concurrency bug you can reproduce *once*: record that one failure, then replay and reverse-execute until you find the exact moment the invariant broke. At the **exhaustive** level, `loom` and `jcstress` give you not one schedule but *every* schedule, and report the specific failing one as a deterministic trace — replay is implicit because the schedule is enumerated, not sampled.

The workflow that ties it together: stress until it fails *once* under `rr` recording, then stop sampling and start *replaying* — you now own a deterministic, debuggable copy of a bug that was nondeterministic five minutes ago. This is the bridge between "I saw it fail" and "I understand exactly why," and it is why making the bug reproducible (step 1 of the workflow) is worth so much effort: a reproducible race is one `rr replay` away from being a fully understood one. This connects directly to the discipline in [the debugger is a microscope](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) — replay turns the microscope on a target that normally won't hold still.

#### Worked example: loom finding a missing release

A developer wrote a lock-free single-producer single-consumer ring buffer in Rust and tested it under heavy x86 stress for an hour — zero failures. They used `Ordering::Relaxed` on the head/tail indices because "x86 is strongly ordered anyway." Under loom, the test failed in *seconds*: loom explored an interleaving (legal under the C11 model, observable on ARM) where the consumer read the new tail index before the producer's data write was visible, returning a torn/garbage element. The fix was `Ordering::Release` on the producer's tail store and `Ordering::Acquire` on the consumer's tail load, establishing the happens-before edge that makes the data write visible. No amount of x86 stress testing would have found this, because x86's TSO model doesn't permit that reordering — but the deployed ARM servers did. Exhaustive, memory-model-aware exploration caught a bug that sampling on the wrong hardware structurally *could not*.

## Model checking the design: TLA+ and litmus tests

Everything so far tests *code*. But the most expensive concurrency bugs are *design* bugs — a flawed protocol, a broken consensus rule, a cache-coherence invariant that doesn't hold — and by the time you have code, the design flaw is baked into thousands of lines. Model checking lets you find the flaw in the *design*, before the code, by writing the algorithm in a specification language and having a tool exhaustively check that your invariants hold across every reachable state.

**TLA+** (and its more programmer-friendly syntax, PlusCal) is the dominant tool. You specify the system as a state machine: variables, an initial predicate, and a next-state relation describing every atomic step any process can take. You write *invariants* ("the balance is never negative," "two nodes never both think they're leader," "no message is delivered twice") and *temporal properties* ("every request is eventually answered"). The TLC model checker then explores **every reachable interleaving of every process's steps** and reports a concrete counterexample trace the moment an invariant breaks — a step-by-step sequence of states leading to the violation. Because it works on the abstract design, it finds protocol bugs that are *interleaving-deep*: the exact sequence of partial failures and message reorderings that breaks consensus.

This is not academic. Amazon Web Services has documented using TLA+ to find serious bugs in the design of DynamoDB, S3, and EBS replication protocols — bugs that surfaced only in interleavings requiring dozens of steps, far beyond what any test could enumerate, and that would have been catastrophic and nearly impossible to debug in production. The model checker found them in the spec, before a line of production code shipped. A sketch of the PlusCal style for the overdraw bug:

```python
""" PlusCal-style pseudocode (real syntax is TLA+/PlusCal, checked by TLC).
    This is illustrative; see Further reading for the real language.

    variables balance = 100;
    process Withdrawer in {1, 2}
      variable local;
    begin
      Read:     local := balance;          # atomic step: read
      Check:    if local >= 40 then        # atomic step: check
      Withdraw:   balance := local - 40;   # atomic step: write (STALE local)
                end if;
    end process

    Invariant:  balance >= 0
    TLC explores BOTH processes interleaving every step and reports:
      Read(1), Read(2), Check(1), Check(2), Withdraw(1), Withdraw(2)
      -> balance = 20 - 40 = -20  VIOLATION
"""
```

TLC hands you that six-step counterexample trace — the exact interleaving where both withdrawers read 100, both pass the check, and the second write overdraws — which is precisely the race condition `-race` could never see. You fix the *design* (make read-check-write a single atomic step, i.e. hold a lock or use a transaction) and re-check until the invariant holds across the entire state space. Because the language is concept-first, this is the natural place to link out to the broader treatment of [consistency models for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects), which is the systems-level cousin of these invariants.

**Litmus tests** are model checking's small, hardware-focused sibling. A litmus test is a tiny multi-thread program plus a question about which final states are *possible* under a given memory model — for example, "can both `r1` and `r2` read 0 in the classic store-buffering test?" Tools like `herd7` and `litmus7` (from the `diy` toolkit) take the memory model of x86, ARM, POWER, or RISC-V as a formal input and tell you *exactly* which outcomes that hardware permits, and run the same test on real silicon to confirm. This is how you answer "is this `Relaxed` atomic actually safe on ARM?" without owning an ARM farm and stress-testing for a week. Litmus tests are why we *know* x86 is TSO and ARM is weakly ordered — they are the formal ground truth beneath every memory-ordering decision. When your bug smells like a memory-model bug (works on x86, fails on ARM, involves `relaxed`/`acquire`/`release`), a litmus test of the exact access pattern is the surgical tool.

## Fault injection

Race detection and exploration assume the threads run; many concurrency bugs only appear when something *fails* mid-flight. **Fault injection** deliberately introduces failures — a thread killed while holding a lock, a partial write, a dropped message, a timeout, a `kill -9` between two steps of a supposedly-atomic update — to test whether your concurrent code's invariants survive the failure interleavings, not just the success ones.

The techniques range from crude to surgical:

- **Crash injection**: kill the process (or a node) at chosen points and verify recovery leaves a consistent state. The discipline of [reproducing it first](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) applies doubly here — you script the crash point so the failure is deterministic, not a one-in-a-thousand power-loss.
- **Latency/error injection**: wrap I/O, lock acquisition, or RPC calls in a shim that randomly delays or fails them, widening failure windows the way `sched_yield` widens race windows. Tooling: `failpoint` libraries (used heavily in TiDB, etcd), `toxiproxy` for network faults, `libfiu` for syscall-level injection in C.
- **Lock-holder death**: in a stress test, abort a thread while it holds a lock and assert that the system either recovers (lease/timeout-based locks) or fails safe — never silently deadlocks the rest. This is how you find the bug where a panic inside a critical section leaves a non-reentrant lock held forever.

```go
// failpoint-style injection: a test-only hook that forces a crash mid-update.
func (s *Store) Commit(key, val string) error {
	s.wal.Append(key, val)         // step 1: write-ahead log
	failpoint.Inject("crashAfterWAL", func() {
		panic("injected crash between WAL and apply")
	})
	s.apply(key, val)              // step 2: apply to state
	return nil
}
// Test enables the failpoint, runs Commit, kills the goroutine after the WAL
// write, restarts, and asserts recovery replays the WAL so state is consistent.
```

Fault injection is the bridge between concurrency testing and resilience testing — it answers "what does my supposedly-atomic operation do when it's interrupted in the middle?" That question is invisible to a race detector (no memory race occurs) and to most stress tests (which run to completion), so it is a genuinely distinct tool in the box. It connects to the broader topic of [delivery semantics and exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once), where fault injection is how you actually verify your "exactly once" claim survives a crash between deliver and acknowledge.

The defining property of a good fault-injection test is that the fault point is **deterministic**, not probabilistic. A chaos-engineering experiment that randomly kills processes in production is valuable for finding *unknown* failure modes, but for *verifying a specific atomicity claim* you want a named injection point you can enable, trigger, and assert on repeatably — exactly like the `failpoint.Inject("crashAfterWAL", ...)` hook above. That determinism is what lets the test become a regression guard: you can prove the recovery path works, commit it, and know the next refactor that breaks recovery will fail the test rather than silently corrupting state during the next real crash. The combinatorics also matter: an operation with $k$ steps has $k+1$ "crash here" points, and a multi-step protocol across two participants has the product of their step counts as candidate failure interleavings, so you enumerate the injection points the way a model checker enumerates schedules — systematically, not randomly — and assert the invariant survives each one. This is fault injection done as a small bounded exploration, which is the most reliable form of it.

## A workflow: reproduce, minimize, fix, regression-guard

Tools are useless without a discipline that turns a flaky symptom into a closed bug. The workflow that works is the same scientific loop as all debugging, specialized for nondeterminism. The figure lays out the five stages as a stack.

![the bug hunting workflow as five stacked stages reproduce minimize diagnose fix and add a regression test](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-4.png)

**1. Reproduce — make it fail often.** A bug you can only see in production is not yet debuggable. Your first job is to raise the manifestation probability from "once a week in prod" to "most runs locally." Use everything from the stress section: widen the window with injected yields, crank up thread count and contention, run under `-race`/TSan (which catches data races from a *single* bad interleaving even if output is right), and run thousands of iterations. The goal is a command you can run that fails reliably. This is the hardest and most important step — see [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) for why everything downstream depends on it.

**2. Minimize — shrink to the core.** A reliable 50,000-line repro is still nearly impossible to reason about. Cut everything that doesn't affect the failure: reduce thread count to the minimum that still races (usually 2), strip unrelated operations, replace real I/O with stubs, shrink data to the smallest that triggers it. Bisection helps here — [binary-search your bug](/blog/software-development/debugging/binary-search-your-bug-with-bisection) by halving the code or the commit range until the smallest failing core remains. A two-thread, ten-line repro is something you can hold in your head; that's the target.

**3. Diagnose — name the shared mutable state.** With a minimal repro, identify *exactly* what is shared and mutable and which happens-before edge is missing. The detector report points at the two accesses; the minimized code makes the missing ordering obvious. This is where you decide the *class* of bug: data race (missing synchronization on a memory access) or race condition (missing atomicity across a sequence). The fix differs by class.

**4. Fix — establish the order.** Add the cheapest synchronization that establishes the needed happens-before edge: an atomic for a single counter, a mutex for a multi-field invariant, a wider critical section for a check-then-act, a channel/ownership transfer to avoid sharing entirely, or the correct memory ordering on a lock-free structure. The series' spine applies: name what's shared, establish happens-before, choose the cheapest mechanism that buys it. Then re-run the repro — it must now pass deterministically.

**5. Regression-guard — lock it down.** This is the step teams skip and regret. A concurrency fix with no test protecting it will be re-broken by the next refactor, and you'll be back at 3 AM. Add a *stress* test (the fuzzed, high-iteration harness from earlier, ideally under `-race`) or, for a primitive, a `loom`/`jcstress` harness that **fails on the old code and passes on the fixed code**. Verify it fails when you revert the fix — a regression test that passes on the buggy code is worthless. Now the bug is closed *and stays* closed.

#### Worked example: closing the payment race

Back to the payment service from the intro. **Reproduce**: a stress test spun up 8 goroutines each withdrawing from one shared account 1000 times, under `-race`, asserting `balance >= 0` after — failed within 50 iterations (detector flagged the data race on the balance field *and* the assertion caught the overdraw). **Minimize**: cut to 2 goroutines, one withdrawal each, stripped logging and metrics — still raced, now ten lines. **Diagnose**: the balance was read and written under *different* lock acquisitions (a `Balance()` call then a `Withdraw()` call), leaving a TOCTOU gap — a race condition, not just a data race. **Fix**: a single `Withdraw` method that holds one lock across check-and-decrement, using a compare-and-act inside the lock. **Regression-guard**: the stress test became a permanent CI job, `go test -race -count=10 ./payments/...`, verified to fail on the reverted code. Total time from "we have a corruption report" to "closed forever": about a day, almost all of it spent on step 1, reproduction. Once it reproduced, the rest was mechanical.

## Measured: detection probability and detector overhead

Honesty about numbers is the whole point of this series, so here is how these tools actually measure up, with the caveats that keep the numbers truthful.

**Detection probability vs iterations.** The table in the stress section is not hypothetical — it is the governing equation. The actionable measurement is: instrument your stress harness to *count manifestations per iteration* and estimate $\hat{p}$, then compute how many runs you need for your confidence target. On the widened counter race above, I have repeatedly measured $\hat{p} \approx 0.4$–$0.6$ with the `sched_yield` in the gap on a 4-core machine — so a handful of iterations suffices. Remove the yield and $\hat{p}$ drops below $10^{-4}$ on the same box — five orders of magnitude, from the *same code*, purely from window-widening. That swing is the single most important empirical fact in this post: **stress testing is not "run it more," it is "engineer $p$ upward by orders of magnitude so the search is cheap."** Measure your $\hat{p}$ before and after adding jitter; if it didn't move by orders of magnitude, your jitter isn't in the right place.

**Detector overhead.** ThreadSanitizer and Go's `-race` impose roughly a **5–10x CPU slowdown** and a large memory multiplier (TSan reserves a big shadow region; expect several times the application's resident memory). These are documented order-of-magnitude figures from the tool authors and match what I see in practice — but the exact factor depends heavily on memory-access density: a memory-bound workload pays more (more accesses to instrument), a compute-bound one pays less. The honest way to report it is a range and the workload. Here is a representative comparison; treat the multipliers as order-of-magnitude, not precise.

| Tool / mode | CPU slowdown | Memory overhead | Coverage on one run | False positives |
|---|---|---|---|---|
| Native (no instrumentation) | 1x | 1x | none — only manifest bugs | n/a |
| Go `-race` / TSan (happens-before) | ~5–10x | ~5–10x | data races on executed paths | none |
| Stress + fuzz (no detector) | ~1–2x | ~1x | races + race conditions, sampled | none |
| Stress + `-race` combined | ~5–15x | ~5–10x | data races + invariants, sampled | none |
| `loom` / `jcstress` (exhaustive) | huge per scope | small scope | *all* interleavings, bounded scope | none |
| TLA+ / TLC (design) | N/A — checks spec | state-space bound | *all* design interleavings | none |

The shape of the trade-off: detectors give precise, false-positive-free data-race detection at a constant 5–10x, but only on what you run. Stress is cheap and catches race *conditions* too, but only samples. Exhaustive tools prove correctness but only for tiny scopes. Model checking proves the *design* but not the code. **No row dominates** — which is exactly why the real answer is "layer them," and why the next section is a decision table, not a single recommendation.

How to measure overhead honestly: warm up (let the JIT/cache settle), run many times, report median and spread not a single number, pin the platform (x86 TSO vs ARM weak ordering can change *which bugs even exist*), and never report a slowdown without naming the workload's memory-access density. A "3.2x slowdown" with no workload named is a number you made up.

## Case studies / real-world

**Go's `-race` and the standard library.** Go shipped its race detector (built on ThreadSanitizer) in Go 1.1, and the Go team ran it across the standard library and the runtime, finding and fixing a long tail of latent data races in packages that had passed tests for years — races in test helpers, in caching layers, in lazy-initialization code. The detector became standard practice: large Go shops run their entire integration suite under `-race` precisely because it catches the latent race from a single bad interleaving rather than waiting for corruption in prod. The lesson that generalized: a detector's value is proportional to the concurrency you drive through it, so you run it under load, not just unit tests.

**jcstress inside the JDK.** The OpenJDK team built `jcstress` to validate the Java Memory Model implementation itself and the `java.util.concurrent` classes. It has been used to verify the semantics of `VarHandle`, `Atomic*` classes, and lazy-initialization idioms (the famous double-checked-locking and "safe publication" patterns) by running tiny actor harnesses billions of times under aggressive JIT optimization and reporting the frequency of every outcome — including JMM-permitted reorderings that occur once in billions of runs. jcstress is how the JDK *knows*, empirically, that a given `volatile` or `final`-field idiom is safe across all the reorderings real JITs produce. It remains the reference tool for any JVM concurrency primitive.

**TLA+ at Amazon Web Services.** AWS engineers documented (in the paper "How Amazon Web Services Uses Formal Methods," CACM 2015) using TLA+ to model-check the designs of core services — DynamoDB, S3, EBS, and a lock service. TLC found **bugs that required interleavings dozens of steps deep**, including a data-loss bug in a replication-and-fault-recovery protocol that no test or code review had caught, because the triggering sequence of partial failures and reorderings was far beyond what any test could enumerate. The bugs were in the *design*, found in the *spec*, before code shipped. The team's stated conclusion: model checking found subtle bugs they would otherwise have shipped, and gave them the confidence to make aggressive optimizations they'd otherwise have feared. This is the canonical evidence that design-level model checking pays for itself on systems where a concurrency bug is catastrophic.

**ThreadSanitizer in Chromium and the kernel.** TSan (and its kernel cousin KCSAN, the Kernel Concurrency Sanitizer) has found thousands of data races in large C/C++ codebases — Chromium runs TSan continuously and has fixed a long stream of renderer and IPC races; the Linux kernel's KCSAN has surfaced numerous data races on fields that "everyone assumed" were safe, leading to added `READ_ONCE`/`WRITE_ONCE` annotations and proper barriers. These are the highest-volume real-world proof that dynamic detection at scale, run continuously in CI and on fuzzers, is the single most cost-effective concurrency-bug-finding technique for large systems code.

## When to reach for this (and when not to)

Every technique here is a cost — instrumentation overhead, harness-writing effort, specification effort, or false-positive triage. Spend deliberately. Before the decision table, it helps to see the whole toolbox at once, split along the one axis that matters most: do you *run the program* to observe a real interleaving, or do you *reason about the design* before any code executes? The tree below organizes every tool we've covered along that dynamic-versus-static split — TSan, stress, and `loom`/`jcstress` on the run-it side; TLA+, litmus tests, and the type system (Rust's `Send`/`Sync`, which prevents the race at compile time) on the design side.

![tree of the concurrency bug toolbox split into dynamic run-it tools and static or design tools each with two examples](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-7.png)

The type-system branch deserves a callout because it is the cheapest tool of all: Rust's `Send` and `Sync` marker traits make a whole class of data races *unrepresentable* — code that would share a non-thread-safe value across threads simply does not compile, so the bug is caught at build time with zero runtime cost and zero false negatives for that class. It doesn't catch race *conditions* (logic bugs are still logic bugs), but it eliminates the data-race class so thoroughly that "fearless concurrency" is not marketing. Where your language offers such a static guarantee, take it first; it's free. The matrix below then maps the common symptom to the first *dynamic* tool to reach for when the type system can't help; the prose after it says when *not* to.

![matrix mapping a suspected data race a rare flaky test a lock free structure and a protocol design each to the right tool and the reason](/imgs/blogs/finding-concurrency-bugs-race-detectors-and-stress-testing-8.png)

**Reach for a dynamic race detector (`-race`, TSan) — almost always, first.** It is cheap to turn on, has zero false positives, and catches the highest-frequency bug class (data races) from a single bad interleaving. Run your tests and a load test under it in CI. *Don't* rely on it alone for rare races (drive concurrency through it with stress), and *don't* expect it to find race conditions that aren't data races — it structurally cannot.

**Reach for stress testing + scheduler fuzzing — when you have a flaky failure or a higher-level invariant to defend.** It is the only tool that catches race *conditions* (check-then-act, ordering bugs) and the way you raise a rare race's probability into the catchable range. *Don't* mistake passing stress runs for proof — they sample; a green stress run bounds the bug rate, it doesn't eliminate it. And *don't* run stress without assertions on your invariants; a stress test that only checks "didn't crash" misses the silent-corruption bugs.

**Reach for exhaustive exploration (`loom`, `jcstress`) — when you hand-roll a lock-free or memory-order-sensitive primitive.** For a custom queue, lock, sequence-lock, or anything using `relaxed`/`acquire`/`release`, exhaustive memory-model-aware checking is the only thing that catches the ARM-only reordering your x86 box hides. *Don't* point it at an integration test or anything with more than two or three threads and a handful of steps — the state space explodes and it never finishes. It is a scalpel for primitives, not a hammer for systems.

**Reach for model checking (TLA+) — when the *design* is the risk.** Distributed protocols, consensus, replication, anything where a subtle interleaving of partial failures loses data or breaks an invariant, and where a production bug would be catastrophic and nearly undebuggable. *Don't* model-check a CRUD endpoint or a single-machine mutex — the specification effort only pays off when the design is genuinely hard and the blast radius is large.

**Reach for fault injection — when "atomic" operations cross failure boundaries.** Crashes mid-update, dropped messages, lock-holder death, exactly-once claims. *Don't* skip it for any system that claims to recover cleanly from a crash — that claim is untested until you've injected the crash.

The decisive rule: **layer cheapest-first.** `-race` on everything (cheap, always-on), stress + assertions for invariants (cheap, catches race conditions), exhaustive exploration for primitives (expensive, narrow), model checking for designs (expensive, high-value), fault injection for resilience claims. Most teams stop after the first two and that is genuinely most of the value; reach for the heavy tools only where the cost of a bug justifies them. The full decision framework for *which concurrency model* to use in the first place — the move that prevents most of these bugs — is in the [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

## Key takeaways

1. **You cannot find a race by hoping it shows up.** Re-running tests samples a heavily-biased distribution over interleavings; a green suite over thousands of runs is weak evidence. Do the probability math: at $p=10^{-6}$ you'd need a million runs for an even chance of a catch.
2. **A dynamic race detector catches a latent data race from one bad interleaving, even when the output is correct** — that's its superpower. Turn `-race`/TSan on in CI today; it's the highest-value, lowest-effort concurrency tool.
3. **Happens-before is precise (no false positives) but only sees executed orderings; lockset is order-insensitive but noisy.** Modern detectors use happens-before with shadow memory and pay a 5–10x slowdown for it.
4. **Detectors find data races, not race conditions.** A properly-locked check-then-act has no data race and a real bug; only stress with invariant assertions, exhaustive exploration, or model checking finds it.
5. **Stress testing works by engineering $p$ upward.** A `sched_yield` in the race window turns a one-in-a-million race into a near-certain one; measure $\hat{p}$ before and after — if it didn't move orders of magnitude, your jitter is in the wrong place.
6. **Exhaustive exploration (`loom`, `jcstress`) proves a small primitive correct across all interleavings and all permitted memory reorderings** — including the ARM-only ones your x86 box hides. Use it for hand-rolled lock-free structures; never for large scopes.
7. **Model checking (TLA+) finds design bugs before code exists** — the interleaving-deep protocol flaws that no test can enumerate. Reserve it for hard, high-blast-radius designs.
8. **The discipline is reproduce, minimize, diagnose, fix, regression-guard.** A concurrency fix without a stress/`loom`/`jcstress` test that fails on the old code will be silently re-broken by the next refactor.
9. **Layer cheapest-first and measure honestly.** No single tool dominates; report slowdowns as ranges with the workload named, warm up, run many times, and pin the platform — x86 TSO and ARM weak ordering change which bugs even exist.

## Further reading

- **Within this series**: [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) (the foundations), [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) and [data races vs race conditions: a precise distinction](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction) (the bug classes these tools target), and the [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) (choosing a model that avoids the bug).
- **Debugging discipline**: [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging), [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection), and [race conditions, the hardest bugs to catch](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch).
- Savage, Burrows, Nelson, Sobalvarro, Anderson, **"Eraser: A Dynamic Data Race Detector for Multithreaded Programs"** (1997) — the lockset algorithm, the foundation of dynamic detection.
- Serebryany & Iskhodzhanov, **"ThreadSanitizer: data race detection in practice"** (2009) — the happens-before + shadow-memory design behind TSan and Go's `-race`.
- Newcombe et al., **"How Amazon Web Services Uses Formal Methods"** (CACM, 2015) — the canonical case for model-checking distributed designs with TLA+.
- Leslie Lamport, **"Specifying Systems"** and the TLA+ Home Page / TLA+ Video Course — the language and the TLC model checker.
- The **`loom`** and **OpenJDK `jcstress`** project documentation — exhaustive interleaving exploration for Rust and the JVM, the right tools for lock-free primitives.
- Herlihy & Shavit, **"The Art of Multiprocessor Programming"** — linearizability and the correctness conditions these tools verify; Maranget, Sarkar & Sewell, **"A Tutorial Introduction to the ARM and POWER Relaxed Memory Models"** for the litmus-test foundations.

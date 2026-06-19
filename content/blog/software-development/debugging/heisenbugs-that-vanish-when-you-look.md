---
title: "Heisenbugs That Vanish When You Look: Debugging the Observer Effect in Software"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn why the act of observing a bug can make it disappear, how to name the four heisenbug families from how they hide, and how to pin one with rr, sanitizers, and a build bisect without scaring it off."
tags:
  [
    "debugging",
    "software-engineering",
    "heisenbug",
    "race-condition",
    "undefined-behavior",
    "record-replay",
    "sanitizers",
    "memory-corruption",
    "rr",
    "observer-effect",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/heisenbugs-that-vanish-when-you-look-1.png"
---

There is a particular kind of 3am despair that you only earn once. The crash is real. You have the customer ticket, you have the stack trace, you have three separate users on three continents reporting the same corrupted output. So you reproduce it locally — and it reproduces, beautifully, ten times in a row. Then you do the most natural thing in the world: you add a single `printf` right before the line that explodes, to see the value of the variable that must be wrong.

And the bug is gone.

You remove the `printf`. The bug comes back. You add it again. Gone. You attach `gdb` and set a breakpoint instead. Gone. You rebuild with `-O0` so the debugger can see your locals. Gone — and not just hidden, *gone*, 0 out of 1000 runs. You rebuild with `-O2`, and there it is again, 847 out of 1000. At this point a junior engineer concludes the universe is malicious and the senior engineer smiles a tired smile, because they know exactly what they are looking at. This is a **heisenbug** — a bug whose behavior changes when you try to observe it, named after Heisenberg's uncertainty principle, where the act of measuring a system disturbs the thing you are measuring. The disappearance is not random. The disappearance is the single most valuable clue you will get, and most engineers throw it away by treating it as bad luck.

![Tree diagram showing the four heisenbug families branching from a single vanishing bug, with each family labeled by the perturbation that hides it: timing, uninitialized memory, optimizer undefined behavior, and memory corruption masked by debug padding](/imgs/blogs/heisenbugs-that-vanish-when-you-look-1.png)

By the end of this post you will be able to do four things. First, **name the family** of a heisenbug from how it vanishes — "it disappears under a debugger" means timing; "it only happens in release" means undefined behavior or uninitialized memory; "it dies under a debug allocator" means corruption. Second, **stop scaring it off** by switching to a low-perturbation observation method — record-replay with `rr`, ring-buffer logging, sampling profilers, post-mortem core dumps — so you can watch the failure without changing the timing that produces it. Third, **pin the root cause** with the right sanitizer: ThreadSanitizer for the race, UBSan for the optimizer-deleted check, MemorySanitizer or Valgrind for the uninitialized read, AddressSanitizer for the overrun. And fourth, **bisect the build** — not the code — when source is identical but a flag flips the bug, because that flip points a finger straight at undefined behavior. This is the same loop the whole series runs on, the one laid out in [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging): observe, reproduce, hypothesize, bisect, fix, prevent. A heisenbug just makes the *observe* step adversarial — your microscope is part of the experiment.

## 1. The observer effect is real in software, not a metaphor

Let me start by killing the mystique. "Heisenbug" sounds like quantum mysticism, but there is nothing quantum about it. The reason observing a bug changes its outcome is brutally mechanical: **your probe is code, and code costs time, memory, and instructions, all of which the bug depends on.** A bug exists because of some precise alignment of conditions — a 80-nanosecond window where two threads can interleave, a stack slot that happens to hold garbage, a value the optimizer decided to keep in a register. When you insert an observation — a print, a breakpoint, a debug build, a sanitizer — you change one or more of those conditions. Sometimes you change them enough to push the bug out of existence.

There are exactly five levers your probe pulls, and every heisenbug story is one or more of them.

**Timing.** A `printf` is not free. Under the hood it formats a string and then makes a `write(2)` syscall, which crosses into the kernel, possibly blocks on a lock for the file or terminal, and returns microseconds later. A few microseconds is an eternity to a CPU — modern cores retire billions of instructions per second, so 5 microseconds is roughly 15,000 instructions of "dead time" you just injected between two operations. If a data race had a window of 80 nanoseconds, inserting 5,000 nanoseconds of syscall in the middle of it does not just shrink the window; it reshuffles the entire schedule, often serializing two threads that used to overlap. The race is still in your code. It just can't happen anymore while you're watching.

**Memory layout.** A debug build and a release build do not lay memory out the same way. Debug builds commonly pad structures, insert guard bytes around allocations, and place locals further apart for the debugger's convenience. AddressSanitizer goes further and wraps every allocation in *redzones* — poisoned bytes that no legitimate access should touch. So a one-byte buffer overflow that, in the tight release layout, lands on the very next field and corrupts it, can in the debug layout land harmlessly in padding that nobody reads. Same bug, different consequence, purely because the bytes moved.

**Optimization.** This is the big one and the most misunderstood. The optimizer does not just make code faster; it *rewrites your program under the assumption that you never invoke undefined behavior.* If your code has signed integer overflow, a null dereference the compiler can prove, or a strict-aliasing violation, the optimizer is allowed to assume those never happen and delete the code that handles them. At `-O0` the compiler is literal and your "safety check" runs; at `-O2` the same check can be optimized away because the compiler proved (using your UB as a premise) that it's dead code. The bug that "only happens in release" is very often this: the optimizer acted on undefined behavior that the unoptimized build happened to tolerate.

**Initialization.** Many debug runtimes zero-fill freshly allocated memory, or fill it with a recognizable poison pattern like `0xCDCDCDCD` or `0xDEADBEEF`. Release builds do not — they hand you whatever bytes were there before. So a bug that reads an uninitialized variable can produce a clean zero in debug (which happens to be a valid, harmless value) and produce stack garbage in release (which is not). The bug — reading before writing — is identical. The *value* you read is build-dependent, and so the crash is build-dependent.

**Scheduling.** A breakpoint stops one thread dead while the others may or may not keep running, and stepping forces a serialization that real concurrent execution never has. A debugger turns a parallel program into a nearly sequential one. Any bug that requires two threads to be genuinely in flight at the same instant simply cannot occur while you single-step. This is why "it works under the debugger" is practically a diagnosis on its own: you almost certainly have a timing bug.

Hold those five levers in your head — timing, layout, optimization, initialization, scheduling — because every heisenbug is a story about which lever your probe pulled. The skill is not memorizing them. The skill is running the inference *backwards*: from "it vanished when I added a log" to "therefore it's timing" to "therefore I need a low-perturbation observation method." That backward inference is the entire game.

> One framing I find clarifying: in normal debugging, your instrument is passive — a thermometer reads the temperature without changing it much. In heisenbug debugging, your instrument is hot enough to change the temperature. The fix is not to give up measuring; it's to switch to a colder thermometer.

## 2. Family one — timing bugs the probe widens or closes

The most common heisenbug, by a wide margin, is a **race condition** whose timing window your probe disturbs. Let me define terms precisely, because the field is sloppy about them. A **data race** is two threads accessing the same memory location concurrently, at least one of them writing, with no synchronization (no happens-before edge) between them. A **race condition** is the broader category: any bug where correctness depends on the relative timing of events. Every data race is a race condition; not every race condition is a data race. (For the full treatment of how to catch races — Helgrind, TSan, the happens-before model — see the sibling on race conditions, the hardest bugs to catch; if you have read [reading a stack trace across languages](/blog/software-development/debugging/reading-a-stack-trace-across-languages) you already know how little a stack trace tells you when the corruption happened in a *different* thread than the one that crashed.)

Here is the canonical reproducer — two threads incrementing a shared counter without a lock:

```c
// race.c — build: cc -O2 -pthread race.c -o race
#include <pthread.h>
#include <stdio.h>

static long counter = 0;          // shared, unsynchronized
#define N 1000000

void *worker(void *arg) {
    for (int i = 0; i < N; i++) {
        counter++;                // read, add, write — NOT atomic
    }
    return NULL;
}

int main(void) {
    pthread_t a, b;
    pthread_create(&a, NULL, worker, NULL);
    pthread_create(&b, NULL, worker, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    printf("expected %d, got %ld\n", 2 * N, counter);
    return 0;
}
```

`counter++` is three machine operations: load `counter` into a register, add one, store it back. When two threads do this with no lock, one thread can load the old value, the other can load the same old value, both add one, both store — and you have lost an update. Run this and you will see something like `expected 2000000, got 1373quelque` — a different wrong number every time. So far this is a plain race, not yet a heisenbug.

It *becomes* a heisenbug the moment you try to debug it the obvious way. Suppose the lost-update only manifests as a downstream symptom — a corrupted total in a report, say — and you want to watch the increment happen. You add a print inside the loop:

```c
for (int i = 0; i < N; i++) {
    counter++;
    fprintf(stderr, "t=%p i=%d counter=%ld\n", (void*)pthread_self(), i, counter);
}
```

Now the `write(2)` behind `fprintf` runs on every iteration. Each one is microseconds. The two threads spend almost all their time in the kernel doing I/O, serialized behind the same stderr lock, and the actual increment becomes a vanishingly small fraction of the loop. The window in which both threads are simultaneously in the load-add-store sequence collapses toward zero. The lost-update rate plummets — and if your symptom only fires when enough updates are lost to cross some threshold, the symptom disappears entirely. You "fixed" it by observing it.

![Graph showing a shared counter with an 80-nanosecond race window that produces a torn write when threads interleave freely, but where adding a 5-microsecond log syscall serializes the writes and makes the bug appear correct](/imgs/blogs/heisenbugs-that-vanish-when-you-look-3.png)

The mechanism, stated rigorously: a data race exists because the language memory model provides **no happens-before edge** between the two unsynchronized accesses. A *happens-before* edge is the formal guarantee that one operation's effects are visible to another — it is established by synchronization (a mutex unlock that happens-before the next lock, a release-store that happens-before the matching acquire-load, thread creation that happens-before the thread's first instruction, a join that happens-after the thread's last). Where such an edge exists, the result is defined. Where it is missing — as between our two raw `counter++` operations — the C and C++ standards declare the behavior *undefined*, which is the same word from section 4: the compiler may assume races don't happen and optimize accordingly, and the hardware may make each core's writes visible to the other in any order its cache-coherence protocol allows. Without that edge, the compiler and CPU are free to reorder, cache, and interleave the operations however they like, and the observable result depends on the exact cycle-by-cycle schedule. Your probe changes the schedule. The bug — the missing happens-before edge — is *still there*; you have merely picked a schedule in which it doesn't bite. This is why "add a lock" is the real fix and "add a log" is not: the lock *creates the missing edge*; the log only reshuffles a schedule that is still, formally, undefined.

Crucially, the probe can also go the *other* way. Sometimes a print **widens** the window. If the race needs thread A to be paused at a specific point while thread B runs past it, and your print happens to pause thread A right there, you can make a one-in-a-billion race fire on every run. That's a gift — a heisenbug you can summon on demand is a heisenbug you can fix. The general principle: a probe perturbs timing, and perturbation can hide *or* expose the bug. Your job is to figure out which, and then to choose probes deliberately.

#### Worked example: the flake that failed 6 of 2000 runs

A team had an integration test that failed about 0.3% of the time — 6 failures in 2000 CI runs over a month, always with a corrupted aggregate count. Re-running passed. Adding debug logging made it pass 5000 times straight, so the logging "fixed" it and was left in, and three weeks later the bug shipped to production anyway (because production didn't have the logging). Classic timing heisenbug: the log perturbation closed the window in CI but production's hot path had no logging.

The diagnosis came from refusing to trust the disappearance. They first made the bug *common* instead of rare, because a 0.3% bug is untestable but a 60% bug is trivially testable. The lever: pin both threads to the same physical core and remove the work between the racing accesses, so the threads contend maximally:

```bash
# Force the race to be common: one core, tight loop, repeat until fail
for i in $(seq 1 200); do
  taskset -c 0 ./repro || { echo "FAILED on run $i"; break; }
done
```

With the work stripped out and both threads on one core, the failure rate jumped from 0.3% to roughly 60% — failing within the first three or four runs almost every time. Now it was observable. They ran it once under ThreadSanitizer:

```bash
cc -O1 -g -fsanitize=thread -pthread repro.c -o repro_tsan
./repro_tsan
```

TSan does not rely on the race actually losing an update on this particular run; it tracks the happens-before graph and reports a data race the instant it sees two unsynchronized accesses to the same address with no edge between them, even if the schedule happened to be benign. It printed the two stack traces — the write in `aggregate_update` and the read in `flush_totals` — with no lock between them. Root cause located. The fix was a single mutex around the shared aggregate. Re-run under the original CI loop: **0 failures in 2000 runs**, then 0 in another 5000 to be sure. The flake rate went from 0.3% to 0%, and this time it was 0% in production too, because the fix was real and not a timing accident.

The lesson is the spine of this whole series. A heisenbug is not a reason to stop reproducing — it's a reason to reproduce *harder and differently*. You make the rare common (more cores, more contention, more iterations, randomized scheduling), then you use a tool that detects the *cause* (TSan finds the missing edge) rather than the *effect* (the lost update), so you don't need the unlucky schedule to occur while you watch.

## 3. Family two — uninitialized memory whose garbage depends on the build

The second family is the one that screams "only in release." A variable is read before it is written. In a debug build, the runtime may have zero-filled or poison-filled that memory, so you read a predictable value — often a harmless zero. In a release build, you read whatever bytes the previous owner of that stack slot or heap block left behind. The bug is the same read-before-write; the *value* differs by build, and so the *behavior* differs by build.

```c
// uninit.c — reads `total` before it is always initialized
#include <stdio.h>

int compute(int *items, int n, int mode) {
    int total;                    // NOT initialized
    if (mode == 1) {
        total = 0;
        for (int i = 0; i < n; i++) total += items[i];
    }
    // mode == 0 falls through: `total` is never set
    return total;                 // reads garbage when mode == 0
}

int main(void) {
    int data[] = {3, 1, 4};
    printf("%d\n", compute(data, 3, 0));   // undefined: uninitialized read
}
```

In a debug build where the stack frame happens to start zeroed, `compute(..., 0)` returns 0 — looks like a sensible default, ships, nobody notices. In release, the stack slot for `total` is reused from a prior call and holds, say, `32767` or a pointer's low bytes, and now a downstream allocation sized by that "total" goes haywire. The crash is 10,000 lines away from the bug, and it only happens in release. Engineers waste days because the symptom and the cause are in different builds *and* different files.

The reason this is a heisenbug specifically: every observation tool you reach for changes initialization. `-O0` debug build → memory zeroed → bug hidden. `gdb` → often runs the debug build → bug hidden. Even adding a print of an *unrelated* local can shift the stack layout enough that `total`'s slot now aliases a different prior value, changing the symptom. You are chasing a ghost made of stale stack bytes.

The method here is precise, and there are two great tools. **Valgrind's memcheck** tracks the definedness of every bit and complains the instant an uninitialized value affects a branch or a syscall:

```bash
# Valgrind: catches use of uninitialized values, no recompile needed
valgrind --track-origins=yes ./uninit
# ==12345== Conditional jump or move depends on uninitialised value(s)
# ==12345==    at 0x... compute (uninit.c:11)
# ==12345==  Uninitialised value was created by a stack allocation
# ==12345==    at 0x... compute (uninit.c:4)
```

`--track-origins=yes` is the magic flag — it tells you not just *that* an uninitialized value was used but *where it was born*, which in this case is the `int total;` declaration on line 4. That's the root cause, handed to you. The cost is real: Valgrind runs the program on a synthetic CPU and is typically 10–30× slower, so it's a "run once to find the bug" tool, not a "leave on in CI for every test" tool.

**MemorySanitizer (MSan)** does the same job — detect reads of uninitialized memory — but as a compile-time instrumentation, so it's far faster (roughly 2–3× overhead) at the cost of needing a rebuild *and* needing the whole program, including libc, to be instrumented (which is the annoying part):

```bash
# MSan: faster than Valgrind, needs recompilation of the whole stack
clang -O1 -g -fsanitize=memory -fsanitize-memory-track-origins=2 uninit.c -o uninit_msan
./uninit_msan
# WARNING: MemorySanitizer: use-of-uninitialized-value
#     #0 ... compute uninit.c:11
#   Uninitialized value was created by an allocation of 'total' ...
```

Both tools find the *cause* (the read of an undefined value) and not just the *effect* (the eventual crash), which is exactly what you need, because the effect is the part that mutates by build.

| Tool | Finds uninitialized reads? | Recompile? | Overhead | When to reach |
| --- | --- | --- | --- | --- |
| Valgrind memcheck | Yes, with origins | No | ~10–30× | Quick triage, no build changes possible |
| MemorySanitizer | Yes, with origins | Yes, whole stack | ~2–3× | CI on a project you fully control |
| Compiler `-Wuninitialized` | Some, statically | No (just `-Wall`) | None | Free first pass, misses cross-function cases |
| Zeroing in debug only | Hides the bug | n/a | n/a | This is the trap, not a tool |

Note the last row, because it's the heisenbug trap in table form: *initializing memory in your debug build is not a fix.* It's a perturbation that hides the bug from you while leaving it live in production. The fix is to initialize the variable in *all* builds — `int total = 0;` — and, better, to turn on `-Werror=uninitialized` and let the compiler reject the read-before-write at build time.

#### Worked example: the report that was right in staging and wrong in prod

A reporting service computed correct numbers in staging and subtly wrong numbers in production for one rarely hit code path. Staging ran a debug build (zero-filled stack); production ran an optimized build with garbage stack. The wrong number was a row count used to size a buffer, and the path that left it uninitialized was the "empty result set" branch — which staging's test data never hit but production occasionally did.

The investigation refused to chase the wrong *number* (which changed run to run in prod, an uninitialized-memory fingerprint) and instead reproduced under MSan with the production-like optimized flags. MSan fired on the first run of the empty-result path, pointing at the declaration. Total time from "MSan run" to "root cause line": about four minutes. The fix was a one-line initializer plus a `-Werror=uninitialized` build flag so it could never regress. Before: intermittently wrong, untraceable from the symptom. After: correct, and the class of bug is now a compile error. The decisive move was recognizing the fingerprint — *a value that differs every run and only in the optimized build is almost always uninitialized memory* — and reaching for the tool that finds the cause regardless of which garbage value showed up today.

## 4. Family three — optimizer-dependent undefined behavior

This is the family that produces the cruelest sentence in debugging: "it only crashes in the release build, and adding any debug code makes it go away." It feels supernatural. It is not. It is **undefined behavior (UB)** that the optimizer acted upon.

Here is the contract you may not have known you signed. The C and C++ standards define behavior only for programs that stay within the rules. The instant your program does something undefined — signed integer overflow, dereferencing a null the compiler can prove, reading past an array bound, violating strict aliasing — the standard imposes *no requirements whatsoever* on what the program does. Compiler writers exploit this aggressively for speed: they assume UB *never occurs*, and they optimize using that assumption as a theorem. If you wrote a check that only makes sense when UB has already happened, the compiler can prove the check is unreachable and delete it. (The sibling post [integer overflow and floating point traps](/blog/software-development/debugging/integer-overflow-and-floating-point-traps) goes deep on why `a + b` can wrap or trap; here the point is what the *optimizer* does with it.)

The textbook example — a self-defeating overflow check:

```c
// ub.c — the deleted overflow check
#include <limits.h>
#include <stdio.h>

int will_overflow(int a, int b) {
    int sum = a + b;              // signed overflow is UB
    if (sum < a) {               // "did it overflow?" — but UB already happened
        return 1;                 // the compiler may DELETE this branch
    }
    return 0;
}

int main(void) {
    printf("%d\n", will_overflow(INT_MAX, 1));
}
```

Reason through what `-O2` does. The compiler knows signed overflow is undefined, so it is entitled to assume `a + b` *never* overflows. Under that assumption, `sum = a + b` always satisfies `sum >= a` when `b >= 0`, so `sum < a` is provably false and the entire `if` branch is dead code. It deletes `return 1`. Your overflow check — the one whose whole purpose was to catch overflow — is gone, because the compiler used "overflow is impossible" as a premise. At `-O0`, the compiler is literal: it computes `INT_MAX + 1`, which wraps to `INT_MIN` on typical hardware, `INT_MIN < INT_MAX` is true, and the check "works." So the bug is masked at `-O0` and exposed at `-O2`. Same source. The optimization level flips it.

![Before-and-after comparison contrasting a debug build at minus O0 where the overflow guard runs and zero crashes occur, against a release build at minus O2 where the compiler deletes the check and 847 of 1000 runs crash with UBSan flagging the line](/imgs/blogs/heisenbugs-that-vanish-when-you-look-2.png)

The method has two halves: **bisect the build** to localize the flag, then **sanitize** to name the UB. When the source is byte-identical between the build that crashes and the build that doesn't, you don't bisect commits — you bisect the *build configuration*. Start coarse:

```bash
# Does it crash at each optimization level?
for opt in -O0 -O1 -O2 -O3; do
  cc $opt -g ub_repro.c -o /tmp/repro_$opt
  if /tmp/repro_$opt; then echo "$opt: clean"; else echo "$opt: CRASH"; fi
done
# -O0: clean ... -O1: clean ... -O2: CRASH ... -O3: CRASH
```

You've now bracketed it between `-O1` and `-O2`. `-O2` is a *bundle* of individual optimization passes, and you can bisect within it. GCC and Clang let you toggle individual flags; `-O2` turns on a known set, and you can subtract them one at a time (`-O2 -fno-strict-overflow`, `-O2 -fno-strict-aliasing`, etc.) to find which one flips the bug:

```bash
# Bisect within -O2: which flag re-hides the bug? (the one that does is the clue)
for flag in -fno-strict-overflow -fno-strict-aliasing -fno-delete-null-pointer-checks -fwrapv; do
  cc -O2 $flag -g ub_repro.c -o /tmp/repro_flag
  if /tmp/repro_flag; then echo "$flag: HIDES the bug -> UB is here"; else echo "$flag: still crashes"; fi
done
# -fno-strict-overflow: HIDES the bug -> UB is signed overflow
# -fwrapv: HIDES the bug -> confirms: define wrap-around and it's gone
```

The flag that hides the bug *names the undefined behavior class*. If `-fwrapv` (define signed overflow as two's-complement wrap) or `-fno-strict-overflow` makes it vanish, your UB is signed overflow. If `-fno-strict-aliasing` fixes it, you have a type-punning aliasing violation. If `-fno-delete-null-pointer-checks` fixes it, the compiler deleted a null check because you dereferenced a pointer before testing it. This is the build-bisect as a diagnostic — the build axis is just another axis to binary-search across, exactly like commits or config, as in [binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection).

![Timeline showing a build bisection that starts clean at minus O0 and minus O1, crashes at minus O2, then narrows through six flag-splitting steps to the strict-overflow culprit, with UBSan confirming signed overflow as the root cause](/imgs/blogs/heisenbugs-that-vanish-when-you-look-5.png)

Then you confirm with **UndefinedBehaviorSanitizer (UBSan)**, which instruments the program to trap *at the exact operation* that invokes UB, before the optimizer has had a chance to weaponize it:

```bash
# UBSan: trap on the UB itself, with file:line
cc -O1 -g -fsanitize=undefined -fno-sanitize-recover=all ub_repro.c -o repro_ubsan
./repro_ubsan
# ub_repro.c:6:17: runtime error: signed integer overflow:
#   2147483647 + 1 cannot be represented in type 'int'
```

There it is: `signed integer overflow` at `ub_repro.c:6`. Not a crash 10,000 lines later — the actual operation, with the actual values, with the file and line. `-fno-sanitize-recover=all` makes it abort on the first finding instead of printing and continuing, which is what you want when hunting a single root cause. Once UBSan names it, the fix writes itself: do the check *before* the overflow, using a form that has no UB —

```c
#include <stdbool.h>
bool will_overflow(int a, int b) {
    return __builtin_add_overflow_p(a, b, (int)0);   // checks WITHOUT overflowing
}
```

`__builtin_add_overflow_p` (GCC/Clang) computes whether the addition would overflow without ever performing the undefined operation, so the optimizer has nothing to exploit. In portable C23 there's `<stdckdint.h>` with `ckd_add`. The point: you don't fix UB by adding a runtime check *after* the fact (the optimizer deletes it); you fix it by never invoking the UB in the first place.

#### Worked example: bisected to -O2 in six steps, root-caused in one UBSan run

A media-encoding library shipped fine for a year, then a new build toolchain bumped the default optimization and a customer reported corrupted output on large files only. The maintainers couldn't reproduce on small files; the corruption needed an input large enough to push an internal offset past `INT_MAX`. Source unchanged from the last good release — only the compiler version and default flags changed.

They bisected the *build*, not the code. Step 1: `-O0` clean, `-O3` corrupt — UB suspected. Steps 2–3: bracketed to `-O1` clean, `-O2` corrupt. Steps 4–6: split `-O2`'s flag set, and `-fwrapv` made the corruption vanish — six bisection steps total to go from "somewhere in -O2" to "signed overflow." One UBSan run then printed `signed integer overflow: 2147483647 + 4096 cannot be represented in type 'int'` at the offset computation. Root cause: an `int` offset that overflowed on files larger than 2 GiB, and the compiler had deleted the bounds check that "protected" it. The fix was a one-line type change from `int` to `size_t` plus a UBSan-clean build in CI. Before: corrupt on large files, only in the new build, unreproducible by the maintainers. After: correct on all sizes, and UBSan in CI would have caught it pre-release. The whole investigation, once they decided to bisect the *build axis*, took an afternoon — the discipline was refusing to stare at the diff (there was none) and instead binary-searching the only axis that had actually changed.

## 5. Family four — memory corruption a debug allocator masks

The fourth family is the mirror image of the third: it appears in *release* and vanishes in *debug*, but the mechanism is memory layout, not the optimizer. You have a buffer overflow, a use-after-free, or a double-free. In the tight release layout, the stray write lands on something live and corrupts it; in the debug layout — with its padding, guard bytes, and fill patterns — the same write lands somewhere harmless, or the debug allocator notices and the program limps on. (The full mechanics of how a freed block gets reused and crashes far away live in [use after free and memory corruption](/blog/software-development/debugging/use-after-free-and-memory-corruption); here the point is why the *debug build* hides it.)

A classic one-byte (fence-post) overflow:

```c
// overrun.c — off-by-one write past the end of a heap buffer
#include <stdlib.h>
#include <string.h>

struct record {
    char name[16];
    int  length;                 // sits right after name in release layout
};

void fill(struct record *r, const char *src) {
    // BUG: copies up to and including index 16 -> one past name[15]
    for (int i = 0; i <= 16; i++) r->name[i] = src[i];   // <= is the bug
    r->length = (int)strlen(r->name);
}
```

In a packed release layout, `name[16]` is the first byte of `length`, so the overrun silently corrupts `length`, and the bug surfaces later as a wrong size, a bad allocation, or a crash in unrelated code. In a debug build that inserts padding between `name` and `length`, byte 16 lands in the pad and `length` is untouched — no symptom. Worse, some debug heaps put guard bytes after each allocation, so the overrun lands in a guard byte that nobody reads. The corruption is real in both builds; only release places something valuable where the stray byte lands.

![Stack diagram showing a one-byte buffer overrun that overwrites the adjacent length field in the packed release layout but lands harmlessly in the eight-byte guard padding of the debug layout, with the AddressSanitizer redzone catching the overflow](/imgs/blogs/heisenbugs-that-vanish-when-you-look-6.png)

The right tool is **AddressSanitizer (ASan)**, which does not rely on the corruption hitting something important. ASan surrounds every allocation with poisoned redzones and checks every memory access against shadow memory; the instant your code touches a redzone byte, it reports — with the allocation site, the access site, and the exact byte offset:

```bash
cc -O1 -g -fsanitize=address overrun.c driver.c -o overrun_asan
./overrun_asan
# ==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
# WRITE of size 1 at 0x... thread T0
#     #0 ... fill overrun.c:13
# 0x... is located 0 bytes to the right of 16-byte region
#  allocated by thread T0 here:
#     #0 ... malloc
#     #1 ... make_record driver.c:7
```

"0 bytes to the right of 16-byte region" is the off-by-one in plain English: you wrote exactly one byte past a 16-byte buffer, at `overrun.c:13`. ASan finds the bug *at the moment of the bad access*, not when the corrupted `length` finally causes a crash three functions later. That temporal locality — error reported at the cause, not the effect — is the whole reason sanitizers beat live debugging for this family.

The heisenbug trap, again in one sentence: *running the corrupting program under a debug allocator or with extra padding can hide the corruption while leaving it live in release.* So when someone says "it crashes in prod but I can't reproduce it in my debug build," your first hypothesis should be memory corruption that the debug layout masks — and your first action is an ASan build, which deliberately makes the corruption *louder*, not quieter.

| Tool | Catches | Recompile? | Overhead | Best for |
| --- | --- | --- | --- | --- |
| AddressSanitizer | Overflow, UAF, double-free | Yes | ~2× CPU, ~3× RAM | First reach for corruption |
| Valgrind memcheck | Overflow, UAF, leaks, uninit | No | ~10–30× | When you can't recompile |
| Debug heap (e.g. glibc `MALLOC_CHECK_`) | Some overruns, double-free | No (env var) | Low | Cheap first signal in prod-ish builds |
| ThreadSanitizer | Data races (not corruption) | Yes | ~5–15× | The timing family, not this one |

Pick the row by the constraint: can you recompile (ASan), or not (Valgrind)? And note the bottom row — TSan is the wrong tool for a corruption bug; matching tool to *family* is the meta-skill this whole post is teaching.

#### Worked example: the crash that moved when you added a field

A team had a use-after-free that crashed in `parse_header` in release, never in debug. The classic pattern: a freed object's memory was reused by a later allocation, so reading through the stale pointer returned *plausible* data in debug (the debug allocator delayed reuse and poisoned freed blocks with `0xDD`, which made the stale read obviously wrong and was caught by an existing assertion) but returned *another live object's* bytes in release (no poisoning, immediate reuse), so the stale read silently succeeded and corrupted state that crashed elsewhere.

What made it a maddening heisenbug: adding an unrelated `int debug_counter;` field to a nearby struct made the crash move to a different function, and removing it moved it back. That sensitivity to an irrelevant change is a *fingerprint* — when a crash relocates because you added a field that the buggy code never touches, you are looking at memory corruption whose victim depends on allocation layout. They stopped editing fields and ran ASan once: `heap-use-after-free`, `READ of size 8`, with both the free site and the allocation site in the report, plus "freed by thread T0 here." The free was in an error path that ran `free(conn)` but left a cached pointer to `conn` live; a later request reused the block. Time from ASan run to root cause: about three minutes. The fix was nulling the cached pointer on free (and, for prevention, a `-fsanitize=address` CI job on the parser). Before: relocating crash, only in release, sensitive to unrelated edits — the textbook corruption heisenbug. After: 0 crashes in 50,000 fuzzed inputs under ASan. The decisive recognition was the *relocation* clue: a bug that moves when you add an unrelated field is corruption, and the cure is to make the corruption loud (ASan), not to keep nudging the layout until it hides again.

## 6. Heisenbugs above C — managed runtimes, async, and the GC

It is tempting to think heisenbugs are a C and C++ problem — that garbage-collected, memory-safe languages are immune. They are immune to *one* family (the corruption family largely goes away when you can't write past a buffer), but the timing and observation-perturbation families are alive and well in Go, Java, Python, and JavaScript. The mechanism shifts, but the principle — *your probe changes the schedule or the layout* — is identical.

**Go.** Go has true OS-thread-backed goroutines, so data races are real and common. The canonical Go heisenbug is a map written from two goroutines without a mutex: it usually works, then occasionally corrupts the map's internal structure and panics with `fatal error: concurrent map writes` — but only when the two writes truly overlap, which adding a `fmt.Println` (a syscall) reliably prevents. The fix is not the print; it's `sync.Mutex` or a `sync.Map`. The tool is the built-in race detector, which is ThreadSanitizer internally:

```go
// race_test.go — run with: go test -race ./...
func TestConcurrentMap(t *testing.T) {
    m := map[int]int{}
    var wg sync.WaitGroup
    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func() { defer wg.Done(); m[1] = 1 }() // unsynchronized write -> race
    }
    wg.Wait()
}
```

```bash
go test -race ./...
# ==================
# WARNING: DATA RACE
# Write at 0x... by goroutine 8:
#   ... race_test.go:NN
# Previous write at 0x... by goroutine 7:
#   ... race_test.go:NN
# ==================
```

`go test -race` reports the race on the *cause* — two unsynchronized writes with no happens-before edge — even if this particular run didn't actually corrupt the map. Like all sanitizers, it finds the missing edge, not the unlucky crash, so it doesn't need the heisenbug to fire while you watch. Leaving `-race` on in CI is the standard prevention for the whole Go timing family.

**Java and the JVM.** The JVM adds a heisenbug source most people forget: the **garbage collector pauses**. A bug that depends on a specific allocation pattern can manifest only when a GC happens mid-operation, and GC timing is nondeterministic — change the heap size, the collector, or add a `System.out.println` that allocates, and you've changed *when* GC runs, which changes whether the bug fires. Java's memory model also permits surprising reorderings of non-`volatile` field writes, so a thread can see a half-constructed object (the classic broken double-checked-locking bug) only under a precise timing the debugger serializes away. The tools: `jstack` for a no-pause thread dump (what is every thread doing right now), `jcmd <pid> Thread.print`, a heap dump (`jmap -dump`) loaded into Eclipse MAT for the allocation family, and the Java Flight Recorder / `async-profiler` for low-perturbation sampling. The discipline is the same — sample and dump rather than freeze.

**Python and async.** CPython's Global Interpreter Lock means two *bytecode* operations can't run truly in parallel, which masks many races — but a race condition (timing-dependent bug) absolutely still happens at the boundaries where the GIL is released (I/O, C extensions, `time.sleep`), and in `asyncio` an `await` is a yield point where another coroutine can run and mutate shared state between your check and your use. The async TOCTOU is the modern Python heisenbug: it only fires when the scheduler happens to run another task at the `await`, and adding a `print` (or any other `await`) changes the interleaving. `faulthandler` (dump all thread stacks on a signal or timeout), `py-spy dump --pid <pid>` (a no-pause stack snapshot of a live process), and `tracemalloc` (allocation tracking for the leak family) are the cold instruments; see [logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) for how to log async flows without serializing them. The point stands across every runtime: managed memory removes the corruption family, but the *observer effect on timing* is universal — wherever there is a schedule, a probe can perturb it.

| Runtime | Surviving heisenbug families | Cold instrument | Race/timing detector |
| --- | --- | --- | --- |
| C / C++ | All four | `rr`, core dump, ASan-redzone | TSan, Helgrind |
| Go | Timing (data races, map writes) | `pprof` sample, goroutine dump | `go test -race` (TSan) |
| Java / JVM | Timing, GC-pause-dependent | `jstack`, heap dump, JFR | TSan-for-Java tools, `-Xcheck` |
| Python | Timing at GIL release, async TOCTOU | `py-spy dump`, `faulthandler` | manual reasoning, `pytest` stress |

## 7. The master move — low-perturbation observation with rr

Everything above shares one idea: when your normal microscope is too hot, switch to a colder one. The single most powerful cold microscope for heisenbugs is **record-replay**, and the best implementation on Linux is `rr` (from Mozilla). I cannot overstate how much `rr` changes heisenbug work. It is, for the timing and concurrency families, close to a cheat code.

Here is what `rr` does. It runs your program once and **records the exact, deterministic execution** — every nondeterministic input the program received: the precise interleaving of threads, the results of every syscall, the values returned by `rdtsc`, signal deliveries, everything. The recording has very low overhead because `rr` serializes threads onto a single core and records scheduling decisions rather than fighting them. Then you replay that recording as many times as you want, and **every replay is byte-for-byte identical** — the same thread interleaving, the same "random" values, the same crash on the same instruction every single time. You debug a *recording*, not a live process, so your debugging actions (breakpoints, stepping, printing) do not perturb anything — the timing is already fixed in the recording. The heisenbug can't run away because it already happened, on tape.

![Matrix comparing five observation methods across their perturbation cost, what they reveal, and whether they are safe for heisenbugs, showing that ring-buffer logging, rr record-replay, and sampling profilers are low-perturbation while print and live debuggers are not](/imgs/blogs/heisenbugs-that-vanish-when-you-look-4.png)

And `rr` supports **reverse execution** — `reverse-continue`, `reverse-step`, `reverse-next` — so you can run *backwards* from the crash to the corruption. This is the killer feature. Normally, when memory is corrupted, you crash at the read and have no idea where the bad write came from; you'd set a hardware watchpoint and re-run, hoping the bug reproduces. With `rr`, you set the watchpoint on the corrupted address and `reverse-continue` — it runs backward until the watchpoint last fired, landing you exactly on the instruction that wrote the bad value, in the right thread, with the full backtrace. You travel from effect to cause in one command.

```bash
# Record once — captures the exact failing execution
rr record ./buggy_program --args-that-trigger-it
# ... it fails, the recording is saved ...

# Replay deterministically under a gdb-like prompt
rr replay
# (rr) continue                 # runs to the crash, same every time
# (rr) print bad_struct->length # inspect the corrupted value
# (rr) watch -l bad_struct->length
# (rr) reverse-continue         # run BACKWARD to the write that corrupted it
# Thread 2 hit watchpoint: bad_struct->length
#   #0 fill overrun.c:13        # the exact bad write, in the right thread
# (rr) bt                       # full backtrace at the moment of corruption
```

For a race, the workflow is even sweeter. Record until you catch a failing run (often you script `rr record` in a repeat-until-fail loop), and now you have *one* deterministic instance of a bug that fails 0.3% of the time. You replay it forever, identically. You set watchpoints, you step both threads, you `reverse-continue` from the torn read to the unsynchronized write — all on a recording that never changes. The race that vanished under every live probe is now sitting still on your desk.

![Before-and-after comparison showing live debugging where adding a log or attaching gdb makes the race vanish and only 6 of 2000 runs fail, versus rr record-replay that captures the failure, reverse-continues to the bad write, and confirms zero failures after the fix](/imgs/blogs/heisenbugs-that-vanish-when-you-look-7.png)

When `rr` isn't available — it's Linux-only and needs certain CPU performance-counter support — there are lighter cold microscopes, and you should know the ladder:

- **Ring-buffer / deferred logging.** Instead of a synchronous `write(2)` per event (microseconds), append a fixed record to a pre-allocated in-memory ring buffer (tens of nanoseconds, no syscall), and only flush to disk when something fails or on demand. The perturbation drops by two or three orders of magnitude, small enough to leave the race window open. This is how you "add logging" to a timing bug without closing it. (Tracing frameworks like LTTng and the kernel's `ftrace` are exactly this idea, industrial-strength.)
- **Hardware tracing (Intel PT).** Intel Processor Trace records the program's control flow in hardware with single-digit-percent overhead, so you can reconstruct exactly which branches executed without inserting a single instrumentation instruction. `perf record -e intel_pt//` captures it; it's the lowest-perturbation way to see control flow.
- **Sampling profilers.** `perf`, `py-spy`, and friends interrupt the program periodically and record the stack, rather than instrumenting every call. Overhead is ~1%, low enough to run in production and low enough not to move a timing bug.
- **Post-mortem core dumps.** Don't attach a live debugger to the misbehaving process at all — let it crash, capture a core dump, and debug the *corpse* offline. `gdb ./prog core` gives you the full state at the moment of death with zero perturbation of the live run. (See [reading a stack trace across languages](/blog/software-development/debugging/reading-a-stack-trace-across-languages) for getting maximum signal from a post-mortem.)

The unifying principle: **make your observation's perturbation smaller than the bug's window.** A print is 5 microseconds; an 80-nanosecond race is 60× smaller, so the print obliterates it. A ring-buffer write is 20 nanoseconds, smaller than the window, so it can coexist. `rr` is even better — it records once and then perturbs *nothing*, because you debug the past. Choose the instrument by comparing its cost to the bug's timescale.

## 8. The disappearance is information — reading the clue

Now we can assemble the backward inference into a routing table, because *how* a heisenbug vanishes tells you which family it's in, which tells you which cold microscope and which sanitizer to reach for. This is the most important habit in the whole post: stop being annoyed that it disappeared, and start asking *what the disappearance tells you*.

![Graph routing a vanishing bug by how it hides: gone under a debugger routes through rr and TSan, only in release routes through a build bisect to UBSan or through MSan and Valgrind, all converging on a confirmed root cause with zero failures after the fix](/imgs/blogs/heisenbugs-that-vanish-when-you-look-8.png)

Read the table as "symptom of disappearance → most likely family → confirming move."

| It vanishes when you... | Most likely family | Confirm / pin it with |
| --- | --- | --- |
| Add a `print` / log line | Timing (race window closed) | Ring-buffer log, then `rr` record-replay, then TSan |
| Attach a debugger / single-step | Timing (threads serialized) | `rr` (debug the recording), TSan |
| Build with `-O0` (debug) | Optimizer UB | Bisect build flags, then UBSan |
| Build with `-O2` and it APPEARS | Optimizer UB or uninit | UBSan (UB) or MSan/Valgrind (uninit) |
| Run a debug build (bug only in release) | Corruption masked by padding, or uninit | ASan (corruption) or MSan (uninit) |
| Run under a debug allocator | Memory corruption | ASan with redzones |
| Run on a different machine only | Layout / timing / CPU model | Reproduce on the affected host; capture with `rr` there |
| Wait long enough (only after hours) | Leak, fragmentation, counter wrap | Run accelerated, watch RSS, `tracemalloc`/heap diff |

Notice the symmetry that trips people up: **"appears at -O2" and "appears at -O0" are different diagnoses.** Appears at `-O2` (gone at `-O0`) is optimizer UB — the optimizer weaponized something. Appears at `-O0`/debug (gone at `-O2`) is rarer and usually means your "fix" was a side effect of debug-build padding or zeroing, i.e. you have corruption or uninit hiding in the release build that the debug build accidentally tolerated. The optimization axis is not "more optimization = more bugs"; it's "a *change* in optimization exposes a latent UB or layout assumption." Either direction of the flip is a clue.

And the most underused inference of all: **"it vanishes under a debugger" is, by itself, ~90% diagnostic of a timing bug.** Debuggers serialize. If serializing the program hides the bug, the bug needed concurrency, full stop. You can skip straight to `rr` and TSan and not waste an afternoon adding prints that will also serialize and also hide it. The disappearance routed you in one step.

## 9. Stress-testing the method — when it gets harder

The clean cases above are the easy 80%. Let me push on the hard cases, because the senior move is knowing what to do when the standard playbook stalls.

**"It only reproduces under load."** Single-threaded local runs never trigger it; only production traffic does. The bug needs contention or a queue depth or a connection-pool exhaustion you can't hit locally. The move: synthesize the load. `stress-ng` for CPU/memory/IO pressure, a load generator (`wrk`, `k6`, `vegeta`) for request volume, and run your `rr` recording or ASan build *under* that load on a staging host. If the bug needs 10,000 concurrent connections, give it 10,000. Often the heisenbug is a resource leak (file descriptors, sockets) that only manifests when the pool runs dry — covered in [resource leaks: fds, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) — and load is just how you reach the limit faster.

**"It only happens after six hours."** Time-dependent heisenbugs — a slow leak, a counter that wraps, fragmentation that eventually fails an allocation, a cache that degrades. You cannot afford a six-hour debug loop. The move: *accelerate the clock.* If it's a leak at +4 MB/min, you don't wait for OOM; you watch RSS climb for ten minutes, confirm the slope, and diff two heap snapshots (`tracemalloc` in Python, `jmap`/MAT in the JVM, `massif` in Valgrind) taken minutes apart to see what's growing. If it's a counter wrap, compute when it wraps and inject a starting value near the boundary. Turn "after six hours" into "after six minutes" by attacking the *rate*, not waiting for the *event*.

**"I can't attach a debugger in production."** The payments service cannot be paused; attaching `gdb` would freeze it and breach SLA, and a breakpoint in a hot path is an outage. This is where low-perturbation observation is not a nicety but a requirement. The move: *never touch the live process.* Capture a core dump on crash (`coredumpctl`, or set up `gcore` on a signal), debug the corpse offline. Use sampling (`perf record -p <pid>` for a few seconds, `py-spy dump --pid` for a no-pause stack snapshot). Use the ring-buffer logs you had the foresight to add. Use distributed tracing and correlation IDs to follow the request across services without stopping any of them — exactly the discipline in [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod). The rule is blunt: **don't attach `gdb` to the payments process in prod.** Reconstruct from artifacts, not from a live freeze.

**"It only reproduces on one host."** Works on every machine but one. Now the variable is the *environment*: a different CPU microarchitecture (memory ordering differs between x86 and ARM!), a different libc, a different kernel, a different glibc malloc tuning, a different NUMA topology. The move: diff the environments aggressively — compiler version, libc version, kernel version, CPU flags (`/proc/cpuinfo`), environment variables, `ulimit`s — and reproduce *on the affected host*. A race that needs ARM's weaker memory model literally cannot reproduce on x86; you must record (`rr` on the affected box) where it actually happens. The host *is* the clue.

**"Two requests have to interleave just so."** The bug needs a specific interleaving of two concurrent operations — request A reads, request B writes, A writes back — a TOCTOU (time-of-check to time-of-use) window. You can't reproduce it by luck. The move: *force the interleaving.* Add a controllable delay or a test-only synchronization point (a "schedule fuzzer" / "interleaving harness") that parks thread A at the check and lets thread B run before A proceeds. Tools like `rr`'s chaos mode (`rr record --chaos`) deliberately randomize scheduling to surface rare interleavings; ThreadSanitizer's deterministic mode and Go's race detector under stress do similar. You stop praying for the interleaving and start *driving* it.

The meta-point across all five stress cases is identical: a heisenbug resists your *default* observation, so you change the observation — accelerate it, synthesize the load, force the interleaving, capture the corpse — until the bug becomes reproducible *and* observable at the same time. That intersection is where you can finally pin it.

## 10. War stories — heisenbugs that shipped

Famous, instructive failures. I'll keep these accurate and flag any illustrative framing.

**The Therac-25 race (1985–1987).** A radiation-therapy machine delivered massive overdoses to several patients, with at least three deaths. One root cause was a race condition: if a skilled operator edited the treatment parameters faster than the software's data-entry phase expected, a one-byte counter that should have flagged an inconsistency wrapped to zero exactly when it was checked, and the safety interlock was bypassed. The bug needed a *specific timing* of operator input — fast, experienced typists triggered it; slow ones never did, which is why it survived testing and only surfaced in the field. It is the canonical heisenbug-with-fatal-consequences: a timing window that the test environment (slow, deliberate input) never opened. The lesson the field took: you cannot test your way to confidence on a concurrency bug by running the happy path; you have to reason about *every* interleaving, because the rare one is the one that kills.

**The Heartbleed read-overflow (2014).** OpenSSL's heartbeat handler trusted a length field in an attacker-controlled packet and `memcpy`'d up to 64 KB from the message buffer back to the attacker — reading far past the end of the actual payload and leaking adjacent memory (private keys, session data). Why did it survive for years? Because in most builds and most allocations, the over-read landed on benign heap bytes, and nothing visibly broke — the bug had no *symptom* under normal observation; it only mattered to an attacker deliberately reading the leaked bytes. AddressSanitizer, running with redzones, flags exactly this class of over-read instantly — and after Heartbleed, fuzzing OpenSSL under ASan became standard. The heisenbug lesson: a memory-safety bug can be completely invisible to ordinary testing precisely because the bad access usually lands somewhere harmless. You need a tool that makes the harmless-looking access *loud*.

**The leap-second cascade (2012).** When a leap second was inserted on June 30, 2012, a Linux kernel bug caused a livelock in the high-resolution timer code: threads that had set timers spun, hammering the CPU, and machines across the internet — Reddit, Mozilla, airline reservation systems — spiked to 100% CPU simultaneously. The trigger was a *once-in-years event* (a leap second) interacting with a timer-reprogramming path that almost never ran. It's a heisenbug of the "only after a rare event" family: untriggerable in any normal test, dependent on a specific moment in wall-clock time. The fix that worked operationally for many was, fittingly, to perturb time deliberately — set the clock with `date` to clear the stuck state. The lesson: some heisenbugs are gated on an event you can only reproduce by *injecting* the rare condition (here, the leap second) rather than waiting for it.

**An illustrative optimizer-UB outage (constructed, not a specific company's incident).** Picture a service that ran cleanly for a year, then a routine compiler upgrade in CI changed the default optimization, and a pointer-arithmetic path that had relied on signed-overflow wrap started corrupting output on large inputs — exactly the family in section 4. No code changed; the toolchain did. The team chased the diff for two days finding nothing, because there *was* no diff — then someone bisected the *build* and the corruption traced to the new optimizer assuming no signed overflow. This scenario is illustrative, assembled from the common pattern rather than a documented public postmortem, but the shape is real and frequent: **a heisenbug whose only "change" was the compiler.** When the source is identical and the behavior changed, the build is a variable, and the build is bisectable.

The thread through all four: heisenbugs ship because they are invisible to ordinary observation — a timing the test never hit, a read that usually lands harmlessly, an event that almost never occurs, a flag that just changed. They are not caught by *more* testing of the happy path. They are caught by tools that make the latent fault loud (ASan, TSan, UBSan), by forcing the rare condition (chaos scheduling, leap-second injection, large inputs), and by reasoning about the *space* of executions rather than the one you happened to run.

## 11. How to reach for this (and when not to)

A decisive section, because every one of these tools has a cost and the wrong reach wastes a night.

**Start by reading the disappearance, then pick the family.** Before you touch a tool, ask how it vanished and route with the table in section 7. "Gone under a debugger" → timing → `rr`/TSan. "Only in release" → UB or uninit → build-bisect + UBSan, or MSan. "Gone in debug build" → corruption → ASan. Picking the family first saves you from running the wrong sanitizer for an hour.

**Reproduce at -O0 first — unless the bug only exists at -O2.** The usual advice is to debug the unoptimized build because the debugger can see your locals and nothing is inlined. That is right *most* of the time. But for the optimizer-UB family, the bug *is* the optimization — reproduce at `-O0` and it's gone, and you'll convince yourself it's fixed when it isn't. So: try `-O0` first, but if it vanishes there, you've just learned it's an optimizer bug and you should switch to UBSan at `-O1`/`-O2` rather than fighting to debug optimized assembly. Don't chase a release-only crash by staring at `-O2` disassembly; bisect the flags to *name* the UB, then fix the UB.

**Reach for `rr` for anything timing-related — it's worth installing.** If you are on Linux and the bug smells like timing or corruption, `rr record` is almost always the fastest path to a deterministic, reverse-debuggable instance. The cost is setup (Linux, performance counters, sometimes a VM) and that recording serializes threads (so a bug that *needs* true parallelism on multiple cores may not record — `rr` runs on one core). For most app-level races, that limitation doesn't bite; record it.

**Don't leave debug logging in as a "fix."** The single most common heisenbug malpractice is: the bug disappeared when you added a log, so you keep the log and call it done. You have not fixed anything; you've added a timing perturbation that happens to hide the bug in *this* environment, and it will resurface in production, on a faster machine, under different load. If a log made the bug vanish, that is *evidence of a timing bug to be fixed*, not a fix. Remove the log and pin the real cause.

**Don't run Valgrind or `rr` in production, and don't attach `gdb` to a critical live process.** Valgrind's 10–30× slowdown and `rr`'s recording overhead are fine on a dev box, fatal on a latency-sensitive prod service. ASan's ~2× CPU and ~3× memory are sometimes acceptable in a canary or a load-test environment but rarely in the hot path of a payments system. In prod, reach for the cold tools: sampling (`perf`, `py-spy`), core dumps, ring-buffer logs, tracing. The rule from section 8 stands: don't freeze the payments process to look at it.

**When one well-placed observation answers it, stop.** Not every release-only crash needs the full sanitizer-and-bisect ceremony. Sometimes a single core dump (`gdb ./prog core`) shows you a null pointer and the fix is obvious. The discipline is proportional response: read the disappearance, form the cheapest hypothesis that explains it, and confirm with the lightest tool that can — escalate to `rr` and sanitizers only when the cheap probe perturbs the bug away.

**Build the prevention in, because heisenbugs are the ones you must not rely on luck to catch.** The reason to run ASan/TSan/UBSan in CI — even though they're too heavy for prod — is that these bug families are *exactly* the ones that slip through ordinary tests. A test suite that passes 10,000 times can still harbor a 0.01% race; TSan finds it on run one because it checks the *cause*, not the *effect*. Make a sanitizer build part of CI for any concurrent or memory-unsafe code. It's the cheapest insurance against the most expensive bugs.

## 12. Tying it back to the loop

Everything here is the series' loop — observe → reproduce → hypothesize → bisect → fix → prevent — with one twist: the *observe* step is adversarial, because your instrument perturbs the experiment. So you adapt each step.

- **Observe**, but with a cold instrument. If the warm instrument (print, debugger) hides the bug, that disappearance is your first observation, and you switch to `rr`, ring-buffer logs, sampling, or a core dump that doesn't change the timing.
- **Reproduce**, harder and differently. A rare heisenbug is reproduced by making it common — more cores, more contention, more iterations, forced interleavings, synthesized load, accelerated time, `rr --chaos`. You don't accept "it's intermittent"; you raise the rate until it's reliable.
- **Hypothesize** from the disappearance. "Gone under a debugger" → hypothesis: timing. "Only in release" → hypothesis: UB or uninit. The way it vanished *is* the hypothesis generator.
- **Bisect** the right axis. When the source is identical, bisect the *build* — optimization level, then individual flags, then compiler version — until one toggle flips the bug, which names the UB. The build is just another axis to binary-search, like commits in [binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection).
- **Fix** the cause, not the symptom. Don't keep the log that hid the race; add the lock. Don't add an after-the-fact overflow check the optimizer will delete; use `__builtin_add_overflow_p` so the UB never happens. Don't zero memory only in debug; initialize in all builds.
- **Prevent** with sanitizers in CI. The families here — race, uninit, UB, corruption — are precisely the ones a sanitizer build catches on run one and an ordinary test misses ten thousand times. Make TSan/ASan/UBSan/MSan part of the pipeline for any code that touches threads or raw memory.

There is also a discipline of *humility* in this loop that is worth naming. The reason heisenbugs humble even strong engineers is that they punish the most natural instinct in debugging — to look closer. Everywhere else, looking closer helps: more logs, a finer breakpoint, a slower step. Here, looking closer with a hot instrument is exactly what destroys the evidence. So the heisenbug forces you to invert a habit you have spent years building, and to ask, before you reach for a probe, "will this probe perturb the thing I am trying to see?" That question — *is my instrument hotter than the phenomenon?* — is the single most valuable thing to carry away. Ask it before every print, every breakpoint, every rebuild, and you will stop accidentally curing the patient you were trying to diagnose.

The mindset shift is the whole post: a heisenbug feels like the universe mocking you, but it is *the most informative kind of bug*, because the very thing that makes it maddening — that it changes under observation — is a precise fingerprint of its cause. Stop being scared of the disappearance. Read it.

## Key takeaways

- **A heisenbug is not random; your probe perturbs one of five levers** — timing, memory layout, optimization, initialization, or scheduling — and that perturbation hides (or exposes) the bug.
- **Name the family from how it vanishes.** Gone under a debugger → timing. Only in release → UB or uninitialized memory. Gone in a debug build → corruption masked by padding. The disappearance is the diagnosis.
- **Switch to a cold instrument.** Make your observation's perturbation smaller than the bug's window: `rr` record-replay (perturbs nothing — you debug the past), ring-buffer logging, sampling profilers, post-mortem core dumps. Never close the race with a synchronous `printf`.
- **`rr` is the single best heisenbug tool.** Record one failing run, replay it deterministically forever, and `reverse-continue` from the crash backward to the exact bad write, in the right thread.
- **Bisect the build, not the code, when source is identical.** Optimization level, then individual flags (`-fwrapv`, `-fno-strict-aliasing`); the flag that hides the bug names the undefined behavior.
- **Match the sanitizer to the family.** TSan for races, UBSan for optimizer UB, MSan or Valgrind for uninitialized reads, ASan for memory corruption. Each finds the *cause*, not the unlucky *effect*.
- **Make the rare common before you try to fix it.** More cores, more contention, more iterations, forced interleavings, synthesized load, accelerated clocks — raise the failure rate until the bug is reliably observable.
- **Never ship the perturbation as the fix.** A log that hides a race is evidence of a timing bug, not a solution. Remove it and fix the real cause; an overflow check the optimizer deletes is no check at all.
- **Don't attach a live debugger to a critical prod process.** Reconstruct from cold artifacts — core dumps, sampling, ring-buffer logs, tracing — and run the heavy tools (Valgrind, `rr`, ASan) in dev and CI, not in the payments hot path.
- **Prevent with sanitizers in CI.** These families slip past ordinary tests that pass ten thousand times; a TSan/ASan/UBSan build catches them on run one because it checks the cause.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post specializes for adversarial observation.
- [Binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — how to binary-search any axis, including the build configuration when the source is unchanged.
- [Integer overflow and floating point traps](/blog/software-development/debugging/integer-overflow-and-floating-point-traps) — the deep treatment of why signed overflow is undefined and how the optimizer reasons about it.
- [Use after free and memory corruption](/blog/software-development/debugging/use-after-free-and-memory-corruption) — the mechanics of allocator reuse and why a corrupting write crashes far from its cause.
- [Mastering an interactive debugger](/blog/software-development/debugging/mastering-an-interactive-debugger) — breakpoints, watchpoints, and the reverse-execution model that `rr` extends to the past.
- The race-conditions sibling in this series (planned) covers the happens-before model, Helgrind, and TSan in full — the timing family this post routes you toward.
- `rr` project documentation (rr-project.org) — recording, replay, reverse execution, and chaos mode for surfacing rare interleavings.
- The AddressSanitizer, ThreadSanitizer, MemorySanitizer, and UBSan wikis (the LLVM/Google sanitizer docs) — flags, output format, and CI integration for each family.
- *Debugging* by David Agans and *Why Programs Fail* by Andreas Zeller — the canonical books on systematic, hypothesis-driven debugging, including the observer effect and delta-debugging.

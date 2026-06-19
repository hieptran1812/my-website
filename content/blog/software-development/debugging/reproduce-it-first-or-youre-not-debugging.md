---
title: "Reproduce It First, or You're Not Debugging: Turning a Ghost Into a Falsifiable Bug"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn to turn an unreproducible ghost into a one-command, fails-every-time reproducer — pin every source of nondeterminism, shrink a 1 MB input to 12 bytes, and replay a prod-only crash on your laptop."
tags:
  [
    "debugging",
    "software-engineering",
    "reproducibility",
    "flaky-tests",
    "nondeterminism",
    "delta-debugging",
    "record-replay",
    "testing",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/reproduce-it-first-or-youre-not-debugging-1.png"
---

There is a particular flavor of helplessness that every working engineer eventually meets. A ticket comes in: "checkout occasionally returns a 500." You read the code. You read it again. You stare at the stack trace — a `NullPointerException` deep in a serializer — and you cannot for the life of you see how the field could ever be null. You run the checkout flow locally. It works. You run it ten more times. It works every single time. The bug is real; customers are hitting it; the error rate in the dashboard is a flat 0.4% that will not go away. And yet on your machine, in front of your eyes, the program is innocent.

This is the moment the whole craft of debugging hinges on, and it is the moment most engineers get wrong. The instinct is to start *guessing*: maybe it's a caching issue, maybe it's a race, maybe we should add a null check right there and ship it. But a null check that you cannot prove fixes anything is not a fix — it is a prayer with a commit message. You will deploy it, the 0.4% will or won't move, and you will have learned nothing either way, because you never had a way to ask the bug a question and get the same answer twice. The figure below shows where this fits: reproduction is not the first interesting step of debugging. It is *step zero*, the foundation the rest of the loop stands on.

![Diagram showing the six-stage debugging loop as a vertical stack with reproduction highlighted as the gating second stage that every later stage depends on](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-1.png)

This post is about that step zero. By the end of it you will be able to take a bug that "happens sometimes" and turn it into a single command that fails 100% of the time, every time, on demand. You will know how to find and pin every hidden input that makes a run non-repeatable — the random seed, the system clock, the locale, the hash-map iteration order, the thread schedule, the dependency version, the working directory. You will know how to shrink a sprawling 200-line failing scenario down to a five-line one that points straight at the cause, and how to take a one-megabyte input that crashes a parser and reduce it to the twelve bytes that actually matter. You will know how to make an intermittent bug reproduce on demand with a repeat-until-fail loop, and how to do the probability math that tells you how many times to loop. And you will know how to drag a bug that only happens in production down onto your laptop, where you can attach a debugger and finally watch it happen.

If you have not read the series intro, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) lays out the spine that everything here serves: **observe → reproduce → hypothesize → bisect → fix → prevent.** This post is the deep dive on the second box. Get it right and every box after it becomes mechanical. Get it wrong and the whole loop collapses, because you are debugging a moving target you can never pin down.

## 1. Why reproduction is not optional: it is the thing that makes debugging *science*

Let me make the strong claim plainly, because it is the load-bearing idea of this entire post. **A bug you cannot reproduce, you cannot fix — and you certainly cannot prove you fixed.** Every other technique in this series, every debugger and sanitizer and bisection script, presupposes that you can make the bug happen when you want it to. Strip that away and the techniques are useless.

Walk through why, stage by stage. To **hypothesize**, you need to form a falsifiable claim — "the field is null because the upstream service returns an empty array when the cart is exactly at the inventory limit." A hypothesis is only worth anything if you can test it, and to test it you must be able to run the failing scenario and watch whether your predicted cause is present. No repro, no test, no real hypothesis — just a hunch you will defend forever because nothing can prove it wrong.

To **bisect** — to binary-search the gap between a known-good and a known-bad commit, or between a working input and a failing one — you need a *test* that returns the same verdict every time you run it at a given point. `git bisect` is built around a command that exits 0 for good and 1 for bad. If that command sometimes says good and sometimes says bad at the *same commit*, the bisection lies to you, narrows toward the wrong commit, and you end up "bisecting" your way to a change that has nothing to do with the bug. Reproducibility is the precondition that makes bisection a tool instead of a coin flip. (We cover bisection in depth in a sibling post; here, just notice that it *requires* a deterministic verdict.)

And to **verify a fix** — the part everyone skips — you need to show the bug reproduced *before* your change and does *not* reproduce *after* it, with nothing else different. If you could never reproduce it on demand, then after your change you only know one thing: it didn't fail *this time*. That is not evidence. A bug that fires 0.4% of the time will, by definition, "not fire" on most runs whether you fixed it or not. The only honest way to claim a fix is to have a repro that fails reliably before and passes reliably after.

There is a deeper point hiding here, and it is worth saying out loud. Debugging is the most ruthlessly empirical activity in software. You have a belief about how the system behaves; the bug is proof that belief is wrong somewhere; your job is to find where. The only instrument you have for measuring reality is *running the program and observing it*. A reproducer is that instrument calibrated and locked down — same input, same environment, same result — so that when you change one thing and the behavior changes, you can attribute the difference to your change rather than to noise. Reproduction is what converts debugging from storytelling into experiment. Until you have it, you are not debugging. You are guessing with extra steps.

It is worth being precise about what "reproduce" even means, because there are degrees of it and they are not equally useful. The weakest form is "I have seen it happen at least once" — a screenshot of a stack trace in Slack. That is an *observation*, not a reproducer; it tells you the bug exists but gives you no lever to pull. The next rung is "it happens sometimes when I do roughly this" — better, but a sometimes-bug is still a moving target you cannot bisect or verify against. The form you actually want is *deterministic on demand*: one command, one input, and the bug fires every single time, with a verdict (exit code, assertion, log line) you can read mechanically. The entire arc of this post is moving a bug up that ladder — from "seen once" to "happens sometimes" to "fails 100% of the time when I run this." Each rung up the ladder is worth real effort, because each rung makes every downstream step — hypothesis, bisection, verification — cheaper and more trustworthy.

A second subtlety: a *good* reproducer is also *fast*. A bug that only reproduces after a 40-minute integration run that spins up six containers is technically deterministic, but you will run it a dozen times during an investigation, and at 40 minutes a run that is hours of dead waiting. Part of the work of reproduction is not just making the bug deterministic but making the loop tight — seconds, not minutes — so you can iterate. A 10-second repro that fails every time is worth more than a 30-minute repro that fails every time, and far more than a "it happens in prod sometimes" that you cannot run at all. Speed and determinism together are what make a reproducer an *instrument* rather than a *museum piece*.

#### Worked example: the fix that "worked" for three weeks

A team I'll describe (composited from several real incidents, presented as illustrative) had a payment-reconciliation job that occasionally double-counted a transaction. Someone added a `DISTINCT` to the query, the duplicate counts stopped appearing in spot checks, and the ticket was closed. Three weeks later the duplicates were back. The `DISTINCT` had masked the symptom on the rows people happened to look at, but the real cause was a retry that re-enqueued the job without an idempotency key, so under a specific timeout pattern the job ran twice. Nobody had ever reproduced the double-count. They had pattern-matched a symptom and shipped a guess. The actual fix — once they built a repro that re-enqueued under the timeout and reliably double-counted — was a one-line idempotency key, and *this* time they could prove it, because the repro that failed 10/10 times before now passed 10/10 times after. The lesson is not "use idempotency keys" (though, yes). The lesson is: the three lost weeks were the price of skipping step zero. For the queueing side of that story, the [message-queue series on idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) is the right place to go deeper; here the point is purely that no repro meant no proof.

## 2. The mechanism: why "the same program" produces different results

If you have only ever debugged deterministic code with `print`, the idea that running the *same binary* on the *same input* could produce different results can feel almost offensive. It shouldn't. A program is never just its source code. A run is the source code *plus* a large set of hidden inputs that the code reads from the world, and any of those changing between runs is enough to change the outcome. Reproduction fails precisely when one of those hidden inputs is not held fixed. The figure below names the worst offenders.

![Diagram showing four hidden inputs feeding into one test run that then produces a flaky pass-then-fail outcome](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-2.png)

Let me make the mechanism concrete for the nastiest category — thread scheduling — because it is the one that makes engineers throw their hands up. When two threads touch the same memory without synchronization, the language memory model gives you *no* guarantee about the order their operations become visible to each other. This is the absence of a *happens-before* edge: a happens-before relationship is the formal guarantee, established by a lock, a mutex, an atomic with the right ordering, or thread start/join, that one operation's effects are visible to another. With no such edge, the compiler is free to reorder instructions, the CPU is free to reorder memory operations, and store buffers can delay a write becoming visible to another core. So two threads incrementing the same counter can both read the old value, both add one, and both write back — and one increment vanishes. The classic lost update. We will look at exactly that interleaving later.

The crucial property for *reproduction* is this: which interleaving you get depends on the OS scheduler, cache state, what else the machine is doing, and a dozen microarchitectural details. On your quiet laptop the threads almost always interleave "nicely" and the bug hides. On a loaded CI box with sixteen noisy neighbors, the bad interleaving surfaces 0.3% of the time. The bug isn't more or less *present* on either machine — the code is identical — the *probability of hitting the bad interleaving* differs. This is the signature of a **heisenbug**: a bug whose appearance changes when you observe it, named after Heisenberg because the act of looking (adding a log line, attaching a debugger, slowing a thread) perturbs the timing enough to make it vanish.

The same "hidden input" framing explains the tamer offenders too, and it is worth naming each because the fix is different for each:

- **Random seeds.** Any call into a PRNG without a fixed seed makes the run depend on entropy you didn't pin. Fuzzers, shuffled test orders, jittered retries, sampled logging — all seeded from the clock or `/dev/urandom` by default.
- **The system clock and timezone.** Code that branches on "now" — token expiry, date rollovers, DST transitions, "is it a business day" — behaves differently depending on *when* you run it. The infamous "only fails near midnight UTC" bug.
- **Locale and encoding.** String sorting, number formatting, case-folding, and date parsing all read the locale. A test that passes in `en_US.UTF-8` can fail in `tr_TR` because Turkish lowercases `I` to a dotless `ı` (the famous Turkish-i bug).
- **Hash-map iteration order.** In many languages the iteration order of a hash set or map is unspecified and, in Python since 3.3, randomized per-process via `PYTHONHASHSEED` for security. Code that accidentally depends on iteration order is fine 999 runs out of 1000 and then isn't.
- **Uninitialized memory.** In C and C++, reading an uninitialized local or heap byte is undefined behavior; the value is "whatever was there," which depends on prior allocations, so the program can work until an unrelated change shifts the heap layout.
- **Environment variables, the working directory, and filesystem state.** `PATH`, `LANG`, `TZ`, `HOME`, feature-flag env vars, the current directory a relative path resolves against, a leftover temp file from a previous run — all inputs, all usually invisible.
- **Network and IO timing.** A response that arrives in 5 ms locally and 50 ms in CI can flip a race between a request completing and a timeout firing.
- **Dependency versions.** A floating version range (`^1.2.0`) means "your machine and CI may resolve different patch releases," which is the entire "works on my machine" genre.

There is one more mechanism worth making concrete, because it is the most surprising to engineers who haven't met it: **uninitialized memory** in C and C++. When you write `int x;` and read `x` before assigning it, the standard says the value is indeterminate — you get whatever bytes were already sitting at that stack slot or heap block. Those bytes are the residue of whatever the program did *before*, which depends on the exact sequence of prior allocations and frees, which depends on the input, the timing, and the allocator's internal state. So a program reading uninitialized memory can compute the "right" answer for months — because the residual bytes happened to be zero, or happened to be a valid pointer — and then a completely unrelated change shifts the heap layout by sixteen bytes and the residue becomes garbage and the program crashes. The bug was *always there*; the crash is new only because the layout changed. This is why "it started crashing after I added an unrelated feature" is such a classic and maddening report: the feature didn't introduce the bug, it moved the furniture so the pre-existing bug finally tripped. The reproduction lever here is MemorySanitizer (`-fsanitize=memory`), which flags the read of an uninitialized value *at the read*, deterministically, instead of waiting for the layout to make it crash.

The diagnostic mindset that follows from the mechanism is simple and powerful: **a bug that won't reproduce is a bug whose hidden inputs you haven't pinned yet.** Reproduction is the disciplined act of finding every one of those inputs and nailing it down until only your code is variable. Everything in the next sections is a technique for nailing down one category. The order matters, too: pin the cheap, certain inputs first (seed, clock, locale, env), because each one you pin shrinks the space of remaining explanations, and you want to be left chasing *only* the genuinely hard nondeterminism (the schedule, the network race) rather than wasting a stress loop on a bug that was really just an unpinned timezone.

## 3. Pinning the pinnable: making a run bit-for-bit repeatable

Start with the inputs you can simply *set*. The figure below is the decision you make every time a bug won't reproduce: is the missing input something you can pin (a seed, a clock, an env var), or something you have to force (a schedule, a load condition)? The pinnable branch is the cheap one, so you always try it first.

![Decision tree branching a non-reproducing bug into pinnable hidden inputs versus hidden timing, with a concrete control listed under each leaf](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-3.png)

The discipline is to pin *each* source explicitly rather than hoping it doesn't matter. Here is what that looks like in Python for a test harness, where the same handful of lines closes the most common holes at once.

```python
import os
import random
import numpy as np

# 1. Random seeds — every PRNG you actually use.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# 2. Hash iteration order — must be set in the ENVIRONMENT before the
#    interpreter starts; setting it inside the process is too late.
#    Run with:  PYTHONHASHSEED=0 pytest ...
assert os.environ.get("PYTHONHASHSEED") == "0", "re-run with PYTHONHASHSEED=0"

# 3. Timezone and locale — pin both, do not inherit the host's.
os.environ["TZ"] = "UTC"
import time
time.tzset()
import locale
locale.setlocale(locale.LC_ALL, "C")

# 4. Working directory — resolve paths from the repo root, not "wherever
#    the test happened to be launched."
os.chdir(os.path.dirname(os.path.abspath(__file__)))
```

A few of these have sharp edges worth calling out, because getting them subtly wrong is its own afternoon lost. `PYTHONHASHSEED` genuinely must be in the environment before the interpreter launches; you cannot set it from inside the program, which is why the snippet *asserts* rather than *sets* it. Timezone via `TZ` only takes effect after `time.tzset()` on Unix. And seeding `random` does nothing for code that uses `numpy`'s generator, or `secrets`, or a framework's own RNG — you have to seed every PRNG the code path actually touches, which means knowing what it touches. That last point is the recurring theme: pinning is only as good as your inventory of hidden inputs.

For the clock specifically, setting `TZ` fixes the *zone* but not the *instant*. If the bug depends on "what time is it right now," you need to freeze the instant itself. In Python the cleanest tool is `freezegun`:

```python
from freezegun import freeze_time

@freeze_time("2026-02-28 23:59:59")
def test_month_rollover_does_not_skip_a_day():
    # Now "now" is deterministic. The DST/leap/rollover bug that only
    # fired in the last second of February will fire every single run.
    invoice = generate_invoice_for_today()
    assert invoice.period_end == date(2026, 2, 28)
```

When the code reaches below the language runtime — a C library calling `gettimeofday`, a subprocess reading the wall clock — `freezegun` can't help, because it patches Python's clock functions, not the kernel's. There the tool is `libfaketime`, an `LD_PRELOAD` shim that intercepts the libc time calls for the whole process tree:

```bash
# Freeze the wall clock for the process AND every child it spawns.
FAKETIME="2026-02-28 23:59:59" \
  LD_PRELOAD=/usr/lib/faketime/libfaketime.so.1 \
  ./run_billing_job
```

#### Worked example: the test that only failed in October

A scheduling service had a test that passed all year and failed for two weeks every autumn. The cause was a date-window calculation that used the host's local timezone, and the CI fleet spanned regions that crossed the DST boundary on different days. For about a fortnight, half the fleet was on summer time and half on winter time, and a window computed in local time was off by an hour, which pushed one boundary case across a day line. Nobody could reproduce it in July. The fix to *reproduction* — not to the bug, to the repro — was three lines: pin `TZ=America/New_York`, freeze the clock to the last second before the fall-back transition, and run. Suddenly the test failed every run instead of seasonally. Once it failed reliably, the actual bug (local-time arithmetic where UTC was required) was obvious in ten minutes, and the verification was trivial: the same frozen-clock test went from red to green. The seasonal flake, measured over the next year of CI, was 0 failures. The whole investigation, which had been reopened three autumns running, was finally closed because step zero finally got done.

## 4. The minimal reproducer: shrink until the cause has nowhere to hide

A reproducer that fires reliably but takes 200 lines, three services, and a seeded database is *a* reproducer. It is not a *good* one. The value of a minimal reproducer is that it points at the cause. When you have shrunk a failing scenario down to five lines, those five lines *are* the bug's neighborhood — there is almost nowhere left for the cause to hide. Shrinking is not busywork; it is the act of deleting everything that is not the bug until only the bug remains.

The systematic way to shrink is **delta debugging**, formalized by Andreas Zeller as the `ddmin` algorithm. The idea is mechanical and beautiful: you have a failing input and a test that returns "still fails" or "no longer fails." You try removing chunks of the input. If the smaller input still fails, you keep it and recurse; if it stops failing, you put that chunk back and try a different one. You bisect the *input* the same way `git bisect` bisects *history* — halving, testing, halving again. The figure below shows the narrowing in action: a one-megabyte input that crashes a parser, halved and halved until twelve bytes remain that still crash it.

![Timeline showing a failing input shrinking from one megabyte through several halving steps down to a twelve-byte minimal case that still fails](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-4.png)

You can do this by hand and it is genuinely effective. The manual version is just disciplined deletion: cut the input in half, re-run the test, and ask "does it still fail?" If yes, throw away the half you cut. If no, restore it and cut the other half. For code rather than data, `git stash` and the editor are your shrinkers — comment out a block, re-run, and keep going. The trap to avoid is removing two things at once and concluding the wrong one mattered; change one thing per step so each result is attributable. The `ddmin` algorithm is exactly this discipline made systematic, with the refinement that it falls back to smaller chunks when halving stops making progress.

For text and data inputs, though, you should reach for an automated shrinker, because it never gets bored and never makes an attribution error. `creduce` (and its newer cousin `cvise`) is the tool for *source-code* inputs — it is how compiler engineers turn a 5,000-line file that crashes the compiler into the 8 lines that actually trigger it. You give it the failing file and a script that exits 0 when the file still triggers the bug, and it relentlessly deletes, simplifies, and renames until nothing more can be removed:

```bash
# interesting.sh exits 0 iff the file still reproduces the compiler crash.
cat > interesting.sh <<'EOF'
#!/usr/bin/env bash
# Compile and grep for the specific internal-compiler-error signature.
clang -c bug.c -o /dev/null 2>&1 | grep -q "internal compiler error"
EOF
chmod +x interesting.sh

# creduce shrinks bug.c in place, re-running interesting.sh after every edit.
creduce ./interesting.sh bug.c
# Result: bug.c shrinks from ~5,000 lines to ~10 that still crash the compiler.
```

The discipline that makes `creduce` safe is the "interestingness" test: it must check for the *specific* failure, not just *any* failure. If your script merely checks "does compilation fail," `creduce` will happily reduce your file to something that fails for an entirely different, trivial reason — a syntax error — and present you a "minimal repro" of the wrong bug. Grep for the exact error signature. This is the single most common way people misuse a shrinker, and it wastes a whole reduction run.

Why is a minimal repro worth this much effort? Because *size is search space*. A 200-line failing scenario has 200 lines where the cause could live, plus all the interactions between them; a 5-line one has five. Every line you delete that doesn't change the failure is a line you have *proven* irrelevant to the bug — you have done a tiny experiment and ruled out a suspect. By the time the reduction is done, what remains is, by construction, the smallest set of facts that still triggers the failure, which means every one of those facts is *load-bearing*. The cause is in there with nowhere to hide. This is also why a minimal repro so often *is* the diagnosis: when the 5,000-line compiler crash reduces to `int a = 1 << 40;`, you don't need to read the compiler's source to know it's a shift-amount overflow — the minimal case names the bug. Shrinking is not a chore you do after understanding the bug; it is frequently *how you come to understand it*.

There is a counterintuitive corner here worth flagging: a minimal repro can change the *category* of bug you think you have. A failure you assumed was about data volume ("it only happens with big payloads") sometimes shrinks to a tiny input, revealing that volume was a red herring and the real trigger was a single malformed field that big payloads merely happened to contain more often. Conversely, a bug that *won't* shrink below a certain size is telling you something real — the size itself is part of the trigger (a buffer that overflows only past N bytes, a hash table that only rehashes past a load factor). Either way, the shrink result is information. Let it surprise you.

#### Worked example: a 1 MB fuzz input shrunk to 12 bytes

A team fuzzing a binary message parser caught a crash: a 1 MB input from `afl-fuzz` segfaulted the parser. The crash backtrace pointed into a length-prefixed field decoder, but 1 MB of random bytes told them nothing about *which* bytes mattered. They wrote a four-line interestingness test — run the parser on the candidate file, exit 0 if and only if it crashes with the *same* signal at the *same* function (grepped from the AddressSanitizer report, so a different crash wouldn't count) — and handed the file to a reducer. The reduction ran in about ninety seconds and halving by halving collapsed 1 MB to 512 KB to 4 KB to 200 bytes to a final **12 bytes**. Those twelve bytes were a header claiming a field length of `0xFFFFFFFF` followed by two bytes of payload: a classic length-lies-about-payload over-read, the Heartbleed shape in miniature. The 12-byte case didn't just reproduce the crash 100% of the time — it *was* the bug report. The fix (validate the claimed length against the remaining buffer) took one line, and the 12-byte input went straight into the regression test suite. Measured: crash localized from "segfault somewhere in a 1 MB parse" to "the bounds check on line 84 is missing," in under two minutes of automated reduction plus the time to read twelve bytes.

## 5. Property-based testing: let the machine find *and* shrink the input for you

Manual delta debugging shrinks an input you *already have*. Property-based testing goes one better: it *finds* a failing input you didn't have, then shrinks it for you automatically. This is the QuickCheck idea (from Haskell), and in Python it lives in the `hypothesis` library. Instead of writing example-based tests ("for input 5, expect 25"), you state a *property* that should hold for all inputs, and the framework generates hundreds of inputs trying to break it. When it finds a breaker, it shrinks that breaker to the smallest example that still breaks the property — exactly delta debugging, built in.

```python
from hypothesis import given, strategies as st

# The property: serializing then deserializing must round-trip.
@given(st.dictionaries(
    keys=st.text(),
    values=st.one_of(st.integers(), st.text(), st.none()),
))
def test_json_round_trips(payload):
    encoded = my_serializer.dumps(payload)
    decoded = my_serializer.loads(encoded)
    assert decoded == payload
```

Run that and `hypothesis` will throw thousands of dictionaries at your serializer. Suppose it finds that `{"": 0}` — an empty-string key — fails to round-trip because your serializer drops empty keys. It will not just report the giant random dictionary it first found the failure on; it will *shrink* it, deleting keys and simplifying values, until it hands you the minimal `{"": 0}`. That minimal case is a pointer straight at the cause: "empty-string keys." And here is the reproduction superpower — `hypothesis` *records the failing example* in a local database (`.hypothesis/`) and replays it first on the next run. So a property test that found a 0.1%-of-inputs bug yesterday will fail *deterministically* today, because the framework saved the seed and the example. It turns a found-once flake into a fails-every-time repro automatically.

The general principle generalizes beyond `hypothesis`: **save the failing input and the seed.** Whatever found the failure — a fuzzer, a load test, a randomized integration run — the instant it fails, persist the exact input and the exact RNG seed that produced it. That artifact *is* your reproducer. A fuzzer that crashes a parser at 3am is worthless if it doesn't write the crashing input to disk; with the input saved, you have a deterministic repro you can shrink and bisect against tomorrow. Make "on failure, dump the seed and the input" a reflex in every randomized harness you write.

There is a subtle reproducibility hazard inside property testing that catches people, and it is worth naming because it is the same hazard as everywhere else: a property test is only deterministic if the *code under test* is deterministic. If your serializer reads the clock, or iterates a set in hash order, or calls a real network, then `hypothesis` can generate the same input twice and get two different verdicts — and now even your shrinker is unreliable, because shrinking assumes "smaller input that still fails" is a stable judgment. So the same pinning discipline from section 3 applies *inside* the property test: freeze the clock, fix the hashseed, stub the network. Property testing finds inputs; pinning makes the verdict on those inputs repeatable; you need both. When a property test is itself flaky, the bug is almost always that the code reaches for an unpinned hidden input, and that is itself a finding — you've discovered a hidden dependency you didn't know the code had.

A related and very practical idea is the **stateful** or **model-based** property test, where `hypothesis` (via its `RuleBasedStateMachine`) generates not a single input but a *sequence of operations* against your object and checks an invariant after each — pushing and popping a stack, inserting and deleting from a cache, opening and closing connections from a pool. When it finds a sequence that violates the invariant, it shrinks the *sequence*, handing you the minimal series of calls that breaks the thing. This is gold for concurrency-adjacent and state-machine bugs, where the failure is not "input X is bad" but "this *order* of operations corrupts the state" — exactly the order-dependent flakes that plague test suites. The reproducer it hands you is a tiny, deterministic call sequence you can drop straight into a unit test.

## 6. Capturing the intermittent bug: the probability math of looping

Now the hard case: the bug is real, you've pinned every input you can think of, and it *still* only fails sometimes. The thread schedule, the network timing, the GC pause — some residual nondeterminism remains. You cannot pin it, so you must *force* it: run the scenario over and over until it fails, then capture that failure. The first tool is the humble repeat-until-fail loop.

```bash
# Run until the first failure, counting iterations. Stops on first red.
n=0
while ./run_flaky_test; do
  n=$((n + 1))
  printf '\rpassed %d times' "$n"
done
echo
echo "FAILED on iteration $((n + 1))"
```

Before you launch that loop and walk away, do the math, because it tells you whether looping is even a viable strategy. Suppose the bug fires with probability $p$ on each independent run. The probability it does *not* fire on a single run is $1 - p$. The runs are independent, so the probability it does not fire on any of $n$ runs is $(1 - p)^n$, and therefore the probability you see it *at least once* in $n$ runs is:

$$P(\text{seen in } n \text{ runs}) = 1 - (1 - p)^n$$

This single formula is your planning instrument. The expected number of runs to see the bug once is $1/p$. So if a flake fails 0.3% of the time, $p = 0.003$ and you expect to wait about $1/0.003 \approx 333$ runs per sighting. To be 95% confident you'll see it at least once, you solve $1 - (1 - p)^n \ge 0.95$, which gives $n \ge \ln(0.05) / \ln(1 - p) \approx 3 / p \approx 1000$ runs. There's the rule of thumb worth memorizing: **to be about 95% sure of catching a bug with per-run probability $p$, run it roughly $3/p$ times.** For a 0.3% flake, loop a thousand times. For a 0.03% flake, loop ten thousand. If $3/p$ is a number you can't run in a reasonable wall-clock budget, looping alone won't cut it and you need to *raise $p$* — which is what stress does.

Raising $p$ means making the bad interleaving more likely per run. The tools:

- **Parallelism and load.** Run the test under `stress-ng` to saturate CPU, memory bandwidth, and IO, so the scheduler thrashes and threads interleave in ways your quiet machine never shows. `stress-ng --cpu 8 --io 4 --vm 2 --timeout 60s` alongside your loop can take a once-in-2000 flake to once-in-50.
- **CPU pinning and core reduction.** Counterintuitively, *fewer* cores often surfaces races faster, because the threads are forced to time-slice on the same core and the preemption points multiply. `taskset -c 0 ./run_flaky_test` pins everything to one core.
- **Scheduling pressure.** Lowering the test's priority or adding other busy processes increases the chance the scheduler preempts your thread at the unlucky moment.
- **Test-order shuffling.** Many flakes are *order-dependent* — test A leaves global state that breaks test B. Shuffling the order (`pytest -p randomly`, Go's `-shuffle=on`, `go test -count=N`) surfaces these. We'll see exactly this in the next section's worked example.

```bash
# Combine: run the test 2000 times under shuffle and load, stop on first fail.
stress-ng --cpu 8 --timeout 5m & STRESS=$!
for i in $(seq 1 2000); do
  if ! taskset -c 0,1 pytest -p randomly --randomly-seed=$i test_orders.py -q; then
    echo "FAILED at seed $i"; break
  fi
done
kill $STRESS
```

#### Worked example: the flake that failed 6 times in 2,000 runs

A backend team had a test suite with one test, `test_currency_conversion`, that failed on CI maybe once a week and never locally. The error was a wrong exchange rate — the test expected 1.10 and got 1.08. Re-running the job always passed, so for months the team just hit "retry" and moved on, paying the tax of a flaky pipeline. Someone finally decided to do step zero. They ran the single test 2,000 times locally in a loop: 0 failures. That was the clue — a test that fails on CI but never in isolation is almost always *order-dependent*, contaminated by some other test's leftover state. So they re-ran with shuffle: `pytest -p randomly` over the *whole module*, 2,000 times. Now it failed 6 times. Six failures in two thousand runs is $p \approx 0.003$, which squares with "about once a week" given the CI cadence — the probability math checked out, which gave confidence the right thing had been caught. Each of the six failures, they captured the random seed (`--randomly-seed`) that produced it. Replaying any one of those seeds made it fail *every time*: a deterministic repro at last. From there it was fast: with a fixed seed, the failing order put `test_central_bank_update` (which mutated a module-level rate cache and never reset it) right before the conversion test. The fix was a fixture that reset the cache. After it, the same 2,000-run shuffled loop produced 0 failures, and they kept it in CI as a regression guard. Flake rate: 0.3% → 0%, proven over 2,000 runs, not asserted. The before-and-after of that shared-state fix is the shape of nearly every order-dependent flake.

![Before and after comparison showing a test failing six of two thousand runs due to shared module cache, then zero of two thousand after the cache is reset per test](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-7.png)

For the full treatment of flakes — when to fix, when to quarantine, how to keep them from rotting your CI — see the sibling post in this series on the flaky test (find it, fix it, or quarantine it). The reproduction techniques here are the *first half* of that fight; the quarantine policy is the second.

## 7. Record and replay: capturing the exact run that failed

Sometimes a bug fires so rarely, or perturbs so badly when observed, that even a stress loop is painful — and when you finally catch it, you cannot afford to lose it. This is where **record-replay** debugging earns its keep. The tool is `rr` (from Mozilla), and the idea is genuinely magical the first time you use it: `rr` records *every nondeterministic input* to your program — the exact results of every system call, every signal, every thread-scheduling decision, the values that came back from the network and the clock — into a trace. Then it can *replay* that trace deterministically, as many times as you want, reproducing the exact same execution down to the instruction. The flaky run becomes a recording you can rewind.

```bash
# 1. Record the program under rr. Run it in a loop until it crashes;
#    rr saves every run, so the crashing one is captured.
rr record ./run_flaky_test
# ... loops, and one run finally crashes. That trace is saved.

# 2. Replay the EXACT failing run under a gdb-like interface — as many
#    times as you want, identically, every time.
rr replay
```

What you get inside `rr replay` is not just deterministic replay — it is *reverse* debugging. You can run `reverse-continue` and `reverse-step` to run the program *backwards* from the crash to the cause. When a value is wrong, set a hardware watchpoint on its memory and `reverse-continue`; execution runs backward until the instant the value was last written, dropping you exactly at the line that corrupted it. For a use-after-free or a data race, where the crash happens thousands of instructions after the actual bug, this collapses hours of "how did we get here" into seconds. The catch: `rr` needs hardware performance counters (so it is finicky in some VMs and unavailable on some CPUs), it serializes threads onto one core (which itself changes timing and can hide the very race you're chasing — though it reproduces whatever it *did* record), and it carries a recording overhead. But for a reproducible-once bug you cannot afford to lose, recording it and replaying it a hundred times is the difference between solving it today and chasing it for a month.

There is a lighter-weight cousin worth knowing for when `rr` is unavailable: **capture-on-condition logging.** You can't log everything always — the volume drowns you and the logging itself perturbs timing. Instead, instrument the suspect code to dump rich state *only* when a cheap invariant trips. "If the counter is ever less than its previous value, write the full thread-local history and the stack to a file and abort." You run under that armed trap in the loop or in production until it fires once, and the dump *is* your reproducer's seed. It is record-replay for the poor: you don't capture the whole run, just enough state at the moment of failure to reconstruct and rerun the failing case deterministically.

In practice the pattern is an `assert`-plus-dump guard placed at the exact invariant you believe should always hold. The trick is that the guard is *cheap* in the common case (one comparison) and *expensive* only when it fires (dump everything), so you can leave it armed in a hot loop or even in production without paying for it on the happy path:

```python
import json, traceback, time

_prev = {"value": None}

def guard_monotonic(name, value, **context):
    p = _prev["value"]
    if p is not None and value < p:
        # Invariant violated — capture EVERYTHING about this moment, once.
        with open(f"/tmp/repro-{name}-{int(time.time())}.json", "w") as fh:
            json.dump({
                "violation": f"{value} < {p}",
                "context": context,           # the request id, the inputs
                "stack": traceback.format_stack(),
                "seed": context.get("seed"),  # the seed to replay this run
            }, fh, indent=2)
        raise AssertionError(f"{name} went backwards: {value} < {p}")
    _prev["value"] = value
```

When that fires in a 3am loop, the JSON file on disk contains the seed and the inputs — everything you need to reconstruct the failing run deterministically tomorrow morning. The discipline is to capture the *seed and inputs*, not just the stack: a stack tells you *where* it broke, but the seed and inputs are what let you *reproduce* it, which is the whole point.

A note on `rr`'s reach: it shines on single-machine, mostly-CPU-bound programs in C, C++, Rust, and Go. It struggles with programs that lean hard on GPUs, exotic syscalls, or hardware it can't model, and because it serializes threads onto one core, it changes the timing of races — it will faithfully replay *whatever interleaving it recorded*, but the act of recording can make a particular race less likely to occur in the first place. The right division of labor is: use a stress loop (which *increases* the chance of the bad interleaving) to *catch* the bug under `rr record`, then use `rr replay` (which is perfectly deterministic) to *study* it. Catching and studying are different jobs, and `rr` is a studying tool.

| Technique | What it captures | Overhead | Reach | When to reach for it |
| --- | --- | --- | --- | --- |
| Repeat-until-fail loop | Nothing; just forces the bug | None | Anything you can run | First move for any intermittent bug |
| Stress + pinning | Raises failure probability | Low–medium | Races, timing bugs | When raw looping is too slow ($3/p$ too big) |
| `rr record`/`replay` | The entire deterministic execution | Medium recording, replay free | C/C++/Rust/Go on supported CPUs | Reproduce-once bugs you can't lose; reverse debugging |
| Capture-on-condition log | State at the failure moment only | Very low | Anywhere, incl. prod | When you can't run a debugger or `rr` |
| Save seed + input | The exact failing input | Negligible | Fuzzers, property tests, randomized runs | Always, in any randomized harness |

## 8. The mechanism up close: an interleaving you can hold in your hand

Let me pay off the data-race mechanism with a concrete, runnable example, because "the schedule is nondeterministic" stays abstract until you watch a write disappear. Here is the smallest lost-update race, in C, two threads each incrementing a shared counter a million times. If increments were atomic the result would be two million. It won't be.

```c
#include <pthread.h>
#include <stdio.h>

long counter = 0;  // shared, unsynchronized — the bug lives here.

void *bump(void *_) {
    for (int i = 0; i < 1000000; i++)
        counter++;     // read, add, write — three steps, not one.
    return NULL;
}

int main(void) {
    pthread_t a, b;
    pthread_create(&a, NULL, bump, NULL);
    pthread_create(&b, NULL, bump, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    printf("counter = %ld (expected 2000000)\n", counter);
    return 0;
}
```

The mechanism is that `counter++` is *not one operation*. It is three: load `counter` into a register, add one, store it back. With no happens-before edge between the two threads, those three-step sequences interleave freely, and the figure below shows the fatal one: both threads load 0, both add to get 1, both store 1 — two increments collapse into one, and the final value is short. Run this and you'll see something like `counter = 1873402`, a different number every time, which is itself the tell of a race.

![Grid showing two threads each reading a shared counter of zero before either writes one back, so the final value is one instead of two and an increment is lost](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-6.png)

The reproduction problem with this bug is that *whether* you observe a lost update at all, and *how many* you lose, depends on the schedule — so it is a heisenbug. Add a `printf` inside the loop and the timing changes enough that you might "lose" far more or far fewer. This is exactly the kind of bug where pinning the schedule by hand is hopeless and you reach for a tool that does not *depend* on catching the bad interleaving by luck. That tool is **ThreadSanitizer.** Compile with `-fsanitize=thread` and TSan instruments every memory access to track the happens-before relation directly; it reports a race the *first* time two threads access the same location without synchronization, whether or not that particular run happened to lose an update.

```bash
# ThreadSanitizer detects the race deterministically — no luck required.
cc -fsanitize=thread -g race.c -o race && ./race
# WARNING: ThreadSanitizer: data race (pid=12345)
#   Write of size 8 at 0x... by thread T2:
#     #0 bump race.c:7
#   Previous write of size 8 at 0x... by thread T1:
#     #0 bump race.c:7
```

This is the deep reason sanitizers are reproduction tools, not just bug finders: they convert a *probabilistic* symptom (sometimes the value is wrong) into a *deterministic* one (the access pattern is always a race, and TSan always sees it). They move the bug from "fires 0.3% of the time" to "flagged 100% of the time," which is precisely the transformation reproduction is about. The Go equivalent is `go test -race`; for memory-corruption bugs the analog is AddressSanitizer (`-fsanitize=address`), which makes a use-after-free or buffer overflow crash *reliably and immediately at the bad access* instead of *eventually and elsewhere* when the corrupted memory is next touched. A whole class of "won't reproduce" memory bugs becomes "reproduces every run, right at the line" the moment you turn on ASan.

| Sanitizer | Finds | Overhead | Flag |
| --- | --- | --- | --- |
| AddressSanitizer | Use-after-free, buffer overflow, leaks | ~2x CPU, ~3x memory | `-fsanitize=address` |
| ThreadSanitizer | Data races, lock-order issues | ~5-15x CPU | `-fsanitize=thread` (Go: `-race`) |
| UBSan | Integer overflow, bad shifts, UB | Low | `-fsanitize=undefined` |
| MemorySanitizer | Reads of uninitialized memory | ~3x CPU | `-fsanitize=memory` |
| Valgrind memcheck | Same as ASan, no recompile needed | ~20-50x CPU | `valgrind --tool=memcheck` |

## 9. "Works on my machine": reproducing the environment, not just the code

A huge fraction of "can't reproduce" is not in the code at all — it is in the *environment*. The program is identical; the world around it differs. Your machine has Python 3.11.4, CI has 3.11.9; your `node_modules` resolved `left-pad@1.3.0`, the build server resolved `1.3.1`; your `LANG` is `en_US.UTF-8`, the container's is `C`. Reproducing the bug means reproducing the *world*, and the discipline has three layers.

First, **lockfiles.** A floating dependency range is a non-pinned hidden input. `package-lock.json`, `poetry.lock`, `Cargo.lock`, `go.sum`, `requirements.txt` with hashes — these pin the exact resolved versions so your machine and CI install bit-identical trees. "We don't commit the lockfile" is, nine times out of ten, the root cause of a "works on my machine" mystery hiding in plain sight. Commit the lockfile; install with the frozen flag (`npm ci`, `poetry install --no-update`, `pip install --require-hashes`).

Second, **containers.** A `Dockerfile` pins the OS, the system libraries, the locale, the timezone, the user, and the working directory — the whole layer below your dependencies. When a bug "only happens in prod," the first question is "what is different about prod's environment," and the cheapest way to answer it is to run *prod's exact image* locally. If you can `docker run` the same image CI runs and the bug reproduces, you have isolated it to the environment and can diff the two worlds methodically.

Third, **recording the exact environment.** When a bug report comes in, capture the environment alongside it: the output of `pip freeze` / `npm ls`, the relevant env vars, the OS and kernel version, the locale and timezone, the git SHA. A bug report without the environment is half a reproducer. Build the habit (or the tooling) to attach an environment fingerprint to every failure.

```bash
# A minimal environment fingerprint to attach to any "won't reproduce" report.
{
  echo "=== git ==="; git rev-parse HEAD
  echo "=== os ===";  uname -a
  echo "=== locale ==="; locale
  echo "=== tz ==="; cat /etc/timezone 2>/dev/null; date +%Z
  echo "=== python ==="; python --version; pip freeze
  echo "=== env (filtered) ==="; env | grep -E '^(LANG|LC_|TZ|PATH|PYTHONHASHSEED|NODE_ENV)='
} > env_fingerprint.txt
```

There is a sharp edge in container-based reproduction worth a warning, because it bites teams who think a `Dockerfile` alone makes them safe. A `Dockerfile` that says `FROM python:3.11` does *not* pin the world — `python:3.11` is a moving tag that points at a different patch release and a different base OS month to month. The same `docker build` run today and in three months can produce different images, which means a bug that reproduced in the image you built in March may *not* reproduce in the image you build in June, even though the `Dockerfile` is byte-identical. The fix is to pin by *digest* (`FROM python:3.11@sha256:...`) and to pin the apt/apk package versions inside, so the image is genuinely reproducible. The same "floating reference" trap that bites dependency ranges bites base-image tags; pin them the same way.

The deeper lesson is that "reproducible build" and "reproducible bug" are the same discipline pointed at different goals. A team that can rebuild a bit-identical artifact from a SHA can also reproduce any bug that artifact ever had, because nothing about the run is unpinned. The matrix below collects the common offenders, the symptom each shows, and the one control that pins it — the lookup table you scan when a bug refuses to reproduce. For the broader container-and-environment story, the [system-design treatment of building reproducible, observable systems](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) goes further than I will here; the reproduction-specific point is just: pin the world, not only the code.

![Matrix mapping each source of nondeterminism to its symptom, the concrete control that pins it, and the repeatable result you get](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-5.png)

## 10. Reproducing the bug that only exists in production

The boss level: a bug that fires in prod and refuses to reproduce anywhere else. You cannot attach a debugger to the payments process serving real traffic. You cannot run a stress loop against live customer data. And yet the bug is *there*, in the dashboard, mocking you. The strategy is to drag the failing case down onto a machine you control by capturing what prod had that you don't: the *real request*, the *real data shape*, and the *real environment*. The figure shows the pipeline.

![Stack showing the path to reproduce a production bug locally by capturing the real request, sanitizing it, building a fixture, and replaying it until it fails](/imgs/blogs/reproduce-it-first-or-youre-not-debugging-8.png)

The first move is **capture the failing input.** When the error fires in prod, your logging and tracing should give you the request that triggered it — the URL, headers, body, and the correlation/trace ID that ties together every service that touched it. (If it *doesn't*, that gap is your first fix, and it's why the series keeps pointing at observability: you cannot reproduce what you didn't record. The [distributed-tracing and observability material in the microservices series](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) is the place to build that capability.) With the exact failing request in hand, you replay it locally against the same code and, very often, it fails immediately — because the bug was never about *load* or *timing*, it was about *that specific input shape* you never tested: the Unicode name, the empty array, the 4097-character field, the null where you assumed non-null.

The second move handles the cases where one request isn't enough — where the bug needs *production-shaped data* in the database, not just in the request. Here you build **prod-data-shaped fixtures**: a sanitized snapshot that preserves the *shape* of prod (the cardinalities, the edge-case rows, the weird legacy records) while stripping the PII (the real names, emails, card numbers). Sanitization that preserves shape is the hard part — replace the values, keep the structure: the row that has a null `region`, the customer with 10,000 orders, the product with an emoji in its name. Those shape edge-cases are usually where the bug lives, and a fixture of clean synthetic data will never reproduce them.

The heavier artillery, when single-request replay isn't enough, is **traffic replay and shadow traffic.** Traffic replay records a slice of real production requests (via a proxy like `goreplay`, or from access logs) and replays them against a staging copy of the service, letting you reproduce bugs that only emerge from the *mix* and *sequence* of real traffic. Shadow traffic (or "mirroring") goes further: it tees live production requests to a new version of the service in parallel with the real one, so the candidate sees real load and real data shapes without serving any user — you watch whether it reproduces the bug or the fix holds, against the actual firehose, with zero customer risk.

A few hard-won cautions about prod reproduction, because this is where teams either move fast or burn a week. First, **the trace ID is the thread that unravels everything** — if your services don't propagate a correlation ID through every hop, your first investment isn't reproducing *this* bug, it's adding the ID so you can reproduce *every* future bug. A failing request you can't tie back to the exact inputs across services is a half-captured reproducer. Second, **sanitization must preserve shape or it's worthless.** Replacing a real customer's 10,000-order history with a synthetic 3-order customer doesn't reproduce the pagination bug that only fires past 8,192 rows; you have to keep the *cardinality* and the *edge-case structure* even as you scrub the *values*. The art is a transformation that is value-destroying and shape-preserving at once. Third, **a prod-only bug that resists single-request replay is telling you the cause is stateful** — it depends on what's already in the database, the cache, or the in-memory state of a long-lived process — and that narrows your hypothesis usefully: stop looking at the request handler and start looking at accumulated state. The failure to reproduce from a single request is itself a clue about *where* the bug lives.

One more discipline that pays for itself: once you *do* reproduce a prod bug locally, **freeze that fixture into the test suite immediately**, before you even fix the bug. The temptation is to fix first and add the test later "if there's time," and there is never time. The fixture that reproduces the bug today is the regression test that prevents it tomorrow, and the marginal cost of saving it while it's in your hands is near zero. This is the seam where reproduction and prevention become literally the same artifact — the captured failing input, checked in, asserting the correct behavior.

#### Worked example: the 500 that only fired for one customer's catalog

Back to the checkout `NullPointerException` from the intro. The dashboard said 0.4% of checkouts. Local runs were clean. The breakthrough was the trace ID: every failing request, pulled from the logs, belonged to merchants whose catalog had a product with no `category` set — a state the UI prevented but a legacy bulk-import had allowed for a few thousand old products. The serializer assumed `category` was always present and dereferenced it. Once they had *one* failing request captured, they built a fixture with a single category-less product, replayed the checkout locally, and it failed 100% of the time. That is the whole game: the bug went from "0.4%, can't reproduce, no idea" to "fails every run on this fixture" the moment the real input shape was captured. The fix (a null-safe default category) was then *provable* — red on the fixture before, green after — and they added the category-less product to the test fixtures permanently so the regression could never return silently. Note what made it tractable: not cleverness, but capturing the one real input the local tests never had. Reproduction was 90% of the work, and the actual code change was four lines.

## 11. War story: famous bugs that were really reproduction problems

It is worth looking at how this plays out at the scale where it makes the news, because the pattern is always the same: the bug was hard not because the *fix* was hard but because *reproducing it on demand* was hard.

**The Therac-25 race condition (1985–87).** The Therac-25 radiation therapy machine delivered massive overdoses to several patients, some fatally. At the heart of it was a data race: if a skilled operator typed the treatment parameters *fast enough*, a concurrent task that should have set up the hardware lost a race with the task that fired the beam, and the machine delivered a high-energy beam without the protective target in place. It was nearly impossible to reproduce because it depended on operator timing — only fast, experienced operators triggered it, and only sometimes. The manufacturer initially could not reproduce it at all and dismissed reports. The bug was a heisenbug in the most literal and tragic sense: the window was a fraction of a second of interleaving. The lesson that hardened into safety-critical practice was exactly the one in this post — a bug you cannot reproduce is a bug you cannot claim to have fixed, and "we couldn't reproduce it" is not the same as "it isn't there." (Concurrency that only fails under specific timing is covered in this series' race/deadlock posts; the reproduction angle is the entry point.)

**Heartbleed (2014).** The OpenSSL Heartbleed bug was a buffer over-read: a request could claim a payload length longer than it actually sent, and the server would copy that many bytes back out of memory, leaking whatever happened to be adjacent — private keys, session data, passwords. The mechanism was a missing bounds check, the kind of thing the C type system does nothing to prevent. What made it so dangerous in the wild and so reproducible *once understood* is that it was deterministic given a crafted request: send a heartbeat with a lying length, get memory back. The reproduction lesson here is the flip side of Therac-25 — once researchers crafted the specific malicious *input*, the bug reproduced 100% of the time, which is exactly why a minimal reproducer (here, a hostile heartbeat packet) is so powerful. The "fuzzing" that finds such bugs is property-based testing pointed at security: generate millions of inputs, save the one that leaks. AddressSanitizer flags this over-read instantly on the crafted input — a probabilistic-looking memory bug turned deterministic by the right tool.

**The Knight Capital deploy (2012).** Knight Capital lost about \$440 million in 45 minutes because a deployment left old, repurposed code (a flag that used to mean one thing now meant another) running on one of eight servers. The bug was not really in the code logic — it was that the *environment* differed across servers: seven had the new code, one had the old. It is the "works on my machine" failure mode at catastrophic scale and speed. The reproduction lesson is brutal and clear: an inconsistent environment across hosts is a hidden, unpinned input, and a bug that only fires on the one out-of-date server is exactly the kind that "can't be reproduced" on the other seven. Pin the world — the same artifact, the same flags, the same everything, on every host — or you are running an uncontrolled experiment in production. For the deploy-consistency side of that, the [microservices material on deployment strategies and safe rollouts](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) is the place to go.

Across all three, the through-line is the thesis of this post. The Therac-25 team couldn't reproduce a timing race and so denied it existed. Heartbleed became tractable the instant someone built the deterministic crafting input. Knight Capital was an unpinned environment difference that "worked on seven machines." Reproduction was the whole battle.

## 12. How to reach for this (and when not to)

Reproduction is step zero, but the *techniques* of reproduction have costs, and a senior engineer knows when each is worth it. Here is the honest guidance.

**Always do the cheap pinning first.** Before you reach for `rr` or `stress-ng`, set the seed, freeze the clock, fix `PYTHONHASHSEED`, pin the timezone and locale, and run in the project's container. This costs minutes and resolves a large fraction of "won't reproduce" outright. It is pure upside; there is never a reason to skip it.

**Reach for the repeat-until-fail loop the moment a bug is "intermittent."** It is free and it gives you the data you need — the failure rate — which feeds the $3/p$ math that tells you whether looping alone is viable. If $3/p$ is more runs than you can afford, escalate to stress and pinning to raise $p$. Don't sit and re-run a flake by hand ten times and conclude anything; that's far too few samples to estimate a rate.

**Reach for `rr` when a bug is reproduce-once and you cannot afford to lose it,** or when the cause is far from the crash and you need reverse debugging. Don't reach for `rr` as your *first* move on a flake you can already reproduce 1 in 50 — the recording overhead and the single-core serialization aren't free, and the serialization can even hide the race. It's a precision tool for the bug you've already cornered, not a dragnet.

**Reach for shrinking when the reproducer is large.** A 5-line repro is worth an hour of `creduce` or manual delta debugging, because it collapses the search space for *every* later step. But don't gold-plate: if your repro is already small and obvious, shrinking it further is wasted motion. Shrink to point at the cause, then stop.

**Don't attach a debugger to a critical prod process.** Attaching `gdb` to the live payments service freezes it; an `rr` recording in prod has overhead you may not be able to spare. For prod, prefer the non-invasive path: capture the input, build a fixture, and reproduce *off* the production host. Shadow traffic and capture-on-condition logging exist precisely so you don't have to break into the running system.

**Don't chase a heisenbug at `-O2` first.** If a memory bug only reproduces in an optimized release build, your first move is to try to reproduce it at `-O0` with sanitizers on, where the behavior is closer to the source and the tools work best. Reproduce in the easiest environment that still shows the bug, not the hardest. Only if it *won't* reproduce at `-O0` (because the bug is in optimizer-introduced behavior or relies on a specific layout) do you stay at `-O2` and reach for harder tools.

**Know when "good enough" is good enough.** The goal of reproduction is a verdict you can trust before and after a fix. If you have a repro that fails reliably enough to prove the fix — say, fails 100% on a fixture, or fails predictably at a known seed — you are done reproducing; go fix the bug. Reproduction is in service of the fix, not an end in itself. The failure mode of theory-minded engineers is polishing a perfect minimal repro long after they had enough to proceed.

## 13. Key takeaways

- **Reproduction is step zero.** A bug you cannot reproduce, you cannot hypothesize about, cannot bisect, and cannot prove you fixed. Every later technique presupposes it. Build the repro first.
- **A non-reproducing bug is an unpinned hidden input.** The run is the code *plus* the seed, the clock, the locale, the env, the schedule, the dependency versions, the working directory. Find the unpinned one and pin it.
- **Pin the pinnable explicitly.** Seed every PRNG you use; freeze the clock and zone; set `PYTHONHASHSEED`; fix the locale and CWD; commit the lockfile; run in the container. Don't hope an input doesn't matter — nail it down.
- **Shrink to point at the cause.** Delta debugging (`ddmin`, `creduce`, property-test shrinking) reduces a sprawling failure to the minimal case. Five lines that fail are a pointer at the bug; always grep for the *specific* failure in your interestingness test.
- **Force what you can't pin, and do the math.** For a bug with per-run probability $p$, expect $1/p$ runs per sighting and loop about $3/p$ times for 95% confidence. If that's too many, raise $p$ with stress, pinning, and shuffle.
- **Sanitizers turn probabilistic bugs deterministic.** TSan flags a race the first run regardless of whether an update was lost; ASan crashes a use-after-free reliably at the bad access. They move a bug from 0.3% to 100% reproducible.
- **Record what you can't afford to lose.** `rr record`/`replay` captures the exact failing execution for deterministic, reversible debugging; capture-on-condition logging and saved seeds are the lighter-weight versions. Always dump the seed and input in any randomized harness.
- **Reproduce the environment, not just the code.** Most "works on my machine" bugs are unpinned worlds — lockfiles, containers, and recorded env fingerprints close the gap.
- **Drag prod bugs onto your laptop.** Capture the real request and trace ID, build a sanitized prod-shaped fixture, replay it; escalate to traffic replay or shadow traffic when the mix matters. Don't debug in the live process.
- **A repro is also a regression test.** The reproducer that fails before your fix and passes after is, with almost no extra work, the test that prevents the bug from ever silently returning. Reproduction and prevention are the same artifact.

## 14. Further reading

- *Why Programs Fail: A Guide to Systematic Debugging* by Andreas Zeller — the canonical text on delta debugging, `ddmin`, and turning debugging into a science. The reproduction-and-shrinking chapters are the source for much of this post.
- *Debugging: The 9 Indispensable Rules* by David J. Agans — rule 3 is literally "Quit Thinking and Look," and the book hammers the reproduce-it-first discipline in plain language.
- The `creduce` / `cvise` documentation and the `rr` project ([rr-project.org](https://rr-project.org/)) — the official guides to automated reduction and record-replay, including the reverse-debugging workflow.
- The Hypothesis documentation (property-based testing and automatic shrinking) and the original QuickCheck paper — for letting the machine find *and* shrink failing inputs.
- The AddressSanitizer and ThreadSanitizer wikis (the Clang/LLVM sanitizers documentation) — for turning probabilistic memory and concurrency bugs into deterministic, reproduce-every-run failures.
- The series intro, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), for the full observe → reproduce → hypothesize → bisect → fix → prevent loop this post serves; and the capstone debugging playbook (`capstone-the-debugging-playbook`, once shipped) for how reproduction fits the whole craft.
- The sibling post on the flaky test — find it, fix it, or quarantine it — for the second half of the intermittent-bug fight: policy, quarantine, and keeping flakes from rotting your CI once you've learned to reproduce them.

---
title: "The Flaky Test: Find It, Fix It, or Quarantine It"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to treat a flaky test as a real bug — reproduce it on demand with repeat-and-shuffle loops, bisect it to the test that poisoned shared state, fix the nondeterminism for good, and quarantine only as a time-boxed last resort."
tags:
  [
    "debugging",
    "software-engineering",
    "flaky-tests",
    "testing",
    "nondeterminism",
    "ci-cd",
    "test-isolation",
    "race-conditions",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-1.png"
---

There is a test in your suite that passes nineteen times and fails the twentieth. You know the one. Someone opened a pull request, CI went red, and the failure was in a test that has nothing to do with their change — a test about user-profile serialization, while they touched the billing module. They click "re-run failed jobs." It goes green. They merge. Nobody investigates, because everybody already knows: *that test is flaky*. It fails sometimes. It's just noise.

This is one of the most expensive lies a team tells itself, and it compounds quietly. The first time a test flakes, an engineer loses ten minutes and a little trust. By the hundredth flake, the team has built a reflex: red build, hit re-run, don't read the error. And the day the suite catches a *real* regression — a genuine bug that would have shipped to customers — the build goes red, somebody hits re-run out of habit, it passes on the retry because the bug is intermittent too, and the broken code sails into production. A flaky suite does not merely waste time. It trains your team to ignore the one signal a test suite exists to send. The figure below frames the whole argument: a flaky test is not random noise to be suppressed, it is a real bug — in the test or in the code under test — that is reporting a real nondeterminism, and it belongs in the same observe-to-prevent loop you would run for any other defect.

![A vertical stack showing the six stage debugging loop applied to a flaky test, from observing that it passes nineteen times and fails the twentieth, through reproducing it a thousand times, to preventing recurrence with a flake dashboard](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-1.png)

By the end of this post you will be able to do four concrete things. You will be able to *reproduce* a flake on demand instead of waiting for it to ambush you — turning a 4% intermittent failure into a test that fails 1 in 25 runs every time you ask it to. You will be able to *classify* the flake into one of a small number of root-cause families, each with a specific tell you can probe for. You will be able to *fix* the nondeterminism at the root — fake the clock, poll for a condition instead of sleeping, isolate shared state, mock the network — so the test passes 2,000 times in a row. And you will know exactly when to *quarantine* a test instead of fixing it, why that decision must be time-boxed with a ticket and an owner, and why the seductive shortcut of "just retry failed tests in CI" is a slow-motion disaster that hides real product bugs. This is the same spine the rest of this series runs on: observe, reproduce, hypothesize, bisect, fix, prevent. We are just pointing it at the test suite. If you have not read the [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) that opens this series, that is the map; this is one hard country on it.

## 1. A flake is a bug, not a weather condition

Let us start by killing the worst metaphor in software testing: the idea that flakiness is *weather*. People talk about flaky tests the way they talk about rain — an external condition that happens to the suite, unpredictable, nobody's fault, wait-it-out. That framing is comforting and completely wrong. A computer is a deterministic machine. Given identical inputs and identical state, a correct test produces an identical result every single time. If the same test sometimes passes and sometimes fails, then *the inputs were not identical* — something differed between the passing run and the failing run, and that something is hidden from you. The flake is the machine telling you, in the only language it has, that there is a variable you have not accounted for.

That variable lives in exactly one of two places, and it matters enormously which. Either the nondeterminism is in the *test* — the test is racing the system, asserting before an async operation finishes, depending on an iteration order that is not guaranteed, leaking state from a previous test — in which case the production code is fine and the test is buggy. Or the nondeterminism is in the *code under test* — there is a genuine race condition, a real time-of-check-to-time-of-use window, an actual ordering bug in your product — in which case the test is doing its job perfectly and has just caught a bug that will also bite real users, intermittently, in production. The single most dangerous habit around flaky tests is treating both cases as "the test is flaky" and silencing them the same way. You will silence real product bugs. You will ship them.

So the first discipline is a mindset shift: when a test flakes, you do not say "that test is flaky." You say "that test is *intermittently failing*, which means there is hidden nondeterminism, and I am going to find it." The word "flaky" is fine as shorthand, but never let it carry the implication of *unfixable* or *not my problem*. Every flake has a root cause. Every root cause is, in principle, findable. Most are findable in an afternoon once you know where to look.

Why does this matter so much in practice? Because of how trust in a test suite decays. A suite's entire value is its *signal*: red means stop, green means go. That signal only works if it is reliable. Suppose your suite of 5,000 tests has a per-test flake rate of just 0.02% — two failures in ten thousand runs, which sounds negligible. The probability that a full suite run is *clean* is $0.9998^{5000} \approx 0.37$. That means roughly **63% of your CI runs will show at least one spurious red** even though every test is individually almost-always-passing. A flake rate that looks like a rounding error at the level of one test becomes a near-certainty at the level of a suite. This is the math behind why large suites rot: flakiness does not add, it multiplies across thousands of tests, and the suite-level reliability collapses long before any single test looks broken.

> The cruel part of that arithmetic is that it gets worse exactly as you succeed. The more tests you write — the better your coverage — the larger the exponent, and the more a tiny per-test flake rate dominates your CI experience. A team that invests heavily in testing and *ignores* flakiness will end up with a suite nobody trusts, which is worse than a small suite everybody trusts. Flakiness is not a cosmetic problem you fix when you have time. It is a tax on the entire investment.

Throughout this post I will use one running bug as the spine: a test called `test_active_user_count` that asserts the system reports exactly one active user after a fresh signup. It passes when you run it alone. It passes most of the time in CI. And about once every twenty-five suite runs it fails with `AssertionError: expected 1, got 2`. We are going to reproduce it, classify it, bisect it to the real culprit, fix it for good, and verify the fix — and along the way we will tour every other family of flake you are likely to meet.

## 2. The root-cause taxonomy: where flakes actually live

Before you can hunt a flake efficiently, you need a map of where they hide. Over years of chasing these, a small taxonomy emerges, and almost every flake you will ever meet belongs to one branch of it. Knowing the branches turns a blind hunt into a checklist. The taxonomy below organizes flakes into three families — leaked state, time and async, and environment — because those three describe *what kind of hidden variable* differed between the passing run and the failing run.

![A taxonomy tree branching a flaky test into three families of hidden state, leaked state with order dependence and unfixed seeds, time and async with sleep then assert, and environment with real network calls](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-2.png)

Let me walk each branch, because the *mechanism* — the reason each one is even possible — is what tells you how to spot it.

**Order dependence.** This is the test that passes alone and fails after some other test ran first. The mechanism is shared mutable state that one test mutates and the next test reads without resetting: a global variable, a module-level singleton, a row in a database that is not rolled back, a temp file left on disk, a cached connection, a memoized value, a registered handler. Test A leaves a footprint; test B trips over it. The defining tell is the one that gives the whole game away: **it passes in isolation, it fails in company.** If `pytest path::test_b` is green but `pytest` (the full suite) is red on `test_b`, you are almost certainly looking at order dependence. The reason this is so common is that test frameworks run tests *in the same process* by default for speed, so any process-level state — a Python module's globals, a class attribute, a connection pool — is shared across every test in that process unless you explicitly tear it down.

**Shared mutable state / test pollution.** This is order dependence's broader parent: any time one test leaks state that affects another. It includes order dependence but also subtler forms — a test that mutates a shared fixture object that other tests in the same file also use, a test that bumps a global counter, a test that monkeypatches a function and forgets to undo it, a test that writes to a cache the next test reads. The mechanism is the same (state outlives the test that created it), but the fix differs: the cure is *isolation and reset*, making each test start from a known clean state and leave no trace.

**Time.** Tests that depend on the wall clock are a bottomless well of flakes, and the mechanisms are wonderfully varied. A `sleep(0.1)` followed by an assertion is a *bet* that the background work finishes within 100 milliseconds — a bet that holds on your idle laptop and loses on a CI box running eight jobs at once. A test that captures `now()` at the top and compares against `now()` later can straddle a second boundary. A test that builds a date from "today" breaks the one day a year it runs on December 31 in a timezone where "tomorrow" is already a different year. A test that hardcodes an expectation about February has a 1-in-1461 chance of running on February 29. A test that assumes the local timezone breaks in CI configured for UTC, or during the one hour a year that does not exist because of a daylight-saving spring-forward. Every one of these is a real, deterministic bug — it fails *exactly* when the clock is in a particular state — that merely *looks* random because the clock is usually not in that state.

**Async / race.** This is the test that asserts on the result of an asynchronous operation before that operation has actually completed. The mechanism is missing synchronization: the test kicks off work on another thread, in another process, on an event loop, or over the network, and then checks the result without a proper *happens-before* relationship guaranteeing the work is done. Sometimes it is a test bug (the test should await or poll). Sometimes it is a real concurrency bug *in the product* — a genuine data race or ordering bug — and the test is correctly flagging it. The line between these is exactly the subject of [race conditions, the hardest bugs to catch](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch); a flaky async test is very often the surface symptom of the race that post dissects.

**Network / external dependencies.** A test that makes a real HTTP call, resolves real DNS, binds a real port, or talks to a real container is at the mercy of everything outside your process. The mechanism is that these resources are *shared and stateful* in ways your test does not control: a port may already be in use by a leftover process, a dependency container may not be ready when your test starts, a DNS lookup may be slow, an external API may rate-limit you, a network blip may drop a packet. None of this is in your code, but all of it surfaces as an intermittent test failure.

**Resource / fixture order.** This family covers nondeterminism in things you forgot are nondeterministic: an unseeded random number generator producing a different value each run, a hash map whose iteration order is not specified (Python randomizes string hashing per process by default for security), a floating-point comparison that is exactly equal on one machine and off by a bit on another, a `set` you iterate over expecting a particular order. The mechanism is that these orderings are *implementation-defined or randomized*, and a test that asserts on a specific order is asserting on something the language never promised to keep stable.

**Infrastructure.** Finally, the flake that is genuinely not your test's fault and not your code's fault: the CI box was overloaded, a noisy neighbor stole the CPU, the test timed out because the machine was thrashing, the disk filled up, the Docker daemon hiccuped. These are real, and they are the *only* category where "it's the environment" is a legitimate diagnosis — but you must earn that diagnosis by ruling out the others first, because "infrastructure" is also the lazy excuse people reach for to avoid investigating a real order-dependence bug.

Here is the taxonomy as a lookup table you can keep next to your keyboard, mapping each cause to its tell and its fix. We will earn every row of it in the sections that follow.

| Root cause | Tell (how it shows up) | How to spot it | Deterministic fix |
| --- | --- | --- | --- |
| Order dependence | Passes alone, fails in suite | Run isolated; shuffle order | Proper teardown / fixtures |
| Test pollution | Fails only after a specific sibling | `--forked`; bisect siblings | Reset shared state per test |
| Sleep-then-assert | Fails under CI load | Re-run on a loaded box | Poll for the condition |
| Real network call | Fails when DNS/API is slow | Run with network disabled | Mock the boundary |
| Unfixed RNG seed | Output differs each run | Diff two run outputs | Pin the seed |
| Hash iteration order | Order varies run to run | Vary `PYTHONHASHSEED` | Sort, or assert order-free |
| Wall-clock dependence | Fails near midnight / Feb 29 | Set `TZ`, fast-forward clock | Fake the clock |
| Async race in product | Fails under parallelism | `-race` / TSan, stress | Fix the synchronization |
| Infrastructure | Random timeouts, no pattern | Rule out all the above first | Right-size CI; retry only here |

The discipline this table encodes is simple: **diagnosis points at the repair.** If you can name which row you are in, you already know the fix. The hard part — and the rest of this post — is earning the right to name the row, which means *reproducing the flake on demand* so you can test your hypothesis instead of guessing.

## 3. Reproducing the flake: the genuinely hard part

Everything downstream depends on reproduction. You cannot confirm a hypothesis about a flake you cannot make fail; you cannot verify a fix for a failure you cannot trigger. This is the same lesson as [reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging), and it bites hardest exactly here, because a flake is *defined* by being hard to reproduce. The whole craft is converting "fails 1 in 25 runs, somewhere, sometime" into "fails 1 in 25 runs, right here, every time I ask."

There is real probability behind why naive reproduction fails. If a flake fails with probability $p$ per run, the chance you catch at least one failure in $n$ runs is $1 - (1-p)^n$. For a 4% flake, a single run catches it only 4% of the time, which is why "I ran it and it passed, must be fixed" is worthless. To be 99% confident of catching a 4% flake you need $n = \lceil \ln(0.01) / \ln(0.96) \rceil = 113$ runs. To be 99% confident of catching a *0.1%* flake you need about 4,600 runs. This is the core reason flakes are hard: the failure is rare enough that you must *amplify your sample size by orders of magnitude* before the bug is forced to show itself. Reproduction is not about one clever run; it is about running enough times, in adversarial enough conditions, that the rare event becomes common.

So the first tool is brute repetition. In Python with `pytest`, the `pytest-repeat` plugin gives you `--count`:

```bash
# Run one test 1000 times in a single invocation; stop on the first failure.
pytest tests/test_users.py::test_active_user_count --count=1000 -x -q

# If it passes 1000x in isolation but flakes in CI, the bug is NOT in this
# test alone -- it is order dependence or pollution. That negative result
# is itself a strong clue: isolate the variable that changed.
```

In Go, repetition is built in — and so is the race detector, which you should reach for reflexively on any concurrency-adjacent flake:

```bash
# -count disables the test cache and forces N real runs; -race instruments
# every memory access to catch data races; -failfast stops at the first red.
go test -run TestActiveUserCount -count=100 -race -failfast ./...

# Shuffle test order with a fixed, printable seed so a failure is replayable.
go test -shuffle=on -count=20 -race ./...
# On failure Go prints: -test.shuffle 1700000000 -- re-run with that exact
# seed to reproduce the same order deterministically.
```

In JavaScript with Jest, you can lean on `--runInBand` plus a repeat loop, or use the `jest-repeat`/`--testNamePattern` combination, but the single most useful Jest flag for flakes is the one that controls isolation, which we will get to. For raw repetition a shell loop is honest and portable:

```bash
# Run the JS suite until it fails, counting iterations. Works for any runner.
i=0
while npx jest test/users.test.js --silent; do
  i=$((i+1))
  echo "pass #$i"
done
echo "FAILED on run #$((i+1))"
```

Repetition alone, though, only catches flakes whose cause is *internal* to the test — a race the test loses some fraction of the time. It will never catch order dependence, because running the same test alone a thousand times never introduces the sibling test that poisons the well. For that you need the second tool: **randomized order**. The figure below shows the full toolkit; no single tactic catches every flake, so you stack them.

![A graph showing four reproduction tactics, repeat with high count, shuffle order, fork isolation, and stress load, all converging on a flaky test becoming reproducible on demand](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-4.png)

`pytest-randomly` shuffles test order every run and, crucially, *prints the seed it used* so you can replay an exact order:

```bash
# Shuffle order; the seed is printed at the top of the run.
pytest -p randomly -q
# Output begins with, e.g.:
#   Using --randomly-seed=1551376399

# A failure under shuffle that passes under default order = ORDER DEPENDENCE.
# Replay the exact failing order with that seed:
pytest -p randomly --randomly-seed=1551376399 -q
```

The third tool is **isolation**, which both reproduces *and* diagnoses pollution. The `pytest-forked` plugin (or `pytest-xdist`'s `--forked`) runs each test in its own forked subprocess, so process-level state cannot leak between tests:

```bash
# Each test gets a fresh process. If a test that flakes in the shared
# process suddenly passes 1000/1000 when forked, the bug is PROCESS-LEVEL
# POLLUTION -- a global, a singleton, a module cache leaking across tests.
pytest --forked --count=1000 tests/test_users.py::test_active_user_count
```

That contrast — flaky in-process, rock-solid forked — is a *diagnosis*, not just a workaround. It tells you the nondeterminism lives in shared process state, which narrows your hunt enormously. (Do not *ship* `--forked` as the fix; forking every test is slow and merely hides the leak. Use it to localize, then fix the leak.)

The fourth tool is **stress**. Many flakes are timing-sensitive and only appear when the machine is under load, because that is when a `sleep`-based wait becomes too short or a scheduler interleaving becomes likely. Reproduce CI's overloaded box on your quiet laptop with `stress-ng`:

```bash
# Saturate 8 CPUs and add memory pressure in the background, THEN run the
# suspect test in a loop. A timing flake that needs a busy box now appears.
stress-ng --cpu 8 --vm 2 --vm-bytes 1G --timeout 120s &
pytest tests/test_users.py::test_active_user_count --count=500 -x
```

The fifth tool, when all else fails, is to reproduce *in CI's exact environment* — same container image, same timezone, same parallelism — and, for the truly nasty ones, **record-replay**. A tool like `rr` records a failing run with enough fidelity to replay it deterministically, so you can attach a debugger to the *exact* failing execution and step backward through it. That is heavy machinery; reach for it only when a flake resists everything above, and lean on the [heisenbugs that vanish when you look](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look) techniques for the ones that change behavior the moment you observe them.

#### Worked example: forcing a 4% flake to fail 1 in 25, on demand

Our running bug, `test_active_user_count`, fails about 4% of the time in CI. Step one: run it alone, 1,000 times. `pytest ...::test_active_user_count --count=1000 -x` — **1000 passed**. Zero failures in isolation. That negative result is gold: the bug is *not* internal to this test. It needs a sibling. Step two: run the whole suite under shuffle, repeatedly. `pytest -p randomly --count=40 -q`. On the 9th suite run it fails: `AssertionError: expected 1, got 2`, and the header reads `Using --randomly-seed=1551376399`. Step three: replay that exact seed — `pytest -p randomly --randomly-seed=1551376399` — and it fails *every single time*. We have converted a 4% intermittent failure into a 100%-reproducible one, pinned to a specific test order. The probability math says we should expect to wait roughly $1/0.04 = 25$ runs on average to see one failure; we got lucky at run 9, but the replayable seed means we never have to be lucky again. Reproduction: done. Now we can hunt the culprit.

## 4. Bisecting to the test that poisoned the well

We know the flake is order-dependent — it needs a specific sibling test to run first. But our suite has 5,000 tests. Which one is poisoning the well? You do not read all 5,000. You **bisect**, exactly as you would bisect a regression in git history (see [binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) for the general technique). The insight is identical: you have a known-bad ordering and you want the *minimal* set of tests that, run before your victim, reproduces the failure. Binary search finds it in $\log_2 N$ steps — for 5,000 tests, about 13 splits instead of 5,000 reads.

The mechanics: take the failing order (you have the seed). Run the first half of the tests before your victim; if it still fails, the culprit is in that half, so recurse into it; if it passes, the culprit is in the other half. `pytest` makes this tractable because you can pass an explicit list of test node IDs in an explicit order. A small script automates the search:

```bash
#!/usr/bin/env bash
# bisect_flake.sh -- find the minimal sibling set that triggers the flake.
# Usage: feed it the ordered list of node IDs from the failing seed, with the
# victim test last. It binary-searches the prefix that still reproduces.

VICTIM="tests/test_users.py::test_active_user_count"
# all_tests.txt: the failing order (from --randomly-seed), victim removed.
mapfile -t TESTS < all_tests.txt

reproduces() {                 # returns 0 (true) if the flake still fails
  pytest "$@" "$VICTIM" -q -p no:randomly >/dev/null 2>&1
  [ $? -ne 0 ]                 # non-zero exit = test failed = reproduced
}

lo=0; hi=${#TESTS[@]}
while (( hi - lo > 1 )); do
  mid=$(( (lo + hi) / 2 ))
  if reproduces "${TESTS[@]:0:mid}"; then
    hi=$mid                    # culprit is within the first $mid tests
  else
    lo=$mid                    # culprit is in the second half
  fi
done
echo "Culprit test: ${TESTS[$lo]}"
```

This is the same `git bisect run` pattern — a binary predicate (does it reproduce?) driving a logarithmic search — pointed at *test order* instead of *commit history*. When it terminates, you have a single guilty sibling. In our case it prints `tests/test_signup.py::test_signup_creates_user`. Now the picture is clear: `test_signup_creates_user` inserts a user row into the shared test database and never deletes it. When `test_active_user_count` runs afterward in the same database, it sees that leftover row *plus* the one it created, counts two, and fails its assertion of one. The grid below shows the exact interleaving — A inserts and forgets to clean up, B counts and finds one row too many.

![A grid showing two test columns where test A inserts a user row with no teardown so the row persists, and test B then counts two rows instead of one and its assertion fails](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-3.png)

Notice *why* this was intermittent in the first place rather than a constant failure. Under the default test order, `test_active_user_count` happened to run *before* `test_signup_creates_user` most of the time, so the leaked row was not there yet. Only when the random shuffle put signup first did the pollution land. That is the signature of order dependence: the bug is fully deterministic *given an order*, but the order itself is the hidden variable. The mechanism — a shared database whose rows outlive the test that created them, because the test framework wraps each test in a transaction it forgot to roll back, or no transaction at all — is exactly the kind of process-level (here, database-level) shared mutable state from our taxonomy.

#### Worked example: the bisection by the numbers

The failing seed produced an order of 5,000 tests with our victim somewhere in the middle; the relevant prefix before the victim was 2,048 tests. The bisection script ran the binary predicate on prefixes of size 1024, then 512 (still reproduced — culprit in first half), then 256, 128, 64, 32, 16, 8, 4, 2, and finally 1. Eleven evaluations — $\lceil \log_2 2048 \rceil = 11$ — each costing one suite-subset run of a few seconds, total under two minutes. Compare that to reading 2,048 tests by hand. The culprit was `test_signup_creates_user`, confirmed because running *only* that test before the victim reproduced the failure 100%, and running the victim with *no* prefix passed 100%. We have not just found a suspect; we have a minimal, deterministic reproducer: two tests, one order, fails every time. That is the standard of proof. Anything less and you are guessing.

## 5. Fixing the determinism: each cause has one repair

Now that we can name the cause, the fix is mechanical — and that is the whole payoff of the taxonomy. The figure below lays out the cause-to-fix mapping for the families we have met; let me then walk the actual code for the most important ones.

![A matrix mapping five flake root causes to their tell, how to spot them, and the one deterministic fix for each, from order dependence cured by teardown to wall clock dependence cured by faking the clock](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-5.png)

**Order dependence and pollution → isolate and reset.** Our database flake is cured by making every test start from a clean database and leave no trace. The right tool is a transactional fixture that rolls back after each test, or an explicit truncate in teardown. In `pytest`:

```python
import pytest
from myapp.db import engine, Session

@pytest.fixture(autouse=True)
def clean_db():
    """Wrap every test in a transaction and roll it back afterward.
    Nothing a test writes survives into the next test -- pollution is
    impossible by construction, not by everyone remembering to clean up."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session                       # the test runs here
    session.close()
    transaction.rollback()              # undo EVERYTHING the test did
    connection.close()
```

The `autouse=True` is the load-bearing word: it applies the reset to *every* test without anyone having to remember. The deeper principle is to make correct isolation the *default* and pollution the thing you have to opt into, rather than the reverse. A test suite where isolation depends on each author remembering to clean up is a suite that will rot, because someone always forgets. After adding this fixture, `test_signup_creates_user` can insert all the rows it wants; the rollback erases them before the next test sees the database. The leak is gone at the root.

The same isolation principle applies in every ecosystem, and the failure modes are identical. In Jest, the trap is shared module-level state: a module imported by two test files keeps its top-level variables across tests in the same worker, so a counter or a cache bumped by one test bleeds into the next. The fix is `jest.resetModules()` in a `beforeEach`, combined with running tests in isolated workers, and resetting any global you mutate:

```js
// Reset module registry and globals before every test so no state leaks.
beforeEach(() => {
  jest.resetModules();        // fresh module instances; no shared globals
  jest.clearAllMocks();       // mock call counts reset to zero
  global.__cache = undefined; // explicitly clear anything you mutate
});

// Run files in isolated workers so a crash or leak can't poison siblings.
// jest --maxWorkers=50% keeps isolation while staying parallel.
```

In Go, the analogous discipline is to avoid package-level mutable state entirely (a `var users = map[...]` at package scope is shared across every test in the package) and to use `t.Cleanup` to register teardown that runs even if the test fails partway through. The cross-language rule is one sentence: *no test may depend on, or leave behind, state that another test can see.* Where that rule is enforced by construction — transactional rollback, module reset, no package globals — order dependence becomes impossible rather than merely discouraged.

**Time → fake the clock.** Never let a test read the real wall clock. Inject a clock you control. In Python, `freezegun` (or `time-machine`) freezes time to a fixed instant:

```python
from freezegun import freeze_time
import datetime

@freeze_time("2026-02-28 23:59:59")
def test_subscription_does_not_expire_early():
    # Time is frozen. "Now" is always this instant, on every machine, in
    # every timezone, on Feb 29, at midnight -- the test can never straddle
    # a second boundary or break on a leap day, because there is no real
    # clock to straddle.
    sub = Subscription(expires=datetime.datetime(2026, 3, 1))
    assert sub.is_active() is True
```

In Go, the same discipline means never calling `time.Now()` directly in code under test — inject a `Clock` interface and pass a fake in tests. In JavaScript, `jest.useFakeTimers()` plus `jest.setSystemTime()` does the job. The mechanism you are defeating is that the real clock is a *global, ever-changing input* your test does not control; faking it converts time from a hidden variable into a pinned one.

**Async → poll for the condition, never sleep.** This is the single most common and most fixable flake in modern code, so it gets its own section below. The short version: `sleep(0.1)` is a bet on how long async work takes; replace it with a loop that *polls for the actual condition* with a generous timeout, so the test waits exactly as long as needed and no longer.

**Network → mock the boundary.** A unit or integration test should not make real HTTP calls. Mock at the network boundary so the test is deterministic and offline:

```python
import responses

@responses.activate
def test_fetches_user_profile():
    # Register a canned response. No real DNS, no real socket, no rate
    # limit, no port conflict, no slow API -- the network is replaced by a
    # deterministic stub, so the test can only fail if YOUR code is wrong.
    responses.add(
        responses.GET, "https://api.example.com/users/42",
        json={"id": 42, "name": "Ada"}, status=200,
    )
    profile = fetch_user_profile(42)
    assert profile.name == "Ada"
```

For tests that genuinely *must* exercise a real service (a small set of true end-to-end tests), the fix is not mocking but *readiness gating*: wait for the dependency to be healthy before the test runs (poll its health endpoint), bind to an ephemeral port (`:0`) instead of a fixed one to avoid conflicts, and give it a real, bounded retry with backoff — the one place retry is legitimate, which we will revisit.

Two specific network flakes are worth calling out because they masquerade as "the test is flaky" when they are really "the test setup is racy." The first is the *port conflict*: a test binds a hardcoded port like `8080`, a previous test (or a leftover process from a crashed run) still holds it, and `bind()` fails with `EADDRINUSE`. The fix is to never hardcode a port — bind `127.0.0.1:0`, let the OS assign a free port, and read back the actual port the kernel chose. The second is the *container-not-ready* flake: your integration test starts a database container and immediately connects, but the database process inside the container has not finished initializing, so the first connection is refused. The fix is to poll the dependency's readiness (a `SELECT 1`, a health endpoint, a TCP connect with retry) until it answers, with a bounded timeout, before the test body runs:

```python
import socket, time

def wait_for_port(host, port, timeout=30.0):
    """Block until a TCP connect succeeds or timeout -- the right way to
    gate on a dependency being ready, instead of sleeping and hoping."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return                      # dependency is accepting connections
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"{host}:{port} not ready within {timeout}s")
```

Both fixes are the same idea as the async poll from section 6: replace a *guess* about timing (a fixed port, an immediate connect) with a *check* of the actual condition (a free port, a ready service). Almost every "environment" flake that is not pure infrastructure dissolves once you stop assuming and start polling.

**Unfixed seed / hash order → pin or assert order-free.** Set `PYTHONHASHSEED=0` and seed every RNG explicitly in a fixture; or, better, write assertions that do not depend on order at all (compare sets, or sort before comparing). A test that asserts `list(d.keys()) == ["a", "b", "c"]` is asserting on an order Python never promised; `sorted(d.keys()) == ["a", "b", "c"]` is deterministic.

The proof standard for *any* of these fixes is the same and it is non-negotiable: re-run the formerly-flaky test under the same adversarial conditions that reproduced it — the same shuffle seed, the same stress, a high count — and show it now passes a large number of times in a row. "I added a fixture and it passed once" proves nothing; you already know a 4% flake passes 96% of the time. The bar is 2,000 clean runs under shuffle, or whatever count gives you 99% confidence given the original flake rate.

#### Worked example: the database flake, measured before and after

Before the fix, `test_active_user_count` failed 4% of the time in CI — concretely, across an instrumented 2,000-run sweep under `pytest-randomly`, it failed **81 of 2,000** times (4.05%), every failure being `expected 1, got 2`. We added the `autouse` transactional-rollback fixture. We then re-ran the *exact same* 2,000-run sweep, same seeds, same shuffle: **0 of 2,000** failures. We also ran the minimal two-test reproducer (`test_signup_creates_user` then `test_active_user_count`) 5,000 times: **0 failures**. The flake rate went from 4.05% to 0%, measured under the conditions that originally exposed it, not under a friendly single run. That is the difference between "I think I fixed it" and "I proved I fixed it." The whole investigation — reproduce, bisect, fix, verify — took an afternoon, and the timeline below is the shape of it.

![A timeline of deflaking one database test from a four percent failure rate on day zero, through reproducing it with shuffle, bisecting to the sibling that leaks a row, fixing with teardown, to verifying zero failures in two thousand runs](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-8.png)

## 6. The sleep-then-assert flake, and the poll that kills it

The async timing flake deserves its own treatment because it is so common, so often "fixed" wrongly, and so cleanly fixable when you understand the mechanism. Here is the archetype, and you have written it:

```python
def test_job_completes():
    job = submit_background_job()
    time.sleep(0.1)                    # "give it time to finish"
    assert job.status() == "done"      # flaky: fails when 0.1s isn't enough
```

The mechanism of the flake is a missing *happens-before* relationship. The test does work A (submit the job) and then asserts on B (the job is done), but there is nothing guaranteeing B has happened when the assertion runs — except the hope that 100 milliseconds is "enough." On your idle laptop, the background worker picks up the job in 5 ms and finishes in 30 ms, comfortably inside the sleep, and the test passes. On a CI box running eight parallel jobs, the worker thread does not even get scheduled for 150 ms, the assertion fires while the job is still queued, and the test fails. Same code, same test, different *scheduling* — the hidden variable is how the OS scheduler shared the CPU, which depends on machine load, which is exactly why the failure correlates with "CI is busy."

The naive instinct is to *increase the sleep*: bump `0.1` to `1.0`. This is wrong on two counts. First, it does not eliminate the flake — it just lowers its probability, because there is no sleep duration that is *guaranteed* to be enough; a sufficiently overloaded box can blow past any fixed timeout. Second, it makes every run slower for everyone, including the 99.99% of runs where the job finished in 30 ms but the test sits there sleeping for a full second anyway. A suite of 500 such tests with one-second sleeps wastes more than eight minutes per run doing nothing. You have traded a flake for a tax and not even removed the flake.

The correct fix is to *poll for the actual condition* with a generous timeout, so the test waits precisely as long as the work takes and no longer, and only fails if the work genuinely never completes:

```python
import time

def wait_until(predicate, timeout=5.0, interval=0.01):
    """Poll predicate() until it is truthy or timeout elapses.
    Returns on success; raises with a clear message on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")

def test_job_completes():
    job = submit_background_job()
    wait_until(lambda: job.status() == "done", timeout=5.0)
    assert job.status() == "done"      # now deterministic: we KNOW it's done
```

Three properties make this correct where the sleep was wrong. It is *fast*: when the job finishes in 30 ms, the poll notices within one 10 ms interval and the test proceeds — no wasted second. It is *robust*: even if the CI box is so loaded the job takes 3 seconds, the test still passes, because the timeout is a generous upper bound, not a precise guess. And it *fails meaningfully*: if the job genuinely never completes — a real bug — the test fails after 5 seconds with a clear message, instead of failing instantly because of a scheduling hiccup. Note the use of `time.monotonic()` rather than `time.time()`; the monotonic clock cannot jump backward if the system clock is adjusted mid-test (an NTP step), which is itself a subtle source of timing flakes.

Most ecosystems ship a battle-tested version of `wait_until` so you do not roll your own. Python has the `tenacity` library and the `pytest`-friendly `wait_for` helpers; the JavaScript world has `@testing-library`'s `waitFor`; Go programs use `require.Eventually` from testify:

```go
// testify's Eventually polls the condition every `tick` until `waitFor`
// elapses. Deterministic and self-documenting -- no magic sleep.
func TestJobCompletes(t *testing.T) {
    job := submitBackgroundJob()
    require.Eventually(t, func() bool {
        return job.Status() == "done"
    }, 5*time.Second, 10*time.Millisecond, "job never completed")
}
```

The figure below contrasts the two approaches as a before-and-after: a fixed sleep that fails under load versus a poll that waits exactly as long as needed.

![A before and after figure contrasting a fixed one tenth second sleep that fails eighty of two thousand times under load against a poll every ten milliseconds with a five second timeout that fails zero of two thousand times](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-6.png)

One caution, and it ties back to the taxonomy. Polling fixes the case where the test was racing the system but the system is *correct* — the work really does complete, the test just asserted too early. Polling does **not** fix a real race *in the product*. If `job.status()` is itself reading a field that two threads write without synchronization, then no amount of polling in the test makes the underlying data race go away; you have a genuine concurrency bug that the flaky test correctly surfaced. The way you tell these apart is to run the system-under-test with a race detector — `go test -race`, or ThreadSanitizer for C/C++/Rust — and see whether it reports a data race. If it does, your bug is in the product, and you fix the synchronization, not the test. This is precisely the boundary where the flaky-test investigation hands off to the race-condition investigation; treat a polling fix that *still* flakes as a strong signal you are in the latter case.

#### Worked example: the async flake that only failed on CI

A team had `test_webhook_delivered` failing about 1.5% of the time, *only* in CI, never locally. The body was the classic `dispatch_webhook(); time.sleep(0.05); assert delivery.received`. Locally, the in-process dispatcher delivered in under 5 ms, so the 50 ms sleep was wildly sufficient. In CI, tests ran with `pytest-xdist -n 8`, so eight test processes fought over the CPU, and the dispatcher thread sometimes did not run for 80–120 ms — past the 50 ms sleep. Instrumenting with a sweep of 2,000 CI-equivalent runs (eight parallel workers, `stress-ng` in the background) reproduced it at **31 of 2,000** (1.55%). The fix was a four-line change to `wait_until(lambda: delivery.received, timeout=3.0)`. Re-running the identical 2,000-run loaded sweep: **0 of 2,000**, and median test duration *dropped* from 50 ms to 7 ms because the test no longer sleeps when delivery is fast. Faster *and* deterministic — the usual outcome when you replace a sleep with a poll.

## 7. Fix it, quarantine it, or delete it — the decision

You have found and can reproduce a flake. Now the judgment call: do you fix it now, quarantine it, or delete it outright? This is where teams go wrong most often, because the easy options (ignore, or silently skip) are the ones that destroy the suite over time. The decision is driven by two axes — *how important is this test* and *how fast can you fix it* — and only one combination ever justifies a quarantine. The figure below is the decision.

![A decision tree for a found flaky test that splits on whether the test is important, with important tests either fixed now or time-boxed quarantined and low value tests deleted](/imgs/blogs/the-flaky-test-find-it-fix-it-or-quarantine-it-7.png)

**Default: fix it now.** If the test guards real behavior and you can fix the determinism in an afternoon — which, for the order-dependence and sleep-then-assert cases above, you can — then fix it now. This is the right answer far more often than people admit, because most flakes are not mysterious; they are a forgotten teardown or a sleep, and the fix is small once you have reproduced the failure. The reason people skip this is that they never reproduced it, so it *feels* mysterious. Reproduction collapses the mystery, and a reproduced flake is usually a quick fix.

**Delete it if it is flaky *and* low-value.** Not every test deserves to exist. A test that is both unreliable and weak — it asserts something trivial, or it overlaps another test, or it tests an implementation detail nobody cares about — should be *deleted*, not fixed and not quarantined. Keeping a low-value flaky test costs you twice: it produces false alarms *and* it is not even buying you meaningful coverage in exchange. Be honest about this. A surprising fraction of flaky tests, examined coldly, turn out to be tests that should never have been written — they were testing the framework, or a getter, or a coincidence. Delete them and feel nothing. The suite gets faster and more trustworthy in one commit.

**Quarantine only as a *time-boxed* stopgap.** Sometimes a test is important — it guards real risk, you do not want to lose its coverage — but the root cause is genuinely hard (a deep race in a third-party dependency, an infrastructure issue you cannot fix today) and you cannot block the entire team's merges while you investigate. *Then*, and only then, you quarantine: you mark the test so it no longer fails the build, but **with three non-negotiable conditions**. First, a *ticket* tracking the quarantine, so it is visible work, not a forgotten skip. Second, an *owner* who is responsible for de-quarantining it. Third, a *deadline* — a time box — after which the quarantine is reviewed and the test is either fixed or deleted. A quarantine without these three is just a permanent skip with extra steps, and a permanent skip is how coverage silently erodes until the test that mattered was disabled two years ago and nobody noticed.

The mechanics of quarantine should make the cost *visible*. In `pytest`, a custom marker plus a ticket reference in the reason:

```python
import pytest

@pytest.mark.flaky_quarantine(
    ticket="JIRA-4821", owner="payments-team", since="2026-06-20")
@pytest.mark.skip(reason="QUARANTINED JIRA-4821: races in vendor SDK; "
                         "owner payments-team; review by 2026-07-20")
def test_payment_reconciliation():
    ...
```

Then a CI job greps for `flaky_quarantine` markers and *fails the build* if any quarantine is older than its review date, or if the count of quarantined tests exceeds a budget. The quarantine list becomes a debt you must pay down, not a junk drawer you can fill forever. Some teams put the quarantine count on a dashboard next to the flake rate, precisely so that "we have 40 quarantined tests" is an embarrassing, visible number rather than an invisible rot.

What you must **never** do is silently skip a test — comment it out, add a bare `@skip` with no reason, or worse, weaken the assertion until it stops failing (the dreaded `assert result is not None` replacing `assert result == expected`). Every one of these *looks* like a fix and is actually a coverage deletion in disguise, with the added insult that the test still exists, still runs, and still gives a false sense of safety. A deleted test is honest: coverage is gone and everyone can see it. A silently-defanged test is a lie that costs you the day a real bug walks through the hole it left.

Here is the disposition as a table, because the decision should be boring and repeatable, not agonized over each time:

| Situation | Importance | Time to fix | Disposition |
| --- | --- | --- | --- |
| Forgotten teardown / sleep | High | Hours | **Fix now** |
| Real product race surfaced | High | Days | Fix now (it's a real bug) |
| Hard vendor/infra cause | High | Unknown | **Quarantine** + ticket + owner + deadline |
| Trivial / duplicate / impl-detail | Low | Any | **Delete it** |
| Asserts a coincidence | Low | Any | Delete it |
| "We might need it someday" | Low | — | Delete it (git remembers) |

## 8. Why "just retry it" is a trap

Every CI system offers, and every tired team eventually reaches for, automatic retry: if a test fails, re-run it; if it passes on the second try, call the build green. `pytest-rerunfailures` gives you `--reruns 2`. Jest has `jest.retryTimes(3)`. GitHub Actions and most CI platforms have a "retry failed jobs" button that becomes muscle memory. It feels like a pragmatic accommodation of reality: tests *are* sometimes flaky, so why not just retry and move on?

Because retry-on-failure does not fix flakes — it *hides real, intermittent product bugs*, which is strictly worse than a flaky test. Walk the logic. The whole premise of "retry and accept the pass" is that an intermittent failure followed by a pass means the failure was spurious. But that premise is exactly backward for the most dangerous bugs you have. If your product has a genuine race condition that corrupts data 2% of the time, a test that exercises it will *also* fail about 2% of the time — and pass on retry 98% of the time. Auto-retry will dutifully mark that build green every single time, and you will ship a data-corruption bug that hits 2% of your real users, having configured your CI to specifically suppress the one signal that was warning you. Retry-on-pass treats "the bug only happens sometimes" as evidence the bug is not real, when "the bug only happens sometimes" is the *definition* of the most expensive bugs in production.

There is a second, slower harm: retry hides the *flake rate itself*, so the suite rots invisibly. If every flake is silently retried into a green, nobody sees the flakiness accumulating; the per-test flake rate climbs, the number of retries per build creeps up, builds get slower (every retry is a full re-run of a test), and one day you have a suite that needs three retry rounds to go green and takes 40 minutes, and you have no idea which tests are responsible because the retries erased the evidence. Auto-retry is a sedative, not a cure — it numbs the pain that was telling you to investigate.

This does not mean retries are *never* legitimate. There is exactly one honest use: retrying a genuinely *external, non-deterministic dependency* with a bounded backoff, in a true end-to-end test, where the flakiness is in the network or a third-party service you do not control and the retry models what your *production* code would also do. Even then, the retry belongs at the boundary (retry the HTTP call), not at the test level (re-run the whole test), and it should be logged so the rate is visible. The distinction is whether the retry is *modeling real resilience your product has* or *suppressing a signal you should be reading*. Retrying the whole test to turn red into green is the latter; it is the trap.

The honest alternative is **detection without suppression**: run the test once, let it fail the build if it fails, but *also* feed every failure into a flake-detection system that watches for the rerun-and-pass signal and tracks it. The point is not to make the red go away silently — the point is to *make the flake visible and assigned* so it gets fixed. That is the subject of the next section.

| Approach | What it does to a real intermittent bug | What it does to the flake rate | Verdict |
| --- | --- | --- | --- |
| Auto-retry, accept pass | Hides it — ships to prod | Hides it — suite rots invisibly | **Trap** |
| Manual "re-run" button | Hides it — trains the reflex | Hides it — no tracking | **Trap** |
| Bounded retry at external boundary | Models real resilience | Logged, visible | OK in true E2E only |
| Run once, fail build, track flake | Surfaces it for investigation | Measured on a dashboard | **Correct** |

## 9. Detecting and tracking flakes in CI before they spread

You cannot fix what you cannot see, and the failure mode of every team that "doesn't have a flaky-test problem" is that they have one and cannot measure it. The mature move is to instrument CI so that flakes are *detected automatically*, *attributed* to specific tests, and *tracked* as a rate over time — turning flakiness from a vibe ("CI's been annoying lately") into a number you can drive down.

The core detection signal is **rerun-and-pass**: a test that fails on one run of a given commit and passes on another run of *the same commit* is, by definition, flaky — the code did not change, so the difference is nondeterminism. You harvest this signal without using it to suppress anything. Many CI systems and test frameworks support recording results to a database keyed by test ID and commit SHA; a small job then flags any test that produced both a pass and a fail for the same SHA. The output is a ranked list of your flakiest tests by flake rate, which is exactly the worklist you want.

A minimal home-grown version: have your test runner emit JUnit XML (every runner does — `pytest --junitxml`, `go test` via `gotestsum`, Jest via `jest-junit`), push each run's results into a small store, and compute per-test flake rate as failures-on-otherwise-green-commits over total runs:

```python
# flake_report.py -- crude but honest flake-rate computation.
# Reads JUnit XML results tagged with the commit SHA; a test that both
# passed and failed for the SAME sha is flaky for that commit.
import collections, glob
from junitparser import JUnitXml

results = collections.defaultdict(lambda: collections.defaultdict(set))
for path in glob.glob("ci-results/*.xml"):
    sha = path.split("/")[-1].split("_")[0]      # encode sha in filename
    for suite in JUnitXml.fromfile(path):
        for case in suite:
            outcome = "fail" if case.result else "pass"
            results[case.name][sha].add(outcome)

print(f"{'test':50s} flake_rate")
for name, by_sha in sorted(results.items()):
    flaky_shas = sum(1 for o in by_sha.values() if o == {"pass", "fail"})
    rate = flaky_shas / max(len(by_sha), 1)
    if rate > 0:
        print(f"{name:50s} {rate:6.1%}")
```

With that list, you set policy. A flake-rate *budget*: the suite's aggregate flake rate must stay under, say, 0.1%, and a PR that pushes it over is blocked or flagged. A *quarantine dashboard*: the count of currently-quarantined tests, with their tickets, owners, and review dates, visible to the whole team so the debt cannot hide. An *auto-quarantine with auto-ticket* flow for the worst offenders, so a test that crosses a flakiness threshold is automatically isolated *and* a ticket is filed with an owner — visible, tracked, time-boxed, never silent. The big engineering organizations that run millions of tests a day all converged on roughly this shape, because at that scale a flake rate of even 0.1% per test makes the suite statistically impossible to ever see fully green (recall the multiplicative math from section 1), so they had no choice but to measure and budget flakiness like any other reliability metric.

The deeper point is cultural, and it is the prevention half of the debugging loop. A team that *measures* its flake rate treats a new flake as a regression to be investigated, the same as a latency regression or an error-rate spike. A team that does not measure it treats each flake as a one-off annoyance to be re-run away, and the rate climbs until the suite is worthless. The instrumentation is not the hard part — a hundred lines of script and a dashboard. The hard part is the commitment to look at the number and act on it. This is exactly the observability discipline that the broader [observability for debugging production](/blog/software-development/debugging/observability-for-debugging-prod) post argues for, applied to your test suite instead of your services: you cannot manage what you do not measure.

## 10. War stories: when intermittent meant catastrophic

It is tempting to think of flaky tests as a developer-experience nuisance — annoying, time-wasting, but ultimately cosmetic. The history of software says otherwise. The exact same class of nondeterminism that makes a test flaky — a race, a timing assumption, an ordering dependency — is, when it lives in production code, the cause of some of the most expensive failures in computing. A flaky test is often the *first, cheapest warning* of a bug that, ignored, becomes a catastrophe. Three cases make the point.

**The Therac-25 race condition.** The Therac-25 was a radiation therapy machine that, between 1985 and 1987, delivered massive radiation overdoses to at least six patients, several fatally. The root cause was a race condition: if an operator entered the treatment parameters and then edited them *quickly* — faster than the software expected — a concurrency bug between the data-entry task and the treatment-setup task left the machine in an inconsistent state, firing a high-power beam without the protective target in place. The bug was intermittent. It depended on operator typing speed, which is exactly the kind of timing variable that makes a bug "only happen sometimes." Experienced operators who typed fast triggered it; the manufacturer initially could not reproduce it because *they* typed slowly. If that race had surfaced as a flaky test during development — a test that "sometimes failed when the setup ran in an unexpected order" — the correct response would have been to treat the intermittency as a real bug and hunt the race, not to re-run until green. The lesson of Therac-25 is the thesis of this post written in the most serious ink possible: *intermittent does not mean unreal; it means timing-dependent, and timing-dependent bugs are still bugs.*

**The leap-second cascades.** On several occasions — notably mid-2012 — the insertion of a leap second (a one-time adjustment that makes a minute 61 seconds long) caused widespread outages: servers locked up, services hung, datacenters saw CPU spikes as the kernel and applications mishandled the impossible `23:59:60` timestamp. The affected code had a *latent* dependency on time being monotonic and well-behaved, an assumption that held every second of every day except the rare leap second. This is precisely the wall-clock flake from our taxonomy, scaled to production: a test that "occasionally fails near midnight" or "breaks on the day they add a leap second" is reporting the same class of bug that took down major services. The fix in tests — fake the clock, never trust the wall clock, test the boundary explicitly (including the leap day and the leap second) — is the same discipline that prevents the production version.

**The Knight Capital deploy.** In 2012, Knight Capital lost roughly \$440 million in 45 minutes because a deployment left old, dormant code active on one of eight servers, and an order flag reactivated it. This was not a race in the classic sense, but it was a *state/ordering* failure of exactly the family that produces order-dependent flakes: behavior that depended on the precise combination of which code was where and what ran first, a combination that was not deterministic across the fleet. A test suite that depended on deploy order, or that passed or failed based on residual state from a previous run, is the small-scale rehearsal of this exact failure mode. The discipline that prevents the flaky test — make every run start from a known, clean, fully-specified state, and never let behavior depend on what happened to run before — is the same discipline that prevents the production catastrophe.

The thread through all three is identical to the thread through this entire post: *a flaky test is a real bug reporting a real nondeterminism.* The systems that failed catastrophically had the same nondeterminism in production that a flaky test would have shown in development — and the difference between a near-miss and a disaster is often whether someone treated the intermittent failure as a signal to investigate or as noise to suppress. Every time you fix a flake instead of re-running it, you are practicing the muscle that, in a more serious system, prevents the headline.

## 11. How to reach for this (and when not to)

Like every tool in this series, the flaky-test playbook has a cost, and a senior engineer knows when *not* to spend it. Here is the honest calculus.

**Reach for full reproduction (1000× repeat, shuffle, stress, fork) when** the flake is recurring, blocks merges, or guards important behavior. The investment — an afternoon to reproduce, bisect, and fix — pays for itself the first time it stops a team of ten from hitting re-run all week. The more a flake costs the team in lost time and lost trust, the more worthwhile the deep hunt.

**Do not chase a flake you cannot reproduce after a genuine, bounded effort** — but be honest about what "genuine effort" means. It means you actually ran it 1,000× under shuffle and stress, not that you ran it twice and gave up. If after a real reproduction campaign the flake still will not appear on demand, *then* quarantine it with a ticket and move on, rather than burning a week on a ghost. The discipline is: reproduce first, and if reproduction genuinely fails after real effort, quarantine honestly rather than pretend.

**Do not "fix" a flake by increasing a sleep or weakening an assertion.** These feel like fixes and are not; they either lower the flake probability without eliminating it (longer sleep) or delete coverage in disguise (weaker assertion). If you find yourself reaching for either, stop — you have not found the root cause yet. The sleep tells you to poll; the failing assertion tells you to investigate, not to soften.

**Do not enable auto-retry-to-green as a policy.** As section 8 argued, it hides real intermittent product bugs and lets the suite rot invisibly. The only legitimate retry is a bounded, logged retry at a true external boundary in a real end-to-end test. Everything else is suppression.

**Do not delete a flaky test that guards real risk just because it is annoying.** Deletion is the right call for *low-value* flaky tests, not important ones. The test that flakes *and* protects your payment reconciliation is a test to fix, not to delete; deleting it because it is inconvenient is how the hole that ships the next disaster gets opened. Use the two-axis decision: importance and fixability, not annoyance.

**Do reach for the race detector early when the flake smells like concurrency.** If a test flakes under parallelism, fails under stress, or involves threads/async, run the system under `-race` or ThreadSanitizer *before* you spend an afternoon assuming it is a test bug. If the detector reports a data race, you have found a real product bug in minutes and the whole investigation shortcuts. The cost of a race-detector run is one slow test invocation; the payoff when it fires is enormous.

**Do escalate from flaky test to the broader investigation when the test is correct.** A flaky test is sometimes the *messenger* for a real race, a real ordering bug, or a real time bug in the product. When you have ruled out test-side causes and the nondeterminism is in the system under test, you are no longer debugging a flaky test — you are debugging a [race condition](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch) or a [heisenbug](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look), and those posts pick up the trail with the heavier tools.

A note on scope: this post is about *finding and fixing* flakes once they exist. The broader discipline of *designing* a test suite that resists flakiness from the start — hermetic test environments, dependency injection of clocks and randomness, isolation by default, the architecture of fast and reliable tests — is the subject of a dedicated testing and quality-engineering series planned for this blog but not yet written. When it lands, it will be the prevention companion to this debugging-focused treatment; for now, the prevention guidance here (isolate by default, inject the clock, mock the boundary, measure the flake rate) is the load-bearing subset.

## 12. Key takeaways

- **A flaky test is a real bug, not weather.** The machine is deterministic; if a test sometimes passes and sometimes fails, a hidden input differed between runs. Find it, do not silence it.
- **Reproduce before you theorize.** A 4% flake passes 96% of the time, so a single passing run proves nothing. Amplify your sample with `--count=1000`, shuffle order, fork isolation, and stress until the failure is on-demand.
- **Classify with the taxonomy.** Almost every flake is leaked state (order dependence, pollution), time/async (sleep-then-assert, clock dependence, races), or environment (network, infrastructure). Naming the family names the fix.
- **Order dependence passes alone and fails in company.** Find the guilty sibling by *bisecting test order*, the same binary search you use on git history — $\log_2 N$ runs, not $N$ reads.
- **Each cause has one deterministic fix.** Reset shared state in teardown; fake the clock; poll for the condition instead of sleeping; mock the network; pin the seed. The diagnosis points straight at the repair.
- **Replace sleeps with polls.** `sleep(0.1)` is a bet that loses under load. A poll with a generous timeout is faster *and* deterministic, and it fails meaningfully when the work genuinely never completes.
- **Prove the fix the way you proved the bug.** Re-run 2,000× under the same shuffle and stress that exposed the flake. "Passed once" is not proof; "0 of 2,000 under shuffle" is.
- **Fix, quarantine, or delete — decide on importance and fixability.** Fix important-and-fixable now; delete flaky-and-low-value; quarantine only important-but-hard, always with a ticket, an owner, and a deadline. Never silently skip or weaken an assertion.
- **Auto-retry-to-green is a trap.** It hides real intermittent product bugs and lets the suite rot invisibly. Run once, fail the build, and *track* the flake rate instead of suppressing it.
- **Measure flakiness like any other reliability metric.** Harvest the rerun-and-pass signal, rank tests by flake rate, budget it, and put the quarantine debt on a dashboard. You cannot drive down a number you do not look at.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the intro map for this whole series and the observe-reproduce-hypothesize-bisect-fix-prevent loop that this post applies to the test suite.
- [Reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — the deep treatment of turning a ghost into a deterministic, one-command reproducer, of which deflaking is a special case.
- [Race conditions: the hardest bugs to catch](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch) — where the flaky async test hands off when the nondeterminism is a real race in the product, with `-race` and ThreadSanitizer.
- [Heisenbugs that vanish when you look](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look) — for the flakes that change behavior the moment you instrument them, and how to pin an observation that perturbs the system.
- [Binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the general bisection technique that section 4 points at test order instead of commit history.
- The `pytest` documentation on fixtures and the `pytest-randomly`, `pytest-repeat`, and `pytest-forked` plugins — the canonical references for isolation, shuffled order, repetition, and per-test process isolation.
- Go's testing package documentation on `-race`, `-shuffle`, and `-count`, and the `testify/require.Eventually` helper — the built-in toolkit for repetition, race detection, and condition polling.
- *Why Programs Fail* by Andreas Zeller — the academic foundation of systematic, hypothesis-driven debugging and delta debugging, the theory under the bisection and minimization in this post.

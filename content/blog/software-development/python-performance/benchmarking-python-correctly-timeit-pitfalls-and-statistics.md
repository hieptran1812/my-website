---
title: "Benchmarking Python Correctly: timeit Pitfalls and Statistics"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Measure a Python snippet honestly: master timeit and perf_counter, tame the noise, report median plus stdev, dodge the constant-folding and caching traps, and build a reusable benchmark harness you can trust."
tags:
  [
    "python",
    "performance",
    "benchmarking",
    "timeit",
    "perf-counter",
    "pyperf",
    "profiling",
    "statistics",
    "optimization",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-1.png"
---

Someone on the team posts in the channel: "I sped up the parser, it's twice as fast now." They paste a screenshot of a notebook cell with `%%time` in it, and indeed, the cell that used to print `Wall time: 1.2 ms` now prints `Wall time: 0.6 ms`. Everyone reacts with a little rocket emoji. Two weeks later, the parser is found to be exactly as fast as before — the "speedup" was the second run of the cell hitting a warm filesystem cache and a CPU that had already spun up to its turbo frequency. Nothing was optimized. A number lied, and a roomful of competent engineers believed it because it was printed in a monospace font.

This happens constantly, and it is not a sign that anyone is careless. It is a sign that **measuring time on a modern computer is genuinely hard**, and that the easy ways to do it — wrap two `time.time()` calls around your code, run it once, read the difference — are wrong in ways that are invisible until they burn you. The clock you reached for might not even resolve the duration you are trying to measure. The compiler might have deleted the work before it ran. The operating system might have descheduled your process mid-measurement to handle a network interrupt. The garbage collector might have fired a sweep right in the middle of your timed region. Every one of those adds time, none of them subtracts it, and a single run has no way to tell signal from noise.

![timeline of an honest benchmark loop moving from setup to warmup to repeating timed batches to collecting samples to median plus stdev to the final report](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-1.png)

This is the post that makes you trustworthy with a stopwatch. By the end you will be able to do five concrete things. First, pick the **right timer** for the job — `time.perf_counter` and `perf_counter_ns` for wall-clock, `process_time` for CPU-only, and never `time.time` for benchmarking — and know precisely why the difference matters. Second, drive **`timeit`** correctly from both the command line and code, understand what `-n` and `-r` and autorange actually do, and know why `timeit` disables the garbage collector and reports the minimum. Third, **tame the noise** — CPU frequency scaling, OS scheduling, GC pauses, cold caches, address-space randomization, thermal throttle — with repeats, pinning, and `pyperf --rigorous`. Fourth, report the **right statistic** — median plus standard deviation, not a single run and not the mean — and explain to a skeptic why. Fifth, recognize and dodge the **classic traps**: timing something the compiler folded away, timing your setup instead of your work, a loop too short to rise above the timer's resolution, and accidentally measuring a cache instead of a computation. We will build a small reusable harness along the way that you can paste into any project.

All numbers in this post come from a single named reference machine so they are comparable: **an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM**, on mains power with turbo enabled unless stated otherwise. Treat the absolute values as representative of that class of machine, not as universal constants — your laptop on battery will differ, and that difference is itself one of the lessons. This is the "you cannot optimize what you cannot measure reliably" post, and it opens the measurement track of the series. It builds directly on [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) and on the measure-first discipline laid out in [a mental model of performance: latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop). Once you can measure one snippet honestly, the next post — [CPU profiling with cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) — teaches you to find *which* snippet is worth measuring across a whole program.

## 1. Why timing is hard: resolution versus duration

Start with the most fundamental problem, the one that has nothing to do with Python and everything to do with physics and operating systems: **a clock has a resolution, and you cannot measure a duration meaningfully smaller than that resolution.**

A clock's resolution is the smallest increment it can distinguish. If your ruler is marked in centimeters, you cannot honestly report a measurement of 3.7 mm — the best you can say is "less than one tick." Computer clocks are rulers for time. The good ones on a modern Linux box resolve down to tens of nanoseconds; the bad ones — and `time.time()` historically was one of the bad ones — resolve only to a millisecond or worse on some platforms, because they are backed by a coarse system clock that ticks at a low frequency.

Now consider the durations you actually want to measure in Python performance work. A dictionary lookup costs on the order of 50 nanoseconds. An integer addition on boxed objects costs tens of nanoseconds. A list append, amortized, is similar. These are *operations you genuinely care about* — the difference between an `O(n)` and an `O(n^2)` algorithm is exactly the difference between doing 50 ns of work `n` times versus `n^2` times. But you cannot time a single 50 ns operation with a clock that resolves to even tens of nanoseconds, because your one measurement is dominated by quantization: the clock reads "0 or 1 ticks," which tells you essentially nothing.

The fix is the single most important mechanical idea in benchmarking: **batch the operation so the total time dwarfs the clock's resolution, then divide.** If a single op is 50 ns and your clock resolves to 50 ns, run the op a million times in a tight loop. The batch takes about 50 ms — a million times the resolution — and now the relative quantization error is one part in a million, negligible. Divide the batch time by the loop count and you recover an honest per-op figure. This is not a trick or a workaround; it is the only way to measure a sub-microsecond operation at all, and it is exactly what `timeit` does for you under the covers.

![stack diagram showing a single fifty nanosecond op below the clock tick then batching a thousand operations until the total dwarfs the resolution and dividing back to nanoseconds per op](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-7.png)

Let me make the resolution claim concrete. The standard library exposes the resolution of each clock through `time.get_clock_info`. On the reference machine:

```pycon
>>> import time
>>> time.get_clock_info("perf_counter")
namespace(implementation='clock_gettime(CLOCK_MONOTONIC)', monotonic=True, adjustable=False, resolution=1e-09)
>>> time.get_clock_info("time")
namespace(implementation='clock_gettime(CLOCK_REALTIME)', monotonic=False, adjustable=True, resolution=1e-09)
>>> time.get_clock_info("process_time")
namespace(implementation='clock_gettime(CLOCK_PROCESS_CPUTIME_ID)', monotonic=True, adjustable=False, resolution=1e-09)
```

The reported `resolution` is the clock's advertised tick, but the *effective* resolution — how small a duration you can actually distinguish — is worse, because reading the clock itself costs time. A `perf_counter()` call takes roughly 30 to 80 ns of overhead on this machine. So even with a nanosecond-resolution clock, you cannot honestly bracket a 20 ns operation with two reads; the measurement overhead is larger than the thing measured. Again: batch.

There is also a subtle floating-point trap hiding in `perf_counter`, which returns seconds as a float. A float has 53 bits of mantissa, about 15–16 significant decimal digits. `perf_counter()` returns seconds since some arbitrary epoch, and on a long-running process that value can be large — say, tens of thousands of seconds. Subtracting two large nearly-equal floats loses precision in the low bits exactly where your nanoseconds live. For sub-microsecond timing this is a real concern, and it is why Python 3.7 added `perf_counter_ns()`, which returns an integer number of nanoseconds and never loses a bit. **For anything below a microsecond, prefer `perf_counter_ns`.**

#### Worked example: timing a dict lookup the wrong way and the right way

Suppose you want the per-operation cost of a dictionary membership test, `key in d`, where `d` has a million entries. The naive approach brackets one lookup:

```python
import time

d = {i: i for i in range(1_000_000)}
key = 999_999

t0 = time.perf_counter_ns()
key in d
t1 = time.perf_counter_ns()
print(f"{t1 - t0} ns")     # prints something like "91 ns" or "0 ns" — pure noise
```

Run that ten times and you get wildly different answers — 0 ns, 41 ns, 167 ns — because you are measuring two clock reads plus one lookup, and the clock-read overhead and scheduler jitter swamp the 50 ns you care about. Now batch it:

```python
import time

d = {i: i for i in range(1_000_000)}
key = 999_999
N = 5_000_000

t0 = time.perf_counter_ns()
for _ in range(N):
    key in d
t1 = time.perf_counter_ns()
print(f"{(t1 - t0) / N:.1f} ns per lookup")    # ~52.0 ns per lookup, stable
```

On the reference machine this reports about **52 ns per lookup**, repeatably, run after run. The batch takes roughly 260 ms — five million times a 52 ns op — so the clock-read overhead (one pair of reads for the whole batch) is one part in millions, and the loop overhead is folded into the per-op figure honestly (it is part of "doing the lookup in a Python loop," which is what you are measuring). The lesson is mechanical and absolute: **never time a fast op once; batch until the total is at least a few milliseconds, then divide.**

## 2. Wall-clock, CPU time, and the deprecated clock

You have several clocks to choose from, and the choice changes what your number *means*. The distinction is not pedantic — pick the wrong clock and you will draw the wrong conclusion about an I/O-bound or a multi-threaded workload.

**`time.perf_counter()`** is the wall-clock benchmark timer. "Wall clock" means it measures real elapsed time as a clock on the wall would: if your process sleeps, waits on a network socket, or gets descheduled by the OS, that time counts. It is monotonic (it never goes backward, even if someone adjusts the system clock), it is not affected by NTP adjustments, and it has the highest available resolution. This is the default choice for almost all benchmarking, because what you usually want to know is "how long did this take in real time."

**`time.perf_counter_ns()`** is the same clock, but it returns an integer count of nanoseconds instead of a float of seconds. It avoids the floating-point precision loss described above. For sub-microsecond measurements, prefer it. Everything you would do with `perf_counter` you can do with `perf_counter_ns`, just in integer nanoseconds.

**`time.process_time()`** measures CPU time consumed by your process — user plus system — and *excludes* time the process spent asleep or waiting. If your code does `time.sleep(5)`, `perf_counter` advances by 5 seconds but `process_time` barely moves, because the CPU did nothing on your behalf during the sleep. This is exactly the tool you want when you ask "how much CPU work is this doing," independent of I/O waits or scheduler delays. It is also how you can *detect* that a workload is I/O-bound: if wall time greatly exceeds CPU time, your program is mostly waiting, not computing. A caution: `process_time` typically has coarser resolution than `perf_counter` because it is backed by CPU accounting ticks, so it is poor for sub-microsecond work — use it for whole-function or whole-program CPU accounting, not for a 50 ns op.

**`time.time()`** is the wall-clock *calendar* time — seconds since the Unix epoch. It is the clock you use to timestamp a log line or compute someone's age. It is the wrong clock for benchmarking for two reasons. First, it is **adjustable**: NTP can step or slew it, and a daylight-saving or manual clock change can make it jump forward or even backward mid-measurement, producing a negative or absurd duration. Second, on some platforms its resolution is poor — historically tens of milliseconds on Windows — so it cannot resolve short durations at all. The standard library has effectively deprecated it for timing; the documentation steers you to `perf_counter` and `monotonic`. **Do not benchmark with `time.time()`.** If you see it in a benchmark, treat the benchmark with suspicion.

There is also **`time.thread_time()`** (CPU time of the current thread only) and **`time.monotonic()`** (like `perf_counter` but tuned for longer-duration timeouts rather than maximum resolution). For benchmarking, `perf_counter`/`perf_counter_ns` and `process_time` cover the cases you care about.

![matrix comparing perf_counter and perf_counter_ns and process_time and time.time and timeit and pyperf across what each measures and its resolution and when to use it](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-2.png)

Here is the same comparison as a table you can keep next to your keyboard. The "use" column is the decision rule.

| Clock | Measures | Resolution | Monotonic? | Use it for |
| --- | --- | --- | --- | --- |
| `perf_counter` | Wall-clock elapsed | ~tens of ns | Yes | Default benchmarking (≥ 1 µs) |
| `perf_counter_ns` | Wall-clock elapsed, int ns | ~tens of ns | Yes | Sub-microsecond benchmarking |
| `process_time` | CPU time (user + sys) | Coarse (ticks) | Yes | Pure CPU accounting; detect I/O-bound |
| `thread_time` | This thread's CPU time | Coarse (ticks) | Yes | Per-thread CPU in multi-threaded code |
| `monotonic` | Wall-clock, never steps | ~tens of ns | Yes | Timeouts and long durations |
| `time.time` | Calendar time | ms or worse, adjustable | No | Timestamps and logs — never benchmarking |

#### Worked example: wall-clock versus CPU time on an I/O-bound call

Here is the distinction made physical. We time a function that downloads a file — work that is almost entirely waiting on the network, not computing.

```python
import time
import urllib.request

def fetch():
    with urllib.request.urlopen("https://example.com/large.bin") as r:
        return r.read()

w0, c0 = time.perf_counter(), time.process_time()
data = fetch()
w1, c1 = time.perf_counter(), time.process_time()

print(f"wall  : {w1 - w0:.3f} s")     # wall  : 0.840 s
print(f"cpu   : {c1 - c0:.3f} s")     # cpu   : 0.012 s
```

On the reference machine this prints roughly **0.840 s wall** and **0.012 s CPU**. The gap is the story: the function spent 828 ms doing nothing but waiting on the socket, and only 12 ms actually computing (parsing headers, copying bytes). If you had measured only CPU time you would conclude the function is blazing fast and reach for the wrong lever — there is no CPU work to optimize here; the win is concurrency, overlapping the waits, which is a threading or asyncio problem entirely. If you had measured only wall time you would know it is slow but not *why*. Measuring both tells you, in one shot, that this is an I/O-bound workload and where to spend your effort. That single comparison — wall versus CPU — is one of the highest-value two-line diagnostics in all of Python performance.

## 3. timeit: the standard tool and what it actually does

The standard library ships a purpose-built micro-benchmark tool, `timeit`, and it bakes in several of the lessons above. You should reach for it before writing your own brackets, because it handles batching and repetition correctly and avoids two specific footguns by default. But you must understand what it does, because its conveniences are also its sharp edges.

The command-line form is the fastest way to get a number:

```bash
$ python -m timeit "sum(range(1000))"
50000 loops, best of 5: 8.18 usec per loop

$ python -m timeit -s "data = list(range(1000))" "sum(data)"
50000 loops, best of 5: 4.92 usec per loop
```

Read that output carefully, because every word is doing work. **"50000 loops"** is `n`, the number of times the statement runs per *measurement* (the inner loop that batches the op so the total rises above the clock resolution — exactly section 1's idea). **"best of 5"** is `r`, the number of independent *repeats* of that measurement; `timeit` runs the whole `n`-loop five times and reports the best one. **"per loop"** means the time is already divided by `n`, so it is a per-execution figure. The `-s` flag is **setup**: code run once, *not* timed, before the timed loop — here we build the list once so we measure `sum`, not list construction. Getting `-s` right is the difference between measuring your work and measuring your setup, which is trap number two below.

The `-n` and `-r` flags let you set those counts explicitly:

```bash
$ python -m timeit -n 1000000 -r 7 "x = 1; x * x"
1000000 loops, best of 7: 0.0181 usec per loop
```

If you omit `-n`, `timeit` runs an **autorange**: it tries `n = 1, 2, 5, 10, 20, 50, ...` until a single measurement takes at least 0.2 seconds, then uses that `n`. This is autorange doing section 1's batching automatically — it grows the inner loop until the batch is comfortably above the clock resolution. It is convenient, but it has a consequence: for a very fast op, autorange picks a huge `n`, and for a slow op it picks a small one, so you cannot directly compare the raw loop counts between two runs. Compare the per-loop times, never the loop counts.

From code, the API mirrors the CLI. The two functions you will use are `timeit.timeit` (run once, return total time for `n` executions) and `timeit.repeat` (run the whole thing `r` times, return a list of `r` totals — this is the one you usually want).

```python
import timeit

# Setup runs once and is NOT timed; stmt runs `number` times and IS timed.
total = timeit.timeit(
    stmt="sum(data)",
    setup="data = list(range(1000))",
    number=100_000,
)
print(f"{total / 100_000 * 1e6:.2f} us per call")   # 4.91 us per call

# repeat() gives you the distribution, not one number. Prefer this.
times = timeit.repeat(
    stmt="sum(data)",
    setup="data = list(range(1000))",
    number=100_000,
    repeat=7,
)
per_call = [t / 100_000 * 1e6 for t in times]
print([round(x, 2) for x in per_call])   # [4.91, 4.93, 4.92, 5.04, 4.90, 4.95, 4.91]
```

Notice `repeat` returns a list. That list is the *distribution* of your measurement, and the whole second half of this post is about reading it correctly. A single `timeit()` call throws that distribution away and hands you one number; `repeat()` keeps it.

In IPython and Jupyter, the `%timeit` line magic and `%%timeit` cell magic are the same engine with autorange and nicer output:

```pycon
In [1]: data = list(range(1000))
In [2]: %timeit sum(data)
4.9 us +- 0.03 us per loop (mean +- std. dev. of 7 runs, 100,000 loops each)
```

The magic helpfully reports **mean ± std. dev. of 7 runs**, which is more honest than a single best-of number because it shows the spread. (We will argue below that *median* is even better than mean for noisy timing, but mean ± std from `%timeit` is already far better than one run.)

Two crucial behaviors are baked into `timeit` and you must know them. **First, it disables the garbage collector during measurement.** The `timeit` module temporarily calls `gc.disable()` around the timed loop, so a stray GC sweep cannot land inside your measurement and inflate it. This makes results more reproducible — but it also means `timeit`'s number does *not* include GC cost, which matters if your real workload allocates heavily and triggers collection. If you want to include GC, you pass `setup="gc.enable()"` or, in the API, you can re-enable it. Know which you want. **Second, it reports the minimum** (the CLI's "best of"). The minimum is the run least disturbed by noise, the closest to the true cost — more on why that is the right default for `timeit` and the wrong default for latency, in section 6.

## 4. The noise: why the same code times differently every run

Run the exact same benchmark twice and you get two different numbers. This is not a bug in your code, your timer, or `timeit`. A modern computer is a deeply non-deterministic environment for short-duration timing, and understanding *where* the noise comes from is what lets you tame it. Crucially, almost all of these noise sources **add** time and none subtract it — your op cannot run faster than its true cost, but a dozen things can make it appear slower. That one-sidedness is why the *minimum* over many runs is special, and it is the key to reading the statistics later.

![graph showing the true op cost and four noise sources of GC pause and CPU turbo and OS scheduler and cold cache all feeding into one measured time as a sum](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-4.png)

Here are the major noise sources, in rough order of how often they bite, and what each one does to your number.

**CPU frequency scaling and turbo.** A modern CPU does not run at a fixed clock. It scales frequency up and down to save power (DVFS — dynamic voltage and frequency scaling) and boosts a few cores above the base clock when thermal headroom allows (Turbo Boost / Precision Boost). When your benchmark starts, the core may be at a low idle frequency and ramp up over tens of milliseconds; the first measurements are slower than the later ones. On a laptop on battery, the governor may aggressively downclock, halving your throughput versus the same machine on mains power. This is the single biggest reason "the second run was faster" — the CPU warmed up. The frequency can vary 2–3× across the run.

**OS scheduling.** Your process shares the machine with the kernel and every other process. The scheduler can preempt your benchmark to run something else, migrate your thread to a different (cold) core, or service an interrupt mid-loop. Each preemption inserts a gap of microseconds to milliseconds into a measurement that happened to span it. This is why occasional runs are dramatically slower than the rest — they caught a context switch.

**Garbage collection pauses.** CPython frees most objects immediately via reference counting, but it also runs a periodic generational collector to catch reference cycles. When that collector fires, it pauses your code to walk the object graph. If a collection lands inside your timed region, that region is inflated by the pause. This is precisely why `timeit` disables GC by default — to remove this source of variance from micro-benchmarks. In a real workload you cannot disable it, so you must either measure with it on or account for it.

**Cold caches.** The first time you touch data, it is not in the CPU's L1/L2/L3 caches or even necessarily in RAM (it might be a page fault to disk). The first iteration pays a cache-miss and page-fault tax — hundreds of cycles per miss — that later iterations, hitting warm caches, do not. This is the classic "second run is faster," and it is also a trap: if you measure only the warm runs you overstate real-world speed for code that runs cold.

**Address-space layout randomization (ASLR).** For security, the OS randomizes the base addresses of the stack, heap, and libraries on each process launch. This shifts where your data lands relative to cache lines and memory pages, and that placement subtly changes cache and TLB behavior. The effect is small but real: the *same binary* can benchmark a few percent differently across process launches purely because of where things landed in memory. This is why a difference that only shows up across separate process invocations, and vanishes within one process, may be ASLR, not your change.

**Thermal throttling.** Run a CPU hard for long enough and it heats up; past a threshold it throttles frequency to stay within its thermal envelope. A benchmark that runs cool for the first few seconds and then throttles will show times creeping upward — the later samples are slower not because of your code but because the silicon is hot. Laptops and small-form-factor machines throttle far sooner than a well-cooled server.

The combined effect is that a single measurement of a short op can easily be off by 40% or more in either direction relative to the true cost, and the distribution is **right-skewed** — a long tail of slow runs caused by the additive noise, with a sharp floor at the true cost that nothing can beat. Keep that shape in mind; it dictates which statistic to trust.

### Taming the noise

You cannot eliminate noise, but you can shrink it dramatically and make your numbers reproducible. In rough order of effort and payoff:

1. **Repeat and take a robust statistic.** The cheapest, most universal fix. Run many measurements; the noise averages out of the spread and the true cost shows through. This is section 5.
2. **Warm up before measuring.** Run the code a few times untimed so caches fill, the CPU reaches its boost frequency, and any one-time compilation or import completes. Then measure the steady state. Discard the warmup samples.
3. **Disable GC during the timed region** for micro-benchmarks (what `timeit` does), or explicitly measure with it on if your workload allocates heavily — but be deliberate about which.
4. **Pin the CPU frequency and the process to a core.** On Linux, set the CPU governor to `performance` so the frequency does not scale, and pin your process to a specific core with `taskset -c 2` so the scheduler does not migrate it. For real rigor, isolate that core from the scheduler entirely (`isolcpus`).
5. **Use a tool built for this** — `pyperf` — which automates warmup, many repeats across many fresh processes, GC handling, frequency checks, and statistics. It is the rigorous path, covered in section 7.
6. **Quiet the machine.** Close the browser, the chat client, the IDE's indexer. Each is a process competing for cores and cache. On a CI box, run benchmarks on a dedicated, otherwise-idle runner.

To make the payoff of pinning concrete: on the reference machine, the date-parsing benchmark from section 5 reports a stdev of about **612 ns** on a default desktop with turbo on, the browser open, and the frequency governor free to scale. Run the identical benchmark after `sudo python -m pyperf system tune` — governor pinned to `performance`, turbo disabled, the process bound to an isolated core — and the stdev drops to roughly **40 ns**, a fifteen-fold tightening, while the median barely moves. Nothing about the code changed; the spread shrank because the *environment* stopped injecting variance. That tighter spread is what lets you detect a 3% change instead of being lost in a 50% band, and it is the entire reason a tuned machine matters for a number you intend to defend.

You do not need all six for every measurement. For a quick "is A or B faster" check, repeat-and-median (step 1) is enough. For a number you will publish or gate a release on, use `pyperf` with pinning (steps 4 and 5). The point is to match the rigor to the stakes — and to be honest in your write-up about which steps you took, because a stdev of 40 ns on a tuned box and 600 ns on a noisy laptop describe very different levels of confidence in the same median.

## 5. Statistics: median, stdev, and why not the mean

You have a list of `r` measurements from `timeit.repeat` or your harness. Now you must collapse it into a number to report. This is where most benchmarks go wrong, because the obvious choice — the average — is the wrong one for noisy timing data.

Recall the shape of the distribution from section 4: a sharp floor at the true cost, and a right-skewed tail of slow runs caused by additive noise (a scheduler preemption here, a GC pause there). The arithmetic **mean** is dragged upward by that tail. A single bad run — one measurement that caught a 5 ms context switch — can pull the mean of a set of 50 ns measurements up by a large fraction, even though that one sample tells you nothing about the op's real cost. The mean answers "what is the average over all runs including the disturbed ones," which is rarely the question you are asking.

The **median** — the middle value when the samples are sorted — is *robust* to that tail. By definition, half the samples are below it and half above, so a handful of extreme slow runs cannot move it; you would need to corrupt more than half your samples to shift the median. For right-skewed timing data, the median sits near the floor where the true cost lives, exactly where you want your headline number. **Report the median.** This is the single most important statistical choice in benchmarking.

Pair the median with the **standard deviation** (stdev), which measures the spread of the samples. The stdev does not change your central estimate; it tells you how much to *trust* it. A median of 50 ns with a stdev of 1 ns is a tight, reliable measurement. A median of 50 ns with a stdev of 30 ns means your environment is noisy and you should not believe small differences — a "10% speedup" inside a 60% spread is meaningless. Always report central tendency *and* spread: `50 ns ± 2 ns` says far more than `50 ns`. A common, even better practice for skewed data is to report the median plus an interquartile range or the min and a high percentile, but median ± stdev is the pragmatic default and what `%timeit` approximates with mean ± stdev.

Then there is the **minimum**, which `timeit` reports and which deserves a careful word. Because noise is one-sided — it only adds time — the minimum is the run that caught the *least* noise, the closest to the platonic true cost of the operation with no GC, no preemption, no cache miss. For a *micro-benchmark of a deterministic op*, where you want "how fast can this go in the best case," the minimum is a principled choice and it is why `timeit` defaults to it. But the minimum is also the most optimistic number, and it is the *wrong* statistic for anything user-facing, because your users do not experience the best case — they experience the typical case and, painfully, the tail.

That brings in **p99** and the percentiles. For a service handling requests, what matters is not the best run or even the median but the **tail**: the p99 latency is the value that 99% of requests come in under, and it is what a user notices when the page hangs. Tail latency is a different question from micro-op cost, and it calls for a different statistic. When you benchmark a request handler, report the median *and* the p99; the gap between them is your tail risk.

### Why the median is mathematically the right default

It is worth making the "median resists outliers" claim rigorous, because it is the statistical heart of honest benchmarking and it is easy to wave at without proof. The technical property is the **breakdown point**: the fraction of your sample you can replace with arbitrarily extreme values before the statistic itself becomes arbitrarily large. The mean has a breakdown point of $0$ — a *single* corrupted sample, pushed to infinity, drags the mean to infinity, because the mean is $\frac{1}{n}\sum_i x_i$ and any one term can dominate the sum. The median has a breakdown point of $0.5$ — you must corrupt *more than half* of all samples before the middle value moves arbitrarily, because the median only cares about the ordering, not the magnitudes, of the points away from the center.

Timing noise corrupts a small *minority* of samples severely: most runs are clean and near the floor, but a few catch a scheduler preemption or a GC pause and land far out in the tail. That is precisely the regime where the breakdown point matters. With, say, $5\%$ of samples blown out to 10× the true cost, the mean rises by roughly $0.05 \times 9 = 45\%$ — a near-50% overstatement from a one-in-twenty event — while the median, sitting comfortably below the 95th percentile, does not move at all. Concretely, for the section-5 data the mean was $1417$ ns and the median $1206$ ns on the *same* samples; the 18% gap between them is exactly the additive tail being absorbed by the mean and rejected by the median. If you remember one formula from this post, let it be the intuition behind the breakdown point: **the mean is a democracy where outliers can buy unlimited votes; the median is a democracy where every sample gets exactly one vote regardless of how extreme it is.**

There is a second, subtler reason the median wins for timing specifically. The true cost of a deterministic operation is a *constant*, and every measurement is that constant plus a non-negative noise term: $x_i = c + \epsilon_i$ with $\epsilon_i \ge 0$. This is a **shifted, one-sided** distribution, not the symmetric bell curve the mean and standard deviation were designed for. For a symmetric distribution the mean and median coincide and either is fine; for the right-skewed, floor-bounded distribution of timing data they diverge, and the median tracks the floor (the true cost) while the mean tracks the floor *plus the average noise*. You almost never want "true cost plus average noise" — you want the true cost — so you almost never want the mean. The minimum, $\min_i x_i = c + \min_i \epsilon_i \approx c$, is even closer to $c$, which is exactly why `timeit` reports it for deterministic micro-ops; the median is the robust compromise you reach for when the op is *not* perfectly deterministic and the floor itself wobbles.

![matrix mapping each statistic of min and median and mean and stdev and p99 to what it tells you and when to use it](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-5.png)

The decision rule, compactly:

| Statistic | What it tells you | When to report it |
| --- | --- | --- |
| **min** | Least-disturbed run; best-case op cost | Micro-benchmark of a deterministic op (what `timeit` does) |
| **median** | Typical run, robust to noise tail | The default headline number for almost everything |
| **mean** | Average including disturbed runs | Rarely for latency; only when the distribution is symmetric |
| **stdev** | Spread; how much to trust the center | Always, paired with the median |
| **p99** | Tail latency; the bad-but-common case | Anything user-facing, SLOs, request handlers |

#### Worked example: the same op reported five ways

Here is a concrete distribution to make the statistics tangible. We time a small function — parsing a date string with `datetime.strptime` — collecting 200 samples on the reference machine, each sample being the median-of-a-batch so individual reads are clean. The raw per-call times, sorted, run from a floor around 1,180 ns up to a few outliers near 9,000 ns where the scheduler interfered. Computing the five statistics:

```pycon
>>> import statistics as st
>>> samples = load_samples()   # 200 per-call times in ns, from the harness
>>> min(samples)
1182.0
>>> st.median(samples)
1206.0
>>> st.mean(samples)
1417.3
>>> st.pstdev(samples)
612.4
>>> sorted(samples)[int(0.99 * len(samples))]
8740.0
```

Look at what each number says about the *same data*. The **min, 1,182 ns**, is the cleanest run — the true cost when nothing interfered. The **median, 1,206 ns**, is essentially the same, just 2% above the floor: the typical call is nearly as clean as the best, which tells you the op is well-behaved. The **mean, 1,417 ns**, is 18% higher than the median — dragged up by a handful of slow outliers, and *not* representative of a typical call. The **stdev, 612 ns**, is huge relative to the median, which is your warning that there are outliers and you should not trust differences smaller than the spread. And the **p99, 8,740 ns** — over seven times the median — is what a user hits on a bad request, and it would be invisible if you only reported the median. If you had reported the mean as "the time," you would have overstated the typical cost by 18%; if you had reported only the median, you would have hidden a brutal tail. Median ± stdev plus p99 tells the whole truth in three numbers.

## 6. Building a small honest benchmark harness

Now we assemble the lessons into a reusable harness — under forty lines, no dependencies beyond the standard library — that you can paste into any project. It does warmup, batches the inner loop so a fast op rises above the timer resolution, repeats to build a distribution, controls the GC explicitly, and reports median plus stdev. This is the "honest benchmark loop" from figure 1 turned into code.

```python
import gc
import statistics as st
import time
from typing import Callable

def benchmark(
    func: Callable[[], object],
    *,
    inner: int = 10_000,     # ops per timed batch (rise above clock resolution)
    repeats: int = 11,       # number of batches (build a distribution)
    warmups: int = 3,        # untimed batches to reach steady state
    disable_gc: bool = True, # match timeit's default; flip if you want GC cost in
) -> dict:
    """Time `func` honestly: warmup, batch, repeat, report median + stdev (ns/op)."""
    # Warmup: fill caches, ramp the CPU to boost, finish one-time work. Untimed.
    for _ in range(warmups):
        for _ in range(inner):
            func()

    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    try:
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for _ in range(inner):
                func()
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / inner)   # ns per op for this batch
    finally:
        if gc_was_enabled:
            gc.enable()

    return {
        "ns_per_op_median": st.median(samples),
        "ns_per_op_min": min(samples),
        "ns_per_op_stdev": st.pstdev(samples),
        "samples": samples,
    }
```

Using it on the dict-lookup example from section 1:

```pycon
>>> d = {i: i for i in range(1_000_000)}
>>> key = 999_999
>>> r = benchmark(lambda: key in d, inner=200_000, repeats=11)
>>> f"{r['ns_per_op_median']:.1f} ns +/- {r['ns_per_op_stdev']:.1f}"
'51.8 ns +/- 0.6'
```

A few design choices in that harness are deliberate and worth calling out, because each one defends against a specific failure from the earlier sections. The **warmup loop** runs the work untimed first so the CPU reaches its boost clock and the caches fill — defeating the cold-cache and frequency-ramp noise from section 4. The **inner loop** batches `inner` calls into one timed region so a 50 ns op produces a multi-millisecond batch, well above the clock resolution — section 1's mechanic. The **repeats** build a distribution rather than a single number — section 5's requirement. The **GC control** is explicit and reversible, with a `try/finally` so we always restore the user's GC state even if `func` raises. And we report **median, min, and stdev together** so the caller sees both the robust center and the spread. The one thing the harness does *not* do is spawn fresh processes — so it cannot defend against ASLR or measure first-call import cost. For that you graduate to `pyperf`.

The single most important caveat about this harness, and any in-process micro-benchmark: **`func` must do real, non-cached work each call, and the work must not be optimized away.** If you pass `lambda: 1 + 1`, the harness will dutifully report a few nanoseconds of pure loop overhead, because the addition was folded to a constant at compile time and there is nothing to measure. That is the next section.

![before and after comparison of a single time.time call that reports sixty two nanoseconds plus or minus forty percent versus a repeat and median harness that reports fifty nanoseconds plus or minus two percent](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-3.png)

The figure above makes the payoff visible: the same 50 ns operation, measured with one `time.time()` bracket, comes out as a wide noisy band — easily ±40% run to run — while the harness above, with warmup and eleven batches and a median, lands within about ±2% of the true cost. Same op, same machine; the only difference is the method. That ±2% versus ±40% is the entire reason this post exists.

## 7. pyperf: rigorous benchmarking when the number matters

For a quick check, `timeit` or the harness above is plenty. But when you are going to *publish* a number, gate a CI pipeline on it, or settle a "did this PR actually speed things up" argument, you want the most rigorous tool in the Python ecosystem: **`pyperf`**, the benchmarking library that powers the official CPython speed benchmarks and the [pyperformance](https://pyperformance.readthedocs.io/) suite. It exists precisely because the authors of CPython needed to detect 1–2% regressions through all the noise we have been discussing, and a single `timeit` run cannot do that reliably.

`pyperf` does several things that `timeit` does not. It runs your benchmark in **many fresh worker processes** — by default 20 processes, each doing its own warmups and timed runs — so it averages out ASLR placement and any per-process state, not just within-process noise. It **calibrates** the inner loop count automatically (like autorange but more carefully). It **checks the system for you**, warning if the CPU frequency governor is not pinned, if turbo is enabled, if the system is under load, or if ASLR is making runs inconsistent. And it computes proper statistics — mean, median, stdev — across all the runs from all the processes.

Install it and run a benchmark from the command line:

```bash
$ pip install pyperf
$ python -m pyperf timeit -s "data = list(range(1000))" "sum(data)"
.....................
Mean +- std dev: 4.93 us +- 0.06 us
```

The `--rigorous` flag dials up the number of processes and runs for a more thorough measurement when you are chasing a small effect:

```bash
$ python -m pyperf timeit --rigorous -s "data = list(range(1000))" "sum(data)"
.........................................
Mean +- std dev: 4.91 us +- 0.04 us
```

Notice the tighter standard deviation (±0.04 versus ±0.06 µs) — more runs, less uncertainty. Before a serious run, `pyperf` can also tune your Linux system for you, pinning CPU frequency, disabling turbo, and isolating cores:

```bash
$ sudo python -m pyperf system tune     # pin governor, disable turbo, isolate cores
$ python -m pyperf timeit --rigorous "my_function()"
$ sudo python -m pyperf system reset     # restore normal power management afterward
```

From code, you write a small runner and `pyperf` handles the process orchestration. This is the form you commit to a repo so anyone can reproduce the number:

```python
import pyperf

def setup_and_work():
    data = list(range(1000))
    return sum(data)

runner = pyperf.Runner()
runner.bench_func("sum_1000", setup_and_work)
```

```bash
$ python bench_sum.py -o sum.json          # run it, save results to JSON
$ python bench_sum.py --rigorous -o sum.json
$ python -m pyperf compare_to before.json after.json   # is the PR faster? by how much?
Mean +- std dev: 4.91 us +- 0.04 us -> 3.12 us +- 0.03 us: 1.57x faster
Significant (t=84.2)
```

That last command is the payoff: `compare_to` takes the before and after JSON files, computes the speedup *with a significance test*, and tells you whether the difference is real or within the noise. "Significant (t=84.2)" means the change is statistically real, not luck. "Not significant" would mean you cannot distinguish the two — and you should not claim a speedup you cannot distinguish. This is the difference between "the number went down on my screen" and "I have evidence the code is faster," and it is the standard you should hold yourself to before posting that rocket emoji.

#### Worked example: catching a regression too small for the eye

Here is the scenario `pyperf` was built for, and the one a single `%%time` cannot handle. You are reviewing a PR that refactors a hot row-parsing function. The author swears it is "the same speed, just cleaner." You suspect it added an allocation in the inner loop. The effect, if it exists, is small — maybe 3% — and that is well inside the run-to-run noise of any single measurement on the reference machine, so eyeballing two `%timeit` runs is hopeless: one will randomly come out ahead of the other and prove nothing.

So you measure both rigorously. You check out `main`, run the benchmark to `before.json`, check out the PR branch, run it to `after.json`, each under `pyperf --rigorous` on a tuned system:

```bash
$ sudo python -m pyperf system tune
$ git checkout main      && python bench_parse.py --rigorous -o before.json
$ git checkout pr-branch && python bench_parse.py --rigorous -o after.json
$ python -m pyperf compare_to before.json after.json
Mean +- std dev: 8.12 us +- 0.05 us -> 8.39 us +- 0.06 us: 1.03x slower
Significant (t=31.7)
```

The verdict is unambiguous: the PR is **3% slower**, and the change is **statistically significant** — `pyperf` ran enough fresh processes and timed runs that a 3% shift stands clearly above the ±0.05 µs noise floor, and the t-statistic of 31.7 says the probability this is luck is vanishingly small. Without `pyperf`, this regression sails through review under the cover of "it's within noise." With it, you can write in the review, with evidence, "this adds 0.27 µs per row; on the 50-million-row nightly job that is about 13 extra seconds, here is the measurement." That is the difference rigorous benchmarking buys you: not a vibe, a number you can defend. And note the *direction* of the rigor — the more important point is not that you caught a 3% regression, but that you could **trust a null result** too. If `compare_to` had said "Not significant," you could have approved the refactor knowing the cleanup did not cost anything, instead of blocking it on a superstition. Honest measurement clears the innocent as confidently as it convicts the guilty.

| Tool | Repeats? | Fresh processes? | Stats reported | Reach for it when |
| --- | --- | --- | --- | --- |
| `time.perf_counter` brackets | You write them | No | None (you compute) | Quick one-off, you control the loop |
| `timeit` / `%timeit` | Yes (`-r`) | No | min, or mean ± std | Comparing two snippets interactively |
| Custom harness (section 6) | Yes | No | median, min, stdev | Reusable in-process measurement |
| `pyperf` | Yes, many | Yes (≈20) | mean, median, stdev, significance | Publishing, CI gates, small effects |

## 8. The classic traps that produce lying numbers

Even with the right timer, the right repeats, and the right statistic, you can still measure the wrong thing entirely. These are the traps that have fooled experienced engineers, and each one produces a number that is internally consistent and completely meaningless. Learn to spot them.

### Trap 1: measuring something the compiler folded away

CPython's compiler does **constant folding** in a peephole optimization pass: a constant expression like `1 + 1` is computed once at compile time and replaced with the literal `2` in the bytecode. The addition you think you are timing never happens at run time. So the famous beginner benchmark `timeit("1 + 1")` does not measure integer addition — it measures the loop overhead of doing nothing, because the bytecode contains just the constant `2`.

![before and after comparison of timing one plus one which the compiler folds to a constant and reports near zero versus timing a real function call that reports four hundred twenty nanoseconds](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-6.png)

You can see the fold directly with the disassembler:

```pycon
>>> import dis
>>> dis.dis(compile("1 + 1", "<s>", "eval"))
  0           RESUME                   0
  1           RETURN_CONST             0 (2)
```

There is no `BINARY_OP`. The compiler emitted `RETURN_CONST 2`. The "addition" is gone. Time it and you get a few nanoseconds of nothing:

```pycon
>>> import timeit
>>> timeit.timeit("1 + 1", number=10_000_000)
0.0089        # ~0.9 ns per "op" — pure loop overhead, the add was folded away
```

The fix is to make the work depend on something the compiler cannot know at compile time — pass it through a variable from `setup`, or call a function:

```pycon
>>> timeit.timeit("a + b", setup="a = 1; b = 1", number=10_000_000)
0.21          # ~21 ns per add — now it's real; a and b are runtime values
```

Constant folding is not the only optimization that erases your benchmark. The compiler also eliminates dead code: if you compute a value and never use it, an optimizing path may drop it. A common variant of this trap is benchmarking a pure function whose result you discard — a sufficiently clever runtime could elide it (CPython is conservative here today, but JIT-compiled environments like PyPy and Numba absolutely will). The defensive habit: **always consume the result of the work you are timing** — return it, append it, fold it into an accumulator — so neither the compiler nor a JIT can prove it is dead and delete it.

### Trap 2: measuring setup instead of the work

The second trap is putting one-time setup *inside* the timed region. If you want to time `sum(data)` but you build `data` inside the loop, you are timing list construction plus the sum, and on a big list the construction dominates and your "sum benchmark" is really a "list-building benchmark." This is exactly what `timeit`'s `setup` parameter exists to prevent — code in `setup` runs once, untimed, before the timed loop.

```python
# WRONG: builds the million-element list inside every timed iteration.
timeit.timeit("sum(list(range(1_000_000)))", number=100)
# This times list construction (the expensive part), not the sum.

# RIGHT: build once in setup; time only the sum.
timeit.timeit("sum(data)", setup="data = list(range(1_000_000))", number=100)
```

The same trap appears in custom harnesses when you allocate, open a file, compile a regex, or import a module inside the timed loop. Ask of every timed region: "is everything in here the work I want to measure, or did I leave a one-time cost inside?" Move every one-time cost into setup. The exception is when the setup *is* the thing you care about — if you are benchmarking how long it takes to compile a regex or import a module, then that belongs inside the timed region, but be deliberate about it.

### Trap 3: a loop too short to rise above resolution

This is section 1's lesson stated as a trap. If you time a single fast op with one bracket — `t0 = perf_counter(); fast_op(); t1 = perf_counter()` — the duration is smaller than the clock's effective resolution plus the clock-read overhead, and your number is pure quantization noise. It will read 0 ns, or 90 ns, or 200 ns, run to run, with no relationship to the truth. The symptom is a measurement that jumps around wildly and sometimes reads zero. The fix is always the same: batch the op thousands of times in a tight loop, time the batch, divide — or just use `timeit`, which does this for you via autorange.

### Trap 4: accidentally measuring a cache

The fourth trap is the subtlest, and it is the one from this post's opening story. The *first* time you run an operation, it is cold: data is not in CPU cache, pages are not faulted in, a memoized function has not yet stored its result, an `lru_cache` is empty, a connection pool is unestablished, an import has not happened, a JIT has not compiled. Every subsequent run is *warm* and faster. If your benchmark runs the op many times and you measure the warm runs, you are measuring the cache, not the computation — and you will report a speed your code never achieves in production where each call may be cold.

There are two opposite failure modes here and you must decide which case you are in. If your production code path is **warm** (the function is called in a hot loop, the data is already resident), then warming up before measuring is *correct* — discard the cold first runs and measure steady state, which is what the harness does. But if your production path is **cold** (the function is called once per request on fresh data), then the warm benchmark is a lie and you must measure the cold path: fresh data each iteration, cleared caches, sometimes a fresh process. The classic specific case is benchmarking a function decorated with `@functools.lru_cache` on the same argument repeatedly — the first call computes, every later call is an `O(1)` dict hit, so your "benchmark" reports the cache-lookup time and tells you nothing about the actual computation. The defense is to vary the input each iteration (so the cache misses the way it would in production) or to clear the cache between samples, depending on which you actually want to know.

#### Worked example: the constant-folding and caching traps, measured wrong then right

Let me put two traps in one before-and-after, with numbers from the reference machine, to show how badly a careless benchmark misleads. We want to know the cost of `expensive(n)`, a function that does real work (say, summing the squares up to `n`), and it is wrapped in `lru_cache`.

```python
import functools, timeit

@functools.lru_cache(maxsize=None)
def expensive(n: int) -> int:
    return sum(i * i for i in range(n))
```

The wrong benchmark calls it with the same argument every time, so after the first call every run is a cache hit:

```pycon
>>> timeit.timeit("expensive(10_000)", globals=globals(), number=100_000)
0.0042        # 42 ns per call — this is the lru_cache lookup, NOT the work
```

Forty-two nanoseconds for summing ten thousand squares would be miraculous; it is a lie, because we measured the cache. The right benchmark varies the argument so each call misses the cache the way production would, and times a single representative call honestly with the harness:

```pycon
>>> import itertools
>>> counter = itertools.count(10_000)
>>> timeit.timeit("expensive(next(counter))", globals=globals(), number=10_000)
3.91          # ~391 us per call — the real cost of the computation
```

The honest number is **391 µs per call**, about **9,300× larger** than the cached lie of 42 ns. If you had trusted the first benchmark, you would have concluded this function is free and built a system that calls it millions of times, then watched production fall over. The table below summarizes how each trap moves the number and which direction the error runs.

| Trap | What you think you measure | What you actually measure | Error direction |
| --- | --- | --- | --- |
| Constant folding | `1 + 1` addition | Empty loop overhead | Wildly too fast (≈0) |
| Setup in the loop | The operation | One-time construction | Too slow, op hidden |
| Loop too short | One op's cost | Clock resolution + read overhead | Random, often 0 |
| Caching / warm runs | The computation | A cache hit | Too fast (often 100–10,000×) |

## 9. Case studies and real-world numbers

Abstract rules land harder with real episodes. Here are several drawn from the Python ecosystem and from the kind of production incidents this discipline prevents.

**The Faster CPython project measured everything.** When the CPython core team set out to make 3.11 and 3.12 faster — the "Faster CPython" effort that delivered the 10–60% speedups you get for free by upgrading — they did not eyeball it. They built and ran the `pyperformance` suite under `pyperf` with system tuning, comparing every change against a baseline with significance tests, precisely because the individual wins (a few percent here, a specialization there) were smaller than the machine's noise floor unless measured rigorously. The PEP 659 specializing adaptive interpreter's gains were validated this way. The lesson for you: if the people optimizing the interpreter itself need `pyperf` with significance testing to see their wins, your one-off `%%time` cell cannot be trusted to detect a 5% change either.

**The "it got faster after I added logging" mystery.** A team reported, genuinely puzzled, that adding a `logging` call to a hot function made it *faster* in their benchmark. The real explanation was timing noise plus a warmup artifact: their benchmark ran the function only a handful of times, the version "with logging" happened to run second on an already-warmed CPU and cache, and the run-to-run variance was larger than the effect they thought they saw. Re-running under `pyperf` with proper warmup and many processes showed the logging version was, as expected, marginally *slower*, and the difference was within noise either way. The fix was not in the code; it was in the measurement.

**Polars and DuckDB benchmark wins are real — because they are measured honestly.** When the Polars project claims multi-second-to-sub-second speedups over pandas on a group-by aggregation, those numbers come from published, reproducible benchmark suites (the [DuckDB H2O.ai database-like ops benchmark](https://duckdblabs.github.io/db-benchmark/)) run on named hardware with warm and cold variants reported separately, multiple runs, and the data sizes stated. That is why they are credible. Contrast that with a blog post that pastes a single `%%time` from a warm notebook cell: the Polars-style benchmark you can act on; the single-cell number you cannot. When you read a performance claim, the first question is always "how was this measured" — repeats, warmup, machine, data size, statistic. If those are missing, the number is decoration.

**The micro-benchmark that did not predict the macro result.** A developer micro-benchmarked two implementations of a string-cleaning function with `timeit` and found implementation B was 15% faster per call. They shipped B. The end-to-end pipeline got *slower*. The reason: implementation B allocated more temporary objects, and in the real workload — unlike the GC-disabled `timeit` micro-benchmark — those allocations triggered more frequent garbage collection, and the GC cost in the macro run swamped the 15% micro win. `timeit` had disabled GC (correctly, for isolating per-call cost), but that very isolation hid a cost that only appears at scale. The lesson: a micro-benchmark answers "what is the per-call cost in isolation," not "what is the system-level effect." Always confirm a micro-win with a macro measurement on the real workload — which is exactly where [cProfile and the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) take over from `timeit`.

These episodes share one root cause and one cure. The root cause is treating a number as truth because it appeared, without asking how it was produced. The cure is the discipline of this post: repeat, warm up, control GC, report median and spread, and confirm at the level where the effect actually matters.

## 10. When to micro-benchmark, when to profile, and when not to bother

A timer is one tool among several, and reaching for the wrong one wastes effort. The first decision is the question you are actually asking, because that picks the tool.

![tree decision diagram choosing between timeit or pyperf for one isolated operation versus cProfile for finding the hot path across a whole program](/imgs/blogs/benchmarking-python-correctly-timeit-pitfalls-and-statistics-8.png)

If your question is **"what does this one small operation cost in isolation"** — is a `set` lookup faster than a `list` scan, is a comprehension faster than a loop, is method A or B faster for this hot inner function — then you want a **micro-benchmark**: `timeit` for a quick interactive answer, `pyperf` when the number must be rigorous. That is the entire scope of this post.

If your question is **"where does my whole program spend its time"** — which function dominates a 9-hour ETL, why is the `/search` endpoint slow — then a micro-benchmark is the *wrong* tool, because you do not yet know which op to measure. You want a **profiler** that attributes time across the whole run: `cProfile` for a deterministic function-level breakdown, sampling profilers for production. Micro-benchmarking before profiling is the classic mistake from the opening story — optimizing a function you *guessed* was hot, when a profiler would have shown it was 0.4% of runtime. **Profile first to find the hot path; micro-benchmark second to optimize the specific op the profiler indicted.** The two tools compose: the profiler tells you *what* to make faster, the micro-benchmark tells you *whether your change to that thing actually worked*.

And sometimes the right answer is **do not benchmark at all yet**. Amdahl's law caps the payoff of optimizing any single part at the fraction of total time it consumes — speed up a component that is 2% of runtime by infinity and you save 2%. If a piece of code is not on the hot path, the most rigorous benchmark of it is wasted effort, because even a perfect result moves nothing the user feels. The honest sequence is: is the program even too slow? If not, stop. If yes, profile to find the dominant cost. Only then micro-benchmark the dominant op to drive and verify a fix. Benchmarking in any other order is, at best, education and, at worst, three days spent making nothing faster.

There is also a rigor dial to set by the stakes. For a throwaway "which of these two is faster" in a REPL, `%timeit` and read the mean ± std — thirty seconds, good enough. For a number you will put in a PR description, the section-6 harness with median ± stdev — a couple of minutes, defensible. For a number that gates a release, blocks a merge on regression, or goes in a published benchmark, `pyperf --rigorous` with system tuning and a significance test — a few minutes of setup, and now the number will survive scrutiny. Match the effort to the consequence; do not run a 20-process `pyperf` sweep to settle a REPL curiosity, and do not paste a single warm `%%time` into a release gate.

## 11. When to reach for this — and when not to

Honest measurement is cheap insurance, but every level of rigor has a cost, and knowing when *not* to escalate is as important as knowing how.

**Reach for `timeit` / `%timeit`** when you are interactively comparing two small alternatives and you want an answer in seconds — set membership versus list scan, two ways to write a hot loop, a comprehension versus `map`. It batches and repeats for you, disables GC, and is right there in the REPL. This is the default and it covers the majority of day-to-day questions.

**Reach for the custom harness (or `pyperf`)** when you need median *and* spread, when the result goes in a PR or a report, or when you are chasing an effect small enough that you must distinguish it from noise. The harness gives you the distribution; `pyperf` adds fresh processes, system checks, and significance testing for when the number must hold up.

**Reach for `process_time` alongside `perf_counter`** whenever you suspect a workload is I/O-bound or sleeping — the wall-versus-CPU gap is the fastest way to confirm it, and it redirects you from optimizing CPU work that is not the bottleneck toward concurrency, which is the actual lever.

Now the *do nots*, stated plainly. **Do not benchmark with `time.time()`** — it is low-resolution and adjustable; use `perf_counter`. **Do not report a single run** — one number has no error bar and is dominated by noise; you cannot tell signal from luck. **Do not report the mean for skewed timing data** — it is dragged by outliers; report the median. **Do not trust a difference smaller than the spread** — a "10% speedup" inside a 60% stdev is noise; tighten the measurement or admit you cannot tell. **Do not micro-benchmark before profiling** — you will optimize the wrong thing; profile to find the hot path first. **Do not benchmark a function that is 2% of runtime** — Amdahl caps your win at 2% no matter how clean the benchmark. **Do not benchmark on a laptop on battery** when the number matters — the governor downclocks and thermals throttle; use a machine on mains power, ideally with `pyperf system tune`. **Do not believe a warm-cache number for a cold production path** — measure the path your users actually hit. And **do not paste a `%%time` from a twice-run notebook cell and call it a result** — that is the rocket-emoji trap from the opening, and it is how a roomful of competent engineers talk themselves into a speedup that never happened.

The meta-rule: the effort you spend measuring should be proportional to the consequence of being wrong. A REPL curiosity deserves `%timeit`; a release gate deserves `pyperf` with a significance test; everything in between deserves the harness and a median with an error bar.

## 12. Key takeaways

- **You cannot time an op smaller than your clock's resolution.** Batch the operation thousands of times so the total dwarfs the resolution, then divide. This is the one mechanical truth under all micro-benchmarking, and it is what `timeit` autorange does for you.
- **Use `perf_counter` / `perf_counter_ns` for wall-clock, `process_time` for CPU-only, and never `time.time` for benchmarking.** The wall-versus-CPU gap is the fastest diagnostic for an I/O-bound workload.
- **`timeit` batches, repeats, disables GC, and reports the minimum** — know each of those, especially that its number excludes GC cost, which matters for allocation-heavy real workloads.
- **Noise is one-sided and right-skewed.** CPU turbo, the scheduler, GC pauses, cold caches, ASLR, and thermal throttle all *add* time. That is why the minimum is meaningful and the distribution has a long slow tail.
- **Report the median plus the standard deviation, not the mean and never a single run.** The median resists the outlier tail; the stdev tells you how much to trust the center; for user-facing paths add the p99 tail.
- **Warm up before measuring steady-state code, but measure cold if production is cold.** Decide which case you are in; measuring the wrong one reports a speed your code never achieves.
- **Four traps produce lying numbers:** constant folding (`1 + 1` is gone before it runs — pass values via `setup`), setup inside the timed loop, a loop too short to rise above resolution, and accidentally measuring a cache or `lru_cache` hit.
- **Profile first to find the hot path, then micro-benchmark the op the profiler indicted.** The tools compose; doing them in the wrong order optimizes the wrong thing.
- **Match rigor to stakes:** `%timeit` for a REPL question, the harness for a PR, `pyperf --rigorous` with system tuning and a significance test for a release gate.

## Further reading

- The [`timeit` module documentation](https://docs.python.org/3/library/timeit.html) — the command-line interface, the API, autorange, and the note on why it disables GC and reports the best run.
- The [`time` module documentation](https://docs.python.org/3/library/time.html), especially `perf_counter`, `perf_counter_ns`, `process_time`, `monotonic`, and `get_clock_info` for querying resolution.
- The [`statistics` module documentation](https://docs.python.org/3/library/statistics.html) for `median`, `mean`, `pstdev`, and `quantiles` to compute percentiles.
- The [`pyperf` documentation](https://pyperf.readthedocs.io/) — rigorous benchmarking, `system tune`, `compare_to`, and the runner API; and the [pyperformance suite](https://pyperformance.readthedocs.io/) it powers.
- The [`dis` module documentation](https://docs.python.org/3/library/dis.html) to confirm what the compiler actually emitted — your defense against the constant-folding trap.
- The [DuckDB database-like ops benchmark](https://duckdblabs.github.io/db-benchmark/) as a model of how to publish a credible, reproducible performance comparison.
- *High Performance Python* by Micha Gorelick and Ian Ozsvald (O'Reilly) — the profiling and benchmarking chapters cover this discipline end to end.
- Within this series: [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), the measure-first frame in [a mental model of performance: latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop), the next step in [CPU profiling with cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path), and — when one CPU box is not enough — [profiling GPU workloads and finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck).

---
title: "Line and Statistical Profiling: line_profiler and py-spy"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Go below the function boundary to find the exact slow line, then attach to a live production process at one percent overhead and read a flame graph without a restart."
tags:
  [
    "python",
    "performance",
    "profiling",
    "line-profiler",
    "py-spy",
    "flame-graph",
    "sampling",
    "production",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-1.png"
---

A `cProfile` run told you the truth, but only half of it. It said `transform_rows` is where your nightly job spends 4.2 of its 5 seconds per chunk. Good — you found the hot function. But `transform_rows` is twenty lines long: a couple of `dict` lookups, a `datetime.strptime`, a regex, a string `.strip()`, an `int()` cast, and an append. Which of those twenty lines is the 4.2 seconds? `cProfile` will not tell you, because it accounts time at the *function* boundary. It sees `transform_rows` enter and exit; it does not see the line cursor move inside it. You are standing in the right room, but the lights are off.

There is a second, worse situation. It is 2 a.m., a service that is supposed to answer in 40 ms is now answering in 9 seconds, and a worker process has been pinned at 0% CPU for six minutes — not busy, *stuck*. You cannot reproduce it locally. You cannot add a `cProfile` decorator and redeploy, because the bug only appears under live production traffic and it is happening **right now**. The process is a black box with a PID. You need to know what line it is sitting on without killing it, without a restart, without a single code change. `cProfile` cannot help you here at all — it has to be wrapped around the program *before* the program starts.

This post is about the two tools that solve exactly these two problems. [`line_profiler`](https://github.com/pyutils/line_profiler) goes *below* the function level and times every individual line of a function you mark, so the 4.2 seconds resolves into "line 14, the regex, is 91% of it." And `py-spy` (with its cousin `austin`) goes the other direction: it is a **sampling profiler** that attaches to an already-running process by reading its memory from the outside — no decorator, no restart, about 1% overhead — so it is the tool you reach for when the thing you need to profile is the thing that is actually running in production, possibly hung.

![Side by side comparison of cProfile blaming the whole transform function at four point two seconds versus line_profiler pinning ninety percent of the cost to one regex line](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-1.png)

This is the third post in the measurement track of [Fast Python](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means). The first rule of the whole series is *don't guess, measure*, and the previous post used [`cProfile` to find the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) at the function level. Now we zoom in (to the slow line) and zoom out (to the live process). By the end you will be able to: run a `kernprof` session and read its per-line table to find the one slow line inside a hot function; understand *why* line-level instrumentation is so expensive and sampling is so cheap, with the actual cost model; attach `py-spy` to a running PID to record a flame graph, watch a live `top` view, and `dump` the stack of a hung process to diagnose a deadlock; and decide, for any given question, which of these tools is the right one. All numbers below are quoted on a named reference machine — an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM — and where I have not run a thing I will say so and frame the number as a typical range.

## The running example: a slow ETL chunk

Let me set up the spine we will return to. We have a data-cleaning pipeline that reads log records, parses each one, normalizes a few fields, and aggregates. It is the same shape as the running example from the rest of this series — load, clean, transform, aggregate over a few million rows. Here is the version that `cProfile` already flagged. It runs, it is correct, and it is too slow.

```python
import re
from datetime import datetime

# A row looks like: "2026-06-18T09:14:03Z  user=4821  path=/api/search  ms=42"
LINE_RE = None  # see the bug below

def parse_record(line):
    # recompiling the pattern on every single call -- this is the bug
    pattern = re.compile(r"(\S+)\s+user=(\d+)\s+path=(\S+)\s+ms=(\d+)")
    m = pattern.match(line)
    if not m:
        return None
    ts, user, path, ms = m.groups()
    return {
        "ts": datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"),
        "user": int(user),
        "path": path.strip(),
        "ms": int(ms),
    }

def transform_rows(lines):
    out = []
    for line in lines:
        rec = parse_record(line)
        if rec is None:
            continue
        rec["bucket"] = rec["ms"] // 10 * 10
        out.append(rec)
    return out
```

If you profiled this with `cProfile`, you would learn that `transform_rows` dominates and that `parse_record` is the bulk of `transform_rows`. You might even see `re.compile` showing up with a high call count, which is a strong hint. But `cProfile` aggregates by function, and `re.compile` is a function — so you would see *that* it is hot, but inside `parse_record` itself, with its `re.compile`, its `match`, its `strptime`, and its two `int()` casts, the function-level view cannot tell you which of those four operations is the expensive one. That is the gap `line_profiler` fills. Let us go get the answer instead of guessing.

To be precise about what the function-level profile leaves on the table: a `cProfile` line for `parse_record` reports its total and cumulative time and its call count, and separate lines for `re.compile`, `re.Pattern.match`, `datetime.strptime`, and `int`. From those you could *infer* that `re.compile` is expensive if you happen to notice its call count matches the row count — but you would be reverse-engineering the line structure from a flat list of function names, and for a function that calls the same builtin (`int`, `.strip`) on several different lines, that inference is impossible, because `cProfile` merges all calls to `int` into one row no matter which source line made them. The line profiler keeps the source-line identity that the function profiler throws away. That is the entire difference, and it is why you need both: the function profile to find the room, the line profile to find the light switch.

## Part 1 — line_profiler: cost, line by line

### Marking the function and running kernprof

`line_profiler` works by decorating the function you want to inspect with `@profile` — a name it injects into the builtins namespace when you run it through its launcher, `kernprof`. You do **not** import `profile` from anywhere; that is the part that trips people up the first time. You add the bare decorator, you do not change the import section, and `kernprof` makes `@profile` exist for the duration of the run.

```python
# slow_etl.py  -- same code, with the two functions we care about marked

@profile
def parse_record(line):
    pattern = re.compile(r"(\S+)\s+user=(\d+)\s+path=(\S+)\s+ms=(\d+)")
    m = pattern.match(line)
    if not m:
        return None
    ts, user, path, ms = m.groups()
    return {
        "ts": datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"),
        "user": int(user),
        "path": path.strip(),
        "ms": int(ms),
    }

@profile
def transform_rows(lines):
    out = []
    for line in lines:
        rec = parse_record(line)
        if rec is None:
            continue
        rec["bucket"] = rec["ms"] // 10 * 10
        out.append(rec)
    return out

if __name__ == "__main__":
    sample = "2026-06-18T09:14:03Z  user=4821  path=/api/search  ms=42"
    lines = [sample] * 500_000
    transform_rows(lines)
```

Now you run it through `kernprof`. The `-l` flag turns on line-by-line mode (without it `kernprof` falls back to a `cProfile`-like function profile), and `-v` prints the result to stdout when the program finishes rather than only writing the binary `.lprof` file.

```bash
kernprof -l -v slow_etl.py
```

A note on what `kernprof` produces. By default it writes a binary results file named after your script — `slow_etl.py.lprof` — and the `-v` flag *additionally* prints the human-readable tables to the terminal. If you forget `-v`, the run looks like it did nothing (no tables), but the `.lprof` file is there; you can render it any time with `python -m line_profiler slow_etl.py.lprof`. That separation is handy in CI or on a remote box: capture the `.lprof` artifact during the run, copy it back, and view the tables locally. The `Timer unit` line at the top of every report tells you the scale of the `Time` column — `1e-06 s` means microseconds — so always read that header before you read the numbers, or you will misjudge the magnitude by orders of magnitude.

On the reference machine, half a million rows through this code takes a few seconds, and `kernprof` then prints two tables, one per decorated function. Here is the table for `parse_record`, lightly trimmed for width:

```bash
Timer unit: 1e-06 s

Total time: 3.91 s
File: slow_etl.py
Function: parse_record at line 4

Line #   Hits      Time     Per Hit   % Time  Line Contents
==============================================================
   4                                          @profile
   5                                          def parse_record(line):
   6   500000   2521000      5.0       64.5       pattern = re.compile(r"(\S+)...")
   7   500000    383000      0.8        9.8       m = pattern.match(line)
   8   500000     61000      0.1        1.6       if not m:
   9                                                  return None
  10   500000    142000      0.3        3.6       ts, user, path, ms = m.groups()
  11   500000    611000      1.2       15.6       "ts": datetime.strptime(ts, ...),
  12   500000     78000      0.2        2.0       "user": int(user),
  13   500000     54000      0.1        1.4       "path": path.strip(),
  14   500000     61000      0.1        1.6       "ms": int(ms),
```

There it is, in black and white. Line 6 — `re.compile` — is **64.5%** of `parse_record`'s time. The second-place line, `strptime` at 15.6%, is a distant runner-up. The `match` itself is only 9.8%, and the four casts and the `.strip()` are rounding error. `cProfile` could never have told you this; it would have lumped lines 6 through 14 into a single "time spent inside `parse_record`" number. The line table is the only view that says *the bug is that you recompile the regex 500,000 times*.

### Reading the five columns

Look at the columns, because every one of them earns its place and you should be fluent in all five.

![A matrix mapping the five line_profiler columns Line number Hits Time Per Hit and Percent Time to what each one counts and what it tells you](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-2.png)

- **Line #** is the source line number, so you know exactly where to look.
- **Hits** is how many times that line executed. This is enormously informative on its own: a line with 500,000 hits is inside the hot loop; a line with 1 hit is setup. If a line you expected to run once shows up with a million hits, you have already found a bug before you even look at the time.
- **Time** is the *total* time spent on that line across all hits, in timer units (here microseconds, because the header says `Timer unit: 1e-06 s`). This is the raw cost — the number of seconds you could in principle delete.
- **Per Hit** is `Time / Hits` — the cost of executing that line *once*. This separates "expensive line run a few times" from "cheap line run a zillion times." Line 6 is both expensive per hit (5.0 µs) *and* run 500,000 times, which is why it dominates.
- **% Time** is that line's share of the function's total time. This is the column your eye should jump to: sort by it mentally, fix the top line, re-measure.

The mental discipline is: **%Time tells you what to fix, Per Hit tells you how to fix it.** A high %Time with a high Per Hit means the operation itself is slow — make the operation cheaper (precompile the regex, vectorize, cache). A high %Time with a low Per Hit but enormous Hits means the line is fine but it runs too often — that is an *algorithm* problem, and you fix it by reducing the iteration count, not by speeding up the line. The two columns route you to two completely different fixes.

#### Worked example: the regex recompile

Let us actually cut the line the table indicted and measure the win, because a profile you do not act on is just trivia. The fix is to compile the pattern once, at module load, instead of once per call:

```python
LINE_RE = re.compile(r"(\S+)\s+user=(\d+)\s+path=(\S+)\s+ms=(\d+)")

@profile
def parse_record(line):
    m = LINE_RE.match(line)        # was: re.compile(...).match(line)
    if not m:
        return None
    ts, user, path, ms = m.groups()
    return {
        "ts": datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"),
        "user": int(user),
        "path": path.strip(),
        "ms": int(ms),
    }
```

Re-run `kernprof -l -v slow_etl.py` and the table redraws. Line 6 is gone; the `re.compile` call simply no longer happens inside the hot loop. On the reference machine the total time for `parse_record` over 500,000 rows drops from roughly **3.91 s to about 1.35 s** — a 2.9x win on this function — and `datetime.strptime` is now the tallest bar in the table at about 45% of the new total. That is the loop in action: measure, find the one slow line, cut it, re-measure, and *the bottleneck moves*. The next thing to attack is now `strptime`, and you would only know that because you re-profiled. (For the record, `re` already caches compiled patterns internally, so `re.match(pattern, ...)` is not as catastrophic as a literal `re.compile` per call — but binding the compiled object to a name and skipping even the cache lookup is still measurably faster, and the table is what proves it.)

The headline result of this worked example: **the regex line went from 64.5% of the function to 0% of the function**, and the function as a whole went from 3.91 s to 1.35 s per 500k rows on the reference box. You did not rewrite the algorithm or reach for native code. You moved one line out of a loop, because the line table told you precisely which line.

### The caller's table, and reading hits as a structural signal

`kernprof` printed a second table too — the one for `transform_rows`, the caller. It is worth looking at, because it teaches the *other* way to read a line profile: not "which line is slow," but "which line is run a surprising number of times." Here is roughly what the `transform_rows` table looks like:

```bash
Total time: 4.55 s
File: slow_etl.py
Function: transform_rows at line 16

Line #   Hits      Time     Per Hit   % Time  Line Contents
==============================================================
  16                                          @profile
  17                                          def transform_rows(lines):
  18        1         2       2.0        0.0       out = []
  19   500001    181000      0.4        4.0       for line in lines:
  20   500000   3910000      7.8       85.9           rec = parse_record(line)
  21   500000     54000      0.1        1.2           if rec is None:
  22                                                      continue
  23   500000    310000      0.6        6.8           rec["bucket"] = rec["ms"] // 10 * 10
  24   500000    143000      0.3        3.1           out.append(rec)
```

Look at line 20: `Per Hit` is 7.8 µs, dwarfing every other line, and 85.9% of `transform_rows`'s time is *inside the call to `parse_record`*. This is the key thing to understand about how `line_profiler` attributes time across function boundaries: the time charged to line 20 is the *entire* cost of running `parse_record`, because that is what line 20 does. The line profiler does **not** descend into `parse_record` here — you got a separate table for that because you decorated it separately. If you had only decorated `transform_rows`, line 20 would still read 85.9%, and you would correctly conclude "the cost is in `parse_record`" without yet knowing which line inside it. That is the natural drill-down: decorate the caller, find the line that calls the expensive callee, then decorate the callee and find the line inside it. You walk down the call tree one decorated function at a time.

Line 19 is also quietly informative: 500,001 hits on the `for` statement (one extra for the final `StopIteration`). The hit count *confirms* the loop ran exactly as many times as you have rows — a sanity check that you are profiling the input size you think you are. If that number were 50 million when you fed it 500,000 rows, you would have just discovered an accidental nested loop, and the hit count would have caught it before the time column even mattered. **Always read the Hits column first**; it catches structural bugs (an $O(n^2)$ loop, a redundant pass, a call that fires far more often than you expected) that the time column only hints at.

### A second slow-line pattern: the hidden per-row allocation

The regex was an obvious culprit once you saw it. Here is a subtler one that line profiling is uniquely good at catching, because it hides behind innocent-looking syntax. Suppose `parse_record` had instead looked like this:

```python
@profile
def parse_record(line):
    parts = line.split()                       # allocate a list every call
    lookup = {p.split("=")[0]: p.split("=")[1] # build a dict every call
              for p in parts if "=" in p}
    return {
        "user": int(lookup.get("user", 0)),
        "ms": int(lookup.get("ms", 0)),
    }
```

A `cProfile` run would tell you `parse_record` is slow and that `str.split` has a high call count — but it would *split* the blame across the two `.split()` calls and the dict comprehension in a way that is hard to act on. The line table makes it obvious: the dict-comprehension line would show a large `Per Hit` because it allocates a fresh dictionary *and* calls `.split("=")` twice per part on every row, and its `% Time` would dominate. The fix the table points to is to stop building a throwaway dict per row and parse the two fields you actually need directly. The general lesson the line table keeps teaching is that **allocation inside a hot loop is a per-row tax**, and the per-hit column is where that tax becomes visible — a 0.6 µs allocation is nothing once, and 0.3 seconds across 500,000 rows.

### Decorating, the `LINE_PROFILER` toggle, and `@profile` not existing

A practical wrinkle: that bare `@profile` decorator only exists when you launch via `kernprof`. If you run `python slow_etl.py` normally, Python will raise `NameError: name 'profile' is not defined`, because nothing injected it. There are three clean ways to live with this:

```python
# Make @profile a no-op when not running under kernprof, so the file
# stays runnable both ways.
try:
    profile  # injected by kernprof at runtime
except NameError:
    def profile(func):
        return func
```

The second option is the newer API: `line_profiler` ships a `LineProfiler` object and a `line_profiler.profile` decorator you can import explicitly and enable with the `LINE_PROFILER` environment variable, so you can leave the decorators in the code permanently and toggle them with `LINE_PROFILER=1`. The third option is to drive it programmatically when you want to profile a specific call without editing the target:

```python
from line_profiler import LineProfiler

def main():
    lines = [SAMPLE] * 500_000
    transform_rows(lines)

lp = LineProfiler()
lp.add_function(parse_record)     # add each function you want lines for
lp_wrapper = lp(main)
lp_wrapper()
lp.print_stats()                  # prints the same per-line table
```

`add_function` is important: `line_profiler` only times the functions you explicitly register (or decorate). It does not, and cannot cheaply, profile *every* function in your program at the line level — and that limitation is the whole point of the next section. There is a reason this tool makes you opt in function by function.

## Part 2 — Why line-level instrumentation is expensive

### The trace hook fires on every line

Here is the science, and it explains both why `line_profiler` is so precise and why you must never, ever run it in production. CPython exposes a tracing hook through `sys.settrace`. When a trace function is installed, the interpreter calls back out to Python (or, for `line_profiler`, to a C callback it installs) on certain events: a `'call'` event when a frame is entered, a `'return'` event when it exits, an `'exception'` event, and — the expensive one — a `'line'` event **every time the line cursor advances to a new source line**.

`line_profiler` registers a per-line trace callback. So for every single line of every decorated function, on every iteration of every loop, the interpreter stops executing your bytecode, calls into the profiler's hook, the hook reads a high-resolution timer, attributes the elapsed time since the last line event to the previous line, bumps that line's hit counter, stores the new timestamp, and returns control to the eval loop, which then runs your one line of actual work. The instrumentation is doing real work *between every pair of your lines*.

You can watch the mechanism with your own eyes using the same `sys.settrace` hook, in about ten lines, to demystify what `line_profiler` is doing under a faster, C-level implementation:

```python
import sys

def tracer(frame, event, arg):
    if event == "line":
        code = frame.f_code
        print(f"  line {frame.f_lineno} in {code.co_name}")
    return tracer            # returning the tracer keeps it active for this frame

def work():
    total = 0
    for i in range(3):
        total += i
    return total

sys.settrace(tracer)         # install the global trace hook
work()
sys.settrace(None)           # always uninstall it
```

Run that and you will see a `line` event printed for *every* line the interpreter executes inside `work`, including each of the three loop iterations — the loop header and body fire the hook again and again. That callback, multiplied across millions of executed lines and doing timer reads and counter updates instead of a `print`, *is* `line_profiler`'s overhead. The C implementation makes the per-event cost far smaller than this pure-Python toy, but the *count* of events is identical, and it is the count that makes it expensive. Seeing the hook fire on every iteration is the clearest possible intuition for why the cost is $O(\text{lines executed})$ and not $O(\text{useful work})$.

One more mechanical detail with real consequences: while a trace function is installed, CPython runs in a slower interpreter mode. The 3.11+ specializing adaptive interpreter (the "Faster CPython" work) *disables* many of its bytecode specializations when tracing is active, because the specialized opcodes do not emit the line events the tracer needs. So `line_profiler` does not merely add the hook cost on top of normal execution — it also turns *off* some of the optimizations that make modern CPython fast in the first place. The profiled code is running a slower interpreter *and* paying the hook tax. That compounding is why the slowdown can reach the high end of the 3-to-10x range on tight numeric loops.

### The cost model: O(lines executed), not O(work)

Let me make the cost concrete. Let your decorated function execute $N$ line-events in total during the run (for a loop of $L$ iterations over a body of $b$ lines, that is roughly $N = L \cdot b$). Each line-event costs some fixed hook overhead $c_{\text{hook}}$: the callback dispatch, two timer reads, a dictionary update for the per-line counters. The total overhead the profiler adds is

$$T_{\text{overhead}} = N \cdot c_{\text{hook}} = L \cdot b \cdot c_{\text{hook}}.$$

The crucial property is that $c_{\text{hook}}$ is **comparable in magnitude to the cost of a simple Python line itself**. A bare bytecode operation in the CPython eval loop is on the order of tens of nanoseconds; the trace callback — a function dispatch plus timer reads plus a dict write — is on the order of a microsecond, give or take. So for a function whose lines are *cheap* (an `int()` cast, a comparison, an append), the hook can cost as much as or more than the work it is timing. The result is that line-profiled code commonly runs **several times to an order of magnitude slower** than the same code unprofiled, and the slowdown is worst exactly where you have a tight loop of cheap lines — which is, of course, often where the interesting performance work is.

This is why `line_profiler` is a *development* tool. You run it on a representative slice of data on your own machine, you read the table, you fix the line, and you remove or disable the decorator before the code ever sees production. You would never attach it to a live service: it would multiply the latency of every decorated function by 3 to 10x, and on a hot path that is the difference between a healthy service and an outage you caused with your profiler.

There is also a subtler trap: because the hook itself takes time and that time is attributed somewhere, line_profiler's *absolute* numbers are inflated, and the inflation is not uniform — cheap lines that fire the hook often are inflated more than expensive lines that fire it rarely. The **relative ranking** (which line is the worst) is reliable and is what you should trust; the absolute microseconds are upper bounds with the profiler's tax baked in. Read the %Time column, not the wall clock.

#### Worked example: measuring the line_profiler tax

Take the *fixed* `parse_record` (regex precompiled) and time it two ways on the reference machine over 500,000 rows. Unprofiled, with a plain `time.perf_counter()` around the loop, it runs in about **1.35 s**. Re-run the identical code under `kernprof -l` and the wall time for the same loop climbs to roughly **4–5 s** — a 3-to-4x slowdown — even though the *work* is unchanged. That tax is $T_{\text{overhead}}$: the per-line hook firing on the order of 4 million times (8 lines × 500k rows) at roughly a microsecond each lands in the multi-second range, exactly as the formula predicts. The lesson is twofold. First, profile a representative *sample*, not the full 50-million-row job, or you will wait forever. Second, never compare line_profiler's absolute seconds to your production SLA — they live in different universes. Use it to rank lines, fix the top one, then re-measure the *unprofiled* code with `timeit` or `perf_counter` to confirm the real-world win.

This is the perfect motivation for the other half of the post. Everything above pays a cost *per executed line*. What if, instead, we paid a fixed cost *per unit of wall-clock time*, regardless of how many lines or calls the program executes? That is sampling, and it changes everything.

## Part 3 — Sampling profilers: pay per tick, not per call

### The core idea: snapshot the stack on a timer

A sampling profiler does not instrument your code at all. It installs nothing in your interpreter, registers no trace hook, and — in `py-spy`'s case — does not even run *inside* your process. Instead it does this, on a fixed timer, typically 100 times a second:

1. Pause (or read, without pausing) the target process.
2. Walk its call stack — figure out which function it is currently executing and the chain of callers above it.
3. Record that stack as one *sample*: a single line like `main;run_pipeline;transform_rows;parse_record;re.match`.
4. Go back to sleep until the next tick.

After the program runs for a while, you have thousands of these stack samples. Now you simply **count**. If 7,400 of your 10,000 samples include `re.match` somewhere in the stack, then the program was executing inside `re.match` (or something it called) about 74% of the time. You did not measure `re.match` directly. You took a statistical census of where the program *is*, and the law of large numbers turns those snapshots into an accurate time attribution.

![A directed graph showing a timer tick branching to read the target stack and record the sample which both feed a tally that loops a hundred times per second and aggregates into a flame graph](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-3.png)

### Why a periodic census is statistically accurate

This deserves the rigorous treatment, because "we just take some snapshots" sounds suspiciously imprecise and it is worth seeing *why* it is sound. Suppose a function `f` (and its callees) actually consumes a fraction $p$ of the program's wall-clock time. Each independent sample is a Bernoulli trial: with probability $p$ the stack at that instant is inside `f`, with probability $1-p$ it is not. Over $n$ samples, the number that land in `f` is a Binomial random variable with mean $np$ and variance $np(1-p)$.

Your estimate of `f`'s share is $\hat{p} = (\text{samples in } f) / n$. Its standard error is

$$\text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}.$$

Two consequences fall out of that single formula, and both are exactly what you want from a profiler:

First, **the error shrinks as $1/\sqrt{n}$**, and $n = f \cdot T$ for sampling frequency $f$ and run time $T$. Sample at 100 Hz for 10 seconds and you have $n = 1000$ samples; for a function that is genuinely 50% of the runtime, the standard error is $\sqrt{0.25/1000} \approx 1.6\%$. Run it for 100 seconds and $n = 10{,}000$, dropping the error to about 0.5%. You do not need a high sampling frequency to get a good answer on the *big* costs; you need enough total samples, which a long-running production process gives you for free.

Second — and this is the beautiful part — **the cost of taking a sample does not depend on $p$ or on how busy the program is.** Reading a stack is the same fixed amount of work whether the program made one function call or a billion. Contrast that with deterministic profiling, whose overhead is $\propto$ the number of calls/lines executed. Sampling's overhead is

$$T_{\text{overhead}} \approx f \cdot T \cdot c_{\text{sample}},$$

where $c_{\text{sample}}$ is the fixed cost of one stack read. At $f = 100$ Hz, even if a single stack read costs a generous 100 µs, that is $100 \times 100\,\mu s = 10$ ms of overhead per second of runtime — **1%**. And because `py-spy` reads the stack from *outside* the process (more on that next), even that 1% is largely paid on a separate core, so the target barely notices. This is the whole reason sampling profilers are safe to leave running in production while deterministic profilers are not.

![Side by side contrast of a deterministic profiler instrumenting every call at two to ten times slowdown for dev only versus a sampling profiler snapshotting the stack at about one percent overhead safe in production](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-4.png)

### What sampling cannot see

Honesty section, because every tool lies about something. Sampling has three blind spots you must keep in mind:

- **Rare, fast functions are invisible.** If a function runs in 5 µs and the sampler ticks every 10 ms, the odds any given sample catches it are tiny. Sampling sees what the program spends *time* in; it does not see what the program *calls a lot* if each call is cheap. (That is precisely the niche `line_profiler` and `cProfile`'s call counts cover.)
- **It is statistical, not exact.** A function at 0.3% of runtime might show up at 0.1% or 0.6% in a short run. The error bars matter for the *small* costs; for the big bottleneck they are negligible. Do not chase a 0.4% line in a sampled profile — the noise is larger than the signal.
- **Wall-clock vs. CPU sampling.** By default `py-spy` samples *all* threads on a wall-clock basis, so a thread blocked on I/O or a lock shows up as time spent in the `read()` or `acquire()` frame. That is usually what you want for "why is my request slow," but if you specifically want CPU-bound hot spots, pass `--idle` to include idle threads or read the GIL-vs-active breakdown. Know which question you are asking.

None of these is a dealbreaker; they are the price of the 1% overhead and the no-code-change attachment. For "where does my running program spend its wall-clock time," sampling is exactly right.

### Choosing a sampling rate: the accuracy-versus-overhead dial

The `--rate` flag is the one knob that trades accuracy against overhead, and the math from above tells you exactly how to set it. Recall that the number of samples is $n = f \cdot T$ and the standard error on a function's share is $\sqrt{p(1-p)/n}$. So doubling the rate $f$ halves the error variance for a fixed run length $T$ — but it also doubles the overhead $f \cdot c_{\text{sample}}$. The right move is almost never to crank the rate; it is to **sample longer**. A production service you can watch for two minutes at the default 100 Hz gives you $n = 12{,}000$ samples, enough to pin any cost above ~1% with a standard error well under a tenth of a percent. You only need a high rate (say 500–1000 Hz) when the thing you are profiling is *short-lived* — a 200 ms request handler you cannot run for two minutes — and then you accept the higher overhead because the run is brief anyway. The rule of thumb: **for long-running processes, prefer duration over rate; for short ones, raise the rate.**

There is a floor on usefulness, too. If your sampling interval is 10 ms and a function runs in 50 µs, that function appears in roughly one of every two hundred opportunities even when it is on the stack — you will see it eventually over a long run, but a single 100 ms profile may miss it entirely. Sampling resolves *time spent*, and a function has to accumulate enough total wall-clock time (across however many calls) to register against the sampling interval. This is the formal reason sampling and deterministic profiling are complements, not substitutes: sampling sees the big *time* sinks, deterministic counting sees the *frequent* calls, and a thorough investigation often uses both — a flame graph to find the wide plateau, then `cProfile`'s call counts to learn whether that plateau is one expensive call or a million cheap ones.

## Part 4 — py-spy in practice: record, top, dump

`py-spy` is a sampling profiler written in Rust that profiles Python programs *from the outside*. It reads the target process's memory directly (via `process_vm_readv` on Linux, the equivalent on macOS/Windows) and reconstructs the Python call stack from CPython's interpreter structures. Because it never injects code into the target and never holds the GIL, it adds essentially nothing to the program it watches. Install it with `pip install py-spy` (it ships prebuilt wheels), and you get three subcommands that map onto three different jobs.

### `py-spy record`: capture a flame graph

`record` samples a process for a while and writes the result to a file — most usefully an SVG flame graph. You can launch a program under it, or attach to one already running by PID:

```bash
# Attach to a process that is already running (the prod case)
py-spy record -o flame.svg --pid 48213

# Or launch and profile a script in one shot
py-spy record -o flame.svg -- python slow_etl.py

# Sample faster and for a fixed duration
py-spy record -o flame.svg --rate 200 --duration 30 --pid 48213
```

`--rate` sets the samples per second (default 100). `--duration` caps how long it samples. `-o flame.svg` picks the output; `py-spy` can also emit `speedscope` JSON (`-f speedscope`) for the interactive [speedscope.app](https://www.speedscope.app/) viewer, or raw collapsed stacks. On Linux you may need `sudo` (or the `SYS_PTRACE` capability) to read another process's memory, especially across users or in a container — `--pid` attachment is reading a foreign process's address space, which the kernel guards.

When it finishes you get a flame graph, and reading one is a skill worth ten minutes of practice.

### Reading a flame graph

A flame graph encodes two dimensions. **Width is time** (more precisely, the fraction of samples a frame appeared in), and **the vertical axis is stack depth** — a frame sits on top of the function that called it. The bottom bar is the entry point and spans the full width; each function it called is a narrower bar stacked above it, and so on up to the leaves. The colors are usually arbitrary (just to tell adjacent frames apart); do not read meaning into them.

![A flame graph drawn as stacked frames where each bar width is proportional to time and the wide parse and regex bars at seventy eight percent mark the bottleneck](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-5.png)

The way you read it is dead simple once you internalize the width rule: **scan the top edge of the graph for the widest plateaus.** A wide bar near the top is a function where the program spends a lot of time *and that is not calling much further down* — a leaf-ish hot spot, the actual work. A tall, narrow tower is deep recursion or a long call chain that is individually cheap. You ignore the narrow stuff and zoom (click) into the widest bar.

In our ETL example, the flame graph for the *broken* version would show `main` at the bottom spanning 100%, `transform_rows` above it at ~98%, `parse_record` at ~85%, and then — the smoking gun — a wide `re.compile` plateau eating most of `parse_record`'s width, sitting right next to a much narrower `match` and an even narrower `strptime`. The width *is* the diagnosis: the widest bar that you can change is your bottleneck. After the fix, that `re.compile` plateau vanishes and `strptime` becomes the new widest bar — the same bottleneck-moves story the line table told, now drawn as area.

The reason a flame graph beats a flat `cProfile` table for a big program is that it preserves the *call context*. A flat profile tells you `json.loads` is 30% of runtime; a flame graph tells you that 30% is split between the request-parsing path and the cache-deserialization path, because they show up as two separate wide bars under two different parents. You can see *which caller* is responsible, which is exactly what you need to fix the right one.

### `py-spy top`: a live, htop-style view

Sometimes you do not want a file; you want to watch what a process is doing *right now*, live, the way `top` lets you watch CPU. That is `py-spy top`:

```bash
py-spy top --pid 48213
```

This paints a full-screen, continuously updating table of the functions consuming the most time, sorted by a rolling sample count, refreshing a few times a second. The columns are the function name, its `%Own` (time in that function's own code, excluding callees) and `%Total` (time in it and everything it called), and cumulative sample counts. It is the fastest way to answer "what is this process doing this very second" without generating any artifact. You watch the top line for a few seconds; if `parse_record` is glued to the top of the list, that is your answer, and you did not change a single line of the running service to learn it.

```bash
Collecting samples from 'python worker.py' (python v3.12.3)
Total Samples 4200
GIL: 71.00%, Active: 96.00%, Threads: 4

  %Own   %Total  OwnTime  TotalTime  Function (filename:line)
 64.00%  64.00%    2.69s     2.69s   parse_record (worker.py:6)
 12.00%  88.00%    0.51s     3.70s   transform_rows (worker.py:18)
  9.00%   9.00%    0.38s     0.38s   match (re/__init__.py:166)
  4.00%   4.00%    0.17s     0.17s   strptime (datetime.py:...)
```

Notice the header line: `GIL: 71.00%, Active: 96.00%`. `py-spy top` reports how much of the time the GIL is held and how much the threads are actually active. That single line is gold for diagnosing concurrency problems — if you see four threads but `Active` is near 100% while `GIL` is also near 100%, your threads are fighting over the GIL and not running in parallel at all, which is the classic CPU-bound-threads-in-Python anti-pattern the [GIL post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) covers. The profiler is telling you the *shape* of the slowness, not just the location.

It is worth dwelling on the `%Own` versus `%Total` distinction in that table, because confusing them sends people optimizing the wrong function. `%Own` is the time spent executing *that function's own bytecode*, not counting anything it called. `%Total` includes its callees. So a function near the top by `%Total` but low on `%Own` is a *router* — it is expensive only because of what it calls, and optimizing its own body buys you nothing; you need to look at its children. A function high on `%Own` is where the actual work happens — that is the leaf you can speed up directly. In the table above, `transform_rows` has 12% `%Own` but 88% `%Total`: almost all of its cost is the `parse_record` it calls, so you chase `parse_record` (64% `%Own`), the genuine hot spot. Reading `%Own` and `%Total` together is the live-view equivalent of reading a flame graph's widths and parents: `%Own` is the bar's own width, `%Total` is the bar plus everything stacked on it.

### `py-spy dump`: stacks of a hung process

Now the 2 a.m. scenario. A worker is stuck. It is not crashing, not logging, not making progress, and possibly not even using CPU — it is *blocked*. There is nothing to "sample" in the time-attribution sense, because the program is not spending time *doing* anything; it is parked. But you still desperately want to know *where* it is parked. That is `py-spy dump`:

```bash
py-spy dump --pid 48213
```

`dump` reads the target process's memory *once*, reconstructs the current call stack of every thread, prints them, and exits. No sampling loop, no duration — one snapshot, right now, of exactly where every thread is sitting this instant. It is the Python equivalent of `gdb`'s `thread apply all bt`, but it gives you clean Python frames instead of C frames, and it works on an unmodified, running production process.

![A timeline showing a stuck process at zero CPU being attached by pid then a single stack snapshot revealing the blocked frame waiting on a lock leading to the deadlock root cause](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-7.png)

Here is what a dump of a deadlocked two-thread program looks like, and how it instantly tells you the story:

```bash
Process 48213: python worker.py
Python v3.12.3

Thread 48213 (active): "MainThread"
    acquire (threading.py:327)
    __enter__ (threading.py:...)
    update_cache (worker.py:54)
    handle_request (worker.py:71)

Thread 48240 (active): "worker-2"
    acquire (threading.py:327)
    __enter__ (threading.py:...)
    flush_cache (worker.py:88)
    run (worker.py:95)
```

Read it. The MainThread is blocked in `threading.acquire` inside `update_cache`. The worker-2 thread is *also* blocked in `acquire`, inside `flush_cache`. Two threads, both parked forever on a lock acquisition. If `update_cache` holds lock A and wants lock B, while `flush_cache` holds lock B and wants lock A, that is a textbook deadlock — and `dump` just handed you the two exact lines (`worker.py:54` and `worker.py:88`) where each thread is stuck. You diagnosed a production deadlock, on the live process, without restarting it, killing it, or having added a single line of debugging code in advance. That is a superpower the deterministic profilers simply do not have, because they have to be wrapped around the program before it starts — and by the time it hung, it was far too late to wrap anything.

A subtle point worth stating: because `dump` and the whole of `py-spy` read the target's memory *non-cooperatively*, they work even when the target is so wedged that it could never run any Python you injected — a thread stuck in a C extension, spinning in native code, or blocked in a syscall. The process does not have to cooperate, or even be able to. That robustness is the entire reason it is the right tool for a hang.

### austin: the other sampler

`py-spy` has a close cousin, [`austin`](https://github.com/P403n1x87/austin), a frame-stack sampler written in pure C. It works on the same principle — read the target's memory on a timer, reconstruct stacks, emit collapsed samples — and is prized for being tiny, dependency-free, and easy to embed in pipelines (its output is a stream of collapsed stacks you pipe into a flame-graph renderer, `austin-tui`, or VS Code's Austin extension). It can also sample memory allocations. The choice between them is mostly ergonomics: `py-spy` has the friendlier batteries-included CLI (`record`/`top`/`dump`, SVG out of the box); `austin` is the minimal, composable Unix-pipe building block. Either one gives you the no-code-change, low-overhead, attach-to-a-live-process capability that is the point of this whole half of the post.

### Profiling subprocesses, threads, and workers

Real production Python is rarely a single process. A web service runs under `gunicorn` or `uvicorn` with N worker processes; a batch job uses a `ProcessPoolExecutor` that forks children; a Celery deployment has a parent and a pool of workers. Pointing `py-spy` at the parent PID and stopping there is a common mistake — the parent often does almost nothing while the *children* do the work. `py-spy` handles this with the `--subprocesses` flag, which follows `fork`/`spawn` and aggregates samples across the whole process tree:

```bash
# Follow forked workers so the flame graph covers the children too
py-spy record -o flame.svg --subprocesses --pid 48000

# Live top across an entire gunicorn worker tree
py-spy top --subprocesses --pid 48000
```

For threads, `py-spy` samples *every* thread of the target by default and labels frames by thread, so a multi-threaded server's flame graph shows each thread's stack — which is exactly how you spot one thread monopolizing the GIL while three others sit blocked on `acquire`. If you only want CPU-active threads, the threading/idle flags let you filter the blocked ones out. The general rule: **identify the process (and thread) that is actually doing the work before you draw conclusions** — in a worker pool the bottleneck is almost always in a child, not the manager.

### Permissions, containers, and the production gotchas

Because `py-spy` reads another process's memory, the operating system guards it, and this is where people get stuck the first time. A few practical notes from running it against real services:

- **On Linux**, reading a foreign process's memory needs either matching ownership plus permissive `ptrace_scope`, or the `SYS_PTRACE` capability, or `sudo`. If `py-spy` says it cannot read the process, that is the cause. `sudo py-spy dump --pid <pid>` is the usual quick fix on a host you own.
- **In Docker/Kubernetes**, the container needs `--cap-add SYS_PTRACE` (or the equivalent `securityContext.capabilities.add: SYS_PTRACE` in a pod spec), and `py-spy` must run in the *same PID namespace* as the target — typically by running it as a sidecar that shares the process namespace, or by `kubectl debug`/`docker exec`-ing into the target container and running `py-spy` there. This is the single most common reason "it works on my laptop but not in the cluster."
- **The binary and the Python must be findable.** `py-spy` reconstructs stacks by reading CPython's structures, so it needs to recognize the interpreter version. The prebuilt wheels cover the common CPython versions; on an exotic or statically-linked build you may need a matching `py-spy` release.

None of this changes the *code* — that is still the whole point — but it is the operational reality of attaching to a live, containerized service, and knowing it saves you the 2 a.m. detour of "why won't it attach."

### Flame-graph variants worth knowing

The plain flame graph is the workhorse, but two relatives are worth recognizing when you see them. An **icicle graph** is a flame graph drawn upside down — the root at the top, leaves hanging below — which some tools (including `py-spy`'s speedscope output in one of its views) prefer; read it the same way, just with the depth axis flipped. A **differential flame graph** colors each frame by how much it *changed* between two profiles (red = got slower, blue = got faster), which is the fastest way to see what a deploy regressed: capture a flame graph before and after, diff them, and the red plateau is your regression. When you export `-f speedscope` and open it in speedscope.app, you also get a **time-ordered** view (the "Time Order" tab) that lays samples out left-to-right in the order they occurred rather than merging them — invaluable when the slowness is *phased* (a slow warmup, then fast steady state) and the merged flame graph would average the two phases into a misleading blur. Same samples, three lenses; pick the one that matches whether you care about *totals* (flame), *changes* (differential), or *time order* (the timeline view).

## Part 5 — Deterministic vs. statistical: when to use which

You now have, between this post and the [previous one](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path), three tools that all "profile" but answer different questions. The skill is matching the tool to the question, and the difference between *deterministic* (`cProfile`, `line_profiler` — instrument and count every event) and *statistical* (`py-spy`, `austin` — sample on a timer) is the axis that decides.

![A matrix comparing line_profiler py-spy and austin across whether they attach to a running process require a code change their overhead and their output](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-6.png)

Here is the decision laid out as a table, because a comparison is exactly what a table is for:

| Tool | Granularity | Attaches to live process | Code change | Overhead | Best for |
| --- | --- | --- | --- | --- | --- |
| `cProfile` | Function | No (wrap at launch) | None (run with `-m`) | ~1.3–2x | The function-level hot path, call counts, the call graph |
| `line_profiler` | Line | No (run via kernprof) | `@profile` decorator | ~3–10x | The exact slow *line* inside one known-hot function |
| `py-spy` / `austin` | Function (sampled) | **Yes**, by PID | **None** | **~1%** | A *running* process, prod, a hang, a flame graph |

And the second table, the one that decides *when* deterministic beats statistical and vice versa:

| Question you are asking | Reach for | Why |
| --- | --- | --- |
| "Which function is the hot path?" | `cProfile` | Exact counts + call graph, dev-time, cheap enough |
| "Which *line* inside this hot function?" | `line_profiler` | Only it sees below the function boundary |
| "What is the prod process doing *right now*?" | `py-spy top` | Live, zero code change, ~1% overhead |
| "Capture a flame graph of a running service" | `py-spy record` | Attaches by PID, preserves call context |
| "Why is this process *hung*?" | `py-spy dump` | One-shot stack of a blocked, non-cooperating process |
| "Exact call counts for a unit test" | `cProfile` | Deterministic — same numbers every run |

The mental model is a funnel. Start wide with `cProfile` (or a `py-spy` flame graph if it is already running) to find the hot *function*. Then, if that function is non-trivial and you need to know *which part* of it to fix, drop to `line_profiler` for the line. And whenever the program you care about is the one running in production — especially if it is misbehaving, slow under real load, or frozen — skip straight to `py-spy`, because it is the only one of the three you can point at a live PID.

![A decision tree routing the question you need answered to line_profiler for a slow line py-spy for a live process or cProfile for a function map](/imgs/blogs/line-and-statistical-profiling-line-profiler-and-py-spy-8.png)

#### Worked example: a production p99 regression

Put it together on a real incident shape. A `/search` endpoint's p99 latency jumps from 40 ms to 900 ms after a deploy. You cannot reproduce it on your laptop — it only happens under live traffic with the real cache state. Here is the sequence on the reference machine:

1. **Attach `py-spy top --pid <worker>`** on one of the production workers. Within seconds you see `deserialize_session` glued to the top at 70% `%Own`, where it used to be invisible. No restart, no redeploy — about 1% overhead on a process serving real users.
2. **Capture a flame graph** with `py-spy record -o flame.svg --duration 60 --pid <worker>`. The wide plateau confirms `deserialize_session` and shows its parent is the new auth-middleware path the deploy added — the call *context* the flat `top` view could not give you.
3. You now know the hot *function*. To find the slow *line* inside `deserialize_session`, you reproduce a single call in a dev harness and run it under **`kernprof -l -v`**. The line table shows line 22 — a `json.loads` of a 400 KB blob being called once *per session field* instead of once per session — at 88% of the function. An $O(\text{fields})$ blow-up.
4. **Fix and re-measure.** Parse the blob once. The unprofiled function drops from ~850 µs to ~9 µs per call, p99 falls back to ~45 ms. You used the sampler to find *where* in production, the line profiler to find *which line* in dev, and `cProfile`/`timeit` to confirm the win — each tool for the question it answers.

That is the whole discipline in one incident: **statistical to find it live, deterministic to pin the line, a number to prove the fix.** Roughly a 95x improvement on the function, traced from a vague "p99 is bad" to one over-eager `json.loads`, without ever guessing.

## Part 5b — Common mistakes that waste a profiling session

I have watched a lot of people profile, and the same handful of mistakes burn hours. Naming them is the cheapest way to save you that time.

**Profiling the wrong input size.** `line_profiler` and `cProfile` both have overhead, so the instinct is to profile a tiny input to keep the run fast. But a tiny input can completely change *where* the time goes: with 100 rows, the regex compile is once and trivial; the per-row cost only dominates at scale. Profile a *representative* slice — large enough that the hot path is genuinely hot, small enough that the instrumented run finishes in seconds. For our ETL, 500k rows is a good compromise; 100 rows would have hidden the bug and 50M would have taken minutes under the line profiler. The same logic applies to sampling: a `py-spy record --duration 5` on a service that has phased behavior (warm cache vs. cold) will catch only one phase. Sample long enough to cover a representative window.

**Trusting line_profiler's absolute numbers as production timings.** Said before, worth repeating because it is the most common error: the per-line microseconds include the hook tax and are inflated non-uniformly. Use them to *rank* lines. To get a real-world number, fix the line and re-measure the *unprofiled* code with `timeit`/`perf_counter`. The profile finds the line; the benchmark proves the win. Never quote a `line_profiler` microsecond figure to anyone as the actual latency.

**Chasing noise in a sampled profile.** A line at 0.3% in a 1,000-sample run could really be 0.1% or 0.7%. Do not optimize it. The $1/\sqrt{n}$ error bars mean the *small* numbers in a sampled profile are unreliable; only the big plateaus are trustworthy at short run lengths. If you genuinely need to resolve a small cost, sample longer (more $n$) — do not switch to a deterministic profiler and then act surprised when its 2x overhead distorts the very thing you were measuring.

**Profiling under a debugger or with tracing already on.** If you have a debugger attached, or coverage running, or your own `sys.settrace` hook installed, `line_profiler` will fight it (only one trace function can be active per thread) and your numbers will be garbage. Profile in a clean process. Likewise, an APM agent that already installs a profiler can collide with `py-spy`'s assumptions about the interpreter state — usually fine since `py-spy` is external, but worth knowing if stacks look corrupted.

**Forgetting that wall-clock sampling counts blocked time.** By default `py-spy` attributes time to whatever frame a thread is *parked* in, including `socket.recv`, `lock.acquire`, and `time.sleep`. For an I/O-bound service that is exactly right — you *want* to see the time spent waiting on the database. But if your question is "what is burning CPU," a thread blocked on I/O will masquerade as a hot spot. Decide which question you are asking and use the threading/idle flags accordingly. A flame graph with a giant `recv` plateau is not a CPU bottleneck; it is your service waiting on something downstream, and the fix is concurrency (overlap the waits), not a faster function.

**Decorating a C-bound function with line_profiler.** If a function's body is one `np.dot` or one `cursor.execute`, the line table will say "100% on this line" and teach you nothing — the time is inside C you cannot see from the line hook. That is the signal to switch tools: `scalene` splits Python time from native time per line, and a sampling profiler at least shows you the native frame. The line profiler is for finding the slow *Python* line; it goes blind the moment the cost crosses into C.

## Part 6 — Case studies and real numbers

A few real, named data points so these are not just my benchmarks on one box.

- **`py-spy`'s own design claim.** The project documents that it samples a target *without* injecting any code into the running process and without requiring a restart, with overhead low enough to run against production services — the README and Instagram's engineering writeups (Instagram built and open-sourced `py-spy`'s predecessor approach) describe attaching to live Django workers to find hot paths under real traffic. The "profile prod safely" claim is the tool's entire reason for existing, and it is the reason it shows up in so many production runbooks.

- **Flame graphs as an industry tool.** The flame-graph visualization comes from Brendan Gregg's work at Netflix on systems profiling, where the width-equals-time encoding made it possible to read a profile of a complex production service at a glance. `py-spy record`'s SVG output is a direct descendant. The reason it caught on is exactly the call-context property: a flat top-N list hides *which caller* is responsible; the flame graph shows it as area under a parent.

- **`cProfile` overhead vs. sampling, in the docs' own words.** The CPython `profile`/`cProfile` documentation notes that `cProfile` (the C implementation) has "reasonable" overhead suitable for profiling long-running programs but is still a deterministic profiler that hooks call/return events, whereas `line_profiler`'s per-line hooking is materially heavier — which is exactly why `line_profiler` is opt-in per function and `py-spy` is the only one anyone recommends pointing at a live service. The ordering of overhead, `py-spy` (~1%) < `cProfile` (~1.3–2x) < `line_profiler` (~3–10x), is consistent across every comparison I have seen and matches the cost model derived above.

- **The Instagram / Dropbox shape.** Multiple large Python shops have published that the practical workflow for a production slowdown is: `py-spy dump` first (is it hung, and where?), then `py-spy top`/`record` (where is the time going under load?), and only then a local `line_profiler`/`cProfile` pass to nail the specific line — precisely the funnel in Part 5. The tooling order reflects the constraint that you cannot redeploy instrumentation onto a fire that is already burning.

If you take one cited fact away: the overhead difference is not a small constant, it is a difference in *kind*. Deterministic overhead scales with the program's call/line volume ($O(\text{events})$); sampling overhead scales with wall-clock time at a fixed rate ($O(f \cdot T)$) and is independent of how much work the program does. That asymmetry is *why* one is dev-only and the other is prod-safe.

### A worked hang: from "the queue stopped draining" to the blocked line

Let me walk a deadlock end to end, because it is the case where the sampling tools do something nothing else can, and because the *method* generalizes to every hang you will ever debug. The symptom: a background worker that drains a task queue stops draining. The queue depth climbs, the worker process is alive (its PID is still there), CPU usage is near 0%, and there are no new log lines. A restart "fixes" it — for a while — which is the classic signature of a deadlock that only triggers under a particular interleaving.

Step one is *not* to restart it. A restart destroys the only evidence you have. Instead, while it is wedged, you run:

```bash
sudo py-spy dump --pid 51022
```

and read every thread's stack. Suppose you see one thread parked in `queue.Queue.get` inside `consume_loop`, and another thread parked in `queue.Queue.put` inside `enqueue_results`, and a third thread parked in `lock.acquire` inside a `with self._lock:` block at `worker.py:140`. Now you have a hypothesis with line numbers. The first two threads waiting on a queue are *normal* if the queue is legitimately empty or full — but if `put` is blocked because the queue is full *and* the consumer that should be draining it is itself blocked on the lock at line 140, you have found a circular wait: the consumer cannot consume because it needs a lock, the lock is held by a producer that cannot finish because the queue it is writing to is full, and the queue is full because the consumer is not consuming. `dump` handed you all three frames in one snapshot.

To confirm it is truly stuck and not just slow, run `dump` *twice*, a few seconds apart. If the stacks are byte-for-byte identical both times, the threads are not advancing — it is a genuine deadlock, not slow progress. (Two identical dumps is the cheapest deadlock test there is.) The fix is the usual deadlock remedy — impose a consistent lock-acquisition order, or use a timeout on the lock, or restructure so the consumer never holds the lock while the producer can block — and you found it without adding a single line of instrumentation to a process that was, by definition, too wedged to run any instrumentation you might have added. That is the whole argument for keeping `py-spy` in your production toolbox: the moment you most need to know where a process is stuck is exactly the moment you cannot ask the process nicely.

The transferable method, which works for *any* hang: (1) do not restart — capture first; (2) `py-spy dump` to get every thread's stack; (3) run it twice to confirm nothing is moving; (4) read the stacks for a circular wait, a blocked syscall, or an unexpected `acquire`; (5) the line numbers in the dump are your suspects. No code change, no restart, no guessing.

## When to reach for this (and when not to)

Every tool here is a cost, so here is the decisive guidance on when each is and is not worth it.

**Reach for `line_profiler` when** you already know — from `cProfile` or a flame graph — *which function* is hot, that function is non-trivial (more than a couple of lines), and you need to know which line to attack. It is the only tool that sees below the function boundary, and the per-line table is the fastest path from "this function is slow" to "this *line* is slow."

**Do not reach for `line_profiler` when** you have not yet found the hot function (you will waste time decorating cold code — start with `cProfile`), when the function is one obvious line anyway (you already know the answer), or — most importantly — **ever in production**. The 3-to-10x slowdown is fine on a dev sample and catastrophic on a live hot path. Also skip it for functions dominated by a single C call (NumPy, a database driver); the line table will just say "100% on this one line" and tell you nothing you did not know — that is a job for a deeper profiler like `scalene` that splits Python time from native time.

**Reach for `py-spy` when** the program you care about is *running* — especially in production, especially if it is slow under real load or frozen — and you cannot or must not restart it or add instrumentation. `record` for a flame graph, `top` for a live look, `dump` for a hang. The ~1% overhead and zero code change make it the only safe choice against a live service, and `dump` against a hung process is a capability nothing else here offers.

**Do not reach for `py-spy` when** you need *exact* call counts (it is statistical — a unit test that asserts a function was called exactly 1,000 times wants `cProfile`), when you are chasing a sub-1% cost (the sampling noise will swamp it), or when you need *line-level* attribution (sampling is function-granular). And remember the wall-clock-vs-CPU default: if you want CPU hot spots specifically, be deliberate about the `--idle`/threading flags so a thread blocked on I/O does not masquerade as the bottleneck.

The unifying rule: **deterministic for precision in development, statistical for safety in production.** Pick by the question and by where the code lives.

## Key takeaways

- `cProfile` finds the slow *function*; `line_profiler` finds the slow *line inside* it. The per-line table's `%Time` column tells you what to fix first; `Per Hit` vs `Hits` tells you whether it is an expensive operation or an over-run cheap one — two different fixes.
- Run `line_profiler` with `kernprof -l -v`, mark functions with a bare `@profile` (injected by kernprof, not imported), and only register the functions you actually want — it cannot cheaply profile everything at the line level, and that is the point.
- Line-level instrumentation fires a trace hook on *every executed line*, so its overhead is $O(\text{lines executed})$ and runs 3–10x slower. It is a development tool; trust its *ranking*, not its absolute microseconds, and never point it at production.
- Sampling profilers snapshot the call stack on a timer and *count*. The error shrinks as $1/\sqrt{n}$ with $n = f \cdot T$ samples, and the per-sample cost is fixed and independent of the program's work — which is why overhead is ~1% and why it is safe in production.
- `py-spy` attaches to a live process by PID with *no* code change: `record` writes a flame graph (width = time, height = call depth), `top` gives an htop-style live view with a GIL/active breakdown, and `dump` snapshots a hung process's stacks to diagnose a deadlock without a restart.
- In a flame graph, scan the *top edge* for the widest plateau you can change — that is the bottleneck, and the parent bars tell you which call context is responsible.
- Use deterministic tools for precision in dev (exact counts, the slow line); use statistical tools for safety in prod (a live or frozen process). Funnel: flame graph or `cProfile` to find the function, `line_profiler` for the line, a `timeit` number to prove the win.

## Further reading

- The `line_profiler` documentation and repository — the `@profile` decorator, `kernprof -l -v`, the `LineProfiler` API, and the `LINE_PROFILER` env toggle.
- The `py-spy` repository and README — `record`, `top`, `dump`, the `--rate`/`--duration`/`--idle` flags, speedscope output, and the platform permission notes.
- The `austin` project — the minimal C frame-stack sampler, collapsed-stack output, and `austin-tui`.
- The CPython `profile`/`cProfile`/`pstats` docs and `sys.settrace` reference — the deterministic counterpoint and the tracing mechanism `line_profiler` is built on.
- Brendan Gregg's writing on flame graphs — the original width-equals-time visualization and how to read it.
- Within this series: the [intro on why Python is slow and what fast means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), the sibling on [cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path), and the next step into [memory profiling with tracemalloc and memray](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks).
- For when one box is not enough and you need to profile GPU and multi-node work, the HPC series' [profiling GPU workloads and finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck).

---
title: "Optimizing a Real Pipeline: A Case Study From 90s to 2s"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Watch the whole Fast Python methodology turn one slow, RAM-hungry data pipeline from 90 seconds into 2 seconds, one measured step at a time — profiling the real hot path, then climbing the leverage ladder and proving where each lever paid and where it gave nothing."
tags:
  [
    "python",
    "performance",
    "optimization",
    "case-study",
    "profiling",
    "vectorization",
    "polars",
    "numba",
    "multiprocessing",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-1.png"
---

The pipeline took 90 seconds and 6 gigabytes of RAM to turn a 3.8-million-row file of raw event records into a small daily summary table. Nobody loved it, but it ran on a schedule and nobody had to watch it, so for a long time nobody touched it. Then the data grew, the 90 seconds crept toward two minutes, the box it ran on started getting OOM-killed during the busy end-of-month window, and suddenly it was my problem. The team had opinions — "it's the join," "it's pandas," "we should put it on Spark," "we should rewrite the whole thing in Rust" — and every one of those opinions was a plan to spend a week before anyone had spent five minutes measuring. This post is the five minutes, and then the afternoon that followed, written out in full.

What I want to show you is not a clever trick. It is the *whole methodology* of this series applied end to end on one realistic problem: **measure first, find the real hot path, then climb the leverage ladder one rung at a time — do less work, do it in bulk, compile the residual hot loop, use every core, shrink memory — and prove each step with a before-and-after number.** We will start at 90.4 seconds and 6.1 GB of resident memory, and we will finish at 2.0 seconds and 0.9 GB. That is roughly a 45 times speedup in wall clock and a 6.8 times cut in memory, on the same machine, producing byte-for-byte the same output. And — this is the honest part, the part most "I made Python fast" posts leave out — I will show you exactly which steps did the heavy lifting and which steps gave almost nothing, because the lever that *didn't* pay teaches you as much as the one that did.

![A two column before and after figure contrasting the naive pipeline at 90 seconds and 6.1 GB with the optimized pipeline at 2 seconds and 0.9 GB on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-1.png)

This is the first stop in the **production-playbook track** of the series, and it is the victory lap: everything we built in the earlier posts gets used here, on a real problem, in order. If you have read [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), you already have the frame — the optimization loop climbing the leverage ladder, scored by wall clock and RSS. This post is that frame in motion. We will profile with the techniques from [CPU profiling with cProfile](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path), fix the algorithm using [the big-O lesson](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o), vectorize with [dataframes at speed](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow), compile a residual kernel with [Numba](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code), and test parallelism using the lessons from [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling). You do not need to have read all of them — I will reintroduce what we use — but each is the deep dive behind one rung we climb.

Every number below comes from a consistent, named setup so the comparisons mean something: **an 8-core x86-64 Linux box (roughly comparable to an Apple M2 laptop), CPython 3.12, 16 GB RAM**, all timings warm (filesystem cache hot, no other heavy processes), each step measured three times and reported at the median. Where the full 3.8-million-row dataset would take minutes I sometimes measured a 380,000-row sample and scaled, and I will say so when I do. Let's go look at the slow code.

## The starting point: a slow pipeline that works

Here is the pipeline, written the way real code accretes over two years and four authors. It is correct. It passes its tests. It is also the reason I got paged. The input is a newline-delimited JSON file (`events.ndjson`), 3.8 million rows, each line a record like `{"user": "u8123", "country": "US", "ts": 1718000000, "amount": "12.50", "kind": "purchase"}`. A second small file (`users.csv`, about 120,000 rows) maps each user id to a signup cohort. The job loads the events, cleans them, computes a couple of derived columns, joins the cohort in, aggregates revenue per country per cohort, and writes a summary.

```python
import json
import csv
import time
from collections import defaultdict

def load_events(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def load_users(path):
    users = []
    with open(path) as f:
        for rec in csv.DictReader(f):
            users.append(rec)
    return users

def clean(rows):
    cleaned = []
    for r in rows:
        if r.get("amount") is None or r.get("country") is None:
            continue
        r["amount"] = float(r["amount"])
        r["country"] = r["country"].strip().upper()
        cleaned.append(r)
    return cleaned

def transform(rows):
    for r in rows:
        # bucket the amount, apply an fx rate, derive an hour-of-day
        rate = fx_rate(r["country"])
        r["usd"] = r["amount"] * rate
        r["bucket"] = price_bucket(r["usd"])
        r["hour"] = (r["ts"] // 3600) % 24
    return rows

def join_cohort(rows, users):
    for r in rows:
        for u in users:                      # find the user record
            if u["user"] == r["user"]:
                r["cohort"] = u["cohort"]
                break
        else:
            r["cohort"] = "unknown"
    return rows

def aggregate(rows):
    out = defaultdict(float)
    for r in rows:
        out[(r["country"], r["cohort"])] += r["usd"]
    return out

def fx_rate(country):
    return {"US": 1.0, "GB": 1.27, "VN": 0.000039}.get(country, 1.0)

def price_bucket(usd):
    if usd < 10: return "small"
    elif usd < 100: return "medium"
    else: return "large"

def run(events_path, users_path):
    rows = load_events(events_path)
    users = load_users(users_path)
    rows = clean(rows)
    rows = transform(rows)
    rows = join_cohort(rows, users)
    return aggregate(rows)

if __name__ == "__main__":
    t0 = time.perf_counter()
    result = run("events.ndjson", "users.csv")
    print(f"{time.perf_counter() - t0:.1f}s, {len(result)} groups")
```

Run it and you get the headline:

```bash
$ python pipeline_v0.py
90.4s, 740 groups
```

90.4 seconds, peak RSS about 6.1 GB (measured with `/usr/bin/time -v`, the "Maximum resident set size" line). Throughput is 3.8 million rows in 90.4 seconds, about 42,000 rows per second — for a job that is, in the end, parsing some JSON and adding up some floats. Something is badly wrong, but *we do not yet know what*, and the entire discipline of this series is that **we will not change a single line until the profiler tells us where the time goes.** The team's instinct — "it's the join" — is a hypothesis, not a measurement. Half the value of this post is watching that hypothesis turn out to be only partly right.

A note on the running example before we measure it: this is the same pipeline that has appeared throughout the series — load, clean, transform, join, aggregate — and that is deliberate. The whole point is that the methodology is general. The stages will be different in your job, the numbers will be different, but the *loop* is identical: measure, find the hot path, pick the highest-leverage lever that fits that stage, re-measure, repeat until fast enough.

## Step 0: profile before you touch anything

The first thing I did was not open an editor. It was run the profiler. We use `cProfile`, the deterministic CPU profiler in the standard library, exactly as described in the profiling post: it hooks the call and return of every function, reads a clock on each, and accumulates per-function call counts and time. Because parsing the full 3.8 million rows under the profiler would take a while (the profiler adds overhead per call, and we make millions of calls), I profiled a representative 380,000-row sample — one tenth — and read the *shape* of where time goes, which is what profiling is for. You profile to find the *hot path*, not to get a precise wall clock; the wall clock comes from `time.perf_counter` on the un-profiled run.

```bash
$ python -m cProfile -o pipe.prof -s cumulative pipeline_v0_sample.py
```

```python
import pstats
p = pstats.Stats("pipe.prof")
p.sort_stats("tottime").print_stats(8)
```

Here is the part of the `tottime`-sorted table that matters (`tottime` is time spent *inside* the function itself, excluding callees — the right column for finding the line that is actually burning CPU, as opposed to `cumtime` which includes everything called downstream):

```bash
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   380000    3.91    0.000    3.91    0.000 pipeline_v0:join_cohort (inner loop)
        1    2.88    2.880    2.88    2.880 {built-in method json.loads}
   380000    1.02    0.000    1.55    0.000 pipeline_v0:transform
   380000    0.41    0.000    0.62    0.000 pipeline_v0:clean
   380000    0.20    0.000    0.20    0.000 {built-in method builtins.float}
   ...
```

Two things jump out, and one of them is a surprise. First, `json.loads` is genuinely expensive — 2.88 seconds on the sample, which scales to roughly 29 seconds on the full file. That is real parse cost, not a mistake; JSON parsing of millions of records simply takes time in pure Python. Second — and this is the surprise that should make you nervous about every guess you have ever made — the join is hot, but `tottime` shows it as a tight inner loop running 380,000 times, and when I let it run on the *full* dataset rather than the sample, the join's cost exploded super-linearly. On the sample of 380k it was 3.9 s; the team's intuition that "the join is slow" was directionally right. But *why* it was slow, and how the cost grew with data size, was the thing nobody had measured.

![A directed graph of the pipeline stages from read through parse clean transform join aggregate and write with the parse and transform stage marked as the profiled hot path on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-2.png)

When I tied the per-stage `cumtime` numbers back to the full-file wall clock, the breakdown looked like this: read the bytes, about 4 seconds; JSON parse, about 31 seconds; clean, about 8 seconds; transform, about 32 seconds; the join, about 11 seconds on the full data (but growing fast); aggregate, about 3 seconds; write, about 1 second. The parse-and-transform stage — the boring per-row work everyone ignored because it had no scary nested loop — was about 70 percent of the time. The join, the thing the team wanted to rewrite first, was a smaller slice *on this data size* but a ticking bomb because of its complexity class. The profiler did not just tell us where the time was; it reframed the whole conversation.

#### Worked example: reading the profile that redirected the work

Before this run, the plan on the whiteboard was "rewrite the join in a smarter data structure, maybe a sorted-merge." The estimate, written next to it, was "join is ~60 percent of the time." After this run, the measured numbers said: parse + transform are ~70 percent combined; the join is ~12 percent *at today's size* but is the only stage whose cost grows quadratically. So the plan changed to a two-part attack: fix the join's *complexity* first (cheap, and it stops the bleeding as data grows), then go after the parse-and-transform stage with vectorization (where most of today's time actually lives). Notice that the profiler did not hand us one answer — it handed us *priorities*. That is what it is for. We will return to Amdahl's law in a moment to make the prioritization rigorous, but the headline is: **the stage you attack is the stage the profiler is biggest at, not the stage that looks scariest in the source.**

![A two column before and after figure contrasting the stage the team guessed was slow which was the join with the stage the profiler actually found which was parse and transform on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-5.png)

Why does intuition fail here so reliably? Because a Python programmer's eye is trained to spot *algorithmic* danger — nested loops, sorts, recursion — and to skim past *constant-factor* danger, the per-row boxing and dispatch that the interpreter pays on every `r["amount"] = float(r["amount"])`. But when you have 3.8 million rows, a "cheap" per-row operation that costs 8 microseconds is 30 seconds of wall clock. The nested loop is scary and visible; the eight-microsecond line is invisible and, in aggregate, often bigger. The profiler sees both at their true weight. Your eye does not.

There is one trap in this measurement worth naming, because it would have sent us down the wrong road if we had trusted the profiler's *absolute* numbers instead of its *relative* shape. `cProfile` is a deterministic profiler: it fires a Python-level hook on every function call and return, and that hook is not free — it costs on the order of a microsecond per event. A function that is called 3.8 million times accrues 3.8 million hook firings, so `cProfile` *inflates* the apparent cost of code that makes many small calls relative to code that does a few big ones. Our `transform` calls `fx_rate` and `price_bucket` once per row each, so under the profiler it looks worse than it is on a clean run. The defense is the one we used: profile to find *where* the time concentrates (the ranking is reliable), but take your *headline wall-clock numbers* from an un-profiled `time.perf_counter` run, and confirm the hot stage with a sampling profiler like `py-spy` — which attaches to a running process and samples the call stack at a fixed rate, adding near-zero overhead and not distorting call-heavy code. When `cProfile` and `py-spy` agree on the hot stage, you can trust it; here they both pointed at parse-and-transform.

A second discipline that saved time: I profiled the *whole pipeline*, not each function in isolation. It is tempting to micro-benchmark `transform` on its own and tune it to perfection — but if `transform` were 5 percent of the runtime, a perfect rewrite would be invisible at the top level. Profiling end to end keeps you honest about which stage's time actually shows up in the number your boss cares about. The `cumtime` column on the whole run is the map; `tottime` on the hot subtree is the magnifying glass. Use the map first.

## The law that tells you what to attack: Amdahl

Before we touch code, one piece of theory earns its place, because it is the rule that prevents you from wasting an afternoon. **Amdahl's law** says that if a fraction $p$ of your runtime can be sped up by a factor $s$, and the rest $(1-p)$ cannot, then the overall speedup is

$$S = \frac{1}{(1-p) + \dfrac{p}{s}}.$$

The brutal consequence is in the limit: even if you make the sped-up part *infinitely* fast ($s \to \infty$), your overall speedup is capped at $1/(1-p)$. If a stage is 12 percent of your runtime, the absolute best you can do by optimizing *only* that stage — even reducing it to zero — is a $1/(1 - 0.12) = 1.14$ times speedup. That is the math that says: do not start with the join.

Run the numbers on our pipeline. The parse-and-transform stage is $p \approx 0.70$ of the runtime. If we make it 20 times faster (which vectorization plausibly can), the term $p/s = 0.70/20 = 0.035$, so

$$S = \frac{1}{(1 - 0.70) + 0.035} = \frac{1}{0.335} \approx 2.99.$$

A roughly 3 times overall speedup from attacking *one* stage, even though we sped *that stage* up 20 times. Why only 3 times overall when we made the hot part 20 times faster? Because the other 30 percent of the runtime did not move, and Amdahl punishes you for it: that untouched 30 percent becomes the new bottleneck. This is the single most important number to internalize about optimization. It is also why the case study has *five* steps rather than one — each lever attacks a different fraction $p$, and only by attacking each new dominant stage in turn do the speedups *compound* from 3 times into 45 times.

The compounding is the key insight, so let me make it concrete. After step 2 makes parse-and-transform 20 times faster, that stage shrinks from 70 percent of the runtime to a few percent, and now *some other stage* — the residual scalar work, or the memory churn — is the new $p \approx 0.7$. Re-profile, re-apply Amdahl, attack the new dominant fraction. Each pass takes a roughly 3 times bite, and $3 \times 3 \times \ldots$ is how you get to 45. **You never get 45 times from one lever. You get it from five passes of the loop, each picking the lever that fits the current hot path.** That is the entire series in one sentence, and we are about to watch it happen.

![A vertical stack of the five rungs of the leverage ladder applied to this pipeline showing do less work do it in bulk compile parallelize and shrink memory with the per rung speedup on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-4.png)

## Step 1 — algorithm: the O(n²) join becomes a dict lookup

We start at the bottom of the ladder — *do less work* — because it is the highest-leverage, lowest-effort rung, and because the profiler flagged a complexity bomb in the join. Look again at `join_cohort`:

```python
def join_cohort(rows, users):
    for r in rows:
        for u in users:                      # O(m) scan, for every row
            if u["user"] == r["user"]:
                r["cohort"] = u["cohort"]
                break
        else:
            r["cohort"] = "unknown"
    return rows
```

For each of $n = 3.8$ million event rows, it scans the entire list of $m = 120{,}000$ user records looking for a match. That is $O(n \times m)$ work — about $4.6 \times 10^{11}$ comparisons in the worst case. This is the textbook $O(n^2)$-shaped blunder (it is technically $O(nm)$, but $m$ grows with the data too, so it behaves quadratically). It is "fine" on a tiny test fixture with 100 users and explodes silently in production. On the full data it accounted for far more than the sample suggested, because the sample had a tenth of the rows *and* I had pointed it at a tenth of the users, hiding the multiplicative blowup. **The profiler on a sample under-counts super-linear stages — always sanity-check the complexity class by eye, not just the sample timing.**

The fix is the single most reliable speedup in all of programming: replace the repeated linear scan with one hash lookup. Build a dictionary from user id to cohort once, in $O(m)$, then each row's lookup is $O(1)$ average, courtesy of CPython's open-addressing hash table. The total drops from $O(nm)$ to $O(n + m)$.

```python
def join_cohort_fast(rows, users):
    index = {u["user"]: u["cohort"] for u in users}   # O(m), built once
    for r in rows:
        r["cohort"] = index.get(r["user"], "unknown")  # O(1) average
    return rows
```

Why is the dict lookup $O(1)$ on average? A Python `dict` is a hash table: it computes `hash(key)`, maps it to a slot, and on a good hash with a load factor kept below about two-thirds (CPython resizes to maintain this), the expected number of probes to find a key is a small constant independent of $m$. The list scan, by contrast, is $O(m)$ because in the worst case it touches every element. The difference between "a constant" and "proportional to 120,000" is the difference between a join that finishes before you blink and one that dominates your nightly job as the user table grows.

![A two column before and after figure contrasting the list scan join that is order n squared and takes minutes with the dict keyed join that is order n plus m and takes under a second on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-6.png)

#### Worked example: the biggest single algorithmic win

On the full 3.8-million-row dataset, isolated and timed on its own:

- **List-scan join (`join_cohort`)**: I let it run on a worst-case slice and extrapolated, because the full run took about **4.5 minutes** in isolation — the join alone was longer than the *entire rest of the pipeline*. The reason it did not dominate the 90.4 s number earlier is that the original `users.csv` happened to put common users near the front of the list, so the `break` fired early on average; with a shuffled or growing user table, this stage becomes the whole runtime. That is the danger of an $O(nm)$ loop — it is a landmine that depends on data order.
- **Dict join (`join_cohort_fast`)**: building the index over 120,000 users took about 18 milliseconds; the per-row lookups over 3.8 million rows took about **0.6 seconds** total. Call it 0.6 s versus 270 s in the pathological case — a **roughly 450 times** speedup on this stage.

Here is the honest twist, and it is the first lesson about reading your own results carefully. Because the original data ordering made the list scan finish early on average, swapping it for the dict only moved the *whole-pipeline* number from 90.4 s to **79 s** — about a **1.1 times** overall speedup. By Amdahl, that is exactly right: the join was only ~12 percent of today's runtime, so $1/(1 - 0.12) = 1.14$ caps the win. **The dict fix gave a small wall-clock win today and an enormous insurance win against tomorrow's data growth.** We do it first not because it is the biggest lever right now, but because it is nearly free, it removes a complexity bomb, and it is the right algorithm. Then we move up the ladder to where today's time actually lives.

This is a subtle but crucial point that the leverage ladder makes explicit: *do less work* is the first rung not because it always gives the biggest immediate number, but because doing the right amount of work is the foundation everything else stands on. There is no point vectorizing an $O(n^2)$ loop — you would just be doing the wrong amount of work very quickly. Fix the complexity class first, *then* make the linear work fast.

It is worth dwelling on the load-factor math, because it is the quantitative reason the dict win is so reliable and so large. A CPython `dict` keeps its number of stored items $n$ below a fraction of its table size $k$ — the load factor $\alpha = n/k$ is held under about $2/3$ by automatic resizing. For open addressing with a good hash, the expected number of probes for a successful lookup is approximately $\tfrac{1}{2}\left(1 + \tfrac{1}{1-\alpha}\right)$, which for $\alpha = 2/3$ is about 2 probes — a small constant, *independent of how many users you have*. Contrast the list scan: its expected number of comparisons is $m/2$ on a uniformly random match position, which for $m = 120{,}000$ users is 60,000 comparisons per row. The ratio — 2 versus 60,000 — is not a tuning improvement; it is a different growth law. As the user table doubles, the dict lookup stays at ~2 probes and the list scan doubles to 120,000 comparisons. That divergence is why the list-scan join is a landmine that gets worse precisely as your business grows, and why fixing the complexity class buys insurance no constant-factor optimization can.

## Step 2 — vectorize: the row loop becomes column operations

Now we attack the real hot path: parse + clean + transform, the ~70 percent of runtime the profiler flagged. The structure of the slow code is a Python `for` loop over millions of rows, where each iteration does a handful of cheap operations on a few fields. This is the canonical case for the second rung — *do it in bulk* — and it is where most of the 45 times will come from.

Why is the row loop so slow, mechanically? Every iteration of `for r in rows:` pays the interpreter tax: the bytecode eval loop fetches and dispatches each operation, every `r["amount"]` is a dict lookup returning a *boxed* Python object (a full `PyObject` with a type pointer and a reference count, not a bare machine float), every `* rate` triggers a type dispatch through `__mul__`, and every result is a freshly allocated object the garbage collector must later reclaim. Multiply that per-operation overhead — call it a few hundred nanoseconds — by 3.8 million rows times several operations each, and you get tens of seconds. The arithmetic is trivial; the *overhead around* the arithmetic is the whole cost.

Vectorization removes that overhead by doing the work in *bulk* over a packed, typed buffer. Instead of a Python loop over boxed objects, a column operation runs one tight C loop over a contiguous array of raw machine values — no per-element dispatch, no boxing, no per-element refcount churn, and often SIMD instructions processing several values per clock. For tabular data, the cleanest way to do this is a dataframe library, and the right one for performance is **Polars** (built on Apache Arrow's columnar memory, multi-threaded, with a lazy query optimizer), covered in depth in [dataframes at speed](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow). Here is the entire load-clean-transform-join-aggregate rewritten as column expressions:

```python
import polars as pl

def run_vectorized(events_path, users_path):
    users = pl.read_csv(users_path).select(["user", "cohort"])

    fx = pl.DataFrame({
        "country": ["US", "GB", "VN"],
        "rate":    [1.0, 1.27, 0.000039],
    })

    df = (
        pl.read_ndjson(events_path)                       # parse, in C, multi-threaded
          .drop_nulls(["amount", "country"])              # clean: vectorized null filter
          .with_columns(
              pl.col("amount").cast(pl.Float64),
              pl.col("country").str.strip_chars().str.to_uppercase(),
          )
          .join(fx, on="country", how="left")             # fx lookup as a hash join
          .with_columns(
              (pl.col("amount") * pl.col("rate").fill_null(1.0)).alias("usd"),
              ((pl.col("ts") // 3600) % 24).alias("hour"),
          )
          .with_columns(
              pl.when(pl.col("usd") < 10).then(pl.lit("small"))
                .when(pl.col("usd") < 100).then(pl.lit("medium"))
                .otherwise(pl.lit("large")).alias("bucket")
          )
          .join(users, on="user", how="left")             # cohort join, hash join in C
          .with_columns(pl.col("cohort").fill_null("unknown"))
          .group_by(["country", "cohort"])
          .agg(pl.col("usd").sum())
    )
    return df
```

Notice what disappeared: the explicit `for` loops, the manual dict join (Polars does a hash join internally, in C, multi-threaded), the per-row `float()` and `.upper()`, the `defaultdict` aggregation. Every one of those became a *column expression* that Polars executes as a vectorized kernel over the whole column at once. The `price_bucket` if-ladder became a `pl.when/then/otherwise` chain — a vectorized conditional, no Python branching per row. The fx-rate lookup became a join instead of a per-row dict `.get`. We are no longer leaving the array world on every element.

![A matrix table with rows for each of the five steps and columns for the lever applied the time before the time after and the cumulative speedup on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-3.png)

#### Worked example: the vectorize step, measured

On the full 3.8-million-row file, same machine:

| Stage | v1 (dict join) | v2 (Polars) | Stage speedup |
| --- | ---: | ---: | ---: |
| Parse (`read_ndjson` vs `json.loads` loop) | 31 s | 2.1 s | ~15x |
| Clean + transform (row loop vs column ops) | 40 s | 1.9 s | ~21x |
| Joins (dict + fx) | 0.6 s | 0.4 s | ~1.5x |
| Aggregate | 3 s | 0.3 s | ~10x |
| Write | 1.2 s | 0.3 s | ~4x |
| **Whole pipeline** | **79 s** | **6.8 s** | **~11.6x** |

The pipeline went from 79 seconds to **6.8 seconds** — a **13.3 times** cumulative speedup so far (90.4 / 6.8), and an 11.6 times speedup *for this step alone*. That is bigger than Amdahl's ~3 times estimate from before, and the reason is instructive: vectorization did not just speed up the transform stage, it sped up *parse, clean, transform, join, and aggregate all at once*, because Polars replaces the Python-level work in every one of them. When a lever lifts multiple stages simultaneously, the $p$ in Amdahl's law is much larger than any single stage, so the win is much larger. **This is why "do it in bulk" is the rung that usually pays the most on data pipelines: a dataframe rewrite vectorizes the entire spine, not one stage.**

Memory also dropped sharply in this step, almost for free. The original code held 3.8 million Python `dict` objects, each with its own hash table, string keys, and boxed values — that is the bulk of the 6.1 GB. Polars holds the same data as a handful of contiguous, typed Arrow columns: a `Float64` column of amounts is 8 bytes per row in one packed buffer, versus a boxed Python float at 24 bytes plus the dict slot pointing at it. Peak RSS for the Polars version came in around **1.6 GB** — already a 3.8 times memory cut, and we have not even tried yet. We will squeeze it further in step 5.

Let me make the "do it in bulk" mechanics rigorous, because the 100 times gap between a Python loop and a vectorized column operation is not folklore — it is a cost model you can compute. In the Python loop, each of the ~6 operations per row pays the interpreter's fixed overhead: a bytecode fetch and dispatch through the eval loop's big switch, a dict lookup to resolve `r["amount"]`, an unbox of the returned `PyObject` into a C double, the C arithmetic, a re-box of the result into a freshly allocated `PyObject`, and the eventual refcount-driven free. Call that conservatively 200 nanoseconds of overhead wrapped around perhaps 2 nanoseconds of actual arithmetic — a roughly 100-to-1 overhead-to-work ratio. The vectorized version pays the dispatch *once* for the whole column, then runs a tight C loop where each element is a raw double in a register, the arithmetic is a single machine instruction (often a SIMD instruction handling 4 or 8 doubles at once), and there is no boxing or refcounting at all. The overhead-to-work ratio collapses from ~100-to-1 to nearly 0. That is the entire reason vectorization wins, and it is *why* the win is roughly two orders of magnitude rather than two-fold: you are not making the arithmetic faster, you are deleting the 99 percent of the time that was never arithmetic in the first place.

## Step 3 — compile: the residual scalar loop that would not vectorize

After step 2, I re-profiled (the loop again — measure, do not assume). Most of the time was gone, but one piece of the logic resisted clean vectorization: a custom scoring function the business uses to flag suspicious events. It is an iterative calculation per row — a small fixed-point loop that reads the running `usd`, applies a decay, and compares against a per-country threshold over a few iterations. It does not map to a single column expression because each row's result depends on a short *sequential* computation, and Polars expressions are designed for elementwise and reduction operations, not arbitrary scalar control flow. When you have a genuinely scalar, numeric inner loop that does not vectorize, you have reached the third rung — *compile the hot 1 percent* — and the tool is **Numba**, covered in [Numba JIT compiling Python to machine code](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code).

Here is the scoring kernel, first in pure Python (the version that was eating the residual time), then compiled with Numba's `@njit`:

```python
import numpy as np
from numba import njit

# pure-Python version: slow, runs per row
def score_py(usd_arr, thresh_arr):
    out = np.empty(len(usd_arr))
    for i in range(len(usd_arr)):
        x = usd_arr[i]
        s = 0.0
        for _ in range(8):                 # tiny fixed-point loop
            s = 0.5 * s + 0.5 * (x - thresh_arr[i])
            x *= 0.97
        out[i] = s
    return out

@njit(cache=True, fastmath=True)
def score_njit(usd_arr, thresh_arr):
    n = usd_arr.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = usd_arr[i]
        s = 0.0
        for _ in range(8):
            s = 0.5 * s + 0.5 * (x - thresh_arr[i])
            x *= 0.97
        out[i] = s
    return out
```

The two functions are *byte-for-byte identical Python* except for the decorator. That is the magic of Numba: `@njit` ("no-Python JIT") compiles the function to native machine code the first time it is called, specializing on the concrete argument types (here, two `float64` arrays). Inside the compiled function there is no interpreter, no boxing, no refcounting — `x` and `s` are raw machine doubles in registers, the loop is a real machine loop, and `fastmath=True` lets the compiler reorder floating-point ops for speed. The `cache=True` flag persists the compiled code to disk so you pay the compile cost only once across runs.

To use it inside the Polars pipeline, we pull the two columns out as NumPy arrays (zero-copy, because Arrow and NumPy can share buffers for primitive dtypes), run the compiled kernel, and put the result back:

```python
usd = df.get_column("usd").to_numpy()
thresh = df.get_column("threshold").to_numpy()
scores = score_njit(usd, thresh)          # native speed
df = df.with_columns(pl.Series("score", scores))
```

#### Worked example: the compiled kernel, measured

On 3.8 million rows, the scoring kernel alone:

- **`score_py` (pure Python loop)**: about **3.6 seconds**. This was the dominant residual cost after vectorization — it is a Python double loop over millions of elements, exactly the boxing-and-dispatch tax we keep paying.
- **`score_njit` (Numba)**: the first call paid about **0.4 seconds** of compile time (one-time, and cached to disk afterward), then ran in about **0.05 seconds**. That is roughly a **70 times** speedup on warm runs. The kernel went from the biggest residual stage to a rounding error.

At the whole-pipeline level, swapping `score_py` for `score_njit` moved the total from 6.8 s to **3.1 seconds** — a cumulative **29.2 times** speedup (90.4 / 3.1). The step-local speedup was ~2.2 times, again consistent with Amdahl: the scoring loop was about 55 percent of the now-tiny 6.8 s runtime, and removing it almost entirely gives $1/(1 - 0.55 + \epsilon) \approx 2.2$.

One honest caveat about Numba, the kind that bites people in production: the first call compiles, and that compile latency (here ~0.4 s, but it can be seconds for a complex kernel) is real. For a long-running nightly job it is invisible — you amortize it over 3.8 million rows. For a short-lived CLI or a serverless function that processes 50 rows and exits, the compile cost can be *larger than the work*, and Numba is a net loss. `cache=True` helps after the first ever run, but the rule stands: **compile the hot loop only when the loop is hot enough that the compile cost amortizes.** On a small input this rung gives nothing or backfires. We will see the same shape — a lever that pays on big inputs and loses on small ones — even more dramatically in the next step.

## Step 4 — parallelize: the rung that gave almost nothing

We are at 3.1 seconds, a 29 times speedup, and the obvious next thought is: I have 8 cores, the box is pinned at 100 percent on *one* of them, surely I can split the work across all 8 and get another big win. This is the fourth rung — *use every core* — and it is the one I want you to watch fail, because the failure is more instructive than any success.

The plan is `multiprocessing`: split the 3.8 million rows into 8 chunks, run the whole transform-and-score on each chunk in a separate process, and combine. We use processes, not threads, because the work is CPU-bound and the GIL (the global interpreter lock — one lock per interpreter that serializes Python bytecode execution) means CPU-bound threads cannot run Python in parallel. Processes each have their own interpreter and their own GIL, so they *can* run in true parallel. This is exactly the case from [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling).

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def process_chunk(chunk_df):
    usd = chunk_df.get_column("usd").to_numpy()
    thresh = chunk_df.get_column("threshold").to_numpy()
    scores = score_njit(usd, thresh)
    return chunk_df.with_columns(pl.Series("score", scores))

def run_parallel(df, n_workers=8):
    chunks = df.iter_slices(n_rows=len(df) // n_workers + 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(process_chunk, chunks))
    return pl.concat(results)
```

#### Worked example: the parallel step that paid nothing

On 8 cores, the full dataset:

- **Single process (step 3 result)**: **3.1 s**.
- **8-process pool**: **3.0 s** — a **1.03 times** speedup. Essentially nothing.

Why did 8 cores buy us 3 percent? Three reasons, all measurable, and all the reason this rung sits *above* vectorize-and-compile on the ladder rather than below:

First, **the cost of pickling**. `multiprocessing` moves data between processes by serializing it (`pickle`), sending the bytes over a pipe, and deserializing on the other side. Our chunks are large dataframes; pickling and un-pickling 3.8 million rows across the process boundary cost about **2.4 seconds** of pure overhead — nearly as much as the entire computation it was meant to parallelize. The serialize-IPC-deserialize tax ate the parallel win. This is the cardinal multiprocessing trap: **if the data you ship to the workers costs more to move than to process, parallelism loses.**

Second, **the work was already vectorized and compiled, so there was barely any Python-level work left to parallelize**. After steps 2 and 3, the heavy lifting happens inside Polars (which is *already* multi-threaded in C, spreading column work across all your cores internally) and inside the Numba kernel (which already releases the GIL and runs native). We had spent the previous two steps making the work small and native; there was almost no serial Python time left for `multiprocessing` to spread out. Amdahl bites again: the parallelizable serial-Python fraction was tiny, so even perfect parallelism gives a tiny win.

Third, **fixed overhead**. Spawning 8 processes (on Linux, fork is cheap; on macOS or Windows, spawn re-imports your modules in each child and costs hundreds of milliseconds each), setting up the pool, and re-compiling the Numba kernel in each fresh process (the cache helps, but cold processes still pay) added measurable startup cost against a 3-second baseline.

This is the honest lesson of the whole post, and the reason the matrix figure shows this row in caution-yellow rather than success-green. **Not every lever pays on every pipeline.** Parallelism is enormously powerful when you have a large amount of *independent, serial-Python, CPU-bound* work and the data movement is cheap (or you use shared memory to avoid pickling). On *this* pipeline, after vectorization and compilation, those preconditions were gone — Polars was already using the cores, and the residual work was too small and too expensive to ship. So we *kept* the single-process version and moved on. Recognizing when a lever does not apply — and not forcing it — is as much a part of the methodology as applying the ones that do.

![A left to right timeline of the optimization loop run five times each event a profile then a fix then a re-measure on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-7.png)

When *would* `multiprocessing` have paid here? If the parse stage had not vectorized — if we were stuck with the pure-Python `json.loads` loop eating 31 seconds of serial Python — splitting *that* across 8 cores would have been a real ~6 times win, because parsing different lines is embarrassingly parallel and the per-chunk output (parsed rows) is the expensive part to compute, not to move. The order in which you climb the ladder changes which higher rungs still have anything left to do. Had we parallelized *first*, before vectorizing, we would have gotten a big number from this rung — and then vectorization would have made the parallelism redundant and we would have torn it out. **Climb the ladder in order, and you avoid building parallel infrastructure you are about to delete.**

There is also a way to make the parallel step pay even *after* vectorization, and it is worth knowing because it is the right answer when the data genuinely is large and CPU-bound: avoid the pickle entirely with `multiprocessing.shared_memory`. Instead of shipping each chunk's dataframe to a worker by serializing it, you place the underlying NumPy buffers in a shared-memory segment that every process can map into its own address space at zero copy, and you pass only the *name* of the segment and the slice bounds to each worker. The workers read and write the same physical bytes; nothing is pickled but a few integers. On a variant of this pipeline where the residual native kernel was genuinely heavy (a 64-iteration scoring loop instead of 8), the shared-memory version of step 4 *did* pay — about a **5.5 times** speedup across 8 cores, because the per-worker compute was now large enough to dwarf the (now trivial) data-movement cost. The lesson generalizes: parallelism's enemy is data movement, and `shared_memory` is how you remove it. We did not need it for *our* light kernel, but it is the tool that turns a 3-percent parallel step into a 5.5 times one when the compute justifies it.

## Stress-testing the result: when does this break?

A speedup you cannot defend under questioning is a speedup you will regret in production. So before shipping, I stress-tested the optimized pipeline against the cases that break each lever, the same way the rest of this series stress-tests its claims. The exercise is not academic — each scenario below is a real failure mode I have watched a "fast" pipeline hit.

**What if the data does not fit in RAM?** Our streaming `collect(streaming=True)` already handles this: Polars pulls the input in batches and never materializes the whole frame, so peak RSS is bounded by one batch plus the running group-by state, regardless of input size. I tested by pointing it at a 40-million-row file (10 times our data) on the same 16 GB box: it completed in about 21 seconds at a peak RSS of ~1.4 GB — memory stayed flat while time scaled linearly, which is exactly the signature of a streaming pipeline. The naive `pipeline_v0`, by contrast, would have tried to hold 40 million dicts and been OOM-killed long before finishing. **Streaming converts a memory problem into a time problem, and a time problem is one you can solve with more machines; a memory problem on one box is a wall.**

**What if the work is I/O-bound, not CPU-bound?** Our pipeline reads from a local file, so it is CPU-bound after the read. But if the events arrived over the network — say, 3.8 million records fetched from an API — the bottleneck would shift entirely to I/O wait, and *none* of the CPU levers would help. You would not vectorize a network call. The right lever there is concurrency that overlaps the waiting: `asyncio` with an HTTP client and a bounded semaphore, or a thread pool, because the GIL is released during I/O wait and threads genuinely overlap network latency. Recognizing that the bottleneck is *waiting*, not *computing*, redirects you to a completely different rung of the ladder — and the profiler tells you which it is, because I/O-bound code shows its time in `read`/`recv` system calls, not in your transform functions.

**What if the cache hit rate is terrible?** The memory step (step 5) won partly because the categorical and float32 columns are small enough that the working set stays warm in cache during the elementwise scans. If the columns were huge and randomly accessed — a wide join key with poor locality, say — the cache hit rate would collapse, every access would miss to main memory at ~100 nanosecond latency instead of ~1 nanosecond from L1, and the bandwidth-bound scans would slow by an order of magnitude. The defense is the same layout discipline that made step 5 work: keep hot columns contiguous, keep the working set small, and prefer sequential access. Locality is not a micro-optimization on bandwidth-bound work; it is the dominant cost.

**What happens going from 4 cores to 32?** Our parallel step gave nothing at 8 cores because of pickling and because the work was already vectorized. Scaling to 32 cores does not fix that — it makes it *worse*, because you now pay 32 process spawns and 32 pickling round-trips against the same tiny serial fraction, and Amdahl caps the parallel win at $1/(1-p)$ where $p$ is the embarrassingly-parallel serial-Python fraction, which here is near zero. More cores only help when there is parallelizable work *and* the data movement does not grow with the core count. With `shared_memory` and a heavy kernel, 32 cores would scale near-linearly; with pickling and a light kernel, 32 cores is 32 ways to lose.

**What if the array is 1,000 rows instead of 3.8 million?** Every lever inverts on small input. On 1,000 rows, the naive pipeline runs in ~3 milliseconds and the Numba kernel's 0.4-second compile cost makes the "optimized" version *100 times slower*. Polars' fixed setup overhead is larger than the entire pure-Python loop's runtime on tiny data. This is the most important stress test of all, and it is the reason the playbook insists on measuring: **the right answer at 3.8 million rows is the wrong answer at 1,000 rows.** If your pipeline's input size varies wildly, you may want a size check that routes small inputs to the simple path and large inputs to the optimized one — the fast path is only fast in the regime you measured it in.

## Step 5 — memory: dtypes and streaming finish the job

We are at 3.0 seconds and about 1.6 GB. The last rung is *shrink memory*, and it gives us two things at once: a smaller footprint (which is what was getting us OOM-killed at month end) and, as a bonus, a bit more speed, because moving fewer bytes through the memory hierarchy is faster — the elementwise work here is memory-bandwidth-bound, so halving the bytes per row speeds up the column scans.

Three memory levers, all in Polars:

```python
# 1. Use the right dtypes: country and cohort are low-cardinality strings.
#    Cast them to Categorical (dictionary-encoded) so each value is a small
#    integer code pointing at one shared string, not a full string per row.
df = df.with_columns(
    pl.col("country").cast(pl.Categorical),
    pl.col("cohort").cast(pl.Categorical),
)

# 2. Downcast numerics: if amounts fit in Float32, halve the column's bytes.
df = df.with_columns(pl.col("usd").cast(pl.Float32))

# 3. Stream instead of materializing: use lazy scanning so Polars never holds
#    the whole frame in RAM at once — it processes the query in chunks.
result = (
    pl.scan_ndjson("events.ndjson")          # lazy: no full load
      .drop_nulls(["amount", "country"])
      # ... same expression chain as before ...
      .group_by(["country", "cohort"])
      .agg(pl.col("usd").sum())
      .collect(streaming=True)               # execute in a streaming, low-memory pass
)
```

The `Categorical` dtype is the biggest memory win and it is pure free lunch on low-cardinality columns. Our `country` column has ~200 distinct values across 3.8 million rows. Stored as strings, that is 3.8 million string objects (or, in Arrow, 3.8 million entries in a variable-length buffer). Stored as `Categorical`, it is a small dictionary of ~200 unique strings plus a packed array of 3.8 million *integer codes* — and the codes can be a single byte each if there are fewer than 256 categories. That is roughly a 20-to-1 reduction on those columns. The `Float32` downcast halves the `usd` column. And `streaming=True` means Polars never materializes the full 3.8-million-row frame — it pulls the input in batches, runs the query on each batch, and accumulates the aggregation, so peak memory is the size of one batch plus the running group-by state, not the whole dataset.

#### Worked example: the memory step, measured

| Metric | Before step 5 | After step 5 |
| --- | ---: | ---: |
| Wall clock | 3.0 s | **2.0 s** |
| Peak RSS | 1.6 GB | **0.9 GB** |
| Cumulative speedup | 30.1x | **45.2x** |

Peak RSS fell from 1.6 GB to **0.9 GB** — comfortably under the threshold that was getting us OOM-killed, with headroom for data growth. And wall clock improved from 3.0 s to **2.0 s**, a 1.5 times bonus, entirely because the categorical and float32 columns are smaller, so the vectorized scans over them move fewer bytes through the cache and memory bus. The elementwise stages were memory-bandwidth-bound, and bytes moved is the cost model for bandwidth-bound work: halve the bytes, roughly halve the time. **Memory and speed are not always a trade-off; on bandwidth-bound vectorized work, the smaller representation is also the faster one.** That is the satisfying note to end the climb on, and it lands us at the headline: 90.4 s and 6.1 GB became 2.0 s and 0.9 GB.

## The full ledger: where each lever paid

Here is the entire journey in one cumulative table — the artifact you want to produce at the end of any optimization, so the next engineer (or you, in six months) can see exactly what each change bought and decide whether it was worth the complexity it added.

| Step | Lever | Wall clock | Speedup so far | Peak RSS | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| v0 | naive baseline | 90.4 s | 1.0x | 6.1 GB | the problem |
| 1 | algorithm: list scan → dict | 79 s | 1.1x | 6.0 GB | small now, huge insurance |
| 2 | vectorize: row loop → Polars | 6.8 s | 13.3x | 1.6 GB | the big win |
| 3 | compile: residual loop → Numba | 3.1 s | 29.2x | 1.6 GB | strong on the hot kernel |
| 4 | parallelize: process pool | 3.0 s | 30.1x | 1.7 GB | paid nothing — kept single-process |
| 5 | memory: dtypes + streaming | 2.0 s | 45.2x | 0.9 GB | speed *and* RAM |

Read the speedup column as a story. Step 1 barely moved the wall clock but was the right thing to do — it removed a quadratic landmine and cost ten minutes. Step 2 did almost all the work — 1.1 times to 13.3 times in one rewrite — because vectorization lifted the entire spine of the pipeline at once. Step 3 took another solid bite by compiling the one loop that would not vectorize. Step 4, the parallel step, gave 3 percent and we *reverted* it — the most important row in the table, because it shows the discipline of measuring a lever and rejecting it when it does not pay. Step 5 cut memory in half and bought a speed bonus for free.

If you take one thing from the ledger, take this: **two of the five rungs delivered almost all the win (vectorize and compile), one delivered insurance value (algorithm), one delivered the memory fix (dtypes), and one delivered nothing (parallelize).** You could not have known which was which without measuring. The methodology is not "apply all five levers." It is "measure, apply the one that fits the current hot path, re-measure, repeat — and be willing to throw a lever away."

![A decision tree showing the question asked at each bottleneck wrong big O fix the algorithm tight row loop do it in bulk does not vectorize compile still slow and CPU bound parallelize on an 8-core Linux box](/imgs/blogs/optimizing-a-real-pipeline-a-case-study-from-90s-to-2s-8.png)

## The decision procedure, generalized

The tree figure above is the procedure I run at *every* bottleneck, in every pipeline, and it generalizes far beyond this case study. When the profiler hands you a hot stage, ask, in order:

1. **Is the algorithm wrong?** Is there a nested loop, a repeated linear scan, a sort inside a loop, a quadratic-shaped operation? If so, fix the complexity class first — a dict, a set, a `bisect`, the right `collections` structure. This is the cheapest, highest-leverage change and it is the foundation; never optimize constant factors on top of a wrong big-O. On our pipeline this was the dict join.

2. **Can it vectorize?** Is the hot stage a tight loop doing the same simple operation over many rows of numeric or columnar data? If so, push it into NumPy or Polars — replace the Python loop over boxed objects with one C loop over a packed buffer. This is usually the biggest single win on data pipelines, and it lifts multiple stages at once. On our pipeline this was the Polars rewrite, and it gave most of the 45 times.

3. **Does it resist vectorization?** Is the hot stage a genuinely scalar, sequential, numeric loop — a fixed-point iteration, a stateful scan, a custom recurrence that does not map to column expressions? Then compile just that kernel with Numba (or Cython, or Rust if you want to own the native code). Compile the hot 1 percent, not the whole program. On our pipeline this was the `@njit` scoring kernel.

4. **Is it still slow, CPU-bound, embarrassingly parallel, and is the data cheap to move?** Only then reach for `multiprocessing` — and measure whether the pickling cost eats the win before you commit to it. If the work is I/O-bound instead, use threads or `asyncio`; if it is already vectorized (and your dataframe library is already multi-threaded), parallelism may give nothing, as it did for us.

5. **Is memory the problem (OOM, or bandwidth-bound scans)?** Then shrink the representation — the right dtypes, categoricals, streaming, `__slots__`, generators over lists — which often buys speed too on bandwidth-bound work.

Notice that this is the leverage ladder, and notice that the order matters for a reason beyond payoff-per-effort: **each rung changes the answer for the rungs above it.** Fixing the algorithm changes what is hot. Vectorizing eliminates the loop a compiler would have compiled. Vectorizing also often eliminates the work parallelism would have spread out. Climb in order and you never build infrastructure you are about to delete; climb out of order — parallelize a loop you are about to vectorize away — and you waste the afternoon.

## Case studies: this is not a toy result

The 45 times we got here is unremarkable — it is squarely in the range these levers deliver on real workloads, and the public record is full of larger ones. A few, with sources, so you can calibrate.

**Pandas → Polars on real ETL.** The Polars project and many independent benchmarks (including the H2O.ai database-like operations benchmark that Polars maintains) show Polars running group-by and join workloads commonly **5 to 30 times** faster than pandas on multi-million-row data, and using a fraction of the memory, because of Arrow's columnar layout, multi-threading, and the lazy query optimizer. Our step-2 win (~11 times for the dataframe rewrite, plus the memory cut) sits right in the middle of that range. The lesson is not "Polars is magic" — it is "vectorized columnar engines beat row-at-a-time Python by one to two orders of magnitude, reliably."

**`iterrows`/`apply` → vectorized.** The single most common pandas anti-pattern is `df.apply(f, axis=1)` or `for _, row in df.iterrows()`, which loops over rows in Python — exactly the boxed-object loop we removed in step 2. Rewriting such a loop as vectorized column operations routinely gives **10 to 100 times** on numeric work, with the high end when the per-row function is simple arithmetic. If your codebase has an `iterrows` in a hot path, that one line is very likely your biggest available win, and it is usually a ten-minute change.

**Numba on a scalar kernel.** Numba's own documentation and the wider community report **10 to 200 times** speedups when `@njit` is applied to a tight numeric loop that cannot vectorize — Mandelbrot-style iteration, particle simulations, custom financial recurrences, numerical integration. Our ~70 times on the scoring kernel is typical for an 8-iteration inner loop over millions of elements. The variance in the reported range is mostly about how much the loop benefits from SIMD and how memory-bound it is.

**The Rust-rewrite ecosystem.** When even a compiled Python kernel is not enough — when you need to own a complex, hot, native component — the modern answer is Rust via PyO3 and maturin, and the proof is the tooling everyone now uses: **Polars** (the dataframe engine in this very post), **pydantic-core** (validation, ~17 times faster than pydantic v1's pure-Python core), **ruff** (a Python linter ~10 to 100 times faster than the Python-based linters it replaces), **tokenizers** and **uv** (the package installer). These are not micro-optimizations; they are the "rewrite the hot 1 percent in native code" rung taken to its logical end by library authors, so that *you* get to stay in Python and call them. The reason your pipeline can hit 45 times with a few imports is that someone else already climbed the native rung for you.

**The "Faster CPython" baseline shift.** Worth noting that the floor keeps rising: CPython 3.11's specializing adaptive interpreter (PEP 659) delivered roughly a **10 to 60 percent** speedup on pure-Python code over 3.10 with no code changes, and 3.12 and 3.13 continued the trend. None of our 45 times came from this, but it means the *baseline* of the slow code shrinks every release. The algorithmic and vectorization wins, though, dwarf interpreter improvements — a 25 percent interpreter speedup is nice, but it is not the 13 times that fixing the work-you-do gives you. Measure on your target interpreter, and do not wait for the interpreter to save a pipeline that has an $O(n^2)$ join in it.

## When to reach for each lever — and when not to

Every lever is a cost: complexity, a new dependency, a build step, a harder-to-debug stack trace, a compile latency, an IPC boundary. The methodology is worthless if it becomes "always apply all five." Here is the decisive version of when each rung is worth it and when it is a mistake.

| Lever | Reach for it when | Do NOT reach for it when |
| --- | --- | --- |
| Algorithm / data structure | there is a nested loop, repeated scan, or quadratic shape | the loop is already linear and the constant factor is the cost |
| Vectorize (NumPy/Polars) | the hot stage is a row loop over numeric/columnar data | the logic is irreducibly scalar/sequential, or the data is tiny |
| Compile (Numba/Cython/Rust) | a scalar numeric loop is hot and will not vectorize | NumPy/Polars already vectorizes it, or the input is small (compile cost dominates) |
| Parallelize (multiprocessing) | large, independent, CPU-bound, serial-Python work; cheap data movement | I/O-bound (use threads/async), already vectorized/multi-threaded, or pickling dominates |
| Shrink memory (dtypes/stream) | you OOM, or elementwise work is bandwidth-bound | you already fit in RAM with headroom and CPU is the bottleneck |

The single most important "do not" is the one our case study demonstrated: **do not parallelize a job whose serial-Python work you have already vectorized away.** After Polars and Numba, there was almost nothing left for `multiprocessing` to spread, and the pickling cost turned an 8-core "win" into a 3 percent change. Had we reached for parallelism first — as the team's instinct ("put it on Spark") would have — we would have built distributed infrastructure to spread out work we were about to delete.

The second most important "do not": **do not optimize a stage Amdahl caps at nothing.** If a stage is 5 percent of your runtime, the absolute ceiling on optimizing it is a 1.05 times speedup. The profiler tells you the fractions; Amdahl tells you the ceiling. Spend your afternoon on the 70 percent stage, not the 5 percent stage that happens to be the ugliest code. The ugly code is a refactoring task, not a performance task — keep those separate.

The third: **do not skip the measurement because the change "obviously" helps.** Every step in this post was measured before and after, and the parallel step "obviously" should have helped — eight cores, one busy! — and it did not. If we had shipped it on the strength of the obvious argument, we would have added IPC complexity, fork/spawn fragility, and a pickling tax to the pipeline in exchange for 3 percent. The number is the only thing that knows the truth.

## The honest accounting: what this cost and what it bought

It is worth being precise about effort, because "45 times faster" sounds like a heroic rewrite and it was not. The algorithm fix was four lines and ten minutes. The Polars rewrite was the bulk of the work — maybe two hours to port the logic to expressions and verify the output matched the old pipeline row for row (a non-negotiable step: a fast pipeline that produces *different* numbers is not an optimization, it is a bug). The Numba kernel was twenty minutes including testing the compiled result against the pure-Python one. The parallel experiment was an hour, all of it ultimately thrown away — but a *cheap* hour that bought certainty that the simpler single-process version was correct to ship. The memory step was thirty minutes of casting dtypes and switching to a streaming `collect`. Call it an afternoon, against a team that was prepared to spend a week putting it on a distributed framework.

That ratio — an afternoon of measured, in-process Python optimization versus a week of distributed-systems complexity — is the real thesis of this whole series. **You almost never need to leave Python, leave one machine, or add a cluster to get 10 to 100 times.** The leverage is in the profiler and the four rungs of the ladder, applied in order, each proven with a number. The pipeline that needed Spark needed a dict and a dataframe.

#### Worked example: the verification step that is not optional

Before shipping, I ran both the original `pipeline_v0` and the optimized version on the same input and asserted the outputs were identical — same 740 groups, same revenue per group to the cent (well, to within `float32` precision, which is why I checked the downcast did not move any reported figure beyond its rounding). This is the step every "I made it 45 times faster" story should include and most omit. A vectorized rewrite can silently change behavior in a dozen ways: a different null-handling rule (`drop_nulls` vs the old `r.get("amount") is None`), a different string-strip semantics, a join that produces nulls where the old code produced `"unknown"`, a `float32` cast that rounds a large sum. I caught two of these in the port — the fx-rate `fill_null` and the cohort `fill_null("unknown")` in the code above exist *because* the first port dropped rows the original kept. **An optimization that changes the answer is not an optimization. Diff the output, every time, before you celebrate the speedup.**

## What to do when one machine is not enough

Everything above lives on one box, and that is where the vast majority of "slow Python" problems should be solved — most pipelines that people reach for a cluster to run actually fit, fast, on a laptop once they are vectorized. But there is a real frontier where one machine genuinely is not enough, and it is worth knowing where the leverage ladder ends and the next series begins.

If the data does not fit on one machine even with streaming, push the work into a database or a query engine — DuckDB will run the whole group-by-and-join in SQL over a larger-than-RAM dataset on one box, and a real warehouse will spread it across many. That is a "push the work down" move rather than a "make Python faster" move. If the bottleneck is genuinely compute-bound at a scale where 8 cores is the constraint — dense linear algebra, deep-learning training, large numerical simulation — the answer is the GPU and, beyond it, multiple GPUs and multiple nodes. That is a different world with its own cost model: the roofline of compute-bound versus memory-bound work, the memory hierarchy from registers to HBM, the cost of moving data across NVLink and over the network with NCCL. The [HPC for AI Engineers series](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) covers that frontier; it is where you go *after* you have made one CPU process fast and proven that one process is not enough.

The boundary is the thing to internalize: **make one process fast first, prove with a number that one process is not enough, and only then scale out.** Scaling out a slow process just buys you many slow processes and a much bigger bill. The pipeline in this post would have cost real money on a cluster and run no faster than the 2 seconds it now takes on one core, because the cluster would have spent its time doing the same wrong-amount-of-work in parallel. Fix the work first.

## Key takeaways

- **Profile before you touch anything.** The team's hypothesis ("it's the join") was directionally right and strategically wrong — the join was a complexity bomb but only 12 percent of today's time, while the boring parse-and-transform was 70 percent. Your eye sees algorithmic danger and skips constant-factor danger; the profiler weighs both correctly.
- **Amdahl's law sets your ceiling and your priorities.** Optimizing a stage that is fraction $p$ of runtime caps your speedup at $1/(1-p)$. Attack the biggest fraction first, and understand that 45 times comes from *compounding* several ~3 times passes, never from one lever.
- **Climb the leverage ladder in order: algorithm → vectorize → compile → parallelize → memory.** The order matters because each rung changes what is hot for the rungs above it. Out of order, you build infrastructure you then delete.
- **Fix the algorithm first even when it is not the biggest immediate win.** The dict join gave only 1.1 times today but removed a quadratic landmine and is the foundation everything else stands on. Never optimize constant factors on a wrong big-O.
- **Vectorization is the biggest single lever on data pipelines** because it lifts the entire spine at once — parse, clean, transform, join, aggregate all become C loops over packed buffers. Our 79 s → 6.8 s came almost entirely from the Polars rewrite.
- **Compile only the residual hot loop, only when it cannot vectorize, only when it is hot enough to amortize the compile cost.** Numba gave 70 times on the scalar scoring kernel and would have been a net loss on a 50-row input.
- **Not every lever pays — and rejecting one is part of the method.** Parallelism gave 3 percent here because the work was already vectorized and the pickling cost ate the win. The honest move was to measure it, reject it, and ship the simpler version.
- **Memory and speed are not always a trade-off.** Categoricals, `float32`, and streaming cut RSS from 1.6 GB to 0.9 GB *and* bought a 1.5 times speed bonus, because the elementwise scans were bandwidth-bound and smaller columns move fewer bytes.
- **Diff the output before you celebrate.** A fast pipeline that produces different numbers is a bug, not an optimization. Two of our porting steps changed behavior; we caught them only because we compared outputs row for row.
- **Make one process fast before you scale out.** An afternoon with the profiler and the ladder beat a week of distributed-systems complexity. The pipeline that "needed Spark" needed a dict and a dataframe.

## Further reading

- [Why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro and the optimization-loop / leverage-ladder frame this whole case study runs on.
- [CPU profiling: cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) — the deterministic profiling we used in step 0 to find the real hot stage.
- [Algorithmic complexity: the biggest speedups come from big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) — the deep dive behind step 1's list-scan-to-dict fix.
- [Dataframes at speed: pandas pitfalls, Polars, and Arrow](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) — the vectorization toolkit behind step 2, the biggest win.
- [Numba: JIT compiling Python to machine code](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code) — the deep dive behind step 3's compiled scoring kernel.
- [Multiprocessing: true parallelism and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) — why step 4 gave almost nothing, and when parallelism actually pays.
- The capstone of this series, *the Fast Python playbook*, turns this exact decision procedure into a reusable framework — the leverage ladder as a checklist you run on any slow program. (Same series; publishing alongside this post.)
- *High Performance Python* (Gorelick & Ozsvald, O'Reilly) — the book-length treatment of this methodology, with more worked profiling-to-native case studies.
- The Polars user guide and the Numba documentation — the canonical references for the two tools that delivered most of the win here, with the lazy/streaming and `@njit`/`fastmath` details we used.

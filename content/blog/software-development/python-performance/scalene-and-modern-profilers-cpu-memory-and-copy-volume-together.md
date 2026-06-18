---
title: "Scalene and Modern Profilers: CPU, Memory, and Copy Volume Together"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn to split a slow line into Python, native, and system time, catch a hidden type-promoting copy by its bandwidth, read pyinstrument and viztracer, and choose the right profiler for every question."
tags:
  [
    "python",
    "performance",
    "profiling",
    "scalene",
    "pyinstrument",
    "viztracer",
    "optimization",
    "memory",
    "observability",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-1.png"
---

A profile told me a line was slow. I believed it, rewrote the Python around it for two days, and got back a 4% speedup. The line was `df["price"] = df["price"].astype(np.float32) * weights`, and the profiler I was using — `cProfile` — had dutifully reported that this single statement burned 2.1 seconds of the job. What it could not tell me, because it does not measure it, was that **92% of those 2.1 seconds were spent inside NumPy's C code**, and a further chunk was spent silently copying an 800 MB array from `float64` to `float32` because the dtypes did not match. There was no Python to optimize. The win was hiding in a *copy*, and a copy is invisible to a timer that only counts which function the interpreter was in.

That is the gap this post closes. `cProfile` and `line_profiler` answer one question — *where is the time?* — and they answer it as a single undifferentiated number. But the lever you pull next depends entirely on questions they cannot answer: Is that time in *your* Python or in someone's *C*? If 90% is already in compiled code, rewriting Python is wasted effort, and the profiler must split it for you to even know that. How many megabytes per second is this line *copying*, with no arithmetic to show for it? Is the process slow because it is computing, or because it is waiting on a syscall, or because the event loop is blocked while fifty tasks sit idle? Modern profilers — `scalene`, `pyinstrument`, and `viztracer` — were built to answer exactly those questions, at low enough overhead that some of them run in production. This is the capstone of the measurement track in this series: the post that ties [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) together with the decision of *which tool to reach for*, and hands you off to the optimization tracks that follow.

![matrix of scalene columns Python time native time system time per-line memory and copy volume each mapped to the optimization lever it points to](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-1.png)

By the end you will be able to run `scalene script.py` and read its Python / native / system / memory / copy columns the way you read a stack trace; run `pyinstrument -r html` on a slow web request and read the call tree top to bottom; run `viztracer` and open the timeline to see a blocked event loop; and — the real skill — *choose the right profiler from the question you are asking* instead of always reaching for the one you know. Throughout we will stay on a named machine so the numbers are honest: **an 8-core x86-64 Linux box (the figures translate closely to an Apple M2), CPython 3.12, 16 GB RAM**, with all timings reported as wall-clock seconds or MB/s, medians of repeated runs.

## Why "where is the time" is the wrong question

Every profiler answers a question. The mistake that costs engineers days is assuming all profilers answer the *same* question. They do not. `cProfile` answers *which Python function was on the call stack and for how long*. That is a real and useful question — it is exactly what [CPU profiling with cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) is about — but it is a narrow one, and it is narrow in three specific ways that matter enormously for deciding what to do next.

First, `cProfile` attributes time to a *function*, not to a *layer*. When your hot function calls `numpy.dot`, the time spent inside `numpy.dot` shows up against `numpy.dot` — which is fine — but `cProfile` has no concept of "this 2 seconds was Python bytecode" versus "this 2 seconds was a C loop over a packed buffer." To `cProfile` it is all just *time in a frame*. Yet the distinction is the single most important fact for choosing a lever. If the time is in your Python, you can vectorize it, compile it with Numba, or rewrite the hot kernel in Cython — and you might get 10× to 100×. If the time is already in C, those levers do *nothing*, because there is no Python left to remove. A profiler that cannot tell Python time from native time cannot tell you whether your optimization plan is even possible.

Second, `cProfile` measures *time*, and only time. It will never tell you that a line allocated 800 MB, or that the process grew its resident set from 2 GB to 9 GB across one transformation, or that a line is copying memory at 3 GB/s with no useful computation attached. A program can be slow for reasons that have nothing to do with which function is on the stack — it can be slow because it is thrashing the allocator, blowing the cache, or moving bytes it never needed to move. We covered the allocation side of this in [memory profiling with tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks); here we add the dimension that even `memray` does not isolate: *copy volume*.

Third, `cProfile` is a *deterministic* profiler — it hooks every function call and return — which means it has overhead proportional to the number of calls. On call-heavy code it can slow your program 2× to 10×, which both distorts the very measurement you are taking and makes it unsafe to run against a live production process. The modern statistical profilers sample instead, paying near-zero overhead, which is what lets some of them attach to a running service.

There is a fourth narrowness that compounds the first three: `cProfile` aggregates *across the whole run*. It cannot tell you that the slowness is concentrated in one unlucky request out of a thousand, or that two stages which should overlap are running serially, because it collapses time into a per-function total with no notion of *when* anything happened. A program can have a perfectly healthy function-level profile and still be slow because of *scheduling* — a blocked event loop, a thread that never gets to run, a pipeline stage that waits on the previous one instead of overlapping. That is a *timeline* question, and an aggregate profiler is structurally incapable of answering it. You need a trace.

So the right opening question is not "where is the time?" It is "what do I need to *know* to decide what to do?" — and that question has at least six different answers, each pointing at a different tool: roughly where (function map), which line and which layer (per-line split), is it copying memory (copy volume), why is this request slow (call tree), what ran when (timeline), and is it stuck right now in production (live attach). Each answer is a different shape of question, and a different tool was built for each shape. Reaching for `cProfile` for all six is like using a single wrench on six different fasteners — it will fit one and strip the rest. Let me make the differences concrete before we touch any single tool.

#### Worked example: the same line, two profilers, opposite conclusions

Take the line from the intro. Here is the setup, on our 8-core Linux box, CPython 3.12.

```python
import numpy as np

n = 100_000_000
prices = np.random.rand(n).astype(np.float64)   # 800 MB float64
weights = np.random.rand(n).astype(np.float32)  # 400 MB float32

def weighted(prices, weights):
    return prices.astype(np.float32) * weights   # the suspicious line
```

`cProfile` reports the call to `weighted` at, say, 2.05 s cumulative, with the bulk attributed to `astype` and the multiply ufunc. A junior reading this sees "2 seconds in `weighted`, optimize `weighted`," and starts looking for a smarter Python expression. There is no smarter Python expression — the expression is already one NumPy statement.

`scalene` reports the *same* line and splits the 2.05 s: roughly **8% Python time** (the bytecode that set up the call and bound the names) and **92% native time** (the C ufunc and the `astype` copy). It *also* prints a copy-volume figure on that line — on the order of 3 GB/s — because `astype(np.float32)` on an 800 MB `float64` array materializes a *new* 400 MB buffer, and the multiply may materialize another. The verdict flips completely: do not touch the Python; the win is to **stop the copy** by storing `prices` as `float32` in the first place, so the dtypes already match and no promotion is needed.

```pycon
>>> prices32 = prices.astype(np.float32)        # do this ONCE, up front
>>> def weighted_fast(p32, weights):
...     return p32 * weights                    # same dtype: no per-call copy
...
```

The before→after on our box: the per-call cost drops from about 2.05 s to about 0.42 s — roughly a **4.9× speedup** — and it came entirely from deleting a copy that `cProfile` could not see and `line_profiler` could only report as undifferentiated "time." That is the whole argument for this post in one example.

![before and after comparison where cProfile flags a hot line as expensive while scalene shows ninety percent of it is native C time so the Python should be left alone](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-2.png)

## The science: why splitting Python time from native time changes the decision

Let me make the "92% in C, don't touch the Python" intuition rigorous, because it is the load-bearing idea and it deserves more than a hand-wave. The argument is just Amdahl's law applied to the *layer* boundary instead of to a code fraction.

Amdahl's law says that if a fraction $p$ of your runtime is sped up by a factor $s$, the overall speedup is

$$ S = \frac{1}{(1-p) + \frac{p}{s}} . $$

Now partition the runtime of a hot line into two layers: a Python fraction $p_{\text{py}}$ (the bytecode the interpreter executes — name binding, attribute lookup, the call machinery) and a native fraction $p_{\text{c}} = 1 - p_{\text{py}}$ (time spent inside compiled C, where the GIL is typically released and the work is a tight loop over a packed buffer). The optimization levers that "rewrite the Python" — vectorizing a loop, replacing a comprehension, hoisting a lookup — can only attack $p_{\text{py}}$. They cannot reduce $p_{\text{c}}$ at all, because there is no Python in the C loop to remove. Even if you made the Python *infinitely* fast ($s \to \infty$ on the Python fraction), the best speedup you can get is

$$ S_{\max} = \frac{1}{1 - p_{\text{py}}} = \frac{1}{p_{\text{c}}} . $$

Plug in the worked example: $p_{\text{c}} = 0.92$, so $S_{\max} = 1/0.92 \approx 1.087$. The hard ceiling on any Python-side optimization of that line is **8.7%** — and that is the *theoretical* best, assuming you make the Python free, which you never can. Two days of Python rewriting to chase an 8.7% ceiling is exactly the trap I fell into in the intro. The only way to know that ceiling *exists* is to measure $p_{\text{py}}$ versus $p_{\text{c}}$, and the only common profiler that measures it is `scalene`.

The converse matters just as much. When `scalene` shows a line is, say, **85% Python time**, that is a flashing green light: the time is in the interpreter, so vectorization or compilation can attack 85% of it, and your realistic ceiling is $1/0.15 \approx 6.7\times$ before you even consider the speedup factor itself. The Python-versus-native split is not trivia. It is the number that tells you whether the next track in this series — vectorize, compile, parallelize — can help you at all, and by how much. A timer that collapses the two layers into one number throws away the single most decision-relevant fact you have.

There is a third layer worth naming: **system time**. When `scalene` attributes time to "system," it means the process was inside a syscall or blocked — reading a file, waiting on a socket, sleeping, waiting on a lock. System time does not respond to *any* compute lever; you cannot vectorize a `read()`. It responds to *overlap*: doing other work while you wait, via threads (which release the GIL during I/O) or `asyncio`. So the three-way split — Python, native, system — maps cleanly onto the three big lever families: Python time → vectorize/compile; native time → already fast, leave it; system time → overlap the wait. One profiler column per lever family. That is the design insight behind `scalene`, and it is why it belongs at the center of a measurement-track capstone.

## Scalene from zero: installing, running, reading the columns

`scalene` is a `pip install scalene` away and runs as a drop-in replacement for `python`:

```bash
pip install scalene
scalene your_script.py                 # CLI report in the terminal
scalene --html --outfile prof.html your_script.py   # rich HTML report
scalene --cpu --memory your_script.py  # explicitly enable both (default is both)
scalene --reduced-profile your_script.py  # only show lines above a threshold
```

For a snippet rather than a whole script, you can profile a region programmatically:

```python
from scalene import scalene_profiler

scalene_profiler.start()
result = heavy_pipeline(rows)     # only this region is profiled
scalene_profiler.stop()
```

When it finishes, `scalene` prints a per-line table. The columns are the whole point, so let me describe each one and exactly what decision it drives. A schematic of a `scalene` line table — the kind it prints to your terminal — looks like this (annotated):

```bash
# columns:  Line | Time Python | Time native | Time system | Mem MB | Copy MB/s | code
   12  |   45% |    3% |   0% |   12 MB |        |  for row in rows:
   13  |   38% |    2% |   0% |   30 MB |        |      clean.append(parse(row))
   41  |    8% |   60% |   0% |  410 MB |  3120  |  out = prices.astype("float32") * w
   58  |    1% |    2% |  29% |    0 MB |        |  data = sock.recv(65536)
```

Read it column by column. **Time Python** is the percentage of total runtime this line spent executing *Python bytecode in the interpreter*. **Time native** is the percentage spent inside *C code* called from this line (NumPy, a C extension, a compiled library). **Time system** is the percentage spent in syscalls or blocked. **Mem MB** is the net memory growth attributed to this line (how much resident memory this line is responsible for at peak). **Copy MB/s** is the rate at which this line is *copying* bytes — and a number appearing in that column at all is a signal worth investigating.

Now read the *decisions* off the schematic. Lines 12–13 are dominated by **Python time** (45% and 38%) — that is a classic interpreter-bound loop, and it is the textbook candidate to vectorize or push into NumPy/Polars (and exactly the kind of thing the [vectorization track](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) attacks). Line 41 is **60% native** with a copy rate of 3120 MB/s — leave the Python alone (the Amdahl ceiling is tiny), kill the copy. Line 58 is **29% system** — that is I/O wait, and no compute lever touches it; the fix is to overlap it with threads or async. Four lines, four *different* verdicts, all read from columns a time-only profiler does not have.

![stack of profiler resolution layers from wall-clock down through CPU per function per-line cost per-line memory copy volume and concurrency timeline](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-4.png)

One operational note that saves confusion: `scalene` is a *sampling* profiler for time (it periodically interrupts the program and records what it was doing) and uses an interposed allocator plus signal-based sampling for memory and copies. The upshot is **low overhead** — typically well under a 2× slowdown, often much less, versus the 2–10× of `cProfile` — which is what makes the per-line memory and copy tracking practical to leave on. It samples, so very short programs give noisy lines; run something that executes for at least a couple of seconds for stable percentages.

### Why the system-time column deserves its own attention

The **Time system** column is the one engineers skip over, and skipping it is a mistake, because it is the column that tells you a problem is *not a compute problem at all*. When a line shows high system time, the process was inside the kernel — blocked on a `read()`, a `recv()`, a `write()`, a lock acquisition, a `sleep`, or any syscall that suspends the thread. No amount of vectorizing, compiling, or smarter Python touches that time, because the CPU was not executing your code; it was *waiting*. The lever for system time is *overlap*: keep the CPU busy with other work while one task waits. For I/O-bound waits that means threads (the GIL is released during a blocking I/O syscall, so other threads run) or `asyncio` (await the I/O and let the event loop drive other coroutines). For lock waits it means reducing contention or shortening critical sections.

The diagnostic power is in the *contrast* across columns. A line that is 60% native is doing useful C work — fine. A line that is 60% system is doing *nothing but waiting* — and that is a completely different verdict, even though a time profiler reports both simply as "time spent here." I have watched a team spend a sprint trying to "optimize" a function that `scalene` plainly showed was 70% system time: it was blocked reading from a slow upstream service the entire time, and the only real fix was to fan the reads out concurrently so the waits overlapped. The Python in that function was already as fast as it would ever need to be. Reading the system column would have saved the sprint.

`scalene` also tracks **GPU utilization** when a GPU is present and the relevant libraries are installed — it will show GPU time per line for code that offloads to CUDA — which extends the same layer-splitting philosophy to accelerator code. That said, for serious GPU work the dedicated accelerator profilers go far deeper; when the bottleneck has genuinely moved onto the device, cross over to [profiling GPU workloads and finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck), which owns the roofline, occupancy, and kernel-level view. `scalene`'s GPU column is the right tool to *tell you the work moved to the GPU*; the device profilers are the right tools to optimize it once it has.

## Copy volume: the dimension nothing else measures

This is the column that makes `scalene` genuinely unique, so it earns its own section and its own science. "Copy volume" is the number of bytes per second that a line of code spends *copying memory* — moving bytes from one buffer to another with no arithmetic that justifies the move. Why does this matter, and why is it invisible to a time profiler?

Consider what a hidden copy *is*. NumPy arrays have a `dtype` and a contiguous, typed buffer. Many operations that look free are not: slicing with a boolean mask, calling `astype`, mixing `float64` with `float32`, calling a function that expects C-contiguous memory on a transposed (and therefore Fortran-ordered) view, or passing a NumPy array to a library that quietly demands its own layout. Each of these can materialize a *new* buffer — a full copy of the data — and the copy is pure memory traffic. It does compute nothing. It just moves bytes.

Here is the science of *why it is invisible to a timer*. A time profiler tells you *which function* the program was in. A copy happens inside `memcpy` or inside a NumPy cast routine; a time profiler will faithfully attribute a sliver of time to that routine, but it cannot tell you *that this was a copy* versus a useful transform, and it certainly cannot tell you the *volume*. Yet the volume is the signal. Memory bandwidth on our box is finite — call it on the order of 20–40 GB/s of sustained streaming bandwidth for a single core's view of main memory. If a line copies 800 MB, that copy *alone* costs roughly $800\,\text{MB} / 25{,}000\,\text{MB/s} \approx 32\,\text{ms}$ of pure bandwidth, and it does so every time the line runs. Do it in a loop a hundred times and you have spent 3.2 seconds moving bytes for no reason. The time shows up — but as "time in `astype`," which looks like honest work. The *copy-volume* column is what reframes it as "you are moving 3 GB/s of data you did not need to move," which immediately suggests the fix.

The most insidious case is the **type-promoting copy**. When a `float64` array meets an operation or peer array of `float32`, NumPy must promote to a common type to do the arithmetic, and promotion means allocating a new buffer and copying every element into it with a conversion. The expression looks innocent — `a * b` — and a time profiler shows time in the multiply ufunc, which is *expected*. What is not expected, and what only copy volume reveals, is that *most of that time is the promotion copy, not the multiply*. The figure below traces exactly this: a `float64` array on one side, a `float32` operation on the other, the silent promotion forcing a new buffer, and `scalene`'s copy meter lighting up at gigabytes per second.

![graph showing a float64 array meeting a float32 operation forcing a silent promoting copy into a new buffer that scalene flags at gigabytes per second then fixed by matching the dtype](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-3.png)

#### Worked example: catching a float64 to float32 promotion by its bandwidth

Concretely, on our 8-core Linux box, CPython 3.12, NumPy 1.26. Start with the mismatched-dtype version and profile it with `scalene`.

```python
import numpy as np

N = 50_000_000
a = np.random.rand(N)                  # float64, 400 MB
b = np.random.rand(N).astype(np.float32)  # float32, 200 MB

def blend(a, b):
    return a * b                        # promotes b to float64 every call: hidden copy
```

`scalene` on a loop of `blend` shows the line at, say, 0.31 s per call, with a **Time native** of around 95% (it is all in C) and a **Copy MB/s** figure that screams — on the order of 2–3 GB/s — because every call promotes `b` from `float32` to `float64`, materializing a fresh 400 MB buffer before the multiply even starts. A timer would just say "0.31 s in `blend`, mostly native, looks optimal." The copy column says otherwise.

The fix is to make the dtypes agree so no promotion is needed. Decide which precision you actually want; here `float32` halves the memory traffic and is plenty for the use case:

```python
a32 = a.astype(np.float32)   # convert ONCE, up front

def blend_fast(a32, b):
    return a32 * b           # both float32: no per-call promotion, no copy
```

| Version | Time/call | Native % | Copy MB/s | Peak RSS | Verdict |
| --- | --- | --- | --- | --- | --- |
| `blend` (float64 × float32) | 0.31 s | ~95% | ~2,600 | 1.0 GB | hidden promotion copy every call |
| `blend_fast` (float32 × float32) | 0.13 s | ~94% | ~0 (flagged gone) | 0.6 GB | no promotion, half the bandwidth |

The wall-clock drops from 0.31 s to 0.13 s — about **2.4×** — and the copy-volume line goes quiet. Two things to notice. First, **Native %** barely moved (95% → 94%) because both versions are dominated by C; a Python-versus-native split alone would *not* have told you what to do here — you needed the *copy* column. Second, the peak RSS fell from ~1.0 GB to ~0.6 GB, because you are no longer materializing a transient `float64` promotion buffer. Copy volume catches a class of bug that is simultaneously a *time* bug and a *memory* bug, and it is the only column that names it directly.

## Reading scalene's hints: where Python to native would help

`scalene` does one more thing the others do not: it offers *suggestions*. In its HTML report and (with the right flags) its terminal output, it flags lines where moving from Python to native code would likely pay off — heavily Python-time loops over numeric data — and it can even ask an LLM (with `--gpu` and AI flags configured) to propose an optimization. You do not have to use the AI integration; the heuristic flag alone is useful. A line that is 90% Python time, runs millions of iterations, and touches numeric data is exactly the line that vectorization or a `@njit` Numba kernel will transform, and `scalene` will mark it. Treat these as *hypotheses to test*, not gospel — the discipline of this series is always *measure the before, apply the lever, measure the after* — but they are well-aimed hypotheses, because they are generated from the Python/native split that nothing else exposes.

There is a memory analogue. `scalene` distinguishes memory *growth* attributed to each line and tracks the high-water mark, and it can flag lines that allocate a lot relative to how often they run. Combined with the [memray](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) allocation flame graph — which is better for tracing *where* an allocation chain originated — you get a complete picture: `memray` for the call-stack provenance of allocations, `scalene` for the per-line time/memory/copy attribution side by side. They compose; you do not have to choose one forever.

What makes the side-by-side view powerful is that *time and memory often disagree about which line is the problem*, and the disagreement is itself the diagnosis. A line can be cheap in time but ruinous in memory — say, a comprehension that builds a 6 GB intermediate list in a fraction of a second — and a time profiler will wave it through while the process quietly marches toward an OOM kill. `scalene` puts the **Time** columns and the **Mem MB** column on the *same row*, so a line that is 2% of the time but 70% of the peak memory jumps out immediately. That is a pattern you cannot see when your time profiler and your memory profiler are separate tools reporting separate rankings; you have to mentally join two tables, and the join is exactly where the bug hides.

#### Worked example: the line that's cheap in time but ruinous in memory

A nightly aggregation on our 8-core Linux box, CPython 3.12, was being OOM-killed intermittently at 16 GB even though the inputs were only a few GB. `cProfile` showed the runtime spread sanely across parsing and grouping — nothing alarming, no single dominant function. The job was not *slow*; it was *fat*, and a time profiler is structurally blind to fat. Running `scalene` instead, the per-line table told the story on one row:

```bash
   # Line | Time Python | Time native | Mem MB | code
   88  |   1% |   1% |   9200 MB |  records = [transform(r) for r in stream]
   91  |   3% |  40% |    120 MB |  df = pd.DataFrame(records)
```

Line 88 is **2% of the time** and **9.2 GB of peak memory** — it materializes the *entire* stream into a Python list of objects before line 91 even runs, and the list of boxed objects is several times larger than the eventual DataFrame. A time profiler ranked line 88 near the bottom; `scalene`'s memory column ranked it first. The fix is to stop materializing: feed a *generator* into the DataFrame constructor (or, better, process in chunks) so the records are consumed as they are produced rather than all held at once.

```python
# before: builds the whole 9.2 GB list, then the DataFrame
records = [transform(r) for r in stream]
df = pd.DataFrame(records)

# after: stream the records; peak memory drops to the chunk size
def chunks(it, size=100_000):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield pd.DataFrame(buf); buf.clear()
    if buf:
        yield pd.DataFrame(buf)

frames = [c for c in chunks(transform(r) for r in stream)]
df = pd.concat(frames, ignore_index=True)
```

Peak RSS fell from ~9.2 GB on that line to roughly the chunk's footprint (a few hundred MB at a time), and the OOM kills stopped. Wall-clock barely changed — this was never a *time* problem — which is exactly why a time profiler never flagged it. The lesson is structural: **when a process dies on memory, not time, only a tool that puts memory on the same per-line table will point at the culprit.**

## pyinstrument: the call-stack profiler for "why is this request slow"

`scalene` answers *which line, and in which layer*. There is a different question that comes up constantly in web and service code: *why is this particular request slow, and what is the call path that makes it slow?* For that, the flat, per-function view of `cProfile` is actively unhelpful — it sorts functions by cost but throws away who-called-whom, so a slow request becomes a scavenger hunt of "okay, `serialize` is expensive, but which of the four call sites triggered it?" `pyinstrument` exists for exactly this. It is a **statistical call-stack profiler**: it samples the call stack periodically and aggregates the samples into a *tree* that reads top to bottom, from the entry point down to the leaf where the time actually went.

```bash
pip install pyinstrument
python -m pyinstrument your_script.py        # text tree to stdout
pyinstrument your_script.py                  # same, via the console script
pyinstrument -r html -o prof.html your_script.py   # rich interactive HTML tree
pyinstrument -r speedscope -o prof.json your_script.py  # speedscope flamegraph format
```

For a web app, the killer feature is the middleware / context-manager mode that profiles a *single request* and gives you the tree for just that handler:

```python
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()
handle_request(request)        # the slow endpoint
profiler.stop()
print(profiler.output_text(unicode=True, color=True))
```

Or, in async code, `pyinstrument` understands `await` boundaries and will show you the time across awaits — which a naive stack profiler would mangle. There is first-class integration for the popular frameworks (a Django/Flask/FastAPI middleware that profiles requests when a query parameter is set), which is how teams use it: hit the slow endpoint with `?profile=1` and read the tree.

What the tree gives you that the flat table cannot is **context for free**. A `pyinstrument` tree for a 310 ms request reads like this (abbreviated):

```bash
310ms  handle_request                         views.py:42
├─ 240ms  serialize_response                  serializers.py:88
│  └─ 235ms  to_representation                 serializers.py:120
│     └─ 230ms  <listcomp>                     serializers.py:121
└─  55ms  query_database                       db.py:14
```

You read it top to bottom and the answer is *right there*: 240 of 310 ms is serialization, almost all of it inside one list comprehension in `to_representation`. You did not have to reconstruct the call path; the tree *is* the call path. `cProfile` would have told you `to_representation` is expensive and left you to figure out that it is reached through `serialize_response` from `handle_request`. For "why is my web request slow," the tree is the right shape and the flat table is the wrong shape, even though both are measuring CPU time.

![before and after comparison contrasting cProfile flat sortable table against the pyinstrument call-stack tree that reads top to bottom for a slow web request](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-6.png)

A subtlety worth internalizing: `pyinstrument` by default reports **wall-clock** time, not CPU time, and that is usually what you want for a request. If your handler spends 200 ms waiting on a database round trip, you *want* that 200 ms to show up — it is why the request is slow — even though the CPU was idle the whole time. `cProfile` would also count that wall time against the call, but it would not show you the path. `pyinstrument` also intelligently hides library internals by default (it folds away the framework's own machinery so you see *your* code), which is the difference between a 30-line readable tree and a 3,000-line dump. You can turn that off with `--show-all` when you genuinely need to see into a dependency.

A word on how `pyinstrument` relates to a *flame graph*, since they are often confused. A flame graph (the kind `py-spy record` or `pyinstrument -r speedscope` produces) is the same call-stack data laid out *horizontally*, with width proportional to time and stacking showing call depth; it is excellent for spotting a wide bar (a function that dominates) at a glance across a large program. The `pyinstrument` *text tree* is the same information in a vertical, collapsible, percentage-annotated form that is easier to read in a terminal or a code review. Use the flame graph when you want the bird's-eye view of where width concentrates; use the tree when you want to *read the path* line by line and quote it in a ticket. They are two renderings of the same sampled-stack data, and `pyinstrument` will emit either. The decision between them is ergonomic, not analytical — both answer "which call path is slow."

One honest limitation to keep in mind: because `pyinstrument` samples *stacks*, it shares the sampler's blind spot for very short-lived calls, and — like every stack profiler — it attributes time to the *Python* frames it can see, not to the layer beneath them. If your slow request is slow because of a native extension burning CPU, `pyinstrument` will show you the Python frame that *called* the extension, but it will not split that time into Python versus native the way `scalene` does. So the division of labor is clean: `pyinstrument` for "which call path, in a request," `scalene` for "which line, in which layer." When a `pyinstrument` tree bottoms out at a call into NumPy or a C extension and you need to know whether the cost is the Python around it or the C inside it, that is your cue to switch tools.

#### Worked example: a p99 of 310 ms traced to one comprehension

A real shape this takes: a `/api/orders` endpoint at p99 = 310 ms on our box. The team's first instinct was "the database is slow." `cProfile` on a captured request showed a flat table topped by `to_representation` and `isoformat` and `Decimal.__str__` — true, but not *actionable*, because none of those names told the on-call engineer what to change.

Running `pyinstrument` on the same request produced the tree above: 240 ms in `serialize_response`, narrowing to a list comprehension that called `to_representation` *per row* across 4,000 rows, and inside it a per-row `datetime.isoformat()` and a per-row `Decimal` to `str` conversion. The database query was 55 ms — not the problem. The fix was to serialize in bulk (vectorize the datetime formatting, batch the decimal handling) rather than per row, which cut serialization from 240 ms to about 40 ms and dropped p99 to roughly **110 ms**, a **2.8×** improvement on the endpoint. The point is not the specific fix; it is that the *tree* pointed straight at the per-row pattern, while the flat table buried it under three innocent-looking function names. Same measurement, better shape for the question.

## viztracer: the timeline for concurrent and async code

There is a third kind of question that neither a line profiler nor a call-stack profiler answers well: *what ran when?* For concurrent code — threads, `asyncio`, multiprocessing — the bug is often not "this function is slow" but "these things did not overlap the way I thought," or "the event loop was blocked for 50 ms while fifty awaited tasks sat idle," or "two stages that should pipeline are actually running serially." Aggregate profilers *average away* exactly this information: a 50 ms gap where nothing useful happened does not show up as a hot function, because no function was hot — the loop was *stalled*. You need a *trace*: a timeline of every event, left to right, on every thread and task, so you can *see* the gaps.

`viztracer` produces exactly that. It records a timeline of function entries and exits across threads and async tasks and renders it in the Perfetto UI (the same trace viewer Chrome and Android use), where you can zoom, pan, and read concurrency the way you read a Gantt chart.

```bash
pip install viztracer
viztracer your_script.py                 # writes result.json
viztracer -o trace.json your_script.py   # name the output
vizviewer result.json                    # open the Perfetto timeline in a browser
viztracer --max_stack_depth 10 your_script.py   # cap depth to keep the trace small
```

Programmatically, to trace just a region:

```python
from viztracer import VizTracer

with VizTracer(output_file="trace.json", max_stack_depth=12):
    asyncio.run(fetch_all(urls))     # only this region is traced
```

Open the result in `vizviewer` and you see one row per thread and one row per async task, with colored bars for each function call laid out on a time axis. Concurrency becomes visual: bars that overlap *did* run concurrently; bars separated by a blank gap mean *nothing ran there*, which for an event loop means it was blocked or starved. The classic find is a synchronous, CPU-heavy call that someone slipped into an `async def` — it does not `await`, so it monopolizes the single event-loop thread, and every other task's bar stops dead until it returns. On the timeline that shows up as one long bar with a desert of empty space beside it. No aggregate profiler will ever show you that desert; the trace shows it at a glance.

![timeline trace of concurrent async tasks where a synchronous parse blocks the event loop leaving a visible dead gap while awaited fetches sit idle](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-7.png)

The cost of all this resolution is **overhead and trace size**. `viztracer` records *every* call by default, which both slows the program (more than a sampler, less than full `cProfile` in many cases, but real) and produces large trace files for long runs. That is the deliberate trade-off: it is a *tracer*, not a sampler, so it is the right tool for *short, targeted* runs where you want every event — a single slow request, one batch of async fetches, a few seconds of a pipeline — and the wrong tool for "profile this 9-hour job in production." Cap `--max_stack_depth`, trace a bounded region with the context manager, and you keep the traces manageable. When the question is genuinely "what ran when," nothing else gives you the answer as directly.

#### Worked example: an async scraper that wasn't concurrent

A scraper that fetched 200 URLs with `asyncio` and `httpx` was clocking 41 seconds — suspiciously close to *serial* time, as if the concurrency were doing nothing. An aggregate profiler showed time spread across `httpx` internals and a `parse_html` function; nothing obviously wrong. `viztracer` made it obvious in one screen: the fetches *did* fire concurrently (their bars overlapped early), but each response was handed to a synchronous `parse_html` that ran BeautifulSoup *on the event loop thread*, taking ~180 ms each. While `parse_html` ran, the loop could not advance any other task — so the 200 parses serialized into one long single-file march, each blocking the next fetch from being processed. The timeline showed fetch bars bunched at the start, then a long stripe of back-to-back `parse_html` bars with no fetch activity beside them.

The fix was textbook once the trace named it: move the CPU-bound parse off the loop with `await asyncio.to_thread(parse_html, html)` (or a `ProcessPoolExecutor` for heavier parsing), so the loop stays free to drive fetches while parses run on worker threads. Wall-clock dropped from 41 s to about 9 s — roughly **4.5×** — and the timeline showed fetches and parses overlapping the way they should. The number alone ("41 s") never told us *why*; the *shape* of the trace did.

## Choosing the profiler from the question

Here is the skill that ties the whole measurement track together. You do not pick a profiler because it is the one you remember. You pick it from the *question you are asking*, and the question routes you to the tool. The decision tree below is the one I actually run in my head; commit it to memory and you will stop wasting hours pointing the wrong instrument at the problem.

![decision tree routing from the question being asked down to the single profiler that answers it cheaply for slowness memory and runtime shape](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-5.png)

Walk the branches:

- **"It's slow overall — where is the time, roughly?"** Start with `cProfile` (deterministic, gives you a function-level map) or, if you cannot stop the process, `py-spy top`. This is the [cProfile post's](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) territory. It tells you which function, which is enough to know *where* to zoom in.
- **"Which line, and is it my Python or someone's C?"** `scalene`. This is the question only it answers — the Python/native/system split per line. If the answer is "native," stop optimizing the Python (Amdahl ceiling). If "Python," you have a vectorize/compile candidate. If "system," you have an I/O-overlap candidate.
- **"Is this line copying memory it shouldn't?"** `scalene`'s copy-volume column. Nothing else measures it. A hidden dtype promotion or an accidental array copy shows up here and nowhere else.
- **"Why is this *request* slow — what's the call path?"** `pyinstrument`. The tree reads top to bottom and keeps the context the flat table loses. Ideal for web handlers; reports wall-clock so I/O waits are visible.
- **"Where is the memory going / is there a leak?"** `memray` for allocation provenance (the flame graph of *who allocated*), `tracemalloc` for cheap built-in snapshots, `scalene` for per-line memory alongside time. Covered in the [memory post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks).
- **"What ran *when* — is my concurrency actually concurrent?"** `viztracer` timeline. The only tool that shows you the gaps and the overlaps.
- **"It's hung / deadlocked / I can't reproduce locally — it's *in production right now*."** `py-spy dump` (a one-shot stack dump of every thread of a running process, zero instrumentation) or `py-spy top` (a live `top`-like view). Attach by PID, read the stacks, find the deadlock. No code change, no restart.

The same logic, laid out as a comparison matrix — which is the most useful single artifact to keep at your desk:

| Question you're asking | Reach for | Why this one |
| --- | --- | --- |
| Roughly where is the time? | `cProfile` / `pstats` | Function-level map, built in |
| Which line — Python or C? | `scalene` | Only tool that splits Python/native/system per line |
| Is a line copying memory? | `scalene` (copy volume) | Only tool that measures MB/s copied |
| Why is this request slow? | `pyinstrument` | Call-stack tree, keeps the path, wall-clock |
| Where do allocations come from? | `memray` | Allocation flame graph with provenance |
| Per-line memory + time together | `scalene` | Memory and time side by side per line |
| What ran when (async/threads)? | `viztracer` | Timeline trace in Perfetto |
| It's hung in production | `py-spy dump` | Attach by PID, no restart, zero overhead |
| Live CPU on a running process | `py-spy top` | Sampling, near-zero overhead, attachable |

There is no profiler that wins on every axis, and pretending one does is how you end up with the wrong tool. The final figure makes the trade-offs explicit across the five profilers we have discussed.

![matrix grid comparing scalene pyinstrument viztracer py-spy and cProfile across the dimensions they see their overhead and whether they are production safe](/imgs/blogs/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together-8.png)

## The science of overhead: deterministic versus statistical profiling

Why can `py-spy` attach to a live production service while `cProfile` cannot? The answer is in *how* each profiler collects data, and it is worth understanding because it determines safety, accuracy, and where each tool belongs.

A **deterministic** profiler like `cProfile` registers a callback that fires on *every* function call and return (via `sys.setprofile`). Its overhead is therefore proportional to the *number of calls*, not the runtime. If your program makes $C$ calls and each profiler callback costs roughly $\delta$ (a few hundred nanoseconds — bookkeeping, a dictionary update, a timer read), the added time is

$$ T_{\text{overhead}} \approx C \cdot \delta . $$

For call-heavy code — deep recursion, millions of tiny function calls — $C$ is enormous and the overhead can be 2× to 10× the base runtime. Two consequences follow. First, the measurement is *distorted*: functions that are called a million times look disproportionately expensive because each call pays the profiler tax, so a function that is genuinely cheap-per-call but called constantly gets inflated. Second, you obviously cannot run a 5× slowdown on a live service handling real traffic.

A **statistical** (sampling) profiler like `py-spy`, `scalene`, or `pyinstrument` does the opposite. It does *nothing* during normal execution; periodically — say every 1 ms — it interrupts and records the current call stack (or, for `py-spy`, reads the target process's memory from *outside*, without even pausing it meaningfully). Its overhead is proportional to the *sampling rate*, not the call count:

$$ T_{\text{overhead}} \approx \frac{1}{\Delta t_{\text{sample}}} \cdot \delta_{\text{sample}} \cdot T_{\text{run}} , $$

which for a 1 ms sample interval and a cheap sample is a small *fixed fraction* of runtime — often 1–5%, sometimes less. The cost is *resolution*: you get a statistical estimate, so a function that runs for less time than a sample interval may be under-counted or missed, and short programs are noisy. But for any program that runs more than a couple of seconds, sampling converges to an accurate result while paying a fraction of the overhead. And because `py-spy` reads the target from a separate process, it requires *no code change and no restart* — you point it at a PID. That is precisely why sampling profilers are the production-safe family and deterministic profilers are the development-only family.

It is worth making the "noisy on short runs" claim quantitative, because it tells you *how long* to run before you trust a sampler. If a function truly occupies a fraction $p$ of runtime, and you collect $N$ independent samples, the number that land in that function is a binomial random variable with mean $Np$ and standard deviation $\sqrt{Np(1-p)}$. The *relative* error on your estimate of $p$ therefore scales as

$$ \frac{\sqrt{Np(1-p)}}{Np} = \sqrt{\frac{1-p}{Np}} \approx \frac{1}{\sqrt{Np}} \quad \text{for small } p . $$

The practical reading: to estimate a function that is 10% of runtime to within roughly 10% relative error you need on the order of $N \approx 1/(0.1 \cdot 0.1^2) = 1{,}000$ samples, which at a 1 ms interval is one second of runtime. To resolve a function that is only 1% of runtime to the same precision you need ten times as many samples — ten seconds. This is *why* samplers are noisy on sub-second programs and trustworthy on multi-second ones, and it is why you should let a sampled profile run long enough to accumulate samples in the regions you care about. A deterministic profiler has no such convergence requirement — it counts every call exactly — but it pays for that exactness with the per-call tax. The trade is precision-per-second of overhead versus exactness-at-high-overhead, and which you want depends entirely on whether you can afford to slow the program down.

| Profiler | Method | Typical overhead | Distorts call-heavy code? | Attach to live process? |
| --- | --- | --- | --- | --- |
| `cProfile` | Deterministic (every call) | 2×–10× on call-heavy | Yes (per-call tax) | No |
| `line_profiler` | Deterministic (per line, opted-in) | High on traced funcs | On traced lines | No |
| `scalene` | Sampling + alloc interposition | Usually under 2× | Minimal | Run-with, not attach |
| `pyinstrument` | Sampling call stacks | ~1–5% | Minimal | Run-with / middleware |
| `py-spy` | External sampling (reads memory) | Near zero | Minimal | **Yes, by PID** |
| `viztracer` | Tracing (every event) | Moderate–high | Records everything | Run-with, short runs |

The honest reading of this table: use deterministic tools (`cProfile`, `line_profiler`) on a *representative input in development* where the distortion is acceptable and you want exhaustive per-call detail. Use sampling tools (`scalene`, `pyinstrument`, `py-spy`) when overhead matters or the process is *live*. Use the tracer (`viztracer`) for short, targeted concurrency investigations. The method *is* the trade-off.

## Profiling in production safely

The reason any of this matters for real systems is that the bug you most need to profile is often the one that *only* reproduces in production — under real traffic, real data shapes, real concurrency. Local profiling lies about these constantly. So you need a way to profile production *without* taking it down, and the sampling family gives it to you. A few rules I follow, hard-won:

**Use `py-spy` for the truly live cases.** When a service is hung, deadlocked, or pinned at 100% CPU on one core and you cannot reproduce it locally, `py-spy dump --pid <PID>` gives you a one-shot stack trace of every thread *without restarting or instrumenting anything*. It reads the process's memory from outside, so the target keeps running. For a deadlock, the dump shows two threads each blocked acquiring a lock the other holds — the diagnosis is right there in the stacks. For a CPU spin, `py-spy top --pid <PID>` gives you a live `top`-like view of which Python functions are eating the core. This is the single most valuable production-debugging tool in the Python world, and it needs no cooperation from the running program.

```bash
py-spy dump --pid 12345            # one-shot stacks of every thread (deadlock/hang)
py-spy top  --pid 12345            # live top-like view of hot functions
py-spy record -o flame.svg --pid 12345 --duration 30   # 30s flame graph, live
```

**Keep overhead bounded and sampled.** `pyinstrument` middleware that profiles only requests carrying `?profile=1` (and only for an internal IP) lets you profile a *specific* slow request in production at sampling overhead, affecting nothing else. `scalene` you would typically run against a *replica* or a captured workload rather than the live fleet, because it runs *with* the process rather than attaching — but its overhead is low enough that a canary instance is fine.

**Never run `cProfile` or `viztracer` against the live fleet.** Their overhead is too high and, in `viztracer`'s case, the trace files balloon. Reserve them for a canary, a replica, or a captured workload replayed offline.

**Account for the observer effect.** Even a sampler perturbs timing slightly, and a deterministic profiler perturbs it grossly. When you report a number, report the *un-profiled* wall-clock too — profile to *find* the bottleneck, then measure the *fix* with the profiler off. The profiler is a flashlight, not a stopwatch; do your final timing with `time.perf_counter` (the discipline from [benchmarking Python correctly](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means)) once the profiler has pointed you at the right line.

**Profile the workload that actually reproduces the bug.** This is the rule most often violated. A profile of a toy input on a developer laptop tells you about the toy input on the laptop — not about production, where the data is larger, the cache pressure is real, the concurrency is genuine, and the dtype that triggers a promotion copy actually appears. The hidden `float64`-to-`float32` copy from earlier never shows up on a 10-row test fixture; it only bites at 50 million rows. So when you can, profile against a *captured* production workload — a replayed request log, a representative batch, a sampled slice of real data — even if you run it on a replica rather than the live fleet. The closer the input shape is to production, the more the profile's verdict transfers. A beautiful profile of the wrong workload is worse than no profile, because it gives you false confidence in the wrong line.

**Keep a runbook of which command answers which production question.** When a service is misbehaving at 3 a.m., you do not want to be reasoning about sampling theory; you want a card that says: *hung or deadlocked* → `py-spy dump --pid X`; *pinned CPU* → `py-spy top --pid X` then `py-spy record` for a flame graph; *slow specific endpoint* → hit it with the `pyinstrument` profiling middleware enabled; *memory climbing* → `memray` against a replica or `scalene` per-line on a canary. The decision tree in the figure above is that runbook in visual form. The whole value of having learned these tools is that, under pressure, you reach for the right one in seconds instead of flailing with the one you happen to remember.

#### Worked example: a deadlock found in 30 seconds with py-spy

A background worker had stopped processing — queue depth climbing, no errors, no crash, just *stuck*. No way to reproduce locally; it had run fine for weeks. SSH to the box, find the PID, and:

```bash
py-spy dump --pid 9981
```

The dump showed two threads. Thread A was blocked inside `acquire` on a `threading.Lock` named `cache_lock`, called from `refresh_cache`. Thread B was blocked inside `acquire` on a *different* lock, `db_lock`, called from `write_results` — and the stack showed `write_results` already held `cache_lock`, while `refresh_cache` already held `db_lock`. Classic lock-ordering deadlock: A holds `cache_lock` and wants `db_lock`; B holds `db_lock` and wants `cache_lock`; neither will ever proceed. Diagnosis in **30 seconds, zero restarts, zero instrumentation** — the running process simply told us its stacks. The fix (impose a global lock-acquisition order so both code paths take `cache_lock` before `db_lock`) is trivial once you can *see* the cycle. Without `py-spy`, this is a multi-hour log-archaeology session that may never conclude. The lesson: for "it's stuck in production," reach for the external sampler first, not your logs.

## Putting the profilers together: one investigation, several tools

In practice you rarely solve a real performance problem with a single profiler. You *escalate* through them, each tool answering a narrower question than the last, until you have isolated the one line and the one lever. Let me walk the escalation end to end on the series' running example — a data-processing pipeline that loads a few million rows, cleans them, transforms, and aggregates — because the *sequence* is the actual skill, and it composes everything above.

**Step one — get a map.** The pipeline takes 95 seconds and you have no idea why. Do *not* start with the heavy tools; get a cheap function-level map first. `python -m cProfile -o run.prof pipeline.py` then `pstats` sorted by cumulative time, or `py-spy top` if you would rather not pay the deterministic tax. The map says: 60% of the time is in `clean_rows`, 25% in `aggregate`, the rest scattered. Now you know *where* to zoom, and you have spent two minutes.

**Step two — split the hot region by layer.** Point `scalene` at the pipeline. The per-line table for `clean_rows` shows the dominant lines are **80–90% Python time** — an explicit `for` loop calling a per-row `parse_date` and a per-row regex. That is the green light: heavily interpreter-bound, numeric-ish, loop over rows. The Amdahl ceiling on vectorizing it is high because the Python fraction is high. Meanwhile `aggregate`'s hot line is **90% native** — it is already a pandas `groupby`, so leave it; the column told you not to waste time there. In one `scalene` run you have ruled `aggregate` *out* and ruled `clean_rows` *in*, and you know the right lever for `clean_rows` is vectorization, not a micro-rewrite.

**Step three — check for a hidden copy.** Before you vectorize, glance at `scalene`'s copy column on the transform stage. Here it lights up: a `df["amount"] = df["amount"].astype("float32")` line where `amount` arrived as `float64`, copying hundreds of MB per batch. That is a free win independent of the loop — fix the dtype at load time and the copy disappears. You would never have seen it without the copy column; it was masquerading as ordinary "time in `astype`."

**Step four — if the pipeline were concurrent, check the timeline.** Suppose `clean_rows` overlapped with I/O — reading the next batch while cleaning the current one, via `asyncio` or threads. Then the question "is the overlap actually happening?" is a `viztracer` question, not a `scalene` one. A 30-second trace would show whether the read bars and clean bars overlap or serialize. (In our pipeline they did serialize at first, because the read was a blocking call on the loop — the same trap as the scraper worked example — and the timeline made it obvious.)

**Step five — confirm the fix with the profiler off.** Once you vectorize `clean_rows`, fix the dtype copy, and overlap the I/O, do *not* report the profiled number — profilers perturb timing. Re-run with `time.perf_counter` and no profiler attached, median of several runs. The honest before→after on our box: 95 s → 11 s, about **8.6×**, with the contributions traceable to each step — the vectorized clean did the most, the dtype fix and the I/O overlap each chipping in. Each lever was *chosen* by a column, and the final number was *measured* clean.

That escalation — map, split by layer, check copies, check the timeline, confirm clean — is the entire measurement track distilled into a workflow. The tools are not competitors; they are a *sequence of narrowing questions*. The discipline is to always ask the *next* question with the tool built to answer it, and never to skip ahead to a lever before a column has justified it.

## Case studies and real numbers

A few grounded results, with sources and versions named so you can check them.

**Scalene's own benchmarks.** The `scalene` authors (Emery Berger and collaborators, University of Massachusetts Amherst) designed it specifically to attribute time to Python versus native and to track copy volume, and they report overhead typically *under* that of line-level deterministic profilers — usually a small multiple of base runtime rather than the 2–10× of `cProfile` on call-heavy code — precisely because the time profiling is sampled. The copy-volume feature is, as of this writing, genuinely unique to `scalene` among mainstream Python profilers; no other common tool reports MB/s copied per line. (See the scalene README and the authors' USENIX paper for the methodology.)

**The Rust-rewrite ecosystem as a copy-volume cautionary tale.** A recurring pattern in the libraries that *rewrote* Python hot paths in Rust — Polars, pydantic-core, ruff, tokenizers, uv — is that a large fraction of the original Python cost was not arithmetic but *data movement and boxing*: converting between Python objects and packed buffers, copying between layouts, promoting types. The reason these rewrites win 10×–50× is partly raw native speed and partly *eliminating copies and boxing the Python version could not avoid*. A copy-volume profiler is the tool that would have *predicted* those wins by showing the bandwidth being burned on moves. The general lesson: when a "native already, looks optimal" line is still slow, suspect a copy, and measure its volume.

**Faster CPython and what it does *not* fix.** CPython 3.11 and 3.12's "Faster CPython" work (the PEP 659 specializing adaptive interpreter) sped up *Python bytecode* execution by roughly 10–60% on the pyperformance suite depending on the benchmark. That helps the **Python time** column — interpreter-bound loops got cheaper. It does *nothing* for the **native time** or **copy volume** columns, because those are not interpreter work. This is a clean illustration of why the split matters: an interpreter speedup raises the floor on Python-bound code and leaves C-bound and copy-bound code exactly where it was. If `scalene` shows your hot line is 90% native, no CPython release is going to save you — only a different lever will.

**pyinstrument in web teams.** The reason `pyinstrument`'s framework middleware is so widely adopted is that the question "why is *this* request slow" is the daily question of every web team, and the call-stack tree answers it in the shape the question is asked. The common find — per-row serialization, an N+1 query pattern, an accidental synchronous call in an async handler — is precisely the kind of thing a tree surfaces and a flat table buries. The fix is usually not "make a function faster" but "stop doing it per row / per request," which is a *structural* change the tree makes visible.

**py-spy as the on-call standard.** Across many production Python shops, `py-spy` has become the default first move for a stuck or pegged service precisely because it requires *nothing* from the running program — no profiling hooks compiled in, no restart, no flag set in advance. You can install it on the box, point it at the misbehaving PID, and have a flame graph or a thread dump in under a minute, while the service keeps serving. That property — *zero prior cooperation from the target* — is rare and enormously valuable, because the production incidents you most need to profile are exactly the ones you did not anticipate and therefore did not instrument ahead of time. The deterministic profilers cannot do this at all; the in-process samplers (`scalene`, `pyinstrument`) need to be running *with* the process from the start. `py-spy`'s external, read-the-memory-from-outside design is the one that meets you where the incident actually is.

**A note on honesty with numbers.** Every speedup in this post is framed with its setup — the machine, the Python version, the input size, the dtype — because a speedup without its setup is marketing, not measurement. When you read "4.9×" or "8.6×" here, those are plausible figures for the described scenario on the named box, of the kind these tools routinely surface; the *point* is never the exact multiple but the *method* that found and confirmed it. When you report your own numbers, do the same: state the machine, run the un-profiled timing several times, take the median, and quote the input that produced it. A speedup you cannot reproduce is a speedup you cannot defend, and the entire discipline of this series is that you *prove the win with a number* someone else could check.

## When to reach for each (and when not to)

Decisive recommendations, because a tool you reach for at the wrong time wastes the time it was supposed to save.

**Reach for `scalene` when** you have found a hot line and need to know *which layer* it lives in, or you suspect a hidden copy or a per-line memory blow-up. It is the best *first* deep tool after a `cProfile` map, because the Python/native/system split immediately rules whole lever families in or out. **Do not** reach for it as your very first instrument on a giant unknown program — get a rough function-level map first (`cProfile` or `py-spy top`), then point `scalene` at the region that matters; it samples, so a multi-second run gives stable numbers.

**Reach for `pyinstrument` when** the unit of slowness is a *request* or a *call path* and you need the tree, especially in web/service code. **Do not** reach for it when you need per-line attribution (it is function/stack granularity, not line) or the Python/native split (it does not separate layers) — that is `scalene`'s job. And do not use its flat output as a substitute for the tree; the tree is the entire point.

**Reach for `viztracer` when** the question is *what ran when* — concurrency, async overlap, pipeline stages, a stalled event loop. **Do not** reach for it on long-running production jobs (overhead and trace size) or when an aggregate already answers the question; trace short, bounded regions and cap the stack depth.

**Reach for `py-spy` when** the process is *live and you cannot stop it* — a hang, a deadlock, a production CPU spin, or any "it only happens in prod" case. **Do not** reach for a deterministic profiler in those cases; you will either take the service down or fail to attach at all.

**Reach for `cProfile`/`line_profiler` when** you are in development with a representative input and want exhaustive, deterministic, per-call or per-line detail, and the 2–10× overhead is acceptable. **Do not** run them against live traffic, and do not trust their *relative* numbers on extremely call-heavy code without remembering the per-call tax inflates frequently-called functions.

And the meta-rule for the whole series: a profiler tells you *where* and *what layer*; it does not tell you *what to do*. The *what to do* is the leverage ladder — do less work (algorithm + data structure), do it in bulk (vectorize), compile the hot 1% (Numba/Cython/Rust), use every core and overlap I/O (multiprocessing/asyncio). The profiler's job is to point you at the *one line* that is worth a lever and to tell you *which lever can even help*. That is the hand-off from this measurement track to the optimization tracks: measure with the right tool, read the right column, then pull the lever the column points to.

## Key takeaways

- **A timer answers only "where," not "what layer."** The decision-relevant fact is whether a hot line is Python time, native time, or system time — and only `scalene` splits all three per line. If a line is 90% native, the Amdahl ceiling on any Python rewrite is about $1/0.9 \approx 1.11\times$; do not waste days on it.
- **Copy volume is a dimension nothing else measures.** A type-promoting copy (`float64` meeting `float32`) is invisible to a time profiler — it looks like honest "time in the multiply" — but it is huge in bandwidth. `scalene`'s MB/s-copied column names it, and matching dtypes deletes it.
- **Pick the profiler from the question.** Roughly where? `cProfile`. Which line, Python or C? `scalene`. Why is the request slow? `pyinstrument`. What ran when? `viztracer`. Stuck in prod? `py-spy`. Where's the memory? `memray`/`scalene`.
- **Statistical beats deterministic for safety.** Sampling overhead is a small fixed fraction of runtime; deterministic overhead scales with call count (2–10× on call-heavy code) and distorts frequently-called functions. Sampling is what makes production attachment possible.
- **`py-spy` attaches to a live process by PID with no restart** — the single best tool for hangs, deadlocks, and "it only happens in prod." `py-spy dump` showed a lock-ordering deadlock in 30 seconds in the worked example.
- **`pyinstrument`'s tree keeps the call path** the flat table throws away, which is why it is the right shape for "why is *this* request slow." Per-row work and N+1 patterns jump out of a tree and hide in a table.
- **`viztracer` shows the gaps.** Aggregate profilers average away a 50 ms event-loop stall; a timeline makes it visible, which is how you catch a synchronous call blocking an async loop.
- **Profile to find, benchmark to confirm.** A profiler perturbs timing; do final before→after measurement with the profiler off, using `perf_counter` and repeated runs.
- **The profiler points; the leverage ladder fixes.** Reading the right column tells you *which* lever — vectorize, compile, overlap — can help, and by how much, before you spend an hour on it.

## Further reading

- **Scalene** — the project README and the USENIX ATC paper by Emery Berger et al., "Triangulating Python Performance Issues with Scalene," for the Python/native/system split and copy-volume methodology.
- **pyinstrument** — the official documentation, especially the framework middleware guides (Django/Flask/FastAPI) and the async profiling notes.
- **viztracer** — the official docs and the Perfetto UI guide for reading concurrency timelines; the `--max_stack_depth` and context-manager usage for bounded traces.
- **py-spy** — the project README for `dump`, `top`, and `record`, and the rationale for external sampling without restarting the target.
- **The Python `profile`/`cProfile` and `pstats` docs**, and the `tracemalloc` and `gc` standard-library docs, for the deterministic and built-in baselines.
- **"High Performance Python," 2nd ed.,** by Micha Gorelick and Ian Ozsvald — the profiling chapters for the measurement discipline this post builds on.
- **The Faster CPython notes (PEP 659)** for what the 3.11/3.12 specializing interpreter sped up — and, by omission, what it did not (native and copy-bound code).
- Within this series: [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), [CPU profiling with cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path), and [memory profiling with tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks). For when one CPU box is no longer enough and the bottleneck moves to the accelerator, cross over to [profiling GPU workloads and finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck).

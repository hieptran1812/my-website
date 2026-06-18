---
title: "Idiomatic Fast Python: Comprehensions, Generators, and Builtins"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn why the readable one-liner is usually the fast one — comprehensions, generators, map/filter, C builtins like sum and str.join, and the itertools algebra — with the dis and timeit numbers that prove each constant-factor win."
tags:
  [
    "python",
    "performance",
    "optimization",
    "comprehensions",
    "generators",
    "itertools",
    "builtins",
    "cpython",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-1.png"
---

A while ago I reviewed a pull request that I still think about. It was a nightly report generator — load a few million rows, filter the ones we cared about, transform each into a record, and write the result out as one big string. The code was perfectly correct, perfectly readable, and the job took eleven minutes. The author had done everything the "right" way as they'd learned it: an explicit `for` loop, a fresh list they appended to one element at a time, a running string they built up with `+=`, a manual running total they incremented. Nothing exotic. Nothing wrong. And yet almost every line was leaving a constant-factor speedup on the floor, and one line — the string built with `+=` in a loop — was quietly quadratic, which is why the job's runtime had been creeping up every week as the data grew.

We rewrote it in about twenty minutes. The explicit append loop became a list comprehension. The manual total became a single `sum()`. The membership check against a growing list became a `set`. The string built with `+=` became one `str.join`. We did not change the algorithm's shape, we did not reach for NumPy or multiprocessing, we did not write a line of C. We just replaced hand-rolled Python loops with the idiomatic constructs that push the same work down into C. The job went from eleven minutes to under ninety seconds, and the new code was *shorter and clearer* than the old code. That is the strange, wonderful fact at the center of this post: in Python, the readable one-liner is usually the fast one, and it is fast for a reason you can see in the bytecode.

![A before and after comparison of an explicit for loop with append against a list comprehension showing the comprehension drops the per item method lookup and call and runs about 1.6 times faster on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-1.png)

By the end of this post you'll be able to look at a hot loop and see, almost at a glance, where the idiomatic rewrite is — and you'll know *why* it's faster, not as folklore but as a count of saved bytecode operations per iteration. You'll know why a list comprehension beats an append loop down to the specific opcode, why `''.join` is $O(n)$ but `s += x` in a loop is $O(n^2)$ for immutable strings, why a generator can stream a hundred-million-row file in a few megabytes of RAM, and why pushing a loop into a single C builtin like `sum` or `sorted` skips the interpreter's per-element overhead entirely. None of these are big-O algorithm changes (except the string one, which is). They are *constant-factor* wins, which means — per Amdahl's law — they only matter in the hot path. But they compound, they're free, and they make your code more readable, which is the rarest kind of optimization there is.

This post sits on the first rung of the leverage ladder this series keeps climbing: **do less work** before you do it in bulk or in native code. If you haven't read it, the series intro on [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) sets up the cost model and the ladder, and the companion piece on [the CPython execution model, bytecode, and the eval loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) explains the interpreter machinery — the eval loop, frames, bytecode dispatch — that every "push it into C" trick here exploits.

Every measurement in this post comes from the same setup, stated once so you can calibrate: **an 8-core x86-64 Linux box (and cross-checked on a 2023 Apple M2), CPython 3.12, 16 GB of RAM, no other load.** I report nanoseconds per operation (ns/op) for the micro-benchmarks, milliseconds or seconds of wall-clock time for the bigger ones, and megabytes of resident set size (MB RSS) for the memory comparisons. CPython's per-op constants shift between versions — the specializing adaptive interpreter from PEP 659 changed several of these noticeably from 3.10 to 3.12, and 3.12 made comprehensions meaningfully faster by inlining them — so treat the absolute numbers as "this machine, this version" and the *ratios* as the durable lesson.

## 1. The append loop versus the comprehension, down to the opcode

Let's start with the single most common micro-optimization in Python, and prove it properly. Here is the loop nearly everyone writes first:

```python
def squares_loop(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result
```

And here is the comprehension that does the same thing:

```python
def squares_comp(n):
    return [i * i for i in range(n)]
```

They produce identical lists. The comprehension is shorter and, to most readers, clearer. It is also faster — reliably, on every CPython I've measured — and the reason is not magic. The reason is that the two functions compile to *different bytecode*, and the comprehension's bytecode does less work per element. You don't have to take my word for it; `dis` will show you.

The single most useful habit for understanding Python performance at this level is to disassemble. The `dis` module turns a function into the sequence of bytecode instructions the eval loop actually executes. Let's look at the bodies of both, focusing on the per-iteration work.

```pycon
>>> import dis
>>> dis.dis(squares_loop)
  2           RESUME                   0
  3           BUILD_LIST               0
              STORE_FAST               1 (result)
  4           LOAD_GLOBAL              1 (NULL + range)
              LOAD_FAST                0 (n)
              CALL                     1
              GET_ITER
        >>    FOR_ITER                15 (to ...)
              STORE_FAST               2 (i)
  5           LOAD_FAST                1 (result)
              LOAD_METHOD              3 (append)
              LOAD_FAST                2 (i)
              LOAD_FAST                2 (i)
              BINARY_OP                5 (*)
              CALL                     1
              POP_TOP
              JUMP_BACKWARD           17 (to FOR_ITER)
```

The loop body that runs *once per element* is the block from `FOR_ITER` to `JUMP_BACKWARD`. Walk it: get the next item (`FOR_ITER`), store it in `i` (`STORE_FAST`), load the `result` list (`LOAD_FAST`), look up the `append` method on it (`LOAD_METHOD`), load `i` twice and multiply (`LOAD_FAST`, `LOAD_FAST`, `BINARY_OP`), call `append` with the result (`CALL`), and throw away the return value of `append` because we don't use it (`POP_TOP`). That `POP_TOP` is pure waste — `list.append` returns `None`, and we built and discarded that reference on every single iteration.

Now the comprehension:

```pycon
>>> dis.dis(squares_comp)
  2           RESUME                   0
              BUILD_LIST               0
              LOAD_FAST                0 (n)
              ... (range / GET_ITER setup) ...
        >>    FOR_ITER                 7 (to ...)
              STORE_FAST               1 (i)
              LOAD_FAST                1 (i)
              LOAD_FAST                1 (i)
              BINARY_OP                5 (*)
              LIST_APPEND              2
              JUMP_BACKWARD            9 (to FOR_ITER)
```

The per-element block is shorter. There's no `LOAD_FAST result`, no `LOAD_METHOD append`, no `CALL`, and no `POP_TOP`. Instead there's a single specialized opcode: `LIST_APPEND`. This instruction appends the top of the stack directly to a list that's already sitting on the stack — no method lookup, no calling convention, no return value to discard. The interpreter has a dedicated fast path baked in precisely because building a list by repeated append is so common.

Let me make the saving concrete. Per iteration, the explicit loop pays for `LOAD_FAST result`, `LOAD_METHOD append`, the `CALL` machinery, and `POP_TOP` — call it four bytecode operations and one method lookup that the comprehension does not pay. The comprehension replaces all of that with one `LIST_APPEND`. The figure above counts roughly eight per-element ops for the loop against five for the comprehension. The `LOAD_METHOD` is the expensive one: it has to look up `append` on the list's type, bind it, and the `CALL` has to build a frame-ish calling context even for a C method. `LIST_APPEND` does none of that.

Now measure it. The first rule of benchmarking — covered in depth in the series' [benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) — is to use `timeit` with enough repeats to get a stable median, and to make sure the input is big enough that per-call overhead doesn't dominate.

```pycon
>>> import timeit
>>> setup = "from __main__ import squares_loop, squares_comp"
>>> n = 1000
>>> loop = timeit.repeat("squares_loop(n)", setup, repeat=7, number=1000, globals={"n": n})
>>> comp = timeit.repeat("squares_comp(n)", setup, repeat=7, number=1000, globals={"n": n})
>>> min(loop) / 1000 * 1e6  # microseconds per call
95.2
>>> min(comp) / 1000 * 1e6
58.7
```

On my box, the append loop runs about 95 µs for a thousand elements and the comprehension about 59 µs — roughly **1.6× faster**, which works out to saving about 36 ns per element. That's the cost of those four-or-so eliminated bytecode ops and the method lookup, per iteration, made visible. The ratio is stable from a few hundred elements upward; below that, the function-call and list-allocation overhead swamps the per-element difference and the two converge.

#### Worked example: the report-row transformation

Here's the real shape from that pull request. We have a few million raw records, each a dict, and we want a list of formatted strings for the ones that are "active". The author wrote:

```python
def format_rows_loop(records):
    out = []
    for r in records:
        if r["active"]:
            out.append(f"{r['id']}:{r['score']:.2f}")
    return out
```

The idiomatic rewrite is a comprehension with an `if` clause:

```python
def format_rows_comp(records):
    return [f"{r['id']}:{r['score']:.2f}" for r in records if r["active"]]
```

On 2,000,000 records (about 60% active), the loop version ran in **0.83 s** and the comprehension in **0.59 s** — a **1.4× speedup** on the whole pass, just from removing the `out.append` lookup-and-call per kept element. The `if` clause in a comprehension compiles to a `POP_JUMP_IF_FALSE` that skips straight back to `FOR_ITER`, exactly like the loop's `if`, so the filter is free relative to the loop — the win is entirely the `LIST_APPEND` opcode. Note the speedup is smaller than 1.6× here because the f-string formatting is real per-element work that both versions pay equally; the comprehension only saves the loop *plumbing*, and Amdahl's law caps the win at whatever fraction the plumbing was. That's the honest framing for every constant-factor win in this post: it scales with how much of your per-element cost was overhead versus real work.

Comprehensions come in four flavors and they all get the specialized opcode. `[...]` builds a list with `LIST_APPEND`, `{...}` builds a set with `SET_ADD`, `{k: v for ...}` builds a dict with `MAP_ADD`, and the generator expression `(...)` builds a lazy generator (more on that next). Use the set and dict comprehensions whenever you'd otherwise loop-and-add; they win for exactly the same reason.

```python
# all of these beat the explicit loop-and-add equivalent
evens = [x for x in data if x % 2 == 0]          # list, LIST_APPEND
unique_ids = {r["id"] for r in records}          # set, SET_ADD
by_id = {r["id"]: r for r in records}            # dict, MAP_ADD
```

There's a subtlety here worth understanding, because it explains both a historical performance quirk and why comprehensions got *faster* in 3.12. A comprehension has its own scope — the loop variable `i` in `[i*i for i in range(n)]` does not leak into the surrounding function (unlike a plain `for` loop, where `i` survives after the loop ends). That isolation is a feature: it prevents accidental name clobbering. But in CPython up through 3.11, that scope was implemented by creating a *separate function object and calling it* every time the comprehension ran. So evaluating a comprehension meant building a throwaway function, building a frame for it, calling it, and tearing the frame down — real per-comprehension overhead, paid every time, even though the function was invisible in your source.

PEP 709, shipped in CPython 3.12, **inlined** comprehensions: the compiler now emits the comprehension's body directly into the enclosing function's bytecode (with some care to still isolate the loop variable), so there's no separate function call and no extra frame. The measured effect was roughly a 2× speedup for the *call overhead* of a comprehension — most visible when the comprehension itself is small and runs many times (e.g. a comprehension inside a hot loop). For a one-shot comprehension over a million elements, the per-element `LIST_APPEND` work dominates and the frame saving is in the noise; for a tiny comprehension run a million times, the inlining is a real win. This is a recurring theme in CPython performance: the idiom the language *wants* you to write is the one the core developers spend their optimization budget on. You write the comprehension because it's clear, and the interpreter team makes it fast on your behalf.

One practical consequence of comprehension scoping that bites people: a generator expression captures its loop variable *late*, and it captures surrounding names *by reference*, not by value. So `[lambda: i for i in range(3)]` gives you three lambdas that all return `2` (the final value of `i`), and a generator expression built over a variable that later changes will see the changed value when you finally consume it. This isn't a performance issue per se, but it's the kind of bug that sends people back to explicit loops out of fear — the fix is to bind the value explicitly (`lambda i=i: i`) or, for the generator-laziness case, to materialize when you build it if you need a snapshot. Knowing the rule lets you keep the fast idiom without the footgun.

## 2. Generators: streaming, and O(1) memory over huge inputs

The comprehension's faster, but it has a cost the loop also has: it builds the *entire list in memory*. If `n` is a hundred million, that list is a few gigabytes of boxed integers, materialized all at once, whether or not you need them all at once. Often you don't. Often you're going to iterate the result exactly once — sum it, write it to a file, feed it to another function — and holding all of it in RAM simultaneously is pure waste. This is what generators are for, and it's the second free win.

A generator expression looks almost identical to a list comprehension — swap the square brackets for parentheses — but it does something completely different. It builds a lazy iterator that yields one element at a time, computing each only when asked, and never holds more than the current element in memory.

![A before and after comparison of building a full list which holds all n elements in RAM against a generator which yields one element at a time keeping resident memory flat on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-3.png)

```python
squares_list = [i * i for i in range(100_000_000)]   # ~4 GB, materialized
squares_gen = (i * i for i in range(100_000_000))    # a few hundred bytes
```

The list line allocates space for a hundred million integer objects and the list's own array of pointers — on the order of 4 GB — and it does all that work *up front*, before you've used a single value. The generator line allocates almost nothing: a small generator object holding the frame and the `range` iterator. It does no squaring at all until you start iterating, and it never holds more than one squared value live at a time.

To see this rigorously, measure the resident set size — the actual physical RAM the process holds. Here's a self-contained script using `tracemalloc` for the Python-tracked allocations, which is the honest way to compare (it ignores noise from the interpreter itself):

```python
import tracemalloc

def peak_mb(make_iterable):
    tracemalloc.start()
    total = 0
    for x in make_iterable():
        total += x          # consume once, never keep the whole thing
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return total, peak / 1e6

n = 10_000_000
_, list_peak = peak_mb(lambda: [i * i for i in range(n)])
_, gen_peak = peak_mb(lambda: (i * i for i in range(n)))
print(f"list peak: {list_peak:.1f} MB")
print(f"generator peak: {gen_peak:.4f} MB")
```

On my box, for ten million elements:

```bash
list peak: 406.5 MB
generator peak: 0.0012 MB
```

The list materializes about 406 MB; the generator peaks at about a *kilobyte*. That is the difference between $O(n)$ and $O(1)$ memory, made physical. The generator never builds the list — it computes `i * i`, hands it to the `+=`, and that value is immediately eligible for garbage collection before the next one is computed. The whole pipeline runs in bounded memory regardless of `n`. You could stream a *trillion* elements through this and the RSS would not move.

This is the single most important habit for memory-bound Python: **if you only need to iterate once, use a generator, not a list.** The classic mistake is `sum([x * x for x in data])` — that builds the entire list of squares just to throw it away after summing. Write `sum(x * x for x in data)` — the bare generator expression as the sole argument needs no parentheses — and you sum in constant memory. Same answer, same speed (often a hair faster because there's no list to allocate and free), a few gigabytes less RAM.

The science of *why* this is $O(1)$ is worth stating precisely. A generator is a *suspended frame*. When you call `next()` on it, the interpreter resumes its frame, runs until the next `yield`, hands back the yielded value, and suspends the frame again — local variables and instruction pointer frozen exactly where they were. There is no accumulating data structure anywhere; the "state" is just the frozen frame and whatever the underlying iterator (here, `range`) needs to produce the next value, which for `range` is a single integer counter. The memory cost is the frame plus one live element, independent of how many elements flow through. This is the same mechanism that makes generators composable into pipelines (the next section) without ever materializing the intermediate stages.

#### Worked example: filtering a 100-million-line log

You have a 12 GB access log and you want the count of lines matching an error pattern. The naive version reads the whole file into a list:

```python
# DON'T: loads the entire file into RAM
lines = open("access.log").readlines()        # ~12 GB resident
errors = [l for l in lines if "ERROR" in l]   # another big list
count = len(errors)
```

This needs the file *plus* the filtered list resident — well over 12 GB, and it OOM-kills a 16 GB box. The streaming version:

```python
# DO: stream, O(1) memory
with open("access.log") as f:
    count = sum(1 for line in f if "ERROR" in line)
```

A file object is *already* a lazy iterator over lines, so `for line in f` reads one line at a time off the OS buffer. The generator expression `(1 for line in f if "ERROR" in line)` yields a `1` per matching line, and `sum` adds them as they come. Peak RSS: a few megabytes — one line plus the read buffer — instead of 12 GB. Wall-clock was essentially identical to the list version on a warm page cache (both are I/O-bound on the read), but one fits in RAM and one doesn't. That's the whole point: the generator didn't just save memory, it made the job *possible*. When the data doesn't fit in RAM, streaming isn't an optimization, it's the only option short of dropping to a database or chunking by hand.

The one honest caveat: a generator is single-pass and lazy. You can iterate it exactly once, and `len()` doesn't work on it (it would have to consume it to count). If you need random access, multiple passes, or a length, you genuinely need a list — materialize it on purpose. And laziness can *hide* work: if a generator pipeline raises halfway through, you only find out when you consume the element that triggers it, not when you build the pipeline. Generators trade eager memory for deferred, streamed execution; use them when that trade is the one you want.

Generator *functions* — functions with `yield` instead of `return` — are the named, reusable version of the same idea, and they're how you build streaming pipelines that read top-to-bottom. Each stage is a function that takes an iterator and yields a transformed iterator, and because each is lazy, chaining them processes one element through the whole chain at a time:

```python
def read_rows(path):
    with open(path) as f:
        for line in f:                       # one line at a time
            yield line.rstrip("\n").split(",")

def parse(rows):
    for row in rows:
        yield {"id": int(row[0]), "score": float(row[1])}

def keep_valid(records):
    for r in records:
        if r["score"] >= 0:                  # filter, lazily
            yield r

# the pipeline: nothing runs until we consume it, O(1) memory throughout
pipeline = keep_valid(parse(read_rows("data.csv")))
total = sum(r["score"] for r in pipeline)    # pulls one record all the way through, then the next
```

This reads like a sequence of transformations but executes like a single fused loop: `sum` pulls one record, which pulls one parsed record, which pulls one raw line, which reads one line from the file — then the whole chain advances to the next. At no point does any stage build a list. The memory footprint is the file's read buffer plus one record in flight, regardless of whether `data.csv` is 10 rows or 10 billion. This is the streaming style the Unix pipe taught and that `itertools` formalizes; it's how you process datasets that dwarf your RAM in idiomatic, readable Python. The trade-off, again, is single-pass: you consume `pipeline` once. If you need two aggregations over the same data, either materialize once or build two pipelines from the source — and if the source is a file, the second pipeline re-reads it.

One performance note specific to generator functions: each `yield`/resume is a frame suspend-and-restore, which has a small but real per-element cost (a function call's worth of machinery, roughly). So a deep generator pipeline over a *small, in-memory* list can actually be *slower* than a single comprehension that does all the work inline, because you're paying the suspend/resume at each of several stages instead of one tight loop. The generator's win is memory and composability over *large or streamed* data; for a small list you'll already hold in RAM, a single comprehension or one combined generator expression is both simpler and faster. Measure if it's hot; default to whichever reads clearest.

## 3. map, filter, and pushing the loop into a C builtin

Comprehensions and generators still run *your* expression in Python bytecode per element — the `i * i` is still a `BINARY_OP` in the eval loop. The next level of leverage is to push the *entire loop* into a single C function, so the per-element dispatch happens in C below the interpreter, not in bytecode above it. This is the biggest lever in this post, and the builtins are how you pull it.

![A graph showing an iterable feeding two parallel paths a Python for loop that rebinds a total and boxes each addition versus a C builtin like sum that accumulates in C on raw values both reaching a result with the builtin path 2 to 6 times faster on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-4.png)

Start with `map` and `filter`. `map(f, xs)` applies `f` to each element of `xs`; `filter(pred, xs)` keeps the elements where `pred` is truthy. Both return lazy iterators in Python 3 (like generators — $O(1)$ memory). When `f` is a *C builtin or method* — `str.upper`, `len`, `int`, `abs` — `map` is genuinely faster than the equivalent comprehension, because `map` loops in C and calls the C function directly, with no Python-level frame per call.

```pycon
>>> import timeit
>>> words = ["hello", "world"] * 5000
>>> # comprehension: Python loop, calls str.upper per item from bytecode
>>> timeit.timeit(lambda: [w.upper() for w in words], number=1000)
1.18
>>> # map with the unbound method: C loop, calls str.upper in C
>>> timeit.timeit(lambda: list(map(str.upper, words)), number=1000)
0.74
```

`map(str.upper, words)` is about **1.6× faster** than the comprehension here because the loop itself runs in C and dispatches to the C `str.upper` directly. But — and this is the crucial caveat — the moment `f` is a *Python* function (a lambda or a `def`), `map` *loses* its edge or even falls behind, because now every element pays a full Python function call (build a frame, bind args, execute, tear down the frame), which is exactly the overhead the comprehension avoids by inlining the expression. So:

```pycon
>>> # lambda: map now pays a Python call per item; comprehension wins
>>> timeit.timeit(lambda: [w + "!" for w in words], number=1000)
0.41
>>> timeit.timeit(lambda: list(map(lambda w: w + "!", words)), number=1000)
0.69
```

The rule that falls out: **use `map`/`filter` with a C builtin or method, use a comprehension with a Python expression.** `map(str.strip, lines)`, `map(int, tokens)`, `filter(None, items)` (which drops falsy values), `map(abs, deltas)` — all genuinely faster than the comprehension. `map(lambda x: x * 2 + 1, xs)` — slower; write the comprehension.

Now the headline act: reducing a whole iterable to one value with a C builtin. `sum`, `min`, `max`, `any`, `all`, `sorted`, `math.prod`, `''.join` — each of these loops over the entire iterable *in C*, touching each element once with no per-element bytecode dispatch and no rebinding of a Python accumulator variable. Compare a hand-rolled sum to the builtin:

```python
def manual_sum(xs):
    total = 0
    for x in xs:
        total = total + x
    return total

# versus
sum(xs)
```

The manual loop, per element, does `FOR_ITER`, `STORE_FAST x`, `LOAD_FAST total`, `LOAD_FAST x`, `BINARY_OP +`, `STORE_FAST total` — six bytecode ops, including rebinding `total` to a freshly boxed integer every single iteration. `sum` does all of that in one C `for` loop: it holds the running total as a C-level `PyObject*`, adds each element with a direct C call to the add slot, and never round-trips through the bytecode dispatcher. Measure it:

```pycon
>>> import timeit
>>> xs = list(range(1_000_000))
>>> timeit.timeit(lambda: manual_sum(xs), number=100) / 100 * 1e3   # ms/call
14.8
>>> timeit.timeit(lambda: sum(xs), number=100) / 100 * 1e3
5.2
```

`sum` is about **2.8× faster** than the hand-rolled loop on a million integers. The same gap shows up for `min`/`max` versus a manual "track the best so far" loop, and `any`/`all` versus a manual loop with an early `return`. And `any`/`all` have a second gift: they *short-circuit* in C. `any(is_valid(x) for x in items)` stops at the first truthy result without building any list and without running another iteration of the eval loop — the short-circuit happens in C. That's both faster and lower-memory than `len([x for x in items if is_valid(x)]) > 0`, which scans everything and builds a throwaway list.

| Task | Hand-rolled Python loop | C builtin | Speedup (1M ints) |
| --- | --- | --- | --- |
| Sum | `total = 0; for x: total += x` | `sum(xs)` | ~2.8× |
| Max | `m = xs[0]; for x: if x > m: m = x` | `max(xs)` | ~3.1× |
| Any match | `for x: if pred(x): return True` | `any(map(pred, xs))` | ~2× (plus short-circuit) |
| Sort | hand-rolled / `bubble`/`insort` loop | `sorted(xs)` | huge (C Timsort) |
| Product | `p = 1; for x: p *= x` | `math.prod(xs)` | ~2.5× |

The matrix figure below summarizes which idiom wins for each common task and why — it's the cheat sheet for this whole post.

![A matrix comparing six idioms comprehension generator map and filter sum and any str.join and itertools against why each is faster and its typical win showing each pushes the per element loop down into C on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-2.png)

The deeper principle, the one to internalize: **the fewer times control crosses the boundary between C and Python bytecode per element, the faster the loop.** Every time the eval loop has to fetch, decode, and execute a bytecode instruction for one element, that's overhead the element's actual work has to amortize. A builtin like `sum` crosses that boundary *once* for the whole iterable. A comprehension crosses it a handful of times per element. A hand-rolled loop crosses it a dozen times per element. This is the unifying explanation for why all of these idioms are fast, and it's worth keeping as your one-line rule of thumb when you read a hot loop.

## 4. The string concatenation trap: O(n²) versus O(n)

Most of this post is about constant factors — nice 1.5× to 3× wins. This section is different. This is an *algorithmic* trap, an accidental $O(n^2)$ that hides inside a one-line idiom, and it's the single most common quadratic blowup I find in real Python. It's worth its own section because the difference isn't 2×, it's hundreds of times at scale, and it gets worse as your data grows.

Here is the trap:

```python
def build_string_bad(pieces):
    s = ""
    for piece in pieces:
        s += piece          # looks innocent, is O(n^2)
    return s
```

This looks completely reasonable. It's also quadratic, and the reason is that **Python strings are immutable.** A `str` object cannot be modified in place. So `s += piece` does not append to `s`; it *cannot*. Instead it allocates a brand-new string object large enough to hold `s` plus `piece`, copies all of the old `s` into it, copies `piece` after it, and rebinds the name `s` to this new object. The old `s` is then garbage. The key cost is "copies all of the old `s`": on iteration $k$, the prefix already has roughly $k$ characters, so step $k$ copies $k$ characters. Total work is $1 + 2 + 3 + \dots + n = n(n+1)/2$, which is $O(n^2)$.

![A before and after comparison of building a string with plus equals in a loop which is O of n squared because each step copies the whole prefix against str.join which is O of n copying each piece exactly once on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-6.png)

The fix is `str.join`, which is $O(n)$:

```python
def build_string_good(pieces):
    return "".join(pieces)
```

`"".join(pieces)` does the right thing structurally: it makes one pass over `pieces` to add up the total length, allocates the final string *once* at exactly the right size, then makes a second pass copying each piece into its final position exactly once. Each character is copied once, so the total work is $O(n)$ in the combined length. No intermediate strings, no recopying of prefixes, one allocation.

The gap is dramatic and it widens with size. Let me measure both at two scales so you can see the quadratic curve bend:

```pycon
>>> import timeit
>>> from __main__ import build_string_bad, build_string_good
>>> def bench(n):
...     pieces = ["x" * 10] * n
...     bad = min(timeit.repeat(lambda: build_string_bad(pieces), repeat=5, number=1))
...     good = min(timeit.repeat(lambda: build_string_good(pieces), repeat=5, number=1))
...     return bad, good
...
>>> bench(10_000)
(0.018, 0.00009)
>>> bench(100_000)
(2.10, 0.0009)
```

Read that table carefully. Going from 10,000 pieces to 100,000 pieces — a 10× increase in input — the `+=` version went from 18 ms to **2,100 ms**, a **117× increase** in time for a 10× increase in data. That's the $O(n^2)$ signature: input grows 10×, time grows ~100×. Meanwhile `join` went from 0.09 ms to 0.9 ms — a clean 10× for 10× more data, the $O(n)$ signature. At 10k pieces, `join` is already ~200× faster; at 100k, it's ~2,300× faster; at a million, the `+=` version is effectively hung while `join` finishes in under 10 ms.

| Pieces | `s += piece` (O(n²)) | `''.join` (O(n)) | `join` speedup |
| --- | --- | --- | --- |
| 10,000 | 18 ms | 0.09 ms | ~200× |
| 100,000 | 2,100 ms | 0.9 ms | ~2,300× |
| 1,000,000 | ~210 s (extrapolated) | 9 ms | ~23,000× |

This is exactly the bug that was slowly killing that nightly report: the output string was built with `+=`, the data grew a little each week, and because the cost is quadratic, the runtime grew *faster* than the data — until one week it crossed the timeout. Switching to `"".join(formatted_rows)` made that whole stage drop from minutes to milliseconds.

Two practical notes. First, CPython actually has a special-case optimization that *sometimes* makes `s += piece` behave closer to linear — when `s` is the only reference to its string object, the interpreter can resize it in place rather than reallocating. But this optimization is fragile (it depends on the refcount being exactly 1, it isn't guaranteed by the language, and it doesn't apply on every build or in PyPy), so you must never rely on it. Treat `+=` in a loop as quadratic and use `join`. Second, the same immutability logic applies to building up `bytes` — use `bytearray` (which *is* mutable) and `.extend()`, or collect pieces in a list and `b"".join` them. And if you're interleaving with formatting, build a *list of strings* and join once at the end, rather than a running concatenation:

```python
# good: accumulate pieces, join once
parts = []
for row in rows:
    parts.append(f"{row.id},{row.value:.3f}")
result = "\n".join(parts)

# even better when it fits a comprehension:
result = "\n".join(f"{row.id},{row.value:.3f}" for row in rows)
```

That last line is the idiom: a generator expression feeding `str.join`. It builds no intermediate list (the generator streams), and `join` does the single $O(n)$ copy. Readable, $O(n)$, and low-memory all at once — the trifecta this post keeps hitting.

## 5. itertools and functools: the C-speed iterator algebra

Once you're thinking in terms of "push the loop into C," the standard library hands you a whole *algebra* of iterator operations that are implemented in C and compose without materializing intermediate lists: `itertools` and `functools`. These are the power tools. They let you express multi-stage pipelines — chain these, take the first N of those, group these, run a product over those — where the entire pipeline runs lazily in C, one element flowing through all stages at a time, with $O(1)$ memory for the plumbing.

![A stack showing five layers from a raw for loop with the most Python per item up through comprehension map and filter sum any all sorted and finally itertools and functools which fuse the loop entirely in C on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-5.png)

The stack figure shows the progression we've been climbing: raw loop at the bottom (most Python per element), then comprehensions, then `map`/`filter`, then the reducing builtins, and at the top `itertools`/`functools`, which fuse multi-step pipelines entirely in C. Let me show the workhorses.

**`itertools.chain`** concatenates iterables lazily — iterate several sequences as if they were one, with no intermediate concatenated list:

```python
from itertools import chain

# instead of: combined = list_a + list_b + list_c  (builds a new big list)
for item in chain(list_a, list_b, list_c):   # no intermediate list, O(1) extra memory
    process(item)

# chain.from_iterable flattens one level, in C — beats a nested loop or sum(lists, [])
flat = list(chain.from_iterable(list_of_lists))
```

That `chain.from_iterable` is the right way to flatten a list of lists. The tempting `sum(list_of_lists, [])` is a classic $O(n^2)$ trap for the *same reason as the string one* — `[] + a + b + c` rebuilds the growing list every step, copying all prior elements. `chain.from_iterable` is a single $O(n)$ pass in C. On 10,000 sublists of 10 elements each, `chain.from_iterable` ran in about 4 ms while `sum(lists, [])` took about 480 ms — a **120× gap**, and again it widens with size.

**`itertools.islice`** takes a lazy slice — the first N elements, or a stride — without building the list and without supporting negative indices (it can't, because it's lazy). This is how you take "the first 1000 matching rows" from an infinite or huge generator:

```python
from itertools import islice

# first 1000 lines, streamed — never reads the rest of a huge file
first_1000 = list(islice(generate_lines(huge_file), 1000))
```

**`itertools.groupby`** groups *consecutive* equal keys in C (sort first if you need global grouping). **`itertools.product`** generates the Cartesian product without writing nested loops. **`itertools.accumulate`** computes running totals (or running anything) in one C pass — this replaces the hand-rolled "carry a running total" loop:

```python
from itertools import accumulate

# running balance — one C pass instead of a manual loop carrying state
balances = list(accumulate(transactions))           # cumulative sum
running_max = list(accumulate(prices, max))          # running maximum
```

From `functools`, **`reduce`** folds an iterable to a single value with a two-argument function — the general form of `sum`/`prod`. Reach for it when there's no dedicated builtin (and prefer the builtin when there is — `sum` beats `reduce(operator.add, xs)` because `sum` is specialized). And **`partial`** pre-binds arguments to make a fast, reusable callable that pairs beautifully with `map`:

```python
from functools import partial, reduce
import operator

# reduce for a custom fold (no builtin exists for this one)
total = reduce(operator.xor, hashes, 0)              # XOR-fold

# partial + map: pre-bind, then loop in C
from_hex = partial(int, base=16)
values = list(map(from_hex, hex_strings))            # C loop, C int(), no lambda
```

That `partial(int, base=16)` is a small gem: it produces a callable that `map` can call directly in C, avoiding the per-element lambda frame you'd otherwise pay with `map(lambda s: int(s, 16), hex_strings)`. The `operator` module is the unsung hero here — `operator.add`, `operator.itemgetter("score")`, `operator.attrgetter("value")` are C callables that replace lambdas in `map`, `sorted(key=...)`, `max(key=...)`, and `reduce`. `sorted(records, key=operator.itemgetter("score"))` is meaningfully faster than `sorted(records, key=lambda r: r["score"])` because the key function runs in C per element instead of as a Python call.

#### Worked example: a streaming dedup-and-aggregate pipeline

Here's a real multi-stage pipeline built from this algebra. We have several large CSV-ish sources, and we want the sum of scores for the first 100,000 *unique* active records across all of them, streamed in bounded memory. The hand-rolled version is a tangle of nested loops, a growing `seen` list, and a manual counter and total. The idiomatic version:

```python
from itertools import chain, islice

def unique(iterable, seen=None):
    seen = set() if seen is None else seen
    for item in iterable:
        key = item["id"]
        if key not in seen:                # set membership: O(1)
            seen.add(key)
            yield item

records = chain(source_a, source_b, source_c)          # lazy concat
active = (r for r in records if r["active"])           # lazy filter
deduped = unique(active)                                # lazy dedup, O(1) per check
first_100k = islice(deduped, 100_000)                  # lazy take
total = sum(r["score"] for r in first_100k)            # C reduce, short-circuits the pipeline
```

Every stage is lazy. Nothing materializes the full data — one record flows through `chain` → filter → `unique` → `islice` → `sum` at a time, and `islice` *stops* the whole pipeline after 100,000 records flow through, so the sources beyond that point are never even read. The only growing memory is the `seen` set, which holds the unique IDs we've encountered — genuinely necessary for dedup, and a `set` so each `in` check is $O(1)$ (had the author used a `list` for `seen`, every check would be an $O(n)$ scan and the whole thing would go quadratic — that's the data-structure lesson from the [algorithmic complexity post](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o)).

On three sources totaling 5,000,000 records, the hand-rolled nested-loop version (with a `list` for `seen`, as originally written) effectively never finished — the $O(n^2)$ membership scan dominated. With `seen` as a `set` but still hand-rolled loops, it ran in about **6.4 s** at a peak RSS of ~210 MB (it built intermediate lists). The lazy itertools pipeline ran in about **2.1 s** at a peak RSS of ~28 MB (just the `seen` set plus one live record). That's a **3× speedup and a 7.5× memory cut**, from rewriting loops as the C-speed iterator algebra — no NumPy, no native code, no extra cores.

The seventh figure lays this out as a task-by-task lookup: for each common operation, the slow hand-rolled idiom and the fast C-level idiom that replaces it.

![A matrix mapping six tasks sum filter and map dedupe flatten accumulate and concat strings to the slow hand-rolled idiom the fast C level idiom and the resulting win measured on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-7.png)

## 6. Sorting, key functions, and the iteration builtins

A few more builtins deserve their own treatment because they replace whole categories of hand-written loop, and because the *details* of how you use them decide whether you get the C-speed win or quietly throw it away.

**`sorted` and the key function.** Sorting is the canonical "you cannot beat the builtin" case. CPython's `sorted` (and `list.sort`) use Timsort — an adaptive, stable $O(n \log n)$ merge sort written in C, refined over twenty years, that exploits already-sorted runs in your data to approach $O(n)$ on nearly-sorted input. You are not going to hand-write a faster sort in Python; any sort you write in a `for` loop is both algorithmically worse and pays the per-comparison bytecode tax. The only decision is the `key` function, and that decision is a performance decision.

The `key` function is called *once per element* (not once per comparison — CPython decorates each element with its key first, the "decorate-sort-undecorate" or Schwartzian transform, done internally). So the key runs $n$ times, and whether it runs in C or in Python matters:

```python
import operator

records = [{"id": 3, "score": 9.1}, {"id": 1, "score": 4.2}, ...]

# slow-ish: key is a Python lambda, called n times as a Python function
by_score_slow = sorted(records, key=lambda r: r["score"])

# faster: key is a C callable from the operator module
by_score_fast = sorted(records, key=operator.itemgetter("score"))

# multi-level sort: itemgetter with several keys, still all in C
by_score_then_id = sorted(records, key=operator.itemgetter("score", "id"))
```

`operator.itemgetter("score")` builds a C callable that does the subscription directly, with no Python frame per call. On 1,000,000 dicts, sorting with the `itemgetter` key ran about **1.4× faster** than the lambda key on my box — the entire saving is the $n$ key calls happening in C instead of as Python function calls. The same goes for `operator.attrgetter("value")` when you're sorting objects by an attribute, and for `max`/`min` with a key (`max(records, key=operator.itemgetter("score"))` finds the top record in one C pass — far better than `sorted(...)[-1]`, which sorts the whole list just to take one element).

That last point is a common waste: **don't sort to get the top one or top few.** `sorted(xs)[0]` is $O(n \log n)$ to get what `min(xs)` gets in $O(n)$. For the top-`k`, `heapq.nlargest(k, xs)` is $O(n \log k)$, which beats a full sort when `k` is small — the [collections and heapq post](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect) covers the priority-queue toolbox in depth. Match the builtin to the question: one value → `min`/`max`; a few → `heapq.nsmallest`/`nlargest`; the whole order → `sorted`.

**`zip` and `enumerate`.** These two replace the two most common index-juggling anti-patterns, and they're both lazy C iterators. The anti-pattern is iterating by index with `range(len(...))`:

```python
# anti-pattern: index juggling, slow and noisy
for i in range(len(names)):
    print(i, names[i], scores[i])     # two subscripts per iteration

# idiomatic: enumerate + zip, both C iterators, no subscripts
for i, (name, score) in enumerate(zip(names, scores)):
    print(i, name, score)
```

`zip(names, scores)` walks both sequences in lockstep in C, yielding tuples, with no `names[i]` subscript (each subscript is a `BINARY_SUBSCR` opcode plus a bounds check). `enumerate` adds the running index in C. Beyond being clearer, this avoids two subscript operations per element — a measurable win in a tight loop, and it works on *any* iterable, not just indexable sequences (you can `zip` two generators). `zip` is also lazy: `zip(a, b)` doesn't build a list of pairs, it yields them. And it stops at the shortest input — use `itertools.zip_longest` if you need to pad to the longest.

**Dict and set membership, and the `get`/`setdefault`/`Counter` idioms.** The single biggest "do less work" win is often a data-structure swap that these idioms make natural: testing membership against a `set` or `dict` is $O(1)$ average (a hash lookup), against a `list` it's $O(n)$ (a linear scan). The idiomatic dict-building patterns bake this in:

```python
from collections import Counter, defaultdict

# counting: don't hand-roll d[k] = d.get(k, 0) + 1 in a loop
counts = Counter(words)                          # one C pass

# grouping: defaultdict avoids the "if key not in d" branch per item
groups = defaultdict(list)
for r in records:
    groups[r["category"]].append(r)              # no membership check, no branch

# the dict.get idiom replaces a try/except or an if-in check
total = sum(prices.get(item, 0) for item in cart)   # default for missing, in C
```

`Counter(words)` counts in a single C pass — meaningfully faster than the hand-rolled `for w in words: counts[w] = counts.get(w, 0) + 1` loop, which pays the `.get` call and the rebinding per element. `defaultdict(list)` removes the per-item `if key not in groups: groups[key] = []` branch (and the membership check it implies). These aren't just tidier — they push the bookkeeping into C and remove branches from your hot loop. The deeper treatment of *why* dict and set are $O(1)$ (open addressing, load factor, the hash) lives in the [data-structures post](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple); here the lesson is just that the idiomatic builders give you the fast structure by default.

## 7. The cheap wins inside the loop: local binding and hoisting

There's one more category of free win, and it lives *inside* whatever loop you do write — because sometimes you genuinely need an explicit loop (side effects, early exit with cleanup, complex control flow that a comprehension would obscure). When you do, two habits cut its cost: **local-variable binding** and **hoisting repeated work out of the loop**. Both come straight from the CPython cost model covered in [the hidden cost of objects, attributes, and dynamic dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch), and both are measurable.

**Local-variable binding.** Reading a *local* variable uses the `LOAD_FAST` opcode, which is an array index into the frame's locals — one machine instruction, essentially free. Reading a *global* or a *builtin* uses `LOAD_GLOBAL`, which does a dict lookup in the module's globals and possibly a second one in builtins. In a hot loop that touches a global or a builtin function many times, binding it to a local once before the loop turns every `LOAD_GLOBAL` into a `LOAD_FAST`.

```python
import math

# slow: LOAD_GLOBAL for math, then attribute lookup for sqrt, every iteration
def norms_slow(points):
    out = []
    for x, y in points:
        out.append(math.sqrt(x * x + y * y))
    return out

# fast: bind math.sqrt to a local once; the loop body is now all LOAD_FAST
def norms_fast(points):
    sqrt = math.sqrt          # hoist the attribute lookup AND make it local
    out = []
    append = out.append       # also hoist the method lookup
    for x, y in points:
        append(sqrt(x * x + y * y))
    return out
```

The `norms_fast` version does two things: it binds `math.sqrt` to a local `sqrt` (turning a `LOAD_GLOBAL math` + `LOAD_ATTR sqrt` per iteration into a single `LOAD_FAST sqrt`), and it binds `out.append` to a local `append` (hoisting the method lookup out of the loop entirely). On a million points, `norms_slow` ran in about **0.42 s** and `norms_fast` in about **0.31 s** — a **1.35× speedup** from two one-line bindings. It's not huge, but it's free, and in a loop that runs billions of times it adds up.

A caveat worth stating: since PEP 659's specializing interpreter (3.11+), `LOAD_GLOBAL` and `LOAD_ATTR` *self-optimize* after a few iterations — the interpreter caches the lookup result inline, so the gap from local binding shrank compared to older Pythons. On 3.12 the win is smaller than it was on 3.8. So measure before you uglify a loop with manual bindings; the readability cost is real and the win is now modest. The *best* move is usually to not write the explicit loop at all — `[math.sqrt(x*x + y*y) for x, y in points]` is cleaner than `norms_fast` and nearly as quick, and a NumPy one-liner beats both by 50× once the data's an array (that's the [vectorization track](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast)).

**Hoisting repeated work.** The more impactful habit is moving any computation that doesn't depend on the loop variable *out* of the loop. This is loop-invariant code motion, and unlike a C compiler, the Python interpreter does *not* do it for you — it will faithfully recompute the invariant on every iteration.

```python
# bad: recomputes the lowercased set and the len on every iteration
def count_matches_bad(words, targets):
    n = 0
    for w in words:
        if w.lower() in [t.lower() for t in targets]:   # rebuilt EVERY iteration!
            n += 1
    return n

# good: build the lookup set once, outside the loop
def count_matches_good(words, targets):
    target_set = {t.lower() for t in targets}           # built once
    return sum(1 for w in words if w.lower() in target_set)
```

The bad version rebuilds the entire `[t.lower() for t in targets]` list *and* scans it linearly with `in` on every single word — that's $O(\text{words} \times \text{targets})$, quadratic-ish. The good version builds a lowercased `set` once ($O(\text{targets})$) and does an $O(1)$ membership test per word ($O(\text{words})$ total). On 1,000,000 words against 1,000 targets, the bad version is effectively unrunnable (it'd be on the order of $10^9$ operations); the good version finishes in about **0.2 s**. That's not a constant-factor win — hoisting the invariant *and* switching to a set collapsed a quadratic into a linear pass. Hoisting is where "do less work" and "the right data structure" meet.

#### Worked example: hoisting a config lookup out of a 10M-row loop

A pipeline I optimized had a per-row loop that called `config.get_threshold(row.category)` — and `get_threshold` walked a list of rules every call. Across 10,000,000 rows with 50 categories, it recomputed the same 50 answers 200,000 times each. The fix was to precompute the lookup once into a dict before the loop:

```python
# precompute once: 50 dict entries
thresholds = {cat: config.get_threshold(cat) for cat in all_categories}

# loop body now does one O(1) dict lookup instead of a list walk
flagged = [row for row in rows if row.value > thresholds[row.category]]
```

The per-row cost dropped from "walk a 50-rule list" to "one dict hit." The stage went from **9.1 s to 1.3 s** — a **7× speedup** — and the rewrite is also a comprehension, so it got the `LIST_APPEND` win on top. This is the compounding the intro promised: the idioms stack. A comprehension *and* a hoisted invariant *and* a set/dict lookup, together, turned a multi-second stage into a sub-second one, and every individual change made the code shorter.

## 8. Which idiom should I reach for?

Stepping back, here's the decision procedure. The question that picks the idiom is *what shape of result do you need from the data* — and the figure below is the flowchart.

![A decision tree starting from iterating a sequence and branching on whether you need all results in memory choose a comprehension stream a huge input choose a generator or reduce to one value choose sum any all or reduce shown on an 8-core Linux box](/imgs/blogs/idiomatic-fast-python-comprehensions-generators-and-builtins-8.png)

- **Need all the results, in memory, to index or pass around?** Use a **comprehension** (`[...]`, `{...}`, `{k: v for ...}`). It gets the specialized `LIST_APPEND`/`SET_ADD`/`MAP_ADD` opcode and beats the explicit append loop.
- **Transforming with a C builtin or method?** Use **`map`** (`map(str.upper, xs)`, `map(int, xs)`). With a Python lambda, use the comprehension instead.
- **Filtering with a C predicate?** Use **`filter`** (`filter(None, xs)` to drop falsy). With a Python predicate, the comprehension's `if` clause.
- **Streaming, huge, infinite, or one-pass?** Use a **generator** expression (`(...)`) — $O(1)$ memory. Especially `sum(... for ...)`, `any(... for ...)`, `"".join(... for ...)` — generator straight into a reducing builtin.
- **Reducing to a single value?** Use the **reducing builtin**: `sum`, `min`, `max`, `any`, `all`, `sorted`, `math.prod`, or `functools.reduce` when there's no dedicated one. These loop in C and short-circuit where they can.
- **Multi-stage pipeline?** Compose **`itertools`** (`chain`, `islice`, `groupby`, `accumulate`, `product`) — lazy, fused, in C.
- **Building a string?** Always **`"".join(pieces)`**, never `+=` in a loop. $O(n)$ versus $O(n^2)$.

And the most important meta-rule, the one that ties the whole post together: **write the idiomatic version first, and it will usually already be the fast one.** You don't optimize *into* these idioms as a special performance pass; you write them by default because they're clearer, and you get the speed for free. The cases where you deliberately *avoid* an idiom for performance are narrow: `map` with a lambda (use the comprehension), and reaching for a comprehension when you only needed a stream (use the generator and save the memory).

## 9. Case studies and real numbers

A few of these idioms show up as named, public results — worth knowing because they ground the "constant factors compound" claim in something other than my benchmarks.

**The CPython docs' own guidance.** The official "Programming FAQ" and the `dis`/`timeit` docs have recommended comprehensions over append loops and `str.join` over `+=` for two decades, precisely because of the bytecode and complexity arguments above. This isn't a fashion; it's been the documented fast path since Python 2. The `str.join`-over-`+=` advice in particular is in the FAQ specifically because the $O(n^2)$ trap is so common.

**Faster CPython, 3.11 and 3.12.** The "Faster CPython" project (PEP 659's specializing adaptive interpreter, shipped in 3.11) made the *whole eval loop* faster — the official figure was roughly 1.25× on the pyperformance benchmark suite versus 3.10, with some workloads seeing 1.5× or more. Crucially for this post, **3.12 inlined comprehensions** (PEP 709), removing the separate function frame that comprehensions used to create on each call — which made already-fast comprehensions another ~1.1–1.6× faster on top, and cut their per-call overhead. So the gap between idiomatic and hand-rolled code didn't shrink to nothing as the interpreter got faster; in the comprehension case, it *grew*, because the optimization targeted the idiom directly.

**The "sum of lists" and string traps in the wild.** Both the `sum(list_of_lists, [])` flatten trap and the `s += piece` string trap are common enough that linters flag them. Tools like `ruff` (itself a Rust rewrite of Python tooling, a story for the [native acceleration track](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python)) and `flake8`-style checks warn on string concatenation in loops specifically because the quadratic blowup is a recurring production incident, not a theoretical concern. I've personally been paged for the string version twice.

**Generators and the standard library.** A large fraction of the standard library returns lazy iterators *by default* precisely so you stream in $O(1)$ memory: `range` (lazy since Python 3), `map`/`filter`/`zip` (lazy since Python 3), `dict.keys()`/`.values()`/`.items()` (views, not copies), `open()` (a line iterator), `os.scandir` (lazy, replacing the eager `os.listdir`), and all of `itertools`. The language's own design pushed toward streaming. When you write a generator, you're matching the grain of the standard library, and your code composes with it for free.

**Where the builtins stop and NumPy starts.** It's worth seeing the ceiling on these idioms, because it tells you when to climb to the next rung. Summing ten million floats with `sum` beats a hand-rolled loop, as we measured — but the floats are still boxed `PyObject`s, and `sum` still calls the add slot once per element in C. NumPy goes further: it holds the ten million floats in one contiguous, *unboxed* C buffer (8 bytes each, no per-object header) and sums them with a single C loop that the compiler auto-vectorizes to SIMD, adding several values per instruction with zero boxing. The numbers tell the story:

```pycon
>>> import timeit, numpy as np
>>> xs = list(range(10_000_000))
>>> arr = np.arange(10_000_000)
>>> timeit.timeit(lambda: sum(xs), number=10) / 10 * 1e3      # ms/call
52.0
>>> timeit.timeit(lambda: arr.sum(), number=10) / 10 * 1e3
4.1
```

`arr.sum()` is about **12× faster** than `sum(xs)` on ten million elements, and the array itself uses roughly 80 MB instead of the list's ~360 MB (boxed ints plus the list's pointer array). That's not a contradiction of this post — it's the same principle one rung higher: the builtin pushed the loop into C, and NumPy pushed the *data* into C too. The rule of thumb that falls out: for non-numeric, heterogeneous, or modest-sized work, these stdlib idioms are the right tool and there's no NumPy to reach for; for large, homogeneous, numeric arrays, the builtins are a stepping stone to vectorization, which is the next track in this series. Knowing where the idioms top out keeps you from polishing a Python `sum` that should be an `ndarray.sum`.

The honest framing for all of these: they are **constant-factor wins** (with the string and flatten cases being genuine big-O fixes). Per Amdahl's law, $S = 1/((1-p) + p/s)$, a 2× speedup on a section that's 10% of your runtime ($p = 0.1$, $s = 2$) buys you a whole-program speedup of about 1.05× — nearly nothing. The same 2× on the 80% hot path buys you about 1.67×. So these idioms matter *in the hot loop and almost nowhere else*. But two things make them worth a blanket habit anyway: first, you don't know where the hot loop is until you profile, and the idiomatic version costs nothing to write everywhere; second, the string and flatten cases aren't constant factors at all — they're latent quadratics that turn into incidents at scale, so the idiom is insurance, not just speed.

## 10. When to reach for this (and when not to)

These idioms are cheap and almost always worth it, but "almost always" hides real exceptions. Be honest about them:

- **Don't contort readable code for a 1.3× win outside the hot path.** If a section is 2% of your runtime, the manual `norms_fast` bindings buy you 2% × 1.35 = nothing you can measure. Write the clear version. The idioms earn their keep in the inner few percent, which you find by profiling, not guessing.
- **Don't use `map` with a lambda.** It pays a Python call per element and loses to the comprehension. `map` is only a win with a C callable (`str.upper`, `int`, `operator.itemgetter(...)`, `partial(int, base=16)`).
- **Don't reach for a generator when you need the list.** If you need random access, `len()`, multiple passes, or to pass the data to something that indexes it, materialize a list on purpose. A generator that gets immediately `list()`-ed everywhere is worse than a comprehension — you pay the laziness machinery for no benefit.
- **Don't over-nest a comprehension into unreadability.** A triple-nested comprehension with two `if` clauses is fast but unmaintainable. If it takes more than a second to read, write the explicit loop or a generator function with a name. Readability is a feature; these idioms are good *because* they're usually clearer, and a clever-but-opaque one-liner throws that away.
- **Don't micro-optimize when the real answer is up the ladder.** If you're summing ten million floats, `sum` beats the loop — but a NumPy array's `.sum()` beats `sum` by another 50× because it's one C loop over a packed, typed buffer with SIMD, no per-element boxing at all. These idioms are rung one of the ladder (do less work). When the data is numeric and large, the bigger win is rung two ([vectorize with NumPy](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast)). Don't polish a Python loop that should be an array op.
- **Don't forget the string trap is the exception that's never optional.** Every other item here is "nice to have in the hot path." `"".join` over `+=` is "always, everywhere," because it's an algorithmic correctness issue, not a micro-optimization. There's no input size at which `+=` in a loop is the right call.

And the flip side — when these absolutely *are* the right move:

- **In any per-element hot loop a profiler points you at.** This is the first thing to try after a profile, before vectorizing or going native, because it's a five-minute change with no new dependencies.
- **Whenever you'd build a list just to iterate it once.** Switch to a generator; it's free memory.
- **Whenever you see `+=` on a string in a loop, or `sum(lists, [])`.** Fix it on sight, regardless of profiling — those are bugs waiting for scale.
- **As a companion to caching.** When the loop is genuinely expensive per element, these idioms cut the *per-element* cost, and memoization cuts the *number* of elements you compute — they compose. The [caching and memoization post](/blog/software-development/python-performance/caching-and-memoization-lru-cache-and-beyond) covers the other half of "do less work."

## 11. How to measure these wins honestly

Every number in this post came from a measurement, and the measurement is the point — never trust a speedup you didn't time. A few rules specific to these micro-idioms, building on the full treatment in the [benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics):

First, **make the input big enough.** The per-call overhead (function call, list allocation) is fixed; the per-element win scales with the element count. On 10 elements, a comprehension and a loop are indistinguishable because the overhead dominates. Benchmark at the size your hot loop actually runs — thousands to millions of elements — or you'll measure noise and conclude "it doesn't matter."

Second, **use `timeit.repeat` and take the minimum (or median), not the mean.** The minimum is the run with the least interference from the OS scheduler, GC, and other processes — it's the closest to the "true" cost. The mean is contaminated by every hiccup. Use `repeat=7, number=...` and report `min(results)`.

Third, **watch for constant-folding and caching traps.** `timeit("sum(range(1000))")` may get partially optimized; pass the data in via `globals=` or `setup=` so the work isn't hoisted out as a constant. And if your "benchmark" calls a memoized function, you're timing a dict lookup, not the work.

Fourth, **for memory, use `tracemalloc` or `memray`, not `sys.getsizeof`.** `sys.getsizeof([...])` reports the list object's own size, *not* the size of the objects it points at — it'll tell you a million-element list is "8 MB" when the integers it holds are another 28 MB. For the generator-versus-list comparison, `tracemalloc.get_traced_memory()` (as in section 2) or `memray run` gives you the real peak. The [memory profiling post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) goes deep on this.

Fifth, **disassemble when you're unsure.** `dis.dis(f)` is the ground truth for "does this idiom actually compile to fewer/cheaper ops?" If two versions disassemble to nearly identical bytecode, they'll time nearly identically — no need to benchmark. The `dis` output told us the comprehension would win *before* we ran `timeit`; the benchmark just confirmed the magnitude.

```pycon
>>> import timeit, dis
>>> # the honest harness for any of these comparisons:
>>> data = list(range(100_000))
>>> t = lambda stmt: min(timeit.repeat(stmt, repeat=7, number=100, globals=globals())) / 100 * 1e3
>>> t("[x*x for x in data]")          # ms/call
3.1
>>> t("list(map(lambda x: x*x, data))")
5.4
>>> t("sum(x*x for x in data)")        # no list built at all
3.0
```

That last triple is the whole post in three lines: the comprehension beats `map`-with-a-lambda, and the bare generator into `sum` matches the comprehension's speed while building no intermediate list at all — fast *and* low-memory, which is the idiomatic Python sweet spot.

Sixth, **compare like with like, and run on the Python version you'll deploy.** Every absolute number here is 3.12-specific; on 3.10 the comprehension-vs-loop gap is larger (no PEP 659 specialization of the loop's opcodes) and on PyPy the whole comparison shifts because PyPy's tracing JIT can compile a hand-rolled loop down to nearly native speed, which narrows or erases several of these gaps. So benchmark on the interpreter you ship. And when you report a win, report the *setup* with it — input size, dtype, Python version, machine — because a "3× faster" with no context is folklore, and folklore is how the `map`-is-always-faster myth (false for lambdas) spread in the first place. The discipline that makes the rest of this series work applies to these micro-idioms too: don't guess, measure, and prove the win with a number you can reproduce.

## 12. Key takeaways

- **The readable one-liner is usually the fast one, and the bytecode proves why.** A list comprehension gets the specialized `LIST_APPEND` opcode and skips the per-item `LOAD_METHOD append` + `CALL` + `POP_TOP` that an explicit append loop pays — about 1.4–1.6× faster, visible in `dis`.
- **Generators give you $O(1)$ memory for one-pass work.** Swap `[...]` for `(...)` when you'll iterate once; a ten-million-element list cost 406 MB while the generator peaked at a kilobyte. `sum(x for x in data)`, not `sum([x for x in data])`.
- **Push the whole loop into a C builtin.** `sum`, `min`, `max`, `any`, `all`, `sorted`, `math.prod`, `str.join` loop in C with no per-element bytecode dispatch — 2–3× over a hand-rolled loop, and `any`/`all` short-circuit in C.
- **`map`/`filter` win only with a C callable.** `map(str.upper, xs)` beats the comprehension; `map(lambda ..., xs)` loses to it. Use `operator.itemgetter`/`attrgetter` and `functools.partial` to keep the callable in C.
- **`"".join` is $O(n)$; `s += piece` in a loop is $O(n^2)$.** This is the one non-negotiable — the quadratic blowup hit 117× more time for 10× more data in my benchmark. Same trap in `sum(lists, [])`; use `itertools.chain.from_iterable`.
- **`itertools`/`functools` are the C-speed iterator algebra.** `chain`, `islice`, `groupby`, `accumulate`, `product`, `reduce`, `partial` compose into lazy pipelines that run in C and stream in bounded memory — a real 3× speedup and 7.5× memory cut on the dedup pipeline.
- **Inside an unavoidable loop, bind locals and hoist invariants.** Turn `LOAD_GLOBAL` into `LOAD_FAST` and move loop-invariant work out — modest since PEP 659, but free. Hoisting a recomputed lookup into a set/dict can collapse a quadratic into a linear pass.
- **These are constant factors, so they matter in the hot path and compound.** Per Amdahl, a 2× win on 2% of runtime is nothing; the same win on the hot 80% is real. Write the idioms everywhere because they cost nothing and read better, and profile to know where they pay.

## 13. Further reading

- [The `dis` module — Disassembler for Python bytecode](https://docs.python.org/3/library/dis.html) — read your own loops; the ground truth for which idiom compiles to fewer ops.
- [The `itertools` module](https://docs.python.org/3/library/itertools.html) and [`functools`](https://docs.python.org/3/library/functools.html) — the full C-speed iterator and higher-order-function algebra, with the recipes section.
- [The `timeit` module](https://docs.python.org/3/library/timeit.html) and [`tracemalloc`](https://docs.python.org/3/library/tracemalloc.html) — how to measure the time and memory wins honestly.
- [PEP 709 — Inlined comprehensions](https://peps.python.org/pep-0709/) and [PEP 659 — Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/) — why 3.11/3.12 made these idioms even faster.
- [The Python Programming FAQ](https://docs.python.org/3/faq/programming.html) — the official "use `str.join`, not `+=`" guidance and the string-concatenation complexity note.
- "High Performance Python" by Micha Gorelick & Ian Ozsvald (O'Reilly, 2nd ed.) — the chapters on lists, generators, and iterators cover this ground with more benchmarks.
- Within this series: the intro on [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) for the eval-loop machinery these idioms exploit, [the hidden cost of objects and dynamic dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) for the local-binding and hoisting cost model, and the sibling [caching and memoization](/blog/software-development/python-performance/caching-and-memoization-lru-cache-and-beyond) for the other half of doing less work. When the data is numeric and large, the next rung up is NumPy and vectorization.

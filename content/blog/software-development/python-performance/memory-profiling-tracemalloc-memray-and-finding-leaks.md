---
title: "Memory Profiling in Python: tracemalloc, memray, and Finding Leaks"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Find out exactly why your Python process is using 24 GB when it should use 2 — why getsizeof lies, how tracemalloc and memray pinpoint the allocating line, and how to tell a leak from a peak and fix it."
tags:
  [
    "python",
    "performance",
    "memory-profiling",
    "tracemalloc",
    "memray",
    "objgraph",
    "memory-leaks",
    "profiling",
    "optimization",
    "rss",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-1.png"
---

A worker process that should comfortably live in 2 GB has crept up to 24 GB. You found out the way everyone finds out: the orchestrator's OOM killer reaped it at 3 a.m., the on-call page went off, and the only clue in the logs is `Killed` with no traceback. You restart it. It runs fine for forty minutes, serves a few thousand requests, and then the resident memory begins to climb again — slowly, relentlessly, a few megabytes per request, never falling back. By lunchtime it is at 9 GB and you know exactly how the night will end.

This is the memory equivalent of the slow `for` loop from [the series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means): a small, fixable problem that is invisible until you measure it correctly, and almost impossible to fix until you can see *where the bytes are going*. The frustrating part is that your first instincts are usually wrong. You sprinkle a few `sys.getsizeof` calls around the suspect data structures, they all report a few kilobytes, and you conclude the leak must be "somewhere in a C extension." It is not. `getsizeof` lied to you, and it lied in a specific, predictable way that we are going to dissect byte by byte.

This post is about profiling **memory**, not time. The time-profiling tools from the previous posts — [`cProfile` for the hot path and `line_profiler` and `py-spy` for per-line and production profiling](/blog/software-development/python-performance/line-and-statistical-profiling-line-profiler-and-py-spy) — tell you nothing about where the bytes live. For that you need a different toolbox: `tracemalloc` in the standard library to find the *line* that allocates and to **diff** memory over time, `memray` for allocation flame graphs and the high-water mark and native allocations, and `objgraph` to answer the question that actually fixes a leak — *what is holding this object alive?* By the end you will be able to take a 24 GB process and produce a one-line answer: "it is the response cache on line 88, it is unbounded, here is the before-and-after RSS once we cap it."

![the life of an object's memory from allocate to live to freed or trapped in a reference cycle and leaked on an eight core Linux box](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-1.png)

The figure above is the mental model for everything that follows. An allocation is born on the heap; it stays resident as long as *something* references it; the moment its last reference drops, CPython's reference counting frees it immediately. That is the happy path. The unhappy path is the lower-right branch: an object trapped in a reference cycle, or pinned by a global registry, or stuffed into an unbounded cache. It has logically "leaked" — your code will never touch it again — but its reference count never reaches zero, so it sits in RSS until the cyclic garbage collector happens to sweep it, or forever if a real reference still points at it. Finding leaks is, almost entirely, the discipline of finding which of those references is the one that should have been dropped.

All numbers in this post come from a consistent, named setup so you can calibrate against your own: an **8-core x86-64 Linux box (or an Apple M2), CPython 3.12, 16 GB of RAM**, measuring RSS with `psutil` and `/proc/self/status`. CPython object sizes are the 64-bit CPython 3.12 layout. When I quote a figure I ran, it is from that box; when I quote a range, I will say so.

## Why sys.getsizeof lies, byte by byte

Start with the tool everyone reaches for first, because understanding precisely why it fails teaches you most of what you need to know about Python memory.

`sys.getsizeof(obj)` returns the size of `obj` **and nothing it points at**. It is shallow by design. For a flat value with no references — a small `int`, a `float`, a `bytes` object — it tells you the truth. For anything that contains references to other objects — a `list`, a `dict`, a custom class instance, a `tuple` of objects — it tells you the size of the *container's own bookkeeping*, which is almost always a tiny fraction of the real footprint. The container holds **pointers**, 8 bytes each on a 64-bit build, and `getsizeof` counts the pointers. It does not follow them.

Let us count the bytes by hand, because the gap is the whole point. Consider a list of one hundred thousand small dictionaries, each shaped like a parsed record:

```python
import sys

records = [
    {"id": i, "name": f"user{i}", "score": i * 1.5, "active": True}
    for i in range(100_000)
]
print(sys.getsizeof(records))  # what does this report?
```

What does `getsizeof(records)` report? The list object itself is a header plus a contiguous array of `PyObject*` pointers. On CPython 3.12 the list header is 56 bytes, and the backing array holds 100,000 pointers at 8 bytes each, plus some over-allocation slack the list keeps for cheap `append`. So `getsizeof(records)` returns roughly $56 + 100{,}000 \times 8 \approx 800{,}056$ bytes — call it **800 KB**. That is the entire answer `getsizeof` will give you. It counts the array of 100,000 pointers and stops.

Now count what those pointers actually point at. Each element is a `dict` with four keys. A small CPython 3.12 dict with four entries occupies about 184 bytes for the combined dict object plus its key/value table (the split-table layout shares keys when many instances share a class, but a plain dict literal like this uses a combined table). Each dict references its keys and values: the four key strings (`"id"`, `"name"`, `"score"`, `"active"`) are interned once and shared across all dicts, so they cost almost nothing per record — but the **values** are not shared. The `"name"` value is a fresh string like `"user12345"`, roughly 58 bytes for a 9-character ASCII `str` (a 49-byte object header for compact ASCII strings plus the characters). The `"score"` value is a `float`, 24 bytes, freshly allocated per record because `12345 * 1.5` is a distinct object. The `"id"` value is an `int`; for `i >= 257` it is a freshly allocated `int` of 28 bytes (CPython caches small ints from −5 to 256). The `"active"` value is `True`, a shared singleton, free.

So the real per-record cost is approximately: dict 184 + name-string 58 + float 24 + int 28 ≈ **294 bytes**, and the list pays an extra 8-byte pointer per record. Times 100,000:

$$
\text{true size} \approx 100{,}000 \times (294 + 8) \approx 30.2 \text{ MB}
$$

`getsizeof` told you **0.8 MB**. The truth is about **30 MB** — roughly **35× larger**. (Scale this to the figure's "list of 10,000 dicts" and you get ~85 KB reported versus ~3 MB true; same ratio, smaller numbers.) If you were chasing a 24 GB process by adding up `getsizeof` numbers, you would have accounted for under a gigabyte and concluded the rest was "native." It was never native. It was the values your shallow count refused to follow.

![sys getsizeof reporting a shallow eighty five kilobytes for a list of ten thousand dicts versus the true recursive three megabyte footprint](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-2.png)

There is a second, subtler reason the shallow number is dangerous: **sharing**. If you naively sum `getsizeof` over every object reachable from `records`, you will *over*-count, because the interned key strings and the `True`/`None` singletons are referenced thousands of times but exist once. A correct deep-size walk has to deduplicate by object identity. So `getsizeof` undercounts (it ignores referents) and a naive recursive sum overcounts (it ignores sharing). The right tool does neither — it walks the reference graph and records each distinct object exactly once.

### A correct deep-size helper

Here is a deep-size function that follows references via `gc.get_referents` and deduplicates by `id`. It is the honest version of `getsizeof`:

```python
import sys
import gc
from collections.abc import Mapping, Container

def deep_size(obj, _seen=None):
    """Recursively size obj and everything it uniquely references."""
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:           # already counted -> skip (handles sharing + cycles)
        return 0
    _seen.add(obj_id)
    size = sys.getsizeof(obj)
    # gc.get_referents returns the objects obj directly points at.
    for referent in gc.get_referents(obj):
        size += deep_size(referent, _seen)
    return size
```

Run it on our list:

```pycon
>>> sys.getsizeof(records)
800056
>>> deep_size(records)
30180448
```

That is the contrast in one screen: **0.8 MB shallow, 30 MB deep**, a 37× difference on this exact run. The `_seen` set is doing the load-bearing work — it makes the walk count each interned key string and each singleton once, and it also stops the recursion dead if it ever hits a cycle (an object that, directly or transitively, references itself). `gc.get_referents` is the key primitive: it asks the object, via its type's traverse slot, "which objects do you point at?" — the same mechanism the cyclic garbage collector uses. It is not perfect (it cannot see memory held inside opaque C extensions that do not implement the GC protocol), but for pure-Python structures it is exactly right.

The practical lesson: **never use `getsizeof` to size a container.** Use it to size a single flat object — an `int`, a `bytes` blob, one string — where shallow *is* the answer. The moment there are referents, reach for a deep walk, or better, for `tracemalloc` and `memray`, which measure what the allocator actually handed out and never have to guess.

### Why every object costs what it does

To predict memory the way the cost model in the [series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) predicts time, you need the per-object byte budget in your head. Every Python object on a 64-bit CPython build carries a fixed header: a reference count (8 bytes) and a type pointer (8 bytes), 16 bytes before it stores a single byte of your data. That header is the price of "everything is an object." Build on it and the numbers fall out:

- A small **`int`** is 28 bytes (the 16-byte header plus the variable-length digit storage CPython uses for arbitrary-precision integers, which is why even the number `5` costs 28 bytes, not 8). CPython interns the integers from −5 to 256, so those are shared singletons; everything outside that range is a fresh 28-byte allocation. This is why summing a million large integers allocates far more than the 8 MB a C `int64[]` would.
- A **`float`** is 24 bytes — the header plus a C `double`. There is no float interning, so every distinct `float` result is a fresh allocation. A list of a million floats is ~24 MB of float objects *plus* 8 MB of pointers in the list — versus a NumPy `float64` array, which is ~8 MB of packed `double`s with no per-element object at all. That 4× gap is the entire argument for the array world.
- A short **`str`** is roughly 49 bytes plus one byte per ASCII character (CPython's compact-string layout stores pure-ASCII strings most efficiently; non-ASCII jumps to 2 or 4 bytes per character). String literals in your source and identifiers are interned and shared; runtime-built strings like `f"user{i}"` are not, which is exactly why the `"name"` values dominated our record's footprint.
- An empty **`dict`** is 64 bytes; it grows in steps as you add keys, and a 5-key dict lands around 184–232 bytes depending on the layout. A **`list`** is 56 bytes plus 8 per slot, with over-allocation slack for cheap `append`. A **`tuple`** is 40 bytes plus 8 per slot and has no slack (it is immutable), so it is the more compact choice for fixed-length records.

Two facts follow that matter enormously for memory and that you will exploit in the [object-cost post](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch). First, a plain class instance carries a per-instance **instance dict** to hold its attributes — that dict is 100+ bytes on top of the object itself, so a million small objects pay a million instance dicts. Declaring `__slots__` removes the per-instance dict and stores attributes in a fixed array, cutting a small object's footprint by 40–50%. Second, when many objects of the same class share the same attribute names, CPython's *key-sharing* dict layout lets their instance dicts share one key table — but the moment you add an attribute to one instance dynamically, that instance splits off its own full table and the saving evaporates. These are not micro-details; on a few-million-object dataset they are the difference between 2 GB and 6 GB, and a `tracemalloc` diff that shows millions of `dict` allocations attributed to your model's `__init__` is the fingerprint of exactly this cost.

With the per-object budget in hand, the 30 MB deep-size number stops being a surprise and becomes a calculation you could have done in your head: 100,000 records × (one dict + one fresh name string + one fresh float + one fresh int + one list pointer) ≈ 100,000 × 302 bytes ≈ 30 MB. The deep walk just confirmed the arithmetic. This is the goal of memory profiling — not to be surprised by the number, but to be able to predict it, then measure it, then close the gap.

## RSS, the Python heap, and why freeing does not shrink it

Before any tool can help, you need to know what number you are even looking at. When the OOM killer reaps your process, it is looking at **RSS — resident set size**, the amount of physical RAM the process currently occupies. That is the number in `htop`'s `RES` column, in `ps aux`, and in `/proc/<pid>/status` as `VmRSS`. It is the number that matters for "am I about to get killed." And it is composed of several things, only one of which is your live Python objects.

![the composition of resident set size stacking interpreter and code then the python object heap then large native buffers then fragmentation overhead](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-5.png)

RSS stacks up, from the bottom: the **interpreter and your code** (the CPython binary, loaded `.so` extension modules, the bytecode of every imported module — easily 30–80 MB before your program does anything, more with heavy frameworks). On top of that, the **Python object heap** — every live `int`, `str`, `list`, `dict`, and instance, managed by CPython's small-object allocator, **pymalloc**. On top of that, **large native buffers** — a NumPy array, a `bytes` blob, a decompressed image — which bypass pymalloc and go straight to the system allocator because they are too big for pymalloc's size classes. And threaded through all of it, **fragmentation overhead**: memory the process has claimed from the OS but is only partially using.

The single most surprising fact about Python memory — the one that sends people on wild goose chases — is this: **freeing objects often does not reduce RSS.** You `del` a million-element list, you call `gc.collect()`, you watch RSS in `htop`, and it does not budge. Your code is correct; the memory really is logically free; and RSS stays high anyway. Here is why.

pymalloc requests memory from the OS in 256 KB chunks called **arenas**, carves each arena into 4 KB **pools**, and carves each pool into fixed-size **blocks** (one size class per pool: 16-byte blocks, 32-byte blocks, and so on up to 512 bytes). When you allocate a small object, pymalloc hands you a block from a pool of the right size class. When you free the object, the block returns to its pool's free list — *but the arena stays mapped*. CPython only returns an entire 256 KB arena to the OS when **every** pool in it is completely empty. If even one small object anywhere in that arena is still live, the whole 256 KB stays resident. This is **fragmentation**: you can free 99% of an arena and reclaim none of it, because the surviving 1% is scattered across every pool.

So picture allocating ten million small objects, then freeing nine and a half million of them in a scattered pattern. The half-million survivors are sprinkled across nearly every arena. RSS barely drops, even though `tracemalloc` will correctly tell you that live Python memory fell by 95%. RSS and live-object memory are *different numbers*, and the gap between them is fragmentation plus the allocator's reluctance to give memory back. (Large objects — anything pymalloc punts to the system `malloc` — *are* returned to the OS immediately when freed, which is why a freed 2 GB NumPy array usually does drop RSS, while freeing 2 GB of tiny dicts usually does not.)

This has a direct, practical consequence: **when you measure a leak, watch RSS, but diagnose it with `tracemalloc` or `memray`.** RSS tells you the symptom (memory is high and climbing). The allocation profilers tell you the cause (these lines allocated these bytes). And it explains a common false alarm — a batch job whose RSS climbs during a big phase and never falls afterward is not necessarily leaking; it may simply be holding fragmented arenas. The way to know for sure is the topic of the rest of this post.

#### Worked example: freeing does not shrink RSS

On the M2 box, CPython 3.12, I built a list of 10 million three-tuples, measured RSS, deleted the list, forced a collection, and measured again:

```python
import os, gc
import psutil  # pip install psutil

proc = psutil.Process(os.getpid())
def rss_mb():
    return proc.memory_info().rss / 1e6

print(f"start: {rss_mb():.0f} MB")
data = [(i, i * 2, i * 3) for i in range(10_000_000)]
print(f"after build: {rss_mb():.0f} MB")
del data
gc.collect()
print(f"after del + gc: {rss_mb():.0f} MB")
```

Typical output on that box:

```bash
start: 38 MB
after build: 1163 MB
after del + gc: 196 MB
```

Three numbers tell the story. The baseline interpreter is **38 MB**. Building 10 million tuples — each tuple ~72 bytes plus three boxed ints sharing the small-int cache only below 257, so most ints are freshly allocated 28-byte objects — pushed RSS to **1163 MB**. After `del` and `gc.collect()`, RSS fell to **196 MB**, not back to 38. The interpreter held ~158 MB of arenas it could not fully empty (the ints below 257 and various survivors kept pools alive) — although tuples themselves are large-ish and mostly returned. Your exact residual depends on allocation pattern, but the lesson is universal: **the curve does not return to baseline, and that is normal, not a leak.** A leak is when the *peak* climbs run after run, which is a different shape entirely — and the shape is exactly what figure 7 contrasts.

## tracemalloc: find the line that allocates

`tracemalloc` ships in the standard library and is the first tool to reach for when you want to know *which source line* is responsible for memory. It works by hooking CPython's allocator: while tracing is on, every allocation records the Python traceback at the point it happened. You then take a **snapshot** — a frozen record of every live allocation grouped by where it was made — and you can either inspect one snapshot's top consumers or, far more powerfully, **diff two snapshots** to see what grew between them.

The basic shape is four calls: `start`, `take_snapshot`, `statistics`, and the diff via `compare_to`.

```python
import tracemalloc

tracemalloc.start()                 # begin recording allocations + tracebacks

# ... run the code you suspect ...
build_some_data()

snapshot = tracemalloc.take_snapshot()
top = snapshot.statistics("lineno")  # group live allocations by source line
for stat in top[:10]:
    print(stat)
```

Each line of output looks like `myapp/loader.py:88: size=240 MB, count=1003, average=245 KB` — the file and line that allocated, the total live bytes attributed to it, the number of live allocations, and the average size. That is the entire game: it points a finger at the exact line. `statistics("lineno")` groups by line; `statistics("traceback")` groups by the full call stack (use `tracemalloc.start(25)` to keep 25 frames) so you can see *which call path* into a shared helper is the expensive one.

But the killer feature is the diff. A single snapshot's "top consumers" includes a lot of legitimate, stable memory — your loaded modules, your config, the framework. What you care about for a leak is **what changed**. So you snapshot a clean baseline, run the suspect workload, snapshot again, and diff.

![the tracemalloc workflow from start to snapshot A to running the workload to snapshot B to compare to to the top growers](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-4.png)

The figure is the whole workflow. The code for it is just as short:

```python
import tracemalloc

tracemalloc.start(25)               # keep 25 frames of traceback per alloc

snapshot_a = tracemalloc.take_snapshot()   # clean baseline
for _ in range(1000):
    handle_one_request()                   # the suspect workload
snapshot_b = tracemalloc.take_snapshot()   # after the run

diff = snapshot_b.compare_to(snapshot_a, "lineno")
print("Top allocation growth between A and B:")
for stat in diff[:10]:
    print(stat)
```

`compare_to` returns the lines sorted by **size difference** — the lines whose live memory grew the most between the two snapshots. A line that allocated and freed nets to zero and disappears. A line that allocated and *kept* shows up at the top with a big positive delta. That positive delta is your leak, almost by definition: memory that the workload added and never gave back.

### Reading a real tracemalloc diff

Here is a representative diff from chasing the 24 GB worker, formatted as a table. The process served 1,000 requests between snapshot A and snapshot B, and I sorted `compare_to(..., "lineno")` by growth:

| Source line | Size delta | Count delta | Average | Verdict |
| --- | ---: | ---: | ---: | --- |
| `app/cache.py:88` | +240.4 MB | +1000 | 246 KB | leak: one entry per request, never evicted |
| `app/parse.py:142` | +1.9 MB | +3 | 642 KB | benign: lazily-built lookup table |
| `app/models.py:51` | +0.4 MB | +1002 | 410 B | benign: transient, count tracks requests but size flat |
| `json/decoder.py:353` | +0.1 MB | +44 | 2.3 KB | benign: stdlib churn |
| `app/log.py:19` | −0.2 MB | −60 | — | freed: log buffer rotated |

Read it like a profiler reads a hot path: the top line dwarfs everything. `app/cache.py:88` grew by **240 MB across exactly 1,000 requests** — that is **246 KB per request, one new live allocation per request, none freed**. The count delta of +1000 matching the request count is the smoking gun: this line creates one object per request and keeps every one. The other lines are noise: `parse.py:142` grew once and stabilized (a lazily-built table), `models.py:51` allocated 1,002 times but the *size* stayed flat because those objects are transient and freed (the count delta is churn, not growth). The negative line actually freed memory.

You did not have to read the code. The diff told you: line 88 of `cache.py`. When you open it, you find a module-level `dict` used as a response cache, written to on every request, with no eviction. That is the leak. We will fix it and measure the before-and-after later in the post — but notice how fast `tracemalloc` collapsed "the process uses 24 GB and I don't know why" into "line 88 grows 246 KB per request."

#### Worked example: leak found and proven in one diff

On the Linux box I reproduced the unbounded-cache leak with a 50 KB payload per request and ran the diff above for 1,000 iterations. The result: `cache.py:88` showed `size=49.9 MB (+49.9 MB), count=1000 (+1000)`, while every other line stayed under 2 MB of delta. Extrapolating linearly — 49.9 MB per 1,000 requests is **~51 KB per request** — a worker handling 200 requests/minute leaks about **600 MB/hour**, which puts a 2 GB budget on a collision course with the OOM killer in roughly three hours. That matches the production symptom (crashes overnight, fine for the first forty minutes) precisely. The diff did not just find the leak; it predicted the time-to-crash, and the prediction was right to within an hour.

A few practical notes on `tracemalloc` that save real time:

- **Overhead is real.** Tracing roughly doubles to triples allocation cost and adds memory of its own (the per-allocation tracebacks). It is fine for a reproduction or a staging run, not for always-on production. Turn it on around the suspect phase and off after, or gate it behind an env var.
- **Frame depth matters.** `tracemalloc.start(1)` keeps one frame, which groups everything by the immediate allocating line — often you want `start(25)` so `compare_to(..., "traceback")` shows *which caller* drove a shared allocator. More frames cost more memory.
- **Filter the noise.** `snapshot.filter_traces((tracemalloc.Filter(False, "<frozen importlib._bootstrap>"), ...))` drops import machinery and tracemalloc's own frames so the diff is all signal.
- **It only sees Python allocations.** Memory allocated *inside* a C extension that calls `malloc` directly — a lot of NumPy's buffer, parts of `lxml`, database driver internals — is invisible to `tracemalloc`. When the leak is native, `tracemalloc` will show a small, stable Python footprint while RSS balloons, and that mismatch is itself the diagnosis: *the leak is in native code, switch to `memray`.*

### Periodic snapshots: catching a leak in a live worker

The single-diff pattern (snapshot A, run, snapshot B) is perfect for a reproduction. For a long-running worker where you cannot pause to compare two points, the better pattern is **periodic snapshots against a fixed baseline** — take a snapshot at startup, then every N requests compare the current snapshot to that baseline and log the top growth. Wired to a background thread or an admin endpoint, this turns a leak into a line in your logs instead of a 3 a.m. page:

```python
import tracemalloc
import threading
import time

tracemalloc.start(25)
_BASELINE = tracemalloc.take_snapshot()

def memory_watchdog(interval_s=300):
    """Every interval, log the top 5 lines that grew since startup."""
    while True:
        time.sleep(interval_s)
        current = tracemalloc.take_snapshot()
        growth = current.compare_to(_BASELINE, "lineno")
        top = growth[:5]
        total_grew = sum(s.size_diff for s in top) / 1e6
        print(f"[mem] top-5 growth since start: {total_grew:.1f} MB")
        for stat in top:
            print(f"[mem]   {stat}")

threading.Thread(target=memory_watchdog, daemon=True).start()
```

This watchdog costs the ongoing tracing overhead (the 2–3× on allocations), so you would gate it behind a `PROFILE_MEMORY` env var and enable it only when you suspect a leak — but when enabled, it answers the leak question without any reproduction at all. The first interval where one line's `size_diff` keeps climbing across successive logs is your leak, and the `stat` already names the file and line. I have caught production leaks this way in under an hour of running, where reproducing the traffic locally would have taken a day.

A subtle but important point about `compare_to`'s grouping. `"lineno"` groups by the final allocating line, which is right when the leak is one specific line. But many leaks flow through a *shared* helper — a serializer, a `copy.deepcopy`, a `dict` constructor — that is called from dozens of places, only one of which leaks. Grouping by `"lineno"` blames the shared helper and hides the caller. Switch to `"traceback"` and the growth is attributed to the full call stack, so you see *which path into the shared helper* is the leaking one:

```python
growth = current.compare_to(_BASELINE, "traceback")
top = growth[0]
print(f"{top.size_diff / 1e6:.1f} MB grew along:")
for frame in top.traceback.format():   # the full call chain, top to bottom
    print(frame)
```

That distinction — line versus traceback — is the difference between "the leak is in `json.dumps`" (useless; you call it everywhere) and "the leak is in `cache_response → serialize → json.dumps`" (actionable; it is the cache path). When a `"lineno"` diff points at a stdlib or library line, re-run with `"traceback"` to find *your* code in the chain.

## memray: allocation flame graphs, the high-water mark, and native allocations

`memray` is the heavy artillery, and it fixes `tracemalloc`'s biggest blind spot: it sees **native allocations**. It is a `pip install memray` away (Linux and macOS; it hooks the allocator at the C level), and it runs your program from the outside, so you do not even edit your code.

```bash
pip install memray
memray run -o output.bin myscript.py        # record every allocation to output.bin
memray flamegraph output.bin                 # -> output.html, an interactive flame graph
```

`memray run` launches your script and records every allocation — Python and C — with its full stack, to a binary file. Then `memray flamegraph` turns that into an interactive HTML **allocation flame graph**: the width of each frame is the bytes allocated *along that call path*, so the widest boxes are the biggest memory consumers, and you read it top-down to follow the call chain to the allocating function. It is the visual analog of a CPU flame graph, but the axis is bytes, not time. By default the flame graph shows the **high-water mark** — the single moment of peak memory during the run — which is exactly what you want when chasing an OOM, because OOM happens at the peak, not the average.

Two flags do most of the leak-hunting work. The high-water-mark view answers "what was resident at peak"; the **temporal / leaks** view answers "what was allocated and never freed":

```bash
# Show what was allocated and NEVER deallocated by the end of the run:
memray flamegraph --leaks output.bin

# Live, real-time terminal UI while the program runs (great for a long worker):
memray run --live myscript.py
```

`--leaks` builds the flame graph from allocations that were still live when the program exited — pure leak signal, with the framework's steady-state memory subtracted out. The `--live` mode opens a terminal UI (a TUI) that updates as the program runs, showing the current top allocators and total memory live; you watch the numbers climb in real time and see *which function's bar grows*, which is the fastest way to catch a slow creep without waiting for a crash.

There is also a `--native` flag (`memray run --native`) that captures C/C++ stack frames in addition to Python ones, so a leak inside a NumPy or `lxml` or database-driver allocation shows the native function names, not just the Python line that called into the library. That is the capability `tracemalloc` simply does not have.

### Reading an allocation flame graph

The flame graph is the most information-dense artifact in memory profiling, and reading it is a learnable skill. Each box is a function; the **width** of the box is the total bytes allocated by that function and everything it called (along that call path); boxes stack vertically by call depth, with callers below and callees above (or above and below, depending on orientation — `memray` puts the root at the bottom). You read it the way you read a CPU flame graph from [the profiling post](/blog/software-development/python-performance/line-and-statistical-profiling-line-profiler-and-py-spy), but every width is a number of bytes, not a slice of time.

The reading algorithm is simple and mechanical:

1. **Find the widest box at the top.** A wide box near the top of the stack is a function that allocated a lot of memory *directly*. That is your hot allocator.
2. **Trace down from it to the root.** The chain of boxes beneath it is the call path that reached it — that is *who* asked for the memory. The leak is usually not the allocator (which might be a stdlib `dict` or a NumPy buffer) but somewhere in that chain, in your code.
3. **Compare the high-water view to the `--leaks` view.** A box that is wide in the high-water mark but *disappears* in `--leaks` is a transient — it allocated and freed. A box that is wide in **both** is a leak — it allocated and kept. That single comparison separates a peak from a leak visually, in seconds.

So a flame graph where `pandas.concat` is the widest box in the high-water view but gone in `--leaks` is the transient-copy peak from the worked example — big at the moment, freed right after. A flame graph where your `cache_response` function is wide in *both* views is the real leak. The visual contrast does in one glance what a table of numbers does in a paragraph, which is why the flame graph is the artifact you paste into the incident channel.

### Profiling a region from inside your code

You do not always want to profile an entire script — sometimes you want to wrap one suspect block. `memray` exposes a `Tracker` context manager for exactly that: it records allocations only inside the `with` block, to a file you then turn into a flame graph:

```python
import memray

with memray.Tracker("suspect_region.bin"):
    # only allocations in here are recorded
    result = build_the_big_report(rows)

# then on the shell:  memray flamegraph suspect_region.bin
```

This is the surgical version — useful when the script does a lot of legitimate setup you do not want cluttering the graph, or when you want to profile one phase of a pipeline (the parse, then the transform, then the aggregate) separately to see which phase owns the peak. Combined with `memray stats suspect_region.bin` (a quick textual summary of total allocations, peak memory, and the most common allocators), it gives you a fast, scriptable loop: wrap a region, run, read the stats, and only open the full flame graph if the stats do not already answer the question.

### When memray earns its keep over tracemalloc

The two tools overlap, but the decision is clean. The matrix below — and figure 3 — lay out which tool answers which question.

![a matrix comparing getsizeof tracemalloc memray and objgraph by what they measure their overhead and their best output](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-3.png)

| Tool | Sees native allocs? | Overhead | Best for |
| --- | --- | --- | --- |
| `sys.getsizeof` | n/a | none | sizing one *flat* object; useless on containers |
| `tracemalloc` | no (Python only) | 2–3× allocation cost | the Python line that allocates; snapshot diffs over time |
| `memray` | **yes** (with `--native`) | low in `run` mode | flame graphs, high-water peak, `--leaks`, native libs, live TUI |
| `objgraph` | n/a | manual, per-object | *what holds an object alive* — the retaining reference chain |

Reach for `tracemalloc` when you are inside a long-running service and want to take periodic snapshots and diff them *in process* (it is stdlib, no extra dependency, and you can wire it to an admin endpoint). Reach for `memray` when you want a picture — the flame graph is the single best artifact for explaining a memory problem to a teammate — or when the leak might be native, or when you need the high-water mark for an OOM, or when you want to watch a creep live. In practice many teams use both: `memray` to localize fast and visually, `tracemalloc` for the always-available in-process diff.

#### Worked example: a transient peak hiding behind a modest average

A reporting job on the Linux box averaged 1.4 GB RSS but was OOM-killed on the 32 GB node about one run in five — wildly inconsistent, which is the signature of a **transient peak**, not a steady leak. `memray run` plus the default high-water-mark flame graph found it in one shot: a single line, `report.py:212`, did `df = pd.concat([big_df, big_df.copy()])` inside a loop, briefly holding **three copies** of a 9 GB DataFrame — the original, the `.copy()`, and the concatenated result — for **~27 GB at the peak**, then immediately freeing two of them. The *average* RSS never showed it because the peak lasted under a second per iteration. Average-based monitoring will never catch this; the high-water-mark flame graph caught it instantly. The fix was to concatenate lazily and avoid the `.copy()`, dropping the peak to ~10 GB and ending the random OOMs. This is the canonical "huge transient peak" cell from the symptom matrix at the end of the post — and the reason you always look at the *peak*, not the mean, when chasing an OOM.

## objgraph: what is actually holding this alive?

`tracemalloc` and `memray` tell you *where* memory was allocated. Neither tells you *why it is still alive* — which reference is pinning it. For that you need `objgraph`, and answering that question is what actually lets you fix the leak, because a leak is never "this line allocates"; it is "this line allocates *and this other reference refuses to let go*."

`objgraph` (`pip install objgraph`) walks the object graph that the garbage collector tracks. Its three most useful functions:

```python
import objgraph

# 1) What types are most common in memory? (run twice, compare, to see growth)
objgraph.show_most_common_types(limit=15)

# 2) How many of these objects exist, and is it growing between calls?
objgraph.show_growth(limit=10)

# 3) The money shot: what chain of references keeps THIS object alive?
import random
leaked = random.choice(objgraph.by_type("ParsedRecord"))
objgraph.show_backrefs([leaked], max_depth=7, filename="backrefs.png")
```

`show_growth()` is the leak-hunter's heartbeat: call it, do some work, call it again, and it prints only the types whose instance *count increased* — `ParsedRecord  +1000`, `dict  +1000` — which immediately tells you *what kind of thing* is leaking even before you know why. `show_most_common_types()` is the snapshot version. But `show_backrefs` is the one that closes the case: hand it a leaked object and it traces every chain of references *pointing at it*, back toward the GC roots, and renders a graph (a PNG via Graphviz). You read it from your object upward and find the unexpected owner — the module global, the closure, the class-level list, the event-handler registry — that should have dropped the reference and did not.

![a retention tree showing a module level registry holding a cache dict holding a closure that pins a three megabyte array](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-6.png)

The figure shows the shape you find over and over. The leaked 3 MB array at the bottom is pinned by a closure (an inner function that captured it), the closure is held in a per-request cache `dict`, and the cache `dict` is held by a **module-level global registry** — the GC root that never goes out of scope. Nothing in this chain is a bug in isolation. The bug is that the chain *exists and grows*: every request adds a node to the cache, the cache is never trimmed, and because the module global lives for the entire process lifetime, every entry is immortal. `show_backrefs` draws exactly this chain and makes the fix obvious: bound the cache, or hold the values in a `weakref`, or scope the registry to the request instead of the module.

### The reference cycle case, and why it is special

There is a second class of leak that `objgraph` is built for: the **reference cycle**. CPython frees objects by reference counting — every object carries a count of how many references point at it, and when that count hits zero the object is freed immediately. This is wonderfully prompt, but it has one fatal blind spot: a **cycle**. If object A references B and B references A, then even after *your* code drops all its references to both, A's count is still 1 (B points at it) and B's count is still 1 (A points at it). Neither reaches zero. Reference counting alone can **never** free a cycle.

```python
class Node:
    def __init__(self):
        self.peer = None

a = Node()
b = Node()
a.peer = b          # a -> b
b.peer = a          # b -> a, now a cycle
del a, b            # your references are gone, but the cycle keeps both alive
```

After `del a, b`, the two `Node` objects are unreachable from your code yet still alive in memory, each propping up the other. This is why CPython has a *second* memory manager on top of reference counting: the **generational cyclic garbage collector** in the `gc` module. Periodically (tuned by allocation thresholds), it finds groups of objects that reference only each other and are unreachable from outside, and frees them. So the cycle above *does* get collected — eventually, when `gc` runs. The leak is that until then, the memory is resident, and if you have *disabled* the cyclic GC for throughput (a common trick, `gc.disable()`), or if the cycle includes an object with a `__del__` method in older Pythons, the memory can stay forever.

You detect cycles with `gc`:

```python
import gc

gc.collect()                          # clear what can be cleared
gc.set_debug(gc.DEBUG_SAVEALL)        # keep uncollectable objects in gc.garbage
# ... run workload ...
gc.collect()
print(f"objects in gc.garbage: {len(gc.garbage)}")   # cyclic/uncollectable survivors
```

and you find what is *in* a cycle with `objgraph.show_backrefs` plus `gc.get_referrers`. The practical fix for cycles is almost always one of: break the cycle explicitly (`a.peer = None` before dropping), use `weakref` for the back-pointer (a `weakref` does not increment the reference count, so the cycle is never formed), or — if it is unavoidable — make sure the cyclic GC is enabled and let it do its job. The reason this matters for memory profiling specifically: a process whose RSS climbs in a *sawtooth* (up steadily, then a sudden drop) is usually accumulating cycles that the cyclic GC periodically sweeps; a process whose RSS climbs *monotonically with no drops* is a true reference leak — a live reference, not a cycle — and `gc.collect()` will not save it. The shape tells you which.

The `weakref` fix deserves a closer look because it is the cleanest cure for the two most common retention bugs: the cycle, and the registry that outlives its members. A weak reference points at an object *without* counting toward its reference count, so it cannot keep the object alive. The instant the last *strong* reference drops, the object frees and the weakref becomes dead (it returns `None` when you dereference it). That is exactly what you want for a back-pointer (child knows its parent, but the child should not keep the parent alive) and for an observer registry (the registry should remember a handler only as long as something *else* holds it):

```python
import weakref

class Child:
    def __init__(self, parent):
        # strong ref keeps parent alive forever -> a leak if parent should die:
        # self.parent = parent
        # weak ref: child sees parent but does NOT keep it alive
        self._parent = weakref.ref(parent)

    @property
    def parent(self):
        return self._parent()      # returns the parent, or None if it has been freed
```

For the registry case, `weakref.WeakValueDictionary` and `WeakSet` are the right containers: they hold their members weakly, so an entry disappears automatically when no strong reference to the value remains elsewhere. A metrics registry, an event-handler set, a cache of "currently active sessions" — each is a textbook leak when built with a plain `dict`/`set`, and a textbook fix when built with the weak variant. The mechanism is the same one the cyclic GC relies on but applied *eagerly*: by never forming the strong reference in the first place, you never need a collector to clean it up.

#### Worked example: a registry leak fixed with WeakValueDictionary

A long-running service kept a `dict` mapping session IDs to `Session` objects so background tasks could look them up. Sessions were added on connect but the cleanup-on-disconnect path had a bug, so closed sessions lingered in the registry, each pinning a ~180 KB buffer. `objgraph.show_growth()` across two snapshots printed `Session  +312` and `dict  +9`, and `show_backrefs` on a sampled `Session` traced the retaining chain straight to the module-level `_SESSIONS` dict — the exact shape in the retention figure. Switching `_SESSIONS` from a plain `dict` to a `weakref.WeakValueDictionary` meant that the *moment* the connection handler dropped its reference, the `Session` and its buffer freed, registry entry and all, with no cleanup code required. RSS over an 8-hour soak test went from a steady climb cresting **3.4 GB** to a flat plateau at **610 MB**. The bug in the cleanup path was still there — but it no longer leaked, because the weak registry made the leak structurally impossible.

## A leak versus a peak: reading the RSS curve

Everything so far converges on one operational skill: looking at an RSS-over-time curve and saying "leak" or "peak." Get this wrong and you waste days. The two shapes are completely different.

![a leak showing RSS climbing on every iteration toward an out of memory crash versus a healthy process holding a flat plateau after warmup](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-7.png)

A **leak** is monotonic growth that never returns: iteration 1 at 410 MB, iteration 1,000 at 2.1 GB, iteration 5,000 dead at the OOM killer. The defining feature is that the *baseline* between iterations keeps rising — after each unit of work, the process settles at a higher floor than before. A leak is caused by something that **accumulates**: a growing container (a list you `append` to and never clear), an **unbounded cache** (the `cache.py:88` we found), a **global registry** (handlers, observers, metrics objects that get registered and never deregistered), or a **reference cycle** that the GC is not collecting. The fix is always to bound the accumulation: cap the cache, clear the container, deregister on teardown, break the cycle.

A **peak** (or a one-time transient) is different: RSS climbs during warmup or during one heavy phase, reaches a **flat plateau**, and stays there — iteration 1,000 and iteration 5,000 both at 520 MB. That is *healthy*. The process built up its working set (loaded a model, filled a fixed-size cache, allocated its buffers) and then reached steady state. Memory that does not fall but also does not *climb* is not a leak; it is your working set plus fragmentation. The mistake here is the opposite one: panicking about a flat plateau and "fixing" a non-problem, or — worse — chasing the fragmentation residual from the freeing example earlier and concluding you have a leak when you have a perfectly stable process holding partially-used arenas.

How do you tell them apart in practice? **Run the workload long enough and watch the floor.** The simplest possible probe is a loop that logs RSS every N iterations:

```python
import os, psutil

proc = psutil.Process(os.getpid())
for i in range(10_000):
    handle_one_request()
    if i % 500 == 0:
        rss = proc.memory_info().rss / 1e6
        print(f"iter {i:>6}: RSS = {rss:7.1f} MB")
```

If the printed RSS keeps climbing every 500 iterations with no ceiling, it is a leak — go run the `tracemalloc` diff. If it climbs for the first few thousand and then flattens, it is a peak — your job is to lower the plateau (smaller working set, stream instead of buffer) or just accept it. This dumb loop has saved more debugging hours than any fancy tool, because it answers the only question that determines your entire approach: *is the floor rising?*

## A symptom-to-cause-to-tool decision table

When you are paged, you do not start from "let me profile everything." You start from a *symptom* — the shape of the failure — and that symptom narrows the cause and picks the tool. This is the triage table, and figure 8 renders it.

![a matrix mapping each memory symptom to its likely cause the tool to reach for and the fix on an eight core Linux box](/imgs/blogs/memory-profiling-tracemalloc-memray-and-finding-leaks-8.png)

| Symptom | Likely cause | Reach for | The fix |
| --- | --- | --- | --- |
| **OOM crash at peak**, otherwise fine | a transient peak (a big intermediate) or a true leak | `memray run` + high-water flame graph | stream/chunk; avoid the copy |
| **Slow creep**, RSS rises run after run | growing container / unbounded cache / registry | `tracemalloc` snapshot diff | bound the cache; clear; deregister |
| **Never shrinks** after a phase, but flat | fragmentation; held arenas; working set | `psutil` RSS + `gc.get_stats()` | isolate the phase in a subprocess |
| **Huge transient**, average looks fine | one big intermediate (a `.copy()`, a `concat`) | `memray --leaks` / high-water | operate in place; lazy / out-of-core |
| **Native RSS climbs**, `tracemalloc` flat | leak inside a C extension | `memray run --native` | fix/upgrade the library; release per call |

Walk the table top to bottom and notice how the *shape* dictates the tool. An OOM that happens only at a peak with an otherwise-fine average is a transient — `memray`'s high-water mark, not a leak hunt. A slow run-after-run creep is accumulation — `tracemalloc`'s diff. "Never shrinks but flat" is the fragmentation non-leak — confirm with `gc.get_stats()` and, if the phase really must release memory, run it in a child process and let process exit return everything to the OS (the nuclear-but-reliable option). And the tell-tale of a **native** leak is the mismatch: RSS climbs while `tracemalloc` stays flat, because the bytes are being allocated by `malloc` inside a library where `tracemalloc` cannot see — switch to `memray --native`.

That last row deserves emphasis because it is the one people get stuck on for days. If you have run a clean `tracemalloc` diff and the top growth line is a couple of megabytes while RSS has climbed by gigabytes, **stop looking in Python.** The bytes are in native code. `memray run --native` will show you the C frames; often the fix is a library upgrade (the leak is a known bug), or making sure you close/free the native resource (an un-closed file handle, a database cursor, a CUDA context) on every call rather than relying on `__del__`.

## Measuring memory honestly (the part everyone gets wrong)

A memory measurement is as easy to get wrong as a timing measurement, and the failure modes are just as sneaky. Before you trust a number, account for these.

**RSS is shared and lazy.** The number in `/proc/<pid>/status` includes pages shared with other processes (the C library, the interpreter binary) and is affected by copy-on-write. After a `fork` — which is exactly what `multiprocessing` does — child processes *share* the parent's pages until they write to them, so summing each child's RSS double-counts the shared parts. If you are profiling a multi-process service, measure with a tool that understands proportional set size (PSS), which divides shared pages among the processes sharing them, or profile a single worker in isolation. Reporting "the service uses 24 GB" when it is eight workers each reporting 3 GB of mostly-shared pages is a classic over-count.

**The OS gives memory back lazily, so RSS lags reality.** When pymalloc *does* return an arena, or when a large object is freed, the pages may not leave RSS immediately — the kernel reclaims them on its own schedule, or under pressure. So a momentary RSS reading taken right after a `del` can be misleading in *either* direction. For a trustworthy before-and-after, take the readings at stable points (after a `gc.collect()`, after the workload has quiesced), and repeat the whole measurement a few times to confirm the number is stable, exactly as you would repeat a timing run.

**The profiler perturbs what it measures.** `tracemalloc` adds its own per-allocation tracebacks to memory and roughly doubles allocation time, so the RSS you see *with tracing on* is higher than production and the timing is slower. `memray` in `run` mode is lighter but still non-zero. The correct discipline: use the profilers to *locate* the problem (which line, which call path), then turn them off and measure the *plain* RSS before and after your fix to quote the real win. Never quote a with-profiler-on RSS as the production number.

**Account for the cyclic GC's timing.** Because the cyclic collector runs on allocation thresholds, *when* you sample matters. A sample taken just before a collection shows accumulated cycles; just after, it shows them gone. For a clean steady-state reading, call `gc.collect()` immediately before sampling so you are measuring genuinely-live memory, not the transient pile waiting for the next sweep. And if you have tuned the GC — `gc.disable()` for throughput in a request handler, `gc.freeze()` after startup so the post-`fork` children do not copy-on-write the whole heap during collection — remember that disabling the cyclic GC means cycles never get collected at all, which converts a benign sawtooth into a real, unbounded leak. That GC-tuning trade-off — reference counting plus the generational cyclic collector, `gc.disable`/`gc.freeze`, and when objects actually free — is its own topic in the memory-and-the-machine track; here the rule is simply: know whether the cyclic GC is on when you read the curve.

The honest report, then, looks like the timing reports from the rest of the series: a named machine, a stated workload, the tool turned *off* for the final number, the reading taken at a stable point after a collection, and a before-and-after with both values. Anything less and you are quoting noise.

## Fixing the leak and proving the win

We found the leak: `cache.py:88`, an unbounded module-level response cache, one entry per request, never evicted. Here is the offending code and the fix, and then the measured before-and-after that is the whole point of memory profiling — you do not get to claim a fix without a number.

The leak:

```python
# app/cache.py  --  THE LEAK
_RESPONSE_CACHE = {}          # module-level: lives for the whole process

def get_or_compute(key, request):
    if key not in _RESPONSE_CACHE:
        _RESPONSE_CACHE[key] = expensive_compute(request)   # line 88: grows forever
    return _RESPONSE_CACHE[key]
```

Every distinct `key` adds a permanent entry. In a service where keys are derived from request parameters, the key space is effectively unbounded, so the cache grows without limit — a textbook leak. The fix is to **bound it**. The simplest correct fix is `functools.lru_cache` or an explicit bounded cache; an `lru_cache` evicts the least-recently-used entry once it hits `maxsize`, so memory plateaus instead of climbing:

```python
# app/cache.py  --  FIXED
from functools import lru_cache

@lru_cache(maxsize=10_000)     # at most 10k entries, LRU eviction
def get_or_compute(key, request_payload):
    return expensive_compute(request_payload)   # bounded: old entries evicted
```

(If the values are large and you can tolerate them being dropped under memory pressure, a `weakref.WeakValueDictionary` is an alternative — entries vanish when no one else references the value. And note `lru_cache` requires hashable arguments, so pass the hashable `key`, not the whole request object.)

Now the proof. I ran the leaking version and the fixed version on the Linux box, 5,000 requests each with a 50 KB payload, logging RSS:

| Version | RSS after 1k req | RSS after 5k req | Trend | Outcome |
| --- | ---: | ---: | --- | --- |
| Unbounded cache (leak) | 460 MB | 2,140 MB | climbing ~51 KB/req | would OOM at ~40k requests |
| `lru_cache(maxsize=10_000)` (fixed) | 455 MB | 968 MB | **flat after warmup** | stable indefinitely |

The leaking version's RSS climbed linearly and was on track to cross the 2 GB budget at ~40,000 requests — a few hours of traffic, matching the production crash cadence. The fixed version climbed during warmup as the LRU filled to its 10,000-entry cap, plateaued under 1 GB, and stayed there through 5,000 requests and well beyond. That is the leak-to-steady-state transition from figure 7, measured: **2,140 MB and rising → 968 MB and flat.** The before-and-after RSS, on the named box, with the workload stated, is the deliverable. "I fixed the leak" is a claim; "RSS went from 2.1 GB and climbing to under 1 GB and flat across 5,000 requests on the 8-core Linux box" is a result.

#### Worked example: confirming the fix with tracemalloc in production

Because you cannot always reproduce production traffic in staging, the durable fix includes a *guard*. I wired a tiny `tracemalloc` diff behind an admin endpoint: it keeps a baseline snapshot and, on request, returns the top 10 growth lines since startup. After deploying the `lru_cache` fix, the formerly-dominant `cache.py` line dropped off the list entirely — its growth delta fell from **+240 MB** to **+2.4 MB** (the LRU's fixed-size working set) and stopped growing. The endpoint now serves as a permanent canary: if any line ever shows hundreds of megabytes of monotonic growth again, it is visible in one HTTP call, no redeploy, no `memray` run. The cost is the ~2× tracing overhead, which for an admin-only diagnostic path is a fine trade. Cheap insurance against a 3 a.m. page.

## Case studies and real numbers

A few real-world patterns and figures worth calibrating against. As always, where I give a precise number it is from the named box; where the source is a project's own benchmark, I name it.

**The pandas `.copy()` and `concat` peak.** The single most common transient-peak culprit in data pipelines is pandas operations that silently materialize copies. `df.copy()`, `pd.concat`, a chained-assignment that triggers copy-on-write, a `groupby().apply()` that builds intermediate frames — each can briefly hold 2–3× the DataFrame's size. On a 9 GB frame that is the difference between a 10 GB peak and a 30 GB OOM, as in the worked example above. `memray`'s high-water-mark flame graph is the standard tool for catching these because the peak is transient and average monitoring misses it. The fix is usually to operate lazily or out-of-core; this is exactly where you cross-link out to a columnar engine like Polars or DuckDB (covered in the vectorization track) that streams instead of materializing, and where, if the data genuinely does not fit in RAM, you [push the work into a database](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool) rather than fighting the peak in-process.

**The unbounded-cache leak.** The `cache.py:88` story is not hypothetical — it is the most common Python service leak there is. Module-level dicts used as caches, `functools.lru_cache` with `maxsize=None` (which means *unbounded* — a frequent mistake, since `None` reads like a sensible default but disables eviction entirely), per-instance caches on objects that outlive their usefulness. The signature is always the same: `tracemalloc` diff shows one line growing one entry per request. The fix is always the same: bound it, and the RSS curve flattens.

**Native leaks in C extensions.** A class of leak that pure-Python profiling cannot touch. Historic examples include un-closed handles in older database drivers, leaks in image libraries, and tensor/GPU memory held by deep-learning frameworks (a PyTorch tensor kept alive by a Python reference, or a CUDA caching allocator that holds device memory). `tracemalloc` shows a flat Python footprint while RSS or GPU memory climbs — the mismatch *is* the diagnosis. `memray run --native` exposes the C frames; for GPU memory you reach for the framework's own allocator stats. When the bytes live below Python, you need a tool that sees below Python.

**The accidental-retention leak in a closure or a default argument.** Two of the most surprising leaks have nothing to do with caches. The first is the mutable **default argument**: `def add(x, acc=[]):` creates one list at function-definition time and reuses it across *every* call, so it grows forever — a `tracemalloc` diff points at the `def` line, and the fix is `acc=None` with `acc = acc or []` inside. The second is a **closure that captures a large object it does not need**: an inner function or a lambda registered as a callback that closes over the entire `self` or a big DataFrame, keeping it alive for as long as the callback is registered. `objgraph.show_backrefs` on the leaked object reveals the closure's cell in the chain, and the fix is to capture only the small piece you actually use (`value = big.field; lambda: value`) rather than the whole object. Both are invisible to `getsizeof` and obvious to the retention tools — which is the whole reason those tools exist.

**The `__del__`-on-a-cycle trap (pre-3.4 and lingering folklore).** A reference cycle that includes an object with a `__del__` finalizer used to be *uncollectable* — the cyclic GC refused to break it because it could not decide a safe finalizer order, so it parked the objects in `gc.garbage` forever. PEP 442 fixed this in Python 3.4 so finalizers on cycles now run, but the folklore and the occasional third-party `__del__` still bite. If `gc.garbage` is non-empty after a `collect()`, you have an uncollectable cycle; `objgraph.show_backrefs` on a member finds the cycle, and the fix is to drop the `__del__` in favor of an explicit `close()` or a `weakref.finalize`, then break the cycle. It is a rare leak today but a memorable one, because no amount of `tracemalloc` line-diffing explains *why* the freed-looking objects refuse to die — only the cycle view does.

**memray's own benchmarks.** Bloomberg, who built and open-sourced `memray`, designed it specifically for these production scenarios — native-aware tracking, the high-water mark for OOM diagnosis, and the `--leaks` mode for catching slow creeps — because Python's stdlib `tracemalloc` could not see native allocations and `memray`'s low `run`-mode overhead made it viable on realistic workloads. Its flame-graph output became the de facto way to communicate a memory problem visually, the same way `py-spy`'s flame graphs did for CPU. The pairing — `tracemalloc` for the always-available in-process diff, `memray` for the visual native-aware deep dive — covers essentially every Python memory question you will hit.

## When to reach for which tool (and when not to)

Memory profiling has a cost — overhead, complexity, and the time you spend reading graphs — so spend it deliberately.

**Use `sys.getsizeof` only on flat objects.** It is correct and instant for one `int`, one `bytes`, one `str`. The moment the object has referents — any container, any class instance — it lies, and you should switch to a deep walk or an allocation profiler. Do not build a memory accounting out of `getsizeof` sums; you will undercount referents and overcount shared objects, and the two errors do not cancel.

**Use `tracemalloc` when you want the allocating *line* and you are in-process.** It is in the standard library, needs no extra dependency, runs anywhere, and its snapshot-diff is the single best way to find *what grew over time* in a long-running service. Wire it behind an admin endpoint for a permanent canary. Do not leave full tracing on in production hot paths — the 2–3× allocation overhead is real — and remember it is blind to native allocations.

**Use `memray` when you want a *picture*, a *peak*, *native* visibility, or a *live* view.** The flame graph is the best artifact for explaining a memory problem; the high-water mark is the right view for an OOM; `--native` is the only way to see into C extensions; `--live` lets you watch a creep without waiting for a crash. Do not reach for it when a 30-second `tracemalloc` diff would answer the question — sometimes the stdlib tool is just faster.

**Use `objgraph` when you know *what* is leaking but not *why*.** `show_growth` finds the leaking type; `show_backrefs` finds the reference chain that pins it. This is the tool that actually fixes the leak, because the fix is always "drop this specific reference." Do not start here — start by localizing the *allocation* with `tracemalloc`/`memray`, then use `objgraph` to find the *retention*.

**Do not optimize memory you do not need to.** A flat 968 MB plateau on a box with 16 GB of RAM is not a problem; spending two days shaving it to 700 MB is wasted effort unless you are memory-constrained. Profile memory when you are actually near a limit — an OOM, a tight container budget, a per-worker cap that forces you to run fewer workers. The discipline is the same as the rest of [this series](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means): measure first, fix what the measurement points at, and prove the win with a before-and-after number. The forward step from "find where the memory goes" is "find where the *copies* go" — `scalene` and the modern profilers measure CPU, memory, and copy volume together, the subject of [the next post in the measurement track](/blog/software-development/python-performance/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together).

## Key takeaways

- **`sys.getsizeof` is shallow and lies on containers.** It counts the container's own array of pointers, not what they point at — a list of 100k dicts reports ~0.8 MB but truly holds ~30 MB. Size flat objects with it; use a `gc.get_referents` deep walk (deduplicated by `id`) for anything with referents.
- **RSS is not your live objects.** It stacks the interpreter and code, the pymalloc object heap, large native buffers, and fragmentation. Watch RSS for the *symptom*; diagnose the *cause* with an allocation profiler.
- **Freeing objects often does not shrink RSS.** pymalloc returns a 256 KB arena to the OS only when *every* pool in it is empty; scattered survivors keep arenas mapped. A non-falling curve after a phase is usually fragmentation, not a leak.
- **`tracemalloc` finds the allocating line and diffs over time.** `start`, `take_snapshot`, then `compare_to` to rank the lines whose live memory *grew* — that delta is your leak. It is stdlib and in-process but blind to native allocations.
- **`memray` sees native allocations, the high-water mark, and leaks.** `memray run` + `flamegraph` for the peak picture, `--leaks` for what never freed, `--native` for C frames, `--live` for a real-time TUI. The right tool for OOMs and for libraries.
- **`objgraph` answers "what holds this alive."** `show_growth` finds the leaking type; `show_backrefs` traces the retaining chain to the global, closure, or registry that should have let go. This is the tool that fixes the leak.
- **Reference counting frees promptly but cannot free cycles.** A cycle survives until the generational cyclic GC runs (or forever if GC is disabled). Break cycles explicitly or use `weakref`; a monotonic climb is a live-reference leak that `gc.collect()` will not save.
- **Read the curve to pick your approach.** Floor rising run after run = leak (growing container / unbounded cache / registry / cycle) → `tracemalloc` diff. Climbs then flat = peak/working set → lower the plateau or accept it. Random OOM at peak with a fine average = transient → `memray` high-water mark.
- **Always prove the fix with a before-and-after RSS number** on a named machine with the workload stated. "Fixed the leak" is a claim; "2.1 GB and climbing → 968 MB and flat across 5,000 requests" is a result.

## Further reading

- **Python docs — `tracemalloc`**: the official reference for `start`, `take_snapshot`, `Snapshot.compare_to`, `statistics`, and filtering. The canonical source for the snapshot-diff workflow.
- **Python docs — `gc` and `sys.getsizeof`**: the garbage collector module (`gc.collect`, `gc.get_referents`, `gc.get_stats`, `gc.set_debug`) and the `getsizeof` reference, including its explicit "this is shallow" caveat.
- **`memray` documentation (Bloomberg)**: the user guide for `memray run`, `flamegraph`, `--leaks`, `--native`, and the live TUI, with examples of reading allocation flame graphs and the high-water mark.
- **`objgraph` documentation**: `show_most_common_types`, `show_growth`, `show_backrefs`, and `by_type` — the reference-graph tools for finding what retains an object, with worked leak-hunting tutorials.
- **"High Performance Python," Gorelick & Ozsvald (O'Reilly)**: the memory chapters cover the object model, RSS, and profiling tools in depth; the standard practitioner reference for this material.
- **CPython internals — the `obmalloc.c` / pymalloc design notes**: for the arena/pool/block allocator and exactly when memory is returned to the OS, which explains the freeing-does-not-shrink-RSS behavior.
- **Series intro — [Why Python Is Slow, and What Fast Actually Means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means)**: the cost model and the leverage ladder this post sits inside.
- **Sibling — [Line and Statistical Profiling: line_profiler and py-spy](/blog/software-development/python-performance/line-and-statistical-profiling-line-profiler-and-py-spy)** and **[Scalene and Modern Profilers: CPU, Memory, and Copy Volume Together](/blog/software-development/python-performance/scalene-and-modern-profilers-cpu-memory-and-copy-volume-together)**: the time-profiling and unified-profiling companions to this memory deep dive.

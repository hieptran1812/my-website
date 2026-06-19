---
title: "Hunting Memory Leaks and Bloat: Reading a Climbing RSS to Its Root Cause"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn why a leak means something completely different in C than in Java or Go, and master the one technique that finds either kind: take two heap snapshots an hour apart, diff them, and follow the retainer path of whatever grew straight to the reference you forgot to release."
tags:
  [
    "debugging",
    "software-engineering",
    "memory-leaks",
    "heap-profiling",
    "garbage-collection",
    "tracemalloc",
    "pprof",
    "valgrind",
    "oom-killer",
    "root-cause-analysis",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/hunting-memory-leaks-and-bloat-1.png"
---

The page came in at 4:07am. The service had been fine all afternoon, fine through the evening deploy, fine when the on-call before me handed off at midnight. Then the dashboard tells the story in one ugly line: resident memory climbing in a near-straight diagonal from 220 MB at noon to 1.6 GB at 2am, never once dropping back, until at 4am it crossed the 2 GB cgroup limit and the Linux OOM-killer sent a `SIGKILL`. The pod restarted, RSS reset to 220 MB, and the climb began again. Every night, the same diagonal. Every night, the same 4am death.

That diagonal is the single most important diagnostic in this entire post. A healthy process that allocates and frees produces a *sawtooth*: memory rises as work arrives, then drops back to a baseline after garbage collection or after the request finishes. A leaking process produces a *staircase that only goes up* — a line that climbs across every GC cycle and never returns to where it started. The shape tells you what kind of bug you have before you have read a single line of code. We will spend this post turning that shape into a name, a line number, and a one-line fix, using the same loop that anchors this whole series: observe the symptom, reproduce it under controlled load, form a falsifiable hypothesis, bisect the gap between belief and truth, fix the held reference, and prevent the regression from coming back. (If that loop is new to you, start with the intro map, [Stop Guessing: The Scientific Method of Debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), and come back.)

![A stacked diagram showing resident set size climbing from 220 MB at noon to 1.6 GB at 2am and crossing a 2 GB cgroup limit so the OOM-killer fires and the engineer is paged at 4am.](/imgs/blogs/hunting-memory-leaks-and-bloat-1.png)

Here is what makes memory leaks special among bugs, and why they deserve a dedicated field manual: the crash happens *nowhere near the cause*. A null-pointer dereference crashes on the line that dereferences the null. A divide-by-zero crashes on the division. But a leak crashes at 4am, hours and gigabytes after the line that actually held onto the memory ran for the last time. The stack trace at the moment of death — if you even get one — points at whatever innocent allocation happened to be the straw that broke the cgroup, which is almost never the culprit. You cannot debug a leak by reading the crash. You have to debug it by reading the *trend*, and that is a fundamentally different skill. By the end of this post you will be able to do five concrete things: read an RSS curve and decide leak versus growth versus fragmentation; take and diff heap snapshots in C, Go, Python, Node, and Java; follow a retainer or dominator path from a leaked object back to the reference that holds it; bound a runaway cache or list so the curve flatlines; and prove the fix with before and after numbers you can defend.

## 1. What a leak actually is (and why the answer depends on your runtime)

The word "leak" hides a trap. It means two genuinely different things depending on whether your language manages memory for you, and if you carry the wrong assumption into the wrong runtime you will look for the bug in the wrong place for hours. Let me make the distinction concrete, because this is the mechanism that everything else in the post rests on.

In an *unmanaged* runtime — C, C++, Rust with `unsafe`, anything where you call `malloc`/`free` or `new`/`delete` yourself — a leak is **allocated memory that no live pointer can reach anymore**. You `malloc`'d 4 MB, you stored the pointer in a local variable, and then you overwrote that variable, or returned from the function, or reassigned the pointer to a fresh allocation, *without* calling `free` first. The 4 MB is still allocated. The operating system still counts it against your process. But there is no longer any pointer anywhere in your program that holds its address. The block is *lost* — unreachable, unrecoverable, dead weight until the process exits. This is a leak in the classic, literal sense: memory has escaped your accounting entirely.

In a *managed* runtime — Java, Go, Python, JavaScript, C#, anything with a garbage collector — that classic leak is **impossible**. The whole job of the garbage collector is to find unreachable memory and reclaim it automatically. The instant the last reference to an object disappears, the object becomes eligible for collection, and the next GC cycle frees it. You literally cannot lose a block the way C loses it. So when a Java service climbs to OOM, you are not looking at lost memory. You are looking at the exact opposite: memory that is *still reachable*, that the GC examined and correctly decided to keep, because somewhere in your program a live reference still points at it. A managed leak is not memory you forgot to free; it is memory you forgot to *stop referencing*. The growing cache with no eviction, the static list you append to forever, the event listener you never removed, the closure that captured a 50 MB buffer it didn't need, the goroutine that blocked and never exited and still owns its stack — every one of these is a reference you left live by accident.

![A before-and-after diagram contrasting a C leak as an unreachable lost block the allocator can never reclaim against a garbage-collected leak as a still-reachable object kept alive by a forgotten reference.](/imgs/blogs/hunting-memory-leaks-and-bloat-2.png)

This is why the same word demands two different hunts. In C you ask *"what did I allocate and forget to free?"* and you look for an allocation site whose pointer escapes without a matching `free`. In Java you ask *"what is still pointing at this and why?"* and you look for a *retainer* — the chain of references that keeps the object alive. The C question is answered by allocation tracking (Valgrind's memcheck, ASan's leak detector). The Java question is answered by retainer analysis (a heap dump and the path to GC roots). If you bring the C question to a Java heap you will hunt for a missing `free` that does not exist; if you bring the Java question to a C heap you will hunt for a retainer when the real problem is a truly lost block. Name your runtime first.

It helps to see the two worlds side by side, because the table is the cheat sheet you carry into every leak:

| | Unmanaged (C / C++) | Managed (Java / Go / Python / JS) |
|---|---|---|
| What "leak" means | block with no live pointer, unreachable | object still reachable, reference forgotten |
| Is a classic lost block possible? | yes — this *is* the leak | no — the GC reclaims unreachable memory |
| The question you ask | "what did I allocate and not free?" | "what is still pointing at this, and why?" |
| The diagnostic | allocation tracking (ASan, Valgrind) | retainer analysis (path to GC roots) |
| The fix | add the missing `free`/`delete` | break the reference; bound or evict |
| Typical culprit | overwritten pointer, early return, error path | unbounded cache, listener, ThreadLocal, closure |

Read that table left to right and you will notice the columns are almost mirror images. The unmanaged leak is a *deficit* — you did too little, you forgot a `free`. The managed leak is a *surplus* — you did too much, you kept a reference you meant to drop. That inversion is why the same word produces such different bug hunts, and why the first question on any leak is "managed or unmanaged?" Everything downstream forks on the answer.

A note on terminology, because we will use these words constantly. **RSS** (resident set size) is the amount of physical RAM your process currently occupies — what the OOM-killer and your cgroup limit actually count, and what your dashboard graphs. It is distinct from **virtual memory** (address space reserved, much of it never touched) and from **heap size** as your runtime reports it internally. **GC roots** are the starting points the garbage collector treats as definitely-alive: global variables, static fields, the local variables on every thread's stack, JNI references. An object is reachable — and therefore kept — if and only if there is a path of references from some GC root to it. A **retainer** is a reference that holds an object alive; the **retainer path** is the chain of retainers from the object back to a root. The **dominator** of an object, in heap-analysis terms, is the object through which *all* paths from the roots must pass to reach it — cut the dominator and the whole subtree it dominates becomes collectable. Hold these four words; they are the vocabulary of every managed-leak hunt.

## 2. The taxonomy: six places a forgotten reference hides

Because a managed leak is always a forgotten reference, and there are only so many ways to forget a reference, the entire problem space collapses into a short taxonomy. If you have a managed leak, it is almost certainly one of these six families. Learning to recognize the family from the symptom is half the battle, because the family tells you which reference to go cut.

![A tree diagram organizing managed memory leaks into growing containers, bindings that outlive their use, and threads that never exit, with leaf causes including unbounded caches, uncleared event listeners, capturing closures, and uncleared thread-local storage.](/imgs/blogs/hunting-memory-leaks-and-bloat-3.png)

The first family is **the growing container**: a `Map`, `List`, `Set`, or array that you add to and never remove from. The unbounded in-memory cache with no eviction policy is the canonical example — every distinct key you ever see adds an entry, and the entries live as long as the cache does, which is usually the lifetime of the process. Its evil twin is the **static or global collection** you append to "temporarily" — a module-level list that accumulates parsed rows, a global registry of every object ever created, a deduplication set that only grows. These are the most common production leaks by a wide margin, and they share a tell: RSS grows in proportion to *cumulative distinct work*, not concurrent work. Ten thousand unique users over a day will leak ten thousand entries even though only fifty are active at once.

The second family is **the binding that outlives its use**. The classic is the **event listener or callback never removed**: you subscribe a handler to an emitter, the handler closes over a big object, the object's owner goes away, but the emitter still holds the handler, the handler still holds the object, and the object never dies. JavaScript front-ends leak this way constantly (a component mounts, adds a `resize` listener, unmounts without removing it), and Node back-ends leak it through `EventEmitter` subscriptions that accumulate — Node even warns you with `MaxListenersExceededWarning` when more than ten pile up on one emitter. Its sibling is the **closure that captures more than it needs**: a closure in JavaScript or Python keeps alive *every* variable in its enclosing scope that it references, and if you accidentally reference one big variable, the whole scope's heavy objects are pinned for as long as the closure lives.

The third family is **the thread or task that never exits**. A thread that blocks forever on a socket read, a goroutine waiting on a channel nobody will ever send to, an async task awaiting a promise that never resolves — each one holds its entire stack alive, plus everything its stack frames reference, plus (crucially) any thread-local storage attached to it. **ThreadLocal that is never cleared** is a notorious Java leak: in a thread pool, threads are reused across thousands of requests, so a `ThreadLocal` value set during request A and never removed stays attached to that pooled thread and survives into requests B, C, and D forever. The Go version is the **goroutine leak**: launch a goroutine per request, have it block on an unbuffered channel because the receiver already returned, and you accumulate one stuck goroutine — and its stack — per request, visible as a climbing goroutine count in `pprof`.

Knowing the taxonomy lets you form a *hypothesis* before you profile. Does RSS track concurrent load (rise and fall with traffic) or cumulative work (only ever rise)? Cumulative-work growth screams container or static collection. Does the leak appear only after long uptime? That favors slow accumulators — caches, dedup sets, listener pile-ups. Does your goroutine or thread count climb alongside RSS? That is your thread-never-exits family, and you can skip straight to a goroutine dump. The taxonomy turns a blank-page hunt into a multiple-choice question.

## 3. Read the curve first: leak vs legitimate growth vs fragmentation

Before you reach for a profiler, spend two minutes reading the RSS curve, because not every rising line is a leak, and the profiler cannot tell you which of three very different problems you have. Reaching for a heap diff when the real issue is allocator fragmentation will waste an afternoon — there is no forgotten reference to find, because nothing is leaked in the heap-object sense at all.

![A grid table mapping three causes of high memory, a true leak, legitimate growth, and allocator fragmentation, to their distinct signatures and their distinct fixes.](/imgs/blogs/hunting-memory-leaks-and-bloat-8.png)

A **true leak** has a signature you now know: RSS rises monotonically *across* GC cycles, and a heap-snapshot diff shows some object type growing without bound. The fix is to find and cut the held reference, or add an eviction policy. This is the case the rest of the post is about.

**Legitimate growth** looks alarming but is not a bug. Many systems are designed to fill memory up to a ceiling and stay there: a database buffer pool grabs RAM up to its configured size and keeps it; a JIT warms up and allocates compiled code; a connection pool or worker pool grows to its max under load; a runtime's own GC heap expands toward its `-Xmx` or `GOMEMLIMIT` ceiling and holds the pages because giving them back is expensive. The tell is that the curve **rises with load and then plateaus** — it has a ceiling and it stops climbing once it reaches steady state, and a heap diff under steady load shows *no* runaway type. The fix, if there even is a problem, is to raise the limit or size the box correctly. The single most common false alarm I see is an engineer paging themselves over a JVM that grew to its `-Xmx` and stayed there, which is the JVM doing exactly what you told it to.

**Fragmentation and bloat** is the subtle one. Here the heap *as your runtime sees it* is not growing — the live-object total is flat — but RSS is high and will not come down, because the underlying allocator is holding pages it cannot return to the OS. This happens because allocators like glibc's `malloc`, `jemalloc`, and `tcmalloc` carve memory into arenas and size classes; if you allocate a million small objects, free most of them, but a few survivors are scattered across many pages, the allocator cannot `munmap` a page that still has one live object on it. The page stays resident even though it is 99% empty. The Java large-object/old-generation analog and the glibc per-arena retention both produce the same fingerprint: **runtime heap utilization low, OS RSS high, gap not closing**. The fix is not a reference cut — there is nothing leaked. It is allocator tuning: switch to or tune `jemalloc` (its `background_thread` and `dirty_decay_ms` settings control how aggressively it returns pages), call `malloc_trim(0)` to nudge glibc to release free pages at the top of the heap, set `MALLOC_ARENA_MAX` to cap glibc's per-thread arenas (a huge RSS reducer for thread-heavy services), or compact the heap. We dig into the diagnostics for this in section 8; for now, the rule is: **if heap-internal live size is flat but RSS is high, you have bloat, not a leak — stop hunting references.**

Here is the two-minute triage you run before anything else, on Linux:

```bash
# Watch RSS over time. The shape decides your whole investigation.
# Column 6 of ps is RSS in KB.
while true; do
  date +%H:%M:%S
  ps -o rss= -p "$PID" | awk '{printf "  RSS %.1f MB\n", $1/1024}'
  sleep 60
done

# Faster: smem gives proportional set size (PSS) too, which is fairer
# when memory is shared across processes.
smem -P "myservice" -c "pid pss rss"

# For the heap-vs-RSS gap that reveals fragmentation, ask the runtime:
#   Go:     curl localhost:6060/debug/pprof/heap?debug=1 | grep -E 'HeapInuse|HeapReleased|Sys'
#   Java:   jcmd $PID GC.heap_info
#   glibc:  cat /proc/$PID/smaps_rollup   (look at Rss vs the heap your runtime reports)
```

If the curve sawtooths and returns to baseline, you are fine — that is normal allocate-and-free behavior. If it rises and plateaus, suspect legitimate growth and check whether you have hit a configured ceiling. If it rises forever across GC cycles, you have a real leak, and now the profiler earns its keep.

## 4. The mechanism: why the GC keeps your "garbage" alive

To debug a managed leak you have to understand exactly what the garbage collector does, because the leak is not a GC failure — it is the GC doing its job correctly on an object you wrongly believe is dead. This is the rigorous mechanism block for this post, and getting it precisely right is what separates "the GC is broken" (it almost never is) from "I left a reference live."

A tracing garbage collector works by **reachability**. It starts from the **root set** — every global variable, every static field, every local variable and operand on every thread's call stack, plus runtime-internal roots like JNI handles or interned strings — and it performs a graph traversal (mark phase) following every reference it finds. Every object it can reach is marked *live*. Every object it cannot reach is, by definition, *garbage*, and the sweep or copy phase reclaims it. The defining theorem of garbage collection is simple and absolute: **an object is reclaimed if and only if it is unreachable from the root set.** There is no "is this object still useful?" judgment — the GC has no idea what "useful" means. It only knows reachable or not.

This is the entire mechanism of a managed leak. Your object is *useless* (you are done with it, it should be freed) but it is *reachable* (some reference path from a root still leads to it). The GC visits it during the mark phase, sees a live path, marks it live, and keeps it. From the GC's perspective this is flawless behavior: there *is* a live reference, so the object *is* live. The bug is not in the collector. The bug is that you, the programmer, left a reference live that you intended to be dead. The cache entry you never evicted is reachable from the cache, which is reachable from a static field, which is a root. The event listener is reachable from the emitter's listener list, which is reachable from a long-lived emitter. The closure's captured buffer is reachable from the closure, which is reachable from whatever holds the closure. In every case there is an unbroken chain from a root to your "garbage," and as long as that chain exists, the GC is contractually obligated to keep your garbage alive.

This mechanism has three immediate, practical consequences. First, **the fix is always to break the chain** — null out the field, remove the listener, evict the entry, let the closure go out of scope, drain the goroutine. Cut any one edge on the path from root to object and the GC reclaims it on the next cycle. Second, **the diagnosis is always to find the chain** — the entire skill of managed-leak hunting is "follow the references backward from the leaked object until you reach the root, and find which edge you can cut." That backward walk is exactly what a heap dump's "path to GC roots" or a DevTools "retainers" panel gives you. Third, **weak references exist precisely to opt out of this** — a weak reference (Java `WeakReference`/`WeakHashMap`, JS `WeakMap`/`WeakRef`, Python `weakref`) does *not* count as a retaining edge, so the GC is free to reclaim an object that is only weakly reachable. This is the right tool for caches and listener registries that should not keep their contents alive on their own.

![A directed graph tracing the retainer path from a garbage-collection root through a module singleton and an unbounded map cache down to a 2 KB parsed payload, with one edge marked as the reference to cut by adding eviction.](/imgs/blogs/hunting-memory-leaks-and-bloat-4.png)

There is one more layer worth understanding because it explains *why a leak looks like a clean diagonal line* rather than a jagged one. Most production GCs are **generational**: they exploit the empirical fact that most objects die young, so they collect a small "young generation" frequently and cheaply, and promote the survivors to an "old generation" that they collect rarely and expensively. A leaked object — one with a forgotten live reference — survives every young collection (it is reachable), gets promoted to the old generation, and then sits there essentially forever, because the full old-gen collection that might examine it runs infrequently and, even when it runs, correctly finds the object still reachable and keeps it. So a leak quietly fills the old generation. The young-gen sawtooth keeps bouncing (that is healthy churn), but underneath it the old-gen baseline ratchets upward one promotion at a time, producing the smooth climbing floor you see on the dashboard. When that floor reaches the heap ceiling, the GC thrashes — it runs full collections back to back trying to reclaim space that is all legitimately reachable — and you see CPU spike and latency collapse just before the OOM. That "GC death spiral" right before the crash is the old generation full of leaked, reachable objects. The number to watch is therefore not the bouncing total heap but the *post-full-GC old-gen occupancy*: if it ratchets up after each full GC, that ratchet is your leak, isolated from all the healthy young-gen noise.

So when someone says "the garbage collector has a leak," they are almost always wrong. The GC is a function of reachability, and reachability is a function of *your* references. The leak is in your reference graph. Now let me show you how to read that graph.

## 5. The killer technique: snapshot, churn, snapshot, diff

If you remember one thing from this entire post, make it this: **the single most effective way to find a managed leak is to take two heap snapshots under steady load and diff them.** Not read one snapshot — diff two. A single snapshot tells you what is in memory right now, which is mostly the legitimate working set drowning out the leak. The *difference* between two snapshots taken an hour apart, while the system does the same kind of work the whole time, isolates exactly what grew. And whatever grew, under steady-state load where the working set should be constant, is your leak. Everything else in this post is a language-specific way of performing this one diff.

![A timeline of the three-snapshot technique: capture a baseline, drive ten thousand requests of steady load, capture a second snapshot an hour later, diff the two to find the type that grew, then follow its retainer to the leak.](/imgs/blogs/hunting-memory-leaks-and-bloat-6.png)

The reasoning is worth making explicit because it is what makes the technique bulletproof. Under steady-state load — the same requests per second, the same mix of work, for the whole window — a *healthy* system reaches a stable working set: the objects in memory at snapshot 1 and snapshot 2 should be roughly the same set, just different instances, because anything allocated for a request is freed when the request ends. A *leaking* system, under that same steady load, accumulates: some object type that should have been freed is being retained instead, so at snapshot 2 there are strictly more of them than at snapshot 1. The diff cancels out the entire stable working set — it appears in both snapshots, so it nets to roughly zero — and leaves behind only the growth. The growth is the leak. This is why the diff is so much sharper than a single snapshot: it subtracts away the noise of normal operation and surfaces the signal of accumulation.

In Chrome DevTools this technique has a name — the **three-snapshot technique** — and it is worth describing precisely because it is the cleanest UI for the idea. You take snapshot 1, perform the action you suspect leaks N times (say, open and close a dialog fifty times), take snapshot 2, perform the action N times again, take snapshot 3. Then in the snapshot 3 view you switch the comparison dropdown to "**Objects allocated between snapshots 1 and 2**." This shows you objects that were created during the first batch and *survived* into snapshot 3 — objects that should have been freed when the dialog closed but were not. If your action is leak-free, that view is nearly empty. If it leaks, you will see a pile of objects (DOM nodes, closures, your own classes) that were allocated during the first batch and are still alive two batches later, and you can click any of them to see its **retainers** — the path back to the root, which is the fix. Two batches matter: the second batch ensures the survivors from batch one really are stuck and not just waiting for a delayed cleanup.

The same diff exists in every runtime, just with different syntax:

| Runtime | Snapshot tool | The diff move | What the diff names |
|---|---|---|---|
| Python | `tracemalloc` | `snap2.compare_to(snap1, 'lineno')` | the source line gaining allocations |
| Python | `objgraph` | `objgraph.show_growth()` | the type whose instance count grew |
| Go | `pprof` heap | `go tool pprof -base old.pb.gz new.pb.gz` | the function holding the growing allocation |
| Node / JS | DevTools heap snapshot | three-snapshot "allocated between 1 and 2" | constructor + retainer path |
| Java / JVM | `jmap` heap dump + Eclipse MAT | dominator tree of two dumps | class + path to GC roots |
| C / C++ | Valgrind `massif` / jemalloc prof | `ms_print` peaks / `jeprof --base` | the call site that allocated the bytes |

![A matrix mapping each runtime to its native heap profiler, its snapshot-diff command, and the leaking object type that the diff names, covering C and C plus plus, Go, Python, Node, and Java.](/imgs/blogs/hunting-memory-leaks-and-bloat-5.png)

Notice the column that matters: every one of these tools, when you give it two snapshots instead of one, names *the thing that grew*. That name — a type, a line, a constructor, a class — is the entire goal of the investigation, because once you know *what* is accumulating, the retainer path tells you *why*, and the why is the fix. Let me make this concrete in three languages.

## 6. Method in Python: tracemalloc diff and objgraph growth

Python ships a heap-diff tool in the standard library: `tracemalloc`. It records the file and line of every allocation, and it can diff two snapshots and tell you, sorted by growth, exactly which source lines are accumulating memory. No third-party dependency, no recompile, works in production behind a feature flag. Here is a complete, runnable diff harness you can drop into a long-running worker.

```python
import tracemalloc
import linecache

# Start tracing as early as possible. frames=25 keeps deeper tracebacks
# so you can see the *caller* that triggered the leaking allocation,
# not just the leaf line inside a library.
tracemalloc.start(25)

_baseline = None

def snapshot_baseline():
    """Call once after warmup, under steady load."""
    global _baseline
    _baseline = tracemalloc.take_snapshot()

def report_growth(top_n=15):
    """Call again an hour later, under the SAME steady load."""
    current = tracemalloc.take_snapshot()
    stats = current.compare_to(_baseline, "lineno")
    print("\n=== top allocations by GROWTH since baseline ===")
    for stat in stats[:top_n]:
        # stat.size_diff is bytes gained; stat.count_diff is objects gained.
        frame = stat.traceback[0]
        line = linecache.getline(frame.filename, frame.lineno).strip()
        print(f"{stat.size_diff/1024:+9.1f} KB  "
              f"{stat.count_diff:+7d} objs  "
              f"{frame.filename}:{frame.lineno}")
        print(f"            {line}")
```

The output, run an hour apart under steady load on a leaking worker, looks like this — and this is what makes the technique feel like magic the first time:

```bash
=== top allocations by GROWTH since baseline ===
 +84211.3 KB  +412904 objs  /app/ingest/parser.py:88
            _ROWS.append(parsed)
   +12.4 KB      +31 objs  /usr/lib/python3.11/json/decoder.py:353
            obj, end = self.raw_decode(s, idx)
    +2.1 KB       +5 objs  /app/server.py:142
            response = handler(request)
```

The first line is the leak, full stop. Eighty-four megabytes and 412,000 objects accumulated on `parser.py:88`, which reads `_ROWS.append(parsed)` — a module-level list named `_ROWS` that the parser appends every parsed row to and never clears. The second and third lines are noise (a handful of KB of churn from normal request handling), and the diff correctly demotes them. You did not read any code to find this; you let the diff name the line. That is the whole point: the diff converts "memory is growing somewhere in 40,000 lines" into "memory is growing on `parser.py:88`."

When `tracemalloc`'s line-level view is not enough — say the accumulating type is allocated in one place but *retained* somewhere else, so the allocation line is innocent — reach for `objgraph`, which works at the object-graph level and can show you *what keeps an object alive*:

```python
import objgraph, gc

# What types are growing? Call twice, a while apart.
objgraph.show_growth(limit=20)
# Prints e.g.:   ParsedRow   412904   +412904   <- the leaking type

# Now: WHO holds a ParsedRow alive? Pick one and walk its referrers
# back toward the roots. This renders the retainer chain to a PNG.
rows = objgraph.by_type("ParsedRow")
objgraph.show_backrefs(
    rows[0], max_depth=8, filename="retainer_chain.png",
)

# Or inspect the live object graph directly:
import sys
obj = rows[0]
referrers = gc.get_referrers(obj)
print("held by:", [type(r).__name__ for r in referrers])
# -> held by: ['list']   (the module-level _ROWS)
```

`objgraph.show_growth()` is the Python equivalent of "which type grew between snapshots," and `show_backrefs` / `gc.get_referrers` walk the retainer chain backward, the Python equivalent of "path to GC roots." Between `tracemalloc` for *where it was allocated* and `objgraph` for *what holds it*, you can localize essentially any Python leak. (For a deeper treatment of Python's allocator internals and these exact tools, the python-performance series has a dedicated walkthrough in [Memory Profiling: tracemalloc, memray, and Finding Leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks), and the object/refcount/GC model that underpins all of this is laid out in [The Python Memory Model: Objects, Refcounts, and the Garbage Collector](/blog/software-development/python-performance/python-memory-model-objects-refcounts-and-the-garbage-collector) — I cross-link rather than re-derive the CPython internals here.)

#### Worked example: the Python worker that flatlined

A batch ingestion worker climbed about 80 MB per hour and got OOM-killed roughly every fourteen hours by its 1.2 GB memory limit. The shape was the classic monotonic staircase, so we knew it was a leak and not load. We started `tracemalloc` with `frames=25`, took a baseline after the warmup burst, drove the worker at its normal steady rate (about 1,100 rows per second), and called `report_growth()` exactly one hour later. The top line, by a factor of several thousand over everything else, was `parser.py:88: _ROWS.append(parsed)` with +84 MB and +412,904 objects. The line existed because someone, long ago, added `_ROWS` as a module-level accumulator to support a "dump everything we parsed" debug endpoint that nobody used anymore — but the append ran on every single row, forever, at module scope, so the list never went out of scope and the GC never had grounds to free it. Every reference was live; the GC was doing its job perfectly.

The fix was one decision: did anything actually need the full history? It did not. We replaced the unbounded list with a bounded LRU of the last 10,000 rows (`collections.OrderedDict` with `popitem(last=False)` on overflow, or `functools.lru_cache` on the wrapping function) so the debug endpoint still worked but the memory was capped. The before/after was unambiguous: RSS had been climbing +80 MB/hr and crossing 1.2 GB every ~14 hours; after the fix it rose during warmup to about 240 MB and then *flatlined* — held within a few MB of 240 for the entire six-day soak we ran before declaring victory. The OOM-kill rate went from roughly twice a day to zero over six days. We proved it not by arguing about the code but by watching the curve go flat, which is the only proof a leak fix should ever be accepted on.

## 7. Method in Go: pprof heap and the -base diff

Go's runtime ships a production-grade heap profiler behind the `net/http/pprof` package, and it supports the exact diff move we keep returning to: `-base`. You expose the pprof endpoints, grab a heap profile now, grab another later, and diff them with `go tool pprof -base`. The diff shows you, per call site, how many bytes and objects *grew* between the two profiles — and under steady load, the growth is the leak.

```go
import (
    "net/http"
    _ "net/http/pprof" // registers /debug/pprof/* handlers
)

func main() {
    // Expose pprof on a private port. NEVER on a public interface.
    go func() { http.ListenAndServe("localhost:6060", nil) }()
    // ... your real server ...
}
```

The investigation session, end to end, looks like this:

```bash
# 1. Capture a baseline heap profile after warmup, under steady load.
#    ?gc=1 forces a GC first so you measure live (in-use) memory, not
#    transient garbage that is about to be collected anyway.
curl -s "http://localhost:6060/debug/pprof/heap?gc=1" > heap_t0.pb.gz

# 2. Let the service run an hour under the SAME steady load.
sleep 3600

# 3. Capture a second profile.
curl -s "http://localhost:6060/debug/pprof/heap?gc=1" > heap_t1.pb.gz

# 4. DIFF them. -base subtracts t0 from t1: what GREW.
#    -inuse_space is the default and the right metric for a leak
#    (live bytes, not cumulative allocations).
go tool pprof -base heap_t0.pb.gz heap_t1.pb.gz

# At the (pprof) prompt:
(pprof) top
#   flat  flat%   sum%        cum   cum%
# 158.40MB 96.1% 96.1%   158.40MB 96.1%  main.(*Cache).Put
#   3.20MB  1.9% 98.0%     3.20MB  1.9%  encoding/json.Unmarshal
(pprof) list main.\(\*Cache\).Put   # show source with per-line bytes
(pprof) web                          # render the diff as a call graph SVG
```

The `top` output names the leak the same way the Python diff did: `main.(*Cache).Put` grew by 158 MB while everything else grew by single-digit MB. `list` then shows you the exact line inside `Put` that holds the bytes — invariably an insert into a map that has no corresponding delete. The `-base` flag is the whole trick; without it you would be reading the cumulative profile, where the legitimate working set buries the leak.

Go has a second leak class that pprof catches beautifully and that has *no analog* in most other runtimes: the **goroutine leak**. A goroutine that blocks forever (on a channel send/receive that will never complete, on a mutex, on a network read with no timeout) never returns, so its stack is never reclaimed and everything its stack references is pinned. The goroutine profile counts them:

```bash
# How many goroutines, grouped by where they're stuck?
curl -s "http://localhost:6060/debug/pprof/goroutine?debug=1" | head -40
# If this count climbs monotonically alongside RSS, you have a goroutine
# leak: each request spawns a goroutine that never exits. The stack
# traces show you EXACTLY which line they're all blocked on, e.g.
#   42031 @ ...  chan receive (nil chan)
# means 42,031 goroutines are all blocked receiving on a channel that
# nobody will ever send to -- one per leaked request.
```

A climbing goroutine count is a dead giveaway, and the dump's grouped stack traces point at the exact blocking line. The fix is almost always a missing `context` cancellation, a missing timeout on a channel receive (use `select` with a `<-ctx.Done()` case), or a producer that returns without closing its channel. As with every leak in this post, the proof is the curve: goroutine count and RSS both flatten once the stuck goroutines can exit.

#### Worked example: the Node service traced by three snapshots

A Node API gateway climbed RSS by about 40 MB per hour and died on its 2 GB container limit around 4am every night — the exact scenario that opens this post. Node uses the same V8 heap as Chrome, so the same heap-snapshot tooling applies. We attached with `node --inspect` (in production you can send `SIGUSR1` to an already-running Node process, or expose the inspector on `localhost` only), opened `chrome://inspect`, and ran the three-snapshot technique against a synthetic steady load that replayed real request shapes at the production rate.

Snapshot 1: baseline, 230 MB heap. We drove 10,000 requests. Snapshot 2. Drove 10,000 more. Snapshot 3, now 480 MB heap. We switched the comparison to "objects allocated between snapshots 1 and 2" and one constructor dominated the list: 9,900-odd `Object` entries plus a matching pile of `Buffer`s, all still alive two batches later. Clicking one and reading its **Retainers** panel gave the whole chain in one screen: the object was held by an entry in a `Map`, the `Map` was held by a property named `requestCache` on a module-level singleton, and the singleton was a GC root. The retainer path *was* the bug: an in-memory `Map` keyed by request id, written on every request, never deleted, never bounded. Each request added roughly a 4 KB entry (parsed body plus metadata), 10,000 requests added ~40 MB, and at the production rate of ~10,000 requests/hour that is exactly the +40 MB/hr we measured. The retainer path explained the slope to the megabyte.

![A before-and-after diagram showing a Node service whose unbounded map climbs resident memory forty megabytes an hour to an out-of-memory kill at two gigabytes, then flatlines at a three-hundred-ten-megabyte plateau once the map is bounded by a ten-thousand-entry LRU.](/imgs/blogs/hunting-memory-leaks-and-bloat-7.png)

The fix was to bound the `Map`. We replaced the bare `Map` with an LRU capped at 10,000 entries (the `lru-cache` package, `max: 10000`, which evicts the oldest entry on overflow), so the cache still served its purpose — deduplicating in-flight retries within a short window — but could never hold more than 10,000 entries at once. The before/after RSS:

| | Unbounded Map | LRU bounded to 10,000 |
|---|---|---|
| RSS at noon | 220 MB | 220 MB |
| Slope | +40 MB/hr, no ceiling | rises to a 310 MB plateau, then flat |
| 4am | OOM-killed at 2 GB | 310 MB, healthy |
| Restarts/day | 1 (nightly OOM) | 0 over 6 days |

The plateau at 310 MB instead of back at 220 is itself informative and worth not panicking over: a full 10,000-entry LRU legitimately holds ~40 MB of live entries plus V8 overhead, so a *higher flat line* is the correct, healthy outcome — bounded retention, not zero retention. We confirmed the leak was dead by repeating the three-snapshot run post-fix: "objects allocated between 1 and 2" was now nearly empty, because the LRU evicted batch one's entries before snapshot 3. The diff that found the leak also certified the fix.

## 8. Method in unmanaged C/C++: the truly lost block

C and C++ are the other half of the world, and the hunt is genuinely different because the leak is a *lost* block, not a retained one. There is no GC and no reachability graph to walk backward; there is a `malloc` with no matching `free`. The tools therefore work by *tracking allocations* and reporting which ones were never freed and where they were allocated.

The first reach is **AddressSanitizer's leak detector** (LeakSanitizer), which is built into modern Clang and GCC and is nearly free to enable. Compile with `-fsanitize=address`, and on a clean exit it reports every block that was still allocated and unreachable, with the allocation stack:

```bash
clang -g -fsanitize=address -O1 leak.c -o leak
./leak
# =================================================================
# ==12731==ERROR: LeakSanitizer: detected memory leaks
#
# Direct leak of 4194304 byte(s) in 1 object(s) allocated from:
#     #0 0x... in malloc
#     #1 0x... in load_config config.c:42   <- the malloc with no free
#     #2 0x... in main main.c:11
# SUMMARY: AddressSanitizer: 4194304 byte(s) leaked in 1 allocation(s).
```

That allocation stack — `config.c:42` — is the leak's birthplace, and for a program with a clean shutdown that is often all you need. But the LeakSanitizer model only reports at exit, and many leaks live in long-running servers that never cleanly exit. For those, and for measuring *how* memory grows over time, the right tool is Valgrind's **massif** heap profiler plus **ms_print**:

```bash
# massif samples the heap over time and records, at each peak, WHICH
# call stacks are responsible for the live bytes.
valgrind --tool=massif --time-unit=ms ./server
# produces massif.out.<pid>

ms_print massif.out.31337
# Renders an ASCII graph of heap size over time PLUS, at each snapshot,
# a tree of the call stacks holding the bytes:
#   99.20% (4,194,304B) load_config (config.c:42)
# If that percentage CLIMBS across snapshots taken over a run, that
# call stack is your leak -- the C equivalent of "the type that grew."
```

`massif` gives you the same diff-over-time signal as the managed profilers: a call stack whose share of the heap climbs across snapshots is the leak, and `ms_print` shows the climb as a graph and names the stack. For production C/C++ services, **jemalloc's and tcmalloc's built-in heap profilers** do the same with far less overhead than Valgrind (Valgrind slows execution 10–50x and is unusable under real load). Set `MALLOC_CONF="prof:true,lg_prof_sample:19"` for jemalloc, dump two profiles minutes or hours apart, and diff them with `jeprof --base=heap.0.heap heap.1.heap ./binary` — the same `-base` diff as Go's pprof, because Go's pprof heap format is literally descended from Google's tcmalloc profiler. The pattern is universal: *snapshot, wait, snapshot, diff, read the call site that grew.*

The C world also gives you the other unmanaged memory bug that is *not* a leak but is constantly confused with one: the **use-after-free** and the heap **buffer overflow**, where you touch memory you already freed or run off the end of a block. Those corrupt memory rather than leak it, and they are a different hunt — ASan catches them too, but the technique is poisoned-redzone detection, not allocation tracking. I treat corruption as its own beast in the sibling post on use-after-free and memory corruption (planned slug `use-after-free-and-memory-corruption` in this series); if your symptom is a crash with a garbage pointer rather than a climbing RSS, that is where to go. Here we stay on the climbing curve.

## 9. Method in the JVM: heap dump, dominator tree, path to GC roots

The JVM has perhaps the most mature leak-analysis tooling of any runtime, built around the heap dump and **Eclipse MAT** (Memory Analyzer Tool). The workflow is the managed-leak workflow in its most refined form: capture the live heap as a file, let MAT compute the dominator tree, find the object that retains the most memory, and read its path to GC roots — the retainer chain that, cut, frees the leak.

```bash
# Capture a live heap dump from a running JVM. live=true forces a GC
# first so you dump only reachable objects -- the leak, not garbage.
jmap -dump:live,format=b,file=heap_t1.hprof <PID>

# Better in modern JDKs (jmap is deprecated for this):
jcmd <PID> GC.heap_dump -all=false heap_t1.hprof

# Capture a SECOND dump an hour later under steady load:
jcmd <PID> GC.heap_dump -all=false heap_t2.hprof
```

You open both in Eclipse MAT. The single most valuable view is the **dominator tree**: it sorts objects by *retained size* — the total memory that would be freed if that object were collected, i.e. the object plus everything only it keeps alive. A leak shows up as one object (or one class's instances) with an enormous, growing retained size. Right-click it and choose **Path to GC Roots → exclude weak/soft references**, and MAT shows you the exact chain of strong references from a root down to the leaked object. That chain is the bug. The two classic JVM patterns it exposes:

- A **`HashMap` (or any collection) growing without bound**, retained by a `static` field, which is a GC root by definition. The path is `static field → HashMap → Node[] → your entries`. Static collections are the #1 JVM leak, and MAT names the static field directly.
- A **`ThreadLocal` never cleared in a pooled thread**. The path runs `Thread → threadLocals → ThreadLocalMap → Entry → your value`. Because the pool reuses the thread across thousands of requests, the value set in one request and never `remove()`d survives forever. MAT's path-to-roots ends at a live `Thread` object, which is the tell.

MAT can also diff two dumps directly — it loads both and computes the delta in instance counts per class, the JVM version of `objgraph.show_growth()`. The class whose instance count climbed between `heap_t1` and `heap_t2` under steady load is your leaking type, and its dominator path is your fix. The before/after proof on the JVM is the same as everywhere else: after cutting the reference (clear the `ThreadLocal` in a `finally`, bound the cache, drop the static collection), you watch the old-generation occupancy after each full GC stop climbing and settle into a flat line — the post-GC live heap, which is the JVM's honest measure of retained memory, stops growing.

A subtlety unique to the JVM that trips people up: because the JVM grabs heap from the OS up to `-Xmx` and is reluctant to give it back, **RSS can look flat-high even after you fix a leak**, because the JVM is holding the pages it once needed. That is the fragmentation/bloat case from section 3 wearing a JVM costume. To see whether you actually fixed the leak, do not watch OS RSS — watch *post-full-GC live heap* via `jstat -gcutil <PID> 1000` (the `OU`, old-gen utilization, column right after a full GC) or `GC.heap_info`. If post-GC live heap is flat, the leak is fixed even if RSS stays high; the high RSS is just the JVM hoarding pages, addressable separately with `-XX:+ShrinkHeapInSteps`, `-XX:MaxHeapFreeRatio`, or a switch to a more page-returning collector like ZGC or Shenandoah. Always measure the leak with the runtime's live-heap number, not the OS's RSS, or you will think a real fix failed.

## 10. War story: real leaks that took down real systems

Memory leaks are not academic; they have melted production at famous companies, and the postmortems teach the patterns better than any toy example. Let me walk through three, accurately, and flag where I am generalizing a pattern rather than citing a specific documented incident.

**The unbounded cache that ate the heap.** The most common production leak in the wild — the one this post's worked examples are drawn from — is an in-memory cache with no eviction policy bolted onto a service "for now." It is invisible in testing because tests use a handful of distinct keys; it is invisible in staging because staging traffic is low-cardinality; and it detonates in production because production has high key cardinality (every user id, every request id, every session) and runs for days. The growth rate is exactly *distinct keys per unit time times entry size*, which is why these leaks have such clean linear slopes. The fix is always the same — bound the cache (LRU/LFU with a max size, or a TTL with active eviction) or make it hold its values weakly. The lesson encoded in countless postmortems: **a cache without an eviction policy is not a cache, it is a memory leak with a hit-rate metric.**

**The ThreadLocal that survived its request.** A documented class of Java leak, common enough that it is in the official `ThreadLocal` javadoc warnings, is the thread-pool ThreadLocal. A framework or your own code stashes per-request state in a `ThreadLocal` for convenience, the request finishes, but nobody calls `ThreadLocal.remove()`. In a pooled-thread server (every servlet container, every modern framework) the thread is returned to the pool and reused, carrying the stale value — and worse, in app-server hot redeploys the ThreadLocal can pin the *entire old classloader* (because the value's class belongs to it), leaking megabytes of loaded classes per redeploy until PermGen/Metaspace OOMs. This is the famous "redeploy your app five times and the server dies" bug. The fix is disciplined `try { set } finally { remove }` and, at the framework level, request-scoped cleanup that clears all ThreadLocals at the end of every request.

**The slow consumer and the unbounded queue.** A systems-level pattern worth naming: a producer faster than its consumer, with an unbounded buffer in between, is a memory leak by construction. The queue grows without limit because items arrive faster than they leave, and RSS climbs until OOM — even though no reference is "forgotten," every queued item is legitimately referenced by the queue. This is the back-pressure problem, and it is why every robust queue is *bounded* and blocks or sheds load when full rather than buffering infinitely. When you see RSS climb under sustained high load specifically (not at idle), suspect an unbounded buffer somewhere in the pipeline before you suspect a classic forgotten-reference leak. The fix is back-pressure, not a reference cut — and it is a system-design problem, covered properly in the message-queue and system-design material rather than re-derived here. The pattern recurs in async runtimes too: an unbounded channel, an unbounded `asyncio.Queue`, an `EventEmitter` with listeners arriving faster than work drains.

The thread that connects all three: a leak is not always a *bug* in the "I forgot to free this" sense. Sometimes it is a *missing bound* — a cache without eviction, a ThreadLocal without cleanup, a queue without back-pressure. The reference is held *on purpose*; the bug is the absence of a policy that releases it. That reframing matters because it changes the fix from "find the mistake" to "add the bound," and the bound is usually a one-line config (max size, TTL, capacity) on something that was unbounded.

## 11. Stress-testing the hunt: when the easy method fails

The snapshot-diff technique is powerful, but production is hostile, and you need answers for the cases where the clean method does not apply. Here is how I stress-test a leak investigation against the real world.

**What if it only reproduces under load?** Many leaks are proportional to throughput — they need real traffic to show a measurable slope. Do not try to reproduce on your laptop with three requests; you will wait days for a megabyte. Instead, drive synthetic steady load (replay production request shapes with a load tool, or shadow real traffic to a canary) at production rate, and take your two snapshots an hour apart *during* that load. The whole technique depends on *steady-state* load: the working set must be constant so the diff cancels it. Idle snapshots are useless for leaks because nothing is churning.

**What if you cannot attach a debugger or profiler in prod?** This is the common case for the most painful leaks. The answer is that every technique here has a low-overhead, always-on variant. Go's pprof endpoints are cheap enough to leave on permanently behind a private port. Python's `tracemalloc` can run continuously with a sampling overhead you can tune (or use `py-spy dump` to snapshot a running process with zero code changes and no restart). The JVM can dump heap with `jcmd` against a live PID without a restart, and you can trigger a dump *automatically* on OOM with `-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/dumps` so the 4am crash leaves you a forensic heap dump to open in the morning. Node responds to `SIGUSR1` to enable the inspector on a running process, and `node --heapsnapshot-signal=SIGUSR2` writes a heap snapshot to disk on a signal. **Set up the always-on capture before the incident**, because the worst time to instrument a leak is during the OOM loop at 4am.

**What if it only leaks slowly — a few MB a day?** Slow leaks defeat the one-hour snapshot because the growth is below the noise floor of normal churn. Two moves: widen the window (snapshot a day apart, not an hour) so the accumulation rises above the noise; and *amplify* the suspected operation — if you suspect a specific endpoint leaks, hammer that one endpoint a few hundred thousand times in a loop to compress days of slow leak into minutes of obvious growth, then diff. Amplification is the leak-hunter's most underused trick: a leak of 4 KB per call is invisible at 1 call/second and screaming at 10,000 calls in a tight loop. There is a clean way to confirm a leak is per-call before you even profile: run the suspect operation N times, force a GC, measure live heap; run it 10N times, force a GC, measure again. If live heap scales linearly with N, you have a per-call leak and the slope (bytes per call) tells you both the size of the leaked object and, multiplied by your real call rate, the exact production slope you measured on the dashboard — which is a satisfying cross-check that you are chasing the right thing.

**What if two different leaks are stacked on top of each other?** Real services often have more than one leak, and a diff will show the biggest one first, masking the others. Fix the dominant leak, redeploy, and *re-run the same diff* — the second leak, previously buried, now sits at the top of the list. Leak hunting is iterative for the same reason debugging in general is iterative: removing the loudest signal reveals the next one. Do not assume the first fix was the only fix; soak again and watch the curve, because a curve that climbs slower but still climbs means there is a second leak underneath the one you just killed.

**What if the leak is in a native extension or a third-party library?** A Python or Node process can leak in C code behind a binding (a NumPy buffer, a database driver's native handle, an image library) that `tracemalloc` and the V8 heap snapshot cannot see, because those tools only track allocations the *managed* runtime made. The tell is RSS climbing while the managed-heap diff shows nothing growing — the same flat-heap-high-RSS signature as fragmentation, but here it is a native leak rather than allocator bloat. Distinguish them by checking whether RSS tracks a specific operation (native leak: amplify that operation and RSS climbs) versus tracking thread count or uptime alone (fragmentation). For native leaks behind a binding, you drop down to the unmanaged tools — `valgrind`, ASan on the extension, or the library's own diagnostics — because the managed profiler is blind to memory the managed runtime never allocated.

**What if the diff shows a *type* that grew but you cannot find why it is retained?** This is when allocation-site tools (`tracemalloc`, massif, pprof) fail you, because they tell you where the object was *born*, not what *holds* it — and the holder is the bug. Switch from allocation-tracking to retainer-tracking: `objgraph.show_backrefs` / `gc.get_referrers` in Python, the Retainers panel in DevTools, Path-to-GC-Roots in MAT. The allocation site and the retainer are often different code, and only the retainer is the fix.

**What if it is not a leak at all — RSS is high but the heap is flat?** You ruled this out in section 3, but it is worth re-stress-testing here because it is the most expensive wrong turn. If the runtime's post-GC live heap is flat while OS RSS is high, stop the reference hunt — there is no forgotten reference. It is allocator fragmentation or page retention. Tune the allocator (`MALLOC_ARENA_MAX`, `jemalloc` decay settings, `malloc_trim`), not your code. Spending a day on a heap diff for a fragmentation problem is the single most common way to waste an afternoon on this class of bug, because the diff will show you *nothing growing* and you will not believe it.

#### Worked example: the leak that was really fragmentation

A C++ service running on glibc showed RSS climbing from 1.2 GB to 3.5 GB over a day, looking exactly like a leak. We ran the full Valgrind massif and jemalloc heap-diff workflow and got a baffling result: **nothing was growing.** The live heap was flat at about 1.1 GB across snapshots hours apart. Every allocation had a matching free. By the section-3 rule, flat live heap plus high RSS equals fragmentation, not a leak. The service was massively multithreaded (about 200 worker threads), and glibc's `malloc` creates up to `8 * num_cores` per-thread arenas by default, each of which holds onto freed pages rather than returning them, so the arena count exploded and each arena hoarded partially-free pages. The "fix" touched no application code at all: we set `MALLOC_ARENA_MAX=4` to cap glibc's arena count, which dropped steady-state RSS from 3.5 GB to 1.4 GB, and we added a periodic `malloc_trim(0)` call to nudge glibc to return free top-of-heap pages. RSS went flat at 1.4 GB. Had we trusted the climbing curve over the flat heap diff, we would have spent days hunting a reference that did not exist. The heap diff *was* the proof — its very emptiness told us the truth.

## 12. Preventing the next leak

Finding a leak is satisfying; never shipping the next one is the real win. The prevention playbook follows directly from everything above, because if you know how leaks are born, you know how to make them impossible.

**Bound everything that grows.** Every cache gets a max size or TTL and an eviction policy — no naked `Map` or `HashMap` used as a cache, ever. Every queue and channel gets a capacity and a back-pressure policy. Every collection that accumulates "for debugging" gets a cap or gets deleted. The single rule that prevents the majority of production leaks is: *if it grows with cumulative work, it must have a bound.* Treat an unbounded container in a long-running process as a code-review red flag the way you treat an unparameterized SQL string as an injection red flag.

**Pair every subscription with an unsubscription.** Every `addEventListener` needs a `removeEventListener`; every `EventEmitter.on` needs an `off`; every observer registration needs a deregistration, ideally in the same lifecycle method or in a `finally`/cleanup hook so it cannot be forgotten. Frameworks that give you a cleanup callback (React's effect cleanup, RAII destructors, Go's `defer`, Python context managers) exist precisely to make this automatic — use them. For caches and listener registries that legitimately should not keep their targets alive, reach for weak references (`WeakMap`, `WeakHashMap`, `weakref`) so the GC can reclaim a value the moment nothing else references it.

**Clear thread-local and per-request state at the boundary.** In any pooled-thread or pooled-anything system, state set during a request must be cleared at the end of the request, in a `finally`, no exceptions. The framework's request-scope cleanup is the right home for this.

**Make memory a tested, gated metric.** The most effective prevention is a CI memory soak: run the service under steady synthetic load for a fixed window, measure RSS slope, and *fail the build* if it exceeds a threshold (say, more than a few MB/hour of upward drift after warmup). This catches leaks before they ship, which is worth more than any production tool, because it turns "we found the leak after the 4am page" into "the leak never merged." The soak does not need to be long to be effective — amplification does the heavy lifting. A test that runs the hot endpoints a few hundred thousand times in a couple of minutes, forces a GC, and asserts that post-GC live heap returned to within a small delta of its pre-run baseline will catch the overwhelming majority of per-call leaks deterministically, because a true leak makes that assertion fail every single time while a leak-free build passes it every single time. That determinism is what makes it a *gate* and not just a dashboard: a leak is one of the few production failures you can turn into a reliable red-or-green CI signal, so do it. Pair the gate with `-XX:+HeapDumpOnOutOfMemoryError` (or the equivalent always-on snapshot for your runtime) in production so that if one slips through, the crash hands you a forensic dump instead of just a restart.

**Watch the right number.** Alert on RSS *slope over hours*, not instantaneous RSS, because the slope is what distinguishes a leak from legitimate high memory. A flat 3 GB is fine; a climbing 1 GB is a time bomb. And measure leak fixes by the runtime's post-GC live-heap number, never by OS RSS alone, so allocator page-hoarding does not fool you into thinking a real fix failed. This is exactly the kind of trend-based, slope-aware monitoring the observability post in this series argues for; if you want the metrics-and-traces version of this story, see [Observability for Debugging Prod](/blog/software-development/debugging/observability-for-debugging-prod).

A close relative of the memory leak is the *handle* leak — file descriptors, sockets, database connections that are opened and never closed. The mechanism is the same (a resource acquired without a matching release), the symptom is different (you hit a descriptor or connection-pool limit, not an OOM, often a deadlock under load), and the diagnosis uses different tools (`lsof`, `/proc/<pid>/fd`, pool metrics). I treat that sibling problem in its own post (planned slug `resource-leaks-fds-sockets-and-connections` in this series); the discipline — pair every acquire with a release, prefer a scope that auto-releases — is identical, which is why "leak" generalizes so cleanly across memory and handles.

## How to reach for this (and when not to)

Every tool here has a cost, and matching the tool to the situation is half of being good at this. Here is the decisive version.

**Read the curve before you touch a profiler.** Two minutes of `ps`/`smem` watching tells you leak versus growth versus fragmentation, and that decision routes the entire investigation. Skipping it is how you end up running a heap diff on a fragmentation problem and finding nothing for a day.

**Reach for the snapshot-diff first, always.** In every managed runtime, two snapshots an hour apart under steady load, diffed, is the highest-yield move and it should be your default. It names the growing type with almost no reasoning required. Do not start by reading code; start by reading the diff, then read the code the diff points at.

**Do not run Valgrind under production load.** Valgrind slows execution 10–50x; it is a development and reproduction tool, not a production tool. In production C/C++, use jemalloc/tcmalloc heap profiling (near-zero overhead) or ASan in a canary, never Valgrind on a live high-traffic process.

**Do not attach an intrusive debugger to a latency-sensitive prod process.** A heap dump on a large JVM can pause the process for seconds (it forces a full GC and walks the whole heap); do it on a canary or a drained instance, not the box serving your p99-sensitive traffic, or schedule it during a low-traffic window. Prefer the always-on low-overhead endpoints (pprof, sampling tracemalloc) over stop-the-world dumps when you can.

**Do not hunt a reference when the heap is flat.** If post-GC live heap is flat and RSS is high, it is fragmentation, not a leak. Reach for allocator tuning (`MALLOC_ARENA_MAX`, jemalloc decay, `malloc_trim`), not a heap diff. The heap diff will correctly show you nothing, and you must believe it.

**Do not fix a leak without proving the curve flatlined.** A leak fix is a hypothesis until the RSS curve (or post-GC live heap) goes flat over a soak. Argue from the curve, not from the code, because the only thing that ends the 4am page is a flat line.

## Key takeaways

- A leak means *lost* memory in C/C++ (a block with no live pointer) but *retained* memory in GC'd languages (a reference you forgot). Name your runtime first; the two demand opposite hunts.
- The GC keeps your "garbage" alive because it is *reachable*, not because the GC is broken. An object lives if and only if a reference path reaches it from a root. The fix is always to cut an edge on that path.
- Read the RSS curve before profiling: a sawtooth is healthy, a rise-then-plateau is legitimate growth, and a monotonic climb across GC cycles is a real leak. High RSS with a flat live heap is fragmentation, not a leak.
- The one technique that finds any managed leak: take two heap snapshots an hour apart under steady load, diff them, and the type that grew is your leak. Follow its retainer or dominator path to the reference that holds it.
- Every runtime has this diff: `tracemalloc.compare_to` in Python, `pprof -base` in Go, the three-snapshot method in Node/Chrome, the MAT dominator tree in Java, `massif`/`jeprof --base` in C/C++.
- Most production leaks are missing *bounds*, not forgotten frees: an unbounded cache, an uncleared ThreadLocal, an unbounded queue. Bound everything that grows with cumulative work.
- Prove every fix with the curve, not the code. The leak is fixed when RSS (or post-GC live heap) goes flat over a soak, and not one minute before.
- Prevent the next leak with a CI memory soak that fails the build on upward slope, an eviction policy on every cache, an unsubscribe for every subscribe, and an OOM-triggered heap dump waiting in prod.

## Further reading

- [Stop Guessing: The Scientific Method of Debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post applies to memory.
- [Observability for Debugging Prod](/blog/software-development/debugging/observability-for-debugging-prod) — how to alert on RSS *slope* and capture a forensic heap dump before the 4am page, not during it.
- [Memory Profiling: tracemalloc, memray, and Finding Leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) — a deeper, Python-specific walkthrough of the exact tools used in section 6.
- [The Python Memory Model: Objects, Refcounts, and the Garbage Collector](/blog/software-development/python-performance/python-memory-model-objects-refcounts-and-the-garbage-collector) — the CPython refcount-plus-GC internals underneath every Python leak.
- The companion sibling posts in this series, on use-after-free and memory corruption (the *other* unmanaged memory bug — a crash, not a climb) and on resource leaks of file descriptors, sockets, and connections (the same discipline applied to handles instead of bytes).
- Valgrind documentation — the `massif` heap-profiler and `ms_print` manual for unmanaged heap-over-time analysis.
- The AddressSanitizer / LeakSanitizer wiki — compile-time leak detection with allocation stacks for C and C++.
- Eclipse Memory Analyzer (MAT) documentation — the dominator tree and path-to-GC-roots workflow for JVM heap dumps.
- The Go `pprof` documentation and Brendan Gregg's writing on production profiling — the `-base` heap diff and low-overhead always-on profiling endpoints.

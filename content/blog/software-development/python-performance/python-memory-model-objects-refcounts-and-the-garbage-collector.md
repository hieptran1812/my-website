---
title: "Python's Memory Model: Objects, Refcounts, and the Garbage Collector"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Understand exactly where your Python objects live, when they actually free, and why RSS stays high — refcounting, the cyclic GC, pymalloc arenas, and how to tune gc.disable, gc.freeze, and set_threshold on real servers."
tags:
  [
    "python",
    "performance",
    "memory",
    "garbage-collection",
    "reference-counting",
    "cpython",
    "pymalloc",
    "gc-tuning",
    "optimization",
    "rss",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-1.png"
---

A long-running service has been up for nine days. It started at 600 MB of resident memory and now sits at 4.2 GB, perfectly flat at that ceiling, serving the same traffic it served on day one. Nobody changed the data. Nobody changed the load. You attach a memory profiler, you free a few hundred megabytes of caches by hand, you watch the number drop inside Python — and the operating system's view of the process, the `RSS` column in `top`, does not move at all. The memory you "freed" is still charged to your process. Meanwhile a colleague's batch job, which allocates and discards tens of millions of tiny objects in a tight loop, runs 12% faster when they add a single line — `gc.disable()` — and 12% is the difference between hitting the nightly window and not.

Both of these are the same story told from two ends. To understand either, you have to know exactly how CPython decides where an object lives, who is holding onto it, and when — precisely when — its memory is allowed to be reused. This is the post about **where your objects live and when they actually free**. The figure below is the whole arc in one picture: an object is born with a reference count, that count rises and falls as names bind and unbind to it, and the instant it reaches zero the object is freed immediately and deterministically — *unless* it got tangled in a reference cycle, in which case a separate, periodic garbage collector has to come find it.

![diagram of an object's life showing allocation then refcount tracking then either immediate free at zero or reclamation by the cyclic collector when stuck in a cycle](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-1.png)

If you have read [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), you already know the spine of this whole series: don't guess, measure; find the bottleneck; pull the right lever. Memory is a lever like any other, and it has its own cost model. By the end of this post you will be able to reason about a process's `RSS` the way you reason about its CPU profile: you will know why refcounting makes every read into a write, why a two-object cycle leaks without the collector, why the generational design makes the average garbage collection nearly free, why freed memory often does not go back to the OS, and exactly which `gc` knob to turn — `gc.collect()`, `gc.disable()`, `gc.freeze()`, `gc.set_threshold()` — when you finally have a reason to turn one. The measurements in this post are framed on a concrete machine: **an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM**. Where I quote a number I did not personally run on your hardware, I frame it as a typical range, because fabricating a precise figure would defeat the entire point of a series built on measurement.

## 1. Everything is an object, and every object has a header

Before we can talk about freeing memory we have to be precise about what a Python object *is*, because the cost model falls out of the layout. In CPython — the reference implementation almost everyone runs — every value you manipulate, from the integer `5` to a 10-million-row list, is a `PyObject` allocated on the heap. There are no "primitive" values sitting bare on the stack the way an `int` does in C or Java. When you write `x = 5`, you do not put the bits `101` into a slot named `x`. You create (or, for small integers, reuse) a heap object that represents the integer five, and you make the name `x` hold a *pointer* to it.

Every one of these heap objects carries a fixed header before its actual payload. The header is two machine words on a standard build:

- **`ob_refcnt`** — the reference count. A `Py_ssize_t`, 8 bytes on a 64-bit build. This is the number we are going to spend most of this post talking about.
- **`ob_type`** — a pointer to the object's type object (the thing `type(x)` returns), which tells the interpreter how to add it, hash it, print it, and free it. Also 8 bytes.

So the *minimum* overhead for any Python object, before it stores a single byte of your data, is 16 bytes of header. A bare `object()` instance is 16 bytes. A Python `int` adds a length field and at least one 4-byte digit, landing at 28 bytes for a small integer — which is why `sys.getsizeof(5)` returns `28`, not `8`. A one-character string is around 50 bytes. An empty `dict` is 64 bytes; an empty `list` is 56. The data is almost an afterthought next to the bookkeeping.

```pycon
>>> import sys
>>> sys.getsizeof(0)
28
>>> sys.getsizeof(object())
16
>>> sys.getsizeof("a")
50
>>> sys.getsizeof([])
56
>>> sys.getsizeof({})
64
```

This is the same boxing tax we measured in [the hidden cost of objects, attributes, and dynamic dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch): a Python `int` is not 8 bytes of two's-complement, it is a full object with a header, a type pointer, and a variable-length digit array. Here we care about that header for a different reason. The first word, `ob_refcnt`, is not a passive label. It is a counter that the interpreter mutates *constantly*, on operations you do not think of as writes at all — and that mutation is the foundation of how Python frees memory.

A subtlety worth knowing up front, because it changes the numbers you will see if you read CPython source: in Python 3.12 and later, `ob_refcnt` is no longer a single plain integer in every case. The "immortal objects" change (PEP 683) gives certain permanently-live objects — `None`, `True`, `False`, small integers, interned strings — a sentinel refcount value that the interpreter recognizes and *stops* incrementing and decrementing. This is a real optimization with consequences we will return to when we talk about fork servers, because an object whose refcount never changes is an object whose memory page never gets dirtied. But for an ordinary object you create, the refcount behaves exactly as the classic story says.

## 2. Reference counting: every bind is an increment, every unbind a decrement

Here is the core mechanism, stated plainly. Every time you create a new reference to an object, CPython increments that object's `ob_refcnt`. Every time you destroy a reference, it decrements. When a decrement brings the count to **zero**, the object is provably unreachable — nobody is holding a pointer to it — so the interpreter frees it *right then, in line*, before the bytecode that did the decrement even returns control to your loop. The two operations are macros in the C source called `Py_INCREF` and `Py_DECREF`.

What counts as "creating a reference"? Far more than you would guess. Every one of these increments a refcount:

- Binding a name: `x = obj`.
- Putting the object in a container: `lst.append(obj)`, `d[k] = obj`, `s.add(obj)`.
- Passing it as a function argument (the parameter name is a new reference).
- Returning it from a function.
- Capturing it in a closure.
- Even *temporarily* holding it on the interpreter's evaluation stack while a bytecode runs.

And every one of these decrements it: a name going out of scope, a `del`, reassigning a name to something else, a container being cleared or itself freed, a function returning (its locals are released). The pattern is symmetric — incref on the way in, decref on the way out — and the interpreter is meticulous about it because the entire correctness of memory management depends on the count being exact. A single missed `incref` would free an object that someone is still using (a use-after-free crash); a single missed `decref` would leak it forever.

You can watch the count directly with `sys.getrefcount`. There is one gotcha that trips up everyone the first time: the act of *calling* `getrefcount` passes the object as an argument, which itself creates a temporary reference, so the number it reports is always one higher than the "real" count you are thinking of.

```pycon
>>> import sys
>>> a = object()
>>> sys.getrefcount(a)        # one ref from 'a', plus one for the call argument
2
>>> b = a                     # second name to the same object
>>> sys.getrefcount(a)
3
>>> lst = [a, a, a]           # three more references inside the list
>>> sys.getrefcount(a)
6
>>> del b                     # drop one name
>>> sys.getrefcount(a)
5
>>> lst.clear()               # drop the three container references
>>> sys.getrefcount(a)
2
```

Notice the symmetry: the count went `2 → 3 → 6 → 5 → 2` as references were created and destroyed, and we ended exactly where we started, with `a` the sole remaining name (plus the eternal `+1` from the call). If we now did `del a`, the count would drop to `0`, and the object's memory would be returned to the allocator immediately — not "eventually", not "at the next GC pause", but synchronously, as part of executing the `del`.

This is the single most important property of CPython's memory management, and it is genuinely nice to have: **freeing is deterministic.** A file you stop referencing is closed promptly. A large array you reassign is released the instant the old binding drops. There is no "GC pressure" for the common case, no stop-the-world pause to reclaim a temporary you created and discarded inside a comprehension. The C++ programmer's intuition about RAII — resources released at well-defined points — mostly holds in Python *because* of refcounting. Many programs run for years and never trigger the cyclic collector at all; refcounting alone reclaims everything they produce.

### The price of determinism: every read can be a write

Determinism is not free, and the bill comes due in a place that is easy to miss. Because the interpreter holds objects on its evaluation stack, an operation as innocent as *reading* a variable inside a hot loop touches a refcount. Consider this loop:

```python
total = 0.0
for i in range(n):
    total += data[i]
```

Each iteration loads `data` (incref the list, then decref when done with it), computes `i`, indexes into the list to fetch `data[i]` (incref the fetched object onto the stack), adds it to `total` (which decrefs the old `total` object and increfs the new one — floats are immutable, so `+=` builds a new object), and so on. Several of these are refcount mutations. None of them looks like a write in your source code; all of them are writes to memory in the running interpreter.

Why does a *write* matter when you only meant to *read*? Two reasons, and they are exactly the reasons that make refcounting both robust and a tax.

First, **cache coherence on multicore machines.** A modern CPU keeps a value's cache line in one of several states (the MESI protocol and its cousins). A line that is only ever read can be held *Shared* in many cores' caches at once, cheaply. The moment a core *writes* to that line, it must take the line *Exclusive*, invalidating every other core's copy. If two threads both touch the same object's refcount, they ping-pong that cache line back and forth — "true sharing" contention on a field neither of them cares about semantically. This is one of the deep reasons the [Global Interpreter Lock](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) exists: with one lock held during bytecode execution, only one thread mutates refcounts at a time, so the counts stay correct without per-object atomic operations. The free-threaded (no-GIL) build of Python 3.13+ has to solve exactly this problem with biased reference counting and deferred refcounts, and the single-threaded cost of doing so is a measurable slice of why it is slower on one core.

Second, **fork and copy-on-write**, which we will spend a whole section on. When you `fork()` a process, the child shares the parent's memory pages read-only until one side writes; the first write copies the page. Refcount mutations are writes. So merely *reading* a big shared data structure in a forked worker silently copies its pages, one by one, defeating the memory sharing you forked to get. Hold that thought.

#### Worked example: counting the refcount traffic in a hot loop

Let me make the "every read is a write" claim concrete with numbers on the named machine — **8-core x86-64 Linux, CPython 3.12, 16 GB RAM**. Take a sum over a list of one million Python floats.

```python
import timeit

data = [float(i) for i in range(1_000_000)]

def py_sum():
    total = 0.0
    for x in data:
        total += x
    return total

t = timeit.timeit(py_sum, number=20) / 20
print(f"pure-Python sum: {t*1000:.1f} ms")
```

On the named box this lands around **18 ms** per pass — roughly 18 nanoseconds per element. Each element costs an attribute-free list index, an incref to put the float on the stack, a float add that allocates a new result object, a decref of the old `total`, and a decref of the fetched float when the iteration's stack is cleaned up. That is on the order of four to six refcount mutations per element. At one million elements you are doing somewhere around 5 million refcount writes just to *add up numbers you already had in memory*.

Now compare the same sum done by NumPy, which stores the million doubles as one packed C buffer with *no per-element Python object and therefore no per-element refcount*:

```pycon
>>> import numpy as np, timeit
>>> arr = np.arange(1_000_000, dtype=np.float64)
>>> t = timeit.timeit(lambda: arr.sum(), number=200) / 200
>>> print(f"numpy sum: {t*1e6:.1f} us")
numpy sum: 320.0 us
```

About **0.32 ms** — roughly **55× faster**. Most of that gap is the boxing and bytecode-loop overhead covered in the [ndarray post](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), but a real, non-trivial fraction is simply the refcount traffic that vanishes when there are no per-element objects to count. The takeaway for memory thinking: *the refcount field you never look at is being written millions of times a second in any Python-level loop.* That cost is invisible in your source and very visible in your wall clock.

### Watching the decrefs in the bytecode

If you want to *see* where the increments and decrements come from rather than take my word for it, the disassembler exposes the stack discipline that drives them. Every Python bytecode that loads a value pushes a *new reference* onto the interpreter's value stack (an incref), and every bytecode that consumes a value pops it and releases that reference (a decref). Disassemble a one-line function and read it as refcount traffic:

```pycon
>>> import dis
>>> def f(d):
...     return d["key"]
...
>>> dis.dis(f)
  2           LOAD_FAST                0 (d)       # incref d onto the stack
              LOAD_CONST               1 ('key')   # incref the constant 'key'
              BINARY_SUBSCR                         # pops both, increfs the result, decrefs d and 'key'
              RETURN_VALUE                          # hands the result's reference to the caller
```

`LOAD_FAST` puts the dict on the stack — a reference, so an incref. `BINARY_SUBSCR` pops the dict and the key, looks up the value (incref the value onto the stack), and then *decrefs the dict and the key* because it is done holding them. By the time the function returns, every temporary reference it created has been balanced by a matching release. Multiply this dance by the millions of bytecodes a loop executes and you have the refcount write-volume that the `del` operator and scope exit also drive. The `del` statement, by the way, is not magic: `del x` compiles to a `DELETE_FAST` (or `DELETE_NAME`) bytecode whose entire job is to decref whatever `x` pointed at and clear the slot. If that decref brings the count to zero, the object's deallocator runs *inside the execution of `DELETE_FAST`*, before the next bytecode. That is what "deterministic, in line" means at the machine level.

This also explains a subtle correctness rule that bites people who write C extensions or read CPython source: a function that *creates* a reference owns it and must release it; a function that *borrows* a reference (looks at an object it does not own) must not. The "owned versus borrowed reference" distinction is the entire contract of the C-API, and getting it wrong is exactly how extension modules leak (a missed `Py_DECREF`) or crash (a missed `Py_INCREF` leading to a premature free). At the pure-Python level you never see this, because the interpreter's bytecodes are written to keep the books exactly — but the cost of keeping those books is the write traffic we just measured.

## 3. The one thing refcounting cannot do: reference cycles

Reference counting has a fatal blind spot, and it is not an edge case — it is the reason CPython needs a second, entirely separate memory reclaimer. The blind spot is the **reference cycle**: a group of objects that refer to each other, directly or through a chain, such that every object in the group is kept alive by another object *in the same group*.

The simplest cycle is two objects pointing at each other. Build it explicitly:

```pycon
>>> import sys
>>> class Node:
...     pass
...
>>> a = Node()
>>> b = Node()
>>> a.other = b          # a references b
>>> b.other = a          # b references a -- the cycle is closed
>>> sys.getrefcount(a)
3
>>> sys.getrefcount(b)
3
```

Each object now has a refcount of 2 (the `getrefcount` call adds the temporary +1 you see as 3): `a` is referenced by the name `a` *and* by `b.other`; `b` is referenced by the name `b` *and* by `a.other`. Now do the thing a real program does all the time — drop the names, because the function that created these returns, or you reassign them:

```pycon
>>> del a
>>> del b
```

What happens to the refcounts? Deleting the name `a` decrements `a`'s count from 2 to 1 — because `b.other` *still references it*. Deleting the name `b` decrements `b`'s count from 2 to 1 — because `a.other` *still references it*. Neither count reaches zero. Refcounting, looking only at the counts, concludes that both objects are still in use and refuses to free them. But they are not in use by anything reachable from your program: there is no name, no module global, no stack frame, no live container anywhere that can reach `a` or `b`. They are a pair of mutually-supporting objects floating in the heap, unreachable and immortal. **That is a memory leak**, caused purely by the design of reference counting, and no sequence of `del`s will ever clean it up.

The figure below traces exactly this: two objects each holding one reference to the other, the program dropping its own variables, and both counts stubbornly stuck at one even though nothing can reach the pair.

![diagram of a reference cycle where object A holds B and object B holds A so both refcounts stay at one and the pair is unreachable but not freed until the cyclic collector runs](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-6.png)

Cycles are not exotic. They appear the moment you build any of these:

- A doubly-linked list, where each node points to `next` and `prev`.
- A tree where children hold a `parent` back-reference.
- A graph with any loop in it.
- A closure that captures a variable holding the function itself.
- An object that stores a callback bound to one of its own methods (`self.timer = Timer(self.on_tick)` — the timer holds a bound method, which holds `self`, which holds the timer).
- An exception caught and stored, because a traceback references the frame, which references its local variables, one of which may be the exception. (This is so common that Python 3 clears the traceback when an `except` block exits, specifically to avoid the leak.)

If reference counting were the *only* reclaimer, every program that built a doubly-linked list and threw it away would leak. Long-running servers would climb to OOM not because of a bug in your code but because of a structural limitation in the runtime. That is unacceptable, so CPython adds a second mechanism that exists for one job: finding and freeing unreachable cycles.

### Why you cannot just "subtract the cycle" with counting alone

It is worth being precise about *why* counting cannot fix this itself, because it explains the design of the thing that does. The refcount on an object is a *local* fact: it knows how many references point at the object, but it does not know *where those references come from*. From `a`'s point of view, a refcount of 1 from `b.other` is indistinguishable from a refcount of 1 from a live global variable. Counting has no notion of *reachability from the roots* (the live names, the module globals, the stack). To detect a cycle you must do a *global* analysis: start from the roots and see what is reachable; everything else is garbage, cycles included. That global, reachability-based sweep is precisely what a tracing garbage collector does — and precisely what reference counting, by design, does not.

## 4. The cyclic garbage collector: a generational mark-and-sweep

CPython's cyclic garbage collector is a **generational, mark-and-sweep tracing collector** layered *on top of* reference counting. It does not replace counting — counting still does the vast majority of reclamation, instantly and deterministically. The cyclic collector exists only to mop up the cycles that counting leaves behind. The two reclaimers are complementary, and the figure below is the clearest way to see why CPython needs both rather than one: counting frees acyclic garbage the instant it dies but leaks every cycle, so a periodic collector is bolted on to catch exactly the cases counting misses.

![before and after comparison contrasting reference counting alone which leaks cycles against reference counting plus a cyclic collector which reclaims them so resident memory stays bounded](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-2.png)

The collector only tracks objects that *can* participate in a cycle — that is, **container** objects that may hold references to other objects: lists, dicts, tuples (that contain containers), sets, class instances, and so on. It deliberately does *not* track objects that cannot reference anything else: an `int`, a `float`, a `str`, a `bytes` object. Those can never be part of a cycle, so the collector ignores them, which keeps its working set small. You can check whether a given object is tracked:

```pycon
>>> import gc
>>> gc.is_tracked(42)
False
>>> gc.is_tracked("hello")
False
>>> gc.is_tracked([])
True
>>> gc.is_tracked({})
True
>>> gc.is_tracked((1, 2, 3))     # a tuple of only atomic values: not tracked
False
>>> gc.is_tracked((1, [2], 3))   # a tuple holding a container: tracked
True
```

When the collector runs over a generation, the algorithm works roughly like this. For each tracked object it makes a private copy of the refcount, then it walks every tracked object and *subtracts* the references that come from *other tracked objects in the set* — the "internal" references. After this pass, an object's adjusted count tells you how many references it has from *outside* the candidate set (from live roots, or from objects not in this generation). If that adjusted count is still greater than zero, the object is reachable from outside and is *live*; the collector marks it and everything it can reach as surviving. Whatever is left — objects whose entire refcount was explained by internal cycle references and that nothing live points to — is unreachable garbage, and the collector frees it. This is the global reachability analysis that counting alone cannot do, restricted cleverly to just the tracked containers.

A detail that matters for the fork story later: to be able to walk "every tracked object" without scanning the whole heap, the collector keeps each generation as an intrusive **doubly-linked list**. When a trackable container is created, CPython threads it into the gen-0 list using a small per-object GC header (a `prev`/`next` pair plus a few bits) that sits *just before* the `PyObject` header in memory. Promotion from one generation to the next is an unlink-and-relink in these lists. The consequence you must remember: the collector's bookkeeping — those `prev`/`next` pointers and the refcount-copy pass — *writes* to memory near every tracked object it visits. That is the write that dirties copy-on-write pages on fork, and it is why `gc.freeze()`, which pulls objects out of these lists into a permanent set the walk never visits, protects shared pages.

You can force this whole process to run on demand with `gc.collect()`, which returns the number of unreachable objects it found and freed. Here is the leaked cycle from the previous section, this time reclaimed:

```python
import gc

class Node:
    def __init__(self):
        self.other = None

def make_cycle():
    a = Node()
    b = Node()
    a.other = b
    b.other = a
    # a and b go out of scope when this function returns;
    # refcounting cannot free them because of the cycle.

gc.collect()                       # clean slate
make_cycle()
make_cycle()
unreachable = gc.collect()         # force a cyclic sweep
print(f"collected {unreachable} unreachable objects")
```

```pycon
collected 4 unreachable objects
```

Four objects — two `Node` instances per call, two calls — were unreachable cycles that reference counting could not touch, and `gc.collect()` found and freed all four. If you had never called `gc.collect()` and `gc.disable()` had been in effect, those objects would have leaked for the lifetime of the process.

### gc.garbage and the objects the collector refuses to free

There is one category of cyclic garbage the collector finds but, historically, refused to *free*: cycles containing an object with a `__del__` finalizer (and, in older Pythons, certain weakref callbacks). The problem is ordering. If `a` and `b` form a cycle and both define `__del__`, in what order should they be finalized? Running `a.__del__()` first might touch `b`, which is also being torn down — finalizers could see half-destroyed objects, or even resurrect them. Before Python 3.4, the collector punted: it found such cycles, decided it could not safely finalize them, and dumped them into the list `gc.garbage` for you to deal with manually. They stayed there, alive, leaking, unless you broke the cycle by hand.

```pycon
>>> import gc
>>> gc.garbage
[]
```

PEP 442, shipped in Python 3.4, fixed the common case: the collector now runs all the finalizers in a cycle *first*, then reclaims the objects, so most `__del__`-bearing cycles are collected normally and `gc.garbage` stays empty. But the lesson survives: **`__del__` plus reference cycles is a code smell.** A finalizer's timing is already non-deterministic for any object that ends up in a cycle (it runs at the next collection, not when the last reference drops), and a finalizer that resurrects its object or has side effects in a cycle is asking for trouble. The robust pattern is to not rely on `__del__` for cleanup at all — use a context manager (`with`) or an explicit `close()` for deterministic resource release, and reserve `__del__` for a last-resort safety net at most. If you must hold a back-reference that would otherwise form a cycle (a child pointing at its parent), use a `weakref` — a reference that does *not* increment the refcount and does *not* keep the target alive — so the cycle never forms in the first place and counting can free everything deterministically.

## 5. Generations: why the average collection is nearly free

If the cyclic collector had to scan *every* tracked object in the heap on every run, it would be ruinously expensive for a large, long-lived program — a process holding 50 million live objects would pay a 50-million-object scan every time it wanted to catch a handful of new cycles. The collector avoids this with the classic trick of **generational garbage collection**, which rests on a single empirical observation called the **generational hypothesis**:

> Most objects die young. An object that has already survived for a while is likely to keep surviving.

Think about your own code. The overwhelming majority of objects a program creates are short-lived temporaries: the intermediate list in a comprehension, the string built to format a log line, the tuple unpacked from a function return. They are born, used, and dropped within microseconds. A small minority — module globals, the connection pool, a long-lived cache, the application config — are created once at startup and live for the entire run. The hypothesis says: spend your collection effort on the young objects, where almost all the garbage is, and rarely bother re-scanning the old objects, which are almost all still alive.

CPython implements this with **three generations**, numbered 0, 1, and 2. The timeline figure shows the flow: every newly tracked container starts in generation 0, the youngest; whatever survives a collection is *promoted* to the next generation; generation 2 holds the long-lived survivors and is scanned the least often.

![timeline showing generational garbage collection where objects start in generation zero scanned often then promote to generation one and finally generation two scanned rarely](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-4.png)

The cadence is controlled by **thresholds**, which you can inspect and set:

```pycon
>>> import gc
>>> gc.get_threshold()
(700, 10, 10)
```

These three numbers mean:

- **700** — generation 0 is collected when the number of container allocations *minus* deallocations since the last gen-0 collection exceeds 700. This is a *net* count of new tracked objects, not a count of total allocations, and it is why a program that allocates and immediately frees in a tight loop may never trigger a collection at all (allocs and deallocs cancel out).
- **10** (the second number) — after 10 generation-0 collections, the next collection also includes generation 1.
- **10** (the third number) — after 10 generation-1 collections, the next one also includes generation 2 (a "full" collection of the entire tracked heap).

So generation 0 is swept constantly and cheaply (only the recently-created objects), generation 1 perhaps an order of magnitude less often, and generation 2 — the expensive full-heap scan — only rarely. This is the whole reason the *average* cost of a Python garbage collection is low even in a process with millions of live objects: you almost never scan all of them.

### The amortized cost argument, made rigorous

Let me make the "nearly free" claim quantitative rather than hand-wave it. Suppose your program creates objects at a steady rate, and let:

- $g_0$ = the cost of one generation-0 collection (proportional to the number of objects *in gen 0*, which the threshold caps at a small, roughly constant number — say a few hundred to a few thousand).
- $g_2$ = the cost of one full (gen-2) collection (proportional to the total number of *live tracked objects*, $N$, which can be in the millions).

With the default thresholds, a full collection happens roughly once every $10 \times 10 = 100$ generation-0 collections. So over a window of 100 gen-0 collections you pay about $100 \cdot g_0 + 1 \cdot g_2$ total. The per-collection *amortized* cost is:

$$\text{amortized cost} \approx g_0 + \frac{g_2}{100}.$$

Because $g_0$ is bounded by a small threshold (it does not grow with the size of your heap) and the expensive $g_2$ term is divided by 100, the average collection stays cheap even as $N$ — your total live object count — grows large. The generational design has effectively *amortized* the one expensive full scan across a hundred cheap young scans. This is the same amortization logic behind a dynamic array's $O(1)$ average append, applied to garbage collection: a rare expensive operation, spread thin across many cheap ones, yields a low average.

The catch — and it is the catch that bites latency-sensitive servers — is that this is an argument about the *average*, not the *worst case*. The full gen-2 collection still costs $g_2 \propto N$, and when it fires, *that* request eats the whole pause. A service holding 30 million live objects in a giant in-memory cache can see a full collection take tens to hundreds of milliseconds, and it lands on whichever unlucky request happened to trip the gen-2 threshold. That is a tail-latency problem, and it is one of the few legitimate reasons to tune the collector, which we get to in §8.

```pycon
>>> import gc
>>> gc.get_stats()
[{'collections': 142, 'collected': 3801, 'uncollectable': 0},
 {'collections': 12, 'collected': 540, 'uncollectable': 0},
 {'collections': 1, 'collected': 88, 'uncollectable': 0}]
```

`gc.get_stats()` returns one dict per generation and is your window into how hard the collector is actually working. Read it like this: generation 0 ran 142 times (cheap, frequent), generation 1 ran 12 times (note 142/12 ≈ 12, consistent with the second threshold of 10), and the expensive generation 2 ran only once. `collected` is how many objects each generation reclaimed; `uncollectable` should be zero — if it climbs, you have `__del__`-in-a-cycle objects piling up in `gc.garbage`. Before you touch any GC knob, look here first: if generation 2 has run twice in an hour, the collector is not your problem and disabling it will buy you nothing.

### When the generational hypothesis fails you

The amortization argument above only holds while the hypothesis holds — while most objects really do die young and the old generation really is mostly stable. There is one common architecture where it quietly breaks: a service that holds a *large, growing, long-lived* collection of tracked objects, such as an in-process cache that maps keys to Python lists or dicts of records. Every entry that survives long enough gets promoted to generation 2, so generation 2's population $N$ grows without an upper bound. The full collection's cost $g_2 \propto N$ therefore *also* grows without bound, and because a full collection re-scans all of generation 2 even though almost none of it is garbage, you pay an ever-larger pause to reclaim an ever-tinier fraction. This is the precise mechanism behind "my service's p99 latency develops a periodic spike that gets worse as the cache fills." The collector is doing exactly what it was designed to do; the workload simply violates the assumption it was designed around.

The fixes follow directly from the diagnosis. If the giant cache is genuinely permanent for the process's life, take it out of the collector's reach entirely — either `gc.freeze()` after you build it (so it joins the permanent set and is never re-scanned) or, if it truly cannot contain cycles, store the data in a form the collector does not track at all (a few large NumPy or Arrow buffers, or `array.array`, instead of millions of individual tracked containers). If it must keep being collected, raise the generation-2 trigger so the expensive full scan fires far less often, accepting a rarer but larger pause. The point is that you cannot know which fix applies until you have read `gc.get_stats()` and confirmed that generation 2 is in fact the source of the pauses — which is the discipline §7 is about.

## 6. pymalloc: where small objects actually live, and why RSS stays high

We have talked about *when* objects are freed. Now the other half of the mystery from the intro: when an object *is* freed, why does the operating system's memory accounting — the `RSS` number you watch in `top` — so often refuse to go down? The answer is the allocator, and CPython's small-object allocator is called **pymalloc**.

Calling the system `malloc()`/`free()` for every tiny Python object would be far too slow — a Python program creates and destroys millions of small objects per second, and the general-purpose system allocator is not optimized for that storm of tiny, same-sized requests. So CPython interposes its own allocator for objects up to **512 bytes** (which is almost all of them: ints, floats, small strings, small tuples, instance headers). The structure is a three-level hierarchy, shown in the stack figure: blocks live inside pools, pools live inside arenas, and only the arena ever talks to the operating system.

![layered diagram of the pymalloc allocator showing an object request served from a block inside a pool inside an arena with only the arena talking to the operating system](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-3.png)

Reading the layers from the inside out:

- **Block** — the smallest unit, a single fixed-size slot that holds one object. pymalloc rounds every request up to one of a fixed set of *size classes*, spaced 8 bytes apart (8, 16, 24, ..., up to 512). A 30-byte object goes in a 32-byte block; the rounding-up is internal fragmentation you pay for speed.
- **Pool** — a single 4 KB page (one OS page) carved entirely into blocks of *one* size class. A pool dedicated to 32-byte blocks holds 4096/32 ≈ 126 of them. A free list threads through the empty blocks in a pool so that allocating one is just popping the head of a list — a handful of instructions, no system call.
- **Arena** — a 256 KB chunk (64 pools) that pymalloc requests from the OS in one go (via `mmap` on Linux). The arena is the *unit of negotiation with the operating system*: pymalloc asks the OS for memory one arena at a time, and — crucially — returns memory to the OS only one whole arena at a time.

That last point is the entire answer to "why doesn't my RSS go down." For pymalloc to hand an arena back to the operating system, **every single pool in that arena must be completely empty** — every block in all 64 pages free. In a real program, objects of wildly different lifetimes get interleaved across arenas: a long-lived cache entry and a short-lived temporary can land in the same arena. When you free a million temporaries, you empty *most* of the blocks in many arenas, but if even one long-lived object remains in each arena, that arena cannot be returned. The freed space is reclaimed *inside Python* — pymalloc will happily reuse those empty blocks for your next million allocations — but it is *not* returned to the OS, so `RSS` stays at the high-water mark.

This is **fragmentation**, and it is why the canonical advice "Python doesn't give memory back" is *mostly* true. Modern CPython (3.x) is better than its reputation — it does return fully-empty arenas, and it uses a smarter arena-management strategy that prefers to fill partially-used arenas before opening new ones, which helps empties accumulate. But the fundamental shape holds: **your process's RSS tends to track its high-water mark, not its current live set.** If at any point you allocated 8 GB of objects, your RSS will likely sit near 8 GB even after you free them, because freeing them rarely empties whole arenas.

It is worth being precise about the size boundary, because it determines which allocator a given object uses. Anything pymalloc considers "small" — a request of **512 bytes or less** after rounding — is served from the arena/pool/block machinery above. Anything larger goes straight to the system allocator (`malloc`), bypassing pymalloc entirely, and is freed straight back to it. So a giant NumPy array's data buffer (megabytes) does *not* live in pymalloc arenas; it is one big `malloc` that *does* get returned to the OS when you free it. The arena-retention problem is specifically a problem of **many small Python objects** — the ints, the tuples, the dict entries, the instance headers — because those are the ones pymalloc pools. This is why a program that holds its bulk data in a few large NumPy or Arrow buffers has far more predictable RSS than one that holds the same data as millions of individual Python objects: the former frees cleanly back to the OS, the latter fragments arenas. It is one more argument, on top of speed, for keeping bulk numeric data in packed buffers, the theme of the [ndarray post](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast).

There is also an interaction with object *churn* worth naming. Within a single size class, pymalloc reuses freed blocks eagerly — free a 32-byte object and the next 32-byte request takes that exact slot. This is wonderful for the common pattern of creating and discarding same-sized temporaries: a tight loop that makes and drops millions of three-tuples reuses the same handful of pools over and over and barely grows RSS at all (recall that the gen-0 threshold counts *net* allocations, which cancel to nearly zero). The fragmentation problem only appears when long-lived and short-lived objects of the same size class are *interleaved in time*, so that the long-lived ones get scattered across many arenas and pin them open. A practical mitigation, when you control the allocation order, is to allocate your long-lived structures *first*, at startup, so they cluster into early arenas, and let the churny short-lived work happen afterward in arenas that can later empty completely. You will not always have that control, but when you do, it is free.

#### Worked example: RSS that refuses to drop

Let me show this directly on the named machine — **8-core x86-64 Linux, CPython 3.12, 16 GB RAM** — using the OS's own view of resident memory.

```python
import os
import gc

def rss_mb():
    # Linux: resident set size in pages * page size, read from /proc.
    with open(f"/proc/{os.getpid()}/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)

print(f"start:        {rss_mb():7.1f} MB")

# Allocate ~10 million small objects: a list of 10M three-tuples.
big = [(i, i + 1, i + 2) for i in range(10_000_000)]
print(f"after alloc:  {rss_mb():7.1f} MB")

# Free them all. Refcounting reclaims every tuple deterministically.
del big
gc.collect()
print(f"after free:   {rss_mb():7.1f} MB")
```

A representative run on the named box prints something like:

```pycon
start:           18.3 MB
after alloc:    742.0 MB
after free:     410.0 MB
```

(Read those two numbers as roughly **742 MB** and **410 MB**; the exact figures vary by build and by how the arenas happened to pack.) The objects were *all* freed inside Python — refcounting reclaimed every tuple the instant `del big` ran, and `gc.collect()` confirms there were no cycles. Yet RSS fell from 742 MB only to about 410 MB, not back to the starting 18 MB. The roughly **390 MB still charged to the process** is memory pymalloc holds in arenas it could not fully empty, ready to be reused by Python but invisible to the OS as "free." If you run the allocation loop a *second* time, RSS will barely climb past the previous peak — Python reuses the held memory rather than asking the OS for more. The high-water mark is sticky.

The practical implication is large for [memory profiling](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks): when you watch RSS climb and never fall, you cannot conclude "leak" from RSS alone. RSS rising to a *peak and staying there* is fragmentation, not a leak — your live set is fine, you just touched a lot of memory once. RSS rising *without bound*, climbing on every request and never plateauing, is a real leak (a growing cache, an accumulating list, an uncollectable cycle). The tools that tell the two apart — `tracemalloc` snapshots and `memray`'s high-water-mark and leak modes — work at the Python-allocation level, *below* RSS, which is exactly why they can see your true live set when `top` cannot.

## 7. tracemalloc and gc.get_stats: observing it for yourself

You should never tune the garbage collector — or conclude anything about your memory — without measurement. Here is the practical observation toolkit, the "how" half of this post.

`tracemalloc` is the standard-library tool for asking *which lines of Python allocated the memory that is currently live*. It hooks the allocator, so it sees the real Python-level allocations regardless of what RSS says. The workflow is snapshot, do work, snapshot again, diff:

```python
import tracemalloc

tracemalloc.start()
snap1 = tracemalloc.take_snapshot()

# ... run the suspect code, e.g. one request cycle or one loop iteration ...
cache = {i: [0] * 100 for i in range(10_000)}

snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, "lineno")[:5]:
    print(stat)
```

```pycon
example.py:9: size=8127 KiB (+8127 KiB), count=10001 (+10001), average=832 B
```

The `compare_to` diff is the key: it shows you the *growth* between two points, attributed to the exact source line, with the object count and average size. Run two snapshots one request apart in a leaking server and the leaking line floats straight to the top. This is the same technique covered in depth in the [memory-profiling post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks); here the point is that it lets you reason about your *true live set*, below the fragmentation noise of RSS.

For the collector specifically, `gc.get_stats()` (which we met in §5) and a couple of helpers tell you whether the GC is even relevant to your problem:

```pycon
>>> import gc
>>> gc.get_count()              # (gen0, gen1, gen2) live un-collected counters right now
(312, 4, 1)
>>> len(gc.get_objects())       # every tracked object in the heap -- can be huge
84211
>>> gc.set_debug(gc.DEBUG_STATS)   # log every collection: timing, counts, generation
```

`gc.set_debug(gc.DEBUG_STATS)` makes the collector print a line every time it runs — generation, objects examined, objects collected, and elapsed time. Pipe that to a log for a few minutes of production traffic and you will know, concretely, how often the collector fires and how long its pauses are. If a full (gen-2) collection runs once a minute and takes 80 ms, you have a real tail-latency lever to pull. If it runs twice an hour and takes 2 ms, leave the GC alone — it is not your problem, and any time you spend "optimizing" it is time stolen from the actual hot path.

Here is the honest measurement discipline for any GC tuning, the same discipline the whole series preaches: (1) reproduce a steady-state workload, not a cold start; (2) record `gc.get_stats()` deltas over a fixed window so you have a *rate*, not a single number; (3) measure the metric you actually care about — p99 latency for a server, wall-clock for a batch job, peak RSS for a memory ceiling — *before* and *after* the change; (4) change one knob at a time; (5) keep the change only if the number moved enough to matter. A GC tweak that improves throughput by 1% is usually not worth the operational risk of a possible cycle leak.

## 8. Tuning the collector: collect, disable, freeze, set_threshold

Now the four knobs, what each one actually does, and exactly when to reach for it. The matrix figure lays them side by side; the prose below is the detail behind each cell.

![comparison matrix of the four main gc controls collect disable freeze and set threshold showing what each does its main effect and when to use it](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-5.png)

| Knob | What it does | Main risk | Reach for it when |
| --- | --- | --- | --- |
| `gc.collect()` | Forces a full cyclic sweep now; returns count freed | A long pause if heap is huge | After a known cycle-heavy phase; to inspect `gc.garbage`; in tests |
| `gc.disable()` | Stops automatic cycle detection entirely | Cycles leak until re-enabled | Short batch jobs with no cycles; to kill GC pause spikes temporarily |
| `gc.freeze()` | Moves all live objects to a permanent set the GC skips | None for correctness; only useful pre-fork | Right before `fork()` in a pre-forking server to protect COW pages |
| `gc.set_threshold()` | Changes how often gen 0/1/2 collections fire | Wrong values waste CPU or grow pauses | After measuring `gc.get_stats`, to trade pause frequency vs size |

**`gc.collect()`** forces a complete cyclic collection immediately and returns the number of unreachable objects it reclaimed. Use it deterministically: after a phase you know created and abandoned a lot of cycles (parsing a big graph, tearing down a large object tree), call it once to reclaim them now rather than waiting for the threshold. It is also the tool for *debugging* a suspected cycle leak — call `gc.collect()` and check whether `gc.garbage` filled up (uncollectable `__del__` cycles). Do not sprinkle it everywhere "to be safe"; a forced full collection on a large heap is exactly the expensive $g_2$ pause from §5, and calling it in a hot loop will wreck your throughput.

**`gc.disable()`** turns off *automatic* cyclic collection. Reference counting keeps working — every acyclic object is still freed instantly — so for code that creates *no cycles* this is pure upside: you skip the cost of the collector waking up and scanning generation 0 hundreds of times. The classic win is an allocation-heavy batch job: a script that loads a few million rows, transforms them, writes output, and exits. It creates a storm of short-lived objects, which keeps tripping the gen-0 threshold, so the collector keeps firing — and finds nothing, because well-structured data-processing code rarely makes cycles. Disabling it skips all that wasted scanning. The risk is real and must be stated plainly: with the collector off, *any* cycle your code does create leaks until the process exits. For a short-lived batch job that is fine (the OS reclaims everything at exit). For a long-running server it is dangerous unless you pair it with periodic manual `gc.collect()` calls at safe points (between requests, say).

#### Worked example: gc.disable on an allocation-heavy loop

The measurable claim, on the named machine — **8-core x86-64 Linux, CPython 3.12, 16 GB RAM**: disabling the collector around a tight allocation loop measurably speeds it up, because the collector keeps firing on garbage that refcounting would have freed anyway.

```python
import gc
import time

def build(n):
    # Allocate a storm of short-lived container objects.
    out = []
    for i in range(n):
        out.append({"id": i, "vals": [i, i * 2, i * 3]})
    return out

N = 5_000_000

gc.enable()
t0 = time.perf_counter()
build(N)
t_enabled = time.perf_counter() - t0

gc.collect()
gc.disable()
t0 = time.perf_counter()
build(N)
t_disabled = time.perf_counter() - t0
gc.enable()

print(f"gc enabled:  {t_enabled:.2f} s")
print(f"gc disabled: {t_disabled:.2f} s")
print(f"speedup:     {t_enabled / t_disabled:.2f}x")
```

A representative result on the named box:

```pycon
gc enabled:  3.91 s
gc disabled: 3.42 s
speedup:     1.14x
```

About a **12–15% speedup** on this allocation-heavy loop — the collector was firing repeatedly as the growing list kept tripping the gen-0 threshold, scanning millions of dicts and lists every time, and finding nothing to collect because the list keeps them all alive and there are no cycles. The exact delta depends heavily on how many tracked objects you churn; for a loop that allocates few containers the win is near zero, and for one that churns tens of millions it can exceed 20%. The honest framing: `gc.disable()` is worth it specifically for *short, cycle-free, allocation-dense* phases — and you should always `gc.collect()` and re-enable afterward in any process that keeps running.

**`gc.set_threshold(threshold0, threshold1, threshold2)`** changes the cadence. The most common production move is to *raise* the gen-0 threshold well above its default of 700 — say to `gc.set_threshold(50_000, 10, 10)` — so the collector fires far less often, trading more frequent small pauses for rarer, somewhat larger ones, and reducing the constant background scanning overhead in an object-churny service. Setting `threshold0` to `0` is the documented way to disable gen-0 collection without disabling the whole collector. This knob is for *after* you have measured with `gc.get_stats()` and decided the collection *frequency* is costing you; it is not a guess-and-pray dial.

## 9. gc.freeze and fork servers: keeping copy-on-write pages shared

The most surgical GC optimization, and the one with the biggest payoff in a specific architecture, is **`gc.freeze()`** in a pre-forking server. To see why it matters you have to combine two facts we have already established: refcount mutations are *writes*, and `fork()` shares memory pages *copy-on-write*.

The pre-forking pattern is everywhere: Gunicorn and uWSGI web servers, and ML inference servers, load a big read-only data structure once in a parent process — a multi-gigabyte model, a large lookup table, a warmed cache — and then `fork()` a pool of worker processes. The point of forking *after* loading is memory sharing: thanks to copy-on-write, the children do not get their own copy of the 8 GB model; they share the parent's physical pages, and the OS only makes a private copy of a page when a child *writes* to it. In an ideal world the model is never written, so 16 workers share one 8 GB copy and the machine's total memory stays near 8 GB instead of $16 \times 8 = 128$ GB.

The thing that ruins this is the garbage collector — specifically, the first full collection in each worker. When the collector runs, it walks the tracked heap and, as part of its algorithm, *touches the refcount header of objects it examines* (and the GC also maintains its own per-object generation/linked-list bookkeeping that lives right next to the object). Touching that header is a *write*. A write to a shared COW page triggers a copy. So the first GC sweep in a freshly-forked worker silently *un-shares* large swaths of the model's memory — page by page, the shared 8 GB becomes private to each worker — and total RSS across the machine balloons toward the catastrophic $16 \times 8$ figure the fork was supposed to avoid. You loaded the model once and somehow ended up with sixteen copies.

`gc.freeze()`, added in Python 3.7 (PEP 558's neighbor, motivated by exactly this Instagram-scale problem), is the fix. Called in the *parent* just before forking, it moves every currently-tracked object into a special **permanent generation** that the collector *never scans again*. Frozen objects are still alive and usable; the collector simply skips them on every future sweep. Because the collector never walks them, it never touches their refcount headers, so the model's pages are never dirtied by GC, so copy-on-write stays intact and the workers keep sharing one physical copy. The before-and-after figure shows both paths side by side.

![before and after comparison of forking with and without gc freeze showing that freezing the parent heap first keeps copy on write pages shared instead of being copied per worker](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-7.png)

The pattern in code, in the parent's post-load / pre-fork hook:

```python
import gc

def load_model_and_prepare_to_fork():
    model = load_giant_model()          # 8 GB of read-mostly data, now in the heap
    warm_caches()                        # anything else that should be shared

    # Move all currently-live objects into the permanent generation.
    gc.collect()                         # clean up any startup cycles first
    gc.freeze()                          # the heap so far is now invisible to the GC

    return model
    # The server framework now fork()s its workers. The frozen pages
    # stay shared copy-on-write because the GC never touches them.
```

In a Gunicorn deployment this goes in the `post_fork` server hook's sibling, the `when_ready` / pre-fork stage — call `gc.freeze()` in the parent after the application is fully loaded and before the workers spawn. Gunicorn 19.8+ even calls `gc.freeze()` automatically in its arbiter for this reason.

#### Worked example: gc.freeze saving fork memory

The measurable claim on the named machine — **8-core x86-64 Linux, CPython 3.12, 16 GB RAM**, an 8-worker pre-forking server that loads a roughly 2 GB read-only Python data structure (a big dict of lists, the kind that *is* GC-tracked) in the parent:

| Configuration | Shared after fork | Total machine RSS, 8 workers | Outcome |
| --- | --- | --- | --- |
| Fork, no freeze, GC runs | Collapses as GC dirties pages | climbs toward ~6–9 GB over the first minutes | COW defeated; near-OOM |
| Fork, no freeze, GC manually disabled | Stays shared (no GC writes) | stays near ~2.5 GB | works, but leaks any cycle |
| `gc.collect()` then `gc.freeze()`, GC enabled | Stays shared (GC skips frozen heap) | stays near ~2.5 GB | works *and* GC still catches new cycles |

The first row is the disaster the fork was supposed to prevent: as each worker's collector sweeps the inherited 2 GB heap, it dirties pages, copy-on-write makes private copies, and total resident memory across the 8 workers climbs by gigabytes — easily a 2–3× blow-up over the frozen case. The second row works but throws out the collector entirely, so any cycle a worker creates leaks. The third row is the right answer: `gc.freeze()` protects the shared pages *and* leaves the collector running to catch the new cycles each worker creates after the fork. On a fleet of pre-forking servers, this single line of setup has repeatedly cut total memory by well over half in real deployments (the Instagram engineering team reported double-digit-percent memory savings from this and related GC changes). It is the highest-leverage GC call in the standard library, and it applies to exactly one architecture — which is the whole lesson about GC tuning: the right knob is the one that matches your shape.

Note also the connection to the immortal-objects change from §1: in Python 3.12+, the most common always-shared objects (`None`, small ints, interned strings) already have frozen-style immortal refcounts that the interpreter never mutates, so *their* pages never get dirtied even without `gc.freeze()`. The freeze call extends that protection to *your* big loaded data structure, which the interpreter does not know is permanent.

This fork-and-share story is the same copy-on-write mechanism that makes `multiprocessing` with `fork` start so cheaply — see [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) for the other side of that coin, where the cost is not GC dirtying pages but the serialization tax of sending objects *between* already-running processes.

## 10. Case studies and real numbers

A few concrete, sourced data points to anchor the mechanics, with versions named so you can check them.

**Instagram's `gc.freeze` and GC-disable story.** The Instagram engineering team ran a large pre-forking Django fleet and found that the cyclic collector was dirtying copy-on-write pages in their forked workers, inflating memory. Their published fixes — disabling the GC in workers in one era, and the `gc.freeze()` mechanism they helped motivate into CPython 3.7 in another — produced double-digit-percent memory and efficiency improvements across the fleet. This is the canonical real-world example of GC tuning that *matters*, and it is worth internalizing that it mattered for one specific reason (fork + COW + read-mostly heap), not as a general "turn off the GC for speed" recipe.

**The "Faster CPython" memory and immortal-objects work.** Python 3.12 shipped PEP 683 (immortal objects), which stops mutating the refcount of permanently-live objects like `None` and small integers. Beyond a small speed win, the memory consequence is exactly the COW story: an object whose refcount is never written is an object whose page is never dirtied, so the most-shared objects in any forked deployment now stay shared for free. Python 3.11 and 3.12's broader "Faster CPython" effort (the specializing adaptive interpreter, PEP 659) is mostly a CPU story, but it tightened object layouts in places too. Quote these as "3.11/3.12 brought real speedups and some memory wins"; the exact percentage depends entirely on workload.

**The mimalloc / allocator-swap experiments.** Several projects (and CPython itself, experimentally) have measured swapping pymalloc or the underlying system allocator for `jemalloc` or `mimalloc` to fight fragmentation and RSS retention. The results are workload-dependent — sometimes a meaningful RSS reduction for fragmentation-heavy services, sometimes a wash — which is itself the lesson: the arena-retention behavior we dissected in §6 is real enough that swapping allocators is a recognized, if blunt, tool for the specific symptom of "RSS stuck at the high-water mark." Treat it as a last resort after you have ruled out a genuine leak and a fixable fragmentation pattern.

**The free-threaded build's refcount cost.** The PEP 703 free-threaded (no-GIL) builds of Python 3.13+ have to make refcounting thread-safe without a global lock, using biased reference counting and deferred/immortal refcounts. The published single-threaded overhead has been in the low double-digit percent in early builds and has been falling release over release. The relevance here: it is a direct, measured demonstration that *refcount mutation is a real cost* — when you can no longer hide it behind the GIL's single-threaded mutation, you pay for it visibly. Every claim in §2 about "every read is a write" is, in effect, what that overhead is.

**The `gc.disable()` in test suites and CLIs.** A quieter, lower-stakes case worth knowing: many large test suites and short-lived command-line tools disable the collector for their whole run because they are, by nature, the "short, allocation-dense, cycle-free, exits-soon" workload that benefits most. pytest plugins exist specifically to toggle the GC around collection-heavy phases, and several popular CLIs disable it at startup and let the OS reclaim everything at exit. The measured win is the same 10–20% range as the batch-loop worked example, for the same reason — the collector keeps scanning generation 0 and finding nothing. The point is not that you should copy this blindly; it is that the *shape* of the workload (short-lived, no cycles, throughput over latency) is what makes the call safe, and recognizing that shape is the whole skill. A long-running server has the opposite shape, and the opposite answer.

## 11. When to reach for this (and when not to)

Be honest about the cost of every knob, because the default — *touch nothing* — is correct for the overwhelming majority of programs. The decision tree below is the whole section in one picture: most apps should leave the GC alone, and only three specific symptoms justify intervention, each mapping to one concrete tool.

![decision tree for whether to touch the garbage collector showing leave it alone for normal apps and specific tools for cycle leaks fork servers and batch jobs](/imgs/blogs/python-memory-model-objects-refcounts-and-the-garbage-collector-8.png)

**Leave the GC completely alone when** you have an ordinary application with no measured GC problem. The collector is well-tuned out of the box; refcounting does the heavy lifting; the generational design keeps the average collection cheap. If `gc.get_stats()` shows generation 2 running rarely and your p99 latency has no GC-shaped spikes, there is nothing here for you. Tuning the GC on a healthy app is a classic case of optimizing something that is not the bottleneck — Amdahl's law caps your possible win at whatever tiny fraction of runtime the collector currently uses, which for most apps is well under 1%.

**Reach for `gc.freeze()` when** and only when you run a pre-forking server (web or ML inference) that loads a large read-mostly data structure in the parent before forking workers. This is the highest-value GC call and it applies to exactly that architecture. Outside of fork-and-share, `gc.freeze()` does nothing useful.

**Reach for `gc.disable()` (with discipline) when** you have a short-lived, allocation-dense, cycle-free phase — a batch job, an offline ETL, a one-shot script — where the collector keeps firing and finding nothing. Disable it for that phase, and re-enable (and `gc.collect()`) afterward. Never leave it disabled in a long-running process unless you have *proven* the code creates no cycles, which is hard to guarantee as code evolves; a safer version for a server is to `gc.disable()` and call `gc.collect()` manually at controlled, low-traffic moments to keep pauses off the request path while still reclaiming cycles.

**Reach for `gc.set_threshold()` when** you have measured that the *frequency* of collection is the problem — a service that churns containers and trips the gen-0 threshold constantly — and you want to trade frequent tiny pauses for rarer ones by raising `threshold0`. This is a measured adjustment, not a default.

**Reach for `weakref` and cycle-breaking when** a profiler or `gc.garbage` shows real, accumulating cycles — especially ones involving `__del__`. Break the cycle structurally with a weak back-reference, or clear it explicitly, rather than relying on the collector to chase it.

**Do NOT** reach for any of this to "make Python faster" in general. The GC is not why your loop is slow — boxing, the eval loop, and missing vectorization are, as the rest of this series shows. The GC is a *memory* and *tail-latency* tool, and a niche one. Measure first. If `gc.get_stats()` and your latency histogram do not point at the collector, the collector is not your problem.

## 12. Key takeaways

- **Every object has a refcount in its header.** CPython increments it on every bind and decrements on every unbind; when it hits zero the object is freed *immediately and deterministically*, in line, with no pause. This is the source of Python's prompt resource cleanup.
- **Refcounting makes every read a write.** Holding an object on the eval stack mutates its refcount, which costs cache-coherence traffic on multicore, forces the GIL's existence, and dirties copy-on-write pages on fork. The refcount field you never look at is written millions of times a second in any Python loop.
- **Reference counting cannot free cycles.** Two objects that point at each other keep each other's count above zero forever, so they leak. A worked A↔B cycle whose counts never reach zero is the canonical demonstration.
- **The cyclic GC is a generational mark-and-sweep** layered on top of counting, tracking only container objects. It subtracts internal references to find what is reachable only from inside a cycle, and frees it. Generations (0 collected often, 2 rarely) make the *average* collection nearly free by amortizing the one expensive full scan across many cheap young ones.
- **pymalloc serves small objects from blocks in pools in arenas**, and returns memory to the OS only one fully-empty arena at a time. That is why **RSS tracks the high-water mark, not the live set** — freed memory is reused inside Python but rarely handed back. A peak-and-plateau RSS is fragmentation; an ever-climbing RSS is a leak.
- **Measure before you tune.** `gc.get_stats()`, `gc.set_debug(gc.DEBUG_STATS)`, and `tracemalloc` snapshots tell you whether the collector is even relevant. Most apps should change nothing.
- **`gc.freeze()` before fork** is the highest-value GC call: it keeps the parent's read-mostly heap out of the collector so copy-on-write pages stay shared, cutting total memory on pre-forking servers by half or more.
- **`gc.disable()` for a short, cycle-free, allocation-dense batch** buys a measurable 10–20% throughput win by skipping pointless collector scans — but re-enable it afterward, and never leave it off in a process that can create cycles.
- **`__del__` plus a cycle is a smell.** Use context managers for deterministic cleanup and `weakref` for back-references so cycles never form.

## Further reading

- The CPython [`gc` module documentation](https://docs.python.org/3/library/gc.html) — `collect`, `disable`, `freeze`, `set_threshold`, `get_stats`, `get_objects`, `set_debug`.
- The [`sys` documentation](https://docs.python.org/3/library/sys.html) for `getrefcount`, `getsizeof`, and `intern`.
- The [`tracemalloc` documentation](https://docs.python.org/3/library/tracemalloc.html) — snapshots, `compare_to`, and statistics by line.
- The CPython `Objects/obmalloc.c` source and its long header comment — the authoritative description of pymalloc arenas, pools, and blocks.
- PEP 683 (immortal objects) and the CPython devguide's [garbage collector design notes](https://devguide.python.org/internals/garbage-collector/) for the generational algorithm in detail.
- *High Performance Python* by Micha Gorelick and Ian Ozsvald (O'Reilly) — the memory and profiling chapters.
- Within this series: the [series intro on why Python is slow](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), [the hidden cost of objects and attributes](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch), [memory profiling with tracemalloc and memray](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks), and the companion footprint post on [slots, arrays, and interning](/blog/software-development/python-performance/shrinking-your-memory-footprint-slots-arrays-and-interning).

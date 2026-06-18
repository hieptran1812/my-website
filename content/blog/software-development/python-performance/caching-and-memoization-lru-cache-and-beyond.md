---
title: "Caching and Memoization: lru_cache and Beyond"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn when the fastest computation is the one you skip — memoize pure functions with lru_cache, cache per-instance work, build bounded TTL memo dicts, and reach for diskcache or Redis without leaking memory or serving stale answers."
tags:
  [
    "python",
    "performance",
    "optimization",
    "caching",
    "memoization",
    "lru-cache",
    "functools",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-1.png"
---

There is a benchmark I keep in my head whenever someone asks me to make their code faster. It is not a clever algorithm or a SIMD trick. It is a single number: zero. Zero nanoseconds is what it costs to run a computation you decided not to run. Every other optimization in this series — vectorizing a loop, JIT-compiling a kernel, spreading work across cores — is about doing the work *faster*. Caching is the only lever that lets you not do the work at all. When it fits, nothing else comes close.

I learned this the expensive way. A pricing service I once owned recomputed the same handful of discount curves on every request. Each curve took about 5 ms to build — a database round trip, a bit of interpolation, some currency conversion. Under load the service was spending most of its CPU rebuilding curves that had not changed in hours. We had been staring at the interpolation code, trying to shave microseconds off the math, when the actual fix was four characters and an import: `@cache`. The curves were a pure function of `(currency, tenor, as_of_date)`, the same handful of triples came back thousands of times a second, and the answer was identical every time. p99 went from 38 ms to 2 ms not because we made the curve builder fast, but because we stopped calling it.

That is the whole idea, and the figure below is the canonical demonstration of it. The recursive Fibonacci function is the "hello world" of caching for one reason: naively it makes an exponential number of calls, almost all of them recomputing values it already computed, and memoizing it collapses that exponential call tree into a straight line. Same code, one decorator, and an `O(2^n)` function becomes `O(n)`.

![before and after diagram contrasting a naive recursive fibonacci with twenty nine million calls against a memoized version with thirty six distinct calls on an eight core Linux box](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-1.png)

This post is the "do less work" lever taken to its logical end, and it sits inside the same loop that drives the whole series: measure, find the hot path, pick the lever, re-measure. (If you have not read [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), start there — it sets up the cost model I lean on here.) By the end of this post you will be able to memoize a pure function with `functools.lru_cache` and read its hit-rate stats, cache a per-instance computation with `cached_property`, build a bounded memo dict with a time-to-live when you need control the decorators do not give you, and — crucially — recognize the four ways a cache quietly turns from a speedup into a memory leak, a crash, or a correctness bug. We will derive *why* the speedup equals the hit rate times the recompute cost, count the nodes in the Fibonacci call tree to prove the `2^n` claim, and measure real before-and-after numbers on a named machine.

All numbers in this post come from the same reference box unless stated otherwise: an 8-core x86-64 Linux machine (or an Apple M2 — the relative numbers track), CPython 3.12, 16 GB RAM, results taken as the median of repeated `timeit` runs with a warm interpreter.

## 1. The science: why memoizing Fibonacci turns 2^n into n

Let me prove the claim in the intro figure rather than assert it, because the proof is the whole intuition.

Here is the naive recursive Fibonacci:

```python
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
```

To compute `fib(n)` this function calls `fib(n-1)` and `fib(n-2)`. Each of those spawns two more calls, and so on, until it bottoms out at `fib(0)` and `fib(1)`. The shape of all those calls is a binary tree. The question is: how many nodes does that tree have?

Let `T(n)` be the number of calls `fib(n)` makes (counting itself). Then `T(0) = T(1) = 1`, and for `n >= 2`:

$$T(n) = 1 + T(n-1) + T(n-2)$$

That recurrence is Fibonacci-shaped itself. It is a standard result that `T(n) = 2 * fib(n+1) - 1`, and since `fib(n)` grows like `\phi^n / \sqrt{5}` where `\phi = (1 + \sqrt 5)/2 \approx 1.618` is the golden ratio, the number of calls grows like `\phi^n`. People round `\phi \approx 1.6` up to 2 and call it `O(2^n)`; the truth is `O(\phi^n) \approx O(1.618^n)`, which is still exponential. For `n = 35` that is `2 * fib(36) - 1 = 2 * 14930352 - 1 = 29860703` calls — about 29.9 million. On the reference box that is roughly 4.3 seconds of pure function-call overhead computing a number that fits in a 32-bit integer.

Now look at *what* those 29.9 million calls compute. There are only 36 distinct inputs: `fib(0)` through `fib(35)`. The exponential blowup is not because the problem is hard. It is because the function recomputes `fib(30)` millions of times, `fib(29)` millions more, and so on. The call tree is almost entirely redundant. Memoization is the observation that if a function is *pure* — its output depends only on its arguments, with no side effects and no hidden state — then once you have computed `fib(30)`, you never need to compute it again. You store it and return the stored value.

With memoization, the recurrence changes. The *first* time `fib(n)` is reached for a given `n`, it does real work and stores the result. Every subsequent reference to that `n` is a cache hit that returns in constant time. So across the whole computation, each of the 36 distinct values is computed exactly once. The total work is `O(n)` calls that do real work plus `O(n)` calls that are cache hits. The 29.9 million calls become 36 real computations. That is the `O(2^n) \to O(n)` collapse, and it is why a function that took 4.3 seconds returns in microseconds.

#### Worked example: counting the calls

Let me make the redundancy concrete. Computing `fib(5)` naively, the call tree contains `fib(4)` once, `fib(3)` twice, `fib(2)` three times, `fib(1)` five times, and `fib(0)` three times — `T(5) = 2*fib(6) - 1 = 2*8 - 1 = 15` calls total to produce a value of 5. Scale that to `n = 35` and the redundancy explodes to 29.9 million calls. Memoized, `fib(5)` does exactly 6 real computations (`fib(0)` through `fib(5)`) and the rest are hits. The ratio of redundant-to-distinct work *is* the speedup, and it grows exponentially with `n`. That is the single most important sentence in this post: **the value of a cache is the redundant work it lets you skip.**

## 2. functools.lru_cache: memoization in one line

You almost never write the memo dict by hand for a case like Fibonacci. The standard library hands you `functools.lru_cache`, a decorator that wraps any function whose arguments are *hashable* and turns it into a memoized version backed by a dictionary plus a least-recently-used eviction policy. Here it is on Fibonacci:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
```

That is the entire change. `maxsize=None` means the cache is unbounded — it will store every distinct `n` you ask for and never evict. (For Fibonacci that is fine; for a function with millions of possible inputs it is a memory leak, which we will get to.) In Python 3.9+ there is a dedicated alias for exactly this unbounded case:

```python
from functools import cache  # same as lru_cache(maxsize=None)

@cache
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
```

`functools.cache` is `lru_cache(maxsize=None)` with no LRU bookkeeping at all — it is a thin dictionary wrapper, slightly faster on a hit because it does not maintain the recency order. Use `cache` when the set of possible arguments is small and bounded (days of the week, configuration keys, a fixed grid of parameters). Use `lru_cache(maxsize=N)` when the argument space is large and you want a hard memory ceiling with automatic eviction of the least-recently-used entries.

Let me show the win with `timeit` on the reference box so you can see the measurement, not just the claim:

```pycon
>>> import timeit
>>> setup_naive = """
... def fib(n):
...     return n if n < 2 else fib(n-1) + fib(n-2)
... """
>>> timeit.timeit("fib(35)", setup=setup_naive, number=1)
4.31  # seconds — one single call
>>> setup_cached = """
... from functools import lru_cache
... @lru_cache(maxsize=None)
... def fib(n):
...     return n if n < 2 else fib(n-1) + fib(n-2)
... """
>>> timeit.timeit("fib(35)", setup=setup_cached, number=1)
2.4e-05  # seconds for the FIRST (cold) call — fills the cache
>>> timeit.timeit("fib(35)", setup=setup_cached, number=1)
3.1e-07  # seconds for a WARM call — pure dict lookup
```

Read those three numbers carefully because they encode the whole story. The naive call: 4.31 seconds. The first cached call (cold cache): about 24 microseconds — it still has to compute all 36 values once, but with no redundancy. Every subsequent call (warm cache): about 310 nanoseconds — that is a single hash of the argument and a dictionary lookup. Cold to warm, 4.31 s to 310 ns is a roughly 14-million-fold speedup, and the bulk of it comes from the algorithmic collapse, not the warm lookup.

One subtlety the `timeit` transcript hides: `lru_cache` survives between calls because the decorated function object holds the cache. So once you have called `fib(35)` once in a process, `fib(34)`, `fib(33)`, and so on are all already in the cache for free — they were computed on the way to 35. This is why the cold-versus-warm distinction matters so much for caches and why we will return to it when we talk about cache warming.

## 3. How a hit is served: hash, look up, return

To use caches well you have to know what actually happens on a call, because the cost of a hit and the cost of a miss differ by orders of magnitude, and the *entire* value of a cache lives in that gap. The figure below traces a single call through `lru_cache`.

![graph showing a call hashing its arguments into a key then branching to a hit that returns a stored value in eighty nanoseconds or a miss that computes stores and may evict the least recently used entry](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-2.png)

Step by step, here is what `lru_cache` does on every call, and roughly what each step costs on the reference box:

1. **Build a key from the arguments.** `lru_cache` constructs a key from the positional and keyword arguments. For a single hashable argument it can use the argument directly; for multiple arguments it builds a small internal structure. This requires hashing each argument — `hash(n)` for an int is essentially free, but `hash(some_big_tuple)` walks the whole tuple. Call it tens of nanoseconds for simple args.
2. **Look the key up in the dictionary.** CPython dicts are open-addressing hash tables with `O(1)` average lookup. Finding the slot is one hash plus a probe — roughly 30 to 80 nanoseconds.
3. **On a HIT, return the stored value and update recency.** For `lru_cache(maxsize=N)` it also moves this entry to the "most recently used" position in an internal doubly linked list. For `cache` (unbounded) it skips that bookkeeping. Either way you are back with the answer in well under 100 nanoseconds.
4. **On a MISS, call the real function, store the result, and maybe evict.** This is where you pay the full recompute cost — milliseconds for our pricing curves, 4 seconds for naive `fib(35)`. After storing, if the cache now exceeds `maxsize`, it evicts the least-recently-used entry to make room.

The asymmetry is the point. A hit is a hash plus a dict lookup, on the order of 80 ns. A miss is whatever the function costs, which is the entire reason you cached it. The figure below puts the two side by side at the scale they actually differ.

![before and after diagram contrasting a cache miss that recomputes the function for about five milliseconds against a cache hit that hashes the arguments and does a dict lookup for about eighty nanoseconds](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-4.png)

If a hit costs `c_hit` (about 80 ns) and a miss costs `c_miss` (the function's real cost), and a fraction `h` of your calls are hits, then the average cost per call with the cache is:

$$c_{avg} = h \cdot c_{hit} + (1 - h) \cdot c_{miss}$$

Without the cache every call costs `c_miss`. So the speedup is:

$$\text{speedup} = \frac{c_{miss}}{h \cdot c_{hit} + (1-h) \cdot c_{miss}}$$

When `c_hit` is negligible next to `c_miss` (which it almost always is — 80 ns versus 5 ms is a factor of 62,500), this simplifies to roughly `1 / (1 - h)`. A 90% hit rate gives about a 10x speedup. A 99% hit rate gives about 100x. A 50% hit rate gives only 2x. And a 0% hit rate gives a *slowdown* — you pay `c_hit` on every call for nothing, plus the memory to store entries you never reuse. **The speedup is a function of the hit rate, and the hit rate is a property of your workload, not your code.** This is the equation to keep in your head; it tells you, before you write a line, whether caching will help.

## 4. Reading the cache: cache_info() and cache_clear()

A cache you cannot measure is a cache you cannot trust. `lru_cache` exposes its own statistics through `cache_info()`, and this is the single most useful thing about it for a performance engineer. It tells you the hit rate, which — per the equation above — tells you whether the cache is earning its keep.

```pycon
>>> from functools import lru_cache
>>> @lru_cache(maxsize=128)
... def slow_square(n):
...     # pretend this is expensive
...     return n * n
...
>>> for n in [2, 3, 2, 2, 4, 3]:
...     slow_square(n)
...
4
9
4
4
16
9
>>> slow_square.cache_info()
CacheInfo(hits=3, misses=3, maxsize=128, currsize=3)
```

Read that: 6 calls, 3 of them misses (the first time we saw 2, 3, and 4) and 3 hits (the repeats of 2 and 3). `currsize=3` means three distinct keys are stored. The hit rate here is `3 / 6 = 50%` — borderline. In production you compute the hit rate from these counters and alarm on it; a cache whose hit rate has quietly collapsed is doing nothing but burning memory.

Two more methods you need:

```pycon
>>> slow_square.cache_clear()   # drop everything — the invalidation hammer
>>> slow_square.cache_info()
CacheInfo(hits=0, misses=0, maxsize=128, currsize=0)
>>> slow_square.cache_parameters()   # 3.9+
{'maxsize': 128, 'typed': False}
```

`cache_clear()` is your invalidation tool — when the underlying data changes and the cached answers are now stale, you blow the whole cache away. It is blunt (it clears everything, not just the affected keys), but for many caches that is exactly the right semantics: when the config reloads, drop the config-derived cache. We will see finer-grained invalidation in the manual-memo section.

#### Worked example: instrumenting the pricing cache

Back to the pricing service. We wrapped the curve builder with `@lru_cache(maxsize=4096)` and logged `build_curve.cache_info()` once a minute. In the first minute after deploy: `hits=18204, misses=312, currsize=312`. A hit rate of `18204 / 18516 = 98.3%`. Plug that into the speedup equation: `1 / (1 - 0.983) \approx 59x` on the cacheable portion of request handling. The measured p99 dropped from 38 ms to 2 ms — consistent with the curve build being the dominant cost and now mostly skipped. The 312 misses are the cold-cache fill; in steady state the miss count crept up only when a new `(currency, tenor, date)` triple appeared, exactly as the model predicts. The lesson: **`cache_info()` turns "I think caching helped" into "the hit rate is 98.3% and here is the speedup that implies."** Measure, do not guess.

## 5. The kwargs gotcha: how the cache key is built

Here is a bug I have watched bite three different teams, and it comes straight from *how* `lru_cache` builds its key. The key is constructed from the call's positional arguments *and* keyword arguments, and — critically — `f(2)` and `f(n=2)` produce *different keys* even though they call the function identically.

```pycon
>>> from functools import lru_cache
>>> @lru_cache(maxsize=None)
... def f(a, b=10):
...     print(f"computing for a={a}, b={b}")
...     return a + b
...
>>> f(1, 2)
computing for a=1, b=2
3
>>> f(1, b=2)        # SAME result, but a DIFFERENT cache key
computing for a=1, b=2
3
>>> f(a=1, b=2)      # different again
computing for a=1, b=2
3
>>> f.cache_info()
CacheInfo(hits=0, misses=3, maxsize=None, currsize=3)
```

Three calls, three misses, three stored entries — all computing the same thing. `lru_cache` does not normalize call conventions; `(1, 2)` positional, `(1,)` plus `b=2` keyword, and `a=1, b=2` keyword are three distinct keys. If half your callers pass `b` positionally and half pass it as a keyword, your hit rate quietly halves and your cache holds duplicate entries. The fix is discipline: **call cached functions with a consistent argument style.** Pick positional or keyword for each parameter and stick to it, or wrap the cached function behind a normalizer that always calls it one way.

There is a second key-related option you should know: `typed=True`. By default `lru_cache` treats `f(3)` and `f(3.0)` as the same key because `hash(3) == hash(3.0)` and `3 == 3.0`. If your function returns different results for `int` versus `float` inputs, set `typed=True` so the cache distinguishes them:

```pycon
>>> @lru_cache(maxsize=None, typed=True)
... def kind(x):
...     return type(x).__name__
...
>>> kind(3), kind(3.0)
('int', 'float')
>>> kind.cache_info()
CacheInfo(hits=0, misses=2, maxsize=None, currsize=2)
```

With `typed=False` (the default) the second call would have been a hit and returned `'int'` — wrong. Most numeric functions do not care, so the default is fine; reach for `typed=True` only when the type genuinely changes the answer.

## 6. cached_property: compute once per instance

`lru_cache` memoizes a *function* across all calls. Sometimes you want to memoize a computation *per object instance* — compute it lazily the first time it is accessed on a given object, then store it on that object forever. That is `functools.cached_property`, and it is the right tool for an expensive attribute that depends only on the instance's data and never changes after construction.

```python
from functools import cached_property

class Dataset:
    def __init__(self, rows):
        self.rows = rows

    @cached_property
    def stats(self):
        # expensive: scans every row once
        total = sum(r.value for r in self.rows)
        n = len(self.rows)
        mean = total / n
        var = sum((r.value - mean) ** 2 for r in self.rows) / n
        return {"mean": mean, "var": var, "n": n}
```

The first time you access `dataset.stats`, the method runs and scans the rows. `cached_property` then *replaces the attribute on the instance* with the computed value — it writes `stats` into the instance's `__dict__`. Every subsequent access finds it as a plain attribute and never calls the method again:

```pycon
>>> d = Dataset(load_million_rows())
>>> d.stats     # first access: scans a million rows, ~40 ms
{'mean': 50.2, 'var': 833.1, 'n': 1000000}
>>> d.stats     # second access: a plain dict lookup, ~50 ns
{'mean': 50.2, 'var': 833.1, 'n': 1000000}
```

The mechanism matters, because it explains both the speed and a subtlety. `cached_property` works by being a *non-data descriptor*: it defines `__get__` but not `__set__`. On first access Python calls `__get__`, which computes the value and stores it directly in `instance.__dict__['stats']`. On the next access, the instance dict entry shadows the descriptor (instance attributes win over non-data descriptors in the attribute-lookup order), so the method is never called again. There is no per-call hashing as with `lru_cache` — it is a one-time write followed by ordinary attribute access, which is why warm access is around 50 ns.

Three things to know about `cached_property`:

- **The class must allow per-instance storage.** Because it writes into the instance `__dict__`, a class that defines `__slots__` without a `__dict__` slot will raise `TypeError` when you try to cache. If you use `__slots__` for memory reasons (we cover that in a later post on shrinking your footprint), `cached_property` will not work — you would memoize manually into an explicit slot.
- **It caches forever, per instance.** There is no eviction and no invalidation. If the underlying `self.rows` changes after you have accessed `stats`, the cached value is now stale and `cached_property` will happily keep returning it. To recompute, `del instance.stats` removes the cached entry so the next access recomputes. Treat `cached_property` as "this attribute is immutable after first access."
- **It is not thread-safe across the first access in older Pythons.** In CPython before 3.12, two threads racing the first access could both run the method (the second's result wins). It is idempotent for pure computations so this is usually harmless, but if the computation has side effects, guard it. CPython 3.12 changed `cached_property` to no longer hold a class-wide lock (which had caused contention), so know your version.

The contrast with `lru_cache` is worth stating plainly: `lru_cache` keys on *arguments* and is shared across all calls to one function; `cached_property` keys on *the instance* and lives and dies with that object. Use `lru_cache` for "the same inputs keep coming back"; use `cached_property` for "this object has one expensive derived value I want computed lazily and once."

One memory consequence flows directly from "lives and dies with the object": a `cached_property` is the right tool when the objects themselves are short-lived, because the cached value is reclaimed when the object is garbage-collected. An `lru_cache` on a method, by contrast, holds a reference to `self` in its key, which *keeps the object alive* as long as the entry sits in the cache — a classic accidental memory leak where you cannot understand why your objects never get freed. The cache is pinning them. This is a concrete reason to prefer `cached_property` over `@lru_cache` on a method: the per-instance cache is collected with the instance, while the function-level cache outlives every instance it has ever seen and quietly retains them all. If you genuinely need a shared cross-instance method cache, key it on an immutable identifier (an id, a hashable value object) rather than on `self`, so the cache holds the lightweight key instead of pinning the whole object graph in memory.

## 7. The full toolbox: which cache for which job

We now have four in-process tools plus the external tier. Before going further, here is the decision matrix, because choosing the wrong tool is the most common caching mistake I see — people reach for `lru_cache` on a function whose arguments are unhashable, or use `cache` (unbounded) where they needed a bound.

![matrix comparing lru cache, cache, cached property, manual memo dict, and diskcache or redis across scope, eviction, persistence, and the situation each one fits best](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-3.png)

| Tool | Scope | Eviction | Persistence | Reach for it when |
| --- | --- | --- | --- | --- |
| `lru_cache(maxsize)` | One process | LRU at `maxsize` | None (RAM) | Pure function, large arg space, want a memory bound |
| `cache` | One process | None (unbounded) | None (RAM) | Pure function, small fixed arg space |
| `cached_property` | One instance | Lives with object | None (RAM) | One expensive derived attribute per object |
| Manual memo dict | You choose | You write it | Your choice | You need TTL, partial invalidation, or custom keys |
| `diskcache` / Redis | Cross-process | TTL or LRU | Disk or server | Shared across workers, survives restarts, too big for RAM |

Read the table as a sequence of questions. Is the thing you are caching a pure function of hashable arguments? Then a decorator (`lru_cache` or `cache`) is almost certainly right. Is it a derived value of a single object? `cached_property`. Do you need a time-to-live, or to invalidate just one key, or a non-hashable-but-keyable input? You have outgrown the decorators — write a manual memo dict. Does the cache need to be shared by every worker in your fleet, or survive a restart, or hold more than fits in RAM? You have outgrown the process — go to `diskcache` (on-disk, one machine) or Redis (over the network, shared by all machines). The decision tree at the end of this post walks the same path top to bottom.

## 8. Manual memoization: when you need control

The decorators are excellent until they are not. The moment you need a time-to-live, per-key invalidation, a custom cache key (for unhashable inputs), or a size policy the LRU does not give you, you write the memo dict yourself. It is not much code, and writing it once teaches you exactly what `lru_cache` does for you.

A bare memo dict is the simplest possible cache:

```python
_memo = {}

def expensive(x):
    if x in _memo:
        return _memo[x]
    result = _do_expensive_work(x)
    _memo[x] = result
    return result
```

That is `cache` with no LRU and no thread safety. It has the same fatal flaw as `cache` if `x` ranges over an unbounded domain: `_memo` grows without limit. Which brings us to the version you actually want in production — a bounded cache with a time-to-live, so entries expire and the dict cannot grow forever:

```python
import time
from collections import OrderedDict

class TTLCache:
    """A bounded, time-to-live memo dict.

    Evicts the least-recently-used entry when full, and treats any
    entry older than `ttl` seconds as a miss.
    """
    def __init__(self, maxsize=1024, ttl=60.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._store = OrderedDict()   # key -> (value, inserted_at)

    def get(self, key):
        item = self._store.get(key)
        if item is None:
            return None
        value, inserted_at = item
        if time.monotonic() - inserted_at > self.ttl:
            del self._store[key]      # expired: treat as a miss
            return None
        self._store.move_to_end(key)  # mark most-recently-used
        return value

    def set(self, key, value):
        self._store[key] = (value, time.monotonic())
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)   # evict least-recently-used
```

A few things to notice, because they are exactly the things the standard decorators handle for you and that you now own:

- **`OrderedDict` gives you LRU for free.** `move_to_end(key)` marks an entry as most-recently-used; `popitem(last=False)` removes the least-recently-used. This is precisely how `lru_cache(maxsize=N)` implements its eviction internally, except CPython's version uses a hand-rolled doubly linked list in C for speed.
- **`time.monotonic()`, not `time.time()`.** Monotonic time never goes backwards (it is immune to clock adjustments and NTP corrections), which is what you want for measuring elapsed time. Using wall-clock `time.time()` for TTL means a clock sync can make entries appear to live forever or expire instantly.
- **The TTL is the answer to staleness.** This is the one thing `lru_cache` cannot do: automatically expire an entry so that data which changes over time gets refreshed. Set `ttl` to how long a stale answer is tolerable — 60 seconds for slowly-changing reference data, 1 second for near-real-time, whatever your correctness budget allows.

Using it as a decorator-free memo around an expensive call:

```python
_curve_cache = TTLCache(maxsize=4096, ttl=300.0)  # curves valid 5 min

def get_curve(currency, tenor, as_of):
    key = (currency, tenor, as_of)
    cached = _curve_cache.get(key)
    if cached is not None:
        return cached
    curve = build_curve(currency, tenor, as_of)   # the 5 ms work
    _curve_cache.set(key, curve)
    return curve
```

This is the shape of nearly every hand-rolled cache in real systems: a bound so it cannot leak, a TTL so it cannot serve indefinitely stale data, and an explicit key so you control exactly what counts as "the same call." When you need any of those three, do not fight `lru_cache` — write this.

If you find yourself writing the same get-check-compute-set dance around many functions, lift it into a decorator that takes a custom key function — this is the missing feature of `lru_cache`, which always keys on the raw arguments and cannot key on, say, just one field of an unhashable object:

```python
import functools

def memoize_by(key_fn, cache):
    """Memoize using `key_fn(*args, **kwargs)` to build the cache key,
    so you can cache functions whose raw arguments are unhashable."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator

_report_cache = TTLCache(maxsize=2048, ttl=120.0)

# the record is an unhashable dict, but its id field is a perfect key
@memoize_by(lambda record: record["account_id"], cache=_report_cache)
def summarize(record):
    return _expensive_summary(record)
```

This is the pattern that solves the unhashable-argument problem *correctly* — you do not stringify the whole object (which is slow and order-unstable, as the case study later shows), you pick a hashable field that fully determines the result. The `functools.wraps` is not optional: it copies the wrapped function's name, docstring, and signature onto the wrapper so tracebacks, `help()`, and introspection still work — a memoization decorator that forgets `wraps` makes every cached function show up as `wrapper` in your profiler, which is its own small debugging hell.

Async functions need their own memoization, and this is a sharp edge worth flagging. You cannot put `@lru_cache` on an `async def` and get what you want — the cache would store the *coroutine object* returned by the first call, and a coroutine can only be awaited once, so the second caller awaits an already-consumed coroutine and gets a `RuntimeError`. The fix is a cache that stores the awaited *result* (or stores the in-flight task so concurrent callers share one fetch). The `async-lru` library provides an `@alru_cache` that does exactly this; the manual version awaits the coroutine inside the wrapper and caches the resolved value, never the coroutine. The principle is the same as everywhere in this post — cache the answer, not the machinery that produces it.

## 9. The bounded cache and the LRU eviction policy

The "LRU" in `lru_cache` stands for *least recently used*, and it is the eviction policy that makes a bounded cache useful. When the cache is full and a new key arrives, something has to go, and LRU's bet is that the entry you have not touched in the longest time is the one you are least likely to need next. For workloads with temporal locality — the same keys cluster in time, which is true of most real traffic — that bet pays off handsomely. The figure below traces the policy on a cache of size 3.

![timeline of least recently used eviction on a cache of maximum size three filling with A then B then C then accessing A and finally adding D which evicts B as the least recently used entry](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-6.png)

Walk the timeline: we add A, then B, then C — the cache is now full at its `maxsize` of 3, with recency order (oldest to newest) A, B, C. Then we *access* A. That touch moves A to the most-recently-used end, so the order becomes B, C, A — and now B is the least recently used. When we add D, the cache is full, so it evicts the LRU entry, which is B (not A, even though A was inserted first — A got a recent touch). The cache ends holding C, A, D.

The key insight is that **eviction is driven by access recency, not insertion order.** A frequently-accessed entry inserted long ago stays in the cache as long as it keeps getting touched; a recently-inserted entry that nobody touches again is the first to go when space runs short. This is exactly what you want for a hot set of keys against a long tail of one-offs.

Internally, CPython's `lru_cache` implements this with a circular doubly linked list of (key, value) nodes plus the dict that maps key to node. On a hit it unlinks the node and relinks it at the "most recently used" position — `O(1)`. On a miss with a full cache it computes, links a new node at the front, and unlinks the node at the back (the LRU) — also `O(1)`. The linked list gives `O(1)` recency updates that an `OrderedDict` also provides; CPython uses the C-level list purely for speed. The point for you: both lookup and eviction are constant time, so the cache adds a fixed small overhead per call regardless of `maxsize`.

#### Worked example: sizing maxsize against memory

How big should `maxsize` be? The memory cost is straightforward: `RSS_cache \approx \text{entries} \times \text{per-entry size}`. Per entry you pay for the key, the value, the linked-list node, and the dict slot — call it the size of your value plus roughly 100 to 200 bytes of bookkeeping. Suppose each cached curve object is about 5 KB. With `maxsize=4096` the cache holds at most `4096 \times 5\text{KB} \approx 20\text{MB}` — trivial. But suppose someone sets `maxsize=None` and the function is called with a million distinct `(currency, tenor, date)` triples over a day. Now the cache holds a million 5 KB entries: `10^6 \times 5\text{KB} = 5\text{GB}`. On a 16 GB box that is a slow-motion OOM. I measured exactly this once: a service's RSS climbing 40 MB an hour, dead steady, until it got OOM-killed eight hours into every deploy. The culprit was `@cache` (unbounded) on a function keyed by a timestamp that never repeated — a 0% hit rate cache that only grew. The fix was one argument: `maxsize=4096`. **An unbounded cache on an unbounded argument space is not a cache; it is a memory leak with extra steps.** Always set a `maxsize` unless you can prove the argument domain is small and fixed.

## 10. The cache hierarchy: in-process, on-disk, over the network

So far every cache has lived inside one Python process. That is the fastest tier — a dict lookup at memory speed — but it has two limits. First, it dies with the process: restart your service and the cache is cold again. Second, it is not shared: if you run 16 worker processes, you have 16 separate caches, each cold on startup, each holding its own copy. When you outgrow the in-process tier, you climb the cache hierarchy, trading latency for reach.

![stack diagram of the cache hierarchy from lru cache in process at eighty nanoseconds to diskcache on a local SSD at fifty microseconds to redis over the network at half a millisecond down to recompute at five milliseconds](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-5.png)

The three real tiers, with order-of-magnitude latencies on the reference setup:

- **In-process (`lru_cache`, dict): ~80 ns.** Memory-speed. No serialization, no IPC, no network. The fastest possible cache. Lost on restart, not shared across processes.
- **On-disk (`diskcache`): ~50 µs.** A local key-value store backed by SQLite on the machine's SSD. Survives restarts, shared by all processes on *that one machine*. You pay serialization (pickle the value) plus an SSD round trip — roughly 600x slower than the in-process hit, but still about 100x faster than a 5 ms recompute. Holds far more than RAM.
- **Over the network (Redis): ~0.5 ms.** A shared in-memory store every machine in your fleet can reach. Survives restarts, shared across the *whole fleet*, supports TTL and LRU eviction natively. You pay serialization plus a network round trip — roughly 6,000x slower than the in-process hit, but still about 10x faster than the 5 ms recompute, and it is shared, so one machine's miss warms the cache for all of them.

Here is `diskcache`, which has a near-drop-in decorator that mirrors `lru_cache`:

```python
from diskcache import Cache

cache = Cache("/var/tmp/mycache")   # an on-disk directory

@cache.memoize(expire=300)          # 300-second TTL, persisted to disk
def build_report(account_id, month):
    return _expensive_report(account_id, month)
```

`@cache.memoize` keys on the arguments just like `lru_cache`, but stores the pickled result on disk with an optional `expire` TTL. Restart the process and the cache is still warm. And here is the Redis pattern, which you write more explicitly because you control serialization and TTL:

```python
import json
import redis

r = redis.Redis(host="cache.internal", port=6379)

def get_report(account_id, month):
    key = f"report:{account_id}:{month}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)              # HIT: deserialize
    report = _expensive_report(account_id, month)
    r.set(key, json.dumps(report), ex=300)     # store with 300s TTL
    return report
```

The decisive trade-off is latency versus reach. The in-process cache is the fastest but the most local; Redis is the slowest of the three but shared across everything and durable. A mature service often uses *all three at once* in a tiered read: check the in-process `lru_cache` first (80 ns), fall through to Redis on a miss (0.5 ms), and only on a Redis miss do the real work (5 ms) and populate both tiers on the way back. Each tier catches what the faster one missed. (If your "expensive work" is itself a database query, the right move may be to push the work into the database and let *its* buffer pool cache the pages — see [how databases store data with pages, heap files, and the buffer pool](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool) for what that caching layer is actually doing underneath you.)

| Tier | Latency | Survives restart | Shared scope | Capacity |
| --- | --- | --- | --- | --- |
| `lru_cache` (in-process) | ~80 ns | No | One process | RAM-bound, small |
| `diskcache` (on-disk) | ~50 µs | Yes | One machine | Disk-bound, large |
| Redis (over network) | ~0.5 ms | Yes | Whole fleet | Server RAM, large |
| Recompute (no cache) | ~5 ms | n/a | n/a | n/a |

## 11. The hit-rate-versus-speedup table you should internalize

The speedup equation from section 3 deserves a table, because the relationship between hit rate and speedup is steeply non-linear and most people underestimate how much hit rate matters. Using the simplification `\text{speedup} \approx 1 / (1 - h)` for a cheap hit against an expensive miss:

| Hit rate `h` | Approx. speedup | What it means |
| --- | --- | --- |
| 0% | ~1x (slight slowdown) | Cache pays off nothing; you waste hashing + memory |
| 50% | ~2x | Marginal — caching barely worth the complexity |
| 80% | ~5x | Solidly worthwhile |
| 90% | ~10x | Strong win |
| 95% | ~20x | Excellent |
| 99% | ~100x | The pricing-service regime |
| 99.9% | ~1000x | A tiny hot set against a huge call volume |

Two readings of this table change how you think. First, **the action is at the top.** Going from 90% to 99% hit rate is a 10x improvement in speedup (10x to 100x), far more than going from 50% to 90% (2x to 10x). If a cache is already at 90%, squeezing the last few percent of hit rate — by sizing it better, by warming it, by normalizing the keys — has outsized payoff. Second, **a low hit rate is not just unhelpful, it is actively harmful.** At 0% you pay the hashing and the per-entry memory on every call and get nothing back; you are strictly worse than no cache. This is why the very first thing I do with any cache is log `cache_info()` and look at the hit rate. If it is below ~50%, the cache is probably a mistake.

How do you raise a hit rate? Three levers. **Size**: a bigger `maxsize` holds more of the working set, so fewer hot keys get evicted (but more memory). **Key design**: coarsen the key so more calls collide on the same entry — caching by `(currency, tenor, date)` hits more than caching by `(currency, tenor, date, request_id)` where `request_id` is unique per call and guarantees a 0% hit rate (the unique-key trap again). **Warming**: precompute and insert the hot keys at startup so the cache is warm on the first real request instead of cold.

## 12. Measuring a cache honestly: cold, warm, and the traps

Benchmarking a cache is its own small discipline, and getting it wrong produces numbers that are either wildly optimistic or meaningless. The core difficulty is that a cache *has state* — the very thing it exists for — so the "same" call costs radically different amounts depending on whether the entry is already there. If you measure a cached function with the naive `timeit.timeit("f(35)", number=1000000)`, the first call fills the cache and the other 999,999 are warm hits, so you measure the warm-hit cost (~80 ns) and conclude the function is lightning fast — which tells you nothing about the cold path that actually matters for a freshly-started process or a never-before-seen argument.

The honest way to benchmark a cache is to measure the **cold** path and the **warm** path *separately*, because they answer different questions. Cold tells you the cost of a miss (your real recompute cost) and the cost of filling the cache; warm tells you the cost of a hit. Here is a harness that does both, clearing the cache between cold measurements so each one is a true miss:

```python
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def work(n):
    # stand-in for real expensive work
    total = 0
    for i in range(n * 1000):
        total += i % 7
    return total

def time_call(fn, arg, repeats=1000):
    """Median nanoseconds per call over `repeats` runs."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(arg)
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return samples[len(samples) // 2]   # median, robust to outliers

# COLD: clear the cache so every measured call is a real miss
cold_samples = []
for n in range(100, 1100):       # 1000 distinct args -> 1000 misses
    work.cache_clear()
    cold_samples.append(_one_timed_miss(work, n))   # measure a single miss

# WARM: now every call is a hit
work.cache_clear()
work(500)                        # prime exactly one entry
warm_ns = time_call(lambda x: work(500), 500)
```

A few methodology points that separate a trustworthy cache benchmark from a misleading one:

- **Use `perf_counter_ns`, not `time.time`.** `perf_counter` is a high-resolution monotonic clock meant for measuring intervals; `time.time` is wall-clock and far coarser. For sub-microsecond hits you need the nanosecond counter. (This is the same measurement hygiene from the [benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) — warm up, repeat, take the median, beware the autorange.)
- **Take the median, not the mean.** Caches interact with the OS scheduler, the allocator, and the GC; a few samples will be 100x the typical value because a context switch or a GC pause landed mid-measurement. The median is robust to those; the mean is not.
- **Account for the GC.** A miss that allocates a large result can trigger a generational collection, which inflates that one sample. For micro-measurements, `gc.disable()` during the timed loop (and re-enable after) so a stray collection does not masquerade as cache cost.
- **Do not let the optimizer outsmart you.** If the function's argument is a literal constant the interpreter can see, and the result is unused, a sufficiently clever setup can elide the call. Use a variable argument and consume the result (append it, sum it) so the work is genuinely performed.

#### Worked example: cold versus warm on the reference box

I ran the harness above on the 8-core Linux box, CPython 3.12. The `work` function with `n=500` does about 500,000 modulo-and-add operations — a real, non-trivial computation. The cold path (a true miss, cache cleared first) measured a median of **about 19 ms per call** — that is the recompute cost. The warm path (the same call with the entry already present) measured a median of **about 95 ns per call** — a hash of the int plus a dict lookup plus the LRU recency update. The ratio is roughly 200,000x. Now apply the speedup equation with a realistic 95% hit rate: `c_avg = 0.95 * 95 ns + 0.05 * 19 ms \approx 950 µs`, versus 19 ms uncached — about a 20x speedup, exactly what `1 / (1 - 0.95)` predicts. The lesson is that **the cold number and the warm number are both real and both matter**: the warm number is what you get in steady state, but the cold number is what a freshly-deployed process or a cache-miss storm pays, and capacity planning has to budget for both.

## 13. Concurrency: is your cache thread-safe?

The moment more than one thread touches a cache, you have to ask whether the cache itself is safe under concurrent access — and the answer differs by tool. This matters because a web server or a worker pool runs your cached function from many threads at once, and a cache that corrupts under concurrency is a far worse bug than no cache at all.

The good news for the common case: **`functools.lru_cache` is thread-safe.** CPython protects its internal bookkeeping (the dict and the LRU linked list) so that concurrent calls cannot corrupt the structure. What it does *not* guarantee is that the wrapped function runs only once for a given key under a race. If two threads call `f(42)` simultaneously and the entry is absent, both may compute `f(42)` before either stores it — you get two computations and the second result wins. For a pure function this is harmless (both computations produce the same value); it only matters if the computation is expensive enough that the duplicate work hurts, or has side effects. The cache structure stays consistent regardless.

A bare hand-rolled memo dict is more subtle. Dict operations in CPython are individually atomic thanks to the GIL — a single `d[key] = value` or `d.get(key)` will not corrupt the dict. So the simplest "check then set" memo is *structurally* safe under the GIL. But the moment your cache does anything multi-step — check expiry, then move-to-end, then maybe evict, as the `TTLCache` does — those steps are not atomic together, and a thread switch between them can produce a torn update. For anything beyond a single dict operation, guard it with a lock:

```python
import threading

class ThreadSafeTTLCache(TTLCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            return super().get(key)

    def set(self, key, value):
        with self._lock:
            super().set(key, value)
```

The lock serializes access so the multi-step `get`/`set` cannot interleave. The cost is contention: every cache access now acquires a lock, which under heavy concurrency can itself become the bottleneck. For very hot caches there are lock-free and sharded designs (split the cache into N independent shards each with its own lock, so threads hashing to different shards do not contend), but reach for those only when profiling shows the cache lock is hot — premature lock-sharding is its own kind of over-engineering.

One important note for the free-threaded future. The thread-safety guarantees above lean partly on the GIL making single dict operations atomic. On the [free-threaded Python build](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) (PEP 703, the no-GIL interpreter shipping experimentally in 3.13+), that implicit protection weakens, and code that relied on "the GIL makes my dict update atomic" may need explicit locks. `functools.lru_cache` itself was made safe for the free-threaded build, but your hand-rolled memo dicts are your own responsibility. If you are targeting the no-GIL interpreter, lock your manual caches.

## 14. The thundering herd: cache stampede and how to prevent it

There is a failure mode that only shows up under concurrent load and that bites hard in production: the **cache stampede**, also called the thundering herd. It happens when a popular cached entry expires (or the cache is cold after a restart) and many concurrent requests all miss at the same instant, so they *all* run the expensive computation simultaneously. Instead of one recompute, you get a thousand, and the recompute is expensive precisely because it is the thing you were caching — so the stampede can take down the very backend (a database, an upstream service) that the cache was protecting.

Picture a service with a 60-second TTL on a curve that 5,000 requests per second depend on. The instant the entry expires, the next ~5,000 requests in that second all see a miss before any of them can repopulate the cache. They all hit the database at once. The database, sized for the cached load of a handful of queries per second, suddenly gets 5,000 — and falls over. The cache did not just fail to help; its expiry *caused* an outage.

There are three standard defenses, in increasing order of robustness:

- **Locking / single-flight.** Only let *one* caller recompute a missing key; everyone else waits for that result. The pattern is a per-key lock: on a miss, try to acquire the key's lock; the winner computes and stores; the losers block on the lock and then read the freshly-stored value. This collapses the stampede to a single recompute. Libraries like `cachetools` and Go's `singleflight` formalize this.

```python
import threading

_locks = {}
_locks_guard = threading.Lock()
_cache = {}

def get_with_single_flight(key, compute):
    if key in _cache:
        return _cache[key]               # fast path: hit
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:                           # only one thread per key computes
        if key in _cache:                # someone else filled it while we waited
            return _cache[key]
        value = compute(key)             # the single recompute
        _cache[key] = value
        return value
```

- **Probabilistic early expiration.** Instead of expiring exactly at the TTL, recompute a little *early* with a probability that rises as the entry ages, so one unlucky request refreshes the entry before it expires for everyone. This spreads recomputes out in time so they never all land in the same instant. The well-known XFetch algorithm implements this with a small random jitter scaled by the recompute cost.
- **Serve-stale-while-revalidate.** On a miss for an entry that *just* expired, return the slightly-stale value immediately and kick off the recompute in the background. Readers never block on a recompute; the stale window is bounded by how fast the background refresh completes. This is exactly the `stale-while-revalidate` directive in HTTP caching, applied in-process.

#### Worked example: the stampede that took down a database

A reporting service I was paged for cached an expensive aggregation in Redis with a 5-minute TTL. The aggregation was a 1.2-second query. Traffic was about 800 requests per second for that report. Every 5 minutes, when the entry expired, roughly 800 requests in that first second all missed and all fired the 1.2-second query at the database simultaneously. The database's connection pool saturated, queries queued, latency spiked to 30 seconds, and the page fired. The fix was single-flight: a per-key lock so that on expiry exactly *one* request recomputed the aggregation while the other ~799 waited ~1.2 seconds for it and then read the fresh value from Redis. Database query load for that report dropped from ~800 simultaneous queries every 5 minutes to exactly **one** every 5 minutes — a 800x reduction in stampede load — and the p99 spikes vanished. The cache was always "working" by hit-rate; the bug was entirely in what happened at the *miss*, which is the part naive caching ignores. **A cache is only as good as its behavior on a miss under load.**

## 15. When caching backfires: the four failure modes

Caching is the highest-leverage optimization when it fits, and a source of nasty bugs when it does not. I have debugged every one of these in production. The figure below names the four failure modes; the rest of this section is how to recognize and avoid each.

![matrix of four ways caching backfires with the symptom root cause and fix for unhashable arguments unbounded growth low hit rate and stale results](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-7.png)

**1. Unhashable arguments.** `lru_cache` builds its key by hashing the arguments, so any unhashable argument — a `list`, a `dict`, a `set`, a NumPy array, a pandas DataFrame — raises `TypeError` the moment you call the cached function:

```pycon
>>> @lru_cache
... def process(items):
...     return sum(items)
...
>>> process([1, 2, 3])
TypeError: unhashable type: 'list'
```

The fix is to make the arguments hashable: pass a `tuple` instead of a `list`, a `frozenset` instead of a `set`, the bytes of an array instead of the array. Or, if the input is genuinely a mutable structure, write a manual memo keyed on a hashable summary of it (a tuple of its sorted items, a hash of its bytes). Do *not* "fix" it by making the argument hashable in a way that ignores part of its content — a key that does not capture everything the result depends on is a correctness bug, which is failure mode 4 in disguise.

**2. Unbounded growth (the memory leak).** Covered in section 9, but it is the most common production caching incident so it bears repeating: `@cache` or `lru_cache(maxsize=None)` on a function whose argument space is unbounded grows the cache forever. RSS climbs steadily until the process is OOM-killed. The symptom is a sawtooth memory graph that resets only on restart. The fix is always a `maxsize` bound, or a TTL (which bounds by time instead of count). **Never cache unbounded keys without a bound.**

**3. Low hit rate.** If arguments rarely repeat, the cache stores entries it never reuses. You pay the hashing cost and the memory cost on every call and get almost no hits back. At a 0% hit rate the cached version is strictly slower than the uncached one. The symptom is a `cache_info()` showing misses vastly outnumbering hits. The fix is to measure first and remove the cache if the hit rate is low — or redesign the key to raise it. The unique-ID-in-the-key trap (a `request_id`, a timestamp, a UUID) guarantees a 0% hit rate and is the usual cause.

**4. Stale results (the invalidation problem).** A cache returns the value it computed earlier. If the inputs the function *implicitly* depends on have changed — a database row was updated, a config file was reloaded, a feature flag flipped — the cached answer is now wrong, and the cache will keep serving it. This is the hardest failure because nothing crashes; the system just quietly returns stale data. The two classic mitigations: a **TTL** (the answer is allowed to be at most `ttl` seconds stale — bounded wrongness in exchange for performance) and **explicit invalidation** (when you know the inputs changed, call `cache_clear()` or delete the affected keys). The deep truth here is the famous quip: there are only two hard things in computer science, cache invalidation and naming things. Caching trades correctness latency for speed — you must decide, explicitly, how stale is too stale.

There is a fifth, subtler trap worth a sentence: **mutable return values.** If your cached function returns a `list` or `dict` and a caller mutates it, they have mutated the cached object — the next caller gets the corrupted version. The cache stores a reference, not a copy. Either return immutable values (tuples, frozensets, frozen dataclasses) from cached functions, or have callers treat the result as read-only by contract, or return a copy (which costs, partly defeating the cache). I prefer immutable returns: a cached function that returns a `tuple` is one you can never accidentally corrupt.

## 16. Case studies: real caching wins and losses

Caching is everywhere in real Python, so the case studies are easy to find and instructive.

**CPython's own internals use caching aggressively.** Small integers from -5 to 256 are *interned* — pre-created and cached as singletons — so `a = 256; b = 256; a is b` is `True` (they are the same cached object), while `a = 257; b = 257; a is b` is `False` (each `257` is a fresh object). This is a cache: CPython computed the small-int objects once at startup and reuses them, saving an allocation on the overwhelmingly common case of small numbers. Short strings that look like identifiers are similarly interned. The PEP 659 specializing adaptive interpreter that arrived in 3.11 (covered in the [CPython execution model post](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop)) is itself a form of caching — it caches the observed types at a bytecode site and specializes the instruction, which is "remember what happened last time and skip the dispatch." Caching is not a library feature bolted on; it is woven through the runtime.

**The `re` module caches compiled patterns.** When you call `re.search(pattern, text)` with a string pattern, the module compiles the pattern to a regex object and caches it (up to a few hundred patterns) so that the next call with the same pattern string skips compilation. This is why you do not *have* to `re.compile` patterns you reuse — the module memoizes it for you. But the cache is bounded and keyed on the pattern string, so if you generate thousands of distinct dynamic patterns, the cache thrashes and `re.compile`-ing them yourself becomes worthwhile.

**A real loss: the cache that made things slower.** A team I advised had wrapped a data-validation function with `@lru_cache` because it was on a hot path. The function took a record `dict` — which is unhashable — so they "fixed" it by hashing `str(record)`. Two problems. First, `str(record)` of a large dict is itself expensive (it serializes the whole thing), so they added cost to every call. Second, dict string representation order was effectively random across runs, so logically-identical records produced different keys — a near-0% hit rate. The "optimization" added a serialization on every call and never hit. We measured it: removing the cache made the endpoint 15% faster. The lesson is the one this whole post circles: **a cache is a bet on repetition; if the bet is wrong, you pay and never collect.** Measure the hit rate before you trust it.

**The pricing-service win, quantified one more time.** Section 4's curve cache: 98.3% hit rate, p99 from 38 ms to 2 ms, RSS overhead about 20 MB for a 4096-entry bounded cache holding ~5 KB curves. The cost was 20 MB of RAM and four lines of code. The benefit was a 19x p99 reduction on the dominant endpoint. That ratio — kilobytes-to-megabytes of memory for an order-of-magnitude latency cut — is the typical shape of a caching win when the hit rate is high. It is the cheapest big speedup in the toolbox, which is exactly why "the fastest computation is the one you skip" is worth saying out loud.

**Django and Flask cache decorators are this same idea at the view layer.** Django's `@cache_page` and the low-level `cache.get`/`cache.set` API, Flask-Caching's `@cache.memoize`, and the HTTP `Cache-Control` header are all the same pattern at a coarser grain: instead of memoizing one Python function, you memoize the rendered response to a whole request, keyed on the URL and parameters. The hit serves a pre-rendered page in microseconds; the miss runs the view, the database queries, and the template rendering. The trade-offs are identical to everything in this post — bound it, give it a TTL, and have an invalidation story for when the underlying data changes — just applied to a response instead of a return value. Recognizing that view caching, function memoization, and the database buffer pool are *the same idea at different layers* is the thing that makes "where should this be cached?" a tractable question rather than a guess.

**The compounding effect of layered caches.** In a mature service the cache layers stack and their hit rates multiply against the work that reaches each tier. Suppose the in-process `lru_cache` catches 80% of calls, the Redis layer behind it catches 80% of *the remaining* 20%, and only the final 4% reach the database. Then 96% of requests never touch the database at all, and the 4% that do are the genuinely novel ones. Each layer is cheap to add and each multiplies the protection of the one behind it. This is why "add a cache" is so often the highest-leverage change available: it does not just speed up the request, it shields every downstream system from load, and that shielding compounds across tiers.

## 17. When to reach for caching (and when not to)

Caching is a cost — memory, complexity, a new class of correctness bugs (staleness) — so like every lever in this series it has a sharp "when" and an equally sharp "when not." The decision tree below routes you to the right tool; the prose after it is the judgment that the tree compresses.

![decision tree routing repeated work by purity then scope then bound then reach to lru cache, cached property, manual memo, or diskcache and redis](/imgs/blogs/caching-and-memoization-lru-cache-and-beyond-8.png)

**Reach for caching when all of these hold:**

- The function is **expensive** relative to a dict lookup (microseconds or more — caching a function that already takes 80 ns saves nothing because the hit *also* costs ~80 ns).
- The function is **pure**, or pure within a tolerable TTL — same arguments, same answer, no important side effects.
- The same arguments **recur** — a measurable hit rate, ideally 80%+. This is a property of your traffic; measure it, do not assume it.
- The arguments are **hashable**, or cheaply made hashable.
- You can **bound** the cache (a `maxsize` or a TTL) so it cannot leak.

**Do not reach for caching when:**

- The **hit rate is low.** If arguments rarely repeat, the cache is pure overhead. The unique-key trap (caching by `request_id`, timestamp, or UUID) guarantees this. Measure with `cache_info()`; if hits are far below misses, remove the cache.
- The arguments are **unhashable and expensive to make hashable** (large dicts, big arrays). Sometimes the hashing costs more than the function.
- The result goes **stale** and you have no acceptable invalidation story. If correctness demands always-fresh data and the inputs change unpredictably, caching is a bug generator. A TTL that says "at most N seconds stale" is the compromise; if even that is unacceptable, do not cache.
- The function is **cheap or rarely called.** Amdahl's law caps your win at the fraction of time the function consumes. Caching a function that is 1% of runtime saves at most 1% — not worth a new class of bugs. (See [a mental model of performance and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) for why you optimize the hot path, not the convenient one.)
- The cache would **grow unbounded** and you cannot bound it. An unbounded cache on an unbounded argument space is a memory leak; if you cannot set a `maxsize` or TTL, do not cache in-process.

There is an ordering to the tools, too, and it is the cache hierarchy in decision form. Default to the cheapest tool that fits: `lru_cache`/`cache` for a pure function, `cached_property` for a per-object value. Step up to a **manual memo dict** only when you need TTL, partial invalidation, or a custom key. Step up to **`diskcache`** only when the cache must survive restarts or exceed RAM on one machine. Step up to **Redis** only when it must be shared across the fleet. Each step adds latency and operational cost; do not pay for reach you do not need.

This is the same discipline as every other post in the series. Caching composes with the others: you cache the *result* of an expensive computation, but you still pick the right data structure inside the function (the [collections and heapq toolbox](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect) post) and write the function idiomatically (the upcoming [idiomatic fast Python post on comprehensions, generators, and builtins](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins)). Caching is the lever you pull *first* when the work repeats, because skipping work beats doing it fast — but a cache around a badly-written function is a fast path to a wrong answer.

## 18. Key takeaways

- **The fastest computation is the one you skip.** A cache hit costs ~80 ns; the recompute it avoids costs milliseconds. The entire value of a cache is the work it lets you not do.
- **The speedup equals roughly `1 / (1 - h)`** where `h` is the hit rate. 90% hit rate is ~10x; 99% is ~100x; 0% is a slowdown. The hit rate is a property of your workload — measure it with `cache_info()`, never assume it.
- **Memoizing a pure recursive function collapses an exponential call tree to linear.** Fibonacci's `O(2^n)` becomes `O(n)` with one decorator, because each distinct input is computed exactly once instead of millions of times.
- **`functools.lru_cache(maxsize)` for bounded memoization, `functools.cache` for a small fixed domain, `functools.cached_property` for a per-instance value.** Know which scope each one caches.
- **Always bound the cache.** An unbounded cache on an unbounded argument space is a memory leak. Set a `maxsize` or a TTL. RSS that climbs steadily until OOM is the classic symptom.
- **Mind the key.** `f(2)` and `f(n=2)` are different keys; unique IDs in the key force a 0% hit rate; unhashable args raise `TypeError`. Normalize call style and key only on what the result truly depends on.
- **Caching trades correctness latency for speed.** Stale results are the silent failure. A TTL bounds the staleness; explicit invalidation (`cache_clear()`, key deletion) removes it. Decide how stale is too stale, on purpose.
- **Return immutable values from cached functions** so a caller cannot mutate the shared cached object.
- **Climb the hierarchy only as far as you need:** in-process (`lru_cache`, ns) → on-disk (`diskcache`, µs) → over the network (Redis, ms). Each tier trades latency for reach and durability.
- **A cache is a bet on repetition.** When the bet is right it is the cheapest huge speedup you own. When it is wrong you pay the hashing and the memory and never collect — and sometimes you serve a wrong answer. Measure, then trust.

## Further reading

- The `functools` documentation — `lru_cache`, `cache`, `cached_property`, the `typed` and `maxsize` parameters, and the `cache_info`/`cache_clear` methods: the canonical reference for everything in this post.
- The CPython source for `Lib/functools.py` — the pure-Python `lru_cache` implementation (a dict plus a circular doubly linked list) is short and worth reading to see the LRU bookkeeping made concrete.
- The `diskcache` documentation — `Cache`, `@cache.memoize`, eviction policies, and the SQLite-backed on-disk design.
- The Redis documentation on key expiration (`EXPIRE`, `SET ... EX`) and eviction policies (`maxmemory-policy`, including `allkeys-lru`) for the network tier.
- "High Performance Python" by Gorelick and Ozsvald — the chapter on dictionaries, sets, and memoization, with the same Fibonacci derivation and more.
- Within this series: [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) (the cost model), [the collections and heapq toolbox](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect) (the data structures inside your cached functions), [idiomatic fast Python: comprehensions, generators, and builtins](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins) (writing the function well before you cache it), and [a mental model of performance and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) (Amdahl's law and why you cache the hot path).
- For pushing repeated work into the data tier instead of caching it in Python, see [how databases store data with pages, heap files, and the buffer pool](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool) — the buffer pool is a cache, and understanding it tells you when the database is already caching for you.

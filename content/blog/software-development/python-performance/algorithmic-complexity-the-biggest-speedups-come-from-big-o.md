---
title: "Algorithmic Complexity: The Biggest Speedups Come From Big-O"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Before you reach for C, fix your big-O: how an O(n squared) loop becomes O(n) with a set, why the right algorithm beats any micro-optimization, and how profiling leads you straight to the quadratic."
tags:
  [
    "python",
    "performance",
    "optimization",
    "big-o",
    "algorithms",
    "data-structures",
    "profiling",
    "complexity",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-1.png"
---

A junior engineer once shipped a deduplication step into a nightly pipeline that, on the test fixture of ten thousand rows, ran in under a second. Three months later the input had grown to two million rows and the same step took just over four minutes. The on-call rotation got paged because the job blew past its 2 a.m. window and collided with the morning's reporting load. The reflex of half the team was to reach for the heavy machinery — "let's rewrite it in Cython," "let's throw it on the GPU," "let's parallelize it across the cluster." Every one of those would have worked, in the sense of buying some multiple. None of them would have fixed the actual problem, which was that the dedup loop checked `if item not in seen` against a growing **list**, turning a linear pass into a quadratic one. The fix was three characters: change `seen = []` to `seen = set()`. The four-minute step dropped to under two seconds, and at the next input size it would have stayed under two seconds while the list version would have crawled toward an hour.

That is the single most important lesson in all of performance work, and it is why this post opens the "do less work" track of the series. **The biggest speedups almost never come from making each operation faster. They come from doing dramatically fewer operations** — from a better algorithm or a better data structure that changes the *shape* of the cost curve, not its constant factor. A constant-factor trick — compiling, vectorizing, parallelizing — multiplies your speed by some fixed number: 5×, 50×, maybe 200× on a good day. A complexity improvement *changes the exponent*, and at large input sizes that beats any constant by an unbounded margin. Compiling a quadratic algorithm just gives you a faster quadratic algorithm; it still falls off a cliff, just a little further down the road.

![before and after comparison of a nested list scan running in minutes versus a set lookup running in milliseconds on a one million row input](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-1.png)

By the end of this post you will be able to do five concrete things. First, read **Big-O notation as a practical cost model** — translate $O(1)$, $O(\log n)$, $O(n)$, $O(n \log n)$, and $O(n^2)$ into real wall-clock predictions for the input sizes you actually run. Second, **spot a hidden quadratic** in code that looks perfectly innocent, because the most expensive line is usually the boring one. Third, **rewrite an $O(n^2)$ membership loop into $O(n)$** with a set or a dict, and understand exactly *why* the rewrite works at the level of operation counts. Fourth, use `timeit` across a sweep of input sizes to **watch the growth curve with your own eyes** and confirm which complexity class you are actually in. Fifth, know **when a complexity change is the wrong lever** — when $n$ is tiny, when the constant dominates, when the better-big-O structure is slower per operation until a crossover point you can compute.

This is the post that sets up the whole "do less work" track, and it sits directly downstream of two earlier posts: [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), which establishes that Python's per-operation cost is high and therefore *doing fewer operations matters more here than almost anywhere else*, and [a mental model of performance: latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop), which gives us the measure-first discipline and Amdahl's law we will lean on throughout. The motto of the series is *don't guess, measure; rewrite 1% in native, not 100%; always prove the win with a number.* This post adds a prefix to it: **before you rewrite anything in native code, fix your big-O.**

## 1. What Big-O actually measures (and what it ignores)

Let us start by being precise about what Big-O notation is, because a lot of working engineers carry around a fuzzy version of it that leads them astray. Big-O describes how the **number of basic operations** an algorithm performs **grows as a function of the input size** $n$, in the limit as $n$ gets large. It is deliberately a *growth rate*, not a count. When we say an algorithm is $O(n^2)$, we are claiming that for large enough $n$, the operation count is bounded above by some constant times $n^2$. We are explicitly *not* claiming it does exactly $n^2$ operations, and we are throwing away every constant factor and every lower-order term.

That last sentence is the source of both Big-O's power and its blind spots. Big-O throws away constants on purpose, because the whole point is to capture the behavior that *dominates at scale*. An algorithm that does $5n$ operations and one that does $100n$ operations are both $O(n)$, and that is correct: as $n$ grows, the ratio between them stays a flat factor of 20, while an $O(n^2)$ algorithm will eventually pass *both* of them no matter how small its own constant is. The exponent always wins in the end. This is why "the right algorithm beats any constant-factor trick" is not a slogan but a mathematical fact: a constant-factor trick lives entirely in the part of the cost that Big-O discards.

But the blind spots are real and you must respect them. First, **Big-O hides the constant**, and the constant is where micro-optimization, vectorization, and compilation all live. A `set` lookup and a `dict` lookup are both $O(1)$ average, but a dict lookup that also has to hash a long string costs more *per call* than one hashing a small int. Second, **Big-O is about the limit**, so for small $n$ a "worse" Big-O can be faster — insertion sort ($O(n^2)$) beats merge sort ($O(n \log n)$) for arrays of a dozen elements, which is exactly why CPython's `list.sort` (Timsort) uses insertion sort on small runs. Third, **Big-O counts abstract operations, not Python operations**, and in Python the per-operation cost is so high and so variable that the constant can swing the answer for moderate $n$. We will keep all three caveats in view; the whole craft is knowing when the exponent matters more than the constant (usually) and when it does not (small $n$, hot constants).

There are three related notations worth distinguishing once. **Big-O** ($O$) is an upper bound — "grows no faster than." **Big-Omega** ($\Omega$) is a lower bound — "grows no slower than." **Big-Theta** ($\Theta$) is a tight bound — both at once. In casual engineering use people say "Big-O" when they often mean $\Theta$, the tight bound, and that is fine; just know that when someone says a hash lookup is "$O(1)$" they mean *average-case* $\Theta(1)$, and the *worst case* is $O(n)$ if every key collides into the same bucket. The average-vs-worst distinction matters enormously for hash structures and we will return to it.

The most important habit Big-O teaches is to **ask how the cost grows with the input, not how long one run takes.** A function that takes 50 ms on your test data tells you almost nothing. A function whose time *quadruples when you double the input* is screaming "$O(n^2)$" at you, and that is the signal that should make you reach for a structure change. We will make this concrete with a growth-rate table that you should genuinely commit to memory.

## 2. The growth-rate table: what the classes mean at real n

Here is the single most useful artifact in this entire post. The columns are the input sizes you actually encounter — a thousand, a million, a billion — and the rows are the complexity classes. The cells are the approximate operation counts. Read it as: "if my algorithm is in this row, and my input is this column, this is roughly how many basic operations I am asking the machine to do."

| Complexity | n = 10³ | n = 10⁶ | n = 10⁹ | What it feels like |
| --- | --- | --- | --- | --- |
| $O(1)$ | 1 | 1 | 1 | Instant, regardless of size |
| $O(\log n)$ | ~10 | ~20 | ~30 | Effectively free |
| $O(n)$ | 10³ | 10⁶ | 10⁹ | Scales linearly, predictable |
| $O(n \log n)$ | ~10⁴ | ~2×10⁷ | ~3×10¹⁰ | A sort; still very tractable |
| $O(n^2)$ | 10⁶ | 10¹² | 10¹⁸ | Fine small, fatal large |
| $O(2^n)$ | 10³⁰⁰ | astronomical | astronomical | Only tiny n ever |

Now anchor those operation counts in time. A rough, defensible rule of thumb for pure-Python work on a modern CPU is that a tight Python loop does on the order of **10–100 million simple operations per second** — call it $10^7$ for a loop body that does real work (a comparison, an attribute access, a method call), because each iteration pays for the bytecode eval loop, reference counting, and boxed objects, all of which we covered in [why Python is slow](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means). With that number you can do the arithmetic that changes how you write code.

At $n = 10^6$, an $O(n)$ algorithm does $10^6$ operations and takes about $10^6 / 10^7 = 0.1$ seconds. Comfortable. The same $n = 10^6$ run of an $O(n^2)$ algorithm does $10^{12}$ operations, which at $10^7$ ops/sec is $10^{12} / 10^7 = 10^5$ seconds — **about 28 hours.** That is the entire story of this post in two lines of arithmetic. The linear version finishes before you finish reading this sentence; the quadratic version is still running tomorrow afternoon. And there is no constant-factor trick that closes a gap of $10^6$. If you compiled the quadratic loop and made each operation 100× cheaper, you would go from 28 hours to about 17 minutes — still vastly slower than the 0.1-second linear version that ran in plain Python.

![the complexity ladder as stacked layers from O(1) at the bottom through O(log n), O(n), O(n log n), and O(n squared) at the top with operation counts at one million elements](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-3.png)

The ladder figure above is worth internalizing. Each rung up multiplies the work at scale. $O(1)$ and $O(\log n)$ are, for practical purposes, free — a billion-element structure still answers a logarithmic query in about 30 steps. $O(n)$ is the honest baseline: you have to look at every element at least once for most problems, so linear is often the best you can hope for. $O(n \log n)$ is the cost of a comparison sort and is genuinely cheap — the $\log n$ factor adds only about 20× at a million elements, not the million× that the jump to $O(n^2)$ costs. And $O(n^2)$ is the line you are always trying to avoid crossing, because it is the first class where "works on the test data" and "works in production" diverge catastrophically.

#### Worked example: the same code, two input sizes

Suppose you have a function `find_duplicates(records)` that, for each record, scans all the others looking for a match — a classic $O(n^2)$ pattern. On a named reference machine — an **8-core x86-64 Linux box, CPython 3.12, 16 GB RAM** — say the inner comparison plus loop overhead costs about 60 ns, so the algorithm does roughly $n^2$ comparisons at 60 ns each.

At $n = 1{,}000$: $n^2 = 10^6$ comparisons × 60 ns ≈ **0.06 seconds.** Nobody notices.

At $n = 100{,}000$: $n^2 = 10^{10}$ comparisons × 60 ns ≈ **600 seconds**, or 10 minutes.

At $n = 1{,}000{,}000$: $n^2 = 10^{12}$ comparisons × 60 ns ≈ **60,000 seconds**, or about **16.7 hours.**

Notice the multiplier. The input grew by 1,000× (from a thousand to a million), but the runtime grew by $1{,}000^2 = 1{,}000{,}000\times$. That is the signature of a quadratic: **the runtime ratio is the square of the input ratio.** When you double the input and the time roughly quadruples, you have found an $O(n^2)$. When you 10× the input and the time roughly 100×es, same diagnosis. This is the single most reliable way to *measure* your way to the complexity class without ever reading the code — and it is exactly what we will do with `timeit` in section 6.

## 3. The canonical fix: list membership becomes set membership

Now let us do the thing this whole track is about — turn a quadratic into a linear — and derive *why* it works at the level of operation counts, because the derivation is what lets you recognize the pattern everywhere it hides.

Here is the slow version. It is the most common performance bug in all of Python, and you will write it yourself at least once a year for the rest of your career if you are not vigilant:

```python
def find_new_users(all_events, known_user_ids):
    """Return events from users we have never seen before."""
    new_events = []
    for event in all_events:                 # outer loop: O(n)
        if event["user_id"] not in known_user_ids:  # inner: O(m) if a list!
            new_events.append(event)
    return new_events

# The trap is in how known_user_ids was built:
known_user_ids = []                          # <-- a LIST
for row in load_known_users():
    known_user_ids.append(row["user_id"])
```

The loop reads innocently. There is one explicit `for`, no nested loop in sight. But `not in` on a **list** is $O(m)$, where $m$ is the length of the list, because CPython has no choice but to walk the list element by element comparing each one until it finds a match or reaches the end. So the line `if event["user_id"] not in known_user_ids` is itself a hidden loop. The outer loop runs $n$ times (number of events), and inside each iteration the membership test runs up to $m$ comparisons (number of known users). The total is $O(n \times m)$, and when $n$ and $m$ are both proportional to the data size, that is $O(n^2)$.

![dataflow graph showing an outer loop over n rows and an inner scan over n rows feeding a compare operation that produces n squared total comparisons and explodes in wall clock time](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-4.png)

The figure above is the mental picture you want for *every* hidden quadratic: two flows of size $n$ — the outer iteration and the inner scan — meeting at a single comparison node, and the product of their sizes is the operation count that explodes. The outer loop is visible in the source. The inner scan is invisible: it is hiding inside an innocent-looking `in`. This is why quadratics are so easy to ship and so hard to spot by reading. You have to know which operations carry a hidden loop. The big three offenders in Python are `x in some_list`, `some_list.index(x)`, and `some_list.remove(x)` — each is $O(m)$ and each is a quadratic waiting to happen if you put it inside a loop over the data.

The fix is to change the data structure backing the membership test from a list to a **set**:

```python
def find_new_users(all_events, known_user_ids):
    """Return events from users we have never seen before."""
    # known_user_ids is now a set: membership is O(1) average.
    return [e for e in all_events if e["user_id"] not in known_user_ids]

# Build it as a set once, up front:
known_user_ids = {row["user_id"] for row in load_known_users()}  # set comprehension
```

A `set` (and a `dict`) in CPython is a **hash table**. When you test `x in some_set`, Python computes `hash(x)` once, jumps directly to the bucket that hash points to, and checks the (usually one) element there. That is $O(1)$ on average — constant time, independent of how many elements the set holds. The membership test no longer scans; it computes an address and looks. So the inner "loop" collapses from $m$ comparisons to about 1. The total goes from $O(n \times m)$ to $O(n \times 1) = O(n)$.

Let me make the operation counts explicit, because this is the heart of the post. With the list, total comparisons $\approx n \cdot m$. With the set, total operations $\approx n \cdot 1 = n$ (plus a one-time $O(m)$ cost to build the set). The ratio of work between the two versions is:

$$\frac{\text{list work}}{\text{set work}} \approx \frac{n \cdot m}{n + m} \approx m \quad \text{(when } n \approx m\text{)}$$

So the speedup *is* the size of the collection you are testing membership against. At $m = 100$, expect about 100× (minus the set's larger constant). At $m = 10{,}000$, about 10,000×. At $m = 1{,}000{,}000$, about a million× in operation count — though in practice the set's per-operation constant (hashing) and other overheads shave that down, and Python's `in` on a list has a tight C-level inner loop that makes its constant relatively small, so the *measured* speedup is typically somewhat less than the raw ratio. But the crucial fact is that **the speedup grows with the input** — it is not a fixed multiple, it is unbounded. That is precisely what a constant-factor trick can never give you.

It is worth being precise about *why* the hash lookup is $O(1)$ on average, because the argument is the foundation under every set/dict fix in this post. A hash table is an array of $k$ slots. To find `x`, CPython computes `hash(x)`, reduces it modulo $k$ to get a slot index, and looks there. If the slot holds `x`, done — one probe. If the slot is empty, `x` is absent — one probe. The only cost beyond constant time is a **collision**: two distinct keys landing in the same slot, which forces a few extra probes to resolve. The number of collisions is governed by the **load factor** $\alpha = n/k$, the fraction of slots occupied. CPython keeps $\alpha$ below about two-thirds by resizing (roughly doubling $k$) whenever the table fills past that threshold. With $\alpha$ bounded by a constant, the *expected* number of probes per lookup is also a constant — it does not grow with $n$. That is the entire content of "average $O(1)$": bounded load factor implies a bounded expected probe count, independent of how many items the table holds. The worst case is $O(n)$ — if an adversary or a pathological hash function forces every key into one slot — but for ordinary data with Python's well-distributed hashes, you get the constant. This is the machinery that makes the list-to-set swap turn $O(n)$ membership into $O(1)$, and it is why the [data-structure post](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) spends so long on open addressing and load factors.

#### Worked example: 10,000× on a real input

On the reference machine — **8-core x86-64 Linux, CPython 3.12, 16 GB RAM** — consider testing membership for $n = 1{,}000{,}000$ lookups against a collection of $m = 1{,}000{,}000$ items.

**List version.** Each `in` scans on average half the list before finding a hit (or all of it for a miss). With a C-level comparison costing roughly 20 ns and an average of $m/2 = 500{,}000$ comparisons per lookup, one lookup costs about $500{,}000 \times 20\text{ ns} = 10$ ms. A million such lookups: $10^6 \times 10\text{ ms} = 10{,}000$ seconds, or **about 2.8 hours.** (Most people kill the job long before it finishes and conclude "Python is too slow for this" — which is the wrong lesson entirely.)

**Set version.** Building the set is a one-time $O(m)$ cost: a million inserts at maybe 80 ns each ≈ 0.08 s. Then each lookup is one hash plus one bucket probe, roughly 40 ns. A million lookups: $10^6 \times 40\text{ ns} = 0.04$ s. Total: about **0.12 seconds.**

The before→after: **2.8 hours to 0.12 seconds — roughly a 84,000× wall-clock speedup**, from a three-character change. Even being conservative and assuming the list scan stops early on average and the constants are kinder, you land somewhere between a few thousand× and tens of thousands×. The headline "10,000×" is not hyperbole; it is the *order of magnitude* you genuinely get when you replace a list-membership-in-a-loop with a set on a million-element input. And — this is the point — there is no amount of Cython, NumPy, or multiprocessing you could apply to the list version that would close that gap, because they all attack the 20 ns constant, and the problem is the $500{,}000$ factor.

| | List membership | Set membership |
| --- | --- | --- |
| Structure | dynamic array | hash table |
| `x in s` complexity | $O(m)$ | $O(1)$ average |
| Per-lookup work at m=1M | ~500,000 comparisons | ~1 probe |
| Build cost | $O(m)$ (just a list) | $O(m)$ (hash each item) |
| Total for 1M lookups | ~2.8 hours | ~0.12 seconds |
| Extra memory | baseline | ~2–3× (load factor) |
| When it wins | tiny m, or order matters | any repeated membership |

The only real cost of the set is memory: a hash table keeps its load factor below a threshold (CPython resizes to keep it under about two-thirds full), so a set of $m$ items uses roughly two to three times the bytes of a bare list of the same items. That is almost always a trade you take gladly — turning hours into milliseconds is worth a 2–3× memory bump on a structure that is usually a small fraction of your total footprint. We dig into exactly how the open-addressing hash table achieves $O(1)$, what the load factor is, and when it degrades, in the sibling post [choosing the right built-in data structure: list, dict, set, tuple](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple).

## 4. Why the right algorithm beats any constant-factor trick

This is the section that justifies the post's place at the *front* of the optimization track, before vectorizing, before compiling, before parallelizing. The claim is strong and I want to defend it rigorously: **a complexity improvement dominates any constant-factor improvement for large enough $n$, and "large enough" is usually smaller than your production input.**

The math is one inequality. Suppose you have an $O(n^2)$ algorithm with a small constant $c_2$, and you are tempted to make each operation $k$ times faster (that is what compiling or vectorizing the inner loop does — it shrinks the constant, not the exponent). Your tuned algorithm costs $\frac{c_2}{k} n^2$. Compare it to switching to an $O(n)$ algorithm with constant $c_1$, costing $c_1 n$. The linear algorithm wins whenever:

$$c_1 n < \frac{c_2}{k} n^2 \quad\Longleftrightarrow\quad n > \frac{c_1 k}{c_2}$$

Read that crossover point: the linear algorithm beats the *k-times-accelerated* quadratic for all $n$ above $\frac{c_1 k}{c_2}$. The speedup factor $k$ from your constant-factor trick appears *linearly* in the crossover — it just pushes the crossover point out a bit. Even a heroic $k = 1000$ (a thousand-fold per-operation speedup, which is more than most native rewrites deliver) only moves the crossover by a factor of 1000. If the un-accelerated crossover was at $n = 100$, the accelerated one is at $n = 100{,}000$ — still smaller than a million-row production input. Past that point, the plain-Python linear algorithm beats the heavily-optimized quadratic, and the margin keeps widening forever.

![before and after comparison showing that micro-optimizing the inner line of a quadratic algorithm caps the win at a small constant factor while changing the data structure to a set yields an unbounded win that grows with n](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-5.png)

The figure crystallizes the choice. On the left, you tune the inner line of a quadratic — rewrite it in C, vectorize it, whatever — and you get *at most* a fixed multiple, then you hit a wall, because the algorithm still does $n^2$ work. On the right, you change the structure and the curve itself bends down to linear, and the win grows without bound as the data grows. The right-hand path is strictly better at scale, and it is usually *less* code. This is why the discipline is: **fix the big-O first, then — only if you still need more — reach for the constant-factor levers** (vectorize, compile, parallelize) on the now-linear algorithm. Those later levers are real and powerful, and the rest of this series is about them. But they are multipliers on whatever complexity you are already paying. Multiplying a quadratic is throwing good money after bad.

There is a beautiful corollary here that surprises people: **vectorizing or compiling a bad algorithm can lose to plain Python with a good algorithm.** Suppose someone is proud that they vectorized the $O(n^2)$ pairwise-comparison loop with NumPy broadcasting, getting a 50× constant-factor win. At $n = 100{,}000$ the vectorized quadratic does $10^{10}$ operations at, say, 1 ns each (NumPy speed) = 10 seconds, and it allocates an $n \times n$ array — $10^{10}$ elements × 8 bytes = **80 GB**, which immediately OOM-kills the process. Meanwhile the plain-Python $O(n)$ set-based version does $10^5$ operations at 100 ns each = 0.01 seconds in a few megabytes. The "optimized" version is both slower *and* crashes on memory. The lesson is not that NumPy is bad — it is essential and we spend a whole track on it — but that **NumPy applied to a quadratic algorithm gives you a fast, memory-hungry quadratic, which is still a quadratic.** Get the complexity right first.

## 5. Amdahl revisited: the algorithm fix is usually the dominant term

We met Amdahl's law in [the mental-model post](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop): if a fraction $p$ of your runtime is spent in the part you speed up by a factor $s$, the overall speedup is

$$S = \frac{1}{(1 - p) + \frac{p}{s}}$$

The law's usual lesson is sobering — it caps your speedup at $\frac{1}{1-p}$ no matter how large $s$ gets, so optimizing a 5%-of-runtime function buys at most 5%. But here is the part people miss: **a hidden quadratic does not stay a small fraction of runtime — it grows to dominate it.** That changes the Amdahl arithmetic completely, and in your favor.

Think about what happens as the input grows. Suppose at small $n$ your pipeline spends 30% of its time in a linear loading step, 60% in a linear transform, and 10% in a quadratic dedup. At small $n$ the dedup is only $p = 0.1$ of the runtime, and Amdahl says fixing it can buy at most $\frac{1}{1 - 0.1} \approx 1.11\times$ — barely worth it. But the dedup is $O(n^2)$ while everything else is $O(n)$. Double the input and the linear parts double while the quadratic part *quadruples*. Keep growing and the $n^2$ term overwhelms the $n$ terms: at large $n$ the dedup is not 10% of runtime, it is 95% or 99%. Now $p \approx 0.99$ and fixing it buys up to $\frac{1}{1 - 0.99} = 100\times$.

This is why **at production scale, the quadratic is almost always the dominant term**, which means fixing it is almost always the single highest-leverage change available. Amdahl's law, applied with the *growth rates* in mind, tells you to hunt for the term with the worst complexity, because that is the term whose fraction $p$ is heading toward 1 as your data grows. The profiler shows you where the time is *now*; the complexity analysis tells you where the time is *going* as the input scales. Put them together and you stop being surprised by the 2 a.m. page.

#### Worked example: how the dominant term flips with scale

On the reference machine, a pipeline has two stages. Stage A (parse and clean) is $O(n)$ and costs about 5 µs per row. Stage B (a dedup that uses `list.index`) is $O(n^2)$ and costs about 50 ns per comparison.

At $n = 10{,}000$ rows: Stage A = $10^4 \times 5\text{ µs} = 0.05$ s. Stage B = $10^8 \times 50\text{ ns} = 5$ s. Already B is 99% of runtime, but the whole job is 5 s and nobody cares.

At $n = 200{,}000$ rows: Stage A = $2 \times 10^5 \times 5\text{ µs} = 1$ s. Stage B = $4 \times 10^{10} \times 50\text{ ns} = 2{,}000$ s ≈ **33 minutes.** The job blew its window. Stage B is now 99.95% of runtime.

Fix Stage B with a set: it becomes $O(n)$ at, say, 200 ns per item, so $2 \times 10^5 \times 200\text{ ns} = 0.04$ s. New total: about 1.04 s, down from 33 minutes — **a roughly 1,900× speedup on the whole job**, almost all of it from the one structure change. Amdahl predicted this: with $p = 0.9995$ and $s$ enormous, $S \approx \frac{1}{1 - 0.9995} = 2{,}000$. The constant-factor levers on Stage A could never have done this; even making Stage A free would have saved 1 second out of 33 minutes.

## 6. Letting the profiler lead you to the quadratic

You should not *guess* that you have a quadratic; you should *measure* it. There are two complementary measurements, and you want both in your toolbox.

The first is the **profiler**, which we covered in depth in [CPU profiling: cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path). A profiler tells you *which function* owns the time on a single representative input. The signature of a quadratic in a `cProfile` report is a function with a huge `tottime` (time in the function itself, not its callees) and a call count that is suspiciously large relative to the input size — or a built-in like `list.index` or the `in` operator showing up with an enormous cumulative time. Here is what running it looks like:

```bash
python -m cProfile -s tottime pipeline.py 2>&1 | head -15
```

```pycon
         50000123 function calls in 41.832 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   38.910   38.910   41.201   41.201 pipeline.py:22(find_duplicates)
   100000    1.840    0.000    1.840    0.000 {method 'index' of 'list' objects}
   100000    0.510    0.000    0.510    0.000 pipeline.py:9(normalize)
        1    0.290    0.290   41.832   41.832 pipeline.py:1(<module>)
```

Two things jump out. `find_duplicates` owns 38.9 of the 41.8 seconds (93% of runtime — Amdahl says fix *this*), and right below it `list.index` is called 100,000 times. The 100,000 calls to `index`, each scanning a growing list, *is* the quadratic. The profiler led you straight to it: one function, one built-in, almost all the time. That is the moment you stop and think "what is the access pattern, and which structure makes it $O(1)$?"

But the profiler on a single input can't tell you the *complexity class* — it tells you the time at one size. To confirm you are quadratic and not just slow, you run the same code at a **sweep of input sizes** and watch the growth curve. This is `timeit` used as a science instrument:

```python
import timeit
import random

def membership_list(haystack, needles):
    s = list(haystack)                     # a LIST
    return sum(1 for x in needles if x in s)

def membership_set(haystack, needles):
    s = set(haystack)                      # a SET
    return sum(1 for x in needles if x in s)

print(f"{'n':>10} {'list (s)':>12} {'set (s)':>12} {'ratio':>10}")
for n in (1_000, 2_000, 4_000, 8_000, 16_000):
    haystack = list(range(n))
    needles  = [random.randrange(n * 2) for _ in range(n)]  # ~half miss
    # number=1 because the list version is already expensive; repeat=3, take the min
    t_list = min(timeit.repeat(lambda: membership_list(haystack, needles),
                               number=1, repeat=3))
    t_set  = min(timeit.repeat(lambda: membership_set(haystack, needles),
                               number=1, repeat=3))
    print(f"{n:>10} {t_list:>12.4f} {t_set:>12.6f} {t_list / t_set:>10.0f}")
```

On the reference machine this prints something close to:

```pycon
         n     list (s)      set (s)      ratio
      1000       0.0091     0.000119         76
      2000       0.0363     0.000231        157
      4000       0.1448     0.000470        308
      8000       0.5790     0.000951        609
     16000       2.3187     0.001889       1228
```

Look at the `list` column. Each time $n$ **doubles**, the list time roughly **quadruples**: 0.009 → 0.036 → 0.145 → 0.579 → 2.319. That 4× per 2× of input is the unmistakable fingerprint of $O(n^2)$. Now look at the `set` column: each doubling of $n$ roughly **doubles** the time (0.000119 → 0.000231 → 0.000470 → ...) — that 2× per 2× is $O(n)$. And the ratio column is the punchline: the speedup *grows* with $n$ (76 → 157 → 308 → 609 → 1228), exactly as the $\frac{nm}{n+m} \approx m$ analysis predicted. Extrapolate the ratio to $n = 1{,}000{,}000$ and you are firmly in five-figure-speedup territory — the "10,000×" headline.

![timeline of the workflow from profile the pipeline to time grows faster than n to spot the quadratic loop to swap list for a set to re-measure and confirm O(n)](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-6.png)

The timeline above is the workflow you run every time you suspect a complexity problem: **profile** to find the function that owns the time, **observe** that its time grows faster than the input, **identify** the quadratic operation (the `in` on a list, the `.index`, the nested loop), **swap** in the structure that makes that operation $O(1)$ or $O(\log n)$, and **re-measure** to confirm the growth curve is now linear and the win is real. The re-measure step is non-negotiable; it is how you prove the change worked and catch the case where you fixed the wrong thing. A note on the `timeit` itself: we use `number=1, repeat=3` and take the `min` because the list version is already slow enough that a single run is a fine measurement, and the minimum of a few repeats filters out noise from the OS scheduler and the garbage collector — the benchmarking discipline that [benchmarking Python correctly](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) treats in full.

## 7. The other workhorses: sort-then-scan, bisect, and Counter

The set/dict fix handles the most common quadratic — membership — but the "do less work" toolkit has three more moves that turn quadratics into $O(n \log n)$ or $O(n)$. Each maps to a recognizable problem shape.

![matrix mapping common problem patterns like dedupe, repeated lookup, nearest in sorted, and top k to their naive quadratic complexity and the better data structure with the improved complexity](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-7.png)

The cheat-sheet figure above is the one to keep next to your keyboard. Four patterns, each with a naive quadratic form and a standard structure that fixes it. Let us take the non-set ones in turn.

**Sort, then scan — when you need order or adjacency.** A huge class of "compare every pair" problems collapses if you sort first. Finding whether any two elements are within some threshold of each other is $O(n^2)$ if you check all pairs, but if you sort ($O(n \log n)$) and then make a single linear pass comparing only *adjacent* elements, it is $O(n \log n + n) = O(n \log n)$. The principle is general: **sorting costs $O(n \log n)$ once and then unlocks $O(n)$ or $O(\log n)$ operations that would otherwise be $O(n)$ or $O(n^2)$ each.** Why does $O(n \log n)$ sort + $O(n)$ scan beat $O(n^2)$? Because $n \log n + n \approx n \log n$, and at $n = 10^6$ that is $2 \times 10^7$ operations versus the $10^{12}$ of the quadratic — a 50,000× reduction in work. The sort pays for itself the instant it lets you avoid even one full quadratic pass.

**`bisect` — binary search on a sorted list.** Once a list is sorted, you never scan it linearly again. The `bisect` module does binary search in $O(\log n)$: `bisect.bisect_left(sorted_list, x)` finds where `x` would be inserted, in about 20 steps for a million-element list instead of 500,000. This is the go-to for "find the nearest value," "count how many are below a threshold," or "look up which bucket a value falls into" against a sorted reference array.

```python
import bisect

# A sorted list of tier thresholds; find each customer's tier.
thresholds = [0, 100, 500, 1_000, 5_000, 10_000]   # sorted ascending
tier_names = ["free", "bronze", "silver", "gold", "platinum", "diamond"]

def tier_for(spend):
    # bisect_right returns the index of the first threshold > spend
    i = bisect.bisect_right(thresholds, spend) - 1
    return tier_names[i]

print(tier_for(0))     # free
print(tier_for(750))   # silver
print(tier_for(9_999)) # platinum
```

That `tier_for` is $O(\log n)$ per call. The naive alternative — a `for` loop scanning the thresholds, or worse, a chain of `if/elif` — is $O(n)$ per call, and if you call it once per row over a million rows against a long threshold list, the difference is real. The catch with `bisect`: the list must be **kept sorted**, and inserting into a sorted list with `bisect.insort` is $O(n)$ because the underlying list must shift elements. So `bisect` shines for *lookups against a mostly-static sorted array*, not for a structure with heavy insertion churn — for that you want a different structure (a balanced tree or just a dict/set if you do not need order).

**`Counter` — counting occurrences.** The "how many times does each item appear?" problem is quadratic if you do it the naive way (for each unique item, scan the whole list counting matches: $O(n^2)$). `collections.Counter` does it in a single $O(n)$ pass using a dict under the covers:

```python
from collections import Counter

# Naive O(n^2): for each unique word, count its occurrences by scanning.
# DON'T do this:
def word_counts_slow(words):
    return {w: words.count(w) for w in set(words)}  # words.count is O(n)!

# Fast O(n): one pass, hash each word once.
def word_counts_fast(words):
    return Counter(words)

# Counter also gives you top-k for free, in O(n log k):
text = "the cat sat on the mat the cat ran".split()
print(Counter(text).most_common(2))   # [('the', 3), ('cat', 2)]
```

The slow version's trap is `words.count(w)`, which is $O(n)$, called once per unique word — another hidden quadratic, the exact same shape as the membership bug. `Counter(words)` hashes each word once and increments a dict counter: one linear pass, done. And `.most_common(k)` uses a heap internally to get the top $k$ in $O(n \log k)$ rather than fully sorting all the counts in $O(n \log n)$ — the **top-k** pattern from the cheat sheet, which `heapq.nlargest` also gives you directly. We cover `deque`, `Counter`, `heapq`, and `bisect` as a complete toolkit in the dedicated [collections and heapq](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) discussion, but the meta-point is consistent: **for every common quadratic, there is a built-in structure that the CPython core developers already wrote in C to make it linear or log-linear. Your job is to recognize the pattern and reach for the structure.**

## 8. The crossover point: when a "worse per-op" structure still wins

A set lookup is $O(1)$ but it is not *free* — it costs more *per operation* than a list scan of length one, because it has to compute a hash. This is the subtlety that the growth-rate table hides, and it is where engineers who half-understand Big-O make their mistakes. The honest statement is: **a structure with better Big-O but a larger constant only wins past a crossover point in $n$.** Below that point, the "worse" structure is actually faster. Knowing where the crossover is keeps you from over-engineering tiny inputs.

Here is the model. A list membership test costs about $c_L \cdot m$ where $c_L$ is the (small, C-level) per-comparison cost and $m$ is the list length. A set membership test costs about $c_S$, a constant that includes hashing — larger than a single comparison but independent of $m$. The set is faster when:

$$c_S < c_L \cdot m \quad\Longleftrightarrow\quad m > \frac{c_S}{c_L}$$

If hashing costs about 5× a single comparison ($c_S \approx 5 c_L$), then the crossover is at $m \approx 5$: for collections of five or fewer elements, scanning a list is actually *faster* than building and probing a set, because you avoid the hash. This is why micro-optimizing tiny collections into sets is pointless or even counterproductive — and why it does not matter, because at $m = 5$ everything is fast anyway. The crossover analysis tells you the structure change matters *exactly when $m$ is large*, which is exactly when you have a performance problem. The two coincide, which is convenient.

| Collection size m | List `in` (scan) | Set `in` (hash) | Winner |
| --- | --- | --- | --- |
| 5 | ~5 comparisons | hash + 1 probe | list (barely) |
| 50 | ~50 comparisons | hash + 1 probe | set |
| 1,000 | ~1,000 comparisons | hash + 1 probe | set (big) |
| 1,000,000 | ~1,000,000 comparisons | hash + 1 probe | set (enormous) |

The same crossover logic applies to the classic $O(n^2)$-vs-$O(n \log n)$ sort question. Insertion sort is $O(n^2)$ with a tiny constant; merge/Timsort is $O(n \log n)$ with a larger one. For $n < \sim 64$, insertion sort wins, which is why CPython's `list.sort` runs insertion sort on small runs and only switches to merging above a threshold. The lesson generalizes: **the asymptotically better algorithm is not always the practically faster one for small inputs, and good library code picks the right one based on size.** When you write your own, the rule of thumb is: if $n$ might ever be large in production, choose the better Big-O even if it loses on your small test fixture — because the test fixture is not where the 2 a.m. page comes from. But do not contort tiny, fixed-size, hot inner collections into fancy structures; for $n$ that is provably small and bounded, the constant is what you optimize, and a plain list or a few `if` statements may genuinely be fastest.

#### Worked example: when NOT to switch to a set

On the reference machine, suppose a hot function checks membership against a *fixed* set of 4 allowed status codes, called 50 million times.

**Tuple/list version.** `if status in (200, 201, 204, 206)` scans up to 4 elements. Each `in` against a 4-tuple costs about 30 ns. Total: $5 \times 10^7 \times 30\text{ ns} = 1.5$ s.

**Set version.** `if status in {200, 201, 204, 206}` hashes `status` then probes — about 35 ns, *slightly slower per call* because of the hash, when the collection is this tiny. Total: $5 \times 10^7 \times 35\text{ ns} = 1.75$ s.

Here the set is marginally *slower* (1.75 s vs 1.5 s) because $m = 4$ is below the crossover. The structure change that wins by 10,000× at $m = 10^6$ *loses* slightly at $m = 4$. (There is a separate, real gotcha: writing `status in {200, 201, 204, 206}` as a literal inside a hot loop rebuilds the set every iteration unless the compiler can hoist it — CPython does constant-fold a frozenset literal here, but a set built from a variable would not be. When in doubt, build the constant collection *once* outside the loop.) The takeaway: **Big-O tells you the structure change matters at scale; the crossover tells you it is a wash or worse at tiny sizes. Apply the fix where $n$ is big, and don't bother where it's small.**

## 9. An end-to-end case: profiling a pipeline to its quadratic

Let us run the whole loop on one realistic problem, the kind that actually pages you. The running example for this series is a data-processing pipeline: load a few million rows, clean them, transform them, aggregate. Here is a version of the clean/enrich stage that someone wrote to "flag returning customers" — events from a user who appears in a known-fraud list and also appears more than once in the batch. It works on the test fixture and is slow in production. We will profile it, find the quadratic (there are two), and fix them.

```python
def enrich_events(events, fraud_user_ids):
    """Tag events from known-fraud users and count repeat offenders."""
    flagged = []
    for event in events:
        uid = event["user_id"]
        # BUG 1: membership test against a LIST -> O(n) each
        is_fraud = uid in fraud_user_ids
        # BUG 2: count occurrences by scanning the whole batch -> O(n) each
        repeat_count = sum(1 for e in events if e["user_id"] == uid)
        if is_fraud or repeat_count > 1:
            flagged.append({**event, "repeat_count": repeat_count})
    return flagged
```

Two hidden loops live inside that single visible `for`. `uid in fraud_user_ids` scans the fraud list ($O(m)$). The generator `sum(1 for e in events if ...)` scans the *entire batch* on every iteration ($O(n)$). The outer loop runs $n$ times, so the total is $O(n \cdot (m + n)) = O(n^2)$ when $m$ and $n$ are comparable. On the test fixture of 5,000 events it runs in about a tenth of a second; on a production batch of 500,000 events it does not finish in the maintenance window.

Step one is to profile, not to guess. We run `cProfile` on a medium-sized input — large enough to make the hot path obvious, small enough to finish:

```bash
python -m cProfile -s tottime enrich.py 2>&1 | head -12
```

```pycon
         100050003 function calls in 58.221 seconds

   Ordered by: internal time

   ncalls   tottime  percall  cumtime  percall filename:lineno(function)
        1    41.770   41.770   58.190   58.190 enrich.py:3(enrich_events)
 50000000     9.120    0.000    9.120    0.000 enrich.py:11(<genexpr>)
   100000     6.840    0.000   16.500    0.000 {built-in method builtins.sum}
        1     0.031    0.031   58.221   58.221 enrich.py:1(<module>)
```

The profile is unambiguous. `enrich_events` and the generator expression inside it own essentially all 58 seconds, and look at the call count on the generator: 50 million calls for a 100,000-row input. That ratio — calls growing like the *square* of the input — is the fingerprint. The `sum(...)` line is the dominant quadratic. The membership test against the fraud list is the secondary one (it hides inside `enrich_events`' own `tottime`). Profiling led us straight there, exactly as the workflow timeline in section 6 promised: profile, see the time concentrated in one function, see the call count exploding faster than $n$, name the quadratic.

Step two is the fix, and it is two structure changes. Make the fraud list a `set` (membership $O(1)$), and replace the per-row rescan with a single `Counter` pass computed once before the loop (counting $O(n)$ total instead of $O(n)$ per row):

```python
from collections import Counter

def enrich_events_fast(events, fraud_user_ids):
    fraud = set(fraud_user_ids)                      # O(m) once, then O(1) lookups
    counts = Counter(e["user_id"] for e in events)   # O(n) once, all counts at hand
    flagged = []
    for event in events:
        uid = event["user_id"]
        repeat_count = counts[uid]                   # O(1) dict get
        if uid in fraud or repeat_count > 1:         # O(1) set lookup
            flagged.append({**event, "repeat_count": repeat_count})
    return flagged
```

Both quadratics are gone. Building the set is $O(m)$, building the `Counter` is one $O(n)$ pass, and the main loop is now $O(n)$ with two $O(1)$ operations inside it. The whole function is $O(n + m) = O(n)$. The logic is identical — same output, same flags, same counts — but the shape of the cost curve has fundamentally changed.

#### Worked example: the pipeline before and after

On the reference machine — **8-core x86-64 Linux, CPython 3.12, 16 GB RAM** — here is the before→after across input sizes, which is the table you would put in the post-mortem:

| Batch size n | `enrich_events` (O(n²)) | `enrich_events_fast` (O(n)) | Speedup |
| --- | --- | --- | --- |
| 5,000 | 0.12 s | 0.004 s | 30× |
| 50,000 | 12.4 s | 0.038 s | 326× |
| 100,000 | 58.2 s | 0.078 s | 746× |
| 500,000 | ~24 min | 0.41 s | ~3,500× |
| 1,000,000 | ~97 min (est.) | 0.83 s | ~7,000× |

Read the speedup column: it *grows* with the input, from 30× at 5,000 rows to roughly 7,000× at a million — the unmistakable signature of trading an exponent for a constant. The 500,000-row production batch went from blowing the maintenance window (24 minutes) to finishing in under half a second. And note what we did *not* do: we did not reach for Cython, NumPy, multiprocessing, or a faster machine. We changed two data structures. The total diff was four lines. *Now*, if 0.83 seconds at a million rows were still too slow, the rest of the leverage ladder is available — but the algorithm fix alone bought three to four orders of magnitude, and it had to come first, because every later lever would have been multiplying a quadratic.

One honest caveat about the table: the entries below the 100,000 row mark for the quadratic column are estimates extrapolated from the measured curve, because nobody actually waits 97 minutes to confirm a number they can compute. When you report a quadratic's runtime at a size you did not run, say so — "extrapolated from the $O(n^2)$ fit" — rather than implying you measured it. The whole point of recognizing the complexity class is that you *can* extrapolate reliably: a quadratic's time at $10\times$ the input is $100\times$ the time, no measurement required. That predictive power is one of the most practical things Big-O gives you.

## 10. Space complexity: the other axis

Everything so far has been about *time* complexity, but the same Big-O machinery describes *memory*, and ignoring the space axis is how you trade a slow program for a crashing one. The set fix that turns hours into milliseconds costs you $O(m)$ extra memory for the hash table; usually trivial, occasionally fatal. The vectorized-quadratic disaster from section 4 — the 80 GB allocation — was a *space* complexity blowup, $O(n^2)$ memory, even though its time was fine. You have to watch both axes.

The most important space-complexity move in the "do less work" toolkit is **streaming**: processing data in a single pass with $O(1)$ or $O(\log n)$ memory instead of materializing the whole thing. A generator expression that yields one transformed row at a time uses constant memory; the equivalent list comprehension materializes all $n$ rows at once, $O(n)$ memory. When the dataset does not fit in RAM, the streaming version is not just more efficient, it is the difference between running and OOM-killing. Consider summing a transformed column over a billion rows:

```python
# O(n) memory: builds the entire list, then sums it. Can OOM on huge inputs.
total = sum([transform(row) for row in read_rows()])   # note the [ ]

# O(1) memory: streams one value at a time, never materializes the list.
total = sum(transform(row) for row in read_rows())     # generator, no [ ]
```

The only difference is the square brackets — the list comprehension versus the generator expression — and on a billion rows it is the difference between a 24 GB peak and a few kilobytes. The time complexity is identical ($O(n)$ for both); the space complexity differs by a factor of $n$. This is the streaming discipline, and it composes with everything: `itertools` chains, `map`/`filter`, and generator pipelines let you process arbitrarily large inputs in bounded memory. We treat the memory hierarchy and zero-copy techniques in depth later in the series, but the headline for this post is: **space complexity is a first-class axis. When time is fine but memory is the constraint, optimize the space Big-O — usually by streaming instead of materializing.**

There is a genuine tension to manage. Some time-complexity wins *cost* space: memoization caches results ($O(\text{distinct inputs})$ memory) to avoid recomputation; an index trades memory for $O(1)$ lookups; the set trades memory for fast membership. The right call depends on which resource is scarce. If you are time-bound and have RAM to spare, build the index. If you are memory-bound, stream and accept a little recomputation. The skill is naming *which* resource is the bottleneck — the same measure-first discipline, applied to RSS instead of wall-clock — and then optimizing the Big-O on that axis. A profiler like `memray` or `tracemalloc` (covered in the [memory-profiling post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks)) tells you the peak, the same way `cProfile` tells you the time.

## 11. Case studies and real-world numbers

These patterns are not academic. Some of the most-cited performance stories in the Python ecosystem are, at their core, complexity fixes wearing different clothes.

**The accidental `O(n^2)` in pandas `iterrows` and string concatenation.** A pervasive real-world quadratic is building a string by repeated concatenation inside a loop: `result += chunk` in a loop over $n$ chunks is $O(n^2)$, because Python strings are immutable, so each `+=` allocates a new string and copies all the characters accumulated so far. Build a list and `"".join(parts)` instead and it is $O(n)$. The CPython interpreter has a special-case optimization that *sometimes* makes `s += chunk` behave like an in-place append for the common case of a single reference, but it is fragile and not guaranteed across implementations — `str.join` on a list is the portable, always-linear answer. The same shape appears in pandas: growing a DataFrame by `df = pd.concat([df, new_row])` inside a loop is $O(n^2)$ because each concat copies the whole frame; collect rows in a list and build the frame once at the end for $O(n)$.

**Polars and DuckDB versus a quadratic pandas join.** When people report "Polars was 100× faster than pandas," part of that is Polars' native, multi-threaded, columnar engine (a constant-factor win we cover in the vectorization track). But a meaningful share of dramatic real-world wins come from the new tool *also* using a better algorithm — a hash join ($O(n + m)$) where the old code did an implicit nested-loop join ($O(n \times m)$), or a sort-merge where the old code rescanned. The honest framing: the engine rewrite gives you a constant factor; the algorithm difference (hash join vs nested loop) gives you a complexity factor, and on a large join the complexity factor is where the surprising headline numbers come from. Push joins and group-bys into a system (Polars, DuckDB, or the database — see the [database series](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool) for engine internals) that uses hash/sort-based algorithms instead of writing the join yourself in a Python loop.

**Ruff, the Rust linter, and "it's not just Rust."** Ruff is often quoted as 10–100× faster than the older Python linters it replaces, and the Rust rewrite is real and matters. But the maintainers have been explicit that a large part of the win is *architectural*: caching, avoiding redundant re-parsing, and data structures that avoid repeated work — algorithmic improvements that would have helped in any language. This is the recurring truth: when a tool gets dramatically faster, look closely and you will usually find a complexity improvement hiding alongside the language change, and the complexity improvement is doing more of the work than people credit. **Native code multiplies; better algorithms change the exponent.**

**Catastrophic regex backtracking — the quadratic you cannot see.** Not every blowup is a loop you wrote by hand. A regular expression like `(a+)+$` matched against a long run of `a` characters that ends without satisfying the pattern can take *exponential* time, $O(2^n)$, because a naive backtracking engine — which is what Python's built-in `re` module uses — tries every possible way to partition the input among the nested quantifiers before it gives up. This is the mechanism behind real ReDoS (regular-expression denial-of-service) incidents: a single innocuous-looking thirty-character request string pins a CPU core for minutes, and a handful of them take down a service. It has caused well-documented production outages, including a 2016 Stack Overflow outage and a 2019 Cloudflare global outage, each traced to one pathological pattern. The fix is the same shape as every other story in this section — a better algorithm, not a faster loop. Rewrite the pattern to remove the nested quantifier (`a+$` instead of `(a+)+$`), anchor it, or use an atomic group or possessive quantifier where the engine supports it; or switch to a guaranteed-linear engine like Google's RE2 (reachable from Python via the `re2`/`pyre2` package) or Rust's `regex` crate, both of which refuse to backtrack and so match in $O(n)$ no matter how adversarial the input. Profiling rarely catches this in advance, because it only fires on hostile input — but when it fires, no amount of constant-factor tuning will save you. The exponent has to go.

**The "Faster CPython" gains are constant-factor — and that is the point.** CPython 3.11 and 3.12, via the PEP 659 specializing adaptive interpreter, made typical Python code roughly 10–60% faster — real, free, and welcome. But notice the magnitude: tens of percent, a constant factor on the *interpreter*, not a change in your program's complexity. The interpreter team can shave the per-bytecode cost; they cannot turn your $O(n^2)$ into $O(n)$. Only you can do that, by picking the right structure. This is the cleanest possible illustration of the hierarchy: the entire heroic multi-year effort of the world's best interpreter engineers buys a constant factor, while a three-character `[]`→`set()` change in your code buys an exponent. Spend your attention accordingly.

## 12. When the complexity lens is the wrong lens

Every technique in this series is a cost, and intellectual honesty requires saying plainly when chasing Big-O is *not* the move. The complexity lens is the right one most of the time at scale, but here is when to set it down.

**When $n$ is provably small and bounded.** If a collection is guaranteed to hold at most a handful of elements — config flags, a fixed enum, the columns of a table — the complexity class is irrelevant. A linear scan of five items is instant, and dressing it up in a hash structure adds code and a hash cost for no benefit. Optimize the constant (or, more likely, optimize *nothing* and move on) when $n$ is small and stays small.

**When you are already linear and need more.** Once you have killed the quadratics and your algorithm is $O(n)$ or $O(n \log n)$, the complexity lens has largely done its job. You usually cannot beat linear for problems that must touch every element. *Now* is when the constant-factor levers — vectorize with NumPy, compile with Numba or Cython, parallelize across cores — become the right tools, because they attack the constant on an algorithm whose exponent is already good. This is the whole reason the series is ordered as it is: algorithm first, then bulk operations, then native code, then parallelism. Each lever is most effective once the previous one has done its part.

**When the bottleneck is I/O, not computation.** If your function spends 95% of its wall-clock waiting on a database, a network call, or the disk, then making the in-memory algorithm asymptotically faster optimizes the 5% — Amdahl caps your win at 5%. The leverage is in the I/O: batch the queries, cache the results, overlap the waits with `asyncio` or threads, push the work into the database. Profile to confirm *where* the time goes before assuming it is your algorithm; sometimes the quadratic is real but it is a quadratic number of *network round trips* (the classic N+1 query problem), and the fix is a single batched query, which is itself a complexity fix at the I/O layer.

**When memory, not time, is the constraint.** A better-time-complexity solution sometimes costs more memory (the set's load factor, a precomputed index, a memoization cache). If you are memory-bound — the process is OOM-killing, or the dataset does not fit in RAM — the right move may be a *streaming* algorithm that processes the data in one pass with bounded memory, even if it does slightly more total work. Time and space complexity are separate axes; optimize the one that is actually killing you.

![decision tree for which structure to use starting from is it slow on big input then branching by whether the pattern is membership in a loop, a sorted range query, or counting occurrences, leading to set or dict, bisect, and Counter](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-8.png)

The decision tree above compresses the practical algorithm: when something is slow and the slowness grows with input size, identify the access pattern and reach for the matching structure. Membership in a loop wants a set or a dict ($O(1)$). A sorted-range or nearest-value query wants `bisect` on a sorted array ($O(\log n)$). Counting occurrences wants `Counter` (one $O(n)$ pass). Most production quadratics are one of these three in disguise, and the fix is almost always shorter than the bug.

## 13. The matrix: which operation costs what

Before the takeaways, here is the reference matrix you should internalize for CPython's built-ins. The whole "do less work" mindset is, in practice, choosing the structure whose operations match your access pattern — and that choice is governed by this table. (The sibling post [choosing the right built-in data structure](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) derives every row from the underlying implementation; here it is the cheat sheet.)

![matrix of common Python operations against their Big-O complexity and the operation count at one million elements covering list membership, set membership, dict get, sorting, append, and front insertion](/imgs/blogs/algorithmic-complexity-the-biggest-speedups-come-from-big-o-2.png)

The figure shows the operations that matter most for avoiding quadratics. In words: `list` membership and `list.index` are $O(n)$ — the quadratic traps. `set` and `dict` membership/get are $O(1)$ average — the fixes. `list.append` is amortized $O(1)$ (the dynamic array doubles its capacity occasionally, amortizing the resize), but `list.insert(0, x)` and `list.pop(0)` are $O(n)$ because every other element must shift — use a `collections.deque` for $O(1)$ operations at both ends. Sorting is $O(n \log n)$. Reading these off correctly is most of the battle: the moment you see an $O(n)$ operation inside a loop over the data, you have a candidate quadratic, and you check whether a structure swap makes it $O(1)$ or $O(\log n)$.

| Operation | Structure | Big-O | The trap / the fix |
| --- | --- | --- | --- |
| `x in s` | list | $O(n)$ | trap in a loop → use a set |
| `x in s` | set/dict | $O(1)$ avg | the fix |
| `s.index(x)` | list | $O(n)$ | trap in a loop → use a dict of index |
| `d[k]` | dict | $O(1)$ avg | direct lookup |
| `s.append(x)` | list | $O(1)$ amort | cheap |
| `s.insert(0, x)` | list | $O(n)$ | trap → use a deque |
| `s.appendleft(x)` | deque | $O(1)$ | the fix |
| `sorted(s)` | any | $O(n \log n)$ | unlocks bisect, adjacency |
| `bisect(s, x)` | sorted list | $O(\log n)$ | nearest / range lookup |

## 14. Key takeaways

- **The biggest speedups come from a better algorithm or data structure, not from making each operation faster.** A constant-factor trick (compile, vectorize, parallelize) multiplies your speed by a fixed number; a complexity improvement changes the exponent and wins by an unbounded margin at scale.
- **Read Big-O as a growth rate, then plug in real $n$.** At $n = 10^6$, an $O(n)$ algorithm does a million operations (~0.1 s in Python); an $O(n^2)$ one does a million million (~28 hours). That gap is the whole game.
- **The canonical quadratic is membership-against-a-list-in-a-loop.** `x in some_list`, `list.index`, and `list.remove` are each $O(n)$ and become $O(n^2)$ inside a loop. Swap the list for a `set` or `dict` and the inner cost drops to $O(1)$ — a speedup that *grows with the input*, routinely 10,000× on a million-element collection.
- **Diagnose complexity by measuring, not guessing.** Run `timeit` across a sweep of input sizes: if doubling $n$ quadruples the time, you have an $O(n^2)$; if doubling $n$ doubles the time, you are $O(n)$. The profiler points to the function; the size sweep names the complexity class.
- **At production scale the quadratic dominates the runtime,** so Amdahl's law makes fixing it the highest-leverage change available — its fraction $p$ of the total heads toward 1 as the data grows.
- **Vectorizing or compiling a bad algorithm gives you a fast bad algorithm.** Get the complexity right *first*, then apply the constant-factor levers to the now-linear code.
- **Mind the crossover.** A better-Big-O structure with a bigger constant only wins past a crossover in $n$. For tiny, bounded collections, a plain list or a few `if`s can be fastest — and it does not matter, because tiny is fast anyway. Apply the fix where $n$ is large.
- **Know the four standard fixes:** membership → set/dict; sorted lookups → `bisect`; counting → `Counter`; top-k → `heapq`. Most production quadratics are one of these in disguise, and the fix is shorter than the bug.
- **Watch space complexity too.** A time win can cost memory (the set's load factor, an index, a memo cache) and a space blowup can crash a program whose time was fine (the 80 GB pairwise array). When memory is the constraint, stream instead of materialize — a generator over a list comprehension is the same time and a factor of $n$ less space.
- **The complexity fix comes first, always.** Once your hot path is $O(n)$ or $O(n \log n)$, then — and only then — reach down the leverage ladder for vectorization, compilation, and parallelism. Those levers multiply; they never change the exponent. So fix the exponent before you spend a single line on the constant.

The discipline that runs through this whole post is the same one that runs through the series: do not guess where the time goes, measure it; do not optimize the constant on a bad algorithm, fix the algorithm; and always prove the win with a before→after number across a sweep of input sizes. Get into the habit of asking, of every loop you write, "what happens to this when the input is a thousand times bigger?" If the answer is "it gets a thousand times slower," you are linear and fine. If the answer is "it gets a million times slower," you have a quadratic, and there is almost certainly a built-in structure — a set, a dict, a `bisect`, a `Counter` — that turns it back into something that scales. That single question, asked relentlessly, will save you more 2 a.m. pages than any native rewrite ever will.

## 15. Further reading

- **CPython `timeit` documentation** — the right way to time a snippet, the `-n`/`-r` flags, and `repeat` + `min`: <https://docs.python.org/3/library/timeit.html>
- **CPython `bisect` and `heapq` documentation** — binary search on sorted lists and the priority-queue/top-k toolkit: <https://docs.python.org/3/library/bisect.html> and <https://docs.python.org/3/library/heapq.html>
- **CPython `collections` documentation** — `deque`, `Counter`, `defaultdict`, and their complexity characteristics: <https://docs.python.org/3/library/collections.html>
- **TimeComplexity wiki page** — the canonical big-O table for every CPython built-in operation: <https://wiki.python.org/moin/TimeComplexity>
- **"High Performance Python" (Gorelick & Ozsvald, O'Reilly)** — the chapter on lists, dicts, and sets derives these costs from the implementations.
- **Within the series**, this post sits between [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) (why doing fewer operations matters most in a slow-per-op language), [a mental model of performance](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) (the measure-first loop and Amdahl's law), [CPU profiling with cProfile](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) (how the profiler leads you to the quadratic), and [choosing the right built-in data structure](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) (the implementation details behind every complexity in this post).

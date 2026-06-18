---
title: "DataFrames at Speed: Pandas Pitfalls, Polars, and Arrow"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Your pandas is slow because it runs row-at-a-time over row-oriented memory. Go columnar: kill iterrows, fix dtypes, and reach for Polars, DuckDB, and Apache Arrow to turn a multi-minute job into seconds."
tags:
  [
    "python",
    "performance",
    "optimization",
    "pandas",
    "polars",
    "duckdb",
    "arrow",
    "dataframes",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-1.png"
---

A data scientist on a team I worked with had a notebook cell that cleaned and scored a marketing dataset. On her sample of fifty thousand rows it ran in about a second, so she shipped the same code into the nightly batch job that processed the full table — a little over five million rows. The next morning the job was still running when the daily dashboards were supposed to refresh, and the on-call engineer found a single Python process pinned at one hundred percent of one core, eight cores otherwise idle, three gigabytes of resident memory and climbing, forty minutes in and not done. The code was not wrong. It produced correct numbers on the sample. It was just built in a way that scaled linearly in the worst possible sense: every single row of the five million paid a full Python-interpreter tax, one row at a time, and the machine's seven other cores and its vector units sat there watching.

The fix took the rest of the morning to understand and about fifteen minutes to apply, and it cut the job from forty-plus minutes to under twenty seconds. There was no new hardware, no cluster, no GPU, no rewrite in C. The entire win came from one idea this whole post is about: **your pandas is slow because it is doing work row-at-a-time over memory that is laid out row-by-row, and the cure is to go columnar** — express your transforms as whole-column operations, fix your dtypes so the columns are compact and typed, and when the data gets big or stops fitting in RAM, reach for tools that were built columnar from the ground up: Polars, DuckDB, and the Apache Arrow format underneath them. This post closes the "vectorize" track of the series, and it is the same lesson as [why a NumPy array crushes a Python loop](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), applied to tables.

![before and after comparison of a pandas iterrows callback running in seconds versus one vectorized column operation running in milliseconds on a five million row input](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-1.png)

If you have read [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), you already know the shape of the problem. The Python interpreter is wonderful for flexibility and developer speed and terrible at doing the same tiny operation a few million times, because each tiny operation drags along the whole machinery of dynamic typing: a boxed object, a type dispatch, a reference-count bump, a trip around the bytecode evaluation loop. A dataframe is, at its core, a few million rows times a handful of columns — exactly the regime where that per-operation tax dominates everything. The discipline of this post is to stop paying that tax per row, and instead pay it once per *column*. Let us build the reasoning from the ground up, prove the numbers, and then climb the ladder from "faster pandas" to Polars to DuckDB to Arrow.

## The setup: a named machine and an honest benchmark

Every number in this post comes from one machine and one set of versions, stated up front so you can calibrate. The box is an 8-core x86-64 Linux server with 16 GB of RAM (the numbers are similar on a 2023 Apple M2 laptop, where the higher single-core speed narrows the gap a little). The software is CPython 3.12, pandas 2.2 (which matters — pandas 2.x has copy-on-write and a pyarrow backend), Polars 1.x, DuckDB 1.x, and pyarrow 16. When I quote a wall-clock time I mean the median of several runs after a warmup, measured with `time.perf_counter` or `timeit`, with the data already in memory or in the OS page cache so we are timing computation, not a cold disk read. When I quote memory I mean resident set size (RSS) or the dataframe's own `memory_usage(deep=True)`, stated which.

The running example is a single realistic table: an events log with five million rows and these columns — an integer `user_id`, a `category` string drawn from a small set like `"electronics"`, `"books"`, `"home"` (low cardinality, maybe twenty distinct values), a floating-point `amount`, and a `country` code string from about forty values. We will load it, clean it, transform it, and aggregate it, and at every step we will ask the same two questions: where did the time go, and which lever do we pull. That is the optimization loop the whole series runs on — measure, find the hot path, pick the lever, re-measure — and on dataframes the lever is almost always "stop looping in Python; operate on whole columns."

Here is the synthetic data so every benchmark below is reproducible:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N = 5_000_000
df = pd.DataFrame({
    "user_id": rng.integers(0, 1_000_000, size=N),
    "category": rng.choice(["electronics", "books", "home", "garden", "toys"], size=N),
    "amount": rng.gamma(2.0, 50.0, size=N),
    "country": rng.choice([f"C{i:02d}" for i in range(40)], size=N),
})
print(df.shape, df.dtypes.to_dict())
```

Note already that `category` and `country` come in as pandas' default `object` dtype, which we will see is a memory and speed disaster. Hold that thought.

## Why a per-row loop is the original sin

Start with the single most common way to make pandas slow, because understanding *why* it is slow is the key that unlocks everything else. Say we want a new column `fee` that is two percent of `amount`, but capped at five dollars. The instinct of someone coming from regular Python is to loop over the rows:

```python
fees = []
for _, row in df.iterrows():
    fee = min(row["amount"] * 0.02, 5.0)
    fees.append(fee)
df["fee"] = fees
```

This is correct. It is also catastrophically slow, and to know why we have to look at what `iterrows` actually does. On each of the five million iterations, pandas constructs a brand-new `Series` object to represent that one row. A `Series` is a full Python object with its own index, its own dtype handling, and its own attribute lookups. Then `row["amount"]` does a hashed lookup into that Series' index to find the value, and — this is the part that hurts — the value it returns is not a raw machine float sitting in a packed buffer. It is a **boxed** Python object: a `PyObject` on the heap with a type pointer and a reference count, because the moment you pull a single scalar out into Python land it has to become a first-class Python object. Then `* 0.02` triggers Python's numeric dispatch (look up `__mul__`, check types, allocate a new float object for the result), `min(...)` is a Python function call, and `.append` is another. Multiply all of that by five million.

Let us count the cost honestly. The per-row body is on the order of a dozen Python bytecode-level operations — constructing the row view, two hashed `__getitem__` lookups, a multiply that boxes a result, a `min` call, a list append. If each of those costs even 50–100 nanoseconds (a reasonable figure for boxed-object Python operations, as established in [the post on object overhead](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch)), then one row costs roughly one to a few microseconds, and five million rows costs several seconds — purely in interpreter overhead, before any *useful* arithmetic. The actual multiply-and-compare we wanted is a handful of nanoseconds; everything else is tax.

There is a further, sneakier cost hiding in `iterrows` that bites correctness as well as speed: because each row is returned as a single `Series`, and a `Series` has one dtype, pandas has to find a common type for *all* the columns in that row. If your row mixes an integer `user_id`, a string `category`, and a float `amount`, the only common type is `object`, so even your nice numeric columns come back to you as boxed Python objects — and worse, a clean `int64` can silently become a float along the way. People who switch from `iterrows` to vectorized code sometimes discover that subtle numeric bugs disappear at the same time, because the column operations preserve each column's true dtype instead of flattening every row to a lowest common denominator. Speed and correctness point the same direction here, which is the comfortable case.

#### Worked example: iterrows vs the vectorized column op

Let us put real numbers on it with `timeit`. On the named 8-core box, CPython 3.12, pandas 2.2:

```python
import timeit

# The per-row loop
def with_iterrows():
    fees = []
    for _, row in df.iterrows():
        fees.append(min(row["amount"] * 0.02, 5.0))
    return fees

# itertuples is faster than iterrows but still per-row Python
def with_itertuples():
    fees = []
    for t in df.itertuples(index=False):
        fees.append(min(t.amount * 0.02, 5.0))
    return fees

# apply(axis=1) is the same trap wearing a nicer hat
def with_apply():
    return df.apply(lambda r: min(r["amount"] * 0.02, 5.0), axis=1)

# The vectorized column op: one C loop over a packed buffer
def vectorized():
    return np.minimum(df["amount"].to_numpy() * 0.02, 5.0)

for name, fn in [("iterrows", with_iterrows), ("itertuples", with_itertuples),
                 ("apply", with_apply), ("vectorized", vectorized)]:
    t = timeit.timeit(fn, number=1)
    print(f"{name:12s} {t:8.3f} s")
```

The output on the named machine is in the ballpark of:

```bash
iterrows        9.812 s
itertuples      3.140 s
apply          11.700 s
vectorized      0.061 s
```

Read that table carefully because it contains three separate lessons. First, `iterrows` is the slowest of the row-wise approaches because every row is materialized as a full `Series` — nearly ten seconds. Second, `itertuples` is about three times faster than `iterrows` (it yields lightweight named tuples instead of `Series`, skipping a mountain of per-row construction) but it is *still* a Python loop over boxed values and *still* over three seconds; "use itertuples instead of iterrows" is a real but tiny win that misses the point. Third, `apply(axis=1)` is, despite its tidy one-liner appearance, the *worst* of the lot here — it is not vectorized at all; it calls your Python lambda once per row with a `Series` argument, with all the same boxing plus extra dispatch overhead. And then the vectorized version: 0.061 seconds. That is roughly a **160× speedup over iterrows** and **190× over apply(axis=1)**, and it is not a clever trick — it is one C loop (`np.minimum`) over one contiguous typed buffer (`df["amount"].to_numpy()`), with the multiply done as a single SIMD-friendly pass and zero per-element boxing.

This is the row-at-a-time trap, and it is the exact same trap as the Python `for`-loop versus NumPy array gap, because **a dataframe column is a NumPy (or Arrow) array**. When you write `df["amount"] * 0.02`, pandas hands the whole column to a single compiled loop that multiplies five million packed floats with no interpreter in the inner loop. When you write `for _, row in df.iterrows()`, you drag all five million floats up into Python one at a time and pay the full tax on each. The rule that follows is blunt and it is the most valuable sentence in this post: **if you are writing a Python `for` loop over the rows of a dataframe, or calling `apply(axis=1)`, you have almost certainly made a mistake.** There is nearly always a column expression that does the same thing in one native pass.

## Translating row logic into column logic

"Just vectorize it" is easy to say and sometimes genuinely non-obvious to do, especially when the row logic has branches. The skill worth building is translating `if/else` per row into whole-column boolean operations. The shift in approach is this: instead of asking "for this one row, which branch?", you compute the branch *condition for every row at once* as a boolean array, then use that boolean array to select. NumPy's `np.where` and pandas' masking are the workhorses.

Suppose the rule is: if `amount` is over one hundred, the fee is one percent; otherwise it is two percent, and either way capped at five dollars. Row-wise that is an `if`. Column-wise:

```python
amount = df["amount"].to_numpy()
rate = np.where(amount > 100.0, 0.01, 0.02)      # one boolean pass, one select
df["fee"] = np.minimum(amount * rate, 5.0)        # two more vectorized passes
```

Each line is a single C loop over the whole column. `np.where(cond, a, b)` evaluates the condition for all five million rows, then picks `a` where true and `b` where false — no Python branch in the inner loop. For more than two branches, `np.select([cond1, cond2, ...], [val1, val2, ...], default=...)` generalizes it. The cost is that you compute *all* branches for *all* rows and then select, which can do more total arithmetic than the row-wise version that short-circuits — but arithmetic on packed buffers is so cheap relative to the interpreter tax that doing "wasteful" vectorized work still wins by one to two orders of magnitude. That trade — do a bit more raw computation to avoid the per-row Python overhead — is the central bargain of vectorization and it almost always pays.

String operations vectorize too, through the `.str` accessor: `df["country"].str.startswith("C0")` runs a compiled loop, not a Python one. Datetime logic vectorizes through `.dt`. Conditional aggregation vectorizes through `groupby`. The pattern to internalize: **find the column accessor or the NumPy ufunc that expresses your per-row idea as a whole-column idea**, and reach for an explicit loop only when you have genuinely exhausted them (which is rarer than people think — and even then, the answer is usually Numba or Cython on the array, not a pandas row loop).

### The cost model, made precise

It helps to write the cost of each approach as a formula, because then the speedup stops being a surprise and becomes arithmetic you can predict. Let $n$ be the number of rows and let the per-row body do $k$ scalar operations (lookups, a multiply, a compare). A Python row loop pays, for each row, a fixed interpreter overhead $c_{\text{py}}$ per operation — the cost of fetching a bytecode, dispatching on type, boxing and unboxing a `PyObject`, bumping reference counts — plus the negligible cost of the actual arithmetic. So the total is roughly

$$T_{\text{loop}} \approx n \cdot k \cdot c_{\text{py}}$$

with $c_{\text{py}}$ on the order of tens of nanoseconds per operation, and for `iterrows` an *extra* large constant per row for building the `Series`. A vectorized column op, by contrast, runs $k$ separate C loops (one per `ufunc`), each of which pays a tiny per-element cost $c_{\text{C}}$ — a single machine instruction or a fraction of one with SIMD — plus a one-time fixed setup $f$ per ufunc call (the dispatch, the output allocation). So

$$T_{\text{vec}} \approx k \cdot f + n \cdot k \cdot c_{\text{C}}$$

The speedup is the ratio $T_{\text{loop}} / T_{\text{vec}}$, and once $n$ is large enough that the fixed setup $f$ is negligible, that ratio approaches $c_{\text{py}} / c_{\text{C}}$ — the per-operation overhead of interpreted, boxed Python divided by the per-operation cost of a packed C loop. That ratio is empirically somewhere between 30 and a few hundred, which is exactly the 50–200× band we keep measuring. This formula also explains the one case where vectorization does *not* win: when $n$ is tiny, the $k \cdot f$ setup term dominates the vectorized cost and the loop's small $n$ keeps its total down, so for, say, ten rows a Python comprehension can actually beat a NumPy round trip. The crossover is usually in the low hundreds of elements; above a few thousand rows, vectorization wins decisively and keeps pulling away. Vectorize when $n$ is large — and dataframes are, by definition, the large-$n$ regime.

There is a second-order effect worth naming: a chain of vectorized operations like `np.minimum(amount * rate, 5.0)` allocates an intermediate array for `amount * rate` before `np.minimum` consumes it, so it makes two full passes over memory and one temporary buffer. For very large arrays that memory traffic — not the arithmetic — becomes the bottleneck, which is the memory-bandwidth wall that [the post on avoiding temporaries](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) covers and that tools like `numexpr` and Polars' fused expressions specifically address by computing the whole expression in one pass without materializing the intermediate. That is one more reason a query-optimizing engine beats a chain of eager pandas calls: it fuses the passes.

![matrix mapping four common pandas anti-patterns to their vectorized fast equivalents including iterrows to column op and object dtype to category](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-2.png)

## The other two pandas traps: concat in a loop and object dtypes

Row loops are the headline pitfall, but two more are nearly as common and just as damaging, and they show up even in code that has no explicit `iterrows`.

### Building a dataframe by concatenating in a loop is quadratic

Here is a pattern that looks innocent and is anything but. You are reading a thousand files, or processing data in chunks, and you accumulate results like this:

```python
result = pd.DataFrame()
for chunk in chunks:                 # say 1000 chunks
    processed = transform(chunk)
    result = pd.concat([result, processed])   # GROWS the result each time
```

Each `pd.concat` allocates a brand-new dataframe and copies *all* the data accumulated so far, because pandas dataframes are backed by contiguous column buffers and you cannot append to them in place. So on iteration $k$ you copy roughly $k$ chunks' worth of data. Summing the copy cost over $n$ iterations gives $1 + 2 + 3 + \dots + n = \frac{n(n+1)}{2}$, which is $O(n^2)$. With a thousand chunks you do on the order of half a million chunk-copies instead of a thousand. This is the exact same quadratic-accumulation blowup described in [the algorithmic-complexity post](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) — the cost curve bends upward and a job that was fine on ten chunks crawls on a thousand.

The fix is to **build a list and concatenate once**:

```python
parts = []
for chunk in chunks:
    parts.append(transform(chunk))
result = pd.concat(parts, ignore_index=True)   # ONE concat, O(n) total
```

Now each chunk is copied exactly once, into the final result, and the total cost is linear. The difference at a thousand chunks of ten thousand rows each is dramatic: the quadratic version can take minutes and balloon memory as it churns through intermediate copies, while the build-once version takes a second or two. The general principle is the same one that says never grow a string with `+=` in a loop or check membership against a growing list: **do not repeatedly rebuild an immutable-ish structure inside a loop; collect the pieces, then build the whole thing once.**

### Object-dtype strings are huge and slow

Now the memory pitfall, which is subtler because the code that creates it looks completely normal. When pandas reads our `category` and `country` columns, it stores them as `object` dtype by default. An `object` column is a NumPy array of *pointers*, one machine word (8 bytes) per row, and each pointer points to a separate Python `str` object living somewhere on the heap. A Python string object is not just its characters — it carries an object header, a length, a hash cache, and the characters themselves, so even a short string like `"books"` costs roughly fifty-plus bytes as a live Python object. With five million rows you are paying 8 bytes per row for the pointer *plus* the cost of millions of scattered string objects, and crucially those string objects are spread all over the heap, so scanning the column means chasing five million pointers to random memory locations — terrible cache behavior.

The fix is to tell pandas the column has few distinct values by converting it to a **`category`** dtype. A categorical column stores the distinct values *once* in a small dictionary (the "categories") and represents each row as a small integer **code** indexing into that dictionary. For a column with twenty distinct values, each row's code fits in a single byte (or at most a few), and there are only twenty actual string objects total instead of five million. The win is enormous in both memory and speed.

![before and after stack showing object dtype strings storing a pointer per row versus category dtype storing small integer codes against a shared dictionary](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-7.png)

#### Worked example: object dtype vs category memory

```python
import pandas as pd

cat_obj = df["category"]                  # object dtype as loaded
cat_cat = df["category"].astype("category")
# pyarrow-backed strings are another good option in pandas 2.x:
cat_arrow = df["category"].astype("string[pyarrow]")

for name, s in [("object", cat_obj), ("category", cat_cat), ("string[pyarrow]", cat_arrow)]:
    mb = s.memory_usage(deep=True) / 1e6
    print(f"{name:16s} {mb:8.1f} MB")
```

On the named machine, for the five-million-row low-cardinality `category` column:

```bash
object             320.4 MB
category             5.0 MB
string[pyarrow]     45.7 MB
```

The `object` column eats about 320 MB — those five million pointers plus all the scattered Python string objects counted via `deep=True`. The `category` version is about 5 MB, roughly **64× smaller**, because it is essentially five million one-byte codes plus a tiny dictionary of five strings. The pyarrow-backed string column lands in between at about 46 MB: it does not deduplicate like a category, but it stores the characters in one contiguous Arrow buffer with an offsets array instead of five million separate heap objects, so it is far more compact than `object` and scans far better. (For a high-cardinality column where almost every value is unique — like a free-text field — `category` would *not* help and could even hurt, since the dictionary would be as large as the data; there, `string[pyarrow]` is the right call. Match the tool to the cardinality.)

The speed benefit follows the memory benefit. A `groupby("category")` on the categorical column is dramatically faster than on the object column, because grouping by an integer code is a tight integer operation over a contiguous buffer, whereas grouping by object strings means hashing and comparing scattered Python string objects. On this dataset a `groupby("category")["amount"].mean()` ran in roughly 0.12 s on the categorical column versus about 0.55 s on the object column — a 4–5× speedup — and the gap widens as the table grows. The same goes for filtering, joining, and sorting on that column.

### Downcasting numerics, and copy-on-write

Two smaller but worthwhile dtype wins round out the "faster pandas" toolkit. First, **downcast numeric columns** that do not need 64 bits. Our `user_id` came in as `int64` (8 bytes per row), but if user IDs fit under about two billion they fit in `int32` (4 bytes) — halving that column's memory. `pd.to_numeric(df["user_id"], downcast="integer")` or an explicit `.astype("int32")` does it; likewise `float32` instead of `float64` for amounts where you can tolerate the reduced precision halves the float column. On a wide table these halvings compound into gigabytes.

Second, pandas 2.x ships **copy-on-write (CoW)**, which changes the rules around an old class of bug and an old class of slowdown. Historically, slicing a dataframe sometimes returned a view and sometimes a copy, and writing to a slice sometimes mutated the parent and sometimes triggered the dreaded `SettingWithCopyWarning`. With CoW (enable it via `pd.options.mode.copy_on_write = True`, and it becomes the default in pandas 3.0), the semantics are clean: a slice never silently shares mutable state, but the actual physical copy is *deferred* until someone writes, so read-only slicing is cheap and copies happen only when truly needed. The practical effect for performance is fewer accidental full-column copies and more predictable memory, and the practical effect for correctness is that the chained-assignment footguns are gone. Turn it on.

## Why row-oriented versus columnar is the whole game

We have been saying "go columnar" as if it were obvious, but it is worth making the *why* rigorous, because it explains not just the pandas fixes above but why Polars and DuckDB exist at all. The question is: given a table, how do you lay it out in memory — by row or by column? — and why does the answer matter so much for analytical work.

![stack diagram contrasting a row-oriented record layout against a columnar layout where each column is one contiguous typed buffer feeding SIMD and compression](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-3.png)

In a **row-oriented** layout (how a CSV file reads, how a transactional database stores rows, how a list of Python dicts is shaped), the values of one record sit together: `user_id`, then `category`, then `amount`, then `country`, then the next record's `user_id`, and so on. That is great if your workload is "fetch one whole record by key" — an online transaction. But analytical dataframe work is almost the opposite: you scan *one column across all rows* — "the mean of `amount`", "filter where `category` equals electronics", "group by `country`". In a row layout, the values of `amount` are scattered through memory with the other columns interleaved between them, so reading the `amount` column means striding across memory, pulling in cache lines that are mostly *other columns* you do not care about.

In a **columnar** layout, each column is one contiguous typed buffer: all five million `amount` floats packed back-to-back, all five million `category` codes back-to-back, and so on. Three concrete physical wins follow, and they are the reason columnar dominates analytics:

1. **Cache locality.** A CPU cache line is 64 bytes. When you scan a packed `float64` column, each 64-byte line you pull from RAM holds eight useful values, and the hardware prefetcher sees the sequential access pattern and reads ahead. In a row layout where each record is, say, forty bytes, a 64-byte line holds maybe one and a half records' worth of `amount` plus a lot of bytes you discard — so you move several times more memory to scan the same column. As established in [the memory-and-locality reasoning](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), memory bandwidth is the bottleneck for this kind of streaming work, and moving less memory is directly faster.

2. **SIMD.** Modern CPUs have vector instructions (SSE, AVX) that apply one operation to 4, 8, or 16 values at once — but only if those values are adjacent in memory and uniformly typed. A packed `float64` column is exactly that shape, so summing or multiplying it runs at several elements per cycle. A row layout cannot feed SIMD because the values are not adjacent.

3. **Compression.** A contiguous typed column compresses far better than mixed-type rows, because adjacent values are similar (run-length encoding for sorted or low-cardinality columns, dictionary encoding for strings, delta encoding for timestamps). This matters both on disk (Parquet files are columnar and routinely 5–10× smaller than the equivalent CSV) and increasingly in memory.

This is precisely why a `category` column is fast: it is a contiguous buffer of small integer codes — maximally cache-friendly, SIMD-able, compressible. It is why pandas operations on numeric columns are fast (NumPy columns are contiguous) and operations on `object` columns are slow (pointer-chasing defeats all three wins). And it is why the next generation of dataframe tools did not patch pandas but rebuilt on a columnar foundation called Apache Arrow. If you want to see the same page-and-buffer reasoning from the database side, [how databases store data](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool) walks through pages, heap files, and the buffer pool — analytical engines lean on columnar storage for exactly the cache and compression reasons above.

It is worth putting a number on how much memory traffic the layout choice saves, because it makes the abstract "cache locality" claim concrete. Suppose each record is forty bytes across all its columns, but the `amount` column you want to sum is eight of those bytes. In a columnar layout, summing `amount` over five million rows reads exactly $5{,}000{,}000 \times 8 = 40$ MB of memory, all of it useful, all of it sequential — the prefetcher and SIMD run at full tilt. In a row-oriented layout, the hardware cannot read just the `amount` bytes; it pulls whole 64-byte cache lines, and since the eight `amount` bytes you want are surrounded by thirty-two bytes of other columns, you drag in roughly $5{,}000{,}000 \times 40 = 200$ MB to touch the 40 MB you need — five times the memory traffic for the same logical work. On a memory-bandwidth-bound operation (and a simple sum or filter is exactly that), five times the bytes moved means roughly five times the time, before you even account for the SIMD you also lost. That factor-of-five is baked into the physics of the layout; no amount of clever code recovers it once the data is stored row-wise. This is the same reason a database serving analytical queries reaches for a column store while one serving single-record transactions stays row-oriented: the access pattern, not the data, decides the optimal layout.

## Apache Arrow: the columnar standard underneath everything

Apache Arrow is a **language-independent columnar memory format**. It specifies exactly how a table's columns are laid out in memory: each column is a contiguous buffer of values plus, where needed, a separate validity bitmap (one bit per row marking null/not-null) and, for variable-length data like strings, an offsets buffer. Crucially, this is a *specification*, not just a library — pandas (via its pyarrow backend), Polars, DuckDB, and many others can all point at the *same* Arrow buffers in memory and agree on what they mean.

That shared agreement enables the property that makes Arrow transformative for Python data work: **zero-copy interchange**. Normally, moving a dataframe from one tool to another means serialization — pandas writes the data into some intermediate format (pickle, CSV, even a copy), the bytes move, and the other tool deserializes them back into its own structures. For a multi-gigabyte table that serialize/deserialize round trip can cost seconds and double your peak memory. With Arrow, if pandas, Polars, and DuckDB all speak Arrow, handing a table from one to another is just handing over a *pointer* to the existing buffers — no copy, no serialization, near-instant regardless of size.

```python
import pandas as pd
import polars as pl
import pyarrow as pa
import duckdb

pdf = df  # our pandas dataframe

# pandas -> Arrow table: zero-copy for Arrow-compatible columns
arrow_tbl = pa.Table.from_pandas(pdf)

# Arrow -> Polars: zero-copy, shares the same buffers
pldf = pl.from_arrow(arrow_tbl)

# DuckDB can query an Arrow table (or a pandas/Polars frame) in place
con = duckdb.connect()
out = con.execute(
    "SELECT category, avg(amount) AS avg_amt "
    "FROM arrow_tbl GROUP BY category ORDER BY avg_amt DESC"
).arrow()   # results come back as Arrow too, zero-copy onward

# Polars back to pandas, routed through Arrow internally
back_to_pandas = pldf.to_pandas()
```

The line `con.execute("... FROM arrow_tbl ...")` is worth pausing on: DuckDB is reading directly from the in-memory Arrow buffers that pandas produced, running a parallel SQL group-by over them, and returning Arrow — and at no point did anybody pickle or copy the five million rows. This is the plumbing that lets you mix tools freely: clean in pandas because you know its API, hand the table to DuckDB for a heavy join it does better, pull the result into Polars for a fast windowed transform, all sharing one set of buffers. Arrow is the lingua franca, and "is it Arrow-native?" is now a real question to ask of any data tool you adopt.

There is a subtlety worth stating honestly: zero-copy works cleanly for numeric and Arrow-string columns, but a pandas `object` column (those scattered Python strings) cannot be zero-copied into Arrow — it has to be converted into a proper Arrow string buffer first, which *is* a copy. That is one more reason to keep your pandas columns in real dtypes (`category`, `string[pyarrow]`, numeric) rather than `object`: it keeps the door to zero-copy open.

### How an Arrow column is actually laid out

It is worth knowing the physical shape of an Arrow column, because once you see it the zero-copy property stops being magic. A primitive column — say `int64` — is exactly two buffers: a contiguous block of the values (8 bytes each, packed back to back) and an optional **validity bitmap**, one bit per row, where a 0 means that row is null. That is it. Any tool that knows "the values start at this pointer, there are this many, the type is int64, and nulls are marked in that bitmap" can read the column with no conversion. A string column is three buffers: a values buffer holding all the characters concatenated together with no separators, an **offsets** buffer of integers marking where each string starts and ends in the values buffer, and the validity bitmap. To read row $i$'s string you slice the characters between `offsets[i]` and `offsets[i+1]` — no per-row object, no pointer chase, just two contiguous buffers. This is why Arrow strings scan so much faster than pandas `object` strings: the characters are in one place and reading them is a sequential walk, not five million heap lookups.

This same layout is what makes the null-handling honest across tools. Pandas historically represented missing numbers as `NaN` (a float trick that forces integer columns with nulls up to float), which is a constant source of dtype surprises. Arrow's separate validity bitmap means an integer column with nulls stays an integer column — the nulls live in the bitmap, not in the values — and that is one reason pandas' pyarrow-backed dtypes have cleaner missing-data semantics than the classic NumPy-backed ones.

### Parquet is Arrow on disk

Arrow is the in-memory format; **Parquet** is its on-disk counterpart, and the two are designed to work together. A Parquet file is columnar (each column stored together), compressed per column (so the low-cardinality `category` column shrinks dramatically, the sorted timestamp column delta-encodes, and so on), and it carries **statistics** — per-column, per-row-group min and max values — in its metadata. Those statistics are what enable predicate pushdown without reading data: a query filtering `amount > 1000` can look at a row group whose recorded `amount` maximum is 800 and skip the entire row group without decoding a single value. Combined with reading only the referenced columns (projection pushdown), this is why `pl.scan_parquet` and DuckDB over Parquet can answer a query while reading a small fraction of the file. The practical advice that falls out: **store analytical data as Parquet, not CSV.** CSV is row-oriented text with no types, no compression, and no statistics — every query re-parses every byte. Converting a pipeline's storage from CSV to Parquet often gives a larger end-to-end speedup than any in-memory tuning, because it changes how many bytes you even touch:

```python
# One-time conversion; thereafter every query reads less and skips more.
pl.scan_csv("events.csv").sink_parquet("events.parquet")   # streaming, low memory
# or with pandas:  pd.read_csv("events.csv").to_parquet("events.parquet")
```

On our five-million-row table the CSV was about 280 MB on disk; the Parquet equivalent was about 38 MB — roughly 7× smaller — and a filtered group-by over it read only the two relevant columns from only the qualifying row groups, a few megabytes, versus re-parsing the entire 280 MB of text every time the CSV was queried.

## Polars: columnar, lazy, multi-threaded, optimized

Polars is a dataframe library built on Arrow from the start, written in Rust, and it changes three things about how a dataframe computation runs. Understanding each one tells you *why* it is routinely 5–30× faster than pandas on real jobs.

First, it is **multi-threaded by default**. Pandas runs on a single core for almost everything — the work that data scientist's job did on one core while seven sat idle. Polars splits a column scan, a group-by, a join across all your cores automatically, with no `multiprocessing`, no pickling, no `ProcessPoolExecutor` to manage. On an 8-core box that alone is a large constant-factor win, and unlike Python's `multiprocessing` it has no per-task serialization tax because the threads share the same Arrow buffers (Polars is Rust, so it has no GIL holding back its own threads — the reasons the GIL blocks Python threads do not apply inside Rust code).

Second, it has a **lazy API**. In pandas, every line executes immediately (eagerly): `df[df.amount > 100]` filters right now, producing a new dataframe, and the next line operates on that. Each step materializes a full intermediate result. In Polars' lazy mode you instead *describe* the whole computation as a query, and nothing runs until you call `.collect()`. That deferral is what unlocks the third thing.

Third, a **query optimizer**. Because Polars sees the whole query before running it, it can rewrite it to do less work. The two most important rewrites are **predicate pushdown** (push filters down to the data source so rows you will throw away are never read or materialized) and **projection pushdown** (only read the columns the query actually uses). If your query reads a 40-column Parquet file but only references three columns and filters to one country, projection pushdown reads three columns instead of forty and predicate pushdown reads only the matching row groups — you can easily avoid reading 90% of the bytes on disk before any computation happens. Pandas, executing eagerly line by line, cannot do this: by the time you write the filter, it has already loaded everything.

![before and after comparison of eager single threaded pandas making copies versus lazy multi threaded Polars with an optimized pushdown plan on a four million row group by](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-4.png)

Here is the same clean-filter-aggregate job in eager pandas and in lazy Polars:

```python
import polars as pl

# pandas, eager: each step makes an intermediate
pdf_result = (
    pdf[pdf["amount"] > 100.0]                      # materialize filtered frame
       .assign(fee=lambda d: (d["amount"] * 0.02).clip(upper=5.0))
       .groupby("category", observed=True)["fee"]
       .sum()
)

# Polars, lazy: describe the whole query, optimize, then run once
ldf = pl.from_pandas(pdf).lazy()                    # or pl.scan_parquet("events.parquet")
pl_result = (
    ldf.filter(pl.col("amount") > 100.0)
       .with_columns((pl.col("amount") * 0.02).clip(upper_bound=5.0).alias("fee"))
       .group_by("category")
       .agg(pl.col("fee").sum())
       .collect()                                   # <-- nothing ran until here
)
```

The Polars version builds a logical plan (you can inspect it with `ldf...explain()` to literally see the pushdowns), the optimizer rewrites it, and then `.collect()` runs it across all cores, never materializing the full filtered intermediate — it fuses the filter, the fee computation, and the aggregation into a streaming parallel pass.

![graph of the Polars lazy execution flow from scan to logical plan to predicate and projection pushdown to parallel execution to the collected result](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-5.png)

#### Worked example: pandas versus Polars on a few-million-row job

The job: from the five-million-row events table, keep rows where `amount > 100`, compute a capped fee, and sum the fee per category. Timed on the named 8-core box, median of several warm runs:

```python
import timeit

def pandas_job():
    f = pdf[pdf["amount"] > 100.0]
    fee = (f["amount"] * 0.02).clip(upper=5.0)
    return fee.groupby(f["category"], observed=True).sum()

def polars_job():
    return (
        pl.from_pandas(pdf).lazy()
          .filter(pl.col("amount") > 100.0)
          .with_columns((pl.col("amount") * 0.02).clip(upper_bound=5.0).alias("fee"))
          .group_by("category").agg(pl.col("fee").sum())
          .collect()
    )

print("pandas", round(timeit.timeit(pandas_job, number=3) / 3, 3), "s")
print("polars", round(timeit.timeit(polars_job, number=3) / 3, 3), "s")
```

Representative output:

```bash
pandas 4.231 s
polars 0.342 s
```

That is roughly a **12× speedup**, and it decomposes into the three factors above. A chunk comes from multi-threading (8 cores doing the group-by in parallel instead of one), a chunk from not materializing the full filtered intermediate dataframe (the lazy fused pass), and a chunk from Polars' generally faster columnar engine and its better group-by algorithm. If we read from a Parquet file with `pl.scan_parquet` instead of an in-memory frame, the gap widens further because projection and predicate pushdown mean Polars reads far fewer bytes off disk than a `pd.read_parquet` followed by an in-memory filter.

A note on the lazy mindset, because it is the genuinely new idea here: **`scan_*` is lazy, `read_*` is eager.** `pl.read_parquet(path)` loads the whole file into memory now; `pl.scan_parquet(path)` returns a lazy frame that reads nothing yet and lets the optimizer push your filters and column selection down into the read. For anything nontrivial, prefer `scan_*` + a chain of expressions + `.collect()`. You write the query; Polars figures out the efficient way to run it. That is the same bargain a SQL database has always offered, brought to the dataframe API.

### The expression API is where the speed lives

The piece that surprises people coming from pandas is that Polars wants you to express transformations as **expressions** — composable descriptions of column computations — rather than as a sequence of indexing and assignment statements. An expression like `pl.col("amount") * 0.02` is not a value; it is a recipe that the engine can analyze, reorder, fuse with neighboring expressions, and run in parallel. This is why a list of expressions inside one `.with_columns([...])` runs them concurrently across cores and fuses passes over the data, whereas the equivalent pandas code (`df["a"] = ...; df["b"] = ...`) runs each assignment eagerly, one after another, each making its own pass. The expression model is the API-level reason Polars can optimize: you hand it intent, not imperative steps.

A few expression patterns cover most real work, and seeing them de-mystifies the API:

```python
result = (
    pl.scan_parquet("events.parquet")
      .filter(pl.col("amount") > 100)                      # row filter (pushed down)
      .with_columns([
          (pl.col("amount") * 0.02).clip(upper_bound=5.0).alias("fee"),
          pl.when(pl.col("amount") > 100)                  # vectorized if/else
            .then(pl.lit("high")).otherwise(pl.lit("low")).alias("bucket"),
          pl.col("category").cast(pl.Categorical),         # dtype control
      ])
      .group_by(["country", "bucket"])
      .agg([
          pl.col("fee").sum().alias("total_fee"),
          pl.col("amount").mean().alias("avg_amount"),
          pl.len().alias("n"),
          (pl.col("amount") > 500).sum().alias("big_orders"),  # conditional count
      ])
      .sort("total_fee", descending=True)
      .collect()
)
```

Every one of those `.agg` expressions runs as a parallel pass over the grouped data, and `pl.when().then().otherwise()` is the Polars spelling of `np.where` — the vectorized branch. There is no `apply`, no loop, no per-row callback anywhere; the whole thing is a declarative query that the engine compiles and runs. Once the expression habit clicks, you stop reaching for row loops because the expression vocabulary covers the cases you used to loop for.

### Joins: where a real engine earns its keep

Joining two large tables is the operation where the difference between "dataframe library" and "query engine" shows most starkly. A pandas `merge` works, but it runs on one core and materializes the full result. Polars (and DuckDB) run a parallel hash join: build a hash table on the smaller table's join key, then probe it once for each row of the larger table — total cost $O(n + m)$ rather than the $O(n \times m)$ of a naive nested comparison, and the probe phase splits across cores.

```python
users = pl.scan_parquet("users.parquet")          # ~1M users, lazy
events = pl.scan_parquet("events.parquet")        # 5M events, lazy

joined = (
    events.join(users, on="user_id", how="inner")  # parallel hash join
          .group_by("country")
          .agg(pl.col("amount").sum())
          .collect()
)
```

On the named box, joining the 5M-row events table to a 1M-row users table and aggregating ran in well under a second in Polars, versus several seconds for the equivalent eager pandas `merge` + `groupby` — and the lazy version, because it sees that only `user_id`, `country`, and `amount` are needed downstream, pushes a projection so it never reads the other user columns at all. This is the leverage of letting an optimizer plan the join rather than executing a hand-written merge: it changes both the algorithm's constant factor (parallelism) and how much data even gets touched (pushdown).

### Thread scaling

Because Polars parallelizes automatically, it is worth knowing how the win scales with cores, and that it does *not* scale perfectly. You can cap Polars' threads with the `POLARS_MAX_THREADS` environment variable to measure. On the group-by job above, going from 1 to 8 threads on the named box looked roughly like this:

| Threads | Wall (s) | Speedup vs 1 thread |
| --- | --- | --- |
| 1 | 1.95 | 1.0× |
| 2 | 1.05 | 1.9× |
| 4 | 0.58 | 3.4× |
| 8 | 0.34 | 5.7× |

The scaling is strong but sub-linear — 8 threads buys about 5.7×, not 8×. This is Amdahl's law in action: parts of the query (planning, the final merge of per-thread partial aggregates, any inherently serial step) do not parallelize, so as you add cores the serial fraction caps the win. The general form, $S = 1/((1-p) + p/s)$ for parallel fraction $p$ across $s$ workers, predicts exactly this diminishing return, and it is the same ceiling discussed for [multiprocessing and the cost of coordination](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop). Still, getting most of an 8-core machine for free, with no `multiprocessing` boilerplate and no pickling, is exactly the kind of high-leverage win this series chases.

## DuckDB: SQL over data that does not fit in RAM

Polars solves "make the in-memory dataframe fast." But sometimes the data does not fit in memory at all — a hundred gigabytes of Parquet, a directory of files larger than your 16 GB box. For that, the right tool is often **DuckDB**: an embedded, columnar, vectorized SQL engine that runs in your process (like SQLite, but column-oriented and built for analytics), parallelizes across cores, and — critically — is **out-of-core**, meaning it streams data through memory and spills to disk when a query's working set exceeds RAM, instead of crashing with an out-of-memory error.

DuckDB's superpower for Python users is that it queries files and dataframes directly, in place, with no import step:

```python
import duckdb

# Query a directory of Parquet files larger than RAM, with SQL.
# DuckDB reads only the columns and row groups the query needs (pushdown),
# parallelizes across cores, and spills to disk if the aggregation is huge.
result = duckdb.sql("""
    SELECT country,
           count(*)        AS n,
           avg(amount)     AS avg_amount,
           sum(amount)     AS total
    FROM 'events/*.parquet'
    WHERE amount > 100
    GROUP BY country
    ORDER BY total DESC
    LIMIT 20
""").df()   # .df() returns a pandas DataFrame; .pl() returns Polars; .arrow() Arrow
```

That single query, pointed at a glob of Parquet files totaling far more than your RAM, runs to completion on a laptop. DuckDB's reader does projection pushdown (only the four referenced columns are read from each file, not all forty), predicate pushdown (Parquet stores per-row-group min/max statistics, so row groups where every `amount` is ≤ 100 are skipped entirely without reading them), and processes the data in vectorized batches across all cores, with the partial aggregates spilling to a temp file if they grow too large. The same query in pandas would require `pd.read_parquet` to materialize the entire dataset in memory first — which, by assumption, does not fit, so it simply fails.

The other thing DuckDB excels at is **joins**. A hash join over millions of rows is the canonical case where a real query engine beats hand-rolled dataframe code: DuckDB builds a hash table on the smaller side and probes it once per row of the larger side, an $O(n + m)$ algorithm, and parallelizes the probe across cores. (Doing a join "by hand" with a Python loop is the $O(n \times m)$ nested-scan disaster the [complexity post](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) warns about — a hash join changes the exponent, exactly the kind of algorithmic win that beats any constant-factor speedup.) When your transform is naturally expressed as SQL — joins, window functions, complex group-bys — DuckDB is often both the fastest and the most readable option, and it returns its results straight into pandas, Polars, or Arrow.

![matrix comparing pandas Polars and DuckDB across execution model parallelism out of core support and what each is best for](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-6.png)

#### Worked example: DuckDB on bigger-than-RAM Parquet

I generated 60 GB of Parquet (well over the 16 GB box's RAM) partitioned across files, with the same schema as our events table scaled up. The task: average `amount` per country for rows with `amount > 100`. Pandas could not run it — `pd.read_parquet("events/")` exhausted memory and the process was killed. DuckDB ran the query in about 38 seconds, reading roughly 9 GB off disk (only the two needed columns, only the qualifying row groups, out of the full 60 GB), parallelized across 8 cores, peaking at about 2.5 GB of RAM because it streamed batches rather than loading everything. Polars with `scan_parquet(...).collect(streaming=True)` ran the same job in a comparable ballpark, around 41 seconds, with similar pushdown and a similar low memory ceiling. The headline: with the right engine, "the data is bigger than my RAM" stops being a wall and becomes a 40-second query on a laptop — and the wrong engine (eager pandas) does not slow down gracefully, it dies.

### Polars or DuckDB? They overlap, but lead in different places

These two tools cover much of the same ground — both columnar, both vectorized, both parallel, both push predicates and projections down, both read Parquet and speak Arrow — so the choice is often about ergonomics and the shape of the workload rather than raw speed, where they trade blows query by query. Reach for **DuckDB** when the work is naturally SQL: multi-table joins, window functions, `GROUP BY ... HAVING`, correlated subqueries, the kind of analytics that reads cleanly as a query and would be a tangle of dataframe operations. DuckDB's out-of-core story is also more battle-tested for the genuinely-larger-than-RAM case, and because it is a full SQL engine it slots in where a database would — you can even register your pandas and Polars frames as tables and join across them. Reach for **Polars** when you are doing dataframe-shaped transformation work — long chains of column expressions, reshaping, custom logic — and want to stay in a Python-native API with strong typing and an expression model that composes. A common and very effective pattern is to use both: DuckDB for the heavy SQL join or the out-of-core scan, handing its Arrow result to Polars (zero-copy) for the expression-heavy transformation, then to pandas for the last-mile plotting or model-feeding. Because they share Arrow, mixing them costs nothing at the boundaries.

A note on DuckDB's streaming execution, since it is the property that makes the out-of-core case work. DuckDB processes data in **vectors** — batches of about a couple thousand values — flowing through the query operators in a pipeline, rather than materializing whole intermediate tables. A filter feeds its surviving rows directly into the next operator a batch at a time, so the memory footprint at any instant is roughly one batch per operator, not the whole dataset. When an operator that *must* hold state — a hash join's build side, a large `GROUP BY`'s aggregate table — grows beyond the configured memory limit, DuckDB spills that state to a temporary file on disk and continues, trading some speed for completion instead of an out-of-memory crash. This is the architectural reason a 16 GB laptop can group-by a 60 GB file: at no point does the engine try to hold 60 GB at once.

## Putting it together: the running pipeline, end to end

Let us return to that data scientist's job and walk the whole optimization loop on it, because the lesson is in the *sequence* of decisions, not any single trick. The original job, abridged, looked like this:

```python
# THE SLOW ORIGINAL — ~40 minutes on 5M rows, one core, 3+ GB RSS
df = pd.read_csv("events.csv")                 # object-dtype strings everywhere
out = []
for _, row in df.iterrows():                   # per-row Python loop (the sin)
    fee = min(row["amount"] * 0.02, 5.0)
    bucket = "high" if row["amount"] > 100 else "low"
    out.append((row["user_id"], row["category"], fee, bucket))
result = pd.DataFrame(out, columns=["user_id", "category", "fee", "bucket"])
summary = pd.DataFrame()
for cat in result["category"].unique():        # group-by reinvented as a loop
    sub = result[result["category"] == cat]
    summary = pd.concat([summary, sub.groupby("bucket")["fee"].sum().to_frame().T])
```

Profiling it (with `cProfile`, sorting by cumulative time, the technique from [the CPU-profiling post in this series](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path)) showed the time piled up in `iterrows` and in `pd.concat` — exactly the two pitfalls. We climbed the ladder one rung at a time, measuring each step:

```python
# STEP 1 — vectorize the per-row loop into column ops
df = pd.read_csv("events.csv", dtype={"category": "category", "country": "category"})
amount = df["amount"].to_numpy()
df["fee"] = np.minimum(amount * 0.02, 5.0)
df["bucket"] = np.where(amount > 100, "high", "low")   # one pass, no row loop

# STEP 2 — one vectorized group-by instead of a concat loop
summary = df.groupby(["category", "bucket"], observed=True)["fee"].sum().unstack()
```

Step 1 (kill `iterrows`, read columns as `category`) took the per-row portion from minutes to under a second and cut memory by reading categoricals instead of object strings. Step 2 (one native `groupby` instead of the concat-in-a-loop) replaced the quadratic accumulation with a single linear pass. That alone took the job from ~40 minutes to roughly 25 seconds and from 3+ GB to under 1 GB of RSS. Then, because this job runs nightly on growing data, we went one rung further and moved the whole thing to Polars reading Parquet:

```python
# STEP 3 — Polars lazy + Parquet, multi-threaded with pushdown
summary = (
    pl.scan_parquet("events.parquet")
      .with_columns([
          pl.min_horizontal(pl.col("amount") * 0.02, pl.lit(5.0)).alias("fee"),
          pl.when(pl.col("amount") > 100).then(pl.lit("high"))
            .otherwise(pl.lit("low")).alias("bucket"),
      ])
      .group_by(["category", "bucket"])
      .agg(pl.col("fee").sum())
      .collect()
)
```

Step 3 brought it under 5 seconds, using all 8 cores and reading only the needed columns from the Parquet file. The cumulative arc — 40 minutes to under 5 seconds, roughly a **500× wall-clock improvement** and a 3×+ memory cut — came entirely from the columnar discipline: stop looping in Python, fix the dtypes, let an Arrow-native engine use the whole machine. Not one line of C was written. That is the leverage ladder this series teaches, applied to tabular data.

## Case studies and real numbers

It is worth grounding the claims in published, reproducible results rather than only my own machine.

**Polars on the H2O.ai database-like benchmark.** The widely cited `db-benchmark` (group-by and join benchmarks over tables of 0.5, 5, and 50 GB) consistently shows Polars and DuckDB finishing group-by and join queries several times faster than pandas, and on the larger sizes pandas often cannot complete the 50 GB cases at all on a normal box while Polars and DuckDB stream through them. The exact multipliers vary by query and version, but the pattern — Arrow-native, multi-threaded, query-optimized engines beating eager single-threaded pandas by 5–30× and degrading gracefully where pandas falls over — is robust across the benchmark suite. Treat the specific numbers as version-dependent and check the current results, but the *shape* is dependable.

**The Rust-rewrite ecosystem.** Polars itself is the proof of the broader thesis this series keeps returning to: the hot path got rewritten in a native language (Rust), the slow Python wrapper is thin, and the result is order-of-magnitude faster while still driven from Python. The same story produced pydantic-core, ruff, the `tokenizers` library, and uv — Python interfaces over Rust engines. You do not have to write the Rust; you benefit by *choosing the Arrow-native, native-engine tool* for the heavy lifting and keeping Python for glue.

**The iterrows-to-vectorized speedup, generalized.** The 100–200× range we measured for `iterrows` versus a vectorized column op is not specific to our toy benchmark; it is the typical range whenever you replace a per-row Python callback with a single columnar pass, and it tracks directly the boxed-PyObject-versus-packed-buffer gap measured throughout the [NumPy](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) and object-cost posts. When the per-row body is heavier (multiple lookups, string work), `iterrows` gets even worse relative to the vectorized form. The honest framing: expect 50–200× from killing a row loop, more if the column also has a bad dtype you fix at the same time.

**Object-versus-category memory, generalized.** The ~60× memory shrink we measured is roughly what you get for a *low-cardinality* string column. The win is proportional to the redundancy: a column where every value is unique gets little to no benefit from `category` (and should use `string[pyarrow]` instead), while a column of, say, country codes or status flags repeated across millions of rows is exactly the case `category` was built for. Always check `df.memory_usage(deep=True)` before and after — measure the win, do not assume it.

## When to reach for each tool (and when not to)

Every tool here is a trade-off, so let us be decisive about the boundaries. The table below is the quick reference; the paragraphs that follow give the reasoning.

| Tool | Execution model | Parallel | Out-of-core | Reach for it when | Skip it when |
| --- | --- | --- | --- | --- | --- |
| pandas 2.x | eager, single-thread | no | no | fits in RAM, you need the ecosystem | the row loop is the bottleneck (fix that first) |
| Polars | lazy, optimized | all cores | streaming | large in-RAM tables, want all cores free | a 10k-row frame; pandas already runs in a ms |
| DuckDB | vectorized SQL engine | all cores | yes, spills to disk | SQL-shaped work, bigger than RAM, query files in place | a one-line NumPy transform |
| Arrow / pyarrow | format, not an engine | n/a | n/a | moving data between the above without copying | you are not crossing a tool boundary |


**Reach for vectorized pandas** when your data fits comfortably in RAM, you are already in the pandas ecosystem, and you need its enormous library surface and integrations (it is the lingua franca of the PyData world, the format scikit-learn, matplotlib, statsmodels, and a thousand tutorials expect). Pandas 2.x with copy-on-write, proper dtypes (`category`, `string[pyarrow]`, downcast numerics), and *no row loops* is genuinely fast for most work. Do not flee pandas at the first slowdown — first profile, kill the `iterrows`/`apply(axis=1)`, fix the dtypes, and de-loop the `concat`. That fixes the large majority of "pandas is slow" complaints without changing tools.

**Reach for Polars** when the table is large (millions to low billions of rows that still fit in RAM, or streams through it), you want to use all your cores without writing `multiprocessing`, and you are willing to learn its expression API. Its lazy `scan_*` + pushdown is a real, free win on Parquet. The cost is the smaller (though fast-growing) ecosystem and a different API — but `.to_pandas()` is zero-copy-ish through Arrow, so you can hand results back to pandas for the last mile. Do not reach for Polars for a 10,000-row dataframe where pandas runs in a millisecond — the multi-threading overhead is not worth it at that size, and you give up ecosystem for nothing.

**Reach for DuckDB** when the natural expression of your transform is SQL (joins, windows, complex aggregations), when the data is bigger than RAM and you need out-of-core streaming with spill-to-disk, or when you want to query Parquet/CSV files directly without an import step. It is the right answer to "push the work down to a real query engine." Do not reach for DuckDB for a tiny in-memory transform that is one line of NumPy — SQL is overkill there.

**Reach for Arrow explicitly** when you are moving data *between* these tools and want to avoid the serialize/deserialize tax — it is the zero-copy bridge. You rarely program against raw Arrow directly; you benefit from it by keeping your columns in Arrow-compatible dtypes and choosing Arrow-native tools so the hand-offs are free.

And the universal "do not": **do not loop in Python over rows, ever, if a column operation exists** — and one almost always does. That single rule, plus right-sizing dtypes, captures most of the available speed before you change a single tool.

![decision tree for choosing pandas for small flexible work Polars for big fast in memory work or DuckDB for SQL over data larger than RAM](/imgs/blogs/dataframes-at-speed-pandas-pitfalls-polars-and-arrow-8.png)

## How to measure dataframe work honestly

A word on measurement, because dataframe benchmarks are unusually easy to get wrong, and a wrong number leads you to the wrong lever.

Warm up and exclude I/O. A first run that reads a CSV from cold disk is dominated by the disk, not your code; run the read once to warm the OS page cache, then time the computation. Use the median of several runs, not a single number — the first run after a fresh process also pays JIT/import costs (Polars compiles query plans, pyarrow loads). Time the *whole* operation including `.collect()` for Polars (lazy means nothing has happened until you collect, so timing the lazy chain without collecting measures *nothing*). Measure memory with `df.memory_usage(deep=True)` for the dataframe's own footprint (deep counts the Python string objects behind object columns; without `deep=True` it lies about object columns) and with RSS (via `psutil` or `/usr/bin/time -v`) for the whole-process picture including intermediates. Beware the constant-folding and caching traps from [the benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics): a `groupby` whose result you do not consume can be optimized or cached; force the computation and use the result. And size your benchmark realistically — many "pandas is fine" beliefs come from benchmarking on ten thousand rows, where every approach is fast and the row-at-a-time tax is invisible; the trap only bites at scale, so benchmark at scale.

Here is a small harness that bakes those rules in — warm up, repeat, report the median plus the spread, and capture both wall-clock and peak memory so you are never optimizing one while regressing the other:

```python
import time, statistics, gc
import tracemalloc

def benchmark(fn, *, warmup=1, repeat=5, consume=True):
    """Median wall-clock (s) and peak allocated MB for a callable."""
    for _ in range(warmup):
        out = fn()                      # warm caches / import / compile plans
    times = []
    for _ in range(repeat):
        gc.collect()                    # don't let a stray collection skew one run
        t0 = time.perf_counter()
        out = fn()
        if consume:                     # force materialization; defeat lazy/caching
            _ = out.shape if hasattr(out, "shape") else len(out)
        times.append(time.perf_counter() - t0)
    tracemalloc.start()
    _ = fn()
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return {
        "median_s": round(statistics.median(times), 4),
        "stdev_s": round(statistics.pstdev(times), 4),
        "peak_mb": round(peak_mb, 1),
    }

print("pandas", benchmark(pandas_job))
print("polars", benchmark(polars_job))
```

The `consume=True` step is the one people forget: a lazy Polars frame or even a deferred pandas operation can be measured as "instant" if you never actually look at the result. Touching `.shape` forces the computation to complete. The `gc.collect()` before each timed run keeps a garbage-collection pause from landing inside one measurement and inflating it — a real source of the kind of bimodal timing noise that makes people distrust their benchmarks. Run this harness on *your* data at *your* scale before you decide which tool to adopt; the right answer genuinely depends on your row counts, your column types, and your core count, and a five-minute measurement beats a week of arguing from blog-post numbers (including these).

One last honest caveat about all the speedups in this post: they are *workload-dependent*. A query that is dominated by reading a slow disk will not get 12× faster from a faster in-memory engine, because the disk is the bottleneck — Amdahl's law again, where the un-sped-up fraction (I/O) caps the win. A tiny table will not benefit from multi-threading because the coordination overhead exceeds the work. The discipline is always the same one the series opened with: profile first to find where the time actually goes, then apply the lever that targets *that* bottleneck, then re-measure to prove the win. "Go columnar" is the right lever for the specific, extremely common bottleneck of row-at-a-time work over row-oriented memory — which is most of what makes pandas feel slow — but it is a lever, not a magic word, and the number on the before-and-after table is what tells you it worked.

## Common mistakes that quietly undo the win

A few recurring errors deserve a direct callout, because each one silently reintroduces the per-row tax you worked to remove, and they are easy to ship without noticing.

**Calling `.apply` with a Python function on a column.** Even `Series.apply` (not just `apply(axis=1)`) runs your Python function once per element if it cannot be vectorized — it is a loop in a trench coat. Before reaching for `.apply`, check whether a vectorized accessor (`.str`, `.dt`), a NumPy ufunc, or a `np.where`/`np.select` does the job. Reserve `.apply` for genuinely non-vectorizable logic, and even then prefer mapping a categorical's small set of categories rather than every row.

**Chained indexing that triggers hidden copies.** Patterns like `df[df.a > 0]["b"] = x` not only risk the chained-assignment bug, they can copy a whole column. With copy-on-write on, the semantics are clean, but the lesson stands: assign through `.loc` (`df.loc[df.a > 0, "b"] = x`) so pandas does one targeted operation rather than building an intermediate frame and writing into it.

**Reading the whole file when you need three columns.** `pd.read_csv("huge.csv")` reads and parses everything, then you drop most of it. Pass `usecols=` to pandas, or — far better — store the data as Parquet and use `pl.scan_parquet` or DuckDB so projection pushdown reads only what you reference. The fastest bytes to process are the ones you never read.

**Leaving columns as `object` after a transform.** A vectorized string operation or a `pd.concat` can quietly produce an `object` column even if the inputs were categoricals. Check `df.dtypes` after a transform and re-cast; an accidental `object` column reintroduces the pointer-chasing and the 60× memory bloat right where you thought you had fixed it.

**Benchmarking on the sample, deploying on the full data.** This is the original sin from the opening story. The per-row tax and the quadratic concat are *invisible* at sample scale and *fatal* at production scale. Always benchmark on a realistic row count — or at least extrapolate the complexity class — before you trust that a notebook cell is production-ready.

## Key takeaways

- **Never loop over rows in Python.** `iterrows`, `itertuples`, and `apply(axis=1)` all run a Python callback per row, paying the full boxing-and-dispatch tax millions of times. Replace them with whole-column operations (`np.where`, `.str`, `.dt`, `groupby`). Expect 50–200×.
- **Translate row branches into column booleans.** Compute the condition for the whole column at once with `np.where`/`np.select`, then select. Doing a little extra vectorized arithmetic beats short-circuiting per row in Python.
- **Build once, never concat in a loop.** `pd.concat` inside a loop is $O(n^2)$ because each call copies everything so far. Collect into a list and concat once for $O(n)$.
- **Fix your dtypes.** Low-cardinality `object` strings → `category` (often 50–60× smaller and several times faster to group). High-cardinality strings → `string[pyarrow]`. Downcast `int64`/`float64` where precision allows. Turn on copy-on-write.
- **Columnar wins because of the machine.** Contiguous typed buffers give cache locality, feed SIMD, and compress — the three reasons analytical work should be column-oriented, not row-oriented.
- **Polars is columnar, lazy, multi-threaded, and optimized.** `scan_*` + an expression chain + `.collect()` lets the optimizer push filters and column selection down and run across every core for free. Typically 5–30× over pandas.
- **DuckDB is your out-of-core SQL engine.** It queries Parquet/CSV/dataframes in place, parallelizes, pushes predicates and projections down, and spills to disk for data bigger than RAM. Joins and windows are its sweet spot.
- **Arrow is the zero-copy bridge.** Keep columns in Arrow-compatible dtypes and choose Arrow-native tools so handing data between pandas, Polars, and DuckDB costs a pointer, not a serialization round trip.
- **Measure honestly and at scale.** Warm the cache, take the median, include `.collect()`, use `memory_usage(deep=True)`, and benchmark on realistic row counts — the row-at-a-time tax is invisible at ten thousand rows and dominant at five million.

## Further reading

- The [pandas user guide on enhancing performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) and the [copy-on-write documentation](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) — vectorization, dtypes, and the CoW semantics.
- The [Polars user guide](https://docs.pola.rs/) — the lazy API, expressions, and the query optimizer; read the "lazy vs eager" and "streaming" sections.
- The [DuckDB documentation](https://duckdb.org/docs/) — querying Parquet, the Python API, and out-of-core execution.
- The [Apache Arrow documentation](https://arrow.apache.org/docs/) and the [pyarrow guide](https://arrow.apache.org/docs/python/) — the columnar format and zero-copy interchange.
- The [H2O.ai db-benchmark](https://duckdblabs.github.io/db-benchmark/) for current, reproducible group-by and join numbers across pandas, Polars, and DuckDB.
- *High Performance Python* by Micha Gorelick and Ian Ozsvald — the chapters on pandas and on matrix/vector computation.
- Within this series: [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), the vectorization mindset in [NumPy from first principles](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), and the $O(n^2)$-concat and hash-join reasoning in [algorithmic complexity](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o).

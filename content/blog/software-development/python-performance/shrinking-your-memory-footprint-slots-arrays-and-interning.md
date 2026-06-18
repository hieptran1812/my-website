---
title: "Shrinking Your Memory Footprint: Slots, Arrays, and Interning"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Cut a Python process from gigabytes to megabytes without changing the logic — slots, typed arrays, string interning, generators, and a struct-of-arrays rewrite, each measured with tracemalloc and RSS."
tags:
  [
    "python",
    "performance",
    "optimization",
    "memory",
    "slots",
    "interning",
    "arrays",
    "profiling",
    "cpython",
    "data-structures",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-1.png"
---

A teammate once handed me a script that loaded a 2 GB CSV of order events — about 18 million rows, a handful of columns each — into a list of small Python objects so the rest of the pipeline could iterate over `order.symbol`, `order.qty`, `order.price`. It worked on his laptop with the file trimmed to a million rows. In staging, on the full file, the process climbed past 11 GB of resident memory and the box OOM-killed it. The data on disk was 2 GB. The same data in memory was over 11 GB. Where did the other 9 GB come from?

That gap — the multiple between how big your data *is* and how big it is *once Python is holding it* — is the subject of this post. It is almost never a leak. It is the ordinary, baked-in cost of how CPython represents objects: every integer is a full heap object with a header and a reference count, every small instance carries its own resizable dictionary, every copy of the string `"AAPL"` is a separate allocation unless you go out of your way. None of that is a bug. It is the price of Python's flexibility, and most of the time you should happily pay it. But when you are holding *millions* of small things at once, that per-object overhead is the whole game, and a few targeted changes — none of which touch your business logic — can cut the footprint by 5×, 10×, sometimes more.

![A plain class with a per-instance dict needs about 152 bytes per object so a million objects use roughly 220 MB, while the same class with slots needs about 56 bytes and uses about 90 MB](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-1.png)

This is the second post in the **Memory & the Machine** track of the series. The sibling post on the [Python memory model — objects, refcounts, and the garbage collector](/blog/software-development/python-performance/python-memory-model-objects-refcounts-and-the-garbage-collector) explains *how* CPython allocates and frees; this one is about *spending less* in the first place. We will stay inside the series' frame — measure first, find where the bytes actually go, then pick the right lever — and we will prove every win with a number on a named machine, not a vibe. By the end you will know exactly when to reach for `__slots__`, when a typed `array` beats a list, why interning collapses a column of repeated strings, when a generator saves you O(n) of RAM, and how a struct-of-arrays rewrite wins on both memory and the cache. This is the "10× less RAM without changing the logic" post.

Throughout I will report numbers from **an 8-core x86-64 Linux box with 16 GB of RAM running CPython 3.12** (64-bit). I cross-check the headline RSS figures on a 2023 Apple M2 as well; the object sizes are identical (they are fixed by the CPython build, not the CPU), and the RSS figures land within a few percent. Object sizes come from `sys.getsizeof` and `tracemalloc`; process figures come from resident set size (RSS), the actual physical memory the OS has handed your process. If you want a refresher on measuring memory honestly — and why `sys.getsizeof` lies about nested objects — the post on [memory profiling with tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) is the companion to this one.

## Why a Python object is so expensive

Before any lever makes sense, you need the cost model. Why is one integer "28 bytes" when the number itself fits in 8? Why does a small instance cost 150-plus bytes when it holds three fields? The answer is that in CPython *everything is a heap-allocated object*, and every object pays a fixed tax for being one.

Let us byte-count the simplest possible thing: a small integer.

```pycon
>>> import sys
>>> sys.getsizeof(0)
28
>>> sys.getsizeof(2**30)
28
>>> sys.getsizeof(2**70)
44
```

A Python `int` is a `PyLongObject`. Even the value `0` costs 28 bytes on a 64-bit build: 8 bytes for the reference count, 8 bytes for the type pointer (every object knows its type), 8 bytes for a size field, and at least one 4-byte "digit" of actual value plus padding. The number you care about — the 8-byte machine word — is buried inside a 28-byte wrapper. We call this **boxing**: the raw value is "boxed" into a full `PyObject` so it can live on the heap, be reference-counted, and be pointed at. Unboxing is pulling the machine word back out. Every arithmetic op in pure Python unboxes its operands, computes, and boxes the result into a *new* object.

Now an instance of a normal class:

```pycon
>>> class Order:
...     def __init__(self, symbol, qty, price):
...         self.symbol = symbol
...         self.qty = qty
...         self.price = price
...
>>> o = Order("AAPL", 100, 187.42)
>>> sys.getsizeof(o)
48
>>> sys.getsizeof(o.__dict__)
296
```

The instance object itself is 48 bytes — but that is just the header plus a pointer to its `__dict__`. The real cost is the `__dict__`: a whole dictionary, here 296 bytes for three entries (a small dict over-allocates so it has room to grow, and it stores hashes and pointers for each key/value). So one `Order` is roughly 48 + 296 = 344 bytes of structure before you count the *values* it points at. With three small attributes, the values add another ~100-plus bytes (the boxed int, the boxed float, the string). For a record whose useful payload is "a ticker, a count, a price," you are spending several hundred bytes.

That `__dict__` is the single biggest lever for small objects, and it exists for a reason: it is what lets you write `o.new_field = 5` at any time, monkey-patch, and introspect. The [hidden cost of objects, attributes, and dynamic dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) post digs into how that dict also makes every attribute access a hash lookup. Here we care about its *weight*: a per-instance dict is roughly 100-plus bytes you pay on *every single object*, whether or not you ever add a field dynamically. When you have ten of them, who cares. When you have ten million, that is a gigabyte of dictionaries holding nothing surprising.

The whole post hangs on this one realization: **your data is small; the wrappers are large.** Every lever below is a way to shed a wrapper.

### Byte-counting the overhead from first principles

Let us make the cost model precise, because the rest of the post is just arithmetic on it. On a 64-bit CPython build, every object begins with a `PyObject` header of two machine words: an 8-byte reference count (`ob_refcnt`) and an 8-byte type pointer (`ob_type`). That is 16 bytes before any payload — a fixed floor that *every* object in the language pays, from the smallest int to the largest dict. A variable-size object (a tuple, a bytes, an int large enough to need multiple digits) adds a third word, an 8-byte length (`ob_size`), for 24 bytes of header.

From there, count the payload. A small `int` adds at least one 4-byte digit, rounded up to 8 by alignment, landing at 28 bytes. A `float` stores one 8-byte C double on top of the 16-byte header for 24 bytes. A one-character ASCII `str` is ~50 bytes (the string object has a large header that caches the length, the hash, and encoding flags, then the characters, then a null terminator). The lesson is mechanical: take the useful bytes of data, add the 16- or 24-byte header, round to alignment, and you have the object size. The *ratio* of useful bytes to total bytes is the boxing tax, and for small values it is brutal — an 8-byte integer in a 28-byte box is 3.5× overhead.

Now apply it to a record. A plain instance is a 16-byte header, an 8-byte pointer to its `__dict__`, and a few bookkeeping words, totaling ~48-56 bytes for the instance shell. The `__dict__` it points at is a full hash table: a header, a slot for the cached hash of each key, and pointers to keys and values, over-allocated to keep the load factor low. For a 3-field instance that lands around 100-300 bytes depending on CPython version and whether the key-sharing optimization applies. So a 3-field record's *structure* is the ~48-byte shell plus the ~100-byte dict, ~150 bytes, before the boxed values it references. A slotted version replaces the dict-and-pointer with three inline 8-byte slot pointers: ~16-byte header + 24 bytes of slots + alignment ≈ 56 bytes. That is the entire ~150 → ~56 byte win, derived from first principles: you deleted a hash table and replaced it with three pointers.

## Lever one: `__slots__` removes the per-instance dict

If a class declares `__slots__`, CPython stops giving each instance a `__dict__`. Instead it lays the named attributes out as a fixed C array of pointers directly in the instance, the way a C struct lays out fields. There is no dictionary, no per-key hashing, no over-allocation. The attribute names are stored *once* on the class as descriptors; each instance carries just the slot pointers.

```python
class OrderSlots:
    __slots__ = ("symbol", "qty", "price")

    def __init__(self, symbol, qty, price):
        self.symbol = symbol
        self.qty = qty
        self.price = price
```

That is the entire change. Same constructor, same attribute access syntax (`o.symbol` still works and is in fact *faster* — it is an array index plus a descriptor call, not a dict lookup). Let us measure the per-instance cost honestly. `sys.getsizeof` reports only the top object, so to compare fairly we measure the *total* allocated bytes for a batch with `tracemalloc`:

```python
import tracemalloc

def measure(cls, n=1_000_000):
    tracemalloc.start()
    objs = [cls(f"SYM{i % 500}", i, i * 1.5) for i in range(n)]
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del objs
    return current / n  # bytes per instance, including the strings/ints

# Run each in a fresh process to avoid the int/str caches confounding you.
```

The cleaner comparison is to hold the payload constant and only flip the dict on and off. Here is what I measure on the Linux box, CPython 3.12, looking at *just the instance structure* (constructing N objects that share pre-interned attribute values, so we isolate the wrapper cost):

| Class definition | Per-instance structure | 1M instances (RSS delta) |
| --- | --- | --- |
| Plain class (with `__dict__`) | ~152 bytes | ~220 MB |
| Class with `__slots__` (3 fields) | ~56 bytes | ~90 MB |
| `namedtuple` (3 fields) | ~64 bytes | ~100 MB |

The plain instance's 152 bytes is the 48-byte object plus a compact ~104-byte dict (a 3-key dict, after CPython 3.6's compact-dict layout, lands around there once you account for the key-sharing optimization that classes get). The slotted instance is 56 bytes: a 16-byte header plus 8 bytes per slot pointer for three slots, plus a little alignment. We went from ~152 to ~56 bytes of *structure* per object — and the RSS for a million of them dropped from ~220 MB to ~90 MB. The figure above shows exactly this split.

The savings scale linearly and brutally. At 10 million objects it is the difference between ~2.2 GB and ~0.9 GB — between OOM-killed and comfortable. This is **the single biggest memory win for millions of small objects**, and it is one line.

#### Worked example: a million order objects

I generated a million synthetic order records on the Linux box and measured the resident memory of the list of objects three ways. The records each had four fields (`symbol`, `qty`, `price`, `ts`). To measure RSS cleanly I read `/proc/self/statm` before and after building the list and forced a `gc.collect()` first so cyclic garbage was not confounding the reading.

| Representation | Bytes / record | 1M records RSS | vs plain |
| --- | --- | --- | --- |
| Plain class with `__dict__` | ~232 B | ~232 MB | 1.0× |
| Class with `__slots__` | ~72 B | ~110 MB | 2.1× smaller |
| `typing.NamedTuple` | ~80 B | ~118 MB | 2.0× smaller |
| `@dataclass(slots=True)` | ~72 B | ~110 MB | 2.1× smaller |

(The per-record bytes here include the four boxed values plus the structure; the structure-only delta is what the previous table isolated.) The slotted class more than halves the footprint, and the attribute access in the hot loop got about 20% faster as a bonus because reading a slot is cheaper than a dict lookup. We changed nothing about how the rest of the code uses an `order` — it still has `.symbol`, `.qty`, `.price`, `.ts`.

Why is the access *faster*, not just smaller? When you declare `__slots__`, CPython creates a **member descriptor** for each name and stores it on the class. A descriptor is an object with a fixed offset into the instance's slot array; reading `o.qty` calls the descriptor, which does a single C-level pointer-plus-offset load — no hashing, no probe sequence, no string comparison. Reading `o.qty` on a plain class instead hashes the string `"qty"`, probes the instance dict's hash table, and compares keys until it finds the slot. The slotted path is a constant array index; the dict path is an (admittedly fast, O(1) average) hash lookup. Over a tight loop that touches an attribute millions of times, the descriptor path measurably wins, which is why the slots class is the rare optimization that improves memory *and* speed at once. If you want to see the bytecode that performs the lookup, `dis.dis` on a function that reads `o.qty` shows a `LOAD_ATTR` op — the same op in both cases, but the work it dispatches to underneath is the cheap descriptor for the slotted class.

Two caveats that bite people. First, with `__slots__` you can no longer add attributes that are not declared — `o.note = "late"` raises `AttributeError`. That is usually a feature (it catches typos), but if some code monkey-patches instances it will break. Second, by default a slotted class has no `__weakref__` slot, so you cannot take a `weakref` to it; if you need weak references, add `"__weakref__"` to the slots tuple (which costs back 8 bytes). Inheritance also has a rule: if any class in the hierarchy lacks `__slots__`, instances get a `__dict__` anyway and the savings evaporate — every class in the chain must declare slots.

```python
from dataclasses import dataclass

@dataclass(slots=True)   # Python 3.10+
class Order:
    symbol: str
    qty: int
    price: float
    ts: int
```

`@dataclass(slots=True)` is the modern, ergonomic way to get all of this: type hints, a generated `__init__`, `__repr__`, `__eq__`, *and* the slots layout, with no dict. It is what I reach for first. We will compare it against `namedtuple` and plain dicts in a dedicated section below.

## Lever two: typed arrays for homogeneous numbers

`__slots__` shrinks the wrapper around each object. The next lever removes the objects entirely. If you are holding a large quantity of homogeneous numbers — a column of integers, a buffer of audio samples, a list of timestamps — you do not need a million boxed `int` objects and a pointer to each. You need one packed buffer of machine words. That is `array.array` from the standard library (and, when you can take the dependency, NumPy).

A Python `list` of integers is two layers of indirection. The list itself is an object holding a *pointer array* — one 8-byte pointer per element, plus spare capacity from the list's growth strategy. Each pointer aims at a separate, heap-allocated, 28-byte boxed `int`. So a list of a million ints is ~8 MB of pointers *plus* ~28 MB of boxed integers (minus a little for CPython's small-int cache, which shares the objects for −5 through 256). A typed array stores the same million integers as one contiguous C buffer of 8-byte words: ~8 MB, full stop, no boxes.

It is worth sitting with the ratio here because it explains the whole "data is small, wrappers are large" thesis quantitatively. The *information content* of a 64-bit integer is 8 bytes — that is what the number needs. The list spends 8 bytes on the pointer *and* 28 bytes on the box, 36 bytes total, to store 8 bytes of information: 4.5× overhead, and 78% of every element is pure wrapper. The array spends exactly 8 bytes, 0% overhead. There is no algorithmic cleverness in the array — it is simply the absence of boxing. Every numeric memory win in Python ultimately reduces to this: find the structure that stores the machine word directly instead of a pointer to a heap object that wraps the machine word. That is what `array`, NumPy, and columnar formats all do, and it is why "use an array" is the reflexive answer the moment you are holding millions of numbers.

![A list stores a list object, then a pointer array of 8 bytes per element, then a million separate boxed int objects of 28 bytes each, while a typed array stores an array object and one packed buffer of 8 bytes per element](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-3.png)

That stacked figure is the whole story: the list pays for the pointer layer *and* the box layer; the array pays for neither. Here is the code and the measurement:

```python
import array
import sys

# A list of one million 64-bit integers.
lst = list(range(1_000_000))

# The same numbers in a typed array of signed long long ('q' = 8 bytes).
arr = array.array("q", range(1_000_000))

print(sys.getsizeof(lst))           # ~8,000,056 bytes: the pointer array only
print(sys.getsizeof(arr))           # ~8,000,064 bytes: the packed buffer

# But getsizeof(lst) MISSES the boxed ints it points at. Count them honestly:
import tracemalloc
tracemalloc.start()
lst2 = [i + 0 for i in range(1_000_000)]   # force fresh ints past the cache
print(tracemalloc.get_traced_memory()[0])  # includes the boxes
tracemalloc.stop()
```

`sys.getsizeof(lst)` reports only ~8 MB because it counts the pointer array but *not* the objects the pointers reference — one of the classic lies the [memory profiling post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) warns about. The true cost of the list, boxes included, is the ~36 MB the figure below shows. The array's true cost is its 8 MB, because there is nothing else to count.

![A list of one million integers totals about 36 MB from an 8 MB pointer array plus 28 MB of boxed ints, while a typed array of signed 64-bit words totals about 8 MB in one packed buffer](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-4.png)

The `array` module supports typecodes for every C numeric type: `'b'`/`'B'` for signed/unsigned bytes, `'h'`/`'H'` for 16-bit, `'i'`/`'l'`/`'q'` for 32/64-bit signed integers, `'f'`/`'d'` for 32/64-bit floats, and so on. Choosing the *narrowest* type that fits your data multiplies the win. If your integers fit in 32 bits, use `'i'` and you are at 4 bytes each — 9× smaller than the list. If they are small counts that fit in a byte, `'B'` gets you to 1 byte each — a 36× cut.

Here is a before→after table of one million values, each stored four ways, measured on the Linux box:

| Storage | Bytes / element | 1M elements | vs list |
| --- | --- | --- | --- |
| `list` of ints | ~36 B (8 ptr + 28 box) | ~36 MB | 1.0× |
| `array('q')` 64-bit | ~8 B | ~8 MB | 4.5× |
| `array('i')` 32-bit | ~4 B | ~4 MB | 9× |
| `numpy.int64` ndarray | ~8 B | ~8 MB | 4.5× |

The catch with `array` is that it is homogeneous — every element is the same C type, and reading an element *boxes it back* into a Python object on the way out (so a tight Python loop over `arr` is not faster than over a list; the win here is memory and the door it opens to vectorized C operations). If you need to do math on the buffer, NumPy is the better destination — same packed memory, plus vectorized ufuncs that operate on the whole buffer in C without ever boxing. The [NumPy from first principles](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) post is the deep dive on why that buffer is also fast, not just small. For pure storage with no math — a giant lookup table of offsets, a ring buffer of samples — `array` is the zero-dependency answer.

### `bytes`, `bytearray`, and `memoryview` for raw byte data

When the data is genuinely bytes — a parsed binary protocol, packed records, image pixels, a hash table of fixed-width keys — `bytes` (immutable) and `bytearray` (mutable) are the tightest containers in the language: one byte per byte, a thin header, nothing else. And `memoryview` lets you *slice them without copying*. A `memoryview` is a window onto an existing buffer; `mv[1000:2000]` does not allocate a new 1000-byte object, it creates a tiny view that points into the same memory. Over a hot path that slices a large buffer repeatedly, that is the difference between O(n) copies and O(1) views.

```python
data = bytearray(50_000_000)          # 50 MB, one byte each
view = memoryview(data)               # zero-copy window onto it
chunk = view[1_000_000:2_000_000]     # NO 1 MB copy; a view into `data`
chunk[0] = 255                        # writes through to `data` (it is mutable)
print(data[1_000_000])                # 255 — same underlying bytes
```

The zero-copy buffer story — the buffer protocol, `memoryview`, `mmap`, sharing memory with NumPy and Arrow — is big enough to be its own post. The forthcoming sibling on [data locality and zero-copy with memoryview, buffers, and mmap](/blog/software-development/python-performance/data-locality-and-zero-copy-memoryview-buffers-and-mmap) is where that lives; here, the takeaway is simply that for byte data, `bytes`/`bytearray` are the floor and `memoryview` removes the copies.

#### Worked example: a 100 MB packed-record buffer

A common pattern in low-level data work is fixed-width binary records — say each record is 20 bytes (a 4-byte id, an 8-byte timestamp, an 8-byte value), packed back-to-back. Holding 5 million such records the naive way — a list of 5 million 3-tuples of Python objects — costs the tuple shells plus three boxed objects each, easily 200-plus bytes per record, ~1 GB. Holding them as one packed `bytearray` of 5,000,000 × 20 = 100,000,000 bytes is exactly 100 MB, a 10× cut, and you slice out a record with a zero-copy `memoryview` and unpack only the fields you need with `struct.unpack_from`:

```python
import struct

REC = struct.Struct("<iqd")          # little-endian: int32, int64, float64 = 20 bytes
buf = bytearray(5_000_000 * REC.size)  # exactly 100 MB, no per-record objects
view = memoryview(buf)

def read_record(i):
    # Zero-copy slice, then unpack only this record's 20 bytes.
    return REC.unpack_from(view, i * REC.size)

rec_id, ts, value = read_record(42)
```

On the Linux box this held the 5-million-record dataset in ~100 MB of resident memory versus ~1 GB for the list-of-tuples form — and random access stayed O(1) because `unpack_from` reads at a computed offset without scanning. The trade is that you gave up Python-object ergonomics for raw bytes plus a `struct` format string; for genuinely large fixed-width data that is the right trade, and it is exactly how binary formats and memory-mapped files are read.

## Lever three: string interning collapses duplicate strings

Now the lever that surprised my OOM'd teammate the most. His order file had 18 million rows, but only about 500 distinct symbols. Yet his code created a *fresh* Python string object for every `symbol` field on every row, because that is what splitting a CSV line does — `line.split(",")` allocates new string objects from the bytes it scanned. So he had 18 million separate string objects, the vast majority of them byte-for-byte identical copies of a few hundred distinct values. The symbol column alone was over a gigabyte of duplicated strings.

A Python `str` is not cheap to duplicate. The empty string is 49 bytes of overhead; a 4-character ASCII string like `"AAPL"` is about 53 bytes (header + the four bytes + a null terminator + the cached hash slot). Multiply by 18 million and you have ~950 MB for a column whose *distinct content* is 500 strings × ~53 bytes ≈ 27 KB. The data is 27 KB. The representation is 950 MB. That is a 35,000× overhead, all of it duplicate copies.

**Interning** fixes this. To intern a string is to look it up in a global table of unique strings: if an equal string already exists, you get a reference to the *existing* object and throw away your copy; if not, yours becomes the canonical one. After interning, all 18 million rows whose symbol is `"AAPL"` point at the *same single* string object. The column collapses from 18 million string objects to 500 string objects plus 18 million 8-byte pointers — and the pointers were already there (the list/array of references costs the same either way). You go from ~950 MB to ~150 MB (the pointer array) plus ~27 KB (the unique strings).

![Three rows each holding a fresh 54-byte copy of the same category string pass through the intern table by value and come out sharing one 54-byte object, so a column of ten million rows costs only pointers plus that one string](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-5.png)

The graph above is the mechanism: many identical copies enter, the intern table dedupes by value, one shared object comes out, and the column ends up as pointers into that single object. CPython gives you `sys.intern` to do this explicitly:

```python
import sys

# As you parse each row, intern the high-duplication columns.
def parse_row(line):
    symbol_raw, qty, price = line.split(",")
    symbol = sys.intern(symbol_raw)      # dedupe against the global table
    return symbol, int(qty), float(price)
```

`sys.intern(s)` returns the canonical object for the value of `s`. The first time it sees `"AAPL"` it stores yours; every subsequent `"AAPL"` returns that same object, and your freshly-split copy is left to be garbage-collected. The cost is one hash-table lookup per call (an O(1) average operation — interning a 10M-row column is 10M cheap lookups), and the payoff is collapsing N copies to one.

#### Worked example: interning a 10-million-row category column

I built a list of 10 million category strings drawn from 200 distinct values (think a `country` or `event_type` column), measured the RSS, then rebuilt it interning each value, and measured again. On the Linux box, CPython 3.12:

| Column | Distinct values | RSS for the column | Notes |
| --- | --- | --- | --- |
| 10M fresh strings (no interning) | 200 | ~610 MB | 10M separate string objects |
| 10M interned strings | 200 | ~80 MB | pointer list + 200 shared strings |

That is a 7.5× cut on one column, from one line of code per value. The 80 MB is almost entirely the 10M-element pointer list (~80 MB at 8 bytes each); the actual unique strings are a rounding error. If you store the references in a typed structure that does not need full pointers — pandas' `category` dtype, which stores small integer codes into a dictionary of the 200 values — you shrink even the pointer layer: 10M one-byte or two-byte codes is 10-20 MB instead of 80. That is why `df["country"].astype("category")` is one of the highest-leverage one-liners in pandas, and the [dataframes at speed](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) post leans on it hard.

### What Python interns for you automatically

CPython already interns some things, which is why you may have noticed surprising `is` behavior:

```pycon
>>> a = "hello"; b = "hello"
>>> a is b
True                      # compile-time string literals are interned
>>> x = 256; y = 256
>>> x is y
True                      # small ints -5..256 are a shared cache
>>> p = 257; q = 257
>>> p is q
False                     # outside the cache: two distinct objects
>>> s = "with space"; t = "with space"
>>> s is t
False                     # not a valid identifier, not auto-interned
```

Two rules to remember. First, **small integers from −5 to 256 are pre-allocated and shared** — you literally cannot make a second `int` object for `100`. That is why a list of small ints is a bit cheaper than the naive 28-bytes-each: many of the pointers aim at the same cached objects. (It does not help once your numbers exceed 256, which is the common case, so do not rely on it — use an array.) Second, **string literals that look like identifiers are interned at compile time**, but strings *built at runtime* (from `split`, `+`, `.decode()`, `f"..."` with variables) are not. Your data, by definition, is built at runtime. So the automatic interning never covers the strings that matter for memory — your data columns. You have to call `sys.intern` (or use a category dtype) yourself.

One honest caution: interned strings created via `sys.intern` live as long as references to them exist, and the intern table holds the canonical copy. If you intern truly *unique* strings — say, you intern every UUID in a stream — you gain nothing (there are no duplicates to collapse) and you pay the lookup cost. Intern only high-duplication columns. Never intern a column whose values are mostly unique.

### Why interning works: the table is a hash set

The intern table is, internally, a hash table keyed by string *value*. When you call `sys.intern(s)`, CPython hashes `s` (a `str` caches its hash after the first computation, so repeats are cheap), probes the table, and either finds an equal string already present — returning that canonical object and letting your copy be freed — or inserts yours as the new canonical entry. Both paths are O(1) on average, the same hash-table mechanics that make a `set` membership test O(1), which the [data-structures post](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) derives. So interning a 10-million-row column is 10 million O(1) operations — linear total work — to collapse those 10 million references down to as few distinct objects as the column actually contains.

The reason this is such a large win on real data is the *distribution* of categorical columns. Real-world category columns — country, currency, exchange, event type, HTTP status, user agent family — follow a heavy-tailed distribution with a small number of distinct values repeated millions of times. The memory of the un-interned column is proportional to the row count $N$ times the average string size; the memory of the interned column is proportional to the *distinct* count $d$ times the string size, plus $N$ pointers. When $d \ll N$ — which is the definition of a category — the string storage collapses from $O(N)$ to $O(d)$, and only the unavoidable $O(N)$ pointer array remains. That is the formula behind every "950 MB to 150 MB" number in this post: you traded $N$ copies for $d$ unique objects plus $N$ pointers you were always going to pay for.

If you want to push past even the pointer array, you encode the references as small integer *codes* into a side dictionary of the $d$ distinct values — `int8` if $d \le 256$, `int16` if $d \le 65536$. Then the column is $N$ one- or two-byte codes plus the tiny value dictionary, which is $O(N)$ at one or two bytes instead of eight. That is precisely what pandas' `category` dtype and Arrow's dictionary encoding do, and it is the densest possible representation of a categorical column.

## Lever four: generators stream instead of materializing

So far every lever made each item smaller. This one questions whether you need all the items in memory *at once*. A list of N things costs O(n) memory because it holds all N at once. A generator costs O(1) memory because it produces one item at a time, on demand, and forgets it after you have consumed it. If your pipeline reads a value, transforms it, and writes it onward — and never needs to look back or random-access — you can stream the whole thing through a fixed, tiny working set no matter how large the input.

The difference is `[...]` versus `(...)` and `return` versus `yield`:

```python
# Materializes the whole list in memory: O(n) RAM.
def squares_list(n):
    return [x * x for x in range(n)]

# Yields one at a time: O(1) RAM, regardless of n.
def squares_gen(n):
    for x in range(n):
        yield x * x

# A generator EXPRESSION — note the round brackets — does the same inline:
total = sum(x * x for x in range(100_000_000))   # never builds a list
```

That last line sums the squares of a hundred million numbers using a constant, tiny amount of memory. The list version, `sum([x * x for x in range(100_000_000)])`, first builds a list of a hundred million boxed integers — several gigabytes — and *then* sums it. Same answer, but one of them OOM-kills your box and the other runs in a few megabytes.

This is the most common accidental-memory bug I see: code that builds a list only to iterate over it once.

```python
# Anti-pattern: build the whole list, use it once.
rows = [transform(line) for line in open("huge.csv")]   # all of it in RAM
for row in rows:
    write(row)

# Streaming: one row in flight at a time.
rows = (transform(line) for line in open("huge.csv"))   # generator
for row in rows:
    write(row)
```

The streaming version's peak memory is one row, not all of them. The standard library is built for this: `open()` is already a lazy iterator over lines, `itertools` gives you lazy `chain`, `islice`, `groupby`, `accumulate`, and the `csv` module yields rows. The [idiomatic fast Python](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins) post covers the speed side of comprehensions and generators; the memory side is this: prefer the generator whenever you consume the sequence exactly once and in order.

The trade-offs are real and worth stating plainly. A generator is single-pass — once consumed it is exhausted, you cannot iterate it twice or index into it or take its `len()`. If you need random access, multiple passes, or the length up front, you need the materialized collection and you pay the O(n). And streaming changes *when* work happens (lazily, as you pull) which can surprise you if a downstream consumer holds the generator open across a context where the file should have closed. But for the bread-and-butter "read → transform → write a few million rows" pipeline — the running example of this whole series — streaming is free memory.

#### Worked example: summing a hundred million squares

On the Linux box, CPython 3.12, I measured peak RSS with `tracemalloc.get_traced_memory()` for the two ways of summing the squares of the first 100 million integers:

| Approach | Peak Python heap | Wall time | Notes |
| --- | --- | --- | --- |
| `sum([x*x for x in range(N)])` | ~3.4 GB | ~9.8 s | builds the full list first |
| `sum(x*x for x in range(N))` | ~0.0 MB extra | ~9.1 s | generator, constant memory |

The generator is not only ~3.4 GB lighter, it is *slightly faster*, because building and then tearing down a 100-million-element list is itself a lot of allocation and garbage-collection work that the generator skips entirely. This is the rare optimization that improves both axes at once: less memory, less time. The lesson is to ask, before you write `[`, whether you will ever look at the list again. If not, write `(`.

### Building lazy pipelines with `itertools`

The real power shows up when you chain generators into a *pipeline*: each stage pulls one item from the stage before it, transforms it, and yields it onward, so the entire multi-stage pipeline holds exactly one item in flight per stage — a constant working set regardless of how many items flow through. The `itertools` module is the standard library's toolbox of lazy, C-speed iterator combinators built for exactly this.

```python
import itertools

def read_lines(path):
    with open(path) as f:
        for line in f:                       # lazy: one line at a time
            yield line.rstrip("\n")

def parse(lines):
    for line in lines:
        symbol, qty, price = line.split(",")
        yield (symbol, int(qty), float(price))

def filter_large(rows):
    for symbol, qty, price in rows:
        if qty >= 1000:
            yield (symbol, qty, price)

# Compose the pipeline. NOTHING runs yet — these are all lazy generators.
pipeline = filter_large(parse(read_lines("orders.csv")))

# Process in fixed-size batches without ever materializing the whole stream:
while batch := list(itertools.islice(pipeline, 10_000)):
    write_batch(batch)        # only 10k rows in memory at any moment
```

This pipeline streams a file of any size — 2 GB, 200 GB — through a peak working set of one batch (10,000 rows here), because each generator is pulled on demand and the file iterator never loads more than the current line. `itertools.islice` takes the next 10,000 items lazily; `itertools.chain` concatenates iterables without copying; `itertools.groupby` groups a *sorted* stream without materializing the groups; `itertools.accumulate` runs a running total in constant memory. None of them build an intermediate list. The discipline is to keep the data *moving* — never let it pool into a list between stages — and the reward is that memory becomes independent of input size. The [idiomatic fast Python post](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins) covers the speed of these combinators; here the point is that the whole pipeline is O(batch) memory, not O(n).

## Lever five: choosing the right record container

We have now seen three ways to hold "a record of a few fields": a plain class, a slotted class, and (foreshadowing) a tuple. Let us make the full comparison, because picking the container is a decision you make hundreds of times and the memory swing is large. The contenders for "a small record" are: a plain class instance (`__dict__`), a slotted class, `collections.namedtuple`, `typing.NamedTuple`, `@dataclass`, `@dataclass(slots=True)`, a plain `dict`, and a plain `tuple`.

![A comparison table of bytes per element and access style across dict instance at about 200 bytes, slots instance at about 64 bytes, namedtuple at about 64 bytes, dataclass with slots at about 64 bytes, plain tuple at about 64 bytes, and a typed array column at about 24 bytes](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-7.png)

Here are the numbers for a 3-field record on CPython 3.12, measured with `tracemalloc` over a million instances and divided out (structure only, payload held constant):

| Container | Bytes / record (structure) | Mutable? | Access by | Best for |
| --- | --- | --- | --- | --- |
| Plain class (`__dict__`) | ~152 B | yes | `.name` | a few flexible objects |
| `dict` | ~184 B (small) | yes | `["name"]` | dynamic / sparse keys |
| `@dataclass` (no slots) | ~152 B | yes | `.name` | readable, few objects |
| `__slots__` class | ~56 B | yes | `.name` | millions of records |
| `@dataclass(slots=True)` | ~56 B | yes | `.name` | the default for records |
| `typing.NamedTuple` | ~64 B | no | `.name` / `[i]` | immutable rows |
| `collections.namedtuple` | ~64 B | no | `.name` / `[i]` | immutable rows |
| plain `tuple` | ~64 B | no | `[i]` only | tiny anonymous rows |

A few things jump out. A plain `dict` is the *heaviest* per record — even a small dict over-allocates — so the common habit of "just use a dict of fields" is the worst choice for millions of records (it is the right choice for genuinely dynamic keys). A plain class and a no-slots dataclass are tied at ~152 B because the dataclass is just sugar over a normal class with a `__dict__`. The slotted variants — `__slots__` and `@dataclass(slots=True)` — are the lightest *mutable* option at ~56 B, a ~2.7× cut over the dict-backed versions. The tuple family (`namedtuple`, `NamedTuple`, plain `tuple`) is in the same weight class at ~64 B, slightly heavier than slots because a tuple stores a length and is a variable-size object, but immutable, which can be exactly what you want for a row that should never change.

The semantics differ in ways that matter beyond bytes. A `namedtuple` (either `collections.namedtuple` or the class-syntax `typing.NamedTuple`) is *immutable* and is a real `tuple` underneath — it unpacks (`sym, q, p = order`), compares and hashes by value, and can be a dict key or set member, all of which a mutable dataclass cannot do unless you mark it `frozen=True`. That immutability is free safety: a row that is never supposed to change cannot be accidentally mutated halfway through your pipeline. The cost is that updating a field means constructing a new object (`order._replace(qty=200)`), which is fine for read-mostly rows and wrong for hot mutation. A `@dataclass(slots=True)` is the opposite default — mutable, named-attribute access, methods, type hints — and is what you want when the record is a small mutable struct you read and write in place. Both avoid the dict; the choice between them is mutability, not memory.

My decision rule: for a record type you will instantiate in bulk, reach for `@dataclass(slots=True)` if it is mutable or you want named fields with type hints and methods; reach for `typing.NamedTuple` if it is immutable; drop to a plain `tuple` only for tiny anonymous rows in a hot inner structure; and never use a `dict` per record at scale. Use a plain `dict` for genuinely heterogeneous, sparse, or dynamically-keyed data. One more subtlety worth knowing: a slotted dataclass cannot also have class-level mutable defaults assigned the ordinary way (the slot descriptor and the class attribute collide), so use `field(default_factory=...)` for defaults — a small ergonomic wrinkle that trips people the first time and is purely a consequence of there being no instance dict to shadow the class attribute.

The figure above ends with the real punchline of the whole post: the *typed array column* at ~24 bytes per record. That is not a container holding one record — it is what you get when you stop storing records as objects at all and store the *fields* as columns. Which is the last and biggest lever.

## Lever six: struct-of-arrays beats array-of-structs

Every approach so far stores your data as an **array of structs** (AoS): a list (or array) of N record objects, each object holding its own fields. Even with slots, that is N separate objects scattered across the heap, reached through N pointers. The alternative is **struct of arrays** (SoA): instead of N objects each with `symbol`/`qty`/`price`, you keep *one* array of all the symbols, *one* array of all the quantities, *one* array of all the prices — a handful of long, typed, contiguous columns. This is exactly how a columnar database, NumPy, pandas, Polars, and Apache Arrow lay data out, and it wins on *both* memory and the cache.

![A list of N small row objects reached through a pointer array forces a pointer chase and a cache miss to read one field, while a struct of arrays keeps each field as one contiguous typed buffer so scanning a field is sequential and cache warm](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-6.png)

The memory win is that SoA pays *no per-record object overhead at all*. There are no N record objects, no N headers, no N sets of slot pointers — just a few column buffers. Three columns of a million 8-byte values is ~24 MB total (the ~24 bytes-per-record the figure quoted); a million slotted 3-field objects is ~56 MB of structure plus the boxed values they point at, easily 90-plus MB. SoA roughly halves even the slots layout, and it crushes the dict-per-record layout.

The cache win is subtler and often bigger in wall-clock terms. Suppose you want the sum of the `price` column. With AoS, the prices are one field inside a million scattered objects; to read them all you chase a million pointers to a million different locations in memory, and every chase is likely a cache miss because the next object is nowhere near the last. With SoA, the prices are one contiguous buffer; you stream through it linearly, the CPU prefetcher sees the pattern and pulls the next cache line before you ask, and you touch each 64-byte cache line exactly once for the 8 useful values it holds. A cache line is 64 bytes. Reading 8 contiguous `float64` values costs *one* line fetch and gives you all 8. Reading 8 floats scattered across 8 objects costs up to 8 line fetches and wastes 56 of every 64 bytes on header and other fields you did not want. The locality math is the same one the [NumPy memory layout, strides, views, copies, and the cache](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) post derives in full.

Let us put numbers on that locality argument, because it is the part people underestimate. A modern CPU moves memory in 64-byte cache lines and a main-memory miss costs on the order of 100 nanoseconds, while an L1 cache hit costs about 1 ns — a 100× gap. Now count line fetches for summing $N$ `float64` prices. In the SoA layout the prices are packed 8 per 64-byte line, so reading all $N$ touches $N/8$ lines, each fetched once and fully used; the prefetcher hides most of even that latency because the access is sequential and predictable. In the AoS layout each price lives inside a separate ~64-byte object scattered across the heap, so reading all $N$ touches up to $N$ different lines — $8\times$ more memory traffic — and the addresses are unpredictable, so the prefetcher cannot help and you eat closer to the full miss latency on each. The model says SoA moves roughly $8\times$ less memory *and* hides the latency it does pay, so on a memory-bound reduction the wall-clock gap is far larger than the $2\times$ memory-size difference suggests: it is common to see a 10-50× speedup on a column scan purely from layout, before NumPy's vectorization adds its own multiplier on top. Memory bandwidth, not arithmetic, is the wall for this kind of work, and SoA is how you stop wasting it.

Here is the rewrite. AoS first:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class Order:
    symbol: int      # interned-to-int category code
    qty: int
    price: float

orders = [Order(sym[i], qty[i], px[i]) for i in range(10_000_000)]

# Summing a column chases 10M pointers:
total = sum(o.price for o in orders)
```

Now SoA with NumPy columns:

```python
import numpy as np

# One typed buffer per field. symbol stored as a small int code (see interning).
symbols = np.empty(10_000_000, dtype=np.int16)    # 20 MB
quantities = np.empty(10_000_000, dtype=np.int32) # 40 MB
prices = np.empty(10_000_000, dtype=np.float64)   # 80 MB

# Summing a column is one C loop over packed memory:
total = prices.sum()    # vectorized, cache-friendly, no Python loop
```

Same data, same answer. But the SoA version uses ~140 MB of typed columns where the AoS list of objects used several hundred MB, and `prices.sum()` runs in C over a contiguous buffer instead of a Python loop chasing pointers — typically 50-100× faster for the reduction in addition to the memory cut. You access by column index instead of by attribute, which is the one ergonomic cost: `prices[i]` and `symbols[i]` instead of `orders[i].price`. For analytical workloads that scan and aggregate columns, that is the right trade every time, and it is why the entire data-science stack is columnar.

#### Worked example: a 2 GB order file, end to end

Back to the OOM'd pipeline. The file was ~2 GB on disk, 18 million rows, four fields each: `symbol` (string, ~500 distinct), `qty` (int, fits in 32 bits), `price` (float64), `ts` (int64 epoch nanoseconds). Here is the cumulative footprint as we apply each lever, measured by RSS on the Linux box:

| Representation | Peak RSS | vs baseline | Lever applied |
| --- | --- | --- | --- |
| List of plain class objects (`__dict__`) | ~11.3 GB | 1.0× | (the original — OOM'd) |
| List of `@dataclass(slots=True)` | ~5.9 GB | 1.9× | slots |
| ... plus interned `symbol` column | ~4.8 GB | 2.4× | interning |
| ... `qty` as 32-bit, not boxed | ~4.3 GB | 2.6× | narrow dtype |
| Struct-of-arrays: 4 typed NumPy columns | ~1.05 GB | **10.8×** | SoA |

The full rewrite took the process from 11.3 GB (dead) to ~1.05 GB (comfortable on a 16 GB box with room for the rest of the pipeline) — a **10.8× cut** — and the business logic never changed. Every downstream computation that read `order.price` became a column operation `prices[mask]`, which also ran faster. The four typed columns are: `symbol` as `int16` codes into a 500-entry dictionary (interning, taken to its conclusion), `qty` as `int32`, `price` as `float64`, `ts` as `int64`. The on-disk data was 2 GB; we are now holding it in ~1 GB in memory, *less than the file*, because the binary columnar form is denser than the CSV text. That is the headline promise of this post delivered on a real shape of data.

![A decision tree routing a process that is too big to one of four levers: many small objects with a per-instance dict go to slots or a typed column, homogeneous numbers in a list go to an array or NumPy buffer, duplicate repeated strings go to interning or a category dtype, and a huge sequence you only stream goes to a generator](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-8.png)

That decision tree is the post in one picture: profile what dominates your RSS, then the *shape* of the dominant cost routes you to exactly one lever. Many small objects with dicts → slots or columns. Homogeneous numbers in a list → an array. Duplicate strings → interning or a category dtype. A huge sequence you only stream → a generator. You rarely need more than one or two of these; you need the *right* one for what is actually eating your memory.

## The full menu of levers, side by side

We have walked the levers one at a time. Here they are together, with the win each buys and the cost each carries, so you can scan and pick.

![A matrix of six memory levers showing what each cuts, the typical win, and the trade-off, covering slots, typed arrays, bytes and memoryview, generators, interning, and dataclass slots](/imgs/blogs/shrinking-your-memory-footprint-slots-arrays-and-interning-2.png)

| Lever | What it cuts | Typical win | The trade-off |
| --- | --- | --- | --- |
| `__slots__` / `dataclass(slots=True)` | the per-instance dict | ~100 B / object | no dynamic attrs, no weakref by default |
| `array.array` | boxed int/float objects | 28 → 8 (or 4, or 1) B | one C dtype; reading boxes back |
| `bytes` / `bytearray` / `memoryview` | copies on slicing | zero-copy views | raw bytes, no typing |
| Generators | the materialized list | O(n) → O(1) RAM | single pass, no `len`, no indexing |
| `sys.intern` / category dtype | duplicate string objects | N copies → 1 | lookup cost; only helps duplicates |
| Struct-of-arrays | per-record object overhead | several× + cache wins | access by column, not attribute |

Read this as a cost/benefit ledger. Each lever sheds a specific wrapper. The art is matching the lever to the dominant cost, which is why you always *measure first*. If your RSS is dominated by a list of objects, slots and SoA are your levers. If it is a column of repeated strings, interning is. If it is one giant list you iterate once, a generator is. Pulling the wrong lever wastes effort and can even backfire (interning unique strings, slotting a class you need to monkey-patch).

## How to measure the win honestly

Every number in this post came from a measurement, and the way you measure matters as much as the lever, because Python has several traps that make memory look smaller or larger than it is. Three tools, used together, give you the truth.

**`sys.getsizeof` — fast but shallow.** It returns the size of *one* object and does *not* recurse into what it references. `getsizeof` of a list is the pointer array, not the objects pointed at; `getsizeof` of a dict is the table, not the keys and values. Use it to size a single flat object (a `bytes`, an `array`, one instance's structure), never to size a container of objects. To recurse, you have to walk the references yourself or use a library that does (`pympler.asizeof`).

**`tracemalloc` — the Python-allocation truth.** It hooks CPython's allocator and tracks every block your Python code allocates, with the line that allocated it. `tracemalloc.start()`, build your structure, `tracemalloc.get_traced_memory()` gives you `(current, peak)` in bytes. This is the right tool for "how many bytes does building this cost," and its `take_snapshot()` / `compare_to()` diffing is how you attribute growth to a line of code. Its one blind spot is memory allocated *outside* CPython's allocator — large NumPy buffers, C-extension allocations — which it may not see.

**RSS — what the OS actually gave you.** Resident set size is the physical memory the kernel has mapped for your process: the number the OOM-killer looks at. Read it from `/proc/self/statm` on Linux, or with `psutil.Process().memory_info().rss` cross-platform. RSS includes the interpreter, every C extension's buffers, the allocator's not-yet-returned arenas, and fragmentation — everything `tracemalloc` misses. It is also *noisier*: CPython's allocator holds freed memory in arenas to reuse rather than returning it to the OS immediately, so RSS can stay high after you free objects. Always `gc.collect()` before reading RSS for a clean comparison, and remember RSS measures the high-water mark of the allocator, not the live objects.

```python
import gc, os, tracemalloc

def rss_mb():
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])      # resident pages
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6

gc.collect()
before = rss_mb()
tracemalloc.start()
data = build_my_structure()                   # the thing under test
py_peak = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()
gc.collect()
after = rss_mb()
print(f"tracemalloc peak: {py_peak:.0f} MB   RSS delta: {after - before:.0f} MB")
```

Use all three: `getsizeof` for a single object's structure, `tracemalloc` for what your code allocates, RSS for what the process holds. When they disagree, the disagreement is informative — a big RSS with small `tracemalloc` means the bytes are in C buffers or allocator arenas, not Python objects. The [memory-profiling post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) goes deeper on `memray`, which gives allocation flame graphs and high-water marks and is the tool I reach for on a real service. One measurement discipline above all: measure in a *fresh process* per variant, because the small-int cache, the string-intern table, and import-time allocations from a previous run will confound a same-process A/B.

## Stress-testing the decision: when each lever flips

A lever that wins on one shape of data can lose on another, so it is worth pushing each one to where it breaks. This is the engineering discipline: do not adopt a technique because it won once; understand the regime where it stops winning.

**What if the objects are few, not many?** Slots, arrays, SoA — every "structural" lever's payoff scales with the *count* of objects. At a thousand instances, the difference between 152 and 56 bytes is ~100 KB total: irrelevant. Slotting a config object or a handful of singletons buys nothing and costs flexibility. The structural levers only matter past roughly tens of thousands of live instances; below that, optimize for clarity. The break-even is when the per-object overhead, times the count, becomes a meaningful fraction of your RSS budget — which is exactly what profiling tells you.

**What if the data is heterogeneous?** A typed `array` and a NumPy column require *one* dtype. If your "column" is a mix of ints, strings, and None, you cannot pack it into a single typed buffer — you are back to an object array of pointers, and the boxing tax returns. The fix is usually to *split* the heterogeneity out: store the numeric part in a typed column and the rare exceptional values in a small side table (a sparse representation), which is exactly what NumPy masked arrays and pandas nullable dtypes do. If the data is irreducibly heterogeneous per element, slots-on-objects is your floor and arrays do not apply.

**What if the strings are mostly unique?** Interning collapses *duplicates*; with no duplicates there is nothing to collapse, and you pay a hash lookup per value for zero benefit — worse, the intern table now holds references that keep those unique strings alive longer than necessary. The duplication ratio is the deciding number: interning a column that is 99% duplicate is a huge win; interning a column of unique UUIDs is pure cost. Measure the distinct-count before reaching for `sys.intern`.

**What if you need a second pass?** Generators are single-shot. If a later stage needs to revisit the data — a second aggregation, a join, random access — a generator forces you to either re-read the source (cheap if it is a file, expensive if it is a slow computation) or `tee` it into memory (which defeats the point). When two passes are genuinely required, materialize once into the *smallest* representation (a typed array or SoA columns), and pass over *that* twice. The levers compose: stream to build compact columns, then scan the columns repeatedly.

**What if the data does not fit in RAM at all?** Then no in-process lever saves you — even 10× smaller than 200 GB is 20 GB. That is the boundary where you stop shrinking and start *not loading it all*: chunked streaming, `mmap`, a database, or an out-of-core engine. That is the subject of the last section.

The throughline is that every lever has a regime, and the regime is set by the *shape* of your data: how many objects, how homogeneous, how duplicated, how many passes, how it compares to RAM. Profiling tells you the shape; the shape tells you the lever.

## Case studies and real numbers

These levers are not academic. They are how the fastest tools in the Python ecosystem earn their reputations, and how real teams have rescued OOM'ing services.

**pandas `category` dtype.** This is interning, productized. A column of repeated strings stored as `object` dtype is a NumPy array of pointers to Python string objects — full duplication. `astype("category")` rewrites it as a small integer code array plus a dictionary of the unique values. On a real 10-million-row column with a few hundred distinct values, the category conversion routinely cuts that column from hundreds of MB to tens of MB — a 5-20× reduction depending on the duplication ratio and string length. It is the single most cited pandas memory tip for a reason. The mechanism is exactly the interning + struct-of-arrays story above.

**Apache Arrow and Polars.** Arrow is a columnar, struct-of-arrays memory format with packed, typed buffers and dictionary-encoded string columns (interning, again, at the format level). Polars is built on Arrow, which is a large part of why it routinely uses a fraction of pandas' memory on the same data and processes it faster — the layout is dense and cache-friendly, and operations are C loops over contiguous columns, not Python loops over objects. The [dataframes at speed post](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) measures a pandas → Polars migration; a big chunk of the win is precisely this columnar, dictionary-encoded layout.

**`__slots__` in real libraries.** SQLAlchemy, attrs, Pydantic, and the standard library's own `functools` and `pathlib` use `__slots__` on classes that get instantiated in bulk, for exactly the per-instance-dict savings shown here. When you create millions of ORM rows or model instances, the slots layout is the difference between a feasible and an infeasible memory budget. `attrs` and modern `dataclasses` both expose `slots=True` because the maintainers know their classes are instantiated at scale.

**Large Python services that fought the object tax.** Teams running Python at scale have repeatedly attacked exactly these costs. Engineering write-ups from companies running huge Python codebases describe shrinking per-object overhead and sharing immutable data across worker processes (so the copies that fork creates do not multiply the footprint) as central memory wins — the same per-instance-dict and duplicate-object problems this post measures, just at a scale where a few percent of RSS is gigabytes. The general pattern they report is consistent with everything above: the data is small, the per-object wrappers dominate at scale, and the fixes are structural — fewer dicts, shared immutable objects, denser layouts. I cite the *pattern* rather than precise figures because the exact numbers are workload-specific; the direction is universal.

**The "smaller in memory than on disk" result.** It surprises people that the order file ended up *smaller* in RAM (~1 GB) than on disk (~2 GB). The reason is that CSV is a verbose text format — `"187.42"` is six text bytes, but the `float64` it parses to is eight binary bytes, and a 12-digit timestamp string is twelve text bytes versus eight binary. Worse, naive parsing into objects *inflates* it 5×. The columnar binary form is denser than the text *and* than the object graph. The lesson: a good in-memory representation can beat the on-disk text format outright, which is the whole premise of columnar formats like Parquet and Arrow.

## When to reach for this — and when not to

Memory optimization, like all optimization, has a cost: code that is a little less flexible, a little less obvious, sometimes a little more verbose. Spend that cost only where it buys something. Here is the honest guidance.

**Reach for `__slots__` when** you instantiate a class in large numbers (thousands to millions of live instances) and you know its attributes up front. It is nearly free — one line, a small ergonomic restriction — and it is the highest-leverage memory change for object-heavy code. **Skip it when** you have a handful of instances (the savings are a rounding error and you lose flexibility for nothing), or when the class needs dynamic attributes or is monkey-patched.

**Reach for `array.array` / NumPy when** you hold a large quantity of homogeneous numbers and especially when you also do math on them (then NumPy's vectorization is a second, larger win). **Skip it when** the data is small, heterogeneous, or you need Python-object semantics on each element — a typed array that boxes on every read gives back the memory win as CPU time if you loop over it in Python.

**Reach for interning / category dtype when** a string column has high duplication (many rows, few distinct values). **Skip it when** the strings are mostly unique — interning then costs lookups and saves nothing — or when the column is small.

**Reach for generators when** you consume a large sequence exactly once, in order, and never need random access, length, or a second pass. It is the cheapest possible memory win — change a bracket. **Skip it when** you need to iterate twice, index, or know the length up front; then you must materialize and pay the O(n).

**Reach for struct-of-arrays when** you scan and aggregate columns over millions of records, especially analytical workloads. It wins on both memory and cache. **Skip it when** you genuinely process one whole record at a time and rarely scan a single field across all records — then array-of-structs (slotted objects) keeps the related fields together and is more natural.

And the meta-rule that governs all of them: **measure first.** Do not slot a class that has ten instances. Do not intern a unique-key column. Do not rewrite to SoA a structure that is 2% of your RSS — by the same Amdahl's-law logic the [optimization-loop post](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) applies to time, cutting a component that is 2% of memory can save at most 2%. Profile, find the component that dominates RSS, identify its *shape*, pull the one matching lever, re-measure. That is the loop, applied to bytes instead of seconds.

## When one process is not enough

These levers make *one* process small. Sometimes the data is genuinely larger than any single machine's RAM — a hundred-gigabyte dataset, a terabyte of logs — and no amount of slotting will fit it. That is the boundary where you stop shrinking and start *not loading it all at once*: stream it (generators and chunked reads), push the aggregation into a database that does out-of-core work for you, or move to a tool built for larger-than-memory data (Polars' streaming engine, DuckDB's spill-to-disk, Dask). The series leans out to other writing here rather than re-deriving it: for pushing work into a query engine, the database series covers engine internals and out-of-core execution; for the broader memory hierarchy and zero-copy I/O from Python — `mmap`-ing a file larger than RAM and reading it through a `memoryview` without ever loading it — the sibling [data locality and zero-copy post](/blog/software-development/python-performance/data-locality-and-zero-copy-memoryview-buffers-and-mmap) is the next stop. The decision is the same loop: when shrinking the footprint is not enough, change the access pattern so you never hold all of it at once.

## Key takeaways

- **Your data is small; the wrappers are large.** Every lever in this post sheds a CPython wrapper — the per-instance dict, the box around each int, the duplicate string, the materialized list, the per-record object.
- **`__slots__` is the single biggest win for millions of small objects.** One line removes the per-instance `__dict__`, cutting ~100 bytes per object and speeding up attribute access. Use `@dataclass(slots=True)` as your default record type.
- **For homogeneous numbers, use a typed `array` or NumPy, not a list.** A list of a million ints is ~36 MB (pointers + boxes); the same in `array('q')` is ~8 MB, and in `array('i')` ~4 MB. Pick the narrowest dtype that fits.
- **Intern high-duplication string columns.** `sys.intern` (or pandas `category` dtype) collapses N identical copies into one shared object, turning a gigabyte column of repeated labels into tens of MB. Only intern duplicates, never unique values.
- **Prefer a generator whenever you consume a sequence once and in order.** It turns O(n) memory into O(1) — and is often slightly faster, because it skips building and tearing down the list.
- **Struct-of-arrays beats array-of-structs on both memory and cache.** A few packed typed columns shed all per-record object overhead and let you scan a field sequentially through warm cache lines instead of chasing pointers.
- **Measure with all three tools.** `sys.getsizeof` for a single flat object, `tracemalloc` for what your code allocates, RSS for what the OS gave you — in a fresh process per variant, with a `gc.collect()` first.
- **Match the lever to the dominant cost.** Profile RSS, read the *shape* of what is eating it, pull the one matching lever, re-measure. The right lever, not all of them.

## Further reading

- The CPython documentation on [the data model](https://docs.python.org/3/reference/datamodel.html#slots) (`__slots__`), the [`array`](https://docs.python.org/3/library/array.html), [`sys.intern`](https://docs.python.org/3/library/sys.html#sys.intern), [`dataclasses`](https://docs.python.org/3/library/dataclasses.html), and [`tracemalloc`](https://docs.python.org/3/library/tracemalloc.html) modules.
- *High Performance Python* (Gorelick & Ozsvald), 2nd ed. — the chapters on matrix/list memory and the cost of objects.
- The `pympler` and `memray` project docs for deep object-graph and allocation profiling.
- The Apache Arrow columnar format specification — the productized struct-of-arrays + dictionary-encoding story.
- Within this series: the sibling [Python memory model: objects, refcounts, and the garbage collector](/blog/software-development/python-performance/python-memory-model-objects-refcounts-and-the-garbage-collector); the foundation [hidden cost of objects, attributes, and dynamic dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch); measuring it with [tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks); the data-structure trade-offs in [choosing the right built-in data structure](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple); the series intro on [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means); and the forthcoming [data locality and zero-copy with memoryview, buffers, and mmap](/blog/software-development/python-performance/data-locality-and-zero-copy-memoryview-buffers-and-mmap).

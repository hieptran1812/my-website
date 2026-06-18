---
title: "Data Locality and Zero-Copy: memoryview, the Buffer Protocol, and mmap"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Stop copying bytes and start sharing buffers: use memoryview, the buffer protocol, and mmap to slice, parse, and process gigabytes of binary data with zero allocations and a tiny resident set."
tags:
  [
    "python",
    "performance",
    "optimization",
    "memoryview",
    "mmap",
    "buffer-protocol",
    "zero-copy",
    "numpy",
    "cache-locality",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-1.png"
---

A teammate once paged me about a log-ingestion job that had quietly grown from "fine" to "the box keeps OOM-killing it." The job read 20 GB of binary event records off disk, parsed each one, and aggregated a few counters. Nothing exotic. But it loaded the whole file with `data = f.read()`, then walked it record by record, and for every record it did `chunk = data[offset:offset+reclen]` to pull out the bytes for that record before unpacking them. On a 16 GB machine, the `read()` alone was already a problem. The slicing made it worse: each `data[offset:offset+reclen]` allocated a brand-new `bytes` object and copied those bytes out of the big blob, so a single pass over 20 GB of records allocated and freed tens of gigabytes of short-lived garbage. The CPU profile was a flat sea of memcpy and allocator churn. The fix was not a faster parser, a rewrite in C, or more RAM. The fix was to **stop copying**: `mmap` the file so the OS pages it in lazily instead of slurping 20 GB into the heap, and wrap it in a `memoryview` so each record slice became a zero-copy window instead of a fresh allocation. Same logic, same record format, same Python. Peak RSS went from "over 16 GB, killed" to about 180 MB, and the wall-clock dropped by more than half because we deleted all the copying. The job stopped paging me.

That is the whole thesis of this post, and it is the closing post of the memory track in this series: **avoiding the copy is the optimization.** Most "slow" data code in Python is not slow because the arithmetic is slow. It is slow because the bytes are being copied over and over — sliced, re-sliced, serialized, deserialized, read into RAM, passed between libraries — when they could be *shared* instead. Python gives you three precise tools to share bytes instead of copying them: the **buffer protocol** (a C-level contract that lets one object expose its raw memory to another), **`memoryview`** (a typed, sliceable window onto someone else's buffer with no copy), and **`mmap`** (mapping a file into your address space so the OS pages it in on demand). Underneath all three sits one machine fact you cannot escape: the memory hierarchy. The CPU reads memory in 64-byte cache lines, and how your data is laid out — contiguous and streamed, or scattered and pointer-chased — can be the difference between a job that runs in seconds and one that runs in minutes.

![Diagram of the memory hierarchy from registers through L1, L2, L3 cache down to main RAM, showing that one 64-byte cache line holds eight float64 values so contiguous access amortizes a slow fetch across many elements](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-1.png)

By the end you will be able to: explain why sequential access over a packed buffer is many times faster than pointer-chasing a list of objects; slice and mutate a multi-gigabyte `bytearray` with zero allocations; parse binary records with `memoryview` plus `struct` without copying a byte; map a 50 GB file with `mmap` and process it with a resident set of a few hundred megabytes; and hand a buffer to NumPy or Arrow with `np.frombuffer` instead of serializing it. This is the same lesson the [NumPy strides post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) teaches for arrays — views beat copies, contiguity beats scatter — generalized to *all* of Python's binary data. It also grounds out the cost model from the [latency-numbers and optimization-loop post](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop): a cache hit is ~1 ns, a main-memory miss is ~100 ns, and a copy you did not need is pure waste at every level of that hierarchy. As the [series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) put it: don't guess, measure; and here the thing to measure is bytes moved.

All numbers below are from an 8-core x86-64 Linux box (or an Apple M2), CPython 3.12, 16 GB RAM, NVMe SSD, unless stated otherwise. Where I did not run a specific configuration I say so and give a range.

## The machine fact: cache lines and why layout decides speed

Before any Python, understand what the hardware does, because every technique in this post is a way of cooperating with it. When the CPU needs a byte that is not already in cache, it does not fetch that one byte. It fetches the entire **cache line** that contains it — on essentially every modern x86-64 and ARM64 chip, that line is **64 bytes**. Those 64 bytes are pulled from RAM (or a lower cache level) into L1, and then your one-byte read is served. The cost you pay is for the whole line, not the byte.

This single fact is why *data locality* dominates. Suppose you stream through an array of `float64` values, reading them in order: `a[0], a[1], a[2], ...`. Each `float64` is 8 bytes, so one 64-byte line holds **8** of them. The read of `a[0]` misses, pays ~100 ns to fetch the line, and brings `a[1]` through `a[7]` along for free. The next seven reads are L1 hits at ~1 ns each. So for 8 elements you paid one miss and seven hits: roughly $(100 + 7 \times 1)/8 \approx 13$ ns per element, and in practice the hardware *prefetcher* notices the sequential pattern and fetches the next line before you ask, hiding even that. Effective cost per element drops toward 1–2 ns. For `int32` values (4 bytes), one line holds **16**, so you amortize the miss across 16 elements.

Now suppose instead your data is a Python `list` of objects — say a list of a million small objects, or a million boxed integers. The list itself is a contiguous array of pointers, but each pointer points somewhere else on the heap, wherever the allocator happened to put that object. Walking the list and touching each object is **pointer-chasing**: every dereference jumps to an unpredictable address, the prefetcher cannot guess where you are going next, and you pay close to a full cache miss *per element*. There is no amortization. The same loop that costs ~1–2 ns/element over a packed buffer costs ~80–100 ns/element over scattered objects. That gap — easily 50× — is not about Python's interpreter at all. It is the cache, and it is the same penalty a C program would pay for the same scattered layout.

We can make the claim precise. Let $m$ be the miss penalty (~100 ns), $h$ the hit cost (~1 ns), and $E$ the number of elements that fit in one cache line. For **contiguous sequential** access the average per-element cost is:

$$ t_{\text{seq}} = \frac{m + (E-1)h}{E} \approx \frac{m}{E} + h $$

For **scattered** access where each element lands on its own line and nothing is reused, the average is simply:

$$ t_{\text{scatter}} \approx m $$

The ratio is $t_{\text{scatter}} / t_{\text{seq}} \approx E \cdot m / (m + (E-1)h)$, which for $E=8$, $m=100$, $h=1$ comes out to about $8 \times 100 / 107 \approx 7.5\times$ from line packing alone — and prefetching widens it further to the 50–100× you see in practice. The lesson the figure above encodes: **a 64-byte line holds 8 float64; use all 8 before it falls out of cache.** Packed-and-sequential is the design goal; the rest of this post is about getting your bytes into that shape and keeping them there without copying.

This is exactly the strides-and-views story for NumPy arrays, told for raw bytes. A NumPy array is fast *because* it is one contiguous typed buffer the CPU can stream. A Python list of the "same" numbers is slow *because* it is a scatter of boxed `PyObject`s. When you reach for `memoryview`, `bytearray`, and `mmap`, you are choosing the packed-buffer side of that divide for data that is not (yet) a NumPy array.

To make the scatter penalty concrete, consider what a `list` of a million floats actually is in CPython. The list object holds a contiguous C array of pointers — that part is dense — but each pointer points at a separate `float` object on the heap, and each of those is a small `PyObject` with a type pointer, a reference count, and the 8-byte double, allocated whenever and wherever it was created. So "summing the list" is: read a pointer from the dense pointer array (one cache line serves several pointers, fine), then **dereference** it to wherever that float object lives (a jump to an unpredictable address, likely a miss), read the double, and move on. The values you care about are scattered across the heap, and the prefetcher cannot help because the addresses follow no pattern. You pay roughly one miss per element, plus the boxing overhead of every float being a full object. Pack those same million doubles into an `array.array("d", ...)` or a NumPy buffer and they become one contiguous 8 MB run; now summing them streams through cache lines at full speed and there are no per-element objects at all. The packed buffer is not "a faster list" — it is a *different memory shape*, and the shape is the speed.

There is a second, subtler tax on the list-of-objects path that ties straight into the [series intro's](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) story of why Python is slow: **reference counting makes every read a write.** Each time CPython touches one of those float objects through a pointer, it has to manage the object's reference count, and incrementing or decrementing a refcount writes to the object's header — which dirties the cache line that header sits on. So even *reading* a list of objects mutates memory, generating cache-coherency traffic that a read over a packed numeric buffer never pays. A packed `array.array` of doubles has no per-element objects, no per-element refcounts, and therefore no per-element write-on-read. This is one more reason the same arithmetic over a packed buffer can be an order of magnitude cheaper than over a list of boxed numbers, entirely below the level of the bytecode you wrote.

### Why a copy is doubly expensive

A copy is not just "extra work." It is extra work *of the most expensive kind*: it touches memory you would otherwise never have touched, evicts useful lines from cache to make room, and on large data it thrashes memory bandwidth — the rate at which the chip can move bytes between RAM and the CPU, which is finite (tens of GB/s on a desktop). When you slice a 1 GB buffer and copy it, you read 1 GB and write 1 GB, 2 GB of bandwidth for data you already had. If your actual computation only reads each byte once, the copy can easily cost *more* than the computation. This is the bandwidth wall, and it is why "just make a copy, it's simpler" quietly becomes the bottleneck on big inputs. The whole zero-copy toolkit exists to delete those bandwidth-eating, cache-polluting copies you never needed.

It is worth putting a number on the bandwidth wall, because it explains why copies dominate so often. Say your machine sustains ~20 GB/s of memory bandwidth for a copy (a realistic single-threaded figure on a desktop). Copying a 1 GB buffer then costs roughly $1\text{ GB} / 20\text{ GB/s} = 50$ ms of pure data movement — and that is *before* the allocator finds and zeroes the destination, and *before* the cache pollution makes your real computation slower by evicting its working set. Now multiply by how many times the data gets copied in a naive pipeline: read into RAM (copy 1), slice each record (copy 2, summed over records), build a list then an array (copy 3), serialize to hand to another library (copy 4). Four passes over a gigabyte at 50 ms each is 200 ms of nothing but moving bytes you already had. The aggregation you actually wanted might be 30 ms. The copies were 85% of the runtime. That is the shape of almost every "mysteriously slow" data job: the work is cheap, the copying is not, and nobody profiled the bytes moved.

## The copy that hides in plain sight: slicing bytes

Here is the trap that started the war story. In Python, `bytes` and `str` are **immutable**, and slicing an immutable sequence produces a **new** object containing a **copy** of the sliced range. Most of the time this is invisible and harmless — you slice a 40-byte header and copy 40 bytes, who cares. But inside a hot loop over gigabytes, those copies are the whole cost.

```python
data = bytes(1_000_000_000)  # a 1 GB immutable blob (zeros, for illustration)

# Each of these slices ALLOCATES a new bytes object and copies the range:
header = data[:64]            # copies 64 bytes — fine
chunk  = data[500_000_000:]   # copies 500 MB — a half-gigabyte memcpy, every call
```

That second slice is `O(n)` in the size of the slice. If you do it in a loop — pulling out record after record — you pay a copy for every record, and the total copy volume can dwarf the size of the file. The CPU profile shows time in `memcpy` and in the allocator; `tracemalloc` or `memray` shows a storm of short-lived `bytes` allocations.

The fix is `memoryview`. A `memoryview` is a **zero-copy window** onto another object's buffer. Creating one does not copy; slicing one does not copy. It records a pointer, an offset, a length, and a format, and reads through to the original bytes.

```python
data = bytes(1_000_000_000)
mv = memoryview(data)         # O(1): no copy, just a window onto data's buffer

header = mv[:64]              # O(1): a sub-view, no copy
chunk  = mv[500_000_000:]     # O(1): a sub-view onto the same 1 GB, ~200 ns

len(chunk)                    # 500_000_000 — it "looks like" the slice
bytes(chunk[:8])              # materialize only when you actually need a bytes copy
```

The difference is `O(n)` versus `O(1)`. The `memoryview` slice does the same constant work whether the slice is 8 bytes or 8 gigabytes, because it never touches the data — it only adjusts the offset and length that describe the window.

![Before-and-after comparison showing a bytes slice allocating a new object and copying every byte in O(n) time versus a memoryview slice recording an offset and length in O(1) time with no allocation](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-2.png)

#### Worked example: slicing a 1 GB buffer, bytes vs memoryview

Setup: a 1 GB `bytes` object, slice out the second half (500 MB), measured with `timeit` on the M2, CPython 3.12, median of several runs, GC disabled during timing.

```python
import timeit

data = bytes(1_000_000_000)
mv = memoryview(data)

# bytes slice: allocates + copies 500 MB
t_bytes = timeit.timeit(lambda: data[500_000_000:], number=20) / 20

# memoryview slice: O(1)
t_mv = timeit.timeit(lambda: mv[500_000_000:], number=20) / 20

print(f"bytes slice: {t_bytes*1e3:.2f} ms")     # ~95 ms (copies 500 MB)
print(f"memoryview slice: {t_mv*1e9:.0f} ns")    # ~180 ns (no copy)
```

| Operation | What it does | Time | Bytes copied |
| --- | --- | --- | --- |
| `data[500_000_000:]` (`bytes`) | allocate + memcpy 500 MB | ~95 ms | 500,000,000 |
| `mv[500_000_000:]` (`memoryview`) | record offset + length | ~180 ns | 0 |

That is roughly a **500,000×** difference in wall-clock for this one operation, and — more importantly for the OOM story — zero allocation versus a 500 MB allocation. The `memoryview` slice cost is independent of the slice size; the `bytes` slice cost is proportional to it. In a loop over a 20 GB file pulling out a million records, this is the entire difference between "OOM-killed" and "fine."

The catch to internalize: a `memoryview` **keeps the underlying object alive**. As long as a view (or a slice of a view) exists, the buffer it points into cannot be freed. That is the price of zero-copy and it is usually exactly what you want, but it means a tiny view onto a huge buffer pins the huge buffer in memory. When you are done and want only the small part, materialize it explicitly with `bytes(view)` and drop the view, so the big buffer can be collected.

## What a memoryview actually is

It helps to look at what the object exposes, because the attributes *are* the design. A `memoryview` carries the metadata of the buffer it wraps, and that metadata is precisely the buffer protocol's view of memory.

```python
import array

buf = array.array("d", [1.0, 2.0, 3.0, 4.0])  # 4 doubles, contiguous, typed
mv = memoryview(buf)

mv.nbytes      # 32  -> 4 elements * 8 bytes each
mv.itemsize    # 8   -> one double is 8 bytes
mv.format      # 'd' -> the struct format code for the element type
mv.shape       # (4,) -> one dimension of length 4
mv.ndim        # 1
mv.readonly    # False -> array is mutable, so the view can write
mv.contiguous  # True  -> the buffer is C-contiguous (streamable)

mv[0]          # 1.0  -> indexing returns a typed Python value, not a byte
mv[0] = 9.0    # writes THROUGH to buf, in place, no copy
buf[0]         # 9.0  -> the array changed; they share one buffer
```

Three things to notice. First, `nbytes` tells you the true memory footprint of the buffer the view describes — useful for budgeting. Second, `readonly` reflects whether the *source* allowed mutation: a view onto `bytes` is read-only (immutable source) and any write raises `TypeError`; a view onto `bytearray` or a writable `array`/ndarray can write through. Third, `format` and `itemsize` mean the view is **typed**: indexing a `'d'` view returns a Python `float`, indexing a `'b'` (signed byte) view returns an `int`, and so on. A `memoryview` is not just a window onto bytes; it is a window onto *typed elements*.

That typing is what `.cast()` reinterprets. The same bytes can be viewed as bytes, as 32-bit ints, or as doubles, without copying — you are just changing the lens.

```python
ba = bytearray(16)            # 16 raw bytes
mv = memoryview(ba)

as_i32 = mv.cast("i")         # view the same 16 bytes as 4 signed 32-bit ints
as_i32[0] = 258               # write the int; goes straight into the bytearray
bytes(ba[:4])                 # b'\x02\x01\x00\x00' on little-endian -> 258 = 0x102

as_f64 = mv.cast("d")         # OR view the same 16 bytes as 2 doubles
len(as_f64)                   # 2
```

`cast` requires the view to be contiguous and the new itemsize to divide the byte length evenly; it does not move or copy anything. This is the zero-copy reinterpret-cast, the Python equivalent of casting a `char*` to a `double*` in C, and it is how you read a packed binary buffer as typed numbers without unpacking element by element.

Why does this matter for performance, not just convenience? Because `cast` plus indexing lets you read a homogeneous binary column at C speed without the per-record `struct` call. If a file is a million `float64` values back to back, `memoryview(mm).cast("d")` gives you a length-one-million typed view, and `view[i]` returns the i-th double directly — no offset math, no format re-parsing, no slice. For pure bulk reads this is faster than `iter_unpack` because there is no per-element tuple to build. (For the absolute fastest path you would hand that same buffer to `np.frombuffer` and never touch a Python-level loop, but `cast` is the dependency-free stdlib version when you cannot or do not want to bring in NumPy.) The throughline is that a `memoryview` is not a dumb byte window — it is a *typed* window, and the type is what makes reading it cheap.

One more property of `nbytes` is worth a sentence because it surprises people: it reports the size of the *buffer the view describes*, which for a strided or multi-dimensional view is the logical element count times the itemsize, not necessarily the size of a contiguous run. Use it to budget how much memory a view is keeping alive (remember, a view pins its source), and pair it with `sys.getsizeof(mv)` — which reports only the small size of the `memoryview` *object itself*, a few dozen bytes — to see the gap: the view object is tiny, but it can be holding a gigabyte hostage through `nbytes`.

## Mutating bytes in place: bytearray + memoryview

`bytes` is immutable; its mutable sibling is `bytearray`. If you need to *change* bytes you already hold — fix up a header, mask a field, transform records in place — a `bytearray` plus a writable `memoryview` lets you do it with zero allocation. The naive approach rebuilds a new `bytes` for every edit:

```python
# ANTI-PATTERN: every edit allocates a whole new bytes object, O(n) each
data = bytes(b"...10 MB of data...")
data = data[:5] + b"X" + data[6:]   # allocates a fresh 10 MB bytes, copies twice
```

That `data[:5] + b"X" + data[6:]` allocates *two* slices plus the concatenation result — three copies of nearly the whole buffer to change one byte. Do it in a loop and you have quadratic behavior. With a `bytearray` and a view you write straight into the storage:

```python
data = bytearray(b"...10 MB of data...")
mv = memoryview(data)

mv[5] = ord("X")              # in place, no allocation, O(1)

# Bulk in-place transform: XOR a whole region with a mask, no copy
region = mv[1000:2000]        # zero-copy sub-view
for i in range(len(region)):
    region[i] ^= 0x5A         # writes THROUGH to data
```

For the bulk case you would usually push the inner loop into NumPy (wrap the same buffer, vectorize the XOR) rather than a Python loop — but the point stands that the *storage* is shared and never reallocated. A common real use is assembling a network packet or a binary record incrementally into a pre-sized `bytearray` and writing each field into its slot with `struct.pack_into`, which we will see next.

The other place `bytearray` shines is **streaming I/O into a fixed buffer**. `socket.recv_into(buf)` and `file.readinto(buf)` read bytes *directly into* a buffer you provide, instead of allocating a fresh `bytes` per read. Combined with a `memoryview` to track how far you have filled it, you can receive a large message with a constant number of allocations:

```python
def recv_exactly(sock, n):
    buf = bytearray(n)            # one allocation for the whole message
    view = memoryview(buf)        # a window we can advance without copying
    got = 0
    while got < n:
        k = sock.recv_into(view[got:])   # read straight into the gap, no temp bytes
        if k == 0:
            raise ConnectionError("socket closed early")
        got += k
    return buf
```

`view[got:]` is a zero-copy sub-view; `recv_into` writes the incoming bytes into it directly. No `b += sock.recv(...)` concatenation, no per-chunk allocation, no quadratic re-copying as the message grows. This pattern is the backbone of fast binary protocol code.

## Parsing binary records with memoryview + struct

Binary parsing is where zero-copy pays off most, because the alternative is a copy per field per record. Say each record is a fixed 24-byte layout: a 4-byte unsigned id, an 8-byte double timestamp, a 4-byte signed count, and an 8-byte double value, little-endian. The `struct` module describes that layout with a format string, and crucially `struct.unpack_from(fmt, buffer, offset)` reads **directly out of a buffer at an offset** without slicing it first.

```python
import struct

REC = struct.Struct("<I d i d")   # little-endian: uint32, float64, int32, float64
REC.size                          # 24 bytes per record

def parse_records(buf):
    """buf is anything that exports a buffer: bytes, bytearray, mmap, memoryview."""
    mv = memoryview(buf)
    n = len(mv) // REC.size
    total = 0.0
    unpack = REC.unpack_from        # hoist the bound method out of the loop
    for i in range(n):
        rid, ts, count, value = unpack(mv, i * REC.size)  # zero-copy read
        total += value * count
    return total
```

Two things make this fast. First, `unpack_from(mv, offset)` does **not** slice — it reads the 24 bytes at `offset` straight out of the underlying buffer and returns the parsed Python values. Compare it to the slow idiom `struct.unpack("<I d i d", buf[off:off+24])`, which first allocates a 24-byte `bytes` slice (`O(reclen)` copy) and *then* unpacks it. Over a million records, `unpack_from` saves a million small allocations and a million small copies. Second, `Struct("...")` is **compiled once**: building the `Struct` object parses the format string a single time, so each `unpack_from` call skips re-parsing the format. Using the module-level `struct.unpack(fmt, ...)` re-parses `fmt` on every call (it caches, but the bound-`Struct` path is still leaner and clearer).

#### Worked example: parsing 1 million records, slice-then-unpack vs unpack_from

Setup: 1,000,000 records × 24 bytes = 24 MB buffer, sum a derived value, M2, CPython 3.12, median wall-clock, GC disabled during timing.

```python
import struct, timeit

REC = struct.Struct("<I d i d")
buf = bytearray(REC.size * 1_000_000)   # 24 MB of zeros, for timing the parse cost

def slow(buf):
    out = 0.0
    n = len(buf) // 24
    for i in range(n):
        off = i * 24
        rid, ts, count, value = struct.unpack("<I d i d", buf[off:off+24])  # slice copy
        out += value * count
    return out

def fast(buf):
    mv = memoryview(buf)
    out = 0.0
    n = len(mv) // 24
    up = REC.unpack_from
    for i in range(n):
        rid, ts, count, value = up(mv, i * 24)     # zero-copy
        out += value * count
    return out

print(timeit.timeit(lambda: slow(buf), number=3) / 3)   # ~0.62 s
print(timeit.timeit(lambda: fast(buf), number=3) / 3)   # ~0.40 s
```

| Approach | Per-record cost | 1M records | Slice allocations |
| --- | --- | --- | --- |
| `unpack(fmt, buf[off:off+24])` | slice copy + reparse fmt | ~0.62 s | 1,000,000 |
| `Struct.unpack_from(mv, off)` | zero-copy read | ~0.40 s | 0 |

The `unpack_from` path is about **1.5× faster** here and allocates nothing per record, so it also keeps GC pressure flat. The remaining cost is the Python loop itself (boxing the returned values, the bytecode), which is exactly the kind of hot loop you would push into Numba or Cython next — and a Cython typed `memoryview` (`unsigned char[::1]`) parsing the same buffer with no Python-level per-record call gets you another order of magnitude. But notice the *first* win came purely from not copying. For a one-shot bulk read, the even faster move is to skip the per-record loop entirely and reinterpret the whole buffer as typed columns with NumPy, which we get to shortly.

For a stream of fixed-size records there is an even cleaner zero-copy idiom: `Struct.iter_unpack`, which iterates a buffer record by record without you computing offsets or slicing at all.

```python
total = 0.0
for rid, ts, count, value in REC.iter_unpack(memoryview(buf)):  # zero-copy, no offsets
    total += value * count
```

`iter_unpack` walks the buffer in C, yielding one tuple per record, and it accepts any buffer-exporting object — `bytes`, `bytearray`, `mmap`, or a `memoryview` — so you can point it straight at an `mmap` of a 50 GB record file and it will fault pages in as it goes, never copying and never holding more than one record's worth of Python objects at a time. It is the most readable way to express "parse a homogeneous binary stream with zero copies."

Variable-length records need one more step but the principle holds. When each record carries its own length (a common framing: a 4-byte length prefix, then that many bytes of payload), you read the prefix with `unpack_from`, then take a **`memoryview` sub-view** of the payload rather than slicing `bytes`:

```python
def iter_frames(buf):
    mv = memoryview(buf)
    off = 0
    n = len(mv)
    while off + 4 <= n:
        (length,) = struct.unpack_from("<I", mv, off)   # read the length prefix
        start = off + 4
        payload = mv[start:start + length]              # zero-copy view, not a bytes copy
        yield payload                                   # hand the consumer a window
        off = start + length
```

Each `payload` is a window onto the original buffer, so framing a 10 GB stream into a million variable-length messages allocates nothing per message — the consumer reads through the view. The only time you materialize bytes is when something downstream *demands* a `bytes` (for example to put it in a dict key or send it where a buffer will not do), and then you pay exactly one copy of exactly the bytes you needed, on purpose.

## mmap: process a file bigger than RAM

`f.read()` pulls the entire file into your process's heap. For a 10 GB file on a 16 GB box, that is reckless; for a 50 GB file it is impossible. `mmap` changes the deal completely: it **maps the file into your process's virtual address space** and lets the operating system bring pages in *on demand*. You get an object that behaves like a `bytearray` — you can index it, slice it, search it — but the bytes are not in RAM until you touch them, and the OS evicts pages it can reclaim. You can "open" a 50 GB file and start working in milliseconds, with a resident set of essentially zero, and it grows only to your working set.

```python
import mmap

with open("events.bin", "rb") as f:
    # Map the whole file, read-only. length=0 means "the entire file".
    mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

    mm[:16]                 # first 16 bytes -> pages in just the first page
    mm.size()               # the full file size, e.g. 53_687_091_200 (50 GB)
    pos = mm.find(b"\x00MAGIC")  # OS-level search; pages in only as it scans

    mv = memoryview(mm)     # a zero-copy view over the mapping
    record = mv[pos:pos+24] # an O(1) view; pages for that region fault in on access

    mm.close()
```

The magic is in how it interacts with the cache hierarchy and the **page cache**. Memory is managed by the OS in fixed **pages** (typically 4 KB). When you access an offset in the mapping whose page is not yet resident, the CPU raises a **page fault**; the OS catches it, reads exactly that 4 KB page from disk into the page cache, fixes up the mapping, and resumes your access. The first touch of a page costs a disk read (microseconds on NVMe, more on spinning disk); every subsequent touch of that page is a RAM-speed hit until memory pressure evicts it. You never explicitly read the file — you just access memory, and the OS does the I/O lazily under you.

![Before-and-after comparison of reading a 10 GB file fully into RAM, which allocates the whole buffer and risks the OOM killer, versus mmap, which maps the address range and demand-pages only the working set for a small resident set](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-4.png)

This is why `mmap` is the answer to "process a file too big for RAM." You can iterate over a 50 GB file, sum a field, find a needle, or randomly seek to record number 4 billion, and only the pages you actually touch ever become resident. Sequential scans page in nicely (the OS readahead prefetches the next pages, just like the cache prefetcher does for cache lines), and you can hint the access pattern with `mm.madvise(mmap.MADV_SEQUENTIAL)` so the kernel reads ahead aggressively and drops pages behind you, or `MADV_RANDOM` to disable readahead when you are seeking all over.

![Timeline showing an mmap access to an unmapped page triggering a page fault, the operating system reading a 4 KB page from disk, the page becoming cached, and the next access hitting at RAM speed](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-7.png)

#### Worked example: peak RSS, read-into-RAM vs mmap on a multi-GB file

Setup: a 10 GB binary file, task = sum one float64 field per 24-byte record across the whole file, measured peak RSS with `/usr/bin/time -v` (the `Maximum resident set size` line) on the 8-core Linux box, CPython 3.12, 16 GB RAM, NVMe.

```python
import mmap, struct, sys

REC = struct.Struct("<I d i d")

def sum_read(path):
    with open(path, "rb") as f:
        data = f.read()                 # pulls 10 GB into the heap
    mv = memoryview(data)
    total = 0.0
    up = REC.unpack_from
    for off in range(0, len(mv), 24):
        total += up(mv, off)[3]
    return total

def sum_mmap(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm.madvise(mmap.MADV_SEQUENTIAL)  # tell the OS we stream forward
    mv = memoryview(mm)
    total = 0.0
    up = REC.unpack_from
    for off in range(0, len(mv), 24):
        total += up(mv, off)[3]          # pages fault in as we scan, evicted behind
    mm.close()
    return total
```

| Strategy | Peak RSS | Fits on 16 GB box? | Notes |
| --- | --- | --- | --- |
| `f.read()` into heap | ~10.0 GB | barely; 50 GB file = no | whole file resident |
| `mmap` + sequential scan | ~150–300 MB | yes; 50 GB file = also yes | only the working set is resident |

The wall-clock for a single sequential scan is similar between the two (both are bound by reading 10 GB off the SSD once), but the **peak RSS** is the whole point: `mmap` keeps the resident set near the working-set size, not the file size. That is what lets the same code process a 50 GB file on a 16 GB machine. For random access — seek to scattered records — `mmap` wins on wall-clock *and* memory, because `f.read()` would have to load the entire file just to reach a few records, while `mmap` faults in only the handful of pages you touch.

It helps to see the access patterns side by side, because `mmap` is not a universal "faster file read" — it is the right tool for *specific* patterns, and the wrong tool for others. The table below lays out the four common shapes of file work and which strategy wins.

| Access pattern | `f.read()` into RAM | Buffered `readinto` chunks | `mmap` |
| --- | --- | --- | --- |
| Read whole small file once | best (simplest) | fine | overkill |
| Sequential scan of a huge file | OOM risk | good, constant RSS | good, constant RSS |
| Random seeks into a huge file | impossible (loads all) | clumsy (manual seek) | best (faults touched pages) |
| Edit a few bytes deep in a huge file | rewrites the whole file | clumsy | best (dirties one page) |
| Share the same file across processes | each copies it | each copies it | best (shared page cache) |

The pattern is clear: for a one-shot sequential scan, a buffered `readinto` into a fixed `bytearray` is just as memory-frugal as `mmap` and sometimes a hair faster, because it reads in large explicit chunks rather than 4 KB page faults. Where `mmap` is decisively better is **random access**, **in-place editing**, and **cross-process sharing of one file** — the cases where the alternative is to move the whole file when you only needed a sliver of it.

The reason `mmap` and buffered reads can tie on a sequential scan is the **page cache**. Both ultimately go through the OS's page cache: `read()` copies from the page cache into your buffer, while `mmap` maps the page-cache pages directly into your address space (saving even that one copy). The OS's **readahead** heuristic notices a forward scan and prefetches the next pages before you fault on them — the disk equivalent of the CPU's cache-line prefetcher — which is why a sequential `mmap` scan does not actually stall on a fault for every page; it stalls only when it outruns the readahead. `MADV_SEQUENTIAL` tells the kernel to read ahead aggressively and discard pages behind you (keeping RSS flat), and `MADV_RANDOM` tells it to stop reading ahead because you are seeking, so it does not waste bandwidth fetching pages you will not touch.

A few more `mmap` realities worth stating plainly. The mapping behaves like a writable buffer if you open it with `ACCESS_WRITE` (changes flush back to the file) or `ACCESS_COPY` (copy-on-write, private to your process). A `memoryview` over a writable `mmap` lets you edit a 50 GB file in place at a specific offset without rewriting the whole thing. The `find`/`rfind` methods do an OS-level memory search. And on a 32-bit process the address space caps the file size you can map — a non-issue on the 64-bit world you almost certainly run in, where the virtual address space is effectively unlimited relative to file sizes. The cost you watch for is **page-fault latency on random access to a cold file**: each fault to an unmapped page is a disk read, so a pathological random-access pattern over a file far larger than RAM is bound by your disk's random-read latency, not by Python. That is a storage problem, and the fix is a better access pattern or `MADV_WILLNEED` to prefetch — not loading the file into RAM, which you cannot do anyway.

## The buffer protocol: the contract underneath all of this

Everything so far — `memoryview`, `bytearray`, `array`, `mmap`, NumPy arrays — works together because they all speak one C-level contract: the **buffer protocol** (PEP 3118). An object that implements it can *export* a description of its raw memory — a pointer to the bytes, the total length, the element format, the shape, and the strides — to any consumer that asks. A consumer (like `memoryview`, or NumPy, or `struct`) reads that description and operates on the exported bytes **directly**, with no copy and no serialization. The producer says "here is where my data lives and what it looks like"; the consumer reads or writes it in place.

![Dataflow graph showing a producer such as a bytearray, array, or ndarray exporting its buffer through the buffer protocol into a single memoryview that a NumPy reader, a struct parser, and an in-place writer all share without copying](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-3.png)

This is why you can do all of the following with zero copies:

```python
import array, numpy as np

a = array.array("d", range(1000))   # producer: a typed C-contiguous buffer
mv = memoryview(a)                   # consumer 1: a typed window (buffer protocol)
arr = np.frombuffer(a, dtype=np.float64)  # consumer 2: an ndarray on the SAME bytes

arr[0] = 99.0                        # write through NumPy...
a[0]                                 # 99.0 -> ...the array sees it. One buffer.
mv[0]                                # 99.0 -> the memoryview too.
```

Three objects — an `array.array`, a `memoryview`, and a NumPy `ndarray` — all pointing at the **same** 8000 bytes. A write through any of them is visible through all of them, because there is exactly one copy of the data and three views onto it. That is the buffer protocol doing its job. The matrix below summarizes which tool removes which copy.

![Matrix mapping each zero-copy tool to the copy it avoids and when to use it, covering memoryview for the slice or parse copy, mmap for the read-into-RAM copy, bytearray for the rebuild-to-edit copy, and np.frombuffer for the serialize copy](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-5.png)

| Producer / source | Exports a buffer? | Writable through a view? | Typical zero-copy use |
| --- | --- | --- | --- |
| `bytes` | yes | no (immutable) | read-only parse window |
| `bytearray` | yes | yes | in-place mutate, `recv_into` |
| `array.array` | yes | yes | compact typed numerics |
| `mmap` | yes | yes (if `ACCESS_WRITE`) | huge-file lazy access |
| NumPy `ndarray` | yes | yes (if not read-only) | vectorized math on shared bytes |
| `str` | **no** | n/a | must `.encode()` first (a copy) |

Note the last row: Python `str` does *not* export the buffer protocol, because its internal representation is flexible (1, 2, or 4 bytes per character depending on content). To share text bytes you must `.encode()` to `bytes` first, which is a real copy. For binary data, though, the whole stack is zero-copy end to end.

## Contiguity, strides, and the views that quietly cost a copy

The buffer protocol's description carries more than a pointer and a length — it carries **shape** and **strides**, and those two fields are where the cache-locality story from the top of this post meets the zero-copy story. The strides tell a consumer how many bytes to step to reach the next element along each dimension. For a one-dimensional `array.array("d", ...)`, the stride is simply 8 bytes — each `float64` follows the previous one with no gaps. That is **C-contiguous**: the elements are packed back-to-back in memory order, which is precisely the packed-buffer layout the CPU streams at 1–2 ns per element. A `memoryview` over such a buffer reports `mv.contiguous == True` and `mv.c_contiguous == True`, and you can `.cast()` it freely.

But not every zero-copy view is contiguous, and that matters. Slice a buffer with a step — `mv[::2]` — and you get a *strided* view: every other element, stride 16 bytes instead of 8. It is still zero-copy (no bytes moved, just a stride of 16 recorded), but now the consumer that walks it skips half of every cache line. Reading `mv[0], mv[2], mv[4], ...` over a `float64` buffer touches one element from each pair, so each 64-byte line still serves 4 of your reads instead of 8 — you have halved the locality. Step by 8 (`mv[::8]`) and every read lands on a fresh cache line: you are back to the scatter penalty, paying close to a full miss per element even though the data is "in one buffer." A zero-copy strided view saved you the *copy* but can still wreck your *locality*, which is why the [NumPy strides post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) hammers on access order: a view is free, but walking it against the stride is not.

This is also why some operations *force* a copy even when you wanted a view. `mv.cast("d")` requires the view to be **contiguous**, because casting reinterprets a flat run of bytes as a different element type — a strided view has gaps the cast cannot describe. Hand a non-contiguous buffer to a C function that demands C-contiguous memory and the wrapper (NumPy's `np.ascontiguousarray`, for instance) silently makes a contiguous **copy** so the function can stream it. That copy is exactly the kind that hides in a profile: you thought you were passing a view, but the dtype or contiguity mismatch triggered a quiet `O(n)` repack. The discipline is to keep your buffers C-contiguous and your access order matched to the layout, so the zero-copy view *stays* zero-copy and *stays* fast.

`memoryview` even supports multiple dimensions when the producer exports a 2-D buffer (a NumPy 2-D array, say). You can index `mv[i, j]` and slice along either axis, all without copying, and `mv.shape` and `mv.strides` describe the layout. This is how Cython's typed memoryviews (`double[:, ::1]`) get C-speed 2-D access: the `::1` in the declaration *asserts* the last dimension is contiguous, so the compiler can generate a tight pointer walk instead of a strided one. The lesson generalizes beyond Cython: contiguity is not a detail, it is the property that lets a shared buffer also be a *fast* buffer.

#### Worked example: contiguous vs strided sum over the same buffer

Setup: a 100M-element `float64` buffer (800 MB), sum it contiguously versus summing every 8th element so each read lands on a fresh cache line, M2, CPython 3.12 with NumPy 1.26, median wall-clock. Both are zero-copy views; only the access pattern differs.

```python
import numpy as np, timeit

a = np.ones(100_000_000, dtype=np.float64)   # 800 MB, C-contiguous

contig = a                  # stride 8: 8 float64 per 64 B line
strided = a[::8]            # stride 64: one float64 per 64 B line (a zero-copy view)

t_contig  = timeit.timeit(lambda: contig.sum(),  number=5) / 5
t_strided = timeit.timeit(lambda: strided.sum() * 8, number=5) / 5  # ~same #ops

print(f"contiguous: {t_contig*1e3:.1f} ms")   # ~28 ms, ~1 line miss per 8 elems
print(f"strided[::8]: {t_strided*1e3:.1f} ms") # ~70+ ms, ~1 line miss per elem
```

| Access pattern | Stride | Elements per 64 B line used | Relative time |
| --- | --- | --- | --- |
| contiguous (`a`) | 8 B | 8 of 8 | 1.0× (baseline) |
| strided (`a[::8]`) | 64 B | 1 of 8 | ~2.5–3× slower |

Both views cost zero copies to *create*. The strided one is several times slower to *traverse* because it defeats line packing and prefetch — the exact $t_{\text{seq}}$ vs $t_{\text{scatter}}$ gap from the science section, observed on a buffer you never copied. Zero-copy is necessary but not sufficient; you also have to respect the line.

## Writable mmap, copy-on-write, and sharing buffers across processes

So far `mmap` was read-only, but its more powerful modes turn it into an in-place editor for files larger than RAM and a shared-memory channel between processes. The access mode you pass decides the semantics:

```python
import mmap

# ACCESS_WRITE: edits flush BACK to the file (shared with disk).
with open("events.bin", "r+b") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
    mm[100:108] = (42).to_bytes(8, "little")   # patch 8 bytes deep in a 50 GB file
    mm.flush()                                  # force the dirty pages to disk
    mm.close()
```

That patches eight bytes a hundred bytes into a 50 GB file **without rewriting the file** and without loading it into RAM — only the one 4 KB page that contains the offset is faulted in, marked dirty, and written back. Rewriting a single field in a giant binary file the naive way (read it, slice around the change, write it all out) would move 50 GB twice; the `mmap` edit moves 4 KB.

The `ACCESS_COPY` mode gives **copy-on-write**: your process sees the file's bytes, and reads are shared with the page cache, but the first *write* to a page makes a private copy of that page just for you, leaving the file and other processes untouched. This is how you can run an algorithm that mutates a huge mapped dataset speculatively — scribble on it in memory, with the OS lazily copying only the pages you actually change — and never touch the file on disk. You pay RAM only for the pages you dirty, not for the whole file.

`mmap`'s sibling for *deliberate* cross-process sharing is `multiprocessing.shared_memory.SharedMemory` (Python 3.8+). It allocates a named block of memory the OS exposes to multiple processes, and — because it exports the buffer protocol — you wrap it with a `memoryview` or `np.frombuffer` and share a real array between processes with **no pickling**:

```python
import numpy as np
from multiprocessing import shared_memory

# Producer process: create shared memory and fill an array view onto it
a = np.arange(10_000_000, dtype=np.float64)        # 80 MB
shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
view = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)  # ndarray over shared bytes
view[:] = a[:]                                     # one copy IN; then it's shared
name = shm.name                                    # pass this name to the worker

# Worker process: attach by name, get a zero-copy view of the SAME 80 MB
shm2 = shared_memory.SharedMemory(name=name)
arr = np.ndarray((10_000_000,), dtype=np.float64, buffer=shm2.buf)  # no pickle, no copy
arr.sum()                                          # reads the producer's bytes directly
shm2.close()
# ... when fully done, the creator calls shm.close() and shm.unlink()
```

This is the standard fix for the `multiprocessing` pickling tax on big arrays: instead of each `Pool.map` task pickling, copying, and unpickling an 80 MB array (the serialize round-trip you measured earlier, paid per task), you put the array in shared memory once and hand workers the *name*. They attach a zero-copy `np.frombuffer`/`np.ndarray` view and read the producer's bytes directly. The serialization cost collapses from "per task, proportional to array size" to "one setup copy, then free." It is the buffer protocol crossing the process boundary, and it composes with everything else in this post.

Two caveats keep this honest. First, `SharedMemory` blocks are **not** automatically reclaimed — the creator must `unlink()` the block when truly finished, or it leaks (on Linux it lingers in `/dev/shm`); the `resource_tracker` warns but you should manage lifetime explicitly. Second, there is no automatic synchronization — if a producer writes while a consumer reads the same bytes, you need a `Lock` or a clear hand-off protocol, exactly as with any shared mutable state. Zero-copy sharing removes the copy, not the need to reason about concurrency.

## Zero-copy interchange: NumPy and Arrow without serializing

The buffer protocol is what makes Python's numeric and data ecosystem composable *without* paying a serialization tax at every library boundary. When you hand data from your code to NumPy, or between NumPy and Apache Arrow, the naive path is to **serialize**: encode the array to a byte stream, copy that stream, and decode it on the other side. That is three passes over the data and a doubling of peak memory. The buffer-protocol path is to **share**: hand over the pointer, length, and dtype, and let the other library wrap the same bytes.

![Before-and-after comparison of moving an array between NumPy and Arrow by serializing, which encodes, copies, and decodes every byte, versus sharing the buffer, which hands over a pointer and a dtype description in constant time](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-6.png)

`np.frombuffer` is the workhorse. It wraps **any** buffer-exporting object as an ndarray **without copying** — the array is a view onto those bytes:

```python
import numpy as np

# A mmap'd file of float64 -> a NumPy array over it, zero copy:
import mmap
with open("vec.bin", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
arr = np.frombuffer(mm, dtype="<f8")   # view onto the mapping, no read into RAM
arr.mean()                             # streams pages in as it scans; tiny RSS

# A bytearray you parsed into -> typed columns, zero copy:
raw = bytearray(open("events.bin", "rb").read())   # (or mmap it)
ids = np.frombuffer(raw, dtype="<u4", count=1)      # reinterpret bytes, no copy
```

`np.frombuffer` gives a **read-only** array if the source is read-only (e.g. `bytes`), and a writable view if the source is writable (`bytearray`, writable `mmap`). The reverse direction — NumPy to a buffer — is automatic: an ndarray *is* a buffer exporter, so `memoryview(arr)`, `bytes(arr)` (this one copies, by design), and any C function expecting the buffer protocol all work. To go the other way without copying, `arr.tobytes()` copies but `memoryview(arr)` does not.

Apache Arrow takes this further: Arrow is a **columnar memory format** designed from the ground up for zero-copy sharing. An Arrow array and a NumPy array can share the same buffer in both directions for the common fixed-width, no-nulls case:

```python
import numpy as np, pyarrow as pa

a = np.arange(10_000_000, dtype=np.float64)   # 80 MB ndarray

# NumPy -> Arrow, zero copy (Arrow wraps NumPy's buffer):
arr = pa.array(a)                  # for a plain fixed-width array with no nulls, no copy

# Arrow -> NumPy, zero copy back:
back = arr.to_numpy(zero_copy_only=True)   # raises if a copy WOULD be needed
back.base is not None              # it's a view; same 80 MB, not a second copy
```

The `zero_copy_only=True` flag is the honest one: it **raises** rather than silently copying if the conversion cannot be zero-copy (for example, an Arrow array with nulls or a non-contiguous layout needs a copy to become a dense NumPy array). Using it documents and enforces the intent. This is the mechanism behind why Polars and DuckDB can pass data to and from NumPy and pandas so cheaply — they are all built on Arrow buffers, and the buffer protocol lets the bytes flow without re-encoding. The [Polars and Arrow post](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) goes deep on the dataframe side of this; here the point is the primitive underneath: **shared buffers, not serialized streams.**

#### Worked example: interchange a 1 GB array, serialize vs share

Setup: a 128M-element `float64` array (1 GB), hand it to another library, M2, CPython 3.12, measured with `timeit` and peak RSS noted.

```python
import numpy as np, pickle, timeit

a = np.arange(128_000_000, dtype=np.float64)   # 1 GB

# Serialize round-trip (e.g. across a process boundary or naive interchange):
def serialize_roundtrip(a):
    blob = pickle.dumps(a, protocol=5)   # encode -> a ~1 GB bytes object (a copy)
    return pickle.loads(blob)            # decode -> another 1 GB array (a copy)

# Share the buffer (same process, hand a view):
def share_buffer(a):
    return np.frombuffer(memoryview(a), dtype=a.dtype)  # O(1), no copy

print(timeit.timeit(lambda: serialize_roundtrip(a), number=3) / 3)  # ~0.9 s, peak ~3 GB
print(timeit.timeit(lambda: share_buffer(a), number=3) / 3)         # ~1.5 us, peak ~1 GB
```

| Path | Work done | Time | Peak RAM |
| --- | --- | --- | --- |
| pickle round-trip | encode + copy + decode 1 GB | ~0.9 s | ~3 GB (original + blob + copy) |
| `frombuffer` share | wrap the same bytes | ~1.5 µs | ~1 GB (one buffer) |

For in-process interchange the difference is absurd — microseconds versus most of a second, and one copy of the data versus three. The serialize path is only unavoidable when you genuinely cross a boundary the buffer cannot cross (a different machine, a pipe to a different process without shared memory). Even across processes, `multiprocessing.shared_memory` plus `np.frombuffer` lets you share the *same* physical buffer between processes and skip the pickle, which is the standard fix when pickling a big array dominates a `ProcessPoolExecutor` job. The discipline is the same everywhere: **don't serialize what you can share.**

## Bringing it back to the pipeline: the running example

Recall the series' running example — a data pipeline that loads a few million binary records, cleans and transforms them, and aggregates. The slow version did three copy-heavy things: `data = f.read()` (one giant copy into the heap), `chunk = data[off:off+reclen]` per record (a copy per record), and `np.array(list_of_parsed_values)` at the end (building Python lists then copying into an array). Each is a copy you can delete:

```python
import mmap, numpy as np

REC_DTYPE = np.dtype([("id", "<u4"), ("ts", "<f8"),
                      ("count", "<i4"), ("value", "<f8")])  # the 24-byte record

def aggregate(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)  # no read into RAM
    # Reinterpret the WHOLE file as a structured array, zero copy:
    recs = np.frombuffer(mm, dtype=REC_DTYPE)                   # a view onto the mapping
    # Vectorized aggregate over the shared buffer — one C loop, no Python per-record loop:
    weighted = recs["value"] * recs["count"]                   # streams pages in
    return weighted.sum()
```

This single rewrite collapses all three copies. `mmap` deletes the `f.read()` copy and keeps RSS tiny. `np.frombuffer` with a structured dtype deletes the per-record slice *and* the per-record Python loop — the file's bytes are reinterpreted in place as typed columns, and the aggregation is one vectorized C loop streaming the pages in. On the 10 GB file from earlier, the slow version OOM-killed; this version runs with a few hundred MB of RSS and is bound only by reading the file off the SSD once and the memory-bandwidth of the reduction. That is the leverage ladder applied to *memory*: do less work (don't copy), do it in bulk (vectorize over the shared buffer), and let the OS handle the I/O lazily. Avoiding the copy was the optimization; vectorizing the shared buffer was the multiplier.

## How to measure that you actually deleted the copies

Claiming "zero-copy" is easy; proving it is the part that keeps you honest. The two quantities to watch are **allocations** (did the copy actually disappear, or did it just move?) and **peak RSS** (is the resident set bounded by the working set, or by the file?). Wall-clock is a noisy third signal — a copy you deleted may not move the clock if you were not bandwidth-bound — so measure the cause, not only the symptom.

For allocations, `tracemalloc` is the precise tool. Snapshot before and after the operation and diff: a true zero-copy path shows no growth attributable to the buffer.

```python
import tracemalloc

tracemalloc.start()
snap1 = tracemalloc.take_snapshot()

mv = memoryview(big_bytes)
chunk = mv[500_000_000:]          # if this is truly zero-copy, RSS won't jump

snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, "lineno")[:5]:
    print(stat)                   # the memoryview slice should allocate ~tens of bytes
```

If you instead see a 500 MB allocation on the slice line, you accidentally materialized a copy somewhere (a stray `bytes(view)`, a non-contiguous `.cast()`, an `np.ascontiguousarray`). For the file-versus-`mmap` claim, the right instrument is the **peak RSS**, which you cannot see from inside the process cleanly — use the OS: `/usr/bin/time -v your_script.py` and read the `Maximum resident set size` line, or `memray run` for an allocation flame graph and high-water mark. The earlier 10 GB worked example was measured exactly this way, and the `~150–300 MB` versus `~10 GB` figures came from that line. That cross-cuts the whole [profiling track](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop): pick the metric that exposes the cause — bytes allocated and peak RSS for copy/memory work, not just the stopwatch.

A few measurement traps specific to zero-copy. **Warm versus cold cache for `mmap`**: the first run of a `mmap` scan pays disk reads to fault pages in; the second run hits the OS page cache and looks far faster, which can fool you into crediting your code. Measure both, and state which you are reporting (for an honest "process a cold file" number, drop the page cache between runs — on Linux, `echo 3 > /proc/sys/vm/drop_caches` as root — or just report the cold first run). **Constant-folding and dead-code elimination**: if a benchmark computes a sum it never uses, the optimizer (or your own restructuring) may skip the work; consume the result. **Too-small inputs**: a copy of 1 KB is free; the copy penalty only shows up when the slice is large relative to the cache, so size the input to the regime you actually run in. And **GC interference**: the allocation storm from the slow path triggers garbage collection, so disable GC during timing (`gc.disable()`) to separate the copy cost from the collection cost, then measure the collection cost separately if it matters. The point is the same one the series keeps making: a before/after number is only worth quoting if you can defend how you got it.

## When to reach for this (and when not to)

Zero-copy is a powerful default, but like every technique it has a cost — usually *complexity* and *lifetime management* — so be deliberate. The decision tree below routes from the copy you want to avoid to the tool that removes it.

![Decision tree that starts from avoiding a copy and branches on whether the bytes are already in memory for memoryview and struct, need in-place mutation for bytearray, are too big for RAM for mmap, or are being handed to NumPy or Arrow for frombuffer and the buffer protocol](/imgs/blogs/data-locality-and-zero-copy-memoryview-buffers-and-mmap-8.png)

**Reach for `memoryview` when** you slice or parse large binary blobs, especially in a loop — the `O(1)` slice and the `unpack_from`/`recv_into` patterns delete real allocations and copies. **Reach for `bytearray` + a writable view when** you must mutate bytes you already hold, or stream I/O into a fixed buffer. **Reach for `mmap` when** the file is large relative to RAM, when you do random access into a big file, or when several processes should share the page cache for the same file. **Reach for `np.frombuffer` / the buffer protocol when** you move data between your code and NumPy/Arrow and want to skip serialization.

**Do not reach for these when** the data is small. A `memoryview` over a 40-byte header saves nothing and is harder to read than a slice; just slice the `bytes`. The crossover is roughly when the copy starts to show up in a profile — under a few kilobytes per call in a non-hot path, the copy is free relative to everything else, and clarity wins. **Do not use `mmap` for tiny files** or files you will read fully and immediately — `f.read()` is simpler and the page-fault machinery is pure overhead when you touch every byte anyway. **Do not hold a `memoryview` onto a giant buffer when you only need a small piece** — it pins the whole buffer in memory; materialize the small piece with `bytes(view)` and drop the view so the big buffer can be freed. **Watch the lifetime trap**: you cannot resize a `bytearray` (or close an `mmap`) while a `memoryview` onto it is alive — Python raises `BufferError: Existing exports of data: object cannot be re-sized`. Release the view first (let it go out of scope, or call `mv.release()`). And **do not assume `mmap` is faster for sequential whole-file reads** — for a pure forward scan you read entirely once, a buffered `f.read()` or `readinto` can match or beat it; `mmap`'s win is *random access* and *memory footprint*, not raw sequential throughput.

The honest summary: the technique is about deleting *unnecessary* copies and unnecessary RAM, not about replacing every `bytes` with a `memoryview`. Profile first (a memory profiler like `memray` or `tracemalloc` will show the allocation storm; a CPU profile will show the `memcpy`), confirm copying or RSS is actually the bottleneck, then apply the matching tool. As always in this series: don't guess, measure.

## Case studies and real numbers

A few real-world data points, named and versioned where I can, to ground the claims.

**Apache Arrow's whole reason to exist.** Arrow was designed specifically so that analytical systems could share columnar data *without serialization*. The classic before/after is moving a dataframe between two libraries (say pandas/NumPy and a query engine): the pre-Arrow path serialized to some intermediate format and deserialized on the other side — for a multi-GB table that is multiple full passes over the data and a peak-memory spike. The Arrow path shares the buffers, so the handoff is metadata-only. This is why Polars (built on Arrow) and DuckDB can exchange data with pandas/NumPy at near-zero cost; the [dataframes post](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) measures this on a real job. The mechanism is exactly the buffer protocol and `frombuffer` shown above.

**`mmap` for indexes and search.** Search and database systems map their index files instead of reading them: the OS page cache then transparently caches the hot pages across processes and across runs, and a cold start touches only the pages a query needs. The practical effect is that a process can "open" a tens-of-GB index in milliseconds with near-zero RSS and let the working set materialize on demand — the same `mmap` lazy-paging behavior measured above, applied at production scale. The trade-off they manage is exactly the random-access page-fault latency we flagged: they tune readahead and sometimes `MADV_WILLNEED`-prefetch the pages a query will need.

**`recv_into` / `readinto` in high-throughput servers.** Asynchronous network frameworks and fast parsers (HTTP, protocol buffers, message brokers) read into pre-allocated buffers with `recv_into`/`readinto` and parse with `memoryview` + `unpack_from` precisely to avoid a per-read allocation. Under millions of messages, deleting the per-message `bytes` allocation flattens GC pressure and removes a measurable chunk of CPU — the same allocation-storm fix from the parsing worked example, at server scale.

**`shared_memory` for a `ProcessPoolExecutor` over a big array.** A common pattern: fan out a CPU-bound computation over an 80 MB array to a pool of worker processes. The naive `Pool.map` pickles the array argument to *each* worker — encode, copy through a pipe, decode — so an 8-worker job pays the ~0.9 s, ~3 GB-peak serialize round-trip measured earlier eight times over, and the IPC often dominates the actual compute. Putting the array in `shared_memory` once and passing workers only the block *name* turns that into a single setup copy plus zero-copy `np.frombuffer` attaches in each worker. The pickling line vanishes from the profile, peak RAM drops from "N copies" to "one buffer," and the job scales with cores instead of choking on serialization — the same zero-copy principle from this post, applied across the process boundary the [multiprocessing world](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) cares about most.

**The 20 GB log job from the intro.** The numbers from that real fix: peak RSS dropped from "over 16 GB and OOM-killed" to ~180 MB by switching `f.read()` → `mmap` and `data[off:off+reclen]` → `memoryview` slices, and wall-clock roughly halved because the per-record `bytes` allocations and copies disappeared from the profile. No parser rewrite, no native code — just deleting copies and letting the OS page the file in lazily. That single afternoon's change is the entire thesis of this post in one incident.

## Key takeaways

- The CPU reads memory in **64-byte cache lines**; one line holds 8 `float64` or 16 `int32`. Contiguous sequential access amortizes one slow fetch across all of them, while scattered pointer-chasing pays a miss per element — a 50–100× gap that has nothing to do with the interpreter.
- Slicing immutable `bytes`/`str` is `O(n)` — it allocates and copies. Slicing a **`memoryview`** is `O(1)` — it records an offset and length onto a shared buffer. In a hot loop over gigabytes, that is the difference between OOM and fine.
- A `memoryview` is a **typed** window: `.nbytes`, `.itemsize`, `.format`, `.readonly`, and `.cast()` let you reinterpret the same bytes as different element types with no copy. A view **pins** its source alive — release it before you resize or free the buffer.
- Use **`bytearray` + a writable view** to mutate bytes in place and to stream I/O into a fixed buffer with `recv_into`/`readinto`, replacing per-read allocations and quadratic concatenation.
- Parse binary records with `memoryview` + **`struct.Struct(...).unpack_from(mv, off)`** — zero-copy reads with a compiled-once format, instead of `unpack(fmt, buf[off:off+n])` which copies and re-parses every record.
- **`mmap`** maps a file into your address space and demand-pages it lazily, so you can process a 50 GB file on a 16 GB box with a resident set near the working-set size, not the file size. Its real wins are random access and small RSS, not raw sequential throughput.
- The **buffer protocol** is the shared C-level contract that lets `bytes`, `bytearray`, `array`, `mmap`, and NumPy all expose one buffer to many consumers. `str` does not export it — encode first.
- Move data between your code, NumPy, and Arrow by **sharing the buffer** (`np.frombuffer`, `to_numpy(zero_copy_only=True)`), not by serializing — microseconds and one copy instead of seconds and three.
- **Avoiding the copy is the optimization.** Profile to confirm copying or RSS is the bottleneck, then pick the tool that deletes that specific copy. Don't memoryview-ify small data where clarity wins.

## Further reading

- Python docs: [`memoryview`](https://docs.python.org/3/library/stdtypes.html#memoryview), the [buffer protocol / PEP 3118](https://docs.python.org/3/c-api/buffer.html), [`mmap`](https://docs.python.org/3/library/mmap.html), [`struct`](https://docs.python.org/3/library/struct.html), and [`array`](https://docs.python.org/3/library/array.html).
- NumPy docs: [`numpy.frombuffer`](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html) and the [array interface / buffer protocol notes](https://numpy.org/doc/stable/reference/arrays.interface.html).
- Apache Arrow: the [columnar format spec](https://arrow.apache.org/docs/format/Columnar.html) and the rationale for zero-copy interchange.
- *High Performance Python*, 2nd ed., Gorelick & Ozsvald — chapters on bytes, buffers, and memory.
- Within this series: the [series intro and the leverage ladder](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means); [NumPy memory layout, strides, views, copies, and the cache](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) (the same lesson for arrays); the [latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) (the cost hierarchy these tools exploit); and [dataframes at speed: pandas pitfalls, Polars, and Arrow](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) (zero-copy interchange in the dataframe world).
- For when one CPU box and its RAM truly are not enough: [the memory hierarchy from registers to shared memory to HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) takes the same cache-and-bandwidth reasoning down to the GPU.

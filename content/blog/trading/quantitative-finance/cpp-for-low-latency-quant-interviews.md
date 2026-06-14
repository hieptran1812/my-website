---
title: "C++ for low-latency: what HFT interviews actually test"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to the C++ that high-frequency trading interviews probe — the memory model, the cache, allocation, undefined behavior, and why the order hot path avoids the abstractions you were taught to love."
tags: ["low-latency", "cpp", "hft", "quant-interviews", "cache", "memory-model", "performance", "undefined-behavior", "market-making", "systems"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — HFT firms interview you in C++ because the few hundred nanoseconds between a market-data packet arriving and your order leaving runs in C++, and every abstraction you add to that path has a measurable cost.
>
> - The thing being tested is not "do you know C++ syntax" but "do you understand what the machine does" — the memory model, the cache, allocation, undefined behavior, and dispatch.
> - The **hot path** is the code from packet-in to order-out. On it you avoid heap allocation, virtual dispatch, exceptions, and anything with unpredictable timing — not because they're bad, but because they cost nanoseconds and, worse, *jitter*.
> - A **cache miss** costs ~100 ns; an **L1 hit** costs ~1 ns. That 100× gap is why a contiguous array beats a linked list and why struct-of-arrays beats array-of-structs — the whole game is keeping data the CPU needs in fast memory.
> - You **measure, you don't guess**: latency is a right-skewed distribution, and the number that matters is the tail (p99), not the average — because you lose the race on your slow days, not your median ones.
> - The one number to remember: a fill that arrives **10 microseconds late** can forfeit on the order of **\$15,000 of edge per day** on a 30-million-share book. That is why the interview is this hard.

Here is a question a Jane Street, Hudson River Trading, Jump, Citadel Securities, Optiver, or IMC interviewer can ask in the first ten minutes: *"Walk me through what happens, in your code, between the moment a price update hits your network card and the moment your buy order leaves it. How long does that take, and where does the time go?"*

If you can answer that — really answer it, with nanoseconds and cache lines and a reason for every abstraction you did or didn't use — you have shown them the one thing they are hiring for. The C++ they test is not the C++ of a tutorial. It is the C++ of a path so short and so hot that a single unnecessary memory access, measured in nanoseconds, is the difference between a trade you win and a trade you lose to a faster competitor.

This post builds that understanding from the ground up. We assume you can read a `for` loop and have seen a pointer, but we define every term that matters — *cache line*, *RAII*, *false sharing*, *vtable*, *undefined behavior* — the first time it appears. We anchor each idea in a worked example with real numbers, and we close with a set of fully-solved interview problems of exactly the kind these desks ask.

![The tick-to-trade hot path: an HFT interview is really an interview about the few hundred nanoseconds between a market-data packet and your order](/imgs/blogs/cpp-for-low-latency-quant-interviews-1.png)

The diagram above is the mental model for the whole post. A packet of market data arrives at your network card. Your code parses it, updates its view of the order book, asks a strategy whether to trade, encodes an order, and hands it back to the card to send. That round trip — call it **tick-to-trade** — is a few hundred nanoseconds at a competitive firm. Everything in this article is about making that path short and, just as importantly, *predictable*.

## Foundations: why C++, and what a nanosecond buys

Let us define the units before anything else, because the whole field is unintuitive until you internalize them.

A **nanosecond** (ns) is one billionth of a second. A **microsecond** (µs) is one thousandth of a millisecond, or one thousand nanoseconds. A **millisecond** (ms) is one thousandth of a second. To make these concrete: light travels about 30 centimeters — roughly one foot — in a single nanosecond. Modern CPUs run at around 3 to 4 billion cycles per second, so one clock **cycle** is roughly a third of a nanosecond. When we say an operation "costs 100 ns," we mean it burns about 300 to 400 cycles the CPU could have spent doing something useful.

A **basis point** is one hundredth of a percent — 0.01%. You will see it later when we put dollars on latency. **Edge** is the trading term for the expected profit, per unit traded, of a strategy — if you expect to make half a cent per share on average, your edge is \$0.005 per share.

### Compiled versus interpreted: where the time actually goes

Programs reach the CPU in one of two broad ways. An **interpreted** or **managed** language — Python, or to a lesser degree Java and C# — does not hand the CPU your code directly. Python compiles your source to an intermediate form called *bytecode*, then a program called the *interpreter* (or for Java, a *virtual machine*, the VM) reads that bytecode one instruction at a time at runtime and decides what the CPU should do for each. That decision — "what does this bytecode op mean, and which machine instructions implement it" — is re-paid on every single operation. It is a translation tax, and on a tight loop it makes the language 10 to 100 times slower per operation than native code.

A **compiled** language like C++ does the translation once, ahead of time. A program called the *compiler* turns your source into actual machine code — the raw bytes the CPU executes natively — before the program ever runs. At runtime there is no interpreter in the loop, no VM dispatching ops. The CPU runs your bytes. You also get something subtler and, for trading, more important: you own every allocation and there is no *garbage collector* — no background process that periodically walks your memory to reclaim what you stopped using, and that can pause your program at an unpredictable moment for an unpredictable duration. A garbage-collection pause of even a few hundred microseconds, landing in the middle of your hot path, is a catastrophe at a firm where the whole budget is hundreds of nanoseconds.

![Why C++: compiled native code runs directly on the CPU while an interpreter re-pays a translation tax on every operation](/imgs/blogs/cpp-for-low-latency-quant-interviews-3.png)

That is the one-sentence answer to "why C++ for HFT": it compiles to native code, you control memory and timing precisely, and there is no VM or garbage collector to inject unpredictable pauses. C and Rust share these properties, and both appear at HFT shops, but C++ has the deepest ecosystem of low-latency libraries and the longest institutional history on these desks, which is why the interview is in C++.

### The latency budget: where nanoseconds matter

Not all of your code is latency-critical. The vast majority of any trading system — risk reporting, configuration, logging, the user interface, the overnight backtest — has no business being micro-optimized, and optimizing it is a waste of effort. The latency budget lives in exactly one place: the **hot path**, the code that runs between a market event and your reaction to it. Understanding which operations are cheap and which are ruinously expensive is the foundation everything else rests on.

![The latency ladder: memory and network operations span six orders of magnitude, and the hot path lives in the fast band where C++ wins](/imgs/blogs/cpp-for-low-latency-quant-interviews-2.png)

Read that ladder carefully, because the entire discipline follows from it. An **L1 cache** hit — fetching data from the small, fast memory physically closest to the CPU core — costs about 1 ns. An **L2** hit is a few nanoseconds; **L3** (shared across cores) is around 12 ns. A read from **main memory** (RAM) that missed all the caches costs about 100 ns. A small **heap allocation** costs 80 to 200 ns. Reading from a fast **NVMe SSD** is tens of microseconds. A network round trip across data centers is hundreds of microseconds — hundreds of thousands of nanoseconds.

The jump from "in cache" (1–12 ns) to "in RAM" (100 ns) is roughly 10× to 100×. That single cliff is responsible for most of what this post teaches. When an interviewer asks why your linked-list traversal is slow, or why your struct is laid out the way it is, the answer almost always reduces to "how many times did we fall off that cliff."

#### Worked example: counting the budget

Suppose your tick-to-trade target is 500 ns, and you measure the path:

- Network card delivers the packet to memory: ~250 ns (much of this is hardware you can offload, but assume it is in your budget).
- Parse the packet and update the order book: ~80 ns.
- Strategy logic decides buy / sell / pass: ~120 ns.
- Encode the outgoing order: ~60 ns.

That sums to 510 ns — already over budget. Now imagine your strategy logic, written naively, does one heap allocation (a `std::string` for a symbol, say) inside that 120 ns block. That allocation might cost 150 ns on a bad day and trigger a cache miss when the allocator walks its free list, costing another 100 ns. Your 120 ns strategy block balloons toward 370 ns, and your 510 ns path becomes 760 ns. You just lost the race to a competitor whose strategy block never touched the heap.

**The intuition:** the budget is so tight that a single avoidable memory access — one allocation, one cache miss — can blow it. That is why the hot path treats every allocation and every pointer-chase as a decision to justify, not a default.

## The memory model: stack, heap, pointers, and RAII

To reason about cost you need a clear picture of *where your data lives*. C++ gives you two main regions: the stack and the heap.

### The stack

The **stack** is a contiguous block of memory, one per thread, typically a few megabytes. It works exactly like a stack of plates: when a function is called, a *frame* — space for its local variables — is pushed on top; when the function returns, the frame is popped off. The CPU tracks the top with a single register, the *stack pointer*. Allocating a local variable is therefore almost free: the CPU just moves the stack pointer down by the size you need, which is a single instruction costing about 1 ns. Freeing it is equally free — the pointer moves back when the function returns. Because the stack is contiguous and reused constantly, the top of it is almost always already in L1 cache, so reads and writes there are fast.

```cpp
void on_tick(double price) {
    double mid = price + 0.5;   // a local on the stack: ~1 ns to "allocate"
    int    qty = 100;           // also on the stack, right next to mid
    // ... mid and qty are freed automatically when on_tick returns
}
```

### The heap

The **heap** is a large, shared pool of memory you ask for explicitly with `new` (or `malloc`). When you write `new Order()`, the C++ runtime calls an *allocator* — a piece of library code whose job is to find a free chunk of the right size somewhere in that pool, mark it used, and hand you back a *pointer* to it. A **pointer** is just a variable holding a memory address: the location of your object, not the object itself. When you are done you must return the memory with `delete` (or `free`), or you have a *memory leak*.

![Stack vs heap: stack allocation bumps a register in one nanosecond while heap allocation searches a free list for 80 to 200 ns](/imgs/blogs/cpp-for-low-latency-quant-interviews-4.png)

The diagram makes the cost difference visible. The stack is orderly: frames stack up in one contiguous, cache-resident region, and allocation is a register bump. The heap is a managed sprawl: buffers, objects, and free blocks scattered across a large region, with metadata and *fragmentation* (gaps left by freed objects that are too small to reuse) in between. Finding a free block means the allocator walks a *free list* — a data structure of available chunks — which is itself often a cache miss, may need to take a lock if multiple threads are allocating, and may trigger a page fault if the operating system has to map fresh memory. That is why a heap allocation is 80 to 200 ns and *jittery*, while a stack allocation is ~1 ns and constant.

### References versus pointers

A **reference** in C++ is an alias for an existing object — another name for the same memory, written `int& r = x;`. It cannot be null and cannot be re-pointed; it is the safe, lightweight way to refer to something without copying it. A **pointer** (`int* p = &x;`) can be null, can be moved to point elsewhere, and can be arithmetic'd. On the hot path you pass large objects by reference (or by pointer) rather than by value precisely to avoid copying them — passing a 200-byte order-book snapshot by value copies all 200 bytes; passing it by reference copies 8 bytes (the address).

### RAII: the idea that makes C++ safe at speed

**RAII** stands for *Resource Acquisition Is Initialization*, and despite the clumsy name it is the single most important idiom in C++. The idea: tie the lifetime of a resource — memory, a file, a lock, a network connection — to the lifetime of an object on the stack. You acquire the resource in the object's *constructor* (the code that runs when it is created) and release it in the *destructor* (the code that runs automatically when it goes out of scope). Because stack objects are destroyed deterministically the instant they leave scope, the resource is always released, exactly once, with no garbage collector and no manual `delete` to forget.

```cpp
{
    std::lock_guard<std::mutex> guard(book_mutex);  // constructor locks
    update_order_book();                            // do work under the lock
}  // guard's destructor runs here, automatically — the lock is released
```

RAII is why C++ can be both fast and safe: you get deterministic, zero-overhead cleanup without a garbage collector's pauses. `std::unique_ptr` and `std::shared_ptr` — the *smart pointers* — are RAII wrappers around heap memory: when the smart pointer is destroyed, it deletes what it owns. Interviewers love RAII questions because it separates people who memorized syntax from people who understand object lifetime.

## The CPU cache and data locality

We have met the cache twice now. It is time to understand it properly, because cache behavior is the dominant factor in real-world C++ performance, and it is the topic HFT interviews probe most relentlessly.

### Cache lines

The CPU never reads a single byte from main memory. It reads in fixed-size chunks called **cache lines**, almost always 64 bytes. When your code touches one byte, the hardware pulls the entire 64-byte line containing it into cache. This is the central fact. It means that if the *next* thing you need is in the same 64-byte line, it is already there — an L1 hit, ~1 ns. If it is somewhere far away in memory, you pay another miss — ~100 ns.

There is also a **prefetcher**: hardware that watches your access pattern and, if it detects a regular *stride* (you are walking through memory at a constant step, like iterating an array), speculatively pulls the next lines in *before* you ask for them, hiding the miss latency entirely. The prefetcher is your best friend, and it only works when your access pattern is predictable and sequential.

### Why contiguous arrays beat linked lists

Now the most famous result in practical C++ performance, and a near-guaranteed interview topic.

![Cache lines: a contiguous array packs many elements per line while a linked list pays a cache miss per node](/imgs/blogs/cpp-for-low-latency-quant-interviews-5.png)

A `std::vector<int>` stores its elements **contiguously** — one after another in a single block of memory. A 64-byte cache line holds 16 four-byte ints. So when you iterate the vector, the very first access misses and loads a line, but the next 15 accesses are already in cache — L1 hits at ~1 ns each. And because you are striding predictably, the prefetcher streams the following lines in ahead of you. Iterating a vector is about as cache-friendly as code gets.

A `std::list<int>` (a doubly-linked list) stores each element in a **separately allocated node** somewhere on the heap, with two pointers (to the previous and next nodes) alongside the value. The nodes can be anywhere. Walking the list means: read this node, follow its `next` pointer to a far-off address, miss the cache (~100 ns), read that node, follow *its* pointer, miss again. There is no stride for the prefetcher to follow because the addresses are effectively random. Every hop is a likely miss.

#### Worked example: array versus linked list, with cycle estimates

Sum 1,000,000 ints. Assume a 3 GHz CPU (one cycle ≈ 0.33 ns), 64-byte lines, an L1 hit at ~1 ns, and a miss at ~100 ns.

- **`std::vector<int>`:** 1,000,000 ints × 4 bytes = 4,000,000 bytes = 62,500 cache lines. The prefetcher hides most miss latency on a clean sequential stride, but even pessimistically counting one ~100 ns miss per line gives 62,500 × 100 ns ≈ 6.25 ms; with prefetching working, real runs land closer to ~1 ms.
- **`std::list<int>`:** 1,000,000 nodes, each its own allocation, each hop a likely miss. 1,000,000 × 100 ns ≈ 100 ms.

That is a roughly 15× to 100× gap on identical logic — summing a million numbers — caused by nothing but memory layout. The list also wastes memory: each node carries two 8-byte pointers, so a list of 4-byte ints is 5× larger than the vector.

**The intuition:** "linked list" sounds elegant in an algorithms course, but on real hardware the constant cache misses make it far slower than a contiguous array for almost any traversal-heavy workload. On the hot path, prefer contiguous storage — `std::vector`, fixed arrays, ring buffers — and treat `std::list` as a near-automatic red flag.

### Array-of-structs versus struct-of-arrays

The cache-line idea has a sharper, more advanced form that comes up at the quant-heavy desks. Suppose each item in your book is a `Quote` with a price, a size, and an id, and you frequently need to do something with *just the prices* — sum them, find the max, run a calculation.

![Array-of-structs vs struct-of-arrays: SoA loads three times fewer cache lines than AoS when you touch a single field](/imgs/blogs/cpp-for-low-latency-quant-interviews-9.png)

The natural layout is **array-of-structs** (AoS): `Quote q[N]`, where each `Quote` holds `{px, sz, id}` side by side. But when you sum only the prices, each cache line you load contains prices *and* the sizes and ids you don't need — so most of every line you fetch is wasted, and you touch more lines than necessary. The alternative is **struct-of-arrays** (SoA): instead of one array of structs, keep three parallel arrays, `double px[N]`, `double sz[N]`, `int id[N]`. Now the prices are packed contiguously with nothing else between them. Summing them touches only the `px` array; every byte you fetch is a price you use.

#### Worked example: AoS versus SoA cache misses

Sum the price field over N = 1,000,000 quotes. Take `Quote = {double px (8 B), double sz (8 B), int id (4 B)}` = 20 bytes, padded to 24 for alignment.

- **AoS:** the prices are 24 bytes apart in memory. A 64-byte line spans about 2.67 quotes, so you touch a line roughly every 2.67 prices: ~1,000,000 / 2.67 ≈ 375,000 line loads.
- **SoA:** the prices are packed 8 bytes apart. A 64-byte line holds 8 prices, so: 1,000,000 / 8 = 125,000 line loads.

That is 3× fewer cache-line loads. At ~100 ns per miss (worst case, no prefetch) that is ~37 ms versus ~12 ms — a 3× speedup from layout alone, with the loop body and the data unchanged. In practice prefetching narrows both, but the *ratio* persists, and on a hot path that runs millions of times a second, 3× is enormous.

**The intuition:** lay your data out to match how you access it. If you scan one field across many records, store that field contiguously. SoA is the workhorse layout of high-performance numerical and trading code for exactly this reason.

### False sharing

There is one more cache phenomenon that is pure interview gold because it is counterintuitive and bites real systems. **False sharing** happens when two CPU cores write to two *different* variables that happen to sit on the *same* 64-byte cache line.

![False sharing: two cores writing different variables on one cache line invalidate each other on every single write](/imgs/blogs/cpp-for-low-latency-quant-interviews-6.png)

Here is the mechanism. To keep memory coherent, the hardware enforces that only one core can *own* a cache line for writing at a time. When Core 0 writes `counterA`, it must take exclusive ownership of the line — which *invalidates* the copy Core 1 holds, even though Core 1 only cares about `counterB`. The next time Core 1 writes `counterB`, it must pull the line back and invalidate Core 0. The two cores ping-pong the line back and forth, paying a coherence cost of roughly 100 ns on every write, even though they never touch the *same* variable. The variables are independent; the cache line is shared. Hence "false" sharing.

The fix is to *pad* the variables so each sits on its own line. In modern C++ you write `alignas(64)` to force a variable to start at a 64-byte boundary:

```cpp
struct Counters {
    alignas(64) std::atomic<long> a;  // a gets its own cache line
    alignas(64) std::atomic<long> b;  // b gets its own cache line
};
// Now Core 0 writing a never invalidates Core 1's line for b.
```

**The intuition:** when you put per-thread data side by side in a struct shared across cores, pad it to cache-line boundaries, or the cores will silently throttle each other. This one shows up because it is invisible in the code — both versions look correct — and only a person who understands the cache coherence protocol will spot it.

## Allocation and why you avoid it on the hot path

We have established that a heap allocation costs 80 to 200 ns and, worse, is unpredictable. On a path with a budget measured in hundreds of nanoseconds, that is unaffordable. So a core skill the interview tests is recognizing hidden allocations and removing them.

![The cost of allocation: heap allocation costs tens to hundreds of nanoseconds while a pool returns a slot in single-digit nanoseconds](/imgs/blogs/cpp-for-low-latency-quant-interviews-7.png)

The diagram shows why `malloc` is both slow and jittery: it may search a free list (a cache miss), it may take a lock if other threads are allocating concurrently, and it may page-fault if the OS must back fresh virtual memory with physical pages — and any of those can land or not land on a given call, so the *distribution* of allocation times has a long tail. The hot path hates tails.

### The techniques: pre-allocation and object pools

Two patterns dominate. **Pre-allocation** means you allocate everything you will need *at startup*, before the market opens, while latency does not matter, and then never allocate on the hot path again. You size your buffers for the worst case you expect and reuse them.

An **object pool** is the reusable version of that idea. At startup you allocate a fixed-size pool of, say, 100,000 `Order` objects. When the hot path needs an order, it calls `pool.acquire()`, which pops a free slot off a freelist you sized in advance — no syscall, no lock contention, no page fault, just a pointer hand-off costing a few nanoseconds. When the order is done, `pool.release()` returns the slot. The memory was allocated once, long ago; the hot path only ever recycles it.

```cpp
template <typename T, size_t N>
class ObjectPool {
    std::array<T, N> storage_;        // all memory allocated up front
    std::vector<T*>  free_list_;      // pointers to unused slots
public:
    ObjectPool() {
        free_list_.reserve(N);
        for (auto& obj : storage_) free_list_.push_back(&obj);
    }
    T* acquire() {                    // ~5-10 ns, no heap touch
        if (free_list_.empty()) return nullptr;  // pool exhausted
        T* p = free_list_.back();
        free_list_.pop_back();
        return p;
    }
    void release(T* p) { free_list_.push_back(p); }
};
```

#### Worked example: remove the allocation from a tick handler

Here is a tick handler with a hidden allocation, of the kind an interviewer hands you and says "make this faster":

```cpp
// BEFORE: allocates on every tick
void on_tick(const Tick& t) {
    std::vector<double> recent_prices;        // heap allocation #1
    recent_prices.push_back(t.price);         // maybe a reallocation
    std::string sym = std::string(t.symbol);  // heap allocation #2
    auto signal = std::make_shared<Signal>(); // heap allocation #3
    // ... use them, then they're all freed at end of scope (3 frees too)
}
```

That handler does three allocations and three frees per tick. At ~120 ns each (alloc + free), that is ~720 ns of pure allocation overhead on every tick — multiple times your entire budget. Here is the rewrite:

```cpp
// AFTER: zero allocation on the hot path
struct TickHandler {
    std::array<double, 64> recent_prices_;    // fixed buffer, on the object
    size_t                 count_ = 0;
    ObjectPool<Signal, 4096> signal_pool_;    // pre-allocated at startup

    void on_tick(const Tick& t) {
        if (count_ < recent_prices_.size())
            recent_prices_[count_++] = t.price;   // no allocation, ~1 ns
        std::string_view sym{t.symbol};            // a view, copies nothing
        Signal* signal = signal_pool_.acquire();   // ~5-10 ns, no heap
        // ... use them; release the signal when done
        signal_pool_.release(signal);
    }
};
```

The `std::vector` became a fixed-size `std::array` living inside the handler object (no per-tick allocation). The `std::string` became a `std::string_view` — a non-owning *view* (a pointer plus a length) that refers to the existing bytes without copying them. The `make_shared` became a pool acquire. The handler now does zero heap operations per tick. We went from ~720 ns of allocation overhead to roughly 10 ns.

**The intuition:** on the hot path, allocate once at startup and reuse. Every `new`, every `make_shared`, every `std::string` and growing `std::vector` is a hidden allocation to hunt down and eliminate. This same discipline shows up in [how alpha signals are computed in production](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) — the research code can allocate freely, but the live path cannot.

## Move semantics versus copies

When you must transfer data between objects on the hot path, you care intensely about whether the transfer *copies* the underlying memory or merely *moves* ownership of it.

![Move vs copy: a copy duplicates the heap buffer while a move transfers ownership of the same buffer](/imgs/blogs/cpp-for-low-latency-quant-interviews-8.png)

Consider a `std::vector` or `std::string` — both hold a small *control block* (a pointer to a heap buffer, a size, and a capacity, roughly 24 bytes) plus the heap buffer itself, which can be large. A **copy** of such an object is *deep*: it allocates a brand-new heap buffer and `memcpy`s all N bytes across, so you end up with two independent copies of the data. The cost grows with N, and it involves an allocation. A **move** does something cleverer: it copies just the control block — the three small fields, 24 bytes — and then nulls out the source's pointer so the source no longer owns the buffer. No new allocation, no `memcpy` of the payload, constant cost regardless of N. The source is left in a valid but empty state.

C++ expresses this with *move constructors* and `std::move`. Writing `auto b = std::move(a);` says "transfer a's resources into b; I promise not to use a's value afterward." Returning a large object from a function is automatically a move in modern C++ (via *return value optimization* and move semantics), which is why returning a `std::vector` by value is cheap, not the disaster a beginner fears.

#### Worked example: copy versus move on a 1 MB buffer

You hold a `std::vector<char>` of 1,000,000 bytes and need to hand it to another object.

- **Copy:** allocate a new 1 MB buffer (~150 ns for the allocation) and `memcpy` 1,000,000 bytes. At, say, 10 GB/s memory bandwidth, the copy alone is ~100 µs. Total: ~100 microseconds, dominated by the byte-by-byte copy.
- **Move:** copy 3 pointers (24 bytes), null one out. ~10 ns. No allocation, no payload copy.

The move is roughly *ten thousand times* faster on a 1 MB buffer, and the gap grows with size. On the hot path, you move; you copy only when you genuinely need two independent copies.

```cpp
void enqueue(std::vector<char> buf) {        // takes buf by value
    queue_.push_back(std::move(buf));        // MOVE into the queue, ~10 ns
}                                            // not a 1 MB copy
// caller:  enqueue(std::move(my_buffer));   // hands over ownership
```

**The intuition:** a copy duplicates the payload; a move transfers ownership of it. When you don't need a second copy, move — it turns an O(N) operation into an O(1) one and removes an allocation. Interviewers test this because misusing it (copying where you meant to move) is one of the most common silent performance bugs.

## Const-correctness and undefined behavior

Two topics that look like "language lawyering" but are really about whether you can be trusted to write code that is correct *and* fast.

### Const-correctness

Marking something `const` declares that it will not be modified. `const` is not primarily about safety theater — it is information for the compiler. When the compiler knows a value cannot change, it can keep it in a register, avoid re-reading it from memory, reorder around it, and skip defensive reloads. A `const` reference (`const Order& o`) also documents intent: "I am reading this, not changing it," which makes code reviewable and lets you pass large objects cheaply by reference without anyone fearing you'll mutate them. On the hot path, const-correctness is a quiet performance lever as well as a correctness one.

### Undefined behavior: the dragon

**Undefined behavior** (UB) is the most important concept in this whole section and a near-certain interview topic. The C++ standard defines what your program must do for *valid* programs. For certain illegal operations — reading past the end of an array, using memory after you freed it, signed-integer overflow, dereferencing a null or dangling pointer, a data race between threads — the standard says nothing at all about what happens. The behavior is *undefined*. The compiler is allowed to assume UB *never occurs*, and it optimizes on that assumption.

This is subtle and dangerous. UB does not mean "it crashes." It means the compiler may do *anything*: produce a value that looks right today and wrong tomorrow, delete a safety check because "this pointer can't be null, since dereferencing null would be UB and I'm allowed to assume UB never happens," or silently corrupt unrelated data. UB bugs are the hardest bugs in C++ because the symptom can appear far from the cause, and "it worked when I tested it" proves nothing.

```cpp
int sum_first_n(const std::vector<int>& v, int n) {
    int total = 0;
    for (int i = 0; i <= n; ++i)   // BUG: <= reads v[n], one past if n == v.size()-... 
        total += v[i];             // out-of-bounds read = undefined behavior
    return total;
}
```

If `n` reaches `v.size()`, `v[i]` reads past the end of the array. That is UB. It might return a garbage number, it might read into adjacent memory, it might crash, it might appear to work for months and then corrupt your order state during a busy open. The fix is `i < n` with a bounds check that `n <= v.size()`.

#### Worked example: spot the undefined behavior

This is a classic interview snippet. The interviewer slides it across and asks, "what's wrong?"

```cpp
const char* make_symbol_label(int id) {
    char buf[16];                          // a local array, lives on the stack
    std::snprintf(buf, sizeof buf, "SYM%d", id);
    return buf;                            // BUG: returns a pointer to a local
}
// caller:
const char* label = make_symbol_label(42);
std::cout << label;                        // reads freed stack memory = UB
```

`buf` is a local array living in `make_symbol_label`'s stack frame. The function returns a *pointer to it* — but the instant the function returns, that frame is popped and the memory is reused by the next function call. The returned pointer is now *dangling*: it points to memory that no longer holds your string. Reading through it is undefined behavior. The treacherous part is that it often *appears* to work in a quick test, because nothing has overwritten that stack slot yet — and then it fails in production when a different call path reuses the frame first.

How to fix it depends on intent: return a `std::string` (which owns its buffer and is moved out cheaply), or have the caller pass in a buffer to fill, or use a fixed-size struct returned by value. The point the interviewer is testing is whether you *see* the dangling pointer and can explain *why* it is UB rather than just "a bug."

**The intuition:** never return a pointer or reference to a local; never read past an array's end or use memory after freeing it. UB is not a crash you can catch — it is the compiler optimizing on a false premise, and the failure can surface anywhere, anytime. On a trading system, a UB bug is a money bug.

## Templates and inlining versus virtual dispatch

The last big mechanism the interview probes: how a function call actually reaches the right code, and why the elegant object-oriented answer is often the wrong one on the hot path.

### Virtual dispatch and the vtable

C++'s mechanism for *runtime polymorphism* — calling the right `on_tick` for whatever strategy object you happen to hold, decided at runtime — is the **virtual function**. It works through a **vtable** (virtual function table): a hidden table of function pointers, one per virtual method, that the compiler builds for each class. Every object of a polymorphic class secretly carries a pointer (the *vptr*) to its class's vtable.

When you call a virtual function through a base-class pointer, the CPU must: load the vptr from the object, load the right function pointer out of the vtable, and then perform an **indirect jump** — a jump to an address it computed at runtime rather than one baked into the instruction. Indirect jumps are hard for the CPU's *branch predictor* (the hardware that guesses where execution is headed so it can run ahead) to predict, and a misprediction stalls the pipeline for roughly 15 to 20 cycles. Worse, the compiler cannot *inline* across a virtual call — it cannot paste the function's body into the caller and optimize the combination — because it does not know at compile time which function will run.

### Templates, CRTP, and inlining

The alternative is *compile-time polymorphism* with **templates**. A template is a recipe the compiler stamps out a concrete version of, for each type you use it with, *at compile time*. Because the exact function is known when the code is compiled, there is no vptr, no vtable lookup, no indirect jump — just a direct call the optimizer can **inline**, pasting the body into the caller and then folding away dead code around it. A common pattern for this is **CRTP** (the Curiously Recurring Template Pattern), where a base class is templated on its derived class so it can call into the derived type's methods with zero runtime indirection.

![Virtual dispatch vs a templated call: a virtual call costs an indirect jump and a possible mispredict while a templated call inlines to nothing](/imgs/blogs/cpp-for-low-latency-quant-interviews-10.png)

```cpp
// Runtime polymorphism: a virtual call on the hot path
struct Strategy {
    virtual void on_tick(const Tick&) = 0;   // virtual: dispatched at runtime
};
void run(Strategy* s, const Tick& t) {
    s->on_tick(t);   // load vptr, load fn ptr, indirect jump, maybe mispredict
}

// Compile-time polymorphism (CRTP): resolved and inlined at compile time
template <typename Derived>
struct StrategyBase {
    void on_tick(const Tick& t) {
        static_cast<Derived*>(this)->handle(t);  // direct call, inlinable
    }
};
struct MyStrategy : StrategyBase<MyStrategy> {
    void handle(const Tick& t) { /* ... */ }     // known at compile time
};
```

#### Worked example: virtual versus templated dispatch cost

A virtual call that hits in cache and predicts correctly costs only a couple of nanoseconds more than a direct call — virtual dispatch is *not* always expensive, and saying "virtuals are slow, period" is a way to fail the interview. The cost shows up in two specific cases. First, when the vtable or the target code is *not* in cache, the loads miss (~100 ns extra). Second, when the call site is *unpredictable* — you are dispatching among many different strategy types in an unpredictable order — the branch predictor misses often, ~15–20 cycles (~5–7 ns) per miss. The deeper, compounding cost is the lost *inlining*: a tiny `on_tick` that does almost nothing could have been inlined and folded into a few instructions, but behind a virtual call it stays a full function call with all its overhead, and the optimizer cannot see through it to optimize the surrounding loop.

The templated version has none of this: the call is direct, inlined, and the compiler optimizes the whole fused result. On a hot path that calls `on_tick` millions of times a second, removing the indirection and unlocking inlining can shave meaningful nanoseconds and, just as importantly, remove jitter.

**The intuition:** virtual dispatch is fine almost everywhere, but on the innermost hot loop you avoid it — not because the indirect jump is catastrophic, but because it blocks inlining and adds unpredictability. Templates and CRTP give you the same polymorphism resolved at compile time, with zero runtime cost. The nuanced answer ("virtuals are usually fine; here's exactly when they hurt") is what distinguishes a strong candidate.

## Measuring latency: measure, don't guess

Everything above is a hypothesis until you measure. The cardinal rule of low-latency engineering — the one every interviewer wants to hear — is **measure, do not guess**. Human intuition about where time goes is reliably wrong; the compiler and the hardware do surprising things; the only truth is the profiler and the timer.

And when you measure latency, you must measure the **distribution**, not the average. Latency is not a single number. It is a right-skewed distribution: most events are fast, but a long tail of slow ones drags out to the right.

![Measure the distribution, not the average: latency is right-skewed and the p99 tail sits far above the median the mean conceals](/imgs/blogs/cpp-for-low-latency-quant-interviews-11.png)

A **percentile** tells you the value below which a given fraction of events fall. The **p50** (the *median*, 50th percentile) is the typical case — half your ticks are faster, half slower. The **p99** is the value below which 99% of events fall — your "1-in-100 bad day." The **p99.9** is your one-in-a-thousand. In the histogram, p50 is 450 ns but p99 is 1500 ns: the worst 1% of your ticks are more than three times slower than typical.

Why does the tail matter more than the average in trading? Because **you lose the race on your slow events, not your median ones**. If a profitable opportunity appears and ten firms react, the trade goes to whoever is fastest *at that instant*. Your median latency is irrelevant in that moment; what matters is whether *this particular reaction* was fast. A strategy with a great median but a fat tail loses the contested, profitable trades — exactly the ones worth winning — and the *mean* hides this entirely, because the mean (~560 ns here) sits between the median and the tail and describes neither.

How you measure matters too. On the hot path you read the CPU's cycle counter (`rdtsc` on x86, or `std::chrono::steady_clock` with care) at the start and end of the path, store the delta into a pre-allocated histogram, and analyze it *off* the hot path. You never log or print inside the measured region — that would allocate and do I/O, perturbing the very thing you measure. This is the same discipline as [backtesting a strategy honestly](/blog/trading/quantitative-finance/backtesting-done-right-quant-research): the measurement must not distort the system, and the summary statistic must be the one that matches how you actually win or lose.

#### Worked example: why the average lies

Two systems, A and B, each handling 1,000,000 ticks:

- **System A:** every tick takes exactly 500 ns. Mean = 500 ns, p50 = 500 ns, p99 = 500 ns, p99.9 = 500 ns.
- **System B:** 990,000 ticks take 300 ns, 9,000 take 2,000 ns, 1,000 take 50,000 ns (a GC-like or allocation-driven hiccup). Mean = (990,000×300 + 9,000×2,000 + 1,000×50,000) / 1,000,000 = (297,000,000 + 18,000,000 + 50,000,000) / 1,000,000 = 365 ns.

By the *mean*, System B (365 ns) looks faster than System A (500 ns). But B's p99.9 is 50,000 ns — every thousandth tick, B stalls for 50 microseconds, which on a busy market open is when the money is. A is slower on average and *vastly* better where it counts. If you reported only the mean, you would ship the worse system.

**The intuition:** report and optimize the tail. The mean is the most misleading number in latency engineering, and "we measure p99 and p99.9, not the average" is the answer that signals you have actually run a low-latency system.

## In the interview room

Here are five fully-solved problems in the style these desks actually use. For each, the point is not just the answer but the *reasoning out loud* — interviewers grade the walk, not the destination.

#### Worked example: problem 1 — spot the bug

> "Read this. What's wrong, and what would you do?"

```cpp
std::string_view get_top_symbol(const OrderBook& book) {
    std::string sym = book.best_bid().symbol();  // a local std::string
    return sym;                                  // returned as a string_view
}
```

**Solution.** The function returns a `std::string_view`, which is a *non-owning* view — a pointer into some bytes plus a length. It views into `sym`, a local `std::string` that owns a heap buffer. The instant the function returns, `sym`'s destructor runs and frees that buffer. The returned `string_view` now points at freed memory: a **dangling view**, and reading through it is undefined behavior. The trap is that it usually *appears* to work, because the freed bytes are not immediately overwritten. The fix is to return a `std::string` by value (the caller then owns the bytes; the return is a cheap move), or to return a view into something whose lifetime *outlives* the call — e.g., a view into the book itself if the book guarantees the symbol's storage persists. The thing to say: "a `string_view` never extends the lifetime of what it views; I have to make sure the underlying storage outlives every view of it."

#### Worked example: problem 2 — make this struct faster

> "This struct is used in a hot array of millions. Improve its layout and explain the win."

```cpp
struct Order {           // assume 8-byte alignment
    bool   is_buy;       // 1 byte, then 7 bytes padding
    double price;        // 8 bytes
    bool   is_active;    // 1 byte, then 7 bytes padding
    double size;         // 8 bytes
    char   id[3];        // 3 bytes, then 5 bytes padding
};                       // total: 8 + 8 + 8 + 8 + 8 = 40 bytes
```

**Solution.** The fields are ordered so that each `bool`/`char` forces *padding* — the compiler inserts unused bytes so that each `double` lands on an 8-byte boundary (its alignment requirement). As written, the struct is 40 bytes, of which 19 are padding. Reorder fields largest-alignment-first to pack them:

```cpp
struct Order {
    double price;     // 8
    double size;      // 8
    char   id[3];     // 3
    bool   is_buy;    // 1
    bool   is_active; // 1   -> 3 + 1 + 1 = 5 bytes, fits with 3 padding to 8
};                    // total: 8 + 8 + 8 = 24 bytes
```

Now it is 24 bytes instead of 40 — a 40% reduction. In an array of millions, that means 40% fewer bytes to stream through cache and, concretely, more orders per 64-byte line: 24-byte orders give ~2.67 per line versus ~1.6 for the 40-byte version, so a scan touches far fewer lines and misses far less. The win is "smaller struct → more elements per cache line → fewer misses on traversal." The interviewer wants you to *name padding and alignment* and connect the byte count to cache-line occupancy — the same logic as the AoS/SoA example, applied within a single struct.

#### Worked example: problem 3 — why is this loop slow?

> "We profiled this and it's the bottleneck. Why?"

```cpp
double total = 0;
for (const auto* order : active_orders)   // active_orders is std::list<Order*>
    total += order->notional();
```

**Solution.** There are *two* layers of pointer-chasing here, both cache-hostile. First, `active_orders` is a `std::list`, so walking it follows a `next` pointer to a separately-allocated node each step — a likely cache miss per node, no stride for the prefetcher. Second, each element is a `Order*` — a pointer to an `Order` that lives somewhere *else* on the heap — so even after reaching the node, dereferencing `order->notional()` is *another* probable miss to wherever that order sits. You are missing the cache roughly twice per iteration. The fix: store the orders contiguously by value in a `std::vector<Order>` (one stride, prefetcher-friendly, the orders themselves are in the lines you load), or if you must keep handles, store them in a contiguous `std::vector` and ensure the pointed-to objects come from a contiguous pool. Either way you collapse two miss sources into a clean sequential scan. Say the magic words: "linked list of pointers to heap objects — that's a cache miss to find the node and another to read the object; I'd make the storage contiguous."

#### Worked example: problem 4 — design a lock-free single-producer single-consumer queue

> "One thread produces ticks, one consumes them. Design the handoff without a lock. What's the trick?"

**Solution.** Use a **ring buffer**: a fixed-size array plus two indices, a `head` (where the consumer reads) and a `tail` (where the producer writes), each wrapping around modulo the array size. Pre-allocate the array at startup so there is zero allocation on the hot path. The producer writes an element at `tail`, then publishes by advancing `tail`; the consumer reads at `head` and advances `head`. Because there is exactly one producer and one consumer, each index is written by only one thread, so you avoid locks entirely — you only need *atomics* with the right memory ordering to ensure the consumer sees the data write before it sees the index advance (a *release* store on the producer's index, an *acquire* load on the consumer's). The two critical low-latency refinements: (1) the array is pre-allocated and reused — no allocation per message; (2) put `head` and `tail` on *separate cache lines* with `alignas(64)`, or the producer's writes to `tail` and the consumer's writes to `head` cause **false sharing** and ping-pong the line between cores. This problem bundles three of the post's themes — pre-allocation, atomics/memory-ordering, and false sharing — which is exactly why it is a favorite. The interviewer is listening for "ring buffer, pre-allocated, release/acquire on the index, and pad the indices to avoid false sharing."

#### Worked example: problem 5 — should this be virtual?

> "We have 50 strategy classes. The dispatcher calls `on_tick` on whichever is active. Should `on_tick` be virtual? Defend your answer."

**Solution.** "It depends on the call pattern, and here is the reasoning." If at any given moment a *single* strategy is active and the dispatcher calls it in a tight loop millions of times, the virtual call is the same target every time — the branch predictor nails it and the vtable load stays hot in cache, so the *direct* cost is tiny (a few nanoseconds). But the *real* cost is the lost inlining: a small `on_tick` that does little work can't be inlined behind a virtual call, so the optimizer can't fuse it with the surrounding loop, and you eat full call overhead each tick. If the hot loop is genuinely the bottleneck, I'd make it compile-time polymorphic — template the dispatcher on the strategy type (or use CRTP) so the call is direct and inlinable, and stamp out one specialized dispatcher per strategy. If instead the dispatch is rare, or you truly need to switch among many strategies unpredictably at runtime, the virtual call is fine and the design clarity is worth more than the nanoseconds. The trap answer is the absolutist one — "always avoid virtuals" or "virtuals are always fine." The strong answer names the *two specific costs* (mispredict on unpredictable sites, lost inlining always) and decides based on whether this call is on the measured hot path. Then add: "but I'd measure it before changing anything."

That last line — *measure before changing* — is the through-line graders reward. The skills overlap with the broader [market-making interview games](/blog/trading/quantitative-finance/market-making-games-quant-interviews) and [Fermi-style estimation](/blog/trading/quantitative-finance/estimation-fermi-problems-quant-interviews): in all of them, the desk is testing whether you reason from first principles and quantify, not whether you recite.

## Common misconceptions

**"Premature optimization is the root of all evil — so I shouldn't worry about any of this."** Donald Knuth's famous line is real, but it is quoted with the qualifier amputated. The full sentence grants that the critical 3% *should* be optimized. The whole skill of low-latency engineering is telling the 97% (leave it readable, don't micro-optimize) from the 3% (the measured hot path, where nanoseconds are dollars). The mistake is not optimizing; it is optimizing *the wrong code*, or optimizing *before measuring*. On the hot path, this work is not premature — it is the job.

**"The compiler will fix it."** The compiler is extraordinary at local optimizations — register allocation, inlining what it can see, vectorizing clean loops, folding constants. It cannot fix your *data layout*: it will not turn your `std::list` into a contiguous array, will not restructure AoS into SoA, will not remove an allocation you asked for, and will not un-pad a struct whose field order you chose. It also cannot see through a virtual call to inline it, or undo a copy you wrote where you meant a move. The architecture-level decisions — how data is laid out and how the path is shaped — are yours; the compiler optimizes within the structure you give it.

**"Virtual functions are slow, always avoid them."** Covered above, but it bears repeating because it is the single most common over-correction. A predictable, cache-hot virtual call costs only a couple of nanoseconds. The cost is real only on unpredictable call sites (mispredicts) or where lost inlining matters (innermost hot loops). Everywhere else, virtuals buy you clean, maintainable polymorphism for a price you cannot measure. The expert position is nuanced, not absolutist.

**"Allocation is fine, modern allocators are fast."** Modern allocators (tcmalloc, jemalloc) *are* impressively fast — for an allocator. But "fast for an allocator" is still 30 to 200 ns with a tail, versus ~1 ns for a stack variable or ~10 ns for a pool slot. And the tail is the killer: an allocation that usually takes 50 ns but occasionally takes 5 µs (lock contention, page fault, OS getting involved) puts a fat spike in your p99.9 exactly when you can least afford it. The hot path doesn't allocate, full stop.

**"Faster hardware will make my slow C++ fast enough."** Hardware helps, but the gaps in this post are *ratios*, not absolute amounts. A cache miss is ~100× an L1 hit on a 2015 CPU and still ~100× on a 2026 CPU — both got faster, but the cliff is the same shape. A linked list that loses to an array by 50× loses by ~50× on next year's chip too. You cannot buy your way out of a bad data layout; you have to fix the layout.

## How it shows up on a real desk

**The tick-to-trade hot path.** This is the heart of it and the thing every section served. A market-data packet arrives at a network card. At a competitive firm the card writes it straight into your process's memory (kernel bypass, via something like a Solarflare/Onload or an FPGA), skipping the operating system. Your code — pre-allocated, no heap, no virtuals on the inner loop, data laid out contiguously — parses the packet, updates the relevant slice of the order book in place, runs a compile-time-dispatched strategy that reads a handful of cache-resident values and decides, encodes an order into a pre-allocated buffer, and hands it to the card. The whole path is a few hundred nanoseconds, and the engineering is relentlessly about *not* doing the expensive things: not allocating, not chasing pointers, not dispatching indirectly, not touching cold memory. Every technique in this article is a tool for keeping that path short and jitter-free.

**The order book itself.** The central data structure is the order book — bids and asks at each price level. A naive design uses a `std::map<price, level>` (a balanced tree), which is pointer-chasing per lookup, every node a potential cache miss. Real low-latency books use a flat array indexed by price *ticks* — a contiguous array where the price directly computes the index, so the level you want is a single cache-friendly indexed read, often already prefetched. The transformation is exactly the array-versus-linked-list and AoS-versus-SoA reasoning from this post, applied to the most-accessed structure in the system. A candidate who can sketch a tick-indexed array book and explain its cache advantage over a `std::map` is demonstrating the whole skill set at once.

**The measurement infrastructure.** Desks invest heavily in measuring tick-to-trade, because you cannot improve what you cannot see, and the number that matters is the tail. They timestamp at the network card (hardware timestamps), again when the strategy decides, again when the order leaves, and reconcile against the exchange's own timestamps. The output is histograms and p99/p99.9 tails, not averages, watched continuously. When the p99.9 creeps up, someone hunts the regression — often a sneaky allocation, a newly cold cache line, or a layout change that broke the prefetcher. This is the same measure-don't-guess discipline as honest [backtesting](/blog/trading/quantitative-finance/backtesting-done-right-quant-research), turned on the system's own latency.

**The 2012 Knight Capital blow-up.** Not a latency story but a vivid systems-correctness one: a deployment left old code active on some servers, and over 45 minutes the firm sent millions of erroneous orders, losing about \$440 million and effectively ending the company. The HFT lesson the desks internalized: in a system this fast, a bug doesn't cost you a wrong answer on a screen — it costs you millions before a human can react. It is why the interview cares so much about correctness (undefined behavior, lifetimes, data races) *alongside* speed. Fast wrong code is the most expensive code there is.

**Why the dollars justify the obsession.** It is fair to ask whether shaving nanoseconds is really worth this much engineering. Put it in money.

![The dollar cost of being late: edge forfeited per day climbs steeply as fills arrive later](/imgs/blogs/cpp-for-low-latency-quant-interviews-12.png)

#### Worked example: the dollar cost of 10 microseconds

Take a market-making strategy with an edge of \$0.005 (half a cent) per share, trading 30,000,000 shares a day. Its gross edge is 30,000,000 × \$0.005 = \$150,000 a day. Now suppose a latency regression makes your fills arrive 10 µs later than a competitor's. You do not lose *all* your trades — you lose the *contested* ones, the profitable opportunities where speed decides the winner, and you get *adversely selected* on others (you get filled precisely when the price is about to move against you, because the faster trader took the good side first). Suppose that costs you 10% of your edge: \$15,000 a day. Over ~250 trading days, that is \$3.75 million a year — from 10 microseconds. Push the lateness to 100 µs and you are mostly adverse-selected, forfeiting a large fraction of the edge; the daily figure climbs toward \$60,000 and the strategy may stop being viable at all.

These are illustrative numbers, not a quote from any specific firm — the exact edge, volume, and speed-sensitivity vary enormously by strategy and venue, and this is educational, not trading advice. But the order of magnitude is real, and it is the honest answer to "why does a few nanoseconds justify a team of C++ specialists." The latency *is* the edge. A microsecond is not a technicality on these desks; it is a line item.

**The intuition for the whole post:** the reason an HFT interview grills you on cache lines and allocation and undefined behavior is that, on the hot path, those details convert directly into dollars of captured-or-forfeited edge, multiplied by enormous volume, every single day.

## When this matters and further reading

If you are interviewing at a low-latency or market-making firm, the takeaway is concrete: be able to walk the tick-to-trade path out loud, put nanoseconds on each stage, and justify every layer you used or avoided with a cache or allocation argument. Practice *spotting* undefined behavior and hidden allocations in code, because reading-and-critiquing is half of what these interviews are. And whenever you give a performance answer, end it with "and I'd measure to confirm" — the measure-don't-guess instinct is what separates someone who memorized this post from someone who could do the job.

If you are simply a curious engineer, the deeper lesson generalizes well beyond trading: performance is dominated by the memory hierarchy, not by clever arithmetic; the average hides the tail that actually hurts you; and the highest-leverage decisions — data layout, allocation strategy, dispatch — are architectural choices the compiler cannot make for you. That is as true in a database, a game engine, or a machine-learning kernel as it is on a trading desk.

To go further: read a CPU architecture primer on the cache hierarchy and branch prediction; work through *what every programmer should know about memory* for the hardware foundations; study a real open-source low-latency library to see pre-allocation, ring buffers, and CRTP in production; and write a tiny benchmark of your own (vector versus list, AoS versus SoA, virtual versus templated) and *measure it* — the numbers will surprise you, and being surprised by your own measurements is the first real step into this field. From here, the natural companions are the interview-craft posts on [mental math and arithmetic speed](/blog/trading/quantitative-finance/mental-math-arithmetic-speed-quant-interviews) and [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews), which test the same first-principles, quantify-everything reflex from the probability side of the same interview loop.

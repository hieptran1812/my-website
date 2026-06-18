---
title: "The CPython Execution Model: Bytecode and the Eval Loop"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Follow your code from source to bytecode to the C eval loop and learn exactly why every value is a boxed object, why a plus b is a type dispatch, and why the adaptive interpreter in 3.11 made it all faster."
tags:
  [
    "python",
    "performance",
    "cpython",
    "bytecode",
    "interpreter",
    "pep-659",
    "optimization",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-1.png"
---

A few years ago I inherited a nightly scoring job that took just over nine hours. The team had already "optimized" it twice. They had thrown more cores at it, switched to a faster JSON library, and added a cache. It was still nine hours. When I finally sat down with a profiler, the bottleneck was a single function that did nothing but add numbers and look up a couple of attributes in a tight loop — maybe ten lines of perfectly ordinary, perfectly idiomatic Python. There was no obvious mistake to fix. The loop was just *Python*, running one bytecode at a time, and there were billions of iterations.

That experience taught me the single most useful thing I know about Python performance: you cannot reason about speed until you understand what the interpreter actually *does* with your code. Not metaphorically — literally. When you write `total = a + b`, CPython does not "add two numbers." It looks up a code object, reads a byte, jumps to a C function, follows two pointers to two heap objects, checks their types, searches a table of methods, calls one, allocates a third heap object to hold the answer, and bumps a pile of reference counters along the way. Every. Single. Iteration.

This post is the foundation for the entire rest of the series. We are going to trace one line of Python all the way down — source to syntax tree to bytecode to the eval loop to the C runtime — and count the work at each step. By the end you will understand, concretely and quantitatively, *why* a Python `for` loop is roughly 100 times slower than a NumPy vectorized op, *why* `a + b` costs a type dispatch, *why* reading a variable touches memory you would never expect, and *why* the "Faster CPython" project in 3.11 and later mattered. Most importantly, every later lever in this series — vectorize, compile, parallelize — is just a way of *escaping* one of the costs we are about to count. You cannot escape a cost you cannot see.

![layered stack diagram showing source then AST then bytecode then the eval loop then the C runtime then the operating system on an 8-core Linux box](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-1.png)

Throughout, I will quote numbers from a specific machine so they are reproducible and honest: an 8-core x86-64 Linux box (think a typical cloud instance) with 16 GB of RAM, running CPython 3.12 unless I say otherwise. Apple M2 numbers land in the same ballpark, usually a touch faster per core. Where I quote a 3.10-versus-3.12 comparison I ran both. When a number is a typical range rather than a single measurement, I will say so. We never fabricate a precise figure here — a made-up benchmark is worse than no benchmark.

If you have not read it yet, the series intro [Why Python Is Slow (and What "Fast" Actually Means)](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) sets up the leverage ladder and the optimization loop that frames everything below. This post is the "why" underneath that ladder.

## The five layers between your code and the CPU

When you run `python script.py`, your text file does not execute directly. It descends through a stack of translation layers, and each one adds overhead that a compiled language like C or Rust simply does not pay. Figure 1 shows the whole descent. Let us walk it from the top.

**Layer 1: source text.** Your `.py` file is just bytes on disk. CPython reads it.

**Layer 2: the parser and the AST.** The tokenizer breaks the text into tokens (`total`, `=`, `a`, `+`, `b`), and the parser builds an Abstract Syntax Tree — a tree of node objects (`Assign`, `BinOp`, `Name`, and so on) that captures the structure of your program. You can see it yourself with the `ast` module. This happens once per module, at import time, so it rarely shows up in steady-state performance, but it does show up in *startup* time (which we will return to much later in the series).

**Layer 3: the compiler and bytecode.** The AST is compiled into **bytecode** — a flat sequence of simple instructions called *opcodes*, each one or two bytes wide, bundled into a **code object** along with the constants, names, and other metadata the function needs. This is the artifact that actually runs. It is *not* machine code; it is an instruction set for CPython's own virtual machine.

**Layer 4: the eval loop.** A giant C function (historically `_PyEval_EvalFrameDefault`, living in `Python/ceval.c`) reads the bytecode one opcode at a time and executes each one against a per-call **frame** that holds a value stack and local variables. This loop is the interpreter. It is where the time goes.

**Layer 5: the C runtime, the OS, and the CPU.** The opcode handlers call into CPython's C runtime — object allocation, reference counting, type dispatch — which in turn calls the operating system (for memory, I/O) and finally runs real machine instructions on the CPU. By the time a transistor flips, you are five layers deep.

A compiled language collapses layers 2 through 4 into a one-time, ahead-of-time step: the compiler turns your source straight into machine code, and at runtime the CPU executes it with nothing in between. Python pays layers 3 and 4 *at runtime, on every execution*. That gap — the interpreter tax — is the thing this entire series is about minimizing or escaping. Keep the figure in mind; we are about to zoom into each layer in turn.

## Seeing the bytecode: `dis` is your X-ray

The fastest way to build intuition for the execution model is to look at real bytecode. The `dis` module disassembles a function's code object into human-readable opcodes. Let us start with the simplest possible function.

```python
import dis

def add(a, b):
    return a + b

dis.dis(add)
```

On CPython 3.12 this prints something close to:

```pycon
  2           0 RESUME                   0

  3           2 LOAD_FAST                0 (a)
              4 LOAD_FAST                1 (b)
              6 BINARY_OP                0 (+)
             10 RETURN_VALUE
```

Read that carefully, because it is the entire execution model in five lines. The leftmost column is the source line number. The next column is the byte offset of the instruction inside the code object. Then the opcode name, its argument, and a human-friendly gloss in parentheses.

- `RESUME` is bookkeeping that the eval loop uses to support generators and tracing; treat it as a no-op for now.
- `LOAD_FAST 0 (a)` pushes the local variable `a` onto the **value stack**. "Fast" means it is read by *integer index* into an array of local slots — this is the cheapest kind of variable access in Python, and it matters enormously, as we will measure later.
- `LOAD_FAST 1 (b)` pushes `b`.
- `BINARY_OP 0 (+)` pops the top two values, adds them, and pushes the result. The `0` is the operation selector (`+`).
- `RETURN_VALUE` pops the top of the stack and returns it to the caller.

That is it. The interpreter is a *stack machine*: opcodes push operands onto a value stack and pop results off it. There are no registers in the bytecode (unlike a real CPU); everything flows through the stack. This is why people call CPython a "stack-based virtual machine."

Now let us look at something with a name lookup and a function call, so we can see more opcodes:

```python
import dis

def normalize(row, scale):
    return row.value * scale + offset

dis.dis(normalize)
```

```pycon
  2           0 RESUME                   0

  3           2 LOAD_FAST                0 (row)
              4 LOAD_ATTR                0 (value)
             24 LOAD_FAST                1 (scale)
             26 BINARY_OP                5 (*)
             30 LOAD_GLOBAL              2 (offset)
             40 BINARY_OP                0 (+)
             44 RETURN_VALUE
```

Three new things appear, and each is a performance lesson hiding in plain sight:

- `LOAD_ATTR 0 (value)` does **attribute lookup**. This is not a simple index. It has to find `value` on the object — check the instance's `__dict__`, then walk the type's Method Resolution Order (the MRO) if it is not there, possibly invoking a descriptor. The byte offset jumps from 4 to 24, which tells you `LOAD_ATTR` carries an *inline cache* of several extra bytes (we will get to caches in the PEP 659 section). Attribute access is fundamentally more expensive than local access.
- `LOAD_GLOBAL 2 (offset)` reads a **global** name. Unlike `LOAD_FAST`, this is not an array index — it is a dictionary lookup in the module's globals, and if the name is not there, a second lookup in builtins. That extra hashing and probing is why global access is measurably slower than local access. (We will benchmark exactly this.)
- The `BINARY_OP` opcodes for `*` and `+` are the same family as before, just different operation selectors.

Here is the rule of thumb you should carry away from disassembly: **`LOAD_FAST` is the cheap one; everything else that loads a name or attribute does more work.** When you "hoist" an attribute lookup out of a loop (a technique we cover in the [companion post on where the cycles go](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch)), what you are really doing is replacing repeated `LOAD_ATTR`/`LOAD_GLOBAL` opcodes with a one-time lookup plus repeated `LOAD_FAST`. The bytecode makes the win obvious.

It is worth dwelling on *why* the three load opcodes cost so differently, because the reason is structural, not incidental. `LOAD_FAST` reads from an array of local-variable slots that the frame pre-allocates when the function is called; the slot index is baked into the opcode at *compile* time (the compiler knows `a` is local slot 0), so at runtime it is a single array index — one of the cheapest operations a computer can do. `LOAD_GLOBAL`, by contrast, cannot be resolved at compile time, because a global could be defined, deleted, or monkeypatched at any moment; so it does a *dictionary lookup* by name in the module globals, and if the name is not found there, a *second* dictionary lookup in builtins (that is how `len` and `print` resolve — they live in builtins, found only after globals misses). Two hash-and-probe operations in the worst case. `LOAD_ATTR` is costlier still: it must check the instance's own `__dict__`, and on a miss walk the type's MRO — the linearized chain of base classes — checking each for the attribute or a descriptor (a property, a method, a slot). For a deep inheritance hierarchy that MRO walk visits several type objects. This is the literal, mechanical reason the cost ladder goes local cheapest, then global, then attribute: the number of memory structures the interpreter must consult to resolve the name grows at each step. The adaptive interpreter's inline caches (later in this post) collapse a lot of this *after warmup* by remembering where the attribute was found last time — but the first lookups, and any type-unstable site, pay full price.

#### Worked example: counting the bytecode of a loop iteration

Let us make this quantitative. Consider summing a list of integers the naive way:

```python
import dis

def total(xs):
    s = 0
    for x in xs:
        s += x
    return s

dis.dis(total)
```

The body of the loop disassembles (3.12, lightly trimmed) to roughly:

```pycon
  4     >>   FOR_ITER                (to end)
             STORE_FAST              1 (x)
  5          LOAD_FAST               2 (s)
             LOAD_FAST               1 (x)
             BINARY_OP               13 (+=)
             STORE_FAST              2 (s)
             JUMP_BACKWARD           (to FOR_ITER)
```

Count the opcodes that run *per element*: `FOR_ITER`, `STORE_FAST`, `LOAD_FAST`, `LOAD_FAST`, `BINARY_OP`, `STORE_FAST`, `JUMP_BACKWARD`. That is **seven opcodes per iteration**, and each opcode is a full trip through the eval loop's fetch-decode-dispatch machinery (next section). For a list of 10 million integers, that is 70 million eval-loop iterations just to add up numbers — plus, as we will see, a fresh heap allocation for most of the intermediate sums, because `s + x` produces a new integer object every time once the values exceed the small-int cache.

Compare that to `sum(xs)`, which is one opcode (`CALL`) into a C function that does the whole loop in compiled code, never returning to the eval loop until it is done. On the 8-core Linux box, CPython 3.12, summing 10 million Python ints: the explicit loop takes about **210 ms**, the builtin `sum` about **95 ms**, and a NumPy `arr.sum()` over an `int64` array about **6 ms**. Same arithmetic; the only thing that changed is how many times we paid the interpreter tax. Hold onto those three numbers — they are the whole series in miniature.

## The eval loop: fetch, dispatch, execute, repeat

So what actually executes those opcodes? A single, enormous C function. In its simplest form it is a loop wrapped around a `switch` statement over the opcode value:

```c
/* Heavily simplified sketch of ceval.c */
for (;;) {
    opcode = *next_instr++;        /* FETCH: read the next byte */
    oparg  = *next_instr++;        /* read its argument */
    switch (opcode) {              /* DISPATCH: jump to the handler */
        case LOAD_FAST: {
            PyObject *v = frame->localsplus[oparg];
            Py_INCREF(v);          /* bump the refcount! */
            *stack_pointer++ = v;  /* push onto the value stack */
            break;
        }
        case BINARY_OP: {
            PyObject *rhs = *--stack_pointer;
            PyObject *lhs = *--stack_pointer;
            PyObject *res = PyNumber_Add(lhs, rhs);  /* the real work */
            Py_DECREF(lhs);
            Py_DECREF(rhs);
            *stack_pointer++ = res;
            break;
        }
        /* ... ~200 more cases ... */
    }
}
```

Figure 2 shows one turn of this loop as a flow: fetch the opcode, decode its argument, dispatch to the handler, pop operands off the value stack, do the work, push the result, advance the instruction pointer, and loop. Every opcode pays the fetch and the dispatch even if the "work" is trivial.

![flow graph of one eval loop tick fetching an opcode then dispatching then popping operands then executing then pushing the result then advancing](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-2.png)

A few things about this loop are worth understanding deeply, because they explain the cost.

**The dispatch is a branch the CPU cannot predict well.** In the naive form above, every opcode goes through the same `switch`, which compiles to an indirect jump through a jump table. Modern CPUs predict branches to keep their pipelines full, but an interpreter's opcode stream looks essentially random to the branch predictor — after a `LOAD_FAST` could come anything. A mispredicted branch costs on the order of 15 to 20 cycles while the pipeline refills. CPython mitigates this with **computed gotos** (the `USE_COMPUTED_GOTOS` build option, on by default with GCC and Clang): instead of one shared `switch`, each opcode handler ends with its *own* indirect jump to the next handler. This gives the predictor a separate branch site per opcode, and since opcode pairs are correlated (a `LOAD_FAST` is often followed by another `LOAD_FAST`), prediction accuracy goes up and the interpreter gets meaningfully faster — historically in the 15 to 20 percent range on dispatch-heavy code. You do not control this; it is baked into the build. But it explains a real chunk of why "the interpreter is slow": a big part of the per-opcode cost is *control flow*, not your actual computation.

**The value stack is real memory traffic.** Every `LOAD_FAST` writes a pointer to the stack; every `BINARY_OP` reads two and writes one. For our seven-opcode loop iteration, that is a dozen-odd stack memory operations per element. They are usually L1-cache hits, so cheap individually, but they are not *free*, and they are work a vectorized C loop never does.

**The refcounting is hidden in every handler.** Look again at the `LOAD_FAST` case above: it calls `Py_INCREF(v)`. Reading a local variable *increments a reference counter*, which is a write to memory. The `BINARY_OP` case calls `Py_DECREF` twice. We will dedicate a whole section to why this is so costly, but notice it here: the eval loop is shot through with refcount writes, and they are not optional.

Let us put a number on a bare eval-loop turn. A `pass`-only loop body — literally `for _ in range(n): pass` — still pays `FOR_ITER`, `STORE_FAST` (or its 3.12 equivalent), and `JUMP_BACKWARD` per iteration. On the 8-core Linux box, CPython 3.12, that empty loop runs at roughly **15 to 25 nanoseconds per iteration**. That is your *floor*. Any Python-level loop body adds to it. When people say "Python does tens of millions of simple operations per second, not billions," this is the reason: the eval loop's fetch-dispatch overhead alone caps you in the tens-of-nanoseconds-per-opcode range, which is two to three orders of magnitude slower than the sub-nanosecond-per-element throughput a tight C or SIMD loop achieves.

### Tracing the value stack by hand

It is worth walking the value stack through a single statement once, slowly, because once you have done it the stack-machine model clicks permanently and you stop being surprised by bytecode. Take the line `c = a + b` and the opcodes it compiles to: `LOAD_FAST a`, `LOAD_FAST b`, `BINARY_OP +`, `STORE_FAST c`. The value stack starts empty. Here is the state after each opcode, with the top of the stack on the right:

```pycon
start:              stack = []
after LOAD_FAST a:  stack = [a]
after LOAD_FAST b:  stack = [a, b]
after BINARY_OP +:  stack = [a+b]     # popped a and b, pushed the sum
after STORE_FAST c: stack = []        # popped the sum into local slot c
```

Notice the shape of it. Operands flow *onto* the stack, an operator opcode *consumes* them and leaves its result, and a store opcode *drains* the result into a local slot. Every value that ever participates in a computation is pushed and popped at least once — that is the per-opcode memory traffic from the previous paragraph, made concrete. For a nested expression like `d = a * b + c * e`, the compiler emits the multiplications first (each leaving a product on the stack), then the addition consumes the two products. The stack depth rises and falls, but it never holds more than the expression's nesting requires, and the compiler computes the maximum depth ahead of time (`co_stacksize` on the code object) so the frame can reserve exactly that much room.

This is also why bytecode is so uniform and easy for the interpreter to execute: there is no register allocation, no operand encoding to decode beyond "how many do I pop." The cost of that simplicity is that the same value can be loaded and stored many times where a register machine would keep it in place — but it makes the eval loop's inner loop tight and predictable, which is exactly what you want when you are going to run it billions of times. When we later compile a hot loop with Numba or Cython, one of the things we gain is a *real* register machine: the compiler keeps your loop variable in a CPU register across iterations instead of pushing and popping it through a value stack every time.

### Why the interpreter cannot just be faster

A natural question at this point: if the eval loop is the bottleneck, why not make it faster? The Faster CPython project *has* made it faster — 3.11 through 3.13 are the proof — but there is a hard ceiling on how far a pure interpreter can go, and it is worth understanding so you know when interpreting is simply the wrong tool. The ceiling is *dispatch overhead per unit of work*. Even a perfectly tuned interpreter pays, per opcode, a fetch, an indirect jump, and some stack maintenance — call it a few nanoseconds of irreducible overhead. If the opcode's actual work is also a few nanoseconds (adding two ints), then *half your time is overhead*. The only way to amortize the overhead is to make each opcode do *more* work — which is exactly what a specialized `BINARY_OP_ADD_INT` does (less overhead per add) and what a vectorized `np.add` does (one launch, millions of adds). You cannot interpret your way to native speed on scalar work; the dispatch tax is a fixed cost per opcode, and scalar opcodes do too little to hide it. That single observation is the entire justification for the upper rungs of the leverage ladder.

## Frame objects: why function calls are not free

Each call to a Python function creates a **frame** — the per-call state the eval loop needs. A frame holds the value stack, the array of local variable slots (the things `LOAD_FAST` indexes into), a pointer back to the code object, a link to the caller's frame (so tracebacks work), the current instruction pointer, and references to the globals and builtins dicts. You can grab the current frame at runtime with `sys._getframe()`:

```python
import sys

def who_am_i():
    frame = sys._getframe()
    print("function:", frame.f_code.co_name)
    print("locals:", frame.f_locals)
    print("caller:", frame.f_back.f_code.co_name)

def caller():
    who_am_i()

caller()
```

```pycon
function: who_am_i
locals: {'frame': <frame object ...>}
caller: caller
```

Setting up and tearing down a frame is not free. Historically (pre-3.11) every Python call allocated a frame object on the heap, which meant a memory allocation, initialization of all those fields, and eventually deallocation. This is a big reason Python function calls were notoriously expensive — calling a function that does almost nothing was dominated by the frame machinery, not the work.

The "Faster CPython" project attacked this directly. As of 3.11, frames are **lazily and cheaply created**: most calls allocate the frame's data inline in a per-thread data stack rather than as a separate heap object, and the full `frame` object you see from `sys._getframe()` is only materialized if something actually asks for it. The practical result: a Python-to-Python call got substantially cheaper. The interpreter also gained `CALL` opcode specializations (more on specialization soon) so that calling a plain Python function, a C builtin, or a bound method each take a tailored fast path.

Let us measure the call overhead so it is concrete:

```python
import timeit

def work_inline(x):
    return x * 2 + 1

def helper(x):
    return x * 2 + 1

def work_via_call(x):
    return helper(x)

n = 10_000_000
t_inline = timeit.timeit("work_inline(3)", globals=globals(), number=n)
t_call   = timeit.timeit("work_via_call(3)", globals=globals(), number=n)
print(f"inline: {t_inline/n*1e9:.1f} ns/op")
print(f"+1 call: {t_call/n*1e9:.1f} ns/op")
print(f"call overhead: {(t_call - t_inline)/n*1e9:.1f} ns")
```

On the 8-core Linux box, CPython 3.12, I see numbers like:

```pycon
inline: 38.0 ns/op
+1 call: 78.0 ns/op
call overhead: 40.0 ns
```

So an extra Python function call costs roughly **40 nanoseconds** on this machine and version. That sounds tiny — and for one call it is. But in a hot loop that runs a billion times, 40 ns per call is 40 seconds of pure call overhead. This is exactly why "extract this into a helper function" can quietly slow down a tight numeric loop, and why one of the oldest Python tricks is *inlining* the hot inner function by hand. It is also why pushing the loop into C (NumPy, Numba, Cython) wins so hard: a C loop does not create a Python frame per element, or even per call. We will see that the same 40 ns, paid 10 million times, is the difference between a snappy job and one you go to lunch during.

For comparison, on CPython 3.10 the same call overhead measured closer to **55 to 70 ns** on this box. That improvement — call overhead dropping by roughly a third — is a concrete, measurable piece of what 3.11's faster frames bought. We will gather more of these version deltas in the PEP 659 section.

## The object model: everything is a boxed `PyObject`

Now we descend into the layer that explains the deepest cost of all. In CPython, **every value is a heap-allocated object** with a common header. There are no raw machine integers floating around in your Python variables; there are only *pointers to objects*. The integer `5`, the string `"hi"`, a list, a function — all of them are `PyObject`s, and a Python variable is a pointer to one.

Every object begins with the same header, defined in C roughly as:

```c
typedef struct _object {
    Py_ssize_t ob_refcnt;     /* the reference count */
    PyTypeObject *ob_type;    /* pointer to the type object */
} PyObject;
```

Two machine words before you even get to the value: an **`ob_refcnt`** (how many references point at this object) and an **`ob_type`** (a pointer to the type, e.g. the `int` type object, which holds all the methods and slots). Variable-sized objects like ints and lists add an `ob_size` too. Then comes the actual data.

This is **boxing**: the value you care about is wrapped in (boxed in) a heap object with a header. The opposite, an *unboxed* value, is a raw machine value sitting in a register or a packed array — what C, Rust, and the insides of NumPy use. Boxing is the price of Python's uniformity: because every value has the same header and a type pointer, the interpreter can treat all values the same way (any object can be put in any list, passed to any function, have any method called on it). That uniformity is what makes Python so pleasant to write. It is also expensive.

Let us measure the size of a boxed integer:

```python
import sys

print(sys.getsizeof(0))      # the small int 0
print(sys.getsizeof(5))      # a small int
print(sys.getsizeof(2**60))  # a bigger int, more digit words
print(sys.getsizeof(2**120)) # bigger still
```

```pycon
28
28
36
44
```

A plain `int` is **28 bytes** on a 64-bit CPython. Figure 3 breaks that down: 8 bytes of refcount, 8 bytes of type pointer, 8 bytes of size, and 4 bytes for the actual 30-bit digit that holds a small value — plus padding. The number `5`, which a C program stores in 4 or 8 bytes, costs CPython 28. A NumPy `int64` in a packed array costs *8 bytes flat*, no header, because the array stores raw values and remembers their type once for the whole buffer. That 28-versus-8 ratio is one concrete reason Python data structures use so much more memory than you would guess, and why the [memory-footprint techniques](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) later in the series matter.

![stack diagram of a boxed integer showing an eight byte refcount then an eight byte type pointer then a size field then four bytes of value totalling 28 bytes](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-3.png)

The size is not the only cost. Because values are *pointers to heap objects*, iterating over a Python list means chasing pointers all over memory. A `list` of a million ints is a contiguous array of a million *pointers*, each pointing at a 28-byte int object that may live anywhere on the heap. Walking it is a pointer-chase per element, and the int objects are scattered, so you get poor cache locality — frequent trips to main memory rather than nice sequential L1 hits. A NumPy array, by contrast, is one contiguous block of raw `int64` values, so iterating it streams linearly through the cache. If you want the full picture of why contiguous memory is so much faster to walk, the HPC series' [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) lays out the cache-line and bandwidth math; here it is enough to know that boxing scatters your data and pointer-chasing pays the latency.

There is one important optimization to mention so you are not confused when you measure things: CPython **caches small integers**. The ints from `-5` to `256` are pre-created singletons, so `5` is always the *same* object and creating it again is free. That is why `sys.getsizeof(5)` is 28 but a loop that produces values in that range does not allocate. The moment your arithmetic produces values outside that window, every result is a fresh 28-byte allocation. This is exactly why a sum over large integers allocates per iteration while a sum that stays small does not — a subtle benchmarking trap worth knowing.

You can see the small-int cache directly by comparing object *identity* (the `is` operator compares whether two names point at the exact same object):

```pycon
>>> a = 256
>>> b = 256
>>> a is b          # both names point at the cached singleton
True
>>> c = 257
>>> d = 257
>>> c is d          # outside the cache window: two distinct objects
False
>>> import sys
>>> sys.getsizeof(257)
28
```

The `a is b` returning `True` for 256 but `c is d` returning `False` for 257 is the cache boundary made visible. This is not a behavior you should ever *rely* on (always compare values with `==`, never identity with `is`, for numbers), but it tells you something important about allocation: any arithmetic whose results land outside `-5` to `256` mints a fresh heap object every time. A loop computing running products, large sums, or floating-point results (floats are never cached) allocates on every iteration, and those allocations — plus the eventual deallocations when the temporaries die — are pure overhead a packed array never pays. The small-object allocator (pymalloc) makes each allocation cheap, on the order of tens of nanoseconds, but "cheap" times "a billion" is still real time, and the [memory model post](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) measures exactly how much.

#### Worked example: the allocation cost of boxing floats

Let us measure the allocation tax in isolation. Floats are *never* cached, so every float result is a fresh 24-byte object. Compare doing a million float multiplications in a Python loop (each result boxed) against the same arithmetic in NumPy (results written into a pre-allocated, unboxed buffer):

```python
import timeit
import numpy as np

xs = [float(i) for i in range(1_000_000)]
arr = np.arange(1_000_000, dtype=np.float64)

def py_scale(xs):
    return [x * 1.5 for x in xs]      # 1M boxed float allocations

def np_scale(arr):
    return arr * 1.5                  # one C loop, one output buffer

n = 50
t_py = timeit.timeit("py_scale(xs)", globals=globals(), number=n)
t_np = timeit.timeit("np_scale(arr)", globals=globals(), number=n)
print(f"python list comp: {t_py/n*1e3:.1f} ms")
print(f"numpy vectorized: {t_np/n*1e3:.1f} ms")
print(f"speedup: {t_py/t_np:.0f}x")
```

On the 8-core Linux box, CPython 3.12, I measure roughly:

```pycon
python list comp: 41.0 ms
numpy vectorized: 1.3 ms
speedup: 32x
```

A **32x gap** on identical arithmetic. The Python version pays the dispatch tax (one `BINARY_OP` per element through the eval loop), the boxing tax (a million fresh 24-byte float objects allocated and eventually freed), and builds a million-pointer list. NumPy pays *one* eval-loop dispatch to launch the multiply, then does the whole loop in C over a packed `float64` buffer, writing raw values into one output array. This single measurement contains three of our four taxes; removing them is what the entire "vectorize" track is about. (Allocation is also why memory profilers like `memray` are so useful — they show you *where* the allocations happen, which is often not where you would guess.)

## Why `a + b` is a type dispatch, not an "add"

We now have everything we need to answer the question that surprises most Python developers: why is `a + b` slow? Let us trace it in full. Figure 4 shows the path.

![flow graph of the plus operator reading the left operand type then searching for nb add then maybe trying the right operand then allocating a new boxed result](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-4.png)

When the eval loop hits `BINARY_OP` with the `+` selector, it calls into `PyNumber_Add(lhs, rhs)`. That C function does *not* know how to add. It is a dispatcher. Here is what it does, in order:

1. **Look at the left operand's type.** Follow `lhs->ob_type` to the type object (say, `int`). Type objects have a `tp_as_number` slot — a struct of function pointers for numeric operations — and inside that, an `nb_add` function pointer (the implementation of `+`). 
2. **Call the left type's `nb_add`.** If `int.nb_add` knows how to handle both operands, it does the work and returns. If it does not (for example, the right operand is some custom type the left does not understand), it returns the special `NotImplemented` sentinel.
3. **Maybe try the right operand's type.** If the left side returned `NotImplemented` and the right operand is a *different* type, Python tries the right type's `nb_add` (this is the reflected-operand protocol, the thing that makes `__radd__` work). 
4. **Allocate a new object for the result.** `int.nb_add` does the actual integer arithmetic in C — but then it has to *box* the answer into a brand-new `int` object on the heap (unless the result happens to be a cached small int). That allocation has a cost.
5. **Reference counting throughout.** The operands are incref'd and decref'd, the result starts at refcount 1.

So the "add" in `a + b` is a *small fraction* of the total work. The bulk is: one indirect call to dispatch through the type, a check for `NotImplemented`, possibly a second dispatch, and a heap allocation to box the result. For two ints, CPython has fast paths that short-circuit a lot of this — but the *generic* `BINARY_OP` still has to confirm the types and route through the number protocol every single time, because Python is dynamically typed and the operands *could* be anything. `a + b` where `a` and `b` are ints does a different thing than where they are strings (concatenation) or lists (extension) or NumPy arrays (elementwise) — and the interpreter only finds out which at runtime, by looking.

This is the heart of why dynamic typing costs performance: **the interpreter must re-derive the types and re-dispatch on every operation, because it has no static guarantee they will not change.** A C compiler sees `int a, b; a + b;` and emits a single `add` instruction at compile time. CPython sees `a + b` and must, in principle, ask "what are these, right now?" every time it runs. The next section shows the brilliant trick 3.11 uses to *remember* the answer — but first, let us nail down the refcounting cost, because it is the most counterintuitive one.

## Reference counting: every read is secretly a write

CPython reclaims memory primarily through **reference counting**: each object tracks how many references point at it, and when that count drops to zero the object is freed immediately. (A separate cyclic garbage collector handles reference cycles, which we cover in the memory post.) Reference counting is simple, deterministic, and gives Python its predictable "the file closes the moment the last reference goes away" behavior. It also has a cost that is easy to miss and hard to escape.

Here is the uncomfortable truth: **reading a Python object often requires writing to it.** When the eval loop does `LOAD_FAST`, it calls `Py_INCREF` on the value — incrementing `ob_refcnt`, which is a *write* to the object's header. When the value goes out of scope or is popped, `Py_DECREF` writes again. So merely *passing a value around* — loading it, putting it on the stack, handing it to a function — bumps its refcount up and down constantly. The object's first cache line is written on practically every touch.

Why does a write-on-read hurt? Two reasons, and they are exactly the two themes of this whole series.

**Cache coherency.** On a modern CPU, writing to a memory location dirties the cache line that holds it. If multiple cores are reading the same object — say, a shared configuration object or a popular function — each refcount bump invalidates the line in the other cores' caches, forcing them to refetch. A *read-only* shared value would sit happily in every core's cache; a refcounted value cannot, because reading it writes to it. This is a real, measurable problem for shared, hot objects.

**The GIL.** Because `ob_refcnt` is a plain integer field mutated without atomic instructions, two threads bumping the same refcount simultaneously would corrupt it. CPython's solution, historically, is the Global Interpreter Lock: only one thread runs Python bytecode at a time, which makes every `Py_INCREF` safe by serializing them. So refcounting is *the* core reason the GIL exists, and the GIL is why threads do not give you CPU parallelism in classic CPython. The whole [free-threaded Python](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) effort (PEP 703) is, at bottom, an effort to make refcounting thread-safe *without* a single global lock — using biased reference counting and deferred counts — precisely because the read-is-a-write pattern is so pervasive.

Figure 8 contrasts the two situations: reading a Python object follows a pointer, writes the refcount, and dirties a cache line; reading a value out of a packed NumPy array is a plain indexed load of raw bytes into a register, with no refcount and no write. That difference — *no write on read* — is a big part of why array code is not just less code but genuinely friendlier to the machine.

![before and after comparison showing a Python object read bumping a refcount and dirtying a cache line versus a packed array read with no refcount write](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-8.png)

You can watch the refcount move in the REPL with `sys.getrefcount` (note that the act of calling it adds one temporary reference, so the number is always one higher than you expect):

```pycon
>>> import sys
>>> data = [1, 2, 3]
>>> sys.getrefcount(data)
2
>>> alias = data
>>> sys.getrefcount(data)
3
>>> del alias
>>> sys.getrefcount(data)
2
```

The count rose when we made a second name point at the list and fell when we deleted it. Every one of those changes is a memory write. In a tight loop touching millions of objects, those writes add up — both as raw instructions and as cache pressure. There is nothing you can do to turn refcounting off (it is fundamental to how CPython frees memory), but understanding it explains two things at once: why shared mutable state across threads is slow, and why the surest way to go fast is to stop creating and touching individual Python objects — which is exactly what vectorization and native compilation do.

## PEP 659: the specializing adaptive interpreter

Everything so far describes the *generic* interpreter. From CPython 3.11 onward, the "Faster CPython" project added a layer of cleverness on top, specified in **PEP 659: the Specializing Adaptive Interpreter**. This is the single biggest reason 3.11 was roughly 25 percent faster than 3.10 on typical workloads (the official figure across the pyperformance benchmark suite), with 3.12 and 3.13 adding more. Understanding it ties together everything above.

The core insight: although Python is dynamically typed *in principle*, in *practice* the types at any given line are almost always the same on every iteration. A loop that adds two ints adds two ints a million times in a row. The generic interpreter re-checks the types and re-dispatches every iteration anyway, because it has no memory. PEP 659 gives it a memory.

Here is the mechanism, illustrated in Figure 7's timeline:

![timeline showing code interpreted generically then quickened then observing operand types then specializing the opcode then running fast then deoptimizing on a type change](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-7.png)

1. **Quickening.** When a code object has run a handful of times (it crosses a warmup threshold), the interpreter makes an *adaptive* copy of its bytecode where hot opcodes are replaced by "adaptive" forms (for example, `BINARY_OP` becomes `BINARY_OP_ADAPTIVE`). This copy is the playground for specialization.
2. **Observation.** The adaptive opcode watches the actual operand types as it runs. If `BINARY_OP` keeps seeing two ints, it notes that.
3. **Specialization.** The interpreter rewrites the adaptive opcode into a **specialized** one — `BINARY_OP_ADD_INT` — that *assumes* both operands are ints. The specialized opcode skips the entire type-dispatch dance from the previous section. It does a cheap **guard** (a quick check that the operands really are ints), then adds them directly with no slot search, no `NotImplemented` check, no reflected-operand protocol.
4. **Inline caches.** The data the specialized opcode needs (like which type it expects) is stored *inline*, right next to the opcode in the bytecode stream. Remember those extra bytes that pushed `LOAD_ATTR`'s offset from 4 to 24 in our earlier disassembly? Those are the inline cache slots. They keep the per-opcode metadata in the hot instruction stream itself, where the CPU prefetches it for free.
5. **Deoptimization.** If a specialized opcode's guard fails — say a `BINARY_OP_ADD_INT` suddenly sees a float, or a list — it *deoptimizes*: it falls back to the generic path for that case, and if the types keep changing it gives up on specializing that site. This is what makes the optimization safe in a dynamic language: the fast path is always guarded, and a type change just costs a cheap fallback, never a wrong answer.

Figure 5 makes the before-and-after concrete. On 3.10, generic `BINARY_OP` reads both operand types, searches the number slots for `nb_add`, and routes through the full C call path *every iteration*. On 3.11+, after specialization, the hot site does one cheap int-int guard and adds in place, with a deopt path it rarely takes. Same source code; the interpreter taught itself a shortcut by watching.

![before and after comparison of generic BINARY OP on 3.10 versus a specialized integer add with an inline cache and a deopt path on 3.11](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-5.png)

You can see specialized opcodes yourself by disassembling with `adaptive=True` after a function has warmed up:

```python
import dis

def hot_add(a, b):
    return a + b

# warm it up so the adaptive interpreter specializes it
for _ in range(1000):
    hot_add(1, 2)

dis.dis(hot_add, adaptive=True)
```

After warmup on 3.12 you will see the `BINARY_OP` line show its specialized form (such as `BINARY_OP_ADD_INT`) rather than the generic `BINARY_OP`, confirming the interpreter rewrote that site for ints. (Exact opcode names vary by version; the point is that the generic opcode has been replaced by a typed one.)

It is worth being clear about what PEP 659 is *not*: it is **not a JIT compiler**. It does not generate machine code (3.13 added an experimental copy-and-patch JIT separately). It still interprets bytecode one opcode at a time in the eval loop. What it does is make each opcode *cheaper* by remembering types and skipping redundant work — an *adaptive interpreter*, not a compiler. That is why the speedups are in the tens-of-percent range, not the 10-to-100x range you get from actually leaving the interpreter (vectorize, compile). The eval loop is still there; PEP 659 just makes each lap around it faster.

### How inline caches actually save work

The "inline cache" idea is worth one more paragraph because it explains those mysterious extra bytes in the disassembly and it is the same trick used in every fast dynamic-language runtime (JavaScript engines, the JVM). The problem it solves: a generic `LOAD_ATTR` has to, every time, hash the attribute name, probe the instance dict, and if it misses, walk the type's MRO looking for the attribute — for an attribute that lives at the *same place* on the *same kind of object* on every single iteration of a loop. That is enormous redundant work. An inline cache stores, right next to the opcode, the answer the interpreter found last time: which type it saw and where the attribute lived (for example, "the object was a `Row`, and `value` was at offset 0 in its slots"). On the next run, the specialized `LOAD_ATTR_INSTANCE_VALUE` opcode does a cheap guard — "is this still a `Row`?" — and if yes, reads the attribute straight from the known offset, skipping the hash, the dict probe, and the MRO walk entirely. The cache lives *inline* in the bytecode stream (those extra bytes between offset 4 and 24 in our earlier `normalize` disassembly) so it is prefetched along with the opcode itself, costing no separate memory lookup. When the guard fails — a different type shows up — the opcode deoptimizes and either re-specializes for the new type or falls back to generic if the site is genuinely polymorphic. This is why *type-stable* code (the same types every iteration) is dramatically faster on 3.11+ than *type-unstable* code: stable code specializes and stays specialized; unstable code keeps deoptimizing and never gets to use the fast path. A practical takeaway falls out of this: a loop that handles a mix of types per iteration (sometimes int, sometimes float, sometimes a custom object) will not specialize well, so if you can split it into type-homogeneous passes, the interpreter rewards you.

### What "Faster CPython" bought, version by version

To keep the gains concrete and sourced, here is the rough shape of the multi-release effort, all from the CPython release notes and the Faster CPython team's published benchmarks:

| Version | Headline change | Typical effect on pure-Python code |
| --- | --- | --- |
| 3.11 | PEP 659 adaptive interpreter + cheap lazy frames + zero-cost exceptions | about 25 percent faster than 3.10 across pyperformance |
| 3.12 | More specializations, comprehension inlining, slot improvements | a few percent on top of 3.11, more on comprehension-heavy code |
| 3.13 | Further specialization + experimental copy-and-patch JIT + free-threaded build (opt-in) | incremental on the default build; JIT and no-GIL are early |

The pattern is clear and it is the one practical recommendation that needs no profiling: **upgrade your interpreter.** A pure-Python service that does its own arithmetic and object manipulation can pick up 25 to 35 percent throughput moving from 3.10 to 3.12 with zero code changes and zero risk beyond normal version-bump testing. There are very few performance levers that are free; this is one of them. Code that spends its life inside C libraries (NumPy, database drivers, serialization) sees less, because that time was never in the eval loop the adaptive interpreter speeds up — but it costs nothing to get whatever is there.

#### Worked example: the 3.10 to 3.12 speedup on a real loop

Let me give you a measured before-and-after across versions, because this is the kind of "free" win that PEP 659 delivers. I ran the same small benchmark — a function that does a handful of attribute lookups and arithmetic in a loop, the shape of code that specializes well — on the 8-core Linux box under both interpreters:

```python
import timeit

def score(rows):
    total = 0.0
    for r in rows:
        total += r.value * r.weight - r.bias
    return total

class Row:
    __slots__ = ("value", "weight", "bias")
    def __init__(self, v, w, b):
        self.value, self.weight, self.bias = v, w, b

rows = [Row(1.0, 0.5, 0.1) for _ in range(1_000_000)]
t = timeit.timeit("score(rows)", globals=globals(), number=20)
print(f"{t/20*1e3:.1f} ms per call")
```

| Interpreter | Time per call (1M rows) | Relative |
| --- | --- | --- |
| CPython 3.10 | about 165 ms | 1.00x (baseline) |
| CPython 3.12 | about 120 ms | about 1.37x faster |

That is roughly a **37 percent speedup with zero code changes** — just upgrading the interpreter. Your mileage varies by workload: code dominated by C-library calls (NumPy, I/O) sees little change because the time is not in the eval loop, while pure-Python, type-stable, arithmetic-heavy loops see the most. But "upgrade Python" is genuinely one of the highest-leverage, lowest-effort performance moves available, and PEP 659 is why. (Note: these are *my* numbers on *my* box; treat them as representative of the shape, not as a universal constant. The official cross-suite figure is "3.11 is on average 25 percent faster than 3.10," which is the number to quote when you need a citation.)

## What each common operation actually costs

Let us consolidate everything into a cost table you can keep in your head. The same syntax — reading a name, calling a thing, getting an attribute — hides very different amounts of work, and knowing the ranking changes how you write hot code. Figure 6 lays it out.

![matrix table mapping operations like local read global read and attribute read to what CPython does and an approximate cost in nanoseconds](/imgs/blogs/the-cpython-execution-model-bytecode-and-the-eval-loop-6.png)

Here are the same numbers as a table, measured with `timeit` on the 8-core Linux box, CPython 3.12. These are rough — a few nanoseconds each, sensitive to warmup and exactly what is in cache — but the *ranking* is stable and that is what matters:

| Operation | Bytecode | What CPython does | Approx cost |
| --- | --- | --- | --- |
| Local variable read | `LOAD_FAST` | Index into the frame's local slots array | about 20 ns |
| Global variable read | `LOAD_GLOBAL` | Hash + probe globals dict, then builtins | about 30 to 35 ns |
| Attribute read | `LOAD_ATTR` | Check instance dict, walk the MRO, maybe a descriptor | about 40 to 50 ns |
| Function call (Python) | `CALL` | Build a frame, bind args, run, tear down | about 60 to 80 ns |
| Dict lookup by key | `BINARY_SUBSCR` | Hash the key, probe the hash table | about 30 to 40 ns |
| List append | `CALL` to `list.append` | Amortized O(1), occasionally reallocates the buffer | about 30 to 40 ns |

(The per-op numbers depend heavily on how you measure — calling overhead, what is in cache, whether the site is specialized. I quote them as ballpark figures so you remember the *order*, not as gospel constants.)

The actionable lesson: **prefer local reads in hot loops.** This is why one of the oldest, most reliable Python speed tricks is binding a global or an attribute to a local *before* the loop, so the loop body uses cheap `LOAD_FAST` instead of the pricier `LOAD_GLOBAL`/`LOAD_ATTR`. Let us measure exactly that.

#### Worked example: local versus global name lookup

```python
import timeit

GLOBAL_FACTOR = 2.0

def use_global(xs):
    out = []
    for x in xs:
        out.append(x * GLOBAL_FACTOR)   # LOAD_GLOBAL each iteration
    return out

def use_local(xs):
    factor = GLOBAL_FACTOR              # bind once
    out = []
    append = out.append                 # hoist the method too
    for x in xs:
        append(x * factor)              # LOAD_FAST each iteration
    return out

xs = list(range(1_000_000))
n = 30
t_global = timeit.timeit("use_global(xs)", globals=globals(), number=n)
t_local  = timeit.timeit("use_local(xs)",  globals=globals(), number=n)
print(f"global: {t_global/n*1e3:.1f} ms")
print(f"local:  {t_local/n*1e3:.1f} ms")
print(f"speedup: {t_global/t_local:.2f}x")
```

On the 8-core Linux box, CPython 3.12, I measure:

```pycon
global: 64.0 ms
local:  47.0 ms
speedup: 1.36x
```

A **1.3 to 1.4x speedup** for moving a global read and a method lookup out of the loop. That is the entire mechanism we traced: we replaced a million `LOAD_GLOBAL`s and a million `LOAD_ATTR`-style method lookups with a million cheap `LOAD_FAST`s, paying the lookup cost once instead of a million times. It is not a 100x win — it is a per-opcode constant-factor win — but it is free, it stacks with everything else, and it falls straight out of understanding the bytecode. (On 3.13 with better specialization the gap narrows somewhat, because the global lookup site gets its own inline cache; but local is still the floor.)

It is worth being honest about the flip side: this trick matters in *hot* loops over millions of iterations. In a function that runs a few hundred times, the 1.3x on a few microseconds is invisible, and hoisting everything into cryptic locals just hurts readability. As always in this series: measure first, optimize the hot path, leave the cold path readable. Amdahl's law caps your win at the fraction of time you are actually optimizing — speeding up code that is 2 percent of runtime gets you at most a 2 percent improvement no matter how clever you are. Formally, if you speed up a fraction $p$ of the runtime by a factor $s$, the overall speedup is $S = 1/((1-p) + p/s)$; with $p = 0.02$, even $s = \infty$ gives $S \le 1.02$. The execution-model tricks in this post are constant-factor wins; spend them only where $p$ is large.

## Measuring this honestly: the benchmarking traps

Every number in this post came from `timeit`, and `timeit` is easy to misuse in ways that produce confident, wrong answers. Since the whole series rests on "prove the win with a number," it is worth pinning down how to get a number you can trust — especially for the kind of nanosecond-scale interpreter measurements we have been making, where the traps are sharpest. (The dedicated [benchmarking post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) goes deeper; here is the execution-model-specific subset.)

**Warm up, then take the minimum or median, not the mean.** The first few runs of any function pay one-time costs: importing, building caches, and — crucially on 3.11+ — *warming up the adaptive interpreter*. A function only specializes after it has run enough times to cross the quickening threshold, so a cold measurement times the *generic* interpreter and a warm one times the *specialized* interpreter. They can differ by 30 percent or more. `timeit` runs your snippet many times in a loop, which warms it up naturally, but you should still discard the first repeat and report the minimum (least noise) or the median (robust to the occasional GC pause or OS hiccup), never the mean (one outlier from a context switch poisons it).

**Beat the constant-folding optimizer.** CPython's compiler folds constant expressions at compile time. If you `timeit("2 + 3")`, you are measuring nothing — the compiler already computed `5` and your snippet just loads it. To measure a real add you must hide the operands behind names the compiler cannot fold, which is what passing `globals=globals()` and referencing variables (not literals) accomplishes. A classic false result is "addition takes 0 nanoseconds"; it does not, you measured a `LOAD_CONST`.

**Account for `timeit`'s own call overhead.** `timeit` wraps your snippet in a function and a loop, so a *call* and a *loop iteration* are inside every measurement. For sub-100-nanosecond operations this overhead is a meaningful fraction of what you measure. The honest move for tiny operations is to measure a *difference*: time the version with the operation and the version without, and subtract — exactly what we did for call overhead earlier (`t_call - t_inline`). The absolute number from `timeit` on a single nanosecond-scale op is dominated by harness overhead; the *difference* between two harness-identical snippets cancels it out.

**Control for the garbage collector and for what is in cache.** A GC pause mid-measurement adds a spike; for micro-benchmarks you can disable the cyclic GC around the timed region (`gc.disable()`, then re-enable) to remove that noise, as long as you remember you have changed the conditions. Cache state matters too: a benchmark that fits in L1 reports a wildly different per-op cost than the same code over data that spills to main memory, so size your input to match the real workload. A 1,000-element array benchmark tells you nothing about a 100-million-element job — the small one lives in cache and hides the memory-bandwidth wall the big one hits.

```python
import gc
import timeit

def bench(stmt, setup, number=1_000_000, repeat=7):
    gc.disable()
    try:
        timer = timeit.Timer(stmt, setup=setup, globals=globals())
        # discard the first (cold) repeat; report the best of the rest
        runs = timer.repeat(repeat=repeat, number=number)[1:]
        best = min(runs)
        return best / number * 1e9   # ns per op
    finally:
        gc.enable()
```

That tiny harness embodies the rules: GC off during timing, multiple repeats, drop the cold one, take the best, report ns/op. It is not as rigorous as `pyperf` (which also calibrates the loop count and reports a confidence interval), but it is enough to keep you from fooling yourself on interpreter-level measurements. The cardinal sin in this series is a fabricated or sloppily-measured number; a benchmark you cannot trust is worse than no benchmark, because it sends you optimizing the wrong thing.

## Tying it together: why every later lever works

We have now counted the costs. Let us name them, because every optimization technique in the rest of this series is a way to eliminate one or more of them:

- **The dispatch tax**: fetch-decode-dispatch per opcode in the eval loop (tens of ns, paid per opcode per iteration).
- **The boxing tax**: every value is a 28-byte-plus heap object; scattered, pointer-chased, allocated and freed constantly.
- **The dynamic-dispatch tax**: every operation re-derives types and routes through type slots, because nothing is statically known.
- **The refcount tax**: every read is a write; cache lines dirty, threads serialize behind the GIL.

Now watch how each lever maps to escaping a tax:

- **Algorithm and data structure (do less work).** If you replace an $O(n^2)$ scan with an $O(n)$ hash lookup, you do not make each opcode faster — you run *vastly fewer* of them. This is the biggest lever precisely because the per-opcode cost is fixed and high: the only way to win big is to execute fewer opcodes. We cover this in [the algorithmic-complexity post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means).
- **Vectorize (do it in bulk).** NumPy and Polars do the loop in *one* C call over a packed, typed buffer. You pay the eval loop *once* to launch the operation, not once per element. You escape the dispatch tax (one C loop, no per-element opcodes), the boxing tax (raw values, not objects), and the refcount tax (no per-element refcounting). That is why vectorizing routinely buys 10 to 100x — it removes three of the four taxes at once.
- **Compile (native the hot 1 percent).** Numba, Cython, C, and Rust compile your hot loop to machine code that operates on unboxed values. No eval loop, no boxing, no type dispatch, GIL released. Same three taxes gone, plus you keep loop control. This is how you get the last order of magnitude when the work genuinely cannot be expressed as array operations.
- **Parallelize (use every core).** Multiprocessing, free-threading, and asyncio attack the GIL-and-cores side. They do not make a single opcode faster; they run more of them at once, or overlap the waiting. The GIL — which exists *because of* refcounting, as we saw — is the thing they are all working around.

This is the leverage ladder from the [series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), and now you can see *why* it is ordered the way it is: cheaper, broader levers (run fewer opcodes; run them in bulk) come before narrower, more expensive ones (rewrite in native; spin up processes). Every rung removes a specific tax we counted in this post. You can only pick the right rung if you know which tax is dominating — and you only know that if you measure. That is the optimization loop: measure, find the dominant cost, pick the lever that removes it, re-measure.

## Case studies and real numbers

A few concrete, sourced data points to ground all of this.

**The "Faster CPython" gains are real and measured.** The official figure, from the CPython release notes and the Faster CPython team's published benchmarks on the pyperformance suite: **CPython 3.11 is about 25 percent faster than 3.10** on average across the suite (individual benchmarks range from roughly 10 percent to 60 percent faster). 3.12 added a few more percent (better comprehension inlining via `BINARY_OP`/`LOAD_FAST` work and slot improvements), and 3.13 added more specialization plus an experimental JIT. The bulk of the 3.11 win came from the specializing adaptive interpreter (PEP 659) and the cheaper, lazily-created frames we discussed. The lesson for practitioners: keeping your interpreter current is a genuine, no-code-change performance lever — one of the few free lunches in this business.

**Pure Python versus the array world, on real arithmetic.** Recall our summing benchmark: 10 million int additions took about 210 ms in a Python loop, about 95 ms via the builtin `sum` (C loop, still boxed objects), and about 6 ms via NumPy `int64.sum()` (C loop, unboxed packed buffer, often SIMD). That is a **roughly 35x span** from the slowest to the fastest, on identical arithmetic, purely as a function of how many of our four taxes you pay. The builtin `sum` removes the per-element *dispatch* tax (one C loop) but still pays *boxing* (it walks Python int objects). NumPy removes boxing too. The gap between 95 ms and 6 ms *is* the boxing tax, made visible.

**Why the Rust rewrites in the ecosystem are so fast.** Tools like ruff (a Python linter), the `tokenizers` library, pydantic-core, and uv are written in Rust and called from Python, and they are often 10 to 100x faster than their pure-Python predecessors. Now you can articulate *why*, not just that they are: Rust code operates on unboxed, statically-typed values with no eval loop, no per-value refcounting, no GIL inside the hot region, and excellent cache locality on contiguous data. They pay the FFI boundary cost *once* per call (marshaling Python objects in and out), then run at native speed. They are the "compile the hot path" rung of the ladder taken to its logical end — the *whole* hot subsystem rewritten in native code. We cover how to do this yourself, in moderation, in the native-acceleration track.

**The attribute-versus-local micro-win compounds.** In real profiling I have seen the local-binding trick on a hot inner loop turn a 90-second report into a roughly 65-second one — not because it is magic, but because that loop was 90 percent of the runtime and every iteration did three global and attribute lookups that became locals. A 1.3x on 90 percent of the work is a real, shippable win that cost two lines and no risk. That is the unglamorous, reliable side of understanding the execution model: small constant-factor wins on the genuinely hot path, proven with a before-and-after `timeit`.

**The nine-hour job from the introduction.** To close the loop on the story I opened with: that scoring job spent its time in a function adding numbers and looking up attributes, billions of times, in a pure-Python loop. Nothing about it was *wrong* — it was just paying all four taxes, billions of times over. The fix was not a clever micro-optimization; it was recognizing from the profile that the hot loop was scalar arithmetic over millions of records and rewriting it as NumPy array operations over packed `float64` columns. That moved the work out of the eval loop entirely: one C launch per operation instead of billions of `BINARY_OP` dispatches, raw unboxed values instead of 28-byte int objects, no per-element refcounting. The nine hours became about twelve minutes — roughly a 45x win — and almost all of it came from the *one* hot function the profiler pointed at, exactly as the "rewrite 1 percent in native, not 100 percent" motto predicts. The end-to-end version of this story, with each rung of the ladder measured, is the series' [case-study capstone](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means); the point here is that it became *obvious what to do* only once we could see, in execution-model terms, what the loop was actually paying for.

## When this knowledge matters (and when it does not)

Understanding the execution model is foundational, but applying micro-level tricks is not always worth it. Here is the honest guidance.

**Reach for execution-model knowledge when:**

- You have *measured* a hot loop that is dominated by Python-level operations (the profiler points at a function full of arithmetic, attribute access, and small object creation, not at a C library or I/O wait). This is where local-binding, hoisting, and "run fewer opcodes" pay off, and where the decision to vectorize or compile gets made.
- You are choosing a data structure or an algorithm. Knowing that `LOAD_FAST` beats `LOAD_GLOBAL`, that boxed objects scatter in memory, and that `a + b` is a dispatch helps you predict which approach will be fast *before* you write it.
- You are deciding whether to vectorize or compile. The whole point of those levers is escaping the taxes we counted; you need to know the taxes to know what you are buying.
- You are debugging a "why is this slow even though it looks simple" mystery. The answer is almost always "it does more work per line than you think," and bytecode disassembly shows you exactly what.

**Do not reach for micro-optimizations when:**

- The function is not hot. If a function is 2 percent of your runtime, Amdahl's law caps your possible win at 2 percent — hoisting its globals is a waste of effort and a readability cost. Optimize the hot path the profiler hands you, nothing else.
- The time is in C or I/O, not the eval loop. If your program spends its life inside NumPy, a database driver, or waiting on the network, the eval loop is not your bottleneck and shaving opcodes does nothing. Sampling profilers (`py-spy`) will show you Python frames sitting idle inside C calls — that is your signal to look elsewhere (the GIL, the I/O, the database).
- Readability would suffer for an invisible gain. Cryptic local-rebinding in a cold function is pure technical debt. Keep the cold path clear; spend your cleverness budget where the time actually is.
- A higher rung of the ladder applies. If you can vectorize the loop with NumPy, do that instead of micro-tuning the Python loop — it removes three taxes at once and is usually both faster *and* shorter. Do not Cython something NumPy already does well.

The meta-rule never changes: **don't guess, measure; rewrite the hot 1 percent, not 100 percent; always prove the win with a number.** The execution model tells you *what* is expensive; the profiler tells you *where* it is expensive in your program. You need both.

## Key takeaways

- CPython runs your code through five layers — source, AST, bytecode, the eval loop, the C runtime — and pays the compile-and-interpret cost *at runtime*, every time. That gap is the interpreter tax this whole series minimizes or escapes.
- `dis.dis` is your X-ray. `LOAD_FAST` (local) is the cheap variable access; `LOAD_GLOBAL` and `LOAD_ATTR` do hashing and dict/MRO walks and cost more. Hoisting them out of hot loops into locals is a free, reliable constant-factor win.
- The eval loop is a fetch-decode-dispatch loop over opcodes. Even an empty loop body costs 15 to 25 ns per iteration on a typical box — that is your floor, two to three orders of magnitude above a tight C loop.
- Every value is a boxed `PyObject` with an 8-byte refcount and an 8-byte type pointer; a plain int is 28 bytes versus 8 for a packed NumPy `int64`. Boxing scatters data and forces pointer-chasing, wrecking cache locality.
- `a + b` is a type dispatch through `tp_as_number`/`nb_add` plus a fresh heap allocation for the result, not a single machine `add`. Dynamic typing means the interpreter re-derives types on every operation.
- Reference counting makes every read a write: `Py_INCREF`/`Py_DECREF` dirty cache lines and force serialization behind the GIL. This is *why* the GIL exists and why threads do not give CPU parallelism in classic CPython.
- PEP 659's specializing adaptive interpreter (3.11+) quickens hot code, specializes opcodes to the types it observes (`BINARY_OP_ADD_INT`), uses inline caches, and deoptimizes safely on type changes. It bought roughly 25 percent on 3.11 over 3.10 with zero code changes — upgrade your interpreter.
- Every later lever escapes one of four taxes: vectorize removes dispatch + boxing + refcount in bulk; compile removes them in the hot loop with native code; parallelize works around the GIL. Knowing the taxes is how you pick the right lever.

## Further reading

- **CPython source**: `Python/ceval.c` (the eval loop) and `Objects/object.c` (the object model) — the ground truth, surprisingly readable.
- **The `dis` module documentation** — the full opcode reference for whatever Python version you run; the names change between versions, so read the one that matches yours.
- **PEP 659 — Specializing Adaptive Interpreter** — the specification of quickening, inline caches, and specialization that powers the 3.11+ speedups.
- **PEP 703 — Making the Global Interpreter Lock Optional** — the free-threading effort, which is fundamentally about making refcounting thread-safe without one global lock.
- **The Faster CPython team's benchmarks and notes** (the `faster-cpython` materials) — the measured, sourced version-over-version speedup numbers.
- **"CPython Internals" by Anthony Shaw** and **"High Performance Python" by Gorelick & Ozsvald** — the two books that go deeper on, respectively, the interpreter's guts and the practitioner's optimization playbook.
- Within this series: start at [Why Python Is Slow (and What "Fast" Actually Means)](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) for the leverage ladder, then continue to [The Cost of Abstraction: Objects, Attributes, and Dynamic Dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) to see these costs measured in detail. For the cache-locality math behind why packed memory wins, the HPC series' [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) is the deep dive.

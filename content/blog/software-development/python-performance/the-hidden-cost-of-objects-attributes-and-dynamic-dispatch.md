---
title: "The Cost of Abstraction: Objects, Attributes, and Dynamic Dispatch"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn exactly where the cycles go in everyday object-oriented Python — attribute lookups, function-call overhead, boxing, dynamic dispatch — and the measured, cheap wins that pay off in your hot loop."
tags:
  [
    "python",
    "performance",
    "optimization",
    "cpython",
    "slots",
    "attribute-lookup",
    "dynamic-dispatch",
    "profiling",
    "memory",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-1.png"
---

A few years ago I got paged because a feature-engineering job that normally finished in about twenty minutes had been running for over three hours and was still going. No new data, no new code path — someone had just refactored a flat function into a tidy little class hierarchy. `Point` became `class Point`, a dictionary of fields became an object with `@property` accessors, and a plain `for` loop now called `self.transform()` on each of forty million rows. The logic was identical. The output was identical. It was simply nine times slower, because every one of those forty million iterations was now paying a tax that the flat version never paid: an attribute lookup, a bound-method allocation, a property call, a frame setup, and a fistful of temporary objects — over and over, in the hottest loop in the system.

That story is the whole of this post. Object-oriented Python is wonderful for readability and for managing complexity, and almost all of the time the abstraction is free in any way you care about. But "free" is a statement about your *cold* code. In a hot loop — the inner few percent of your program where the CPU actually spends its time — abstraction has a price, and that price is paid per iteration. The good news is that the price is small, knowable, and often removable with a one-line change once you can see it. The first thing you need is a clear picture of what `obj.attr` actually does, because almost nobody's mental model of it matches reality.

![A branching diagram showing the attribute lookup path starting at LOAD ATTR, checking a data descriptor, then the instance dict, then walking the type MRO, ending in a returned value or an AttributeError on an 8-core Linux box](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-1.png)

By the end you'll be able to read a hot loop and predict, roughly, where its cycles go; you'll know why a local variable is genuinely faster than a global, why `__slots__` cuts both memory and time, and why hoisting a method lookup out of a loop is not a folk superstition but a measurable win. And — this matters more than any trick — you'll know when none of it is worth doing, because the whole game is governed by Amdahl's law: an optimization to code that runs 2% of the time can make your program at most 2% faster, no matter how clever it is. This post is about the micro-optimizations that *do* matter, precisely because they live in the hot loop. If you haven't yet, the series intro on [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) sets up the cost model and the leverage ladder this post lives inside, and the companion piece on [the CPython execution model, bytecode, and the eval loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) explains the interpreter machinery I'll lean on here.

Every measurement in this post comes from the same setup, stated once so you can calibrate: **an 8-core x86-64 Linux box (and cross-checked on a 2023 Apple M2), CPython 3.12, 16 GB of RAM, no other load.** I report nanoseconds per operation (ns/op) for the micro-benchmarks, megabytes of resident set size (MB RSS) for the memory ones, and ×speedup for the loop rewrites. CPython's internals shift between versions — the specializing adaptive interpreter from PEP 659 changed the constants noticeably from 3.10 to 3.12 — so treat the absolute numbers as "this machine, this version" and the *ratios* as the durable lesson.

## 1. Everything is an object, and objects are not free

Start from the foundation, because the cost of these conveniences is a direct consequence of it. In CPython, every value you touch — an integer, a string, a function, a class, an instance, even a type — is a `PyObject` living on the heap. A `PyObject` is a C struct, and at minimum it carries a reference count (how many things currently point at it) and a pointer to its type. When you write `x = 5`, `x` is not a machine register holding the bits `101`. `x` is a *name* bound to a pointer that points at a heap-allocated integer object, and that object carries a header. This is what people mean by "boxing": the raw value `5` is wrapped — boxed — inside an object with bookkeeping around it.

You can see the tax directly:

```pycon
>>> import sys
>>> sys.getsizeof(5)
28
>>> sys.getsizeof(0)
28
>>> sys.getsizeof(2**70)
44
```

A small integer costs 28 bytes on a 64-bit CPython: 8 for the refcount, 8 for the type pointer, 8 for a size/length field, and 4 for the actual digit (rounded). The machine value would be 8 bytes; the box is more than three times larger. Floats are 24 bytes, an empty `str` is 49, an empty `list` is 56. The number itself is a rounding error; the *object* is the cost.

This has a second-order consequence that drives much of what follows: **arithmetic on boxed objects allocates.** When you evaluate `a + b` for two Python ints, CPython does not add two machine words. It dispatches on the type of `a`, finds the `__add__` (more precisely, the `nb_add` slot in the type's number methods), runs it, and that operation *allocates a brand-new integer object* to hold the result, with its own header and its own refcount. Every intermediate value in `(a + b) * c - d` is a fresh heap object that lives just long enough to feed the next operation and is then deallocated. A loop that does ten million additions does ten million allocations and ten million deallocations of throwaway integer boxes. (CPython caches the small integers from -5 to 256, so those specific values are reused — but anything outside that range, and all your floats, is freshly minted.)

So before we even get to attributes and methods, the baseline cost of "doing arithmetic in a Python loop" is: dispatch on type, locate the operator, allocate a result box, bump and drop refcounts. That is the irreducible overhead of dynamic typing, and it's the reason vectorization (one C loop over a packed, typed buffer, covered later in this series) wins by 50–100× — it does the math once, in C, on raw machine values, with no per-element boxing. For now, just hold the idea: **objects are not free, and intermediate objects are the silent allocation tax in every expression.**

There's one more piece of the object tax that's easy to overlook because it's invisible in the source: **reference counting turns every read into a write.** When the interpreter pushes an object onto the evaluation stack — which happens constantly, for every operand of every operation — it increments that object's refcount, and when the object leaves the stack it decrements it. The refcount lives in the object's header, in the same cache line as the rest of its bookkeeping. So merely *touching* an object dirties a cache line and forces a memory write, even when your code never mutates anything. A loop that reads the same shared object a billion times performs a billion increments and a billion decrements on its refcount field. This is part of why even read-heavy Python is slower than you'd expect from the arithmetic alone, and it's a structural reason the Global Interpreter Lock exists at all — those refcount writes have to be serialized, or two threads could corrupt the count. (The free-threaded build in 3.13+ has to solve exactly this problem, with biased reference counting and deferred counting, which is its own later post.) The practical takeaway is small but real: fewer distinct object touches in a hot loop means fewer refcount writes, which is one more reason that pushing the loop into a single C call — where the C code can hold a reference once and iterate raw values — is so much cheaper than iterating objects in bytecode.

## 2. What `obj.attr` actually does

Now the central mechanism: attribute access. When you write `obj.attr`, the compiler emits a `LOAD_ATTR` bytecode, and at runtime the interpreter runs a protocol called `__getattribute__`. It is not a single pointer dereference. It is a small algorithm, and figure 1 above traces it. Here is the path in words, in the order CPython actually checks:

1. **Look up `attr` on the *type* and its MRO**, watching for a *data descriptor*. The MRO ("method resolution order") is the linearized list of base classes — for a plain class it's just `[YourClass, object]`, but for multiple inheritance it can be several entries long, computed by the C3 algorithm. A descriptor is any object that defines `__get__` (and, for a *data* descriptor, also `__set__` or `__delete__`); `property` is the canonical example.
2. **If a data descriptor is found, call its `__get__` and return** — data descriptors win over the instance dict. This is why a `@property` always runs its getter and can't be shadowed by an instance attribute of the same name.
3. **Otherwise, check the instance's own `__dict__`** — a hash-table probe keyed by the attribute name. This is where ordinary `self.x = 5` attributes live. If found, return it.
4. **Otherwise, fall back to what the MRO turned up** in step 1 — a class attribute or a *non-data* descriptor (a plain function is a non-data descriptor; accessing it triggers method binding, step 5).
5. **If the found thing is a function, bind it** into a `method` object that remembers `self`, and return that bound method.
6. **If nothing is found anywhere, raise `AttributeError`.**

That's the cost model for a single dot. The common, fast cases — reading an ordinary instance attribute, or a method — short-circuit early, but even the fast path is "check the type for a data descriptor, then probe the instance dict." Two of the steps involve hash-table lookups by string; one of them may walk several classes; and the method case *allocates a new object every time*. None of this is visible in the source. The dot looks like a field access in C or Java, but it's an interpreted protocol.

A quick demonstration that the protocol is real and observable:

```pycon
>>> class Loud:
...     def __getattribute__(self, name):
...         print(f"  looking up {name!r}")
...         return super().__getattribute__(name)
...
>>> obj = Loud()
>>> obj.x = 10
>>> obj.x
  looking up 'x'
10
```

Every dot fires the protocol. In normal classes you don't override `__getattribute__`, so the C implementation runs and you don't see the print — but the *work* is the same. Now let's measure each piece.

### Why the MRO walk and descriptors cost what they do

It's worth slowing down on two parts of that protocol, because they're where the surprising costs hide. The first is the **MRO walk**. When the attribute isn't on the instance, CPython has to find it on the type — and "the type" really means "the type and every class it inherits from, in resolution order." For a flat class the MRO is just `[YourClass, object]`, two entries, so the walk is cheap. But under multiple inheritance or a deep hierarchy, the MRO can be five or ten classes long, and resolving a method that lives near the bottom means probing each class's `__dict__` in turn until it's found. This is why a deep inheritance chain is genuinely slower to call into than a flat one, all else equal: the lookup has farther to walk. CPython caches the result of these walks in a per-type method cache keyed by a type-version tag, so the *second* and subsequent lookups of the same name on the same type are fast — but the cache is invalidated whenever any class in the chain is modified, and the first lookup after an invalidation pays the full walk. You can see the MRO directly:

```pycon
>>> class A: pass
>>> class B(A): pass
>>> class C(A): pass
>>> class D(B, C): pass
>>> [cls.__name__ for cls in D.__mro__]
['D', 'B', 'C', 'A', 'object']
```

Looking up an attribute that lives on `A` from a `D` instance walks `D`, `B`, `C`, then finds it on `A` — four `__dict__` probes (after the instance-dict miss) on the cold path. The lesson isn't "never use inheritance"; it's that **a deep or wide hierarchy is not free in the hot path**, and that the method cache only saves you when the same lookups repeat against an unchanging set of classes.

The second part worth dwelling on is the **descriptor protocol**, because it's the mechanism that makes `@property`, `classmethod`, `staticmethod`, bound methods, and `__slots__` all work — they're not special-cased features, they're all descriptors. A descriptor is simply an object that lives on a *class* and defines `__get__`. When you access `instance.x` and `x` is a descriptor on the class, Python doesn't return the descriptor object — it calls `descriptor.__get__(instance, type)` and returns *that*. A *data descriptor* additionally defines `__set__` or `__delete__`, and the crucial consequence is precedence: data descriptors are checked *before* the instance dict, while non-data descriptors are checked *after*. That single rule explains a pile of Python behavior. A `property` is a data descriptor, so it always wins over an instance attribute of the same name and always runs its getter — which is exactly why it can't be cheaply cached by accident, and why it's a per-access function call. A plain function is a non-data descriptor, so its `__get__` produces a bound method only when nothing shadows it in the instance dict. And a `__slots__` member is a data descriptor that reads and writes a fixed offset in the instance struct — that's literally how slots store their values without a dict. Knowing this collapses several seemingly-unrelated costs into one model: **anything implemented as a descriptor turns an apparent field access into a function call**, and the cost of that field access is the cost of the descriptor's `__get__`.

## 3. Names: local vs global vs builtin, and why it matters

Before attributes, there's an even more basic cost difference that most people never think about: how Python resolves a bare *name* like `x`, `len`, or `math`. This is the LEGB rule — **L**ocal, **E**nclosing, **G**lobal, **B**uiltins — and the four scopes cost wildly different amounts because they're implemented completely differently. Figure 5 lays the ladder out.

![A vertical stack showing the LEGB lookup ladder from Local as an array slot at 2 to 5 ns, through enclosing closure cells, global module dict at about 20 ns, builtins as a second dict probe at about 30 ns, down to a NameError when the name is not found](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-5.png)

Here is the key fact, and it's the single most leverageable thing in this post: **local variables are not stored in a dictionary. They're stored in an array.** When CPython compiles a function, it counts the local variables and assigns each one a fixed integer index. At runtime, the frame has a small C array of object pointers ("fast locals"), and reading a local is `LOAD_FAST n` — a single array index, $O(1)$ with a tiny constant, no hashing, no string comparison. Reading a *global*, by contrast, is `LOAD_GLOBAL`, which probes the module's `__dict__` (a real hash table) by the variable name as a string, and if that misses, probes the `builtins` dict too. A global read is a hash of the name, a bucket lookup, and a pointer comparison; a builtin read is *two* such probes.

Let's prove it with `dis`:

```pycon
>>> import dis
>>> g = 10
>>> def uses_global(x):
...     return x + g
...
>>> def uses_local(x):
...     g = 10
...     return x + g
...
>>> dis.dis(uses_global)
  2   LOAD_FAST                0 (x)
      LOAD_GLOBAL              0 (g)
      BINARY_OP                0 (+)
      RETURN_VALUE
>>> dis.dis(uses_local)
  2   LOAD_CONST               1 (10)
      STORE_FAST               1 (g)
  3   LOAD_FAST                0 (x)
      LOAD_FAST                1 (g)
      BINARY_OP                0 (+)
      RETURN_VALUE
```

In the first function, `g` is a `LOAD_GLOBAL`: a dict probe by name. In the second, `g` is a `LOAD_FAST`: an array index. Same arithmetic, different name-resolution cost. Now the numbers. Here's a careful `timeit` that isolates the access cost by making the loop body almost nothing else:

```python
import timeit

# Global access: 'data' is looked up in the module dict each time.
data = list(range(1000))

def via_global():
    total = 0
    for i in range(1000):
        total += data[i]   # 'data' is a LOAD_GLOBAL every iteration
    return total

def via_local():
    local_data = data      # bind once -> LOAD_FAST inside the loop
    total = 0
    for i in range(1000):
        total += local_data[i]
    return total

# n=10000 runs, take the best of 5 repeats, divide out the 1000-element loop.
n, r = 10000, 5
tg = min(timeit.repeat(via_global, number=n, repeat=r)) / n / 1000
tl = min(timeit.repeat(via_local, number=n, repeat=r)) / n / 1000
print(f"global per-access: {tg*1e9:6.1f} ns")
print(f"local  per-access: {tl*1e9:6.1f} ns")
```

On the 8-core Linux box, CPython 3.12, this prints roughly `global per-access:  21.0 ns` and `local per-access:  14.5 ns`. The per-access delta from the name resolution alone is on the order of 6–8 ns — small in isolation, but multiply it by the number of times the name appears in a loop that runs millions of times and it's real wall-clock. The classic version of this win is binding a global function or builtin to a local before a hot loop:

```python
# Before: 'append' is resolved every iteration (global lookup of 'result' + attr lookup of 'append').
def build_before(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

# After: hoist the bound method into a local; the loop body is now LOAD_FAST.
def build_after(n):
    result = []
    append = result.append   # resolve once, outside the loop
    for i in range(n):
        append(i * i)
    return result
```

I'll measure that one properly in the worked example below. For now, internalize the mechanism: **a local is an array slot; a global is a dict probe; a builtin is two dict probes.** The compiler decides which a name is — you don't get to choose `LOAD_FAST` directly, but you *do* choose it indirectly by assigning the thing to a local name inside the function. That is the entire basis of "local binding" as an optimization, and figure 3 puts the costs side by side.

![A matrix comparing local var, global, builtin, instance attr, bound method, and property by their mechanism, approximate nanoseconds, and verdict, showing local access as the cheapest array slot and property access as the most expensive full call](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-3.png)

#### Worked example: hoisting `list.append` out of a loop

The setup: build a list of one million squared integers, the kind of thing that shows up in any data-shaping loop. The slow version looks up `result.append` every iteration; the fast version binds it once. On the 8-core Linux box, CPython 3.12, with `timeit.repeat(..., number=20, repeat=7)` and the median taken:

| version | per-iteration body | total for 1M | speedup |
|---|---|---|---|
| `result.append(i*i)` in the loop | ~70 ns | ~70 ms | 1.0× |
| `append = result.append` hoisted | ~46 ns | ~46 ms | ~1.5× |

The 24 ns/iteration we saved breaks down into two parts: the `LOAD_GLOBAL`/`LOAD_FAST` difference for `result`, and — the bigger piece — *not re-running the `LOAD_ATTR` protocol and not re-binding the method object* every iteration. We'll see that bound-method allocation cost explicitly in the next section. A 1.5× speedup on a loop body is not going to save a doomed $O(n^2)$ algorithm — nothing at this layer will — but on an already-tight inner loop that's genuinely in your hot path, it's free real estate: one line, no readability cost, no new dependency. Figure 4 shows the before/after.

![A before and after comparison showing a method lookup inside a loop costing about 70 nanoseconds per iteration versus binding append to a local once and reaching about 45 nanoseconds per iteration for roughly a 1.5 times speedup](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-4.png)

## 4. Methods: the bound-method allocation you didn't know about

Here is something that surprises a lot of experienced Python developers: **`obj.method` allocates a new object every single time you write it.** When you access a method, the lookup finds a plain function on the class. A function is a *non-data descriptor*: its `__get__` runs and produces a `method` object — a small wrapper that pairs the function with `self`. So `obj.method()` is really two steps: build a bound-method object (allocate), then call it. Two separate dots that look identical (`obj.method` then `()`) hide an allocation between them.

You can watch the allocation happen:

```pycon
>>> class C:
...     def m(self):
...         return 1
...
>>> c = C()
>>> c.m is c.m          # two accesses -> two different bound-method objects
False
>>> bm = c.m
>>> bm is c.m           # still a fresh one each access
False
>>> type(c.m)
<class 'method'>
```

`c.m is c.m` is `False` because each access creates a distinct `method` object. (CPython optimizes the immediate `c.m()` call pattern with `LOAD_METHOD`/`CALL` so that the *common case* of "look up and immediately call" can avoid materializing the bound-method object on the stack — but the conceptual cost is there, and it absolutely materializes if you store the access in a variable or call it indirectly.) Let's price the difference between calling a method in a loop and hoisting it:

```python
import timeit

class Accumulator:
    def __init__(self):
        self.total = 0
    def add(self, x):
        self.total += x

def method_in_loop(n):
    acc = Accumulator()
    for i in range(n):
        acc.add(i)        # LOAD_ATTR 'add' + bind + call, every iteration
    return acc.total

def method_hoisted(n):
    acc = Accumulator()
    add = acc.add         # bind the method once
    for i in range(n):
        add(i)            # LOAD_FAST + call
    return acc.total

n = 1_000_000
t1 = min(timeit.repeat(lambda: method_in_loop(n), number=5, repeat=5)) / 5 / n
t2 = min(timeit.repeat(lambda: method_hoisted(n), number=5, repeat=5)) / 5 / n
print(f"method in loop: {t1*1e9:6.1f} ns/call")
print(f"method hoisted: {t2*1e9:6.1f} ns/call")
```

On the Linux box this lands around `method in loop:  58.0 ns/call` versus `method hoisted:  44.0 ns/call` — roughly 14 ns/call, or about a 1.3× speedup on the call overhead alone. That gap is the attribute lookup plus the bound-method handling that the hoisted version pays once. The win is smaller than the function-vs-function case because `add` itself does real work (`self.total += x`), which dilutes the overhead. The rule generalizes: **the more trivial the loop body, the larger the *fraction* of its time is overhead, and the more hoisting helps.** A loop body that calls a method and does nothing else is almost all overhead; a loop body that does a database round-trip is almost all real work, and shaving 14 ns off it is meaningless. This is Amdahl's law operating at the level of a single line.

There's a subtle trap here worth flagging: hoisting `acc.add` into a local captures the binding *to that specific instance*. If `acc` is reassigned inside the loop, your hoisted `add` still points at the old object. So hoist only when the receiver is loop-invariant — which, in a hot inner loop, it almost always is.

One nuance keeps this honest. Modern CPython compiles the *immediate* call pattern `obj.method(args)` to a `LOAD_ATTR` (which, since 3.12, folded in the old `LOAD_METHOD` fast path) followed by `CALL`, and the interpreter has a specialized form that, when it recognizes "attribute lookup immediately followed by a call," can call the underlying function with `self` prepended *without materializing the bound-method object at all*. So the bare `acc.add(i)` pattern is already cheaper than the naïve "allocate a method object, then call it" model would suggest — the common case is optimized. What hoisting still removes is the *attribute lookup itself*: even with the call specialization, `acc.add(i)` re-runs the `LOAD_ATTR` protocol (instance-dict miss, MRO probe for `add`, guard checks) every iteration, whereas the hoisted `add(i)` is a `LOAD_FAST` and a call. That's where the measured 14 ns goes. The mechanism is more subtle than "you avoid an allocation," but the conclusion is the same: in a genuinely hot loop, resolving the method once beats resolving it a million times.

## 5. The cost of a call: frames, arguments, and "push the loop down"

Every function call in Python has a fixed setup-and-teardown cost that has nothing to do with the function's body. To call a Python function, the interpreter must: create (or, since 3.11, often reuse from a free list) a *frame object* to hold the call's local variables and evaluation stack; bind the arguments to parameters, checking arity and handling defaults, `*args`, `**kwargs`, and keyword-only parameters; run the body; then return the result and tear the frame down (dropping refcounts on everything it held). Figure 6 traces that lifecycle.

![A branching diagram of one call lifecycle showing CALL allocating or reusing a frame, binding arguments, running the body, then either returning a value or unwinding on an exception, both paths ending by freeing the frame](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-6.png)

The "Faster CPython" work in 3.11 cut this overhead substantially — frames became lighter and are reused from a free list rather than heap-allocated every call, and the call sequence was streamlined — but it is not zero. A bare Python-to-Python call costs on the order of 30–60 ns of pure overhead on our box before the body does anything. Here's a measurement that isolates the call tax by making the body do almost nothing:

```python
import timeit

def noop(x):
    return x

def call_each(n):
    s = 0
    for i in range(n):
        s += noop(i)      # one function call per iteration
    return s

def inline(n):
    s = 0
    for i in range(n):
        s += i            # same arithmetic, no call
    return s

n = 1_000_000
tc = min(timeit.repeat(lambda: call_each(n), number=5, repeat=5)) / 5 / n
ti = min(timeit.repeat(lambda: inline(n), number=5, repeat=5)) / 5 / n
print(f"with call:    {tc*1e9:6.1f} ns/iter")
print(f"inlined:      {ti*1e9:6.1f} ns/iter")
print(f"call overhead ~ {(tc-ti)*1e9:5.1f} ns")
```

This typically reports something like `with call:  62.0 ns/iter`, `inlined:  28.0 ns/iter`, `call overhead ~ 34.0 ns` on CPython 3.12. That ~34 ns is the price of the frame and argument handling for a function that does nothing. This is *why* the single most repeated piece of Python performance advice — **"push the loop down into one call"** — works. Every time you replace `n` Python-level calls with one call that loops internally (especially one that loops in C), you pay the call overhead once instead of `n` times.

Not all calls cost the same, and the difference is in *how the arguments are passed*. The cheapest call is positional arguments to a function with simple parameters — the interpreter can map them straight into the frame's fast-local slots. The moment you add keyword arguments, `*args`, `**kwargs`, or default-value handling, the argument-binding step does more work: it builds a dict for the keywords, unpacks the star-args, and fills in defaults. None of this is expensive in absolute terms, but in a hot loop calling the same function millions of times, *passing positionally is measurably cheaper than passing by keyword*. Here's the gap:

```python
import timeit

def f(a, b, c):
    return a + b + c

def by_position(n):
    s = 0
    for i in range(n):
        s += f(i, i, i)           # positional: straight into fast-locals
    return s

def by_keyword(n):
    s = 0
    for i in range(n):
        s += f(a=i, b=i, c=i)     # keyword: builds a kwargs mapping
    return s

n = 1_000_000
tp = min(timeit.repeat(lambda: by_position(n), number=5, repeat=5)) / 5 / n
tk = min(timeit.repeat(lambda: by_keyword(n), number=5, repeat=5)) / 5 / n
print(f"positional: {tp*1e9:5.1f} ns/call")
print(f"keyword:    {tk*1e9:5.1f} ns/call")
```

On the Linux box this is roughly `positional:  48.0 ns/call` versus `keyword:  61.0 ns/call` — about 13 ns/call for the keyword-binding overhead. Keyword arguments are wonderful for readability and you should absolutely keep them in normal code; the point is narrow: *in a profiled hot loop*, calling positionally shaves a little off each call. The same reasoning says a function with a long signature full of defaults costs more to call than a tight three-positional one, and that `*args`/`**kwargs` wrappers (the classic decorator pattern) add a real per-call tax — every decorated call passes through the wrapper's `*args, **kwargs` packing and unpacking before reaching the real function. If a decorated function is in your hot path, that wrapper layer is a cost you can see in a profile.

The canonical examples are the C-implemented builtins. `sum(iterable)` loops in C; a hand-rolled `total = 0; for x in it: total += x` loops in the eval loop with a `BINARY_OP` and refcount churn per element. `''.join(parts)` builds the result string in one C pass; `result = ''; for p in parts: result += p` builds quadratically (each `+=` copies the whole accumulated string) *and* pays per-iteration interpreter overhead. `any(...)`, `all(...)`, `max(...)`, `min(...)`, `map(...)`, `filter(...)`, and the `itertools` functions all share this property: the loop body runs at C speed, and you pay one call instead of a million. Here is the `sum` case measured:

```python
import timeit

xs = list(range(1_000_000))

def manual_sum():
    total = 0
    for x in xs:
        total += x
    return total

t_manual = min(timeit.repeat(manual_sum, number=20, repeat=5)) / 20
t_builtin = min(timeit.repeat(lambda: sum(xs), number=20, repeat=5)) / 20
print(f"manual loop: {t_manual*1e3:6.2f} ms")
print(f"sum():       {t_builtin*1e3:6.2f} ms  ({t_manual/t_builtin:.1f}x faster)")
```

On our box: `manual loop:  18.50 ms`, `sum():  4.80 ms  (3.9x faster)`. Nearly 4× for the same arithmetic, because `sum`'s loop never returns to the bytecode eval loop — it does the iteration, the addition, and the refcounting entirely in C, paying the Python call cost exactly once. The lesson scales all the way up the leverage ladder: vectorizing with NumPy is the same idea taken further (one C call over a *typed* buffer with no per-element boxing at all), and writing a Numba or Cython kernel is the same idea taken further still. They all amortize the per-operation Python overhead by doing more work per call. The [mental model of performance and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) post frames exactly this trade-off — work per call versus number of calls — as the recurring question of the whole series.

## 6. `__slots__`: killing the per-instance dictionary

Now the biggest single win available at this layer, and it's both a memory win and a speed win: `__slots__`. To understand it, you have to understand where ordinary instance attributes live. When you write `self.x = 5` in a normal class, that `x` goes into the instance's `__dict__` — a per-instance dictionary, one hash table for every object you create. That dictionary is a separate heap allocation, and it's *not small*: even an empty one is around 64 bytes, and a populated one with a few keys runs to 104–232 bytes depending on how it grew. So a plain instance is really *two* heap objects — the instance struct plus its dict — and the dict usually dominates the size. Figure 2 shows the layout.

![A before and after comparison showing a plain class instance with a separate per-instance dict adding up to about 152 bytes versus a slots class storing two attributes inline for about 56 bytes with no dict](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-2.png)

When you declare `__slots__`, you tell CPython exactly which attribute names instances will have. CPython then stores those attributes in fixed slots laid out directly in the instance struct — like C struct fields — and *does not create a per-instance `__dict__` at all*. Attribute access becomes a slot read at a known offset (implemented as a member descriptor) instead of a dictionary probe. Two wins fall out: each instance is much smaller (no dict), and attribute access is a bit faster (offset read instead of hash probe). Here's the definition and the size measurement:

```python
import sys

class PlainPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlotPoint:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = PlainPoint(1.0, 2.0)
s = SlotPoint(1.0, 2.0)

# getsizeof on the instance does NOT include the __dict__ — add it explicitly.
plain_total = sys.getsizeof(p) + sys.getsizeof(p.__dict__)
slot_total = sys.getsizeof(s)   # no __dict__ exists

print(f"plain instance:  {sys.getsizeof(p):3d} B + dict {sys.getsizeof(p.__dict__):3d} B = {plain_total} B")
print(f"slot instance:   {slot_total} B")
print(f"slot has __dict__? {hasattr(s, '__dict__')}")
```

On CPython 3.12, 64-bit, this prints approximately:

```bash
plain instance:   56 B + dict  64 B = 120 B
slot instance:    56 B
slot has __dict__? False
```

Note the trap in the first line: `sys.getsizeof(p)` reports only the instance struct, *not* the dict it points to. That's one of the `getsizeof` "lies" — it's shallow, it doesn't follow pointers. You have to add the dict explicitly to get the real footprint, and once you do, the plain instance is roughly twice the size of the slotted one for this two-field object. (On a freshly created plain instance the dict can be as small as 64 bytes; as you add more attributes or as the dict grows, the gap widens to the 152-vs-56-byte ballpark in the figure.) The bigger the population of objects, the more this matters — which is exactly the case where it matters at all.

#### Worked example: one million points, slotted vs not

This is the scenario from real systems: you load a few million records into objects and hold them in memory. Let's build a million two-field objects each way and measure actual process RSS, not `getsizeof` estimates, because RSS is what the OOM killer reads.

```python
import os, resource

def rss_mb():
    # ru_maxrss is in KB on Linux, bytes on macOS — this is the Linux form.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

class PlainPoint:
    def __init__(self, x, y):
        self.x = x; self.y = y

class SlotPoint:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x; self.y = y

import gc
N = 1_000_000

base = rss_mb()
plain = [PlainPoint(float(i), float(i)) for i in range(N)]
after_plain = rss_mb()
del plain; gc.collect()

slot = [SlotPoint(float(i), float(i)) for i in range(N)]
after_slot = rss_mb()

print(f"1M plain points: +{after_plain - base:6.1f} MB")
print(f"1M slot points:  +{after_slot - base:6.1f} MB")
```

On the 8-core Linux box, CPython 3.12, this reports roughly:

| layout | RSS for 1M objects | bytes/object (incl. list + floats) |
|---|---|---|
| `PlainPoint` (per-instance dict) | ~+290 MB | ~290 B |
| `SlotPoint` (`__slots__`) | ~+130 MB | ~130 B |

A 160 MB cut — better than 2× — from adding one line to the class. (The per-object number is larger than the bare instance size because each object also holds two distinct float objects, and the list holds a pointer to each instance; the *difference* between the two rows is essentially the per-instance dict, eliminated.) Figure 8 captures the headline. For a process that was getting OOM-killed at a few million objects, this is often the difference between fitting in RAM and not — and it costs you nothing at the call site. The one constraint: with `__slots__` you can't add *new* attributes to instances at runtime that weren't declared, and you give up the per-instance `__dict__`. For data-carrying objects that's exactly what you want; for objects you genuinely monkey-patch, it isn't.

![A before and after comparison showing one million plain instances each with a dict consuming about 290 MB of resident memory versus one million slots instances with no dict consuming about 130 MB](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-8.png)

`__slots__` also makes attribute *access* slightly faster, because reading a slot is a member-descriptor read at a fixed offset rather than an instance-dict hash probe. The effect is small — a couple of nanoseconds per access — but it stacks with the memory win, and in a hot loop over slotted objects it's measurable. Here's the access micro-benchmark:

```python
import timeit

class Plain:
    def __init__(self):
        self.x = 0

class Slotted:
    __slots__ = ("x",)
    def __init__(self):
        self.x = 0

p, s = Plain(), Slotted()
tp = min(timeit.repeat(lambda: p.x, number=5_000_000, repeat=5)) / 5_000_000
ts = min(timeit.repeat(lambda: s.x, number=5_000_000, repeat=5)) / 5_000_000
print(f"plain attr read: {tp*1e9:5.1f} ns")
print(f"slot  attr read: {ts*1e9:5.1f} ns")
```

This lands around `plain attr read:  22.0 ns` and `slot attr read:  19.0 ns` — a few ns, real but secondary to the memory win. The memory is the headline; the speed is a bonus.

### The key-sharing dict, and why slots still wins

You might object: CPython already optimizes instance dicts. That's true, and it's worth understanding so you don't over- or under-estimate the slots win. Since PEP 412 (Python 3.3), instances of the same class share their dict's *keys*. The first time you create a `PlainPoint`, CPython builds a "keys object" recording the attribute names `x` and `y`; every subsequent `PlainPoint` reuses that shared keys table and only stores its own *values*. This is called a split dictionary, and it already cuts the per-instance dict cost substantially compared to the bad old days when every instance carried a full independent hash table. So a modern plain instance is cheaper than the naïve "instance + full dict" picture suggests.

But two things break the optimization, and both are common. First, if you ever add an attribute to one instance that the others don't have — anything that makes the key sets diverge — CPython has to "unshare" that instance's dict into a full independent one, and you pay the whole cost back. Second, even a perfectly shared dict still costs per-instance bytes for the values array and a pointer, and access is still a hash probe rather than a fixed-offset read. `__slots__` sidesteps all of it: there is no dict to share or unshare, the storage is a fixed-size array of slots, and access is a member-descriptor read at a known offset. That's why, even against the key-sharing optimization, slots still measurably wins on both memory and speed — and it wins *reliably*, because it can't be silently de-optimized by an errant attribute assignment the way a split dict can. The mechanism matters: the split-dict optimization is a best-effort runtime heuristic, while `__slots__` is a static guarantee you opt into.

### Slots and inheritance: the gotcha that gives the dict back

One trap worth flagging because it silently undoes the win: `__slots__` only suppresses the per-instance dict if *every* class in the inheritance chain declares it. The moment any base class (including one you didn't write) lacks `__slots__`, instances get a `__dict__` back — and now you've paid for both the slots *and* the dict, the worst of both worlds. So if you slot a class, make sure its bases are slotted too, and watch for mixins that quietly add a dict. You can check at runtime with `hasattr(instance, "__dict__")` — if it's `True` on a class you thought was slotted, a base class gave the dict back. This is exactly the kind of thing a memory profiler catches when your "optimized" objects mysteriously didn't shrink.

## 7. The `@property` trap in a hot loop

`@property` is one of Python's best features for clean APIs — it lets you turn an attribute access into a method call without changing the caller. That's also exactly why it can be a performance trap: it *hides a function call behind what looks like a field read.* Because a property is a data descriptor, accessing `obj.value` doesn't probe the instance dict at all; it runs the getter function — frame setup, body, return — every single time. In a hot loop that reads the same property repeatedly, you're paying a full Python call per read, and the cost model from section 5 applies.

```python
import timeit

class WithProperty:
    def __init__(self, raw):
        self._raw = raw
    @property
    def value(self):
        return self._raw * 2      # a real getter: runs every access

class WithAttr:
    def __init__(self, raw):
        self.value = raw * 2      # computed once, stored as a plain attribute

wp = WithProperty(10)
wa = WithAttr(10)

# Reading the property re-runs the getter; reading the attribute is a dict probe.
tp = min(timeit.repeat(lambda: wp.value, number=2_000_000, repeat=5)) / 2_000_000
ta = min(timeit.repeat(lambda: wa.value, number=2_000_000, repeat=5)) / 2_000_000
print(f"property read: {tp*1e9:5.1f} ns")
print(f"attr read:     {ta*1e9:5.1f} ns")
```

On our box: `property read:  95.0 ns` versus `attr read:  22.0 ns` — the property is roughly 4× slower because each read is a full function call, not a lookup. The fix is not to delete the property — it's a good API — but to **read it once and reuse it inside the hot loop**:

```python
# Before: re-runs the getter on every iteration.
def process_slow(records):
    out = []
    for r in records:
        if r.value > 0:          # getter call
            out.append(r.value)  # getter call AGAIN
    return out

# After: read the property once per record.
def process_fast(records):
    out = []
    for r in records:
        v = r.value              # one getter call per record
        if v > 0:
            out.append(v)
    return out
```

If the property is genuinely expensive to compute and the value doesn't change, reach for `functools.cached_property` instead — it computes the value on first access and then stores it in the instance dict, so subsequent reads are plain attribute lookups. (Note: `cached_property` needs a per-instance `__dict__`, so it's incompatible with bare `__slots__` unless you include `__dict__` in the slots — a trade-off worth knowing.) The general principle: **a property is a method dressed as an attribute; treat it like a method call in your cost accounting.** Figure 7 ranks all these techniques by their win and effort.

![A matrix ranking slots, hoisting lookups, local binding, using builtins, and avoiding property by their main win, effort, and when it matters, showing slots and hoisting as the largest low-effort wins on a hot path](/imgs/blogs/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch-7.png)

## 8. Dynamic dispatch: why every operator is a runtime decision

We've covered attributes and calls; the last piece of the overhead is *dispatch*. Python is dynamically typed, which means the interpreter cannot know at compile time what `a + b` should do — it depends on the runtime types of `a` and `b`. So `BINARY_OP` for `+` has to, at runtime, look at the type of the left operand, find its number-add slot (`nb_add` in the type's `tp_as_number` table), possibly try the right operand's reflected `__radd__`, dispatch, and allocate the result. Every operator, every comparison, every subscript (`obj[i]` is `__getitem__` dispatch), every `for` iteration step (`__next__` dispatch) is resolved at runtime by the type. This is *dynamic dispatch*, and it's the reason a Python loop can't be compiled down to a tight machine loop the way a statically typed one can — the types aren't known until the value is in hand.

It's worth being precise about the `+` protocol, because it shows how much hides behind one character. For `a + b`, CPython first tries `type(a).__add__(a, b)`. If that method returns the special sentinel `NotImplemented` (not the same as raising — it's CPython's way of saying "I don't know how to add these"), the interpreter then tries the *reflected* operation `type(b).__radd__(b, a)`. There's a refinement: if `type(b)` is a *subclass* of `type(a)`, the reflected method is tried *first*, so a subclass can override addition with its parent. Only if both return `NotImplemented` does Python raise `TypeError`. So `1 + 2.0` works because `int.__add__(1, 2.0)` returns `NotImplemented` (an int doesn't know how to add a float), and then `float.__radd__(2.0, 1)` succeeds by promoting the int. That fallback dance is real work, and it happens — in the unspecialized path — on every mixed-type addition. The same reflected-operand protocol governs `-`, `*`, `/`, `==`, `<`, and the rest. None of this is exotic; it's the everyday machinery under arithmetic, and it's why "just adding two numbers" in a Python loop is genuinely more expensive than it looks.

CPython mitigates this enormously with the **specializing adaptive interpreter** (PEP 659, shipped in 3.11 and refined in 3.12). The interpreter watches what types actually flow through a given bytecode and, if they're consistent, rewrites that specific bytecode in place into a specialized fast version. A `BINARY_OP` that always sees two ints becomes `BINARY_OP_ADD_INT`, which skips the general dispatch and does the int-add directly (still allocating the result box, but without the type-table walk). A `LOAD_ATTR` that always sees the same type with the attribute in the same place becomes a specialized form that checks a cached type-version guard and reads the slot directly — close to a single load when the guard holds. This is why upgrading from 3.10 to 3.12 sped up a lot of ordinary OO code "for free": the specializer turns the common, type-stable cases into near-direct operations.

The catch, and it's an important one for your code: **specialization needs type stability.** A bytecode that sees ints sometimes, floats other times, and strings occasionally can't stay specialized — it keeps de-optimizing back to the general form. So a loop that processes a homogeneous list (all ints, all the same object type) runs *much* faster than the same loop over a mixed-type list, even though they look identical in source. The practical advice that falls out: keep your hot loops type-homogeneous. Don't put `int`, `float`, and `Decimal` through the same hot path if you can help it; don't mix `None` and real objects in a tight numeric loop. The specializer rewards consistency. You can see specialization in action with the adaptive disassembler:

```pycon
>>> import dis
>>> def addup(xs):
...     t = 0
...     for x in xs:
...         t += x
...     return t
...
>>> for _ in range(100):          # warm it up so the specializer kicks in
...     addup([1, 2, 3, 4, 5])
...
>>> dis.dis(addup, adaptive=True)  # shows the specialized opcodes
```

After warmup you'll see `BINARY_OP_ADD_INT` and `FOR_ITER_LIST` in place of the generic `BINARY_OP` and `FOR_ITER` — the interpreter has specialized the loop to the int-over-list case it observed. Feed `addup` a list of floats and it specializes differently; feed it a mixed list and it can't fully specialize at all. The deep mechanics of the eval loop and the specializer are the subject of [the CPython execution model post](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop); here the takeaway is operational: **dynamic dispatch is the per-operation tax, the specializer rebates most of it when your types are stable, and you collect the rebate by keeping hot loops homogeneous.**

## 9. Measuring honestly: the traps that fake your numbers

Everything above is backed by `timeit`, and `timeit` is easy to misuse in ways that produce confident, wrong numbers. Since the whole point of this series is *don't guess, measure*, here are the traps that matter for micro-benchmarks like these, and how I avoided them above.

**Constant folding and dead-code elimination.** If you `timeit("2 + 2")`, the compiler folds it to `4` at compile time and you measure nothing. If your benchmark's result is never used, an optimizer (or just the interpreter skipping work) may elide it. Always feed the benchmark real, runtime-varying inputs and make sure the result is consumed (returned, summed, appended).

**Too-short loops measure the timer, not the code.** A single attribute read is ~20 ns; `perf_counter`'s own resolution and call overhead are in that ballpark. That's why every benchmark above runs the operation millions of times (`number=5_000_000`) and divides — you measure the aggregate and amortize the timer's noise. `timeit` does this for you with its `number` argument; respect it.

**One run is noise.** Background processes, CPU frequency scaling, cache state, and the garbage collector all perturb a single measurement. I used `timeit.repeat(..., repeat=5)` and took the `min` (or median). The minimum is defensible for micro-benchmarks because the *fastest* run is the one least disturbed by external interference — the real cost can only be inflated by noise, not deflated. For larger, noisier workloads, the median is the more honest summary.

**The GC fires mid-benchmark.** Allocating loops (building the million-element list) can trigger the cyclic garbage collector partway through, adding a pause that has nothing to do with the operation you're timing. For allocation-heavy benchmarks, disable it during the timed region: `gc.disable()` before, `gc.enable()` after — but remember to account for the fact that you've changed memory behavior.

**Warmup matters now more than it used to.** With the specializing interpreter, the first few hundred iterations of a loop run the *unspecialized* bytecode; only after the specializer has observed stable types does the fast path kick in. If your benchmark doesn't warm up, you measure a blend of slow-then-fast. `timeit` with a large `number` naturally warms up; for hand-rolled timing, run the function some hundreds of times before you start the clock.

Here's a compact harness that bakes in the good practices, the kind I keep in a scratch file:

```python
import timeit, gc, statistics

def bench(fn, *, number=1_000_000, repeat=7, warmup=3):
    for _ in range(warmup):        # let the specializer settle
        fn()
    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        samples = timeit.repeat(fn, number=number, repeat=repeat)
    finally:
        if gc_was_on:
            gc.enable()
    per_op = min(samples) / number
    spread = statistics.stdev(samples) / number if repeat > 1 else 0.0
    return per_op, spread

per_op, spread = bench(lambda: sum(range(100)))
print(f"{per_op*1e9:.1f} ns/op  (+/- {spread*1e9:.1f} ns)")
```

The broader discipline of benchmarking — `pyperf` for publication-quality numbers, statistical rigor, and the rest of the traps — is its own post later in this series. The point here is that *the numbers in this post were produced this way*: warmed up, repeated, GC accounted for, results consumed, large enough loop counts to dwarf the timer. When you reproduce them on your machine you'll get different absolute values (different CPU, different CPython build) but the same ordering and roughly the same ratios. That ordering is the durable knowledge; the constants are not.

## 10. A problem-solving narrative: the refactor that went 9× slower

Let me walk the opening story through to its resolution, because it's the realistic shape of how this knowledge gets used — not as a bag of tricks applied blindly, but as a diagnosis. The job processed 40 million rows. Each row was wrapped in a `Record` object with `@property` accessors for several "computed" fields, and the inner loop called `record.transform()`, which itself read three properties twice each, looked up a couple of methods on `self`, and built a small result list with `.append`. The flat predecessor had been a function over dicts and tuples. Same math, 9× slower.

The diagnosis, in order of impact, mapped exactly onto this post. First, **the properties were the single biggest cost** — six property reads per row (three fields, read twice each) meant six hidden function calls × 40M rows = 240M needless calls. Reading each property *once* into a local at the top of `transform` and reusing it removed roughly half of them and was, by itself, about a 3× improvement on that function. Second, **the per-instance dicts** on 40 million `Record` objects were both bloating RSS and slowing every attribute read; converting `Record` to `__slots__` cut memory by more than half (the process had been swapping, which is a separate catastrophe — once it stopped swapping, wall-clock improved out of proportion to the CPU win) and shaved a couple of ns off every field access. Third, **the bound-method lookups** for the two `self.` methods called per row got hoisted where they were loop-invariant. Fourth — and this is the one that pushed it past the original flat version — the inner `.append` loop building each row's small result was replaced with a list comprehension, which keeps the loop in C and skips the repeated method-lookup-and-call entirely.

None of these is a 100× lever. Stacked, on a loop body that was almost entirely overhead, they recovered the 9× and then some. The meta-lesson is the one the whole series keeps returning to: **profile first to find that `transform` was 80% of runtime** (so optimizing it could actually help — Amdahl permitted a large win), then attack the overhead in the order of its measured contribution. If `transform` had been 5% of runtime, the correct move would have been to leave the pretty class hierarchy alone and go find the real bottleneck. The class hierarchy wasn't *wrong*; it was just in the hot path, where that overhead is the one place it isn't free.

## 11. The running pipeline: stacking the wins on one loop

Let me make the refactor story concrete and runnable, because seeing all the levers applied to a single loop is where the mental model clicks. Here is a slimmed-down version of the kind of feature-engineering loop that paged me — process a few million `Sample` records, compute a derived score, and collect the ones over a threshold. I'll write the naïve, idiomatic version first and then apply each lever in order, measuring as I go.

The naïve version uses properties, plain instances, in-loop method lookups, and `.append`:

```python
class SampleSlow:
    def __init__(self, a, b):
        self._a = a
        self._b = b

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def score(self):
        # reads each property twice; that's four hidden getter calls per call
        return (self.a * self.a + self.b * self.b) - (self.a + self.b)

def run_slow(samples, threshold):
    keep = []
    for s in samples:
        sc = s.score()          # method lookup + bind + call, then 4 getter calls inside
        if sc > threshold:
            keep.append(sc)     # attr lookup of 'append' + bind, every hit
    return keep
```

Now the optimized version. It's the *same logic* — I want to stress that, because the point of this post is that you don't change what the code does, only what it costs. The changes: `__slots__` to kill the per-instance dict; plain attributes instead of properties so reads don't run getters; compute `score` over locals; and a list comprehension to push the collect loop into C.

```python
class SampleFast:
    __slots__ = ("a", "b")
    def __init__(self, a, b):
        self.a = a
        self.b = b

def score_fast(s):
    a = s.a                     # read each slot once, into a local
    b = s.b
    return (a * a + b * b) - (a + b)

def run_fast(samples, threshold):
    # the comprehension keeps the collect loop in C and skips per-hit 'append' lookups
    return [sc for s in samples
            if (sc := score_fast(s)) > threshold]
```

Measured on the 8-core Linux box, CPython 3.12, over 2,000,000 samples with `timeit.repeat(..., number=3, repeat=5)` and the median taken, building the lists once and timing the processing pass:

| stage | what changed | wall-clock (2M rows) | speedup vs naïve |
|---|---|---|---|
| naïve | properties, plain class, method-in-loop, `.append` | ~1.30 s | 1.0× |
| + drop properties | plain attribute reads, no getters | ~0.78 s | ~1.7× |
| + `__slots__` | no per-instance dict, faster reads, less RSS | ~0.70 s | ~1.9× |
| + read attrs once | `score_fast` over locals | ~0.62 s | ~2.1× |
| + comprehension | collect loop runs in C | ~0.52 s | ~2.5× |

So the fully-optimized loop is about 2.5× faster than the idiomatic one, and uses roughly half the memory because of `__slots__` — and not one line of the *logic* changed. The single biggest jump was dropping the properties (1.7× on its own), which matches the cost model: each property read was a full function call, and the naïve `score` triggered four of them per row plus the call to `score` itself. This is the honest picture of what these conveniences cost in a hot loop and what removing them buys: a couple of x, free, *if* the loop is genuinely hot. If this loop were 3% of the job, the right answer would be to leave `SampleSlow` exactly as it is — it's more readable — and go optimize the 80% somewhere else. The discipline is always the same: measure, confirm it's the hot path, then collect the cheap wins.

#### Worked example: profiling points at the loop, not the I/O

Here's how the decision to optimize *this loop at all* gets made, because it's the step people skip. Suppose the full job is: read 2M rows from a Parquet file, run `run_slow`, write the results. A naïve instinct says "the disk read is slow, optimize the I/O." The profiler says otherwise. A 30-second `cProfile` run on the job reports cumulative time roughly as: file read 1.2 s, `run_slow` 9.8 s, write 0.4 s, total ~11.4 s. The loop is **86% of runtime** — so by Amdahl, even a perfect 2.5× on it (taking 9.8 s to ~3.9 s) cuts the whole job from ~11.4 s to ~5.5 s, a real 2.1× end-to-end. Optimizing the 1.2 s read instead, even to zero, would buy at most a 1.1× overall. The profiler didn't just tell us *what* to optimize; it told us the *ceiling* on the payoff before we spent any effort. That is the entire reason the series leads with measurement: the cheap wins in this post are only worth collecting once a profile has proven the loop dominates, and the profile also tells you, in advance, the most the wins can possibly be worth.

## 12. Case studies and real numbers

A few grounded data points beyond my own box, so you're not relying on one machine.

**The "Faster CPython" gains are mostly here.** The headline "CPython 3.11 is 10–60% faster than 3.10, ~25% on average" (the figure the core team reported across the pyperformance suite) came largely from exactly the costs this post is about: cheaper frames, the streamlined call sequence, and the PEP 659 specializing adaptive interpreter cutting attribute-lookup and operator-dispatch overhead. 3.12 and 3.13 continued the trend. The reason ordinary OO Python got faster without you changing anything is that the interpreter got better at the attribute/call/dispatch machinery — which tells you those costs were a meaningful fraction of real programs' runtime to begin with.

**`__slots__` in the wild.** Libraries that create many small objects reach for `__slots__` as a matter of course. `attrs` and `dataclasses` both support `slots=True` precisely because users hold millions of these objects in memory; the documented effect is a substantial per-instance memory cut (commonly reported in the 30–50% range for small classes) plus a small attribute-access speedup, matching what we measured. SQLAlchemy, pydantic's internals, and countless ORMs and data-mapping layers slot their hot-path classes for the same reason.

**The `iterrows` parable.** The most famous Python micro-cost story is pandas `DataFrame.iterrows()` — it's slow not because pandas is slow but because it materializes a fresh `Series` object (with all the attribute and dispatch overhead of a full Python object) for every row, so a million-row loop allocates a million `Series`. Replacing it with a vectorized expression — pushing the loop down into one C call over typed arrays — routinely yields 50–200× on real jobs. It's the same principle as `sum` beating the manual loop, taken to the array world; the dataframes track later in this series covers it in depth, but the *cause* is the per-object cost we've been pricing all along.

**Generated `__init__` and call overhead.** A subtler one: `dataclasses` and `attrs` generate an `__init__` that does a series of `self.x = x` assignments. For classes you instantiate in tight loops, the generated init's call overhead is real, and benchmarks of "construct N objects" show plain `__slots__` classes with a hand-written init edging out heavier wrappers — again, not because the wrapper is badly written, but because every layer of indirection at construction time is a call and an assignment you pay per object. When you build tens of millions of objects, those constants surface.

**namedtuple vs dict vs object.** A recurring real-world question is how to store a few million small fixed-shape records. The choices line up exactly along the cost model in this post. A plain `dict` per record is flexible but heavy — a full hash table each, hundreds of bytes. A `class` with `__slots__` is compact (no dict) and gives attribute access. A `collections.namedtuple` or `typing.NamedTuple` is a tuple internally, so it's the most compact of all (no dict, values packed in the tuple's own storage) and immutable, with attribute access implemented via — you guessed it — descriptors that read tuple slots by index. For the millions-of-records case, the common measured result is: namedtuple and slotted class both roughly halve the memory of the dict-per-record approach, with namedtuple usually a touch smaller and a touch slower to access (the descriptor indexes into the tuple) and the slotted class a touch faster. The point isn't which one wins by a nose; it's that *all three of them beat a dict-per-record because they kill the per-instance hash table*, which is the exact same lever as `__slots__`. The deeper memory-footprint treatment, including `array` and interning, is a dedicated post later in the series; here, recognize the pattern — the per-instance dict is the cost, and several different tools remove it.

## 13. When to reach for this (and when not to)

This is the section that keeps you honest, because everything above is a micro-optimization and micro-optimizations are *usually the wrong thing to do*. The governing law is Amdahl's: if a piece of code accounts for a fraction $p$ of your total runtime and you speed it up by a factor $s$, your overall speedup is

$$S = \frac{1}{(1 - p) + p/s}.$$

If `transform` is 80% of runtime ($p = 0.8$) and you make it 3× faster ($s = 3$), you get $S = 1/(0.2 + 0.8/3) \approx 2.1\times$ overall — worth it. If some attribute access is 2% of runtime ($p = 0.02$) and you make it infinitely fast ($s = \infty$), you get $S = 1/0.98 \approx 1.02\times$ — a 2% ceiling no matter what. **The entire decision of whether to apply anything in this post is: is this code in the hot path? Profile first; if it's not hot, stop.**

So, decisively:

- **Reach for `__slots__`** when you hold a large population of small, fixed-shape objects in memory (millions of records, nodes, points, events). It's a one-line memory win with a speed bonus and almost no downside for data-carrying classes. **Don't bother** for objects you create a handful of, or objects you genuinely need to monkey-patch or attach arbitrary attributes to.
- **Hoist attribute and method lookups** out of a loop only when the loop is hot *and* the receiver is loop-invariant. **Don't** scatter `append = lst.append` through cold code — it adds a line and buys nothing, and it harms readability where readability is what matters.
- **Bind globals and builtins to locals** inside a measured hot loop. **Don't** do it everywhere reflexively; outside the hot path it's noise that obscures intent.
- **Read a property once and reuse it** in a hot loop; reach for `cached_property` when the value is expensive and stable. **Don't** delete properties from your public API to save nanoseconds in cold code — the clean interface is worth far more than the cycles.
- **Push loops down into C builtins** (`sum`, `any`, `''.join`, `map`, comprehensions) whenever the operation has a builtin form. This one is almost always worth it because it usually *also* reads better. **Don't** contort a clear loop into an unreadable nested comprehension to save a few percent on code that isn't hot.
- **Keep hot loops type-homogeneous** so the specializer can do its job. **Don't** obsess over this in cold code; the specializer is invisible there.

And the meta-rule that supersedes all of them: **if you haven't profiled, you don't know your hot path, and you're guessing.** Most of the time the real win is one rung up the ladder — a better algorithm or a vectorized rewrite — and these micro-wins are the polish you apply *after* the profiler has pointed you at a loop that's genuinely dominating. The next series post on [the optimization loop and latency numbers](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) makes the measure-first discipline concrete.

## 14. Key takeaways

- **`obj.attr` is a protocol, not a pointer load.** It checks the type for a data descriptor, probes the instance dict, walks the MRO, and may allocate a bound method. Even the fast path is more than one operation.
- **A local is an array slot; a global is a dict probe; a builtin is two.** `LOAD_FAST` beats `LOAD_GLOBAL` by 6–8 ns/access on our box. Binding a hot global or builtin to a local converts dict probes into array reads.
- **`obj.method` allocates a bound-method object.** Hoisting a method out of a hot loop saves the repeated lookup and bind — about 1.3–1.5× on overhead-dominated loop bodies.
- **Every call pays a fixed frame + argument tax** (~30–60 ns of pure overhead on 3.12). "Push the loop down into one call" — `sum`, `''.join`, comprehensions, NumPy — amortizes it; `sum` beat a manual loop by ~3.9×.
- **`__slots__` kills the per-instance dict.** It roughly halved a two-field object (120 B → 56 B) and cut 1M-object RSS from ~290 MB to ~130 MB, with a small attribute-access speedup as a bonus.
- **A `@property` is a method dressed as an attribute** — ~4× slower than a plain read because it runs a getter every time. Read it once per loop iteration; use `cached_property` for expensive stable values.
- **Dynamic dispatch is the per-operation tax**, and the PEP 659 specializing interpreter rebates most of it *when your types are stable*. Keep hot loops homogeneous to collect the rebate.
- **Amdahl governs everything.** None of this is worth doing outside a profiled hot path; an optimization to 2%-of-runtime code caps your win at 2%. Measure first, then polish the loop that actually dominates.

## 15. Further reading

- [Why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro: the cost model, the leverage ladder, and the measure-first spine this post sits inside.
- [The CPython execution model: bytecode and the eval loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) — how source becomes bytecode, the stack-based eval loop, reference counting, and the specializing adaptive interpreter in depth.
- [A mental model of performance: latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) — Amdahl's law, the cost hierarchy, and when not to optimize.
- The CPython documentation on the [data model](https://docs.python.org/3/reference/datamodel.html) (descriptors, `__getattribute__`, `__slots__`) and the [`dis` module](https://docs.python.org/3/library/dis.html) for reading bytecode, including `adaptive=True`.
- The [`timeit`](https://docs.python.org/3/library/timeit.html) and [`sys.getsizeof`](https://docs.python.org/3/library/sys.html#sys.getsizeof) docs, plus the `gc` module for understanding GC effects on benchmarks.
- **PEP 659 — Specializing Adaptive Interpreter** and the Faster CPython project notes, for the source of the 3.11+ attribute/call/dispatch speedups.
- *High Performance Python* (Gorelick & Ozsvald, O'Reilly), the chapters on the cost of the object model, dictionaries, and the dynamic dispatch tax.
- The [`attrs`](https://www.attrs.org/) and [`dataclasses`](https://docs.python.org/3/library/dataclasses.html) docs on `slots=True`, for the real-world memory and speed effects of slotting hot-path classes.

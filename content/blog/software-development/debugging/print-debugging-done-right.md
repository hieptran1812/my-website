---
title: "Print Debugging Done Right: The Most-Used Tool, Used Deliberately"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Turn the print statement from a guilty habit into a precision instrument: structured trace lines, cheap object dumps, bisection with prints, and the disciplines that keep it from lying or leaving a mess."
tags:
  [
    "debugging",
    "software-engineering",
    "print-debugging",
    "logging",
    "tracing",
    "observability",
    "concurrency",
    "heisenbug",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/print-debugging-done-right-1.png"
---

At 2am the checkout service is double-charging maybe one customer in fifty thousand. You cannot reproduce it on your laptop. You cannot attach a debugger to a production pod that is serving live traffic without freezing every request that lands on it. The bug only happens when two retries of the same request interleave in a way you have never managed to trigger by hand. You have a debugger. It is, in this moment, useless to you. What you reach for instead is the oldest tool there is: you make the program tell you what it is doing, out loud, as it runs. You add a print.

There is a tired piece of folklore that print debugging is what beginners do until they learn to use a "real" debugger. It is repeated so often that good engineers feel a flicker of shame when they add a `print` instead of setting a breakpoint. That folklore is wrong, and believing it makes you slower. Print debugging — done deliberately — is not the fallback. In a large and important class of bugs it is the *correct first tool*, the one a breakpoint cannot match: async and event-driven code where stopping the world changes the very timing you are trying to observe; production systems you cannot attach to; tight loops running millions of iterations where stepping is a fantasy; heisenbugs that evaporate the instant you pause them; distributed code spread across processes that have no single call stack to inspect. In all of those, a well-formed line of output beats the microscope, because the microscope's act of looking disturbs the specimen.

This post is about the difference between the print that beginners scatter and the print that a principal engineer wields. The two look superficially the same — both call a function that writes to a stream — but everything else is different. The deliberate print carries a tag you can grep for, the *name* of the variable and not just its value, a timestamp, a thread or request id, and the call site. It dumps a whole object in one cheap call instead of hand-listing five fields. It bisects: a single print at the midpoint of a function halves the region where a value goes wrong, exactly the same binary search you would run with [git bisect](/blog/software-development/debugging/binary-search-your-bug-with-bisection) over commits, but inside one function over one run. It fires *conditionally* — once, on iteration 3,847,221, and never again. And it cleans up after itself: gated behind a debug flag, greppable by a tag before you commit, caught by a linter so a stray print never ships. Figure 1 shows the decision that makes all of this matter — the four contexts where a breakpoint is the wrong instrument and print is the right one.

![A branching flow diagram showing a wrong-value symptom splitting into four contexts where a breakpoint fails and all four routing to a structured print that localizes the bug without shifting timing](/imgs/blogs/print-debugging-done-right-1.png)

By the end you will be able to do five concrete things. Write a structured trace line that survives interleaved output from a dozen threads. Dump a whole object cheaply in Python, Go, JavaScript, C, and Rust. Bisect a wrong value down to one line in about eight prints. Recognize the print that *lies* — the buffered line that never flushed before the crash, the reordered output from threads, the print whose timing masks a race — and confirm it. And do all of it without leaving a single stray print in the codebase. We will stay on the series' spine throughout: observe, reproduce, hypothesize, bisect, fix, prevent. Print debugging is, properly understood, just the cheapest way to *observe* — and the discipline in this post is about making that observation honest.

## 1. Why print is not the beginner's tool

Start with the claim that print is for amateurs, because demolishing it correctly teaches the whole subject.

A debugger's superpower is that it freezes time. You hit a breakpoint and the process stops dead; every local variable, every frame on the call stack, the whole heap, is sitting still for you to inspect. For a huge number of bugs that is exactly what you want and the sibling post [the debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) makes the case for it with full force. But that superpower is also its fatal limitation, and the limitation is not a matter of skill or taste. It is physics. A breakpoint *stops the world*. The moment it does, three things become impossible.

The first impossibility is timing. Enormous categories of modern software are correct only because of *when* things happen relative to each other: the event loop processes callbacks in a particular order; two requests race for the same row; a timeout fires after 30 milliseconds; a buffer drains before it overflows. The instant you stop one thread at a breakpoint, every timing relationship in the program changes. Other threads keep running while the stopped one sits frozen for the seconds or minutes you spend reading variables. The interleaving you were trying to observe — the precise overlap of two operations that produces the bug — will never happen the same way again while a human is in the loop. This is not a small effect. A breakpoint can stretch a 30-microsecond window to a 30-second one. The race you were hunting hides, because you have changed the conditions under which it occurs. We will return to this — it is the heart of the heisenbug worked example later — but the structural point stands: a debugger cannot observe a timing bug without disturbing the timing.

The second impossibility is production. You have a bug that only manifests under real traffic, at real scale, on real data you cannot fully reproduce in staging. To use a debugger you must attach to the running process and set a breakpoint — and when that breakpoint hits, *that process stops serving traffic*. In a service handling thousands of requests per second, a single breakpoint hit is an outage: every in-flight request on that worker hangs, health checks fail, the load balancer evicts the node, and now you have caused a far worse incident than the one you were debugging. You cannot attach gdb to the payments process under load. The bug, meanwhile, is happening *right now* on that exact box, and the only safe way to watch it is to make the program emit a line as it goes.

The third impossibility is scale of iteration. Suppose a value is corrupted somewhere inside a loop that runs five million times. A debugger lets you step, line by line, through one iteration. To reach the iteration where corruption first appears you would have to step through millions of correct ones, or set a breakpoint and continue, continue, continue, millions of times. It is hopeless by hand. You could set a *conditional* breakpoint — and you should, when you can attach — but even an in-debugger conditional breakpoint that fires once at iteration 3.8 million costs you the overhead of evaluating the condition five million times, and if you cannot attach a debugger at all, the loop is utterly opaque to stepping. A single conditional *print* inside the loop, firing exactly once, hands you the answer with almost no overhead.

It is worth making the async case fully concrete, because it is the most counter-intuitive and the one engineers most often get wrong. In a single-threaded event loop — Node.js, Python's `asyncio`, a UI thread — there is exactly *one* thread of execution that runs callbacks to completion, one at a time, pulling the next ready callback off a queue each turn. Correctness depends on the *order* those callbacks run and on *which* callbacks are ready when. Now set a breakpoint inside one callback. The event loop thread is now stopped *inside that callback*, which means it cannot pull the next callback off the queue, which means every timer, every I/O completion, every queued continuation is frozen behind you for as long as you sit at the breakpoint. When you finally continue, the timers that should have fired during your five-second pause all fire at once, in a burst, in an order that the un-paused program would never have produced. A race between two awaited operations that depended on one completing a few milliseconds before the other is now completely scrambled — and not in a way you can reason about, because the scrambling depends on exactly how long you happened to sit at the breakpoint. A print, by contrast, runs *as part of* the callback, takes microseconds, and perturbs the ordering by an amount small enough that the real interleaving still occurs and the trace shows you the true order. This is why a breakpoint in async code does not merely slow things down — it changes the program into a different program, and the bug you were chasing is no longer present to be found.

So the honest framing is not "debugger good, print bad." It is: each instrument owns a set of contexts, and a senior engineer chooses by the bug in front of them. Print owns async and timing-sensitive code, production, tight loops, distributed systems, and any situation where the *act of stopping* would destroy the evidence. The debugger owns the deep single-point inspection — when you are frozen at a fault and want to read forty locals and walk the stack and call functions in the live process. Treating one as a junior version of the other is how you end up using the wrong tool for an hour. The skill is not "graduating" from print to debugger; the skill is knowing which one the bug demands.

## 2. The structured trace print: never print a bare value

Here is the single most common mistake, the one that makes print debugging deserve its bad reputation. An engineer writes:

```python
print(total)
```

Then they run the program, and a number scrolls past in a wall of other output. `1837`. And now they have *three new problems*. Which `total` was that — the function has the variable in four places? Which call produced it — the function ran a thousand times? And in a concurrent program, which thread or request emitted it — there are twelve of them and they are all writing to the same stream? A bare `print(total)` answers one question (the value, this once) and creates three. That is a bad trade, and it is the trade that gives print debugging its amateur smell.

The fix is to treat every diagnostic print as a *structured trace line*. A trace line carries, at minimum, five things, and figure 2 contrasts the bare print against the structured one so you can see exactly what each field buys you.

![A two-column before-and-after diagram contrasting a bare print of a value against a structured trace line carrying a tag, variable name, value, timestamp, and thread id, showing the structured form is greppable and sortable](/imgs/blogs/print-debugging-done-right-2.png)

The five fields are: a **tag** (a unique token like `DBG:accum` you can `grep` for, so your lines are findable in a flood of other output and removable later); the **name** of the variable (so the line is self-documenting — `total=1837`, not a naked `1837`); the **value** itself (printed with a representation that shows type and structure, not a lossy stringification); a **timestamp** (so you can order lines and measure durations even when output interleaves); and a **context id** — the thread name, the request id, the worker pid — so that in a concurrent or distributed program you can tell *whose* line this is. Figure 3 lays out the anatomy of one disciplined line field by field.

![A vertical stack diagram breaking a single structured trace line into its component fields of grep tag, variable name, value, timestamp, context id, and source location with the question each field answers](/imgs/blogs/print-debugging-done-right-3.png)

In Python the language gives you a beautiful shortcut for the name-and-value part. Since 3.8 the f-string `=` specifier prints both:

```python
import threading, time

def trace(tag, **kw):
    t = time.monotonic()
    ctx = threading.current_thread().name
    parts = " ".join(f"{k}={v!r}" for k, v in kw.items())
    print(f"DBG:{tag} t={t:.6f} thread={ctx} {parts}", flush=True)

# self-documenting, name comes for free:
x = compute()
print(f"{x=}", flush=True)            # -> x=1837

# full structured line when it matters:
trace("accum", total=total, i=i)      # -> DBG:accum t=14123.394812 thread=worker-3 total=1837 i=88421
```

Two details in that snippet are load-bearing and easy to miss. First, `{v!r}` uses `repr()` rather than `str()`. The `repr` of a string shows its quotes (`'42'` vs `42` reveals a string-versus-int bug you would otherwise never see), the `repr` of a float shows full precision (so `0.30000000000000004` is visible, not a friendly rounded `0.3`), and the `repr` of `None` is the literal `None` rather than an empty string. The most insidious print bug is a lossy `str()` that hides the very type confusion you are chasing. Always print the `repr`. Second, `flush=True` — we will spend an entire section on why, but the short version is that without it the line you most need may never reach disk before the crash.

The grep tag is what makes a trace line *removable*. Before you commit, you run `git grep "DBG:"` and every diagnostic line in your change is listed for deletion. A consistent, unusual tag turns cleanup from "read every line hoping to spot the prints" into one command. Pick a tag no real code would contain — `DBG:`, `XXXTRACE`, your initials — and use it religiously. The discipline of a tag is the difference between a print you can find and a print that ships to production by accident.

## 3. Dump a whole object cheaply

The second amateur tell is hand-building a print of an object's fields:

```python
print("user id", u.id, "email", u.email, "plan", u.plan, "trial", u.trial_ends)
```

This is slow to write, easy to get wrong (you will forget the one field that mattered), and it does not nest — when `u.plan` is itself an object you are back to typing its fields by hand. Every major language ships a one-liner that dumps a whole object, recursively, with field names and types, and you should reach for it instead. Figure 4 lines them up across five languages.

![A four-column comparison matrix of object-dump primitives across Python, Go, JavaScript, and Rust covering recursive dumps, type-aware output, self-documenting calls, and JSON-safe serialization](/imgs/blogs/print-debugging-done-right-4.png)

In **Python**, `pprint` pretty-prints any nested structure with indentation, and `vars(obj)` returns the instance `__dict__` so `pprint(vars(u))` dumps every attribute. For anything that must survive as a log line or be diffed later, `json.dumps(obj, default=str, indent=2)` serializes arbitrary objects — the `default=str` is the trick that stops it from crashing on a `datetime` or a `Decimal` or a custom class by stringifying anything it does not know how to encode:

```python
from pprint import pprint
import json

pprint(vars(u))                                  # every attribute, indented
print(json.dumps(u.__dict__, default=str, indent=2))  # log-safe, diffable
```

In **Go**, the `fmt` verbs are the whole toolkit. `%v` prints a value, `%+v` adds the *field names* (so a struct prints as `{ID:7 Email:a@b.com Plan:pro}` instead of `{7 a@b.com pro}`), and `%#v` prints the full Go-syntax representation including the type name — invaluable when you cannot tell a `nil` slice from an empty one or an `int32` from an `int64`:

```go
log.Printf("user %+v", u)   // {ID:7 Email:a@b.com Plan:pro Trial:2026-07-01}
log.Printf("user %#v", u)   // main.User{ID:7, Email:"a@b.com", Plan:"pro", ...}
```

The difference between `%v` and `%+v` in Go is the single highest-value print habit you can build in that language. The bare `%v` of a struct is a tuple of values with no names; under any real struct it is unreadable. `%+v` costs one character and makes the line self-documenting. Make it a reflex.

In **JavaScript / Node**, `console.log` stops at a shallow depth and prints `[Object]` for anything nested — which is exactly the field you wanted. `console.dir(obj, { depth: null })` prints the whole tree, and `console.table(rows)` renders an array of objects as an actual table, which is the fastest way to eyeball a list of records:

```javascript
console.dir(order, { depth: null });   // full nested tree, no [Object] truncation
console.table(items);                  // array of objects as a real table
console.log(JSON.stringify(order, null, 2));  // serialized, indented
```

In **C**, there is no reflection, so you write a small dump helper once per struct and call it — and you write to `stderr`, not `stdout`, for reasons we will make precise in the flush section:

```c
void dump_conn(const struct conn *c) {
    fprintf(stderr, "DBG conn fd=%d state=%d retries=%d buf=%zu\n",
            c->fd, c->state, c->retries, c->buf_len);
    fflush(stderr);   /* stderr is usually unbuffered, but be explicit */
}
```

In **Rust**, the `dbg!` macro is the gold standard of a self-documenting print: `dbg!(&x)` prints the *file, line, the expression as source text, and the value* using the `Debug` formatter, then returns the value so you can wrap it inline without restructuring code. And the `{:#?}` format specifier pretty-prints any type that derives `Debug`:

```rust
let total = dbg!(items.iter().map(|i| i.price).sum::<u64>());
// [src/cart.rs:42:17] items.iter().map(|i| i.price).sum::<u64>() = 1837
println!("{order:#?}");   // pretty, multi-line Debug dump
```

`dbg!` is what every other language's print should aspire to: it tells you *where* it fired, *what expression* produced the value, and the value, in one call, and it is transparent enough that you can drop it into the middle of an expression without changing the program's structure. When you reach for a print in Rust, reach for `dbg!`.

The lesson across all five languages is the same. You should never be hand-listing fields. One call dumps the object with names and types; that is faster to write, complete by construction, and far less likely to hide the field you needed.

## 4. Bisecting a value with prints

Now the technique that turns print from a guess into a method, and the one that ties this post directly to the series' spine. You have a function — say 200 lines — and you know that a value is *correct when it enters* and *wrong by the time it leaves*. Somewhere in those 200 lines it goes bad. The amateur reads all 200 lines hoping to spot it. The disciplined engineer *bisects*: put one print at the midpoint of the function. If the value is still correct there, the bug is in the second half; if it is already wrong, the bug is in the first half. Either way, one print has *halved* the suspect region. Repeat, and a 200-line function is localized to one line in about $\log_2(200) \approx 8$ prints. Figure 5 walks the narrowing.

![A left-to-right timeline showing a value bisection where prints at the midpoint of successively halved regions narrow a wrong accumulator from a 200-line function down to a single off-by-one line](/imgs/blogs/print-debugging-done-right-5.png)

This is the exact same binary search as `git bisect`, but the search space is *positions inside one run* rather than commits in history. The sibling post on [bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) develops the commit-search version in depth; here is the in-function version, made concrete:

```python
def settle_invoice(inv):
    total = 0
    # ... 50 lines of setup ...
    for line in inv.items:
        total += line.qty * line.unit_price
    print(f"DBG:mid1 {total=}", flush=True)   # midpoint: line ~100
    # ... 50 lines of discounts ...
    total = apply_tax(total, inv.region)
    # ... 50 lines of rounding/fees ...
    print(f"DBG:mid2 {total=}", flush=True)   # 3/4 point: line ~150
    return total
```

Run it once. Suppose `mid1` shows `total=1837` (correct, the expected pre-tax subtotal) and `mid2` shows `total=2104` (wrong — you expected `2020`). The bug is in the ~50 lines between them, the tax-and-fees region. Move both prints into that region's quarter points and run again. Two runs, four prints, and you have gone from "somewhere in 200 lines" to "somewhere in 12 lines." Two more prints and you are on the line: `apply_tax` is multiplying by `1.0 + rate` but `rate` arrived as a percentage (`8`, not `0.08`), so an `apply_tax` that should add 8% is adding 800%. Found in three runs and roughly eight prints, without reading the bulk of the function.

The reason this is powerful is the same reason binary search is powerful: each probe carries *one bit* of information (first half or second half), and a 200-line function needs only about eight bits to pin a line. Reading the function linearly extracts information one line at a time — 200 reads in the worst case. Bisection extracts it eight probes at a time. When the function is long, or the logic is dense and easy to misread, or the bad value is produced by a subtle interaction you would not spot by eye, bisection wins decisively. It also works when you *cannot* attach a debugger at all — it is just prints — which is precisely the production and tight-loop case from section 1.

#### Worked example: which of 5 million iterations corrupts the accumulator

Here is the canonical tight-loop bug, with real numbers. A batch job sums 5,000,000 transactions into a running `int64` accumulator and produces a total that is wrong by a small, weird amount — off by exactly `4,294,967,296` on some runs, which to a trained eye is suspicious: that number is $2^{32}$, the size of a 32-bit overflow. Hypothesis: one transaction's amount is being read or cast as a 32-bit integer somewhere, and on iterations where the partial sum crosses a 32-bit boundary the high bits are lost. But which iteration? Stepping five million times in a debugger is hopeless, and even a conditional breakpoint must evaluate its predicate five million times.

The move is a *conditional print that fires only when the invariant breaks*, plus a one-shot guard so it prints once and then goes quiet:

```python
prev = 0
fired = False
for i, tx in enumerate(transactions):
    total += tx.amount
    # invariant: a correct running int64 sum never goes DOWN when amounts are positive
    if total < prev and not fired:
        print(f"DBG:overflow first drop at i={i} "
              f"prev={prev} total={total} amount={tx.amount!r} "
              f"type={type(tx.amount).__name__}", flush=True)
        fired = True
    prev = total
```

Run once. The line fires exactly once:

```bash
DBG:overflow first drop at i=3847221 prev=2147480000 total=-2147483648 amount=12345 type=numpy.int32
```

There it is, in a single run with one print's worth of output: iteration 3,847,221, the running sum had climbed to just under $2^{31}$, and `amount` is a `numpy.int32` — the transactions were loaded into a NumPy array with the default 32-bit dtype, and the running sum wrapped around the 32-bit signed boundary. The print did three things a debugger could not do cheaply here: it scanned five million iterations at full speed, it fired *only* on the one that mattered (the `not fired` guard kept it from flooding the next ten thousand iterations once the sum stayed negative), and it captured the *type* of the offending value, which was the actual root cause. The conditional print plus a one-shot guard is the single most useful pattern for tight-loop bugs. The fix — load the column as `int64` — is one line; finding it was the whole job, and one print did it.

## 5. Is this line even reached?

A separate and embarrassingly common bug is not a wrong value but a wrong *path*: a branch you were certain executed never runs, or the one you were certain was dead runs every time. An exception handler that "can't be hit" is swallowing errors. A cache check that you assumed was a hit is silently always a miss. A feature flag you think is on is off. For this you do not print a value; you print *presence*. The crudest, most honest debugging line in the world is:

```python
print("DBG: HERE 7", flush=True)
```

Sprinkle numbered `HERE` markers down a function, run it, and read which numbers appear and in what order. The ones that *don't* appear tell you which branches never ran — and a branch that never runs is frequently the whole bug. This is reachability debugging, and it answers a question values cannot: *did control flow get here at all?* When a function "isn't doing anything," it is usually because an early return, a swallowed exception, or a guard clause is bailing out before the code you are staring at ever runs, and a `HERE` marker at the top of each block finds it in one run.

The structured version tags the marker and includes context so it works under concurrency:

```python
def trace_here(n):
    import threading
    print(f"DBG:HERE {n} thread={threading.current_thread().name}", flush=True)

def handle(req):
    trace_here(1)
    if not req.valid():
        trace_here(2)          # if this prints, validation is rejecting it
        return error()
    trace_here(3)
    result = cache.get(req.key)
    if result is None:
        trace_here(4)          # if this ALWAYS prints, the cache never hits
        result = compute(req)
    trace_here(5)
    return result
```

Run this against the bug and read the markers. If you see `1, 3, 4, 5` on every single request and *never* a marker showing a cache hit, your "cache" has a key-mismatch bug — it is storing under one key and reading under another, so it is a 0% hit-rate cache wearing a cache's costume. The reachability print found in one run what could otherwise be hours of staring: the code is correct, the *keys* are wrong. Reachability prints are also how you confirm the *negative* — proving a line is dead is as valuable as proving one is hit, and a debugger makes the negative awkward to establish (you would have to convince yourself the breakpoint *never* fired across thousands of requests) while a `HERE` marker's absence states it plainly.

Figure 6 organizes all of these moves — trace, reachability, selective, and dump — into the family they actually form, so you can pick the right one by the question you are asking rather than reaching for a bare print by reflex.

![A taxonomy tree of print-debugging techniques branching into trace, reachability, selective, and dump families with concrete leaf techniques like greppable trace lines, HERE markers, conditional one-shot prints, and deep object dumps](/imgs/blogs/print-debugging-done-right-6.png)

## 6. Conditional and one-shot prints

We met the conditional print in the loop worked example; it deserves its own treatment because it is what makes print usable in code that runs at scale. The principle: a print is only useful if you can *read* its output, and output you cannot read is noise. So you gate the print so it fires only in the situations you care about.

The simplest gate is a value condition — only print when `i == target`, or only when a field is in the suspicious range:

```python
if order.id == "ord_91af" or amount > 1_000_000:
    trace("suspect", order=order, amount=amount, path=request_path)
```

The second gate is a **one-shot** guard, so a condition that becomes permanently true does not flood the log. We used `fired = True` above. A reusable version:

```python
_seen = set()
def trace_once(key, **kw):
    if key in _seen:
        return
    _seen.add(key)
    trace(f"once:{key}", **kw)

# fires the first time each distinct error code appears, then never again:
trace_once(f"err-{code}", code=code, req=req.id)
```

The third gate is **rate limiting** — print at most once per N events or once per second — for a high-frequency event where you want a *sample*, not every occurrence:

```python
import time
_last = {}
def trace_rate(key, every=1.0, **kw):
    now = time.monotonic()
    if now - _last.get(key, 0) >= every:
        _last[key] = now
        trace(f"rl:{key}", **kw)
```

The fourth gate, and the most important for production, is a **debug flag** — the print exists in the code permanently but only emits when an environment variable or config flag turns it on:

```python
import os
DEBUG = os.environ.get("APP_DEBUG", "").lower() in ("1", "true", "yes")
def dbg(*a, **kw):
    if DEBUG:
        print("DBG:", *a, **{**kw, "flush": True})

# in code:
dbg(f"{total=} {region=}")     # silent in prod, on with APP_DEBUG=1
```

This is the bridge from print debugging to real logging, and it is the single most important discipline in this entire post for not leaving a mess. A flag-gated print is not a temporary hack you must remember to delete; it is a *permanent diagnostic* that costs one boolean check when off and gives you a trace when on. Now you can turn diagnostics on for one production pod, or one request, without redeploying and without slowing everyone down. The next step up is to stop hand-rolling `dbg` and use the standard library's `logging` (Python), `log/slog` (Go), or a structured logger (JS `pino`, Rust `tracing`), with the diagnostic lines at `DEBUG` level so they are filterable, leveled, and removable as a class. The sibling post on logging as a debugging instrument — a deliberate, leveled, structured logging discipline rather than ad-hoc prints — picks up exactly here; that post is the natural next read once your prints have outgrown being temporary.

In Python specifically, the upgrade path has a beautiful waypoint: when a print is not enough and you want to actually stop and poke around, `breakpoint()` drops you into `pdb` at that exact line — and it too respects an environment variable (`PYTHONBREAKPOINT=0` disables every `breakpoint()` call in the process, or you can point it at `ipdb` or a remote debugger). So the progression is `print` → flag-gated `logging.debug` → `breakpoint()` when you finally do want to freeze and inspect. Each step is a deliberate upgrade, not an admission that the previous tool was childish.

## 7. The print that lies

Now the dangerous part, the reason print debugging earns a *different* kind of distrust among people who have been burned: sometimes the output is *wrong*. Not incomplete — actively misleading. The log tells you execution stopped at line 8 when it actually died at line 10. The output shows events in an order that never happened. The bug vanishes the instant you add the print. A senior engineer knows the three ways print lies and how to defeat each one. Figure 7 shows the most common lie — buffered output — in before-and-after form.

![A two-column before-and-after diagram showing buffered stdout holding the final prints in memory so a crash drops them and the log blames the wrong line, versus a flushed or stderr write where every line reaches disk before the next statement](/imgs/blogs/print-debugging-done-right-7.png)

### Lie #1: the buffer that never flushed

This is the most common and the most damaging. Standard output is *buffered*. When a program's stdout is connected to a pipe or a file (as it almost always is in production — your logs go to a file or a log collector, not a terminal), the C runtime and the language runtime collect your printed bytes in an in-memory buffer, typically 4 or 8 kilobytes, and only write them to the actual file descriptor when the buffer fills or the program exits *cleanly*. The reason is performance: a `write()` syscall per print would be ruinously slow, so the runtime batches them.

Here is the trap. Your program prints "step 9: about to call risky()" and then `risky()` *crashes the process* — a segfault, an `abort()`, an `os._exit`, an OOM kill, a `kill -9`. The buffer holding "step 9" was never flushed, because the program did not exit cleanly; it was killed. So your log ends at "step 8," and you spend an hour convinced the bug is near step 8, when it is actually at step 10. The output *lied about where the program was* — not because the print was wrong, but because the line you needed died in a buffer in memory that the crash discarded. This is the mechanism, and it is worth stating precisely because it explains a whole genre of "the logs make no sense" confusion: buffered output is only guaranteed to reach disk on a *clean* exit, and the bugs you are chasing are, by definition, the ones that do not exit cleanly.

There is a deeper layer to this mechanism, and it is the part that catches even experienced engineers: *the buffering mode depends on what the stream is connected to, and it changes between your laptop and production.* The C standard library — which every language's runtime ultimately sits on top of for I/O — chooses a buffering mode at the first write to a stream by asking the operating system whether the file descriptor is a terminal, via `isatty()`. If it is a terminal (you running the program in your shell), stdout is *line-buffered*: every newline flushes, so on your laptop every print appears immediately and you never see the lie. If it is *not* a terminal (a pipe to a log collector, a redirect to a file, a container's captured stdout — which is to say, production), stdout is *fully buffered*: it accumulates 4 or 8 kilobytes before a single `write()`, and nothing reaches disk until the buffer fills or the program exits cleanly. This is the precise reason the buffered-log lie is a *production-only* bug that you cannot reproduce on your laptop: the same code, the same prints, behave differently because the stream's buffering mode silently switched when stdout stopped being a terminal. An engineer who does not know this will run the program locally, see every line appear in order, conclude the logging is fine, and be baffled when production's log ends in the wrong place. The fix works precisely because it overrides this automatic choice: `flush=True` forces a `write()` regardless of mode, stderr is unbuffered by the same standard regardless of whether it is a terminal, and `stdbuf`/`-u`/`PYTHONUNBUFFERED` override the `isatty()`-driven default outright.

The fixes, in order of preference:

```python
# 1. Flush every diagnostic print (Python 3.3+):
print("step 9: calling risky()", flush=True)

# 2. Write to stderr, which is unbuffered by default in C and line-buffered in Python:
import sys
print("step 9", file=sys.stderr)

# 3. Make the whole stream unbuffered for the run:
#    $ python -u script.py          (Python: -u = unbuffered)
#    $ PYTHONUNBUFFERED=1 python ... (same, via env)
#    $ stdbuf -oL -eL ./program      (line-buffer any program's stdout/stderr)
```

In **C** the same trap and the same fix: `stdout` is fully buffered when not a terminal; `stderr` is unbuffered. So `fprintf(stderr, ...)` survives a crash that `printf(...)` would lose, and after any `printf` you actually need before a possible crash, call `fflush(stdout)` (or `setvbuf(stdout, NULL, _IONBF, 0)` once at startup to disable buffering entirely while debugging). In **Go**, `fmt.Println` writes to `os.Stdout` which is unbuffered at the `os.File` level — but if you wrapped it in a `bufio.Writer` for speed, you must `Flush()` or a crash loses the tail. The rule across every language is the same: **if the line might be the last thing before a crash, flush it or send it to stderr.** When in doubt during a crash investigation, send every diagnostic to stderr and flush; the tiny performance cost is irrelevant next to a misleading log.

#### Worked example: the log that blamed the wrong line

A service crashes intermittently and the logs always end with the same line: `processing batch, size=500`. Every engineer who looks at it concludes the crash is *inside* batch processing, near that line, and spends days reading that code. It is clean. The real story: the next thing the program does after that print is call a C extension that, on a particular malformed input, segfaults. The "size=500" print and the next several diagnostic prints were sitting in stdout's 8 KB buffer; the segfault killed the process; the buffer was discarded; the log ended at the last line that *happened* to get flushed when an earlier buffer filled. The crash was 40 lines of execution *past* where the log stopped.

The confirmation was one change: run with `PYTHONUNBUFFERED=1` (or add `flush=True` to the diagnostic prints around the suspect region). Suddenly the log ended with `calling parse_ext(buf), len=4096` — a line that had never appeared before — and the segfault was localized to the C extension call in one run. Before: days lost to a log that pointed 40 lines too early. After: root cause in one unbuffered run. The bug was never where the log said it was, because the log was lying, and the lie had a precise, knowable cause.

### Lie #2: reordered, interleaved output from threads

When several threads print to the same stream, two distortions appear. First, **interleaving**: thread A prints half its line, thread B's line cuts in, and you get a garbled hybrid — because a single `print` may be more than one `write()`, and the OS can switch threads between them. Second, and subtler, **apparent reordering**: even if each line is atomic, the order lines appear in the log is the order they reached the *buffer*, not necessarily the order the *events* happened, because each thread may have its own buffering and the timestamps you did not include would have revealed the true order. If you read a threaded log top-to-bottom and assume that is the order of events, you can convince yourself of a causal story that is exactly backwards.

The defenses are the structured-trace fields from section 2, now earning their keep. **Timestamp every line** with a high-resolution monotonic clock so you can sort by true time rather than trusting file order. **Tag every line with the thread or request id** so you can `grep` one thread's lines out of the interleaved mess and read that thread's story in isolation. And **make each line a single atomic write** — build the whole line as one string and emit it in one call, or route through a logging framework whose handler holds a lock so lines never split. In Python, `logging` acquires a lock per emit, which is one concrete reason to graduate from raw `print` to `logging.debug` the moment more than one thread is involved: it makes each line atomic for free. Once your lines carry a timestamp and a thread id, an interleaved log stops lying — you sort by time, filter by thread, and reconstruct the true sequence.

### Lie #3: the print that changes the timing and hides the bug

This is the heisenbug, the most unsettling lie of all: you add a print to diagnose an intermittent failure, and the failure *stops happening*. Remove the print, it comes back. The print is "fixing" the bug. It is not, of course — it is *masking* it, by changing the program's timing just enough to make a race lose its window. This is the precise inverse of the breakpoint problem from section 1: a breakpoint slows a thread so much the race never overlaps; a print slows it a *little*, which can be exactly enough to move two operations out of their dangerous overlap.

#### Worked example: works with logging, breaks without

A cache layer occasionally returns stale data — about 6 times in 2,000 runs of the stress test, a 0.3% failure rate. An engineer adds `print(f"DBG cache write {key}", flush=True)` inside the write path to watch it, runs the stress test 2,000 times, and gets *zero* failures. Triumphant, they conclude the print revealed nothing because the bug "doesn't happen here." Then they remove the print to ship, and the 0.3% failure rate returns in production. The print was the fix. The print was hiding the bug.

What is actually going on is a data race on the cache entry: two threads — a writer updating the entry and a reader serving it — touch the same memory with no synchronization (no lock, no `happens-before` edge from a mutex or atomic), so the reader can observe a half-updated entry. Why does that even cause a bug? Because without a synchronization edge, the compiler and CPU are free to reorder the writer's stores: it may publish the entry's *pointer* before it has finished writing the entry's *fields*, and a reader that sees the new pointer then reads garbage fields. The `print`, with its `flush=True`, performs a `write()` syscall — and a syscall is a synchronization point with the kernel and a yield to the scheduler. That syscall slows the writer thread by a few microseconds at exactly the moment that, in the un-printed version, the reader was slipping into the gap. The print closes the window. The bug is still a bug; it is just no longer *observable* with that print in place.

How do you confirm a heisenbug is a timing race and not something the print legitimately fixed? Three moves, in order. First, **swap the print for a workload that perturbs timing without changing logic** — insert a `time.sleep(0.0001)` or a busy-spin at the same spot; if a no-op delay also makes the bug vanish, the cause is timing, not the print's content. Second, **run a thread sanitizer**, which detects the race *directly* by tracking happens-before relationships rather than by waiting for the bad interleaving to manifest. In Go that is the built-in detector:

```bash
go test -race ./cache/...     # reports the exact two stacks racing on the entry
```

In C, C++, and Rust it is `-fsanitize=thread`:

```bash
clang -fsanitize=thread -g cache.c -o cache && ./cache
# ThreadSanitizer: data race on entry->value
#   Write of size 8 by thread T2 ... cache.c:88
#   Previous read of size 8 by thread T1 ... cache.c:54
```

ThreadSanitizer finds the race even on runs where the bug does *not* manifest, because it does not need the dangerous interleaving to happen — it reasons about the *absence of a synchronization edge* between the two accesses. That is the decisive confirmation: a tool that proves the race exists regardless of timing, immune to the very heisenbug effect that makes the print lie. Third, once confirmed, **fix the synchronization** (a mutex around the entry, or an atomic publish with release/acquire ordering), and re-run the stress test: 0 failures in 2,000 runs *with the print removed*, because the bug is actually gone rather than merely hidden. Before: 6/2,000 failures, masked to 0/2,000 by a print that fixed nothing. After: a real synchronization edge, 0/2,000 with no print, confirmed by `-race`. The print did not solve the bug — but the fact that the print *masked* it was itself the clue that pointed straight at a timing race, and that is a piece of evidence a debugger could never have given you.

## 8. A short language tour

Pulling the language-specific moves together, because in practice you are debugging in whatever language the bug lives in, and each has its own idiomatic print.

**Python.** The f-string `=` specifier (`print(f"{x=}")`) gives you self-documenting name-and-value for free; always add `flush=True` for crash-survival; use `pprint`/`json.dumps(default=str)` to dump objects; gate diagnostics behind `logging.debug` with a level so they are filterable and removable as a class; and treat `breakpoint()` as the upgrade when a print is no longer enough and you want to freeze and inspect. The standard library's `faulthandler` (`python -X faulthandler`) is the natural partner: it prints a Python traceback on a segfault or fatal signal, recovering the stack a buffered print would have lost.

```python
import logging, faulthandler, sys
faulthandler.enable()                         # traceback on hard crash
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr,
                    format="%(asctime)s %(threadName)s %(message)s")
logging.debug("total=%r region=%r", total, region)   # leveled, atomic, greppable
```

**JavaScript / Node.** `console.log` for quick values, `console.dir(obj, {depth: null})` to dump deep objects, `console.table(rows)` for arrays of records, `console.trace("got here")` to print a stack trace *without* throwing (the fastest way to answer "who called this?"), and the `console.time(label)` / `console.timeEnd(label)` pair to measure how long a span took without a profiler:

```javascript
console.time("query");
const rows = await db.query(sql);
console.timeEnd("query");          // query: 412.83ms
console.trace("unexpected null user");   // prints the call stack, keeps running
```

The JavaScript trap to know is that `console.log` of an object logs *a live reference*, and many runtimes render that reference lazily when you expand it in the devtools console — so if you mutate the object after the log, you may see the *mutated* state, not the state at the moment of the log. When you need a true snapshot of an object's state at a point in time, log a deep copy: `console.log(structuredClone(obj))` or `console.log(JSON.parse(JSON.stringify(obj)))` freezes the values as they were. This "the log shows the wrong state" surprise is a close cousin of the buffered-log lie — both are cases where the output does not reflect the moment you think it does — and the fix in both is to capture the value eagerly rather than trust a deferred read.

**Go.** `log.Printf("%+v", x)` for field-named struct dumps, `%#v` when you need the type, and the standard logger writes to stderr with a timestamp by default. For structured, leveled output that survives into production, `log/slog` (Go 1.21+) gives you key-value pairs that are greppable and machine-parseable:

```go
slog.Debug("settled invoice", "total", total, "region", inv.Region, "id", inv.ID)
```

Go has one print-debugging trap worth calling out specifically because it bites people coming from other languages: a `%+v` on a struct does *not* recurse into pointer fields by default — a nested `*Address` prints as a bare hex pointer like `0xc000123450` rather than the address's contents, which is exactly the field you wanted. The fix is to print the dereferenced value, or to reach for a deep-printer like the `davecgh/go-spew` library's `spew.Dump(x)`, which walks pointers and prints the whole graph with types. And because Go's race detector is built into the toolchain and essentially free to run, the idiomatic Go debugging loop for a suspected concurrency bug is to narrow with a few `slog.Debug` lines and then immediately run `go test -race ./...`; the detector will point at the exact two goroutines and the exact field they raced on, which is the confirmation step from section 7 made trivial.

**C.** `fprintf(stderr, ...)` with an explicit `fflush(stderr)` (or `setvbuf` to disable buffering during a debug session), because `stdout` is fully buffered to a pipe and will eat your last lines on a crash. Pair it with a sanitizer build (`-fsanitize=address` for memory bugs, `-fsanitize=thread` for races) so the print narrows the region and the sanitizer names the exact fault.

**Rust.** `dbg!(&x)` for a self-documenting print that includes file, line, and the source expression; `{:#?}` for pretty `Debug` dumps; and `eprintln!` to write to stderr. For anything beyond throwaway, the `tracing` crate gives spans and structured fields that work the same in a test and in production.

The recurring theme is that every language has, somewhere, a print that already does the disciplined thing — carries the name, dumps the structure, marks the site, survives a crash — and learning it is a five-minute investment that pays out on every bug for the rest of your career.

## 9. How to not leave a mess

A print's worst failure mode is not that it is imprecise; it is that it *ships*. A `print(user.password)` left in a login handler is a security incident. A debug print in a hot loop is a performance regression. A wall of `DBG` lines in production logs is noise that buries the signal during the *next* incident. So the discipline of removal is as important as the discipline of the print itself, and it has four layers.

**Layer one: the grep tag.** Every diagnostic print carries a consistent, unusual token — `DBG:`, `XXXTRACE`, your initials — so `git grep "DBG:"` before you commit lists every diagnostic line in the change for deletion. This is why section 2 insisted on a tag. A tagged print is findable; an untagged `print(x)` blends into legitimate output and survives.

**Layer two: the debug flag.** The prints you want to *keep* — the ones that will help during the next incident — should not be deleted at all; they should be gated behind a flag or moved to `logging.debug`, so they exist permanently but emit only when turned on. The decision per print is binary: temporary (tagged, deleted before commit) or permanent (gated, leveled, kept). There is no third category of "untagged print I'll remember to remove," because you will not remember.

**Layer three: the linter.** Configure your linter to reject stray prints. Python's `flake8` with `flake8-print` flags every `print` and `pprint` call. ESLint's `no-console` rule flags `console.log`. Go's `vet` and `staticcheck` catch leftover debug constructs. Rust's clippy warns on `dbg!` (it is `clippy::dbg_macro`) precisely so a `dbg!` never reaches `main`. These run in CI and fail the build, which means a stray print cannot be merged even if every human reviewer misses it:

```bash
# CI fails the build if any debug print survives:
flake8 --select=T20 .            # T20x = flake8-print: no print / pprint
eslint --rule 'no-console: error' src/
cargo clippy -- -D clippy::dbg_macro
```

**Layer four: structured logging instead of raw print.** The deepest fix is to not use raw `print` for diagnostics at all, but a logging framework with levels. A `logging.debug(...)` line is *filterable* (off by default in production, on by raising the level), *removable as a class* (you can strip an entire level), *atomic* (the handler locks), and *greppable* (structured fields). It is everything a disciplined print is, made permanent and safe. The progression of maturity is: bare print → tagged print → flag-gated print → `logging.debug`. By the time you reach `logging.debug` you have a diagnostic that helps you today *and* during the incident six months from now, with no mess and no cleanup, because it was designed to stay.

The combined effect is that a print never reaches production by accident. The tag makes it findable, the flag makes the keepers safe, the linter makes a stray one un-mergeable, and structured logging makes the whole category permanent and filterable. That is the difference between print debugging that earns a bad reputation and print debugging that a staff engineer is happy to see in a diff.

## 10. Distributed print debugging: the correlation id

The hardest context in section 1's list — and the one where print's advantage over a debugger is most absolute — is the distributed bug: a request enters an API gateway, fans out to a cart service, which calls a pricing service, which reads from a cache and a database, and *somewhere* across those four processes on four machines the answer comes back wrong. There is no single call stack to freeze. There is no process to attach a debugger to that would show you the whole story, because the story spans four address spaces that share no memory. You could attach four debuggers, but the moment you stop one process at a breakpoint the request times out, the upstream retries, and the interleaving you were chasing is gone — the production-and-timing problems from section 1 compound. This is print's home turf, and the technique that makes it work is the **correlation id**.

A correlation id (also called a trace id or request id) is a single unique token — a UUID, or a short random string — generated once at the edge of the system when a request first arrives, then *threaded through every call* so that every log line emitted anywhere in the system, on any machine, for that one request carries the same id. Now your distributed log is not a hopeless interleaving of thousands of concurrent requests' lines; it is filterable. You `grep` the one id, across all four services' logs (or query it in your log aggregator), and you get *that single request's story*, in order, across every process it touched. The bug that was invisible because it was scattered across four machines becomes a single readable trace.

The mechanism that makes this work is propagation: the id must travel with the request. In an HTTP system it rides in a header — by convention `X-Request-Id` or, under the W3C Trace Context standard, `traceparent` — and every service reads it from the incoming request and writes it on every outgoing call:

```python
import uuid, logging, contextvars

# one context variable holds the id for the duration of a request,
# even across async awaits, so every log line can reach it:
request_id = contextvars.ContextVar("request_id", default="-")

def with_correlation(handler):
    def wrapper(req):
        # read from upstream, or mint a new one at the edge:
        rid = req.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        request_id.set(rid)
        return handler(req)
    return wrapper

class CorrelationFilter(logging.Filter):
    def filter(self, record):
        record.rid = request_id.get()   # inject the id into every record
        return True

# every log line now carries rid=... with zero per-call effort:
logging.basicConfig(
    format="%(asctime)s rid=%(rid)s %(threadName)s %(message)s")
logging.getLogger().addFilter(CorrelationFilter())

def call_pricing(item):
    headers = {"X-Request-Id": request_id.get()}   # propagate downstream
    return http.post("http://pricing/quote", json=item, headers=headers)
```

Notice two design choices that matter. First, the id is stored in a `contextvars.ContextVar`, not a thread-local — context variables survive across `await` points in async code, which a thread-local does not, so the same machinery works in a synchronous thread-pool server and an async event-loop server without change. Second, the id is injected by a *logging filter*, so every existing log line in the codebase automatically gains `rid=...` without any author having to remember to add it; the diagnostic discipline is built into the logging configuration, not into every call site. That is the difference between a correlation scheme that works and one that has gaps: it must be automatic, or someone will forget the one line that mattered.

#### Worked example: the duplicate charge across three services

Back to the opening scenario — checkout double-charges roughly one customer in fifty thousand. It cannot be reproduced locally; it only happens under production concurrency; and a debugger is useless because the bug is spread across the gateway, the order service, and the payment service. The approach is pure correlated print debugging. Every service already logs at `INFO` with a correlation id, so when a customer reports a double charge, support hands over the two charge ids, and a single query — `rid=<that request's id>` — pulls every line that request produced across all three services, in time order.

The trace tells the story in about twenty lines: the gateway received *one* checkout request; the order service logged `creating order` *once*; but the payment service logged `charging card` *twice*, 1.8 seconds apart, with the same correlation id. Between those two charge lines, the order service logged `timeout calling payment, retrying`. There it is: the payment call took longer than the order service's 1.5-second timeout, the order service retried, but the *first* call had already succeeded on the payment service's side — the payment service is not idempotent, so the retry charged the card a second time. No single-process debugger could have seen this, because the bug is the *interaction* of a timeout in one service with a non-idempotent handler in another, visible only when you line up both services' logs for the one request. The fix is an idempotency key on the charge so a retry with the same key is a no-op; the diagnosis was entirely a matter of threading a correlation id through and grepping for it. This pattern — timeout plus non-idempotent retry producing a duplicate — is common enough that the message-queue and microservices material treats idempotency as a first-class concern; the point here is that the *diagnosis* of such a cross-service bug is a print-and-correlate job, not a debugger job, and the correlation id is what makes the scattered prints into one readable trace. This is also exactly where structured, distributed tracing (OpenTelemetry, a `traceparent` header, a tracing backend) is the productionized form of the same idea — the correlation id grown up into spans with timing — which the system-design observability material develops in full.

The takeaway generalizes past this one bug: in any system spanning more than one process, the correlation id is the single most valuable piece of print discipline you can adopt, because it is the only thing that turns a multi-machine flood of interleaved output back into one request's coherent story. Adopt it before you need it; threading an id through after an incident has started is far harder than having it there all along.

## 11. Print vs debugger vs structured logging

It is worth being decisive about which instrument to reach for, because the three are not competitors but specialists, and reaching for the wrong one by habit is a real cost. Figure 8 lays out the trade-off as a matrix you can consult mid-incident.

![A comparison matrix scoring print, the interactive debugger, and structured logging across async timing-sensitive bugs, tight loops, live production, deep one-time inspection, and how much cleanup each leaves behind](/imgs/blogs/print-debugging-done-right-8.png)

Here is the same trade-off as a table you can keep:

| Context | print / trace | interactive debugger | structured logging |
| --- | --- | --- | --- |
| Async / event-driven, timing matters | **Best** — no timing shift | A breakpoint changes the timing you're observing | Good — async-safe, no pause |
| Tight loop, millions of iterations | **Best** — gated, fires once | Stepping is hopeless; conditional bp evaluates N times | Sampled at a level |
| Live production, can't attach | Good — if flag-gated | **Cannot** — a breakpoint hit is an outage | **Best** — built for prod |
| Deep one-time inspection at a fault | Dump one object | **Best** — all locals, walk the stack, call functions | Log the struct |
| Distributed, across processes | **Best** — correlation id ties it together | No single stack to inspect | **Best** — traces span services |
| How much mess it leaves | Tag + grep + linter to clean | Nothing to clean | Stays, but filtered by level |
| Overhead when off | Zero if flag-gated | N/A | Near-zero at higher level |

Read it as ownership, not ranking. The debugger *owns* deep single-point inspection — frozen at a fault, forty locals, walk the stack, evaluate expressions in the live process; nothing else comes close and the microscope post is the place to master it. Structured logging *owns* production and distributed systems — it is async-safe, leveled, filterable, and a correlation id stitches one request's story across a dozen services. Print *owns* the contexts in between and the contexts the other two cannot reach: timing-sensitive code where any pause changes the answer, tight loops where stepping is a fantasy, the quick one-shot dump, and the situation where you need an answer *right now* with no setup. The mature instinct is to read the bug, not the habit: a timing race wants a flushed trace line and a sanitizer, not a breakpoint; a deep "why is this object in this state" wants the debugger; a "one customer in fifty thousand, in production" wants structured logging with a correlation id. Reaching for the wrong one is not a moral failing, but it is an hour you did not need to spend.

## 12. War story: when print was the only tool that could see

Some of the most consequential bugs in computing history were timing bugs in systems where stopping the world was either impossible or itself catastrophic — exactly the territory where print-style tracing, not breakpoints, is the only viable instrument.

Consider the **Therac-25** radiation therapy machine of the mid-1980s, whose software race condition delivered massive radiation overdoses to several patients. The fatal bug only manifested when an experienced operator typed the treatment setup *quickly* — fast enough that a data-entry edit and a concurrent setup routine interleaved in a window that slow typing never opened. This is the purest possible example of a timing-dependent race, and it is also the purest example of why a breakpoint is useless against such a bug: the failure existed *only* at full operator speed, and any tool that paused execution would have closed the very window that caused harm. The investigation that finally characterized it depended on tracing what the software actually did, in order, as it ran at speed — the discipline this post is about. The accuracy point matters here: the Therac-25 case is documented and the timing-race characterization is well established in the safety-engineering literature; it stands as the canonical lesson that races hide from any instrument that changes timing, and that ordered traces of a running system are sometimes the only way to see them.

Or consider the everyday but instructive case of the **buffered-log production crash**, which is not one famous incident but a pattern that recurs in essentially every large system's history: a service crashes, the logs end at a line that turns out to be innocent, and an entire team chases the wrong code for days because the real failing line died unflushed in a buffer. The Knight Capital trading loss of 2012 — which destroyed the company in 45 minutes through a deployment that left old code active on one of eight servers — was in part a story about *observability under speed*: the failing path was emitting signals that, in the chaos of a live market open, were not surfaced in time to act on. The general lesson, applicable far below that scale, is the one from section 7: in a system that can die uncleanly, your last and most important diagnostic line is exactly the one most likely to be lost, and unbuffered, flushed, stderr-bound output is what recovers it. (Where these incidents are documented, the details above reflect the public record; where I have generalized to the recurring pattern, I have flagged it as a pattern rather than a specific company's postmortem.)

The thread connecting these is that the bugs that hurt most are timing- and crash-related, and those are precisely the bugs a breakpoint cannot observe and a buffered print can hide. The engineer who understands *how* their output can lie — buffered, interleaved, timing-perturbing — is the one who can trust it when it matters. For a deeper treatment of how production observability is designed so the signal survives the incident, the system-design material on building observability in by design is the place to go; this post is the ground-level discipline that makes those higher-level systems trustworthy.

## 13. How to reach for this (and when not to)

Every technique has a cost, and the mark of a senior engineer is knowing when *not* to use the one they are good at. Here is the honest guidance.

**Reach for print first** when the bug is timing-sensitive (any async, event-loop, or concurrent code where a pause changes the answer); when you cannot attach a debugger (production under load, an embedded target, a CI box you do not control); when the bug is deep in a tight loop or a high-iteration path; when the bug spans processes or services and there is no single stack to freeze; and when you simply want the *fastest* answer to "what is this value here" with zero setup. In all of these, a flushed, tagged, structured trace line is the right move, and you should feel zero shame reaching for it.

**Reach for the debugger instead** when you are frozen at a fault and need to inspect many things at once — forty locals, the full call stack, the ability to evaluate expressions and call functions in the live process. A single print answers one question per rebuild; if you have *twenty* questions about one frozen moment, that is a debugger, and trying to print your way through it is the genuinely amateur move. The microscope post makes this case in full.

**Reach for structured logging instead** when the diagnostic should be *permanent* — when it will help during the next incident, not just this one — and when you are in production or distributed systems where leveled, filterable, correlation-tagged output is the only thing that scales. The moment your prints have outgrown being temporary, stop adding prints and add `logging.debug` lines.

And the explicit *don'ts*. Don't add a print when one well-placed breakpoint with the whole stack visible answers it faster — print is not always right, and stubbornly printing through a problem that wants inspection wastes time. Don't print inside a hot loop without a gate — an ungated print in a five-million-iteration loop produces five million lines you cannot read and slows the program enough to possibly mask the bug. Don't trust an unflushed log during a crash investigation — assume the last line is a lie until you have run unbuffered. Don't leave an untagged print in a diff — tag it or gate it, every time. Don't print a secret — `print(user.password)` is a security incident waiting to ship; the linter exists partly to catch this. And don't conclude a bug is "fixed" because a print made it vanish — that is the heisenbug telling you it is a timing race; confirm with a sanitizer before you believe it.

Tie it back to the loop the whole series runs on — observe, reproduce, hypothesize, bisect, fix, prevent. Print debugging is the cheapest, most universally available way to *observe*, and the disciplines here — structure, flush, gate, bisect, clean — are what make that observation *honest* and *repeatable* rather than a guess. Used deliberately, print is not the tool you apologize for. It is, in a large and important class of bugs, the fastest path to the root cause there is.

## Key takeaways

- **Print is the correct first tool, not the beginner's fallback,** for async and timing-sensitive code, production you cannot attach to, tight loops of millions of iterations, heisenbugs, and distributed systems — anywhere a breakpoint's act of stopping the world would destroy the evidence.
- **Never print a bare value.** Every diagnostic line carries a grep tag, the variable *name*, the value as a `repr` not a lossy `str`, a timestamp, and a thread or request id. That structure is what makes output greppable, sortable, and trustworthy under interleaving.
- **Dump whole objects with the built-in primitive** — `pprint` / `json.dumps(default=str)` in Python, `%+v` / `%#v` in Go, `console.dir({depth:null})` in JS, `dbg!` and `{:#?}` in Rust. Never hand-list fields.
- **Bisect with prints.** A print at the midpoint of a function halves the suspect region; about eight prints localize a wrong value in a 200-line function. It is binary search inside one run, the same method as `git bisect` over commits.
- **Gate your prints** — conditional (`if i == target`), one-shot (fire once), rate-limited, and above all flag-gated behind an environment variable so a diagnostic can be permanent and free when off.
- **Flush, or it lies.** Buffered stdout discards your last lines on an unclean crash, so the log blames the wrong line. Use `flush=True`, write to stderr, or run unbuffered during any crash investigation.
- **A print that makes a bug vanish is masking a timing race, not fixing it.** Confirm with a thread sanitizer (`-race`, `-fsanitize=thread`) that detects the race regardless of timing, then fix the synchronization.
- **Leave no mess:** tag and `git grep` before commit, gate the keepers behind a flag, let the linter (`flake8-print`, `no-console`, `clippy::dbg_macro`) fail the build on strays, and graduate temporary prints to `logging.debug` so they are leveled, filterable, and permanent without being noise.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the intro map for this whole series and the observe-reproduce-hypothesize-bisect-fix-prevent loop that print debugging serves.
- [The debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) — the sibling case for when stopping the world is exactly right, and how to wield breakpoints, watchpoints, and post-mortem inspection.
- [Binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the full treatment of bisection over commit history with `git bisect run`, the same method this post applies inside a single function.
- The sibling post on logging as a debugging instrument (planned in this series) — the deliberate, leveled, structured logging discipline your prints graduate into once they have outgrown being temporary; the natural next read after this one.
- *Debugging* by David J. Agans — nine rules including "Quit Thinking and Look" and "Make It Fail," the canonical practitioner's text on observing rather than guessing.
- *Why Programs Fail* by Andreas Zeller — the academic treatment of systematic debugging, delta debugging, and turning symptoms into falsifiable hypotheses.
- The ThreadSanitizer and AddressSanitizer documentation (the LLVM/`google/sanitizers` wiki) — how the sanitizers detect data races and memory errors regardless of whether the bad interleaving manifests on a given run, the decisive confirmation tool for a print-masked heisenbug.
- Brendan Gregg's writing on tracing and observability — the bridge from ad-hoc prints to `bpftrace`, `perf`, and production-grade dynamic tracing when print debugging needs to scale to a live system.

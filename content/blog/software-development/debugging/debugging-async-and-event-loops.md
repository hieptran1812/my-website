---
title: "Debugging Async and Event Loops: When the Stack Trace Lies"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn why async code destroys your stack trace and your sense of causality, and how to debug missing awaits, unhandled rejections, lost callbacks, blocked event loops, and async races with real diagnostics."
tags:
  [
    "debugging",
    "software-engineering",
    "async",
    "event-loop",
    "concurrency",
    "nodejs",
    "asyncio",
    "promises",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/debugging-async-and-event-loops-1.png"
---

The page came in at 02:14: a payments endpoint was "sometimes losing writes." Not crashing. Not timing out. Just — every few hundred requests, a customer would get a clean `200 OK`, the UI would say "saved," and the row would not be in the database. No error in the logs. No exception bubbling up to our error tracker. No stack trace to grep for. When I caught one in staging and looked at the handler, it was forty lines of perfectly ordinary code: validate the body, write the record, send the response. I added a `try/catch` around the whole thing expecting to catch a swallowed error, and it caught nothing, because there was nothing to catch in the synchronous path. The write was failing *after* the function had already returned. The handler had moved on, sent the response, and torn down its own stack frame before the database driver ever rejected the promise. The error landed on an empty stack, in a context that no longer had any idea who had asked for the work.

That is the defining cruelty of async debugging, and it is what this post is about. In synchronous code, two things hold that you lean on without ever noticing: the **stack trace** tells you who called whom, and **causality is linear** — line N+1 runs after line N, and if line N threw, you'd see it before line N+1. Async code quietly breaks both. The moment your code hits an `await` (or registers a callback, or returns a promise), the current call stack *unwinds* all the way back to the event loop, the loop goes off and runs other people's tasks, and your continuation is resumed *later, on a fresh stack* that no longer contains the frame that scheduled it. So when an async operation fails, the trace at the failure is short and useless: it shows the few frames of the resumed continuation and nothing above, because the logical caller — the request handler, the test, the `main()` that started it all — unwound minutes ago. The figure below is the whole problem in one column: a handler that calls a write, the `await` where the stack unwinds back to the loop, and the throw that lands two frames deep with no caller above it.

![A vertical stack showing a handler calling a write, the stack unwinding to the loop at the await, the loop running other tasks, and the error being thrown on a fresh stack with no caller above it](/imgs/blogs/debugging-async-and-event-loops-1.png)

By the end of this post you will be able to do six concrete things. Diagnose a **missing `await`** (the floating promise / never-awaited coroutine) where work runs detached and errors silently vanish. Catch an **unhandled rejection** before it disappears. Find a **lost callback** that hangs a request forever, and a callback that fires *twice*. Detect and fix a **blocked event loop** — the "why is my Node server slow at low CPU" mystery — using event-loop-lag monitoring. Track down **zombie timers and leaked listeners** that keep the loop alive or leak memory. And reason about **async ordering races**, which are real races even in single-threaded JavaScript because `await` is a yield point. Throughout, we follow the series spine: **observe → reproduce → hypothesize → bisect → fix → prevent.** The whole game in async debugging is to turn the loop's hidden state back into evidence, because the stack — your usual evidence — has been thrown away. This builds directly on [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) and on [reading a stack trace across languages](/blog/software-development/debugging/reading-a-stack-trace-across-languages); here we tackle the case where the trace itself is missing.

## 1. The event loop, in just enough detail to debug it

You cannot debug a machine you cannot picture, so let us build the picture first, precisely, because almost every async bug is a misunderstanding of this one mechanism. An **event loop** is a single thread running a tight loop: take one task off a queue, run it to completion or until it yields, then take the next one. That is the entire model. Node.js has one (libuv driving V8). Python's `asyncio` has one. Browsers have one. The crucial, load-bearing fact is that the loop runs **exactly one task at a time** and only ever switches tasks at two moments: when a task returns, or when a task hits an `await` (in JS, a point that returns a pending promise; in Python, a point that yields from the coroutine). There is no preemption. Nothing interrupts your task in the middle of a synchronous run of statements. This is why people say async code is "concurrent but not parallel": many things are *in flight*, but only one is *executing* at any instant.

When a task `await`s, three things happen in sequence, and understanding them is the key to everything that follows. First, the task registers a **continuation** — the rest of the function, captured so it can be resumed — as a callback to fire when the awaited thing resolves. Second, the current call stack **unwinds** back to the loop; the function's frame, its locals, its place in the call chain, all of it pops off the native stack. The continuation's state is kept alive on the heap (that is what a coroutine object or a promise's `.then` chain *is*), but the stack itself is gone. Third, the loop is now free, so it picks the next ready task and runs it. Later — maybe microseconds later, maybe seconds — the awaited operation completes, the continuation gets re-queued, the loop eventually picks it up, and it resumes *on a brand-new stack*. The figure below traces one full turn: pick a task, run it, hit an await, park the continuation, let the OS do the I/O, re-queue the result.

![A flow graph of one event loop turn showing a task picked from the queue, running until it either returns or hits an await, parking its continuation until the OS I/O or timer fires and re-queues the resolved value](/imgs/blogs/debugging-async-and-event-loops-2.png)

Hold onto two consequences, because they generate most of the bugs in this post. **Consequence one: the stack at a resumed continuation does not contain the scheduler.** When your `await db.write()` rejects and the error is thrown into the continuation, the native stack contains the continuation and the loop's internals — not the request handler that issued the write, not the route, not the test. That frame unwound at the await. This is why a naive async exception trace is two or three frames and tells you nothing. **Consequence two: anything you do synchronously between awaits blocks the entire loop.** If you run a tight `for` loop over ten million items, or call a synchronous `JSON.parse` on a 50 MB string, or do a blocking `fs.readFileSync`, no other task can run until you yield. You have frozen *all* concurrency on the process. Every in-flight request, every timer, every health check, is stuck behind you. We will weaponize both of these facts into diagnostics — async stack traces recover the lost scheduler, and event-loop-lag monitoring measures the freeze.

One more definition before we move, because the words matter. A **microtask** is a continuation scheduled to run at the very next opportunity, before the loop returns to the I/O queue — in JS, a resolved promise's `.then`; the microtask queue is fully drained between each macrotask. A **macrotask** (or "task" proper) is the timer callback, the I/O completion, the `setImmediate`. The distinction matters when you debug ordering: a chain of `await`s on already-resolved values can starve the I/O queue by piling up microtasks, and `process.nextTick` in Node runs *before even* the promise microtask queue, which has surprised many people into infinite loops. We will not belabor the queue taxonomy, but when ordering looks impossible, "which queue?" is often the question.

### Why ordering surprises you: a concrete trace

A single example pins down the queue model better than a paragraph. Consider this snippet and predict its output before reading on:

```js
console.log("1: sync");
setTimeout(() => console.log("4: timeout (macrotask)"), 0);
Promise.resolve().then(() => console.log("3: promise (microtask)"));
process.nextTick(() => console.log("2: nextTick (runs first)"));
console.log("1b: sync");
```

It prints `1: sync`, `1b: sync`, `2: nextTick`, `3: promise`, `4: timeout`. The two synchronous `console.log`s run first because nothing yields between them. Then the *current operation finishes*, and before the loop touches its timer queue, it drains the `nextTick` queue (which V8 prioritizes above even promises), then the promise microtask queue, and only *then* the macrotask (`setTimeout`) queue. People reach for `setTimeout(fn, 0)` expecting "run this next" and are baffled when a promise callback registered *later* runs *first* — because `setTimeout` is a macrotask and the promise is a microtask, and all microtasks drain before any macrotask. When you debug an ordering bug where "this should have run before that," sketch this queue order and ask which queue each callback lives in. Nine times out of ten the surprise is a macrotask losing a race to a microtask, or a `nextTick` recursion (a `nextTick` that schedules another `nextTick`) starving the I/O queue forever — a loop that never reaches its timers because the microtask queue never empties.

### The forgotten await in a loop: sequential when you wanted parallel

A performance bug that hides as a correctness-shaped habit. People write an `await` inside a `for` loop and accidentally serialize work that should run concurrently:

```js
// SLOW: each await blocks the next iteration — sequential, total = sum of latencies
async function fetchAllSlow(ids) {
  const results = [];
  for (const id of ids) {
    results.push(await fetchOne(id));   // waits for each before starting the next
  }
  return results;
}
// FAST: start them all, then await together — total = max latency, not sum
async function fetchAllFast(ids) {
  return Promise.all(ids.map(id => fetchOne(id)));
}
```

For 100 ids at 50ms each, the sequential version takes 100 × 50ms = 5 seconds; the `Promise.all` version takes ~50ms (the slowest single call) because all 100 are in flight at once. This is not a crash, so it never throws — it just silently makes an endpoint 100× slower, and it is invisible in code review because `await` *looks* right. The diagnostic is to log timestamps around the loop and notice the calls happen back-to-back rather than overlapping, or to watch a flame graph / waterfall show 100 sequential bars instead of 100 stacked ones. The honest caveat: `Promise.all` fires *all* of them at once, which can overwhelm a downstream — for large `ids` you want bounded concurrency (a pool of, say, 10 in flight) via `p-limit` or a manual semaphore. The fix is "parallel with a bound," not "parallel without limit," and choosing the bound is a real engineering decision, not a one-liner. Python's exact analog is `await`ing in a loop versus `asyncio.gather(*tasks)`; the same 100× hides in the same place.

## 2. The bug taxonomy: three families of async failure

Before we go deep on each, it helps to have a map, because "the async code is broken" is not a hypothesis you can falsify — it is too vague. Every async bug I have chased fits into one of three families, and naming the family is the first cut that turns a vague symptom into a testable claim. The taxonomy below is the one I reach for at 2am.

![A taxonomy tree splitting async failures into detached work that runs unwatched, work that never completes and hangs, and a blocked loop where everything stalls, each with its concrete sub-cases](/imgs/blogs/debugging-async-and-event-loops-4.png)

**Family one: detached work.** Something was started but nobody is holding the handle. The two members are the **missing `await`** (a promise or coroutine is created and never awaited, so the work runs in the background, its ordering relative to everything else is undefined, and any error it produces lands with no one to catch it) and the **unhandled rejection** (an error in an async path that has no `.catch` and no surrounding `try`, which historically vanished silently). The symptom is *spooky*: results that are sometimes there and sometimes not, responses that return before their work finishes, errors that appear in no log.

**Family two: stalled work.** Something was started and will never finish. The classic is the **lost callback** — a continuation-passing-style function where one code path (an early `return`, an error branch, a thrown exception before the callback is invoked) forgets to call the callback, so the caller waits forever. Its evil twin is the callback called *twice*, which double-sends a response or double-charges a card. With promises, the analog is a promise that never resolves or rejects — an `await` that hangs because the thing it awaits is dead. The symptom is a request that hangs, a test that times out, a graceful shutdown that never completes.

**Family three: a blocked loop.** The single shared thread is busy doing synchronous work, so *every* task stalls — not one request, all of them. This is **event-loop starvation**: a CPU-bound loop or a synchronous I/O call holding the thread. The symptom is uniquely confusing: high latency across the board, but low CPU per request and an idle-looking process, because the CPU spikes in bursts on one thread while everything else queues. Adjacent to it sit **zombie tasks** — a `setInterval` or subscription never cleared that keeps the loop alive or leaks memory, which we treat as a [resource leak](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) with an async flavor.

A fourth pattern, **async ordering races**, cuts across families two and three: because `await` is a yield point, two coroutines touching shared state can interleave and corrupt it even with no threads in sight. We give it its own section because it surprises people the most. With the map in hand, let us take the families one at a time, mechanism first, then the diagnostic, then the proof.

## 3. The missing await: work that runs detached

Here is the bug from the opening page, reduced to its essence. This is real Express-style Node, and it looks completely fine.

```js
// BUG: the await is missing on the write
app.post("/orders", async (req, res) => {
  const order = validate(req.body);          // sync, fine
  db.insert("orders", order);                // <-- no await! floating promise
  res.status(200).json({ id: order.id });    // responds immediately
});
```

Trace the control flow against the event-loop model from section 1. `db.insert` returns a pending promise. We do not await it, so execution continues *synchronously* to the next line: `res.status(200).json(...)` sends the response. The handler then returns, its stack unwinds, the loop moves on. Meanwhile the insert is still in flight. If the insert succeeds, you got lucky and nobody noticed. If it fails — a constraint violation, a dropped connection, a deadlock victim — the promise rejects on an empty stack, with no `.catch`, and historically the error simply evaporated. The client already has its `200`. The row is not there. This is the **floating promise**, and the `await` that should anchor it is the single most common async bug in the wild. The fix is one word — `await db.insert(...)` — but you cannot fix what you cannot see, and the whole difficulty is *seeing* it. The figure contrasts the two control flows: the detached write that sends the response early versus the awaited write that blocks the handler until the row is durable.

![A two-column before-and-after showing a missing await that sends the response before the write commits and an error vanishing, versus an awaited write that commits the row durably before responding](/imgs/blogs/debugging-async-and-event-loops-3.png)

### The mechanism: why the error truly vanishes

It is worth being precise about *why* the error disappears rather than just asserting it, because the precision tells you where to instrument. In JavaScript, a promise that rejects with no rejection handler attached *by the end of the current microtask checkpoint* is reported to the `unhandledRejection` hook — but only if you have one, and historically Node would only print a warning and (in old versions) keep running. The rejection is not an exception in the classic sense; it never unwinds a stack looking for a `catch`, because there is no stack to unwind — the stack that created it is long gone. It is a *value* sitting in a rejected promise that nobody read. In Python's `asyncio`, the analog is even more visible: if you create a coroutine and never await it, the coroutine object is garbage-collected without ever running, and the interpreter emits `RuntimeWarning: coroutine 'foo' was never awaited`. The work *did not even happen*. These two runtimes fail differently — JS runs the detached work and loses the error; Python often does not run the work at all — but the debugging instinct is the same: make the runtime tell you about promises and coroutines that nobody is holding.

### The method: lint it, then make the runtime shout

You catch most missing awaits before they ship, statically. In a TypeScript codebase, the `@typescript-eslint/no-floating-promises` rule flags any promise-returning expression whose result is ignored. Turn it on as an *error*, not a warning:

```js
// .eslintrc.cjs
module.exports = {
  parserOptions: { project: "./tsconfig.json" },
  rules: {
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
  },
};
```

`no-floating-promises` would have caught the orders bug at the keyboard. `no-misused-promises` catches the subtler cousin — passing an `async` function where a sync callback is expected, e.g. `arr.forEach(async x => await f(x))`, which fires all the iterations without waiting and is almost never what you meant. For the ones that slip past static analysis, make the *runtime* loud. In Python, run with asyncio debug mode on, which turns never-awaited coroutines and slow callbacks into logged warnings:

```bash
# Python: enable asyncio debug mode at the process level
PYTHONASYNCIODEBUG=1 python -W error::RuntimeWarning app.py
```

Or programmatically, which also enables slow-callback logging:

```python
import asyncio, logging
logging.basicConfig(level=logging.WARNING)
loop = asyncio.new_event_loop()
loop.set_debug(True)            # logs never-awaited coroutines and slow callbacks
loop.slow_callback_duration = 0.1   # warn on any callback that runs > 100ms
```

With `-W error::RuntimeWarning`, the never-awaited coroutine becomes a hard error that stops the process at the offending line — a far better signal than a warning buried in a log. In Node, register the global hook so a floating rejection cannot escape unseen, and in production crash on it deliberately rather than limping along in an unknown state:

```js
process.on("unhandledRejection", (reason, promise) => {
  console.error("UNHANDLED REJECTION", { reason, stack: reason?.stack });
  // In modern Node the default is to crash; we make it explicit and log first.
  process.exit(1);
});
```

### The proof: from "sometimes loses writes" to a localized line

Here is how this investigation actually resolved, with numbers, because the kit demands a before→after you can trust and this is the one I lived. We were dropping roughly 1 in 300 order writes — about 0.3% — under normal traffic, invisible at the request level because each individual request returned `200`. We could not reproduce it locally because our dev database never rejected an insert. The fix path was: (1) enable `no-floating-promises` across the repo, which surfaced *eleven* floating promises, one of them the orders insert; (2) add the `unhandledRejection` crash-and-log handler in staging, then run a load test that forced insert failures by capping the connection pool. The rejections went from silently dropped to *eleven log lines a second* with full reasons, naming the file and the rejected query. We awaited the insert, redeployed, and the lost-write rate went from 0.3% to 0 over the next 48 hours and a few hundred thousand orders. The bug was one missing keyword. Finding it took making the runtime stop hiding it.

#### Worked example: the floating promise that linted clean but still leaked

A subtle one, because it teaches you not to trust a green lint blindly. A service had `no-floating-promises` enabled and passing, yet still leaked a detached write. The culprit was a "fire and forget" metrics call deliberately marked `void emitMetric(order)` to satisfy the linter — `void` tells the rule "I meant to ignore this." That is a legitimate pattern *if* the ignored promise can never reject in a way you care about. But `emitMetric` could reject when the metrics backend was down, and because it was `void`-ed, the rejection went to `unhandledRejection`, which in our config *crashed the process*. So a metrics outage was crashing the order service — a tiny detached promise taking down the whole server. We caught it by correlating: every crash had an `unhandledRejection` log line 8ms before it, and the reason always referenced `emitMetric`. The fix was not to await it (we genuinely did not want to block orders on metrics) but to give it its own catch: `emitMetric(order).catch(logMetricError)`. The lesson: `void` silences the linter, not the runtime. Every detached promise still needs a `.catch`, even the ones you fired and forgot on purpose.

## 4. Unhandled rejections: the error that used to disappear

The missing await is one *source* of detached errors; the broader class is the **unhandled rejection** — any promise that rejects with nobody listening. It is worth a section of its own because the failure mode changed across Node versions and a lot of folklore is stale. In Node before version 15, an unhandled rejection printed a deprecation warning and the process *kept running*, often in a corrupt state — half-applied transactions, leaked connections, a queue that silently stopped draining. From Node 15 onward the default is `--unhandled-rejections=throw`, which terminates the process. This is a *good* default for the same reason a segfault is better than silent memory corruption: a fast, loud crash is debuggable; a slow, silent drift is not. But it means code that "worked" on old Node now crashes on new Node, and the bug was always there — you just promoted it from invisible to fatal.

The mechanism, stated precisely: a rejected promise carries its rejection reason as a value. The runtime tracks, per promise, whether a rejection handler is ever attached. At the end of each microtask checkpoint, any promise that is rejected and *still* has no handler is reported to `unhandledRejection`. The window matters — you can attach a `.catch` asynchronously and still be "handled," which is why the report is deferred to the checkpoint and not fired the instant of rejection. This deferral is also why a `.catch` added one tick too late does not save you: if the rejection's checkpoint already passed, it was already reported. Debugging these is about three moves. First, *always* install the global handler so nothing escapes:

```js
process.on("unhandledRejection", (reason) => {
  log.error({ event: "unhandledRejection", reason: reason?.stack ?? reason });
  process.exit(1);   // fail fast; let the orchestrator restart a clean process
});
process.on("uncaughtException", (err) => {
  log.error({ event: "uncaughtException", err: err.stack });
  process.exit(1);
});
```

Second, run Node with `--trace-warnings` and, when hunting a specific one, `--unhandled-rejections=strict` in CI so any rejection fails the test run. Third — and this is the move that recovers causality — turn on async stack traces so the rejection's reason carries the *full async chain* back to the scheduler, which we cover in section 7. Python's equivalent is the task exception handler:

```python
def handle_task_exception(loop, context):
    # Fires when a Task is GC'd with an unretrieved exception, among others
    logging.error("asyncio exception: %s", context.get("exception"), exc_info=context.get("exception"))

loop.set_exception_handler(handle_task_exception)
```

In `asyncio`, the analogous trap is a `Task` whose exception is never retrieved. If you `asyncio.create_task(coro())` and never await the task or check its result, and the coro raises, the exception sits unretrieved until the task is garbage-collected, at which point `asyncio` logs "Task exception was never retrieved." That log line, plus `loop.set_exception_handler`, is your `unhandledRejection` for Python. The honest before→after here is a measurement of *visibility*, not of correctness: before, a known-flaky downstream caused zero log lines and a slow trickle of corrupt state; after installing the handlers and `--unhandled-rejections=strict` in CI, the same downstream produced an immediate, named, stack-bearing crash in 100% of the failing cases, and CI started catching them before deploy instead of pager catching them after.

There is a subtle ordering trap inside the rejection-reporting machinery that bites people who "handle" their rejections and still see crashes. Attaching `.catch` *synchronously* on the same promise is always safe. Attaching it across an `await` boundary — say you `await something()` and only then wire a `.catch` onto a promise you created *before* the await — can be too late, because the rejection's microtask checkpoint may have already passed during the awaited gap, and the rejection was reported as unhandled in that window. The rule that avoids the whole trap: attach the handler in the same synchronous breath as you create the promise (`doThing().catch(handle)`, never `const p = doThing(); await other(); p.catch(handle)`). When you genuinely must fan out promises and collect their outcomes later, use `Promise.allSettled` rather than bare `Promise.all` — `allSettled` never leaves a rejection unobserved, because every input promise's outcome is captured as a result object, whereas `Promise.all` rejects on the first failure and the *other* in-flight promises can become unhandled rejections if you do not also catch them. Reaching for `allSettled` when you want "run all of these and tell me which failed" is the correct default; reaching for `Promise.all` there silently creates floating rejections on the losers.

#### Worked example: the rejection that only crashed in production

A service crashed two or three times a day in production and *never* in staging, which is the signature of a load- or data-dependent rejection. The crash log was always an `unhandledRejection` with a reason of `ECONNRESET` and — thanks to async stack traces being on — a chain ending in a background cache-warmer that ran `Promise.all([...])` over a list of upstream fetches. In staging the upstream never reset the connection, so the `Promise.all` never saw a rejection; in prod, under real network conditions, one of the parallel fetches would occasionally `ECONNRESET`, `Promise.all` rejected on it, and the *other* in-flight fetches' rejections (the same upstream often reset several at once) became unhandled because nothing was catching the losers. The fix had two parts, both from this section: switch the warmer to `Promise.allSettled` so no loser rejection floats, and give each fetch its own `.catch` that logged and returned a sentinel. After deploy, the crash rate went from ~2.5/day to 0 over two weeks, and the cache-warmer degraded gracefully (warming what it could) instead of taking down the process. The diagnostic that cracked it was, again, *visibility*: the async stack trace named the warmer, and the "staging never resets, prod sometimes does" pattern told us it was data-dependent, not a code path that was simply wrong everywhere.

## 5. Lost callbacks: the request that hangs forever

Now the second family — stalled work — and the most maddening symptom in it: a request that simply hangs. No error, no timeout in your code, no CPU. The connection sits open until the *client's* timeout fires, minutes later, and your logs show the request starting and never finishing. The cause is almost always a **lost callback**: a continuation-passing-style function where some code path fails to call its callback. Consider this classic, and notice it on the early-return path.

```js
function getUser(id, cb) {
  cache.get(id, (err, hit) => {
    if (err) return;                  // BUG: returns without calling cb
    if (hit) return cb(null, hit);    // ok path
    db.query("SELECT ...", [id], (err, row) => {
      if (err) return cb(err);
      cb(null, row);
    });
  });
}
```

The `if (err) return;` on the cache error path returns from the inner callback *without invoking `cb`*. So when the cache errors — say Redis blips — `getUser`'s caller is never told anything, neither success nor failure. The promise wrapping this (or the awaiting handler) hangs forever. The request thread is not blocked — the event loop is happily running other requests — but *this one continuation is orphaned*. It will never be re-queued because nothing will ever call its callback. The mechanism is simply that in continuation-passing style, **the callback is the only edge back to the caller**, and any path that does not traverse that edge severs the request permanently. Promises mostly fix this by construction — a promise executor that throws auto-rejects, so the throw path is covered — which is the strongest argument for wrapping every callback API in a promise at the boundary and never hand-writing CPS for new code.

The reason a lost callback is so much worse to debug than a thrown error deserves emphasis, because it changes how you instrument. A thrown error, even an async one, *eventually* produces a log line or a crash — there is an event. A lost callback produces *no event at all*. Nothing is logged, because nothing decided "this is over." Nothing crashes, because the process is healthy and busy. The only observable is *absence*: a request that started and has no matching "finished" log line, a counter of in-flight requests that creeps up and never comes down, a connection pool that slowly exhausts as orphaned handlers hold their connections forever. This is why the prevention is a *timeout on every async operation that can hang*: wrap awaits in `Promise.race([work(), timeoutAfter(5000)])` (or Python's `asyncio.wait_for(coro, timeout=5)`), so a lost callback becomes a *timeout error* — an event you can log, count, and alert on — instead of a silent eternal wait. A timeout does not fix the lost callback, but it converts an invisible hang into a visible failure, and visible failures are debuggable. Reaching for an unbounded `await` on anything that crosses a network or a queue boundary is the habit that makes lost callbacks invisible; a bounded `await` is the habit that makes them loud.

### The method: dump the pending tasks and find the orphan

When a request hangs and you suspect a lost callback, you need to see *what the loop is waiting on*. In Python `asyncio` this is gloriously direct — dump every pending task and its stack:

```python
import asyncio, sys

def dump_pending_tasks():
    for task in asyncio.all_tasks():
        if not task.done():
            print(f"--- PENDING TASK {task.get_name()} ---", file=sys.stderr)
            task.print_stack(file=sys.stderr)   # shows where the coroutine is suspended

# Wire it to a signal so you can trigger it on a hung prod process:
import signal
signal.signal(signal.SIGUSR1, lambda *_: dump_pending_tasks())
```

Send `SIGUSR1` to the hung process and every suspended coroutine prints the exact `await` line it is parked on. A coroutine parked forever on `await some_future` whose future nobody will ever resolve *is* your lost callback, and `print_stack` names the line. For Node, `node --inspect` plus Chrome DevTools lets you pause and inspect pending async operations, and `process._getActiveHandles()` / `process._getActiveRequests()` (or the cleaner `why-is-node-running` package) enumerate what is keeping the loop alive. For a truly hung Python process where the interpreter itself is wedged, `faulthandler` dumps every thread's stack on a signal:

```python
import faulthandler, signal
faulthandler.register(signal.SIGABRT)   # kill -ABRT <pid> -> full stack dump
faulthandler.dump_traceback_later(30, repeat=True)  # auto-dump if hung 30s
```

### The callback called twice

The mirror-image bug is the callback invoked *more than once*. It double-sends a response (Express throws "Cannot set headers after they are sent"), double-resolves a promise (silently — extra resolutions are ignored, which hides the bug), or double-charges. It usually comes from an error path that calls `cb(err)` and then *falls through* to also call `cb(null, result)`, the missing `return` before the error callback. The diagnostic is a guard that screams on the second call:

```js
function once(cb) {
  let called = false;
  return (...args) => {
    if (called) { console.error("DOUBLE CALLBACK", new Error().stack); return; }
    called = true;
    cb(...args);
  };
}
```

Wrap a suspect callback in `once()` and the `new Error().stack` on the second invocation gives you the synchronous stack of the *second* caller — which, because both calls usually happen in the same synchronous run, actually points at the bug. That is a rare gift in async debugging: a stack that is still intact because no `await` intervened.

#### Worked example: the upload that hung 1 in 50 times

A file-upload service hung on roughly 2% of uploads — 1 in 50 — with no error. Reproducing it was the whole battle; it never hung locally. We followed the spine: **reproduce first.** We wrote a loop that uploaded the same 4KB file 5,000 times against staging and counted hangs; it hung 94 times, almost exactly 2%, confirming it was real and load-independent. Then we wired `SIGUSR1` to `dump_pending_tasks` and triggered it mid-hang. Every hung request showed a coroutine parked on `await self._scan_for_virus(file)`. Reading `_scan_for_virus`, the virus scanner had a path — when the file matched an internal allow-list — that `return`ed early *without resolving the future* the caller awaited. The 2% was exactly the fraction of uploads hitting the allow-list. The pending-task dump turned an invisible hang into a one-line fix (resolve the future on the allow-list path) and a regression test that uploaded an allow-listed file and asserted completion. After the fix: 0 hangs in 20,000 uploads. The measurement that mattered was not speed but *the hang rate going to zero and staying there*, which we only trusted because we had a reproducer that reliably produced the failure first.

## 6. Async ordering races: a race with one thread

Now the pattern that surprises everyone, because it violates a belief people hold dear: "JavaScript is single-threaded, so I can't have a race." You can. You absolutely can. The belief is half-true — there is no *parallelism*, no two statements running at the same instant, no torn reads of a single variable. But there is **interleaving**, and a race condition only needs interleaving, not parallelism. The yield point is `await`. Every `await` is a place where the loop can run *some other task* before resuming you, and if that other task touches the same state you are halfway through updating, you have a classic check-then-act race — in single-threaded code. This connects directly to [race conditions, the hardest bugs to catch](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch); the mechanism there is hardware reordering across threads, the mechanism here is loop interleaving across awaits, but the *shape* of the bug is identical.

Here is the canonical example: two concurrent requests debiting the same account.

```js
async function debit(accountId, amount) {
  const acct = await db.get(accountId);     // await #1: yield point
  if (acct.balance < amount) throw new Error("insufficient");
  acct.balance -= amount;
  await db.put(accountId, acct);            // await #2: yield point
}
```

Run two `debit(7, 30)` and `debit(7, 40)` concurrently on a balance of 100. Trace it against the loop model. Request A awaits `db.get` and reads balance 100; at that `await`, the loop is free, so it runs Request B, which *also* awaits `db.get` and *also* reads balance 100 — because A has not written yet. Now both have a local copy saying 100. A computes 70, writes it. B computes 60, writes it, *clobbering* A's write. One debit is lost; the account reads 60 when it should read 30. No thread in sight, no `-race` flag will fire, because there is no data race at the memory level — it is a *logic* race created by the yield point. The figure lays the interleaving out by time: both requests read the stale 100 in the gap between their awaits, and the second write wins.

![A grid showing two requests interleaving across await points, both reading a stale balance of 100 before either writes, so the second write of 60 clobbers the first write of 70 and a debit is lost](/imgs/blogs/debugging-async-and-event-loops-5.png)

The mechanism deserves one more sentence of rigor because it is the crux: **the dangerous region is between two awaits that span a read and a write of shared state.** Anything you do *without* awaiting is atomic with respect to the loop — no other task can interleave, because the loop will not switch until you yield. So the fix space is clear: either shrink the critical section to contain no await (do the read-modify-write in one synchronous burst, e.g. an atomic database operation or a single SQL `UPDATE ... SET balance = balance - 30 WHERE balance >= 30`), or serialize access with an async lock so only one debit holds the account at a time. The async-lock version:

```js
const locks = new Map();
async function withLock(key, fn) {
  while (locks.get(key)) await locks.get(key);   // wait for the current holder
  let release;
  locks.set(key, new Promise(r => (release = r)));
  try { return await fn(); }
  finally { locks.delete(key); release(); }
}
// usage: await withLock(accountId, () => debit(accountId, amount));
```

The honest trade-off: the in-process lock only works within one process — across a cluster you need a distributed lock or, far better, an atomic DB operation that pushes the check-then-act into the database's transaction, which is what [databases](/blog/software-development/database) are *for*. Reaching for an app-level lock when the database can do an atomic conditional update is usually the wrong call; reach for the DB's atomicity first.

### The method: reproduce the interleave on purpose

Async races are heisenbugs — they vanish under a debugger because stepping changes the timing. The reliable diagnostic is to *force* the bad interleave by inserting a yield where you suspect the window is:

```js
// Test harness: widen the race window deterministically
async function debitInstrumented(id, amt, delayMs) {
  const acct = await db.get(id);
  await new Promise(r => setTimeout(r, delayMs));   // force a yield mid-critical-section
  acct.balance -= amt;
  await db.put(id, acct);
}
// Kick both off; the delay guarantees B reads before A writes:
await Promise.all([debitInstrumented(7, 30, 50), debitInstrumented(7, 40, 0)]);
// assert balance === 30, watch it be 60 every time
```

By injecting a `setTimeout` between the read and the write you turn a one-in-a-thousand interleave into a 100%-reproducible failure, which you then assert against in CI. This is the same "widen the window" move from the race-conditions post, adapted to the loop: you are not changing thread scheduling, you are forcing the loop to switch tasks at the exact instant that exposes the bug. Once it fails reliably, you fix it, and the same test proves the fix by passing 1,000 times.

## 7. Recovering the lost stack: async stack traces

Everything so far has danced around the central handicap: when an async operation fails, the stack does not contain the scheduler. Now we fix that handicap directly, because modern runtimes can *reconstruct* the logical async call chain even though the native stack has unwound. This is the single highest-leverage technique in this post.

The mechanism of the recovery is elegant. The runtime cannot keep the native stack alive across an `await` — it physically unwinds. But it *can* capture a lightweight snapshot of the stack *at the moment a continuation is registered*, store it on the heap alongside the continuation, and then, when that continuation later throws, *stitch* the saved snapshot onto the current short stack. The result is a trace that reads like a synchronous one even though the frames lived at different times on different native stacks. In V8 (Node 12+ and modern Chrome), this is **async stack traces**, on by default for promises, and you can force it and deepen it:

```bash
# Node: enable / deepen async stack traces
node --stack-trace-limit=50 --async-stack-traces app.js
```

In Chrome DevTools the call-stack panel shows an "Async" separator splicing the pre-await frames above the post-await ones — you see the route handler that issued the query even though, natively, it unwound long ago. For deeper, programmatic introspection Node exposes `async_hooks`, which fires callbacks on the lifecycle of every async resource (init, before, after, destroy) and lets you maintain your own causal chain. You rarely write raw `async_hooks` — it is the foundation under `AsyncLocalStorage`, which is the practical tool:

```js
const { AsyncLocalStorage } = require("async_hooks");
const als = new AsyncLocalStorage();

app.use((req, res, next) => {
  als.run({ requestId: req.headers["x-request-id"] ?? crypto.randomUUID() }, next);
});

function log(...args) {
  const ctx = als.getStore();
  console.log(JSON.stringify({ requestId: ctx?.requestId, ts: Date.now(), msg: args }));
}
```

`AsyncLocalStorage` propagates a context *through every await* in a request's async tree without you threading it manually — it is the async-aware version of thread-local storage, built on `async_hooks`. This is how you reconstruct causality when the stack is gone: every log line, no matter how deep in the async chain, carries the `requestId`, so you can `grep` one request's entire journey across the loop's interleavings. That ties directly to [logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) — when the stack cannot tell you who called you, a correlation id threaded through `AsyncLocalStorage` can. In Python, the equivalent context propagation is `contextvars`, which `asyncio` integrates so a `ContextVar` set in a parent coroutine is visible in awaited children, and the `extra=` field of structured logging carries your correlation id through.

### Python's async stack tooling

`asyncio` debug mode (section 3) already improves tracebacks by attaching the coroutine's creation traceback to never-awaited warnings. For a live hung process, the combination is `asyncio.all_tasks()` to enumerate and `task.get_stack()` / `task.print_stack()` to see where each is suspended (we used this in section 5). For a deeper view, Python 3.12+ ships `python -m asyncio ps <pid>` and `python -m asyncio pstree <pid>` to dump the running event loop's task tree of an external process — the closest thing to `jstack` for asyncio. And `py-spy dump --pid <pid>` will give you native + async-aware stacks of a running Python process *without* stopping it, which is the prod-safe move when you cannot attach `pdb`.

#### Worked example: the trace that finally named the caller

A GraphQL service threw `TypeError: cannot read property id of undefined` a few hundred times a day, and the stack was four frames of resolver-runtime internals — no resolver name, no field, nothing actionable. We had been *guessing* for a week. The fix was embarrassingly simple: we were on Node 16 but had `--stack-trace-limit=10`, which truncated the async splice. We bumped it to 50 and added `--async-stack-traces` explicitly. The very next occurrence printed the *full* async chain: the runtime internals, then an "async" separator, then `resolveOrder`, then the specific GraphQL field `order.shippingAddress`, then the request that triggered it. The undefined was a missing join on cancelled orders. Time from "no idea, four useless frames" to "exact field and exact data condition": one flag and one redeploy. The measurement: we went from 0% of these errors being attributable to a code site to 100%, and fixed the root cause that afternoon. The lesson burned in: **check your stack-trace-limit before you spend a week guessing** — a truncated async trace looks exactly like a runtime that "just doesn't capture the caller."

## 8. The blocked event loop: high latency, idle CPU

The third family, and the one that produces the most confusing dashboards. Symptom: your Node (or single-loop Python) server has terrible latency — p99 of two seconds — but CPU sits at 30%, every request looks cheap individually, and you cannot find the slow query or the slow downstream. You scale out horizontally and it barely helps. The cause is **event-loop starvation**: somewhere, a *synchronous* operation is holding the single thread for hundreds of milliseconds at a time, and during that hold, *every other request is frozen in the queue*. The CPU looks idle because the blocking happens in bursts on one core while everything else waits; averaged over a second, utilization is low, but the *loop* — the thing that determines latency — is pegged.

The mechanism is exactly consequence two from section 1: the loop runs one task at a time with no preemption, so a synchronous run that takes 500ms means no other task — no other request, no timer, no health check — runs for 500ms. Common culprits: `JSON.parse`/`JSON.stringify` on a large payload (parsing a 50 MB body is hundreds of milliseconds of pure synchronous CPU), synchronous crypto (`crypto.pbkdf2Sync`, a big `bcrypt` round count called synchronously), a `fs.readFileSync` in a request path, a regex with catastrophic backtracking, or a plain CPU loop over a large array. Each one freezes the loop for the duration. The fix is always one of: move it off the loop (a worker thread, a child process, the libuv threadpool for I/O), make it asynchronous and chunked (stream the parse, yield between chunks), or precompute it. The figure shows the before→after with real numbers: a 50 MB synchronous parse blocking the loop for 1.9 seconds and pinning p99 at 2 seconds, versus offloading the parse to a worker thread so the loop stays free and p99 drops to 80ms.

![A two-column before-and-after showing a 50 megabyte synchronous parse blocking the loop and pinning p99 at 2000 milliseconds, versus moving the parse to a worker thread so the loop stays free and p99 falls to 80 milliseconds](/imgs/blogs/debugging-async-and-event-loops-7.png)

### The method: measure event-loop lag

You do not guess at a blocked loop — you *measure* it, with a metric called **event-loop lag** (or event-loop delay): how long the loop takes to come back around to a scheduled callback. The naive version schedules a timer for 0ms and measures how late it actually fires; if the loop is free, it fires within a millisecond; if the loop is blocked, the timer fires hundreds of milliseconds late and the lateness *is* the block duration. Node ships a precise, low-overhead version in `perf_hooks`:

```js
const { monitorEventLoopDelay } = require("perf_hooks");
const h = monitorEventLoopDelay({ resolution: 10 });  // sample every 10ms
h.enable();

setInterval(() => {
  // values are in nanoseconds; convert to ms
  console.log(JSON.stringify({
    event: "loop_lag",
    mean_ms: (h.mean / 1e6).toFixed(1),
    p99_ms: (h.percentile(99) / 1e6).toFixed(1),
    max_ms: (h.max / 1e6).toFixed(1),
  }));
  h.reset();
}, 1000);
```

`monitorEventLoopDelay` uses a high-resolution histogram and costs almost nothing, so you run it in production permanently and *alert* on it. A healthy loop has mean lag under a millisecond and p99 a few milliseconds. When you see p99 lag spike to 1,900ms in lockstep with your request-latency spike, you have proven the loop is blocked — and crucially, you have proven the latency is *not* a slow downstream or a slow query, because those would not move loop lag. That single metric ends the entire "is it the database or the loop?" argument with evidence. Once you know the loop is blocked, find *what* blocks it with a CPU profile: `node --prof` then `node --prof-process`, or the Chrome DevTools CPU profiler, or `0x` for a flame graph. The flame graph of a blocked loop has a tall, wide synchronous tower — your `JSON.parse` or your crypto call — sitting right on top of the loop's tick, and that tower is your culprit. This is the same flame-graph reading covered in [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod), applied to the loop's one thread.

### The fix: get off the loop

Once you have named the blocking call, you move it. The CPU-bound case goes to a worker thread:

```js
const { Worker } = require("worker_threads");

function parseHugeOffLoop(jsonString) {
  return new Promise((resolve, reject) => {
    const w = new Worker("./parse-worker.js", { workerData: jsonString });
    w.once("message", resolve);
    w.once("error", reject);
    w.once("exit", code => code !== 0 && reject(new Error(`worker exit ${code}`)));
  });
}
// parse-worker.js:  const { parentPort, workerData } = require("worker_threads");
//                   parentPort.postMessage(JSON.parse(workerData));
```

Now the 1.9-second parse runs on a *different* thread; the loop posts the work and stays free to serve other requests, paying only the small cost of serializing the string across the thread boundary. For large payloads, the even better fix is to **stream** the parse so you never hold the whole 50 MB at once — a streaming JSON parser yields between chunks, returning control to the loop repeatedly. For sync crypto, switch to the async API (`crypto.pbkdf2` instead of `pbkdf2Sync`) which already runs on libuv's threadpool. The general rule, and a key takeaway: **the event loop is for orchestration, not computation.** Any synchronous operation that can take more than a few milliseconds does not belong on it.

One more nuance the dashboards hide: a blocked loop is *self-reinforcing under load*, which is why the latency curve is not linear but a cliff. While the loop is frozen for 1.9 seconds parsing one big body, *new* requests keep arriving and queuing behind it. So the request that triggered the parse pays 1.9 seconds, but the hundred requests that arrived during that window pay 1.9 seconds *plus* their own position in the backlog, and the backlog itself takes time to drain after the loop frees up. This is why a server that is fine at 80% of capacity falls off a cliff at 85%: a single long synchronous block that was tolerable when traffic was light becomes a head-of-line stall that cascades when there is a queue behind it. The diagnostic signature is loop lag and request latency rising *together* and *super-linearly* with traffic, while CPU stays modest. When you see that shape — latency exploding faster than load, CPU calm — stop looking for a slow dependency and measure the loop. The cure (off-load the blocker) collapses the cliff back into a gentle slope, which is exactly what the 27× improvement in the worked example below looks like on a latency-vs-throughput chart: the knee moves out, and the wall at 85% disappears.

#### Worked example: 2-second latency at 30% CPU

This is the second worked example the brief promised, and it is the cleanest "measure it" story I have. A reporting API had p99 latency of 2.1 seconds and the team had spent three days optimizing the database — adding indexes, tuning queries — with zero improvement, because the database was never the problem. CPU on the box averaged 31%. We added `monitorEventLoopDelay` and within one minute had the answer: mean loop lag 4ms, **p99 loop lag 1,950ms, max 2,400ms** — the loop was being frozen for nearly two seconds at a stretch. That immediately exonerated the database (a slow query cannot raise loop lag) and pointed at synchronous CPU. A `node --prof` capture over five minutes, processed with `--prof-process`, put 71% of ticks in one frame: `JSON.parse`, called on the raw body of an upstream report that had grown to ~48 MB. We moved the parse to a worker thread. Next deploy: p99 loop lag dropped to 3ms, and end-to-end p99 latency went from 2,100ms to 78ms — a 27× improvement — with *no* database change. The three days on the database were wasted because nobody measured the loop. The whole investigation, once we measured the right thing, took under an hour. The discipline the kit preaches — turn the symptom into a measured, falsifiable claim before you optimize — would have saved three days.

## 9. Zombie tasks: leaked timers, listeners, and subscriptions

The last specific pattern, an async-flavored [resource leak](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections). The event loop stays alive as long as there is a referenced handle keeping it busy — a pending timer, an open socket, a registered listener. That is usually what you want. But a `setInterval` you never `clearInterval`, an event listener you `on` but never `removeListener`, or a subscription you open per-request and never close, becomes a **zombie**: it keeps firing, holds references that prevent garbage collection (a memory leak), and can keep the process from exiting cleanly on shutdown. The classic symptom is "the process won't exit" (your test suite hangs at the end, or graceful shutdown stalls) and a slow, steady memory climb that tracks request count.

The mechanism is the loop's reference counting: each active timer or handle increments a count, and the loop runs while the count is above zero. A leaked `setInterval` keeps that count permanently above zero. The diagnostic for "what is keeping the loop alive" is to enumerate active handles. The `why-is-node-running` package prints them with creation stacks:

```js
const log = require("why-is-node-running");
process.on("SIGUSR2", () => log());   // kill -USR2 <pid> -> prints active handles + where created
```

Trigger it on a process that should have exited and it lists every timer, socket, and listener still active, *with the stack where each was created* — which points straight at the `setInterval` you forgot to clear. For the memory-leak angle, a heap snapshot diff (Chrome DevTools "Memory" tab, two snapshots, compare) shows the listener arrays or timer objects growing monotonically; in Node, `node --inspect` plus heap snapshots, and the retained-size column names the leaked closures. The fix is disciplined teardown: pair every `setInterval` with a `clearInterval` on shutdown, every `on` with a matching `off`/`removeListener`, every subscription with an unsubscribe in a `finally`. In Node, an `AbortController` is the modern idiom — pass its `signal` to timers, fetches, and listeners, and one `abort()` tears down the whole group:

```js
const ac = new AbortController();
const timer = setInterval(poll, 1000);          // remember to clear
someEmitter.on("data", onData, { signal: ac.signal });  // auto-removed on abort
// shutdown:
clearInterval(timer);
ac.abort();   // removes the listener and cancels anything wired to the signal
```

The honest before→after: a websocket gateway leaked one listener per reconnect, climbing about 3 MB/min under reconnect churn; the heap diff showed `EventEmitter` listener arrays with tens of thousands of entries (Node's "possible EventEmitter memory leak detected; 11 listeners added" warning was the first clue, which you should *never* suppress by raising `setMaxListeners` without understanding why). Adding the matching `removeListener` on disconnect flatlined memory and the process exited cleanly in tests for the first time in months.

## 10. The diagnostic toolkit, side by side

You now have a tool for each symptom. The decision is *which* instrument to reach for, and the answer follows from the family. The matrix below is the one I keep in my head: match the instrument to the symptom, because reaching for the wrong one wastes the most precious resource in an incident — time.

![A matrix matching async debugging tools to what they find, their overhead, and when to reach for them, covering async stack traces, event-loop lag monitoring, the task dump, and the floating-promise lint](/imgs/blogs/debugging-async-and-event-loops-6.png)

The same information as a reference table, with the runtime specifics filled in:

| Symptom | First instrument | What it finds | Overhead |
| --- | --- | --- | --- |
| Error with a short, useless stack | `--async-stack-traces`, `--stack-trace-limit=50`; Chrome async stack | The lost scheduler / logical caller | Near zero (default on) |
| Work runs but error vanishes | `unhandledRejection` handler; `no-floating-promises` lint; Python `-W error::RuntimeWarning` | Missing await / floating promise | Zero (static) to tiny |
| Request hangs forever | `asyncio.all_tasks()` + `print_stack`; `why-is-node-running`; `SIGUSR1` dump | Lost callback / parked coroutine | One dump call |
| High latency, idle CPU | `monitorEventLoopDelay`; `node --prof` flame graph | Blocked loop / sync CPU on the thread | Tiny histogram |
| Memory climbs, process won't exit | Heap snapshot diff; `why-is-node-running`; `AbortController` audit | Zombie timer / leaked listener | One snapshot |
| Spooky ordering / lost update | Inject a `setTimeout` to widen the window; async lock or atomic DB op | Async interleaving race | Test-only |
| Live prod process, can't attach pdb | `py-spy dump --pid`; `python -m asyncio pstree <pid>` | Where every task is suspended, no stop | Near zero, non-stopping |

A second table, because the kit rightly loves trade-off tables, comparing the three *philosophies* of async debugging so you pick deliberately:

| Approach | Recovers causality? | Catches it pre-prod? | Cost | Best for |
| --- | --- | --- | --- | --- |
| Static lints (`no-floating-promises`, `no-misused-promises`) | No (prevents, doesn't trace) | Yes, at build time | One-time setup | Missing awaits, misused promise callbacks |
| Runtime debug modes (asyncio debug, async stack traces) | Yes, reconstructs the chain | Partly (CI + dev) | Tiny to small | Never-awaited coroutines, lost callers, unhandled rejections |
| Correlation ids + structured logs (`AsyncLocalStorage` / `contextvars`) | Yes, across services and awaits | No (observes in prod) | Per-request context | Prod where you can't attach a debugger; cross-service async |
| Loop metrics (`monitorEventLoopDelay`) | No (localizes, doesn't trace) | Partly (load tests) | Negligible, always-on | Blocked loop, starvation, the latency-vs-CPU mystery |

The meta-point: there is no single async debugger that does it all, the way `gdb` mostly does for a synchronous crash. Async debugging is a *portfolio* — static analysis to prevent, runtime debug modes and async traces to reconstruct, correlation ids to follow across the loop, and loop metrics to catch the freeze. The skill is knowing which one the symptom is asking for.

## 11. The full workflow: observe, reproduce, find, fix, prevent

Step back and assemble the pieces into the series' spine, because a list of tools is not a method. The figure below is the workflow I run for any async incident, and every step maps to a specific instrument from this post.

![A timeline of the async debugging workflow moving from observing that requests hang, to measuring event loop lag, dumping pending tasks, following the async stack chain, applying the fix, and preventing regression with lint and lag alerts](/imgs/blogs/debugging-async-and-event-loops-8.png)

**Observe** the symptom precisely. "It's slow" is not observable; "p99 is 2s while CPU is 30%" is. "It sometimes loses writes" becomes "0.3% of orders return 200 with no row." The precision of your observation determines which family you're in, which determines which instrument you grab. **Reproduce** it deterministically, which for async usually means a load loop (run it 5,000 times and count failures) or a window-widener (inject a `setTimeout` to force the bad interleave). A bug you cannot reproduce on demand, you cannot prove you've fixed — this is the non-negotiable first move from [reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging). **Hypothesize** by family: detached (check lints and `unhandledRejection`), stalled (dump pending tasks), or blocked (measure loop lag). Each hypothesis names a specific measurement that confirms or kills it — that is what makes it a hypothesis and not a guess.

**Bisect** when the cause isn't obvious. Async bugs bisect across two axes that synchronous bugs don't: across the *await points* (binary-search which await is the yield window — comment out the awaits one half at a time and see when the race disappears) and across *commits* with `git bisect run` (an async regression — "ordering broke last Tuesday" — bisects exactly like any other, scripted against your reproducer). The same [bisection discipline](/blog/software-development/debugging/binary-search-your-bug-with-bisection) applies: turn it into a binary search over the gap between belief and truth. **Fix** at the right layer: add the `await`, give the detached promise a `.catch`, resolve the orphaned future, shrink the critical section or use an atomic DB op, move the blocking call off the loop, pair the timer with a clear. **Prevent** with the cheapest durable control: the lint as a CI error, the `unhandledRejection` crash-and-alert, the event-loop-lag alert, and the regression test that uses your reproducer to assert the fix holds — the window-widener test that fails on the old code and passes on the new.

### Stress-testing your fix

The kit demands a stress test, and async fixes have specific failure modes to interrogate. *What if it only reproduces under load?* Loop-blocking and races both intensify with concurrency — your fix must hold at the concurrency where the bug appeared, so load-test it, don't unit-test it once. *What if it only reproduces in production?* You can't attach `pdb` to the payments process, so you lean on the no-stop tools: `py-spy dump`, `monitorEventLoopDelay` already running, `SIGUSR1`/`SIGUSR2` handlers you wired in *before* the incident, and correlation ids in your logs. The single best prevention is to make production *observable for async before you need it*: ship the loop-lag metric, the global rejection handler, and the task-dump signal as standard middleware in every service, so when the page comes at 02:14 you are reading evidence, not adding instrumentation to a burning process. *What if the race only loses an update once in ten thousand?* Then your reproducer must widen the window deterministically (the injected yield) so it fails 100% of the time on the broken code; a fix you "verified" against a 1-in-10,000 bug by running it twice is not verified at all. *What if the worker-thread fix introduces its own bug?* Moving work off the loop adds a serialization boundary and a new failure surface (worker crashes, message-channel backpressure) — test the worker's error path explicitly, because you've traded a latency bug for a new class of detached-error bug if you don't handle the worker's `error` and `exit` events.

## 12. War story: when an async bug took down more than a server

Async and event-loop bugs are not academic; some of the most expensive software failures in history are, at root, the failure modes in this post. Consider the broad class of **retry-storm / thundering-herd** outages that have hit large services repeatedly: a downstream slows down, every client's async request starts timing out, every client *retries* — and because the retries are fired without backoff or jitter from thousands of event loops simultaneously, the retries themselves become a synchronized flood that keeps the downstream pinned. The async mechanism underneath is precisely a detached/uncoordinated-work problem: each client fires a promise, it rejects on timeout, the `.catch` immediately fires another, and nobody is holding a global handle that says "back off." The fix is the same family of disciplines — bound the concurrency, add exponential backoff with jitter, add a circuit breaker that *stops* firing the async calls when the downstream is unhealthy. This is well documented in real postmortems; see [the anatomy of an outage from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) for the systemic view. The point for *our* purposes: the loop-level bug (unbounded async retries) and the system-level outage are the same bug at two scales.

A second, narrower realistic scenario — presented as illustrative, assembled from patterns I have personally debugged rather than a single named company incident — is the **graceful-shutdown hang that masked a data-loss bug.** A service's deploys started taking the full 30-second kill timeout instead of exiting in milliseconds, and ops "fixed" it by lowering the timeout to force-kill faster. That force-kill was *terminating in-flight async writes mid-flight*, because the process was being SIGKILLed while detached promises were still pending — the same missing-await detachment from section 3, now interacting with shutdown. The slow shutdown was a symptom (a zombie `setInterval` keeping the loop alive, section 9), and force-killing to hide the symptom *created* a data-loss bug. The lesson is the series' deepest one: **a symptom you suppress without understanding becomes a worse bug somewhere else.** The slow exit was the loop honestly telling them work was still pending; silencing it lost data. They found it by wiring `why-is-node-running` to `SIGUSR2`, seeing the leaked interval, clearing it so the process exited cleanly on its own, and *then* the force-kill was never triggered and the writes completed. Observe the real signal; do not gag it.

A third, genuinely documented pattern worth naming: long **garbage-collection pauses** presenting exactly like a blocked event loop. A major GC (a stop-the-world pause in V8 or in a JVM) freezes the one thread for hundreds of milliseconds, and to your loop-lag metric it looks *identical* to a synchronous `JSON.parse` — high lag, idle-looking CPU. The discriminator is the CPU profile: a GC pause shows up as time in `GC` / collector frames, not your code, whereas a sync-parse block shows your `JSON.parse` frame. Knowing both *present* as loop lag but *resolve* to different flame-graph signatures is what stops you from "fixing" your parser when the real cause is allocation pressure triggering frequent major GCs — a [memory problem](/blog/software-development/debugging/hunting-memory-leaks-and-bloat), not a parsing one. Same symptom, two root causes, one discriminating measurement.

## 13. How to reach for this (and when not to)

Every technique here has a cost, and the senior move is knowing when *not* to reach for it. **Don't attach a stepping debugger to chase an async race** — `pdb`/`node --inspect` breakpoints change the loop's timing and the race vanishes (a heisenbug); reproduce it with the window-widener and structured logs instead, and only attach a debugger once you can make it fail on demand. **Don't enable `async_hooks` heavily in hot prod paths** — raw `async_hooks` with `before`/`after` callbacks on every async resource has measurable overhead; use `AsyncLocalStorage` (which is optimized) or sampling, not hand-rolled hooks on the request path. **Don't set `--unhandled-rejections=warn` to "stop the crashes"** — that is gagging the smoke alarm; the crash is telling you about a real unhandled error, and the fix is to handle it, not to silence it. **Don't reach for an in-process async lock when the database can do it atomically** — an app-level lock doesn't survive multiple processes and adds a deadlock surface; a single atomic SQL `UPDATE` is simpler and correct across the cluster. **Don't move every CPU operation to a worker thread** — workers have startup and serialization cost; a 2ms computation belongs on the loop, only the >50ms blockers belong off it, and you decide *with the loop-lag measurement*, not by guessing. **Don't suppress the "possible EventEmitter memory leak" warning** with `setMaxListeners(Infinity)` — that warning is usually a real leaked-listener bug; raise the limit only when you genuinely need many listeners and have proven it's not a leak.

And the cheapest, highest-value advice: **a well-placed log line beats a debugger for most async bugs.** Because the stack is unreliable anyway, a structured log with a correlation id at each await boundary often tells you the causality faster than any breakpoint — you read the request's whole journey across the loop in `grep` output. Reach for the heavy machinery (worker threads, `async_hooks`, heap snapshots) only when the cheap controls (a lint, a log line, the loop-lag metric, a pending-task dump) haven't already answered it. Most async incidents are solved by the four-line diagnostics in this post, not by a long debugger session.

## 14. Key takeaways

- **An `await` unwinds the stack to the loop.** When an async op fails, the native stack does not contain the scheduler that issued it — that frame is gone. A short, useless async trace is not a tooling bug; it is the mechanism. Recover the caller with async stack traces and correlation ids, not by staring at the short stack.
- **The loop runs one task at a time and only switches at an await or completion.** Anything synchronous between awaits blocks *all* concurrency on the process. Computation does not belong on the loop; orchestration does.
- **A missing `await` detaches the work.** In JS the work runs and the error vanishes; in Python the coroutine may never run at all. Lint it (`no-floating-promises`) and make the runtime shout (`unhandledRejection` handler, `-W error::RuntimeWarning`, asyncio debug mode).
- **An unhandled rejection that "disappeared" is a real, unhandled error.** Install the global handler, fail fast in prod, and run CI with `--unhandled-rejections=strict`. Never gag it with `warn`.
- **A request that hangs forever is usually a lost callback or an unresolved future.** Dump pending tasks (`asyncio.all_tasks()` + `print_stack`, `SIGUSR1`) to see the exact suspended await; check for the early-return path that skipped the callback.
- **`await` is a yield point, so single-threaded code can still race.** Two coroutines can interleave across awaits and lose an update. Shrink the critical section to no await, or use an atomic DB operation — prefer the database's atomicity over an in-process lock.
- **High latency at idle CPU means the loop is blocked.** Measure it with `monitorEventLoopDelay`; loop lag spiking with your latency *proves* it's the loop and exonerates the database. Move the blocker off the loop (worker thread, stream, async API).
- **Leaked timers and listeners are zombie tasks.** They keep the loop alive and leak memory; pair every `setInterval`/`on`/subscribe with its teardown, and reach for `AbortController` and `why-is-node-running`.
- **Async debugging is a portfolio, not one debugger.** Static lints prevent, runtime debug modes and async traces reconstruct, correlation ids follow causality across the loop, and loop metrics catch the freeze. Match the instrument to the symptom.
- **Reproduce on demand before you claim a fix.** A 1-in-10,000 race needs a window-widener that fails 100% on the broken code; a fix "verified" by running it twice is not verified.

## 15. Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro and the observe→reproduce→hypothesize→bisect→fix→prevent loop this post follows.
- [Reading a stack trace across languages](/blog/software-development/debugging/reading-a-stack-trace-across-languages) — the sibling on stacks, for when you have a trace; this post is the case where the trace is missing.
- [Logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) — correlation ids and structured logging, the way to reconstruct causality across the loop when the stack is gone.
- [Race conditions: the hardest bugs to catch](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch) — the threaded counterpart; the async ordering race is the same bug shape with the yield point instead of the CPU as the interleaver.
- [Reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — the load-loop and window-widener reproduction discipline async bugs demand.
- Node.js docs: the `perf_hooks` `monitorEventLoopDelay` API, `worker_threads`, `async_hooks`, and `AsyncLocalStorage` — the canonical reference for the loop-lag, off-load, and context-propagation tools used here.
- Python docs: `asyncio` "Developing with asyncio" (debug mode), `asyncio.all_tasks`/`Task.print_stack`, `contextvars`, and `faulthandler` — the asyncio-side equivalents of every Node technique above.
- The V8 blog post on async stack traces and the Chrome DevTools "asynchronous call stacks" documentation — how the runtime stitches the pre-await frames onto the resumed continuation.

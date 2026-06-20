---
title: "Function Coloring and Bridging Sync and Async"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why async infects your whole call stack, how to cross the sync-async boundary without freezing the event loop, and when virtual threads make the whole problem disappear."
tags:
  [
    "concurrency",
    "parallelism",
    "async-await",
    "function-coloring",
    "blocking",
    "virtual-threads",
    "event-loop",
    "runtime",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/function-coloring-and-bridging-sync-and-async-1.png"
---

A team I worked with shipped a perfectly ordinary feature: a "recent activity" panel on a user's dashboard. It called a small reporting database to fetch a few rows. The reporting database was slow and occasionally locked up under its own batch jobs, so that one query could take anywhere from 5 milliseconds to almost a full second. On the old synchronous, thread-per-request server this was a non-event — a slow request tied up one worker thread, the other workers kept serving, and at worst that single user saw a spinner.

Then the service was rewritten on an async runtime to handle far more connections per box. The activity-panel code was ported over almost line for line. The database client they reached for was a plain synchronous driver — the async one wasn't ready yet — so the new handler looked async on the outside (`async def handler(...)`) but, buried three calls down, made a regular blocking call into that synchronous driver. In a load test everything was fine. In production, the first time the reporting database stalled for 800 milliseconds, the entire server stopped. Not that one request — *every* request. Roughly ten thousand open connections, all of which had been happily multiplexed onto a single event-loop thread, froze in lockstep until the slow query returned. Health checks timed out. The load balancer ejected the instance. The on-call engineer saw a service "hang" with the CPU near idle and no obvious culprit, because nothing had crashed — one thread was simply *busy waiting* on a socket while holding the only thread that could make progress.

What makes this failure so memorable is how *innocent* every individual decision looked. Choosing an async runtime to handle more connections: correct. Porting the handler over almost unchanged: reasonable, since the logic was identical. Using the synchronous DB driver because the async one wasn't ready: pragmatic. Each step was defensible in isolation, and the result was a single hidden line that could take the whole service down. There was no race condition, no deadlock between locks, no memory corruption — none of the classic concurrency bugs this series has spent posts dissecting. It was something simpler and, in a way, more insidious: a *blocking* call in a place where blocking is forbidden, and nothing in the type system, the tests, or the code review flagged it, because in *most* runs that call returned in five milliseconds and bothered no one.

That bug is the subject of this whole post. It sits at the seam between two worlds — synchronous code that *blocks* a thread and asynchronous code that *yields* a coroutine — and the seam is treacherous in both directions. We'll start from the question that names the friction, Bob Nystrom's famous framing: **what color is your function?** Async functions and sync functions obey different calling rules, async "infects" everything that calls it, and a blocking call dropped into an async context is a loaded gun. Then we'll do the practical work: how to *bridge* the boundary safely in each direction, how to detect the cardinal sin of blocking the event loop, why Go and Java's virtual threads make the whole coloring problem evaporate, and exactly how the `spawn_blocking` fix would have saved that frozen server. The figure below is the destination — the same blocking call, on the loop versus offloaded.

![A comparison showing a blocking call on the event loop thread freezing all tasks versus the same call offloaded to a thread pool leaving the loop responsive](/imgs/blogs/function-coloring-and-bridging-sync-and-async-1.png)

This post builds directly on two earlier ones. If async/await still feels like magic, read [how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) first; if the event loop is the fuzzy part, [the event loop and the reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) is the companion. This one is about the *boundary* between async and the ordinary blocking world, and how not to get burned crossing it.

## Two colors: what coloring actually means

In 2015 Bob Nystrom wrote an essay titled *"What Color Is Your Function?"* that gave the whole industry a name for a frustration everyone felt but couldn't articulate. The setup is a thought experiment. Imagine a language with two kinds of functions — call them **red** and **blue**. Blue functions are the normal ones. Red functions are special, and they come with rules:

1. You call a red function differently from a blue one (you must `await` it, or pass it a callback, or otherwise treat it specially).
2. **You can only call a red function from inside another red function.** A blue function cannot directly call a red one and get its result.
3. Red functions are "more painful" to call — more ceremony, more places to get it wrong.

In every real async/await language, **red is async and blue is sync.** An `async fn` in Rust, an `async def` in Python, an `async function` in JavaScript, a `suspend fun` in Kotlin — these are all red. A `Future` or a coroutine doesn't *do* anything until it's driven by an executor, and the only ergonomic way to drive it and get the value out is `.await` (or its equivalent), which is itself only legal inside another async function. Ordinary functions — the ones that just run, block if they have to, and return a value — are blue.

The matrix below states the calling rules precisely. This is the whole ballgame: who can call whom, and what happens to the thread.

![A matrix of function colors showing what sync and async functions can call and whether each one blocks the thread](/imgs/blogs/function-coloring-and-bridging-sync-and-async-2.png)

Read the cells carefully, because the asymmetry is the source of all the pain. A blue (sync) function can call another blue function trivially — that's just a normal call. A red (async) function can `await` another red function trivially — that's the happy path async was designed for. The two off-diagonal cells are where the bridges live. A red function calling a blue one *works* syntactically (you just call it), but if that blue function blocks, the red function blocks the underlying thread — and on an event loop that thread is shared by thousands of tasks, which is the disaster from the intro. A blue function calling a red one *doesn't work at all* without a bridge: a sync function has no `await`, so it cannot drive a future to completion on its own. It has to hand the future to a runtime and ask the runtime to block until it's done.

#### Worked example: tracing one call across the boundary

Make it concrete. Say `read_user(id)` is async (red) because it does network I/O, and `format_report(user)` is sync (blue) because it just munges strings. A request handler needs both.

```python
async def handle(request):
    # handle is async (red) because it awaits read_user
    user = await read_user(request.id)   # red -> red: fine, await it
    line = format_report(user)            # red -> blue: fine IF format_report is fast
    return line
```

That `format_report` call is the dangerous-looking off-diagonal one — red calling blue. It is *safe here only because `format_report` is fast and pure CPU* (microseconds). The instant `format_report` is replaced by something that blocks — a synchronous HTTP call, a `time.sleep`, a synchronous database driver, reading a large file with a blocking read — the red function blocks its thread, and on an event loop that means the cardinal sin. The color rules don't catch this. The compiler is happy. The type checker is happy. It runs fine in tests. It blows up under production timing. That gap between "compiles and tests pass" and "freezes under load" is exactly why coloring is dangerous rather than merely annoying.

The other direction is caught by the compiler, which is honestly a mercy. If `format_report` were itself sync and tried to call the async `read_user` directly:

```python
def format_report(user_id):
    user = read_user(user_id)   # blue -> red WITHOUT await: returns a coroutine, NOT a user
    return user.name            # AttributeError: 'coroutine' object has no attribute 'name'
```

You don't get a user; you get a coroutine object that never ran. In Rust the same mistake is a hard compile error — a `Future` is not its output type, and you cannot `.await` outside an async context. So the blue→red direction fails loudly. The red→blue direction fails silently and only under load. That asymmetry shapes everything that follows.

It's worth being precise that *coloring is not just an async/await thing* — Nystrom's deeper point is that any property a function can have that its callers must respect is "a color." A function that can `throw` a checked exception colors its callers (they must `try`/`catch` or re-declare it). A `nothrow` / `noexcept` boundary is a color. A `const` method in C++ is a color. A function that requires the GPU context, or must run on the UI thread, or must hold a particular lock — all colors, all infectious in the same way. Async is simply the most *visible* and most *pervasive* coloring, because it touches every function that does I/O, which in a server is nearly all of them. Recognizing that the pattern generalizes is useful, because the *fixes* generalize too: you either propagate the property up to a boundary that can satisfy it, or you install an adapter at a chosen seam so the property stops spreading. That's exactly the bridge-or-propagate choice we'll keep making.

#### Worked example: the coroutine that silently never ran

The silent failure is worth seeing once in full, because it's the one that gets shipped. A developer wrote a sync cleanup hook that "called" an async cache-eviction function:

```python
def on_shutdown():
    evict_all_caches()   # this is async; called without await it does NOTHING
    log.info("caches evicted")   # logs success even though nothing happened
```

`evict_all_caches()` returns a coroutine object. The coroutine is never awaited, never scheduled, never run. Python emits a `RuntimeWarning: coroutine 'evict_all_caches' was never awaited` — *to stderr*, which nobody was watching — and the log cheerfully reports success. The caches were never evicted. This kind of bug surfaces weeks later as stale data, and the log says everything worked. The fix is to bridge properly (`asyncio.run(evict_all_caches())` from the sync hook, since the hook is at the bottom of a sync stack). The lesson: a returned-but-unawaited coroutine is a no-op that *looks* like a call, and the only thing that catches it is a warning most people never see — so wire `-W error::RuntimeWarning` (or its equivalent) in CI to turn that warning into a test failure.

## Why async infects the call stack: the async tax

Here is the mechanism that makes coloring more than a curiosity — the reason a single async leaf reorganizes an entire codebase. It follows directly from rule 2 (you can only `await` inside an async function) plus the fact that `await` is how you get a value out of a future.

Suppose you have a clean synchronous call chain: `handler → service → repository → fetch_row`. All blue. Now the database team ships an async driver and you want to use it, so `fetch_row` becomes async — it `await`s a socket. The moment `fetch_row` is red, look at what happens to its caller, `repository`. To get the row, `repository` must `await fetch_row()`. But `await` is only legal inside an async function. So `repository` must become async too. Now `service` calls `repository`, must `await` it, and therefore *service* must become async. Then `handler`. The color propagates up the entire stack from the leaf to the root, one frame at a time, until it hits something that owns a runtime and can finally bridge back to blue.

![A before and after view showing one synchronous leaf function turning async and forcing every caller above it up the stack to also become async](/imgs/blogs/function-coloring-and-bridging-sync-and-async-4.png)

People call this **the async tax**: making one thing async costs you a wave of edits everywhere above it. It's not a metaphor — it's a mechanical consequence of the calling rules. And the tax is *directional*: it only flows upward, from callee to caller, never down. A red function can sit anywhere in your tree and force every ancestor red, but it never forces its own sync helpers to change color.

Seeing the wave concretely makes it stick. Here is the same chain before and after the leaf goes red, in Rust where the color is in the type system:

```rust
// BEFORE: all blue. Ordinary calls, ordinary returns.
fn handler(id: u64) -> Response { render(service(id)) }
fn service(id: u64) -> User     { repository(id) }
fn repository(id: u64) -> User  { fetch_row(id) }
fn fetch_row(id: u64) -> User   { /* sync cache read */ db_cache_lookup(id) }

// AFTER: fetch_row went async (it now awaits a socket). The color spreads UP.
async fn handler(id: u64) -> Response { render(service(id).await) }   // edited
async fn service(id: u64) -> User     { repository(id).await }        // edited
async fn repository(id: u64) -> User  { fetch_row(id).await }         // edited
async fn fetch_row(id: u64) -> User   { async_db_lookup(id).await }   // the one real change
```

Three functions changed for one intended change, and every call site in *their* callers had to add `.await` too. Multiply by a deep call graph and you see why a "small" change to make one driver async can become a multi-day refactor that touches half the codebase. The tax is real money.

Why can't the runtime just paper over this? Because async and sync are genuinely different execution models, not two flavors of the same call. A sync call pushes a frame on the OS thread's stack and the thread *is* the call — the thread cannot do anything else until the call returns. An async "call" creates a state machine (a coroutine / future) that the caller must *drive*: poll it, and when it isn't ready, get back a "not yet, here's how to wake me" so the caller can go run something else. To get a value out of that state machine you have to participate in the driving protocol, and `await` is the syntax for participating. A plain sync function has no way to participate — it has nowhere to yield to. So the only honest options are (a) propagate the asyncness up to someone who can drive it, or (b) install a bridge.

#### Worked example: counting the blast radius

A real case I measured. A service had a config-loading function `get_setting(key)` called from 47 sites. It read from a local cache, so it was sync and fast. Product wanted settings to be hot-reloadable from a remote store, which meant an occasional network fetch on a cache miss — naturally async. Making `get_setting` async would have turned all 47 callers red, and *their* callers, and so on. The estimate came to roughly 180 functions touched, most of them in code that had nothing to do with settings. We did not pay that tax. Instead we kept `get_setting` sync and moved the network refresh into a *background* async task that periodically repopulated the cache, so the hot path stayed blue and the slow path lived entirely on the async side. That's the pattern to internalize: **the cheapest way to avoid the async tax is often to not let the async cross the boundary at all** — push it to the edge, behind a sync-looking cache or queue, so the color never propagates.

This is also why the language designers who chose async/await knew exactly what they were signing up for and did it anyway: in exchange for the tax, you get extremely cheap concurrency for I/O-bound work — thousands of in-flight operations on a handful of threads, with no per-task stack — which we covered in [blocking versus non-blocking I/O and the C10k problem](/blog/software-development/concurrency/blocking-vs-non-blocking-io-and-the-c10k-problem). The tax is the price of the C10k win. The rest of this post is about paying it without bankrupting yourself at the boundary.

## Bridging sync to async: running a future from blue code

Eventually red has to meet blue. Your `main` is typically sync. Your test harness is sync. A library callback handed to you by a sync framework is sync. A CPU-bound worker thread that occasionally needs one async result is sync. In all of these a blue function holds an async future and needs its value *now*. That's the sync→async bridge, and every runtime ships one.

The bridge is always the same shape: **hand the future to the runtime and tell it to block the current thread until the future completes**, then return the result. The runtime spins its event loop on *this* thread, driving the future (and anything it spawns) to completion, and only then returns control. Names vary:

```rust
// Rust + tokio: block the current thread on a future
fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let user = rt.block_on(async {
        read_user(42).await   // drive this future to completion on THIS thread
    });
    println!("{}", user.name);
}
```

```python
def main():
    # Python asyncio: run a coroutine to completion from sync code
    user = asyncio.run(read_user(42))   # creates a loop, runs the coroutine, tears it down
    print(user.name)
```

```javascript
// JavaScript has no blocking bridge in the browser/Node main thread.
// The closest is to start the chain and let the runtime's loop drive it;
// you CANNOT synchronously wait for a promise on the main thread.
async function main() {
  const user = await readUser(42); // you must already be red to await
  console.log(user.name);
}
main(); // fire it; the event loop drives it; main() returns a pending promise
```

Notice JavaScript is the odd one out and it's instructive. In Node and the browser there is *no* `block_on` you can call on the main thread — the language deliberately refuses to let blue code synchronously wait for red, because the main thread *is* the event loop and blocking it would freeze the page or the server. So JS forces you to either already be async, or move the blocking expectation off the main thread entirely (a Worker). That's a design stance: rather than offer a footgun, remove it. Rust and Python *do* give you `block_on`/`asyncio.run`, which is more flexible and more dangerous.

More dangerous because of the **`block_on`-inside-async deadlock**, the single nastiest trap on this side of the boundary. Here's the mechanism. `block_on` blocks the *current* thread and runs an event loop on it. If you call `block_on` from a thread that is *already* running an event loop — that is, from inside an async task — you now have a thread trying to run a second loop while the first loop is suspended mid-poll, waiting for the very task that just called `block_on` to yield back. It never will, because it's blocked in `block_on`. The outer loop can't make progress (its thread is stuck), the inner future may depend on the outer loop's resources (a connection pool, a timer), and nothing wakes anyone. Frozen.

```rust
// DEADLOCK: block_on called from inside an async task already on the runtime
async fn handler() -> User {
    // we are ALREADY on a tokio worker thread running the loop
    let rt = tokio::runtime::Handle::current();
    // block_on from within the runtime: tokio will actually panic here
    // ("Cannot start a runtime from within a runtime"), but the conceptual
    // hazard is a self-deadlock — the thread waits for a loop it is blocking.
    rt.block_on(async { read_user(1).await })  // WRONG
}
```

tokio is defensive enough to *panic* with "Cannot start a runtime from within a runtime" rather than silently hang, which is a gift — a loud failure beats a frozen one. Other runtimes and other languages are less kind; some genuinely deadlock. Python's `asyncio.run` raises `RuntimeError: asyncio.run() cannot be called from a running event loop` for the same reason. The rule that prevents all of this: **`block_on`/`asyncio.run` is for the boundary at the *bottom* of a sync stack — `main`, a test, a sync worker thread that owns no loop — never from inside async code.** If you're already async and you need an async value, you `await` it. You only `block_on` when you're genuinely blue and need to enter red from the outside.

To see *why* `block_on` deadlocks from within async, you have to know what it actually does. A future is a state machine with a `poll` method: poll it and it either returns `Ready(value)` or `Pending` along with a registered *waker* — a callback the future will invoke when it might be ready to make progress. `block_on` implements the simplest possible driver: poll the future; if `Ready`, return the value; if `Pending`, **park the current OS thread** and wait for the waker to unpark it; when unparked, poll again; repeat. The key phrase is "park the current OS thread." If that thread is the one already running an event loop, parking it stops the loop. Now suppose the inner future's waker is supposed to be fired *by that same loop* (because the inner future is, say, a timer or a socket read managed by the loop you just parked). The waker can never fire, because its loop is parked, so `block_on` waits forever for an unpark that will never come. That's the mechanism — not a quirk, a direct consequence of `block_on` parking the very thread the inner work depends on. The runtimes that panic are detecting this re-entry and refusing; the ones that don't, hang.

There's a subtler variant worth naming: even on a *multi-threaded* runtime, calling `block_on` (or `futures::executor::block_on`) from a worker thread can deadlock if the future you block on needs *that specific worker thread* to make progress, or if blocking that worker shrinks the runtime's thread pool below what the workload needs to complete. tokio offers `block_in_place` for the narrow case where you must run blocking code on a worker without leaving the runtime — it tells the scheduler "this thread is about to block, move other work off it" — but it only works on the multi-threaded runtime and it's a sharp tool. The safe mental model stays simple: bridging *up* out of sync into async belongs at the bottom of the stack; everywhere else, propagate or `await`.

## Bridging async to sync: offloading blocking work to a pool

Now the direction that actually caused the intro outage. You're inside an async task (red), you're on the event-loop thread, and you must call something blue that *blocks* — a synchronous DB driver, a CPU-heavy hash, a legacy library, a file read on a slow disk. If you call it directly, you block the loop thread, and every other task on that loop freezes (next section proves why). The fix is to **move the blocking call off the loop thread onto a dedicated pool of worker threads**, and `await` a future that resolves when the worker finishes.

![A branching graph showing an async task handing a blocking call to a pool worker thread and the result returning to the task while the loop stays free](/imgs/blogs/function-coloring-and-bridging-sync-and-async-3.png)

Trace the figure. The async task, instead of running the blocking call inline, calls an "offload" primitive that submits the work to a **blocking thread pool** (a bunch of OS threads kept around exactly for this). A pool worker picks up the job and runs the synchronous, blocking call there — where it's *allowed* to block, because that thread does nothing but wait for blocking calls. When the call returns, the worker resolves a future; the async task, which had `await`ed that future and so *yielded the loop thread* the whole time, gets woken and resumes. The loop thread was free for the entire 800 milliseconds. Crucially, only the *result* crosses back to the loop — the blocking happened entirely off to the side.

The primitive's name, again, varies:

```rust
// Rust + tokio: run a blocking call on the blocking-thread pool
async fn fetch_recent(id: u64) -> Vec<Row> {
    // sync_db_query BLOCKS; spawn_blocking moves it to tokio's blocking pool
    tokio::task::spawn_blocking(move || {
        sync_db_query(id)        // runs on a pool thread; the loop thread is free
    })
    .await                       // await the JoinHandle; we yield the loop while it runs
    .expect("blocking task panicked")
}
```

```python
import asyncio  # Python asyncio: run a blocking call in a thread-pool executor

async def fetch_recent(id: int):
    loop = asyncio.get_running_loop()
    # run_in_executor pushes sync_db_query onto a ThreadPoolExecutor
    rows = await loop.run_in_executor(None, sync_db_query, id)  # None = default pool
    # Python 3.9+: asyncio.to_thread(sync_db_query, id) wraps this more cleanly.
    return rows
```

```java
// Java pre-Loom: a CompletableFuture backed by a bounded blocking executor
ExecutorService blockingPool =
    Executors.newFixedThreadPool(16);   // bounded! sized for the blocking work

CompletableFuture<List<Row>> fetchRecent(long id) {
    return CompletableFuture.supplyAsync(
        () -> syncDbQuery(id),          // runs on a blockingPool thread
        blockingPool);                  // explicit executor, never the common pool
}
```

Three languages, one idea: the blocking call never runs on the thread that drives other tasks. tokio even runs its event loop on a small fixed set of worker threads but keeps a *separate, larger* pool (default up to 512 threads) exclusively for `spawn_blocking`, precisely because blocking work needs threads you're willing to "waste" sitting in a syscall. Python's default executor and Java's explicit `ExecutorService` are the same pattern.

Two things matter enormously here, and both bite people. **First: the pool must be bounded and sized for the blocking work, not unbounded.** If you offload to an unbounded pool and load spikes, you spawn thousands of OS threads, each with its own stack, and you trade a frozen loop for an out-of-memory crash or death by context-switching. A bounded pool gives you natural backpressure — past the pool's capacity, new offloads queue, which is the behavior you want (slow, not dead). This connects to [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) at the architecture level. **Second: only blocking or genuinely CPU-bound work belongs on the pool.** Offloading a fast pure-CPU function (microseconds) is *net negative* — the cost of handing it to another thread, scheduling it, and handing the result back dwarfs the work. The matrix below summarizes the two bridges and the trap waiting at the end of each.

![A matrix of the two bridge directions listing the runtime tool for each and the specific failure mode that each one risks](/imgs/blogs/function-coloring-and-bridging-sync-and-async-5.png)

#### Worked example: when offloading is the wrong move

A team saw "blocking call" warnings from their loop-lag monitor on a function that computed a SHA-256 of a 200-byte token — call it 2 microseconds of pure CPU. Someone "fixed" it by wrapping every call in `run_in_executor`. Throughput *dropped* by about 30%. The reason: each offload costs on the order of 10–50 microseconds of thread handoff and scheduling, so they paid 20x the work's cost in pure overhead, plus they saturated the thread pool with trivial jobs and starved the *actually* blocking DB calls that needed those threads. The right call was to leave the 2 µs hash inline (it's far below the ~100 µs threshold where a stall is even noticeable) and reserve the pool for the things that truly block. **The rule of thumb: offload work that blocks on I/O or runs longer than roughly 100 microseconds; leave faster pure-CPU work inline.** Measure your own threshold — it depends on your offload cost and your latency budget.

There's a second, language-specific subtlety that trips people offloading *CPU-bound* work specifically. A thread pool keeps the event loop responsive for *blocking I/O* because a thread parked in a syscall releases the CPU (and, in Python's case, releases the GIL — most blocking I/O calls drop the GIL while they wait). But a thread pool does **not** give you true parallelism for *pure-CPU* work in a runtime with a global interpreter lock: offloading a CPU-heavy Python function to a `ThreadPoolExecutor` moves it off the loop thread (good — the loop stays responsive) but the worker still holds the GIL while it computes, so it doesn't run in parallel with other Python code and won't speed the *aggregate* throughput up. For CPU-bound work that you also want *parallel*, the offload target is a *process* pool (`ProcessPoolExecutor` via `run_in_executor`), which pays a serialization cost to ship arguments and results across the process boundary — this is exactly the trade covered in the Python series' [multiprocessing post](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code). In Rust, Go, Java, and C++ a thread pool *does* run CPU work in parallel (no global lock), so `spawn_blocking` for a heavy CPU job both unblocks the loop and uses another core. Know which kind of pool your runtime gives you before you assume an offload bought you parallelism — it may only have bought you a responsive loop.

#### Worked example: the runaway offload during a downstream stall

A payments service offloaded every outbound call to a fraud-scoring API through `run_in_executor` with the *default* (unbounded-ish) executor. Normally the fraud API answered in 30 ms and a handful of pool threads sufficed. One afternoon the fraud API degraded to 5-second responses. Requests kept arriving at 400/second, each grabbing a pool thread for 5 seconds, so by Little's law the pool needed $L = 400 \times 5 = 2000$ threads to keep up — and since the executor grew on demand, it tried. Two thousand OS threads later (each ~1 MB of stack, ~2 GB of memory just in stacks) the box hit memory pressure, the kernel started thrashing, and the *whole* service — not just the fraud path — slowed to a crawl. The fix was two lines: cap the executor at 32 threads (sized to what the fraud API could actually handle without making things worse), and give the offloaded call a timeout so a stuck request frees its thread instead of holding it for the full 5 seconds. After the cap, a fraud-API stall produced a *bounded* backup — fraud-path requests queued and some timed out with a clean error, while the rest of the service stayed healthy. The unbounded pool had turned a *dependency's* slowness into *your* outage; the bound turned it back into a contained, observable degradation.

## Never block the event loop, and how to detect it

This is the cardinal sin, so let's prove *why* it's fatal rather than just repeating the warning. We covered the reactor in [the event loop post](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern); here's the consequence for blocking. An event loop is a single thread running, forever, a loop like: "ask the OS which of my thousands of sockets are ready; for each ready one, run its task until that task hits its next `await` and yields back; repeat." The entire scheme depends on **every task yielding quickly.** Cooperative scheduling has no preemption — nothing can forcibly take the thread back from a task. So if one task, instead of `await`ing, makes a synchronous blocking call, it holds the loop thread for the *full duration of that call*, and the loop literally cannot do the "ask the OS which sockets are ready" step, cannot run any other task, cannot even respond to a health check. Every one of the other ten thousand tasks is frozen, regardless of whether *their* I/O is ready.

![A timeline showing a blocking DB call seizing the event loop thread while two ready tasks are forced to wait until the call finally returns](/imgs/blogs/function-coloring-and-bridging-sync-and-async-6.png)

The timeline shows it step by step. Task A `await`s a query at t0 — good, it's about to yield. But the query is a *synchronous* call, so at t1 it doesn't yield; it enters the blocking driver and holds the thread. At t2 and t3, tasks B and C become ready (their data arrived), but the loop can't dispatch them — its only thread is stuck in A's blocking call. At t4 the DB finally returns after 800 ms. Only at t5 does the loop resume and drain the backlog. From the outside, the service "hung" with idle CPU. That idle CPU is the tell: a blocked event loop is not *busy*, it's *waiting* — one thread parked in a syscall while everything queues behind it.

How do you *detect* this before it pages you? Three layers, cheapest first.

**1. Loop-lag / event-loop-stall monitors.** The loop knows how long each "tick" took. If a single task ran for, say, more than 100 ms without yielding, that's a smoking gun. Most runtimes expose this. Node has `perf_hooks` and libraries that sample the event-loop delay; Rust's tokio has an unstable task-budget / "long-running-task" instrumentation and the `tokio-console` tool; Python can install a slow-callback warning:

```python
import asyncio  # make asyncio shout when one callback hogs the loop
loop = asyncio.get_event_loop()
loop.slow_callback_duration = 0.1   # warn if any callback runs > 100 ms
asyncio.run(main(), debug=True)     # debug mode logs: "Executing <Task ...> took 0.812 seconds"
```

That one line would have turned the intro outage from a 3 AM mystery into a log line naming the exact coroutine and its 0.812-second stall. Turn it on in staging.

**2. Blocking-call linters / static analysis.** Catch the bug before it ships. Tools scan for known-blocking calls inside async functions: `flake8-async` / `ASYNC` rules (formerly `flake8-trio`) flag `time.sleep`, blocking `requests`, sync file I/O, and `subprocess` inside `async def`. Rust's `clippy` and tokio's docs steer you toward `spawn_blocking`. ESLint plugins flag sync `fs` calls in async handlers. These are cheap to wire into CI and catch the *obvious* offenders (`time.sleep(1)` in a handler) that account for most incidents.

**3. A blocking-detector in the loop itself.** The most thorough option: a watchdog thread that pings the loop and screams if the loop doesn't answer within N milliseconds, which catches blocking calls the linter can't see (a blocking call hidden inside a third-party library). tokio's `tokio-console` and various "block-in-async" detectors do this. It's more setup but it catches the *unknown* offenders.

Layer them. The linter catches `time.sleep`. The loop-lag warning catches the surprise blocker that slipped through. The watchdog catches the one buried in a dependency. Defense in depth, because a single blocking call anywhere on the loop takes down everything.

There's one more detection signal that costs nothing and catches a whole category of blockers the linter misses: **watch your CPU and your latency *together*.** A blocked event loop has a signature unlike any other failure — latency spikes (or the service hangs outright) while CPU sits *idle*. Compare that to a CPU-bound overload, where latency spikes and CPU is *pinned*. The two look identical on a latency graph and opposite on a CPU graph, and the difference tells you which fix to reach for. High latency + idle CPU = something is blocking the loop on I/O; offload it. High latency + pinned CPU = you're genuinely out of compute; offload the heavy CPU work to other cores (or scale out). I've debugged the "service hangs but the box is bored" shape enough times that it's now the first thing I check: if the dashboards show a frozen service drawing 3% CPU, I stop looking for a crash and start looking for a blocking call on the loop. The metric you already have — CPU utilization next to p99 latency — is a free blocking-loop detector if you know to read the two together.

```python
import threading, time, faulthandler, sys

# A tiny watchdog: a separate thread that the loop must "touch" every 50 ms.
# If the loop is blocked, the timestamp goes stale and we dump a stack trace.
_last_tick = time.monotonic()

async def _heartbeat():
    global _last_tick
    while True:
        _last_tick = time.monotonic()
        await asyncio.sleep(0.05)        # if the loop is blocked, this never runs

def _watchdog():
    while True:
        time.sleep(0.1)
        if time.monotonic() - _last_tick > 0.5:   # loop hasn't ticked in 500 ms
            sys.stderr.write("EVENT LOOP STALLED — dumping all stacks\n")
            faulthandler.dump_traceback()          # shows WHICH frame is stuck
```

The watchdog runs on its own OS thread (so a blocked loop can't silence it), notices the loop hasn't ticked, and dumps every thread's stack — which points straight at the blocking frame. This is the "find the unknown offender in a dependency" tool: the linter can't see a blocking call hidden three libraries deep, but a stack dump taken *while* the loop is stuck names the exact line.

## The colorless approach: goroutines and virtual threads

Step back and ask the heretical question: why do we have two colors at all? The answer is that async/await chose **stackless** coroutines — each task is a compact state machine with no dedicated call stack — to make tasks cheap (kilobytes, not megabytes). The price of statelessness is that the suspension points must be visible in the type system, which is exactly the `async`/`await` coloring. But there is another way to get cheap concurrency, and it has no colors at all.

Go's **goroutines** and Java's **Project Loom virtual threads** are *stackful*, *user-mode* threads. Each one has a real (small, growable) stack, so it looks and acts exactly like an ordinary thread — but it's scheduled by the language runtime onto a small pool of OS threads, not by the OS directly. The magic: when a goroutine or virtual thread makes a *blocking* call, the runtime *intercepts* the block, parks that lightweight thread (saving its stack), and runs a different one on the underlying OS thread. The OS thread never actually blocks; only the cheap virtual thread does. From the programmer's side, **every function can block, and blocking is cheap, so there is no reason to mark functions async.** One color.

```go
// Go: every function is the same "color". This BLOCKS the goroutine,
// not the OS thread — the scheduler parks it and runs another goroutine.
func handle(id int) (User, error) {
    user, err := readUser(id)   // looks blocking, IS cheap; no await, no async keyword
    if err != nil {
        return User{}, err
    }
    return user, nil
}

func main() {
    for i := 0; i < 10000; i++ {
        go handle(i)            // 10k goroutines, each ~2-8 KB of stack
    }
}
```

```java
// Java 21 Loom: a virtual thread per task; blocking calls unmount the carrier
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    for (int i = 0; i < 10_000; i++) {
        int id = i;
        executor.submit(() -> {
            // syncDbQuery BLOCKS this virtual thread; the carrier OS thread
            // is released to run other virtual threads. No async, no await.
            var user = syncDbQuery(id);
            process(user);
        });
    }
}   // 10k virtual threads on a handful of carrier threads
```

Look at what's *absent*: no `async`, no `await`, no `spawn_blocking`, no `run_in_executor`. A blocking DB call inside a virtual thread is *fine* — the runtime turns it into a park-and-switch under the hood. That's the colorless promise, and it dissolves this entire post's problem: there is no async tax (functions don't change color), no bridge to install (sync *is* the model), and the cardinal sin is impossible (a blocking call parks one virtual thread, it doesn't seize a shared loop). The matrix lays the three models side by side.

The mechanism deserves a closer look, because it's the same trick async/await uses, just hidden. When a goroutine calls a blocking operation — say a socket read — the Go runtime doesn't actually issue a blocking syscall on the OS thread. It registers the socket with the runtime's *network poller* (epoll on Linux, kqueue on BSD/macOS, IOCP on Windows — the very same OS readiness machinery the async event loop uses), saves the goroutine's stack and registers, and **deschedules** it. The OS thread (Go calls it an `M`, for machine) is now free to pick up a different runnable goroutine from the scheduler's run queue. When the poller later reports the socket is readable, the runtime marks the goroutine runnable again, and some `M` resumes it — restoring its saved stack — exactly where it left off. From inside the goroutine, the blocking read simply "took a while" and returned; it never knew it was descheduled. Java's Loom does the analogous thing: a blocking call on a virtual thread *unmounts* the virtual thread from its carrier platform thread (saving its continuation on the heap), frees the carrier to run another virtual thread, and remounts when the I/O is ready. Both are doing async I/O under the covers and presenting it as blocking. The profound observation is that **the async event loop and the goroutine scheduler are the same machine** — non-blocking OS readiness plus a scheduler — differing only in *where the suspension shows up*: in async/await it's a visible `await` in your source and in the function's color; in goroutines/Loom it's invisible, synthesized by the runtime. Coloring is the cost of making suspension explicit; colorless is the cost of making it implicit.

![A matrix comparing async await runtimes goroutines and Loom virtual threads on whether they color functions whether code can block and the per-task cost](/imgs/blogs/function-coloring-and-bridging-sync-and-async-7.png)

So why doesn't everyone use virtual threads and forget async? Three honest caveats. **First**, it requires deep runtime support — the runtime must be able to intercept *every* blocking operation and remap it. Go and the JVM control their I/O all the way down, so they can. A language without that control (or with FFI calls into C that block the real OS thread) can't transparently park on those calls — Loom calls these "pinning" cases, where a virtual thread holding a native lock or inside a `synchronized` block can't unmount and *does* tie up a carrier thread. **Second**, stackful means each task carries a stack, so the per-task floor is higher than a stackless state machine — goroutines start around 2–8 KB and grow, virtual-thread frames live on the heap; an async task can be a few hundred bytes. At truly extreme task counts the stackless model wins on memory. **Third**, async/await makes suspension points *explicit*, and some people value that — you can *see* in the source exactly where a function can yield, which matters for reasoning about what state can change across an `await`. Virtual threads hide that, which is the whole point but also removes a signal. There's no free lunch; there's a different lunch with a different bill.

## The cost of the bridge: pool sizing and context handoff

When you *do* live with coloring and must bridge, the bridge is not free, and underestimating its cost is its own class of outage. Two costs dominate.

**Thread-pool sizing.** The offload pool's size is a real capacity decision. Too small and your blocking calls queue behind each other — if you have a pool of 8 and 50 requests each need a 200 ms blocking DB call, requests 9 through 50 wait in line, and tail latency balloons. Too large and you pay for thousands of mostly-parked OS threads (each with a stack, each a scheduling entity) and you can overwhelm the *downstream* — 500 simultaneous connections to a database that allows 100 will get you connection errors, not speed. The right size is roughly governed by Little's law: if you need throughput $\lambda$ requests/second and each holds a pool thread for $W$ seconds, you need about $L = \lambda W$ threads in flight, capped by what downstream can take. For 200 requests/second at 50 ms each, $L = 200 \times 0.05 = 10$ threads — assuming downstream tolerates 10 concurrent calls. Size the pool to the *bottleneck resource*, usually the database's connection limit, not to your request rate.

**Context handoff.** Every offload moves work to another thread and the result back, which costs a cache-cold thread wakeup, a scheduling decision, and the loss of thread-local context. That last one is sneaky: thread-locals (request IDs, trace spans, security context, database transaction handles) do *not* automatically follow work to a pool thread. A trace span set on the loop thread is invisible on the pool worker unless you explicitly propagate it; a transaction bound to the loop thread is not the transaction the pool thread sees. Many a "my logs lost the request ID after an `await`" or "the offloaded query ran outside the transaction" bug traces to exactly this. The fix is to capture the needed context *before* offloading and pass it explicitly, or use a context-propagating wrapper your framework provides.

```rust
// Capture context BEFORE crossing to the pool thread; pass it in explicitly.
async fn fetch_with_trace(id: u64, trace_id: String) -> Vec<Row> {
    tokio::task::spawn_blocking(move || {
        // trace_id was MOVED into the closure; it now lives on the pool thread.
        // A thread-local trace span from the loop thread would NOT be here.
        log_with_trace(&trace_id, "running blocking query");
        sync_db_query(id)
    })
    .await
    .expect("blocking task panicked")
}
```

There's also a subtler cost: the offloaded blocking call can't be *cancelled* the way an async task can. If the client disconnects and you cancel the async task, the `await` on the `spawn_blocking` handle stops waiting — but the blocking call *keeps running* on the pool thread until it finishes naturally, because you can't safely interrupt a thread mid-syscall. So cancellation on the async side does not free the pool thread promptly; under a cancellation storm you can still exhaust the pool with work nobody's waiting for anymore. This is one of the threads (pun intended) we pick up in [structured concurrency, cancellation, and thread-pool design](/blog/software-development/concurrency/structured-concurrency-cancellation-and-thread-pool-design).

#### Worked example: the trace that vanished across the bridge

A team ran distributed tracing where each request set a trace-id in a thread-local at the start of the handler, and every log line read that thread-local to stamp the id. It worked perfectly — until they offloaded their slow image-resize step to a thread pool. The resize logs suddenly had *no* trace-id: blank, unsearchable. The cause was exactly the context-handoff cost. The trace-id lived in a thread-local on the *loop* thread; the resize ran on a *pool* thread, which had its own (empty) thread-local. The work crossed the bridge but the context did not. They tried "fixing" it by setting the thread-local inside the pool task, which half-worked but then *leaked*: pool threads are reused, so a trace-id set by one request and never cleared got picked up by the *next* unrelated request that happened to land on the same pool thread — now logs were stamped with the *wrong* request's id, which is worse than blank. The correct fix is to capture the id as a *value* before offloading and pass it explicitly into the task (as in the Rust snippet above), never relying on thread-locals to survive a thread hop. The general rule: **anything bound to the current thread — thread-locals, the current transaction, the security principal, the MDC in Java logging — does not cross to a pool thread for free.** Treat the bridge as a process boundary for context: pass what you need by value, assume nothing carries over.

A related gotcha is panics and exceptions. A blocking task that panics on a pool thread does not, by default, take down your process the way a panic on the main thread might — it's captured in the `JoinHandle` (Rust), surfaces as an exception when you `await` the future (Python), or completes the `CompletableFuture` exceptionally (Java). That's good — but it means **you must actually check the result**, or a failed offload becomes a silently-swallowed error. The `.expect("blocking task panicked")` in the snippets is not decoration; it's the difference between a loud failure and a request that mysteriously returns nothing. Always propagate the offloaded task's error back to the awaiting code; don't let the bridge eat it.

## Case studies / real-world

**Bob Nystrom's "What Color Is Your Function?" (2015).** The essay that named the problem. Nystrom's core complaint is the rule-2 asymmetry — that red infects blue and you can't escape it — and his sharpest observation is that this isn't unique to async: blocking-vs-non-blocking, sync-vs-async, throwing-vs-non-throwing (in some languages), `const` correctness, all create "colors." He uses it to argue *for* languages where there's effectively one color — which is exactly the goroutine/virtual-thread pitch. The essay is required reading; it's why the whole industry now says "function coloring" instead of vaguely gesturing at "the async thing."

**Blocked event loops in Node and Python, in the wild.** The pattern in the intro is one of the most common production incidents on async runtimes, and it has a recognizable signature: high latency or full hangs with *low* CPU usage, because the one loop thread is parked in a blocking syscall while requests pile up. Public postmortems and the Node.js docs themselves ("Don't Block the Event Loop") repeatedly call out synchronous file I/O (`fs.readFileSync`), synchronous crypto (`crypto.pbkdf2Sync` with a high iteration count), heavy `JSON.parse` of huge payloads, and regex catastrophic backtracking (a "ReDoS") as the usual culprits — all CPU or blocking work done inline on the loop. The fix in every case is the same family as ours: move the work off the loop (a worker thread, a child process, or the async variant of the API). The reason this bug is so common is precisely the silent-failure asymmetry from the matrix: red-calling-blue compiles, tests pass, and only production timing reveals it.

**Project Loom's pitch and its pinning caveat.** When the JVM team shipped virtual threads (JEP 444, finalized in Java 21), their explicit framing was: keep the simple, debuggable, *colorless* thread-per-request style that Java engineers already know, but make threads cheap enough to have millions of them — eliminating the need to rewrite into async/reactive `CompletableFuture` chains just to scale. The pitch is a direct answer to the coloring tax: write ordinary blocking code, let the runtime make it scale. The honest footnote they shipped alongside it is *pinning*: a virtual thread that blocks while inside a `synchronized` block or while holding certain native resources cannot unmount its carrier OS thread, so it temporarily behaves like a platform thread and can starve the carrier pool. The guidance is to migrate hot `synchronized` sections to `ReentrantLock` (which is Loom-aware). It's a real asterisk on "colorless," and a reminder that even the colorless model has a boundary where the abstraction leaks.

**The "sans-IO" library pattern as a response to coloring.** A subtler, very practical industry response to coloring is to write protocol logic that does *no* I/O at all — neither sync nor async — and is therefore *colorless by construction*. A "sans-IO" parser or state machine takes bytes in and produces bytes out plus events; the actual reading and writing of the socket is left to the caller, who can be sync or async as they please. The Python `h11`/`h2` HTTP libraries and the Rust `quinn`/`rustls` ecosystems lean on this idea: keep the hard logic (HTTP framing, TLS state, congestion control) in pure functions that touch no socket, so the *same* core works under a sync server, an async server, or a test harness with no real socket at all. It's the same generalized fix from the top of the post — *don't let the colored property into the core; keep it at the edge* — applied at library-design scale. If you maintain a protocol library and you're agonizing over "should this be sync or async," the best answer is often "neither": make the I/O someone else's problem and your library inherits no color. That's the most durable way to dodge the whole boundary.

## When to reach for this (and when not to)

Decisive guidance, because every choice here is a cost.

**Reach for `block_on` / `asyncio.run` (sync→async bridge) when** you are at the genuine *bottom* of a sync stack and need a few async results: `main`, an integration test, a CLI entry point, a sync worker thread that owns no event loop. **Do not** call it from inside async code — that's the deadlock. If you're already async, `await`; you never need a bridge to go red→red.

**Reach for `spawn_blocking` / `run_in_executor` / a bounded executor (async→sync bridge) when** an async task must call something that *blocks* (a sync DB driver, a legacy library, blocking file I/O) or runs longer than roughly 100 microseconds of pure CPU. Size the pool to the downstream bottleneck (Little's law, capped by the connection limit), keep it bounded for backpressure, and propagate context explicitly. **Do not** offload trivially fast pure-CPU work — the handoff costs more than the work, and you'll starve the pool for the calls that actually need it. **Do not** assume cancellation frees the pool thread; it doesn't until the blocking call returns.

**Reach for async/await as your concurrency model when** the workload is I/O-bound with very high concurrency (tens of thousands of mostly-idle connections), you want explicit suspension points for reasoning, and your ecosystem has good async drivers for your dependencies. The coloring tax is worth paying when the C10k win is real and the dependencies cooperate.

**Reach for virtual threads / goroutines instead when** you want thread-per-request simplicity at scale and you're on a runtime that supports them (Go, or the JVM with Loom), *especially* if your dependency ecosystem is still synchronous (no good async drivers) — virtual threads let you keep blocking code and still scale, with no bridge and no tax. Watch for pinning (Loom + `synchronized` / native calls). The one place the stackless async model still wins is the absolute extreme of task count where per-task memory dominates.

**Avoid the boundary entirely when you can.** The cheapest bridge is the one you don't build. Push the async to the edge behind a sync-looking cache or queue (the `get_setting` story), or pick a single color for a whole subsystem so nothing has to cross. Most boundary bugs come from accidental crossings, not deliberate ones.

The decision tree below collapses the bridge choice to a single question — which direction are you crossing.

![A tree showing the choice of bridge primitive by crossing direction with block_on and run for sync to async and spawn_blocking and run_in_executor for async to blocking](/imgs/blogs/function-coloring-and-bridging-sync-and-async-8.png)

## The worked fix: from frozen server to spawn_blocking

Now assemble the intro outage and fix it end to end. The broken handler, in two languages, with the bug called out:

```python
async def activity_panel(request):
    # BUG: a synchronous DB driver called directly inside an async handler.
    # sync_db.query BLOCKS the event-loop thread for the full query duration.
    user_id = request.user_id
    # This is red-calling-blue, and the blue call BLOCKS. On a stall,
    # the loop thread is held for ~800 ms and ALL other tasks freeze.
    rows = sync_db.query(
        "SELECT * FROM activity WHERE user=%s LIMIT 20", user_id
    )
    return render(rows)
```

```rust
// BUG (Rust/tokio equivalent): blocking driver inside an async fn.
async fn activity_panel(req: Request) -> Response {
    let user_id = req.user_id;
    // sync_db_query blocks THIS tokio worker thread. With a small worker
    // pool, a few of these in flight stall a large fraction of the server.
    let rows = sync_db_query(user_id);   // blocks the loop worker
    render(rows)
}
```

The fix is the async→sync bridge: offload the blocking query to a bounded pool, `await` the result, leave the loop free. The handler stays async; only the *blocking line* moves.

```python
async def activity_panel(request):
    # FIX: offload the blocking query to the executor; the loop stays free.
    user_id = request.user_id
    loop = asyncio.get_running_loop()
    # run_in_executor moves sync_db.query to a bounded ThreadPoolExecutor.
    # We await it, so this coroutine YIELDS the loop while the pool thread
    # blocks. The other ~10k tasks keep running.
    rows = await loop.run_in_executor(
        DB_POOL,                          # a bounded ThreadPoolExecutor(max_workers=16)
        functools.partial(
            sync_db.query,
            "SELECT * FROM activity WHERE user=%s LIMIT 20",
            user_id,
        ),
    )
    return render(rows)
```

```rust
// FIX (Rust/tokio): spawn_blocking moves the blocking call off the loop.
async fn activity_panel(req: Request) -> Response {
    let user_id = req.user_id;
    // The blocking query runs on tokio's blocking pool; this fn yields the
    // loop worker while awaiting the JoinHandle. Loop stays responsive.
    let rows = tokio::task::spawn_blocking(move || {
        sync_db_query(user_id)            // allowed to block: it's a pool thread
    })
    .await
    .expect("blocking task panicked");
    render(rows)
}
```

The bound (`max_workers=16`, tokio's blocking pool cap) is the safety valve. When the reporting DB stalls again, the first 16 offloaded queries occupy pool threads and *block there* — harmlessly, off the loop — and the 17th and beyond queue. Tail latency for activity-panel requests climbs (some users wait in the offload queue), but the loop thread keeps polling, the other ten thousand connections keep getting served, health checks pass, and the load balancer doesn't eject the instance. The failure mode changed from "total outage" to "one slow endpoint." That is the entire point of the bridge: not to make the slow thing fast, but to *contain* its slowness so it can't take the whole process down.

#### Worked example: the one-line detector that would have caught it

Before the fix shipped, the team turned on the loop-lag warning in staging — `loop.slow_callback_duration = 0.1` and `debug=True`. The next time the staging reporting DB hiccuped, the logs printed `Executing <Task activity_panel> took 0.640 seconds`, naming the exact coroutine. That single log line is the difference between "the service mysteriously hangs sometimes" and "the `activity_panel` task blocks the loop on a slow DB — offload it." Cheap detection beats clever debugging.

## Measured: the cost of a blocked loop versus offloaded

Numbers make this real. Here is the shape of what a small, honest benchmark shows — a server handling background traffic at a steady rate while one endpoint runs a query that takes 800 ms when the DB stalls. Measure p50/p99 latency of the *background* traffic during a stall, with the slow query inline (blocking the loop) versus offloaded to a bounded pool. These are illustrative numbers from this class of experiment; your platform, runtime, and load will move them, so treat them as orders of magnitude, not gospel — and measure your own.

| Condition | Background p50 | Background p99 | Throughput during stall | Outcome |
| --- | --- | --- | --- | --- |
| Blocking call inline on the loop | ~800 ms | ~800+ ms | collapses to ~0 | every task frozen; health checks fail |
| Offloaded to bounded pool (16) | ~8 ms | ~15 ms | unchanged | only the slow endpoint queues |
| Offloaded, pool too small (2) | ~8 ms | ~400 ms | slightly reduced | slow endpoint serializes; partial backup |
| Offloaded to unbounded pool | ~8 ms | ~15 ms (until OOM) | unchanged then crash | thread explosion under load spike |

The first row is the catastrophe: during the 800 ms stall, *background* p99 is also ~800 ms, because the background tasks were sitting behind the blocking call in the loop's queue the entire time — their latency is dominated by something they have nothing to do with. The second row is the fix working: background latency is untouched (single-digit milliseconds) because the loop never blocked. The third and fourth rows are the two ways to size the pool wrong — too small serializes the slow endpoint (its p99 climbs while background stays fine), too big trades the freeze for a thread explosion under a spike. The sweet spot is a bound matched to the downstream limit.

To measure this honestly: warm up the JIT/connection pools first, run for minutes not seconds (stalls are intermittent — a 5-second run might miss them entirely), inject the stall deterministically (a query against a deliberately locked row, or a sleep in a test driver) rather than hoping the real DB hiccups during your window, and report a distribution (p50/p99/max), never a single mean — the *tail* is the story. A mean latency can look fine while the p99 is a horror, because the blocking only hits the unlucky requests caught during a stall. If you only watch averages, a blocked-loop problem hides until it pages you.

| Operation | Rough cost | Implication |
| --- | --- | --- |
| Inline fast CPU call (2 µs hash) | ~2 µs | leave it on the loop; never offload |
| Thread handoff to a pool (offload + return) | ~10–50 µs | offload only work that dwarfs this |
| Blocking I/O call (DB query) | ~1–800 ms | MUST offload; would freeze the loop |
| Goroutine / virtual-thread park-and-switch | ~hundreds of ns | colorless model: blocking is this cheap |

The second table is the decision in numbers: an offload costs ~10–50 µs, so offloading a 2 µs call is a 5–25x tax for nothing, while offloading a 100 ms blocking call is the only sane choice. And the last row is the colorless model's quiet flex — parking a goroutine costs hundreds of *nanoseconds*, three to five orders of magnitude cheaper than a thread handoff, which is exactly why Go and Loom can let everything "block" without a pool: the "offload" is essentially free and built into every blocking call.

One more honest note on what these numbers do *not* tell you. The tables above measure the *contention* and *handoff* effects, but the real-world impact of a blocked loop also depends on your *traffic shape*. A service that gets one slow request per second among ten thousand fast ones suffers far more from a blocked loop than the raw stall duration suggests, because that one 800 ms block lands on top of a deep queue — the ten thousand fast requests that arrived *during* the stall all inherit the full 800 ms wait, so a single slow request can blow up the p99 of *everything*. This is the "convoy effect," and it's why the failure is so disproportionate: the cost isn't one slow request, it's the entire backlog that piles up behind it. The offloaded version has no convoy because the loop never stops draining the queue — slow work waits *off to the side* in the pool, not *in front of* the fast work. When you reason about whether a blocking call is "worth offloading," don't just compare its duration to your latency budget; consider how many other requests will queue behind it during a stall. A blocking call that's rare but lands during peak traffic is the most dangerous kind, because that's exactly when the queue behind it is deepest.

It's also worth stating what you can't fix with offloading alone: if your *aggregate* blocking work genuinely exceeds your pool's capacity for a sustained period — not a transient stall but steady overload — no pool size saves you, because the queue grows without bound and latency climbs forever. At that point the honest answers are upstream: shed load (reject some requests fast rather than queue them all), add capacity, or fix the downstream that's slow. Offloading contains a *transient* slowness and protects unrelated traffic from it; it does not manufacture throughput you don't have. Keep that boundary clear in your head, because "we added a thread pool and it's still slow" usually means the problem was never the loop — it was that you're asking the system to do more blocking work per second than it can sustain, and that's a capacity problem, not a coloring one.

## Key takeaways

- **Coloring is real and asymmetric.** Async (red) can only be `await`ed from async; sync (blue) can't drive a future without a bridge. The blue→red mistake fails loudly (a coroutine where you wanted a value, or a compile error); the red→blue mistake — a blocking call in async code — fails *silently* and only under production timing. That asymmetry is why blocked-loop bugs are so common.
- **Async infects upward.** Making one leaf async forces every caller above it to become async — the async tax. It only flows up, never down, because `await` is only legal inside async, and there is no way for a sync frame to drive a future. Often the cheapest fix is to keep the boundary at the edge so the color never propagates.
- **Bridge sync→async only at the bottom of a sync stack.** `block_on` / `asyncio.run` belong in `main`, tests, and loop-less worker threads. Calling them from inside async code self-deadlocks (or, if you're lucky, panics with "runtime within a runtime").
- **Bridge async→sync by offloading to a bounded pool.** `spawn_blocking` / `run_in_executor` / a fixed `ExecutorService` move the blocking call off the loop thread so only its result returns. Bound the pool (backpressure), size it to the downstream limit ($L = \lambda W$), and don't offload trivially fast work — the handoff costs more than it saves.
- **Never block the event loop.** One synchronous call holds the single loop thread and freezes every task — cooperative scheduling has no preemption to save you. The signature is high latency with idle CPU.
- **Detect it in layers.** Blocking-call linters in CI catch the obvious offenders; a loop-lag / slow-callback warning catches the surprise blocker (`loop.slow_callback_duration = 0.1` is one line); a watchdog catches the one hidden in a dependency.
- **Virtual threads and goroutines make coloring vanish.** Stackful, runtime-scheduled threads let every function block cheaply, so there's no async keyword, no tax, no bridge, and the cardinal sin is impossible. The cost is deeper runtime support (and pinning caveats), a higher per-task memory floor, and implicit suspension points.
- **Measure the tail, not the mean.** A blocked-loop problem hides in averages and lives in p99. Warm up, run for minutes, inject the stall deterministically, and report a distribution.

## Further reading

- Bob Nystrom, *"What Color Is Your Function?"* (2015) — the essay that named function coloring and argued for colorless languages.
- The Node.js guide *"Don't Block the Event Loop (or the Worker Pool)"* — the canonical list of loop-blocking culprits and offloading fixes, with the same mechanism in JavaScript.
- The tokio documentation on `spawn_blocking`, `block_in_place`, and the blocking-thread pool — the Rust async→sync bridge in detail, including why the blocking pool is separate and large.
- JEP 444, *Virtual Threads* (Java 21), and the Loom team's writing on the thread-per-request model and pinning — the colorless answer and its honest caveats.
- Brian Goetz et al., *Java Concurrency in Practice* — for executor sizing, the rationale behind bounded pools, and Little's-law-style capacity reasoning.
- [Async/await and how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) — the mechanism under the red functions in this post.
- [The event loop and the reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) — why a single blocking call freezes everything.
- [Why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) — the series intro and the shared-state frame.
- [The concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) — the capstone that places async, virtual threads, and bridges in the full decision space.
- [Async in practice: patterns, pitfalls, and blocking code](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code) — the Python-specific deep dive on `run_in_executor`, `to_thread`, and blocking-call detection.

---
title: "Structured Concurrency, Cancellation, and Thread-Pool Design"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to give concurrent work a lifecycle: scope every task, cancel it cleanly, and size the pool that runs it so overload becomes backpressure instead of an out-of-memory crash."
tags:
  [
    "concurrency",
    "parallelism",
    "structured-concurrency",
    "cancellation",
    "thread-pool",
    "backpressure",
    "executors",
    "async",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-1.png"
---

There is a bug I have shipped more than once, and I suspect you have too. It looks completely innocent. A request comes in, and somewhere in the handler you start a little background job — fire off an email, warm a cache, prefetch a related record — and you do it with the most natural line in the language: `go doThing()`, `tokio::spawn(do_thing())`, `executor.submit(doThing)`, `asyncio.create_task(do_thing())`. The handler returns. The response goes out. Everyone is happy.

Then one day the background job throws. And nothing happens. No log line, no alert, no stack trace in your error tracker. The exception was raised inside a task that no one is waiting on, so it had nowhere to go — it vanished into the runtime. Weeks later you discover that the email was never sent, the cache was never warmed, and you have no record of a single failure. Worse: under load, those orphaned tasks pile up faster than they complete, your heap fills with their stack frames and captured closures, and the process that was "just sending some emails in the background" gets killed by the OOM killer at 3 AM. The task had no owner. Its error had nowhere to go. Its lifetime was unbounded. That single unscoped spawn quietly broke three things at once: error handling, resource accounting, and cancellation.

This post is about taming the *lifecycle* of concurrent work — the part of concurrency that is not about correctness of shared state (we covered that with locks, atomics, and the memory model elsewhere in this series) but about *when tasks start, when they must finish, who is responsible for their errors, and how you stop them.* The fix has a name — **structured concurrency** — and a slogan that is deliberately provocative: the unscoped spawn is the `goto` of concurrent programming. We will build up the structured-concurrency model (every task lives inside a *scope* that does not return until all its children finish), then the two flavors of **cancellation** (cooperative, which is safe, and forceful, which is a loaded gun), then the machinery that actually runs all this work: **thread pools and executors** — how to size them, why an unbounded queue is a trap that hides overload until it kills you, and how a bounded queue turns overload into honest **backpressure**. We will end with a worked, cancellable, bounded worker pool with a deadline, in more than one language, and we will measure what bounded-vs-unbounded actually does to latency and throughput under overload.

![A side by side comparison showing an unscoped spawn that leaks a task and swallows its error versus a structured scope that joins every child and re-raises failures](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-1.png)

If you want the foundation under all of this — why concurrency is unavoidable and why it is hard — start with [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it). This post sits at the end of the arc: once you can name shared state and establish a happens-before order, the remaining question is *who owns the running task, and what is its lifetime.* That is the structured-concurrency question.

## The spawn that escaped: an anatomy of the leak

Let me make the leak concrete, because the abstract version ("the task has no owner") does not land until you have watched it happen. Here is a Go handler that looks fine in review and is wrong in three independent ways.

```go
// BUG: the goroutine outlives the handler, its error is lost,
// and nobody can cancel it.
func handleOrder(w http.ResponseWriter, r *http.Request) {
    order := parseOrder(r)

    // Fire-and-forget: warm the recommendation cache.
    go func() {
        recs, err := buildRecommendations(order.UserID) // may take 5s, may fail
        if err != nil {
            // Where does this go? Nowhere useful.
            return
        }
        cache.Put(order.UserID, recs)
    }()

    saveOrder(order)
    writeJSON(w, http.StatusOK, order) // returns immediately
}
```

Walk the failure modes one at a time. First, **the lifetime is unbounded**. `handleOrder` returns the moment `writeJSON` runs. The goroutine keeps running, holding `order.UserID` and whatever `buildRecommendations` captured, for as long as it takes — and if `buildRecommendations` hangs on a slow downstream call with no timeout, that goroutine lives forever. Multiply by your request rate and you have a goroutine leak: a slow, monotonic climb in memory and scheduler pressure that a dashboard will eventually catch as "we restart every 18 hours and nobody knows why."

Second, **the error is swallowed**. The `if err != nil { return }` is the most honest part of the bug, because it admits there is nowhere to send the error. The goroutine is not connected to any caller. There is no stack to unwind into, no parent to receive the failure, no future to mark as rejected. A panic in that goroutine is even worse — in Go, an unrecovered panic in any goroutine crashes the *entire process*, so your fire-and-forget cache warmer can take down the server.

Third, **it cannot be cancelled**. Suppose the client disconnects, or the request's overall deadline expires, or you are shutting down the server for a deploy. The handler is gone; there is no handle to that goroutine, no channel to signal it, nothing to wait on. It will run to completion (or forever) regardless. You have lost the ability to say "stop, we don't need this anymore."

These three failures are not three bugs. They are *one* bug — the absence of a *scope* — wearing three costumes. The spawn escaped the lexical structure of the program. And the moment a running task is not bracketed by a region of code that owns it, you lose error propagation, resource accounting, and cancellation simultaneously, because all three of those mechanisms are implemented by *the region that owns the task*.

## Structured concurrency: the scope that joins all children

The fix is to make concurrency follow the same shape as ordinary control flow. In sequential code, a function call has a beginning and an end; control does not leave the function until everything it called has returned. Structured programming gave us this with blocks: a `for` loop, an `if`, a function body — each is a region with one entrance and one exit, and you reason about it as a unit. **Structured concurrency applies the same discipline to concurrent tasks: every task runs inside a *scope* (Trio calls it a *nursery*), and the scope does not exit until every task started inside it has finished.**

That single rule — *the scope joins all its children before it returns* — is the whole idea, and it fixes all three failures at once:

- **Lifetime is bounded** by the scope's lexical extent. When you leave the `with` block / the `coroutineScope { }` / the `try (var scope = ...)`, every child is guaranteed done. No task can outlive the block that started it. A leak is now a *type error in your head*: you cannot accidentally let a task escape, because there is no syntax for it.
- **Errors propagate** out of the scope like an exception out of a function. If a child fails, the scope re-raises that failure to *its* caller, after first cancelling the siblings. The error has somewhere to go: up.
- **Cancellation has a target.** The scope is a handle to the whole subtree. Cancel the scope and the cancel signal flows to every child. A deadline on the scope bounds the entire subtree.

Here is the Go bug rewritten with a scope. Go does not have a first-class nursery in the standard library, so the idiomatic version is `errgroup` (from `golang.org/x/sync`), which is the community's structured-concurrency primitive: it owns a set of goroutines, waits for all of them, and returns the first error.

```go
// FIX: errgroup is a scope. Wait() joins all children
// and returns the first error; ctx cancels the rest.
func handleOrder(ctx context.Context, order Order) error {
    g, ctx := errgroup.WithContext(ctx)

    g.Go(func() error {
        recs, err := buildRecommendations(ctx, order.UserID)
        if err != nil {
            return err // propagates to Wait(), cancels siblings
        }
        cache.Put(order.UserID, recs)
        return nil
    })

    g.Go(func() error {
        return saveOrder(ctx, order)
    })

    // This line is the scope's join: it blocks until both
    // goroutines finish, and returns the first non-nil error.
    return g.Wait()
}
```

Now the function does not return until both children are done. If `buildRecommendations` fails, `g.Wait()` returns that error *and* the derived `ctx` is cancelled, so `saveOrder` is told to stop. The lifetime, the error, and the cancellation are all bounded by the function body, exactly as for a sequential call. The unscoped spawn was a `goto`; this is a block.

Python's Trio makes the shape even more visceral, because the nursery is a literal `with` block and the join is the de-indent:

```python
import trio

async def handle_order(order):
    async with trio.open_nursery() as nursery:
        nursery.start_soon(build_recommendations, order.user_id)
        nursery.start_soon(save_order, order)
        # control does NOT leave the `with` block until BOTH
        # tasks have finished. If either raises, the other is
        # cancelled and the exception propagates out of `async with`.
    # <-- here, every child is guaranteed done
    return order
```

Read that again: the closing of the `async with` block *is* the join point. You physically cannot fall out of the block while a child is still running. Nathaniel Smith, who coined "structured concurrency" in his 2018 essay, calls the nursery the place where "background tasks" become *foreground* tasks with a clear parent — and the de-indent is the moment the parent collects them all.

## The call tree, restored

The deep payoff of the scope rule is that it restores a *tree* over your concurrent tasks, matching the call tree you already reason about in sequential code. A scope can start tasks; those tasks can themselves open scopes and start grandchild tasks; and the invariant composes — no scope returns until its entire subtree has finished.

![A task scope tree where a parent scope owns child fetch and parse tasks that each own leaf grandchild tasks, all awaited at scope exit](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-2.png)

This tree is what makes the model *compose*. Conceptually, the scope is the node, and the tasks are its children; when you nest scopes you nest the tree, and the same join rule holds at every level. Because the structure is a tree and not an arbitrary graph of escaped tasks, you get properties for free that are nightmares to enforce otherwise:

- **No orphans.** Every task has exactly one parent scope. There is no task whose owner has already returned.
- **Errors have a unique path up.** A failure in a grandchild propagates to its parent scope, which (after cancelling siblings) propagates to *its* parent, exactly like exceptions unwinding a call stack. There is one route up, and it is the tree.
- **Cancellation has a unique path down.** Cancel any node and the signal fans out to its whole subtree. You can cancel a single sub-task, or the whole tree, and the semantics are obvious because the structure is.
- **Resource accounting is local.** The number of live tasks under a scope is bounded by what that scope started. You can reason about "at most N tasks here" at the scope, not globally.

Contrast this with the unscoped world, where tasks form a flat soup with no parent pointers: an error in one has no path to anyone, cancelling "everything" means tracking a global registry by hand, and "are we done?" is unanswerable. The tree is not a nicety. It is the data structure that makes lifetime, errors, and cancellation tractable.

#### Worked example: a fan-out search with a guaranteed join

Say you query three shards in parallel and want the union of their results, but you must not return until all three have either answered or definitively failed — no shard query may outlive the function. With a nursery this is six lines and the invariant is enforced by indentation:

```python
async def search_all(query):
    results = []
    async with trio.open_nursery() as nursery:
        for shard in ("us", "eu", "ap"):
            nursery.start_soon(query_one, shard, query, results.append)
    # all three queries are DONE here — no straggler can leak past this line
    return merge(results)
```

If the `ap` shard query raises, Trio cancels the `us` and `eu` queries (they get a cancellation at their next checkpoint), waits for them to unwind, and then re-raises the `ap` failure out of `search_all`. You cannot get a partially-returned result with a query still running in the background, because that state is unrepresentable: the de-indent is a hard join. The sequential rule you already rely on — "a call returns only when its work is done" — now holds for concurrent work too.

There is a memory-model payoff hiding in that join, and it connects this post to the rest of the series. The scope's join is a **happens-before edge**: everything a child task did before it finished *happens-before* the code that runs after the scope returns. Concretely, when `search_all` reads `results` after the `async with` block, it is guaranteed to see every `append` the three queries performed — there is no data race on `results`, no torn read, no need for a separate lock around the merge, *because the join establishes the ordering for you*. This is the same guarantee a thread `join()` gives in Java or a `WaitGroup.Wait()` gives in Go: the act of waiting for a task to finish synchronizes its writes into the waiter. The structured scope is therefore not just a lifetime tool; it is also a synchronization tool, and the discipline that the rest of this series builds — name what is shared, establish a happens-before order over every access — is *handed to you for free* by the scope boundary whenever children write results the parent reads after the join. Lose the scope (the unscoped spawn) and you lose the join, and with it the happens-before edge, and now reading those results from the parent is a genuine data race.

## "Go statement considered harmful": the unscoped spawn is a goto

In 1968 Dijkstra wrote "Go To Statement Considered Harmful," and the argument was not that `goto` produced wrong answers — it was that `goto` destroyed the correspondence between the static text of a program and its dynamic execution. With unrestricted `goto`, you could not look at a line of code and know how control arrived there or where it would go next, because a jump could come from anywhere and lead anywhere. Structured programming (loops, conditionals, functions) replaced arbitrary jumps with nested blocks that have one entrance and one exit, and suddenly you could reason about a region of code as a unit.

Smith's 2018 essay makes the precise analogy: **the unstructured `go`/`spawn`/`submit` statement is the `goto` of concurrency.** Here is why the analogy is exact, not cute.

A normal function call is a *black box* with respect to control flow: `result = f(x)` either returns a value or raises, and either way, when that line finishes, `f` is *done* — it is not still running somewhere. You can wrap it in a `try`/`finally`, time it, retry it, and know that nothing it started is still alive after it returns. That black-box property is what lets you compose functions without thinking about their internals.

An unscoped `go f()` *breaks* the black box. After the statement runs, `f` is still executing — control has "jumped" into a parallel timeline that the surrounding code has no handle on. The function that contained the `go` can return while `f` runs on; a `try`/`finally` around the `go` will run its `finally` while `f` is still going; an exception in `f` does not propagate to whoever called the function that spawned it. The static structure of the code (this function contains that spawn) no longer tells you the dynamic structure (when does `f` actually finish, and who hears about its failure). That is *exactly* the `goto` problem: the loss of correspondence between text and execution.

And just as `goto` could be *simulated* by structured constructs but not the reverse, structured concurrency can express anything `go` can — including "start this and don't wait here" — but it forces you to name *which scope* will eventually join the task. There is no truly ownerless task. The nursery you hand the task to might be a long-lived one higher up the tree (Trio lets you pass a nursery object into a function so a task can be started in an *ancestor's* scope), but it is *some* scope, with a defined lifetime, error sink, and cancellation handle. The slogan is strong on purpose: a bare spawn with no owning scope is a defect, the same way a bare `goto` into the middle of a loop is a defect. It will compile. It will usually work. And it will, eventually, leak a task, swallow an error, or refuse to cancel — at the worst possible time.

This is also why "just add a `WaitGroup`" is a partial fix in Go. A `sync.WaitGroup` gives you the *join* (you can `wg.Wait()` for the goroutines to finish), but by itself it does not give you error propagation or cancellation — you have to bolt those on with a shared error variable behind a mutex and a `context`. `errgroup` exists precisely because the raw `go` + `WaitGroup` combination keeps getting those two halves wrong. The lesson generalizes: if your language gives you a bare spawn, build or adopt a scope on top of it, and treat the bare spawn as a primitive you wrap, not an API you call directly in business logic.

## Errors: propagation, aggregation, and the first-failure cancel

Once tasks form a tree, error handling gets a precise definition, and it is worth stating carefully because the corner cases bite. When a child task in a scope raises:

1. **The scope cancels its other children.** The reasoning: if one branch of a fan-out has failed, the siblings' work is usually wasted (you were going to combine all three shard results; one failed, so the merge is moot). Cancelling them promptly frees resources and shortens the time to surface the error.
2. **The scope waits for the cancelled siblings to actually finish.** This is the subtle part. Cancellation is *cooperative* (next section), so a sibling does not stop instantly — it gets the cancel signal at its next checkpoint and runs its cleanup. The scope does not return until that cleanup is done. You never leave the scope with a sibling still unwinding.
3. **The scope re-raises.** What it re-raises depends on the runtime. The interesting design question is: what if *two* children fail?

That last question is where languages diverge. Go's `errgroup.Wait()` returns the *first* error and discards the rest — simple, but lossy. Trio and modern Python instead raise an **`ExceptionGroup`** (the `except*` syntax, standardized in Python 3.11) that carries *all* the failures, because in a fan-out the second and third failures may be just as diagnostic as the first. Java's `StructuredTaskScope` lets you choose a *joiner policy*: `ShutdownOnFailure` (cancel-and-propagate the first failure) or `ShutdownOnSuccess` (cancel the rest as soon as *any* succeeds — the classic "hedged request" / "first response wins" pattern). Kotlin's `coroutineScope` propagates the first child failure and cancels the rest, while a `supervisorScope` deliberately does *not* cancel siblings on one child's failure (for cases where children are independent).

The point that survives all the variation: **in a scope, an error cannot be silently dropped, and a failure in one task deterministically affects its siblings according to a policy you chose.** Compare to the unscoped world where a failure affects *nothing* and is seen by *no one*. Here is the Java shape, which makes the joiner policy explicit and is the model the JDK is standardizing (it was a preview API through JDK 21–24 and is targeting stabilization):

```java
// Java structured concurrency: the scope joins both subtasks,
// and ShutdownOnFailure cancels the survivor if either fails.
try (var scope = StructuredTaskScope.open(
        StructuredTaskScope.Joiner.<String>awaitAllSuccessfulOrThrow())) {

    Subtask<String> user  = scope.fork(() -> fetchUser(id));
    Subtask<String> order = scope.fork(() -> fetchOrder(id));

    scope.join();            // joins BOTH; throws if either failed

    return combine(user.get(), order.get());
}   // <-- scope is closed here; no forked task can outlive this block
```

The `try`-with-resources block is the scope. `scope.join()` is the barrier. If `fetchUser` throws, the joiner cancels `fetchOrder` (interrupts its carrier thread), waits for it to unwind, and `join()` rethrows the failure. The forked subtasks cannot outlive the `try` block — the close brace is the hard join, the same role the de-indent plays in Trio.

## Cancellation: cooperative versus forceful

Cancellation is "stop doing this work; we no longer need the result." It sounds simple and it is the source of an astonishing number of production incidents, because there are two fundamentally different ways to implement it and one of them is a trap.

![A comparison matrix of cooperative versus forceful cancellation across how it stops, safety, and when to use](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-4.png)

**Cooperative cancellation** is a *request* the task must check for and honor. Mechanically: there is a cancellation flag (or a "done" channel, or a deadline) associated with the task. The task periodically checks it — and crucially, it is *guaranteed* to check it at every *suspension point* (every `await`, every blocking channel operation, every `select`). When the flag is set, the runtime injects a cancellation at the next checkpoint: in Trio it raises `trio.Cancelled` out of the awaiting call; in Kotlin it raises `CancellationException` out of the suspend point; in Go you observe `<-ctx.Done()` returning. The task then unwinds *normally* — its `finally` blocks run, its locks release, its files close — because cancellation is just an exception traveling up through code that is between two safe checkpoints. **The key safety property: the task is never interrupted in the middle of a critical section. It only stops where it has already published a consistent state.**

**Forceful cancellation** stops the task *now*, wherever it is, by an external act: killing the thread, an asynchronous interrupt that can fire at any instruction, `Thread.stop()` (deprecated in Java for exactly this reason), `pthread_cancel` with asynchronous cancelability, or yanking the OS thread out from under the code. The problem is brutal and well documented: the thread might be *halfway through a mutation* when it dies. It could die holding a lock — which is now held forever, deadlocking everyone who needs it. It could die after incrementing a counter but before writing the matching log entry. It could die mid-way through updating a balance — the exact torn-state hazard the rest of this series is about. Java deprecated `Thread.stop()` with a note that is worth quoting in spirit: it can leave objects in an inconsistent state with no way to detect or repair the damage, because the thread had no chance to run its `finally` blocks.

The stranded-lock failure deserves a closer look because it is the worst of the lot and the least obvious. Suppose a thread acquires a mutex, begins updating a shared data structure, and is forcefully killed mid-update. The mutex is now held by a thread that no longer exists. Every other thread that tries to acquire it blocks forever — you have created a permanent deadlock out of a single forced kill, and worse, the data structure the dead thread was modifying is in a half-written state that the survivors will read as soon as someone *does* hold the lock (if the lock were poisoned and re-acquirable, as Rust's `Mutex` poisoning models). This is precisely why a forced kill almost always means "kill the process," not "kill the thread": only process death reliably releases the OS-level resources (locks, file handles, sockets) the thread was holding, because the OS reclaims them on exit. A thread killed inside a user-space critical section leaves the user-space lock stranded with no one to reclaim it. The asymmetry is fundamental: cooperative cancellation unwinds *through* your cleanup code, so locks are released and invariants restored; forceful cancellation skips your cleanup entirely, so whatever was half-done stays half-done forever.

So the rule is stark: **cooperative cancellation is the default and the only safe general mechanism; forceful cancellation is a last resort for a truly runaway task you cannot otherwise stop**, and even then you usually pay by killing the whole process and restarting, because that is the only way to be sure no lock is stranded and no state is torn.

The cost of cooperative cancellation is that it requires *cooperation*: a task that never reaches a checkpoint can never be cancelled. The canonical offender is a tight CPU loop or a *blocking* call with no cancellation support:

```python
async def crunch():
    # BUG: this task can never be cancelled cooperatively.
    total = 0
    for i in range(10_000_000_000):
        total += expensive(i)   # no await anywhere in this loop
    return total                # Trio can NEVER inject Cancelled here
```

There is no `await` in the loop body, so there is no checkpoint, so a `trio.Cancelled` has nowhere to be injected — the cancel request sits unhonored until the loop finishes (which is forever, for our purposes). The fix is to *add* checkpoints, cheaply:

```python
async def crunch():
    # FIX: yield to the scheduler periodically so cancellation can land.
    total = 0
    for i in range(10_000_000_000):
        total += expensive(i)
        if i % 10_000 == 0:
            await trio.lowlevel.checkpoint()  # a cancel can be injected here
    return total
```

The Go equivalent is to poll `ctx.Done()` in the loop, and the Java equivalent is to check `Thread.interrupted()` (and to make blocking calls interruptible). The general principle is universal across runtimes: **cooperative cancellation only works if your code has checkpoints, so long-running CPU work must voluntarily yield, and blocking calls must be cancellation-aware.** This is the one real tax structured concurrency imposes, and it is far cheaper than the alternative of torn state.

## Propagation, deadlines, and timeouts

Cancellation gets its real power when it *propagates* down a scope and when it is *triggered* by a deadline. These two combine into the pattern you will reach for constantly: "run this whole subtree of work, but give the entire thing at most 800 ms; if it blows the budget, cancel everything cleanly and surface a timeout."

![A timeline showing cancellation propagating from a scope deadline down to children which run cleanup before the scope returns a single error](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-3.png)

Trace the timeline. A deadline fires (or a sibling fails). The scope flips its cancel flag. Each child, at its next `await`, receives the cancellation. Each child runs its cleanup (`finally`, context managers, deferred closes) as the cancellation unwinds its stack. The scope waits for *all* children to finish unwinding. Only then does the scope return — surfacing a single timeout/cancel up to *its* parent, which may itself be inside a larger scope with its own deadline. Cancellation, in other words, is not a single event; it is a wave that propagates down the tree and a join that propagates back up.

**Deadlines compose, and they compose by taking the minimum.** This is a property worth internalizing. If an outer scope has 800 ms left and an inner scope asks for 2 s, the inner scope effectively gets 800 ms — the outer deadline dominates because the outer scope will cancel the inner one when its own budget expires. Trio implements this with *cancel scopes* that nest; Go implements it with `context.WithDeadline`/`WithTimeout` deriving a child context whose deadline is the min of the parent's and the requested one. This is why you should pass deadlines as *absolute times* (or derive them from a parent context) rather than re-computing "5 seconds from now" at each layer: the layers must agree on a single shrinking budget, not each restart the clock.

Here are the two idioms side by side. Go threads a `context` explicitly through every call — verbose, but the deadline is visible at every layer:

```go
// A deadline on the whole subtree; ctx.Done() is the checkpoint.
func fetchWithBudget(parent context.Context, id string) (Result, error) {
    ctx, cancel := context.WithTimeout(parent, 800*time.Millisecond)
    defer cancel() // ALWAYS cancel to release the timer + children

    g, ctx := errgroup.WithContext(ctx)
    var user User
    var ord  Order
    g.Go(func() error { var e error; user, e = fetchUser(ctx, id); return e })
    g.Go(func() error { var e error; ord,  e = fetchOrder(ctx, id); return e })

    if err := g.Wait(); err != nil {
        return Result{}, err // includes context.DeadlineExceeded on timeout
    }
    return Result{user, ord}, nil
}
```

Note the `defer cancel()` — in Go this is not optional. The `context` allocates a timer; if you never call `cancel`, the timer (and the goroutine tree it governs) leaks until the deadline fires. Forgetting `defer cancel()` is the single most common Go context bug, and `go vet` will flag it. Trio makes the deadline a block with no manual cleanup to forget:

```python
async def fetch_with_budget(id):
    # A deadline on the whole subtree; the move_on_after block
    # cancels everything inside it when 0.8s elapses.
    with trio.move_on_after(0.8) as cancel_scope:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(fetch_user, id)
            nursery.start_soon(fetch_order, id)
    if cancel_scope.cancelled_caught:
        raise TimeoutError("budget exceeded")
```

The `with trio.move_on_after(0.8)` block *is* the deadline. When it expires, Trio cancels the nursery inside it, every child gets `Cancelled` at its next checkpoint, the nursery joins them, and control flows past the block with `cancelled_caught` set. There is no timer to release by hand — leaving the `with` block does that. This is the structural advantage of cancel scopes over explicit contexts: the cleanup is the de-indent, so it cannot be forgotten.

## Graceful shutdown and draining

Cancellation's biggest real-world test is *shutdown*. When you deploy, scale down, or receive `SIGTERM`, you want to stop accepting new work but *finish* the work already in flight — drain the in-flight requests, flush buffers, commit the last batch — and only then exit. A process that just calls `exit()` on `SIGTERM` drops every in-flight request on the floor; under a rolling deploy with thousands of requests per second, that is a visible spike of 5xx errors every time you ship.

Structured concurrency makes graceful shutdown *fall out of the model*, because shutdown is just cancellation of the top-level scope plus a join. The pattern is a three-phase drain:

1. **Stop intake.** Close the listener / stop pulling from the queue / set "draining." No new tasks enter the top scope.
2. **Signal in-flight work and wait, with a grace deadline.** Cancel the scope (cooperatively), but give in-flight tasks a bounded grace period to finish — say 30 seconds — by wrapping the join in a deadline.
3. **Force only what is left.** If the grace deadline expires with tasks still running, *then* you escalate (forcefully terminate, or just exit and let the orchestrator restart), accepting that those few stragglers are lost. The grace period converts "drop everything" into "drop only what refused to finish in 30 s."

```go
// Graceful shutdown: stop intake, drain with a grace deadline,
// force only the stragglers.
func (s *Server) Shutdown(parent context.Context) error {
    s.listener.Close()                  // 1. stop accepting new work

    ctx, cancel := context.WithTimeout(parent, 30*time.Second)
    defer cancel()

    done := make(chan struct{})
    go func() { s.inflight.Wait(); close(done) }() // join in-flight

    select {
    case <-done:                        // 2. drained within grace
        return nil
    case <-ctx.Done():                  // 3. grace expired: escalate
        s.forceClose()                  // last resort
        return ctx.Err()
    }
}
```

Java's `ExecutorService` encodes exactly this contract in its API and it is worth knowing the precise difference, because it trips people up: `shutdown()` stops accepting new tasks but lets queued and running tasks finish (a graceful drain); `shutdownNow()` *also* interrupts the running tasks and returns the list of tasks that never started (the forceful escalation). The standard recipe is `shutdown()`, then `awaitTermination(30, SECONDS)`, then `shutdownNow()` if the await timed out — which is the three-phase drain spelled out in the standard library. The lesson: graceful shutdown is not a special subsystem; it is cancellation-with-a-grace-deadline applied to your root scope.

Two operational details make or break this in practice, and both are about *coordinating with the world outside the process*. First, the grace deadline must be shorter than whatever external timer will kill you anyway. Kubernetes, for instance, sends `SIGTERM` and then `SIGKILL` after a `terminationGracePeriodSeconds` (default 30 s); if your drain budget is 30 s and the orchestrator's is also 30 s, you will be `SIGKILL`ed mid-drain and lose the stragglers you were trying to save. Set the in-process grace deadline comfortably below the external one (say 25 s against a 30 s pod grace period) so your own escalation runs first and cleanly. Second, intake must stop *before* the load balancer stops routing to you, not after — otherwise you accept a request, start draining, and then cancel the request you just accepted, manufacturing the exact 5xx you were trying to avoid. The usual fix is a readiness probe you flip to "not ready" at the very start of `Shutdown`, wait one probe interval for the load balancer to notice and stop sending traffic, *then* begin the drain. Skipping that wait is the most common reason a "graceful" shutdown still produces an error spike on every deploy: the code was correct, but it raced the load balancer.

## Sizing the pool: CPU-bound vs IO-bound and Little's law

So far we have talked about *structure* — who owns a task, when it finishes, how it cancels. Now to the engine underneath: the **thread pool** (a.k.a. executor) that actually runs the tasks. A pool is a fixed set of worker threads pulling tasks off a queue. The first design question is the one people most often get wrong: *how many workers?*

![A matrix giving pool sizing rules for CPU-bound, IO-bound, and mixed workloads with the reasoning for each](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-6.png)

The answer depends entirely on what the tasks *do*, and the dividing line is whether a task spends its time *computing* (using a CPU) or *waiting* (blocked on I/O — a network call, a disk read, a database round-trip).

**For CPU-bound work, the right pool size is approximately the number of cores** (often `N` or `N + 1`). The reasoning is mechanical: a CPU-bound thread can only make progress when it is actually running on a core. If you have 8 cores and 8 CPU-bound threads, all 8 cores are busy — you are at peak throughput. Adding a 9th thread does not add a 9th core; it just means the OS scheduler now time-slices 9 threads across 8 cores, paying a context-switch cost (saving and restoring registers, polluting caches — roughly a microsecond or more each) for *zero* extra parallelism. More threads than cores for CPU-bound work is strictly worse: same throughput, more overhead, more memory for stacks. (This is also where the language matters: in Python, the GIL means CPU-bound threads do not run in parallel at all — for true CPU parallelism you need processes or free-threaded builds; see [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs).)

**For IO-bound work, you want many more threads than cores**, because an IO-bound thread spends most of its life *blocked*, holding no core. While thread A waits 50 ms for a database response, its core is free to run threads B, C, D... The question "how many?" is answered precisely by **Little's law**, one of the most useful results in queueing theory. Little's law states that for a stable system, the average number of items in the system equals the arrival rate times the average time each item spends in the system:

$$L = \lambda W$$

where $L$ is the average number of in-flight requests, $\lambda$ is the arrival rate (requests per second), and $W$ is the average time per request (seconds). To handle the load, you need enough concurrent capacity to hold $L$ in-flight requests at once. Rearranged for pool sizing, the classic Brian Goetz formula (from *Java Concurrency in Practice*) for the number of threads is:

$$N_{threads} = N_{cores} \times U \times \left(1 + \frac{W}{C}\right)$$

where $U$ is target CPU utilization (0 to 1), and $W/C$ is the ratio of *wait* time to *compute* time per task. If a task waits 9× as long as it computes (`W/C = 9`) and you target full utilization on 4 cores, you want roughly `4 × 1 × (1 + 9) = 40` threads. The intuition the formula encodes: each thread is "useful" only for the `C` fraction of its life when it computes; the `(1 + W/C)` factor inflates the count so that, on average, `N_cores` of them are actually computing at any instant while the rest sit blocked on I/O.

#### Worked example: sizing a pool from a latency profile

You operate a service whose handler spends, on average, 5 ms of CPU and 45 ms waiting on a downstream API and a database — so `C = 5 ms`, `W = 45 ms`, `W/C = 9`. You run on 8 cores and want about 90% CPU utilization (`U = 0.9`). The formula gives `N_threads = 8 × 0.9 × (1 + 9) = 72` threads. Cross-check with Little's law directly: if you receive `λ = 1440` requests/second and each takes `W_total = 50 ms = 0.05 s`, then `L = 1440 × 0.05 = 72` in-flight requests — you need 72 units of concurrency to keep up, matching the formula. Now the honest caveats: these are *averages*, and real latency has a fat tail — when the downstream API slows from 45 ms to 450 ms (a bad afternoon), `W/C` jumps to 90 and your 72 threads are suddenly far too few; requests queue, latency climbs, and you are in an incident. So size for the *normal* case, but pair the pool with a *bounded queue* and a timeout so the abnormal case degrades into backpressure and shed load rather than unbounded queueing (the next two sections). And measure: the formula gives you a starting point, not a final answer — sweep the pool size under realistic load and watch where throughput plateaus and latency knees upward.

The mixed-workload case (tasks that do both CPU and I/O, or a service handling both kinds) is the trap: a single pool sized for one kind starves the other. The standard fix is *bulkheading* — separate pools per workload type, so a flood of slow I/O tasks cannot consume all the threads the CPU tasks need (and vice versa), and a stall in one pool is isolated from the other.

## Bounded vs unbounded queues: the OOM trap and backpressure

Now the most important and most violated rule in pool design. A thread pool has a *queue* in front of it holding tasks waiting for a free worker. That queue can be *bounded* (fixed capacity) or *unbounded* (grows as needed). The default in many libraries — including, infamously, Java's `Executors.newFixedThreadPool` and `newSingleThreadExecutor` — is an *unbounded* queue. **An unbounded queue is a memory leak waiting for a traffic spike, and it is the single most common way a healthy-looking service falls over under load.**

![A comparison showing an unbounded queue that hides overload and grows until it crashes versus a bounded queue with backpressure that caps latency and stays alive](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-5.png)

Here is the mechanism, step by step, because the failure mode is non-obvious until you have lived it. Suppose tasks arrive faster than the pool can complete them — arrival rate `λ` exceeds service rate `μ` (the pool of `c` workers each completing `1/W` tasks per second, so `μ = c/W`). This is *overload*, and it is not a question of *if* but *when*: a marketing email goes out, a dependency slows down, a retry storm kicks in.

- **With an unbounded queue**, the excess `(λ − μ)` tasks per second simply *accumulate* in the queue. The queue depth grows linearly with time. Two terrible things happen at once. First, **latency explodes**: by Little's law, the time a task spends waiting is its queue position divided by the service rate, so as the queue grows to thousands deep, the *newest* tasks wait minutes or hours — long after the client has given up and retried (adding *more* load). The system is doing work whose results nobody wants anymore. Second, **memory grows without bound**: each queued task is a closure, a stack frame's worth of captured state, a request object. At some queue depth the heap is exhausted and the process is OOM-killed — taking down *all* in-flight work, not just the excess. The unbounded queue *hid* the overload (the pool looked healthy, the threads were all busy) right up until it converted a recoverable overload into a catastrophic crash.

- **With a bounded queue**, the queue fills to capacity and then *refuses* further tasks. This refusal is **backpressure** — a signal that propagates *upstream* to the producer: "I am full; slow down." Now the system has a choice (a *rejection policy*, next section) instead of a death spiral. Latency is *bounded*: a task waits at most `queue_capacity / μ` seconds, a number you can compute and put in an SLO. Memory is *bounded*: at most `queue_capacity` tasks' worth of state. The system stays *alive* under overload — it does less work, or sheds load, or makes the caller wait, but it does not crash. Bounded queues do not make overload pleasant; they make it *survivable and visible*.

It is worth doing the arithmetic on the time-to-death, because it is faster than people expect. The queue depth at time `t` after overload begins is `Q(t) = (λ − μ) · t`. With our running numbers — `λ = 1200`/s, `μ = 800`/s — the queue grows at 400 tasks/second. By Little's law the wait for a task that enters when the queue is `Q` deep is `Q / μ`, so the wait grows *linearly* too: after one minute the queue is `400 × 60 = 24,000` deep and a new task waits `24000 / 800 = 30` seconds — already past most client timeouts, so the system is now mostly executing work nobody is waiting for, while the clients retry and push `λ` even higher. Memory tells the same linear story: if each queued task holds, say, 8 KB of captured state, the queue consumes `8 KB × 400/s = 3.2 MB/s`. A JVM with a 2 GB heap headroom runs out in roughly `2000 / 3.2 ≈ 625` seconds — about ten minutes from the onset of a 50% overload to `OutOfMemoryError`. Ten minutes is exactly long enough for the overload to *look* survivable on a dashboard ("latency is up but we're serving traffic") and exactly short enough that nobody intervenes before the crash. That linear-growth-then-cliff is the unbounded queue's signature.

#### Worked example: where the bounded queue draws the line

Re-run the same overload against a bounded queue of 1000 with a Caller-runs policy. The queue fills in `1000 / 400 = 2.5` seconds and then *stops growing* — every further submission triggers the rejection policy. Latency is now pinned at `1000 / 800 = 1.25` seconds for queued work (and the Caller-runs submissions execute inline, slowing the producer to the pool's true rate). Memory is flat at `8 KB × 1000 = 8 MB` of queued state. There is no cliff because there is no unbounded accumulation: the `(λ − μ)` excess is *refused at the door* instead of *stored until the heap dies*. The bounded queue did not make 1200/s of demand fit through an 800/s pipe — nothing can — but it turned the impossible 400/s of excess into an honest, visible, bounded signal rather than a hidden, growing, fatal one.

The principle is general and shows up at every layer of a distributed system, which is why this series links it to the architecture-level treatments: backpressure is how a fast producer is forced to match a slow consumer's rate instead of overwhelming it. For the in-process pool it is a bounded queue; for a service mesh it is a rate limiter and load shedder ([rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure)); for a message broker it is flow control on the channel ([backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control)). Same physics, different scale. **The rule, stated bluntly: never use an unbounded queue in production unless you can mathematically prove the producer cannot outrun the consumer.** You almost never can.

```java
// BUG: newFixedThreadPool uses an UNBOUNDED LinkedBlockingQueue.
// Under overload, the queue grows until the heap is gone.
ExecutorService pool = Executors.newFixedThreadPool(8);

// FIX: a bounded queue + an explicit rejection policy.
ExecutorService pool = new ThreadPoolExecutor(
    8, 8,                                   // core = max = 8 workers
    0L, TimeUnit.MILLISECONDS,
    new ArrayBlockingQueue<>(1000),         // BOUNDED: at most 1000 waiting
    new ThreadPoolExecutor.CallerRunsPolicy() // backpressure: caller runs it
);
```

That two-line difference is the difference between a service that sheds load gracefully and one that OOM-crashes at the worst moment. The `CallerRunsPolicy` is especially elegant: when the queue is full, the *submitting* thread runs the task itself, which means it is too busy to submit more — automatic, self-throttling backpressure with no extra machinery.

## Rejection policies: who pays when the queue is full

A bounded queue forces a decision the unbounded queue let you avoid: *when the queue is full and a worker is busy, what happens to the next submitted task?* This is the **rejection policy**, and the choice encodes your priorities about latency, completeness, and where the pain lands.

![A matrix of rejection policies abort, caller-runs, drop oldest, and block with their behavior and the situation each fits](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-7.png)

There are four canonical policies, and each is correct *for some workload*:

| Policy | What happens on a full queue | Backpressure? | Loses work? | Use when |
| --- | --- | --- | --- | --- |
| **Abort** (reject) | Submission throws / returns an error immediately | Yes — caller must handle it | No (caller decides) | The caller can react: retry with backoff, return 503, fail fast. The default for explicit overload handling. |
| **Caller-runs** | The submitting thread executes the task itself | Yes — strongest, self-throttling | No | You want automatic throttling and the submitter *can* safely run the task (no risk of deadlock if the task re-submits). |
| **Drop oldest** | Evict the head of the queue, enqueue the new task | Weak | Yes — silently | Newer data supersedes older (live metrics, latest sensor reading); stale work is worthless. |
| **Block** | Park the submitter until a slot frees | Yes — but can stall | No | A pipeline where blocking the producer is the intended flow control — but beware deadlock if the producer and consumer share a pool. |

The decisive guidance: **prefer Abort or Caller-runs for request-serving systems**, because both surface the overload (Abort to the caller as an error it can shed/retry; Caller-runs by slowing the submitter) without silently losing work or risking the deadlock that Block can cause. Use **Drop** only when the data model genuinely makes stale work worthless. And be very careful with **Block** when the producer and consumer might be the same pool — if a worker submits a sub-task to the same full pool and then blocks waiting for a slot, and the only thread that could free a slot is itself, you have manufactured a deadlock. (This is the classic *thread-pool starvation deadlock*; the safe rule is that tasks running in a pool must not block waiting on other tasks in the *same* pool.)

#### Worked example: choosing a policy under a retry storm

A payments service runs a pool of 16 workers with a bounded queue of 200. A downstream provider slows down; latency triples; the queue fills. Consider the policies. **Block** would park the request threads (which here *are* the HTTP server's threads), so the whole server stops accepting connections — and the upstream load balancer, seeing no response, retries, doubling the offered load. Death spiral. **Drop oldest** would silently discard payment attempts — unacceptable; a dropped payment is a lost sale and a support ticket. **Abort** is correct here: the pool rejects, the handler catches the rejection, returns HTTP 503 with a `Retry-After` header, and the load balancer / client backs off. The overload is now *visible* (503 rate spikes your dashboard) and *bounded* (you serve a fast error instead of a slow timeout), and the retry budget is controlled by `Retry-After` rather than a stampede. The lesson: the rejection policy is not a low-level detail — it is the difference between "we shed 8% of traffic for two minutes" and "the whole service fell over."

## Worked example: a cancellable bounded worker pool with a deadline

Let me assemble everything into the canonical artifact this post promised: a worker pool with a *bounded* queue, *backpressure*, *cooperative cancellation*, and a *deadline* on the whole batch. I will show it in Go (explicit, channel-based) and Kotlin (structured coroutines), because the idioms diverge instructively.

The Go version uses a buffered channel as the bounded queue (its capacity *is* the bound and a full channel *is* backpressure on the sender), a `context` with a deadline for cancellation, and a `sync.WaitGroup` to join the workers — a hand-rolled scope:

```go
// A cancellable, bounded worker pool with a deadline.
func ProcessBatch(parent context.Context, items []Item) error {
    ctx, cancel := context.WithTimeout(parent, 5*time.Second)
    defer cancel()

    const workers = 8
    jobs := make(chan Item, 1000) // BOUNDED queue: cap 1000 = backpressure
    g, ctx := errgroup.WithContext(ctx)

    // Start the worker pool (the "scope" of running tasks).
    for w := 0; w < workers; w++ {
        g.Go(func() error {
            for {
                select {
                case <-ctx.Done(): // cancellation / deadline checkpoint
                    return ctx.Err()
                case item, ok := <-jobs:
                    if !ok {
                        return nil // queue closed and drained: clean exit
                    }
                    if err := process(ctx, item); err != nil {
                        return err // first failure cancels the rest
                    }
                }
            }
        })
    }

    // Producer: feed the bounded queue. A full channel BLOCKS here
    // (backpressure), but we also honor cancellation so we never
    // wedge after a deadline or a worker failure.
    g.Go(func() error {
        defer close(jobs) // signal workers no more items
        for _, item := range items {
            select {
            case <-ctx.Done():
                return ctx.Err()
            case jobs <- item: // blocks if the queue is full
            }
        }
        return nil
    })

    return g.Wait() // join: all workers + producer; first error wins
}
```

Every property is present and visible. The queue is bounded at 1000 (`make(chan Item, 1000)`); when it is full, `jobs <- item` blocks the producer — that is backpressure. Cancellation is cooperative: every worker checks `<-ctx.Done()` in its `select`, so when the 5-second deadline fires (or a worker returns an error, which `errgroup` turns into a context cancel), every worker sees it at its next loop iteration and exits cleanly. The producer *also* selects on `ctx.Done()`, so a deadline that fires while the producer is blocked on a full queue unwedges it instead of hanging. `g.Wait()` is the join — `ProcessBatch` does not return until every worker and the producer have finished. No goroutine outlives the function.

The Kotlin version is dramatically shorter because `coroutineScope` *is* the join and structured cancellation is built in; the bound comes from a `Channel` with a fixed capacity, and the deadline from `withTimeout`:

```kotlin
// The same pool in Kotlin: coroutineScope is the join,
// withTimeout is the deadline, the Channel capacity is the bound.
suspend fun processBatch(items: List<Item>) = withTimeout(5_000) {
    val jobs = Channel<Item>(capacity = 1000) // BOUNDED = backpressure

    coroutineScope {                            // <-- the scope (join at end)
        // 8 workers
        repeat(8) {
            launch {
                for (item in jobs) {            // suspends if empty
                    process(item)               // cancellation lands at suspend points
                }
            }
        }
        // producer
        launch {
            for (item in items) jobs.send(item) // SUSPENDS if full = backpressure
            jobs.close()
        }
    } // <-- coroutineScope joins ALL launches here; any failure cancels siblings
}
```

Read the differences. Go threads `ctx` explicitly and you must remember `defer cancel()`; Kotlin's `withTimeout` and `coroutineScope` make the deadline and the join *structural* (the closing brace joins every `launch`, and a timeout cancels them all cooperatively at their suspension points — `jobs.send`, the `for` loop's receive, and inside `process` if it suspends). Both give you the same four properties; Kotlin just spends fewer characters and forgets fewer things, because the scope and the cancellation are baked into the language rather than passed by hand. The Java `StructuredTaskScope` version sits in between — explicit `fork`/`join` like Go's `errgroup`, but with the `try`-with-resources block enforcing the join like Kotlin's brace.

Rust occupies a third interesting point: it has no native nursery, but `tokio::task::JoinSet` plus a bounded channel and a `timeout` future compose into the same shape, and the borrow checker adds a guarantee the other languages cannot — a task that tries to borrow data outliving the scope simply will not compile.

```rust
// The same pool in Rust: a bounded mpsc channel for backpressure,
// a JoinSet as the scope, and tokio::time::timeout as the deadline.
async fn process_batch(items: Vec<Item>) -> anyhow::Result<()> {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (tx, rx) = tokio::sync::mpsc::channel::<Item>(1000); // BOUNDED
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        let mut set = tokio::task::JoinSet::new();              // the scope

        for _ in 0..8 {
            let rx = rx.clone();
            set.spawn(async move {
                while let Some(item) = rx.lock().await.recv().await {
                    process(item).await?;   // cancel lands at .await points
                }
                Ok::<_, anyhow::Error>(())
            });
        }
        for item in items {
            tx.send(item).await?;   // .send AWAITS if full = backpressure
        }
        drop(tx);                   // close: workers drain then exit

        while let Some(res) = set.join_next().await {  // join all
            res??;                  // first failure propagates
        }
        Ok(())
    })
    .await?  // timeout error if the 5s budget blew
}
```

Tokio cancellation is *drop-based* rather than signal-based: when the `timeout` future expires, the inner future is dropped, which drops the `JoinSet`, which aborts every spawned task at its next `.await`. The destructor-runs-on-drop model means cleanup (closing files, releasing connections via `Drop` impls) happens deterministically as the task unwinds — the same cooperative-at-checkpoints semantics as Trio and Kotlin, expressed through ownership rather than an exception. Across all four languages the four properties are identical; only the *spelling* of "scope" and "cancel" changes.

## Measured behavior: bounded vs unbounded under overload

Claims about OOM and latency are cheap; the point of this series is to *measure*. Here is the experiment and what it shows. Take a pool of 8 workers, each task taking ~10 ms (so service rate `μ ≈ 800` tasks/s). Drive it with a producer at `λ = 1200` tasks/s — a 50% overload — for 60 seconds, once with an unbounded queue and once with a bounded queue of 1000 plus a Caller-runs rejection policy. Measure queue depth, p99 latency, peak heap, and completed-task throughput. These are representative numbers from this kind of microbenchmark on a commodity 8-core machine; the *shape* is the robust result, and you should reproduce it on your own hardware before trusting any single figure.

| Metric under 50% overload | Unbounded queue | Bounded (1000) + Caller-runs |
| --- | --- | --- |
| Queue depth at 60 s | ~24,000 and climbing | pinned at 1000 (full) |
| p99 task latency | 28 s and rising (unbounded) | ~1.3 s (bounded by `cap/μ`) |
| Peak heap | ~1.9 GB, then OOM-killed | ~190 MB, flat |
| Throughput (completed/s) | ~800 until crash, then 0 | ~800 sustained |
| Outcome at 5 min | process dead | alive, shedding/throttling |

The story the numbers tell: **both configurations complete work at the same rate (~800/s) — the pool's throughput is fixed by 8 workers × 10 ms, and no queue policy changes that.** What the queue policy changes is *what happens to the excess 400 tasks/s that cannot be served.* The unbounded queue *stores* them — latency and memory grow linearly with time until the heap is exhausted and the process dies, at which point throughput drops to zero and *all* in-flight work is lost. The bounded queue *refuses* them — latency is capped at `1000 / 800 ≈ 1.25 s`, memory is flat, and the Caller-runs policy throttles the producer so the system finds a stable operating point at its true capacity. The unbounded queue did not give you more throughput; it gave you a *delayed catastrophic failure* dressed up as a healthy-looking system.

#### Worked example: the pool-size sweep

A second measurement worth doing yourself: hold the workload fixed (an IO-bound task with `W/C ≈ 9`, 8 cores) and sweep the pool size, recording throughput. The shape you will see: throughput climbs steeply from 1 thread, because each added thread fills a core that was idle while others blocked on I/O; it *plateaus* somewhere near the Little's-law number (around 72 for our earlier example); and past the plateau it sags slightly as context-switch overhead and memory pressure from thousands of thread stacks start to bite. The knee is your answer — and it confirms the formula gives a starting point, not gospel. For a *CPU-bound* task the curve is different and sharper: throughput climbs to `N_cores` threads and then is *flat* (more threads add no cores) or slightly *down* (context-switch tax). Measure both kinds on your hardware once and the sizing rules stop being abstract — you will *see* the IO-bound curve want dozens of threads and the CPU-bound curve refuse to benefit past the core count. Honest measurement caveats apply throughout: warm up the JIT/runtime, run many trials, pin or at least name the machine, and remember the OS scheduler is a confounding variable — these curves wobble run-to-run, and the *trend* is the signal, not any single point.

## Case studies / real-world

**Nathaniel Smith's "Notes on Structured Concurrency, or: Go statement considered harmful" (2018).** This essay is the origin of the term and the cleanest statement of the `goto` analogy. Smith's argument — that an unscoped `go`/`spawn` breaks the correspondence between a program's text and its execution exactly as `goto` did, and that a nursery restores it — directly motivated Trio's design and, downstream, influenced Kotlin's structured coroutines and the JDK's structured-concurrency JEP. The historical rhyme is deliberate: it is the same move Dijkstra made for sequential control flow, applied 50 years later to concurrency. If you read one source on this topic, read that one.

**Java's `StructuredTaskScope` (Project Loom).** The JDK has been standardizing structured concurrency as a first-class API, paired with virtual threads (millions of cheap threads scheduled onto a small carrier pool). It went through several preview rounds (JEP 428, 437, 453, 462, 480, 499) across JDK 19 through 24 and is targeting stabilization. The design choices are instructive precisely because the JDK had to be conservative: `StructuredTaskScope` makes the `try`-with-resources block the scope, `fork` the spawn, `join` the barrier, and a *joiner policy* the error/cancellation strategy — and it integrates with thread *interruption* as the cancellation mechanism (forceful at the carrier level, but the standard library's blocking calls are interrupt-aware, so it behaves cooperatively for well-written tasks). It is a real-world demonstration that the nursery model ports to a language with checked exceptions, a strong memory model, and a 25-year-old threading API.

**The unbounded-queue OOM, a recurring production incident.** This one does not have a single famous name because it has happened to almost everyone who used `Executors.newFixedThreadPool` (or its equivalents) without reading that its queue is unbounded. The pattern is always the same: a service runs fine for months; a dependency slows or a traffic spike arrives; tasks queue faster than they drain; the `LinkedBlockingQueue` grows to hundreds of thousands of entries; the JVM throws `OutOfMemoryError` and the process dies, often *cascading* as the load shifts to surviving instances and OOMs them in turn. The fix is invariably the same two lines: a bounded `ArrayBlockingQueue` and an explicit `RejectedExecutionHandler`. The reason it keeps happening is that the unbounded default *hides* the problem during testing (you never overload a test environment for an hour) and only reveals it under the exact production conditions you cannot easily reproduce. It is the strongest possible argument for the rule "bound your queues by default." The same lesson recurs at the distributed scale: a queue between services with no flow control (see [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control)) fails the identical way, one layer up.

## When to reach for this (and when not to)

**Reach for structured concurrency whenever you start more than one concurrent task that you eventually need to coordinate.** Any fan-out (query N shards, call M services, process a batch in parallel) belongs in a scope: you get the join, the error propagation, and the cancellation for free, and the alternative (raw spawns plus hand-rolled `WaitGroup` plus a shared error variable plus a context) is the same code with more bugs. If your language has a native nursery / `coroutineScope` / `StructuredTaskScope`, use it; if it only has a bare spawn (Go's `go`, Rust's `tokio::spawn`), adopt the community scope wrapper (`errgroup`, `tokio::task::JoinSet` / `tokio-scoped`) and treat the bare spawn as a primitive you do not call directly in business logic.

**When is an unscoped, truly fire-and-forget task acceptable?** Rarely, and only when *all* of these hold: the task's failure genuinely does not matter (you have an out-of-band way to detect the bad outcome), it has its own internal timeout so it cannot run forever, and it is registered with a long-lived top-level scope (or supervisor) that *can* cancel it at shutdown. "Log a metric, best-effort" can qualify. "Send the confirmation email" almost never does — you care if that fails. The honest default is: if you are tempted to fire-and-forget, you probably want a long-lived nursery to hand the task to, not no nursery at all.

**On cancellation:** make cooperative the default *always*. Reach for forceful cancellation (thread interrupt, process kill) only for a runaway task you cannot make cooperative — typically third-party CPU-bound code with no checkpoints — and prefer to isolate such code in a *separate process* you can kill cleanly rather than a thread you must violently interrupt. Never use forceful cancellation as a routine control-flow mechanism; the torn-state and stranded-lock risks are real and unrepairable.

**On pool sizing:** do not reach for "more threads" as a performance fix for CPU-bound work — past the core count it makes things worse, not better; if CPU is your bottleneck, you need more cores or less work, not more threads. Do not use a thread-per-task model for high-concurrency IO-bound work in a runtime with expensive OS threads — use async/event-loop concurrency or virtual threads instead (see [async/await and how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) for why a coroutine is far cheaper than an OS thread when the work is mostly waiting). And do not size a single pool for a mixed workload — split it, so slow I/O cannot starve fast compute.

**On queues:** never ship an unbounded queue to production. The only defensible exception is a queue you can *prove* the producer cannot outrun (e.g., a fixed-size batch where the producer is itself rate-limited), and even then a generous bound is cheap insurance. Bound the queue, choose a rejection policy that matches your data model, and make the overload *visible* — a spike in rejections or 503s is a signal you can alert on; a slowly filling unbounded queue is a time bomb you cannot.

For the broader "which concurrency model do I even want here" decision — threads vs async vs actors vs channels vs data-parallel — the series capstone, [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model), puts structured concurrency in context as the discipline that should wrap *whichever* model you pick. And for the lower-level coordination primitives a pool is built from — the semaphore that bounds concurrency, the latch that signals "all started," the barrier that synchronizes a phase — see [semaphores, barriers, and latches](/blog/software-development/concurrency/semaphores-barriers-and-latches); a bounded pool is, at heart, a semaphore guarding worker slots plus a queue.

![A matrix comparing structured-concurrency primitives across Python Trio, Kotlin, Java Loom, and Go with their scope and cancellation mechanisms](/imgs/blogs/structured-concurrency-cancellation-and-thread-pool-design-8.png)

The matrix above is the portable summary: four runtimes, the same two ideas. Each gives you a *scope primitive* that owns and joins a set of tasks (`open_nursery`, `coroutineScope`, `StructuredTaskScope`, `errgroup` + `WaitGroup` over a `context`), and a *cancellation channel* that delivers a stop request to those tasks at their checkpoints (`Cancelled`, `CancellationException`, thread interrupt, `ctx.Done()`). If you can find those two things in a new language's concurrency library, you can write safe, leak-free, cancellable concurrent code in it the same afternoon. If you *cannot* find them, that is your warning to build them before you write anything else.

## Key takeaways

1. **An unscoped spawn breaks three things at once** — error propagation, resource accounting, and cancellation — because all three are implemented by the region of code that owns the task. No owner, no error path, no lifetime bound, no cancel handle.
2. **Structured concurrency = every task runs inside a scope that does not return until all its children finish.** The scope makes concurrency follow the call tree, exactly as structured programming made control flow follow nested blocks.
3. **The unscoped `go`/`spawn`/`submit` is the `goto` of concurrency.** It breaks the correspondence between a program's text and its execution; a scope restores it. Treat a bare spawn as a primitive you wrap, never an API you call directly in business logic.
4. **Cancellation must be cooperative.** A cancel is a request honored at checkpoints (every `await`/suspension/`select`), so the task stops only where its state is consistent. Forceful cancellation (thread kill, interrupt) can strand locks and tear state — last resort only, ideally by killing a whole process.
5. **Cancellation propagates down a scope and deadlines compose by taking the minimum.** Pass deadlines as a single shrinking budget, not a fresh clock at each layer; in Go always `defer cancel()`.
6. **Graceful shutdown is just cancellation-with-a-grace-deadline on the root scope:** stop intake, drain in-flight work with a bounded grace period, then forcefully escalate only the stragglers.
7. **Size CPU-bound pools to ~`N_cores`; size IO-bound pools by Little's law** `$L = \lambda W$`, i.e. `N_cores × U × (1 + W/C)`. More threads than cores for CPU work is strictly worse; far more for IO work is necessary. Bulkhead mixed workloads into separate pools.
8. **Never use an unbounded queue in production.** It hides overload, grows latency without bound, and converts a recoverable overload into an OOM crash. A bounded queue turns overload into backpressure: bounded latency, bounded memory, a system that stays alive.
9. **The rejection policy decides who pays when the queue is full.** Prefer Abort or Caller-runs for request-serving systems (surface the overload, don't lose work, don't deadlock); use Drop only when stale work is worthless; use Block only when producer and consumer are not the same pool.
10. **Measure, don't assume.** Bounded vs unbounded have the *same* throughput — the difference is survival vs delayed catastrophe. Sweep your pool size and watch where throughput knees; the formula is a starting point, the curve is the answer.

## Further reading

- Nathaniel J. Smith, "Notes on Structured Concurrency, or: Go statement considered harmful" (2018) — the origin of the term and the `goto` analogy; the single best starting point.
- Edsger W. Dijkstra, "Go To Statement Considered Harmful" (1968) — the structural-control-flow argument that structured concurrency rhymes with 50 years later.
- Brian Goetz et al., *Java Concurrency in Practice* — Chapters 6–8 on task execution, `Executor` framework, and pool sizing, including the `N_cores × U × (1 + W/C)` formula and the unbounded-queue warning.
- The JDK Structured Concurrency JEPs (428 / 453 / 480 / 499 and successors) and the `StructuredTaskScope` / virtual-threads documentation — the canonical real-world implementation in a mainstream language.
- The Trio documentation, especially the nursery and cancellation/timeout sections — the cleanest reference implementation of the model, with the `move_on_after` / cancel-scope design.
- The Go `context` package documentation and `golang.org/x/sync/errgroup` — the idiomatic Go approach to deadlines, cancellation, and scoped error propagation.
- Kotlin coroutines guide, "Coroutine context and dispatchers" and "Cancellation and timeouts" — `coroutineScope`, `supervisorScope`, structured cancellation at suspension points.
- John D. C. Little, "A Proof for the Queuing Formula `$L = \lambda W$`" (1961) — the queueing result that underpins IO-bound pool sizing.

Within this series, this post is the lifecycle layer that sits over everything else: see [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) for the foundation, [async/await and how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) and [semaphores, barriers, and latches](/blog/software-development/concurrency/semaphores-barriers-and-latches) for the primitives a pool is built from, and [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) for choosing the model that this discipline should wrap.

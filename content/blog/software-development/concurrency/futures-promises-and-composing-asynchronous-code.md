---
title: "Futures, Promises, and Composing Asynchronous Code"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a future turns a not-yet-ready value into something you can chain, combine, race, and fan out without drowning in nested callbacks."
tags:
  [
    "concurrency",
    "parallelism",
    "futures",
    "promises",
    "async-await",
    "callbacks",
    "composition",
    "error-handling",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/futures-promises-and-composing-asynchronous-code-1.png"
---

Here are two pieces of code that do exactly the same thing: load a user, then load that user's orders, then load the line items for the first order. The first one is the version most of us wrote in 2013.

```javascript
getUser(userId, (err, user) => {
  if (err) return done(err);
  getOrders(user.id, (err, orders) => {
    if (err) return done(err);
    getItems(orders[0].id, (err, items) => {
      if (err) return done(err);
      done(null, { user, orders, items });
    });
  });
});
```

The logic marches to the right. Every step nests inside the previous one's callback, and the error check is copy-pasted three times because there is no other place to put it. Miss one `if (err) return` and a failure vanishes without a trace. This shape has a name — the **pyramid of doom**, or just *callback hell* — and it is the thing futures were invented to kill.

Here is the second version, with futures and `await`:

```javascript
const user = await getUser(userId);
const orders = await getOrders(user.id);
const items = await getItems(orders[0].id);
return { user, orders, items };
```

Same three dependent steps, same order, same error semantics — but it reads top-to-bottom like ordinary sequential code, and a single `try/catch` around it covers all three failure points. The right-drifting pyramid became a flat list. That collapse, from nested callbacks to a linear chain, is the whole subject of this post, and figure 1 puts the two shapes side by side.

![A nested three deep callback pyramid with error checks scattered across levels next to a flat composed future chain where one handler covers all errors](/imgs/blogs/futures-promises-and-composing-asynchronous-code-1.png)

But `await` is just sugar. Underneath it is a **future** — a handle to a value that is not ready yet — and the real skill is composing those handles: chaining them, waiting for all of them, racing them, propagating errors through them, and fanning a request out to a dozen of them at once without serializing the whole thing by accident. That last mistake, awaiting inside a loop, is the single most common async performance bug I see in code review, and by the end of this post you will see exactly why it turns three 100-millisecond calls into a 300-millisecond wall-clock wait when it should have been 100.

This post is part of the [Concurrency & Parallelism, From the Ground Up](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) series. The series spine is simple: concurrency is about correctness under nondeterministic scheduling, and the discipline is to name what is shared, establish an order over every access, and pick the cheapest mechanism that buys that order. Futures are that mechanism for a *single asynchronous value*. The value lives in one place, exactly one writer settles it, and the happens-before edge from "the value was produced" to "the continuation observes it" is built into the type. Get that edge right and a whole class of races simply cannot happen.

## A future is a handle to a value that is not ready yet

Start with the type, because the type is the idea. A **future** is an object that represents a value that will exist *later*. Right now it is empty — call that state **pending**. At some point in the future the underlying work finishes and the future becomes **settled**: either **fulfilled** with a value, or **rejected** with an error. Crucially, a future settles **exactly once** and is then immutable. You cannot un-fulfill a future or change its value after the fact. That one-shot, write-once property is what makes futures safe to share across tasks: there is nothing to race on once it is settled.

Now the split that confuses almost everyone the first time. In most libraries the *reader* side and the *writer* side are two different objects:

- A **future** is the **reader** handle. You hold it, you register a continuation on it, you `await` it. You cannot complete it.
- A **promise** is the **writer** handle. Whoever holds it is responsible for completing the future, exactly once, with a value or an error.

In Rust the reader side is `Future` and the writer-ish side is a channel sender or `oneshot::Sender`. In Java the future is `Future<T>` (or the more capable `CompletableFuture<T>`, which is both), and the writer calls `complete(value)`. In C++ the pair is literally named: `std::promise<T>` is the writer, and you get the reader by calling `promise.get_future()`. The names are not arbitrary; the *promise* is the obligation to produce, the *future* is the right to consume.

JavaScript is the famous exception that fuses them. The `Promise` object is *both* — the executor function receives `resolve`/`reject` (the writer capabilities) and the `Promise` itself is the reader you call `.then()` on. So JS programmers say "promise" for everything, while Rust, Java, C++, and Scala programmers keep the two words distinct. When you read across ecosystems, mentally translate: a JS "promise" is a Rust "future you can also complete from outside."

Here is the producer/consumer split made concrete in C++, where the two objects are named separately:

```cpp
#include <future>
#include <thread>

std::promise<int> writer;            // the writer handle
std::future<int> reader = writer.get_future();  // the reader handle

std::thread producer([&writer] {
    int result = expensive_computation();
    writer.set_value(result);        // settle exactly once
});

int value = reader.get();            // blocks until settled, then reads
producer.join();
```

The producer thread owns the `promise` and is the only thing that can `set_value` (or `set_exception`). The consumer thread owns the `future` and can only *read*. Try to call `set_value` twice and C++ throws `std::future_error` with `promise_already_satisfied` — the one-shot rule enforced at runtime. That asymmetry is the entire safety story: one writer, one settle, many possible readers, and a clean happens-before edge from the `set_value` to the `get`.

Why does this matter for correctness, not just ergonomics? Because the future *replaces* the shared-variable-plus-flag pattern that is the source of so many data races. The tempting but wrong way to "return a value from a thread" is to write into a shared field and flip a boolean: `result = x; ready = true;`. The reader spins on `while (!ready) {}` and then reads `result`. On any real CPU with a relaxed memory model, that is broken — the reader can observe `ready == true` *before* the write to `result` is visible, because the compiler and the hardware are free to reorder two independent stores (the [reordering post](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) walks through exactly how). A future closes that hole because the library author already inserted the release/acquire barrier inside `set_value`/`get`. You do not get to forget it. The type *is* the synchronization.

This is also why a future is safe to *hand to multiple readers*. Once settled, a future is immutable, and immutable shared data is never a race — there is no write to interleave with. C++'s `std::shared_future` makes this explicit: it is a future you can copy and read from many threads, precisely because the value is frozen after the single settle. A plain `std::future` is move-only (one reader) by default; you opt into multi-reader sharing only when you mean it. The "exactly one writer, then frozen, then any number of readers" shape is the safest concurrent data structure there is, and a future hands it to you packaged.

What does the lifecycle actually look like in time? It is a tiny state machine — pending, then a single transition to fulfilled or rejected, then the continuation fires. Figure 2 lays out that timeline. Notice there is no path back to pending; the arrow only goes forward, which is precisely why a settled future is immutable and safe to read concurrently.

![A timeline showing a future moving from pending through running work to either fulfilled or rejected and then firing its continuation exactly once](/imgs/blogs/futures-promises-and-composing-asynchronous-code-2.png)

#### Worked example: the happens-before edge a future buys you

Suppose thread A computes a configuration object and thread B needs to read it. Without a future you would share a plain variable and a flag, and now you have a data race: B might see the flag set before A's writes to the object are visible, depending on the memory model (the [memory-models post](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) covers exactly why). With a future, the library guarantees that *everything thread A did before calling `set_value` happens-before everything thread B does after `get` returns*. The settle is a release, the read is an acquire, and the future's internals carry the barrier. You did not have to reason about acquire/release yourself — you reached for a handle whose type encodes the ordering. That is the recurring move of this whole series: name the shared value (the config), establish the order (settle happens-before read), pick the cheapest mechanism that buys it (a future), done.

## Callbacks and how they spiral into callback hell

Before futures, the universal way to "return later" was the **callback**: you pass a function in, and the asynchronous operation calls it when the work is done. The simplest convention, and the one Node.js standardized, is *error-first callbacks* — the callback's first argument is an error (or null), the rest is the result.

```javascript
function getUser(id, callback) {
  db.query("SELECT * FROM users WHERE id = ?", [id], (err, rows) => {
    if (err) return callback(err);
    if (rows.length === 0) return callback(new Error("not found"));
    callback(null, rows[0]);
  });
}
```

One callback in isolation is fine. The problem is *composition*. The moment step B depends on the result of step A, and step C depends on B, you cannot write them as three statements — each one has to live inside the previous callback's body, because that is the only scope where the previous result exists. So they nest. Three dependent steps means three levels of indentation; five means five. The code drifts right until it falls off the screen, and that is callback hell.

There are three distinct pathologies hiding in that pyramid, and it is worth naming them separately because futures fix each one:

1. **Rightward drift.** Each dependency adds a level of nesting. The structure of the code stops reflecting the logic ("do A, then B, then C" should be three lines, not three nested scopes).
2. **Scattered, repetitive error handling.** There is no single place a failure can propagate to. You re-handle the error at every level, and the instant you forget one `if (err) return callback(err)`, the failure is silently swallowed — the callback for the *success* path runs with `undefined` data, and you get a confusing crash three steps later instead of at the source.
3. **Inversion of control.** You handed your continuation to someone else's function and you are trusting *them* to call it — exactly once, with the right arguments, on the right thread. A buggy library that calls your callback twice, or never, or synchronously when you expected async, corrupts your control flow and there is no type that stops it.

That third one is subtle and worth dwelling on. When you `await` a future, *you* are in control — the runtime resumes your function. When you pass a callback, *they* are in control. Promises/A+, the JavaScript spec we will look at in the case studies, exists largely to nail down the answers to "exactly once? what thread? sync or async?" so that the inversion is at least *predictable*.

Here is the same load-user-then-orders-then-items flow in raw callbacks, so the pyramid is concrete:

```javascript
function loadDashboard(userId, done) {
  getUser(userId, (err, user) => {
    if (err) return done(err);                 // error handling, copy 1
    getOrders(user.id, (err, orders) => {
      if (err) return done(err);               // error handling, copy 2
      getItems(orders[0].id, (err, items) => {
        if (err) return done(err);             // error handling, copy 3
        done(null, { user, orders, items });   // success, buried 3 deep
      });
    });
  });
}
```

Three `if (err) return done(err)` lines that are nearly identical, one success line buried at the bottom of the pyramid, and a control flow you have to read inside-out. Now watch what a future does to it.

It gets worse when the steps are not a straight line. Suppose step B and step C are *independent* — both depend on A but not on each other — and you want them to run at the same time, then combine. In callbacks there is no clean way to express "wait for both of these to finish." You end up hand-rolling a counter: launch both, have each callback decrement a shared `pending` count, and only call the final continuation when the count hits zero — while also tracking whether either failed, and being careful not to call the continuation twice. That manual join logic is fiddly, easy to get wrong (off-by-one on the counter, double-invocation on error, a race on the shared count if the callbacks fire on different threads), and it is *re-implemented from scratch at every fan-out site*. The combinators we will reach in a few sections — `all`, `allSettled` — exist precisely to package that join logic once, correctly, so you never hand-roll a completion counter again. Callbacks give you no vocabulary for "combine these N outcomes"; futures give you a whole one.

There is one more callback hazard worth naming because it is genuinely subtle: **synchronous-versus-asynchronous invocation of the callback.** Some callback APIs invoke your callback *synchronously* in the common case (the value was cached, so why wait?) and *asynchronously* otherwise. That inconsistency — sometimes the callback runs before the registering function returns, sometimes after — is the "release Zalgo" bug Isaac Schlueter named: it means the order of your statements is no longer deterministic, and code that relied on "the callback runs later" breaks intermittently. Promises/A+ banned it outright by mandating that continuations *always* run asynchronously, on a future tick, never synchronously. That guarantee — "a `.then` callback never runs before the `.then` call returns" — is one of the quiet reasons futures are easier to reason about than raw callbacks: the timing is *predictable*.

## Continuation-passing, and how futures flatten it

The callback style above has a formal name: **continuation-passing style**, or CPS. Instead of a function *returning* a value, you pass it the "continuation" — the function representing what to do next with that value. CPS is completely general; any program can be mechanically rewritten into it. The trouble is that the rewrite is exactly the nesting we just saw: the continuation of the continuation of the continuation, growing one scope deeper per step.

To see why CPS *forces* nesting, look at what a continuation is: it is "the rest of the computation," captured as a function. In direct style you write `let a = f(); let b = g(a); return h(b);` — three statements, the rest of the program is implicit in what comes after each line. In CPS you make "the rest" explicit by passing it in: `f(a => g(a, b => h(b, done)))`. Each function no longer *returns*; it *calls its continuation with the result*. And the continuation of `f` literally *contains* the continuation of `g`, which contains the continuation of `h` — so they nest by construction. The deeper the data dependency, the deeper the nest. There is no way to flatten it while keeping the continuation as a bare lambda argument, because the inner lambda needs to close over the outer's result.

A future is the trick that lets you write CPS *without* the nesting. The key insight is that a future is a **reified continuation point** — it is a first-class object representing "the value that will flow into the next step." Because it is an object you can return it, store it, pass it around, and most importantly attach the next step to it with a method instead of by nesting inside a lambda. The continuation is still there — it is the argument to `.then` — but it no longer has to *contain* the next continuation, because the next continuation attaches to the future that `.then` *returns*. The nesting becomes chaining: the recursion in the source code unrolls into a left-to-right pipeline of `.then` calls, each operating on the future the previous one produced.

That method is usually called `then` (JavaScript), `map`/`and_then` (Rust), or `thenApply`/`thenCompose` (Java). And here is the move that flattens everything: **`then` returns a new future.** So instead of nesting, you *chain* — each `.then()` hands you back a fresh future you can call `.then()` on again, at the same indentation level.

```javascript
getUser(userId)
  .then((user) => getOrders(user.id))
  .then((orders) => getItems(orders[0].id))
  .catch((err) => handle(err));
```

The pyramid is gone. Three steps, three lines, all at one indent, and *one* `.catch` at the bottom that every step's failure flows into. The continuation that used to be nested is now an argument to `.then`, and the chaining happens horizontally through return values instead of vertically through nesting.

There is one more thing `.then` does that pure CPS does not, and it is the difference between `map` and `flatMap`. Notice the first callback, `(user) => getOrders(user.id)`, *returns another future* (`getOrders` is async). A naive `map` would give you a `Future<Future<Orders>>` — a future of a future, nested values you would have to unwrap manually. But `then` is smart: if your callback returns a future, `then` **flattens** it, "adopting" the inner future's eventual state as its own. This flattening is `flatMap` (called `thenCompose` in Java, `and_then` in Rust, `flatMap` in Scala). It is exactly the monadic bind, and it is the reason chains stay flat even when every step is itself asynchronous.

It is worth seeing the two operations distinctly. `map` (or JS `then` with a *plain* return) transforms the value:

```javascript
getUser(userId)
  .then((user) => user.name.toUpperCase())   // map: User -> String, stays a Future<String>
  .then((name) => console.log(name));
```

`flatMap` (JS `then` with a *future* return) sequences a dependent async step and flattens:

```javascript
getUser(userId)
  .then((user) => getOrders(user.id))        // flatMap: User -> Future<Orders>, flattened to Future<Orders>
  .then((orders) => console.log(orders.length));
```

JavaScript blurs the two because `then` does both depending on what you return; Rust and Scala keep them as separate methods (`map` vs `and_then`/`flatMap`) so the types are explicit. Either way, the mechanism is the same: chaining replaces nesting, and flattening keeps a chain of async steps from turning into a tower of nested futures.

#### Worked example: why flattening is not optional

Walk a two-step chain through the types and watch where a nested future would derail you. You start with `getUser(id)`, type `Future<User>`. You want the user's orders, and `getOrders` is async — its type is `User -> Future<Orders>`. Now apply the two operators:

- With a plain `map`, the result type is `Future<map's-return>` = `Future<Future<Orders>>`. The outer future fulfills the moment `getOrders` is *called* (returning the inner future as its value) — but the inner future has not settled, so what you are holding is "a settled handle to an unsettled handle." If you `await` the outer one you get back *another future*, not the orders, and you have to await *again*. Every async step in the chain would add a layer, so a five-step chain becomes `Future<Future<Future<Future<Future<T>>>>>` — the exact tower we are trying to avoid, just hidden behind method calls instead of indentation.
- With `flatMap`, the runtime *adopts* the inner future: the outer result does not settle until the *inner* one does, and it settles with the inner's value, type `Future<Orders>`. The tower collapses to a single layer no matter how long the chain.

That adoption is the whole trick, and it is why Promises/A+ spends a paragraph on "if the value returned from `onFulfilled` is a thenable, adopt its state." Without flattening, futures would compose into nested garbage; with it, a hundred dependent async steps stay exactly one future deep.

## Composing with `then`, `map`, and `flatMap` across languages

Let me make the chaining idiom concrete in the three languages whose APIs diverge meaningfully, because the *shape* transfers but the *names* do not. The four operations you reach for over and over — chain a dependent step, combine all results, race for the first, handle an error — map to a different concrete method in each language, and figure 8 lines them up so you can translate one ecosystem's idiom into another's. (Python's `asyncio` has the same model with `await`; I will not re-derive it here — the [asyncio from the ground up post](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) owns that story.)

![A matrix mapping chain combine race and error operations to their concrete method names in JavaScript Promise Rust Future and Java CompletableFuture](/imgs/blogs/futures-promises-and-composing-asynchronous-code-8.png)

**Rust** is the most explicit because the future is cold and the types are visible. The idiomatic chain uses `.await` inside an `async fn`, which the compiler desugars into a state machine — but you can also compose combinators directly:

```rust
async fn load_dashboard(user_id: u64) -> Result<Dashboard, AppError> {
    let user = get_user(user_id).await?;          // ? propagates the error
    let orders = get_orders(user.id).await?;      // flatMap: each step is a future
    let items = get_items(orders[0].id).await?;
    Ok(Dashboard { user, orders, items })
}
```

The `?` operator is Rust's error propagation: if `get_user` returns `Err`, the function returns that error immediately, skipping the rest — the exact short-circuit a `.catch` does in JS, but as a language feature on `Result`. Each `.await` is a flatMap point: it drives the inner future to completion and unwraps its value before the next line runs.

**Java**'s `CompletableFuture` keeps `map` and `flatMap` as distinct methods, which makes the flattening explicit and is a common source of bugs when people pick the wrong one:

```java
CompletableFuture<Dashboard> loadDashboard(long userId) {
    return getUser(userId)
        .thenCompose(user -> getOrders(user.id()))   // flatMap: returns a future, flattened
        .thenApply(orders -> orders.get(0))           // map: plain transform
        .thenCompose(order -> getItems(order.id()))   // flatMap again
        .thenApply(items -> new Dashboard(items));    // map
}
```

The rule in Java is mechanical: if your lambda returns a `CompletableFuture`, you **must** use `thenCompose`, not `thenApply`. Use `thenApply` and you get a `CompletableFuture<CompletableFuture<List<Item>>>` — a nested future that never gets awaited, so the outer future "completes" with an inner future still pending, and downstream code sees a value of the wrong type. The compiler catches it (Java is statically typed), but the error message is a wall of nested generics that takes a minute to decode.

**JavaScript** is the loosest — `then` is both `map` and `flatMap`, and `async`/`await` is sugar over the chain:

```javascript
async function loadDashboard(userId) {
  const user = await getUser(userId);
  const orders = await getOrders(user.id);
  const items = await getItems(orders[0].id);
  return { user, orders, items };
}
```

Each `await` is a `.then` flatMap point. The function is paused at the `await`, the future is driven, and the function resumes with the unwrapped value when it settles. This is the *same code* as the `.then` chain above, rewritten by the engine into a state machine — which is exactly the coroutine mechanism the sibling [async/await post](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) takes apart. For composing a *single* dependent chain, `await` is almost always clearer than explicit `.then`; the combinators below are where the chaining methods earn their keep.

## Combinators: `all`, `race`, `any`, `allSettled`

Chaining handles *dependent* steps — B needs A's result. The far more interesting case is *independent* steps: you have N futures that do not depend on each other and you want to combine their outcomes. That is what **combinators** are for, and choosing the wrong one is how you silently drop failures or waste latency. Figure 3 is the cheat sheet; the four rows are the ones you will reach for constantly.

![A matrix comparing all race any and allSettled across what they wait for what they return and how they behave when one input fails](/imgs/blogs/futures-promises-and-composing-asynchronous-code-3.png)

**`all` — wait for all, fail fast.** Takes N futures, returns a future of N results, in input order. It fulfills when *every* input fulfills. The catch is the failure mode: if *any* input rejects, the combined future rejects immediately with that first error — it does **not** wait for the others. This is "all or nothing," and it is exactly right when you genuinely need every result (load the user *and* their settings *and* their permissions; if any fails, the page can't render).

```javascript
const [user, settings, perms] = await Promise.all([
  getUser(id),
  getSettings(id),
  getPermissions(id),
]);
// rejects as soon as any one of the three rejects
```

**`race` — first to *settle* wins, success or failure.** Returns a future that settles with the outcome of whichever input settles first — fulfilled *or* rejected. The classic use is a timeout: race the real work against a timer that rejects.

```javascript
const result = await Promise.race([
  fetchData(url),
  rejectAfter(5000, new Error("timeout")),
]);
// whichever settles first — the data, or the timeout error
```

The gotcha with `race`: because the *first to settle* wins, a fast rejection beats a slow success. If your "real" call fails in 10 ms and your timeout is 5000 ms, `race` rejects in 10 ms — usually what you want, but know it.

**`any` — first to *fulfill* wins, ignore failures until all fail.** Like `race`, but it only counts *fulfillments*. Rejections are tolerated; `any` keeps waiting for a success and only rejects (with an aggregate of all errors) if *every* input rejects. Use it for redundancy: query three mirror servers, take the first one that answers.

```javascript
const data = await Promise.any([
  fetch(mirror1),
  fetch(mirror2),
  fetch(mirror3),
]);
// first successful response; rejects only if all three fail
```

The `race` vs `any` distinction trips people up constantly: `race` settles on the first to *finish* (so one fast error sinks it), `any` settles on the first to *succeed* (so it shrugs off errors). Reach for `any` when you want resilience; `race` when you want a strict deadline.

**`allSettled` — wait for all, never reject.** This is the partial-failure workhorse. It waits for *every* input to settle (success or failure), and returns N *status records* — each tagged `fulfilled` with a value or `rejected` with a reason. It never rejects itself, because a failed input is data, not an exception.

```javascript
const results = await Promise.allSettled([
  sendEmail(a),
  sendEmail(b),
  sendEmail(c),
]);
const sent = results.filter((r) => r.status === "fulfilled").length;
const failed = results.filter((r) => r.status === "rejected");
// e.g. "2 of 3 emails sent, 1 failed" — you decide what to do
```

This is the one you want for fan-out where partial success is acceptable: send 100 notifications, and you want to know which 3 failed without the other 97 being thrown away. `all` would discard all the successes the moment the first one failed; `allSettled` keeps them.

These names are JavaScript's, but every mature ecosystem has the same four. Rust's `futures` crate has `join!`/`try_join!` (≈ `all`), `select!` (≈ `race`), `join_all`/`try_join_all`, and `FuturesUnordered` for streaming completion. Java's `CompletableFuture` has `allOf` (≈ `all`, though you collect results yourself), `anyOf` (≈ `race`), and you build `allSettled` by mapping each future's `.handle(...)` to a status. Go does not have futures at all — it expresses the same patterns with goroutines, a `sync.WaitGroup` (≈ `all`), and a `select` over channels (≈ `race`). Same four shapes, different syntax.

It is worth seeing the Go version, because the lack of a future type makes the underlying join logic visible — Go forces you to write the completion counter that `Promise.all` packages. The `all`-shaped pattern is a `WaitGroup` plus a slice the goroutines write into by index (each index is owned by exactly one goroutine, so no lock is needed):

```go
func fetchAll(ids []int) ([]Item, error) {
    items := make([]Item, len(ids))
    errs := make([]error, len(ids))
    var wg sync.WaitGroup
    for i, id := range ids {
        wg.Add(1)
        go func(i, id int) {        // each goroutine owns one slot
            defer wg.Done()
            items[i], errs[i] = fetchItem(id)
        }(i, id)
    }
    wg.Wait()                       // the "join": block until all Done
    return items, errors.Join(errs...) // collect outcomes (Go 1.20+)
}
```

`wg.Wait()` is the join — it blocks until every goroutine has called `Done`, exactly what `Promise.all` does. The Go `errgroup` package wraps this and adds fail-fast cancellation (when one goroutine errors, it cancels a shared `context` so the others can bail) — which is `Promise.all`'s fail-fast behavior plus the sibling-cancellation that bare `all` lacks. The point of showing Go here is that a "future" is not magic: it is a packaged completion counter with a happens-before edge, and when the language does not give you one, you write the counter by hand.

The `race`-shaped pattern in Go is a `select` over result channels, taking whichever sends first:

```go
func race(a, b func() Result) Result {
    ch := make(chan Result, 2)      // buffered so the loser doesn't leak
    go func() { ch <- a() }()
    go func() { ch <- b() }()
    return <-ch                     // first to send wins; the other's send is buffered
}
```

`<-ch` returns the first result produced — `Promise.race`. Note the buffered channel: without it the losing goroutine would block forever trying to send into a channel nobody reads, leaking a goroutine. That leak is the Go-flavored version of the "the loser kept running" problem that hot futures have, and it is a reminder that *racing* always raises the question of what happens to the work that lost.

| Combinator | Waits for | Returns | If one input fails |
| --- | --- | --- | --- |
| `all` / `join!` | every input to fulfill | all N values, in order | rejects fast with first error |
| `race` / `select!` | first input to settle | that one outcome | first failure wins too |
| `any` | first input to fulfill | first success value | tolerates failures, rejects only if all fail |
| `allSettled` | every input to settle | N status records | never rejects, failures are data |

## Sequential versus concurrent: the await-in-a-loop anti-pattern

Now the bug I promised — the one I flag in code review more than any other. You have a list of IDs and you want to fetch each one. The "obvious" loop is wrong:

```javascript
// ANTI-PATTERN: serializes N independent requests
async function fetchAll(ids) {
  const results = [];
  for (const id of ids) {
    results.push(await fetch(`/api/item/${id}`)); // awaits each before starting the next
  }
  return results;
}
```

This is *correct* but *slow*, and the slowness is structural, not incidental. The `await` inside the loop means: start request 1, **wait for it to finish**, then start request 2, wait, then request 3. The requests are independent — none of them needs another's result — but you have forced them into a strict sequence. If each call takes 100 ms and there are 10 IDs, this loop takes **1000 ms**. The requests could all have been in flight at the same time, finishing in roughly 100 ms total, but `await` paused the loop at each iteration.

The mechanism is worth stating precisely, because "async" misleads people into thinking it is automatically concurrent. **`await` suspends the current task until the awaited future settles.** It does *not* run anything in the background while you wait — it yields the thread to the event loop, which may run *other* tasks, but *this* function does not advance past the `await` until the future is done. So awaiting inside a loop means iteration `i+1` cannot even *begin* until iteration `i` has fully settled. The concurrency you wanted never started, because you never launched the work before awaiting it.

The fix is to **start all the futures first, then await the combined result**:

```javascript
// FIXED: launches all requests, then joins
async function fetchAll(ids) {
  const futures = ids.map((id) => fetch(`/api/item/${id}`)); // start all now
  return Promise.all(futures);                                // await the join
}
```

The difference is the `.map`: it calls `fetch` for every id *immediately*, producing an array of already-in-flight futures, *then* `Promise.all` waits for the set. Now all 10 requests overlap, and the wall-clock time is roughly the slowest single request, not the sum. Figure 4 contrasts the two timelines.

![A before and after figure contrasting awaiting each request in a loop which costs N times one latency against starting all requests then joining which costs about one latency](/imgs/blogs/futures-promises-and-composing-asynchronous-code-4.png)

The same trap exists in Rust, and the fix is the same shape. The anti-pattern:

```rust
// ANTI-PATTERN: each .await blocks the next iteration
let mut results = Vec::new();
for id in ids {
    results.push(fetch(id).await?);  // serialized
}
```

The fix uses `join_all` (or the `join!` macro for a fixed set) to drive them concurrently:

```rust
use futures::future::try_join_all;

// FIXED: build the futures, then drive them together
let futures = ids.into_iter().map(|id| fetch(id));
let results: Vec<_> = try_join_all(futures).await?;  // all in flight, join once
```

In Java the anti-pattern is calling `.get()` (or `.join()`) inside a loop, which blocks the thread on each future in turn; the fix is to collect the futures and combine with `allOf`. The lesson is language-independent and it is the single most important performance idea in this post: **`await`/`get` is a *wait*, not a *launch*. Launch first (build the futures), then wait once (join them).** Sequential composition is for *dependent* steps; concurrent composition is for *independent* ones, and confusing the two by awaiting in a loop is how you accidentally turn parallel work serial.

#### Worked example: where the latency actually goes

Say you fetch 10 items, each call 100 ms, over a connection that allows plenty of concurrent requests. Sequential `await`-in-loop: each call waits for the previous, so total $\approx 10 \times 100 = 1000$ ms. Concurrent `Promise.all`: all 10 start at $t=0$, all finish near $t=100$, total $\approx \max(100, 100, \ldots) = 100$ ms — a 10x speedup, for free, by moving one `await` outside the loop. The general rule: for N independent calls each taking $L$, sequential is $N \cdot L$ and concurrent is $\approx L + \text{(scheduling overhead)}$. The speedup ceiling is N, capped in practice by how many requests the server, the connection pool, or the OS will let you run at once — which is why "fan out to all N at once" is sometimes *too* aggressive and you bound it with a concurrency limit (more on that in the worked example near the end).

## Error propagation and the dropped-error trap

Composition is only half the story; the other half is what happens when a step *fails*. The good news is that futures give you a clean error story: a rejection **short-circuits** the chain. When a future rejects, every downstream `then`/`map` is skipped and control jumps straight to the first error handler (`catch`, `?`, `exceptionally`). It works just like a thrown exception in synchronous code, but threaded through asynchronous steps. Figure 7 shows the short-circuit: a rejection in the middle skips the success steps and lands at the catch.

![A timeline showing a rejected future skipping the downstream map steps and jumping straight to the catch handler](/imgs/blogs/futures-promises-and-composing-asynchronous-code-7.png)

That is the mechanism. The *trap* is that this short-circuit only works if there **is** a handler — and async errors are dropped silently in ways synchronous errors never are. Here is the classic dropped-error bug:

```javascript
// BUG: the error is silently lost
function notifyUser(id) {
  sendEmail(id); // returns a promise, but we never await it or attach .catch
}
notifyUser(42);
// if sendEmail rejects, nobody handles it — an unhandled rejection
```

`notifyUser` calls `sendEmail`, which returns a promise, and then *throws that promise away* — it does not `await` it, does not `return` it, does not attach `.catch`. If `sendEmail` rejects, the rejection has nowhere to go. In Node.js this produces an `unhandledRejection` warning (and, since Node 15, *crashes the process* by default); in a browser it logs to the console and is otherwise lost. Either way, your caller has no idea the email failed. This is the **floating promise** or fire-and-forget bug, and it is endemic because the code *looks* fine — it compiles, it runs, it just quietly loses failures.

The fix is to never let a promise float: `await` it, `return` it, or attach a handler.

```javascript
// FIXED: propagate the error to the caller
async function notifyUser(id) {
  await sendEmail(id); // now a rejection propagates out of notifyUser
}
// or, if you truly want fire-and-forget, handle it explicitly:
sendEmail(id).catch((err) => logger.error("email failed", err));
```

The same trap exists in every language, and the better-typed ones lean on the compiler to stop it. **Rust** makes the floating-future bug nearly impossible because futures are *cold* — a future that is never `.await`ed does literally *nothing* (we will get to why in the next section), and the compiler emits a `#[must_use]` warning: "unused `Future` that must be used: futures do nothing unless you `.await` or poll them." Combined with `Result` and `?`, Rust forces you to either propagate the error or explicitly discard it:

```rust
// the compiler warns if you ignore the Result or the Future
async fn notify_user(id: u64) -> Result<(), AppError> {
    send_email(id).await?;   // ? propagates; dropping the Result warns
    Ok(())
}
```

**Java** sits in between. `CompletableFuture` propagates exceptions through the chain (they surface in `exceptionally` or `handle`, or get wrapped in a `CompletionException` when you call `.join()`), but nothing forces you to attach a handler — a `CompletableFuture` whose exceptional completion is never inspected is the JVM's version of the floating promise. The discipline is to always terminate a chain with `.exceptionally(...)` or `.handle(...)`, or to `.join()` it somewhere that propagates.

| Language | Error mechanism | Floating-error protection |
| --- | --- | --- |
| JavaScript | rejection short-circuits to `.catch` / `try-catch` | runtime `unhandledRejection`, crashes by default in modern Node |
| Rust | `Result` + `?` short-circuits | compile-time `#[must_use]` on unused futures; very hard to drop |
| Java | exception propagates to `exceptionally` / `handle` | none at compile time; discipline only |
| Go | explicit error return per goroutine | none; `errgroup` helps collect them |

There is a second, sneakier error trap specific to combinators: **`Promise.all` hides partial failures.** If you fan out 100 writes with `all` and write #37 fails, `all` rejects with #37's error the instant it happens — and you lose track of which of the other 99 succeeded (they may still be in flight, their results discarded). For idempotent reads that is fine; for non-idempotent writes it is a correctness bug, because some writes committed and you no longer know which. The fix is `allSettled`: collect every outcome, then decide. This is the partial-failure handling the worked example at the end builds out in full.

A third trap is **error context loss across the await boundary.** In synchronous code a thrown exception carries a stack trace that points at the line that threw. Across an `await`, the call stack that *registered* the continuation is gone by the time the continuation runs — the function suspended, the stack unwound, and the engine resumed you later on a fresh stack. Naively, the trace you get points at the event loop's internals, not at *your* call site, which makes async bugs maddening to debug. Modern runtimes fight this with **async stack traces**: V8 stitches together the logical chain of `await`s so the trace reads like the synchronous one would, and Rust's `?` with a good error type (`anyhow`, `eyre`) attaches context at each propagation point. The practical rule: when you write a custom error-handling combinator, *preserve the cause* — wrap, do not replace. A `catch` that does `throw new Error("request failed")` and drops the original is throwing away the one piece of information that would have told you *why*. Always chain the cause (`{ cause: err }` in modern JS, `.context(...)` in Rust, the two-arg `Throwable` constructor in Java) so the trace survives the async hop.

#### Worked example: the swallowed-rejection bug, traced

Here is the bug in a form I have actually debugged at 3 AM. A request handler kicks off an audit-log write but does not await it, because "logging shouldn't block the response":

```javascript
app.post("/transfer", async (req, res) => {
  const result = await doTransfer(req.body);   // awaited: errors propagate
  auditLog(result);                            // NOT awaited: returns a floating promise
  res.json({ ok: true });
});
```

The transfer succeeds, the response goes out, everyone is happy — until `auditLog` starts failing silently because its database is down. The floating promise rejects, nobody is listening, and in older Node the rejection is logged and forgotten; the response already said `ok: true`, so no alert fires. You discover weeks later that audit records have gaps, with no error in any log to point at when it started. The fix is to decide *explicitly* what a logging failure means: either it is fire-and-forget and you attach a `.catch` that records the failure somewhere durable, or it matters and you `await` it. What you must not do is let the promise float — the silence is the bug.

```javascript
// explicit fire-and-forget: the failure is handled, just not awaited
auditLog(result).catch((err) => metrics.increment("audit.write.failed", { err }));
```

The difference between the broken and fixed versions is one `.catch`. That is the entire floating-promise lesson: a future's error has to go *somewhere*, and "nowhere" is a choice the language will let you make silently unless you are using a runtime or a linter that makes it loud.

## Cold versus hot futures, and why it matters

This is the deepest conceptual difference between ecosystems, and it explains a dozen surprising behaviors. **When does a future start doing its work?** There are two answers, and they are opposites.

A **hot** (eager) future starts its work the moment it is *created*. JavaScript promises are hot: the instant you call `fetch(url)`, the request is on the wire — before you ever call `.then` or `await`. The promise is a *handle to work already in progress*. A **cold** (lazy) future does *nothing* until something *drives* it — until it is `.await`ed or polled. Rust futures are cold: `let f = fetch(url);` builds a future object that has not sent a single byte; the work only begins when you `f.await` it (or hand it to an executor that polls it).

This is not a trivia distinction; it changes how you write code. Figure 6 lays out the consequences side by side.

![A matrix comparing cold Rust futures against hot JavaScript promises across when they start what drives them and whether dropping them cancels the work](/imgs/blogs/futures-promises-and-composing-asynchronous-code-6.png)

The mechanism behind cold futures is **poll-driven** execution. A Rust `Future` is a state machine implementing one method, shown simplified:

```rust
trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output>;
}
// poll returns Poll::Pending (not ready, will wake later) or Poll::Ready(value)
```

An **executor** (Tokio, async-std) repeatedly calls `poll`. Each `poll` either returns `Ready(value)` — done — or `Pending`, meaning "not yet; I have registered a waker, call me again when there's progress." The future is a passive object; *it does not run itself*. Nothing happens until the executor polls it, which only happens after you `.await` it (or `spawn` it). That is what "cold" means at the metal: no driver, no work.

JavaScript's hot model is different at the metal too. When you call `fetch`, the engine immediately kicks off the I/O and hands you a promise. The event loop will settle that promise when the I/O completes, *whether or not anyone is listening*. The promise runs itself, eagerly, driven by the event loop rather than by your `await`.

Three practical consequences fall out of cold vs hot, and they are exactly where people get burned:

**1. Cancellation by dropping.** A cold Rust future cancels for free: if you `drop` it (let it go out of scope) before it completes, its work simply *stops* — there is no executor polling it anymore, so the state machine never advances. This is why Rust's `select!` and timeouts compose so cleanly: the loser of a race is dropped, and dropping *is* cancellation. A hot JS promise cannot be cancelled by dropping it — the `fetch` is already in flight, and forgetting the promise does **not** stop the request. (You need an explicit `AbortController` to cancel it.) That asymmetry is the single biggest source of "wait, the request still fired?" confusion when JS programmers move to Rust or vice versa.

**2. Eager side effects.** Because JS promises are hot, this surprises people:

```javascript
const p = chargeCard(amount); // the charge ALREADY HAPPENED, right here
if (shouldCharge) {
  await p; // awaiting later doesn't make the charge conditional — too late
}
```

The card was charged the instant `chargeCard` was *called*, not when it was awaited. In Rust the equivalent `let p = charge_card(amount);` charges *nothing* until `p.await` — so the `if` would actually gate the charge. Hot futures make "create" and "start" the same event; cold futures separate them, which is usually what you want for conditional or cancellable work.

**3. Re-await and memoization.** A JS promise is settled once and *caches* its result — awaiting the same promise twice gives the same value instantly the second time (it is a value, not a recipe). A cold Rust future is a one-shot recipe — you cannot `.await` the same future twice (it is consumed), and there is no caching; if you want to reuse a result you store the *value*, not the future. Hot = a settled value you can re-read; cold = a recipe you run once.

| Property | Cold future (Rust) | Hot future (JS Promise) |
| --- | --- | --- |
| Starts work | on first poll / `.await` | at creation |
| Driven by | an executor polling it | the event loop |
| Drop = cancel | yes, dropping stops it | no, work keeps running |
| Re-await | consumed, run once | cached, re-read freely |
| Conditional start | natural (gate the `.await`) | too late (already started) |

Neither model is "better" — they are different trade-offs. Hot is convenient for "just do this now," cold is precise for "describe the work, then decide when and whether to run it." Java's `CompletableFuture` is hot like JS (the work is usually already submitted to an executor); Kotlin coroutines have *both* (`launch` is hot, a `suspend fun` you call is effectively cold until invoked). Knowing which one you are holding tells you whether dropping it cancels, whether creating it has side effects, and whether you can await it twice.

Kotlin is the instructive middle case because the language makes the choice *visible at the call site*. A `suspend fun` does nothing until you call it from a coroutine — that is cold, like a Rust future you have not awaited. But `async { ... }` returns a `Deferred<T>` that starts running *immediately* on the dispatcher — that is hot, like a JS promise. And `launch { ... }` starts a hot fire-and-forget coroutine. So in one language you pick coldness or hotness per call, and the API names tell you which: `coroutineScope { val a = async { fetchA() }; val b = async { fetchB() }; a.await() + b.await() }` fans out two hot coroutines and joins them — and because they are launched inside a `coroutineScope`, they are *structured*: if one throws, the scope cancels the other and the whole block fails as a unit. Kotlin took the cold-vs-hot lesson and the structured-concurrency lesson and baked both into the standard library, which is why its async story is often held up as the cleanest of the mainstream languages. The takeaway for picking *any* async API: find out, before you write a line, whether the thing you are holding is cold or hot, because that single fact determines whether `if (cond) { await(f) }` actually gates the work or merely gates the *waiting* for work that already started.

## Cancellation of a future, and structured fan-out

Cold-vs-hot leads straight into **cancellation**, which is where futures get genuinely hard and where this post hands off to a sibling. The question is simple to ask: you started a future, and now you do not want its result anymore — how do you stop it?

For cold futures, cancellation is *structural*: drop the future and it stops, because nothing polls it. Rust's timeout is exactly this — `tokio::time::timeout(dur, fut)` races your future against a timer, and if the timer wins, your future is dropped mid-poll, unwinding its state. The catch is that cancellation happens *at an await point*: a cold future can only be cancelled when it is suspended (returns `Pending`), so synchronous work between await points runs to completion. You cannot interrupt a tight CPU loop with no `.await` in it.

For hot futures, cancellation is *cooperative and explicit*. A JS `fetch` keeps going unless you pass an `AbortSignal` and call `controller.abort()`; the underlying operation has to *check* the signal and bail out. Java's `CompletableFuture.cancel(...)` is famously weak — it does not actually interrupt the running computation, it just completes the future exceptionally so *downstream* steps see a `CancellationException`; the upstream work keeps running. Go's idiom is the cleanest of the hot-ish models: pass a `context.Context`, and the goroutine `select`s on `ctx.Done()` to bail.

The reason cancellation is so fiddly is that a bare future has no notion of *who owns it* or *when its lifetime ends*. If you fan out 10 futures and one fails, what cancels the other 9? With raw `Promise.all`, nothing — they keep running, their results discarded, wasting work and possibly committing side effects you no longer want. The disciplined answer is **structured concurrency**: bind a set of futures to a *scope* so that (a) the scope does not exit until all its children settle, and (b) if one child fails, the scope cancels its siblings. That turns fan-out into a single, cancellable unit with a clear lifetime — exactly the model the [structured concurrency post](/blog/software-development/concurrency/structured-concurrency-cancellation-and-thread-pool-design) develops in full, and the reason raw `Promise.all` is a leaky abstraction for anything with side effects. The related question of *which functions can even be async* — why a sync function cannot transparently call an async one — is the function-coloring problem, taken up in the [bridging sync and async post](/blog/software-development/concurrency/function-coloring-and-bridging-sync-and-async).

A concrete example of why the missing-lifetime problem bites: cancellation safety in Rust's `select!`. When you race two futures with `select!`, the *loser* is dropped the moment the winner finishes. If the loser was, say, halfway through reading from a socket — it had pulled some bytes off the wire into a temporary buffer but not yet returned them — dropping it *loses those bytes*. The next read starts from a torn position. A future is "cancellation-safe" only if dropping it at any await point leaves no half-finished mutation behind; many are *not*, which is why Rust's docs annotate which combinators are safe to use inside `select!`. Hot futures dodge this particular issue (you cannot drop-cancel them, so there is no mid-poll drop) but pay for it elsewhere — the loser of a JS `race` keeps running and may *complete* its side effect after you have already moved on, which is the same bug from the other direction. Either way, racing and fanning out *demand* an answer to "what happens to the work that did not win," and a bare future does not give you one. The scope does.

The takeaway for *this* post: cancellation is a property of how the future is *driven*, not of the future itself. Cold futures cancel by dropping (cheap, structural); hot futures need an explicit signal threaded through (cooperative). And bare combinators like `all` do *not* give you sibling-cancellation for free — you need a scope for that.

## Worked example: fan out N requests, await all, handle partial failure

Let me build the canonical pattern end to end, because it ties together fan-out, combinators, error propagation, and partial failure — and it is a pattern you will write hundreds of times. The job: given a list of N item IDs, fetch all of them concurrently, and produce a result that distinguishes the successes from the failures instead of letting one bad ID sink the whole batch. Figure 5 is the dataflow — one request fans out to N futures, then a single join merges them.

![A graph showing one request fanning out to three concurrent futures that each finish at different times and then merging at a single join into one combined result](/imgs/blogs/futures-promises-and-composing-asynchronous-code-5.png)

Here is the naive version with `Promise.all`, and why it is wrong for this job:

```javascript
// FRAGILE: one bad ID rejects the whole batch, losing the good results
async function fetchItems(ids) {
  const items = await Promise.all(ids.map((id) => fetchItem(id)));
  return items; // if any fetchItem rejects, this throws and we lose all of them
}
```

If item #37 fails, `Promise.all` rejects with #37's error, the function throws, and the 99 items that *did* load are gone — discarded along with the rejection. For a read where you can just retry the whole thing, fine. For "fetch what you can and show the rest," it is wrong. Switch to `allSettled`:

```javascript
// ROBUST: fan out, wait for all, partition into ok and failed
async function fetchItems(ids) {
  const results = await Promise.allSettled(ids.map((id) => fetchItem(id)));
  const ok = [];
  const failed = [];
  results.forEach((r, i) => {
    if (r.status === "fulfilled") ok.push(r.value);
    else failed.push({ id: ids[i], error: r.reason });
  });
  return { ok, failed }; // e.g. { ok: 97 items, failed: [{id, error}, ...] }
}
```

Now all N fetches run concurrently (the `.map` starts them all), the function waits for every one to settle, and you get back a clean partition: the items that loaded, and the IDs that did not (with their errors) so you can retry just those or render a partial page. No good result is ever thrown away.

In production you usually need one more thing: **bounded concurrency.** Firing 10,000 fetches at once will exhaust the connection pool, trip the server's [rate limits](/blog/software-development/system-design/rate-limiting-and-backpressure), or run you out of file descriptors. So you cap the in-flight count — a semaphore, a worker pool, or a batching helper. A simple bounded fan-out:

```javascript
async function fetchItemsBounded(ids, limit = 20) {
  const ok = [], failed = [];
  let cursor = 0;
  async function worker() {
    while (cursor < ids.length) {
      const i = cursor++;            // claim the next index
      const id = ids[i];
      try { ok.push(await fetchItem(id)); }
      catch (err) { failed.push({ id, error: err }); }
    }
  }
  // run `limit` workers concurrently; each pulls work until the list is drained
  await Promise.all(Array.from({ length: limit }, worker));
  return { ok, failed };
}
```

This launches exactly `limit` concurrent workers, each looping until the shared cursor drains the list, with per-item `try/catch` so one failure never sinks the batch. It is the same fan-out/fan-in shape, throttled. (The deeper version of "don't overwhelm the downstream" is backpressure, which the [message-queue backpressure post](/blog/software-development/message-queue/backpressure-and-flow-control) covers at the system level.)

The Rust equivalent uses `try_join_all` for the fail-fast variant or `join_all` for the collect-everything variant, and `buffer_unordered` on a stream for bounded concurrency:

```rust
use futures::stream::{self, StreamExt};

// bounded fan-out: at most 20 fetches in flight, collect all outcomes
async fn fetch_items(ids: Vec<u64>) -> (Vec<Item>, Vec<(u64, AppError)>) {
    let results: Vec<_> = stream::iter(ids)
        .map(|id| async move { (id, fetch_item(id).await) })
        .buffer_unordered(20)        // cap concurrency at 20
        .collect()
        .await;
    let mut ok = Vec::new();
    let mut failed = Vec::new();
    for (id, res) in results {
        match res {
            Ok(item) => ok.push(item),
            Err(e) => failed.push((id, e)),
        }
    }
    (ok, failed)
}
```

`buffer_unordered(20)` is the Rust idiom for "run up to 20 of these futures concurrently, yield results as they finish" — bounded fan-out and partial-failure handling in four lines. The shape is identical to the JS worker pool: start the work, cap the in-flight count, collect every outcome (ok and failed) rather than letting one error abort the set.

## Measured: sequential versus concurrent latency

Now the numbers, because "concurrent is faster" is only true when the work is actually independent and I/O-bound, and the gap is worth measuring rather than asserting. The setup that matters: N independent calls, each with latency $L$ dominated by waiting (network, disk), running on a runtime that lets them overlap.

How to measure honestly: warm up first (JIT, connection pools, DNS cache — the first call is always an outlier), run the batch many times, report a median not a single run, and name the confound. The big confound here is that *concurrent* does not mean *unbounded* — the server, the connection pool, and the OS all cap how many requests truly overlap, so above some concurrency the curve flattens and you stop getting speedup. Measure the real ceiling, do not assume it is N.

Here is a representative shape for N = 10 independent calls at $L \approx 100$ ms each (these are order-of-magnitude figures from the model, not a single benchmarked machine — the *ratios* are the point, and they hold across runtimes):

| Strategy | Wall-clock (10 calls, ~100 ms each) | Speedup | Why |
| --- | --- | --- | --- |
| Sequential (`await` in loop) | ~1000 ms | 1x baseline | each call waits for the previous |
| Concurrent (`Promise.all`, unbounded) | ~110 ms | ~9x | all overlap, total ≈ slowest + overhead |
| Bounded (limit = 4) | ~280 ms | ~3.5x | 3 batches of ~100 ms run in sequence |
| Bounded (limit = 10) | ~110 ms | ~9x | all 10 fit under the limit, fully overlapped |

Three honest observations. First, the speedup ceiling is N (here 10x) but you rarely hit it — scheduling overhead, the event loop, and connection setup eat the last bit, so ~9x is the realistic best. Second, *bounded* concurrency trades some speedup for safety: a limit of 4 runs in roughly $\lceil 10/4 \rceil = 3$ batches, so ~3x, and that is often the *right* trade because unbounded fan-out can knock over the thing you are calling. Third — and this is the one people forget — **none of this speedup applies to CPU-bound work.** Concurrency overlaps *waiting*. If each "call" is a tight compute loop with no I/O, running 10 of them concurrently on one thread is *not* faster (it may be slightly slower from scheduling overhead); you need actual *parallelism* — multiple cores — which is a different mechanism entirely (the [concurrency vs parallelism post](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws) draws that line). Futures buy you concurrency for I/O-bound work; they do not magically parallelize computation.

There is a fourth subtlety that catches people benchmarking this for the first time: the latency model $N \cdot L$ for sequential and $\approx L$ for concurrent assumes the calls are *truly independent at the resource level*. If all N requests hit the same single-threaded database connection, the server serializes them anyway, and your "concurrent" client sees them complete one after another — the concurrency on the client does not create concurrency on the server. The same is true of a connection pool of size 4: you get at most 4-way overlap no matter how many futures you launch, so the effective ceiling is $\min(N, \text{pool size})$, and beyond the pool size the extra futures queue. This is why I always say *measure the real ceiling* rather than assuming N: the limiting resource is usually downstream of your client, and the right concurrency limit is the one that saturates that resource without overwhelming it — which is exactly the backpressure question the system grows into. The honest benchmark therefore reports not just "9x faster" but "9x up to the pool's 4-connection ceiling, then flat," because that second clause is what tells you where to tune.

#### Worked example: when concurrency does *not* help

You have 8 image-resize operations, each 50 ms of pure CPU, on a single-threaded event loop. Naively you `Promise.all` them expecting ~50 ms. You get ~400 ms — the same as sequential — because the event loop has one thread and CPU work cannot overlap on one thread; each resize runs to completion before the next starts. The `await` points never yield during a tight compute loop, so "concurrent" degenerates to "sequential" for CPU work. The fix is *parallelism*: worker threads (Node `worker_threads`, a thread pool, Rust's `rayon`, Java's `ForkJoinPool`) that run on multiple cores. The rule of thumb that falls out: **futures/async for I/O-bound concurrency; threads/cores for CPU-bound parallelism.** Reaching for the wrong one is why "I made it async and it didn't get faster" is such a common complaint.

## Case studies / real-world

**The Promises/A+ specification (2012–2013).** Early JavaScript had a zoo of incompatible "thenable" libraries (jQuery's Deferred, Q, when.js, Bluebird) that disagreed on the details — did `.then` run synchronously or async, what happened if you returned a thenable, how did errors propagate? The [Promises/A+ spec](https://promisesaplus.com/) nailed down the contract: `then` must return a new promise, callbacks must run *asynchronously* (never synchronously, to avoid the "sometimes-sync-sometimes-async" Zalgo bug Isaac Schlueter wrote about), a thenable returned from `then` must be *adopted* (the flattening / `flatMap` behavior), and rejections propagate to the nearest `onRejected`. Native `Promise` in ES2015 implemented A+, which is why `.then` chaining behaves identically across every modern JS runtime today. It is a small spec with an outsized payoff: it made composition *predictable*, which is the whole reason `await` could be built on top of it.

**Rust's decision to make futures lazy (2016–2019).** When Rust designed `async`/`await`, the team deliberately chose *cold*, poll-driven futures over the eager model JavaScript and C# use. The reasoning, documented in the async working group's posts and Aaron Turon's "Zero-cost futures in Rust," was that lazy futures compile to a state machine with no heap allocation and no runtime scheduler baked in — you get to choose the executor (Tokio, async-std, embedded) — and, critically, that **cancellation becomes dropping**: a future you stop polling stops running, with no special cancellation protocol. The cost is the `#[must_use]` footgun (a future you forget to `.await` does nothing, which surprises newcomers) and "cancellation safety" subtleties (a future dropped mid-`select!` must leave its state consistent). It is a clean example of a deliberate cold-vs-hot trade: Rust paid newcomer confusion for zero-cost, executor-agnostic, drop-cancellable futures.

**A real callback-hell refactor: Node.js and the move to promises (2015–2018).** The Node.js ecosystem lived in error-first callbacks for years, and the pyramid of doom was a constant complaint. The escape was incremental: `util.promisify` (Node 8, 2017) wrapped callback APIs into promise-returning ones, and `fs.promises` / the `node:` promise APIs gave first-class promise versions of the standard library. Teams refactored deeply nested callback chains into flat `async`/`await` and consistently reported the same two wins — fewer lines and *centralized* error handling (one `try/catch` replacing N scattered `if (err) return`), which is exactly the flattening this post is about. The migration also surfaced the floating-promise bug at scale, which is why modern Node *crashes* on unhandled rejection by default and why linters ship a `no-floating-promises` rule: the very mistake the callback era hid silently is now loud.

## When to reach for this (and when not to)

Futures are the right tool for **composing asynchronous values** — a single result that arrives later, and the chaining, combining, and racing of several such results. Reach for them, and avoid them, by this checklist:

**Reach for futures/async when:**

- The work is **I/O-bound** — network, disk, database, RPC — and you want to overlap the *waiting*. This is the home-run case: fan out N calls, `all`/`allSettled` them, get an ~Nx latency win for free.
- You have a **dependent chain** of async steps (A then B then C) and want it to read like sequential code. `await` flattens the pyramid; one `try/catch` covers the chain.
- You need **first-to-finish** semantics — a timeout (`race`), redundant requests (`any`), or a deadline. Combinators express these in one line.
- You want **partial-failure** handling over a batch — `allSettled` keeps the successes when some inputs fail.

**Do not reach for futures (or know the caveat) when:**

- The work is **CPU-bound.** Async overlaps waiting, not computation. Ten compute loops on one thread are not faster concurrent; you need *parallelism* (threads, cores, a worker pool). "I made it async and it didn't speed up" is almost always this.
- You are **awaiting in a loop** for independent work. That serializes it. Build the futures first, then `all` them. If you genuinely need them sequential (each depends on the last), the loop is fine — but be sure that is the case.
- You need **streaming** of many values over time, not one value-that-arrives-later. A future settles once; a *stream* (async iterator, channel, observable) yields repeatedly. Use a channel or async iterator — and when communication, not a single result, is the point, [channels](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) and message passing fit better than futures.
- You need **strict cancellation and lifetime guarantees** over a fan-out with side effects. Bare `Promise.all` does not cancel siblings on failure and does not bound a lifetime; reach for structured concurrency (a scope) instead.
- The whole program is **simple and synchronous.** Do not async-ify a script that makes two sequential calls and finishes — you add coloring, error-handling ceremony, and a runtime for no latency benefit. Measure first.

The decision in one line: **futures for asynchronous *values* and the I/O concurrency that overlaps their waiting; threads/cores for CPU *parallelism*; channels/streams for ongoing *communication*; a scope for *cancellable lifetimes*.** Most async bugs come from using a future where one of the other three was the right tool.

## Key takeaways

1. **A future is a write-once handle to a not-yet value.** It is pending, then settles exactly once to fulfilled or rejected, then is immutable. The *promise* is the writer side (completes it once); the *future* is the reader side (awaits it). JavaScript fuses both into `Promise`; most other languages keep them distinct.
2. **`then`/`flatMap` flattens continuation-passing.** Each `then` returns a *new* future, so chaining replaces nesting — the callback pyramid becomes a flat list — and if a step returns a future, `then` flattens it (`flatMap`/`thenCompose`/`and_then`), keeping chains from nesting into futures-of-futures.
3. **Pick the combinator deliberately.** `all` = wait for all, fail fast on the first error; `race` = first to settle wins (a fast error sinks it); `any` = first to *succeed* wins, tolerates failures; `allSettled` = wait for all, never reject, failures are data. The wrong choice silently drops failures or wastes latency.
4. **`await` is a *wait*, not a *launch*.** Awaiting inside a loop serializes independent work to N times one latency. Build the futures first, then `all` them once — that is the ~Nx win, and forgetting it is the most common async perf bug.
5. **Errors short-circuit to the first handler — if one exists.** A rejection skips downstream steps and lands at `catch`/`?`/`exceptionally`. A *floating* promise (never awaited, never `.catch`ed) drops the error silently; modern runtimes crash on it, Rust's `#[must_use]` warns at compile time.
6. **Cold vs hot decides start, cancel, and re-await.** Cold futures (Rust) start only when polled and cancel by dropping; hot futures (JS) start at creation and keep running if dropped. Know which you hold: it tells you whether creating it has side effects and whether dropping it cancels.
7. **Bound your fan-out.** Unbounded `all` over thousands of inputs exhausts pools and trips rate limits. Cap concurrency with a semaphore, worker pool, or `buffer_unordered`, and use `allSettled` so one failure never discards the batch.
8. **Concurrency overlaps waiting, not computation.** Futures speed up I/O-bound work; CPU-bound work needs real parallelism (threads/cores). Reaching for the wrong one is why "async didn't make it faster" happens.

## Further reading

- [Promises/A+ specification](https://promisesaplus.com/) — the small, precise contract that made JavaScript promise composition predictable (asynchronous callbacks, thenable adoption, error propagation).
- [MDN: Using Promises](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises) — the canonical reference for `then`/`catch`/`finally`, `Promise.all`/`race`/`any`/`allSettled`, and the common composition mistakes.
- *Rust Async Book* (the `async`/`await` chapter) — why Rust futures are cold and poll-driven, the `Future` trait, executors, and cancellation by dropping.
- *Java Concurrency in Practice*, Brian Goetz — the futures/task chapters for the JVM model that `CompletableFuture` later built on.
- Aaron Turon, "Zero-cost futures in Rust" — the design rationale for lazy, state-machine futures and what that buys (no allocation, executor choice, drop-cancellation).
- [Concurrency & Parallelism, From the Ground Up — series intro](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) — the shared-state and happens-before frame this post sits inside.
- [async/await and how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work) — the state-machine transform that turns the `.then` chains here into linear `await` code.
- [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) — the Python event-loop-and-coroutine story this post deliberately links out to rather than re-deriving.

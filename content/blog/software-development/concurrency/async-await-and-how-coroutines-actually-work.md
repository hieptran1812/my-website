---
title: "Async/Await and How Coroutines Actually Work"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Async/await is not a thread you spawned and forgot — it is your function rewritten into a resumable state machine, and once you can see that machine you can finally reason about what suspends, what resumes, and what starves the loop."
tags:
  [
    "concurrency",
    "parallelism",
    "async-await",
    "coroutines",
    "futures",
    "event-loop",
    "state-machine",
    "cooperative-scheduling",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/async-await-and-how-coroutines-actually-work-1.png"
---

The first time I shipped an async service, it lied to me. The code read top to bottom like ordinary blocking code — `let header = read(socket).await; let body = read(socket).await; write(socket, reply).await;` — and for a while I believed the mental picture that matched the syntax: a worker that sits and waits at each `.await`, the way a thread sits and waits inside a blocking `read()`. Then one afternoon a single request handler did a little too much CPU work between two awaits, and the entire service's tail latency exploded. Not that one request — *every* request. Thousands of unrelated connections, all serviced by what I had assumed were independent "workers," froze in lockstep for 80 milliseconds. There were no extra threads. There was one thread, and one request had been hogging it.

That incident is the whole lesson of this post in miniature. `await` *looks* like blocking, but it is the opposite of blocking. When a thread blocks inside `read()`, the OS parks the thread and runs something else on the core. When a coroutine hits `.await` and the data is not ready, the **function** suspends — it saves its place and its local variables and hands the thread back — and that same thread immediately runs other work. The thread never blocks. So if your function refuses to ever reach an `.await` — because it is grinding through a CPU loop — it never hands the thread back, and every other task on that thread starves. The syntax hid a machine, and the machine is what you actually have to reason about.

![Sequential await source on the left compiles into a state machine on the right with explicit suspension points and saved locals](/imgs/blogs/async-await-and-how-coroutines-actually-work-1.png)

This post is about that machine. We are going to demystify `async`/`await` completely: what a **coroutine** is (a function that can suspend partway through and resume later, keeping its locals), the two ways to build one (**stackful**, which owns a real stack and can suspend anywhere, versus **stackless**, which the compiler rewrites into a **state machine** that can suspend only at `await` points), how exactly an `async fn` *desugars* into that state machine, who *drives* it (the **executor** / event loop, by repeatedly calling `poll` or `resume`), and why **cooperative scheduling** means a non-awaiting coroutine is a loaded gun pointed at your latency. We will hand-desugar an `async fn` into its state machine — by hand, field by field — and we will measure why a million async tasks fit in memory where a million threads do not. By the end you should be able to look at any `async` function and *see the states*. This sits squarely on the series' spine: **shared mutable state plus nondeterministic scheduling is the hazard** — async swaps preemptive nondeterminism for *cooperative* scheduling, which is easier to reason about precisely because suspension happens only at points you can see. It builds on [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) and feeds the [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) at the end of the series.

## What a coroutine actually is: suspend and resume

Start with an ordinary function. It has one entry point (the top), it runs straight through, and when it returns it is *gone* — its stack frame is popped, its locals are destroyed, and there is no "later" for it. You can call it again, but that is a fresh invocation from the top with fresh locals. A normal function is a one-shot.

A **coroutine** removes the one-shot restriction. A coroutine is a function that can **suspend** in the middle — stop executing, but *not* be destroyed — and later **resume** from exactly where it stopped, with all of its local variables intact. It has, in effect, more than one entry point: the top, and every point where it previously suspended. Where a normal function call is "run to completion," a coroutine call is "run until the next suspension point, then return control to whoever called you; they may resume you later."

The defining property is **state preservation across suspension**. Consider a coroutine with three locals and two suspension points:

```python
def counter():             # generator-style coroutine (illustration only)
    total = 0              # local state
    while True:
        n = yield total    # SUSPEND here; resume with the value sent in
        total += n         # local 'total' survived the suspension
```

Each `yield` is a suspension point. When the coroutine suspends at `yield`, `total` does not vanish the way a normal function's locals vanish at return — it is preserved, sitting somewhere, ready for the resume. The resume jumps *back into the middle of the loop*, not to the top. That is the entire trick: a coroutine is a function whose execution state (instruction position plus locals) survives between calls. Python's generators and `asyncio` coroutines are exactly this; this post stays language-agnostic and uses Python only as the cleanest first illustration, so for the full Python story see [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) — it owns that territory.

There are two flavors of coroutine, and the difference between them is the most consequential design decision in this entire area:

- **Symmetric vs asymmetric.** Asymmetric coroutines have a clear caller/callee relationship: the coroutine suspends back *to its resumer* (like `yield` returning to whoever called `next()`). Symmetric coroutines can transfer directly to *any* other coroutine. Async/await is asymmetric — you suspend back to the executor, never directly to a sibling task.
- **Stackful vs stackless.** This is the big one. Does the coroutine carry its own *call stack* so it can suspend from arbitrarily deep inside nested function calls? Or is it a flat object that can suspend only at the syntactic `await` points the compiler could see? That distinction decides everything about cost, ergonomics, and which language you are using. We take it head-on next.

Either way, the contract is the same and worth stating precisely, because the rest of the post is just consequences of it: **a coroutine is a value you can call repeatedly; each call advances it from where it last suspended to the next suspension point, preserving its locals; eventually it completes and yields a final result.** Hold that sentence. Everything — futures, executors, the state machine — is a way to implement it.

It is worth being honest about how *old* this idea is, because async/await is often presented as a recent invention when in fact the coroutine predates the subroutine-only mindset by decades. Melvin Conway described coroutines in 1963 in the context of a one-pass compiler, precisely so a lexer and a parser could each be written as a straight-line loop while alternately handing control back and forth instead of one driving the other through callbacks. That is the same shape as a producer and a consumer taking turns — exactly the producer/consumer spine this series keeps returning to. Simula 67 had coroutines; so did Modula-2, Lua, and a long line of languages before "async/await" was a keyword anyone typed. What changed recently is not the concept but the *packaging*: compilers got good enough at the stackless state-machine transformation that you can write coroutine code that looks identical to blocking code, and runtimes got good enough at the `epoll`-driven executor that one thread can hold a hundred thousand of them. The async keyword is a 1963 idea with a 2015 compiler.

The reason the distinction between suspension and return matters so much in practice is that it changes what "the value" of a function is. A normal function's value is its *result* — you call it, you get a number. A coroutine's value is the *coroutine object itself* — a thing that, when you poke it, makes a little progress and either suspends or finishes. In Python you can hold a generator in a variable and call `next()` on it whenever you like. In Rust you can hold a `Future` in a variable and it does *nothing at all* until an executor polls it (Rust futures are famously "lazy" — constructing one runs none of its body; only `.await`-ing it or handing it to `spawn` causes a poll). In JavaScript an `async function` call returns a `Promise` immediately, and the body runs eagerly up to the first `await`. These differences in *when the body first runs* are real and bite people — but they are all variations on the one contract: the function is now a value you advance, not a result you receive.

## Stackful vs stackless: own a stack, or be a state machine

Here is the fork in the road. To let a function suspend and resume, you need to preserve its execution state. There are exactly two ways to do that, and they have completely different cost profiles.

**Stackful coroutines** give each coroutine its *own contiguous call stack* — a separate region of memory, just like a thread's stack but usually smaller and growable. When the coroutine suspends, the runtime saves a handful of CPU registers (stack pointer, instruction pointer, a few callee-saved registers) and switches to another stack. Because the coroutine has a real stack, it can suspend **anywhere** — including from three function calls deep, inside a library you didn't write, in a helper that has no idea it is running inside a coroutine. The suspension is a runtime operation (swap stacks), not a compile-time transformation. Goroutines in Go are stackful. So are fibers, green threads, Lua coroutines, and Kotlin's coroutines at the implementation level for some cases. The cost is the stack: even a small goroutine starts around 2–8 KB and can grow, so a million of them is gigabytes.

**Stackless coroutines** have no stack of their own. Instead, the **compiler transforms the coroutine's body into a state machine** — a plain struct/object that holds (a) which suspension point we are paused at (an integer "state" tag) and (b) the local variables that are live across that suspension point (as fields). Suspending is just "set the state field, save the live locals into fields, return." Resuming is "look at the state field, jump to the matching label, restore locals from fields." Because the whole thing is one flat object, it can suspend **only at the syntactic points the compiler could see and rewrite** — the `await` expressions. You cannot suspend from inside an ordinary function called by the coroutine, because that function was not rewritten; it has no state field to set. This is why stackless async is "viral" (`async` infects the call chain: to await inside `f`, `f` must itself be `async`). The payoff is enormous: a stackless task is just the bytes of its live locals — often tens to hundreds of bytes — so a million tasks fit in tens of megabytes. Rust, JavaScript, C#, and Python's `async`/`await` are all stackless.

![Matrix comparing stackful and stackless coroutines across own stack, suspend anywhere, memory per task, and example languages](/imgs/blogs/async-await-and-how-coroutines-actually-work-2.png)

Let me make the trade-off concrete, because "suspend anywhere" sounds like a pure win until you count the bytes.

#### Worked example: why "suspend anywhere" costs kilobytes

Suppose you want 200,000 concurrent connections, each handled by one coroutine. With **stackful** coroutines, each needs a stack. You cannot make the stack tiny, because the coroutine might call deep into a parser or a TLS library and need stack room; Go starts goroutine stacks at about 2 KB *and grows them on demand* precisely so deep calls don't overflow. Say steady-state each uses 8 KB. That is $200{,}000 \times 8\text{ KB} = 1.6\text{ GB}$ of stacks. Workable on a big box, painful on a small one.

With **stackless** coroutines, each task holds only its live locals. A connection handler that is parked awaiting a socket read might have live: a buffer pointer, a length, a state tag, a couple of integers — call it 200 bytes. That is $200{,}000 \times 200\text{ B} = 40\text{ MB}$. Same workload, a 40× difference, and the gap widens as you scale. The stackless task pays nothing for stack depth it isn't currently using, because there *is* no reserved stack — only the handful of variables that happen to be alive across the current `await`.

The flip side is ergonomics. Stackful lets you write a perfectly ordinary recursive descent parser and `await` from its deepest leaf; nothing in the call chain needs to know. Stackless forces the `async` keyword up the entire chain and forbids awaiting from non-`async` callees. That viral `async` is the price of bytes. The series' [event loop and reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) post covers the machinery underneath both; here we focus on the transformation that makes stackless possible.

This viral property has a famous name: **function coloring.** In a stackless language there are effectively two colors of function — "red" (`async`, can await) and "blue" (ordinary, cannot await) — and the rules are asymmetric. A red function can call a blue function freely. A blue function *cannot* await a red one — it can only call it to get the future and then it is stuck, because it has no suspension point of its own to wait at. So once any leaf in your call graph needs to await (a database driver, an HTTP client), every function on the path from `main` down to that leaf must be colored red. You cannot "sneak" an await into the middle of a blue call tree. This is why migrating a large synchronous codebase to async is rarely a local change: it ripples all the way up. Stackful languages have *no* colors — every function is the same color because suspension is a runtime operation, not a compile-time keyword, so any function can be parked by the scheduler regardless of who wrote it. Go's deliberate omission of `async`/`await` is exactly a refusal to color functions; the cost it pays for that uniformity is the per-goroutine stack. When people say async is "infectious" or complain about "what color is your function," this is the precise mechanism they are describing, and it falls directly out of "stackless can only suspend at points the compiler rewrote."

There is a quieter ergonomic cost too: **debugging.** A blocking thread has a real, contiguous call stack — when it crashes or you attach a debugger, you see `main → serve → handle → read_header → recv`, a clean lineage. A suspended stackless task has *no* such stack; it is a flat state-machine struct parked at state 2, and the "call stack" that led there has long since unwound (that is the whole point — the thread went and did other work). So async stack traces are notoriously shallow and confusing: the trace shows the executor's poll loop, not the logical chain of awaits that led to the current suspension. Runtimes fight this with async-aware tracing (tokio's `tracing`, .NET's async stack-trace reconstruction, V8's async stack tags), but it is a genuine tax that stackful coroutines mostly avoid because they keep a real stack. Trade-offs all the way down: the same flatness that makes a stackless task cost bytes also makes it harder to see where it came from.

| Property | Stackful (goroutines, fibers) | Stackless (Rust, JS, C#, Python) |
| --- | --- | --- |
| Execution state stored as | a real call stack | a compiler-generated state machine object |
| Can suspend from deep in nested calls? | yes, anywhere | no, only at syntactic `await` |
| Memory per unit | ~2–8 KB stack (growable) | ~bytes of live locals |
| `async` is viral up the call chain? | no | yes |
| Suspension is | a runtime stack switch | a compile-time rewrite + a return |
| Who can suspend it | the scheduler, often preemptively | the executor, only at await points |

## How async/await desugars into a state machine

This is the heart of the post. When you write a stackless `async fn`, the compiler does not produce a function that "waits." It produces a **struct that implements a `poll` method**, where each `await` is split into a numbered state. Let me build it up in slow motion, because once you have seen the transformation once you will see it forever.

Take a deliberately small async function. Pseudocode first, then real Rust:

```rust
async fn handle(conn: &mut Conn) -> Reply {
    let header = read_header(conn).await;   // await point 1
    let body = read_body(conn, header.len).await; // await point 2
    build_reply(header, body)               // pure, no await
}
```

There are two `await`s, so there are three places execution can be: at the start (before await 1), parked at await 1, and parked at await 2. (After the final expression there is no more suspension — the function completes.) The compiler enumerates those as **states**: call them `Start`, `AwaitingHeader`, `AwaitingBody`, plus a terminal `Done`. The locals that must survive a suspension become **fields** of the state machine: `header` is computed before await 2 and used after it, so `header` must be stored as a field; `conn` is used across both awaits, so it is a field too. A local that lives entirely between two awaits without crossing one does *not* need a field — it can stay on the real stack during a single `poll` call.

What does the compiler emit? Conceptually, an enum (the state) wrapped in a struct (the saved locals), with a `poll` method that is one big `match`/`switch` on the current state. Here is a faithful hand-written version of what `rustc` generates for `handle`. It is verbose precisely because the compiler is doing the bookkeeping you usually never see:

```rust
use std::task::{Context, Poll};

// The compiler-generated state machine. You never write this — it is what
// `async fn handle` becomes. Each await is a variant; locals that cross an
// await are carried as fields.
enum HandleState<'a> {
    Start { conn: &'a mut Conn },
    AwaitingHeader { conn: &'a mut Conn, fut: ReadHeaderFut<'a> },
    AwaitingBody { conn: &'a mut Conn, header: Header, fut: ReadBodyFut<'a> },
    Done,
}

impl<'a> Future for HandleState<'a> {
    type Output = Reply;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Reply> {
        loop {
            match &mut *self {
                HandleState::Start { conn } => {
                    // begin await point 1: create the sub-future, move to next state
                    let fut = read_header(conn);
                    *self = HandleState::AwaitingHeader { conn, fut };
                    // loop around and poll the new state immediately
                }
                HandleState::AwaitingHeader { conn, fut } => {
                    match fut.poll(cx) {
                        Poll::Pending => return Poll::Pending, // SUSPEND: yield to executor
                        Poll::Ready(header) => {
                            // header is now live across await 2 -> save it as a field
                            let fut2 = read_body(conn, header.len);
                            *self = HandleState::AwaitingBody { conn, header, fut: fut2 };
                        }
                    }
                }
                HandleState::AwaitingBody { conn: _, header, fut } => {
                    match fut.poll(cx) {
                        Poll::Pending => return Poll::Pending, // SUSPEND again
                        Poll::Ready(body) => {
                            let reply = build_reply(header.clone(), body);
                            *self = HandleState::Done;
                            return Poll::Ready(reply);          // COMPLETE
                        }
                    }
                }
                HandleState::Done => panic!("polled after completion"),
            }
        }
    }
}
```

Read what each `await` became. `let header = read_header(conn).await` turned into: create the sub-future, enter the `AwaitingHeader` state, and on each `poll` ask the sub-future if it is ready. If it returns `Poll::Pending`, the whole state machine returns `Poll::Pending` — **that return is the suspension.** Control goes back to the caller of `poll` (the executor), the thread is free, and the state machine sits in memory at `AwaitingHeader` with `conn` saved. When something later calls `poll` again, the `match` jumps straight to `AwaitingHeader` — *not* to the top — and re-asks the sub-future. The locals are in fields, so they are exactly as they were. That is suspend-and-resume, built entirely out of "return early" and "switch on a tag."

A few details that matter:

- **`await` is a loop, not a single check.** A sub-future can return `Pending` many times before `Ready`. Each `Pending` is a separate suspension; each later `poll` re-enters the same state. The state only advances when the sub-future is `Ready`.
- **Locals become fields only if they cross an await.** This is why async closures and async blocks have a *size* you can measure — the struct is exactly big enough to hold the largest set of simultaneously-live cross-await locals. A handler that holds a 64 KB buffer across an await makes a 64 KB-ish future. (This is the source of the infamous "large future" warnings.)
- **No magic threads.** Nothing here spawns a thread. `poll` runs on whatever thread the executor calls it from. Suspension is `return Poll::Pending`. Resumption is `the executor calls poll again`. The whole apparatus is single-function machinery.

![Acyclic state machine graph with start, two await states, a pending state, the executor that re-polls, and a done state](/imgs/blogs/async-await-and-how-coroutines-actually-work-4.png)

The same transformation happens in every stackless language; only the spelling differs. In **C#**, the compiler builds a struct implementing `IAsyncStateMachine` with an `int` field named `<>1__state` and a `MoveNext()` method that is a `switch` on that state — `MoveNext` is C#'s `poll`. In **JavaScript** (and TypeScript's downlevel output), an `async function` becomes a generator threaded through a helper that resumes it on each `Promise` resolution; the suspension points are the `await`s, and the engine resumes by calling the generator's `.next()`. In **Python**, an `async def` compiles to a coroutine object whose `.send()` drives it from one `await` to the next, and `await`ing on something that isn't ready raises out via the iterator protocol back to the loop. Different nouns — `MoveNext`, `.next()`, `.send()`, `poll` — same verb: *advance the state machine to the next suspension point.*

To make the cross-language sameness concrete, here is roughly what a C# compiler emits — the structure is public and stable, so this is faithful rather than invented. The `async` method becomes a struct whose `MoveNext` is a `switch` on an integer state, with hoisted locals as fields:

```csharp
// hand-sketch of the C# compiler output for an async method with one await.
// `<>1__state` is the real generated field name; MoveNext is the poll.
struct HandleStateMachine : IAsyncStateMachine
{
    public int <>1__state;          // -1 = start, 0 = awaiting, -2 = done
    public Conn conn;               // hoisted local (crosses the await)
    private TaskAwaiter<Header> awaiter;

    public void MoveNext()
    {
        switch (<>1__state)
        {
            case -1:                                  // Start
                awaiter = ReadHeader(conn).GetAwaiter();
                if (!awaiter.IsCompleted) {           // not ready -> SUSPEND
                    <>1__state = 0;
                    awaiter.OnCompleted(MoveNext);    // register the resume
                    return;
                }
                goto case 0;
            case 0:                                   // resumed: header ready
                Header h = awaiter.GetResult();
                Finish(conn, h);
                <>1__state = -2;                      // Done
                return;
        }
    }
    public void SetStateMachine(IAsyncStateMachine sm) { }
}
```

`OnCompleted(MoveNext)` is C#'s waker: "when this awaiter finishes, call `MoveNext` again." Squint past the syntax and it is the exact same machine as the Rust `poll`: a tag, hoisted locals, a switch that either suspends by registering a resume callback or advances on ready. JavaScript's transpiled output (what TypeScript or Babel emits when targeting older engines) is the same idea wearing generator clothing — a `switch (_state)` inside a function that a `__awaiter` helper resumes via `.next()` each time a `Promise` settles. Every stackless async language is this machine; only the spelling and the laziness rules differ.

There is one more piece of the mechanism that the simple version above quietly skips, and it is the single hardest part of stackless async: **self-reference.** Consider an async function that takes a reference to one of its own locals across an await — say it makes a buffer, then awaits a read *into a slice of that buffer*. After the transformation, both the buffer and the pointer-into-the-buffer become fields of the same state-machine struct. Now the struct contains a pointer to *itself*. If the executor ever *moved* that struct in memory (copied it to a new address, as moving a value normally does), the internal pointer would still point at the old location — a dangling pointer, instant memory corruption. Rust's answer is the `Pin` type: once a future has been polled, it is *pinned* — guaranteed not to move — so self-pointers stay valid. That is why `poll` takes `self: Pin<&mut Self>` and why `Pin` is the concept everyone trips over when learning Rust async. Languages with a garbage collector and indirection-by-default (JS, C#, Python) sidestep this because their objects already live behind a stable handle; Rust, which puts values directly inline for speed, has to name the constraint explicitly. The lesson worth keeping: a stackless coroutine that holds a reference across an await is a *self-referential* object, and something — a `Pin`, a GC handle, a box — has to guarantee it never moves while suspended.

## Who drives it: the executor and the event loop

A state machine that implements `poll` does nothing on its own. `poll` is *pull-based*: it only advances when someone calls it. That someone is the **executor** (Rust/tokio's term), the **event loop** (JS, Python `asyncio`), or the **scheduler** (Go, for its stackful goroutines). Whatever the name, its job is the same: hold a set of tasks, repeatedly call `poll`/`MoveNext`/`.next()` on the ones that are *ready to make progress*, and otherwise sleep efficiently until something becomes ready.

Here is the crucial mechanism, and it is what makes async efficient rather than a busy-wait disaster. When a future returns `Poll::Pending`, how does the executor know *when* to poll it again? It must not spin in a loop re-polling — that would burn a core. The answer is the **waker**. When a leaf future (a socket read, a timer) returns `Pending`, it first stashes a callback — the `Waker` from the `Context` that was passed into `poll` — with the underlying I/O reactor. The reactor registers the file descriptor with the kernel via `epoll` (Linux), `kqueue` (BSD/macOS), or `IOCP` (Windows) and goes to sleep in a single `epoll_wait` syscall covering *all* parked tasks. When the kernel reports the fd is readable, the reactor calls the stored waker, which marks that one task "ready," and the executor pops it off the ready queue and calls `poll` again. No spinning. One thread can hold tens of thousands of parked tasks, all blocked in a single `epoll_wait`, and wake exactly the ones whose I/O completed.

![Layered stack from your async function down through future, executor, event loop, epoll, and the kernel](/imgs/blogs/async-await-and-how-coroutines-actually-work-5.png)

So the full life of one suspension is: your `async fn` hits `.await` → its state machine `poll`s a leaf future → the leaf registers its fd with the reactor and stores the waker → the leaf returns `Pending` → your state machine returns `Pending` → the executor sets that task aside and runs *other* ready tasks → eventually `epoll_wait` reports the fd ready → the reactor fires the waker → the task is re-queued → the executor `poll`s it again → this time the leaf returns `Ready` and your state machine advances to its next state. That round trip is the entire async model. Everything else is optimization.

![Timeline of one task running to an await, yielding the thread, the executor running other tasks, the kernel signaling readiness, and the task resuming](/imgs/blogs/async-await-and-how-coroutines-actually-work-3.png)

A minimal but real executor makes this concrete. Here is a toy single-threaded executor in Rust, stripped to its essence — it is genuinely how the simplest runtimes work:

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

struct Task {
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
    queue: Arc<Mutex<VecDeque<Arc<Task>>>>, // the ready queue this task re-enters
}

impl Wake for Task {
    fn wake(self: Arc<Self>) {
        // "this task can make progress now" -> put it back on the ready queue
        self.queue.lock().unwrap().push_back(self.clone());
    }
}

fn run(queue: Arc<Mutex<VecDeque<Arc<Task>>>>) {
    loop {
        let task = match queue.lock().unwrap().pop_front() {
            Some(t) => t,
            None => break, // nothing ready; a real loop would epoll_wait here
        };
        let waker = Waker::from(task.clone());
        let mut cx = Context::from_waker(&waker);
        // poll the task once; if Pending it parked itself via its waker
        let _ = task.future.lock().unwrap().as_mut().poll(&mut cx);
    }
}
```

Notice there is no thread per task. There is one loop, pulling tasks off a ready queue and polling each once. A task that returns `Pending` simply is not re-queued until its waker fires. The JavaScript event loop is the same shape with different vocabulary: the **microtask queue** holds continuations created by resolved `Promise`s, and after each macrotask the engine drains the microtask queue to completion before rendering or handling the next event. Python's `asyncio` loop is the same shape again: `loop.run_forever()` pops ready callbacks, runs them, and uses `selectors` (an `epoll` wrapper) to sleep until I/O.

The waker is the part most people never see, and it is worth lingering on because it is *the* mechanism that makes async efficient rather than a busy-wait. Naively, an executor could just re-poll every pending task in a tight loop: poll task 1 (Pending), poll task 2 (Pending), … poll task N (Pending), back to task 1. That works, and it is also a catastrophe — it pins a CPU core at 100% doing nothing, asking sockets "ready yet? ready yet?" millions of times a second. The waker removes the busy-wait by inverting the question. Instead of the executor asking "is task K ready?", each parked leaf future *tells* the system "wake me when my fd is ready" by handing its `Waker` to the reactor, then the reactor blocks the whole thread in one `epoll_wait` that covers every parked fd at once. When the kernel returns from `epoll_wait` with "fds 7, 19, and 204 are readable," the reactor fires exactly those three wakers, which push exactly those three tasks onto the ready queue, and the executor polls only those three. Ten thousand parked connections cost one sleeping thread and zero CPU until something actually happens. This is the difference between $O(N)$ polling per loop iteration and $O(\text{ready})$ — and at the C10k scale that is the difference between a server that works and one that melts.

The vocabulary differs across runtimes but the mechanism is identical. In tokio, the `Waker` is an explicit object threaded through `Context`. In JavaScript, the "waker" is implicit: resolving a `Promise` enqueues its `.then` continuations onto the microtask queue, which the engine drains. In Python's `asyncio`, a `Future` parks by registering a callback via `add_done_callback`, and setting the future's result schedules that callback with `loop.call_soon`. In every case, "this task can make progress now" is a *push* from whatever the task was waiting on, never a *poll-everything* sweep by the scheduler. When you read "the event loop," picture this: a ready queue, a reactor blocked in one `epoll_wait`, and wakers that move tasks from "parked" to "ready" exactly when their I/O completes.

This is also where the producer/consumer spine of the series reappears, because a channel between two async tasks is built from precisely this machinery. When a consumer `await`s an empty channel, it parks — it stores its waker in the channel and returns `Pending`. When a producer sends a value, the channel's send path fires that stored waker, moving the consumer back onto the ready queue. The channel is shared mutable state (the buffer and the parked-waker slot), so it must establish a happens-before order over those accesses — exactly the frame this series keeps returning to — but the *waiting* is async suspension, not a thread blocking on a condition variable. An async producer/consumer pipeline is the same dataflow you would build with [condition variables](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly), with the blocking-wait replaced by a suspend-and-wake. The relationship between the executor (which drives state machines) and the reactor (which talks to `epoll`) is the subject of the [event loop and reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) post; here the point is just *who calls poll, and how it avoids spinning.*

## Cooperative scheduling: yield at await, or starve the loop

Now the consequence that bit me in production, stated as a principle. **Async scheduling is cooperative.** A task keeps the thread until it *voluntarily* yields, and the only place it yields is at an `await` that returns `Pending`. There is no timer interrupt that forcibly preempts a running task mid-computation (in most stackless runtimes). The scheduler is polite by construction: it can only switch tasks at the boundaries the task itself offers. That is wonderful for reasoning — your code runs atomically between awaits, so a lot of "shared mutable state" hazards on a single-threaded executor simply evaporate, because no other task can interleave except at an await you can point to. It is terrible if a task forgets to be polite.

Picture two tasks on one executor thread. Task A awaits a socket read every few microseconds; it is a good citizen, yielding constantly. Task B does this:

```python
async def hash_everything(items):   # the classic async footgun
    total = 0
    for x in items:            # 10 million iterations, NO await anywhere
        total = expensive_hash(total, x)   # pure CPU, never yields
    return total
```

While `hash_everything` runs, it never returns `Pending`, so it never hands the thread back. The event loop cannot run Task A, cannot service new connections, cannot fire timers, cannot even cancel anything — it is *blocked inside your coroutine*, not blocked on I/O. For the full duration of that loop, every other task is frozen. If `expensive_hash` over ten million items takes 80 ms, your service's p99 latency just acquired an 80 ms cliff that has nothing to do with load. This is the single most common async outage I have debugged, and it is invisible to thread-based intuition because there *is no other thread* to absorb the work.

![Before and after contrasting a coroutine that awaits and yields the loop against a CPU loop that never awaits and starves every other task](/imgs/blogs/async-await-and-how-coroutines-actually-work-6.png)

The same footgun has a second form: calling a **blocking** API from inside a coroutine. A synchronous `requests.get(url)`, a `time.sleep(5)`, a blocking file read, a CPU-heavy regex — any of these holds the OS thread without yielding to the loop, and on a single-threaded executor that means the whole loop stalls for the duration. The fix is always one of three moves, and choosing among them is half of writing good async code:

- **Use the async version** of the call so it actually awaits (`await client.get(url)` instead of blocking `requests.get`). This is the right fix when an async equivalent exists.
- **Offload to a thread pool** for genuinely blocking or CPU-bound work (`tokio::task::spawn_blocking`, `loop.run_in_executor(...)`, C#'s `Task.Run`). The blocking work runs on a separate OS thread so it does not stall the loop; the coroutine awaits the handle.
- **Yield periodically** inside long CPU loops (`tokio::task::yield_now().await`, `await asyncio.sleep(0)`) so other tasks get a turn. This is a band-aid for code you cannot move off the loop, and it must be applied deliberately.

The deeper point: **on an async runtime, "don't block the event loop" is not a style guideline, it is a correctness requirement.** A blocking call in async code is the moral equivalent of holding a global lock across I/O — covered, in the lock world, by the series' [deadlock](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them) post. The mechanism differs but the symptom is identical: one piece of code monopolizes a shared resource (here, the executor thread) and everything that needs that resource waits.

#### Worked example: measuring the starvation

To make starvation undeniable, instrument it. Run an async server with a heartbeat task that is *supposed* to print every 10 ms, then submit one request that does 100 ms of un-awaited CPU work. With a correctly cooperative workload the heartbeat ticks like a metronome: 10, 20, 30, 40 ms. The moment the CPU request lands, the heartbeat goes silent — no tick at 50, 60, 70, 80, 90, 100, 110, 120, 130 ms — and then resumes at 140 ms, exactly one CPU-burst later. You will see a single gap equal to the CPU burst, dropped right into the heartbeat stream. That gap *is* the starvation, and it is exactly as long as the longest stretch your code runs without awaiting. The remedy — `run_in_executor` for the CPU work — restores the metronome, because the burst now runs on a thread-pool thread and the loop keeps ticking. I have used precisely this heartbeat trick to localize "mystery latency" outages to the offending handler within minutes.

## Go's stackful goroutines vs the stackless world

Go made the opposite bet from Rust/JS/C#/Python, and seeing the contrast cements the whole concept. Go has no `async`/`await` keywords *at all*. You write what looks like ordinary blocking code, you put `go` in front of a function call to run it concurrently, and a goroutine — a **stackful coroutine** — runs it. When that goroutine does a "blocking" operation (a socket read, a channel receive), the Go runtime transparently parks the goroutine, registers the fd with its internal `netpoller` (which is `epoll`/`kqueue` underneath), and schedules another goroutine onto the OS thread. The blocking is an illusion maintained by the runtime; underneath, it is the same `epoll` reactor everyone else uses.

Because goroutines are stackful, they can suspend from anywhere — no viral `async`, no colored functions, no `.await` peppered through your code. You call a function, it might block, the runtime handles it. That is a genuine ergonomic win, and it is why Go code is so pleasant to write concurrently. Go also made its scheduler **preemptive** (since Go 1.14): a background monitor can interrupt a goroutine that has run too long without yielding — even a tight CPU loop — by setting an asynchronous preemption signal, so one goroutine cannot starve the others the way a non-awaiting coroutine starves a single-threaded executor. Go's scheduler is the famous **GMP** model: G = goroutines, M = OS threads, P = logical processors (scheduling contexts); the runtime multiplexes many Gs onto few Ms across P contexts, and work-steals to balance them.

![Matrix of language async models showing Go as stackful with the Go scheduler and Rust, JS, Python, and C-sharp as stackless with their drivers](/imgs/blogs/async-await-and-how-coroutines-actually-work-7.png)

Here is the same connection handler in Go (stackful, no `async`) and Rust (stackless, explicit `await`), side by side, so the divergence is concrete:

```go
// Go: stackful goroutine. No async keyword. "Blocking" reads are
// transparently parked by the runtime; one goroutine per connection scales.
func handle(conn net.Conn) {
    defer conn.Close()
    header := readHeader(conn) // looks blocking; runtime parks the goroutine
    body := readBody(conn, header.len)
    conn.Write(buildReply(header, body))
}

func serve(ln net.Listener) {
    for {
        conn, _ := ln.Accept()
        go handle(conn) // launch a stackful coroutine; cheap (~KBs)
    }
}
```

```rust
// Rust: stackless. `async`/`.await` are explicit; the function is a state
// machine, and `tokio::spawn` hands it to the executor.
async fn handle(mut conn: TcpStream) {
    let header = read_header(&mut conn).await; // explicit suspension point
    let body = read_body(&mut conn, header.len).await;
    let _ = conn.write_all(&build_reply(header, body)).await;
}

async fn serve(listener: TcpListener) {
    loop {
        let (conn, _) = listener.accept().await.unwrap();
        tokio::spawn(handle(conn)); // hand the state machine to the executor
    }
}
```

They express the *same* concurrency. Go hides the machine inside the runtime and gives you preemption and "suspend anywhere" at the cost of per-goroutine stacks. Rust exposes the machine, gives you bytes-per-task and zero-cost-when-idle at the cost of viral `async` and cooperative-only scheduling (no preemption — a non-awaiting Rust future *can* starve its executor, exactly as the Python example did). Neither is "better" in the abstract; they sit at different points on the same trade-off curve. The series' [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) turns this into a decision procedure.

The preemption difference deserves one more beat, because it is the most underappreciated distinction between the two camps and it directly determines how dangerous the "CPU hog" footgun is in each. In Go, a goroutine that enters a tight CPU loop *will* be preempted: since Go 1.14 the runtime can interrupt a goroutine at a safe point via an asynchronous signal even if it never makes a function call, so one greedy goroutine cannot freeze the others — it just shares the cores fairly with them. The Go scheduler is effectively a small operating system in user space, with timeslices and preemption. In the stackless cooperative world (Rust/tokio, JS, Python, C#), there is *no* such safety net by default: the executor only regains control when a task awaits, so a task that never awaits owns its thread until it finishes, full stop. This is not a defect to be fixed — it is the deliberate trade for the zero-cost, bytes-per-task model, and it is *why* "don't block the event loop" is repeated so insistently in those ecosystems and barely mentioned in Go's. Go moved the burden of fairness into the runtime; the stackless languages left it with you. When you choose a model, you are also choosing who is responsible for keeping one task from starving the rest: the runtime (Go) or the programmer (everyone else). That single choice explains most of the cultural difference between "just write blocking-looking code and `go` it" and "audit every loop for a missing `await`."

A subtle corollary worth naming: because Go can preempt, Go can use *all your cores* for a CPU-bound parallel workload simply by launching goroutines — the runtime spreads them across OS threads (the `M`s) up to `GOMAXPROCS`. A single tokio executor, by contrast, is often configured multi-threaded too (a work-stealing pool of worker threads), but a *single* future still runs on one thread at a time, and a single `!Send` future is pinned to one worker. So "use more cores" is automatic-ish in Go and a deliberate architectural choice in the stackless world. This is yet another reason the right answer for CPU-bound work is usually real threads (or Go), and the right answer for I/O-bound high-concurrency work is usually stackless async — they are optimized for opposite ends of the $C/(W+C)$ axis.

| Axis | Go goroutines (stackful) | Rust/JS/C#/Python (stackless) |
| --- | --- | --- |
| Keywords | none — `go f()` | `async` / `await` |
| Function coloring | none | viral `async` up the chain |
| Suspend from deep calls | yes | no — only at `await` |
| Preemptible CPU loop | yes (async preemption) | no (cooperative only) |
| Memory per unit | ~2–8 KB growable stack | ~bytes of live locals |
| Who parks on I/O | runtime netpoller | leaf future + waker + reactor |

## Cancellation lives at the suspension points

There is a subtle, beautiful corollary of "stackless coroutines suspend only at await": **cancellation also happens only at await.** In a cooperative model, you cannot forcibly kill a task mid-computation the way you might (dangerously) kill a thread — there is no safe place to interrupt arbitrary running code. Instead, cancellation is *checked at suspension points.* When you cancel an async task, what actually happens depends on the language, but the shape is the same: the next time the task would be polled (or is sitting parked at an await), it is told to stop, and it unwinds from that suspension point — running its cleanup (`Drop` in Rust, `finally` in Python/C#, deferred cleanup in structured-concurrency scopes) as it goes.

This has a sharp consequence: **a task that never awaits can never be cancelled.** Your 80 ms CPU loop is not just un-yielding — it is un-*cancellable*, because there is no suspension point at which the cancellation can take effect. The same property that lets one task starve the loop also makes it impossible to interrupt. In Rust, dropping a `Future` cancels it (its state machine is destroyed at whatever state it was parked in, running each field's destructor); in Python, `task.cancel()` arranges for a `CancelledError` to be raised *at the next await*; in C#, a `CancellationToken` is checked at await boundaries you thread through. All three are "cancellation = a thing that happens at a suspension point."

Here is the shape concretely. A common pattern is "race a piece of work against a timeout, and cancel the loser." In Rust with `tokio::select!`, the moment one branch completes, the *other* future is dropped — and dropping is cancellation:

```rust
use tokio::time::{timeout, Duration};

async fn fetch_with_deadline(url: &str) -> Result<Bytes, Timeout> {
    // if the fetch hasn't completed in 2s, the fetch future is DROPPED.
    // Dropping destroys its state machine at whatever await it was parked on,
    // running each field's destructor (closing the socket, freeing buffers).
    match timeout(Duration::from_secs(2), fetch(url)).await {
        Ok(bytes) => Ok(bytes),
        Err(_elapsed) => Err(Timeout),   // the in-flight fetch was cancelled
    }
}
```

The crucial detail: the cancellation took effect *at a suspension point* inside `fetch` — wherever it was parked awaiting the socket. There was a clean place to stop, run cleanup, and unwind, precisely because the only place `fetch` could be paused was at an `await`. Python expresses the same idea differently but with the same mechanism — `asyncio.wait_for(fetch(url), timeout=2)` arranges for a `CancelledError` to be *raised at the next await* inside `fetch`, which propagates up through `finally` blocks that close resources. C# threads a `CancellationToken` that participating awaits check. All three: cancellation is a thing that lands at a suspension point and unwinds with cleanup.

#### Worked example: the un-cancellable task

Take the starvation footgun from earlier — the 80 ms un-awaited CPU loop — and now try to *cancel* it after 10 ms. You cannot. `task.cancel()` (Python) or dropping the future (Rust) only schedules the cancellation to take effect *at the next await*, and there is no next await; the loop has none. So the cancel request sits queued while the loop runs to its natural end, 70 ms after you asked it to stop. The very same property that lets this task starve the loop — "no suspension points" — makes it impossible to interrupt. Contrast a *thread* doing the same loop: you still cannot safely force-kill it mid-instruction, but at least it does not freeze every other unit of work, because the OS preempts it on a timer. This is the sharpest practical reason the "never run long without awaiting" rule matters: un-yielding code is both un-scheduling-friendly *and* un-cancellable. The fix is the same as before — offload the CPU loop to a thread pool, or insert deliberate `yield_now().await` points that double as cancellation checkpoints.

This is exactly why **structured concurrency** matters and why it composes so cleanly with async. If every task's lifetime is bounded by a scope, and cancellation propagates to children at their await points, you get the guarantee that no task outlives its scope and that cancelling the scope reliably tears down its whole subtree — at the awaits, with cleanup. The mechanics of scopes, nurseries, and propagation are the subject of [structured concurrency, cancellation, and thread pool design](/blog/software-development/concurrency/structured-concurrency-cancellation-and-thread-pool-design); the relevant fact here is that the *transformation* we built (await = suspension point) is precisely the hook cancellation uses. The suspension points are not just where you yield for I/O — they are the only places the rest of the system can get a word in: to schedule another task, to wake you, or to cancel you.

## Worked example: desugar an async fn by hand

Let me do the full transformation manually, on a fresh function, so you can run the algorithm yourself on any async code you meet. Here is the source — a tiny pipeline that fetches a value, then fetches a second value that depends on the first, then combines them:

```rust
async fn combine(id: u64) -> String {
    let a = fetch_first(id).await;        // await 1
    let b = fetch_second(a.key).await;    // await 2; needs `a`
    format!("{}-{}", a.name, b.value)     // uses `a` and `b`; no await
}
```

**Step 1 — find the suspension points.** Scan for `await`. There are two: after `fetch_first` and after `fetch_second`. Two awaits means three "before/at" positions plus a terminal: `Start`, `Awaiting1`, `Awaiting2`, `Done`. These are your state tags.

**Step 2 — find the cross-await locals.** For each local, ask: is it written before an await and read after one? `a` is produced before await 2 and read by `fetch_second(a.key)` *and* by the final `format!`, so `a` crosses await 2 — it must be a **field**. `b` is produced at the end and used immediately with no await between — it never crosses an await, so it can live on the real stack during a single `poll`; no field needed. `id` is used to make the first future and not after — it is consumed in `Start`, so it lives only in the `Start` variant.

**Step 3 — assign locals to states.** `Start` holds `id`. `Awaiting1` holds the in-flight `fetch_first` future. `Awaiting2` holds `a` (the saved cross-await local) and the in-flight `fetch_second` future. `Done` holds nothing.

**Step 4 — write `poll` as a `match` with a driving loop.** Each arm either advances to the next state (looping to poll it immediately) or, on a sub-future's `Pending`, returns `Pending` to suspend. Here is the hand-desugared result:

```rust
enum Combine {
    Start { id: u64 },
    Awaiting1 { fut: FetchFirstFut },
    Awaiting2 { a: First, fut: FetchSecondFut }, // `a` saved across await 2
    Done,
}

impl Future for Combine {
    type Output = String;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<String> {
        loop {
            match &mut *self {
                Combine::Start { id } => {
                    let fut = fetch_first(*id);
                    *self = Combine::Awaiting1 { fut };
                }
                Combine::Awaiting1 { fut } => match fut.poll(cx) {
                    Poll::Pending => return Poll::Pending,        // suspend at await 1
                    Poll::Ready(a) => {
                        let fut = fetch_second(a.key);
                        *self = Combine::Awaiting2 { a, fut };    // save `a` as a field
                    }
                },
                Combine::Awaiting2 { a, fut } => match fut.poll(cx) {
                    Poll::Pending => return Poll::Pending,        // suspend at await 2
                    Poll::Ready(b) => {
                        let out = format!("{}-{}", a.name, b.value); // `b` on the stack
                        *self = Combine::Done;
                        return Poll::Ready(out);                  // complete
                    }
                },
                Combine::Done => unreachable!("polled after Ready"),
            }
        }
    }
}
```

Trace one execution to see suspend/resume in action. The executor calls `poll`. We are in `Start`; we build the first future and switch to `Awaiting1`; the loop re-enters and polls the first future. Suppose the network is slow: `fetch_first` returns `Pending`. We return `Poll::Pending`. **The function has suspended at await 1.** The `Combine` value sits in memory in the `Awaiting1` state, holding the in-flight future. The thread is free; the executor runs other tasks. Later the socket becomes readable, the waker fires, the executor calls `poll` again. The `match` lands in `Awaiting1`, re-polls the first future — now `Ready(a)`. We build the second future, save `a` into the `Awaiting2` field, loop, poll the second future. If it's ready, we `format!` and return `Ready`. If not, we suspend again with `a` safely stored. At no point did we re-run `fetch_first`, and `a` survived the second suspension because it was a field. That is the entire algorithm: **enumerate awaits as states, promote cross-await locals to fields, emit a `match` that returns `Pending` at each await and advances on `Ready`.** Run it on your own code and you will never again wonder what `async` "is."

## Measured behavior: a million tasks vs a million threads

Now the numbers, honestly framed. The reason async exists is a memory and scheduling story, and the gap is large enough that order-of-magnitude estimates make the point without me fabricating false precision. (Measure on your own platform; these are defensible magnitudes, and I'll say where they come from.)

**Memory.** An OS thread reserves a stack — Linux defaults to a *virtual* reservation of 8 MB per thread, of which physical pages are faulted in lazily, but even the committed and bookkeeping overhead lands around 1 MB of real footprint per thread in practice once stacks touch a few pages plus kernel structures. So a thread-per-task design at 100,000 tasks wants on the order of tens of gigabytes of address space and many gigabytes resident — and the kernel's thread tables and scheduler do not love six-figure thread counts. An async task, as we computed, is the bytes of its live cross-await locals — call it a few hundred bytes — so 100,000 tasks is tens of megabytes. **That is the headline: hundreds of bytes vs ~1 MB per unit, a roughly 1000× difference in per-task memory.** A million async tasks on one box is routine; a million threads is not.

**Context-switch cost.** Switching between OS threads is a kernel operation: trap into the kernel, save/restore the full register set, swap page-table and TLB context if crossing processes, run the scheduler. On commodity x86 this is on the order of **1–5 µs** per switch, plus indirect cache-pollution costs that can dwarf the direct cost. Switching between async tasks on one thread is a function return and a `match` — **tens of nanoseconds**, no kernel trap, no TLB flush. For an I/O-bound workload that switches constantly, that is a 100×-plus difference *per switch*, and at high connection counts the switches dominate.

![Matrix comparing async tasks and OS threads on memory per unit, scheduling, blocking call behavior, and CPU work](/imgs/blogs/async-await-and-how-coroutines-actually-work-8.png)

**Throughput at the C10k frontier.** The classic result that birthed async servers: a thread-per-connection server hits a wall somewhere in the low tens of thousands of connections — not because the CPU is busy (the connections are mostly idle, waiting on I/O) but because the threads themselves cost too much memory and the scheduler thrashes. An event-loop / async server holds the same 10,000+ connections in one or a few threads, each connection a cheap parked task in a single `epoll_wait`, and serves them with a fraction of the memory and far less scheduler overhead. This is *the* use case async was built for: **many connections, each mostly idle, each doing little CPU work.** The win is not throughput-per-request — a single thread doing real CPU work is plenty fast — it is *density*: how many concurrent, mostly-waiting tasks you can hold without the per-task overhead crushing you.

#### Worked example: where the thread model breaks even

It is tempting to read "async beats threads" as universal. It is not, and the break-even is instructive. Suppose each unit of work spends time $W$ waiting on I/O and time $C$ on CPU, and you have $N$ concurrent units on a machine with $k$ cores. Threads waste memory ($\sim 1\text{ MB} \times N$) and switch cost ($\sim 1\text{–}5\,\mu s$ per context switch), but they get *real parallelism* across cores for free and the OS preempts CPU hogs. Async saves the memory and switch cost but runs CPU work on one thread at a time per executor and starves on a hog. So the model that wins depends on the ratio $C / (W + C)$ and on $N$. When $C/(W+C)$ is tiny (mostly waiting) and $N$ is large (tens of thousands), async dominates by orders of magnitude — this is the web server, the proxy, the database connection pool. When $C/(W+C)$ approaches 1 (CPU-bound) regardless of $N$, threads-across-cores win and async wins nothing, because the bottleneck is compute and async gives you concurrency, not parallelism. When $N$ is small (a few dozen units) the per-task overhead of threads is negligible and the simplicity of threads usually wins. The honest decision is not "async vs threads" in the abstract — it is "where does *this* workload sit on the $C/(W+C)$ axis, at *this* $N$." Most network services sit far toward the waiting end at large $N$, which is why async became the default there; most batch-compute jobs sit at the CPU end, which is why they use thread pools and `rayon`/`ForkJoinPool` instead.

| Metric | Async tasks (one executor thread) | OS thread per task |
| --- | --- | --- |
| Memory per unit | ~hundreds of bytes (live locals) | ~1 MB real footprint (stack + kernel) |
| Switch cost | ~tens of ns (return + `match`) | ~1–5 µs (kernel trap + register save) |
| Scheduling | cooperative — yields at `await` | preemptive — kernel timeslice |
| One unit blocks | stalls the whole loop (bug) | only that thread (fine) |
| CPU-bound work | hogs the loop (bug) | scales across cores (good) |
| Practical concurrency ceiling | ~10^6 parked tasks | ~10^3–10^4 threads |

How to measure this honestly on your own box: spawn N tasks vs N threads, have each park on a timer or a socket, measure RSS and a heartbeat's jitter. **Warm up** (let allocators and the JIT settle), **run many times** (scheduling is nondeterministic), report a distribution not a single number, and **name the platform** — the thread-stack default, the scheduler, and even the OS page size shift the numbers. Do *not* conclude "async is faster" from a single hot-loop microbenchmark — async is not faster at CPU work; it is denser at concurrent I/O. The honest claim is narrow and important: **for many concurrent, I/O-bound, mostly-idle tasks, async wins on memory and switch cost by orders of magnitude; for CPU-bound work it wins nothing and can lose badly.**

## Case studies / real-world

**Rust's `Future` / poll design (the zero-cost bet).** Rust deliberately chose a *pull-based, stackless* future: `Future::poll` is called by an executor, returns `Poll::Pending` or `Poll::Ready`, and the future is responsible for stashing a `Waker` so it can be re-polled. The design rationale (documented in the Rust async book and the RFCs that landed `Future`, `Pin`, and `async/await`) was *zero overhead*: an idle task should cost only its saved state, with no heap allocation forced by the language and no runtime imposed by the standard library. The trade-off is real and was argued at length: poll-based futures need `Pin` to make self-referential state machines sound (a future that holds a pointer into its own buffer across an await must not move), which is the single hardest concept in Rust async. The lesson: stackless async buys density and predictability, and the bill is paid in type-system complexity. Rust shipped `async`/`await` stable in late 2019 after years of this design work.

**The Node.js / V8 event loop and the microtask queue.** Node's concurrency is a single-threaded event loop (over libuv, which wraps `epoll`/`kqueue`/IOCP) plus a thread pool for genuinely blocking work like file I/O and DNS. The well-known and frequently-mis-taught detail is the **microtask queue**: resolved `Promise` callbacks (and `await` continuations) run on a *microtask* queue that is fully drained after each macrotask and before the loop proceeds, which is why a `Promise.resolve().then(...)` runs before a `setTimeout(..., 0)`. The classic Node failure is exactly our footgun: a synchronous CPU loop or a synchronous `fs.readFileSync` in a request handler blocks the single loop and stalls every other request — the Node docs literally headline the guidance "Don't Block the Event Loop." That guidance is not advice; it is the cooperative-scheduling correctness requirement we derived, restated for one specific runtime.

**Go's goroutine scheduler (GMP) and async preemption.** Go's runtime multiplexes goroutines (G) onto OS threads (M) across logical processors (P), parks goroutines on I/O via the integrated netpoller, and work-steals across Ps for balance. The historically important fix was **asynchronous preemption**, added in Go 1.14: before it, a goroutine in a tight loop with no function calls (hence no cooperative yield point) could starve the scheduler — the exact stackful analogue of a non-awaiting coroutine. The fix was to let the runtime deliver a signal that preempts such a goroutine at a safe point. The lesson is symmetric with the Rust/Node story: stackful coroutines give you "suspend anywhere" ergonomics, but you still need *some* preemption to defend against CPU hogs — whereas the stackless cooperative world hands that responsibility entirely to you, the programmer, every time you write a loop without an `await`.

## When to reach for this (and when not to)

Async/await is not a general speed-up and it is not free. Reach for it when the shape of your problem matches what the machine is good at, and *avoid* it — actively avoid it — when it doesn't.

**Reach for async when:** the work is **I/O-bound** (network, disk, database, RPC) and you have **many concurrent units** that spend most of their time *waiting* — the C10k server, a crawler fanning out thousands of requests, a proxy holding tens of thousands of mostly-idle connections, a chat backend. Here async's density (hundreds of bytes per parked task, tens-of-nanoseconds switches, one `epoll_wait` for everyone) is exactly the right tool, and threads would drown in their own overhead. Also reach for it when you want **deterministic, cooperative** scheduling on a single thread so that your between-await code is effectively atomic — it kills a whole class of data races by construction.

**Do not reach for async when:** the work is **CPU-bound**. Async gives you *concurrency*, not *parallelism* — a single executor thread runs one task at a time, and a CPU loop hogs it. For CPU-bound parallel work you want real threads across cores (a thread pool, `rayon`, `ForkJoinPool`, Go's parallel goroutines), not coroutines. Reaching for async here buys you nothing and *adds* the risk of starving the loop. Likewise, **do not** sprinkle `async` on a program that has a handful of tasks and no I/O concurrency pressure — you are paying the viral-`async`, `Pin`, and debugging-the-state-machine tax for no density you needed; plain threads or even sequential code are simpler and just as fast. And **never** make a blocking call (a sync HTTP client, `time.sleep`, a blocking lock held across I/O, a heavy synchronous parse) from inside a coroutine without offloading it — that single line converts "many concurrent tasks" into "one task at a time with extra steps," and it is the defect I have spent more on-call hours chasing than any other in async systems.

One more anti-pattern to retire explicitly, because it is so common: **do not reach for async to make sequential code "faster."** Async is not an optimization you sprinkle on a slow function; it is a *structure* for holding many concurrent waits cheaply. If you have one request that does five database queries in sequence and each must finish before the next starts, making them `async` saves you nothing — the latency is the sum of the five, async or not. Async only helps when there is *concurrency to exploit*: five queries that could run *at the same time* (await them together with `join!`/`Promise.all`/`asyncio.gather`), or many independent requests in flight at once. If the work is inherently sequential and single, plain blocking code is simpler and exactly as fast. The question to ask before reaching for async is never "is this slow?" but "do I have many things waiting at once, or independent things that could overlap?" If the answer is no, async is pure overhead and complexity.

The decision in one line: **async for high-concurrency I/O-bound waiting; threads for CPU-bound parallelism; and if you choose async, the iron rule is never block the loop.** When you genuinely need both — a mostly-async server that occasionally does heavy CPU — combine them: async for the I/O, a `spawn_blocking`/`run_in_executor` thread pool for the CPU bursts, so the loop keeps ticking while the heavy work runs elsewhere. The full decision tree across locks, channels, actors, and async lives in [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model); composing the resulting async pieces into pipelines is the job of [futures, promises, and composing asynchronous code](/blog/software-development/concurrency/futures-promises-and-composing-asynchronous-code).

## Key takeaways

- **`await` is not blocking; it is suspending.** A blocking call parks the *thread*; an `await` that isn't ready suspends the *function* and frees the thread to run other tasks. That inversion is the whole source of both async's power and its footguns.
- **A stackless `async fn` is a state machine, not a thread.** The compiler enumerates each `await` as a numbered state, promotes the locals that cross an await into fields, and emits a `poll`/`MoveNext`/`.next()` method that is a `switch` on the state. Suspension is "return Pending"; resumption is "the executor calls poll again."
- **Stackful vs stackless is the core trade-off.** Stackful (goroutines, fibers) owns a real stack, can suspend anywhere, needs no `async` keyword, and costs kilobytes per unit. Stackless (Rust, JS, C#, Python) is a flat state machine, suspends only at syntactic `await`, makes `async` viral, and costs bytes per unit.
- **The executor drives everything by pulling.** `poll` does nothing until called; a leaf future stashes a `Waker` with an `epoll`/`kqueue` reactor and returns `Pending`; the kernel signals readiness; the reactor fires the waker; the executor re-polls. No spinning, one `epoll_wait` for all parked tasks.
- **Cooperative scheduling means a non-awaiting coroutine starves the loop.** Between awaits your code runs uninterrupted; a CPU loop or a blocking call with no `await` monopolizes the single executor thread and freezes every other task. "Don't block the event loop" is a correctness requirement, not a style tip.
- **Cancellation happens at suspension points.** You cannot interrupt arbitrary running code; cancellation lands at the next `await`, runs cleanup, and unwinds — which is exactly why a task that never awaits can never be cancelled, and why structured concurrency composes with async.
- **Async wins on density for I/O-bound concurrency, not on raw CPU speed.** ~hundreds of bytes vs ~1 MB per unit and ~tens of ns vs ~1–5 µs per switch make a million parked tasks routine; for CPU-bound parallel work, reach for threads across cores instead.
- **When you must do CPU or blocking work in async code, offload it** to a thread pool (`spawn_blocking`, `run_in_executor`, `Task.Run`) so the loop keeps ticking — or yield periodically inside long loops as a deliberate band-aid.

## Further reading

- **Within this series:** [Why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) (the intro and the shared-state frame), [The event loop and the reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) (the `select`/`poll`/`epoll` machinery under the executor), [Futures, promises, and composing asynchronous code](/blog/software-development/concurrency/futures-promises-and-composing-asynchronous-code) (combining the state machines into pipelines), [Structured concurrency, cancellation, and thread pool design](/blog/software-development/concurrency/structured-concurrency-cancellation-and-thread-pool-design) (scopes, cancellation propagation, offloading), and [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) (when async vs threads vs actors).
- **Python specifics:** [asyncio from the ground up: event loops and coroutines](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) — the cleanest concrete walkthrough of a real stackless coroutine runtime, with `.send()`-driven coroutines and the selector loop.
- *The Rust Async Book* (`rust-lang.github.io/async-book`) — the `Future`/`poll`/`Waker`/`Pin` model first-hand, including why poll-based futures need `Pin` for self-referential state machines.
- The Node.js guide *"Don't Block the Event Loop"* and the V8/libuv documentation on the macrotask/microtask queues — the cooperative-scheduling correctness requirement, restated for JavaScript.
- The Go runtime scheduler design docs and the Go 1.14 *asynchronous preemption* release notes — the stackful GMP model and why even goroutines needed preemption against CPU hogs.
- Marlin & Wirth's foundational notes on coroutines, and the Kotlin coroutines design documents — the suspend/resume contract from first principles, across stackful and stackless implementations.

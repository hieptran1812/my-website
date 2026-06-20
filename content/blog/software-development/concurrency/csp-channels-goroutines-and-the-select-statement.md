---
title: "CSP, Channels, Goroutines, and the Select Statement"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Build a bounded, cancellable worker-pool pipeline out of channels and goroutines, and learn exactly when a channel beats a mutex and when it does not."
tags:
  [
    "concurrency",
    "parallelism",
    "csp",
    "channels",
    "goroutines",
    "select",
    "pipelines",
    "go",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/csp-channels-goroutines-and-the-select-statement-1.png"
---

Here is a small program that does real concurrent work and contains not one lock, not one `volatile`, not one atomic, and not one shared variable that two tasks touch at the same time:

```go
func main() {
	jobs := make(chan int, 100)
	results := make(chan int, 100)

	// Start three workers. They share nothing but the two channels.
	for w := 0; w < 3; w++ {
		go worker(jobs, results)
	}

	// Feed the workers, then close the input.
	for j := 1; j <= 9; j++ {
		jobs <- j
	}
	close(jobs)

	// Collect exactly nine results.
	sum := 0
	for i := 0; i < 9; i++ {
		sum += <-results
	}
	fmt.Println(sum)
}

func worker(jobs <-chan int, results chan<- int) {
	for j := range jobs {
		results <- j * j // pretend this is expensive
	}
}
```

Three workers run in parallel, each grabbing jobs off a queue and pushing squared results back. There is no race condition here, and not because we were careful with locks — there is no race because **no two goroutines ever name the same piece of mutable memory**. The job `j` lives on exactly one worker's stack at a time. Ownership of a value moves from the producer to a channel to one consumer, like a baton in a relay, and only one runner holds the baton at any instant. That is the whole idea behind Communicating Sequential Processes, the model Tony Hoare described in 1978 and the model Go made famous: *don't communicate by sharing memory; share memory by communicating.*

This post is about making that idea concrete. We will build channels up from nothing — typed conduits that carry values between independently-running tasks — and we will see the sharp difference between an **unbuffered** channel (a synchronous rendezvous, where the sender waits for a receiver to be standing right there) and a **buffered** one (which decouples the two parties up to a fixed capacity). We will spawn cheap **goroutines** by the thousand, wait on several channels at once with the **select** statement, wire up **fan-out/fan-in** pipelines, cancel work cleanly with the **done-channel** and **context** idioms, learn the precise rules for **closing** a channel, and reproduce the dreaded `fatal error: all goroutines are asleep - deadlock!` so we understand exactly why it happens. The figure below previews the single most important distinction in the whole post — what a send does on an unbuffered versus a buffered channel — and the rest of the article earns every claim in it with mechanism, runnable code in Go (with Rust, Kotlin, and Clojure contrasts), and honest measurements.

![two side by side states comparing an unbuffered channel where the sender blocks waiting for a rendezvous against a buffered channel where the sender returns immediately until the buffer fills](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-1.png)

If you have not read [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), that post frames the hazard this whole series circles: shared mutable state plus nondeterministic scheduling. Channels are one of the cleanest ways to dissolve that hazard rather than fight it, and the companion post on [message passing versus shared memory and the CSP philosophy](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy) makes the philosophical case. Here we get our hands dirty.

## Channels: typed conduits between tasks

A channel is a typed, first-class conduit. You make one with a value type and a capacity, you send values in with one operator, you receive them out with another, and the runtime guarantees that each value sent is received exactly once by exactly one receiver. In Go:

```go
ch := make(chan int)      // unbuffered channel of int
ch <- 42                  // send: put 42 in
x := <-ch                 // receive: take a value out
```

The type matters. A `chan int` carries `int`; a `chan Job` carries `Job` structs; a `chan struct{}` carries no data at all and exists purely for signaling (the empty struct occupies zero bytes, so a "send" on it is a pure synchronization event). The compiler will not let you send a string down a `chan int`, which means a whole class of "wrong message in the wrong queue" bugs is caught before the program runs.

Two refinements make channels safer in practice. First, **directionality**: a function parameter can be typed `<-chan int` (receive-only) or `chan<- int` (send-only). In our opening example, `worker(jobs <-chan int, results chan<- int)` says, right in the signature, that this function only *reads* jobs and only *writes* results. A worker that accidentally tried to send into `jobs` would not compile. The arrow points the way the data flows, and the type system enforces the contract. Second, the **range** loop: `for j := range jobs` pulls values until the channel is closed and the buffer is drained, then ends the loop cleanly. We will return to closing shortly, because the closed-channel rules are where most channel bugs actually live.

What is a channel *made of*? Underneath, Go's channel is a small struct (`hchan`) holding a ring buffer (for buffered channels), a mutex, and two queues of waiting goroutines — one for blocked senders, one for blocked receivers. A send acquires the channel's internal lock, and then one of three things happens: if a receiver is already waiting, the value is copied straight into the receiver's stack and that receiver is made runnable (a *direct handoff*, no buffer involved); if there is room in the buffer, the value is copied into the ring and the send returns; otherwise the sending goroutine is parked on the sender queue and the scheduler runs someone else. So yes, there is a lock *inside* the channel — but you never see it, never hold it across your own code, and never have to reason about its ordering. The channel turns a hard, open-ended "coordinate these N goroutines" problem into a narrow, well-tested primitive. That is the trade: you give up direct shared memory and you get a conduit with airtight semantics.

This is the first place to plant a flag that we will keep returning to: a channel establishes a **happens-before** edge. When goroutine A sends a value and goroutine B receives it, everything A did *before* the send is guaranteed visible to B *after* the receive. You do not need a separate memory barrier; the channel operation is the barrier. That single guarantee is why our opening worker pool is race-free without a `sync.Mutex` anywhere — the handoff of `j` from `main` to a worker, and of the result back, each carry a happens-before edge that orders the memory accesses. The [memory models and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) post derives that relation in full; here, just trust that a channel send/receive pair is one of the strongest, simplest happens-before edges you can buy.

It is worth being precise about *which* operations order *which*, because the Go memory model spells it out and the rules occasionally surprise people. For an **unbuffered** channel, the *receive* completes before the *send* completes — that is, the receiving goroutine is guaranteed to be at the rendezvous before the sender is allowed to return, so the send acts as a full barrier in both directions. For a **buffered** channel of capacity C, the rule is the elegant one: the *k*-th receive on the channel happens-before the *(k+C)*-th send completes. In plain English: the send that fills the buffer past its capacity is held until the receive that made room has happened, which is exactly the slack the buffer gives you, stated as a memory-ordering guarantee. And **closing** a channel happens-before a receive that returns the zero-value-because-closed — so if you close a channel *after* writing some shared state, a receiver that observes the close is guaranteed to see that state. These are not academic curiosities; they are the rules that make the `done`-channel cancellation idiom and the `WaitGroup`-then-`close` shutdown pattern provably correct rather than "works on my machine."

The contrast with shared-memory concurrency is the whole point of CSP. In a lock-based design you have *N* goroutines, *one* piece of shared memory, and a mutex whose job is to ensure that the *N* accesses to that one location are properly ordered. You reason about who holds the lock, in what order locks are acquired (to avoid the deadlock from the [deadlock post](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them)), and whether every access path takes the lock. In a channel-based design you have *N* goroutines and *no shared memory at all* — the data is copied from one goroutine's stack to another's at each send, so there is exactly one writer and exactly one reader of any given value, and they never overlap in time. The race is not *prevented*; it is *structurally impossible*, because the thing a race needs — two goroutines accessing the same memory concurrently with at least one writing — never exists. That is what Hoare's slogan means operationally: you do not synchronize access to shared memory, you avoid sharing the memory by passing it.

## Unbuffered channels: the synchronous rendezvous

An unbuffered channel — `make(chan int)`, capacity zero — has no place to *store* a value. So a send and a receive must meet in time. When you write `ch <- v` on an unbuffered channel, the sending goroutine **blocks** until some other goroutine executes `<-ch`. At that instant the value is copied directly from sender to receiver, and both proceed. This is a *rendezvous*: a synchronization point where two independent timelines briefly touch.

The blocking is the feature, not a bug. An unbuffered send is a guarantee that "by the time my send returns, someone has taken this value." That makes unbuffered channels perfect for handoff and for signaling completion. Consider the classic ping-pong:

```go
func main() {
	ch := make(chan string) // unbuffered
	go func() {
		fmt.Println(<-ch) // receive blocks until main sends
		ch <- "pong"      // send blocks until main receives
	}()
	ch <- "ping"          // blocks until the goroutine receives
	fmt.Println(<-ch)     // blocks until the goroutine sends
}
```

Walk the order. `main` reaches `ch <- "ping"` and blocks because nobody is receiving yet. The goroutine reaches `<-ch`, the rendezvous completes, `"ping"` is handed over, and *both* continue: `main` moves on to `<-ch` and blocks again, while the goroutine prints `ping` and reaches `ch <- "pong"`. They rendezvous a second time, `"pong"` flows back, and the program prints `pong`. Every send is paired with a receive in lockstep. There is no buffer, no queue, no possibility of "I sent it but you haven't gotten to it yet." The two goroutines are tightly coupled in time.

#### Worked example: why an unbuffered send is a barrier

Suppose worker goroutine W computes a result and does `done <- result` on an unbuffered `done` channel, and the main goroutine does `r := <-done`. Here is the exact interleaving the runtime allows, step by step:

1. W finishes its computation, writing `result = 144` into its own stack and into the heap object it built.
2. W executes `done <- 144`. No receiver is ready, so W parks on the sender queue. The scheduler picks another goroutine.
3. Later, `main` executes `r := <-done`. Now there *is* a waiting sender. The runtime copies `144` directly into `main`'s `r`, marks W runnable again, and returns.
4. Because the send happened-before the receive, every memory write W made in step 1 — including the heap object — is guaranteed visible to `main` in step 4.

That last point is the payoff. `main` can read the heap object W built and is guaranteed to see the *final*, fully-written version, with no torn reads and no stale cache lines, **without any explicit synchronization on that object**. The unbuffered channel did the ordering. Contrast this with the lock-based world of [mutual exclusion and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections), where you would protect the shared object with a mutex and reason about who holds it when. With a channel, the object is *owned* by exactly one goroutine at a time, and ownership transfers atomically at the rendezvous. There is nothing to lock because there is nothing shared.

The cost of an unbuffered channel is exactly that coupling: the sender cannot get ahead of the receiver, not even by one item. If the receiver is slow, every send waits. That is sometimes precisely what you want (it is automatic backpressure — see below) and sometimes a throughput killer. Which brings us to buffering.

## Buffered channels: decoupling up to a capacity

A buffered channel — `make(chan int, 4)`, capacity four — has an internal ring buffer that can hold up to four values. Now the rules change. A send succeeds *immediately* if there is room in the buffer; the value is copied into the ring and the sender moves on without waiting for a receiver. A send only **blocks when the buffer is full**. Symmetrically, a receive succeeds immediately if the buffer is non-empty, and only blocks when the buffer is empty.

So a buffered channel **decouples** sender and receiver up to its capacity. The sender can race ahead by up to N items before it has to wait for the consumer to catch up. This smooths out bursts: if your producer emits in clumps but your consumer drains steadily, a buffer absorbs the clumps. The matrix below lays the two kinds side by side along the axes that actually matter when you choose one — capacity, the exact condition under which a send blocks, whether the parties are decoupled, and the use it fits.

![a comparison table contrasting unbuffered and buffered channels across capacity, the condition that blocks a send, whether they decouple the parties, and their best use](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-4.png)

A subtlety worth internalizing: **a buffered channel does not make sends "non-blocking."** It makes them *less often* blocking. The moment the buffer fills — because the consumer fell behind — the next send blocks just like an unbuffered one. People reach for a big buffer hoping to "never block," and what they actually build is a large queue that hides a throughput mismatch until it overflows into latency. If your producer is permanently faster than your consumer, no finite buffer saves you; you have a capacity problem, not a buffering problem. A buffer buys you *slack for transient bursts*, not a license to ignore backpressure.

Here is the same producer-consumer with a buffer, and a measurement intuition:

```go
func main() {
	ch := make(chan int, 64) // buffer of 64
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for v := range ch {
			_ = v * v // consume
		}
	}()
	for i := 0; i < 1_000_000; i++ {
		ch <- i // blocks only when 64 items are unconsumed
	}
	close(ch)
	wg.Wait()
}
```

With capacity 64, the producer can stay 64 items ahead of the consumer, which means the two goroutines hand off in larger batches and the scheduler parks/unparks them far less often than the unbuffered version would. Each park-and-wake of a goroutine costs on the order of a microsecond (a scheduler operation plus, often, a cache-cold resume), so cutting the number of handoffs cuts overhead. We will put real numbers on the buffered-versus-unbuffered gap in the measurement section; the headline is that for high-frequency, small-payload handoffs, a modest buffer can lift throughput several-fold, and beyond a point (once you are no longer scheduler-bound) more buffer does nothing but cost memory.

One more rule that bites beginners: the **zero value of a channel is `nil`**, and `nil` channels block *forever* on both send and receive. `var ch chan int` (never `make`d) is nil; sending or receiving on it parks the goroutine permanently. This looks like a bug, and usually is — but it is occasionally a *feature*: setting a channel variable to `nil` inside a `select` disables that case, which is a clean way to "turn off" one branch of a multi-way wait. Hold that thought for the select section.

There is a deeper design lesson in buffering that is easy to miss. A buffer's capacity is, quietly, a *policy decision about how much work-in-flight you are willing to tolerate*. Capacity zero says "I want a strict handshake; no work may be in flight that the next stage hasn't accepted." Capacity *N* says "I will tolerate up to *N* units of buffered, not-yet-processed work before I throttle the producer." That number is a real engineering parameter with a cost: every buffered item is memory you are holding and latency you are adding (an item sitting in slot 50 of a 64-slot buffer waits for 50 ahead of it to drain before anyone looks at it). When you write `make(chan T, 64)`, you are not "optimizing" — you are declaring a bound on in-flight work, and you should be able to defend the 64. The honest default for a channel you are unsure about is **unbuffered** (force yourself to confront the rate mismatch immediately) or a small buffer sized to the number of consumers; a large buffer should be a deliberate, measured choice, never a reflex. We will see in the measurement section exactly where buffering stops helping and starts merely hiding problems.

## Goroutines: lightweight tasks the runtime multiplexes

A goroutine is not an OS thread. It is a function scheduled by the Go runtime onto a small pool of OS threads. You start one with the word `go` in front of a call: `go worker(jobs, results)`. The cost of starting a goroutine is tiny — a small heap allocation for its stack and a couple of pointer updates — and the initial stack is about 2 KB, growing and shrinking on demand. An OS thread, by contrast, reserves on the order of a megabyte of stack and costs a system call to create. That difference is why you can have a *million* goroutines but not a million threads.

The runtime uses an M:N scheduler: M goroutines multiplexed onto N OS threads (N defaults to the number of CPUs, `GOMAXPROCS`). When a goroutine blocks on a channel, a system call, or a mutex, the runtime parks it and runs another goroutine on the same OS thread — no kernel context switch needed for the channel case, just a userspace stack swap costing tens of nanoseconds rather than the ~1–5 microseconds of a kernel thread switch. This is the same idea as the event loop and coroutines from [async/await and how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work), but Go hides it inside the runtime: you write straight-line blocking code, and the runtime turns every blocking channel operation into a cooperative yield point. You never see a callback; you never color a function (the [function coloring](/blog/software-development/concurrency/function-coloring-and-bridging-sync-and-async) problem simply does not arise in Go).

#### Worked example: a million goroutines, almost free

Start a million goroutines that each block on a channel, and the program uses a few gigabytes of memory (a million stacks at ~2 KB-plus each) but consumes almost no CPU, because blocked goroutines are off the run queue entirely:

```go
func main() {
	const n = 1_000_000
	done := make(chan struct{})
	var ready sync.WaitGroup
	ready.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			ready.Done()
			<-done // park here, costing no CPU
		}()
	}
	ready.Wait()
	fmt.Println("a million goroutines are parked")
	close(done) // wake them all
}
```

Try the equivalent with a million OS threads and you will exhaust address space or hit the kernel's thread limit long before you finish. The lesson is not "goroutines are magic" — it is that a *userspace* scheduler over a small thread pool changes the economics. A goroutine is cheap enough that the idiomatic Go answer to "should I reuse this worker or spawn a new one?" is usually *just spawn one*, because spawning costs less than the bookkeeping of a pool — until you need to *bound* concurrency, which is exactly what worker pools are for, and which we build at the end.

Other languages reach the same destination by different roads. Kotlin coroutines are also cheap userspace tasks scheduled on a thread pool, suspended at `suspend`-function call sites. Rust's `async` tasks on a `tokio` runtime are state machines polled by an executor. Java's Project Loom virtual threads are the JVM's version of goroutines — millions of them, parked cheaply, multiplexed onto carrier threads. The CSP *channel* idiom layers on top of whichever lightweight-task primitive the language gives you.

There is one scheduler detail worth knowing because it explains a real class of "my program froze" reports. Go's scheduler is *cooperative with preemption help*: a goroutine yields at function calls, channel operations, and (since Go 1.14) at safepoints the runtime can inject asynchronously, so a tight CPU loop with no function calls can still be preempted. But a goroutine that makes a **blocking system call** — a synchronous file read, a CGo call into a C library that blocks — takes its OS thread *with it* into the kernel. The runtime notices (a background monitor thread, the *sysmon*, watches for threads stuck in syscalls) and spins up or hands off another OS thread so the other goroutines keep running, but if you make enough simultaneous blocking syscalls you can exhaust the thread pool and stall. This is why "wrap blocking calls so the scheduler can see them" matters, and why a channel-based design that keeps blocking confined to a bounded worker pool is more robust than one that fires off an unbounded number of goroutines each making a blocking call. The pool bounds not just your logical concurrency but your OS-thread pressure.

A second economic point: because a goroutine's stack starts tiny and grows by *copying* (the runtime allocates a bigger stack and relocates the goroutine onto it when it would overflow), deeply recursive goroutines pay a copy cost, and a goroutine that briefly needs a big stack keeps it until it shrinks at the next GC. None of this matters for the typical few-frame worker, but it is why "a goroutine is free" is an approximation, not a law — the right framing is "a goroutine is cheap enough that bounding their *count* matters far more than bounding their *creation rate*."

## The select statement: waiting on many channels at once

A goroutine often needs to wait not on one channel but on *several* — a work channel, a cancellation signal, a timeout — and react to whichever is ready first. That is what `select` does. It is the channel analogue of the OS `select`/`epoll` call that drives an event loop: block on a set of operations, wake on the first ready, with a way to say "if none is ready right now, do something else instead."

```go
select {
case job := <-jobs:
	process(job) // a job arrived
case <-done:
	return // someone asked us to stop
case <-time.After(2 * time.Second):
	log.Println("idle for 2s") // timeout fired
}
```

The semantics are precise and worth stating exactly. `select` evaluates all its cases. If **exactly one** case is ready (its channel op can proceed without blocking), that case runs. If **several** are ready, the runtime picks one **uniformly at random** — this is deliberate, and it prevents starvation, so a busy `jobs` channel cannot perpetually shut out the `done` case. If **none** is ready and there is **no `default`**, the `select` blocks until one becomes ready. If there is a `default`, and no case is ready, the `default` runs immediately — making the whole `select` non-blocking. The timeline below shows the common shape: a `select` parks on three channels, and the first one to become ready fires while the others stay pending.

![a left to right timeline showing a select blocking on three channels where the result channel becomes ready first and fires while the work and done channels remain pending](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-3.png)

Three patterns fall straight out of these rules. The **timeout** pattern uses `time.After`, which returns a channel that delivers a value after the duration; race your real work against it and whichever wins, wins. The **non-blocking** pattern uses `default`: `select { case v := <-ch: use(v); default: }` peeks at a channel without committing to a block — handy for "drain if there's anything, otherwise carry on." And the **disable-a-case** pattern uses the nil-channel rule: assign `nil` to a channel variable used in a `select` case and that case can never be ready, so it is effectively switched off until you reassign it.

How does `select` actually pick, mechanically? The runtime locks all the involved channels in a fixed address order (to avoid a lock-ordering deadlock — see [deadlock and the four conditions](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them)), scans the cases for one that can proceed, and if it finds several, uses a fast pseudo-random permutation to choose. If none can proceed and there is no `default`, it enqueues the goroutine on the wait queue of *every* channel simultaneously, then unlocks and parks. When any one of those channels later becomes ready, the goroutine is dequeued from *all* of them and resumed on the winning case. That "wait on all, wake on one, cleanly dequeue from the rest" machinery is the heart of `select`, and it is why a single `select` over N channels is cheap and correct where hand-rolling the equivalent with N goroutines and a shared flag would be a race waiting to happen.

Other languages spell it differently but mean the same thing. Rust's `crossbeam` crate provides a `select!` macro over its channels; `tokio::select!` does it for async futures. Clojure's `core.async` has `alt!` and `alts!`. Kotlin has `select { onReceive(...) }` over its channels. The primitive — *first-ready-of-many, with a default and a timeout* — is universal to CSP-style systems.

A common mistake with `select` is worth a direct warning, because it produces a subtle leak. The `time.After` timeout case allocates a *new* timer every time the surrounding loop iterates, and that timer is not garbage-collected until it fires, even if a different case won. In a hot loop that selects on `work` and `time.After(time.Minute)` and almost always takes the `work` case, you accumulate up-to-a-minute's worth of pending timers — a slow leak that a profiler will eventually surface as a pile of `time.Timer` objects. The fix is to hoist the timer out of the loop and `Reset` it (or use `time.NewTimer` and `Stop`/`Reset` explicitly), so there is exactly one timer that you re-arm rather than a fresh one per iteration. The general principle: anything a `select` case *allocates* is allocated on *every* evaluation of the `select`, whether or not that case wins, so keep per-iteration allocation out of `select` cases.

A second `select` subtlety is the **non-blocking send** and its danger. `select { case ch <- v: ; default: }` tries to send and, if no receiver is ready and the buffer is full, falls through to `default` and *drops the value*. This is the right primitive for a metrics or logging channel where dropping under overload is preferable to blocking the hot path — but it is exactly wrong for data you must not lose, and the bug is silent: under load the channel fills, the `default` fires, and your data quietly vanishes. Use non-blocking send only where dropping is the *designed* behavior, and say so in a comment, because the next reader will assume the send always lands.

## Fan-out and fan-in: building pipelines

Once you have channels and cheap tasks, a *pipeline* is the natural way to structure concurrent work: a series of stages connected by channels, where each stage is a goroutine (or a set of them) that receives from an input channel, does one transformation, and sends to an output channel. The shape that earns its keep is **fan-out/fan-in**: one stage *fans out* its work to several identical worker goroutines reading the same input channel, and a *fan-in* stage merges their outputs back into one channel. The graph below shows the topology — a source feeding three workers that all merge into one sink.

![a pipeline graph showing a source node fanning jobs out to three parallel worker nodes whose outputs merge into one sink node that collects all results](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-2.png)

Fan-out is trivial because *multiple goroutines can receive from the same channel*, and the runtime hands each value to exactly one of them. So to parallelize a stage, you just start more goroutines reading the same input:

```go
// gen emits the numbers 1..n on a channel, then closes it.
func gen(nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for _, n := range nums {
			out <- n
		}
	}()
	return out
}

// sq squares every value it receives. Fan-out = start several of these.
func sq(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			out <- n * n
		}
	}()
	return out
}
```

Fan-in — merging several channels into one — needs a little more care, because you must close the merged output *exactly once*, *after* every input is drained. The canonical Go merge uses a `sync.WaitGroup` to count the input goroutines and a single closer:

```go
func merge(cs ...<-chan int) <-chan int {
	out := make(chan int)
	var wg sync.WaitGroup
	wg.Add(len(cs))
	for _, c := range cs {
		go func(c <-chan int) {
			defer wg.Done()
			for v := range c { // drain this input fully
				out <- v
			}
		}(c)
	}
	go func() {
		wg.Wait()  // after ALL inputs are drained...
		close(out) // ...close the merged output exactly once
	}(  )
	return out
}
```

Now the full pipeline reads like a sentence: `merge(sq(in), sq(in), sq(in))` fans the squaring stage out to three workers and fans their results back in. Each value flows source → one worker → merge → sink, and only one goroutine ever owns a given value. No locks, no shared counters, no races — the structure itself enforces single ownership.

There is a classic mechanistic point hiding here. Fan-out gives you *load balancing for free*: because each worker pulls the next job only when it finishes its current one, a fast worker naturally grabs more jobs than a slow one. You do not partition the work up front and risk one worker getting all the hard jobs; you let the workers *pull* at their own pace. This is the same self-balancing property that makes the work-stealing schedulers of [data parallelism, fork/join, and work stealing](/blog/software-development/concurrency/data-parallelism-fork-join-and-work-stealing) effective, expressed at the channel level. The shared input channel is, in effect, a tiny work queue.

#### Worked example: pull-balancing beats push-partitioning

Say you have 1000 jobs and 4 workers, and the jobs vary wildly in cost: 990 of them take 1 ms and 10 of them take 100 ms, randomly ordered. Two strategies:

1. **Push-partition up front.** Split the 1000 jobs into four slices of 250 and hand one slice to each worker. By bad luck, all ten 100 ms jobs land in worker 3's slice. Worker 3 now has 240 ms of light work plus 1000 ms of heavy work, while the other three finish their ~250 ms slices and sit idle. Total time is bounded by the unluckiest worker: roughly 1240 ms, with three cores idle for most of the second half.
2. **Pull from a shared channel (fan-out).** Put all 1000 jobs on one channel; each worker pulls the next job when it finishes the last. The heavy jobs are spread across whichever workers happen to be free when they come up, and no worker is idle while jobs remain. Total time is close to (total work / 4) — about `(990 + 1000)/4 ≈ 500` ms — roughly **2.5x faster**, with no idle cores until the very end.

The pull model self-balances because *demand-driven dispatch* matches each job to a worker that is actually free, exactly when it is free. You did not write any balancing logic; the shared channel *is* the balancer. The only cost is the per-pull synchronization on the channel, which for jobs measured in milliseconds is utterly negligible. This is why "many small jobs on one channel, several workers pulling" is the default Go concurrency shape, and why you almost never want to partition work statically unless the jobs are uniform.

## Cancellation: the done-channel and context

Pipelines have a dark side: **goroutine leaks**. If a downstream stage stops reading — because the consumer found what it wanted and quit, or because an error aborted the run — every upstream goroutine blocked on a send into a now-unread channel is stuck *forever*. It will never be garbage-collected (a goroutine is a GC root), so its stack, and everything its closures reference, leaks. In a long-running server, leaked goroutines accumulate until the process dies. The figure contrasts the bug with the fix: a goroutine blocked on a send with no receiver, versus the same goroutine made cancellable.

![two side by side states showing a goroutine that blocks forever on a send with no receiver leaking memory, versus the same goroutine selecting on its result and a done channel so a context cancel returns it cleanly](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-6.png)

The fix is a **cancellation channel** that the goroutine watches alongside its real work. The convention is a `done` (or `quit`) channel of `chan struct{}`; *closing* it broadcasts the signal, because — by the closed-channel rules in the next section — a receive on a closed channel returns immediately for *every* receiver. So one `close(done)` wakes all the goroutines waiting on it at once.

```go
func sq(done <-chan struct{}, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			select {
			case out <- n * n: // normal: emit the result
			case <-done:       // cancelled: stop, don't leak
				return
			}
		}
	}()
	return out
}
```

The key line is the `select`. Every send is now `select { case out <- v: ; case <-done: return }`, so a blocked send can *always* be unblocked by closing `done`. That single change converts a leak-prone pipeline into one that tears down cleanly on cancellation.

Go standardizes this pattern as `context.Context`. A `context` carries a cancellation signal (its `Done()` method returns a channel that is closed when the context is cancelled, times out, or its deadline passes), plus optional deadline and request-scoped values. The idiom threads a `ctx` through every function in a call tree, and every blocking operation selects on `ctx.Done()`:

```go
func sq(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			select {
			case out <- n * n:
			case <-ctx.Done(): // cancelled, timed out, or deadline hit
				return
			}
		}
	}()
	return out
}

// Caller controls the lifetime:
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel() // always cancel to release resources, even on success
```

Calling `cancel()` (or hitting the timeout) closes `ctx.Done()`, which every selecting goroutine sees, so the whole tree unwinds. The `defer cancel()` is not optional — forgetting it leaks the context's own timer goroutine. Cancellation in CSP is *cooperative*: there is no "kill this goroutine" primitive (and there should not be — killing a goroutine mid-mutation would corrupt state). Instead, every goroutine periodically checks the done signal and returns on its own. This is the same cooperative-cancellation discipline that structured concurrency formalizes, and it is the right one: a task gets to clean up after itself.

Rust expresses cancellation through dropping: a `tokio` task is cancelled by dropping its `JoinHandle` or via a `CancellationToken` that futures `.await` on. Kotlin coroutines are cancelled by cancelling their `Job`/`CoroutineScope`, and well-behaved suspend functions check for cancellation at each suspension point. Clojure's `core.async` uses a "kill channel" closed to signal `go` blocks to stop. The shape repeats: **a closed signal channel that every task watches.**

Here is the Kotlin version of a cancellable pipeline stage, to show how structured concurrency makes the teardown automatic rather than manual:

```kotlin
fun CoroutineScope.square(input: ReceiveChannel<Int>) = produce {
    for (n in input) {          // iterates until input closes or we're cancelled
        send(n * n)             // suspends if the buffer is full
    }
    // No explicit close: `produce` closes its channel when the block returns,
    // and a parent-scope cancellation makes the `for` loop throw and unwind.
}
```

The difference from Go is instructive. In Go you thread a `context` and `select` on `Done` by hand at every blocking point; the discipline is yours to keep. In Kotlin's structured concurrency, the *scope* owns the children: cancel the scope and every coroutine launched in it is cancelled, the suspending `send`/`for` throws a `CancellationException`, and the stack unwinds through normal exception handling — you cannot forget to propagate it, because the runtime does it. Both reach leak-free cancellation; Go makes the cancellation channel explicit and visible, Kotlin makes it structural and automatic. That trade — explicit-and-visible versus structural-and-automatic — recurs across every concurrency model, and structured concurrency (the subject of the series' later posts) is the argument that automatic is worth giving up some visibility for.

## Closing channels and the closed-channel rules

Closing is where channel code most often goes wrong, so the rules deserve to be stated exactly. `close(ch)` marks a channel closed. After that:

- **Receiving from a closed channel never blocks.** It returns immediately. If there are still buffered values, you get them in order; once drained, every further receive returns the **zero value** of the element type. The two-value receive form tells them apart: `v, ok := <-ch` sets `ok` to `false` once the channel is closed *and* drained. This is exactly what lets `for v := range ch` terminate — `range` stops when the channel is closed and empty.
- **Sending to a closed channel panics.** `ch <- v` on a closed channel is a `panic: send on closed channel`. There is no recovering from this in the normal flow; it is a programming error.
- **Closing a closed channel panics.** A second `close(ch)` is a `panic: close of closed channel`.
- **Closing a nil channel panics**, and as noted, send/receive on a nil channel blocks forever.

From these four facts comes the governing convention, and you should tattoo it on the inside of your eyelids: **only the sender closes a channel, and only when no more sends will happen.** A receiver must never close — it cannot know whether the sender is done, and closing under the sender's feet causes a send-on-closed panic. When *multiple* goroutines send to one channel, none of them may close it unilaterally; instead a coordinator closes it after all senders have finished (the `WaitGroup`-then-`close` pattern from our `merge`). The matrix below summarizes how each operation behaves on an open versus a closed channel, so you can see the panics and the zero-value rule at a glance.

![a table showing how send, receive, close, and a select case each behave on an open channel versus a closed channel, including the send on closed panic and the zero value with ok false on receive](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-5.png)

#### Worked example: the "who closes?" decision

You have three producer goroutines feeding one `chan int`, and one consumer ranging over it. Who closes?

- *Not a producer.* If producer 1 closed when it finished, producers 2 and 3 would panic on their next send. Wrong.
- *Not the consumer.* The consumer cannot know all producers are done, and closing would race against in-flight sends. Wrong.
- *A coordinator.* Wrap the producers in a `WaitGroup`, and in a separate goroutine do `wg.Wait(); close(ch)`. The close happens strictly after the last send, so no send ever races the close, and the consumer's `range` ends cleanly.

This is the exact structure of the `merge` function earlier, and it generalizes: the right to close belongs to whoever can prove "no more sends will occur," which with multiple senders is a coordinator, never an individual sender. Get this wrong and you ship a panic that fires only under the precise timing where a producer sends just after another closes — a heisenbug that passes every test and crashes in production at peak.

A handy special case: when a channel is used *purely as a broadcast signal* (the `done` channel), you never send on it at all — you only ever `close` it. Closing is the broadcast: every receiver waiting on `<-done` wakes at once, and any number of goroutines can `<-done` safely. This is why `chan struct{}` plus `close` is the idiomatic cancellation broadcast.

## Deadlock by channel: "all goroutines are asleep"

Channels remove data races, but they do not remove deadlocks. A channel deadlock is a *circular wait*: every goroutine is blocked on a channel operation that only some *other* blocked goroutine could satisfy, so nobody can move. The simplest possible case is a single goroutine sending on an unbuffered channel with nobody receiving:

```go
func main() {
	ch := make(chan int) // unbuffered
	ch <- 1              // blocks forever: no receiver exists
	fmt.Println(<-ch)    // unreachable
}
```

Run it and Go prints the unmistakable:

```bash
fatal error: all goroutines are asleep - deadlock!
```

How does the runtime *know*? This is a genuinely elegant piece of engineering. The Go scheduler tracks how many goroutines exist and how many are runnable. Every time it parks a goroutine (on a channel, a mutex, a `WaitGroup`), it decrements the count of runnable goroutines. When that count hits **zero** while goroutines still *exist* — every goroutine is parked, none is runnable, and no timer or network poller is pending to wake anyone — the runtime concludes that nothing can ever make progress and aborts with the fatal deadlock error. It is not a heuristic; for the pure-channel case it is a sound detection: if all goroutines are blocked on channels and none can be woken, the program is provably stuck. The figure shows the shape: two goroutines each blocked on a channel the other was supposed to feed, which the runtime then detects as all-asleep.

![a graph showing the main goroutine blocked sending on channel A and a worker goroutine blocked receiving on channel B where neither channel will ever be fed, leading the runtime to report all goroutines asleep](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-7.png)

The crucial caveat: this detection only fires when *literally every* goroutine is asleep. A *partial* deadlock — where two goroutines are stuck in a channel cycle but a third goroutine (or the network poller, or a timer) is still alive — will **not** be reported. The program will simply leak the two stuck goroutines and run on, slowly bleeding memory, with no error at all. This is why "it didn't deadlock in testing" proves nothing about a real server: in a server there is *always* something runnable (the accept loop, a health-check ticker), so the all-asleep detector never trips even when half your worker goroutines are wedged. You catch partial deadlocks with goroutine-leak detection and dumps (`runtime.NumGoroutine()` climbing without bound; a `pprof` goroutine profile showing many goroutines parked at the same channel op), the techniques covered in [finding concurrency bugs with race detectors and stress testing](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing).

It is worth contrasting this with how *other* CSP systems handle the same hazard, because it reveals a genuine design trade-off. Rust's standard channels do **not** have a runtime all-asleep detector — if every thread blocks on a channel, the process simply hangs, and you diagnose it with a debugger or a thread dump (`SIGQUIT` to dump stacks). What Rust gives you instead is a *compile-time* guarantee about data races (the `Send`/`Sync` system), trading runtime deadlock detection for static data-race prevention. Erlang/Elixir take yet another stance: a process blocked in `receive` forever is not a crash, it is just an idle process, and the *supervisor* model is the answer — a supervisor restarts a process that has wedged or crashed, so the system heals rather than detecting the wedge. Three runtimes, three philosophies: Go *detects* the total deadlock and aborts loudly, Rust *prevents* the related race at compile time and lets a deadlock hang, Erlang *recovers* from a wedged process by restarting it. None is strictly best; they reflect different bets about whether a stuck program should crash, hang, or self-heal. Knowing which bet your runtime made tells you which diagnostic to reach for at 3 AM.

The practical upshot for Go specifically: do not rely on the all-asleep detector as your deadlock safety net. It catches exactly the *unit-test* and *small-program* case — which is genuinely useful, because it turns a whole category of beginner mistakes into an instant, unambiguous error rather than a silent hang — but it is structurally blind to the production case where a server's main loop keeps the runtime busy. Treat a clean run as "no *total* deadlock," not "no deadlock," and lean on goroutine-count monitoring and periodic goroutine-profile snapshots for the partial deadlocks that actually page you.

#### Worked example: the unbuffered-handoff deadlock

A frequent real bug: send a result on an unbuffered channel *before* starting the receiver.

```go
func main() {
	results := make(chan int) // unbuffered
	results <- compute()      // (1) blocks: no receiver yet
	go consume(results)       // (2) never reached
}
```

Line (1) blocks forever because the receiving goroutine on line (2) has not been started — and never will be, because line (1) never returns. The whole program is one goroutine, parked, so the all-asleep detector fires immediately. The fixes are instructive because each maps to a concept above: (a) **start the receiver first** (`go consume(results)` before the send), so a receiver is ready at the rendezvous; or (b) **buffer the channel** (`make(chan int, 1)`), so the send has a slot and returns without a receiver. Option (a) keeps the rendezvous guarantee; option (b) decouples and removes it. Choosing between them *is* the unbuffered-versus-buffered decision, now with real stakes.

## Worker pools with backpressure

We can now assemble the centerpiece. A **worker pool** bounds concurrency: instead of spawning one goroutine per job (which, for jobs that hold a scarce resource — a database connection, a file handle, an outbound API rate limit — would blow that resource), you spawn a *fixed* number of workers and feed them through a channel. The channel does double duty: it distributes jobs (fan-out) and it provides **backpressure** — when all workers are busy and the job channel's buffer is full, the producer's next send blocks, automatically throttling the producer to the rate the workers can sustain.

```go
func pool(numWorkers int, jobs <-chan Job, results chan<- Result) {
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func() {
			defer wg.Done()
			for job := range jobs { // each worker pulls the next job
				results <- process(job)
			}
		}()
	}
	go func() {
		wg.Wait()      // all workers finished (jobs channel closed)
		close(results) // safe single close after the last send
	}()
}
```

The producer feeds `jobs` and closes it when done; the workers `range` over `jobs`, so they all exit when it is closed and drained; the coordinator closes `results` after every worker exits. Bounded workers, fan-out load balancing, fan-in via the shared `results` channel, clean shutdown via close — every idea in this post in fifteen lines.

The backpressure is the part people miss. If `jobs := make(chan Job, numWorkers)` (a small buffer) and the producer is faster than the workers, the producer fills the buffer, then *blocks on its next send* until a worker frees a slot. The system self-throttles. Compare this to the alternative of `go process(job)` per job with no bound: under a load spike you spawn ten thousand goroutines, ten thousand database queries, and either tip the database over or exhaust memory. The bounded pool degrades gracefully — it gets *slower* (the producer waits) rather than *falling over*. This is exactly the [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) principle from message queues, realized in-process with a channel; the [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) post in system design frames it at the architecture scale. The channel *is* your in-process bounded queue, and its capacity *is* your concurrency-limited admission control.

How many workers? That is a tuning question with a real answer, covered in the measurement section, but the shape of the answer is: for CPU-bound work, around `GOMAXPROCS` (one busy worker per core); for I/O-bound work, many more, because each worker spends most of its time blocked on I/O and contributes little CPU, so you need enough of them to keep the CPUs fed while others wait. Sizing the pool to the *nature of the work* is the whole game.

## Worked example: a bounded pipeline with cancellation

Let us pull every thread together into one program: a bounded worker pool, wired as a fan-out/fan-in pipeline, that respects a context deadline and cancels cleanly without leaking a single goroutine. This is the artifact the whole post was building toward.

```go
package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// stage 1: produce jobs until cancelled or exhausted.
func produce(ctx context.Context, n int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for i := 1; i <= n; i++ {
			select {
			case out <- i: // emit job i
			case <-ctx.Done(): // cancelled: stop producing
				return
			}
		}
	}()
	return out
}

// stage 2: a pool of `workers` goroutines, each squaring jobs.
func square(ctx context.Context, in <-chan int, workers int) <-chan int {
	out := make(chan int)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for n := range in { // fan-out: each worker pulls
				time.Sleep(time.Millisecond) // pretend it's work
				select {
				case out <- n * n: // emit result
				case <-ctx.Done(): // cancelled mid-send: bail
					return
				}
			}
		}()
	}
	go func() {
		wg.Wait()  // all workers done...
		close(out) // ...close fan-in output exactly once
	}()
	return out
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel() // release the timer goroutine no matter what

	jobs := produce(ctx, 1000)
	results := square(ctx, jobs, 4) // bounded: exactly 4 workers

	sum, count := 0, 0
	for r := range results { // drains until square closes `out`
		sum += r
		count++
	}
	fmt.Printf("processed %d jobs before the 50ms deadline; sum=%d\n", count, sum)
}
```

Trace what happens at the deadline. After 50 ms, `ctx`'s `Done()` channel closes. Every goroutine that is parked on a `select` — the producer trying to emit, any worker trying to send a result — sees `<-ctx.Done()` become ready and `return`s. The workers' deferred `wg.Done()` fire, the `WaitGroup` reaches zero, the closer goroutine closes `out`, and `main`'s `range results` loop terminates. **No goroutine is left blocked**; the entire pipeline unwinds from the cancellation signal outward. Because the workers are bounded to four, the program also never runs more than four units of "work" at once, regardless of how many jobs `produce` was willing to emit — the `in` channel and the pool provide the backpressure. Run it and you will see it processed however many jobs fit in 50 ms (a few dozen, given the artificial 1 ms per job across four workers), then stopped cleanly. Remove the `<-ctx.Done()` cases and instead let the consumer quit early, and you would leak every worker still trying to send into an unread `out` — the exact leak from the cancellation section.

Here is the analogous pipeline in **Rust** with `crossbeam` channels and a scoped thread pool, to show the same CSP shape in a language with ownership-based safety:

```rust
use crossbeam_channel::{bounded, select, Receiver};
use std::thread;
use std::time::{Duration, Instant};

fn square_pool(input: Receiver<i64>, workers: usize, deadline: Instant) -> Receiver<i64> {
    let (tx, rx) = bounded::<i64>(workers);
    thread::scope(|s| {
        for _ in 0..workers {
            let input = input.clone();
            let tx = tx.clone();
            s.spawn(move || {
                while let Ok(n) = input.recv() {
                    if Instant::now() >= deadline {
                        break; // cooperative cancellation by deadline
                    }
                    select! {
                        send(tx, n * n) -> _ => {},
                        default(Duration::from_millis(1)) => break,
                    }
                }
            });
        }
    });
    rx
}
```

Rust gives you the same fan-out (multiple workers `clone` the receiver and pull from the shared channel), the same fan-in (they `clone` the sender), the same `select!` for "send the result *or* notice we should stop," and the compiler additionally proves at *build time* that no two threads alias the same mutable data — `Send`/`Sync` bounds make a data race a type error, not a runtime hope. The CSP idea is identical; Rust just refuses to compile the unsafe version. **Kotlin** would express the same pipeline with `produce { }` coroutine builders, `Channel`s, and structured `coroutineScope { }` for clean cancellation; **Clojure's** `core.async` would use `chan`, `go` blocks, and `alt!` with a kill channel. The vocabulary changes; the grammar — channels, lightweight tasks, select, fan-out/fan-in, cooperative cancellation — does not.

## Measured: throughput, buffering, and pool sizing

Now the honest part. Concurrency advice is worthless without numbers, and numbers are worthless without saying how they were taken. Everything below is the *shape* you will reliably observe; the exact figures depend on your CPU, your Go version, and your payload, so I give order-of-magnitude ranges and tell you how to measure your own. **Method:** warm up first (run the benchmark once and discard it so the scheduler, the allocator, and the caches are hot), run many iterations, report the median (not the mean — a single GC pause skews the mean), and pin `GOMAXPROCS` so the OS scheduler is not silently changing your core count between runs. Go's `testing.B` benchmarks do most of this for you; use `go test -bench` rather than hand-rolling timers, and run with `-count=10` to see the variance.

The first measurement: **buffered versus unbuffered throughput** for a tight producer-consumer that hands off small integers. The cost being measured is per-handoff scheduler overhead — each time a send finds no waiting receiver, the sender parks and a context switch (userspace) ensues. A buffer lets the sender deposit several items before blocking, amortizing that overhead.

| Channel configuration | Relative handoff throughput | Why |
| --- | --- | --- |
| Unbuffered (cap 0) | 1.0x (baseline) | Every send is a rendezvous; sender parks if receiver isn't waiting |
| Buffered (cap 1) | ~1.2–1.5x | One slot of slack reduces some park/wake cycles |
| Buffered (cap 64) | ~2–4x | Sender stays ahead in batches; far fewer park/wake events |
| Buffered (cap 4096) | ~2–4x (no better than 64) | Already scheduler-overhead-free; extra capacity only costs memory |

The lesson lives in the last two rows: throughput rises with buffering up to a point, then **flattens**. Once the buffer is big enough that the sender rarely blocks, you have eliminated the scheduler overhead, and more capacity buys nothing but memory and worse latency-under-overflow. There is no magic number — a small buffer (capacity equal to the number of consumers, or a few dozen for a high-rate stream) captures most of the win. A megabyte-sized channel is almost always a smell: it is masking a producer-consumer rate mismatch that a buffer cannot actually fix.

The second measurement: **pool sizing**. Take a fixed batch of jobs and vary the worker count, measuring total wall-clock time. Two regimes, and they look completely different.

| Workers | CPU-bound jobs (8-core box) | I/O-bound jobs (each blocks ~10 ms) |
| --- | --- | --- |
| 1 | 1.0x (baseline time) | 1.0x |
| 4 | ~3.5–4x faster | ~4x faster |
| 8 | ~6–7x faster (near core count) | ~8x faster |
| 16 | ~6–7x (no gain; oversubscribed) | ~15x faster |
| 64 | slightly *slower* (context-switch + cache thrash) | ~30–50x faster (until the I/O resource saturates) |
| 1000 | much slower (scheduling overhead dominates) | gains flatten; the downstream resource is the limit |

Read the two columns against each other and the whole worker-sizing rule falls out. **CPU-bound work** peaks at roughly one worker per core (`GOMAXPROCS`); beyond that, extra workers do not get extra CPUs, they just add context-switch and cache-eviction overhead, so the curve flattens and then *bends down* — the same Amdahl-and-contention story from [concurrency versus parallelism and the scaling laws](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws). **I/O-bound work** keeps scaling far past the core count, because each worker spends ~99% of its time parked on I/O contributing no CPU, so you need *many* workers to keep the cores fed — until the downstream resource (the database's connection limit, the API's rate limit) saturates, at which point adding workers just deepens the queue. The pool size you want is "enough to saturate the bottleneck resource, and not one more." Finding that number is an experiment, not a formula, but the experiment is fifteen minutes with `go test -bench` and a worker-count parameter.

A third, sharper measurement worth taking yourself: **the latency cost of an oversized buffer.** Put a 100,000-slot buffer on a channel whose consumer is 10% slower than the producer, and watch end-to-end latency climb steadily as the buffer fills — by the time an item is consumed, it sat in the queue for the time it took to drain everything ahead of it (this is just Little's law, $L = \lambda W$: a longer queue $L$ at a fixed arrival rate $\lambda$ means a longer wait $W$). The buffer did not fix the rate mismatch; it converted it from "blocking the producer" into "hidden tail latency," which is strictly worse because it is invisible until a user complains. Bounded buffers make the backpressure *visible*; unbounded ones hide it until it becomes an outage.

To make the buffer-versus-throughput claim concrete, here is the benchmark you would actually write — the thing that produced the shape in the first table, so you can reproduce it on your own hardware instead of trusting my ranges:

```go
func benchHandoff(b *testing.B, capacity int) {
	ch := make(chan int, capacity)
	done := make(chan struct{})
	go func() { // consumer
		for range ch {
		}
		close(done)
	}()
	b.ResetTimer() // discard setup time; this is the warm-up discipline
	for i := 0; i < b.N; i++ {
		ch <- i
	}
	close(ch)
	<-done
}

func BenchmarkUnbuffered(b *testing.B) { benchHandoff(b, 0) }
func BenchmarkBuf64(b *testing.B)      { benchHandoff(b, 64) }
func BenchmarkBuf4096(b *testing.B)    { benchHandoff(b, 4096) }
```

Run it with `go test -bench=. -benchmem -count=10` and read the `ns/op` column. You will see the unbuffered case cost the most nanoseconds per handoff, the capacity-64 case drop sharply, and the capacity-4096 case land right next to capacity-64 — the plateau. The `-count=10` is not optional: a single run can be skewed by a GC pause or a scheduler hiccup, and you want to see the *spread* and take the median. The `-benchmem` flag shows you allocations per operation, which for a pure channel handoff should be zero — if it is not, you are accidentally allocating in the loop and measuring the allocator instead of the channel. This is the honest-measurement discipline in miniature: warm up (`ResetTimer`), repeat (`-count=10`), and check that you are measuring what you think (`-benchmem`).

A word on **platform and nondeterminism**, because it is the difference between a measurement and a number you made up. Channel handoff cost depends on whether the sender and receiver run on the *same* OS thread (a cheap userspace swap) or get scheduled onto *different* cores (a cross-core wakeup with cache effects), and that depends on `GOMAXPROCS`, the OS scheduler, and load. The same benchmark can report meaningfully different numbers on a quiet laptop versus a busy CI box versus a cloud VM with noisy neighbors. So when you report a throughput figure, name the machine, pin `GOMAXPROCS`, run on an otherwise-idle box, and present a range or a median, never a single suspiciously-precise number. A concurrency benchmark that reports "exactly 142.7 ns/op" with no error bars and no platform is a number to distrust.

## When to reach for this (and when not to)

Channels are a beautiful tool and a bad default. Reach for channels and CSP when:

- **You are passing ownership of data between stages.** A pipeline — parse → validate → transform → write — where each item is owned by exactly one stage at a time is the channel sweet spot. The handoff *is* the synchronization; there is nothing to lock.
- **You need to coordinate multiple concurrent activities** — a `select` over work, cancellation, and timeout; a fan-out worker pool; a pub/sub of events. Channels turn "wait for whichever happens first" into three readable lines.
- **You want backpressure for free.** A bounded channel between a fast producer and slower consumers throttles the producer automatically. This is the cleanest in-process flow control there is.
- **You want cancellation and lifetimes to compose.** `context` threading plus `select`-on-`Done` gives you cooperative, leak-free teardown across a whole call tree.

Reach for a **mutex** instead — see [mutual exclusion, mutexes, and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) — when:

- **You are guarding a small piece of shared state with short critical sections** — incrementing a counter, updating a map, reading-and-modifying a struct field. A `sync.Mutex` around three lines is simpler, faster, and more obvious than routing every read and write through a channel and a dedicated owner goroutine. The Go team says this outright: *"Use whichever is most expressive and/or most simple."* A shared cache behind a mutex is not a CSP failure; it is the right call.
- **You need a cross-entity invariant** — "transfer must debit A and credit B atomically." That is a single critical section over two pieces of state; a channel-per-account actor model makes it *harder*, because now you need a transaction protocol across two owners. Lock both, do the transfer, unlock.
- **The contention is low and the state is tiny.** A mutex on an uncontended counter is ~25 nanoseconds; a channel handoff is hundreds of nanoseconds plus a goroutine. Do not pay for a conduit you do not need.

And reach for an **actor** (a goroutine that owns state and processes a mailbox of messages, one at a time) when you have *one logical entity* with state that *many* clients mutate concurrently — a connection's session, a game character, an account's running balance — and you want all access serialized through that one owner without explicit locks. The matrix below compares the three models on the axes that drive the choice: how they communicate, who owns the state, and whether backpressure is built in.

![a table comparing channels, a shared mutex, and the actor model across how they communicate, who owns the state, and whether backpressure is built in](/imgs/blogs/csp-channels-goroutines-and-the-select-statement-8.png)

The anti-patterns are worth naming bluntly. **Do not build a "channel of channels of channels" to avoid a three-line mutex** — over-channeling produces code that is harder to follow than the locks it replaced, with worse performance. **Do not use an unbuffered channel as a shared variable** — if two goroutines both need the current value of something, that is shared state; use an atomic or a mutex. **Do not reach for a giant buffer to "fix" blocking** — that hides a rate mismatch as latency. And **do not forget that channels can still deadlock** — a channel cycle is every bit as fatal as a lock cycle; the [deadlock](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them) post's discipline applies. The honest summary: channels are for *communication and coordination*; mutexes are for *protecting shared state*; reaching for the wrong one makes the code worse. Measure, then choose.

## Case studies / real-world

**The Go pipelines blog post (the canonical reference).** The Go team's official article "Go Concurrency Patterns: Pipelines and cancellation" (blog.golang.org, 2014) is the source of the `gen` → `sq` → `merge` structure and the explicit `done`-channel cancellation idiom used throughout this post. Its central warning is the one we built the cancellation section around: *"goroutines and channels...are easy to leak if you're not careful."* It predates `context` (which later standardized the `done`-channel pattern into the standard library) and remains the clearest worked treatment of why fan-in must close exactly once and how to tear a pipeline down without leaks. If you read one external source after this, read that.

**A real channel deadlock in production tooling.** Channel deadlocks are common enough that Go's own ecosystem has shipped detectors for them. The static analyzer in `go vet` and third-party tools like `dingo-hunter` and the `staticcheck` suite specifically look for the "send with no possible receiver" and "missing close" shapes, because they recur constantly in real codebases — a worker pool where the result channel is never drained on the error path, a `select` missing its `default` that wedges under an unexpected ordering, a `close` that a second goroutine can reach. The pattern is always the same circular wait we diagrammed: someone blocked on a channel that someone else, also blocked, was supposed to feed. The fix is always one of the four moves from the deadlock post — start the receiver first, buffer the channel, add a `default`/timeout to the `select`, or add a cancellation path.

**Clojure's `core.async` and the channels-everywhere experiment.** Rich Hickey's `core.async` (2013) brought CSP channels and a `go`-block macro to the JVM and JavaScript, with `alt!`/`alts!` as its `select`. It is a faithful CSP port — channels, lightweight `go` blocks compiled to state machines, fan-in/fan-out — and the community's hard-won lesson from years of using it mirrors the "when not to" section above: channels shine for *event coordination and decoupling pipeline stages*, and they are overkill (and a debugging burden) when a simple atom or a function call would do. The same conclusion the Go team reached, validated independently on a different runtime: CSP is a coordination tool, not a replacement for ordinary shared state behind a lock.

**Kubernetes, etcd, and the cost of getting `context` right at scale.** Go's largest production systems — Kubernetes, etcd, Docker, the Go standard library's own HTTP server — are built on exactly the channel-and-`context` patterns in this post, and their issue trackers are an honest record of how the idiom behaves under real load. A recurring class of bug across these projects is the *missing `<-ctx.Done()` case*: a worker goroutine that ranges over a channel or blocks on a send without also selecting on the request's context, so that when the client disconnects or the request times out, the goroutine keeps running — doing now-pointless work and, in the worst cases, holding a database transaction or a lock open. The fixes that landed are uniformly the discipline this post prescribes: thread the `context` everywhere, select on `ctx.Done()` at every blocking point, and `defer cancel()` to release the timer. The lesson these systems teach at scale is that the channel primitives are sound, but *cancellation hygiene is a property of the whole call tree* — one stage that forgets to honor the context reintroduces the leak for the entire pipeline. The `errgroup` package (`golang.org/x/sync/errgroup`), now ubiquitous in production Go, exists precisely to make this discipline harder to forget: it bundles a worker pool, a shared context that is cancelled when any worker errors, and a `Wait` that returns the first error — structured concurrency, retrofitted onto channels and `context`.

## Key takeaways

- **A channel is a typed conduit that transfers ownership and establishes a happens-before edge.** Send-then-receive guarantees the sender's prior writes are visible to the receiver — that is why channel-based code is race-free without explicit locks.
- **Unbuffered means rendezvous; buffered means decoupled up to capacity.** An unbuffered send blocks until a receiver is ready (a synchronization guarantee); a buffered send blocks only when the buffer is full (slack for bursts, not a license to ignore rate mismatches).
- **Goroutines are cheap userspace tasks, not OS threads.** A million parked goroutines cost memory but no CPU; spawn freely, and bound concurrency with a pool only when a scarce resource demands it.
- **`select` waits on many channels and fires the first ready one**, picking randomly among ties to prevent starvation; `default` makes it non-blocking and `time.After` gives it a timeout.
- **Fan-out is free load balancing** (workers pull from a shared input at their own pace); **fan-in must close the merged channel exactly once**, after every input drains, via a coordinator.
- **Only the sender closes, and only when no more sends will happen.** Send-on-closed and double-close panic; receive-on-closed returns the zero value with `ok=false`, which is what ends a `range`.
- **Closing a `done`/`context` channel is a broadcast cancel** — every selecting goroutine wakes at once; cancellation in CSP is cooperative, so every blocking op must `select` on `Done`.
- **Channels deadlock too.** The runtime detects only the *all-goroutines-asleep* case; partial deadlocks leak silently, so watch goroutine counts in long-running services.
- **Pick the tool by the job:** channels for communication and coordination, a mutex for protecting small shared state, an actor for one entity many clients mutate. Measure before you choose; do not over-channel.

## Further reading

- **Tony Hoare, *Communicating Sequential Processes* (1978, and the 1985 book).** The original formalism. Dense, but the source of every idea in this post.
- **"Go Concurrency Patterns: Pipelines and cancellation"** (The Go Blog, 2014). The canonical worked treatment of fan-out/fan-in and `done`-channel cancellation.
- **"Go Concurrency Patterns" (Rob Pike, Google I/O 2012 talk)** and its follow-up "Advanced Go Concurrency Patterns" (2013). Where the `select`-driven idioms were popularized.
- ***The Go Programming Language* (Donovan and Kernighan), chapters 8–9.** The clearest book treatment of goroutines, channels, `select`, and the memory model.
- **The `context` package documentation and "Go Concurrency Patterns: Context" (The Go Blog, 2014).** How cancellation and deadlines compose across a call tree.
- **Rich Hickey, *core.async* (Clojure) design notes and talks.** CSP on the JVM, with an honest account of where channels help and where they do not.
- **Within this series:** [why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [message passing versus shared memory and the CSP philosophy](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy), [deadlock and the four conditions](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them), [data parallelism, fork/join, and work stealing](/blog/software-development/concurrency/data-parallelism-fork-join-and-work-stealing), and the capstone [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- **For Python's take on these patterns**, see [asyncio from the ground up: event loops and coroutines](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines), which owns the Python concurrency story this series links out to.

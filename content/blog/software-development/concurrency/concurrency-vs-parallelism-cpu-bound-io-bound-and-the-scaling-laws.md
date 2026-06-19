---
title: "Concurrency vs Parallelism: CPU-Bound, IO-Bound, and the Scaling Laws"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make the concurrency-versus-parallelism distinction airtight, learn to classify CPU-bound from IO-bound work, and derive the scaling laws that cap how fast more cores can ever make you."
tags:
  [
    "concurrency",
    "parallelism",
    "amdahls-law",
    "gustafsons-law",
    "cpu-bound",
    "io-bound",
    "scalability",
    "performance",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-1.png"
---

Two stories, same hardware, opposite lessons.

In the first, a colleague rewrote a single-threaded image-processing batch job to use all eight cores of the build box. He expected an 8× speedup. He got 1.05×. The job spent most of its wall-clock time in one function that decoded a manifest, parsed configuration, and sorted file paths — work that simply could not be split. Eight cores sat mostly idle while one core ground through the part that mattered. The machine had eight times the compute and almost none of the gain.

In the second, a different service — a chat backend — held ten thousand simultaneous client connections on a *single* operating-system thread. No thread pool, no eight cores, no parallelism at all. Each connection spent 99.9% of its life waiting: waiting for the client to type, waiting for a database round-trip, waiting on a downstream API. A single thread, by refusing to *block* on any one wait, kept all ten thousand conversations alive at once. One core, ten thousand things happening.

Same class of hardware. In the first case more cores bought nothing; in the second case a single core did the work of a thousand. The reason these two stories point in opposite directions is that they are about two genuinely different ideas that the industry routinely smears into one word: **concurrency** and **parallelism**. The image job needed parallelism and the structure of its work denied it. The chat server needed concurrency and got it from a single thread. If you can tell these two apart — and tell which one your workload actually wants — you will stop being surprised by speedups that never arrive.

This post makes the distinction airtight and then quantifies the ceiling. We will pin down what concurrency and parallelism each mean, walk the four quadrants of having neither, one, or both, and learn the single most useful classification you can apply to any workload: **CPU-bound versus IO-bound**. Then we go to the math — **Amdahl's law**, which tells you why a job that is 95% parallel can never beat about 20× no matter how many cores you throw at it, and **Gustafson's law**, the optimistic reframing that explains why supercomputers scale to millions of cores anyway. We will close on the overhead floor: the point where adding workers makes your program *slower*, with a measured scaling curve to prove it. The series spine — *concurrency is correctness under nondeterminism, parallelism is throughput under finite hardware* — starts here, because this is the post where those two halves get their precise meanings. For why any of this is unavoidable in the first place, see [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it); for the decision framework that ties every model together, see [the concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

![Matrix of the four quadrants showing parallel and not-parallel rows against concurrent and not-concurrent columns with a real workload in each cell](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-1.png)

## The precise definitions, and the four quadrants

Start with the cleanest one-line statements, the ones worth memorizing:

> **Concurrency is dealing with many things at once. Parallelism is doing many things at once.**

Rob Pike, in a much-cited talk, put it as "concurrency is about *structure*, parallelism is about *execution*." That phrasing is the key. Concurrency is a property of how your program is *organized*: it has multiple independent logical threads of control — tasks, requests, connections — whose progress is interleaved rather than strictly ordered. Parallelism is a property of how your program *runs*: multiple operations are physically executing at the same instant, which requires multiple execution units (cores, hardware threads, machines).

These are independent axes. That independence is the whole point, and it is what people miss when they treat the words as synonyms. Concurrency is about the *shape* of the work — whether it is decomposed into pieces that can make progress independently. Parallelism is about the *hardware* — whether more than one piece is literally advancing at the same physical moment. You can have either without the other.

A useful way to keep them straight: concurrency is a *design-time* decision and parallelism is a *run-time* fact. When you write your program as a set of independent tasks — connections, jobs, requests — that the runtime can interleave, you have *made it concurrent*; that is a choice in your source code, true whether you ever run it on one core or a hundred. Whether those concurrent tasks then *execute simultaneously* depends on the machine you deploy on and the scheduler: run a concurrent program on a single core and it runs concurrently but not in parallel; run the same source on eight cores and the runtime may run eight of its tasks in parallel. The same concurrent program can be non-parallel on Monday's single-core container and parallel on Tuesday's eight-core box, with no source change at all. So concurrency is something you *write*, and parallelism is something the hardware *grants* — and conflating them leads you to expect a parallel speedup from a machine that has only one core, or to add cores to a program whose source was never made concurrent and so can't use them.

There is one more subtlety worth naming early, because it trips people up: parallelism is one *way* to realize concurrency, but not the only way, and concurrency is one *way* to get useful work out of parallelism, but not the only way. A program decomposed into many tasks (concurrent) *can* be sped up by running those tasks on many cores (parallel) — that is the common case. But you can also be parallel without being concurrent in the everyday sense: a single `parallel-for` over an array is one logical activity whose arithmetic happens to run on many lanes; there is no juggling of distinct jobs, just one job accelerated. The cleanest mental separation is: concurrency = *how many things are in progress*; parallelism = *how many things are physically executing this instant*. The first is bounded by how you structured the work; the second is bounded by your core count.

Two more definitions we will lean on. A **process** is a program in execution with its own private memory address space; an operating system runs many processes and isolates them from each other. A **thread** is a unit of execution *within* a process; threads of one process share that process's memory, which is exactly what makes them both cheap to coordinate and dangerous to get wrong. When we say "parallel," we mean two threads (or two processes) running on two different cores at the same instant. When we say "concurrent," we mean two or more tasks are in flight — started but not finished — whether or not any two of them are advancing simultaneously.

Because the two axes are independent, there are four quadrants, and every one of them is a real, shippable design. The figure above lays them out; here is each cell in words.

**Concurrent and parallel.** Multiple tasks, multiple cores, genuinely simultaneous. A video encoder that splits a clip into independent segments and renders each on its own core. A web server that runs a thread pool across eight cores, each handling its own requests. This is what most people picture when they say "multithreaded," but it is only one of four boxes.

**Concurrent, not parallel.** Multiple tasks in flight, but only one core advancing them, interleaved. This is the chat server from the intro: a single-threaded event loop juggling ten thousand connections, switching among them whenever one would block. It is the JavaScript runtime in a browser. It is Python `asyncio` on one thread. There is real concurrency — ten thousand conversations are *dealt with* at once — and zero parallelism, because at any given nanosecond exactly one of them is executing a CPU instruction.

**Parallel, not concurrent.** This one surprises people, but it is the bread and butter of high-performance computing. A single logical operation — sum a billion numbers, multiply two matrices, apply one filter to every pixel — split across many cores. Conceptually there is *one* task; we are not juggling independent jobs, we are accelerating a single job by doing its arithmetic on many lanes at once. SIMD instructions, GPU kernels, and a `parallel-for` over one array are the canonical examples. The work is parallel in execution but not concurrent in structure: there is no interleaving of distinct logical activities, just one activity sped up.

**Neither.** One task, one core, start to finish in order. The plain script. The for-loop. Most code anyone has ever written. There is nothing wrong with this quadrant; it is correct, simple, and debuggable, and the entire point of the rest of this series is to help you decide when leaving it is worth the cost. A sequential program has no interleavings to reason about, no shared state to synchronize, no scheduler nondeterminism to fear — which is exactly why it should remain your default until a measured bottleneck justifies the move. Every step out of this quadrant trades simplicity for either throughput (parallelism) or responsiveness and overlap (concurrency), and that trade is only worth making when you can name the resource you are recovering.

The trap that this taxonomy guards against is the unspoken assumption that "make it faster" means "add threads," as if there were a single dial. There are two dials, they recover two different resources, and turning the wrong one wastes effort and often makes things worse. The image-batch engineer turned the parallelism dial on a problem whose bottleneck was its serial fraction; the chat-server engineer turned the concurrency dial and recovered idle waiting that no number of cores would have touched. The whole rest of this post is about reading your workload precisely enough to know which dial is even connected to anything.

The reason this taxonomy matters is that the four quadrants demand four different tools and exhibit four different failure modes. Reaching for parallelism (more cores) when your problem lives in the "concurrent, not parallel" quadrant gets you the chat-server-on-a-thread-pool — needless threads, needless context switches, needless locks, and no throughput gain because the bottleneck was never the CPU. Reaching for concurrency (an event loop) when your problem is "parallel, not concurrent" gets you an elegant single-threaded program that uses one of your eight cores while a number-crunch waits. Diagnosing *which quadrant you are in* is the first move, and the lever that puts you in the right quadrant is the next concept: whether your work is bound by the CPU or by waiting.

## CPU-bound versus IO-bound: the one classification that decides everything

If you remember one diagnostic from this post, make it this: **is your program bound by the CPU or bound by IO?** Almost every correct decision about concurrency and parallelism falls out of that single question, and almost every wrong decision comes from never asking it.

**CPU-bound** work spends its time *computing*. The processor is busy — the bottleneck is arithmetic, the limit is how many instructions per second your cores can retire. Hashing a password, encoding video, compressing a file, multiplying matrices, running a regex over a gigabyte of text, training a model, ray-tracing a frame: in all of these the cores are pinned near 100% and the program is gated by raw compute.

**IO-bound** work spends its time *waiting*. The processor is mostly idle — the bottleneck is some external thing the CPU has asked for and is now blocked on: a disk seek, a network round-trip, a database query, a reply from another service, a user's keystroke. A web request that calls three downstream APIs spends milliseconds in your code and hundreds of milliseconds waiting for those APIs. A log shipper that reads files and POSTs them spends its life in the kernel's network stack. The CPU could do a thousand other things during each wait, and that latent capacity is exactly what concurrency reclaims.

![Matrix comparing CPU-bound and IO-bound work across bottleneck, right tool, scaling axis, and example rows](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-3.png)

The figure above is the cheat sheet. The columns are the two kinds of work; the rows are the four things you need to know about each. CPU-bound: the bottleneck is the cores, the right tool is parallelism, it scales with core count up to the number you have, and it looks like hashing or encoding. IO-bound: the bottleneck is the wait, the right tool is concurrency, it scales with how many operations you keep in flight (which can be thousands), and it looks like HTTP calls or DB queries. The whole rest of this post is an unpacking of that table.

How do you actually tell which you have? Three reliable methods, in order of rigor.

First, **look at CPU utilization while the workload runs.** Run `top`, `htop`, your platform's activity monitor, or a profiler. If your process is sitting near 100% of one core (or N×100% across N cores), it is CPU-bound — the processor is the thing it is waiting on, which is itself. If your process shows low CPU but the wall-clock time is large, it is IO-bound — the time is going somewhere the CPU isn't busy. A program that takes 10 seconds of wall-clock time but only 0.3 seconds of CPU time is spending 9.7 seconds waiting; that is IO-bound by definition.

Second, **compare wall-clock time to CPU time directly.** On a Unix shell, the `time` command prints both: `real` is wall-clock, and `user + sys` is CPU time. The ratio is the diagnosis.

```bash
$ time sha256sum bigfile.bin     # CPU-bound: real is close to user
real    0m4.812s
user    0m4.790s
sys     0m0.018s

$ time curl -s https://example.com/slow-endpoint > /dev/null   # IO-bound: real >> CPU
real    0m1.604s
user    0m0.011s
sys     0m0.007s
```

In the first run, `user` (4.79 s) almost equals `real` (4.81 s): the CPU was doing arithmetic the entire time, classic CPU-bound. In the second, `real` is 1.6 s but the CPU time is a combined 18 *milliseconds*: the program spent 99% of its wall-clock time waiting on the network, textbook IO-bound. This one comparison resolves most ambiguity in seconds.

Third, **reason about the operations.** Does the hot path touch the network, the disk, a database, another process, or a human? Then it waits, and it is IO-bound. Does the hot path only touch registers, caches, and RAM, doing arithmetic and comparisons? Then it computes, and it is CPU-bound. Mixed workloads exist — a job that reads a file (IO) then compresses it (CPU) then uploads it (IO) — and there the right move is to classify *each stage* and apply the right tool to each. The classification is per-bottleneck, not per-program.

A common confusion worth defusing: "memory-bound" is a sub-case of CPU-bound for our purposes here. A workload that stalls waiting for RAM (a cache-miss-heavy graph traversal) is not *waiting on IO* in the blocking sense — the core can't go do something else useful, it is stalled on the memory subsystem. From the concurrency-vs-parallelism standpoint it behaves like CPU-bound work: more cores can help (each gets its own cache and memory channel), an event loop cannot. We will keep "IO-bound" reserved for work that blocks on something *outside the box's compute and memory*, where the CPU is genuinely free to do other work during the wait. That distinction — can the CPU do something else while it waits? — is the line that separates the two regimes.

It is worth being precise about *why* the wall-clock-versus-CPU-time test is so reliable, because the mechanism behind it is the same mechanism that makes concurrency work. When a thread issues a blocking system call — `read()` on a socket, a database query, a disk read — the kernel does not spin the CPU waiting for the answer. It marks the thread "not runnable" (sleeping on that IO event) and *takes the core away*, scheduling some other runnable thread instead. The clock keeps ticking on your program's wall-clock time, but *no CPU time is charged to your thread* during that sleep, because your thread is not on a core. That is exactly why `real` can be 1.6 s while `user + sys` is 18 ms: 1.58 s of that wall-clock time was spent with your thread asleep, off-core, charged to nobody. CPU-bound work, by contrast, never sleeps — it stays runnable and on-core the whole time, so `user` accrues at the same rate the wall clock does. The diagnosis is not a heuristic; it is reading the kernel's own accounting of where your thread was.

One more practical wrinkle: utilization figures lie if you read them carelessly. A multithreaded process pinning all 8 cores shows "800%" CPU on tools that report per-process totals, or "100%" on tools that normalize to the whole machine — know which convention your tool uses before you conclude anything. And a process that *looks* 100%-busy on one core but is actually thrashing on cache misses is CPU-bound in the sense that matters here (cores help, async doesn't), even though the "real work per cycle" is low. The clean rule survives all of this: if taking the core away and giving it to another thread would let *useful other work* happen, you are IO-bound and concurrency wins; if there is no other work the core could usefully do because *your own* work needs every cycle, you are CPU-bound and only more cores help. Hold that test — "could the core be doing something else right now?" — and the right tool is never in doubt.

## Why IO-bound work wants concurrency and CPU-bound work wants parallelism

Now the heart of it: *why* does the right tool differ? Because the two regimes have a different thing to recover, and the mechanisms recover different things.

![Before and after comparison of interleaving tasks on one core versus running tasks at once on many cores](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-2.png)

The figure contrasts the two mechanisms. On the left, concurrency on one core: task A runs until it blocks on IO, then the runtime switches to task B, which runs while A's IO completes in the background — never two at the same instant, but the core is never idle. On the right, parallelism on N cores: task A on core 0 and task B on core 1 advance at the very same instant. Concurrency *fills the gaps*; parallelism *adds lanes*. Hold that image — it is the mechanism behind everything below.

**IO-bound work has idle time to reclaim.** When a thread issues a network read and the data hasn't arrived, the thread *blocks*: the operating system parks it, marks it not-runnable, and gives the core to someone else. If you wrote your program as one thread doing one request at a time, then during every one of those waits the core does nothing for *your* program. The latent capacity — the CPU cycles available during the wait — is enormous. A 100 ms network round-trip on a 3 GHz core is 300 *million* cycles of doing nothing. Concurrency's job is to fill those cycles with *other* tasks' work.

Crucially, this does not require parallelism. A single core, switching among tasks every time one would block, can keep thousands of IO operations in flight, because at any instant only one task needs the CPU — the other 9,999 are waiting on the network, and waiting is free. This is why a single-threaded event loop serves 10,000 connections (the famous "C10k" problem). It is also why threads work for IO-bound concurrency even under a global interpreter lock: a thread blocked on IO is not holding the CPU, so other threads run. The python-performance series proves this empirically in [threading done right: IO-bound concurrency and its limits](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) and explains the event-loop machinery in [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines); the lesson generalizes to Go's goroutines, Java's virtual threads, and Rust's `async`/`tokio`.

To make the saving concrete, watch what happens to three IO requests that each take 100 ms of pure waiting. Done the naive way — issue request 1, wait for it, then issue request 2, wait, then request 3, wait — the three waits run *back to back*: 100 + 100 + 100 = 300 ms of wall-clock time, and the CPU was idle for essentially all 300 ms. Done concurrently — issue all three requests first (which takes microseconds, the CPU just hands them to the kernel and moves on), then wait for all three to come back — the three waits *overlap*: while request 1's reply is in flight, request 2's and request 3's are also in flight, so the total wait is about 100 ms, set by the slowest single request rather than the sum. Three serial waits collapsed into one overlapped wait. The figure traces both schedules step by step.

![Timeline contrasting three blocking IO waits running back to back against three overlapped waits issued together and awaited once](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-4.png)

The top row is the blocking schedule: three 100 ms waits in series, 300 ms total. The bottom row is the overlapped schedule: a single "issue all three" step that does no waiting, then one combined await where the three waits run on top of each other, 100 ms total. Nothing about the *work* changed — the same three requests, the same network, the same one core — only the *structure* changed, from serial-wait to overlapped-wait, and the wall-clock time dropped 3×. Scale that to a thousand requests and the serial version takes 100 seconds while the overlapped version takes about 100 milliseconds: the saving grows with the number of in-flight operations, which is exactly why IO-bound concurrency scales to thousands. This is also the precise reason "add more cores" does nothing here — the cores were never the bottleneck; the *serialized waiting* was, and concurrency, not parallelism, is what un-serializes it.

**CPU-bound work has no idle time to reclaim — only more lanes can help.** If your thread is computing, the core is busy. There is no wait to overlap with someone else's work, because there is no wait at all. Switching to another task does not help; that task also needs a busy core. The *only* way to go faster is to do the arithmetic on more cores at once — that is parallelism, and it needs real hardware execution units. Concurrency on one core does nothing for a CPU-bound job except add context-switching overhead.

Here is the trap, stated as a rule and then demonstrated in code: **threads do not give you parallelism for CPU-bound work if a global lock serializes execution.** In a runtime with a global interpreter lock (CPython is the canonical example, but the principle is general — any shared lock around the interpreter or a coarse resource has the same effect), only one thread executes bytecode at a time. For IO-bound work that lock is released during the wait, so threads still overlap waiting. For CPU-bound work the lock is *held* the whole time a thread computes, so N threads take turns on one core and you get no speedup — sometimes a slowdown, from the switching overhead. The fix for CPU-bound work is genuine parallelism: separate processes (which each have their own interpreter and their own lock), or a language without the global lock, or offloading the hot loop to native code.

Let me show the bug and the fix concretely, in two languages where the idioms diverge. First, the bug — Python threads on a CPU-bound sum, which the global lock serializes (this is the illustration; the python-performance series owns the deep Python story, so we link out rather than re-derive it in [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs)):

```python
import threading, time

def cpu_work(n):
    total = 0
    for i in range(n):
        total += i * i          # pure arithmetic, no IO, never blocks
    return total

N = 50_000_000

# The BUG below: 4 threads on CPU-bound work. The GIL serializes them onto
# one core, so this is NOT faster than running the work once, may be slower.
start = time.perf_counter()
threads = [threading.Thread(target=cpu_work, args=(N,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"4 threads: {time.perf_counter() - start:.2f}s")   # ~ same as serial x4
```

Four threads, four times the work, and roughly the wall-clock time of doing it serially four times — because only one thread holds the lock and computes at a time. The fix is to use *processes*, which sidestep the global lock entirely because each process has its own interpreter:

```python
import multiprocessing, time

def cpu_work(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == "__main__":
    N = 50_000_000
    start = time.perf_counter()
    # FIX: 4 processes -> 4 real cores -> near-4x speedup on CPU-bound work.
    with multiprocessing.Pool(4) as pool:
        pool.map(cpu_work, [N] * 4)
    print(f"4 processes: {time.perf_counter() - start:.2f}s")  # ~ 4x faster
```

The cost of processes is real — separate memory, and arguments and results must be serialized ("pickled") across the process boundary, which the python-performance series quantifies in [multiprocessing: true parallelism and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling). The point that generalizes: for CPU-bound work you need *separate execution units that do not share a serializing lock*.

In a language with no global lock, threads *are* the parallelism primitive, and the same CPU-bound sum scales across cores without the process workaround. Here is Go, where goroutines are scheduled across all available cores by default:

```go
package main

import (
	"fmt"
	"sync"
)

func cpuWork(n int) int {
	total := 0
	for i := 0; i < n; i++ {
		total += i * i // pure arithmetic, runs on whatever core the scheduler picks
	}
	return total
}

func main() {
	const N = 50_000_000
	var wg sync.WaitGroup
	results := make([]int, 4)
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(idx int) { // each goroutine can land on a different core
			defer wg.Done()
			results[idx] = cpuWork(N)
		}(w)
	}
	wg.Wait() // 4 goroutines, 4 cores, near-4x speedup — no GIL to serialize them
	fmt.Println(results)
}
```

And Rust with [Rayon](https://github.com/rayon-rs/rayon), where data parallelism over a slice is a one-line change from the sequential iterator — `iter()` becomes `par_iter()` and the work fans across cores:

```rust
use rayon::prelude::*;

fn main() {
    let data: Vec<u64> = (0..50_000_000).collect();

    // Sequential: one core.
    // let sum: u64 = data.iter().map(|&x| x * x).sum();

    // Parallel: Rayon splits the range across a work-stealing thread pool.
    // No global lock; on an 8-core box this is close to 8x for big enough N.
    let sum: u64 = data.par_iter().map(|&x| x * x).sum();

    println!("{sum}");
}
```

Same shape, three languages, one principle: CPU-bound work scales only by recruiting more execution units, and that requires either processes (to dodge a global lock) or a runtime where threads run truly in parallel. Now the IO-bound side, where the right answer is concurrency, not more cores. Here is Go again, issuing many HTTP requests concurrently from a handful of goroutines — the goroutines spend their lives blocked on the network, so even one core overlaps thousands of waits:

```go
package main

import (
	"io"
	"net/http"
	"sync"
)

func fetch(url string, wg *sync.WaitGroup) {
	defer wg.Done()
	resp, err := http.Get(url) // blocks on the network — but the goroutine yields
	if err != nil {
		return
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body) // drain and discard
}

func main() {
	urls := loadURLs() // say 1000 URLs
	var wg sync.WaitGroup
	for _, u := range urls {
		wg.Add(1)
		go fetch(u, &wg) // 1000 concurrent fetches; the waits all overlap
	}
	wg.Wait()
}
```

The same job in Rust with `tokio`, where `async`/`.await` cooperatively yields the single (or few) OS threads back to the runtime on every `.await`, so a small thread pool drives thousands of in-flight requests:

```rust
use futures::future::join_all;

async fn fetch(url: String) -> usize {
    let body = reqwest::get(&url).await.unwrap()  // yields the thread while waiting
        .bytes().await.unwrap();
    body.len()
}

#[tokio::main]
async fn main() {
    let urls: Vec<String> = load_urls(); // say 1000 URLs
    // Spawn all fetches; the runtime overlaps every network wait on a few threads.
    let futures = urls.into_iter().map(fetch);
    let sizes = join_all(futures).await;
    println!("fetched {} responses", sizes.len());
}
```

Notice that the IO-bound code uses concurrency primitives (goroutines, `async` tasks) and does *not* care how many cores it has — one core can drive a thousand overlapping waits. The CPU-bound code uses parallelism primitives (processes, a work-stealing pool) and cares about exactly one thing: how many cores it has. That mismatch — using the wrong primitive for the bound — is the single most common concurrency mistake, and you now have the rule to avoid it.

#### Worked example: which tool for a 200-URL scrape that hashes each page

You need to fetch 200 web pages and compute a SHA-256 of each. Each fetch averages 150 ms of network wait; each hash takes 5 ms of CPU. Classify the stages and pick the tool.

The fetch stage: 200 fetches × 150 ms = 30 seconds of *waiting* if done serially. This is IO-bound — the CPU is idle during each fetch. Concurrency wins: issue all 200 fetches and overlap the waits. With even modest concurrency the 30 seconds of serial waiting collapses toward the time of the slowest single fetch plus scheduling overhead — call it well under 1 second of wall-clock for the network portion. No extra cores needed.

The hash stage: 200 hashes × 5 ms = 1 second of *compute* if done serially. This is CPU-bound — the core is busy. Concurrency on one core buys nothing here; parallelism does. On 8 cores, 1 second of serial hashing becomes about 0.13 seconds. The right design is concurrency for the fetches (an async client, or threads, keeping all 200 in flight) feeding a small parallel pool of processes (or a no-global-lock thread pool) for the hashes. Use *both* tools, each on the stage it fits. A single-tool answer — "just throw threads at it" or "just throw cores at it" — gets one stage right and the other wrong.

## Task parallelism versus data parallelism

When the work *is* parallel, there are two distinct ways to split it, and they have different ergonomics, different load-balancing behavior, and different failure modes. Knowing which one you have shapes the code you write.

![Tree taxonomy of parallel work splitting into task parallelism and data parallelism with concrete examples under each](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-5.png)

**Task parallelism** runs *different* operations at the same time. Each worker does a *different job*. A web request handler that, in parallel, calls the auth service, loads the shopping cart, and computes a shipping quote is task-parallel: three different functions, three different code paths, running at once and then joined. A media pipeline whose stages — decode, resize, watermark, encode — each run on their own worker is task-parallel. The defining feature is *heterogeneity*: the workers are not interchangeable; each does a specific, distinct piece. The classic risk is imbalance — if the shipping-quote call takes 400 ms and the others take 20 ms, your parallel section is gated by the slow one, and the fast workers sit idle. Task parallelism scales only up to the number of distinct tasks you have; three tasks cannot use eight cores no matter what.

**Data parallelism** runs the *same* operation over *different* pieces of data. Each worker does the *same job* on its own chunk. Summing a billion numbers by giving each of 8 cores 125 million of them and adding the partial sums is data-parallel. Applying one filter to every pixel of an image, where each core owns a tile, is data-parallel. SIMD instructions (one instruction, many data lanes) and GPU kernels (thousands of threads running the identical kernel over different elements) are the hardware embodiment of data parallelism. The defining feature is *homogeneity*: the workers are interchangeable, each running the identical code over its slice. Data parallelism scales as far as you can subdivide the data, which is often *very* far — millions of elements, millions of lanes — which is precisely why it is the engine of high-performance and machine-learning computing. MapReduce, `Rayon`'s `par_iter`, Java's parallel streams, NumPy's vectorized ops, and every GPU workload are data-parallel.

A compact comparison:

| Aspect | Task parallelism | Data parallelism |
| --- | --- | --- |
| What is split | Different operations | One operation over chunks |
| Workers are | Heterogeneous, specialized | Homogeneous, interchangeable |
| Scales up to | Number of distinct tasks | Number of data chunks (often huge) |
| Load balancing | Hard — slowest task gates the join | Easy — even chunks, work-stealing |
| Canonical form | Pipeline stages, fan-out service calls | MapReduce, SIMD, GPU kernel, parallel-for |
| Typical primitive | Thread/goroutine per task, futures | Parallel-for, `par_iter`, map over partitions |

The two compose. A real system is often task-parallel at the top (a pipeline of stages) and data-parallel inside a stage (each stage processes its batch across cores). The reason to name them separately is that they fail differently and balance differently. If your "parallel" code isn't speeding up, ask which kind it is: a task-parallel section caps at the number of tasks and is gated by the slowest one; a data-parallel section caps at the number of chunks and is usually limited instead by overhead or by the serial fraction — which is exactly where the scaling laws come in.

The load-balancing distinction deserves a moment because it is where task parallelism quietly disappoints. With data parallelism, if you split a billion-element array into 8 equal chunks, each worker has the same amount of work, so they finish together and no core sits idle waiting for a straggler. Even when chunks are *uneven* (some rows of an image are cheaper to filter than others), a *work-stealing* scheduler — Rayon, the JVM's `ForkJoinPool`, Go's runtime, Intel TBB — lets an idle worker steal a sub-chunk from a busy one, so the load self-balances at run time without you partitioning perfectly up front. Task parallelism has no such luxury: the three service calls in the fan-out example are *indivisible distinct jobs*, so if one takes 400 ms and the others 20 ms, the whole parallel section takes 400 ms and there is nothing to steal — you cannot hand half of "call the shipping service" to an idle worker. The practical consequence: task parallelism's speedup is gated by its *slowest task* (the critical path), while data parallelism's speedup is gated by its *total work divided by workers* (modulo overhead). When you can express a problem as data-parallel, you usually should, precisely because it balances and scales so much more gracefully — and that is why the heavy-iron computing world (HPC, deep learning, big-data) is overwhelmingly data-parallel.

#### Worked example: a render pipeline that is task-parallel then data-parallel

A video service ingests a clip and runs four stages: demux (5 ms), decode (40 ms), apply a filter to every frame (200 ms), and encode (60 ms). Naively the stages are *sequential* — decode needs demux's output, the filter needs decoded frames — so the per-clip latency is 5 + 40 + 200 + 60 = 305 ms, and that order is a *data dependency*, an inherently serial chain you cannot parallelize away for a single clip.

Two different parallel wins are available, and they are different *kinds*. First, *task parallelism across clips*: run the demux of clip 2 while clip 1 is being filtered, decode of clip 3 while clip 2 is being encoded — a classic pipeline where each stage is a worker and clips flow through. With four stages busy at once on four workers, *throughput* rises to roughly one clip per 200 ms (the slowest stage, the filter, sets the pipeline's rate), even though each individual clip still takes 305 ms of latency. This is task parallelism: heterogeneous stages, gated by the slowest one.

Second, *data parallelism inside the filter stage*: that 200 ms filter applies the same operation to every frame independently, so split the frames across 8 cores and the filter stage drops from 200 ms to about 25 ms. Now the per-clip latency falls to 5 + 40 + 25 + 60 = 130 ms, *and* the pipeline's bottleneck stage is no longer the filter (25 ms) but the encode (60 ms), so throughput rises to one clip per 60 ms. The lesson: the *same system* uses task parallelism to overlap distinct stages across clips and data parallelism to accelerate one stage internally — and once you accelerate one stage, the bottleneck moves, so you re-profile and attack the new slowest stage. Naming the two kinds is what lets you see both moves.

## Amdahl's law: the serial fraction is the ceiling

Here is the mechanism that explains the intro's image job, and it is worth deriving from scratch because the derivation *is* the explanation — once you see where the terms come from, the ceiling is obvious.

Take any task and split its total work, on one worker, into two parts: a fraction `$p$` that *can* be parallelized and a fraction `$(1-p)$` that *cannot* — the inherently serial part, the setup, the part with a true data dependency, the single-threaded section. Normalize the one-worker time to 1. So the serial part takes time `$(1-p)$` and the parallel part takes time `$p$`, and together they sum to 1.

Now run it on `$N$` workers. The serial part cannot be sped up — by definition it runs on one worker — so it still takes `$(1-p)$`. The parallel part, *ideally*, splits perfectly across `$N$` workers, so it takes `$p/N$`. The total time on `$N$` workers is therefore:

$$T(N) = (1-p) + \frac{p}{N}$$

Speedup is the ratio of one-worker time to `$N$`-worker time, and one-worker time is 1:

$$S(N) = \frac{1}{(1-p) + \dfrac{p}{N}}$$

That is **Amdahl's law**, stated by Gene Amdahl in 1967. Everything important about parallel scaling is in that one fraction. Now take the limit as `$N \to \infty$` — infinite workers, the parallel part shrinks to zero:

$$S_{\max} = \lim_{N\to\infty} \frac{1}{(1-p) + \dfrac{p}{N}} = \frac{1}{1-p}$$

This is the punchline and it is brutal: **the maximum possible speedup is one over the serial fraction.** Not the parallel fraction — the *serial* one, the part you couldn't split. The serial fraction is a hard ceiling that no amount of hardware can break, because it is the time of the part that never goes away.

![Before and after diagram of Amdahl's law showing a fixed serial part bounding total time as the parallel part shrinks with more workers](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-6.png)

The figure shows it geometrically. On one worker, a 5%-serial / 95%-parallel job spends 0.05 of its time serial and 0.95 parallel, total 1.0. Add workers and the parallel block shrinks — toward zero with enough of them — but the serial block does not move. The total time bottoms out at 0.05, which is a 20× speedup, and you cannot go below it. The serial sliver, irrelevant on one core, *becomes the entire job* once you have enough cores. That is why the image-batch rewrite got 1.05× instead of 8×: its serial fraction wasn't 5%, it was most of the runtime, and one over a large serial fraction is a small number.

It pays to understand *what counts as serial*, because engineers routinely underestimate their own serial fraction. The serial part is not only the obvious single-threaded setup. It includes: any section guarded by a lock that all workers contend on (while one holds it, the rest wait — that is serial execution wearing a parallel costume); the final reduction or merge step that combines partial results (summing 8 partial sums is serial, however briefly); reading the input and writing the output if those are single-streamed; any data dependency that forces ordering (step B needs step A's result); and — subtly — the *thread spawn and join* themselves, which happen on one thread. A job you *believe* is 99% parallel often measures at 90% once you account for the lock contention and the merge, and that difference moves your ceiling from 100× down to 10×. The discipline Amdahl forces is to *measure the serial fraction empirically* — run on 1, 2, 4, 8 workers, fit the curve, and back out `$p$` — rather than eyeball it, because the eyeball is optimistic and the law is unforgiving.

Let me put the numbers in a table, because the shape of the curve is the lesson.

| Serial fraction `$(1-p)$` | Parallel `$p$` | Max speedup `$1/(1-p)$` | Speedup at 8 cores | Speedup at 64 cores |
| --- | --- | --- | --- | --- |
| 1% | 99% | 100× | 7.5× | 39.3× |
| 5% | 95% | 20× | 5.9× | 15.4× |
| 10% | 90% | 10× | 4.7× | 9.1× |
| 25% | 75% | 4× | 2.9× | 3.8× |
| 50% | 50% | 2× | 1.8× | 2.0× |

Read across the 5% row: a job that is 95% parallelizable — which sounds *excellent* — caps at 20× no matter how many cores you buy, reaches only 5.9× on 8 cores, and limps to 15.4× even on 64 cores. The 64-core box delivers 15.4× while costing 8× the cores of an 8-core box for less than 3× the speedup. Read the 50% row: a half-serial job cannot beat 2×, ever, with infinite cores. The serial fraction is the tyrant.

![Matrix table mapping serial fraction of one, five, ten, and fifty percent to maximum speedup of one hundred, twenty, ten, and two](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-7.png)

The figure distills the ceiling: 1% serial → 100×, 5% → 20×, 10% → 10×, 50% → 2×. Memorize the 5% → 20× point; it is the one that most often surprises people, because 95% parallel *feels* like it should give nearly the full core count, and it gives a fifth of it on 64 cores. The practical upshot is that the highest-leverage parallel optimization is usually *shrinking the serial fraction*, not adding cores — moving setup out of the critical path, removing a global lock, batching the single-threaded part. One percentage point off the serial fraction at the low end (1% → 0.5%) doubles your ceiling from 100× to 200×; that is a far better trade than doubling your hardware.

#### Worked example: is the 64-core box worth it for a 92%-parallel job

You have a batch job measured at 92% parallel, 8% serial. It runs on an 8-core machine today in 100 seconds. A vendor offers a 64-core machine. Should you take it, and what will you actually get?

First the ceiling. Max speedup is `$1/(1-p) = 1/0.08 = 12.5\times$`. So no matter what, this job cannot run faster than `$100/12.5 = 8$` seconds. That is the floor on wall-clock time, set entirely by the 8% you can't parallelize.

Now the two machines. At 8 cores: `$S = 1/(0.08 + 0.92/8) = 1/(0.08 + 0.115) = 1/0.195 = 5.13\times$`, so `$100/5.13 \approx 19.5$` seconds. At 64 cores: `$S = 1/(0.08 + 0.92/64) = 1/(0.08 + 0.0144) = 1/0.0944 = 10.6\times$`, so `$100/10.6 \approx 9.4$` seconds. The 64-core machine — eight times the cores — takes you from 19.5 s to 9.4 s, a 2.07× improvement for 8× the parallel hardware. You are paying for 56 extra cores to claw back 10 seconds, and you are already within 18% of the 8-second floor that you can never beat.

The decision: if those 10 seconds matter (a latency-critical pipeline), maybe. But the *better* engineering move is to attack the 8% serial part. Cut it to 4% and the ceiling jumps from 12.5× to 25×, and even the 8-core machine would then hit `$1/(0.04 + 0.92/8) = 1/0.155 = 6.45\times$` → 15.5 s — beating where you started, on hardware you already own. Amdahl tells you to profile the serial fraction before you sign the hardware purchase order.

## Gustafson's law: scale the problem, not the speedup

If Amdahl's law were the whole story, supercomputers with a million cores would be pointless — the serial fraction would cap everyone at a few hundred times at best. Yet supercomputers scale to millions of cores and stay useful. The resolution came from John Gustafson in 1988, and it is less a contradiction of Amdahl than a different question.

Amdahl asks: *given a fixed problem, how much faster does it run with more cores?* That is **strong scaling** — same problem, more workers, measure the speedup. Under strong scaling the serial fraction is fixed and you hit the ceiling. But Gustafson observed that in practice nobody buys a thousand-core machine to run yesterday's problem faster. They buy it to run a *bigger* problem in the *same time* — a finer simulation grid, a larger model, more data, higher resolution. That is **weak scaling** — grow the problem with the workers, keep the runtime roughly constant.

The two terms are worth carving into memory because benchmark reports constantly equivocate between them and the equivocation hides bad news. **Strong scaling** holds the *total work* fixed and adds workers; the metric is "time to finish this exact job," and a strong-scaling plot that flattens is Amdahl's ceiling showing up. **Weak scaling** holds the *work per worker* fixed and adds workers (so total work grows with `$N$`); the metric is "can I do `$N$` times the work in the same wall-clock time," and a weak-scaling plot that stays flat-and-high is the good news that your serial fraction isn't growing. A vendor who shows you a gorgeous near-linear *weak-scaling* curve has told you the machine handles bigger problems well — and has told you *nothing* about whether it makes your *current, fixed* problem finish faster, which is a strong-scaling question with a much grimmer answer. Always ask which scaling a speedup claim refers to.

![Before and after comparison of a fixed problem saturating under Amdahl versus a grown problem scaling under Gustafson](/imgs/blogs/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws-8.png)

The figure puts the two side by side. On the left, Amdahl's fixed-size world: same input, the serial part is a fixed fraction, speedup saturates and flattens. On the right, Gustafson's grown-problem world: bigger input, the serial part becomes a *shrinking* fraction of a larger total, and speedup keeps rising nearly linearly with cores. The key move is that the serial work (setup, coordination) often does *not* grow when you grow the data — you parse the config once whether you simulate a thousand cells or a billion — so as the parallel work balloons, the serial fraction shrinks toward zero and Amdahl's ceiling recedes.

Here is the law. Suppose on `$N$` workers the parallel part takes time `$p$` and the serial part takes `$s$`, with `$s + p = 1$` being the *measured* time on the parallel machine. If you ran that same (now large) workload on one worker, the serial part would still take `$s$` but the parallel part would take `$N \cdot p$` (it was already split across `$N$`). The scaled speedup is:

$$S(N) = \frac{s + N \cdot p}{s + p} = s + N \cdot p = N - s(N-1)$$

Read that last form: the speedup grows *linearly* in `$N$` with slope `$p$`, offset down only by the serial fraction. If your serial fraction `$s$` is small and the problem is large, the speedup tracks the core count almost one-to-one. With `$s = 0.05$`, on 1000 cores Gustafson predicts `$S = 1000 - 0.05 \times 999 = 950\times$` — versus Amdahl's hard cap of 20× for the *fixed* problem. Same 5% serial fraction, two wildly different answers, because they are answering two different questions.

| | Amdahl (strong scaling) | Gustafson (weak scaling) |
| --- | --- | --- |
| Problem size | Fixed | Grows with cores |
| Question answered | Same job, how much faster? | Bigger job, same time? |
| Serial fraction | Fixed → caps speedup | Shrinks as problem grows |
| Speedup behavior | Saturates at `$1/(1-p)$` | Nearly linear in `$N$` |
| Right metric | Time to finish a set job | Throughput / problem size at fixed time |
| Where it rules | Latency-bound, fixed workload | Capacity planning, HPC, big-data |

Neither law is "correct" and the other "wrong"; they model different goals. If your job has a fixed size and you need it *done faster* — render this specific frame, finish this nightly batch sooner — you live under Amdahl, and the serial fraction is your enemy. If your job can *grow* and you want to do *more* in the same time — simulate more, serve more, index more — you live under Gustafson, and adding cores keeps paying off as long as you scale the work to match. The honest answer to "will more cores help?" is "for which question?" Most production capacity-planning is weak scaling (handle more traffic), and most one-off acceleration is strong scaling (make this run faster) — knowing which one you are in tells you whether to expect the ceiling or the linear ramp.

#### Worked example: weak versus strong scaling on the same cluster

A genomics team runs an alignment job. On 1 node it processes 1 genome in 1 hour; the per-genome serial setup is 3 minutes (0.05 of the hour), the alignment is 57 minutes (0.95) and is data-parallel.

*Strong scaling (Amdahl)* — same single genome, more nodes. Max speedup `$= 1/0.05 = 20\times$`, so even on a 256-node cluster one genome cannot finish faster than 3 minutes (the serial setup). At 16 nodes: `$S = 1/(0.05 + 0.95/16) = 1/0.109 = 9.2\times$` → about 6.5 minutes. Past ~20 nodes, adding hardware for *this one genome* is nearly pointless; you are pinned near the 3-minute setup floor.

*Weak scaling (Gustafson)* — 256 genomes on 256 nodes, one each, same hour. Each node does its own setup (3 min) and its own alignment (57 min) in parallel with the others, so all 256 genomes finish in about 1 hour — 256× the throughput of one node, for 256× the hardware. Here the serial fraction never bit, because the serial work *replicated* per node rather than gating a shared job, and the problem grew exactly with the cores.

Same cluster, same job kernel, same 5% serial fraction. Asking "finish one genome faster" caps you at 20×; asking "process more genomes in the hour" scales to 256× and beyond. The cluster purchase is justified by the second question, not the first — and a team that benchmarked only single-genome speedup would wrongly conclude the cluster was a waste.

## The overhead floor: where speedup goes negative

The scaling laws above are *optimistic*: they assume the parallel part splits *perfectly* with zero coordination cost. Reality charges a tax, and past a point that tax exceeds the work, so adding workers makes the program *slower*. This is the part the textbook curves omit and the part that bites you in production.

Every parallel decomposition pays for things the serial version never did. **Spawn cost**: creating a thread is roughly microseconds (tens of µs for an OS thread; far less for a goroutine or virtual thread, but never free); creating a process is more like milliseconds. **Coordination**: workers must be handed work and their results collected — a queue, a channel, a barrier, a join. **Synchronization**: if workers touch shared state, they contend on locks or atomics, and contended synchronization serializes them right back (the very thing parallelism was supposed to avoid). **Communication**: in data parallelism the partial results must be combined (the "reduce"), and across processes or machines the data must be serialized, copied, and shipped. **Scheduling and context switches**: more runnable threads than cores means the OS time-slices them, and each switch costs roughly 1–5 µs of direct cost plus a cache-pollution tail that can dwarf it. **False sharing and cache effects**: two cores writing to variables on the same cache line ping-pong that line between their caches, which can make "parallel" code several times slower than serial — a hazard we cover in depth in its own post.

Model it crudely. The optimistic time was `$(1-p) + p/N$`. Add an overhead term that *grows* with the number of workers — call it `$c \cdot N$` for coordination that scales with worker count, or `$c \cdot \log N$` for a tree reduction:

$$T(N) = (1-p) + \frac{p}{N} + c \cdot N$$

The first two terms *decrease* (or stay flat) as `$N$` grows; the third *increases*. There is a minimum. Differentiate and the optimal worker count is finite — there is a sweet spot `$N^*$` beyond which every added worker adds more overhead than it removes work, and the curve turns *up*. Past `$N^*$`, more cores means *slower*. This is not a corner case; it is the normal shape of a real scaling curve, and the reason "just add more threads" so often backfires.

Here is a measured-style scaling table for two workloads on a hypothetical 8-physical-core (16-hyperthread) machine. These are illustrative orders of magnitude — measured honestly they would jitter run-to-run, and the exact numbers depend on the chip, the OS scheduler, and the workload — but the *shape* is what every real measurement shows.

| Workers | CPU-bound speedup | IO-bound speedup | Note |
| --- | --- | --- | --- |
| 1 | 1.0× | 1.0× | baseline |
| 2 | 1.9× | 2.0× | both near-linear early |
| 4 | 3.7× | 4.0× | CPU still scaling well |
| 8 | 6.8× | 8.0× | CPU near physical-core limit |
| 16 | 7.4× | 16× | CPU: hyperthreads add little, overhead grows |
| 32 | 6.9× | 31× | CPU goes *backward*; IO still rising |
| 256 | 5.1× | 220× | CPU: pure overhead; IO: thousands of waits overlap |

Read the CPU-bound column top to bottom. It scales nicely to 8 (the physical-core count), barely improves from 8 to 16 (hyperthreads share execution resources, so the second logical core per physical core adds maybe 20–30%), and then goes *negative* past 16: at 32 and 256 workers the speedup is *lower* than at 8, because there are far more threads than cores, the OS thrashes context-switching among them, caches get polluted, and any shared synchronization contends. The CPU-bound work has a sweet spot near the physical core count and punishes you for exceeding it.

Now read the IO-bound column. It scales nearly linearly far past the core count — 16×, 31×, 220× — because the "workers" are mostly *blocked on IO*, not competing for cores. A thousand connections waiting on the network use almost no CPU; you can keep thousands in flight on a handful of threads (or one event loop) and throughput keeps climbing until you saturate the network, the downstream service, or memory for connection state. This is the empirical signature of the two regimes: CPU-bound speedup peaks at the core count and then falls; IO-bound concurrency scales to thousands. Measuring this curve for *your* workload is how you find your sweet spot — and the fact that it *has* a sweet spot, beyond which more is worse, is the practical face of the overhead floor.

#### Worked example: where the overhead floor turns the curve down

Put numbers on the crude model `$T(N) = (1-p) + p/N + cN$`. Take a job that is 99% parallel `$(p = 0.99,\ 1-p = 0.01)$`, normalized to 1 time unit on one worker, with a per-worker coordination cost of `$c = 0.0008$` units (sub-millisecond queue and synchronization overhead per worker, which is realistic for a contended shared queue). Without the overhead term, Amdahl alone would cap this at `$1/0.01 = 100\times$`. With the overhead term, the curve has a peak and then descends.

Evaluate it. At 8 workers: `$T = 0.01 + 0.99/8 + 0.0008 \times 8 = 0.01 + 0.124 + 0.0064 = 0.140$`, speedup `$1/0.140 = 7.1\times$`. At 32 workers: `$T = 0.01 + 0.99/32 + 0.0008 \times 32 = 0.01 + 0.031 + 0.0256 = 0.066$`, speedup `$15.0\times$`. At 64 workers: `$T = 0.01 + 0.0155 + 0.0512 = 0.077$`, speedup `$13.0\times$` — *lower* than at 32. The curve peaked somewhere around 35 workers and is now going *backward*: the `$cN$` overhead term (0.0512 at 64 workers) has grown larger than the parallel-work term it was supposed to be helping (0.0155), so each added worker costs more coordination than it saves in computation. The optimum `$N^*$` is where the derivative is zero, `$N^* = \sqrt{p/c} = \sqrt{0.99/0.0008} \approx 35$`, and past it you are paying to go slower. The lesson is sharp: even a *99%-parallel* job — far better than most real code — has a finite best worker count set by overhead, not by Amdahl's ceiling, and blindly setting your pool size to "all 64 hyperthreads" can land you on the wrong side of the peak.

How do you measure this honestly, so your numbers mean something? Four disciplines, all mandatory:

```python
import time, statistics

def benchmark(fn, workers, warmup=3, runs=15):
    # 1. WARM UP: discard the first few runs. The first run pays for cold caches,
    #    JIT warmup, lazy imports, page faults, and thread-pool spin-up.
    for _ in range(warmup):
        fn(workers)
    # 2. MANY RUNS: scheduling and contention make a single run noisy.
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(workers)
        samples.append(time.perf_counter() - start)
    # 3. REPORT SPREAD, not just a point. Median resists outliers; stdev shows noise.
    return {
        "median": statistics.median(samples),
        "stdev": statistics.pstdev(samples),
        "min": min(samples),  # the "best case" the machine can do
    }

# 4. CONTROL THE ENVIRONMENT: pin frequency (disable turbo/throttling if you can),
#    quiesce background load, name the platform (CPU model, core count, OS) in the
#    write-up. A speedup number without a platform is not reproducible.
```

The four disciplines: **warm up** (discard cold-start runs — caches, JIT, lazy init, and thread-pool spin-up all distort the first iterations); **run many times** and report the spread, not a single number (median and standard deviation, because scheduling makes single runs noisy); **control the environment** (pin CPU frequency, quiesce background load, and *name the platform* — an x86 box and an ARM box, or a turbo-boosting laptop and a steady server, give different answers); and **never extrapolate past what you measured** — a curve that scales to 8 tells you nothing about 64 except that you should go measure 64. The python-performance series treats benchmarking rigor at length; the same statistics apply in any language. The honest scaling curve, warts and downturn included, is the only thing that tells you where your real sweet spot is.

## Case studies / real-world

**Amdahl in production: the global lock that capped a service.** A widely-told pattern (and one I have personally debugged) is the service that scales fine to 4 cores and then flatlines — every request, deep in a shared code path, grabs one global mutex (a shared cache, a metrics registry, a connection pool's lock). That lock is the serial fraction made concrete: while one thread holds it, every other thread waits, so the "parallel" service has a single-file section that no number of cores can widen. The fix is always to *shrink the serial fraction* — shard the lock, make the structure lock-free, or remove the shared state — which is Amdahl's lesson applied: attack the serial part, not the core count. The original Python global interpreter lock is the most famous instance of this exact shape, which is why [the GIL post](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) exists; the language-agnostic lesson is that *any* coarse shared lock is an Amdahl ceiling.

**Gustafson and MapReduce: data-parallel scaling that actually works.** Google's MapReduce (2004) and its open-source heir Hadoop are the canonical proof that data parallelism plus weak scaling scales to thousands of machines. The framework splits an enormous dataset into chunks ("map" runs the same function over each chunk, in parallel, on whatever machines are free) and combines partial results ("reduce"). It is data-parallel by construction — homogeneous workers, the identical map function over different shards — and it lives squarely in Gustafson's world: you don't run MapReduce to make a *fixed* small job faster (the framework overhead would make it slower), you run it because the dataset is huge and grows, and the serial coordination is a tiny fraction of a petabyte-scale job. That is exactly why it scales near-linearly to thousands of nodes where a strong-scaling, fixed-problem framing would have hit Amdahl's wall. The system-design view of this kind of partitioned pipeline is in [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects).

**The C10k problem: concurrency without parallelism, at scale.** In 1999 Dan Kegel framed the "C10k problem" — how does one server handle ten thousand simultaneous connections? The answer that won was *not* ten thousand threads (the per-thread memory and context-switch overhead crushes the machine well before 10k); it was event-driven concurrency — `epoll`/`kqueue` and a single-threaded (or few-threaded) event loop that multiplexes thousands of mostly-idle, IO-bound connections. nginx, Node.js, Redis, and every async runtime since are built on this insight: IO-bound concurrency scales to tens of thousands of connections on a handful of cores precisely because the connections are *waiting*, not *computing*. It is the "concurrent, not parallel" quadrant operating at production scale, and it validates the central claim of this post — that the right tool follows the bound, not the core count. The event-loop machinery is dissected in [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines).

## When to reach for this (and when not to)

This whole post is a decision aid, so let me make the recommendations decisive.

**Reach for parallelism (more cores: processes, a no-global-lock thread pool, a data-parallel framework) when** your workload is CPU-bound *and* has a low serial fraction *and* the problem is large enough that the per-task work dwarfs the spawn and coordination overhead. Hashing a directory of large files, encoding video, training, batch numeric crunching, anything where a profiler shows the cores pinned near 100%. Cap your worker count near the *physical* core count for compute-bound work; past it you pay the overhead floor for nothing.

**Reach for concurrency (an event loop, async/await, threads-for-IO, goroutines) when** your workload is IO-bound — the profiler shows low CPU and the time is going to network, disk, database, or other services. A web server, a scraper, a proxy, a chat backend, a log shipper. Here you can keep thousands of operations in flight on a few cores, and more cores buy almost nothing because the bottleneck was never compute. Threads work for IO-bound concurrency even under a global lock, because the lock is released during the wait.

**Do NOT add threads to a CPU-bound task under a global interpreter lock** expecting a speedup — you will get serialization plus switching overhead, sometimes a slowdown. Use processes or a language without the global lock.

**Do NOT add cores to an IO-bound task** expecting a speedup — the cores will sit idle waiting; use concurrency to overlap the waits instead, on the cores you already have.

**Do NOT expect linear speedup from a job with a meaningful serial fraction.** Compute `$1/(1-p)$` first. If it is 4×, no machine on earth will give you 8×; profile and shrink the serial part before you buy hardware. The single highest-leverage parallel optimization is usually reducing the serial fraction, not adding workers.

**Do NOT keep adding workers past the measured sweet spot.** The scaling curve turns up (slower) past a point set by core count, context-switch cost, and contention. Measure where *your* curve bends and stop there; "more threads" is not a monotone improvement.

**Do NOT parallelize a job that is already fast enough**, or one whose total work is smaller than the spawn-and-coordinate overhead. A serial loop over a thousand small items will beat a parallel one that pays microseconds-per-task of overhead. The simplest correct thing — one thread, one core, in order — is the right default until a measurement says otherwise.

## Key takeaways

- **Concurrency is dealing with many things at once (structure); parallelism is doing many at once (execution).** They are independent axes — you can have either, both, or neither — and conflating them is the root of most misapplied effort.
- **CPU-bound versus IO-bound is the single most useful classification.** Compare wall-clock time to CPU time: if `real ≈ user+sys`, you are CPU-bound; if `real >> user+sys`, you are IO-bound. The diagnosis dictates the tool.
- **IO-bound work wants concurrency** (overlap the waits — thousands of in-flight operations on a few cores); **CPU-bound work wants parallelism** (more execution units — capped near the physical core count).
- **Threads do not give CPU-bound parallelism under a global lock.** Use processes (separate interpreters) or a language without the global lock; threads still help IO-bound work because the lock is released during the wait.
- **Task parallelism** runs different jobs at once (caps at the number of tasks, gated by the slowest); **data parallelism** runs one job over many chunks (scales to millions of lanes — the engine of HPC and ML).
- **Amdahl's law: `$S = 1/((1-p)+p/N)$`, and the ceiling is `$1/(1-p)$`.** A 95%-parallel job caps near 20× — the serial fraction is the tyrant. The best parallel optimization is usually shrinking the serial part, not adding cores.
- **Gustafson's law reframes it: scale the problem with the cores** (weak scaling) and speedup stays near-linear, because the serial fraction shrinks against a growing problem. Strong scaling hits Amdahl's wall; weak scaling does not.
- **There is an overhead floor.** Spawn, coordination, synchronization, communication, and context-switch costs grow with workers, so past a sweet spot more workers means *slower*. Measure honestly — warm up, run many times, report the spread, name the platform.

## Further reading

- Gene Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities" (1967) — the original two-page paper that states the law.
- John L. Gustafson, "Reevaluating Amdahl's Law" (Communications of the ACM, 1988) — the weak-scaling reframing.
- Rob Pike, "Concurrency Is Not Parallelism" (talk, 2012) — the clearest statement of structure-versus-execution.
- Dan Kegel, "The C10k problem" — the essay that framed event-driven IO-bound concurrency at scale.
- Dean and Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004) — data parallelism and weak scaling in production.
- [Why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) — the series intro that motivates everything here.
- [The concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) — the capstone decision framework.
- [The GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) and [multiprocessing: true parallelism and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) — the Python-specific story of the global-lock ceiling and the process fix.

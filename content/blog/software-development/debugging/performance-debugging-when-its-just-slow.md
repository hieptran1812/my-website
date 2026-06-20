---
title: "Performance Debugging: When the Bug Report Just Says It's Slow"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Turn a vague slow into a measured win by profiling the dominant cost, fixing exactly one thing, and proving the p99 dropped."
tags:
  [
    "debugging",
    "software-engineering",
    "performance",
    "profiling",
    "flame-graphs",
    "latency",
    "n-plus-one",
    "observability",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/performance-debugging-when-its-just-slow-1.png"
---

The ticket is two words long. "It's slow." Maybe there is a screenshot of a spinner, maybe a Slack thread where three people have already guessed three different causes. One says it's the database. One says it's "probably the network." One has already opened a pull request adding a cache to a function they have a hunch about. Nobody has measured anything. By the time you pick up the ticket, the team has spent half a day optimizing code that was never the problem, and the endpoint is exactly as slow as it was this morning.

This is the most common way performance work goes wrong, and it is entirely self-inflicted. "It's slow" is a bug report like any other — a symptom, not a diagnosis — and the cardinal sin is the same one we commit with crashes and flaky tests: we guess where the bug is instead of measuring where it is. Performance just makes the temptation worse, because everyone has a pet theory about what's slow, every theory is plausible, and an optimization always feels like progress even when it changes nothing. You can spend a week making code faster and move the user-visible latency by zero, because the code you sped up was never on the hot path.

![A timeline showing the performance debugging loop of reproduce, baseline the percentile, profile to the dominant cost, form one hypothesis, fix one thing, and re-measure to verify the win](/imgs/blogs/performance-debugging-when-its-just-slow-1.png)

The thesis of this post is one sentence: **measure, don't guess.** Profile first, optimize the actual hot spot, verify the win — and verify that the bottleneck didn't just move somewhere else. That is the same `observe → reproduce → hypothesize → bisect → fix → prevent` loop this whole series runs on, applied to latency. We reproduce under representative load, baseline the percentiles, profile to find the dominant cost, form one falsifiable hypothesis, change exactly one thing, and re-measure to prove it. If you have read the intro to this series — [Stop Guessing: The Scientific Method of Debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — this is that method with a stopwatch attached.

By the end of this post you will be able to: read a flame graph and point at the one frame that costs you money; tell the difference between a request that is *busy* on the CPU and one that is *blocked* waiting on something, and pick the right profiler for each; recognize the half-dozen classic culprits — the N+1 query, lock contention, stop-the-world GC, accidental quadratic blowups, the missing index — each by its fingerprint in a profile or a log; and benchmark honestly enough that your "p99 went from 1200 ms to 80 ms" is a number people can trust. We will work two real investigations end to end. Let's start with why guessing fails so reliably.

## 1. Latency is a sum, and only the biggest term matters

Before you can find the slow part, you have to understand what "slow" is made of. The wall-clock time a request takes — the number the user actually feels — is not one thing. It is a sum of every place the request spends time, and those places are made of very different stuff.

![A vertical stack of the terms that add up to total latency: on-CPU compute, I/O wait, lock wait, GC pauses, network round trips, and queueing delay](/imgs/blogs/performance-debugging-when-its-just-slow-2.png)

A single request's latency decomposes, roughly, into: time spent **on the CPU** actually computing (parsing, serializing, looping, hashing); time spent **waiting on I/O** with the CPU idle (a disk read, a database round trip, an HTTP call to another service); time spent **waiting on a lock** (your thread is runnable but parked because another thread holds the mutex it needs); time **stolen by the garbage collector** (a stop-the-world pause that freezes your thread mid-request); time **on the network** (the round trips themselves, plus serialization on the wire); and time spent **queueing** — sitting in line before any work even starts, because every worker is busy.

If you write that as a sum, latency $L = t_{cpu} + t_{io} + t_{lock} + t_{gc} + t_{net} + t_{queue}$, then the whole game becomes obvious: you want to find which term is largest, because optimizing any other term is mathematically capped. This is just Amdahl's law wearing work clothes. If a function you are about to optimize accounts for fraction $p$ of total time, and you make that function infinitely fast — zero — the best possible speedup of the whole request is $1/(1-p)$. Optimize a piece that is 5% of the time and you cannot, even in principle, make the request more than about 5.3% faster. The cache the eager engineer added to a function that was 3% of the time? Its ceiling was a 3% win, and that is *if* the cache hit every time and cost nothing — which it never does.

This is precisely why guessing is so dangerous. Your intuition about which term dominates is shaped by what you happened to work on last, what looks expensive in the source code, and what you are afraid of. None of those correlate with the actual distribution of time. The function that *looks* expensive — a gnarly nested loop with scary indentation — might run on tiny inputs and cost nothing. The innocent-looking line `return jsonify(orders)` might be 70% of the request because `orders` is fat with fields nobody asked for. You cannot see this by reading code. You can only see it by measuring, which is the entire reason profilers exist.

So the very first discipline of performance debugging is to refuse to optimize anything until you know which term in that sum is the dominant one. Everything that follows in this post is a technique for measuring one of those terms. CPU profiling and flame graphs measure $t_{cpu}$. Off-CPU profiling measures $t_{io} + t_{lock} + t_{gc}$ — the blocked time. Query logs and `EXPLAIN` measure the database slice of $t_{io}$. GC logs measure $t_{gc}$ directly. Tracing across services attributes $t_{net}$ to the right hop. Pick the measurement that matches the term you suspect dominates, and let it tell you whether you were right.

A worked sense of scale helps here. Suppose a request takes 1000 ms and you have a rough split: 700 ms serializing a response, 180 ms in the database, 70 ms in auth, 50 ms everywhere else. If you spend a sprint making auth twice as fast, you save 35 ms — the request goes from 1000 ms to 965 ms, a 3.5% win nobody will notice. If instead you cut the serialization in half, you save 350 ms — a 35% win, a different-feeling product. Same effort, ten times the payoff, and the only difference is that you spent the first hour measuring instead of guessing.

There is one term in that sum that behaves very differently from the others, and it is worth pausing on because it ambushes so many investigations: the **queueing** term, $t_{queue}$. The other terms — CPU, I/O, lock, GC, network — are roughly constant per request regardless of how many requests are in flight. Queueing is not. Queueing time depends on *utilization*, and it does so nonlinearly. When a system's busiest resource (a CPU, a thread pool, a connection pool, a single-threaded event loop) is, say, 50% utilized, requests barely wait. As utilization climbs toward 100%, the wait time does not climb gently — it goes vertical. A simple queueing model captures the shape: the average time in the system scales roughly as $1/(1 - \rho)$, where $\rho$ is utilization. At 50% load the factor is 2; at 90% it is 10; at 95% it is 20; at 99% it is 100. The same service that answers in 20 ms at half load can answer in 400 ms at 95% load *without any code getting slower* — purely because requests are stacking up behind each other waiting for a worker.

This is why a service can look perfectly healthy in a profiler on your laptop and fall over in production, and why "we added more traffic and it got disproportionately slower" is not a paradox but a law. It is also why capacity has a cliff: running a pool at 95% utilization to "save money" leaves you one traffic spike away from the latency going vertical. When your latency is dominated by $t_{queue}$, the fix is rarely faster code — it is more capacity, fewer items contending for the scarce resource, or shedding load before the queue builds. A profiler attached to a single request will never show you queueing, because the request that *is* waiting in line is, by definition, not running. You see queueing only by measuring under concurrency and watching latency as a function of load. Keep this term in mind every time someone proposes to "just optimize the code" to fix a system that is slow specifically when it is busy.

## 2. The percentile trap: averages lie about the tail

There is a second way "it's slow" goes wrong before you have even opened a profiler, and it is in how you measure "slow" in the first place. If you summarize latency with an average, you are already lost, because the average is a single number trying to describe a distribution that is almost never symmetric — and for latency, the part that hurts users lives in the tail the average hides.

Consider a thousand requests. Nine hundred and ninety of them take 40 ms. Ten of them take 2000 ms — they hit a slow path, or a GC pause landed on them, or they queued behind a fat request. The average is $(990 \times 40 + 10 \times 2000) / 1000 = 59.6$ ms. Sixty milliseconds! Looks healthy. Ship it. But one request in a hundred takes two full seconds, and on a page that fires twenty backend calls to render, the probability that *at least one* of them is slow is enormous: $1 - 0.99^{20} \approx 18\%$. Almost a fifth of page loads are dragged out by the tail the average swallowed. The user experience is defined by the worst calls, not the typical one.

This is why you report **percentiles**, not averages. The p50 (median) is the typical request — half are faster, half slower. The p99 is the latency that 99% of requests beat; one in a hundred is worse. The p999 (the 99.9th percentile) is the one-in-a-thousand tail, and at scale — a service doing millions of requests an hour — the p999 is not an edge case, it is happening constantly to real people. A healthy-looking p50 with an ugly p99 is the signature of a *tail* problem: a slow path some requests take, queueing under load, a GC pause, a lock some requests fight over while most sail past.

| Metric | What it answers | What hides here |
| --- | --- | --- |
| Average (mean) | "What's a typical number?" | The entire tail; a few slow requests barely move it |
| p50 / median | "What does the middle request feel?" | All tail pain; looks great while p99 burns |
| p95 | "The slow-ish 1-in-20" | The genuinely bad outliers |
| p99 | "The bad 1-in-100" | Still misses the 1-in-1000 cascade at scale |
| p999 | "The pathological tail" | Needs lots of samples to be stable |
| max | "Worst single request seen" | Noisy; one freak GC pause dominates it |

The practical rule: **always baseline p50 and p99 together, under representative load.** The gap between them is itself a diagnosis. If p50 and p99 are close (40 ms and 60 ms), the system is uniformly slow — every request pays the same cost, and you are looking for a fat term in the latency sum that everybody hits. If p50 is great and p99 is terrible (40 ms and 1200 ms), you have a *tail* problem — some requests take a slow path, and your job is to find what's special about the slow ones. Queueing, GC, a code path that only triggers for certain inputs, a lock that only contends under concurrency: these all live in the gap between p50 and p99.

A subtle trap inside the trap: percentiles do not average. You cannot take the p99 of two services and add them to get the p99 of the combined call, and you cannot average p99s across machines to get a fleet p99 — that arithmetic is meaningless. You have to compute percentiles from the raw distribution (or from histograms designed to merge, like HDR histograms). Many a "our p99 is fine" has been a per-host p99 averaged into a lie while the real, globally-computed p99 was twice as bad. Measure the distribution, compute the percentile from it, and report p50 and p99 side by side or you are flying blind. The python-performance series builds this intuition from the ground up in [A Mental Model of Performance: Latency Numbers and the Optimization Loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop), and it pairs well with the rest of this post.

There is a second, even nastier version of the same arithmetic mistake: **tail amplification across fan-out.** Suppose a single page renders by calling ten backend services in parallel and waits for all of them. The page is as slow as the *slowest* of the ten responses. Even if each service has a perfectly respectable p99 of 100 ms, the probability that all ten beat 100 ms is $0.99^{10} \approx 0.90$ — so roughly one page in ten is dragged past 100 ms by *some* service hitting its tail. The page's effective p90 is the services' p99. Fan out to a hundred dependencies, which large systems routinely do across their full call graph, and a per-service p99 of 100 ms means the *typical* page waits on someone's slow tail almost every single time. This is the reason that at scale you cannot tolerate a fat tail anywhere: each service's p99 becomes the next layer's median, and the pain compounds upward through the call graph. It is also why "our service is fast at p50" is cold comfort to the page that calls fifty services — the page lives in your tail, not your median.

A practical corollary for how you instrument: record latency as a histogram, not as a running average or a single gauge. A histogram (Prometheus histograms, HDR histograms, the `summary` types in most metrics libraries) lets you compute *any* percentile after the fact and merge across hosts correctly. A pre-computed average throws away the distribution forever — once you've averaged, the tail is gone and no amount of later analysis brings it back. The cheapest mistake to avoid in all of performance work is logging the wrong summary statistic at collection time, because you cannot reconstruct what you didn't keep.

## 3. Guessing versus the flame graph

Now the heart of it. Let me show you exactly how guessing fails and measuring wins, with a real shape of investigation, because the contrast is the whole lesson.

![A before and after comparison showing a team guessing the database was slow and changing nothing versus profiling with a flame graph and finding the real cost in JSON serialization](/imgs/blogs/performance-debugging-when-its-just-slow-3.png)

Here is the scenario, and it is a true one in spirit — I have debugged this exact shape more than once. An internal API endpoint, `GET /api/orders`, is slow. The dashboard shows p50 around 40 ms, p99 around 1200 ms. The team's instinct fires immediately: it's the database. It is always the database. Someone adds a Redis cache in front of the orders query, ships it, and watches the dashboard. Nothing moves. p99 is still 1200 ms. The cache hit rate is 95%, the query is now served from memory in microseconds, and the endpoint is exactly as slow as before. That is the moment the guess should have died — but instead people start guessing again. Maybe it's the connection pool. Maybe it's the load balancer.

The reason the cache did nothing is that the database was never the dominant term. We just hadn't measured. So we measure. We attach a sampling CPU profiler to the running process under load — for a Python service, `py-spy`, which can profile a process you don't even own without restarting it:

```bash
# Profile the live process for 30 seconds of real load, emit a flame graph.
# --pid attaches to a running process; no code change, no restart.
py-spy record --pid 48213 --duration 30 --rate 200 --output orders.svg

# If you want raw text you can grep instead of an SVG:
py-spy dump --pid 48213
```

The flame graph it produces is the single most valuable artifact in performance debugging, and reading it is a skill worth ten cache layers.

## 4. How to read a flame graph

A flame graph looks intimidating the first time — a jagged skyline of colored rectangles — but it encodes exactly two things, and once you know them you can read any flame graph in any language in about ten seconds.

![A flame graph diagram where the request handler is the full width, serialize_json is seventy percent of the width and the dominant cost, and the database and auth frames are narrow](/imgs/blogs/performance-debugging-when-its-just-slow-4.png)

**Width is time. Depth is the call stack.** That's it. The horizontal axis is not time-ordered — it is not a timeline — it is *total time*, the sum of samples in which that function was on the stack. A frame that is half the width of the graph was on the CPU for half the samples, which means it cost half the time. The vertical axis is the call stack: a frame sitting on top of another means the lower one called the upper one. Depth is just how deep the call chain went; depth costs nothing by itself. **You are hunting for width.** The single widest frame, wherever it sits, is where the time goes. A tall thin tower is a deep call chain that is cheap. A wide flat frame is the cost.

When we look at the `orders.svg` from our slow endpoint, the bottom frame is `handle_request` at 100% width — of course, everything happens inside it. Above it, the width splits. `fetch_rows` (the database call everyone blamed) is 18% of the width. `check_auth` is 7%. And one frame, `serialize_json`, is **70% of the width** — and stacked on top of it, `encode_field` is a hot leaf burning most of that. Seventy percent of the request's CPU time is spent turning Python objects into JSON. The database, the thing two engineers were certain was the problem, is less than a fifth of the time. The guess was not just wrong, it was pointed at almost the smallest term.

Why was serialization 70%? Because the ORM query was doing `SELECT *` and the response object carried forty fields per order — including a couple of big text blobs and a nested list of line items — and the client only rendered six of those fields. We were serializing ten times the data anyone needed, and JSON encoding is roughly linear in the size of what you encode. The fix is not a cache and not an index. It is to fetch and serialize only the six fields the client uses.

#### Worked example: the over-fetched response

The endpoint baselined at **p50 40 ms, p99 1200 ms** under a load test replaying production traffic shape (about 120 requests/second, realistic order sizes). The flame graph from a 30-second `py-spy record` showed **70% of on-CPU samples in `serialize_json` → `encode_field`**, 18% in the DB fetch, 7% in auth. Hypothesis, written down as one falsifiable claim: *the dominant cost is serializing an over-fetched object; trimming the response to the six rendered fields will cut p99 by more than half.* The fix changed exactly one thing — the query selected six columns instead of all forty, and the serializer was given a narrow schema:

```python
# Before: fetch everything, serialize a 40-field object per order.
orders = (
    session.query(Order)
    .filter(Order.customer_id == cid)
    .all()
)
return jsonify([o.to_dict() for o in orders])  # 40 fields, 2 blobs each

# After: fetch only what the client renders; serialize a narrow shape.
rows = (
    session.query(
        Order.id, Order.status, Order.total_cents,
        Order.created_at, Order.customer_name, Order.item_count,
    )
    .filter(Order.customer_id == cid)
    .all()
)
return jsonify([
    {"id": r.id, "status": r.status, "total": r.total_cents,
     "created": r.created_at.isoformat(), "customer": r.customer_name,
     "items": r.item_count}
    for r in rows
])
```

Re-measured under the *same* load test: **p50 18 ms, p99 180 ms.** The p99 dropped from 1200 ms to 180 ms — a 6.7x improvement on the tail — and crucially, a follow-up flame graph confirmed the bottleneck had genuinely moved: `serialize_json` was now 22% of a much smaller total, and the database fetch was the new widest frame. That last check is not optional. The bottleneck always moves when you fix one; you re-profile to find out where it went and decide whether it is now small enough to stop. (The over-fetch-then-serialize anti-pattern is so common in Python web services that the python-performance series treats it as a category; see [CPU Profiling: cProfile and Finding the Hot Path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) for the deterministic-profiler version of the same hunt.)

The tools to make this flame graph vary by runtime, but the reading skill transfers completely:

| Runtime | Capture command | Notes |
| --- | --- | --- |
| Native C/C++/Rust | `perf record -F 99 -g -- ./app` then `perf script \| flamegraph.pl` | System-wide, kernel + user frames; needs symbols |
| Python | `py-spy record -p PID -o out.svg` or `austin` | Sampling, no restart, no code change |
| Go | `import _ "net/http/pprof"` then `go tool pprof -http=:8080 /debug/pprof/profile` | Built-in, flame view in the browser UI |
| JVM (Java/Kotlin/Scala) | `async-profiler` `./profiler.sh -d 30 -f out.html PID` | Low overhead, mixed Java + native stacks |
| Node.js / browser | Chrome DevTools Performance panel, or `node --prof` | Flame chart is time-ordered; flame *graph* is aggregated |

Notice one terminology trap worth flagging: Chrome's "flame chart" and a true "flame graph" are different. A flame *chart* is time-ordered left to right — it shows *when* things happened. A flame *graph* is aggregated by total time — it shows *how much* each function cost in total. For finding the dominant cost you want the aggregated graph; for understanding a single slow request's sequence you want the time-ordered chart. Both are useful; know which one you're staring at.

It is worth understanding the mechanism that makes a sampling profiler trustworthy, because it explains both why it works and where it can mislead you. A sampling profiler does not instrument every function call — that would be a *tracing* profiler, and it would add huge overhead and distort the very timings you are trying to measure. Instead, a sampler interrupts the program at a fixed frequency — say 99 or 200 times a second — and at each interruption it records the current call stack: the full chain of "who called whom" frozen at that instant. After thirty seconds at 200 Hz you have collected roughly 6,000 stack samples. The fraction of those samples in which a given function appears *is* the fraction of time that function was on the CPU, by the law of large numbers. If `serialize_json` shows up in 4,200 of 6,000 samples, it was on the CPU 70% of the time. The flame graph is just those samples folded together: each unique stack becomes a column whose width is its sample count.

This statistical nature has two consequences worth internalizing. First, the overhead is tiny and roughly constant regardless of how fast your functions are — you take one stack snapshot per tick, period — which is exactly why samplers are safe to run in production where tracing profilers are not. Second, sampling has a resolution limit: a function that runs for less time than the gap between samples may be missed entirely or over/under-counted by chance, and you need *enough* samples for the percentages to be stable. Profile for a couple of seconds and the numbers are noisy; profile for thirty seconds under steady load and the wide frames are rock solid. The fix for noise is always more samples — longer duration or higher rate — not a different tool. And a frequency chosen to be a prime number like 99 Hz rather than a round 100 Hz is a deliberate trick: it avoids aliasing with periodic activity in the system (timers, frame loops) that ticks at round intervals and would otherwise correlate with your sampling and bias the result.

One more reading skill separates people who can use flame graphs from people who just look at them: the difference between *self* time and *total* time. A frame's total width includes everything its children did — `handle_request` is 100% wide because it contains the whole request. That tells you where to *descend*, not where the cost *is*. The actual cost lives in the **leaves** and in *self time* — the slice of a frame's width that is not covered by any child sitting on top of it. A frame that is wide but fully covered by a child is just a pass-through; the child is the real cost. A wide frame with a lot of exposed top edge (no children, or thin children) is burning CPU itself. So you scan for width to find the expensive *subtree*, then look for the widest *leaf* or the widest *exposed self-time* within it to find the line that actually costs money. In our example, `serialize_json` is wide, but the exposed hot leaf is `encode_field` — that is the function to attack.

## 5. On-CPU versus off-CPU: the "slow but the CPU is idle" case

There is a whole class of slowness that a CPU profiler is structurally blind to, and missing it is one of the most common ways performance investigations stall. A CPU profiler samples what is *running on the CPU*. If your request is slow because it is **blocked** — parked, waiting on a disk read, a database round trip, a network call, or a lock another thread holds — then during all that waiting your thread is *off the CPU*, and the CPU profiler sees nothing. You profile a request that takes 900 ms, the flame graph shows 40 ms of actual CPU work, and the other 860 ms is simply absent. The profiler isn't broken. The time was spent not-running.

![A decision tree that routes a slow request by CPU usage, sending CPU-bound requests to a CPU flame graph and CPU-idle requests to off-CPU, lock, or GC analysis](/imgs/blogs/performance-debugging-when-its-just-slow-5.png)

So the very first fork in any latency investigation is: **is the request busy or blocked?** Look at CPU utilization while the slow request runs. If a core is pinned near 100%, the request is doing genuine on-CPU work, and a CPU flame graph will find the wide frame — that's the case we just handled. But if the CPU is near idle while the request crawls, the time is being spent *waiting*, and a CPU profiler will tell you nothing useful. You need a different instrument: an **off-CPU profile**, which measures where threads go to sleep and how long they stay asleep.

Off-CPU profiling flips the question. Instead of "where is the CPU spending cycles?" it asks "where do threads block, and for how long?" On Linux you can build an off-CPU flame graph with eBPF — the kernel records every time a thread is scheduled off the CPU and the stack at that moment:

```bash
# Off-CPU flame graph: where threads BLOCK and for how long (microseconds).
# Requires root and a recent kernel with eBPF.
offcputime-bpfcc -df -p $(pgrep -n myservice) 30 > offcpu.folded
flamegraph.pl --title="Off-CPU time" --countname=us offcpu.folded > offcpu.svg

# Quick triage without flame graphs: what syscalls is it blocked in?
strace -f -c -p $(pgrep -n myservice)   # -c summarizes time per syscall
```

The `strace -c` summary alone is often enough to break the case open. It prints a table of every syscall the process made, the total time spent in each, and the count. If 95% of the time is in `recvfrom` or `read` on a socket, your request is blocked waiting for a downstream service or the database — that is $t_{io}$ dominating, and the next step is to trace *which* downstream call. If the time is in `futex`, your threads are blocked on locks — that is $t_{lock}$, and you have contention. The syscall-level view of what a process is actually doing is so useful for this that it gets its own treatment in [Seeing What a Process Really Does: Syscall Tracing](/blog/software-development/debugging/seeing-what-a-process-really-does-syscall-tracing); reach for it whenever the CPU looks idle but the clock keeps ticking.

This on-CPU versus off-CPU split is the single most clarifying distinction in performance debugging, and most people who "can't find the slow part with a profiler" are using a CPU profiler on a blocked workload. The decision is mechanical: check CPU utilization first. Busy means CPU flame graph. Idle means off-CPU, and then within off-CPU you split again — blocked in a syscall is I/O or network, blocked in a futex is a lock, frozen in lockstep across all threads is GC. Three different fixes hide behind one symptom ("slow"), and the off-CPU profile is what tells them apart.

There is a particularly treacherous variant of the blocked case that deserves its own warning: the **event-loop stall**. In a single-threaded async runtime — Node.js, Python's `asyncio`, a Go program where one goroutine hogs a thread — the model is that the loop juggles thousands of concurrent operations by never blocking, handing control back to the loop at every `await`. The whole illusion of concurrency depends on no single task ever holding the loop for long. The moment one callback does something synchronous and slow — a 50 ms JSON parse of a huge payload, a synchronous file read, a tight CPU loop with no `await` — it freezes the *entire* event loop. Every other in-flight request, no matter how cheap, waits behind that one blocking call, because there is only one thread and it is busy. The symptom is bizarre at first glance: thousands of requests, all of which should be fast, all spiking in latency at the same instants, while CPU usage on the box is moderate. The off-CPU profile of any individual request shows it "waiting," but the real culprit is a *different* request that monopolized the loop. The fix is to move the blocking work off the loop — to a worker thread or process pool — so the loop stays responsive. The diagnostic tells: in Node, the event-loop lag metric (how long the loop took to come back around) spikes; in Python `asyncio`, `loop.slow_callback_duration` warnings fire. Knowing this pattern exists is half the battle, because it is invisible to anyone who assumes "async means it can't block."

A note on the stress-test angle, because this is where real systems get interesting. What if the request is fast in isolation and only slow *under load*? Then the dominant term is almost certainly queueing or lock contention — terms that are zero when one request runs alone and explode when many run concurrently. You will never see them by profiling a single request on your laptop. You have to reproduce under representative concurrency, which is why "reproduce under representative load" is the first step of the loop and not an afterthought. What if it's only slow on one host? Then suspect that host's hardware, a noisy neighbor, or a local resource (a full disk, a saturated NIC) — and your off-CPU profile on that host will show a wait that the healthy hosts don't have. What if it's only slow after the process has been up for six hours? Then suspect a slow leak (a cache that never evicts, a connection pool that grows), fragmentation, or a data structure that degrades as it fills — and the tell is that a restart "fixes" it temporarily while the real bug is still there. What if you can't attach a profiler in prod at all? Then `py-spy`/`async-profiler`/eBPF sampling — which attach to a running process with low overhead and no restart — are exactly the tools designed for that constraint; they are how you profile the patient without stopping its heart.

## 6. The classic culprits, each by its fingerprint

Most production slowness is not exotic. It is one of a small handful of recurring culprits, and the enormous practical advantage is that **each one has a distinctive signature** — a telltale shape in a profile, a log, or a scaling experiment. Once you have the fingerprints memorized, "it's slow" becomes a quick triage rather than an open-ended hunt.

![A matrix mapping each slow symptom such as N plus one queries, lock contention, GC pauses, accidental quadratic, and missing index to how it shows up, how you spot it, and the targeted fix](/imgs/blogs/performance-debugging-when-its-just-slow-6.png)

Here is the catalog, each entry framed as *how it shows up → how you spot it → the fix*, because that is the workflow.

**The N+1 query.** Shows up as: latency that grows with the size of the data, often fine in dev (small tables) and terrible in prod (big tables). How to spot it: turn on the query log and count round trips — you will see one query to fetch a list, then *one more query per row* in the list (1 + N). The ORM made it invisible by lazy-loading a relationship inside a loop. The fix: replace the N follow-up queries with a single join or an eager `IN (...)` load. This is the number-one web performance bug, full stop, and it gets its own worked example below.

**Lock contention.** Shows up as: throughput that refuses to scale with cores — you add machines or threads and the system gets no faster, sometimes slower. How to spot it: an off-CPU or lock profile shows threads piling up blocked on the same mutex; in the syscall view, lots of time in `futex`. The fix: shrink the critical section (do less while holding the lock), shard the lock so different keys use different locks, or switch to a lock-free or read-optimized structure. The tell is the *flat scaling curve*: if doubling the cores doesn't roughly double throughput, something is serializing you.

**GC pauses.** Shows up as: p999 spikes while the CPU and the code both look fine — the median is great, but one request in a thousand freezes for tens or hundreds of milliseconds. How to spot it: GC logs (enable them — `-Xlog:gc` on the JVM, `GODEBUG=gctrace=1` in Go) show stop-the-world pauses that line up in time with the latency spikes, and a high allocation rate is the upstream cause. The fix: cut the allocation rate (reuse buffers, avoid per-request garbage, pool objects), or tune the collector — but reducing allocation is almost always the real fix. GC pressure is a *tail* phenomenon, which is exactly why it hides from the average and shows up in p999.

**Accidental O(n²).** Shows up as: time that grows *super-linearly* — double the input and the time quadruples. Fine on test data, catastrophic in production. How to spot it: the scaling experiment. Run it on input of size $n$, then $2n$, then $4n$, and plot the time. Linear work doubles; quadratic work quadruples; that 4x-per-2x is the unmistakable fingerprint. The usual cause is a nested loop, or a membership test against a list (`x in some_list` is $O(n)$) inside a loop, making the whole thing $O(n^2)$. The fix is almost always a better data structure — a `set` or `dict` turns the $O(n)$ membership test into $O(1)$ and collapses the loop to linear.

**Missing index.** Shows up as: one specific query that is slow, and slower as the table grows. How to spot it: `EXPLAIN ANALYZE` the query and look for a sequential scan (`Seq Scan`) over a large table where you expected an index lookup. The fix: add an index on the filter or join column. The database goes from reading every row to jumping straight to the matching ones.

| Culprit | Latency term | Fingerprint | First fix |
| --- | --- | --- | --- |
| N+1 query | $t_{io}$ / $t_{net}$ | 1 + N rows in the query log; grows with data | One join or eager load |
| Lock contention | $t_{lock}$ | Throughput flat as cores rise; `futex` in `strace` | Shrink/shard the critical section |
| GC pauses | $t_{gc}$ | p999 spikes; GC log pauses; high alloc rate | Reduce allocation rate |
| Accidental O(n²) | $t_{cpu}$ | Time grows 4x when input grows 2x | Better data structure (set/dict) |
| Missing index | $t_{io}$ | `Seq Scan` in `EXPLAIN`; one slow query | Index the filter column |
| Cache misses / bandwidth | $t_{cpu}$ | High cycles-per-instruction; `perf stat` cache-miss rate | Improve data locality |
| Serialization overhead | $t_{cpu}$ | Wide `serialize`/`encode` frame in flame graph | Trim payload; faster codec |
| Chatty network | $t_{net}$ | Many small round trips in a trace | Batch; coalesce calls |

Two more deserve a sentence each because they are sneaky. **Cache misses and memory bandwidth**: code that is on-CPU but slow because it is thrashing the memory hierarchy — `perf stat` will show a high cache-miss rate and low instructions-per-cycle, and the fix is data locality (arrays of structs versus structs of arrays, sequential versus random access). **Serialization overhead**: the culprit in our worked example above — turning objects into bytes (JSON, protobuf, pickle) is pure CPU and scales with payload size, so an over-fetched response makes you pay twice, once in the database and again in the encoder. Each of these has a fingerprint; none of them requires you to guess.

The accidental-quadratic culprit deserves a worked example of its own, because its fingerprint — the scaling experiment — is the single most powerful diagnostic for any "it gets slow as the data grows" report, and it requires no profiler at all.

#### Worked example: the batch job that grew super-linearly

A nightly job reconciled two lists — incoming transactions against known accounts — and flagged transactions whose account wasn't in the known set. It ran in 8 seconds when it was written. A year later it took 40 minutes, and the on-call engineer's first instinct was "the database must be slow" (it wasn't; the job was pure in-memory Python). The bug report was the classic shape: *fine at launch, catastrophic as data grew.* Instead of guessing, we ran the **scaling experiment** — feed it $n$, then $2n$, then $4n$ inputs, and watch the time:

```python
import time

def reconcile(transactions, known_accounts):
    flagged = []
    for txn in transactions:
        # `known_accounts` is a LIST, so `in` scans it: O(n) per check.
        if txn.account_id not in known_accounts:   # the quadratic line
            flagged.append(txn)
    return flagged

for n in (10_000, 20_000, 40_000, 80_000):
    txns = make_transactions(n)
    accounts = make_accounts(n)            # also grows with n
    t0 = time.perf_counter()
    reconcile(txns, accounts)
    print(f"n={n:>7}  {time.perf_counter() - t0:6.2f}s")
```

The output was the unmistakable fingerprint of $O(n^2)$:

| Input size $n$ | Time | Ratio vs previous |
| --- | --- | --- |
| 10,000 | 0.21 s | — |
| 20,000 | 0.83 s | 3.9x |
| 40,000 | 3.31 s | 4.0x |
| 80,000 | 13.4 s | 4.0x |

Double the input, *quadruple* the time — that 4x-per-2x is quadratic, full stop. Linear work would have shown 2x-per-2x. The cause was the line `txn.account_id not in known_accounts`: `known_accounts` was a Python *list*, and the `in` operator on a list is a linear scan, $O(n)$. Running it once per transaction, with the account list also growing, made the whole loop $O(n^2)$. The fix was one word — make `known_accounts` a `set`, turning the membership test from $O(n)$ to $O(1)$:

```python
def reconcile(transactions, known_accounts):
    known = set(known_accounts)          # O(n) once, then O(1) lookups
    return [t for t in transactions if t.account_id not in known]
```

Re-running the scaling experiment after the fix: the time grew *linearly* (2x-per-2x), and at the production size of about 600,000 records the job dropped from **40 minutes to under 3 seconds** — roughly an 800x improvement, because at that scale $n^2/n = n$ is itself enormous. We verified the win by re-running the scaling experiment and confirming the line was now straight: doubling the input doubled the time, not quadrupled it. The lesson generalizes hard: any `x in list`, `list.index(x)`, or `del list[i]` inside a loop over the same-sized data is a quadratic landmine, and the scaling experiment finds it in four runs with nothing but a stopwatch. The python-performance series treats this whole family — wrong data structure, hidden linear scan, quadratic accumulation — in depth; it is the cheapest, highest-leverage class of fix in this entire post.

## 7. The N+1 query, the number-one web performance bug

The N+1 query deserves its own section because it is the single most common serious performance bug in web applications, it is almost always introduced by accident, and it is invisible in the source code. You have to see the round trips to believe it.

![A grid showing an N plus one pattern where the app issues one list query for eight hundred rows then loops issuing one user query per row, producing eight hundred and one serial round trips, fixed by a single join](/imgs/blogs/performance-debugging-when-its-just-slow-7.png)

Here is the mechanism, because the *why* is the whole point. Modern ORMs map a database row to an object and let you navigate relationships as attribute access. So you write what looks like innocent, readable code:

```python
# Looks fine. Is a disaster.
orders = session.query(Order).filter(Order.shipped == False).all()  # 1 query
for order in orders:
    # order.customer triggers a SEPARATE query, lazily, per order.
    print(order.id, order.customer.name)   # +1 query, EVERY iteration
```

The first line runs one query and returns, say, 800 unshipped orders. Then the loop accesses `order.customer` — and because the relationship is *lazy-loaded*, the ORM quietly issues a brand-new `SELECT * FROM customers WHERE id = ?` for each order. That is 800 additional queries, each one a full network round trip to the database, executed *serially* one after another. One logical operation ("list orders with their customers") becomes 1 + 800 = 801 round trips. If each round trip is even 1.5 ms of network and database time, that's 1.2 seconds of pure round-trip latency, and none of it shows up as CPU — it is $t_{io}$ accumulated 801 times.

The reason it hides: in development, the `orders` table has 12 rows, so it's 1 + 12 = 13 queries, runs in 4 ms, nobody notices. In production with 800 unshipped orders it's 801 queries and the endpoint takes over a second. "The API got slow after the data grew" is the classic N+1 bug report, and the cause is data size multiplying an invisible per-row query.

How you spot it: turn on SQL logging and *count*. In SQLAlchemy, `echo=True` on the engine; in Django, inspect `connection.queries`; in Rails, the log shows every query. You will see the same `SELECT ... WHERE id = ?` repeated hundreds of times with different ids — that repetition *is* the fingerprint. Or instrument it directly:

```python
# Count queries per request to catch N+1 in tests and prod sampling.
from sqlalchemy import event

query_count = 0

@event.listens_for(engine, "before_cursor_execute")
def count_queries(conn, cursor, statement, params, context, executemany):
    global query_count
    query_count += 1

# After handling a request:
#   if query_count > 50: log.warning("possible N+1: %d queries", query_count)
```

The fix is to fetch the related data in *one* query instead of N. With a join (eager loading), the ORM issues a single `SELECT` that pulls orders and their customers together:

```python
# Eager-load the relationship: ONE query with a join, not 1 + N.
from sqlalchemy.orm import joinedload

orders = (
    session.query(Order)
    .options(joinedload(Order.customer))   # JOIN customers in one round trip
    .filter(Order.shipped == False)
    .all()
)
for order in orders:
    print(order.id, order.customer.name)   # no extra queries; already loaded
```

801 round trips collapse to 1. And the supporting fix is an index: the join needs an index on `customers.id` (usually the primary key, so it's there) and, if you filter or join on `orders.customer_id`, an index on that column too, so the database can find matching rows without scanning the whole table.

![A before and after comparison showing eight hundred and one serial round trips at p99 eleven hundred milliseconds collapsing to a single indexed join at p99 forty milliseconds](/imgs/blogs/performance-debugging-when-its-just-slow-8.png)

#### Worked example: the API that got slow after the data grew

The report: "the orders endpoint was fast at launch, now it's slow, and it gets worse every week." That last clause — *worse as data grows* — is the N+1 tell, written right into the bug report. Baseline under load: **p50 80 ms, p99 1100 ms**, and a load test confirmed latency rose roughly linearly with the number of unshipped orders. Turning on `echo=True` and replaying one request, the log showed **1 list query followed by 800 single-row customer lookups — 801 round trips**, exactly the fingerprint. Hypothesis, written down: *the dominant cost is N+1 round trips to fetch customers per order; a single eager-loading join plus an index on `customer_id` will collapse it to one round trip and flatten the latency curve.*

The fix changed exactly one thing in the query — added `joinedload(Order.customer)` — plus a migration adding an index on `orders.customer_id`. Re-measured under the same load: **p50 22 ms, p99 40 ms.** The p99 went from 1100 ms to 40 ms — a 27x improvement — and, critically, the latency curve flattened: doubling the order count no longer doubled the time, because there was now one round trip regardless of row count. We verified the win two ways: the query log showed 1 query instead of 801, and the load test's latency-versus-data-size line went from a rising slope to flat. The database side of this — why a query is fast in dev and slow in prod, and how the planner chooses a scan versus an index — is exactly the territory of [Why Queries Are Fast in Dev and Slow in Prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod) and [Reading EXPLAIN ANALYZE Like a Staff Engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer), and the index mechanics live in [B-Trees: How Database Indexes Work](/blog/software-development/database/b-trees-how-database-indexes-work).

The general lesson of N+1 generalizes far past ORMs: any time a loop does I/O per iteration — one HTTP call per item, one cache lookup per item, one file read per item — you have an N+1 in disguise, and the fix is the same in spirit: batch the I/O so N round trips become one. A loop that calls a microservice once per item is an N+1 against that service; the trace will show it as a fan of identical calls. Whenever you see per-iteration I/O, your alarm should go off.

## 8. Benchmark honestly, or your numbers are fiction

Everything above ends in "re-measure to verify the win," and that step is where good performance work goes to die if you're sloppy, because measuring latency is genuinely hard to do honestly. A benchmark that lies is worse than no benchmark — it gives you false confidence to ship a regression. So before we talk workflow, the rules of honest measurement.

**Warm up first.** The first runs of almost anything are unrepresentative: caches are cold, the JIT hasn't compiled the hot path yet, connection pools are empty, the page cache hasn't loaded your data. If you measure the first hundred requests you're measuring startup, not steady state. Run the system under load for a warm-up period, *discard* those samples, then measure. A JVM benchmark that doesn't warm up is measuring the interpreter, not the compiled code, and can be 10x off.

**Report the distribution, not one number.** Run the workload many times and report p50 and p99 with the spread, not a single "it took 43 ms." A single number hides variance, and variance is information — if your measurements swing from 40 ms to 400 ms run to run, you have a tail problem *in the benchmark itself* that you need to understand before you trust any conclusion. Use a real load tool (`wrk`, `k6`, `hey`, JMeter) that reports percentiles, not a stopwatch around one call.

**Use representative data and load.** This is the one people skip and it invalidates everything. An N+1 is invisible on 12 rows. A quadratic blowup is invisible on a tiny input. Lock contention is invisible at concurrency 1. If you benchmark on toy data at low concurrency you will measure a system that does not have the bug you're chasing — and "fix" nothing while reporting a win. Reproduce the *shape* of production: realistic data sizes, realistic concurrency, realistic request mix.

**Change one thing.** If you change the query *and* add a cache *and* bump the pool size in one deploy and latency improves, you have learned nothing about *which* change helped — and you may be carrying a regression masked by a win. One change, re-measure, attribute the delta. This is the same discipline as bisection: isolate the variable.

**Watch for the bottleneck moving.** When you fix the dominant term, the next-largest term becomes the new dominant one. Your fix might cut p99 in half and reveal that the database is now the bottleneck — or it might reveal that you've shifted load onto a downstream service that now falls over. Re-profile after the fix, not just re-measure the top-line number. The number can improve while the system gets more fragile.

| Pitfall | What it does to your number | The fix |
| --- | --- | --- |
| No warm-up | Measures startup/cold caches, often 2–10x slow | Run a warm-up phase, discard it |
| Reporting the average | Hides the entire tail | Report p50 and p99 |
| Toy data | The bug isn't present to measure | Use production-shaped data sizes |
| Concurrency of 1 | Lock/queue contention is absent | Load test at realistic concurrency |
| Changing many things | Can't attribute the delta | One change per measurement |
| Not re-profiling | Miss that the bottleneck moved | Re-profile, not just re-measure top line |
| Coordinated omission | Load tool stalls, under-reports tail | Use a tool that corrects for it (`wrk2`) |

That last row is subtle and famous enough to name: **coordinated omission.** Many load generators send a request, wait for the response, *then* send the next — so when the server is slow, the load generator slows down too and simply doesn't send the requests that would have hit the slow window. The result is a benchmark that systematically under-reports the tail, sometimes by an order of magnitude, because the slowest moments get the fewest samples. Gil Tene's `wrk2` and a few other tools correct for this by sending at a fixed rate regardless of response time. If your benchmark's p99 looks suspiciously good, coordinated omission is a prime suspect.

## 9. The workflow, start to finish

Pull it all together and the workflow is short, and it is the series' loop with measurement bolted onto every step. Internalize this and "it's slow" stops being scary.

**Reproduce and baseline.** Recreate the slowness under representative load — realistic data, realistic concurrency — and record p50 and p99. This is your control. You cannot prove a win against a number you never wrote down, and you cannot fix what you can't reproduce. If it only happens in prod, baseline in prod with low-overhead sampling.

**Profile to find the dominant cost.** Check CPU first: busy means a CPU flame graph, idle means off-CPU/lock/GC analysis. Read the profile for *width* — the dominant term in the latency sum. Do not skip to a fix because you have a hunch; the hunch is exactly what the profile is there to overrule.

**Form one falsifiable hypothesis.** Write it down as a sentence with a prediction: "the cost is JSON serialization of an over-fetched object; trimming to six fields will cut p99 by more than half." A hypothesis you can't disprove is a guess in disguise. The prediction is what you'll check.

**Fix exactly one thing.** Change the single thing your hypothesis names. Not three things. One. So that when you re-measure, the delta is attributable.

**Re-measure to verify — and check the bottleneck didn't just move.** Run the same load test, compare p50/p99 to the baseline, and re-profile. Did the predicted improvement happen? Did the dominant cost shift to something new? If the win is real and big enough, stop. If the bottleneck moved and the new dominant term is worth chasing, repeat the loop. Each pass either confirms a number or it does not count.

**Prevent the regression.** Once fixed, leave a tripwire so it doesn't come back: a query-count assertion in tests (`assert query_count < 10`) catches a re-introduced N+1; a latency budget in CI catches a creeping regression; a p99 alert in production catches the next one before a user files a two-word ticket. Performance, like every bug class in this series, is cheaper to *prevent* than to re-debug. The observability that makes the p99 alert possible in the first place is the subject of [Observability for Debugging in Production](/blog/software-development/debugging/observability-for-debugging-prod), and when the slow hop is in another service entirely you'll want distributed tracing to find which one — the cross-service investigation I cover in the companion piece on debugging across service boundaries (linked from the series index once shipped).

A word on how this loop feels in practice, because the steps look tidy on paper and are messier in a war room. The discipline that matters most is the one people skip under pressure: *writing the hypothesis down before you change anything.* When the page is firing and three people are pasting suggestions into Slack, the temptation is to apply the first plausible fix immediately. Resist it. A sentence — "I believe the dominant cost is X; if I do Y, p99 will drop below Z" — costs ten seconds and converts an argument into an experiment. If the fix lands and Z happens, you were right and you have proof. If Z doesn't happen, you learned that X was *not* the dominant cost, which is real progress — you've eliminated a suspect, exactly as you would in a crash investigation. The teams that resolve performance incidents fastest are not the ones with the best hunches; they are the ones whose hunches are written as falsifiable predictions and checked against a number. Everyone else is just taking turns guessing, and guessing does not converge.

One more practical point about the loop: keep your baseline and your fix on the *same* measurement harness, the same load profile, the same data. The most common way a "win" turns out to be fake is that the before number was measured one way (cold, low concurrency, dev data) and the after number another way (warm, different load, prod-shaped data). If the harness changed, the comparison is meaningless — you might be comparing your fix to noise, or worse, hiding a regression behind a more favorable measurement setup. Lock the harness, vary only the code, and the delta is trustworthy. This is the entire reason the loop says "re-measure under the *same* load" and not just "re-measure."

## War story: the GC pause that froze a trading gateway, and a thundering herd

Two real-shaped stories, because performance failures at scale teach the percentile lesson better than any toy example.

The first is a **stop-the-world GC pause**, a pattern that has bitten essentially every large JVM and Go shop at some point and was a documented contributor to outages at companies running latency-sensitive services. Picture a JVM-based gateway with a p50 of 2 ms — beautifully fast. But once every few minutes, a single request takes 400 ms, and on a busy day those 400 ms freezes line up with downstream timeouts and a cascade of retries. The p50 dashboard is pristine; the team insists the service is "fast." The truth is in the p999, where a 400 ms tail is screaming. The cause: the service allocated a fresh multi-megabyte buffer per request, the allocation rate was enormous, and the garbage collector had to run a stop-the-world pause to keep up — freezing *every* thread, including the one mid-request, for the length of the collection. The CPU flame graph showed nothing wrong (GC barely registers in on-CPU sampling of application code); the smoking gun was in the GC log, where pause durations lined up exactly with the latency spikes, and an allocation profile showed the per-request buffer. The fix was not a GC tuning flag — it was pooling and reusing the buffer so the allocation rate dropped by an order of magnitude, the collector ran far less often, and the p999 tail collapsed. The lesson is the percentile trap made physical: an average-healthy service can be quietly destroying one user in a thousand, and only the tail percentile and the GC log reveal it.

The second is a **thundering-herd retry storm**, the failure mode where the cure becomes the disease. A service slows down briefly — maybe a GC pause, maybe a dependency hiccup. Clients time out. Clients *retry immediately*. Now the struggling service is hit with its normal load *plus* a flood of retries, all at once, all synchronized — a thundering herd. The extra load makes it slower, which causes more timeouts, which causes more retries, and the whole thing spirals into a self-sustaining outage that doesn't recover even after the original trigger is gone. The signature is unmistakable once you know it: a sharp synchronized spike in request rate that correlates with rising latency, and a system that won't recover on its own. The off-CPU and trace view shows requests queueing, the upstream metrics show the retry multiplier, and the fix is in the *clients*: exponential backoff with jitter (so retries spread out instead of synchronizing), retry budgets (cap retries as a fraction of traffic), and circuit breakers (stop hammering a service that's clearly down). This is a performance bug whose root cause is not in the slow service at all but in how everyone reacts to slowness — a reminder that "it's slow" can be a system-level emergent property, not a single function you can profile.

The *why* under the GC story is worth making concrete, because it explains the whole shape of the symptom. A tracing garbage collector has to occasionally determine which objects are still reachable from the program's roots (the stack, globals, registers) so it can free the rest. For some collection phases it needs a consistent snapshot of the heap, and the simplest way to get one is to *stop every application thread* — the stop-the-world pause — walk the object graph, and then resume. The duration of that pause scales with how much live data must be traced and how much garbage must be reclaimed, and the *frequency* scales with the allocation rate: the faster you produce garbage, the more often the collector must run. So a service that allocates a multi-megabyte buffer per request at thousands of requests per second is effectively *commanding* the collector to run constantly, and every run freezes the thread that happens to be mid-request. That is why the fix was reducing the allocation rate rather than tuning a pause-target flag: the flag changes how the pause is scheduled, but the buffer churn is what *forces* the pauses. Cut the garbage, and the collector simply has less to do. This is also why GC pauses are a *tail* phenomenon and never a median one — most requests sail through between collections, and only the unlucky one that is running when the world stops pays the full pause. Average latency barely twitches; p999 screams.

Both stories share the moral of this whole post: the team that *guessed* chased the wrong thing for hours (tuning GC flags that didn't help; restarting the "slow" service that wasn't the root cause). The team that *measured* — read the GC log, watched the p999, traced the retry multiplier — found it in minutes. Measurement is not slower than guessing. It is faster, because guessing has no convergence guarantee and measurement does. And both stories are reminders that the slowest part of a system is frequently not where the code *looks* expensive — it is in the runtime's accounting (GC), in how clients react to slowness (retries), or in a single line of innocent-looking ORM navigation. You cannot read your way to these answers. You measure your way to them.

## How to reach for this (and when not to)

Performance debugging has a cost, and the discipline includes knowing when *not* to spend it.

**Reach for a profiler when you don't know where the time goes — which is more often than you think.** If anyone on the thread is guessing, stop and profile. A 30-second `py-spy record` or an `async-profiler` run is cheaper than an afternoon of arguing, and it settles the question with evidence. The sampling profilers (`py-spy`, `async-profiler`, eBPF) are low-overhead enough to run in production, so "I can't reproduce it locally" is not an excuse to keep guessing.

**Don't optimize without a profile, ever.** This is the cardinal rule restated. An optimization you applied because it "felt slow" has, at best, a coin-flip chance of touching the dominant term, and it adds complexity (a cache to invalidate, a buffer pool to manage) whether or not it helps. Complexity you took on for no measured benefit is pure cost.

**Don't optimize a term that isn't the bottleneck — Amdahl will laugh at you.** Before you spend a day on something, ask what fraction of total time it is. If it's 5%, your ceiling is 5%, and you should go find the 70% frame instead.

**Don't micro-optimize when the architecture is the problem.** Sometimes the answer isn't a faster function but a different shape: cache the whole computed result, precompute it offline, denormalize the data, move the work out of the request path entirely. If the dominant cost is "we recompute this expensive thing on every request," no amount of making the computation faster beats not doing it.

**Don't trust a benchmark you ran once on toy data.** A win you can't reproduce, or a win measured on 12 rows, is not a win. If you can't reproduce the improvement under representative load, you haven't proven anything — you've told yourself a story.

**Don't keep optimizing past "good enough."** Performance work has diminishing returns. Once the p99 is inside the budget the product needs, stop and spend the effort elsewhere. The goal is a fast-enough system, not a benchmark trophy. Knowing when you're done is part of the craft.

**Don't attach a heavyweight profiler to a fragile prod process blind.** Sampling profilers are safe; some deterministic profilers and debuggers add real overhead. On a latency-critical service, prefer the low-overhead sampler, profile a canary or one host, and know your tool's overhead before you point it at the thing handling money.

## Key takeaways

- **"It's slow" is a symptom, not a diagnosis.** The cardinal sin is guessing where the time goes. Measure first.
- **Latency is a sum** of CPU, I/O wait, lock wait, GC, network, and queueing. Find the dominant term; Amdahl's law caps everything else.
- **Report percentiles, never averages.** p50 and p99 together — the average hides the tail, and the tail is what users feel. A healthy p50 with an ugly p99 is a tail problem.
- **Read a flame graph by width.** Width is total time; the widest frame is the cost, however deep the stack. Depth is free; width is money.
- **Check CPU first: busy or blocked.** CPU near 100% means a CPU flame graph. CPU idle means off-CPU, lock, or GC analysis. Using a CPU profiler on a blocked workload is the most common stall.
- **Learn the fingerprints.** N+1 (1 + N in the query log, grows with data), lock contention (throughput flat as cores rise), GC (p999 spikes), accidental O(n²) (time 4x when input 2x), missing index (`Seq Scan`). Each maps to one signature and one fix.
- **The N+1 query is the number-one web perf bug.** Per-iteration I/O is an N+1 in disguise; the fix is always to batch N round trips into one.
- **Fix one thing, then re-measure — and re-profile to see where the bottleneck moved.** Verification is half the loop.
- **Benchmark honestly:** warm up, use representative data and concurrency, report the distribution, change one variable, and beware coordinated omission.
- **Prevent the regression:** query-count assertions, latency budgets in CI, p99 alerts in prod. Cheaper than re-debugging.

## Further reading

- [Stop Guessing: The Scientific Method of Debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro and the `observe → reproduce → hypothesize → bisect → fix → prevent` loop this post applies to latency.
- [Seeing What a Process Really Does: Syscall Tracing](/blog/software-development/debugging/seeing-what-a-process-really-does-syscall-tracing) — `strace`/`bpftrace` for the off-CPU, blocked-on-I/O case where the CPU looks idle.
- [Observability for Debugging in Production](/blog/software-development/debugging/observability-for-debugging-prod) — the metrics, logs, and traces that turn a p99 alert into a diagnosis before a user files a ticket.
- [CPU Profiling: cProfile and Finding the Hot Path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) and [Line and Statistical Profiling: line_profiler and py-spy](/blog/software-development/python-performance/line-and-statistical-profiling-line-profiler-and-py-spy) — the Python profiling toolkit, flame graphs, and the over-fetch hunt in depth.
- [A Mental Model of Performance: Latency Numbers and the Optimization Loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) — latency budgets and the measure-fix-verify loop from first principles.
- [Why Queries Are Fast in Dev and Slow in Prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod) and [Reading EXPLAIN ANALYZE Like a Staff Engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) — the database side of N+1, slow queries, and the planner's scan-versus-index decision.
- [B-Trees: How Database Indexes Work](/blog/software-development/database/b-trees-how-database-indexes-work) — why an index on the filter column turns a sequential scan into a jump.
- Brendan Gregg, *Systems Performance* and the FlameGraph / off-CPU material — the canonical reference for `perf`, eBPF, flame graphs, and the on-CPU versus off-CPU split.
- Gil Tene, "How NOT to Measure Latency" — the definitive talk on percentiles, coordinated omission, and honest benchmarking.

---
title: "Case Study: The Leak That Took Down Prod"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Walk a real slow-resource-leak outage end to end, from the 3am OOM-kill page to the unbounded cache that retained 1.4 GB to the LRU bound and the CI memory test that catches the next one before it ships."
tags:
  [
    "debugging",
    "software-engineering",
    "memory-leaks",
    "production-debugging",
    "observability",
    "git-bisect",
    "root-cause-analysis",
    "heap-profiling",
    "incident-response",
    "resource-leaks",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/case-study-the-leak-that-took-down-prod-1.png"
---

The page came in at 4:07am on a Thursday. I know the exact minute because the PagerDuty incident is still pinned in a Slack channel we now call `#the-cache-incident`, kept around as a teaching artifact. The alert text was blunt: `service=orders pod=orders-7f9c-q4m2 OOMKilled exit_code=137`. Exit 137 is the number you learn to hate on call: it is `128 + 9`, the shell's way of saying the process took a `SIGKILL`, and in a container under a memory cgroup the thing that sends that `SIGKILL` is almost always the Linux OOM-killer. The pod had grown past its 2 GB limit and the kernel reaped it. Kubernetes did its job, restarted the pod, and traffic kept flowing. By the time I had my laptop open and VPN connected, the service was green again. Every dashboard said healthy. If I had been a little lazier, I would have acknowledged the page, muttered something about a flaky node, and gone back to sleep.

I did not go back to sleep, because of one line on the memory dashboard — and that line is the whole reason this post exists. Resident set size, the actual physical memory the process held, was not a spike. It was a *staircase*. Pod memory climbed in a near-straight diagonal from about 220 MB after each restart up to the 2 GB ceiling over roughly two days, then died, restarted back to 220 MB, and started climbing again. Three such teeth were already visible on the seven-day graph, each one ending in an OOM-kill, each one a little less than 50 hours wide. This was not a node problem and it was not a one-off. This was a *leak* — a slow, patient, monotonic accumulation of memory the process would never give back — and a leak does not get better on its own. It gets worse until it crosses a threshold, and then it takes down prod. Over the next 70 hours it would escalate from one pod dying overnight to three pods dying simultaneously at Saturday-evening peak, turning a curiosity into a customer-facing outage.

![A timeline of the incident showing the Tuesday deploy, the first overnight page, repeated OOM-kills, and the eventual root cause three days later](/imgs/blogs/case-study-the-leak-that-took-down-prod-1.png)

This is a case study, so I am going to tell it as a story — beginning, middle, and end — but it is a story with a spine, and the spine is the one this whole series is built on: **observe, reproduce, hypothesize, bisect, fix, prevent.** We will start at the alert and the dashboard (observe), decide how to investigate a leak *safely in production* without pausing the fleet (reproduce), form four competing hypotheses and falsify three of them with evidence (hypothesize), narrow the cause in time and by endpoint with bisection (bisect), bound the leaking cache (fix), and then build the guardrail that turns this from a recurring 3am page into a unit test that fails in CI (prevent). If the loop itself is unfamiliar, read the intro map first, [Stop Guessing: The Scientific Method of Debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), and come back. Everything here is an application of it.

A note on what this is and is not. The numbers, the service names, the stack traces, and the fix are an **illustrative composite** drawn from the way these incidents actually go — not a specific company's documented postmortem. I have changed details and rounded figures, but every technique, command, and order-of-magnitude is real and reproducible. If you have shipped a service that retained references it should have dropped, you will recognize every step. By the end you will be able to look at a climbing-RSS dashboard, decide whether you are even looking at a leak, profile it in production without making the outage worse, walk a retainer chain to the exact line, prove the fix flatlined the curve, and install the alert that catches the next one before a customer ever notices.

## 1. Observe: the dashboard told me it was a leak before I read any code

The single most valuable skill on call is reading the *shape* of a metric, not its value. A value tells you where you are; a shape tells you what is happening. Memory in a healthy long-running service is supposed to look like a **sawtooth**: it rises as requests arrive and allocate working memory, and it falls back toward a baseline after garbage collection (in a managed runtime) or after the request handler returns and frees what it borrowed (in a manual one). The teeth of a healthy sawtooth all return to roughly the same floor. The floor is flat. That flatness is the promise that the process is in steady state — it gives memory back as fast as it takes it.

A leak breaks that promise. A leak is memory the program allocated and then *can no longer reach to free*, or — more subtly in a garbage-collected runtime — memory the program *can* still reach through some reference it forgot about, so the collector is forbidden from reclaiming it. Either way the floor rises. Every GC cycle leaves a little more behind than the last. On the dashboard this turns the sawtooth into a *staircase that only climbs*: the teeth still rise and fall with load, but the bottom of each tooth sits a little higher than the bottom before it, and if you draw a line through the troughs it points straight at the ceiling. The shape is diagnostic. Before I read a single line of code, the shape told me three things: this is a leak and not a spike, it is unbounded (the line does not asymptote toward some higher steady state, it climbs at a roughly constant rate), and it is fast enough to matter (a service that leaks 1 KB a day is a leak you can ignore for years; this one was killing a 2 GB pod every two days).

![A before and after comparison contrasting a healthy sawtooth that returns to baseline against a leaking staircase that climbs across every restart until it OOM-kills](/imgs/blogs/case-study-the-leak-that-took-down-prod-2.png)

Here is the actual climb rate, because the numbers drive everything that follows. After a restart the pod settled at 220 MB. By the next morning it was at about 1.1 GB. That is roughly 900 MB of growth over 24 hours, or **+38 MB/hour**, dead linear — I checked the slope across three independent restart cycles and it was within 10% each time. Linearity is itself a clue. A leak whose rate scales with *accumulated state* (say, a list you keep appending to and occasionally scan) would grow super-linearly or in bursts. A leak whose rate is constant is almost always proportional to *throughput*: every unit of work leaves behind a fixed amount of garbage. At a steady ~3,000 requests/minute, +38 MB/hour works out to about 210 bytes leaked per request. Hold that number; we will use it later to confirm we found the right thing, because a good root cause has to *explain the rate*, not just exist.

The first discipline of observation is to not touch anything yet. The strongest temptation at 4am is to *do something* — bump the memory limit, add a pod, restart the deployment — and every one of those actions destroys evidence or masks the problem. Bumping the limit to 4 GB just moves the OOM-kill from every two days to every four. Restarting resets the climb and throws away the very accumulation you need to profile. The right first move is purely observational: confirm the shape across multiple cycles, capture the slope, and check the *correlated* metrics. Did the leak start at a deploy? I overlaid the deploy markers on the memory graph and there it was — the staircase began the evening of the previous Tuesday, with the v2.34.0 rollout. Before Tuesday: flat floor, healthy sawtooth, weeks of it. After Tuesday: the climb. That overlay is worth a hundred guesses. It does not tell me *what* leaks, but it tells me *when it started*, and "when" is half of "what." For the full discipline of reading prod metrics this way — what to chart, how to overlay deploys, how to tell a leak from a slow degradation — see [Observability for Debugging Prod](/blog/software-development/debugging/observability-for-debugging-prod). The rest of this post is what you do *after* the dashboard has told you it is a leak.

### Why a leak is even possible: the GC can only collect what you cannot reach

It is worth being precise about the mechanism, because the mechanism is what makes the fix obvious later. People say "garbage collection means you can't leak memory in Java/Go/Python," and that is one of the most expensive half-truths in our field. A tracing garbage collector reclaims an object when that object is no longer **reachable** from a set of *roots* — the stack frames of live threads, static fields, registers, JNI handles, and so on. The collector starts at the roots and walks every reference; anything it cannot reach is garbage and gets freed. The crucial word is *reachable*. The collector cannot read your intent. If you put an object into a `static HashMap` and never remove it, that object is reachable from a GC root *forever*, and the collector is not merely allowed to keep it alive — it is *required* to. You have not "forgotten to free" the object the way you would in C; you have done something subtler and just as fatal: you have kept a reference you no longer need, and the runtime faithfully honors it.

That is why GC languages do not eliminate leaks; they change the *shape* of them. In C, a leak is `malloc` with no matching `free`: the allocation is genuinely lost, no pointer to it remains, and a tool like Valgrind's `memcheck` can find it precisely because "no pointer remains at exit" is a decidable property. In Java, Go, or Python, the leak is the opposite — a reference that very much *does* remain, sitting in some long-lived container, growing without bound. There is nothing for Valgrind to flag because nothing is "lost"; every byte is reachable. This is the central idea of the whole memory-leak family, and it is worth internalizing before you ever take a heap snapshot. [Hunting Memory Leaks and Bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) develops it in full across C, Java, and Go; here I will lean on one consequence of it: in a managed runtime, *finding a leak means finding the long-lived container that holds references it should have dropped, and then finding the line that put them there and never took them out.* That sentence is the entire investigation in compressed form.

### Why the kill happened where it did: cgroups, RSS, and the OOM-killer

The other half of the mechanism is *why the pod died the way it did*, and getting this right keeps you from chasing the wrong number. In a container the memory limit is enforced by a control group (cgroup), and the kernel tracks the cgroup's memory usage as the sum of resident pages charged to it — anonymous pages (the heap), page-cache pages it touched, and a few kinds of kernel memory. When that charged total crosses the `memory.limit_in_bytes` you set in the pod spec (here, 2 GB), the kernel does not gently ask the process to slow down. It invokes the OOM-killer *for that cgroup*, picks the process (in a single-process container, the one process), and sends `SIGKILL` — signal 9, uncatchable, no cleanup, no flush, no shutdown hook. The process is simply gone, which is why the exit code is `137` (`128 + 9`) and why you never see a graceful log line at the moment of death. If you have ever wondered why your "graceful shutdown" handler never ran during an OOM, this is why: `SIGKILL` cannot be handled, by design.

Two subtleties bite people here. First, the number the cgroup kills on is *not always the same number your runtime reports as heap usage.* A JVM's reported heap might be 1.4 GB while the cgroup's charged memory is 2.0 GB, because the cgroup also counts off-heap allocations (direct byte buffers, thread stacks, metaspace, the JIT code cache, mapped files) that the heap number excludes. So when you debug a container OOM, watch `container_memory_working_set_bytes` (what the cgroup actually charges and kills on), not just the runtime's heap gauge — they can diverge by hundreds of megabytes, and chasing the wrong one sends you hunting an on-heap leak when the growth is off-heap, or vice versa. In our incident they tracked together because the leak *was* on-heap (a `HashMap` of objects), but I confirmed that early rather than assuming it. Second, the working-set number deliberately excludes *reclaimable* page cache, so a brief spike of file I/O does not look like a leak — which is exactly why the working-set metric is the right one to alert on. The OOM-killer is a blunt instrument operating on a precise accounting; knowing which bytes it counts is what lets you point your profiler at the right pool.

## 2. Decide: how do you investigate a leak in prod without making it worse?

There is a particular flavor of dread that comes from realizing the only place the bug reproduces is the place you absolutely cannot break. The leak grows over *days* of real, organic traffic. I could not reproduce +38 MB/hour on my laptop in five minutes, because the climb is a function of accumulated production traffic with its real distribution of request shapes. The honest situation was: the bug lives in prod, and I have to study it in prod, and prod is currently the thing paying everyone's salary. This is the central tension of [Debugging in Production Without Making It Worse](/blog/software-development/debugging/debugging-in-production-without-making-it-worse), and it deserves its own deliberate decision rather than a reflex.

The wrong move is to attach a heavyweight profiler to a live, load-balanced pod and hope. A full heap dump on a 1.8 GB JVM does a stop-the-world pause that can run several seconds; on a pod actively serving 500 requests/second, several seconds of pause means hundreds of failed requests, blown latency SLOs, and possibly a cascade as upstream retries pile on. Worse, a stop-the-world during an already-stressed memory state can itself trigger the OOM-killer (the dump writer needs scratch memory). So the rule I follow is: **isolate one victim, then study the victim as hard as you like.** Do not experiment on the fleet; experiment on one pod you have deliberately removed from the fleet.

![A flow showing one pod cordoned and drained out of the load balancer so its heap can be snapshotted while the rest of the fleet keeps serving traffic](/imgs/blogs/case-study-the-leak-that-took-down-prod-3.png)

Concretely, the safe-investigation procedure was four steps. First, pick the victim: the pod with the *highest* current RSS, because it is furthest along the leak and its heap will show the accumulation most clearly. `kubectl top pods` ranked them; `orders-7f9c-q4m2` was at 1.8 GB, the oldest survivor of the current cycle. Picking the *most-leaked* pod is not an aesthetic choice — it is signal-to-noise engineering. The leaked container is now 1.4 GB of the 1.8 GB heap, so the thing you are hunting is ~78% of everything, and it will be impossible to miss in the dump. A freshly-restarted pod at 240 MB has the same leak but it is buried under a normal-sized heap where it is only a few percent of the total and easy to overlook. Always profile the worst victim, not a convenient one. Second, drain it: cordon it out of the load balancer so it receives no new requests, but — critically — *do not let it restart*. A restart resets the heap and destroys two days of accumulated evidence.

```bash
# Stop new traffic reaching the victim pod without killing it.
# (Adjust to your service mesh; the principle is: remove from rotation, keep alive.)
kubectl label pod orders-7f9c-q4m2 traffic=drain --overwrite
# Verify it dropped out of the Service endpoints (no longer a backend):
kubectl get endpointslices -l kubernetes.io/service-name=orders \
  -o jsonpath='{range .items[*].endpoints[*]}{.targetRef.name}{"\n"}{end}' \
  | grep q4m2 || echo "victim drained: no longer an endpoint"
```

Third, snapshot it twice with a wait in between, because the *diff* between two snapshots is worth far more than either snapshot alone (we will see exactly why in the next section). Fourth, once you have what you need, let the pod be reaped normally — it has already given you everything. The five healthy pods never noticed; the user-facing impact of the entire investigation was zero. This is the difference between debugging prod and breaking prod: you pay a small, bounded capacity cost (one drained pod) instead of an unbounded reliability cost (a paused or perturbed fleet).

#### Worked example: why one drained pod beats a fleet-wide pause

Put numbers on the trade-off so the decision is not vibes. The service ran 6 pods, each handling about 500 requests/second, so ~3,000 req/s total. Option A, pause the fleet for a 5-second heap dump on every pod: that is `3,000 req/s × 5 s = 15,000` requests delayed or dropped, plus a retry storm from upstream, plus the risk that the dump itself OOMs a pod already near its ceiling. Option B, drain one pod and dump only it: the drained pod's `500 req/s` redistributes across the remaining 5 pods, raising each from 500 to 600 req/s — a 20% bump, comfortably inside headroom — and `0` requests are dropped because the load balancer simply stops routing to the drained one. The dump's multi-second pause now happens on a pod that *nobody is talking to*, so its latency is irrelevant. Same diagnostic information (a full heap from a deeply-leaked pod), roughly `15,000 → 0` failed requests. The capacity cost is real but bounded and recoverable; the reliability cost of Option A is neither. When someone proposes profiling the live fleet, this is the arithmetic to put on the whiteboard.

### The reproduction harness: making a two-day leak appear in an hour

The deeper reproduction problem is *time*. The leak takes two days to kill a pod, and nobody investigates a two-day-feedback-loop bug efficiently. The fix is to *compress* the leak: drive the same growth that takes two days of organic traffic into one hour of replayed traffic, so the diff between two snapshots an hour apart is large enough to read clearly and the verification later is fast enough to iterate on. The way to compress a throughput-proportional leak is simply to push throughput at the victim faster than production does — but with *representative* request shapes, because a leak that is sensitive to which endpoint is hit (and this one was) will not reproduce if you replay the wrong mix.

So I built a small traffic-replay harness from real captured requests. We already logged request lines (method, path, anonymized body hash) in our access logs, so I sampled a representative hour — preserving the real distribution across endpoints — into a 60,000-request batch and replayed it at the drained pod with a fixed concurrency.

```python
# replay.py -- fire a captured, representative batch at the drained pod.
# The point is REPRESENTATIVE shape, not raw speed: a leak sensitive to
# one endpoint only reproduces if that endpoint's share is preserved.
import asyncio, aiohttp, json

async def fire(session, line):
    rec = json.loads(line)                       # {method, path}
    async with session.request(rec["method"], BASE + rec["path"]) as r:
        await r.read()                           # drain body so connections recycle

async def main():
    sem = asyncio.Semaphore(50)                  # match prod concurrency
    async with aiohttp.ClientSession() as s:
        async def bounded(line):
            async with sem: await fire(s, line)
        with open("captured_hour.jsonl") as f:   # 60k representative requests
            await asyncio.gather(*(bounded(l) for l in f))

asyncio.run(main())
```

Replaying that 60,000-request hour at the drained pod reproduced the climb at the same ~36 MB/hour the order-detail endpoint drove in prod — a faithful, on-demand reproduction of a bug that otherwise only appears after two days of waiting. This harness is the workhorse of the whole investigation: it is what made the snapshot diff legible (a big, clean delta over one controlled hour), what let me bisect endpoints by load, and what let me verify the fix by replaying the *identical* batch against the patched build. Build the compressed reproducer early; everything downstream gets faster.

## 3. Hypothesize: four suspects, and the test that kills three of them

With a safely-drained pod in hand, the temptation is to dive straight into a heap dump and start reading. Resist it for sixty seconds and write down your hypotheses *first*, because a hypothesis is what makes the heap dump interpretable. A heap dump is millions of objects; without a question, it is noise. With a question, it is an answer. The discipline here is exactly the scientific method: each hypothesis must be **falsifiable**, and each must come with a *cheap confirming test* you can run before the expensive one. I wrote four.

![A hypothesis tree branching into a memory-leak family and an FD-leak family, with three of the four leaves marked as ruled out and one surviving suspect](/imgs/blogs/case-study-the-leak-that-took-down-prod-4.png)

**Hypothesis 1 — an unbounded in-memory cache.** Something is caching per-request data in a long-lived map and never evicting it. Confirming signature: a single large collection on the heap whose size tracks total requests served, full of entries keyed by something high-cardinality. This is the textbook GC-language leak and it fits a *linear, throughput-proportional* climb perfectly.

**Hypothesis 2 — a goroutine/thread leak holding references.** Each request spawns a background worker (a goroutine, a thread, a timer) that never terminates, and each live worker keeps its captured closure variables alive. Confirming signature: goroutine/thread count climbing in lockstep with memory; the heap dominated by stack-rooted closures rather than one big container.

**Hypothesis 3 — a file-descriptor / connection leak.** An HTTP client or DB connection is created per request and never closed on some path, so sockets pile up in `CLOSE_WAIT` and their kernel buffers and wrapper objects accumulate. Confirming signature: file-descriptor count climbing, many sockets stuck in `CLOSE_WAIT`, and eventually `EMFILE` ("too many open files") rather than (or before) an OOM.

**Hypothesis 4 — an event-listener / callback leak.** A subscription or listener is registered per request on a long-lived event source and never removed, so the source's listener list grows without bound and pins everything the listeners close over. Confirming signature: a long-lived publisher/event-bus object retaining a growing list of callbacks.

These are not equally likely, and the point of the cheap-test-first rule is to spend evidence in the order that eliminates the most uncertainty per minute. The cheapest tests do not require the heap dump at all. **Goroutine/thread count** is one number from `/debug/pprof/goroutine?debug=1` (Go) or `jstack | grep -c '"'` (JVM) — I checked it and it was flat at ~340 across the whole climb. Hypothesis 2 falsified in thirty seconds: if threads were leaking, their count would climb with memory, and it did not. **File-descriptor count** is one number from `ls /proc/PID/fd | wc -l` and the socket states are one command of `ss`. I checked those next.

```bash
# FD count for the victim pod's main process (run inside the pod or via its PID).
ls /proc/$(pgrep -f orders-service)/fd | wc -l
# 412   -> and it stayed ~410-415 across the whole 2-day climb: FLAT.

# Are sockets piling up in CLOSE_WAIT (the FD-leak smoking gun)?
ss -tan state close-wait | wc -l
# 3     -> three. Not hundreds. Not climbing.
```

That settled hypothesis 3 too. **The file-descriptor count was flat at ~412 the entire climb, and `CLOSE_WAIT` sockets stayed in single digits.** This matters enormously, because an FD/connection leak presents with a *staircase RSS too* — the kernel buffers and the Java/Go socket wrapper objects do grow — and it is the single most common false trail in leak hunting. The way you tell the two families apart is exactly this: chart FD count alongside RSS. If RSS climbs and FD count is flat, it is a memory leak; if FD count climbs (and you eventually get `EMFILE` rather than OOM), it is a descriptor leak. I ruled out the FD family on evidence, not intuition, and I am spelling it out because *ruling out the wrong family fast is most of the speed* in these investigations. The full mechanics of descriptor and connection leaks — `lsof`, `/proc/PID/fd`, the `CLOSE_WAIT` ladder, the FD-table limit — live in [Resource Leaks: FDs, Sockets, and Connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections); here it was the *alternative I considered and falsified*, and that falsification is part of the proof that I found the real cause rather than the first plausible one.

![A matrix contrasting a memory leak and a file-descriptor leak across the symptom metric, the eventual killer, and the best tool to reach for](/imgs/blogs/case-study-the-leak-that-took-down-prod-6.png)

So after sixty seconds of cheap tests, two of four hypotheses were dead, both falsified by a flat count where the hypothesis predicted a climbing one. That left hypothesis 1 (unbounded cache) and hypothesis 4 (listener leak) — and the way to distinguish *those* two is the heap dump, because both present as "a long-lived container holding a growing collection of things," and only the dump tells you whether that container is a cache map or a listener list. Now the heap dump has a question to answer, and the question is sharp: *which long-lived object owns a collection whose size tracks total requests served?*

### The honest dead end I am not going to hide

Before the heap dump confirmed anything, I spent a real twenty minutes down a wrong path, and I am including it because the clean retelling where every step is correct is a lie that makes you feel worse about your own messy investigations. My *first* instinct, before I disciplined myself into writing the four hypotheses, was hypothesis 3 — the connection leak — because the previous leak I had personally chased at a different job *was* a connection leak, and we pattern-match on our scars. I had already started writing a `bpftrace` script to count `socket()` syscalls without matching `close()` when I forced myself to do the cheap test first and the FD count came back flat. That flat number saved me an hour of instrumenting the wrong thing. The lesson is not "I am clever"; the lesson is the opposite — my experienced gut sent me at the wrong family, and *only the cheap falsifying measurement pulled me back.* This is why the rule is cheap-test-first and why hypotheses must be falsifiable: your intuition is a hypothesis generator, not an oracle, and the whole value of the method is that it lets cheap evidence overrule expensive intuition.

## 4. Profile: the heap diff and the dominator path to the exact line

Two snapshots, one hour apart, on the drained pod. The reason for *two* and not one is the most important technique in this entire post, so I will state it as a rule: **a single heap dump tells you what is big; a diff of two heap dumps tells you what is growing, and growth is the leak.** A single snapshot of a leaking 1.8 GB JVM is dominated by all sorts of legitimately large things — the JIT code cache, class metadata, a few big-but-bounded caches that are *supposed* to be there. Trying to find the leak in one snapshot is like trying to find a slow drip in a photograph of a flooded basement. But take two snapshots an hour apart and *subtract* them, and everything that is steady-state cancels out. What remains — the objects whose count or retained size went *up* between the two — is, by definition, the thing that does not stop growing. That is the leak, isolated by arithmetic.

For a JVM service the tooling is `jmap` to capture and Eclipse MAT (Memory Analyzer Tool) to diff and walk. For Go it would be `go tool pprof -base profile1 profile2` on two heap profiles; for Python, `tracemalloc.take_snapshot()` twice and `snapshot2.compare_to(snapshot1, 'lineno')`; for Node, two `.heapsnapshot` files loaded into Chrome DevTools with the "Comparison" view. The concept is identical across all of them — *capture, wait, capture, diff* — and the cross-runtime mechanics are covered in [Hunting Memory Leaks and Bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat). Here is the JVM capture:

```bash
# Snapshot A on the drained victim. -dump:live forces a full GC first,
# so only genuinely reachable objects remain -- this is the whole point:
# a leak is reachable, so it survives the GC and shows up in the live set.
jmap -dump:live,format=b,file=/tmp/heapA.hprof $(pgrep -f orders-service)

# ... wait one hour while the (drained) pod sits idle-ish, or replay a
# fixed batch of representative traffic at it so growth is reproducible ...

jmap -dump:live,format=b,file=/tmp/heapB.hprof $(pgrep -f orders-service)
```

Then open both in MAT and run the **histogram comparison**: it lists every class by how many instances and how many bytes it gained between A and B. The leak announces itself as the row at the top of the "delta" column. In our case the top delta rows were, in order: `byte[]` (+~360 MB), `java.util.HashMap.Node` (+~31 MB), `java.lang.String` (+~18 MB), and our own `OrderResponse` (+~22 MB). On their own those classes are generic — `byte[]` and the `HashMap.Node` entry objects are in every Java heap — and a junior reaction is to despair that "everything grew." But the histogram is only the *what*. The question that matters is *who is holding them*, and for that you switch from the histogram to the **dominator tree**.

The dominator tree is the single most powerful view in heap analysis and it is worth understanding *why* it works. Object X *dominates* object Y if every path from a GC root to Y passes through X — meaning if X were freed, Y would become unreachable and freeable too. The "retained size" of X is the total memory that would be reclaimed if X went away: X plus everything X uniquely dominates. So the dominator tree, sorted by retained size, answers exactly the question you care about: *what single object, if I could make it go away, would give me back the most memory?* For a leak, that object is the long-lived container doing the holding. In our heap, one object sat at the top of the dominator tree with a retained size of **1.4 GB**: a `java.util.HashMap` referenced by a static field named `responseCache` on a class called `CacheService`.

![A dominator path running from a static GC root through the responseCache map and its 892 thousand entries down to the exact source line that inserts without evicting](/imgs/blogs/case-study-the-leak-that-took-down-prod-5.png)

From there the **path-to-GC-root** (in MAT, "Merge Shortest Paths to GC Roots," excluding weak/soft references) walks you the rest of the way: `CacheService.responseCache` (the static `HashMap`) → 892,000 `HashMap.Node` entries → each node's key was a `String` request id, each node's value was an `OrderResponse` holding a `byte[]` of the serialized payload, averaging about 1.6 KB retained per entry. Multiply it out: `892,000 entries × ~1.6 KB ≈ 1.4 GB`. The arithmetic *closes* — the dominator's retained size, the entry count, and the per-entry size all agree, and they agree with the rate we measured back in section 1. Remember the +38 MB/hour, the ~210 bytes per request? At ~3,000 req/min that is ~180,000 requests/hour; `180,000 × 1.6 KB ≈ 288 MB/hour` of *gross* insertions, and after accounting for the fraction of requests that hit the cache path and the entries that overwrite duplicate ids, the *net* growth lands right around the +38 MB/hour we charted. **A root cause that explains the rate is a root cause you can trust.** One that merely exists is a suspect; one whose arithmetic reproduces the observed climb is a conviction.

#### Worked example: the diff that found the leak in two snapshots

Concrete numbers, because the technique is the lesson. Snapshot A (taken right after the pod was drained, at ~1.81 GB RSS): the `responseCache` map held 847,000 entries, retained size 1.33 GB. I replayed a fixed batch of 60,000 representative production requests at the drained pod — captured from real traffic, anonymized — over the next hour. Snapshot B (RSS now ~1.85 GB): `responseCache` held 892,000 entries, retained 1.40 GB. The diff: `+45,000 entries, +70 MB`, from `+60,000 requests`. So 45,000 of the 60,000 replayed requests added a new entry (the other 15,000 were duplicate request ids that overwrote), and *zero* entries were ever removed — the entry count only went up. The map's size tracked requests served with a slope of ~0.75 entries per request and a floor that never dropped. That is the unbounded cache, caught red-handed by subtraction. No other object on the heap had a meaningful positive delta. The histogram comparison took ten minutes to capture and one minute to read, and it pointed at one map. Hypothesis 1 confirmed; hypothesis 4 (listener leak) fell out, because the dominator was unambiguously a cache *map keyed by request id*, not an event-source listener list.

### Stress-testing the diagnosis: what if it had been harder?

A clean case study can make this look easier than it is, so let me stress-test the diagnosis against the ways real leaks resist you — because the technique has to survive those cases or it is not a technique, it is a lucky guess.

*What if the leak only reproduced under concurrent load?* Some leaks are not throughput-proportional but *concurrency*-proportional — they only manifest when two requests interleave on a shared structure (a leak in an error path that only fires under contention, or a per-connection buffer that is only orphaned when a client disconnects mid-stream). The single-threaded replay would miss those. The defense is to replay at *production concurrency* (the `Semaphore(50)` in the harness was not arbitrary — it matched the real in-flight count), and if the leak still does not appear, to deliberately inject the adversarial condition: replay with random client disconnects, or with the downstream dependency returning errors, to exercise the error paths where leaks love to hide. Here, the leak was plain throughput-proportional, so simple replay sufficed — but I confirmed that by checking the climb rate matched at both `Semaphore(10)` and `Semaphore(50)`; equal rates meant concurrency was not a factor.

*What if it only reproduced after six hours, not one?* A leak whose growth is sub-linear (it accumulates state that only some later requests touch) can hide in a one-hour window. The tell is in the snapshot diff: if snapshot B over snapshot A shows a delta that is *smaller* than the rate would predict, your window is too short or your batch is unrepresentative. I sanity-checked the +70 MB delta against the +36 MB/hour expectation and they agreed, which is itself evidence the reproduction is faithful. When they disagree, lengthen the window or fix the batch before trusting any conclusion.

*What if it only happened on one host?* Then it is probably not a code leak at all but an environment difference — a different kernel, a different allocator (glibc `malloc` arenas versus jemalloc fragmentation can masquerade as a leak), a noisy neighbor, a config drift. The discriminator is reproducibility: a code leak reproduces on *any* host given the traffic; a one-host leak does not. Ours reproduced on every drained pod across two availability zones, which ruled out environment and kept the hunt on the code. *What if you genuinely cannot attach anything in prod* — a locked-down payments box where you may not run `jmap`? Then you fall back to the lightest possible signal: the working-set slope from metrics you already export, plus a one-time controlled reproduction in a staging replica fed the captured traffic. You can find a leak with nothing but a memory graph and a representative replay; the heavy tools make it faster, not possible-versus-impossible.

The line itself, once MAT named the field, was a thirty-second `grep`:

```java
// CacheService.java, line 84 -- the entire bug, in one method.
public OrderResponse getOrCompute(String requestId, Supplier<OrderResponse> compute) {
    OrderResponse cached = responseCache.get(requestId);   // line 83
    if (cached != null) return cached;
    OrderResponse fresh = compute.get();
    responseCache.put(requestId, fresh);                   // line 84: put, never evict
    return fresh;
}
```

There it is. A cache keyed by `requestId` — which is *unique per request* — so the hit rate is essentially zero (only the rare retry with a reused id ever hits), the cache provides no benefit, and every single distinct request adds a permanent entry. The map is a `static final HashMap`, so it is rooted forever, so the GC can never touch any of it, so it grows by one entry per request until the pod dies. This is not an exotic bug. It is one of the most common leaks in production software, and it got past code review because in isolation the method *looks* like a perfectly reasonable memoization cache. The defect is not in the lines you see; it is in the *invariant that is missing* — there is no bound, no eviction, no TTL — and missing invariants are invisible in a diff. We will come back to that under "prevent," because the systemic fix is to make that invariant impossible to omit.

## 5. Bisect: when did this start, and which endpoint feeds it?

I already knew *where* the leak was. So why bisect at all? Two reasons, and they are different axes of bisection that this series treats as one idea — binary-searching the gap between belief and truth. First, **bisect in time** to find the commit that introduced the regression, because the commit's diff and its message tell you the *intent*, which you need to fix it correctly rather than ripping out something that was load-bearing. Second, **bisect by input** to confirm that this cache is actually the dominant driver of the climb and not merely *a* leak among several, because a service this size could plausibly have more than one and I did not want to ship a fix that only solved 30% of the climb. Bisection is the tool for both. The general technique — turning "somewhere in this range" into "exactly here" in `log2(N)` steps — is the subject of [Binary Search Your Bug With Bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection); here is how both axes played out.

**Bisecting in time with `git bisect`.** The deploy overlay said the staircase started Tuesday evening with v2.34.0, but a release can carry dozens of commits, and "the release that exposed it" is not always "the commit that caused it." The range from the last-known-good tag to v2.34.0 was 312 commits. The beauty of `git bisect` is that I do not have to reason about those 312 commits at all; I have to answer one yes/no question — "does this commit leak?" — about `log2(312) ≈ 9` of them. The trick for a *leak* is writing a test that answers that question fast and deterministically, because the real leak takes days to manifest. I wrote one that drives 50,000 requests through the cache path in-process and asserts the cache stays bounded:

```bash
#!/usr/bin/env bash
# leak-probe.sh -- exit 0 if cache stays bounded, 1 if it leaks.
# Used as: git bisect run ./leak-probe.sh
set -euo pipefail
./gradlew -q test --tests 'CacheLeakProbe' > /tmp/probe.out 2>&1 || true
# The probe fires 50k unique-id requests, then asserts the cache holds
# far fewer entries than requests sent (a bounded cache must evict).
if grep -q 'CACHE_BOUNDED' /tmp/probe.out; then
  exit 0   # good: cache stayed small
else
  exit 1   # bad: cache grew ~1:1 with requests -> leaks
fi
```

```bash
git bisect start
git bisect bad v2.34.0
git bisect good v2.33.0
git bisect run ./leak-probe.sh
# ... 9 automated steps later ...
# a3f1c9e is the first bad commit
```

Nine steps, fully automated, and it landed on commit `a3f1c9e`: *"add response cache to cut p99 on the order-detail endpoint."* The commit message is the confession. Someone was fighting a latency problem on `/orders/{id}` and added a cache to memoize responses — a reasonable instinct — but keyed it on the full request id (unique per call) instead of the order id (the thing actually worth caching), and used a plain `HashMap` with no bound. The cache never hit (wrong key) and never shrank (no bound). The *intent* was a latency optimization; the *effect* was a memory leak. Knowing the intent is what lets me fix it correctly in the next section: I should not delete the cache (latency was a real problem), I should make it cache the right key with a real bound.

#### Worked example: nine bisect steps over 312 commits

The arithmetic of bisection is worth seeing explicitly because it is what makes a 312-commit search feel trivial. Each `git bisect` step halves the remaining range, so the number of steps to localize a single bad commit in a range of `N` is `ceil(log2(N))`. For `N = 312`, `log2(312) ≈ 8.3`, so `ceil` is `9` — exactly the nine steps it took. The harness ran each step automatically: check out the midpoint commit, run `leak-probe.sh` (which fires 50,000 unique-id requests and asserts the cache stayed bounded), and exit 0 or 1 so `git bisect run` knows which half to keep. Each step took about 90 seconds (build plus the 50k-request probe), so the whole search was `9 × 90 s ≈ 13.5 minutes` of wall time, fully unattended. Compare that to reading 312 commit diffs by hand looking for "the one that leaks" — that is hours, and you will miss it, because the leaking line *looks fine in isolation*. The leverage of bisection is that it never asks you to *understand* a commit; it only asks you to *test* one, and a good test is far cheaper than understanding. That is the whole reason this series treats bisection as a core reflex: it converts an `O(N)` reading problem into an `O(log N)` testing problem.

**Bisecting by input.** To confirm this cache was *the* driver and not one of several, I bisected across endpoints by load. The service had four hot endpoints. I drained a fresh pod, replayed traffic for *only* the order-detail endpoint at it for thirty minutes, and watched RSS: it climbed at ~36 MB/hour — almost the entire production rate of 38. Then I replayed the other three endpoints combined at it: RSS climbed at ~2 MB/hour, the normal allocator noise floor. That is a clean bisection of the *input space*: the leak is essentially 95% attributable to one endpoint, the same endpoint commit `a3f1c9e` touched. Two independent bisections — over commits and over inputs — converged on the same place. Convergence from independent directions is how you earn confidence that you have *the* root cause and not *a* contributing factor.

## 6. The fix: bound the cache, and verify the curve flatlines

The fix is almost boring once the root cause is this clear, and that is the sign of a good investigation — the fix should feel inevitable. There were three honest options, and choosing among them is a real engineering decision, not a formality.

| Option | Change | Pro | Con |
| --- | --- | --- | --- |
| Delete the cache | Remove `getOrCompute` entirely | Simplest; leak gone for sure | Reintroduces the p99 latency the commit was fighting |
| Bound it (LRU + TTL) | Cap size, evict oldest, expire entries | Keeps a real cache benefit, bounds memory | Slightly more code; pick sane limits |
| Fix the key only | Key on order id, keep `HashMap` | Cache finally *hits* | Still unbounded over distinct orders -> slower leak, not no leak |

Option 3 is a trap worth naming: fixing the *key* (order id instead of request id) makes the cache actually useful — it would finally get hits — but a plain `HashMap` keyed on order id is *still unbounded* over the lifetime of the process; it just leaks more slowly because there are fewer distinct orders than distinct request ids. A slower leak is still a leak, and it would have come back in a month with a quieter staircase and a harder page. The only option that addresses the *missing invariant* — "this cache must have a bound" — is option 2. I keyed on order id (so it hits) *and* bounded it with an LRU eviction policy and a TTL (so it cannot grow forever). Caffeine is the standard JVM library for this; the equivalent exists in every ecosystem.

```java
// CacheService.java, after the fix. Bounded by BOTH size and age.
private final Cache<String, OrderResponse> responseCache = Caffeine.newBuilder()
    .maximumSize(10_000)                 // hard cap: at most 10k entries, evict LRU
    .expireAfterWrite(Duration.ofMinutes(5)) // and nothing lives past 5 minutes
    .recordStats()                       // so we can alert on hit rate, see "prevent"
    .build();

public OrderResponse getOrCompute(String orderId, Supplier<OrderResponse> compute) {
    return responseCache.get(orderId, id -> compute.get());  // key on orderId, not requestId
}
```

`maximumSize(10_000)` makes the memory ceiling explicit and provable: at ~1.6 KB per entry, the cache can hold at most `10,000 × 1.6 KB ≈ 16 MB`, full stop, regardless of how many requests arrive — bounded by *configuration*, not by hope. `expireAfterWrite` adds a second, independent bound on *age*, so even a slow trickle of distinct keys cannot accumulate. And keying on `orderId` means the cache finally does its job — in the replay it hit ~40% of the time, actually cutting p99 the way the original commit intended. Two bounds and a corrected key, and the diff is six lines.

The fix is a hypothesis like any other, and a hypothesis is worthless until tested. **Verification is not optional and it is not "deploy and watch."** Deploy-and-watch on a leak that takes two days to manifest means a two-day feedback loop and a real risk of another outage if you are wrong. Instead I verified the way you verify any leak fix: reproduce the original climb under controlled load, apply the fix, and watch the curve under *identical* load. On a fresh drained pod running the patched build, I replayed the same captured production traffic — the same 60,000-request batches that grew the cache 45,000 entries an hour before — for twelve hours, and charted RSS.

![A before and after chart contrasting the original plus thirty-eight megabyte per hour staircase against the flat steady memory line after the cache was bounded](/imgs/blogs/case-study-the-leak-that-took-down-prod-7.png)

The before curve climbed +38 MB/hour, dead linear, headed for OOM in ~50 hours. The after curve rose for the first few minutes as the cache filled to its 10,000-entry cap, then went **flat at 540 MB and stayed there for the full twelve hours** — the LRU was now evicting one entry for every one inserted, so net growth was zero. That is the entire proof, and it is the proof you must demand of yourself: not "the code looks right," not "it deployed cleanly," but *the curve that used to climb is now flat under the same load that used to kill it.* The number that defines this whole class of fix is the slope: it went from +38 MB/hour to ~0 MB/hour. A leak fix that does not flatline the curve has not been verified, no matter how clean the diff reads. When I rolled it to prod, RSS settled at ~560 MB across the fleet and the staircase was gone — flat floor, healthy sawtooth, for the weeks since.

#### Worked example: proving "flat" honestly over 12 hours

"Flat" is a claim, and claims need numbers, because eyeballing a graph fools you — a leak of 1 MB/hour looks flat on a 12-hour view but kills a 2 GB pod in three months. So I did not eyeball it. I logged RSS once a minute for the 12-hour replay and fit a line to the troughs (the post-GC floors), because the floor is what matters — the peaks are just transient working memory. The fitted slope was `+0.4 MB/hour`, with a standard error that comfortably included zero; over 12 hours the floor moved from 538 MB to 543 MB, which is allocator and metaspace noise, not a leak. Compare to the before run on the same harness: slope `+38.1 MB/hour`, floor moving from 221 MB to 678 MB over the same 12 hours, no overlap with zero, unmistakable climb. That is what "flat" should mean: a *measured* slope statistically indistinguishable from zero over a window long enough that a real leak would have shown itself. Stating it as a slope with an error bar — not "the graph looks flat" — is the difference between a verified fix and a hope. And it directly informs the alert we are about to build: if `+0.4 MB/hour` is noise and `+38 MB/hour` is an outage, the alert threshold lives somewhere sane in between.

## 7. Prevent: turn a 3am page into a CI failure

Shipping the fix closes the incident. It does not close the *bug*, because the bug is not really "this one cache" — the bug is that our system allowed an unbounded cache to reach production, climb for two days, and OOM-kill the fleet before anyone noticed. If I fix only the cache, I have fixed *this* instance and left the *class* wide open; the next engineer fighting a latency problem will reach for the same `static HashMap` and we will be back in `#the-cache-incident` in six months. Prevention means asking *why the system let this happen* and installing guardrails at each layer where it could have been caught earlier. This is [Root Cause Analysis and the Five Whys](/blog/software-development/debugging/root-cause-analysis-and-the-five-whys) applied for real, and the five-whys ladder is worth walking out loud.

*Why did prod OOM-kill?* Because a cache grew unbounded. *Why was it unbounded?* Because it used a plain `HashMap` with no eviction and no TTL. *Why did an unbounded cache get merged?* Because code review saw a normal-looking memoization method and there was no rule or lint against unbounded long-lived maps. *Why did it run for two days before paging?* Because our only alert was on the OOM-kill itself — a *lagging* indicator that fires when the damage is already done — and nothing watched the *slope* of memory, the leading indicator. *Why did no test catch it?* Because we had no test that runs the service under sustained load and asserts memory stays bounded. Five whys, and notice that only the first answer is about the cache. The other four are about *the system*: a missing lint, a lagging alert, a missing load test. Fix only the cache and four systemic holes remain. The point of five-whys is to climb from the instance to the class, and each rung becomes a guardrail.

![The six phase debugging loop drawn as a stack, mapping observe, reproduce, hypothesize, bisect, fix, and prevent onto the concrete artifact each phase produced in this incident](/imgs/blogs/case-study-the-leak-that-took-down-prod-8.png)

Before the guardrails, one more honest question deserves an answer: *why did code review miss this?* Three people approved commit `a3f1c9e`, all competent, none careless. They missed it because the defect is not in the code that is present — it is in the code that is *absent*. Reviewers read diffs, and a diff shows you what changed; it does not highlight the invariant you failed to add. The method `getOrCompute` looks exactly like ten thousand correct memoization caches; the only thing wrong is a bound that is not there, and "a thing that is not there" does not appear in a diff with a red minus or a green plus. This is the deepest lesson of the incident and the reason prevention cannot be "review more carefully." Human review is structurally bad at catching missing invariants, because absence has no syntax to catch the eye. The fix is to move the check from human attention to machine enforcement — a lint that *knows* a long-lived cache must be bounded and fails the build when it is not. You do not catch a missing bound by looking harder; you catch it by making the unbounded shape refuse to compile or refuse to merge. With that understood, we built four guardrails, one per systemic why, ordered from fastest-feedback to last-resort:

**Guardrail 1 — a CI memory test (catches it before merge).** The cheapest place to catch a leak is before it ships, so we promoted the `leak-probe` from the bisection into a permanent CI test. It drives 100,000 distinct requests through the hot endpoints in-process and asserts that heap usage after a forced GC stays under a threshold. A leak makes it fail; a bounded cache passes. This is the test that, had it existed, would have failed commit `a3f1c9e` in CI and this incident would never have happened.

```python
# ci_memory_test.py -- fails the build if the service leaks under load.
import gc, tracemalloc

def test_hot_path_does_not_leak():
    client = start_test_service()
    warm(client, n=5_000)                 # let caches/JIT reach steady state
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for i in range(100_000):              # 100k DISTINCT requests
        client.get(f"/orders/{unique_order_id(i)}")
    gc.collect()
    after = tracemalloc.take_snapshot()
    # Net growth across 100k requests must be small. A leak grows ~linearly
    # with request count; a bounded cache plateaus.
    grew = sum(s.size_diff for s in after.compare_to(before, "filename"))
    assert grew < 50 * 1024 * 1024, f"leaked {grew/1e6:.1f} MB over 100k reqs"
```

**Guardrail 2 — an RSS-slope alert (catches it hours after deploy, not days).** The OOM-kill alert is a tombstone; it fires after death. We added a *leading* alert on the *slope* of RSS: if the 6-hour linear-regression slope of pod memory exceeds a threshold — we set it at +15 MB/hour, comfortably above the +0.4 noise floor and well below the +38 of the incident — page someone. This would have fired Thursday *morning*, after the first few hours of climb, instead of at 4am after the first death. Alerting on the *derivative* of a metric rather than the metric itself is the general lesson: for anything that should be in steady state, the rate of change is the leading indicator and the absolute value is the lagging one.

```yaml
# Prometheus alert: page on sustained memory growth, the LEADING indicator.
- alert: PodMemorySlopeHigh
  # deriv() over a 6h window: bytes/second of sustained growth.
  expr: deriv(container_memory_working_set_bytes{container="orders"}[6h]) > 4400
  # 4400 B/s ~= 15 MB/hour, above the 0.4 MB/h noise floor, below the 38 MB/h leak.
  for: 30m
  labels: { severity: page }
  annotations:
    summary: "orders pod memory climbing >15 MB/h for 30m -- possible leak"
```

**Guardrail 3 — an explicit max-size invariant (makes the bug shape impossible).** We added a lint/architecture rule: any cache or long-lived collection must be a bounded type (Caffeine, Guava `CacheBuilder`, or an explicitly size-capped structure); a raw `static HashMap`/`ConcurrentHashMap` used as a cache fails review. The principle is to make the *missing invariant* — "this must be bounded" — a thing the code *cannot omit*, by making the unbounded shape un-mergeable. You cannot rely on every future engineer remembering to bound a cache; you make the bound the default and the unbounded version the thing that trips a check. This is the heart of [Building Debuggable Systems](/blog/software-development/debugging/building-debuggable-systems): the best fix for a bug class is a design where that class cannot recur.

**Guardrail 4 — cache hit-rate metric (catches "the cache is useless" directly).** We wired up `recordStats()` and exported cache hit rate. A cache with a near-zero hit rate is either keyed wrong or pointless — exactly this bug's signature — and now it is visible on a dashboard and alertable. Had this metric existed, the ~0% hit rate would have screamed "this cache is broken" from day one, independent of the memory symptom. Each guardrail catches the bug at a different layer, and the layers are *redundant on purpose*: CI catches it before merge, the hit-rate metric catches a wrong key, the slope alert catches a climb hours in, and the OOM alert remains as the last-resort tombstone. Defense in depth means no single guardrail has to be perfect.

## 8. War story: the cache leaks that take down real systems

This composite is fiction in its details but its *shape* is one of the most common real outages in our industry, and it is worth grounding in the patterns behind documented incidents, presented accurately as classes rather than as any one company's report.

The **unbounded-cache OOM** is a genre. The canonical real version is a service that adds an in-memory cache to cut latency, keys it on something effectively unique (a session id, a request id, a full URL with query string, a UUID), and never bounds it — so the "cache" is really an append-only log of every request the process has ever served, and it grows until OOM. It is common enough that the standard caching libraries (Guava, Caffeine, `functools.lru_cache`, Go's `golang-lru`) all default to *requiring* you to set a bound, precisely because the unbounded version is such a reliable foot-gun. The lesson the industry encoded into those libraries is the same one we encoded into guardrail 3: make the bound mandatory.

The **listener/subscriber leak** is the second genre — hypothesis 4, which we ruled out but which is real and nasty. A long-lived event source (an event bus, an observable, a DOM node, a signal handler registry) accumulates subscriptions because something registers a listener per request/component and never unregisters it on teardown. Every live listener pins everything it closes over. This is the dominant leak in long-running single-page web apps (a component subscribes on mount, forgets to unsubscribe on unmount, and after enough navigation the page holds thousands of dead components alive) and in any pub/sub backend where `subscribe` lacks a matching `unsubscribe` on the error path. The retainer chain looks different from a cache leak — it walks through a listener *list* on a publisher rather than a map — which is exactly why the dominator-tree view matters: it tells the two apart instantly.

There is a reason these two genres dominate, and it is worth naming because it tells you where to look first on your *next* leak. Both are violations of the same rule: **anything you add to a long-lived container, you must also have a path that removes.** A cache adds on `put` and must remove on eviction; a subscriber list adds on `subscribe` and must remove on `unsubscribe`; a connection pool adds on `acquire` and must remove on `release`. The leak is always the missing half of a pair — the `put` with no eviction, the `subscribe` with no unsubscribe, the `acquire` with no release on the error path. When you go hunting a leak, do not start by reading code top to bottom; start by listing every long-lived container in the service and asking, for each, *where does something get removed from this, and is that removal guaranteed on every path including errors?* The container whose removal is missing or conditional is your leak. That single question would have found `responseCache` in about two minutes of reading, because the answer for it was "nowhere — nothing is ever removed."

The **slow-resource-leak-that-becomes-an-outage** has a famous structural cousin in the broader reliability literature: the *bounded resource that silently fills*. Thread pools that leak threads until `OutOfMemoryError: unable to create new native thread`; connection pools that leak connections until every request blocks waiting for one that will never return; file-descriptor leaks that hit `EMFILE` and take down a process that "wasn't even using much memory." They share our incident's defining feature — a slow, monotonic climb toward a hard limit, invisible on a "current value looks fine" dashboard, lethal the moment the limit is crossed, often at peak load when headroom is thinnest. The defense is always the same trio we built: bound the resource explicitly, alert on the *slope* not the level, and test under sustained load. The leaks that take down prod are almost never clever. They are ordinary omissions of a bound, running long enough to matter. For how these failures cascade across a service fleet and the postmortem discipline around them, see the system-design material on outages and observability ([Observability for Debugging Prod](/blog/software-development/debugging/observability-for-debugging-prod) is the in-series companion).

## 9. How to reach for this (and when not to)

Every technique here has a cost, and a field manual that does not tell you when *not* to use a technique is selling, not teaching. Here is the honest decision guide.

**Read the shape before you read the code.** This costs nothing and saves the most time. If the metric is a spike that returns to baseline, you do not have a leak — you have a load problem or a GC-tuning problem, and heap-diffing it is wasted effort. Only the monotonic, returns-to-a-higher-floor staircase is a leak. Spend the thirty seconds to classify the shape before you spend an hour on tooling.

**Drain one pod; never profile the live fleet.** Reach for the drain-and-snapshot procedure whenever the bug only reproduces in prod and the diagnostic is heavyweight (a heap dump, a several-second pause, an attached profiler). Do *not* drain a pod when a cheap, safe observation answers the question — if the goroutine count or FD count settles it in one command, you never needed the heap dump at all. And never, ever attach an intrusive debugger or take a stop-the-world dump on a *live, in-rotation* pod serving a latency-sensitive or payments path. The capacity cost of one drained pod is bounded; the reliability cost of perturbing the fleet is not.

**Diff two snapshots; do not stare at one.** Single-snapshot analysis is for when you have a hard ceiling and need to know what is big *right now* (bloat, not leak). For a leak — something that *grows* — always diff. The diff cancels the legitimately-large steady-state objects and isolates the thing that does not stop. Staring at a single dump of a leaking heap is the most common way to waste an afternoon.

**Bisect when the cause's *origin* or *driver* is unclear, not when you already know the line.** I bisected here even knowing the line, but only because I needed the *intent* (to fix it right) and the *attribution* (to be sure it was the dominant driver). If you already have the line *and* understand the intent *and* it is obviously the whole problem, skip bisection and fix it. Bisection is a search tool; do not run a search when you are already standing on the answer.

**Verify by flatlining the curve, not by deploying and hoping.** Always reproduce the climb under controlled load and prove the slope went to zero *before* you trust the fix in prod, especially for slow leaks where deploy-and-watch is a multi-day feedback loop. The one time to skip controlled verification is a true emergency mitigation (bump the limit to stop the bleeding *right now*) — but then treat that as a temporary tourniquet, not a fix, and come back to verify properly.

**Build the guardrail every time; this is the one step nobody regrets.** The CI memory test, the slope alert, the bounded-by-default rule — these are cheap to add in the calm after an incident and they are the difference between learning from a leak and being doomed to repeat it. The only time to defer prevention is if the service is being decommissioned anyway. Otherwise, the page you prevent is worth ten times the page you handle.

## 10. Key takeaways

- **The shape of the memory curve diagnoses the bug before the code does.** A sawtooth that returns to a flat floor is healthy; a staircase that climbs across every restart and never resets is a leak. Classify the shape first; it tells you it is a leak, whether it is bounded, and whether it is fast enough to matter — all before you open an editor.
- **In a GC language, a leak is a reference you kept, not memory you lost.** The collector keeps everything reachable from a root. Finding the leak means finding the long-lived container holding references it should have dropped, and the line that inserted without removing.
- **Investigate prod by isolating one victim, never by perturbing the fleet.** Drain one pod out of rotation, keep it alive, snapshot it. The capacity cost is bounded and recoverable; the reliability cost of a fleet-wide pause is neither.
- **Write falsifiable hypotheses with cheap confirming tests, and run the cheap tests first.** A flat goroutine count killed the thread-leak hypothesis in thirty seconds; a flat FD count killed the descriptor-leak hypothesis in another thirty. Ruling out the wrong family fast is most of the speed.
- **Diff two heap snapshots; the delta is the leak.** A single dump shows what is big; subtraction cancels the steady state and isolates what grows. Then the dominator tree names the one object whose retained size would free the most, and the path-to-root walks to the exact line.
- **A root cause must explain the rate, not just exist.** When the entry count, the per-entry size, and the dominator's retained size all multiply out to the +38 MB/hour you measured, you have a conviction. When they do not, you have a suspect.
- **Bisect over time to learn intent and over input to confirm the driver.** `git bisect run` with a fast leak probe found the commit in nine automated steps; replaying one endpoint confirmed it drove 95% of the climb. Convergence from two independent directions earns confidence.
- **Verify a leak fix by flatlining the slope, measured.** Not "the diff looks right" and not "it deployed" — reproduce the climb under identical load and prove the fitted slope went from +38 MB/hour to statistically indistinguishable from zero over a long-enough window.
- **Prevent the class, not the instance, with five-whys and layered guardrails.** A CI memory test catches it before merge, a bound-by-default rule makes the bug shape un-mergeable, a slope alert catches a climb hours in, and the OOM alert stays as the last-resort tombstone. The leaks that take down prod are ordinary omissions of a bound — so make the bound impossible to omit.

## Further reading

- [Stop Guessing: The Scientific Method of Debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe→reproduce→hypothesize→bisect→fix→prevent loop this case study is one full application of.
- [Hunting Memory Leaks and Bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) — the cross-runtime mechanics of heap snapshots, diffs, dominator trees, and retainer paths in C, Java, Go, Python, and Node.
- [Resource Leaks: FDs, Sockets, and Connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) — the descriptor-leak family we considered and ruled out: `lsof`, `/proc/PID/fd`, `CLOSE_WAIT`, and the FD-table ceiling.
- [Debugging in Production Without Making It Worse](/blog/software-development/debugging/debugging-in-production-without-making-it-worse) — the drain-one-victim discipline and the full toolkit for studying a bug in prod without breaking it.
- [Observability for Debugging Prod](/blog/software-development/debugging/observability-for-debugging-prod) — reading the dashboard, overlaying deploys, and alerting on the slope rather than the level.
- [Binary Search Your Bug With Bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — bisecting in time with `git bisect run` and across inputs to localize a cause in `log2(N)` steps.
- [Root Cause Analysis and the Five Whys](/blog/software-development/debugging/root-cause-analysis-and-the-five-whys) — climbing from the instance to the class so you fix the system, not just the symptom.
- [Building Debuggable Systems](/blog/software-development/debugging/building-debuggable-systems) — designing bounded-by-default resources and leading-indicator metrics so this bug class cannot recur.
- The series capstone, `capstone-the-debugging-playbook` (planned), collects every loop, tool, and guardrail from this series into one reference.
- *Debugging* by David J. Agans, and *Why Programs Fail* by Andreas Zeller — the canonical texts on systematic debugging and the scientific method applied to software faults. The Eclipse MAT documentation on dominator trees and the Caffeine and Guava caching guides are the practical references behind this post's tooling.

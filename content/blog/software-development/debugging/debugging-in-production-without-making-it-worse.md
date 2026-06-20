---
title: "Debugging in Production Without Making It Worse: First, Do No Harm"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to investigate a live production system without turning a degradation into an outage, by ranking every probe by its blast radius and reaching for zero-perturbation reads, py-spy, and bpftrace before you ever pause a process."
tags:
  [
    "debugging",
    "software-engineering",
    "production",
    "observability",
    "py-spy",
    "bpftrace",
    "ebpf",
    "profiling",
    "sre",
    "troubleshooting",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/debugging-in-production-without-making-it-worse-1.png"
---

At 02:41 the pager goes off: the payments service is slow, p99 has tripled, and the on-call channel is already filling with the special kind of quiet that means everyone is staring at the same Grafana panel. You SSH onto a box — which you should not have done, but you did — and your fingers do the thing they have done ten thousand times on your laptop. `gdb -p $(pgrep payments)`. You hit enter. And for the next thirty seconds the process does not serve a single request, because `gdb` stopped all of its threads to give you a prompt. The load balancer's health check times out. The instance gets marked unhealthy and pulled. The requests that were on that box get retried onto the others, which were already hot, which now tip over too. You did not debug the latency spike. You converted it into an outage, and the postmortem is going to have your name in the timeline.

This is the defining hazard of debugging production, and it is the one almost nobody teaches: **the act of looking can be the thing that breaks it.** Every technique that makes you fast on a laptop — attach a debugger, set a breakpoint, crank up logging, take a heap dump, run an exploratory query to confirm a hunch — is, in production, a way to perturb a live system that thousands of users are depending on *right now*. The debugger that is your microscope on a laptop is a sledgehammer in prod, because the thing it does best, stopping the world, is the one thing a live service cannot survive. The verbose log flag that gives you everything you need fills the disk in nine minutes or quadruples this month's logging bill. The heap dump that shows you exactly which object is leaking pauses the JVM for four seconds and doubles its memory while it writes. The "quick query" you ran to check a theory took a table lock and stalled every checkout for the duration.

So production debugging is not a harder version of laptop debugging. It is a *different discipline with a different first principle*, and the principle is the oldest one in medicine: **first, do no harm.** Before you touch anything, you ask not "what will tell me the most?" but "what is the cheapest probe, in terms of how much it perturbs the live system, that can still answer my question?" Every tool you own gets re-sorted along a single axis — **blast radius**, the amount of damage the probe itself can do — and you reach for them strictly from lowest to highest, climbing only when the cheaper rung genuinely cannot answer. The whole toolkit, re-understood, is fundamentally about **low-perturbation, read-only observation**, with the dangerous high-perturbation tools reserved for an instance you have first made safe to break.

![Diagram showing a live production symptom flowing into four rungs of investigation ordered by rising blast radius, from read-only dashboards through sampling and reversible probes to a drained node, all converging on a root cause found with zero user impact](/imgs/blogs/debugging-in-production-without-making-it-worse-1.png)

By the end of this post you will be able to take a live prod symptom — "it's slow," "memory is climbing," "we're throwing 500s on one host" — and run a disciplined, blast-radius-ordered investigation that finds the root cause without making the incident worse. You will know why attaching a debugger freezes the process and why that is an outage, not an inconvenience; you will have a ranked toolkit from zero-perturbation reads through `py-spy` and `bpftrace` up to a core dump on a drained canary; you will know the six rules that keep a single probe from cascading into a self-inflicted outage; and you will have walked two real investigations — a memory bloat found live with sampling, and a latency spike traced with eBPF — both resolved with measured numbers and zero customer impact. This is the same observe → reproduce → hypothesize → bisect → fix → prevent loop the whole [series](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) is built on, but with a hard constraint bolted on top of every step: *the investigation must not become the incident.*

## 1. The mechanism: why a probe in prod is fundamentally different

Before we reach for any tool, we need to be precise about *why* the laptop reflexes are dangerous, because the mechanism dictates the entire strategy. It is not superstition or caution-for-caution's-sake. Each dangerous technique perturbs the running process in a specific, mechanical way, and once you can name the mechanism, the safe alternative becomes obvious.

**A debugger freezes the process — and a freeze is an outage.** When you attach `gdb` or `lldb` to a running process, the operating system sends it a `SIGSTOP`-class signal and the debugger uses `ptrace` to take control. To give you a stable view of memory and let you set breakpoints, it stops *every thread* in that process. While you sit at the `(gdb)` prompt thinking, none of those threads run. On a laptop that is exactly what you want: time stands still so you can inspect it. In production, a service process is shared by hundreds or thousands of concurrent requests, and "time stands still" means every one of those requests is now blocked. The thread pool stops draining. The event loop stops turning. The load balancer's health probe — which expects a response within, say, two seconds — times out, the orchestrator marks the instance unhealthy, and it gets pulled from rotation. The traffic it was carrying retries onto the remaining hosts, which were already loaded, and you have manufactured a cascading failure out of a fact-finding mission. The mechanism is simple and unforgiving: **a debugger's superpower is stopping the world, and a live service is precisely the kind of thing that must never stop.**

**A heap dump pauses the runtime and doubles its memory.** Take a managed runtime — the JVM, CPython, the Go runtime, V8. To produce a consistent snapshot of the heap, the runtime generally has to reach a *safepoint* where no thread is mutating object graphs, which in practice means a stop-the-world pause. A `jmap -dump` on a multi-gigabyte JVM heap can pause the process for several seconds and, while it writes, can transiently need close to double the live-set memory. On a box that was already memory-pressured — which is *exactly* the situation you take a heap dump to investigate — that extra pressure can push it into swap or trigger the OOM killer. The dump that was supposed to explain the memory problem becomes the thing that crashes the process. The mechanism: **capturing the full heap requires a consistent view, a consistent view requires a pause, and the pause plus the copy is itself a load spike on an already-stressed host.**

**A verbose log flag is an unbounded write amplifier.** Flip a service from `INFO` to `DEBUG` or `TRACE` and you have not added a little information — you have multiplied the write rate by ten or a hundred, on every request, on every host, all at once. Three things can break. First, the local disk: debug logging on a busy service can write gigabytes in minutes, and a full disk takes down everything that needs to write to it, including the service itself. Second, the logging pipeline: your log shipper, your aggregation cluster, your indexed log store all have throughput limits, and a fleet-wide debug flag is a self-inflicted denial-of-service against your own observability stack. Third, the bill: cloud log ingestion is priced per gigabyte, and a debug flag left on across a fleet can turn a routine month into a five-figure surprise. The mechanism: **log level is a multiplier on a hot path, and multipliers on hot paths compound across requests, hosts, and downstream sinks.**

**An exploratory query takes locks and competes for the one resource everyone needs.** You have a hunch that a particular row is in a bad state, so you run a `SELECT` to check — or worse, a `SELECT ... FOR UPDATE`, or a schema-introspection query, or a `COUNT(*)` on a big table. On the production primary, that query competes for buffer pool, for I/O, and possibly for locks with the live traffic. A long-running read can hold a snapshot that bloats the database's version store; a careless write or DDL can take a table lock that stalls every transaction touching it. The mechanism: **the production database is a shared, contended resource, and your diagnostic query is just more contention on the exact thing the incident is already about.**

**Redeploying with a debug build changes the very behavior you are studying.** The instinct when a laptop bug is stubborn is to rebuild with more instrumentation. In prod, a rebuild is a deploy, and a deploy is the highest-blast-radius action there is: it rolls new code across the fleet, restarts processes, clears caches, and — critically — *changes timing*. A build with optimizations turned off and logging turned up runs slower and schedules differently, which means a race or a latency bug may vanish (a heisenbug, which [its own post](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look) covers) or, just as likely, the rollout itself trips a different problem. The mechanism: **a deploy mutates code, state, and timing simultaneously, so it is the worst possible way to observe a running system you do not yet understand.**

Once you see the five mechanisms side by side, the strategy writes itself. Every one of them is a form of *perturbation* — freezing, pausing, amplifying, contending, mutating. The safe alternatives all share one property: they observe the running process **without changing it**. You read state that is already emitted instead of stopping to inspect it; you sample stacks from the outside instead of halting from the inside; you measure in the kernel instead of in the hot path; and when you genuinely need the dangerous tools, you first move the work onto an instance you have made disposable. That is the whole game, and the rest of this post is its tactics.

## 2. Blast radius: the axis that re-sorts your whole toolkit

On a laptop you sort tools by *power* — which one tells you the most, fastest. In production you sort them by **blast radius**, and you accept less power in exchange for less risk, climbing the power ladder only when forced. Blast radius is the combination of three things: does the probe **pause** the process (and for how long), how much **extra load** does it add (CPU, memory, I/O, network, log volume), and how **reversible** is it (can you undo it instantly, or did you just open a one-way door). A tool with zero pause, negligible load, and instant reversibility has essentially zero blast radius and you can use it on a live host without a second thought. A tool that stops the world has the maximum blast radius and you must never point it at an instance serving real traffic.

Here is the toolkit re-sorted along that axis. Notice that the most *powerful* tools — the interactive debugger, the full heap dump — are at the bottom, because power and blast radius are correlated: the more completely a tool lets you inspect a process, the more completely it has to stop or strain it.

![Comparison matrix ranking six production probes from read-only telemetry through py-spy and bpftrace up to gdb attach, scored by whether they pause the process, their blast radius, and when to reach for each one](/imgs/blogs/debugging-in-production-without-making-it-worse-3.png)

| Probe | Pauses process? | Extra load | Reversible? | What it gives you |
| --- | --- | --- | --- | --- |
| Dashboards, metrics, traces, logs you already have | No | Zero (already emitted) | N/A (read-only) | What and when; where time went |
| `ss`, `lsof`, `/proc/<pid>` reads, `ps`, `top` | No | Negligible | N/A (read-only) | Sockets, FDs, RSS, CPU, threads |
| Thread dump (`jstack`, `py-spy dump`) | Milliseconds or none | Tiny | N/A | What every thread is doing now |
| Sampling profiler (`py-spy top`, `perf record`, async-profiler) | No | ~1% CPU | Stop anytime | Where CPU time is spent |
| eBPF / `bpftrace` one-liner | No | Sub-1%, in-kernel | Detach anytime | Syscalls, latency, lock waits, live |
| Dynamic log-level bump | No (config change) | Multiplies log volume | Yes, flip it back | More detail on a hot path |
| Feature-flagged debug path / 0.1% sampling | No | Bounded by sample rate | Yes, toggle flag | Detail on a controlled fraction |
| Core dump on a *drained* node | Process ends | High, but offline | Replace the node | Full post-mortem state, offline |
| Heap dump | Seconds | 2x memory transiently | N/A | Full object graph (do on drained node) |
| `gdb`/`lldb` attach to live process | **Full stop** | Whole host | Detach (damage done) | Everything — never on a live node |

The discipline is to start at the top of that table and stop the instant you have your answer. Most production incidents — and this is the part engineers under-believe until they have lived it — are solved on the first two rows, by *reading telemetry that already exists*. You do not need to probe a live process to learn that a deploy fifteen minutes ago tracks exactly with the latency spike; the deploy marker and the p99 line on the same dashboard told you. You do not need a heap dump to suspect a connection leak; `ss -s` showing 9,800 sockets in `CLOSE-WAIT` told you. The skill of prod debugging is, more than anything, the discipline to extract the maximum from the zero-blast-radius rungs before you reach for anything that touches the process.

[Observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) is the sibling post on getting everything you can out of those top rungs — metrics, traces, and logs as a debugging toolkit rather than a monitoring dashboard. This post is the complement: what to do when the already-emitted telemetry is *not enough*, and you have to actively probe a live process. The thesis is that even then, you have a long ladder of safe options before you ever reach the dangerous bottom.

## 3. Rung one: read-only first, the probes that cannot hurt anything

The lowest rung is everything you can read without changing a single bit of the process's behavior. These tools open files the kernel already maintains, or read counters the process already publishes. They have, to a very good approximation, zero blast radius, and you should exhaust them before anything else.

Start with what is already on the screen: the dashboards, the RED metrics (rate, errors, duration) for request-shaped problems and the USE metrics (utilization, saturation, errors) for resource-shaped ones, the distributed traces, the existing logs. Cross-reference the symptom against the deploy timeline. An enormous fraction of incidents are a recent change, and diffing a metric across a deploy marker is just [bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) applied to a running system — the bad version is on one side of the marker, the good version on the other.

When the dashboards are not specific enough, drop to the host and read the kernel's own bookkeeping. These commands inspect a live process without touching it:

```bash
# Sockets: are we leaking connections? How many in CLOSE-WAIT or TIME-WAIT?
ss -tanp | awk '{print $1}' | sort | uniq -c | sort -rn
ss -s                                  # one-line summary of socket states

# File descriptors: are we leaking FDs toward the ulimit?
ls -1 /proc/$(pgrep -f payments)/fd | wc -l     # current open FDs
cat /proc/$(pgrep -f payments)/limits | grep 'open files'   # the ceiling
lsof -p $(pgrep -f payments) | wc -l            # same, with detail

# Memory & threads: read RSS and thread count from /proc, no pause
grep VmRSS /proc/$(pgrep -f payments)/status    # resident set size
ls -1 /proc/$(pgrep -f payments)/task | wc -l    # live thread count
cat /proc/$(pgrep -f payments)/status | grep -E 'Threads|VmRSS|VmSwap'

# CPU right now, per-thread, without attaching anything
top -H -p $(pgrep -f payments)          # -H shows threads
```

Every one of those is a read. `ss` reads the kernel's socket tables. `/proc/<pid>/fd` is the kernel's view of the process's open file descriptors. `/proc/<pid>/status` is a text file the kernel synthesizes on read. `top -H` samples `/proc` periodically. None of them stop, pause, or load the target process in any way you can measure. This is the rung where a [file-descriptor or socket leak](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) becomes visible — a steadily climbing FD count toward the `ulimit`, or a pile of sockets stuck in `CLOSE-WAIT` (the remote closed, your app never did) — entirely from the outside, with zero risk.

The one nuance on this rung is the thread dump versus the heap dump, because they sit on opposite ends of the cost spectrum despite both being "dumps." A **thread dump** captures what every thread is doing *right now* — its current stack — and is cheap: for the JVM, `jstack <pid>` triggers a brief safepoint measured in milliseconds; for Python, as we will see, `py-spy dump` does not pause the process at all. A **heap dump**, by contrast, serializes the entire object graph, needs a stop-the-world pause that can run for seconds on a large heap, and transiently doubles memory. So a thread dump belongs near the top of the ladder (reach for it freely; it tells you about deadlocks, stuck threads, and where the work is) and a heap dump belongs near the bottom (drained node only). Conflating them — "let me just grab a dump" — is how a cheap question becomes an expensive accident.

#### Worked example: a connection leak found without touching the process

A service is slowly dying: every hour or so it stops accepting new connections and has to be restarted, and the latency graph shows a sawtooth that resets at each restart. On a laptop you would attach a debugger and watch the connection pool. In prod, you do not need to. From the host, read-only:

```bash
$ ls -1 /proc/$(pgrep -f orders)/fd | wc -l
9847
$ cat /proc/$(pgrep -f orders)/limits | grep 'open files'
Max open files            10000    10000    files
$ ss -tanp | grep -c CLOSE-WAIT
9102
```

The story is fully told by three reads and zero perturbation. The process is at 9,847 open file descriptors against a ceiling of 10,000 — minutes from exhaustion — and 9,102 of its sockets are in `CLOSE-WAIT`, the state that means *the remote peer closed the connection and your code never called `close()`*. That is a textbook FD leak: a code path that opens a connection to a downstream service and, on some branch (probably an error branch), forgets to release it. You have localized the bug class, the resource, and even the likely shape of the fix (find the un-`close()`d connection, almost certainly an exception path), and you did it by reading three files the kernel already maintains. No debugger, no dump, no risk. The restart-every-hour was the FD ceiling being hit; the sawtooth was the leak refilling after each restart. Now you can go read the code for the one downstream client that does not close on error, confirm it in a unit test, and ship the fix — having spent your entire investigation on the zero-blast-radius rung.

## 4. Rung two: low-overhead sampling, watching without stopping

When the read-only rung tells you *that* something is wrong but not *where in the code* the time or the memory is going, you climb to sampling. The defining property of a sampling tool is that it observes the process from the *outside*, by periodically reading its stacks or its kernel events, **without injecting code, without recompiling, and without pausing**. This is the rung that replaces the single most dangerous laptop reflex — "attach a debugger and see where it's stuck" — with something you can safely run on a live host.

The mechanism that makes this safe is worth understanding, because it is exactly the mechanism the debugger lacks. A sampling profiler like `py-spy` is a *separate process*. It reads the target process's memory through the kernel's `process_vm_readv` (on Linux), walks the interpreter's call-stack data structures from the outside, and assembles a stack trace — all without the target ever knowing it happened and without stopping it. Contrast that with `gdb`, which uses `ptrace` to *become the controller* of the target and must stop it to get a consistent view. Same goal (see the stacks), opposite blast radius (zero pause versus full stop), because one reads from outside and the other halts from inside.

![Before and after comparison showing a gdb attach halting every thread on a live host and triggering a health-check failure versus py-spy reading process stacks without pausing, leaving the process serving traffic with zero user impact](/imgs/blogs/debugging-in-production-without-making-it-worse-2.png)

For Python, `py-spy` is the tool, and these two commands are the heart of safe live profiling:

```bash
# A one-shot snapshot of every thread's current stack, no pause, no code change.
# Reads the running process from the outside via process_vm_readv.
sudo py-spy dump --pid $(pgrep -f payments)

# A live, top-like view of where CPU time is going, sampled ~100x/sec.
# Press 'q' to stop; the target never paused.
sudo py-spy top --pid $(pgrep -f payments)

# Record a flame graph over 30 seconds of live traffic, then walk away.
sudo py-spy record --pid $(pgrep -f payments) --duration 30 --output flame.svg
```

`py-spy dump` is the one to internalize: it gives you, instantly and with no pause, the current stack of every thread in the process — the production equivalent of "what is it doing *right now*." If a thread is wedged on a lock, blocked on a slow downstream call, or spinning in a hot loop, the dump shows it in the stack, and a few dumps a second apart show you whether a thread is *stuck* (same stack every time) or *busy* (different stacks). This single command resolves an enormous fraction of "the service is hung / slow and I don't know why" incidents, and it does so with a blast radius indistinguishable from zero.

Every major runtime has an equivalent. The principle — sample from outside, never pause — is universal; the table maps it across stacks:

| Runtime | Live sampling tool | Pauses? | One-liner |
| --- | --- | --- | --- |
| Python | `py-spy` | No | `py-spy dump --pid <pid>` |
| JVM | async-profiler | No (sampled) | `asprof -d 30 -f flame.html <pid>` |
| Go | built-in `pprof` | No | `go tool pprof http://host/debug/pprof/profile` |
| Native (C/C++/Rust) | `perf record` | No (sampled) | `perf record -F 99 -p <pid> -g -- sleep 30` |
| Any (kernel + user) | `bpftrace` / eBPF | No (in-kernel) | see the next section |

The `perf` invocation deserves a note on the overhead knob: `-F 99` sets the sampling frequency to 99 Hz, which is the standard low-overhead choice (99 rather than 100 to avoid lockstep with periodic timers). Sampling at 99 times a second adds roughly one percent of CPU overhead — negligible — and it is *adjustable*: if even that is too much during a delicate incident, drop to `-F 49`. This is the whole philosophy of the sampling rung in one flag: you can dial the perturbation down to whatever the system can spare, because sampling cost is a frequency you control, not a fixed pause you cannot avoid.

A close cousin worth knowing about is **continuous profiling** — tools like Pyroscope, Parca, or the cloud providers' always-on profilers that run a low-frequency sampler in production *all the time* and store the flame graphs. The payoff is that when an incident hits, you do not have to start profiling reactively; you already have the flame graph from five minutes before the spike *and* from during it, and you diff them. It is the sampling rung made proactive: the cheapest probe of all is the one that was already running, the same way the cheapest log line is the one you already wrote before the bug struck.

## 5. The modern prod-safe tracer: eBPF and bpftrace

The sharpest tool on the low-overhead rung deserves its own section, because it changes what is possible to ask of a live system: **eBPF**, usually wielded through `bpftrace`. eBPF lets you run small, verified programs *inside the kernel*, attached to events — a syscall entry, a function return, a network packet, a user-space function in your process. Because the probe runs in the kernel right where the event happens, it does not pause your process, does not require a redeploy, and adds overhead measured in fractions of a percent. It is the closest thing production has to "set a breakpoint that does not stop anything."

The mechanism that makes eBPF safe is genuinely clever and worth a sentence, because it is *why* you can trust it on the payments box. Before the kernel will run your eBPF program, an in-kernel **verifier** statically proves the program will terminate (no unbounded loops), will not read out-of-bounds memory, and will not crash the kernel. A program that does not pass the verifier simply does not load. So unlike a kernel module — which can panic the box — an eBPF probe is *provably bounded* before it ever runs. That is the property that lets you point it at a production process: the worst it can do is observe.

What this buys you in an incident is the ability to answer questions you could previously only get from a debugger, without the debugger's pause. A few `bpftrace` one-liners that are safe to run on a live host:

```bash
# Distribution of how long every read() syscall takes, system-wide, live.
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_read { @start[tid] = nsecs; }
  tracepoint:syscalls:sys_exit_read /@start[tid]/ {
    @ns = hist(nsecs - @start[tid]); delete(@start[tid]); }'

# Count which files our process is opening, live, to catch a hot path.
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat
  /pid == '$(pgrep -f payments)'/ { @[str(args->filename)] = count(); }'

# Histogram of time spent inside a specific user-space function (a suspected
# hot lock acquire), attached by symbol, no recompile, no pause.
sudo bpftrace -e 'uprobe:/app/payments:acquire_lock { @t[tid] = nsecs; }
  uretprobe:/app/payments:acquire_lock /@t[tid]/ {
    @lock_ns = hist(nsecs - @t[tid]); delete(@t[tid]); }'
```

The last one is the powerful pattern: a `uprobe` (user-space probe) attaches to a function *by symbol* in your already-running binary, and the matching `uretprobe` fires on return, so the difference is the function's wall-clock duration — emitted as a histogram, with no code change and no pause. You can ask "how long is `acquire_lock` actually taking in production, right now, under real load?" and get a real answer in seconds, on the live process, without redeploying a thing. For a hot-lock latency bug, that is the difference between a two-minute investigation and a two-day one.

The honest trade-offs, because no tool is free: eBPF needs a recent-enough kernel and root (or specific capabilities), `uprobe` symbol resolution needs symbols in the binary (strip them and you must attach by address), and very high-frequency probes on extremely hot paths can add measurable overhead — a `uprobe` on a function called ten million times a second is not free, so you scope to the specific function you suspect rather than tracing everything. But within those limits, eBPF is the modern answer to "I need to see inside a live production process and I cannot stop it." Brendan Gregg's BPF work is the canonical reference, and it is worth your time.

## 6. Rung three: controlled, reversible probes

Sometimes reading and sampling genuinely cannot answer the question — you need *more detail from the application itself* than it normally emits. This is the rung where you change the application's behavior, but you do it in ways that are **bounded and reversible**: a small, controlled increase in what the system tells you, with an instant undo. The governing principle here is reversibility. Any change you make to a live system during an incident must be one you can take back in a single action, because the whole point of "do no harm" is that if your probe makes things worse, you can immediately make them un-worse.

**Dynamic log-level bump without a redeploy.** The naive way to get more logs is to redeploy with a higher log level — which, as we established, is the highest-blast-radius action there is. The disciplined way is to change the log level on a *running* process without restarting it, and every mature logging setup supports this. A Spring Boot service exposes it over an admin endpoint; many services watch a config file or a feature-flag value; a Python service can wire a signal handler to flip levels. The point is that you bump the level, *get your detail*, and flip it back, all without a deploy:

```bash
# Spring Boot Actuator: raise one logger to DEBUG on the live process, no restart.
curl -X POST http://localhost:8080/actuator/loggers/com.acme.payments \
  -H 'Content-Type: application/json' \
  -d '{"configuredLevel": "DEBUG"}'

# ... read the detailed logs you needed ...

# Then put it back. This is the reversibility that makes it safe.
curl -X POST http://localhost:8080/actuator/loggers/com.acme.payments \
  -H 'Content-Type: application/json' -d '{"configuredLevel": "INFO"}'
```

Two refinements turn this from "still a bit risky" into "safe": scope the bump to **one logger** (the package you suspect, `com.acme.payments`, not the root logger — you do not need the framework's debug spew) and ideally to **one instance** (bump the level on a single host so even if the volume is higher than you expected, only one host is affected). Narrow and reversible: that is the recipe.

**A feature-flagged debug path.** A step deeper: ship, ahead of time, a debug code path that is dark by default and lights up only behind a flag — extra validation, a detailed trace of a specific decision, a comparison of two code paths' outputs. Because it is behind a flag, you can enable it for a *fraction* of traffic, or for *one tenant*, or for *internal users only*, and turn it off instantly if it misbehaves. This is dynamic instrumentation done the disciplined way: the code is already deployed and reviewed, so lighting it up is a config change (reversible) rather than a deploy (not), and the blast radius is bounded by whatever fraction you target.

**Sample 0.1% of requests for heavy instrumentation.** When you need expensive detail — full request/response capture, a per-request profile, a deep trace — the move is to apply it to a tiny, random fraction of traffic. Capturing everything for every request would blow your storage and your latency budget; capturing it for one request in a thousand is statistically plenty to find a pattern and costs almost nothing. This is the same logic as tail-based sampling in distributed tracing: you do not need every needle, you need *enough* needles to characterize the haystack. The 0.1% is a dial — turn it up if the bug is rare, down if the overhead bites.

**Dynamic logpoints.** Some platforms (production debuggers and live-debugging tools) let you add a "logpoint" — a log line injected at a specific line of code at runtime, without a redeploy, that captures a few named variables when execution passes through. Used carefully (one logpoint, on a cold-ish path, with a rate limit), this gives you the "I wish I had logged this variable" detail without the deploy loop. Used carelessly (a logpoint on a hot line with no rate limit), it is just the verbose-log blast radius wearing a disguise. The discipline is the same as everywhere on this rung: narrow scope, bounded volume, instant off-switch.

The unifying property of rung three is that everything here has a **flip-it-back**. A redeploy does not — once the new build is rolling out, undoing it is *another* deploy with its own risk. That is precisely why a reversible log bump beats a debug-build redeploy, and why this entire rung sits *above* "redeploy with instrumentation" on the safety ladder despite both being "add detail to the app."

## 7. Rung four: capturing state for offline analysis

The bottom rung is for when you truly need the dangerous, high-power tools — a full heap dump, a core dump, an attached interactive debugger, an expensive query against production-shaped data. You do not give these up; you *relocate the work* onto an instance you have first made safe to break, so the high blast radius lands on something disposable instead of on live traffic. This is the rung's whole insight: **you can use the most invasive tool in your kit if you first move the patient out of the operating theater.**

![Stack diagram of the four-rung perturbation ladder rising from read-only telemetry through low-overhead sampling and controlled reversible probes to capturing full state offline on a drained node](/imgs/blogs/debugging-in-production-without-making-it-worse-4.png)

**Take a node out of the load balancer and debug it freely.** This is the single most important technique on this rung, and it is the one that unlocks every other dangerous tool. If you have N replicas behind a load balancer, you can mark one as **draining** — it finishes its in-flight requests, the load balancer stops sending it new ones, and the other N−1 replicas absorb the traffic. Now that one instance is serving *nobody*, and you can do anything you want to it: attach `gdb` and stop the world (there is no world to stop), take a heap dump and pause for ten seconds (no requests to delay), run `strace -f` on every syscall, even crash it on purpose. The blast radius collapses to zero because the perturbation no longer touches live traffic.

![Graph showing a load balancer with N replicas where the failing replica is selected and marked draining, in-flight requests finish, N minus one replicas keep serving while the drained node is debugged offline with gdb or a heap dump, then re-added or replaced](/imgs/blogs/debugging-in-production-without-making-it-worse-6.png)

```bash
# AWS: deregister one target so the ALB drains it gracefully.
aws elbv2 deregister-targets --target-group-arn $TG_ARN \
  --targets Id=i-0abc123,Port=8080

# Kubernetes: cordon-equivalent for a single pod — remove it from the Service
# by deleting the label the Service selector matches, keeping the pod running.
kubectl label pod payments-7d9f-x4k2 app- --overwrite
# The pod keeps running and is debuggable; the Service no longer routes to it.

# HAProxy via the runtime API: set one server to maintenance (drain) state.
echo "set server be_payments/srv3 state drain" | socat stdio /var/run/haproxy.sock

# Now the instance serves no traffic. Do the dangerous thing safely:
sudo gdb -p $(pgrep -f payments)          # stop-the-world is fine; nobody's home
jmap -dump:live,format=b,file=/tmp/heap.hprof $(pgrep -f payments)   # pause is fine
```

Two rules make draining safe rather than dangerous in itself. First, **drain the instance that is already failing, not a healthy one.** If one host is the sick one (high latency, climbing memory, the errors are coming from it), that is the one whose state you want to capture, *and* pulling the already-sick host out actually relieves the system — you are removing a bad server, not a good one. Second, **never drain the last healthy replica.** If you are down to two healthy hosts and you pull one, the survivor takes all the traffic and may tip over; now you have an outage caused by your own debugging. Capacity-check before you drain: you need enough remaining replicas to carry the load comfortably.

**A core dump on one canary instead of the fleet.** When a process is crashing intermittently in prod and you cannot catch it in a debugger, configure *one* instance to write a core dump on crash (`ulimit -c unlimited`, a sane `core_pattern`, or `coredumpctl` on systemd). The crash is captured to disk on that one host, the dump is copied off, and you do the full post-mortem analysis — open the core in `gdb`, walk the stack at the moment of death, inspect every variable — entirely offline, with no impact on the other instances. Reading that core dump is its own deep skill (the planned sibling post **reading a core dump and post-mortem analysis** covers it end to end: symbolization, walking frames, reconstructing the crash). The production-safety point is just that you capture the dump on *one* sacrificial host, not fleet-wide.

**Mirror or shadow traffic to a debug instance.** A more advanced move: configure your proxy or service mesh to *duplicate* a copy of live production traffic and send the shadow copy to a separate debug instance, while the real responses still come from the production path. The debug instance sees real production requests — the exact inputs that trigger the bug — but its responses are discarded, so you can run it with a debugger attached, with verbose logging, even with sanitizers compiled in, and it cannot affect a single user. This is how you get "reproduce it with real prod inputs" without any of the prod risk: the bug-triggering request hits your fully-instrumented copy, you catch it there, and production never knew.

**Reproduce in staging with prod-shaped data.** The safest capture of all is the one that does not touch prod: take the *shape* of production — a representative slice of data (anonymized), realistic traffic patterns, the same configuration — into a staging environment where you can be as invasive as you like. The bug that needs a debugger, a TSan build, an `rr` record-replay session — all of that is free in staging. The catch, and it is a real one, is that some prod bugs depend on scale, on real concurrency, on a specific host's state, or on data you cannot easily replicate; for those, the drained-node and shadow-traffic techniques are what you fall back to. ([Reproduce it first](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) is the sibling on the art of reconstructing a bug's conditions.)

## 8. The rules of engagement: how not to make it worse

Tactics matter, but the thing that actually keeps you from causing an outage is a short list of *rules* you follow regardless of which tool you reach for. These are the do-no-harm rules, and they are not bureaucracy — each one exists to block a specific way a well-intentioned probe cascades into a real incident.

![Matrix mapping six production-debugging rules to what each one prevents and the failure mode that occurs if it is ignored, from confounded results to a self-inflicted outage from pulling the last healthy replica](/imgs/blogs/debugging-in-production-without-making-it-worse-8.png)

**Change one thing at a time.** If you bump the log level *and* restart a pod *and* flush a cache in the same minute, and the symptom changes, you have no idea which action did it — and if it got *worse*, you do not know what to undo. The scientific method does not stop applying because you are in an incident; if anything it matters more. One change, observe the effect, then the next change. This is the same falsifiability discipline the [hypothesize-and-falsify](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) post is built on, applied under fire.

**Prefer reversible over irreversible.** Given two probes that would answer the question, take the one you can undo. A log bump (flip it back) beats a redeploy (undo is another deploy). A drain (re-add the node) beats a kill (it is gone). A flag (toggle off) beats a config file edit you have to remember to revert. Reversibility is the property that lets a mistake stay a non-event.

**Have a rollback before you act.** Before you change anything, know exactly how you will undo it, and have that command ready. Before you bump the log level, have the "set it back to INFO" command in your buffer. Before you drain the node, know how to re-add it. The time to figure out the undo is *before* the change, not after it has made things worse and the clock is running.

**Never debug on the only healthy replica.** This is the cardinal sin and it has a body count in real postmortems. If the fleet is degraded and you reach for the one host that is still healthy — to attach a debugger, to drain it, to experiment — and it falls over, you have just taken the whole service down with your own hands. Touch the *sick* hosts. Leave the healthy ones serving.

**Do the dangerous probe on the instance that is already failing.** Closely related, and it is a gift hidden in plain sight: the host that is already misbehaving is *both* the one whose state you most want to capture *and* the one that is safest to perturb, because it is already not carrying its share well. Drain the leaking host, dump the heap on the crashing host, `strace` the wedged host. You get the best data and you do the least additional harm, in one move.

**Announce it, and coordinate with on-call and incident command.** During an incident there may be several engineers acting at once. If you silently bump a log level while someone else is mid-rollout and a third person is restarting pods, your signals tangle and someone's change masks or amplifies another's. Say what you are about to do in the incident channel *before* you do it ("I'm going to drain host i-0abc and take a heap dump, ~2 min"), so there is one coherent picture of what is being changed. In a formal incident this is incident command's job — there is a single person coordinating actions so two responders do not collide. This forward-references the SRE discipline of [incident response](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) and the broader idea that reliability is something you engineer deliberately, the way the planned **reliability-is-a-feature, the SRE mindset** sibling argues; treating production as a thing you operate with discipline rather than poke at is the difference between a controlled investigation and a second outage. The error-budget framing in [reliability, SLOs, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) is the policy layer on top of these tactics.

These six rules compose into one sentence you can hold in your head at 3am: **make the smallest reversible change to the sickest non-essential host, announce it, and know your undo.** Everything else in this post is in service of being able to follow that sentence under pressure.

## 9. Worked example: a memory bloat investigated live, zero impact

Let me run a full investigation end to end, because the rungs and rules only become real when you watch them applied to a single bug. This is a memory bloat — the kind of thing that on a laptop you would attack with a heap dump in the first thirty seconds, and the entire skill in prod is *not* doing that until you have made it safe.

![Timeline of a memory bloat investigation showing the alert at plus four megabytes per minute, draining one host, a py-spy dump confirming live code, a tracemalloc snapshot diff, identifying an unbounded cache, and a fix that flattens resident memory with zero customer impact](/imgs/blogs/debugging-in-production-without-making-it-worse-5.png)

#### Worked example: the cache with no ceiling

**The symptom.** The alert fires at 09:14: resident memory on the `recommendations` service is climbing at roughly +4 MB/minute across all twelve hosts, and the oldest hosts are at 85% of their memory limit. Left alone, the fleet OOM-kills in about an hour. This is a classic [memory leak / bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) shape: a steady, linear climb that resets on restart.

**Rung one — read-only.** First, confirm the climb from the outside, no perturbation:

```bash
$ for h in rec-01 rec-05 rec-09; do
    ssh $h "grep VmRSS /proc/\$(pgrep -f recommendations)/status"; done
rec-01: VmRSS:  3814220 kB
rec-05: VmRSS:  3801104 kB
rec-09: VmRSS:  3798880 kB
```

All three hosts are near 3.8 GB and climbing in lockstep — so this is not one bad host, it is something in the code or the load that affects every instance equally. That rules out a single corrupted host and points at a leak in a common code path. A read-only check, zero risk, and it has already narrowed the hypothesis.

**Rung four — drain one host, because a heap dump is expensive.** I want to know *which objects* are growing, and on a managed runtime that means a heap snapshot — which is exactly the expensive, pausing operation we never do on a live host. So I drain one host first. The fleet has twelve hosts and plenty of headroom for eleven to carry the load, so this is safe (rule: never the last replica; check capacity first). I announce it in the incident channel — "draining rec-09 to profile memory, ~3 min" — and pull it:

```bash
$ kubectl label pod recommendations-rec-09 app- --overwrite
pod/recommendations-rec-09 labeled        # now serving no traffic
```

**Rung two on the drained host — py-spy first, cheapest thing that helps.** Before the expensive snapshot, a free check: is the growth in live application code or in something else? A `py-spy dump` shows the threads are deep in the recommendation-scoring path, not stuck in GC or in a native library — so the growth is application objects, not a runtime artifact. That took one second and zero pause and it confirmed where to look.

**The state capture — tracemalloc diff on the drained host.** Now, on the host that is serving nobody, I can afford the expensive probe. I use Python's `tracemalloc` to snapshot allocations, wait, snapshot again, and diff — which tells me exactly which lines allocated the memory that is still live and growing:

```python
import tracemalloc, time, signal

tracemalloc.start(25)          # keep 25 frames of traceback per allocation
snap1 = tracemalloc.take_snapshot()
time.sleep(300)                # let it run 5 minutes on the drained host
snap2 = tracemalloc.take_snapshot()

# Which lines allocated the memory that grew between the two snapshots?
for stat in snap2.compare_to(snap1, 'lineno')[:10]:
    print(stat)
```

The top line of the diff is unambiguous:

```bash
recommendations/scoring.py:88: size=612 MiB (+184 MiB), count=2.4M (+720k)
```

A single line allocated 184 MB in five minutes, with 720,000 new objects that never got freed. I open `scoring.py:88` and it is a memoization cache — `@lru_cache(maxsize=None)` — keyed on the full request, including a per-request timestamp. Because the timestamp makes every key unique, the cache never gets a hit and never evicts; it just grows forever. That is the leak: **an unbounded cache with an accidentally-unique key.** The fix is two characters of intent — a bounded `maxsize` and a key that excludes the timestamp:

```python
# Before: unbounded, keyed on a per-request timestamp => never evicts, grows forever.
@lru_cache(maxsize=None)
def score(user_id, item_id, request_ts): ...

# After: bounded cache, keyed only on the stable fields that actually repeat.
@lru_cache(maxsize=100_000)
def score(user_id, item_id): ...
```

**The proof, measured honestly.** Ship the fix on a canary first, watch its `VmRSS` for an hour: the +4 MB/min slope goes flat — it climbs to the working-set size and then holds steady, exactly what a bounded cache should do. Roll it to the fleet, re-add `rec-09` to rotation, and the fleet-wide memory graph flattens. The honest way to state the result: RSS growth went from **+4 MB/min to ~0 MB/min** (flat within measurement noise) over a one-hour observation window on the canary, and the OOM-kill-in-an-hour trajectory was eliminated. **Customer impact during the entire investigation: zero** — every probe ran on a drained host, the eleven live hosts never saw a debugger, never saw a pause, never saw a heap dump. The bug that on a laptop you would have heap-dumped in thirty seconds was instead resolved by a disciplined climb up the rungs, and nobody using the product felt a thing.

## 10. Worked example: a latency spike traced with bpftrace, no redeploy

The second investigation is a latency spike, and it shows off the eBPF rung — answering a question that classically required a debugger or a debug build, on the live process, with no pause and no deploy.

#### Worked example: the hot lock that only bit under load

**The symptom.** At peak traffic, the `inventory` service's p99 latency jumps from 80 ms to 1,200 ms, but only at peak — at 3am it is fine, and it never reproduces in staging because staging never sees peak concurrency. CPU is not saturated, the database is healthy, and the obvious suspects are all clean. This is the worst kind of prod bug: it only happens under real load, on real hosts, and you cannot make it happen anywhere you can attach a debugger.

![Before and after comparison contrasting shipping a debug build that changes timing and risks an outage against attaching a bpftrace uprobe to the live process to histogram lock wait times and find the hot mutex, dropping p99 from twelve hundred to eighty milliseconds](/imgs/blogs/debugging-in-production-without-making-it-worse-7.png)

**The wrong move.** The laptop instinct is to rebuild with timing logs around every suspicious section and deploy it. But a debug build changes timing — and this is a *concurrency* latency bug, so changing timing may make it vanish (and waste a deploy) or, worse, the deploy itself disturbs the fleet at peak. We do not redeploy to debug a timing bug. We trace it where it lives.

**Rung two — sample first.** A `perf record -F 99` flame graph over thirty seconds at peak shows a fat stack frame in a function called `reserve_stock`, and within it a lot of time *not* on CPU — the frame is wide but the on-CPU samples are sparse, which is the signature of a thread *waiting*, typically on a lock, rather than computing. So the hypothesis sharpens: threads are blocking on a lock inside `reserve_stock`, and only at peak concurrency does the contention get bad enough to blow the p99.

**Rung two, sharper — bpftrace the lock directly.** A flame graph suggests a lock; `bpftrace` *proves* it by measuring the lock-acquire time on the live process, with a `uprobe`/`uretprobe` pair, no redeploy, no pause:

```bash
# Histogram of how long reserve_stock's lock acquire actually takes, live, at peak.
sudo bpftrace -e '
  uprobe:/app/inventory:lock_reserve   { @start[tid] = nsecs; }
  uretprobe:/app/inventory:lock_reserve /@start[tid]/ {
    @acquire_us = hist((nsecs - @start[tid]) / 1000);
    delete(@start[tid]);
  }'
```

Run that for sixty seconds at peak and the histogram is damning:

```bash
@acquire_us:
[0, 1)            812340 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[1, 2)             40221 |@@                                      |
[512, 1024)         9102 |                                        |
[1024, 2048)       21847 |@                                       |
[2048, 4096)       14903 |                                        |
```

Most lock acquisitions are instant (under a microsecond), but a long tail of them take **1–4 milliseconds** — and at peak, with thousands of requests a second all funneling through this one lock, those millisecond waits stack up into the 1,200 ms p99. The root cause is a **coarse lock**: `lock_reserve` holds a single global mutex around the *entire* stock-reservation operation, including a slow-ish bit of work that does not actually need the lock, so under high concurrency threads queue behind it. The fix is to narrow the critical section — hold the lock only around the actual shared-state mutation, do the slow work outside it — or to shard the lock by SKU so different products do not contend. This is a [lock-contention / latency](/blog/software-development/debugging/deadlocks-livelocks-and-starvation) problem, diagnosed without ever stopping the process.

**The proof.** Ship the narrowed critical section, and at the next peak the `bpftrace` histogram's millisecond tail is gone — acquisitions stay under a microsecond even under full concurrency — and p99 drops from **1,200 ms back to 80 ms**, its off-peak baseline. State it honestly: the before and after are *measured* at peak on real traffic (the histogram and the p99 panel), not extrapolated, and the entire diagnosis happened on the live process with **no redeploy, no pause, and no customer-visible perturbation** — the eBPF probe added well under one percent overhead and was detached the moment we had the histogram. A timing bug that only existed under real production load was caught with the one class of tool that can watch a live system without changing its timing.

## 11. War story: famous ways the probe became the outage

It is worth grounding the do-no-harm principle in real incidents, because the failure mode — the investigation causing or worsening the outage — is one of the most documented patterns in postmortem history.

**The retry storm / thundering herd.** A canonical self-inflicted cascade: a service degrades slightly, clients retry, the retries add load, the added load degrades it more, which triggers more retries, and the system collapses under a load it created. The debugging-relevant version is when the *responder* makes it worse — for example, draining or restarting instances during the degradation reduces capacity at the exact moment retries are spiking, and the smaller fleet tips over. The lesson is the capacity rule from §8: before you pull a host, confirm the remaining hosts can carry the load *including* any retry amplification. A degradation is a delicate state; removing capacity from it is how a blip becomes an outage. (The [message-queue series](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) covers retry/backoff and idempotency in depth; the debugging takeaway is just to respect the fragility of a system already under stress.)

**The GC-pause outage made worse by a heap dump.** A widely-seen pattern on the JVM: a service is suffering long garbage-collection pauses (the symptom is latency spikes correlated with GC). A responder, reasonably wanting to know what is on the heap, runs `jmap -dump` on a *live* instance. The heap dump itself requires a stop-the-world pause and transiently inflates memory — on a box already memory-pressured and GC-thrashing, that extra pause and pressure can be the straw that triggers an OOM kill or a multi-second freeze that fails health checks. The investigation tool caused the very outage it was investigating. The do-no-harm version: drain the instance first (§7), *then* dump the heap; or use a sampling allocation profiler (async-profiler's allocation mode) that does not require the stop-the-world dump at all.

**Knight Capital, 2012 — the deploy as the highest-blast-radius action.** Knight Capital lost about \$440 million in 45 minutes when a deployment left old, repurposed code active on one of eight servers, and that one server began sending erroneous orders at market open. It is not a debugging story per se, but it is the sharpest possible lesson about the rung-four mechanism from §1: a deploy mutates code, state, and timing across a fleet, and a partial or inconsistent rollout is catastrophic. The debugging-relevant takeaway is exactly why we put "redeploy with a debug build" near the bottom of the safety ladder: shipping new code to a production fleet to investigate a problem is the *most* invasive thing you can do, and partial rollouts (some hosts new, some old) are their own bug class. When you can answer the question with a `py-spy dump` or a `bpftrace` one-liner instead of a deploy, you have removed the entire category of deploy risk from your investigation.

**The verbose-log disk fill.** A less famous but extremely common one, seen in countless internal postmortems: during an incident, a responder flips the log level to `DEBUG` fleet-wide to gather detail, the debug volume fills local disks within minutes, and services that need to write — to logs, to temp files, to the database's write-ahead log on a co-located disk — begin failing. The detail-gathering probe took the system down. The do-no-harm version is the §6 recipe: bump *one* logger on *one* host, with a rollback ready, and watch the disk while you do it.

The thread through all four is the same: in production, the observer is part of the system. There is no neutral act of looking. Every probe consumes some resource — CPU, memory, I/O, a pause, a lock, a deploy — and during an incident the system has the least of those resources to spare. Respecting that is the entire discipline.

## 12. Stress-testing the playbook against the hard cases

A playbook is only worth having if it survives the awkward cases — the bugs that do not present cleanly. Let me push the blast-radius ladder against the situations that actually make production debugging hard, because the answer is rarely "give up and attach a debugger"; it is almost always "there is a safe rung that fits this case too."

**What if it only reproduces under real load?** This is the §10 latency bug. The trap is that staging never hits the concurrency that triggers it, so the instinct is to load-test prod or to deploy timing instrumentation. Neither is safe. The right move is the one we used: trace it *where the load already is* — on the live process, with a sampling profiler or a `bpftrace` `uprobe` that adds sub-1% overhead. Load-dependent bugs are precisely the case the sampling rung was built for, because the load is a resource you cannot manufacture safely but the live system is already generating for free. You do not bring the load to the debugger; you bring a zero-pause probe to the load.

**What if it only happens on one host?** A single misbehaving instance — high latency, climbing memory, a pile of errors all sourced from one host — is the *easiest* case to debug safely, because it is the case where draining is unambiguously correct (rule §8: probe the failing instance) and unambiguously safe (you are removing a bad server, not a good one). Pull that one host from rotation and you have a perfect specimen: the exact state that produces the bug, now serving nobody, fully available for `gdb`, a heap dump, an `strace -f`, whatever you need. The one-host bug is the gift case. The danger is misdiagnosing *which* host is sick and draining a healthy one, so confirm the symptom is actually sourced from the host you are about to pull before you pull it — read the per-host metrics first (rung one), then drain (rung four).

**What if it only appears after six hours?** A bug that needs a long uptime to manifest — a slow leak, a counter that overflows, a connection pool that fragments, a cache that degrades — defeats any reproduce-on-demand approach, because you cannot wait six hours per attempt. The answer is to capture the *trajectory*, not a single snapshot. Continuous profiling and time-series metrics record the slow drift so that when the symptom finally appears, you have the whole six-hour history to diff against the first hour. For a slow leak specifically, the §9 technique works: drain a host that has *already* been up six hours (so it already exhibits the bloat), and take the `tracemalloc` diff there — you do not have to wait, because you debug an instance that has already done the waiting for you. The general principle: when a bug needs time, find the instance that has already spent the time, and probe *it*.

**What if it only happens when two requests interleave?** A [race condition](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch) in prod is brutal because the interleaving is non-deterministic and adding any probe can change the timing enough to hide it (a [heisenbug](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look)). Here the zero-perturbation rungs earn their keep most of all: a sampling profiler and a `bpftrace` lock-wait histogram do not change timing the way a debugger or a log line does, so they can observe a timing-sensitive bug without scaring it off. The §10 hot-lock investigation is exactly this case — a contention bug that only blows up under concurrent interleaving, caught with the one tool class (eBPF) that watches timing without altering it. When you genuinely need to *step through* the interleaving, that is the case for capturing a core dump or an `rr` record-replay session on a drained or shadow instance, where you can replay the exact interleaving offline as many times as you want.

**What if you cannot attach a debugger at all?** Sometimes the platform forbids it — a managed runtime, a locked-down container, no `ptrace` capability, a serverless function with no host access. This is where the ladder's top rungs shine, because they never needed `ptrace` in the first place. Read-only telemetry, the request traces you already emit, the metrics, and — where the platform allows it — a sampling profiler bundled into the application itself (an in-process continuous profiler, or an admin endpoint that dumps thread stacks on request) all work without ever attaching an external debugger. The whole reason this post leads with "observe what is already emitted" is that it is the *only* technique that works everywhere, including the environments where you have no host and no debugger and no shell.

#### Worked example: the bug that only bit after the cache warmed

**The symptom.** A search service is fine for the first few minutes after every deploy, then p99 degrades over the next twenty minutes and plateaus high. It never reproduces in staging (which gets restarted and never runs long enough) and never on a freshly-deployed prod host (which is, briefly, fast). The "only after N minutes" shape immediately says: do not try to reproduce on a fresh instance — find the instance that has *already aged into the bug*.

**The investigation.** Rung one, read-only: per-host metrics show the degraded hosts are the ones up longest, and their RSS is higher and their GC frequency has climbed — but memory is not the symptom, latency is. Hypothesis: something that grows with uptime is making each request do more work. Rung two on a *long-up* host (no drain needed yet, sampling is safe live): `py-spy top` on a 25-minute-old host shows a surprising amount of time in a cache-eviction routine that barely registered on a fresh host. The story clicks: an in-memory cache with a bad eviction policy degrades to near-linear scan as it fills, so every request gets slower as the cache grows — fast when empty, slow when full. To confirm without guessing, drain one aged host and `bpftrace` the eviction function's duration; the histogram shows it climbing from microseconds (fresh) to milliseconds (aged), perfectly tracking the p99 curve.

**The proof.** Replace the eviction policy (a proper LRU with O(1) eviction instead of the accidental linear scan), and the post-deploy degradation curve flattens: p99 stays at its fresh-deploy baseline of ~70 ms even after an hour of uptime, versus the old plateau around 900 ms. Measured honestly on real traffic over a multi-hour window after the fix, with the entire diagnosis done via live sampling plus one drained-host `bpftrace` confirmation — zero customer impact, and crucially, *zero attempts to reproduce on a fresh instance*, because the bug lived in uptime and we went straight to the aged instance that already had it.

The pattern across all five hard cases is one idea: **the bug already exists somewhere in production — find the instance, the load, or the moment where it already lives, and bring a zero-perturbation probe to it, rather than trying to manufacture the conditions somewhere safe.** Production is, perversely, the best reproduction environment you have; the discipline is observing it without disturbing it.

## 13. How to reach for this (and when not to)

Here is the decisive guidance, because every tool on the ladder has a cost and the skill is knowing when *not* to reach for the powerful ones.

**Reach for read-only telemetry first, always, no exceptions.** Before you touch the process, look at what it has already told you — dashboards, traces, logs, the deploy timeline, and the `/proc`/`ss`/`lsof` reads. A large majority of incidents are diagnosable from this rung alone, and it has zero blast radius. If you find yourself reaching for a debugger before you have read the existing telemetry, stop; you are about to take risk you do not need to take.

**Reach for sampling (`py-spy`, `bpftrace`, `perf`) when you need to know *where* in the code, and you would otherwise want a debugger.** This is the rung that should replace your "attach gdb" reflex on a live host. `py-spy dump` for "what is it doing right now," `py-spy top`/`perf`/async-profiler for "where is CPU going," `bpftrace` for "how long is this syscall/function/lock taking." All zero-pause, all dial-able overhead, all safe on live traffic.

**Reach for a reversible probe (log bump, flag, 0.1% sample) when sampling cannot get application-level detail.** And do it narrow: one logger, one host, one flag, a rollback ready.

**Reach for the dangerous tools (heap dump, core dump, gdb, expensive query) only after you have moved the work onto a drained or shadow instance.** The rule is absolute: never point a stop-the-world tool at a process serving live traffic. Drain first, then do anything you want.

**When NOT to do each:**

- **Don't attach `gdb`/`lldb` to a process serving live traffic.** Ever. It freezes the process; the freeze fails health checks; the failed health check pulls the node; you have an outage. If you need the debugger, drain the node first.
- **Don't take a heap dump on a memory-pressured live host.** It pauses and doubles memory — the two things a pressured host cannot afford. Drain it, or use a sampling allocation profiler.
- **Don't flip the log level to DEBUG fleet-wide.** One logger, one host. The fleet-wide debug flag is a self-inflicted DoS on your disks, your log pipeline, and your bill.
- **Don't run an exploratory query on the production primary.** Run it on a read replica, or if you must, with a tight `LIMIT` and a statement timeout, and never anything that takes a lock.
- **Don't redeploy a debug build to investigate a timing bug.** The deploy changes the timing you are studying and is the highest-blast-radius action there is. Trace it live with `bpftrace` instead.
- **Don't drain the last healthy replica, and don't probe the only working host.** Touch the sick ones; leave the healthy ones serving.
- **Don't debug silently during a multi-responder incident.** Announce every change in the incident channel before you make it, so signals do not tangle.

The meta-rule that subsumes all of these: **prefer the cheapest probe that can answer the question, and prove to yourself it is reversible before you run it.** If you cannot answer "how do I undo this if it makes things worse" in one sentence, you are not ready to run it on production.

## 14. Prevention: make prod debuggable before the incident

The deepest lesson of production debugging is the same as the deepest lesson of [logging](/blog/software-development/debugging/logging-as-a-debugging-instrument) and [observability](/blog/software-development/debugging/observability-for-debugging-prod): the time to make a system debuggable is *before* the incident, not during it. Every safe technique in this post is dramatically more effective if you have prepared for it.

**Ship the safe tools onto the hosts already.** `py-spy`, `bpftrace`, `perf`, and a recent kernel should be present (or one `apt install` away) on production hosts *before* you need them at 3am. An incident is the worst time to discover that the box does not have the profiler. Bake them into the image.

**Run continuous profiling in production.** An always-on, low-frequency sampler (Pyroscope, Parca, the cloud profilers) means that when an incident hits, you already have the flame graph from before *and* during, and you diff them. The cheapest probe is the one that was already running.

**Build the drain switch and rehearse it.** Make "drain one instance" a one-command, well-understood operation that every on-call engineer has done before. If the first time anyone drains a node is during an incident, they will do it wrong. Practice it in a game-day.

**Wire dynamic log levels and feature-flagged debug paths in advance.** A service that can have one logger bumped to DEBUG and flipped back, without a restart, is a service you can investigate safely. A service where the only way to get more logs is a redeploy forces you toward the dangerous rung. Build the safe path before you need it.

**Keep enough capacity headroom to drain.** If your fleet runs so hot that pulling one host tips the rest over, you *cannot* use the single most important safe-debugging technique. Headroom is not just a reliability property; it is a debuggability property — it is what makes the drained-node investigation possible at all.

These are investments, and they pay off the way all good prevention does: the next incident is one you can investigate *calmly*, on a drained host, with the right tools already present, instead of one where the investigation itself becomes the disaster. This is the prevention half of the series' observe → reproduce → hypothesize → bisect → fix → **prevent** loop, applied to the meta-level: you are preventing not just the bug, but the *next investigation's* potential to do harm.

## 15. Key takeaways

- **First, do no harm.** In production the act of looking can be the thing that breaks it; pick the cheapest probe that can still answer the question, never the most powerful one you happen to know.
- **Sort tools by blast radius, not by power.** Pause cost, added load, and reversibility — not "how much does it tell me" — decide what is safe on a live host. Power and blast radius are correlated, which is why the most powerful tools are the most dangerous.
- **A debugger freezes the process, and a freeze is an outage.** `gdb` attached to a live service stops all threads, fails the health check, and pulls the node. Never point a stop-the-world tool at a process serving traffic.
- **Read-only first, always.** Dashboards, traces, `ss`, `lsof`, `/proc` — most incidents are diagnosable from the zero-perturbation rung, and you should exhaust it before touching the process.
- **Sample, don't stop.** `py-spy dump`, `perf -F 99`, and `bpftrace` read a live process from the outside or in-kernel with dial-able sub-1% overhead and no pause — they are the modern replacement for the "attach a debugger" reflex.
- **eBPF is the prod-safe tracer.** A verified in-kernel probe answers "how long is this lock/syscall/function taking, live, right now" with no redeploy and no pause — the question that used to need a debug build.
- **For the dangerous tools, move the work offline.** Drain one instance (the *sick* one, never the last healthy one), then take the heap dump, the core dump, or the gdb session — the high blast radius lands on something disposable.
- **Follow the rules of engagement:** change one thing, prefer reversible, have a rollback ready, never touch the last healthy replica, probe the failing instance, and announce every action in the incident channel.
- **Prepare before the incident.** Ship `py-spy`/`bpftrace` onto the hosts, run continuous profiling, build and rehearse the drain switch, wire dynamic log levels, and keep enough headroom that draining a node is actually safe.

## 16. Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the intro that frames the whole observe → reproduce → hypothesize → bisect → fix → prevent loop this post operates inside.
- [Observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) — the sibling on getting the most out of the read-only rung: metrics, traces, and logs as a debugging toolkit.
- [Hunting memory leaks and bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) and [resource leaks: FDs, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) — the leak-shaped bugs the §3 and §9 examples chase.
- **Reading a core dump and post-mortem analysis** (planned sibling) — the deep skill of opening the core you captured on a drained node and walking the crash offline.
- [Anatomy of an outage: lessons from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) and [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — the SRE incident-response and reliability framing that surrounds these tactics.
- Brendan Gregg, *BPF Performance Tools* and the flame-graph material — the canonical reference for `bpftrace`, `perf`, and low-overhead production tracing.
- The `py-spy` and async-profiler documentation — the sampling profilers that watch a live process without pausing it.
- David Agans, *Debugging: The 9 Indispensable Rules* — "make it fail," "quit thinking and look," and the discipline that underlies doing no harm under pressure.

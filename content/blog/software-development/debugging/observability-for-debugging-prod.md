---
title: "Observability for Debugging Prod: When You Cannot Attach a Debugger"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to debug a bug you cannot reproduce locally by reading metrics, traces, and logs together, pivoting from a p99 spike on a Grafana panel to one exemplar trace to the single log line that explains it."
tags:
  [
    "debugging",
    "software-engineering",
    "observability",
    "distributed-tracing",
    "prometheus",
    "opentelemetry",
    "metrics",
    "production",
    "troubleshooting",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/observability-for-debugging-prod-1.png"
---

At 14:07 on a Tuesday the pager goes off: checkout p99 latency is over budget. You pull up the service on your laptop, run the exact request that's slow in staging, and it returns in 90 milliseconds. You run it again. 88 milliseconds. You add a `print`, run it a third time, still fast. The bug is real — the dashboard says so, customers are feeling it, the on-call channel is filling up — but it does not exist anywhere you can put your hands on it. You cannot attach `gdb` to the payments process. You cannot add a log line and redeploy in the eleven minutes you have before this becomes a status-page incident. The single most important debugging tool you have ever used, the debugger, is useless here, because the bug only happens in a place you are not allowed to stop the world.

This is the defining problem of debugging production: **you can watch the system, but you cannot touch it.** Everything the earlier posts in this series taught you — set a breakpoint, add a watchpoint, bisect the commit, reproduce it until it fails — assumes you can run the failing code under your control. In prod you can't. The request that's slow is one of forty thousand per second, it ran on a host you'll never SSH into, it touched six services, and by the time you've finished reading the alert it has already completed and been forgotten by every process that handled it. You have exactly one source of truth about what happened: the telemetry the system emitted while it was happening. If that telemetry is good, you find the root cause in five minutes. If it's bad, you spend two days guessing.

The thesis of this post is simple and I want you to hold onto it the whole way through: **observability is the debugger for systems you can only watch.** A debugger lets you stop a process and inspect its state. Observability lets you reconstruct, after the fact, the state of a request that already finished, by stitching together the three signals the system emitted as it ran — metrics that tell you *what* is wrong and *when*, traces that tell you *where* the time or the error went, and logs that tell you *why* the code did what it did. Used as a monitoring dashboard, these three signals tell you the house is on fire. Used as a *debugging toolkit* — the way we will use them here — they walk you from the smoke alarm to the exact wire that shorted.

![Diagram showing the prod debugging loop that pivots from a metric spike to an exemplar trace to the log line that explains it, ending at a config push as the root cause](/imgs/blogs/observability-for-debugging-prod-1.png)

By the end of this post you will be able to take a vague prod symptom — "it's slow," "we're throwing 500s," "it's broken for some users" — and run a disciplined investigation: start at the symptom metric, pivot to a representative trace at that exact timestamp, and drill to the log line for that one request. You will know which dashboard to open first (RED for request-shaped problems, USE for resource-shaped ones), how to read a flame graph to find the span that ate the latency, why high-cardinality fields like user id are both gold and a budget hazard, how tail-based sampling catches the 0.1% of requests that are slow, and how diffing a metric across a deploy marker is just `git bisect` applied to a running system. This is the same observe → reproduce → hypothesize → bisect → fix → prevent loop the whole [series](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) is built on — except "reproduce" becomes "find the one captured request that already failed," because in prod you don't get to make it happen again on demand.

## 1. The mechanism: why you cannot debug prod the way you debug a laptop

Before we reach for tools, it's worth being precise about *why* the laptop techniques fail in production, because the reason dictates the whole strategy. It is not that production is mysterious. It's that production violates three assumptions every interactive debugger silently depends on.

**Assumption one: you can stop the process.** A debugger works by halting execution at a breakpoint and letting you inspect memory while time stands still. In prod, time cannot stand still. If you attach `gdb` to a live service and hit a breakpoint, every in-flight request on that process blocks. The thread pool drains, the load balancer's health check times out, the instance gets marked unhealthy and pulled, and now you've turned a latency blip into an outage. The process is shared by thousands of concurrent requests; you cannot freeze it to look at one. This is why the very first rule of prod debugging is *do not attach a debugger to a process serving live traffic* — not because it won't work, but because it works by stopping the one thing you must not stop.

**Assumption two: you can reproduce on demand.** Bisection, the workhorse of this series, needs a test you can run repeatedly that's red on the bad commit and green on the good one. Production bugs frequently have no such test. The bug depends on a specific input you don't have ("the request from tenant 8841 with an empty cart and a stale coupon"), a specific concurrency interleaving ("two checkout requests hit the same inventory row in the same 3 ms window"), a specific machine state ("this one host's connection pool was exhausted because a slow query held connections"), or a specific moment in time ("the cache was cold for the ninety seconds after the deploy"). You can't reconstruct those on your laptop because *you don't know what they are yet* — that's the bug. The whole point of the investigation is to discover the input, the interleaving, the host, the moment.

**Assumption three: you can add instrumentation cheaply.** On a laptop, when you want to know a value, you add a `print`, recompile, and rerun — a thirty-second loop. In prod that loop is a deploy: build, test, canary, roll out, wait for the bug to recur. It can take an hour, and the bug may not recur for a day. So the instrumentation you wish you had must *already be there* before the bug strikes. This is the deepest mental shift of production debugging: you cannot add the print *after* you see the symptom, so you must have emitted enough signal *during* normal operation that the failing request left a trail. Observability is not something you turn on during an incident. It is the trail you laid down in advance so that the incident is debuggable at all. ([Logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) is the sibling post on writing that trail well.)

So the strategy follows directly from the constraints. You cannot stop the process, so you reconstruct from emitted signal. You cannot reproduce on demand, so you find the one already-captured failing request. You cannot add a print, so you mine the structured telemetry that's already there. Three signals carry that telemetry, and each answers a different debugging question.

## 2. The three signals as a debugging toolkit, not a monitoring lecture

Most observability writing treats metrics, logs, and traces as "the three pillars" and stops at definitions. For debugging, definitions are useless; what matters is the *question each one answers* and the *order you ask them in*. Let me reframe the three pillars as a debugger's toolkit.

**Metrics answer WHAT and WHEN.** A metric is a number aggregated over time and dimensions — requests per second, error rate, p99 latency, CPU utilization, queue depth. It is cheap because it's pre-aggregated: a billion requests collapse into a handful of time series. That cheapness is also its limit — a metric tells you *that* p99 latency doubled at 14:03, but it cannot tell you *which* request was slow or *why*, because the individual requests were summed away. In the debugging loop, the metric is your *symptom detector and your clock*. It is where you confirm something is actually wrong (not just a single user complaining) and pin down exactly *when* it started, which is often the single most valuable fact in the whole investigation because it lets you correlate with a deploy, a traffic shift, or a config push.

**Traces answer WHERE.** A trace follows one request as it travels through the system, recording each unit of work — each *span* — with a start time, a duration, a service name, and a parent. Reassembled, the spans of a trace form a tree (often drawn as a waterfall) that shows you exactly which service, which database call, which downstream dependency consumed the time or threw the error. A trace is per-request and expensive relative to a metric, but it is the only signal that can answer *where in the call graph* the latency or failure lives. In the loop, the trace is your *localizer*: the metric told you something is slow at 14:03; the trace tells you it's the pricing-DB call inside the checkout service, not the cache and not the network.

**Logs answer WHY.** A log line is a timestamped record of an event or decision inside the code — an exception with a stack trace, a "took branch B because feature flag X was off," a "tenant_id was null so I fell back to the default catalog." Logs are per-event and the most expensive of the three at high volume, but they carry the richest detail: the actual exception message, the actual variable value, the actual decision the code made. In the loop, logs are your *explainer*: the trace told you *where* (the pricing-DB span), the logs for that span tell you *why* (it threw `TenantNotFound` because the request arrived with a null tenant id after the config push stripped a header).

![Matrix comparing metrics, traces, and logs across what they answer, their granularity, their cost driver, and when to reach for each during a prod investigation](/imgs/blogs/observability-for-debugging-prod-2.png)

The crucial insight — and the reason this is a *toolkit* and not three separate dashboards — is that **no single signal debugs anything alone.** A metric spike with no trace to pivot to is just anxiety: you know something's wrong and you have nowhere to go. A trace with no metric is a needle in a haystack: which of the forty thousand traces per second do you even look at? A log with no trace id is a confession with no context: you found a `TenantNotFound` exception, but was it the one causing the latency spike, or one of the thousand harmless ones you throw every minute? The power is in the *pivots between them*: metric → trace → log, each one cutting the search space by an order of magnitude. The repo's [system-design post on observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) covers how to architect those pivots into your system; here we use them.

| Signal | Debugging question | Granularity | What it cannot tell you |
| --- | --- | --- | --- |
| Metric | What is wrong, and when did it start? | Aggregate per service or resource | Which request, or why |
| Trace | Where did the time or error go? | One request, a tree of spans | The exception text or variable value |
| Log | Why did the code do that? | One event, one line | Whether it is the cause or noise (needs a trace id) |

Notice the last column. Each signal's blind spot is exactly the next signal's strength. That is not a coincidence; it is why the drill-down order is fixed.

## 3. The workflow: symptom metric, exemplar trace, log line for that trace id

Here is the core loop of this entire post, the thing I want you to internalize so deeply that you run it on muscle memory at 3am. Three steps, always in this order.

**Step one: start at the symptom metric.** Do not start in the logs. Logs at prod scale are a firehose — millions of lines a minute — and grepping them without a timestamp and a trace id is how you lose a day. Start at the metric that represents the symptom. If the page says "latency," open the latency panel. Find the exact moment it changed. Pin the timestamp. This is your `t0`, and from here on, every other signal gets filtered to a tight window around `t0`.

**Step two: pivot to an exemplar trace at that timestamp.** A metric is an aggregate, so you cannot debug it directly — you need to get from "p99 doubled" to "here is one actual request that was slow." That bridge is the *exemplar*: a representative trace, captured at the moment of the spike, linked directly from the metric. Modern stacks attach exemplars to histogram buckets — when a request lands in the slow bucket, the metrics system records its trace id alongside the count. So when you see the spike, you click it and land on a real trace that was part of that spike. Without exemplars you do it manually: query your trace store for traces in the `[t0, t0+2m]` window with duration above p99, and pick one.

**Step three: drill to the logs for that trace id.** Now you have one trace. Read its waterfall, find the span that consumed the time or threw the error, and pull the logs *for that exact trace id and span*. Because every log line carries the trace id (you propagated it — this is the correlation id discipline from the logging post), you can filter the firehose down to the dozen lines that this one request emitted. Those lines tell you *why*.

That's it. Symptom metric (when) → exemplar trace (where) → span logs (why). Every prod investigation in this post is an instance of this loop. The reason it works is that it *monotonically narrows the search space*: you go from all-of-production to one-time-window to one-request to one-log-line, cutting roughly 10× at each step. Contrast that with the rookie move of starting in the logs and grepping for "error" — you start with everything and stay with everything.

![Before and after diagram contrasting the monitoring mindset that ends in guessing with the debugging mindset that attaches an exemplar trace to the spike and reaches root cause in five minutes](/imgs/blogs/observability-for-debugging-prod-3.png)

The difference between a monitoring setup and a debugging setup is exactly whether step two is possible. A monitoring dashboard shows you the spike and stops — it tells you the system is unhealthy and then strands you, because there is no link from the aggregate to a single request. A debugging setup makes the metric *clickable down to a trace*, and the trace *clickable down to logs*. If you take one architectural lesson from this post, it is: **wire your exemplars and propagate your trace ids, so that every aggregate is one click from a representative request.** That single capability is the difference between a five-minute root cause and a two-day guess.

Let me make the descent concrete as a set of layers, because seeing it as a stack helps you remember where you are during a stressful incident.

![Stack diagram showing the drill-down layers from the service-level objective down through the RED panel, the histogram bucket, the exemplar trace, the slow span, and finally the single span log line](/imgs/blogs/observability-for-debugging-prod-4.png)

You enter at the top — a single number, the SLO, which is true of all traffic. You descend one layer at a time, and at each layer the number of things you're looking at shrinks by an order of magnitude, until at the bottom you're looking at one log line from one request. The skill of prod debugging is largely the discipline to *descend in order* and not skip layers — because every layer you skip is a search space you re-expand.

## 4. RED and USE: which dashboard to open first

When the page fires, you have seconds, not minutes, to decide where to look. The wrong first dashboard costs you ten minutes of scrolling. Two methods tell you which dashboard to open first, and which one you pick depends on the *shape* of the symptom.

**RED is for request-shaped symptoms.** RED stands for **Rate, Errors, Duration**, measured *per service* (the framing comes from Tom Wilkie, building on Google's Four Golden Signals). Rate is requests per second; Errors is the fraction that failed; Duration is the latency distribution (p50, p95, p99). RED is what you open when the symptom is something a *user* experiences — "it's slow," "we're throwing 500s," "requests are timing out." It's request-centric: it answers "which of my services is misbehaving from the caller's point of view?" When checkout p99 spikes, you open the RED panel for the checkout service, see the Duration line jump, and you've already localized to one service before you've touched a trace.

**USE is for resource-shaped symptoms.** USE stands for **Utilization, Saturation, Errors**, measured *per resource* (Brendan Gregg's method). Utilization is how busy a resource is (CPU at 92%); Saturation is how much work is queued waiting for it (run-queue length, connection-pool wait); Errors is resource-level errors (failed allocations, dropped packets). USE is what you open when the symptom is something a *machine* experiences — "this host is hot," "we're out of connections," "the disk is full," "OOM kills." It's resource-centric: it answers "which resource on which host is starved?" When a single instance's latency is bad but the others are fine, USE on that host tells you its connection pool is saturated.

The two methods are complementary, not competing. RED finds the *symptom* from the request side; USE finds the *resource cause* from the machine side. A classic investigation uses both: RED tells you the checkout service's Duration spiked (request side), and USE tells you the database host's connection-pool saturation went to 100% at the same moment (resource side) — and now you have both ends of the story.

![Matrix mapping the five RED and USE signals to the underlying telemetry each reads and the suspect each one points you toward during an investigation](/imgs/blogs/observability-for-debugging-prod-6.png)

Here is the decision in table form — the thing to glance at when the page fires.

| Symptom you were paged for | Shape | Open first | Why |
| --- | --- | --- | --- |
| "Checkout is slow" / p99 up | Request | RED Duration, per service | Localizes the slow service before you trace |
| "We're throwing 500s" | Request | RED Errors, per service + endpoint | Finds the failing route |
| "Traffic dropped to zero" | Request | RED Rate, per service | Distinguishes outage from load shift |
| "This host is on fire" | Resource | USE Utilization, per host | Finds the busy resource |
| "Out of DB connections" | Resource | USE Saturation, pool wait time | Finds the starved pool |
| "OOMKilled" / restarts | Resource | USE Errors + memory, per pod | Confirms the resource that failed |

And as a decision tree, because the very first fork — is this symptom request-shaped or resource-shaped? — is the one that determines your first ten minutes.

![Decision tree splitting an incoming page into request-shaped symptoms that start at RED panels and resource-shaped symptoms that start at USE panels, with leaves for latency, errors, CPU, and pool saturation](/imgs/blogs/observability-for-debugging-prod-8.png)

A practical note: build these dashboards *before* the incident, one RED panel per service and one USE panel per resource class, and put a deploy-marker annotation on every one of them (more on that in section 8). During an incident you should be *reading* dashboards, not *building* them. The five minutes you spend writing a PromQL query during an outage is five minutes the outage continues.

## 5. Localizing with traces: reading the waterfall

You've used the metric to find *when* and the RED panel to find *which service*. Now the trace tells you *where inside that request* the problem lives. Reading a trace waterfall well is a learnable skill, and there are five patterns that account for the overwhelming majority of what you'll find.

**Pattern one: the one slow span.** The waterfall is a set of horizontal bars, one per span, nested by parent-child, laid out on a time axis. In a healthy trace the bars are short and the total is small. In a slow trace, one bar is dramatically longer than the rest — the request spent 90% of its time in a single span. That span *is* your localization: it names the service and operation that ate the latency. This is the happy case, and it's more common than you'd think, because most latency regressions are one dependency going from fast to slow.

**Pattern two: the fan-out that waits on the slowest child.** A parent span often fans out to several children in parallel — checkout calls pricing, inventory, and the cache all at once. The parent cannot return until *all* children return, so the parent's duration equals the *slowest* child, not the average. This is the mechanism behind tail-latency amplification: if a service fans out to 10 backends and each has a 1% chance of being slow, the request has roughly a $1 - 0.99^{10} \approx 9.6\%$ chance of hitting at least one slow backend, so the request's tail is ten times worse than any single backend's tail. When you see a parent span whose duration matches one specific child's duration, that child is your suspect — and the fan-out structure explains why a backend that's "only slow 1% of the time" causes a visible p99 regression.

**Pattern three: the retry storm — N identical spans.** Look for the same downstream call repeated two, three, or more times in a row, each one ending in a timeout or error, the last one succeeding (or the whole thing failing). That's a retry. A retry that fires three times turns one 800 ms timeout into a 2.4-second request, and a retry storm across a fleet can take down the very dependency it's retrying (a thundering herd — more in the war story). The signature is *N near-identical spans* with the same operation name and similar durations. If you see them, the bug isn't (only) the slow dependency; it's the retry policy amplifying it.

**Pattern four: the missing span.** Sometimes the bug is what *isn't* there. A request that should have called the inventory service has no inventory span at all — the call was skipped, short-circuited by a feature flag, or swallowed by a code path that returned early. A missing span is invisible if you're only looking at what's slow, which is why you must know what a *healthy* trace looks like: compare the broken trace against a known-good one and ask "what spans does the good one have that this one doesn't?" The absence is the clue.

**Pattern five: the gap — time with no span.** Occasionally there's a stretch of the waterfall where the parent is running but no child span covers it. That gap is un-instrumented work: a slow in-process computation, a lock wait, a GC pause, a DNS lookup nobody traced. The gap tells you *where to add instrumentation* even if it can't yet tell you what's in it.

![Graph of a trace waterfall showing the gateway fanning out through the checkout service to a fast cache, a slow pricing database, a retry of the slow call, and a missing inventory span, all converging on a response that waits on the slowest child](/imgs/blogs/observability-for-debugging-prod-5.png)

Reading a waterfall, then, is pattern-matching against these five shapes: one long bar (the slow span), a parent matching its slowest child (fan-out), repeated bars (retry), an absent bar (missing span), or a bar with internal dead time (gap). Each shape points at a different class of bug and a different fix. And every one of them is invisible in metrics — the metric just says "slow"; only the trace shows you the shape.

#### Worked example: a p99 latency regression localized in four clicks

Let me run the full loop on a concrete regression, with numbers, the way it actually goes.

**The symptom (metric).** At 14:03 the checkout service's p99 latency steps from 90 ms to 180 ms on the Grafana panel. p50 is unchanged at 40 ms — so it's a *tail* problem, not a broad slowdown; most requests are fine, a slice are terrible. The p99 line is computed with PromQL from a histogram:

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{service="checkout"}[5m]))
)
```

`http_request_duration_seconds_bucket` is a Prometheus histogram: each request increments the counter for every bucket whose upper bound `le` ("less than or equal") is at least its latency. `rate(...[5m])` gives the per-second increase of each bucket over a 5-minute window; `sum by (le)` aggregates across instances; `histogram_quantile(0.99, ...)` interpolates the 99th-percentile latency from the bucket counts. The step from 90 to 180 at 14:03 is unambiguous and sharp — a step, not a ramp, which screams *configuration change*, not *organic load growth*.

**The pivot (exemplar trace).** The panel has exemplars enabled, so the spike is dotted with little diamonds, each carrying a trace id. I click one in the slow region and land on a trace whose total duration is 840 ms. p99 in the panel said ~180 ms but this exemplar is 840 ms — that's expected; exemplars catch the worst requests in the bucket, and the worst are far past p99.

**The localization (waterfall).** The waterfall shows the checkout span (840 ms) with three children: `cache.get` at 4 ms, `inventory.reserve` at 12 ms, and `pricing.quote` at **812 ms**. One slow span (pattern one), and it's the pricing call. Before today, `pricing.quote` ran in about 5 ms. So a call that took 5 ms now takes 812 ms — a 160× regression on one dependency, and because the parent fans out and waits on the slowest child, that 812 ms becomes the whole request.

**The explanation (logs).** I drill into the `pricing.quote` span's logs, filtered by its trace id. The pricing service logged: `cache miss for price_table; falling back to full table scan; rows=1.2M`. The pricing service had an in-memory price cache; something stopped it from being populated. I check the deploy timeline (section 8): pricing service shipped a config change at 14:02 that set `PRICE_CACHE_ENABLED=false` — a typo in an environment override that was meant for a different service. With the cache off, every quote did a full scan. p99 jumped one minute after the deploy. Total elapsed from page to root cause: under ten minutes, four clicks. The fix is a one-line config revert; the *finding* came entirely from the metric → exemplar → trace → log drill-down, none of which I could have done with a debugger.

That investigation is impossible without the pivots. The metric alone says "slow at 14:03" and leaves you guessing across six services. The trace alone is one of millions and you'd never find it. The log alone (`cache miss; full table scan`) is one of thousands of benign-looking lines. Together, in order, they converge in minutes.

## 6. Cardinality: why user id is gold and why it costs money

Here is the tension at the heart of debugging with telemetry, and the thing that separates engineers who *think* they have observability from those who actually can debug with it: **the fields that make telemetry debuggable are exactly the fields that make it expensive.**

The mechanism is *cardinality* — the number of distinct values a field can take. A field like `http_method` has cardinality ~5 (GET, POST, PUT, DELETE, PATCH). A field like `status_code` has cardinality ~40. A field like `user_id` has cardinality in the *millions*. A field like `request_id` has cardinality equal to your total request count — effectively infinite. For metrics, cardinality is multiplicative and brutal: every unique combination of label values creates a separate time series that must be stored, indexed, and kept in memory. A counter `http_requests_total{service, endpoint, status, region}` with 20 services × 200 endpoints × 40 statuses × 5 regions is 800,000 time series — and that's before you add `user_id`. Add `user_id` with a million users and you've multiplied by a million; you've created a *cardinality explosion* that will OOM your Prometheus and bankrupt your metrics bill.

So why is high cardinality *gold* for debugging? Because the high-cardinality fields are exactly the ones that localize a bug to a specific request. "Some users get 500s" is undebuggable. "Users in tenant 8841 get 500s" is a five-minute fix. The `tenant_id`, the `endpoint`, the `user_id`, the `request_id` — these are what let you go from "0.2% of requests fail" to "all of the failures are from tenant 8841 on the `/checkout` endpoint." Strip them out to save money and you've made your telemetry cheap *and undebuggable*. Keep them all in metric labels and you've made it debuggable *and unaffordable*.

The resolution is to put cardinality where it belongs:

| Field type | Example | Put it in | Why |
| --- | --- | --- | --- |
| Low cardinality | service, status_class, region | Metric labels | Cheap, aggregatable, pre-computed |
| Medium cardinality | endpoint, http_status | Metric labels (carefully) | Useful for slicing, watch the multiplication |
| High cardinality | user_id, tenant_id, request_id | Traces and logs | Per-request, not per-time-series |
| Unbounded | request_id, trace_id | Trace and log attributes only | Never a metric label, ever |

The rule: **metrics carry low-cardinality dimensions for aggregation; traces and logs carry high-cardinality fields for localization.** A trace already costs one record per request, so adding `user_id` to a span is free — it doesn't multiply anything. A metric costs one time series per label combination, so adding `user_id` multiplies your storage by your user count. This is also why the workflow in section 3 is correct: you *aggregate* on the cheap low-cardinality metric to find *when*, then *pivot* to the expensive high-cardinality trace to find *which one*. You never try to slice the metric by `user_id`; you slice the *traces* by it, because that's where the cardinality is affordable.

There is one important exception: a *bounded* high-value dimension can earn its place as a metric label if it's the one you constantly debug by. If `tenant_id` has 500 values (not a million) and tenant-specific bugs are your most common incident, a `errors_total{tenant_id}` metric with 500 series is cheap and lets you *alert* per tenant. The skill is knowing which one dimension is worth the cardinality. Everything else lives in traces and logs.

#### Worked example: finding the 0.2% of requests that 500 via a high-cardinality error metric

The second classic prod bug: not slow, but *occasionally wrong*. The error rate metric shows `/checkout` throwing 500s at 0.2% of requests — 2 in every 1,000. The other 99.8% are fine. This is the nightmare case for a debugger, because you cannot reproduce it: 998 times out of 1,000 the code works, and you have no idea what's special about the 2.

**Find the WHEN and the WHO (metric).** First, is 0.2% new or always-there? I diff the error-rate metric over the last week:

```promql
sum(rate(http_requests_total{service="checkout", status=~"5.."}[5m]))
/ sum(rate(http_requests_total{service="checkout"}[5m]))
```

It's been flat at ~0% and stepped to 0.2% three days ago — a regression, not eternal background noise. Now, *who*? Here's where one well-chosen, bounded high-cardinality label pays for itself. We have an `errors_total{endpoint, error_type}` metric — `error_type` is bounded (a few dozen exception classes), so it's an affordable label. I break the errors down by type:

```promql
topk(5, sum by (error_type) (rate(errors_total{service="checkout", status=~"5.."}[5m])))
```

100% of the new 500s are `error_type="NullPointerException"`. Not a database timeout, not a downstream outage — a null deref in our own code, on a small slice of requests.

**Find WHICH request (exemplar trace).** I pivot from the error metric to an exemplar of a failing request — a trace tagged `error=true` in the spike window. The trace shows the checkout span failing inside `pricing.quote` with a span status of ERROR. So far so good: it's the pricing call again, but failing this time, not slow.

**Find WHY (log).** I pull the logs for that trace id and span. The log line:

```log
ERROR pricing.quote tenant_id=null trace_id=a1f3...  NullPointerException at PriceResolver.resolve(PriceResolver.java:88)  "Cannot read field 'currency' of null tenant"
```

`tenant_id=null`. The pricing code did `tenant.getCurrency()` without null-checking `tenant`. So *which* requests fail? The ones that arrive with a null tenant. I go back and query traces with `tenant_id` unset over the window — they're 0.2% of traffic, and *all* of them 500. The high-cardinality `tenant_id` field, which lives in the trace and log (not as a metric label), is what let me characterize the failing population precisely: requests from a specific client SDK version that stopped sending the tenant header after an upgrade three days ago. The metric found the *rate* and the *type*; the trace found *a* failing request; the log's high-cardinality `tenant_id=null` found the *whole failing class*. Fix: null-check the tenant and 400 (not 500) on a missing header, and patch the SDK. The 0.2% goes to 0.0% — verified by watching the same `errors_total{error_type="NullPointerException"}` query flatline for an hour after deploy.

Note what made this debuggable: `error_type` was a *bounded* metric label (cheap), and `tenant_id` was a *high-cardinality* trace/log field (also cheap, because it's per-request). Had we tried to put `tenant_id` on the metric, we'd have a million series and a dead Prometheus. Had we *not* put it on the trace, we'd never have characterized the failing class. Right cardinality in the right place is the whole game.

## 7. The tail-latency hunt: tail-based sampling for "slow for 0.1% of users"

Section 5's regression was an easy tail problem — the slow requests were a third of traffic, so any exemplar caught one. The hard tail problem is "it's slow for 0.1% of users," and it exposes a brutal limitation of the naive approach.

The limitation is *sampling*. You cannot store a trace for every request at scale — at 40,000 requests per second, full tracing would be petabytes a day and would cost more than the service it's tracing. So you sample: you keep, say, 1% of traces and drop the rest. *Head-based sampling* makes the keep/drop decision at the *start* of the request, before you know anything about it — flip a weighted coin at the gateway, and if it says keep, propagate that decision to every downstream span. It's cheap and simple. And it's *useless for the tail*, because the decision is made before the request is slow. If 0.1% of requests are slow and you sample 1% of all requests at the head, you keep roughly 1 in 100 of the slow ones — and the slow ones are 1 in 1,000 — so you capture about 1 slow trace per 100,000 requests. You'll wait an hour to get a handful, and your exemplars will overwhelmingly be the *fast* requests you didn't need.

The fix is *tail-based sampling*: make the keep/drop decision at the *end* of the request, after you know its outcome. Buffer all spans of a trace until the request completes, then apply a policy — *keep it if it was slow, or errored, or touched a rare code path; drop it if it was a boring fast success.* Now you keep 100% of the slow and erroring traces and 1% of the boring ones, so your trace store is dense with exactly the requests you need to debug and sparse on the ones you don't. The 0.1% tail is now fully captured.

The cost is real and worth stating plainly: tail-based sampling requires buffering every trace's spans somewhere (memory in a collector tier) until the trace completes, which means a fleet of collector instances holding in-flight traces, and it requires all spans of a trace to route to the *same* collector (so it can make one decision). It's more infrastructure than head-based sampling. But for debugging the tail, it's the difference between catching the bug and never seeing it.

| Sampling strategy | Decision made | Catches the slow tail? | Cost |
| --- | --- | --- | --- |
| Head-based, 1% | At request start, random | No — keeps 1% of slow ones | Cheap, stateless |
| Head-based, 100% | Always keep | Yes, but stores everything | Storage explosion |
| Tail-based | At request end, by outcome | Yes — keeps all slow/error | Collector must buffer traces |
| Tail-based + rate cap | At end, keep slow up to a budget | Yes, bounded | Buffer + per-policy logic |

Here is a tail-sampling policy in the shape an OpenTelemetry Collector uses — keep everything slow or errored, plus a small fraction of the rest:

```yaml
processors:
  tail_sampling:
    decision_wait: 10s          # buffer spans up to 10s for a trace to finish
    num_traces: 100000          # in-flight traces held in memory
    policies:
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }   # all errored traces
      - name: keep-slow
        type: latency
        latency: { threshold_ms: 500 }            # all traces over 500ms
      - name: sample-the-rest
        type: probabilistic
        probabilistic: { sampling_percentage: 1 } # 1% of boring fast successes
```

With this policy, the "slow for 0.1% of users" hunt becomes tractable: every one of those slow requests is in your trace store, tagged and queryable. You query `duration > 500ms AND service = checkout` and you get *all* of them, not a random 1%. Then you look for what they have in common — same region? same SDK version? same tenant? same downstream? same time-of-day? — using the high-cardinality fields from section 6. The pattern in the captured population *is* the root cause hypothesis.

#### A note on the math of catching rare events

It's worth making the probability explicit, because it justifies the whole tail-sampling argument. If a bug affects a fraction $p$ of requests and you sample a fraction $s$ at the head independently of outcome, the probability that a given captured trace is a buggy one is just $p$ — the sampling doesn't enrich for the bug at all. To capture $k$ buggy traces you need to wait for roughly $k / (p \cdot s)$ requests. For $p = 0.001$ (0.1%), $s = 0.01$ (1%), and $k = 10$ traces to find a pattern, that's $10 / (0.001 \times 0.01) = 1{,}000{,}000$ requests of head-sampled waiting. With tail sampling, $s_{\text{slow}} \approx 1$ for the slow population, so you need about $k / p = 10{,}000$ requests — a hundredfold faster to a diagnosable sample. That two-orders-of-magnitude speedup is why, for rare-event debugging, *where* you make the sampling decision matters more than almost anything else.

## 8. Diffing across the deploy marker: bisection on a live system

Most prod regressions are *caused by a change* — a deploy, a config push, a feature-flag flip, a dependency upgrade. And the single most powerful debugging move in production is also the cheapest: **align the metric regression with the change timeline and you've localized the cause to one change, without a single trace.** This is `git bisect` applied to a running system — the same binary-search-the-gap logic from the [bisection post](/blog/software-development/debugging/binary-search-your-bug-with-bisection), except the axis you're searching isn't commits, it's *time*, and the boundary you're looking for is the deploy marker where the metric stepped.

The mechanism that makes this work is the *deploy marker*: every time you ship, you emit an annotation onto your dashboards — a vertical line at the deploy timestamp, tagged with the version. Grafana supports annotations; many teams push them automatically from CI. Now, when a metric steps, you don't have to *guess* whether a deploy caused it — you *look*: is there a deploy marker right before the step? If the p99 line steps from 90 ms to 180 ms at 14:03, and there's a `pricing v1.8.4` marker at 14:02, the deploy is your prime suspect with near-certainty. The step plus the marker is a falsifiable hypothesis (in the sense of the [hypothesize-and-falsify post](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope)): *if v1.8.4 caused it, rolling back to v1.8.3 should return p99 to 90 ms.* And you can test it directly — roll back and watch.

![Timeline showing a baseline p99, a version deploy marker, the latency step one minute later, the moment an exemplar trace is pulled, and the rollback that returns p99 to baseline](/imgs/blogs/observability-for-debugging-prod-7.png)

There's a subtlety: you must *version-label your metrics* for the cleanest version of this. If every metric carries a `version` label (`http_request_duration_seconds_bucket{service="checkout", version="v1.8.4"}`), you can plot p99 *per version on the same panel* and literally see the new version's line sitting above the old one's during a canary — the regression is visible *before* you fully roll out, while both versions are serving traffic side by side:

```promql
histogram_quantile(0.99,
  sum by (le, version) (
    rate(http_request_duration_seconds_bucket{service="checkout"}[5m])
  )
)
```

This `sum by (le, version)` query is the canary's safety net: during a deploy both versions emit metrics, and the per-version p99 lines diverge the instant the bad version starts serving. You catch the regression at 1% of traffic instead of 100%. This is *bisection in the deploy dimension* — instead of bisecting across thousands of commits with `git bisect run`, you've labelled the live metric with the version and let the divergence between two lines do the bisection for you in real time.

When the regression *isn't* a deploy — when the metric ramps rather than steps, or steps with no marker nearby — that absence is itself a clue: it points you away from "we shipped something" and toward "the environment changed" (traffic grew, a dependency degraded, data accumulated, a cache slowly filled). The shape of the metric change — sharp step vs. gradual ramp — and its alignment (or not) with a deploy marker is one of the highest-signal, lowest-effort reads in all of prod debugging. Always look at the deploy markers first.

## 9. A real war story: the retry storm that took down a healthy service

Let me tell you about a failure mode that I've watched (and helped cause) more than once, because it ties together traces, the fan-out pattern, and a counterintuitive truth: *the bug was not in the service that fell over.* This is a realistic, illustrative reconstruction of a well-known class of incident — the thundering-herd retry storm — not a specific company's documented postmortem, but the pattern is real and has felled many production systems.

A payments service (call it `charge`) depends on a fraud-scoring service (`fraud`). Normally `fraud` responds in 20 ms. One afternoon, `fraud` has a minor blip: a slow database query pushes its p99 to 1.2 seconds for about ninety seconds. Not great, but survivable — `fraud` is still *up*, just briefly slow. The `charge` service has a retry policy: if `fraud` takes longer than 500 ms, retry, up to 3 times. Sensible-looking policy. Here's what happens.

When `fraud` slows to 1.2 s, every `charge` request times out at 500 ms and *retries*. Now `fraud` is receiving its normal load *plus* a flood of retries — effectively 3× the traffic. The extra load makes `fraud` *slower*, which makes *more* requests time out, which generates *more* retries. This is a positive feedback loop, and it's vicious: within thirty seconds `fraud` goes from "slightly slow" to "completely saturated," and now it's returning errors, and `charge` is failing every payment. A 90-second minor blip in `fraud` has been amplified by `charge`'s retry policy into a total payments outage. The service that fell over (`fraud`) was the *victim*; the bug was in the *caller's* retry policy.

How do you debug this from the outside, with no debugger? The trace tells the whole story. Pull an exemplar of a failing `charge` request during the storm and look at the waterfall: you see `fraud.score` called **four times in a row** — the original plus three retries — each timing out at 500 ms, total 2+ seconds, the request ultimately failing. That's pattern three from section 5: *N identical spans*, the unmistakable signature of a retry storm. The metric corroborates: `fraud`'s *inbound* request rate (RED Rate) tripled at the exact moment its latency spiked, even though `charge`'s *user-facing* request rate was flat. Inbound traffic to a backend tripling while the front-end traffic is flat is the fingerprint of retry amplification — there's no other way to triple a backend's load without tripling the user load.

The fixes are well-known and worth naming, because the *debugging* points straight at them: cap retries and add *exponential backoff with jitter* (so retries spread out instead of synchronizing into a herd); add a *circuit breaker* (when `fraud` is failing, stop calling it for a few seconds instead of hammering it); and *budget retries* (allow retries only up to a small fraction of total traffic, so a storm can't triple the load). The deeper lesson for *this* post: the trace's retry signature and the metric's inbound-rate triple together localized a bug that lived in the *opposite* service from the one that was alarming. If you'd only looked at `fraud`'s dashboard you'd have spent the incident debugging a healthy service. The cross-service view — which the repo's [debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale) post and the [anatomy of an outage](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) postmortem collection go deeper on — is what saves you. Debugging across service boundaries, where the cause and the symptom live in different services, is a whole discipline of its own (a sibling post in this series, `debugging-across-service-boundaries`, is dedicated to it).

This is also where the SRE mindset earns its keep: treating reliability as a feature means building the retry budgets, circuit breakers, and per-version dashboards *before* the storm, so that when it hits, the system degrades gracefully and the telemetry already shows you the cause. The SRE companion piece `reliability-is-a-feature-the-sre-mindset` makes that case in full.

## 10. The tooling: PromQL, OpenTelemetry, Jaeger, Loki, in practice

Let me ground all of this in the actual tools, because "use traces" is useless advice without knowing how a span gets created and how a trace id reaches a log line. The modern open-source stack is roughly: **Prometheus** for metrics (queried with PromQL), **Grafana** for dashboards, **OpenTelemetry** for instrumentation (the vendor-neutral standard for emitting traces, metrics, and logs), **Jaeger** or **Grafana Tempo** for trace storage and the waterfall UI, and **Loki** for logs. The thing that makes them a *debugging toolkit* rather than three disconnected products is the *trace id*, propagated everywhere, so you can pivot between them.

Here's how a span is created and how the trace id gets onto a log line, in Python with OpenTelemetry — the snippet that makes the whole drill-down possible:

```python
from opentelemetry import trace
import logging

tracer = trace.get_tracer("checkout")
log = logging.getLogger("checkout")

def quote_price(tenant_id: str, sku: str) -> float:
    # A span: one node in the trace waterfall. It records start, duration,
    # status, and attributes, and it carries the trace context downstream.
    with tracer.start_as_current_span("pricing.quote") as span:
        span.set_attribute("tenant_id", tenant_id)   # high-cardinality -> on the span, not a metric
        span.set_attribute("sku", sku)

        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")      # the id that links metric exemplar -> trace -> log

        if tenant_id is None:
            # This log line is what you read at the bottom of the drill-down.
            # Because it carries trace_id, you can filter the firehose to THIS request.
            log.error("null tenant on price quote", extra={"trace_id": trace_id, "sku": sku})
            span.set_status(trace.StatusCode.ERROR, "null tenant")
            raise ValueError("tenant_id is required")

        price = _lookup_price(tenant_id, sku)         # the downstream work being timed
        span.set_attribute("price", price)
        return price
```

Three things in that snippet are doing the heavy lifting for debuggability. First, the `with tracer.start_as_current_span(...)` block *is* the span — it's automatically timed and placed in the waterfall, and it propagates the trace context to anything it calls, so the downstream DB call becomes a child span automatically. Second, the high-cardinality `tenant_id` goes on the *span attribute*, not into a metric label — exactly the cardinality discipline from section 6. Third, and most importantly, the log line carries `trace_id` in its structured fields, so when you've found the slow trace you can query Loki for `{service="checkout"} | json | trace_id="a1f3..."` and get the dozen lines this one request emitted, out of the millions emitted that minute.

On the query side, here are the PromQL idioms you'll use constantly, annotated so you actually understand what each computes:

```promql
# Rate (R in RED): requests per second, per endpoint, over a 5m window.
sum by (endpoint) (rate(http_requests_total{service="checkout"}[5m]))

# Errors (E in RED): fraction of requests that are 5xx.
sum(rate(http_requests_total{service="checkout", status=~"5.."}[5m]))
  / sum(rate(http_requests_total{service="checkout"}[5m]))

# Duration (D in RED): p99 latency from a histogram, the symptom panel.
histogram_quantile(0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{service="checkout"}[5m])))

# USE saturation: connection-pool wait time, the resource-side cause.
max by (instance) (db_connection_pool_wait_seconds{service="checkout"})

# Per-version p99 for canary diffing: see the bad version's line rise.
histogram_quantile(0.99,
  sum by (le, version) (rate(http_request_duration_seconds_bucket{service="checkout"}[5m])))
```

The `rate(...[5m])` wrapper is non-negotiable on counters: a Prometheus counter only increases (it resets to zero on restart), so the raw value is meaningless — you always take its `rate` to get a per-second change, and the `[5m]` window smooths out scrape jitter. `histogram_quantile` interpolates a percentile from bucket boundaries, which is *approximate* (its accuracy depends on how finely you bucketed) — a subtlety worth knowing, because if your buckets are coarse around the latencies you care about, your p99 panel can be off by a lot and mislead the whole investigation.

A few honest words on tool trade-offs, because the kit demands honesty about cost. Tracing every request is too expensive; sample (section 7). High-cardinality metric labels will kill Prometheus; keep cardinality in traces and logs (section 6). Log volume at full verbosity will bankrupt you and bury the signal; log at the *decision points* (the branch taken, the fallback hit, the error), not every line, and rely on the trace for the structure. The art is putting each piece of information in the cheapest signal that can still answer the question it needs to answer — `when` in metrics (cheapest), `where` in traces (per-request), `why` in logs (richest). Get that allocation wrong and you either go broke or go blind.

## 11. Correlation is not causation: the trap of the coincident spike

Everything so far has a hidden failure mode I need to warn you about, because it sends more prod investigations down a rabbit hole than any other single mistake: **two metrics moving together at the same time does not mean one caused the other.** When you're staring at a wall of dashboards during an incident, your brain is a pattern-matcher desperate for a story, and it will *gleefully* hand you a false one. The CPU spiked at 14:03; the latency spiked at 14:03; therefore the CPU caused the latency. Maybe. Or maybe a third thing — a slow downstream dependency — caused *both* the latency (requests pile up) *and* the CPU (the runtime busy-waits on the blocked threads). The coincident spike is a *hypothesis*, never a conclusion.

The reason this is dangerous in observability specifically is that you have *hundreds* of metrics, and during an incident *many* of them move at once. With enough time series, something will always correlate with your symptom by pure coincidence — this is the same statistical trap as p-hacking. If you have 500 metrics and you eyeball them for "what moved at 14:03," you will find a dozen, most of them downstream effects of the real cause or pure noise. Picking the most visually dramatic one and declaring it the root cause is exactly how teams spend an hour optimizing a CPU that was never the bottleneck.

The discipline that saves you is the same one from the [hypothesize-and-falsify post](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope): a correlated metric is a *suspect*, and a suspect must be *confirmed by an independent test before you act on it.* There are three confirmation moves that turn correlation into causation, and you should reach for them every time the stakes are real:

- **The timing test — which moved first?** Causation flows forward in time. Zoom in to second-level granularity and ask which metric stepped *first*. If the latency stepped at 14:03:01 and the CPU stepped at 14:03:04, the CPU is downstream — it can't be the cause of something that started three seconds earlier. The leading indicator is the suspect; the lagging one is an effect. This single zoom-in resolves a huge fraction of "which one caused which" questions, and it costs you fifteen seconds.
- **The trace test — is there a mechanism?** A real cause leaves a mechanism you can see in a trace. If "the slow database caused the latency," then an exemplar trace from the spike should show a long database span. If you pull the trace and the database span is 4 ms, your correlation was a coincidence — the database is innocent, no matter how dramatically its metric moved. The trace is the independent witness that confirms or destroys the metric-level story.
- **The intervention test — does removing the suspect remove the symptom?** The strongest confirmation: change the suspect and watch the symptom. Roll back the deploy, disable the feature flag, fail over the host, scale up the pool. If the symptom clears, you've proven causation; if it doesn't, your suspect was innocent and you've saved yourself from "fixing" the wrong thing. This is the live-system equivalent of commenting out a line to see if the bug goes away.

Here's the trap in a concrete shape. During the retry storm from section 9, you'd see `fraud`'s CPU at 100%, its latency at 1.2 s, its error rate climbing, *and* its garbage-collection time up, *and* its thread count up — five metrics screaming at once. A panicked responder picks "CPU 100%" and pages the platform team to add capacity. But the timing test shows `fraud`'s *inbound request rate* tripled *before* the CPU saturated — the request rate led, the CPU lagged. The CPU was an *effect* of the retry flood, not the cause. Adding capacity would have helped marginally and missed the real fix entirely (cap the retries). The leading metric — inbound rate — was the suspect; the dramatic metric — CPU — was a red herring. Always ask *what moved first*, and always confirm with a trace before you act.

This is also why the deploy-marker diff from section 8 is so valuable: a deploy marker is a *known intervention at a known time*, so it sidesteps the correlation trap. You're not asking "which of 500 metrics happened to move," you're asking "did this specific, deliberate change line up with the symptom" — a far narrower, far more trustworthy question. Whenever you can anchor your investigation to a known event (a deploy, a config push, a traffic shift you can see in the rate metric, a failover) rather than to a coincidental metric movement, do it.

## 12. Structured logs: the discipline that makes the bottom of the drill-down work

The whole drill-down ends at a log line, and whether that final step takes five seconds or five hours comes down entirely to *how you logged.* Two log lines can describe the same event; one is debuggable and one is useless. Let me show you the difference, because this is where the rubber meets the road and where most teams quietly fail.

Here is the useless version — the one most code emits by default:

```log
ERROR: failed to process request
```

And here is the debuggable version — structured, with the keys that let you pivot:

```json
{
  "level": "error",
  "ts": "2026-06-20T14:03:07.412Z",
  "service": "pricing",
  "msg": "price quote failed",
  "trace_id": "a1f3c2...",
  "span_id": "9b2e...",
  "tenant_id": null,
  "sku": "WIDGET-42",
  "error_type": "NullPointerException",
  "version": "v1.8.4"
}
```

The difference is not cosmetic; it is the difference between a log that participates in the drill-down and one that doesn't. Walk the keys: `trace_id` is what links this line to the exemplar trace you came from — without it, you found a slow trace and then have to *guess* which of the thousand `ERROR: failed` lines belongs to it. `tenant_id: null` is the high-cardinality field that, in section 6, characterized the *entire failing class* of requests. `error_type` is the bounded field you can also aggregate into a metric. `version` ties the line back to the deploy. `ts` at millisecond precision lets you order events within a request. Each key is a *pivot point* — a field you can filter or group by — and the unstructured version has none of them.

The mechanism that makes structured logs work for debugging is that they're *queryable as data, not as text*. With Loki or any structured log store you write `{service="pricing"} | json | trace_id="a1f3c2..."` and get exactly this request's lines, or `{service="pricing"} | json | tenant_id="" and error_type="NullPointerException"` and get the whole failing population, instantly, out of millions of lines. The unstructured `grep "failed"` gives you every failure ever, with no way to isolate the one request or the one class. At prod scale, *text grep does not scale and structured query does* — that is the entire argument for structured logging, and it's why the [logging-as-an-instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) sibling post treats it as foundational.

Three rules make logs debuggable in practice. First, **always carry the trace id** (and span id) — it's the join key for the whole drill-down; a log without a trace id is an orphan you can't connect to anything. Second, **log decisions, not noise** — the branch taken, the fallback hit, the validation that failed, the retry attempted — because those are the *why* you'll be reading; logging every function entry buries the one line that matters under a million that don't. Third, **put values in fields, not in the message string** — `tenant_id: null` as a field is queryable; `"failed for tenant null"` as prose is not, and the difference is whether you can find the failing class in one query or never.

#### Worked example: the intermittent 503 that only one host produced

Let me run a third investigation that exercises the correlation discipline and the structured-log pivot together, because it has a twist the first two didn't: the symptom was *not* uniform across the fleet.

**The symptom (metric).** The `api` service throws 503s at about 0.4% of requests — low, but customers notice because retries fail too. The error-rate metric, summed across the fleet, is a flat 0.4%, which looks like a uniform low-grade problem. p50 and p99 latency are *normal* — this is purely an error problem, not a latency one, which already rules out the slow-dependency story.

**The correlation trap, avoided.** Several metrics moved when the 503s started: memory was a little higher, GC time was up, connection count was up. The tempting story: "memory pressure is causing the 503s." But I apply the timing test and the trace test before acting. The trace test first: I pull an exemplar of a 503 and the trace shows the request *never reached application code* — it was rejected at the server's accept layer with the connection refused. That's not a memory-pressure-in-the-handler story at all; the request died before any handler ran. The memory correlation was a red herring (memory is always a little higher under any load bump). The mechanism the trace revealed — rejected at accept — points somewhere completely different: the listen backlog or the worker pool is full.

**The twist — it's one host (USE, per instance).** Here's where summing across the fleet hid the truth. I break the 503 rate down *per instance* instead of fleet-wide:

```promql
sum by (instance) (rate(http_requests_total{service="api", status="503"}[5m]))
```

The 503s are *not* spread across 50 hosts at 0.4% each. They are *concentrated on one host* throwing them at ~20%, and the other 49 hosts at 0%. The fleet-wide 0.4% was `20% ÷ 50 hosts`. One sick host, averaged into invisibility. Now USE on *that host*: its connection-pool saturation (queue depth waiting for a worker) is pinned at 100%, while every other host is near 0. One instance's worker pool is exhausted; it can't accept new connections, so it 503s a fifth of its traffic.

**The why (structured log, per host).** I query that host's logs filtered to the 503 trace ids: `{service="api", instance="api-prod-37"} | json | status=503`. The lines show `worker pool exhausted; 0 idle workers; 200 queued` repeating. Why only this host? The logs from an hour earlier on `api-prod-37` show a burst of `slow upstream call: 30000ms timeout` — a handful of requests to a flaky dependency hit the 30-second timeout and *held a worker each for the full 30 seconds*, and because they never freed, the pool drained on that one host and never recovered (the other hosts happened not to route those specific slow requests). The fix: a much shorter upstream timeout plus a pool that sheds load instead of queueing unboundedly. After deploy, the per-instance 503 query goes to 0 on all hosts and stays there.

Three lessons in one example. The **correlation trap**: the obvious memory correlation was wrong, and the trace test killed it before I wasted an hour on it. The **aggregation trap**: summing the error rate across the fleet hid a single sick host inside a deceptively-uniform 0.4% — *always be able to break a metric down per instance.* And the **structured-log pivot**: filtering one host's logs by trace id and status went straight from "503s" to "worker pool exhausted by a 30-second-timeout upstream call" in two queries. None of it was reproducible locally; all of it came from telemetry that was already there.

## 13. How to reach for this (and when not to)

Observability is powerful and it is not free, so here is the decisive guidance on when to use which signal and when to *not* reach for this at all.

**Reach for the metric → trace → log drill-down when the bug is in production and you cannot reproduce it locally.** That is the whole use case. If you *can* reproduce it on your laptop, do that instead — a local debugger gives you far more than any trace, because you can stop the world, inspect every variable, and step through. Don't run a distributed-tracing investigation for a bug you can catch in `pdb` in thirty seconds. The prod toolkit is for the bugs that *only* exist in prod.

**Start at the metric, always.** The single most common rookie mistake is diving into logs first and grepping for "error." At prod scale that's a firehose with no filter; you'll drown. Confirm the symptom and find `t0` in the metric, *then* descend. The discipline of descending in order is most of the skill.

**Don't attach a debugger to a process serving live traffic.** I've said it twice and I'll say it a third time because it's the one that causes outages. A breakpoint blocks every concurrent request on that process. If you absolutely must inspect live process state, use a *non-stopping* tool — `py-spy dump` for a Python stack sample, `jstack` for a JVM thread dump, an `eBPF`/`bpftrace` probe — that reads state without halting. Never a breakpoint in prod. (When you *can* get a core dump or a crashed process, post-mortem debugging is a different and safe story — covered elsewhere in this series.)

**Don't put high-cardinality fields in metric labels.** `user_id` or `request_id` as a Prometheus label is the classic way to OOM your metrics backend and get paged for the *monitoring* going down during an incident. High cardinality belongs in traces and logs. If you find yourself wanting to slice a metric by user, that's the signal to pivot to traces, not to add a label.

**Don't trace everything; don't log everything.** Both are tempting and both are how you go broke and bury the signal. Sample traces (tail-based for the rare-event tail), and log at decision points, not every line. More telemetry is not more observability past a point — it's more *cost* and more *noise*, and noise actively hurts debugging because the signal you need is buried in it.

**Don't build dashboards during an incident.** If you're writing PromQL at 3am, you've already lost time. Build the RED-per-service and USE-per-resource dashboards, with deploy-marker annotations, *before* you need them. During an incident you should be reading, pivoting, and forming hypotheses — not authoring queries.

**Reach for the deploy-diff first when a metric steps.** Before you trace anything, look at the deploy markers. If the step lines up with a deploy, you've done most of the localization for free, and a rollback is a one-step falsification test. Tracing is for *after* you've used the cheap deploy-diff to narrow the suspect.

| Situation | Reach for | Don't |
| --- | --- | --- |
| Reproducible locally | A real debugger (`pdb`, `gdb`, `delve`) | Don't trace what you can breakpoint |
| Prod, can't reproduce | Metric → trace → log drill-down | Don't grep logs first |
| Need live process state | `py-spy dump`, `jstack`, `bpftrace` | Never a breakpoint in prod |
| Rare-event tail (0.1%) | Tail-based sampling | Don't rely on 1% head sampling |
| Metric stepped sharply | Diff across the deploy marker | Don't trace before checking deploys |
| Want to slice by user/tenant | High-cardinality on traces/logs | Don't add it as a metric label |

## 14. Key takeaways

- **Observability is the debugger for systems you can only watch.** You cannot stop the process, reproduce on demand, or add a print and redeploy in time, so you reconstruct the failed request from the telemetry it already emitted.
- **The three signals answer different questions: metrics WHAT and WHEN, traces WHERE, logs WHY.** None debugs alone; the power is in the pivots between them, each cutting the search space ~10×.
- **Run the drill-down in order: symptom metric → exemplar trace at that timestamp → logs for that trace id.** Starting in the logs is starting with everything; starting in the metric monotonically narrows.
- **Open RED for request-shaped symptoms, USE for resource-shaped ones.** RED (Rate, Errors, Duration per service) finds the misbehaving service from the request side; USE (Utilization, Saturation, Errors per resource) finds the starved resource from the machine side.
- **Read the waterfall by pattern:** one long span (the slow dependency), a parent matching its slowest child (fan-out tail amplification), N identical spans (a retry storm), a missing span (a skipped call), a gap (un-instrumented work).
- **High-cardinality fields are gold for debugging and expensive as metric labels.** Keep `user_id`, `tenant_id`, `request_id` in traces and logs (per-request, cheap there); keep metric labels low-cardinality (per-time-series, multiplicative).
- **Tail-based sampling catches the rare slow tail; head-based sampling doesn't.** Decide keep/drop at the *end* of a request by outcome, and you store all the slow and erroring traces and 1% of the boring ones — a ~100× faster path to a diagnosable sample.
- **Diffing a metric across a deploy marker is bisection on a live system.** A sharp step aligned with a deploy marker localizes the cause to one change for free; version-label your metrics so canary divergence shows up at 1% rollout.
- **The cause often lives in a different service from the symptom** (the retry storm took down a healthy `fraud` service; the bug was in `charge`'s retry policy). The cross-service trace view is what saves you.
- **A coincident spike is a suspect, not a verdict.** With hundreds of metrics, something always correlates by chance; confirm causation with the timing test (which moved first), the trace test (is there a mechanism), or the intervention test (remove it and watch) before you act.
- **Always be able to break a metric down per instance.** A fleet-wide average hides a single sick host; a deceptively uniform 0.4% error rate was really one host at 20% averaged across fifty.
- **Structure your logs around the join keys.** Carry the trace id, put high-cardinality values in fields not the message string, and log decisions not noise — the difference between a five-second query and a five-hour grep is whether the final log line is queryable as data.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the intro map for this whole series and the observe → reproduce → hypothesize → bisect → fix → prevent loop this post is an instance of.
- [Logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) — the sibling post on writing the structured log line, with trace id, that the bottom of every drill-down here depends on.
- [Binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the bisection logic that the deploy-marker diff in section 8 applies to the time axis.
- [Hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) — turning "the deploy caused it" into a falsifiable, rollback-testable hypothesis.
- [Observability: metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — the architecture side: how to build the exemplar links and trace propagation that make the pivots possible.
- [Debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale) and [Anatomy of an outage: lessons from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) — going wider on cross-service investigations and real incident structure.
- The Prometheus documentation on [histograms and `histogram_quantile`](https://prometheus.io/docs/practices/histograms/), and the [OpenTelemetry](https://opentelemetry.io/docs/) docs on traces, context propagation, and the Collector's tail-sampling processor.
- Brendan Gregg's writing on the [USE method](https://www.brendangregg.com/usemethod.html), and Tom Wilkie's RED method talks — the two complementary entry points to a production investigation.
- *Distributed Systems Observability* by Cindy Sridharan, and Google's *Site Reliability Engineering* chapters on monitoring and the Four Golden Signals, for the conceptual foundations behind RED and USE.

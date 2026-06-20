---
title: "Distributed Tracing in Practice: Following One Request Through Every Service"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to read a trace waterfall, propagate context across every hop, sample without throwing away your errors, and localize a p99 regression to one slow downstream call in seconds instead of an afternoon."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "distributed-tracing",
    "opentelemetry",
    "observability",
    "context-propagation",
    "tail-sampling",
    "latency",
    "debugging",
    "jaeger",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/distributed-tracing-in-practice-1.png"
---

At 2:14 on a Tuesday afternoon the checkout p99 latency on our dashboard went from 250 milliseconds to a little over 800. Not an outage — nothing was down, the error rate stayed flat, the success ratio that fed our SLO barely twitched. But the page felt slow, a few customers complained, and the latency graph had a step in it that lined up suspiciously well with a deploy that had gone out at 2:09. The on-call engineer — me — opened the metrics. The checkout service's own latency was up. So was the order service's. So was the gateway's. Every service in the path showed elevated latency, because every service in the path was *waiting* on the slow one. Metrics told me the whole neighborhood was on fire; they could not tell me which house started it. I spent the next forty minutes opening dashboards, reading logs, and guessing. The thing that finally answered the question — in about four seconds, once I thought to look — was a single trace.

The trace showed one request that had taken 815 milliseconds, drawn as a waterfall of nested bars. Almost all of it — 800 of those 815 milliseconds — was a single span: a call from the payments service to an external card processor that, before the 2:09 deploy, had taken about 5 milliseconds. The deploy had changed a connection-pool setting and the payments service was now opening a fresh TLS connection to the processor on every call instead of reusing a warm one. Auth was fast. Inventory was fast. The database was fast. The 90% of the latency lived in exactly one hop, and the waterfall put that hop right in front of me with no guessing at all. That is the entire pitch for distributed tracing: it is the only telemetry that gives you **cross-service causality** — the actual path of one request through N services, with the time attributed to each hop — and once you can read it, a class of incident that used to eat an afternoon collapses into seconds.

![A graph diagram showing one request fanning out from an API gateway root span through an order service to auth, inventory, and payments spans, with the payments span calling an external card processor as the long pole that owns most of the latency](/imgs/blogs/distributed-tracing-in-practice-1.png)

This post is the practitioner's guide to that skill. We will build the mechanism from the ground up — what a span is, what a trace is, how a trace id and span id ride a header across every hop so the spans reassemble into one tree — and then we will spend most of our time on the things that actually decide whether tracing works in production: **context propagation done right** (and the broken trace when one service drops the header), **reading the waterfall** to find the long pole and the deepest error, **sampling** (head-based versus tail-based, and why tail-based plus always-keep-errors is the SRE default), and **correlation** that ties a trace to its logs and links a metric spike to a representative trace. It sits in the middle of this series' spine: you *define* reliability with SLIs and SLOs, you *measure* it with observability, and tracing is the pillar of observability that answers "where did the time go?" when the metric says something is wrong but not what. If you have not yet read the companion pieces on [the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) or on [when to reach for metrics, logs, or traces](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which), this is the deep dive on the third pillar — the one that turns a fleet of services from a black box into a glass box.

## 1. Why metrics and logs leave a gap tracing fills

Start with the failure of the other two pillars, because that gap is the whole reason tracing exists. You already run **metrics**: numbers aggregated over time — request rate, error rate, p99 latency, queue depth. Metrics are cheap, they are the backbone of your alerts, and they are wonderful at telling you *that* something is wrong and *how much*. They are terrible at telling you *where*. A metric is an aggregate; by the time a thousand requests have been rolled into a p99 number, the identity of any single request — its path, its slow hop — has been averaged away. When checkout p99 jumps, the latency metric for *every* service on the path jumps, because they are all stuck waiting on the same downstream call. The metric points at the whole call graph.

You also run **logs**: discrete events, usually one line per interesting thing that happened, with a timestamp and some fields. Logs are great for "what exactly happened in this one service" — the exception, the SQL that ran, the user id. But a log line lives inside one service. To follow a request across five services using logs alone, you would have to find the matching lines in five different log streams, correlate them by timestamp (which drifts) and by some shared id (which you have to have remembered to put in every line), and mentally reconstruct the ordering. People do this. It is miserable, it is slow, and it falls apart the moment two requests interleave or a service fans out to three downstreams in parallel.

A **trace** is the missing pillar. It is purpose-built for the question metrics and logs cannot answer: *for this one request, where did the time go, and which hop failed?* A trace records the request's journey across every service as a tree of timed operations, with the parent-child structure that tells you what called what and the durations that tell you what was slow. It carries causality. Here is the same comparison laid out as a decision matrix, because the right answer is almost always "use all three, but reach for the right one first."

![A matrix comparing metrics, logs, and traces across the questions they answer, the cardinality they tolerate, and their relative cost, showing that traces uniquely answer where time went across service hops](/imgs/blogs/distributed-tracing-in-practice-6.png)

| Pillar | The question it answers | Cardinality it tolerates | Cost driver | Reach for it when |
| --- | --- | --- | --- | --- |
| Metrics | Is it broken? How much? Is it trending? | Low — every label combination is a new time series, so labels must be bounded | Number of time series | You need an alert, an SLO ratio, or a trend over weeks |
| Logs | What exactly happened inside one service? | High — free text, any field, but per-service | Volume (bytes ingested) | You have a specific service and need the exception, the stack trace, the exact values |
| Traces | Where did the time go across N hops? Which hop failed? | Very high — high-cardinality attributes on spans are fine | Volume × sampling rate | A request is slow or failed and you don't know *which service* owns the problem |

The three are complementary, not competing. A mature setup uses metrics to *detect* (the SLO burn alert fires), traces to *localize* (which hop owns the latency or the error), and logs to *explain* (the exact exception in that hop). The dream is that they are *linked*: the metric spike hands you an exemplar trace, the trace hands you the span, and the span hands you its logs. We will wire exactly that linkage in section 9. For the architecture-time view of designing observability into a system from the start — instrumentation boundaries, the three pillars as a design concern — cross-link out to the system design piece on [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design); here we are squarely in the *running it* layer: you already have services in production and you need to see through them.

### The cost of not having tracing, stated honestly

Before we build, let me put a number on the gap, because the principle of this series is that every practice must justify its cost. In that 2:14 incident, **mean time to identify** — the clock from "the alert fired" to "we know which component is the problem" — was about forty minutes without tracing and about four seconds with it. That is not a one-off. Across a representative quarter of latency incidents on a service mesh I have operated, the distribution looked roughly like this (illustrative, from incident records, not a published benchmark): without tracing, localization took a median of 25 minutes and a long tail past 90 minutes when the slow hop was a transitive dependency nobody suspected. With a trace in hand, localization took under a minute in the large majority of cases, because the waterfall *shows* you the long pole. If your **MTTR** (mean time to recovery — the full clock from detection to resolution) is dominated by the time spent *guessing which service is the problem*, tracing attacks the largest term directly. That is the proof angle, and we will keep returning to it.

## 2. The mechanism: spans, traces, and the tree

Let us build the data model precisely, because everything else depends on getting these three nouns right.

A **span** is the atomic unit. One span represents one operation in one service: an HTTP request handler running, a database query executing, a call to a downstream service, a chunk of business logic you chose to time. A span has, at minimum:

- a **span id** — a unique 64-bit (8-byte) identifier for this operation,
- a **trace id** — a 128-bit (16-byte) identifier shared by *every* span in the same request's journey,
- a **parent span id** — the span id of the operation that caused this one (empty for the very first span, the *root*),
- a **start time** and an **end time** (so its **duration** is end minus start),
- a **name** (a low-cardinality operation name like `GET /orders/{id}` or `SELECT orders`),
- **attributes** — arbitrary key-value tags (`http.status_code=200`, `db.system=postgresql`, `user.id=8831`, `region=us-east-1`),
- a **status** (`OK`, `ERROR`, or unset), and optionally
- **events** — timestamped points inside the span (an exception, a cache miss).

A **trace** is the set of all spans that share a trace id, arranged by their parent-child links into a tree. The root span is the first operation — usually the gateway or the entry service's request handler — and it has no parent. Every other span points at its parent. Reassemble the tree and you have the complete causal structure of the request: who called whom, in what order, and how long each took. That tree is exactly what figure 1 draws — a gateway root, an order service child, and the order service's own children for auth, inventory, and payments, with payments calling out to the external card processor.

The crucial property is that **spans are created independently in each service but reassemble into one tree because they all carry the same trace id and the correct parent span id.** The order service does not know or care that the gateway exists; it just sees an incoming request that carries a trace id and a parent span id in its headers, creates its own span as a child of that parent, and passes its *own* span id down to whatever it calls next. Each service does a small, local thing — read the incoming context, start a child span, pass context onward — and the global tree emerges. This is the same "small local rule, global structure" property that makes the whole thing scale to thousands of services without any central coordinator. The reassembly happens later, in the backend, when all the spans for a trace id arrive and get sorted into a tree.

#### Worked example: the anatomy of our checkout trace

Take the 2:14 request concretely. Seven spans, one trace id (call it `4bf92...`, a 16-byte value rendered as 32 hex characters):

- Root span: `gateway: POST /checkout`, parent = none, duration 815ms.
- Child: `order-service: handleCheckout`, parent = the gateway span, duration 805ms.
- Child of order: `auth-service: verifyToken`, duration 12ms, status OK.
- Child of order: `inventory-service: reserve`, duration 18ms, status OK, with a grandchild `postgres: SELECT stock`, duration 6ms.
- Child of order: `payments-service: charge`, duration 800ms — the long pole.
- Child of payments: `HTTP POST card-processor`, duration 790ms, an external span.

Notice the arithmetic the tree makes obvious. The order service's 805ms is *not* the sum of its children (12 + 18 + 800 = 830) because auth, inventory, and payments run partly in parallel — the waterfall will show that overlap. And the 790ms external call sits *inside* the 800ms payments span, so the payments service itself added only ~10ms of its own work; the rest was waiting. Every one of those facts is visible at a glance in the tree and invisible in the metrics. That is the leverage.

## 3. Context propagation: the one thing that must work

Here is the uncomfortable truth that separates a tracing system that works from one that is a graveyard of disconnected fragments: **a trace is only as complete as your context propagation.** If the gateway starts a trace but the order service does not read the incoming trace id and start its span as a child, the order service starts a *brand new* trace. Now you have two traces where there should be one, neither of which tells the full story, and the slow payments call is stranded in a trace that does not connect back to the user-facing request. The single most common reason tracing "doesn't work" in practice is not the SDK, not the backend, not sampling — it is a hop somewhere that drops the context.

**Context** is the small bundle of identifiers that must travel with the request: at minimum the trace id and the current span id (which becomes the *parent* span id for the next hop), plus sampling and vendor flags. For synchronous HTTP hops, the industry standard is the **W3C Trace Context** specification, which defines a header called `traceparent`. It looks like this:

```bash
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

Four dash-separated fields: `00` is the version; `4bf92f...4736` is the 16-byte (32 hex char) trace id; `00f067aa0ba902b7` is the 8-byte (16 hex char) parent span id; `01` is the trace-flags byte, whose low bit is the **sampled** flag (1 = this trace is being recorded, 0 = it is not). A companion `tracestate` header carries vendor-specific key-value pairs. The whole job of propagation is: on the way *out* of a service, **inject** the current context into these headers; on the way *in* to a service, **extract** them and make them the parent of the new span. Get inject-and-extract right at every boundary and the tree is whole. Miss one boundary and the tree is severed there.

![A graph diagram showing context propagation where service A injects the traceparent header, a synchronous HTTP hop reaches service B, and a publish path puts the trace id into a Kafka message header that a worker extracts on consume, so all spans reassemble into one trace](/imgs/blogs/distributed-tracing-in-practice-7.png)

### Inject and extract, in real code

With OpenTelemetry's auto-instrumentation, the common cases — incoming HTTP requests, outgoing HTTP calls with a standard client, popular database drivers — are handled for you: the instrumentation library injects and extracts automatically. The places you must do it *by hand* are the boundaries the auto-instrumentation does not know about: a custom transport, a homegrown RPC, and — the big one — **async boundaries** where the request leaves a synchronous call stack and travels through a queue or a background job. Here is the manual inject on the producing side and extract on the consuming side, in Python with the OpenTelemetry API:

```python
from opentelemetry import trace, context as otel_context
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer("orders")

# --- Producing side: publishing a message to a queue ---
def publish_order_event(producer, order_id):
    # Start a span for the publish operation itself.
    with tracer.start_as_current_span("publish order.created") as span:
        span.set_attribute("messaging.system", "kafka")
        span.set_attribute("messaging.destination", "order.created")
        span.set_attribute("order.id", order_id)

        # Inject the CURRENT context (trace id + this span id + flags)
        # into a plain dict that becomes the message headers.
        carrier = {}
        inject(carrier)  # writes "traceparent" (and "tracestate") into carrier

        headers = [(k, v.encode()) for k, v in carrier.items()]
        producer.send("order.created", value=serialize(order_id), headers=headers)


# --- Consuming side: a worker pulling from the queue ---
def handle_message(message):
    # Rebuild the carrier dict from the message's headers.
    carrier = {k: v.decode() for k, v in (message.headers or [])}

    # Extract the propagated context. This returns a Context whose
    # current span is the producer's span -> our new span's parent.
    ctx = extract(carrier)

    # Start our span as a child of the extracted context.
    with tracer.start_as_current_span("process order.created", context=ctx) as span:
        span.set_attribute("order.id", parse_order_id(message))
        do_the_work(message)
```

Two things are doing the work here. `inject(carrier)` serializes the active span's context into the W3C headers and writes them into whatever dict you give it — for HTTP that dict is your outgoing request headers; for a queue it is the message headers. `extract(carrier)` reads those headers back into an OpenTelemetry `Context` object, and passing that as `context=ctx` to `start_as_current_span` makes the new span a child of the producer's span. That single `context=ctx` argument is the seam that stitches the consumer's work back onto the request that triggered it. Without it, the worker starts a fresh, parentless trace and the link is gone.

This is also exactly the boundary where most teams' traces break, because a queue *feels* like the end of the request — the HTTP response already went back to the user — so people forget that the downstream work is still part of the same logical operation. The architecture-level treatment of how messages carry metadata across a queue (headers, ordering, delivery semantics) is covered in the message-queue series; here the operational rule is blunt: **every queue you publish to is a context-propagation boundary, and you must inject on publish and extract on consume or the trace dies at the queue.**

### Starting and decorating a manual span

For the business logic you want to time that the auto-instrumentation does not cover, you create spans by hand. The pattern is small and worth memorizing:

```python
from opentelemetry import trace

tracer = trace.get_tracer("payments")

def charge_card(order_id, amount_cents, card_token):
    with tracer.start_as_current_span("charge_card") as span:
        # High-cardinality attributes are fine and welcome on spans.
        span.set_attribute("order.id", order_id)
        span.set_attribute("payment.amount_cents", amount_cents)
        span.set_attribute("payment.provider", "card-processor")
        try:
            result = call_card_processor(card_token, amount_cents)
            span.set_attribute("payment.auth_code", result.auth_code)
            return result
        except ProviderTimeout as e:
            # Record the error ON the span so the waterfall shows it red.
            span.set_status(trace.Status(trace.StatusCode.ERROR, "processor timeout"))
            span.record_exception(e)
            raise
```

Note the `set_status(... ERROR ...)` and `record_exception` calls. They are what make a failed span show up red in the waterfall and carry the exception as a span event. The deepest red span in a trace is, nine times out of ten, the origin of a failure that bubbles up to the user as a gateway 500 — and we will lean on that fact hard in section 5.

## 4. The broken trace: when one service drops the header

Now the failure mode, because the principle of context propagation is best learned through its breakage. Picture our checkout path again, but the order service is running an old internal HTTP client that does *not* inject `traceparent` on its outgoing calls. The gateway starts trace A and calls the order service. The order service does its work — but when it calls the payments service, it sends no `traceparent`. The payments service, seeing no incoming context, starts a brand-new trace B with itself as the root. The slow 800ms payments span now lives in trace B, which has no connection to trace A and no idea it was serving a user checkout. When you open trace A to debug the slow checkout, it ends at the order service. The payments call — the actual culprit — is a *blind spot*. You see the request go in and you see it take a long time, but the tree is amputated exactly where the problem is.

![A before and after diagram contrasting a broken trace where the order service drops the traceparent header and the payments span is orphaned and invisible against a fixed trace where the order service injects the header and the eight hundred millisecond payments span becomes visible end to end](/imgs/blogs/distributed-tracing-in-practice-2.png)

This is not hypothetical. Broken traces are the dominant tracing failure in real fleets, and they come from a short, recurring list of causes:

| Cause of broken trace | Where it bites | The fix |
| --- | --- | --- |
| A service uses a client/framework with no auto-instrumentation | Outgoing calls carry no `traceparent` | Add the instrumentation library, or inject manually |
| An async/queue boundary | The worker starts a fresh trace | Inject on publish, extract on consume (section 3) |
| A proxy or gateway strips unknown headers | `traceparent` deleted in transit | Allowlist the trace headers in the proxy config |
| A thread pool or executor loses the context | The active context is per-thread; work moves threads | Re-attach the context inside the worker (`context.attach`) |
| Mixed propagation formats (B3 vs W3C) | One service speaks B3, the next only reads W3C | Standardize on W3C `traceparent`, or configure a multi-format propagator |

The last row deserves a word. Before W3C Trace Context was standardized, Zipkin's **B3** headers (`X-B3-TraceId`, `X-B3-SpanId`, `X-B3-Sampled`) were the de facto format, and a lot of older infrastructure and service meshes still emit and expect them. If service A injects W3C and service B's instrumentation only reads B3, the context is dropped even though *both* services are "instrumented." The fix is to standardize — modern OpenTelemetry defaults to W3C — or to configure a **composite propagator** that reads and writes both formats during a migration. This is exactly the kind of cross-service mismatch that the debugging series covers as a class; the deep dive on [debugging across service boundaries](/blog/software-development/debugging/debugging-across-service-boundaries) treats the general problem of a bug that hides *between* services rather than inside one, and a broken trace is the observability version of that bug.

#### Worked example: finding and fixing the gap

Here is how you actually find a broken trace, because "some traces are short" is not an alert. The tell is structural: you have traces that *should* be deep (a checkout touches five services) but are showing up only one or two services deep, and you have a population of suspicious *root* spans in a downstream service that should never be a root. Payments should never be the root of a trace — it is always called by something. So the diagnostic query is simply: *show me traces whose root span is in a service that is never an entry point.* In our incident, that query lights up: hundreds of traces rooted at `payments-service: charge` with no parent. That is the smoking gun — payments is being entered without context, which means whoever calls it (the order service) is not propagating.

We fixed it by upgrading the order service's HTTP client to the auto-instrumented one, which injects `traceparent` on every outgoing call. The measurable result, the proof: before the fix, the median *trace depth* for checkout requests was 2 services and only ~30% of payment latency was attributable to a user-facing trace; after the fix, median trace depth was 5 services and ~99% of payment spans connected to their originating checkout. End-to-end visibility went from "blind past the order service" to "complete," and the very next latency regression in payments was localized in seconds instead of being invisible. The cost of the fix was a dependency bump and a deploy. The payoff was that the most expensive part of the request graph stopped being a blind spot.

### The sneakiest gap: context lost inside one process

The broken traces above happen *between* processes, where you can at least point at the network hop. The cruelest gaps happen *inside* one process, because the network looks fine and the headers are present — the context just evaporates somewhere in your own code. The mechanism is that the "current span" lives in thread-local (or async-task-local) storage. It is implicitly attached to the thread that is running and implicitly available to any code that thread calls. The moment your work hops to a *different* thread — a thread pool, an executor, a background task, a callback fired on another thread — the new thread has no current span, so any span you start there is parentless and gets a fresh trace. The request did not leave the process, but the trace broke anyway.

This is maddening to debug because it depends on the *internal threading model* of each service, which the auto-instrumentation cannot always see. The fix is to capture the context on the calling thread and re-attach it on the worker thread by hand:

```python
from concurrent.futures import ThreadPoolExecutor
from opentelemetry import context as otel_context
from opentelemetry import trace

tracer = trace.get_tracer("orders")
pool = ThreadPoolExecutor(max_workers=8)

def enrich_order(order_id):
    # Capture the CURRENT context on the calling thread, before we hand
    # the work to a pool thread that won't inherit it automatically.
    captured = otel_context.get_current()

    def task():
        # Re-attach the captured context inside the worker thread, so the
        # span we start below is a child of the caller's span.
        token = otel_context.attach(captured)
        try:
            with tracer.start_as_current_span("enrich_order") as span:
                span.set_attribute("order.id", order_id)
                return do_enrichment(order_id)
        finally:
            otel_context.detach(token)

    return pool.submit(task)
```

The `otel_context.get_current()` on the calling thread plus `attach`/`detach` inside the worker is the in-process analogue of inject-and-extract across the network. Many libraries provide a `ContextVarsRuntimeContext` or an instrumented executor that does this for you, and async runtimes propagate context through `await` automatically because the task-local storage follows the coroutine — but the second you reach a raw thread pool, a `multiprocessing` boundary, or a third-party callback, you are back to capturing and re-attaching by hand. The rule generalizes: **anywhere control leaves the current call stack — across the network or across a thread — the context must be carried explicitly, or the trace breaks there.** That one sentence is the whole theory of propagation, and almost every broken trace you will ever debug is a violation of it.

## 5. Reading the waterfall: the SRE's core skill

A trace backend (Jaeger, Grafana Tempo, Zipkin) renders a trace as a **waterfall**: every span drawn as a horizontal bar, positioned left-to-right by its start time, sized by its duration, and indented by its depth in the tree. The root span is the full width at the top; its children sit below, each starting where it started and ending where it ended. Learning to *read* this picture quickly is the single highest-leverage observability skill you can build, because the geometry of the waterfall maps directly onto classes of bugs. Let me give you the vocabulary.

![A timeline diagram of a request waterfall showing the gateway accepting at time zero, auth and inventory running in parallel, the payments call as an eight hundred millisecond long pole containing a seven hundred ninety millisecond external card processor span, and the response sent at eight hundred fifteen milliseconds](/imgs/blogs/distributed-tracing-in-practice-3.png)

**The long pole.** One span is far wider than the rest and the root's total duration tracks it almost exactly. This is the 90%-of-the-latency-in-one-hop pattern — our 2:14 incident. When you see a long pole, you are done localizing: that span owns the problem. Click into it, read its attributes, and you usually have your answer (in our case, the external card-processor span with a duration that exploded from 5ms to 790ms). The long pole is the most common and most satisfying read, and it is *exactly* the read that metrics cannot give you because the metric had smeared the latency across every service in the chain.

**Serial versus parallel fan-out.** When a service calls three downstreams, the waterfall shows you immediately whether it did so *in parallel* (the three child bars overlap in time, starting together) or *in series* (the bars form a descending staircase, each starting only after the previous finished). A staircase is a performance smell: if those three calls do not depend on each other, running them serially is leaving latency on the table. I have fixed many a slow endpoint by spotting a staircase in the waterfall and turning three sequential calls into one parallel fan-out — and the waterfall *proves* the win, because after the change the three bars line up vertically and the total shrinks to the longest single call instead of the sum. This is one of the few places where a picture directly suggests the fix.

**The retry, shown as N identical spans.** A retry loop renders as several spans with the *same operation name* to the *same downstream*, back to back, the first few ending in error and the last (maybe) succeeding. If you see `HTTP POST card-processor` three times in a row, each ~timing out, you are looking at a retry storm in miniature — and if every service in the chain is *also* retrying, you have the multiplicative retry amplification that turns one slow dependency into a self-inflicted load spike. The waterfall makes the retries countable. (The principled fix — retries with exponential backoff and jitter, and a budget — is a resilience topic; the waterfall is how you *see* that your retries are misbehaving.)

**The deepest error span.** When a request fails, the user sees a single 500 from the gateway, but the *origin* of the failure is almost always far down the tree. A trace propagates errors upward: the card processor times out, payments marks its span ERROR and re-raises, order marks its span ERROR, the gateway returns 500. All four spans are red. The one that *matters* is the **deepest** red span — the leaf of the error, the place the failure actually started. Reading "find the deepest ERROR span" off the waterfall takes you straight to the root cause instead of the symptom. The gateway's 500 is the symptom; the card processor's timeout is the disease.

**The missing span.** Sometimes the bug is a span that *isn't there.* You expect a checkout to call the fraud-check service, but the waterfall has no fraud-check span — the service was never called, perhaps because a feature flag short-circuited it or a conditional branch was wrong. A missing span is a hard read (you have to know what *should* be there) but a powerful one: it catches "we silently skipped a step" bugs that produce no error and no latency anomaly, only a wrong result.

![A tree diagram mapping each waterfall shape to a root cause, splitting latency shapes into a single long span or a serial staircase and error shapes into the deepest error span or a missing span that reveals a service was never called](/imgs/blogs/distributed-tracing-in-practice-8.png)

That tree is the cheat sheet. Slow or failed request at the root; latency-shaped problems split into the long pole and the serial staircase; error-shaped problems split into the deepest error span and the missing span. When you open a trace, classify the shape first and the diagnosis follows. This is the mental discipline that makes the difference between staring at a waterfall and *reading* it.

### The practice: a manual span that makes the long pole legible

A waterfall is only as informative as the spans in it. If the payments service's call to the card processor is *not* its own span — if it is buried inside an un-instrumented `charge_card` function — then the waterfall shows an 800ms payments span and you cannot tell whether payments was slow or its dependency was slow. The fix is to wrap the outbound call in its own child span, which we did in section 3 with `start_as_current_span("charge_card")` plus the attributes. The rule of thumb: **instrument every network call and every expensive piece of work as its own span,** so the waterfall attributes time to the smallest unit that can be slow. Under-instrumenting hides the long pole inside a parent; over-instrumenting (a span per function call) drowns you in noise and costs storage. The sweet spot is one span per I/O boundary and per significant computation.

## 6. OpenTelemetry: the SDK, the collector, the backends

Everything so far has been vendor-neutral on purpose, because the open standard for all of it is **OpenTelemetry** (OTel) — a CNCF project that defines the API, the SDK, the wire protocol (OTLP), and a collector, so that instrumentation is decoupled from storage. You instrument once against the OpenTelemetry API and you can export to Jaeger, Grafana Tempo, Zipkin, or a commercial backend without touching your code. That decoupling is the whole reason to standardize on it: your instrumentation outlives your choice of backend.

![A stack diagram showing spans flowing from application code with automatic and manual instrumentation through the OpenTelemetry SDK and OTLP exporter into a collector that does tail sampling and batching before reaching a Jaeger or Tempo backend rendered in a Grafana waterfall](/imgs/blogs/distributed-tracing-in-practice-4.png)

The pieces, top to bottom:

- **Instrumentation.** Your app code produces spans two ways. **Auto-instrumentation** libraries wrap common frameworks (your web server, HTTP client, DB driver, queue client) and emit spans with no code changes — for many languages you literally launch the process with an agent or import a one-liner and you get incoming-request and outgoing-call spans for free. **Manual instrumentation** is the `start_as_current_span` calls you add for your own business logic and any boundary the auto libraries miss. The realistic split is "auto for the plumbing, manual for the parts that matter to your domain."
- **The SDK.** The OpenTelemetry SDK in your process collects spans, applies a **span processor** (almost always the **batch span processor**, which buffers spans and ships them in batches to avoid a network call per span), applies any head-sampling decision, and hands finished spans to an **exporter**.
- **The exporter.** Usually the **OTLP exporter**, which speaks the OpenTelemetry Protocol over gRPC or HTTP to the next hop — typically the collector. You *can* export straight to a backend, but routing through the collector is the production pattern.
- **The Collector.** A standalone process (a sidecar, a daemonset, or a central deployment) that receives spans, runs them through **processors** (batching, attribute scrubbing, and — crucially — **tail sampling**, section 7), and **exports** them to one or more backends. The collector is where you centralize policy: change the sampling rule once in the collector instead of redeploying every service.
- **The backend.** **Jaeger** or **Grafana Tempo** (cheap object-storage-backed) or **Zipkin** stores the spans, reassembles traces by trace id, and serves the waterfall UI. Grafana ties traces, metrics, and logs together in one pane.

Here is a minimal but production-shaped SDK setup in Python, showing the batch processor and OTLP export to a local collector:

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "payments-service",
    "service.version": "2026.06.20",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
# Batch, don't export per-span: one network call per batch, not per span.
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True),
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
)
trace.set_tracer_provider(provider)
```

The `service.name` resource attribute is not optional cosmetics — it is how the backend labels each span's owning service in the waterfall, and it is what lets you query "show me traces rooted in a service that is never an entry point" from section 4. Set it, set the version (so you can correlate a regression with a deploy, as we did at 2:14), and set the environment. Three attributes, enormous downstream value.

For the deeper microservices-instrumentation view — how to roll OpenTelemetry across a whole service fleet, the language-by-language auto-instrumentation story, and propagation through a service mesh — the microservices series has a dedicated piece on [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry). This post stays on the SRE operational concerns: reading traces under incident pressure, sampling economics, and correlation.

## 7. Sampling: keeping the interesting traces without going broke

If you record and store *every* span of *every* request, tracing becomes the most expensive thing you run. A busy service handling tens of thousands of requests per second, each producing a dozen spans, generates a firehose of telemetry that dwarfs your metrics and rivals your logs. So you **sample**: you keep some traces and drop the rest. The entire art of sampling is *which* traces to keep, because the naive answer — keep a random 1% — throws away exactly the traces you most need: the rare errors and the rare slow requests. There are two fundamentally different strategies, and the difference between them is the difference between a tracing system that helps in an incident and one that shrugs.

![A before and after diagram contrasting head based sampling that flips a coin at the trace root and discards rare errors against tail based sampling that buffers the whole trace in the collector and keeps it if it errored or was slow plus a small baseline](/imgs/blogs/distributed-tracing-in-practice-5.png)

**Head-based sampling** decides at the *start* of the trace, at the root, before anything has happened. The gateway flips a weighted coin: with probability *p* (say 1%) it sets the sampled flag to 1 and the whole trace is recorded; otherwise it sets the flag to 0 and the trace is dropped. The decision propagates in the trace-flags byte of `traceparent`, so every service downstream honors the same decision consistently — you never get half a trace. Head-based is **cheap** (no buffering — you know immediately whether to keep a span, so you can drop unsampled spans before they ever leave the process) and **simple**. Its fatal flaw: it decides *before it knows whether the trace is interesting.* A request that is about to error or about to be slow is sampled at the same 1% as a boring success. So 99% of your errors are thrown away. In an incident, you go to find a trace of the failing request and there usually isn't one. Head-based sampling optimizes for cost and sacrifices exactly the traces that justify having tracing at all.

**Tail-based sampling** decides at the *end* of the trace, after every span has been collected, when you actually know what happened. The collector **buffers** all the spans of a trace (holding them in memory for a few seconds until the trace looks complete), then applies a policy: **keep it if any span errored, keep it if the total duration exceeded a latency threshold, and keep a small random baseline of the normal ones** (so you still have a representative sample of healthy traffic for baselining). This keeps the *interesting* traces — every error, every slow request — and throws away the boring bulk. The cost is real: the collector must hold whole traces in memory and reassemble them, which needs more collector resources and careful sizing, and it requires that all spans of a trace reach the *same* collector instance (so you route by trace id). But the payoff is decisive: **in an incident, the trace you need is there**, because tail sampling specifically kept it.

The SRE default is unambiguous: **tail-based sampling, with always-keep-errors and always-keep-slow, plus a small baseline of normal traffic, and the sampling rate as your cost lever.** You tune the baseline rate (1%? 0.1%?) to hit your storage budget while the error-and-slow rules guarantee you never lose a debuggable trace. Here is a tail-sampling policy as it appears in an OpenTelemetry Collector configuration:

```yaml
processors:
  tail_sampling:
    # Hold spans this long waiting for the whole trace to arrive.
    decision_wait: 10s
    num_traces: 100000
    expected_new_traces_per_sec: 2000
    policies:
      # 1. ALWAYS keep any trace that contains an error.
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      # 2. ALWAYS keep any trace slower than 500ms end to end.
      - name: keep-slow
        type: latency
        latency:
          threshold_ms: 500
      # 3. Keep a 1% baseline of everything else for healthy-traffic comparison.
      - name: baseline
        type: probabilistic
        probabilistic:
          sampling_percentage: 1

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling, batch]
      exporters: [otlp/tempo]
```

Read the three policies as an OR: a trace is kept if it errored, *or* it was slow, *or* it won the 1% baseline lottery. The `decision_wait: 10s` is the buffering window — long enough for stragglers, short enough to bound memory. The `num_traces` and `expected_new_traces_per_sec` size the in-memory buffer. This single config is the difference between "we have a trace of the outage" and "we sampled it away."

| Dimension | Head-based sampling | Tail-based sampling |
| --- | --- | --- |
| When the decision is made | At trace start (root), before anything happens | At trace end, after all spans are collected |
| Keeps rare errors? | No — they're sampled at the same low rate | Yes — explicit always-keep-errors policy |
| Keeps slow requests? | No, unless they win the coin flip | Yes — explicit latency-threshold policy |
| Cost | Low — no buffering, drop early | Higher — buffer whole traces, route by trace id |
| Where it runs | In the SDK / at the gateway | In the collector |
| Best for | Very high volume where cost dominates and errors are common enough to catch | Almost everyone who wants to debug incidents |

#### Worked example: the cost lever, with numbers

Suppose payments handles 5,000 requests/second, each trace averages 10 spans, and a stored span costs roughly (illustratively) one ten-thousandth of a cent to ingest and retain. At 100% sampling that is 50,000 spans/second — a number that will dominate your observability bill. Drop to a flat 1% head-based and you store 500 spans/second, a 100× cut, but you keep only ~1% of the errors. Now switch to tail-based with always-keep-errors plus a 1% baseline: if your error rate is 0.2% (10 req/s erroring) and your slow rate is 0.3% (15 req/s over 500ms), you store *all* of those (25 req/s of interesting traces × 10 spans = 250 spans/s) *plus* 1% of the remaining ~4,975 healthy req/s (≈50 req/s × 10 = 500 spans/s). Total ≈ 750 spans/second — only 50% more than flat head-based 1%, but now you have captured **100% of your errors and slow requests** instead of 1% of them. That is the deal tail sampling offers: a small multiple on storage buys you every trace that matters. The sampling rate (the baseline percentage and the latency threshold) is the dial you turn to trade cost against fidelity, and you turn it knowing the errors are always safe.

## 8. Sampling is not the only thing that breaks — a stress test

Let me stress-test the tracing setup the way the series demands, because a practice you cannot break under pressure is a practice you do not understand.

**What if two incidents overlap?** Tracing helps more here, not less. Each request still gets its own trace; the two incidents show up as two distinct waterfall shapes (say, one long-pole-on-payments and one error-shaped-on-inventory). You localize each independently by reading the shape. Metrics would blur them together into "everything is slow and erroring"; traces keep them separable because each trace is one request's truth.

**What if the slow hop is an external dependency you don't control?** The waterfall still localizes it — you see the external span (the card processor) as the long pole even though you cannot fix the processor itself. That is still enormously valuable: it tells you the problem is *not yours*, which redirects the entire response (open a vendor ticket, trip a circuit breaker, fail over to a backup provider) instead of having three of your engineers comb your own code for a bug that isn't there. Knowing the problem is external is a finding.

**What if the collector falls over under the tail-sampling memory load?** This is the real operational risk of tail sampling. The mitigations: size `num_traces` and `decision_wait` conservatively, run the collector with horizontal scaling and a **trace-id-aware load balancer** in front (so all spans of a trace land on the same collector — the `loadbalancing` exporter does exactly this in a two-tier collector setup), and set memory limits with a `memory_limiter` processor so the collector sheds load gracefully instead of OOM-ing. If the collector is at risk, you degrade to head-based as a fallback rather than lose all telemetry. The collector is now a production dependency; treat it like one, with its own SLO and its own alerts.

**What if a service is un-instrumented entirely?** Then it appears in the trace as a *gap* — the parent span's duration includes time spent in the un-instrumented service, but there is no child span for it, so the waterfall has an unexplained stretch of "missing" time under a parent. This is the cousin of the broken trace: not a severed tree, but a hole in it. The read is "there is 200ms here I cannot account for," and the fix is to instrument that service. Recognizing the *unaccounted-for time* under a parent span is itself a diagnostic skill.

**What if the trace id is in the headers but the clocks are skewed?** Span timestamps come from each service's own clock, and if those clocks disagree, the waterfall can show a child span starting *before* its parent or ending *after* it — physically impossible, an artifact of clock skew. Good backends correct for this by trusting the parent-child structure over raw timestamps, but the operational lesson is to run NTP everywhere and not panic when a waterfall shows a small negative offset; it is a clock problem, not a causality problem. The practical tell is the *magnitude*: a few milliseconds of overlap is almost always skew and safe to ignore; hundreds of milliseconds of a child apparently preceding its parent points at a genuinely misconfigured clock that will also be corrupting your logs' timestamps and your incident timelines, so it is worth fixing for reasons well beyond tracing.

**What if a span is enormous because the operation legitimately took minutes?** Long-running operations — a batch import, a video transcode, a slow report — produce a single span that dwarfs everything and can blow your tail-sampling latency threshold so that *every* trace touching that operation gets kept, swamping storage. The fix is to break the long operation into child spans for its phases (so the waterfall shows *where* the minutes went, not just that it took minutes) and to set the latency keep-threshold per operation rather than globally, so a five-minute batch job is not judged by the same 500ms bar as a checkout. A single undifferentiated multi-minute span is both a sampling problem and a missed diagnostic opportunity; subdividing it fixes both at once.

## 9. Correlation: tracing is a hub, not an island

The final piece — and the one that turns tracing from a nice tool into the center of your observability — is **correlation**: tying a trace to its metrics and its logs so you can move between the three pillars without losing your place. Three links matter.

**Trace to logs.** Every log line a service emits while handling a request should include the **trace id** (and ideally the span id) as a structured field. Then, from any span in the waterfall, you can pivot to "show me every log line for this trace id" and read the exact exception, the SQL, the values — the *explanation* that the span's attributes only hint at. The mechanism is to pull the active trace id out of the context and inject it into your structured logger:

```python
import logging
from opentelemetry import trace

def log_with_trace(message, **fields):
    span = trace.get_current_span()
    ctx = span.get_span_context()
    # Render the ids as the conventional hex strings.
    trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else None
    span_id = format(ctx.span_id, "016x") if ctx.span_id else None
    logging.info(message, extra={
        "trace_id": trace_id,
        "span_id": span_id,
        **fields,
    })
```

With `trace_id` in every log line and a log backend like Loki, the trace UI links straight to the logs and back. The trace tells you *where* and *how long*; the logs tell you *exactly what*. That is the trace-to-log loop, and it is the most useful correlation you will build.

**Metric to trace, via exemplars.** An **exemplar** is a single example data point attached to a metric — specifically, a trace id stapled onto a bucket of a latency histogram. When your p99 latency metric spikes, the exemplar lets you click the spike on the Grafana panel and jump directly to a *representative trace* that landed in that high-latency bucket. This closes the detect-then-localize loop with one click: the SLO burn alert fires on the metric, you click the exemplar, and you are looking at the waterfall of an actual slow request. Prometheus and OpenTelemetry both support exemplars; enabling them on your latency histograms is the single highest-leverage correlation feature, because it connects the cheap thing that *alerts* (the metric) to the expensive thing that *explains* (the trace) at the exact moment you need the connection.

**High cardinality belongs on spans.** A point that trips up engineers coming from a metrics background: on metrics, high-cardinality labels are forbidden — every distinct value of `user.id` would create a new time series and blow up your storage. On **spans, high cardinality is fine and encouraged.** Put the user id, the order id, the request id, the full URL, the customer tier — whatever you might want to filter or group by during an incident — directly on the span as an attribute. Spans are individual records, not aggregates, so there is no cardinality explosion; you are just adding fields to a row. This is *why* tracing answers questions metrics cannot: "show me the slow traces *for enterprise-tier customers in us-west*" is a trivial span query and an impossible metric query. The cardinality you cannot afford on a metric, you put on a span for free.

#### Worked example: the full incident, end to end

Tie it all together with the 2:14 incident, now with the complete toolchain. (1) **Detect**: the multi-window SLO burn-rate alert on checkout latency fires — the metric saw p99 cross threshold. (2) **Correlate to a trace**: I click the exemplar on the p99 panel and land on a real 815ms checkout trace. (3) **Localize**: the waterfall shows the long pole — the 790ms external card-processor span inside the 800ms payments span; everything else is fast. (4) **Explain**: I pivot from that span to its logs via the trace id and see "establishing new TLS connection to processor" on *every* call instead of once — the connection pool is not being reused. (5) **Confirm the cause**: the payments span's `service.version` attribute matches the 2:09 deploy, and the changelog shows a connection-pool setting changed. (6) **Fix and verify**: roll back the pool setting, watch the next trace show a 5ms processor span, and watch the p99 metric drop back to 250ms. Total time from alert to confirmed root cause: a few minutes, most of it spent reading rather than guessing. Without tracing, step 3 alone — *which service* — was the forty-minute afternoon. The whole chain (detect on metrics → correlate via exemplar → localize on the waterfall → explain via trace-linked logs) is the observability loop working as designed, and tracing is the hub that connects the cheap detector to the precise explanation.

## 10. War story: how tracing was born and what it teaches

The canonical origin is **Google's Dapper**, the internal tracing system described in a 2010 paper that essentially defined the field — trace id and span propagation through every RPC, low-overhead sampling, and a backend that reassembles the tree. Dapper's hard-won lessons are the ones we have been living: that propagation must be *built into the RPC framework* so it is automatic and ubiquitous (a tracing system with gaps is barely a tracing system), that *sampling is mandatory* at scale (you cannot keep everything), and that the value is overwhelmingly in *latency analysis* — finding the long pole. Twitter's **Zipkin** and Uber's **Jaeger** were the open-source children of that idea, and the format wars between them (B3 versus everything else) are why the W3C eventually standardized `traceparent` so that a service from one vendor's ecosystem could propagate context to another's. OpenTelemetry is the convergence: one API, one protocol, the format wars settled.

The operational lesson from the lineage is the one this whole post orbits: **the system is only as good as its weakest propagation boundary.** Dapper worked because Google could mandate that *every* RPC carried the context, fleet-wide, with no opt-out. Most companies cannot mandate that overnight; they have a polyglot fleet, some legacy services, a service mesh speaking one format and an app speaking another, and a few queues nobody remembered to instrument. So real-world tracing is a *coverage* problem more than a technology problem. The teams that get value from tracing are not the ones with the fanciest backend; they are the ones who treated context propagation as a fleet-wide invariant — who added the trace headers to the proxy allowlist, who instrumented the queues, who standardized on W3C, who wrote the "payments should never be a root span" detector to *find* the gaps. Coverage is the game. The waterfall is only as honest as the spans it contains.

A second, smaller war story worth internalizing: a **retry-storm cascade** read from traces. A downstream service slowed down; the service above it, configured to retry three times with no backoff, turned each slow request into three; the service above *that* also retried three times, turning three into nine; and so on up the stack, so one slow dependency became an exponential pile of duplicate requests that buried it completely. On the metrics it looked like a sudden, inexplicable load spike with no traffic increase at the front door. On a *trace*, it was obvious in one waterfall: the same downstream call appearing 3, then 9, then 27 times, the retry amplification drawn as a fan of identical spans. The metric said "load spike from nowhere"; the trace said "we are doing this to ourselves, here is the multiplication." That is the difference between detecting a symptom and seeing a mechanism.

The arithmetic is worth making explicit because it explains *why* the trace's fan of identical spans is so dangerous and so diagnostic. If each of $L$ layers in the call stack retries up to $r$ times, the load multiplication factor at the bottom is $r^{L}$. With three layers each retrying three times, the deepest service sees up to $3^{3} = 27$ times its normal request volume — from a *single* user request at the top. The slowdown that triggered the retries gets worse under the extra load, which triggers more retries, which is the positive-feedback loop that turns a blip into an outage. The fix is principled (retries with exponential backoff and jitter, plus a per-request retry *budget* so the total amplification is capped, not multiplied) and lives in the resilience layer, not here. But tracing is how you *see* the $r^{L}$ in the first place: the count of identical sibling spans in the waterfall is the per-layer $r$, and the depth at which they stack is $L$. The number that the metric hid as an anonymous load spike is right there, countable, in the tree.

## 11. Span names, attributes, and the discipline that keeps traces searchable

A trace you cannot search is a trace you will not use, and searchability comes down to two unglamorous disciplines: **span names** and **attribute conventions**. Get them right and "show me the slow checkout traces for enterprise customers in us-west since the 2:09 deploy" is a one-line query. Get them wrong and you have a backend full of traces you cannot filter, which in an incident is nearly as useless as having no traces at all.

**Span names must be low cardinality.** The name is the operation, not the instance: `GET /orders/{id}`, not `GET /orders/8831`. If you bake the order id into the *name*, every request becomes a distinct operation name, your backend's "group by operation" view shatters into millions of one-off entries, and you can no longer ask "what is the p99 of the get-order operation?" The variable parts — the id, the user, the query — go in **attributes**, where high cardinality is welcome; the name stays a stable template. This is the single most common instrumentation mistake, and the auto-instrumentation gets it right (it uses the *route template*, not the resolved path) precisely because the library authors learned this lesson the hard way.

**Attributes should follow the OpenTelemetry semantic conventions.** OTel publishes a standard vocabulary of attribute keys — `http.request.method`, `http.response.status_code`, `db.system`, `db.statement`, `messaging.system`, `server.address`, `url.path` — so that a span from any language or library is queryable with the *same* key. The payoff is concrete: when you can rely on `http.response.status_code` meaning the same thing across your whole fleet, you can write a single tail-sampling policy or a single dashboard query that works everywhere. When every team invents its own key (`status`, `statusCode`, `http_status`, `code`), every query becomes a per-service special case. Adopt the conventions; do not reinvent them. Here is a manual span that follows them, plus a couple of well-chosen high-cardinality attributes for incident filtering:

```python
with tracer.start_as_current_span("GET /orders/{id}") as span:
    # Semantic-convention keys: queryable fleet-wide.
    span.set_attribute("http.request.method", "GET")
    span.set_attribute("url.path", f"/orders/{order_id}")
    span.set_attribute("server.address", "orders.internal")
    # High-cardinality business attributes: cheap on spans, gold in an incident.
    span.set_attribute("order.id", order_id)
    span.set_attribute("customer.tier", tier)        # "enterprise" / "free"
    span.set_attribute("deployment.version", BUILD)  # ties a regression to a deploy
    result = fetch_order(order_id)
    span.set_attribute("http.response.status_code", 200)
```

Notice `customer.tier` and `deployment.version`. Neither could live on a metric without exploding cardinality, but on a span they are free, and they are exactly the dimensions you slice by in an incident: *is this only hurting enterprise customers? did it start with this deploy?* The discipline is simple to state and easy to skip under deadline pressure: **stable low-cardinality names, semantic-convention keys for the plumbing, and generous high-cardinality business attributes for the questions you will ask at 2am.**

#### Worked example: an attribute that closed an incident in one query

Months after the connection-pool incident, a different alert fired: elevated checkout errors, but only a trickle — maybe 2% of requests, not enough to blow the SLO, but climbing. The metric could tell us the error rate and nothing about *who*. Because every checkout span carried `customer.tier` and `deployment.version`, the query "group the erroring traces by `customer.tier`" returned the answer instantly: 100% of the errors were `enterprise` tier, and grouping by `deployment.version` showed they all came from a canary that had rolled out to 5% of traffic — and that 5% happened to include a large enterprise account on a feature flag. The errors were a null-pointer in a code path only enterprise contracts hit. Two `group by` queries on span attributes — `customer.tier` then `deployment.version` — turned "2% of checkouts are failing for some reason" into "the canary build breaks the enterprise contract path; roll it back" in under a minute. That is the dividend of high-cardinality attributes: the question you did not know you would need to ask is answerable because you put the field on the span when you had no specific reason to, except the general principle that span attributes are cheap and incidents are expensive.

### Rolling instrumentation across a real fleet without a big-bang

One last practical concern, because "instrument everything" is a slogan, not a plan. You will not instrument a forty-service fleet in a sprint, and you should not try. The pragmatic rollout order, by leverage:

1. **The edge and the critical path first.** Instrument the gateway and the two or three services on your highest-value request path (checkout, login, the thing that pages you). One complete trace through the money path is worth more than partial coverage of forty internal services.
2. **Auto-instrument the plumbing fleet-wide.** Turn on auto-instrumentation everywhere it is one line — it costs almost nothing and immediately gives you incoming-request and outgoing-call spans, which is 80% of the value. This is also how you *find* the broken-trace gaps: the moment most services emit spans, the un-instrumented ones show up as holes.
3. **Standardize the propagator and the collector before you scale.** Pick W3C, deploy the collector with tail sampling, and make those the paved road *before* every team rolls their own — otherwise you spend the next year reconciling B3-versus-W3C gaps and five different sampling configs.
4. **Add manual spans where incidents actually happen.** Let the postmortems drive it: every time an incident's localization was slow because a span was missing, add that span. The instrumentation that pays off is the instrumentation aimed at the questions you have actually had to answer under pressure.

This staged order means you get value in week one (the critical path is traced) and you never block on a fleet-wide migration. It also keeps the cost honest: you are not paying to store spans for services nobody has ever had to debug. The architecture-time question of *where service boundaries should be* so that traces are meaningful is a system-design concern; the operational question of *which boundaries to instrument first* is answered by following the money and the pages.

## How to reach for this (and when not to)

Tracing is powerful and it is not free, so be deliberate about where it earns its keep.

**Reach for tracing when** you run more than a couple of services and a request crosses service boundaries; when your incidents routinely involve the question "*which* service is slow?"; when you have latency that the metrics flag but cannot localize; when you have errors that surface as a generic gateway 500 whose true origin is somewhere downstream; or when you are debugging a fan-out and need to know whether calls ran in series or parallel. Anywhere the hard part of an incident is *localization across services*, tracing is the highest-leverage instrument you can add, and the always-keep-errors tail-sampling default means it is there exactly when you need it.

**Do not bother (or do less) when**: you run a single service or a monolith with no service-to-service calls — there is no cross-service causality to capture, and a profiler or in-process timing gives you the same answers more cheaply; the architecture-level "where is my time going inside one process" question is better served by the profiling techniques in the [python-performance series](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) than by distributed tracing. Do not turn on **100% sampling** "to be safe" — you will drown in storage cost for traces nobody ever looks at; use tail sampling with always-keep-errors instead and tune the baseline down. Do not instrument *every function* as a span — one span per I/O boundary and per significant computation is the sweet spot; a span per function call costs storage and buries the long pole in noise. Do not deploy tail sampling without sizing the collector and putting it behind a trace-id-aware load balancer — an OOM-ing collector that loses all your traces is worse than honest head-based sampling. And do not treat tracing as a *replacement* for metrics and logs — it is the third pillar, the localizer, not the detector (metrics) or the explainer (logs); its power is in correlation with the other two.

The honest summary of the trade-off: tracing's cost is instrumentation effort (which OpenTelemetry's auto-instrumentation makes mostly free for plumbing) plus storage (which tail sampling makes affordable) plus a new production dependency (the collector). Its benefit is collapsing the most expensive part of a cross-service incident — figuring out *which component is the problem* — from an afternoon of guessing to seconds of reading. For any system with real service-to-service fan-out, that trade is one of the best in all of observability.

## Key takeaways

1. **A trace is a tree of spans tied together by a shared trace id.** Each span is one timed operation in one service; the parent span id links them; the tree reassembles in the backend. It is the only telemetry that carries cross-service causality — the path of one request through N services.
2. **Context propagation is the whole game.** Inject the W3C `traceparent` header on the way out, extract it on the way in, at *every* boundary — including async and queue boundaries. One hop that drops the header severs the trace and hides everything downstream. Standardize on W3C and treat propagation coverage as a fleet-wide invariant.
3. **Learn to read the waterfall.** The long pole is 90% of latency in one hop; a serial staircase is a parallelization opportunity; repeated identical spans are retries; the deepest red span is the origin of a 500; a missing span means a service was never called. The shape points at the bug class before you read a log.
4. **Sample tail-based, always keep errors and slow traces.** Head-based is cheap but throws away the rare traces you most need; tail-based buffers the whole trace and keeps every error and slow request plus a small baseline. The baseline rate is your cost lever; the error policy guarantees the trace is there in an incident.
5. **OpenTelemetry decouples instrumentation from storage.** Auto-instrument the plumbing, manually instrument your domain logic, route through a collector (where sampling policy lives), and export to Jaeger or Tempo. Your instrumentation outlives your backend choice.
6. **Correlate the three pillars.** Put the trace id in every log line (trace-to-log), enable exemplars so a metric spike links to a representative trace (metric-to-trace), and pile high-cardinality attributes onto spans (which, unlike metrics, tolerate them). Tracing is the hub of observability, not an island.
7. **The measured payoff is localization time.** Cross-service latency and error incidents go from "which service is it?" guessing — a median of tens of minutes — to seconds of reading the waterfall. Tracing attacks the largest term in cross-service MTTR directly.

## Further reading

- **The intro map for this series**: [reliability is a feature — the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — where observability fits in the define → measure → budget → respond → learn loop.
- **The pillar-selection companion**: [metrics, logs, and traces — when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which) — the three pillars and the question each answers.
- **The sibling on log economics**: [logging at scale without going broke](/blog/software-development/site-reliability-engineering/logging-at-scale-without-going-broke) — structured logs, sampling, and trace-id correlation from the logs' side.
- **Debugging the gaps between services**: [debugging across service boundaries](/blog/software-development/debugging/debugging-across-service-boundaries) — the general problem of a bug that hides *between* services, of which the broken trace is the observability case.
- **The microservices instrumentation deep dive**: [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) — rolling OTel across a whole service fleet.
- **The architecture-time view**: [observability — metrics, logs, traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — designing instrumentation in from the start.
- **Canonical sources**: the Google *Dapper* paper (Sigelman et al., 2010), the [W3C Trace Context specification](https://www.w3.org/TR/trace-context/), the [OpenTelemetry documentation](https://opentelemetry.io/docs/), and the [Grafana Tempo](https://grafana.com/docs/tempo/) and [Jaeger](https://www.jaegertracing.io/docs/) docs for the backends. For sampling, the OpenTelemetry Collector's `tailsamplingprocessor` README is the practical reference.

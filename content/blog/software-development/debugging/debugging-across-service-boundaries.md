---
title: "Debugging Across Service Boundaries: Find the Service That Owns the Bug"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to localize a failure to one service out of six from a distributed trace, before you ever open a repository, so you stop reading the wrong codebase."
tags:
  [
    "debugging",
    "software-engineering",
    "distributed-tracing",
    "opentelemetry",
    "microservices",
    "observability",
    "correlation-ids",
    "localization",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/debugging-across-service-boundaries-1.png"
---

A user files a ticket: "Checkout returns a 500. Sometimes." You open the gateway logs and there it is — `HTTP 500, POST /checkout` — with a timestamp and a request id and absolutely nothing else useful. There is no stack trace, because the gateway did not throw. It merely forwarded a request into a fleet of six services and got back an error it does not understand. The actual failure happened somewhere downstream, in a process that lives on a different host, in a different language, behind two more hops, and the only thing that survived the journey back to you is the corpse of an HTTP status code.

This is the defining problem of debugging across service boundaries, and it breaks every instinct you built up debugging a single process. In a monolith, an exception carries a stack trace, and the stack trace *is* the localization: frame by frame, it names the function that threw, the function that called it, and the path all the way back to `main`. The bug announces its own address. In a distributed system that contract is gone. No single stack trace spans services. Each hop is a separate process with its own stack that unwinds and vanishes the moment it returns a response. A 500 the user sees at hop one can originate five hops deep, and the gateway's timeout is frequently not the bug at all — it is the *symptom* of a slow database two services further down that nobody at the gateway can see. Causality is hidden because there is no shared memory and no shared stack to walk.

![A diagram of one user request fanning through six services where the gateway returns a 500 but the real error lives five hops down at the auth service](/imgs/blogs/debugging-across-service-boundaries-1.png)

So the first job is not *fixing*. It is not even *reading code*. The first job is **localization**: deciding which of the six services owns the bug, before you open a single repository. Read the wrong service's code and you can burn a day confirming that it is innocent. The whole discipline of this post is learning to point at the guilty service from a distance — from the trace, the logs, and the access logs of the mesh — and only *then* descending into that one service to read its code and reproduce it in isolation. We will lean hard on distributed tracing as the primary instrument, fall back to correlation IDs and `grep` when full tracing is not deployed, learn the five waterfall shapes that each name a specific failure class, and walk two real investigations end to end: a user-facing 500 that turned out to be an auth 403 mis-mapped three hops down, and a p99 latency complaint that was a database N+1 hiding in the orders service. By the end you will turn "something in the cluster is broken" into "service C, line 88, here is the reproducer" — which is exactly the **observe → reproduce → hypothesize → bisect → fix → prevent** loop from the [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), applied across processes instead of within one.

## Why a cross-service bug is even possible: the mechanism

Before any tool helps you, internalize *why* this class of bug exists, because the mechanism dictates the method. When you call a function inside one process, the runtime maintains a call stack: a contiguous region of memory where each call pushes a frame holding its return address and locals. When something throws, the runtime walks that stack from the top, and every frame it passes becomes a line in your stack trace. The stack trace is free localization precisely because the stack is *physically there in memory* and the language runtime has the metadata to symbolize each return address back to a file and line. The caller and callee share an address space, share a thread, and share a single linear notion of "what happened before what."

Cross a service boundary and every one of those guarantees evaporates. Service A calls service B over HTTP or gRPC. From A's perspective it made a network call and is now blocked on a socket `read()`, waiting for bytes. A's stack at that moment shows the HTTP client library, not anything inside B. B, on a completely different machine, receives the bytes, builds its *own* fresh call stack starting from its request handler, does its work, possibly calls C, and eventually writes a response back onto the socket. The instant B writes that response, its stack unwinds and is gone — the frames are popped, the locals are reclaimed, and the only evidence that B ran is whatever B chose to *emit* on the way: a log line, a metric, a span. There is no shared memory between A and B, so there is no way for A's runtime to reach into B and ask "what was your stack when you failed?" The causal link — "A's slowness was caused by B waiting on C waiting on D's database" — exists in reality but is recorded *nowhere* unless you deliberately record it.

That is the mechanism in one sentence: **a distributed request has a causal structure that is real but invisible, because no process can observe another process's stack, and each stack is destroyed when its response is sent.** Distributed tracing is the engineering response to exactly this. It reconstructs the missing causal structure by having every service emit a small record (a *span*) for the work it did, tagged with a shared *trace id* that travels with the request from hop to hop, plus a *parent span id* that records who called whom. Stitch those spans together by trace id and parent id and you have rebuilt, after the fact, the cross-process call tree that the runtime could not give you for free. The trace is the stack trace that distributed systems do not have — assembled from breadcrumbs rather than read from memory.

This also explains the most painful failure mode of all: the *broken trace*. If even one service in the chain forgets to forward the trace context to its downstream calls, the breadcrumb trail snaps. Everything below that service still runs and still fails, but its spans get a *new, unrelated* trace id (or none), so they no longer attach to your request's tree. You get a trace that looks suspiciously short and ends right above the service that actually broke — a blind spot located, by cruel coincidence, at the exact hop you most need to see. Understanding that the trace is a manually-propagated reconstruction, not a free read of memory, is what lets you recognize a missing span as a *propagation bug* rather than as evidence the request stopped there.

## Localize first, read code last: the inverted workflow

Here is the single most important habit shift, and it is genuinely the opposite of how you debug a monolith. In a monolith you read the stack trace and jump straight to the throwing line — observe and localize happen in the same instant. Across services, **you must localize from the trace before you open any service's code at all.** The trace tells you *which* of six repositories to clone, which service's logs to tail, which team to page. Reading code first is a category error: you are guessing at the suspect, and with six services your prior is wrong five times out of six.

![A vertical flow showing the cross-service debugging order from symptom to opening the trace to finding the deepest error span to reading that one service's logs and reproducing it in isolation](/imgs/blogs/debugging-across-service-boundaries-2.png)

The workflow has a fixed shape. **Symptom** — a 500 at the gateway, or a p99 latency alert — gives you a trace id (or a correlation id, or a timestamp window to search). **Open the trace** — pull up that one trace id in your tracing UI. **Find the deepest error or slow span** — not the top-most error, the *deepest* one, because errors bubble *up* and the one at the bottom of the tree is the origin while the rest are echoes. **Now read that service's logs** — only now, having named the suspect, do you tail logs and open code, and you open *one* repository, not six. **Reproduce it in isolation** — pull the offending service up alone, replay the request that triggered the span, and confirm you can make the bug happen on demand. **Root cause** — you have localized to one service out of six and turned a fleet-wide mystery into an ordinary single-process bug, which is the kind you already know how to kill.

Notice what this buys you. The expensive, slow, high-variance part of cross-service debugging is the *search* over services. Tracing collapses that search from "read six codebases hoping to get lucky" to "look at one waterfall, point at one span." Everything downstream of the point — reading code, writing a reproducer, fixing — is the ordinary single-process debugging you do every day, which we cover in [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) and [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope). The cross-service skill is purely the localization step. Get good at that step and the rest is muscle memory.

There is a discipline to resisting the urge to read code early. When the ticket says "checkout 500s," the engineer who owns the checkout BFF feels a pull to open the BFF code immediately — it has their name on it. Resist. The BFF is hop one of six; the odds it contains the bug are no better than any other service, and probably worse, because BFFs mostly forward. Open the trace, find the deepest red span, and let the *data* tell you where to look. I have watched senior engineers lose two hours reading the gateway and the BFF line by line for a bug that lived in the auth service, simply because they started with the code they knew instead of the trace they had.

## The trace waterfall: every span on one clock

The single artifact that makes localization possible is the **trace waterfall**: a view that lays every span of one request onto a single shared horizontal time axis. Each span is a bar; its left edge is when that operation started, its width is how long it took, and its vertical indentation shows the parent-child nesting (who called whom). Once you have all six services' spans on one clock, two things that were invisible in the logs become obvious by *eye*: the long pole (the one bar that is wider than all the others, i.e. the slow hop) and the deepest red bar (the first span that errored, i.e. the origin of the failure).

![A left-to-right timeline of a trace waterfall where the gateway request enters at zero, fans out through services, and the database span at service F is the long pole returning at over a second](/imgs/blogs/debugging-across-service-boundaries-3.png)

Walk the waterfall left to right. At `0ms` the gateway receives the request and opens its span. At `8ms` service A (the BFF) calls service B. Around `20ms` service C fans out to both D and E in parallel — you see two child bars starting at nearly the same x-coordinate, which is the visual signature of a fan-out. At `30ms` service F begins a database query. And then the picture tells the whole story: F's span does not return until `1130ms`. That one bar is wider than the entire rest of the trace combined. It is the long pole, and because the gateway has a one-second timeout, the gateway's own span ends at `1140ms` with a timeout error. The user got a 500; the *cause* is a database query in F that took 1.1 seconds. Without the waterfall you would have six services each logging "I was slow" and no way to tell which slowness caused which. With the waterfall, the answer is a bar you can point at across the room.

The clock-alignment is the whole trick, and it is not free — it depends on every service stamping its spans with timestamps that are at least roughly comparable. In practice spans carry both a start time and a duration measured by the *emitting* service's monotonic clock, so within a single service the durations are exact; across services there can be small skew from imperfectly-synchronized wall clocks, but tracing backends correct for it using the parent-child causal ordering (a child cannot start before its parent's span, so the backend nudges skewed timestamps into a consistent tree). You almost never need to worry about skew at the resolution that matters — a 1100ms long pole is unmistakable even with 20ms of clock skew. The cases where skew bites are sub-millisecond races, and for those you reach for the techniques in [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering), not the waterfall.

To read a waterfall fluently, train your eye on three features in order. First, **width** — scan for the widest bar; that is your long pole and your latency suspect. Second, **color** — scan for red (errored) bars; the *deepest* (most indented, latest in the call tree) red bar is your error origin. Third, **gaps** — look for a parent bar that is much wider than the sum of its visible children, because that gap is *unaccounted time* and usually means a missing span: a child the trace failed to capture. Width, color, gap. Master those three reflexes and you can localize most cross-service failures in under a minute.

## The propagation contract: how the trace id survives six hops

The waterfall only exists because the trace id physically travels with the request through every hop, and that propagation is a *contract* that each service must honor. The modern standard is **W3C Trace Context**, which defines an HTTP header named `traceparent` carrying four fields packed into one string: a version, the 16-byte trace id (shared by every span in the request), the 8-byte id of the *current* span (which becomes the parent id of the next hop), and a one-byte flags field (notably the "sampled" bit). It looks like `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`. The rule each service must follow is simple to state and easy to get wrong: on the way *in*, read `traceparent` from the incoming request to learn your parent; on the way *out*, write a fresh `traceparent` (same trace id, your span id as the new parent) onto every downstream call. Forward the context, always, on every egress.

OpenTelemetry (OTel) — the vendor-neutral standard for generating telemetry — automates this if you let it, and seeing the real code dissolves the mystery. Here is a Python service that participates in a trace correctly: it extracts incoming context, creates a server span, and *injects* context into its outgoing call. This is the part that, when omitted, breaks the trace.

```python
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import SpanKind
import requests

tracer = trace.get_tracer("orders-service")

def handle_request(incoming_headers):
    # On the way IN: rebuild the parent context from the caller's traceparent.
    ctx = extract(incoming_headers)
    with tracer.start_as_current_span(
        "orders.handle_request", context=ctx, kind=SpanKind.SERVER
    ) as span:
        span.set_attribute("http.route", "/checkout")
        result = call_payments(amount=4200)  # cents, not dollars
        return result

def call_payments(amount):
    with tracer.start_as_current_span(
        "orders.call_payments", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("payment.amount_cents", amount)
        downstream_headers = {}
        # On the way OUT: inject the CURRENT context so payments joins our trace.
        inject(downstream_headers)
        # If you forget this inject(), payments starts a NEW trace -> broken trace.
        resp = requests.post(
            "http://payments.internal/charge",
            json={"amount": amount},
            headers=downstream_headers,
            timeout=2.0,
        )
        if resp.status_code >= 500:
            span.set_status(trace.StatusCode.ERROR, "payments 5xx")
        return resp
```

The two lines that matter most are `extract(incoming_headers)` and `inject(downstream_headers)`. The first connects you to your caller; the second connects your callee to you. Auto-instrumentation libraries (`opentelemetry-instrumentation-requests`, `-flask`, `-grpc`, and their equivalents in Go, Java, Node, and Ruby) do both for you by monkey-patching the HTTP client and server, which is why teams that adopt OTel auto-instrumentation rarely hit propagation bugs. The teams that *do* hit them are the ones with a hand-rolled HTTP client, a message queue hop that does not carry headers (the trace context has to ride in the message metadata — see [tracing across the queue boundary](/blog/software-development/message-queue/observability-tracing-across-the-queue-boundary) if your break is at a queue), or a service written before tracing was adopted that nobody re-instrumented.

When that contract is broken at one service, the consequence is severe and specific.

![A before and after comparison where service C drops the traceparent header and orphans the downstream services into a blind spot, versus forwarding the header so all six spans are visible](/imgs/blogs/debugging-across-service-boundaries-4.png)

In the broken case, A and B propagate correctly, so their spans share a trace id. Then service C handles the request but fails to inject context into its calls to D, E, and F. Those three services still run, still do work, and still might be the ones that fail — but their spans get a brand-new trace id, so they form a *separate, orphaned* trace tree that does not show up when you pull up the user's trace. Your trace ends at C and looks complete-but-short. The blind spot is precisely below C, which is exactly the region you need to see if the real bug is in D, E, or F. In the fixed case, C forwards the header, and the waterfall shows all six of six spans — the whole tree is visible and the deepest error or longest span falls right out. The lesson: when a trace looks suspiciously short and ends at a service that "shouldn't" be the leaf, your first hypothesis is not "the request stopped there" — it is "*that service dropped the context*," and you go check its egress instrumentation.

## When you have no tracing: correlation IDs and grep

Plenty of systems do not have full distributed tracing yet. You can still localize — you fall back to **correlation IDs**, which are the poor cousin of trace context and surprisingly effective. A correlation id is a single unique string (a UUID generated at the edge) that you attach to the incoming request and then (a) log on every log line in every service and (b) forward as a header on every downstream call. It carries far less than a full trace — no span tree, no per-hop durations, no parent-child structure — but it does the one thing you need most: it lets you `grep` the same id across every service's logs and assemble, by timestamp, a rough timeline of which services touched the request and what each one said.

The implementation is a few lines. Generate-or-propagate at the edge, stash it in a context variable, log it on every line, forward it on every call:

```python
import contextvars, uuid, logging

correlation_id = contextvars.ContextVar("correlation_id", default="-")

def middleware(request, call_next):
    # Reuse the caller's id if present, else mint one at the edge.
    cid = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    correlation_id.set(cid)
    response = call_next(request)
    response.headers["x-correlation-id"] = cid
    return response

# A logging filter injects the id into every record automatically.
class CorrelationFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id.get()
        return True

# Outbound: forward the same id on every downstream call.
def call_downstream(url, payload):
    headers = {"x-correlation-id": correlation_id.get()}
    return requests.post(url, json=payload, headers=headers, timeout=2.0)
```

With every service logging structured JSON that includes `correlation_id`, localization becomes a log query. If your logs are centralized (Loki, Elasticsearch, CloudWatch), you query the id across all services and sort by timestamp. If you only have files on hosts, you `grep` and merge:

```bash
# Across centralized logs, one query gives you the whole request's footprint:
#   {namespace="prod"} | json | correlation_id="3f9a-..." 
# sorted by timestamp, across every service.

# On raw host logs, grep each service and merge by time:
for svc in gateway bff orders payments auth inventory; do
  grep '"correlation_id":"3f9a8c12-7b4e-4a1d-9e22-0c6f5d8a1b33"' \
    /var/log/$svc/app.log | sed "s/^/[$svc] /"
done | sort -k2   # sort by the timestamp field so the timeline is chronological
```

The merged output reads like a hand-assembled trace: `[gateway] received POST /checkout`, `[bff] calling orders`, `[orders] calling payments`, `[auth] returning 403 forbidden`, `[orders] mapping downstream error to 500`. The 403 line, deep in the timeline and earlier than the 500, is your origin — the same "deepest error is the cause" logic as a waterfall, reconstructed by hand. It is slower and grainier than a real trace, and it gives you no durations to find a long pole easily, but it gets you to one suspect service. The honest comparison of the two approaches:

| Capability | Distributed tracing | Correlation IDs + grep |
| --- | --- | --- |
| Per-hop latency (find the long pole) | Yes, exact span durations | No — only log timestamps, coarse |
| Parent-child call tree | Yes, full waterfall | No — flat timeline, infer nesting by hand |
| Find deepest error span | Yes, by indentation and color | Partial — by timestamp order in the merged log |
| Detect a missing hop (broken trace) | Yes — gap in the tree | Hard — you just see fewer lines |
| Setup cost | OTel SDK + collector + backend | One header + one log field |
| Works with no infra investment | No | Yes — runs on `grep` |
| Sampling loses some requests | Often yes (head sampling) | No — every line is logged |

Correlation IDs are the thing you can ship *this afternoon* with no new infrastructure, and they are the right first move on a legacy fleet that has no OTel. They are also a useful backstop even when you *do* have tracing, because tracing is often sampled — only a fraction of requests get full spans to control cost — while logs are usually complete. When the one request that failed was not sampled, the correlation id in the logs is all you have. For the broader logging discipline this rests on, see [logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument).

## The five localizing patterns: reading shapes in the waterfall

Once you can read a waterfall, you discover that cross-service failures come in a small number of recognizable *shapes*, and the shape itself names the failure class and points at one suspect. This is the heart of the skill: you stop reasoning about six services and start pattern-matching the picture. There are five shapes worth memorizing.

![A matrix mapping five waterfall shapes to the suspect service and the confirming check, covering the slow hop, error bubbling up, retry amplification, fan-out wait, and timeout cascade](/imgs/blogs/debugging-across-service-boundaries-5.png)

**The slow hop (long pole).** One span is dramatically wider than the rest — 90% of the total request time lives in a single bar. The suspect is whatever service owns that span. The confirming check is to read that span's attributes (the DB statement, the URL, the payload size) and that service's logs at that timestamp. This is the most common and the easiest: latency complaints almost always reduce to one long pole, and the waterfall hands it to you. The trap is mistaking a *parent's* width for the cause — a parent span is wide because its child was slow, so always descend to the *deepest* wide span, not the outermost one.

**The error that bubbles up.** You see multiple red error spans stacked up the chain — F errored, E errored, C errored, the gateway errored. It is tempting to blame the top one (the gateway, because that is what the user saw), but errors propagate *upward*: F failed, E saw F fail and failed too, on up to the gateway. The **deepest** error span is the origin; everything above it is an echo bubbling the failure up. The suspect is the service owning the deepest red span. The confirming check is `status == error` at the *leaf* of the error chain. This single rule — deepest error span is the root cause — solves a huge fraction of cross-service 500s, and we will watch it work in the first worked example.

**Retry amplification.** You see *N* spans where you expected one — three or nine bars all hitting the same downstream target, all to the same URL, clustered in time. This is a retry loop: some layer is configured to retry on failure, so one logical call became three (or, if two layers each retry three times, nine). The suspect is the downstream that keeps failing *and* the layer doing the retrying, because retries against an already-struggling dependency amplify load and often make the outage worse. The confirming check is to count the spans hitting the same target and read the retry config. We will dissect this shape in its own section because it is sneaky.

**The fan-out wait.** A parent span fans out to several children in parallel (their bars start at the same x) and the parent's span ends only when the *slowest* child returns — the parent is gated by its slowest dependency. If the parent waited 800ms, scan its children and find the one child that took 800ms; the rest finished in 50ms and are irrelevant. The suspect is the slowest sibling in the fan-out. The confirming check is to sort the children by duration and look at the top one. The trap is averaging — the *mean* child latency is meaningless when the parent waits on the *max*.

**The timeout cascade.** This is the cruelest shape and the one that fools people most. The gateway has a 1-second timeout; you see it error at exactly 1000ms. The naive read is "the gateway timed out, the gateway is the problem." But 1000ms is suspiciously round — it is the *timeout value*, not a natural duration. Look *below* the timeout: there is a child span at 1200ms that never got to finish reporting because its parent gave up at 1000ms. The real long pole is *below the cutoff* and partially hidden, because the parent abandoned the call before the child's span closed cleanly. The suspect is the real long pole hiding under the timeout. The confirming check is to follow the chain *past* the timeout boundary and find the operation that was still running when the parent quit. Always distrust a duration that exactly equals a configured timeout; it is a symptom, never a cause.

| Shape in the waterfall | What it means | Suspect service | Confirming check |
| --- | --- | --- | --- |
| One span is ~90% of total | Slow hop / long pole | Owner of the widest deepest span | Read that span's attributes and logs |
| Stacked red spans up the chain | Error bubbling up | Owner of the *deepest* error span | `status == error` at the leaf |
| N spans to the same target | Retry amplification | Failing downstream + retrying layer | Count retries, read retry config |
| Parent gated by slowest child | Fan-out wait | Slowest sibling in the fan-out | Sort children by duration |
| Duration equals the timeout value | Timeout cascade | Real long pole below the cutoff | Follow the chain past the timeout |

Internalize these five and you will localize most cross-service failures by glance. The picture does the reasoning; you just have to know which shape you are looking at.

## Retry amplification: one call becomes nine spans

Retry amplification deserves its own section because it is both a *localization* signal and a *cause of outages*, and confusing the two will send you fixing the wrong thing. The mechanism: many HTTP clients and service meshes are configured to retry failed requests automatically — a reasonable default, since transient network blips are real. But retries compose multiplicatively through layers. If the BFF retries 3× on failure, and the orders service it calls *also* retries 3× on its own downstream, then one user request can become 9 calls to the deepest dependency. In the trace, this shows up as a tell-tale cluster: nine spans, same target, same time window, most of them red.

![A graph showing one logical call from service B to service C exploding into three retry attempts that each hit service C, tripling its load and producing nine spans for a single user call](/imgs/blogs/debugging-across-service-boundaries-6.png)

Here is the cruelty of it. The downstream (service C) is slow — say it is briefly overloaded. The layer above sees a timeout and retries, which sends *another* request to the already-overloaded C, which makes C slower, which causes more timeouts, which causes more retries. This is a *retry storm* (the cousin of a thundering herd), and it can take a service that was merely slow and drive it fully into the ground. The retries that were meant to improve reliability become the mechanism of the outage. The trace shows nine spans hammering C; C's own metrics show 3× its normal request rate with no increase in upstream user traffic — the extra load is entirely self-inflicted.

Localizing this is straightforward once you know the shape: count the spans to a single target. If you see *N* spans for what should be one logical call, you have retry amplification. The detection one-liner against a trace export (if your backend lets you dump spans as JSON) is just a group-and-count:

```bash
# Dump one trace's spans and count how many hit each downstream target.
# An N>1 count for a single logical call is retry amplification.
otel-cli span list --trace-id "$TRACE_ID" --json \
  | jq -r '.spans[] | select(.kind=="CLIENT") | .attributes["http.url"]' \
  | sort | uniq -c | sort -rn
#   9 http://service-c.internal/lookup    <- one logical call, nine attempts
#   1 http://payments.internal/charge
#   1 http://inventory.internal/reserve
```

The fix is *not* in the failing downstream alone — it is in the retry policy. The right shape is: retry only idempotent operations, cap total retries across the whole call chain (a "retry budget"), add jitter and exponential backoff so retries do not synchronize into a herd, and put a circuit breaker in front of a failing dependency so you stop calling it entirely while it recovers. But the localization lesson is the one to carry: when you see N spans for one call, the suspect is *both* the downstream that is failing *and* the layer whose retry config is amplifying. For the deeper treatment of retry budgets, circuit breakers, and idempotency, the [microservices resilience and the message-queue idempotency posts](/blog/software-development/microservices/debugging-distributed-systems-in-production) are the place to go; here the job is just to recognize the nine-span signature.

## A decision tree: which localizing move for which symptom

You will not always have the luxury of staring at a full waterfall and recognizing a shape. Often you start from a *symptom* — a one-line alert — and need to know your first move. Mapping symptom to move keeps you from the worst failure mode of cross-service debugging, which is opening six codebases and reading hopefully.

![A decision tree routing from the symptom at the open trace to the right localizing move, where slow goes to find the long pole, wrong result goes to find the deepest error span, and a short trace goes to check who dropped the traceparent](/imgs/blogs/debugging-across-service-boundaries-7.png)

Three symptom classes cover most of what lands in your queue. **Too slow** — a p99 latency complaint, a "checkout feels sluggish" ticket, an SLO burn alert on latency. Your first move is *find the long pole*: open a slow trace and look for the one span that is 90% of the time. Latency is almost always one dominant span, and the waterfall hands it to you. **Wrong result** — a 500, a 4xx, a corrupted or missing field in the response, "I got charged twice." Your first move is *find the deepest error span*: open the failing trace and descend to the leaf-most red bar, because errors bubble up and the deepest one is the origin. **Trace looks short / spans missing** — the trace ends at a service that should not be the leaf, or a parent has a big unaccounted-for gap. Your first move is *check who dropped the traceparent*: this is a propagation bug, and the service at the bottom of the truncated trace is the one whose egress instrumentation to inspect — it ran fine but failed to forward context, so its children are orphaned into a separate trace.

The tree's value is that it commits you to a *first move* before you have all the information, which is exactly when the temptation to flail is strongest. "Slow → long pole, wrong → deepest error, short → propagation" is the whole map, and it routes you to one localizing action that names one suspect service. From there you are back in the inverted workflow: read that one service's logs, reproduce it alone, fix.

It is worth saying what is *not* on this tree, because the gaps are instructive. Intermittent failures — fails 1 in 1000, fine the rest of the time — are not a separate symptom; they are a slow-or-wrong that you cannot reproduce on demand, and the move is to capture a trace of a *failing* instance (turn up sampling, or add tail-based sampling that keeps all error traces) and then apply the same tree. Failures only under load are a long-pole or retry-amplification shape that only appears at high request rates, so you reproduce under load (we cover that stress-test in the war story). And failures only on one host are a localization to the *infrastructure* layer rather than the service code — a bad node, a clock skew, a corrupted local cache — which the trace exposes when every failing trace shares one `host.name` attribute. The tree gets you to a suspect service; the host attribute, when present, gets you to a suspect *instance*.

## The toolbox: four instruments and when to reach for each

Distributed tracing is the primary tool, but it is not the only one, and a complete cross-service debugger keeps four instruments and knows which gap each one fills. Reaching for the wrong one wastes time; reaching for the right one localizes in minutes.

![A matrix comparing four cross-service localization tools by what they give you, their setup cost, and when to reach for each, covering distributed tracing, correlation IDs, mesh access logs, and contract tests](/imgs/blogs/debugging-across-service-boundaries-8.png)

**Distributed tracing** (OpenTelemetry generating spans, exported to Jaeger, Grafana Tempo, or Zipkin) gives you the full span tree on one clock — the waterfall. Its setup cost is real: an OTel SDK in every service, a collector to receive spans, and a backend to store and query them. It is your default first move *whenever it is available*, because it answers "which service and how slow" in one picture. Its weakness is sampling — to control cost, most deployments trace only a fraction of requests, so the specific request that failed may not have a trace. Tail-based sampling (decide whether to keep a trace *after* seeing all its spans, so you always keep errors and slow traces) mitigates this and is worth configuring precisely for debuggability.

**Correlation IDs** are the fallback we already met: one header, one log field, `grep` across logs. They give you a flat per-request timeline with no durations and no tree, but they cost almost nothing and work on a fleet with zero tracing infrastructure. Reach for them when tracing is not deployed, or when the failing request was not sampled and the logs are all you have.

**Service mesh access logs** are an underused instrument. If you run a mesh (Istio/Envoy, Linkerd), every sidecar proxy emits an access log for every call it relays, including the per-hop response code and latency, *without any application instrumentation at all*. This is gold when the application's own tracing is broken — when a service dropped the traceparent, the *mesh* still saw the call and logged its status and timing, so the mesh access logs can fill the exact blind spot that the broken trace left. Reach for mesh logs when the trace context is broken and you need ground-truth per-hop status that does not depend on app instrumentation.

**Contract tests** are not a debugging tool at all — they are a *prevention* tool — but they belong in this toolbox because the cheapest cross-service bug is the one that never ships. A consumer-driven contract test (Pact is the common framework) encodes the consumer's expectations of a provider's API as an executable test that runs in the provider's CI; if the provider renames a field or changes a status code that a consumer depends on, the build fails *before* deploy. They cost a Pact suite per consumer-provider pair and a CI integration, and they pay off by killing the entire class of "service B started returning a slightly different shape and service A broke in prod" bugs. The deep treatment is in [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing); here the point is that contract tests turn a future 2am page into a red CI check.

| Tool | What it gives you | Setup cost | Reach for it when |
| --- | --- | --- | --- |
| Distributed tracing | Full span tree on one clock | OTel SDK + collector + backend | Default first move, whenever available |
| Correlation IDs | Flat per-request log timeline | One header + one log field | No tracing deployed, or request unsampled |
| Mesh access logs | Per-hop status and latency, no app code | Free with an Envoy/Linkerd sidecar | Trace context is broken, need ground truth |
| Contract tests | Catch contract breaks in CI | Pact suite per consumer | Prevention — stop the field-rename bug shipping |

The mental model is layered: tracing localizes fastest, correlation IDs are the no-infra fallback, mesh logs are the ground-truth backstop when app instrumentation lies, and contract tests stop a whole bug class at the door. A mature team runs all four. For the design-time view of how to build observability in from the start rather than bolting it on, see [observability: metrics, logs, traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) and the debugging-series companion [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod).

## Worked example: a user-facing 500 that was a mis-mapped 403

Let me walk a real-shaped investigation end to end, because the shape of the reasoning matters more than any single fact.

#### Worked example: localizing a checkout 500 to an auth 403 three hops down

The page comes in at 14:02. Symptom: **"POST /checkout returns 500 for about 4% of users, started ~13:40."** The gateway dashboard confirms a 500 rate that jumped from baseline 0.1% to 4.0% at 13:38. No deploy to the gateway in the window. Six services sit behind /checkout: gateway → BFF (A) → orders (B) → a routing/aggregation service (C) → which fans out to payments (D) and auth (E), and payments calls the orders database via service F.

Step one, **do not read code.** Pull a *failing* trace. I filter the tracing UI for `http.status_code=500 AND http.route=/checkout` in the last 30 minutes and grab one trace id. The waterfall loads. Six spans, nicely nested. I run my three reflexes. Width: nothing is a dramatic long pole — total request time is 140ms, which is normal; this is not a latency bug. Color: there is red. Three red spans, stacked — the gateway span is red (500), the BFF span is red (500), the C span is red (500). But the C span fans out to two children, D (payments) and E (auth), and **the E span is red too — and it is the deepest red span, and its status code is 403.**

Step two, apply the rule: the deepest error span is the origin; everything above it is an echo. So the *origin* is auth (E) returning a 403 — forbidden — and that 403 is bubbling up. But the user is seeing a 500, not a 403. Somewhere between E's 403 and the gateway, the status got *rewritten*. I read the spans bottom-up: E returns 403. Its parent C has status 500. So **C received a 403 from auth and mapped it to a 500.** That is the bug's location: service C mis-maps a downstream 403 into a 500. The user should have gotten a clean 403 ("your session expired, please log in"); instead they got a 500 ("something broke, contact support"), which generated the panic ticket.

Step three, *now* read code — and only C's code. I open service C's repository (not the other five) and search for where it handles downstream auth responses. Time to suspect from page to one-service localization: under five minutes, entirely from the trace, having opened zero source files until I knew which one to open. In C I find the handler:

```python
# service C: aggregation/routing layer
def call_auth(req):
    resp = auth_client.check(req.token)
    if resp.status_code != 200:
        # BUG: any non-200 becomes a generic 500. A 403 is a *client* error,
        # an expired token, not a server fault — it must pass through as 403.
        raise InternalServerError("auth check failed")
    return resp
```

There it is. `if resp.status_code != 200: raise InternalServerError` collapses every non-200 — including the perfectly-expected 403 for an expired session — into a 500. Step four, **reproduce in isolation.** I pull service C up alone, point its `auth_client` at a stub that returns 403, replay the captured request, and confirm: C returns 500. The bug reproduces deterministically on one service on my laptop, with no other service running. It is now an ordinary single-process bug.

Why 4% and why 13:38? Because at 13:38 a token-lifetime config change shortened session TTLs, so more users hit expired tokens, so auth returned more 403s, so C mis-mapped more of them to 500s. The auth 403s were *correct*; the bug was C's handling of them. The fix is to pass client errors through:

```python
def call_auth(req):
    resp = auth_client.check(req.token)
    if resp.status_code == 200:
        return resp
    if 400 <= resp.status_code < 500:
        # Client errors pass through unchanged — a 403 is a 403, not a 500.
        raise HTTPError(status=resp.status_code, body=resp.json())
    raise InternalServerError("auth upstream 5xx")  # only real server faults
```

Before: 4.0% of checkouts returned a 500 and the on-call was paged. After: those requests return a 403 with a "session expired, please sign in again" message, the 500 rate drops back to 0.1% baseline, and the panic stops. Total time from page to fix: about 40 minutes, of which the localization — the genuinely hard part in a six-service fleet — took five. The deepest-error-span rule did the entire localization. To *prevent* recurrence, a contract test asserting that C propagates 4xx as 4xx would have failed in CI the day someone wrote `!= 200`, and that is exactly the prevention layer from the toolbox.

## Worked example: a p99 complaint that was an N+1 three hops down

The second investigation is a latency case, which exercises the long-pole reflex instead of the deepest-error reflex.

#### Worked example: finding an N+1 query in the orders service from a p99 alert

Symptom: **"p99 latency on GET /order-history crossed 1.2s, SLO is 400ms, started gradually over the last two weeks."** No single deploy correlates — it crept up, which already hints at a data-volume problem (something that gets worse as a table grows) rather than a code change. The endpoint touches gateway → BFF (A) → orders (B) → which calls F (the orders database) and inventory.

First move, from the decision tree: *too slow → find the long pole.* I pull a p99-representative slow trace (filter for traces over 1 second on that route) and open the waterfall. Three reflexes. Width: there is unmistakably one dominant region — but it is not one wide bar. It is the orders (B) span, which is 1100ms wide, and *inside* B there are **dozens of thin child spans, all to the same database, stacked back to back.** Color: no red — nothing errored; this is pure latency. Gap: B's 1100ms is fully accounted for by those dozens of tiny DB spans, each ~18ms, lined up sequentially.

That sequential stack of identical short DB calls is the unmistakable signature of an **N+1 query**: the service runs one query to fetch a list (the N orders), then loops and runs *one more query per row* to fetch each row's detail (N additional queries), so a request that should be 2 queries becomes N+1 = (say) 51 queries. Each is fast — 18ms — but 51 × 18ms ≈ 918ms of serial round-trips, and it gets worse as users accumulate order history, which explains the gradual two-week creep. The long pole is B, and the cause is inside B: an N+1 against F.

I have localized to one service (orders, B) and one cause (N+1) entirely from the waterfall — the deepest layer of detail came from B's child spans being individually visible, which is why per-call DB spans are worth instrumenting. Now I read B's code:

```python
# orders service (B): the N+1
def get_order_history(user_id):
    orders = db.query("SELECT id FROM orders WHERE user_id = %s", user_id)  # 1 query
    result = []
    for o in orders:
        # One query PER order -> N more queries. 50 orders = 51 round-trips.
        items = db.query("SELECT * FROM order_items WHERE order_id = %s", o["id"])
        result.append({"order": o, "items": items})
    return result
```

The fix is the standard one — replace N+1 with a single batched query (an `IN` clause or a join), turning 51 round-trips into 1:

```python
def get_order_history(user_id):
    orders = db.query("SELECT id FROM orders WHERE user_id = %s", user_id)
    if not orders:
        return []
    ids = tuple(o["id"] for o in orders)
    # One query fetches ALL items for ALL orders. 51 round-trips -> 2.
    rows = db.query("SELECT * FROM order_items WHERE order_id IN %s", (ids,))
    by_order = collections.defaultdict(list)
    for r in rows:
        by_order[r["order_id"]].append(r)
    return [{"order": o, "items": by_order[o["id"]]} for o in orders]
```

The measured before→after: p99 on /order-history dropped from 1,200ms to roughly 90ms, the number of DB spans inside the B span fell from 51 to 2, and the SLO breach cleared. The deep mechanics of why N+1 is a latency killer and how to spot it from the query plan live in the [database N+1 and slow-query posts](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod) and [python-performance profiling](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path); the cross-service point here is that the *waterfall localized it to B in seconds* — without per-call DB spans you would have seen only "B is slow" and spent an afternoon guessing among B's many queries. The trace told you it was N+1 by showing you fifty-one identical bars.

## Contract and version bugs across the boundary

Not every cross-service bug is a slow span or an error span — some are *silently wrong*, and those are the nastiest because the trace looks completely green. The classic is a **contract bug**: service B changes the shape of its response — renames a field from `userId` to `user_id`, changes a status code from 404 to 200-with-empty-body, makes a previously-optional field required, or changes a unit (cents to dollars) — and service A, which consumed the old shape, now silently misreads it. No span errors; A just produces a wrong result downstream. The user sees a wrong total or a missing item, files a ticket, and your green trace mocks you.

The mechanism that makes this possible is the same loose coupling that makes microservices attractive: services are deployed independently and communicate over a wire format (JSON, Protobuf) that is *not* checked by a shared compiler. In a monolith, renaming a field is a compile error at every call site. Across a service boundary there is no compiler spanning both sides — B's change compiles fine in B's repo, A's code compiles fine in A's repo, and the mismatch only manifests at runtime when A parses B's new response. The type system that would have caught it does not cross the network.

Localizing a contract bug is harder than localizing an error, because there is no red span to chase. The technique: when a result is *wrong* but no span errored, do not look for the deepest error — look for the deepest *change*. Find the request id, pull the trace, and inspect the *span attributes* — the actual request and response payloads, which a well-instrumented service records as span attributes or events. Walk the tree and compare each hop's output to what its consumer expected; the hop where the shape diverges from the contract is your suspect. If you do not record payloads on spans (many teams do not, for privacy and size reasons), you fall back to the correlation id and the raw request/response logs at each hop. Either way you are diffing the *data* across the boundary rather than chasing an error.

```python
# Recording enough payload shape on a span to diagnose a contract bug later.
# Record SHAPE (keys, types), not full PII payloads.
with tracer.start_as_current_span("orders.parse_payments_response") as span:
    body = resp.json()
    span.set_attribute("payments.response.keys", sorted(body.keys()))
    span.set_attribute("payments.response.amount_type",
                       type(body.get("amount")).__name__)
    # If payments renamed `amount` to `amount_cents`, the keys attribute
    # shows `amount_cents` while orders' code still reads `amount` -> wrong.
```

The real defense is prevention, and it is the strongest argument for **contract tests** and explicit **API versioning**. A consumer-driven contract test pins the exact shape A depends on and runs in B's CI; the day someone renames a field B's build goes red and the bug never ships. Versioning the API (or using a schema with explicit compatibility rules, like Protobuf's field-number discipline or Avro's schema resolution) makes incompatible changes structurally impossible to deploy silently. The [microservices contract-testing and versioning posts](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) cover the build-time machinery; the debugging lesson is recognition: a *wrong-but-green* result across services is a contract bug, and you localize it by diffing payload shape across hops, not by chasing a red span that does not exist.

## Sampling and cost: keeping the traces you actually need

Tracing every request at full fidelity would be wonderful and is usually unaffordable. A high-traffic service can emit millions of spans per minute, and storing, indexing, and querying all of them costs real money — so every production tracing setup samples, keeping only some fraction of traces. The *way* you sample, however, decides whether tracing is useful for debugging or actively useless, because the requests you most need — the rare failures — are exactly the ones naive sampling throws away.

**Head-based sampling** decides whether to keep a trace at the *very first* span, before the request has done anything, typically by flipping a weighted coin: keep 5%, drop 95%. The decision propagates down the chain (the "sampled" bit in the `traceparent` flags), so either the whole trace is kept or none of it is. This is cheap and simple, and it is the default in most setups. It is also a debugging trap. Your failures are rare — a 1-in-1000 error, a p99.9 latency spike — and head sampling keeps them with the same 5% probability as the boring requests, so 95% of your failures have *no trace at all*. You get paged for an error, pull up the request, and the trace is simply not there because the coin came up tails. You are left with correlation IDs and `grep`, which works but is slower and grainier, as we saw.

**Tail-based sampling** inverts the timing: the collector buffers *all* spans of a trace until the request completes, and only *then* decides whether to keep it — armed with the full picture. That lets you write the rule that matters: **always keep a trace if any span errored, or if the total duration exceeded your SLO; sample the fast, successful ones at 1%.** Now 100% of your failures have a complete trace and the boring traffic is cheap. This is the single highest-value tracing configuration change for debuggability, and it is worth the cost: tail sampling needs the collector to hold a trace's spans in memory briefly and to receive every span (more network than head sampling, which drops at the source), but the debugging payoff — never losing the trace of the request that actually broke — is enormous. If your tracing exists mainly to debug incidents, tail-based always-keep-on-error is the configuration you want.

A complementary trick is **exemplars**: a metric data point (say, the p99 latency bucket of a histogram) carries a pointer to one representative trace id that landed in that bucket. When you see a latency spike on a dashboard, you click the spike and jump straight to a real trace that exhibited it — no searching, the metric *is* linked to a representative request. Exemplars are the bridge from "a graph went red" to "here is the waterfall of a request that caused it," and they make the metrics-to-trace handoff one click instead of a manual hunt.

The last cost lever is **span attributes**, the key-value tags you attach to spans (the DB statement, the user tier, the route, the host). They are what make a localized span *diagnosable* — but every distinct attribute value is *cardinality*, and high-cardinality attributes (a raw user id, a full URL with query params, a millisecond timestamp) explode storage and index cost in some backends. The discipline is to record high-*value*, bounded-cardinality attributes (route template not full URL, error class not error message, user *tier* not user id) on every span, and to put the genuinely high-cardinality stuff (the specific id, the full payload) in *logs* keyed by the correlation id, which are cheaper to store and which you only read once tracing has already pointed you at the suspect. Trace to localize, log to diagnose — and keep the attribute cardinality on the trace side modest so the bill stays sane.

## War story: the retry storm that took down a healthy database

Let me tell the kind of cross-service incident that becomes a legend on a team, constructed to be realistic and to braid together the patterns above — this is an illustrative composite of a very common failure mode, not a specific company's documented postmortem.

It is a Tuesday afternoon. A product service depends on a recommendations service, which depends on a feature-lookup service, which reads from a shared cache that falls through to a database. The database is healthy — p99 query time 8ms, CPU 30%, plenty of headroom. At 15:10 the cache cluster does a routine node rotation and, for about 40 seconds, its hit rate drops while the new node warms. During those 40 seconds, more reads fall through to the database. The database, being healthy, serves them — at maybe 25ms instead of 8ms, slightly slower because of the extra load, but fine.

Here is where it goes wrong. The feature-lookup service has a 20ms timeout on its database reads. During the cache miss window, some reads cross 20ms, so feature-lookup *times out* and *retries* — 3 attempts, no backoff, no jitter. Each retry is another database query. The database, now serving 3× the read load, slows to 40ms. More reads cross 20ms. More retries. Meanwhile the recommendations service has *its own* 100ms timeout on feature-lookup and *its own* 3 retries, so each user request becomes 3 recommendations calls, each becoming 3 feature-lookup calls, each becoming 3 database queries: one user request is now up to 27 database queries. The database goes from 30% CPU to 100% in ninety seconds, query latency blows past every timeout, and *everything* starts failing. The product service shows 500s. The cache rotation finished long ago — the original trigger is gone — but the system is now in a self-sustaining retry storm, feeding on itself.

The page says "database down." It is the natural conclusion — the database is at 100% CPU and timing out. A team that trusts that conclusion spends an hour failing over the database, scaling it up, and staring at slow-query logs, and *none of it helps*, because the database is a victim, not the culprit. The break in the case came from the trace. Someone pulled a single failing user request and counted the spans: **27 database spans for one user request.** That number is impossible for a healthy code path — order-history fetches a handful of rows, not 27 — and it is the unmistakable signature of retry amplification stacked across two layers (3 × 3 × 3). The localization was not "the database is slow"; it was "*something is multiplying one call into 27*," and the two retrying layers were feature-lookup and recommendations.

The fix had nothing to do with the database. It was: add exponential backoff with jitter to both retry layers so retries stop synchronizing into a herd; cap the total retry budget across the chain so one user request can never become 27 queries; and put a circuit breaker in front of feature-lookup so that when it starts failing, the recommendations service *stops calling it* for a few seconds and lets it recover instead of hammering it. With a circuit breaker and a retry budget, the *next* cache rotation caused a 40-second blip in recommendations quality and zero database impact — the system degraded gracefully instead of cascading. The measured outcome: the same cache-rotation event that previously caused a 12-minute full outage became a sub-minute, invisible-to-users quality dip.

The lessons are three. First, **the symptom is not the cause** — the database at 100% CPU was the most visible thing and the least causal thing; the cause was retry configuration two layers up. Second, **retries are a foot-gun that turns slow into down** — a dependency that is merely slow becomes a dependency that is overwhelmed once retries amplify the load, and the amplification is multiplicative through layers. Third, **the trace localized what the metrics could not** — every service's metrics said "I am slow and erroring," which is true and useless; the *count of spans per request* in a single trace named the failure class instantly. When the metrics all point at the victim, the trace points at the cause. For the design patterns that prevent this — circuit breakers, retry budgets, bulkheads, backpressure — see [anatomy of an outage](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) and [debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale).

## Stress-testing the method: when the easy path fails

The clean workflow — pull a trace, read the shape, localize — assumes you *have* a trace of the failing request. Half the craft of cross-service debugging is what you do when that assumption breaks. Let me stress-test the method against the hard cases.

**What if the failing request was not sampled?** Most tracing samples — keeps 1% or 5% of traces to control cost — and a rare failure (1 in 1000) is statistically likely to be in the un-sampled 99%. You have a 500 in the logs and no trace for it. Two moves: first, fall back to the correlation id in the logs and reconstruct the timeline by `grep`, which still gets you to a suspect service. Second, and better long-term, switch to **tail-based sampling**, where the decision to keep a trace is made *after* all its spans arrive, so you can keep 100% of error traces and slow traces while sampling the boring fast ones. Configure your collector to always keep traces with an error span or a duration over your SLO, and the failing request will be there next time. This is the single highest-value tracing config change for debuggability.

**What if the trace is broken — a service dropped the context?** Then the trace ends above the real culprit and you are blind exactly where you need sight. Move to **mesh access logs**: the Envoy or Linkerd sidecar logged every hop's status and latency regardless of app instrumentation, so you can reconstruct the per-hop picture the broken trace hid. Then go fix the propagation bug in the service that dropped the header (check its egress: is it using an un-instrumented HTTP client? a queue hop that does not carry headers?) so the *next* incident has a complete trace.

**What if it only reproduces under load?** Retry amplification, lock contention, connection-pool exhaustion, and thundering herds frequently do not show up at one request per second — they need concurrency. You reproduce with a load generator (`vegeta`, `wrk`, `k6`, or `hey`) replaying the request at production-like RPS while tracing is on, then pull a *failing* trace from under load:

```bash
# Replay /checkout at 200 req/s for 60s, capture status distribution,
# then pull a failing trace from the tracing UI for that window.
echo "POST http://gateway.internal/checkout
Content-Type: application/json
@checkout-payload.json" \
  | vegeta attack -rate=200 -duration=60s \
  | vegeta report -type='hist[0,100ms,250ms,500ms,1s,2s]'
# Watch the histogram's tail bucket grow; that tail is your failing traces.
```

**What if you cannot attach a debugger in prod?** You usually cannot and usually should not — you are not going to `gdb` the payments process under live traffic. This is exactly why tracing and logging exist: they are the *non-intrusive* observability that replaces the debugger across the boundary. You localize from the trace to one service, then reproduce *that one service in isolation* on your laptop or in staging, where you *can* attach `pdb`/`gdb`/`delve` freely. The cross-service phase is debugger-free by necessity; the single-service phase, after localization, is where the interactive debugger from [the debugger is a microscope](/blog/software-development/debugging/mastering-an-interactive-debugger) comes back into play. The boundary between "localize in prod with traces" and "reproduce in isolation with a debugger" *is* the workflow.

**What if it only happens on one host?** Then every failing trace shares a `host.name` or `pod` attribute that the healthy ones do not, and your localization is to an *instance*, not a service version. The cause is usually environmental — a bad disk, a skewed clock, a stale local cache, a corrupted config on one node, a noisy neighbor. The trace exposes it the moment you group failing traces by host and see them all land on one. The fix is often "drain and replace that node," and the prevention is "why did one node diverge?" — which is an infrastructure question, not a code question.

## How to reach for this, and when not to

Every technique here has a cost, and a senior engineer is decisive about when *not* to reach for the heavy machinery.

**Reach for distributed tracing first, always, when it exists.** It is the lowest-effort, highest-information move for any "which service" question. There is essentially no scenario where reading six codebases beats reading one waterfall. If your fleet has tracing and you are debugging a cross-service issue by reading code, you are doing it wrong.

**Do not build a full tracing stack to debug one incident.** If you are paged at 3am and there is no tracing, the answer is *not* "let me deploy OpenTelemetry across six services right now." It is correlation IDs and `grep` — minutes, not days. Build tracing afterward, in daylight, as prevention. The right tool for the incident in front of you is the one already deployed plus the cheapest fallback.

**Do not attach a debugger to a production service across the boundary.** You cannot meaningfully `gdb` one process in a six-service request anyway — you would freeze one hop while the others time out. Localize with traces, then reproduce the *single* suspect service in isolation and debug it there with full debugger power. Never pause a live prod process that other services are waiting on.

**Do not chase a green-but-wrong result as if it were an error.** If no span is red but the result is wrong, stop looking for the deepest error span — there isn't one. Switch to diffing payload shape across hops; it is a contract bug, and the error-chasing reflex will waste your night.

**Do not trust a duration that equals a timeout.** A span that took exactly 1000ms against a 1000ms timeout is a *symptom*; the cause is below the cutoff. Always follow the chain past any duration that suspiciously matches a configured timeout.

**Do not fix the victim.** When metrics point at a database at 100% CPU or a downstream that is timing out, ask whether it is the culprit or the victim of amplification upstream. Count spans per request; if one logical call became many, the cause is a retry config above, not the struggling service below.

**Do invest in prevention proportional to pain.** Contract tests and explicit versioning cost real engineering time; spend it on the boundaries that have *actually* burned you, not on every pair of services speculatively. The boundary between payments and orders deserves a contract test; an internal read-only metrics endpoint probably does not.

## Key takeaways

- **Localize before you read code.** With six services, your prior on which one owns the bug is wrong five times out of six. Find the service from the trace first, then open exactly one repository.
- **No single stack trace spans services** — each hop's stack is destroyed when it returns a response. The distributed trace is the stack trace you do not get for free; it is reconstructed from spans tagged with a shared trace id propagated hop to hop.
- **Read the waterfall by width, color, gap.** The widest deepest bar is your latency long pole; the deepest red bar is your error origin; a gap between a parent and its visible children is a missing span, usually a propagation bug.
- **The deepest error span is the cause; everything above it is an echo.** Errors bubble up. The leaf-most red span names the service that originated the failure.
- **A duration equal to a configured timeout is a symptom, not a cause.** Follow the chain past the timeout to find the real long pole hiding below the cutoff.
- **N spans for one logical call means retry amplification.** Suspect both the failing downstream and the retrying layer; uncapped retries turn a slow dependency into a down one.
- **A broken trace blinds you at exactly the hop you need.** When a trace looks short and ends where it shouldn't, the bottom service dropped the `traceparent` — check its egress instrumentation, and use mesh access logs to fill the blind spot.
- **A green-but-wrong result is a contract bug.** Diff payload *shape* across hops rather than chasing a red span that does not exist; prevent it with consumer-driven contract tests in CI.
- **Correlation IDs are the no-infra fallback.** One header plus one log field gives you a `grep`-able cross-service timeline when full tracing is unavailable or the request was unsampled.
- **After localization it is an ordinary single-process bug.** Reproduce the one suspect service in isolation and debug it with the full power of an interactive debugger — the cross-service skill is purely the localization step.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post applies across processes.
- [Observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) — the sibling on using metrics, logs, and traces to debug what you cannot attach a debugger to.
- [Distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering) — when the cross-service bug is a timing/ordering hazard rather than a slow or errored span.
- [It's the network: packet and protocol tracing](/blog/software-development/debugging/its-the-network-packet-and-protocol-tracing) — when the failure is in the bytes on the wire between hops, below the application span.
- [Observability: metrics, logs, traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — building the trace-context plumbing in from the start instead of bolting it on.
- [Distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) — the deep how-to on instrumenting a service fleet with OTel and a Jaeger/Tempo backend.
- [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) — the prevention layer that kills the field-rename contract bug in CI before it ships.
- The OpenTelemetry documentation (context propagation, spans, sampling), the W3C Trace Context specification, and Jaeger / Grafana Tempo / Zipkin docs for the trace backends — the canonical references for the tools used throughout this post.

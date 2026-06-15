---
title: "Distributed Tracing and Observability With OpenTelemetry"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "In microservices a single request crosses many services, so you cannot debug what you cannot see, and this is the practitioner's guide to observability: the three pillars of logs, metrics, and traces; distributed tracing and context propagation in depth; OpenTelemetry and the collector; head versus tail sampling; the cardinality trap; and how you make every request traceable and every log correlated."
tags:
  [
    "microservices",
    "observability",
    "distributed-tracing",
    "opentelemetry",
    "distributed-systems",
    "software-architecture",
    "backend",
    "monitoring",
    "telemetry",
    "sampling",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-1.webp"
---

At 2:47 on a Thursday afternoon the ShopFast checkout button started feeling slow. Not broken — slow. The customer-success Slack channel filled up with the same vague report a dozen times: "checkout is taking forever." The on-call engineer pulled up the dashboards and found exactly the kind of unhelpful picture that makes you want to throw your laptop into the sea. The gateway's p99 latency had climbed from 220ms to 640ms. That was the only signal. Every individual service had a green health check. CPU was fine everywhere. There were no error logs spiking. The graph just said *something* in the checkout path got slower, and the checkout path touched the gateway, the order service, the payment service, and the inventory service — four services, any of which could be the culprit, plus a third-party payment processor sitting behind the payment service that the dashboards could not see at all. The engineer had a number that said "it's slow" and absolutely no way to answer the only question that mattered: *slow where?*

This is the defining problem of operating microservices, and it is the reason observability is not a nice-to-have you bolt on after launch but a load-bearing part of the architecture. In a monolith, "checkout is slow" is a tractable question: you attach a profiler to one process, you look at one flame graph, and the hot path is right there. The instant checkout becomes four services across a network, that flame graph shatters into four separate processes on four separate machines, and the call that ties them together — the actual journey of one user's request as it hops from gateway to order to payment to inventory and back — exists in no single place. It is smeared across four sets of logs, four sets of metrics, and four machines that do not know about each other. The engineer who can reassemble that journey on demand resolves the incident in four minutes. The engineer who cannot spends two hours grepping logs by timestamp and guessing.

![A vertical stack diagram showing your question at the top resting on the three pillars of traces metrics and logs which all rest on a shared trace id that stitches them together](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-1.webp)

The figure above is the whole architecture of seeing your system, drawn as a stack. At the bottom is a single shared identifier — a trace id — that every part of the system stamps onto everything it emits. Resting on it are the three pillars that everyone cites and few people use well: logs, which tell you *exactly what happened* at a point in time; metrics, which tell you *how often and how bad* in aggregate; and traces, which tell you *where in a request* the time went. And resting on all three is the only thing that matters at 2:47 on a Thursday: your ability to walk up to a system you did not design that morning and ask it a question you had never thought of before. By the end of this post you will be able to instrument the ShopFast services with OpenTelemetry so that every checkout request is a single traceable story, every log line carries the trace id that links it to that story, and the engineer on call can go from "checkout is slow" to "the payment service is spending 400 milliseconds waiting on the external processor" in the time it takes to open one trace. We will build that capability from first principles, then make it survive 50,000 requests a second without bankrupting you on storage.

## Monitoring tells you the knowns; observability lets you ask new questions

Start with a distinction that sounds like a buzzword fight but is actually the entire mindset shift, because getting it wrong is why so many teams have a hundred dashboards and still cannot debug anything. **Monitoring** is what you do when you already know the questions. You decide in advance, "I care about request rate, error rate, and p99 latency," you build a dashboard with those three graphs, and you set an alert when any of them crosses a line. Monitoring is about *known unknowns*: you know which questions matter, you just do not know their current answers, so you put the answers on a wall. It is essential, and most of this section's golden-signals material belongs to it. But monitoring has a hard ceiling: it can only answer the questions you thought to ask when you built the dashboard.

**Observability** is the property of a system that lets you answer questions you *did not* anticipate — the *unknown unknowns* — without shipping new code to go find the answer. The 2:47 incident is the test case. Nobody had built a "which service in the checkout path is slow today" dashboard, because there are too many possible slow paths to pre-build a dashboard for each. A monitored-but-not-observable system forces you to add logging, deploy, and wait for the problem to recur. An observable system already emitted enough high-fidelity, correlatable data that you can slice it a new way *right now*: "show me the traces for slow checkouts in the last ten minutes, broken down by which span took the longest." You ask a brand-new question of data you already have. That is the difference, and it is worth being precise about because it changes what you instrument. Monitoring pushes you toward a few pre-aggregated numbers. Observability pushes you toward rich, dimensional, per-event data you can group and filter after the fact.

The honest framing for a senior is that you need both, and they trade off. Pre-aggregated metrics are cheap to store and instant to query but throw away the detail you need for novel questions. Raw per-event data answers any question but is expensive and slow to query at scale. The craft is knowing which question lives in which pillar, and the rest of this post is largely about exactly that: when you reach for a metric, when for a log, and when only a trace will do. A junior thinks observability is "install Datadog." A senior knows it is a set of deliberate decisions about what data to emit, at what cardinality, at what sampling rate, correlated by what key — decisions that cost money on one side and debugging-blindness on the other.

## The three pillars, and the one that is unique to distributed systems

The three pillars are logs, metrics, and traces. You almost certainly already use the first two; the third is the one microservices forced into existence, and it is the one this post spends the most time on. But you cannot use traces well without understanding how all three divide the labor, so let us take them in turn and tie each to the ShopFast checkout.

A **log** is a timestamped record of a discrete event: "order 8821 created," "payment authorization failed: card declined," "cache miss for product 552." Logs are the highest-fidelity pillar — a log line can carry arbitrary structured detail — and therefore the most expensive per unit and the easiest to drown in. The single most important upgrade you can make to logs in a microservices system is to make them **structured** (emit JSON, not free text) and to stamp every line with the **trace id** of the request that produced it. A free-text log line — `2026-06-15 14:47:02 payment failed for order 8821` — is a needle that you find by grepping and reading. A structured line — `{"ts":"...","level":"error","svc":"payment","trace_id":"4bf92f...","order_id":"8821","msg":"PSP declined","psp_code":"51"}` — is a row you can query, group, and, crucially, *pivot from a trace into*. The trace id is the join key that turns four disconnected log streams into one story, and we will build that link explicitly later. The relationship between a structured log event and the request that caused it is exactly the kind of thing the sibling post on [the anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice) treats as table stakes for any service worth shipping.

A **metric** is an aggregate number sampled over time: requests per second, error count, a latency histogram, CPU utilization, queue depth. Metrics deliberately throw away per-event detail in exchange for being tiny and fast. A counter that says "payment service handled 1.2 million requests this minute, 340 of them errored" is a few bytes regardless of how many requests it summarizes, which is why metrics power your dashboards and alerts. Two acronyms tell you which metrics to emit, and they are worth committing to memory because they are the difference between a dashboard that helps and a wall of noise. **RED** — Rate, Errors, Duration — is the request-centric trio you want for every *service*: how many requests per second, what fraction errored, and the latency distribution. **USE** — Utilization, Saturation, Errors — is the resource-centric trio you want for every *resource* (CPU, memory, disk, a connection pool): how busy it is, how much work is queued waiting because it is full, and its error count. RED tells you the *service* is unhealthy; USE tells you *which resource* is the reason. The deeper treatment of which signals to alert on and how to set thresholds belongs to the sibling post on [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices); here it is enough to know that metrics are your monitoring layer.

A **trace** is the pillar microservices created. A trace is the complete record of one request as it travels through your system — every service it touched, in what order, how long each step took, and how the steps nested. It is the only pillar that natively understands that "checkout" is not one event in one process but a *causal tree of work* spread across many. Metrics can tell you the payment service's p99 is high; logs can tell you a specific payment failed; but only a trace can tell you that *this particular checkout* spent 400 of its 600 milliseconds inside the payment service's call to the external processor, while the order and inventory services were idle the whole time. Traces are what make the 2:47 question answerable, and they are the reason this post exists.

![A stack diagram of the three pillars with your question on top resting on traces metrics and logs all anchored to a shared trace id](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-1.webp)

The figure restates the division of labor as a load-bearing structure, and the ordering is deliberate. Your *question* sits on top — observability exists to serve questions, not to fill disks. The three pillars carry it, each answering a different shape of question: traces for "where in the request," metrics for "how often and how bad," logs for "exactly what." And the whole thing rests on the trace id, because without a shared identifier stamped across all three, you have three disconnected datasets and you are back to grepping by timestamp. Notice what the picture refuses to say: it does not claim any one pillar is best. The senior move is not "traces are the new hotness, log less" — it is using each pillar for the questions it answers cheaply and refusing to ask metrics to do logs' job or vice versa. The matrix later in this post makes that allocation explicit.

#### Worked example: which pillar answers "find the 400ms span in a 600ms request"

Take the 2:47 incident and run it through all three pillars to feel the difference concretely. The symptom is a metric: gateway p99 for `POST /checkout` rose from 220ms to 640ms. Good — the metric did its monitoring job and told you *that* something is slow and *how* slow, cheaply, for every request. Now you want to know *where*. Can a metric answer it? Only if you happened to pre-build a per-service-per-endpoint latency breakdown for the checkout path, and even then it gives you averages, not the actual slow request. Can a log answer it? You can grep the payment service logs around 14:47, but the logs say "authorization succeeded in 410ms" only if someone thought to log that duration, and even then you have to manually correlate it to the gateway request by timestamp across four services and pray the clocks agree. The trace answers it directly and unambiguously: open any slow checkout trace, look at the waterfall, and the payment span is visibly 400ms wide while its siblings are slivers. One pillar makes the question trivial; the other two make it a research project. That is the entire argument for traces in three sentences.

## A trace is a tree of spans

Let us get precise about what a trace actually *is*, because the vocabulary is small and once you have it the rest of distributed tracing falls into place. The atom of tracing is a **span**. A span represents a single unit of work with a start time and a duration — "handle the HTTP request at the gateway," "execute the SQL query," "call the payment service," "authorize with the external processor." A span has a name, a start timestamp, a duration, a status (ok or error), and a bag of key-value **attributes** that describe it: `http.method=POST`, `http.route=/checkout`, `db.system=postgres`, `order.id=8821`, `payment.amount=49.99`. Attributes are how you make a span *queryable* — they are the dimensions you will later filter and group by.

A **trace** is a tree of spans that all share one **trace id** and are linked by **parent-child** relationships. The first span — the one with no parent — is the **root span**, usually the entry point (the gateway receiving the request). When the gateway calls the order service, the order service's work becomes a **child span** whose parent is the gateway span. When the order service calls payment, payment is a child of order. The tree structure is the causal structure of the request: parent spans *caused* their children, and a span's duration encloses its children's durations. This is why a trace is so much more informative than a flat list of timings — it preserves *who called whom*, so you can see not just that payment was slow but that payment was slow *as a consequence of* the order service calling it, which the order service called as a consequence of the gateway. The nesting is the explanation.

Two span ids matter at every hop: the span's own **span id** (unique to that span) and its **parent span id** (the span id of whoever called it). Combine the shared trace id with the parent links and you can reconstruct the entire tree from spans that were emitted by four different services on four different machines and never spoke to each other directly. Each service emits its spans independently to a collector; the collector (or the backend) stitches them into a tree by trace id and parent id. This is the central trick of distributed tracing and it is worth saying slowly: *no single service ever sees the whole trace.* The gateway sees its span and knows it called order. The payment service sees its span and knows order called it. The tree only exists once all the spans are gathered and joined by id — which is why the collector and backend exist, and why the *propagation* of those ids across service boundaries is the make-or-break mechanic we turn to next.

## Context propagation: how a trace survives a network hop

Here is the question that everything hinges on. The gateway starts a root span with trace id `4bf92f3577b34da6` and span id `a1b2`. It then makes an HTTP call to the order service. The order service is a different process on a different machine; it has no idea the gateway span exists. How does the order service know to create its span *as a child* of `a1b2`, under the same trace id `4bf92f3577b34da6`, instead of starting a brand-new unrelated trace? The answer is **context propagation**: the gateway puts the trace context into the outgoing request's headers, and the order service reads it out of the incoming request's headers. The trace id rides along *with the request itself*, on every hop, so each service can attach its work to the right tree.

The standard format is **W3C Trace Context**, a single HTTP header called `traceparent` that looks like this: `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`. It is four dash-separated fields: a version (`00`), the 16-byte trace id, the 8-byte parent span id (the *caller's* span id, which becomes the callee's parent), and an 8-bit flags field whose low bit is the **sampling flag** (`01` = this trace is being recorded, `00` = it is not). That single header is the entire mechanism. The gateway sets `traceparent` to its own trace id and span id; the order service reads it, sees trace id `4bf92f...` and parent `a1b2`, and creates its span under that trace with parent `a1b2`. When the order service in turn calls payment, it sets `traceparent` with the *same* trace id but *its own* span id as the new parent. The trace id is constant across the whole journey; the parent span id changes at every hop to point at the immediate caller. There is a companion header, `tracestate`, that carries vendor-specific key-value data, but `traceparent` is the load-bearing one.

![A branching graph showing the client starting a root span then the gateway setting traceparent and fanning to the order service which calls both the payment service and the inventory service while spans from gateway and payment converge on the collector](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-3.webp)

The figure traces one trace id across the ShopFast checkout. The client starts the root span; the gateway receives it, sets `traceparent`, and calls the order service; the order service fans out to payment and inventory, propagating the same trace id to each. Notice the branch: the order service calls *two* downstreams, so the tree branches there, and the payment span (marked slow, 400ms) and the inventory span (fast, fine) are *siblings* under the order span. The dashed edges to the collector show the other half of the mechanic — every service independently exports its finished spans to the collector, which reassembles the tree by trace id. The thing to internalize from the picture is that propagation and collection are two separate flows: the trace id flows *forward* with the request through the headers, and the finished spans flow *sideways* out to the collector. Get the forward flow wrong — drop the header on one hop — and the trace *breaks*: the spans after the broken hop start a new orphan trace, and your beautiful waterfall splits into two disconnected halves. This is the single most common tracing bug, and it almost always lives at a boundary where someone wrote their own HTTP client and forgot to propagate headers, or at a queue hop where the trace context was not put onto the message.

There is one more concept that rides the same rails: **baggage**. Baggage is arbitrary key-value data you attach to the trace context that *propagates to every downstream service*, separate from span attributes. Where an attribute is local to one span, a baggage item set at the gateway — say `baggage: tenant=acme,plan=enterprise` — is readable by *every* service the request later touches, because it travels in headers alongside `traceparent`. Baggage is powerful and dangerous: it is the clean way to thread a tenant id or a feature-flag decision through a deep call graph without adding it to every function signature, but every byte of baggage is sent on every hop, so a careless `baggage: full_user_profile_json=...` multiplies your header size across the whole tree. Use it for small, high-value identifiers; never for payloads.

## Reading a trace waterfall: where the time actually went

When you open a trace in Jaeger, Tempo, or any backend, you see a **waterfall**: each span drawn as a horizontal bar, positioned left-to-right by its start time and sized by its duration, indented to show parent-child nesting. The waterfall is the single most useful debugging artifact microservices produce, and learning to read one fluently is a genuine skill. The root span is the full width of the request — it starts at zero and ends when the response goes out. Its children sit underneath, each starting somewhere inside the parent's span and ending before the parent does. A child that nearly fills its parent's width is where the parent spent most of its time; a child that is a thin sliver was fast. Gaps matter too: if a parent span is 600ms wide but its children only account for 250ms of bars, the missing 350ms was spent *in the parent itself* — local computation, serialization, or waiting on something not instrumented.

![A timeline waterfall of a checkout trace where the gateway root span spans six hundred milliseconds the order span nests inside it inventory is a thin fast bar and the payment span is a wide four hundred millisecond bar driven by an external processor call](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-4.webp)

The figure is the 2:47 incident, solved, drawn as the waterfall the on-call engineer wished they had opened first. Read it left to right by start offset. The gateway root span is the full 600ms. The order span starts at 5ms and runs 580ms — almost the whole request lives inside order's orchestration. Inside order, two children: inventory starts at 20ms and is a clean 30ms bar (fine, fast, not the problem), and payment starts at 60ms and is a *400ms* bar — two-thirds of the entire request. Drill into payment and its own child, the external PSP call, is 380ms of that 400ms. The waterfall has localized the problem to a single span in under five seconds of looking: the payment service is spending almost all of its time waiting on the third-party processor, and the fix is on the payment-to-PSP edge (a timeout, a faster processor, a fallback) — *not* in the order service, *not* in inventory, *not* in the gateway, all of which the dashboards could not have ruled out. This is the payoff. Every other technique in this post exists to make sure that when you need this waterfall, it is there, it is complete, and you can find the slow trace among millions.

A few reading heuristics that separate a fluent operator from a beginner. **Wide bars are suspects, but check whether the width is the span's own work or its children** — a wide span full of wide children is just an honest orchestrator; a wide span with thin children is hiding local slowness or an un-instrumented wait. **Sequential children that could be parallel are an optimization target** — if order calls inventory, *then* payment, *then* shipping, one after another, and they do not depend on each other, you are paying their latencies in series; the waterfall shows the staircase and the fix is to fan them out concurrently. **A span marked error (red) with a long tail** often means a retry storm — the span retried a failing downstream several times before giving up, and each attempt is visible. **A trace that ends abruptly** with a downstream span that never completed points at a timeout or a crash mid-request. You learn to scan a waterfall the way a doctor reads an X-ray: not reading every pixel, but pattern-matching to the handful of shapes that mean trouble.

## OpenTelemetry: the vendor-neutral standard

For years, instrumenting your services meant picking a vendor and littering your code with that vendor's SDK — Datadog's tracer, New Relic's agent, Zipkin's client — and switching vendors meant re-instrumenting everything. **OpenTelemetry** (OTel) ended that. It is a vendor-neutral, open standard (a CNCF project, the merger of the older OpenTracing and OpenCensus efforts) that defines how you generate, collect, and export telemetry — traces, metrics, and logs — independent of where you send them. You instrument once against the OTel API, and you can point the data at Jaeger today, Grafana Tempo tomorrow, and a commercial vendor next quarter by changing *configuration*, not code. For a microservices fleet where you may use different backends for different signals or migrate vendors as you scale, this decoupling is worth a great deal.

OTel has a few parts worth naming. The **API** is what your application code calls — `tracer.start_span(...)`, `span.set_attribute(...)` — and it is deliberately minimal and stable. The **SDK** is the implementation behind the API that actually builds spans, applies sampling, and batches exports; you configure it at startup. **Auto-instrumentation** is the magic that makes adoption realistic: language-specific libraries that hook into common frameworks (your HTTP server, your gRPC stubs, your database driver, your HTTP client) and emit spans *for code you did not write*, so you get gateway-receives-request, order-calls-payment, and the SQL query spans for free, and you only hand-write spans for your own business logic. **OTLP** (the OpenTelemetry Protocol) is the wire format every OTel component speaks — a gRPC or HTTP protocol for shipping telemetry — and it is what makes the whole thing interoperable. And the **Collector** is a standalone process that receives OTLP, processes it, and exports it onward, which we will treat in depth next because it is where most of the production decisions live.

```python
# ShopFast order service (Python / FastAPI). Set up the OTel SDK once at startup.
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Resource attributes identify THIS service in every span it emits.
resource = Resource.create({
    "service.name": "order-service",
    "service.version": "2.4.1",
    "deployment.environment": "prod",
})

provider = TracerProvider(resource=resource)
# Batch, don't export one span per network call: huge efficiency win (see optimization).
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://otel-collector:4317"),  # OTLP gRPC to the collector
        max_export_batch_size=512,
        schedule_delay_millis=2000,
    )
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("order-service")
```

That block runs once when the service boots. From then on, any code can grab the tracer and create spans. In practice you let auto-instrumentation handle the framework-level spans and you hand-write spans only around the business logic you care about — the part no library knows is meaningful.

```python
# Hand-written spans around business logic, with attributes that make the span queryable.
@app.post("/orders")
async def create_order(req: OrderRequest):
    with tracer.start_as_current_span("create_order") as span:
        # Attributes turn a span into a row you can later filter and group by.
        span.set_attribute("order.item_count", len(req.items))
        span.set_attribute("order.amount_usd", req.total)
        span.set_attribute("customer.tier", req.customer_tier)
        try:
            await reserve_inventory(req.items)       # child span (auto-instrumented HTTP)
            payment = await charge_payment(req)       # child span; this is the slow one
            span.set_attribute("payment.psp_latency_ms", payment.latency_ms)
            order = await persist_order(req, payment) # child span (auto-instrumented DB)
            span.set_status(trace.StatusCode.OK)
            return order
        except PaymentDeclined as e:
            span.set_status(trace.StatusCode.ERROR, "payment declined")
            span.record_exception(e)                  # attaches the stack trace to the span
            raise
```

Notice what makes this good: every attribute is a *dimension you might later want to slice by*. `customer.tier` lets you ask "are enterprise customers' checkouts slower?" `payment.psp_latency_ms` lets you correlate slow traces with PSP latency. `record_exception` on error means the trace itself carries the failure detail, so a failing trace is self-explanatory. The here-is-a-span-around-my-logic pattern is the same in every language; here it is in Go for the payment service, which is where the trace context gets propagated onward.

```go
// ShopFast payment service (Go). Propagating context across the call to the external PSP.
func (s *PaymentService) Charge(ctx context.Context, req ChargeRequest) (*Charge, error) {
    // ctx already carries the trace context extracted from the inbound gRPC metadata
    // by the otelgrpc server interceptor. We start a child span under it.
    ctx, span := s.tracer.Start(ctx, "charge_payment")
    defer span.End()
    span.SetAttributes(
        attribute.String("psp.provider", "stripe"),
        attribute.Float64("payment.amount_usd", req.Amount),
    )

    // The otelhttp transport injects traceparent into the outbound request automatically,
    // so the PSP call shows up as a child span if the PSP also speaks OTel (and as a
    // timed external span either way).
    client := http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)}
    httpReq, _ := http.NewRequestWithContext(ctx, "POST", s.pspURL, body)

    start := time.Now()
    resp, err := client.Do(httpReq)        // THIS is the 380ms span in the waterfall
    span.SetAttributes(attribute.Int64("psp.latency_ms", time.Since(start).Milliseconds()))
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, "psp call failed")
        return nil, err
    }
    return parseCharge(resp), nil
}
```

The critical line is `http.NewRequestWithContext(ctx, ...)` paired with the `otelhttp` transport: because the context carries the trace, the outbound request automatically gets a `traceparent` header injected, and the PSP call becomes the child span we saw eating 380ms in the waterfall. Forget to pass `ctx` (a startlingly common bug — people grab a background context out of habit) and the trace silently breaks at exactly the hop you most need to see. The connection between idiomatic context-passing and an unbroken trace is the practical heart of instrumenting Go, and it rhymes with the broader discipline the sibling on [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) lays out: every network boundary is a place where something — a header, a deadline, a trace context — has to be deliberately carried across, because nothing crosses a process boundary by accident.

## The Collector: receive, process, export

You *could* have every service export its spans straight to your tracing backend. You should not, at any real scale, and the reason is the **OpenTelemetry Collector** — a standalone process (a sidecar, a daemonset, or a central deployment) that sits between your apps and your backends and does three things, captured in its config structure: **receivers** take telemetry in, **processors** transform it, and **exporters** send it out. The collector is the single most important operational component in an OTel deployment because it decouples your application code from your backend choices and gives you one place to enforce sampling, batching, and cardinality control without touching application code.

![A vertical stack showing apps emitting OTLP at the top flowing down through collector receivers then processors that batch and tail sample then exporters to Jaeger Tempo and a vendor then the backends that store and query](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-6.webp)

The figure shows the pipeline as a stack because that is how the data flows: in at the top through receivers, down through the processing stages, out the bottom to backends. Why is this worth a separate process rather than library calls in each app? Several reasons that compound. First, **decoupling**: your apps speak only OTLP to the collector; switching from Jaeger to Tempo to a commercial vendor is a collector config change, not a redeploy of forty services. Second, **batching and buffering**: the collector aggregates spans from many services and ships them in efficient batches, and it can buffer when the backend is briefly unavailable so a backend hiccup does not drop your telemetry or block your apps. Third, **tail-based sampling**, which we will see *requires* a component that sees whole traces — something an individual app, which only sees its own spans, fundamentally cannot do. Fourth, **cardinality and PII control** in one enforceable place: you can strip a high-cardinality attribute or redact a credit-card number in the collector's processors, so a careless `span.set_attribute("user.email", ...)` in one service gets scrubbed before it ever reaches storage. The collector is where the practitioner's discipline lives.

```yaml
# otel-collector-config.yaml — the receive -> process -> export pipeline for ShopFast.
receivers:
  otlp:
    protocols:
      grpc:                     # apps push spans here on :4317
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:                        # batch spans before export: fewer, bigger network calls
    send_batch_size: 1024
    timeout: 5s
  memory_limiter:               # refuse new data before the collector OOMs under a flood
    check_interval: 1s
    limit_mib: 1500
  attributes/scrub:             # cardinality + PII control, enforced centrally
    actions:
      - key: user.email         # drop a high-risk, high-cardinality attribute everywhere
        action: delete
      - key: http.url           # strip query strings that carry unbounded values
        action: update
        value: "redacted"

exporters:
  otlp/tempo:                   # primary backend
    endpoint: tempo:4317
  jaeger:                       # a second backend, for free, no app change
    endpoint: jaeger-collector:14250

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes/scrub]
      exporters: [otlp/tempo, jaeger]
```

Read the `service.pipelines.traces` block as the wiring diagram: OTLP comes in, passes through the memory limiter (back-pressure protection), then batching, then attribute scrubbing, then fans out to both Tempo and Jaeger. Adding a third backend or changing a sampling rule is an edit here, not a fleet-wide code change. This central choke point is also where you most directly apply the lessons of [handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation): the `memory_limiter` is the collector refusing to take down itself (and back-pressure the apps) when the backend is slow, which is the same fail-the-cheap-thing-not-the-expensive-thing instinct applied to your telemetry plane.

## Sampling: you cannot store every trace

Here is the cost reality that everything in production tracing bends around: at scale you cannot afford to store every trace. A single moderately-detailed trace through the ShopFast checkout might be ten to twenty spans, each a few hundred bytes of attributes — call it 5 to 10 kilobytes per trace. That sounds trivial until you multiply by traffic. The arithmetic is brutal, and it forces a decision: **sampling** — keeping a representative subset of traces and discarding the rest. The question is not *whether* to sample but *how*, and the two strategies make opposite trade-offs that you must understand to choose.

**Head-based sampling** decides at the *start* of the trace, at the root span, whether to record it — usually a simple probability ("keep 1% of traces") propagated to every downstream service via that sampling flag in the `traceparent` header. Its virtue is that it is cheap and stateless: the decision is made once, instantly, at the root, and every service obeys the flag, so a not-sampled trace generates almost no overhead anywhere. Its fatal flaw is that it decides *blind* — at the moment the request arrives, you do not yet know whether it will error or be slow, so a 1% head sample keeps 1% of your *errors* and 1% of your *slow* requests, which are exactly the traces you most want and which are rare. Head sampling optimizes for cost and uniform coverage; it is poor at preserving the interesting tail.

**Tail-based sampling** decides at the *end* of the trace, after all spans have arrived and you can see the whole thing — its total latency, whether any span errored, which services it touched. Now you can keep *every* error, *every* slow trace, and a small percentage of the boring fast successes. This is what you actually want: full visibility into problems, cheap storage for the routine. Its cost is that it requires *buffering*: something has to hold all the spans of an in-flight trace until the trace completes before it can decide, and that something has to see *all* the spans of a trace, which means routing all spans of a given trace id to the same collector instance (consistent hashing by trace id, an idea the sibling [database partitioning post](/blog/software-development/database/database-partitioning-and-sharding) treats in its general form). Tail sampling buys you the interesting traces at the price of collector memory and a more complex deployment.

![A matrix comparing head based and tail based sampling across when the decision is made whether it sees the outcome whether it keeps all errors its memory cost and its best fit](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-7.webp)

The matrix lays the trade-off out row by row. Head decides at the root, blind, near-zero memory, keeps errors only by luck — best when load is uniform and you mostly want a representative sample cheaply. Tail decides after the trace ends, sees the full outcome, keeps all errors by policy, costs collector memory to buffer — best when you specifically want to keep the slow and erroring traces, which in practice is almost always what an on-call engineer reaches for. The senior answer is frequently *both*: a modest head sample to bound the firehose before it even reaches the collector, then tail sampling on what survives to keep the interesting subset. Here is a tail-sampling policy in collector config that encodes exactly the decision tree we want.

```yaml
# Tail-sampling processor: keep all errors, keep all slow traces, sample the rest at 1%.
processors:
  tail_sampling:
    decision_wait: 10s          # buffer a trace up to 10s for its spans to all arrive
    num_traces: 100000          # max in-flight traces held in memory
    policies:
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }   # any errored span -> keep the whole trace
      - name: keep-slow
        type: latency
        latency: { threshold_ms: 1000 }           # total trace > 1s -> keep
      - name: sample-rest
        type: probabilistic
        probabilistic: { sampling_percentage: 1 } # everything else: keep 1%
```

![A decision tree for tail sampling that asks if any span errored then keep all errors otherwise asks if latency exceeds one second then keep all slow else sample one percent of fast successes](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-9.webp)

The tree draws the same policy as the decision it is: when a trace completes, ask "did any span error?" — if yes, keep it (100% of errors). If no, ask "was it slower than a second?" — if yes, keep it (100% of slow). Otherwise it is a fast success, the boring 99% of your traffic, and you keep 1% of those just to have a representative sample of healthy behavior. This single policy is the difference between a tracing bill that scales linearly with traffic and one that scales with your *problem rate*, which grows far slower. Let us put real numbers on it.

#### Worked example: trace storage cost at 50k requests per second

ShopFast peaks at 50,000 requests per second. Assume each request produces one trace averaging 8KB of stored spans after compression. **Full sampling** — store every trace — generates 50,000 × 8KB = 400 MB/s = roughly 34.5 TB/day. At a representative managed-tracing price of around \$0.10 per GB ingested, that is about \$3,450 per day, or **\$1.05 million per year**, just for traces, and you will almost never look at 99.99% of them. That is the bill that makes finance ask hard questions. Now **1% head sampling**: you keep 1% of traces, blindly. Storage drops 100× to about 345 GB/day, roughly \$10,500 per year — a 100× saving. But you also kept only 1% of your errors and slow requests, so when the 2:47 incident hits, the specific slow traces you need have a 99% chance of having been thrown away, and you are debugging from the 1% that happened to survive. Now **tail sampling** with the policy above. Suppose your real error rate is 0.5% and your slow-trace (>1s) rate is another 1.5%, so 2% of traces are "interesting" and kept at 100%, plus 1% of the remaining 98% kept as a healthy baseline. You store 2% + (0.98 × 1%) ≈ 2.98% of traces — about 1.03 TB/day, roughly \$31,000 per year. That is 34× cheaper than full sampling, three times the cost of blind head sampling — and *you kept every error and every slow trace*. You are paying a small premium over head sampling to never again be missing the trace you need. For most teams that is the obvious buy, and it is the number to bring to the budget conversation: tail sampling costs about 3% of full sampling while preserving 100% of the traces an engineer actually opens.

## The cardinality trap

Sampling controls trace cost. The analogous trap on the *metrics* side — and the one that has blown up more monitoring bills than any other single mistake — is **cardinality**. Cardinality is the number of distinct time series a metric generates, and it is the product of all the distinct values of all the *labels* (dimensions) attached to that metric. A metric is stored as one time series per unique combination of label values. `http_requests_total{service="payment", status="200"}` is one series; add `status="500"` and now it is two; add `method` with 5 values and you have 2 × 5 = 10; add `route` with 20 endpoints and you have 200. Each series is cheap individually, but the count *multiplies*, and a single ill-chosen label can multiply it into the millions.

The classic disaster is adding a **high-cardinality label** to a metric — most infamously `user_id`. It feels reasonable: "I want per-user request rates." But ShopFast has, say, 5 million users. The instant you add `user_id` as a label on `http_requests_total`, you have multiplied your series count by 5 million. A metric that was 200 series becomes 200 × 5,000,000 = *one billion* series. Your time-series database, which was happily storing a few hundred thousand series, now has to track a billion, most of which see a single data point and then go cold forever. Memory explodes, ingestion chokes, query latency goes through the roof, and your monthly bill — most metrics vendors price on active series — goes from hundreds of dollars to tens of thousands or simply falls over. The label that *seemed* like richer observability destroyed the metrics system.

The rule is sharp and worth memorizing: **labels on metrics must be low-cardinality and bounded** — `service`, `status_code`, `method`, `region`, `route` (a *templated* route like `/orders/:id`, never the raw URL with the id in it). High-cardinality data — `user_id`, `order_id`, `trace_id`, raw URLs, full error messages — belongs on **logs and span attributes, not metric labels**, because logs and spans are stored per-event (you pay once per event, not per-unique-combination-forever) while metrics pay the multiplicative series cost. This is precisely the division of labor the three pillars are *for*: you want per-user detail, you put `user_id` on the span and the structured log, where it is just a field on a row; you never put it on a metric label, where it detonates. The cardinality trap is the single most important reason juniors must understand the pillars are not interchangeable.

#### Worked example: the cardinality blowup from one label

ShopFast runs a healthy metrics setup: about 40 services, each emitting RED metrics. Take request count with labels `service` (40), `route` (~25 per service), `method` (4), `status_code` (~8). Per service that is 25 × 4 × 8 = 800 series; across 40 services, about 32,000 active series. The metrics vendor charges roughly \$0.10 per series per month at this tier, so that is about \$3,200/month — a normal, defensible bill. Now a well-meaning engineer adds `customer_id` to the request metric to build a per-customer dashboard. ShopFast has 5,000,000 customers, but realistically only ~500,000 are active in a given month and actually generate a metric data point. The series count is now 32,000 × 500,000 = **16 billion** series. Even at a steep volume discount the bill is now measured in *millions* per month, the ingestion pipeline cannot keep up, and queries that used to return in 200ms now time out. The fix is not "tune the database" — it is *delete the label*. Move `customer_id` to the span attribute and the structured log, where serving the same per-customer question costs one field per event instead of one immortal series per customer. The lesson: **a single high-cardinality metric label can multiply your cost by five or six orders of magnitude, and the only real defense is to never add it in the first place.** The collector's `attributes` processor and the metrics SDK's view configuration are where you enforce a label allow-list so this cannot happen by accident.

## Correlating the three pillars by trace id

The three pillars are most powerful not in isolation but *stitched together*, and the thread that stitches them is the trace id. We have seen it stamped on logs; the same id ties metrics to traces through a mechanism called **exemplars**. An exemplar is a sample data point attached to a metric bucket that carries the trace id of one request that landed in that bucket. So when your p99 latency histogram shows a spike in the 500ms–1s bucket, the exemplar attached to that bucket gives you the trace id of an *actual* request that took that long — and you click straight from the metric spike into the trace waterfall of a real slow request. That is the bridge from monitoring (the metric told you *that* it is slow) to observability (the trace tells you *where*), and it is the single most valuable correlation in a mature setup.

![A branching graph showing an alert flowing to a metric exemplar carrying a trace id then to the trace waterfall then to the slow payment span then to that span logs and the trace and logs converging on the root cause](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-8.webp)

The figure is the 2:47 incident debugged the way a correlated system lets you debug it, as one unbroken chain. The alert fires on a p99 metric. The metric carries an exemplar with a trace id, so you pivot directly to the trace waterfall. The waterfall shows the slow payment span. You click the payment span and see *its logs* — filtered to that span by trace id and span id — which read "PSP retry x3, timeout." Now you know not just *where* (payment's PSP call) but *what* (it retried three times against a timing-out processor) and the root cause falls out in minutes. Contrast this with the uncorrelated path, where each pillar is a separate tool with a separate query language and you copy-paste timestamps between four tabs hoping the clocks line up. The structured log line that makes this work is mundane but load-bearing.

```json
{
  "ts": "2026-06-15T14:47:02.317Z",
  "level": "error",
  "service": "payment-service",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "msg": "PSP call exhausted retries",
  "psp.provider": "stripe",
  "psp.attempts": 3,
  "psp.last_status": 504,
  "order.id": "8821"
}
```

The `trace_id` and `span_id` fields are the entire trick. Your logging library reads them out of the current OTel context automatically (every mature OTel language SDK offers a log-correlation hook), so every log line emitted *while a span is active* carries that span's ids with zero per-call-site effort. Then your log backend and your trace backend share that key, and "show me the logs for this span" is a single filtered query. The before-and-after of having versus not having this is stark.

![A before and after comparison contrasting four log streams with no shared key that take thirty minutes to correlate by timestamp against logs that all carry one trace id so you click a span and see its logs in seconds](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-2.webp)

The figure contrasts the two worlds. On the left, the world most teams actually live in: four services each logging to their own stream with no shared key, so correlating "which payment log goes with this gateway request" means eyeballing timestamps across four tabs and guessing — a half-hour archaeology dig per incident, performed under pressure at 2:47am. On the right, the world this post is teaching you to build: one trace id stamped on every log in every service, so you click a span in the waterfall and its logs appear instantly, filtered by that id. The only difference between the two pictures is whether somebody wired the trace id into the logger — a few lines of configuration per service — and that few lines is the difference between a four-minute incident and a two-hour one. This is the cheapest, highest-leverage observability upgrade available, and it is the first thing I do on any service that lacks it.

## The decision matrix: which pillar answers which question

Everything so far comes down to a single allocation decision you make constantly: a question arrives, and you must know which pillar answers it cheaply. Getting this allocation right is most of operational maturity, because the failure mode is using the wrong pillar — building a metric for something that needed a trace (and learning nothing about *where*), or scanning logs for something a metric would have answered instantly (and burning your log bill). The matrix below is the cheat sheet.

![A matrix comparing logs metrics and traces across the question each answers their storage cost their cardinality limit and when to reach for each](/imgs/blogs/distributed-tracing-and-observability-with-opentelemetry-5.webp)

The figure rows out the four properties that decide the allocation. *What question it answers*: logs say what exactly happened, metrics say how often and how bad, traces say where in the request. *Storage cost*: logs are expensive per event (you pay per line), metrics are cheap because they aggregate, traces are high but tamed by sampling. *Cardinality*: logs and span attributes tolerate unbounded high-cardinality fields (per-event storage), metric labels must be strictly low-cardinality (the multiplicative series trap), traces tolerate per-span detail. *When to reach for it*: logs for forensic detail on a specific event, metrics for dashboards and alerts on aggregate health, traces for "this cross-service request is slow and I need to know where." Read the matrix as a routing table. The same routing in a markdown table for the questions you will actually face:

| Question you are asking | Reach for | Why not the others |
| --- | --- | --- |
| "Is the payment service healthy right now?" | Metrics (RED) | Cheap, instant, alertable; logs/traces too detailed |
| "How many checkouts errored in the last hour?" | Metrics (error counter) | An aggregate count, exactly what metrics are for |
| "*Where* did this slow checkout spend its time?" | Trace (waterfall) | Only traces preserve the cross-service call tree |
| "What exactly did order 8821's payment do?" | Logs (by trace_id) | Per-event forensic detail; metrics threw it away |
| "Are enterprise customers' checkouts slower?" | Traces grouped by `customer.tier` attribute | High-cardinality slice; never a metric label |
| "Did p99 latency regress after the deploy?" | Metrics, then trace via exemplar | Metric detects, exemplar pivots you to the real slow trace |
| "Which of 8 services is the bottleneck?" | Trace (waterfall) | The defining trace question |

The decision matrix is also the answer to the trade-off the kit demands you state explicitly: you are always trading **cost against fidelity**. Metrics are the cheapest and lowest-fidelity (aggregates, no per-request detail). Logs are high-fidelity and high-cost (every event, every field). Traces are high-fidelity and *would* be high-cost except sampling lets you buy down the cost while keeping the fidelity *where it matters* (the errors and the slow ones). The senior move is to push each question to its cheapest sufficient pillar: alert on metrics, drill with traces, confirm with logs, and never pay log prices for a metrics question or lose trace fidelity to a metrics-shaped tool.

## Optimization: making observability production-grade

A naive OTel setup works in a demo and falls over in production, usually in one of three ways: it floods the network exporting one span per call, it bankrupts you storing every trace, or it detonates your metrics bill with a careless label. Each has a concrete fix with a measurable win, and tuning these is what separates "we have tracing" from "we have tracing we can afford and that helps."

**Batch your exports.** The single biggest efficiency mistake is the `SimpleSpanProcessor`, which exports each span the instant it ends — one network round-trip per span. At 50k requests/second with 10 spans each, that is 500,000 export calls per second per fleet, drowning the collector in connection overhead. The `BatchSpanProcessor` (shown in the SDK snippet earlier) buffers spans and ships them in batches of, say, 512 every couple of seconds. The win is dramatic: 500,000 individual exports collapse to roughly 1,000 batched exports per second, a 500× reduction in network calls, with a corresponding drop in CPU spent on serialization and TLS handshakes. The cost is a few seconds of export latency (a span ends now but ships a couple seconds later) — irrelevant for telemetry, which is not real-time. Always batch.

**Sample with intent, not panic.** We covered the math: full sampling at 50k req/s is ~\$1M/year; tail sampling that keeps every error and slow trace plus 1% of successes is ~\$31k/year — a 34× saving with *no* loss of the traces you actually open. The optimization is not "sample less" (that loses the errors); it is "sample *smartly*" — head-sample to bound the firehose, tail-sample to keep the signal. Measure the win two ways: storage bill (down 34×) and *coverage of incidents* (did the trace you needed survive? With tail sampling, yes, by policy).

**Control cardinality at the source and at the collector.** A single bad metric label can 1,000,000× your series count. The fix is a label allow-list enforced in two places: the metrics SDK (a `View` that drops disallowed attributes before they become labels) and the collector's `attributes` processor (a delete action as a backstop). The win is the difference between a \$3,200/month metrics bill and a \$3M/month one. Measure it by tracking *active series count* as a first-class operational metric — alert when it grows faster than your service count, because runaway series is almost always a leaked high-cardinality label, and catching it the day it ships beats discovering it on the invoice.

**Tune the collector for throughput and safety.** Set `memory_limiter` so the collector sheds load before it OOMs rather than crashing and losing all in-flight telemetry — the same load-shedding discipline the sibling on [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) applies to request traffic, here applied to the telemetry plane. Run the collector as a horizontally-scaled deployment behind a load balancer that hashes by trace id (so tail sampling sees whole traces on one instance). Size `num_traces` and `decision_wait` to your real trace duration — too short and slow traces get decided before their spans arrive (you lose the slow ones you were trying to keep); too long and memory balloons. A good starting point: `decision_wait` at roughly 2× your p99.9 trace duration.

#### Worked example: the cost of getting all three optimizations wrong vs right

Put the three together for ShopFast at 50k req/s. *Naive*: `SimpleSpanProcessor` (500k exports/s, collector CPU pegged, dropping spans under load), full sampling (~\$1M/year traces), and a leaked `user_id` metric label (~\$3M/year metrics). Total observability bill heading toward \$4M/year, and the system is *also* unreliable because the collector is overwhelmed. *Tuned*: `BatchSpanProcessor` (1k exports/s, collector idle), tail sampling (~\$31k/year traces), bounded metric labels (~\$38k/year metrics). Total roughly \$70k/year — a **57× reduction** — and the system is *more* reliable because nothing is overwhelmed, *and* you kept every error trace. The lesson the numbers teach is that observability optimization is not penny-pinching; it is the difference between a tool you can afford to run at full fidelity and one you are forced to cripple or switch off when the bill arrives. The cheapest observability is the kind that survives the finance review.

## Stress-testing the design

The kit demands we pose the real problems and stress the design against them. Here are the three that actually page you, each run through the architecture we have built.

**"A request is slow but which of 8 services is it?"** This is the 2:47 incident and the canonical justification for tracing. Without traces you bisect by hand — add timing logs to one service, deploy, wait for it to recur, repeat — a process that can take days for an intermittent slowdown. With traces and exemplars, the path is: alert fires on the gateway p99 metric → click the exemplar → land in a real slow trace's waterfall → read which span is widest → done, in minutes. The stress test it survives: even if the slowness is *intermittent* (one in a thousand requests), tail sampling guarantees the slow traces are kept, so you have examples to open. The stress test it *fails* if you got sampling wrong: 1% head sampling would have thrown away 99% of those rare slow traces, and you would be debugging from luck. This is why sampling strategy is not a detail — it determines whether the tool works when you need it.

**"Traces cost too much to store."** The naive reaction is to turn tracing off or crank head sampling down to 0.1%, both of which destroy the tool's value the moment you most need it. The design's answer is tail sampling: keep 100% of errors and slow traces (the few percent you actually open), sample the boring 99% down hard. The worked example showed this drops the bill 34× *without* losing the interesting traces. The deeper stress test: what about a *traffic spike* — Black Friday at 5× normal load, 250k req/s? Full sampling would 5× your already-huge bill; tail sampling scales with your *problem rate*, not raw traffic, and your problem rate does not rise 5× just because traffic did (it might rise less, or proportionally to errors). Plus the `memory_limiter` and horizontal collector scaling absorb the volume. Tail sampling is *more* important under load, not less.

**"Logs aren't correlated — we can't tell which logs go with which request."** This is the before-picture from the correlation section, and at 2:47 it is the difference between a four-minute and a two-hour incident. The design's answer is stamping `trace_id` and `span_id` on every structured log via the OTel log-correlation hook, so every log line is joinable to its trace. The stress test: what about logs from *async* work — a message consumed off a queue minutes after the request that produced it? Here the trace context has to ride *on the message* (in message headers, the same way it rides in HTTP headers), so the consumer can continue the trace or link to it. This is the exact discipline the message-queue sibling on [the anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) implies for any production queue: the message is a network hop too, and the trace context must cross it deliberately, or the trace breaks at the broker and your async work becomes an un-correlatable orphan.

## Case studies

**Google's Dapper — where this all started.** Distributed tracing as we practice it descends almost directly from Google's internal system **Dapper**, described in a widely-read 2010 paper. Dapper introduced the now-standard vocabulary — traces, spans, the propagation of a trace context across RPC boundaries — and, critically, established two principles that OTel still honors. First, **low overhead through sampling**: Dapper sampled aggressively (a small fraction of requests) precisely because Google's scale made full tracing impossible, the same cost reality this post is built around. Second, **ubiquitous deployment through transparent instrumentation**: Dapper got its coverage by instrumenting Google's *shared RPC and threading libraries* rather than asking every team to add tracing by hand — the direct ancestor of OTel's auto-instrumentation. The lesson Dapper teaches is that tracing only works if it is *cheap to adopt* (auto-instrument the common libraries) and *cheap to run* (sample), and a tracing system that fails either test does not get used.

**Uber and Jaeger.** Uber built and open-sourced **Jaeger** (now a CNCF project, and a common OTel backend) to trace requests across a fleet that grew into the thousands of microservices — the same fleet whose organizational sprawl prompted Uber's **DOMA** (Domain-Oriented Microservice Architecture) reorganization. At that service count, a request can touch dozens of services, and Uber's public engineering writing repeatedly makes the point that *no human can hold the call graph in their head* — tracing is not a debugging luxury but the only way to understand the system's behavior at all. The lesson: tracing's value grows super-linearly with service count, because the number of possible cross-service paths explodes, and past a few dozen services there is no substitute. If you are at five services, tracing is helpful; at fifty, it is the only thing standing between you and flying blind.

**The OpenTelemetry convergence.** Before OTel, the ecosystem was fragmented: OpenTracing (an API standard) and OpenCensus (Google's instrumentation libraries) competed, vendors each had proprietary agents, and switching observability vendors meant re-instrumenting your entire fleet. OpenTelemetry, formed in 2019 from the merger of OpenTracing and OpenCensus, deliberately solved the *vendor lock-in* problem by separating instrumentation (the API you code against, stable and neutral) from the backend (configured, swappable). The practical lesson for a senior making a build-versus-buy decision: **instrument with OTel even if you use a commercial backend**, because the OTel API decouples your forty services' instrumentation from your vendor contract, so re-negotiating or switching vendors becomes a collector config change instead of a multi-quarter re-instrumentation project. The teams that learned this the hard way had instrumented natively against a vendor SDK and discovered the switching cost was effectively a rewrite.

**The trace that found the bug.** A recurring pattern in production engineering write-ups (Shopify, Stripe, and others have published variants) is the intermittent latency bug that *only* yields to tracing. The shape is always similar: a small fraction of requests are slow, metrics show only an elevated p99 with no obvious cause, and logs across services do not obviously correlate. The resolution comes from opening the slow traces (kept by tail sampling) and noticing a pattern in the waterfall — a particular downstream span that is occasionally 50× its normal width, or a sequence of calls that should be parallel running in series for a subset of requests, or a retry storm visible as repeated child spans. The lesson is the one this whole post serves: *the trace makes visible the thing no aggregate could*, because the bug lives in the *shape of an individual request's journey*, and only the trace preserves that shape. The teams that resolve these bugs in an afternoon are the ones who had tail sampling keeping the slow traces; the teams that spend a week are the ones who sampled blind and threw the evidence away.

## When to reach for this (and when not to)

Observability is a cost — engineering time to instrument, money to store, complexity to operate — so be honest about when it pays. **Metrics (RED/USE) are non-negotiable from day one, at any scale.** Even a single service needs request rate, error rate, latency, and resource utilization; they are cheap, they power your alerts, and there is no scale at which you should run blind. Start here always.

**Structured logging with trace ids is the second thing, and it is nearly free.** The moment you have *two* services that call each other, uncorrelated logs start costing you debugging time, and stamping a trace id on every log line is a few lines of config per service. Do this early; it pays for itself in the first multi-service incident.

**Distributed tracing's value scales with your service count and your call-graph depth.** At one or two services, a profiler and good logs often suffice, and full tracing may be over-engineering — the request does not cross enough boundaries for the waterfall to reveal much you could not get from logs. The crossover is somewhere around the point where a single request touches *four or more* services, or where you have enough services that no one person holds the call graph in their head. Past that point — which most real microservices fleets are well past — tracing moves from helpful to essential, and the 2:47 incident is unsolvable without it. If you are still in a monolith or a handful of services, the honest advice is the same as the series' recurring theme, echoed in [what are microservices and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them): do not pay the distributed-tracing tax until you have the distributed-systems problem that requires it. But the day you cross into "which of eight services is slow," you needed it yesterday — so instrument with OTel *before* you cross, because the API decoupling means the cost of being ready is low and the cost of retrofitting under incident pressure is high.

**Tail sampling and serious cardinality control are scale problems** — you do not need them at 100 req/s, where you can afford to store everything and your metrics have no cardinality pressure. They become essential as you climb toward thousands of requests per second, where the storage math turns into real money. The sequencing for a growing system: metrics from day one, correlated logs as soon as you have two services, basic tracing (head sampling) as you cross four services, tail sampling and cardinality discipline as traffic and cost grow into six figures a year.

## Key takeaways

- **Monitoring answers the questions you anticipated; observability lets you ask new ones.** Build dashboards for the known signals, but instrument with rich, dimensional, correlatable data so you can slice it a new way during the incident you did not predict.
- **The three pillars are not interchangeable.** Logs say *what exactly*, metrics say *how often and how bad*, traces say *where in the request*. Push every question to its cheapest sufficient pillar and never ask metrics to do logs' job.
- **A trace is a tree of spans stitched by trace id and parent links** — and no single service ever sees the whole trace. The collector reassembles it, which is why context propagation across every hop is make-or-break.
- **The `traceparent` header is the whole propagation mechanism.** Drop it on one hop — a hand-rolled HTTP client, a queue with no context on the message — and the trace silently splits into orphans at exactly the boundary you most need to see.
- **Read the waterfall by width and nesting.** The widest span is the suspect; a wide span with thin children hides local slowness; sequential children that could be parallel are an optimization target.
- **Instrument with OpenTelemetry, run the Collector, decouple from your backend.** The OTel API frees you from vendor lock-in, and the collector is the one enforceable place for batching, sampling, and cardinality control.
- **You cannot store every trace at scale; tail-sample with intent.** Keep 100% of errors and slow traces, sample the boring successes hard — roughly 3% of full-sampling cost while preserving 100% of the traces an engineer actually opens.
- **The cardinality trap is the metrics analog of unbounded cost.** Never put `user_id`, `order_id`, or a raw URL on a metric label; a single high-cardinality label can multiply your series count a million-fold. High-cardinality data lives on spans and logs, where you pay per event, not per immortal series.
- **Correlate by trace id and the pillars become one tool.** Stamp the trace id on every log, attach exemplars to your metrics, and you can pivot from an alert to a real slow trace to its span's logs to the root cause in minutes instead of hours.
- **The senior bar: every request traceable, every log correlated, the bill survivable.** Junior observability is "we installed a vendor." Senior observability is a deliberate set of decisions about what to emit, at what cardinality, at what sampling rate, correlated by what key.

## Further reading

- Benjamin H. Sigelman et al., *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure* (Google, 2010) — the founding paper; read it for sampling and transparent instrumentation as first principles.
- *OpenTelemetry documentation* (opentelemetry.io) — the canonical reference for the API, SDK, Collector, OTLP, and language-specific auto-instrumentation.
- *W3C Trace Context* specification — the definitive description of the `traceparent` and `tracestate` headers your propagation depends on.
- Cindy Sridharan, *Distributed Systems Observability* (O'Reilly) — the book that popularized the three-pillars framing and the monitoring-versus-observability distinction.
- Charity Majors, Liz Fong-Jones, and George Miranda, *Observability Engineering* (O'Reilly) — high-cardinality, event-based observability and why pre-aggregation limits you.
- The sibling [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) — the monitoring layer this post leans on: which signals to alert on and how to set thresholds.
- The sibling [debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production) — putting these pillars to work under incident pressure, with the trace as your primary instrument.
- The sibling [testing microservices from unit to chaos](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos) — how you verify resilience and observability *before* the 2:47 page, not during it.

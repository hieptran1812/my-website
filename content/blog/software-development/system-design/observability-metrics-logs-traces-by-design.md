---
title: "Observability by Design: Metrics, Logs, and Traces That Actually Help"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn to design observability into a system the way a senior does — knowing exactly what metrics, logs, and traces are each FOR, alerting on symptoms not causes, controlling the cardinality that blows up your bill, and using correlation IDs and traces to find root cause at 3am in minutes instead of hours."
tags:
  [
    "system-design",
    "observability",
    "metrics",
    "logging",
    "distributed-tracing",
    "opentelemetry",
    "slo",
    "architecture",
    "distributed-systems",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/observability-metrics-logs-traces-by-design-1.webp"
---

It is 3:07 a.m. and your phone is buzzing. The page says `checkout p99 latency > 2s`. You are now the person standing in front of a distributed system you half-remember, with a customer-facing problem, a manager who will ask "what happened" in five hours, and a single question that determines whether you go back to sleep in ten minutes or in two hours: **can I see what my system is doing right now?** Not what it was doing in aggregate last week. Not what some dashboard's vanity chart of total requests looks like. Can I take *this specific slow checkout*, follow it across the six services it touched, and find the one hop that ate two seconds? If the answer is yes, you are debugging. If the answer is no, you are guessing — grepping logs on six hosts by hand, correlating timestamps in your head, restarting services to "see if it helps," and praying. The difference between those two 3am experiences is not talent or luck. It is whether somebody designed observability into the system before the incident, or bolted monitoring on after.

This post is about treating observability as a **design property**, not an afterthought you buy from a vendor and forget. The core claim a senior internalizes is simple and uncomfortable: *you cannot operate what you cannot see, and you only see what you instrumented on purpose*. Observability is the property that lets you ask new questions about your system's internal state from the outside — including questions you did not anticipate when you wrote the code — and get answers from the data you are already emitting. That last clause is the whole game. The questions you will need to answer at 3am are, almost by definition, the ones you did not foresee, which means observability is fundamentally about preserving enough *detail* and *correlation* in your telemetry that a future you, under pressure, can slice it in a way present-you never foresaw. Figure 1 lays out the three signal types we have to combine to make that possible, and the property each one wins.

![Matrix comparing metrics, logs, and traces across cost per event, cardinality tolerance, latency to insight, retention, query power, and cross-service causality](/imgs/blogs/observability-metrics-logs-traces-by-design-1.webp)

The three signals — metrics, logs, and traces, often called the three pillars — are not three interchangeable ways to see the same thing. Each one is good at a specific job and bad at the others, and the senior skill is knowing which question each one answers, what each one costs, and how to wire them together so a metric spike leads you to a trace leads you to a log line. Get the wiring right and a 3am page becomes a two-minute click-through. Get it wrong — emit everything as logs, slap an unbounded `user_id` on every metric, sample your traces so aggressively the slow request is never captured — and you will pay a fortune for telemetry that still cannot answer the question that woke you up. By the end of this post you should be able to design the metrics-logs-traces strategy for a real service, alert on the four golden signals instead of on causes, control metric cardinality so your bill does not explode, choose a trace sampling strategy that actually keeps the requests you care about, and reason about all of it the way you would defend it in a design review.

A note on where this sits relative to the rest of this blog. This is the architect's decision layer: not how a time-series database stores samples internally or how a span exporter batches over gRPC, but *what to measure*, *what to alert on*, *where the cost traps are*, and *how to design for debuggability*. It pairs tightly with the reliability post on [SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — observability is how you *measure* the SLOs that reliability post teaches you to *set* — and with [articulating trade-offs with CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond), because every observability decision is a trade-off between cost, detail, and latency-to-insight.

## 1. Monitoring versus observability, and why the distinction is not pedantic

Start with the distinction that the word "observability" was coined to make, because if you treat it as a synonym for "monitoring with a fancier name" you will build the wrong thing. **Monitoring** is the practice of watching a predefined set of signals for predefined failure conditions: CPU over 80%, disk over 90%, error rate over 1%, the health check returning non-200. Monitoring answers **known-unknowns** — questions you knew to ask in advance, encoded as dashboards and alerts. It is necessary and it is not going away. The trouble is that monitoring can only catch the failures you predicted. The dashboard tells you CPU is fine, memory is fine, error rate is fine — and yet checkout is slow, because the failure is some combination of conditions nobody put on a dashboard: requests from one specific region, hitting one specific shard, for one specific product category, during a cache cold-start, that together produce a 2-second p99 that no single predefined metric reveals.

**Observability** is the property that lets you answer **unknown-unknowns**: questions you did not know to ask until the incident forced them. The formal definition borrows from control theory — a system is observable if you can determine its internal state from its external outputs — but the practical definition is more useful: a system is observable if, when something goes wrong in a way you did not anticipate, you can figure out *what* and *why* from the telemetry you are already emitting, without shipping new code and waiting for the bug to recur. The litmus test is whether you can slice your telemetry along a dimension you did not pre-aggregate. "Show me p99 latency" is monitoring. "Show me p99 latency *for requests to shard 7, from the EU region, on the checkout endpoint, where the user is on the enterprise tier*" — and getting an answer without having built that exact dashboard in advance — is observability.

The mechanism that separates the two is **high-cardinality dimensions**. Cardinality is the number of distinct values a dimension can take. `http_status` has low cardinality (a handful of values). `region` has low cardinality (a dozen). `user_id` has cardinality in the millions; `request_id` is effectively infinite — every request a unique value. Monitoring lives in the low-cardinality world because that is what cheap pre-aggregated metrics can afford. Observability requires high cardinality, because the dimension that explains your incident — the specific user, the specific request, the specific shard-region-tier combination — is almost always a high-cardinality slice you did not pre-compute. This is the single most important architectural fact in observability: **the signals differ primarily in how much cardinality they can carry and what that cardinality costs.** Metrics are cheap precisely because they refuse high cardinality. Logs and traces are expensive precisely because they embrace it. Everything else follows from that trade.

Why is this not pedantic? Because it changes what you build. A team that thinks "observability = more dashboards" buys a monitoring tool, builds forty dashboards, and is still helpless against the novel failure. A team that thinks "observability = preserve high-cardinality detail and correlation" instruments structured logs with a request id on every line, propagates a trace context across every service boundary, links metric exemplars to traces, and — crucially — keeps enough raw detail that they can ask a brand-new question at 3am and get an answer. The first team has spent money. The second team can actually operate the system. That is the difference the rest of this post is about making real.

## 2. The three pillars: what each is actually FOR

The fastest way to waste money on observability is to use the wrong pillar for the job — to record as logs what should be a metric (and pay 50x), or to try to alert off traces (and miss the spike because you sampled it away). Each pillar has a job it is uniquely good at. Let me make the jobs concrete.

**Metrics are for cheap, aggregate, always-on measurement and alerting.** A metric is a number sampled over time — a counter that only goes up (requests served, errors), a gauge that goes up and down (queue depth, memory in use), or a histogram that buckets a distribution (request latency). The defining property is that metrics are *pre-aggregated at emit time*: instead of recording every request, you increment a counter, and at the end of each scrape interval you have one number per series. That aggregation is why metrics are cheap — a service handling 50,000 requests per second emits not 50,000 data points per second but a handful of counter increments that collapse into one sample per series per scrape. It is also why metrics *cannot* answer "what happened to *this* request": the individual events were summed away the instant they were recorded. Metrics tell you *that* something is wrong (error rate jumped, p99 climbed) and they tell you fast and cheap, which is exactly what you want a pager built on. They cannot tell you *why* for any specific request. That is not a defect; it is the trade that makes them affordable.

**Logs are for high-cardinality, per-event detail.** A log line is a record of a single discrete event: a request was received, a payment failed, a cache was evicted, with whatever context the developer attached. Logs carry arbitrary high-cardinality fields — the user id, the request id, the full error message, the SQL that failed — which is exactly what you need when you have narrowed a problem to a specific event and want to know everything about it. The price is volume: a busy service emits an enormous number of log events, each one a full record that must be transported, indexed, and stored, and **logs are typically 10 to 100 times more expensive per event than metrics** because nothing is aggregated away — every event is a full row you pay to move, index, and retain. Logs answer "*why did this specific thing happen*," and they answer it with rich detail, but they do not answer it cheaply, and they do not answer "*is the system healthy*" without expensive aggregation queries over millions of lines.

**Traces are for causality across services.** A trace records the path of a single request as it flows through a distributed system, as a tree of **spans** — one span per unit of work (an HTTP handler, a database query, a downstream RPC) — each with a start time, a duration, and a parent. The trace's superpower is *causality across service boundaries*: it shows you that the slow checkout spent 9ms in the gateway, 18ms in inventory, and 1,840ms waiting on the payment service, all stitched together by one trace id even though those are four separate processes on four separate hosts. No metric can give you that — metrics are per-service aggregates with no notion of which gateway request caused which payment call. No log can give you that without heroic manual correlation. Traces answer "*where, across all my services, did this request spend its time, and what called what*." Their cost sits between metrics and logs, and they have a distinctive cost-control knob — sampling — that we will spend a whole section on. Figure 2 shows a single trace flowing across services and converging on a collector.

![Graph showing a request fanning from a gateway root span into checkout, payment, inventory, and database spans that all export to a collector](/imgs/blogs/observability-metrics-logs-traces-by-design-2.webp)

Here is the senior heuristic that ties them together, and it is worth memorizing: **meter what you alert on, log what you investigate, trace what you correlate.** You alert on metrics because they are cheap and always-on and fast. When an alert fires, the metric tells you a service or endpoint is unhealthy. You then open a trace to see *where across services* the latency or error lives. The trace points you at a specific span on a specific service, and you read the *logs* for that request — joined by the request id — to see the full detail of why that one operation failed. Metric → trace → log, narrowing from "something is wrong" to "this exact line of code on this exact request did this exact thing." A system designed so that path is one click at each step is observable. A system where each step is a manual context-switch into a different tool with no shared identifier is just three expensive silos.

## 3. Metrics done right: RED, USE, and the four golden signals

Metrics are where most teams either under-instrument (no useful signals, just CPU graphs) or over-instrument (ten thousand series nobody looks at, a five-figure bill). The discipline that fixes both is to instrument according to a *method* rather than by instinct, and there are two methods worth knowing because they cover the two kinds of thing you measure. Figure 3 lays out the four golden signals that sit at the top of the alerting hierarchy.

![Matrix mapping latency, traffic, errors, and saturation to what each measures, when to alert, and which metric type sources it](/imgs/blogs/observability-metrics-logs-traces-by-design-3.webp)

**RED is for request-driven services** — anything that handles requests and returns responses (an API, a web service, an RPC handler). RED stands for **Rate** (requests per second), **Errors** (the fraction that failed), and **Duration** (the latency distribution, which you record as a histogram so you can compute p50, p99, p999 after the fact). For every service and ideally every endpoint, you want those three. RED is the minimum viable instrumentation for a service, and it is shockingly powerful: rate tells you load, errors tell you correctness, and duration tells you the user experience. If you instrument nothing else, instrument RED per endpoint, because those three numbers catch the overwhelming majority of "is this service okay" questions.

**USE is for resources** — anything with finite capacity that work contends for (a CPU, a connection pool, a disk, a thread pool, a queue). USE stands for **Utilization** (the fraction of the resource that is busy), **Saturation** (how much work is queued waiting for the resource, the degree to which it is *over*-subscribed), and **Errors** (resource-level failures). USE is what catches the failures RED misses, because a resource can be saturated long before request errors show up — a connection pool at 100% utilization with a growing wait queue is the cause of the latency that RED will only show you as a *symptom*. Figure 5 lays out the two families and their fields as a tree, because the most common instrumentation mistake is measuring only RED (you see the symptom but not the cause) or only USE (you see CPU graphs but have no idea if users are suffering).

![Tree splitting metric methods into RED for services with rate, duration, and errors and USE for resources with utilization and saturation](/imgs/blogs/observability-metrics-logs-traces-by-design-5.webp)

Sitting above both methods are the **four golden signals** from Google's SRE practice — **latency, traffic, errors, saturation** — which are essentially RED's three plus USE's saturation, elevated to "the four things you should alert on for any user-facing system." The reason they are singled out is the single most important alerting principle in this whole post: **alert on symptoms, not causes.** Latency, traffic, errors, and saturation are *symptoms the user can feel* (or that predict imminent user pain, in saturation's case). CPU at 90% is a *cause* — and it may be a perfectly healthy state if the box is doing useful work and latency is fine. If you page on CPU, you will be woken up by non-problems and you will *miss* real problems where CPU is fine but latency is terrible. If you page on the golden signals, you get woken up exactly when users are actually hurting, regardless of which of a hundred possible causes produced it. The cause is what you diagnose *after* the page, with traces and logs. The symptom is what you page on.

Why symptoms over causes, stated as a senior would defend it in review: there are a small, stable number of symptoms (the user is slow, the user gets errors, the user can't connect) and an unbounded, ever-changing number of causes (a bad deploy, a slow query, a saturated pool, a noisy neighbor, a DNS hiccup, a cert expiry, a thundering herd). If you alert on causes you must enumerate the causes in advance — back to the known-unknowns trap of monitoring — and you will always be one novel cause behind. If you alert on symptoms, a brand-new cause you never anticipated still trips the same latency or error alert, because it still makes users slow or broken. Symptom-based alerting is observability's principle applied to paging: catch the failure by its user-visible effect, then use high-cardinality detail to find the never-before-seen cause.

#### Worked example: designing the RED metrics for a checkout flow

Let me make this concrete with the checkout flow that woke us at 3am. The flow is: gateway receives `POST /checkout`, calls the checkout service, which calls payment, inventory, and the orders database. We want RED metrics that will both alert us and, when an alert fires, immediately narrow the problem.

For each service-and-endpoint we emit three metrics. Rate is a counter: `http_requests_total{service, endpoint, method}` — incremented on every request, and the rate is its derivative. Errors is the same counter sliced by status: `http_requests_total{service, endpoint, status}` where we can compute error fraction as `status=~"5.."` over total. Duration is a histogram: `http_request_duration_seconds{service, endpoint}` with buckets chosen to bracket our SLO — if our p99 target is 300ms we want bucket boundaries densely around there (say 50ms, 100ms, 200ms, 300ms, 500ms, 1s, 2s) so the p99 estimate is accurate near the threshold we care about.

Now the cardinality budget. We have, say, 5 services and ~8 endpoints each = 40 service-endpoint pairs. For rate-and-errors we slice by `status` (~6 distinct values that actually occur) and `method` (~3), so the counter is roughly 40 × 6 × 3 = 720 series. For duration the histogram has ~9 buckets plus `+Inf`, sum, and count, so ~12 series per service-endpoint = 40 × 12 = 480 series. Total: about 1,200 active series for full RED across the whole checkout subsystem. That is *tiny* — a single modern Prometheus handles millions of series — and it is enough to alert on the golden signals and to immediately see, the moment we are paged, *which service and which endpoint* the latency lives in. The trace then tells us *where inside that service*. We will return to what happens if we get greedy and add `user_id` to these labels in section 8; for now, note that we got powerful per-endpoint RED for ~1,200 series because we kept every label low-cardinality. Here is the Prometheus query that computes the error-rate symptom we would alert on:

```promql
# Error rate (fraction of 5xx) for the checkout endpoint over 5 minutes.
# This is a SYMPTOM (errors users feel), not a cause (e.g. CPU).
sum(rate(http_requests_total{service="checkout", endpoint="/checkout", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{service="checkout", endpoint="/checkout"}[5m]))
```

And the p99 latency symptom, computed from the histogram:

```promql
# p99 request latency for checkout over 5 minutes, in seconds.
# histogram_quantile reads the bucketed _bucket series.
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket{service="checkout", endpoint="/checkout"}[5m])) by (le)
)
```

Notice both queries aggregate *away* the high-cardinality detail (no per-user, no per-request) precisely because metrics are the cheap aggregate layer. When one of these fires, we pivot to the trace and the log to recover the detail.

## 4. Logs done right: structured, leveled, and sampled

Logs are the pillar most teams get most wrong, because logging feels free — you write `log.info("processing order " + orderId)` and it works on your laptop — and the cost only shows up at scale, where it shows up brutally. Three disciplines turn logs from a liability into the high-cardinality detail layer they should be.

**Structured logging.** A log line should be a structured record — JSON or a key-value format — not a prose sentence. The difference is whether your logs are *queryable*. `log.info("processing order " + orderId + " for user " + userId)` produces a string you can only grep. `log.info("processing order", {"order_id": orderId, "user_id": userId, "request_id": reqId})` produces a record you can filter, aggregate, and join: "show me all logs where `user_id = 8821` and `request_id = abc` and `latency_ms > 500`." Structured logs are the difference between logs as a high-cardinality query layer and logs as a haystack you grep. Every log line should carry, at minimum, a timestamp, a level, a message, the **request id / trace id** (so it joins to the trace), and the relevant entity ids. This is non-negotiable for a system you intend to debug across services.

**Levels, and meaning them.** Log levels (`ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`) are a cost and signal-to-noise control, but only if you use them with discipline. `ERROR` means a human needs to know something failed in a way that needs action. `WARN` means something recoverable happened that might matter. `INFO` is the normal narrative of what the service did. `DEBUG`/`TRACE` is detail you want during an investigation but cannot afford always-on. The failure mode is everything-is-INFO (or worse, everything-is-ERROR), which destroys the signal — when every line is an error, no line is. A disciplined service runs at INFO in production, can be flipped to DEBUG for a specific component during an incident, and never pages a human off a log line that is not genuinely actionable.

**Sampling and the cost of "log everything."** At high request rates, logging every request at INFO is often the single largest line item in your observability bill, and most of those lines are never read. The senior move is to log *selectively*: always log errors and slow requests in full, but *sample* the successful-and-fast common case — keep 1% of the boring "200 OK in 12ms" lines, because the hundred-thousandth identical success line teaches you nothing the metric did not already. This is where the meter-versus-log decision bites hardest. If you want to know *how many* requests succeeded, that is a **metric** (a counter), not a log line per request — counting via logs is paying 50x to compute a number a counter gives you for free. Logs are for the *detail of a specific event*, not for counting events. The discipline is: **count with metrics, detail with logs.** Figure 4 contrasts the before-and-after of a logs-only debugging session against a trace-correlated one — the same incident, vastly different time-to-root-cause.

![Before-and-after comparing logs-only debugging that takes about forty minutes against trace-correlated debugging that takes about two minutes](/imgs/blogs/observability-metrics-logs-traces-by-design-4.webp)

Here is a structured-logging snippet in Python that emits JSON and always includes the trace context, so every line joins to its trace:

```python
import logging, json, time
from opentelemetry import trace

class JsonFormatter(logging.Formatter):
    def format(self, record):
        span = trace.get_current_span()
        ctx = span.get_span_context()
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "msg": record.getMessage(),
            # join key: every log line carries the trace and span id
            "trace_id": format(ctx.trace_id, "032x") if ctx.trace_id else None,
            "span_id": format(ctx.span_id, "016x") if ctx.span_id else None,
        }
        # attach any structured fields passed via `extra={"fields": {...}}`
        if hasattr(record, "fields"):
            payload.update(record.fields)
        return json.dumps(payload)

log = logging.getLogger("checkout")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log.addHandler(handler)
log.setLevel(logging.INFO)

# usage: structured, high-cardinality, joinable to the trace
log.info("charge failed", extra={"fields": {
    "order_id": "ord_8821",
    "user_id": "usr_5520",
    "amount_cents": 4999,
    "gateway_latency_ms": 1840,
}})
```

That single line is now a queryable record with a `trace_id` that links it directly to the slow span in figure 2. That link — log line to trace, by shared id — is the wiring that turns three pillars into one investigation.

## 5. Traces done right: spans, context propagation, and the gap that kills you

Tracing is the pillar that most directly answers the 3am question, and it has one prerequisite that, if you get it wrong, makes the whole thing useless: **context propagation.** A trace is only as complete as the chain of services that pass the context along. The mechanics are straightforward. When a request enters your system at the edge, you start a **root span** and generate a **trace id** — a unique identifier for the whole request. Each unit of work creates a **child span** with its own span id and the parent's span id, so the spans form a tree. The critical part is what happens at every service boundary: the calling service must inject the trace context (trace id, current span id, sampling decision) into the outgoing request — for HTTP, into the standard `traceparent` header — and the receiving service must extract it and continue the same trace. Do that at every hop and you get the complete tree in figure 2. Miss it at *one* hop and you get a **trace gap**: the trace simply ends at the service that did not propagate, and the slow downstream work it did is invisible, an orphan you cannot see.

This is the failure mode that bites hardest in practice, because it is silent. Everything looks instrumented — each service emits spans, the collector receives them, the UI shows traces — but one service, maybe an older one, maybe a third-party proxy, maybe a queue consumer that did not extract context from the message, fails to propagate. The result is a trace that looks complete but is missing the exact branch where your latency lives. I have watched a team spend an hour staring at a trace that showed checkout calling payment and payment "returning fast," concluding payment was fine — when in fact payment was calling a fraud-check service that *had not been instrumented to propagate context*, so the 1.8 seconds it spent there was a gap the trace never showed. The lesson: **a trace is a chain, and a chain is only as good as its weakest link.** Designing for tracing means designing for *propagation at every boundary*, including the async ones (message queues, where the context must ride in the message envelope — see how a [message system carries producer-to-consumer metadata](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers)) and the third-party ones.

Here is the OpenTelemetry span instrumentation for the checkout service's call to payment, showing both the span creation and the context propagation that prevents a gap:

```python
from opentelemetry import trace, propagate
import requests

tracer = trace.get_tracer("checkout")

def call_payment(order):
    # child span under the current checkout span
    with tracer.start_as_current_span("payment.charge") as span:
        span.set_attribute("order.id", order.id)
        span.set_attribute("amount.cents", order.amount_cents)

        headers = {}
        # inject trace context into the outgoing request -> no gap downstream
        propagate.inject(headers)

        try:
            resp = requests.post(
                "https://payment.internal/charge",
                json=order.to_payment_payload(),
                headers=headers,           # carries `traceparent`
                timeout=2.0,
            )
            span.set_attribute("http.status_code", resp.status_code)
            if resp.status_code >= 500:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
            return resp
        except requests.Timeout:
            # record the failure ON the span so the trace shows WHERE it broke
            span.set_status(trace.Status(trace.StatusCode.ERROR, "payment timeout"))
            span.record_exception(TimeoutError("payment 2s timeout"))
            raise
```

The `propagate.inject(headers)` line is the entire difference between a complete trace and a gap. It writes the `traceparent` header so the payment service — and anything it calls, as long as *it* propagates too — joins the same trace. Note also that we record the timeout *on the span*: when the trace renders, that span is marked errored at exactly the spot the request died, which is what turns a trace from a latency picture into a root-cause picture.

#### Worked example: debugging the p99 spike with the trace

Back to 3am. The page fired off the p99 query from section 3: `checkout p99 > 2s`. Here is how the wiring pays off, step by step, and why it takes two minutes instead of forty.

First, the **metric** told us the *what and where-ish*: p99 on `/checkout` crossed 2s, error rate is normal (so requests are succeeding, just slowly — that already rules out a hard failure and points at a latency source). The metric is aggregate, so it cannot tell us *which* requests or *where inside*. But our metrics emit **exemplars** — a metric exemplar is a sampled example trace id attached to a histogram bucket, so the p99 bucket carries a pointer to an actual slow trace. We click the exemplar on the 2s bucket and jump straight to a real slow trace. (No exemplars? We instead query traces for `service=checkout duration>2s` over the last 5 minutes and open one — slightly slower, same destination.)

Second, the **trace** tells us *where across services*: the slow trace's span tree shows gateway 9ms, checkout 1,910ms total, of which inventory 18ms, orders-db 9ms, and **payment 1,860ms**. The latency is entirely in the payment span. Without the trace we would not know whether payment, inventory, or the database was the culprit — the metric only said "checkout is slow." The trace localized it to payment in one view.

Third, the **log** tells us *why*: we take the trace id from the slow span and query logs for `trace_id = <that id>`. The payment service's logs for that request show `charge slow, gateway_latency_ms=1820, retries=2, upstream=fraud-check`. Now we know: payment is slow because *its* call to the fraud-check service is slow and being retried twice. (And if fraud-check had not propagated context, we would have had the gap from earlier — which is itself a finding: "payment is slow and we are blind past it, fix propagation.")

Total elapsed: metric to exemplar to trace to log, a few clicks, a couple of minutes — because every step shared an identifier with the next. Compare the logs-only path in figure 4: no exemplar, no trace, so we would grep six services' logs for anything slow in the last five minutes, try to correlate by timestamp with no shared id, and slowly reconstruct by hand what the trace showed in one view. That is the forty-minute version. The difference is not the tools' raw power; it is whether the *correlation* was designed in.

## 6. Correlation: the wiring that makes three pillars one investigation

Everything in the last three sections converges on a single design property: **correlation.** Three pillars that cannot reference each other are three silos; three pillars joined by shared identifiers are one investigation. The shared identifier is a **correlation id** — variously a request id, a trace id, or both — minted once at the edge and threaded through everything the request touches. Figure 9 shows the id flowing from the edge into logs, spans, downstream calls, and metric exemplars.

![Graph showing an edge proxy minting a request id that propagates into the service, then into structured logs, a trace span, a downstream call, and a metric exemplar](/imgs/blogs/observability-metrics-logs-traces-by-design-9.webp)

The design rules for correlation are short and absolute. **Mint the id at the very edge** — the load balancer or API gateway — so that even the entry point's own logs carry it, and so a client can be handed the id (in a response header) to quote in a support ticket. (For where this edge sits, see [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7).) **Propagate it on every call**, synchronous and asynchronous, in a standard header (`traceparent` for trace context, plus your own `X-Request-Id` if you like a human-friendly handle). **Stamp it on every log line** (the structured-logging snippet above does this automatically from the span context). **Attach it to metric exemplars** so a metric spike carries a pointer to a representative trace. Do these four and any signal becomes a launchpad into the others: a metric spike → its exemplar trace → the trace's span tree → the per-span logs by trace id. Skip any one and you reintroduce a manual correlation step — exactly the forty-minute grind we are trying to design away.

The deepest version of this property is the **exemplar**, because it links the cheap-aggregate world to the expensive-detail world directly. A histogram bucket is an aggregate — it tells you "37 requests landed in the 2s+ bucket" but throws away which ones. An exemplar attaches *one* real example to that bucket: a trace id of an actual request that landed there. So when you see the p99 bucket light up, you do not have to go hunting for an example slow request — the metric is already holding one by the hand. This is the single highest-leverage piece of observability wiring, because it solves the metric's fundamental weakness (no per-event detail) by pinning a pointer to the detail layer onto the aggregate. A senior designing observability makes exemplars a first-class requirement, not a nice-to-have, because they are what make the metric-to-trace pivot one click instead of a query you write by hand under pressure.

## 7. The trade-off matrix: cost, cardinality, latency-to-insight, retention, query power

Now the decision section, because every observability choice is a trade and a senior names the cost of each. The three pillars are not ranked best-to-worst; they each win a column and lose others, which is *why you run all three* rather than picking one. Figure 1 at the top is this matrix; here it is again as a markdown table you can defend in a review, with the reasoning for each cell.

| Property | Metrics | Logs | Traces |
| --- | --- | --- | --- |
| **Cost per event** | Cheapest (pre-aggregated) | 10–100x metrics (full row per event) | Medium (per-request, but sampled) |
| **Cardinality tolerance** | Low (cardinality = cost, blows up) | Highest (arbitrary fields free-ish) | High (per-request id is the point) |
| **Latency to insight** | Seconds (scrape + alert) | Minutes (index + query) | Minutes (assemble span tree) |
| **Practical retention** | Months–years (cheap, downsampled) | Days–weeks (expensive at volume) | Days (sampled, still bulky) |
| **Query power** | Aggregate only (no per-event) | Full-text + structured filter | Span tree + cross-service causality |
| **Causality across services** | None | Weak (manual id correlation) | Native (the entire point) |

Read the matrix as a senior would. **Metrics win cost and latency-to-insight and retention**, which is exactly why they are the alerting and long-term-trend layer — cheap enough to keep forever, fast enough to page on. They *lose* cardinality and per-event detail and causality, so you never try to debug a specific request with them. **Logs win cardinality and query power** — arbitrary fields, full-text search, the richest per-event picture — which is why they are the investigation layer once you have narrowed to a specific event. They *lose* on cost (10–100x), so you sample the boring case and never use them to count. **Traces win causality across services**, uniquely and completely, which is why they are the localization layer that turns "checkout is slow" into "payment's fraud-check call is slow." They *lose* on retention and on always-on completeness (sampling means not every request is captured), so you design sampling to keep the requests that matter.

The architectural consequence of this matrix is the layering we have been building toward: **alert on metrics (cheap, fast), localize with traces (causal), investigate with logs (detailed).** Each pillar is used for the column it wins and never stretched to a column it loses. The teams that get observability wrong are almost always the teams that stretch one pillar across all three jobs — logging everything (so logs do the metric's counting job at 50x the cost), or alerting off log queries (so a slow expensive query stands between an incident and the page), or trying to debug specific requests from dashboards (impossible, the detail was aggregated away). Use each pillar for its column. That single discipline is most of what separates an affordable, debuggable observability stack from an expensive, useless one.

## 8. The optimization lens: cardinality budgets and the cost trap that bites

Now the part that determines whether your observability bill is \$400 a month or \$40,000: **cardinality.** This is where the senior optimization lens matters most, because the cost of observability does not scale with your traffic in the way people expect — it scales with the *cardinality of your telemetry*, and cardinality can explode silently from a one-line code change. Figure 7 shows the before-and-after of a cardinality blowup.

![Before-and-after showing an unbounded user id label producing twenty million metric series and a large bill versus a bucketed tier label keeping it to thirty series](/imgs/blogs/observability-metrics-logs-traces-by-design-7.webp)

The mechanism of the explosion is multiplicative, and that is what makes it dangerous. A metric's cost is roughly the number of distinct **time series** it produces, and the number of series is the *product* of the cardinalities of all its labels. A metric `http_requests_total` with labels `{service, endpoint, status}` produces (services) × (endpoints) × (statuses) series — a few thousand, as we computed in section 3. Add one label with high cardinality and you multiply the whole thing by that cardinality. Add `user_id` with 2 million distinct values and you have just multiplied a few thousand series by 2 million. Each metric series costs memory in your time-series database (it must hold the series in RAM to ingest it) and costs money in a hosted observability product (most price per active series). Twenty million series is not "a bigger version of two thousand series" — it is a categorically different system, one that crashes your Prometheus with an out-of-memory error or hands you a five-figure monthly invoice for telemetry nobody can even query usefully (a 2-million-way breakdown is not a chart, it is noise).

The defense is a **cardinality budget**: treat the number of series each metric can produce as a resource you allocate on purpose, and review every label for its cardinality before it ships. The rule of thumb is brutal and correct: **labels on metrics must be low-cardinality and bounded.** `status` (bounded, ~6 values), `endpoint` (bounded, dozens), `region` (bounded, dozens), `tier` (bounded, 3) — all fine. `user_id`, `request_id`, `session_id`, `email`, `full_url_with_query_string`, `error_message` — all forbidden on metrics, because each is unbounded or near-unbounded and will multiply your series into oblivion. The high-cardinality dimension you genuinely need for that user belongs on a **log or a trace**, not a metric. This is the meter-versus-log decision viewed through cost: the cheap aggregate layer (metrics) *must* stay low-cardinality to stay cheap; the high-cardinality detail you need lives in the layers built to carry it (logs, traces) at their higher per-event cost but lower always-on volume.

#### Worked example: computing the cost of metric cardinality

Let me put real numbers on the trap so the magnitude is undeniable. Suppose your hosted metrics vendor charges \$0.25 per active series per month (a representative order of magnitude). You have a service with 10 metrics, and today each metric carries the labels `{service, endpoint, status}` producing about 3 series each on average across your endpoints — call it 10 metrics × 40 service-endpoint pairs × ~3 status values ≈ 1,200 active series. Your monthly metrics bill for this service is 1,200 × \$0.25 = **\$300/month.** Reasonable.

Now a well-meaning engineer adds `user_id` to those metrics "so we can see per-user behavior," and you have 2 million active users in a month. Every one of those 10 metrics now multiplies by the number of distinct users it sees. Even if each metric only sees, say, the full 2 million users over the billing month, you go from 1,200 series to on the order of 10 metrics × 2,000,000 users ≈ **20,000,000 active series.** The bill: 20,000,000 × \$0.25 = **\$5,000,000/month** at face value — and even with vendor cardinality-tier discounts and the fact that not every user hits every endpoint, you are realistically looking at a jump from a few hundred dollars to tens of thousands of dollars per month, for a "feature" that produces a 2-million-way breakdown nobody can read. (The figure shows a more conservative \$48k/month against the controlled \$40/month for the same insight via a 3-value `tier` label — the exact multiplier depends on your traffic, but the *shape* is always this violent.)

The fix costs nothing and loses nothing you actually needed. If you want to know per-*tier* behavior, label by `tier` (3 values): 10 × 40 × 3 = 1,200 series, no change in cost, and a chart you can actually read. If you genuinely need to investigate a *specific* user's requests, that is what the **logs and traces** are for — query `user_id = 5520` in the log/trace store, which is built to carry that cardinality, instead of paying to pre-compute a 2-million-way metric breakdown you will look at approximately never. **Right cardinality on the right pillar** is the single highest-leverage cost optimization in observability, and it is a *design-time* decision: you catch it in code review by asking "what is the cardinality of this label?" — or you discover it on a Tuesday when the metrics bill arrives.

The same cost logic applies to **sampling traces**, which is the trace pillar's version of cardinality control. Capturing and storing a full trace for every single request at high QPS is expensive — each trace is many spans, each span a record. The optimization is to *sample*: keep a representative or interesting subset and drop the rest. We will look at *how* to sample in the next section, but note the senior framing now: metrics control cost via low cardinality, logs control cost via level-and-sampling, traces control cost via sampling. All three pillars have a cost knob, and observability that ignores all three knobs is observability that arrives as a surprise invoice.

## 9. Sampling strategies: head versus tail, and what each loses

Traces are too expensive to keep all of at scale, so you sample — and *how* you sample determines whether you keep the traces that matter or throw them away. There are two strategies, and the difference between them is one of the more consequential observability design decisions, because it is a direct trade between cost-predictability and keeping-the-interesting-traces.

**Head sampling** decides whether to keep a trace *at the start*, at the root span, before you know anything about how the request will turn out. You flip a weighted coin — "keep 1% of traces" — and that decision is propagated in the trace context so every service either records or drops consistently (you never want half a trace). Head sampling's virtue is that it is cheap and simple and decided up front: each service knows immediately whether to bother recording, so you spend no resources on spans you will discard, and your trace volume is predictable (1% of traffic, full stop). Its fatal weakness is that it is *blind to the outcome*: it decides before the request is slow or errors, so it keeps a random 1% — which means the slow, errored, interesting requests are kept only 1% of the time, same as the boring ones. The trace that would have explained your 3am incident had a 99% chance of being thrown away by head sampling precisely because it could not know, at the start, that this request was the one that mattered.

**Tail sampling** decides whether to keep a trace *at the end*, after all spans are collected, when you know the outcome — how long it took, whether it errored, what services it touched. This is the strategy you actually want for debugging, because you can encode rules like "keep 100% of errored traces, keep 100% of traces slower than 1 second, keep 1% of the fast successful rest." Now the interesting traces are *always* kept and only the boring common case is sampled down. Tail sampling's cost is operational: you must *buffer all the spans of a trace somewhere until the trace completes* so you can make the keep/drop decision with full information, which requires a stateful collector tier that holds in-flight traces in memory and routes all spans of a given trace to the same decision-maker. That buffering is more infrastructure, more memory, and a harder thing to scale than the stateless head-sampling coin flip — but it is what keeps the slow and errored traces that head sampling discards.

| Property | Head sampling | Tail sampling |
| --- | --- | --- |
| **Decision point** | At root span, up front | After trace completes |
| **Keeps slow/errored traces?** | No (blind 1%) | Yes (rule-based, 100%) |
| **Cost predictability** | High (fixed % of traffic) | Lower (depends on error/slow rate) |
| **Infrastructure** | Stateless, cheap | Stateful buffer tier, costlier |
| **Best for** | High-volume, cost-capped, uniform | Debuggability, keep-the-interesting |

The senior pattern is often a *hybrid*: head-sample lightly to cap raw volume at the source (drop the obvious flood early), then tail-sample at the collector to *guarantee* you keep every errored and slow trace plus a small fraction of the healthy ones. The two are not mutually exclusive, and the combination gives you both a bounded cost and the guarantee that the trace you need at 3am — the slow one, the errored one — is the one you definitely kept. The mistake is to head-sample at 1% and stop, then discover during an incident that the slow request you desperately want to inspect was a coin-flip casualty. If you take one rule from this section: **never let your sampling strategy throw away your errors and your slow tail.** Those are the traces that exist to be debugged.

## 10. Alerting that does not destroy your on-call: burn rates and the page-vs-ticket line

An observability stack that pages well is a force multiplier; one that pages badly trains your team to ignore the pager, which is worse than no pager at all. The discipline here is twofold: **alert on symptoms** (covered in section 3 — golden signals, not causes) and **alert at the right urgency**, which is where SLO burn-rate alerting comes in. Figure 6 shows a burn-rate alert firing.

![Timeline showing errors jumping from baseline to eight percent, the one-hour burn rate exceeding fourteen times, a page firing, and the burn falling after a fix](/imgs/blogs/observability-metrics-logs-traces-by-design-6.webp)

The naive alert is a *threshold*: "page when error rate > 1%." It is bad in two opposite ways at once. It is too *sensitive* — a brief 2% blip for thirty seconds that the system recovers from on its own pages a human at 3am for nothing, the definition of alert fatigue. And it is too *insensitive* in another regime — a steady 0.9% error rate, just under the threshold, never pages but is silently burning your reliability into the ground over days. The threshold has no notion of *how much it matters over time*. The fix, which ties observability directly to the [SLOs and error budgets](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) the reliability post defines, is to alert on **burn rate**: how fast you are consuming your error budget relative to the rate that would exhaust it over the SLO window.

The idea is precise. If your SLO is 99.9% over 30 days, your error budget is 0.1% of requests — and a "normal" burn rate spends that budget evenly across the 30 days. A burn rate of *1x* exactly exhausts the budget at the end of the window (acceptable). A burn rate of *14.4x* means you are spending budget 14.4 times faster than sustainable — at that rate you would exhaust a 30-day budget in roughly 2 days, which is an emergency that warrants a **page**. A burn rate of *3x* means you would exhaust it in ~10 days — a real problem but not a wake-someone-up problem, which warrants a **ticket**. Burn-rate alerting maps the *severity* of an alert to *how fast you are losing reliability*, which is exactly the thing you actually care about, instead of to a raw threshold that ignores duration. The standard refinement is **multi-window, multi-burn-rate**: page on a fast burn (e.g. 14.4x over 1 hour *and* still burning over 5 minutes, to avoid paging on a momentary spike that already recovered) and ticket on a slow burn (e.g. 3x over 6 hours). The dual-window requirement — fast burn must be confirmed by a recent shorter window — is what kills the false page from a 30-second blip while still catching the real sustained burn fast.

Here is the burn-rate alert as a Prometheus rule, the fast-page variant:

```yaml
# Page if we are burning the 30-day error budget at >14.4x
# over BOTH a 1h and a 5m window (fast burn, confirmed, not a momentary blip).
groups:
  - name: checkout-slo
    rules:
      - alert: CheckoutErrorBudgetFastBurn
        expr: |
          (
            sum(rate(http_requests_total{service="checkout",status=~"5.."}[1h]))
            / sum(rate(http_requests_total{service="checkout"}[1h]))
          ) > (14.4 * 0.001)
          and
          (
            sum(rate(http_requests_total{service="checkout",status=~"5.."}[5m]))
            / sum(rate(http_requests_total{service="checkout"}[5m]))
          ) > (14.4 * 0.001)
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Checkout burning 30d error budget >14.4x (fast)"
```

The `page-vs-ticket` decision deserves its own crisp statement, because it is where teams either protect their on-call or burn them out. **Page only for things that need a human to act *now*** — a fast budget burn, a hard outage, a symptom the user is feeling that will not self-resolve. **Ticket everything else** — slow burns, degraded-but-functioning states, capacity warnings, anything that can wait until business hours. The test for whether something should page is not "is it bad" but "does a human need to act on it *in the next few minutes*, and is there an action they can take." If the answer to either half is no, it is a ticket, not a page. Every page that does not pass that test trains your team to trust the pager less, until the night the real page comes and someone swipes it away half-asleep because the last ten were noise. Alert fatigue is not a personal failing; it is a *design failure* of the alerting layer, and you fix it by paging only on confirmed, actionable, user-facing symptoms.

## 11. Dashboards that answer questions versus vanity dashboards

A quick but important section, because dashboards are where observability most often degrades into theater. A **vanity dashboard** is one that looks impressive — twenty graphs, total requests ever, a world map of traffic, a big number that goes up and to the right — and answers no question you actually ask during an incident. It exists to be shown in a status meeting, not to be used at 3am. The tell is that nobody can state, for a given graph, *what decision it informs*. If a graph does not change what you would *do*, it is decoration.

A dashboard that *works* is organized around the questions you ask in the order you ask them. The top-level service dashboard shows the **four golden signals** for the service — latency (p50/p99), traffic, errors, saturation — because those are the first four questions of any incident ("is it slow, is load weird, are we erroring, are we saturated"). Below that, the same signals **broken down by the dimensions you actually pivot on** — by endpoint, by region, by dependency — because the second question is always "where." The dashboard should let you go from "the service is unhealthy" to "it is the EU region on the checkout endpoint" in two glances, and then hand you off to a trace. The senior heuristic: **a good dashboard mirrors the diagnostic path** — symptom (golden signals), then localization (breakdowns), then a link out to traces — and ruthlessly omits anything that does not serve that path. Every graph should answer "is this normal, and if not, where" for a thing you would act on. Total-requests-since-launch answers neither. Cut it.

There is a real-time-versus-investigation distinction worth naming too. Dashboards are for *known* questions you ask repeatedly (the golden signals, the standard breakdowns) — they are monitoring, in the section-1 sense. The *novel* question, the unknown-unknown, is not answered by a pre-built dashboard; it is answered by *ad-hoc querying* of high-cardinality data ("group p99 by this weird dimension I just thought of"). A mature observability setup has both: a small number of high-signal dashboards for the questions you always ask, and a powerful ad-hoc query layer over high-cardinality telemetry for the questions you have never asked before. The failure mode is investing entirely in dashboards (you can answer only pre-built questions, helpless against novel failure) or having no dashboards at all (you re-derive the golden signals from scratch every incident). Build the few dashboards that mirror the diagnostic path; keep the cardinality and query power to go off-road when the incident is novel.

## 12. OpenTelemetry: the standard that makes this portable

A short section on the connective tissue, because the single biggest practical shift in observability over the last few years is the consolidation onto **OpenTelemetry (OTel)** as the vendor-neutral standard for generating and shipping telemetry. Before OTel, every observability vendor had its own agent, its own SDK, its own wire format, and instrumenting your code meant marrying a vendor — rip-and-replace if you ever wanted to switch. OTel breaks that lock-in by standardizing three things: the **API/SDK** you call in your code to create spans and metrics, the **wire protocol** (OTLP) that carries telemetry, and the **collector**, a vendor-neutral agent that receives OTLP, processes it (batching, sampling, scrubbing, enrichment), and exports it to whatever backend you choose. Instrument once against the OTel API, run a collector, and you can point the backend at Prometheus, Jaeger, a hosted vendor, or three of them at once, switching backends without touching application code.

The architectural payoff is that **instrumentation becomes a property of your code, and routing becomes a property of your infrastructure**, decoupled. The collector tier is where the design leverage lives: it is where you do **tail sampling** (the stateful buffering from section 9), where you **scrub PII** out of telemetry before it leaves your trust boundary, where you **enrich** spans with deployment metadata (which version, which region, which host), and where you **fan out** to multiple backends. Putting sampling and scrubbing in the collector rather than in every service means you change those policies in one place, not in fifty codebases. For a senior designing observability today, the default answer to "what do we instrument with" is OpenTelemetry, and the default architecture is application-instrumented-with-OTel-SDK → OTLP → collector-tier (sampling, scrubbing, enrichment) → backend(s). It is not the only choice, but it is the one that keeps you from re-instrumenting your entire fleet the next time you want to change vendors or add a backend, and that portability is worth a lot. The decision tree in figure 8 captures the runtime side — given a page, which signal do you reach for — which OTel makes uniform across whatever backends you chose.

![Decision tree mapping a 3am page to metrics for system-wide pain, USE saturation gauges for resource limits, traces for one slow request, and correlated logs for one odd request](/imgs/blogs/observability-metrics-logs-traces-by-design-8.webp)

## 13. Case studies: an incident, a bill, and a postmortem

Three real-shaped stories, each teaching one lesson, because observability is best understood through what its absence costs.

**Traces find a root cause that metrics could not — the cross-service latency hunt.** A team running a microservices checkout had good metrics: per-service RED, golden-signal dashboards, the works. One afternoon checkout p99 crept from 200ms to 1.4s. The metrics were unambiguous that checkout was slow and that *every individual downstream service's own p99 looked normal* — payment said it was fast, inventory said it was fast, the database said it was fast. This is the classic blind spot of per-service metrics: each service measures *its own* time and sees nothing wrong, yet the request is slow, because the latency lives in the *gaps between* services — connection acquisition, serialization, a retry, queueing — that no single service's metric captures. They had recently added distributed tracing. Opening one slow trace showed the truth instantly: checkout was making payment, inventory, and shipping calls *sequentially* that could have been parallel, and a recent deploy had added a fourth sequential call, so the per-service times were all fine but they now *summed* to 1.4s. No metric showed this because the problem was the *shape of the call graph*, which only a trace makes visible. The lesson: **per-service metrics see symptoms; only traces see cross-service causality.** A system without tracing is blind to exactly the failures that live between services — which, in a microservices architecture, is most of them.

**A cardinality explosion arrives as an invoice — the \$40,000 surprise.** A growing startup on a hosted metrics product shipped a release where, reasonably enough, someone added a `customer_id` label to a handful of business metrics to "see per-customer usage." It passed code review — nobody asked the cardinality question. It worked perfectly in staging (a dozen test customers). In production, with tens of thousands of real customers and several metrics each, the active series count went from low tens of thousands to several million within a day. The metrics did not break — the hosted product happily ingested them — but the next monthly invoice was roughly an order of magnitude larger than the last, the increase entirely attributable to per-series billing on the new high-cardinality label. The cruel part: the per-customer breakdown was *never once queried*, because a tens-of-thousands-way breakdown is not a usable chart. They had paid five figures for telemetry that answered no question. The fix was a one-line revert plus a new code-review checklist item — "what is the cardinality of every metric label you are adding" — and the high-cardinality per-customer questions were moved to the log/trace store where that cardinality is what those pillars are *for*. The lesson: **cardinality is a cost you provision at design time, or a surprise you receive in the mail.** Put the cardinality question in code review, because the engineer adding the label is the only person positioned to catch it before it ships.

**An alert-fatigue postmortem — the page nobody read.** A team had grown its alerting organically: every incident's retro added "an alert so we catch it next time," and over two years that became ~200 alerts, most of them cause-based thresholds ("CPU > 80%", "disk > 70%", "queue depth > 1000") that fired constantly and self-resolved. On-call engineers received dozens of pages a night, the overwhelming majority non-actionable, and had — entirely rationally — learned to glance and dismiss. Then a real outage came: a genuine fast error-budget burn paged at 2am, and the on-call engineer, conditioned by hundreds of noise pages, acknowledged and dismissed it without reading, going back to sleep. The outage ran for over an hour before a customer escalation woke someone properly. The postmortem's honest root cause was not the engineer — it was the *alerting design*: pages that were not actionable, not symptom-based, and not tuned to urgency had trained the team to ignore the pager, so the one page that mattered was indistinguishable from the noise. The fix was severe and correct: delete every alert that was not a confirmed, actionable, user-facing symptom; convert cause-based thresholds to tickets or dashboards; rebuild paging on multi-window burn-rate alerts (section 10). Page volume dropped by over 90%, and every remaining page was real. The lesson: **alert fatigue is a design failure, and the cure is fewer, better alerts — page only on confirmed actionable symptoms, ticket the rest.** A pager that cries wolf is worse than no pager, because it actively trains your team to miss the wolf.

## 14. Stress-testing the design: what breaks at 3am, at scale, at a gap

Now the senior move — take the observability design and stress it, because a design you have not stress-tested is a design you do not yet trust.

**The 3am incident: which signal finds root cause fastest?** Stress-test the diagnostic path. A page fires. If your only signal is metrics, you know *that* checkout is slow and you are stuck — metrics cannot tell you *which request* or *where inside*, so you devolve to guessing or to manual log-grepping (the forty-minute path). If you have metrics *with exemplars* feeding into *traces* that are *tail-sampled to keep slow requests*, joined to *structured logs by trace id*, you go metric → exemplar → trace → log in minutes. The stress test reveals the weak links: no exemplars means a manual hunt for a slow trace; head sampling means the slow trace might be gone; unstructured logs means you cannot join by trace id. **Each missing piece of wiring adds a manual step at the worst possible time.** The design passes only if every pivot — metric to trace, trace to log — is a click, not a query you compose under pressure.

**At scale (10x traffic): what breaks?** Stress the cost dimensions. At 10x traffic, your *metrics* cost barely moves if your cardinality is bounded (10x more requests increment the same counters — series count is unchanged), which is exactly why low cardinality is the optimization that makes metrics scale gracefully. But your *logs* cost moves 10x if you log per-request, and your *traces* cost moves 10x if you sample at a fixed percentage. So at 10x the design holds *only if* logs are sampled (keep errors and slow, sample the boring success) and traces are tail-sampled (keep interesting, drop common). The team that logged every request and head-sampled at a fixed 1% finds their log bill 10x'd and their trace store no more useful than before. The stress test forces the cost-knob discipline: **the pillar whose cost scales with raw traffic (logs, naive traces) must be sampled, or 10x traffic is 10x bill.** Bounded-cardinality metrics scale for free; everything else needs a knob.

**At a region failure or a hot key: does observability still see?** The subtle one. When a whole region fails, can you *tell*, and can you tell *which* region — i.e., is `region` a dimension on your golden signals? If you aggregated region away (to save cardinality, perhaps over-zealously), a single-region outage shows as a *partial* global degradation that is hard to localize. The lesson is that cardinality control is a *balance*: too much cardinality bankrupts you, too little blinds you to exactly the dimension you need (region, AZ, shard, deploy version). The senior keeps the *bounded* dimensions that matter operationally — region, AZ, endpoint, status, deploy version — and forbids the *unbounded* ones — user id, request id. A hot key (one shard, one tenant getting hammered) is visible only if `shard` or `tenant_tier` is a (bounded) dimension; if not, the hot key shows as unexplained aggregate latency you cannot localize. **The dimensions you keep are the questions you can answer; choose them on purpose.**

**The trace gap from an un-propagating service.** Already covered in section 5, but it belongs in the stress test as the failure that hides from you: a service that does not propagate context produces a trace that *looks complete but is missing a branch*. The stress test for this is a periodic audit — does every service, including async consumers and third-party hops, propagate `traceparent`? — because the gap is silent until the incident lives in the gap. The design must include *propagation as a checked invariant*, not an assumed one. A contract test that asserts the trace context survives each hop is cheap insurance against the one missing link that hides your root cause.

## 15. When to reach for each pillar (and when not to)

Decisive recommendations, the way a senior would close a design review.

**Always instrument RED metrics per service and endpoint, from day one.** They are cheap, bounded, and catch most "is this okay" questions. This is the non-negotiable floor; a service with no RED metrics is a service you cannot operate. Add USE metrics for the resources that constrain you (connection pools, queues, thread pools) the moment you have a resource that saturates.

**Alert only on the four golden signals, as symptoms, with multi-window burn-rate alerts.** Do not page on causes (CPU, memory, disk) — graph those, ticket those, but do not wake a human for them. Page only on confirmed, actionable, user-facing symptoms. This is how you keep on-call trustworthy.

**Add distributed tracing the moment you have more than two or three services in a request path.** In a monolith, traces add little over good logs and metrics — the request stays in one process. In a microservices architecture, traces are not optional; they are the *only* thing that sees cross-service causality, and most of your hard failures live between services. Use OpenTelemetry so you are not locked to a vendor, and *propagate context at every hop* including async and third-party.

**Use structured logs for per-event detail, joined to traces by id, sampled for the common case.** Always log errors and slow requests in full; sample the boring success. Never use logs to count what a metric can count — that is paying 50x for a number you can get for free.

**Control cardinality as a design-time budget.** Low-cardinality bounded labels on metrics (status, endpoint, region, tier, deploy version); forbid unbounded labels (user id, request id) on metrics and move that cardinality to logs and traces. Tail-sample traces to keep the slow and errored ones. Put the cardinality question in code review.

**When *not* to over-invest:** a small single-service app with low traffic does not need tail-sampling infrastructure, a stateful collector tier, or fifty dashboards — RED metrics, structured logs, and basic alerting are plenty, and the full distributed-tracing apparatus is cost you do not yet need to pay. Observability should be proportional to operational complexity. Scale the investment with the number of services, the traffic, and the cost of being down — not ahead of it. The discipline of [evolutionary architecture](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) applies here too: design the observability you need now with a clear path to add tracing and tail-sampling when service count and traffic actually demand it.

## 16. Key takeaways

- **You cannot operate what you cannot see, and you only see what you instrument on purpose.** Observability is a design property — correlation IDs, propagation, exemplars, bounded cardinality — decided before the incident, not a tool bought after.
- **Monitoring answers known-unknowns; observability answers unknown-unknowns.** The difference is high-cardinality detail you can slice along a dimension you did not pre-aggregate. The 3am question is always one you did not foresee.
- **Each pillar wins one column: metrics are cheap aggregate alerting, logs are high-cardinality detail, traces are cross-service causality.** Use each for its column. Stretching one pillar across all three jobs is how you build an expensive useless stack.
- **Meter what you alert on, log what you investigate, trace what you correlate.** Alert on metrics, localize with traces, investigate with logs — metric → trace → log, each joined by a shared id so every pivot is one click.
- **Alert on symptoms, not causes.** The four golden signals (latency, traffic, errors, saturation) are user-visible symptoms that catch novel failures; causes are unbounded and always one behind. Page only on confirmed, actionable symptoms via multi-window burn-rate alerts; ticket everything else.
- **Cardinality is a cost you provision at design time or receive as an invoice.** Metric series multiply by label cardinality — an unbounded `user_id` label can turn a \$300 bill into tens of thousands. Keep metric labels bounded; push high-cardinality detail to logs and traces.
- **Sampling decides whether you keep the trace that matters.** Head sampling is cheap but blind to outcome — it discards your slow and errored traces. Tail-sample (or hybrid) so you *always* keep the errors and the slow tail; those exist to be debugged.
- **Alert fatigue is a design failure.** A pager that cries wolf trains your team to miss the wolf. Fewer, better, symptom-based alerts beat two hundred noisy thresholds every time.
- **Standardize on OpenTelemetry and a collector tier.** Instrument once, route anywhere; do sampling, PII scrubbing, and enrichment in the collector so policy changes in one place, not fifty codebases.

## 17. Further reading

- [Google SRE Book — Monitoring Distributed Systems (the four golden signals)](https://sre.google/sre-book/monitoring-distributed-systems/) — the canonical source for symptom-based alerting and the golden signals.
- [Google SRE Workbook — Alerting on SLOs (multi-window, multi-burn-rate)](https://sre.google/workbook/alerting-on-slos/) — the burn-rate alerting math used in section 10.
- [OpenTelemetry documentation](https://opentelemetry.io/docs/) — the vendor-neutral standard for instrumentation, OTLP, and the collector.
- [Dapper, Google's large-scale distributed tracing infrastructure (paper)](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/) — the original distributed-tracing paper that everything since builds on.
- The RED method (Tom Wilkie) and the USE method (Brendan Gregg) — the two instrumentation methods from section 3, worth reading in the authors' own words.
- Sibling posts in this series: [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) (observability measures the SLOs reliability sets), [articulating trade-offs with CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond), and [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) (where context propagation rides the message envelope).
- Forward in this series: [capacity planning and autoscaling](/blog/software-development/system-design/capacity-planning-and-autoscaling) (saturation metrics drive autoscaling) and [testing distributed systems with chaos and load](/blog/software-development/system-design/testing-distributed-systems-chaos-and-load) (you verify observability by breaking things on purpose and checking you can see it).
- Mechanism deep-dive: [anatomy of a message system — producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers), for how trace context must ride the message envelope across async hops.

---
title: "Observability for APIs: Logs, Metrics, Traces, and SLOs"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "You cannot operate a contract you cannot see — learn to instrument an API with structured logs, RED metrics, distributed traces, a correlation id that ties a log line to a trace to a support ticket, percentile latency, and SLOs with error budgets that gate your releases."
tags:
  [
    "api-design",
    "api",
    "observability",
    "metrics",
    "tracing",
    "opentelemetry",
    "slo",
    "http",
    "monitoring",
    "operability",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-1.png"
---

At 02:14 on a Tuesday I got paged for the Payments API. The alert said, in full: `payments-api error rate elevated`. That was the entire message. No route, no status code, no caller, no idea whether one customer was affected or ten thousand. I opened the only dashboard we had — a single line labeled "average response time" — and it read 92 ms, flat and green, exactly as it had all week. So I had an alert that said something was wrong and a dashboard that swore everything was fine, and the only way to reconcile them was to start SSH-ing into boxes and grepping logs that turned out to be a wall of unstructured `printf` lines with no request id, no timestamp I could trust, and a customer's full card PAN sitting in plaintext in three of them. It took us forty minutes to discover that a single downstream ledger service had started taking four seconds on one in every hundred writes, that the *average* stayed at 92 ms because ninety-nine fast requests drowned one catastrophic one, and that the one in a hundred was disproportionately our largest merchant, whose retries were now stacking up and tipping us toward a real outage.

We did not have an outage that night because we got lucky, not because we could see. And that is the lesson this whole post is built on: **you cannot operate a contract you cannot see.** You can design the cleanest resource model, the most honest error envelope, and the most careful versioning policy in the world, and none of it matters at 02:14 if you cannot answer three questions in under a minute — *is it broken, where is it broken, and why.* Those three questions map onto the three classic pillars of observability, and each pillar exists precisely because the other two cannot answer its question. **Metrics** tell you *that* something is wrong (the error rate moved, the p99 climbed). **Traces** tell you *where* it is wrong (the ledger span owns 600 of the 720 milliseconds). **Logs** tell you *why* it is wrong (`ledger: lock wait timeout on row 88123`). The figure below is the whole argument in one frame: three pillars, one shared request id, one view you can debug an alert from.

![a diagram showing logs metrics and traces all stitched together by one shared request id flowing into a single view that lets an on-call engineer move from a fired alert to a fix](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-1.png)

By the end of this post you will be able to instrument an API as the long-lived, externally consumed contract it is: emit **structured, leveled JSON logs** carrying a correlation id and no secrets; record **RED metrics** (Rate, Errors, Duration) per route and per status without blowing up your bill on cardinality; propagate **W3C trace context** so a request's path across services is one connected trace; alert on **p95/p99 latency** instead of an average that lies to you; and define **SLIs, SLOs, and error budgets** with **burn-rate alerts** that page a human on the symptoms your users feel and gate your releases when you are spending reliability faster than you promised. We will keep returning to the series' running example — a "Payments & Orders" API for a fictional commerce platform — and we will instrument `POST /payments`, trace a slow payment across services, and write a 99.9% availability SLO with the error budget math worked out to the minute. This is part of the **["Designing APIs That Last"](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems)** series, and it composes directly with the post on **[error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract)** — the correlation id you put in an error body is the same id you will search your logs and traces for — and with **[API performance](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency)**, where the tail latency you instrument here is the tail latency you go fix there.

## 1. The principle: an SLO turns "reliable enough" into a number you can defend

Start from the spine of this series: an API is a contract you design for a caller you will never meet, on a timeline of years. A contract has terms. "We will be up" is not a term — it is a wish, and an unmeasurable one. Every system is down sometimes; the only honest question is *how often, for how long, and measured how.* Observability is how you measure it, and a **Service Level Objective (SLO)** is how you turn that measurement into a term of the contract you can be held to and can hold others to.

Three acronyms anchor the rest of this post, so define them once, precisely:

- An **SLI (Service Level Indicator)** is a *measurement* of one aspect of service quality, expressed as a ratio of good events to valid events. For an API, the canonical availability SLI is `successful requests / valid requests`, and the canonical latency SLI is `requests faster than a threshold / valid requests`. An SLI is a number you compute from telemetry, between 0 and 1.
- An **SLO (Service Level Objective)** is a *target* you set for an SLI over a window: "99.9% of valid requests succeed, measured over a rolling 28 days." It is an internal promise.
- An **SLA (Service Level Agreement)** is an *externally contracted* SLO with consequences — a refund, a credit, a penalty — if you miss it. The SLA is almost always *looser* than the internal SLO, on purpose: you alert yourself at 99.9% so you never get near the 99.5% you owe a customer money for.

Here is the principle that makes the whole machine work, and it is worth stating as a rule because everything downstream is a consequence of it:

> **Choose a target below 100%, and the gap between your target and 100% becomes a budget you are *allowed* to spend. That budget — the error budget — is the single number that resolves the eternal fight between shipping features and keeping things stable, because it makes "how much risk can we take this month" a quantity instead of an argument.**

Let me derive why the budget exists and why it must not be 100%. Suppose your SLO is 99.9% availability over 30 days. The complement, $100\% - 99.9\% = 0.1\%$, is the fraction of requests you are *permitted* to fail. That fraction is not waste — it is the resource you spend on shipping. If you set the target at 100%, the error budget is zero, which means *any* change that risks *any* failure is forbidden, which means you ship nothing, which is itself a failure of the product. A 100% reliability target is not safe; it is a way to guarantee you fall behind. The right target is "as reliable as your users actually need and no more," and the slack between that and perfection is the fuel for every deploy.

Now turn the percentage into time, because minutes are what an on-call engineer feels. A 30-day month is $30 \times 24 \times 60 = 43{,}200$ minutes. A 99.9% availability SLO permits $0.1\%$ of that to be bad:

$$\text{budget} = 43{,}200 \times 0.001 = 43.2 \text{ minutes per month}.$$

So **`99.9% ≈ 43 min/month`** of downtime, full stop. That single number reframes everything. A 30-minute incident is not "an incident" in the abstract; it is *70% of your entire monthly budget*, gone in one night. A second 30-minute incident the same month puts you over. The table later in this post lays the famous "nines" out — 99% is over 7 hours a month, 99.99% is barely 4 minutes — but the move that matters is the one we just made: we converted a vague feeling ("pretty reliable") into a defendable, bankable quantity. That is what an SLO buys you, and you cannot have it without observability, because the SLI is computed entirely from the telemetry we are about to instrument.

The deeper point — the one that ties back to "an API is a contract" — is that **your callers are building their own SLOs on top of yours.** A merchant integrating `POST /payments` is making a promise to *their* customers about checkout reliability, and that promise is a function of yours. When you publish an SLO and hold yourself to it, you are giving your callers a number they can compose into their own contract. When you do not, every caller has to guess, and they will guess pessimistically, retry aggressively, and build defensive complexity that you then have to support. The SLO is a load-bearing term of the contract, not an ops detail.

There is a composition math here worth making explicit, because it changes how you think about depending on others. If your payments path calls three downstream services in *series* — a fraud check, a ledger write, and a notification — and each independently meets a 99.9% availability SLO, your path's *best-case* availability is not 99.9%; it is the product of the three, since any one of them failing fails the whole request:

$$0.999 \times 0.999 \times 0.999 = 0.999^3 \approx 0.997 = 99.7\%.$$

That dropped you from a 43-minute monthly budget to roughly a 130-minute one *before you wrote a single line of buggy code*, purely from dependency arithmetic. The lesson is not "trust nobody"; it is that **your achievable SLO is bounded by the SLOs of everything you hard-depend on**, so a high reliability target forces you to either reduce the number of synchronous dependencies on the critical path (move the notification to async, after the response), add redundancy, or make a dependency's failure non-fatal (degrade gracefully rather than 500). Observability is what lets you see *which* dependency is eating your budget, via the per-dependency RED metrics we will instrument in section 7. You cannot manage a budget you cannot attribute, and you cannot attribute one you cannot see.

One more framing keeps the whole post honest: an SLO is a statement about *the past*, computed over a trailing window, while an alert is a statement about *the present*, computed over a short window. The error budget lives in the gap. You measure the SLI continuously, you judge the SLO over 28 days, and you *act* — page, freeze, ship — based on how fast the present is consuming the window's budget. Holding those three timescales distinct is what separates a working SLO program from a wall of dashboards nobody trusts.

## 2. Structured logs: the "why," one request at a time

A log line is a record of one thing that happened in one request. Its job is to answer *why* — why did this specific payment fail, why did this caller get a 403, why did this request take four seconds. Logs are the highest-fidelity, highest-cardinality, and most expensive pillar, and the single decision that separates a useful logging setup from a useless one is **structure.**

An unstructured log line is a string built for a human to read:

```bash
[14:02:09] payment failed for user 88123 amount 49.99 reason declined retry=2
```

You cannot reliably query that. To find "all declined payments over \$40 in the last hour for tenant `acme`" you are reduced to regular expressions over free text, and the instant someone reorders the fields or rewords `reason declined` to `status: declined`, every query and alert built on it silently breaks. This is the same fragility we diagnosed in the [error design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract): a machine should never have to parse prose.

A **structured log** is the same event emitted as a machine-readable object — almost always one JSON object per line (newline-delimited JSON, or NDJSON), so each line is independently parseable:

```json
{
  "ts": "2026-06-20T14:02:09.412Z",
  "level": "warn",
  "msg": "payment declined",
  "request_id": "req_9f2c7a1e",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "service": "payments-api",
  "route": "POST /v1/payments",
  "status": 402,
  "decline_code": "insufficient_funds",
  "tenant_id": "acme",
  "amount_cents": 4999,
  "currency": "USD",
  "latency_ms": 38,
  "retry": 2
}
```

Now "declined payments over \$40 for tenant `acme`" is a structured query — `level=warn AND decline_code=* AND amount_cents>4000 AND tenant_id=acme` — that survives rewordings, survives new fields, and joins cleanly to your traces via `trace_id` and to a support ticket via `request_id`. Five disciplines turn structured logs from a liability into an asset.

**Level it.** Every line carries a severity — `debug`, `info`, `warn`, `error`. The level is the dial that lets you run quiet in production (info and up) and loud in an incident (drop to debug for one service) without redeploying. Reserve `error` for things a human should eventually look at; do not log a routine `404 not found` at `error`, or your error stream becomes noise and you will train yourself to ignore it.

**Correlate it.** Every line for a single request carries the same `request_id`, and every line for a single distributed request carries the same `trace_id` (more on the distinction in the tracing section). This is the thread that lets you pull *all* the logs for one failed payment, across every service it touched, with one query. The correlation id is the most important field in the object, and we devote a whole section to it below.

**Strip secrets and PII from it.** This is not optional and it is not a nice-to-have; it is a security and compliance control. Logs get shipped to third-party systems, replicated, retained for months, and read by far more people than your database. A card PAN, a full `Authorization: Bearer <token>` header, a password, a national id, a raw `Set-Cookie` — none of these may ever reach a log. Redact at the logging boundary, by an allowlist of fields you *may* log rather than a denylist of fields you must strip, because a denylist always loses to the next field someone adds. The PAN in our 02:14 incident was a reportable event waiting to happen.

**Sample it.** At scale you cannot afford to log every line of every request at full fidelity. Log every error and every warning; sample the `info`-level "request completed" lines (keep 1%, or 1-in-N, or all of a slow tail). The structured fields make sampling safe because the metrics — which you do keep at 100% — carry the aggregate truth; the logs are for drilling into examples.

**Make it consistent.** The same field means the same thing in every service: `latency_ms` is always milliseconds, `status` is always the HTTP status integer, `tenant_id` is always the tenant. A shared logging library that stamps `service`, `ts`, `request_id`, and `trace_id` automatically is worth building once, because hand-rolled per-service logging drifts within a quarter and then your cross-service queries return half the data.

There is a structural reason structured logs win that is worth stating plainly: a log line written for a human is optimized for *reading one at a time*, but in production you never read one at a time — you read ten thousand, filtered and aggregated, during an incident. The unit of consumption is the *query*, not the *line*, and a query needs fields, not prose. The moment you accept that production logs are a queryable dataset rather than a scrolling console, every design choice follows: emit JSON because it parses; name fields consistently because you join on them; keep the level because you filter on it; carry the correlation id because you group by it. The console-readable string was always a fiction; in production nobody reads the console.

A word on **log volume and where the lines go**. Each service writes its JSON lines to stdout (or a local file); a collector — Fluent Bit, Vector, the OTel Collector, or a cloud agent — tails them, batches them, and ships them to a central store (an aggregator like Loki, Elasticsearch/OpenSearch, or a managed log platform) where they are indexed for query. Two things bite teams here. First, **the indexed store is the expensive part**, priced roughly by ingested bytes and indexed fields, which is the real reason you sample info-level lines and cap field counts — an unbounded `extra` blob of free-form keys is a cardinality bomb for the *log* index just as a user-id label is for metrics. Second, **clocks**: the collector should stamp ingestion time but you must log an event time (`ts`) from the service, because cross-service ordering during an incident depends on a trustworthy event timestamp, and machine clocks drift. When in doubt, lean on the *trace* (which records true relative timing via span parentage) for ordering and the logs for detail.

#### Worked example: a logging middleware that does the right thing by default

Here is the shape of a request-logging middleware that stamps every line with the correlation context, redacts by allowlist, and emits one structured "completed" line per request. The point is that *individual handlers should not think about any of this* — observability that depends on every developer remembering to log correctly will not survive contact with a deadline.

```python
import json, time, uuid, logging

# Fields we are ALLOWED to log. Anything not here is dropped — allowlist, not denylist.
SAFE_FIELDS = {"route", "status", "tenant_id", "amount_cents", "currency",
               "decline_code", "retry", "latency_ms"}

def observe(handler):
    def wrapped(request):
        # Honor an inbound id (set by the gateway) or mint one; never trust it blindly for security.
        request_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:12]}"
        trace_id = current_trace_id()           # from the tracing context, section 4
        start = time.monotonic()
        ctx = {"request_id": request_id, "trace_id": trace_id,
               "service": "payments-api", "route": f"{request.method} {request.route}"}
        try:
            response = handler(request)
            status = response.status
            return response
        except Exception:
            status = 500
            log("error", "unhandled exception", ctx, status=500)
            raise
        finally:
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            level = "error" if status >= 500 else "warn" if status >= 400 else "info"
            # Sample info-level completions; always keep warn/error.
            if level != "info" or should_sample(rate=0.01) or latency_ms > 1000:
                log(level, "request completed", ctx, status=status, latency_ms=latency_ms)
    return wrapped

def log(level, msg, ctx, **fields):
    safe = {k: v for k, v in fields.items() if k in SAFE_FIELDS}
    line = {"ts": iso_now(), "level": level, "msg": msg, **ctx, **safe}
    print(json.dumps(line))   # one JSON object per line → your collector ships it
```

Notice the design choices that are really contract decisions. The middleware *always* emits the correlation id, so it is impossible to have a request with no trace back to its logs. It samples `info` but *never* drops an error or a slow request (`latency_ms > 1000`), so your examples bias toward the failures you actually need. And it filters fields through `SAFE_FIELDS`, so a developer who passes `card_number=...` into a log call simply has it dropped — the safe path is the default path. Build the trap-free version once and make it the only way to log.

## 3. Metrics: the RED method, percentiles, and the cardinality trap

A metric is a number sampled over time — a counter that only goes up, a gauge that moves up and down, or a histogram that records a distribution. Metrics are cheap (you aggregate, you do not store every event), they are the basis of every dashboard and SLO, and they answer one question: *is it broken, right now, in aggregate.* For a request-driven service like an HTTP or gRPC API, the canonical recipe is the **RED method**, coined by Tom Wilkie: for every request, record its **R**ate, its **E**rrors, and its **D**uration.

- **Rate** — requests per second, a counter you take the rate of. `rate(http_requests_total[5m])` tells you traffic. A sudden drop is as alarming as a spike: it usually means requests are failing upstream of you and never arriving.
- **Errors** — the count (or rate, or fraction) of requests that failed. For an HTTP API the honest definition is "responses with status `5xx`" for server faults; `4xx` is usually the *client's* fault and should be tracked separately so a misbehaving caller does not page you. The error *ratio* — `5xx / total` — is your availability SLI inverted.
- **Duration** — how long requests take, as a **histogram** so you can compute percentiles. Not an average. The whole next subsection is about why.

The sibling method for the resources *underneath* your service is **USE** (Brendan Gregg): for every resource — CPU, memory, the DB connection pool, a thread pool, a queue — record its **U**tilization, **S**aturation, and **E**rrors. RED watches the requests; USE watches the machinery serving them. Together with Google's **four golden signals** (latency, traffic, errors, saturation) they cover the API boundary. The matrix below lays out how they relate; they overlap heavily, which is fine — they are three lenses on the same goal of seeing both the requests and the resources.

![a comparison matrix of the RED method the USE method and the four golden signals showing what each measures what it applies to and an example metric for the payments API](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-2.png)

The three metric *types* matter because they answer different questions and cost differently:

- A **counter** monotonically increases (`http_requests_total`, `payments_charged_total`). You never read its value directly; you read its `rate()`. Counters survive restarts because the rate function handles resets.
- A **gauge** is an instantaneous value that goes up and down (`db_pool_in_use`, `inflight_requests`, `queue_depth`). This is your USE saturation signal.
- A **histogram** buckets observations so you can ask "what fraction of requests were faster than 200 ms" and estimate any percentile. `http_request_duration_seconds` is a histogram. It is the most expensive metric type because each bucket is its own time series, which leads directly to the cardinality discussion.

Here is what RED metrics look like instrumented at the boundary, in Prometheus-style client code, recorded per route and per status:

```python
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "route", "status"])          # bounded labels only

DURATION = Histogram(
    "http_request_duration_seconds", "Request latency",
    ["method", "route", "status"],
    buckets=[.01, .025, .05, .1, .2, .3, .5, .8, 1, 2, 4, 8])

def record(method, route, status, seconds):
    REQUESTS.labels(method, route, status).inc()
    DURATION.labels(method, route, status).observe(seconds)
```

And here is the availability SLI and the p99 latency, as PromQL — the two queries that feed your SLO dashboard:

```promql
# Availability SLI over 28 days: fraction of valid requests that did NOT 5xx.
1 - (
  sum(rate(http_requests_total{status=~"5.."}[28d]))
  /
  sum(rate(http_requests_total[28d]))
)

# p99 latency for POST /payments over 5 minutes, from the histogram.
histogram_quantile(0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{route="/v1/payments", method="POST"}[5m]))
)
```

### 3.1 Percentiles, not averages — and why the average lies

This is the single most important quantitative idea in the post, and the one my 02:14 dashboard got catastrophically wrong, so let me derive it rather than assert it. An **average (mean)** latency is the sum of all latencies divided by the count. A **percentile** is a threshold: the **p99** is the latency value below which 99% of requests fall, so 1% of requests are *slower* than the p99. The p50 is the median. The reason you alert on p95 and p99 and almost never on the mean is that **the mean is dominated by the many fast requests and is structurally blind to the slow tail that your worst-affected users actually experience.**

![a before and after contrast showing an average of eighty milliseconds that looks healthy and green while the p99 of four thousand milliseconds reveals the slow tail that the worst users feel](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-3.png)

#### Worked example: an average that hides a four-second tail

Take 100 requests to `POST /payments`. Ninety-nine of them complete in a healthy 40 ms. One of them — the ledger lock-wait from our incident — takes 4000 ms (4 seconds). Compute the mean:

$$\text{mean} = \frac{99 \times 40 + 1 \times 4000}{100} = \frac{3960 + 4000}{100} = \frac{7960}{100} = 79.6 \text{ ms}.$$

Your average-latency dashboard reads **79.6 ms** and is green. It looks *better* than a typical week, in fact, because the 99 fast requests pulled it down. Now compute the percentiles over the same 100 requests, sorted ascending. The 99th value (index 99 of 100) is 40 ms; the 100th — the single slow one — is 4000 ms. So:

$$p99 = 40 \text{ ms}, \qquad p100 \;(\text{max}) = 4000 \text{ ms}.$$

With only one slow request in a hundred, even the p99 here sits at 40 ms — which tells you something subtle and important: **the percentile you alert on must match the rate of the failure you care about.** If 1-in-100 requests is catastrophic, the p99 catches it only when you have enough volume that "1%" reliably includes it; for a 1-in-100 event you watch the **p99.9** or the **max**, or you alert directly on the count of requests over a threshold. Push the bad fraction up to a more realistic incident — say 5 of the 100 requests take 4000 ms — and now:

$$\text{mean} = \frac{95 \times 40 + 5 \times 4000}{100} = \frac{3800 + 20000}{100} = 238 \text{ ms},$$

$$p95 = 40 \text{ ms (the 95th sorted value)}, \qquad p99 \approx 4000 \text{ ms}.$$

The mean has crept to 238 ms — easy to dismiss as "a bit slow today." But the p99 has snapped to 4000 ms, a 100× jump, screaming that one request in twenty is now taking four seconds. *That* is the signal you page on. The lesson generalizes: **the mean averages your worst users away; the high percentile is the only number that represents them.** A merchant whose checkout hangs for four seconds does not care that the average customer waited 40 ms. They experienced 4000 ms, and so did 1% (or 5%) of everyone, and at a million requests a day that 1% is ten thousand furious people.

Two practical warnings that follow directly. First, **you cannot average percentiles.** The p99 of two servers is *not* the average of their two p99s — percentiles are not linear. To get a fleet-wide p99 you must aggregate the underlying *histogram buckets* and compute the quantile once (exactly what the `histogram_quantile(0.99, sum by (le) (...))` query above does). Averaging per-host p99s is a classic, silent, deeply wrong metric. Second, **histograms estimate percentiles from buckets**, so your p99 is only as precise as your bucket boundaries near the values you care about; that is why the histogram above has fine buckets at 100–800 ms, where SLO thresholds usually live.

### 3.2 Cardinality: the metric label that bankrupts you

The seductive mistake with metrics is to add labels that *feel* useful: `user_id`, `request_id`, `email`, a raw URL with the order id in the path. Here is why that one decision can multiply your monitoring bill by a thousand. In a metrics system, **every unique combination of label values is a separate time series**, and the number of series is the *product* of the cardinalities of all labels — the dimensions multiply, they do not add.

Suppose `http_requests_total` carries `method` (5 values), `route` (40 templated routes), and `status` (15 distinct codes you actually emit). The series count is:

$$5 \times 40 \times 15 = 3{,}000 \text{ series}.$$

Entirely fine. Now add `tenant_id` and you have, say, 500 tenants:

$$5 \times 40 \times 15 \times 500 = 1{,}500{,}000 \text{ series}.$$

Already large but bounded, and arguably worth it because per-tenant RED is genuinely useful. Now add `user_id` with 2 million users:

$$3{,}000 \times 500 \times 2{,}000{,}000 = 3 \times 10^{12} \text{ series}.$$

That is three *trillion* series. Your metrics backend will fall over, your bill will be five or six figures a month, and you will have learned the hard way the iron rule of metric labels:

> **A metric label's value set must be bounded and small. Put high-cardinality identifiers — user id, request id, email, order id, full URL — in logs and traces, never in a metric label.** Metrics are for low-cardinality aggregates; the per-request detail belongs in the pillars built to carry it.

The matrix below restates the division of labor: logs and traces are high-cardinality and sampled; metrics are low-cardinality and kept at 100%, which is exactly why they can be cheap enough to keep forever and feed your SLOs.

![a matrix comparing logs metrics and traces by the question each answers their cardinality and the cost driver for each pillar so you size and sample them differently](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-4.png)

This is the discipline that companies like Honeycomb built a product category around: high-cardinality, high-dimensionality *event* analysis lets you slice by `user_id` and `build_id` and `region` at query time precisely because those events are stored as traces and wide structured events, not as pre-aggregated metric series. The trade-off is explicit — you pay storage and query cost for the cardinality, and you sample to keep it bounded — but you never pay it by exploding a metric. Keep the line clean: aggregate truth in metrics, per-request truth in traces and logs, and the same `trace_id` joining the two.

## 4. Distributed tracing: following a request across services

A modern API request is rarely served by one process. `POST /payments` in our platform hits a gateway, then the payments API, which calls a fraud-check service, writes to a ledger service (which talks to a database), and fires a notification — five hops, each on a different host, possibly in a different language. When the request takes 720 ms, *which hop owns the time?* Metrics tell you the payments endpoint is slow; they cannot tell you the slowness lives in the ledger's database lock. That is the question **distributed tracing** answers.

Define the vocabulary precisely, because the words are load-bearing:

- A **span** is a single timed operation — one unit of work with a start time, a duration, a name (`POST /v1/payments`, `ledger.write`, `db.query`), and a bag of attributes (`http.status_code`, `db.system`, `tenant_id`). A span is the tracing equivalent of a structured log line, but with a duration.
- A **trace** is the tree of spans for *one* logical request. It has a single **trace id** shared by every span, and each span has a **span id** and a **parent span id**, so the spans assemble into a tree showing causality and timing. The root span is the inbound request; its children are the downstream calls it made.
- **Context propagation** is how the trace id and parent span id travel from one service to the next. Service A, when it calls service B over HTTP, injects the context into a request header; service B extracts it and makes its spans children of A's span. Without propagation you get five disconnected single-span traces instead of one tree.

The standard header for this — and the reason cross-vendor, cross-language tracing works at all — is the **W3C Trace Context** `traceparent` header, an open recommendation that defines a single, vendor-neutral format every tool understands:

```http
POST /ledger/write HTTP/1.1
Host: ledger.internal
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate: payments=t61rcWkgMzE
Content-Type: application/json
```

Read the `traceparent` value left to right: `00` is the version; `4bf92f3577b34da6a3ce929d0e0e4736` is the 16-byte **trace id** (the same id you saw in the structured log object in section 2 — that is not a coincidence, it is the whole point); `00f067aa0ba902b7` is the 8-byte **parent span id** (this caller's current span, which becomes the parent of the callee's span); and `01` is the trace-flags byte whose low bit means "this trace is sampled, record it." The `tracestate` header carries vendor-specific key-value context alongside it. Because the format is a published standard, your gateway (one vendor), your services (an SDK), and your tracing backend (another vendor) all interoperate without a private agreement.

You almost never assemble that header by hand. The vendor-neutral standard for *producing* spans and propagating context is **OpenTelemetry (OTel)** — a CNCF project that gives you one set of SDKs, one wire protocol (OTLP), and one set of semantic conventions (agreed attribute names like `http.route` and `http.status_code`) so you can instrument once and export to any backend. The strategic value is that OTel decouples your instrumentation from your vendor: you put OTel in your code, and you can change your tracing/metrics/logs backend without touching a line of application code. This is the same "design to a standard, not a vendor" principle this series applies to error envelopes (RFC 9457) and trace context (W3C).

Here is auto- plus manual instrumentation of the payments path in Python with OpenTelemetry — the framework auto-instruments the inbound request and the outbound HTTP calls; you add one manual span around the ledger write because that is the business operation you care about timing:

```python
from opentelemetry import trace
tracer = trace.get_tracer("payments-api")

def create_payment(request):
    # The inbound server span and the trace context are created by OTel's
    # framework instrumentation; this code runs INSIDE that span.
    with tracer.start_as_current_span("ledger.write") as span:
        span.set_attribute("tenant_id", request.tenant_id)
        span.set_attribute("amount_cents", request.amount_cents)
        # The outbound call is auto-instrumented: OTel injects the
        # traceparent header so the ledger's spans join THIS trace.
        result = ledger_client.write(request.to_ledger_entry())
        span.set_attribute("ledger.entry_id", result.entry_id)
        return Payment.from_ledger(result)
```

### 4.1 Finding the slow hop

Now the payoff. The trace for our 720 ms payment, drawn as the span tree, immediately localizes the latency. The figure shows the gateway span (5 ms), the payments handler (40 ms of its own work) fanning out to a fraud check (50 ms, fine), a ledger write (600 ms — the culprit), and a notification (25 ms, fine); the ledger span has a child `db.query` span that spent 580 of its 600 ms waiting on a row lock.

![a span tree for one payment request showing the gateway payments api fraud check ledger write and notify spans with the ledger write and its database lock wait owning six hundred of the total seven hundred twenty milliseconds](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-5.png)

#### Worked example: from a p99 alert to a one-line root cause in three minutes

Trace the actual debugging path, because this is where the three pillars pay off together. (1) A **metric** alert fires: `histogram_quantile(0.99, ...)` for `POST /v1/payments` crossed 800 ms. You know *that* it is broken. (2) You open a **trace** for one of the slow exemplars — modern backends link a slow metric sample directly to an example trace id. The span tree shows `ledger.write` at 600 ms with a `db.query` child at 580 ms. You now know *where*. (3) You pivot to **logs** filtered by that `trace_id`, and the ledger service's structured line reads `{"level":"warn","msg":"lock wait timeout","trace_id":"4bf92f35...","table":"ledger_entries","wait_ms":580,"blocking_txn":"txn_8841"}`. You now know *why*: a long-running batch transaction is holding a row lock. Total time from page to root cause: about three minutes, because each pillar handed you cleanly to the next via the shared id. Compare that to the forty minutes of SSH-and-grep from my opening story, where the pillars did not exist and nothing was correlated. That delta — 40 minutes versus 3 — is the entire return on investment for instrumentation, and it compounds on every incident for years.

Two operational notes. **Sampling**: you cannot store a span for every request at scale, so you sample. *Head sampling* decides at the root (keep 1% of traces) — cheap but blind, because you might drop the slow ones. *Tail sampling* buffers the whole trace and decides after it completes (keep all errors and all traces over 500 ms, plus 1% of the rest) — more expensive, far more useful, because it biases retention toward the traces you will actually want. **Trust boundaries**: never let an *external* caller's `traceparent` link into your internal trace without scrutiny, and never propagate your internal trace ids back out to untrusted callers; the correlation id you echo to a client (next section) is a *separate, safe* identifier.

Tracing is paradigm-agnostic, which matters for this series: the same machinery works for gRPC and GraphQL, not just REST. For **gRPC**, OpenTelemetry's instrumentation creates a span per RPC and propagates context through gRPC metadata rather than HTTP headers, so a `payments.Charge` call that fans out to three downstream RPCs traces exactly like the REST path above; the deadline you set on a gRPC call even shows up as the span's timeout boundary. For **GraphQL**, where a single `POST /graphql` can resolve dozens of fields across many backends, tracing is *more* valuable, not less — a span per resolver is how you finally see the N+1 fan-out that a single endpoint metric completely hides. The point is that you instrument once with OTel's vendor-neutral SDK and the wire-level propagation differs by transport, but the trace tree — and the "find the slow hop" workflow — is identical. That is exactly why instrumenting to a standard rather than re-rolling per-protocol telemetry pays off across the whole surface of an API.

## 5. The correlation id: one thread through everything

We have now seen the same identifier appear in three places — the `request_id`/`trace_id` in the structured log, the `trace_id` in the `traceparent` header, and the id a slow metric exemplar points at. Pull that thread out explicitly, because the **correlation id** is the cheapest, highest-leverage piece of observability you will ever ship, and it is the bridge between your telemetry and your *human* support process.

The pattern is a contract between the gateway, the services, the error body, and the support desk:

1. **Mint at the edge.** The gateway (or the first service) generates a `request_id` for every inbound request if the client did not supply one, and stamps it into an `X-Request-Id` header that propagates to every downstream service. Alongside it, the tracing layer establishes the `trace_id`. Some teams use one id for both; keeping them distinct lets you expose the request id externally while keeping the trace id internal.
2. **Stamp every log line and span** with it (the middleware in section 2 does this automatically).
3. **Echo it in every response**, success and failure, as a header — and crucially, **put it inside every error body**, the way the [error design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) prescribes for the `problem+json` envelope. This is the single most useful field in an error:

```http
HTTP/1.1 500 Internal Server Error
Content-Type: application/problem+json
X-Request-Id: req_9f2c7a1e

{
  "type": "https://api.example.com/problems/internal-error",
  "title": "An unexpected error occurred",
  "status": 500,
  "detail": "We could not process this payment. Our team has been notified.",
  "request_id": "req_9f2c7a1e",
  "instance": "/v1/payments"
}
```

4. **Let support search by it.** When a merchant opens a ticket — "payment failed at 2:14 PM, here is the error" — the *only* thing your support engineer needs is that `request_id`. They paste it into the log search, get every structured line for that request across every service, click through to the trace, and see exactly what happened. No reproduction, no guessing, no "can you send a screenshot." The request id you echoed turned a support ticket into a database lookup.

This is observability serving the *product*, not just the on-call. A caller who can quote `req_9f2c7a1e` and get a precise answer experiences a fundamentally more trustworthy API than one who hears "we can't find any record of your request." The correlation id is a developer-experience feature as much as an operational one — which is why it belongs in the contract, documented, and stable. Notice what the error body does *not* contain: no stack trace, no SQL, no internal hostname. It hands the caller a safe, searchable handle (`request_id`) and keeps the sensitive *why* in the logs that only your team can read. That split — safe handle out, full detail in — is the whole discipline.

## 6. SLIs, SLOs, and error budgets: spending reliability on purpose

We defined the SLI/SLO/SLA vocabulary in section 1 and derived that 99.9% over 30 days is a 43-minute budget. Now we make the budget *operational* — the thing that gates releases — and that requires the ladder in the figure below: an SLI you measure, an SLO you target, an error budget you may spend, and a **burn rate** that turns "spending too fast" into a page.

![a layered stack showing the ladder from the measured SLI up to the SLO target then the error budget then the burn rate and finally the release gate that freezes or ships](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-6.png)

First, the nines table, because it grounds every SLO conversation in minutes a human can feel. (Downtime here is the permitted *bad* time per period; an availability SLI is `good/valid`, so the bad budget is $1 - \text{SLO}$ of the period.)

| Availability SLO | Bad fraction | Per 30-day month | Per year |
| --- | --- | --- | --- |
| 99% ("two nines") | 1% | ~7.2 hours | ~3.65 days |
| 99.9% ("three nines") | 0.1% | ~43 minutes | ~8.76 hours |
| 99.95% | 0.05% | ~21.6 minutes | ~4.38 hours |
| 99.99% ("four nines") | 0.01% | ~4.3 minutes | ~52.6 minutes |
| 99.999% ("five nines") | 0.001% | ~26 seconds | ~5.26 minutes |

Read that table as a *cost* table, not an aspiration table. Each nine you add roughly multiplies your engineering cost — redundancy, faster rollback, more on-call, more testing — by a large factor, while the budget shrinks tenfold. Five nines (26 seconds a month) is achievable only for a tiny class of systems and is almost never what your callers actually need. **Pick the lowest number of nines your callers genuinely require**, because every extra nine is paid for with feature velocity you could have spent elsewhere. This is the SLO version of "design by force, not fashion."

You also need *two* SLOs for an API, not one, because there are two ways to fail a caller:

- An **availability SLO**: `successful requests / valid requests ≥ 99.9%`. ("Successful" = not a `5xx`; you decide whether to count `429` and certain `4xx` — usually a `400` is the *caller's* fault and is excluded from the SLI's "valid" denominator or counted as good, so a buggy client cannot burn your budget.)
- A **latency SLO**: `requests faster than 300 ms / valid requests ≥ 99%`. Note this is a *ratio against a threshold*, not "the p99 must be under 300 ms" — though the two are equivalent, the ratio form composes into an error budget exactly like availability does. A request that succeeds but takes 8 seconds satisfies availability and *violates* latency; both matter.

A subtlety that decides whether an SLO is honest or theater: **what counts in the denominator, and what counts as "good."** The SLI is `good events / valid events`, and both halves are choices you must make on purpose. For the denominator — *valid* requests — you almost always exclude requests that are the *caller's* fault: a malformed `400`, an unauthorized `401`, a `404` for a resource that never existed. If a single buggy client hammering a bad endpoint could burn your error budget, your SLO measures the caller's bugs instead of your reliability, which is useless. So `4xx` (client errors) are typically excluded from "valid" or counted as good; only `5xx` (your faults) and timeouts count against you. For the numerator — *good* — you must also decide where the SLI is *measured*: at the load balancer (sees requests that never reached your code, including total outages, which is the honest place); at the service (misses the "we are completely down" case because a down service emits no metrics); or from synthetic probes and client-side RUM (sees what the user saw, including your CDN and DNS). The textbook answer is to measure availability **as close to the user as you can** — at the edge or via probes — so a total outage, the worst failure, actually registers. A service-level metric that goes silent during an outage will show you a *perfect* SLI during your worst hour, which is exactly backwards. Pick the measurement point deliberately and write it down next to the SLO, because two teams measuring "availability" at different points are not measuring the same thing.

### 6.1 The error budget and the burn rate

The error budget is the absolute count of bad events you are allowed in the window. With a 99.9% SLO and, say, 10 million requests in 28 days, the budget is:

$$\text{budget} = 10{,}000{,}000 \times (1 - 0.999) = 10{,}000{,}000 \times 0.001 = 10{,}000 \text{ failed requests}.$$

You may fail 10,000 requests this window and still meet the SLO. The **burn rate** is how fast you are consuming that budget *relative to the rate that would exactly exhaust it over the window.* A burn rate of 1 means you will spend exactly your budget by the end of the window — sustainable, on target. A burn rate of 2 means you are spending twice as fast and will run out at the halfway point. A burn rate of 14.4 means you are on track to burn the *entire month's* budget in $28 \text{ days} / 14.4 \approx 2 \text{ days}$. Burn rate is the derivative that converts a slow-moving budget into a fast-moving alert.

#### Worked example: a burn-rate alert that gates a release

You ship a deploy at 09:00. Over the next hour, `POST /payments` runs 600,000 requests and 3,000 of them return `5xx` — an error ratio of $3000 / 600000 = 0.5\%$. Your SLO allows $0.1\%$. The instantaneous burn rate is:

$$\text{burn rate} = \frac{\text{observed error ratio}}{\text{budgeted error ratio}} = \frac{0.005}{0.001} = 5.$$

At a 5× burn, you will exhaust a 28-day budget in $28 / 5 = 5.6$ days. That is a real fire but not an instant page; it goes to a *slow-burn* alert. Now suppose the deploy is worse: 600,000 requests, 8,640 of them `5xx`, an error ratio of $1.44\%$:

$$\text{burn rate} = \frac{0.0144}{0.001} = 14.4 \implies \text{budget gone in } 28/14.4 \approx 1.94 \text{ days}.$$

That is the canonical Google SRE **fast-burn threshold**: a burn rate of 14.4 over a 1-hour window means you would consume the entire month's budget in under two days, which is a *page-someone-now* event. The recommended pattern is **multi-window, multi-burn-rate** alerts: page on a fast burn (14.4× over 1 hour *and* still elevated over 5 minutes, to avoid flapping on a brief blip) and open a ticket on a slow burn (e.g. 3× over 6 hours). Here is the fast-burn alert as a Prometheus rule:

```yaml
groups:
  - name: payments-slo
    rules:
      - alert: PaymentsFastBudgetBurn
        # 14.4x burn over 1h AND confirmed over 5m, on the 99.9% SLO (0.001 budget).
        expr: |
          (
            sum(rate(http_requests_total{route="/v1/payments",status=~"5.."}[1h]))
            / sum(rate(http_requests_total{route="/v1/payments"}[1h]))
          ) > (14.4 * 0.001)
          and
          (
            sum(rate(http_requests_total{route="/v1/payments",status=~"5.."}[5m]))
            / sum(rate(http_requests_total{route="/v1/payments"}[5m]))
          ) > (14.4 * 0.001)
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Payments burning error budget at >14x — month's budget gone in <2 days"
```

Now the cultural payoff, which is the entire reason error budgets exist as a *management* tool and not just an alerting one. **When the budget is healthy, you ship aggressively** — the team has earned the right to take risk, because they are well within their reliability promise. **When the budget is exhausted, releases freeze** until reliability work earns it back. This is not a punishment; it is an *automatic, depoliticized* decision rule. The product manager who wants to ship and the SRE who wants stability are no longer arguing about feelings — they are reading the same number off the same dashboard. This was Google's key insight in the SRE book: the error budget aligns dev and ops by giving them a shared, objective currency for risk. The figure below shows the freeze in action over a month.

![a timeline showing the monthly error budget starting full then a bad deploy causing a fourteen times fast burn that triggers a release freeze followed by a rollback that protects the remaining budget](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-7.png)

## 7. What to instrument at the boundary, and how to alert on it

Pull the pieces together into a concrete instrumentation policy for the API boundary. **Every single request, on its way in and out, records a fixed set of dimensions**, and from those dimensions every dashboard and SLO is derived. For the Payments & Orders API, every request emits:

- **`method`** and **`route`** — the *templated* route (`POST /v1/payments`, not `POST /v1/payments/pay_88123`), so the path parameter does not explode cardinality.
- **`status`** — the HTTP status code, as a metric label *and* a log field. Per-status is non-negotiable: `200`, `402` (declined), `409` (idempotency conflict), `429` (rate limited), `500` must be distinguishable, because they mean completely different things to your SLI.
- **`latency`** — observed into the duration histogram, and logged on the tail.
- **`caller`** — the API client / key id (low cardinality, bounded by your client count), as a metric label, so you can answer "is the elevated error rate one misbehaving caller or everyone."
- **`tenant`** — the tenant/merchant id. Bounded if you have hundreds of tenants; if you have millions of *end users*, the user id goes in logs/traces only, never a metric label (section 3.2).
- **The four golden signals** fall out of these directly: **latency** (the histogram), **traffic** (the request rate), **errors** (the `5xx` rate), and **saturation** (USE gauges on the pool, the queue, CPU).

The two comparison tables that pin down the methodology:

| | RED (requests) | USE (resources) | Four golden signals |
| --- | --- | --- | --- |
| **Measures** | Rate, Errors, Duration | Utilization, Saturation, Errors | Latency, Traffic, Errors, Saturation |
| **Best for** | Request-driven services (your API) | Pools, CPU, queues under the service | Any user-facing system |
| **Coined by** | Tom Wilkie | Brendan Gregg | Google SRE |
| **Payments example** | p99 on `POST /payments`, 5xx ratio | DB pool 95% used, queue depth 400 | p99 latency + 5xx rate per route |

| | SLI | SLO | SLA |
| --- | --- | --- | --- |
| **What it is** | A *measurement* (good/valid ratio) | A *target* over a window | A *contracted* target with penalties |
| **Audience** | Engineers (computed from telemetry) | The team (internal promise) | Customers (external, legal) |
| **Example** | `1 - 5xx/total = 0.9994` | "≥ 99.9% over 28 days" | "≥ 99.5% or we credit 10%" |
| **Relationship** | Feeds the SLO | Stricter than the SLA | Looser than the SLO, on purpose |

### 7.1 Page on symptoms, not causes

The final discipline, and the one that decides whether your pager is a tool or a source of trauma: **alert on symptoms your users feel, not on causes that may or may not lead to symptoms.** A symptom is "the p99 of `POST /payments` is over budget" or "the 5xx ratio crossed the burn threshold" — a user is, right now, having a bad time. A cause is "the DB connection pool is 90% utilized" or "a node restarted." Causes are valuable *diagnostic* signals, but if you *page* on every cause you will get woken for a pool at 90% that drained itself in thirty seconds and never affected a single user — and after a week of that, you will silence the pager and miss the real one. The figure shows the routing rule.

![a decision tree routing a fired signal into a user-facing symptom that pages on-call versus an internal cause with no user impact that only files a ticket](/imgs/blogs/observability-for-apis-logs-metrics-traces-and-slos-8.png)

The rule that follows: **page on SLO burn (the symptom); ticket on saturation and other causes (the early warning).** A pool at 90% becomes a *page* only when it actually causes the latency or error SLO to burn. This keeps the pager rare and trustworthy — every page means a user is hurting and the on-call's job is to make it stop — and it keeps the saturation signals as the cheaper, non-paging early warning you review during the day. Tie this back to the contract: your SLO *is* the definition of "a user is hurting," so paging on SLO burn means paging exactly when you are breaking the promise you published. The pager and the contract say the same thing.

A few more boundary signals worth emitting and dashboarding, none of which should page on their own: dependency health (the error and latency RED metrics *of your downstream calls*, so you can tell "we are slow" from "the ledger is slow"); rate-limit rejections (`429` count per caller — a spike is either an abusive client or your limits being too tight, per the [rate-limiting post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection)); and deploy markers (annotate the dashboard with each release, so a metric that moves at 09:00 lines up visibly with the 09:00 deploy — the single fastest way to spot a bad release).

### 7.2 Dashboards that an on-call can read at 2 a.m.

A dashboard is not a place to dump every metric you collect; it is the *first screen* an on-call engineer opens when paged, and its job is to answer "is it broken, where, and how bad" before they are fully awake. Design it like an interface, not a junk drawer. The discipline that works, again from the RED idea, is **one consistent service dashboard layout, repeated for every service**, so a half-asleep engineer debugging a service they have never touched already knows where everything is. A useful layout reads top-to-bottom as a triage path:

1. **The SLO row, first and biggest**: the availability SLI and the latency SLI as big numbers against their targets, plus the *remaining error budget* for the window as a fuel gauge. This single row answers "are we breaking the promise, and how much budget is left." Nothing else belongs above it.
2. **The RED row**: rate (requests/sec), error ratio (5xx and 4xx as *separate* lines — a 4xx spike is the caller's problem, a 5xx spike is yours), and a latency panel showing **p50, p95, and p99 together** so the *spread* between them is visible at a glance. A widening gap between p50 and p99 is the classic "the tail is degrading" signal that a single number hides.
3. **The per-route and per-caller breakdown**: the same RED, split by route and by caller, so "errors are up" instantly becomes "errors are up *on `POST /payments`, for caller `acme`*" — the difference between a five-minute fix and an hour of flailing.
4. **The dependency row (USE-ish)**: the RED metrics of your downstream calls plus saturation gauges (DB pool in use, queue depth, inflight requests). This is where "we are slow" resolves into "the ledger is slow" or "we have exhausted the connection pool."
5. **Deploy and incident markers** overlaid on every time series, so a metric that turns red at 09:00 visibly lines up with the 09:00 release.

The anti-pattern is the "wall of graphs" dashboard with forty panels and no hierarchy, where the signal is buried in noise and the on-call scrolls past the one panel that mattered. If a panel has never helped resolve an incident, it does not belong on the *triage* dashboard — move it to a detailed drill-down view you open *after* triage points you there. The percentiles-not-averages rule applies to dashboards too: never put a lone "average response time" panel where a percentile belongs, because — as the worked example proved — that panel will be green during the exact incident you most need to see. The dashboard that lied to me at 02:14 had a single average-latency line; the dashboard that would have saved forty minutes had a p99 panel split by route, with the ledger dependency's RED right below it.

## 8. Synthetic and real-user monitoring, and the audit log

Two more monitoring surfaces complete the picture, plus a security-logging note that is easy to forget and expensive to forget.

**Real-user monitoring (RUM)** is the telemetry we have discussed so far: it measures *actual* requests from *actual* callers. Its strength is fidelity — it is the ground truth of what users experienced. Its weakness is that it only sees traffic you *have*; if a route gets no traffic at 3 a.m., RUM tells you nothing about whether it works at 3 a.m.

**Synthetic monitoring** fills that gap: a scheduled probe that calls your API on a fixed cadence (every 30 seconds) from multiple regions, exercising the critical path — create an order, charge a payment, fetch a receipt — and asserting on status, latency, and body shape. Synthetics give you (1) coverage of low-traffic paths, (2) a consistent baseline unaffected by traffic mix, (3) the *outside* view including DNS, TLS, and your CDN that RUM (measured inside your service) misses, and (4) the first alert when you are *completely* down and there is no real traffic to generate an error metric. A synthetic check for the payment path:

```bash
# Synthetic probe run every 30s from 3 regions; alerts if status != 201 or latency > 800ms.
curl -sS -o /dev/null -w '%{http_code} %{time_total}s\n' \
  -X POST https://api.example.com/v1/payments \
  -H 'Authorization: Bearer <synthetic-test-token>' \
  -H 'Idempotency-Key: synthetic-probe-fixed-key' \
  -H 'Content-Type: application/json' \
  -d '{"amount_cents": 100, "currency": "USD", "source": "tok_test_synthetic"}'
```

Use them together: synthetics tell you *the path is up and fast from the outside*; RUM tells you *what real users actually experienced*. Synthetics catch the total outage and the cold path; RUM catches the failure that only your largest merchant's payload triggers. Neither replaces the other.

Finally, **audit and security logging is a distinct stream from operational logging, and it has different rules.** Operational logs are for debugging and are sampled, retained briefly, and stripped of PII. An **audit log** records *who did what to what, when* — "client `acme` issued `POST /v1/refunds` for `pay_88123`, amount \$49.99, at 14:02:09, result success" — and exists for security forensics, compliance, and answering "did this actually happen and who authorized it" months later. Audit logs are **append-only, tamper-evident, retained for years, never sampled, and access-controlled**, because they are evidence. Security signals also deserve their own metrics and alerts: a spike in `401`/`403`, a single caller hitting `429` thousands of times, a sudden geographic shift in callers — these are abuse or attack signals, and they page a security on-call, not the service on-call. Keep the streams separate; conflating audit and debug logging means you either over-retain debug noise (expensive, a privacy risk) or under-retain audit evidence (a compliance failure).

## 9. Case studies: where these ideas come from

These practices are not invented here; they are the distilled, battle-tested conventions of teams that operate enormous APIs. Naming the sources accurately matters, both for credit and so you can read the originals.

**Google SRE and the error budget.** The error-budget model — pick an SLO below 100%, treat the complement as a budget, gate releases on the burn rate, and use it to align dev and ops — is the central operational idea of Google's *Site Reliability Engineering* book (O'Reilly, 2016) and its sequel *The Site Reliability Workbook*. The four golden signals (latency, traffic, errors, saturation) and the multi-window multi-burn-rate alerting approach (including the canonical 14.4× fast-burn-over-1-hour figure) come directly from that body of work. If you read one source on SLOs, read the SRE workbook's "Implementing SLOs" chapter.

**The RED method (Tom Wilkie).** RED — Rate, Errors, Duration for every service — was articulated by Tom Wilkie (then at Weaveworks, later Grafana Labs) as a request-centric counterpart to Brendan Gregg's resource-centric **USE method** (Utilization, Saturation, Errors). RED is deliberately simple precisely so that every service is instrumented the same way and every dashboard looks the same, which is what makes an on-call engineer able to debug a service they have never seen. The discipline of "the same three metrics on every service" is the point.

**OpenTelemetry adoption.** OpenTelemetry, formed in 2019 from the merger of the earlier OpenTracing and OpenCensus projects, is now a CNCF project and the de facto vendor-neutral standard for traces, metrics, and logs. Its strategic contribution is decoupling instrumentation from backend: you instrument with OTel SDKs and the OTLP protocol once, and you can switch observability vendors without re-instrumenting. It carries the **W3C Trace Context** `traceparent`/`tracestate` recommendation as its propagation standard, which is why cross-vendor, cross-language traces connect at all.

**Honeycomb and high cardinality.** Honeycomb (founded by ex-Facebook/Parse engineers, notably Charity Majors and Christine Yu) built a product and a strong point of view around the limits of pre-aggregated metrics: when you need to ask "why is *this* user, on *this* build, in *this* region slow," low-cardinality metrics cannot answer, because the answer lives in dimensions you did not pre-aggregate. Their model stores wide, high-cardinality structured *events* (effectively traces) and computes aggregates at query time — paying storage and sampling cost in exchange for the ability to slice by any dimension after the fact. It is the practical embodiment of "put high-cardinality detail in events/traces, not metric labels."

**RFC 9457 and W3C Trace Context as standards.** The reason the correlation id, the error body, and the trace header all interoperate is that each follows a published standard: error bodies follow **RFC 9457** (`problem+json`), trace propagation follows the **W3C Trace Context** recommendation, and HTTP semantics follow **RFC 9110**. Designing to the standard rather than a private convention is what lets a caller, a gateway, and three backends from different teams all agree on the wire — the same principle that runs through this entire series.

**Exemplars: the bridge from a metric back to a trace.** A practice worth calling out, because it is what made the three-minute root-cause workflow in section 4.1 possible, is the **exemplar** — a feature of the Prometheus/OpenMetrics ecosystem where a metric sample (a histogram bucket observation) carries an attached `trace_id` of one representative request that landed in that bucket. The payoff is enormous: you are looking at a p99-latency panel, you see the slow bucket light up, you click the exemplar, and you land *directly on a trace* of an actual slow request. Without exemplars, "the p99 is bad" leaves you hunting for a slow trace by hand among millions; with them, the metric *is* a launchpad into the trace, which is a launchpad (via `trace_id`) into the logs. This is the concrete mechanism behind "metrics tell you that, traces tell you where, logs tell you why" — the pillars are not three separate tools you mentally correlate; they are linked by shared ids, and exemplars are the link from the cheapest pillar to the richest. When you evaluate an observability stack, ask specifically whether it threads exemplars from histograms to traces, because that single feature is the difference between a forty-minute incident and a three-minute one.

## 10. When to reach for this (and when not to)

Observability is not free, and more of it is not always better. Be decisive about the trade-offs.

**Do instrument RED metrics and structured logs from day one, on every service.** They are cheap, they are the foundation everything else builds on, and retrofitting them during an incident is too late. There is no project too small for "structured logs with a request id" and "rate, errors, duration per route." This is the floor, not a luxury.

**Do adopt OpenTelemetry rather than a vendor SDK**, even for a small system, *because* it is small now — the cost of instrumenting with OTel is the same as a vendor SDK, but it preserves your freedom to change backends later without re-instrumenting. The exception: if your platform's managed observability has a deeply integrated agent and you will never leave it, the lock-in may be an acceptable price for zero-config; weigh it honestly.

**Do not write SLOs you have no intention of defending.** An SLO that never gates a release and whose breach triggers no action is theater. If you are not willing to freeze a release when the budget is exhausted, you do not have an SLO; you have a number on a slide. Better to have *one* defended SLO (availability on your critical path) than ten decorative ones.

**Do not start with five nines, or with an SLO at all if you have no traffic.** A pre-launch service with ten requests a day cannot compute a meaningful SLI — the denominator is too small for a percentage to mean anything. Start with structured logs and RED metrics, get real traffic, *then* set an SLO grounded in what callers actually need. And never set more nines than your callers require; each nine is paid for in velocity.

**Do not put high-cardinality identifiers in metric labels — ever.** No `user_id`, no `request_id`, no `email`, no raw URL. This is the mistake that turns a \$200/month metrics bill into a \$20,000 one overnight (section 3.2). Those identifiers belong in logs and traces, which are built to carry them.

**Do not alert on the average latency, and do not alert on causes.** Alert on percentiles (the symptom your worst users feel) and on SLO burn (the symptom that means you are breaking your promise). Saturation and other causes are non-paging early warnings. A pager that fires on causes is a pager you will learn to ignore, and an ignored pager is worse than no pager because it gives false confidence.

**Do not log secrets or PII, and do not conflate audit and debug logs.** Redact by allowlist at the logging boundary; keep the append-only, long-retained audit stream separate from the sampled, short-lived debug stream. The PAN-in-the-log mistake from the opening is a reportable security incident, not a typo.

## 11. Key takeaways

- **You cannot operate a contract you cannot see.** Metrics tell you *that* it broke, traces tell you *where*, logs tell you *why* — and a shared correlation id stitches all three into one debuggable timeline. Instrument all three from day one.
- **Structured, leveled JSON logs with a correlation id, no secrets, and sampling** are the floor. Stamp the id automatically in middleware; redact by allowlist; never make correct logging depend on every developer remembering to do it.
- **RED for requests, USE for resources, the four golden signals as the union.** Record Rate, Errors, and Duration per route and per status, as counters and histograms, on every service.
- **Alert on percentiles, never the average.** An average of 80 ms hides a p99 of 4000 ms; the high percentile is the only number that represents your worst-affected users. And you cannot average percentiles — aggregate the histogram buckets and compute the quantile once.
- **Guard cardinality with your life.** A metric label's value set must be bounded and small; high-cardinality identifiers (user id, request id, order id) go in logs and traces, never a metric label, or your bill and your backend both explode.
- **Propagate W3C `traceparent` and instrument with OpenTelemetry.** One vendor-neutral standard for the wire and the SDK means cross-service, cross-vendor traces connect, and you can change backends without re-instrumenting.
- **An SLO turns "reliable enough" into a defendable number.** 99.9% over 30 days is a 43-minute monthly error budget; pick the *fewest* nines your callers need, set both an availability and a latency SLO, and never aim for 100%.
- **The error budget gates releases via the burn rate.** Healthy budget → ship fast; exhausted budget → freeze. A 14.4× burn over an hour means the month's budget is gone in under two days — page now. The budget depoliticizes the dev-vs-ops fight by making risk a shared number.
- **Page on symptoms, not causes, and keep audit logging separate.** Page on SLO burn (a user is hurting); ticket on saturation (an early warning). Keep the append-only audit stream apart from the sampled debug stream.

## 12. Further reading

- **Google, *Site Reliability Engineering* and *The Site Reliability Workbook*** (O'Reilly) — the source for SLIs/SLOs, error budgets, the four golden signals, and multi-window burn-rate alerting. The "Implementing SLOs" chapter is the canonical reference.
- **OpenTelemetry documentation** (`opentelemetry.io`) — the vendor-neutral standard for traces, metrics, and logs; SDKs, the OTLP protocol, and the semantic conventions for HTTP attributes.
- **W3C Trace Context** (`www.w3.org/TR/trace-context`) — the `traceparent` and `tracestate` header specification that makes distributed tracing interoperable across tools and languages.
- **Tom Wilkie, "The RED Method"** and **Brendan Gregg, "The USE Method"** — the two complementary instrumentation methodologies; read both to cover requests and resources.
- **RFC 9457** (problem+json) and **RFC 9110** (HTTP semantics) — the standards your error bodies and status codes follow, so your telemetry's `status` and error `request_id` fields are well-defined.
- Within this series: the intro hub **["What is an API"](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems)**; the capstone **["The API Design Playbook"](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2)**; and the siblings **[error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract)** (the correlation id in the error body), **[API performance](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency)** (the tail latency you go fix), **[rate limiting](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection)** (the `429` signals you watch), and **[deprecation and sunset](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely)** (the long-lived contract these SLOs defend).

---
title: "Metrics, Logs, and Traces: When to Use Which"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn the three pillars of observability from first principles — why metrics answer what and when, traces answer where, and logs answer why — and how to chain them so you debug a latency spike in minutes without blowing your telemetry bill on a cardinality explosion."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "observability",
    "metrics",
    "logs",
    "tracing",
    "opentelemetry",
    "prometheus",
    "cardinality",
    "monitoring",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/metrics-logs-and-traces-when-to-use-which-1.png"
---

At 14:03 on a Tuesday, a single number on a dashboard doubled. The p99 latency on your checkout endpoint went from a comfortable 250 milliseconds to a sweaty 1.4 seconds, and a page fired. You now have a problem and three completely different tools sitting in front of you to solve it, and most engineers — even good ones — reach for the wrong one first. They open the logs, because logs feel like "the truth," and they start grepping through millions of lines hoping the answer falls out. Twenty minutes later they are still grepping, the budget is still burning, and the actual answer — that one of twelve downstream service hops got slow because a database query lost its index — was sitting two clicks away in a trace the whole time.

This post is about not doing that. It is about knowing, before the page even fires, which of the three pillars of observability answers which question, so that during an incident your hands move on muscle memory instead of panic. Metrics, logs, and traces are not three competing products you pick between; they are three complementary instruments that each see a different slice of reality, and the skill that separates a senior on-call from a junior one is knowing the handoff: which signal tells you *what* broke, which tells you *where*, and which tells you *why*. Get the order right and a multi-service latency mystery collapses from a forty-minute grep marathon into a four-minute investigation.

By the end of this post you will be able to do five concrete things. You will be able to look at any observability question and name the pillar that answers it cheapest. You will understand the cardinality-versus-volume-versus-sampling cost model well enough to predict, before you ship a label, whether it will quietly ten-thousand-x your metrics bill. You will be able to run the metric-to-trace-to-log workflow on a real latency spike. You will know the difference between *monitoring* (watching the failures you predicted) and *observability* (asking new questions about the failures you did not), and why that difference is the whole game. And you will understand how OpenTelemetry lets you instrument your code once and get all three signals correlated by a single trace id. The figure below is the map we will keep returning to: three pillars, three questions, one workflow.

![A vertical stack diagram showing the three pillars of observability where metrics answer what and when, traces answer where, and logs answer why, all correlated by a trace id and governed by a cost lever](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-1.png)

This is the second post in the observability track of our field manual, and it sits in a specific place in the larger loop. The whole series follows one spine: you **define** reliability with an SLI and an SLO, you **measure** it with observability, you **spend** the resulting error budget, you **respond** to incidents, you **learn** from them in postmortems, and you **engineer** the permanent fix. This post is the foundation of the *measure* stage. Before you can alert on an SLO, burn an error budget honestly, or debug an incident at three in the morning, you need telemetry — and you need to understand what kind of telemetry does what. The three pillars are the raw material of every other measurement in this series. If the intro to the whole field manual is new to you, start with [the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset); it frames why reliability is a number you engineer rather than a wish.

## 1. The three pillars, and the one question each answers

Let us define the three pillars precisely, because the entire post hangs on these definitions being crisp. A **metric** is a number measured over time. A **log** is a timestamped record of a discrete event. A **trace** is the recorded path of a single request as it moves through a system. That is it. Everything else — Prometheus, Loki, Jaeger, OpenTelemetry, your bill — is plumbing around those three ideas. The reason all three exist, and the reason you cannot collapse them into one, is that each one is shaped to answer a different question and to make a different trade-off.

A **metric** answers *what* happened and *when*. It is a cheap, aggregable number sampled at regular intervals: requests per second, error ratio, p99 latency, CPU saturation, queue depth. The defining property of a metric is that its storage cost does not grow with how much traffic you serve — a counter that increments a billion times costs the same to store as one that increments a thousand times, because you only ever keep the *aggregate* over each scrape interval, not the individual events. That is what makes metrics the right substrate for dashboards, for alerts, and for your service level indicators, the SLIs that quantify reliability. A metric is how you know, at 14:03, that p99 doubled. What a metric fundamentally cannot do is carry detail. It cannot tell you *which* user, *which* request, or *which* line of code, because the moment you try to attach that kind of identifying detail to a metric you trigger the cardinality explosion we will spend a whole section on. Metrics are deliberately low-cardinality. That limitation is not a bug; it is the price of the bounded cost that makes them safe to alert on.

A **log** answers *why*. It is a discrete, timestamped event with as much detail as you care to write: the exception and its stack trace, the decision a piece of code made and the inputs that drove it, the exact SQL that ran, the user id, the request id, the feature flags that were on. A log line is the highest-fidelity record you have of a single moment. Its strength is detail; its weaknesses are two. First, logs are expensive at volume, because unlike a metric, every single event costs storage — a service doing fifty thousand requests per second writing even one log line per request produces a torrent of data that you pay to ingest, index, and retain. Second, logs are hard to aggregate. You can grep a log, you can index it, you can sum it up after the fact, but a pile of discrete events is not naturally a time series, and trying to compute a reliable p99 by post-processing raw logs is both slow and lossy compared to a metric that was built for it. Logs tell you why one thing happened. They are a terrible way to find out that something happened at all.

A **trace** answers *where*. It follows one request across every service it touches and records each hop as a **span** — a named, timed unit of work with a start, an end, and a set of attributes. The spans of one request share a single **trace id** and nest into a tree, so a trace shows you that the request spent twelve milliseconds in auth, forty in the cart service, eighty in pricing, and one-point-one seconds parked in a database call. The thing a trace gives you that nothing else can is **cross-service causality**: in a system of a dozen microservices, a trace is the only signal that can tell you *which hop* of the twelve actually ate the latency, rather than leaving you to guess and check each service's own dashboards in turn. A metric on the gateway tells you the whole request was slow; only a trace tells you *where* inside the request the time went. The catch is that traces are usually **sampled** — you keep a fraction of them, often one percent or less at high volume — because keeping a full span tree for every request would cost as much as keeping a log line for every request, for the same reason. Sampling is the trace's version of the trade-off the other two pillars also make.

Notice the shape of this. Three pillars, three questions — what, where, why — and three different cost trade-offs that are not coincidences but consequences of the question each answers. The pillar that has to be cheap enough to alert on (metrics) gives up detail. The pillar that has to carry full detail (logs) gives up cheapness. The pillar that has to follow causality across a fleet (traces) gives up completeness through sampling. You do not get to have all three properties in one signal; the universe does not offer that deal. So you carry all three pillars and you learn the handoff between them. Everything else in this post is that handoff, made concrete.

It is worth pausing on *why* you cannot just have one super-signal that does everything, because engineers new to observability often ask exactly that. "If a log can carry any field, and I can derive a count or a latency from logs, why do I need metrics at all? Why not log everything and compute the rest?" The answer is the cost model, and it is fundamental, not incidental. To compute a reliable p99 latency from logs, you must store and process *every* latency observation — every request's duration as a discrete event — and then sort or approximate quantiles over them on every query. At low volume that is fine. At fifty thousand requests per second it is ruinous: you are paying log-volume prices to answer a question that a histogram answers for the price of a dozen bucket counters. Conversely, you cannot make a metric carry the stack trace of the one exception that caused the spike, because a metric is an aggregate by construction — it has thrown away the individual events that a stack trace lives inside. And you cannot make either one reconstruct the causal tree of a request across services, because neither carries the parent-child span relationships that a trace is built around. The three signals are not three implementations of the same idea that history happened to leave us with; they are three genuinely different data structures — an aggregated time series, a stream of discrete records, and a tree of timed spans — and each is the *right* structure for its question and the *wrong* structure for the others. That is why the boundary between them is real and worth respecting.

There is also a useful framing borrowed from systems thinking. Metrics live in the world of **aggregates**: they answer questions about populations of events ("what fraction of all requests in the last five minutes failed?"). Logs and traces live in the world of **instances**: they answer questions about individual events ("what happened to *this* request?"). Monitoring and alerting are aggregate questions, which is why they ride on metrics. Debugging a specific failure is an instance question, which is why it rides on logs and traces. When you feel uncertain which pillar to reach for, ask yourself whether your question is about a *population* or an *individual* — that single distinction routes most questions correctly before you even think about cost.

## 2. The complementary workflow: what, then where, then why

The single most useful thing in this entire post is the order. When something breaks, you investigate in a fixed sequence: **metric, then trace, then log.** What and when, then where, then why. Each pillar narrows the search space for the next, and skipping a step is the mistake that turns a four-minute investigation into a forty-minute one.

Walk it through. The metric is your tripwire and your first orientation: an alert on an SLI tells you *that* something is wrong and *when* it started, and a quick look at a dashboard tells you the rough shape — error rate up, latency up, one region or all of them, started at 14:03 sharp or ramped over an hour. The metric does not tell you why and it should not; its job is to point. Now you take the *when* and the *what* and you jump to a trace — ideally an **exemplar**, which is a trace id that Prometheus can attach to a specific bucket of a latency histogram, so that from the spiking p99 bar on your dashboard you can click straight through to an actual slow request that landed in that bucket. The trace tells you *where*: it shows the span tree, and one span — the database call — is fat, 1.1 seconds out of the 1.4-second total, while every other hop is fast. Now you know the *where*, and the search space has collapsed from "the entire system" to "this one database call." Finally you take the trace id and you pull the logs for that span, and there is your *why*: a structured log line emitted inside that database call, with the query text and a note that it did a full table scan because an index got dropped in last night's migration. Three pillars, three clicks, three questions answered in the only order that works.

Why does the order matter so much? Because each pillar is good at narrowing and bad at the others' jobs. If you start with logs — the instinct of most engineers, because logs feel authoritative — you are searching millions of undifferentiated events with no idea *when* the problem started or *where* in the call graph to look. You are using the most expensive, least aggregable signal as a search tool, which is exactly backwards. If you start with traces, you have no idea whether anything is even wrong, because a single trace is a sample of one and tells you nothing about the aggregate. The metric has to come first because only the metric sees the whole and sees the trend. The trace has to come second because only the trace localizes across services. The log comes last because only the log has the detail, and by the time you reach it you have narrowed the haystack from millions of lines to the handful attached to one span. The figure below shows this exact investigation on a real clock.

![A horizontal timeline of a latency spike investigation showing a metric alert at 14:03, an exemplar trace at 14:05, the slow span found at 14:06, the explaining log line at 14:07, and the fix shipped restoring p99 by 14:40](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-4.png)

This handoff is also why the three pillars have to be *correlated*, not just *collected*. Having metrics, logs, and traces in three separate systems with no shared identifier is like having three witnesses to a crime who refuse to talk to each other. The metric says "latency spiked at 14:03," the trace system has a million traces from that minute, and the log system has ten million lines, and you are back to guessing which trace and which lines belong together. The thing that makes the workflow fast is that a metric exemplar carries a trace id, and a span carries a trace id, and a log line carries that same trace id, so each handoff is a click and not a search. We will see exactly how OpenTelemetry wires that up in section 7. First, let us look at each pillar's real artifacts, because the ideas only become real when you can see the query, the span, and the log line.

There is a deeper reliability principle hiding in this workflow that is worth naming, because it connects observability back to the rest of the SRE discipline. The reason you invest in this metric-trace-log chain at all is **mean time to recovery** — MTTR, the average time from when an incident starts to when it is resolved. MTTR decomposes into two parts: time to *detect* (how long until you know something is wrong) and time to *diagnose-and-fix* (how long from knowing to resolving). Metrics and symptom-based alerting attack the first part — a good SLI alert detects user pain in seconds to minutes instead of waiting for a customer to complain. Traces and logs attack the second — the faster you can localize and explain a failure, the faster you fix it. The whole three-pillar investment is, at bottom, an MTTR-reduction program, and that is why it sits in the *measure* stage of the series loop: you measure not for its own sake but so that when the error budget starts burning, you can see *what* is spending it, find *where* it is leaking, and learn *why* fast enough to stop the bleed before the budget is gone. Observability is the instrument panel that makes the rest of reliability engineering possible.

#### Worked example: the MTTR arithmetic of getting the order right

Make the speed claim concrete. Take a team whose cross-service latency incidents historically took 45 minutes to diagnose because they investigated by grepping each service's logs in turn — the *why* pillar used to answer a *where* question, one service at a time, six services on average at roughly seven minutes per service of pull-and-grep. Now give them the workflow: a metric alert that fires within 60 seconds of the spike (time to detect), an exemplar trace that localizes the slow span in about 2 minutes (where), and a filtered log read of that one span that surfaces the cause in another 2 minutes (why). Time to diagnose drops from 45 minutes to roughly 4. If incidents like this happen, say, twice a month and each one burns error budget at the service's full error rate while unresolved, cutting 41 minutes off each diagnosis is 82 minutes a month of budget saved, plus 41 minutes a month of on-call human time returned, plus — the part that does not show on a dashboard — an on-call who is calm instead of frantic because the tooling does the narrowing for them. The arithmetic is illustrative, but the shape is real and repeatable: the order of the pillars is not a style preference, it is the single biggest lever on time-to-diagnose, and time-to-diagnose is most of MTTR for anything past the smallest systems.

## 3. Before the pillars: what to even measure

Knowing that metrics answer *what* is not the same as knowing *which* what to measure. A service emits a near-infinite number of possible numbers, and a junior team's instinct is to graph all of them, which produces a wall of dashboards nobody reads and no signal during an incident. The discipline that fixes this predates the three-pillar framing and is worth carrying into it: the **four golden signals** and the **RED** and **USE** methods. They tell you which handful of metrics actually matter so the *what* pillar stays small, sharp, and alert-able.

The **four golden signals**, from the Google SRE practice, are **latency** (how long a request takes), **traffic** (how much demand the system is under — requests per second), **errors** (the rate of failed requests), and **saturation** (how full the system's most constrained resource is — CPU, memory, disk, connection pool). If you can only watch four numbers per service, watch these four, because between them they catch nearly every user-facing failure: the service is slow (latency), overwhelmed (traffic plus saturation), or broken (errors). The genius of the four golden signals is that they are *symptom-oriented* — they measure what the user experiences, not the dozens of internal causes that might produce it — which is exactly the property that makes a good alert, as we saw with the symptom-based error-ratio rule.

Two complementary mnemonics specialize the golden signals for two common cases. **RED** — Rate, Errors, Duration — is the request-driven view, ideal for any service that handles requests: the *rate* of requests, the rate of *errors*, and the *duration* (latency) distribution. It is the golden signals minus saturation, focused on the request flow, and it is the right default for an API or a web service. **USE** — Utilization, Saturation, Errors, from Brendan Gregg — is the resource-driven view, ideal for diagnosing infrastructure: for each resource, its *utilization* (percent busy), its *saturation* (queued work it cannot get to), and its *errors*. RED tells you the *service* is unhealthy; USE tells you *which resource* is the bottleneck causing it. A practiced on-call reads RED to know there is pain and USE to know what is starved.

The reason this section comes *before* the pillars in practice is that these methods are how you keep the metrics pillar low-cardinality and high-signal. You do not metric everything; you metric the golden signals per service (a small, bounded set), you put the high-cardinality detail in logs and traces, and your dashboards become readable and your alerts become trustworthy. When you instrument a new service, the first question is not "what can I measure?" but "what are the RED metrics for this thing?" — and the answer is three or four series, not three hundred. That restraint is what makes the *what* pillar cheap enough to alert on, which is the whole reason metrics exist.

## 4. Metrics in practice: cheap numbers over time

A metric in the Prometheus model is a named time series with a set of labels, sampled on a scrape interval. There are four metric types and you should know what each is for, because choosing the wrong one is a common and expensive mistake. A **counter** only ever goes up (until a process restart resets it): total requests, total errors, total bytes sent. You never read a counter's raw value; you read its *rate* of change. A **gauge** can go up or down and represents a current value: memory in use, queue depth, number of in-flight requests, temperature. A **histogram** buckets observations into ranges and lets you compute quantiles after the fact — this is how you get a p99 latency without storing every individual latency. A **summary** computes quantiles client-side; prefer histograms in almost all cases because they aggregate across instances and histograms are what `histogram_quantile()` works on.

Here is the canonical pair you will write a hundred times: a counter for requests labeled by status, and a histogram for latency. The PromQL to turn them into the two numbers an SRE actually watches — the error ratio and the p99 — looks like this.

```promql
# Error ratio over a 5-minute window: a service level indicator for availability.
# This is a number between 0 and 1; multiply by 100 for a percentage.
sum(rate(http_requests_total{job="checkout", code=~"5.."}[$__rate_interval]))
  /
sum(rate(http_requests_total{job="checkout"}[$__rate_interval]))

# p99 latency over a 5-minute window, computed from a histogram.
# le is the "less than or equal to" bucket boundary label.
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket{job="checkout"}[$__rate_interval])) by (le)
)
```

Read what those two queries cost. The `http_requests_total` counter has labels for `job` and `code`. If `job` has ten values and `code` has, say, eight realistic values (a handful of 2xx, 3xx, 4xx, 5xx), that is eighty time series — eighty numbers that Prometheus keeps and updates regardless of whether you serve a hundred requests per second or a hundred thousand. *The traffic volume does not change the storage.* That is the whole magic of a metric, and it is why metrics are the only signal cheap enough to evaluate an alert rule against every fifteen seconds, forever, on every service you run. The p99 query reads from a histogram whose cost is the number of buckets times the label combinations — a dozen buckets across our eighty series is under a thousand numbers, and from those thousand numbers you can answer "what is the 99th percentile latency right now" for the cost of a single arithmetic pass. You could not do that from logs without reading and sorting every event.

The proof that metrics are the right substrate for reliability shows up the moment you write a recording rule and an alert. A recording rule precomputes an expensive query on a schedule so your dashboards and alerts read a cheap, named series instead of re-evaluating the raw PromQL every time.

```yaml
groups:
  - name: checkout-sli
    interval: 30s
    rules:
      # Precompute the error ratio as a named SLI series, evaluated once per 30s.
      - record: job:checkout_errors:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[5m]))
            /
          sum(rate(http_requests_total{job="checkout"}[5m]))
  - name: checkout-alerts
    rules:
      # Page only when the SLI is bad AND sustained — symptom-based, not cause-based.
      - alert: CheckoutHighErrorRatio
        expr: job:checkout_errors:ratio_rate5m > 0.01
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "Checkout error ratio above 1% for 5 minutes"
          runbook: "https://runbooks.example.com/checkout-errors"
```

That alert pages on a *symptom the user feels* — more than one percent of checkouts failing for five straight minutes — not on a cause you guessed in advance. The metric is what makes that possible: it is the aggregate, evaluated cheaply and continuously, that turns "is the service healthy?" into a number with a threshold. We go deep on symptom-versus-cause alerting and on multi-window burn-rate alerts in the alerting posts of this track; for the foundations of metrics themselves — naming, the RED and USE methods, histogram buckets, recording rules at scale — see the sibling post [metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right). The one-line takeaway here is that a metric is a *cheap, aggregable, bounded-cost number over time*, and the price of those four properties is that it cannot carry detail.

## 5. Logs in practice: discrete events with the detail

A log is where the detail lives. When the metric has told you *what* and the trace has told you *where*, the log tells you *why*, and "why" almost always means a specific value: the exception, the SQL, the decision, the inputs. The single most important upgrade you can make to your logs is to make them **structured** — emit them as key-value records (usually JSON) rather than free-form English strings — because a structured log can be filtered, aggregated, and correlated, while a free-text log can only be grepped. Here is the same event as a string and as structured data, so the difference is concrete.

```bash
ERROR checkout failed for user during db call after 1100ms
```

That string is fine for a human reading one line and useless for a machine reading a million. Now the structured version, the kind you actually want flowing into Loki or Elasticsearch:

```json
{
  "timestamp": "2026-06-20T14:06:51.221Z",
  "level": "error",
  "service": "checkout",
  "msg": "order persistence failed",
  "trace_id": "7f3a9c2e1b8d4a6f",
  "span_id": "a1b2c3d4e5f60718",
  "user_id": "u_88421",
  "order_id": "o_5512309",
  "db_query": "SELECT * FROM orders WHERE user_id = ? FOR UPDATE",
  "db_duration_ms": 1104,
  "db_rows_scanned": 4200000,
  "error": "context deadline exceeded"
}
```

Look at what that one record carries that no metric ever could: the exact user, the exact order, the exact query, the number of rows it scanned, and — crucially — the `trace_id`. The `db_rows_scanned` of 4.2 million against a `FOR UPDATE` is the smoking gun: a query that should hit one row is scanning the whole table, which means the index is gone. That is the *why*, and it lives in the log because the log is the only pillar with room for it. Notice also that this log line is not something you went looking for blindly; you arrived at it through the `trace_id`, having already narrowed from "the system is slow" to "this database span is slow." The log was the third click, not the first grep.

The cost model of logs is the opposite of metrics, and you must respect it or it will bankrupt your observability budget. A metric's cost is bounded by cardinality and independent of volume; a log's cost is dominated by *volume* — every event is a separate stored, indexed record. A service handling fifty thousand requests per second that logs one line per request produces 4.3 billion log lines per day. At even a few hundred bytes a line, that is on the order of a terabyte a day of raw logs from one service, and you pay to ingest it, to index it for search, and to retain it. This is why mature shops do three things ruthlessly. They **log at the right level** — `info` and above in production, `debug` only when toggled on for an investigation. They **sample high-volume logs** — keep all errors, but maybe one in a hundred success lines, because the hundredth identical "request succeeded" line teaches you nothing. And they **push detail out of logs and into traces** where it belongs, because a span attribute is cheaper than a log line for the same field and is structurally correlated for free. The sibling post [logging at scale without going broke](/blog/software-development/site-reliability-engineering/logging-at-scale-without-going-broke) is the deep dive on sampling, levels, retention tiers, and the economics of a log pipeline; here the point is narrower: a log answers *why* with maximum detail, and you pay for that detail by the event.

There is a second principle about logs that trips up teams who came from the print-debugging tradition: a log line should record a *fact about what happened*, not a *guess about what is wrong*. The most useful production logs are the ones that capture decisions and their inputs — "chose fallback path because primary returned 503," "rejected request because rate limit of 100/min exceeded, count was 142," "retried 3 times then gave up after 4.2s." Each of those is a fact with the data you need to understand the decision, and each is far more valuable during an incident than a vague "something went wrong here." When you instrument logging, write for the on-call engineer at 3am who has the trace id and needs the one missing piece of context the metric and the trace could not carry. That framing — log the decision and its inputs, at the boundary where a choice was made — produces logs that earn their storage cost. Logs that merely narrate control flow ("entering function," "got here," "about to call DB") are noise that costs money and teaches nothing; delete them or demote them to `debug`.

A quick note on a fourth thing that is technically a log but behaves like a metric: the **event-count log derivative**. Most log pipelines (Loki, the OpenTelemetry Collector, Fluent Bit) can match log lines and emit a *counter metric* from them — for example, count log lines where `level=error` and turn that into a `log_errors_total` metric. This is the bridge you use when the only signal for a failure mode lives in logs but you need to *alert* on it: you derive a low-cardinality metric from the high-volume logs and alert on the metric, keeping the log volume for forensics and the metric for detection. It is the same principle as everything else in this post — put the aggregate question on a metric and the detail question on a log — applied at the pipeline level. Just be careful not to label that derived metric with anything high-cardinality, or you have reintroduced the explosion through the back door.

## 6. Traces in practice: the causal path of one request

A trace is the pillar most teams adopt last and regret not adopting first, because it is the only one that gives you cross-service causality. In a monolith you can reason about a request by reading the code top to bottom. In a system of a dozen services, a single user action fans out into a tree of calls across process and network boundaries, and the question "why was *this* request slow?" has no answer you can read off any single service's logs or metrics. The trace is the answer. It assigns the request a trace id at the edge, and every service it touches creates a span — a timed unit of work — that carries that trace id and points to its parent span. Reassembled, the spans form a tree that shows exactly where the time went. The figure below is one such trace for our slow checkout.

![A graph diagram showing one checkout request fanning out from the gateway across auth, cart, pricing, cache, and the orders database, with the database span consuming 1.1 seconds of the 1.4 second total](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-6.png)

Here is what a single span looks like in the OpenTelemetry data model, serialized to JSON. A span has a name, a trace id shared with its siblings, a span id, a parent span id that builds the tree, a start and end time, and a bag of attributes — the high-cardinality detail that metrics cannot hold but traces happily can.

```json
{
  "name": "orders.persist",
  "trace_id": "7f3a9c2e1b8d4a6f",
  "span_id": "a1b2c3d4e5f60718",
  "parent_span_id": "0099aabbccddeeff",
  "start_time": "2026-06-20T14:06:50.117Z",
  "end_time": "2026-06-20T14:06:51.221Z",
  "kind": "CLIENT",
  "attributes": {
    "db.system": "postgresql",
    "db.statement": "SELECT * FROM orders WHERE user_id = ? FOR UPDATE",
    "db.rows_affected": 1,
    "net.peer.name": "orders-primary",
    "user_id": "u_88421",
    "order_id": "o_5512309"
  },
  "status": { "code": "ERROR", "message": "context deadline exceeded" }
}
```

That span took 1.104 seconds — the difference between `start_time` and `end_time` — and it is a child of the cart service's span, which is a child of the gateway's root span. When you view the trace, the database span is visibly the fat bar in the flame graph, and every other span is a sliver. *That* is the localization that nothing else provides. A metric on the gateway told you the whole request was 1.4 seconds; the trace tells you 1.1 of those 1.4 seconds were a single database call, and now you know precisely which service to blame and which log to read. Note also the `user_id` and `order_id` sitting happily in the span attributes — those are exactly the high-cardinality fields you must *never* put on a metric label, and they cost almost nothing here because a span is a sampled, per-request record, not an always-on aggregate.

Which brings us to the trace's trade-off: **sampling**. You cannot keep every trace at high volume for the same reason you cannot log every request — the cost scales with traffic. So you sample. There are two strategies and the choice matters. **Head-based sampling** decides at the start of the request, before you know anything, to keep or drop the trace — simple, cheap, and decided at the edge, but it throws away traces blindly, so a rare error might never be sampled. **Tail-based sampling** buffers the spans and decides *after* the request finishes, so you can keep one hundred percent of error traces and slow traces and sample only the boring fast successes — far better signal, at the cost of running a collector that holds spans in memory long enough to decide. The practical rule: tail-sample so you keep what is interesting (errors, high latency) and drop what is not, and accept that your traces are a curated sample, not a census. The sibling post [distributed tracing in practice](/blog/software-development/site-reliability-engineering/distributed-tracing-in-practice) covers context propagation across async boundaries, sampling strategies, and span design in depth. For now: a trace answers *where* with cross-service causality, and you pay for it with sampling.

It is worth being precise about what sampling does and does not cost you, because the word makes nervous engineers want to keep everything. Sampling does *not* hurt your aggregate accuracy when you also have metrics — your p99, your error rate, your traffic are all measured by the metrics pillar, which sees every request, so the numbers on your SLO dashboard are exact regardless of how few traces you keep. Sampling only affects which *individual* traces are available for forensics, and the whole point of tail-based sampling is to bias that retention toward the traces you would actually want to look at — the slow ones and the failed ones. The fast, successful, identical traces you drop are the ones you would never have opened anyway. So the honest framing is: metrics give you the exact aggregate, sampled traces give you a representative-plus-curated set of individual examples, and between them you have both the population view and a good supply of instances to inspect. You are not losing reliability measurement to sampling; you are losing only the boring traces, on purpose.

One more practical point on span design, because badly designed spans waste the trace's superpower. A span should correspond to a *unit of work worth timing on its own* — an outbound call to another service, a database query, a significant computation — not to every function call (which produces an unreadable thousand-span tree) and not to nothing (which produces a flat, useless trace). The right granularity is the granularity at which you would ask "did *this* take too long?" The fraud-scoring call in our war story was its own span, which is exactly why the trace could point a finger at it; if the entire downstream-call block had been one undifferentiated span, the trace would have said "downstream was slow" and left you back to guessing. Span boundaries are an instrumentation decision, and the decision is: draw a span around each thing whose latency you might one day need to attribute independently.

## 7. The decision table: which pillar, when

You now have all three pillars and their trade-offs, so let us compress everything into a table you can keep in your head. The decision is always driven by two things: the *question* you are asking, and the *cost axis* you can afford to push on. The figure and the table below are the same content — read whichever your brain prefers.

![A matrix decision table comparing metrics, logs, and traces across what each answers, what its cost scales with, and its cardinality limits](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-2.png)

| Signal | Answers | Best for | Cost scales with | Cardinality | Aggregation |
| --- | --- | --- | --- | --- | --- |
| **Metrics** | What and when | Dashboards, alerts, SLIs | Label combinations (cardinality) | Low — no user id | Native and cheap |
| **Logs** | Why | Root-cause detail, audit, the exception | Raw event volume | High — any field | Hard, post-hoc only |
| **Traces** | Where | Cross-service latency, the slow hop | Sampling rate kept | High — per request | Per-trace, not aggregate |

The trade-offs are not arbitrary; each one is the direct consequence of the question. Metrics must be cheap enough to evaluate alerts against continuously, so they give up detail and stay low-cardinality. Logs must carry full detail, so they give up cheapness and you pay per event. Traces must follow causality across a fleet, so they give up completeness and sample. Here is the same logic as a decision tree — start from the question and the branch lands you on the pillar.

![A decision tree starting from the question being asked and branching to metrics for trends and alerts, traces for cross-service latency, and logs for detailed root cause](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-8.png)

A few sharp rules fall out of this table that are worth stating as commandments. Never alert directly off logs when a metric can carry the same signal — derive a metric from the log if you must (most log pipelines can emit a counter for matching lines), and alert on the metric, because alerting needs the bounded cost and the aggregation. Never put a high-cardinality field — user id, request id, full URL with query params, email — on a metric label; that field belongs in a log or a span. Never expect a single trace to tell you whether something is *wrong*; a trace is a sample of one, so use a metric to detect and a trace to localize. And never reach for a trace when a metric will do — if the question is "is the error rate up?", that is a metric question, and pulling traces to answer it is slow and pointless. The pillar follows the question; the question is never "which tool do I like?"

#### Worked example: routing four real questions to the right pillar

Suppose four questions land in your on-call channel within an hour. "Did the error rate go up after the 2pm deploy?" is a *what-and-when* question — metric: query the error-ratio SLI before and after the deploy timestamp; done in one PromQL query, no logs needed. "The mobile app team says checkout is slow for some users — where is the time going?" is a *where* question — trace: pull tail-sampled slow traces for the checkout flow and read the flame graph; the fat span names the culprit service. "Why did order o_5512309 fail to persist?" is a *why* question about one specific event — log: filter by `order_id` and read the exception and inputs. "Are we within our latency SLO this month?" is a *what-over-time* question — metric: the p99 histogram aggregated over the 30-day window against the SLO threshold. Four questions, three pillars, zero ambiguity once you ask "what am I actually asking — what, where, or why?" The pillar is not a preference; it is determined by the question, and naming the question out loud is half the skill.

## 8. OpenTelemetry: one SDK, three signals, one trace id

The reason the metric-to-trace-to-log handoff is a click and not a search is **correlation**, and the reason correlation is now achievable without heroics is **OpenTelemetry** (OTel), the vendor-neutral CNCF standard for generating telemetry. Before OTel, you instrumented metrics with one library, traces with a vendor-specific tracer, and logs with a third thing, and stitching them together was a project. OTel collapses that: you instrument your code once with one SDK that emits all three signals, it propagates a trace id through the request context so every signal a request produces carries the same id, and a collector exports each signal to whatever backend you like — metrics to Prometheus, traces to Tempo or Jaeger, logs to Loki. The figure shows the shape.

![A vertical stack showing OpenTelemetry instrumenting a service once, emitting three signals through the SDK, propagating a trace id onto each signal, and exporting through a collector to Prometheus, Tempo, and Loki backends](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-7.png)

The mechanism that makes the whole workflow possible is **context propagation**. When a request enters your system, OTel creates a span and stores its trace id and span id in the request context. As your code calls downstream services, the SDK injects that context into the outgoing request headers (the W3C `traceparent` header), and the downstream service extracts it and continues the same trace. Meanwhile, any metric exemplar you record and any log line you emit *within* that context automatically picks up the active trace id. The result is the thing we keep coming back to: the spiking p99 bar on your dashboard carries an exemplar trace id, that trace id opens the exact trace, and that trace id filters the exact logs. Here is what wiring a span and a correlated log looks like in practice, in Python — small, idiomatic, and the kind of thing you actually write.

```python
from opentelemetry import trace
import logging, json

tracer = trace.get_tracer("checkout")
log = logging.getLogger("checkout")

def persist_order(order_id: str, user_id: str):
    # Open a span; everything inside shares this span's trace id.
    with tracer.start_as_current_span("orders.persist") as span:
        span.set_attribute("user_id", user_id)      # high-cardinality: fine on a span
        span.set_attribute("order_id", order_id)     # would be catastrophic as a metric label
        ctx = span.get_span_context()
        try:
            rows = run_query(order_id)               # the slow DB call
            span.set_attribute("db.rows_affected", rows)
        except DeadlineExceeded as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR)
            # Emit a structured log carrying the SAME trace id for correlation.
            log.error(json.dumps({
                "msg": "order persistence failed",
                "order_id": order_id,
                "user_id": user_id,
                "trace_id": format(ctx.trace_id, "032x"),
                "error": str(e),
            }))
            raise
```

The `trace_id` in that log line is the entire point — it is the thread that ties the log back to the span and the span back to the exemplar on the metric. Three signals, one id, one click between each. OpenTelemetry is the practical answer to "how do I get the correlated workflow without building three integrations by hand," and adopting it is one of the highest-leverage observability investments a team can make. For the architecture-level view of designing telemetry into a system from the start — span boundaries, what to instrument, how to budget the three signals at design time — cross-link out to the system-design treatment, [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design); this post is the operator's view of *using* the three signals once they exist.

Two more things about OpenTelemetry are worth knowing before you adopt it, because they shape how much of it you build by hand. First, **auto-instrumentation** does a surprising amount of the work for free. For many common languages and frameworks, an OTel agent or library will automatically create spans around incoming HTTP requests, outgoing HTTP calls, and database queries, and propagate context across them, without you writing any of the span code yourself. You get a usable trace of the request flow on day one, and you add *manual* spans (like the `orders.persist` example above) only for the in-process work the auto-instrumentation cannot see. This is why the cost of adopting tracing has fallen so far in recent years: you no longer hand-instrument every service from scratch. Second, the **Collector** is the piece that earns its keep operationally. Rather than have every service talk to every backend, you run the OpenTelemetry Collector as a pipeline that receives all three signals, processes them (this is where tail-based sampling lives, where you scrub sensitive fields, where you batch and retry), and exports them to the right backends. The Collector is where your sampling policy, your cost controls, and your routing live, decoupled from application code, so you can change the sampling rate or add a backend without redeploying a single service. When you design an observability rollout, design the Collector deliberately — it is the control point for the entire cost model from the previous section.

A practical adoption sequence falls out of all this. Start by turning on auto-instrumentation so you get baseline traces and the standard HTTP and database metrics with near-zero effort. Add a Collector and route metrics to Prometheus, traces to Tempo or Jaeger, logs to Loki, with a tail-sampling policy that keeps errors and slow traces. Then, incrementally, add manual spans around the in-process work you most often need to attribute, and add `trace_id` to your structured logs so the correlation closes the loop. You do not boil the ocean; you get the metric-trace-log chain working end to end on your single most important service first, prove the workflow shortens an incident, and expand from there. That incremental path is how observability adoption succeeds in practice instead of stalling as a six-month platform project nobody finishes.

## 9. The unified cost model: three knobs on one bill

Step back and look at the three pillars as a single budget, because the most senior way to think about observability is as one bill with three knobs, each pillar's cost driven by a different lever. Get the levers straight and you can reason about your whole observability spend without surprises.

Metrics cost scales with **cardinality** — the number of distinct time series, which is the product of label-value counts. The knob is your label design. Turn it carelessly (add `user_id`) and the bill explodes by orders of magnitude; turn it deliberately (golden signals, low-cardinality labels) and it stays flat regardless of traffic. Logs cost scales with **volume** — the number of events ingested, indexed, and retained. The knobs are log level, sampling rate, and retention period. Turn them carelessly (debug-in-prod, log-per-request, retain-everything-forever) and you pay terabyte prices; turn them deliberately (info-and-above, sample successes, tier retention) and you keep the forensic detail at a fraction of the cost. Traces cost scales with **sampling rate** — the fraction of traces you keep. The knob is the sampling policy. Turn it carelessly (keep 100%) and you pay log-like volume; turn it deliberately (tail-sample errors and slow traces, drop boring successes) and you keep the debugging power at a sliver of the cost.

Notice the pattern: in all three pillars, the careless setting pays *volume* prices and the deliberate setting pays *bounded* prices, and the deliberate setting loses you almost nothing that you would actually have queried. That is the central insight of observability cost management. You are never choosing between "expensive and complete" and "cheap and useless"; you are choosing between "expensive and mostly redundant" and "cheap and almost-as-useful," because the data you drop under good policies — the millionth identical success log, the boring fast trace, the high-cardinality label that belongs elsewhere — is data you would never have looked at. The cost lever is always the same move applied to a different knob: *keep the aggregate cheaply, keep the interesting instances deliberately, drop the redundant rest.*

This is also why the three pillars resist being collapsed into one product even though several vendors sell "one platform." The platform can store all three for you; it cannot change the fact that each signal's cost is driven by a different physical lever, and that you must tune each lever separately. A team that treats observability as one undifferentiated firehose — "just send everything to the platform" — gets one undifferentiated enormous bill. A team that understands the three knobs tunes cardinality on metrics, volume on logs, and sampling on traces independently, and lands a bill that is a fraction of the firehose for the same — often better — debugging capability. The single most leveraged cardinality knob is the one we turn next, because it is the one that goes wrong most violently.

## 10. The cardinality explosion: the single most expensive mistake

Everything above has hinted at cardinality. Now we make it rigorous, because the cardinality explosion is the most common way a well-meaning engineer accidentally multiplies the observability bill by four orders of magnitude with a one-line change, and understanding it is the difference between an observability stack you can afford and one you cannot.

**Cardinality** is the number of distinct time series a metric produces, and it is the *product* of the number of distinct values of every label. A metric `http_requests_total` with a `route` label that has 20 values and a `code` label with 8 values produces $20 \times 8 = 160$ series. Add a `method` label with 4 values and you are at $20 \times 8 \times 4 = 640$. The cost is multiplicative, not additive, because each combination of label values is a separate series that Prometheus stores, indexes, and keeps in memory. So far so manageable — 640 series is nothing. The explosion happens the instant someone adds a label whose value space is *unbounded by request*, and the canonical example is `user_id`.

![A before and after diagram showing that adding user_id as a metric label turns ten endpoint series into ten million series and melts the time series database, while putting user_id in a log or trace keeps the metrics bill stable and still queryable](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-5.png)

Watch the arithmetic. Suppose you have 10 endpoints and you are tracking a request counter — 10 series, trivially cheap. A product manager asks, reasonably, "can we see request counts per user?" and an engineer adds `user_id` as a label. You have one million users. The cardinality is now $10 \times 1{,}000{,}000 = 10{,}000{,}000$ series. Ten million. Each series carries its own samples, its own index entry, its own slice of memory; Prometheus is built to handle millions of series but ten million from one careless label, multiplied by the dozens of other metrics that label might leak into, will exhaust memory and either OOM-kill the server or send your managed-metrics bill — which is almost always priced per series or per "active time series" — into numbers that get noticed by finance. And it gets worse, because cardinality is *sticky*: even after you remove the bad label, the series you already created persist in storage for the full retention window, so a five-minute mistake costs you weeks of bloat.

The fix is not "be careful with labels," though you should be. The fix is structural and it is the lesson of this entire post: **high-cardinality data belongs in logs and traces, not in metrics.** You want per-user analysis? Put `user_id` on the *span* and on the *log line*, where it costs you one attribute on a sampled, per-request record instead of multiplying your always-on series count. Then you query it where high cardinality is cheap — you filter traces by user, or you aggregate logs by user — and your metrics stay a tidy 10 series that you can still alert on safely. This is *the* practical reason the three pillars exist as separate things: the cost model of each is what makes it safe to carry a certain kind of data. Metrics carry low-cardinality aggregates cheaply; logs and traces carry high-cardinality detail at a cost that is bounded by volume and sampling rather than exploded by label combinations. Put the data where its cost model fits and the bill stays sane.

#### Worked example: the label that cost the budget

A team adds three labels to their main request metric to "improve observability": `user_id` (1,000,000 values), `session_id` (5,000,000 active in the window), and `request_id` (unique per request — effectively unbounded). The base metric had a `route` label with 25 values. Before the change: 25 series. After: the `request_id` label alone makes every single request its own series, so the active series count over a 5-minute window equals the number of *distinct requests in that window*. At 50,000 requests per second over 5 minutes that is 15 million distinct requests, hence on the order of 15 million active series — from one metric. The managed-metrics vendor, billing roughly per active series, sends an invoice an order of magnitude larger than the entire previous monthly bill. The correct version keeps the metric at 25 series with only the `route` label, and moves `user_id`, `session_id`, and `request_id` into span attributes and structured log fields, where they cost a fixed amount per sampled request and are exactly where you would go to investigate a specific user or session anyway. Same questions answerable; bill reduced by roughly 99.99%. The lesson is a rule you can apply mechanically: *if a label's value space grows with traffic or users, it does not go on a metric.*

## 11. Monitoring versus observability: known versus unknown failures

There is a distinction underneath all of this that is worth making explicit, because it reframes *why* you collect three pillars rather than just metrics. **Monitoring** is watching the failure modes you predicted: you decide in advance what might go wrong, you build a dashboard and an alert for each, and you watch them. **Observability** is the property of a system that lets you ask *new* questions about failure modes you did *not* predict, from the telemetry you already have, without shipping new code. The figure contrasts them.

![A before and after diagram contrasting monitoring on fixed dashboards for predicted failures against observability with high-cardinality telemetry that answers new questions without a deploy](/imgs/blogs/metrics-logs-and-traces-when-to-use-which-3.png)

The distinction is not academic; it determines whether your next incident is a five-minute investigation or a five-hour one. Monitoring handles the *known unknowns* — the disk that might fill, the dependency that might time out, the error rate that might climb. You predicted these, so you have a dashboard and an alert, and when one fires you know what it means. But production systems fail in ways nobody predicted: a specific combination of a new device type, a particular API version, and a feature flag interacts to produce timeouts for exactly the users in one region on one app version. No dashboard exists for that, because no one predicted it. With *monitoring only*, your move is to form a hypothesis, write new instrumentation to test it, deploy, wait, and repeat — hours per loop. With *observability*, you already have high-cardinality telemetry (rich span attributes, structured logs) and you can slice it live: group failing requests by device, by API version, by flag, by region, until the pattern jumps out — minutes, no deploy. The thing that *buys* observability is exactly the high-cardinality detail in logs and traces that you must keep *out* of metrics. That is the deep reason the pillars are not redundant: metrics give you cheap monitoring of the known, while logs and traces give you explorable observability of the unknown.

Put plainly: monitoring tells you the system is broken; observability tells you why, *even when you never predicted that particular break.* A mature stack does both. It alerts on a small number of symptom-based metric SLIs (monitoring) so you learn *fast* that something is wrong, and it carries high-cardinality traces and logs (observability) so you can answer *new* questions when the break is novel. If you only invest in metrics, you can detect every failure you predicted and debug none of the ones you did not. If you only invest in logs and traces, you can debug anything but you will not find out it is broken until a user tells you. You need both halves, which is to say you need all three pillars.

## 12. Stress-testing the workflow: when the clean story breaks

The metric-to-trace-to-log workflow is clean in the worked example. Real incidents are messier, so let us stress-test it against the cases that actually happen on call, because knowing where the workflow strains is what separates someone who has read about observability from someone who has used it under fire.

**What if the trace was not sampled?** You followed the metric to the right minute, but when you go to pull an exemplar trace, the slow request was dropped by head-based sampling and you have nothing. This is the strongest argument for *tail-based* sampling: configure the collector to keep 100% of error traces and traces above a latency threshold, so the interesting requests are never the ones thrown away. If you are stuck with head sampling and a missing trace, fall back: the metric still localized you in *time*, and the structured logs for that window — filtered to errors and that service — will still carry the *why*, just without the cross-service *where*. The workflow degrades gracefully because the pillars overlap at the edges.

**What if the metric never fired?** The latency spiked only for a thin slice of users — one region, one app version — and the global p99 barely moved, so no alert. This is the monitoring blind spot from section 9, and the answer is observability: because your traces and logs carry `region` and `app_version` as high-cardinality attributes, you can slice latency by those dimensions *after* a customer reports it and find the spike that the aggregate metric hid. It also argues for SLIs sliced by the dimensions that matter (per-region availability), so the metric *does* fire next time — but you cannot pre-slice by every dimension, which is exactly why you keep the high-cardinality signals.

**What if two incidents overlap?** Two unrelated problems are burning at once and the logs are a blur of both. This is where the `trace_id` earns its keep: you do not read logs by time window, you read them by trace, so the spans and logs of incident A share one trace id and incident B's share another, and the correlation untangles what a time-ordered grep would have hopelessly interleaved. Correlation is not a nicety; under concurrent incidents it is the only thing that keeps the investigation tractable.

**What if the bill is the incident?** Sometimes the failure *is* the observability stack — a cardinality explosion from a bad deploy is hammering the metrics backend, or a logging loop is flooding the pipeline and you cannot even see straight. Treat your observability stack as a production system with its own SLIs: alert on ingestion rate, on active series count, on dropped spans. A sudden 10x in active series is itself a symptom worth paging on, because it usually means someone just shipped the `user_id`-as-a-label mistake, and catching it in minutes instead of at month-end invoice time is the difference between a quick rollback and a budget conversation with finance.

## 13. War story: the grep marathon and the click

Let me tell you the story this whole post is really about, in the composite, illustrative form it takes across many teams (the specifics here are representative rather than any single named company's documented incident). A payments team had excellent metrics and excellent logs and no tracing. Their dashboards were beautiful; their alerting was tight; their logs were structured and searchable. They were, by the standards of five years earlier, a model observability shop. And every cross-service latency incident took them roughly forty-five minutes to diagnose, every single time, because they had two of the three pillars and the missing one was the *where*.

The pattern was always the same. A metric alert fired — checkout p99 over threshold, good, the monitoring worked. The on-call would open the dashboards and confirm the spike was real and global. Then they would start *guessing*. Was it the auth service? Pull auth's logs, grep, nothing obviously wrong. The pricing service? Pull its logs, grep, looks fine. The inventory service? The database? Each guess was a few minutes of pulling and grepping a different service's logs, building a mental model of the call graph by hand, because nothing tied the request together across services. They were using logs — the *why* pillar — to answer a *where* question, one service at a time, and it was slow because that is not what logs are for. Forty-five minutes, every incident, burning error budget the whole time.

They adopted OpenTelemetry tracing. The next cross-service latency incident, the on-call followed the metric to the minute, clicked an exemplar trace straight off the spiking histogram bar, saw a flame graph where one downstream span — a call to a third-party fraud-scoring API — was 900 milliseconds of a 1.1-second request while every internal hop was a sliver, and had the answer in under five minutes: the external dependency was slow, not their code, and the fix was a tighter timeout and a fallback. Forty-five minutes to five. Same metrics, same logs; the only thing that changed was adding the pillar that answers *where*, and letting the three signals hand off to each other through a shared trace id instead of being investigated in isolation. The measured result, averaged over the following quarter, was a roughly 70% reduction in mean time to identify the responsible service for cross-service incidents. That is the entire thesis of this post made operational: you need all three, and you need them correlated, because each one is bad at the others' jobs and the handoff between them is where the speed comes from.

## 14. How to reach for this (and when not to)

Decisive recommendations, because every pillar costs money and attention and you should not over-invest in any of them. Here is how to allocate.

**Always have metrics, and make them your SLIs and your alerts.** This is non-negotiable for any service that matters. Metrics are cheap, they are the only signal you can afford to alert on continuously, and they are the substrate of your error budget. Start here, keep the cardinality low and deliberate, and alert on a small number of symptom-based SLIs. If you do one thing, do this.

**Have structured logs, and spend money to keep them affordable.** Logs are where root cause lives, so you need them — but they are the pillar most likely to bankrupt you, so invest in the controls: structured JSON, sane log levels in production, sampling of high-volume success lines, retention tiers (hot for a week, cheap cold storage for compliance), and a relentless habit of pushing high-cardinality detail into span attributes instead of fattening every log line. Do not log a line per request at fifty thousand requests per second and expect the bill to be reasonable.

**Add tracing once you are more than two or three services deep, and not really before.** This is the honest "when not to" of tracing. For a monolith or two services, tracing's marginal value is low — you can reason about the call path by reading the code, and metrics plus logs cover you. The value of a trace is *cross-service causality*, and that value scales with the number of service hops. Once a single user action fans out across five, eight, a dozen services, tracing stops being a nice-to-have and becomes the only thing that can answer "where did the time go" in finite time. Adopt it then. Adopt OpenTelemetry specifically, so you instrument once and get correlation for free, rather than bolting on a vendor tracer later.

**Do not put high-cardinality data on a metric, ever.** This is the one rule that has no exceptions worth the risk. User id, request id, session id, full URL, email, raw error message — none of these go on a metric label. They go on a span and in a log. Violating this is how you turn 10 series into 10 million, and it is the single most expensive observability mistake there is.

**Do not start an investigation with logs.** Start with the metric (what and when), narrow with the trace (where), finish with the log (why). Reaching for the grep first is the reflex of someone who has not internalized the workflow, and it is slow because it uses the most expensive, least aggregable pillar as a search tool. Train the order until it is muscle memory, because at 3am muscle memory is all you have.

And one more, on over-investment: do not build a tracing pipeline for an internal cron job that runs once an hour, do not keep 100% of traces "just in case" when tail sampling the interesting ones gives you the same debugging power at a fraction of the cost, and do not retain raw logs at full fidelity for a year when a metric and a sampled trace would answer every question you will actually ask. The three pillars are tools with costs; match the investment to the value, and the value is set by how much user pain the service can cause and how many hops a request crosses. For the broader question of debugging production with these signals in hand, the debugging-series post [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) picks up where this one leaves off.

## Key takeaways

- **Three pillars, three questions.** Metrics answer *what and when*, traces answer *where*, logs answer *why*. No single pillar is enough; you carry all three because each is bad at the others' jobs.
- **The workflow has a fixed order: metric, then trace, then log.** Detect with the metric, localize with the trace, explain with the log. Starting with logs is the classic slow mistake.
- **Each pillar's cost model is the consequence of its question.** Metrics are bounded-cost and low-cardinality so they can be alerted on continuously; logs cost per event so they carry detail; traces are sampled so they can follow causality across a fleet.
- **Cardinality is multiplicative and the explosion is the most expensive mistake.** Adding `user_id` to a metric can turn 10 series into 10 million. High-cardinality data belongs in logs and traces, never on a metric label.
- **Correlation is what makes the handoff a click, not a search.** A shared `trace_id` across metric exemplars, spans, and log lines is the thread that ties the three pillars together; without it you have three witnesses who do not talk.
- **Monitoring watches predicted failures; observability answers new questions about unpredicted ones.** Metrics give you monitoring of the known; high-cardinality logs and traces give you observability of the unknown. You need both.
- **OpenTelemetry is the practical unifier.** One SDK, three signals, context propagation that stamps a trace id on everything — adopt it once you are several services deep and you get the correlated workflow without building three integrations.
- **Match investment to value.** Always have metrics; keep logs affordable with structure, levels, and sampling; add tracing when you cross enough service hops that cross-service causality is the bottleneck. Do not over-collect what you will never query.

## Further reading

- [Reliability Is a Feature: The SRE Mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the intro map for this whole field manual and the define-measure-budget-respond-learn-engineer loop this post lives inside.
- [Metrics and Time Series Done Right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right) — the deep dive on the metrics pillar: naming, RED and USE, histograms, recording rules, and PromQL at scale.
- [Logging at Scale Without Going Broke](/blog/software-development/site-reliability-engineering/logging-at-scale-without-going-broke) — the economics of logs: structured logging, levels, sampling, retention tiers, and the pipeline.
- [Distributed Tracing in Practice](/blog/software-development/site-reliability-engineering/distributed-tracing-in-practice) — context propagation, head versus tail sampling, span design, and reading a flame graph.
- [Observability: Metrics, Logs, and Traces by Design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — the architecture-time view of designing telemetry into a system, complementing this operator-time view.
- [Observability for Debugging Prod](/blog/software-development/debugging/observability-for-debugging-prod) — using the three signals to run a real production debugging session.
- The Google SRE Book and SRE Workbook (Chapters on monitoring distributed systems and on SLIs) — the canonical source for symptom-based monitoring and the golden signals.
- The OpenTelemetry documentation and the Prometheus documentation (querying, histograms, recording and alerting rules) — the authoritative references for the tools used throughout this post.

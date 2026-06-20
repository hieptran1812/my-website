---
title: "Metrics and Time Series Done Right: From the Data Model to PromQL That Tells the Truth"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn the Prometheus data model, the four metric types, the RED and USE menus, the PromQL that turns counters and histograms into honest SLIs, and how to avoid the cardinality explosion that takes your monitoring down with your service."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "prometheus",
    "promql",
    "metrics",
    "observability",
    "histogram",
    "cardinality",
    "red-method",
    "use-method",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/metrics-and-time-series-done-right-1.png"
---

At 3:47 on a Tuesday morning the pager went off for a service that, by every dashboard we owned, looked perfectly healthy. CPU was at 40%, memory was fine, the pods were all `Running`. And yet the API was returning errors to a third of our users. The dashboards were green because we were measuring the wrong things — utilization of the box instead of the experience of the user — and worse, when I tried to query the one panel that might have told me the truth, the latency panel, Prometheus timed out. Someone had shipped a change two days earlier that added a `user_id` label to a request counter. Our series count had gone from about fifty thousand to twelve million overnight, the Prometheus server was thrashing on garbage collection, and the very tool we needed to diagnose the outage was itself part of the outage.

That night taught me, expensively, that metrics are not free and not automatic. A metric is a model of your system, and a bad model lies to you confidently. You can have terabytes of data and still not be able to answer "are users in pain right now?" You can add a label that feels harmless and quietly multiply your storage cost by a hundred. You can read a p99 latency off a dashboard that is mathematically meaningless because your histogram buckets are in the wrong place. Getting metrics right is not about collecting more — it is about collecting the right shape of data, naming it well, querying it honestly, and bounding it so it can never take you down.

This post is the second stop on the observability leg of our reliability loop: we **define** reliability with SLIs and SLOs, then we have to **measure** it, and metrics are the workhorse of that measurement. By the end you will be able to read the Prometheus data model in your sleep, pick the right one of the four metric types for any quantity, instrument an HTTP service with the RED method, write the PromQL that turns a raw counter into a rate and a raw histogram into a believable p99, spot a cardinality bomb before it goes off, and precompute your expensive SLI queries with recording rules so your dashboards and burn-rate alerts read in milliseconds. Figure 1 is the mental model the whole post hangs on — the data model that says a time series is nothing more than a metric name, a set of labels, and a stream of samples.

![A layered stack showing that a Prometheus series is a metric name plus a label set forming one unique series of timestamped samples, with cardinality counting unique series and memory cost scaling with it](/imgs/blogs/metrics-and-time-series-done-right-1.png)

If you have not read it yet, this post sits downstream of the series intro, [reliability is a feature, the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), and it is the partner to [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain): that post tells you *what* to measure, this one tells you *how* to measure it correctly. We will reference siblings on when to use logs and traces instead, on building dashboards that tell the truth, and on alerting that does not cry wolf, all of which build on the metrics foundation laid here.

## 1. The data model: a time series is a name plus labels

Everything in Prometheus — and in the broader OpenMetrics world that Prometheus standardized — reduces to one idea. A **time series** is identified by a **metric name** and a set of **labels** (key-value pairs), and it holds a stream of **samples**, where each sample is a timestamp and a floating-point value. That is the whole data model. It is worth internalizing because almost every metrics mistake is a violation of this model.

Concretely, a single series looks like this in the exposition format a target serves on its `/metrics` endpoint:

```
http_requests_total{route="/checkout", method="POST", code="200"} 84213
```

Read that left to right. `http_requests_total` is the metric name. The part in braces is the **label set**: `route`, `method`, and `code`, each with a value. The number at the end is the current sample value. The full identity of the series is the name *plus every label* — change any label value and you have a different series. So `http_requests_total{route="/checkout", method="POST", code="200"}` and `http_requests_total{route="/checkout", method="POST", code="500"}` are two completely separate series that happen to share a name.

This is the single most important consequence of the model, so let me state it plainly: the number of distinct series your system produces equals the number of distinct label-set combinations across all your metrics. We call that number **cardinality**, and it is the resource that governs almost everything about how Prometheus behaves — its memory footprint, its query speed, its disk usage, and its cost. Prometheus holds the most recent sample of every active series in memory in a structure called the head block, and it builds an inverted index over every label name and value so it can answer queries like "all series where `route=/checkout`." Both of those grow with cardinality. A few hundred thousand series is comfortable on a modest server. A few million starts to hurt. Ten or twenty million and you are in the territory of the outage I opened with.

Samples are appended on a schedule. Prometheus **pulls** — it reaches out to each target on a fixed **scrape interval** (commonly 15 or 30 seconds), does an HTTP `GET` on `/metrics`, parses every line, and appends one sample per series with the scrape timestamp. There is no per-sample timestamp coming from the application in the common case; the timestamp is when Prometheus scraped. This matters because it sets the resolution of everything downstream: if you scrape every 15 seconds, you cannot see anything that happens and resolves inside a 15-second window, and any rate you compute needs several scrapes to be meaningful.

### Why the model is shaped this way

The label-set design is what makes metrics cheap to aggregate and slice. Because every dimension you care about — which route, which status code, which method, which datacenter — is a label rather than baked into the metric name, you can sum across some labels and group by others at query time. You instrument once and ask many questions. Compare that to the bad old days of `graphite`-style dotted names like `prod.checkout.post.200.count`, where the dimensions were positional in the name and you could only slice the way you had pre-encoded. Labels give you a relational model in disguise: name is the table, labels are the columns, and PromQL is the query language.

The cost of that flexibility is exactly the thing that bites people: every unique combination of label values is a row, and rows are not free. We will spend a whole section on the discipline this demands, but the seed of it is right here in the data model. A label is a dimension; the number of series is the *product* of the dimensions' cardinalities; and a product blows up fast. Two labels with 40 values each give you 1,600 series; add a third with 100,000 values and you have 160 million. The model is beautiful and it is a loaded gun.

There is one more piece of the model that confuses newcomers: there are also a handful of **reserved labels** that Prometheus adds for you. The `__name__` label *is* the metric name — `http_requests_total{...}` is sugar for `{__name__="http_requests_total", ...}`, which is why you can match metric names with regexes like `{__name__=~"http_.*"}`. The `job` and `instance` labels come from the scrape config: `job` is the logical group of targets (all the `api` pods), and `instance` is the specific target (`10.0.3.7:8080`). These are added at scrape time, not by your code, and they are how the same application metric from twenty different pods stays distinguishable — twenty series differing only in `instance`, which `sum without (instance)` then collapses into a fleet number. Understanding that the metric name is itself just a label, and that `job`/`instance` are attached by the scraper, demystifies a lot of PromQL that otherwise looks like magic.

## 2. The four metric types: counter, gauge, histogram, summary

Prometheus has exactly four metric types, and choosing the right one is the first place instrumentation goes right or wrong. The type is not stored on the server — on the wire it is just series and samples — but it is a contract about what the numbers *mean* and how you are allowed to query them. Figure 2 lays out all four against the questions they answer and, critically, whether you can aggregate them across instances.

![A matrix comparing counter gauge histogram and summary across their behavior, the function used to read them, and whether they can be aggregated across instances, showing summary cannot be averaged](/imgs/blogs/metrics-and-time-series-done-right-2.png)

**Counter.** A counter only ever goes up (or resets to zero when the process restarts). It counts events: requests served, errors returned, bytes sent, tasks completed. The raw value of a counter is almost never interesting — nobody cares that the process has served 84,213,902 requests since it started. What you care about is the *rate*: requests per second right now. So you almost never read a counter directly; you wrap it in `rate()`. By convention a counter's name ends in `_total`: `http_requests_total`, `errors_total`, `bytes_sent_total`. Counters aggregate beautifully: the total request rate across ten instances is just the sum of the ten per-instance rates.

**Gauge.** A gauge goes up and down and represents a current value: temperature, queue depth, memory in use, number of in-flight requests, current number of connected clients. You read a gauge raw, or you take its `avg`, `max`, `min`, or `sum` across instances depending on what makes sense. `rate()` on a gauge is meaningless — there is no monotonic accumulation to differentiate. Gauges are the right tool for "how full is this thing right now."

**Histogram.** A histogram samples observations — almost always durations or sizes — and counts them into a set of cumulative **buckets**. When you observe a request that took 0.27 seconds, the histogram increments every bucket whose upper bound (`le`, for "less than or equal") is at least 0.27, plus a running `_sum` of all observed values and a `_count` of all observations. So a single histogram metric `http_request_duration_seconds` actually exposes many series: one `_bucket` series per `le` boundary, plus `_sum` and `_count`. The payoff is that you can compute approximate quantiles — p50, p90, p99 — *after the fact, on the server, aggregated across all instances*, using `histogram_quantile()`. This is the single most important metric type for SRE latency work, and we will give it its own section because there is a sharp caveat in how it computes those quantiles.

**Summary.** A summary also tracks a distribution, but it computes the quantiles **on the client**, inside the application, at observation time. It exposes series like `http_request_duration_seconds{quantile="0.99"}` with the p99 already calculated. This sounds convenient and it is a trap for any multi-instance service. Quantiles cannot be averaged: the average of three instances' p99 latencies is *not* the fleet p99. If instance A's p99 is 100ms, B's is 100ms, and C's p99 is 2000ms because C is sick, the true fleet p99 is dominated by C's tail, but `avg` gives you 733ms, which is a number that describes no real request. Because the quantile is baked in on the client, you cannot recombine. Summaries are fine for a single-instance process or for tracking a fixed quantile you will never aggregate; for anything that runs on more than one box, reach for a histogram.

| Type | Goes | Read with | Aggregate across instances | Name suffix |
| --- | --- | --- | --- | --- |
| Counter | Up only, resets to 0 | `rate()` / `increase()` | Yes — sum the rates | `_total` |
| Gauge | Up and down | raw value, `avg`/`max`/`sum by` | Yes — sum or average | none / unit |
| Histogram | Observations into buckets | `histogram_quantile()` on `_bucket` | Yes — sum buckets, then quantile | `_seconds`/`_bytes` + `_bucket` |
| Summary | Observations, client quantiles | read the `quantile` label | **No** — cannot recombine quantiles | `_seconds` + `quantile` label |

The decision rule is mechanical and worth memorizing: **counting events that only go up → counter; a level that moves both ways → gauge; a distribution you want quantiles from → histogram; a distribution on a single instance where you will never aggregate → summary (and even then, prefer a histogram).** Get this choice right and the rest of your PromQL falls into place; get it wrong and you will fight the query language forever trying to extract a number the type cannot give you.

A common confusion worth heading off: people reach for a gauge to track "requests per second" and set it from the application. Do not. A self-reported rate gauge bakes the averaging window into the application, loses the ability to re-window at query time, and breaks on restart in ways a counter does not. The right pattern is *always* a counter that the application only ever increments, with the rate computed at query time by `rate()`. The application's job is to count truthfully; the query's job is to turn counts into rates over whatever window the question needs. This separation — applications count, queries derive — is the single most important instrumentation habit, and it is why counters vastly outnumber gauges in a well-instrumented service. Reserve gauges for genuine levels: a queue depth, a pool's in-use count, a cache size, a temperature, a "number currently connected." If the quantity is fundamentally "how many times has X happened," it is a counter, full stop.

The histogram-versus-summary decision deserves one more concrete framing because it is the choice people most often get wrong. Ask: will this metric ever be aggregated across more than one process? If your service runs on more than one pod — and in production it almost always does — the answer is yes, and a summary's client-side quantiles become useless the moment you have two instances, because there is no correct way to combine `instance A's p99` with `instance B's p99` into a fleet p99. The histogram defers the quantile computation to query time, *after* the buckets from all instances have been summed, which is the only place the fleet-wide quantile can be computed correctly. Summaries earn their place in exactly one situation: a single-instance process (a leader, a singleton job) where you want a quantile cheaply without choosing buckets, and you will never scale it out. Everywhere else, histogram.

## 3. PromQL essentials: rate, irate, and the rate-interval

PromQL is the query language, and for SRE work you need a surprisingly small core of it done correctly. The first thing to understand is the difference between an **instant vector** (one value per series at a single point in time) and a **range vector** (a window of samples per series, written with a duration in square brackets like `[5m]`). The rate functions take a range vector and collapse it back to an instant vector.

**`rate()`** computes the per-second average rate of increase of a counter over the range you give it. `rate(http_requests_total[5m])` reads every sample of that counter in the last five minutes and returns the average requests-per-second over that window. It is smart about counter resets — if the process restarted and the counter dropped from 84,000 to 200, `rate()` recognizes the reset and does not report a giant negative spike. It also extrapolates slightly to the edges of the window. The window length is a smoothing knob: a longer window (`[5m]`) is smoother and lags more; a shorter window (`[1m]`) is twitchier and more responsive but noisier.

**`irate()`** is the "instant rate" — it uses only the last two samples in the range to compute the most recent rate of change. It reacts fast and is great for graphing volatile, fast-moving signals where you want to see spikes, but it is too jumpy for alerting because a single scrape gap or burst can swing it wildly. The rule of thumb: `rate()` for alerts and SLOs (you want stability), `irate()` only for high-resolution dashboards of fast signals, and never `irate()` in an alert expression.

There is a subtle constraint that trips up everyone. For `rate()` to work it needs at least **two samples** inside the window, and to be stable it really wants four or so. If your scrape interval is 30 seconds and you write `rate(...[30s])`, you might catch zero or one sample and get gaps or `NaN`. The window must comfortably exceed the scrape interval — a common floor is four times the scrape interval. This is where **`$__rate_interval`** comes in. It is a Grafana template variable that automatically computes a rate window appropriate to the current scrape interval and the graph's pixel resolution, guaranteeing you always have enough samples. In a Grafana panel you write `rate(http_requests_total[$__rate_interval])` and stop worrying about it. Outside Grafana, pick a fixed window that is at least 4× your scrape interval — for a 15-second scrape, `[1m]` is the practical floor and `[5m]` is the comfortable default for alerting.

### Aggregation: sum and avg by

Raw `rate()` gives you one series per label combination. To get a service-level number you aggregate. The two operators you will use constantly are `sum by (...)` and `avg by (...)`. To get the total request rate of a service across all instances and routes:

```promql
sum(rate(http_requests_total[5m]))
```

To keep it broken out by status code so you can see errors separately:

```promql
sum by (code) (rate(http_requests_total[5m]))
```

The golden rule of aggregating rates is **rate first, then sum** — `sum(rate(...))`, never `rate(sum(...))`. The reason is counter resets: if you sum raw counters across instances first and one instance restarts, the summed counter takes a big backward step, and `rate()` applied afterward sees a reset that was not real and mis-handles it. Computing the rate per series first lets each series' reset be handled correctly, and then you sum well-behaved per-second rates. Burn this into your fingers: `sum(rate(...))`.

### increase, offset, and comparing windows

Two more functions earn their keep in SRE work. **`increase()`** is `rate()` multiplied by the window length — it gives you the *total count* of events over a window rather than a per-second rate, which is what you want when you say "how many errors in the last hour." `increase(http_requests_total{code=~"5.."}[1h])` is the absolute error count this hour, the natural unit for an error-budget conversation ("we have spent 4,200 of our 8,640 allowed errors this month"). Under the hood it is the same counter-reset-aware computation as `rate()`, just not divided by seconds.

The **`offset`** modifier shifts a query back in time, which is how you compare *now* against *then*. `sum(rate(http_requests_total[5m] offset 1w))` is last week's request rate at this same moment, and dividing the current rate by it gives you a week-over-week traffic ratio — invaluable for spotting a traffic drop (a partial outage upstream) or a spike (a thundering herd) against the normal weekly pattern. You will reach for `offset` constantly when you want "is this number abnormal?" rather than "what is this number?"

Finally, a word on **label matching** in selectors, because it is where queries quietly select the wrong series. `code="500"` is an exact match; `code=~"5.."` is a regex match (any 5xx); `code!~"2.."` excludes the 2xx successes. The regex matchers are how you carve out errors without enumerating every status code, and the negative matchers (`!=`, `!~`) are how you say "everything except." A subtle gotcha: a regex like `code=~"5.."` matches the *whole* label value anchored, so `5.. ` means exactly three characters starting with 5 — fine for HTTP codes, but be deliberate when matching free-form labels. Getting the selector right is half of writing honest PromQL; a query that silently drops the `code=~"5.."` filter reports your *total* rate as your *error* rate and makes a healthy service look like it is on fire.

### The vector-matching trap in ratios

When you write the availability SLI as `errors / total`, PromQL has to match the left-hand series to the right-hand series by their labels — this is **vector matching**, and it is the second most common source of a wrong PromQL answer after forgetting `by (le)`. If both sides are aggregated to the same label set with `sum by (job)`, they match cleanly job-to-job and you get one ratio per job. But if the two sides carry different labels — say the error rate still has a `code` label and the total does not — the division finds no matching pairs and returns *nothing*, an empty result that looks exactly like "zero errors." The discipline is to aggregate both sides of a ratio to the *same* label set before dividing, which is exactly what the recording rules in section 8 do. When a ratio query returns empty and you cannot see why, check that both sides expose identical labels; nine times out of ten that is the bug.

## 4. Histograms and the bucket-boundary caveat

The histogram is where SRE latency measurement lives, and `histogram_quantile()` is where most people quietly get a wrong number and trust it. Figure 3 shows the mechanic that makes a histogram work: one observation lands in the cumulative buckets at or above it.

![A branching diagram showing a single 0.27 second latency observation incrementing the le 0.3, le 1.0, and le plus infinity buckets while leaving le 0.1 untouched and adding to the sum and count series](/imgs/blogs/metrics-and-time-series-done-right-3.png)

To compute a p99 latency from a histogram you write, idiomatically:

```promql
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)
```

Read it inside out. `http_request_duration_seconds_bucket` is the set of cumulative bucket series, one per `le`. We take the `rate()` of each over five minutes (so the buckets reflect the recent traffic, not all-time totals). We `sum ... by (le)` to combine across all instances and routes while *keeping the `le` dimension* — that `by (le)` is mandatory and the most common thing people forget; without it `histogram_quantile` has nothing to interpolate over. Then `histogram_quantile(0.99, ...)` walks the cumulative buckets to find the boundary below which 99% of observations fell, and **linearly interpolates within the bucket** to estimate the value.

That last phrase is the entire caveat. **`histogram_quantile` does not know the real distribution inside a bucket — it assumes observations are spread uniformly across the bucket's width and interpolates linearly.** So your quantile is only as accurate as your bucket boundaries are close to the value you care about. Concretely: suppose your p99 truly falls between the `le=1.0` and `le=2.5` buckets. The function will report somewhere between 1.0 and 2.5 seconds, interpolated linearly — but if 99% of the requests in that bucket actually cluster near 1.1s, the linear assumption is wrong and you might read 1.8s. The estimate can be off by the full width of the bucket. And if your p99 falls in the last finite bucket before `+Inf`, the function cannot interpolate at all (the `+Inf` bucket has no upper bound) and returns the lower bound of that bucket — your p99 reads as exactly the boundary, which is silently capped.

The practical consequence: **you only get accurate quantiles near your bucket boundaries, so you must put your boundaries where your SLO thresholds and your real latencies live.** The default Prometheus client buckets (`.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10`) are fine for a service whose latencies are in the tens to hundreds of milliseconds. But if your SLO says "p99 under 300ms" and your real p99 is 280ms, you want fine-grained buckets right around 300ms — say `.2, .25, .3, .35, .4` — so the interpolation is tight there. If your service is much faster (a cache returning in single-digit milliseconds) the default coarse low end will give you garbage; add `.001, .002, .003`. Pick buckets around the values that decide pass-or-fail for your SLO. A histogram with thoughtful buckets is one of the highest-leverage instrumentation decisions you can make.

There is a newer answer to the bucket-choice problem worth knowing about: **native histograms** (also called sparse or exponential histograms), introduced in recent Prometheus versions. Instead of you choosing fixed buckets, native histograms use exponentially-spaced buckets at a configurable resolution, automatically covering many orders of magnitude with far fewer series and no manual boundary tuning. They are the future and worth adopting for new services, but classic explicit-bucket histograms are still everywhere and you must understand their caveat to read existing dashboards correctly.

One more subtlety that separates people who read histograms correctly from people who quote nonsense: **you cannot recover the true maximum from a histogram, and the `_sum` is the only exact aggregate you get.** The `_count` and `_sum` series let you compute an *exact* average (`rate(_sum) / rate(_count)`), which is genuinely useful and often overlooked — the average latency is precise even when the percentiles are interpolated. But there is no "max" bucket; the largest value you can report is bounded by your highest finite `le`, and anything above it lands in `+Inf` and is invisible except as a count. So when an SLO is about the worst case, a histogram can tell you "1% of requests exceeded 2.5 seconds" (the count in `+Inf` over the total) but not "the slowest request took 9 seconds." If you need the true max or the true tail beyond your buckets, that is a job for a trace or a log of slow requests, not a histogram. Knowing the *exact* boundary of what a histogram can and cannot tell you keeps you from over-claiming a number in a postmortem.

A related discipline: when you compare a percentile across two time ranges (this week's p99 vs last week's), make sure both are computed over comparable windows and traffic volumes. A p99 computed over a 5-minute window during a low-traffic trough is a noisy estimate built from a handful of observations; the same p99 over a busy hour is statistically solid. Percentiles are estimates, and their confidence scales with the number of observations behind them — a fact worth remembering before you alert on a p99 wobble that is really just thin overnight traffic.

#### Worked example: reading a p99 honestly

Suppose `sum(rate(http_request_duration_seconds_bucket[5m])) by (le)` over the last five minutes gives cumulative rates of: `le=0.1 → 800`, `le=0.25 → 950`, `le=0.5 → 990`, `le=1.0 → 998`, `le=2.5 → 999.5`, `le=+Inf → 1000`. The total is 1000 observations/sec. The p99 is the value below which 990 of every 1000 observations fall. From the cumulative counts, 990 is reached exactly at the `le=0.5` boundary — so the p99 is approximately 0.5 seconds. Now notice the fragility: the *next* boundary up, `le=1.0`, holds 998. If our true p99 had been at 991 observations, the function would interpolate linearly between `le=0.5` (990) and `le=1.0` (998), giving roughly `0.5 + (991-990)/(998-990) × (1.0-0.5) = 0.5 + 0.0625 = 0.56s`. But the real requests in that 0.5-to-1.0 bucket might all cluster at 0.51s — the linear interpolation has no way to know, and it would over-report by 50ms. If our SLO threshold is 0.6s, that 50ms error could flip a "passing" reading into a "failing" alert. The fix is a bucket at `le=0.6` so the boundary sits right on the decision line. The arithmetic is honest only when your boundaries straddle the value you are judging.

## 5. RED and USE: the two standard instrumentation menus

You do not have to invent what to measure from scratch. Two battle-tested menus cover the vast majority of cases, and Figure 4 puts them side by side. The trick is knowing which applies to what you are looking at: RED for things that *serve requests*, USE for things that are *resources*.

![A matrix contrasting the RED method which watches request-driven services through rate errors and duration against the USE method which watches resources through utilization saturation and errors](/imgs/blogs/metrics-and-time-series-done-right-4.png)

**The RED method** (popularized by Tom Wilkie) is for **request-driven services** — anything that handles requests and returns responses. You measure three things per service:

- **Rate** — requests per second. `sum(rate(http_requests_total[5m]))`.
- **Errors** — failed requests per second (or the error *ratio*, which is usually more useful). `sum(rate(http_requests_total{code=~"5.."}[5m]))`.
- **Duration** — the latency distribution, read as percentiles from a histogram. `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`.

RED maps almost one-to-one onto three of the **four golden signals** from the Google SRE book (latency, traffic, errors — the fourth being saturation, which USE covers). If you instrument every service the same RED way, every service dashboard looks the same, on-call engineers can read any service's health at a glance without learning a new layout, and your SLIs fall right out: availability is `1 − errors/rate`, latency SLI is the duration percentile.

**The USE method** (from Brendan Gregg) is for **resources** — CPUs, disks, network interfaces, memory pools, connection pools, thread pools, queues. For every resource you check three things:

- **Utilization** — the percentage of time the resource was busy (CPU % busy, disk % time doing I/O).
- **Saturation** — the degree to which the resource has extra work it cannot service yet, usually a queue length (run-queue depth, disk I/O queue, connection-pool wait count). Saturation is the early-warning signal: a resource at 100% utilization with a growing queue is in trouble; at 100% with no queue it is merely fully used.
- **Errors** — the count of error events for that resource (disk errors, dropped packets, failed allocations).

The two methods are complementary, not competing. RED tells you *that* users are in pain (high error rate, high p99). USE tells you *why* — which resource is saturated and causing it. During an incident you start at RED ("users are seeing errors, latency is up") and drill into USE ("the database connection pool is saturated, every connection is checked out and requests are queuing"). Instrument both and you have the symptom and the cause.

| | RED method | USE method |
| --- | --- | --- |
| Applies to | Request-driven services | Resources (CPU, disk, pools, queues) |
| Question | Are users in pain? | Which resource is the bottleneck? |
| Signals | Rate, Errors, Duration | Utilization, Saturation, Errors |
| Maps to | Golden signals: traffic, errors, latency | Golden signal: saturation |
| Your SLIs come from | Errors/rate, duration percentile | Saturation as leading indicator |
| Best for | Dashboards, alerts on symptoms | Diagnosis, capacity planning |

Here is USE made concrete with the metrics you would actually scrape. For a CPU you get utilization from the node exporter, saturation from the run-queue, and errors are rare:

```promql
# CPU utilization: fraction of time NOT idle, per instance
1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

# CPU saturation: run-queue length above the core count signals waiting work
node_load1 / on(instance) count by (instance) (node_cpu_seconds_total{mode="idle"})

# Memory saturation: swapping is memory's queue
rate(node_vmstat_pswpin[5m]) + rate(node_vmstat_pswpout[5m])
```

For an application's own resources — the thread pool, the database connection pool, the request queue — you instrument them yourself with gauges, and saturation is almost always a *queue or wait count*:

```promql
# Connection-pool saturation: in-use vs the pool's capacity
db_pool_connections_in_use / db_pool_connections_max

# Request-queue saturation: depth is the leading indicator of latency
max by (instance) (http_request_queue_depth)
```

The reason saturation is the *leading* indicator deserves a moment, because it is the deepest idea in USE. Utilization tells you a resource is busy; saturation tells you it has run out of headroom and work is now *waiting*. By Little's Law, the number of items waiting in a queue is the arrival rate times the wait time ($L = \lambda W$), so a growing queue depth is a direct, early measurement of latency that is *about to* happen — the requests are stacked up but have not yet timed out into errors. A resource at 100% utilization with an empty queue is healthy (fully used, nobody waiting); the same resource at 100% with a queue climbing from 2 to 200 is seconds away from a latency cliff. Alerting on saturation buys you the lead time that alerting on errors does not — by the time errors appear, the users have already felt the pain. This is also why a pure-utilization alert ("CPU > 80%") is a poor page: 80% utilization may be perfectly fine if nothing is queuing, and you will cry wolf, whereas a saturation alert fires only when work is genuinely backing up.

A note on the boundary between this post and architecture: *which* resources to bulkhead, how to shed load, and how to design for graceful degradation are design-time decisions covered in the system-design series — see [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) and [capacity planning and autoscaling](/blog/software-development/system-design/capacity-planning-and-autoscaling). Here we are concerned with *measuring* utilization and saturation so you know when those designs are being stressed. The backpressure and load-shedding mechanics that respond to a saturated queue are treated in [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure).

## 6. Instrument an HTTP service with RED (the practice)

Theory is cheap; let us instrument a real service. Here is a Go HTTP handler with RED instrumentation using the official Prometheus client. It defines a counter for rate-and-errors and a histogram for duration, with carefully bounded labels.

```go
package main

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// Counter: gives us Rate and Errors. Labels are BOUNDED:
	// route is a template, method and code are small fixed sets.
	httpRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total HTTP requests by route, method, and status code.",
		},
		[]string{"route", "method", "code"},
	)

	// Histogram: gives us Duration. Buckets chosen around our
	// 300ms p99 SLO so interpolation is tight near the threshold.
	httpDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request latency in seconds.",
			Buckets: []float64{0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 2.5},
		},
		[]string{"route", "method"},
	)
)

// instrument wraps a handler, recording RED metrics. The route
// label is the TEMPLATE ("/orders/:id"), never the raw path.
func instrument(route string, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		h(rec, r)
		elapsed := time.Since(start).Seconds()
		code := strconv.Itoa(rec.status)
		httpRequests.WithLabelValues(route, r.Method, code).Inc()
		httpDuration.WithLabelValues(route, r.Method).Observe(elapsed)
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/orders", instrument("/orders", ordersHandler))
	mux.Handle("/metrics", promhttp.Handler())
	http.ListenAndServe(":8080", mux)
}
```

The two things to notice are the things that keep you out of trouble. First, the **`route` label is the template** (`/orders` or `/orders/:id`), never the raw path with the order ID in it — that single decision is the difference between forty route values and forty million. Second, the **histogram buckets are chosen around the 300ms SLO**, not left at the defaults, so the p99 interpolation is accurate at the threshold that decides pass-or-fail.

The Python equivalent with the official client is just as direct:

```python
from prometheus_client import Counter, Histogram
import time

REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["route", "method", "code"],
)

DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["route", "method"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 2.5),
)

def handle(route, method, request):
    start = time.perf_counter()
    status = 200
    try:
        status = process(request)   # your real work
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(route, method, str(status)).inc()
        DURATION.labels(route, method).observe(elapsed)
```

Now the PromQL that turns this instrumentation into the three RED signals plus the SLIs:

```promql
# Rate: requests per second across the whole service
sum(rate(http_requests_total[$__rate_interval]))

# Rate broken out by route, to find the busy endpoints
sum by (route) (rate(http_requests_total[$__rate_interval]))

# Error rate: 5xx responses per second
sum(rate(http_requests_total{code=~"5.."}[$__rate_interval]))

# Error RATIO: the availability SLI, a fraction in [0,1]
sum(rate(http_requests_total{code=~"5.."}[$__rate_interval]))
  /
sum(rate(http_requests_total[$__rate_interval]))

# Duration: p99 latency, the latency SLI
histogram_quantile(
  0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket[$__rate_interval]))
)

# Duration: p50 and p99 together for one panel
histogram_quantile(0.50, sum by (le) (rate(http_request_duration_seconds_bucket[$__rate_interval])))
histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[$__rate_interval])))
```

That error-ratio query is your **availability SLI** — the fraction of requests that failed — and it is the input to your SLO and your error budget. The latency percentile is your **latency SLI**. Two queries, and you have the numbers the entire rest of the reliability loop runs on. For the full treatment of turning these ratios into SLOs and budgets, see [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) and the SLO posts in this series.

### Proof: what good instrumentation buys you

Before this RED instrumentation, the team I described in the intro had per-host CPU and memory dashboards and nothing else. When users complained, the mean time to even *identify which service* was at fault was around 25 minutes of clicking through host graphs. After adding RED to every service — one counter, one histogram, identical PromQL everywhere — the same identification took under 2 minutes: open the service-overview dashboard, see which service's error ratio spiked and when, and start there. That is not a fabricated number from a vendor case study; it is the very ordinary result of measuring the symptom (user-facing errors and latency) directly instead of inferring it from resource graphs. The leverage of RED is that it standardizes *what* you look at, so every responder is fast on every service.

## 7. Cardinality explosions: the failure mode that kills Prometheus

Now the dark side, and the reason I started this post at 3:47 in the morning. **Cardinality is the number of unique label-set combinations, which equals the number of series, and it is the resource that, when it runs away, takes your monitoring down.** Figure 5 shows the exact incident.

![A before and after comparison showing that adding a user_id label and a full_path label pushed the series count from fifty thousand to twelve million causing a Prometheus out of memory crash, while bounded labels brought it back to fifty thousand](/imgs/blogs/metrics-and-time-series-done-right-5.png)

The mechanism is multiplication. Suppose `http_requests_total` has these labels and value-counts: `route` (40 templates), `method` (5 values), `code` (8 values). The number of series for that one metric is at most 40 × 5 × 8 = 1,600. Comfortable. Now a well-meaning engineer adds a `user_id` label "to debug a specific customer's errors." If you have 200,000 active users, the cardinality becomes 1,600 × 200,000 = 320 million *potential* series, and even at the active subset it is millions. Add `full_path` — the raw URL including query strings and IDs — which is effectively unbounded, and you have multiplied by another large and ever-growing factor. The series count goes from fifty thousand across your whole fleet to twelve million, the Prometheus head block can no longer fit in RAM, the process spends all its time in garbage collection, queries time out, and eventually the OOM killer reaps it. Your monitoring dies precisely when you need it most.

The insidious part is that the offending change looks harmless in review. `WithLabelValues(route, method, code, userID)` is one extra argument. It passes tests. It works fine in a dev environment with three users. It only detonates in production at scale, days later, and by then nobody connects the OOM to that diff.

It is worth being precise about *why* cardinality hurts, because "it uses more memory" undersells it. Prometheus keeps every active series' most recent sample and its label set in the in-memory head block, and it maintains an **inverted index** mapping every label name and every label value to the set of series that carry it. Both structures scale with the number of series, but the index scales with the number of *distinct label values* too, which is why a single label with a million values is far worse than a million series spread across ten well-bounded labels. On top of memory, query cost scales with series touched: a query like `sum(rate(http_requests_total[5m]))` must read and rate every matching series, so a metric with twelve million series makes even a simple aggregation slow, and that slowness compounds when dashboards and alerts re-run constantly. And the cost is not only RAM and CPU — it is disk (every series writes a chunk every couple of hours) and, if you run a remote-write backend like Thanos or a managed service billed per series, it is literally money. A cardinality leak is simultaneously a memory leak, a performance regression, a storage-cost spike, and a billing surprise.

There is also a churn dimension people miss. A label whose values *change over time* — a deploy hash, a pod name in a frequently-restarting deployment, a timestamp-bearing value — creates *new* series continuously even if the count active at any instant is modest. Those old series do not vanish instantly; they linger through the staleness window and stay on disk for the retention period. So `rate(prometheus_tsdb_head_series_created_total[5m])` climbing steadily, even while `prometheus_tsdb_head_series` looks stable, is the signature of a *churning* label — a pod name or a version string in a label position — and it bloats your blocks and slows long-range queries even though a snapshot looks fine. Watch the creation rate, not just the instantaneous count.

### The rules that prevent it

The discipline is simple to state and must be enforced without exception:

1. **Never put an unbounded or high-cardinality value in a label.** User IDs, request IDs, trace IDs, email addresses, session tokens, raw URLs with IDs in them, full SQL queries, error messages with embedded values — none of these belong in a metric label. They belong in **logs** (searchable, no cardinality cost in the metrics system) or **traces** (where high-cardinality context like a trace ID is exactly the point). This is one of the clearest cases of choosing the right observability pillar for the job — see the sibling on when to use metrics, logs, and traces, and the architecture-level treatment in [observability metrics logs traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design).
2. **Bound every label to a small, fixed set of values.** `code` (a handful of HTTP statuses), `method` (GET, POST, ...), `route` as a *template* (`/orders/:id`, not `/orders/8675309`). If you cannot enumerate the possible values on a whiteboard, it is too high-cardinality for a label.
3. **Templatize paths before they become labels.** Most web frameworks expose the matched route pattern (`r.Pattern`, `request.matched_route`, etc.). Use that, never the raw path. If you must derive it, normalize aggressively: replace numeric and UUID path segments with placeholders.
4. **Use exemplars for the high-cardinality needle.** If you genuinely need to jump from a slow-latency bucket to the specific trace that was slow, that is what **exemplars** are for: a histogram bucket can carry a small number of exemplar trace IDs as out-of-band data that does not create new series. The metric stays low-cardinality; the exemplar links you to the high-cardinality trace.

#### Worked example: the 50k → 12M incident and its fix

Here is the audit and the fix with the real arithmetic. Before the bad deploy, the fleet ran about **50,000 active series** total. The `http_requests_total` metric contributed 40 routes × 5 methods × 8 codes = 1,600 series, multiplied across ~30 services with similar shapes — call it 48,000, plus assorted gauges. Healthy.

The deploy added two labels to the request counter on the busiest service: `user_id` and `full_path`. The busiest service alone served around 180,000 distinct users that week, and `full_path` (raw URLs with IDs and query params) contributed thousands of distinct values per route. The counter's cardinality on that one service went from 1,600 to roughly 1,600 × 180,000 (capped by which combinations actually occurred) — observed as about **12 million active series** fleet-wide, a 240× explosion. Prometheus, sized for ~100k series, OOM-crashed.

The fix was two lines of code and a confirmation: drop `user_id` and `full_path` from the labels, keep `route` (template), `method`, and `code`. Series fell back to ~50,000. To find *who else* was at risk before it happened again, we ran a cardinality-audit query (next section) and set a per-job series limit. The before→after is stark: **12,000,000 series → 50,000 series, OOM crashes → stable**, and the data we actually lost (per-user request counts) was never something a metric should have held — it moved to logs, where a query like "requests by this user" is a normal log search with no cardinality cost.

### The cardinality-audit queries

You should know how to find your worst offenders before they find you. These are the queries every SRE should have bookmarked. To see which metric names have the most series:

```promql
# Top metric names by series count (run against Prometheus itself)
topk(10, count by (__name__)({__name__=~".+"}))
```

To see total active series and head-block size, Prometheus exposes metrics about itself:

```promql
# Total active series in the head block
prometheus_tsdb_head_series

# Series created per second — a rising line is a cardinality leak
rate(prometheus_tsdb_head_series_created_total[5m])
```

To find which *label* on a given metric is exploding its cardinality, the `count_values` and label-count approach helps, but the most practical day-to-day tool is the Prometheus UI's "TSDB Status" page (Status → TSDB Status), which lists the top series by metric, the label names with the most values, and the label-value pairs in the most series. Make checking that page part of every onboarding. And in CI or as a Prometheus rule, enforce a limit: the scrape config supports `sample_limit` (rejects a target whose scrape exceeds N samples) and you can alert on `prometheus_tsdb_head_series` crossing a threshold so a leak pages you while it is still a leak and not yet an outage.

```yaml
# In prometheus.yml scrape config: reject a target that
# tries to expose more than 100k series in one scrape.
scrape_configs:
  - job_name: "api"
    sample_limit: 100000
    static_configs:
      - targets: ["api:8080"]
```

## 8. Recording rules: precompute the expensive queries

Some PromQL expressions are expensive — a `histogram_quantile` over a high-cardinality `_bucket` rate, summed `by (le)` across thousands of series, evaluated every time a dashboard refreshes or an alert fires. If five dashboards and three alerts all recompute the same SLI ratio every fifteen seconds, you are doing the same heavy query dozens of times a minute. **Recording rules** fix this by evaluating an expression on a schedule and saving the result as a brand-new, cheap time series. Figure 6 shows the flow.

![A branching diagram showing a recording rule evaluating an expensive raw bucket expression every thirty seconds into a new cheap SLI series that both a dashboard and a burn rate alert read in five milliseconds, versus an ad hoc query taking twelve hundred milliseconds without the rule](/imgs/blogs/metrics-and-time-series-done-right-6.png)

A recording rule is a YAML entry that says "evaluate this expression on this interval and store it under this new metric name." The naming convention is `level:metric:operation` — for example `job:http_requests:rate5m` reads as "aggregated to the job level, derived from http_requests, the 5-minute rate." Here is a rule group that precomputes the request rate, the error ratio (the availability SLI), and the p99 latency for every job:

```yaml
groups:
  - name: red_recording_rules
    interval: 30s
    rules:
      # Request rate per job
      - record: job:http_requests:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      # 5xx error rate per job
      - record: job:http_requests_errors:rate5m
        expr: sum by (job) (rate(http_requests_total{code=~"5.."}[5m]))

      # Availability SLI: error ratio per job (the number SLOs use)
      - record: job:slo_errors:ratio5m
        expr: |
          sum by (job) (rate(http_requests_total{code=~"5.."}[5m]))
            /
          sum by (job) (rate(http_requests_total[5m]))

      # p99 latency per job
      - record: job:http_latency:p99_5m
        expr: |
          histogram_quantile(0.99,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m])))
```

Now a dashboard panel for p99 latency is just `job:http_latency:p99_5m{job="api"}` — a single cheap series read, not a multi-thousand-series interpolation. The win is threefold: **speed** (dashboards that loaded in over a second now load in milliseconds), **consistency** (every dashboard and every alert reads the *same* definition of the SLI, so they never disagree — a constant source of confusion when one panel computes the error ratio slightly differently from the alert), and **cost** (one evaluation instead of dozens).

Recording rules are the bridge to burn-rate alerting, which is the subject of the alerting sibling. A multi-window burn-rate alert needs the SLI error ratio computed over several windows (5m, 1h, 6h). Precompute each window once as a recording rule and your alert expressions become trivial comparisons against the budget. The error-budget math itself — burn rate = observed error rate ÷ (1 − SLO), and how a 14× burn exhausts a 30-day budget in roughly two days — is developed in [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) and operationalized in the alerting post; here the point is that recording rules are what make those alerts fast and consistent.

Concretely, the recording rules that feed a multi-window burn-rate alert look like this — the same error-ratio expression at several windows, each stored as its own series:

```yaml
groups:
  - name: slo_burn_inputs
    interval: 30s
    rules:
      - record: job:slo_errors:ratio5m
        expr: |
          sum by (job) (rate(http_requests_total{code=~"5.."}[5m]))
            / sum by (job) (rate(http_requests_total[5m]))
      - record: job:slo_errors:ratio1h
        expr: |
          sum by (job) (rate(http_requests_total{code=~"5.."}[1h]))
            / sum by (job) (rate(http_requests_total[1h]))
      - record: job:slo_errors:ratio6h
        expr: |
          sum by (job) (rate(http_requests_total{code=~"5.."}[6h]))
            / sum by (job) (rate(http_requests_total[6h]))
```

With these in place, the alert that catches a *fast* burn (a deploy melting the budget in hours) compares the 5-minute and 1-hour ratios against a high multiple of the budget; the alert that catches a *slow* burn (a steady leak draining the budget over days) compares the 1-hour and 6-hour ratios against a lower multiple. Each alert reads precomputed series and evaluates in milliseconds, and — critically — every alert and dashboard agrees on what "the error ratio" is, because they all read `job:slo_errors:ratioNm`. The full multi-window, multi-burn-rate alert construction is the alerting sibling's job; the metrics contribution is making the inputs cheap, consistent, and always available.

### A caution on recording-rule cardinality and ordering

Two operational gotchas with recording rules. First, a recording rule *creates new series*, so a rule that aggregates `by (job, route, code)` can itself add meaningful cardinality — keep recording rules aggregated to the coarsest label set the dashboard or alert actually needs, usually just `by (job)`. A rule that preserves a high-cardinality label is a quieter version of the same cardinality mistake. Second, rules within a group evaluate **sequentially top to bottom**, so a rule can depend on a series produced by an earlier rule in the *same* group; rules in *different* groups evaluate independently and possibly out of order, so never have a rule in group B depend on a rule in group A computed in the same cycle — you may read a stale value. Order dependent rules within one group.

#### Worked example: the dashboard that went from 1.2s to 5ms

A team's main SLO dashboard had eight panels, each running a `histogram_quantile` over a histogram with around 12,000 active `_bucket` series, summed `by (le)` across instances and routes. Each panel query took roughly **1,200ms** to evaluate, and with eight panels refreshing every 15 seconds plus three alerts re-running the error-ratio query, the Prometheus query engine was spending a meaningful fraction of every minute on these same expressions. We added four recording rules (rate, error rate, error ratio, p99), evaluated every 30 seconds. Each dashboard panel and each alert switched to reading the single precomputed series — a **~5ms** read of one series instead of a 1,200ms aggregation over thousands. The dashboard load time dropped from over a second to imperceptible, the query-engine load fell by more than an order of magnitude, and — the subtle win — the alert and the dashboard now read byte-for-byte the same number, so "the dashboard says we're fine but the alert fired" stopped happening. 1,200ms → 5ms per read, and one source of truth.

## 9. Pull, scrape intervals, and staleness

A few operational mechanics of the metrics pipeline decide whether your numbers are trustworthy, and Figure 7 lays out the scrape lifecycle that governs them.

![A timeline of the pull model showing scrapes at zero, fifteen, and thirty seconds providing the samples a one minute rate window needs, and a missed scrape marking the series stale after five minutes](/imgs/blogs/metrics-and-time-series-done-right-7.png)

**Pull vs push.** Prometheus pulls — it initiates the scrape — which has real advantages for reliability: the monitoring system controls the cadence (no application can accidentally flood it), a target that is down simply fails to be scraped (giving you a clean `up == 0` signal for free), and you can scrape the same target from a test instance without touching production config. The trade-off is that pull needs *service discovery* — Prometheus has to know what to scrape — which it solves with Kubernetes/Consul/EC2 service discovery. For workloads that do not fit pull — short-lived batch jobs that finish before any scrape lands, or events behind a network boundary — there is the **Pushgateway** (a holding pen a batch job pushes its final metrics to, which Prometheus then scrapes), but it is a deliberate exception, not the default. Use it for batch jobs' completion metrics; do not use it as a general push endpoint, because it breaks the clean liveness signal and accumulates stale series.

**Scrape interval.** This is the resolution knob. A 15-second interval gives you fine resolution at the cost of more samples (and thus more storage and more series-churn cost); 30 or 60 seconds is cheaper but blurs short events. The interval also sets the *floor* on your rate windows, as we saw: a rate window must comfortably exceed the interval, so a 60-second scrape forces minimum 4-5 minute rate windows for stability, which means slower detection. There is a direct tension between scrape cost and detection speed, and the right answer depends on the SLO — a payments service might justify a 15-second scrape for fast detection, an internal batch dashboard is fine at 60.

**Staleness.** When a scrape that should have happened does not — the target was briefly unreachable, or it disappeared from service discovery — Prometheus does not keep returning the last value forever. After a configured staleness window (default 5 minutes) it marks the series **stale** and stops returning it in queries, so a `rate()` over a vanished target correctly goes to no-data rather than reporting a frozen, misleading value. This is why a `rate()` that suddenly returns nothing during an incident often means "the target stopped being scraped," not "the rate is genuinely zero" — and it is exactly why you also alert on `up == 0` and on `absent()` of a metric you expect to always exist, so a target that silently disappears pages you instead of just going quiet.

This staleness behavior interacts with a trap that catches everyone at least once: the difference between "zero" and "no data." If a counter for a rare error has never been incremented since the process started, its series may not exist at all — and `sum(rate(rare_errors_total[5m]))` returns *empty*, not zero. An alert written as `sum(rate(rare_errors_total[5m])) > 0` will never fire on its own absence, but worse, a panel will show a gap rather than a flat zero line, and a tired responder may misread "no data" as "all good." The fix is to initialize your counters to zero at startup (`promauto` and the Python client let you pre-declare the label combinations you care about so the series exists at value 0 from the start), and to use `or vector(0)` in queries where an empty result should read as zero. The discipline is to make "nothing has gone wrong" look like an explicit zero, not like missing data, because a human under pressure cannot tell a healthy silence from a broken pipeline.

```promql
# Alert input: a target we expect is not being scraped
up{job="api"} == 0

# Alert input: a metric that should always exist has vanished
absent(http_requests_total{job="api"})
```

## 10. Naming, units, and the golden-signals instrumentation menu

Good metrics are *conventionally* named so that anyone can read them and aggregation works. The Prometheus naming conventions are not bureaucracy — they make queries composable.

- **Counters end in `_total`.** `http_requests_total`, `errors_total`, `bytes_processed_total`. The suffix signals "this is a monotonic counter, wrap me in `rate()`."
- **Use base units, always.** Seconds, not milliseconds. Bytes, not kilobytes. Ratios in `[0,1]`, not percentages. So `http_request_duration_seconds`, `queue_size_bytes`, `cache_hit_ratio`. Base units mean every query and dashboard speaks the same language; you never have to remember whether a given metric was in ms or µs. You display in friendlier units in Grafana; you *store* in base units.
- **The unit goes in the name.** `..._seconds`, `..._bytes`, `..._ratio`, `..._info` (for a constant-1 gauge carrying version labels). The name is self-documenting.
- **Namespace with a prefix.** `myapp_http_requests_total`, so your application metrics do not collide with library or exporter metrics.

The **four golden signals** — latency, traffic, errors, saturation — are the canonical "what to measure" list from the Google SRE book, and they map cleanly onto the metric types and methods we have built. Figure 8 is the decision tree that ties type selection to labeling discipline.

![A decision tree that asks whether you are counting events, reading a current level, or capturing a distribution, leading to counter gauge or histogram respectively and to the rule that every label must be bounded with no raw ids](/imgs/blogs/metrics-and-time-series-done-right-8.png)

Here is the golden-signals menu as concrete metrics and queries, so you have a copy-and-adapt checklist for any new service:

| Golden signal | Metric type | Metric | PromQL |
| --- | --- | --- | --- |
| Traffic | Counter | `http_requests_total` | `sum(rate(http_requests_total[5m]))` |
| Errors | Counter | `http_requests_total{code=~"5.."}` | error ratio (good/total) |
| Latency | Histogram | `http_request_duration_seconds_bucket` | `histogram_quantile(0.99, ...)` |
| Saturation | Gauge | `queue_depth`, `pool_in_use` | `max(queue_depth)` vs capacity |

Instrument every service against this menu, name everything by convention, bound every label, and your observability is boringly consistent — which is exactly what you want at 3am. The dashboards-sibling, dashboards that tell the truth, takes these signals and lays them out so a tired responder reads health in five seconds; the alerting sibling, alerting that does not cry wolf, turns the error and latency SLIs into symptom-based, multi-window burn-rate pages.

One last instrumentation habit that pays for itself: the **`_info` gauge** pattern. You often want to attach slowly-changing metadata — the build version, the git SHA, the region, the deployment color — to a service without exploding cardinality on your high-traffic metrics. The pattern is a single constant-`1` gauge named `..._info` whose labels carry the metadata: `build_info{version="2.4.1", commit="a1b2c3", region="us-east-1"} 1`. It is one series per distinct build, which is naturally bounded (you do not deploy a thousand versions a day), and you can join it onto your other metrics at query time with vector matching to slice by version — for example to confirm that the error rate spiked exactly when version 2.4.1 rolled out. This keeps the version *out* of your request counter's labels (where it would multiply cardinality and churn on every deploy) while still letting you correlate behavior to releases. Metadata goes in an `_info` gauge; it does not go in your hot-path labels.

## War story: the metric that took down the monitoring

The incident I have been threading through this post is real in its shape and worth telling end to end, because it is the most common self-inflicted metrics disaster I have seen, across more than one company.

A team wanted to debug why a *specific* enterprise customer was getting intermittent errors. The fastest-seeming fix was to add a `customer_id` label to the request and error counters so they could filter the dashboard to that one customer. It shipped on a Thursday. It worked — for that customer, on a low-traffic staging-like view. Over the weekend, traffic from all customers flowed through the new labels. By Monday the Prometheus server's memory had climbed from a steady 6GB to 28GB and then the pod was OOM-killed. It restarted, replayed its write-ahead log, climbed again, and OOM-ed again — a crash loop. During that window, every dashboard and every alert was blind. A genuinely unrelated latency regression in the checkout path went unnoticed for over an hour because the system that would have caught it was itself down.

The root cause in the postmortem was not "an engineer made a mistake" — it was that *nothing prevented* an unbounded label from reaching production. The contributing factors were: no `sample_limit` on the scrape config, no alert on `rate(prometheus_tsdb_head_series_created_total[5m])`, no code-review guideline about label cardinality, and no easy alternative for the actual need (per-customer debugging). The fixes were all of those: a `sample_limit` so a runaway target is rejected rather than ingested, an alert on series-creation rate so a cardinality leak pages while it is small, a lint rule and review checklist flagging any new label, and — for the real need — moving per-customer error investigation to **logs** (a structured log line per error with the `customer_id` field, queryable in Loki without any cardinality cost in Prometheus). The blameless framing mattered: the engineer who added the label was the one who wrote the clearest part of the postmortem, because nobody was hunting for someone to punish. For how this kind of investigation is run scientifically, the debugging series' [stop guessing, the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) is the companion read.

The measured aftermath: series count capped at ~80,000 (with a 100k `sample_limit` as the hard ceiling), zero monitoring-OOM incidents in the following year, and the per-customer debugging that started it all was *faster* in logs than it ever was as a metric label, because logs let you see the actual error messages and request context, not just a count.

## How to reach for this (and when not to)

Metrics are powerful and they are not the answer to everything. Reach for a **metric** when you want an aggregate, low-cost, always-on signal you can alert on and trend over time: rates, error ratios, latency percentiles, saturation levels — the golden signals. Metrics are cheap per data point and answer "how much / how often / how fast" across all requests.

Reach for **logs** when you need the high-cardinality detail of an individual event: *this* request, from *this* user, with *this* error message and *these* parameters. The moment you find yourself wanting to put a user ID, request ID, or full URL in a metric label, that is your signal to use a log line instead. Logs are the right home for unbounded cardinality.

Reach for **traces** when you need to follow a single request across many services and see where the time went. A trace ID is high-cardinality by design and belongs in the tracing system, with an **exemplar** linking a slow latency bucket to the specific trace. The when-to-use-which decision is its own sibling post; the short version is that metrics tell you *something is wrong and roughly where*, traces tell you *which service and span*, and logs tell you *exactly what happened to that one request*.

And explicitly, do **not**:

- Do not add a label you cannot enumerate on a whiteboard. If the value set is "all user IDs" or "all URLs," it is a log field, not a label.
- Do not use a **summary** for a multi-instance service expecting to read a fleet percentile — its client-side quantiles cannot be aggregated, and the average of percentiles is a lie. Use a histogram.
- Do not leave histogram buckets at the defaults if your SLO threshold or real latencies fall in a sparse part of the range — you will read interpolated quantiles that are off by the bucket width exactly where it matters.
- Do not scrape every service at 15 seconds reflexively; match the interval to the SLO's required detection speed and pay the storage cost only where it buys you something.
- Do not skip recording rules for your SLI definitions. The first time a dashboard and an alert disagree because they computed the ratio slightly differently, you will wish you had one precomputed source of truth.

## Key takeaways

1. **A time series is a metric name plus a label set, holding timestamped samples. Cardinality — the count of unique label-set combinations — is the resource that governs Prometheus's memory, speed, and cost.** Everything else follows from this.
2. **Pick the metric type from the question: counter for events that only go up (read with `rate()`), gauge for a current level, histogram for a distribution you want quantiles from, summary only for a single instance.** Summaries cannot be aggregated across instances — never average percentiles.
3. **Rate first, then sum: `sum(rate(...))`, never `rate(sum(...))`,** so counter resets are handled per series. Use `rate()` for alerts and `$__rate_interval` in Grafana; reserve `irate()` for high-resolution dashboards only.
4. **`histogram_quantile()` interpolates linearly within a bucket, so your quantiles are only accurate near your bucket boundaries.** Put buckets where your SLO thresholds and real latencies live, not at the defaults.
5. **Instrument request-driven services with RED (Rate, Errors, Duration) and resources with USE (Utilization, Saturation, Errors).** RED tells you users are in pain; USE tells you which resource is why. Your availability and latency SLIs fall right out of RED.
6. **Never put an unbounded value — user IDs, request IDs, full URLs — in a label.** Bound every label, templatize routes, and send the high-cardinality needle to logs and traces (or attach it as an exemplar). A `sample_limit` and a series-creation-rate alert catch a leak before it OOMs your monitoring.
7. **Precompute expensive and shared expressions with recording rules** named `level:metric:operation`, so dashboards and alerts read one cheap, consistent series in milliseconds instead of recomputing a thousand-series aggregation each time.
8. **Name by convention: counters end in `_total`, store in base units (seconds, bytes, ratios), put the unit in the name.** Conventional names make queries composable and dashboards readable at 3am.

## Further reading

- **Google SRE Book — "Monitoring Distributed Systems"** and the **SRE Workbook — "Implementing SLOs"** chapters, for the golden signals and the SLI-as-a-ratio foundation this post's PromQL produces.
- **Prometheus documentation — "Metric types," "Querying basics / functions," "Histograms and summaries," and "Recording rules,"** the authoritative reference for the four types, `rate`/`irate`, `histogram_quantile`, and rule syntax.
- **Tom Wilkie — "The RED Method"** and **Brendan Gregg — "The USE Method,"** the two instrumentation menus, straight from the people who named them.
- **Prometheus blog — "Native Histograms,"** for the modern answer to the bucket-choice problem.
- Within this series: [reliability is a feature, the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) (the loop this measuring step sits in), [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) (what to measure), and the siblings on metrics, logs, and traces when to use which; dashboards that tell the truth; and alerting that does not cry wolf (which all build on the metrics laid down here). The error-budget arithmetic these SLIs feed lives in [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability).
- For the architecture-time side of these topics, the system-design series covers [observability metrics logs traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) and [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads).

---
title: "SLOs, Golden Signals, and Alerting for Microservices: Get Paged Only When It Matters"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Turn a flood of microservice metrics into a handful of reliability targets and burn-rate alerts that wake you only when users are actually hurting — with the error-budget and composed-availability math worked out in full."
tags:
  [
    "microservices",
    "slo",
    "sli",
    "observability",
    "alerting",
    "error-budget",
    "sre",
    "burn-rate",
    "distributed-systems",
    "software-architecture",
    "backend",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-1.webp"
---

At 03:11 on a Saturday, the ShopFast on-call engineer's phone buzzed for the fourteenth time that night. The pages all said variations of the same thing: `HighCPUUsage on order-svc-7f9c`, `MemoryPressure on payment-worker-3`, `DiskIOLatency on redis-cache-2`. She had acknowledged every one, looked at the box in question, found it busy but fine, and gone back to sleep — until the next one fired four minutes later. By 06:00 she had silenced the whole alerting pipeline out of exhaustion. And that is exactly when the real incident started: a slow payment provider began timing out, checkouts started failing, and customers flooded support. The page that would have told her about *that* never arrived, because she had muted everything to survive the night of noise. The post-mortem's most damning line was four words long: **"alerts fired; nobody listened."**

This is the failure mode that observability tooling, by itself, does not solve. You can instrument every service with metrics, logs, and distributed traces — the previous post in this track covered [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) — and still be completely blind to whether your users are happy, while being completely buried under alerts about things that do not matter. Raw signals are not reliability. **Reliability is a decision: which signals you elevate into a small number of targets you actually promise to hold, and which conditions are worth waking a human at 3am.** That decision is what Site Reliability Engineering calls SLOs, golden signals, error budgets, and symptom-based alerting, and getting it right is the difference between an on-call rotation people fight to leave and one they trust.

By the end of this post you will be able to do the whole loop, concretely, for the ShopFast checkout journey. You will define an SLI as a measured ratio and turn it into an SLO with a real time window. You will compute the error budget that SLO implies — down to the minute — and use it to decide when to ship features and when to freeze and fix. You will instrument the four golden signals with real Prometheus queries, write a recording rule for an SLI, and write a *multi-window, multi-burn-rate* alert that pages you in minutes for a sudden outage and quietly opens a ticket for a slow leak. You will work out why a checkout journey across five services at 99.95% each is only 99.75% available — and what that means for whose SLO you have to tighten. And you will leave with an error-budget policy you can paste into a runbook. The figure below is the shape of the whole journey: a funnel from a flood of raw signals down to one page that actually deserves your attention.

![A vertical funnel from raw signals down through SLIs and SLOs to an error budget and finally a single burn-rate alert worth a page](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-1.webp)

A quick orientation for where this sits. Observability gives you the *data*; this post gives you the *judgment* layer on top of it. It builds directly on the resilience work from Track 4 — [timeouts, retries, circuit breakers, and bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) keep a service alive, [partial-failure handling and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) keep the journey serving when a dependency dies, and [health checks](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) let the platform heal instances automatically — but none of those tells you *whether your reliability is good enough* or *when to be woken about it*. That is the SLO's job. When a page does fire, you will need to chase the cause, which is the next post on [debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production); and the velocity that error budgets unlock gets spent safely through [deployment strategies like blue-green, canary, and feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags). Let's start with the three words everyone confuses.

## SLI, SLO, SLA: a measurement, a target, and a contract

Three acronyms, endlessly muddled, and the confusion is expensive because each one has a different audience and a different consequence when it's violated. Define them in plain language and the rest of the post falls out almost mechanically.

An **SLI — Service Level Indicator** — is a *measurement*. Specifically, it is almost always a **ratio of good events to total events**, expressed as a percentage. "What fraction of checkout requests in the last five minutes returned a success in under one second?" is an SLI. The reason it is a ratio and not, say, an average latency is that ratios are the language of reliability: 99.9% is a number you can promise, budget, and burn, while "average latency was 240ms" tells you nothing about the 0.5% of users who waited eight seconds and gave up. The SLI is the raw truth about how the service is treating users, distilled to one number between 0 and 100%.

An **SLO — Service Level Objective** — is a *target* for that indicator, **over a stated time window**. "99.9% of checkout requests succeed in under one second, measured over a rolling 28-day window" is an SLO. The window matters enormously — we'll return to it — and the target is a deliberate choice, not an aspiration. The SLO is the line you draw and then *defend*: above it, you are free to take risks and ship fast; below it, you stop and fix. It is internal. Your team owns it, your team enforces it, and crucially, **a good SLO is set lower than 100% on purpose**, because the gap below 100% is the budget that buys you the freedom to deploy.

An **SLA — Service Level Agreement** — is a *contract* with an external party, and it has **consequences with money or legal weight attached**. "We guarantee 99.5% checkout availability per month, or you get a 10% service credit" is an SLA. The audience is your customers, your sales team, and possibly a court. The cardinal rule is that **your SLA is always looser than your internal SLO**, so that you breach your private target and start fixing *long before* you breach the public contract and owe refunds. If your SLA promises 99.5% and your SLO targets 99.9%, you have a comfortable buffer; if they are equal, you will be writing refund checks the first time you're a hair late.

The figure below lays out the three side by side. Read it as: SLI is what you measure, SLO is the target you defend internally, SLA is the promise you make externally with penalties — and they should be ordered SLA < SLO, with the SLI being the live number that tells you where you stand against both.

![A matrix comparing SLI SLO and SLA across what each is, its audience, an example, what happens on breach, and how to set it](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-2.webp)

The single most important consequence of these definitions: **you set SLOs per user-facing journey, not per machine resource.** Nobody — not a customer, not the CEO, not the on-call engineer — cares whether a particular CPU is at 80%. They care whether a customer can complete a checkout. So a ShopFast SLO is phrased in the language of the journey: "checkout succeeds quickly," "the product page loads," "search returns results." It is emphatically *not* "the order service's CPU stays below 70%" — that's a resource metric, useful for capacity planning, but it is a *cause*, and as we'll see, causes are exactly the wrong thing to build your reliability targets and alerts around.

### Why a journey, not a service or a server

There's a subtlety worth pausing on, because it shapes everything downstream. ShopFast's checkout journey touches the cart service, the order service, the inventory service, and the payment service. You *could* set an SLO on each service individually — and you should, as we'll see, for internal accountability. But the SLO that matters to the business is the one on the **journey the user actually experiences**, measured as close to the user as you can get: at the API gateway or the BFF, where you can see "this checkout attempt succeeded in 0.7 seconds" or "this one failed." That number is the truth about user happiness. The per-service SLOs exist to make the journey SLO achievable and to assign blame when it isn't — but the journey SLO is the one you put in front of leadership and the one your alerts defend.

This is why a junior engineer's instinct — "let's alert when any server's CPU is high" — produces the 3am noise storm from the intro. High CPU on one box is a *cause* that *might* eventually hurt a *journey*, or might be completely harmless. The journey SLI is the *symptom*: it tells you, unambiguously, whether users are being hurt right now. Build your targets and alerts on symptoms.

## The four golden signals: what to measure on every service

Before you can define an SLI, you need to know what to measure. Google's SRE book distilled service monitoring to **four golden signals**, and the elegance is that these four, measured on every service, tell you almost everything about its health. Let me define each one in ShopFast terms.

**Latency** is how long a request takes — but with one essential refinement: you must **separate the latency of successful requests from the latency of failed ones**. A fast failure (an instant 500) can otherwise hide in a healthy-looking average, and a server that's failing fast will look *better* on latency than a healthy one, which is exactly backwards. So you measure, for example, the p99 latency of *successful* checkout requests. Latency is almost always reported as percentiles — p50, p90, p99, p99.9 — never as a mean, because the mean is dominated by the bulk and blind to the tail, and the tail is where your angriest users live.

**Traffic** is how much demand the service is receiving — requests per second for an HTTP service, messages consumed per second for a queue worker. Traffic is the denominator of your error and latency ratios and the context for everything else: a 10-error-per-minute spike means nothing at 100,000 RPS and everything at 50 RPS.

**Errors** is the rate of requests that fail. This is subtler than it sounds, because "fail" includes explicit failures (HTTP 5xx, gRPC `UNAVAILABLE`) *and* implicit ones (a 200 response with the wrong body, a request that succeeded but took 30 seconds and the user already left, a 200 that returned an empty cart when the cart was full). Defining what counts as an error *is the act of defining your SLI*, and it's where most of the real thinking lives.

**Saturation** is how *full* the service is — how close it is to a resource limit that will tip it over. This is the leading indicator: CPU utilization, memory pressure, connection-pool occupancy, queue depth, thread-pool saturation. Latency, traffic, and errors tell you how the service is doing *now*; saturation tells you how much *headroom* is left before it falls over. A service at 95% pool utilization is serving fine this instant and is one traffic bump away from a cliff.

![A stacked diagram of the four golden signals latency traffic errors and saturation feeding a verdict on whether the service is serving with headroom](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-3.webp)

Two adjacent frameworks are worth knowing because you'll hear them constantly. **RED** — Rate, Errors, Duration — is the golden signals minus saturation, popularized by Tom Wilkie for *request-driven services*: it's exactly the three signals you compute from request logs or histograms, and it maps cleanly onto SLIs. **USE** — Utilization, Saturation, Errors, coined by Brendan Gregg — is the *resource-centric* view, applied to each resource (CPU, disk, network): how busy is it, how much work is queued, are there errors? The mental shortcut: **RED for services, USE for resources, golden signals for the union.** You build SLIs from RED (they're user-facing), and you use USE-style saturation metrics to predict and diagnose, not to page on.

### Latency split is non-negotiable

One more time on the latency point, because it trips up even experienced teams. Imagine you alert when p99 latency exceeds one second. Your payment dependency starts hard-failing instantly — every payment call returns a 500 in 5ms. Your checkout p99 *drops*, because failed requests are fast, and your latency alert stays green while every checkout is broken. The fix is to compute latency only over the requests you'd count as "good" in your SLI, and let the *errors* signal catch the fast failures. The two signals are complementary: errors catch fast failures, latency catches slow successes, and together they define "good." A "good" checkout is one that **both** succeeded **and** was fast — anything else burns budget.

### The four kinds of SLI, and picking the right ones

Not every service's "good" looks the same, and a senior picks the SLI *type* that actually reflects what the user cares about. There are four common shapes, and most journeys need two or three of them rather than one.

**Availability** SLIs ask "did the request succeed?" — the good/total ratio of non-error responses. This is the default for any request-driven service and it's the first SLI you should define for checkout: the fraction of checkout attempts that didn't 5xx. **Latency** SLIs ask "was it fast enough?" — the fraction of requests served under a threshold (not the percentile itself, but the *ratio* of requests beating it, so it composes with availability into one budget). For checkout we folded availability and latency into a single "success AND under 1s" SLI, which is the cleanest pattern when both matter equally.

**Quality** SLIs ask "was the response *correct and complete*, even if it was a 200?" — these catch the silent failures that availability misses. A search service that returns 200 with an empty result set when it should have found matches is *available* but *broken*; a quality SLI ("fraction of searches that returned a non-degraded result") catches it. ShopFast's product page, which assembles data from five services, has a natural quality SLI: "fraction of page loads that rendered with full data" versus a degraded version missing reviews or recommendations. **Freshness** SLIs ask "how stale is the data?" — essential for any asynchronous or [event-driven](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) pipeline where there's no synchronous request to measure. ShopFast's inventory-sync pipeline, which propagates stock counts via events, has no latency-of-a-request to track; its SLI is "fraction of stock updates reflected in the search index within 60 seconds." This is the right SLI shape for the eventually-consistent reads we covered in [data consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice): the user-facing promise isn't "instant," it's "fresh within a bound," and freshness SLIs measure exactly that bound.

The senior move is to **pick the smallest set of SLIs that captures real user pain and no more.** Three or four SLIs per journey — availability, latency, maybe a quality SLI and a freshness SLI for the async parts — is plenty. A dozen SLIs per service is a sign you're measuring causes, not symptoms, and you'll drown in budgets you can't reason about. Each SLI you define is a budget you have to track and an alert you have to tune; spend that cost only where users actually feel the difference.

### Rolling window vs. calendar window: a quiet but real choice

One decision hides inside every SLO and changes its behavior more than people expect: is the time window **rolling** (the last 28 days, continuously sliding) or **calendar-aligned** (this calendar month, resetting on the 1st)? They sound interchangeable; they're not.

A **rolling window** is what engineers usually want for alerting and budget tracking. "99.9% over the last 28 days" means an incident's impact on your budget *gradually ages out* over 28 days — a bad day three weeks ago slowly stops counting, and you don't get a sudden budget reset that masks a chronic problem. Rolling windows give a smooth, honest picture and are the right default for burn-rate alerts. The cost is that they're slightly harder to explain to non-engineers ("why did our budget change overnight when nothing happened?" — because a good day 28 days ago rolled off).

A **calendar window** ("99.9% this month") resets to a full budget on the 1st, which matches how SLAs and business reporting work — customers and finance think in calendar months. The danger is the reset: an incident on the 30th and an incident on the 1st look identical to a calendar-aligned budget even though they're a day apart, and a chronic low-grade problem can hide behind monthly resets, never accumulating enough in any single month to trigger action. The pragmatic answer most teams land on: **track and alert on a rolling window internally (28 days is common, because it's exactly four weeks and so isn't skewed by which weekdays a calendar month happens to contain), and report the calendar-month number to customers for the SLA.** The 28-day choice in the OpenSLO spec above is deliberate for exactly this reason — four clean weeks of traffic, no weekend-vs-weekday skew.

## Defining the ShopFast checkout SLI in code

Enough definitions. Let's make the checkout SLI real. We'll instrument the order service to emit a histogram of checkout latency labelled by outcome, then build the SLI as a PromQL ratio. The instrumentation, in Go with the Prometheus client, looks like this — note that we record *both* the duration and the outcome on the same series so we can split good from bad.

```go
// internal/metrics/checkout.go
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Histogram buckets chosen around our 1s SLO threshold so we can read
// "fraction under 1s" straight off the bucket boundaries.
var CheckoutDuration = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "checkout_request_duration_seconds",
		Help:    "End-to-end checkout latency, labelled by outcome.",
		Buckets: []float64{0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0},
	},
	[]string{"outcome"}, // "success" | "client_error" | "server_error"
)

func ObserveCheckout(seconds float64, outcome string) {
	CheckoutDuration.WithLabelValues(outcome).Observe(seconds)
}
```

The critical design choice is in the bucket boundaries: we put a bucket edge *exactly at our SLO threshold of 1.0 seconds*. That lets us read "fraction of checkouts under 1s" directly off the cumulative histogram with no estimation error, which matters because our whole SLO hinges on that threshold. (If your SLO threshold doesn't line up with a bucket boundary, your SLI is only as precise as your nearest bucket — a classic, silent source of error.) We also exclude `client_error` from the bad count: a 400 because the user's card was declined is the *user's* fault, not ours, so it shouldn't burn our budget. This is an opinionated but standard call; define it explicitly so the whole team agrees.

Now the SLI as a Prometheus query. The "good" events are successful checkouts that completed in under 1 second; the "total" is all checkouts we hold ourselves responsible for (successes plus server errors, excluding client errors).

```promql
# checkout_sli: fraction of checkouts that were good (success AND < 1s)
# over the last 5 minutes. This is our raw SLI.
(
  # good = successes that landed in the <= 1.0s buckets
  sum(rate(checkout_request_duration_seconds_bucket{outcome="success", le="1.0"}[5m]))
)
/
(
  # total = everything we own: successes + server errors (any latency)
  sum(rate(checkout_request_duration_seconds_count{outcome="success"}[5m]))
  +
  sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[5m]))
)
```

This single ratio is the heartbeat of ShopFast checkout reliability. When it reads 0.9995, 99.95% of the checkouts we're responsible for were both successful and fast in the last five minutes. When it dips, users are being hurt. Everything from here — the SLO, the error budget, the alerts — is built on this number.

### Recording rules: don't compute the SLI on every dashboard load

Computing that ratio across millions of series on every dashboard refresh and every alert evaluation is wasteful and slow. Prometheus's answer is **recording rules**: you compute the expensive expression once per evaluation interval and store the result as a new, cheap-to-query time series. Here's the SLI as a recording rule, plus the per-window good/bad rates the burn-rate alerts will need.

```yaml
# prometheus/rules/checkout_sli.yml
groups:
  - name: checkout_sli
    interval: 30s
    rules:
      # The raw 5m good-ratio SLI, recorded once so dashboards & alerts reuse it.
      - record: checkout:sli_good_ratio:5m
        expr: |
          sum(rate(checkout_request_duration_seconds_bucket{outcome="success", le="1.0"}[5m]))
          /
          (
            sum(rate(checkout_request_duration_seconds_count{outcome="success"}[5m]))
            + sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[5m]))
          )
      # The error ratio (1 - good) over several windows for burn-rate alerting.
      - record: checkout:error_ratio:5m
        expr: 1 - checkout:sli_good_ratio:5m
      - record: checkout:error_ratio:1h
        expr: |
          1 - (
            sum(rate(checkout_request_duration_seconds_bucket{outcome="success", le="1.0"}[1h]))
            /
            (
              sum(rate(checkout_request_duration_seconds_count{outcome="success"}[1h]))
              + sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[1h]))
            )
          )
      - record: checkout:error_ratio:6h
        expr: |
          1 - (
            sum(rate(checkout_request_duration_seconds_bucket{outcome="success", le="1.0"}[6h]))
            /
            (
              sum(rate(checkout_request_duration_seconds_count{outcome="success"}[6h]))
              + sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[6h]))
            )
          )
```

Now `checkout:error_ratio:1h` and `checkout:error_ratio:6h` are pre-computed series the alert rules can reference cheaply. This pattern — record once, alert and dashboard off the recorded series — is the foundation of every production SLO setup. It also means your alert math and your dashboard math are *literally the same expression*, so what you alerted on is exactly what you'll see when you investigate.

## Error budgets: turning the SLO into a currency you can spend

Here is the idea that reorganizes how an engineering org thinks about reliability. An SLO of 99.9% says, equivalently, that you are *allowed* to be bad 0.1% of the time. That 0.1% is not a failure to be ashamed of — it is a **budget**, an allowance you get to spend. Spend it on deploys, on experiments, on the occasional self-inflicted incident from shipping fast. As long as you stay within budget, you're meeting your promise, and the budget *should be used* — an unused error budget means you're being too conservative and shipping too slowly. Run out of budget, and you stop shipping risky changes and pour effort into reliability until the budget recovers.

The error budget is the **inverse of the SLO**, and the first thing to do with any SLO is convert it into a concrete amount of allowable badness in human units — minutes per month — because "0.1%" doesn't viscerally communicate "you may be down for this long."

#### Worked example: error budget from an SLO

Take ShopFast's checkout SLO: 99.9% over a 30-day window. The error budget is `1 - 0.999 = 0.001`, or 0.1% of the time. Now convert to minutes.

A 30-day month is `30 × 24 × 60 = 43,200` minutes. The budget is `0.001 × 43,200 = 43.2` minutes — that is **43 minutes and 12 seconds** of "bad" per month. That's the whole allowance. If checkout is fully broken for 43 minutes in a month, you've spent the entire budget and you're exactly at your SLO. One more bad minute and you've breached it.

Now feel how a tighter SLO bites. Bump the target to 99.95% and the budget *halves* to 21 minutes 36 seconds. Go to 99.99% ("four nines") and it collapses to `0.0001 × 43,200 = 4.32` minutes — **4 minutes 19 seconds per month**. And five nines, 99.999%, leaves you `25.9` seconds per month. Each extra nine cuts the budget by 10×, and the cost of *achieving* each nine roughly multiplies. This is why "we want five nines" is almost always the wrong answer for a checkout flow: 26 seconds of badness per month means a single bad deploy blows your entire annual budget in one afternoon, and the engineering cost to never have a bad deploy is astronomical. Below is a reference table you'll want pinned somewhere.

| SLO target | Budget (% bad) | Per 30 days | Per year |
|---|---|---|---|
| 99% | 1% | 7h 12m | 3d 15.6h |
| 99.5% | 0.5% | 3h 36m | 1d 19.8h |
| 99.9% ("three nines") | 0.1% | 43m 12s | 8h 45m |
| 99.95% | 0.05% | 21m 36s | 4h 23m |
| 99.99% ("four nines") | 0.01% | 4m 19s | 52m 35s |
| 99.999% ("five nines") | 0.001% | 25.9s | 5m 15s |

The reason error budgets are revolutionary is *organizational*, not technical. They **align the incentives of development and operations**, which have been at war since the dawn of software. Dev wants to ship features fast; ops wants stability and resists change. The error budget dissolves the conflict by making it a single shared number with a shared rule: *while there's budget, dev ships freely and ops doesn't block them; when the budget's exhausted, everyone — dev included — works on reliability until it recovers.* Reliability stops being a vague virtue ops nags about and becomes a quantity both sides watch on the same dashboard. Now the argument "can we ship this risky change?" has a data-driven answer: "we have 31 minutes of budget left this month — yes, with a canary," or "we're 8 minutes over — no, we're frozen."

### The error-budget policy: write down what happens at zero

An error budget is toothless without a *policy* — a pre-agreed, written-down set of actions triggered by budget levels, decided *before* an incident when everyone is calm. Negotiating "should we freeze?" mid-incident at 3am, with a VP on the call, never goes well. Write it once:

```yaml
# error-budget-policy.yaml — agreed by eng + product, reviewed quarterly
slo:
  journey: checkout
  objective: 99.9          # percent of good checkouts
  window_days: 30          # rolling
  budget_minutes: 43.2     # 0.1% of 30 days

policy:
  # Budget remaining is checked daily and gates the team's actions.
  - when: budget_remaining > 50%
    state: green
    actions:
      - "Ship freely. Canary + auto-rollback required for risky changes."
      - "Run a planned chaos experiment this week if none in 2 weeks."

  - when: budget_remaining between 10% and 50%
    state: yellow
    actions:
      - "Ship, but every risky change needs a named reviewer + rollback plan."
      - "No non-essential migrations or infra changes."

  - when: budget_remaining < 10%
    state: red
    actions:
      - "Feature freeze on the checkout journey. Reliability work only."
      - "Every deploy must reduce risk or fix a known reliability issue."
      - "Daily budget review until back above 25%."

  - when: budget_remaining <= 0%
    state: exhausted
    actions:
      - "Hard freeze. Page eng lead. Open a reliability incident."
      - "Post-mortem required for what consumed the budget."
```

The magic of writing it down is that the freeze is no longer a fight. When the budget hits zero, the freeze isn't ops being difficult — it's a policy the *whole team agreed to*, including the product manager who signed off when the budget was healthy. The error budget converts an emotional, political argument into a mechanical rule, and that is its real power.

## Alerting on symptoms, not causes

Now we can solve the 3am noise storm from the intro precisely. The root cause of that disaster was a single bad principle: **alerting on causes**. Every alert that woke the engineer — high CPU, memory pressure, disk I/O latency — was a *cause*: an internal condition that *might* lead to user pain, or might be completely benign. A box at 90% CPU during a traffic spike is doing its job. A pod that restarted and rejoined cleanly hurt nobody. Paging on causes means paging on a vast, noisy set of conditions, the overwhelming majority of which are harmless, which trains the on-call to ignore alerts — and then the one cause that *does* matter gets ignored too.

The fix, stated as a single rule SRE has converged on industry-wide: **page on symptoms, not causes.** A symptom is *user-facing pain you can measure* — your journey SLI burning through its error budget. A cause is *anything inside the system that might explain that pain.* You **page** on symptoms, because a symptom means a real human is being hurt right now and the page is justified. You **do not page** on causes; you put them on dashboards and in tickets, and you consult them *after* a symptom alert fires, to find out *why*.

![A before and after comparison contrasting noisy cause-based alerting with quiet symptom-based burn-rate alerting](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-4.webp)

The before/after above is the whole transformation. On the left, cause-based alerting fires fifty times a day, mostly on harmless conditions, burying the one real signal and training the on-call to mute everything. On the right, symptom-based alerting fires only when the checkout SLO is actually burning — two pages a week, each one a real user-facing problem — and *then* the on-call uses the cause dashboards (CPU, pool depth, the four golden signals on each service) to drill down and find why. The causes didn't disappear; they got demoted from "wakes a human" to "informs a human who's already awake because users are hurting."

The decision matrix below is the test to apply to every candidate alert. For each condition, ask one question: *is a user being hurt right now?* If yes, page. If a user *will* be hurt soon but isn't yet (disk filling, days of runway left), open a ticket. If no user is hurt and none will be soon, it belongs on a dashboard or in a log, not in anyone's pager.

![A decision matrix testing several conditions for whether a user is hurt now and whether each should page, ticket, or stay on a dashboard](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-5.webp)

There is one well-known exception worth naming: **"will hurt soon with high certainty"** symptoms are sometimes worth a *page*, not just a ticket — the classic is "at the current fill rate, the disk runs out of space in 90 minutes." That's a cause, but it's a cause with a near-certain, near-term, user-facing consequence and a clear action, so a page is defensible. The discipline is that this is a *narrow, deliberate* exception, not a license to page on every cause "just in case." When in doubt, the answer is: symptoms page, causes inform.

### Why not just alert when the SLI crosses a threshold?

The naive symptom alert is: "page when `checkout:sli_good_ratio:5m < 0.999`." This is better than cause alerting but has two opposite failure modes, and resolving them is what burn-rate alerting is for.

If you alert on a *short* window (5 minutes), you get **noise**: every brief, self-healing blip — a 30-second dependency hiccup, a single bad deploy that auto-rolled-back — drops the 5-minute ratio below the line and pages you, even though no meaningful budget was spent. If you alert on a *long* window (the full 30 days), you get **slow detection**: a total outage might burn for an hour before the 30-day average drops below 99.9%, because one hour of badness barely moves a 30-day average. Short window = sensitive but noisy. Long window = stable but slow. You cannot win with a single threshold on a single window.

## Burn-rate alerting: the technique that fixes both

The insight that resolves the tension is to alert not on the *level* of the SLI but on the **rate at which you're burning the error budget** — and to do it over *multiple windows at once*. This is **multi-window, multi-burn-rate alerting**, the recommendation from the SRE Workbook, and it's the single most important alerting technique in this post.

**Burn rate** is how fast you're consuming the budget relative to the rate that would exactly exhaust it over the SLO window. A burn rate of **1×** means you're spending budget at exactly the pace that uses it all up precisely at the end of the 30-day window — sustainable, the budget lasts exactly as designed. A burn rate of **2×** means you're spending twice as fast; the 30-day budget will be gone in 15 days. A burn rate of **14.4×** is the magic number from the SRE Workbook: it exhausts the *entire* 30-day budget in `30 / 14.4 ≈ 2.08` days, or equivalently, burns **2% of the 30-day budget in a single hour** (`1h / (30d × 24h) × 14.4 = 0.02`). Concretely, burn rate is just the ratio of your current error rate to your budgeted error rate: if your SLO allows 0.1% errors and you're currently seeing 1.44% errors, your burn rate is `0.0144 / 0.001 = 14.4×`.

#### Worked example: the 14.4× fast-burn page

ShopFast's checkout SLO is 99.9% over 30 days, so the budgeted error rate is 0.1% and the total budget is 43.2 minutes. Suppose the payment provider degrades and 1.5% of checkouts start failing. Burn rate = `0.015 / 0.001 = 15×` — just past the 14.4× threshold.

How long until the alert *should* fire, and how much budget will be gone? At 14.4×, you burn 2% of the 30-day budget per hour. Two percent of 43.2 minutes is `0.02 × 43.2 = 0.86` minutes — under a minute of budget per hour of fast burn. That feels small, but the *point* of the fast-burn alert is to catch this *early*, before it compounds. The SRE Workbook pairs the 14.4× burn rate with a **1-hour long window and a 5-minute short window**, and the alert fires when *both* windows show the high burn. The 1-hour window provides confidence (this isn't a 90-second blip), and the 5-minute window provides *speed* (once the 1-hour window is elevated, the alert resolves quickly when the problem stops, so it doesn't keep paging after you've fixed it). Result: this 15× burn pages you in roughly 2 to 5 minutes, having spent well under a minute of budget — early enough to act, certain enough not to be noise.

The dual structure is the whole trick: a **fast-burn** alert with a high burn rate over short windows catches sudden, severe problems in minutes and **pages**; a **slow-burn** alert with a low burn rate over long windows catches gradual leaks over hours and **opens a ticket** (no 3am page for a slow leak — fix it during business hours). The matrix below is the standard two-tier configuration; memorize the shape and tune the numbers to your SLO.

![A matrix contrasting fast-burn and slow-burn alert tiers across burn rate, window lengths, budget consumed, time to fire, and severity](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-6.webp)

Here's the fast-burn page as a real Prometheus alert rule, referencing the recorded error-ratio series from earlier. The `and` between the long and short windows is the multi-window logic — both must be elevated for the alert to fire.

```yaml
# prometheus/rules/checkout_burn_alerts.yml
groups:
  - name: checkout_burn_rate
    rules:
      # FAST BURN: 14.4x over 1h, confirmed by 5m. Pages. Catches sudden outages.
      - alert: CheckoutErrorBudgetFastBurn
        expr: |
          checkout:error_ratio:1h > (14.4 * 0.001)
          and
          checkout:error_ratio:5m > (14.4 * 0.001)
        labels:
          severity: page
          journey: checkout
        annotations:
          summary: "Checkout burning error budget 14.4x — paging."
          description: >
            Checkout error rate is >1.44% over both 1h and 5m windows.
            At this rate the 30-day budget is gone in ~2 days.
            Runbook: https://wiki.shopfast.io/runbooks/checkout-burn
          dashboard: "https://grafana.shopfast.io/d/checkout-slo"

      # SLOW BURN: 1x over 6h, confirmed by 30m. Ticket only. Catches slow leaks.
      - alert: CheckoutErrorBudgetSlowBurn
        expr: |
          checkout:error_ratio:6h > (1 * 0.001)
          and
          checkout:error_ratio:30m > (1 * 0.001)
        labels:
          severity: ticket
          journey: checkout
        annotations:
          summary: "Checkout slowly leaking error budget (1x over 6h)."
          description: >
            Sustained ~0.1%+ error rate over 6h. Not urgent, but if it
            continues the monthly budget will run out. Investigate in hours.
          dashboard: "https://grafana.shopfast.io/d/checkout-slo"
```

Notice the `severity` label: `page` routes to PagerDuty and wakes someone; `ticket` routes to a queue someone triages during business hours. That routing — driven by severity, driven by burn rate, driven by symptoms — is how you get woken *only when it matters*. The production-grade configuration from the SRE Workbook actually uses three or four tiers (e.g., 14.4× / 6× / 3× / 1× over 1h / 6h / 1d / 3d windows) so that the *severity scales with the speed of the burn*, but the two-tier version above captures the essential idea and is enough for most teams.

#### Worked example: noisy alert fires 50×/day — tune it

Suppose a team has an old threshold alert: "page when 5-minute checkout error ratio > 0.1%." It fires roughly 50 times a day, almost always self-resolving within two minutes, and the on-call has started ignoring it. Walk through the fix.

First, *quantify the noise*: 50 pages/day, of which (say) 49 self-resolve in under three minutes with zero customer reports. The signal-to-noise ratio is 1:49 — catastrophic. The on-call's trust is gone, which is the real damage.

Now replace it with the multi-window fast-burn rule above. What happens to those 49 blips? A 90-second spike to 0.5% errors produces a burn rate of `0.005 / 0.001 = 5×` for 90 seconds. The 1-hour window barely moves — 90 seconds of 5× burn raises the 1-hour error ratio by roughly `(90s / 3600s) × 0.005 ≈ 0.000125`, nowhere near the `14.4 × 0.001 = 0.0144` threshold. So the 1-hour window stays green, the `and` fails, and **no page fires**. The 49 blips vanish. The one real outage — sustained 1.5% errors for ten minutes — drives the 1-hour window past the threshold within a few minutes and pages once. New signal-to-noise: roughly 1:0. The on-call gets one page a day or fewer, and *trusts every one of them*. The measurable win: pages/day from ~50 to ~1, mean-time-to-acknowledge from "ignored" to under two minutes, and — the metric that actually matters — zero missed incidents because nobody's muting the pipeline anymore.

## The dependency problem: your journey availability is a product

Now the part that catches teams who set per-service SLOs and assume they compose for free. They don't. **A journey's availability is the *product* of the availabilities of its critical dependencies** — and products of numbers below 1 only ever get smaller. This is the composed-availability math, and it has sharp consequences.

The reasoning is simple. For a checkout to succeed, the cart service *and* the order service *and* the inventory service *and* the payment service all have to work (assuming each is on the critical path — a dependency you can't degrade gracefully around). If they fail independently, the probability that *all* of them work is the product of their individual availabilities. Multiply four numbers each just below 1 and you get a number meaningfully further below 1.

#### Worked example: five services at 99.95% compose to 99.75%

ShopFast's checkout journey has five services on its critical path, each with its own SLO of 99.95% availability. Naively you'd hope the journey is also around 99.95%. It is not.

Composed availability = `0.9995^5`. Compute it: `0.9995^2 = 0.99900025`, `^4 ≈ 0.998001`, `^5 ≈ 0.99750`. So the journey is **99.75% available** — worse than every single one of its dependencies. Translate to budget: 99.75% over 30 days is `0.0025 × 43,200 = 108` minutes of allowed badness, versus the 21.6 minutes each individual service gets at 99.95%. The journey's users experience **five times more downtime than any one service's SLO suggests**, purely from composition.

Now flip the question the way a senior does: if the *journey* needs to hit 99.9% (43.2 min/month), how good does each of five critical dependencies need to be? Solve `x^5 = 0.999`, so `x = 0.999^(1/5) ≈ 0.99980` — each service needs **99.98%**, *tighter* than the journey target, to leave room for the others. This is the core lesson: **per-service SLOs must be tighter than the journey SLO they roll up into**, and you have to do the multiplication to know how much tighter. Set every service to "99.9% is fine" and your five-service journey lands around 99.5% — a budget of 216 minutes/month, five times what you promised.

![A branching graph showing the checkout journey fanning out to four critical dependencies and composing into a lower overall availability](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-7.webp)

This is also why you need **dependency SLOs and explicit error-budget allocation**. The journey owner (the checkout team) holds the journey SLO, but it depends on services owned by other teams. So the checkout team negotiates an SLO *with each dependency team* — "we need payment to be 99.98% available on the checkout path" — and that becomes the payment team's commitment, with its own budget and alerts. The journey's 43.2-minute budget is *allocated* across dependencies: maybe the payment service (most fragile, talks to an external provider) gets the largest slice, the cart service a small one. When a dependency overspends *its* allocated slice, that's a signal to the dependency team, not a mystery in the journey owner's dashboard. This is the SLO equivalent of the [database-per-service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) ownership boundary: each team owns and defends its own reliability commitment, and the commitments compose into the journey's.

### Critical path vs. graceful degradation changes the math

A crucial refinement that the naive product hides: the multiplication only applies to dependencies on the **critical path** — ones whose failure *fails the journey*. If a dependency can be **degraded around**, it drops out of the product. ShopFast's recommendation service is called during checkout to show "customers also bought," but if it's down, checkout proceeds without recommendations. So recommendations is *not* on the critical path, contributes *nothing* to the composed availability, and needs no SLO that gates checkout. This is exactly the [graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) pattern paying off in the SLO math: every dependency you can make optional is a dependency that *stops dragging your journey availability down*. The architectural move — make the payment call critical but the recommendation call optional, with a timeout and a fallback — directly improves your composed SLO. Reliability is designed in, not just measured.

This is also the practical answer to "why is distributed availability so hard?" — and it's the same family of trade-off as the [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc): every additional service on the critical path is another factor below 1 in the product and another component that can partition away. The fewer hard dependencies on the critical path, the higher your achievable availability. Sometimes the highest-leverage reliability work isn't making a service more reliable — it's *removing it from the critical path*.

## An incident, end to end: the budget burns, the page fires

Let's run the whole machine on one incident, because seeing the parts work together is what makes them click. ShopFast's payment provider — an external PSP — starts degrading at 03:00, its p99 climbing from 200ms to 4 seconds. Here's the timeline, and notice how each reliability primitive plays its part.

![A six-event timeline of a payment slowdown burning the checkout error budget, triggering a fast-burn page, and recovering after mitigation](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-8.webp)

At **T+0**, payment p99 spikes to 4 seconds. The checkout service's [timeout on the payment call](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) is 2 seconds, so checkouts that hit the slow path now *time out and fail* — the timeout converts "slow" into "fast failure," which is correct, but those failures are server errors that count against the SLI.

At **T+2m**, the checkout SLI drops to 96% — a 4% error rate, a burn rate of `0.04 / 0.001 = 40×`. The 5-minute window plummets immediately; the 1-hour window starts climbing.

At **T+5m**, the 1-hour window crosses `14.4 × 0.001`, the `and` with the 5-minute window is satisfied, and **the fast-burn alert fires a page.** This is the *only* page in the whole incident. Note what did *not* page: the payment service's high latency (a cause — on the dashboard), the order service's blocked threads (a cause), the elevated CPU on the gateway (a cause). One symptom, one page.

At **T+9m**, the on-call — awake, with a trusted page and a runbook link — opens the SLO dashboard, sees latency and errors spiking, drills into a [distributed trace](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) of a failing checkout, and sees the 4-second payment span lit up red. Cause identified in minutes because the dashboard funnels page → overview → signals → trace. They apply the runbook mitigation: trip the payment circuit breaker manually and shed the synchronous payment retries (the retries were *amplifying* load on an already-slow dependency — the classic [retry-storm](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) failure mode). Checkout switches to its degraded path: accept the order, queue the payment for async retry, confirm to the user.

At **T+18m**, the SLI recovers to 99.95% — the degraded path keeps checkouts succeeding while payment is bypassed. At **T+30m**, the budget review shows ~23% of the monthly budget spent in this one incident (18 minutes of badness against a 43.2-minute budget ≈ 42% — close enough to "a big chunk, not catastrophic"). The error-budget policy moves the team from green to yellow: ship carefully for the rest of the month. The incident is contained, *one* human was woken, the cause was found in minutes, and the budget tells everyone exactly how much risk-appetite remains.

That is the entire value proposition of this post in one incident: symptom-based burn-rate alerting woke exactly one person at exactly the right time, the dashboard drove them to the cause fast, the resilience patterns gave them a mitigation, and the error budget quantified the cost. Compare that to the intro's 14-pages-then-mute-then-miss-the-real-one disaster. Same underlying outage; opposite outcome.

## Dashboards that drive you to the cause

A page is the start of an investigation, not the end, and a good dashboard turns a 30-minute hunt into a 3-minute one. The structure that works, used at most mature SRE shops, is a **funnel**: a top-level service-overview that the page links to, the four golden signals one click down, and drill-down to traces one click below that.

![A branching graph showing a burn-rate page leading from the SLO overview through the four golden signals down to a specific slow trace](/imgs/blogs/slos-golden-signals-and-alerting-for-microservices-9.webp)

The **service overview** is the first screen: the journey SLI as a single big number, the SLO target as a line, the error budget remaining as a gauge, and the burn rate as a sparkline. One look tells the on-call "how bad, how fast, how much budget left." Below it, the **four golden signals** for the journey and each critical dependency, so a glance shows *which signal* is off and *which service* is the source. And each panel links to **traces** filtered to the failing requests, so one more click lands on the exact slow span. The dashboard mirrors the page → overview → signals → trace path from the incident above. Here are the core PromQL queries each golden-signal panel runs:

```promql
# --- LATENCY: p99 of *successful* checkouts (the latency that counts) ---
histogram_quantile(0.99,
  sum(rate(checkout_request_duration_seconds_bucket{outcome="success"}[5m])) by (le)
)

# --- TRAFFIC: checkout requests per second, by outcome ---
sum(rate(checkout_request_duration_seconds_count[5m])) by (outcome)

# --- ERRORS: server-error rate as a fraction of owned traffic ---
sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[5m]))
/
(
  sum(rate(checkout_request_duration_seconds_count{outcome="success"}[5m]))
  + sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[5m]))
)

# --- SATURATION: DB connection-pool occupancy on the order service ---
max(db_pool_in_use_connections{service="order"})
/
max(db_pool_max_connections{service="order"})

# --- ERROR BUDGET REMAINING over the 30d window (1 = full budget) ---
1 - (
  (1 - avg_over_time(checkout:sli_good_ratio:5m[30d]))   # actual badness
  / 0.001                                                # budgeted badness (0.1%)
)
```

That last query is worth dwelling on: it computes **how much of the 30-day budget remains**, normalized so 1.0 is "full budget untouched" and 0.0 is "exhausted." It's the number the error-budget policy gates on and the gauge the overview shows. (In a precise implementation you'd compute actual bad-event count over good+bad-event count over the window rather than averaging the 5m ratio, but the shape is what matters here.) Render this on the overview and the freeze decision becomes a glance, not a debate.

## SLOs as code: OpenSLO and the GitOps loop

Everything above — the SLI query, the SLO target, the burn-rate alert thresholds — should live in version control, reviewed in pull requests, and generated rather than hand-written. Hand-writing burn-rate alert YAML for forty services, with four windows each, is how you get inconsistency and silent gaps. The emerging standard for declaring SLOs portably is **OpenSLO**, a YAML spec you can feed to a generator (like Sloth or OpenSLO's own tooling) that emits the Prometheus recording and alerting rules for you.

```yaml
# slo/checkout.openslo.yaml — the single source of truth for the SLO
apiVersion: openslo/v1
kind: SLO
metadata:
  name: checkout-availability
  displayName: ShopFast Checkout Availability
spec:
  service: checkout
  description: 99.9% of checkouts succeed in under 1s over 28 days.
  budgetingMethod: Occurrences        # count good/total events, not time-slices
  timeWindow:
    - duration: 28d
      isRolling: true
  objectives:
    - displayName: Fast successful checkouts
      target: 0.999                   # the SLO: 99.9%
      # The SLI: ratio of good events to total events.
      indicator:
        metadata:
          name: checkout-good-ratio
        spec:
          ratioMetric:
            good:
              metricSource:
                type: Prometheus
                spec:
                  query: |
                    sum(rate(checkout_request_duration_seconds_bucket{outcome="success", le="1.0"}[{{.window}}]))
            total:
              metricSource:
                type: Prometheus
                spec:
                  query: |
                    sum(rate(checkout_request_duration_seconds_count{outcome="success"}[{{.window}}]))
                    + sum(rate(checkout_request_duration_seconds_count{outcome="server_error"}[{{.window}}]))
  alertPolicies:
    - alertPolicyRef: fast-and-slow-burn   # generator expands to the multi-window rules
```

The win here is uniformity and review. The SLO target, the SLI definition, and the alert policy are one reviewed artifact; a change to the checkout SLO is a pull request with a diff a human reads, not a midnight edit to a YAML file nobody owns. The generator guarantees every service's burn-rate alerts use the same battle-tested windows, so you never ship a service with a subtly-broken alert. And because it's in Git, you get history: "we loosened checkout from 99.95% to 99.9% on this date, here's the PR and the reasoning." For a fleet of forty services, SLOs-as-code is the difference between consistent reliability engineering and forty snowflakes.

## Trade-offs: the decisions you actually have to make

No reliability decision is free; each buys something and costs something. Here are the ones that matter, with the cost named explicitly.

| Decision | What you gain | What you pay | When it wins |
|---|---|---|---|
| Tight SLO (99.99%) | Strong reliability promise, happy users | Tiny budget (4 min/mo), slow shipping, high eng cost | Payments, auth — where failure is catastrophic |
| Loose SLO (99.5%) | Big budget (3.6h/mo), fast shipping | Visible downtime, weaker promise | Internal tools, async batch jobs, new services |
| Multi-window burn-rate | Fast detection + low noise + trusted pages | More complex rules, needs recording rules | Any user-facing journey SLO |
| Single-threshold alert | Trivial to write and understand | Either noisy (short window) or slow (long) | Throwaway prototypes only |
| Per-journey SLOs | Measures real user pain, business-aligned | Need to instrument at the edge/BFF | Always, for the targets you defend |
| Per-service SLOs | Team accountability, dependency budgets | Don't reflect user experience alone | Internal accountability, not the headline metric |
| Symptom-only paging | Trusted pages, no fatigue | Slightly slower to specific cause | Always — causes go on dashboards |
| Page on near-certain causes | Catch slow disasters (disk fill) early | Risk of cause-alert creep | Narrow, deliberate exceptions only |

The decision that trips teams up most is the **SLO target itself**, so it deserves its own principle: **set the SLO to match user expectation, not 100%.** Users do not perceive the difference between 99.9% and 99.99% for a shopping cart — both feel "basically always up." But the *cost* difference between them is enormous: an extra nine roughly multiplies engineering effort and divides your shipping velocity. The SRE maxim is that **100% is the wrong reliability target for almost everything**, because the marginal user-perceived value of each nine drops sharply while the cost rises sharply, and a target of 100% leaves *zero* error budget — meaning *any* change is a budget violation, which freezes you permanently. The right target is the lowest one your users won't notice, which leaves you the most budget to ship.

### Optimization: tuning to cut pages without going blind

The production-grade move is to *measure* your alerting and tune it like any other system. Pull the numbers: how many pages per week, what fraction were actionable (led to a fix), what fraction self-resolved before anyone acted, and what's the on-call's MTTA. A healthy on-call rotation runs at **roughly 2 or fewer pages per on-call shift**, with **over ~90% of pages actionable**. If you're at 50 pages/day with 2% actionable (the intro's state), you're not alerting, you're spamming.

The concrete tuning levers, with their measurable effects: **(1)** Move cause alerts off the pager — that alone took the intro team from 50 to ~1 page/day. **(2)** Add the long-window confirmation (`and` with the 1h window) — kills the self-resolving blips, typically a 5–10× noise reduction with zero loss of real-incident coverage. **(3)** Right-size the SLO — if your "99.99%" SLO is causing weekly pages for blips users don't notice, dropping to 99.9% multiplies your budget by 10 and your page rate drops accordingly, while users feel nothing. **(4)** De-flap with `for:` durations and grouping so a single incident produces one page, not forty. The measurable win across all four: pages/week down 10–50×, actionable fraction up from single digits to >90%, MTTA down because every page is now trusted, and — the real prize — *zero missed incidents*, because nobody mutes a pipeline they trust.

## Stress-testing the design

A senior pressure-tests a design before production does. Run the checkout SLO setup against the hard cases.

**"An incident burns half the budget — page or not?"** Depends entirely on *rate*, which is why burn-rate alerting is right. A slow leak that consumes half the monthly budget over three weeks should *never* page — it should open a ticket and get fixed in business hours (slow-burn tier). A fast burn that consumes half the budget in twenty minutes absolutely pages (fast-burn tier). Same total budget spent; opposite urgency. The burn-rate design encodes this distinction natively, which a level-threshold alert cannot.

**"A noisy alert fires 50×/day"** — covered in the worked example. The fix is structural, not a band-aid: move it from the pager to a dashboard if it's a cause; convert it to multi-window burn-rate if it's a real symptom. Never solve alert fatigue by *raising the threshold* on a cause alert — you'll just miss the real one. Solve it by *changing what you alert on*.

**"A dependency degrades — whose SLO breaks?"** The journey SLO breaks first, visibly, because the journey is the product of its dependencies. *Whose fault* it is depends on the allocation: if the payment service blew through its allocated slice of the budget, that's the payment team's reliability incident, surfaced by *their* dependency SLO. The journey owner sees the symptom and pages; the dependency SLO assigns the cause. This is exactly why you allocate the budget across dependencies — so a degradation has a clear owner instead of a finger-pointing meeting.

**"What breaks at 10× traffic?"** Two things. First, **saturation** — the leading signal — will hit a limit (connection pool, thread pool, CPU) before latency and errors visibly degrade, which is why saturation is the *predictive* signal you watch even though you don't page on it. Second, your SLI denominator grows, so the *same number* of errors becomes a *smaller* ratio; an absolute count that mattered at 1× becomes invisible at 10×. Ratios scale correctly with traffic, which is another reason SLIs are ratios and not counts. Your burn-rate alerts keep working unchanged because they're ratio-based.

**"A service is deployed mid-incident."** A deploy during an active burn either helps (it's the fix) or makes it catastrophically worse. The error-budget policy is the guardrail: if the budget's exhausted, you're in freeze, and the only deploys allowed are reliability fixes. The deploy strategy matters too — a [canary with auto-rollback](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) lets you ship even in yellow state, because a bad change gets caught on 1% of traffic and rolled back before it burns meaningful budget. SLOs and safe deploys reinforce each other: the budget tells you *how much* risk you can take, the canary *limits the blast radius* of each risk.

## On-call hygiene: making the page worth answering

All of this — SLOs, golden signals, burn-rate alerts — exists to serve one human at the end of the chain: the on-call engineer holding the pager. If that person doesn't trust the page, none of the machinery matters. So the last discipline is **on-call hygiene**, the set of practices that keep a page actionable, rare, and survivable.

**Every page must be actionable.** This is the prime directive. When a page fires, the engineer must be able to *do something about it right now*. If a page fires and the correct response is "there's nothing I can do, it'll fix itself," that page should not exist — it should be a ticket or a dashboard panel. The actionability test is brutal but clarifying: for every alert in your pager, ask "if this fires at 3am, what does the on-call *do*?" If the answer is "acknowledge it and go back to sleep," delete it. The fast-burn checkout alert passes this test: there's a runbook, a mitigation (trip the breaker, shed retries), and a clear escalation path.

**Every page needs a runbook.** A runbook is a short, specific document linked directly from the alert (note the `Runbook:` line in the alert's `description` annotation, and the `dashboard:` annotation alongside it) that tells a *half-asleep, possibly-junior* engineer exactly what to check and what to try. Not a textbook — a checklist. "1. Open the checkout SLO dashboard. 2. Which golden signal is off? 3. If errors: which dependency? Check the trace. 4. If payment: trip breaker `checkout-payment-cb` via this command. 5. If that doesn't help in 10 minutes, escalate to the payment team's on-call." The runbook is what lets the page wake *anyone* on the rotation, not just the one engineer who happens to know this system. Writing the runbook is also a forcing function: if you can't write a runbook for an alert, you don't understand the failure well enough to be paging on it.

**No flapping.** An alert that fires, resolves, fires, and resolves repeatedly — *flapping* — is almost as corrosive as constant noise, because it trains the on-call to wait-and-see instead of acting. The cures are built into the alert design: the multi-window `and` requires sustained badness (a flap on the 5-minute window won't fire if the 1-hour window is calm), the `for:` clause requires a condition to persist before firing, and alert *grouping* collapses many related alerts from one incident into a single notification. A single payment outage should produce *one* page, not one page per affected service per minute. Grouping by `journey` or `incident` label is how you get there.

**Sustainable load and good handoffs.** The SRE-recommended ceiling is roughly **two incidents per on-call shift** — enough that the skill stays sharp, few enough that the engineer can actually handle each one and still do project work. Above that, you have a reliability problem *or* an alerting problem, and the error budget plus the page-rate metrics tell you which. Pair this with clean shift handoffs (what's ongoing, what's degraded, what's been silenced and why) and a blameless post-mortem culture, and on-call becomes a rotation people can sustain rather than one that burns them out. The whole point of everything in this post — symptom-based alerting, burn-rate tiers, error budgets — converges here: **the metric that ultimately matters is whether the human holding the pager trusts it and can sleep.** Build for that human, and the reliability follows.

## Case studies

**Google SRE — the origin of golden signals and error budgets.** Google formalized all of this in the SRE book and SRE Workbook, and the most important contribution wasn't technical — it was the *error-budget policy as a treaty between dev and ops*. The rule "while budget remains, the SRE team will not block launches; when it's exhausted, launches freeze" turned reliability from a perpetual argument into a shared number with shared consequences. The four golden signals and the 14.4× / multi-window burn-rate alerting recipe both come directly from these books, and they remain the canonical reference. The deep lesson Google teaches: reliability is a *product feature with a cost*, and the error budget is how you price it.

**A team that cut alert fatigue with burn-rate alerts.** The migration from level-threshold or cause-based alerting to multi-window burn-rate alerting is one of the most reliably-reported wins in the SRE community, and it consistently follows the shape of the worked example above: page volume drops by an order of magnitude or more, the fraction of actionable pages climbs toward 100%, and — the qualitative change that matters most — on-call engineers start *trusting* their pager again. Teams that have written about this transition (across the SRE and observability communities) report the same arc: the old system fired constantly and was ignored; the new system fires rarely and is acted on every time. The lesson: alert *quality* beats alert *quantity*, and burn-rate alerting is the mechanism that buys quality.

**The over-tight-SLO regret.** A recurring, hard-won lesson is the team that sets an aspirational SLO — "we want four nines!" — that's *tighter than users actually need*, and then suffers for it. With a 4-minute monthly budget, every routine deploy and every brief dependency blip threatens the SLO, so the team is either perpetually frozen (can't ship) or perpetually breaching (the SLO is a lie nobody believes). The regret is always the same: they set the target by aspiration rather than by measuring what users actually tolerate, and the over-tight SLO either strangled their velocity or became meaningless. The correction is to *loosen* the SLO to the real user-expectation level, which simultaneously restores shipping velocity and makes the SLO honest again. The lesson: an SLO you can't actually hold while shipping is worse than a looser one you can defend — pick the target your users need, not the one that sounds impressive.

**Amazon and the dependency-availability discipline.** Amazon's well-documented obsession with availability includes the practice of treating each service's dependency graph as a reliability liability and aggressively making dependencies *non-critical* — the graceful-degradation move that drops a service out of the composed-availability product. The lesson reinforces the math above: the cheapest way to raise a journey's availability is often not to make a dependency more reliable, but to make the journey *survive that dependency's failure*, removing it from the critical path entirely.

## When to reach for this (and when not to)

SLOs and burn-rate alerting are not free — they cost instrumentation, recording rules, alert tuning, and the organizational work of agreeing on targets and policies. Reach for the full machinery when:

- **You have a user-facing journey whose reliability matters to the business** — checkout, login, search, the API your customers pay for. These deserve a journey SLO, an error budget, and burn-rate alerts.
- **You're being paged too much or missing real incidents** — the symptoms of cause-based alerting. Migrating to symptom-based burn-rate alerting is the highest-leverage reliability work you can do.
- **Dev and ops are fighting about velocity vs. stability.** The error budget is the treaty that ends the fight.

Don't over-invest when:

- **The service has no users yet or trivial traffic.** An SLI is a ratio; at 50 requests a day, a single error is 2% and your "SLO" is statistical noise. Wait until you have enough traffic for the ratio to mean something (rule of thumb: you need enough events per window that one bad event doesn't dominate — a few thousand per window minimum).
- **It's an internal batch job or async pipeline with no real-time user.** A latency SLO on a nightly job is theater. Track freshness or completion-time SLOs instead, and don't page on them at 3am.
- **You'd be setting an SLO you can't measure.** An SLO you can't compute from real telemetry is a wish, not an objective. Instrument first, set the SLO second.

And the honest senior caveat: **don't set an SLO on a service nobody will defend.** An SLO with no error-budget policy, no owner, and no consequence when it's breached is just a number on a dashboard. The discipline — the policy, the freeze, the burn-rate page — is what makes an SLO real. A number without the discipline is worse than no SLO, because it gives false confidence.

## Key takeaways

1. **SLI is a measurement, SLO is a target, SLA is a contract.** SLI = good/total ratio; SLO = the internal target you defend over a window; SLA = the external promise with penalties. Always order them SLA < SLO, and set SLOs on user-facing journeys, never on CPU.
2. **The four golden signals are latency, traffic, errors, saturation.** Split latency by outcome (good-request latency only), build SLIs from RED, use saturation as the leading indicator you watch but don't page on.
3. **The error budget is the inverse of the SLO, and it's a currency.** 99.9% over 30 days = 43m 12s of allowed badness. Spend it on velocity while it lasts; freeze and fix when it's gone. It aligns dev and ops on one shared number.
4. **Page on symptoms, not causes.** A page means a user is hurt now. Causes (high CPU, a restarted pod) go on dashboards and in tickets, consulted *after* a symptom alert fires.
5. **Use multi-window, multi-burn-rate alerting.** Fast-burn (14.4× over 1h, confirmed by 5m) pages now for sudden outages; slow-burn (1× over 6h) opens a ticket for slow leaks. This beats both alert fatigue and slow detection.
6. **Journey availability is the product of its critical dependencies.** Five services at 99.95% compose to 99.75%. Per-service SLOs must be *tighter* than the journey SLO; allocate the budget across dependencies and make non-critical ones degradable to drop them from the product.
7. **Set the SLO to user expectation, not 100%.** Each extra nine multiplies cost and divides velocity for diminishing user-perceived value. The right target is the lowest one users won't notice, leaving you the most budget to ship.
8. **Write the error-budget policy down before the incident.** A pre-agreed policy turns the freeze from a 3am political fight into a mechanical rule the whole team already signed.
9. **Put SLOs in code and build dashboards that funnel page → overview → signals → trace.** Uniform, reviewed, version-controlled SLOs scale to a fleet; a funnel dashboard turns a 30-minute hunt into a 3-minute one.
10. **Measure your alerting and tune it.** Target ~2 or fewer pages per shift, >90% actionable. The on-call trusting the pager is the metric that matters; everything else serves it.

## Further reading

- *Site Reliability Engineering* (Beyer, Jones, Petoff, Murphy — the Google SRE book), especially the chapters on Service Level Objectives and Monitoring Distributed Systems (the four golden signals).
- *The Site Reliability Workbook* (Beyer et al.), especially "Alerting on SLOs" — the canonical source for multi-window, multi-burn-rate alerting and the 14.4× recipe.
- *Implementing Service Level Objectives* (Alex Hidalgo) — a book-length, practical treatment of SLIs, SLOs, and error budgets.
- The OpenSLO specification and the Sloth SLO-generator project — for declaring SLOs as code and generating Prometheus rules.
- [Distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) — the data layer this judgment layer sits on.
- [Resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) and [handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) — the mechanisms that *defend* the budget.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — why composed distributed availability is fundamentally hard.
- Coming next in this series: [debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production) (chasing the cause after a page) and [deployment strategies: blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) (spending the budget safely).

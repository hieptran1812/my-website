---
title: "Dashboards That Tell the Truth: Designing for Signal Over Noise"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Design dashboards that answer is-the-service-healthy in five seconds and where-is-it-broken in thirty, by building a tiered hierarchy of honest, code-versioned panels instead of a wall of forty no one reads."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "dashboards",
    "grafana",
    "observability",
    "golden-signals",
    "red-method",
    "slo",
    "promql",
    "incident-response",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/dashboards-that-tell-the-truth-1.png"
---

It is 3:14am. Your phone is buzzing against the nightstand, the alert says `SLOErrorBudgetFastBurn`, and you have maybe ninety seconds of clear thought before adrenaline turns your hands clumsy. You open the dashboard. And there it is: forty-eight panels, three rows of them above the fold, every line a different color, a forest of sparklines that all look slightly wrong because at 3am everything looks slightly wrong. You scroll. There are twelve more panels below. Somewhere in this thicket is the one chart that would tell you checkout is failing, but you cannot find it, because the person who built this board built it to show *everything* and so it shows *nothing*. You burn four minutes scanning. The outage burns four minutes with you.

That dashboard had a job, and it failed at the job. The job of a dashboard is not to display data. Data is cheap; Prometheus will happily give you ten thousand time series for free. The job of a dashboard is to **answer a specific question fast** — and the two questions that matter at 3am are "is the service healthy?" and "where is it broken?". A good dashboard answers the first in five seconds and the second in thirty. The forty-eight-panel board answers neither, because nobody ever asked it a question; it was built by accumulation, one panel bolted on after every incident, never a panel removed, until it became a museum of every metric anyone ever cared about and a tool for none of them.

This post is about building dashboards that tell the truth — fast, honestly, and at the right altitude for whoever is reading them at 3am. We will design a **dashboard hierarchy**: a single service-overview board that triages in five seconds, a set of deep-dive boards that localize the fault in thirty, and the capacity and business boards that serve completely different audiences. We will go through the design principles that make a panel readable under stress — eye path, consistent time ranges, colors that mean something, annotations that correlate a regression with the deploy that caused it, percentiles instead of averages, rates instead of raw counters. We will catalog the anti-patterns that make dashboards lie: the wall of sixty panels, the board that glows green while users are down, the truncated y-axis, the dual-axis trick, the average that hides the tail. And we will treat dashboards as **code** — versioned, reviewed, and pruned — because a dashboard built once by hand rots the moment the service changes. Figure 1 shows the hierarchy we are going to build.

![A vertical stack diagram of the dashboard hierarchy showing the service overview at the top, deep-dive subsystem boards below it, then capacity and business dashboards, with a user-journey SLI feeding in and all of them built as reviewed code](/imgs/blogs/dashboards-that-tell-the-truth-1.png)

This is the measure-and-respond joint in the SRE loop. We have spent earlier posts in this series defining reliability — picking [SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain), [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something), and getting [metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right). A dashboard is where all that measurement becomes a decision under pressure. If the SLI is the number and the SLO is the promise, the dashboard is the windshield you steer by. A dirty windshield gets you killed. Let us clean it.

## 1. A dashboard is a tool with a job

Start with the heresy: most dashboards should not exist, and most panels on the dashboards that should exist should be deleted. This sounds extreme until you watch how dashboards actually get used. Instrument your own Grafana — Grafana ships usage analytics in the enterprise tier, and even the open-source version logs dashboard views — and you will find a brutal power law. A handful of boards get opened during every incident. The vast majority get opened once, by their author, the week they were built, and never again. They are not free. Every dead dashboard is a search result the on-call has to skip past at 3am, a query load on your Prometheus, and a lie waiting to happen when the metric it charts gets renamed and the panel silently goes blank.

So before you draw a single panel, answer three questions, and if you cannot answer all three, do not build the dashboard:

1. **What single question does this board answer?** Not "monitoring for the payments service" — that is a topic, not a question. "Is the payments service meeting its SLO, and if not, which dependency is the cause?" is a question. A board with a question has a natural panel budget: the panels that help answer it stay, the rest go.
2. **Who reads it, and in what situation?** The on-call at 3am mid-incident is a different reader from the capacity owner doing a quarterly review, who is different again from the VP looking at a weekly business summary. They need different altitudes, different time ranges, different vocabularies. One board cannot serve all three. The belief that it can — the *single pane of glass* — is the most expensive myth in observability.
3. **What will the reader do as a result?** A dashboard that does not change a decision is wallpaper. The overview's decision is "do I escalate / page someone else / roll back?". The deep-dive's decision is "which subsystem do I dig into?". The capacity board's decision is "do we order more nodes this quarter?". If you cannot name the action, you are decorating, not engineering.

This is the same discipline you apply to alerts. A good alert pages on a symptom a human must act on; a good dashboard panel earns its place by helping a human act faster. Both are governed by the cost of false signal. An alert that cries wolf trains the on-call to ignore it — we cover that in the sibling post on [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf). A dashboard panel that adds noise does the same thing more quietly: it trains the eye to skim, and a skimming eye misses the one chart that mattered.

### The five-second test and the thirty-second test

Here are the two acceptance tests I apply to every dashboard hierarchy, and they are how you know you have succeeded:

- **The five-second test (the overview):** glance at the top of the overview for five seconds. Can you tell whether the service is healthy? If the answer requires reading numbers, comparing two charts, or scrolling, you have failed. Health should read like a traffic light — a glance, a verdict. The way this works in practice is one big SLO/error-budget panel and three or four golden-signal panels arranged so that "all green, budget healthy" versus "something is red" is preattentive — your visual cortex flags it before you consciously read anything.
- **The thirty-second test (the path to the deep-dive):** from the overview, in thirty seconds, can you name *which part* of the service is broken and open the right deep-dive board? This is the triage path. The overview localizes; the deep-dive diagnoses. If localizing takes minutes of squinting, the hierarchy is wrong.

These two tests do all the work. Every design principle in this post — eye path, percentiles, annotations, color discipline, the panel budget — exists to pass them. Keep them in mind as your north star. A dashboard that passes both is worth its weight; a dashboard that passes neither is the forty-eight-panel board that lost you four minutes.

There is a deeper reason the panel budget matters, and it is worth making explicit because it is the principle behind every cut you will make. A dashboard is read under a *cognitive load budget*, not just a screen-space budget. The human working-memory span is small — famously around four to seven chunks — and a stressed, sleep-deprived on-call is at the bottom of that range. Each panel they have to consider is a chunk. Past roughly seven panels, adding a panel does not add information to the reader; it *evicts* a different panel from their attention, because they cannot hold more. So the marginal panel on a crowded board has negative value: it does not just fail to help, it actively crowds out a panel that would have. This is why "is it useful?" is the wrong question for the overview and "is it more useful than what I would delete to make room?" is the right one. The budget is not arbitrary; it is set by the reader's wetware. Design within it or design against the only person who matters at 3am.

## 2. The dashboard hierarchy: overview, deep-dive, capacity, business

The single most important structural decision is to stop building *one* dashboard and start building a *hierarchy*. Different questions live at different altitudes, and trying to answer all of them on one screen is exactly how you get the wall of panels. Figure 2 contrasts the two worlds.

![A before-and-after diagram contrasting a 52-panel everything board where averages hide the tail and triage takes minutes against a 6-panel RED-plus-SLO overview where rate errors and p99 sit top-left and triage takes thirty seconds](/imgs/blogs/dashboards-that-tell-the-truth-2.png)

There are four tiers, and each is designed from its own job, audience, and question. The matrix in Figure 3 lays them out side by side, and it is worth internalizing because it kills the single-pane-of-glass myth more decisively than any argument.

![A four-by-three matrix mapping the service overview deep-dive capacity and business dashboards to the question each answers its primary viewer and its refresh and time range](/imgs/blogs/dashboards-that-tell-the-truth-3.png)

| Tier | Question it answers | Primary viewer | Panel count | Refresh / range |
| --- | --- | --- | --- | --- |
| **Service overview** | Is it healthy? Where is it broken? | On-call, mid-incident | 5–8 | 10s / last 1h |
| **Deep-dive (per subsystem)** | Why is *this* subsystem slow / erroring? | On-call, after the overview localizes | 8–15 | 30s / last 6h |
| **Capacity / planning** | Will we run out of headroom? | Capacity owner, weekly | 6–12 | 1h / last 90d |
| **Business / exec** | Is the business feeling it? | PM, exec, standups | 4–8 | 5m / last 30d |

**The service overview** is the heart of the hierarchy and the only board most people will ever open during an incident. It is one screen — no scrolling — and it answers the two questions from section 1. At the top it carries the SLO and error-budget status (the verdict), then the golden signals, then a single user-journey SLI. Its job is triage, not diagnosis. It tells you the patient has a fever; it does not tell you which organ. We design it in detail in section 4.

**The deep-dive boards** are per-subsystem: one for the database, one for the cache, one for the message queue, one for each downstream dependency. These are where you go *after* the overview has localized the problem. They can be denser — ten or fifteen panels — because you arrive at them with a hypothesis, not a blank mind. The deep-dive for the database shows connection-pool saturation, slow-query counts, replication lag, lock waits — the stuff you need once you already know the database is the suspect. You do not stare at the database deep-dive during normal operation. You arrive there from the overview, with a question.

**The capacity board** serves a completely different reader on a completely different clock. The on-call cares about the last hour; the capacity owner cares about the last ninety days and the trend line projected forward. This board answers "when do we run out?" — disk growth rate, request growth, the headroom on the connection pool, the saturation trend. Its time range is weeks, its refresh is hourly, and it is reviewed in a meeting, not during a page. Putting capacity panels on the overview is a category error: a ninety-day disk-growth trend is noise to someone trying to triage a five-minute outage.

**The business board** speaks the language of revenue, signups, conversion, and active users. Its audience is product managers and executives. It deliberately hides the engineering detail — no p99, no connection pools — and surfaces the metrics the business steers by. The link between the business board and the overview is the user-journey SLI: when checkout success drops on the overview, revenue drops on the business board, and the two should be visibly correlated. But they are separate boards because their readers ask different questions and act on different timescales.

### Why the single-pane-of-glass myth is so seductive and so wrong

Every few years a vendor sells leadership on "one dashboard for the whole company," and every time it collapses under its own contradictions. The reason is structural, not a failure of effort: the design constraints of the four tiers are mutually exclusive. The overview must fit on one screen and refresh every ten seconds; the capacity board must show ninety days and refresh hourly. The overview must be readable by a sleep-deprived on-call who knows the system cold; the business board must be readable by a VP who does not know what p99 means and should not have to. A panel tuned for one reader is wrong for the others. You cannot put a ninety-day trend and a one-hour incident view on the same axis without lying about one of them. So you build four boards, each honest about its own question, and you link them. The hierarchy *is* the single pane of glass — it is just spread across four panes, each of which tells the truth.

The links between tiers matter as much as the tiers themselves. A well-built hierarchy is *navigable*: from the overview you click the errors panel and land on the errors deep-dive, scoped to the same time range and the same service. Grafana data links make this literal — a panel can carry a URL that opens another dashboard with the current variables and time range preserved, so the on-call never has to manually re-pick a time window when they drill down. The result is a tree the eye walks: start at the verdict, follow the worst signal one level down, and arrive at the diagnosis with no friction. A hierarchy whose tiers do not link is just four separate boards the on-call has to find and align by hand at 3am, which is most of the way back to the wall of panels.

#### Worked example: cutting a 52-panel board to a 6-panel overview

Here is the redesign that motivated this post, panel by panel, because the *reasoning for each cut* is the lesson. The original "Checkout — Everything" board had 52 panels across nine rows. Here is what was on it and what happened to each group:

- **Eight per-host CPU panels, one per pod.** *Cut.* Eight separate gauges is eight things to scan; nobody compares pod 3 to pod 6 during triage. Replaced by one saturation panel showing max pool and max CPU across the fleet. Eight panels to one.
- **Six raw counter panels** (`http_requests_total`, `db_queries_total`, and four more), each an ever-rising line. *Cut entirely.* A raw counter tells you nothing at a glance. The one that mattered (request volume) came back as a single `rate()` panel; the rest were noise.
- **The mean-latency panel.** *Cut and replaced* with a single p50/p95/p99 panel. One panel, three lines, the whole distribution.
- **Eleven JVM/runtime panels** (heap, GC pauses, thread counts, class loading). *Moved* to the runtime deep-dive. Useful when you already suspect a GC problem; pure noise on the overview.
- **Nine database panels** (pool, slow queries, replication lag, locks, buffer cache, and more). *Moved* to the database deep-dive. You arrive there from the overview, not before.
- **Four "total since deploy" vanity counters** (total orders, total revenue, total users). *Deleted or moved to the business board.* Cumulative totals never catch an outage; they only ever go up.
- **The SLO panel** that did exist, buried in row four. *Promoted to top-left* as the verdict, with budget burn added.
- **Seven assorted panels** nobody could explain (added during old incidents, never removed). *Deleted* after confirming via usage analytics that none had been opened in the last quarter's incidents.

What survived became the six-panel overview: SLO + budget burn, request rate, error ratio, p50/p95/p99 latency, saturation, and a newly added checkout journey SLI. Fifty-two panels became six, and the forty-six that left did not lose any information the on-call needed for triage — every one of them is either reachable in a deep-dive or was noise. Measured result on the next game day: time-to-triage dropped from a stopwatch-measured 4 minutes 10 seconds of scanning the old board to 28 seconds on the new one. The deletion *was* the improvement.

## 3. Design principles: how to make a panel readable under stress

A dashboard is read by a human eye, and the human eye has rules. The eye lands top-left and sweeps right and down — that is the reading path in every left-to-right language, and it is hardwired enough that it survives sleep deprivation. So **the most important thing goes top-left.** On the overview, top-left is the SLO and error-budget verdict, because that is the answer to "is it healthy?". Everything else flows down and right from there in decreasing order of triage importance. If your most important panel is in the bottom-right corner below the fold, you have built a board that fights the eye, and at 3am the eye wins.

Here are the principles that pass the five-second and thirty-second tests. Each is a rule I have watched a board violate during a real incident, with real cost.

**Consistent time range across all panels.** Every panel on a board shares one time picker. If one panel shows the last hour and another shows the last six because someone hardcoded a range, you will misread a correlation — the deploy spike that lines up on one panel is offset on the other, and you will chase a ghost. Grafana makes this the default; do not override it per-panel without a very good reason, and if you do, put the range in the panel title so nobody is fooled.

**Color that means something.** Red and green are the most powerful preattentive signal you have, which means they are too valuable to waste on decoration. Reserve red for "this is bad and you should look," green for "this is within SLO," and a neutral color for everything informational. The cardinal sin is a board where every series gets a random color from the palette, so red is just "the third line" and carries no meaning. Tie your thresholds to the SLO: a latency panel turns yellow at the SLO threshold and red past it, so the color *is* the verdict. We will see the PromQL and threshold config for this in section 4.

**Percentiles, not averages, on latency.** This one deserves its own section and gets one (section 5), but state the principle now: never plot the average of a latency distribution, because the average is a liar. Plot p50, p95, and p99. The average can sit at a comfortable 80ms while one in a hundred users waits two seconds, and those are exactly the users who are about to churn or page you.

**Rate, not raw counters.** A Prometheus counter only goes up. Charting a raw counter gives you an ever-climbing line that means nothing — you cannot see a spike, only an accumulation. Always wrap counters in `rate()` so you see the per-second behavior, which is what you actually care about: requests per second, errors per second. A raw `http_requests_total` line is one of the most common rookie dashboard mistakes and it makes the panel useless.

**Annotations for deploys and incidents.** This is the highest-leverage feature most teams ignore. An annotation is a vertical line on every panel marking an event — a deploy, a config change, a feature-flag flip, the start of an incident. The single most common cause of a regression is a change, and the fastest way to confirm "did a deploy cause this?" is to see the deploy annotation land exactly where the latency line bent. Without annotations you are correlating a graph against your memory of the deploy schedule; with them, causality is visible. We will write the annotation query in section 7. This is non-negotiable on the overview.

**A panel budget, ruthlessly enforced.** The overview gets five to eight panels. Not because eight is magic, but because eight is roughly how many a human can scan in five seconds. Every panel you add past that dilutes the rest. When someone asks to add a panel to the overview, the question is not "is this useful?" — almost everything is useful — but "is this more useful than the least useful panel currently here, which I will now delete to make room?". A fixed budget forces that trade-off and keeps the board honest.

**Aggregate, then drill — never the reverse.** A common mistake is to show per-instance breakdowns on the overview: one line per pod, one panel per host. With three pods that is tolerable; with thirty it is a hairball nobody can read. The overview should show the *aggregate* — fleet-wide rate, fleet-wide p99 — and let the deep-dive break it down by pod when you need to find the one bad host. The reasoning is the same eye-path logic: the overview answers "is the service healthy?", which is a question about the service, not about pod 17. Per-instance detail belongs one level down, surfaced only when the aggregate points you there. A board that leads with thirty per-pod lines has confused "all the data" with "the answer."

**Defaults that do not surprise.** The overview should open to a sensible time range (the last hour is usually right for triage), refresh on its own, and require zero clicks to be useful. If the on-call has to pick a time range, expand a collapsed row, or select a variable before the board says anything, you have added friction at the worst possible moment. Set the default time range, set the auto-refresh, default the variables to "all" or to the most-paged service, and make the board answer the question the instant it loads. Every click you remove is a second of clarity returned to someone who has very few seconds to spare.

#### Worked example: the eye-path audit

Take your current overview and do this. Cover the bottom two-thirds of the screen with your hand. Can you answer "is it healthy?" from the top third alone? On a well-designed board, yes: the SLO verdict and the golden signals are all in that top third. On the forty-eight-panel board, the SLO panel was in row four, position three — dead center, below the fold, invisible to a five-second glance. We moved it to top-left, and triage time on that service dropped from a measured four-plus minutes of scanning to under thirty seconds, with no change to the underlying metrics at all. The data was always there. The *design* was the bug.

## 4. Building the service overview: RED, SLO, and the golden signals

Now we build the overview concretely, with real PromQL, for a running example: a checkout service in a webshop. This is the board the on-call opens first. Its layout follows the eye path, shown in Figure 5 — the SLO verdict and golden signals occupy the top-left, detail flows down and right.

![A two-row three-column grid showing the overview layout with the SLO and budget burn panel top-left then request rate and error ratio across the top row and the checkout SLI latency percentiles and saturation across the bottom row](/imgs/blogs/dashboards-that-tell-the-truth-5.png)

The overview is built on two well-worn method frameworks. **RED** — Rate, Errors, Duration — is the request-centric view, perfect for a request-driven service like checkout. **The four golden signals** from the Google SRE book — latency, traffic, errors, saturation — are the superset: RED's Duration is golden-signal latency, RED's Rate is traffic, errors are shared, and the golden signals add saturation (how full your resources are). For a request-serving service, RED plus a saturation panel plus the SLO is the whole overview. Let us write each panel.

### The SLO and error-budget panel (top-left, the verdict)

This is the first thing the eye hits and the answer to "is it healthy?". It shows the current SLO compliance and how much error budget remains in the window. Assume a 99.9% availability SLO over 30 days, where availability is the ratio of good (non-5xx) requests to total. The error budget is the allowed 0.1% of requests that may fail. Here is the PromQL for budget remaining as a fraction:

```promql
# Good = requests that did NOT return 5xx, over the 30-day window
# Budget remaining = 1 - (errors_so_far / errors_allowed)
1 - (
  (
      sum(increase(http_requests_total{job="checkout", code=~"5.."}[30d]))
    /
      sum(increase(http_requests_total{job="checkout"}[30d]))
  )
  / (1 - 0.999)
)
```

The numerator is the observed error ratio over the window; dividing by `(1 - 0.999)` = `0.001` expresses it as a fraction of the budget; subtracting from 1 gives budget *remaining*. A value of `1.0` means the full budget is intact; `0.0` means it is exhausted; negative means you have blown the SLO. Render this as a Grafana stat panel with thresholds: green above 0.25 (more than a quarter of the budget left), yellow from 0 to 0.25, red below 0. Now the color is the verdict. A glance tells you not just whether you are meeting the SLO but how much room you have left before you must stop shipping features — the [error budget being the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) that governs that decision.

Querying `increase(...[30d])` on every refresh is expensive. In production you precompute the error ratio with a recording rule so the panel reads a cheap pre-aggregated series:

```yaml
groups:
  - name: checkout-slo.rules
    interval: 30s
    rules:
      - record: job:http_requests:error_ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="checkout"}[5m]))
      - record: job:http_requests:rate5m
        expr: sum(rate(http_requests_total{job="checkout"}[5m]))
```

The overview's budget panel then reads from the long-window equivalent, and your burn-rate alerts (a sibling topic) read from the same recording rules. One source of truth for the number, reused by the dashboard and the alert, so the page and the panel never disagree.

### The RED panels (rate, errors, duration)

**Rate (traffic).** Requests per second, the heartbeat of the service. A sudden drop is as alarming as a spike — it can mean an upstream load balancer stopped sending you traffic:

```promql
sum(rate(http_requests_total{job="checkout"}[$__rate_interval]))
```

Note `$__rate_interval`, Grafana's built-in variable that picks a rate window safely scaled to the panel's resolution, so you never get the jagged or empty graph that comes from a window narrower than the scrape interval.

**Errors.** Plot the error *ratio*, not the raw count, because a thousand errors per second is fine at a million requests per second and catastrophic at two thousand. The ratio is the SLI:

```promql
sum(rate(http_requests_total{job="checkout", code=~"5.."}[$__rate_interval]))
/
sum(rate(http_requests_total{job="checkout"}[$__rate_interval]))
```

Set the threshold at your SLO boundary: green below 0.001 (0.1%), red above. Now the panel turns red exactly when you are out of SLO, and the color carries the meaning.

**Duration (latency).** Percentiles from a histogram, never the average:

```promql
histogram_quantile(0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{job="checkout"}[$__rate_interval]))
)
```

Plot three series on one panel — p50, p95, p99 — by repeating the query for quantiles 0.50, 0.95, 0.99. Three lines, one panel, the whole shape of the latency distribution visible at a glance. The gap between p50 and p99 is the tail, and the tail is where the pain lives (section 5).

### The saturation panel and the journey SLI

**Saturation** is the fourth golden signal: how full are your constrained resources? For checkout the binding constraint is usually the database connection pool, so chart pool utilization, plus CPU as a backstop:

```promql
max(db_connection_pool_in_use{job="checkout"})
/
max(db_connection_pool_max{job="checkout"})
```

A pool at 95% is a latency cliff waiting to happen — requests queue for a connection and the tail explodes. Saturation is the leading indicator; latency is the lagging one.

**The journey SLI** is the panel that would have saved the outage in section 6: a direct measurement of whether the user-facing flow works end to end. For checkout, that is the success rate of synthetic or real checkout completions, not the health of any single server. We make the full argument for this in section 6 and in the sibling post on [monitoring the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server). On the overview it gets one prominent panel, because it is the closest thing you have to the truth about whether customers are happy.

That is the whole overview: SLO/budget, rate, errors, p50/p95/p99 latency, saturation, journey SLI. Six panels. It fits on one screen. It passes the five-second test. And it is the only board the on-call needs to open before they know whether to escalate.

## 5. Why averages lie and percentiles tell the truth

This is the single most important honesty principle in dashboard design, so it gets its own section with the math. The claim: **the average of a latency distribution is almost always a lie, and the lie systematically hides exactly the users you most need to protect.**

Latency distributions are not symmetric. They are right-skewed with a long tail — most requests are fast, a few are very slow, and the slow ones are dragged out by garbage collection pauses, lock contention, cold caches, a slow downstream, or a retry. The mean of a right-skewed distribution sits well below the tail, so a mean of 80ms is perfectly compatible with a p99 of 2 seconds. If you chart the mean, the panel says "80ms, all good," and one in every hundred requests is taking 2 seconds. At a thousand requests per second that is ten users *every second* having a miserable experience, invisible on your board.

Here is the arithmetic that makes it concrete. Suppose 99 requests take 50ms and 1 request takes 5000ms:

$$\bar{x} = \frac{99 \times 50 + 1 \times 5000}{100} = \frac{4950 + 5000}{100} = 99.5\text{ ms}$$

The average is 99.5ms — looks healthy. But the p99 is 5000ms. The single slow request barely moves the mean (it contributes 50ms of the 99.5) while utterly dominating the experience of the unlucky 1%. The average has *averaged away* the only signal that matters. This is why every serious latency panel plots percentiles. Figure 7 collects this and the other chart lies into one reference.

![A five-by-two matrix listing common dashboard lies including the truncated y-axis the average on latency the raw counter the dual-axis overlay and meaningless colors each paired with its honest default fix](/imgs/blogs/dashboards-that-tell-the-truth-7.png)

#### Worked example: the tail that paged at p99 while p50 slept

A payments team I worked with ran a dashboard with a single "API latency" panel showing the mean. For weeks it read a steady 90ms. Customer complaints about "the app hanging at checkout" trickled in, but the dashboard said everything was fine, so the complaints were filed as user error. We added p50, p95, and p99 to the panel. p50 was 45ms — half the users were *faster* than the old mean suggested. p99 was 3.2 seconds. The tail had been there the whole time; the mean had hidden it. The 3.2-second tail traced to a connection-pool exhaustion under load — requests queueing for a database connection. Once we could *see* it on the percentile panel, the fix (raising the pool size and adding a saturation panel as a leading indicator) was obvious. Mean latency: still 90ms before and after, because the fix helped the slow 1% and barely touched the fast 99%. The percentile panel was the difference between a problem you can see and a problem you cannot.

A caveat that keeps you honest: `histogram_quantile()` in Prometheus interpolates within histogram buckets, so its precision is bounded by your bucket boundaries. If your buckets jump from 1s to 5s, a p99 reported as "3.2s" is really "somewhere between 1s and 5s." That is usually fine for triage — you only need to know the tail is bad, not its third decimal — but if you need precise tail numbers, define finer buckets around your SLO threshold. And remember you cannot average percentiles across instances: averaging the per-pod p99 does *not* give you the fleet p99. Aggregate the raw histogram buckets first (the `sum by (le)` in the query above), then take the quantile. Averaging quantiles is its own subtle lie, and one many dashboards commit.

### The other chart lies: truncated axes and dual-axis tricks

The average is the most common dashboard lie, but it has siblings, and a dashboard that tells the truth has to defend against all of them. Two deserve a close look because they are easy to commit by accident.

**The truncated y-axis** makes a molehill look like a mountain or a mountain look like a molehill, depending on which way you want to mislead — and the danger is that you usually do it *without* meaning to. Grafana, by default, auto-scales the y-axis to fit the data, which means a panel showing error rate bouncing between 0.40% and 0.42% will draw that 0.02-point wiggle as a dramatic full-height sawtooth. The on-call glances at it, sees a chart that looks like it is exploding, and panics over a 0.02% movement that is pure noise. The honest fix depends on the panel's job: for a panel whose absolute level matters (latency, error ratio against an SLO), anchor the y-axis at zero so the height of the line is proportional to the actual value. For a panel where you genuinely care about *change*, plot the percentage change explicitly rather than letting an auto-scaled absolute axis fake a trend. Either way, the rule is: never let the axis range be an accident, because an accidental axis range is an accidental lie. Put the axis minimum in the panel config and review it like any other decision.

**The dual y-axis** is the most seductive lie because it looks sophisticated. You overlay request rate (thousands per second) and error rate (a fraction near zero) on one panel with two different y-axes, and now the two lines wander across the same space. The eye reads "these move together" or "these diverge here" — but that apparent correlation is an artifact of where you happened to set the two scales. Slide one axis and the "correlation" vanishes or reverses. A dual-axis panel can make any two unrelated series look causally linked, which is exactly the cognitive trap an incident does not need. The honest fixes: split into two stacked panels sharing the same time axis (so vertical alignment still lets you spot real correlation without faking the magnitudes), or normalize both series to a comparable unit (both as percentage-of-baseline, say) so a single honest axis serves both. Reserve the dual axis for the rare case where the two units are genuinely related and you have thought hard about both scales — and even then, label both axes loudly. When in doubt, two panels beat one clever overlay.

Both of these share a root with the average: they let the *presentation* distort the *data*. A dashboard that tells the truth treats every axis choice, every overlay, every color as a claim it has to stand behind. The discipline is to ask, of every panel, "could a reasonable on-call misread this under stress?" — and if the answer is yes, fix the presentation until the answer is no.

## 6. The anti-pattern that hurts most: green while users are down

Now the failure that motivates this whole post, because it is the one that does the most damage and is the hardest to catch: **the dashboard that looks green while users are down.** Figure 6 shows the before and after.

![A before-and-after diagram showing a server-monitoring board glowing green while checkout fails for twelve minutes contrasted with a user-journey SLI panel that turns red within thirty seconds of the same failure](/imgs/blogs/dashboards-that-tell-the-truth-6.png)

Here is how it happens, and it has happened to almost every team. The overview measures the *servers*: CPU, memory, disk, host up/down, maybe per-process restart counts. During an incident, every one of those panels is green. The hosts are up. CPU is nominal. Memory is fine. And checkout has been completely down for twelve minutes, because the payment gateway integration is returning errors that the application catches, logs, and turns into a friendly "please try again" page — which means the user fails but the *server* is perfectly healthy. The dashboard is monitoring the machine. The machine is fine. The user is not. Nothing on the board measures the user, so the board lies, confidently and in green.

The reason this is so dangerous is that it inverts the purpose of monitoring. You built the dashboard to catch outages, and during the one outage that matters it reassures you that everything is fine. The on-call, trusting the green board, does not even get paged — because the alerts are wired to the same server metrics — and the outage is discovered when a customer tweets at the company account. Detection time: twelve minutes, by Twitter. That is a catastrophic failure of the monitoring system disguised as a healthy system.

#### Worked example: the green-board checkout outage and the one-panel fix

The timeline of the real incident: 14:02 the payment gateway begins returning 502s to the checkout service. The checkout service catches them and renders a retry page — correct application behavior, terrible for visibility. 14:02 to 14:14, the overview is entirely green: hosts up, CPU 40%, memory 55%, no host alerts. 14:14 a customer tweets "can't check out, been trying for 10 minutes." 14:16 an engineer sees the tweet, manually tries checkout, confirms it is broken, and only *then* finds the gateway 502s buried in the logs. Detection took fourteen minutes and required a customer to do the monitoring for us.

The fix was a single panel and a single SLI: **checkout success rate**, measured from the application's own instrumentation of the end-to-end flow — did the user reach the order-confirmation step? We added a synthetic prober too, hitting the real checkout path every thirty seconds from outside the cluster, so even a total application failure would be caught. The PromQL for the success-rate panel:

```promql
sum(rate(checkout_attempts_total{result="success"}[$__rate_interval]))
/
sum(rate(checkout_attempts_total[$__rate_interval]))
```

We re-ran the incident in a game day. With the journey panel on the overview, the same gateway failure turned the panel red within thirty seconds — success rate dropped to 4% — and the burn-rate alert wired to this SLI paged immediately. Detection: under one minute instead of fourteen. MTTR fell from the original fourteen-plus minutes to roughly two. The lesson is brutal and simple: **monitor the user journey, or your green board will betray you during the exact outage it was built to catch.** A server panel tells you the machine is alive. Only a journey SLI tells you the customer is being served. The full treatment of this principle is in the sibling post on [monitoring the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server); on the dashboard, the rule is to put one journey SLI panel on the overview, always, near the top, where the eye lands.

## 7. The dashboard as an incident tool: designing for the 3am flow

The overview is not a reporting artifact; it is a *triage instrument*, and it should be designed for one specific workflow: the half-asleep on-call following the worst signal to its source. Figure 4 traces that flow.

![A branching graph showing the on-call flow from page to overview to picking the worst golden signal then branching to the errors or latency deep-dive checking deploy annotations and mitigating](/imgs/blogs/dashboards-that-tell-the-truth-4.png)

The flow is: the page fires, the on-call opens the overview, reads the golden signals in five seconds, identifies the *single worst* one, and follows it into the matching deep-dive. They do not read all six panels equally; they scan for the one that is red and chase it. That is why the overview's job is to make the worst signal *jump out* — color discipline and eye path exist to serve this exact moment. If errors are red, the on-call clicks through to the errors deep-dive (top endpoints by 5xx, errors by downstream dependency, recent deploy correlation). If p99 is red, they go to the latency deep-dive (latency by endpoint, by pod, saturation of the pool). One signal, one path, thirty seconds.

The highest-leverage thing you can add to support this flow is **deploy and incident annotations.** When the on-call lands on a panel and sees the latency line bend sharply upward, the very next question is "what changed?" — and the answer is almost always a deploy. An annotation that lands a vertical line exactly where the bend starts answers that question in one second. Here is a Grafana annotation query that pulls deploy events from a Prometheus metric your CI emits on every deploy:

```promql
# Annotation query: render a vertical line at each deploy
changes(deploy_timestamp_seconds{job="checkout"}[$__rate_interval]) > 0
```

Or, if you record deploys as a metric with the version in a label, drive the annotation off that and put the version in the annotation text so the line reads `deploy v2.4.1` and you know precisely which release to roll back. Wire your incident-management tool to drop an annotation at incident-declared and incident-resolved too, so post-incident the board shows the whole story: deploy here, latency bent there, incident declared, mitigation applied, recovery. That annotated timeline is also your postmortem's first exhibit — the correlation between the change and the regression, drawn automatically.

Designing for the 3am flow also means **labeling panels for a tired human.** A panel titled `histogram_quantile(0.99, ...)` is a query; a panel titled "p99 latency (SLO: 250ms)" is information. Put the SLO threshold in the title and as a threshold line on the graph, so the on-call sees not just the current value but how far it is from the line that matters. Put units on every axis. Put the time range in the title if it differs from the board default. Every bit of cognitive load you remove from the panel is a bit of clarity you give the on-call at the moment they have the least to spare.

The most concrete way to make color carry meaning is to wire the panel's thresholds to the SLO in the dashboard JSON itself, so the color is computed from the number, not painted by hand. Here is the threshold and field config for the p99 latency panel, tied to a 250ms SLO:

```json
{
  "fieldConfig": {
    "defaults": {
      "unit": "s",
      "min": 0,
      "thresholds": {
        "mode": "absolute",
        "steps": [
          { "color": "green",  "value": null  },
          { "color": "yellow", "value": 0.20  },
          { "color": "red",    "value": 0.25  }
        ]
      }
    }
  },
  "title": "p99 latency (SLO: 250ms)"
}
```

Now the panel goes yellow at 200ms (approaching the budget) and red at 250ms (the SLO line), and `min: 0` anchors the axis so the height of the line is honest. The color *is* the verdict, computed from the SLO, and it cannot drift because it lives in the reviewed JSON. Apply the same pattern to the error-ratio panel (red at 0.001) and the budget panel (red below 0), and the whole overview reads as a traffic light: any red is bad, all green is healthy, and the on-call's five-second glance is preattentive instead of analytical.

### Stress-testing the overview

Good design survives the hard cases. Let us stress-test the overview against the situations that actually happen:

- **What if two incidents overlap?** The overview shows aggregate golden signals, so two simultaneous problems can blur — errors up *and* latency up, which one is primary? This is where the deep-dives earn their keep: the on-call opens both the errors and latency deep-dives, and the annotations usually disambiguate (a deploy four minutes ago points at one; a dependency outage at the other). The overview triages; it does not have to diagnose two things at once.
- **What if the dashboard itself is part of the outage?** If Grafana or Prometheus is down, your overview is blind. This is why critical alerts must not depend solely on the same Prometheus the dashboard reads — you need a meta-monitor (a second Prometheus, or a dead-man's-switch alert that pages if the primary stops reporting). The dashboard is a triage tool, not the alerting system; the [alert that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) is what wakes you, and it must survive the dashboard being down.
- **What if the on-call has never seen this service?** During a broad incident, someone may get paged for a service they do not own. A well-designed overview is *self-documenting*: panel titles carry SLO thresholds, a text panel at the top links the runbook and the deep-dive boards, and the journey SLI names the user flow in plain English ("checkout success rate"). A stranger should be able to triage from the overview alone, because the design did the explaining.
- **What if the budget is already spent?** The SLO panel should make this unmissable — red, "0 min budget, SLO breached this window." That does not change the triage flow, but it changes the stakes: every additional minute of this outage is now eating into next window's standing with your stakeholders, and it sharpens the escalation decision.

### Designing the deep-dive the overview points to

When the overview localizes the problem — "errors are red" — the on-call clicks through to a deep-dive, and that board has a different job: not "is it healthy?" but "*why* is this subsystem unhealthy, and what do I do about it?". The deep-dive can be denser because the on-call arrives with a hypothesis. But density is not license for the wall of panels; the deep-dive is still designed, still ordered by the eye path, still honest. It just goes one layer deeper on one subsystem.

Take the errors deep-dive for our checkout service. The overview said errors are up; the deep-dive's job is to answer "which endpoint, which dependency, since when?". Three panels do most of the work. First, errors broken down by endpoint, so you see whether it is the whole service or one route:

```promql
topk(5,
  sum by (route) (
    rate(http_requests_total{job="checkout", code=~"5.."}[$__rate_interval])
  )
)
```

Second, errors broken down by downstream dependency, because a checkout 5xx is usually a *downstream* failure surfacing — the payment gateway, the inventory service, the database. If your client libraries emit a per-dependency error metric, this panel points straight at the culprit:

```promql
sum by (dependency) (
  rate(downstream_request_errors_total{job="checkout"}[$__rate_interval])
)
```

Third, the deploy annotation overlay (section 7's query) so that "since when?" answers itself: if the errors bend upward exactly at the `deploy v2.4.1` marker, you have your root cause and your mitigation (roll back) in one glance. The latency deep-dive mirrors this shape: p99 by endpoint, p99 by pod (to catch a single bad host), and the saturation panels (pool, CPU, queue depth) that usually explain a latency cliff. The discipline is the same as the overview — every panel earns its place by helping answer the deep-dive's one question — but the question is narrower, so the panels are more specific. A good deep-dive is built lazily, the first time an incident makes you wish you had it, with exactly the panels you wished for in that moment. That is how it stays matched to reality instead of speculative.

## 8. Building dashboards as code

A dashboard you build once by hand in the Grafana UI is a liability the moment the service changes. Someone renames a metric, the panel goes blank, and nobody notices until the next 3am page when the on-call opens the board and finds an empty chart where the error rate should be. Hand-built dashboards rot, drift between environments (staging's board does not match production's), cannot be code-reviewed, and cannot be rolled back. The fix is to treat dashboards exactly like the rest of your infrastructure: **as code, versioned in git, reviewed in pull requests, and applied by automation.** Figure 8 shows the lifecycle.

![A timeline showing the dashboard-as-code lifecycle from authoring Grafana JSON in git through pull-request review Terraform deploy use during an incident and pruning unused panels](/imgs/blogs/dashboards-that-tell-the-truth-8.png)

There are three common ways to do this, and they sit on a spectrum from "export the JSON" to "generate everything from a library":

| Approach | What it is | Best for |
| --- | --- | --- |
| **Grafana JSON in git** | Export the board's JSON model, commit it, apply via provisioning or API | Small teams, getting started, few boards |
| **Terraform `grafana_dashboard`** | Manage dashboards as Terraform resources alongside the rest of your infra | Teams already on Terraform; consistent apply / rollback |
| **Jsonnet + grafonnet / Grafana Foundation SDK** | Generate JSON from a typed library; one function builds a RED overview for any service | Many services that should share one consistent board shape |

For most teams, start with Grafana JSON in git and graduate to Terraform when you have more than a handful of boards. Here is a minimal Terraform resource that puts a dashboard under version control and applies it idempotently:

```yaml
# main.tf — manage the checkout overview as code
resource "grafana_dashboard" "checkout_overview" {
  config_json = file("${path.module}/dashboards/checkout-overview.json")
  folder      = grafana_folder.sre.id
  overwrite   = true
}

resource "grafana_folder" "sre" {
  title = "SRE Service Overviews"
}
```

The JSON file lives in the repo, the panels and PromQL are reviewed in the pull request just like application code, and `terraform apply` makes production match the repo. A change to the SLO threshold is now a diff someone approves, not a silent click in a UI that nobody can audit. When a deploy renames a metric, the same PR that renames the metric updates the dashboard query, and the review catches the mismatch *before* it goes blank at 3am.

The real payoff of code is **consistency through generation.** With jsonnet (via the grafonnet library) or the newer Grafana Foundation SDK, you write a single function — `redOverview(service)` — that produces a correct RED-plus-SLO overview for any service, with the right panels, the right thresholds tied to that service's SLO, the right eye-path layout. Spin up a new service, call the function with its name, and it gets a battle-tested overview for free. No more thirty subtly-different overviews where one team forgot the journey SLI and another plotted the average. The design discipline of this entire post gets *encoded once* and applied everywhere. That is the difference between dashboards as folklore and dashboards as engineering.

Here is the shape of that generator, sketched in grafonnet-style jsonnet. The point is not the exact API (it shifts between library versions) but that one function encodes the *design rules* — top-left SLO, percentile latency, rate-wrapped errors — so every service inherits them:

```yaml
# overview.libsonnet — one function builds a correct overview per service
local g = import 'grafonnet/main.libsonnet';

local redOverview(service, sloTarget) =
  g.dashboard.new('%s — overview' % service)
  + g.dashboard.withRefresh('10s')
  + g.dashboard.time.withFrom('now-1h')
  + g.dashboard.withPanels([
    // top-left: the SLO verdict, color-coded to budget remaining
    sloPanel(service, sloTarget) { gridPos: { x: 0, y: 0, w: 6, h: 6 } },
    ratePanel(service)           { gridPos: { x: 6, y: 0, w: 6, h: 6 } },
    errorRatioPanel(service, sloTarget),
    latencyPercentilesPanel(service),   // p50, p95, p99 — never the mean
    saturationPanel(service),
    journeySliPanel(service),           // the user-journey panel, always present
  ]);

{ redOverview:: redOverview }
```

Every service's overview is now `redOverview('checkout', 0.999)` — one line, and it cannot forget the journey SLI, cannot plot the average, cannot bury the SLO panel, because the function does not let it. When you improve the design (say you discover the saturation panel should also show queue depth), you change the function once and every service's board improves on the next apply. That is the compounding return of code: the lesson learned in one incident propagates to every overview in the fleet automatically. Hand-built boards cannot do that; each is a snowflake that has to be fixed by hand, which means most of them never get fixed.

### The discipline of deleting panels nobody uses

#### Worked example: reading the budget panel during a burn

Tie the overview's SLO panel back to the arithmetic, because the panel is only useful if you can read the number it shows. The checkout service runs a 99.9% availability SLO over a 30-day window. At that target the error budget is 0.1% of requests, which over 30 days of roughly continuous traffic works out to about 43.2 minutes of full downtime-equivalent per month — the same nines-to-downtime arithmetic the [error budget post](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) derives. Now a bad deploy ships at 14:00 and starts erroring 2% of requests instead of the allowed 0.1%. That is a 20x burn rate: you are spending budget twenty times faster than the SLO allows. The budget panel, reading the recording rule from section 4, drops visibly — and the arithmetic says a 20x burn empties a 30-day budget in roughly `30 days / 20` ≈ 1.5 days if left unchecked. The panel turning from green through yellow toward red *is* that countdown made visible. The on-call does not have to compute the burn rate in their head; the color and the trend show it. That is the whole point of putting the budget front and center: it converts an abstract policy ("meet 99.9%") into a live gauge the on-call steers by, and it makes the ship-or-freeze decision a reading off a panel rather than an argument in a meeting. When the panel hits red, the conversation is over — reliability work comes first until the budget recovers in the next window.

Code does not just let you build dashboards; it lets you *delete* them with confidence, because a deletion is a reviewable diff you can revert. And deletion is the most underrated dashboard discipline. Instrument dashboard and panel views — Grafana's usage insights, or the access logs — and once a quarter, look at what actually gets opened. The panels nobody viewed during the last quarter's incidents are not earning their place; they are noise the eye has to skip. Delete them. The panels that *were* opened during incidents are your real overview — promote them, make them prominent. This is the feedback loop that keeps a dashboard matched to reality: **a dashboard built once and never matched to real incidents is a fossil.** Match it. After every significant incident, ask "what did we wish the dashboard had shown us?" and "what did we never look at?" — add the first, delete the second. A dashboard that evolves with its incidents stays a tool; one that only accretes panels becomes the forty-eight-panel wall we started with.

## 9. War story: the dashboard that watched the wrong thing

Two real-shaped stories, because the failures here are instructive and common enough that some version of them has happened at almost every company that runs services at scale.

**The vanity-metric board.** A streaming company had a beautiful executive dashboard — concurrent viewers, total minutes streamed, a world map of active sessions glowing with traffic. It was genuinely impressive and it was on a TV in the office. During a major incident, when a CDN misconfiguration caused a fifth of streams to fail to start, the executive board barely flinched: total concurrent viewers dipped a few percent (the streams that *did* start kept playing fine), and the pretty map stayed lit. The number that mattered — *stream start success rate* — was not on the board, because it was not a vanity metric; it was a pain metric, and pain metrics do not look good on a TV. The lesson: vanity metrics (totals, cumulative counts, anything that mostly goes up and to the right) make beautiful dashboards and terrible incident tools. The metrics that catch outages are ratios of good-to-total over short windows — exactly the unglamorous SLIs that nobody puts on the office TV.

**The leap-second cascade and the missing annotation.** This one is closer to home for many teams. A service started timing out across the fleet at midnight UTC. The on-call opened the overview: latency red, errors climbing, every host showing high CPU. They spent forty minutes chasing a "load problem," scaling up, restarting pods, none of it helping, because the real cause was a clock issue — a kernel-level event that spiked CPU on every host simultaneously. What would have cut the forty minutes to five was a single annotation: an NTP/clock-event marker on the timeline showing that *every* host changed behavior at exactly 00:00:00 UTC, which immediately rules out a gradual load problem and points at a synchronized external event. The dashboard had all the symptom panels and none of the *change* panels — no deploy annotations, no infra-event annotations — so the on-call could see *that* everything broke at once but had no on-board hint as to *why*. The fix, beyond the specific kernel patch, was to add infrastructure-event annotations (deploys, config pushes, NTP steps, cloud-provider maintenance) so that "everything broke at the same instant" becomes "everything broke at the same instant, and here is the marked event that caused it." Correlation drawn automatically beats correlation reconstructed from memory at 3am, every time.

Both stories share a root cause: the dashboard measured the wrong thing for the moment it was needed. The streaming board measured vanity instead of pain. The timeout board measured symptoms without changes. The cure in both cases is the same discipline — design each board from the question it must answer during an incident, put the user-pain SLI front and center, and annotate the changes — and then *verify it against real incidents* and prune what did not help.

There is a third, quieter war story that every team eventually lives: the board that was *right* the day it was built and silently went wrong. A team built a careful overview, shipped it, and moved on. Eight months later a refactor renamed `http_request_duration_seconds` to `request_latency_seconds`. The dashboard's latency panel did not error — it just went flat at "no data," rendering as an empty axis. Nobody noticed, because a healthy service produces a boring panel and an empty panel looks boring too. Three weeks later, during a real latency incident, the on-call opened the overview and the latency panel was blank. They had no idea whether latency was fine or whether the panel was broken, which is the worst possible state: a monitoring tool you cannot trust during the exact moment you need it. The fix is structural, and it is the whole argument for dashboards-as-code in one anecdote: had the dashboard been a JSON file in the same repo as the metric rename, the pull request that renamed the metric would have shown a diff in the dashboard query too, and a reviewer would have caught it — or a CI check that validates every dashboard query against the live metric catalog would have failed the build. Hand-built boards have no such safety net; they rot in the dark and reveal the rot at the worst time. Code gives you a place to catch the rot before it reaches 3am.

## 10. How to reach for this (and when not to)

The dashboard hierarchy is not free, and like every reliability practice it has a point past which more investment is waste. Here is the decisive guidance.

**Build the service overview for every service that can page you.** This is non-negotiable. If a service has an SLO and an on-call rotation, it needs a six-to-eight-panel RED-plus-SLO overview with a journey SLI, designed for the 3am flow. This is the highest-leverage dashboard you will ever build and it pays for itself the first time it cuts triage from four minutes to thirty seconds.

**Build deep-dives lazily, driven by incidents.** Do not pre-build a deep-dive for every subsystem on day one; you will guess wrong about what you need. Build the database deep-dive the first time the overview points at the database and you wish you had one. Let real incidents tell you which deep-dives to build, and build exactly the panels you wished you had during that incident. This keeps deep-dives matched to reality instead of speculative.

**Build the capacity and business boards only when someone will actually use them on a cadence.** A capacity board nobody reviews quarterly is a fossil; build it when there is a capacity owner who will look at it. The business board is the same — build it when an exec or PM will actually steer by it, not because a hierarchy diagram says you should have one.

**When it is not worth it:** do not build a dashboard hierarchy for an internal batch job that nobody pages on — a single "did the last run succeed, and how long did it take" panel is plenty. Do not build a six-panel overview for a service with no SLO and no users; the panel budget is for services where triage speed has a cost. Do not invest in jsonnet generation until you have enough services that hand-maintaining boards is the bottleneck — for three services, Grafana JSON in git is fine, and the generation library is premature. And do not chase the single pane of glass; spend that energy on four honest boards instead. The cost of a dashboard is not the time to build it — it is the attention it taxes every time someone scans past a panel that does not earn its place. Spend that attention budget as carefully as you spend the error budget.

One more honest caution about scope creep, because it is the failure mode of teams that take this advice *too* far. Having read all this, you might be tempted to add a journey SLI for every micro-flow, a deep-dive for every class, annotations for every config key. Resist it. The overview is a six-to-eight panel instrument precisely because restraint is the discipline. The goal is not the most thorough dashboard; it is the *fastest correct triage*. A board with one perfect journey SLI beats a board with ten mediocre ones, because the on-call can only follow one signal at a time. Build the smallest set of panels that pass the five-second and thirty-second tests, and stop. The next panel can always be added during the next incident, when you actually know you need it — and a panel earned by a real incident is worth ten panels added on speculation.

The recurring thread of this series is the loop: define reliability with SLIs and SLOs, measure it with observability, spend the error budget deliberately, respond to incidents, learn, and engineer the fix. The dashboard is where measurement meets response — it is the instrument the on-call reads at the moment the error budget is burning. A dashboard that tells the truth makes that moment shorter, calmer, and correct. A dashboard that lies — green while users are down, average hiding the tail, forty panels of noise — makes it longer, more frantic, and wrong. Design for the truth.

## Key takeaways

1. **A dashboard is a tool with a job.** The overview's job is to answer "is it healthy?" in five seconds and "where is it broken?" in thirty. If a panel does not help answer a question someone will ask, delete it.
2. **Build a hierarchy, not a dashboard.** Service overview (triage), deep-dives (diagnosis), capacity (planning), business (the money). Each has its own question, audience, time range, and refresh — the single pane of glass is a myth.
3. **Design for the eye.** Most important top-left, consistent time range, color that means something (red = bad, green = within SLO), thresholds tied to the SLO. The eye lands top-left and wins every fight with your layout.
4. **Percentiles, not averages; rates, not raw counters.** The average of a latency distribution hides the tail where the pain lives. A raw counter is meaningless; wrap it in `rate()`. Plot p50, p95, p99 on every latency panel.
5. **Put the SLO and error budget front and center.** The first panel the eye hits should be the verdict: how much budget is left. That single panel decides whether you ship or freeze.
6. **Annotate deploys and incidents.** The fastest way to confirm "did a change cause this?" is a vertical line where the regression started. Annotations turn correlation-from-memory into correlation-on-screen.
7. **Monitor the user, not just the server.** A green server board will betray you during the one outage it was built to catch. Always put a user-journey SLI panel on the overview.
8. **Build dashboards as code.** Versioned, reviewed, applied by Terraform or generated by jsonnet. Hand-built boards rot, drift, and go blank silently. Code makes them consistent, auditable, and revertible.
9. **Delete panels nobody uses.** Instrument dashboard views, and once a quarter prune the panels no incident ever opened. Match the board to real incidents or it becomes a fossil.
10. **Spend the attention budget like the error budget.** Every panel taxes the eye. The discipline of dashboards is the discipline of subtraction — signal over noise, the question over the data.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the define-measure-budget-respond-learn-engineer loop this post fits into.
- [Choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) — the journey SLI on your overview is only as honest as the SLI behind it.
- [Metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right) — counters, gauges, histograms, and the `rate()`/`histogram_quantile()` mechanics every panel depends on.
- [Alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) — the dashboard triages; the alert wakes you. They must agree and the alert must survive the dashboard being down.
- [Monitor the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server) — the full argument behind the journey-SLI panel that prevents the green-board outage.
- [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — the budget panel on your overview and the ship-or-freeze decision it drives.
- [Observability: metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — the architecture-time view of the instrumentation your dashboards read.
- The Google SRE Book and SRE Workbook (free online) — the four golden signals, monitoring philosophy, and the "monitor symptoms, not causes" principle; the Grafana, Prometheus, and OpenTelemetry docs for the panel, query, and instrumentation specifics; Tom Wilkie's RED method write-up for the request-centric overview.

---
title: "Capacity Planning and Forecasting: The Outage You Can See Coming Weeks Away"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Stop being surprised by load: forecast demand at the p99, find your real per-unit capacity with a load test, size the fleet with headroom and N+1, and run below the saturation cliff before production finds it for you."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "capacity-planning",
    "load-testing",
    "forecasting",
    "saturation",
    "autoscaling",
    "headroom",
    "queueing-theory",
    "finops",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/capacity-planning-and-forecasting-1.png"
---

The worst page I ever got for a capacity problem arrived at 9:47 on a Tuesday morning, which is the wrong time for a capacity outage because everyone is awake to watch it. The service was a payments API that ran perfectly well at 4,000 requests per second on a fleet of fifteen instances. That morning a partner integration went live a day earlier than their change calendar said it would, dumped roughly 7,000 requests per second of new traffic onto us with no warning, and the p99 latency went from 110 milliseconds to 6.5 seconds in under ninety seconds. Not gradually. Not with a polite warning shoulder. It was flat, flat, flat, and then it was a wall. By the time I had the dashboard open the request queue was 40,000 deep, healthy instances were being marked unhealthy because their own health checks timed out behind the backlog, and the autoscaler was dutifully adding instances that were getting starved at the database connection pool the moment they came up. We were not out of CPU. We were not out of memory. We had run a single resource — the connection pool — straight off a cliff, and everything downstream of that cliff fell with it.

Here is what stung in the postmortem: every number we needed to prevent that outage was knowable a week earlier. The partner's go-live was on a calendar. Our per-instance capacity was measurable with a load test we had never run. The connection pool ceiling was a config value we could have read in thirty seconds. The fleet size we needed for the combined load was a division problem a junior could do. Running out of capacity is not like a cosmic-ray bit flip or a once-in-a-decade hardware fault. It is the most *foreseeable* class of outage there is — a slow-motion failure you can watch approaching for weeks if you are looking, and that blindsides you only if you are not. Capacity planning is the discipline of looking.

That is the whole thesis of this post, and it is worth saying plainly before we go deep: **you forecast demand, you know your per-unit capacity, you keep headroom for spikes, and you test to find the real limit before production does.** Four moves. Get all four right and you will never again be surprised by load. Get any one wrong and you are back at 9:47 on a Tuesday watching a wall.

![A flow showing latency staying flat as utilization rises from fifty to seventy percent, then bending sharply at the eighty-five percent knee and exploding past ninety-five percent toward an infinite queue, with the safe rule to run below the knee branching off](/imgs/blogs/capacity-planning-and-forecasting-1.png)

By the end of this post you will be able to do the concrete work: read the saturation curve of your own system and find where its knee sits, decompose your demand into trend and seasonality and spikes and forecast each, run a load test that finds your real breaking point instead of guessing it, size a fleet from that number with the right headroom and redundancy, and decide how much spare capacity is worth paying for given your SLO. This is the *running-it* layer of reliability. If you want the architecture-time side — how to design a system that scales and how autoscaling control loops actually work — that lives in the system-design post on [capacity planning and autoscaling](/blog/software-development/system-design/capacity-planning-and-autoscaling), and I will cross-link it where the line between design and operation gets thin. This post is about the operational discipline: the forecast, the load test, the headroom number, and the re-plan trigger that keeps you off the cliff.

A quick note on where this sits in the larger reliability loop. The whole series runs on one spine — define reliability with an SLI and SLO, measure it, spend the error budget deliberately, reduce toil, respond to incidents, learn, engineer the fix — and the error budget is the currency that ties it together. Capacity is woven through every link of that chain. Saturation is one of the four golden signals you measure. Running out of capacity is the fastest way to burn an error budget there is. And the question "how much capacity do we need?" is, at bottom, the same question as "how reliable do we want to be, and what will we pay for it?" If you have not read it, the intro to the whole series, [reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), frames that loop; this post is one of its load-bearing beams.

## 1. The saturation cliff: why systems don't degrade gracefully

Start with the single most counterintuitive fact about capacity, the one that catches good engineers off guard: **systems do not slow down gradually as they fill up. They are fine, and then they are not.** The mental model most people carry — that doubling the load roughly doubles the latency, that a system at 90% capacity is a little slower than one at 45% — is wrong, and it is wrong in a way that will hurt you. The truth is that latency stays nearly flat across a wide range of utilization and then, past a threshold, climbs almost vertically. Plotted, it looks like a hockey stick: a long flat handle, then a sudden blade. SREs call the bend the *knee* or the *saturation cliff*, and it is the single most important feature of any capacity curve.

You can feel the shape without any math. A grocery store with two open registers and four shoppers has no line; with eight shoppers it has a short line; with twenty shoppers it has a line out the door and the *time you spend waiting* has grown far faster than the number of shoppers did. The registers did not get slower. They are processing customers at the same rate they always did. What changed is that arrivals started colliding with each other, and every collision means waiting, and waiting *compounds* because the person behind you waits for your wait plus their own. That compounding is the cliff.

### A little queueing theory, just enough to be dangerous

The reason the cliff exists, and the reason it is mathematically unavoidable rather than a quirk of bad code, comes from queueing theory. Define utilization as the fraction of time a resource is busy, written $\rho$ (rho), where $\rho = 0$ means idle and $\rho = 1$ means saturated. For a simple queue (the M/M/1 model — random arrivals, random service times, one server) the average number of requests waiting in the system is:

$$L = \frac{\rho}{1 - \rho}$$

Look at what that fraction does as $\rho$ climbs toward 1. At $\rho = 0.5$, $L = 0.5/0.5 = 1$ — about one request in the system on average. At $\rho = 0.8$, $L = 0.8/0.2 = 4$. At $\rho = 0.9$, $L = 0.9/0.1 = 9$. At $\rho = 0.95$, $L = 0.95/0.05 = 19$. At $\rho = 0.99$, $L = 0.99/0.01 = 99$. The denominator $(1 - \rho)$ is shrinking toward zero, so the quotient is racing toward infinity. **As utilization approaches 100%, the queue length — and the time a request spends waiting — goes to infinity.** That is the cliff, in one equation. It is not a bug. It is arithmetic.

The latency consequence follows by Little's Law, which says the average time a request spends in the system is the average number in the system divided by the arrival rate ($W = L / \lambda$). Since $L$ blows up as $\rho \to 1$, so does $W$. The practical upshot is brutal and worth memorizing: **the last 10% of your nominal capacity is not usable capacity.** A resource that is "90% utilized" is not 90% of the way to full; in latency terms it is already most of the way up the blade of the hockey stick. The headroom between the knee and 100% is a trap — it shows up as "spare capacity" on a utilization dashboard but delivers exploding latency if you try to use it.

This is why you must run *below* the cliff, and it is the entire justification for headroom, which we will quantify in Section 4. The figure above traces a representative curve: at 50% utilization a request takes 100 milliseconds, at 70% maybe 130 milliseconds, and you are still on the flat handle. The knee — where the curve starts bending up — sits somewhere around 80–85% for a typical service. Push to 95% and the same request takes 4 seconds; nudge to 100% and the queue is unbounded and you are in an outage. The rule that falls out of this is simple: **find your knee, then run with enough headroom that normal peaks never reach it.**

### Which resource saturates first?

The cliff is not always CPU. In fact CPU is one of the *friendlier* bottlenecks because it degrades comparatively gracefully and is easy to watch. The dangerous saturations are the ones that hit a hard ceiling and then drop you straight off:

| Resource | The hard limit | Failure signature | Easy to miss? |
| --- | --- | --- | --- |
| CPU | Cores × time | Run-queue grows, latency rises | No, well-monitored |
| Memory | Total RAM | OOM kill, crash-loop, then cascade | Sometimes, if slow leak |
| Connection pool | Fixed max (e.g. 200) | New requests block, then time out | Yes, very |
| File descriptors / sockets | ulimit, ephemeral ports | `EMFILE`, `connection refused` | Yes, very |
| Disk IOPS | Provisioned ceiling | I/O wait climbs, everything stalls | Yes, on cloud volumes |
| Thread pool | Fixed worker count | Requests queue, then reject | Yes |
| Downstream service | Its own capacity | Your latency = their saturation | Yes, blame goes upstream |

The lesson from my payments outage is right there in the table: we were watching CPU and memory, both of which were comfortable, while a connection pool with a hard ceiling of 200 saturated invisibly. **You can only run below a cliff you can see.** Saturation is one of the four golden signals — latency, traffic, errors, and saturation — precisely because it is the leading indicator that all the others are about to go bad. We measure it as "how full is the most constrained resource," and the most constrained resource is rarely the one you instinctively watch. The metrics-and-time-series side of instrumenting saturation is covered in the sibling post on [metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right); here the point is conceptual: enumerate every fixed-ceiling resource on the request path, and put a saturation gauge on each.

Here is a Prometheus recording rule that turns "saturation" into a single number you can alert on — the worst-utilized resource across the fleet, expressed as a fraction of its limit:

```yaml
groups:
  - name: capacity.saturation
    rules:
      # connection pool: in-use / configured max
      - record: job:db_pool_utilization:ratio
        expr: |
          max by (service) (
            db_pool_connections_in_use / db_pool_connections_max
          )
      # CPU: 1m rate of cpu seconds vs cores requested
      - record: job:cpu_utilization:ratio
        expr: |
          sum by (service) (rate(container_cpu_usage_seconds_total[5m]))
          /
          sum by (service) (kube_pod_container_resource_requests{resource="cpu"})
      # the headline saturation SLI: the single worst resource
      - record: job:saturation:worst
        expr: |
          max by (service) (
            label_replace(job:db_pool_utilization:ratio, "res", "pool", "", "")
            or
            label_replace(job:cpu_utilization:ratio, "res", "cpu", "", "")
          )
```

That `job:saturation:worst` series is the one you put on the dashboard and the one you alert on when it crosses, say, 0.75 — which, if your knee is at 85%, gives you a margin before the curve bends. Watch the *worst* resource, not the average, because the cliff is decided by whichever resource saturates first, and the average will look healthy right up until the moment the constrained one falls off.

There is a deeper point hiding in that recording rule, and it is the single most useful habit you can build around capacity. **Saturation is per-resource, and the resource that bites you is rarely the one on your default dashboard.** Most teams ship with CPU and memory graphs because that is what the cloud provider hands them for free, and so those are the two resources they instinctively watch. But the dangerous ceilings — the ones with a hard limit and a vertical cliff right behind it — are usually the bespoke ones: a connection pool sized in a config file someone set two years ago, a thread pool whose `maxThreads` defaulted to 200, an ephemeral-port range that exhausts under high connection churn, an IOPS budget on a cloud volume that throttles silently the moment you exceed it. None of those appear on a stock dashboard. You have to go enumerate them. The exercise I run with every team I join: draw the request path end to end, and at every hop write down the *fixed-ceiling resources* — the pools, the limits, the quotas — and then go confirm each one has a saturation gauge. The first time you do this you will find at least one resource that is closer to its cliff than anyone realized, and it will be the one nobody was watching. That is not a coincidence; it is the rule. The unwatched resource is exactly where the next capacity outage lives, because if it had been watched, someone would have already raised its ceiling.

A practical aside on memory specifically, because it behaves differently from the other resources and trips people up. CPU and connection-pool saturation produce *latency* cliffs — requests queue and slow down. Memory saturation produces a *crash* cliff: when you run out of memory, the kernel's OOM killer terminates the process outright, the instance restarts, and in a fleet that restart shifts its load onto its neighbors, which can push *them* over their own cliffs, which is a cascading failure that started from one instance quietly leaking a few megabytes an hour. Memory saturation is therefore especially dangerous as a *slow* phenomenon, which is exactly why the soak test in Section 3 exists — a 20-minute load test will never show you a leak that takes eight hours to fill the heap.

## 2. Demand forecasting: planning for the load you will have, not the load you have

If the cliff tells you where the edge is, forecasting tells you how fast you are walking toward it. The job of a demand forecast is to answer one question with a defensible number: *how much load will this system have to serve over the planning horizon, and with how much confidence?* Get a credible answer and capacity planning becomes a tidy arithmetic exercise. Skip it and you are sizing for the load you happen to have today, which is exactly how a fleet that was perfect last quarter becomes an outage this quarter.

![A vertical stack decomposing future demand into organic trend at thirty percent per year, daily and weekly seasonality, known events like a launch or Black Friday at five times normal, and unpredictable black swans, all feeding a single rule to plan for the ninety-ninth percentile of demand rather than the mean](/imgs/blogs/capacity-planning-and-forecasting-2.png)

Demand is not one thing; it is a sum of components with very different shapes, and you forecast each separately because each behaves differently and each fails differently. The decomposition in the figure above is the working model:

**Organic trend** is the slow, steady growth that comes from the business itself succeeding — more users, more usage per user, more features that each make a few more calls. It is the easiest to forecast because it is the most regular: fit a line (or an exponential if growth compounds) to your historical traffic and extend it. A service growing 30% year over year is growing about 2.2% per month, which compounds; "30% a year" does not mean "add 30% on the last day of December," it means the curve is climbing under you every single week.

**Seasonality** is the repeating cycle: the daily peak when your users are awake, the weekly rhythm where weekdays differ from weekends, the yearly pattern where retail spikes in November and a tax service spikes in April. Seasonality matters enormously because *you size for the peak of the cycle, not its average.* A service whose daily peak is 3× its daily mean must be sized for the peak; the mean is a comforting number that describes a moment of the day when nothing is at risk.

**Inorganic spikes** are the step-changes that don't come from organic growth at all: a product launch, a marketing campaign that lands, Black Friday, a feature that goes viral, a TV mention, an app-store feature placement — or the one nobody plans for, a competitor's outage routing their users to you in a wave. These are the dangerous ones because they don't show up in your historical trend line. They are discontinuities, and a trend-fit model is blind to them by construction.

**Black swans** are the spikes you genuinely cannot predict in timing or size. You cannot forecast them. What you *can* do is size your headroom and your scale-up speed so that a sudden multiple of normal load is survivable for long enough to react. That is a headroom decision, not a forecast.

### Time-series forecasting, in just enough depth

The honest version of forecasting decomposes a historical traffic series into trend + seasonality + residual noise, projects the trend and seasonality forward, and quantifies the uncertainty as a confidence interval. You do not need a PhD or an exotic model. A few weeks of per-minute request counts and a classical decomposition gets you most of the value. Here is a compact, real forecaster using `statsmodels`-style seasonal decomposition and a Holt-Winters exponential smoothing fit — the kind of script you actually run, not pseudocode:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# daily peak RPS for the last 12 weeks (one value per day)
peaks = pd.Series(history_daily_peak_rps, index=pd.date_range(end="today", periods=84, freq="D"))

# Holt-Winters: additive trend + weekly seasonality (period = 7)
model = ExponentialSmoothing(
    peaks, trend="add", seasonal="add", seasonal_periods=7
).fit()

# forecast the next 90 days of daily peaks
fc = model.forecast(90)

# residual standard deviation -> a crude but honest confidence band
resid_std = np.std(model.resid)
upper_95 = fc + 1.96 * resid_std        # plan to roughly here, not to fc

print(f"forecast peak in 90 days (mean):  {fc.iloc[-1]:,.0f} rps")
print(f"forecast peak in 90 days (p97.5): {upper_95.iloc[-1]:,.0f} rps")
```

The two printed numbers are the whole point. The *mean* forecast is the line through the middle of the cloud of possibility. The *upper bound* is the number you actually size for. **You plan for the p99 of demand, not the mean of demand** — because a fleet sized for the mean is, by definition, under-provisioned roughly half the time, and the half it is under-provisioned for is precisely the busy half. The confidence interval is not academic decoration; it is the difference between "we expect about 12,000 rps" and "there is a real chance of 15,500 rps and we need to survive that." A capacity plan that quotes only the mean is a plan to be on the cliff every other peak.

There is a subtlety worth stating because it trips people up. The right percentile to plan for depends on your tolerance for being wrong, which ties straight back to your SLO and error budget. If a brief saturation event costs you a chunk of error budget you cannot afford, plan to the p99 or higher of demand. If you have autoscaling that reacts in seconds and a generous budget, you can plan closer to the mean and let elasticity absorb the rest. The forecast percentile and the headroom percentage are two dials on the same machine, and Section 4 connects them.

### Organic versus inorganic: forecast one, plan around the other

The single most important distinction in forecasting is between organic growth — the trend-and-seasonality part your model can predict from history — and inorganic events, which it cannot. They demand completely different responses, and conflating them is how teams get the math right and still get surprised.

Organic growth you *forecast*. You fit the model, project the trend and seasonality, read off the p99, and size the standing fleet to that. The model works because the future looks like the past plus a slope; next Tuesday's peak is genuinely predictable from the last twelve Tuesdays. This is the bread-and-butter of capacity planning and it is where a Holt-Winters or Prophet-style decomposition earns its keep.

Inorganic events you *plan around*, because they are discontinuities the model is structurally blind to. A trend line fit to last quarter's traffic has no way to know that marketing is launching a campaign next Thursday, or that your biggest competitor is about to have a multi-hour outage that sends their users to you. The defense is not a better model — it is a *process*: a shared calendar of known events (launches, sales, marketing pushes, partner go-lives) that capacity planning reviews on a cadence, plus a standing spike buffer for the events you cannot calendar. The calendar handles the known unknowns; the spike buffer handles the unknown unknowns. A capacity plan that has a beautiful forecast but no event calendar will be blindsided by the first launch, every time, because the launch is not in the history the forecast was built from.

#### Worked example: the calendar event the forecast missed

A media-streaming service had a genuinely good organic forecast — Holt-Winters, weekly seasonality, a confidence band, the works. Steady-state peak was 8,000 rps, forecast to grow to 9,500 rps over the year, and the fleet was sized correctly for the p99 of *that*. Then a major sporting final landed on their platform via a last-minute rights deal. The event was on a public calendar months in advance, but it was on the *business development* team's calendar, not the SRE one, and the two never met. Traffic that evening hit 24,000 rps — 3× the forecast peak — for a four-hour window. The forecast was not wrong; it had simply never been told the event existed. The fleet, sized at p99 of 9,500 rps with normal headroom, topped out around 13,000 rps of usable capacity and spent most of the match on the cliff. The fix in the postmortem was not a better model. It was one line: *capacity planning gets read access to every team's event calendar, and reviews it monthly.* The forecast handles the slope; the calendar handles the steps. You need both, and the cheap one — the calendar — is the one teams forget.

## 3. Load testing: measure your capacity, don't guess it

Here is the question that exposes most capacity plans as fiction: *how many requests per second can one of your instances actually serve before it hits the knee?* If the answer is "I think around a thousand?" you do not have a capacity plan, you have a hope. The per-unit capacity number is the foundation everything else is built on — the forecast tells you the demand, but you cannot turn demand into a fleet size without knowing what one unit can carry. And you cannot know that by reading the code or by intuition. You have to **measure it**, by deliberately driving a single instance up its saturation curve until you watch it bend.

![A matrix laying out the four load tests as rows with the question each answers, what it finds, and the risk if skipped as columns, showing load test finding the per-unit knee at eight hundred requests per second, stress test finding cliff behavior, soak test finding slow degradation, and spike test finding scale-up lag](/imgs/blogs/capacity-planning-and-forecasting-3.png)

There are four distinct tests, and they answer four different questions. People conflate them and then are surprised when "the load test passed" did not predict the 3am outage — because the load test answers a different question than the soak test does. The matrix above lays them out:

- **Load test** ramps traffic steadily upward and watches for the knee. This is the test that gives you per-unit capacity: the requests-per-second at which latency starts climbing nonlinearly. You ramp until the curve bends, note the rps at the bend, and that is your per-unit number. Critically, the knee is *not* the point where the instance falls over — it is the point where it starts to degrade. You size to stay comfortably below the knee, not below the crash.
- **Stress test** keeps pushing *past* the limit to see how the system behaves when overloaded. Does it shed load gracefully and keep serving a reduced rate, or does it collapse entirely — fall into a death spiral where retries pile on, the queue grows unbounded, and throughput actually *decreases* under increasing load? The latter is called congestion collapse, and a system that does it needs load shedding (cross-link: the system-design treatment of [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) covers the shedding mechanisms). You want to know which one you have *before* a real overload tells you.
- **Soak test** runs at a sustained, realistic load for hours — overnight, ideally — to find the failures that only show up over time: memory leaks that slowly eat the heap until an OOM kill, file-descriptor leaks, log volumes that fill a disk, connection pools that fragment, caches that grow without bound. A system can pass a 20-minute load test beautifully and still OOM-crash at 4am after eight hours at the same load. The soak test is the only one that finds that, and it is the one teams skip most often because it is slow and boring.
- **Spike test** slams a sudden multiple of normal load on instantly, with no ramp, to see whether your scaling reacts fast enough. This is the test that catches the autoscaling lag problem we will dissect in Section 6 — the gap between "load arrives" and "new capacity is serving." A spike test is the closest thing to a Black Friday rehearsal you can run on a Tuesday.

### Synthetic load is not real traffic

A warning that has burned me personally: a load test that hammers one endpoint with identical requests will overstate your capacity, sometimes wildly. Real traffic has *shape*. It hits many endpoints in a realistic mix, with realistic cache hit ratios, realistic payload sizes, realistic think-time between requests from the same user, and a realistic distribution of cheap reads versus expensive writes. A synthetic test that sends 10,000 identical GET requests to a heavily-cached endpoint will report a per-unit capacity that evaporates the moment real users send their messy mix of cache-missing, write-heavy, fan-out-triggering requests. **Test the traffic shape you will actually receive**, ideally by replaying a captured sample of production traffic, or at minimum by weighting your synthetic mix to match your real endpoint distribution.

Here is a `k6` load test that ramps to find the knee using a realistic endpoint mix — the kind you commit to the repo and run in CI before a known event:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const p99 = new Trend('endpoint_latency', true);

export const options = {
  scenarios: {
    ramp_to_knee: {
      executor: 'ramping-arrival-rate',   // model arrivals, not VUs
      startRate: 100,
      timeUnit: '1s',
      preAllocatedVUs: 500,
      stages: [
        { target: 400, duration: '2m' },   // warm up
        { target: 600, duration: '2m' },
        { target: 800, duration: '2m' },   // expect the knee around here
        { target: 1000, duration: '2m' },  // push just past it
      ],
    },
  },
  thresholds: {
    // the knee is where p99 crosses our SLO budget
    endpoint_latency: ['p(99)<300'],
  },
};

export default function () {
  // realistic mix: 70% reads, 25% search, 5% writes
  const r = Math.random();
  let res;
  if (r < 0.70)      res = http.get('https://staging.api/v1/orders/123');
  else if (r < 0.95) res = http.get('https://staging.api/v1/search?q=widget');
  else               res = http.post('https://staging.api/v1/orders', '{"sku":"x"}');
  p99.add(res.timings.duration);
  check(res, { 'status 2xx': (x) => x.status < 300 });
  sleep(Math.random() * 0.5);  // think-time, so we model arrivals not a hammer
}
```

The `ramping-arrival-rate` executor matters: it models a fixed *arrival rate* (requests per second arriving regardless of how the system responds), which is how real load behaves, rather than a fixed number of virtual users that politely wait for slow responses and thereby hide the cliff. When you run this against a single instance and chart p99 latency against offered rps, the knee jumps out of the chart — the rps where the latency line bends from flat to vertical is your per-unit capacity.

#### Worked example: establishing per-instance capacity from a load test

Run the ramp above against one instance and you get a table like this:

| Offered load | p50 latency | p99 latency | On the cliff? |
| --- | --- | --- | --- |
| 400 rps | 45 ms | 95 ms | No, flat |
| 600 rps | 52 ms | 130 ms | No, flat |
| 750 rps | 68 ms | 220 ms | Approaching knee |
| 800 rps | 90 ms | 290 ms | At the knee |
| 850 rps | 180 ms | 700 ms | Past the knee |
| 900 rps | 600 ms | 3,400 ms | Off the cliff |

Read it carefully. The instance does not *crash* until somewhere past 900 rps. But your SLO says p99 must stay under 300 ms, and that line is crossed right around 800 rps. **Your per-unit capacity is 800 rps, not 900** — capacity is defined by your SLO, not by the crash point. The 100 rps between the knee and the crash is the unusable last-10% we proved with queueing theory in Section 1; it exists on the chart but it costs you your latency SLO to use it. Write down 800 rps as the number, because the entire fleet-sizing calculation in the next section divides by it.

### Where the load test should run, and what it must include

Two operational notes that separate a load test you can trust from one that lies to you. First, **test an instance that is configured exactly like production** — same instance type, same JVM or runtime flags, same connection-pool sizes, same sidecars, same resource limits. A load test against a beefier dev box, or against a pod without the production memory limit, measures a capacity you will not have when it counts. The number is only useful if the thing you measured matches the thing you'll deploy. Second, **isolate the single instance** so you are measuring one unit's capacity, not the fleet's load balancer. Point the load generator at one pod directly (or pin it with a header-based route) so the rps you offer is the rps that pod receives. If you load-test through the balancer you are measuring the balancer's spreading behavior, not the per-unit knee, and the per-unit knee is the number the formula needs.

The other thing a real load test must include is the *downstream*. When that single instance is pushed to 800 rps, what is it doing to the database, the cache, the downstream APIs? In my payments outage the per-instance web capacity was fine — it was the *shared* database connection pool that saturated, and a single-instance load test that talked to a dedicated test database would never have revealed it, because the contention only appears when the whole fleet competes for the same shared pool. So you run two tests: the single-instance test to find the per-unit web knee, and a *fleet-scale* test against the shared production-shaped downstream to find where the shared resource saturates. The fleet capacity is the *lower* of the two. It does not matter that one instance can do 800 rps if 20 instances collectively exhaust the database pool at an aggregate of 9,000 rps — then 9,000 is your real ceiling, and you've found a downstream cliff that no per-unit number would have shown.

### Make load testing a habit, not a heroic one-off

The load test that prevents an outage is the one you ran *last week*, not the one you scramble to run during the incident. The teams that never get surprised by capacity treat load testing as routine: a single-instance ramp test in CI on every significant change (so a code change that halves per-unit capacity gets caught in review, not in production), a fleet-scale test before every known high-traffic event, and a periodic soak test on a schedule to catch the slow leaks. The chaos-engineering discipline overlaps here — a *game day* where you deliberately remove an instance at simulated peak and watch whether the N+1 math holds in reality is a load test wearing a different hat. Here is a minimal chaos experiment, in the style of a Chaos Mesh `PodChaos` spec, that validates the N+1 redundancy assumption under load:

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: validate-n-plus-one-at-peak
spec:
  action: pod-kill           # remove one instance...
  mode: one                  # ...exactly one
  selector:
    namespaces: [checkout]
    labelSelectors:
      app: checkout-api
  duration: "10m"            # run while a load test holds the fleet at peak
  # success criterion (checked out-of-band): p99 stays under the SLO
  # with one fewer instance, proving the headroom math is real
```

You run that *while* a load generator holds the fleet at its forecast peak. If p99 stays under your SLO with one instance gone, your N+1 headroom is real. If it doesn't, your fleet-sizing math was optimistic and you've learned it on a game day instead of at 3am. This is the difference between a capacity plan that is a spreadsheet and one that is *tested* — and tested capacity is the only capacity you can actually rely on.

## 4. Headroom and the fleet-sizing formula

Now we have the two numbers everything hinges on: the forecast peak demand (from Section 2) and the per-unit capacity (from Section 3). Turning those into a fleet size is arithmetic, but it is arithmetic with two crucial adjustments — headroom and redundancy — that the naive division misses, and missing them is exactly how you end up sized correctly for the average and on the cliff at the peak.

![A vertical stack building the fleet size from peak demand of twelve thousand requests per second divided by the per-unit knee of eight hundred to a base need of fifteen instances, then dividing by seventy percent target utilization to reach twenty-two for headroom, then adding one for N plus one redundancy to reach twenty-three instances](/imgs/blogs/capacity-planning-and-forecasting-4.png)

The naive version is just demand over per-unit capacity. If peak demand is 12,000 rps and one instance handles 800 rps, then you "need" 12,000 / 800 = 15 instances. That number is wrong in two directions, and both directions are dangerous.

**First adjustment: headroom for the cliff.** If you run 15 instances at 12,000 rps, each instance is at exactly its knee — 800 rps each — which means the *whole fleet* is sitting on the saturation cliff at peak. Any blip above the forecast, any instance that's a little slower than the others, any GC pause that shifts load to its neighbors, and you bend off the knee fleet-wide. You do not run at the knee; you run at a target utilization *below* it. If your knee is at 800 rps and you target 70% utilization, your *effective* per-unit capacity for planning is 800 × 0.70 = 560 rps. The headroom-adjusted fleet is 12,000 / 560 ≈ 21.4, round up to 22. Those extra 7 instances are not waste — they are the gap between the knee and your operating point, the margin that keeps normal peaks from reaching the cliff.

The formula, written out, is:

$$N_{\text{instances}} = \left\lceil \frac{D_{\text{peak}}}{C_{\text{unit}} \times U_{\text{target}}} \right\rceil + R$$

where $D_{\text{peak}}$ is forecast peak demand, $C_{\text{unit}}$ is per-unit capacity at the knee, $U_{\text{target}}$ is your target utilization (the headroom dial, typically 0.6–0.75), and $R$ is the redundancy term.

**Second adjustment: redundancy for unit loss.** A fleet sized to exactly carry peak load has *no spare unit* — so the moment one instance dies (a node failure, a deploy that takes one out, a hardware fault), the remaining instances must each carry more, which can push them over the knee, which is a cascading failure that starts from a single routine instance death at peak. The fix is **N+1 redundancy**: size the fleet so that with one instance gone, the survivors are still below the knee. At 22 instances, losing one drops you to 21, and 12,000 / 21 = 571 rps per instance — still under the 800 knee, so we are fine; in this case N+1 is satisfied by adding just one buffer instance to make the loss comfortable, giving 23. If you need to survive losing an entire availability zone (say a third of your fleet at once), that is **N+2** thinking or zone-aware over-provisioning, and the redundancy term grows accordingly — you size so that two-thirds of the fleet can carry the full peak.

The figure traces exactly this build-up: 12,000 rps demand, divided by the 800 rps knee gives a base of 15, divided by the 70% target gives 22 for headroom, plus the N+1 buffer gives 23. The point of drawing it as a stack is that **each layer is a deliberate decision with a reason**, not a fudge factor. Headroom is the cliff margin. Redundancy is the unit-loss margin. Skip either and you have sized for a world that does not have peaks or failures in it.

### How much headroom? Tie it to your provisioning lead time

The target utilization dial — how much headroom to hold — is not a universal constant. It depends on one thing above all others: **how long it takes you to add capacity.** This is the rule most teams get wrong.

If provisioning a new instance takes a week (you have to order hardware, or your scaling involves a change-management approval, or your image bakes for hours), then your headroom must cover a *week's worth of growth plus a week's worth of possible spike*, because that is how long you are stuck with the fleet you have. If autoscaling can add a warm instance in 30 seconds, you need far less standing headroom because elasticity covers the gap — you only need enough headroom to survive the scale-up lag, which we will see in Section 6 can still be longer than a fast spike. The principle:

> Your standing headroom must cover the demand growth that can occur within your provisioning lead time, plus a spike buffer.

This is why "what's the right utilization target?" has no single answer. A fleet of bare-metal database servers with a multi-week procurement cycle might run at 40% utilization because it cannot react quickly. A stateless web tier on Kubernetes with sub-minute autoscaling might run at 70%+ because it can react. Same company, same SLO, completely different headroom — because the lead times differ. The cost section (Section 8) makes the money side of this dial explicit, but the engineering rule is: **headroom is insurance against the time you cannot react, so price it against your reaction time.**

#### Worked example: sizing a fleet from a load test, with a 12-month re-plan trigger

Let me run the whole calculation end to end with the numbers from the figure, then add the part most plans forget — the trigger that tells you when to redo it.

Inputs: per-unit capacity from the load test is **800 rps at the knee**. Current forecast peak demand is **12,000 rps**. Organic growth is **30% per year**. Target utilization is **70%** (stateless web tier, fast autoscaling, so we can run warm). We want **N+1** at peak.

Sizing today: effective per-unit = 800 × 0.70 = 560 rps. Base fleet = ⌈12,000 / 560⌉ = ⌈21.4⌉ = 22. With one instance lost, 12,000 / 21 = 571 rps each, just under the knee, so we add one buffer instance for comfortable N+1: **23 instances** is the provisioned fleet. (Round the spike buffer up, never down — fractional instances do not exist and the cliff is unforgiving.)

Now project forward. With 30% annual growth, in 12 months peak demand becomes 12,000 × 1.30 = 15,600 rps. Re-running the formula: ⌈15,600 / 560⌉ = ⌈27.9⌉ = 28, plus N+1 buffer = **29 instances** a year out. So the plan is not a single number; it is a trajectory: 23 now, ~26 at six months (12,000 × 1.14 ≈ 13,700 rps → 25 + 1), 29 at twelve months.

The re-plan trigger is the safety net. Set an alert that fires when sustained peak utilization crosses 70% — meaning the fleet is approaching the operating point you sized it for and the buffer is being eaten by growth faster than the plan assumed. When that alert fires, you re-run the forecast and the formula. **A capacity plan without a re-plan trigger is a snapshot, and snapshots go stale.** The trigger turns it into a living loop. Here is the forecast table that comes out of this, the artifact you actually hand to the team and the budget owner:

| Horizon | Forecast peak (rps) | Effective unit cap (rps) | Base instances | +N+1 | Provision by |
| --- | --- | --- | --- | --- | --- |
| Now | 12,000 | 560 | 22 | 23 | provisioned |
| +6 months | 13,700 | 560 | 25 | 26 | 5 months out |
| +12 months | 15,600 | 560 | 28 | 29 | 11 months out |
| Re-plan trigger | sustained util > 70% | — | — | — | re-forecast now |

## 5. The holiday spike that wasn't planned: a war story in two timelines

Let me tell this one as a paired story, because the contrast is the whole lesson, and because I have lived both sides of it. The figure shows the two outcomes side by side: the fleet sized for the mean that hit the cliff, and the fleet that was pre-scaled and load-tested and rode the same surge out flat.

![A before and after comparison showing on the left a fleet of fifteen instances sized for the mean hit by five times traffic that saturates to one hundred percent utilization and sends latency from one hundred milliseconds to eight seconds in an outage, and on the right a pre-scaled fleet of ninety instances load-tested to six times that holds latency flat at one hundred twenty milliseconds with zero errors](/imgs/blogs/capacity-planning-and-forecasting-5.png)

**Timeline one — the fleet sized for the mean.** A retail checkout service ran comfortably all year at a fleet sized around its daily average, with a bit of slack for the daily peak. The capacity "plan," such as it was, looked at the average request rate, saw plenty of margin, and called it done. Nobody decomposed demand into trend plus seasonality plus events. Nobody noticed that the marketing team had committed to a Black Friday push that historically drove 5× normal traffic. The fleet was 15 instances. On the morning of the sale, traffic ramped from the usual ~3,000 rps toward 15,000 rps over about twenty minutes. The fleet hit its knee around 12,000 rps. Past that, the queueing-theory cliff did exactly what the math says it must: p99 latency went from 100 ms to 800 ms to 3 s to 8 s. Checkout requests timed out. Frustrated users retried, which *added* load — the retry storm that turns a saturation event into a death spiral. The autoscaler was configured but its cooldown and warm-up meant new instances took minutes to help, and by then the database connection pool behind the fleet had saturated too, so the new instances came up and immediately starved. The site was effectively down for the most lucrative two hours of the year. The postmortem's root cause was one line: *the fleet was sized for the mean, and Black Friday is not the mean.*

**Timeline two — the fleet that was planned.** Same service, different year, after the lesson landed. Six weeks before the event, the team pulled the previous year's traffic and the marketing calendar, forecast a 5× peak, and added margin for the forecast being wrong — they planned to survive **6×**, not 5×, because the confidence interval on a marketing-driven spike is wide. They load-tested a single instance to confirm the per-unit knee (it was 800 rps, just like our running example), ran a spike test to confirm the autoscaler reacted fast enough, ran a soak test overnight at the projected sustained load to confirm nothing leaked. They sized the fleet for 6× — roughly 18,000 rps of headroom-adjusted capacity, about 90 instances — and crucially they **pre-scaled**: they did not trust the autoscaler to react fast enough to a near-instant surge, so they scheduled the fleet up to its event size before the sale started and scheduled it back down after. They also load-tested the *database* path and bumped its connection pool ceiling so the downstream would not become the new cliff. On the day, traffic surged to 5×. The fleet, sized and warmed for 6×, sat at a comfortable utilization the whole time. p99 latency held flat at ~120 ms. Zero customer-facing errors. The most lucrative two hours of the year went off without a single page.

Same surge. Same code. The only differences were a forecast, a load test, headroom for the forecast being wrong, and the decision to pre-scale rather than trust elasticity against a fast spike. **That is the entire discipline, demonstrated.** The cost of the planned version was real — 90 warm instances for a day or two costs money, which we will address head-on in Section 8 — but it was a tiny fraction of the revenue lost in the unplanned version, and it bought certainty on the day that mattered most.

The stress-test reasoning that the second team did and the first did not is worth making explicit, because it generalizes. They did not just plan for the expected peak; they asked the adversarial questions. *What if the spike is bigger than forecast?* (Plan for 6×, not 5×.) *What if an instance dies mid-event?* (N+1, sized in.) *What if the bottleneck moves downstream when we scale the web tier?* (Load-test the DB path too, raise its ceiling.) *What if the autoscaler is too slow for the ramp?* (Pre-scale, don't trust elasticity against a step.) Every one of those questions corresponds to a way the first team got surprised. Capacity planning is, in large part, the practice of getting surprised on paper, in advance, where it is cheap.

## 6. Autoscaling: dynamic capacity and the four ways it betrays you

Autoscaling is the most attractive idea in capacity planning: instead of standing up a big fleet and paying for headroom you mostly don't use, let the system add and remove capacity automatically as load changes. When it works, it is genuinely wonderful — it turns headroom from a fixed cost into an elastic one, it absorbs the daily seasonality without anyone touching anything, and it ties straight into the self-healing systems discipline (a sibling post, planned under the slug `self-healing-systems-and-their-traps`, goes deep on automated remediation and its failure modes; until it ships, the short version lives here). But autoscaling is not magic capacity, and treating it as a substitute for planning is one of the most common and most dangerous mistakes I see. It betrays you in four specific ways, and a planner has to know all four.

![A flow showing a six times traffic spike in thirty seconds hitting an autoscaler that adds web pods, but the scale-up lag of ninety seconds is too slow while the added clients overwhelm a fixed database of two hundred connections that exhausts the pool and explodes latency, with both failure paths leading to the fix of pre-scaling plus load shedding](/imgs/blogs/capacity-planning-and-forecasting-7.png)

**Betrayal one: scale-up is slower than the spike.** Autoscaling has a reaction time. The control loop has to *observe* high utilization (a metric scrape interval, often 15–60 seconds), *decide* to scale (a stabilization window to avoid flapping), *provision* the instance (boot, pull images, warm caches, pass health checks — often 60–120 seconds for a real service), and only then does the new capacity start serving. Add it up and the realistic gap between "load spikes" and "new capacity helps" is one to three minutes. A spike that goes 6× in 30 seconds — a viral moment, a flash sale, a competitor's outage redirecting traffic — outruns the autoscaler entirely. By the time new instances are serving, you have already spent two minutes on the cliff, which is two minutes of outage. **This is exactly why the second war-story team pre-scaled.** For a known event, you scale up *before* it on a schedule; you do not gamble that elasticity will win a race against a step function. The spike test from Section 3 exists precisely to measure this gap so you know whether you can trust the autoscaler or must pre-scale.

**Betrayal two: the bottleneck is downstream and fixed.** This is the one that took down my payments API, and it is the one in the figure. Your web tier autoscales beautifully — Kubernetes adds pods, they pass health checks, they start serving. But every one of those new pods opens connections to a database whose connection pool has a *fixed* ceiling of 200. The autoscaler has done its job perfectly and made the problem *worse*, because now there are more clients competing for the same 200 connections, the pool saturates, and every pod — old and new — is blocked waiting for a connection. **Autoscaling only helps above the layer that scales.** If your true bottleneck is a fixed-capacity downstream — a database, a third-party API with a rate limit, a license-limited service, a single-writer queue — then adding stateless capacity in front of it does nothing but pile more pressure on the chokepoint. You have to identify the binding constraint (the database connection pool, the downstream rate limit) and either scale *it* or protect it with backpressure and load shedding. The DB replication and connection-pooling side of this lives in the database series; here the planning lesson is: **find the constraint that does not scale, because that is your real capacity ceiling, not the web tier you can grow effortlessly.**

**Betrayal three: scaling has a cost and a limit you forgot to raise.** Cloud accounts have quotas — instance limits, IP limits, API rate limits on the scaling actions themselves. I have watched an autoscaler hit a regional vCPU quota mid-spike and simply stop scaling, silently, while traffic kept climbing. The autoscaler "worked"; the quota was the cliff. Before you rely on elasticity for an event, *check that your account limits are above the fleet size you'll need*, and request increases in advance — quota increases can take days to approve. This is a planning task, not a runtime one.

**Betrayal four: it scales down at the wrong moment.** Aggressive scale-down saves money but can remove capacity right before a recurring peak, forcing a scale-up race every single day. Worse, during a partial outage where a fraction of requests are failing fast, the *measured* load can look low (failed requests are cheap), tricking the autoscaler into scaling *down* exactly when you are about to recover and need the capacity back. Scale-down should be slow and conservative; scale-up should be fast — asymmetric by design, because the cost of scaling up too eagerly is a few extra instance-minutes, while the cost of scaling down too eagerly is an outage.

#### Worked example: does the autoscaler win the race?

Put real numbers on betrayal one so you can decide whether to trust elasticity or pre-scale. Say a flash sale will drive a step from 4,000 rps to 20,000 rps — a 5× spike — over roughly 60 seconds. Your autoscaler is a Kubernetes HPA with a 15-second metrics scrape, a 60-second stabilization window before it acts on a scale-up, and pods that take 90 seconds to boot, warm their caches, and pass readiness. Add those up: the autoscaler does not even *decide* to scale for ~75 seconds after the load arrives, and the new pods are not serving until ~165 seconds after that. So for the first roughly two and a half minutes of the spike, you are running on the fleet you started with. If that starting fleet was sized for 4,000 rps with a little headroom — say it tops out at 6,000 rps of usable capacity — then for two and a half minutes you are trying to serve 20,000 rps through a 6,000 rps fleet. That is not "a bit slow." That is the cliff, for 150 seconds, which at a 400× burn rate (Section 7) is enough to vaporize a month of error budget several times over. The conclusion is forced: **for a step spike that outruns your scale-up time, you cannot trust the autoscaler — you pre-scale.** You schedule the fleet up to its event size before the sale opens, and you let the autoscaler manage the gentle ups and downs *around* that floor, not the step itself. The spike test exists to give you exactly these three numbers — scrape interval, stabilization window, boot time — so you can do this arithmetic before the event instead of discovering it during one.

Here is the scheduled pre-scale, expressed as a Kubernetes CronJob that raises the HPA floor before the event and lowers it after — the artifact that turns "we'll pre-scale" from a Slack message into something that actually happens:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: prescale-checkout-for-sale
spec:
  schedule: "30 8 * * 5"     # 08:30 Friday, 30 min before the 09:00 sale
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: prescale
              image: bitnami/kubectl
              command:
                - /bin/sh
                - -c
                - |
                  # raise the autoscaler's minimum to the event-sized fleet
                  kubectl patch hpa checkout-api -n checkout \
                    --type merge \
                    -p '{"spec":{"minReplicas":90}}'
```

A matching CronJob scheduled for after the sale patches `minReplicas` back down to the everyday floor, so you stop paying for 90 warm instances the moment the event is over. The autoscaler still operates above that floor for the unexpected, but the floor itself guarantees the capacity is *already warm* when the step arrives — no race to lose.

The unifying lesson across all four: **autoscaling is dynamic headroom, not infinite capacity, and it has a reaction time and a ceiling.** It reduces how much standing headroom you must hold, but it does not eliminate the need to know your forecast, your per-unit capacity, and your downstream constraints. A capacity plan that says "we'll just autoscale" is not a plan; it is a deferral of the planning to the worst possible moment — mid-incident. Plan the fleet, use autoscaling to ride the seasonality efficiently, pre-scale for known step-changes, and protect the fixed downstream with backpressure. The architecture of the autoscaling control loop itself — target tracking, predictive scaling, the math of stabilization windows — is the system-design post on [capacity planning and autoscaling](/blog/software-development/system-design/capacity-planning-and-autoscaling); this post is about how to *operate* with it without being betrayed.

## 7. Capacity as a reliability signal: saturation, SLOs, and the error budget

Everything so far connects to the series' spine through one idea: **capacity is not a separate concern from reliability — it is one of the four golden signals, and running out of it is one of the fastest ways to burn an error budget.** Let me make that connection precise, because it is what turns capacity planning from an infrastructure chore into an SRE discipline.

![A left-to-right timeline of the capacity planning loop running from load testing to find the knee, to forecasting demand with trend and spikes, to sizing the fleet with headroom and N plus one, through the one-week provisioning lead buffer, to watching saturation as a golden signal, ending at the re-plan trigger that fires at seventy percent headroom](/imgs/blogs/capacity-planning-and-forecasting-6.png)

Recall the four golden signals: latency, traffic, errors, and **saturation**. Saturation is the odd one out because it is not directly a measure of user experience — a user does not feel "the connection pool is 90% full." But saturation is the *leading* signal: it is the one that goes bad *first*, before latency climbs and before errors appear. By the time latency and errors degrade, you are already on the cliff. By the time saturation crosses your knee, you have a head start. That is why saturation earns a place among the golden four — it is your early-warning system for the capacity outage, and watching it is how you turn a 3am page into a Tuesday-afternoon re-plan. The figure traces the full loop: you load-test to find the knee, forecast demand, size with headroom, account for provisioning lead, **watch saturation as the golden signal**, and re-plan when it crosses the trigger.

### Saturation as an SLI input, and the budget connection

Here is where it ties to the error budget — the currency of the whole series. Your availability SLO gives you an error budget: at 99.9% you have 0.1% of requests, or about 43.2 minutes per month, that are allowed to be bad. A saturation event spends that budget *fast*, because when you go off the cliff you do not lose 0.1% of requests — you lose a large fraction of them for the duration of the saturation, with elevated latency that may itself violate a latency SLO. The arithmetic is worth doing.

#### Worked example: how a saturation event burns the error budget

Suppose your SLO is 99.9% availability over 30 days, giving 43.2 minutes of error budget. Now suppose a capacity miss puts you on the cliff during a peak, and for that period 40% of requests fail or violate the latency SLO. The *burn rate* — how fast you're spending budget relative to the steady allowance — is the observed bad-event rate divided by the SLO's allowed bad-event rate:

$$\text{burn rate} = \frac{\text{observed bad fraction}}{1 - \text{SLO}} = \frac{0.40}{0.001} = 400\times$$

A 400× burn rate means you are spending budget 400 times faster than the SLO permits. Your entire 30-day budget of 43.2 minutes is gone in 43.2 / 400 ≈ **0.108 minutes, about 6.5 seconds of real time** at that intensity. In practice a saturation event lasts not seconds but minutes — the 8-second-latency outage in the war story ran for two hours. **A single unplanned capacity event can blow a month's entire error budget many times over.** That is the budget argument for headroom in one calculation: the spare capacity you "waste" by running at 70% instead of 95% is cheap insurance against an event that can vaporize a month of budget in minutes. The multi-window burn-rate alerting that catches this — fast burn over a short window plus slow burn over a long one — is the sibling discipline covered in the alerting posts, and capacity is one of its most important triggers.

This is also the cleanest way to *justify* a headroom budget to a skeptical finance partner: headroom is not idle waste, it is the cost of the error budget you are choosing to protect. How much capacity buys the SLO is the real FinOps question, and it is answerable in dollars and nines rather than vibes. A fleet running at 95% utilization is cheaper on the invoice and ruinous on the budget; a fleet at 70% costs more on the invoice and protects the budget. **The right headroom is the one where the marginal cloud dollar equals the marginal error-budget dollar** — and once you frame it that way, "how much spare capacity?" becomes a business decision with a number, exactly like the SLO target itself (the sibling post on [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) makes the parallel argument for the SLO target, and [monitor the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server) makes the case for measuring saturation in terms of user-felt pain rather than raw resource gauges).

## 8. The cost dimension: capacity is money, and headroom has a price

Capacity is money. Every instance of headroom is an instance you pay for and mostly don't use, and an honest capacity discipline confronts that directly rather than pretending reliability is free. The whole job is to find the right point on a dial between two failure modes: over-provision and you waste spend; under-provision and you risk an outage. This is the FinOps balance, and it is the same shape as the error-budget balance — too much reliability is as much a mistake as too little, just a more expensive and less visible one.

![A matrix comparing running at ninety-five percent utilization with lowest cost but no spike survival and a fast-burning budget, running at seventy percent with balanced spend that rides a moderate spike and meets the SLO, and running at forty percent with double the cost that rides a large spike but overbuys nines](/imgs/blogs/capacity-planning-and-forecasting-8.png)

The matrix above lays the trade-off out at three operating points. Run at 95% utilization and your invoice is as low as it goes, but you are sitting on the cliff with no spike survival and your error budget burns the instant anything moves. Run at 40% and you can ride a 2.5× spike without thinking, but you are paying roughly double for capacity that mostly idles — you have *overbought nines* that your users cannot perceive and your budget does not need. Run at 70% — the balanced point for a fast-autoscaling stateless tier — and you ride a moderate spike, meet a 99.9% SLO, and pay a defensible bill. The right column is which one *survives* what, and the question is always: what do you actually need to survive, given your forecast and your reaction time?

The honest framing of the cost dial:

| Operating point | Cloud cost | Risk profile | When it's right |
| --- | --- | --- | --- |
| Run hot (90%+) | Lowest | On the cliff; one spike from outage | Batch jobs with no latency SLO; cost-critical, failure-tolerant |
| Balanced (~70%) | Moderate | Rides normal peaks; survives N+1 | Most user-facing services with an SLO |
| Conservative (~40%) | High | Rides big spikes; deep redundancy | Can't react fast (long lead time); high blast-radius services |
| Pure autoscale | Variable, often lowest | Great for slow ramps; loses races to fast spikes | Elastic stateless tiers with proven fast scale-up |

Two cost levers reduce the headroom bill without giving up safety, and both are worth knowing. The first is **autoscaling**, which turns standing headroom into elastic headroom — you pay for the big fleet only during the peak, not all night. The second is **tiered capacity**: hold a small amount of expensive, instantly-available headroom (warm instances, on-demand) for fast spikes, and a larger amount of cheap, slower-to-provision headroom (reserved capacity, spot instances with fallback) for sustained growth. You do not need all your headroom to be the expensive instant kind; you need *enough* of it to cover the spike you can't react to, and the rest can be the cheap kind that covers slow growth.

The stress test for the cost dial is the same adversarial questioning as everywhere else, just pointed at the budget. *What if we cut headroom to save 20%?* Then quantify the error-budget risk of the next unplanned spike — if it would blow a month's budget, the 20% saving is not a saving, it is a deferred outage. *What if finance demands we run at 90%?* Then show them the burn-rate arithmetic from Section 7: 90% utilization means the next 1.2× spike puts you on the cliff, and one cliff event costs more in lost revenue and burned budget than a year of the headroom they want to cut. **Capacity decisions framed only as cost lead to outages; capacity decisions framed as cost-versus-budget lead to the right number.** The discipline is not to minimize spend or maximize headroom — it is to make the trade-off explicit, in dollars and nines, and then choose deliberately.

## 9. How to reach for this (and when not to)

Capacity planning is a real cost in engineering time and cloud spend, and like every reliability practice it has a point past which it stops paying off. Here is the decisive guidance.

**Do the full discipline — forecast, load test, headroom, re-plan trigger — when:**

- The service is user-facing and has an SLO. The arithmetic linking saturation to error-budget burn (Section 7) makes capacity a first-class reliability concern here.
- Demand has a known peak that's a large multiple of the mean — anything with strong seasonality or scheduled events (retail, ticketing, tax, anything with a launch calendar). The mean will lie to you.
- Provisioning is slow. The longer your lead time, the more you must plan ahead, because you cannot react. A multi-week procurement cycle makes forecasting non-negotiable.
- The bottleneck is a fixed downstream — a database, a license-limited service, a rate-limited third party. You *cannot* autoscale your way out of these, so you must plan for them.

**Don't over-invest when:**

- It's an internal batch job with no latency SLO and generous deadlines. If a job that runs nightly takes 40 minutes instead of 30 because you ran it hot, nobody is harmed. Run those at high utilization and save the money — there is no cliff that hurts a user here.
- The service is genuinely elastic, stateless, and has a *proven* fast scale-up that you've spike-tested, *and* its downstreams scale too. Then lean on autoscaling and hold minimal standing headroom. But "we have autoscaling" is not the same as "we've spike-tested it and it wins the race" — prove it before you trust it.
- The traffic is small, flat, and stable, with no events on the horizon. A service doing a steady 50 rps with no seasonality does not need a Holt-Winters forecast; it needs a saturation alert and a sanity-check that one instance can carry it with room to spare.
- You'd be buying a nine the user can't feel. Section 8's "overbought nines" — don't run at 40% utilization for a service whose users would never notice a brief degradation, just because conservative *feels* safe. Match the headroom to the actual risk, priced against the actual budget.

The meta-rule, the one that generalizes the whole post: **the cost of capacity planning should be proportional to the cost of getting capacity wrong.** A payments API at peak gets the full treatment. A nightly internal report gets a saturation alert and a shrug. Spend your planning effort where the cliff hurts.

## 10. Key takeaways

- **Systems don't degrade linearly — they fall off a cliff.** Queueing theory guarantees it: as utilization $\rho \to 1$, queue length $\rho/(1-\rho) \to \infty$. The last 10% of nominal capacity is unusable, so find your knee and run below it.
- **Forecast demand at the p99, not the mean.** Decompose load into organic trend, seasonality, known events, and unforecastable spikes, project each, and size for the top of the confidence interval — a fleet sized for the mean is under-provisioned every busy peak by construction.
- **Measure per-unit capacity, never guess it.** Load-test one instance to find the knee (the rps where p99 crosses your SLO, *not* the crash point), stress-test for overload behavior, soak-test for slow degradation, spike-test for scale-up lag. Test the real traffic shape, not a synthetic hammer.
- **Fleet size = ⌈peak demand / (per-unit × target utilization)⌉ + redundancy.** Headroom keeps you off the cliff; N+1 (or N+2 for zone loss) survives a unit failure at peak. Both are deliberate decisions, not fudge factors.
- **Size headroom against your provisioning lead time.** If you can't add capacity for a week, hold a week's growth-plus-spike buffer; if autoscaling reacts in seconds, hold less. Headroom is insurance against the time you cannot react.
- **Autoscaling is dynamic headroom, not infinite capacity.** It loses races to fast spikes (pre-scale for known events), can't help when the bottleneck is a fixed downstream, hits account quotas, and scales down at the wrong time. Plan the fleet; use elasticity to ride seasonality efficiently.
- **Saturation is a golden signal and a leading indicator.** It goes bad first, before latency and errors. A single unplanned capacity event can burn a month's error budget many times over — a 40% bad-request cliff is a 400× burn rate.
- **Capacity is money; price headroom against the error budget.** The right operating point is where the marginal cloud dollar equals the marginal error-budget dollar. Match the planning effort to the cost of getting it wrong.

## Further reading

- *Site Reliability Engineering* (the Google SRE Book), the chapter on **Software Engineering in SRE** and the load-balancing/handling-overload chapters — the canonical treatment of saturation, load shedding, and graceful degradation under overload.
- *The Site Reliability Workbook*, the chapter on **managing load** and the alerting-on-SLOs chapter — multi-window burn-rate alerting, which is how you catch a capacity-driven budget burn in time.
- Brendan Gregg's **USE method** (Utilization, Saturation, Errors) — a systematic checklist for finding which resource saturates first; the missing-link between "the system is slow" and "this specific resource is the cliff."
- The classic queueing-theory primers on **Little's Law** and **M/M/1** behavior — enough to understand *why* the cliff is arithmetic, not bad luck.
- Within this series: [reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) for the define→measure→budget→respond→learn loop this fits into; [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) for the SLO target that prices your headroom; and [monitor the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server) for measuring saturation as user-felt pain. The planned sibling on self-healing systems and their traps goes deeper on automated remediation and why autoscaling can betray you.
- System-design (architecture-time companions): [capacity planning and autoscaling](/blog/software-development/system-design/capacity-planning-and-autoscaling) for the autoscaling control loop in depth, and [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) for the load-shedding mechanisms that keep an overload from becoming a congestion collapse.

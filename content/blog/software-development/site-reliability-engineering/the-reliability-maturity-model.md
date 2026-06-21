---
title: "The Reliability Maturity Model: Find the Rung You're On, Then Climb One"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A five-rung ladder from chaotic firefighting to engineered, self-improving reliability — with the telltale symptoms of each level, a seven-dimension self-assessment to find your real position, and the single next practice that unblocks the rung above."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "maturity-model",
    "self-assessment",
    "error-budget",
    "slo",
    "toil",
    "postmortem",
    "reliability",
    "roadmap",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-reliability-maturity-model-1.png"
---

There is a particular kind of meeting that happens after a bad quarter. The third major outage in eight weeks has just been mitigated, the customer-success team is forwarding angry emails, and a director asks the question that has no good answer: "Why does this keep happening, and what do we need to *do* about it?" Around the table, everyone has a favorite answer. The platform engineer wants to buy a chaos-engineering tool. The team lead wants more headcount. The newest hire, who read half of the Google SRE book on the plane, wants to "define some SLOs." The on-call veteran just wants the pager to stop screaming at 3am for a disk that auto-recovers in ninety seconds anyway.

All of them are partly right and all of them are about to waste six months, because they are arguing about *which* practice to adopt without agreeing on *where the team is standing*. Reliability is not a single switch you flip. It is a ladder you climb, and the rung you're on determines which practice will actually help and which will be expensive theater. The team that buys the chaos tool while still answering "is it up?" by refreshing the homepage will learn nothing from breaking things on purpose — they have no steady state to compare against. The team that "defines some SLOs" while alerting on every CPU spike will just have a beautiful number that nobody believes and a pager that still cries wolf.

This post gives you a map: a five-rung **maturity model** for reliability, from Level 0 (reactive firefighting) to Level 4 (self-improving, data-driven reliability). For each rung you'll get the telltale symptoms you can actually observe in your own team, the specific pain that defines it, and — most importantly — the *one or two* practices that move you up to the next rung. Then you'll get a seven-dimension self-assessment rubric you can score your team on in twenty minutes, which almost always reveals that you're lower than you thought and *uneven* across dimensions: strong on deploys, embarrassing on postmortems. Figure 1 is the whole ladder in one frame; pin it somewhere, because everything that follows is just zooming into one rung at a time.

![A vertical ladder of five reliability maturity levels from reactive firefighting at the bottom to self-improving optimization at the top, with the single unlocking practice labeled on each rung transition](/imgs/blogs/the-reliability-maturity-model-1.png)

By the end you'll be able to do three concrete things: place your team on the ladder honestly (per dimension, not as a single flattering number), name the single highest-leverage next move instead of boiling the ocean, and recognize the three anti-patterns — cargo-culting a high-level practice without the foundation, declaring victory too early, and reliability theater — before they eat your quarter. This is the meta-skill the whole series serves: the [intro to the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) tells you reliability is a feature you engineer; this post tells you in what *order* to build it, and where each later technique fits on the climb.

## 1. Why a ladder, and why order matters

Let me be precise about what a maturity model is and is not, because the genre has a bad reputation — and deservedly so. Plenty of "maturity models" are vendor questionnaires designed to make you feel inadequate so you buy the level-5 product. That is not what this is. A useful maturity model makes exactly one claim, and it has to be *true* for the model to earn its place: that the practices have a **dependency order**. You cannot usefully do practice B until practice A is in place, because B consumes the thing A produces.

That dependency is the spine of this whole series. Recall the loop every SRE practice serves: **define reliability (SLI/SLO) → measure it (observability) → spend the error budget → reduce toil → respond to incidents → learn (postmortem) → engineer the fix.** Notice that each step feeds the next. You cannot spend an error budget you haven't defined. You cannot define one without measuring the underlying signal. You cannot measure a signal you have not instrumented. The loop *is* the dependency graph, and the maturity ladder is that same graph drawn vertically, with the levels being the natural plateaus where a set of practices becomes coherent.

Here is the principle stated as plainly as I can: **you climb in order because each rung's practice consumes the previous rung's output.** Error-budget governance (Level 3) is a policy that says "stop shipping features when the budget is exhausted." That policy is meaningless without an error budget, which is `1 − SLO` measured over a window. The SLO (Level 2) is meaningless without an SLI — a signal you actually collect — which requires monitoring (Level 1). And monitoring is meaningless if "the system" is a black box you only know about when a customer phones (Level 0). Skip a rung and the upper practice has nothing to stand on. It does not fail loudly; it fails *quietly*, by producing numbers and rituals that look like the real thing but carry no information.

Let me make the dependency concrete with a tiny piece of arithmetic, because the rest of the post will lean on it. An error budget is the amount of unreliability you are *allowed* to spend:

$$\text{error budget} = 1 - \text{SLO}$$

If your SLO is 99.9% availability over 30 days, your budget is `0.1%` of the time, which is:

$$0.001 \times 30 \times 24 \times 60 = 43.2 \text{ minutes per month}$$

Now ask: how would a Level 0 team even compute that? They have no SLO, so `1 − SLO` is undefined. They have no measured availability, so they can't tell how much of the 43.2 minutes they've spent. The budget is not "hard to manage" at Level 0 — it is *literally uncomputable*. That is what "you can't skip rungs" means in practice. The math has no inputs until the lower rungs are built.

This per-rung dependency is also why the model is honest about *cost*. Each rung asks for a specific, bounded investment, and the payoff is that it unblocks the rung above. You do not adopt every practice in the series at once. You adopt the one practice that your current symptoms say is the bottleneck. The whole point of a maturity model is to stop you from buying the chaos tool when what you need is a dashboard.

Before we zoom into each rung, Figure 2 is the same ladder rendered as a diagnostic table: each level paired with its telltale symptom and the single practice that unblocks the rung above. The rest of the post is mostly an expansion of this one table, so it's worth reading it as a checklist — find the row whose symptom matches your team, and the right-hand column is your next move.

![A five-row maturity ladder table pairing each level with its telltale symptom what hurts about it and the single unlocking practice that advances to the next rung](/imgs/blogs/the-reliability-maturity-model-2.png)

### The series, mapped onto the ladder

It helps to see immediately where each part of this field manual lives. Level 1 is the [metrics and observability](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right) track: you instrument the four golden signals and build dashboards. Level 2 is the [SLI/SLO](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter) track plus the [blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) and humane on-call. Level 3 is the [error budget as governance](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability), [progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery), [toil reduction](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team), and [chaos engineering](/blog/software-development/site-reliability-engineering/chaos-engineering-breaking-on-purpose). Level 4 is [learning from incidents at scale](/blog/software-development/site-reliability-engineering/learning-from-incidents-at-scale) and self-healing. Read this post as the table of contents, sorted by the order you should actually do things.

## 2. Level 0 — Reactive: the firefight that never ends

Every Level 0 team believes it is more mature than it is, and the tell is in the language. People say "we're pretty solid, we just had a bad month." They've had a bad month every month. Reliability at Level 0 is a *property of the people on the team*, not a property of the system — which means it walks out the door when those people quit, and it doesn't scale past their working hours.

### The symptoms you'd observe

You are at Level 0 if you recognize most of this list. The only metric anyone trusts is "is it up?", usually answered by a human loading the homepage. Alerts are configured in one of two failure modes: alerts on *everything* (so the channel is a wall of noise nobody reads, and the real page is buried under forty disk-space warnings that auto-resolve), or alerts on *nothing* (so you find out about outages from Twitter or a customer). A small number of heroes carry the institutional knowledge and the pager, and when the senior on-call is on vacation, incident response degrades visibly. The *same incident recurs* — the database connection pool exhausts itself every few weeks, someone restarts the service, and nobody ever fixes the root cause because there is no mechanism that would make them. Deploys are *scary*; they happen on Friday afternoons against everyone's better judgment, with a rollback plan of "we'll figure it out." Postmortems are either absent or *blameful* — a meeting where someone gets thrown under the bus, which guarantees that the next person to make a mistake will hide it.

### What hurts

The pain at Level 0 is human before it is technical. The heroes burn out. On-call is dreaded, attrition is high among exactly the people you most need to keep, and the team operates in a permanent state of low-grade adrenaline that masquerades as productivity. "We're so busy" is not a sign of health here; it's the symptom. The technical pain is that you cannot make any *promises*. A customer asks "what's your uptime?" and the honest answer is "we don't know." You cannot do capacity planning because you have no baseline. And critically, you cannot *improve*, because improvement requires a measurement to improve against, and you have none.

### The one practice that moves you up

This is the most important rule in the whole post, and it is counterintuitive to a Level 0 team in crisis: **do not start with SLOs, postmortems, or chaos engineering. Start with basic measurement.** The single unlocking practice is *instrument the system so that a machine, not a human, can answer "is it healthy?"* In practice that means the [four golden signals](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — latency, traffic, errors, saturation — emitted as metrics and put on one dashboard. That's it. You are not trying to be sophisticated. You are trying to stop being blind.

Here is the minimum viable instrumentation: an HTTP server that exposes a Prometheus metrics endpoint with a request counter labeled by status, and a latency histogram. This is the foundation literally everything above it stands on.

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["route"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

def handle(method, route, fn):
    start = time.perf_counter()
    status = "500"
    try:
        result = fn()
        status = "200"
        return result
    finally:
        LATENCY.labels(route).observe(time.perf_counter() - start)
        REQUESTS.labels(method, route, status).inc()

if __name__ == "__main__":
    start_http_server(9090)  # scraped by Prometheus
```

With that scraped every fifteen seconds, you can finally write a PromQL query that answers "what fraction of requests are failing right now?" — the single question a Level 0 team cannot answer:

```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))
```

The proof that you've left Level 0 is not a fancy dashboard. It's that the *machine* now knows whether the service is healthy, and a human doesn't have to load the homepage. The first time an alert fires before the customer emails, you've cleared the rung.

#### Worked example: the recurring connection-pool outage

A team I'll keep anonymous had a service that fell over every two to three weeks. The pattern: latency would creep up, then requests would start timing out, then someone would restart the service and "fix" it. Over six months they restarted that service roughly a dozen times. Each restart took about 25 minutes of one engineer's evening, plus the customer impact. Twelve incidents at, conservatively, \$3,000 of engineering-plus-impact cost each is \$36,000 spent *not solving the problem*.

They were Level 0 specifically because they had no instrumentation that would have shown the connection pool saturating. Once they added a single gauge for `db_connections_in_use` against `db_connections_max`, the next "incident" took four minutes to diagnose: the pool was at 100% and a slow query was holding connections. That four-minute diagnosis is the entire value of Level 1. The fix came later; the *visibility* came first, and visibility is the rung. You cannot diagnose what you cannot see, and you cannot fix what you cannot diagnose. The order is forced.

## 3. Level 1 — Aware: you can see, but you can't promise

Level 1 is where most teams that *think* they're doing SRE actually live. They have Prometheus and Grafana. They have dashboards — sometimes dozens of them. They have alerts that fire and get acknowledged. They might even have a wiki full of runbooks. And yet something is missing, and the missing thing is the difference between *watching* a system and *governing* it.

### The symptoms you'd observe

At Level 1 you have basic monitoring and dashboards that someone built and the team mostly trusts. Alerts exist and get acknowledged — the pager goes off, someone clicks "ack," and the incident gets handled, more or less. There may be a wiki of tribal runbooks, which is a real step up: knowledge is starting to leave people's heads and enter documents, even if those documents are stale half the time. But here's the tell that you're *still* at Level 1: **there are no SLOs.** You measure latency and errors, but you've never decided what "good enough" means. So every alert is a judgment call. Is 2% errors a page-worthy incident or a Tuesday? Nobody knows, because there's no agreed threshold, so the answer depends on who's on call and how they're feeling. Toil — the manual, repetitive operational work — is real and heavy, but *unmeasured*, so nobody can argue for the automation budget to reduce it.

### What hurts

The pain at Level 1 is subtle and corrosive: you have data but no *decisions*. Dashboards tell you what's happening but not what to *do* about it. Alerts fire but you can't prioritize them, because without an SLO there's no objective notion of "how bad." This is where alert fatigue sets in hard — you're past the Level 0 problem of being blind, and into the Level 1 problem of *drowning in signal*. Engineers start ignoring alerts because most of them don't matter, and then they ignore the one that does. The deeper pain is that you still can't make a credible reliability promise. Product asks "can we commit to 99.9% in this contract?" and you genuinely don't know, because you've never measured availability against a target.

### The one practice that moves you up

The unlocking practice for Level 1 → Level 2 is the keystone of the entire discipline: **define an SLO on a user-facing SLI, and start tracking it.** An SLI (Service Level Indicator) is a measured signal of user experience — typically a *ratio* of good events to total events. An SLO (Service Level Objective) is your target for that ratio over a window. Picking even one good SLO transforms everything above it: alerts become "are we burning the budget?" instead of "did a number cross a line someone guessed at," and you finally have the inputs to compute an error budget.

The discipline of [choosing an SLI that reflects user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) deserves its own treatment, but the mechanical version is short. You already have the metrics from Level 1. Now you write a *recording rule* that turns them into an SLI, and you declare a target:

```yaml
groups:
  - name: slo_recording
    rules:
      # SLI: the "good fraction" of requests over a rolling 30-day window.
      - record: job:availability_sli:ratio_rate30d
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[30d]))
          /
          sum(rate(http_requests_total[30d]))
      # Latency SLI: fraction of requests served under 300ms.
      - record: job:latency_sli:ratio_rate30d
        expr: |
          sum(rate(http_request_duration_seconds_bucket{le="0.3"}[30d]))
          /
          sum(rate(http_request_duration_seconds_count[30d]))
```

With the SLI recorded, the SLO is just a number you commit to — say 99.9% availability and 99% of requests under 300ms — and the error budget falls out as arithmetic. This is the moment "is it reliable enough?" stops being an argument and becomes a calculation. That shift, from opinion to arithmetic, is the substance of clearing the rung. The full discipline of [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) — picking targets that match what users can actually perceive, not aspirational nines — is what separates a real Level 2 from a cargo-culted one.

There's a cultural cost to this rung that's worth naming, because it's where the leap to "managed" actually happens and it's not purely technical. Defining an SLO forces a conversation nobody at Level 1 has had: *how reliable does this service actually need to be?* That conversation has to include product and the business, not just engineering, because the SLO is a promise with a cost and someone other than the on-call has to own it. The first time a team writes down "99.9% and no more" and means it, they've made a *decision* about acceptable failure — which is precisely the thing Level 0 and Level 1 teams avoid by pretending every failure is unacceptable (and then triaging by panic). The maturity leap from Aware to Managed is, at bottom, the leap from "all failures are emergencies" to "we have decided how much failure is acceptable and we budget for it." That decision is uncomfortable, which is exactly why so many teams stall at Level 1 with great dashboards and no SLOs — the tooling is easy and the decision is hard.

The proof you've left Level 1: you can now answer "are we within budget this month?" with a number, and your alerts start pointing at user pain instead of at every twitchy resource gauge. We'll see exactly how to rebuild alerting on that foundation in section 11.

## 4. Level 2 — Managed: reliability becomes a number you operate

Level 2 is where reliability stops being vibes and becomes *managed*. You have SLOs. You have error budgets. Your alerts page on user-visible symptoms rather than on every possible cause. Your postmortems are blameless and produce tracked action items. Your on-call is a humane, fair rotation rather than two heroes carrying the world. And you've started *measuring* toil even if you haven't fully reduced it yet. This is a genuinely good place to be, and — fair warning — it's the most dangerous rung for a different reason: it's where teams *declare victory*. More on that anti-pattern in section 10.

### The symptoms you'd observe

At Level 2 the error budget is defined and visible on a dashboard everyone can see. Alerting is **symptom-based**: you page when users are in pain (errors are up, latency is over the SLO threshold, the budget is burning fast) rather than on causes (a single node's CPU, one pod restarting). Incidents follow a recognizable shape — there's an incident commander, a comms lead, a [clear structure](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident) — and afterward there's a [blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) that produces action items somebody owns. On-call is a rotation with a reasonable load and a [humane design](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call): handoffs, sane paging hours, follow-the-sun if you're global. Toil is *named and counted* — you know roughly what fraction of the team's time goes to repetitive ops work, even if it's still high.

### What hurts

The pain at Level 2 is that you have all the right *artifacts* but they don't yet *govern* anything. The error budget exists, but when it's exhausted, nothing stops the team from shipping more risky features — the budget is a thermometer, not a thermostat. Postmortems produce action items, but the action items sit in a backlog and rot because there's no mechanism forcing them to close. Toil is measured at, say, 60% of the team's time, and everyone agrees that's too high, but the automation work keeps getting deprioritized for features. In other words, Level 2 has *visibility and process* but not yet *enforcement and engineering*. You know what's wrong; you haven't built the machinery that fixes it automatically.

### The practices that move you up

Level 2 → Level 3 takes *two* tightly-linked practices, and they reinforce each other. The first is **error-budget governance**: a written, agreed policy that the budget actually *controls* something — most often the release process. The second is **progressive delivery with automated rollback**, which is what makes spending the budget safe enough to keep shipping. Let me take them in turn because they are the heart of what "engineered reliability" means.

Error-budget governance turns the budget from a number you watch into a rule that changes behavior without a meeting. The classic policy: while the budget has room, the team ships features freely; when the budget is exhausted, the team *stops* feature work and spends its time on reliability until the budget recovers. This is the famous trick of the [error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — it aligns dev and ops without anyone having to argue, because the *arithmetic* decides. A written policy looks like this:

```yaml
# error-budget-policy.yaml — an agreement, version-controlled, not a vibe.
service: checkout-api
slo:
  availability: 99.9%        # 43.2 min/month of budget
  window: 30d
policy:
  budget_remaining_gt_50pct:
    action: ship_freely
  budget_remaining_10_to_50pct:
    action: ship_with_review     # senior approval on risky changes
  budget_remaining_lt_10pct:
    action: freeze_features      # reliability work only
    alert: "#eng-leadership"
  budget_exhausted:
    action: hard_freeze
    require: error_budget_postmortem
review_cadence: weekly
```

That policy is only humane if you can *also* ship without spending the whole budget on every deploy. That's what [progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) buys you: instead of a big-bang release, you route a small fraction of traffic to the new version, watch the SLIs, and automatically roll back if they degrade. Here's a canary with an automated analysis gate, in the Argo Rollouts style:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout-api
spec:
  strategy:
    canary:
      steps:
        - setWeight: 5          # 5% of traffic to the new version
        - pause: { duration: 5m }
        - analysis:
            templates:
              - templateName: success-rate-and-latency
        - setWeight: 25
        - pause: { duration: 10m }
        - analysis:
            templates:
              - templateName: success-rate-and-latency
        - setWeight: 50
        - pause: { duration: 10m }
        - setWeight: 100
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate-and-latency
spec:
  metrics:
    - name: success-rate
      interval: 1m
      failureLimit: 2           # roll back after 2 bad readings
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{job="checkout-api-canary",status!~"5.."}[2m]))
            /
            sum(rate(http_requests_total{job="checkout-api-canary"}[2m]))
      successCondition: result[0] >= 0.999
```

When those two practices are in place, a bad deploy spends a sliver of the budget on 5% of traffic for five minutes and then rolls itself back — not the entire budget on 100% of traffic until a human notices. That's the engineering that earns Level 3.

#### Worked example: a deploy that would have blown the budget

Suppose your SLO is 99.9% over 30 days — a 43.2-minute monthly budget. A bad release ships a bug that errors on 8% of requests. At your traffic of, say, 2,000 requests per second, that's 160 failing requests per second. In a full-traffic, big-bang deploy that takes 20 minutes to notice and roll back, you've served `160 × 60 × 20 = 192,000` failed requests. Against a monthly total of roughly `2000 × 60 × 60 × 24 × 30 ≈ 5.2 billion` requests, that single bad deploy burns `192,000 / 5,200,000,000 ≈ 0.0037%` — which against a 0.1% budget is about 3.7% of your *entire month's* budget in 20 minutes. Do that five times in a month and you're frozen.

Now run the same bug through the canary. It hits 5% of traffic for 5 minutes before the analysis gate trips: `160 × 0.05 × 60 × 5 = 2,400` failed requests, which is `2,400 / 5,200,000,000 ≈ 0.000046%` — about 0.05% of the monthly budget. Same bug, **80× less budget burned**, because progressive delivery shrinks the blast radius in both traffic *and* time. That ratio is the entire argument for the practice, and it's why Level 3 teams ship *more* often, not less, while spending *less* budget.

## 5. Level 3 — Engineered: the budget runs the release train

Level 3 is where reliability stops being something a team *does* and becomes something the *system* enforces. The error budget governs releases automatically. Observability is rich enough that you can ask questions you didn't anticipate when you built the instrumentation — you can slice an incident by customer, region, and version on the fly. Toil is below 50% of the team's time and *falling*, because automation work is funded by the error-budget policy. Progressive delivery with automated rollback is the default deploy path. The resilience patterns — [circuit breakers, bulkheads, and load shedding](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding), [timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) — are in place and have been *tested under fault injection*. And capacity is planned, not discovered during an outage.

### The symptoms you'd observe

You're at Level 3 when the budget actually *stops* things — a release genuinely doesn't go out when the budget is exhausted, and nobody overrides it casually. Your observability is high-cardinality and exploratory: when an incident happens, you don't just see "errors are up," you can pivot to "errors are up *for enterprise customers in us-east-1 on version 4.2*" without having pre-built that dashboard. Toil is tracked as a metric and trending down. Deploys are boring — they're canaries that mostly succeed and occasionally roll themselves back without paging anyone. You run [chaos experiments](/blog/software-development/site-reliability-engineering/chaos-engineering-breaking-on-purpose) and game days, and they mostly confirm the system behaves as designed, with occasional valuable surprises. Capacity is forecasted from growth trends, with headroom for the next spike.

### What hurts

The pain at Level 3 is real but it's a *good* pain — the pain of diminishing returns and of organizational, not technical, limits. The cost of the next nine is now visibly high: going from 99.9% to 99.99% (from 43.2 minutes to 4.32 minutes of monthly budget) might require multi-region active-active, which is genuinely expensive, and you have to ask whether users can even tell the difference. Learning is still mostly *per-incident*: each postmortem is good, but you're not yet systematically mining patterns *across* incidents to find the recurring contributing factors. And the resilience engineering is sophisticated enough that it has its own failure modes — a [self-healing system](/blog/software-development/site-reliability-engineering/self-healing-systems-and-their-traps) can mask a degradation until it's catastrophic, and an automated rollback can fight a human operator if you're not careful.

### The practice that moves you up

The unlocking practice for Level 3 → Level 4 is the subtlest one in the model, and it's why most organizations never reach Level 4: **learn from incidents at the level of trends, not individual events.** A Level 3 team writes a good postmortem for each incident. A Level 4 team treats the *corpus* of postmortems as a dataset, mines it for recurring contributing factors, and feeds those patterns directly into the engineering roadmap. This is [learning from incidents at scale](/blog/software-development/site-reliability-engineering/learning-from-incidents-at-scale), and it requires that you've been writing structured, blameless postmortems long enough to *have* a corpus worth mining — which is precisely why you can't skip to it.

The mechanical foundation is tagging every postmortem with structured, queryable metadata, so that "we keep getting bitten by config pushes" stops being a hallway anecdote and becomes a chart. A minimal incident record:

```json
{
  "incident_id": "INC-2026-0418",
  "severity": "Sev2",
  "detected_by": "burn-rate-alert",
  "time_to_detect_min": 4,
  "time_to_mitigate_min": 22,
  "service": "checkout-api",
  "contributing_factors": ["config-push", "missing-canary", "stale-runbook"],
  "trigger": "deploy",
  "action_items": [
    {"id": "AI-881", "owner": "team-checkout", "status": "open",
     "title": "Gate config pushes through the canary pipeline"}
  ],
  "budget_burned_pct": 12.5
}
```

Once a few dozen of those exist, the learning becomes a query, not a memory:

```python
from collections import Counter
import json, glob

factors = Counter()
for path in glob.glob("postmortems/*.json"):
    with open(path) as f:
        inc = json.load(f)
    factors.update(inc["contributing_factors"])

# The top recurring factor is your highest-leverage roadmap item.
for factor, count in factors.most_common(5):
    print(f"{factor:24s} appeared in {count} incidents")
```

When `config-push` shows up in eleven of your last forty incidents, you don't write an eleventh postmortem action item — you fund an engineering initiative to make config pushes safe, and you've just let your incident history *plan your roadmap*. That feedback loop, from the corpus of learning back into prioritized engineering, is the engine of Level 4.

## 6. Level 4 — Optimizing: reliability as a product with a budget

Level 4 is rare, and most teams don't need it — which I'll say more forcefully in section 12. At Level 4, reliability is data-driven and continuously improving. Learning from incidents at scale feeds the roadmap as a first-class input. Self-healing automation handles the well-understood failure modes *where it's safe to do so* (and only there). And — this is the cultural marker — the organization treats reliability as a *product feature* with its own budget, roadmap, and stakeholders, not as a cost center or a tax.

### The symptoms you'd observe

A Level 4 organization has reliability metrics that trend over quarters and *inform planning*: the SLO targets themselves get revisited based on what users actually value and what the business can sustain. Incident patterns are mined and the top recurring factors become roadmap initiatives, with measurable before-and-after results. Self-healing is present but *bounded* — the system auto-remediates the failure modes it understands well, and explicitly escalates to humans for anything novel. Capacity, cost, and reliability are optimized together: the team can articulate "this nine costs us \$X/month and saves users Y minutes" and make a *deliberate* trade-off rather than reflexively chasing more nines. And reliability work has executive sponsorship and a line in the budget, because it's understood to drive retention and revenue.

### What hurts

The pain at Level 4 is almost entirely about *judgment under diminishing returns and complexity*. The systems are sophisticated enough that the failure modes are emergent — the self-healing interacts with the autoscaler interacts with the circuit breaker in ways no single person fully models. The biggest risk is over-automation: auto-remediating something you don't fully understand, which can turn a small problem into a large one by hiding it or by the remediation itself misfiring. And there's an organizational pain — keeping reliability funded when things are *going well* is harder than funding it during a crisis, because the absence of outages looks like the work isn't needed. Level 4 teams have to actively defend the budget that keeps them at Level 4.

### Staying at Level 4

There is no "Level 5" in this model, and that's deliberate. Level 4 is not a finish line; it's a *steady state* you maintain by continuing to measure, learn, and deliberately trade off. The unlocking "practice" for staying here is simply *keep running the loop, and keep the budget funded*. The danger is complacency — a Level 4 team that stops investing slides back down the ladder surprisingly fast, because reliability is a property you *maintain*, not a milestone you *reach*.

The clearest sign of a real Level 4 organization, versus one that merely owns the tooling, is how it treats its *SLOs* over time. At lower rungs the SLO is fixed once and defended forever. At Level 4 the SLO target itself is a variable the org tunes deliberately: if users genuinely can't tell 99.95% from 99.99%, a Level 4 team will *lower* the target to reclaim the error budget and spend it on velocity — the opposite of the reflexive nine-chasing that traps less mature teams. The willingness to spend budget *on purpose*, to ship faster when the reliability is already better than users perceive, is the mark of an org that treats reliability as a tunable product dimension rather than a moral imperative. Conversely, when an SLO is consistently *missed*, a Level 4 team asks whether the target was right before it asks the team to work harder — because an SLO nobody can hit is either a too-aggressive target or a real engineering gap, and the data tells you which.

The second marker is *bounded* self-healing. A mature org can articulate, in writing, exactly which failure modes the automation is allowed to remediate without a human and which it must escalate. The boundary is drawn by understanding: auto-restart a pod that OOMs in a known, benign way — yes; auto-scale-out on a traffic spike that matches a learned pattern — yes; auto-remediate a novel error signature the system has never seen — no, escalate to a human, because remediating what you don't understand is how a small incident becomes a large, *hidden* one. The discipline isn't "automate everything"; it's "automate the understood, escalate the novel, and never let the automation outrun your observability." That bound is the difference between self-healing that buys reliability and self-healing that buys a new, worse class of outage.

## 7. Maturity is per-dimension, not a single number

Here is the single most useful correction this post makes to the way people usually think about maturity: **your team does not have "a level." It has a level per dimension, and they're almost always uneven.** You can be Level 3 on deployment (you've got canaries and automated rollback because the platform team handed you a paved road) and Level 0 on learning (your postmortems are blameful, or you don't write them, or the action items never close). A single overall number would hide exactly the thing you most need to see.

![A graph showing four reliability dimensions at different maturity levels feeding into an overall level that equals the floor, with the lowest dimension determining the next move](/imgs/blogs/the-reliability-maturity-model-6.png)

Why does this happen? Because the dimensions are unlocked by *different practices owned by different people*. Deployment maturity often comes "for free" if your company has a strong platform team — you inherit a canary pipeline without ever having done the cultural work of blameless postmortems. Monitoring maturity comes from tooling investment. But postmortem maturity is *cultural* — it requires leadership to genuinely not punish people for honest mistakes — and culture doesn't come bundled with a Helm chart. So a team can buy its way to Level 3 deploys while its learning culture sits at Level 0, and the mismatch is invisible until an incident exposes it.

There's an important consequence for how you read your "overall" level. **Your effective overall maturity is closer to the *floor* of your dimensions than the average.** A team that's L3 on five dimensions and L0 on learning is *not* "averaging to L2.4." It's a team whose recurring incidents never get fixed, because the L0 learning dimension means root causes never become engineering work — so all the L3 deployment sophistication just lets them ship the same bug faster. The lowest dimension is the bottleneck, and bottlenecks set the throughput of the whole system. This is the same logic as the [theory of constraints](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team) applied to reliability: improving a non-bottleneck dimension is wasted effort.

That's also why the self-assessment in the next section scores *seven dimensions separately*. The number you care about isn't the average; it's the *minimum* and the *spread*. The minimum tells you your next move. The spread tells you whether you have a balance problem — usually too much tooling investment relative to cultural investment, because tooling is easier to buy than culture is to build.

## 8. The self-assessment: find your real level in twenty minutes

Now the practical artifact. Below is a seven-dimension rubric. For each dimension, read the four descriptions (L0 through L3 — I deliberately stop at L3, because L4 is a refinement of L3 that most teams don't need to score) and pick the one that *honestly* describes your team's *typical* behavior, not its best day. Be ruthless. The most common mistake is scoring your aspiration instead of your reality; if you "have runbooks" but they're stale and nobody trusts them at 3am, that's not the runbook rung.

![A seven-row by four-column rubric mapping each reliability dimension across maturity levels zero through three with a short behavioral description in every cell](/imgs/blogs/the-reliability-maturity-model-4.png)

### The rubric

| Dimension | L0 Reactive | L1 Aware | L2 Managed | L3 Engineered |
|---|---|---|---|---|
| **Measurement** | "Is it up?" answered by a human loading the page | Dashboards and metrics exist; no targets | SLIs and SLOs defined on user-facing signals | Error budget computed and tracked; SLOs revisited |
| **Alerting** | Alert on everything or nothing; channel is noise | Some alerts fire and get acknowledged | Symptom-based, paging on user pain | Multi-window burn-rate alerts; tuned false-positive rate |
| **Incident response** | Ad hoc heroics; depends who's on call | Incidents acknowledged; informal handling | Defined roles, rotation, incident commander | Practiced IC, drills, two-incident playbook |
| **Learning** | Blame, or no postmortem at all | Notes filed somewhere | Blameless postmortems with owned action items | Action items close; trends mined across incidents |
| **Toil** | Unmeasured; "we're just busy" | Felt but not counted | Measured as a percentage of team time | Under 50% and falling; automation funded |
| **Deployment** | Scary manual deploys, no rollback plan | Scripted deploys, manual rollback | CI/CD with gates and a tested rollback | Canary + automated rollback by default |
| **Resilience** | Hope it holds; no timeouts | Some timeouts and retries, untested | Circuit breakers and bulkheads in place | Chaos-tested under real fault injection |

### How to score it honestly

Score each dimension, then write down two numbers: your **minimum** (the lowest dimension) and your **spread** (highest minus lowest). The minimum is your effective overall level and points at your next move. The spread tells you whether your problem is "we're uniformly early" (small spread, low floor — invest broadly in the foundation) or "we're lopsided" (large spread — invest in the laggard, usually a cultural dimension like Learning, because the tooling dimensions raced ahead).

A few honesty checks, because self-assessment is where teams fool themselves. For **Learning**, the test is not "do we write postmortems" but "did last quarter's action items actually *close*?" If they're still open, you're L1 at best regardless of how good the documents are. For **Toil**, the test is a number: if you can't say "roughly X% of our week is toil," you're below L2 by definition, because L2 *is* having the measurement. For **Alerting**, the brutal test is your false-positive rate: count last week's pages and ask how many represented real user pain. If it's under half, you're noisy, and noisy alerting is an L0/L1 symptom no matter how sophisticated the rules look.

#### Worked example: scoring a realistic team

Let me score a real-shaped team — call it the Checkout team. Walking the rubric:

- **Measurement: L2.** They have dashboards and they defined an availability SLO of 99.9% six months ago. They check it. Good.
- **Alerting: L1.** They have alerts, but a lot of them are cause-based (CPU, memory, pod restarts) and the on-call complains about noise. Symptom-based alerting isn't really in. *L1.*
- **Incident response: L2.** They have a rotation and a loose incident-commander convention. Decent.
- **Learning: L0.** This is the one that matters. They "do postmortems," but the meetings tend to find someone to blame, and the action items go into a backlog that never gets prioritized. Of last quarter's 14 action items, 2 closed. *That's L0, not L1* — filing blameful notes that nobody actions is worse than no postmortem because it teaches people to hide.
- **Toil: L1.** Everyone agrees toil is heavy. Nobody has measured it. *L1.*
- **Deployment: L2.** They inherited a CI/CD pipeline with a manual rollback button that works. No canary yet.
- **Resilience: L1.** Some timeouts and retries exist, configured years ago, never tested. No circuit breakers.

What's the team's level? The *average* is about 1.3 — which would tempt them to say "we're solidly L1, let's invest in SLO tooling." But the **minimum is L0 on Learning**, and that's the floor that's actually killing them. Their recurring incidents never get fixed because the action items never close, so all their L2 measurement just documents the same fire repeatedly. The honest read is: **L1 overall, L0 on Learning, and Learning is the bottleneck.**

The targeted next move is therefore *not* "define more SLOs" or "buy a canary tool" — those would improve dimensions that aren't the constraint. The move is to fix Learning: introduce genuinely [blameless postmortems](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) and, critically, a mechanism that forces action items to close — a weekly review where open items are tracked and a policy that a service can't ship features while it has an overdue Sev1 action item. That single cultural change unblocks everything, because it's the thing that converts their good measurement into actual fixes. No big-bang transformation; one targeted move at the constraint.

![A graph contrasting a team that bought a chaos tool at Level 0 with no baseline against a team that earned a Level 2 SLO baseline first, showing only the second produces a measurable signal](/imgs/blogs/the-reliability-maturity-model-3.png)

## 9. The skipped-rung failure: a chaos tool at Level 0

Let me tell the cautionary tale promised in the intro, because it's the most common expensive mistake and it perfectly illustrates *why* the order matters. A team — frustrated by recurring outages, and having read that the most mature companies do chaos engineering — bought a chaos-engineering platform. The thinking was reasonable on its face: "the best teams break things on purpose to find weaknesses, so let's break things on purpose." They scheduled a game day, killed a database replica, and watched.

And they learned *nothing*. Here's why, and it's the whole lesson. [Chaos engineering](/blog/software-development/site-reliability-engineering/chaos-engineering-breaking-on-purpose) is the practice of forming a *hypothesis about steady-state behavior*, injecting a fault, and checking whether the system stayed within its steady state. The entire method depends on having a defined, measured steady state to compare against. This team was at Level 0: no SLO, so no definition of "healthy"; thin monitoring, so no measurement of whether the experiment degraded user experience. When they killed the replica, the system... did something. Latency maybe went up. Errors maybe happened. But they had no SLI to measure "maybe," no baseline to compare it to, and no error budget to quantify the impact. The experiment produced *noise*, not signal, because there was no steady-state ruler to measure the deflection against.

The principle here is exact: **chaos engineering is a Level 3 practice that measures a deviation from steady state, and steady state is a Level 2 artifact (the SLO/SLI).** Running it at Level 0 is like trying to measure how much a bridge sways in the wind without first measuring where the bridge sits when it's calm. You see it move; you can't say whether that's normal or a crisis. The same money and time, spent on defining one SLI and one SLO (the Level 1→2 move), would have given them the ruler — and *then* the chaos experiment would have produced a number: "killing the replica burned 4% of our monthly budget in eight minutes, here's the fallback that didn't trip." That's a finding. Noise is not.

This is the general shape of every skipped-rung failure, and it's worth internalizing as a pattern:

| Skipped-rung attempt | What's missing below it | What you actually get |
|---|---|---|
| Chaos engineering with no SLO | A measured steady state | Noise; "something happened" |
| Error-budget policy with no SLI | The budget is uncomputable | A policy that can't fire |
| Burn-rate alerts with no SLO | A target rate to burn against | An alert with no threshold |
| Trend-learning with no postmortems | A corpus to mine | An empty dataset |
| Self-healing with no observability | Knowledge of *what* to heal | Automation masking real failures |

In every row, the upper practice *consumes* the lower rung's output. Without the input, the practice doesn't fail loudly — it runs, produces something that looks like a result, and quietly carries no information. That quiet failure is what makes skipped rungs so seductive and so wasteful: you get the *appearance* of maturity without the substance. Earn the foundation first.

## 10. The three anti-patterns

There are three ways teams reliably waste their reliability investment. They're worth naming explicitly so you can catch yourself.

### Anti-pattern 1: cargo-culting a high-level practice

This is the skipped-rung failure generalized. You see what mature organizations *do* — chaos engineering, error budgets, self-healing — and you adopt the visible practice without the invisible foundation it rests on. Cargo-culting is seductive because the high-level practices are the ones that get conference talks and blog posts, while the foundational work (instrument the golden signals, write a real SLO) is unglamorous. The fix is the maturity model itself: before adopting a practice, ask "what does this practice *consume*, and do I have it?" If chaos engineering consumes a steady-state SLO and you don't have one, you're cargo-culting.

### Anti-pattern 2: declaring victory at Level 2

This one is subtler and catches good teams. You did the hard work to reach Level 2 — SLOs, blameless postmortems, a humane rotation — and it feels like a destination. It's not. The trap is that Level 2 has all the *artifacts* of maturity without the *enforcement*. The error budget exists but doesn't govern; the postmortems happen but the action items rot; the toil is measured but never reduced. A team that declares victory at L2 slowly decays, because nothing is pushing the loop forward. The error budget becomes a thermometer nobody acts on. The tell is when your "Action Items" backlog has items older than two quarters and your error budget has been negative for a month with no change in behavior. The fix is the Level 2→3 move: make the budget *govern* releases, so the system enforces what the process merely documents.

### Anti-pattern 3: reliability theater

Theater is when you have the *appearance* of reliability practice with none of the function. Dashboards that nobody looks at — built once for a review, never opened since. Postmortems written and filed and never read or actioned. SLOs defined in a doc but not actually wired to any alert or any decision. An on-call rotation that exists on paper while the same two people quietly handle everything. Theater is worse than doing nothing, because it consumes real effort and produces *false confidence* — leadership believes the system is being managed because the artifacts exist. The fix is to ruthlessly test every artifact against "does a decision depend on this?" A dashboard nobody uses to make a decision should be deleted. A postmortem nobody actions should not be written until you fix the action-item process. An SLO not wired to an alert or a release gate is a wish, not an objective.

![A before-and-after contrast showing a team that declared victory at Level 2 with unread dashboards and unactioned postmortems versus a team that climbed to Level 3 where the budget gates releases and trends feed the roadmap](/imgs/blogs/the-reliability-maturity-model-8.png)

The common thread across all three anti-patterns: **the artifact is not the practice.** A dashboard is not observability; observability is *asking and answering questions*. A postmortem document is not learning; learning is *the action item that closes and the recurrence that stops*. An SLO in a doc is not an objective; an objective is *a target that changes a decision*. The maturity model keeps you honest by always asking what *function* a rung serves, not what artifacts it produces.

## 11. Rebuilding alerting as you climb — a concrete trace

To make the climb tangible, let me trace one dimension — alerting — across the rungs, because it's the dimension where the difference between levels is most visible in day-to-day pain, and where the principle (symptom over cause, page on user impact) earns its keep.

At **Level 0**, alerting is `alert if CPU > 80%` and forty other cause-based rules. The on-call gets paged when a single node's CPU spikes — which happens constantly and usually means nothing, because the load balancer just routes around it. The pager fires 200 times a week, the on-call ignores most of it, and the one real alert gets lost in the flood.

At **Level 1**, you've got dashboards and you've pruned the worst noise, but the alerts are still cause-based. You page on "error rate > 5%" with a threshold someone picked by feel. It's better — maybe 60 pages a week — but it still fires on transient blips and still doesn't reflect whether *users* are actually hurting over a meaningful window.

At **Level 2**, you alert on the *symptom* (the SLI) over a window, not the cause. This already cuts noise dramatically because transient spikes don't burn enough budget to matter:

```yaml
groups:
  - name: slo_symptom_alerts
    rules:
      - alert: CheckoutErrorBudgetBurning
        expr: |
          (1 - job:availability_sli:ratio_rate1h) > 0.01
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "Checkout error rate over 1% for 5m — burning budget fast"
```

At **Level 3**, you graduate to **multi-window, multi-burn-rate** alerting — the technique from the Google SRE Workbook that catches both fast burns (a sudden outage) and slow burns (a steady low-grade error rate that quietly drains the budget over days), while keeping the false-positive rate low. The *burn rate* is how fast you're consuming the budget relative to "even" consumption:

$$\text{burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}$$

A burn rate of 1 means you'll exactly exhaust the 30-day budget in 30 days. A burn rate of 14.4 exhausts it in roughly two days. You page urgently on a *fast* burn confirmed over a short window, and page less urgently on a *slow* burn confirmed over a long window:

```yaml
groups:
  - name: multiwindow_burnrate
    rules:
      # Fast burn: 14.4x over 1h AND 5m (catches acute outages, low FP).
      - alert: ErrorBudgetFastBurn
        expr: |
          (
            job:availability_sli:error_ratio_rate1h > (14.4 * 0.001)
            and
            job:availability_sli:error_ratio_rate5m > (14.4 * 0.001)
          )
        labels: { severity: page }
        annotations:
          summary: "Fast burn: ~2 days to exhaust the 30-day budget"
      # Slow burn: 3x over 6h AND 30m (catches the quiet drain).
      - alert: ErrorBudgetSlowBurn
        expr: |
          (
            job:availability_sli:error_ratio_rate6h > (3 * 0.001)
            and
            job:availability_sli:error_ratio_rate30m > (3 * 0.001)
          )
        labels: { severity: ticket }
        annotations:
          summary: "Slow burn: budget draining over days"
```

#### Worked example: the alert-volume before and after

Here's the measured payoff of climbing this dimension, from a real-shaped migration. Before (Level 0/1, cause-based): **roughly 200 pages per week**, of which by the on-call's own count maybe 8 represented real user pain — a 96% false-positive rate. The on-call was exhausted and had started reflexively acknowledging without looking. After moving to symptom-based then multi-window burn-rate alerting (Level 2→3): **about 12 pages per week**, of which around 10 represented real user pain — a roughly 17% false-positive rate. Alert volume fell **about 94%**, and — the number that matters most — the *real* alerts stopped getting lost. MTTR on genuine incidents dropped from 90 minutes to 22 minutes, largely because the on-call now *trusts* the pager and responds immediately instead of assuming it's noise. The full discipline lives in [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf); the point here is that this 94% reduction was only *possible* because the SLO existed. You cannot build symptom-based or burn-rate alerts without the SLI/SLO foundation. The rung order is forced, again.

![A timeline of one team climbing from Level 0 to Level 3 over four quarters, adding one practice each quarter while pages per week and MTTR fall steadily](/imgs/blogs/the-reliability-maturity-model-7.png)

## 12. War stories and how the model reads them

The maturity model is most convincing when you use it to read real outages, because the famous failures are almost always *rung mismatches* — a practice operating without its foundation.

### The Google error-budget model

The origin story of error budgets, documented in the Google SRE Book, is itself a maturity-model insight. Google's SRE teams found that the perpetual dev-versus-ops conflict — developers want to ship, operators want stability — dissolves once you have a *measured* error budget. The budget gives both sides a shared, objective currency: while there's budget, ship; when it's exhausted, stabilize. But notice the precondition. This only works at Level 2+, because the budget has to be *measured* (you need the SLI/SLO) and at Level 3 it has to *govern* (the policy has to actually freeze releases). A Level 0 team that tries to "adopt the Google model" by writing an error-budget policy gets nothing, because they have no measured budget for the policy to act on. The model is real; it requires the foundation.

### Cascading failures and the retry storm

A recurring class of major outages is the *retry storm* / thundering herd: a dependency gets briefly slow, every client retries aggressively without backoff, the retries multiply the load, and the brief slowness becomes a full cascading collapse. The retry-amplification factor is brutal — if every client retries `r` times on failure, a backend already at the edge sees up to `(1 + r)×` the load exactly when it can least handle it; with `r = 3` that's a 4× surge precisely during a wobble. The fix — [retries with exponential backoff and jitter](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right), plus [circuit breakers](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding) — is a Level 3 resilience practice. But here's the maturity reading: a team often *has* the breakers configured (they read the same blog posts) but has *never tested them* under fault injection, so they discover during the real outage that the breaker's threshold was wrong or it was never wired in. Configured-but-untested resilience is an L2 symptom masquerading as L3. The chaos-testing (the actual fault injection) is the thing that promotes it, which is why chaos engineering sits at Level 3 — it *validates* the resilience the team thinks it has. For the architecture-time view of these patterns, the system-design treatment of [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) is the companion read; this series covers *operating and validating* them in production.

### The untested backup

The most painful war story is also the simplest: a company has backups, religiously, for years. Disaster strikes, they go to restore — and the backups are corrupt, or incomplete, or the restore procedure has bit-rotted and nobody has ever run it end to end. The maturity reading is exact: *having backups is L1; having backups you've actually restored from in a drill is L3.* The [backups that actually restore](/blog/software-development/site-reliability-engineering/backups-that-actually-restore) discipline is precisely the difference between the artifact (a backup file exists) and the function (you can recover within your RTO). Untested backups are the canonical reliability theater — an artifact that produces false confidence and fails exactly when called upon. RPO (how much data you can afford to lose) and RTO (how long recovery can take) are meaningless numbers until a *drill* proves you can hit them.

### The config-push outage

Many of the largest internet outages of the last decade trace to a single bad configuration push that propagated everywhere at once. The maturity reading is the [progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) gap: the team treated config as somehow different from code — code went through a canary, but config pushed globally and instantly. A bad config then had the maximum possible blast radius. The Level 3 fix is to route *config changes through the same progressive-delivery and automated-rollback path as code*, so a bad config hits 5% and rolls back. The trend-learning (Level 4) fix is the one from section 5: when `config-push` shows up in eleven postmortems, you fund the initiative to make config a first-class, gated deploy artifact.

## 13. The economics of each rung, and the cost of the next nine

The maturity model isn't just a sequence of practices; it's a sequence of *trades*, and being explicit about what each rung costs and returns is what keeps a reliability program funded and honest. Let me put numbers on it, because "invest in reliability" loses every budget fight against a feature with a revenue projection, while "this rung costs roughly N engineer-weeks and removes M hours of toil per week" wins.

The classic anchor is the **nines-to-downtime** table, because it makes the *target* concrete and immediately exposes the diminishing returns that govern the upper rungs:

| SLO | Allowed downtime / month | Allowed downtime / year | Roughly what it demands |
|---|---|---|---|
| 99% (two nines) | 7.3 hours | 3.65 days | Basic monitoring; manual recovery is fine |
| 99.9% (three nines) | 43.2 minutes | 8.77 hours | SLOs, symptom alerts, fast human response |
| 99.95% | 21.6 minutes | 4.38 hours | Automated rollback, tested failover |
| 99.99% (four nines) | 4.32 minutes | 52.6 minutes | Multi-region, no-human-in-the-loop recovery |
| 99.999% (five nines) | 25.9 seconds | 5.26 minutes | Active-active everywhere; recovery faster than a human can react |

Read that table as a *cost curve*. Each additional nine cuts the allowed downtime by 10×, and the practices required to hit it roughly multiply in cost. Going from two nines to three is mostly *process* — define SLOs, fix alerting, respond fast — and it's the cheapest, highest-leverage rung you'll ever climb. Going from four nines to five often means the recovery has to happen *faster than a human can wake up and read a page*, which forces fully automated, no-human-in-the-loop failover across regions — a different and far more expensive class of engineering. The crucial business insight: the user-perceived benefit of each nine *shrinks* while the cost *grows*, so there's a service-specific point where the next nine is simply not worth buying. The maturity model's job is to make sure you climb the cheap, high-leverage rungs first and stop deliberately at the rung the stakes justify, rather than reflexively chasing nines no user will ever notice.

#### Worked example: pricing a rung against the toil it removes

Say your team spends 60% of its time on toil — roughly three of five engineers' time, on a five-person team, going to manual deploys, manual incident recovery, and answering "is it healthy?" by hand. That's expensive, and worse, it's the work that *doesn't compound*: toil scales linearly with the service while engineering scales the service sublinearly. Suppose the Level 2→3 move (progressive delivery with automated rollback, plus measuring and attacking the top toil source) costs you, conservatively, eight engineer-weeks to build and roll out. If it takes toil from 60% to 35% — a 25-percentage-point reduction on a five-person team — you've recovered roughly **1.25 engineers' worth of capacity, every week, forever**. Eight engineer-weeks of investment that returns more than one engineer per week pays back in under two months and compounds after that. *That* is the argument that wins the budget fight, and it's the argument the maturity model hands you: each rung has a measurable toil-and-incident return, not a vague "more reliable." The full method for measuring and attacking toil this way is in [toil: the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team) and [automating yourself out of the pager](/blog/software-development/site-reliability-engineering/automating-yourself-out-of-the-pager).

### Stress-testing the climb

A good operating model survives the awkward questions, so let me pose the ones a skeptical staff engineer will raise. *What if two dimensions are tied for the floor?* Pick the one whose practice is the cheapest unlock — usually a process or cultural change beats a tooling project for speed-to-value, and you re-score next quarter anyway. *What if the floor is Learning (culture) and leadership won't make postmortems blameless?* Then your reliability program is blocked at an organizational level, and the honest move is to escalate *that* — the model is now a tool for showing leadership exactly which rung their own behavior is capping. *What if a reorg drops you a level?* That's normal — maturity walks out with the people who carried it, which is the entire argument for encoding practices in *systems* (gated pipelines, written budget policy, structured postmortems) rather than in heroes' heads, so the rung survives turnover. *What if the budget is already exhausted when you're trying to climb?* Then you're in the L2→L3 enforcement scenario by force: the budget being negative *is* the signal to freeze features and spend on reliability, which is exactly the engineering work that climbs the rung. The exhausted budget isn't a reason you can't climb; it's the mechanism that funds the climb.

## 14. How to reach for this — and when not to

A maturity model is a tool, and like every tool it has a wrong way to use it. Let me be decisive about both.

**Use the model to find your *next* move, not to plan a two-year transformation.** The entire value is in identifying the single highest-leverage practice — the one that unblocks the most. Score the seven dimensions, find the floor, fix the floor. Then re-score in a quarter. A reliability program that tries to jump straight to Level 4 by adopting every practice at once will collapse under its own weight and produce a lot of theater. One rung at a time.

![A decision tree that starts from the seven-dimension score, branches on whether monitoring or SLOs exist, and points to the single next practice to adopt](/imgs/blogs/the-reliability-maturity-model-5.png)

**Don't chase a level your service doesn't need.** This is the most important "when not to." Not every service should be Level 3, let alone Level 4. An internal batch job that runs nightly and has no real-time users does not need a 99.99% SLO, multi-window burn-rate alerting, or chaos engineering. Setting a 99.999% SLO for an internal admin tool is *negative* value — you'll spend enormous effort chasing nines that no user can perceive, while your customer-facing checkout flow sits at Level 1. **Match the rung to the stakes.** The right SLO for a service is the lowest one your users can't tell apart from perfect; everything beyond that is wasted budget. A blog summarized the trade-off well: the cost of each additional nine roughly multiplies, while the user-perceived benefit shrinks — so the fifth nine is almost never worth it outside life-critical or revenue-critical paths.

**Don't auto-remediate what you don't understand.** The Level 4 self-healing temptation is real, but auto-remediation of a poorly-understood failure mode is dangerous — at best it masks a degradation (so it gets worse out of sight), at worst the remediation itself misfires and amplifies the problem. The [self-healing systems and their traps](/blog/software-development/site-reliability-engineering/self-healing-systems-and-their-traps) post details this; the maturity reading is that self-healing is a *late*-Level-3/Level-4 practice precisely because it requires the observability to *know* what you're healing and the discipline to bound it to well-understood cases.

**Don't use the model to shame a team.** The point of an honest, usually-humbling self-assessment is to *direct effort*, not to assign a grade. A team scoring L0 on Learning isn't a bad team — it's a team whose leadership hasn't yet made postmortems safe. The model points at the *system* to change, which is the same blameless principle that makes postmortems work. Use it to advocate for the investment the floor dimension needs, not to rank people.

**When the model genuinely doesn't apply:** very early-stage products with no users and no traffic don't need this. If you're pre-product-market-fit with ten users, defining SLOs is premature optimization — you don't yet know what "good" means to users who don't exist in numbers. Get to the point where an outage actually costs you something, *then* start climbing. The model is for systems whose unreliability has a cost; if yours doesn't yet, building it from first principles is its own kind of theater. The companion to this is the [production-readiness review](/blog/software-development/site-reliability-engineering/production-readiness-reviews) (planned in this series) — the checklist a service should pass *before* it earns a high-stakes rung — and [building an SRE culture and team](/blog/software-development/site-reliability-engineering/building-an-sre-culture-and-team) (also planned), which is how an organization sustains the climb past any single hero.

## 15. Key takeaways

- **Reliability is a ladder, not a switch.** There's a recognizable progression from reactive firefighting (L0) to self-improving optimization (L4), and the rung you're on determines which practice will help.
- **You climb in order because each rung consumes the previous rung's output.** Error-budget governance needs an error budget needs an SLO needs an SLI needs monitoring. Skip a rung and the upper practice fails *quietly* — it produces artifacts with no information.
- **The single next move is to fix your lowest dimension, not to adopt the trendiest practice.** Buying a chaos tool at Level 0 is theater; defining one SLO at Level 1 is leverage.
- **Maturity is per-dimension, and your effective level is the floor, not the average.** You can be L3 on deploys and L0 on postmortems; the L0 dimension is the bottleneck that caps the whole system.
- **Score your team honestly across seven dimensions** — measurement, alerting, incident response, learning, toil, deployment, resilience — using typical behavior, not your best day. You're almost certainly lower and more uneven than you think.
- **The three anti-patterns are cargo-culting, declaring victory at L2, and reliability theater** — and they share one root: mistaking the artifact (a dashboard, a postmortem doc, an SLO in a wiki) for the practice (a decision that depends on it).
- **Match the rung to the stakes.** Not every service needs Level 3; an internal batch job at 99.99% is wasted effort. The right SLO is the lowest one users can't tell apart from perfect.
- **Re-score every quarter and fix one rung at a time.** Reliability is a steady state you maintain, not a milestone you reach — Level 4 teams that stop investing slide back down fast.

## 16. Further reading

- *Site Reliability Engineering* (the Google "SRE Book"), especially the chapters on **Embracing Risk**, **Service Level Objectives**, and **Eliminating Toil** — the canonical source for the error-budget and toil concepts the upper rungs depend on.
- *The Site Reliability Workbook* — the practical companion, with the **Alerting on SLOs** chapter that defines the multi-window, multi-burn-rate alerting used in section 11.
- The **Prometheus and Alertmanager documentation** — recording rules, alerting rules, and routing for the artifacts shown throughout.
- **Brendan Gregg's USE method** and the **RED method** — two complementary lenses for the golden-signals instrumentation that gets you off Level 0.
- This series' [intro: reliability is a feature, the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) and [the reliability stack mental model](/blog/software-development/site-reliability-engineering/the-reliability-stack-a-mental-model) — the why and the master map this ladder organizes.
- The practice posts for each rung: [SLI, SLO, SLA](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter) and [the error budget](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) (L2→L3), [the blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) and [toil](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team) (L2), [progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) and [chaos engineering](/blog/software-development/site-reliability-engineering/chaos-engineering-breaking-on-purpose) (L3), and [learning from incidents at scale](/blog/software-development/site-reliability-engineering/learning-from-incidents-at-scale) (L4).
- The forthcoming [capstone: the SRE playbook](/blog/software-development/site-reliability-engineering/capstone-the-sre-playbook), which assembles the whole climb into one operating manual.

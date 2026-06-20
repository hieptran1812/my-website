---
title: "The Reliability Stack: A Mental Model for the Whole Discipline"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Get the one master map that organizes all of Site Reliability Engineering — the closed loop of define, measure, govern, automate, respond, learn, and engineer — so every later technique has a place to live and reliability becomes a continuous process you run, not a project you finish."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "error-budget",
    "slo",
    "sli",
    "observability",
    "incident-management",
    "reliability",
    "mental-model",
    "resilience",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-reliability-stack-a-mental-model-1.png"
---

Most teams I have walked into do not have a reliability problem. They have a reliability *organization* problem. They have a metrics dashboard nobody reads, an alert channel that pages 40 times a night, a quarterly outage that keeps coming back wearing a slightly different hat, and a postmortem template that gets filled out, filed, and never read again. Every one of those things is a real practice. The trouble is that they are sitting in a pile, not in a loop. Nothing feeds anything else. The dashboard does not inform the alert, the alert does not connect to a budget, the budget does not exist, the postmortem does not drive an engineering fix, and so the same disk fills up at 3am every month forever. The team is busy. They are just busy in a circle that never closes.

This post is the map that closes the circle. It is the hub of this entire forty-post series, and its single job is to give you one mental model — the *reliability loop* — that organizes everything else you will ever read about running systems. Once you can see the loop, every later technique stops being a disconnected trick and becomes a stage with a job: a thing that takes a specific input from the stage before it and hands a specific output to the stage after it. SLIs and SLOs are not "the metrics thing," they are the *define* stage. Observability is not "the Grafana thing," it is the *measure* stage. On-call is not "the suffering thing," it is the *respond* stage. They are gears, and the error budget is the shaft that drives them all.

You can see the whole machine in the figure below. Read it top to bottom, but understand it as a loop: the last stage feeds back into the first. We draw it as a stack because a closed ring is hard to read and easy to mislabel, but the arrow from *Engineer* back up to *Define* is the most important arrow in this series — it is the one that makes reliability a continuous process instead of a one-time sprint.

![A vertical stack of the seven reliability loop stages from define through engineer with the error budget governing change in the middle](/imgs/blogs/the-reliability-stack-a-mental-model-1.png)

By the end of this post you will be able to do three concrete things. First, place any reliability practice you have ever heard of — chaos engineering, burn-rate alerts, runbooks, blue-green deploys, RPO targets, blameless retros — onto exactly one stage of this loop and know what it consumes and what it produces. Second, diagnose *which stage your team is weakest at*, because a team is only as reliable as its leakiest gear, and the fix is almost never "try harder at the stage that is already working." Third, explain to a skeptical director why reliability is a process and not a project, with the arithmetic to back it up. That last one matters more than it sounds: the single most expensive mistake in this discipline is treating reliability as a hardening sprint you can finish, when it is in fact a loop you have to keep turning. We will trace one real improvement all the way around the loop, contrast a team that runs it against a team that does not, and end with a decisive guide to where this whole thing is and is not worth the cost.

This is the conceptual map, so it forward-references the deep dives rather than re-deriving them. When I say "the burn-rate alert catches this," I will point you at the observability and error-budget posts where the PromQL lives. When I say "the breaker absorbs it," I will point you at the resilience-design post where the YAML lives. The map is the thing. Hold the map, and the rest of the series is just walking the territory.

One more framing before we start, because it changes how you should read everything below. The reason a *map* is worth a whole post — the reason I am not just listing forty topics in a table of contents — is that the value of this discipline is not in any single practice but in the *connections between them*, and connections are exactly what a list cannot show. A list of forty SRE topics tells you what exists; the loop tells you what *flows*. And it is the flow, not the inventory, that decides whether your reliability rises or rots. So read the sections that follow not as "here is another practice" but as "here is the next gear, and here is the gear it drives." If you finish this post able to draw the seven arrows from memory and name what travels along each one, you will have gotten more out of it than from any individual deep dive, because you will know where every deep dive *fits*.

## 1. Why you need a model at all: the pile versus the loop

Let me start with the failure mode, because it is so common it is almost invisible. A team adopts reliability practices the way a kitchen accumulates gadgets: one at a time, each because somebody read a blog post or survived an outage, and none of them wired together. They install Prometheus after a blind outage. They write a runbook after a confusing incident. They add PagerDuty after someone missed a page. They run a "reliability sprint" after a bad quarter. Each addition is locally sensible. The sum is a pile of tools that do not talk to each other.

Here is the giveaway that you have a pile and not a loop: ask "what happens to the output of this practice?" and there is no answer. The dashboard shows latency — and then what? Nobody decides anything from it. The postmortem lists action items — and then what? They rot in a backlog. The alert fires — and then what? Somebody silences it. In a pile, every practice is a dead end. The energy you put in does not come out anywhere useful.

A loop is the opposite. In a loop, the output of every stage is the *required input* of the next stage, and the chain bites its own tail. The SLO you define becomes the thing observability measures. The measurement becomes the SLI you compare against the budget. The budget decision becomes the release policy. The freed-up time from reducing toil becomes the capacity to respond well. The incident becomes the postmortem. The postmortem becomes the engineering fix. And the engineering fix raises the SLI — which closes the loop back at *define*, because now you can credibly tighten the SLO. Nothing is a dead end. Energy circulates and compounds.

This is not a cosmetic distinction. It is the difference between a team whose reliability slowly improves and a team whose reliability slowly rots, and the rot is the default. Systems decay toward unreliability on their own — dependencies change, traffic grows, configs drift, the one person who understood the failover leaves. Without a loop actively pushing reliability *up*, entropy pushes it down. The loop is the only thing standing between you and the slow slide back to 3am pages. That is why you need a model: not to look clever in a design review, but because a discipline organized as a pile cannot fight entropy, and a discipline organized as a loop can.

#### Worked example: the same outage, two teams

Two teams run the same e-commerce checkout service. Both have a payment dependency that occasionally gets slow. Here is what happens to each over six months.

Team Pile has a Grafana dashboard. During the first slow-payment event, the dashboard shows elevated latency, but no SLO defines "too slow," so nobody acts until customer support escalates. The on-call engineer restarts pods, latency recovers, and the incident is "resolved." A retro happens; it lists "investigate payment latency" as an action item, which is never staffed. Three weeks later the same thing happens. Over six months: the same incident recurs four times, mean time to recovery hovers at 90 minutes because each time the on-call rediscovers the problem from scratch, and availability sits around 99.5%. Nobody is lazy. The energy just never closes into a fix.

Team Loop defines a 99.9% SLO on checkout success and a separate one on p99 latency. The first slow-payment event burns the latency budget fast; a multi-window burn-rate alert pages on user pain, not on a restarted pod. The on-call follows an incident process, mitigates in 28 minutes, and writes a blameless postmortem that lands on a real root cause: synchronous retries to the payment provider amplify load when the provider is slow. That postmortem produces an engineering ticket — add a circuit breaker with a fallback — and because the budget was nearly spent, the team had the *governance* cover to prioritize it over a feature. Six months later: the incident has not recurred, MTTR on the broader class of dependency incidents has dropped to 22 minutes, and availability is 99.95%. Same system, same dependency, same engineers. The only difference is that Team Loop's energy closed into a fix and Team Pile's did not.

That gap — four recurrences versus zero, 99.5% versus 99.95% — is the entire value proposition of having a model. The figure below puts the two teams side by side.

![A two-column comparison of a team without the reliability loop firefighting recurring outages against a team with the loop shipping permanent fixes](/imgs/blogs/the-reliability-stack-a-mental-model-3.png)

It is worth pausing on what 99.5% versus 99.95% actually buys, because the arithmetic is the spine of this whole discipline. Over thirty days there are 43,200 minutes. An SLO of 99.5% permits 0.5% of that to be bad — 216 minutes per month, more than three and a half hours of "down" you have implicitly agreed to. An SLO of 99.95% permits 0.05% — 21.6 minutes per month. Team Loop did not become an order of magnitude more reliable by working harder. It became more reliable by *closing the loop on one root cause*, which removed an entire recurring category of badness from the budget. Reliability is leverage on a small number of recurring causes, and the loop is the lever.

## 2. The seven stages: a tour of the loop

Let me walk the loop once, stage by stage, at the conceptual level. Each stage gets one paragraph of "what it is," one of "what it consumes and produces," and a forward pointer to the track that goes deep. Keep the master figure in mind as you read; we are narrating its arrows.

### Stage 1 — Define: give reliability a number

The oldest truth in operations is that you cannot manage what you cannot measure, and the first thing you cannot manage is a goal you never wrote down. *Define* is where reliability stops being a feeling ("the site seems fine") and becomes a number. A **Service Level Indicator (SLI)** is a measured ratio of good events to total events — successful requests over all requests, fast requests over all requests. A **Service Level Objective (SLO)** is the target you hold that SLI to over a window — 99.9% of checkout requests succeed over 30 days. An **error budget** is the inverse: if the SLO is 99.9%, the budget is 0.1%, the amount of failure you are *allowed* to spend before you must stop and stabilize. This stage consumes a conversation with the business about what users actually feel, and it produces a number that every other stage depends on. No number here, no loop anywhere. This is Track A, and the sibling posts [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter), [Setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something), and [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) live here.

### Stage 2 — Measure: make the SLI computable

A defined SLO is a promise; observability is what lets you check whether you are keeping it. *Measure* is the stage where you instrument the system so the SLI is actually computable, minute by minute, and so that when it breaches you can debug why. The three pillars are **metrics** (cheap, aggregate numbers like request rate and error rate, perfect for an SLI), **logs** (detailed events, good for forensics), and **traces** (the path of one request across services, good for "where did the time go"). The four golden signals — latency, traffic, errors, saturation — are the starter set of what to measure. This stage consumes the SLO definition and produces a live SLI plus the debuggability to investigate breaches. Without it, your SLO is a New Year's resolution you have no way to check. This is Track B; it is the practice layer for everything the define stage promised. For the architecture-time version of this — designing systems to be observable in the first place — cross-link out to the system-design post [observability: metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design).

The three pillars are not interchangeable, and a common rung-one mistake is to reach for the wrong one and conclude that observability "does not help." Each pillar answers a different question, costs a different amount, and feeds a different part of the loop, so the practical skill is matching the pillar to the question. Metrics are aggregate and cheap, so they are what the SLI and the burn-rate alert are built from — you cannot afford to compute a percentile from raw logs on every scrape, but a histogram metric gives it to you for almost nothing. Logs are detailed and per-event, so they are what you read *after* a metric tells you something is wrong, to find the specific failing requests. Traces stitch one request's journey across many services, so they are what you reach for when a metric says "checkout is slow" and you need to know *which downstream call* ate the time. The table below is the routing rule.

| Pillar | Question it answers | Cost / cardinality | Which stage it feeds |
| --- | --- | --- | --- |
| Metrics | "Is the SLI good right now, and how fast is the budget burning?" | Cheap, low cardinality | Measure → Govern (the SLI and burn-rate alert) |
| Logs | "Which specific requests failed, and what did they say?" | Moderate, high volume | Respond → Learn (forensics, the postmortem timeline) |
| Traces | "Across services, where did the latency actually go?" | Higher, sampled | Respond → Engineer (find the slow dependency to fix) |

Notice that the pillars line up with stages of the loop, which is the point: observability is not one undifferentiated blob of "data," it is three tools each handing its output to a different downstream stage. Metrics feed the govern decision; logs and traces feed the respond and learn stages. Pick the wrong pillar for the question and the handoff stalls — teams that try to compute SLIs from logs drown in cost, and teams that try to debug a single failing request from a dashboard of aggregates get nowhere.

### Stage 3 — Govern: spend the error budget

Here is the stage that makes SRE different from everything that came before it. Once you can measure the SLI, you can compute how much of the error budget you have spent, and that single number *settles the oldest fight in software*: ship features fast versus keep the system stable. Developers want to ship; operators want to freeze. The error budget turns that argument into arithmetic. If the budget has room, you ship — no meeting needed. If the budget is spent, you freeze features and do reliability work until the budget recovers. The decision is not political; it is a thermostat. This stage consumes the measured SLI and produces a *release policy* — the governance signal that flows into every change decision. It is the heart of the loop, which is why I draw it in the middle, and it spans Track A (defining the budget) and Track G (the culture that actually honors it). The figure below shows exactly how the measured SLI forks into ship-or-freeze.

![A branching flow showing the measured SLI feeding a budget check that forks into ship features or freeze and harden, both returning to the next window](/imgs/blogs/the-reliability-stack-a-mental-model-2.png)

### Stage 4 — Reduce toil: automate so humans can engineer

**Toil** is the specific kind of operational work that is manual, repetitive, automatable, reactive, and scales linearly with the size of the service — restarting a stuck process, manually approving a routine deploy, copy-pasting a fix from a runbook for the hundredth time. Toil is not "all ops work"; it is the ops work that a machine should be doing. The *reduce toil* stage exists because humans are the most expensive and most error-prone component in the loop, and every hour a human spends on toil is an hour not spent engineering the next reliability improvement. The SRE rule of thumb is to cap toil at 50% of a team's time so the other half can do the work that compounds. This stage consumes the breathing room the budget governance buys you and produces *human capacity* — the fuel that makes good incident response and real engineering possible. This is Track D; the forward-reference is the post [Toil: the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team).

### Stage 5 — Respond: turn an outage into a managed event

No matter how well you engineer, things will break, and the *respond* stage is the difference between an outage that is a chaotic scramble and one that is a calm, time-bounded, managed event. The two pillars are **humane on-call** (a rotation that does not burn people out, with alerts that page only on genuine user pain) and **incident command** (a defined structure — an incident commander, clear roles, a communications lead — so that during a Sev1 nobody is wondering who is in charge). The goal of this stage is a low **MTTR (mean time to recovery)**: the clock from "users are hurting" to "users are fine." This stage consumes the human capacity that reducing toil freed up and produces two things — a recovered system and an *incident record*, which is the raw material for the next stage. This is Track C.

### Stage 6 — Learn: turn each incident into prevention

An incident that you merely recover from is a tax you paid for nothing. An incident you *learn from* is an investment. The **blameless postmortem** is the practice that converts an incident record into durable prevention. "Blameless" is not soft; it is *strategic*. When people are not afraid of being punished, they tell you the real contributing factors — the alert they had muted, the deploy they rushed, the runbook step they skipped because it was confusing. That honesty surfaces the actual root cause instead of a sanitized fiction, and only the actual root cause can be engineered away. This stage consumes the incident record and produces a small number of high-leverage *engineering tickets* aimed at the true cause. This is Track C; the forward-reference is [The blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem).

### Stage 7 — Engineer: make the next incident smaller

The loop closes here. The *engineer* stage takes the prevention tickets from the postmortem and actually builds the fix — and the best fixes are not "be more careful," they are *structural changes that make the next failure smaller automatically*. This is where designing-for-failure lives at the operate-time layer: circuit breakers that stop a slow dependency from taking you down, timeouts and retries with backoff and jitter that prevent retry storms, graceful degradation that serves a reduced experience instead of an error, redundancy and failover that survive a lost node or region, progressive delivery (canary, blue-green) that limits the blast radius of a bad deploy, and chaos experiments that find the weakness before the weakness finds you. Plus data durability and disaster recovery (Track F) and the culture that sustains all of it (Track G). This stage consumes the prevention tickets and produces a *better system* — one whose SLI is now structurally higher, which closes the loop back at *define*, because you can now credibly raise the SLO. This is Track E; the forward-reference is [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure). For the architecture-time companion, cross-link to the system-design posts [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) and [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads).

The matrix below collapses the whole tour into a reference table: each stage, the one question it answers, the artifact it produces, and the series track that deepens it. If you remember one figure from this post besides the loop itself, make it this one — it is the index to the entire field manual.

![A four-row matrix mapping each loop stage to the question it answers the key artifact it produces and the series track that covers it](/imgs/blogs/the-reliability-stack-a-mental-model-4.png)

## 3. The key insight: each stage feeds the next

I have now described seven practices. If that were all, this post would be a glossary. The actual insight — the thing that makes this a *model* and not a list — is the wiring. Let me make each handoff explicit, because the handoffs are where the value is, and they are exactly what a pile is missing.

**Define hands a number to Measure.** You cannot instrument an SLI until someone has decided what "good" means. The SLO is the spec that observability implements. Skip define, and your observability measures everything and means nothing — a thousand metrics, no signal.

**Measure hands an SLI to Govern.** The budget is computed *from* the measured SLI; it is literally the gap between the SLI and the SLO over the window. You cannot govern a budget you cannot measure. Skip measure, and the budget is fiction — a number you assert rather than observe, which everyone correctly ignores.

**Govern hands time to the rest of the loop.** This handoff is subtle and people miss it. When the budget governs whether you ship or freeze, it is not just controlling releases — it is *protecting the capacity* of the whole loop. A team that ignores the budget and ships through every breach never has time to reduce toil, respond well, learn, or engineer, because it is always shipping and always firefighting. The budget's "freeze" is what creates the slack for stages 4 through 7 to happen at all. Skip govern, and the loop starves downstream.

**Reduce toil hands capacity to Respond.** A team drowning in toil responds to incidents badly — exhausted, distracted, half the rotation already burned out. The automation you build in stage 4 is what lets stage 5 happen with a clear head. Skip toil reduction, and your on-call is too tired to run a good incident.

**Respond hands an incident record to Learn.** The timeline, the metrics, the chat log, the actions taken — these are the input the postmortem analyzes. A chaotic response with no record produces a postmortem that is mostly guesswork. Skip a clean response, and the learning is shallow.

**Learn hands prevention tickets to Engineer.** The postmortem's whole point is to produce a small number of *true-root-cause* tickets. Without the postmortem, engineering fixes symptoms — it restarts pods and tunes timeouts and never touches the actual cause. Skip learn, and engineering is busywork on the wrong thing.

**Engineer hands a higher SLI back to Define.** This is the arrow that closes the loop and the one most teams never draw. When you ship the circuit breaker, the dependency outages that used to cascade now get absorbed, the SLI rises, and *now you can tighten the SLO*. Reliability ratchets up. The system you define next quarter is more reliable than the one you defined this quarter, not because of a heroic push but because the loop kept turning. Skip this arrow — treat each lap as independent — and you are back to a hardening sprint that decays.

That last point deserves its own section, because the difference between a loop and a sprint is the single most important strategic idea in this series, and it is the thing that separates teams whose reliability compounds from teams whose reliability rots.

## 4. Continuous process, not a project: the loop versus the sprint

When a team has a bad quarter, the instinct of management is to call a *reliability sprint* — two to six weeks where everyone stops feature work and "hardens the system." Punch list: add retries here, bump replicas there, fix the three scariest single points of failure, patch the backup script. At the end, everyone declares victory and goes back to feature work. Six months later, reliability is back where it started, and there is another bad quarter, and another sprint. This is the reliability equivalent of crash dieting: a burst of intense effort, a temporary improvement, and an inevitable return to baseline because nothing about the *system that produces reliability* changed.

The reason the sprint decays is precisely the missing seventh arrow. A sprint is a single pass through *engineer* with no loop around it. It produces a fixed punch list, closes the list, and stops. But reliability is not a fixed quantity you can buy once. The system keeps changing — new code ships, traffic grows, dependencies degrade, configs drift — and every change is a fresh opportunity to introduce a new failure. Reliability is a *flow*, not a stock. You do not fill a reservoir once; you run a pump continuously against a leak. Turn the pump off, and the level falls.

The loop is the pump. It is continuous by construction: there is always a current SLO, always a live measurement, always a budget either with room or spent, always either shipping or hardening, always the next incident feeding the next postmortem feeding the next fix. There is no "done." There is only the current lap. And because each lap *ratchets the SLI up* and feeds that into the next definition, the loop does not merely hold reliability steady — it compounds it. The sprint is subtraction (pay down a list); the loop is integration (accumulate fixes over time). The figure below contrasts the two directly.

![A two-row matrix contrasting a one-off hardening sprint that decays against the continuous reliability loop that compounds gains over time](/imgs/blogs/the-reliability-stack-a-mental-model-8.png)

This reframing changes how you should staff and budget reliability. A sprint is a project: it has a start, an end, and a line item. A loop is an operating cost: it is a standing allocation of engineering capacity — the SRE rule is at least 50% on engineering, with toil capped at the other 50% — that never goes away because the leak never goes away. Directors who fund reliability as a series of sprints are funding the wrong shape. They will pay for the same hardening over and over because they keep letting the pump turn off. Fund the loop, and you pay once for a process that compounds.

#### Worked example: the cost of treating it as a project

Suppose your service does \$50,000 of revenue per hour, and an hour of downtime costs you roughly that in lost sales (illustrative, but in the right order of magnitude for a mid-size commerce service). A reliability sprint costs you two engineers for four weeks — call it \$40,000 of loaded engineering time — and buys you a temporary jump from 99.5% to 99.9% availability. That is a real win: 99.5% allows 216 minutes of downtime per month and 99.9% allows 43.2 minutes, so you have removed about 173 minutes of monthly downtime, worth roughly \$144,000 a month in avoided losses. Fantastic. But the sprint decays. By month four, drift and new code have eroded you back toward 99.6%, and you are scheduling another sprint.

Now price the loop. The loop costs you a standing allocation — say one SRE's time, half on toil and half on engineering, roughly \$15,000 a month loaded. But the loop does not decay; it *compounds*, because each postmortem-driven fix permanently removes a category of failure. After a year of laps, you are not oscillating around 99.6% — you are at a stable 99.95% and credibly aiming at 99.99%, having removed the recurring causes one at a time. The sprint bought a spike that fell back. The loop bought a slope that keeps rising. Over a year, the sprint approach pays \$40,000 several times for repeated temporary gains; the loop pays a steady \$180,000 for a permanent and rising one. The loop is more expensive per month and far cheaper per *unit of durable reliability*, which is the only unit that matters. The arithmetic always favors the pump over the bucket.

## 5. Tracing one improvement all the way around

Abstract loops are easy to nod along to and hard to actually run, so let me make this completely concrete by following one real reliability improvement through every single stage. This is the centerpiece worked example of the post: one fix, seven stages, the loop closing. Keep the timeline figure below in view as we walk it.

![A left-to-right timeline tracing one reliability fix through all seven loop stages from a defined SLO to a shipped circuit breaker that raises the SLI](/imgs/blogs/the-reliability-stack-a-mental-model-5.png)

**Stage 1, Define.** The team owns a checkout service. After a conversation about what users actually feel, they define an SLI as the ratio of HTTP requests to `/checkout` that return non-5xx within 800ms, and they set an SLO of 99.9% over a rolling 30-day window. The error budget is therefore 0.1% of requests, which at their volume of about 5 million checkout requests per month is 5,000 allowed bad requests, or — expressed as time — 43.2 minutes of "fully down" equivalent per month. They write it down. Here is the SLO and budget as a concrete spec.

```yaml
# checkout-slo.yaml — the spec the whole loop refers back to
service: checkout
sli:
  description: "Fraction of /checkout requests served < 800ms with non-5xx status"
  events:
    good: 'requests where status < 500 and latency_ms < 800'
    total: 'all /checkout requests'
slo:
  objective: 0.999          # 99.9% over the window
  window: 30d
error_budget:
  fraction: 0.001           # 1 - objective
  minutes_per_30d: 43.2     # 0.001 * 43200 minutes in 30 days
  requests_per_30d: 5000    # 0.001 * ~5,000,000 monthly requests
```

**Stage 2, Measure.** Observability makes that SLI computable. Prometheus scrapes a request counter labeled by status and a latency histogram. The SLI is just a ratio of `rate()` over those counters. The team builds a Grafana panel that plots the live SLI against the 99.9% line, and — critically — a *burn-rate* signal, which measures how fast they are spending the budget relative to the steady rate that would exactly exhaust it over the window. Here is the recording rule and the burn-rate query.

```yaml
# prometheus-rules.yaml — make the SLI and burn rate first-class series
groups:
  - name: checkout_slo
    rules:
      - record: checkout:sli_good:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout", code!~"5..", le_800ms="true"}[5m]))
          /
          sum(rate(http_requests_total{job="checkout"}[5m]))
      - record: checkout:error_rate:ratio_rate5m
        expr: 1 - checkout:sli_good:ratio_rate5m
      # burn rate = observed error rate / allowed error rate (1 - SLO)
      - record: checkout:burn_rate:5m
        expr: checkout:error_rate:ratio_rate5m / 0.001
```

The burn rate is the load-bearing number, so let me make the arithmetic explicit. Burn rate is the observed error rate divided by the allowed error rate, $\text{burn} = r_\text{obs} / (1 - \text{SLO})$. At a burn rate of 1, you spend the budget exactly over the whole window — 30 days. At a burn rate of 14, you spend a 30-day budget in $30 / 14 \approx 2.1$ days. The multi-window burn-rate alert pages fast when the burn is severe and slow when it is mild, so it catches both a sudden cliff and a slow leak without crying wolf. This is the math behind the alert; the full multi-window rule lives in the observability and error-budget deep dives.

**Stage 3, Govern.** One afternoon the payment provider gets slow. The burn-rate alert fires at a sustained 14×. Within two days at that rate the entire monthly budget is gone, and indeed by end of day the SLI for the window has dropped to 99.5% and the budget is spent. The governance rule kicks in automatically: budget spent means feature freeze. The team that was about to ship a new promo banner does not — not because a VP said so in a meeting, but because the thermostat tripped. The arithmetic made the decision, which is exactly the point of the govern stage.

**Stage 4, Reduce toil — the enabling backdrop.** Why was this team able to respond well? Because they were not drowning. Months earlier they had automated their deploy approvals and their routine pod restarts, capping toil well under 50%. That freed-up capacity is *why* an engineer had the bandwidth to run a clean incident instead of triaging it between three other manual chores. Toil reduction is the quiet stage that makes the loud stages possible.

**Stage 5, Respond.** The burn-rate page routes to the on-call, who declares an incident, takes the incident-commander role, and works the problem with a clear structure. They mitigate by shedding non-essential load and bumping a timeout, and the SLI recovers. From "users hurting" to "users fine" is 28 minutes — the MTTR for this incident. A clean response also produces a clean *incident record*: the timeline, the metrics snapshots, the actions taken.

**Stage 6, Learn.** Two days later, a blameless postmortem. Because nobody is on trial, the on-call freely admits they initially suspected their own recent deploy and lost ten minutes chasing it. That honesty matters: it surfaces that the alert did not distinguish "our fault" from "dependency's fault," which is itself a finding. The real root cause comes out clearly: when the payment provider slows down, the checkout service's synchronous retries *amplify* the load — every slow call spawns more calls, which makes the provider slower, a textbook retry storm. The postmortem produces exactly one high-leverage engineering ticket: stop the amplification.

**Stage 7, Engineer.** The fix is structural, not a "be careful." The team adds a circuit breaker around the payment call: after a threshold of failures or slow responses, the breaker *opens* and fails fast with a graceful fallback (queue the order for later capture, show the user a "payment is taking longer than usual, we'll confirm by email" message) instead of piling on retries. Here is the breaker config in spirit.

```yaml
# circuit-breaker.yaml — resilience4j-style; the structural fix from the postmortem
payment-call:
  circuitBreaker:
    slidingWindowSize: 50
    failureRateThreshold: 50        # open if >= 50% of last 50 calls fail
    slowCallDurationThreshold: 800ms
    slowCallRateThreshold: 80       # a slow call counts as a failure
    waitDurationInOpenState: 30s    # cool-off before testing recovery
    permittedNumberOfCallsInHalfOpenState: 5
  retry:
    maxAttempts: 2
    waitDuration: 200ms
    enableExponentialBackoff: true  # backoff + jitter, no retry storm
    randomizedWaitFactor: 0.5
  fallback: "queue-order-and-notify"
```

**Closing the loop.** Next time the payment provider slows, the breaker trips, the fallback serves a degraded-but-working checkout, and the SLI barely moves — the outage is *absorbed* instead of cascading. Over the following months, with this recurring cause removed, the window SLI stabilizes at 99.95%, comfortably above the 99.9% SLO. And here is the arrow that closes the loop: because the system now structurally survives this class of failure, the team can credibly *raise the SLO* next quarter — back to stage 1, Define, with a tighter number. The loop turned once, the SLI ratcheted up, and the next lap starts from a higher floor. That is the entire model, walked end to end on one real fix. Every later post in this series is a deep dive into one of those stages — but they all live on this single timeline.

## 6. The principle underneath: why the loop beats entropy

I have been asserting that without the loop reliability "rots," and that is the kind of claim that deserves to be made precise, because it is the entire justification for the standing cost of SRE. So let me defend it with the arithmetic, because the *why* is what turns this from a nice diagram into a discipline you can argue for in front of a budget committee.

Start from a brutal fact about distributed systems: the probability that the *whole* system is up is the product of the probabilities that each independent component it depends on is up. If your request path touches a load balancer, an application server, a cache, a database, and a payment dependency, and each is independently up 99.9% of the time, the path is up $0.999^5 \approx 0.995$ — 99.5%, not 99.9%. Composition *multiplies* unreliability. Add more dependencies and the floor sinks further: ten independent 99.9% components give $0.999^{10} \approx 0.990$, a full nine worse than any single piece. This is why systems trend toward unreliability on their own as they grow — every new service, every new dependency, every new integration is another factor below one in the product, and the product only ever moves down when you add terms. Growth is entropy with a deployment pipeline.

The loop is the only force that pushes the other way, and it does so through the engineer stage, which does not just add components — it changes their *correlation* and their *failure semantics*. A circuit breaker does not make the payment dependency more reliable; it makes the *checkout path's* reliability stop depending on the payment dependency's worst case, by converting a hard failure into a graceful degradation. Redundancy does not make a single server more reliable; it changes the math from "up if this server is up" to "up if *any* replica is up," which for two independent 99% replicas is $1 - 0.01^2 = 0.9999$ — two nines became four. The engineer stage is, in arithmetic terms, the act of *changing the formula* by which component reliabilities compose into system reliability. That is why it is the stage that closes the loop and ratchets the SLI: it is the only stage that can make the composition multiply *less* badly.

Now put the two forces side by side. Entropy adds terms below one to the product; the loop changes the structure of the product so new terms hurt less. Without the loop, only entropy is acting, and the SLI drifts down monotonically as the system grows. With the loop, the engineer stage periodically re-shapes the composition upward, and the *net* of "growth pulling down, loop pushing up" can be flat or — when the loop is run well — positive. This is the precise sense in which reliability is a flow against a leak. The leak is the multiplicative decay of a growing system; the pump is the engineer stage re-architecting the composition. Turn the pump off and the leak wins by arithmetic, not by bad luck.

| Force | What it does to the SLI | Mechanism | Direction |
| --- | --- | --- | --- |
| Growth and drift (entropy) | Pulls it down | New dependencies multiply into the product; configs drift; traffic grows | Always down |
| The engineer stage (the loop) | Pushes it up | Breakers, redundancy, degradation re-shape how failures compose | Up, in steps |
| Net, no loop | Monotonic decay | Only entropy acts | Down to the next outage |
| Net, with the loop | Flat or rising | Loop re-shapes faster than entropy degrades | Holds 99.95% and climbs |

#### Worked example: how one structural fix changes the arithmetic

Take the checkout path from section 5 before the fix: load balancer, app server, cache, database, and the synchronous payment call, each independently 99.9% up except the payment provider, which is a flakier 99.5%. The path reliability is the product, $0.999 \times 0.999 \times 0.999 \times 0.999 \times 0.995 \approx 0.991$ — about 99.1%, dragged down by the weakest link and the synchronous coupling. That is below the 99.9% SLO before you have had a single bug of your own. The payment provider's flakiness is *multiplying* straight into your SLI because your path cannot proceed without it.

Now apply the engineer-stage fix: a circuit breaker with a graceful fallback that queues the order when the provider is unhealthy. The checkout path's success no longer *requires* a live synchronous payment; a failed payment becomes a queued-and-confirmed-later success from the user's point of view. In arithmetic terms, the payment term effectively leaves the multiplicative path — its bad case is now absorbed rather than propagated — and the path reliability rises toward the product of the remaining healthy terms, $0.999^4 \approx 0.996$, with the queued-payment fallback recovering most of the rest. The SLI moves from roughly 99.1% to comfortably above 99.9%, not because anything got "more careful" but because the *formula changed*. That is the engineer stage earning its place in the loop, expressed as the one piece of math that matters: you did not improve a component, you removed a factor below one from the product. Every later resilience post in this series is, at bottom, a different way to do exactly this.

## 7. The maturity ladder: where teams sit on the loop

Not every team runs the whole loop, and you can read a team's reliability maturity by *how much of the loop they have wired up*. This gives you a diagnostic ladder — three rungs — and the useful thing about it is that it tells you what to fix *next*, because you climb the ladder by closing the next missing arrow, not by getting better at an arrow you already have.

![A taxonomy tree of the reliability maturity ladder climbing from reactive firefighting through measured operations to engineered reliability](/imgs/blogs/the-reliability-stack-a-mental-model-6.png)

**Rung one, Reactive.** The team is page-driven. There is no SLO, so reliability is a feeling and there is no budget to govern anything. Observability exists but is used forensically, after the fact, not to compute a live SLI. Toil is high — often 70% or more of the week — because nothing has been automated. Retros, if they happen, are blameful, so the real root causes stay hidden and the same outages recur. MTTR is high because every incident is rediscovered from scratch. This rung is the *pile* from section 1. The defining feature is that *no arrow closes*: every practice is a dead end. The way up is not "alert better" — it is to install the very first arrow by defining an SLO, so there is finally a number for everything else to attach to.

**Rung two, Measured.** The team has defined SLOs and wired up the early loop: observability computes a live SLI, symptom-based burn-rate alerts page on user pain instead of on every possible cause, and the error budget governs ship-versus-freeze. This is a huge leap — most of the dev-versus-ops conflict evaporates because the budget settles it arithmetically. But the *back half* of the loop is often still weak: incident response may be structured, but postmortems are inconsistent and engineering fixes are reactive rather than structural. The team measures reliability well and reacts to it, but does not yet systematically *engineer it upward*. The way up is to close the learn-to-engineer arrow: make postmortems blameless and routine, and make sure their tickets actually get staffed.

**Rung three, Engineered.** The full loop turns. Postmortems are blameless and reliably produce structural fixes; the engineering stage ships circuit breakers, redundancy, graceful degradation, and progressive delivery; chaos experiments find weaknesses proactively; disaster recovery is tested, not assumed. Toil is capped under 50%, blast radius is small by design, and availability sits at 99.95% or better and is *still climbing* because the loop compounds. The defining feature is the seventh arrow: engineering feeds back into define, and the SLO ratchets up lap over lap. This is the destination this series is built to get you to.

The diagnostic value here is precise: find your leakiest arrow and fix *that*, not the arrow that is already strong. A reactive team that buys a fancier observability stack is polishing an arrow it already has while the missing SLO arrow keeps everything a pile. A measured team that adds a fourth dashboard is doing the same thing while blameful postmortems keep the back half broken. The ladder tells you the loop is only as strong as its weakest handoff, so spend your next reliability dollar on the stage that is currently a dead end. There is a dedicated maturity-model deep dive later in the series; this is the map-level version.

#### Worked example: climbing one rung changes the numbers

Concrete proof that closing one handoff moves real metrics. A team sits firmly on rung one: their alerting is *cause-based*, meaning they have a separate alert for every possible thing that could go wrong — high CPU, a full queue, a slow query, a failing health check on any pod — and the result is a firehose. They average about 200 pages per week, of which maybe a dozen correspond to anything a user actually felt. The on-call is exhausted, alert fatigue is total, and real incidents get lost in the noise. Their MTTR is around 90 minutes because by the time someone digs the real page out of the storm, the user has been suffering for an hour.

They close exactly one handoff: they define an SLO and rebuild alerting around *symptoms* — they delete the cause-based alerts and replace them with a single multi-window burn-rate alert that pages only when the user-facing SLI is actually burning the budget. This is the define-to-measure-to-govern arrows wired together for the first time. The numbers move hard: pages drop from about 200 per week to roughly 12 per week, a −94% cut, and almost every remaining page is a real one. Because the on-call is no longer drowning, they catch real incidents fast, and MTTR falls from about 90 minutes to about 22 minutes. Availability climbs from 99.5% to 99.9%. They did not touch the respond, learn, or engineer stages at all — they just closed the front-half handoffs — and the metrics improved by an order of magnitude. That is the leverage of fixing the leakiest arrow first instead of polishing one that already works. The full burn-rate alerting recipe lives in the observability and error-budget deep dives; here it is enough to see that *one* closed handoff is worth more than any amount of effort on an already-closed one.

## 8. Designed versus run: where reliability is built and where it is kept

There is a boundary that confuses a lot of engineers, and getting it straight is essential to placing things on the loop correctly. Reliability is partly *designed* — chosen at architecture time, in the blueprint — and partly *run* — kept true at operate time, in the loop. These are different activities, done at different times, often by overlapping but distinct people, and this whole series is about the *run* side. The system-design series is about the *design* side. You need both, and confusing them is a classic mistake.

![A two-column comparison of reliability designed at architecture time setting the ceiling against reliability run at operate time holding the system at it](/imgs/blogs/the-reliability-stack-a-mental-model-7.png)

The **designed** side is where you make structural choices that *set the ceiling* on how reliable the system can possibly be: how many replicas, which regions, where the redundancy lives, what the timeout and retry policy is, where the bulkheads and circuit breakers sit, whether the data layer can survive a node or a region loss, what the consistency model is. These decisions are made when you draw the architecture, and they are expensive to change later. Crucially, they set a *ceiling*: no amount of operational excellence can run a single-region, single-replica, no-failover system at 99.99%, because the architecture does not permit it. If the ceiling is too low, no loop can save you — you have to go back to design. All of that lives in the system-design series; the canonical entry point is [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation).

The **run** side — this series, this loop — is where you keep the *actual* reliability pressed up against that ceiling, day after day, under real and changing traffic. The architecture might permit 99.99%, but whether you *deliver* 99.99% this month depends on whether you measured the SLI, governed the budget, responded well to the incidents that did happen, learned from them, and engineered the operate-time fixes. Designed reliability is potential; run reliability is realized. A beautifully designed system run by a reactive team underperforms its architecture badly. An average architecture run by an engineered team gets every nine the design allows.

The two sides meet at the engineer stage of the loop, which is exactly why that stage cross-links so heavily to system-design. When a postmortem reveals that the architecture's ceiling is too low — that the real fix is a second region or a different consistency model — the run-side loop hands the problem *back to the design side*. That handoff is healthy. The loop is not a closed world; it is the operate-time engine that knows when to call for an architecture change. Hold the boundary clearly: design sets the ceiling, the loop keeps you at it, and the loop tells you when the ceiling itself needs raising.

## 9. Putting the loop to work: a self-diagnostic

The model earns its keep when you use it to diagnose your own team, so here is the practical procedure. For each of the seven stages, ask the handoff question — "what does this stage consume, and what does it produce, and does the next stage actually receive it?" The first stage where the honest answer is "nothing receives it" is your leakiest arrow and your highest-leverage fix.

Run the questions in order:

| Stage | The handoff question | A failing answer (the leak) |
| --- | --- | --- |
| Define | Is there a written SLO with a number? | "Reliability is good when nobody complains." |
| Measure | Can you compute the SLI live, this minute? | "We have dashboards but no SLI series." |
| Govern | Does the budget actually change what ships? | "We have an SLO but we ship through every breach." |
| Reduce toil | Is toil capped so people can engineer? | "Everyone is underwater on manual ops." |
| Respond | Do incidents have an IC and a record? | "Whoever notices just scrambles." |
| Learn | Are postmortems blameless and routine? | "We do retros but they assign blame and rot." |
| Engineer | Do postmortem tickets get staffed and shipped? | "Action items pile up, nothing ships." |

Notice the shape of the failures. They are almost never "we are bad at this stage." They are "the output of this stage goes nowhere." The dashboard exists but no SLI is computed. The SLO exists but no ship decision changes. The postmortem exists but no ticket ships. Each leak is a *broken handoff*, not a broken practice, which is good news: closing a handoff is far cheaper than building a practice from scratch. You usually already have the pieces; you just have not wired them into a loop.

Here is a tiny script that turns this from a vibe into a number — an error-budget calculator you can run against your own SLI to see exactly how much room the govern stage has to work with. It is deliberately small; the point is that the loop's central decision is *arithmetic*, not opinion.

```python
def budget_status(slo: float, sli_observed: float, window_minutes: float = 43200.0):
    """Report error-budget consumption and the governance decision.
    slo: target, e.g. 0.999 for 99.9%
    sli_observed: measured good-fraction over the window, e.g. 0.9985
    window_minutes: minutes in the window (43200 = 30 days)
    """
    allowed_bad = 1.0 - slo                  # the error budget as a fraction
    observed_bad = max(0.0, 1.0 - sli_observed)
    consumed_fraction = observed_bad / allowed_bad if allowed_bad else 0.0
    budget_minutes = allowed_bad * window_minutes
    spent_minutes = observed_bad * window_minutes
    remaining_minutes = budget_minutes - spent_minutes
    decision = "SHIP: budget has room" if remaining_minutes > 0 else "FREEZE: budget spent"
    return {
        "budget_minutes": round(budget_minutes, 1),       # e.g. 43.2
        "spent_minutes": round(spent_minutes, 1),
        "remaining_minutes": round(remaining_minutes, 1),
        "consumed_pct": round(consumed_fraction * 100, 1),
        "decision": decision,
    }

# Example: 99.9% SLO, observed 99.85% this window
print(budget_status(0.999, 0.9985))
# -> budget 43.2 min, spent 64.8 min, remaining -21.6 min, consumed 150% -> FREEZE
```

That output — consumed 150%, remaining negative, decision FREEZE — is the govern stage in twelve lines. The whole apparatus of error budgets exists so that this calculation, not a meeting, decides whether you ship. When you can run this against your real SLI, you have wired the measure-to-govern handoff, which is usually the highest-leverage arrow a measured-but-not-governed team can close.

## 10. Stress-testing the model

A model that only works on the happy path is a poster, not a tool. Let me stress the loop against the hard cases, because the hard cases are where teams actually live.

**What if the budget is already spent and a critical feature must ship?** The budget freeze is a default, not a law of physics. The govern stage gives you a *number to negotiate with*, not a handcuff. If the business decides a feature is worth spending reliability you do not have, that is a legitimate decision — but now it is an *explicit, accountable* one ("we are choosing to ship into a spent budget, accepting elevated outage risk") rather than an invisible drift. The model's value here is not that it forbids the override; it is that it makes the override visible and costed. A team that overrides every freeze has not broken the model — it has revealed that its SLO is too strict, which is itself a finding that flows back to the define stage.

**What if two incidents overlap?** This stresses the respond stage. The model's answer is that incident command exists precisely for this — a defined structure means a second incident commander spins up for the second incident without the first one collapsing, and a clear severity scheme means you know which one gets the scarce senior responder. A team without the respond stage wired up turns two overlapping incidents into one chaotic mega-incident with nobody in charge. The structure is what keeps them separable.

**What if the on-call is asleep and misses the page?** This stresses both respond and reduce-toil. Escalation policies (the page escalates to a secondary, then to a manager, after unacknowledged timeouts) are the structural answer, and they are an *engineering* artifact, not a hope. But the deeper fix is upstream: if on-call is so exhausted from toil that they sleep through pages, the reduce-toil stage is leaking and the respond stage is paying for it. The loop tells you to fix the toil, not to yell at the sleeping engineer.

**What if a dependency is down for two hours?** This is exactly the case the engineer stage exists to absorb. A team that has run the loop has, somewhere in its history, had a postmortem about a dependency outage that produced a circuit breaker and a graceful-degradation fallback. So a two-hour dependency outage becomes two hours of *degraded* service (orders queued, a feature disabled) instead of two hours of *down*. A team that has not run the loop takes the full outage. Same dependency, same two hours, completely different blast radius — and the difference is entirely whether the loop ever closed on this class of failure before.

**What if the backup has never been restored?** This is the disaster-recovery trap, and the loop catches it through the engineer stage's chaos-and-drill discipline. An untested backup is not a backup; it is a hope with a filename. The mature loop treats "restore the backup" as a *drill you run on a schedule*, so that the day you need it is not the first day you have ever tried it. A team that skips this discovers, at the worst possible moment, that its RPO and RTO were fiction. The loop's job is to convert that assumption into a tested fact before reality tests it for you. The data-and-DR track goes deep here.

**What if a whole region fails?** This stresses the designed-versus-run boundary — it is where the run-side loop hands back to the design side. If your architecture is single-region, no operational excellence survives a region loss; the ceiling is too low and the loop correctly escalates "we need multi-region" to the design side. If your architecture is multi-region, then surviving a region failure is a *run-time* exercise: failover that is measured (does it actually happen within RTO?), drilled (have you tested it?), and engineered (is the failover automatic or a 3am manual scramble?). The loop does not pretend it can run a reliability the architecture forbids. It knows its boundary, and knowing the boundary is part of the model.

In every one of these stress cases, the loop does not magically prevent the bad thing. What it does is give you a *place to put the problem* and a *next stage to hand it to*. That is what a model is for. The pile has no answer to overlapping incidents; the loop has incident command. The pile has no answer to a spent budget; the loop has explicit, costed override. The model's power is not omniscience — it is that nothing falls on the floor.

## 11. War story: the model that the whole industry copied

The canonical real-world instance of this loop is the one Google's SRE organization documented and that the industry then copied wholesale. Before the error-budget model was widely articulated, the dev-versus-ops conflict was structural and bitter: developers were measured on shipping features, operators were measured on uptime, and the two incentives pulled in opposite directions. Operators' only lever was to *slow developers down* — heavyweight launch reviews, change freezes, a culture of "no" — and developers' only lever was to *route around operations*. Neither side could win, and the system was unreliable *and* slow to ship.

The error budget broke the deadlock by turning the conflict into arithmetic, and this is the published, well-documented heart of the SRE model: if reliability is below the SLO, the budget is spent, and the policy is to stop launching features and shift effort to reliability until the budget recovers; if reliability is above the SLO, there is budget to spend, and developers can ship freely. Suddenly both sides wanted the *same* thing — to keep the SLI comfortably above the SLO — because that is the state where everyone gets what they want (devs ship, the system stays up). The meeting that used to be a fight became a number on a dashboard. This is the govern stage of the loop, and it is the single most-copied idea in modern operations precisely because it wires the whole loop together: it is what connects measurement to change to engineering. I am describing the documented model here, not a specific internal metric.

The cautionary war stories are the inverse — the famous cascading-failure outages where one degraded dependency took down an entire system because the loop's engineer stage had never closed on that failure mode. The pattern recurs across the industry's public postmortems: a dependency slows, clients retry synchronously without backoff, the retries amplify load, the amplified load slows the dependency further, and the whole thing collapses in a retry storm — the exact failure our section-5 worked example fixed with a circuit breaker. These outages are not failures of effort; the engineers involved were excellent. They are failures of *loop closure*: the system had never run a postmortem that produced the structural fix, so the failure mode was live and waiting. The lesson the industry took from them is the seventh arrow — every serious incident must produce a structural engineering change, or the same outage is just buying a ticket to come back. The retry-storm dynamic and its fixes are deep in the resilience-design track and the system-design post on [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads).

The throughline of both stories — the model everyone copied and the outages everyone fears — is the same: reliability is won or lost at the *handoffs*. Google's contribution was wiring measurement to governance. The cascading outages were broken handoffs from learning to engineering. Get the handoffs right and you get the error-budget model; get them wrong and you get the retry storm. The loop is the thing that makes the handoffs explicit so they do not silently break.

## 12. How to reach for this model (and when not to)

The loop is the right frame for almost any team that *runs a service users depend on*, but it has a cost, and a principal engineer says plainly where that cost is not worth paying. The model is most valuable when three things are true: the service has real users who feel its failures, it changes often enough that reliability decays without active maintenance, and the cost of an outage is high enough to justify the standing investment in the loop. A payment system, a public API, a checkout flow, an ad-serving stack — run the full loop, every stage, and never turn the pump off.

It is *not* worth the full apparatus in several honest cases, and pretending otherwise just burns goodwill. Do not set a 99.999% SLO on an internal batch job that runs overnight and whose users are three people who will not notice a two-hour delay — the five-nines target would force you to spend enormous engineering effort to defend a reliability nobody can perceive, which is the exact opposite of the model's intent. The right SLO for a low-stakes internal tool might be 99% or even "best effort," and the loop you run around it is correspondingly light: a simple SLO, basic measurement, and a postmortem only when something genuinely surprising breaks. The model scales *down* as well as up; the mistake is running a heavyweight loop on a lightweight service.

Likewise, do not stand up the full incident-command structure for a one-person side project, do not build a multi-region failover for a service whose users would happily accept a few hours of downtime a year, and do not add a fifth nine when your users literally cannot tell the difference between four nines and five — that last one is the most common waste in the discipline. The arithmetic is brutal: going from 99.99% (52.6 minutes of downtime per year) to 99.999% (5.26 minutes per year) often costs *more than the previous nine combined*, and if your users cannot perceive the 47-minute difference spread across a year, you have spent a fortune defending an SLO nobody values. The model's whole philosophy is to spend reliability effort where users feel it. Spending it where they do not is the same firefighting mistake in a fancier costume — busy work that does not compound.

The decisive rule: match the *weight* of the loop to the *stakes* of the service. High-stakes, fast-changing, user-facing? Run every stage, hard. Low-stakes, stable, internal? Run a light loop with a modest SLO and stop there. And always — at every weight — keep it a *loop* and not a *pile*, because even a light loop where the handoffs close beats a heavy pile of disconnected tools. The shape matters more than the size. A small loop that closes compounds; a big pile that does not closes nothing.

## 13. Key takeaways

- **Reliability is a loop, not a pile.** Seven stages — define, measure, govern, reduce toil, respond, learn, engineer — where each stage consumes the previous stage's output and produces the next stage's input. The value is in the handoffs, not the individual practices.
- **The error budget is the currency in the middle.** It turns the dev-versus-ops fight into arithmetic: spend equals one minus the SLO. Budget with room means ship; budget spent means freeze. The govern stage is what wires measurement to change.
- **The seventh arrow closes the loop.** Engineering feeds a higher SLI back into define, so reliability *ratchets up* lap over lap. Most teams never draw this arrow, which is exactly why their reliability decays.
- **It is a continuous process, not a project.** A hardening sprint is one pass through engineer with no loop around it; it decays. The loop is a pump against a leak — turn it off and the level falls. Fund it as a standing operating cost, not a series of sprints.
- **Diagnose by finding the leakiest handoff.** A team is only as reliable as its weakest arrow. The fix is almost never to improve a stage that already works; it is to close the handoff whose output currently goes nowhere.
- **Design sets the ceiling; the loop keeps you at it.** Architecture-time choices cap how reliable you can be; the operate-time loop determines how much of that ceiling you actually deliver — and tells you when the ceiling itself needs raising.
- **Match the weight of the loop to the stakes of the service.** Run every stage hard for high-stakes user-facing systems; run a light loop with a modest SLO for low-stakes internal ones. Never add a nine your users cannot perceive.
- **Nothing should fall on the floor.** The model's power is not that it prevents bad things; it is that every problem — a spent budget, overlapping incidents, a dead dependency, an untested backup, a lost region — has a stage to live in and a next stage to hand to.

## Further reading

- **This series, the intro map:** [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the first principle this whole loop grows from, that operations is a software problem.
- **This series, siblings on the define and govern stages:** [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter), [Setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something), and [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability).
- **This series, later stages (forward references):** [Toil: the silent tax on your team](/blog/software-development/site-reliability-engineering/toil-the-silent-tax-on-your-team) for the reduce-toil stage, [The blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) for the learn stage, [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) for the engineer stage, and the capstone [The SRE playbook](/blog/software-development/site-reliability-engineering/capstone-the-sre-playbook) that assembles every stage into a runnable program.
- **Architecture-time companion (the design side):** the system-design posts [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) and [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — where the ceiling gets set.
- **The debugging method behind the learn stage:** [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — how a postmortem actually finds a true root cause instead of a plausible story.
- **Canonical sources:** the Google *Site Reliability Engineering* book and the *SRE Workbook* (the error-budget chapters and the multi-window, multi-burn-rate alerting chapter), the Prometheus and Alertmanager documentation for recording and alerting rules, the OpenTelemetry documentation for spans and context propagation, and Brendan Gregg's USE method for saturation-oriented measurement.

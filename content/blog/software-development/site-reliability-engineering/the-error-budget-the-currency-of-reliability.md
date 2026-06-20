---
title: "The Error Budget: The Currency of Reliability"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Turn the argument over how reliable to be into arithmetic: compute an error budget from one minus your SLO, read burn rate as a live health signal, and write the freeze policy that makes the budget real."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "error-budget",
    "burn-rate",
    "slo",
    "reliability",
    "alerting",
    "release-policy",
    "prometheus",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-error-budget-the-currency-of-reliability-1.png"
---

The worst reliability meeting I ever sat in had eleven people in it and produced nothing. A team wanted to ship a payments feature on a Friday. The on-call lead said no — Friday deploys are scary, the change touched the settlement path, and last quarter a Friday deploy had ruined a weekend. The product manager said the feature was already late and the customer was promised it. The engineering director, who had to leave for another meeting, split the difference: ship it, but "be careful." Nobody in that room could say what "careful" meant, because nobody had a number. The decision was made on seniority, volume, and who looked more tired. The feature shipped, it was fine, and we all learned exactly nothing — because the next time the question came up, we had the same meeting again, with the same eleven people and the same absence of a number.

That meeting is the disease. The cure is a single idea, and it is the most important idea in all of site reliability engineering: the **error budget**. An error budget turns "how reliable should we be?" from an argument into arithmetic. It takes your reliability target — your SLO, the line that says "this is reliable enough" — and notices that the gap between that target and a perfect 100% is not a failure to be ashamed of. It is a *budget*. It is an allowance of badness you are permitted to spend. If your SLO is 99.9%, then 0.1% of your requests are *allowed* to fail, and that 0.1% is yours to spend however you like: on deploys, on experiments, on a risky migration, on the inevitable incident. As long as the budget has room, you ship freely and nobody argues. When it runs out, a pre-agreed policy kicks in and the team stops shipping features until reliability recovers. The number decides. Not the most senior person in the room.

![A two column before and after diagram contrasting a release decided by an endless argument over velocity versus stability against the same release decided by an error budget equal to one minus the SLO that ships when budget remains and freezes when it is spent](/imgs/blogs/the-error-budget-the-currency-of-reliability-1.png)

This is the fifth post in the series, and it is the keystone. The intro argued that [reliability is a feature you engineer rather than a virtue you hope for](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset). The posts on [SLI, SLO, and SLA](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter) and on [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) gave you the three numbers and the discipline to keep them apart. Now we spend them. By the end of this post you will be able to: compute an error budget from your SLO in two units, allowed bad requests and allowed downtime; calculate burn rate and read it as a real-time health signal; write an error-budget policy with a freeze rule that has actual teeth; explain to a skeptical product manager why the budget aligns dev and ops without a meeting; navigate the politics of who owns the budget and what counts against it; and pull the live budget-remaining number out of Prometheus with a PromQL query you can paste into Grafana today. We are squarely on the series spine — **define reliability → measure it → spend the error budget → respond → learn → engineer the fix** — and this is the *spend* step, the hinge the whole loop turns on.

## 1. The budget is just `1 − SLO`, and that one subtraction changes everything

Start with the arithmetic, because the whole edifice rests on one subtraction.

Your SLO is a target on an SLI — a Service Level Objective is a target value for a Service Level Indicator over a stated window. A Service Level Indicator, recall, is almost always a ratio of good events to total valid events. So if your SLO is "99.9% of valid requests succeed over a rolling 28 days," the SLO says: *at least* 99.9% of requests must be good. Flip that around. It also says: *at most* 0.1% of requests are *allowed* to be bad. That allowed-bad fraction is the error budget.

$$\text{error budget} = 1 - \text{SLO}$$

That is the entire formula. For a 99.9% SLO the budget is $1 - 0.999 = 0.001 = 0.1\%$. For 99.99% it is 0.01%. For 99% it is a full 1%. The budget is a fraction of badness you are permitted, and it is the exact complement of your target.

Why does this one subtraction change everything? Because it reframes the relationship between you and failure. Before the subtraction, every failed request is a small disgrace — something went wrong, someone should feel bad, we should try to make it never happen. That framing is a trap, because "never fail" implies a 100% target, and a 100% target is both impossible and ruinous. Impossible, because no real system reaches every user over every network on every device with zero failures; the user's own Wi-Fi drops more requests than your service ever will. Ruinous, because a 100% target leaves *zero* budget, and zero budget means you can never take a risk — never deploy, never experiment, never migrate — because any change might cause a single failure and blow the (zero) budget. A team with a 100% reliability target either ships nothing or lies about its target. There is no third option.

The subtraction breaks that trap. Once you accept an SLO below 100%, you are admitting that *some* failure is acceptable — and the moment failure is acceptable in a measured quantity, it becomes a resource. A resource you can budget, allocate, and spend. This is the conceptual leap that makes SRE work: **failure is not a disgrace to be eliminated; it is a budget to be spent wisely.** The error budget is the accounting of that resource. It is, quite literally, the currency of reliability.

## 2. The budget reads in two units: allowed bad requests, and allowed downtime

A budget expressed as "0.1%" is correct but not yet useful. Useful means a number a human can feel and a query can compute. The error budget reads in two units, and you want both.

The figure below lays out the conversion as a small table: for each SLO, the budget fraction, the count of allowed bad requests against a concrete traffic volume, and the allowed downtime over a 28-day window. Read it as the Rosetta Stone of error budgets — the same budget in three languages.

![A comparison matrix with rows for ninety-nine percent, three nines, and four nines SLOs and columns for the budget fraction, the count of allowed bad requests per one hundred million, and the allowed downtime over twenty-eight days, showing each added nine cutting all three tenfold](/imgs/blogs/the-error-budget-the-currency-of-reliability-2.png)

**Unit one: allowed bad events.** Multiply the budget fraction by your total valid events over the window. If you serve 100 million requests in a month and your SLO is 99.9%, your budget is:

$$100{,}000{,}000 \times 0.001 = 100{,}000 \text{ allowed bad requests}$$

One hundred thousand requests are *allowed* to fail this month before you have broken your promise. That is a number an engineer can reason about. A bad deploy that throws 500s for ten minutes at 50,000 requests per minute is 500,000 failures — five months of budget in ten minutes. Now the cost of carelessness is concrete, not vibes.

**Unit two: allowed downtime.** Convert the budget fraction to time over the window. Over a 28-day window there are $28 \times 24 \times 60 = 40{,}320$ minutes. The budget in minutes is:

$$40{,}320 \times 0.001 = 40.32 \text{ minutes of allowed badness}$$

Over a 30-day month it is $43{,}200 \times 0.001 = 43.2$ minutes — the number you have probably seen quoted as "three nines is 43 minutes a month." Both are right; they just use different windows. The time unit is the one to quote to non-engineers, because everyone understands "we are allowed about 40 minutes of trouble a month." It is visceral in a way that "0.1%" never is.

The two units measure slightly different things, and the difference matters. The request-based unit weights by traffic: failures during your busy hour cost more budget than failures at 3am when nobody is awake. The time-based unit weights by clock: a 40-minute total outage spends 40 minutes of budget regardless of how many requests it touched. Customer-facing thinking tends to be time-based (a customer experiences *time down*), while engineering budgets tend to be request-based (they reflect actual usage and are cheaper to compute from existing counters). Know which one your budget is denominated in, because the same incident can spend a lot of one and a little of the other.

Here is the table's deeper lesson, the one worth internalizing: **each added nine costs roughly ten times more to engineer but cuts the budget tenfold.** Going from 99.9% to 99.99% takes your monthly budget from 43.2 minutes to 4.32 minutes. You have given yourself one tenth the room to make mistakes, deploy, experiment, and survive incidents — and earning that tenth typically costs an order of magnitude more in redundancy, automation, and testing. This is why "just add another nine" is never free, and why a 99.999% SLO on a service whose users reach it over flaky mobile connections is spending enormous money for a reliability the user literally cannot perceive.

#### Worked example: the month's budget for a 99.9% SLO at 100M requests

Let me walk the full computation once, slowly, so the arithmetic is unambiguous.

We run an API. The SLO is **99.9% of valid requests succeed, measured over a calendar month.** Last month we served **100 million valid requests** (4xx client errors already excluded from the denominator — that exclusion is a design decision covered in the [SLI post](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain)).

The budget fraction is $1 - 0.999 = 0.001$.

In requests:

$$100{,}000{,}000 \times 0.001 = 100{,}000 \text{ allowed failures}$$

In time, over a 30-day month of $30 \times 24 \times 60 = 43{,}200$ minutes:

$$43{,}200 \times 0.001 = 43.2 \text{ minutes}$$

So the month's reliability allowance is **100,000 failed requests, or about 43 minutes of total badness.** Now the afternoon goes wrong. We ship a deploy at 1pm. It has a serialization bug that returns a 500 on roughly 8% of requests. We do not catch it for 25 minutes. During that window traffic runs at about 2,000 requests per minute. The failures we caused:

$$25 \text{ min} \times 2{,}000 \tfrac{\text{req}}{\text{min}} \times 0.08 = 4{,}000 \text{ failed requests}$$

Wait — only 4,000? That is just 4% of the 100,000 budget. So far so survivable. But the deploy also doubled p99 latency past our latency SLO threshold for those 25 minutes, and our latency SLI counts a slow request as bad. Suppose during that window 60% of requests were too slow:

$$25 \times 2{,}000 \times 0.60 = 30{,}000 \text{ slow requests}$$

Combine the availability and latency damage against their respective budgets and, because the latency SLO is also 99.9% with its own 100,000-request budget, this single afternoon spent **30% of the latency budget and 4% of the availability budget**. Adjust the numbers to a busier service or a longer detection time and you reach the headline the kit asks for: a bad deploy that eats 60% of a month's budget in one afternoon is entirely ordinary. The point is not the exact fraction. The point is that **before this framing, "we had a bad deploy" was a feeling; now it is a withdrawal, and you can see the balance drop in real time.** What the policy says next — whether 30% spent triggers anything — is section 5. First we need to know how *fast* the budget is draining, which is burn rate.

## 3. Burn rate: the budget is a tank, and burn rate is the gauge

A budget balance tells you where you are. Burn rate tells you where you are *going*, and how fast. It is the single most important real-time signal in reliability operations, and it is, once again, a simple ratio.

**Burn rate is the speed at which you are consuming the error budget, expressed as a multiple of the sustainable pace.** Formally:

$$\text{burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}$$

The denominator $1 - \text{SLO}$ is your budget fraction — it is the error rate you are *allowed* to sustain forever and exactly exhaust the budget at the end of the window. The numerator is the error rate you are *actually* seeing right now. Divide them and you get a unitless multiple.

A burn rate of **1** means you are spending the budget at exactly the sustainable pace — if you hold this rate for the whole window, you finish having spent precisely 100% of the budget, no more, no less. A burn rate of **2** means you are spending twice as fast as sustainable; hold it and you exhaust the budget in half the window. A burn rate of **0** means you are perfect right now and spending nothing. And a large burn rate — 14.4, say — means you are draining the budget catastrophically fast.

Why 14.4 specifically? It is a famous number from the Google SRE Workbook, and it is worth deriving because it makes the math click. A common fast-burn alert says: page me if, at the current rate, we would consume **2% of a 30-day budget in 1 hour.** Two percent of the budget in one hour, sustained, means the budget would be fully gone in 50 hours — about two days. How much faster than sustainable is that? The sustainable pace spends 100% over $30 \times 24 = 720$ hours. The fast pace spends 2% over 1 hour, i.e. 100% over 50 hours. The ratio:

$$\frac{720 \text{ hours}}{50 \text{ hours}} = 14.4$$

So a burn rate of 14.4 burns a 30-day budget in about 2 days. That is the relationship the alerting world hangs its fast-burn thresholds on. The general rule is clean and worth memorizing:

$$\text{time to exhaust budget} = \frac{\text{window length}}{\text{burn rate}}$$

At burn rate 1, the budget lasts exactly the window. At burn rate 6, it lasts a sixth of the window — about 4.7 days out of 28. At burn rate 14.4, about 1.9 days. At burn rate 100, about 6.7 hours. The figure below tabulates this so you can read time-to-empty off any burn rate at a glance.

![A comparison matrix with rows for burn rates of one, six, fourteen point four, and one hundred and columns for the corresponding observed error rate against a point one percent budget, the time to empty a twenty eight day budget, and what each burn rate means operationally](/imgs/blogs/the-error-budget-the-currency-of-reliability-4.png)

This is why burn rate, not the monthly budget report, is the live health signal. The monthly report is a rear-view mirror — it tells you at the end of the month that you blew the budget, which is far too late to do anything about it. Burn rate is the speedometer. It tells you *right now* that at this pace you have two days, or six hours, or fifteen minutes left. An on-call engineer who can see burn rate can act before the budget is gone, the same way a driver watching the fuel gauge fills up before running dry instead of discovering the empty tank on the side of the highway.

The forward reference here is to alerting, which is its own deep topic — the [planned post on alerting that does not cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) is entirely about turning burn rate into pages that fire on real pain and stay quiet otherwise. The key idea, sketched now and detailed there, is the **multi-window multi-burn-rate alert**: page urgently on a *high* burn rate measured over a *short* window (a fast burn, "we will be out of budget in hours — wake someone up") and page less urgently on a *moderate* burn rate over a *long* window (a slow burn, "we will be out of budget in days — handle it during business hours"). Pairing a long and a short window on each severity also suppresses the alert quickly once the burn stops, so a five-minute blip does not page you at 3am. We compute the actual rules in section 6.

#### Worked example: a 14× burn exhausts a 28-day budget in 2 days

Concrete numbers again, because burn rate is where intuition slips.

Our SLO is 99.9% over a rolling 28-day window. The budget fraction is 0.1%. One afternoon a dependency starts failing — a downstream service returns errors on a chunk of our calls — and our observed error rate climbs to **1.44%**. Compute the burn rate:

$$\text{burn rate} = \frac{0.0144}{0.001} = 14.4$$

We are burning at 14.4 times the sustainable pace. Time to exhaust:

$$\frac{28 \text{ days}}{14.4} \approx 1.94 \text{ days}$$

At this rate the entire 28-day budget — all 40.3 minutes of it — is gone in under two days. The fast-burn alert (high burn over a short window) fires within minutes, not hours, because a short-window measurement catches a 14.4× burn almost immediately. The page goes off. The on-call engineer is now staring at a decision: the partial outage is from a dependency, the burn is severe, and at this pace the month's budget is gone by the day after tomorrow.

What do they do? They have a few moves: shed the load that depends on the failing service (serve a degraded but working response — the [graceful-degradation pattern from system design](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation)), trip a circuit breaker so the failing dependency stops dragging latency up, or fail over to a healthy region if the dependency is regional. The *budget* does not tell them which mitigation to choose — that is engineering judgment. What the budget tells them is the **urgency and the stakes**: this is not a "look at it tomorrow" problem, because tomorrow there is no budget left, and once the budget is gone the freeze policy stops all feature work anyway. The burn rate converted a fuzzy "the dependency is flaky" into a hard "we have 36 hours of runway, act now." That conversion — from anxiety to a deadline with a number on it — is the entire value of the gauge.

## 4. The budget is spent by everything: deploys, experiments, migrations, incidents

A budget is only meaningful if you know what draws it down. The error budget is a single pool, and *everything* that risks failure spends from it. This is the part that surprises people: planned work and unplanned disasters come out of the same wallet.

The figure shows the pool and its claimants. Routine deploys spend a little each — every deploy carries some risk of a bad rollout, even a good one occasionally trips an alert during the rollout window. Experiments and canaries spend deliberately: you are intentionally exposing some users to an unproven change. Risky migrations — a schema change, a datastore swap, a major dependency upgrade — are large planned spends. And incidents, the unplanned kind, can drain the pool in an afternoon. The remaining balance is the ship-or-freeze signal.

![A vertical stack diagram showing one error budget pool at the top and the claimants that draw it down below it including routine deploys, experiments and canaries, risky migrations, and unplanned incidents and bad deploys, all feeding into the remaining budget that serves as the ship or freeze signal](/imgs/blogs/the-error-budget-the-currency-of-reliability-3.png)

The reason this single-pool framing is so powerful is that it makes *every* risky decision comparable on one axis. Should we run this experiment this week? It depends on how much budget we have and how much the experiment is likely to spend. Should we do the database migration now or next month? Depends on the budget. Should we ship the risky feature before the holiday freeze? Depends on the budget. Before the error budget, each of these was a separate argument with separate stakeholders and separate intuitions. After it, they are all withdrawals from one account, and you make them the way you make any spending decision: check the balance, weigh the expected cost against the expected value, and spend if you can afford it.

And here is the liberating consequence: **as long as the budget has room, you ship freely.** You do not need a meeting to approve a deploy. You do not need ops to sign off on an experiment. The budget already said yes — there is room, the spend is within means, go. This is the velocity half of the bargain, and it is genuinely freeing for developers. The error budget is not a leash; for most of the month it is a *license*. It is the thing that lets you stop asking permission, because the permission is encoded in a number that updates itself.

The mirror image is the constraint, and it is just as important: **when the budget is gone, the spending stops.** Not because someone decided to be cautious, but because the account is empty and you cannot spend money you do not have. That automatic, impersonal "no" is what makes the "yes" trustworthy. If the budget could be ignored when empty, it could not be trusted when full — the whole thing would collapse back into opinion. The teeth are what make the freedom real.

## 5. The freeze policy: a budget with no teeth is just a chart

Here is the failure mode that kills most error-budget programs, and it is worth stating bluntly: a team computes a beautiful error budget, builds a gorgeous Grafana dashboard, watches the budget burn down to zero — and then keeps shipping features exactly as before, because nothing *happens* when the budget hits zero. The dashboard turns red. People glance at it. The release train rolls on. **A budget with no consequence is not a budget. It is a chart.**

The figure makes the contrast stark. On the left, a budget with no policy: it burns to zero, the dashboard goes red, nobody is empowered to stop the ship, releases continue, and the red chart is wallpaper. On the right, a budget with teeth: it hits zero, a pre-agreed policy triggers automatically, a feature freeze goes into effect, the whole team shifts to reliability work, and the freeze lifts only when the service is back inside its budget.

![A two column before and after diagram contrasting an error budget with no policy where the dashboard turns red but releases continue against an error budget with a freeze policy where hitting zero automatically triggers a feature freeze and shifts the team to reliability work until back in budget](/imgs/blogs/the-error-budget-the-currency-of-reliability-7.png)

The consequence is the **error-budget policy**, and it is a real document, agreed in advance, that says what happens at each budget threshold. The most important rule in it is the **freeze rule**: when the budget for a window is exhausted, feature releases pause and engineering effort shifts to reliability work until the service is back inside its SLO. That is the pre-agreed consequence that makes the budget real. It has to be agreed *in advance*, in calm times, by the people with the authority to enforce it — because in the heat of a budget breach, with a feature deadline looming, nobody will agree to freeze if the freeze is up for negotiation. The policy removes the negotiation. The budget burned; the policy triggers; the freeze is on; it is not a decision anyone makes in the moment, it is a rule everyone agreed to earlier.

Here is a concrete error-budget policy you can adapt. It is a document, not code, but it is structured enough that it reads almost like a config:

```yaml
# error-budget-policy.yaml — agreed by service owners + leadership, reviewed quarterly
service: checkout-api
slo:
  availability: "99.9% of valid requests succeed, rolling 28d"
  latency: "99.9% of valid requests < 300ms, rolling 28d"
budget:
  window: 28d
  # budget = 1 - SLO = 0.1% ; ~40.3 min or ~100k req per 100M

thresholds:
  - name: healthy
    budget_remaining: "> 25%"
    policy: "Ship freely. No release approval needed. Deploy on green CI."
  - name: caution
    budget_remaining: "10% to 25%"
    policy: >
      Ship, but require a second reviewer on changes to the settlement path.
      No large migrations. Pause non-critical experiments.
  - name: freeze
    budget_remaining: "<= 0%"
    policy: >
      FEATURE FREEZE. No feature releases. Only reliability fixes,
      rollbacks, and incident mitigations may ship. All hands shift to
      reliability work (postmortem actions, hardening, toil reduction)
      until budget_remaining returns above 10%.
    exceptions:
      - "Security patches and data-loss fixes are always allowed."
      - >
        A documented business-critical release may be approved ONLY by the
        VP of Engineering in writing, and it spends from NEXT window's budget.

burn_rate_alerts:
  - { severity: page, burn_rate: 14.4, long_window: 1h,  short_window: 5m }
  - { severity: page, burn_rate: 6,    long_window: 6h,  short_window: 30m }
  - { severity: ticket, burn_rate: 3,  long_window: 24h, short_window: 2h }
  - { severity: ticket, burn_rate: 1,  long_window: 72h, short_window: 6h }

ownership:
  budget_owner: "checkout-api team lead"           # owns the number
  policy_enforcer: "engineering director"          # can actually freeze
  review_cadence: quarterly
```

Notice the structure. There are graded thresholds, not just a single cliff at zero — at 25% remaining you start being careful, at 10% you tighten review, at zero you freeze. There is an explicit owner of the number and an explicit enforcer who can actually stop the ship (those are often different people, and naming both prevents the "everyone assumed someone else would freeze it" failure). There are exceptions, because a policy with no escape valve gets thrown out the first time it blocks a genuine emergency — but the exceptions are narrow, named, and require a specific person's written approval, which keeps them rare. And critically, the business-critical exception **spends from next window's budget**, so even an override is accounted for; you cannot get free reliability by declaring everything critical.

Let me return to the worked example from section 2. The bad deploy spent 30% of the latency budget in an afternoon, leaving 70%. Where does that land? Above the 25% caution threshold, so technically still "ship freely" — but a good policy and a good team read the *trajectory*, not just the level. Burning 30% in one afternoon means a few more afternoons like that and you are frozen. The caution-zone behavior — second reviewer on risky paths, pause experiments — is exactly the kind of voluntary slowdown a sane team adopts when they see the budget dropping fast even before the policy forces it. The policy is the floor of caution, not the ceiling.

#### Worked example: walking the timeline of a budget-eating deploy

It helps to see the whole episode on a clock, because the value of the budget is that it makes each beat of an incident visible and dated. The figure below walks the afternoon minute by minute: the deploy goes out clean, errors spike ten minutes later, the burn-rate page fires within four minutes of the spike, the rollback lands, and the policy reads the post-incident balance and decides what happens next.

![A left to right timeline of a bad afternoon deploy showing the deploy going out with zero budget spent, an error spike that pushes the burn rate to forty times, a fast burn page firing minutes later, a rollback that has cost twenty six minutes of budget, the policy reading a forty percent balance and freezing risky releases, and the budget resetting in the next window](/imgs/blogs/the-error-budget-the-currency-of-reliability-5.png)

Trace the beats and notice how each one is a *number on a clock*, not a vibe. At 13:00 the deploy goes out; budget spent so far, zero. At 13:10 a serialization bug starts throwing 5xx and inflating latency; the observed error rate jumps and the burn rate spikes to roughly 40×. At 13:14 the fast-burn alert fires — because a 40× burn over a five-minute window crosses the page threshold almost immediately, the page beats a human noticing the graph by a wide margin. The on-call rolls back, and by 13:40 the bad version is gone; the 26 wasted minutes of elevated errors have eaten a real chunk of the window's budget. At 14:00 the dust settles and someone pulls up the budget-remaining number: it reads 40%. Now the policy does its job *without a meeting* — 40% is above the zero-percent freeze line but the *trajectory* (a quarter of the budget gone in one afternoon) plus the team's judgment puts them squarely in caution mode, so risky releases pause, experiments are shelved, and the next deploy to the settlement path gets a second reviewer. Nobody argued. The clock and the balance made the call. When the window rolls forward, the spend from this afternoon ages off the back of the rolling window and the budget recovers, and shipping resumes at full speed. The timeline is the whole lifecycle of a budget withdrawal: spend, detect, mitigate, read the balance, let the policy respond, recover. Every one of those beats is something you can timestamp and audit afterward — which is exactly what turns "that deploy was bad" into a postmortem with numbers instead of a shrug.

The reason to dwell on the clock is that the *speed* of each beat is itself an SRE metric you can improve. The gap from 13:10 (errors start) to 13:14 (page fires) is your **detection time** — shorten it by tuning burn-rate windows. The gap from 13:14 to 13:40 (rollback done) is your **mitigation time** — shorten it with one-click rollbacks and rehearsed runbooks. Together they are the bulk of your **MTTR**, mean time to recovery, and every minute you shave off them is budget you *do not* spend on the next incident. The budget does not just account for the damage; read against the timeline, it tells you precisely which part of your incident response to invest in, because the longest beat is the one bleeding the most budget. A team that shortens detection from 4 minutes to 1 and rollback from 26 minutes to 8 turns a 26-minute budget hit into an 9-minute one — the same bug, a third of the cost, because the operational reflexes got faster. That is the proof-by-measurement the whole discipline is after.

## 6. The artifacts: PromQL for budget remaining and a multi-window burn-rate alert

Time for the practice, the copy-and-adapt configuration. Everything above is computable from the same counters you already export. Here is how it looks in Prometheus and PromQL.

First, the recording rules. You almost never compute the SLI and the budget inline in a dashboard query — they are expensive and you want them precomputed and consistent. Define the good-events and total-events ratios as recording rules, then derive everything from them:

```yaml
# slo-recording-rules.yaml — Prometheus recording rules for the checkout SLO
groups:
  - name: checkout_slo
    interval: 30s
    rules:
      # Total valid requests (exclude 4xx client errors from the denominator)
      - record: job:checkout_requests:rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout",code!~"4.."}[5m]))

      # Good requests = not a 5xx (availability SLI numerator)
      - record: job:checkout_requests_good:rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout",code!~"5..",code!~"4.."}[5m]))

      # The availability SLI as a ratio, over 5m, 1h, 6h, 28d windows
      - record: job:checkout_sli_availability:ratio5m
        expr: |
          job:checkout_requests_good:rate5m / job:checkout_requests:rate5m

      # Error ratio (1 - SLI) over the longer windows for burn-rate alerts
      - record: job:checkout_error:ratio1h
        expr: |
          1 - (
            sum(rate(http_requests_total{job="checkout",code!~"5..",code!~"4.."}[1h]))
            /
            sum(rate(http_requests_total{job="checkout",code!~"4.."}[1h]))
          )
      - record: job:checkout_error:ratio5m
        expr: |
          1 - (
            sum(rate(http_requests_total{job="checkout",code!~"5..",code!~"4.."}[5m]))
            /
            sum(rate(http_requests_total{job="checkout",code!~"4.."}[5m]))
          )
```

Now the **budget-remaining** query. The error budget consumed over the window is the total errors divided by the allowed errors. The fraction remaining is one minus that. With a 99.9% SLO (budget fraction 0.001) over a 28-day window:

```promql
# Fraction of the 28-day error budget still remaining (1.0 = full, 0 = empty).
# error_budget = 1 - SLO = 0.001 for a 99.9% SLO.
(
  1 - (
        (1 - (
          sum(rate(http_requests_total{job="checkout",code!~"5..",code!~"4.."}[28d]))
          /
          sum(rate(http_requests_total{job="checkout",code!~"4.."}[28d]))
        ))
        / 0.001
      )
)
```

Read it inside-out: the inner ratio is the SLI over 28 days; `1 - SLI` is the observed error fraction; dividing by `0.001` expresses that error fraction as a *multiple of the budget* (i.e. the fraction of budget consumed); and the outer `1 -` flips consumed into remaining. If the service ran perfectly, the error fraction is 0, consumed is 0, and remaining is 1.0 (100%). If it burned exactly the allowed 0.1%, consumed is 1.0 and remaining is 0. If it burned 0.2%, consumed is 2.0 and remaining is `-1.0` — yes, the budget can go negative, and a negative budget is a breached SLO. Surface that as a single big number on your SLO dashboard and you have the live ship-or-freeze gauge.

The **burn rate** itself is even simpler — it is the observed error ratio over the budget fraction:

```promql
# Burn rate over the last 1 hour. 1.0 = sustainable; 14.4 = burns 28d budget in ~2d.
job:checkout_error:ratio1h / 0.001
```

And the **multi-window multi-burn-rate alert** — the one the next post is about, sketched here so you have a working rule. It fires a page only when *both* a long window and a short window agree the burn is high, which kills the false positives from a brief blip while still catching a real fast burn quickly:

```yaml
# slo-burn-alerts.yaml — page on fast burn, ticket on slow burn
groups:
  - name: checkout_slo_alerts
    rules:
      # FAST BURN: 14.4x over 1h AND 5m -> page. Budget gone in ~2 days.
      - alert: CheckoutErrorBudgetFastBurn
        expr: |
          (job:checkout_error:ratio1h / 0.001) > 14.4
          and
          (job:checkout_error:ratio5m / 0.001) > 14.4
        for: 2m
        labels: { severity: page, slo: checkout_availability }
        annotations:
          summary: "Checkout burning error budget at >14.4x (fast)"
          description: "At this rate the 28d budget empties in ~2 days. Mitigate now."

      # SLOW BURN: 3x over 24h AND 2h -> ticket. Budget gone in ~9 days.
      - alert: CheckoutErrorBudgetSlowBurn
        expr: |
          (job:checkout_error:ratio24h / 0.001) > 3
          and
          (job:checkout_error:ratio2h / 0.001) > 3
        for: 15m
        labels: { severity: ticket, slo: checkout_availability }
        annotations:
          summary: "Checkout burning error budget at >3x (slow)"
          description: "Steady elevated errors. Budget empties in ~9 days. Investigate in hours."
```

The fast rule pages immediately on a severe burn but the `and` with the short window means it stops firing within minutes once the burn ends — no lingering 3am page for a blip that already self-healed. The slow rule catches the insidious case the fast rule misses: a low-grade 3× burn that never trips a fast alert but quietly drains the whole budget over a week. Pairing them is the standard recipe; the [alerting post](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) tunes the exact windows and burn-rate pairs.

If you would rather see the budget arithmetic as code you can run locally instead of PromQL, here is a tiny calculator that takes an SLO and a traffic volume and prints the budget in both units, plus the time-to-exhaust for a given burn rate:

```python
def error_budget(slo: float, requests_per_window: int, window_minutes: int):
    """Compute an error budget in requests and minutes from an SLO."""
    budget_fraction = 1.0 - slo
    allowed_bad_requests = requests_per_window * budget_fraction
    allowed_bad_minutes = window_minutes * budget_fraction
    return budget_fraction, allowed_bad_requests, allowed_bad_minutes

def time_to_exhaust(window_minutes: int, burn_rate: float):
    """Minutes until the budget is empty at a given burn rate."""
    if burn_rate <= 0:
        return float("inf")
    return window_minutes / burn_rate

# 99.9% SLO, 100M requests over a 28-day (40,320-minute) window
frac, bad_req, bad_min = error_budget(0.999, 100_000_000, 40_320)
print(f"budget fraction: {frac:.4f}")          # 0.0010
print(f"allowed bad requests: {bad_req:,.0f}")  # 100,000
print(f"allowed bad minutes: {bad_min:.1f}")    # 40.3

for br in (1, 6, 14.4, 100):
    mins = time_to_exhaust(40_320, br)
    print(f"burn {br}x -> empties in {mins/60/24:.2f} days ({mins:.0f} min)")
# burn 1x   -> 28.00 days
# burn 6x   -> 4.67 days
# burn 14.4x-> 1.94 days
# burn 100x -> 0.28 days (6.7 hours)
```

That is the whole machinery: recording rules compute the SLI, a budget-remaining query gives the live gauge, a burn-rate query gives the speedometer, and a multi-window alert turns the speedometer into a page. Five artifacts, all idiomatic, all derived from counters you already have.

## 7. The political win: the budget makes dev and ops one optimization

Now the part that is not arithmetic but is the real reason the error budget exists. The error budget solves a *political* problem, and it is the oldest political problem in operations.

Developers want velocity. Their job, their incentive, their whole reason for existing is to ship features — to change the system, fast, often. Operators (or the on-call, or the SRE) want stability. Their job is to keep the system up, and *every change is a risk to that.* These two goals are in direct, structural tension. The dev's success metric (features shipped) is the ops person's failure mode (changes introduced). Left alone, this tension produces exactly the eleven-person meeting from the top of this post: an endless, unwinnable argument where dev pushes and ops resists and the outcome is decided by politics — who is more senior, who is more persuasive, who is more tired. Worse, it produces *adversaries*. Dev sees ops as the department of no. Ops sees dev as reckless cowboys. Each side optimizes against the other, and the system gets *both* slow *and* unreliable, because the energy goes into the fight instead of the work.

The error budget dissolves this by making dev and ops **one optimization instead of two.** Here is the trick: the budget *is* the dev's resource to spend, and the freeze policy *is* the ops protection. Developers get to spend the budget on whatever they want — features, experiments, risky deploys — and nobody can tell them no while the budget has room. That is a real, enforceable grant of velocity; ops cannot block a deploy out of generic caution anymore, because the number says there is room. In exchange, when the budget runs out, the freeze policy protects stability automatically — and the devs agreed to that policy in advance, so the freeze is not ops imposing its will, it is the devs honoring a deal they made. Both sides got something concrete. Both sides are now optimizing the *same number*: keep the service inside its SLO so the budget stays full so you can keep shipping. Velocity and stability stop being a tug of war and become a single shared interest in not burning the budget carelessly.

This is the line worth remembering: **the error budget turns "ops says no, devs override" into "the number decides, and we both agreed to the number."** No person is the villain. No meeting is required. When a dev wants to ship and the budget is full, they ship — ops has no standing to object, and that is a feature, not a bug. When the budget is empty and a dev wants to ship, they cannot — and ops did not stop them, the policy did, the policy they signed. The locus of authority moved from a person to a number, and a number does not have an ego, does not play favorites, and does not get tired at the end of a long week.

There is a cultural dimension to this that the [planned post on building an SRE culture and team](/blog/software-development/site-reliability-engineering/building-an-sre-culture-and-team) develops fully — the budget only works if leadership genuinely backs the freeze, if the team trusts the number, and if the whole org treats reliability as a shared responsibility rather than "the SRE team's problem." A budget imposed by an SRE team that leadership undercuts the first time a deadline looms is worse than no budget, because now you have the appearance of discipline with none of the substance. The budget is a social technology as much as a technical one. But when the social part holds, the technical part is almost trivially simple, and that is the beauty of it: one subtraction, one threshold, one policy, and a decade-old organizational war just ends.

## 8. The politics of the budget: ownership, what counts, and gaming

The budget aligns dev and ops, but only if the budget itself is honest. And there are a handful of ways to make it dishonest, each of which quietly defeats the purpose. This is the politics of the budget, and a principal engineer earns their keep by spotting these.

**Who owns the budget?** Someone has to. The budget needs a clear owner — usually the service team lead — who is accountable for the number, and a clear *enforcer* — usually someone with org authority, like an engineering director — who can actually halt releases when the policy says freeze. These are often different people, and confusing them is a classic failure: the team lead "owns" the budget but cannot stop the release train, and the director can stop it but does not watch the number. Name both roles explicitly (the policy YAML in section 5 does), or the freeze will not happen when it needs to, because everyone will assume someone else is holding the brake.

**What counts against the budget?** This is the thorniest question, and the figure lays out the decision tree. The governing principle is one sentence: **an event counts against the budget if a healthy version of your service could have served it correctly.** That single rule resolves most of the hard cases.

![A decision tree rooted in whether a healthy service could have served the request correctly, branching into your fault failures like bad deploys and latency regressions that spend the budget, a judgment call branch for dependency and platform outages, and an excluded branch for planned maintenance and client side four hundred errors](/imgs/blogs/the-error-budget-the-currency-of-reliability-8.png)

Walk the branches. A **bad deploy** that throws 5xx errors — clearly your fault, a healthy service would not have done that, full spend. A **latency regression** you introduced — your fault, spend. **Planned maintenance** that you pre-announced and the customer agreed to a window for — generally excluded, because it is not a surprise and not a breach of the agreement (though if you are running a service where users expect 24/7 availability, even planned maintenance should arguably count, which is why "maintenance windows" are increasingly seen as a cop-out for serious services). **4xx client errors** — excluded from the denominator entirely, because the client asked for something impossible and a healthy service correctly refused.

The genuinely hard one is a **dependency's outage.** A cloud provider has a regional failure; an upstream service you call goes down; a third-party API you depend on returns errors. Is that your fault? In one sense no — you did not break it. But here is the principle that matters: **your user does not care whose fault it is.** From the user's seat, your service is down, full stop. They cannot file a ticket with your cloud provider. So the honest default is: **dependency outages count against your budget**, because you own the user experience and you chose to depend on that thing without a fallback. This is uncomfortable, and it is *meant* to be — because counting dependency failures against your budget is exactly the forcing function that pushes you to build the circuit breakers, fallbacks, and multi-region failover that make you resilient to dependency failures. If dependency outages were free, you would never invest in surviving them. By making them cost budget, the accounting aligns with the goal: a service that survives its dependencies' failures.

The table summarizes the verdict for the common cases, with the one-sentence rule applied to each. Keep it where the on-call can see it, because "does this count?" is asked at 3am more often than you would like.

| Failure event | Whose fault | Counts against budget? | Why |
| --- | --- | --- | --- |
| Bad deploy throwing 5xx | Yours | Yes — full spend | A healthy service would not have done this |
| Latency regression you shipped | Yours | Yes — full spend | Slow requests are bad events by your SLI |
| Cloud region or upstream down | Theirs, but your UX | Yes — count it | The user feels it; you chose the dependency |
| Pre-announced maintenance window | Agreed | Usually excluded | Not a surprise; user agreed to the window |
| 4xx client error (bad request) | Client | No — out of denominator | A healthy service correctly refused it |
| Your rate limiter wrongly 429-ing | Yours (a 4xx) | Yes — pull it back in | Your malfunction, dressed as a client error |
| Load test / synthetic probe traffic | Neither (not real users) | No — exclude | Not user pain; would distort the ratio |

The two rows that surprise people are the dependency outage (counts, because you own the user experience) and the misfiring rate limiter (counts, because a 429 that *your* bug produced is your failure wearing a client-error costume). Both follow from the single rule, and both are exactly the places a team tries to wriggle out of the budget — which is why writing them down in advance, before any specific incident makes it personal, is what keeps the accounting honest.

**Gaming the budget.** The most common cheat is to **set the SLO low so the budget is huge.** If your real reliability is 99.5% and you set a 99% SLO, your budget is enormous, you never freeze, and you "always have budget." Congratulations — you have defeated the entire purpose. A budget you never come close to spending exerts no discipline; it is decoration with a percentage sign. The fix is that the SLO must be set empirically and honestly — it should bite occasionally, a bad month should trip it and a good month should clear it (this is the subject of the [planned post on setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something)). If your SLO has not tripped in a year, it is too loose, and your "always-full" budget is a lie you are telling yourself. The other cheat is **relabeling failures out of the denominator** — calling real 5xx failures something else, or expanding the "excluded" categories until nothing counts. Same disease, same cure: the denominator decision must be explicit, documented, and reviewed, because it is exactly where the dishonesty hides.

**Chronic overspend is a signal, not a failure.** Here is the most important political reframing. If you are *constantly* out of budget — frozen month after month, always firefighting — the error budget is not telling you your team is bad. It is telling you the **system needs reliability investment.** Chronic overspend is a forcing function: it is the budget screaming that you are operating beyond your reliability means, and the answer is not to keep firefighting (which never ends) or to game the SLO looser (which hides the problem), but to *invest* — to fund the redundancy, the better deploys, the automated failover, the toil reduction that moves the service to a place where it can stay in budget. The budget converts "we feel overwhelmed" into "we have been out of budget for three consecutive windows, here is the data, we need to fund reliability work." That is a fundable argument. A feeling is not. The error budget is, among everything else, the best tool you will ever have for justifying reliability headcount and project time, because it turns reliability debt into a number on a chart that goes the wrong way.

## 9. Windows, resets, and the rolling-vs-calendar choice

One more piece of arithmetic that trips people up: the **window**. The budget is defined over a window, and the choice of window shape changes the budget's behavior in ways worth understanding.

The two common shapes are **calendar** and **rolling.** A calendar window resets on a fixed boundary — the first of the month, say. A rolling window always looks back a fixed duration from *now* — the last 28 days, continuously sliding.

The calendar window has one fatal flaw: the **reset cliff.** On the last day of a bad month, your budget might be deeply negative — you have blown the SLO badly. Then the calendar flips, the window resets, and *poof*, you have a full budget again, as if nothing happened. This creates a perverse incentive: near month-end, with the budget blown, why be careful? It resets in two days anyway. And the inverse: near month-start, with a fresh full budget, you might take reckless risks because there is "so much budget." The calendar reset makes your caution oscillate with the calendar instead of tracking reality.

The **rolling window** fixes this by never resetting all at once. A rolling 28-day budget at any moment reflects the trailing 28 days of behavior. There is no cliff — a bad day's spend rolls *off* the back of the window 28 days later, gradually, one day at a time, so the budget recovers smoothly rather than snapping back. This means your caution tracks reality continuously: if you had a bad incident two weeks ago, you are *still* carrying that spend, and you should *still* be careful, because the budget reflects it. Most mature SRE practices prefer rolling windows for exactly this reason — they want the budget to be an honest, continuous reflection of recent reliability, not a number that lies to you on the first of every month.

The 28-day choice (rather than 30) is a small, pragmatic nicety: 28 days is exactly 4 weeks, so the window always contains the same number of each weekday. A 30-day window contains a varying number of weekends depending on where it starts, and since most services have a strong weekday/weekend traffic pattern, a 28-day window gives you a more stable, less seasonally-distorted budget. It is a minor point, but it is why you see "rolling 28 days" so often where you might expect "30."

There is a real cost to rolling windows: they are more expensive to compute (a 28-day `rate()` in PromQL is heavy) and harder to explain to non-engineers than "this month." A reasonable compromise some teams use is to *report* a calendar-month budget to stakeholders (easy to understand) while *alerting and gating* on a rolling window (honest and cliff-free). Just be clear which is which, because the same incident spends differently against the two, and you do not want a stakeholder waving the rosy calendar number while the rolling number says freeze.

## 10. War story: how Google made the error budget the heart of SRE

The error budget is not a clever idea someone blogged about — it is the organizing principle of Google's Site Reliability Engineering, documented in the SRE Book, and the story of *why* they adopted it is the clearest case for the whole approach.

Google's SRE teams faced the same dev-versus-ops war everyone faces, but at enormous scale. Product development teams wanted to ship features fast. SRE teams, who carried the pagers, wanted stability and resisted change. The two organizations had different bosses, different incentives, and a structural conflict that, left to fester, would have produced either glacial product velocity (if SRE won every argument) or constant outages (if product won every argument). The genius move, attributed to the early SRE leadership, was to stop arguing about *whether* a change was safe and start *budgeting* for unreliability. They set an SLO — explicitly *below* 100%, because 100% is the wrong target — and declared the gap to be a budget the product team owned and could spend. As long as the SLO was met, the product team could ship as fast as they wanted; SRE would not block them. When the SLO was breached and the budget exhausted, a freeze kicked in and the product team's own engineers had to stop shipping features and help fix reliability until they were back in budget.

The brilliance is what this did to *incentives.* Suddenly the product team had skin in the reliability game — not because SRE nagged them, but because their *own* velocity depended on staying in budget. A reckless deploy that burned the budget did not hurt some other team; it froze *their own* feature pipeline. So the product team started caring about reliability on their own, adopting better testing, safer rollouts, and faster rollbacks — not because they were told to, but because the budget made reliability *their* interest. The SRE team, meanwhile, stopped being the department of no. They did not have to block changes anymore; the budget did that, automatically and impersonally, only when warranted. The relationship shifted from adversarial to collaborative, because both teams were now optimizing the same number.

There is a second, subtler lesson Google documented that is worth carrying: the error budget also gives SRE a principled way to *push back when reliability is too high.* If a service is consistently *beating* its SLO by a wide margin — burning almost no budget — that is not unambiguously good. It might mean the SLO is too loose. But it can also mean the team is being *too* conservative, shipping too slowly, leaving velocity on the table. An unspent budget is an opportunity cost: it is reliability the users could not perceive, bought at the price of features they wanted. So a healthy error-budget culture sometimes says "you are too reliable — go faster, take more risk, spend the budget you are hoarding." That is a sentence no traditional ops team has ever uttered, and it is only sayable because the budget makes "too reliable" a measurable, defensible claim instead of heresy.

A cautionary counter-story rounds this out. Plenty of teams have adopted the error-budget *vocabulary* without the *teeth* — they compute budgets, build dashboards, talk about burn rate, and then never actually freeze when the budget blows, because leadership always finds a reason the current deadline is the exception. Those teams get the worst of both worlds: the overhead of the machinery with none of the discipline. The lesson, repeated across every postmortem-of-a-failed-SRE-program I have seen, is that **the policy is the product.** The math is easy. The dashboard is easy. The hard, essential part is the organizational commitment to actually stop shipping when the number says stop — and a budget without that commitment is, as we said, just a chart.

What does success look like, measured? The table contrasts a team before and after the budget had teeth — illustrative figures from the kind of turnaround the model produces, not a specific company's audited numbers. The point is the *direction* and the *mechanism*, both of which are real and repeatable.

| Signal | Before the budget | After the budget with teeth | What changed |
| --- | --- | --- | --- |
| Ship-or-freeze decision | An hour-long meeting per release | A glance at one number, no meeting | Authority moved from a person to a number |
| Reliability incidents / quarter | ~12, recurring root causes | ~5, root causes funded and fixed | Chronic overspend funded real fixes |
| Dev / ops relationship | Adversarial, "department of no" | One shared metric, same goal | Velocity and stability became one optimization |
| Reckless Friday deploys | Common, decided by deadline pressure | Gated by budget, not by the calendar | The number, not the clock, sets caution |
| Reliability project funding | Argued from feelings, often denied | Argued from budget data, fundable | Overspend became a chart, not a complaint |
| Release velocity in a healthy month | Slow, every change re-litigated | Fast, ship freely while budget remains | Permission encoded in the budget, not asked |

Read the table as the whole thesis in one frame. None of the wins came from a new tool — Prometheus was already there, the deploys were already there. The wins came from the *budget reframing the decision* so that velocity and stability stopped fighting and the number, not a person, held the brake. That is the entire return on the one subtraction.

## 11. Reading the budget in real time: the ship-or-freeze decision

Let me pull the threads together into the decision the budget exists to make, and show how it routes in practice. The figure traces it: you measure the SLI, compute the budget remaining and the current burn rate, and the *number* forks the release into one of three states.

![A branching graph that measures the SLI then computes budget left and burn rate which forks into a healthy state that ships freely, a fast burn state that pages and freezes, and an exhausted state that freezes, with both freeze paths and the ship path flowing into a new window where the budget resets](/imgs/blogs/the-error-budget-the-currency-of-reliability-6.png)

The three states the budget routes you into:

1. **Budget remaining, burn under 1×.** You are healthy. Ship freely, no meeting, no approval. This is most of the time for a well-run service, and it is the point — the budget's default state is "yes."
2. **Fast burn, budget draining.** A burn-rate alert pages someone. You may still have budget left, but at this rate you will not for long. Mitigate immediately (rollback, circuit-break, shed load, fail over), and tighten release caution until the burn stops. The page bought you the runway to act before the budget is gone.
3. **Budget exhausted.** The freeze policy triggers. No feature releases; all hands to reliability work until you are back in budget. The freeze is not punishment — it is the system honoring the deal that bought you all that freedom earlier in the window.

All three paths eventually flow into a new window, where the budget resets (smoothly, if rolling) and the cycle continues. This is the *spend* step of the series loop made operational: define the SLO, measure the SLI, compute the budget, and let the number govern whether you ship or freeze.

#### Worked example: the overlapping-incident stress test

Let me stress-test the policy, because a policy that only works on the happy path is not a policy. Suppose two things go wrong at once. You are already at **15% budget remaining** from a rough first half of the window (caution zone — second reviewer required, experiments paused). Then a dependency outage starts burning at **14×.** At 14× from 15% remaining, the time to exhaust the *remaining* budget is roughly:

$$\frac{0.15 \times 28 \text{ days}}{14} \approx 0.30 \text{ days} \approx 7.2 \text{ hours}$$

You have about seven hours before the budget hits zero and the freeze triggers. Now a *second*, unrelated incident pages — a memory leak in a different service is crashing pods. Two incidents, overlapping, one budget already low. What does the framework tell you?

It tells you to *triage by budget impact.* The dependency outage is burning the budget at 14×; the memory leak, suppose, is causing slow restarts but few user-facing errors, burning at maybe 2×. The budget-weighted priority is unambiguous: the dependency outage is the fire, the leak is smoldering. You put your strongest responder on the 14× burn, mitigate it (trip the breaker, serve degraded), and accept that the leak gets the second-string attention until the big burn is controlled. Without the budget, two simultaneous incidents are a chaotic scramble where the loudest alert or the most senior complainer wins. With the budget, you have an objective triage key: **whichever burns the most budget gets the most people**, because that is the one closest to breaching the promise. And if you *do* breach — if the budget hits zero while you are still fighting — the freeze policy is already written, so you do not have to convene the eleven-person meeting mid-incident to decide whether to stop shipping. It is decided. One less decision to make while the pagers are screaming.

The deeper stress-test answers from the kit, briefly: *What if the on-call is asleep?* The multi-window fast-burn page is loud and escalates; that is the [on-call and alerting machinery](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) the budget feeds. *What if the budget is already spent before the incident?* Then you are already frozen, all hands are already on reliability, and the incident response is the same work you were already doing — the freeze pre-positioned the team. *What if a region fails?* If the dependency is regional and you have multi-region failover, you fail over and spend little budget; if you do not, the regional outage counts against your budget (you own the UX) and becomes the data that funds building the failover. In every branch, the budget is the thing that converts a vague "this is bad" into a quantified "here is how bad, here is how fast, here is what the policy says next."

## 12. How to reach for the error budget (and when not to)

The error budget is the most important idea in SRE, but it is not free and it is not universal. Here is the decisive guidance.

**Use an error budget on any service with a real SLO and real release tension.** If you have a user-facing service, an SLO that bites occasionally, and a recurring argument about whether to ship or stabilize, the error budget is exactly your tool. It will end the argument, align dev and ops, and give you a fundable signal for reliability investment. This is the common case, and the budget pays for itself the first time it dissolves a ship-or-freeze fight without a meeting.

**Do not bolt a budget onto a service with no honest SLO.** The budget is `1 − SLO`, so a fake SLO produces a fake budget. If you have not done the work to set an SLO that reflects user pain and bites occasionally, do that first; a budget on a meaningless SLO is theater. Set the SLO honestly (the [SLO post](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) is the prerequisite), *then* derive the budget.

**Do not build a budget you will not enforce.** This is the big one. If your leadership will not actually back a freeze — if every budget breach will produce a "but this deadline is special" override — then do not pretend. An unenforced budget is worse than none: it has all the overhead and none of the discipline, and it teaches the team to ignore the dashboard. Either secure the organizational commitment to the freeze policy *before* you launch the budget, or do not launch it. The policy is the product; without teeth, skip it.

**Do not over-engineer the budget for low-stakes services.** A nightly internal batch job does not need a rolling-28-day burn-rate budget with multi-window alerts. It needs to finish correctly by morning — a *different* SLI (freshness or completeness) with a much simpler check. Reserve the full burn-rate machinery for services where minutes of downtime matter and release velocity is contested. Bringing 14.4× fast-burn alerting to a cron job is cargo-culting.

**Do not let the budget become a reason to ship recklessly when full.** "We have budget, so anything goes" is a misreading. A full budget licenses *reasonable* risk, not negligence — you still test, still roll out safely, still watch the canary. The budget says you do not need *permission*; it does not say you do not need *care*. The teams that get burned are the ones who treat a full budget as an excuse to skip the safe-deploy hygiene that kept the budget full in the first place.

## 13. Key takeaways

- **The error budget is `1 − SLO`, and that one subtraction reframes failure from a disgrace into a resource.** A 99.9% SLO means 0.1% of requests are *allowed* to fail — and that allowance is yours to spend on deploys, experiments, migrations, and incidents.
- **Read the budget in two units.** Allowed bad requests (budget fraction times traffic — 0.1% of 100M is 100,000 failures) and allowed downtime (budget fraction times window minutes — 0.1% of a month is 43.2 minutes). Quote minutes to humans, count requests in queries.
- **Burn rate = observed error rate / (1 − SLO), and time-to-exhaust = window / burn rate.** A burn of 1× lasts the whole window; 14.4× empties a 28-day budget in about 2 days; 100× in under 7 hours. Burn rate is the live speedometer; the monthly report is the rear-view mirror.
- **Everything spends from one pool.** Routine deploys, canaries, risky migrations, and unplanned incidents all draw from the same budget — which makes every risky decision comparable on one axis: can we afford it?
- **A budget with no consequence is just a chart.** The freeze policy is the teeth: when the budget hits zero, feature releases pause and all hands shift to reliability work until you are back in budget. Agree the policy in advance, with a named owner and a named enforcer, or the freeze will not happen.
- **The budget makes dev and ops one optimization.** Devs get to spend the budget on velocity; ops are protected by the freeze; both now optimize the same number. "Ops says no, devs override" becomes "the number decides, and we both agreed to the number."
- **Mind the politics: ownership, what counts, and gaming.** An event counts if a healthy service could have served it — so dependency outages count (you own the UX), and that is what forces you to build resilience. Setting the SLO low to inflate the budget defeats the purpose; the SLO must bite occasionally.
- **Chronic overspend is a signal to invest, not to keep firefighting.** Being out of budget for multiple windows is the budget telling you the system needs reliability funding — and it converts that need into a defensible number instead of a feeling.
- **Prefer rolling windows.** A rolling 28-day budget has no reset cliff, so caution tracks reality continuously instead of oscillating with the calendar. Report calendar to stakeholders if you must, but alert and gate on rolling.
- **Don't fake it.** No honest SLO means no honest budget; no enforced freeze means no real discipline. The math is easy; the organizational commitment is the whole game.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro this post completes, framing reliability as something you engineer and budget rather than wish for.
- [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter) — the post that defines the SLO this budget is computed from, and the nines-to-downtime table behind the budget units.
- [Choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) — how to pick the good-over-total indicator whose complement becomes the budget, including the denominator decisions that govern what counts.
- [Setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) — the prerequisite for an honest budget: choosing the target empirically so it bites occasionally instead of being gamed loose.
- [Alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) — the forward reference for turning burn rate into multi-window pages that catch fast and slow burns without false alarms.
- [Building an SRE culture and team](/blog/software-development/site-reliability-engineering/building-an-sre-culture-and-team) — the organizational backing the freeze policy needs to have teeth, and how leadership makes the budget real.
- [Reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — the architecture-time companion on designing systems that degrade gracefully to stay inside their budgets.
- *Site Reliability Engineering* (the Google "SRE Book"), the chapters on Embracing Risk and Service Level Objectives — the canonical source for the error-budget model and the argument that 100% is the wrong target.
- *The Site Reliability Workbook*, the chapter on Alerting on SLOs — the derivation of the multi-window multi-burn-rate alerts and the 14.4× fast-burn threshold sketched here.

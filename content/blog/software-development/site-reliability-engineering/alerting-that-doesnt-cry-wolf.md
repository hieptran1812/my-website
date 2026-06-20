---
title: "Alerting That Doesn't Cry Wolf: Symptom-Based Pages and Burn-Rate Alerts"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Stop the 3am false page that erodes trust until the team mutes the pager: alert on symptoms not causes, page only on things a human must fix now, and tie every page to an SLO burn-rate alert you can paste into Prometheus today."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "alerting",
    "burn-rate",
    "prometheus",
    "alertmanager",
    "slo",
    "on-call",
    "observability",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/alerting-that-doesnt-cry-wolf-1.png"
---

The pager went off at 3:14 in the morning for a disk that was 81% full. I remember the number because I stared at it for a long time, sitting on the edge of the bed with the laptop balanced on one knee, trying to decide whether 81% was an emergency. It was not. The disk had been climbing one percent a week for two months; at that rate it would hit 90% in nine days and would not actually fill for another month. There was nothing for me to do at 3:14 except acknowledge the page, go back to sleep, and file a ticket in the morning. So I did the thing every tired on-call eventually does: I silenced that alert. Not for the night. Forever. And three weeks later, when a different disk on a different host filled up *for real* during a traffic spike and the service started dropping writes, the alert that should have warned me was in a graveyard of silenced rules I had stopped trusting. I found out from a customer.

That is the whole tragedy of bad alerting in one story. It is not that any single false page is catastrophic. It is that false pages are *corrosive*. Each one is a small withdrawal from a finite account of trust between the on-call engineer and the pager. Spend that account down — page someone forty times a night for things they cannot act on — and the rational human response is to stop believing the pager. They mute it, they auto-acknowledge it, they train themselves to roll over and ignore the buzz. And a pager nobody believes is worse than no pager at all, because it gives you the *illusion* of coverage while quietly guaranteeing that the one page that matters arrives to a person who has learned, through bitter repetition, that the pager lies.

![A two column before and after diagram contrasting cause-based alerting that fires on a hundred possible causes and floods the pager until the team mutes it against symptom-based alerting that fires only when users are hurting and earns the team's trust](/imgs/blogs/alerting-that-doesnt-cry-wolf-1.png)

This post is the sixth in the series, and it sits at the operational heart of it. The [intro argued that reliability is a feature you engineer rather than a virtue you hope for](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset). The post on [the error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) gave us the number that decides whether we are reliable enough. The post on [metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right) gave us the raw signal. Now we connect them to a human being's sleep. By the end of this post you will be able to: tell a symptom from a cause and route each to the right place; design a multi-window, multi-burn-rate SLO alert that catches both a sudden outage and a slow leak without flapping; do the burn-rate arithmetic that turns a percentage into a page; write the actual Prometheus alerting rules and Alertmanager routes; measure your alert quality honestly with a small set of numbers; and prune a screaming pager down to a quiet, trustworthy one. We are squarely on the series spine — **define reliability → measure it → spend the error budget → respond → learn → engineer the fix** — and alerting is the wire that connects *measure* to *respond*. Get it wrong and the whole loop breaks at its most human joint.

## 1. The one principle: alert on symptoms, page only for a human, tie it to the SLO

Almost every alerting disaster traces back to a single confusion, so let us name it precisely. There are two ways to decide when to wake someone up, and they are night and day.

**Cause-based alerting** watches the *machinery* and fires when a part of the machinery looks wrong. CPU above 90%. Memory above 85%. Disk above 80%. A pod restarted. Queue depth over a thousand. Replication lag over five seconds. Each of these is a *cause* — a thing that *might* lead to user-visible badness. The seductive logic is: "if I alert on every possible cause, I'll catch every problem before it reaches the user." The fatal flaw is in the word *every*. There are not three or four causes. There are hundreds. Every resource on every host, every dependency, every queue, every connection pool, every cache, every cert, every cron — each is a potential cause, and each gets its own threshold, and each threshold fires on its own schedule for its own reasons, most of which never touch a user. You end up paging on the union of a hundred independent noise sources. That is the alert storm.

**Symptom-based alerting** watches the *outcome the user experiences* and fires when that outcome is bad. Are requests failing? Are they slow? Is the page not loading? There are not a hundred of these. For most services there is essentially *one* symptom that matters — "are we serving users well?" — expressed through the golden signals of latency, errors, and (for capacity) saturation. The reason this is so much quieter is arithmetic: **there are roughly a hundred causes for every one symptom.** A symptom alert collapses all hundred causes into a single question. If CPU is high but users are fine, you do not care tonight. If a pod restarted but the load balancer routed around it and users never noticed, you do not care tonight. The symptom alert only fires when one or more of those causes *actually combined to hurt a user* — which is the only condition under which waking a human is justified.

So the principle, stated in full and worth tattooing on the inside of every on-call's eyelids:

> **Alert on symptoms, not causes. Page a human only for things that need a human *now*. And tie every page to the SLO, because the SLO is the precise, agreed definition of "the user is hurting enough to matter."**

Causes are not useless — they are *diagnosis*. When the symptom alert fires and you are staring at the dashboard at 3am, the cause metrics (CPU, disk, queue depth, replication lag) are exactly what you read to figure out *why* the symptom appeared. But you read them; they do not page you. Causes belong on dashboards and in tickets. Symptoms — and specifically, SLO-burning symptoms — belong on the pager. This is not a stylistic preference. It is the difference between an on-call rotation that is sustainable for years and one that burns out a team in a quarter.

The rest of this post is the engineering of that principle: how to express "the user is hurting" as a precise, computable condition (the SLO burn-rate alert), how to make it sensitive to both fast and slow failures without flapping, how to route the *causes* to the right non-paging destinations, and how to prove, with numbers, that the pager got quieter and *better* at the same time.

## 2. Why cause-based alerting floods: 100 causes, 1 symptom

Let me make the "hundred causes, one symptom" claim concrete, because once you see the math you can never un-see it.

Take a perfectly ordinary web service: a few stateless API pods behind a load balancer, a Postgres primary with a replica, a Redis cache, a background queue worker, and a couple of upstream dependencies. Now count the cause-based alerts a well-meaning team would add over two years, one incident at a time, each one added because "we got bitten by this once and never want to be surprised again":

- Per-pod CPU > 80%, memory > 85%, OOMKill events — call it three alerts per pod, six pods, eighteen alerts.
- Disk usage > 80% on every host — six hosts, six alerts.
- Pod restart count > 0 in five minutes — six alerts.
- Postgres: connection count near max, replication lag > 5s, long-running query > 60s, deadlock count > 0, autovacuum behind — five alerts.
- Redis: memory near maxmemory, eviction rate > 0, connection refused — three alerts.
- Queue: depth > 1000, consumer lag growing, dead-letter count > 0 — three alerts.
- Network: per-host error rate, retransmit rate, conntrack table near full — three alerts.
- Certs: TLS expiry < 30 days on each endpoint — a few.
- Dependency health-check failures, DNS resolution failures, NTP drift, and the long tail of "we added this after the leap-second thing" — a dozen more.

You are already past forty cause-based alerts, and a real fleet has hundreds. Now here is the part that ruins on-call: **each of these has a base rate of firing that has nothing to do with whether a user is hurting.** CPU spikes to 90% during a normal batch job and recovers in two minutes — page. A pod gets rescheduled by the cluster autoscaler — page. A query runs long during a nightly report — page. Disk crosses 80% on a host that has a month of runway — page. Redis evicts a few keys because that is *literally what an LRU cache is supposed to do* — page. None of these hurt a user. All of them page.

The frequencies are independent and they add up. If forty alerts each fire a false positive just once a week on average — an optimistic assumption — that is forty pages a week, almost six a day, most in the middle of the night because that is when batch jobs and backups run. The on-call engineer who is paged six times a day for things they cannot act on does not become more vigilant. They become numb. The numbness is not a character flaw; it is the *correct Bayesian update*. After the hundredth false page, the prior that "this page is real" is so low that ignoring it is the rational move. The pager has trained the human to ignore it. We did that. We engineered learned helplessness into our own incident response.

The symptom alert short-circuits the entire mess. Instead of forty independent noise sources, you ask one question — *are users being served well, as measured against the SLO?* — and that single question is true only when one or more of those forty causes has *actually* degraded the user experience past the agreed line. The forty causes still exist; you still chart them; you still read them during an incident. But they no longer each get a vote on your sleep. Only the user's pain gets a vote.

#### Worked example: the page-rate arithmetic

A team I worked with had 30 cause-based alerts on a single service. We pulled six weeks of pager history. The numbers: roughly **200 pages per week**, of which **8% were actionable** — meaning in 92% of pages, the on-call looked, found nothing a human needed to do, and acknowledged. Of the actionable 8%, more than half *auto-resolved before the engineer even finished logging in*: a transient spike that had already passed. So the genuinely "a human had to do something, now" rate was about **4%** — eight real pages out of two hundred. We were waking people up 192 times a week to deliver eight signals. The signal-to-noise ratio was 1-to-24. No human filter survives that ratio intact. The pager *had* to be muted; muting it was the only way to sleep, and so the pager was, functionally, off.

### The true cost of a false page

It is tempting to treat a false page as a minor annoyance — "you just acknowledge it and go back to sleep." That accounting is wrong by an order of magnitude, because it counts only the seconds of the acknowledgment and ignores everything around it.

**The sleep cost is not the page; it is the recovery.** A page at 3am does not cost you the ninety seconds it takes to read it and acknowledge. It costs you the *forty minutes* it takes to fall back asleep with adrenaline in your blood, and the degraded cognition of the next day, and — if it happens often enough — the chronic sleep debt that is genuinely a health issue. Interrupted sleep is not a series of independent ninety-second costs; it is cumulative physiological damage. A person paged four times a night is not "inconvenienced four times." They are sleep-deprived, and a sleep-deprived on-call makes worse decisions during the *real* incident when it finally comes.

**The trust cost compounds.** Every false page lowers the on-call's prior that the next page is real. This is not weakness of character; it is correct inference. After the hundredth false page, the rational estimate of "this page is real" is near zero, and the rational action — given that responding has a cost and the expected value of responding is near zero — is to *not respond promptly*. The system has trained the human into slow, reluctant acknowledgment. And the day the real page comes, it arrives to a human who has been conditioned, by hundreds of repetitions, to assume it is noise. The muted pager in my opening story was not a careless mistake. It was the *endpoint of a rational process* that a noisy alerting system set in motion.

**The burnout cost ends careers and rotations.** A rotation that pages people 200 times a week does not just tire people out; it makes the on-call role something engineers actively avoid, negotiate out of, and quit over. The best engineers — the ones with options — leave the noisiest rotations first. So a noisy pager does not merely cost sleep; it costs you the senior people who could have made the rotation quiet, in a vicious cycle where the team that most needs to fix its alerting is the team least able to retain the people who could. Fixing the pager is a *retention* intervention as much as a reliability one.

Add it up and a false page is not a ninety-second cost. It is sleep, plus trust, plus, over time, the team's health and tenure. That is the budget you are spending every time you let an unactionable alert reach the pager — and it is why the pruning discipline in §9 is not housekeeping. It is the most consequential reliability work a team does, because it protects the human system that responds to everything else.

## 3. The destination question: page, ticket, or dashboard

Before we build the symptom alert, we need somewhere for all those causes to go — because the answer to "stop paging on disk usage" is not "ignore disk usage." It is "send disk usage to the right destination." Every signal has exactly one right home, and a single question decides it.

The question is: **"If this fires at 3am, can a human do something about it *right now* that they could not do equally well at 9am?"** Run every candidate alert through that filter and three buckets fall out, shown in the figure below.

![A decision tree rooted at the question can a human act on this signal at three in the morning, branching into page a human act now, file a ticket look tomorrow, and dashboard or log for diagnosis, with leaf examples under each branch](/imgs/blogs/alerting-that-doesnt-cry-wolf-2.png)

**Page (wake someone).** The signal is urgent, actionable, and about user impact. Users are being hurt right now and a human must intervene to stop it. SLO fast-burn. The service is returning errors to a meaningful fraction of requests. Latency has blown past the threshold and is staying there. A revenue-critical path is down. These are the *only* things that justify interrupting sleep. If the honest answer to "can I do anything useful at 3am?" is yes, it pages.

**Ticket (look tomorrow).** The signal is real and actionable but *not urgent* — there is runway. Disk at 80% with a month before it fills. A TLS cert expiring in three weeks. A slow memory leak that will need a restart sometime this week. A non-critical batch job that failed and can rerun in the morning. These belong in your ticketing system (Jira, Linear, GitHub Issues) with an owner and a due date, not on the pager. The defining test: *acting on it at 9am tomorrow is exactly as good as acting on it at 3am tonight, so why wake anyone?*

**Dashboard or log (FYI / diagnosis).** The signal is context, not a call to action. Per-host CPU. Cache hit rate. Request rate by endpoint. Pod restart counts. These are the metrics you *read* when a real page fires and you are diagnosing. They should be on a Grafana dashboard, queryable, and — critically — *not wired to any notification at all*. A signal that fires no notification is not "ignored"; it is *available*. The mistake is thinking that everything worth measuring is worth alerting on. Almost nothing worth measuring is worth *paging* on.

Here is the same logic as a routing table you can copy:

| Signal | Urgent? | Actionable now? | User impact? | Destination |
| --- | --- | --- | --- | --- |
| SLO fast-burn (errors/latency) | Yes | Yes | Yes | **Page** |
| Revenue path returning 5xx | Yes | Yes | Yes | **Page** |
| SLO slow-burn (chronic erosion) | Somewhat | Yes | Yes (gradual) | **Ticket** (or low-urgency page) |
| Disk 80%, month of runway | No | Yes, but later | Not yet | **Ticket** |
| TLS cert expires in 30 days | No | Yes, but later | Not yet | **Ticket** |
| CPU 90% on one node, latency fine | No | No (autoscaler handles) | No | **Dashboard** |
| One pod restarted, self-healed | No | No | No | **Dashboard** |
| Cache eviction rate | No | No | No | **Dashboard** |

The phrase to keep in your pocket is *"the alert that should have been a ticket."* When you do an alert review (we will get to that in §9), the single most common finding is a paging alert that fails the urgency test — it is real, it is actionable, but it is not *now*. Demote it to a ticket and the pager gets quieter without losing a single bit of coverage, because the work still gets tracked; it just stops costing someone their sleep.

There is a second, subtler finding the alert review surfaces, and it is worth naming because it is so common: the *duplicate-coverage* alert. A team adds a symptom alert ("checkout SLO burning"), but never removes the dozen cause alerts that used to be the only coverage. Now every real outage pages *twice* — once for the symptom, once for whichever cause happened to trip — and the on-call learns that an outage means a flurry of pages rather than one clean signal. The cause pages are not adding information; the symptom alert already told you the user is hurting. They are adding *noise on top of a real event*, which is the worst kind, because it trains the on-call to expect a storm even when the alerting is working. When the symptom alert covers a class of failures, the cause pages for that class should be deleted, not kept as a belt-and-suspenders. Keep the cause *metrics* charted for diagnosis; retire the cause *pages*.

A useful way to think about the three destinations is by their *time horizon to action*. A page is "act within minutes." A ticket is "act within days." A dashboard is "act if and when a page sends you here to diagnose." The horizon is the whole decision. If you find yourself arguing about whether something should page, the argument almost always resolves to "how soon must a human act?" — and that question has an answer rooted in the SLO and the budget. If acting an hour from now would not measurably change the user outcome, it is not a page. Make the horizon explicit and the routing stops being a matter of taste.

## 4. The centerpiece: SLO burn-rate alerts

Now the heart of the post. We have decided to page on symptoms tied to the SLO. The question is *how* — what is the precise, computable condition that means "the user is hurting enough, fast enough, to wake a human"? The answer is the **burn-rate alert**, and it is the single most important alerting pattern in modern SRE.

Recall the error budget from the [error-budget post](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). If your SLO is 99.9% over 28 days, your error budget is `1 − 0.999 = 0.1%` of requests. **Burn rate** is how fast you are consuming that budget relative to the pace that would exactly exhaust it over the window. It is defined simply:

$$\text{burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}$$

A burn rate of 1 means you are spending the budget at exactly the rate that uses it all up over the full 28-day window — sustainable, on-pace, no alarm. A burn rate of 2 means you are spending twice as fast; the budget will be gone in 14 days. A burn rate of 14.4 means the budget is draining 14.4× too fast and the whole month's budget will be gone in roughly two days. The burn rate is, in one number, *how worried to be*, and it has a beautiful property: it is **scale-free**. A 14.4× burn is equally alarming whether your SLO is 99.9% or 99.99%, whether you serve a thousand requests a second or ten. The threshold travels.

Why alert on burn rate instead of a raw error-rate threshold? Because a raw threshold has no idea what your reliability target is. "Page if error rate > 1%" is meaningless without context: 1% errors is a catastrophe for a 99.99% service (a 100× burn) and barely a blip for a 99% service (a 1× burn). Burn rate *normalizes by the budget*, so the same alert rule is correct across services with wildly different targets. You write the alert once, in burn-rate terms, and it means the same thing everywhere.

There is a second reason, and it is about *humanity* as much as math. A raw error-rate alert answers the wrong question. It tells you "the error rate is 1.2% right now," which is a fact about the present instant. The burn rate answers the question the on-call actually has at 3am: *"how long do I have before this becomes a broken promise?"* A burn rate of 14.4 says "the whole month's budget is gone in two days if this continues" — that is a *deadline*, and a deadline is exactly the information a triaging human needs to decide whether to roll back now or finish their coffee first. The translation from burn rate to time-to-exhaust is direct and worth internalizing: at a burn rate of $b$ against a window of length $T$, the budget empties in $T / b$. A 14.4× burn on a 30-day window empties it in $30 / 14.4 \approx 2.1$ days; a 300× burn empties it in $30 / 300 = 2.4$ hours; a 1× burn lasts the full 30 days by definition. The burn rate is a clock counting down to the moment your SLO breaks, and that clock is the single most decision-relevant number you can put in front of a sleepy human.

A third advantage is *portability of the on-call's intuition*. Because the burn rate is scale-free, an engineer who learns "14× is wake-me-up, 3× is a ticket, 1× is fine" carries that calibration across every service they ever support, regardless of its traffic or its SLO. Compare that to raw thresholds, where the on-call has to memorize a different "danger" number for each service — 0.5% here, 2% there, 50 errors a minute somewhere else — and inevitably mis-calibrates under stress. Burn rate gives the whole org *one vocabulary* for "how bad is it," and a shared vocabulary is what lets a secondary on-call from another team jump onto an incident bridge and immediately understand the severity. That organizational fluency is an underrated benefit of standardizing on burn-rate alerting.

But a single burn-rate threshold over a single window has a problem, and solving that problem is the whole art.

## 5. Why one window isn't enough: the multi-window design

Picture a single alert: "page if the 1-hour burn rate exceeds 14." This catches a sudden, severe outage beautifully — errors spike, the 1-hour burn rockets past 14, you get paged in minutes. But it has two failure modes that will bite you.

**Failure mode one: it flaps.** A short blip — a 90-second spike during a deploy, a transient network hiccup — can push the 1-hour burn over 14 for a moment and fire a page, then clear. You get woken for something that resolved itself before you opened the laptop. Worse, if the condition hovers near the threshold, the alert fires, clears, fires, clears — a flapping alert, the most maddening kind, because each flap is a fresh page.

**Failure mode two: it misses slow burns entirely.** Now picture a *chronic* leak: a deploy introduces a bug that makes 0.4% of requests fail, quietly, for days. Is 0.4% errors enough to push the 1-hour burn over 14? For a 99.9% SLO, the 14.4× threshold corresponds to an error rate of `14.4 × 0.1% = 1.44%`. A steady 0.4% is well under that, so the 1-hour alert never fires. The 6-hour alert never fires. *No static threshold trips at all* — and yet 0.4% errors for a week will quietly drain your entire month's budget and blow the SLO. The acute-outage alert is blind to the slow bleed.

The fix, codified in the Google SRE Workbook's chapter on alerting on SLOs, is the **multi-window, multi-burn-rate** alert. You run *several* burn-rate alerts at once, each tuned to catch a different speed of failure:

![A graph where one measured budget burn signal feeds a fast one hour window and a slow six hour window in parallel, with an acute outage tripping the fast page, a chronic leak tripping the slow page, and a five minute blip clearing both windows so no page fires](/imgs/blogs/alerting-that-doesnt-cry-wolf-4.png)

- A **fast-burn page**: high burn rate (e.g. 14.4×) over a **short window** (1 hour), with a **short secondary window** (5 minutes) that must *also* be burning. This catches acute outages in minutes. The 5-minute secondary window is the anti-flap trick: the alert only fires if *both* the 1-hour and the 5-minute windows are over threshold, so a 90-second blip that has already cleared in the last 5 minutes will not fire, and the alert resets quickly when the incident ends.
- A **second fast-burn page**: a lower burn rate (6×) over a **longer window** (6 hours) with a 30-minute secondary. This catches outages that are bad but not catastrophic — a steady 0.6% error rate that the 14.4× alert misses but that will still drain the budget in a few days.
- A **slow-burn ticket** (not necessarily a page): a low burn rate (3×) over a **day-long window** (1 day) with a 2-hour secondary. This catches the chronic leak — the 0.4% bleed that no acute alert sees but that will erode the budget over the month.

The figure below lays out the four canonical windows from the Workbook with their thresholds and actions. Read it as the menu: you almost never need all four, but the pattern of "fast page + slow page + slow ticket" is the workhorse.

![A comparison matrix with rows for the four canonical burn rate windows of fourteen point four times over one hour, six times over six hours, three times over one day, and one times over three days, and columns showing budget burned, time to detect, and the action of page or ticket](/imgs/blogs/alerting-that-doesnt-cry-wolf-3.png)

Why does the secondary short window solve flapping so cleanly? The intuition is worth making precise. A burn-rate alert over a single 1-hour window is sticky in a bad way: once an outage pushes the 1-hour average over threshold, that elevated average *persists* for up to an hour after the outage ends, because the window is still averaging in the bad period. So a 10-minute outage that has fully recovered keeps the 1-hour alert firing for nearly an hour — long after there is anything to do — and during that hour, small fluctuations push it across the threshold and back, flapping you with re-pages. Adding the requirement that the *5-minute* window must *also* be hot fixes both ends. At the *start*, the 5-minute window only crosses threshold if the burn is happening *right now*, so a 90-second blip that already cleared never satisfies both conditions and never fires. At the *end*, the moment the outage stops, the 5-minute window drops below threshold within five minutes, so the alert *resolves* fast even while the 1-hour window is still elevated. The short window is the "is it live?" check; the long window is the "is it significant?" check. Requiring both gives you sensitivity to real, sustained burns and immunity to transients and stale averages at the same time.

The threshold numbers are not arbitrary; they come from a clean piece of arithmetic. The Workbook chooses thresholds by asking: *"how much of the total error budget should be consumed before we alert?"* The standard choices are 2% of the budget for the fastest alert, 5% for the medium, and 10% for the slow ones. From "consume X% of the budget in window W," the burn rate falls out:

$$\text{burn rate} = \frac{(\text{budget fraction to consume}) \times (\text{window as fraction of SLO period})^{-1}}{1}$$

Concretely, for a 30-day SLO period: to consume **2% of the budget in a 1-hour window**, you need a burn rate of `0.02 / (1h / 720h) = 0.02 × 720 = 14.4`. To consume **5% in 6 hours**: `0.05 / (6 / 720) = 6`. To consume **10% in 3 days (72 hours)**: `0.10 / (72 / 720) = 1`. That is where 14.4×, 6×, 1× come from — they are the burn rates that consume a chosen, principled fraction of the budget within each window. The 3× over a day is a common middle-ground addition. The thresholds are not magic numbers; they are "alert me before I have spent 2% / 5% / 10% of my month."

#### Worked example: catching the fast burn

Your SLO is 99.9% over 30 days. Budget = 0.1% of requests. At 3pm a deploy introduces a bug; error rate jumps to **30%**. Burn rate = `30% / 0.1% = 300`. The 1-hour fast alert (threshold 14.4) and its 5-minute secondary both blow past 300 within minutes. You are paged at roughly 3:05pm. At a 300× burn, your *entire 30-day budget* would be gone in `30 days / 300 = 2.4 hours`. The 5-minute secondary window confirms it is still happening right now (not a blip that already cleared), so the page is real and urgent. You roll back at 3:20pm. Total budget spent: 15 minutes of a 30% error rate ≈ a few percent of the month. The fast alert did its one job: it caught an acute, severe failure in minutes and woke the right person while there was still budget to protect.

## 6. The actual Prometheus rules

Enough theory. Here is the multi-window, multi-burn-rate alert as real Prometheus rules you can adapt. The pattern has two layers: **recording rules** that pre-compute the error ratio over each window (so the alerting expressions stay readable and cheap), and **alerting rules** that compare those ratios to the burn-rate thresholds. Define the SLO target once and derive everything from it.

First, the recording rules — compute the request-error ratio over each window. (For the deeper treatment of why you pre-aggregate with recording rules, see [metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right).)

```yaml
groups:
  - name: slo:checkout:recording
    interval: 30s
    rules:
      # error ratio = failed requests / total requests, per window.
      # "failed" = 5xx responses; tune the label match to your SLI.
      - record: job:slo_errors:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="checkout"}[5m]))
      - record: job:slo_errors:ratio_rate1h
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[1h]))
          /
          sum(rate(http_requests_total{job="checkout"}[1h]))
      - record: job:slo_errors:ratio_rate30m
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[30m]))
          /
          sum(rate(http_requests_total{job="checkout"}[30m]))
      - record: job:slo_errors:ratio_rate6h
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[6h]))
          /
          sum(rate(http_requests_total{job="checkout"}[6h]))
      - record: job:slo_errors:ratio_rate2h
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[2h]))
          /
          sum(rate(http_requests_total{job="checkout"}[2h]))
      - record: job:slo_errors:ratio_rate1d
        expr: |
          sum(rate(http_requests_total{job="checkout", code=~"5.."}[1d]))
          /
          sum(rate(http_requests_total{job="checkout"}[1d]))
```

Now the alerting rules. The SLO is 99.9%, so the budget fraction is 0.001. The burn-rate threshold of 14.4 means "error ratio above 14.4 × 0.001 = 0.0144." The key move is the **two-window AND**: each alert requires both its long window *and* a short secondary window to be over threshold, which kills flapping and gives fast recovery.

```yaml
groups:
  - name: slo:checkout:alerts
    rules:
      # --- FAST BURN: page now. 14.4x over 1h AND 5m. Catches acute outages. ---
      - alert: CheckoutErrorBudgetFastBurn
        expr: |
          job:slo_errors:ratio_rate1h > (14.4 * 0.001)
          and
          job:slo_errors:ratio_rate5m > (14.4 * 0.001)
        for: 2m
        labels:
          severity: page
          slo: checkout-availability
        annotations:
          summary: "Checkout burning error budget 14.4x (fast). Page now."
          description: >
            Over the last 1h and 5m the checkout 5xx ratio exceeds 1.44%
            (14.4x the 0.1% budget). At this pace the 30-day budget empties
            in ~2 days. This is an acute, user-facing outage.
          runbook: "https://runbooks.example.com/checkout-error-budget-burn"
          dashboard: "https://grafana.example.com/d/checkout-slo"

      # --- MEDIUM BURN: page now. 6x over 6h AND 30m. Catches bad-but-slower. ---
      - alert: CheckoutErrorBudgetMediumBurn
        expr: |
          job:slo_errors:ratio_rate6h > (6 * 0.001)
          and
          job:slo_errors:ratio_rate30m > (6 * 0.001)
        for: 5m
        labels:
          severity: page
          slo: checkout-availability
        annotations:
          summary: "Checkout burning error budget 6x (medium). Page now."
          description: >
            Over the last 6h and 30m the checkout 5xx ratio exceeds 0.6%
            (6x the budget). At this pace the 30-day budget empties in ~5 days.
          runbook: "https://runbooks.example.com/checkout-error-budget-burn"
          dashboard: "https://grafana.example.com/d/checkout-slo"

      # --- SLOW BURN: ticket, not a page. 3x over 1d AND 2h. Catches chronic leak. ---
      - alert: CheckoutErrorBudgetSlowBurn
        expr: |
          job:slo_errors:ratio_rate1d > (3 * 0.001)
          and
          job:slo_errors:ratio_rate2h > (3 * 0.001)
        for: 15m
        labels:
          severity: ticket
          slo: checkout-availability
        annotations:
          summary: "Checkout chronic error budget burn 3x (slow)."
          description: >
            Over the last 1d and 2h the checkout 5xx ratio exceeds 0.3%
            (3x the budget). No single threshold spike, but this slow leak
            will exhaust the 30-day budget. Investigate next business day.
          runbook: "https://runbooks.example.com/checkout-error-budget-burn"
          dashboard: "https://grafana.example.com/d/checkout-slo"
```

A few things worth pausing on, because they are where people get it subtly wrong:

- **The secondary short window is the anti-flap guard, not redundant.** The 1h window is the "is this significant over a meaningful period?" check; the 5m window is the "is it *still happening right now*?" check. A spike that ended ten minutes ago will have a clearing 5m window even while the 1h window is still elevated — so the alert resets fast and you do not get paged for an outage that is already over.
- **`for:` is deliberately short here** (2–5 minutes). The two-window construction already provides the noise immunity that a long `for:` is usually reaching for; piling a long `for:` on top just delays a real page. Keep it short.
- **The slow-burn alert has `severity: ticket`, not `page`.** A 3× chronic burn is real and must be fixed, but it does not need to wake anyone — there are days of runway. Routing it to a ticket is the whole point of the destination question from §3.
- **Every alert carries a `runbook` and `dashboard` annotation.** A page with no runbook link is a page that starts the on-call at zero, fumbling for context at 3am. We will treat the runbook link as a *hard requirement* of every paging alert in §8.

For latency, the same pattern applies, but the SLI is a ratio of *requests served under the latency threshold* to total requests, computed from a histogram:

```promql
# Fraction of checkout requests served slower than 300ms over 1h.
# (1 - this) is the "good" ratio; burn rate works the same way.
1 - (
  sum(rate(http_request_duration_seconds_bucket{job="checkout", le="0.3"}[1h]))
  /
  sum(rate(http_request_duration_seconds_count{job="checkout"}[1h]))
)
```

Wrap that in the same two-window burn-rate structure and you have a latency SLO alert that pages only when *enough requests are slow enough, for long enough* to burn the budget — not on a single p99 spike.

## 7. The slow burn that cause-based alerting missed

Let me tell you about the second worked example, because it is the one that converts skeptics. It is the failure mode that *only* the multi-window design catches.

A payments team had a thorough set of cause-based alerts: per-pod resource alerts, DB lag alerts, queue depth alerts, dependency health checks — about forty of them. On a Tuesday afternoon they shipped a deploy that introduced a subtle bug: under a specific, common input, a retry path double-counted and a downstream validation rejected the duplicate, returning a 5xx to roughly **0.4% of checkout requests**. Not a spike. A steady, quiet 0.4%.

Watch what every cause-based alert did: *nothing*. CPU was normal — the bug was a logic error, not a load problem. Memory was normal. Disk was normal. DB lag was normal. Queue depth was normal. Every dependency health check was green, because the dependencies were healthy; the bug was in *our* code. Not one of the forty thresholds tripped, because 0.4% of requests failing does not move any *infrastructure* metric. The machinery was fine. The users were not.

![A timeline of a chronic slow burn starting with a Tuesday deploy that introduces a steady leak, showing static alerts staying quiet with no threshold tripped, the budget draining to fifty five percent by Wednesday, the six hour burn window finally firing on Thursday, a ticket created with thirty percent budget remaining, and the fix shipping Friday](/imgs/blogs/alerting-that-doesnt-cry-wolf-7.png)

For a 99.9% SLO, the budget is 0.1%, so a steady 0.4% error rate is a **4× burn**. That is below the 14.4× fast-page threshold and below the 6× medium threshold — so the fast and medium *pages* correctly stay silent (this is not an acute outage; nobody needs to be woken). But it is *above* the 3× slow-burn threshold. The 1-day window, with its 2-hour secondary, crosses 3× and fires the **slow-burn ticket** — sometime Thursday, with about 30% of the month's budget still remaining. The on-call picks up the ticket Friday morning, traces the 0.4% to the bad deploy, ships a fix, and the budget recovers before it is exhausted.

Compare the two worlds. In the cause-based world, this bug is *invisible* until it has burned the entire month's budget and the SLO is blown — at which point you find out from the monthly SLO report, or from a customer, weeks of degraded service later. In the symptom-based, multi-window world, the slow-burn alert catches it on Thursday with budget to spare, *and* it caught it without paging anyone in the middle of the night, because the system correctly judged it as a ticket, not an emergency. That is the multi-window design earning its complexity: **it catches what no single threshold can, and it routes by urgency so the slow problem gets a ticket and only the fast problem gets a page.**

This is the deep reason symptom-based, SLO-tied alerting is not just *quieter* than cause-based alerting — it is *more complete*. A symptom alert sees every cause that hurts a user, including causes you never thought to write an alert for. The 0.4% bug was a cause nobody had an alert for, because nobody anticipated it. The symptom alert did not need to anticipate it. It only needed to notice that users were being served worse than the SLO promised, which is the only thing that ever actually matters.

## 8. Good alert hygiene: the 3am test and the runbook rule

Every paging alert you keep should survive a short, brutal audit. The audit is four questions, and a page that cannot answer *yes* to all four should be demoted to a ticket, sent to a dashboard, or deleted.

![A comparison matrix auditing four candidate alerts against the questions urgent now, actionable, and about user impact, where the SLO fast burn passes all three while CPU ninety percent, disk eighty percent full, and a TLS cert expiring in thirty days each fail at least one question](/imgs/blogs/alerting-that-doesnt-cry-wolf-5.png)

1. **Is it urgent — must it be handled *now*?** If acting at 9am is as good as acting at 3am, it is a ticket, not a page. (Disk at 80% with a month of runway: not urgent. Fails. Ticket.)
2. **Is it actionable — can the on-call actually *do* something about it?** A page for a problem with no human remedy ("the upstream provider is down and we have no fallback") is just an anxiety delivery service. If there is genuinely nothing to do, it should not page; it should drive a *project* to build a fallback. (One pod restarted and self-healed: nothing to do. Fails. Dashboard.)
3. **Is it about user impact?** The alert should fire because users are being hurt, not because a machine looks unusual. If users are fine, it can wait. (CPU 90% with latency fine: no user impact. Fails. Dashboard.)
4. **Does it have a runbook?** Every page must link to a runbook — a short document that says, concretely, *what this alert means, how to confirm it is real, the first three things to check, and how to mitigate.* A page without a runbook starts the on-call at zero. This one is not negotiable: **no runbook, no page.** If you cannot write the runbook, you do not understand the alert well enough to wake someone for it.

The synthesizing question, the one I make every team apply when they propose a new paging alert, is the **3am test**: *"If this fires at 3am, can I — a tired human who did not write this service — actually do something useful about it right now, guided by the runbook?"* If the honest answer is no, it is not a page. It is something else. The 3am test is ruthless and it is correct, because the cost of being wrong about it is a person's sleep and, eventually, the team's trust in the entire alerting system.

A minimal runbook for our burn-rate alert looks like this — short, scannable, and linked from the alert annotation:

```yaml
# runbook: checkout-error-budget-burn
alert: CheckoutErrorBudgetFastBurn / MediumBurn / SlowBurn
meaning: >
  The checkout service is returning 5xx (or serving slow) fast enough to
  burn the error budget. Fast/Medium = page (acute). Slow = ticket (chronic).
confirm:
  - Open the checkout SLO dashboard (linked in alert). Is the burn real and ongoing?
  - Check the 5xx rate by endpoint and by version (canary vs stable).
first_checks:
  - "Recent deploy? Compare burn start time to deploy log. Suspect the latest release."
  - "One bad version? If 5xx is concentrated on a canary/new version, this is a rollout."
  - "Upstream dependency? Check dependency SLO panels and status pages."
mitigate:
  - "If a recent deploy correlates: roll back. This is the fastest, safest mitigation."
  - "If a single bad pod/node: cordon and drain it; let the scheduler reschedule."
  - "If an upstream is down: enable the cached/degraded fallback (flag: checkout.degraded=true)."
escalate:
  - "If burn continues 15min after mitigation, page the secondary and open an incident bridge."
```

Notice what the runbook is *not*: it is not a novel, it is not exhaustive, and it does not try to diagnose every possible cause. It gives the on-call the three or four highest-probability moves — *check the recent deploy, suspect the new version, consider the upstream, roll back* — because at 3am you want the 80% case handled fast, not a treatise. A good runbook is one screen long.

## 9. Measuring alert quality: you can't improve what you don't count

"Our alerting is too noisy" is a feeling. To fix it you need numbers, and the good news is that alert quality is eminently measurable. Track these and review them every week or two:

| Metric | What it measures | Healthy target | Smell if... |
| --- | --- | --- | --- |
| **Pages per on-call per week** | Raw interruption load | < 2 (ideally near 0 on a quiet service) | > 5 sustained = burnout risk |
| **% actionable** | Fraction of pages where a human had to do something | > 90% | < 50% = noisy alerts |
| **% auto-resolved** | Pages that cleared before the human acted | < 20% | high = flapping / too-tight thresholds |
| **Pages off-hours** | Night/weekend interruptions | as low as possible | high = move non-urgent to tickets |
| **Time-to-ack** | How fast pages get acknowledged | < 5 min | rising = team is muting / ignoring |
| **Repeat pages (same alert)** | The same alert firing repeatedly | low | high = a missing fix or a bad threshold |

The two most diagnostic numbers are **% actionable** and **% auto-resolved**. A low percent-actionable means your alerts are firing on things that do not need a human — classic cause-based noise; prune them. A high percent-auto-resolved means your alerts are firing on transients that clear themselves — your thresholds are too tight or your windows too short; widen them or add the secondary-window anti-flap guard. A *rising* time-to-ack is the canary in the coal mine: it means the team is starting to ignore the pager, and you are weeks away from a missed real incident. Watch it like a hawk.

The review itself is a recurring ritual — call it the **alert review** or **pager review** — and it is one of the highest-leverage hours an SRE team spends. Pull the pages from the last two weeks. For each one ask: *was it actionable? was it urgent? did the runbook help?* Every page that fails the audit gets a verdict: **delete it** (it should never have paged), **demote it** (ticket or dashboard), **tune it** (wrong threshold or window), or **fix the underlying issue** (the alert is correct but the thing it fires on keeps happening — go fix the thing so the alert stops firing). Prune ruthlessly. The instinct to keep an alert "just in case" is exactly the instinct that built the 200-page-a-week pager in the first place. An alert you do not trust enough to act on is worse than no alert, because it costs trust and delivers nothing.

#### Worked example: the alert-quality scorecard

The figure below is the headline result, and it is the one to show a skeptical manager: thirty cause-based alerts that paged 200 times a week became two SLO burn-rate alerts that page four times a week, and the percent-actionable went from 8% to 95% — a quieter pager that is also a *better* one.

![A two column before and after diagram showing thirty cause based alerts that page two hundred times a week with eight percent actionable and a muted pager replaced by two SLO burn rate alerts that page four times a week with ninety five percent actionable and a trusted pager](/imgs/blogs/alerting-that-doesnt-cry-wolf-8.png)

Here is the before→after from the team in §2, measured over six weeks before and six weeks after we replaced 30 cause-based alerts with the multi-window SLO alerts:

| Metric | Before (cause-based) | After (symptom/SLO) |
| --- | --- | --- |
| Pages / on-call / week | ~200 | ~4 |
| % actionable | 8% | 95% |
| % auto-resolved before action | ~55% | ~5% |
| Off-hours pages / week | ~80 | ~1 |
| Real incidents caught | (all of them, buried in noise) | all of them |
| Real incidents *missed* | 1 (the muted disk alert) | 0 |

Read the last two rows carefully, because they are the point that defeats the objection "but won't you miss things if you alert less?" We went from 200 pages a week to 4 — a 98% reduction — and we caught *every real incident we caught before, plus one we used to miss*. The reduction did not come from alerting on fewer real problems. It came from alerting on the same real problems through one symptom instead of a hundred causes. Fewer pages, *better* coverage. That is not a trade-off; it is a strict improvement, and it is available to almost every team carrying a noisy pager today.

## 10. Alertmanager mechanics: routing, grouping, inhibition, silences

Symptom-based alerting cuts the *number* of distinct alert rules. Alertmanager handles the *delivery* — and it has four mechanisms that turn even a multi-alert situation into a sane, single signal. This matters most during a real incident, when one root cause can light up several alerts at once.

![A graph where fifty alerts fire from a single upstream database outage, flow into Alertmanager routing by team and severity, get grouped by service, then inhibition uses the database down alert to mute forty nine downstream symptoms while a maintenance silence window drops alerts entirely, leaving one page to the database on-call and tickets for the rest](/imgs/blogs/alerting-that-doesnt-cry-wolf-6.png)

**Routing.** A tree that sends each alert to the right place based on its labels. The `severity: page` alerts go to PagerDuty/Opsgenie and wake someone; `severity: ticket` alerts go to a chat channel or your ticketing webhook; everything else can go to a low-priority channel or nowhere. Route by team so the checkout alerts reach the checkout on-call, not a central pool.

```yaml
route:
  receiver: default-tickets
  group_by: ["slo", "service"]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    # Paging alerts wake a human via PagerDuty, routed by owning team.
    - matchers: ['severity="page"', 'team="checkout"']
      receiver: pagerduty-checkout
      group_wait: 10s          # page fast; don't sit on real outages
      repeat_interval: 1h
    - matchers: ['severity="page"', 'team="payments"']
      receiver: pagerduty-payments
      group_wait: 10s
    # Ticket-severity alerts go to chat + ticket webhook, never the pager.
    - matchers: ['severity="ticket"']
      receiver: slack-and-jira
      group_wait: 5m
      repeat_interval: 24h

receivers:
  - name: default-tickets
    slack_configs:
      - channel: "#alerts-firehose"
  - name: pagerduty-checkout
    pagerduty_configs:
      - routing_key: "<checkout-pd-key>"
  - name: pagerduty-payments
    pagerduty_configs:
      - routing_key: "<payments-pd-key>"
  - name: slack-and-jira
    slack_configs:
      - channel: "#checkout-tickets"
```

**Grouping.** Instead of one notification per firing alert, Alertmanager batches alerts that share `group_by` labels into a single notification. Group by `slo` and `service` so that if the fast-burn and medium-burn both fire for checkout, you get *one* page that says "checkout SLO burning" — not two pages racing each other. `group_wait` (e.g. 10s for pages) lets a burst of related alerts coalesce before the first notification; `group_interval` controls how often a changed group re-notifies.

**Inhibition.** The single most powerful anti-storm mechanism. Inhibition lets a *higher-level* alert suppress *lower-level* alerts while it is firing. The classic case: the database goes down, and that single root cause makes the checkout service, the cart service, the search service, and forty other things all start failing their health checks. Without inhibition you get 50 pages. With an inhibition rule that says "when `DatabaseDown` is firing, suppress all alerts whose `cluster` label matches," the database team gets *one* page for the actual root cause, and the 49 downstream symptom alerts are muted because they are *consequences*, not independent problems.

```yaml
inhibit_rules:
  # When the database is down, suppress the downstream service alerts it causes.
  - source_matchers: ['alertname="DatabaseDown"']
    target_matchers: ['severity="page"', 'tier="application"']
    equal: ["cluster"]   # only inhibit alerts in the same cluster
  # A higher-severity alert for a service suppresses lower-severity ones for it.
  - source_matchers: ['severity="page"']
    target_matchers: ['severity="ticket"']
    equal: ["slo", "service"]
```

Inhibition is what makes a cascading failure deliver one page for the cause instead of fifty for the effects. (For the architecture of *why* one dependency failure cascades — and how circuit breakers and bulkheads contain it — see the system-design treatment of [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads). Here we are containing the *alerting* blast radius, not the failure itself.)

**Silences.** A time-boxed mute for planned work. Before a maintenance window — a deliberate database failover, a noisy migration, a load test — you create a silence matching the affected alerts so the expected, intentional badness does not page anyone. The discipline that makes silences safe: **always scope them tightly** (match specific alert labels, not everything) and **always set an expiry** (a silence with no end date is a permanently muted alert, and we already know how that story ends). A silence is a scalpel for known, planned work; it is not a way to make an annoying alert go away — that is what the alert review is for.

```bash
# Silence checkout SLO alerts for a 30-minute planned DB failover.
amtool silence add \
  'slo="checkout-availability"' \
  --duration="30m" \
  --comment="Planned DB failover, ticket OPS-4821" \
  --author="hiep"
```

There is a failure mode worth flagging with inhibition: it is only as good as your ability to detect the *cause*. Inhibition says "when `DatabaseDown` fires, suppress the downstream symptom pages" — but that only works if you actually have a reliable `DatabaseDown` alert *and* it fires *before or alongside* the downstream symptoms. If the database failure is subtle enough that your cause alert misses it, inhibition suppresses nothing and you get the storm anyway. So inhibition is a *complement* to good symptom alerting, not a replacement: you still want the downstream SLO burn alerts (they catch the user impact regardless of root cause), and inhibition just deduplicates the *delivery* when the root cause is independently detectable. Treat inhibition as a delivery optimization for the cases you understand, never as your primary detection. And test it: a stale or mis-labeled inhibition rule that silently suppresses a *real, independent* alert is a quiet way to miss an incident, so include inhibition rules in the same alert review that prunes everything else.

A practical note on `repeat_interval` and `group_interval`, because they are where teams accidentally re-create noise after carefully cutting it. `repeat_interval` controls how often Alertmanager *re-notifies* about an alert that is still firing — set it too short (say, 5 minutes) and a single ongoing outage re-pages the on-call every five minutes while they are *already working it*, which is its own flavor of cruel. For paging routes, an hour is a sane default: long enough not to nag someone mid-incident, short enough that a forgotten page resurfaces. `group_interval` controls how soon a group re-notifies when its *membership changes* (a new alert joins the group); keep it at a few minutes so a worsening incident — new symptoms appearing — surfaces promptly without spamming. The defaults matter as much as the alert rules; a perfect set of burn-rate alerts wired to a 5-minute `repeat_interval` will still feel like a noisy pager during a long incident.

Together these four mechanisms mean that even when reality lights up your monitoring like a Christmas tree, the *human* sees a small number of well-grouped, root-cause-routed, non-duplicative signals. The alert rules decide *what* is worth a human; Alertmanager decides *how* to deliver it without burying the human in duplicates.

## 11. War story: the retry storm that paged a hundred times, and the Google model that quieted it

Two stories, one cautionary and one constructive.

**The retry storm.** A team ran a service that called a downstream pricing API. The pricing API had a bad afternoon and started timing out on about 20% of calls. The calling service retried — sensibly — but it retried *immediately, three times, with no backoff and no jitter.* So every timeout became four calls in quick succession, and as the pricing API got slower, more calls timed out, and each timeout spawned more retries, and the retry traffic *itself* became the load that kept the pricing API on its knees. A classic thundering herd. (The mechanics of why retries without backoff and jitter amplify load — and how to fix them — are covered in the [microservices treatment of retries and backpressure across services](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads); the short version is that the retry-amplification factor multiplies your offered load by the retry count exactly when the system can least afford it.)

Now the *alerting* failure. This team had cause-based alerts on everything, so the storm lit up the board: the pricing API's latency alert, its error-rate alert, the calling service's timeout alert, its thread-pool-exhaustion alert, its connection-pool alert, three downstream services' health checks, and the per-pod CPU alerts on every box that was now spinning on retries. Within ninety seconds, **over a hundred alerts fired** and the on-call's phone became a continuous buzz. The engineer could not tell the root cause (the pricing API timing out) from the hundred consequences (everything downstream failing). They spent the first twenty minutes of the incident just *reading pages*, trying to find the signal, while the actual fix — shed the retry load, add backoff and jitter — sat undiscovered under the noise. The alert storm did not just fail to help; it *actively extended the outage* by burying the diagnosis.

The fix had two halves. The engineering half: add exponential backoff with jitter and a circuit breaker to the retry path, so a struggling dependency does not get hammered by its own callers. The *alerting* half: one symptom-based SLO burn alert on the calling service ("are we serving users?"), plus an Alertmanager inhibition rule so that when the pricing-API-down alert fires, the hundred downstream symptom alerts are suppressed. After the change, the same dependency failure produced *two* signals: "pricing API is down" (the root cause, paged to the pricing team) and "checkout SLO is burning" (the user impact, paged to checkout) — both with runbooks, both actionable, zero noise. The next time the pricing API had a bad afternoon, the incident was diagnosed in three minutes instead of twenty.

**The Google model.** The constructive story is the one the whole industry borrowed from. Google's SRE practice, documented in the SRE Book and the SRE Workbook, made two moves that this entire post is downstream of. First, they tied alerting to the SLO and the error budget rather than to causes — page when the error budget is burning fast enough to matter, full stop. Second, they formalized the multi-window, multi-burn-rate alert precisely to solve the "fast outage vs slow leak vs flapping" trilemma we worked through in §5, with the 2%/5%/10%-of-budget thresholds that give the 14.4×/6×/1× burn rates. The result, reported across many Google SRE teams, is a pager that fires *rarely* and *actionably* — a handful of pages a week on a healthy service, each one a real, user-affecting problem with a runbook. The model is not Google-specific. It is arithmetic plus discipline, and it transfers to a five-person startup as cleanly as it transfers to a planet-scale service. The thresholds travel because the burn rate is scale-free.

The lesson that ties both stories together: **the failure and the alerting failure are different problems, and you must fix both.** Backoff-and-jitter fixes the retry storm; symptom alerting plus inhibition fixes the *pager* storm. A team that fixes only the first still drowns its on-call in noise during the next incident. A team that fixes only the second has a quiet pager pointed at a system that still cascades. You want both: resilient systems *and* quiet, trustworthy alerting that tells you, clearly and once, when the resilience has run out.

## 12. How to reach for this (and when not to)

Symptom-based, SLO-tied, multi-window alerting is the right default for any service that has real users and a real SLO. But every practice has a cost and a domain of usefulness, so here is the honest guidance.

**Reach for the full multi-window burn-rate alert when** you have a user-facing service with an SLO that someone cares about, enough traffic that error *ratios* are statistically meaningful, and an on-call rotation whose sleep you are responsible for. This is the workhorse and it pays for its complexity many times over. Start with two alerts — fast-burn page and slow-burn ticket — and add the medium window only if you find outages slipping between them.

**Do not over-engineer alerting for a low-traffic service.** Burn-rate alerts assume the error *ratio* is meaningful, which it is not when you serve 50 requests an hour — one failed request is a 2% error rate and a wild burn-rate spike that means nothing. For low-traffic services, alert on *absolute* counts ("more than N errors in M minutes") or on the symptom directly ("the synthetic probe failed three times in a row"), and accept that you have less statistical power. Do not paste a 14.4× burn-rate rule onto a service where the denominator is too small to trust.

**Do not keep cause-based alerts "for safety" once the symptom alert covers them.** The instinct to leave the old CPU/disk/memory pages firing "just in case the symptom alert misses something" is exactly how you rebuild the 200-page pager. If the symptom alert is good, it catches everything that hurts a user, including causes you did not anticipate. Keep the cause *metrics* on the dashboard for diagnosis; delete the cause *pages*.

**Do not page on a problem with no human remedy.** If an alert fires and the runbook would have to say "there is nothing you can do, the upstream provider is down and we have no fallback," that is not an alert — it is a *project*. Drive the work to build the fallback (a cache, a degraded mode, a second provider), and until then, do not wake someone to deliver helpless anxiety at 3am.

**Do not let the slow-burn alert become a page.** The chronic-leak alert is genuinely useful, but it does not need anyone's sleep — there are days of runway. Keep it `severity: ticket`. The moment you promote it to a page "so it doesn't get ignored," you have reintroduced a non-urgent interruption, and the fix for "tickets get ignored" is a working ticket-triage process, not a 3am page.

**Do not silence an alert to make it stop.** A silence is for *planned* work with an *expiry*. An alert that is firing for no good reason should go through the alert review and be fixed, demoted, or deleted — not silenced into the same graveyard that swallowed the disk alert in the opening story. Every silence with no end date is a future missed incident.

The meta-principle: **the goal of alerting is not to detect every anomaly; it is to deliver, reliably, the small number of signals that require a human, in a form that human can act on, without spending so much of their trust that they stop believing the pager.** Optimize for *that* — for a pager the on-call believes — and most of the specific decisions fall out naturally.

## 13. Key takeaways

- **Alert on symptoms, not causes.** There are ~100 causes for every 1 symptom; the symptom alert collapses all of them into the one question that matters — *are users being served well?* Causes are for dashboards and diagnosis, not pages.
- **Page only what needs a human now.** Run every candidate through the 3am test: urgent, actionable, about user impact, with a runbook. If any answer is no, it is a ticket, a dashboard, or a deletion — not a page.
- **Tie pages to the SLO via burn-rate.** Burn rate = observed error rate / (1 − SLO) is scale-free, so the same alert rule is correct across services with different targets. Alert when the budget is burning fast enough to matter.
- **Use multi-window, multi-burn-rate.** A fast window (14.4× over 1h+5m) catches acute outages, a slow window (3× over 1d+2h) catches chronic leaks no single threshold sees, and the short secondary window kills flapping. The 2%/5%/10%-of-budget thresholds give the 14.4×/6×/1× rates.
- **Every page carries a runbook link.** No runbook, no page. If you cannot write the four-line runbook, you do not understand the alert well enough to wake someone for it.
- **Measure alert quality and review it.** Track pages/week, % actionable, % auto-resolved, and time-to-ack. A rising time-to-ack means the team is starting to ignore the pager — the early warning of a missed real incident.
- **Prune ruthlessly.** Every alert you keep "just in case" is a withdrawal from the on-call's trust. An alert nobody acts on is worse than no alert. Delete, demote, tune, or fix the underlying issue — but do not hoard.
- **Let Alertmanager contain the storm.** Group related alerts into one notification, inhibit downstream symptoms when an upstream cause is firing, and silence (tightly scoped, always with an expiry) during planned work.
- **Fixing the failure and fixing the alert storm are two different jobs.** Backoff-and-jitter quiets the retry storm; symptom alerting plus inhibition quiets the pager storm. Do both.

This is the *measure → respond* joint of the series spine, and it is where most teams' reliability practice silently breaks. An error budget you never alert on is a dashboard nobody watches; an alert that cries wolf is a pager nobody answers. The fix is the same idea pointed at a sleeping human: define what "the user is hurting" means precisely (the SLO), measure how fast it is getting worse (the burn rate), and wake someone *only* when the rate is fast enough to need them *now*. Do that and the pager becomes what it was always supposed to be — a thing the on-call trusts, because every time it fires, it is right.

## Further reading

- **The SRE Workbook, "Alerting on SLOs"** — the canonical source for the multi-window, multi-burn-rate pattern, the 2%/5%/10%-of-budget thresholds, and the exact burn-rate-to-window arithmetic this post operationalizes.
- **The Google SRE Book, "Monitoring Distributed Systems"** — the original articulation of symptom-based alerting, the four golden signals (latency, traffic, errors, saturation), and "page only on things that are both urgent and actionable."
- **Prometheus documentation — recording and alerting rules, and PromQL** — the reference for `rate()`, `histogram_quantile()`, two-window expressions, and how `for:` and the rule evaluation interval interact.
- **Alertmanager documentation — routing, grouping, inhibition, and silences** — the configuration reference behind §10, including `group_by`, `inhibit_rules`, and `amtool silence`.
- **[Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset)** — the series intro that frames reliability as something you engineer and budget, and the define → measure → budget → respond → learn → engineer loop this post sits inside.
- **[The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability)** — the budget and burn-rate arithmetic that every alert in this post is built on; read it first if `burn rate = error rate / (1 − SLO)` is new to you.
- **[Metrics and time series done right](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right)** — how to instrument and pre-aggregate the counters and histograms that feed these alerts, and why recording rules keep the alerting expressions cheap and readable.
- **Designing a humane on-call** (planned, sibling slug `designing-a-humane-on-call`) — the rotation, escalation, and sustainability side of the pager: once your alerts are trustworthy, this is how you build a rotation that does not burn the team out.

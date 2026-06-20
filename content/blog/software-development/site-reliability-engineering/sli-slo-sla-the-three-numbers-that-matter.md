---
title: "SLI, SLO, SLA: The Three Numbers That Matter"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make the three terms everyone confuses precise: compute an SLI from raw request counts, draw an SLO line on it with a window, and write an SLA with teeth that sits safely below."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "sli",
    "slo",
    "sla",
    "error-budget",
    "reliability",
    "observability",
    "prometheus",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-1.png"
---

A vice president once stood up in an incident review I was running and said the words that have started a thousand bad reliability programs: "I want this service to be a hundred percent reliable. No more downtime. Ever." Everyone nodded. It sounded responsible. It sounded like leadership. And it was, quietly, the most expensive and least achievable sentence anyone said all quarter — because "a hundred percent reliable" is not a target, it is a category error. It mixes up the thing you measure, the line you draw on that measurement, and the promise you make to a customer, all into one fuzzy wish that cannot be engineered, budgeted, or even checked.

The fix is three numbers, and the discipline to keep them apart. An **SLI** — a Service Level Indicator — is a *measurement*: a number, almost always a ratio of good events to total events, that tells you how the service is actually behaving right now. An **SLO** — a Service Level Objective — is the *target* you draw on that measurement: the line that says "this is reliable enough," over a stated window. An **SLA** — a Service Level Agreement — is the *contract* you sign with a customer, with consequences (refunds, credits, penalties) if you miss, and it is always looser than your internal SLO on purpose. The SLI is the measurement, the SLO is the line, the SLA is the promise with teeth. Get those three straight and almost everything else in reliability engineering — error budgets, burn-rate alerts, the decision to freeze a release or ship it — falls out as arithmetic instead of argument.

![A vertical stack showing SLI as the good over total measurement at the base, the SLO target drawn on it in the middle, and the looser SLA contract at the top with an error budget and buffer branching off](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-1.png)

This post is the second stop in the series — right after [the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), which argued that reliability is a feature you engineer rather than a virtue you hope for. Here we make that concrete. By the end you will be able to: write down the exact ratio that defines an availability SLI and a latency SLI; compute both from a week of raw request logs with the actual good and total counts; turn a percentage into minutes of allowed badness per month so you can feel what a "nine" costs; draw a defensible SLO with a window and a consequence; explain to a salesperson why the customer SLA must be looser than the internal SLO; and recognize the four mistakes that quietly break most SLO programs. We are still on the series spine — **define reliability (SLI/SLO) → measure it → spend the error budget → respond → learn → engineer the fix** — and this is the *define* step. You cannot measure, budget, or spend anything until these three numbers are precise. So let us make them precise.

## 1. The SLI: a measurement, and almost always a ratio

Start with the indicator, because everything else is built on it. A **Service Level Indicator** is a quantitative measure of some aspect of the service's behavior. That is the whole definition. It is a *number*, it changes over time, and it is supposed to reflect something a user would actually notice.

The trap is that "a number that reflects behavior" is a huge space, and most of it is junk. CPU utilization is a number. The count of log lines per second is a number. The temperature of the rack is a number. None of those are good SLIs, because a user does not feel CPU at 70% — a user feels *their request failed* or *their page took four seconds to load*. The discipline of choosing a good SLI is a whole topic of its own (the next post, [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain), is entirely about it), but the structural rule we need right now is simpler and almost universal:

**A good SLI is a ratio of good events to total valid events, expressed as a percentage, bounded between 0% and 100%, where higher is better.**

$$\text{SLI} = \frac{\text{good events}}{\text{total valid events}} \times 100\%$$

That shape is not an accident. It is the shape that makes an SLI composable with everything downstream. Because it is bounded 0–100% and higher-is-better, you can subtract it from a target to get a budget. Because it counts events, you can compute it over any window by just summing the numerator and denominator over that window. Because it ignores the *magnitude* of badness and only counts the *fraction* of badness, it does not get distorted by one giant request or one tiny one. If your candidate "SLI" is not a ratio of good over total — if it is a raw latency in milliseconds, or a count of errors with no denominator, or an average that a few outliers can drag around — you have probably not chosen an SLI yet. You have chosen a metric. They are not the same thing, and the difference is exactly the difference between "p99 latency is 412ms" (a metric) and "99.1% of requests completed under 300ms this week" (an SLI).

### Two ways to count: request-based and windows-based

There are two standard ways to turn the good/total idea into an actual number, and both show up constantly, so it is worth naming them.

The **request-based** SLI counts individual events. You define "good" for one request, then sum:

$$\text{SLI}_{\text{request}} = \frac{\sum \text{good requests}}{\sum \text{valid requests}}$$

For an availability SLI, "good" usually means "did not return a server error" — the response status was not a 5xx. For a latency SLI, "good" means "the request finished faster than some threshold." We will compute both from real counts in a moment.

The **windows-based** (or "good-minutes") SLI chops time into small fixed buckets — say one minute each — labels each *bucket* as good or bad, and takes the ratio of good buckets to total buckets:

$$\text{SLI}_{\text{window}} = \frac{\text{number of good minutes}}{\text{total minutes}}$$

A minute might be "good" if, during that minute, the error rate stayed under some threshold and the p95 latency was under some limit. Windows-based SLIs are how a lot of customer-facing SLAs are actually written, because "this many minutes of the month the service was usable" is something a lawyer can put in a contract and a customer can understand. The two definitions usually agree closely, but not always — a service that is broken for one user for the whole month but fine for everyone else looks terrible request-by-request for that one user and perfectly fine minute-by-minute in aggregate. Knowing which definition you are using, and why, is part of doing this honestly.

### What goes in the denominator is a design decision

Here is the subtlety that separates a defensible SLI from a misleading one: *what counts as a "valid" event in the denominator.* This is not a clerical detail — it changes the number, and people game it.

The figure below traces a single request through this classification. The request arrives and is counted in the total. It is then labeled good or bad. The SLI is the running ratio.

![A branching flow diagram in which a request arrives and is counted in the total, then splits into a good path for fast successful responses and a bad path for server errors or slow responses, both feeding into the total valid count that produces the SLI ratio](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-2.png)

The standard convention for an HTTP availability SLI is: count 5xx responses as *bad*, count 2xx and 3xx as *good*, and **exclude 4xx client errors from the denominator entirely**. Why exclude 4xx? Because a 404 or a 400 is usually the *client's* fault — they asked for something that does not exist, or sent a malformed request — and you do not want a buggy mobile app hammering a nonexistent endpoint to make your service's reliability number look bad when your service is behaving exactly as designed. But this exclusion is also exactly where people cheat: relabel a class of real failures as 4xx and your SLI magically improves while users suffer just as much. So the denominator decision is a place to be honest, document the convention, and review it. A good rule: a request belongs in the denominator if a healthy version of your service *could* have served it correctly. If your service should have handled it and didn't, it counts against you, whatever status code you slapped on it.

The 4xx exclusion has sharp edges worth naming, because not every 4xx is innocent. A `429 Too Many Requests` is technically a 4xx, but if *your* rate limiter is misconfigured and throttling legitimate traffic, that 429 is your fault and your user feels pain — yet the default 4xx exclusion sweeps it out of the denominator and your SLI stays green while users get rejected. A `401`/`403` flood can mean your auth service is broken (your fault) or that someone is probing you (not your fault) — the same status code, two different verdicts. And a `400` can be your client SDK serializing a request wrong, which is arguably your fault if you ship that SDK. The honest move is not to blindly exclude all 4xx but to exclude the ones that genuinely indicate a well-formed-but-impossible request, and to *count* the 4xx classes that indicate your own malfunction. Many mature teams pull 429s back into the bad bucket for exactly this reason. The principle stands: the denominator encodes a judgment about whose fault a failure is, and that judgment deserves to be explicit and reviewed, not buried in a regex nobody reads. The classification flow shown above traces the happy path; the edge cases here are why the convention needs a comment block above the PromQL explaining each exclusion in plain words.

### Worked example: computing two SLIs from a week of logs

Let us make this fully concrete with numbers, because the entire point of an SLI is that it is computable from raw counts. Suppose we run an API and pull a week of access logs. After excluding 4xx client errors, we have these aggregate counts:

```bash
Total valid requests (excludes 4xx):   8,000,000
  of which 5xx server errors:              5,600
  of which 2xx/3xx successes:          7,994,400

Latency, of the 8,000,000 valid requests:
  finished in < 300 ms (our "fast"):   7,968,000
  finished in >= 300 ms (too slow):       32,000
```

The **availability SLI** is the success ratio:

$$\text{SLI}_{\text{avail}} = \frac{7{,}994{,}400}{8{,}000{,}000} = 0.99930 = 99.930\%$$

The **latency SLI** is the fast ratio:

$$\text{SLI}_{\text{latency}} = \frac{7{,}968{,}000}{8{,}000{,}000} = 0.99600 = 99.600\%$$

Two numbers, two ratios, both bounded 0–100%, both higher-is-better, both computed by dividing a count of good things by a count of total things. That is an SLI. Notice we report them separately — availability and latency are *different* aspects of behavior, and a request can succeed (good for availability) while being slow (bad for latency). You do not average them into one mush; you track each as its own SLI with its own target. We will hold onto these two numbers — 99.930% and 99.600% — because in the next section we draw a line on them.

### When request-based and windows-based disagree

It is worth lingering on the two counting methods from earlier, because the case where they *disagree* teaches you something important about what an SLI can and cannot see. Recall the request-based SLI counts individual events (good requests over valid requests), while the windows-based SLI labels fixed time buckets good or bad and counts good buckets over total buckets. In a uniformly healthy or uniformly broken service the two numbers track each other closely. They split apart exactly when failure is *concentrated* — clustered in time, or clustered on a subset of users.

Take a concrete divergence. Suppose over a 30-day month your service handles 100 million requests, and 99.95% of them succeed request-by-request — a great-looking availability SLI of 99.95%. But suppose almost all of those failures were jammed into a single 90-minute incident on the 14th, when a bad shard took down one region. Compute the windows-based SLI for the same month at one-minute granularity: the month has 43,200 minutes, and you label a minute "bad" if the per-minute error rate that minute crossed some threshold. During the 90-minute incident, those 90 minutes are all bad; the rest of the month is clean. That gives a windows-based availability of:

$$\frac{43{,}200 - 90}{43{,}200} = \frac{43{,}110}{43{,}200} = 99.79\%$$

Now look at the gap. Request-based says 99.95% — *met* a 99.9% SLO. Windows-based says 99.79% — *missed* the same 99.9% SLO. Same month, same incident, two honest numbers, opposite verdicts. Which is right? Both are, for different questions. The request-based number answers "what fraction of attempts worked?" The windows-based number answers "what fraction of the time was the service usable?" A 90-minute total outage is barely a blip in request-based terms if your traffic that hour was low, but it is a 90-minute hole in windows-based terms regardless of how many requests it touched. Customer SLAs are very often written windows-based precisely because a customer experiences *time down*, not *requests failed* — a one-hour outage during their batch window ruins their day even if it overlaps a low-traffic hour for you. Engineering SLOs are very often request-based because they weight by actual usage and are cheaper to compute from existing counters. The lesson: state which method your SLI uses, because the *same target* can be met under one method and missed under the other, and that ambiguity is exactly where an argument with a customer is born.

## 2. The SLO: the line you draw on the indicator

An SLI by itself does not tell you whether the service is okay. 99.930% availability — is that good? Is that a fire? You cannot say, because "good enough" is not a property of the measurement. It is a *decision*. The decision is the SLO.

A **Service Level Objective** is a target value (or range) for an SLI, measured over a stated window. That is it. "99.9% of valid requests succeed, measured over a rolling 28 days" is an SLO. The SLI is 99.930% (the thing we measured); the SLO is 99.9% (the line we decided was the boundary of acceptable). The SLO is an *internal* engineering goal — it is the threshold that defines "reliable enough" for *your* team, and it is the single most important reliability decision you will make, because everything downstream keys off it: the error budget is `1 − SLO`, your alerts fire when you are burning that budget too fast, and your release policy is gated on whether you have budget left.

### Every SLO needs a window and a consequence

Here is a rule worth tattooing somewhere visible: **a target without a window and a consequence is not an SLO — it is a slogan.**

The before-and-after below shows why. On the left, "aim for 99.9%" — which 99.9%? Over the last hour? Last quarter? Since launch? Without a window, the number is unmeasurable, because you cannot compute a ratio without telling me what range of events to sum over. And without a consequence — without something that *happens* when you cross the line — nobody will ever act on it. The number becomes wallpaper. On the right, the same target becomes real: 99.9% over a rolling 28-day window gives you a concrete error budget of 43.2 minutes per month, and the consequence is explicit — if you burn that budget, you freeze risky releases until it recovers.

![A two-column before and after diagram contrasting a naked target of 99.9 percent with no window and no consequence against a complete objective of 99.9 percent over twenty-eight days with a budget of 43.2 minutes per month and a freeze-shipping consequence](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-3.png)

The **window** matters more than people expect. A rolling 28-day window is the most common choice, and 28 is deliberate — it is exactly four weeks, so every window contains the same number of weekends and weekdays, which keeps the budget from wobbling just because the calendar has five Mondays this month. A *rolling* window (always "the last 28 days") behaves very differently from a *calendar* window (resets on the 1st). A calendar window resets your budget on a fixed date, which creates a perverse "spend it before it expires" incentive at month-end and a brutal "we're already out and it's the 3rd" panic at month-start. A rolling window has no reset cliff; the budget continuously heals as old bad events age out the back of the window. Most mature SLO setups use rolling windows for alerting and a 28- or 30-day window for reporting. Pick one, write it down, and never let "99.9%" travel without it.

### Worked example: did we meet a 99.9% SLO, and what did it cost?

Back to our two measured SLIs. Suppose our SLO is **99.9% availability over the 28-day window**, and the week we measured is representative.

Our measured availability SLI was 99.930%. Our SLO line is 99.900%. We are *above* the line — the SLO was met. By how much? The gap between where we are and the floor is our margin:

$$99.930\% - 99.900\% = 0.030\% \text{ of headroom remaining}$$

Now turn that into something you can feel. The **error budget** is everything below the SLO: `1 − SLO = 1 − 0.999 = 0.001 = 0.1%`. That is the fraction of requests you are *allowed* to fail and still meet the objective. Of our 8,000,000 requests, the budget is:

$$0.001 \times 8{,}000{,}000 = 8{,}000 \text{ failures allowed}$$

We actually had 5,600 failures (the 5xx count). So we spent 5,600 of our 8,000 allowed, leaving 2,400 — we consumed **70% of the budget** this week. We met the SLO, but we are not comfortable; at this rate, one bad afternoon could blow through the remaining 30%. That is the kind of sentence an error budget lets you say — and it is the entire subject of the sibling post on [the error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). The SLO is what makes the budget arithmetic possible: no SLO line, no budget, no "we spent 70% this week," no principled decision about whether to ship.

### The nines, in minutes you can feel

Percentages lie to your intuition. 99.9% and 99.99% look almost identical — one extra nine, who cares — but the difference is enormous when you convert it into *allowed downtime*, because the badness you are allowed shrinks by 10x with every nine. This table is the single most useful thing to memorize in this whole post, so here it is, computed against a 30-day month (43,200 minutes) and a 365-day year:

| SLO ("nines") | Allowed bad fraction | Downtime per month | Downtime per year |
| --- | --- | --- | --- |
| 99% (two nines) | 1% | 7.2 hours | 3.65 days |
| 99.5% | 0.5% | 3.6 hours | 1.83 days |
| 99.9% (three nines) | 0.1% | 43.2 minutes | 8.77 hours |
| 99.95% | 0.05% | 21.6 minutes | 4.38 hours |
| 99.99% (four nines) | 0.01% | 4.32 minutes | 52.6 minutes |
| 99.999% (five nines) | 0.001% | 25.9 seconds | 5.26 minutes |

Read down that table slowly, because it reframes every reliability conversation. Three nines — which sounds very strict — gives you 43.2 minutes of badness a month. That is *one* slightly-too-long deploy. Four nines gives you 4.32 minutes a *month* — you cannot even acknowledge a page, open a laptop, and SSH into a box in four minutes, let alone diagnose and fix. Four nines means the fix must be *automatic*, because a human in the loop cannot respond fast enough to stay inside the budget. Five nines — 26 seconds a month — means you cannot deploy a config change by hand, ever; the whole system has to be self-healing and the humans only watch. Each nine is not "a bit better." Each nine is roughly **10x more expensive to build and operate**, because the cheap ways to fail are used up and you are now buying redundancy, automation, and multi-region failover to claw back the next tenth of a percent. The table is why "just make it a hundred percent" is not a request; it is a request for infinite money.

## 3. Why a 100% SLO is a mistake, not an aspiration

Let us kill the VP's sentence properly, because it is the most common and most damaging reliability mistake, and understanding *why* it is wrong teaches you most of what the error budget is for.

The figure below splits the problem into three independent dead ends. Chasing 100% fails on the cost path, the user path, and the change path simultaneously — and any one of them alone is fatal to the idea.

![A decision tree rooted at a 100 percent target that branches into three paths, a cost path where each nine is ten times harder leading to exploding spend on idle high availability, a user path where the user's own network drops more than your last nine, and a change path where a zero budget means no shipping and reliability rots](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-7.png)

**The cost path.** We just saw it in the nines table: each nine costs roughly 10x more. The first nine (90%) is "the service is on." The second (99%) is "we restart it when it crashes." The third (99.9%) is "we have redundancy and good deploys." The fourth (99.99%) is "we have automatic failover and no manual deploys." The fifth (99.999%) is "we are running active-active across regions with automated everything and a team that does nothing but reliability." Each step buys you a smaller slice of availability for a larger pile of money. 100% is the limit of that curve, where the cost goes to infinity for a benefit that goes to zero. There is no amount of money that buys 100%, because hardware fails, networks partition, and the laws of physics include cosmic rays flipping bits in your RAM.

**The user path.** This is the argument that actually ends the debate, because it is not about your budget — it is about reality. Your user reaches your service over their home WiFi, their phone's flaky LTE, their ISP's oversubscribed last mile, their corporate proxy, and a dozen hops of public internet, *before* a single packet touches your infrastructure. The measured reliability of a typical consumer internet path is somewhere around two to three nines all by itself — the user's own connection drops more requests than your fifth nine ever could. So if your service is already at 99.99% and you spend a fortune to reach 99.999%, **the user cannot tell.** The improvement is buried under the noise of their own network. You spent 10x to move a number nobody can perceive. The honest framing: there is a reliability level above which *additional reliability is invisible to the user*, and engineering past it is pure waste. Find that level. Set your SLO at it. Stop.

**The change path.** This is the deepest one, and it is the seed of the entire error-budget idea. The error budget is `1 − SLO`. If your SLO is 100%, your error budget is `1 − 1 = 0`. Zero budget means zero tolerance for *any* failure — and since every deploy, every config change, every schema migration carries some risk of failure, a zero budget means **you can never change anything.** And a system you never change does not stay reliable; it rots. Dependencies drift, certificates expire, the OS goes end-of-life, the one person who understood the failover leaves. A 100% SLO does not produce a perfectly reliable system; it produces a *frozen* system that slowly degrades while everyone is too scared to touch it. The non-zero error budget is what *buys you the right to ship* — it is permission, denominated in allowed failures, to take the risks that keep a system alive. The whole genius of the SRE model is that it reframes "how reliable should we be?" from a religious argument into this: pick the lowest number of nines your users can't tell apart from perfect, and spend everything below it on velocity.

#### Worked example: the marginal cost of one more nine

Put rough numbers on the cost path so the 10x rule stops being a slogan. Say your service runs single-region behind a load balancer and reliably delivers 99.9% (three nines) on a fleet that costs you a baseline `B` per month to operate. You decide the next nine — 99.99% — is worth it. What does it take? Three nines fails on single-region incidents: a bad deploy, a zone outage, a dependency blip. To absorb a *zone* outage you now run hot across at least two availability zones, which roughly doubles compute and adds cross-zone data replication. To absorb the *deploy* class of failure you build automated canary rollouts and automatic rollback so a bad release is caught in seconds, not the minutes a human takes. To absorb the *dependency* class you add circuit breakers and graceful degradation so a slow downstream sheds load instead of cascading. None of that is a knob you turn — it is months of engineering plus a standing increase in infrastructure spend, plausibly `1.5B` to `2B` per month, to move availability from 99.9% to 99.99%.

Now feel the trade. That move bought you `43.2 − 4.32 = 38.9` fewer minutes of allowed monthly downtime — a real improvement. But the *next* nine, 99.99% to 99.999%, requires active-active across *regions* with automated global failover and a team whose full-time job is reliability, and it buys you only `4.32 − 0.26 ≈ 4.06` more minutes a month. Same engineering-and-money step change, roughly a tenth of the benefit of the previous nine. That is the 10x rule made arithmetic: equal cost steps, geometrically shrinking benefit. Somewhere on that curve is the nine where the cost step exceeds what the extra minutes are worth to your users and your business — and the discipline of this whole post is finding that nine and stopping there, not one further.

The stress-test that seals it: ask "if I spend `2B` and reach 99.999%, and the user's home WiFi runs at 99.9%, what does the user experience?" The user's path is the *weakest link* — their experienced reliability is bounded by their own connection, around three nines, no matter how many nines you stack behind it. You spent `2B` to move a number the user mathematically cannot observe, because their own 99.9% connection drops far more requests than your shiny fifth nine ever will. The money bought a dashboard improvement and zero customer-felt improvement. That is not reliability engineering; that is reliability theater, and the nines table plus the user-path argument are how you say so with numbers in the room.

### How to reach for the right SLO target

So what number *should* you pick? The honest answer is empirical, not aspirational. Look at what the service has actually delivered over the last few months — measure the SLI historically. If it has been running at 99.95% without anyone trying, setting an SLO of 99.99% is writing a check your architecture can't cash; you'll be in violation constantly and the SLO will become noise everyone ignores. Set the SLO at or slightly below what you reliably deliver today, so that *most* of the time you are comfortably inside it and the budget is meaningful when it does get tight. Then, separately, decide whether *users need more* than you deliver — if they do, that is a project (more redundancy, better deploys), not an SLO edit. The SLO describes the line; the project moves the line. Don't confuse them. The full treatment of choosing the number lives in the sibling post [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something); the rule for here is just: pick a number you can actually hold, that users would notice if you dropped below, and not one nine more.

## 4. The SLA: the promise with teeth

Now the third number, and the one that confuses even senior engineers because it lives half in legal and half in engineering. A **Service Level Agreement** is a *contract* with a customer that specifies a level of service and, crucially, the **consequences** if you fail to deliver it — refunds, service credits, penalties, sometimes the right to terminate. The SLA is the only one of the three numbers with teeth. Miss your SLI target, nothing automatic happens. Miss your SLO, your own release policy kicks in (you stop shipping). Miss your SLA, *money leaves the building* — you owe the customer.

That single difference — consequences — changes everything about how you set it. The SLI and SLO are tools you use on yourself to run the service well. The SLA is a promise to someone who will hold you to it in dollars. So the SLA must be a number you can keep even on a bad month, because the downside of missing it is real and external.

### The cardinal rule: SLA looser than SLO

Here is the relationship that ties the three numbers together, and the one most teams get backwards the first time:

$$\text{SLA target} < \text{SLO target} \le \text{what you can actually deliver}$$

Your customer-facing SLA must be **looser** (a lower target) than your internal SLO. You promise the customer 99.5% but you engineer toward 99.9% internally. The gap is not sloppiness — it is the entire point. It is a *buffer*, and the before-and-after below shows what the buffer buys you.

![A two-column before and after diagram showing that when the internal SLO equals the customer SLA at 99.5 percent any miss of the SLO is instantly a miss of the SLA and credits are owed immediately, versus setting the internal SLO stricter at 99.9 percent so the SLO alerts fire first and the team can fix the problem before the SLA is ever breached](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-8.png)

If your internal SLO and your customer SLA are the *same* number, then the very moment you cross your SLO you have also breached the SLA — there is no warning, no margin, and no time to fix it before you owe credits. Your own alerting and your contractual liability trigger at the same instant. That is a terrible place to be. But if your SLO is 99.9% and your SLA is 99.5%, then crossing the SLO is an *internal* event — your dashboards go red, your release freezes, your on-call investigates — while the SLA is still comfortably intact. You have the whole gap between 99.9% and 99.5% as room to detect, diagnose, and fix the problem *before* it ever becomes a customer-facing breach. The internal SLO is your early-warning line; the external SLA is the line where it gets expensive. You always want to hit the warning line first. That is why the SLA is looser: so that the SLA breaching is something that *should essentially never happen* if your SLO discipline is working, because the SLO breaches and self-corrects long before.

### Worked example: setting the SLO and SLA pair for a new API

Concretely, suppose you are launching a paid API and need to write both numbers. Here is the reasoning, step by step.

First, measure (or estimate from a similar service) what you can actually deliver. Say your architecture — load-balanced, two availability zones, decent deploys — reliably hits about 99.93% in normal operation, dipping to maybe 99.85% on a bad month with an incident.

Second, set the **internal SLO** at a number you hold *most* months but that bites when something is wrong: **99.9%**. You clear it in good months (99.93% > 99.9%) with a little room, and you breach it in bad months (99.85% < 99.9%) — which is *correct*, because a bad month *should* trip your internal alarm so you investigate and freeze releases. An SLO that never trips is set too loose; one that always trips is set too tight. 99.9% is the Goldilocks line here.

Third, set the **customer SLA** below the worst month you expect, with margin: **99.5%**. Even your bad-month 99.85% clears 99.5% comfortably. For the SLA to breach, you would need a *catastrophic* month — well below anything your normal operations produce — which means by the time you owe a credit, something genuinely went badly wrong and you knew about it for a long time (because your 99.9% SLO tripped weeks of budget-burn ago). The SLA breach becomes the rare, loud, "we truly failed this customer" event it should be, not a monthly accounting routine.

Fourth, write the **consequence**: a service credit. A typical schedule pays the customer a percentage of their monthly fee back if availability drops below the SLA, laddered by severity. The figure below shows such a schedule, and notice that every contractual tier sits *below* the internal 99.9% SLO — the first credit only triggers once you are already well past your own internal line.

![A four-row credit schedule matrix mapping monthly availability bands to service credits and to their relationship with the internal SLO, where 99.5 percent and above meets the SLA with zero credit and sits inside the SLO buffer, 99.0 to 99.5 percent is a minor breach paying 10 percent credit with the SLO already missed, 95 to 99 percent is a major breach paying 25 percent, and below 95 percent is a severe breach paying 50 percent and triggering an incident review](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-6.png)

In contract language, the same schedule reads like this — a real artifact you can adapt:

```yaml
sla:
  service: "Payments API"
  metric: "monthly_availability_percent"
  definition: "successful_requests / valid_requests over a calendar month"
  exclusions:
    - "4xx client errors"
    - "scheduled maintenance announced >= 7 days ahead"
    - "force majeure"
  target: 99.5            # the contractual promise (looser than internal SLO 99.9)
  credit_schedule:
    - { if_availability_below: 99.5, credit_percent: 10 }
    - { if_availability_below: 99.0, credit_percent: 25 }
    - { if_availability_below: 95.0, credit_percent: 50 }
  claim_window_days: 30    # customer must request the credit within 30 days
  cap: "50% of the monthly fee"
```

Three things in that YAML earn their keep. The **exclusions** define the denominator just like the SLI did — 4xx is out, announced maintenance is out — and these are negotiated, so read them carefully on any SLA you sign or write. The **credit schedule** ladders the penalty so a small miss costs a little and a catastrophe costs a lot. And the **claim window** plus **cap** are the legal guardrails: the customer has to actually ask for the credit within a window, and your total liability is capped, so an SLA can't accidentally bankrupt you on one terrible month. The engineering numbers (the SLI, the SLO) live upstream of all this; the SLA is where they meet the contract.

## 5. The three numbers in one picture, and how they compose

Let me put the whole relationship in one place, because the composition is the thing to walk away with. The first figure of this post stacked them; here is the same idea as prose you can recite.

The **SLI** is the measurement — `good / total`, bounded 0–100%, computed continuously from raw counts. The **SLO** is the line you draw on the SLI — a target plus a window — and it is *internal*. The **error budget** is `1 − SLO`, the allowed badness below the line, and it is the currency that lets you decide whether to ship. The **SLA** is a *separate, looser* target written into a customer contract with a *consequence*, set below the SLO so that you breach your own internal line (and react) long before you breach the contractual one (and pay). Measurement, line, budget, contract. They nest: the SLA sits below the SLO sits below what you can deliver, and the SLI is the live number that tells you where you actually are inside all of it.

The reason this composition is powerful is that it converts arguments into arithmetic. "Should we ship the risky feature this week?" is no longer a debate about who is more cautious — it is a lookup: how much error budget is left? If the budget is healthy, ship and learn. If it is nearly spent, freeze and stabilize. "Are we reliable enough?" is no longer a vibe — it is `SLI ≥ SLO?`. "Do we owe this customer money?" is no longer a panic — it is `monthly availability < SLA target?`, with a credit schedule that answers itself. Three numbers, three precise questions, three arithmetic answers. That is the whole reason to keep them straight.

There is a second, quieter benefit to the composition, and it is organizational: the three numbers each belong to a different *owner*, and keeping them distinct keeps the ownership clean. The SLI belongs to whoever instruments the service — it is a measurement fact, not a negotiation, and it should be computed the same way whether the news is good or bad. The SLO belongs to the service owner together with product, because deciding "how reliable is reliable enough" is a product trade-off (more nines means slower features) as much as an engineering one. The SLA belongs to sales, legal, and engineering jointly, because it is a commercial promise with legal consequences that engineering has to be able to keep. When a team collapses these into one number, they also collapse the ownership, and you get the classic dysfunction where a salesperson promises a customer "five nines" in a deal because nobody told them that the *SLI* has never measured above three. Keeping the numbers separate keeps the conversations separate: the engineer reports the SLI, the service owner sets the SLO, and only then does sales write an SLA that engineering has signed off on as keepable. The arithmetic discipline and the organizational discipline reinforce each other — precise numbers force clear ownership, and clear ownership keeps the numbers precise.

### A comparison table to nail the distinctions

The single most common mistake in this whole area is using one of these words when you mean another — especially saying "SLA" when you mean "SLO," which happens in roughly every other meeting. Here is the table that pins each term to what makes it different:

| | SLI | SLO | SLA |
| --- | --- | --- | --- |
| What it is | A measurement | An internal target | An external contract |
| Form | A ratio, good/total | A target + window | A target + window + consequence |
| Audience | Engineers, dashboards | Engineering org | Customers, legal |
| Consequence of missing | None (it's just data) | Internal: freeze releases | External: refunds/credits |
| Who sets it | SRE / service owner | Service owner + product | Sales + legal + engineering |
| Strictness | n/a (it's the measurement) | Stricter (the early line) | Looser (the safe promise) |
| Example | "99.93% succeeded this week" | "99.9% over 28 days" | "99.5%/mo or 10% credit" |

If you internalize one row, make it the strictness row: **SLO stricter, SLA looser.** Everything else follows. And if you catch yourself or a colleague saying "our SLA is 99.9%, so let's alert when we drop below it" — stop. That sentence has confused the internal target with the customer contract. You alert on the *SLO*, which is stricter, precisely so the alert fires before the *SLA* (looser) is anywhere near breaching.

## 6. The practice: computing an SLI in Prometheus, and writing an SLO spec

Enough theory — let's make the numbers real in the tools, because an SLO that nobody measures is the fourth great mistake (we'll get to all four). The point of this section is that everything above is *computable* from your existing telemetry with a few lines of PromQL and a YAML spec, and once it is, the numbers stop being slideware and start driving alerts.

### The PromQL: an availability SLI from request counters

Assume you export the standard HTTP server counter `http_requests_total` with a `code` label (the status) and you scrape it in Prometheus. The availability SLI — successful over valid, excluding 4xx — is a ratio of two `rate()`s. Here is the recording rule that computes it over a 28-day window:

```yaml
groups:
  - name: payments_api_sli
    interval: 30s
    rules:
      # numerator: non-5xx responses, excluding 4xx from the count entirely
      - record: job:http_requests_good:rate28d
        expr: |
          sum(rate(http_requests_total{job="payments-api", code!~"5..", code!~"4.."}[28d]))

      # denominator: all valid requests = total minus 4xx client errors
      - record: job:http_requests_valid:rate28d
        expr: |
          sum(rate(http_requests_total{job="payments-api", code!~"4.."}[28d]))

      # the SLI itself: good / valid, a number in [0, 1]
      - record: job:sli_availability:ratio28d
        expr: |
          job:http_requests_good:rate28d
          /
          job:http_requests_valid:rate28d
```

Read what those three rules do. The first sums the rate of "good" responses — anything that is not a 5xx and not a 4xx, i.e. 2xx and 3xx — over a 28-day window. The second sums all *valid* requests by excluding 4xx (the client's fault) from the denominator, exactly matching our SLI definition. The third divides them, giving a number between 0 and 1 that is your live availability SLI. That `job:sli_availability:ratio28d` series is the thing you graph, the thing your SLO compares against, and the thing your alerts watch. It is the same `good / total` ratio from section 1, now updating every 30 seconds.

### The latency SLI with a histogram

The latency SLI — fraction of requests faster than 300ms — needs a histogram, because you have to count requests by their duration bucket. With a Prometheus histogram `http_request_duration_seconds_bucket`, the fast count is just the bucket with `le="0.3"`:

```promql
# fraction of requests served faster than 300ms over 28 days
sum(rate(http_request_duration_seconds_bucket{job="payments-api", le="0.3"}[28d]))
/
sum(rate(http_request_duration_seconds_count{job="payments-api"}[28d]))
```

The numerator counts requests that fell into the "≤ 0.3 seconds" bucket (Prometheus histogram buckets are cumulative, so `le="0.3"` is *all* requests at or under 300ms). The denominator is the total request count. The ratio is your latency SLI — again `good / total`, again bounded 0–1. Note that this is *not* `histogram_quantile()` — we are not asking "what is the p95 latency" (a metric); we are asking "what fraction was fast enough" (an SLI). The histogram-quantile version is useful for dashboards, but the *SLI* is the ratio against a threshold, because that is what composes into a budget. If you want the same number from a SQL store instead of Prometheus, it is the same ratio:

```sql
SELECT
  SUM(CASE WHEN status_code NOT BETWEEN 500 AND 599
           AND status_code NOT BETWEEN 400 AND 499
           THEN 1 ELSE 0 END)::float
  /
  SUM(CASE WHEN status_code NOT BETWEEN 400 AND 499
           THEN 1 ELSE 0 END)               AS availability_sli
FROM request_log
WHERE ts >= NOW() - INTERVAL '28 days';
```

Same definition, different engine: count the good rows, divide by the valid rows (excluding 4xx from both), get a ratio. The SLI does not care whether it lives in a time-series database or a data warehouse; it is arithmetic over event counts.

### The SLO spec: target plus window, as code

The SLO itself deserves to be a checked-in artifact, not a number in someone's head. A minimal, tool-agnostic SLO spec — the kind Sloth, OpenSLO, or a homegrown generator can consume — looks like this:

```yaml
apiVersion: openslo/v1
kind: SLO
metadata:
  name: payments-api-availability
spec:
  service: payments-api
  description: "Fraction of valid requests that succeed"
  indicator:
    ratioMetric:
      good:  { metric: "job:http_requests_good:rate28d" }
      total: { metric: "job:http_requests_valid:rate28d" }
  objectives:
    - target: 0.999          # the SLO: 99.9%
  timeWindow:
    - duration: 28d
      isRolling: true        # rolling, not calendar — no reset cliff
  alerting:
    burnRate: true           # generate multi-window burn-rate alerts
```

That file is the SLO. It names the indicator (the good/total ratio we built in PromQL), states the target (0.999), states the window (28 days, rolling), and asks for burn-rate alerting. Check it into the repo next to the service it governs. Now the SLO is *measured* — a generator turns it into the recording rules and the burn-rate alerts automatically, and the number on the dashboard is the number in the contract review, because they come from the same source of truth. This is what "an SLO nobody measures" looks like when fixed: the objective is code, the measurement is code, and they cannot drift apart.

### Proof: what changes when you do this

The before→after here is not about the service's reliability — it is about the *conversation*. Here is the measurable shift teams report after moving from vibes to a defined SLI/SLO:

| Question | Before (no SLO) | After (defined SLO) |
| --- | --- | --- |
| "Are we reliable enough?" | Argued in meetings, no answer | `SLI ≥ SLO`, answered on a dashboard |
| "Should we ship the risky change?" | HiPPO decides (highest-paid person) | Error budget left? Arithmetic decides |
| Time to know we're in trouble | After customers complain | Burn-rate alert, often hours earlier |
| "Do we owe this customer?" | Scramble + legal panic | Lookup against SLA credit schedule |
| Alert relevance | Page on CPU, disk, every cause | Page on SLO burn (user pain) |

None of those rows require the service to get *more* reliable. They require the service to get *measured*, against a line, with a budget. That is the entire return on defining the three numbers: the same system, now legible, now governable by arithmetic instead of argument. The reliability improvements come later, in the budget-spending and engineering posts; this post buys you the ability to *talk about* reliability precisely, which is the precondition for improving it.

## 7. Picking the SLI that actually reflects pain

We've been assuming "success ratio" and "fast ratio" are the right SLIs, and for a request/response API they usually are. But the choice of *which* indicator deserves a moment, because a beautifully computed SLI on the wrong signal is worse than no SLI — it gives you false confidence. The full treatment is the next post, [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain); here is the structural filter that keeps you honest.

A candidate SLI has to pass two tests: does it **track user pain** (when this number drops, do users actually hurt?), and is it **measurable** (can you compute it cheaply and reliably from telemetry you have?). The matrix below runs four common candidates through both filters.

![A four-row decision matrix scoring SLI candidates on whether they track user pain and whether they are measurable, where request success ratio scores high on both and is recommended, latency under 300ms tracks pain well and needs a histogram so is recommended, CPU utilization tracks no user pain and is rejected as a vanity metric, and an uptime ping every 60 seconds is weak on pain and blind to slow responses so is rejected](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-4.png)

Walk the rows. **Request success ratio** tracks pain directly (a failed request is a user who didn't get what they asked for) and is trivially measurable from logs — use it. **Latency under a threshold** also tracks pain (a slow page is a user who left) and is measurable with a histogram — use it, accepting the slightly higher instrumentation cost. **CPU utilization** tracks *no* user pain — a box at 90% CPU serving every request perfectly is fine, and a box at 20% CPU returning 500s is on fire — so despite being trivial to measure, it is a vanity metric and must be rejected as an SLI (it is a useful *cause* signal for debugging, just not a *symptom* SLI). **An uptime ping every 60 seconds** is the sneaky one: it feels like availability, but it only checks whether one synthetic request to one endpoint succeeded once a minute. It is blind to slow responses, blind to the 30% of real requests failing while your health-check endpoint stays green, and blind to everything between pings. It is weak on pain and weak on coverage — reject it as a primary SLI.

The deeper rule the matrix encodes: **measure the symptom, not the cause.** Users feel symptoms — failed requests, slow pages. They do not feel causes — high CPU, full disks, a wedged thread pool. SLIs are about symptoms. Causes are for your debugging dashboards (and the system-design treatment of [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) goes deep on wiring those up). Keep the two layers separate: the SLI is the user's experience as a ratio; the cause metrics are how you explain a bad SLI. Confusing them is how teams end up paging on-call at 3am for a CPU spike that no user ever noticed.

## 8. Watching the budget burn over the window

The SLO gives you a budget; the budget gets *spent* over the window. Visualizing that spend is how the abstraction becomes a thing you can feel in your gut, and it is the bridge to burn-rate alerting (a whole topic of its own later in the series). The timeline below tracks a single 28-day window's budget — starting at the full 43.2 minutes and ending barely inside the line.

![A five-event timeline tracking a 28-day error budget that starts with 43.2 minutes available on day one, holds at 38 minutes by day seven, drops sharply to 21 minutes burned after a bad deploy on day twelve, recovers slightly to 17 minutes left after the day thirteen rollback, and finishes the window at 9 minutes left having met the SLO](/imgs/blogs/sli-slo-sla-the-three-numbers-that-matter-5.png)

Read the story in the timeline. Days 1 through 7, the service hums along and the budget barely moves — normal background failures nibble a few minutes. Then day 12: a bad deploy ships, error rate spikes, and in a single afternoon the budget goes from ~38 minutes down to having *burned* 21 — more than half the month's allowance gone in hours. That is the shape of a real budget catastrophe: not a slow drift but a cliff. Day 13, on-call catches it, rolls back, and the bleeding stops with 17 minutes left. The rest of the window heals slowly as the bad afternoon ages toward the back of the rolling window, finishing at 9 minutes left — *just* inside the SLO. You met it, but barely, and the postmortem writes itself: one deploy, one window's budget, a freeze you should have called sooner.

This is the precise reason the SLO needs a *consequence*, and why "freeze releases when the budget is low" is the canonical one. After day 12, with the budget more than half gone, the correct policy response is to stop shipping anything risky until the window recovers — because another bad deploy now would breach the SLO outright. The error budget turns "should we be more careful?" into a measured trigger: below some threshold of remaining budget, the policy *automatically* shifts the team from feature work to reliability work, no meeting required. This is the heart of the error-budget model, and the dedicated post on [the error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) builds the full burn-rate machinery (fast and slow burn windows, multi-window alerts) on top of exactly this picture.

#### Worked example: a 14x burn rate empties the budget in two days

Let's quantify the cliff, because "burn rate" is the number that makes budget spending predictable. **Burn rate** is how fast you are spending budget relative to "even" spending. A burn rate of 1 means you'd spend exactly your whole budget evenly across the window and end at zero. A burn rate of 14 means you're spending 14x faster than that.

$$\text{burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}$$

Our SLO is 99.9%, so `1 − SLO = 0.001` (0.1% allowed error rate). Suppose the bad deploy pushes the observed error rate to 1.4%:

$$\text{burn rate} = \frac{0.014}{0.001} = 14\times$$

At 14x, how long until the 28-day budget is fully spent? The window is 28 days; at 1x you'd take all 28 days; at 14x you take:

$$\frac{28 \text{ days}}{14} = 2 \text{ days}$$

Two days. A 1.4% error rate — which sounds almost fine, "98.6% of requests are working!" — *empties a month of error budget in 48 hours.* That is why you cannot wait for the monthly report to notice; by then the budget is long gone. It is also why burn-rate alerts watch *multiple* windows: a fast window (say 1 hour) catches the 14x deploy-disaster before it eats the month, while a slow window (say 6 hours) catches a low-grade 2x–3x leak that would otherwise drain the budget over a week without ever tripping the fast alarm. The math here — error rate over `1 − SLO` — is the same arithmetic every alert in the later posts is built from. Internalize it now: high error rate plus tight SLO equals a fast-draining budget, and the burn rate is exactly how fast.

## 9. War story: the Google error-budget model and a credit that wasn't

Two real-world threads make the three numbers concrete: where the model came from, and what happens when the SLA does its job.

**The Google error-budget model.** The discipline in this post is not folklore — it is the formalization Google published in the *Site Reliability Engineering* book, and its central, somewhat radical claim is this: **100% is the wrong reliability target for basically everything.** The book argues exactly what section 3 argued — that the marginal cost of each nine is enormous, that users can't perceive reliability above their own connection's limit, and that a 100% target leaves zero budget for the changes that keep a system healthy. Out of that came the error-budget bargain that aligns two teams who are usually at war: developers want to ship fast (which risks reliability), SREs want stability (which slows shipping). The error budget ends the fight without a meeting — *if there is budget left, devs ship freely; if the budget is spent, the team freezes features and fixes reliability until it recovers.* Both sides agreed to the rule in advance, so the day-to-day decision is arithmetic, not politics. That bargain is the entire reason SLOs are worth the effort, and it only works because the three numbers are kept precise: a fuzzy SLO can't produce a crisp budget, and a fuzzy budget can't settle the ship-or-freeze argument. (The architecture-time view of this same model lives in the system-design post on [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation); this post is the running-it view.)

**The SLA that paid out — and the one that didn't.** Major cloud providers publish SLAs with exactly the credit-ladder structure we wrote: a compute SLA might promise 99.99% monthly availability with a 10% credit below that, 25% below 99%, and 100% below 95%. When a large provider has a multi-hour regional outage, customers in that region whose monthly availability dropped below the SLA file for credits, and the provider pays — that is the SLA doing its job, the consequence with teeth. But here is the instructive part, and it is the lesson of the whole post: those same providers run *internal* SLOs far stricter than their published SLAs. The public number might be 99.99%, while the internal objective the team actually engineers and alerts against is tighter still, precisely so that the internal SLO trips and triggers a response *long* before the contractual SLA is anywhere near breaching. The customer-facing breach that pays out is, by design, the rare catastrophe — the regional outage, the once-a-year event — not the routine bad week, because the routine bad week was caught and fixed at the internal SLO line weeks of budget earlier. When you read a vendor's SLA, you are seeing the looser, public number; the stricter internal one you never see is what actually runs the service. That gap, between the promise and the engineering target, *is* the buffer this whole post has been about. (To avoid putting fabricated figures in your mouth: treat the specific percentages here as illustrative of the structure — check any current vendor SLA for its exact tiers before quoting them.)

### Stress-testing the SLO/SLA pair

A definition is only as good as the edge cases it survives, so put our 99.9% SLO / 99.5% SLA pair through the wringer the way a real incident does.

*What if the dependency is down for two hours?* A downstream you call goes hard-down for 120 minutes and every request that needs it fails. Against a 99.9% SLO with a 43.2-minute monthly budget, a single 120-minute outage doesn't just dent the budget — it *triples* it. You are instantly and deeply in violation of the SLO, your burn-rate alert screams within minutes, and your release freeze engages. But the *SLA* (99.5%, ~3.6 hours/month allowed)? A 120-minute outage spends about 2 hours of a 3.6-hour SLA budget — painful, but you have *not* breached the contract, assuming the rest of the month is clean. That gap is the buffer doing exactly its job: the internal line breached and triggered a full response, while the customer line held. Had you set SLO = SLA, the same outage would have breached the contract and owed credits with no warning. The stress test confirms the design: the looser SLA absorbs the same incident that blows the stricter SLO, buying you the right to fail occasionally without paying every time.

*What if two incidents overlap in the same window?* The rolling-window budget is unforgiving about clustering: two 30-minute incidents in the same 28 days don't average out, they *add*, and against a 43.2-minute budget two such incidents (60 minutes) put you over. This is the correct behavior — it reflects that a user who hit both incidents had a genuinely bad month — and it is the reason the release-freeze consequence matters. After the first incident eats most of the budget, the freeze should already be on, so the *second* incident is far less likely to be a self-inflicted bad deploy. The budget doesn't just measure pain; its consequence reduces the chance of the next self-inflicted wound.

*What if the budget is already spent and a critical security fix must ship?* The error-budget policy is a default, not a suicide pact. A freeze stops *risky feature work*; it does not stop a fix that *reduces* risk. Shipping a security patch or a reliability fix while frozen is exactly correct — those changes spend budget to *protect* the remaining budget. The policy's job is to stop the team from adding new risk while reliability is fragile, not to forbid the very fixes that restore it. Encode this exception explicitly in the policy so nobody has to argue it during an incident.

## 10. The four mistakes that quietly break SLO programs

Most SLO programs don't fail loudly; they rot quietly through four specific mistakes. Name them so you can catch them in review.

**Mistake 1: an SLI that isn't a ratio of good over total.** Someone proposes "our SLI is p99 latency, target under 300ms." That's a *metric*, not an SLI — it's an unbounded number in milliseconds, not a bounded good/total ratio, and you can't subtract it from a target to get a budget. The fix is to reframe it as a ratio: not "p99 < 300ms" but "fraction of requests faster than 300ms ≥ 99%." Now it's bounded 0–100%, higher-is-better, and composes into a budget. **Test for this:** can you write `1 − SLI` and get a meaningful "fraction of badness"? If not, it isn't an SLI yet.

**Mistake 2: an SLO with no window.** "We target 99.9%" with no time window is the slogan we warned about — it is literally uncomputable, because a ratio needs a range of events to sum over. It also invites cherry-picking: in a bad week, point at the good quarter; in a good week, point at the good day. **The fix:** every SLO carries a window, and it's stated every time the number is. "99.9% over a rolling 28 days" — never just "99.9%."

**Mistake 3: confusing the internal SLO with the customer SLA.** This is the most common and most dangerous. Someone alerts on the SLA number, or sets the SLO equal to the SLA, or quotes the strict internal SLO to a customer as a promise. Each version collapses the buffer that protects you. **The fix:** SLO stricter, SLA looser, and they live in different documents with different audiences. Alert on the SLO. Promise the SLA. Never the other way around, and never the same number.

**Mistake 4: an SLO nobody measures.** A beautiful SLO doc that no dashboard computes is theater. If the number on the slide and the number a query would return have never been compared, you don't have an SLO — you have a hope with a percentage sign. **The fix** is section 6: the SLO is code, the SLI is code (PromQL/SQL), they share a source of truth, and a dashboard shows the live SLI against the SLO line. If you can't pull up the current SLI in ten seconds, the SLO isn't real.

There's a fifth, subtler failure worth a sentence: **an SLO no human has authority to enforce.** If the budget burns and nobody is empowered to freeze releases, the consequence is fiction and the whole apparatus is decoration. The error-budget policy needs a real owner who can actually stop the ship. That's an organizational fix, not a technical one, but it's the difference between an SLO that governs and an SLO that decorates.

## 11. How to reach for these (and when not to)

The three numbers are not free, and not every service needs all three. Here is the decisive guidance.

**Every user-facing service needs an SLI and an SLO.** If real people (or paying systems) depend on it, you need to know its good/total ratio (SLI) and you need a line that says reliable-enough (SLO). This is the floor. Without it you cannot prioritize reliability work against feature work, because you have no budget to spend. Define them even if they're rough at first — a measured-but-imperfect SLO beats a perfect-but-imaginary one.

**Only services with external contractual commitments need an SLA.** An SLA is a legal instrument with financial consequences; you write one when a customer is paying for a guarantee, not for every internal microservice. Your internal recommendation-service does not need an SLA — it needs an SLO (and the *teams that depend on it* might agree to an internal SLO-like target, sometimes called an SLA loosely, but with no money attached, just an internal agreement). Reserve the real, teeth-bearing SLA for paying customers and legal contracts. Writing an SLA for an internal batch job is cargo-culting.

**Do not chase nines users can't perceive.** Section 3 is the policy: find the reliability level above which the user's own network drowns out your improvements, set your SLO there, and stop. A 99.999% SLO for a service whose users reach it over flaky mobile connections is spending 10x for invisibility. Be especially ruthless here on internal and batch systems — a nightly ETL job does not need three nines of availability; it needs to finish correctly by morning, which is a *different* SLI (a freshness or completeness ratio) and a much looser SLO.

**Do not set an SLO so loose it never trips, or so tight it always does.** A loose SLO (one you clear every single month with room to spare) gives you no budget signal and no reason to act — it is decoration. A tight SLO (one you breach constantly) trains everyone to ignore the alarm, the boy who cried wolf. The right SLO bites occasionally — a bad month should trip it, a good month should clear it. If yours hasn't tripped in a year, tighten it; if it's red half the time, either loosen it or, better, fix the service so the existing line is achievable.

**When you genuinely need more reliability, that's a project, not an SLO edit.** Raising the SLO number does not make the service more reliable — it just makes you violate the new number. If users need 99.99% and you deliver 99.9%, the answer is engineering (redundancy, better deploys, automatic failover), tracked as work, after which you *earn* the right to raise the SLO. The SLO describes reality; projects change reality. Keep that arrow pointing the right way.

## 12. Key takeaways

- **SLI is the measurement, SLO is the line, SLA is the promise with teeth.** An SLI is a ratio of good events to total valid events (bounded 0–100%, higher is better); an SLO is a target on that SLI over a window; an SLA is a customer contract with consequences for missing.
- **A good SLI is almost always `good / total`.** If your candidate isn't a bounded ratio you can subtract from a target to get a budget, it's a metric, not an SLI. p99 latency is a metric; "fraction under 300ms" is an SLI.
- **The denominator is a design decision.** Exclude 4xx client errors (the client's fault), count 5xx as bad, and document the convention — because the denominator is exactly where people cheat.
- **Every SLO needs a window and a consequence.** "99.9%" alone is a slogan; "99.9% over a rolling 28 days, freeze releases if the budget burns" is an SLO. Rolling windows avoid the month-end reset cliff.
- **The nines table is worth memorizing.** 99.9% = 43.2 min/month, 99.99% = 4.32 min/month, each nine ~10x more expensive. Convert percentages to minutes to make cost felt.
- **Never target 100%.** It costs infinite money, sits below the noise of the user's own network, and — most importantly — leaves zero error budget, which means you can never ship, which means the system rots. The non-zero budget is what buys the right to change.
- **SLA looser than SLO, always.** Set the internal SLO stricter (the early-warning line) and the customer SLA looser (the expensive line), so you detect and fix breaches before any credit is owed. SLO 99.9% internal, SLA 99.5% external is a healthy pair.
- **Error budget = `1 − SLO`, and burn rate = error rate / (1 − SLO).** A 1.4% error rate against a 99.9% SLO burns at 14x and empties a 28-day budget in 2 days — which is why you watch burn rate over multiple windows, not the monthly report.
- **Watch for the four mistakes:** an SLI that isn't a ratio, an SLO with no window, confusing the SLO with the SLA, and an SLO nobody measures. Each one quietly hollows out an otherwise sensible program.
- **Define before you measure.** This is the *define* step of the series spine; you cannot measure, budget, or spend reliability until these three numbers are precise. Make them precise, then everything downstream becomes arithmetic.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro this post builds on, framing reliability as something you engineer and budget rather than wish for.
- [Choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) — the next post, on picking indicators that track symptoms users feel, not causes that only you see.
- [Setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) — how to choose the actual target number empirically so the line bites occasionally without crying wolf.
- [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — the full burn-rate machinery and the ship-or-freeze policy built on the `1 − SLO` budget introduced here.
- [Reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — the architecture-time companion: designing systems that degrade gracefully to stay inside their budgets.
- *Site Reliability Engineering* (the Google "SRE Book"), chapters on Service Level Objectives and Embracing Risk — the canonical source for the error-budget model and the "100% is the wrong target" argument.
- *The Site Reliability Workbook*, the chapters on implementing SLOs and on alerting on SLOs — practical recipes for the recording rules, multi-window burn-rate alerts, and SLO specs sketched here.
- The Prometheus documentation on recording and alerting rules, and the `rate()` and `histogram_quantile()` functions — the reference for the PromQL artifacts in section 6.

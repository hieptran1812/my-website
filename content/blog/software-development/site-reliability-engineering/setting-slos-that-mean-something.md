---
title: "Setting SLOs That Mean Something: The Number Is a Business Decision"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Choose an SLO target the way a principal SRE does: price each nine, read your achieved performance, find the user-perception threshold, compute the dependency-chain ceiling, set percentile latency targets, and negotiate the budget with stakeholders."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "slo",
    "error-budget",
    "availability",
    "latency",
    "percentiles",
    "reliability",
    "prometheus",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/setting-slos-that-mean-something-1.png"
---

A staff engineer I worked with once spent a full sprint chasing the last fraction of a percent on a service that was already at 99.95% availability. He shaved a few hundred milliseconds off a p99 nobody had complained about, added a second redundant cache layer, and rewrote a retry path so it could survive a failure mode that had happened exactly once in two years. The work was genuinely good. It was also, in hindsight, close to worthless — because the service sat behind a corporate VPN whose own uptime was nearer 99.7%, and behind a third-party identity provider that took itself offline for maintenance one Sunday a month. No user on earth could perceive the difference between 99.95% and 99.99% on that path. The VPN dropped more requests in a single bad afternoon than our extra nine would save in a year. We spent a sprint of senior engineering time buying reliability that physically could not reach the user.

That is the most common way an SLO program fails: not by being too lax, but by treating the target as a virtue to maximize instead of a number to *choose*. An SLO target — the line that says "this service is reliable enough" — looks like an engineering metric, but it is really a business decision wearing an engineering costume. Set it too strict and the team drowns in pages for badness no user noticed, mutes the alert out of self-defense, and ends up with *worse* reliability than if they had no SLO at all. Set it too loose and the SLO never triggers, never protects anyone, and becomes a number on a dashboard that everyone has learned to ignore. The right target lives in a narrow band: high enough that missing it actually corresponds to user pain, low enough that hitting it is achievable and that the next nine is worth what it costs. Finding that band is a skill, and almost nobody is taught it.

![A vertical stack showing the four inputs to an SLO target which are user perception, achieved performance, dependency ceiling, and cost of the next nine, all feeding into a chosen target that sits in a narrow band](/imgs/blogs/setting-slos-that-mean-something-1.png)

This post is the fourth stop in the series. It builds directly on [the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), which argued reliability is a feature you engineer rather than a virtue you hope for, and on [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter), which made the three terms precise. There we said *what* an SLO is — a target drawn on an SLI over a window. Here we answer the harder question that everyone skips: *what number do you actually put there, and how do you defend it?* By the end you will be able to read the nines-to-downtime table from memory and feel what each nine costs; explain why each added nine costs roughly ten times the previous one; choose an availability target from four real inputs instead of vibes; compute a dependency-chain ceiling so you never promise more than your dependencies can deliver; set latency SLOs on percentiles instead of the lie that is an average; tier multiple journeys at different targets; and walk into a room with product and leadership and negotiate the number as a shared commitment rather than impose it as an ops edict. We are on the series spine — **define reliability (SLI/SLO) → measure it → spend the error budget → respond → learn → engineer the fix** — and this is still the *define* step, but the part of it that decides everything downstream. The error budget you get to spend is exactly `1 − SLO`. Pick the SLO wrong and every burn-rate alert, every release-freeze decision, every on-call rotation inherits the mistake.

## 1. The cost of each nine: the table you should know cold

Before you can choose a target you have to feel what each target *costs the user* — not in money yet, but in raw allowed downtime. Reliability is quoted in "nines": 99% is "two nines," 99.9% is "three nines," and so on. The number of nines tells you the fraction of time the service is allowed to be down, and the most useful thing you can do as an SRE is to stop hearing "99.9%" as an abstract percentage and start hearing it as *"forty-three minutes a month."* Because that is what it is.

The conversion is pure arithmetic. The error budget — the allowed badness — is `1 − SLO`. Multiply that fraction by the length of the window and you get the allowed downtime in real time units. A month is about 30.44 days on average, which is 43,830 minutes; a 30-day month is 43,200 minutes; a 365-day year is 525,600 minutes. So a 99.9% target over a 30-day month allows `(1 − 0.999) × 43,200 = 0.001 × 43,200 = 43.2` minutes of downtime. That is the whole calculation. Here is the table, which you should genuinely memorize, because it ends more arguments than any slide deck:

| Availability (SLO) | Nines | Downtime / month (30 d) | Downtime / year | Error budget |
| --- | --- | --- | --- | --- |
| 90% | one nine | 3 days (72 h) | 36.5 days | 10% |
| 99% | two nines | 7.2 hours | 3.65 days | 1% |
| 99.5% | — | 3.6 hours | 1.83 days | 0.5% |
| 99.9% | three nines | 43.2 minutes | 8.77 hours | 0.1% |
| 99.95% | — | 21.6 minutes | 4.38 hours | 0.05% |
| 99.99% | four nines | 4.32 minutes | 52.6 minutes | 0.01% |
| 99.999% | five nines | 25.9 seconds | 5.26 minutes | 0.001% |

![A figure showing how downtime shrinks dramatically as availability climbs from 99 percent at 7.2 hours per month down to 99.999 percent at 26 seconds per month](/imgs/blogs/setting-slos-that-mean-something-2.png)

Read down that column and a few things jump out. Two nines — 99% — sounds respectable until you see it allows **7.2 hours of downtime every month**. That is most of a working day, every month, gone, and nobody would call a service that's down for a workday a month "reliable." Three nines, 99.9%, is the workhorse target for most user-facing services: 43.2 minutes a month, about ten minutes a week, is roughly the band where a typical web service lives if it is run with discipline. Four nines, 99.99%, is **4.32 minutes a month** — that is so little downtime that a single bad deploy, a single slow database failover, a single thirty-second blip will *blow your entire month's budget in one event*. And five nines, 99.999%, is **26 seconds a month** — so tight that you cannot even safely do a manual rollback within budget; the only way to hit it is full automation with instant failover, and even then you're one human reaction time away from missing.

### The exponential cost of the next nine

Here is the part that the nice clean table hides, and it is the single most important idea in this whole post: **each additional nine costs roughly ten times as much as the one before it.** This is not a rule of thumb someone made up; it falls out of the arithmetic of the budget itself.

Going from 99% to 99.9% shrinks your allowed downtime from 7.2 hours/month to 43.2 minutes/month — you have cut the permitted badness by a factor of ten. To actually *achieve* a tenfold reduction in failures, you have to eliminate ninety percent of whatever was causing the old failures. The easy failures — the ones that cause most of your downtime — you kill first and cheaply. But each subsequent nine forces you to hunt down rarer and rarer failure modes: the once-a-quarter rack power event, the leap-second bug, the BGP route flap, the cloud zone that browns out for ninety seconds. Those tail failures are expensive precisely because they are rare: you cannot reproduce them on demand, you have to build redundancy that sits idle 99.99% of the time waiting for them, and you have to staff people who can respond in seconds rather than minutes. The cost of removing the *last* 10% of failures is enormous compared to the cost of removing the *first* 10%.

So the cost curve of reliability is not linear — it is roughly exponential in the number of nines. Each nine demands about an order of magnitude more engineering effort, more redundant infrastructure, more on-call discipline, and more architectural complexity than the last. This is why "as reliable as possible" is not an engineering goal; it is a budget hole. The right framing is a **cost/benefit tradeoff**: every nine has a price, and you buy a nine only when the value it delivers to the user and the business exceeds that price.

It helps to make the cost concrete by walking up the ladder of what each nine *physically requires*, because the exponential isn't abstract — it's a list of increasingly expensive engineering commitments. To hold **two nines** (99%) you mostly need a service that doesn't fall over constantly: a single deployment, basic health checks, a human who can restart it within a couple of hours. That's a normal, single-region, single-instance-with-a-restart kind of operation, and most teams can do it without thinking hard. To hold **three nines** (99.9%) you need redundancy so that a single instance dying doesn't take you down: multiple replicas behind a load balancer, automated restarts, a deploy process that can roll back, and on-call coverage that responds within tens of minutes. That's a real but ordinary step up — most competent web services live here. To hold **four nines** (99.99%) the human stops being able to react fast enough: 4.32 minutes a month is less time than it takes a person to read a page, log in, and diagnose, so *everything that matters must be automated*. Automated canary analysis that aborts a bad deploy on its own. Automated failover that promotes a standby database in seconds without a human in the loop. Circuit breakers that shed a failing dependency before it cascades. Multi-zone or multi-region deployment so a single zone browning out doesn't end your month. And to hold **five nines** (99.999%) — 26 seconds a month — you're in the territory of telephone switches and stock exchanges: full active-active multi-region with instant traffic shifting, hardware redundancy at every layer, formally verified failover, and an operations culture where every change is treated as a potential outage. Each rung up that ladder is roughly ten times the cost and complexity of the one below, and the people who staff and build each rung are progressively scarcer and more expensive.

The reason this matters for *choosing* an SLO is that the exponential cost curve sets up the central tradeoff: on one side, the value of an extra nine — usually some marginal revenue, conversion, or trust that you might capture by being more reliable — and on the other, the order-of-magnitude jump in cost to buy it. For a payments processor where every minute of downtime is measured in tens of thousands of dollars of failed transactions and regulatory exposure, four or even five nines can pencil out. For a content site where a minute of downtime means a few readers refresh and come back, three nines is generous and four would be a waste. The number you choose is *where on that curve the marginal value of the next nine drops below its marginal cost* — and that point is different for every service, which is exactly why "as reliable as possible" is the wrong answer and "as reliable as the user needs, no more" is the right one.

![A before and after contrast showing the naive view that more nines is always better versus the real view where each nine costs ten times more so the target is chosen where value exceeds cost](/imgs/blogs/setting-slos-that-mean-something-3.png)

#### Worked example: pricing the jump from three nines to four

Suppose your service runs at 99.9% today and leadership asks for 99.99%. Let's price it honestly. At 99.9% you have 43.2 minutes/month of budget; at 99.99% you have 4.32 minutes/month. You must cut your downtime by ninety percent. Where does today's downtime come from? Say a representative month looks like this: a 12-minute deploy that went bad and got rolled back, a 9-minute database failover during a primary crash, an 8-minute incident from a dependency timeout, two 4-minute blips from autoscaling lag, and a 6-minute config push that broke a route. That is 43 minutes — right at budget for three nines.

To get to four nines you must reduce that 43 minutes to **under 4.32 minutes**. The 12-minute bad deploy has to become a sub-minute automated rollback, which means canary analysis with automatic abort (see the system-design treatment of [reliability, SLOs, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation)). The 9-minute database failover has to become a sub-30-second automated failover, which means a hot standby with health-checked automatic promotion — real money in standby infrastructure that sits idle most of the time. The dependency timeout needs a circuit breaker with a cached fallback. The autoscaling lag needs pre-provisioned headroom (you pay for capacity you usually don't use). The config push needs a staged rollout with automated validation. Every single one of those is real engineering work plus real standing infrastructure cost, and you have to land *all* of them, because at 4.32 minutes/month a single one of those events blowing past budget fails the month. That is what "the next nine costs 10×" means in practice: you are not optimizing one thing, you are systematically eliminating every category of failure and buying redundancy against the ones you can't eliminate. If users genuinely need four nines — a payments processor, a stock exchange, an emergency dispatch system — it is worth it. If they don't, you just spent a quarter of senior engineering time buying downtime nobody will ever notice.

## 2. The first input: what users actually need and can perceive

Now we start choosing the number, and the first question is the one teams most often skip: **what do users actually need, and what can they even perceive?** There is a ceiling on useful reliability set not by your engineering but by physics and human attention, and past that ceiling every nine you add is invisible.

Consider the chain a request travels through before it reaches a human. The user is on a phone on a cellular network, or a laptop on home WiFi, or a desktop behind a corporate proxy. That last-mile network is not reliable. Consumer cellular and WiFi connections routinely drop one in several hundred to one in a thousand requests just from radio interference, handoffs between towers, and congestion — call it somewhere around 99.7% to 99.9% reliable on a good day, much worse on a train. The user's device itself stalls, backgrounds your tab, runs out of memory, and gets interrupted by notifications. By the time you account for the network and device that *you do not control*, the user is already experiencing something well short of perfect — and crucially, **they cannot distinguish your contribution to failure from the network's contribution.** A request that fails because your server returned a 500 looks, to the user, exactly like a request that failed because their elevator passed a dead zone.

This gives us a hard principle: **once your service's reliability is meaningfully better than the user's own network and device, additional nines are imperceptible.** If the user's path drops 0.2% of requests before they ever reach you, then the difference between your service being 99.99% reliable (you drop 0.01%) and 99.999% reliable (you drop 0.001%) is utterly lost in the noise of their own connection. You spent 10× the engineering effort to improve the user's experience by a factor that is one twentieth of the variance their own WiFi introduces. They will never, ever notice. This is exactly the trap my colleague fell into in the opening story.

The same logic applies to latency, and even more sharply, because human perception of speed has well-studied thresholds. Below about 100 milliseconds, an interaction feels instantaneous — the user perceives no delay at all. Up to about 1 second, the user notices the delay but stays in flow; their train of thought is not broken. Past about 1 second they start to feel the wait, and somewhere around 3 to 10 seconds — depending on the task and whether there's feedback — they begin to abandon. These thresholds matter enormously for choosing a latency SLO: **there is no user-perceptible reward for making a 90ms response 70ms,** because both are already below the instantaneous threshold. If your p99 latency is 90ms and you spend a sprint getting it to 70ms, the user cannot tell. But moving a p99 from 1,200ms to 800ms is a real, felt improvement, because you crossed from "I notice this is slow" toward "this feels responsive."

So the first input to your SLO is a question, not a number: **what is the threshold past which the user stops perceiving improvement?** For availability, it's roughly where your reliability exceeds the user's own path reliability. For latency, it's the perception threshold for the specific interaction — instantaneous for a keystroke autocomplete, a second or two for a search results page, longer for a deliberate "generate report" action the user expects to take time. You set the SLO at or just inside the threshold the user can perceive, and not a nine beyond it.

There's a subtlety worth being honest about: *perceptibility is per-interaction, not per-service*, and it depends on how the user reaches you. A request from a server-to-server integration — say a partner's backend calling your API in a tight retry loop — perceives reliability completely differently from a human on a phone. The machine notices every dropped request and retries it, so for an API consumed by machines, the relevant "user perception" is really "does the consumer's retry budget absorb my failures?" If your API drops 0.1% of requests but every consumer retries up to three times with backoff, the *effective* failure rate the consumer experiences is the cube of yours, vanishingly small — which means the consumer can tolerate a looser SLO than a human-facing path could. Conversely, a human watching a single page load has no retry loop; one failure is one visible failure. So when you ask "what can the user perceive?", be precise about *which* user: a human staring at a result, or a machine with a retry budget that hides your blips. The answer changes the target.

There's also a cumulative-perception effect that pushes the other way. A user who makes one request a week against your service barely interacts with your reliability at all — even at 99% they'd hit a failure only every two years of weekly use. But a user who makes hundreds of requests an hour — a power user, a dashboard that polls, a mobile app that syncs constantly — multiplies your per-request failure rate by their request volume, so they feel a much higher *cumulative* failure probability than your per-request SLI suggests. A 99.9% per-request availability means a heavy user making 1,000 requests a day expects about one failure a day. Whether that's perceptible depends entirely on what the failure costs them and whether your client retries it. The lesson is to think about reliability from the perspective of your *heaviest realistic user session*, not the average request, because that's where the cumulative pain concentrates.

## 3. The second input: your current achieved performance

The second input is the most concrete and the most often ignored: **what is your service actually doing right now?** You cannot choose a sensible target in a vacuum; you choose it relative to your achieved reliability. And there are two ways to get this wrong that are equally bad.

The first failure is **setting an SLO you are already missing.** Suppose your service is genuinely running at 99.5% — you measured it over the last quarter, it's real — and someone declares an SLO of 99.9%. Congratulations: you are now *out of budget on day one*. Every week your burn-rate alerts fire, every release decision says "freeze, we're over budget," and the team learns within a month that the SLO is a fantasy disconnected from reality. They stop trusting it. An SLO you are already violating doesn't motivate improvement; it just generates noise and teaches everyone that the number is fiction. If you want to *reach* 99.9% you set a roadmap to get there and you set the SLO at what you can actually hold *today*, then ratchet it up as you earn each nine (more on ratcheting in §9).

The second failure is the opposite and subtler: **setting an SLO you are already crushing by a wide margin.** If your service is reliably running at 99.99% and you set the SLO at 99.9%, you have a *useless* SLO. It will essentially never trigger. Your error budget is ten times bigger than you're actually using, so the budget never runs low, the burn-rate alerts never fire, and the SLO provides zero signal and zero protection. You could have a slow degradation that takes you from 99.99% to 99.95% — a real regression that users might start to feel — and your 99.9% SLO would sit there green and silent the whole time. The SLO is supposed to be a tripwire that fires when reliability degrades to the edge of acceptable. If you set it so far below your achieved performance that you'd have to fall off a cliff to trip it, it's not a tripwire, it's decoration.

So the rule is: **measure your achieved performance over a representative window (a quarter is good), and set the SLO close to it** — close enough that you can hold it, but tight enough that meaningful degradation trips it. A common and sane practice is to set the SLO a notch below your reliable achieved performance, so you have realistic headroom but the SLO still bites when things slide. Here is how you measure achieved availability honestly in Prometheus, as a ratio of good requests to total over a rolling 30-day window:

```promql
# Achieved availability over the trailing 30 days: good requests / total valid requests.
# "Good" = not a 5xx server error. Adjust the job label to your service.
sum(rate(http_requests_total{job="checkout", code!~"5.."}[30d]))
/
sum(rate(http_requests_total{job="checkout"}[30d]))
```

Run that, and you get your real number — say 0.9992, i.e. 99.92%. *Now* you can choose an SLO, because you know what you're actually delivering. Picking a target without first running this query is choosing a number with your eyes closed.

![A before and after figure contrasting an SLO chosen blind that is either already missed or already crushed against an SLO anchored to measured achieved performance with realistic headroom](/imgs/blogs/setting-slos-that-mean-something-4.png)

#### Worked example: reading achieved performance into a target

Your checkout service, measured over the last quarter with the query above, is delivering 99.92% availability, and the number is stable — it doesn't swing wildly month to month. What SLO do you set? Not 99.99%: you'd be out of budget constantly, since 99.92% is well below it, and the team would learn to ignore the alarm. Not 99% or 99.5%: those are so far below your achieved 99.92% that the SLO would never trip even if you regressed badly — a slide to 99.6% (a real, user-felt degradation) would still sit comfortably inside a 99.5% SLO. The honest target lives right around your achieved number. Setting it at 99.9% gives you a tiny bit of headroom below your achieved 99.92%, so you start in budget but not by a mile, and a regression of even a couple hundredths of a percent starts eating budget noticeably. We'll confirm 99.9% is the right landing in the full worked example in §8, but the achieved-performance reading alone already rules out everything above 99.95% and everything below 99.8%.

## 4. The third input: the dependency-chain ceiling

Here is the input that catches even experienced teams, and it is pure arithmetic that you can't argue your way around: **your SLO can never exceed the combined reliability of the things you depend on.** If your service calls three other services to do its job, and any one of them being down makes you fail, then your availability is *capped* by the product of theirs. You cannot be more reliable than your weakest critical path allows, no matter how flawless your own code is.

The math is the math of independent serial dependencies. If your request must successfully call service A (available with probability `p_A`), service B (probability `p_B`), and service C (probability `p_C`), and all three must succeed for your request to succeed, then the probability *your* request succeeds is the product:

$$\text{availability}_{\text{you}} \le p_A \times p_B \times p_C$$

Multiplying availabilities together always gives a smaller number than any single one of them, because each is below 1. This is why a chain of "pretty reliable" services produces a service that is *less* reliable than any of its links. Three dependencies each at 99.9% give a ceiling of `0.999 × 0.999 × 0.999 = 0.997002`, which is about **99.7%** — already a full nine *worse* than each dependency, just from chaining three of them. Add more dependencies and the ceiling keeps dropping. This is the single most important sanity check before you commit to an availability SLO: **add up your dependency chain and confirm the ceiling is above the number you want to promise.** If it isn't, no amount of effort on your own code can save you, because the failure is structural.

![A graph showing a request fanning out through three dependencies which are auth at 99.95 percent, database at 99.99 percent, and payments at 99.9 percent, multiplying down to a combined ceiling near 99.84 percent](/imgs/blogs/setting-slos-that-mean-something-5.png)

#### Worked example: computing a dependency-chain ceiling

Your order service must, on the critical path, call the auth service (99.95% available), the primary database (99.99%), and the payments gateway (99.9%). All three are required — if any fails, the order fails. What's the highest availability you can honestly promise?

$$\text{ceiling} = 0.9995 \times 0.9999 \times 0.999$$

Compute it step by step. `0.9995 × 0.9999 = 0.99940005`. Then `0.99940005 × 0.999 = 0.99840065`. So the ceiling is about **99.84%**. That is the most your order service can ever achieve, assuming your own code never adds a single failure of its own — which it will, so the real ceiling is a touch lower.

Now suppose someone has already written down a 99.9% SLO for the order service. That target is **above the ceiling.** It is *mathematically impossible* to hit, no matter how heroically the team works, because even a perfect order service inherits 99.84% from its dependencies. The 99.9% SLO will be missed every single month, forever, and the team will burn out chasing a number physics won't let them reach. The honest options are exactly two: **lower the target** to something the chain supports (say 99.8%, which leaves a sliver of room below the 99.84% ceiling for your own faults), or **fix the architecture** so the dependencies stop being serial single points of failure. You could make payments asynchronous so a payments blip doesn't fail the order synchronously (process the order, settle payment in the background with retries). You could add a cached fallback for auth so a brief auth outage doesn't block existing sessions. You could make the database read path survive a primary blip via a replica. Each of those *removes* a multiplicand from the critical-path product and raises the ceiling. But you must do one or the other. You cannot promise 99.9% on a chain that caps at 99.84%; the arithmetic is not negotiable, only the architecture is.

This is also why decoupling matters so much for reliability. Every dependency you can move off the synchronous critical path — by caching it, by making it asynchronous, by degrading gracefully when it's down — raises your ceiling. The architectural moves for this (circuit breakers, fallbacks, async processing, bulkheads) are covered in the system-design series; here the point is just that the dependency-chain calculation tells you *when* you're forced to make them. If your target sits above your ceiling, that's not an SLO problem, it's an architecture signal.

A few refinements make the dependency math more honest, because the naive product is a worst case and reality is usually a little kinder. First, the product assumes every request hits every dependency. If only 30% of your requests touch the payments gateway (the rest are read-only browsing that never reaches payments), then payments' 99.9% only drags down 30% of your traffic, and your blended ceiling is higher than the naive product suggests. To weight it properly you compute the ceiling per request *type* and blend by traffic share — a read request that touches only auth and the database has a ceiling of `0.9995 × 0.9999 ≈ 99.94%`, while a checkout that also hits payments caps at 99.84%, and your overall number is the traffic-weighted average. Second, the product assumes failures are *independent* — that auth being down and payments being down are unrelated events. In reality they often correlate (a shared network partition takes out both at once, a shared cloud zone failure hits everything in it), and *correlated* failures are actually slightly better for the product math than independent ones, because two dependencies failing at the same moment is one outage, not two — though they're worse operationally because everything breaks at once. Third, the product assumes *any* dependency failure fails your request, which is only true for hard, synchronous, no-fallback dependencies. The moment you add a fallback — serve stale data when the cache is down, queue the write when the database is slow, approve provisionally when fraud-check times out — that dependency stops being a hard multiplicand and starts being a soft one, and your ceiling climbs.

So the practical procedure is: list your critical-path dependencies, mark each as hard (failure fails the request) or soft (you have a fallback), multiply only the *hard* ones weighted by the fraction of traffic that touches them, and treat that product as your ceiling. Then if the ceiling is below your desired SLO, your move is to convert hard dependencies to soft ones — which is precisely the graceful-degradation work — until the ceiling rises above your target. The dependency-chain calculation isn't just a sanity check you run once; it's a design tool that tells you *which specific dependency to decouple next* to earn the nine you want. The cheapest nine to buy is almost always the one you get by turning your worst hard dependency soft.

## 5. Percentiles, not averages: why an average latency SLO is a lie

Everything so far has been about availability. Latency needs its own treatment, because the way most teams first try to write a latency SLO is actively misleading. The instinct is to say "average response time under 200ms." That sounds reasonable. It is a lie, and here's why.

An average collapses an entire distribution of response times into a single number, and in doing so it hides exactly the part you care about: the slow tail. Suppose 99% of your requests complete in a brisk 50ms, and 1% take a brutal 5,000ms because they hit a cold cache, a lock, or a slow shard. The average is `0.99 × 50 + 0.01 × 5000 = 49.5 + 50 = 99.5ms` — under 100ms, looks fantastic, "well within our 200ms SLO." But one in every hundred requests took five full seconds. If a user makes ten requests to load a page, the chance that at least one of those ten hits the 5-second tail is `1 − 0.99^10 ≈ 9.6%` — almost one in ten page loads has a five-second stall in it. Your average says you're great. Your users are watching a spinner. **The average is a number that is true and useless at the same time.**

The fix is to set latency SLOs on **percentiles**, which measure the tail directly instead of averaging it away. The p99 latency is the value such that 99% of requests are faster than it and 1% are slower — so a "p99 < 800ms" SLO is a promise that *at most* 1% of requests take longer than 800ms. That directly bounds the slow tail, which is the thing users actually feel. The relationship between percentiles and the user experience is what you want to reason about: p50 (the median) is the typical experience, p95 is the experience of your less-lucky users, p99 is the experience of your unluckiest one-in-a-hundred, and p99.9 is the experience during the worst moments. Each higher percentile costs more to hold (the tail is where the rare, expensive failures live, exactly like the rare failures behind each extra nine), so you pick the percentile that matches how much of the tail you're willing to leave unprotected.

| Latency statistic | What it tells you | Why it's right or wrong for an SLO |
| --- | --- | --- |
| Average (mean) | One number for the whole distribution | Wrong — a small slow tail is averaged away; hides p99 entirely |
| p50 (median) | The typical request's experience | Useful context, but ignores the tail that hurts |
| p95 | The experience of your unluckier 1-in-20 | Good SLO target for many services; bounds most of the felt slowness |
| p99 | The unluckiest 1-in-100 | Strong SLO target; catches the tail that average and p50 miss |
| p99.9 | The worst moments, 1-in-1000 | Use for critical paths; expensive to hold, very tight |

![A matrix comparing latency statistics average, p50, p95, and p99 across what they reveal and whether they hide the slow tail that users actually feel](/imgs/blogs/setting-slos-that-mean-something-6.png)

The practice for measuring percentiles is the Prometheus histogram. You instrument your handler with a histogram metric that buckets request durations, and then `histogram_quantile()` reconstructs the percentile from the buckets. Here is the recording rule for a p99 latency SLI over a rolling window:

```promql
# p99 request latency for the checkout service over a 5-minute window.
# Requires an http_request_duration_seconds histogram with le buckets.
histogram_quantile(
  0.99,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{job="checkout"}[5m])
  )
)
```

And here is the more SLO-shaped version: not "what is the p99?" but "what fraction of requests are faster than the threshold?" — which turns latency into a good/total ratio exactly like availability, so it composes with the error-budget machinery. If your latency SLO is "99% of requests under 800ms," the SLI is the ratio of requests in the sub-800ms buckets to all requests:

```promql
# Fraction of checkout requests completing under 800ms over 30 days.
# This is the latency SLI: good (fast) requests / total requests.
sum(rate(http_request_duration_seconds_bucket{job="checkout", le="0.8"}[30d]))
/
sum(rate(http_request_duration_seconds_count{job="checkout"}[30d]))
```

Note the threshold — `le="0.8"` for 800ms — has to be an actual bucket boundary in your histogram. This is a real practical constraint: **you can only set latency SLO thresholds at boundaries you instrumented.** If you want a p99 SLO at 800ms, you must have a histogram bucket at 0.8 seconds. Choosing your bucket boundaries to land on the thresholds users notice (the perception thresholds from §2) is part of designing a latency SLO that means something. Define "fast" at the threshold the user actually perceives — 800ms might be the line for a checkout confirmation, 100ms for an autocomplete, 2 seconds for a report — and instrument a bucket there.

A word on *why the threshold-based form is better than the raw quantile* for an SLO, because this trips people up. When you write `histogram_quantile(0.99, ...)` you get a number in seconds — "the p99 is 740ms right now" — which is great for a dashboard but awkward as an SLO, because the quantile estimate is noisy when traffic is low and it interpolates between buckets, so its precision is fake at the edges. The threshold form — "what fraction of requests were under 800ms?" — sidesteps all of that. It's a clean good-over-total ratio, it composes directly with the error-budget arithmetic (`budget = 1 − target`, just like availability), it's stable at low traffic because you're counting events rather than estimating a quantile, and it states the SLO the way users actually experience it: "almost all of my requests were fast enough." The two forms answer different questions — the quantile says *how slow is the tail*, the threshold-ratio says *how much of the tail crossed the line users care about* — and for an SLO you almost always want the second. Keep the quantile for your dashboards and incident debugging; write the SLO on the threshold ratio.

One more practical note on percentiles that catches teams: **percentiles do not average across services or time windows.** You cannot take the p99 of service A and the p99 of service B and average them to get "the p99." You cannot take yesterday's p99 and today's p99 and average them to get the two-day p99. Percentiles are properties of a distribution, and to combine them you have to combine the underlying distributions — which, with Prometheus histograms, means summing the *bucket counts* first and computing the quantile from the summed buckets, exactly as the recording rule above does with `sum by (le)`. This is why histograms are the right instrument and why naively-stored "p99 per minute" metrics are a trap: you can't aggregate them correctly after the fact. If your monitoring stores pre-computed percentiles instead of histogram buckets, your cross-service and cross-window SLO math is quietly wrong, and you won't find out until an audit. Store the buckets, aggregate the buckets, compute the quantile last.

## 6. Multi-tier SLOs: different journeys deserve different targets

A single service almost never has a single SLO, because a single service almost never does a single thing. Your application has many user journeys, and they are not equally important. The checkout flow losing requests costs you revenue and trust directly. The "recommended for you" widget on the sidebar failing costs you... a slightly emptier sidebar. Setting one SLO across all of it forces a bad compromise: either you hold the whole service to the checkout's strict bar (and burn money making the recommendation widget five-nines reliable for no reason), or you hold it to the widget's loose bar (and let checkout failures hide inside an aggregate that looks fine). **The fix is to tier your SLOs by the criticality of each journey.**

The principle is that the SLO should match the *cost of failure* for that specific journey. A journey whose failure costs revenue, safety, or irrecoverable trust gets a strict target. A journey whose failure is a minor cosmetic annoyance gets a loose one. This isn't just budget-saving; it's *correct*, because it focuses your reliability engineering where user pain actually lives. When the recommendation service is having a bad day, you genuinely do not want to page anyone at 3am, freeze releases, or spend budget — its 99% SLO has plenty of room and nobody is hurt. When checkout dips, you want every alarm in the building going off. Tiering encodes that judgment into numbers so the alerting and the release-freeze decisions follow automatically.

| Journey | Tier | Availability SLO | Latency SLO | Why this target |
| --- | --- | --- | --- | --- |
| Checkout / payment | Critical | 99.95% | p99 < 800ms | Direct revenue and trust; failure is felt and costs money |
| Login / auth | Critical | 99.95% | p95 < 300ms | Gateway to everything; a login outage blocks the whole product |
| Search / browse | Important | 99.9% | p95 < 500ms | Core to the experience but degrades gracefully (cached results) |
| User profile / settings | Standard | 99.5% | p95 < 1s | Used occasionally; a brief outage is a minor annoyance |
| Recommendation widget | Best-effort | 99% | p95 < 1.5s | Cosmetic; failure means an empty box, not a broken product |

![A matrix showing five user journeys tiered by criticality from checkout at 99.95 percent down to a recommendation widget at 99 percent each with its own availability and latency target](/imgs/blogs/setting-slos-that-mean-something-7.png)

There is a real engineering consequence to tiering beyond the numbers: it justifies — and demands — **graceful degradation**. If your recommendation widget has a best-effort 99% SLO, then when it's failing your product must *keep working without it*. The page renders, the recommendations box just shows nothing or a static fallback, and checkout is completely unaffected. That isolation is what lets you set a loose SLO on the widget honestly: a widget failure that took down checkout would not be a 99% problem, it would be a checkout-tier incident. So tiering and isolation go together — you can only set a loose SLO on a journey if its failure is genuinely contained. The bulkheads and fallbacks that make this possible are architecture-layer concerns ([graceful degradation in the system-design series](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) covers them), but the SLO tiering is what tells you *which* journeys need to be made independently failable, and how hard.

A clean way to express a multi-tier SLO set in config is a per-journey SLO spec. Here's a YAML SLO definition for the checkout journey with both an availability and a latency objective, a window, and the burn-rate alerting windows derived from it — the kind of artifact you'd check into your repo and feed to an SLO tool like Sloth or Pyrra:

```yaml
# slo/checkout.yaml — the SLO spec for the critical checkout journey.
service: checkout
tier: critical
objectives:
  - name: availability
    description: "Fraction of checkout requests that are not 5xx."
    target: 0.999            # 99.9% — three nines, ~43.2 min/month budget
    window: 30d
    sli:
      events:
        good: 'sum(rate(http_requests_total{job="checkout", code!~"5.."}[5m]))'
        total: 'sum(rate(http_requests_total{job="checkout"}[5m]))'
  - name: latency
    description: "Fraction of checkout requests completing under 800ms."
    target: 0.99             # 99% of requests under the threshold
    threshold: 0.8           # 800ms — a histogram bucket boundary
    window: 30d
    sli:
      events:
        good: 'sum(rate(http_request_duration_seconds_bucket{job="checkout", le="0.8"}[5m]))'
        total: 'sum(rate(http_request_duration_seconds_count{job="checkout"}[5m]))'
alerting:
  # Multi-window burn-rate alerts derived from the budget (1 - target).
  page:   { burn_rate: 14.4, long: 1h,  short: 5m }   # fast burn → page
  ticket: { burn_rate: 3,    long: 6h,  short: 30m }  # slow burn → ticket
```

That spec is the bridge between the *number you chose* and the *alerts that protect it*. The burn-rate alerting machinery — why a 14.4× burn over an hour pages and a 3× burn over six hours files a ticket — is the subject of the error-budget post; here the point is that the spec only makes sense once the target is chosen correctly, because every threshold in it is derived from `1 − target`.

## 7. The two ways an SLO fails: too strict and too loose

We've alluded to both failure modes; now let's name them precisely, because recognizing which one you've committed is half of fixing it. An SLO fails in exactly two directions, and they are mirror images.

**The too-strict SLO that everyone ignores.** This is the more insidious failure, because it looks like rigor. You set the SLO higher than the service can reliably hold — maybe above the dependency ceiling, maybe above achieved performance, maybe just past the user-perception threshold. Now the burn-rate alerts fire constantly, for badness that no user actually noticed. The on-call gets paged at 3am because the SLO dipped, investigates, and finds... nothing a user would care about, a transient blip well inside what the network swallows anyway. This happens a few times and the team does the only rational thing: **they mute the alert.** Maybe they raise the threshold informally, maybe they route it to a channel nobody watches, maybe they just learn to ack-and-ignore. And now you are in the worst possible state — *worse than having no SLO at all* — because the alert that's been crying wolf will also be ignored on the night it's real. A too-strict SLO doesn't produce reliability; it produces alert fatigue, and alert fatigue produces missed incidents. The cure is to bring the SLO down to a number that, when violated, genuinely corresponds to user pain — so that a page *means something* and the team trusts it.

**The too-loose SLO that never protects anyone.** The opposite failure is quieter and so it survives longer. You set the SLO so far below achieved performance that nothing ever trips it. The service could degrade significantly — a real regression users start to feel — and the SLO sits green the whole way down, because the slide didn't go far enough to cross the lazy line you drew. The error budget never runs low, so the burn-rate alerts never fire, so the release-freeze logic never engages, so the SLO provides exactly zero protection. It's a number on a dashboard that makes everyone feel covered while covering nothing. The cure is to tighten the SLO toward achieved performance until it actually bites when reliability degrades to the edge of acceptable.

| Failure mode | What it looks like | What it produces | The fix |
| --- | --- | --- | --- |
| Too strict | Fires for badness users don't notice | Alert fatigue, then a muted alert — worse than no SLO | Lower until a violation means real user pain |
| Too loose | Never fires even during real regressions | Zero protection; a number that decorates a dashboard | Tighten toward achieved performance until it bites |
| Above dependency ceiling | Mathematically impossible to hit | Permanent budget-out, team burnout | Lower the target or fix the architecture |
| Already crushing it | Budget never runs low | No signal; regressions hide below the line | Tighten to a notch below achieved |

The narrow band between these two failures is where a good SLO lives. It's set just below your reliable achieved performance, at or inside the user-perception threshold, under the dependency ceiling, and at a target where the next nine isn't worth its cost. Hit that band and the SLO does its job: it stays green during normal operation, it goes red when reliability degrades to where users start to hurt, and the team trusts a red as a real signal worth acting on.

### Stress-testing the choice

Before you commit a target, stress-test it the way you'd stress-test any decision. *What if a key dependency is down for two hours?* At 99.9% your monthly budget is 43.2 minutes — a two-hour dependency outage blows nearly three months of budget in one event, which tells you that either you need a fallback for that dependency or your SLO is implicitly assuming the dependency never has a bad day, which is a fantasy. *What if two incidents overlap?* Your budget is shared across all causes; two simultaneous incidents burn it twice as fast, so a target with no slack between achieved and SLO has no room to absorb a bad week. *What if the budget is already spent mid-month?* Then your SLO's whole point kicks in — you freeze risky releases until the window rolls over — and if that happens *every* month, your SLO is above achieved performance and must come down. *What if a region fails entirely?* A target that can only be met with all regions healthy isn't really a 99.9% target; it's a 99.9%-if-nothing-major-breaks target, which is not what you promised. Each of these "what ifs" either confirms your target has the right slack or reveals that it's secretly too strict for the reality your service lives in.

## 8. Choosing the number: the full worked example

Let's put all four inputs together on one real decision, because the inputs interact and the point is to watch them converge. You own the **checkout service**, and you need to set its availability SLO and its latency SLO.

**Input 1 — user perception.** Checkout is a deliberate, high-stakes action. The user has decided to buy; they will tolerate a second or so of "processing your order" because they expect a payment to take a moment, but a *failure* — a 500, a hang past a few seconds — costs you the sale and the trust. So users care a great deal about checkout *succeeding* and care moderately about it being *fast*. The perception threshold for latency here is generous: a p99 under about 800ms feels responsive for a confirmation step, and pushing it to 400ms would not measurably change conversion. So latency target: **p99 < 800ms**, and there's no user reward for going tighter.

**Input 2 — achieved performance.** You run the availability query from §3 over the last quarter and get **99.92%**, stable. That immediately rules out anything at or above 99.95% (you'd be out of budget) and anything at or below 99.8% (you'd never trip it on a real regression). The honest target is right around 99.9%.

**Input 3 — dependency ceiling.** Checkout calls auth (99.95%), the database (99.99%), and the payments gateway (99.9%) on the critical path. From §4's worked example, the ceiling is `0.9995 × 0.9999 × 0.999 ≈ 99.84%`. Wait — that's a problem. Your achieved 99.92% is *above* the 99.84% ceiling the chain should allow. How? Because in practice not every request touches every dependency at its worst moment, and some failures overlap rather than add — the real-world ceiling has some slack the naive product underestimates. But the calculation is still a warning: **99.84% is roughly where the structural ceiling sits, so a 99.95% SLO would be fighting the architecture, and even 99.9% leaves almost no room for your own faults.** This input pushes you to be conservative and not go above 99.9%, and it flags that if you ever want *more* than 99.9% you must decouple a dependency (make payments async, cache auth) rather than just trying harder.

**Input 4 — cost of the next nine.** Going from 99.9% to 99.99% on checkout would, from §1's pricing exercise, require automated sub-minute rollback, sub-30-second database failover, circuit breakers on every dependency, and pre-provisioned headroom — a quarter of senior engineering work plus standing infrastructure cost. The benefit: shrinking allowed downtime from 43.2 to 4.32 minutes a month. Is that worth it for checkout? It might be, *eventually*, for a large-revenue business — but not today, when you're at 99.92% and the dependency ceiling sits at 99.84%. The next nine isn't buyable yet; you'd have to fix the architecture first.

**The decision.** All four inputs converge on **99.9% availability, p99 < 800ms latency, 30-day window**. Achieved performance (99.92%) says you can hold it with a sliver of headroom. User perception says higher availability is invisible and 800ms latency is plenty fast. The dependency ceiling (99.84%) says don't go above 99.9% without re-architecting. The cost of the next nine says four nines isn't worth it today. Here's the SLO spec you'd write down and defend:

```yaml
# The defensible checkout SLO, with the reasoning encoded in comments.
service: checkout
tier: critical
window: 30d
objectives:
  - name: availability
    target: 0.999            # 99.9% — at achieved (99.92%) minus a sliver;
                             # below the ~99.84% naive dependency ceiling's
                             # real-world headroom; the next nine isn't
                             # buyable without re-architecting dependencies.
    budget_minutes_per_month: 43.2
  - name: latency
    target: 0.99             # 99% of requests under threshold
    threshold_seconds: 0.8   # p99 < 800ms — at the user-perception line for
                             # a checkout confirmation; tighter buys nothing.
review: quarterly            # re-read achieved performance and ratchet if earned
```

Notice that the comments *are* the negotiation. When product asks "why not 99.99%?", the answer is in the file: it's above the dependency ceiling, the next nine costs a quarter of engineering, and users can't perceive it. When leadership asks "why not 99.5%, it'd be easier?", the answer is also there: you're already at 99.92%, so 99.5% would never trip and would hide real regressions. The number isn't a guess; it's the intersection of four constraints, each of which you can show your work for.

## 9. Negotiating the SLO: it's a shared commitment, not an ops edict

The most important thing about an SLO target is something the arithmetic can't give you: **buy-in.** An SLO that ops imposes on a reluctant product team is a number on a wiki that gets overridden the first time a launch deadline collides with a release freeze. An SLO that product, engineering, and leadership *agreed to together* is a shared commitment that holds under pressure, because everyone in the room signed up for the tradeoff. So the final skill in setting an SLO is negotiating it, and the key is to reframe the conversation away from "how reliable should we be?" (which product always answers "totally reliable, obviously") toward the tradeoff that's actually being made.

The reframe is the **error budget.** Instead of asking "what's the SLO?", you tell stakeholders: *"This SLO gives you an error budget — `1 − SLO` — that you get to spend. A 99.9% SLO means 43.2 minutes a month where the service can be down without us having failed our commitment. That budget is what pays for shipping features fast: every risky deploy, every experiment, every new feature spends a little of it. A higher SLO gives you a smaller budget — less room to ship — and costs more engineering to hold. A lower SLO gives you more room to move fast but a worse user experience. Which trade do you want to make?"* Suddenly product is not choosing a virtue, they're choosing a budget they get to spend on the thing they care about — velocity. That's a conversation they can actually engage with, because it's framed in their currency.

The second move is to **make the cost of a higher target explicit.** When someone says "I want 99.99%," don't argue — price it. Pull up the §1 exercise: "99.99% means 4.32 minutes a month, which means a single bad deploy blows the whole budget, which means we need automated canary rollback, hot-standby failover, and circuit breakers on every dependency — that's roughly a quarter of the team for a quarter, plus ongoing infrastructure cost, and our dependency chain caps us at about 99.84% anyway so we'd have to re-architect three integrations first. Here's what that quarter of work would *not* get done instead." Now the higher target has a visible price tag and a visible opportunity cost, and the conversation becomes a real prioritization decision rather than a wish. Often, faced with the actual cost, the business decides 99.9% is plenty — which it usually is.

The third move is to get the agreement **written down and owned jointly.** The SLO spec lives in the repo, product and engineering both reviewed it, leadership signed off on the error-budget policy (what happens when the budget is spent — releases freeze, reliability work gets prioritized). When the budget runs out mid-quarter and someone wants to ship anyway, you're not an ops gatekeeper saying no; you're pointing at a policy everyone agreed to. The SLO has teeth precisely because it was negotiated, not imposed.

![A graph showing the SLO negotiation flowing from a too-high ask through reframing as an error budget and pricing the next nine to a jointly owned shared commitment](/imgs/blogs/setting-slos-that-mean-something-8.png)

### Achievable versus aspirational, and ratcheting up over time

There's a useful distinction to keep in the negotiation: the difference between an **achievable** SLO and an **aspirational** one. The achievable SLO is what you can hold *today*, given current achieved performance and architecture — that's the one you commit to, the one that drives alerting and release decisions, the one with teeth. The aspirational SLO is where you *want* to be — the better number you'll reach after a roadmap of reliability work. Keep them separate. If you commit to the aspirational number before you've earned it, you've just set a too-strict SLO that you'll miss every month (§7). Instead, commit to the achievable number, *and* write down the aspirational one with the roadmap to reach it.

Then you **ratchet.** As you do the reliability work — eliminate a class of failures, decouple a dependency, add automated rollback — your achieved performance climbs. Each quarter you re-run the achieved-performance query (§3), and when you've genuinely held a better number for a sustained period, you tighten the SLO to match. The SLO walks up behind your real reliability, never ahead of it. This is how you avoid both failures at once: you're never committed to a number above achieved (too strict), and you don't let the SLO lag so far behind achieved that it stops biting (too loose). Ratcheting turns the SLO from a fixed line into a moving floor that rises as you earn it — and crucially, each ratchet-up is a small re-negotiation: "we earned a nine, here's the new budget and what it cost; do we want to keep climbing or is this enough?" The SLO stays a living, shared commitment rather than a number someone set once and forgot.

## 10. The window matters as much as the number

There's an input to an SLO that people forget is even a choice: the **window** over which you measure it. "99.9% availability" is meaningless until you say *over what period* — 99.9% over a rolling 30 days is a very different commitment from 99.9% over a calendar quarter, and both differ from 99.9% measured per-day. The window controls how much a single bad event hurts you, how fast your budget recovers, and how the SLO feels day to day. Choosing it badly can turn an otherwise well-chosen target into a too-strict or too-loose one without changing a single nine.

Consider the two common shapes. A **rolling window** (say "the trailing 30 days, recomputed continuously") means your error budget is always measured over the last 30 days no matter what day it is. A bad incident hurts for exactly 30 days and then ages out — the day the incident falls off the back of the window, that budget comes back. This is smooth and forgiving: there's no artificial reset, no "the clock starts over on the first of the month," and a single outage doesn't dominate forever. A **calendar window** (say "this calendar month") resets the budget on a fixed boundary. It's easier to reason about for reporting ("how did we do in March?") and it's what SLAs usually use because contracts are billed monthly, but it has a sharp edge: an outage on the 1st and an identical outage on the 30th cost the same budget, yet the first leaves you nervous for the whole month while the second is forgiven in a day when the calendar flips. Calendar windows also create a perverse incentive near the boundary — "we're almost out of budget but it resets Tuesday, so let's just hold deploys two days and start fresh" — which is gaming, not reliability.

The window also interacts with the *length* of the window in a way that directly changes how strict the SLO feels. A shorter window makes the SLO more volatile: over a 7-day window, a single 43-minute incident is `43 / (7 × 1440) ≈ 0.43%` of badness, which blows a 99.9% (0.1% budget) target four times over for that week. Over a 30-day window the same 43-minute incident is `43 / 43,200 ≈ 0.1%` — right at budget. Over a 90-day window it's a third of budget. So **the same incident is a catastrophe, a tie, or a non-event purely depending on the window length.** Longer windows are more forgiving of individual incidents but slower to react to a sustained regression (it takes longer for a slow burn to consume a big budget); shorter windows react fast but are jumpy and punish single events harshly. The standard, sane default is a **rolling 30-day window** — long enough to absorb a single bad day without panic, short enough that a sustained regression burns visibly within a couple of weeks, and aligned with how people naturally think about a month of operation. Use a calendar month only when your SLA is billed that way and you need the internal SLO to match the contract's accounting.

#### Worked example: the same incident across three windows

You run at a 99.9% SLO and suffer one 60-minute incident this period. How much budget did it cost, and are you in trouble? It depends entirely on the window:

- **7-day rolling window.** Budget = `0.001 × 7 × 1440 = 10.08` minutes. A 60-minute incident is `60 / 10.08 ≈ 595%` of budget — you blew nearly six weeks' worth of a 7-day budget in one event. With this window, a single hour-long incident puts you deep in the red and freezes releases for days. Probably too strict a window for most services.
- **30-day rolling window.** Budget = `0.001 × 30 × 1440 = 43.2` minutes. A 60-minute incident is `60 / 43.2 ≈ 139%` of budget — you're over for the month, but by a recoverable amount; in ~13 days the incident ages out of the window and you're back under. Sane: one bad incident matters, but doesn't end the world.
- **90-day rolling window.** Budget = `0.001 × 90 × 1440 = 129.6` minutes. A 60-minute incident is `60 / 129.6 ≈ 46%` of budget — barely a dent. With this window you could absorb two such incidents and still be in budget, which is forgiving but slow to flag a service that's quietly getting worse.

Same nine, same incident, three completely different stories. The window is not a footnote on the SLO; it's part of the SLO, and you choose it with the same deliberateness you choose the number — usually landing on 30 days unless a billing cycle or a specific risk profile argues otherwise.

## 11. War story: the SLO that paged 200 times a week, and the one that protected nothing

Two real-shaped failures, both of which I've watched happen, to make the two directions concrete.

**The too-strict SLO.** A platform team set a 99.99% availability SLO on an internal API gateway because four nines "sounded like the bar a platform should hold." The gateway's achieved performance was around 99.9%, and its dependency chain (it fronted a dozen backend services) capped it well below 99.99% anyway. The result was exactly what the arithmetic predicts: the burn-rate alerts fired constantly. The 4.32-minutes-a-month budget got eaten by routine deploy blips, autoscaling lag, and dependency hiccups that no consumer of the gateway ever noticed — internal services retry transient failures anyway, so a 200ms blip on the gateway was invisible to everyone downstream. The on-call was getting paged 30 to 40 times a week for SLO burn that corresponded to *zero* reported user pain. Within two months the team had quietly routed the burn-rate alerts to a Slack channel they muted. The SLO was now decorative, and — this is the dangerous part — *the alert that would fire during a real outage was the same muted alert.* They had engineered themselves into a state worse than no SLO. The fix was to bring the target down to 99.9% (matching achieved performance and the dependency reality), at which point the alerts dropped from ~35/week to about 4/week, every one of which corresponded to something worth looking at, and the on-call started trusting the page again. **Alert volume 35/week → 4/week, and the alerts that remained were real.** That is the signature of a target that was too strict, corrected.

**The too-loose SLO.** A different team owned a search service and, scarred by a previous over-strict experience, set the SLO at 99% — comfortably below their achieved 99.95%. For a year it sat green and silent. Then a gradual regression set in: a slow memory leak in a downstream index node degraded p99 latency and pushed availability down to 99.6% over the course of six weeks. Real users were feeling it — search felt sluggish, occasional results failed — and support tickets ticked up. But the SLO was *99%*, and 99.6% is comfortably inside 99%, so the SLO stayed green the entire time, the burn-rate alerts never fired, and nobody connected the rising support tickets to a reliability regression until a customer escalated. A 99% SLO on a service that runs at 99.95% is not a tripwire; it's a number that can only fire during a catastrophe, which means it provides no protection against the slow, ordinary regressions that are how most services actually degrade. The fix was to tighten the SLO to 99.9% — a notch below achieved — at which point the *next* slow regression would burn budget visibly within days and trip an alert while it was still small. **The lesson: a loose SLO doesn't protect you from the failures you actually get, which are gradual, not catastrophic.**

The two stories are the same lesson from opposite sides: the SLO has to sit in the narrow band where a violation means real user pain *and* a normal day stays green. Too high and you mute it. Too low and it's blind to the regressions you actually suffer. The four inputs — perception, achieved performance, dependency ceiling, cost of the next nine — exist precisely to land you in that band.

The Google SRE model that originated this whole discipline frames it the same way: the SLO target is chosen so that the error budget it produces is *spendable* — enough room to ship features at a healthy pace, tight enough that running out means you genuinely should slow down and invest in reliability. If your budget never runs out, your SLO is too loose; if it's always out, your SLO is too strict or your service is genuinely under-reliable. The target is calibrated to make the budget a useful currency, which is the subject of the [error-budget post](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). And the SLI you draw the SLO on has to reflect real user pain in the first place — a target on a metric users don't feel is pointless no matter how well you choose the number, which is why [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) comes right before this.

## 12. How to reach for this (and when not to)

Setting an SLO target deliberately, from the four inputs, is worth the effort for any service whose reliability someone actually cares about. But like every practice in this series, it has a cost and a wrong place to apply it. Here's the decisive guidance.

**Do set a deliberate SLO when** the service is user-facing or business-critical, when reliability decisions (release freezes, on-call priority, where to invest) need a shared basis, and when you have enough traffic to measure an SLI honestly (an SLI computed from a handful of requests a day is statistical noise). For these, walk the four inputs: read the nines table, measure achieved performance, find the perception threshold, compute the dependency ceiling, price the next nine, and negotiate the number with stakeholders.

**Don't over-invest in SLO precision when** the service is a low-traffic internal tool, a batch job, or a best-effort background process. A 99.999% SLO on an internal nightly batch job is absurd — it runs once a day, nobody is staring at it, and the cost of holding five nines on it is pure waste. Set a loose, honest target (or a simple "it ran successfully" success-rate SLO over a long window) and move on. Don't bring the full apparatus to a service where the cost of failure is "someone re-runs it in the morning."

**Don't add a nine users can't perceive.** This is the opening story's lesson and it's worth repeating as a rule: if your service is already more reliable than the user's own network and device, the next nine is invisible. Spend that engineering effort somewhere the user can feel it — on a different service, on latency, on a feature — rather than on a nine that the user's WiFi erases before they ever see it.

**Don't set an SLO above your dependency ceiling and call it a goal.** If the chain caps you at 99.84%, a 99.9% SLO isn't ambitious, it's impossible, and committing to it just burns the team out chasing physics. Either fix the architecture (decouple a dependency) or set the target where the chain actually allows. The dependency-chain calc is a five-minute sanity check that saves quarters of wasted effort.

**Don't impose an SLO without buy-in.** A target nobody agreed to won't survive its first collision with a launch deadline. The negotiation isn't bureaucratic overhead; it's what gives the SLO teeth. If you skip it, you've written a number, not made a commitment.

## 13. Key takeaways

- **An SLO target is a business decision disguised as a number.** It's a cost/benefit tradeoff, not a virtue to maximize. "As reliable as possible" is a budget hole, not a goal.
- **Know the nines table cold:** 99% is 7.2 h/month, 99.9% is 43.2 min/month, 99.99% is 4.32 min/month, 99.999% is 26 sec/month. Hear percentages as minutes of allowed downtime.
- **Each added nine costs roughly 10× the previous one.** The cost of reliability is exponential in the number of nines, because each nine forces you to eliminate rarer, more expensive failure modes. Buy a nine only when its value exceeds its price.
- **Choose the number from four inputs:** what users can perceive (don't add nines past their own network's noise), your achieved performance (don't set an SLO you're already missing or already crushing), the cost of the next nine versus its value, and the dependency-chain ceiling.
- **Your SLO can't exceed your dependencies' combined reliability.** Multiply the availabilities of serial critical-path dependencies; that product is your ceiling. Three 99.9% dependencies cap you at ~99.7%. If your target is above the ceiling, fix the architecture or lower the target.
- **Set latency SLOs on percentiles, never averages.** An average hides the slow tail that users feel — a 200ms average can conceal a 5s p99 hurting 1% of requests. Use p95/p99 and define "fast" at the threshold users perceive.
- **Tier your SLOs by criticality.** Checkout deserves 99.95%; the recommendation widget deserves 99% and graceful degradation. One SLO across all journeys forces a bad compromise.
- **An SLO fails in two directions.** Too strict pages for badness users don't notice until the team mutes it (worse than no SLO); too loose never trips and protects nothing. The right target sits in the narrow band where a violation means real user pain and a normal day stays green.
- **Negotiate the SLO as a shared commitment.** Reframe it as the error budget product gets to spend, make the cost of a higher target explicit, and get it owned jointly. An imposed SLO has no teeth.
- **The window is part of the SLO.** The same incident is a catastrophe over a 7-day window, a tie over 30 days, and a non-event over 90. Default to a rolling 30-day window; use a calendar month only to match an SLA's billing cycle.
- **Commit to the achievable target, write down the aspirational one, and ratchet up as you earn each nine.** The SLO should walk up behind your real reliability — never ahead of it, never so far behind it stops biting.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro: reliability is engineered and budgeted, not hoped for. The frame this whole post sits inside.
- [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter) — the precise definitions this post builds on; what an SLO *is* before we chose what number to make it.
- [Choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) — a target is only as good as the indicator it's drawn on; this covers picking the right SLI first.
- [The error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — what `1 − SLO` buys you and how it aligns dev and ops; the budget your chosen target produces.
- [Reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — the architecture-layer view: how to *design* for the SLOs you set, including the graceful degradation that makes tiering honest.
- [Capacity planning and autoscaling](/blog/software-development/system-design/capacity-planning-and-autoscaling) — the headroom and pre-provisioning that buying an extra nine often requires; the infrastructure cost behind the cost-of-the-next-nine.
- *Site Reliability Engineering* (Google, O'Reilly), Chapter 4, "Service Level Objectives" — the canonical treatment of choosing targets and the error-budget model.
- *The Site Reliability Workbook* (Google, O'Reilly), Chapters 2 and 5 — implementing SLOs and the multi-window multi-burn-rate alerting derived from them.

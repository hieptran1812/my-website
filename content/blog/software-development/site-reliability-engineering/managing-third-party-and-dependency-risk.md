---
title: "Managing Third-Party and Dependency Risk: Don't Let Someone Else's Outage Become Yours"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Your reliability is capped by the product of your hard dependencies' reliability, and some of those are vendors you can't fix, can't see into, and can't page. Learn the dependency-chain math, how to convert hard dependencies to soft ones, and how to design so a payment processor's outage becomes a degraded checkout instead of yours."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "dependency-risk",
    "third-party",
    "vendor-management",
    "circuit-breakers",
    "graceful-degradation",
    "multi-vendor",
    "supply-chain",
    "availability",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/managing-third-party-and-dependency-risk-1.png"
---

At 14:11 on a Tuesday, our checkout conversion rate fell off a cliff. Not a gentle slope, a cliff: the graph went from a steady 3.1% to 0.0% in the span of one scrape interval. Our service was healthy. CPU was flat, memory was flat, our error rate on every endpoint we owned was a clean zero. The dashboards we had built to watch ourselves all glowed green. And yet not a single customer could complete a purchase, because the one thing we did not own, the payment processor, had gone down. We had no pager into them. We had a status page that said "investigating" and a Twitter feed full of other companies discovering the same thing at the same moment. For the next 113 minutes there was nothing for the engineers on the bridge to *fix*. We could only watch our revenue graph sit at the bottom of the chart and wait for someone else's incident to end.

That is the particular helplessness of dependency risk, and it is worth naming early because it changes how you have to think. Most of the SRE craft is about making *your* systems better: better SLOs, better alerts, faster rollbacks, calmer on-call. But a large fraction of your real-world outages will not originate in your code at all. They will arrive from outside, through a vendor you pay, a cloud service you assumed was always there, or a transitive dependency you did not even know you had. You cannot make those things reliable. You did not write them, you cannot deploy a fix, and the person who can is not going to answer your page. The uncomfortable truth is that **your reliability is capped by your dependencies' reliability**, and you do not control most of the factors that set that cap.

This post is about the one thing you *can* control. You cannot make a third party reliable, but you can design so that their outage does not become yours. That is the entire thesis, and it is more achievable than the helplessness suggests. The goal is not to never depend on anyone, which is impossible, nor to negotiate a perfect contract, which buys you a refund and not uptime. The goal is to architect your service so that when, not if, a dependency fails, the failure degrades you instead of destroying you: a checkout that falls back to a second processor instead of going to zero, a login that serves a cached token instead of erroring, a feature that quietly disappears instead of taking the whole page down with it. We will work through the dependency-chain math that sets your ceiling (figure 1 shows it), the crucial distinction between hard and soft dependencies, the techniques to convert one into the other, how to hold vendors to an SLO while designing as if they will miss it, and two worked examples with real arithmetic, including the redesign of that very payment outage. This sits squarely in the SRE loop: it is how you spend your error budget wisely by not letting a single vendor spend it all for you.

![Diagram showing how five hard external dependencies each at three nines multiply down to cap a service near ninety-nine point five percent availability](/imgs/blogs/managing-third-party-and-dependency-risk-1.png)

By the end you will be able to draw your own dependency inventory, compute your availability ceiling from it, identify which dependencies must stay hard and which you can soften, and write the circuit-breaker, cache, and failover patterns that turn a vendor's bad day into a footnote in your own postmortem rather than the headline.

## 1. The dependency-chain math: why every hard dependency lowers your ceiling

Let me start with the arithmetic, because it is the load-bearing idea of the whole post and it is genuinely simple once you see it. Availability is usually quoted as a fraction of time a system is up: 99.9% ("three nines") means the system is up 99.9% of the time, which works out to roughly 43.8 minutes of downtime per month, or about 8.77 hours per year. We covered how to choose and defend those numbers in [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something); here we ask a different question: given the availabilities of the things you depend on, what is the best availability *you* can ever achieve?

A **hard dependency** is one whose failure fails your request. If your checkout request cannot complete without calling the payment processor, then any time the processor is down, your checkout is down too. Mathematically, for your request to succeed, you need yourself *and* every hard dependency to be up simultaneously. If failures are independent, the probability that they are all up at once is the **product** of their individual availabilities:

$$A_{\text{you}} = A_{\text{own}} \times \prod_{i=1}^{n} A_{\text{dep}_i}$$

This is the part that surprises people. Multiplication of numbers below 1 always shrinks the result, and it shrinks faster than your intuition expects. Suppose your own code is essentially perfect, 99.99%, and you have five hard dependencies each at a respectable 99.9%. Your ceiling is:

$$0.9999 \times (0.999)^5 = 0.9999 \times 0.99501 \approx 0.9949$$

So roughly **99.5%** — about 3.6 hours of downtime per month — even though every single component is at three nines or better and your own code is at four nines. You did not write a single bug. You simply assembled five reliable things into a chain, and the chain is weaker than any link in it. Figure 1 at the top draws exactly this: five external boxes at 99.9% feeding into a ceiling box at 99.5%.

The general rule falls right out of the formula: **every hard dependency you add lowers your ceiling.** Each one multiplies in another factor below 1. A sixth dependency at 99.9% drops you to ~99.4%; a tenth would put you near 99.0%. There is no amount of internal engineering excellence that escapes this. You can have flawless code, instant rollbacks, and a perfect on-call rotation, and your availability will still be bounded above by the product of the things you cannot control.

This has one blunt implication that should govern your architecture: **minimize the number of hard dependencies on the critical path.** The critical path is the set of operations that must succeed for the core user journey to work — for an e-commerce site, "can the user buy the thing." Every service, database, vendor API, and external system that sits on that path and can fail the request is a multiplicative tax on your ceiling. The cheapest reliability win available to most teams is not adding redundancy; it is *removing* a hard dependency from the critical path, either by deleting it or by converting it to soft (we get to how in section 3).

#### Worked example: the six-dependency ceiling

Take a real shape. You run a service whose request flow, on the critical path, calls six third parties: an auth provider, a feature-flag service, a payment processor, a fraud-scoring API, a tax-calculation API, and an address-validation API. Each advertises 99.9% availability. Your own code, generously, is 99.99%.

Your ceiling:

$$0.9999 \times (0.999)^6 = 0.9999 \times 0.99403 \approx 0.9939$$

That is **99.4%**, or about 52.6 hours of unavailability per year that you literally cannot prevent by improving your own service. If your SLO target is 99.9% (8.77 h/yr), you have already lost the bet before you wrote a line of code, because your dependency stack alone consumes six times your entire annual budget. No deploy freeze, no extra replica, no faster rollback closes that gap. The only fix is structural: reduce how many of those six are hard. We will finish this example in section 9 by softening three of them and watching the ceiling rise.

The lesson the math teaches, stated as a principle: **count your hard dependencies before you promise an SLO.** If a product manager wants four nines and your critical path crosses eight hard third parties at three nines each, the SLO is arithmetically impossible and the honest answer is to renegotiate either the SLO or the dependency graph. Reliability is a number you engineer, and the dependency product is the first term in the equation.

A small table makes the nines tangible, because the abstract percentages hide how steep the cost of each extra nine actually is, and how quickly a chain of dependencies eats them:

| Availability | Downtime per year | Downtime per month | Downtime per week |
| --- | --- | --- | --- |
| 99% (two nines) | 3.65 days | 7.31 hours | 1.68 hours |
| 99.9% (three nines) | 8.77 hours | 43.8 minutes | 10.1 minutes |
| 99.95% | 4.38 hours | 21.9 minutes | 5.04 minutes |
| 99.99% (four nines) | 52.6 minutes | 4.38 minutes | 1.01 minutes |
| 99.999% (five nines) | 5.26 minutes | 26.3 seconds | 6.05 seconds |

Look at the jump from three nines to four: you go from 8.77 hours of allowed downtime a year to under an hour. Now recall that *one* hard dependency at three nines spends all 8.77 of those hours by itself. The table and the product formula together explain why teams that casually promise four nines while standing on a stack of three-nines vendors are promising something the arithmetic forbids.

One honest caveat about the multiplication: it assumes dependency failures are **independent**, and in the real world they are not always. If two of your dependencies both run in the same cloud region, a region outage takes both down at once — they are *correlated*, and the simple product understates your risk during such an event (the joint failure is more likely than independence would predict). Conversely, dependencies that fail truly independently are exactly what makes redundancy work, which is why section 9 insists that a second vendor must be *genuinely* independent of the first. So treat the product as a clean upper-bound estimate for planning — it tells you the ceiling you cannot exceed — while remembering that correlated failures can push your *actual* availability below even that pessimistic-looking number. The math is a floor for your pessimism, not a ceiling for it.

## 2. Hard versus soft: the classification that decides everything

The single most important act in managing dependency risk is classifying each dependency as hard or soft, because that label determines whether it counts against your ceiling and how you must treat it. Figure 2 lays the distinction out as a small matrix.

![Matrix comparing hard and soft dependencies across whether they fail the request and how each is survived](/imgs/blogs/managing-third-party-and-dependency-risk-2.png)

A **hard dependency** is one whose failure fails your request. No auth, no login; no primary database, no read or write; no payment processor, no checkout. These are the dependencies that multiply into your ceiling, and they are non-negotiable in the sense that the user journey genuinely cannot complete without them.

A **soft dependency** is one you can survive without. The recommendation API is soft: if it is down, you show a generic "popular items" list or hide the carousel, and the user still buys. The avatar service is soft: a default silhouette is a perfectly acceptable substitute for a profile picture. The analytics sink is soft: if your event pipeline is down, you buffer or drop events, and not one user notices. A soft dependency's failure costs you a feature, a little quality, or some internal data — but not the request.

Here is the principle that makes this classification the center of gravity for the whole topic: **a soft dependency does not multiply into your availability ceiling.** Because you survive its outage, its uptime does not appear in the product formula from section 1. If you can move a dependency from hard to soft, you remove its factor from the equation entirely. That is why the chain math and the classification are two sides of one coin: hard dependencies set your ceiling, and reclassifying them is how you raise it.

Two cautions before we go further. First, *hard and soft are properties of how you use a dependency, not of the dependency itself.* The same recommendation API is hard if your homepage renders a 500 when it times out, and soft if your homepage renders fine without it. The label describes your code's behavior under the dependency's failure, which means you can change the label by changing your code. Second, *people misclassify constantly, almost always in the optimistic direction.* Everyone agrees the recommendation widget is "not critical" — until the homepage controller awaits its response synchronously with no timeout, and a slow recommendation API turns into a slow homepage, which turns into exhausted request threads, which turns into a fully-down site. On paper it was soft. In production it was hard, because nobody made the code actually survive its absence. The gap between the intended classification and the enforced one is where most dependency-driven outages live.

| Property | Hard dependency | Soft dependency |
| --- | --- | --- |
| Failure fails your request? | Yes | No |
| Counts against availability ceiling? | Yes (multiplies in) | No |
| Examples | Auth, primary DB, payment on checkout | Recommendations, avatars, analytics |
| Survival technique | Cache, replica, second vendor, defer | Hide feature, default value, fire-and-forget |
| Goal | Reduce the count; convert to soft | Keep soft (verify it really degrades) |

The takeaway is a habit: for every external call on a request path, ask out loud, "if this returns an error or hangs for 30 seconds, what does the user see?" If the answer is "an error page" or "a spinner forever," it is hard whether you wanted it to be or not. If the answer is "a slightly worse but working page," it is genuinely soft. Write the answer down for every dependency; that document is your inventory, and we build it properly in section 6.

## 3. The central technique: converting hard dependencies to soft

If reducing hard-dependency count is the goal and you cannot always delete a dependency, the workhorse technique is **conversion**: take something that is hard and engineer it into something soft. There are three reliable ways to do it, and most real systems use all three in different places. This is the operational sibling of [graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — that post covers the degradation philosophy across your whole service; here we focus on the specific maneuver of softening a dependency. Figure 3 contrasts the hard and the softened call paths.

![Before and after diagram showing a hard direct vendor call that fails versus a softened call with a breaker timeout and cache fallback that degrades](/imgs/blogs/managing-third-party-and-dependency-risk-3.png)

**Technique one: cache its data so you survive its outage.** Many "hard" dependencies are read-mostly. Your auth provider issues tokens that are valid for an hour; your feature-flag service returns a config that changes a few times a day; your pricing service returns prices that move slowly. If you cache the last successful response and you are willing to serve it when the live call fails, the dependency becomes soft for the duration of its outage. The user gets a slightly stale answer instead of an error. The trade-off is staleness, and you have to decide per dependency how stale is acceptable. A feature flag that is ten minutes stale is fine; a fraud score that is a day stale may not be. The point is that *staleness is almost always better than failure* for read paths, and caching is the lever that lets you choose it.

**Technique two: make the call async or fire-and-forget.** If the dependency does not need to be on the critical path synchronously, take it off. The analytics event does not need to block the user's purchase; emit it to a local buffer or a queue and let a background worker ship it. The "send a welcome email" call does not need to complete before you return "account created"; enqueue it. By moving the call off the synchronous request path, its failure can no longer fail the request. This pattern leans on durable queuing — if you want the deferred work to actually happen, you need an at-least-once queue with retry, which is its own topic; the message-queue series covers delivery guarantees and backpressure in depth, and you should link your async fallbacks to a real queue rather than a fire-and-forget that silently drops on failure.

**Technique three: provide a fallback or default.** When you cannot cache and cannot defer, you can often still supply a reasonable default. The recommendation API is down, so you serve a static "bestsellers" list compiled weekly. The personalization service times out, so you render the generic homepage. The currency-conversion API fails, so you show prices in the base currency with a note. A default is a worse answer, but a worse answer keeps the request alive. The discipline is to *design the default deliberately* rather than discovering at 3am that the absence of a default means a null-pointer exception.

The harder, more valuable habit underneath all three is this: **ruthlessly question whether a hard dependency on the critical path needs to be hard.** Engineers reach for synchronous, blocking, must-succeed calls by default because they are the simplest to write. But "I need this data to render the page" is rarely as absolute as it sounds. Do you need the *live* fraud score, or is a cached one from five minutes ago acceptable for 99% of orders, holding only the suspicious ones for a live check? Do you need the recommendation *now*, or can the page render and the carousel fill in via a later asynchronous fetch that fails silently? Every time you successfully challenge a "hard" assumption, you remove a factor from your availability product. This is the highest-leverage reliability work most teams never do, because it requires saying "this feature is allowed to be slightly broken sometimes," which is a product decision as much as an engineering one.

Here is a compact fallback chain in code, the pattern you reach for when softening a read dependency. It tries the live call with a tight timeout, falls back to a cached value, and finally to a static default — three levels, each strictly better than an error:

```python
import time
from functools import lru_cache

CACHE_TTL_SECONDS = 600        # serve cache up to 10 min old on failure
STALE_OK_SECONDS = 86400       # in an outage, serve up to 1 day old

_last_good = {}                # key -> (value, timestamp)

def get_recommendations(user_id, live_client):
    key = f"recs:{user_id}"
    try:
        # 1. live call, tight timeout so a slow vendor can't hang us
        value = live_client.fetch(user_id, timeout_ms=200)
        _last_good[key] = (value, time.time())
        return value, "live"
    except (TimeoutError, ConnectionError):
        # 2. last-known-good cache, even if a bit stale during an outage
        cached = _last_good.get(key)
        if cached:
            value, ts = cached
            if time.time() - ts < STALE_OK_SECONDS:
                return value, "cache-stale"
        # 3. static default: a worse answer, but a working page
        return STATIC_BESTSELLERS, "default"
```

Notice what this does to the classification: the recommendation call started hard (a timeout would have failed the page) and is now soft (the page always gets *some* answer). It no longer appears in your availability product. That is the conversion, made concrete.

## 4. Third-party and vendor risk: the outage you literally cannot fix

So far the dependencies could have been your own internal services. Now narrow to the case that gives this post its name: **third-party vendors and cloud services**, the dependencies you do not operate and cannot fix. These deserve their own treatment because the usual SRE reflexes — page the owner, deploy a fix, roll back the bad change — are all unavailable to you. The owner is a company you have a contract with, not a teammate; the fix is on their roadmap, not yours; and there is nothing to roll back because you did not deploy anything.

Think about the canonical events. In October 2016, a large distributed-denial-of-service attack against Dyn, a managed DNS provider, made a long list of major sites unreachable for hours, even though those sites' own infrastructure was perfectly healthy — they simply could not be *resolved*. In February 2017, an Amazon S3 outage in the US-EAST-1 region took down a stunning breadth of the internet, including services that did not even realize they used S3, because so many other AWS services and third-party tools depended on it transitively. In July 2024, a faulty content update from a single security vendor caused widespread Windows host failures across airlines, banks, and hospitals worldwide — an outage that originated entirely outside the affected organizations' own engineering. The common thread is that the victims could do nothing about the root cause. Their reliability was, for those hours, entirely in someone else's hands.

What you actually have during a vendor outage is thin: their **status page** (often updated slowly, and sometimes hosted on the same infrastructure that is down), a support ticket queue, and whatever resilience you built *before* the incident. That last item is the only one you control, which is the whole reason this post exists. The vendor's reliability is an input you cannot change; your response to their failure is the only variable.

A few hard-won attitudes about vendor risk:

- **Their status page is not your monitoring.** Status pages lag, sometimes by tens of minutes, and occasionally never acknowledge an incident your users are clearly experiencing. Monitor the vendor *yourself* by tracking the success rate and latency of your own calls to them. You will frequently know they are degraded before their status page does, and that head start is worth real minutes of MTTR.
- **"It's the vendor's fault" does not help your users.** When checkout is down, your customers do not care whose fault it is. The outage is yours from the user's perspective, the support load is yours, and the trust damage is yours. Attribution is for the postmortem; resilience is for right now.
- **You will use a vendor you forgot you had.** The S3 outage taught everyone that you depend on far more third parties than your architecture diagram shows, because your vendors depend on vendors. Section 7 is about discovering those.
- **Concentration risk is real.** When everyone uses the same cloud region, the same DNS provider, or the same auth SaaS, a single vendor outage becomes a correlated, internet-wide event. Your "independent" failure assumption from the section-1 math quietly breaks, because your dependency and your competitor's dependency are the same box.

The practical consequence is that vendor calls deserve your most defensive engineering. They are the dependencies most likely to fail in ways you cannot fix, for durations you cannot predict, with no warning you control. Treat every synchronous call across your trust boundary as a potential multi-hour hang, and build accordingly. The next section is the how.

## 5. Graceful handling of a down provider: timeout, break, cache, fail over

When a provider goes down, there is a specific sequence of defenses that, layered together, turn the outage into a degraded mode. Figure 4 shows the breaker-wrapped call path; figure 5 shows the multi-vendor failover that sits behind it for the truly critical calls.

![Graph showing a request passing through a circuit breaker that either calls the vendor with a short timeout and refreshes the cache or fast-fails to a cached fallback](/imgs/blogs/managing-third-party-and-dependency-risk-4.png)

**Timeout fast.** The first and most important defense is a tight, explicit timeout on every external call. The default in most HTTP clients is effectively infinite or absurdly long (30, 60, even 120 seconds). A vendor that is *down* often fails fast with a connection refused, which is fine; a vendor that is *degraded* is far more dangerous, because it accepts your connection and then hangs. Without a timeout, your request thread is now stuck for as long as the vendor wants to hold it. Multiply that by your traffic and you exhaust your thread pool or connection pool, at which point *every* request — even ones that do not touch the vendor — starts queueing and failing. A slow dependency, untimed, is how a single soft-looking vendor takes down your entire service. Set the timeout to a small multiple of the vendor's normal p99 latency, not to some round number you picked out of comfort.

**Circuit-break the vendor call.** A timeout protects one request; a circuit breaker protects all of them. The breaker watches the failure rate of calls to a given dependency, and when failures cross a threshold, it "opens" — meaning subsequent calls fail *immediately* without even attempting the vendor, for a cooldown period. This does two things. It stops you from wasting thread-time and timeout-budget hammering a dead vendor, and it gives the vendor room to recover instead of being pounded by your retries the moment they come back. After the cooldown, the breaker goes "half-open," lets a trial request through, and either closes (vendor recovered) or re-opens (still down). We cover the mechanics, the bulkhead pattern that isolates one dependency's pool from another, and load shedding in [circuit breakers, bulkheads, and load shedding](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding) and at the architecture level in [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right). The key point for dependency risk: the breaker is what converts "vendor is slow" from a thread-pool-exhausting catastrophe into a fast, contained fast-fail that triggers your fallback.

Here is a breaker-plus-cache configuration in the shape resilience4j uses, wrapping a vendor call so its slowness cannot exhaust your threads:

```yaml
resilience4j:
  circuitbreaker:
    instances:
      paymentVendor:
        slidingWindowType: COUNT_BASED
        slidingWindowSize: 50              # last 50 calls
        failureRateThreshold: 50           # open at 50% failures
        slowCallDurationThreshold: 300ms   # a >300ms call counts as "slow"
        slowCallRateThreshold: 80          # open if 80% are slow
        waitDurationInOpenState: 10s       # cooldown before half-open
        permittedNumberOfCallsInHalfOpenState: 3
  timelimiter:
    instances:
      paymentVendor:
        timeoutDuration: 250ms             # hard cap; cancel after 250ms
        cancelRunningFuture: true
  bulkhead:
    instances:
      paymentVendor:
        maxConcurrentCalls: 20             # cap threads this vendor can hold
```

The `slowCallDurationThreshold` and `bulkhead.maxConcurrentCalls` lines are the unsung heroes. They mean a *degraded* vendor — slow, not dead — also trips the breaker and can never tie up more than 20 of your threads. That is the difference between absorbing a vendor's gray failure and being dragged into it.

**Cache last-known-good.** We met this in section 3; in the provider-down context it is the most common fallback target. When the breaker is open, you serve the last successful response. For read-mostly vendor data — config, prices, tokens, catalogs — this single technique converts most vendor outages into invisible degradation.

**Fall back to a degraded mode or a secondary provider.** For the truly critical calls where stale data will not do — taking a payment, sending a transactional message, serving an asset — the strongest defense is **multi-vendor**: a second provider you can fail over to. A backup payment processor, a secondary email/SMS provider, a multi-CDN setup with two independent content networks. When the primary trips its breaker, the router sends traffic to the secondary. Figure 5 sketches this for payments, including the queue-and-retry path for the rare case where *both* are down.

![Graph showing a checkout request routed by a health-checked payment router to a primary processor a secondary processor or a deferred retry queue](/imgs/blogs/managing-third-party-and-dependency-risk-5.png)

**Queue and retry later for deferrable writes.** Some operations do not need to happen *now*, only *eventually*. If both payment processors were down (extraordinarily rare if they are truly independent), you can accept the order optimistically, queue the charge, and retry it when a processor recovers — telling the user "your order is confirmed, payment is processing." That trades a small risk (a charge that ultimately fails and an order you have to cancel) for keeping the storefront functional through a total payment outage. Whether that trade is acceptable is a business decision, but the *capability* to make it is an engineering one.

**Fail the feature soft if it is non-critical.** And for the soft dependencies, the answer is the simplest of all: just turn the feature off. Hide the recommendation carousel, drop the analytics events, show the default avatar. The page renders, the user proceeds, and the only cost is a missing non-essential feature.

| Failover strategy | RTO (time to recover) | Cost / complexity | When it's worth it |
| --- | --- | --- | --- |
| Cache last-known-good | Instant (already in memory) | Low | Read-mostly data; staleness tolerable |
| Static default / hide feature | Instant | Very low | Soft dependencies, non-critical UI |
| Queue and retry later | Minutes to hours (deferred) | Medium (need durable queue) | Deferrable writes; email, analytics |
| Automatic secondary vendor | Seconds (health-check failover) | High (integrate + test 2 vendors) | Critical path; revenue-bearing calls |
| Multi-region / multi-cloud | Minutes (DNS / LB failover) | Very high | Cloud-provider-level resilience only |

The decisive principle: **match the failover strategy to the dependency's criticality, and no higher.** A second payment processor is worth its integration and reconciliation cost because payments are revenue. A second avatar service is absurd; just show a default. Most teams either under-invest everywhere (no fallbacks at all) or, having been burned once, over-invest everywhere (multi-vendor for things nobody would miss). The inventory in the next section is what lets you spend the resilience budget where it actually matters.

Now stress-test the design, because a fallback that only works in the easy case is not a fallback. Walk the hard scenarios:

- **What if the dependency is down for two hours, not two minutes?** A breaker plus cache handles a brief blip, but stale data ages. For a price cache, two-hour-old prices may be fine; for a fraud score, two hours may be too stale to trust, so your fallback for long outages might be stricter (hold suspicious orders for manual review) rather than serving very old scores. Design the fallback for the *long* outage, not the convenient short one, and decide explicitly how stale is too stale.
- **What if two dependencies fail at once?** Independent failures are unlikely to coincide, but correlated ones (same cloud region, same DNS) are not. If your auth provider and your payment processor both ride the same cloud and the region fails, both fallbacks fire simultaneously — make sure they do not contend for the same resource (the same thread pool, the same cache) or one will starve the other. Bulkheads, which give each dependency its own isolated pool, exist precisely for this.
- **What if the fallback itself fails?** The secondary processor could be down too, or your cache could be cold (just restarted, nothing to serve). This is why the chain in section 3 has *three* levels and why the payment design has a queue behind the secondary: each fallback needs its own fallback until you reach a level that cannot fail (a static default, an honest "try again shortly" with the order preserved).
- **What if the vendor comes back unhealthy?** A recovering vendor often comes back at reduced capacity, and if every client slams it the instant the breaker half-opens, they go right back down — a thundering herd. The breaker's half-open trickle plus retries with backoff and jitter (covered in the timeouts-and-retries sibling) are what let the vendor recover gracefully instead of being re-killed by your enthusiasm.

A design that survives all four of these is genuinely resilient to the dependency; one that only handles the two-minute blip will be back in the postmortem the first time the outage runs long.

## 6. Mapping your dependencies: the inventory you build before the outage

You cannot manage risk you cannot see. The foundational artifact of dependency management is a **dependency inventory**: a living document that lists every dependency, classifies it, and records what happens when it fails. Figure 7 shows an excerpt of the shape.

![Matrix showing a dependency inventory excerpt with each dependency classified by hard or soft blast radius fallback existence and vendor service-level agreement](/imgs/blogs/managing-third-party-and-dependency-risk-7.png)

The columns that matter:

- **Dependency** — the name, and ideally the specific endpoints you call.
- **Hard or soft** — using the section-2 definition, *as your code actually behaves*, not as you wish it behaved.
- **Criticality / blast radius if down** — what breaks for users when this fails. "No revenue," "no logins," "missing avatars," "delayed emails." This is the column that focuses attention.
- **Fallback exists?** — yes/no, and what it is. "Yes, second processor." "Cache, 10 min stale." "None yet" — and that last entry is a backlog item, because a hard dependency with no fallback is your most acute risk.
- **Vendor SLA** — what they promise, used to set your own expectations and to track them against reality (section 8).

Here is a fuller inventory rendered as a table, the kind you would actually keep in a runbook or a config file:

| Dependency | Class | Blast radius if down | Fallback | Vendor SLA |
| --- | --- | --- | --- | --- |
| Payment processor (primary) | Hard | No checkout, lost revenue | Secondary processor, auto-failover | 99.99% |
| Auth-as-a-service | Hard | No new logins | Cache valid JWTs, 1h grace | 99.9% |
| Primary database | Hard | No reads/writes | Read replica for reads | self-run |
| Object store (uploads) | Hard | No file uploads | None yet — **risk** | 99.9% |
| Transactional email | Soft | Delayed receipts | Queue + secondary provider | 99.5% |
| Recommendation API | Soft | Generic list shown | Static bestsellers | none |
| Feature-flag service | Soft | Last config served | Cache, fail-open to defaults | 99.9% |
| Analytics ingestion | Soft | No event data (temp) | Local buffer, drop on overflow | none |

This table does several jobs at once. It tells you your hard-dependency count for the section-1 math (four hard here). It surfaces the acute risks — the object store with "none yet" in the fallback column is the thing that will hurt you, and now it is visible and prioritizable instead of a surprise. It tells incident responders, during an outage, exactly what is expected to degrade and how, which shortens diagnosis enormously. And it forces the honest classification conversation: someone will claim the feature-flag service is hard ("we read it on every request!"), and the inventory makes you decide whether you fail-open to default flag values (soft) or genuinely cannot serve without it (hard, and a problem).

The principle here ties straight back to the SRE loop: **you cannot define, measure, or budget reliability for dependencies you have not enumerated.** The inventory is the "define" step applied to your external surface. Build it before the outage, because the alternative is building it *from* the outage — reconstructing your dependency graph at 3am from stack traces while revenue burns, which is exactly the postmortem action item nobody enjoys writing. Keep it in version control next to your code so it reviews like code, drifts less, and can even be partially generated (section 7).

A practical way to keep the inventory honest is to derive part of it mechanically. You can scrape your service mesh or your tracing data to list which external hosts you actually call, then reconcile that against the hand-maintained inventory. Anything in the traces that is not in the inventory is a dependency you forgot you had — which is precisely the dangerous category.

```promql
# Distinct external dependencies this service actually called in the last day,
# from client-side spans / metrics tagged with the upstream peer.
count by (upstream_service) (
  sum by (upstream_service) (
    rate(http_client_requests_total{service="checkout"}[1d])
  )
)
```

Run that, diff it against your written inventory, and you will almost always find a surprise. That diff is the cheapest dependency discovery you will ever do.

A few habits keep the inventory from rotting into the stale wiki page nobody trusts. **Tie criticality to a tier, not a vibe.** Give each dependency a tier — tier 0 is "outage = full user-facing outage" (auth, payment, primary DB), tier 1 is "outage = major degradation" (a key feature down), tier 2 is "outage = minor or invisible" (analytics, avatars). The tier drives the level of investment: tier 0 dependencies *must* have a fallback and a tested failover, tier 1 *should* have a fallback, tier 2 just needs to degrade cleanly. This converts the inventory from a flat list into a prioritized work queue — the tier-0 rows with "fallback: none" are your sprint backlog, in order. **Review the inventory on a cadence and after every dependency-driven incident.** New dependencies sneak in with every feature; the trace-diff catches them, but only if someone runs it. Make "update the dependency inventory" an explicit step in your design-review checklist and a standing postmortem action item. **Store it where it is useful during an incident** — in the runbook the on-call opens, not in a doc three clicks deep that nobody finds at 3am. The inventory earns its keep precisely in the minutes when an engineer is staring at a green self-dashboard and a red conversion graph, asking "what do we depend on that could cause this?" — and the answer is one glance away instead of a 25-minute archaeology dig.

## 7. The supply chain: the dependencies behind your dependencies

The inventory in section 6 captures the dependencies you *call at runtime*. But there is a whole second layer of dependency risk that does not show up in your request traces at all: the **supply chain** — the libraries you build with, the base images your containers start from, the package registries you pull from, the CI/CD provider that builds and ships you, the DNS that resolves every name, and the certificate authority whose trust your TLS depends on. Figure 6 stacks these layers.

![Layered stack diagram showing a service atop its direct vendors atop libraries and base images atop CI and DNS atop the certificate authority atop the cloud provider](/imgs/blogs/managing-third-party-and-dependency-risk-6.png)

These are dependencies in the truest sense — your service cannot exist or deploy without them — but they fail in ways that feel different from a vendor API timing out:

- **The package registry goes down** and your CI build cannot fetch a dependency, so you cannot ship the fix for your *current* outage. Now a vendor outage and a registry outage have correlated to block your only escape route.
- **A transitive library** four levels deep in your dependency tree ships a breaking change, or worse, a malicious or broken version, and your next build pulls it. You depended on it, you just never knew its name.
- **DNS fails** — and DNS is a dependency of *everything*, including your ability to reach all your other dependencies. The Dyn incident was a DNS outage that took down sites whose own servers never went down.
- **A certificate expires or a CA has an incident**, and suddenly every TLS handshake to or from the affected service fails. Certificate expiry is one of the most common self-inflicted "dependency" outages precisely because the CA and the expiry date are easy to forget you depend on.
- **The CI/CD provider has an outage** and you cannot deploy, which is invisible until the moment you urgently need to.

The defining trait of supply-chain dependencies is that they are **transitive and mostly invisible.** You did not choose the library your library depends on. You did not list DNS in your architecture diagram because it felt like physics rather than a service. This is why supply-chain outages surprise even careful teams: the dependency you got taken down by was one you never consciously took on.

What you can actually do:

- **Pin and vendor what you can.** Lock dependency versions (lockfiles, pinned image digests) so a build is reproducible and a surprise upstream publish cannot silently change what you ship. Consider mirroring critical packages and base images into a registry you control, so a public-registry outage does not block your builds.
- **Generate a software bill of materials (SBOM).** An SBOM enumerates every library and version in your build, transitive ones included. It is the supply-chain analogue of the runtime inventory: you cannot reason about a dependency you have not listed, and the SBOM lists the ones that never appear in traces.
- **Monitor the boring infrastructure dependencies explicitly.** Alert on certificate expiry *well* before it happens (30 days out, not 30 minutes). Track DNS resolution success. Know your CI provider's status. These are the dependencies that fail rarely and catastrophically, which is the worst combination because rarity breeds complacency.
- **Reduce blast radius of registry/CI outages on your escape routes.** Keep the ability to deploy a known-good image even when the registry or CI is down — a cached last-good image, a break-glass deploy path. The worst time to discover you cannot build is during an incident.

The principle: **map the dependency graph deep enough to include the things you treat as physics.** DNS, the CA, the registry, and CI feel like the ground beneath your feet, which is exactly why their failure is so disorienting. Putting them in the inventory — even with "fallback: none, monitor expiry" — converts a category of total surprise into a known, watched risk.

## 8. Dependency SLOs and vendor SLAs: hold them to it, design as if they'll miss

You should hold your vendors to a service-level objective, track their reliability, and have contractual teeth — and you should design your system as if none of that will save you on the day it matters. Both halves of that sentence are true and they do not contradict each other.

First, the terms, because they are easy to conflate. An **SLA** (service-level agreement) is the contractual promise a vendor makes you, usually with a financial penalty (a service credit) if they miss it. An **SLO** (service-level objective) is the internal target you hold a dependency to, which may be stricter than the SLA — you might *require* 99.95% from a vendor whose SLA only promises 99.9%, because your own SLO demands it. The gap between them is risk you are knowingly carrying.

**Do hold the vendor to a number.** Track the success rate and latency of your calls to each vendor as first-class SLIs, exactly as you would for an internal service. This gives you three things: early detection (you often see degradation before their status page admits it), an objective record for contract conversations ("you promised 99.9%, our measurements show 99.4% this quarter"), and the data to decide whether a vendor is worth keeping. A vendor that chronically misses its SLA is a reliability liability you are paying for, and you cannot have that conversation without measurements.

Here is a recording rule and an alert that turn vendor calls into a tracked SLI, so a degrading vendor pages you on your terms rather than theirs:

```yaml
groups:
  - name: vendor-slo
    rules:
      # Vendor success ratio over a rolling 5m window
      - record: vendor:success_ratio5m
        expr: |
          sum by (vendor) (rate(vendor_requests_total{outcome="success"}[5m]))
          /
          sum by (vendor) (rate(vendor_requests_total[5m]))

      # Page when a critical vendor's success rate drops below its SLO floor
      - alert: VendorBelowSLO
        expr: vendor:success_ratio5m{tier="critical"} < 0.99
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "Vendor {{ $labels.vendor }} success {{ $value | humanizePercentage }} < SLO"
          runbook: "https://runbooks.internal/vendor-degraded"
```

**And design as if they will miss it anyway**, because the SLA does not buy you uptime — it buys you a refund. This is the hard truth that the contract obscures. When the payment processor is down for two hours, the SLA credit you receive next month is a few percent off your monthly bill. It does not restore the revenue you lost during those two hours, and it does not restore your users' trust, which is the more expensive casualty. Users who hit a broken checkout do not file a claim against the vendor's SLA; they go to a competitor and they remember. **The contract is a financial backstop, not an availability mechanism.** Treat the SLA as a way to recover *some money* and to hold the vendor accountable over time, never as a reason to skip building your own resilience.

This resolves the apparent tension into a clean operating posture: *measure the vendor to manage the relationship; engineer your fallbacks to manage the outage.* The measurements feed your quarterly vendor review and your renewal decisions. The fallbacks feed your survival on the bad day. One is about the contract; the other is about the user. Confusing them — assuming the SLA will protect your users — is how teams end up with a vendor that "promised 99.99%" and a checkout that was down for two hours with nothing to fall back to.

There is a subtler point about *what to alert on* for a vendor. The naive instinct is to page whenever the vendor returns any error. But if you have done the conversion work — the vendor call is now soft, with a cache and a fallback — then a vendor error does *not* hurt your user, so paging a human at 3am for it is exactly the alert-fatigue trap the series warns against. The right thing to alert on is **user-facing impact**, measured as a burn rate against *your own* SLO. Burn rate is the multiple of your error-budget consumption versus the steady rate that would exactly exhaust the budget over the SLO window; a burn rate of 1 spends the budget exactly on schedule, and a burn rate of 14 exhausts a 30-day budget in roughly two days. A multi-window burn-rate alert on your user-facing SLI catches the case where the vendor outage is *leaking through your fallback* and actually hurting users (the fallback failed, the cache was cold, the secondary was also down) while staying silent when the fallback is doing its job:

```yaml
groups:
  - name: checkout-burn
    rules:
      # Page only when the USER-facing error budget burns fast AND the
      # short window agrees — i.e., the vendor outage got past our fallback.
      - alert: CheckoutFastBurn
        expr: |
          (
            checkout:error_ratio5m  > (14 * 0.001)
            and
            checkout:error_ratio1h > (14 * 0.001)
          )
        labels:
          severity: page
        annotations:
          summary: "Checkout burning budget at 14x — fallback not absorbing the outage"
          runbook: "https://runbooks.internal/checkout-degraded"
```

Note the `0.001` is `1 - SLO` for a 99.9% target. The vendor SLI alert from earlier is *informational* (severity could be a ticket, not a page) — it tells you the vendor is degraded so you know why the breaker tripped. The burn-rate alert is the *page* — it fires only when users are actually feeling it. That split is the whole discipline: **observe the dependency, but page on the user.** A vendor can be on fire and your on-call can stay asleep, because the fallback you built is quietly absorbing the outage exactly as designed, and that silence is the measured proof your resilience works.

#### Worked example: what the SLA credit actually buys

Your payment processor's SLA promises 99.99% and includes a 10% monthly service credit if they fall below it. They have a 2-hour outage. Over a 30-day month (43,200 minutes), 2 hours is 120 minutes, so their availability that month was about 99.72% — well under the 99.99% promise, so the credit triggers. Say your processing bill is \$8,000/month; the credit is \$800.

Now the other side of the ledger. During those 2 hours you would normally have processed, say, \$50,000 in orders at a 3% margin, and with no fallback you captured \$0 of it. The lost margin is \$1,500, plus an unknowable amount of trust and future purchases from customers who hit a broken checkout. The \$800 credit covers about half of one hour's lost margin and none of the trust. The arithmetic makes the principle concrete: **the SLA credit is real money, but it is an order of magnitude smaller than the cost of the outage it "compensates."** Build the fallback; bank the credit as a bonus.

## 9. Single-vendor risk and when multi-vendor is worth it

The most acute form of dependency risk is depending on a **single vendor for a critical function** with no alternative. One payment processor, one auth provider, one CDN, one cloud region. When that vendor fails, your critical function fails completely, and your only options are to wait and to apologize. Multi-vendor — running two independent providers for the same function — eliminates the single point of failure, but it is expensive and complex, and it is not always worth it. Knowing *when* it is worth it is the judgment this section is about. Figure 8 shows the ceiling math for softening, which is the cheaper alternative you should exhaust first.

![Before and after diagram showing how softening three of six hard dependencies raises the availability ceiling from ninety-nine point four to about ninety-nine point seven percent](/imgs/blogs/managing-third-party-and-dependency-risk-8.png)

The cost of multi-vendor is real and worth stating plainly:

- **Double integration.** Two SDKs, two sets of credentials, two webhook formats, two sets of quirks and edge cases. Every feature you add must work against both.
- **Reconciliation and consistency.** With two payment processors you have two sources of transaction truth to reconcile, two refund flows, two dispute processes. With two databases you have a consistency problem. The state-management burden is often larger than the integration burden.
- **The failover path itself must be tested**, or it is theater. A secondary provider you have never actually cut over to is a liability disguised as a safety net — it may have stale credentials, an unfunded account, or a code path that was never exercised. We will stress-test exactly this below.
- **You inherit the *worse* of the two vendors' constraints** in some dimensions (the lower rate limit, the smaller feature set you can rely on across both).

So when is it worth that cost? The decision rule: **multi-vendor is worth it when the function is on the critical revenue/trust path, the vendor's outages are frequent or long enough to threaten your SLO, and the providers are genuinely independent.** Payments clear this bar — revenue-bearing, and processors do have multi-hour outages. Transactional messaging often clears it for businesses where the message *is* the product (a 2FA code, a booking confirmation). A CDN clears it for sites where availability is the business. But personalization, recommendations, internal analytics, avatars — these do not clear the bar, because softening them (section 3) gets you almost all the resilience for a fraction of the cost.

That "genuinely independent" qualifier matters more than it looks. Two payment processors that both settle through the same banking rail, or two CDNs that both run on the same cloud, or two SaaS tools that both depend on the same DNS provider, are *correlated* — they can fail together, which defeats the entire point. Before you invest in a second vendor, check that its failure is actually independent of the first. Correlated redundancy is redundancy you paid for and do not have.

#### Worked example: softening three dependencies raises the ceiling

Return to the six-hard-dependency service from section 1, ceiling 99.4%. Before reaching for a second vendor for each, apply the cheaper fix — soften the ones you can:

- **Feature flags** → cache the config and fail-open to last-known-good. Now soft. Removed from the product.
- **Tax calculation** → cache rates (they change rarely) and serve cached on failure, with a daily refresh. Now soft. Removed.
- **Address validation** → make it advisory: if it is down, accept the address as entered and validate asynchronously. Now soft. Removed.

That leaves three genuinely hard dependencies (auth, payment, fraud-scoring). The new ceiling:

$$0.9999 \times (0.999)^3 = 0.9999 \times 0.99700 \approx 0.9969$$

So **99.7%**, up from 99.4%. In annual terms, the ceiling moved from about 52.6 hours of forced downtime to about 26.3 hours — **roughly half**, recovered without integrating a single second vendor, just by changing how three calls behave under failure. Figure 8 draws this. Only *now*, with the ceiling lifted and three hard dependencies remaining, do you evaluate multi-vendor for the highest-value of them (payment), because each hard dependency that survives the softening pass is a candidate for the expensive treatment. The order of operations is the lesson: **soften first (cheap, big wins), then go multi-vendor on what's left (expensive, targeted).**

## 10. War story: the payment outage we redesigned away

Here is the full arc of the outage that opened this post, because it is the most concrete possible illustration of every principle above. Some details are composited from common patterns, but the shape is real and the redesign is exactly the kind teams ship after this class of incident.

**The architecture, before.** Checkout called a single payment processor synchronously, on the critical path, with the SDK's default timeout (effectively very long). There was no fallback. The processor was, by the inventory's honest classification, a hard dependency with "fallback: none." On paper everyone knew payments were critical; in practice nobody had asked the section-3 question — does this *have* to be a single hard call? — because the integration worked and there had never been a problem. The processor's SLA promised 99.99%, and we had quietly treated that promise as if it were a guarantee of our own uptime.

**The incident.** At 14:11 the processor began returning 503s and, worse, hanging on a fraction of requests. Because we had no tight timeout, those hung requests held our checkout threads. Within ten minutes our checkout service's thread pool was saturated — not just the payment step, but *every* checkout request, including ones that had not reached the payment step yet, was now queueing behind hung threads. So we had two failures stacked: the processor was down (their fault, unfixable by us), and our own service was degrading from thread exhaustion (our fault, fixable but not in the moment). The bridge could see the processor's status page say "investigating" and could do precisely nothing about the root cause. We failed checkout fully for 113 minutes. Lost revenue and a wave of support tickets, all from a vendor outage we had no control over.

**What the postmortem found.** Three contributing factors, in the [blameless](/blog/software-development/site-reliability-engineering/learning-from-incidents-at-scale) spirit of "what about the system let this hurt so much," not "who messed up":

1. The payment call was a hard dependency with no fallback — a single point of failure on the revenue path.
2. The missing timeout turned the *vendor's* slowness into *our* thread exhaustion, amplifying a contained vendor outage into a full service degradation. This is the [cascading-failure pattern](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — a slow dependency consuming a shared resource — applied to a third party.
3. We had no measurement of the vendor's reliability and learned of the outage from our conversion graph, not from a vendor SLI, costing us minutes of detection.

**The redesign.** Three changes, mapping one-to-one to the sections above:

- **Tight timeout plus circuit breaker on the payment call** (section 5), so a slow processor fast-fails in 250ms and trips the breaker rather than holding threads. This alone would have prevented our self-inflicted thread exhaustion — the vendor outage would have stayed the vendor's outage.
- **A secondary payment processor with automatic, health-checked failover** (section 9), genuinely independent of the primary (different settlement rail, verified). When the primary's breaker opens, the router sends new charges to the secondary. Multi-vendor was worth it here by every criterion: revenue path, real outage frequency, independent providers.
- **A queue-and-retry path for the rare double-down** (section 5), so that in the extraordinary case where *both* processors are unavailable, we accept the order optimistically and charge asynchronously when a processor recovers, telling the user their order is confirmed.

**The result, measured.** Six weeks later the primary processor had another, shorter outage — about 35 minutes. This time: the breaker tripped within seconds, the router failed over to the secondary, and **checkout conversion never left its normal band.** Zero user-facing checkout errors. We learned about the primary's outage from our own vendor SLI dashboard (and our automatic failover alert), not from angry users. The 113-minute total outage had become a 35-minute vendor incident that our users never saw. We did not make the processor more reliable — we could not. We made its outage stop being ours.

That is the whole thesis in one before-and-after. The vendor's reliability was unchanged and uncontrollable. The only variable we touched was our own design, and it turned a revenue-ending outage into a non-event.

## 11. A second war story: the DNS provider and the dependency nobody listed

The payment outage was a dependency we knew about and underprotected. This one was a dependency we did not know we had at all, and it is the more instructive failure mode because no inventory built from the architecture diagram would have caught it.

**The setup.** A mid-size SaaS product, healthy and boring, ran across three availability zones with a sensible amount of redundancy. The architecture diagram showed the load balancer, the app tier, the database with a replica, a cache, and a handful of vendor APIs — all classified, all with fallbacks where it mattered. By the section-6 standard the inventory looked complete. What the diagram did not show, because it felt like physics, was the managed DNS provider that every one of those components used to resolve every name: the database hostname, the cache endpoint, the vendor API domains, even the internal service names. DNS was not in the inventory because nobody thought of it as a dependency. It was the ground.

**The incident.** The DNS provider had a regional incident — resolution started failing for a fraction of queries and slowing for the rest. The symptom in our systems was bizarre and scattered: intermittent connection failures to the database (it could not resolve the hostname), the cache "disappearing" (same), vendor calls timing out (same), and our own service-to-service calls failing unpredictably. Every individual subsystem reported a different, plausible-looking error, and the on-call engineer spent the first 25 minutes chasing each one separately — restarting the cache, failing the database to its replica (which also could not be resolved, so that made it worse), checking the vendor status pages (all green, because *they* were fine). The actual root cause — one shared dependency underneath all of them — was invisible precisely because it was shared. It looked like five unrelated failures, not one.

**What broke the deadlock.** A senior engineer who had lived through the Dyn outage asked the right question: "what do all of these have in common?" The answer was DNS, and a single `dig` against the provider confirmed resolution was failing. The moment the *shared* dependency was named, the incident collapsed from five mysteries into one known event with a known shape. Mitigation was to fail over to a secondary DNS provider (which, thankfully, had been configured at the registrar level years earlier and never used) and to extend DNS cache TTLs so resolved names stayed valid longer, riding out the provider's degradation.

**The lessons, mapped to this post.** First, **the dependency you got taken down by is often one you never consciously took on** (section 7) — DNS, the CA, the registry, CI. These belong in the inventory even though they feel like physics. Second, **a shared dependency creates correlated failure that masquerades as many independent failures**; when several unrelated subsystems degrade at once, suspect a shared dependency underneath before you debug each symptom in isolation. This is a debugging discipline as much as a reliability one — the [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) (form a hypothesis that explains *all* the symptoms, not each one separately) is exactly what cracked it. Third, **the cheapest insurance is the redundancy you set up before you need it and hope never to use** — the secondary DNS provider configured years earlier turned a potential multi-hour outage into a 40-minute one.

**The result.** Postmortem action items: DNS, the CA, and the package registry went into the inventory with explicit monitoring (resolution success rate, cert expiry alarms 30 days out, registry reachability checks in CI). The secondary DNS provider, previously a dusty config nobody trusted, got a quarterly failover drill so it would actually work next time. And the team added a "what's underneath all of these?" prompt to the incident runbook for the specific pattern of many-simultaneous-unrelated failures. The next DNS hiccup, eight months later, was diagnosed in under four minutes because the pattern was now a known shape rather than a novel mystery.

The two war stories bracket the whole topic. The payment outage was a *known* dependency that was underprotected — fixed by softening and multi-vendor. The DNS outage was an *unknown* dependency that was uninventoried — fixed by mapping the graph deep enough to include the things you treat as physics. Most real dependency incidents are one of these two shapes, and the inventory plus the conversion techniques address both.

## 12. How to reach for this (and when not to)

Dependency-risk engineering has a cost, and like every reliability practice it can be over-applied. Here is the decisive guidance.

**Do this, always:** Build the dependency inventory (section 6) — it is cheap, it is the foundation for everything else, and there is no service for which it is not worth having. Put a tight timeout and a circuit breaker on *every* external call (section 5) — this is close to free and prevents the single most common dependency-driven self-inflicted outage (thread exhaustion from a slow vendor). Classify every dependency hard or soft honestly (section 2). These three are baseline hygiene for any production service.

**Do this where the math justifies it:** Soften hard dependencies to soft (section 3) wherever staleness or a default is acceptable — this is the highest-leverage work and you should do a lot of it, but it requires the product-side agreement that a feature may sometimes be degraded. Track vendor SLIs (section 8) for your critical vendors. Generate an SBOM and monitor cert expiry / DNS / CI (section 7) — proportional to how much a supply-chain outage would hurt you.

**Do NOT do this:** Do not go multi-vendor for a soft dependency — a second avatar service or a backup recommendation API is pure cost for resilience you do not need; just soften it. Do not over-invest in a second vendor whose failure is *correlated* with the first (same rail, same cloud, same DNS) — you will pay for redundancy you do not actually have. Do not treat a vendor SLA as an availability mechanism — it is a refund, not uptime (section 8). Do not build a failover path you never test — an untested secondary is a liability that gives false confidence; if you cannot afford to drill the failover, you cannot afford to rely on it. And do not try to eliminate all dependency risk — you cannot, and the attempt produces a baroque system that is *less* reliable because it is too complex for anyone to operate. The goal is not zero dependency risk; the goal is that no single dependency's outage can take you fully down.

The honest framing: every fallback, every cache, every second vendor is itself code that can have a bug, drift out of sync, or fail in a novel way. Resilience is not free and it is not strictly additive. Spend it where the inventory says the blast radius is largest and the fallback is missing, and leave the low-blast-radius soft dependencies to degrade gracefully. The inventory tells you where to spend; this section tells you not to spend everywhere.

This connects back to the whole SRE loop. Dependency risk is a way your error budget gets spent by someone other than you, and the [SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — reliability is a number you engineer and budget — applies directly: budget for the dependency outages you know are coming, design so they cost you degradation rather than downtime, and stop trying to make uncontrollable things controllable. The architecture-time companion to all of this is [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure), which assumes every component will fail and builds accordingly; this post is what that assumption looks like when the failing component is a vendor you cannot page.

## Key takeaways

- **Your availability is bounded by the product of your hard dependencies' availability.** Five hard deps at 99.9% cap you near 99.5%; every hard dependency you add lowers the ceiling. Count them before you promise an SLO.
- **Hard versus soft is the classification that decides everything.** A hard dependency's failure fails your request and multiplies into your ceiling; a soft one you survive and it does not. The label describes your code's behavior, so you can change it.
- **Converting hard dependencies to soft is the highest-leverage reliability work most teams skip.** Cache its data, make the call async, or provide a default — each conversion removes a factor from your availability product.
- **You cannot fix a vendor's outage; you can only design around it.** Their status page is not your monitoring, their fault is still your user's broken experience, and your pre-built resilience is the only variable you control.
- **Timeout fast, circuit-break the vendor, cache last-known-good, fail over to a secondary, queue deferrable writes, fail soft for non-critical features** — layered, these turn a provider outage into a degraded mode.
- **An SLA buys you a refund, not uptime.** Hold vendors to a tracked SLO and use the contract for accountability, but design as if they will miss it, because the credit never restores lost revenue or trust.
- **Map your dependencies before the outage maps them for you** — an inventory with class, blast radius, and fallback-exists is the cheapest risk reduction available, and it must include the supply-chain dependencies (DNS, the CA, the registry, CI) you treat as physics.
- **Multi-vendor only where the math justifies it** — critical revenue/trust path, real outage frequency, genuinely independent providers. Soften first (cheap), then go multi-vendor on what survives the softening (expensive, targeted).

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the budget-it-don't-wish-it framing that governs this whole topic.
- [Setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) — where the nines-to-downtime math and the achievable-target conversation come from.
- [Graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — the degradation philosophy that softening a dependency is one application of.
- [Circuit breakers, bulkheads, and load shedding](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding) — the mechanics of breaking the vendor call and isolating its blast radius.
- [Timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) — why a missing timeout turns a slow vendor into your thread exhaustion.
- [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — the architecture-time stance that every component, including every vendor, will fail.
- [Cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — the system-design companion on how a slow dependency consumes shared resources and propagates.
- *Site Reliability Engineering* (Google) and *The SRE Workbook*, especially the chapters on managing service dependencies, addressing cascading failures, and embracing risk — the canonical treatment of dependency-driven availability.

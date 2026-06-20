---
title: "Choosing SLIs That Reflect User Pain: Measure What People Feel, Not What Your Box Is Doing"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to pick the three service-level indicators that actually move when users are unhappy, specify them rigorously, and stop paging yourself for a CPU number nobody feels."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "sli",
    "golden-signals",
    "observability",
    "prometheus",
    "reliability",
    "latency",
    "error-budget",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/choosing-slis-that-reflect-user-pain-1.png"
---

At 3:14am the pager went off for a host at 92% CPU. The on-call engineer rolled over, squinted at the phone, opened the dashboard, and found the box happily serving every single request in 48 milliseconds. Nobody was suffering. No user had filed a ticket. The error rate was zero. The checkout flow was humming. And yet here we were, awake, staring at a red graph, because months earlier somebody had decided that "CPU above 90%" was a thing worth waking a human for. That engineer acknowledged the page, muttered something unprintable, and went back to sleep — and in doing so quietly trained themselves to ignore the pager, which is exactly how a real outage slips past an exhausted on-call three weeks later.

That page was a lie. Not a technical lie — the CPU really was at 92% — but a lie about what mattered. The thing we were measuring (CPU) had nothing to do with the thing we cared about (whether users could check out). This is the single most common, most expensive mistake in operating a service: measuring what is easy to measure instead of what users feel. You can instrument three hundred numbers off a modern service — CPU, memory, GC pauses, thread-pool depth, socket counts, file descriptors, heap fragmentation, connection-pool saturation, disk queue length. Almost none of them belong on the contract you make with your users. Only a tiny handful do. The discipline of this post is choosing those few — the Service Level Indicators, or **SLIs**, that move when users are unhappy and stay perfectly still when users are fine.

![A comparison of a CPU-based SLI that pages at 3am while every request still succeeds fast versus a success-rate SLI that only moves when users feel real pain](/imgs/blogs/choosing-slis-that-reflect-user-pain-1.png)

By the end of this post you will be able to do something concrete: walk up to any service — a web API, a checkout flow, a nightly data pipeline, an object store — and pick the two or three SLIs that genuinely track user pain, specify each one precisely enough that two engineers would compute the identical number, and write the PromQL that measures it. You will also be able to defend why you *rejected* the tempting metrics — the CPU SLI, the memory SLI, the "is the process running" SLI — that look responsible but page you for nothing. We are at the very front of the SRE loop here: **define reliability (SLI) → measure it → spend the error budget → reduce toil → respond to incidents → learn → engineer the fix.** Everything downstream — your SLO, your error budget, your burn-rate alerts, your on-call sanity — inherits the quality of the SLIs you pick at this step. Choose a bad SLI and the rest of the loop is built on sand. This is the foundation, and it is worth getting right.

A note on where this sits. This is the *operations* layer — how you run a live service and decide what "reliable enough" means in numbers. It is a sibling of, not a substitute for, the architecture-time work of designing systems that degrade gracefully; for that, see the system-design treatment of [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation). And it is the first concrete step after the [SRE mindset intro](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) that frames reliability as a feature you engineer, not a wish you make.

## 1. What an SLI actually is (and the one test that separates good from bad)

Let me define the term before I lean on it. A **Service Level Indicator (SLI)** is a carefully defined quantitative measure of some aspect of the level of service you provide. In almost every useful case it is a *ratio of good events to total valid events*, expressed as a percentage:

$$ \text{SLI} = \frac{\text{good events}}{\text{total valid events}} \times 100\% $$

"Good" means the event made the user happy — the request returned a success status quickly enough, the data was fresh enough, the read returned the right bytes. "Total valid" means every event that *should* count — usually every real user request, minus the things that are not really user traffic (synthetic health checks, your own monitoring probes). An SLI built this way has a beautiful property: it lives between 0% and 100%, it is directly comparable across services, and it is the exact input your **SLO** (Service Level Objective — the target you commit to, like 99.9%) and your **error budget** (the SLO's complement, the amount of unreliability you are allowed to spend) need downstream. We will cover SLOs and the three numbers — SLI, SLO, and **SLA** (the contractual Service Level Agreement with penalties) — in the sibling post [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter). Here we stay laser-focused on the SLI: the raw measurement of user happiness.

Now, the one test. Here is the property that separates a good SLI from a bad one, and you should tattoo it somewhere:

> **A good SLI moves when users are unhappy and stays still when users are fine.**

That is it. That is the whole discipline compressed into one sentence. Run any candidate metric through that test. CPU at 90% with every request served in 48ms? Users are fine, but the metric is screaming — it *moves when users are fine*, so it fails the test, so it is a bad SLI. The fraction of checkout requests that returned a 5xx error? When that climbs, users genuinely cannot buy things; when it is zero, users are buying happily. It *moves when users are unhappy and stays still when they are fine* — it passes, so it is a good SLI. This single test will save you from ninety percent of bad SLI decisions, and it is the reason CPU, memory, disk, thread count, and "the process is alive" are all wrong as user-facing SLIs. None of them have a tight relationship with user happiness. They can move wildly while users are fine, and — more dangerously — they can sit perfectly still while users suffer.

That last point deserves emphasis because it is the subtle failure. A bad SLI is not just one that cries wolf. It is also one that stays *silent during a real fire*. Consider a service whose CPU is a comfortable 40%, memory is fine, the process is up — every health check green — but a misconfigured downstream timeout means one in twenty user requests hangs for 30 seconds and then fails. Your CPU SLI says everything is perfect. Your users are filing angry tickets. The metric that *should* protect them — the success rate of real requests, the fraction served fast enough — would have caught it instantly. The infrastructure metrics are blind to it. This asymmetry is why we anchor SLIs to user-felt outcomes: only an outcome-based metric is faithful in both directions.

### Why we obsess over picking only a few

You might ask: why not just track everything and call them all SLIs? Because an SLI is not a dashboard panel. It is a *commitment*. Each SLI you elevate to that status becomes the basis of an SLO, which becomes an error budget, which decides whether you ship features or freeze to fix reliability, which decides whether someone gets paged. Every SLI carries the full weight of the loop behind it. Track three hundred and you have committed to nothing, because nobody can reason about three hundred competing signals at 3am. Track three — availability, latency, and maybe one more — and you have a contract a human can hold in their head and defend in a meeting. The scarcity is the point. The art is choosing *which* three.

## 2. The four golden signals: latency, traffic, errors, saturation

The single most useful starting menu comes from the Google SRE book, and it is called the **four golden signals**. The book's advice is blunt: if you can only measure four things about your user-facing system, measure these four. They are latency, traffic, errors, and saturation. Each one maps to a distinct flavor of user pain, and together they cover the space of "is this service healthy from where the user stands" with very few blind spots.

![A taxonomy tree of the four golden signals splitting into user-facing latency, traffic, and errors plus the capacity-side saturation signal](/imgs/blogs/choosing-slis-that-reflect-user-pain-2.png)

Let me walk each one and, crucially, say what flavor of user pain it represents and whether it makes a good *SLI* or merely a good *signal to watch*.

**Latency** — how long a request takes. This is the user staring at a spinner. Latency is the canonical SLI for any interactive service, because a slow response is a degraded response even when it eventually succeeds. The critical refinement: you must separate the latency of *successful* requests from the latency of *failed* ones, because a fast failure (an immediate 500) can look deceptively good on a naive latency graph and mask a real problem. You also almost never want the *average*. Averages hide the tail. You want a high percentile — **p99**, the latency below which 99% of requests fall — because the user who waited 8 seconds does not feel better knowing the average was 200ms. Latency is a golden signal *and* a great SLI.

**Traffic** — how much demand the service is handling, typically requests per second. This one is different: traffic is rarely an SLI by itself, because high traffic is not user pain — it is usually success. (Your users showing up in droves is a good day.) Traffic matters as *context*: it is the denominator that makes your error rate and latency meaningful, and a sudden traffic collapse can itself be a symptom (did the load balancer stop routing to you? did the upstream die?). Watch it, alert on anomalies in it, but it is usually not the SLI you commit an SLO to.

**Errors** — the rate of requests that fail. This is the user who clicked "buy" and got an error page, or worse, got a success page but the order never went through. Errors are the most direct measure of "the service did not do the thing the user asked." The error rate, expressed as the complement of availability (good = not-an-error, SLI = successes / total), is the bedrock SLI for almost every request-driven service. The subtlety is defining what counts as an error — and that is harder than it sounds, which is why we devote a whole section to it below.

**Saturation** — how "full" the service is, how close some constrained resource is to the limit that will start hurting users. This is the one people misuse. Saturation is your CPU, your memory, your queue depth, your connection pool, your disk I/O headroom. It is genuinely useful — but as a *leading indicator for capacity and alerting*, not as a user-facing SLI. A service at 90% saturation that still serves every request fast is *fine right now*; saturation tells you that you are *about to* have a problem, which is gold for capacity planning and proactive alerts, but it is not a measure of current user pain. This is the heart of the CPU-SLI mistake, and we will return to it in depth.

So the golden signals give you four candidates, but notice they are not all equal as SLIs. Latency and errors are your SLI workhorses. Traffic is context. Saturation is a capacity/alerting signal that masquerades as an SLI and bites the unwary. Already the menu is narrowing toward the few that matter.

One more nuance on latency that trips up almost everyone the first time. When you measure latency you are measuring a *distribution*, not a single number, and the shape of that distribution carries the user pain. A service can have a beautiful 100ms median and still be making one in a hundred users wait three seconds — and those one-in-a-hundred users are real people having a bad time, often the very ones with the most data, the biggest carts, the slowest networks. The mean would hide them entirely; even the median (p50) hides them. This is why SRE practice insists on the tail percentiles, p95, p99, sometimes p99.9, and on phrasing the SLI as "the fraction of requests faster than a threshold" rather than "the p99 value." The threshold framing is subtle but important: it turns a percentile (which does not aggregate cleanly across instances and is awkward to hold to an SLO) into a clean good/total ratio (fast requests over total requests) that aggregates trivially and slots straight into the error-budget arithmetic. We will lean on exactly this trick when we write the PromQL, because the histogram makes the fast/total count almost free.

## 3. RED and USE: two menus, and which one to order SLIs from

The golden signals are the foundation, but two complementary frameworks make the choice even sharper by telling you *which lens to look through*: RED for services, USE for resources. Knowing which menu to order from is half the skill.

![A matrix contrasting the RED method measured per service against the USE method measured per resource, showing why RED yields SLIs and USE yields capacity alerts](/imgs/blogs/choosing-slis-that-reflect-user-pain-3.png)

**The RED method** — Rate, Errors, Duration — was popularized by Tom Wilkinson and is the request-centric view. For every user-facing service, measure:

- **Rate** — requests per second (this is "traffic" from the golden signals).
- **Errors** — the number or fraction of those requests that failed.
- **Duration** — the distribution of how long requests took (this is "latency").

RED is the menu you order SLIs from. Notice it is the golden signals minus saturation, and that is not a coincidence — RED is deliberately the *user-experience* slice. Every item in RED is something the user directly feels: how much they asked, how often it broke, how long it took. If you build SLIs only from RED, you will rarely build a bad one, because RED is constructed from request outcomes, and request outcomes are exactly what users experience.

**The USE method** — Utilization, Saturation, Errors — comes from Brendan Gregg and is the resource-centric view. For every constrained *resource* (CPU, memory, disk, network interface, connection pool), measure:

- **Utilization** — the percentage of time the resource was busy.
- **Saturation** — the degree to which the resource has extra work queued that it cannot service (run-queue length, swap activity, queue depth).
- **Errors** — error events from that resource (dropped packets, ECC errors, failed allocations).

USE is the menu you order *capacity alerts and debugging signals* from — not SLIs. USE answers "is this resource the bottleneck?" which is a fabulous question when you are diagnosing *why* an SLI went bad, or planning capacity so it doesn't go bad next quarter. But utilization and saturation are not user pain. A disk at 99% utilization that still serves reads in 2ms is not hurting anyone. The user does not have a CPU. The user has a request, and they want it to succeed and be fast.

So the rule writes itself: **build SLIs from RED, build capacity alerts from USE.** When someone proposes a CPU SLI, you now have the vocabulary to reject it precisely — "CPU is a USE-method utilization signal, it belongs in capacity alerting, not in our SLO; the user-facing SLI for this service comes from RED, which is the error rate and the duration." The two menus are not competitors. They are different tools for different jobs, and the most common SLI mistake is reaching for the USE tool (resource metrics) when the job calls for the RED tool (request metrics).

| Question you are answering | Reach for | Example metric | Good SLI? |
| --- | --- | --- | --- |
| Are users getting what they came for? | RED | success rate, p99 latency | Yes — this is the SLI |
| How loaded is the service right now? | RED rate / USE | requests/sec | No — context only |
| Is a resource about to become the bottleneck? | USE | CPU utilization, queue depth | No — capacity alert |
| Why did the SLI go bad just now? | USE | disk saturation, GC pauses | No — debugging signal |

## 4. The SLI menu by service type

Different services have different shapes, and the SLIs that reflect user pain differ accordingly. A request/response API and a nightly data pipeline make their users unhappy in completely different ways, and an SLI that fits one is blind to the other. Here is the menu, organized by the three service shapes you will encounter most.

![A matrix of recommended SLIs for request and response services, data pipelines, and storage systems showing how the menu differs by service shape](/imgs/blogs/choosing-slis-that-reflect-user-pain-4.png)

### Request/response services (APIs, web apps, RPC backends)

These serve a request and return a response while the user (or a calling service) waits. The user pain is: it failed, or it was slow, or it came back incomplete. The SLI menu:

- **Availability** — the fraction of valid requests that succeeded. Good = a success status (typically 2xx or 3xx, and specifically *not* a 5xx server error). This is your bedrock SLI. When availability drops, users are getting error pages.
- **Latency** — the fraction of valid requests served faster than a threshold the user notices. Note the framing: not "the p99 latency" as a raw number, but "the fraction of requests faster than X" — that turns latency into a good/total ratio you can hold to an SLO. Good = a successful request that completed in under, say, 300ms (or 1s for a heavier flow). This is the spinner SLI.
- **Quality** — the fraction of responses that were *complete* rather than degraded. Many services degrade gracefully under load: they drop the personalized recommendations, skip the expensive enrichment, serve a stale cache. The request technically "succeeded" with a 200, but the user got a worse answer. A quality SLI (good = a full, non-degraded response) captures pain that availability and latency miss. This one is optional and you only need it if your service actually degrades; if it does, it is worth it.

### Data-processing / pipeline services (ETL, streaming, batch, async jobs)

These consume input and produce output asynchronously; the user is not waiting on a synchronous response, so availability and latency *as defined above do not even apply*. Asking "what fraction of requests returned 2xx" is meaningless for a Kafka consumer chewing through events. The pain here is different — the data is stale, or wrong, or missing — so the SLI menu changes completely:

- **Freshness** — the fraction of data (or the fraction of time) that is more recent than a threshold. Good = the output reflects input from less than, say, 10 minutes ago. A pipeline can be perfectly "up" — every process running, zero errors logged — while quietly serving data that is six hours stale because a job is stuck. Freshness is the SLI that catches the stale-but-up failure that availability is structurally blind to.
- **Correctness** — the fraction of records that were processed correctly (the right transformation, no corruption, no dropped fields). Good = output that matches what the input should have produced. You usually measure this with sampling or a reconciliation check against a source of truth.
- **Coverage / completeness** — the fraction of valid input records that were actually processed and made it into the output. Good = the record appears in the output. A pipeline that silently drops 3% of events has 97% coverage, and your users are missing 3% of their data.
- **Throughput** — whether the pipeline keeps up with input rate (the backlog is not growing without bound). This is the pipeline analog of saturation; it is more of a capacity/leading signal than a pure SLI, but a chronically lagging pipeline is a freshness failure waiting to happen.

### Storage systems (object stores, databases, file systems)

The user pain is: my data is gone, or I can't get to it, or reads are slow. The menu:

- **Durability** — the fraction of stored objects that are not lost or corrupted over time. This is the SLI you care about most and measure least often, because data loss is rare and catastrophic. Good = the object is still there and intact. Cloud object stores quote durability in absurd numbers of nines (eleven nines) precisely because the pain of the rare failure is so severe.
- **Availability** — the fraction of read/write operations that succeed. Good = the operation completed without error.
- **Latency** — the fraction of operations served faster than a threshold (read under 100ms, say).

Notice the pattern across all three shapes: every SLI is phrased as a fraction of good over total, every "good" is defined from the user's standpoint, and not one of them is a resource metric. The menu by service type is your second filter (after the golden-signals / RED-vs-USE filter) for narrowing three hundred candidate metrics down to three.

| Service shape | SLI #1 (the bedrock) | SLI #2 | Optional SLI #3 | What availability/latency would MISS |
| --- | --- | --- | --- | --- |
| Request/response | Availability (success rate) | Latency (fast-enough rate) | Quality (full response) | Degraded-but-200 responses |
| Data pipeline | Freshness (data age) | Correctness | Coverage / throughput | Stale-but-up; silently dropped records |
| Storage | Durability | Availability | Latency | The rare catastrophic data loss |

## 5. The user-journey lens: SLIs on journeys, not components

Here is the mental model that ties the whole menu together and keeps you honest. Do not pick SLIs on internal components. Pick them on **critical user journeys** — the specific things a user came to your product to *do*. Can the user log in? Can they search? Can they add to cart and check out? Can they upload a photo and have it appear? Each of those is a journey, and the question an SLI must answer is brutally simple:

> **Are users able to do the thing they came to do?**

This reframing is more powerful than it looks, because it flips your attention from the inside of the system to the outside. An internal-component view says "the auth service is at 99.99%, the cart service is at 99.95%, the payment service is at 99.9%, the inventory service is at 99.99% — great, everything is green." A user-journey view asks: "what fraction of users who started a checkout actually completed one?" And those two views can give wildly different answers, because the checkout journey *traverses all four services in series*, and a user only succeeds if every link holds. The math is unforgiving here. If checkout depends on four independent services at 99.9% each, the journey availability is roughly $0.999^4 \approx 0.996$, or 99.6% — a budget four times worse than any single component, and not one component-level dashboard would have shown you that 0.4% of users hitting a wall. The journey SLI shows it immediately, because it measures the *whole path the user walks*.

This is also why measuring at the component level systematically *understates* user pain. Each component reports only on the requests that reached it and that it answered; it cannot see the user whose request died in the gap between two services, or timed out at the load balancer before the app ever got it. The journey SLI, measured as close to the user as you can get, sees all of it. So the discipline is: enumerate your handful of critical user journeys (most products have five to fifteen that matter), and put your SLIs there. A login journey gets a login-success SLI and a login-latency SLI. A checkout journey gets a checkout-success SLI and a checkout-latency SLI. You are not trying to cover every internal call — you are trying to answer, for each thing users care about, whether they can do it.

A practical consequence: this naturally caps how many SLIs you have, which is the scarcity we keep returning to. You do not have three hundred user journeys. You have a dozen, and the top three or four carry most of the value (the revenue path, the core read path, the sign-up path). Put two SLIs on each of the top journeys and you have your handful — derived from user value, not from whatever was easy to instrument.

### The arithmetic of why journey SLIs matter more than nines on paper

Before we leave the journey lens, it is worth making the cost of the wrong altitude concrete, because the numbers are surprising. People reason about reliability in "nines," and it helps to know what each nine actually buys in downtime. The error budget of an SLO is the time (or fraction of requests) you are allowed to be unreliable, and it shrinks fast as you add nines:

| SLO (over 30 days) | Allowed unreliability | Downtime budget / month |
| --- | --- | --- |
| 99% (two nines) | 1% | about 7.2 hours |
| 99.9% (three nines) | 0.1% | about 43.2 minutes |
| 99.95% | 0.05% | about 21.6 minutes |
| 99.99% (four nines) | 0.01% | about 4.3 minutes |
| 99.999% (five nines) | 0.001% | about 26 seconds |

Now combine that with the series-dependency math from a moment ago. The checkout journey traverses four services. Suppose each one is independently held to a respectable 99.95% — that is 21.6 minutes of monthly budget each, which feels generous. But the journey only succeeds when *all four* hold, so the journey availability is roughly $0.9995^4 \approx 0.998$, which is 99.8%, a budget of about **86 minutes a month** — four times worse than any single component, and a full nine short of what each component proudly reports. If you only ever looked at component SLIs, you would believe you were running a 99.95% checkout and you would be wrong by a factor of four. The journey SLI tells you the truth — 99.8% — because it measures the whole path as the user walks it. This is not a rounding detail; it is the difference between an SLO you can actually meet and one you are silently violating every month. The arithmetic is the proof: dependencies in series *multiply* their failure probabilities, so the user-visible journey is always less reliable than its most reliable component, and only a journey-level SLI captures that. Put your SLIs where the multiplication happens.

## 6. Why CPU, memory, and disk are bad SLIs (the saturation trap in full)

We have circled this point several times; now let me nail it down, because it is the single most important lesson in this post and the reason that 3am page was a lie.

CPU, memory, disk, and the rest are **saturation** signals (in golden-signals terms) or **utilization/saturation** signals (in USE terms). They measure how full a *resource* is. They are genuinely valuable — for *capacity planning* (am I going to run out of headroom next quarter?) and for *leading-indicator alerting* (this disk will fill in 4 hours at the current rate, page someone now while it's a 9-to-5 problem). But they fail the one test catastrophically, in both directions:

- **They move when users are fine.** A box at 90% CPU serving every request in 50ms is delivering a perfect user experience. An SLI on CPU < 80% would have a fat error budget burned by *nothing the user can feel*, and it would page you at 3am for a number, training your on-call to ignore the pager. This is the false-positive direction, and it is corrosive.
- **They stay still when users suffer.** A box at 30% CPU whose downstream dependency is timing out is failing one in ten user requests while every resource metric sits comfortably green. An SLI on resource metrics is *blind* to the most common real outages, which come from logic errors, bad deploys, dependency failures, and misconfiguration — none of which need to show up as a resource spike. This is the false-negative direction, and it is the one that actually gets you fired.

The fix is not to throw resource metrics away — it is to *file them correctly*. They are capacity and alerting signals. Watch saturation to know when to add capacity. Set a *leading* alert on saturation (disk will fill in N hours) so you fix it during business hours. But do not commit an SLO to them, do not put them in the error budget, and do not let them page on-call as if they were user pain. The user-facing SLI for that same service is the RED-method success rate and latency. CPU answers "will I have a problem soon?"; the success rate answers "do I have a problem now?" — and on-call should be woken only for *now*.

#### Worked example: the CPU SLI that would page you for nothing

Imagine a service running at a steady 88% CPU during peak hours — efficient, well-packed, every request served in 45ms, error rate 0.01%. Now suppose, against all advice, you set an SLI as "CPU < 80%" with a 99% SLO. During the 6 peak hours each day, CPU exceeds 80%, so those hours count as "bad." That is 6 bad hours out of 24, or 25% badness, which against a 99% SLO obliterates your entire monthly error budget in a *single day* — while delivering a flawless user experience the whole time. Your error budget says "freeze all features, this service is in crisis." Your users say "this is the best the service has ever been." The SLI is not just useless; it is actively *anti-correlated* with reality, manufacturing a phantom crisis out of healthy, efficient operation. Contrast the correct RED SLI: success rate 99.99% and fast-enough rate 99.97% — both comfortably above target, error budget barely touched, on-call asleep, because nothing is actually wrong. Same service, same hour. The only difference is whether you measured the resource or the user.

The lesson generalizes: any time a proposed SLI is a *resource* number rather than a *request outcome*, you have probably reached for the USE menu when the job needed RED. Push back, re-derive from the user journey, and the right SLI appears.

## 7. Measuring at the point closest to the user

Where you measure changes the number — sometimes by a lot — and the rule is simple: **measure as close to the user as you can.** For a request/response service that usually means at the load balancer (or even the client), not deep inside the application code.

![A request path from the user through the load balancer to the application and its dependencies, marking the load balancer as the SLI measurement point that captures dropped connections the internal counter misses](/imgs/blogs/choosing-slis-that-reflect-user-pain-5.png)

Here is why the measurement point matters so much. Suppose you count "good requests" using a counter inside your application — the app increments `requests_total` and `requests_failed_total` as it handles each request. That counter can only see requests the application actually received and finished. It is *blind* to:

- Requests that the load balancer dropped because every backend was unhealthy (the user got a 502, your app never saw it).
- Requests that timed out at the load balancer because the app was too slow to respond at all (the connection was killed before your code returned, so your "failed" counter never incremented).
- Requests rejected by the load balancer's own limits, TLS handshake failures, connection resets.

Every one of those is a real user failure, and an internal counter misses every one. Measure at the load balancer and you catch them, because the load balancer sees the request the moment it arrives and records its fate whether or not a backend ever responds. The number you get measuring at the LB is *closer to the truth of what the user experienced* than the number you get measuring inside the app — and the SLI's whole purpose is to reflect user experience. The trade-off is honesty for proximity: the closer you measure, the more user pain you capture, at the cost of needing to instrument an edge component you may not fully control. For most teams the load-balancer or ingress layer is the sweet spot — close enough to the user to be honest, owned enough to be reliable.

There is a corollary about *what to exclude*. The denominator of your SLI — "total valid requests" — should count what a user would call a real request and exclude what they would not. The classic exclusion is *health checks*: your load balancer and Kubernetes probes hammer `/healthz` constantly, and those are not user traffic. If you leave them in the denominator, a flood of trivially-successful health checks will inflate your availability and *hide* real user failures behind a wall of green probe traffic. Exclude them. Likewise exclude synthetic monitoring probes, internal warm-up traffic, and load tests — anything that is not a human or a real client doing a real thing. Getting the denominator right is as important as getting the numerator right, and it is the part people forget.

## 8. Request-based vs window-based SLIs (and when each fits)

There are two ways to compute the good/total ratio, and they behave differently enough that picking the wrong one will distort your reliability picture. Both are legitimate; they suit different traffic shapes.

![A comparison of a request-based SLI counting each request against a window-based SLI counting each good minute, showing which traffic shapes each one favors](/imgs/blogs/choosing-slis-that-reflect-user-pain-7.png)

**Request-based SLI** = good requests / total requests, over a window. You count every request, classify each as good or bad, and divide. Each request gets an equal vote. This is the most intuitive and usually the right default. It directly answers "what fraction of requests succeeded?" and it scales naturally with traffic — a busy minute counts more than a quiet one, which is fair, because a busy minute is more users.

**Window-based SLI** = good time windows / total time windows. You slice time into small buckets (say, 1-minute windows), declare each window "good" or "bad" by some rule (e.g., the window is bad if its error rate exceeded 1%), and then your SLI is the fraction of *windows* that were good. Each minute gets an equal vote, regardless of how many requests it contained.

The difference matters most at the extremes of traffic shape:

- **High, steady traffic → request-based.** When every minute has thousands of requests, request-based is precise and fair. A bad minute with a thousand failed requests is genuinely worse than a bad minute with ten, and request-based reflects that. Window-based would flatten both to "one bad minute," throwing away information.
- **Low or bursty traffic → window-based often fits better.** Imagine a service that gets 5 requests in a quiet hour, and all 5 fail. Request-based says 0% availability for that hour — technically true but it makes a 5-request blip look like a total outage and can violently swing a low-traffic SLI. Window-based smooths this: that hour is "some bad minutes" against a backdrop of mostly-good minutes, which is a fairer picture of a service that is mostly idle. Window-based also protects against the inverse: a brief, severe spike during an otherwise-busy period can get *diluted* in a request-based number if traffic was huge, where window-based would flag those minutes as clearly bad.

There is a sharp trade-off lurking here. Request-based can *hide a short, total outage* if traffic was low during it (few requests means few failures means small dent in the ratio). Window-based can *over-penalize a brief blip* in a quiet period (one bad minute is a big fraction of a few hundred minutes). Neither is universally right. The honest practice: default to request-based for high-volume user-facing services, and reach for window-based when traffic is spiky or low and you want every bad *minute* to count regardless of volume. Many mature SRE setups actually compute both and alert on whichever is more conservative. The point is to *choose deliberately* and write down which one your SLI uses, because the two can disagree by a lot during exactly the incidents you care about.

#### Worked example: when the two SLIs disagree by a full nine

Take a service that handles 10,000 requests per minute during the day and 50 requests per minute overnight. At 3am, a deploy breaks it completely for 10 minutes: every overnight request fails. That is 500 failed requests. Over the full 30-day month the service sees roughly 250 million requests, so request-based availability drops by only $500 / 250{,}000{,}000 = 0.0002\%$ — the SLI barely twitches, reading something like 99.97% when it would otherwise read 99.9702%. A request-based view essentially *shrugs off a total 10-minute outage* because it happened when few people were watching. Now compute it window-based: those 10 minutes are 10 fully-bad windows out of roughly 43,200 minutes in the month, a hit of $10 / 43{,}200 = 0.023\%$, dropping a window-based SLI from 100% to about 99.977%. That is a hundred times larger a dent for the *same* outage. Which is right? It depends on what you mean by reliable. If you care that the service was *entirely down for ten minutes* regardless of who noticed, window-based is honest. If you care about *total user-requests served*, request-based is honest. The lesson is not that one wins — it is that the *same* incident produces a 100× difference in the number depending on a choice many teams never consciously make. Make it consciously, write it in the spec, and you will never be ambushed by your own SLI during a postmortem.

## 9. Specifying an SLI rigorously: the spec template

A vague SLI is worse than no SLI, because it gives false confidence and two engineers will compute two different numbers from it and argue forever. The cure is a rigorous specification. Every SLI must pin down four things — numerator, denominator, measurement point, and window — precisely enough that any engineer would produce the identical number.

![A stack showing the derivation of an SLI specification from a user journey down through defining good, defining total, the measurement point, the window, and finally the good over total ratio](/imgs/blogs/choosing-slis-that-reflect-user-pain-6.png)

Here is the template I use. It is deliberately boring, because boring is unambiguous.

```yaml
sli:
  name: checkout-availability
  journey: "User completes checkout"
  # NUMERATOR — what counts as a GOOD event
  good: >
    HTTP requests to POST /api/checkout that return a 2xx or 3xx
    status code (a 5xx is bad; a 4xx caused by user input is EXCLUDED,
    see denominator)
  # DENOMINATOR — what counts as a VALID event at all
  total: >
    All HTTP requests to POST /api/checkout, EXCLUDING:
      - health/readiness probes to /healthz and /ready
      - synthetic monitoring traffic (User-Agent: synthetic-probe)
      - 4xx client errors caused by malformed user input (not our fault)
  # WHERE we measure
  measurement_point: "Layer-7 load balancer access logs (closest to user)"
  # OVER WHAT WINDOW
  window: "28-day rolling"
  unit: "ratio of good to total, expressed as a percent"
```

Walk through the four fields and the judgment each demands:

1. **Numerator (what counts as good).** Be explicit about the success criteria. For availability: "2xx or 3xx, and specifically not 5xx." For latency: "successful AND completed in under 300ms." The trap is sloppy boundaries — does a 3xx redirect count as good? (Usually yes, the user got where they were going.) Does a request that succeeded but took 9 seconds count as good for the *latency* SLI? (No.) Write it down.

2. **Denominator (what counts at all).** This is where the exclusions live, and it is the field people skip. Exclude health checks and synthetic traffic, as discussed. The harder judgment call is *4xx client errors*: a 404 or a 400 caused by the user sending garbage is arguably not *your* unreliability — you correctly rejected bad input. Most teams exclude user-caused 4xx from both numerator and denominator, but they *keep* 429 (rate-limited) and certain 4xx that indicate your own bug. There is no universal answer; the rule is to decide explicitly and document it, because this single choice can swing the SLI by a percentage point.

3. **Measurement point.** State it — "L7 load balancer access logs" — because, as section 7 showed, the number depends on where you stand. An SLI spec that does not say where it is measured is not reproducible.

4. **Window.** A rolling window (28 days is common, aligning to four weeks; 30 days also seen) over which the ratio is computed. The window length trades responsiveness against stability: a short window reacts fast but is noisy; a long window is stable but sluggish to recover after a bad incident. 28 days is a sane default for an SLO-backing SLI; you will compute *burn rate* over much shorter windows for alerting, which is a topic for the alerting posts.

With those four fields pinned, the SLI is a single unambiguous number, and you can hand it to the SLO. We carry this exact spec into [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something), where the target and error budget get attached.

## 10. The PromQL: measuring a latency SLI with histograms

Theory is cheap; let me show the artifact that actually computes this in production, because "fraction of requests faster than a threshold" sounds simple and has one beautiful trick that makes it both cheap and correct: **histograms**.

A naive approach would try to compute the p99 latency and check it against your threshold. That is fragile — percentiles do not average across instances, they are expensive to compute exactly, and p99 alone is not a good/total ratio. The histogram approach is better. You instrument your service to emit a **histogram** metric — Prometheus's histogram exposes cumulative buckets, each counting requests that finished *at or below* a latency boundary. If one of your bucket boundaries is exactly your latency threshold (say 0.3 seconds), then the count in that bucket *is* your "good" count, for free.

Suppose your service exports `http_request_duration_seconds_bucket` with a bucket boundary at `le="0.3"` (300ms) and `le="+Inf"` (the total). The latency SLI — the fraction of successful requests served in under 300ms — is exactly:

```promql
# Fraction of requests faster than 300ms, over the last 28 days.
# Numerator: requests that finished in <= 0.3s (the le="0.3" bucket count).
# Denominator: all requests (the le="+Inf" bucket count == total).
sum(rate(http_request_duration_seconds_bucket{job="checkout", le="0.3"}[28d]))
  /
sum(rate(http_request_duration_seconds_bucket{job="checkout", le="+Inf"}[28d]))
```

That ratio is your window-based latency SLI as a fraction between 0 and 1. No percentile estimation, no averaging-of-percentiles sin — just a count of fast requests over total requests, which is exactly the good/total shape an SLI wants. Multiply by 100 for a percent.

If you also want the classic *p99 latency number* for a dashboard panel (humans like seeing "p99 = 280ms" even though the SLI is the ratio above), `histogram_quantile` interpolates it from the same buckets:

```promql
# p99 latency in seconds, for a human-readable dashboard panel.
# This is NOT the SLI itself; the SLI is the fast/total ratio above.
histogram_quantile(
  0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{job="checkout"}[5m]))
)
```

The availability SLI is even simpler, since it is just good statuses over total. With a counter `http_requests_total` labeled by status code:

```promql
# Availability SLI: fraction of checkout requests that did NOT 5xx,
# over 28 days. We treat 5xx as "bad" and exclude health checks via
# the route label, mirroring the spec's denominator.
1 - (
  sum(rate(http_requests_total{job="checkout", route!="/healthz", code=~"5.."}[28d]))
    /
  sum(rate(http_requests_total{job="checkout", route!="/healthz"}[28d]))
)
```

To make these cheap to query and to back an SLO and alerts, you precompute them as **recording rules** so Prometheus evaluates them on a schedule and stores the result as a new, fast-to-read series:

```yaml
groups:
  - name: checkout_sli
    interval: 30s
    rules:
      # Window-based latency SLI: fraction served under 300ms.
      - record: sli:checkout_latency:ratio_rate28d
        expr: |
          sum(rate(http_request_duration_seconds_bucket{job="checkout", le="0.3"}[28d]))
            /
          sum(rate(http_request_duration_seconds_bucket{job="checkout", le="+Inf"}[28d]))
      # Availability SLI: fraction of non-5xx, health checks excluded.
      - record: sli:checkout_availability:ratio_rate28d
        expr: |
          1 - (
            sum(rate(http_requests_total{job="checkout", route!="/healthz", code=~"5.."}[28d]))
              /
            sum(rate(http_requests_total{job="checkout", route!="/healthz"}[28d]))
          )
```

Now `sli:checkout_availability:ratio_rate28d` is a single series you can graph, alert on, and subtract from your SLO to get the error budget. These recording rules are the practical bridge from "we defined an SLI" to "Prometheus computes it every 30 seconds." For the deeper treatment of how to model these metrics correctly — counters vs gauges vs histograms, cardinality, the `rate()` and `$__rate_interval` mechanics — see the planned metrics post `metrics-and-time-series-done-right`, and the system-design view of [observability with metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design).

## 11. Worked example: choosing SLIs for a checkout flow

Let me put the whole method to work on a real journey, end to end, including the rejection of the tempting bad SLI.

**The service.** An e-commerce checkout flow. A user clicks "place order," which hits `POST /api/checkout`, which validates the cart, reserves inventory, charges the card, and writes the order. It is the revenue path — when it breaks, the company loses money by the minute. This is a textbook critical user journey.

**Step 1 — start from the journey, not the components.** The journey is "user completes checkout." Internally it touches the cart service, the inventory service, the payment gateway, and the orders database. I am explicitly *not* going to put SLIs on each component. I am going to put SLIs on the journey, measured at the edge, because that is what answers "can the user actually buy the thing."

**Step 2 — pick from the menu.** Checkout is a request/response service, so the menu is availability, latency, and optionally quality. The user pain is "it failed" and "it was slow." I pick the two bedrock SLIs:

- **Availability**: fraction of checkout requests that succeeded.
- **Latency**: fraction of checkout requests served in under 1 second (checkout is a heavier flow than a simple read, so 1s is the threshold users tolerate before they start to worry the charge didn't go through).

I skip a quality SLI here because checkout does not degrade gracefully — it either places the order or it doesn't; there is no "partial checkout."

**Step 3 — specify precisely.** Now the spec template, filled in:

```yaml
sli:
  name: checkout-availability
  journey: "User completes checkout"
  good:  "POST /api/checkout returning 2xx (order placed)"
  total: >
    All POST /api/checkout requests, excluding /healthz probes,
    synthetic traffic, and 4xx caused by user input (invalid card
    field, empty cart). A 402 payment-declined is EXCLUDED from
    'total' because the bank declined, not us; a 5xx is bad.
  measurement_point: "L7 load balancer access logs"
  window: "28-day rolling"
---
sli:
  name: checkout-latency
  journey: "User completes checkout"
  good:  "POST /api/checkout that succeeded AND completed in < 1.0s"
  total: "All successful POST /api/checkout (same exclusions as above)"
  measurement_point: "L7 load balancer access logs"
  window: "28-day rolling"
```

Note the genuinely hard judgment baked into the availability denominator: a **402 payment-declined** is the bank's "no," not our outage. If we counted every declined card as "bad availability," our SLI would tank every time customers used expired cards, and it would *move when our service is perfectly fine* — failing the one test. So we exclude bank declines from the total. But we keep 5xx (our crash) and we keep 429 (we rate-limited a legitimate user, which *is* our reliability problem). This is exactly the denominator discipline from section 9, and it is the difference between an SLI that tracks our reliability and one that tracks the population's credit-card hygiene.

**Step 4 — reject the tempting bad SLI.** Someone on the team proposes: "let's also add a CPU < 80% SLI on the checkout boxes, so we know they're healthy." Here is the precise rejection: *CPU is a USE-method utilization signal; it belongs in capacity alerting, not in our SLO.* During peak shopping events the checkout boxes will run hot — that is them doing their job well, serving a flood of orders fast. A CPU SLI would burn our error budget hardest exactly during our biggest revenue moments, when users are happiest, and would page on-call during the one event where we most need them watching real user pain. The thing the CPU number is *trying* to protect against — "the boxes can't keep up and users start failing" — is already captured, correctly and directly, by the availability and latency SLIs, which will move the instant users actually fail. So we watch CPU on a *capacity dashboard* and set a *leading* alert ("CPU sustained > 85% for 30 min, scale up during the event"), and we keep it out of the SLO entirely. The bad SLI is rejected with reasoning, not vibes.

**The result we can measure.** With these two SLIs computed via the recording rules from section 10, checkout currently sits at 99.95% availability and 99.7% fast-enough — both comfortably above the 99.9% / 99% targets we will set in the SLO post. When a bad deploy ships a checkout regression, the availability SLI drops to 98.5% within minutes and the error budget burns visibly, paging on-call for *real user pain*. Before we did this — back when the only alert was "CPU > 90%" — that same regression page would have *never fired* (CPU was normal; the bug was a logic error), and we would have learned about the outage from angry customers. After: caught in minutes, by a metric that moved because users were unhappy. That is the entire value of choosing the right SLI, made concrete.

## 12. Worked example: SLIs for an async data pipeline

Now the case where availability and latency *do not even apply*, to show why the menu has to change with the service shape.

**The service.** A streaming pipeline consumes user-activity events off Kafka, enriches them, aggregates them into per-user features, and writes the features to a store that the recommendation system reads at request time. There is no synchronous user waiting on this. A user never sends a "request" to the pipeline. So asking "what fraction of requests returned 2xx" is meaningless — there are no requests. The pipeline's user pain is entirely different.

![A data pipeline where the ingest stage can stall and let lag grow while staying up, with freshness and completeness SLIs catching the stale data that availability misses before the served view goes wrong](/imgs/blogs/choosing-slis-that-reflect-user-pain-8.png)

**The failure that availability is blind to.** Suppose the enrichment job hits a poison message and gets stuck retrying it forever. The process is *up* — it is running, it is logging, its health check is green, its CPU is fine. Every infrastructure metric says "healthy." But it has stopped advancing: the Kafka consumer lag is climbing, and the feature store is now serving features computed from data that is two hours old. Downstream, the recommendation system is showing users recommendations based on what they did this morning, not what they are doing right now. Users feel it — the product feels *stale and dumb* — but every availability and uptime metric you have is bright green. This is the **stale-but-up** failure, and it is structurally invisible to availability. You need different SLIs.

**The menu for a pipeline:**

- **Freshness**: the fraction of time (or the fraction of served reads) where the data is less than 10 minutes old. Good = the most recent processed event is under 10 minutes behind real time. This is the SLI that screams the instant the job gets stuck, because the data age starts climbing immediately.
- **Completeness / coverage**: the fraction of input events that made it all the way to the output. Good = the event appears in the computed features. This catches the silent-drop failure, where a pipeline quietly discards 2% of events and nobody notices the data is subtly incomplete.

**Specifying the freshness SLI.** Freshness needs a clock and a "last successfully processed event timestamp." A common implementation exports a gauge with the timestamp of the newest event the pipeline has fully processed; freshness is then *now minus that timestamp*. The SLI is the fraction of time that lag is under the threshold:

```yaml
sli:
  name: features-freshness
  journey: "Recommendations reflect recent user activity"
  good:  "Data age (now - last_processed_event_timestamp) < 600 seconds"
  total: "All evaluation intervals during which the pipeline should be running"
  measurement_point: "Pipeline's last-processed-watermark gauge"
  window: "28-day rolling"
```

And the PromQL, where `pipeline_last_processed_timestamp_seconds` is the watermark gauge:

```promql
# Freshness gauge: current data age in seconds. The pipeline is
# "fresh" when this is under 600s (10 minutes). Alert/SLI on the
# fraction of time this stays under threshold.
time() - max(pipeline_last_processed_timestamp_seconds{job="feature-pipeline"})
```

A window-based freshness SLI then asks: over the 28-day window, what fraction of 1-minute evaluation intervals had data age under 600 seconds? When the poison message stalls the job, data age crosses 600s within ten minutes, the freshness SLI starts burning budget, and on-call gets paged for *the actual user pain* — stale recommendations — that no availability metric could ever have surfaced. The completeness SLI, meanwhile, runs as a reconciliation: periodically count input events in a window against output records and ratio them, catching the silent 2% drop that freshness alone would miss. Two SLIs, both phrased as good/total, both anchored to user pain, neither one an availability or latency metric — because the service shape demanded a different menu. That is the whole point of section 4 made real.

## 13. War story: how Google made the error budget arithmetic work — and what a bad SLI does to it

The canonical real-world case is Google's own error-budget model, documented in the SRE book, because it is the clearest illustration of *why the SLI must reflect user pain* — the entire model collapses if it doesn't.

Google's insight was organizational, not just technical. Developers want to ship features fast; SRE wants the service to stay reliable; these pull in opposite directions and, without a shared number, the argument is settled by whoever is loudest in the meeting. The error budget dissolves the conflict by turning "is it reliable enough?" into arithmetic: you pick an SLO (say 99.9%), the error budget is its complement (0.1% of requests are *allowed* to fail), and as long as the budget has room, developers ship freely; when it is exhausted, the team freezes features and fixes reliability until the budget recovers. No meeting, no politics — the *number* decides. We unpack this fully in the error-budget posts, but here is the part that matters for SLIs: **the entire mechanism is only as good as the SLI underneath it.** The error budget is computed directly from the SLI ($\text{budget remaining} = \text{SLO} - \text{SLI}$, roughly). If the SLI does not reflect user pain, the budget is measuring the wrong thing, and the whole elegant model becomes a precise way of optimizing for nonsense.

Picture it with the bad SLI from section 6. If a team had set a CPU-based SLI, their error budget would burn during peak traffic (high CPU) and recover during quiet hours (low CPU) — utterly decoupled from whether users were happy. They would freeze feature work during their busiest, healthiest periods and ship recklessly during quiet periods when a bug would do real damage. The error-budget model would be running perfectly, mechanically, while pointed at exactly the wrong target. The genius of Google's model is not the arithmetic; the arithmetic is trivial. The genius is *insisting the SLI measure user-perceived reliability* so that the budget the arithmetic protects is a budget users actually care about. Get the SLI right and the budget aligns the org around user happiness. Get the SLI wrong and the budget aligns the org around a phantom.

There is a second, humbler real-world lesson worth naming: many famous outages were *invisible to the dashboards in the room* precisely because those dashboards watched resources, not user journeys. The classic shape — repeated across countless public postmortems — is a config push or a bad deploy that breaks a user-facing path while every host-level metric stays green, so the team chases CPU and memory graphs for the first painful minutes while users are down and the actual signal (the success rate of the affected journey) was never on a screen. The fix in every one of those postmortems rhymes: *put an SLI on the user journey and alert on it*, so the next time the dashboard turns red the instant users hurt, not twenty minutes later when the tickets pile up. For the architecture-time analysis of how these outages unfold, see the system-design study of [anatomy of an outage and lessons from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems); for the disciplined way to find root cause once you are in one, see [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging).

## 14. Stress-testing your SLI choices

A choice you have not stress-tested is a choice you do not understand. Let me push the checkout and pipeline SLIs against the hard cases, because the edge cases are where bad SLIs reveal themselves.

**What if a dependency is down for two hours?** Say the payment gateway has an outage. Our checkout availability SLI will immediately reflect it — checkout requests 5xx because we cannot charge cards, the SLI drops, the budget burns, on-call is paged for genuine user pain (users cannot buy). This is *correct* — it is exactly when we want to know. The subtlety: should a *third-party's* outage burn *our* budget? The honest answer is yes for the SLI (the user does not care whose fault it is — they still couldn't check out), but you may track a separate "errors excluding upstream outages" view for the postmortem to assign causes fairly. The user-facing SLI stays user-facing; the cause attribution is a separate concern. Designing the fallback so a dependency outage does *not* take you down is the resilience work covered in [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — but the SLI's job is just to tell the truth about user pain, and it does.

**What if traffic is very low at 4am?** Our request-based checkout SLI gets noisy when only a handful of orders come in overnight — five failed orders out of ten in a quiet hour reads as 50% availability and could swing the 28-day number or trip a naive alert. This is precisely the case section 8 warned about, and it is why for low-traffic windows we either lean on the window-based variant or compute the SLI over a long enough rolling window that a handful of overnight failures cannot dominate. The stress test reveals which SLI flavor we should have picked.

**What if the pipeline is *up* but completely wrong?** The poison-message stall is caught by freshness. But what if the job runs fine, stays fresh, yet a bad code deploy makes it compute *wrong* features — fresh garbage? Freshness is green, completeness is green, and users get current-but-incorrect recommendations. This is the gap that the *correctness* SLI fills, and the stress test is exactly how you discover you needed a third SLI. No single SLI is complete; the stress test tells you where the next one goes.

**What if two journeys fail at once, or the budget is already spent?** SLIs are per-journey, so two failing journeys give you two burning budgets — and that is a feature, because it tells you *both* things are wrong and lets you triage by which user pain is worse (checkout down beats search-suggestions down). And if the budget is already spent before the incident? Then your SLO was either too tight or your reliability work is overdue — but the SLI still faithfully reports current pain regardless of budget state. The SLI measures reality; the budget is a policy layered on top. Keeping those two ideas distinct is what lets you reason clearly when everything is on fire at once.

## 15. How to reach for this (and when not to)

SLIs are powerful, but like every practice they have a cost, and part of being a principal-level operator is knowing when *not* to add one.

**Do** put SLIs on your top critical user journeys — the revenue path, the core read path, the sign-up flow. **Do** pick availability and latency as your bedrock pair for request/response services, freshness and completeness for pipelines, durability and availability for storage. **Do** specify each one with the four-field template so it is reproducible. **Do** measure at the load balancer or client. **Do** keep the count small — two SLIs on each of three-to-five top journeys is plenty; that is your handful.

**Do not** create an SLI for every internal component — you will drown in numbers nobody can act on, and component SLIs systematically understate journey pain. **Do not** elevate a resource metric (CPU, memory, disk) to an SLI — file those as capacity/USE signals with *leading* alerts, never in the SLO. **Do not** chase a fourth or fifth SLI on a journey when two already capture the pain — each SLI is a commitment with a budget and a page attached, and more is not better; more is noise. **Do not** add a quality or correctness SLI for a service that has no degraded mode or no correctness risk — you would be measuring a failure that cannot happen. And **do not** over-specify the threshold to false precision: if you genuinely do not know whether the latency threshold should be 250ms or 300ms, pick one, ship it, and let real user-behavior data (the latency at which conversion or engagement actually drops) move it — an approximately right SLI in production beats a perfectly debated one in a doc.

The meta-rule: an SLI earns its place only if a human will *act* on it. If it goes red and the right response is "huh, weird" and a shrug, it was never an SLI — it was a dashboard panel wearing a costume. Reserve the word SLI for the two or three numbers per journey that, when they move, mean a user is hurting and someone should do something. Everything else is observability, and observability is good, but it is not the contract.

There is also a question of *when in a service's life* to add SLIs, and the honest answer is: later than you think for a brand-new service, earlier than you think for one carrying real traffic. A prototype with ten internal users does not need a formal SLI program — you would be specifying numerators and rolling windows for a service whose entire user base could be polled in a Slack channel. Adding the ceremony there is its own kind of toil. But the moment a service is on the revenue path or a real customer depends on it, the SLI is overdue, because that is the moment an outage costs something and the moment you need a number to tell you about it before the customer does. The trigger is not "the service exists"; the trigger is "a human outside the team will be hurt if this breaks." When that becomes true, walk the four steps from the checkout example — start at the journey, pick from the menu, specify the four fields, reject the resource-metric temptation — and you will have your two or three SLIs in an afternoon.

A final word on *evolving* an SLI, because they are not carved in stone. The first version of an SLI is a hypothesis about what user pain looks like, and production will teach you where it was wrong. Maybe the latency threshold you guessed at 300ms turns out to be the wrong cliff — real conversion data shows users abandon at 500ms, not 300ms, so the threshold moves. Maybe you discover a whole class of failure (the degraded-but-200 response) that your availability SLI was blind to, so you add a quality SLI. Maybe a journey you thought was critical turns out to be barely used, and one you ignored turns out to be where the money is. Treat the SLI set as a living artifact: review it quarterly, move it when the data argues for it, and never let "we've always measured it this way" outweigh "this number no longer reflects what users feel." The discipline is not picking the perfect SLI on day one — it is keeping the SLI faithful to user pain as the service, and the users, change underneath it.

## 16. Key takeaways

- **The one test:** a good SLI moves when users are unhappy and stays still when they are fine. Run every candidate metric through it; most resource metrics fail it in both directions.
- **An SLI is a ratio of good events to total valid events**, anchored to user-felt outcomes, living between 0% and 100% — the exact input your SLO and error budget need.
- **The four golden signals** (latency, traffic, errors, saturation) are your starting menu; latency and errors make great SLIs, traffic is context, and saturation is a capacity signal that masquerades as an SLI.
- **Build SLIs from RED** (Rate, Errors, Duration — per service) and **build capacity alerts from USE** (Utilization, Saturation, Errors — per resource). Reaching for USE when you needed RED is the root of most bad SLIs.
- **The menu changes by service shape:** request/response wants availability + latency; pipelines want freshness + completeness; storage wants durability + availability. Availability is structurally blind to a stale-but-up pipeline.
- **Pick SLIs on critical user journeys, not internal components** — answer "can users do the thing they came to do?" Component SLIs multiply to understate real journey pain.
- **CPU, memory, and disk are saturation signals**, useful for capacity and leading alerts but terrible SLIs: a box at 90% CPU serving every request in 50ms is fine, and an SLI on it would page you for nothing.
- **Measure as close to the user as you can** (load balancer or client), and **get the denominator right** — exclude health checks and synthetic traffic, decide deliberately on user-caused 4xx and third-party declines.
- **Specify rigorously:** numerator (what is good), denominator (what counts at all), measurement point, and window — precise enough that two engineers compute the identical number.
- **Choose request-based for steady high traffic and window-based for bursty or low traffic** — and write down which you used, because they disagree during exactly the incidents you care about.

## 17. Further reading

- *Site Reliability Engineering* (the Google SRE Book), Chapter 4 "Service Level Objectives" and Chapter 6 "Monitoring Distributed Systems" — the source of the four golden signals and the good-events-over-total SLI definition.
- *The Site Reliability Workbook*, Chapter 2 "Implementing SLOs" — the practical SLI specification template, the good/total ratio, request-based vs window-based SLIs, and worked menus by service type.
- Tom Wilkinson, "The RED Method" — Rate, Errors, Duration as the request-centric monitoring menu for services.
- Brendan Gregg, "The USE Method" — Utilization, Saturation, Errors as the resource-centric menu for systems, and why it is for capacity and debugging rather than user-facing SLIs.
- The Prometheus documentation on histograms and `histogram_quantile`, and on recording and alerting rules — the mechanics behind the latency-SLI PromQL above.
- Within this series: [reliability is a feature — the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) (the loop this post starts), [SLI, SLO, SLA: the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter), and [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) (where these SLIs get their targets and error budgets). The planned metrics post `metrics-and-time-series-done-right` covers counters, gauges, histograms, and `rate()` in depth.
- Cross-cutting: the system-design treatments of [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) and [observability with metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design).

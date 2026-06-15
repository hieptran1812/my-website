---
title: "From a Vague Ask to Real Requirements: Functional, Non-Functional, and SLOs"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a senior turns a one-line product ask into a crisp requirements document, defines SLIs, SLOs, and SLAs with real error-budget math, and lets those numbers drive every architecture decision that follows."
tags:
  [
    "system-design",
    "requirements",
    "slo",
    "sla",
    "error-budget",
    "availability",
    "architecture",
    "distributed-systems",
    "scalability",
    "reliability",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/turning-vague-asks-into-requirements-and-slos-1.webp"
---

A product manager drops a message in your channel: "We want to add a social feed to the app, and it should feel instant and never go down." A junior engineer reads that and starts drawing boxes — a feed service, a database, maybe a cache because instant means cache, right? Three weeks later they demo something that works on a laptop with twelve rows in the table, and the first time real traffic hits it, the "instant" feed takes four seconds and the "never goes down" service falls over because nobody ever asked how many users, how fresh the reads needed to be, or what happened when a write was lost.

The senior reads the exact same message and sees something completely different. They see a sentence with three adjectives — *social*, *instant*, *never goes down* — and they know that adjectives are not requirements. Adjectives are the absence of requirements. The entire job, before a single box is drawn, is to convert each of those words into a number that someone is willing to sign their name under. "Instant" becomes "p95 read latency under 200 milliseconds." "Never goes down" becomes "99.9% availability, which is a 43-minute monthly downtime budget, and here is what each extra nine would cost you." "Social feed" becomes a list of capabilities with a read-to-write ratio and a freshness tolerance attached to each one.

![A taxonomy tree of requirement types splitting into functional and non-functional branches with sub-dimensions](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-1.webp)

The figure above is the map for this whole article. A requirement splits into two branches. The functional branch — *what the system does* — is the one juniors capture, because it is the one the product manager talks about. The non-functional branch — *how well it does it* — is the one that actually decides your architecture, and it is almost always missing from the ask. By the end of this post you will be able to take any one-line product request and walk out of the room with a requirements document that has functional capabilities, quantified non-functional dimensions, a defined set of SLIs and SLOs with an explicit error budget, a capacity and cost envelope, and a clear record of what you said no to and why. You will also be able to defend every one of those numbers in a design review, which is the difference between a requirements doc that holds up and one that gets relitigated every sprint.

This is the second post in the **System Design, Like a Senior** series. The first, [how seniors approach ambiguous system design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems), covered the overall loop — clarify, constrain, sketch, stress-test, iterate. This post zooms all the way into the *constrain* step, because that is where most designs are silently won or lost.

## 1. Why juniors capture only the functional half

Let me be precise about the failure mode, because naming it correctly is half of avoiding it. When a product owner describes a feature, they describe behavior: the user opens the feed, sees posts from people they follow, can like and comment, and the newest posts appear at the top. Every word of that is functional. It describes a function from inputs to outputs. A junior engineer is reward-shaped to translate behavior into code, so they hear behavior and immediately start thinking about endpoints, schemas, and UI states. That instinct is not wrong — those things have to exist — but it is dangerously incomplete, because the behavior description contains zero information about the conditions under which that behavior must hold.

Consider the sentence "the newest posts appear at the top." Functionally trivial: sort by timestamp, descending. But the senior's brain immediately spawns a half-dozen questions that have nothing to do with sorting and everything to do with whether the system is buildable at all. How soon after I post does it need to appear in *my* feed? In my *followers'* feeds? Within a second? Within a minute? Is it acceptable that a friend on the other side of the world sees my post thirty seconds later than a friend in the same city? When two posts arrive in the same millisecond, does it matter which one wins, or is any consistent order fine? If we lose a post during a server crash — it was accepted but never persisted — is that a minor annoyance or a trust-destroying catastrophe?

None of those questions are answered by the functional description, and every one of them changes the architecture. The first question is a *freshness* requirement and it decides your caching strategy. The second is a *consistency* requirement and it decides whether you can serve reads from asynchronous replicas. The last is a *durability* requirement and it decides your replication factor and whether writes must be acknowledged by a quorum before you return success. The functional spec is the same regardless of how you answer those — but the systems you would build to satisfy each answer are wildly different in cost, complexity, and failure behavior.

This is why the senior treats the functional spec as the *easy* part, often the part you can write in twenty minutes, and spends the bulk of the requirements-gathering effort on the non-functional dimensions. The functional spec tells you what to build. The non-functional spec tells you whether what you build will survive contact with reality.

### A quick functional pass, done right

Before we leave the functional half, do it properly, because a sloppy functional spec hides non-functional landmines. The technique I use is to write each capability as a single sentence in the form *actor → action → object → outcome*, and then immediately annotate each one with a rough read/write classification and an importance tier. For the feed:

| Capability | Type | Tier | Hidden NFR it implies |
| --- | --- | --- | --- |
| User posts an item to their followers | Write | Core | Durability — losing a post is unacceptable |
| User reads their home feed | Read | Core | Latency + freshness — defines "instant" |
| User likes a post | Write | Core | High volume, low value per event |
| User comments on a post | Write | Core | Ordering matters within a thread |
| User searches posts | Read | Secondary | Staleness tolerance is generous |
| User sees a like count | Read | Secondary | Approximate is fine |

The right-hand column is the entire point. You are not just listing features; you are using each feature to *surface* the non-functional requirement it secretly depends on. A like count that can be approximate is a completely different engineering problem from a bank balance that cannot. The functional pass is a hunting expedition for hidden non-functional requirements, and if you finish it with an empty right-hand column you did it wrong.

## 2. The non-functional dimensions that actually drive architecture

Here is the senior's checklist — the non-functional dimensions that, in my experience, determine 90% of the architecture. For each one I will give you the junior's vague version and the number you replace it with. The discipline is simple and relentless: *every adjective becomes a number*.

**Latency.** Junior: "it should be fast." Senior: latency is a distribution, not a single value, so you specify percentiles. p50 (the median) tells you the typical experience; p95 and p99 tell you the tail, which is where users actually feel pain because the slowest requests are disproportionately the ones that involve the most data or the unluckiest cache misses. "Fast" is meaningless. "p50 under 50ms, p95 under 200ms, p99 under 500ms, measured at the server, for the home-feed read endpoint" is a requirement. Note that you must specify *which endpoint* and *where measured* — server-side and client-side latency differ by the network round trip, and the feed read and the post write have different budgets.

**Throughput.** Junior: "lots of users." Senior: requests per second at peak, and the peak-to-average ratio. A system averaging 5,000 QPS but peaking at 50,000 QPS during a morning rush is a 10× different sizing problem from one that is flat at 5,000. You also separate read QPS from write QPS, because the read/write ratio decides whether you optimize for read replicas and caching or for write throughput and partitioning.

**Availability.** Junior: "never goes down." Senior: a target expressed in nines, translated into a monthly downtime budget, *with the cost of each extra nine made explicit*. We will spend a whole section on this because it is where the most money gets wasted.

**Durability.** Junior: doesn't think about it. Senior: the probability that committed data survives. Durability and availability are different — a system can be available (serving requests) while having lost data, and it can be durable (lost nothing) while being unavailable. You specify durability as "we tolerate zero acknowledged-write loss" or, more honestly, "11 nines of durability," which is the order of magnitude cloud object stores quote and means you expect to lose roughly one object per ten million stored per ten thousand years.

**Consistency.** Junior: assumes everything is instantly consistent everywhere, because that is how a single laptop database behaves. Senior: specifies, per read path, how stale a read may be and whether a user must always see their own writes. This is the requirement that most directly drives the CAP and PACELC choice, and we will connect it to [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) explicitly.

**Scalability.** Junior: "should scale." Senior: defines the growth curve and the planning horizon. Designing for today's load plus 5× headroom over an 18-month horizon is a concrete target. "Should scale" is a wish; "should hold p95 under 200ms as we go from 1M to 10M daily active users without a re-architecture" is a requirement you can test against a design.

**Cost.** Junior: ignores it, then is shocked by the cloud bill. Senior: treats the dollar figure as a first-class non-functional requirement and derives a per-request cost. If a feature must serve a billion reads a month and the business can only afford to spend \$2,000 a month on it, you have a hard ceiling of \$0.000002 per read, and that number alone eliminates whole categories of design.

**Security and compliance.** Junior: "we'll add auth." Senior: specifies authentication, authorization model, encryption in transit and at rest, audit requirements, and — critically — *data residency*. If your users include EU residents, GDPR may require their data to stay in-region, and that single requirement can force a multi-region partitioning strategy that doubles your infrastructure. Compliance requirements are non-negotiable and architecture-defining, and they are almost never in the original ask.

The senior's superpower is not knowing exotic algorithms. It is having this checklist memorized and running it, ruthlessly, against every vague ask, replacing each adjective with a number before any box is drawn.

### Why latency is a distribution, not a number — and why the tail bites

I want to dwell on latency specifically, because it is the dimension where the gap between the junior's intuition and reality is widest. The junior thinks of latency as a single number — "the API responds in 80ms" — because that is what they see when they hit the endpoint once from their laptop. But a real service handling thousands of requests per second produces a *distribution* of latencies, and the shape of that distribution is what determines the user experience. The median (p50) might be a comfortable 40ms, while p99 is 600ms and p99.9 is two full seconds. Those slow requests are not noise to be ignored; they are real users having a bad time, and at scale there are a lot of them.

Here is the arithmetic that makes the tail matter more than people expect. Suppose a single feed render requires the backend to make ten internal service calls — fetch the follow graph, fetch recent posts from several shards, fetch like counts, fetch the user's read state, and so on — and the response cannot be assembled until all ten return. If each internal call independently has a p99 of 100ms (meaning a 1% chance of exceeding 100ms), then the probability that *all ten* come back under 100ms is 0.99 raised to the tenth power, which is about 0.904. So roughly 1 minus 0.904, or **9.6%** of feed renders will have at least one slow internal call and exceed 100ms. Your dependencies' p99 became your service's p90. This is the fan-out tail-amplification effect, and it is why services that fan out to many backends obsess over tail latency: a backend's rare slow case becomes the parent's common slow case once you multiply the fan-out. It is also why the requirement must specify the *percentile that carries the promise* — promising "p99 < 100ms" on a service that fans out to ten p99-100ms backends is promising something the math says you cannot deliver without first fixing the backends' tails.

The practical consequence for the requirements doc is twofold. First, you specify multiple percentiles, not one: a strict bar on p50/p95 (the typical experience) and a looser but still-bounded ceiling on p99 (the tail you'll tolerate). Second, you specify latency *per endpoint*, because a feed read and a post write and a search query have completely different distributions and completely different user tolerance. A user staring at a spinner waiting for their feed has a tight budget; a user who just tapped "post" and got an optimistic UI update has a much looser one, because the write can complete in the background. Lumping them into one "API latency" SLO either over-constrains the write or under-constrains the read. The senior writes a latency requirement *per user-facing operation*, each with its own percentiles, each justified by what the user is actually experiencing at that moment.

#### Worked example: translating "the feed should feel instant" into numbers

Let me do the translation that the title of this section promises, because "feels instant" is the canonical vague latency ask and turning it into numbers is a teachable procedure. Start with what we actually know about human perception, which is well-studied: responses under roughly 100ms feel instantaneous (the user perceives no delay); responses under roughly 1 second keep the user's train of thought unbroken but feel like the system is working; beyond about 10 seconds the user mentally checks out and may abandon the task. The feed render is a foreground, blocking interaction — the user is looking at a spinner and waiting — so "feels instant" maps to the under-100ms-to-under-1-second band, leaning toward the fast end because the feed is the app's most-used screen and slowness there colors the whole product.

Now turn that perceptual band into percentiles, accounting for the fan-out math above. We want the *typical* render to feel instant, so we put a hard bar at **p95 < 200ms** measured server-side (allowing roughly 50ms more client-side for the network round trip, landing the typical user comfortably under the 250ms-or-so threshold where delay starts to register). We accept that the tail will be slower because of the ten-way fan-out, so we set a **p99 ceiling of 400ms** — still under the one-second flow-breaking threshold, but acknowledging the math that says we cannot hold p99 to 200ms without enormous over-investment in every backend's tail. We deliberately do *not* chase p99.9; at the feed's scale a one-in-a-thousand 800ms render is invisible to the product and not worth the cost of eliminating.

So "the feed should feel instant" becomes, precisely: **p50 < 50ms, p95 < 200ms, p99 < 400ms, measured server-side at the feed-read endpoint, over a rolling 28-day window.** That is four numbers and a measurement context, each traceable to a fact about human perception and a fact about the system's fan-out structure. It is testable — you can measure it and find out you're failing. It is defensible — you can explain in a review why p95 carries the strict promise and p99 gets a looser one. And it is *bounded* — it tells you exactly how much over-provisioning is justified (enough to hold p95 at 200ms) and how much is waste (anything spent chasing p99 below 400ms). That is what it means to translate an adjective into a requirement.

## 3. The same product, four different requirement profiles

Here is the insight that separates a senior requirements doc from a junior one: **a single product does not have a single non-functional profile**. Different features have radically different requirements, and trying to satisfy all of them with one global SLO is the single most expensive mistake I see teams make. They pick "99.99% availability and strong consistency" for the whole product because the payment flow needs it, and then they pay 99.99%-and-strong prices for the like counter, which would be perfectly happy at 99.5% and eventual.

![A matrix of four product features against their required consistency, availability, and latency profiles](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-2.webp)

Look at the matrix. The payment write needs strong consistency (you cannot double-charge or lose a charge), high availability (99.99%, because a failed payment is lost revenue and lost trust), and can tolerate relatively relaxed latency (300ms is fine — nobody abandons a purchase over a quarter second). The feed read is the opposite shape: eventual consistency is fine (a five-second-stale feed is invisible to users), availability of 99.9% is plenty, but latency must be tight (100ms p99) because the feed must "feel instant." The like counter is the most relaxed across the board — eventual, 99.5%, 150ms — because a wrong count for a few seconds harms no one. The search index can be the stalest thing in the system; nobody expects a post to be searchable the instant it is created.

If you give all four the payment write's profile, you over-build three of them by an order of magnitude. If you give all four the like counter's profile, you under-build the payment flow and eventually lose someone's money. The senior's move is to *partition the product into requirement profiles* and design each part to its own SLO. This is not over-engineering — it is the opposite. It is refusing to pay for reliability that the business does not need on three-quarters of the system so you can afford it on the quarter that does.

This per-feature partitioning is also what makes the architecture legible later. When you decide the feed can read from asynchronous replicas while payments cannot, you can point at this matrix and say "because the feed's consistency requirement is *eventual* and the payment's is *strong*." The matrix is the justification, written down, before anyone argues about it in a review. We go much deeper on how those consistency choices cascade into CAP and PACELC trade-offs in [cap theorem and pacelc](/blog/software-development/database/cap-theorem-and-pacelc) and in the upcoming [articulating tradeoffs cap pacelc and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond); here the point is narrower: the requirement profile *precedes* and *determines* that trade-off. You do not pick CAP first. You pick it because the requirement told you to.

## 4. SLI, SLO, SLA: the three words people use interchangeably and shouldn't

These three acronyms get mangled constantly, including by people who should know better, so let me define them with the precision a senior uses, because the distinctions are load-bearing.

An **SLI** — Service Level Indicator — is a *measurement*. It is a number you actually compute from real traffic. The canonical form is a ratio of good events to total events: the fraction of requests that completed successfully under 200ms, the fraction of minutes the service was reachable, the fraction of writes that were durably persisted. An SLI is a fact about the past. It is not a goal; it is a thermometer.

An **SLO** — Service Level Objective — is a *target* for an SLI. It is the line you draw and say "the SLI must stay above this." "99.9% of feed reads complete under 200ms over a rolling 28-day window" is an SLO. The SLO is an internal engineering commitment. It is the number that decides whether you ship the next feature or stop and fix reliability. It should be *achievable but not trivially so* — set it too low and it gives you no protection; set it to 100% and you have made an impossible promise that will be violated the first time a single request times out.

An **SLA** — Service Level Agreement — is a *contract* with an external party, almost always with financial or legal consequences for breach. "If availability drops below 99.5% in a calendar month, the customer gets a 10% credit" is an SLA. The crucial senior insight is that **your SLA should always be weaker than your SLO**. You promise customers 99.5% (the SLA) but target 99.9% internally (the SLO), so that you have a comfortable margin before you ever owe anyone money or breach a contract. If your SLA and SLO are the same number, you will be paying out credits constantly, because real systems hover around their target, not safely above it.

![A pipeline showing the relationship from measured SLI to internal SLO target to external SLA promise with a safety gap](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-3.webp)

The pipeline above shows the safety gap. You measure the SLI continuously. You set the SLO above the SLA. The space between SLO and SLA is your buffer — the room to have a bad week without breaching a contract. Many systems have an internal SLO and no external SLA at all, which is fine; not everything is a paid product with contractual guarantees. But you should never have an SLA without a stricter internal SLO behind it, because that is a promise with no margin.

One more distinction that trips people up: an SLO is not the same as an *alert threshold*. You alert when you are *on track to miss* the SLO — when your error budget is burning too fast — not when you have already missed it. Alerting at the SLO boundary means you find out about reliability problems only after you have already broken your commitment, which defeats the purpose. The SLO defines the budget; the alerts watch the *burn rate* of that budget. More on burn rate shortly.

### Choosing a good SLI is harder than it looks

The hardest part of this triad is not the SLO target — that comes from the business need — it is defining the SLI so that it actually measures what users experience. A badly chosen SLI gives you a number that looks healthy while users suffer, which is worse than no SLI at all because it actively lies to you. There are a few traps worth naming.

The first trap is **measuring at the wrong place**. If you measure availability by polling your service from inside your own datacenter, you will miss every problem that lives between the user and your front door — DNS failures, CDN outages, a misconfigured load balancer that returns 200 to health checks but errors to real traffic. The user-experienced SLI should be measured as close to the user as you can manage, which usually means at the load balancer or edge, counting real user requests, not synthetic probes from a friendly network location. Synthetic probes are a useful supplement but a dangerous primary signal, because they don't experience what users experience.

The second trap is **the wrong aggregation**. Averages lie about latency. If you report "average latency was 90ms" you have hidden the fact that 5% of requests took 800ms, because the average smears the tail into the bulk. SLIs for latency must be percentile-based — "the fraction of requests under 200ms" — never average-based. The same goes for availability: a daily average of 99.95% can hide a complete 30-minute outage that happened to fall on a low-traffic day, which is why the SLI should be computed over a *request-count* basis (good requests / total requests) rather than a *time* basis when traffic is uneven. Request-based SLIs naturally weight the outage by how many users it actually affected.

The third trap is **counting the wrong events as good**. Earlier I noted that the latency SLI should only consider *successful* requests — a request that errors out in 5ms should not count as a "fast" success. But the subtler version is what counts as success at all. A request that returns a 200 with an empty or corrupt body is technically available but functionally broken; if your SLI only checks the HTTP status code, you'll report perfect availability while serving garbage. Good SLIs are defined in terms of the *outcome the user cares about* — "the feed loaded with content" — which sometimes requires instrumenting the response body or the client, not just the status line. This is more work, and most teams start with status-code-based SLIs and refine toward outcome-based ones as they mature. The point for the requirements doc is to be honest about what your SLI does and does not capture, and to note the gap so you can close it later.

## 5. The error-budget math everyone gets wrong in their head

This is the part where I am going to make you do arithmetic, because the entire discipline of SLOs rests on one piece of math and most engineers carry a wrong intuition about it. The error budget is the *complement* of the SLO. If your availability SLO is 99.9%, your error budget is 0.1% — that is the fraction of the time you are *allowed* to be down. The whole trick is converting that percentage into a wall-clock number that humans can feel.

A 30-day month has 30 × 24 × 60 = 43,200 minutes. An SLO of 99.9% means you are allowed to be down 0.1% of that, which is 43,200 × 0.001 = **43.2 minutes per month**. That is the famous number. Internalize it: three nines is roughly 43 minutes a month, which is about one moderately bad incident. Not one per quarter — one per *month*. People dramatically overestimate how much downtime 99.9% permits; it is less than a single long lunch.

Now watch what each additional nine does. It does not subtract a fixed amount; it *divides by ten*.

![A matrix mapping availability nines to monthly downtime budget and relative cost to achieve it](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-4.webp)

| Availability SLO | Error budget | Downtime / month | Downtime / year |
| --- | --- | --- | --- |
| 99% (two nines) | 1% | 7.2 hours | 3.65 days |
| 99.9% (three nines) | 0.1% | 43.2 minutes | 8.76 hours |
| 99.99% (four nines) | 0.01% | 4.32 minutes | 52.6 minutes |
| 99.999% (five nines) | 0.001% | 25.9 seconds | 5.26 minutes |

Stare at the right two columns. Going from two nines to three nines takes you from "you can be down for the better part of a workday each month" to "one incident a month." Going from three to four takes you to *four minutes a month* — which means you cannot have a human in the loop for recovery, because by the time a pager fires, a person reads it, opens a laptop, and SSHes in, four minutes are gone. Four nines forces automated failover. Five nines — 26 seconds a month — means even your automated failover has to be near-instant and you essentially cannot deploy risky changes during business hours.

The cost column in the figure is the part juniors never see. Each nine roughly triples to ten-times your cost, because it forces a qualitatively harder architecture: two nines needs a single well-run server; three nines needs redundancy and health checks; four nines needs multi-AZ automated failover and no single points of failure; five nines needs multi-region active-active, which means solving cross-region consistency, which is one of the genuinely hard problems in distributed systems. The jump from three nines to five nines is not 2× the work — it can be 10× to 30× the cost and a permanent on-call burden. We will return to this as the central optimization lesson of the post.

#### Worked example: deriving SLOs and an error budget for the feed service

Let me make this concrete. We are building the home-feed *read* service. From the requirements interrogation (next section) we have established: the business wants the feed to "feel instant," cares about retention, and is a consumer app where a brief outage is annoying but not financially catastrophic. Let me derive real SLOs.

First, **latency**. Research on perceived performance says interactions under about 100ms feel instant and under about 1 second keep the user's flow unbroken. The feed read is a foreground, blocking interaction — the user is staring at a spinner — so we want it to feel instant. I set: **p50 < 50ms, p95 < 200ms, p99 < 400ms**, measured server-side at the feed read endpoint. Why p95 and not p99 for the "feels instant" bar? Because at the feed's scale, p99 will occasionally be blown by a cold cache or a slow replica, and chasing p99 < 100ms would force enormous over-provisioning for marginal user benefit. We hold p95 to the instant bar and give p99 a looser but still-bounded budget. This is itself a requirement decision — *which percentile carries the strict promise* — and it should be written down.

Second, **availability**. This is a consumer feed, not a payment system. If the feed is down for 40 minutes once a month, users are annoyed but no money is lost and no trust is permanently destroyed. The business need maps to **99.9%**, giving us a **43.2-minute monthly error budget**. We explicitly reject 99.99%: it would require multi-AZ automated failover for the whole read path and roughly 10× the infrastructure cost, to save 39 minutes of monthly downtime that no user will remember. That rejection, written in the doc with its cost, is the senior move.

Third, the **SLI definitions** that make those SLOs measurable. Availability SLI: the fraction of feed-read requests that return a 2xx or 3xx status (not 5xx) over a rolling 28-day window, measured at the load balancer. Latency SLI: the fraction of successful feed-read requests served under 200ms server-side. Note that latency SLI is conditioned on *successful* requests — you do not want a request that fails fast in 5ms to count as a "fast" success, so the latency and availability SLIs are tracked separately and the latency one only looks at the requests that actually succeeded.

So the feed read service requirement, in one block:

```yaml
# Feed read service — SLO definitions
slo:
  availability:
    objective: 99.9%            # 43.2 min/month error budget
    window: 28d_rolling
    sli: ratio(non_5xx_responses / total_requests)
    measured_at: load_balancer
  latency:
    objective: 95% < 200ms      # p95 carries the "feels instant" promise
    p99_ceiling: 400ms          # looser tail bound
    window: 28d_rolling
    sli: ratio(successful_requests_under_200ms / successful_requests)
    measured_at: server
error_budget:
  monthly_minutes: 43.2
  policy: freeze_features_if_budget_exhausted
```

That block is buildable, alertable, and defensible. Every number traces back to a business need we can articulate. That is a requirements artifact; "the feed should feel instant and never go down" is not.

## 6. Interrogating stakeholders to surface the hidden requirements

The numbers in that worked example did not fall from the sky. They came from a structured interrogation of the people who made the request. This is a *skill*, and it is one of the highest-leverage things a senior does, because the requirements that sink projects are almost always the ones nobody thought to state. The product manager who asked for the feed is not hiding the durability requirement from you maliciously — it genuinely never occurred to them that a post could be silently lost, because to them the database just works.

Your job is to ask the questions that surface those buried requirements. Over the years I have collected a set of questions that reliably extract the non-functional dimensions, phrased the way you actually ask them in the room — not in jargon, but in consequences the stakeholder can reason about.

**To surface freshness/consistency:** "After someone posts, how soon does it *have* to show up in their followers' feeds — instantly, within a second, within a minute? Is it okay if it shows up for some friends a few seconds before others?" Notice I never say "eventual consistency." I describe the observable behavior and let them tell me their tolerance. Their answer — "oh, a few seconds is totally fine" — is a freshness SLO that just unlocked asynchronous replication and aggressive caching.

**To surface durability:** "Say someone hits post, the app says 'posted,' and then a server crashes and that post is gone forever. How bad is that — a shrug, or a support ticket, or a news story?" For a feed post, usually "annoying, they'd repost." For a payment, "that's a catastrophe, we can never do that." That one question is the difference between fire-and-forget writes and quorum-acknowledged durable writes, and it changes your replication design entirely.

**To surface the real availability need:** "If this is down for ten minutes during peak hours, what actually happens? Who notices, who calls, and does the company lose money?" The honest answer for most features is "users are annoyed and it recovers." The number of times the answer is genuinely "we lose thousands of dollars a minute" is far smaller than the number of times someone *initially asks* for five nines. Always price the requested availability against the real cost of being down, because the gap between the two is where you save the company a fortune.

**To surface scale and growth:** "How many users today? What do you expect in a year? Is there a launch or a marketing push that could 10× this overnight?" The launch-spike question matters enormously — a system sized for steady growth can be destroyed by a single viral moment, and knowing a spike is coming changes whether you build for it or build a graceful-degradation path.

**To surface cost constraints:** "What's the budget for running this? Is there a number the finance team would be unhappy to see on the cloud bill?" Engineers are weirdly shy about asking this, and it is one of the most important numbers, because it is a hard ceiling that eliminates designs before you waste time on them.

**To surface compliance:** "Where do our users live, and is any of this data regulated — health, financial, personal data of EU residents?" Ask this *early*, because a data-residency requirement discovered late can invalidate an entire architecture.

The meta-skill here is to ask in terms of *consequences*, never in terms of mechanisms. Never ask "do you need strong consistency?" — they don't know what that means and they'll say yes to be safe, costing you dearly. Ask "is it a problem if two people briefly see different counts?" and let their answer reveal the requirement. You are translating, in real time, from the language of product consequences into the language of non-functional requirements. That translation is the senior's core competency in this phase.

There is also a political dimension worth naming. When you ask "what happens if this is down for ten minutes," you are not just gathering a number — you are getting the stakeholder *on record* about the real stakes, which protects you later. If they say "it's fine, users are annoyed," and you build to 99.9%, nobody can come back after an incident and demand to know why you didn't build five nines. The interrogation is also documentation of agreed-upon stakes.

### The durability question deserves its own paragraph

Of all the hidden requirements, durability is the one stakeholders most reliably forget, because to them "the database" simply never loses data. But durability is not free and it is not binary — it is a probability that committed data survives, and the cost rises steeply as you push it toward certainty. The interrogation question — "the app says 'posted,' then a crash loses it forever, how bad is that?" — sorts every write in the system into one of three durability tiers, and each tier maps to a concrete write path.

Tier one is **best-effort**: losing the write is a shrug. A "user is typing" indicator, an analytics ping, a non-critical metric. You can fire-and-forget these, buffer them in memory, and accept that a crash drops whatever was in flight. The write path is cheap and fast because it doesn't wait for durable acknowledgment.

Tier two is **durable-on-acknowledgment**: the user was told it succeeded, so it must survive a single-node failure. This is most writes — a feed post, a comment, a profile edit. The write path must persist to durable storage and replicate to at least one other node *before* returning success, so that losing the node that took the write doesn't lose the data. This is where replication factor and quorum acknowledgment enter the design, and it is meaningfully more expensive than tier one because the write latency now includes a replication round trip.

Tier three is **never-lose-it-ever**: a payment, a legal record, a medical result. Here you want multi-replica synchronous durability, often across availability zones, plus the ability to prove the write happened (an audit log, an append-only ledger). The write path is the slowest and most expensive, and you accept that latency cost because the alternative — losing someone's money — is unacceptable. The replication and acknowledgment mechanics behind these tiers are covered in [database replication sync async logical physical](/blog/software-development/database/database-replication-sync-async-logical-physical); the requirements job is to *assign each write to a tier* by asking the durability question, because that assignment is what tells the implementer which write path to build.

The senior insight is that you do not pick one durability level for the whole system any more than you pick one availability level. You tier the writes, and the cheap tiers subsidize the expensive ones — fire-and-forget analytics costs almost nothing, which leaves budget for the synchronous cross-zone durability that payments demand. A team that makes every write tier-three pays enormously for durability that best-effort writes never needed; a team that makes every write tier-one eventually loses something it promised to keep.

### Writing it all down: the requirements artifact

The output of the interrogation is not a memory — it is a document, and the document has a shape. I keep a lightweight template that forces every dimension to be filled in, which makes the gaps obvious: an empty field is an un-asked question, and an un-asked question is a future incident. The artifact is short, scannable, and lives next to the design doc.

```yaml
# Requirements: Home Feed (v1)
functional:
  - user posts an item to followers      # write, core
  - user reads home feed                  # read, core
  - user likes a post                     # write, core, high-volume
  - user searches posts                   # read, secondary
non_functional:
  latency:
    feed_read:   { p50: 50ms, p95: 200ms, p99: 400ms }   # "feels instant"
    post_write:  { p95: 500ms }                          # background-able
  throughput:
    read_qps_peak: 7000
    write_qps_peak: 700
    read_write_ratio: "10:1"
  availability:
    feed_read: 99.9%        # 43.2 min/month; rejected 99.99% (10x cost)
    payment:   99.99%       # different profile, different service
  durability:
    feed_post: tier2_durable_on_ack
    payment:   tier3_never_lose
  consistency:
    feed_read: eventual      # up to 5s stale OK (from interrogation)
    payment:   strong
  scale_horizon: "1M -> 10M DAU over 18 months, hold p95"
  cost_ceiling: "$50k/month all-in"       # forces CDN+cache as requirement
  compliance:
    data_residency: "EU users stay in-region (GDPR)"
out_of_scope_v1:
  - realtime_collab_editing  # needs OT conflict resolution, 3x timeline
  - offline_mode             # needs multi-device conflict resolution
  - e2e_encryption           # breaks server-side ranking + search
```

That document is the deliverable of this whole phase. Notice what it makes impossible: you cannot leave a dimension blank without it being visibly blank, you cannot smuggle in a requirement without recording its cost, and you cannot quietly accept scope without putting it in the out-of-scope list with a reason. It is boring, and boring is exactly the property you want, because every line is a decision that someone agreed to and that the architecture will now be derived from.

## 7. Stacking a requirement from business goal down to a single number

It helps to see how a complete requirement is *layered*, because the layers are how you check that your SLO actually serves a real human need rather than an arbitrary engineering preference. A well-formed requirement is a stack: it starts at a business goal and descends, layer by layer, to a single measurable indicator.

![A layered stack showing a requirement descending from business goal to user journey to capability to SLO to SLI](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-7.webp)

At the top sits the **business goal**: *retain users*. That is the thing the company actually cares about; nobody at the executive level cares about p95 latency for its own sake. Below it is the **user journey** that serves that goal: *the user opens the feed and sees fresh, relevant content*. If that journey is slow or broken, users churn, and retention — the business goal — suffers. Below the journey is the **capability** the system must provide: *rank the user's network's recent posts and render them*. Below the capability is the **SLO**: *p95 feed render under 200ms at 99.9% availability* — the engineering target that ensures the capability is delivered well enough to serve the journey. And at the very bottom is the **SLI**: *the ratio of feed-read requests that succeed under 200ms* — the single number you actually measure and alert on.

The discipline of building this stack is what keeps SLOs honest. Every time someone proposes an SLO, you walk *up* the stack and ask "which user journey does this serve, and which business goal does that journey advance?" If you cannot answer, the SLO is arbitrary and you are about to spend money for no reason. The classic case is an engineer who wants p99 < 50ms on an internal batch endpoint because low latency feels virtuous — but no user journey depends on that endpoint's latency, so the SLO serves nothing and the over-provisioning it demands is pure waste. The stack catches that.

It also works in the other direction. When the business goal changes — say the company pivots from growth to monetization — you walk *down* the stack and discover which SLOs need to change. If retention stops being the priority and revenue-per-user takes over, the payment flow's SLOs get stricter and the feed's might loosen. The stack makes the requirement traceable, and traceable requirements are the ones that survive a year of changing priorities. This connects directly to the next post on [back of the envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design), where these stacked numbers become the inputs to your capacity math.

## 8. From requirements to a capacity and cost envelope

Requirements are not just constraints to satisfy — they are *inputs to arithmetic*. Once you have the throughput, storage, and latency numbers, you can compute a capacity and cost envelope before you build anything, and that envelope is what tells you whether the design is even feasible. This is the bridge between requirements and the back-of-envelope estimation that the next post covers in depth; here I want to show how the requirement numbers flow directly into a cost ceiling.

#### Worked example: capacity and cost envelope for the feed

Take our feed and put numbers on it. Suppose the interrogation surfaced: 10 million daily active users, each opening the feed an average of 20 times a day, with a peak-to-average ratio of 3×. Each user posts an average of twice a day.

**Read QPS.** 10M users × 20 reads = 200M feed reads per day. Spread over 86,400 seconds, that is about 2,300 reads/second average. With a 3× peak factor, peak read QPS is about **7,000 reads/second**. That is the load the read path must sustain at p95 < 200ms.

**Write QPS.** 10M users × 2 posts = 20M posts per day, about 230 writes/second average, **~700 writes/second peak**. The read/write ratio is roughly 10:1, which immediately tells the senior that this is a read-heavy workload where caching and read replicas will pay off enormously, and write throughput is not the bottleneck.

**Storage.** Suppose a post averages 1KB of text and metadata. 20M posts/day × 1KB = 20GB/day = about **7.3TB/year** of raw post data, before replication. At a replication factor of 3 for durability, that is roughly 22TB/year of stored bytes. Over a three-year retention horizon, call it **65TB**. That is a number you can price.

**Bandwidth.** Each feed read returns, say, 30 posts at 1KB plus overhead — call it 50KB per read response. 7,000 reads/second × 50KB = 350MB/second of egress at peak, or about **2.8 Gbps**. That number decides your CDN and egress cost, and at typical cloud egress prices of around \$0.08–\$0.12 per GB it is a real line item — 350MB/s sustained is roughly 900TB/month of egress, which at \$0.09/GB is on the order of **\$80,000/month** in raw egress alone if you serve it all from origin. *That* number is exactly why a feed lives behind a CDN and aggressive caching, and it is invisible until you do the arithmetic.

**Cost ceiling check.** If finance said the feature must run under \$50,000/month all-in, the \$80,000/month naive egress figure already blows the budget, which forces the caching-and-CDN design *as a requirement, not an optimization*. The requirement-driven cost envelope turned a "nice to have" cache into a "the business case fails without it" cache. That is the kind of conclusion you want to reach on a whiteboard in week one, not in a budget review in month six.

This is the whole point of computing the envelope from requirements: it surfaces feasibility and forces architecture decisions *cheaply*, before you have written code. The detailed mechanics of these estimates — how to pick the right unit costs, how to round sanely, how to sanity-check against known system limits — are the subject of [back of the envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design). For now, internalize that **every non-functional number is also an input to a cost model**, and the cost model frequently makes the architecture decision for you.

## 9. Trade-offs: the requirements decision matrix

Now we get to the heart of thinking like a senior, which is refusing to recommend anything without naming its cost. Requirements are not free to satisfy; every dimension you tighten, you pay for in another dimension. The senior holds an explicit trade-off matrix in their head and makes the trade visible in the design review rather than letting it hide.

The three dimensions that trade against each other most sharply in distributed systems are **consistency**, **availability**, and **latency**. This is not arbitrary — it is the practical face of the CAP and PACELC theorems, which say (roughly) that during a network partition you must choose between consistency and availability, and even when there is no partition you trade consistency against latency. Rather than re-derive those theorems here (we do that properly in [cap theorem and pacelc](/blog/software-development/database/cap-theorem-and-pacelc) and in the forthcoming [articulating tradeoffs cap pacelc and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond)), let me show how the requirement *profile* of each feature picks a corner of the trade space.

| Feature | Wants | Trades away | Concrete consequence |
| --- | --- | --- | --- |
| Payment write | Strong consistency, high durability | Latency and some availability | Synchronous quorum writes; rejects writes during a partition rather than risk a double-charge |
| Feed read | Low latency, high availability | Consistency (accepts staleness) | Serves from async replicas and cache; a read may be a few seconds stale |
| Like counter | High availability, low latency | Consistency and exactness | Eventually-consistent counter; the count can be briefly wrong |
| Inventory decrement | Strong consistency | Latency, availability | Must not oversell; serializes the decrement, accepts higher latency |

Read across each row: what the feature *wants*, what it must *trade away* to get it, and the *concrete architectural consequence*. The payment write wants strong consistency and durability, so it trades away latency (synchronous quorum writes are slower) and even some availability (during a partition it would rather reject the write than risk a double-charge — it picks C over A). The feed read makes the opposite trade: it wants low latency and high availability, so it accepts staleness, reading from asynchronous replicas and caches. The like counter trades exactness for availability. The inventory decrement, like the payment, picks consistency over latency because overselling is unacceptable.

The senior move is to make this table *the* artifact of the design review. When someone challenges "why is the feed eventually consistent?" you point at the row: because the feed *wants* low latency and high availability, and the only way to get both is to trade away strong consistency, and the requirement interrogation established that a few seconds of staleness is acceptable. The trade is justified by the requirement, the requirement is justified by the business need, and the whole chain is written down. Nobody can relitigate it without producing a new requirement.

This is also where you connect requirements to the deeper consistency mechanics. The choice of "eventual consistency" for the feed is not a vibe — it has a precise meaning and precise failure modes, covered in [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual). The requirements phase decides *which* consistency model you need; the mechanism deep-dive tells you how to implement and operate it. Keep those two activities separate in your head: requirements *choose*, mechanisms *implement*.

## 10. How a single requirement propagates into the architecture

I want to show, concretely, how one number — a freshness tolerance — ripples through the entire design. This is the payoff of doing requirements properly: you do not make architecture decisions by taste, you make them by *propagation* from a stated requirement.

![A branching graph showing how a freshness requirement forks into a caching decision and a consistency decision that converge on a datastore choice](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-8.webp)

Start at the top of the figure with the requirement we extracted in the interrogation: *"feed reads can be up to 5 seconds stale."* Watch it fork. Down the left branch, "5 seconds stale is acceptable" directly unlocks a **caching path with a 5-second TTL** — you can cache feed responses for up to five seconds and serve the vast majority of reads from cache without anyone noticing the staleness. Down the right branch, the same requirement establishes that **eventual consistency is acceptable**, which unlocks reading from **asynchronous replicas** rather than forcing every read to hit the leader. Both branches then converge on the **datastore choice**: because we tolerate staleness and eventual consistency, a leaderless or async-replicated store is acceptable, and we are *not* forced into the expensive strong-consistency machinery that the payment path requires.

Now flip the requirement and watch the whole architecture change. If the interrogation had instead revealed "feed reads must show the user's own writes immediately" (read-your-writes consistency), the caching TTL collapses (you cannot serve a stale cache to a user who just posted), the async-replica read path becomes unsafe (the replica might not have the user's write yet), and you are pushed toward routing a user's reads to the leader or implementing sticky read-your-writes session guarantees. Same feature, one different requirement, completely different architecture.

This is why I keep insisting that requirements *precede* architecture. The architecture is not a creative act performed in a vacuum; it is the *consequence* of the requirements, mechanically propagated. A senior who has the requirements nailed down can derive most of the architecture by following the propagation, and — crucially — can *defend* every choice by tracing it back up the chain to a requirement and a business need. The connection between consistency requirements and the underlying replication choices is explored in depth in [distributed replication leader multi-leader leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless); here the lesson is that the *requirement* is the thing that selects among those options.

## 11. The optimization angle: over-specifying availability is a cost trap

Here is the optimization lesson that I think is the single most valuable thing in this entire post, because it saves real money and real on-call suffering, and almost every team gets it wrong in the same direction. **The instinct is always to over-specify reliability, and over-specification is enormously expensive.** Nobody ever got fired in a planning meeting for asking for more nines, so everyone asks for more nines, and the cumulative cost across an organization is staggering.

![A before-and-after comparison of an over-specified five-nines SLO versus a right-sized three-nines SLO and their cost and on-call implications](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-9.webp)

Look at the two designs. On the left, someone asked for 99.999% availability for the feed because "we want it to be really reliable." That is a 26-second monthly budget, which forces multi-region active-active deployment, cross-region consistency machinery, near-instant automated failover, and a 24/7 on-call rotation that wakes people up for sub-minute blips. The cost is on the order of 30× the right-sized design, and the human cost — engineers paged for things that don't matter — is corrosive in a way that doesn't show up on the bill but shows up in attrition.

On the right is the right-sized design: 99.9%, a 43-minute monthly budget, a single region with a warm failover, and an on-call rotation that only fires for genuine incidents. It costs roughly 5× a non-redundant baseline (not 30×), and the on-call is *calm*, which means the engineers stay. The two designs serve the *same feature*. The only difference is that someone matched the SLO to the actual business need instead of chasing nines for their own sake.

The senior's optimization rule: **the right SLO is the loosest one the business can actually tolerate, not the tightest one you can imagine.** Every nine you add past the business need is pure cost with negative human return. The optimization is not "make it more reliable" — it is "make it exactly as reliable as it needs to be and not one nine more." You *measure the win* in dollars (the 30× → 5× cost reduction), in on-call pages avoided (count the pages that would have fired under five nines for blips that nobody would have noticed under three nines), and in engineering time freed to build features instead of operating gold-plated infrastructure.

There is a second optimization hidden in the error budget itself: the error budget is *a budget to be spent, not a number to maximize*. This is the SRE insight that flips reliability on its head. If you have a 43-minute monthly budget and you are using only 5 minutes of it, you are being *too cautious* — you are leaving 38 minutes of risk-taking on the table that you could spend on shipping features faster, deploying more aggressively, or running chaos experiments. A team that never spends its error budget is a team that is over-investing in reliability at the expense of velocity. The optimal state is to spend *most* of your budget most of the time, which means you are moving as fast as your reliability target allows. Over-specifying the SLO and then under-spending the budget is the double cost trap: you pay for reliability you don't need, and then you move slower than you could because you're protecting a budget that's far larger than your business requires.

## 12. The art of saying no and deferring scope

A requirements document is defined as much by what it *excludes* as by what it includes, and a senior is far more willing to say no than a junior, because a senior knows that every requirement accepted is a cost incurred forever. When the product manager says "and it should also support real-time collaborative editing of posts, and offline mode, and end-to-end encryption," the junior writes it all down. The senior asks "which of these is needed for launch, and what does each one cost us?"

Saying no is a skill with a structure. You do not say "no, that's too hard" — that makes you the obstacle. You say "yes, and here is the cost, so let's decide together whether it's worth it now or later." End-to-end encryption on the feed, for example, fundamentally breaks server-side ranking and search (you cannot rank content you cannot read), so accepting that requirement means giving up the ranking capability — a trade you surface explicitly and let the product owner make with open eyes. Offline mode means building conflict resolution for edits made on multiple devices, which is a whole [saga pattern distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions)-class problem of its own. Each "also" carries a cost, and the senior's job is to make that cost visible *before* it is accepted, so the decision to take it on is informed.

The technique I use is a three-bucket sort for every requested feature: **must-have for launch**, **fast-follow** (built right after launch, designed-for now but not built yet), and **explicitly out of scope** (with the reason recorded). The third bucket is the most valuable and the one juniors skip. Writing down "real-time collaborative editing — OUT OF SCOPE for v1 because it requires operational-transform conflict resolution that would triple the timeline, revisit if usage data shows demand" does three things: it documents the decision so it doesn't get relitigated weekly, it preserves the reasoning so a future engineer understands why, and it leaves a clear trigger ("if usage data shows demand") for when to reconsider. A requirements doc with an empty out-of-scope section is a requirements doc that will suffer endless scope creep, because nothing is ever officially excluded.

Deferring is subtly different from excluding, and the distinction matters architecturally. When you *defer* a feature to fast-follow, you may still need to *design for it now* — leaving the seams in the architecture so it can be added later without a rewrite. When you *exclude* a feature, you may deliberately *not* design for it, accepting that adding it later would be a significant change. Knowing which features to design-for-but-not-build versus exclude-entirely is a judgment call, and getting it right is a big part of what [evolutionary architecture designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) is about. The requirements phase is where you make that call, and you make it by asking "how likely is this, and how expensive is it to retrofit?"

## 13. The whole transformation, before and after

Let me bring it all together by showing the transformation the senior performs, side by side, because the contrast is the lesson.

![A before-and-after comparison of a vague one-line ask transformed into structured measurable requirements](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-5.webp)

On the left of the figure is the input: three adjectives. "Feed feels instant." "Never goes down." "Handles lots of users." Every word is an opinion. None of it is buildable, testable, or defensible. You cannot write a test for "feels instant." You cannot size a server for "lots of users." You cannot price "never goes down" — and if you tried, you'd build five nines and waste a fortune.

On the right is the output of the senior's work: three requirements. "p95 < 200ms for feed read" — testable, measurable, alertable. "99.9% SLO with a 43-minute monthly error budget" — priced, achievable, with a clear policy for what happens when it's exhausted. "50,000 QPS peak with 5× headroom" — a sizing target you can validate a design against. Every adjective on the left became a number on the right, and every number traces back to a business need surfaced in the interrogation.

That transformation *is* the job in this phase. It is not glamorous; there are no clever algorithms in it. It is the disciplined, almost boring work of replacing every adjective with a number and writing down the reason for each number. But it is the work that determines whether everything downstream — the architecture, the capacity plan, the cost model, the on-call burden — is built on a foundation of facts or a foundation of opinions. A design built on the left column will be argued about forever and will fail in production in ways nobody predicted. A design built on the right column can be reviewed, defended, and operated.

Notice also that the right column is *falsifiable*. "p95 < 200ms" can be wrong — you can measure it and discover you're at 350ms, and then you have a concrete problem to solve. "Feels instant" can never be wrong, which is exactly why it's useless as a requirement. The senior prefers requirements that *can* be violated, because only a requirement that can be violated can be verified.

## 14. Watching the error budget burn over a month

The error budget is not a number you compute once and file away. It is a *living balance* that burns down over the month, and watching the burn rate is how you manage reliability in real time. This is where SLOs stop being a planning artifact and become an operational tool.

![A timeline showing a monthly error budget burning down from 43 minutes through a series of small and large incidents](/imgs/blogs/turning-vague-asks-into-requirements-and-slos-6.webp)

Follow the timeline. The month starts with the full **43.2-minute budget** in the bank. On day 6, a minor blip — a slow deploy, a brief replica hiccup — burns 3 minutes. No problem; you've got 40 left, and the burn rate is well within budget. Then on day 14, a bad deploy causes a 28-minute partial outage. That single event spends most of the remaining budget, leaving only about 12 minutes for the rest of the month. This is the moment the error budget *changes behavior*: with 12 minutes left and two weeks to go, the team enacts a **deploy freeze** on risky changes — the budget policy kicks in. The rest of the month is conservative: small, safe, well-tested changes only. The system survives to day 30 with about 4 minutes of budget to spare.

The crucial insight is that the error budget is what *connects reliability to release velocity*. When the budget is healthy, you ship aggressively — you can afford to take risks, because you have downtime budget to absorb a mistake. When the budget is nearly exhausted, you slow down and stabilize. This turns a perpetual and unwinnable argument — "should we ship fast or be reliable?" — into a *measurable, automatic decision*: ship fast while the budget allows, slow down when it doesn't. The budget is the referee. It removes the politics, because nobody is arguing about gut feelings; they're looking at a number that says "you have 12 minutes of downtime budget left this month, so no, we are not deploying that risky migration today."

The burn-rate alerting that watches this is itself a piece of design. You don't alert when the budget hits zero — that's too late. You alert on the *rate* of burn: a fast-burn alert fires if you're consuming the budget so quickly that you'll exhaust it within hours (a real incident in progress), and a slow-burn alert fires if you're on track to exhaust it by month-end even though no single event is dramatic (a slow degradation that would otherwise sneak past you). Multi-window, multi-burn-rate alerting is the operational mechanism that makes the error budget actionable, and it is one of the genuinely good ideas to come out of the SRE discipline.

## 15. Case studies

Requirements discipline is not academic. Some of the most instructive production stories are stories about requirements that were captured well or captured badly. Here are four, with the concrete lesson each teaches.

**Google SRE and the invention of the error budget.** The error-budget framework as we use it today was codified in Google's SRE practice, and the originating problem was organizational, not technical: the SRE team (who wanted reliability) and the product team (who wanted velocity) were in perpetual conflict, and every release was a negotiation. The error budget resolved it by turning reliability into a *currency*. The product team gets to spend the error budget on shipping features; if they spend it all, releases freeze until the budget recovers. This aligned incentives — the product team now *cared* about reliability because outages spent their feature budget — and it converted an unwinnable argument into arithmetic. The lesson: a well-defined SLO with an error budget is as much a tool for aligning humans as it is for measuring systems.

**Amazon and the latency-is-money requirement.** Amazon famously established, through experimentation, that added latency on their pages measurably reduced sales — the often-quoted order of magnitude is that every additional 100ms of latency cost a noticeable fraction of revenue. The lesson for requirements is not the specific number (which varies by product and is frequently misquoted) but the *practice*: they treated latency as a first-class business requirement with a measured dollar impact, not as an engineering nicety. When you can attach a revenue number to a latency SLO, the requirement stops being a matter of taste and becomes a business decision, and that is exactly the altitude a senior wants to operate at.

**Netflix and chaos as a durability/availability requirement.** Netflix built Chaos Monkey — a tool that deliberately kills production instances — because they made an explicit non-functional requirement: *the system must survive the loss of any single instance with no user-visible impact.* Rather than hope that requirement was met, they *continuously tested* it in production. The lesson is that a non-functional requirement like "tolerate instance failure" is only real if you verify it; a durability or availability requirement that is never tested under real failure conditions is a wish, not a requirement. The requirement ("survive instance loss") drove a verification practice (chaos engineering), which is the requirement made operational.

**The over-specified five-nines internal tool.** This is a composite from incidents I've seen across multiple companies, not a single named one, so I'll frame it as a pattern rather than a specific number: an internal tool — a dashboard, an admin panel, something only employees use — gets a five-nines availability requirement because someone in the planning meeting said "it should always be up." The team dutifully builds multi-region failover for a tool that twelve people use during business hours. The infrastructure costs many times what the tool is worth, the on-call rotation pages engineers at 3am for an internal dashboard nobody is looking at, and eventually someone does the math and realizes that 99% (7 hours of monthly downtime, none of it at 3am because nobody uses it at night) would have been completely fine and a fraction of the cost. The lesson is the cost trap of section 11, made flesh: the absence of a requirements interrogation — nobody asked "what actually happens if this is down at 3am?" — led to enormous waste. The fix was free: one good question, asked early.

## 16. When to reach for rigorous SLOs (and when not to)

Like everything in engineering, requirements rigor has a cost, and a senior knows when to dial it up and when to keep it light. Applying the full SLI/SLO/error-budget machinery to a weekend prototype is its own kind of over-engineering.

**Reach for full requirements and SLO discipline when:** the system is user-facing and its reliability affects revenue or trust; it is operated by a team over time (so the requirements need to be a durable, shared artifact); it has external customers with contractual expectations (so SLAs are in play); it is expensive enough that right-sizing the SLO saves meaningful money; or it has compliance and data-residency constraints that are architecture-defining and non-negotiable. For anything that real users depend on and a team operates, the few hours of requirements work pays for itself many times over in avoided rework and avoided over-provisioning.

**Keep it light when:** you are building a prototype to validate an idea (capture the functional requirements and a rough latency sense, skip the formal SLOs — you'll throw the code away anyway); the system is internal and low-stakes (a quick availability target, no error-budget machinery); or you genuinely do not yet know the load and are building to *learn* what the requirements are (in which case the first version's job is to surface the real numbers, and you formalize SLOs once you have real traffic data). Imposing rigorous SLOs on a system whose load you cannot yet estimate is theater — you'll invent numbers, build to them, and discover they were all wrong once real traffic arrives.

The judgment is the same judgment as everywhere in this series: match the rigor to the stakes. A senior is not someone who applies maximum process everywhere; a senior is someone who knows *which* process matters *when*. The requirements interrogation always matters — even the weekend prototype benefits from one good question about expected load. But the full SLO-and-error-budget apparatus is a tool for systems that real people depend on and real teams operate, and applying it to a throwaway is the inverse mistake of skipping it on a payment system. The next post, [back of the envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design), is what you reach for in that "I don't know the load yet" case — it's how you produce the rough numbers that become your first SLOs.

## 17. Key takeaways

- **Adjectives are the absence of requirements.** "Fast," "reliable," "scalable" are opinions. Your entire job in this phase is to replace every adjective with a number that someone will sign their name under. A requirement you cannot violate is a requirement you cannot verify.
- **Juniors capture the functional half; the non-functional half decides the architecture.** Latency percentiles, throughput, availability, durability, consistency, scalability, cost, and compliance are what actually drive design. Run the checklist against every ask.
- **One product has many requirement profiles.** Do not impose a single global SLO. The payment write and the like counter live in opposite corners of the consistency/availability/latency trade space, and forcing one profile on both either over-builds or under-builds most of the system.
- **Know the error-budget math cold.** 99.9% is 43.2 minutes of downtime per month. Each nine divides the budget by ten and roughly multiplies the cost. Internalize the nines-to-downtime table so you can do it in your head in a meeting.
- **SLI is measured, SLO is your internal target, SLA is the weaker external promise.** Always keep the SLO stricter than the SLA so you have margin before you owe anyone money, and alert on the budget *burn rate*, not on the SLO boundary.
- **Interrogate in consequences, never in mechanisms.** Never ask "do you need strong consistency?" Ask "is it a problem if two people briefly see different counts?" Translate product consequences into non-functional requirements in real time.
- **Over-specifying reliability is the most common and most expensive mistake.** The right SLO is the loosest the business can tolerate, not the tightest one available. Match the nines to the real cost of downtime, and spend your error budget rather than hoarding it.
- **Requirements precede and determine architecture.** You do not pick CAP first; the requirement picks it for you. A single freshness number propagates into caching, consistency, replication, and datastore choices. Trace every architecture decision back up to a requirement and a business need.
- **A requirements doc is defined by what it excludes.** Maintain an explicit out-of-scope bucket with reasons and triggers. Saying no, with the cost made visible, is a senior skill — and the empty out-of-scope section is the seed of endless scope creep.

## 18. Further reading

- [How seniors approach ambiguous system design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems) — the overall clarify-constrain-sketch-stress-iterate loop this post lives inside.
- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — turning the requirement numbers into capacity and cost math, the natural next step.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — how the consistency/availability/latency trade space, which the requirement profile selects, actually works.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the formal foundation behind the trade-off matrix in section 9.
- [Consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the precise meaning and failure modes of the consistency choice your freshness requirement makes.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the replication mechanisms a durability and consistency requirement selects among.
- [Evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) — how to defer scope while leaving the seams to add it later.
- *Site Reliability Engineering* (Google, O'Reilly) — the originating text for SLIs, SLOs, error budgets, and burn-rate alerting. Chapters 3 and 4 are the canonical reference for everything in sections 4 through 6 and 14.

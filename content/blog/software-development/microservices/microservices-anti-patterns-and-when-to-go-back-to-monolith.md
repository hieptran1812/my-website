---
title: "Microservices Anti-Patterns and When to Go Back to the Monolith: The Honest Counterweight"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A field guide to the ways microservices go wrong — the distributed monolith, nanoservices, the shared database, the god service, chatty call chains, premature decomposition — and the increasingly common, genuinely smart decision to consolidate back to a modular monolith or coarser services."
tags:
  [
    "microservices",
    "anti-patterns",
    "distributed-monolith",
    "modular-monolith",
    "consolidation",
    "nanoservices",
    "distributed-systems",
    "software-architecture",
    "backend",
    "migration",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-1.webp"
---

This is the post where I stop selling you microservices.

Thirty-eight articles into this series, I have walked you through service boundaries, the saga pattern, circuit breakers, mTLS, distributed tracing, canary deploys, and how to migrate a monolith with the strangler fig. All of it is real, all of it works, and all of it is what a senior engineer reaches for when the system genuinely needs it. But every one of those techniques exists to *pay for a cost you chose to take on*. And the most senior thing I can teach you — the thing that separates someone who has shipped microservices from someone who has actually *lived with them at 3am* — is knowing when that cost was a mistake, how to recognize it, and how to walk it back.

I have a company in mind while I write this. I will call it ShopFast. It is a composite of three real teams I have either worked on or cleaned up after, and it follows a depressingly common arc: a two-engineer startup with a clean monolith reads a few conference talks, decides microservices are "how you scale," and over the course of a year splits itself into **forty services** operated by **three engineers**. Their checkout flow makes **eight synchronous network hops**. Most of those forty services still read and write the same shared Postgres database. The team spends more time wiring services together and chasing latency across the call graph than building features. They are, in the words of one of their engineers, "drowning." Figure 1 is the map of how they got there — a taxonomy of the anti-patterns we are going to diagnose, every one of which ShopFast hit.

![A tree diagram grouping microservices anti-patterns into two families, wrong boundaries and missing operational maturity, with examples under each branch](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-1.webp)

By the end of this post you will be able to do three concrete things. First, **diagnose** the major microservices anti-patterns from their symptoms — you will recognize a distributed monolith, a nanoservice sprawl, a god service, and a chatty call chain by the tells they leave in your deploy logs, your traces, and your team's calendar. Second, **measure** whether you are over-split, with real signals and a couple of diagnostic scripts. Third, **consolidate** — execute the increasingly common, increasingly respected move of merging services back into a modular monolith or coarser services, recovering velocity and cutting cost, and understand why doing this is a *sign of engineering maturity, not failure*. We will follow ShopFast from 40 services back to 6, with numbers.

If you read only one post in this series, the senior version of the lesson is this: **the goal was never microservices. The goal is a system you can change safely and cheaply.** Sometimes that system is microservices. Very often it is a modular monolith. And consolidating back when you over-split is one of the smartest, most senior moves you can make. Let me show you why.

## The thesis: microservices are a tax, not a feature

Before the catalog of failures, you need the frame that makes every one of them legible. Microservices buy you exactly two things that a monolith cannot:

1. **Independent deployability.** Team A can ship its service without coordinating with Team B. This is the team-scaling property — it lets dozens of teams move in parallel without colliding in one codebase.
2. **Independent scalability and fault isolation.** You can scale the one hot service to fifty replicas without scaling everything, and a crash in the recommendation service does not (if you did your resilience homework) take down checkout.

That is the *entire* upside. Everything else people attribute to microservices — clean code, good boundaries, testability — you can get from a well-structured monolith. (We covered this in detail in [monolith-first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith).)

And the bill for those two benefits is steep. You pay it in three currencies:

- **Operational overhead.** Every service needs a CI/CD pipeline, a deploy target, health checks, dashboards, alerts, on-call coverage, secrets, a network identity, and a place in the service mesh. Forty services is forty of each. This cost grows roughly linearly with service count, and it does *not* go to zero when a service is small.
- **Cognitive overhead.** A request that was one stack trace in the monolith is now a distributed trace across eight services owned by four teams. Debugging requires correlation IDs, tracing infrastructure, and a mental map no single person holds. (This is exactly why [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) is not optional once you split.)
- **Latency and consistency overhead.** Every in-process method call you turned into a network call added a network hop — serialization, a round trip, a chance of timeout, a chance of partial failure. A transaction that was ACID in one database is now a saga across several, with eventual consistency and compensating actions to write and test.

So the honest decision is not "monolith bad, microservices good." It is: **you should only pay the microservices tax when you have both the scale that needs it (enough teams, enough traffic) AND the operational maturity to afford it (CI/CD, observability, platform automation).** Take on the tax without the scale and you get all the cost and none of the benefit. Take it on without the maturity and you get a system you cannot operate. Both are the failure modes we are about to catalog.

Let me put one more number on the tax, because juniors consistently underestimate it. Every service you stand up carries a *fixed operational cost that is independent of how much code is inside it*. Concretely, a single new service in a typical Kubernetes-based shop needs: a Git repository and CODEOWNERS, a CI pipeline (build, test, scan, push), a CD pipeline (deploy, rollback, health-gate), a Dockerfile and base-image upgrade cadence, Kubernetes manifests (Deployment, Service, HPA, PodDisruptionBudget, NetworkPolicy), liveness and readiness probes, a dashboard, a set of alerts wired to an on-call rotation, secrets and config wiring, a service-mesh identity and mTLS cert, an entry in service discovery, an SLO, and a runbook. That is fifteen-plus artifacts *before a single line of business logic exists*. Call it conservatively one to three engineer-weeks to set up properly and a few hours per month forever to maintain — per service. Multiply by forty and you have ShopFast's problem: the team is spending its entire capacity feeding the fixed overhead of forty services and has nothing left for product. The logic inside a nanoservice might be a day of work; the wrapper around it is the rest of the iceberg, and the wrapper does not shrink when the logic does.

This is the asymmetry that makes the consolidation math so favorable later. When you merge two services into one, you delete a *whole copy* of that fixed overhead — one fewer pipeline, one fewer rotation, one fewer mesh identity, one fewer set of alerts — while the business logic just moves into the same process and keeps working. Consolidation does not delete features; it deletes the *per-service tax* on features. Hold that thought; it is why the 40 → 6 merge later recovers cost and velocity at the same time.

## The anti-pattern catalog

Every anti-pattern below follows the same three-beat structure: the **symptom** (what you observe), the **cause** (why it happened), and the **fix** (what to do about it). Figure 6, the symptom-and-fix matrix, is the compressed version you can pin above your desk; the prose is where the reasoning lives.

![A matrix mapping five microservices anti-patterns to their observable symptom and their consolidation fix](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-6.webp)

### 1. The distributed monolith — the master anti-pattern

**Symptom:** Your services cannot be deployed independently. To ship a feature you have to release service A, then B, then C, in a specific order, in the same window, or it breaks. A change to one service's API forces simultaneous changes to three others. You have a "release train" where everything goes out together every two weeks because nothing can go out alone.

**Cause:** You split the *code* into separate deployables but you did not split the *coupling*. The services share data (a shared database, or one reaching into another's tables), share synchronous request paths that assume a specific call order, or share data models so tightly that any schema change ripples across the fleet. You have the network boundary without the independence — which means you got every cost of microservices and none of the benefits. This is the master anti-pattern because almost every other failure on this list collapses into it.

I gave this its own full post — [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith) — because it is the single most common way teams go wrong. Here I want you to internalize the *one-line test*: **if two services must be deployed together more often than not, they are not two services. They are one service in a trench coat.**

**Fix:** Either merge the services that co-deploy back into one (the consolidation move — most of this post), or break the coupling that forces co-deployment: give each service its own database, replace the synchronous cross-service joins with asynchronous events or local data copies, and version your APIs so a producer can change without breaking consumers ([api versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) covers the contract side). The diagnostic for "which services co-deploy" is below, and it is the single most useful script in this article.

### 2. Nanoservices — too fine-grained, the overhead exceeds the logic

**Symptom:** You have a service whose entire job is to validate an email address, or to format a currency, or to wrap a single database table with three CRUD endpoints. The service is 200 lines of business logic wrapped in 2,000 lines of Dockerfile, Kubernetes manifests, health checks, retry config, and tracing boilerplate. The overhead of *operating* it dwarfs the logic *inside* it. You have forty of these and a three-person team.

**Cause:** You took "small" as the goal instead of "independently changeable by a team." Somewhere the lesson "services should be small" mutated into "smaller is always better," and you kept splitting past the point of diminishing returns. The tell is that your services do not map to *business capabilities* or *team boundaries*; they map to *nouns* or *functions*. A service per database table is the classic nanoservice smell.

This is ShopFast's headline problem. Figure 3 is their before-and-after: forty nanoservices with an eight-hop checkout, a shared database, and three engineers drowning, consolidated down to six well-bounded services with a two-hop checkout and a team that ships again.

![A before-and-after diagram contrasting forty chatty nanoservices on a shared database against six well-bounded services with clear data ownership](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-3.webp)

**Fix:** Merge nanoservices up to the level of a business capability owned by one team. The right granularity question is never "is this small?" It is **"can one team own and independently deploy this, and does the value of independent deployment exceed the operational cost?"** For ShopFast, "pricing," "tax," and "promotions" were three nanoservices on the same hot path, changed by the same engineer, never deployed apart. They became one Pricing service. The merge math is the worked example below.

### 3. The shared database — the coupling that hides

**Symptom:** Multiple services connect to the same database and read or write each other's tables. A schema migration requires coordinating across teams. You cannot tell who owns a table. Two services contend on the same rows and you get lock contention and mysterious deadlocks under load.

**Cause:** Splitting the application tier is easy — you carve the code into services in an afternoon. Splitting the *data* is the hard, slow part, so teams skip it and leave everything on the one database "for now." A year later "for now" is load-bearing. The shared database is the single most effective way to turn microservices into a distributed monolith, because the database becomes the hidden coupling that forces co-deployment and co-evolution.

The rule that defines microservices — [database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) — exists precisely to prevent this. A service owns its data; nobody else touches its tables; if you need its data you ask it through an API or you subscribe to its events.

**Fix:** Give each service its own schema or database, expose data through APIs and events instead of shared tables, and where one service needs another's data on a hot path, keep a *local read-replica copy* updated by events (the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) pattern is the reliable way to publish those events). For ShopFast, untangling the shared Postgres was the hardest and most valuable part of the consolidation — it is what actually broke the deploy coupling.

### 4. Entity services and the god "User service"

**Symptom:** There is one service — almost always called `user-service` or `account-service` — that *every other service* calls on *every request*. It is the most-deployed, most-paged, most-feared service in the fleet. When it has a bad day, the entire platform has a bad day. Its on-call rotation is a hazing ritual.

**Cause:** You decomposed by **entity** (User, Product, Order) instead of by **capability** (Checkout, Catalog, Identity). Entity decomposition feels natural — they are the nouns in your data model — but it is a trap, because real features cut *across* entities, so every feature ends up calling every entity service. The god service is the convergence point: everyone needs "the user," so everyone depends on the user service, and now it is a single point of failure and a coordination bottleneck dressed up as a microservice.

**Fix:** Decompose by business capability and bounded context, not by entity. ([Service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design) is the full treatment.) Where a god service is unavoidable as a *source of truth*, stop calling it synchronously on every request. Instead, let consuming services keep a **local cached copy of the slice of user data they actually need**, kept fresh by `UserUpdated` events. Checkout does not need to call the user service to learn the customer's shipping address on every request if it already has a local copy updated whenever the address changes. This converts a hard runtime dependency into a soft, eventually-consistent one, and it is often *the* move that lets you take the god service off the critical path.

The pattern in code: the checkout service subscribes to user-change events and maintains exactly the columns it needs in its own table, so the hot path never makes a network call to the god service.

```python
# checkout keeps a LOCAL read-model of the user slice it needs,
# fed by events from the user service. No sync call on the hot path.
@subscribe("user.updated")          # event from user-svc via the broker
def on_user_updated(evt):
    db.upsert("user_view", {
        "user_id":        evt["user_id"],
        "default_address": evt["default_address"],   # only what checkout needs
        "tier":           evt["tier"],
        "updated_at":     evt["occurred_at"],
    })

def start_checkout(user_id, cart):
    # local lookup — microseconds, no dependency on user-svc being up
    u = db.get("user_view", user_id)
    return build_order(u["default_address"], u["tier"], cart)
```

The cost is eventual consistency — the local copy may lag the source by the event-propagation delay (typically tens to hundreds of milliseconds with a healthy broker). For shipping address and tier that lag is harmless; for something like a fraud block you would still call synchronously. The senior judgment is *which* fields tolerate staleness and which do not — push the tolerant ones to local copies and keep only the intolerant ones on the synchronous path, and the god service stops being a single point of failure for the things that can live without it.

### 5. Chatty synchronous call chains — the N+1 across services

**Symptom:** A single user action triggers a long chain of synchronous service-to-service calls. Your trace for one checkout shows eight spans in series. p99 latency is the *sum* of every hop's p99 plus the network time between them, so it balloons. Worse, you find the cross-service N+1: a service loops over N items and makes one downstream call per item, so rendering a cart with 30 line items makes 30 calls to the pricing service.

**Cause:** You modeled service calls like in-process method calls — cheap, instant, always available. The [fallacies of distributed computing](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) say otherwise: the network is not reliable, latency is not zero, and bandwidth is not infinite. When you turn a method call into a network call you pay for it every single time, and when you chain them you pay cumulatively. ShopFast's eight-hop checkout is the canonical version; figure 4 is its trace.

![A branching graph showing a checkout request fanning through eight synchronous service hops down to a shared database, with latency compounding across the chain](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-4.webp)

There is a piece of latency math here that every senior should have memorized, because it explains why chatty chains are so much worse than they look. When you chain calls **in series**, p99 latencies do *not* simply add — the tail gets worse than additive. The probability that *at least one* of N independent hops lands in its slow tail rises with N: if each hop has a 1% chance of a slow response, a chain of 8 hops has roughly a `1 - 0.99^8 ≈ 7.7%` chance that *some* hop is slow, so the *combined* p99 is pulled up by whichever hop happened to be slow on that request. This is "tail amplification," and it is why an 8-hop checkout can have a p99 far worse than the sum of the individual p99s would suggest. Every hop you remove does not just subtract its own latency; it removes a chance for the whole request to hit a tail. Collapsing 8 hops to 2 is a tail-latency win out of proportion to the raw milliseconds.

**Fix:** Three moves, in order of preference. (a) **Collapse the chain** — if pricing, tax, and promotions are always called together in sequence, they belong in one service so the calls become in-process. (b) **Parallelize and batch** — call independent services concurrently instead of in series, and replace the N+1 with a single batch call (`getPrices([sku1..sku30])` instead of 30 calls). (c) **Precompute via events** — if checkout needs derived data, compute it asynchronously and have it ready before the request arrives. We go deep on the measurement and tuning in [performance and cost optimization in microservices](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices); here the point is that the chatty chain is usually a *boundary* smell, and the durable fix is to move the boundary, not to micro-optimize the calls.

### 6. Microservices without the prerequisites

**Symptom:** You adopted microservices but you deploy them by hand, one at a time, with a two-hour coordination call. You have no distributed tracing, so when a request is slow you have no idea which service is to blame. You have no service mesh, no centralized logging, no platform team. Every incident is an archaeology dig. Your "microservices" are slower to ship than the monolith they replaced.

**Cause:** You bought the architecture without building the foundation it stands on. Microservices are not a free-floating idea; they sit on top of a platform of automation and observability. Without [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) you cannot deploy services independently, which is the *entire point* — so you get the cost (many deployables) without the benefit (independent deploys). Without observability you are blind across the network boundary. Figure 8 is the stack: the fleet on top is only stable because of the layers underneath it.

![A layered stack showing a microservices fleet resting on ownership, observability, platform automation, and CI/CD, with a danger layer at the bottom warning that no platform means distributed pain](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-8.webp)

**Fix:** Build the platform *first*, or do not split. This is non-negotiable, and it is the single most common reason a small team's microservices migration becomes a disaster. The worked example below quantifies exactly how a team without CI/CD and observability ends up slower after splitting. If you do not have these prerequisites and cannot build them, the modular monolith is not a compromise — it is the correct architecture for your maturity level. Stay there until the foundation exists.

### 7. Premature decomposition — baking in the wrong boundaries

**Symptom:** Your service boundaries keep being wrong. A "simple" feature requires changes across four services because the thing it touches was split across all four. You spend more time redrawing boundaries (moving code between services, migrating data) than building features. Every quarter someone proposes merging two services or splitting one.

**Cause:** You split before you understood the domain. Service boundaries are the hardest, highest-stakes decision in the architecture, and you made them when you knew the *least* — at the start, when the domain was still being discovered. A wrong boundary is far more expensive to fix in microservices than in a monolith, because moving logic across a service boundary means moving data across a database boundary and rewriting network contracts, where in a monolith it is a refactor your IDE can do. This is precisely why [monolith-first](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) is the default senior advice: build a modular monolith, let the seams emerge from real usage, and extract services along seams you have *observed* rather than guessed.

**Fix:** If the boundaries are wrong, you must redraw them — and redrawing often means *merging back* to a coarser grain where the domain is still fuzzy, then re-splitting later along better seams. Do not be precious about it. The cost of a wrong boundary compounds every single day you keep it; the cost of merging back is a one-time refactor. The math overwhelmingly favors the merge. There is a worked example on the cost of premature decomposition below.

### 8. Resume-driven and hype-driven decomposition

**Symptom:** You cannot articulate, in one sentence, the *business* problem the split solves. The justification is "this is how Netflix does it," or "microservices are how you scale," or — said quietly — "it'll look good on the team's roadmap / my resume." The architecture serves the engineers' career goals or the industry's fashion, not the product.

**Cause:** Microservices became a status symbol. For about a decade, "we're moving to microservices" signaled that you were a serious, scaling engineering org, and "we're a monolith" signaled that you were behind. This is backwards. The senior signal is choosing the *simplest architecture that solves your actual problem* — and for most companies, most of the time, that is a modular monolith.

**Fix:** Apply the decision matrix (figure 2, next section). If you cannot point to a concrete benefit — *these* specific teams will deploy independently, *this* specific service needs to scale 50× while the rest does not — you do not have a reason to split. "It's modern" is not a reason. "Netflix does it" is not a reason; Netflix has thousands of engineers and runs at planetary scale, and you, dear reader, almost certainly do not.

### 9. The ground-up microservices startup

**Symptom:** A brand-new company, pre-product-market-fit, building twenty services on day one. They have more services than customers. They spend their precious early runway on infrastructure instead of finding out whether anyone wants the product.

**Cause:** The belief that you should "build it right from the start" and that microservices are "right." But you do not yet know what the product *is*, which means you do not know what the boundaries are, which means you are guaranteed to draw them wrong — premature decomposition with extra steps. And you are spending the scarcest resource a startup has (time before the money runs out) on operational overhead instead of learning.

**Fix:** Start with a monolith. Always, basically always, for a new product. [What are microservices and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them) makes the full case; the short version is that a startup's job is to find product-market fit as fast as possible, and a monolith is the fastest way to change everything — which is what you do constantly before PMF. You split *after* you have a product, traffic that needs it, and teams that need to deploy independently. Not before.

### The anti-pattern checklist

Here is the catalog as a checklist you can run against your own system in five minutes. Each "yes" is a red flag; two or three is a diagnosis.

```yaml
# anti-pattern self-audit — answer honestly, count the "yes"es
distributed_monolith:
  - "Do services routinely deploy together in a fixed order?"   # master anti-pattern
  - "Does changing one service's API force changes in others?"
  - "Is there a release train because nothing ships alone?"
nanoservices:
  - "Is there a service with more YAML/Dockerfile than business logic?"
  - "Do you have more than ~3 services per engineer (with automation)?"
  - "Does a service wrap a single table with bare CRUD?"
shared_database:
  - "Do multiple services read/write the same tables?"
  - "Does a schema migration require cross-team coordination?"
  - "Can you NOT name the single owner of a given table?"
god_service:
  - "Is there one service every request depends on?"            # usually user/account
  - "Did you decompose by ENTITY (User, Order) not CAPABILITY?"
chatty_chain:
  - "Does one user action make 5+ synchronous hops in series?"
  - "Is there a per-item downstream call inside a loop (N+1)?"
missing_prerequisites:
  - "Do you deploy services by hand / with a coordination call?"
  - "When a request is slow, can you NOT see which hop is to blame?"
premature_or_hype:
  - "Have you redrawn a service boundary more than once?"
  - "Can you NOT state the business problem the split solves?"
# 0-1 yes: healthy. 2-3: investigate. 4+: you are over-split — start the consolidation playbook.
```

Print it, run it in a design review, and let the count make the argument for you. Numbers defuse the emotional attachment people develop to their service count.

## Should you do microservices? The decision matrix

Here is the scorecard. For each axis, where does a modular monolith win and where do microservices win? This is the explicit "should you?" decision, and figure 2 is the visual version.

![A decision matrix scoring five axes against few-teams-low-scale versus many-teams-high-scale, showing the modular monolith wins on most axes until both team count and scale are high](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-2.webp)

| Axis | Modular monolith wins when | Microservices win when |
| --- | --- | --- |
| **Team count** | 1–3 teams; everyone can coordinate in one codebase | Many teams (5+) that need to deploy without colliding |
| **Traffic / scale** | One process can serve the load; uniform scaling is fine | Hot paths need to scale 10–50× independently of cold paths |
| **Domain clarity** | Domain is new or shifting; boundaries unknown | Domain is well-understood; stable bounded contexts |
| **Operational maturity** | No platform team, manual deploys, thin observability | CI/CD, tracing, mesh, platform automation already in place |
| **Deploy-independence need** | Releasing the whole app together is acceptable | Teams are blocked waiting on each other's releases |
| **Fault isolation need** | A crash taking the app down is tolerable | One component must not be able to take down the rest |

Read the matrix as an **AND, not an OR**. Microservices are the right call when you have *many teams* AND *high scale* AND *operational maturity* AND a *clear domain*. Miss any of those and the modular monolith is usually the better engineering choice. The most common real situation — a 3–8 person team, moderate traffic, a domain still being discovered, and no platform — points squarely at the modular monolith, and the matrix says so on every row.

Notice the one row that is **required in both columns**: operational maturity. There is no scale of traffic or team count at which you can run microservices *without* CI/CD and observability. If that row is red, you are not ready, full stop — go build the platform or stay on the monolith.

#### Worked example: the operational-maturity prerequisite

Let me make the prerequisite concrete with numbers, because this is the failure I see most often.

A team of 6 engineers runs a monolith. Deploys take 12 minutes via a single pipeline. Mean time to diagnose a production issue is ~30 minutes, because every error is one stack trace in one log stream and a senior engineer can usually eyeball it. They ship roughly **15 features per week**.

They read the hype and split into 12 services. But they skip the prerequisites — no per-service CI/CD (they deploy by hand), no distributed tracing (each service logs to its own stdout). Watch what happens to each metric:

- **Deploy.** A feature that touches 3 services now needs 3 manual, ordered deploys plus a coordination call. Deploy time per feature goes from 12 minutes to roughly **90 minutes** including coordination. Worse, with no contract testing, ordering mistakes cause a broken-deploy incident about once a week.
- **Diagnosis.** A slow request is now spread across 3–4 services with no correlation IDs and no trace. Mean time to diagnose goes from 30 minutes to **3+ hours** of cross-grepping logs by timestamp. On-call burnout sets in within two months.
- **Throughput.** Between the deploy tax, the incident tax, and the debugging tax, feature throughput drops from 15/week to about **6/week**. The team got *60% slower* by adopting the architecture that was supposed to make them faster.

The architecture was not the problem; the *missing platform underneath it* was. Figure 7 is the contrast — without CI/CD and observability, the same team ships changes more safely on a modular monolith than on a service fleet they cannot operate.

![A before-and-after diagram comparing microservices without operational maturity against a modular monolith, showing faster single deploys, one log stream, and easier in-process debugging on the monolith side](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-7.webp)

The lesson, stated as a rule: **the operational maturity to run N services must precede the split to N services. If you cannot deploy and observe a service fleet, you do not have microservices — you have a monolith you have to deploy N times and debug blind.**

## How to know you over-split: the signals

Diagnosis comes before treatment. Here are the concrete signals that you have over-split, in rough order of how reliably they predict it. None is conclusive alone; two or three together is a clear verdict.

1. **You spend more time on glue and ops than on features.** Track it for a sprint. If "wiring service A to talk to service B," "chasing latency across the call graph," "coordinating a multi-service deploy," and "debugging a cross-service issue" together exceed feature work, you are over-split. This is the single strongest signal.
2. **Every feature touches many services.** A healthy split means most features live inside one service. If your average feature requires changes across four or five services, your boundaries are wrong — the things that change together are spread across services that should be one.
3. **Latency and cost ballooned after the split.** If p99 went up and your cloud bill went up after you split — more network hops, more idle service overhead, more inter-service traffic — and you did not get a proportional benefit, you paid the tax for nothing.
4. **A small team is drowning in services.** A rough heuristic: a team can sustainably own and operate **2–4 services per engineer** *with* good platform automation, and fewer without it. Three engineers running forty services (ShopFast) is more than 13 services per engineer — wildly over the line. The on-call burden alone will burn the team out.
5. **Services co-deploy constantly.** If you watch your deploy history and the same two or three services almost always go out together, they are coupled and should probably be merged. This one you can *measure*, and the script below does exactly that.

Here is the deploy-coupling detector. Point it at your deploy log (here a simple CSV of `timestamp,service`), and it tells you which service *pairs* co-deploy within a short window — your merge candidates.

```python
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

# CSV rows: deploy_timestamp_iso, service_name
WINDOW = timedelta(minutes=30)   # deploys within this window = "co-deployed"

def load_deploys(path):
    rows = []
    with open(path) as f:
        for ts, svc in csv.reader(f):
            rows.append((datetime.fromisoformat(ts), svc))
    return sorted(rows)

def co_deploy_pairs(rows):
    # For each deploy, find deploys of OTHER services within WINDOW.
    pair_counts = defaultdict(int)
    deploy_counts = defaultdict(int)
    for ts, svc in rows:
        deploy_counts[svc] += 1
    n = len(rows)
    for i in range(n):
        ts_i, svc_i = rows[i]
        nearby = {svc_i}
        j = i + 1
        while j < n and rows[j][0] - ts_i <= WINDOW:
            nearby.add(rows[j][1])
            j += 1
        for a, b in combinations(sorted(nearby), 2):
            pair_counts[(a, b)] += 1
    return pair_counts, deploy_counts

def report(path):
    rows = load_deploys(path)
    pairs, counts = co_deploy_pairs(rows)
    print(f"{'PAIR':<40} {'CO-DEPLOYS':>10} {'COUPLING':>9}")
    for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1]):
        # coupling = how often the LESS-deployed of the pair shipped WITH the other
        denom = min(counts[a], counts[b]) or 1
        coupling = c / denom
        if coupling >= 0.70:   # the merge-candidate threshold
            print(f"{a + ' + ' + b:<40} {c:>10} {coupling:>8.0%}")

if __name__ == "__main__":
    report("deploys.csv")
```

If two services co-deploy in 70% or more of the less-frequent one's releases, they are not independent — they are one unit of change pretending to be two. That pair is your highest-value merge candidate. Run this monthly; it is the cheapest architecture-health signal you have.

And here is the service-count-vs-team-size sanity check — a heuristic, not a law, but it catches the egregious cases like ShopFast:

```python
def service_budget(num_engineers, has_cicd, has_observability, has_platform_team):
    # Baseline sustainable ownership: ~2 services/engineer with full automation.
    base_per_eng = 2.0
    # Each missing prerequisite slashes how many services a team can sustainably run.
    if not has_cicd:          base_per_eng *= 0.4
    if not has_observability: base_per_eng *= 0.5
    if not has_platform_team: base_per_eng *= 0.6
    budget = round(num_engineers * base_per_eng)
    return max(budget, 1)

def verdict(num_services, num_engineers, **prereqs):
    budget = service_budget(num_engineers, **prereqs)
    ratio = num_services / budget
    if ratio <= 1.0:   status = "OK"
    elif ratio <= 1.5: status = "STRETCHED"
    else:              status = "OVER-SPLIT — consolidate"
    print(f"{num_services} services, budget {budget} -> {status} ({ratio:.1f}x)")

# ShopFast today:
verdict(40, 3, has_cicd=False, has_observability=False, has_platform_team=False)
# -> 40 services, budget 1 -> OVER-SPLIT — consolidate (40.0x)

# ShopFast after building the platform AND consolidating:
verdict(6, 3, has_cicd=True, has_observability=True, has_platform_team=False)
# -> 6 services, budget 4 -> STRETCHED (1.5x)
```

ShopFast scores **40× over budget** today. Even after they build CI/CD and observability, forty services for three engineers is hopeless — the only fix is to consolidate. The script makes the case in one line, which is exactly what you want when you are arguing for a consolidation in a design review against someone emotionally attached to the service count.

### The organizational signals — the ones that hurt most

The five signals above are technical and measurable. There is a second tier that is organizational, harder to put a number on, and ultimately more damaging because it costs you people. Watch for these:

- **On-call is a punishment.** When engineers dread their on-call week because forty services means a page every couple of hours, you are bleeding morale and, eventually, headcount. Sustainable on-call is the single best proxy for "do we have the right number of services for this team?" If it is unsustainable, you are over-split regardless of what the architecture diagram looks like.
- **Nobody owns the whole picture.** When a customer-facing bug spans four services and four engineers each say "not my service," and it takes a war room to figure out where the bug actually lives, you have spread one coherent responsibility across too many boundaries. Ownership fragmented past the point of accountability.
- **New engineers take months to ship.** In a healthy monolith a new hire ships something real in their first week. If onboarding to your microservices fleet takes a month of "which service do I even change for this?", the cognitive map is too large for the value it delivers.
- **Every estimate has a "coordination" line item.** When your sprint planning routinely includes "and then we coordinate the deploy across teams," the deploy independence you split for does not actually exist — you got the boundaries without the independence, which is the distributed monolith wearing a project-management hat.

These signals matter because the whole point of microservices was to let an organization *scale its people*. If the architecture is instead burning your people — through on-call, through fragmented ownership, through slow onboarding — it is failing at the one job that justified its cost, and that is the loudest possible signal to consolidate.

## The consolidation move: merging back is maturity, not failure

Now the heart of the post. You have diagnosed over-splitting. What do you actually *do*?

You consolidate — you merge services back into coarser services or a modular monolith. And the first thing I want to do is **kill the shame around it**, because that shame is the main reason teams keep suffering with too many services. Consolidating is not admitting failure. It is responding to evidence. The teams that get this right treat their service count as a *dial they can turn in both directions* based on data, not a one-way ratchet that only ever goes up. Turning the dial down when the data says to is one of the most senior moves in this entire series.

Figure 9 is the decision flow for *which* services to merge. It comes down to two signals you can measure: **co-deployment** (do they ship together?) and **chattiness** (do they call each other synchronously on a hot path?). Independent services that rarely co-change stay split.

![A branching decision graph showing that services which co-deploy should merge, synchronously chatty services should merge or go async, and independent services stay split](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-9.webp)

The consolidation playbook, step by step:

1. **Identify the merge clusters.** Run the co-deploy detector and the trace analysis. Group services that co-deploy frequently or sit on the same synchronous hot path. For ShopFast, the clusters were obvious: pricing + tax + promotions (always changed together, all on checkout's path), and cart + wishlist + saved-items (one team, one data model, three deployables).
2. **Pick the target grain.** For each cluster, decide: merge into one coarser *service*, or fold back into a *modular monolith module*? If the cluster is genuinely high-scale and team-owned, a coarser service is right. If it is just operational overhead with no scaling benefit, a monolith module is right. ShopFast went to six services for the parts that needed independent scaling and a single modular monolith for the long tail of admin/reporting nanoservices.
3. **Merge the code first, the data second.** Pull the services' code into one deployable, turning the network calls between them into in-process function calls immediately — that alone kills the latency and the deploy coupling. Then unify or clean up the data: where they shared a database, that is now fine (it is one service again); where they each had a database, you merge schemas carefully.
4. **Keep the API stable.** Consumers of the old services should not notice. The merged service exposes the same endpoints (or you put a thin facade in front during the transition). This is the strangler fig in reverse, and the techniques transfer directly from [strangler fig: migrating a monolith to microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) — you can consolidate incrementally, one cluster at a time, behind a stable interface, with no big bang.
5. **Measure the win.** Latency, cost, deploy coordination, on-call burden — before and after. If you cannot show the win, you merged the wrong things; back up and re-cluster.

The fear most teams have about merging back to a monolith is regression to the *big ball of mud* — the unstructured tangle that microservices were partly a reaction to. This fear is legitimate but solvable: the target is a **modular monolith**, where the boundaries you learned the hard way are *preserved as enforced module boundaries inside one deployable* rather than thrown away. You keep the clean seams; you just stop paying the network-and-pipeline tax to enforce them. The enforcement moves from "the network won't let module A touch module B's data" to "the build won't let module A import module B's internals." Tooling like Shopify's Packwerk (for Ruby), Java's module system or ArchUnit, or simple dependency-linting in any language makes those boundaries *fail the build* if violated:

```ruby
# package.yml — a modular-monolith module boundary (Packwerk style)
# the "pricing" module: its internals are private; only its public API leaks out
enforce_privacy: true              # other modules can't reach into internals
enforce_dependencies: true         # declared deps only — no surprise coupling
dependencies:
  - "modules/catalog"              # pricing may depend on catalog's public API
  - "modules/promotions"           # ...and promotions
# NOT allowed to depend on: checkout, payment, fulfillment.
# A violating `require` fails CI — the boundary is real without a network hop.
```

That single config does what a service boundary did — keeps modules from reaching into each other's guts and creating hidden coupling — but at build time, for free, with zero latency cost and zero deploy coordination. This is the crux of why the modular monolith is so often the right target for a consolidation: **modularity was always the goal; the network was just one expensive way to enforce it.** When you can enforce it at the build instead, you keep the discipline and delete the tax.

Here is what merging three nanoservices into one looks like in code — the network calls simply become method calls. Before, the checkout service made three HTTP round trips:

```python
# BEFORE: three network hops, three failure points, three timeouts to tune
async def price_cart(cart):
    base = await http_get(f"http://pricing-svc/price", json=cart)        # hop 1
    tax  = await http_get(f"http://tax-svc/tax", json={**base})          # hop 2
    promo= await http_get(f"http://promo-svc/apply", json={**tax})       # hop 3
    return promo  # p99 of this block ~= sum of three p99s + 3x network
```

After merging pricing, tax, and promotions into one `pricing` service, the same logic is three function calls in one process:

```python
# AFTER: one service, in-process calls, one transaction, one deploy
class PricingService:
    def price_cart(self, cart):
        base  = self._base_price(cart)      # in-process: ~microseconds
        taxed = self._apply_tax(base)       # in-process
        final = self._apply_promos(taxed)   # in-process
        return final                        # p99 collapses; one failure domain
```

The eight-hop checkout in figure 4 becomes a two-hop checkout: gateway → checkout service (which now does pricing in-process) → payment. Seven of the eight network hops are gone. The numbers are the next worked example.

When you do the merge in practice, the riskiest part is the data, not the code. If pricing, tax, and promotions each had their own small database, you have to unify them, and the safe way is to migrate the data behind the still-running old endpoints, verify row counts match, then cut over. A pragmatic merge plan, expressed as a deployment manifest plus a migration step, looks like this:

```yaml
# consolidation plan: pricing + tax + promotions -> one "pricing" service
merge:
  target_service: pricing
  absorbs: [tax-svc, promo-svc]
  strategy: incremental          # never big-bang; one cluster at a time
  steps:
    - move_code:                 # pull the logic into one repo/module
        from: [tax-svc, promo-svc]
        into: pricing/internal/{tax,promos}   # in-process packages now
    - data:
        # tax & promo had tiny own DBs; fold their tables into pricing's schema
        migrate: [tax_rules, promo_rules]
        verify: row_counts_match    # gate the cutover on parity
    - facade:
        # keep old URLs alive during transition so callers don't change
        route: { "tax-svc/*": "pricing/tax/*", "promo-svc/*": "pricing/promos/*" }
    - retire:
        when: facade_traffic == 0   # decommission only after callers move
        delete: [tax-svc, promo-svc]   # delete the per-service overhead
  expected_win:
    checkout_hops: "5 -> 2"
    p99_ms: "~700 -> ~180"
    deployables_removed: 2
```

The two principles that keep a consolidation safe: **keep the old interface alive behind a facade so consumers never break**, and **gate the cutover on a measurable parity check** (row counts, shadow-traffic diff) so you never lose data. Retire the old services only when their traffic hits zero. This is the strangler fig running in reverse, and it means a consolidation is just as incremental and reversible as the original split was supposed to be.

Let me spell out the *mechanics* of merging two chatty services back into one, because juniors often expect it to be a huge rewrite and it usually is not. Take two services, `tax-svc` and `promo-svc`, that the checkout service calls in series on every request. The merge is five concrete mechanical steps, in this order:

1. **Collapse the network call into an in-process call.** Move `promo-svc`'s code into `tax-svc`'s repository as an internal package. The HTTP client call `await http_get("http://promo-svc/apply", ...)` becomes a plain function call `apply_promos(...)` in the same process. This single change deletes a serialization, a round trip, a timeout to tune, and a partial-failure mode — and on ShopFast's path it cut roughly **120 ms of p99** for *that hop alone*, before any other work.
2. **Merge the two schemas.** `promo-svc` had its own tiny `promo_rules` table; fold it into the surviving service's database as `promotions.promo_rules`. The data migration is a one-time copy with a row-count parity check; because the tables had no foreign-key relationship to anything else (they were a nanoservice), this is genuinely a half-day of work, not a rewrite. Where there *are* relationships, you migrate behind the still-running old endpoint and diff the results before cutting over.
3. **Retire the duplicate deploy pipeline.** Delete `promo-svc`'s CI pipeline, CD pipeline, Kubernetes manifests, dashboard, alerts, and on-call entry. This is where the *operational* savings land — you have just deleted a whole copy of the fixed per-service tax I described at the top, while the business logic kept working unchanged.
4. **Keep one team owning the result.** Assign the merged service to a single team with a single CODEOWNERS entry. Two services that two half-teams co-owned become one service one team fully owns — the accountability fragmentation from the god-service and chatty-chain anti-patterns disappears here.
5. **Verify the win and only then delete.** Shadow-traffic the merged service against the old pair, confirm identical outputs, watch the parity dashboard for a day, then delete the old services when their traffic hits zero.

The before/after on this single two-service merge: checkout dropped from making **2 synchronous hops to 0** for pricing logic (it now calls one in-process function), p99 for the pricing block fell from roughly **~240 ms to ~8 ms**, and ShopFast deleted **2 of its 40 deployables** with their pipelines and pager rotations. That is the whole consolidation in miniature, repeated cluster by cluster until you are at six services.

#### Worked example: the 40 → 6 consolidation, with numbers

This is ShopFast's actual consolidation, with before/after metrics across the four currencies that matter.

**Before (40 nanoservices, shared DB, 3 engineers):**

| Metric | Before |
| --- | --- |
| Checkout network hops | 8 (synchronous, in series) |
| Checkout p99 latency | ~900 ms (sum of 8 hops + network + DB contention) |
| Cloud cost (compute) | ~\$9,200/month (40 services × idle baseline + replicas) |
| Deploy coordination per feature | ~90 min (multi-service ordered manual deploys) |
| On-call burden | 3 engineers covering 40 services; ~6 pages/night |
| Feature throughput | ~5/week (rest of time is glue + ops + incidents) |

**After (6 services + 1 modular monolith for the long tail, db-per-service, same 3 engineers):**

| Metric | After | Change |
| --- | --- | --- |
| Checkout network hops | 2 (gateway → checkout → payment) | −6 hops |
| Checkout p99 latency | ~180 ms | **−80%** |
| Cloud cost (compute) | ~\$5,000/month (fewer idle baselines; right-sized replicas) | **−46%** |
| Deploy coordination per feature | ~12 min (most features now touch one service) | **−87%** |
| On-call burden | 3 engineers covering 7 deployables; ~1 page/night | **−83% pages** |
| Feature throughput | ~13/week | **+160%** |

Where did the wins come from? The latency drop is almost entirely the **six eliminated network hops** — turning network calls into in-process calls. The cost drop is **eliminating the idle baseline of 34 services** (each nanoservice was running replicas with their own memory floor, sidecars, and overhead even at near-zero traffic). The deploy and on-call wins come from **most features now living in one service**, so there is nothing to coordinate. And the throughput nearly tripled because the engineers got their *time* back — they stopped doing glue and ops and started doing features again.

That is the optimization angle of this entire post: **consolidation is a performance and cost optimization.** You recover latency, cloud spend, and — most valuable of all — engineering time. The senior framing is that velocity *is* a production metric, and this team's velocity went up 2.6× by deleting 34 services.

#### Worked example: the cost of premature decomposition

The other worked example is the one that should make you cautious about splitting in the first place: what does it cost to redraw a wrong boundary?

ShopFast originally split "Orders" and "Fulfillment" as two services because they sounded like two things. In practice, almost every change touched both — placing an order, reserving stock, generating a pick list, and updating status were one tightly-coupled workflow that the early team had guessed wrong. Over six months:

- **Engineering time on the boundary:** roughly **40 engineer-days** spent moving logic back and forth across the Orders/Fulfillment line, migrating data between their two databases, versioning the API between them as it churned, and debugging the saga that spanned them. In a monolith, this would have been a series of IDE refactors — call it 4 engineer-days.
- **Incidents caused by the boundary:** the cross-service saga had compensating-transaction bugs that caused ~3 production incidents (orders stuck in a half-fulfilled state), each ~half a day of cleanup.
- **The fix:** merge Orders and Fulfillment into one service. Once merged, the workflow became a single local transaction, the saga vanished, and the boundary churn stopped.

The lesson: **a wrong boundary in microservices costs roughly 10× what the same wrong boundary costs in a monolith**, because moving the boundary means moving data and rewriting network contracts, not just moving code. This is the entire quantitative case for monolith-first: when you do not yet know the boundaries, keep them cheap to change, and a monolith's boundaries are cheap to change while a service fleet's are not. ShopFast paid ~36 engineer-days extra plus three incidents to learn a boundary they could have discovered for free inside the monolith.

## A consolidation, not a regression: the journey

I want to be precise about the *shape* of the right journey, because "go back to the monolith" can sound like surrender, and it is not. Figure 5 is ShopFast's actual timeline: monolith → hype-driven over-split → velocity wall → consolidation → recovery. The consolidated state is *not* the same as where they started. They came out the other side with a **modular monolith plus six well-bounded services**, a real CI/CD pipeline, real observability, and — crucially — *boundaries they have now observed in production* rather than guessed.

![A six-event timeline tracing a team from a fast monolith through a hype-driven split to forty nanoservices, a velocity wall, consolidation to six services, and recovered velocity with lower cost](/imgs/blogs/microservices-anti-patterns-and-when-to-go-back-to-monolith-5.webp)

It is worth naming ShopFast's worst offenders, because the pattern repeats everywhere. Three nanoservices did the most damage relative to their size. The first was a `currency-format-svc` whose entire job was to format a number as a localized currency string — pure, stateless logic that should never have crossed a network boundary, yet it sat on the checkout path and added a hop plus a failure mode to format `\$19.99`. The second was an `address-validate-svc` that wrapped a single third-party API with no logic of its own, so it was a network hop whose only job was to make another network hop. The third was a `cart-count-svc` that returned the integer count of items in a cart by querying the shared database — a service for one `SELECT COUNT(*)`. None of the three had any business logic worth isolating; each had a full Dockerfile, pipeline, and pager rotation. Folding all three into the services that actually used them removed three deployables and three failure modes in an afternoon, and nobody missed them.

The on-call numbers tell the same story even more starkly. Before consolidation, ShopFast's three engineers shared a rotation across forty services and were paged roughly **six times a night** — most pages were a nanoservice timing out under the cross-service load it should never have been carrying, or a cascading failure rippling through the eight-hop checkout. The engineer on call could not sleep, and the team lost one person to burnout-driven attrition during the worst quarter. After consolidating to seven deployables with collapsed call chains and a real circuit breaker on the payment boundary, pager volume fell to about **one page a night**, most weeks zero — an **80%+ reduction** that, more than any latency or cost number, is what actually saved the team. The lesson a senior takes from this: pager volume is a first-class architecture metric. If your on-call is unsustainable, your architecture is wrong, no matter how elegant the diagram looks.

That is the senior arc. You are not regressing to the naive monolith of year one. You are arriving at a *deliberate, modular* architecture sized to your actual team and traffic, with the option to split further *later* along seams you have measured. The dial turns both ways. Some of those six services may eventually split again as ShopFast grows to thirty engineers — and that will be correct then, because the conditions in the decision matrix will have changed. Architecture is not a destination; it is a response to your current scale, team shape, and maturity, and it should change as those change. ([Conway's law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices) is the org-side of why your service boundaries should track your team boundaries — and why, when your org is small, your service count should be small too.)

## Stress-testing the consolidation decision

A senior does not just propose a design; they try to break it. Let me stress-test the consolidation thesis against the hard cases.

**"We have 40 services and a 3-person team. What now?"** This is ShopFast, and the answer is: consolidate aggressively, but in a controlled order. Do *not* attempt a 40→1 big bang. Cluster by co-deployment and chattiness (figure 9), then merge the highest-value cluster first — for ShopFast that was the 8-hop checkout, because it had the biggest latency and the most incidents. Behind a stable gateway, merge pricing+tax+promotions into checkout, ship it, measure the latency drop, prove the win. Then take the next cluster. Each merge is small, reversible, and shows a number. Within a quarter you are at six services and the team is breathing again. The key is that *consolidation can be incremental* — the strangler fig works in reverse.

**"Every feature needs five services changed. Is that always over-splitting?"** Almost always, but check one thing first: is it a *boundary* problem or a *coordination* problem? If five services change because the feature genuinely spans five bounded contexts owned by five teams, that may be acceptable cross-team coordination (with good contract testing to decouple the deploys). But if the five services are owned by one team and change together every time, that is a boundary problem — those five things are one thing, and you should merge them. The test: *who owns the services, and do they change together by necessity (one workflow) or by accident (one team, badly split)?* Necessity might be fine; accident is over-splitting.

**"We adopted microservices and have no observability. Should we add observability or consolidate?"** Both, but in the right order. First, **stop the bleeding** — you cannot operate what you cannot see, so add at least basic distributed tracing and centralized logging immediately; this is triage. Then run the diagnosis: how over-split are you (the service-budget script), and which services co-deploy (the coupling script)? If you are wildly over budget like ShopFast, consolidate down to a number you can actually operate *while* you mature the platform. Observability and consolidation are not either/or — observability tells you *what* to consolidate, and consolidation reduces *how much* you have to observe. They reinforce each other.

**"What breaks when we merge two services that were independently scaled?"** This is the real risk of consolidation, and you must check it before merging. If service A needs 50 replicas (hot) and service B needs 2 (cold), merging them means the cold logic now scales to 50 replicas too — you waste resources, and worse, a memory leak or slow path in the cold code now runs on 50 instances. **Do not merge services with genuinely different scaling profiles.** Co-deployment and chattiness say "merge," but a divergent scaling profile says "keep split." When the signals conflict, scaling profile usually wins — that is exactly the kind of independent-scaling need that justified a service in the first place. ShopFast's payment service stayed separate for this reason (and because it is an external-facing PCI boundary): it scales and fails independently of everything else.

## Case studies

These are real, and I am framing each fairly — including the nuance that several of them are *narrow* lessons that the internet over-generalized.

**Segment's monolith U-turn.** Segment is the most-cited consolidation story, and rightly so. They split into microservices — in their case, roughly one service per destination integration, which grew into a large fleet — and hit exactly the failure modes in this post. The specific killer was a *shared-library distributed monolith*: every destination service depended on a shared codebase, so updating that shared dependency meant testing and redeploying the whole fleet, and a single failing destination could starve the others of resources. The operational overhead of running so many services per engineer became crushing, and the team spent more time on the fleet than on features. In a widely-read 2018 engineering write-up, they described consolidating back to a monolith and recovering developer productivity dramatically — fewer things to deploy, one codebase to test, one place to fix a bug. The honest lesson is *not* "microservices are bad" — it is "Segment over-split relative to their team size and operational maturity, the shared library turned their fleet into a distributed monolith, and consolidating was the correct, mature response." It is the canonical proof that the dial turns both ways, and that a *shared dependency* couples services every bit as tightly as a shared database.

**Amazon Prime Video's consolidation.** In 2023, an Amazon Prime Video team published a write-up describing how they took a specific audio/video monitoring system that had been built as distributed serverless microservices (Step Functions plus Lambda) and consolidated it into a single process, cutting infrastructure cost for that system by around 90%. The internet briefly declared "Amazon abandons microservices," which is wrong and unfair — this was *one team optimizing one cost-sensitive, high-throughput data-processing workload* where the serverless orchestration overhead and inter-component data transfer dominated the cost. The accurate lesson is precise: **for a high-throughput pipeline where the overhead of distribution exceeds the benefit, consolidating into one process can be a 10× cost win.** It is a worked example of the chatty-chain and nanoservice anti-patterns being *expensive*, and consolidation being the optimization — exactly this post's thesis, applied at Amazon's scale.

**Istio's control-plane consolidation.** Istio, the service mesh, originally shipped its control plane as several separate microservices (Pilot, Mixer, Citadel, Galley). Operating and upgrading that fleet of control-plane components was painful, and in Istio 1.5 (2020) the project consolidated them into a single binary called `istiod`. This is a beautiful example because the people who *build* microservices tooling decided their own *control plane* was over-split — the components co-deployed, co-versioned, and gained nothing from being separate. Consolidating made Istio dramatically simpler to install and operate. ([Service mesh: when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) covers the mesh itself.)

**Shopify's majestic monolith.** Shopify is the counterweight to the counterweight: a company operating at genuinely massive scale that has *deliberately stayed a monolith* — a Ruby on Rails "majestic monolith" — for its core commerce platform. Rather than splitting into microservices, they invested heavily in **modularity within the monolith** (componentization, enforced module boundaries, their "Packwerk" tooling for static boundary enforcement). The lesson is the most important one in this post: **you can get clean boundaries, team ownership, and a system you can change safely without paying the distribution tax — by building a *modular* monolith.** Modularity is the goal; microservices are just one way to enforce it, and not always the cheapest. Shopify proves that "monolith" and "scale" are not opposites.

**The counter-case — Monzo, where many services is right.** It is only fair to show the other side, so you do not over-correct into "always consolidate." Monzo, the UK bank, runs well over a thousand microservices, and for them it is the *correct* architecture — and notice why, because it maps exactly onto the decision matrix. They have hundreds of engineers (many teams that need to deploy independently), they operate a regulated bank where fault isolation is genuinely life-or-death for the business (a bug in one service must not take down payments), and — critically — they invested enormously in *platform* first: a deployment tool that makes shipping a service trivial, uniform service templates so every service looks the same, and deep observability. Monzo can run 1,500 services because they have all four matrix conditions maxed out: many teams, high scale, hard fault-isolation needs, and world-class operational maturity. They are not a counter-example to this post; they are the *proof of the rule* — microservices at extreme scale work when, and only when, the prerequisites and the genuine need are both present. ShopFast had neither, which is why the same architecture that serves Monzo destroyed ShopFast.

The through-line across all five: the service count is a tunable parameter, not a one-way ratchet, and the most sophisticated engineering orgs in the world turn it both *down* (Segment, Prime Video, Istio) and *up* (Monzo) based on whether they meet the conditions. None of the consolidators is embarrassed; all of them published the story as a *win*. And the company running the most services on the list got there by satisfying every condition first.

## When microservices are the wrong choice

Let me be decisive, because hedging here helps no one. Microservices are the **wrong** choice when:

- **You are a startup before product-market fit.** You do not know your boundaries; you will draw them wrong; you will spend runway you cannot spare on operational overhead. Build a monolith and find out if anyone wants your product.
- **You have a small team (roughly under 8–10 engineers).** You cannot operate enough services to justify the overhead, and a single codebase lets a small team move fastest. The coordination cost microservices solve does not exist yet — you can all just talk to each other.
- **You lack the operational platform.** No CI/CD, no observability, no platform automation. Without these, microservices are strictly slower and more dangerous than a monolith. Build the platform first or do not split.
- **Your domain is still being discovered.** If your boundaries are shifting quarter to quarter, baking them into network-and-database boundaries makes every wrong guess 10× more expensive to fix. Keep them cheap to change — in a monolith — until they stabilize.
- **You are doing it for hype, resume, or status.** If you cannot name the concrete business problem the split solves, you do not have a reason. "It's modern" is not a reason.
- **Your load is uniform and modest.** If you do not have hot paths that need to scale 10–50× independently of cold paths, independent scalability — half the entire value proposition — buys you nothing.
- **You are building a mostly-CRUD application.** If the system is fundamentally forms over a database — create, read, update, delete with some validation — there is no rich domain to carve into bounded contexts and no independent-scaling story; a monolith over one database will be simpler, faster, and cheaper to run for years, and splitting it just scatters the same CRUD across a network.
- **Your domain is genuinely small.** If the entire business fits in a few bounded contexts that one team understands end-to-end, microservices add coordination overhead to solve a coordination problem you do not have. A handful of clean modules in one process expresses the same boundaries with none of the network tax.
- **You cannot afford a platform team.** Running a service fleet implies someone owns the paved road — the deploy tooling, the mesh, the observability stack, the service templates. If you have no one to build and maintain that and no budget to hire them, every product team will reinvent it badly, and you will get forty subtly-different services nobody can operate. No platform owner means stay on the monolith.

And microservices are the **right** choice when, and roughly only when, you have all of: **many teams that are blocking each other in one codebase, high or very uneven traffic that needs independent scaling, a domain you understand well enough to draw stable boundaries, AND the operational maturity (CI/CD, observability, platform) to run a fleet.** That is a real and important situation — it is why Netflix, Uber, Amazon, and Monzo run microservices — but it is a situation most teams are not in, and pretending otherwise is how you end up as ShopFast.

The default, for most teams, most of the time, is a **modular monolith**: one deployable, clean enforced internal boundaries, ready to extract services along observed seams *if and when* the conditions above are met. Start there. Earn your way to microservices. And if you over-shot, consolidate back without shame — it is the mature move, and your latency, your cloud bill, and your team's sanity will all thank you.

## Key takeaways

1. **Microservices are a tax, not a feature.** They buy independent deployability and independent scalability, and you pay for them in operational, cognitive, and latency overhead. Only pay when you have the scale that needs it *and* the maturity to afford it.
2. **The distributed monolith is the master anti-pattern.** If two services must deploy together more often than not, they are one service. Almost every other failure collapses into this one. Measure it with the co-deploy detector.
3. **Smaller is not better — independently-changeable-by-a-team is better.** Nanoservices put more code in the Dockerfile than in the logic. The right grain is a business capability one team owns, where independent deployment is worth more than the operational cost.
4. **A shared database turns microservices into a distributed monolith.** Database-per-service is the rule that actually enforces independence; skipping the data split is the most common way the architecture quietly fails.
5. **Decompose by capability, not by entity.** Entity decomposition breeds god services that everything depends on. Real features cut across entities; boundaries should track bounded contexts and team ownership.
6. **The prerequisites come first.** No CI/CD and no observability means no microservices — you will be slower and blinder than the monolith you left. Build the platform before you split.
7. **Premature decomposition is expensive — a wrong boundary costs ~10× in services what it costs in a monolith.** When you do not know your boundaries, keep them cheap to change. That is the whole case for monolith-first.
8. **Consolidation is maturity, not failure.** The service count is a dial that turns both ways. Merging back to coarser services or a modular monolith — incrementally, behind a stable interface, with measured wins — is one of the most senior moves you can make. Segment, Amazon Prime Video, and Istio all turned the dial down and published it as a win.
9. **The goal was never microservices — it is a system you can change safely and cheaply.** Sometimes that is microservices. Very often it is a modular monolith. Choose by the matrix, not by fashion.

## Further reading

- Sam Newman, *Building Microservices* (2nd ed., O'Reilly) — the canonical text, and notably honest about when *not* to split and how to migrate incrementally.
- Sam Newman, *Monolith to Microservices* (O'Reilly) — the migration companion; the patterns reverse cleanly for consolidation.
- Chris Richardson, *Microservices Patterns* (Manning) and microservices.io — the pattern catalog, including the anti-patterns.
- Segment engineering blog, "Goodbye Microservices: From 100s of problem children to 1 superstar" (2018) — the consolidation case study, told by the team that lived it.
- Amazon Prime Video tech blog, "Scaling up the Prime Video audio/video monitoring service and reducing costs by 90%" (2023) — read it for the *precise* scope; it is a single-workload optimization, not a verdict on microservices.
- Shopify engineering, "Deconstructing the Monolith" and the Packwerk modularity tooling — how to get clean boundaries without distribution.
- This series: [monolith-first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith), [what are microservices and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them), [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), [performance and cost optimization in microservices](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices), [Conway's law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices), [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability), [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry), [strangler fig: migrating a monolith to microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices), and the capstone, [designing a complete microservices system end-to-end](/blog/software-development/microservices/designing-a-complete-microservices-system-end-to-end).

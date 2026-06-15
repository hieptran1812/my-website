---
title: "Strangler Fig: Migrating a Monolith to Microservices Without a Big-Bang Rewrite"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The practical playbook for incrementally peeling services off a monolith behind a facade, solving the data problem deliberately with CDC and expand-contract, and knowing when to stop."
tags:
  [
    "microservices",
    "strangler-fig",
    "monolith-migration",
    "distributed-systems",
    "software-architecture",
    "change-data-capture",
    "backend",
    "migration",
    "database-per-service",
    "refactoring",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-1.webp"
---

A team I advised had a five-year-old e-commerce monolith they called ShopFast. It was, by every internal measure, the most successful and most hated piece of software in the company. Successful because it processed every order, every payment, every shipping notification the business had ever made. Hated because it was a single 600,000-line Rails-and-Postgres deploy that took 40 minutes to ship, where a change to the tax calculation could break the password-reset email, where the on-call rotation was a coin flip on which of a dozen unrelated subsystems had fallen over, and where the payments code — the code that touched real money and lived under PCI scope — was tangled into the same process, the same database, and the same deploy as the marketing email blaster. They had decided, correctly, that they needed to break it up. Then they made the decision that nearly killed the company: they decided to rewrite it.

The plan was a clean-room microservices platform, built in parallel by a "tiger team," that would replace the monolith all at once on a glorious cutover weekend. Eighteen months later the new system could process about 60% of what the old one did, the monolith had grown three hundred new features in the meantime (because the business does not freeze for your rewrite), and the cutover date had slipped four times. They were carrying two systems, paying for both, and the new one was always behind. This is not an unusual story. It is the *default* story of the big-bang rewrite, and the whole point of this article is to teach you the path they should have taken instead.

![A branching diagram showing a routing facade in front of the ShopFast monolith sending each request path to either the monolith or an extracted notifications or payments service while a shared database sits behind both](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-1.webp)

The figure above is the shape of the right answer, and it is the shape the whole article builds toward. You put a thin routing facade at the edge — an API gateway or a reverse proxy — in front of the monolith. The facade looks at each request and decides, per path, whether to send it to the old monolith or to a new service you have carved off. On day one, every path routes to the monolith and the facade is a no-op. Then you extract one bounded context at a time — notifications first because it is async and low-risk, payments later because it is high-value and needs isolation — and you flip routes over to the new services one at a time, slowly, reversibly, while the business keeps shipping features the entire time. The monolith shrinks. Eventually it is "strangled" down to a small core, or to nothing, and you retire it. This is the **strangler fig** pattern, and by the end of this post you will be able to run it: choose the first service to extract, stand up the facade, solve the genuinely hard part (the data), measure your progress as a percentage of traffic, and — the senior move almost everyone forgets — know when to stop.

This is the practitioner's layer. The *mechanics* of how change-data-capture works, how the outbox pattern guarantees reliable publishing, how a saga rolls back — those live in their own deep-dives, which I link where they matter. Here the question is operational and immediate: you have a monolith, the business will not give you a feature freeze, and you need to come out the other side with a system you can actually run. How do you do that without betting the company on a cutover weekend?

## 1. Why the big-bang rewrite almost always fails

Let me be precise about *why* the rewrite fails, because "rewrites are bad" is a slogan, not an argument, and there are real cases where a rewrite is the right call. The big-bang rewrite — replace the whole system at once with a new one — fails for a small number of structural reasons that compound.

**You freeze features, but the business does not freeze.** A rewrite that aims to replace the whole system needs the whole system finished before it can ship anything. That means months — usually many — during which the new system delivers zero value to a single user. But your competitors keep shipping, your customers keep asking for things, and the business keeps making commitments. So one of two things happens: either you genuinely freeze the old system (and the business revolts, because they cannot stop selling for nine months while you refactor), or — far more common — the old system keeps getting features bolted on *while you rewrite*, which means your rewrite is chasing a moving target. Every feature added to the monolith is a feature your new system now also has to build, except it is permanently behind. This is the single deadliest dynamic in the whole genre.

**The new system never catches up.** Joel Spolsky's famous 2000 essay "Things You Should Never Do" called the rewrite "the single worst strategic mistake that any software company can make," and his core observation has aged perfectly: the old code is *ugly* but it is *correct*. Every weird conditional, every special case, every `if customer_id == 4471` hack you sneer at — each of those is a bug fix, a hard-won lesson, a real customer edge case the old system survived. When you rewrite, you throw away years of accumulated correctness and rediscover every one of those bugs the hard way, in production, on new code. The "messy" old code is messy *because reality is messy*, and the clean rewrite is clean *because it does not yet know about reality*.

**You carry both systems.** Until the cutover, you are running and paying for two systems — two sets of infrastructure, two on-call rotations, two places a feature might need to be built. And because the cutover is all-or-nothing, you cannot retire the old one until the new one is 100% done, so you carry that double cost for the entire (long, slipping) duration. The Netscape lesson — the one Spolsky was writing about — is the canonical telling: Netscape rewrote their browser engine from scratch for version 6, spent roughly three years on it, shipped nothing usable in the meantime, watched their market share collapse, and effectively never recovered. They are the cautionary tale precisely because the rewrite was technically reasonable and strategically fatal.

**The risk is concentrated at the worst possible moment.** A big-bang cutover puts all of the risk — data migration, behavior parity, performance, every integration — into a single event, usually a weekend, usually after everyone is exhausted from the eighteen-month march. If anything is wrong, it is all wrong at once, in production, with the old system already being decommissioned. There is rarely a clean rollback because the data has moved. This is the opposite of how you want to ship anything risky: you want risk spread thin across many small reversible steps, not concentrated into one irreversible leap.

![A before and after comparison contrasting a big-bang rewrite that freezes features for nine months and ships value only at a risky cutover against a strangler fig that ships a slice every few weeks and keeps shipping features in parallel](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-2.webp)

The figure contrasts the two risk profiles directly. On the left, the big-bang rewrite: a long feature freeze, a new system chasing a moving target, and all the risk dumped into one cutover. On the right, the strangler fig: ship a real slice every two to four weeks, keep shipping features on the monolith in parallel, and make every cutover a reversible per-route flip. The strangler is not "the slow safe option versus the fast risky option." It is *both safer and delivers value sooner*, because the first extracted service is in production earning its keep in weeks, not at the end of an eighteen-month march that may never end.

When is a rewrite actually right? When the system is small enough that "the whole thing" is a few weeks of work, not years — then the big-bang and the strangler converge and the ceremony of incrementalism is not worth it. Or when the old system is genuinely unmaintainable at the language/platform level (a dead language, an unsupported runtime) *and* small. The rewrite fails specifically for *large, valuable, actively-used* systems, which is exactly the kind of system anyone reading this is trying to migrate.

## 2. The strangler fig pattern

The name comes from Martin Fowler, who borrowed it from the strangler fig vines he saw in Australia. The vine seeds in the canopy of a host tree, grows roots down around the trunk, and gradually envelops the host until the original tree dies and rots away, leaving the fig standing in its exact shape. Fowler's "StranglerFigApplication" pattern (he originally called it "StranglerApplication" in 2004) applies the metaphor to software: **you grow the new system around the edges of the old one, gradually moving functionality across, until the old system is enveloped and can be removed.**

The mechanics are simple to state and the whole rest of this article is about doing them well:

1. **Put a facade at the edge.** Insert a routing layer — a proxy or API gateway — between the clients and the monolith. At first it routes everything to the monolith. It is transparent; clients do not change.
2. **Extract one slice.** Pick one bounded context. Build it as a new service. The service implements the same external behavior for the routes it owns.
3. **Route that slice to the new service.** Flip the facade so the paths that slice owns go to the new service instead of the monolith. Do this gradually (0% → 1% → 10% → 100%) so you can catch problems at low blast radius and reverse instantly.
4. **Repeat.** Extract the next slice. The monolith shrinks with each one.
5. **Retire (or stop).** When the monolith is reduced to nothing — or to a small, well-factored core you have decided is fine to keep — you remove the dead code or freeze it.

The reason this works where the big-bang fails is that **every step ships value and every step is reversible.** The first extracted service goes to production and starts earning the migration's keep in weeks. If a slice has a bug, you flip its route back to the monolith and you are safe — the old code is still there, still working, because you have not deleted it yet. You delete the old code only *after* the new service has carried 100% of traffic cleanly for long enough to trust it. The migration de-risks continuously instead of concentrating risk at the end.

The metaphor carries one more lesson that people miss: the fig does not strangle the *whole* tree in a day, and it does not necessarily strangle *every* tree. Some monoliths get strangled to nothing; many get strangled down to a small core that turns out to be perfectly fine to keep — which is the topic of the last section. The pattern is incremental envelopment, and incrementalism means you get to stop and evaluate at every step, including the step where you decide you are done.

There is an important sibling pattern for the parts of the monolith that do not have a clean *edge* you can route at — internal code that other internal code calls directly, not over HTTP. For those you cannot put a facade at the network edge because there is no network call to intercept. You reach for a different refactoring discipline instead, covered in section 7: you introduce an internal interface *inside* the monolith and swap the implementation behind it without a long-lived branch. Strangler fig is for the network edges; that in-process technique is for the internal seams. A real migration uses both.

## 3. The facade: the routing layer that makes it all possible

The facade is the single most important piece of machinery, and it is mechanically simple: a thing that sits in front of the monolith and decides, per request, where the request goes. In practice it is a reverse proxy (Nginx, Envoy) or an API gateway, and the routing decision is "does this path/method belong to a service we have extracted yet, and if so, what fraction of this traffic should go to the new service?" The gateway is the same component this series covers in depth in [The API Gateway and Backend for Frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend); here we are using it specifically as the migration's control point.

The simplest possible facade is a path-based routing rule. Here is an Nginx config that fronts ShopFast: everything goes to the monolith *except* `/notify`, which we have extracted to a new notifications service.

```nginx
# facade.conf — routes old-vs-new per path
upstream monolith   { server shopfast-monolith:8080; }
upstream notif_svc  { server notifications-svc:9090; }

server {
  listen 80;

  # extracted slice: notifications go to the new service
  location /notify/ {
    proxy_pass http://notif_svc/;
    proxy_set_header X-Request-Id $request_id;
  }

  # everything else still goes to the monolith
  location / {
    proxy_pass http://monolith/;
    proxy_set_header X-Request-Id $request_id;
  }
}
```

That is a real strangler facade. The client's URL did not change; `/notify/send` still works; it just lands on a different process now. But a static path rule cannot do a *gradual* ramp, and the gradual ramp is what keeps the migration safe. For that you want the routing decision to be dynamic — driven by a config or a feature-flag service so you can change the split without a redeploy. Here is the same idea as an Envoy weighted route, which lets us send a *percentage* of `/pay` traffic to the new payments service while the rest stays on the monolith:

```yaml
# envoy: weighted routing for the payments ramp (10% to the new svc)
routes:
  - match: { prefix: "/pay" }
    route:
      weighted_clusters:
        clusters:
          - name: payments_svc   # the new service
            weight: 10
          - name: monolith       # the safe fallback
            weight: 90
  - match: { prefix: "/" }       # everything else
    route: { cluster: monolith }
```

![A layered stack showing the transition edge built from a path router, a feature flag controlling the ramp percentage, a shadow comparator that verifies the new path, the new service as the canary target, and the monolith route as a safe fallback](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-6.webp)

The figure shows the facade as it actually grows up: not one component but a thin stack. A **path router** decides which slice a request belongs to. A **feature flag** controls the ramp percentage so you can go 1% → 10% → 100% without a deploy and, critically, flip back to 0% in seconds if something breaks. A **shadow comparator** (section 8) can fork a copy of read traffic to the new service and diff the responses *without* the new service actually serving the user, so you verify correctness before you risk a single real request. Behind it sit the new service (the canary target) and the monolith route (the safe fallback that is always one flag-flip away).

The discipline this buys you is enormous. Because routing is dynamic and percentage-based, your *unit of migration risk* is one percent of one path's traffic, reversible in seconds. Compare that to the big-bang's unit of risk: the entire system, irreversible. The facade is how you convert a terrifying migration into a long sequence of boring, safe, individually-reversible steps. It is worth over-investing in it early — a good routing-and-flag layer is the highest-leverage thing you build in the whole migration.

There is a category of detail that bites every real migration at the facade, and it is worth naming because the textbook diagrams hide it: **the cross-cutting state the monolith assumed it owned.** A monolith typically authenticates a request once, attaches a session, and then every internal module reads from that in-process session object for free. The moment a request can land on *either* the monolith *or* a new service, that free shared context evaporates — the new service has no access to the monolith's in-memory session. So the facade has to take on a job the monolith never had to think about: turning the request's auth into something *both* backends can read. In practice that means the facade (or a thin auth layer in front of it) validates the session or token once and forwards a normalized identity downstream — a signed header or a propagated JWT — so the extracted service can authorize without reaching back into the monolith. This is exactly the token-propagation problem the series covers in [Authentication and Authorization: OAuth2, JWT, and Token Propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation), and it surfaces *immediately* at your first extraction, not someday. If you forget it, your first extracted service either cannot tell who the user is or — worse — trusts an unvalidated header.

```nginx
# the facade normalizes identity so BOTH backends can authorize
location /notify/ {
  # auth_request validates the monolith session once, at the edge
  auth_request /_auth;
  auth_request_set $user_id $upstream_http_x_user_id;
  proxy_set_header X-User-Id   $user_id;          # forward normalized identity
  proxy_set_header X-Request-Id $request_id;       # one trace id across old+new
  proxy_pass http://notif_svc/;
}
```

The same logic applies to the **request id / trace context**: during a migration a single user action may bounce between the monolith and one or more new services, and if you do not propagate one correlation id across the boundary (the `X-Request-Id` above), you lose the ability to follow a request across the seam — precisely when you most need it, because the seam is where the new failures live. Propagating a trace id from the facade through every backend is the cheap prerequisite for the [distributed tracing](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) you will lean on heavily during the transition. The senior habit is to set up identity-forwarding and trace-id propagation *before* the first extraction, as part of standing up the facade, because retrofitting them after you have three services in flight is miserable.

One more facade-level decision people get wrong: **where to draw the routing granularity.** You can route at the path level (`/notify/*` to the new service), the method level (`POST /orders` new, `GET /orders` still monolith — useful for splitting reads from writes), or even the *content* level (route based on a header or a percentage). Coarser is simpler and safer to reason about; finer lets you migrate a slice of a path at a time, which is invaluable when a single path is partly migrated. Start coarse (whole path), get finer only when a path genuinely needs to be split mid-migration, and keep the routing rules in version control and code-reviewed — a routing config is now production-critical infrastructure, and a fat-fingered weight that sends 100% of payments to a half-built service is an outage. Treat the facade config with the same rigor as application code.

#### Worked example: the facade overhead budget

A fair objection to the facade is "you have added a network hop to every request — what does that cost?" Put numbers on it for ShopFast. Before the migration, the client hit the monolith directly: median end-to-end latency ~85 ms. Inserting an Envoy facade in front adds one local proxy hop. Measured: the facade adds **p50 ~0.4 ms, p99 ~1.8 ms** of overhead — it is an in-datacenter L7 proxy doing a header parse and an upstream pick, not a database call. Against an 85 ms request that is sub-2% at the tail and invisible at the median. For the *extracted* services there is a *real* added hop (client → facade → service is genuinely one more network segment than the in-process call used to be), and that is the honest cost of services we quantify across the series — but the *facade itself* is cheap. The reason to measure and publish this number early is political as much as technical: someone *will* claim the facade is "slowing everything down," and "p99 +1.8 ms on an 85 ms request" ends the argument with data instead of opinion. Measure the overhead, set a budget (say, facade overhead must stay under 1% of p50), and alert if it regresses.

## 4. Choosing the first extraction

The first extraction is the most important one and the one teams most often get wrong, because the instinct is to attack the most painful part of the monolith first — usually the tangled core, the orders-and-inventory-and-pricing knot that everything depends on. That is exactly backwards. The first extraction is where you *learn the playbook* — how your facade behaves, how you handle the data, how you run a parallel system, how you do a cutover — and you want to learn it on something **valuable but not the riskiest core**. You want a clean teacher, not the final boss.

The criteria for a good first extraction:

- **A clear seam with low coupling.** A bounded context (the domain-driven-design term for a chunk of the model with a clear responsibility — we cover finding them in [Service Boundaries with Domain-Driven Design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design)) that touches the rest of the system through a narrow, well-understood interface. The fewer foreign keys reaching into its data and the fewer callers reaching into its code, the cheaper it is to peel off.
- **Real pain that extraction relieves.** High change-rate (a team is constantly editing it and deploy contention hurts) or a divergent scaling profile (it needs ten times the resources of everything else, or it has spiky load you would like to scale independently). If extracting it does not relieve a real, present pain, you are extracting for fashion.
- **Low blast radius if it goes wrong.** Async, idempotent, retryable work is ideal because a hiccup degrades gracefully rather than failing a user-facing transaction. A notification that arrives a few seconds late is fine; a payment that double-charges is not.

![A decision tree for choosing the first extraction that forks on whether the context has a clear seam and low coupling, whether it is tangled into core data, and whether it has high change-rate or scaling pain, pointing to notifications first and payments later](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-3.webp)

The decision tree in the figure walks the logic. Does the context have a clear seam and low coupling? If it is tangled deep into core data, defer it — untangle it *in place* first (branch by abstraction, schema separation) before you try to extract it over a network. If it has a clear seam *and* a real pain (high change-rate or scaling), it is a candidate. Among the candidates, start with the one that is async and low-risk. For ShopFast, that is **notifications**: it consumes events ("order placed," "payment succeeded") and sends emails, SMS, and push. It is naturally async, naturally idempotent (sending a duplicate email is recoverable), it has its own clear data (templates, delivery logs, user preferences) with few foreign keys reaching in, and it has been a deploy-contention headache because the marketing team is always editing templates and getting blocked behind the core team's deploys. It is valuable, it relieves real pain, and if it breaks, nobody loses money. That is the perfect first extraction.

**Payments comes second**, deliberately. It is the highest-value extraction — isolating the PCI-scoped, money-touching code is the whole reason the migration has executive support — but it is also the riskiest, and its data is the hardest to split. You do it *second*, after notifications has taught you the facade, the parallel-run discipline, and the cutover mechanics on something forgiving. Attacking payments first, before you have run the playbook once, is how migrations blow up. (The instinct to extract the modular monolith's cleanest seams first is exactly the discipline we set up in [Monolith First and the Modular Monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) — a well-modularized monolith makes the first extraction hours of work instead of a quarter.)

#### Worked example: big-bang timeline versus incremental slices

Put numbers on the two paths for ShopFast. The big-bang plan estimated 14 months to rebuild the whole monolith as services, with a feature freeze (or, realistically, a moving target the rewrite never catches). Value delivered to users during those 14 months: **zero**, until a single high-risk cutover. Probability of the cutover slipping at least once, based on the genre: high; their actual plan slipped from 14 to 22 months.

The strangler plan: stand up the facade in **2 weeks**. Extract notifications in **weeks 3–6** — shipped, in production, relieving deploy contention, in week 6. Extract payments in **months 2–4** (the data split is the slow part). Catalog in **months 5–6**. Orders, the hard core, in **months 7–9**. By month 9 there are four services live and the monolith is a thin shell. Value delivered: a real, production slice every few weeks starting in week 6, with the business shipping features the whole time. Even if the strangler takes *longer* in total wall-clock than the big-bang's optimistic estimate (it usually does not, but suppose it did), it delivers value 8 months sooner and never once bets the company on a cutover weekend. The expected value is not close. This is the entire case for incrementalism in one paragraph.

## 5. The hard part is the data, not the code

Here is the thing that separates engineers who have actually done a migration from those who have only read about it: **moving the code is easy; moving the data is the entire problem.** You can stand up a new payments service in a sprint. Splitting the payments data out of a shared database that the monolith and a dozen reports and three batch jobs all read and write — *that* is the part that takes months, and it is the part that goes wrong.

The reason microservices insist on a database per service (the rule we treat as foundational in [Database Per Service: The Rule That Defines Microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices)) is that a shared database is the thing that secretly couples "independent" services back together. If the new payments service and the monolith both read and write the same `payments` and `orders` tables, they are not independent — they share a schema, so neither can change it without coordinating; they share locks, so one can block the other; and you have built a [distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith) — the worst of both worlds, with the network costs of services and the coupling of a monolith. So the *goal* of the data split is: the new service owns its data in its own store, and nobody else touches that store directly.

But you cannot get there in one step, because on day one the monolith *is* still writing those tables and you cannot just yank them out from under it. So the data split is itself a staged migration, and there are a few strategies, from most-pragmatic-interim to most-correct-end-state:

**1. Share the database at first (the pragmatic interim anti-pattern).** The new service reads and writes the monolith's tables directly. This is an *anti-pattern* — it is the distributed monolith — but as a *temporary interim* during a transition it is sometimes the pragmatic choice: it lets you extract the *code* quickly and prove the service boundary at the application layer before you take on the data split. The rule is that it must be *temporary and acknowledged*, with a plan to split the data, not a place you stop. Many migrations that "failed" actually succeeded at extracting code and then stopped at the shared database, and now have all the operational cost of services with none of the independence.

**2. Split with views and synchronization.** As an intermediate step you can give the new service its own schema with database *views* onto the monolith's tables, or a one-directional sync, so it has a clean read model while writes still flow through the old path. This decouples the *read* shape before you tackle writes.

**3. CDC to keep the new store in sync during the transition.** This is the workhorse. You use change-data-capture — tailing the monolith database's write-ahead log — to stream every change to the relevant tables into the new service's private store in near-real-time. The new service gets its *own* database, kept current by CDC, while the monolith remains the writer during the transition. We go deep on this in section 6 and the mechanism itself is covered in [Change Data Capture and the Outbox Pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern).

**4. Dual writes (and why they are dangerous → use the outbox).** The tempting shortcut is to have the application write to *both* the old database and the new store in the same code path. This is dangerous because the two writes are not atomic: the process can crash between them, the network to the new store can fail, and now your two stores disagree with no record of which is right. There is no transaction spanning both. The correct fix is the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing): you write the change *and* an outbox row in one local transaction, then a relay reliably publishes the change to the new store, giving you at-least-once delivery with a single atomic write. Never hand-roll dual writes; route through the outbox or CDC.

![A before and after comparison contrasting an interim shared database where both systems write the same tables and recreate hidden coupling against a split where the new service owns a private store kept in sync by change-data-capture with an expand-contract cutover](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-7.webp)

The figure draws the destination. On the left is the interim shared database: both systems writing the same tables, hidden coupling, a schema neither side can evolve alone, and the distributed-monolith risk. On the right is where you are headed: a private store per service, kept in sync during the transition by a CDC stream with lag under a couple of seconds, and an expand-contract cutover (section 6) that flips the writer from the monolith to the new service without downtime. The left side is a stop you pass through; the right side is where you stop.

Here is the honest comparison of the data strategies, because the right choice depends on the slice and you will use different ones for different extractions:

| Data strategy | What it costs | When it is the right interim | Risk if you stop here |
| --- | --- | --- | --- |
| Shared database (read+write) | Hidden schema coupling; neither side can evolve alone | Fastest way to extract the *code* and prove the boundary | Distributed monolith — never a valid end state |
| Views / one-way sync | Extra read model to maintain | Decouple the *read* shape before tackling writes | New service is read-only; writes still coupled |
| CDC into a private store | A CDC pipeline to operate; eventual lag | The workhorse for splitting a live, busy table | Acceptable end state once the writer is flipped |
| Dual writes (hand-rolled) | Non-atomic; diverges on the first crash | Almost never — it looks easy and is a trap | Silent data divergence with no record of truth |
| Outbox + relay | An outbox table and a relay to run | When the new service must *emit* its changes reliably | Acceptable end state; this is the correct write path |

The pattern most migrations follow for a busy table is: shared DB to extract the code fast (acknowledged, dated), then CDC to give the new service its own read-current store, then expand-contract to flip the writer, then outbox for the new service's own outbound events. Notice "dual writes" is the only row with no valid use — it is on the table specifically so you recognize and reject it when a well-meaning engineer proposes "just write to both." The reason it fails is worth internalizing: a write to store A and a write to store B are two separate operations with no shared transaction, so any crash, timeout, or deploy between them leaves the two stores disagreeing, and — critically — *with no record of which write succeeded*, so you cannot even reconcile. CDC and the outbox both solve this by making the propagation derive from a *single* atomic local write (the WAL entry, or the outbox row in the same transaction as the business change), so there is always exactly one source of truth and the second store is a *consequence* of it, not a racing peer.

## 6. The Payments data split, concretely: CDC and expand-contract

Let me make the data split real, because the abstractions above only become useful when you see the mechanism. We are extracting payments. The monolith owns the `pay_charges` and `pay_refunds` tables. The new payments service needs its own store, and during the transition the monolith is still the writer (it still serves `/pay` for the 90% of traffic the facade has not yet flipped). We need the new service's store to stay current so that when we flip a request to the new service, it sees the same data the monolith would have.

**Step 1: stand up CDC.** We point Debezium at the monolith's Postgres write-ahead log, filtered to the `pay_` tables, and stream the changes into the payments service's own database. CDC reads the replication log, so it adds essentially no load to the monolith's query path and captures *every* change, including ones made by batch jobs and admin tools, which is exactly why it beats dual writes. Here is the connector config (secrets are placeholders):

```json
{
  "name": "payments-cdc",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "monolith-db.internal",
    "database.user": "REDACTED_CDC_USER",
    "database.password": "REDACTED_CDC_PASSWORD",
    "database.dbname": "shopfast",
    "table.include.list": "public.pay_charges,public.pay_refunds",
    "plugin.name": "pgoutput",
    "slot.name": "payments_cdc_slot",
    "publication.autocreate.mode": "filtered",
    "heartbeat.interval.ms": "10000",
    "tombstones.on.delete": "true",
    "snapshot.mode": "initial"
  }
}
```

The `snapshot.mode: initial` does the one-time backfill of all existing rows, then CDC tails the log for ongoing changes. A consumer in the payments service applies each change to its own tables. Now the new store is a live, lagging-by-a-second replica of the monolith's payments data — read-only, and always current.

![A branching dataflow showing change-data-capture tailing the monolith database into the new payments service store while a dual-read comparator reads from both and a drift alert blocks the cutover on any mismatch](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-8.webp)

The figure shows the whole transition topology for payments. The monolith still writes its tables; CDC (Debezium) tails the log and feeds the new payments store, keeping it under two seconds behind; and a **dual-read comparator** reads the same record from both stores and diffs them, alerting if they ever disagree by more than a tiny threshold. The drift alert *blocks the cutover* — you do not flip writes to the new service until the comparator has been green for long enough to trust the sync. This is the verification gate that makes the cutover safe.

**Step 2: expand-contract the cutover.** Now the genuinely delicate part: flipping the *writer* from the monolith to the new payments service, without downtime and without losing a write. You do not do this with a flag flip and a prayer. You use the **expand-contract** (also called parallel-change) migration, which has three phases:

- **Expand:** Prepare both sides to coexist. The new service can already *read* its store (CDC-fed) and you teach it to *write* its store. You make the schema changes additive — new columns/tables are added, nothing is removed or renamed — so both old and new code work simultaneously.
- **Migrate / cut over:** Flip the writer. For a window, you accept that the source of truth is moving. The cleanest approach for payments (where correctness matters more than a few seconds of latency) is a brief write-pause-and-drain on the affected keys: stop the monolith from accepting *new* payment writes for the cutover window, let CDC drain the last in-flight changes (you watch the lag hit zero), then enable writes on the new service and flip the facade route to 100%.
- **Contract:** Once the new service is the sole writer and has been verified, remove the old write path, stop the CDC connector, and delete the dead columns/tables from the monolith. This is the cleanup that, in the contract phase, you can finally do safely because nothing reads the old path anymore.

Here is the additive (expand) schema migration on the new service's store — note it only *adds*, so the cutover is reversible right up until the contract phase:

```sql
-- expand phase: additive only, both old and new code tolerate this
ALTER TABLE charges ADD COLUMN idempotency_key TEXT;
ALTER TABLE charges ADD COLUMN migrated_from_monolith BOOLEAN DEFAULT TRUE;
CREATE UNIQUE INDEX CONCURRENTLY uq_charges_idem
  ON charges (idempotency_key) WHERE idempotency_key IS NOT NULL;
-- NOTHING is dropped or renamed here. The contract phase (weeks later)
-- drops migrated_from_monolith once 100% of writes are native.
```

#### Worked example: the Payments data split, with numbers

Concrete numbers from a real-shaped migration. The `pay_charges` table has 42 million rows. The initial CDC snapshot backfills them in about 90 minutes (the new store is on faster disks and the backfill is a bulk copy). Once snapshot completes, steady-state **CDC lag is 400–900 ms** at the monolith's normal write rate of ~120 payment writes/second, spiking to ~2.5 s during the nightly batch reconciliation — well within tolerance for a read-replica that nobody is cutting over to yet.

The **dual-read verification** runs for two weeks before any cutover: for every payment read served by the monolith, an async job reads the same charge id from the new store and diffs the two. Over 14 days and ~140 million comparisons, the mismatch rate settles at **0.003%**, all traced to a known timestamp-precision difference (microseconds vs milliseconds) which we normalize in the comparator. Zero *semantic* mismatches. That is the green light.

The **cutover window** itself: we pick a low-traffic window (03:00–03:08 local), pause new monolith payment writes (existing in-flight ones drain), watch CDC lag fall to **0 ms** (takes ~40 seconds), enable native writes on the new service, and flip the facade route from 10% to 100% in one config push. Total user-visible write-unavailability for `/pay`: **~90 seconds**, and even that is masked because the client retries (payments are idempotent on the idempotency key). The reads were already served from the new store at 100% a week earlier. The whole risky operation is 90 seconds, instantly reversible until the moment we drop the old columns weeks later in the contract phase. That is what "deliberate" looks like.

## 7. Branch by abstraction for the internal seams

Not everything has a network edge you can route at. The facade works for *requests that cross the process boundary* — HTTP calls, queue messages. But a huge amount of a monolith's coupling is *internal*: one module calls another module's function directly, in-process. You cannot put a proxy between two function calls. For those internal seams you use **branch by abstraction**, a refactoring discipline that lets you swap an implementation gradually, in production, without a long-lived feature branch.

![A layered stack showing the controlled gradual swap behind an internal interface, from a call router at the top through a feature flag controlling the ramp percentage, a shadow comparator that verifies the new path, the new remote implementation as the target, and the old in-process path kept as a safe fallback](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-6.webp)

The four steps:

1. **Introduce an interface** over the thing you want to replace. The monolith's calling code now goes through this interface instead of calling the concrete implementation directly.
2. **Reimplement behind the interface.** Build the new implementation — which, for our purposes, calls out to the new service over the network — behind the same interface. Now there are two implementations: the old in-process one and the new remote one.
3. **Swap, gradually.** Use a flag to route some/all calls to the new implementation. Same ramp discipline as the facade: 1% → 100%, reversible.
4. **Remove the old implementation** once the new one carries everything.

Here is the interface for ShopFast's notification call inside the monolith. Before extraction, the order code called `EmailSender` directly. We introduce a `Notifier` interface, keep the old in-process implementation, and add a new one that calls the extracted notifications service:

```python
# 1. the interface the monolith now depends on
class Notifier:
    def notify(self, user_id: str, template: str, data: dict) -> None: ...

# 2a. the OLD implementation (in-process, the monolith's original code)
class InProcessNotifier(Notifier):
    def notify(self, user_id, template, data):
        EmailSender.send_now(user_id, template, data)  # legacy path

# 2b. the NEW implementation (calls the extracted service)
class RemoteNotifier(Notifier):
    def __init__(self, client): self.client = client
    def notify(self, user_id, template, data):
        # fire async; notifications are not on the critical path
        self.client.enqueue(user_id=user_id, template=template, data=data)

# 3. swap behind a flag — reversible, gradual
def get_notifier(flags) -> Notifier:
    if flags.percent_enabled("notifications.remote", user_id):
        return RemoteNotifier(notif_client)
    return InProcessNotifier()
```

The calling code — `get_notifier(flags).notify(...)` — does not change as you ramp. You move from 0% remote to 100% remote with a flag, watching error rates and latency, and you can fall back to the in-process path in seconds. Once you are at 100% remote and stable, you delete `InProcessNotifier` and the legacy `EmailSender`. This in-process technique is how you strangle the *inside* of the monolith, and combined with the edge facade it lets you migrate code that has no clean network boundary yet. It is the in-process cousin of the canary flag we use at the edge, and it follows the same reversible-ramp philosophy as the deployment strategies in [Deployment Strategies: Blue-Green, Canary, Feature Flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags).

## 8. Keeping both running and consistent during the (long) transition

A migration is not a moment; it is a *state you live in for months*. For that whole time you have two systems that must agree, and "must agree" is doing a lot of work. The discipline that makes this survivable is **parallel run with verification** — and the cheapest, most powerful version is the shadow / dual-read check.

A **shadow read** forks a copy of read traffic to the new service *without* using its response to serve the user. The user gets the monolith's answer (safe), and asynchronously you ask the new service the same question and compare. You are verifying the new service's *correctness* under real production traffic, with zero user risk, before you ever let it serve a real request. Here is a shadow-and-diff check on the gateway/comparator side:

```python
def handle_get_charge(charge_id, flags):
    # serve from the safe source of truth (monolith) for now
    primary = monolith.get_charge(charge_id)

    if flags.enabled("payments.shadow_verify"):
        # async, off the critical path: ask the new service the same thing
        spawn(verify_shadow, charge_id, primary)

    return primary  # user always gets the monolith's answer during shadow

def verify_shadow(charge_id, expected):
    candidate = payments_svc.get_charge(charge_id)
    if not charges_equal(expected, candidate):
        metrics.incr("payments.shadow.mismatch",
                     tags={"field": first_diff(expected, candidate)})
        log.warning("shadow mismatch on %s", charge_id)  # blocks cutover
```

The mismatch metric is your cutover gate (we used it in the worked example: 0.003% mismatch, all explained, green light). You do not flip writes until shadow has been clean. This single technique — verify with real traffic before you take real risk — is the difference between a migration that surprises you in production and one that does not.

It helps to think of each slice as moving through an explicit **state machine**, because "the migration" is really a set of slices each at a different stage, and a senior tracks them individually rather than as one fuzzy percentage. A slice is in exactly one of these states: *not started* (100% monolith), *shadow* (new service receives forked read traffic, serves nothing — verifying correctness), *canary* (new service serves a small live percentage), *ramping* (climbing toward 100% as the error budget holds), *primary* (100% live, monolith path still present as fallback), and *retired* (old code deleted, irreversible). The transitions between states are *gated*: shadow → canary requires the mismatch rate under threshold for N days; canary → ramping requires the canary's golden signals (latency, error rate, saturation) within the [SLO](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices); primary → retired requires two clean weeks at 100%. Writing these gates down turns "are we done?" from a vibe into a checklist, and it makes the *reversible/irreversible* boundary explicit: everything up to and including *primary* is reversible (flip the flag), and only *retired* is not. The discipline is to never let a slice reach *retired* until you would bet your on-call shift on it, because that is the one transition you cannot take back.

The rollback procedure deserves to be a runbook, not improvisation, because you will execute it under pressure. For a *read* path it is trivial: flip the route weight to 0% and 100% of reads serve from the monolith in seconds, because the monolith never stopped being able to serve them. For a *write* path mid-cutover it is more delicate, which is exactly why the expand phase keeps everything additive and keeps the monolith's write path intact until the contract phase: if the new service is the writer and you must roll back, you re-enable the monolith's write path (still present), point reads back at the monolith, and let CDC — which you have *not* torn down yet — reconcile any writes the new service took during its brief tenure. The fact that you kept the old path and the CDC pipeline alive *through* the cutover, deleting them only weeks later in the contract phase, is precisely what makes the write rollback possible. A migration that deletes the old write path at cutover has thrown away its own rollback, which is how a small regression becomes a data-loss incident.

**Measuring progress** is the other half of living in the transition. The honest, motivating metric is **percent of live traffic on the new services** (not "percent of code rewritten," which is unfalsifiable and always optimistic). You can read it straight off the facade: what fraction of requests, weighted by importance, route to extracted services? You track it per slice and in aggregate, and you put it on a dashboard the whole org can see, because a migration that cannot show progress loses funding.

![A timeline of the traffic ramp on a new service going from zero percent in shadow only, to a one percent canary, ten percent, fifty percent, one hundred percent live with the monolith path off, and finally the old code deleted](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-9.webp)

The figure shows the ramp for one service as the percent of live traffic over six weeks. Week 0 is 0% — shadow only, verifying correctness with no user risk. Week 1 is a 1% canary where you watch p99 latency and error rate like a hawk. Week 2 you are at 10%, confirming the error budget holds. Week 3, 50% — half the fleet. Week 4, 100% live and the monolith path is switched off (but the code is still *there*, one flag-flip from coming back). Week 6, after two clean weeks at 100%, you finally delete the old code. The slow ramp is not timidity; it is how you keep blast radius small. A bug at 1% affects 1% of users for the minutes it takes to flip back; the same bug shipped at 100% in a big-bang affects everyone at once with no flag to flip.

#### Worked example: catching a regression at 1% instead of 100%

During the payments ramp, at the 1% canary stage (week 1), the new service's p99 latency on `/pay/charge` was **310 ms** versus the monolith's **140 ms** — a 2.2× regression. At 1% traffic this showed up as a small bump on the dashboard affecting a few hundred requests, and the error budget was untouched because the absolute volume was tiny. We flipped the route back to 0% in **under 20 seconds**, investigated, found the new service was making a synchronous call to the fraud-check API that the monolith had been doing in a connection-pooled, warmed path. We added the same pooling, re-ran the canary at 1%, saw **155 ms** p99, and resumed the ramp. Total user impact: a few hundred slightly-slow (but successful) requests over a few minutes. Had this been a big-bang cutover, the same bug would have hit 100% of payments at once, blown the latency SLO across the board, and there would have been no 20-second flip-back — the rollback would have been "restore the old system and migrate the data back," i.e., an incident. The ramp converted a potential outage into a dashboard blip. That is the entire value proposition of incremental cutover in one number: 1% blast radius versus 100%.

## 9. The migration sequence and the decision matrix

Let me zoom out to the whole sequence and then make the approach choice explicit, because the matrix is the senior's tool for not getting talked into a rewrite.

![A timeline of the strangling sequence from putting a facade in front of the monolith, to extracting notifications, payments with CDC, catalog with branch-by-abstraction, orders, and finally retiring or freezing the monolith core](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-5.webp)

The figure lays out ShopFast's full sequence over roughly a year. Month 0: facade in front of the monolith, routing everything to it. Month 1: notifications extracted (the low-risk teacher). Month 3: payments extracted via CDC and the ramp (the high-value, hard-data slice). Month 6: catalog, using branch-by-abstraction for its internal callers. Month 9: orders split, three or four services now live. Month 12: the monolith is either fully retired or — and this is the part section 11 is about — strangled down to a small, well-factored core you have *decided* to keep. Notice that value ships continuously across the whole timeline, and notice that the order is deliberate: forgiving first, hard-data second, internal-seam third, core last.

Now the explicit decision. There are four real approaches to "this monolith needs to change," and a senior chooses among them on evidence, not fashion.

The matrix scores the four approaches across the dimensions that actually decide the outcome. **Big-bang rewrite**: very high risk, 9–18 months to any value, months of feature freeze, one huge data cutover, near-zero reversibility — it loses on every axis and is the default trap. **Strangler fig**: low staged risk, value in 2–4 weeks, no freeze, per-slice CDC data complexity (the real cost), reversible per-route — it wins almost everywhere, paying mainly in data-split work. **Branch by abstraction**: even lower risk for *internal* seams, value in days, no data move at all (it is an in-process swap), reversible by swapping the implementation back — but it only addresses internal coupling, not the network boundary, so it is a *complement* to the strangler, not a substitute. **Stay a modular monolith**: zero migration risk, value already shipped, no freeze, shared schema, n/a reversibility — and for many teams this is the *right* answer, which the matrix makes uncomfortable but honest.

| Dimension | Big-bang rewrite | Strangler fig | Branch by abstraction | Stay modular monolith |
| --- | --- | --- | --- | --- |
| Risk profile | Very high (all at once) | Low, staged per slice | Low, in-process swap | None |
| Time to first value | 9–18 months | 2–4 weeks | Days | Already shipped |
| Feature freeze | Months | None | None | None |
| Data-split complexity | One huge cutover | Per-slice CDC | No data move | Shared schema (the cost) |
| Reversibility | Near zero | Per-route flag flip | Swap impl back | n/a |
| Best when | System is small | Large valuable system, real pain | Internal seam, no network boundary | Pain is org/modularity, not scale |

The matrix's verdict: for a large, valuable, actively-used monolith with real present pain, the strangler fig (using branch-by-abstraction for the internal seams) is the answer in almost every case. The big-bang is right only for small systems. And staying a modular monolith is right more often than ambitious engineers want to admit — which the next-to-last section makes the case for.

![A decision matrix scoring big-bang rewrite, strangler fig, branch by abstraction, and staying a modular monolith across risk profile, time to first value, feature freeze, data-split complexity, and reversibility](/imgs/blogs/strangler-fig-migrating-a-monolith-to-microservices-4.webp)

## 10. Avoiding the distributed monolith

The most expensive way to "succeed" at this migration is to chop the monolith's code into services and *keep the coupling* — to end up with services that cannot deploy independently because they share a database, share synchronous chains where one being down takes them all down, and share a release train because their contracts are too tightly bound. That is the **distributed monolith**, and it is strictly worse than the monolith you started with: you have added network latency, partial failure, and operational overhead, and gained none of the independence that was supposed to justify it. We catalogue its failure modes in [Shared Data Anti-Patterns and the Distributed Monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith); here is how the strangler specifically avoids it.

**The data split is not optional.** The single biggest cause of the distributed monolith is stopping at "shared database during transition" and never finishing the split. The shared DB is fine as an *acknowledged temporary interim* with a dated plan to split; it is fatal as a *destination*. If your migration's definition of done does not include "each service owns its data," you are building a distributed monolith on purpose. The CDC + expand-contract work in section 6 is the *expensive* part of the migration precisely because it is the part that buys you real independence; skipping it is skipping the point.

**Watch the synchronous call depth.** If extracting payments means orders now makes a synchronous network call to payments which makes a synchronous call to notifications which calls back to orders, you have built a fragile chain where any one being slow stalls all of them — and you have possibly built a cycle, which is even worse. The fix is to prefer async events at the seams (choreography, covered in [Event-Driven Microservices](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — not in the link list but the principle holds) and to keep synchronous chains shallow. A service boundary that turns one in-process call into a four-hop synchronous network chain has multiplied your failure surface, not reduced it.

**Independent deployability is the test.** The honest acceptance test for "is this really a service" is: *can you deploy it, by itself, without coordinating a release with any other service or the monolith?* If the answer is no — if a payments change requires a synchronized orders deploy because they share a schema or a tightly-coupled contract — you have not finished extracting it. Independent deployability is the *definition* of the thing you are migrating toward, and it is the thing the distributed monolith fails. We pick this thread up in the sibling [Microservices Anti-Patterns and When to Go Back to Monolith](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith).

#### Stress test: three ways the migration goes wrong

**"The new service has a bug mid-migration — can you route back?"** Yes, instantly, *if* you built the facade and the flag right, and this is the whole reason you did. The new service is at, say, 10%; you flip the route to 0% and 100% of traffic is back on the monolith in seconds because the monolith's code path was never deleted. This is why you delete old code only after weeks at 100%, not at cutover. The migration that *cannot* route back is one that did dual-writes without an outbox (so the data already diverged) or deleted the old path too early — both avoidable. The reversibility is a property you engineer, not a hope.

**"The data got out of sync during transition."** This is the genuinely scary one, and it is why the dual-read comparator and the CDC-lag monitor exist. If shadow/dual-read shows the new store disagreeing with the monolith, you do *not* cut over — the drift alert blocks the cutover by design (figure 8). If you somehow cut over and *then* discover drift, you have a real incident: you flip reads back to the monolith (your source of truth, which you kept writing during the ramp precisely so you could fall back), reconcile by re-running the CDC backfill, and re-verify before trying again. The discipline that prevents this from being catastrophic is *keeping the monolith as the source of truth until shadow has been clean for weeks*, and never deleting the old write path until the contract phase. If you violated those, drift becomes data loss; if you respected them, drift is a delayed cutover, which is annoying but safe.

**"The migration stalls at 60% forever."** This is the most common *real* failure, and it is not a technical one — it is organizational. The high-value, easy slices get extracted, the team declares partial victory, leadership reallocates the migration team to feature work, and the monolith sits at 60% migrated forever, carrying *both* the operational cost of services *and* the operational cost of the monolith. Sometimes stalling at 60% is actually *fine* — if the remaining 40% is a stable, low-change core with no extraction signal, leaving it as a small monolith is the correct end state (section 11). The failure is stalling *by accident* — never deciding whether the remaining core should be extracted or kept, and paying the in-between cost indefinitely. The senior move is to make the stop *deliberate*: either fund the migration to a chosen end state, or formally declare the remaining core a kept monolith and stop pretending it is a migration in progress.

## 11. Knowing when to stop

This is the section that separates the senior from the enthusiast, and it is the one most migration write-ups never mention because it is unglamorous: **the right end state is very often a small, well-factored monolith plus a few extracted high-value services — not zero monolith.**

The strangler-fig metaphor implies the host tree dies completely, and that biases people toward "we are not done until the monolith is gone." But the *engineering* goal was never "no monolith"; it was "relieve the specific pains that justified the migration." Once you have extracted the slices that had a real signal — the high-change-rate context that suffered deploy contention, the divergent-scaling context that wasted money co-located, the PCI-scoped payments code that needed isolation — the *remaining* monolith is, by definition, the part that *had no extraction signal*. It is the stable, low-change, normally-scaling core. Extracting *it* would cost you all the operational tax of more services (more deploys, more dashboards, more network hops, more partial-failure handling) to relieve a pain that does not exist. That is negative-value work.

The discipline here is the same cost-of-change reasoning from [Monolith First and the Modular Monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith): you extract against a *measurable signal*, and when the next extraction has no signal — no deploy contention, no scaling dollar figure, no blast-radius requirement — you *stop*. The remaining monolith, now small and well-factored (you cleaned it up as you strangled the painful parts off), is a perfectly good permanent component. ShopFast's likely correct end state is not "twelve microservices and no monolith"; it is "a notifications service, a payments service, maybe a catalog service, and a tidy orders-and-checkout core that ships fine as a single unit because nothing about it hurts."

The trap on the other side is the migration that runs out of *value* but keeps going on *momentum* — extracting the last stable core into three services because the roadmap said "decompose the monolith" and nobody updated the goal when the pains were already relieved. That is how you end up with a sprawling fleet whose last few services exist only because someone finished a checklist. Stop when the value runs out, not when the monolith is gone. A senior writes the *stopping condition* into the migration plan at the start: "we will extract contexts X, Y, Z against these signals, and we will reassess whether the remaining core warrants extraction — defaulting to keeping it." Making the stop a *decision* rather than an *omission* is the whole point of section 10's third stress test.

## Case studies

**Segment's monolith U-turn (the over-decomposition lesson).** Segment publicly documented going *from* a microservices architecture *back* toward a monolith in their 2018 post "Goodbye Microservices." They had decomposed into 140+ services and found the operational overhead — shared libraries that had to be updated across every service, deploy and dependency management across the fleet, the cognitive load — outweighed the benefits for their workload. The lesson for a migration *toward* services is the mirror image: decomposition has a real, ongoing cost, and the right number of services is the number your pains justify, not the maximum you can chop into. It is direct evidence for the "know when to stop" discipline: more services is not more better.

**Shopify's modular monolith (the wise non-migration).** Shopify runs one of the largest Rails monoliths in the world and has deliberately *not* big-bang-rewritten it into microservices. Instead they invested in modularity *in place* — their Packwerk tool enforces component boundaries inside the monolith — extracting only specific high-value pieces. This is the "stop partway / mostly-don't" outcome at enormous scale: a huge, successful business decided that a well-factored monolith plus selective extraction beat a full decomposition. The lesson: the end state of a good migration can absolutely be "a great monolith with a few services around it," even for a company most engineers would assume *must* be all-microservices.

**The Netscape rewrite (the big-bang cautionary tale).** Netscape's decision to rewrite their browser engine from scratch for Netscape 6 is the canonical big-bang failure, immortalized in Joel Spolsky's 2000 essay. They spent roughly three years rebuilding, shipped nothing competitive in the interim, watched Internet Explorer take the market, and never recovered their position. Order-of-magnitude lesson: a *technically defensible* rewrite of a large, valuable, actively-used system can be *strategically fatal* purely because of the time-with-no-value and the moving-target dynamics — exactly the failure modes the strangler fig is engineered to avoid. The genre repeats constantly; Netscape is just the most public grave marker.

**The general strangler success genre (incremental migrations that worked).** Many large engineering organizations — payments companies, marketplaces, and SaaS platforms — have documented incremental, facade-fronted migrations on their engineering blogs over the past decade. The shared shape is always the same: a routing/edge layer, one bounded context extracted at a time, CDC or events to handle the data, a slow traffic ramp with verification, and a deliberate decision about the remaining core. The accurate, order-of-magnitude takeaway is that the *successful* migrations look boring — long, incremental, value-shipping, reversible — and the *failed* ones look dramatic (a tiger team, a cutover weekend, a rewrite). When a migration story is exciting, it is usually a story about a fire.

## When to reach for this (and when not to)

**Reach for the strangler fig when:** you have a *large, valuable, actively-used* monolith with a *specific present pain* (deploy contention, a divergent-scaling context, a part that needs isolation like payments), and the business cannot afford a feature freeze (it never can). This is the default for any real monolith decomposition. Pair it with the in-process branch technique for the internal seams that have no network boundary.

**Reach for the in-process branch technique (on its own) when:** the coupling you want to break is *internal* — module-to-module in-process calls — and you are not (yet) moving it across a network. It is the lowest-risk tool in the box and often all you need to clean up a monolith *in place* without extracting anything.

**Do a big-bang rewrite only when:** the system is genuinely small (the whole thing is weeks, not years, so incrementalism's overhead is not worth it), or it is small *and* on a dead platform you must abandon. For anything large, the big-bang is the trap, full stop.

**Stay a modular monolith — possibly forever — when:** your pain is *organizational or modularity-related*, not a scaling or isolation signal. If the monolith hurts because it is a big ball of mud, fix the boundaries *in place* (it is far cheaper than extraction and gets most of the benefit). If it hurts because too many teams contend for one deploy, that is a [Conway's-law](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices) and ownership problem that may have a cheaper fix than a distributed system. Microservices are a continuous operational tax; do not pay it to relieve a pain a refactor would fix.

The senior framing: migrate *incrementally* behind a facade, solve the *data* problem deliberately (CDC, expand-contract, never naked dual-writes), keep *shipping features* the whole time, and *stop when the value runs out* — which is often well before the monolith is gone.

## Key takeaways

- **The big-bang rewrite almost always fails on large systems.** You freeze features while the business does not, the new system chases a moving target and never catches up, you carry both, and all the risk lands on one cutover. (Junior: rewrites feel clean. Senior: the mess in the old code is encoded correctness you will rediscover the hard way.)
- **Strangle, don't rewrite.** Put a facade at the edge, route one slice at a time to a new service, ramp traffic slowly and reversibly, repeat. Every step ships value and every step can be rolled back.
- **The facade is the highest-leverage thing you build.** Dynamic, percentage-based, flag-driven routing turns the migration's unit of risk from "the whole system, irreversible" into "1% of one path, reversible in seconds." Over-invest in it early.
- **Choose the first extraction for learning, not for maximum pain relief.** Low coupling, clear seam, async and low-risk (notifications), valuable but not the riskiest core. Do payments *second*, after the playbook is proven.
- **The hard part is the data, not the code.** Standing up a service is a sprint; splitting its data out of a shared database is months. Use CDC to keep a private store in sync during the transition and expand-contract to flip the writer without downtime.
- **Never hand-roll dual writes.** Two non-atomic writes diverge on the first crash. Use CDC or the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) so the change and its propagation are atomic.
- **Use the in-process branch technique for internal seams.** No network edge means no facade; introduce an interface, build the new (remote) implementation behind it, swap behind a flag, delete the old one. It is the in-process cousin of the canary.
- **Verify with real traffic before you take real risk.** Shadow reads and a dual-read comparator prove correctness at zero user risk; the mismatch metric is your cutover gate. Measure progress as percent of live traffic, not percent of code.
- **Avoid the distributed monolith.** Finish the data split, keep synchronous chains shallow, and accept nothing as "a service" until it deploys independently. A shared database at the end is failure wearing a success costume.
- **Know when to stop.** A small, well-factored monolith plus a few high-value extracted services is very often the *right* permanent end state. Extract against a measurable signal; when the next extraction has no signal, stop on purpose.

## Further reading

- Martin Fowler, ["StranglerFigApplication"](https://martinfowler.com/bliki/StranglerFigApplication.html) — the original pattern, the metaphor, and why incremental envelopment beats replacement.
- Sam Newman, *Monolith to Microservices* (O'Reilly) — the definitive book-length playbook for incremental decomposition, with deep treatment of the strangler, the in-process branch technique, and every flavor of the data split.
- Joel Spolsky, ["Things You Should Never Do, Part I"](https://www.joelonsoftware.com/2000/04/06/things-you-should-never-do-part-i/) — the Netscape rewrite essay; the canonical argument for why throwing away working code is a strategic mistake.
- Segment Engineering, "Goodbye Microservices: From 100s of Problem Children to 1 Superstar" — the over-decomposition reversal and the clearest real-world evidence for knowing when to stop.
- This series and its mechanism deep-dives: [Monolith First and the Modular Monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith), [Service Boundaries with Domain-Driven Design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design), [Database Per Service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), [Shared Data Anti-Patterns and the Distributed Monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), [The API Gateway and Backend for Frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), [The Transactional Outbox and Reliable Event Publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing), [Change Data Capture and the Outbox Pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and the forward-looking [Microservices Anti-Patterns and When to Go Back to Monolith](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith) and [Conway's Law and Team Topologies for Microservices](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices).

---
title: "What Are Microservices, and When Not to Use Them"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A precise, junior-friendly definition of what a microservice actually is, the spectrum from monolith to nanoservices, the real costs you buy and benefits you get, and a decisive guide to when the boring monolith wins on every axis."
tags:
  [
    "microservices",
    "monolith",
    "software-architecture",
    "distributed-systems",
    "backend",
    "system-design",
    "modular-monolith",
    "scalability",
    "domain-driven-design",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/what-are-microservices-and-when-not-to-use-them-1.webp"
---

A few years ago I watched a three-person startup spend its entire first quarter not building features but building infrastructure. They had read that "real" companies use microservices, so before they had a single paying customer they had split their product into eleven services, each with its own repository, its own database, its own deploy pipeline, and its own little README that nobody kept up to date. Adding a field to the signup form — a one-line change in a sane system — meant editing three services, coordinating two deploys, and discovering on the third try that a fourth service had been silently caching the old schema. They were not moving fast. They were moving like a company ten times their size, paying all the costs of scale with none of the scale. When they finally tore it all down and put everything back into one well-organized application, they shipped more in two weeks than they had in the previous two months.

This is the first post in a forty-part series on building microservices, and I am going to open it the most honest way I know how: by telling you that microservices are a *trade*, not an *upgrade*. They are not the next rung on a ladder where monoliths are the beginner level and microservices are the expert level. They are a different shape of system that you adopt to solve specific problems — mostly *organizational* problems, problems of too many people trying to change the same code at once — and they make almost everything else harder in exchange. The whole point of this series is to teach you to build, connect, deploy, secure, observe, and operate a fleet of services *well*. But you cannot operate something well until you understand what you are getting into, and the most senior thing I can teach you in this very first post is the judgment to know when *not* to reach for them at all.

The figure below shows the heart of the matter on a single feature — checking out a cart — in our running example system, an e-commerce platform I will call **ShopFast** that we will use across the whole series. On the left, the feature lives inside one application, and the order code calls the inventory code with an ordinary function call: microseconds, in the same process, wrapped in a database transaction that either fully commits or fully rolls back. On the right, the same feature is spread across services that own their own data and talk over the network, and that one function call has become an HTTP or gRPC request that can be slow, can time out, can half-succeed, and can never be wrapped in a single transaction again. Everything good and everything painful about microservices is visible in the difference between those two pictures.

![A side-by-side comparison showing the ShopFast checkout feature as a single in-process function call in a monolith versus the same feature spread across independently deployed services that communicate over the network](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-1.webp)

By the end of this post you will be able to do four concrete things. You will be able to define a microservice precisely enough to tell a real one from a thing people merely *call* a microservice. You will be able to place any system on the spectrum from monolith to nanoservices and say where it should sit. You will be able to list, from memory, the real costs you buy and the real benefits you get — and recite which is which under pressure in a design review. And, most importantly, you will be able to look at a team and a product and say, with reasons, "you should not do this yet." Let us build that judgment from the ground up, starting with the definition, because almost every microservices disaster I have seen began with someone using the word to mean something it does not mean.

## 1. What a microservice actually is

Let me give you the definition first and then defend every word of it, because the definition is doing real work and most people only remember half of it.

A **microservice** is an independently deployable unit of software that owns a single business capability and the data behind it, and that communicates with other services only over the network through well-defined interfaces. That is the whole thing. There are four load-bearing phrases in that sentence, and a system is only genuinely a microservices architecture if it honors all four. Drop one and you have something else — sometimes something worse than a monolith wearing a microservices costume.

**Independently deployable.** This is the one that matters most and the one people forget first. A microservice can be built, tested, and pushed to production *on its own schedule*, without coordinating a release with any other service. If deploying your "order service" requires you to simultaneously deploy your "payment service" because they share a library that changed, or a database table whose schema both depend on, then they are not two services — they are one service in two repositories, which is strictly worse than one service in one repository. Independent deployability is the property that delivers nearly all of the organizational benefit, and it is the property that gets quietly violated the moment you let two services share a database or a deploy step. Hold onto this; we will keep coming back to it.

**Owns a single business capability.** A service should map to a *thing the business does* — taking orders, charging cards, tracking inventory, shipping parcels — not to a technical layer like "the database access service" or "the validation service." This is the difference between slicing your system vertically (by what it does for the customer) and horizontally (by technical role). Vertical slices can change independently because a business capability tends to change as a unit; horizontal slices cannot, because almost every feature cuts across all the layers, so a feature change touches every horizontal service at once. We will spend an entire post later in the series on finding these boundaries with [domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design), because getting the boundaries right is the single hardest and most consequential thing in this whole discipline.

**Owns the data behind it.** A real microservice has its own datastore that no other service touches directly. The order service has the orders database; if the shipping service wants to know about an order, it asks the order service, it does not reach into the orders database and run a query. This rule — database-per-service — is so central that it gets its own dedicated post later, titled, not subtly, [the rule that defines microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices). The instant two services share a database table, they are coupled at the deepest possible level: neither can change that table's schema without breaking the other, which means neither can deploy independently, which means, per the first phrase, they are not really separate services.

**Communicates only over the network through well-defined interfaces.** Services do not share memory, do not share code that carries business logic, do not call each other's internal functions. They send each other messages — an HTTP request, a gRPC call, an event on a queue — across a network boundary, through an interface that is a published contract. This is the phrase that imports all the pain, because the network is slow, unreliable, and capable of failing in ways an in-process call simply cannot, a fact so important that the entire fifth post in this series is about the [fundamentals and fallacies of inter-service communication](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies).

Now contrast this with what people *think* a microservice is. The most common wrong definition is "a small service" — as if the defining property were lines of code. It is not. A perfectly good microservice can be tens of thousands of lines if it owns a genuinely large business capability; a 200-line service that reaches into someone else's database is not a microservice, it is a distributed bug. The "micro" in microservices was always a misleading prefix. It does not mean *small in size*. It means *small in scope of responsibility* — one capability, cleanly owned. James Lewis, who helped popularize the term, has said he regrets the "micro" because it makes people optimize for the wrong thing. Optimize for *independence* and *clear ownership*, not for smallness.

The cleanest test I know for whether something is genuinely a microservice is the **deploy-independence test**: pick any one service, change its internal implementation in a backward-compatible way, and ask whether you can deploy *only* that service to production without touching, rebuilding, or coordinating with any other. If the answer is yes, you have a real service. If the answer is "well, we'd also have to bump the shared library and redeploy the two services that use it," then you have discovered that those services are secretly one unit, and you are paying network and operational costs for a coupling you never escaped. Run that test on a system claiming to be microservices and you will quickly learn how many real services it actually has — which is often far fewer than the number of repositories suggests.

There is one more piece of vocabulary worth installing now, because it names the most common way the definition gets violated: the **distributed monolith**. A distributed monolith is a system that *looks* like microservices — many services, many repositories, lots of network calls — but where the services cannot actually be deployed independently, because they share a database, share a release schedule, or are so chattily coupled that changing one always forces changing several others. It is the worst of both worlds: you pay the full network-latency tax, the full operational tax, and the full distributed-data tax of microservices, while collecting *none* of the independent-deploy benefit, because nothing can ship on its own. A plain monolith is a coherent, well-understood thing; a distributed monolith is a monolith that has been smeared across a network and made slower and harder to operate for no gain. Almost every microservices horror story — including the one in my opening — is really a distributed-monolith story. The four-property definition above is precisely the set of properties you must protect to avoid sliding into it, and we devote an entire later post to the [shared-data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith) that cause it.

## 2. The spectrum: monolith to nanoservices

The biggest conceptual mistake juniors make — and plenty of seniors too — is treating this as a binary choice: monolith *or* microservices, old-and-bad *or* new-and-good. It is not a switch. It is a dial, and there are several meaningful positions along it, illustrated in the figure below. Knowing the positions lets you choose deliberately instead of lurching from one extreme to the other.

![A spectrum pipeline showing four architecture positions from a single-deploy monolith through a modular monolith and microservices to over-split nanoservices](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-2.webp)

At the far left is the **monolith**: one application, one deploy unit, one database. Everything runs in a single process (or a set of identical replicas of that one process). Module A calls module B with a function call. There is exactly one thing to deploy, one thing to monitor, one place to look when something breaks. The monolith has an undeserved bad reputation. A *well-built* monolith is a fantastic way to run a system, and the overwhelming majority of successful software in the world is monolithic. The bad reputation comes from the **big ball of mud** — a monolith with no internal structure, where everything calls everything, where the order logic and the payment logic and the inventory logic are tangled together so thoroughly that no one can change one without breaking the others. But a tangled monolith is a *failure of discipline*, not a property of monoliths. The cure is structure, not distribution.

Which brings us to the second position, the one I want you to fall in love with: the **modular monolith**. This is still one deploy unit and (usually) one database, but the code inside is organized into well-separated modules with explicit, enforced boundaries between them. The order module and the inventory module live in the same process and can call each other with fast in-process calls — but they call each other only through defined interfaces, not by reaching into each other's internals, and ideally a build-time rule or an architecture-fitness test fails the build if someone violates a boundary. The modular monolith gives you most of the *design* benefit of microservices (clear ownership, separable concerns, the ability to reason about one capability at a time) with *none* of the distribution cost (no network, no separate deploys, no distributed transactions). It is, for the vast majority of teams, the correct answer, and it gets its own post: [monolith first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith). When in doubt, this is where you should be.

The third position is **microservices** proper: those well-separated modules are now separate deployable units, each in its own process, each owning its own data, talking over the network. You have crossed the line where the costs in this post become real and unavoidable. For a system of meaningful size — say five to thirty services owned by several teams — this is a coherent and powerful place to be. The benefit you bought is independence; the price you paid is everything else in section 4.

And at the far right is the failure mode, **nanoservices**: you kept splitting past the point of usefulness, until you have hundreds of tiny services, each so small that it does almost nothing, and the *coordination* between them dwarfs the work they do. A service that does nothing but validate a phone number, called over the network, with its own deploy pipeline and on-call rotation, is a nanoservice. You have all the costs of distribution multiplied by a huge number, and the business logic is now smeared so thin across the network that no one can find it. Monzo, which famously runs over 1,500 services, is sometimes cited as proof you can go very fine — but Monzo also built enormous tooling to make that survivable, and they are a bank with thousands of engineers. For everyone else, the lesson of the far right of the spectrum is: *more services is not more better*. There is an optimum, and it is almost never "as many as possible."

The skill is not picking the rightmost position. The skill is knowing which position your specific team and product belong in *today*, and moving along the dial deliberately as your forcing functions change.

## 3. The running example: ShopFast as a monolith and as services

Let us make this concrete with ShopFast, the e-commerce platform we will carry through the whole series. ShopFast has seven recognizable business capabilities: a **catalog** of products, a **cart**, **orders**, **payments**, **inventory**, **shipping**, and **notifications**. In the monolith, those are seven modules in one codebase. In the microservices version, they are seven services. Same capabilities, different shape.

Here is what the *same* operation — placing an order — looks like inside the monolith. It is just code calling code, all inside one database transaction:

```python
# ShopFast monolith: place_order is a single in-process call path.
# Everything below runs in ONE database transaction.
def place_order(user_id: int, cart_id: int) -> Order:
    with db.transaction():                      # one ACID transaction
        cart = cart_module.get_cart(cart_id)    # in-process call, ~1us
        inventory_module.reserve(cart.items)    # in-process call, ~1us
        order = order_module.create(user_id, cart.items)
        payment_module.charge(user_id, order.total_cents)
        shipping_module.schedule(order.id)
        notification_module.enqueue_receipt(order.id)
        return order
    # If charge() raises, the WHOLE transaction rolls back automatically:
    # no reservation, no order, no shipment. Atomicity is free.
```

Read what is happening there and appreciate how much you are getting for free. Five capabilities cooperate, and if any one of them fails — the payment is declined, the inventory is gone — the database rolls back *everything*. No order is half-created. No inventory is left reserved for a sale that never happened. You did not write a single line of cleanup logic; the database's transaction did it for you. The calls are microseconds because they are function calls in the same process. If you want to change how `reserve` works, you change one file, run the tests for the whole app, and deploy one thing.

Now here is the same operation in the microservices version. The order service is the orchestrator, and every other capability is a network call to a separate service that owns its own data:

```python
# ShopFast order service: the same operation is now a fan-out of
# network calls, each of which can be slow, fail, or partially succeed.
async def place_order(user_id: int, cart_id: int) -> Order:
    cart = await cart_client.get_cart(cart_id)          # HTTP/gRPC, may time out
    await inventory_client.reserve(cart.items)          # network call, +ms
    order = orders_db.create(user_id, cart.items)       # our OWN db, local txn

    try:
        await payment_client.charge(user_id, order.total_cents)
    except PaymentError:
        await inventory_client.release(cart.items)      # MANUAL rollback!
        orders_db.mark_failed(order.id)
        raise

    # shipping and notifications happen ASYNCHRONOUSLY via an event:
    await event_bus.publish("order.placed", {"order_id": order.id})
    return order
```

Look at everything that changed. There is no longer a single transaction wrapping the whole thing, because the inventory, payment, and order data each live in a *different* database owned by a *different* service, and you cannot put a single ACID transaction across separate databases on separate machines. So when the payment fails *after* you reserved inventory, the database will not undo the reservation for you — you have to write that compensation by hand, calling `inventory_client.release` yourself, and you have to hope *that* call does not also fail. This hand-rolled "do a thing, and if a later step fails, undo the earlier steps" pattern has a name — the **saga pattern** — and it is so central that it gets a [dedicated post in this series](/blog/software-development/microservices/the-saga-pattern-in-practice) and a [mechanism deep-dive in the database track](/blog/software-development/database/saga-pattern-distributed-transactions). Notice also that shipping and notifications no longer happen inline; they are fired off as an *event* that those services will pick up later, which means they happen *eventually*, not immediately, and now you are living with [eventual consistency](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

That is the trade in two code blocks. The monolith version is simpler, faster, and atomic. The microservices version lets the cart, inventory, payment, shipping, and notification teams each own, deploy, and scale their piece independently — at the cost of writing, by hand, all the failure handling that the database used to do for free. The figure below makes the fan-out explicit: one shopper click becomes a cascade of network calls across services that each guard their own data.

![A service topology graph showing a single shopper click entering the API gateway and the order service fanning out to payment, inventory, and shipping services each owning separate data](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-3.webp)

## 4. The real costs you are buying

Whenever someone proposes microservices, the right first question is not "what do we gain" but "what do we pay," because the costs are concrete, immediate, and arrive whether or not the benefits ever do. There are five of them, and I want you to be able to recite all five. They are not abstract; each one is a specific category of work and risk that simply did not exist in your monolith.

**Network latency and partial failure.** Every in-process call you replace with a network call adds latency — typically somewhere from a few hundred microseconds on a fast local network to several milliseconds, plus the cost of serializing and deserializing the request and response. That is the cheap part. The expensive part is *partial failure*: a network call can fail in ways a function call never can. It can time out (you sent the request, you do not know if it arrived). It can succeed on the server but fail to return the response (so the work was done, but you think it failed). The remote service can be up but slow, which is often worse than being down, because slow calls hold your threads hostage. None of these failure modes exist inside a single process; all of them are now your problem, every single call. The first lie engineers tell themselves here is one of the famous *fallacies of distributed computing*: "the network is reliable." It is not, and a whole track of this series — [resilience patterns like timeouts, retries, circuit breakers, and bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — exists only because of this cost. The figure below shows what changes the instant a call crosses the wire.

![A before and after figure contrasting a one-microsecond in-process function call that always returns against a network call that adds milliseconds, can time out or return an error, and must obey a versioned contract](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-5.webp)

**Distributed data and no easy transactions.** In the monolith, "do these five things atomically or do none of them" is one line: `with db.transaction()`. In microservices, the five things touch five databases, and there is no transaction that spans them. You have two unappealing options. You can use distributed transaction protocols like two-phase commit, which are slow, fragile, and hold locks across the network — almost nobody uses them at scale. Or you accept eventual consistency and build sagas, where each step has a compensating action to undo it, and you live with windows of time where the system is in a partially-completed state (the inventory is reserved but the order is not yet confirmed). This is not a tooling problem you can buy your way out of; it is a fundamental consequence of splitting your data, and it changes how you have to *think about correctness*. Every "transaction" becomes a workflow that can be interrupted halfway, and you have to design for what the world looks like at every halfway point. The mechanism deep-dives live in the database track: the [transactional outbox and CDC](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and the broader [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

**Operational and observability overhead.** This is the cost that the three-person startup in my opening story drowned in, and it is the one most often underestimated. Every service you add needs its own deploy pipeline, its own monitoring and alerting, its own logging, its own health checks, and its own place in your on-call rotation. A single bug report that used to mean "grep the one log file" now means correlating logs across four services to figure out where the request actually died — which is why distributed tracing stops being a nice-to-have and becomes mandatory. The figure below shows the *stack* of operational concerns each individual service must carry; notice that the business logic, the part you actually care about, is the small box at the bottom. Everything above it is tax, and you pay it per service.

![A layered stack figure showing the operational tax each microservice carries, with CI and CD, observability, resilience, security, and on-call stacked above the small layer of actual business logic](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-8.webp)

**Organizational and coordination cost.** Here is the subtle one, and the one that separates a senior view from a junior one. Microservices do not just change your code; they change how your *people* have to work. To get the benefit of independent deployment, you need teams that own services end to end — which means you need enough people to staff those teams. A service with no clear owning team becomes an orphan that nobody maintains. And the *interfaces* between services become interfaces between *teams*: changing an API now requires a conversation, a version negotiation, a contract test, sometimes a multi-week migration where both the old and new versions must run at once. This is **Conway's Law** in action — your system's structure will mirror your organization's communication structure — and it cuts both ways, which is why we devote a whole post to [Conway's Law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices). If you have three engineers, you do not have three teams; you have one team, and giving that one team eleven services to coordinate just means they are coordinating with themselves across an unnecessary network.

**Eventual consistency leaking into the product.** The final cost is that distributed data does not just complicate your backend — it changes what your *users* experience, and sometimes what your product manager has to accept. When the order service confirms an order and publishes an event, the analytics service, the recommendation service, and the email service all update *some milliseconds or seconds later*. A user who places an order and immediately checks their order history might, for a beat, not see it. "Read-your-own-writes" — the simple expectation that you can see the thing you just did — is free in a monolith and a genuine engineering effort in microservices. You will find yourself explaining to non-engineers why the dashboard number is briefly wrong, and you will be designing UI affordances ("your order is processing") to paper over consistency windows that did not exist before. The mechanism is covered in the [eventual-consistency-in-practice post](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice).

Five costs: latency and partial failure, distributed data, operational overhead, organizational coordination, and eventual consistency. Memorize them. When someone says "let's go microservices," your job is to ask which of these five they have a plan for.

It is worth being precise about *when* each cost actually arrives, because they do not arrive at the same time, and underestimating the schedule is how teams get surprised. The latency and the operational-overhead costs arrive *immediately*, on day one, the moment you draw the first network boundary — there is no honeymoon on those two. The distributed-data and eventual-consistency costs arrive on the day you ship your *first feature that spans two services*, which is usually within the first few weeks, because almost any real feature touches more than one capability. The organizational-coordination cost arrives more slowly and insidiously, ramping up as the number of services and the number of cross-service features grows, until one day you notice that every change requires a meeting. The reason the costs sink teams is precisely this staggered arrival: the early costs feel manageable, the team congratulates itself, and the largest cost — coordination — only reveals its full weight months later, by which point the architecture is hard to reverse. A senior engineer prices in the *whole* cost curve up front, not just the day-one bill.

## 5. The real benefits you are getting

I have spent a lot of words on costs, deliberately, because the costs are what get skipped in the hype. But microservices are not a mistake — they exist because, *at the right scale and for the right reasons*, the benefits are enormous and genuinely unavailable any other way. There are four, and the honest framing is that each benefit is the *flip side* of a cost: you pay the network tax to buy independence, you pay the operational tax to buy isolation. Let us name the four.

**Independent deployment and scaling.** This is the headline benefit, the one everything else is downstream of. When the payment team can deploy a fix at 2pm without waiting for the catalog team's release train, and when you can run forty replicas of the inventory service during a flash sale while running three of the rarely-used returns service, you have something a monolith simply cannot give you. In a monolith, *everything* deploys together — so the slowest, riskiest part of the system gates the release of everything else, and *everything* scales together — so to handle ten times the read traffic on the catalog, you have to run ten times as many copies of the *entire* application, including the parts that have no extra load. Independent deploy means small, frequent, low-risk releases. Independent scale means you spend compute only where the load actually is. For a system with wildly uneven load across capabilities, the cost savings of scaling each service to its own demand can be substantial and very real.

#### Worked example: independent scaling saves real money

Suppose ShopFast runs a flash sale. The catalog and cart endpoints see a 20× traffic spike; the returns and account-settings endpoints see no change at all. In the monolith, the *whole application* is one deploy unit, so to serve the spike you must scale the entire monolith. Say the monolith needs 4 replicas normally at a cost of \$0.20 per replica-hour; to absorb 20× you might run 60 replicas (you cannot scale just the hot path), costing 60 × \$0.20 = \$12.00 per hour during the sale. In the microservices version, you scale *only* catalog and cart. If catalog and cart together are 30% of the system's compute, you run roughly 0.30 × 60 ≈ 18 hot-path replicas plus the unchanged baseline for everything else (say 6 replicas), for about 24 × \$0.20 = \$4.80 per hour. Over a 12-hour sale that is \$144 versus \$57.60 — a 60% saving on the spike, every sale, because you stopped paying to scale the cold parts of the system. That is the benefit, quantified. Note, though, that you only realize it if your load is genuinely uneven; if every capability scales together, this benefit evaporates and you are left holding only the costs.

**Team autonomy.** When a team owns a service end to end — its code, its database, its deploy, its on-call — they can move at their own pace. They choose when to refactor, they are not blocked waiting on a shared release, they are not afraid that their change will break someone else's unrelated feature because the blast radius is bounded by the service boundary. This is, fundamentally, an *organizational scaling* tool. Amazon's famous "two-pizza teams" — teams small enough to be fed by two pizzas — work precisely because each team owns its services and its APIs, so hundreds of teams can ship in parallel without a central coordination bottleneck. The benefit here is not technical throughput; it is *human* throughput. You can put a hundred engineers to work without them constantly stepping on each other.

**Fault isolation.** In a well-designed microservices system, a failure in one service does not take down the others. If the recommendation service falls over, the catalog still serves, the cart still works, and the checkout still completes — the page just renders without recommendations. In a monolith, a memory leak or an unbounded query in *any* module can exhaust the resources of the *whole* process and take everything down with it, because it is all one process sharing one heap and one thread pool. Microservices give you bulkheads: the blast radius of a failure is, at least in principle, contained to one service. I say "in principle" because this benefit is *not automatic* — without circuit breakers and timeouts, a slow downstream service can drag down its callers in a cascade, turning a one-service problem into a system-wide outage. Fault isolation is a benefit you have to *engineer*, which is why the resilience track exists.

**Technology heterogeneity.** Because services communicate only over the network through language-agnostic contracts, each service can be written in whatever language and use whatever datastore fits its job. The image-processing service can be in Rust for speed; the data-science service can be in Python for the libraries; the high-throughput ingestion service can use a different database than the transactional order service. In a monolith, everything shares one runtime and, usually, one primary datastore. Heterogeneity is a real benefit, but I will be honest: it is the *most overrated* of the four. Most organizations are better off standardizing on a small number of languages and datastores than letting every team pick its own, because every additional language is another thing to secure, monitor, hire for, and keep patched. Use heterogeneity surgically, not as a default.

Four benefits: independent deploy and scale, team autonomy, fault isolation, technology heterogeneity. Notice that three of the four are fundamentally about *people and organization*, not about raw technical performance. That is the deepest truth about microservices and the one I most want you to carry: **microservices are primarily a solution to an organizational scaling problem, dressed up as a technical architecture.** If you do not have an organizational scaling problem — if you do not have many teams blocked on each other — you probably do not have the problem microservices solve, and you will pay all the costs for none of the benefit.

## 6. The trade-off matrix

Let us put the whole comparison in one place, because a senior engineer carries this table in their head and reaches for it in every architecture review. The figure below scores the three live positions — monolith, modular monolith, and microservices — across the five axes that actually decide the call: deploy independence, data consistency, operational cost, team scaling, and call latency. Read it as "what does each option give me on each axis," and notice how much green sits on the monolith side until you reach the two axes — deploy independence and team scaling — where microservices, and *only* microservices, win.

![A decision matrix scoring monolith, modular monolith, and microservices across deploy independence, data consistency, operational cost, team scaling, and call latency, showing the monolith winning most axes until team scaling and deploy independence](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-4.webp)

Here is the same comparison as a table you can copy into a design doc, with the reasoning made explicit:

| Axis | Monolith | Modular monolith | Microservices |
| --- | --- | --- | --- |
| Deploy independence | None — all ships together | None — still one unit | Per service — the core win |
| Data consistency | ACID, transactions free | ACID, transactions free | Eventual; sagas by hand |
| Operational cost | Low — one thing to run | Low — one thing to run | High — multiply per service |
| Team scaling | Hard past ~8 engineers | Medium — modules help | Scales to many teams |
| Call latency | ~1 microsecond, in-process | ~1 microsecond, in-process | +milliseconds per hop |
| Refactoring across boundaries | Compiler-checked, easy | Compiler-checked, easy | Cross-service migration |
| Debugging a request | One log, one stack trace | One log, one stack trace | Distributed tracing needed |
| Blast radius of a bug | Whole process | Whole process | One service (if engineered) |
| Tech choice | One stack | One stack | Per service (use sparingly) |

The pattern in that table is the whole argument of this post. Microservices win exactly two rows decisively — deploy independence and team scaling — and those two rows are about *organizational* scale, not about whether your code runs better. On every other axis the monolith or modular monolith is equal or better. So the decision rule writes itself: **adopt microservices when, and only when, the two rows they win are the rows that are actually hurting you.** If your pain is "we have eight teams blocked on one release train," microservices are the cure. If your pain is "we want our code to be cleaner" or "we read that we should," you want a modular monolith, and you can have all the cleanliness with none of the distributed-systems tax. The fact that there is no single dominant column is exactly why this is a genuine engineering trade-off and not a fashion choice; for the broader skill of [articulating trade-offs like this in a review](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond), the system-design track goes deeper.

## 7. When NOT to use microservices

This is the most important section in the post, so I am going to be blunt. There are several situations where microservices are not merely overkill but actively self-defeating — where the monolith (ideally a *modular* one) wins on every axis you actually care about. If you are in one of these, do not split. Here they are, with the reasoning, because "don't" without "why" is useless.

**You are an early-stage startup that has not found product-market fit.** Before you know what your product *is*, your domain boundaries are guesses, and microservice boundaries are extremely expensive to move. When you split a monolith and put a network boundary between two modules, you have *frozen* that boundary — moving it later means a cross-service data migration, dual-running two versions, and rewriting contracts, which is an order of magnitude harder than moving a boundary inside a single codebase where the compiler helps you. Early on, your boundaries *will* be wrong, because you are still discovering the domain. A monolith lets you move those boundaries cheaply with a refactor; microservices punish you for every wrong guess. The right move pre-PMF is a modular monolith that lets you *learn* the boundaries before you *commit* to them.

**You have a small team.** This is, in my experience, the single most reliable predictor that microservices are a mistake. The benefits of microservices — independent deploy, team autonomy, fault isolation across team boundaries — are *organizational* benefits, and a small team does not have an organizational scaling problem to solve. Three engineers do not need to deploy independently of each other; they sit next to each other and can just coordinate. Three engineers with eleven services are not eleven autonomous teams; they are three people context-switching across eleven on-call rotations, eleven deploy pipelines, and eleven sets of dependencies to patch. The coordination they hoped to remove between teams reappears as coordination *within* their own heads, plus all the network and operational tax on top. A rough rule of thumb: if you cannot staff each service with a team that can own it, you do not have enough people for microservices. Below roughly two or three genuine teams, the monolith wins.

**Your domain boundaries are unclear.** Even with a big team, if you do not yet understand how your domain decomposes — if you cannot confidently say "this is the order capability and it is separate from the inventory capability and here is the clean interface between them" — then any boundaries you draw will be wrong, and wrong service boundaries produce the worst of all architectures: the **distributed monolith**, where services are technically separate but so entangled that they must always deploy together. You get every cost of microservices and none of the benefits. The cure is to *not split until the boundaries are clear*, which usually means running as a modular monolith long enough to learn where the natural seams are. We have a whole post on the [shared-data anti-patterns that create distributed monoliths](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith).

**Your scale is low.** If your traffic fits comfortably on a few servers, the independent-scaling benefit is worth nothing to you, because you are not scaling anything. The latency cost of network hops, however, is just as real at low scale as at high scale. So at low scale you pay the full latency and operational tax and collect none of the scaling reward. A system serving a thousand requests a second does not need to be distributed; it needs to be *correct and cheap to operate*, and a monolith is both.

The figure below turns all of this into a decision tree you can actually walk through with your team. Notice that most paths through it end at "stay a modular monolith," not because microservices are bad, but because the *forcing functions* that justify them — many teams blocked on each other, a part of the system that genuinely needs to scale separately, clear and stable domain boundaries — are simply not present for most teams most of the time.

![A decision tree asking whether you have a forcing function such as multiple blocked teams or a part needing separate scale, with most branches routing the reader to stay a modular monolith and only clear forcing functions leading to extracting or splitting services](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-6.webp)

There is a positive reframing of all this, due to Martin Fowler, called **MonolithFirst**: start with a monolith, and only extract services once you have learned, by living with the system, where the real boundaries are and which parts genuinely need to be independent. The complementary idea, **MicroservicePremium**, is Fowler's name for the fact that microservices carry a *premium* — a fixed cost you pay just to play — and you should only pay that premium when the system has grown complex enough that the premium is worth it. Both ideas point the same direction: the default is a monolith, and the burden of proof is on the person who wants to split. We expand the practical recipe in [monolith first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith).

## 8. How services actually talk: a concrete look

To make the network boundary tangible rather than abstract, let us look at the actual shapes a service-to-service call takes, because "they communicate over the network" is doing a lot of hiding. There are two broad styles, and your choice between them is one of the most consequential design decisions in the whole system; we devote a full post to [REST vs gRPC vs GraphQL](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis), but here is the gist so the costs above feel real.

The **synchronous request/response** style — the order service calls the payment service and *waits* for the answer — is the most intuitive and the one that maps most directly onto the function call it replaced. Here is what reserving inventory might look like as a gRPC contract, defined in a `.proto` file that is the published interface between the two services:

```protobuf
// inventory.proto — the contract between order service and inventory service.
// This file IS the interface. Both teams agree on it; either can regenerate
// a typed client or server from it in their own language.
syntax = "proto3";
package shopfast.inventory.v1;          // note the v1: APIs are versioned

service Inventory {
  rpc Reserve(ReserveRequest) returns (ReserveResponse);
  rpc Release(ReleaseRequest) returns (ReleaseResponse);
}

message ReserveRequest {
  string idempotency_key = 1;           // so a retry does not double-reserve
  repeated LineItem items = 2;
}
message LineItem {
  string sku = 1;
  int32  quantity = 2;
}
message ReserveResponse {
  bool reserved = 1;
  string reservation_id = 2;
}
```

Notice three things that did not exist in the monolith and now must exist. There is a *version* in the package name (`v1`), because the contract will change and old callers must keep working during the transition — the discipline of [API versioning and contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) is born here. There is an *idempotency key*, because network calls can be retried and you must not reserve inventory twice for one logical request — the whole topic of [idempotency across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) is born here too. And the request and response are now *explicit messages* that must be serialized and deserialized, with a cost in both latency and the ongoing maintenance of keeping the two teams' understanding of these messages in sync.

The **asynchronous event** style is the other shape: instead of calling the shipping service and waiting, the order service publishes an `order.placed` event to a message broker and moves on; the shipping service consumes that event whenever it is ready. This decouples the two services in time — the shipping service can be down for a minute and the order still completes, because the event waits in the broker. The cost is that shipping now happens *eventually*, and you have to reason about ordering, duplicates, and what happens if the event is processed twice. The mechanism deep-dives live in the message-queue track — the [anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) and [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — and the microservices-specific patterns are in [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration).

To see how few moving parts even a *small* slice of ShopFast becomes once it is distributed, here is a stripped-down `docker-compose` sketch of just three services and their infrastructure. Count the boxes; every one of them is something to configure, monitor, secure, and keep running:

```yaml
# docker-compose.yml — a MINIMAL slice of ShopFast as services.
# Three services, three databases, a gateway, and a broker — and this is small.
services:
  gateway:
    image: shopfast/gateway:1.4.2
    ports: ["8080:8080"]
    environment:
      ORDER_URL: "http://order:9001"
      CATALOG_URL: "http://catalog:9002"

  order:
    image: shopfast/order:2.1.0
    depends_on: [order-db, broker]
    environment:
      DB_DSN: "postgres://order-db:5432/orders"   # its OWN database
      BROKER_URL: "amqp://broker:5672"
  order-db:
    image: postgres:16

  inventory:
    image: shopfast/inventory:1.9.3
    depends_on: [inventory-db]
    environment:
      DB_DSN: "postgres://inventory-db:5432/inventory"  # a DIFFERENT database
  inventory-db:
    image: postgres:16

  broker:
    image: rabbitmq:3.13
```

That is three services, but already two separate databases, a gateway, and a message broker — six processes to run, configure, and observe for what would be a handful of modules and one database in a monolith. Multiply by ten services and you start to feel the operational tax in your bones. This is the concreteness the term "operational overhead" was hiding.

Two pieces of that diagram deserve a closer look because they are infrastructure that *only exists because you went distributed* — they are pure tax that a monolith does not pay. The first is the **API gateway**, the single front door through which all external traffic enters before being routed to the right internal service. In a monolith there is no gateway, because there is only one thing to send requests to; the moment you have many services, something has to sit in front of them to terminate TLS, authenticate the caller once, apply rate limits, and route each request to the owning service — and that something is a new component you build, run, and keep highly available, because if it falls over, *everything* behind it is unreachable. We give it a full post: [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend). The second is **service discovery**: in the compose file above I hardcoded `http://order:9001`, but in a real cluster the order service has many replicas that come and go as they scale, crash, and redeploy, with changing IP addresses. Some mechanism has to answer the question "where is a healthy instance of the order service right now?" — that is service discovery and load balancing, another component that simply does not exist in a monolith where a module is just an import. It gets its own post too: [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing). The point of naming these is to drive home that the operational tax is not vague unease; it is a specific, enumerable list of new infrastructure — gateway, discovery, broker, per-service databases, per-service pipelines — each of which is a real thing a real human now has to own.

## 9. Optimization and production-grade reality: making the network call survivable

Suppose you have decided, for good reasons, to run microservices. The single most important thing to internalize is that a naive remote call — just calling the other service and waiting — is *not production-grade*, and the gap between the naive call and the production call is where most of the real engineering of this discipline lives. Let us make the bottleneck concrete and then fix it with numbers.

The core failure mode is the **cascade**. The order service calls the payment service synchronously. One day the payment service's dependency, the external payment provider, gets slow — not down, just slow, p99 going from 80ms to 4 seconds. The order service's threads that are waiting on payment now block for 4 seconds each instead of 80ms. With a fixed thread pool, those blocked threads pile up; soon every thread in the order service is stuck waiting on payment, so the order service stops responding to *anything* — including requests that have nothing to do with payment. The gateway's threads, in turn, block waiting on the order service, and now the whole site is down. A single slow dependency, four hops deep, has taken down the entire system. This is the cascade, and it is the defining failure mode of synchronous microservices.

The fix is a stack of resilience layers around every remote call, and the key insight is that you measure your way to the right numbers rather than guessing. A **timeout** caps how long you will wait — set to, say, the dependency's p99 plus headroom, maybe 200ms, so a slow call fails fast instead of holding your thread. A **retry with backoff and jitter** handles the transient blip — but capped at two attempts, because retries *amplify* load on an already-struggling service, and unbounded retries are how a small problem becomes a retry storm. A **circuit breaker** watches the error rate and, when too many calls to payment fail, *stops calling payment at all* for a cooldown window, failing fast and letting the downstream recover instead of pounding it. And a **bulkhead** caps how many concurrent calls to payment any one caller may have outstanding, so even if payment is slow, only a bounded slice of your threads can be stuck on it — the rest keep serving. These are the subject of the [resilience-patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads); here is the shape in code:

```python
# A production-grade remote call: timeout, bounded retry with jitter,
# circuit breaker, and a bulkhead concurrency limit. NEVER call raw.
breaker = CircuitBreaker(fail_max=20, reset_timeout_s=30)   # open after 20 fails
bulkhead = asyncio.Semaphore(50)                            # max 50 concurrent

@breaker
async def charge_payment(user_id: int, cents: int) -> bool:
    async with bulkhead:                                    # cap concurrency
        for attempt in range(2):                            # at most 2 tries
            try:
                return await payment_client.charge(
                    user_id, cents, timeout_s=0.2           # 200ms hard timeout
                )
            except (TimeoutError, TransientError):
                if attempt == 0:
                    await asyncio.sleep(0.05 + random.random() * 0.05)  # jitter
                else:
                    raise                                   # give up; fail fast
```

#### Worked example: latency math for a fan-out request

Let us quantify why fan-out depth matters and why p99 is the number that bites. Suppose the ShopFast checkout request, at the gateway, must call four services *sequentially*: cart, inventory, order, payment. Say each call has a p50 of 10ms and a p99 of 60ms. If you naively sum the p50s, you get 4 × 10 = 40ms and feel fine. But latency does not add at the median; *tail* latency dominates a fan-out. With four sequential dependencies, the probability that *all four* come in under their p99 is 0.99⁴ ≈ 0.96 — meaning roughly 4% of checkout requests will hit at least one 60ms+ call, and the *overall* p99 of the chained request is closer to 60 + 60 + 60 + 60 = 240ms in the bad case, plus the gateway's own overhead. The lesson: every synchronous hop you add inflates the tail, and the tail is what your users notice and your SLO measures. The optimizations follow directly: make calls *parallel* where they have no data dependency (cart and inventory can be fetched concurrently, cutting two sequential hops to one), push non-critical work (shipping, notifications) *off the synchronous path* and onto events so they do not inflate the user-facing latency at all, and set timeouts at the p99 so one slow call fails fast rather than dragging the whole chain. After those three changes, the same checkout might present a p99 of ~120ms (one parallel critical hop of payment plus order, with the rest async) instead of 240ms — a 2× improvement that came entirely from understanding the latency *shape*, not from faster hardware.

#### Worked example: the deploy cost of a three-engineer team

Now the organizational bottleneck, quantified, because this is the cost that sinks small teams. Take a team of three engineers. In a monolith, a deploy is one pipeline: one build, one test suite (say 8 minutes), one rollout, one thing to watch. Call it 20 minutes of attention per deploy, and they deploy maybe twice a day — 40 minutes a day of deploy overhead, shared. Now split into eleven services. Each service has its own pipeline, its own test suite, its own rollout to watch, its own dependency updates, its own security patches, its own on-call. Even if each service's *individual* deploy is faster (say 10 minutes), the team now maintains *eleven* pipelines, and a feature that spans four services means four coordinated deploys for one change — 40+ minutes for one feature that was a single deploy before. Add eleven sets of dependency upgrades (a CVE in a common library means eleven patch-and-deploy cycles instead of one), eleven sets of dashboards, and an on-call rotation spread so thin that one person is effectively on call for eleven services they half-remember. Conservatively, the operational overhead per engineer *triples*, and the team's feature throughput roughly *halves* — for a team that had no organizational scaling problem to solve in the first place. That is not a hypothetical; it is precisely the curve the next figure traces.

## 10. Stress test: what breaks when you split too early

A senior engineer does not just design a system; they stress-test the design by asking "what breaks?" before reality asks it for them. So let us stress-test the premature split — the three-person team that went to eleven services — and watch exactly where and when it fails, because the failure is not a single dramatic outage. It is a slow bleed of velocity, and the figure below traces its timeline.

![A timeline showing how splitting six services across three engineers leads within weeks to features spanning four services, hand-built sagas to replace lost transactions, repeated re-splitting of wrong boundaries, and velocity dropping to half of the monolith baseline](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-9.webp)

In **week one**, everything feels great. The services are clean, small, and the team feels modern and professional. This is the honeymoon, and it is dangerous precisely because it feels like success. By **week three**, the first real feature arrives — say, "apply a discount code at checkout" — and it turns out to touch the cart, order, payment, and pricing services all at once. What would have been a one-file change is now a four-service change requiring four coordinated deploys and a careful ordering so that no service is deployed expecting a contract the others have not shipped yet. By **week six**, a feature needs the kind of atomicity the database used to provide for free — "reserve inventory and charge the card, or do neither" — and since those now live in separate services with separate databases, the team has to build a saga by hand, with compensating actions and a state machine to track partial progress, which is a week of work to replace a single `BEGIN TRANSACTION`.

By **week ten**, the team realizes that one of their original boundaries was wrong — the "pricing" and "promotions" split they guessed at does not match how the domain actually works, because every promotion *is* a pricing rule. So they have to merge two services back together, which means a cross-service data migration and untangling two databases that should have been one. They do this twice, because the *next* boundary turns out to be wrong too. By **week fourteen**, the measurable result is in: the team's feature throughput is roughly *half* what it was as a monolith, the on-call burden has tripled, and the "modern architecture" they adopted to go faster has made them slower. Nothing dramatic broke. No single outage. The system just quietly became expensive to change — which, for a startup racing to find product-market fit, is the most dangerous failure mode of all, because it is invisible until you compare against the counterfactual.

Now run the *opposite* stress test, on a system that genuinely *should* be microservices — say ShopFast at the scale of a real e-commerce company, fifty teams, millions of orders a day. What breaks *there* if you stayed a monolith? The release train: fifty teams trying to merge into one deploy, where one team's bad commit blocks all fifty from shipping, where the build takes an hour and the test suite is flaky, where you can only deploy twice a week because the coordination cost of a release is enormous, where scaling the read-heavy catalog means running fifty copies of the entire monolith including the write-heavy order code. *That* is the failure mode microservices solve, and at that scale they solve it beautifully. The stress test, run both directions, gives you the rule: premature splitting bleeds velocity through coordination tax; staying monolithic too long bottlenecks an organization through the release train. The art is reading which failure mode is actually approaching *you*.

## 11. Case studies

Theory is cheap; let us look at what actually happened to companies that lived these trade-offs, because the real stories are more instructive than any principle, and they are all genuinely documented.

**Amazon and the two-pizza teams.** Amazon's transition, beginning in the early 2000s, from a large monolith to a service-oriented architecture is one of the founding stories of the whole movement. The organizational principle came first: small autonomous teams, each "small enough to be fed by two pizzas," each owning its services end to end with hard API contracts between them. A famous (and widely retold) internal mandate required all teams to expose their functionality only through service interfaces over the network — no shared databases, no back doors, no reaching into each other's data. The lesson Amazon teaches is the deepest one in this post: microservices were, for them, fundamentally an *organizational* tool to let thousands of engineers ship in parallel without a central coordination bottleneck. The architecture followed the org design, exactly as Conway's Law predicts. They did not adopt services to make the code prettier; they adopted services to make *the company* scale.

**Netflix's move off the monolith.** Netflix migrated from a monolithic data center application to a cloud-based microservices architecture on AWS over a multi-year effort that began around 2009, triggered in part by a major database corruption incident that took down their DVD-shipping system for days. At their scale — serving a huge fraction of global internet video traffic — the independent-scaling and fault-isolation benefits are enormous and genuinely necessary; you cannot run that load as one deployable. Netflix is also the origin of *chaos engineering*: they built Chaos Monkey to randomly kill production instances, precisely because in a large microservices fleet, partial failure is constant and the only way to be sure your system survives it is to *cause* it deliberately and continuously. The lesson: at genuine hyperscale, microservices are not optional, but they only work if you invest correspondingly in resilience tooling — fault isolation is engineered, not free.

**Segment's famous monolith U-turn.** This is the case study every junior should read twice, and the figure below traces it. Segment, a customer-data company, started with a monolith, then, around 2015, split into microservices — eventually growing to well over a hundred services and repositories, one per data destination. It seemed like the right call. But by 2017 the operational overhead had become crushing: every service needed its own dependencies, its own deploys, and shared libraries had to be updated across more than a hundred repos, which meant a single common change rippled into an enormous coordination effort. Developer productivity *fell*. So Segment did the thing nobody talks about: they *consolidated back*, merging their fleet into a smaller number of services (a system they called Centrifuge), and publicly wrote about why. The lesson is not "microservices are bad" — it is that *more services is not strictly better*, that the per-service operational tax is real and compounds, and that the senior move is to be willing to reverse a decision when the evidence says the trade-off has flipped. Going back to a (modular) monolith is not an admission of failure; it is engineering maturity. We return to this idea in [when to go back to monolith](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith).

![A timeline of Segment's architecture journey from a single monolith in 2013 through a per-destination microservice split that grew to over 140 repositories, drowning operations and dropping velocity, until they consolidated back to a smaller service called Centrifuge](/imgs/blogs/what-are-microservices-and-when-not-to-use-them-7.webp)

**Monzo's 1,500+ services.** At the opposite extreme, Monzo, a UK digital bank, famously runs well over 1,500 services on Kubernetes. This is sometimes cited as proof that you can go extremely fine-grained — and you can, *if* you build the tooling to make it survivable. Monzo invested heavily in standardization: a single language (Go) for almost everything, a uniform service template so every service looks the same, automated tooling for creating and deploying services, and a service-mesh-like network layer with strict rules about which services may call which. The lesson is the inverse of Segment's: a very large service count is *survivable* only with enormous investment in platform tooling and ruthless standardization. Monzo did not get there by letting every team do its own thing; they got there by making every service nearly identical in shape, so that the per-service tax was driven as close to zero as possible. If you are not willing to build that platform, you cannot run that many services — which is precisely why most companies should not try.

Put the four side by side and the meta-lesson emerges. Amazon shows microservices as *organizational scaling*. Netflix shows them as *necessary at hyperscale, paired with resilience investment*. Segment shows that the *trade-off can flip* and reversing is wise. Monzo shows that *extreme granularity demands extreme tooling*. None of the four adopted microservices because they were fashionable; each had a specific forcing function, and each paid a specific, deliberate price. That is the posture to copy.

## 12. So when, exactly, do you reach for them?

Let me collapse everything into a decision you can defend in a review, because vague advice helps no one. Reach for microservices when *all* of these are true, and be very skeptical when any is false.

Reach for them when you have **multiple teams blocked on each other** — when the coordination cost of shipping has become the bottleneck, when teams are queuing behind a shared release train, when one team's risky change keeps blocking another team's safe one. This is the single strongest signal, because it is the organizational scaling problem microservices were built to solve. Reach for them when **a specific part of the system genuinely needs to scale or be deployed independently** — a hot path with wildly different load from the rest, a component with a different reliability or compliance requirement, a piece that must release on a different cadence. Note that this often justifies extracting *one or two* services from a monolith, not splitting the whole thing — the strangler-fig approach we cover in [migrating a monolith to microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices). And reach for them when **your domain boundaries are clear and stable** — when you have lived with the system long enough to know, with confidence, where the natural seams are, so that the boundaries you freeze into network contracts are the *right* ones.

Do *not* reach for them when you are pre-product-market-fit, when you have a small team, when your boundaries are still guesses, or when your scale is low. In every one of those cases the modular monolith wins on every axis you care about: it is cheaper to operate, faster to change, keeps your transactions atomic, and — crucially — keeps your boundaries *soft* so you can move them as you learn. The senior posture is to *default to the monolith and demand a forcing function before splitting*. You should be able to finish the sentence "we are splitting service X out of the monolith because ___" with a concrete, present-tense pain, not a future hypothetical or a fashion. If you cannot finish that sentence, you are not ready, and the honest thing — the thing that separates a senior from someone merely chasing the new — is to say so.

For the broader meta-skill of reasoning through this kind of open-ended architectural ambiguity, where there is no single right answer and the job is to surface the trade-offs and pick deliberately, the system-design track's [how seniors approach ambiguous system design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems) and [back-of-the-envelope estimation](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) are the natural companions to this post — the first teaches the reasoning posture, the second the numbers to back it up.

## Key takeaways

- **A microservice is independently deployable, owns one business capability and its data, and talks only over the network.** All four properties must hold. Drop one — most commonly independent deployability or owned data — and you have a distributed monolith, which is worse than a monolith.
- **It is a spectrum, not a switch.** Monolith → modular monolith → microservices → nanoservices. The modular monolith is the underrated sweet spot and the right default for most teams; nanoservices are a real failure mode where coordination dwarfs the work.
- **Microservices are a trade, not an upgrade.** They win decisively on exactly two axes — deploy independence and team scaling — and cost you on every other: latency, distributed data, operational overhead, organizational coordination, and eventual consistency.
- **The benefits are mostly organizational.** Three of the four real benefits — team autonomy, fault isolation across boundaries, and independent deploy — are about scaling *people*, not code. If you do not have an organizational scaling problem, you probably do not have the problem microservices solve.
- **Default to the monolith; demand a forcing function before splitting.** A junior reaches for microservices because they read they should; a senior reaches for them only when multiple teams are blocked, a part genuinely needs separate scale, and the domain boundaries are clear and stable.
- **Splitting too early bleeds velocity invisibly.** No dramatic outage — just features that span four services, hand-built sagas replacing free transactions, and re-splitting of wrong boundaries until throughput halves. The danger is that it feels modern while it makes you slow.
- **The network call is not free and not reliable.** Every synchronous hop inflates tail latency and introduces partial failure; a production call needs timeouts, bounded retries, a circuit breaker, and a bulkhead, or one slow dependency cascades into a full outage.
- **Be willing to go back.** Segment's monolith U-turn is engineering maturity, not failure. When the trade-off flips, reversing the decision is the senior move.

## Further reading

- Sam Newman, *Building Microservices*, 2nd edition (O'Reilly) — the definitive practitioner's book; chapters on boundaries and on "is your organization ready?" are essential before any split.
- Chris Richardson, *Microservices Patterns* (Manning) — the pattern catalog this series builds on; pairs especially well with the data and resilience tracks.
- Martin Fowler, "MonolithFirst" and "MicroservicePremium" (martinfowler.com) — short, sharp essays arguing the default should be a monolith and that microservices carry a fixed premium worth paying only past a complexity threshold.
- Segment Engineering, "Goodbye Microservices: From 100s of problem children to 1 superstar" — the public write-up of the monolith U-turn; read it for the operational-tax reality.
- Next in this series: [monolith first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) for the practical recipe, and [service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design) for how to find the seams before you cut.
- For the mechanisms this post only gestured at: the [saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions) for distributed transactions, [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) for eventual consistency, and [how seniors approach ambiguous system design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems) for the decision posture.

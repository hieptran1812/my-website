---
title: "Service Boundaries With Domain-Driven Design: Drawing the Lines That Decide Everything"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn the single hardest microservices skill — drawing the right service boundaries — through Domain-Driven Design taught in plain language, so you split by business capability instead of building a chatty distributed monolith."
tags:
  [
    "microservices",
    "domain-driven-design",
    "bounded-context",
    "service-boundaries",
    "ddd",
    "software-architecture",
    "distributed-systems",
    "backend",
    "eventstorming",
    "aggregates",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/service-boundaries-with-domain-driven-design-1.webp"
---

A team I worked with once split their monolith into microservices over a long, proud quarter. They drew the lines the way it felt natural to draw them: a User service, an Order service, a Product service, a Payment service. Clean nouns, one service per table, everyone got a repo. Six months later their checkout flow made seven synchronous calls across four services for a single button press, every "place order" needed a distributed transaction nobody trusted, two of the four services could only deploy together because changing an order field meant changing a user field, and three teams shared one Postgres instance because "the data was related." They had paid the full operational tax of microservices — the network hops, the YAML, the on-call rotations — and kept every coupling of the monolith. They had built a *distributed monolith*: the worst of both worlds, slower and more fragile than the thing they replaced.

The mistake was not technical. Their Kubernetes was fine, their gRPC was fine, their CI was fine. The mistake was the boundaries. They had cut the system along the wrong seams, and once a boundary is wrong, no amount of good engineering downstream can save you — you are forever paying for a partition that does not match how the business actually changes. This is the most important and least taught skill in the whole discipline: **where do you draw the line between one service and the next?** Get this right and the rest of microservices is mostly plumbing. Get it wrong and you will spend years fighting the symptoms — chatty calls, sagas you did not need, shared databases you cannot untangle — without ever naming the cause.

![A before-and-after comparison contrasting an entity-split distributed monolith on the left with a capability-aligned set of services on the right](/imgs/blogs/service-boundaries-with-domain-driven-design-1.webp)

The figure above is the whole post in one image. On the left, the team that split by entity: services that share a database, fan out seven calls per checkout, and deploy in lockstep. On the right, the same system split by *business capability*: each service owns its data, runs its core transaction locally, and ships on its own schedule. The tool that gets you from left to right is **Domain-Driven Design** (DDD), a body of practice from Eric Evans's 2003 book that, stripped of its jargon, is really one idea: *let the structure of the business decide the structure of the software.* By the end of this post you will be able to take a tangle of features, run an EventStorming session over it, find the **bounded contexts** that should become your services, identify the **aggregates** that define your transaction boundaries, place an **anti-corruption layer** where a foreign model would otherwise leak in, and defend each boundary with a trade-off matrix and a count of cross-service calls. We will build all of it on one running example — a fictional e-commerce system called **ShopFast** — so the abstractions always have something concrete to grab onto.

This is post three in the *Microservices, From Junior to Senior* series, and it sits deliberately right after [monolith-first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith). That post argued you should usually start with a monolith and only split when you have a reason. This post is about *how* to split when the time comes — and, just as importantly, about discovering the seams *before* you cut, so that the modules inside your monolith already follow the lines your future services will. Boundaries are a design decision you can make on day one, on paper, before a single service exists.

## 1. Why the boundary is the decision that dominates everything

Let me make the stakes concrete before we reach for theory, because the theory only lands once you feel the pain it prevents.

A service boundary is, at bottom, a line that says "this data and this logic live on one side; that data and that logic live on the other; and the only way across is a network call over a contract." That last clause is the whole story. *Inside* a boundary, a method call is free: nanoseconds, in-process, transactional, refactorable in one commit. *Across* a boundary, the same call becomes a network round trip — milliseconds, fallible, untransactional, and frozen behind a contract that two teams must agree to change. The boundary is where cheap becomes expensive. So the only question that matters when you draw it is: **am I putting things that change together and call each other constantly on the same side, and things that change independently on opposite sides?**

When you get this wrong in the "everything is too connected" direction — you draw a boundary through the middle of something that is really one thing — three specific diseases appear, and they are the three signatures of a distributed monolith. First, **chattiness**: a single user action triggers a storm of cross-service calls, because the data it needs was scattered across services. Your latency budget, which had room for one or two network hops, gets eaten by seven, and your p99 balloons because the slowest of seven independent calls dominates. Second, **distributed transactions**: an operation that must be atomic — reserve stock *and* charge the card *or* neither — now spans services, so you cannot use a database transaction, and you are forced into a [saga](/blog/software-development/database/saga-pattern-distributed-transactions) with its compensating actions and its eventual consistency, to solve a problem you only have because the boundary was wrong. Third, the **shared database**: when two services both need the same data and the boundary did not give either of them ownership, the tempting fix is to let them both read the same tables — and now you have coupled their schemas, their migrations, and their deploys forever, which is the single most destructive anti-pattern in the field.

Each of these is the subject of its own post later in this series — the saga pattern in [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), the shared-database trap in [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), data ownership in [database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices). The reason I am front-loading them here is that **all three are usually symptoms of a boundary mistake, not independent problems.** A senior engineer, shown a system drowning in sagas and chatty calls, does not first reach for a better saga framework. They first ask: are the boundaries wrong? Should these two services be one? Did we split a single consistency requirement across a network? The boundary is the root; the patterns are the bandages. You want the patterns *after* you have done everything you can with the boundaries, not instead of doing it.

There is a beautiful, slightly uncomfortable corollary here, and it is **Conway's Law**: organizations ship systems whose structure mirrors their communication structure. If your boundaries do not match your team boundaries, every change becomes a cross-team coordination meeting. We have a whole post on this — [Conway's Law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices) — but flag it now, because the deepest reason to draw boundaries by business capability is that *capabilities map to teams*, and a service that one team can own end to end is a service that can actually move fast. A boundary is simultaneously a technical line, a data line, and an org line. The art is finding the one place to draw it that satisfies all three.

## 2. The two wrong ways to split (that everyone tries first)

Before the right way, the wrong ways — because they are seductive, they look reasonable in a whiteboard sketch, and nearly every team reaches for one of them before they have heard of DDD. Naming them precisely is half the cure.

![A tree diagram branching from a single question into three decomposition strategies showing technical-layer and entity as poor choices and business capability as the strong choice](/imgs/blogs/service-boundaries-with-domain-driven-design-2.webp)

### Wrong way one: split by technical layer

The first instinct, especially for teams steeped in layered architecture, is to make the *layers* into services. A "presentation service" (the UI/API tier), a "business logic service," a "data access service." It feels like microservices because there are multiple deployable units, and it maps cleanly onto the n-tier diagram everyone already has in their head.

It is a disaster, and the reason is structural: **almost every feature you will ever build cuts vertically through all three layers.** Adding a "gift wrapping" option to checkout touches the UI (a checkbox), the logic (price adjustment, validation), and the data (a new column). With layer-as-service, that one feature now requires a coordinated change across three services and three teams, three deploys in the right order, and a contract negotiation between the logic tier and the data tier. You have taken changes that *naturally* travel together — the vertical slice of one feature — and forced them across the most expensive boundaries in your system. The classic symptom is that nothing ever ships without all three layer-teams in the room. You optimized for an architectural diagram instead of for the direction of change.

### Wrong way two: split by entity (the "noun service")

The second instinct is subtler and far more common, because it produces nice clean service names that sound right in a standup: a **User service**, an **Order service**, a **Product service**, an **Address service**. One service per database table, more or less — one service per *noun* in your data model. This is what the ShopFast team in the intro did, and it is the road to the distributed monolith.

The problem is that **a business operation almost never lives inside a single noun.** "Place an order" reads the user, reads products, reads addresses, writes an order, decrements inventory, and triggers a payment. With a noun-per-service split, that one operation becomes an orchestra of cross-service calls — and worse, the *data* an operation needs is spread across the noun-services, so to do anything useful a service has to constantly ask other services for data. You get **anemic services**: thin CRUD wrappers around a table, with no real behavior, because all the behavior is in the operations that span them. The User service does `getUser`, `createUser`, `updateUser` — it has no logic, because there is no such thing as "user logic" in isolation; users only matter in the context of ordering, or support, or billing, and those are where the real rules live.

The deepest tell of the entity trap is the **"God entity"** — usually User or Account — that *everything* depends on, because every part of the system references it. When every service calls the User service, the User service becomes a single point of failure, a latency bottleneck on every request, and a deployment chokepoint nobody can change without breaking ten consumers. You took the most-referenced row in your database and turned it into the most-called service in your fleet. That is not decomposition; it is centralization with extra network hops.

Here is the thing both wrong ways share: **they optimize for the structure of your *artifacts* — your layers, your tables — rather than the structure of your *business*.** Layers and tables are how the software is built today; they are not how the business changes. DDD's entire contribution is to redirect your attention from "what does my data model look like" to "what does my business actually *do*, and where do its rules and language naturally cluster." That redirection is the whole game.

## 3. The right way: decompose by business capability and subdomain

The right way is to split by **business capability** — the things the business *does*, the verbs, the outcomes it is responsible for producing. ShopFast's capabilities are: maintain a catalog of products, take orders, take payment, track inventory, fulfil and ship orders, and notify customers. Each is a self-contained responsibility with its own rules, its own data, and (crucially) its own *language*. A service aligned to a capability owns a use case from start to finish: it has the data it needs locally, it can complete its core transaction without asking anyone, and it can change its internal rules without negotiating with another team.

In DDD terms, a business capability corresponds to a **subdomain** — a distinct area of the business problem — and the software model of a subdomain lives inside a **bounded context**, which is the thing that becomes your service. We will define those terms precisely in the next section. For now, the heuristic to carry: **a good service owns a capability end to end.** Ask of any candidate service, "can it complete its primary job using only its own data and an event or two?" If yes, the boundary is probably right. If it cannot take a step without three synchronous calls to fetch data it does not own, the boundary is wrong, and you have re-discovered the distributed monolith.

There is one more layer of sophistication that separates a senior's boundaries from a junior's, and it is the distinction between **core, supporting, and generic** subdomains. Your **core domain** is the thing that makes your business special — for ShopFast, perhaps the ordering and recommendation experience that wins customers. That is where you invest your best engineers, your richest models, your custom code. **Supporting subdomains** are necessary but not differentiating — inventory tracking, say — where solid but unremarkable code is fine. **Generic subdomains** are solved problems you should buy, not build: payment processing (use Stripe), notifications (use a provider), authentication (use an identity provider). This classification tells you *where to spend* and *where a boundary should be a thin anti-corruption layer over a vendor* rather than a hand-crafted domain model. A junior models everything richly; a senior models the core richly and wraps the generic with the thinnest possible adapter. We will see exactly that when we put an anti-corruption layer around Payment.

## 4. The DDD toolkit, in plain language

DDD has a reputation for being dense and abstract — a wall of patterns with intimidating names. It is not, once you strip it to the four ideas that actually matter for drawing service boundaries. Let me define each in a sentence, then go a level deeper.

**Ubiquitous language** is the agreement that everyone — engineers, product managers, domain experts — uses *the same words for the same concepts*, and that those words appear directly in the code. If the warehouse staff say "pick" and "pack" and "dispatch," then your code has `pick()`, `pack()`, and `dispatch()`, not `processStage1()`. The payoff is that the model and the conversation never drift apart; a bug report in business language maps directly onto a method. The deeper payoff, and the one that matters for boundaries, is the next idea: language is the *detector* for where a boundary belongs.

**A bounded context is the boundary within which a particular model and its language are consistent and unambiguous.** Inside a bounded context, every term means exactly one thing. The magic — and the reason it maps onto a service — is that *the same word can mean different things in different contexts, and that is fine.* "Customer" in the Support context is a person with tickets and a satisfaction score; "Customer" in the Billing context is an account with a payment method and an invoice history; "Customer" in the Marketing context is a segment with a lifetime-value estimate. These are three different models of the same real-world person, and trying to force them into one shared "Customer" object is precisely how you build a coupled mess. The bounded context says: let each context have its own model, its own meaning of the word, and translate at the boundary. **A bounded context is the unit that maps to a microservice.** When people ask "how big should a service be," the honest answer is "one bounded context," and the rest of this post is about how to find them.

**An aggregate is the consistency boundary *inside* a context** — a cluster of objects that must change together and stay valid as a unit, with one object (the *aggregate root*) as the only entry point. An Order is an aggregate: the order header and its line items must be consistent (the total must equal the sum of the lines; you cannot have an order with zero items in PLACED state), so you load, modify, and save them as one unit, in one transaction, through the Order root. The aggregate is the single most underrated concept in this whole post, because **the aggregate is your transaction boundary**, and the transaction boundary is what tells you whether two pieces of data can live in different services. If two things must be updated atomically — in the same transaction, with an invariant spanning them — they belong in the same aggregate, which means the same context, which means the same service. We will hit this hard in the stress-test section, because "an invariant that spans two services" is the most expensive boundary mistake there is.

**Context mapping is the catalogue of ways two bounded contexts can relate**, and it is how you reason about the lines *between* services rather than the services themselves. The four you actually use are: **shared kernel** (two contexts co-own a small shared model, like a `Money` value type — powerful but coupling, use sparingly); **customer/supplier** (one context is downstream of another and they negotiate the upstream's contract, the downstream being the "customer" whose needs the "supplier" agrees to serve); **conformist** (the downstream simply accepts the upstream's model as-is, no translation, because it has no leverage to negotiate — common when the upstream is a big platform team or a third party); and the most important one, **anti-corruption layer** (the downstream builds a translation layer that maps the upstream's model into its own clean language, so the foreign model *never leaks in*). The anti-corruption layer is your defensive weapon: when you integrate with a messy legacy system or an external vendor whose model you do not control, you put an ACL at the boundary so your beautiful domain stays beautiful. We will build one for ShopFast's payment integration.

## 5. Finding boundaries in practice: EventStorming and four signals

Theory is lovely; the practical question is *how do you actually discover the contexts in a real system?* The single best technique is **EventStorming**, invented by Alberto Brandolini — a workshop where you get the domain experts and engineers in a room (physical or virtual) and cover a wall with sticky notes representing **domain events**: things that have happened in the business, phrased in the past tense. "Order placed." "Payment captured." "Stock reserved." "Item shipped." "Customer notified." You do not start with data models or services; you start with *events*, because events are the language of the business and they are model-agnostic.

![A four-step pipeline showing an EventStorming pass moving from posting domain events to sorting them on a timeline to clustering by language to naming the candidate bounded contexts](/imgs/blogs/service-boundaries-with-domain-driven-design-3.webp)

The pass goes in four moves, shown above. **One:** everyone posts domain events as orange stickies, fast and messy, no filtering — you want hundreds. **Two:** sort them left to right on a timeline, the natural order in which they happen, which reveals the end-to-end flow of the business. **Three:** cluster them — and this is where the magic happens — by looking for two things. Look for *pivot events* (an event that clearly ends one phase and begins another, like "Order placed" handing off from shopping to fulfilment) and look for *where the language changes*. When the words on the stickies shift — when you move from "cart, item, price, discount" to "parcel, weight, carrier, tracking number" — you have crossed a boundary between contexts, because a change in vocabulary is a change in model. **Four:** draw a line around each cluster and name it. Those names are your candidate bounded contexts, and therefore your candidate services.

Beyond the workshop, four concrete signals tell you a boundary is in the right place, and you should sanity-check every candidate against all four.

The first and strongest is **where the language changes.** I cannot overstate this. When the same word means two different things — "Order" as a billable cart versus "Order" as a physical shipment, "Customer" as a ticket-haver versus an invoice-payer — that ambiguity is a boundary screaming to be drawn. The famous example is "customer means different things to billing versus support": billing's customer has a credit limit and a payment method; support's customer has a contact history and a sentiment. If you build one Customer service for both, you couple two teams who genuinely need different things, and every change fights the other context's needs. Two meanings, two contexts.

The second signal is **data cohesion**: which data is accessed together, almost always, in the same operations? Data that is read and written as a unit wants to live in one service, because separating it creates exactly the chatty cross-service reads that define the distributed monolith. If you find yourself saying "well, the Order service will need to call the Pricing service on every single order operation," that is data cohesion telling you Pricing and Ordering might be one context.

The third signal is **rate of change**: things that change together, for the same business reason, belong together; things that change for *different* reasons and on *different* cadences belong apart. This is the Single Responsibility Principle scaled up to services — a service should have one reason to change. The promotions logic changes weekly with marketing campaigns; the shipping-rate logic changes quarterly with carrier contracts. Different reasons, different cadences, different contexts. Putting them in one service means the fast-changing promotions code keeps forcing redeploys of the stable shipping code.

The fourth signal is **team ownership**, the Conway's Law lever. A boundary you can hand to one team to own end to end — with the autonomy to change its internals, its database, its deploy schedule without coordinating — is a boundary that will actually let you move fast. If a candidate service would have to be co-owned by three teams, it is probably either too big (split it) or drawn across a natural team seam (move the line). The boundaries that survive contact with reality are the ones that also work as ownership boundaries.

## 6. The ShopFast contexts: running EventStorming on a real example

Let us actually do it. ShopFast sells physical goods online. We get the warehouse lead, the customer-support lead, two product managers, and four engineers in a room, and we EventStorm. After an hour the wall has a few hundred events, and when we sort and cluster them, six clusters emerge cleanly — the language genuinely changes between each.

![A branching service-topology graph showing the six ShopFast bounded contexts connected by named events and contracts rather than a shared database](/imgs/blogs/service-boundaries-with-domain-driven-design-4.webp)

The six contexts, as the figure shows, are **Catalog** (the source of truth for what products exist, their descriptions, categories, images — language: product, SKU, category, variant), **Ordering** (taking an order, the cart-to-confirmed-order journey — language: cart, line item, discount, order total, place), **Payment** (charging the customer, mostly a thin wrapper over an external PSP — language: charge, authorization, capture, refund), **Inventory** (tracking stock levels and reservations — language: stock-on-hand, reserved, available, replenish), **Shipping** (fulfilling and delivering orders — language: parcel, weight, carrier, label, tracking), and **Notifications** (emails and SMS — language: template, channel, recipient, send). Notice how *cleanly the vocabulary partitions*: nobody in the Shipping cluster talks about discounts, and nobody in the Ordering cluster talks about carrier labels. That linguistic separation is the boundary made visible.

Notice also how they *integrate*: not through a shared database, but through **named events and explicit contracts**. Ordering, when an order is placed, emits an `OrderPlaced` event that Shipping consumes; Payment emits `PaymentCaptured` that both Ordering and Notifications consume; Inventory exposes a "reserve stock" contract that Ordering calls. The edges in the figure are events and contracts, not foreign-key joins. This is the difference between integration (services collaborating across explicit, versioned contracts) and coupling (services reaching into each other's data). The whole point of the boundary is to convert the second into the first.

Now I want to be honest about the judgment calls, because EventStorming does not hand you the answer — it gives you candidates, and you size them. Two real questions came up for ShopFast. First: is **Inventory** part of **Ordering**, or its own context? It depends on whether stock reservation is part of the order's consistency requirement (more on this in the stress-test) and on rate of change — ShopFast decided Inventory changes for warehouse reasons (replenishment, audits, multiple warehouses) on a different cadence than ordering, and a separate warehouse team would own it, so: separate context. Second: are **Notifications** really their own service, or just a library? Notifications has almost no business logic — it is close to a generic subdomain. ShopFast made it a service anyway, but a *thin* one, because many contexts need to send messages and centralizing it avoids every team reinventing email templating. These are exactly the sizing judgments the next section is about.

## 7. The same word, two models: "Order" across contexts

The deepest lesson in this whole post hides inside one ordinary noun, so let me slow down on it, because once you *feel* this, bounded contexts stop being abstract.

![A before-and-after comparison showing the Order aggregate as a billable cart in the Ordering context versus a physical shipment in the Shipping context](/imgs/blogs/service-boundaries-with-domain-driven-design-5.webp)

"Order" exists in both the Ordering context and the Shipping context. The junior instinct — and the entity-service instinct — is "great, one Order, one Order service, both use it." But look at what Order actually *is* in each, shown above. In **Ordering**, an Order is a *billable cart*: it has line items with prices, a customer, a payment reference, a status like PLACED, and an invariant that the total equals the sum of the lines and there is at least one item. The Ordering context cares deeply about money and items and discounts. In **Shipping**, an Order is a *physical shipment*: it has parcels, weights, a destination address, a carrier, a tracking number, and a status like IN_TRANSIT. The Shipping context does not care about prices *at all* — it never reads the order total, because that is not its job. It has fields Ordering lacks (weight, carrier) and lacks fields Ordering has (price, discount).

These are *two different models of the same real-world thing*, and that is not a problem to be eliminated — it is the natural, correct state of affairs, and bounded contexts are precisely the mechanism that lets both coexist without coupling. If you forced a single shared Order object across both services, every change to Shipping's carrier logic would risk breaking Ordering's billing, the object would accumulate every field both contexts ever needed (a bloated "God object"), and the two teams would be in each other's way forever. Instead, each context has its *own* Order model, in its *own* database, and they synchronize through an event: when Ordering finishes, it publishes `OrderPlaced` carrying only the data Shipping needs (the items, quantities, and address — *not* the prices), and Shipping constructs its *own* Order from that. The same identity (the order ID) links them; the models are independent.

This is the resolution to the question that confuses every newcomer: "but if I split User and Order into different services, won't I duplicate data?" Yes — and that duplication is *correct*. Each context keeps the *projection* of shared data that it needs, in the shape it needs, and they stay loosely synchronized through events. Shipping does not call Ordering to get the address on every operation; it received the address in the `OrderPlaced` event and stored its own copy. A little duplication in exchange for decoupling is one of the best trades in distributed systems, and getting comfortable with it is a rite of passage from junior to senior.

## 8. Aggregates: the transaction boundary in code

We have placed the boundaries *between* services. Now we go *inside* one and find the transaction boundary, because this is where boundary-thinking becomes code you can copy.

Recall the definition: an aggregate is a cluster of objects that must stay consistent as a unit, with one root as the only entry point, and it is the boundary of a single transaction. The rule that makes aggregates useful is **invariants live inside the aggregate, enforced by the root.** An invariant is a business rule that must *always* be true: an order's total equals the sum of its lines; a confirmed order has at least one item; you cannot remove the last item from a placed order. The aggregate root is the gatekeeper that guarantees no operation ever leaves the aggregate in a state that violates an invariant.

Here is ShopFast's Order aggregate root, written so the invariants are enforced in code rather than scattered across the codebase. Notice that there is no public way to mutate the line items except through methods on the root, and every method re-checks the invariants:

```python
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4


class OrderStatus(Enum):
    DRAFT = "DRAFT"
    PLACED = "PLACED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("cannot add money of different currencies")
        return Money(self.amount + other.amount, self.currency)


@dataclass(frozen=True)
class LineItem:
    sku: str
    quantity: int
    unit_price: Money

    @property
    def subtotal(self) -> Money:
        return Money(self.unit_price.amount * self.quantity, self.unit_price.currency)


class Order:
    """Aggregate root. The ONLY way to touch line items is through these methods,
    so the invariants below can never be violated by outside code."""

    def __init__(self, customer_id: UUID):
        self.id: UUID = uuid4()
        self.customer_id = customer_id
        self.status = OrderStatus.DRAFT
        self._lines: list[LineItem] = []

    # --- invariant-preserving behavior, not anemic setters ---

    def add_item(self, sku: str, quantity: int, unit_price: Money) -> None:
        if self.status != OrderStatus.DRAFT:
            raise ValueError("can only add items to a DRAFT order")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._lines.append(LineItem(sku, quantity, unit_price))

    def place(self) -> "OrderPlaced":
        # invariant: a placed order has at least one item and a positive total
        if not self._lines:
            raise ValueError("cannot place an order with no items")
        if self.total.amount <= 0:
            raise ValueError("order total must be positive")
        self.status = OrderStatus.PLACED
        # the aggregate emits the event the rest of the system reacts to
        return OrderPlaced(order_id=self.id,
                           customer_id=self.customer_id,
                           items=[(l.sku, l.quantity) for l in self._lines])

    @property
    def total(self) -> Money:
        return sum((l.subtotal for l in self._lines),
                   start=Money(Decimal("0")))


@dataclass(frozen=True)
class OrderPlaced:
    order_id: UUID
    customer_id: UUID
    items: list[tuple[str, int]]  # only what downstream needs: NOT the prices
```

Three things in this code are the whole point. First, **there are no public setters for line items** — you cannot do `order.lines.append(...)` from outside; you go through `add_item`, which checks the rules. This is the difference between a rich domain model and the *anemic* model the entity-service trap produces. Second, **`place()` enforces the invariants and emits an event** — the aggregate is the source of truth for what is valid, and the event carries only what downstream contexts need (note `OrderPlaced` does *not* include prices, because Shipping does not need them — that is the bounded-context discipline from the last section, encoded). Third, **the entire aggregate is loaded, mutated, and saved in one database transaction**, which is possible precisely because it all lives in one service's one database.

Now the load-bearing rule for boundaries: **reference other aggregates by ID, never by object.** Notice `Order` holds `customer_id`, not a `Customer` object. The Customer lives in another context; the Order only knows its identity. This is what *keeps the boundary clean*: if the Order aggregate held a full Customer object, you would have just dragged the Customer context's model into the Ordering context, and you would need a cross-service call to load it every time. Holding only the ID means the aggregate stays self-contained and the transaction stays local. The DDD rule "reference across aggregates by identity" is, at the service level, the rule "do not pull another service's model into yours."

#### Worked example: an invariant that must stay in one transaction

Here is the kind of concrete reasoning that should drive every boundary decision. ShopFast has a rule: **an order's total must always equal the sum of its line items, and a placed order must have at least one item.** That is an invariant spanning the order header and the line items.

Question: can the order header and the line items live in *different* services? Walk through it. If `Order` lives in an Order-Header service and `LineItem` lives in a Line-Item service, then placing an order requires updating both atomically — add the lines *and* flip the header to PLACED *and* verify the total matches — across two databases. There is no shared transaction across two databases, so you would need a two-phase commit (slow, blocking, operationally hated) or a saga (eventually consistent, meaning there is a window where the header says PLACED but the lines have not committed, *violating the invariant*). Either way you have spent enormous complexity to support a split that buys you *nothing*, because the header and lines always change together, are always read together, and have no independent rate of change. The invariant is the proof: **header and lines belong in the same aggregate, the same context, the same service, the same transaction.** The cost of getting this wrong is not theoretical — it is a permanent tax of distributed coordination on your single most frequent write path. We will return to the inverse — what to do when an invariant *genuinely* must span two services — in the stress-test.

## 9. The anti-corruption layer: defending your model at a boundary

Some boundaries face inward (toward another team's clean context) and some face outward (toward a legacy system or a third-party vendor whose model you do not control and would not choose). The outward-facing ones are dangerous, because a foreign model — Stripe's payment intents, a legacy ERP's product schema, a shipping carrier's idiosyncratic API — will leak into your code and corrupt your clean domain language if you let it. The defense is the **anti-corruption layer** (ACL): a translation layer at the boundary that maps the upstream's vocabulary onto yours, so nothing foreign crosses the line.

![A layered stack showing an anti-corruption layer sitting between a clean domain model and an external payment provider SDK translating the foreign model](/imgs/blogs/service-boundaries-with-domain-driven-design-7.webp)

The figure shows the layering. Your **domain model** speaks your ubiquitous language (`Charge`, `Refund`, `PaymentResult`). Behind a **port** — an interface *you* define in *your* terms — sits the **ACL translator**, which maps your domain calls to and from the vendor's model and back, wrapping the **vendor SDK client** and ultimately the **external PSP API**. The point of the port is dependency inversion: your domain depends on the interface *you* own, not on Stripe's SDK. The point of the ACL is that the words "PaymentIntent," "client_secret," and "charges.create" — Stripe's vocabulary — appear *only* inside the translator, never in your domain code.

Here is the port your Payment context defines, in its own language:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class PaymentOutcome(Enum):
    CAPTURED = "CAPTURED"
    DECLINED = "DECLINED"
    PENDING = "PENDING"


@dataclass(frozen=True)
class ChargeRequest:
    order_id: str
    amount: Decimal
    currency: str
    customer_ref: str


@dataclass(frozen=True)
class ChargeResult:
    outcome: PaymentOutcome
    provider_charge_id: str
    decline_reason: str | None = None


class PaymentGateway(ABC):
    """The port: the Payment context's OWN language. Nothing here mentions
    Stripe, PaymentIntents, or client secrets."""

    @abstractmethod
    def charge(self, request: ChargeRequest) -> ChargeResult: ...
```

And here is the anti-corruption layer that implements it against a specific vendor. Every line of vendor-specific translation is quarantined in this class:

```python
import stripe


class StripePaymentGateway(PaymentGateway):
    """Anti-corruption layer. Stripe's model is translated into ours here and
    NOWHERE else. If we swap Stripe for Adyen, only this class changes."""

    def charge(self, request: ChargeRequest) -> ChargeResult:
        # translate OUR model -> THEIR model (units, naming, structure)
        intent = stripe.PaymentIntent.create(
            amount=int(request.amount * 100),       # we use dollars; Stripe wants cents
            currency=request.currency.lower(),      # they want lowercase
            customer=request.customer_ref,
            metadata={"order_id": request.order_id},
            confirm=True,
        )

        # translate THEIR model -> OUR model (status strings, error shapes)
        return self._to_domain(intent)

    def _to_domain(self, intent) -> ChargeResult:
        mapping = {
            "succeeded": PaymentOutcome.CAPTURED,
            "requires_payment_method": PaymentOutcome.DECLINED,
            "processing": PaymentOutcome.PENDING,
        }
        outcome = mapping.get(intent.status, PaymentOutcome.PENDING)
        reason = None
        if outcome is PaymentOutcome.DECLINED and intent.last_payment_error:
            reason = intent.last_payment_error.get("decline_code")
        return ChargeResult(
            outcome=outcome,
            provider_charge_id=intent.id,
            decline_reason=reason,
        )
```

Look at what this buys you. The rest of the Payment service — and certainly the Ordering service that consumes it — only ever sees `ChargeResult` with a `PaymentOutcome`. They never see the string `"requires_payment_method"`, never deal in cents, never import the Stripe SDK. When ShopFast inevitably wants a second payment provider for a new region, or wants to swap Stripe for Adyen, you write one new ACL implementing the same `PaymentGateway` port, and *nothing else changes*. The boundary held. This is exactly why a generic subdomain like payment should be a *thin* context wrapped in an ACL rather than a richly modeled one: you do not own the model, so you defend against it instead of embracing it. This same pattern is how you integrate with a strangled legacy monolith during a migration — the new services put an ACL in front of the old system's schema — which is the subject of [the strangler-fig migration post](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices).

## 10. The trade-off matrix: scoring the three decomposition axes

A senior never recommends an approach without naming its cost, so let us put the three decomposition strategies side by side on the axes that actually decide whether a service can be operated independently. This is the decision matrix the kit demands, and it is the single artifact I would put on the wall in a boundary design review.

![A decision matrix scoring technical-layer, entity, and business-capability decomposition across coupling, transaction locality, team ownership, and chattiness](/imgs/blogs/service-boundaries-with-domain-driven-design-6.webp)

The matrix above, and the table below, score the same four properties. **Coupling** is how much a change in one service forces a change in another. **Transaction locality** is whether your core operations can complete in a single local transaction or must span services. **Team ownership** is whether one team can own a service end to end. **Chattiness** is how many cross-service calls a typical operation triggers.

| Property | Split by technical layer | Split by entity (noun) | Split by business capability |
| --- | --- | --- | --- |
| Coupling | Tight — every feature spans all layers | Medium — operations span nouns | Loose — a capability is self-contained |
| Transaction locality | Poor — a write crosses 3 layer-services | Often distributed across noun-services | Local — the core write is one transaction |
| Team ownership | Shared — layer teams co-own every feature | By noun — but ops cross owners | By capability — one team, end to end |
| Chattiness | Very high — vertical slices fan out | High — data scattered across nouns | Low — data is local to the capability |
| When it might win | Almost never for business systems | Tiny CRUD apps with truly isolated nouns | The default for any non-trivial system |
| The hidden cost | A coordination meeting per feature | The God-entity bottleneck (User svc) | Requires real domain understanding up front |

Read the bottom two rows carefully, because they are the honest part. Capability decomposition is not free: its cost is that **it demands real domain understanding before you can draw the lines**, which is exactly why teams reach for the easier layer/entity splits — those you can do from the data model alone, without talking to a domain expert. That is the trap. The cheap-to-draw boundaries are the expensive-to-operate ones, and the expensive-to-draw boundary (capability) is the cheap-to-operate one. You pay either at design time, with EventStorming and conversations, or forever at runtime, with sagas and chatty calls and shared databases. Pay at design time. It is vastly cheaper, and a wrong boundary, once services are live and other teams depend on the contracts, is brutal to move.

There is one honest case for entity-ish splitting: a *truly* simple CRUD system where the nouns really are independent and operations really do stay within a single noun. If you are building a small admin tool where "manage tags" and "manage categories" never interact, entity services are fine. But the moment operations start spanning nouns — and in any real business they do, immediately — the entity split decays into a distributed monolith. Treat it as the exception that proves the rule.

## 11. Sizing services: avoid nano-services, and the "deploy together" heuristic

Once you have the right *axis* (capability), the next mistake is the wrong *granularity*. Microservice enthusiasm pushes people toward ever-smaller services — and there is a failure mode at the small end that is just as bad as the distributed monolith at the large end: **nano-services**, or **entity services** so fine-grained that each does almost nothing and the real work happens in the chatter between them.

The symptom of nano-services is the same chattiness and distributed-transaction pain as the entity trap, because they *are* the entity trap taken to its logical extreme — a service per noun, then a service per *field*. A "PriceService," an "AddressService," a "DiscountService" as separate deployables means that computing an order total — a single conceptual operation — fans out to three network calls, and any consistency requirement across them becomes a distributed transaction. The operational cost is real and measurable: each service is a deploy pipeline, a set of dashboards, an on-call rotation, a network hop with its own failure modes, and a contract that must be versioned. A fleet of 200 nano-services where 30 capability-services would do is a fleet that drowns its team in operational overhead while delivering *worse* latency and reliability. (Monzo famously runs 1,500+ services, but they are a bank with a deeply specialized platform and tooling investment to match — it is an existence proof that it *can* work at extreme scale, not a recommendation for your 12-person team.)

The single best heuristic for right-sizing, and the one I want you to carry out of this post, is this: **two services that always deploy together should be one service.** If every time you ship a change to service A you must also ship a coordinated change to service B — because they share a contract that always co-evolves, or a piece of logic neither can move without the other — then the boundary between them is fake. You are paying the full cost of a network boundary (latency, partial failure, contract versioning) and getting none of the benefit (independent deployability), which was the *entire reason* to split. Merge them. The corollary heuristic: **a service should be able to be rewritten by one team in one or two sprints.** If it is so big that no one understands it, split it; if it is so small that it has no behavior of its own, merge it. The right size is "one business capability, owned by one team, deployable on its own schedule."

Here, to make the boundary contract concrete, is what a well-sized capability service's interface looks like — note that it exposes *behavior* (capabilities), not CRUD on a table, and references other contexts only by ID:

```protobuf
// Ordering bounded context: a capability interface, NOT a table CRUD.
// It exposes what the business DOES, and references Catalog/Customer by ID.
syntax = "proto3";
package shopfast.ordering.v1;

service OrderingService {
  // capabilities (verbs the business performs), not getId/setId
  rpc PlaceOrder(PlaceOrderRequest) returns (PlaceOrderResponse);
  rpc CancelOrder(CancelOrderRequest) returns (CancelOrderResponse);
  rpc GetOrderSummary(GetOrderSummaryRequest) returns (OrderSummary);
}

message PlaceOrderRequest {
  string customer_id = 1;            // a REFERENCE into Customer context, by ID
  repeated RequestedItem items = 2;  // sku + qty; Ordering looks up prices itself
}

message RequestedItem {
  string sku = 1;                    // a REFERENCE into Catalog context, by ID
  int32 quantity = 2;
}

message PlaceOrderResponse {
  string order_id = 1;
  string status = 2;                 // PLACED, REJECTED
  Money total = 3;
}
```

A junior reviewing this might ask "where's `CreateOrder`, `UpdateOrder`, `DeleteOrder`?" The answer is that those are CRUD on a row, and this is not a row, it is a capability. `PlaceOrder` and `CancelOrder` are *what the business does*; they carry invariants and emit events. The interface is the boundary, and a capability-shaped interface is the tell that you sized the service right.

## 12. The optimization angle: measuring chattiness and fixing a bad boundary

Now the production-grade angle the kit demands: where is the real bottleneck a bad boundary creates, how do you make it better, and how do you *measure* the win? The bottleneck is almost always **synchronous cross-service call fan-out on a hot path**, and the metric that exposes it is *cross-service calls per business operation* plus its effect on *p99 latency*.

![A before-and-after comparison showing checkout dropping from seven cross-service calls to a single local transaction after the boundary is moved](/imgs/blogs/service-boundaries-with-domain-driven-design-8.webp)

Here is the mechanism that makes chattiness so damaging on the p99, and why it is worse than the average suggests. When a single operation makes N independent synchronous calls, its latency is roughly the *maximum* of the N call latencies (if parallel) or their *sum* (if sequential), and either way the tail compounds: even if each downstream service has a 1% chance of a slow response, the probability that *at least one* of seven calls is slow is roughly 1 minus 0.99 to the seventh power, about 6.8% — so your operation's p99 is dragged up by the worst of seven dice rolls, not one. This is the "tail at scale" effect, and it is why reducing fan-out from seven calls to one is not a 7× improvement on average but a far larger improvement at the tail where your SLO actually lives. For the deeper mechanics of why distributed calls fail in correlated and surprising ways, see [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies).

#### Worked example: counting cross-service calls before and after a boundary fix

ShopFast's checkout, under the entity split, made these synchronous calls per "place order" button press: three reads to the User service (profile, default address, saved payment method), one read to the Product service per distinct item to get current prices (say two items, so two calls), one write to the Order service, and one write coordinated as part of a two-phase commit with Inventory. That is roughly **seven cross-service calls** on the critical path. With each cross-service call adding around 18ms at p99 (a modest gRPC round trip including serialization and a downstream DB hit), and several of them sequential because of data dependencies (you need the address before you can compute shipping, you need prices before you can compute the total), the checkout p99 landed around **140ms** — and that is *before* any downstream service has a bad day.

Now we redraw the boundary so that **Ordering owns the checkout transaction end to end**: it stores the customer's chosen address and a payment-method token *as part of the order* (received at the start of checkout, not fetched per-call), it owns the price snapshot taken when items were added to the cart (so it does not re-fetch prices from Catalog at place-time — it already has them), and it commits the order in a single local database transaction. Stock reservation and payment are handled *asynchronously after* the order is placed, via events and a saga, rather than synchronously inside the place-order call. The result, shown in the figure: **zero synchronous cross-service calls on the place-order path, one local DB transaction, and a checkout p99 around 22ms** — roughly a 6× improvement at the tail, and a checkout that no longer fails just because the User service is having a slow minute. The win is measurable: cross-service calls per operation dropped from 7 to 0, p99 from ~140ms to ~22ms, and the operation's availability went from the *product* of seven services' availabilities (if any of seven 99.9% services is down, you fail ~0.7% of the time) to the availability of one. The boundary move *is* the optimization; no amount of faster gRPC would have gotten you there.

The general technique this illustrates: when an operation is chatty, the fix is rarely "make the calls faster" and almost always "**move the boundary so the data the operation needs is local**, and push the rest of the work off the synchronous path onto events." A boundary that makes the hot path local is the highest-leverage performance optimization in microservices, and it is invisible to anyone profiling individual service latency — you have to look at the call graph of the *operation*. This pushes a lot of the system toward eventual consistency, which is its own discipline; for when that is and is not safe, see [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

## 13. Context mapping: reasoning about the lines between services

We have spent most of the post on the contexts themselves. The lines *between* them deserve their own treatment, because how two contexts relate determines who absorbs the cost when one of them changes — and that is a strategic decision, not just a technical one.

![A graph showing the Ordering context relating to its neighbours through shared kernel, customer-supplier, anti-corruption layer, and conformist relationships](/imgs/blogs/service-boundaries-with-domain-driven-design-9.webp)

The figure maps ShopFast's Ordering context to four neighbours using four different relationship patterns, and the choice in each case is deliberate. With a small shared `Money` value type, Ordering uses a **shared kernel** — both contexts co-own one tiny, stable model, accepting tight coupling on that one type because the alternative (every context defining its own Money and translating constantly) is worse for something so fundamental and so unlikely to change. With Inventory, Ordering is in a **customer/supplier** relationship: Ordering is the downstream customer whose needs Inventory (the supplier) agrees to serve, so when Ordering needs a new field in the "stock reserved" contract, the two teams negotiate it and Inventory's roadmap accommodates it. With Payment's external PSP, Ordering sits behind an **anti-corruption layer** — the defensive pattern from section 9 — because the upstream model is foreign and out of its control. And with Catalog, Ordering is a **conformist**: it simply accepts Catalog's product model as given, because Catalog is the authoritative source of product truth and Ordering has no leverage or need to reshape it, so it takes the model as-is rather than building a translation layer it does not need.

The strategic insight is **who absorbs change**, which the relationship encodes precisely. In a conformist relationship, the *downstream* absorbs every upstream change — fine when the upstream is stable and authoritative, dangerous when it is volatile. In a customer/supplier relationship, the *upstream* agrees to absorb the downstream's needs — appropriate between two of your own teams who can negotiate. With an ACL, *neither leaks into the other* — the downstream pays a translation cost to stay insulated, which is exactly what you want against a third party or a legacy system. With a shared kernel, *both* must agree on every change — the highest coupling, justified only for a tiny, stable, genuinely shared concept. When you draw a context map, you are deciding, for every boundary, who pays when it moves. A senior makes that choice consciously; a junior lets it happen by accident and then wonders why one team's release keeps breaking another's.

## 14. Stress-testing the design: what breaks, and when

The kit asks for a problem-solving narrative that stress-tests the design, and the most important stress test for boundaries is the one I flagged earlier: **what happens when an aggregate's invariant genuinely spans two services?**

Set it up concretely. ShopFast wants this rule: *you can never sell more units than you have in stock.* That is an invariant relating Ordering (which creates demand by placing orders) and Inventory (which holds the stock count). If both lived in one aggregate, a database transaction would enforce it trivially: decrement stock and create the order line in one atomic commit, and the constraint `stock >= 0` is checked by the database. But we deliberately put Ordering and Inventory in *separate* contexts — separate services, separate databases — because they change for different reasons and are owned by different teams. So the invariant now spans a network boundary, and there is no transaction that can hold both sides. What do we do?

There are exactly three honest answers, and choosing between them is the senior-grade judgment.

**Answer one: it was the wrong boundary — merge them.** If "never oversell" is a *hard, money-critical invariant that must never be violated even momentarily*, and if Ordering and Inventory turn out to always change and deploy together anyway, then maybe they are really one context, and the right move is to merge them so the invariant is enforced by a local transaction. Always consider this first. The existence of a hard invariant spanning your boundary is *evidence the boundary might be wrong*, and you should take that evidence seriously rather than immediately reaching for a saga. The "deploy together → one service" heuristic applies.

**Answer two: relax the invariant to a reservation with a window.** This is usually the right answer for e-commerce, and it reflects how the *real business* works. Inventory does not actually enforce "never oversell" as an instantaneous global invariant — it enforces "reserve stock for an order, hold it for 15 minutes, and release it if the order is not completed." Stock reservation is a *business process with a time window*, not an instantaneous constraint. So Ordering, at place time, sends a "reserve" command to Inventory; Inventory atomically (in *its* local transaction) decrements available stock and creates a reservation, or rejects if insufficient; and if payment later fails, a compensating "release reservation" event returns the stock. This is a [saga](/blog/software-development/microservices/the-saga-pattern-in-practice), and it works precisely because the business invariant was never truly instantaneous — it tolerates a reservation window. The key realization: **most invariants people think must be instant are actually time-windowed business processes, and modeling them honestly dissolves the cross-service-transaction problem.** You are not weakening correctness; you are matching the model to how the business genuinely operates.

**Answer three: accept eventual consistency with detection and correction.** For invariants that are *not* money-critical — say, "a customer's loyalty points should equal the sum of their order rewards" spanning Ordering and Loyalty — you let the two services update independently and run a reconciliation job that detects and corrects drift. You accept that for short windows the two sides may disagree, because the cost of inconsistency is low and the cost of distributed coordination is high. This is the right trade when the invariant is a "should" rather than a "must." The discipline here — idempotent updates, reconciliation, detecting violations — is its own topic in [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice).

The meta-lesson across all three: **a cross-service invariant is a design smell that demands a decision, not a thing you paper over.** Either the boundary is wrong (merge), or the invariant is actually a time-windowed process (saga), or it is a soft "should" (reconcile). What you must *never* do is the fourth, lazy option — let two services share a database table so a database constraint can enforce the invariant across both. That "solves" the invariant by destroying the boundary, and you are back to the distributed monolith. Resist it every time.

Two more quick stress tests, because boundaries must survive real failure. **What breaks at 10× traffic?** A capability-aligned boundary scales gracefully because each service can be scaled independently to match its own load — Catalog reads scale separately from Ordering writes — whereas the God-entity (User service) under the entity split becomes a bottleneck that *every* request hits, so it saturates first and takes everything down. The boundary determines your scaling unit. **What breaks when a downstream service is down?** With the chatty entity split, checkout calls the User service synchronously, so when User is down, checkout is down — the failure propagates because the boundary made the dependency synchronous and on the hot path. With the capability split, checkout completes locally and only the *asynchronous* post-order steps (shipping, notification) are delayed, degrading gracefully rather than failing. The boundary determines your blast radius. Both of these — independent scaling and contained failure — are supposed benefits of microservices, and *both are forfeited by a bad boundary.* The boundary is not one factor among many; it is the precondition for every other microservices benefit.

## 15. Case studies: how the best teams draw their lines

Theory and a fictional store are useful, but the patterns earn their authority from real systems at scale. Three cases, each teaching one lesson.

### Amazon: capability-aligned teams, "you build it, you run it"

Amazon's famous reorganization in the early 2000s — the "two-pizza teams" mandate, where every team was small enough to be fed by two pizzas and owned a service end to end — is the canonical example of boundaries drawn by capability and aligned to teams. The mandate, attributed to a 2002 internal memo, required teams to expose their functionality only through service interfaces over the network, with no direct database sharing and no back doors. This is Conway's Law deployed *intentionally*: by making teams small, autonomous, and aligned to a business capability, Amazon ensured the service boundaries matched the team boundaries, so a team could change its service without coordinating with others. The lesson for boundaries: **the unit of decomposition and the unit of ownership should be the same**, and "you build it, you run it" forces the boundary to be one a single team can actually own. The interface-only rule is the same discipline as our protobuf example — no shared database, integration through contracts. Amazon did not split by entity; they split by what a small team could own and operate.

### Uber: DOMA, domain-oriented architecture

By the late 2010s Uber had grown to thousands of microservices, and they hit the nano-service wall from section 11 — so many fine-grained services that the chattiness, the operational overhead, and the difficulty of understanding the system became the bottleneck. Their response, which they wrote up publicly as **DOMA (Domain-Oriented Microservice Architecture)**, was essentially to apply DDD retroactively: group the sprawl of microservices into **domains** (their word for a collection of related services serving one business capability — close to a bounded context), put a clear gateway interface in front of each domain, and use *layers* to control which domains can call which (so dependencies flow in one direction, avoiding the tangled call graph). DOMA also explicitly uses the **anti-corruption layer** concept — a translation layer at a domain's edge so internal changes do not leak out. The lesson: **you can over-decompose**, and the correction is to re-aggregate fine-grained services into capability-aligned domains with explicit boundaries and a clean dependency direction. Uber's journey is a real-world demonstration that the right granularity is "business capability," and that going finer than that is a mistake you eventually have to undo.

### The classic distributed monolith: split by entity, reaped by sagas

The third case is not one company but a pattern repeated at hundreds, including the ShopFast-like team from the intro, and it is worth stating as a composite lesson because it is the most common failure in the field. A team splits a monolith by entity — User, Order, Product, Payment, Inventory as services, one per major table. It feels like progress; there are services now. Within a year the symptoms arrive: every feature touches multiple services, so velocity *drops* below the monolith's; the User service is called by everything and becomes a reliability bottleneck; "place order" needs a distributed transaction so the team adopts a saga framework they do not fully trust; and because the data was never cleanly owned, two services end up sharing a database "just for this one query," which couples their schemas forever. The team concludes "microservices are hard" or "microservices were a mistake" — when in fact *the boundaries* were the mistake. Segment's well-known public reversal — they consolidated a sprawl of microservices back toward a monolith around 2018 because the operational overhead exceeded the benefit — is a real instance of this lesson, though their root cause was as much per-customer destination sprawl as a pure boundary error. The accurate, non-fabricated takeaway: **when microservices hurt more than they help, the first thing to audit is whether the boundaries follow business capabilities or just the old table layout.** More often than not, a distributed monolith is an entity-split monolith wearing a network costume. For the full anatomy of when to retreat, see [microservices anti-patterns and when to go back to monolith](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith).

## 16. When to reach for DDD boundaries (and when it is overkill)

DDD and EventStorming are a cost — workshops, domain-expert time, the discipline of maintaining ubiquitous language and context maps. When is that cost worth it?

**Reach for full DDD boundary work when:** the domain is genuinely complex with rich, contested business rules (e-commerce, fintech, logistics, healthcare, insurance — anywhere the *business logic* is the hard part, not the technology); you are decomposing a monolith and the boundaries will be expensive to move once services are live; multiple teams will own different parts and you need the boundaries to also work as ownership lines; or the same words demonstrably mean different things to different parts of the business (the surest sign bounded contexts will pay off). In these cases the up-front investment in finding the right seams is the cheapest money you will spend, because the alternative is paying it forever at runtime.

**Do not over-invest in DDD when:** the domain is genuinely simple (a CRUD admin tool, a small internal app) where the nouns really are independent and there is no rich behavior — here the ceremony of bounded contexts and aggregates is overhead with no payoff, and a modest service or even a monolith is correct; you are at the prototyping stage where the domain is still being discovered and any boundary you draw now will be wrong (start with a monolith, as [monolith-first](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) argues, and let the boundaries reveal themselves); or you are a tiny team where the operational cost of *any* extra service outweighs the decoupling benefit. The mature position: **DDD's strategic patterns (ubiquitous language, bounded contexts, context mapping) are valuable at almost any scale and cost little to apply on paper, but turning every bounded context into a separately deployed service is a cost you should defer until you have a concrete reason — team autonomy, independent scaling, or independent deployability — to pay it.** You can have bounded contexts as *modules inside a monolith* long before you have them as services, and that is exactly the modular-monolith path. Find the boundaries early; deploy them separately late.

## 17. Key takeaways

- **The boundary is the decision that dominates everything.** A wrong boundary cannot be fixed by good engineering downstream; it taxes you forever with chattiness, distributed transactions, and shared databases. Get the seams right before you cut.
- **Never split by technical layer (UI/logic/data) or by entity (User/Order services).** Both optimize for the structure of your artifacts instead of the direction of business change, and both produce a distributed monolith. Split by **business capability / subdomain** so each service owns a use case end to end.
- **A bounded context is the unit that maps to a service.** Find them where the *language changes* — the same word meaning different things ("Order" as billable cart vs physical shipment) is a boundary screaming to be drawn.
- **The aggregate is your transaction boundary.** Data with a shared invariant that must hold atomically belongs in one aggregate, one context, one service, one transaction. Reference other aggregates and other contexts by ID, never by object — that is what keeps the boundary clean.
- **A cross-service invariant is a design smell demanding a decision**, not a thing to paper over: either the boundary is wrong (merge), or the invariant is a time-windowed process (saga), or it is a soft "should" (reconcile). Never share a database to enforce it.
- **Defend outward-facing boundaries with an anti-corruption layer** so a vendor's or legacy system's foreign model never leaks into your clean domain. Generic subdomains (payment, notifications) should be thin contexts wrapped in an ACL, not richly modeled.
- **Two services that always deploy together should be one service.** Avoid nano-services as fiercely as you avoid the distributed monolith; the right size is one capability, one team, one independent deploy schedule.
- **Measure boundaries by cross-service calls per operation and p99, not per-service latency.** The highest-leverage performance fix in microservices is moving a boundary so the hot path's data is local — invisible to anyone profiling one service at a time.
- **Find the boundaries early (on paper, via EventStorming), deploy them separately late.** Bounded contexts can be modules in a monolith long before they are services; do the strategic DDD work cheaply up front and defer the operational cost of separate deployment until you have a concrete reason to pay it.

## 18. Further reading

- **Eric Evans, *Domain-Driven Design: Tackling Complexity in the Heart of Software*** (2003) — the foundational text. Dense, but the strategic-design chapters (bounded contexts, context mapping, ubiquitous language) are the ones that matter for service boundaries.
- **Vaughn Vernon, *Implementing Domain-Driven Design*** (2013) — the practical companion to Evans, with concrete code for aggregates, repositories, and domain events. Vernon's "Effective Aggregate Design" essays (free online) are the best treatment of aggregate sizing.
- **Sam Newman, *Building Microservices* (2nd ed.)** — the practitioner's bible for this whole series; its chapters on decomposition, information hiding, and the dangers of the distributed monolith are directly on this topic.
- **Alberto Brandolini, *EventStorming*** — the definitive guide to the workshop technique for discovering bounded contexts.
- **Chris Richardson, *Microservices Patterns*** — strong on decomposition strategies (by business capability vs by subdomain) and on the aggregate-as-transaction-boundary discipline.
- **Uber Engineering, "Introducing Domain-Oriented Microservice Architecture" (DOMA)** — the public write-up of re-aggregating sprawl into capability-aligned domains.
- Continue in this series with the consequences of getting boundaries right: [database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), and [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice). For the mechanism deep-dives, see [the saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions) and [evolutionary architecture](/blog/software-development/system-design/evolutionary-architecture-designing-for-change).

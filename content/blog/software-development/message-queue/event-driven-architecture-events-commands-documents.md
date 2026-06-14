---
title: "Event-Driven Architecture: Events vs Commands vs Documents"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn the one distinction that makes or breaks event-driven systems — the difference between an event (a fact about the past), a command (an instruction to one handler), and a document (pure data) — plus event notification vs event-carried state transfer, choreography, eventual consistency, and how a disguised command silently recouples your services."
tags:
  [
    "message-queue",
    "event-driven",
    "event-driven-architecture",
    "choreography",
    "eventual-consistency",
    "microservices",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "system-design",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/event-driven-architecture-events-commands-documents-1.webp"
---

Here is a sentence I have heard in more design reviews than I can count: "We'll make it event-driven so the services are decoupled." It sounds right. It is usually wrong, because nine times out of ten the thing the team is about to publish is not an event at all. It is a command wearing an event's clothes. They publish a message called `SendWelcomeEmail`, congratulate themselves on the new decoupled architecture, and ship it. Six months later the order service cannot be deployed without coordinating with the email team, a "new event" requires a meeting, and nobody can explain why the supposedly decoupled system feels exactly as tangled as the synchronous one it replaced. The decoupling never happened. The publish-subscribe plumbing was real, but the *intent* was still imperative, and intent is the thing that actually couples or decouples services.

This post is about the single distinction that separates event-driven architecture that works from event-driven architecture that is just RPC with extra latency: the difference between an **event**, a **command**, and a **document**. An event is a fact about something that already happened — `OrderPlaced`, `PaymentTaken`, `AddressChanged`. It is fire-and-forget. The producer announces it to the world and does not know, or care, who is listening; zero consumers or fifty consumers, the producer's code is identical either way. A command is an instruction — `ChargeCard`, `ReserveInventory`, `SendEmail`. It is imperative, it is addressed to exactly one handler, and the sender expects it to be carried out. A document is neither; it is pure data in motion, a payload handed from one place to another with no behavioral expectation attached. These three are not interchangeable. Choosing the wrong one is the most common and most expensive mistake in this entire style of building systems, and the figure below lays out exactly how they differ on the four dimensions that matter.

![A matrix comparing event, command, and document across direction, cardinality, coupling, and sender expectation, showing events as outward-announcing with zero-to-many consumers and loose coupling](/imgs/blogs/event-driven-architecture-events-commands-documents-1.webp)

By the end of this post you will be able to look at any message in your system and classify its intent with confidence; you will know why a disguised command recouples services that publish-subscribe was supposed to decouple; you will understand the central design tradeoff between *event notification* (thin events that make consumers call back for details) and *event-carried state transfer* (fat events that ship the data so nobody has to call back); you will see how *choreography* lets services compose into emergent flows with no central brain; and — just as important — you will know the real, unglamorous price of all this, which is eventual consistency, implicit flow, and debugging that spans services. Event-driven architecture is not free decoupling. It is a trade, and you should make it with both eyes open.

This is post twenty-one in a forty-part message-queue series. It builds directly on two earlier posts you should read first if you have not: [what a message queue actually is and why asynchronous decoupling matters](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling), and [the three messaging models — queue, pub/sub, and log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models). Those posts give you the transport primitives. This post is about what you *put on* those primitives and what the choice means. It also sets up two siblings still to come: the [saga pattern, orchestration versus choreography](/blog/software-development/message-queue/saga-pattern-orchestration-vs-choreography), which takes the choreography idea here and makes it transactional, and [event sourcing and CQRS with an event log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log), which takes events to their logical conclusion by making the log the source of truth.

## 1. What "event-driven" actually means

Strip away the buzzword and "event-driven" names a specific *architectural style*: a way of organizing a system around the production, detection, and consumption of events, where the flow of control is driven by things that happen rather than by procedures that call each other. In a request-driven system, service A decides what should happen next and tells service B to do it: A *calls* B. In an event-driven system, service A announces that something happened, and any service that cares about that fact reacts on its own initiative. A does not call anyone. The dependency arrow flips: instead of A depending on B, B depends on the *event* that A emits, and A depends on nothing downstream at all.

That inversion is the whole point, and it is worth dwelling on because it is subtle. In the request-driven world, when you add a new thing that needs to happen after an order is placed — say, you now want to award loyalty points — you go edit the order service. You add a call to the loyalty service. The order service grows a new dependency, a new failure mode, a new reason to redeploy. Every new reaction makes the order service fatter and more entangled. In the event-driven world, you do not touch the order service at all. The order service already emits `OrderPlaced`. You write a brand-new loyalty service that subscribes to `OrderPlaced` and awards points. The order service does not know the loyalty service exists. It never will. This is the property people mean when they say "decoupled," and it is real — *when you do it right*.

### A style, not a technology

It is important to separate the style from the plumbing. You can build an event-driven system on Kafka, on RabbitMQ, on AWS SNS plus SQS, on Google Pub/Sub, on NATS, or on a hand-rolled outbox table polled by a cron job. The broker is transport. Event-driven architecture is a discipline about *what messages mean* and *who is allowed to know about whom*. You can run Kafka and build a tightly coupled request-driven system on top of it (people do this constantly — they just turned their RPC calls into request-reply over a topic). And you can build a genuinely event-driven system on the humblest transport imaginable. The technology does not make you event-driven. The intent of your messages and the direction of your dependencies do.

The figure below contrasts the two worlds directly. On the left, request-driven: the order service reaches out and calls billing, then mail, then shipping, blocking on each, knowing all three by name. On the right, event-driven: the order service emits one fact and goes back to its own business; consumers react asynchronously and independently, and adding a new reaction means writing a new subscriber, not editing the order service.

![A before-and-after diagram contrasting request-driven architecture where the order service synchronously calls billing, mail, and shipping, against event-driven architecture where the order service emits one event and independent consumers react asynchronously](/imgs/blogs/event-driven-architecture-events-commands-documents-2.webp)

### The four properties of an event-driven system

When people describe a system as event-driven, they usually mean it exhibits four properties at once, and it is worth naming them because a system can have some and not others. First, **asynchrony**: the producer does not wait for consumers. It emits and moves on. Second, **broadcast semantics for events**: a single emitted fact can reach zero, one, or many consumers, and the producer's behavior does not change with the count. Third, **inversion of dependency**: consumers depend on producers' events, not the reverse, so producers can be written and deployed with no knowledge of consumers. Fourth, **reaction over instruction**: the system's behavior emerges from services reacting to facts, rather than from one service commanding others through a script.

A system that has the publish-subscribe plumbing but lacks the inversion of dependency — because its "events" are actually addressed instructions — is event-driven in form only. That is the trap. We will spend a whole section on it because it is the single most common way teams think they have decoupled and have not.

### Asynchronous does not imply event-driven

One more confusion to clear before we go deeper: asynchrony and event-driven-ness are *not the same thing*, even though they usually travel together. You can send a *command* asynchronously — drop `ChargeCard` onto a single-consumer queue and let the billing service pick it up when it can. That is asynchronous (the sender does not block) but it is not event-driven (it is still an imperative instruction to one named handler). Conversely, you could in principle deliver an *event* synchronously (an in-process observer pattern fires listeners on the same call stack). The two dimensions are orthogonal: synchronous-versus-asynchronous is about *whether the sender waits*, while event-versus-command is about *whether the message is a fact or an instruction*. Event-driven architecture combines the two — asynchronous delivery of facts — but you should hold them apart in your head, because conflating them is how people convince themselves that "we put it on a queue" means "we are decoupled," when an asynchronous command queue couples the sender to the handler exactly as much as a synchronous call does. The queue bought you load leveling and retry; it did not buy you decoupling. Only the event intent does that.

## 2. Three intents: event, command, document

Every message that crosses a service boundary carries an *intent*, whether you named it or not. The intent answers a deceptively simple question: what does the sender expect to happen as a result of sending this? There are exactly three useful answers, and they correspond to the three message types. Getting precise about which one you mean is the most valuable habit you can build in this style of architecture, because the intent — far more than the payload or the transport — determines how the message couples your services.

### An event is a fact about the past

An **event** is a statement that something happened. It is named in the past tense, and that naming convention is not cosmetic — it enforces the right mental model. `OrderPlaced`. `PaymentTaken`. `CustomerAddressChanged`. `InventoryReserved`. `ShipmentDispatched`. Each is an immutable fact about a thing that already occurred. You cannot argue with a fact. You cannot "decline" `OrderPlaced` — the order *was* placed; that is history now. The producer emits the event to announce the fact to whoever is interested, and crucially, **the producer does not know who is interested**. The producer is not requesting anything. It is reporting.

The defining behavioral property of an event is **fire-and-forget with zero-to-many consumers**. The order service publishes `OrderPlaced` and is done. There might be no consumers (in dev, before anyone has subscribed). There might be one (just analytics). There might be eight (analytics, email, fraud, loyalty, inventory, tax, the data lake, and a fourth team you have never met). The order service's code is *byte-for-byte identical* in all of these cases. It publishes one event. That invariance — that the producer's logic does not change as consumers come and go — is the litmus test for whether you have a real event. If adding a consumer requires touching the producer, you do not have an event.

### A command is an instruction to one handler

A **command** is an instruction to do something. It is named in the imperative: `ChargeCard`, `ReserveInventory`, `SendWelcomeEmail`, `CancelOrder`. The sender expects the command to be carried out, and it is addressed to **exactly one handler**. There is a specific service whose job is to handle `ChargeCard`, and the sender is talking to *that service*, even if a broker sits in between. The sender depends on the existence and correctness of that handler. If no handler picks up the command, that is a problem — a command with no handler is a bug, whereas an event with no consumers is perfectly fine.

Commands are not evil. A great deal of useful work in distributed systems is genuinely command-shaped. "Charge this card" is a command. There is one billing service, you want it to charge the card, and you expect that to happen. Sending it asynchronously through a queue (a single-consumer work queue, not a broadcast) is a legitimate and powerful pattern — it buys you load leveling and retry. But you must *know* it is a command, because a command couples the sender to the handler. The sender knows the handler exists, expects it to act, and usually cares about the result. That coupling is fine when it is intentional. It is poison when it is accidental, hiding inside something you called an event.

### A document is pure data transfer

A **document** (sometimes called just a "message" in the narrow sense, or a "data message") is neither a fact nor an instruction. It is a payload — a chunk of data moved from one place to another with no behavioral expectation attached. A CSV of yesterday's transactions dropped onto a topic for a batch job to pick up. A serialized customer record shipped from one system to another during a migration. A rendered PDF handed off to a storage service. The sender is not announcing a fact and not requesting an action. It is transferring data. Documents are the simplest intent and the least interesting architecturally, but naming them matters because a document mistaken for an event invites consumers to attach behavior that the sender never intended, and a document mistaken for a command invites a handler to assume an instruction that was never given.

The figure below organizes all three intents — and the sub-patterns they spawn — into a single taxonomy you can keep in your head. Notice that the event branch splits further, into notification and state transfer; that split is the subject of section four and is where most of the interesting design tension lives.

![A taxonomy tree rooting at message intent and branching into event, command, and document, with the event branch further splitting into thin notification and fat state-transfer patterns and the command branch into synchronous and asynchronous forms](/imgs/blogs/event-driven-architecture-events-commands-documents-6.webp)

### The one-line test for each intent

When you cannot decide what a message is, ask one question per candidate. For an event: *"If nobody listened, would the sender care?"* If the answer is no — the sender just wants to announce the fact and would shrug if no one consumed it — it is an event. For a command: *"Does the sender expect a specific thing to be done, and would it be a bug if no one did it?"* If yes, it is a command. For a document: *"Is the sender attaching any behavioral expectation at all, or just handing over bytes?"* If it is just bytes, it is a document. These three questions resolve the vast majority of real cases, and they force you to think about *the sender's expectation*, which is exactly the dimension that determines coupling.

### Where the intent shows up in the wire format

The intent is not just a mental label — it leaves fingerprints in the message itself, and learning to read those fingerprints lets you classify a message you did not write. An event carries a *past-tense type name* and *no addressee*: nothing in `OrderPlaced` says where it should go, because it does not go anywhere in particular; it is published to a topic and whoever subscribes, subscribes. A command carries an *imperative type name* and is *routed to a specific destination*: `ChargeCard` goes to the billing service's command queue, and that routing is part of the message's identity. A document carries *neither verb tense nor routing intent*; it is a blob with a content type. Look at any message in a real system and you can usually tell its intent from these two signals alone — the verb tense of its name and whether it is addressed.

A useful sharpening: events and commands also differ in *who owns the name*. The producer owns the event's name, because the producer is asserting the fact and gets to decide what to call it — the order service decides it emits `OrderPlaced`. The *consumer* effectively owns the command's name, because a command is named for the operation the handler performs — the billing service defines what `ChargeCard` means and what fields it requires, and senders must conform to billing's contract. This ownership flip is another reason commands couple and events do not: with a command, the sender is constrained by the receiver's contract; with an event, the receiver is constrained by the sender's fact, and any number of receivers can interpret that fact however they like.

## 3. Why a disguised command recouples your services

This is the heart of the post, so let us be concrete and a little merciless. A **disguised command** is a message that is structurally a command — it instructs a specific handler to do a specific thing, and the sender expects it to happen — but has been *named* and *published* as if it were an event. The classic tell is a message named in the imperative but emitted on a broadcast topic: `SendWelcomeEmail` published to a "user events" topic. It looks event-driven. It is RPC.

### How the disguise happens

The disguise almost always happens by accident, through a perfectly reasonable-sounding chain of decisions. The product manager says: "When a user signs up, send them a welcome email." An engineer translates that literally. They make the signup service publish `SendWelcomeEmail` to a topic, and the email service consumes it and sends the email. Decoupled, right? The services do not call each other directly; there is a broker in between. But look at what actually happened. The signup service *knows* an email should be sent. It is *instructing* the email service to send it. It has encoded "the thing that should happen after signup" — a downstream concern — into its own logic. The signup service now owns a piece of the email service's responsibility. It is coupled to the email behavior.

Watch how this bites. Suppose the company decides welcome emails are spammy and should be replaced by an in-app notification. In a properly event-driven system, the signup service emits `UserSignedUp` (a fact), the email service subscribes and sends an email, and to change the behavior you simply *stop* the email service from subscribing and write a notification service that subscribes instead. The signup service is never touched. But in the disguised-command system, `SendWelcomeEmail` is baked into the signup service's code. To change the behavior you have to go edit the signup service to stop emitting `SendWelcomeEmail` and start emitting `SendWelcomeNotification` — and now you are redeploying signup to change an email policy. The coupling you thought you removed is still there; you just routed it through a broker.

### The deeper symptom: the producer knows too much

The diagnostic for a disguised command is that **the producer knows something it should not have to know**. The signup service, emitting `SendWelcomeEmail`, knows that a welcome email is part of the signup flow. Why should it? Signup's job is to create accounts. Whether the company sends welcome emails is a *marketing* concern that should live entirely in a downstream service. When the producer encodes downstream policy into its messages, the producer becomes a chokepoint for changing that policy, and the whole promise of decoupling evaporates.

Reframe the same need as an event and the knowledge moves to where it belongs. The signup service emits `UserSignedUp` — a fact it is uniquely qualified to assert, because it is the thing that created the user. It knows nothing about emails, notifications, loyalty, analytics, or anything else. The email service, which *is* the right place for email policy, subscribes to `UserSignedUp` and decides — on its own — to send an email. Now the email policy lives in the email service. Marketing changes the policy by changing the email service. The signup service is blissfully unaware. *That* is decoupling.

#### Worked example: modeling "customer changed their address" three ways

Let us make this fully concrete with a single requirement modeled three ways, because the same business fact can be expressed as an event, a state-transfer event, or a command, and each couples the services completely differently. The requirement: a customer updates their shipping address in the account service, and as a result the shipping service should use the new address, the billing service should update the address on file, and a confirmation email should go out.

**Modeling one — as a thin event (event notification).** The account service emits `CustomerAddressChanged { customerId: 4471, version: 7 }`. That is it — an id and a version number, roughly two hundred bytes. It is a fact: this customer's address changed, here is who and which revision. Shipping, billing, and email each subscribe. When each receives the event, it calls back to the account service's API — `GET /customers/4471/address` — to fetch the actual new address, then updates its own state. The account service knows about none of these consumers. Coupling: the account service is decoupled from *who* reacts, but every consumer is *runtime-coupled* to the account service's address API — when account is down, consumers cannot fetch, and they all hammer account with read traffic the instant the event lands.

**Modeling two — as a fat event (event-carried state transfer).** The account service emits `CustomerAddressChanged { customerId: 4471, version: 7, address: { line1, line2, city, region, postal, country } }` — the full new address, maybe four kilobytes. Now shipping, billing, and email each receive everything they need *in the event itself*. No callback. Each updates its local copy of the address directly from the event payload. Coupling: the account service is decoupled from who reacts *and* consumers no longer call back at runtime — they can process the event even if account is completely offline. The cost: the address data is now duplicated in four services, and if a consumer is slow, its copy is briefly stale. We pay in duplication and staleness to remove the runtime callback.

**Modeling three — as a command.** The account service, after saving the new address, emits `UpdateShippingAddress { customerId: 4471, address: {...} }` to shipping, `UpdateBillingAddress {...}` to billing, and `SendAddressChangeConfirmation {...}` to email. These are three commands, addressed to three specific handlers. Coupling: the account service now *knows about all three downstream services*. It knows shipping needs the address, billing needs the address, and email needs to send a confirmation. It has absorbed knowledge of the entire downstream flow. Add a fourth reaction — say, a fraud check on address changes — and you must edit the account service to emit a fourth command. This is the disguised-command failure in full: the producer became the place you edit to change downstream behavior. The supposedly event-driven system is a synchronous orchestrator that happens to use a broker.

The lesson is stark. The *same business fact* produced three radically different coupling profiles depending purely on intent. The two event models keep the account service ignorant of downstream consumers; the command model makes it the orchestrator of all of them. If your goal was decoupling, modeling three defeated it entirely, and it is exactly what teams build when they reach for "event-driven" without internalizing the intent distinction.

## 4. Event notification vs event-carried state transfer

The worked example above already introduced the second great design decision in event-driven architecture, and it deserves its own section because it is where experienced teams spend real design effort. Both options are genuinely events — facts about the past, fire-and-forget, zero-to-many consumers — so both preserve the decoupling of intent. They differ in *how much data the event carries*, and that single choice ripples into coupling, payload size, staleness, and operational behavior.

### Event notification: the thin event

An **event notification** is a thin event: it says *something changed* and carries just enough identifying information for an interested consumer to go find out more. `CustomerAddressChanged { customerId: 4471 }`. `OrderUpdated { orderId: 9912 }`. `ProductPriceChanged { sku: "ABC-123" }`. The event is a doorbell, not a delivery. When a consumer hears it, if it needs the details, it calls back to the source system's API to fetch the current state. The notification's job is purely to *wake the consumer up*; the data comes from the callback.

The appeal of event notification is its tiny payload and its simplicity. The event is a few hundred bytes regardless of how big the underlying entity is. The source system stays the single authority for the data — there is one place to read the truth, and consumers always get the *current* value when they call back, never a stale snapshot. The schema of the event is trivially small and stable: it is just an id and maybe a version.

The cost is **runtime coupling through the callback**. Every consumer now depends on the source system's read API being available at the moment it processes the event. If the source is down, consumers cannot complete their work. Worse, there is the **thundering herd**: when a popular entity changes and fifty consumers all receive the notification at once, they all call back simultaneously, hammering the source's read API with a synchronized spike. And there is a subtle ordering hazard — by the time a consumer calls back, the entity may have changed *again*, so the consumer might read version 9 in response to a notification about version 7, which is sometimes fine and sometimes a correctness bug.

### Event-carried state transfer: the fat event

**Event-carried state transfer** (often abbreviated ECST) is the fat event: the event carries the full data the consumer needs, so the consumer never has to call back. `CustomerAddressChanged { customerId: 4471, version: 7, address: {...} }` with the entire address embedded. The consumer updates its own local copy directly from the payload. The event is self-contained.

The win is **the elimination of runtime coupling**. The consumer can process the event with the source system completely offline, because everything it needs is in the event. No callback, no thundering herd, no read-API dependency. Consumers can also build and maintain their *own local read model* of the data — shipping keeps its own copy of customer addresses, updated by events — which means at query time it never has to leave its own process. This is enormously powerful for autonomy and resilience: a consumer that holds a local replica of the data it needs can keep functioning during outages of the systems that own that data.

The cost is **data duplication and staleness**. The address now lives in four places. Each consumer's copy is only as fresh as the last event it processed; if a consumer is lagging by thirty seconds, its address copy is thirty seconds stale. You also pay in payload size — fat events are bigger, which matters at high volume — and in **schema coupling on the data shape**: every consumer now parses the embedded address structure, so changing that structure is a coordinated, breaking change across all consumers, whereas a notification's tiny schema almost never changes.

![A before-and-after diagram contrasting a thin event notification that carries only an id and forces consumers to call back for data, against a fat event-carried state transfer that ships the full record so consumers need no callback but hold a copy that can go stale](/imgs/blogs/event-driven-architecture-events-commands-documents-4.webp)

### Choosing between them

The decision comes down to which coupling you can least afford. Reach for **event notification** when the data is large or changes constantly (you do not want to ship a megabyte on every tiny change), when consumers rarely need the full data (most just need to know *that* something changed), when staleness is unacceptable (you need the consumer to always read the current truth), or when the source system's read API is robust and can handle the callback load. Reach for **event-carried state transfer** when consumer autonomy and resilience matter most (consumers must keep working during source outages), when the data is small and changes infrequently, when you want consumers to maintain fast local read models, or when the callback load would overwhelm the source.

Many mature systems use both, even for the same entity, on different topics — a thin `OrderUpdated` notification for consumers that just need to invalidate a cache, and a fat `OrderSnapshot` state-transfer event for consumers building a local read model. The two are not mutually exclusive; they are tools for different consumer needs. The connection to [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) is direct: CDC streams are very often event-carried state transfer, shipping each row change as a fat event so downstream systems can rebuild their own copy of the table.

```python
# Event notification: thin event, consumer calls back for details
def emit_address_changed_notification(customer_id: int, version: int):
    event = {
        "type": "CustomerAddressChanged",
        "customerId": customer_id,
        "version": version,
        "occurredAt": now_iso8601(),
    }
    # ~200 bytes; consumers will GET /customers/{id}/address to fetch the data
    producer.send("customer-events", key=str(customer_id), value=json.dumps(event))


# Event-carried state transfer: fat event, fully self-contained
def emit_address_changed_ecst(customer_id: int, version: int, address: dict):
    event = {
        "type": "CustomerAddressChanged",
        "customerId": customer_id,
        "version": version,
        "occurredAt": now_iso8601(),
        "address": address,  # the entire new address travels with the event
    }
    # ~4 KB; consumers update their local copy with zero callbacks
    producer.send("customer-events", key=str(customer_id), value=json.dumps(event))
```

Notice the producer is decoupled from consumers in *both* versions — that is the event intent doing its job. The notification-versus-state-transfer choice is orthogonal to the event-versus-command choice. You first decide it is an event (a fact, fire-and-forget); then you decide how fat to make it.

## 5. Choreography and emergent flows

Once your services communicate through events, a remarkable thing becomes possible: multi-step business processes can run with **no central coordinator**. This is **choreography**. In choreography, each service reacts to events emitted by other services and emits its own events in turn, and the end-to-end flow *emerges* from these local reactions without anyone scripting it from above. There is no orchestrator holding the master plan. Each service knows only its own piece: "when I see this event, I do my work and emit that event."

### The order flow, choreographed

Consider an order fulfillment flow. The order service, on receiving an order, emits `OrderPlaced`. The payment service subscribes to `OrderPlaced`; when it sees one, it charges the card and emits `PaymentTaken` (or `PaymentFailed`). The shipping service subscribes to `PaymentTaken`; when it sees one, it dispatches the shipment and emits `Shipped`. The notification service subscribes to `Shipped` and emails the tracking number. No service "runs the order flow." There is no order-flow class anywhere in the codebase. The flow — place, charge, ship, notify — is an emergent consequence of four services each reacting to the previous one's fact.

![A graph showing a choreographed order flow where the order service emits OrderPlaced, payment reacts and emits PaymentTaken or PaymentFailed, shipping reacts to PaymentTaken and emits Shipped, notification reacts to Shipped, and a reversal service reacts to PaymentFailed](/imgs/blogs/event-driven-architecture-events-commands-documents-9.webp)

The figure above traces the chain. Notice the branch at the payment service: on `PaymentTaken` the flow proceeds to shipping; on `PaymentFailed` a reversal service reacts to refund or release the order. Nobody coded an `if payment_failed` branch into a central orchestrator. The payment service simply emits one of two facts, and different services react to each. The flow's branching logic is *distributed across the subscriptions*, which is the defining characteristic of choreography and also, as we will see in section eight, the source of its biggest operational headache.

#### Worked example: adding a new reaction to a choreographed flow

Here is the payoff of choreography, measured in the only currency that matters for maintainability: *how many services must change to add a new behavior?* Take the choreographed order flow above and add a requirement: "When an order is placed, also reserve inventory." Let us count the changes in two architectures.

**In a choreographed event-driven system:** the inventory service is a brand-new service that subscribes to `OrderPlaced` and reserves stock. Number of *existing* services modified: **zero**. The order service is not touched — it already emits `OrderPlaced`. The payment service is not touched. Shipping is not touched. You write one new service, wire its subscription, deploy it. Nothing that already works is at risk, because nothing that already works was changed. This is the autonomy dividend: new reactions are *additive*.

**In a request-driven orchestrated system:** the order service is the orchestrator. To add inventory reservation, you edit the order service to call the inventory service at the right point in the flow. Number of services modified: **at least one, and it is the most critical one** — the order service, the heart of the system, the thing you least want to redeploy on a Friday. You also now have to decide where in the synchronous call chain the inventory call goes, what happens if it fails partway, and how to roll back the charge if inventory reservation fails after payment succeeded. Every new reaction makes the orchestrator bigger and riskier.

Now scale the comparison. Add five more reactions over a year — fraud scoring, loyalty points, tax calculation, a data-lake sink, a partner webhook. In the choreographed system, that is five new services, **zero** modifications to existing ones. In the orchestrated system, that is five more edits to the order service, which is now a thousand-line procedure that calls eight downstream services, blocks on each, and cannot be reasoned about by anyone. The choreographed system's existing services never grew. *That* is what "loose coupling" buys you, concretely: the cost of adding a reaction stays flat instead of growing with every addition.

### Choreography versus orchestration, stated plainly

It helps to put the two extremes side by side, because they are the poles of a real design axis and most systems live somewhere between them. In **orchestration**, one service — the orchestrator — holds the flow as an explicit script and *commands* the others through each step: it tells payment to charge, waits, tells shipping to ship, waits, tells notification to notify. The flow lives in one place, is easy to read, and is easy to change *in that place* — but the orchestrator knows every participant, the participants are coupled to being commanded, and the orchestrator is a single point of both knowledge and failure. In **choreography**, there is no orchestrator; each service reacts to events and emits events, and the flow is distributed across subscriptions. No participant knows the whole flow, no participant commands another, and the flow is maximally decoupled — but it lives nowhere legible.

Note the intent difference baked into this contrast: **orchestration runs on commands, choreography runs on events.** An orchestrator's messages are imperative ("charge this card") because it is directing; a choreographed service's messages are facts ("payment was taken") because it is announcing. This is the same event-versus-command distinction from section two, now operating at the level of whole flows. When you choose choreography you are choosing events as your coordination primitive, with all the decoupling and all the illegibility that implies. When you choose orchestration you are choosing commands, with all the legibility and all the coupling. The saga sibling post goes deep on choosing between them for transactional flows; the short version is that pure choreography is wonderful for loosely related reactions and becomes painful for tightly sequenced, must-roll-back-together flows, where a thin orchestrator earns its keep.

### The dark side of emergence

I am being deliberately one-sided to make the decoupling vivid, so let me immediately add the counterweight, because choreography is not a free win. The flip side of "no central brain" is "no central place to understand the flow." When the order process is a script in the order service, you can read it top to bottom and know exactly what happens. When it is choreographed across six services' subscriptions, *the flow exists nowhere as a single artifact*. To understand it, you have to trace which service subscribes to which event, in your head or in a diagram, across the whole system. We will return to this in section eight — it is the central tradeoff of the style, and choreography is precisely where you pay for it. The saga sibling post, [orchestration versus choreography](/blog/software-development/message-queue/saga-pattern-orchestration-vs-choreography), is entirely about when to pull a central orchestrator back in to regain that legibility, especially for flows that need transactional rollback.

## 6. Eventual consistency as the price of decoupling

Here is the bill for everything good in the previous sections. The moment you make your system asynchronous — the moment the producer stops waiting for consumers — you give up **immediate consistency** and accept **eventual consistency**. There is now a real window of time during which different parts of your system disagree about the state of the world, and you must design for that window rather than pretend it does not exist. This is not a flaw in event-driven architecture. It is the *definition* of it. If everyone always agreed instantly, you would be synchronous, and you would not have the decoupling.

### What eventual consistency actually looks like

When the order service emits `OrderPlaced`, the order exists *in the order service immediately*. But the analytics service has not counted it yet. The email service has not sent the confirmation yet. The loyalty service has not awarded points yet. For some window — milliseconds usually, sometimes seconds, occasionally much longer if a consumer is lagging — the system is in a state where the order is real but its consequences have not all propagated. If a user places an order and immediately checks their loyalty points, they might not see the points yet. The system is *inconsistent* in that instant. Given enough time and no new changes, every consumer catches up and the system *converges* — that is the "eventual" in eventual consistency.

![A timeline showing an OrderPlaced event emitted at time zero, analytics updating at eight milliseconds, email and fraud at forty milliseconds, a loyalty consumer lagging and going stale at three hundred milliseconds, and all consumers converged by two seconds](/imgs/blogs/event-driven-architecture-events-commands-documents-8.webp)

The figure above shows a realistic convergence profile. The event is emitted at T+0. Analytics, a fast in-memory consumer, updates within eight milliseconds. Email and fraud, which do more work, land around forty milliseconds. The loyalty service, suppose it is briefly backed up, lags to three hundred milliseconds — during that window, a user checking points sees stale data. By two seconds, every consumer has caught up and the system has converged. That convergence window is not a bug to be eliminated; it is an inherent property you must account for in the user experience and the correctness of every read. For a deeper treatment of where eventual consistency sits among the other models, see [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

### Designing for the window

You cannot make the window zero — that would be synchronous — but you can make it tolerable. A few standard moves. **Set user expectations**: show "your order is being processed" rather than asserting a final state the system has not reached. **Make reads tolerant of staleness**: a loyalty balance that is two seconds behind is usually fine; a bank balance might not be, which tells you the bank balance should not be eventually consistent. **Bound the lag and alert on it**: monitor consumer lag (covered at length in the consumer-offset posts of this series) so that a window that should be milliseconds does not silently become minutes. **Make consumers idempotent**: because delivery is at-least-once, the same event may arrive twice, and a consumer that double-counts loyalty points has turned eventual consistency into permanent corruption. Idempotency is the subject of [making at-least-once delivery safe with idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe), and it is non-negotiable in any real event-driven system.

There is also a measurable, not just philosophical, side to this. Suppose your order topic carries five thousand events per second and a consumer normally processes them in twelve milliseconds each with plenty of headroom. A deploy slows that consumer to forty milliseconds each — now it processes at twenty-five thousand events per second of capacity against five thousand per second of arrival, so it keeps up, barely, and the window stays small. But if a dependency stall pushes per-event time to four hundred milliseconds, the consumer's capacity drops to twenty-five hundred per second against five thousand arriving — it is now falling behind by twenty-five hundred events every second, and the convergence window grows without bound until you fix the consumer or scale it out. The point is that the convergence window is not a fixed property of "being event-driven"; it is a dynamic quantity that depends on consumer throughput versus arrival rate, and it can blow up silently. This is why lag monitoring is the heartbeat of an event-driven system: the window is healthy only as long as every consumer's processing rate exceeds its arrival rate, and the moment that inverts, eventual consistency stops being "a few milliseconds" and becomes "however long until someone notices."

The trade you are making is precise and worth stating plainly: you are trading **request-time errors for eventual-consistency bugs**. In a synchronous system, if billing is down, the order *fails immediately* and the user sees an error — annoying, but obvious, and the system is never in a half-done state. In an event-driven system, the order *succeeds immediately* and the billing event sits in a queue waiting for billing to recover — resilient, but now there is a window where the order exists and the charge does not, and if your code assumes "order placed implies charged," you have a bug that only manifests inside the convergence window. You did not remove the failure. You moved it from request time, where it is loud, to eventual-consistency time, where it is quiet and weird. Whether that is a good trade depends entirely on your domain, which is what [the CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) formalize.

## 7. Loose coupling and service autonomy

Let us name precisely what event-driven architecture buys, because "loose coupling" is thrown around so loosely that it has nearly lost meaning. EDA decouples services on three distinct axes, and a system can be decoupled on one and coupled on another, so it pays to be specific.

![A stack showing three layers of decoupling — temporal where producer and consumer need not run together, spatial where neither knows the other's address, and data where each owns its own schema — all resting on a durable bus that buffers, retains, and replays](/imgs/blogs/event-driven-architecture-events-commands-documents-7.webp)

### Temporal decoupling

**Temporal decoupling** means the producer and consumer do not need to be running at the same time. The producer emits `OrderPlaced` and the event sits durably in the bus. The consumer can be down for maintenance, crashed, or not even written yet, and when it comes up, the event is waiting. This is the property a durable log gives you that a direct synchronous call cannot: a synchronous call requires both ends alive *right now*; an event tolerates a consumer that is offline. Temporal decoupling is why event-driven systems absorb downstream outages gracefully — the events queue up and drain when the consumer recovers, rather than failing the upstream operation.

### Spatial decoupling

**Spatial decoupling** means the producer and consumer do not know each other's location or identity. The producer publishes to a topic; it does not have the consumer's address, does not know how many consumers there are, does not know their names. Consumers subscribe to the topic; they do not need to register with the producer or be discovered by it. The bus is the only thing either side knows about. This is what allows you to add a consumer without touching the producer — the producer has no list of consumers to update, because it never had one. Spatial decoupling is the structural basis for the "add a reaction without editing the producer" property we measured in section five's worked example.

### Data decoupling and the limits of it

**Data decoupling** means each service owns and evolves its own data schema independently. With event-carried state transfer, each consumer maintains its own local model of the data it cares about, shaped however suits that consumer, rather than sharing a database or a common object model with the producer. The order service's internal representation of an order and the analytics service's representation can be wildly different; the event is the contract between them, and each side maps the event to its own model.

But here is the honest caveat: **the event schema itself is a coupling point**, and it is the one most teams underestimate. The producer and all consumers must agree on the shape of the event. Change `OrderPlaced`'s structure carelessly — rename a field, change a type, remove something a consumer depends on — and you break every consumer at once. Event-driven architecture moves coupling *out of the call graph and into the event schema*. That is a better place for it (schemas evolve more slowly and more visibly than code call chains), but it is not *no* coupling. This is why mature event-driven shops invest heavily in **schema registries**, **schema compatibility rules** (you may add optional fields but not remove or repurpose existing ones), and **versioned events**. The decoupling is real, but it relocates the contract to the schema, and the schema must be governed with the same discipline you would give a public API — because that is exactly what it is.

## 8. The downsides: implicit flow and harder debugging

I have spent seven sections selling event-driven architecture, so let me now spend a serious one telling you why it can be a nightmare, because every senior engineer who has operated one of these systems carries the scars, and pretending the downsides are minor is how teams adopt EDA and then drown in it.

### The flow is implicit and lives nowhere

The single biggest cost is that **the end-to-end business flow is implicit**. In a synchronous orchestrated system, the flow is a procedure you can read: `placeOrder` calls `chargeCard` calls `reserveInventory` calls `ship`. It is right there in one function, top to bottom. In a choreographed event-driven system, *that procedure does not exist anywhere*. The order flow is an emergent property of which service happens to subscribe to which event, scattered across six codebases owned by four teams. To answer the question "what happens when an order is placed?" you cannot read a function — you have to trace event subscriptions across the entire system, often by reading config or asking around, because no single artifact describes the flow.

This makes onboarding brutal and makes change scary. A new engineer cannot find "the order code" because there is no order code; there is order-shaped behavior distributed across services that each know only their fragment. Worse, you can *break the flow by accident* in a way that is invisible at the change site: if someone refactors the payment service and accidentally stops emitting `PaymentTaken`, the shipping service silently stops shipping, and *nothing in the payment service's tests or code review will reveal it*, because the dependency is in shipping's subscription, not in payment's code. The coupling did not disappear; it became *invisible*, which is arguably worse than visible coupling because you cannot see what you are about to break.

### Debugging spans services and there is no stack trace

When something goes wrong, **there is no stack trace that spans the flow**. A request-driven failure gives you a stack trace through every service involved, end to end. An event-driven failure gives you... a missing email. The order was placed (you can see that), but the customer never got their confirmation. Why? You do not know where it broke, because the "flow" is six independent reactions and any one of them could have silently failed, dropped an event, crashed mid-processing, or never subscribed in the first place. You are now doing forensics: checking each service's logs, checking consumer lag on each topic, checking dead-letter queues, reconstructing the timeline by hand.

This is survivable, but *only* if you invest upfront in the tooling that makes the implicit flow observable, and that investment is not optional — it is the cost of admission. **Distributed tracing** (propagating a trace id through every event so you can reconstruct the causal chain in a tool like Jaeger) is the single highest-leverage thing you can build; without it, debugging a choreographed flow is archaeology. **Correlation ids** on every event let you grep one order's entire journey across all services. **Dead-letter queues** (covered in [dead-letter queues, retries, and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff)) catch events that no consumer could process so they do not vanish silently. **Consumer lag monitoring** tells you a consumer has fallen behind before users notice. And some teams build an explicit **flow visualizer** or maintain a hand-drawn map of which service subscribes to which event, precisely because the system refuses to tell you on its own.

### The other eventual-consistency bugs

Beyond the flow being implicit, event-driven systems breed a specific *class* of bug that synchronous systems mostly do not have: bugs that live in the convergence window or arise from message-ordering and duplication. **Out-of-order delivery**: if `AddressChanged v8` arrives before `AddressChanged v7` (because they went through different partitions or were retried), a naive consumer applies the stale one last and corrupts its copy — which is why version numbers and partition keys matter, a subject we cover in [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees). **Duplicate delivery**: at-least-once means events repeat, and a non-idempotent consumer double-applies. **The dual-write problem**: a service that updates its database and *then* publishes an event can crash in between, leaving the database changed but the event never sent — a half-done state with no error anywhere, which is exactly what the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) exists to solve. None of these exist in a synchronous monolith. They are the tax of going event-driven, and you pay it whether you planned for it or not.

The honest summary: event-driven architecture trades a *legible, fragile* system for a *resilient, illegible* one. The synchronous system fails loudly and is easy to understand but falls over when a dependency hiccups. The event-driven system shrugs off dependency outages but hides its flow and breeds consistency bugs. Neither is universally better. The trade is real, and you should make it deliberately, with the observability investment budgeted in *from day one*, not bolted on after the first 3 a.m. incident where a missing email turns into a four-hour spelunking expedition.

## 9. Designing good events

If the whole game is getting the intent right, then designing the events themselves is where you win or lose. Here is the accumulated craft, compressed into rules that experienced teams converge on.

### Name events as past-tense facts

Name every event in the **past tense**, after the business fact, from the producer's point of view. `OrderPlaced`, not `PlaceOrder` (that is a command) and not `OrderService.handleOrder` (that is an implementation detail). The past-tense name is a forcing function: it makes you ask "what *fact* am I asserting?" and if you cannot phrase it as a fact that already happened, you probably have a command, not an event. `SendEmail` fails the test — it is imperative, it names a downstream action, it is a disguised command. `UserSignedUp` passes — it is a fact the producer is uniquely qualified to assert. The naming convention is not pedantry; it is the cheapest possible guard against the disguised-command trap.

### Make events about business facts, not technical operations

Good events describe *business* facts, not *database* operations. `OrderPlaced` is a business fact. `OrderRowInserted` is a database operation leaking into your event stream. The difference matters because business facts are stable — orders will always be "placed" — while technical operations are coupled to your current implementation. If you emit `OrderRowInserted` and later change your storage, your events break or lie. Emit the fact at the level your domain experts would recognize. (The exception is deliberate change-data-capture streams, which *are* row-level by design and serve a different purpose — there the row change *is* the fact you want.)

### Carry enough context, but own the staleness decision

Decide deliberately, per event, where it sits on the notification-versus-state-transfer spectrum, and put *enough* context in the event for its primary consumers. A common failure is the over-thin event — `OrderUpdated { orderId }` — that forces every consumer to call back for everything, recreating runtime coupling. Another is the over-fat event that ships the entire aggregate including data no consumer needs, bloating the topic. Carry the fields consumers actually use; include a **version number** so consumers can detect and reject out-of-order updates; include an **event id** so consumers can deduplicate; include a **timestamp** (`occurredAt`) so consumers can reason about ordering and staleness; and include enough of a **key** that the broker can partition by entity to preserve per-entity ordering.

```json
{
  "eventId": "evt_01HR8K3M9X",
  "type": "OrderPlaced",
  "occurredAt": "2026-06-14T11:02:33.481Z",
  "version": 1,
  "aggregateId": "order_9912",
  "aggregateType": "Order",
  "data": {
    "customerId": 4471,
    "items": [{ "sku": "ABC-123", "qty": 2, "price": 1999 }],
    "currency": "USD",
    "totalCents": 3998
  }
}
```

That envelope — `eventId`, `type`, `occurredAt`, `version`, `aggregateId`, `data` — is the shape mature systems converge on. The `eventId` enables deduplication, `occurredAt` and `version` enable ordering decisions, `aggregateId` enables partitioning for per-entity ordering, and the `type` and a separate `version` on the *schema* enable evolution. Note that monetary amounts are integer cents (`totalCents: 3998` for \$39.98), never floats — a discipline worth keeping everywhere money appears.

### Version the schema and evolve it compatibly

Treat the event schema as a public contract, because it is one. Adopt **backward-compatible evolution**: you may add new optional fields (old consumers ignore them), but you may not remove fields, rename fields, or change a field's type or meaning, because some consumer depends on the old shape. When you genuinely must make a breaking change, introduce a *new event type* (`OrderPlacedV2`) and run both in parallel until every consumer migrates, then retire the old one. A **schema registry** with enforced compatibility rules turns this from a tribal-knowledge discipline into a machine-checked guarantee, which is exactly what you want for a contract that many teams depend on. The relevant rule for the whole style: the producer's job is to keep its events readable by consumers it has never met, including consumers that will subscribe years from now.

### Make consumers idempotent and tolerant

Finally, design every consumer to be **idempotent** (processing the same event twice has the same effect as once, via a dedup key on `eventId`) and **tolerant of out-of-order and stale data** (check the `version` and ignore an update older than what it already has). These are not optional refinements; they are the baseline correctness requirements imposed by at-least-once delivery and asynchronous convergence. A consumer that is not idempotent will eventually double-process and corrupt state. A consumer that is not order-tolerant will eventually apply a stale update and corrupt state. Build them right from the start, because retrofitting idempotency onto a corrupted production data set is a far worse afternoon than designing it in.

The two disciplines fold neatly into one consumer skeleton. The consumer checks whether it has already seen this `eventId` (idempotency), and it checks whether this event's `version` is newer than what it last applied for this entity (order tolerance). Both checks happen in the same transaction as the state update, so a crash cannot leave the dedup record and the state out of sync:

```python
def handle_address_changed(event: dict, db) -> None:
    event_id = event["eventId"]
    customer_id = event["data"]["customerId"]
    version = event["version"]

    with db.transaction():
        # Idempotency: if we have processed this exact event, skip it.
        if db.dedup_seen("address-consumer", event_id):
            return  # duplicate delivery, harmless no-op

        # Order tolerance: ignore an event older than what we already applied.
        current = db.get_applied_version(customer_id)  # None if never seen
        if current is not None and version <= current:
            db.dedup_record("address-consumer", event_id)  # still mark seen
            return  # stale or out-of-order update, do not regress state

        # Safe to apply: update the local copy and advance the version.
        db.upsert_address(customer_id, event["data"]["address"])
        db.set_applied_version(customer_id, version)
        db.dedup_record("address-consumer", event_id)
```

Three things make this robust. The dedup check and the state update share one transaction, so the consumer can crash anywhere and rerun safely — either the whole thing committed or none of it did. The version check prevents a redelivered or reordered older event from clobbering a newer state, which is the out-of-order hazard made concrete. And the dedup record is written *even when we skip on staleness*, so a later redelivery of that same stale event is short-circuited at the cheaper idempotency check instead of redoing the version comparison. This pattern — idempotency key plus a per-entity applied-version watermark, both committed atomically with the state change — is the workhorse of correct event consumers, and once you have written it once you tend to factor it into a shared library every consumer reuses.

![A grid showing an event-driven system where the order service, payment service, and event bus sit in the top row and the shipping, audit, and email services sit in the bottom row, each service both emitting and consuming facts through the shared bus rather than calling peers directly](/imgs/blogs/event-driven-architecture-events-commands-documents-5.webp)

The figure above shows the steady state you are aiming for: a set of services that each emit and consume facts through a shared bus, none of them calling each other directly. The order service emits `OrderPlaced`; payment reacts and emits `PaymentOK`; shipping reacts to `PaymentOK`; an audit service consumes *everything* for compliance; email consumes the facts it cares about. No service holds another's address. No service orchestrates the others. The flow lives in the subscriptions. That is event-driven architecture done deliberately — and you can see, looking at it, both the beauty (add a service, change nothing else) and the danger (where, exactly, does the order flow live?).

## Event notification fan-out in one picture

It is worth pausing on the cleanest possible illustration of why event intent decouples, because it is the image to keep in your head when you are tempted to reach for a command. One producer emits one fact. Many independent reactors consume it. The producer named none of them.

![A graph showing the order service emitting OrderPlaced to an event bus, which fans out to four fully independent consumers — email, analytics, loyalty, and fraud — none of which the producer knows about](/imgs/blogs/event-driven-architecture-events-commands-documents-3.webp)

In the figure above, the order service emits `OrderPlaced` to the bus, and four consumers — email, analytics, loyalty, fraud — each react on their own. The order service's code does not mention email, analytics, loyalty, or fraud. It does not know they exist. Tomorrow you add a fifth consumer (a partner webhook) and a sixth (a data-lake sink); the order service still does not change. This is the fan-out that intent makes possible. The same plumbing carrying a command would force the producer to name each recipient, and every new recipient would mean editing the producer. The difference between those two worlds is entirely the intent of the message, not the broker, not the topic, not the wire format. Get the intent right and the decoupling follows; get it wrong and no amount of infrastructure will save you.

## Case studies and war stories

### The disguised-command monolith-in-disguise

A team I worked with migrated a monolith to microservices and proudly went "event-driven" on Kafka. A year later, deploys were *more* coordinated than in the monolith. The post-mortem was illuminating: nearly every "event" was an imperative command. The checkout service published `CreateInvoice`, `ReserveStock`, `SendConfirmation`, and `NotifyWarehouse` — four commands addressed to four services, all from checkout. Checkout *was* the orchestrator; it had simply moved its function calls onto Kafka topics. Adding a new step meant editing checkout. Changing invoice logic meant coordinating with checkout's team because checkout decided *when* invoices were created. The fix was a months-long reframing: checkout was reduced to emitting one fact, `OrderConfirmed`, and each downstream service was given ownership of its own reaction. The lesson the team learned the hard way: *the broker does not decouple you; the intent does.* They had bought a Ferrari and driven it in first gear.

### The thundering herd from a thin event

An e-commerce platform used event notification for product changes: `ProductChanged { sku }`, a tiny event, and consumers called back to the product service for details. It worked beautifully until a merchant ran a bulk price update on ten thousand products. Ten thousand `ProductChanged` events fired in seconds. Forty consumers each received all of them and each called back to fetch the product — *four hundred thousand synchronized read requests* slamming the product service's API in a few seconds. The product service fell over, which made the callbacks time out, which made consumers retry, which made it worse. The thin-event design had hidden a fan-amplification factor of forty inside an innocuous-looking notification. The fix was to switch the high-volume product-change path to event-carried state transfer — ship the product data in the event so consumers never call back — which removed the callback storm entirely at the cost of fatter events, a trade they were happy to make once they understood it. The lesson: *a thin event's runtime coupling to the source API is invisible until the day it is not.*

### The silent flow break

A payments company ran a choreographed settlement flow. A routine refactor of the ledger service changed an internal method and, as a side effect, stopped emitting `LedgerEntryPosted` in one code path. The ledger service's tests passed — it was not testing the event emission for that path. Code review passed — the reviewer was looking at ledger logic, not at who consumes the event. The reconciliation service, which subscribed to `LedgerEntryPosted`, simply received fewer events and silently reconciled less. The gap was discovered *eleven days later* during a monthly audit, by which point a meaningful number of transactions had gone unreconciled. There was no error anywhere. No stack trace, no exception, no failed request — just a missing event nobody noticed, because the dependency lived in a different service's subscription. The lesson, learned expensively: *in a choreographed system, you can break a flow without any signal at the change site, which is why contract tests on event emission and end-to-end flow monitoring are not luxuries.* After this, they added consumer-side alerts on expected event rates — if `LedgerEntryPosted` volume drops anomalously, page someone — which would have caught it in minutes instead of days.

### The eventual-consistency support nightmare

A SaaS product made user permission changes event-driven for resilience: the admin service emitted `PermissionsChanged`, and each downstream service updated its local permission cache from the event. Resilient, fast, decoupled. But the convergence window was sometimes several seconds, and the support flow went like this: an admin revokes a user's access, immediately tells the user "you're locked out now," the user refreshes — and is *still in*, because their session service has not processed the `PermissionsChanged` event yet. Support tickets piled up: "I revoked access but they can still log in!" The system was working exactly as designed; the *expectation* was wrong. The product had asserted an immediate guarantee on top of an eventually consistent mechanism. The fix was partly technical (make the critical path — session invalidation — synchronous, while leaving the non-critical caches eventually consistent) and partly UX (the admin UI now says "access will be revoked within a few seconds" instead of "access revoked"). The lesson: *eventual consistency is fine until you make a promise the convergence window cannot keep; know which of your reads must be strongly consistent and do not make those event-driven.*

## When to reach for event-driven architecture (and when not to)

**Reach for event-driven architecture when** you have multiple consumers interested in the same facts and that set of consumers changes over time — this is the scenario EDA is built for, where the additive "just subscribe" property pays off enormously. Reach for it when you need services to remain available during each other's outages, because temporal decoupling lets the system absorb downstream failures gracefully. Reach for it when you want teams to move independently — spatial and data decoupling let teams deploy their services without coordinating, which is the organizational payoff that often matters more than the technical one. Reach for it when the natural shape of your domain is "things happen and other things react," which describes a huge swath of business processes. And reach for it when load leveling matters — buffering spikes in a durable log so downstream consumers process at their own pace, the core benefit covered in the [load-leveling post](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling).

**Be cautious or avoid it when** you need strong, immediate consistency on a read — if a user action must be instantly and globally reflected (a bank balance, a permission revocation, a uniqueness check), making that path event-driven introduces a convergence window that can be a correctness or security problem; keep those paths synchronous. Avoid it when the flow is genuinely a simple linear sequence with exactly one consumer per step and no prospect of new reactions — that is a command pipeline, and modeling it as broadcast events adds indirection for no decoupling benefit; just send the commands. Be wary when your team lacks the observability maturity to operate it — without distributed tracing, correlation ids, dead-letter queues, and lag monitoring, a choreographed system *will* eventually produce a 3 a.m. incident that takes hours to diagnose, and you should build that tooling before, not after. And reconsider when the cognitive cost outweighs the benefit — for a small system with a small team and a stable, well-understood flow, a synchronous monolith you can read top-to-bottom may be genuinely easier to operate than a decoupled system whose flow lives nowhere.

The decision is rarely "all events" or "all requests." Mature systems are *hybrids*: synchronous request-reply for the operations that need an immediate answer and strong consistency, asynchronous commands for work that should be load-leveled and retried, and broadcast events for facts that an open-ended set of consumers reacts to. The skill is choosing the right intent for each interaction, which is exactly what the event-versus-command-versus-document distinction equips you to do.

## Key takeaways

- **Intent, not transport, decouples your services.** An event is a fact about the past (fire-and-forget, zero-to-many consumers, producer is blind to them); a command is an instruction to exactly one handler (sender expects it done); a document is pure data. The broker is irrelevant to which one you have.
- **A disguised command — an imperative message published as an event — recouples everything.** If the producer knows about a downstream behavior or names a recipient, you have a command wearing an event's clothes, and the decoupling you think you bought never happened.
- **The litmus test for an event:** if nobody listened, would the sender care? If no, it is an event. If adding a consumer requires editing the producer, it was never an event.
- **Event notification (thin) versus event-carried state transfer (fat) is the core payload tradeoff.** Thin events keep payloads small but force runtime callbacks and risk thundering herds; fat events remove the callback and enable local read models at the cost of data duplication and staleness.
- **Choreography makes flows emergent and additive.** Adding a new reaction means writing a new subscriber and modifying zero existing services — the autonomy dividend — but the end-to-end flow then lives nowhere as a single artifact.
- **Eventual consistency is the price, not a defect.** You trade loud request-time errors for quiet eventual-consistency bugs that live inside the convergence window; design reads to tolerate staleness and keep strongly-consistent paths synchronous.
- **The flow is implicit and debugging spans services.** There is no stack trace across a choreographed flow; budget distributed tracing, correlation ids, dead-letter queues, and lag monitoring from day one, not after the first incident.
- **Name events as past-tense business facts, version the schema compatibly, and make consumers idempotent and order-tolerant.** These are baseline correctness requirements under at-least-once, asynchronous delivery, not optional polish.

## Further reading

- [What is a message queue: async decoupling and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) — the foundational "why async" post this one builds on.
- [Queue vs Pub/Sub vs Log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) — the transport primitives that carry events, commands, and documents.
- [Saga pattern: orchestration vs choreography](/blog/software-development/message-queue/saga-pattern-orchestration-vs-choreography) — when to pull a central orchestrator back into choreographed flows for transactional rollback.
- [Event sourcing and CQRS with an event log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log) — taking events to their conclusion by making the log the source of truth.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the non-negotiable consumer discipline for asynchronous delivery.
- [Transactional outbox pattern for reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — solving the dual-write problem so a state change and its event are atomic.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — event-carried state transfer at the database level.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the formal grounding for the consistency trade you make going event-driven.
- Martin Fowler, "What do you mean by 'Event-Driven'?" and "Focusing on Events" — the canonical articulation of the notification-versus-state-transfer distinction.

---
title: "Designing a Complete Microservices System End to End: The ShopFast Capstone"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The grand finale: design the entire ShopFast platform as a senior would in a real design review — requirements and scale, service decomposition, sync versus async, data and sagas, resilience, observability, deployment, security, and scale — then stress-test the whole thing at 10x, estimate the bill, and reflect on what is over-engineered."
tags:
  [
    "microservices",
    "system-design",
    "software-architecture",
    "distributed-systems",
    "saga",
    "kubernetes",
    "backend",
    "scalability",
    "observability",
    "capstone",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/designing-a-complete-microservices-system-end-to-end-1.webp"
---

It is 10 a.m. on a Tuesday and you are standing at a whiteboard. Six engineers are in the room, two more are dialed in, and your VP of Engineering is leaning against the back wall with arms folded. The brief on the slide behind you says: "ShopFast — re-platform for 10x growth. Design review. 90 minutes." ShopFast is the food-delivery-meets-e-commerce platform we have used as the running example through this entire series. Today you have to put the whole thing together: take it from a hand-wavy "we'll use microservices" into an architecture you can defend decision by decision, name every trade-off out loud before someone else does, and survive the stress questions — "what happens on Black Friday?", "what happens when the payment provider goes down?", "is this over-engineered?" This post is that design review, written out in full.

This is the capstone of the journey. Every prior post in this series taught one piece in depth: how to draw a service boundary, how to run a saga, how to wrap a remote call so it survives a slow dependency, how to ship without taking the fleet down. A junior who has read those posts knows the pieces. A senior is the person who can sit at that whiteboard and assemble the pieces into one coherent system, *and* know which pieces to leave on the table because the simpler thing is correct. The hardest skill in this field is not knowing more patterns. It is knowing which patterns this particular system, at this particular scale, with this particular team, actually needs — and having the spine to say "we don't need that yet" when the room wants to gold-plate.

So we will reason the way a real design review reasons, in the order the decisions actually depend on each other. Requirements and scale first, because everything downstream is justified by a number. Then the honest gate: should this even be microservices? Then decomposition, communication, data, resilience, observability, deployment, security, and scale — pulling forward the deep-dive for each topic so you can go re-read the mechanism when you need it. Then we stress-test the whole design against four concrete disasters, put a dollar figure on it, trace the evolution path that got us here from a monolith, and finish with the most senior section of all: what I would do differently, and what in this very design is over-engineered. By the end you should be able to walk into your own version of that room and run the review yourself.

![A branching architecture diagram of the ShopFast platform showing an edge gateway fanning into catalog and pricing, order and cart, identity, payment, inventory and shipping, all meeting on a Kafka event bus that feeds a notification service](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-1.webp)

That diagram is the destination. By the end of this post every box, every arrow, and every label on it will be a decision you can defend. Let us earn it.

## Step 1: Requirements and scale — the numbers that justify everything

The single biggest mistake in a design review is starting with the architecture. You start with the requirements, because the architecture is just the cheapest correct answer to the requirements, and you cannot tell whether an answer is correct until you know the question. A senior spends the first ten minutes of a 90-minute review refusing to draw a single box, and instead writing numbers on the board. This is the discipline the [back-of-the-envelope estimation](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) post drills, and it is the difference between a design and a daydream.

So, ShopFast. Let us pin the functional requirements first — what the system must *do*:

- Browse a catalog of products and restaurants, search and filter, see live pricing and availability.
- Add to a cart, check out, pay, and get an order confirmed.
- Track an order from "confirmed" through "preparing", "out for delivery", "delivered".
- Receive notifications (push, email, SMS) at each step.
- Manage an account, addresses, payment methods, and order history.

Now the non-functional requirements — the ones that actually shape the architecture, because *how well* and *how much* matter far more than *what*:

- **Scale.** 20 million registered users, 2 million daily active users. Average steady-state traffic of about 5,000 requests per second (RPS) across all services, with a normal daily peak around 15,000 RPS at the dinner rush. Black Friday and major promotions push that to a planned **10x of the daily peak — call it 150,000 RPS at the very top**, sustained for a few hours.
- **Latency SLOs.** The browse and search path must serve p99 under 200ms — this is where users abandon. The checkout path is allowed to be slower because the user expects it to "process": p99 under 1.5 seconds end to end, p50 under 400ms. Order tracking can be eventually consistent and serve from a read model.
- **Availability.** The checkout path targets 99.95% (about 4.4 hours of downtime a year). Browse targets 99.9%. Notifications are best-effort — 99% is fine, and a delayed notification is annoying, not catastrophic. **Not all paths deserve the same number, and pretending they do is how you spend a fortune protecting the parts that do not matter.**
- **Data volume.** About 2 million orders a day at peak. Each order with its line items and events is roughly 4 KB. That is 8 GB of new order data a day, ~3 TB a year before compression. The product catalog is small — a few million SKUs, single-digit GB. The event log (every state change) is the big one: tens of millions of events a day.
- **Consistency.** Money must be correct: no double charges, no lost charges, every charge reconcilable against an order. Inventory must never oversell a limited item. Everything else — recommendation freshness, order-history lag, notification timing — can be eventually consistent.
- **Growth.** Planning for 3x users over two years and expansion into a second geographic region in year one.

Write those down and the architecture starts designing itself. A p99-under-200ms read path that is read-heavy screams *caching and read replicas*. "No double charge, ever" screams *idempotency and a saga, not a naive RPC chain*. "150k RPS for three hours, twice a year" screams *elastic autoscaling, not 150k-RPS standing capacity you pay for 365 days*. "Second region in year one" is a flag you plant now so you do not paint yourself into a single-region corner, but it is *not* a reason to build full active-active multi-region on day one. We will come back to every one of these.

### The honest gate: should ShopFast even be microservices?

Before we decompose anything, we run the gate from [what microservices are and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them) and [microservices anti-patterns](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith). Microservices are not a goal. They are a cost you pay to buy a specific thing: **independent deployability and independent scaling along team and capability boundaries.** If you do not need that thing, you are paying the cost for nothing, and the cost is real — network calls instead of function calls, distributed transactions instead of `BEGIN/COMMIT`, distributed tracing instead of a stack trace, an entire platform team instead of a `git push`.

Does ShopFast need it? Walk the test honestly:

- **Team count.** ShopFast has roughly 50 engineers across 7 product teams. That is past the point where one monolith codebase creates constant merge contention and deploy coupling. Above ~5 teams, the org pressure for independent deploys is real. This is a genuine signal, and it is the Conway's-law argument we will return to.
- **Differential scaling.** The browse path does 30x the traffic of the checkout path and has completely different hardware needs (cache-bound versus transaction-bound). Scaling them as one unit means over-provisioning the cheap part to feed the expensive part. This is a genuine signal.
- **Differential availability and blast radius.** A bug in recommendations should never be able to take down checkout. In one monolith, a memory leak in any module can OOM the whole process. This is a genuine signal.
- **Differential compliance.** Payment touches PCI scope; isolating it into its own service shrinks the audit surface dramatically. Genuine signal.

So yes — ShopFast, at this scale and team size, clears the gate. But here is the senior move, and it is the most important sentence in this whole section: **we clear the gate for a handful of services, not for fifty.** The failure mode is not "monolith versus microservices." It is "right-sized services versus a distributed monolith of nano-services that share a database and deploy together anyway." We will decompose into roughly **nine services**, and we will defend keeping it that small. If ShopFast were a 4-person startup with 200 orders a day, the correct answer would be a [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith), and we would say so to the VP without flinching. It is not, so we proceed.

## Step 2: Service decomposition — bounded contexts to a service map

Now we draw boxes — but we draw them along the right seams. The wrong way to decompose is by technical layer (a "controllers service", a "database service") or by noun ("a User service" that everything depends on). The right way is by **bounded context**: a slice of the business with its own ubiquitous language, its own data, and ideally its own team. This is the core of [service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design), and it is where most microservice architectures live or die. Get the boundaries right and the rest is plumbing; get them wrong and you spend the next three years doing distributed surgery on chatty, coupled services.

Let me show the decomposition as a taxonomy, grouping the nine services under the three core business subdomains they belong to.

![A tree diagram grouping the nine ShopFast services under three business domains: shopping owned by the catalog team, checkout owned by the checkout team, and fulfilment owned by the operations team](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-7.webp)

Here is the service map, each with a one-line charter:

| Service | Owns (bounded context) | Owning team | Datastore |
|---|---|---|---|
| **Catalog** | Products, restaurants, menus, descriptions, search index | Catalog | Postgres + Elasticsearch |
| **Pricing** | Prices, promotions, surge/dynamic pricing rules | Catalog | Postgres + Redis |
| **Cart** | The in-progress, not-yet-ordered basket | Checkout | Redis (DynamoDB-backed) |
| **Order** | The order aggregate; orchestrates checkout saga | Checkout | Postgres |
| **Payment** | Charges, refunds, payment methods (PCI scope) | Checkout | Postgres (encrypted) |
| **Inventory** | Stock levels, reservations, holds | Operations | Postgres (sharded) |
| **Shipping** | Delivery dispatch, courier assignment, tracking | Operations | Postgres + PostGIS |
| **User/Identity** | Accounts, auth, addresses, profile | Platform | Postgres |
| **Notification** | Push/email/SMS fan-out | Platform | DynamoDB (queue-fed) |

It is worth capturing that service map as a single declarative artifact the platform can read — a service catalog. This is the source of truth that drives ownership routing, on-call paging, and dependency auditing; when an alert fires, the platform looks up the owning team here:

```yaml
# service-catalog.yaml — one entry per service, the org's source of truth
services:
  - name: order
    domain: checkout
    team: checkout
    tier: critical            # drives SLO and on-call priority
    datastore: postgres
    sync_deps: [payment, inventory]      # synchronous gRPC callees
    publishes: [OrderPlaced, OrderCancelled, OrderConfirmed]
    consumes: []
    slo: { availability: "99.95", p99_ms: 1500 }
  - name: payment
    domain: checkout
    team: checkout
    tier: critical
    datastore: postgres-encrypted
    pci_scope: true           # isolates the audit surface
    sync_deps: []
    external_deps: [psp-primary, psp-backup]   # third-party processors
    publishes: [PaymentCharged, PaymentRefunded]
    slo: { availability: "99.95", p99_ms: 800 }
  - name: notification
    domain: platform
    team: platform
    tier: best-effort         # async consumer; can lag without harm
    datastore: dynamodb
    sync_deps: []
    consumes: [OrderConfirmed, OrderCancelled, ShipmentDispatched]
    slo: { availability: "99", p99_ms: 5000 }
```

That file is small but it encodes a lot: which service is in PCI scope, which depend synchronously on which (the dependency graph you audit for cycles — a cycle is a distributed-monolith smell), what each publishes and consumes, and the per-service SLO tier. Critically, Notification's `tier: best-effort` is *declared*, which tells the alerting system not to page anyone at 3 a.m. for a Notification blip.

Notice the deliberate choices. **Cart and Order are separate services** even though they feel related: a cart is high-churn, ephemeral, read-write-heavy, and tolerant of loss (you can ask the user to re-add an item); an order is durable, money-touching, and must never be lost. Different consistency, different store, different failure tolerance — different service. **Pricing is separate from Catalog** because prices change on a completely different cadence than product descriptions and are owned by a different part of the business (revenue/growth versus content), and because surge pricing is computationally heavy and you do not want it competing for resources with catalog reads. These are exactly the seams DDD's bounded contexts reveal.

And notice what is *not* a service. There is no "Logging service", no "Validation service", no "Email service" separate from Notification, no "Database service." Those are the nano-service smell — they create chatty cross-service calls for what should be a library, a sidecar, or a single capability. Cutting along business capability, not technical function, is the rule.

### Aligning services to teams — Conway's law, applied

The reason the table has an "owning team" column is not bookkeeping. By [Conway's law](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices), your system's structure will mirror your org's communication structure whether you plan it or not, so you plan it. Each service has **exactly one owning team**; no service is co-owned. The Checkout team owns Cart, Order, and Payment — the three services that participate in the single most important transaction in the business, so the people who reason about checkout end-to-end sit together. The Catalog team owns Catalog and Pricing. The Operations team owns Inventory and Shipping (the physical-world services). A thin Platform team owns Identity and Notification *and* the paved road — the shared CI/CD, the mesh, the observability stack — so the stream-aligned teams do not each reinvent it.

Nine services, three core domains, a platform team underneath. That is a deliberately *small* number for 50 engineers, and that restraint is the point. Monzo runs 1,500+ services with a few hundred engineers; that works for them because of an extraordinary platform investment, and copying it without that investment is how you get a 1,500-service mess that three teams cannot operate. Start small, split only under pressure.

## Step 3: Communication — what is synchronous, what is an event

Now the arrows. The single most consequential communication decision in a microservices system is **sync versus async per interaction**, because it determines your coupling and your failure modes. The fallacies of distributed computing — the network is reliable, latency is zero, bandwidth is infinite — are the subject of [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), and the whole reason this decision is hard is that every arrow you draw is a place the network can fail.

The rule of thumb a senior carries into the room: **use a synchronous call when the caller genuinely cannot proceed without the answer right now; use an asynchronous event when the caller is announcing that something happened and does not need anyone to do anything before it continues.** Read that twice. "Charge this card and tell me if it worked" is synchronous — the checkout cannot proceed without the answer. "An order was placed" is an event — the order service does not need notifications or analytics to finish before it returns to the user.

Apply it to ShopFast:

- **Client → Gateway → Catalog/Search:** synchronous. The user is waiting for products. REST/JSON over HTTP at the edge for browser/mobile friendliness, served from cache and read replicas.
- **Order → Payment (charge):** synchronous gRPC. The saga step blocks on the result; we need the charge outcome to decide the next step. gRPC because it is an internal, high-throughput, strongly-typed call where REST's overhead and looseness buy us nothing. The full case for gRPC over REST or GraphQL on internal calls is in [REST vs gRPC vs GraphQL for service APIs](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis).
- **Order → Inventory (reserve):** synchronous gRPC. The saga must know the reservation succeeded before charging.
- **Order → "order placed" / "order confirmed":** asynchronous events on the bus. Shipping, Notification, Analytics, and the search-popularity updater all *subscribe*; the Order service does not call them and does not wait for them. This is the heart of [event-driven microservices](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration).
- **Inventory → "stock low" / "out of stock":** asynchronous events that Pricing (for surge) and Catalog (for availability badges) consume.

The internal contract between Order and Payment is a protobuf gRPC service — strongly typed, versioned, and the single source of truth both teams generate clients from. Notice the idempotency key is a first-class field, not an afterthought:

```protobuf
syntax = "proto3";
package shopfast.payment.v1;

service PaymentService {
  // Charge is idempotent: the same idempotency_key returns the same result,
  // so a client retry after a timeout never double-charges.
  rpc Charge(ChargeRequest) returns (ChargeResponse);
  rpc Refund(RefundRequest) returns (RefundResponse);
}

message ChargeRequest {
  string order_id        = 1;
  string customer_id     = 2;
  int64  amount_minor    = 3;  // cents; never use float for money
  string currency        = 4;  // ISO 4217, e.g. "USD"
  string payment_method  = 5;  // token; raw card data never crosses here
  string idempotency_key = 6;  // dedupe key, required
}

message ChargeResponse {
  string charge_id   = 1;
  Status status      = 2;      // AUTHORIZED, DECLINED, PENDING
  string decline_code = 3;     // populated only on DECLINED
}

enum Status { STATUS_UNSPECIFIED = 0; AUTHORIZED = 1; DECLINED = 2; PENDING = 3; }
```

Money is `int64` minor units, never a float — a floating-point cent is how you end up off by a penny across a billion transactions. Raw card data never appears in this contract; only a tokenized `payment_method` crosses the wire, keeping the Order service out of PCI scope. The contract is versioned in its package name (`v1`) so we can evolve it without breaking callers, the discipline from [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing).

Why does this split matter so much? Because **synchronous calls couple availability multiplicatively.** If checkout synchronously called Notification, and Notification went down, checkout would go down — for a feature the user does not even need to complete their purchase. By making Notification an event subscriber, Notification can be down for an hour and checkout never notices; the events queue up and get delivered when it recovers. You have decoupled the availability of an optional capability from a critical one. That single design choice is worth more than any amount of clever code.

### The edge: API gateway and BFF

Every external request enters through one place: the [API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend). The gateway terminates TLS, authenticates the request (validates the JWT), enforces global rate limits, and routes to the right service. We layer a thin **BFF per client type** — a mobile BFF and a web BFF — because the mobile app wants small, aggregated payloads (one call returns "home screen" = nearby restaurants + active order + promos) while the web app wants finer-grained calls. The BFF does the fan-out and aggregation so the client does not, and so the public contract is tailored per client without polluting the underlying services.

Here is the request path as layers, each adding exactly one guarantee before any business logic runs:

![A vertical stack showing the six layers a checkout request descends through: CDN and edge, API gateway and BFF, service mesh, the order service, resilience wrappers, and finally the Postgres and cache datastore](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-4.webp)

That layering is not decoration. Each layer has one job and fails independently: the CDN absorbs static and bot traffic so it never reaches your services; the gateway rejects unauthenticated and over-limit requests so they never reach business logic; the mesh handles mTLS and connection pooling so services do not; the resilience wrappers handle timeouts and breakers so a slow dependency does not block a thread forever. A request that reaches your Postgres has passed five gates, each of which shed a category of load or failure.

### Checkout as a saga

The most important flow in the system is checkout, and it spans Order, Inventory, Payment, and Shipping — four services, four databases (one of them an external payment processor). There is no `BEGIN/COMMIT` that spans those. So checkout is a **saga**: a sequence of local transactions, each with a compensating transaction that semantically reverses it on failure. We choose **orchestration over choreography** here — the Order service is an explicit orchestrator that drives each step — because checkout is a critical, money-touching flow where you want one place that knows the whole state machine and where you can see exactly where a stuck order is. (Choreography, where services react to each other's events with no central brain, is great for loosely-coupled fan-out but terrible for a flow you need to debug at 3 a.m.) The full mechanics — step ordering, isolation anomalies, idempotent compensations — are in [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice) and the deeper [saga mechanism deep-dive](/blog/software-development/database/saga-pattern-distributed-transactions).

![A flow graph of the checkout saga: place order, reserve inventory, charge payment, then a branch where a charged path creates a shipment and confirms the order, and a declined path runs the release-inventory compensation and ends in a cancelled order](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-2.webp)

The forward path is reserve → charge → ship → confirmed. The backward branch is the senior detail: when `chargePayment` is declined, the saga runs only the compensations for steps that actually committed — here, `releaseInventory` — and ends in a clean cancel. If `createShipment` had failed instead, the saga would compensate *both* `chargePayment` (refund) and `reserveInventory` (release), in reverse order. The compensation chain is always the reverse-suffix of the steps that succeeded.

Here is the orchestrator as an explicit, durable state machine. The key properties: every step records its outcome so a crash-and-recover resumes from the right place, every forward call carries the saga's idempotency key, and a failure unwinds only the committed steps in reverse.

```go
// CheckoutSaga drives the place-order flow. Each step is a local txn in a
// remote service; each has a compensation. State is persisted after every
// transition so a process crash resumes from the last durable step.
func (s *Orchestrator) Run(ctx context.Context, saga *Saga) error {
    steps := []Step{
        {Name: "reserve", Do: s.reserveInventory, Undo: s.releaseInventory},
        {Name: "charge",  Do: s.chargePayment,    Undo: s.refundPayment},
        {Name: "ship",    Do: s.createShipment,   Undo: nil}, // async, no undo
    }

    for i, step := range steps {
        // Idempotent: the key dedupes a retried call so we never act twice.
        err := step.Do(ctx, saga, saga.IdempotencyKey)
        if err == nil {
            saga.Completed = append(saga.Completed, step.Name)
            s.persist(ctx, saga) // durable checkpoint after each success
            continue
        }

        // Step i failed. Compensate i-1, i-2, ... 0 in reverse order.
        log.Warn("saga step failed, compensating", "step", step.Name, "err", err)
        for j := i - 1; j >= 0; j-- {
            if steps[j].Undo != nil {
                if uerr := steps[j].Undo(ctx, saga); uerr != nil {
                    // Compensation MUST eventually succeed: retry with backoff,
                    // and if it cannot, page a human. Never silently drop it.
                    s.enqueueRetry(ctx, saga, steps[j].Name)
                }
            }
        }
        saga.State = "cancelled"
        s.persist(ctx, saga)
        return fmt.Errorf("checkout failed at %s: %w", step.Name, err)
    }
    saga.State = "confirmed"
    s.persist(ctx, saga)
    return nil
}
```

The two senior details: the `Undo` for `ship` is `nil` because shipment is created via an async event after confirmation, so there is nothing to compensate on the synchronous path. And compensation is treated as something that *must* eventually succeed — if a refund fails, we do not drop it; we retry it and, failing that, page a human, because an uncompensated charge is real money owed to a customer. We will see the orchestrator code's resilience wrapper shortly.

## Step 4: Data — database per service, polyglot stores, outbox

Now the rule that actually defines microservices, more than any other: **[database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices).** Each service owns its data privately. No other service reads or writes its tables. The only way to its data is through its API or its events. This is non-negotiable, because the moment two services share a database, you no longer have two services — you have a [distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), with all the coupling of a monolith and all the operational pain of distribution, which is the worst of both worlds.

The upside of private data is that each service can choose the *right* store for its access pattern — **polyglot persistence.** The choices for ShopFast, and the reasoning:

| Service | Store | Why this store |
|---|---|---|
| Catalog | Postgres + Elasticsearch | Relational source of truth; ES for full-text search and faceting |
| Pricing | Postgres + Redis | Rules in Postgres; computed prices cached hot in Redis |
| Cart | Redis + DynamoDB | Sub-ms reads; DynamoDB for durability and TTL eviction |
| Order | Postgres | Strong consistency, relational integrity for the order aggregate |
| Payment | Postgres (encrypted) | ACID ledger; column encryption for PCI |
| Inventory | Postgres (sharded by SKU) | Strong consistency for stock counts; sharded for write throughput |
| Shipping | Postgres + PostGIS | Geospatial queries for courier dispatch |
| Notification | DynamoDB | Write-heavy, key-value, simple access pattern, auto-scaling |

The cost you pay for private data is that you **cannot do a join across services.** You cannot write `SELECT * FROM orders JOIN users JOIN products`. So you handle cross-service reads two ways. For a screen that needs data from several services, the **BFF fans out** and composes (parallel calls, then merge). For a read that is hot and would be expensive to compose every time — like "my order history with product names and images" — you build a **read model / CQRS view**: the Order service consumes product and user events and maintains a denormalized, query-optimized copy of exactly the fields that screen needs. The mechanism and its sharp edges are in [event sourcing and CQRS in microservices](/blog/software-development/microservices/event-sourcing-and-cqrs-in-microservices).

### The outbox: how the saga publishes reliably

Here is a trap that catches every team once. The Order service commits the order to its Postgres, then publishes "order placed" to Kafka. What if the process crashes *between* the commit and the publish? The order exists but the event was never sent — Shipping never hears about it, the food never gets cooked, and you have a paying customer with a silent order. You cannot wrap the Postgres commit and the Kafka publish in one transaction; they are two different systems.

The fix is the **[transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing)**: in the *same* local transaction that writes the order, you also insert the event into an `outbox` table in the *same* database. Now the order and the intent-to-publish commit atomically. A separate relay process (or change-data-capture tailing the WAL, as in [CDC and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern)) reads the outbox and publishes to Kafka, marking rows as sent. If the relay crashes, it re-reads unsent rows and republishes — which means events are delivered **at-least-once**, so every consumer must be idempotent. This is the single most important reliability pattern in the data layer, and skipping it is how you get phantom orders.

Here is the write-and-outbox in one transaction, then the event schema the relay publishes. The event is versioned, carries a unique `event_id` for consumer-side dedupe, and is a *fact about the past* ("an order was placed"), not a command:

```sql
-- Both rows commit atomically, or neither does. No dual-write window.
BEGIN;
INSERT INTO orders (id, customer_id, total_minor, status)
  VALUES ('ord_8a1f', 'cus_42', 4250, 'CONFIRMED');
INSERT INTO outbox (id, aggregate_id, type, payload, created_at)
  VALUES ('evt_3c9d', 'ord_8a1f', 'OrderPlaced',
          '{"order_id":"ord_8a1f","customer_id":"cus_42","total_minor":4250}',
          now());
COMMIT;
```

```json
{
  "event_id": "evt_3c9d",
  "type": "OrderPlaced",
  "version": "1.0",
  "occurred_at": "2026-06-15T18:42:07Z",
  "aggregate_id": "ord_8a1f",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "data": {
    "order_id": "ord_8a1f",
    "customer_id": "cus_42",
    "total_minor": 4250,
    "currency": "USD",
    "items": [{ "sku": "PIZZA-MARG", "qty": 1, "price_minor": 4250 }]
  }
}
```

Three things make this event production-grade. The `event_id` lets Shipping, Notification, and the analytics consumer each dedupe independently, so at-least-once delivery is safe. The `version` field lets us evolve the schema without breaking old consumers. And the `trace_id` carries the distributed trace across the async boundary, so a single checkout's trace stays connected even after it crosses the event bus — without it, async hops are invisible black holes in your traces.

### Eventual consistency in the UX

Database-per-service plus async events means parts of the system are **eventually consistent**, and the senior move is to design the *user experience* around that honestly rather than pretend it is instant. When a user places an order, the Order service confirms it synchronously (they see "Order confirmed!"), but the order-history read model might lag by a few hundred milliseconds. So the order page reads from the Order service directly for the just-placed order, and from the read model for the history list — read-your-own-writes for the thing the user just did, eventual consistency for the rest. The full set of UX patterns for hiding eventual consistency is in [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice), and the formal models behind it in [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

#### Worked example: the checkout latency budget across services

Let me make the saga concrete with a latency budget, because "p99 under 1.5s" is meaningless until you allocate it across hops. A checkout request must fit its work inside the budget *with margin*, because p99s compound — the p99 of a chain is worse than the worst single p99 when calls are sequential.

The synchronous critical path for checkout, sequentially:

| Step | Operation | Budget (p99) |
|---|---|---|
| 1 | Gateway: TLS, authn, route | 15 ms |
| 2 | Order svc: validate cart, open saga | 25 ms |
| 3 | → Inventory: reserve stock (gRPC + Postgres write) | 80 ms |
| 4 | → Payment: charge via external PSP | 600 ms |
| 5 | Order svc: persist order + outbox (one txn) | 60 ms |
| 6 | Gateway: serialize response | 10 ms |
| | **Sum of the synchronous path** | **790 ms** |

Shipping creation and all notifications happen *asynchronously* off the bus, so they are **not** in this budget — that is precisely why we made them events. The synchronous total is ~790 ms p99, comfortably inside the 1.5s SLO with ~700 ms of headroom for the long tail and retries. The dominant cost is the external payment call at 600 ms — it is the one step we do not control, which is exactly why it gets the biggest timeout, the breaker, and the most attention. If we had naively made Shipping and Notification synchronous too (say +200 ms and +150 ms), the path would be ~1,140 ms p99 and one slow notification provider would blow the SLO. The async split bought us the budget. That is a design decision you can defend with a number.

## Step 5: Resilience — keeping the critical path alive

Everything above assumes the happy path. Production is not the happy path. The eighth fallacy is "the network is homogeneous"; the real one is "everything is always up." A microservices system has dozens of network calls per user action, and at scale *something* is always degraded. Resilience is the discipline of making one degraded dependency a *partial* problem instead of a *total* outage. The full toolkit — timeouts, retries, circuit breakers, bulkheads — is in [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads).

The contrast that matters is the naive call versus the hardened call.

![A before-and-after diagram contrasting a naive critical path with no timeout, a retry storm, and a hard dependency against a hardened path with a budgeted timeout, a circuit breaker with jitter, and degraded optional features so checkout still works](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-9.webp)

The rules ShopFast applies on every critical-path call:

- **Timeouts everywhere, budgeted.** No call without a deadline. The timeout is derived from the latency budget above, not guessed. The Payment call gets 800 ms (above its 600 ms p99, below the point where the user gives up). A call with no timeout is a thread that blocks forever, and blocked threads are how one slow service drains the gateway's connection pool and takes down everything.
- **Retries with jitter, but only on idempotent operations and only with a budget.** Retry a timed-out inventory read, yes. Retry a payment charge blindly? Absolutely not — you will double-charge. Retries are capped (2 attempts), use exponential backoff with jitter to avoid synchronized retry storms, and respect a per-request retry budget so retries can never amplify load more than ~10%.
- **Circuit breakers.** When Payment's error rate crosses a threshold, the breaker opens and we *fail fast* instead of piling requests onto a dying service. This is what stops a cascade — the breaker turns "wait 800 ms then fail" into "fail in 1 ms", freeing threads instantly.
- **Bulkheads.** The thread pool / connection pool for calls to Payment is isolated from the pool for calls to Inventory. If Payment goes slow and saturates its pool, Inventory calls still have their own threads. One drowning dependency cannot pull the others under.
- **Idempotency on every mutation.** Every checkout carries an idempotency key; the Order service dedupes on it so a client retry never creates two orders. Every event consumer dedupes on event id. This is the safety net under at-least-once delivery, covered in [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) and the mechanism deep-dive on [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe).
- **Graceful degradation of optional features.** This is the senior instinct. Recommendations down? Show a static "popular near you" list. Pricing surge engine slow? Fall back to base price. Reviews service down? Hide the reviews section. The checkout path *never* hard-depends on an optional feature. [Handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) is the whole playbook.
- **Rate limiting and load shedding at the edge.** The gateway sheds excess load *before* it reaches services, and prioritizes checkout over browse when capacity is tight, per [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding).
- **Health checks and self-healing.** Every service exposes liveness and readiness probes so Kubernetes restarts the dead and stops routing to the not-ready, per [health checks, readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing).

Here is the critical-path call to Payment with the whole stack applied — timeout, breaker, bulkhead, capped jittered retry — using a resilience library so the policy is declarative, not hand-rolled per call site:

```java
// Order service: the wrapped call to Payment. Every critical-path remote call
// goes through a config like this. Note: retry is safe ONLY because Charge is
// idempotent (the idempotency key dedupes on the Payment side).
CircuitBreakerConfig breaker = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)              // open if >50% of calls fail
    .slowCallRateThreshold(80)             // ...or >80% are "slow"
    .slowCallDurationThreshold(Duration.ofMillis(800))
    .waitDurationInOpenState(Duration.ofSeconds(5))   // probe again after 5s
    .build();

Bulkhead bulkhead = Bulkhead.of("payment",
    BulkheadConfig.custom().maxConcurrentCalls(50).build()); // isolated pool

Retry retry = Retry.of("payment", RetryConfig.custom()
    .maxAttempts(2)                        // 1 retry, no more
    .intervalFunction(IntervalFunction.ofExponentialRandomBackoff(100, 2.0))
    .retryOnException(e -> e instanceof TimeoutException) // never retry a DECLINE
    .build());

ChargeResponse resp = Decorators.ofSupplier(() ->
        paymentClient
            .withDeadline(Deadline.after(800, MILLISECONDS))  // hard timeout
            .charge(req))
    .withBulkhead(bulkhead)
    .withCircuitBreaker(circuitBreaker)
    .withRetry(retry)
    .withFallback(List.of(CallNotPermittedException.class),
        e -> ChargeResponse.pending())     // breaker open -> "payment pending"
    .get();
```

Read the order of the decorators carefully, because it is load-bearing: the timeout is innermost (on the raw call), then retry, then breaker, then bulkhead, then fallback. The fallback only fires when the breaker is *open* — meaning Payment is already known-bad — and it returns a graceful "pending" rather than an error the user sees as a failure. Crucially, the retry only fires on a `TimeoutException`, **never** on a `DECLINED` response, because a decline is a definitive business answer (the card was refused) and retrying it would be both useless and, if the call were not idempotent, dangerous.

## Step 6: Observability and SLOs — knowing what is happening

You cannot operate what you cannot see, and a distributed system is invisible by default. When checkout is slow, the question "which of the six hops is slow?" has no answer without instrumentation. The three pillars — traces, metrics, logs — plus SLOs are the subject of [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) and [SLOs, golden signals, and alerting](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices).

ShopFast's observability design:

- **Distributed tracing on every request.** A trace id is generated at the gateway and propagated through every hop (gRPC metadata, Kafka headers). One checkout becomes one trace with a span per service, so when the budget blows you can see *exactly* which span ate the time. OpenTelemetry is the vendor-neutral instrumentation; traces sample at 100% for errors and ~1% for success to control cost.
- **Golden signals per service.** Latency (p50/p99), traffic (RPS), errors (rate), and saturation (CPU/memory/pool utilization). These four, dashboarded per service, catch most problems.
- **SLOs with error budgets.** Checkout success-rate SLO is 99.95% over 28 days. That is an error budget of ~0.05% — about 21 minutes of "down" per month. When we have spent the budget, deploys freeze and the team focuses on reliability. When the budget is healthy, the team ships features faster and takes more risk. The error budget turns "how reliable should we be?" from an argument into a number.
- **Alert on symptoms, not causes.** Page on "checkout success rate below SLO" and "checkout p99 above 1.5s" — the things users feel — not on "CPU above 80%" which may be totally fine. Cause-based alerts are noise; symptom-based alerts are signal. A high CPU that is not hurting the SLO is not a page; it is a dashboard.
- **The on-call story.** One team owns each service's pager. Runbooks link from each alert. When [debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production), the trace id is the thread you pull — paste it into the trace UI and the whole request graph lights up.

#### Worked example: the error budget math during an incident

Suppose Payment has a bad afternoon. The PSP's p99 spikes and 2% of charges time out for 25 minutes before our breaker stabilizes things and we fail over to the backup PSP. How much budget did that burn?

Checkout SLO is 99.95% over 28 days. Total checkout requests in 28 days at, say, an average of 300 checkout RPS: 300 × 86,400 × 28 ≈ 725 million requests. The error budget is 0.05% of that ≈ **363,000 allowed failures** for the month. During the 25-minute incident, at 300 RPS that is 300 × 1,500 s = 450,000 requests, of which 2% failed = **9,000 failures**. So this incident burned 9,000 / 363,000 ≈ **2.5% of the monthly error budget.** That is a healthy, recoverable bite — annoying but not a crisis, and the number tells us we can absorb a few of these a month and still hit the SLO. If a single incident had burned 60% of the budget, *that* is the signal to freeze deploys and harden. The error budget converts a vague "was that bad?" into a precise "it cost us 2.5%, we have margin." That is how seniors talk about reliability.

## Step 7: Deployment and platform — shipping without fear

The whole point of microservices is independent deployability, so the platform must deliver exactly that. The building blocks are [containerizing microservices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices), [Kubernetes for microservices](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials), [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability), [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags), and [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management).

ShopFast's platform:

- **Containers.** Every service ships as a minimal, multi-stage Docker image (distroless base, non-root user, pinned digests). Small images deploy faster and have a smaller attack surface.
- **Kubernetes.** Services run as Deployments behind Services, with HorizontalPodAutoscalers driven by CPU and custom metrics (RPS, queue depth). Each service has resource requests/limits so the scheduler can pack pods and so one greedy pod cannot starve a node.
- **CI/CD per service.** Each service has its own pipeline and deploys independently — that is the whole reason we paid the microservices tax. A merge to a service's main branch runs tests, builds the image, and rolls it out. Forty-plus deploys a day across the fleet, none of them coordinated.
- **Canary + feature flags.** Risky changes roll out to 5% of traffic first; if the golden signals hold for 10 minutes, the rollout proceeds; if error rate climbs, it auto-rolls back. Feature flags decouple *deploy* from *release* — code ships dark and is turned on later, gradually, and can be killed instantly without a redeploy.
- **Config and secrets.** Config is environment-injected (ConfigMaps); secrets live in a secrets manager (Vault / cloud KMS), are mounted at runtime, never baked into images, never committed. We will show this in YAML with `REDACTED` placeholders.

### The mesh decision — made honestly

Do we need a [service mesh](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one)? This is exactly the kind of decision a junior says yes to reflexively ("everyone uses Istio") and a senior interrogates. A mesh gives you uniform mTLS, retries, timeouts, and traffic-shifting *without* per-service code — the sidecar handles it. The cost is real: a sidecar proxy on every pod (latency +1–3 ms per hop, memory +50–100 MB per pod), a steep operational learning curve, and a new failure mode (the control plane).

The honest answer for ShopFast at nine services: **adopt a lightweight mesh (Linkerd, not full Istio) primarily for mTLS and traffic-shifting, and turn off the features we do not need.** At nine services we are right on the boundary — we could do mTLS with a library and live without the mesh. We choose the mesh because we want uniform, code-free mTLS for the zero-trust posture below, and because we are planning for growth past 20 services where the per-service-code approach stops scaling. But we explicitly reject full Istio's complexity at this size; Linkerd is the right-sized tool. **If ShopFast had four services, the answer would be "no mesh, do mTLS in a shared library" — and we would say so.** Naming the threshold is the senior move.

Here is what a service's Kubernetes deployment looks like, with the resilience, probes, autoscaling hooks, and redacted secrets all visible:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  labels: { app: order-service, team: checkout }
spec:
  replicas: 6
  selector: { matchLabels: { app: order-service } }
  template:
    metadata:
      labels: { app: order-service }
      annotations:
        linkerd.io/inject: enabled            # mesh sidecar for mTLS
    spec:
      containers:
        - name: order
          image: registry.shopfast.io/order-service@sha256:REDACTED_DIGEST
          ports: [{ containerPort: 8080 }]
          resources:
            requests: { cpu: "500m", memory: "512Mi" }
            limits:   { cpu: "1",    memory: "1Gi" }
          livenessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet: { path: /readyz, port: 8080 }
            periodSeconds: 5
          env:
            - name: PSP_TIMEOUT_MS
              value: "800"
            - name: PAYMENT_API_KEY
              valueFrom:
                secretKeyRef: { name: payment-secrets, key: api-key }   # from Vault, REDACTED
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: order-service }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: order-service }
  minReplicas: 6
  maxReplicas: 40
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
```

That single manifest encodes a half-dozen decisions from this series: mesh injection for mTLS, requests/limits for bin-packing, liveness/readiness for self-healing, a budgeted PSP timeout from the latency analysis, secrets pulled from Vault (never baked in), and an HPA that scales 6→40 replicas for Black Friday.

## Step 8: Security — zero-trust east-west, OAuth2 north-south

Security in a microservices system splits into two problems with two different answers, and conflating them is a common mistake.

**North-south** is the traffic between the outside world and your system — users, mobile apps, partners. The answer is **OAuth2 / OIDC with JWTs**, covered in [authentication and authorization with OAuth2 and JWT](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation). The user authenticates with the Identity service, gets a short-lived access token (a signed JWT), and presents it on every request. The gateway validates the token's signature and expiry once at the edge, then propagates the verified identity (the token, or a derived internal token) to downstream services so each can make authorization decisions about *this user*. The JWT carries claims (user id, roles, scopes); services check scopes for authorization ("does this token have `orders:write`?").

**East-west** is service-to-service traffic inside the cluster. The answer is **zero-trust with mTLS**, covered in [service-to-service security with mTLS and zero-trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust). The principle: the network is not a trust boundary. Just because a request came from inside the cluster does not mean it is allowed. Every service presents a cryptographic identity (an mTLS certificate, issued and rotated by the mesh), every connection is mutually authenticated and encrypted, and authorization policies state explicitly which service may call which ("only the Order service may call Payment's `charge` method"). This is why we accepted the mesh: it makes mTLS uniform and code-free. Combined, a compromised browse service still cannot call Payment, because the policy forbids it and the certificate proves who is calling.

The PCI dimension reinforces the decomposition: by isolating all card data into the Payment service (encrypted at rest, the only service in PCI scope), an auditor reviews one small service instead of the whole platform. The boundary we drew for domain reasons pays off again for compliance.

## Step 9: Scale and optimization — caching, cost, and (maybe) multi-region

Now we make it fast and cheap, and decide honestly whether it needs to be multi-region. The relevant deep-dives: [caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services), [performance and cost optimization](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices), and [multi-region microservices and data locality](/blog/software-development/microservices/multi-region-microservices-and-data-locality).

**Caching is the highest-leverage optimization on the read path.** The browse path is 30x the traffic of checkout and is overwhelmingly read-only, so we cache aggressively in layers: CDN for static assets and cacheable catalog pages; a Redis cache in front of Catalog and Pricing for hot products; the read models for composed views. A 95% cache hit rate on the catalog read path means only 5% of browse traffic ever reaches Postgres — that is the difference between needing 4 database replicas and needing 40. The cache invalidation strategy (event-driven: a "product updated" event evicts the cached entry) is the hard part, and it is where most caching bugs live.

**Cost optimization is a first-class design constraint, not an afterthought.** The biggest lever is autoscaling: we provision for the *daily* peak (15k RPS) standing and let the HPA scale up for promotions, rather than paying for 150k-RPS capacity 365 days a year. We use spot/preemptible instances for stateless, interruption-tolerant services (Catalog read replicas, Notification workers) and on-demand for stateful ones (Order, Payment). We right-size resource requests from real utilization data — over-requesting CPU/memory is the most common and most invisible cloud waste.

**Multi-region — the honest call.** The requirements flagged a second region in year one. Do we build active-active multi-region now? **No.** Active-active multi-region is one of the most expensive things in distributed systems — you fight data locality, cross-region replication lag, conflict resolution, and a doubling of operational surface. We build **single-region with cross-region async backups and a documented failover-to-a-warm-standby plan**, and we keep the data layer's design *compatible* with future multi-region (no assumptions that break under partitioning). We plant the flag without building the cathedral. When the second region's traffic justifies it — when latency to far-away users actually hurts the SLO, or when regulatory data residency forces it — we revisit. Building it on day one for a problem we do not have yet is the textbook over-engineering this series keeps warning against.

#### Worked example: Black Friday 10x capacity and the scaling math

This is the question the VP will ask, so we answer it with numbers, not vibes. Daily peak is 15,000 RPS; Black Friday plans for 10x = **150,000 RPS** at the top, sustained ~3 hours.

Take the Order (checkout) service. At daily peak, checkout is ~5% of total traffic ≈ 750 RPS, served by 6 replicas, so each replica handles ~125 RPS at 60% CPU. The relationship is roughly linear in this range, so 10x checkout traffic = 7,500 RPS needs ~60 replicas at the same per-pod load. Our HPA maxes at 40 — so we raise the Black-Friday ceiling to 70 (with headroom) and **pre-warm** to 50 the night before, because cold-start autoscaling during a spike is too slow (pods take ~30s to become ready, and a thundering spike outruns reactive scaling). The browse path, at 150k × 0.95 ≈ 142k RPS, is mostly absorbed by the 95% cache hit rate, so only ~7,000 RPS reach the Catalog service and its Postgres replicas — the cache is what makes 10x affordable.

Now the database. Order writes scale to ~7,500 writes/sec at peak. A single Postgres primary can sustain that with tuning, but it is the riskiest single point, so we shard Inventory by SKU (the highest-contention writes) and keep Order on a beefy primary with synchronous standby and read replicas for the read model. Kafka, at ~50,000 events/sec peak (orders + inventory + shipping events), needs partitions sized for that throughput — say 50 partitions on the orders topic so consumers can parallelize.

The scaling sequence, as a timeline:

![A timeline of Black Friday scaling: a capacity test two weeks out replaying 10x load, pre-warming pods and freezing deploys the day before, traffic hitting 10x at midnight, the autoscaler scaling the order service 4x, shedding optional reads, and the checkout SLO holding at the half-hour mark](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-8.webp)

The key insight is that **the spike is survivable only because of decisions made weeks earlier**: the load test that found the real per-pod limit, the pre-warming that beat cold-start latency, the cache that absorbed the read flood, and the load-shedding rules that sacrifice recommendations to protect checkout. You do not survive 10x by scaling reactively at midnight; you survive it by having designed for it. That is the whole difference.

## The headline decisions, as a matrix

Before we stress-test, let me put the four biggest architecture decisions side by side, the way I would on the whiteboard — what each buys, what it costs, and when it wins. This is the artifact the VP photographs.

![A decision matrix comparing database-per-service, saga over two-phase-commit, async events, and a service mesh across what you gain, what you pay, and when each wins](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-3.webp)

| Decision | What you gain | What you pay | When it wins |
|---|---|---|---|
| **DB per service** | Team autonomy, polyglot stores, clear ownership | No cross-service joins, eventual consistency | Always, in true microservices — it is the defining rule |
| **Saga over 2PC** | Availability, works with external steps | No isolation, must hand-write compensations | Any multi-service transaction with a non-transactional step |
| **Async events** | Loose coupling, decoupled availability | Eventual reads, harder debugging | Fan-out flows where the caller does not need the result now |
| **Orchestration over choreography** | One place knows the state, debuggable | A central component to own | Critical, money-touching flows you must debug fast |
| **Service mesh (Linkerd)** | Uniform mTLS, traffic-shifting, no per-svc code | Sidecar latency/memory, control-plane complexity | ~10+ services where library-based mTLS stops scaling |
| **Single-region now** | Far cheaper, far simpler ops | Higher latency for distant users, region is an SPOF | Until distant-user latency or data residency actually hurts |

Every row names a cost. That is the discipline. There is no free architecture; there is only the architecture whose costs you chose on purpose.

## Stress-testing the whole design

A design is a hypothesis until you try to break it. Here are four concrete disasters and what the design does under each. This is the part of the review where you earn your seniority, because the room is trying to find the crack.

**1. Black Friday, 10x traffic.** Covered in the worked example above. The cache absorbs the read flood (95% hit rate → only 5% reaches the DB), the HPA plus pre-warming scales the checkout path 10x, load shedding sacrifices recommendations to protect checkout, and the SLO holds. The thing that *would* break — and we tested for it — is cold-start autoscaling losing the race against the spike, which is why we pre-warm. **Result: survivable, by design, because we tested it two weeks out.**

**2. A region fails.** Our honest single-region design means a full region outage *is* downtime for new traffic until we fail over to the warm standby (RTO ~15 minutes, RPO ~1 minute given async replication). We accepted this trade explicitly: active-active would prevent it but costs more than the risk justifies at our current scale and SLO. The error budget can absorb a rare regional failover; if region failures became frequent enough to threaten the SLO, *that* data would justify going multi-region. **Result: degraded with a documented, tested failover — a conscious trade, not an oversight.**

**3. The payment provider goes down.** This is the scariest one because checkout *needs* payment. The design: Payment service has a **circuit breaker** that opens fast when the primary PSP errors, and a **secondary PSP** to fail over to. While both are down, the breaker fails fast (no thread-blocking cascade), and checkout returns a clear "payment temporarily unavailable, your cart is saved" — the Cart service holds the basket, so the user loses nothing but time. Crucially, because Payment is **bulkheaded**, a dead PSP does not drain the Order service's threads, so browse and the rest of checkout-prep stay healthy. **Result: checkout pauses gracefully; the rest of the platform is unaffected; carts are preserved.**

**4. A hot product — one SKU goes viral.** Ten thousand people try to buy the same limited item in the same minute. The danger is a hot partition: all the writes hit one Inventory shard and one row, and row-level contention serializes them. The design: Inventory uses optimistic concurrency on the stock row and the reservation is a fast local transaction, but at extreme contention we add a **per-SKU in-memory reservation queue** in front of the hot row, and once stock hits zero we fail fast with "sold out" instead of letting 10,000 requests queue on the database. Pricing's surge engine, hearing the "stock low" event, may raise the price — which naturally sheds some demand. **Result: the hot row is protected, the oversell invariant holds, and excess demand is shed at the edge rather than melting the database.**

Four disasters, four defensible answers, each tracing back to a specific pattern from this series. That is what "I have stress-tested this design" means.

## The evolution path: it did not start like this

Here is the most important context for the whole design, and the thing juniors most need to hear: **ShopFast did not start as nine services. It started as a [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith), and it should have.** The architecture on the whiteboard today is the *result* of eight years of pressure-driven evolution, not a thing anyone designed up front. If a four-person ShopFast had tried to build this in 2019, they would have spent all their runway on distributed-systems plumbing and shipped nothing.

![A before-and-after diagram contrasting the 2019 modular monolith with one deploy unit, in-process calls, and a shared database against the 2026 architecture with per-team deploys, network calls, and a database per service](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-5.webp)

The 2019 ShopFast was one deploy unit with four well-bounded modules (catalog, cart, order, user) sharing one Postgres. In-process calls, zero network latency, one `BEGIN/COMMIT` for checkout, a stack trace when anything broke. It was *correct and fast to build* — and crucially, it was *modular*, so the seams where services would later split were already clean module boundaries. That is the strangler-fig precondition: a modular monolith is the launchpad, a big-ball-of-mud monolith is a trap.

The splits, each triggered by a *real pain*, never by a roadmap that said "do microservices":

![A timeline of ShopFast's evolution from a 2019 modular monolith through extracting payment for PCI scope, splitting catalog for read scaling, introducing the order saga and database-per-service, adding the event bus with outbox and CDC, adopting a mesh and tracing at forty services, to multi-region with nine core services in 2026](/imgs/blogs/designing-a-complete-microservices-system-end-to-end-6.webp)

- **2020 — extract Payment.** Trigger: PCI compliance audit. Pulling card handling into one service shrank the audit scope from "everything" to "one service." Pain-driven, clear ROI.
- **2021 — split Catalog.** Trigger: browse traffic was 30x checkout and growing, but scaling the monolith meant scaling everything. Catalog got its own service and its own read-replica fleet.
- **2022 — Order saga + database-per-service.** Trigger: the shared-database coupling was causing cross-team deploy conflicts; the checkout `BEGIN/COMMIT` was the last thing binding the modules. Splitting it forced the saga.
- **2023 — event bus + outbox.** Trigger: too many synchronous calls were coupling availability; moving fan-out to events decoupled it. The outbox came in with the bus to fix the dual-write problem.
- **2024 — mesh + tracing.** Trigger: at ~40 internal services (we had over-split in places — see the honesty section) debugging was impossible without traces and mTLS-in-code did not scale.
- **2026 — consolidate to 9 core + multi-region prep.** Trigger: we had *too many* services and merged several nano-services back together (the reverse migration from [microservices anti-patterns](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith)), landing on the right-sized nine.

The migration mechanics — running the new service alongside the old code, routing a slice of traffic, and strangling the monolith path by path — are the [strangler-fig pattern](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices). The lesson for the room: **you do not design this system; you grow it, pruning as you go.** The whiteboard architecture is a snapshot of a living thing.

## Case studies: how real platforms structure this

ShopFast is fictional, but every decision in it is borrowed from real platforms that have published how they actually run. Order-of-magnitude accurate, no fabricated specifics.

**Uber and DOMA.** Uber's early architecture grew into thousands of microservices, and the sprawl became its own problem — the same over-splitting ShopFast hit at "40 services." Uber's published response was **DOMA (Domain-Oriented Microservice Architecture)**: group the thousands of services into a few dozen *domains* with clear gateways between them, so a team reasons about a domain, not a thousand services. The lesson ShopFast applies directly: **organize services into domains, and resist the nano-service explosion.** The tree diagram grouping our nine services into three domains is DOMA in miniature.

**DoorDash's migration off a monolith.** DoorDash publicly described moving from a Python/Django monolith to services, explicitly to get independent deployability and per-capability scaling as their order volume grew — and they emphasized doing it incrementally, service by service, exactly the strangler-fig path. The lesson: **the migration is incremental and pain-driven, and the monolith-first start is normal, not shameful.** This is precisely ShopFast's evolution timeline.

**Amazon's two-pizza teams.** Amazon's organizing principle is that a team should be small enough to feed with two pizzas, and each team owns its service end-to-end — build it, run it, page for it. This is Conway's law turned into an explicit org rule, and it is why ShopFast assigns exactly one owning team per service with no co-ownership. The lesson: **service boundaries and team boundaries are the same boundary; you cannot get the architecture right while getting the org wrong.**

**Netflix and chaos engineering.** Netflix popularized deliberately injecting failure in production (Chaos Monkey and successors) to prove resilience patterns actually work, rather than assuming they do. ShopFast's stress-test discipline — "what happens when the PSP is down?" answered with a *tested* breaker-and-failover, not a hopeful one — is chaos engineering as a design-review habit. The lesson: **a resilience pattern you have not tested under failure is a guess; you must break it on purpose, regularly.**

**Monzo's service count.** Monzo, a bank, runs well over a thousand microservices with a relatively small engineering org, sustained by an extraordinary investment in a uniform platform (a paved road so consistent that creating and operating a service is nearly free). The lesson is a *warning*: Monzo's count works *because* of that platform, and copying the count without the platform investment is how you get an unoperable mess. ShopFast deliberately stays at nine. The lesson: **a high service count is a function of platform maturity, not architectural sophistication — earn it before you spend it.**

## What I would do differently — and what is over-engineered

This is the most senior section, and the one I would not let any junior skip. A design review that ends with "and it is all perfect" is a review that failed, because every real system has compromises and a good architect names their own before someone else does. Here is where I would push back on ShopFast's own design.

**Things that are over-engineered for the stated requirements:**

- **The service mesh is borderline.** At nine services, we justified Linkerd mostly on the *future* growth path and uniform mTLS. An equally defensible call is "no mesh; do mTLS in a shared library and traffic-shifting at the gateway" until we cross ~15 services. The mesh is a real operational burden, and I would only keep it if the team has the platform maturity to run it well. If the platform team is two people, **drop the mesh** — it will own you instead of you owning it.
- **Separate Pricing and Catalog might be premature.** We split them on a clean domain boundary, but at ShopFast's current volume they could be one service with two modules until the surge-pricing computation actually competes for resources with catalog reads. Splitting on a *future* concern is the classic over-decomposition mistake. I would seriously consider merging them and re-splitting later under real pressure.
- **Full CQRS read models everywhere would be over-engineering.** We use them only for the genuinely hot composed views (order history). Building event-sourced read models for every cross-service query is a common trap — most cross-service reads are fine as a BFF fan-out with caching, and CQRS adds real complexity (the read model can lag, can get out of sync, needs rebuilding). Use it surgically, not universally.

**Things I would do differently with hindsight:**

- **I would have started the outbox earlier.** ShopFast added the outbox in 2023, *after* getting burned by phantom orders from the dual-write problem in 2022. The outbox is cheap and should go in the moment you publish events from a database transaction. That ordering was a real, avoidable scar.
- **I would have invested in the platform/paved-road before splitting past ~6 services.** ShopFast over-split to ~40 services in 2024 *before* the observability and CI/CD platform was ready, and operating that fleet without good traces was the genuinely painful period. The platform investment should *lead* the service count, not lag it.
- **I would push harder on keeping things in the monolith longer.** Looking back, two or three of the early extractions did not pay for themselves and were later merged back. The default should be "keep it in the modular monolith" with a high bar to extract — extraction is easy to do and expensive to undo.

**And the honest bottom line a senior must be able to say to the VP:** if ShopFast's traffic and team were a tenth of what they are, *none* of this would be justified — the right answer would be a modular monolith with one database, and this entire architecture would be elaborate, expensive theater. The architecture is correct *for these requirements*. Change the requirements and the correct architecture changes. That conditional is the entire job.

## The journey: junior to senior

This is the last post in the series, so let me name the arc explicitly, because it is the real takeaway under all the patterns.

**The junior** learns the pieces. They can build a single well-structured service ([anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice)), make it talk to another over gRPC or events, give it a database, put it in a container, and deploy it to Kubernetes. They know what a saga is, what a circuit breaker is, what a JWT is. This is real, valuable skill, and it is most of what the first thirty-five posts in this series taught. A junior who has internalized all of it is a strong, productive engineer.

**The senior** does three things the junior does not. First, they **assemble** — they can take fifty patterns and compose them into one coherent system where the pieces reinforce instead of fight each other, the way this whole post did. Second, they **subtract** — they know that every pattern is a cost, they can name the cost out loud, and they have the judgment and the spine to leave patterns on the table when the simpler thing is correct. The senior's most powerful sentence in a design review is "we don't need that yet." Third, they **reason from requirements** — they never start from "we'll use microservices"; they start from QPS and SLOs and team size and dollar budgets, and they let those numbers design the architecture, so that every box on the whiteboard traces back to a requirement they can point at.

The progression of this series mirrored that arc. We started with "what is a microservice and when should you *not* build one" — the subtraction lesson, first, on purpose. We spent thirty-odd posts on the pieces. And we ended here, assembling them into ShopFast while continuously asking "is this justified?" If there is one thing to carry out of forty posts, it is this: **the goal was never to build microservices. The goal was to build the simplest system that meets the requirements, and sometimes — often — that system has more than one service.** Knowing when, and how many, and how to connect and operate and secure and scale them — *and when to stop* — is what it means to have made the journey from junior to senior.

## Key takeaways

- **Start from requirements, not architecture.** QPS, latency SLOs, data volume, team size, and budget design the system. If you cannot point a box back to a number, you cannot defend it.
- **The "should this be microservices?" gate is real.** Microservices buy independent deployability and scaling along team/capability seams, at a steep cost. Below ~5 teams or without differential scaling, the modular monolith wins. Say so.
- **Decompose by bounded context, align to teams.** Cut along business capability, never technical layer. One owner per service. Nine right-sized services beat forty nano-services every time.
- **Sync only when the caller needs the answer now; events for everything else.** This single rule decides your coupling and your failure blast radius. Async-ifying optional work is how checkout survives a dead notification provider.
- **Database-per-service plus saga plus outbox is the data spine.** Private data buys autonomy and costs you joins and isolation; the saga gives weak-but-usable atomicity; the outbox makes publishing reliable. Skip the outbox and you get phantom orders.
- **Resilience is making one degraded dependency partial, not total.** Budgeted timeouts, jittered capped retries on idempotent ops only, breakers, bulkheads, and graceful degradation of optional features. The critical path never hard-depends on an optional one.
- **SLOs and error budgets turn reliability arguments into numbers.** Alert on symptoms users feel, not causes. The error budget tells you when to ship and when to freeze.
- **You grow this architecture, you do not design it.** Start as a modular monolith, strangle out services under real pressure, prune over-splits. The whiteboard is a snapshot of a living system.
- **The senior skill is subtraction.** Every pattern is a cost. Name it. Leave the unneeded ones on the table. "We don't need that yet" is the most valuable sentence in the room.

## Further reading

- Sam Newman, *Building Microservices* (2nd ed., O'Reilly) — the canonical text on decomposition, integration, and evolutionary migration.
- Chris Richardson, *Microservices Patterns* (Manning) and microservices.io — the pattern catalog behind sagas, the outbox, API composition, and CQRS.
- Matthew Skelton and Manuel Pais, *Team Topologies* — the org/Conway's-law half of the design, why team boundaries are service boundaries.
- The series Track 1: [what microservices are and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them), [monolith-first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith), [service boundaries with DDD](/blog/software-development/microservices/service-boundaries-with-domain-driven-design).
- The data spine: [database-per-service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), [the transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing), and the mechanism deep-dives on [CDC and the outbox](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) and [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).
- Operating it: [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads), [distributed tracing and observability](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry), [SLOs and golden signals](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices), [Kubernetes for microservices](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials).
- The org and migration: [Conway's law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices), [strangler-fig migration](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices), [anti-patterns and when to go back to monolith](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith).
- How seniors reason about all of this: [how seniors approach ambiguous system-design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems) and [back-of-the-envelope estimation](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design).

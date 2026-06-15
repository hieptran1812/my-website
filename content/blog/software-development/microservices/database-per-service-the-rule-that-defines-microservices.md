---
title: "Database per Service: The Rule That Defines Microservices"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The one rule that separates real microservices from a distributed monolith: each service owns its data privately. Learn why a shared database recouples your teams, the hard consequences you must design around, and how to actually get there from a single Postgres."
tags:
  [
    "microservices",
    "database-per-service",
    "distributed-systems",
    "data-ownership",
    "software-architecture",
    "backend",
    "polyglot-persistence",
    "api-composition",
    "read-models",
    "data-consistency",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/database-per-service-the-rule-that-defines-microservices-1.webp"
---

The ShopFast team thought they had built microservices. They had eleven repositories, eleven CI pipelines, eleven Kubernetes deployments, eleven Slack channels with on-call rotations. The architecture diagram on the wiki had eleven neat boxes with arrows between them. By every surface measure, this was a microservices system. And then one Tuesday, a backend engineer on the catalog team shipped a tiny migration — renaming a column from `price` to `list_price` on the `products` table, the kind of cleanup you do without a second thought — and within the hour three other services started throwing errors in production. The orders service crashed because its reporting query did a `JOIN products ON ...` and selected `price`. The pricing service crashed. The recommendation service crashed. None of those teams had been in the room for the catalog change. None of them had been told. They did not need to be told, because the catalog team had no idea those queries existed.

The reason all four services died from one column rename is the single most important fact about that architecture, and it had nothing to do with the eleven boxes on the wiki. Behind all eleven services sat one shared Postgres database, and every service connected to it directly and read whatever tables it found useful. The `products` table did not belong to the catalog service in any meaningful sense — it belonged to *everyone*, which is to say it belonged to *no one*, which is to say nobody could change it without a cross-team negotiation that, in practice, never happened until production told them it should have. ShopFast had eleven deployable units and exactly one thing they could not deploy independently: their data. That one shared dependency quietly turned the whole thing back into a monolith, just a monolith with more network hops and worse failure modes. They had paid the entire cost of microservices and kept the worst property of the monolith.

This is the post about the rule that would have saved them. It is, in my opinion, *the* rule — the one that genuinely separates microservices from a distributed monolith, more than any other single decision you will make. The rule is this: **each service owns its own data privately, and no other service is ever allowed to touch its tables.** Other services can ask for data through the owning service's API, or they can subscribe to events the owning service publishes, but they may never reach around the service and read or write its database directly. That sentence sounds almost trivially simple. It is, I promise you, the most expensive and consequential constraint in this entire series, and getting it right is the difference between a system that lets a hundred engineers ship in parallel and a system where one column rename pages four teams at 2am.

![A before and after comparison showing four ShopFast services sharing one Postgres schema with free cross-table JOINs on the left versus each service owning a private schema reached only through its API or events on the right](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-1.webp)

By the end of this post you should be able to do four concrete things. First, look at any proposed architecture and tell, in thirty seconds, whether it is microservices or a distributed monolith wearing a microservices costume — the test is "can two services see each other's tables?" Second, name the hard consequences the rule forces on you and design around each one deliberately rather than discovering them in an incident: no cross-service JOINs, no distributed ACID transaction, mandatory data duplication, queries that span services done by composition or read models, and referential integrity that is now your code's job and not the database's. Third, choose the right point on the isolation spectrum — shared schema, separate schema, separate database, separate cluster — knowing exactly what each one costs. And fourth, plan the painful migration from a shared database to private ownership without taking the system down. We will use ShopFast as the running example throughout, and we will keep coming back to that four-table reporting JOIN, because it is the perfect specimen of everything this rule makes hard and everything it makes possible.

## Why a shared database is the silent killer

Let me state the why before the how, because if you do not feel the why in your gut you will cut corners on the how. A shared database is seductive precisely because it works. On day one it is the path of least resistance. You already have a Postgres instance. The orders service needs the customer's email for the receipt — it is right there in the `customers` table, one `JOIN` away, no network call, strongly consistent, sub-millisecond. Why on earth would you stand up a whole customer service and make an HTTP call when the data is sitting in the same database? The junior engineer who reaches for that `JOIN` is not being lazy or stupid. They are being locally rational. The cost of the shared database is real but it is *deferred* and *diffuse*, while the cost of the API call is small but *immediate* and *concrete*. Human beings, and engineering teams, reliably choose the immediate concrete cost over the deferred diffuse one, which is exactly why this anti-pattern is so common and so durable.

So you have to make the deferred cost vivid. Here it is. When the catalog team and the orders team both read and write the same `products` table, that table's schema becomes a *shared contract between two teams that neither team controls*. The schema is, effectively, an API — but it is an API with no versioning, no documentation, no deprecation policy, no contract tests, and no idea who its consumers are. Every column is public. Every index assumption is shared. The catalog team cannot rename a column, cannot split a table, cannot change a type, cannot add a `NOT NULL` constraint, cannot drop an unused field — cannot do any of the ordinary evolution that a healthy codebase requires — without potentially breaking a consumer they have never met. In Martin Fowler's vocabulary this is the **integration database** anti-pattern: the database has become the integration point between services, and an integration database is one of the most tightly-coupled, hardest-to-change artifacts a software organization can produce. It couples teams at the most rigid layer in the entire stack, the one place where a mistake corrupts data rather than just throwing an error.

Trace the consequences one layer down. Because the schema is a shared contract, schema changes require cross-team coordination, which means they get batched, delayed, and feared. Because changes are feared, the schema ossifies; nobody wants to be the one who broke checkout, so the `products` table accretes columns nobody dares remove and grows into a 90-column monster that serves six different access patterns badly. Because every service connects to the same database, you cannot scale one service's data tier independently — if the catalog's read traffic needs ten read replicas but the orders write path needs strong single-leader consistency, tough, it is one cluster and one configuration for all of you. Because every service shares one connection pool budget against one Postgres `max_connections`, a connection leak in the recommendation service can starve checkout of connections and take down payments. Because the database is the integration point, a slow query written by one team — an unindexed report scanning the `orders` table — can saturate the shared buffer cache and degrade every other service's latency at once. The shared database is not just a coupling problem. It is a coupling problem, an availability problem, a scaling problem, and an evolution problem, all wearing the disguise of a convenient `JOIN`.

And here is the part that makes it the *silent* killer rather than the loud one: none of this shows up in the architecture diagram. The diagram shows eleven services and clean arrows. The shared database is drawn, if at all, as one small box at the bottom that everyone's arrow points to, and that box is where all the coupling lives. You cannot see it in the code review for any single service, because each service's code looks fine in isolation. You only see it when you try to do the thing microservices were supposed to let you do — deploy one service without coordinating with the others — and discover that you cannot, because they all share the one thing you forgot to split.

## What "owns its data" actually means

So the rule is "each service owns its data." Ownership is a word we throw around; let me make it precise, because the precision is the whole point. A service owns a piece of data when **it is the only code that can read or write that data's storage directly, and every other part of the system can reach that data only by asking the owning service** — either by calling its API (a query or a command) or by consuming an event it chose to publish. The storage itself — the schema, the tables, the database, whatever the physical boundary is — is private. It is an implementation detail of the service, as private as a private field on a class. Nobody else's connection string points at it. Nobody else's credentials can authenticate to it. The service's API is its public surface; its tables are its private internals, and the wall between them is enforced, not just documented.

![A vertical stack showing other services and clients at the top reaching a service only through its public API or its domain events, then the owning service with its business logic, and at the bottom a private store locked by database roles](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-4.webp)

This is exactly the encapsulation principle you already trust from object-oriented design, lifted up a level to the service. You would never make a class's fields public and let other classes mutate them directly; you expose methods and keep the fields private so you can change the internal representation freely as long as the methods keep their contract. Database-per-service is that same discipline applied to persistent state. The tables are the private fields. The API and the events are the public methods. The freedom you buy is identical: as long as the owning service keeps its API contract stable, it can change its tables however it likes — rename columns, split tables, switch from Postgres to DynamoDB, add a cache, denormalize for speed — and no other service can possibly notice, because no other service was ever allowed to see the tables. That freedom *is* independent deployability for the data tier. It is the thing ShopFast did not have.

Three clarifications that trip people up. **First, "owns" means writes *and* reads.** A surprisingly common half-measure is to let other services read the tables directly ("it's just a read, what could go wrong?") while routing writes through the API. This does not work, because a reader is still coupled to your schema — a reader's query breaks when you rename that column just as surely as a writer's does. ShopFast's outage was caused entirely by *readers*. If anyone but the owner can issue a `SELECT` against your tables, you do not own your data. **Second, "owns" is about the logical data, not necessarily a separate physical database server.** A service can own its data living in a private schema inside a shared Postgres instance, as long as nobody else can query that schema. The isolation can be logical or physical; we will spend a whole section on that spectrum. **Third, ownership is exclusive and singular.** Exactly one service owns each piece of data and is the source of truth for it. Other services may *cache* or *replicate* copies of that data for their own use, but those copies are read-only projections of someone else's truth, and everyone knows which service is the authority. Two services that both claim to be the source of truth for the customer's email is not ownership; it is a future data-corruption incident with a date on it.

## The hard consequences you are now signing up for

Here is where I lose the people who wanted database-per-service to be free. It is not free. The moment you split the data, you lose a set of guarantees that a single relational database gave you for nothing, and you must now provide equivalents yourself, in application code, at the cost of latency, complexity, and consistency. I am going to lay all five consequences out bluntly, because a senior engineer's value is in seeing the full bill before signing, not in being pleasantly surprised by the convenient parts and ambushed by the hard ones.

**Consequence one: no cross-service JOINs, ever.** This is the one that hits first and hardest. In the shared database, ShopFast's monthly revenue report was one beautiful query that joined `orders`, `order_items`, `products`, and `customers` and let Postgres's query planner do all the work — index scans, hash joins, the lot — and returned a fully-stitched result in a few milliseconds. Once those four tables live in four private stores owned by four services, that query is impossible. There is no engine that can see all four tables at once, because no credential can reach all four stores. The data that used to be joined in the database now has to be joined *somewhere else* — in the calling application, by composition, or in a precomputed read model. Either way, the cheap, planner-optimized in-database `JOIN` is gone, and you replace it with network calls or with denormalized copies. This is not a small inconvenience; cross-entity queries are the bread and butter of most business applications, and you have just outlawed the cheapest way to do them.

**Consequence two: no distributed ACID transaction across services.** In the monolith, when checkout reserved inventory, charged the card, and created the order, all three writes happened in one database transaction. Either all three committed or all three rolled back, atomically, with the database guaranteeing it. Once inventory, payments, and orders are three services with three private databases, there is no transaction that spans all three. There is no `BEGIN; ... COMMIT;` that brackets writes in three different Postgres instances and rolls them back together. Two-phase commit across services exists in theory but is, in practice, an availability and latency disaster that almost nobody runs in production microservices. So you give up atomicity across services and replace it with a **saga**: a sequence of local transactions, each committing in one service, with compensating actions to undo the earlier steps if a later one fails. The saga gives you a much weaker guarantee — eventual consistency with explicit rollback logic you write yourself — in place of the bulletproof ACID transaction the database used to hand you for free. We forward-link the full mechanics to [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice) and the [saga deep-dive in the database series](/blog/software-development/database/saga-pattern-distributed-transactions); for now, internalize that the atomic multi-entity write is gone.

**Consequence three: data must be duplicated.** If the orders service needs the customer's name and email on every order, and it cannot `JOIN` the `customers` table because it does not own it, then the orders service must keep its own copy of the name and email — denormalized into the order at the moment it is placed, or replicated into a local table kept up to date by customer events. Duplication is no longer a code smell to be normalized away; in a microservices data architecture it is a deliberate, necessary design choice. You will store the customer's name in the orders database, the catalog database, and the shipping database, and you will accept that those three copies can briefly disagree when the customer changes their name. Normalization was the right answer inside one database; *denormalization across services* is the right answer between them. This flips an instinct that good engineers spent their whole careers building, and it takes real discipline to stop fighting it.

**Consequence four: cross-service queries need composition or read models.** Because the JOIN is gone, any query whose answer lives in more than one service has to be assembled. You have exactly two tools. **API composition**: the caller fans out to each owning service, gets the pieces, and stitches them together in memory — live, fresh, but as slow as your slowest call and as available as the product of all callees' availabilities. **Read models** (also called materialized views or CQRS read sides): a separate, denormalized data store that subscribes to the owning services' events and continuously projects their data into a shape that answers the query in a single local read — fast and decoupled, but eventually consistent, lagging the source by however long the event pipeline takes. Most real systems use both, picking per query. We will reimplement ShopFast's report both ways, with numbers, later in this post.

**Consequence five: referential integrity is now your job.** A single relational database enforces, for free, that an `order_items.product_id` actually refers to a row in `products` — the foreign key constraint makes it impossible to insert an orphan. Across services, that constraint cannot exist, because the foreign key would have to cross a database boundary, and it cannot. So nothing stops the orders service from referencing a `product_id` that the catalog service has since deleted. Referential integrity across services is no longer a database guarantee; it is an application concern you must design for — usually by *not* hard-deleting (soft-delete and tombstone instead), by tolerating dangling references gracefully, and by reconciling asynchronously. The database used to be your safety net for this. Now you are the net.

Sit with that list. Five guarantees — joins, atomic transactions, normalization, single-query reads, referential integrity — that a single database gave you for free, all gone, all now your responsibility, all paid for in latency, code, and weaker consistency. That is the real price of the rule. The reason we pay it anyway is that the alternative — the shared database — costs us the one thing microservices exist to buy: the ability for many teams to evolve and deploy independently. You are trading data-layer guarantees for organizational scalability. That trade is sometimes wrong; the [monolith-first post](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) argues you should not pay it until you have to. But when you do pay it, pay it with eyes open.

## The isolation spectrum: schema, database, cluster

"Each service owns its data privately" does not mandate a separate physical database server per service on day one. Isolation is a spectrum, and where you sit on it is one of the most practical decisions in this whole topic, because it trades isolation strength against operational cost. Let me lay out the four rungs.

![A matrix comparing shared schema, separate schema, separate database, and separate cluster across isolation strength, blast radius, operational cost, cross-query ease, and deploy independence](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-3.webp)

**Rung 0 — Shared schema (the anti-pattern).** Every service reads and writes the same tables in the same schema. This is ShopFast's starting point and it is not database-per-service at all; it is the integration database. Isolation: none. Blast radius of a schema change: the whole company. The only thing it has going for it is the free `JOIN` and the lowest possible ops cost — one database to run. We list it only to be the thing you are escaping.

**Rung 1 — Separate schema per service, shared database instance.** Each service gets its own schema (in Postgres, a schema; in MySQL, often a separate logical database) inside one shared physical Postgres instance, and — this is the load-bearing part — database permissions are configured so that each service's login role can only touch its own schema. The orders service literally *cannot* `SELECT` from the catalog schema, because the grant does not exist. This is genuine logical isolation: the schema is now a private contract. You get most of the coupling benefit (nobody can JOIN your tables, nobody can read them, you can evolve them freely) at a fraction of the operational cost (one Postgres to back up, patch, and monitor). What you do *not* get: independent failure (one instance crashing takes everyone down), independent scaling (one shared resource budget), or independent technology choice (everyone is on this Postgres version). For a small-to-mid system, or early in a migration, **separate-schema-shared-instance is the sweet spot** and I reach for it constantly. The key is that the role grants are real, not a gentleman's agreement.

**Rung 2 — Separate database per service.** Each service gets its own database — its own logical database in the same managed service, or its own instance entirely. Now you have independent failure domains (one database down does not take the others with it), independent scaling (size each one for its load), independent configuration, and the freedom to upgrade or migrate one without touching the others. The cost is operational: more databases to provision, back up, monitor, patch, secure, and pay for. Connection management gets more complex. This is the canonical "database per service" people picture, and it is the right target for services that have real, divergent load or that you genuinely need to fail and scale independently.

**Rung 3 — Separate cluster (or separate engine) per service.** The strongest isolation: not just a separate database but a separate cluster, possibly a separate database *engine* entirely. The orders service runs its own Postgres cluster; the catalog service runs its own Elasticsearch cluster; the session service runs its own Redis. This is full physical isolation plus polyglot persistence (next section). Blast radius is minimal — these systems share nothing — but the operational cost is highest: you are now running and being paged for multiple distinct database technologies, each with its own backup story, scaling story, failure modes, and on-call expertise. You reach for this rung for the handful of services whose access pattern genuinely demands a different engine, or whose isolation requirements (a payments database with strict compliance) justify the cost.

The senior move is **to pick the weakest rung that still gives you a private schema, and climb only when a specific pain forces you to.** Most teams over-isolate too early, standing up a separate RDS instance per service when they have eight services and three engineers, and drown in operational toil. Almost as many under-isolate forever, staying on the shared schema and calling it microservices. The right answer is usually: start at rung 1 (separate schema, shared instance, *real grants*), and promote individual services to rung 2 or 3 when their load, failure-isolation needs, or technology fit demand it. Isolation is per-service; you do not have to make the same choice for all eleven.

## Locking the schema down so the rule is enforced, not requested

A rule that is merely written in the architecture doc is not a rule; it is a suggestion, and suggestions lose to deadlines. The thing that turned ShopFast's wiki diagram into a lie was that *nothing stopped* the orders service from connecting to the catalog tables. The connection string was right there in the config; the credentials worked. If you want database-per-service to hold, you must make it *impossible* for a service to touch another's data, not merely impolite. In a shared-instance, separate-schema setup, the enforcement mechanism is database roles and grants. Here is the actual Postgres for locking the catalog schema to exactly one login.

```sql
-- Run once as a superuser when provisioning the catalog service.
-- The catalog service gets its own schema and its own login role,
-- and that role can touch nothing outside its schema.

CREATE SCHEMA catalog;

CREATE ROLE catalog_svc LOGIN PASSWORD 'rotate-me-via-secrets-manager';

-- Lock down the default: no role can create objects in or even see
-- the public schema by accident. Revoke the permissive defaults.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE shopfast FROM PUBLIC;

-- The catalog role owns its schema and can do anything WITHIN it.
GRANT USAGE, CREATE ON SCHEMA catalog TO catalog_svc;
ALTER DEFAULT PRIVILEGES IN SCHEMA catalog
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO catalog_svc;
ALTER DEFAULT PRIVILEGES IN SCHEMA catalog
  GRANT USAGE, SELECT ON SEQUENCES TO catalog_svc;

-- Crucially: catalog_svc is NOT granted USAGE on the orders schema,
-- the customers schema, or anyone else's. A query like
--   SELECT * FROM orders.orders;
-- run as catalog_svc fails with "permission denied for schema orders".
-- The rule is now enforced by the database, not by code review.
```

The line that matters most is the *absence* of a grant: `catalog_svc` is never given access to any schema but `catalog`. When the orders service's reporting query tries to `JOIN` the catalog tables, it does not get wrong data — it gets `ERROR: permission denied for schema catalog`, in the CI pipeline, the first time anyone tries, long before production. That error message is the rule doing its job. You have converted a discipline problem (please don't read other services' tables) into a mechanism (you cannot). This is the single highest-leverage thing you can do to keep a shared-instance setup honest, and ShopFast did not do it, which is the entire reason a column rename could ripple across four services.

For separate-database or separate-cluster setups, the enforcement is even simpler and stronger: each service's credentials only exist for its own database, stored in its own secret, and the network policy (a Kubernetes `NetworkPolicy`, a security group) only permits each service's pods to reach its own database endpoint. The orders pods literally cannot open a TCP connection to the catalog database. That is the strongest enforcement there is, and it is the one I trust most, because it removes the credential from the building entirely. The principle is the same at every rung: **make the wrong thing impossible, not merely discouraged.** We cover the network and credential side of this more fully under [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management) and [service-to-service security](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

## A real per-service schema and its migration

Let me make the owning service concrete with the orders service's private schema and a migration, because "owns its data" should feel like code, not philosophy. Note two things in this schema that reflect the rule directly: there is no foreign key to `customers` (it lives in another service), and the customer's name and email are *denormalized* into the order, copied at the moment of placement so the orders service never needs to call out to render a receipt.

```sql
-- orders service migration: 0007_create_orders.sql
-- This schema is OWNED by the orders service. No other service's
-- role can touch it. Note what is and is not here.

CREATE TABLE orders.orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- A reference to the customer, but NOT a foreign key: the
    -- customers table lives in another service's database, so the
    -- DB cannot enforce this. Integrity is the application's job.
    customer_id     UUID NOT NULL,
    -- Denormalized customer fields, copied at order-placement time.
    -- The orders service is NOT the source of truth for these; it
    -- snapshots them so a receipt never needs a cross-service call.
    customer_name   TEXT NOT NULL,
    customer_email  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    total_cents     BIGINT NOT NULL,
    currency        CHAR(3) NOT NULL DEFAULT 'USD',
    placed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orders.order_items (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        UUID NOT NULL REFERENCES orders.orders(id),
    -- product_id points at the catalog service. Again, no FK.
    product_id      UUID NOT NULL,
    -- Denormalized product snapshot: name and the price AT THE TIME
    -- OF SALE. Even if the catalog later changes the price, the
    -- historical order must show what the customer actually paid.
    product_name    TEXT NOT NULL,
    unit_price_cents BIGINT NOT NULL,
    quantity        INT NOT NULL CHECK (quantity > 0)
);

CREATE INDEX idx_orders_customer ON orders.orders(customer_id);
CREATE INDEX idx_order_items_order ON orders.order_items(order_id);
```

The denormalized `unit_price_cents` is worth dwelling on, because it reveals something the shared-database version got *wrong*. In the old shared schema, `order_items` referenced `products.price` directly, which meant a historical order's apparent total could silently change when the catalog updated a price — a genuine correctness bug that the shared `JOIN` hid. Copying the price at sale time is not just a microservices necessity; it is the *correct* model of the domain. The order should record what the customer paid, not what the product costs today. Database-per-service forced ShopFast to confront a modeling error they had been living with. This happens a lot: the constraints of the rule push you toward sounder domain models, because they make you decide explicitly who owns each fact and when each fact is captured. The boundaries you draw here are the same ones you drew in [service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design) — each aggregate owns its data, and the data boundary follows the service boundary.

## Reimplementing the four-table report: API composition

Now the centerpiece. ShopFast's monthly report listed every order with the customer's name, the products' names, and the current stock level of each product — data spanning orders, customers, catalog, and inventory. In the shared database it was one four-table `JOIN`. With four private stores, the first way to rebuild it is **API composition**: a small composer (it can live in the reporting service, or a backend-for-frontend) calls each owning service, then stitches the pieces together in memory. The cardinal sin here is doing it sequentially; the right way is to fan out concurrently, so the total time is the *slowest* call, not the *sum* of all calls.

![A graph showing the report service fanning out concurrently to the order, customer, inventory, and catalog services and composing their responses into a single joined view](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-2.webp)

```go
// reportcomposer.go — assemble the order report by fanning out to
// the four owning services concurrently. The owning services are the
// ONLY code allowed to read their own data; we go through their APIs.

func (c *Composer) OrderReport(ctx context.Context, month string) ([]ReportRow, error) {
    // Step 1: orders is the "spine" of the report. Fetch it first;
    // everything else hangs off the IDs it returns.
    orders, err := c.orderSvc.OrdersForMonth(ctx, month)
    if err != nil {
        return nil, fmt.Errorf("orders fetch: %w", err)
    }

    custIDs := uniqueCustomerIDs(orders)
    prodIDs := uniqueProductIDs(orders)

    // Step 2: fan out the three remaining lookups CONCURRENTLY.
    // errgroup cancels the others if any one fails, and waits for all.
    var (
        customers map[string]Customer
        stock     map[string]int
        products  map[string]Product
    )
    g, gctx := errgroup.WithContext(ctx)

    g.Go(func() error {
        var e error
        // ONE batched call for all customer IDs, not one per order.
        customers, e = c.customerSvc.BatchGet(gctx, custIDs)
        return e
    })
    g.Go(func() error {
        var e error
        stock, e = c.inventorySvc.StockLevels(gctx, prodIDs)
        return e
    })
    g.Go(func() error {
        var e error
        products, e = c.catalogSvc.BatchGet(gctx, prodIDs)
        return e
    })

    if err := g.Wait(); err != nil {
        return nil, fmt.Errorf("fan-out: %w", err)
    }

    // Step 3: do the JOIN in memory. This is the JOIN that Postgres
    // used to do for us; now it is application code.
    return stitch(orders, customers, stock, products), nil
}
```

Two design decisions in there carry all the weight. **Batching**: `customerSvc.BatchGet(custIDs)` takes a list of IDs and returns them all in one round trip, instead of calling the customer service once per order. If a month has 50,000 orders and you called per-order, you would make 50,000 network calls and the report would never finish; batched, it is one call (or a handful of paginated calls). This is the N+1 problem from [inter-service communication fundamentals](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) — the cure is the same across services as it is across SQL rows. **Concurrency**: the three lookups run in parallel via `errgroup`, so their latencies overlap rather than stack. Let us put real numbers on why that matters.

#### Worked example: the JOIN report as composition, with latency

Take the per-call p99 latencies from figure 2: orders 12ms, customers 9ms, inventory 15ms, catalog 11ms. The composer must fetch orders first (it provides the IDs), then fans out the other three.

If you fanned out *sequentially* — a beginner's mistake — total latency is the **sum**: 12 + 9 + 15 + 11 = **47ms**, and that is the happy path. Worse, sequential calls compound tail latency: the chance that *at least one* of four independent calls hits its p99 is roughly 1 − 0.99⁴ ≈ 4%, so about 4% of reports see a tail spike, and on a sequential path those spikes add up.

If you fan out the last three *concurrently* after the orders call, total latency is `orders + max(customers, inventory, catalog)` = 12 + max(9, 15, 11) = 12 + 15 = **27ms**. The slow inventory call hides the other two entirely. You cut the report's p99 nearly in half — 47ms down to 27ms — by changing nothing but the call structure. That is why the composer uses `errgroup` and not a `for` loop.

Now stress it. **What is the report's availability?** It depends on all four services being up, so availability multiplies: if each service is 99.9% available, the report is 0.999⁴ ≈ **99.6%** available, which is 3.5 hours of report downtime a month — worse than any single service. Composition makes the consumer *less* available than its dependencies, the core fallacy from the comms post. **What about caching?** Customer and product data change slowly; cache `BatchGet` results for, say, 60 seconds and a warm report skips the customer and catalog calls entirely, dropping to `orders + inventory` = 12 + 15 = 27ms but with two of three dependencies removed from the availability product (now 0.999² ≈ 99.8%). **What happens at month-end when everyone runs the report at once?** The fan-out multiplies load on the four downstream services by the report's QPS — a classic amplification. This is precisely when API composition starts to hurt, and it is the cue to reach for a read model instead.

API composition is the right tool when the data must be **fresh to the second** (you cannot tolerate lag), the fan-out is **small** (a handful of services, not fifteen), and the **query rate is modest**. It is the wrong tool when any of those is false: a 15-service fan-out has terrible availability and latency, and a high-QPS report hammers your downstreams. For those, you precompute.

## Reimplementing the report: an event-fed read model

The second way to rebuild the four-table report flips the work from read time to write time. Instead of fanning out every time someone asks, you maintain a **read model**: a separate, denormalized store that already holds the joined shape, kept continuously up to date by subscribing to the owning services' events. When the report runs, it is a single local query against a purpose-built table — no fan-out, no availability product, just one fast read. The cost is that the read model lags the source by however long the event pipeline takes to propagate a change: it is eventually consistent.

![A graph showing the order, customer, and catalog services publishing domain events to a Kafka bus, a projector consuming them with a couple seconds of lag, and writing into a denormalized read model that answers the report with a single local query](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-5.webp)

```python
# report_projector.py — a consumer that subscribes to domain events
# from the owning services and maintains a denormalized read model.
# This read model lives in the reporting service's OWN store; it is a
# read-only PROJECTION of other services' data, not a second source
# of truth. The owning services remain authoritative.

def handle(event: Event) -> None:
    match event.type:

        case "OrderPlaced":
            # Insert the order spine with the snapshot fields the
            # event carried. Catalog/inventory fields fill in later
            # or are joined from rows we already projected.
            db.execute(
                """INSERT INTO report.order_view
                       (order_id, customer_id, customer_name,
                        product_id, product_name, qty, unit_price_cents,
                        placed_at)
                   VALUES (%(order_id)s, %(customer_id)s, %(customer_name)s,
                           %(product_id)s, %(product_name)s, %(qty)s,
                           %(unit_price_cents)s, %(placed_at)s)
                   ON CONFLICT (order_id, product_id) DO NOTHING""",
                event.payload,
            )

        case "CustomerRenamed":
            # The customer service is the source of truth for names.
            # When it changes, we update our denormalized copies.
            db.execute(
                """UPDATE report.order_view
                       SET customer_name = %(new_name)s
                     WHERE customer_id = %(customer_id)s""",
                event.payload,
            )

        case "StockLevelChanged":
            # Maintain a tiny stock side-table the report joins LOCALLY.
            db.execute(
                """INSERT INTO report.stock (product_id, on_hand)
                   VALUES (%(product_id)s, %(on_hand)s)
                   ON CONFLICT (product_id)
                   DO UPDATE SET on_hand = EXCLUDED.on_hand""",
                event.payload,
            )

    # Record the consumed offset so a restart resumes, not replays.
    db.commit()
```

The report query against this read model is now boring, which is exactly what you want: `SELECT * FROM report.order_view JOIN report.stock USING (product_id) WHERE placed_at >= ...`. One service, one store, one local query, no fan-out, no availability product. The events flow through an event bus; how you publish them *reliably* — without losing an event when the service crashes between committing its database write and publishing — is the [transactional outbox pattern](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing), which we forward-link and which the database series covers in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern). The consistency model you are now living in — the read model trails the truth — is exactly the subject of [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and the sibling [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice). This read-side projection is the read half of CQRS, deep-dived in [event sourcing and CQRS in microservices](/blog/software-development/microservices/event-sourcing-and-cqrs-in-microservices).

#### Worked example: read-model freshness lag versus composition

Suppose the report runs 200 times a day and a month has 50,000 orders. Compare the two implementations on the dimensions that matter.

**Latency.** Composition: 27ms p99 (worked above), and that 27ms is paid *every* run. Read model: one local indexed query over the projected `order_view`, roughly **4ms** p99 — a 7× improvement — because there is no network fan-out at read time at all. The work moved to write time, where it is amortized across the event stream.

**Freshness.** This is what you pay. The read model lags the source by the event-pipeline latency: producer publishes (a few ms), Kafka delivers (single-digit ms), projector consumes and applies (a few ms). End to end, call it **~2 seconds** under normal load. So if a customer renames themselves and the report runs one second later, the report shows the *old* name. For a monthly revenue report, a 2-second lag is utterly irrelevant. For a "is this order paid right now?" check at the till, 2 seconds might be unacceptable and you would compose live instead. The freshness requirement decides the tool.

**Lag under stress.** The 2-second figure is the *healthy* lag. If the projector falls behind — a traffic spike, a slow deploy, a poison message — consumer lag grows, and the read model can trail by minutes. You must *monitor consumer lag as a first-class SLI* and alert when it exceeds, say, 30 seconds, because a silently-stale read model serving "fresh-looking" data is a nasty class of bug. **Backfill cost:** when you first build a read model, or change its shape, you must replay history to populate it; replaying 50,000 orders plus all customer and price events is a one-time batch job you must plan for (and the event log must be retained long enough to replay — see [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees)).

**The verdict.** Read models win decisively for high-QPS, lag-tolerant, wide-fan-out queries; composition wins for low-QPS, freshness-critical, narrow queries. Most systems run a portfolio: live composition for the checkout summary the customer is staring at, an event-fed read model for the analytics dashboard and the search index. Figure 8 contrasts the two for the same report side by side.

![A before and after comparison of the monthly report implemented as a single shared-database four-table JOIN versus the same report rebuilt as either a concurrent four-call fan-out or a precomputed event-fed read model with its lag](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-8.webp)

## The cost of duplicating data, in numbers

Duplication is the consequence engineers resist most, so let me make its real cost concrete, because the resistance usually comes from imagining the cost is bigger than it is. The customer's display data — name, email, maybe a tier — gets copied into the orders service (for receipts), the catalog/recommendations service (for personalization), and the shipping service (for labels). Three copies of the same fact. The instinct screams "that's wasteful and it'll drift." Let us price it.

#### Worked example: storage cost of duplicating customer data across three services

ShopFast has 5 million customers. The duplicated fields per customer — `customer_id` (16 bytes UUID), `name` (~40 bytes), `email` (~40 bytes), `tier` (8 bytes), plus row overhead — call it **~150 bytes per customer per service**. Duplicated into three services:

- Per service: 5,000,000 × 150 bytes ≈ **750 MB**.
- Across three services: 3 × 750 MB ≈ **2.25 GB** of duplicated customer data total.

At a representative managed-Postgres storage price of roughly \$0.115 per GB-month, 2.25 GB costs about **\$0.26 per month**. The *duplication* of the customer's identity across three services costs you a quarter of a dollar a month. The storage cost of duplication is, for the overwhelming majority of systems, a rounding error — disk is the cheapest resource you have. People hold onto the normalization instinct as if storage were 1985-priced, and it simply is not. (Large *blobs* — full documents, images — are a different story and you would store a reference, not a copy; but identity fields and small attributes? Copy them freely.)

The cost that is *not* a rounding error is **keeping the copies in sync**. That is real engineering: each service must subscribe to `CustomerRenamed` / `CustomerEmailChanged` events and update its copy, idempotently, handling out-of-order and duplicate delivery (see [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe)). During the propagation window the copies disagree — the orders service might still show the old name for a couple of seconds. You design for that: copies are eventually consistent, and you choose which fields even *need* propagation (a placed order's snapshot of the name at purchase time is *immutable history* and should NOT be updated when the customer renames, while a "current customer" projection should). So the real cost of duplication is not bytes; it is the event plumbing and the eventual-consistency reasoning. That cost is genuine, but it is *code and thought*, not dollars, and it buys you the decoupling — which is the whole point.

## Polyglot persistence: the right tool per service

Owning your data unlocks a freedom the shared database forbade: each service can choose the **datastore that fits its access pattern**, instead of forcing every workload into one engine. This is **polyglot persistence**, and database-per-service is its precondition — you cannot put the catalog in Elasticsearch if the catalog tables are trapped in the shared Postgres everyone else depends on.

![A tree grouping ShopFast services by access shape into relational for ACID writes with the order service on Postgres, search for faceted text with the catalog service on Elasticsearch, and key-value for low-latency lookups with the session service on Redis](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-6.webp)

Map ShopFast's services to engines by their actual access shape. The **orders service** needs strong transactional writes, multi-row consistency within an order, and relational integrity inside its own boundary — that is a relational database, Postgres, exactly right. The **catalog service** serves faceted full-text search ("waterproof hiking boots under \$120, in stock, 4+ stars") that a relational `WHERE` clause does badly and a search engine does brilliantly — Elasticsearch or OpenSearch, with the catalog projected into it. The **session service** needs sub-millisecond key-value lookups with automatic expiry and no durability requirements — Redis with a TTL, where a relational database would be absurd overkill. Each is the right tool because each service *owns* its data and is therefore free to pick.

```yaml
# docker-compose excerpt — three services, three PRIVATE stores,
# three different engines. No store is shared; no service has a
# connection string to another's store.

services:
  order-svc:
    image: shopfast/order-svc:1.4.2
    environment:
      DATABASE_URL: postgres://order_svc:***@orders-db:5432/orders
    # order-svc has NO env var pointing at catalog-es or session-redis.

  order-db:                       # private to order-svc
    image: postgres:16
    volumes: ["orders-data:/var/lib/postgresql/data"]

  catalog-svc:
    image: shopfast/catalog-svc:2.1.0
    environment:
      ES_URL: http://catalog-es:9200       # search engine, not SQL

  catalog-es:                     # private to catalog-svc
    image: elasticsearch:8.13.0
    environment: { discovery.type: single-node }

  session-svc:
    image: shopfast/session-svc:0.9.5
    environment:
      REDIS_URL: redis://session-redis:6379/0  # KV with TTL

  session-redis:                  # private to session-svc
    image: redis:7-alpine
    command: ["redis-server", "--maxmemory-policy", "allkeys-lru"]

volumes:
  orders-data:
```

Now the honest cost, because polyglot persistence is routinely oversold. Every engine you adopt is a new thing your team must **run, back up, monitor, secure, patch, scale, and be paged for**, with its own failure modes and its own operational expertise. Three engines means three backup strategies, three sets of metrics dashboards, three on-call runbooks, three CVE feeds to track. A team that adopts a new datastore for every service ends up operating a zoo it cannot feed. The discipline is: **default to one or two boring, well-understood engines** (Postgres covers an astonishing range — relational, JSON, even decent full-text and vector search), and adopt a specialized engine only when a service's access pattern genuinely justifies the operational tax. "We use Postgres for almost everything, Elasticsearch for the two services that need real search, and Redis for caching and sessions" is a mature, defensible polyglot posture. "Every service picks its favorite database" is how you end up with a Cassandra cluster that one engineer understood and who left in 2024. Polyglot persistence is a *permission*, not an *obligation*. The choice of which datastore to reach for, and the trade-offs of each, is the architect's-layer decision covered in the database series; here the point is narrower — the rule *enables* the choice, and the choice has an operational bill.

The target topology, then, looks like figure 9: each service sitting above its own private store, the engine matched to the workload, and not a single arrow crossing from one service into another service's store.

![A grid showing the order, catalog, and session services each sitting directly above its own private store with Postgres for orders, Elasticsearch for catalog, and Redis for sessions, and no arrows crossing between a service and another service's store](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-9.webp)

## Stress test: a report needs data from five services

Let me run the stress test the kit demands, because the rule's failure modes only show up under pressure. Two scenarios.

**Scenario A: a new report needs data from five services.** Product wants a "customer lifetime value" dashboard joining orders, payments, returns, support tickets, and loyalty points — five owning services. Try API composition: a five-way fan-out. Latency is fine if concurrent (`max` of five calls, maybe 30ms). But availability is `0.999⁵ ≈ 99.5%`, roughly 3.6 hours of dashboard downtime a month, and at dashboard refresh rates you are multiplying load on five services. If the dashboard is high-traffic, composition is the wrong call. So you build a read model fed by five event streams, projecting a denormalized `customer_ltv` table. Now read latency is one local query (~5ms), availability depends only on the read model's store, and the five services are shielded from the dashboard's read load entirely. The cost: five event streams to consume and keep correct, and a freshness lag of seconds. **The lesson:** the more services a query spans, the more the answer tilts from composition toward a read model, because the availability product and the load amplification of a wide fan-out get unacceptable fast. A query spanning five-plus services is almost always a read-model job, not a composition job.

**Scenario B: two services need a transaction across their data.** Checkout must reserve inventory *and* create the order, atomically — either both or neither, or you sell stock you do not have or take an order you cannot fulfill. There is no cross-database transaction. The naive fix — call inventory, then call orders, and hope — fails the moment the second call fails after the first succeeded: you have reserved inventory for an order that does not exist, leaking stock forever. The correct fix is a **saga**: a local transaction in inventory (reserve), then a local transaction in orders (create); if the order creation fails, you fire a *compensating* transaction in inventory (release the reservation) to undo the first step. The saga gives you eventual atomicity with explicit rollback logic, in place of the database's instant atomicity. It is more code, it is eventually consistent, and you must handle the compensations idempotently. This is the entire subject of [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), forward-linked. **The lesson:** when you find yourself wanting a transaction across two services, that is a signal — either your service boundary is wrong (the two things that must be atomic perhaps belong in *one* service and one transaction), or you must accept a saga. Often the cleanest answer is to redraw the boundary so the atomic operation stays inside a single service's database, where ACID still works. The desire for a cross-service transaction is frequently the boundary telling you it was misplaced.

That second lesson is the deepest one in this post. **The hard consequences of database-per-service are not just costs to absorb; they are signals about your boundaries.** A query that needs a five-way `JOIN` and a write that needs a two-service transaction are both the architecture telling you something about where the seams should be. A senior reads those signals; a junior fights them with ever-more-elaborate distributed plumbing. Sometimes the right move is not a fancier saga or a cleverer read model — it is to move two tables back into the same service because they were never two bounded contexts to begin with.

## Optimization: making the read paths production-grade

Once the data is split, the read paths are where the latency and cost live, and there are three levers, all of which we have touched and now make precise with numbers.

**Lever 1 — read models to kill fan-out.** We saw the report drop from 27ms (four-call composition) to 4ms (local read-model query), a 7× win, by precomputing. The general rule: any query that runs far more often than its underlying data changes is a candidate for a read model. ShopFast's product page joins catalog, inventory, pricing, and reviews — four services, hit on every page view at thousands of QPS. Composing it live would multiply page-view QPS across four services; a read-model "product page document" projected into one store turns it into a single ~3ms read and shields the four services from the read flood entirely. Measure the win in two numbers: **read p99** (27ms → 3ms) and **downstream QPS amplification eliminated** (4× → 0×).

**Lever 2 — caching for the freshness-tolerant.** For composition paths you keep live, cache the slow-changing pieces. Customer and catalog data change rarely; a 60-second cache on `BatchGet` cuts those calls out of most requests. If a report's customer/catalog cache hit rate is 95%, then 95% of reports skip two of three fan-out calls, dropping their latency and removing those services from the availability product for the cached path. The cost is staleness bounded by the TTL, and cache invalidation on the rare write (or just accept the TTL window). Cross-service caching has its own deep post, [caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services); the point here is that caching and read models are the same idea at different freshness/cost trade-offs — a read model is a cache you keep fresh via events instead of TTL.

**Lever 3 — deliberate denormalization at write time.** The cheapest read is the one that needs no join at all because the data was already shaped right when it was written. The orders schema snapshotting `customer_name` and `unit_price_cents` is exactly this: the receipt query needs *zero* cross-service calls because the data it needs was denormalized into the order at placement. Denormalization moves cost from the hot read path to the cold write path, where it is paid once. The measure of the win: **cross-service calls per read eliminated** — the receipt went from one customer-service call per render to zero.

#### How to measure the win

Instrument three numbers per cross-service query path and watch them in a dashboard: **read p99 latency** (the user-facing cost), **fan-out factor** (how many downstream services one read touches — your availability and amplification driver), and **read-model lag** (the freshness SLI, if you have a read model; alert above your tolerance, e.g. 30s). A healthy mature system trends fan-out toward 1 for hot paths (read models / denormalization), keeps composition for the cold, freshness-critical paths, and never lets read-model lag silently grow. If you can show a graph where the product-page p99 dropped from 27ms to 3ms and downstream QPS to the four services dropped to near zero after shipping the read model, you have measured the win and you can defend the read model's event-plumbing cost to anyone who asks.

## How you get there from a shared database

Almost nobody starts at database-per-service; almost everybody arrives from a shared database, and the migration is the hardest, scariest part of this whole topic because you are operating on the live data tier of a running business. The cardinal rule is **separate logically before you separate physically, and migrate one bounded context at a time** — never a big-bang.

![A six-event timeline of migrating off a shared database from auditing cross-table JOINs through splitting into per-service schemas, locking grants, replacing JOINs with APIs and read models, moving the hottest schema to its own database, and finally leaving the shared database empty](/imgs/blogs/database-per-service-the-rule-that-defines-microservices-7.webp)

Here is the sequence that has worked for me, mapped to figure 7. **Month 0 — audit.** Find every cross-table `JOIN` and every write that crosses what will become a service boundary. This is the inventory of pain; you cannot plan the migration until you know which couplings exist. Database query logs and a `grep` for table names across the codebase get you most of the way. **Month 1 — split into per-service schemas inside the same instance.** Move each service's tables into its own schema. No physical change yet, no new database, low risk — it is mostly `ALTER TABLE ... SET SCHEMA` and updating connection search paths. **Month 2 — lock the grants.** This is the moment of truth: configure each service's role so it can only touch its own schema (the SQL from earlier). Every cross-schema `JOIN` and `SELECT` now fails loudly — in staging, ideally — and you have a worklist of exactly the couplings you must break. **Month 3 — replace the broken JOINs** with API calls (for the freshness-critical, low-volume ones) and read models (for the hot, lag-tolerant ones), and replace cross-service writes with sagas. This is the bulk of the engineering work, and you do it incrementally, one query at a time, behind the grants you already locked. **Month 5 — promote the hottest schema to its own database** for failure and scale isolation, once the logical separation is solid and you have a service whose load justifies the operational cost. **Month 8 — the shared database holds nothing**; every schema has graduated, and you can decommission the integration database that started it all.

The technique that makes the JOIN-replacement step survivable is the **strangler fig**: you do not rewrite the access path in one cut, you route new reads through the new API or read model while the old `JOIN` still works, compare results to catch drift, then flip traffic and remove the old path. We forward-link the full migration playbook to [strangler fig: migrating a monolith to microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) and to [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), which catalogs every way this migration goes wrong. The data-tier mechanics — how to split a hot table, how to keep two copies consistent during the cutover — lean on [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) and [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), because CDC is often how you keep a new read model populated from a still-shared source during the transition. The honest truth: this migration takes months to years for a real system, it never goes as cleanly as the six-step timeline implies, and the single most common failure is *stopping halfway* — splitting some services while leaving a shared `customers` table that quietly recouples everything, which is how you build a distributed monolith on purpose.

## When to reach for this (and when not to)

Database-per-service is non-negotiable *if you are doing microservices at all* — it is the defining rule, and a microservices architecture with a shared database is not microservices, it is a distributed monolith and will give you every cost of microservices with none of the benefits. But that conditional ("if you are doing microservices at all") is doing enormous work, and the honest senior position is that you should question it before you pay this rule's price.

**When the rule is worth its cost:** you have multiple teams that need to deploy and evolve independently, and the coordination tax of a shared schema is actively slowing them down. You have services with genuinely divergent data needs — one needs search, one needs sub-ms KV, one needs strong relational writes — and the shared database serves all of them badly. You have isolation requirements (a payments database with compliance boundaries) that demand physical separation. You have scale where one data workload needs to be tuned and scaled independently of the others. In all of these, the cost of the rule buys you something proportionate.

**When the monolith's shared database wins:** you are a small team, early in a product, where the dominant risk is not shipping fast enough, not coordination overhead. A single well-structured database with a clean module boundary inside one application gives you ACID transactions, free joins, referential integrity, and one thing to operate — and you can draw clean *logical* boundaries (separate schemas, no cross-module queries) inside it that make a future split far easier. This is the [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith), and it is the right answer far more often than the conference talks admit. The two-pizza team with one product does not need eleven databases; it needs one database with disciplined boundaries and the option to split later. **Splitting the database is the most expensive and least reversible decision in the whole microservices transition** — you can merge two services back into one repo in an afternoon, but un-splitting two databases is a genuine data migration. So split data last, split it deliberately, and only split the data that a real, present pain forces you to. The rule defines microservices; whether you should be doing microservices at all is the prior question, and the answer is "less often than you think" — which is the entire thesis of [what are microservices and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them).

## Trade-offs at a glance

To consolidate the decision, here is the isolation spectrum as a table you can take to a design review. The figure-3 matrix shows the same five dimensions; this adds the "when it wins" column the kit asks for.

| Option | Isolation | Blast radius | Ops cost | Cross-query | When it wins |
| --- | --- | --- | --- | --- | --- |
| Shared schema | None | Whole DB / all teams | Lowest (one DB) | Free in-DB JOIN | Never for microservices; it *is* the distributed-monolith trap |
| Separate schema, shared instance | Logical | One schema (but shared instance can still take all down) | Low (one DB to run) | No JOIN; compose / read model | Small-to-mid systems; early migration; the pragmatic default at rung 1 |
| Separate database per service | Strong | One database | Medium (many DBs) | No JOIN; compose / read model | Services with divergent load or real failure-isolation needs |
| Separate cluster / engine | Physical + polyglot | One cluster only | Highest (many engines) | No JOIN; compose / read model | The few services whose access pattern or compliance demands a distinct engine |

And the query-implementation trade-off, the other decision you make constantly:

| Approach | Freshness | Read latency | Availability | Best for |
| --- | --- | --- | --- | --- |
| API composition (concurrent fan-out) | Real-time | Slowest callee (~27ms for 4 calls) | Product of all callees (drops fast) | Low-QPS, freshness-critical, narrow fan-out |
| Event-fed read model | Lagged (~2s, can grow) | One local query (~4ms) | One store only | High-QPS, lag-tolerant, wide fan-out |
| Write-time denormalization | Snapshot at write | Zero cross-service calls | Owning store only | Immutable history (order line prices, receipts) |

## Case studies

**Amazon's move off the shared Oracle database.** For its first decade, much of Amazon ran against a small number of large, shared relational databases — famously a heavy Oracle footprint — and the company hit exactly the wall this post is about: the shared database became the integration point, and teams could not evolve their data or scale independently because they all shared the same tables and the same monster database. The well-documented response, over many years, was a sweeping move toward service-oriented architecture in which each service owned its own data behind a hard API boundary — the era that produced the internal mandate (associated with Jeff Bezos's "API mandate") that all teams expose data and functionality only through service interfaces, with *no* direct database access between teams, and that eventually birthed AWS itself, including purpose-built databases (DynamoDB and others) precisely so services could pick the right store. The lesson is the one this whole post argues: at sufficient scale, the shared database is the binding constraint on organizational velocity, and breaking it — expensive and multi-year as it was — was what let thousands of teams ship in parallel. Treat the order-of-magnitude as the lesson, not any precise internal number.

**Shopify's schema isolation inside a modular core.** Shopify is the instructive counter-example: rather than shattering into thousands of services with thousands of databases, Shopify invested heavily in a *modular monolith* with strong internal boundaries, including disciplined data ownership where modules are not supposed to reach into each other's tables, enforced by tooling. It is a reminder that the *principle* — each component owns its data and you don't cross-query someone else's tables — delivers most of the decoupling benefit and can be enforced *inside* a monolith via schema isolation and lint/tooling, without paying the full operational tax of physically separate databases. You can have the rule's discipline at rung 0.5 (logical ownership inside one process) long before you pay for rung 2. Many teams would be better served by Shopify's path than by a premature service explosion.

**The shared-database coupling horror story.** The generic but utterly real failure mode — and the one ShopFast lived in the intro — is the team that "does microservices" with a shared database and discovers, usually during an incident, that they have a distributed monolith: a schema change in one service breaks others without warning, no service can deploy without coordinating data changes, one service's runaway query degrades everyone via the shared instance, and a connection leak in a minor service starves a critical one of connections and takes down checkout. This pattern is so common it has a name in the literature — the **distributed monolith** — and the engineering-blog post-mortems describing it (teams who split their services but kept a shared database and got the worst of both worlds) are numerous enough that it is the default cautionary tale of the field. The concrete lesson: the box-and-arrow diagram lies; the only honest test of whether you have microservices is whether two services can see each other's tables, and if they can, you do not, no matter how many deployable units you have. Segment's famous reversal — splitting into many services, drowning in the operational and coupling cost, and consolidating back toward a monolith — is the adjacent cautionary tale: the cost of getting decomposition (including data decomposition) wrong is high enough that going *back* was the right call.

## Key takeaways

- **The defining test of microservices is data ownership, not box count.** If two services can read or write each other's tables, you have a distributed monolith, full stop — no matter how many repos, pipelines, and deployments you have. The honest question in any review is "can service A see service B's tables?"
- **A shared database is an integration database, and an integration database recouples your teams at the most rigid layer there is.** It quietly destroys independent deployability, independent scaling, and schema evolution, and none of it shows on the architecture diagram.
- **"Owns its data" means exclusive read *and* write access to a private store, reached only through the service's API or its events.** Readers couple to your schema just as hard as writers; the half-measure of "let them just read it" is what kills you.
- **Splitting the data costs you five guarantees the database gave you free:** cross-service JOINs, distributed ACID transactions, normalization, single-query reads, and referential integrity. You now provide equivalents — composition, sagas, denormalization, read models, application-level integrity — in code, at the price of latency and weaker consistency.
- **Isolation is a spectrum; pick the weakest rung that still gives a private schema and climb only on real pain.** Separate schema with locked grants on a shared instance is the pragmatic default; promote individual services to their own database or cluster when load, failure-isolation, or technology fit demand it.
- **Enforce the rule with mechanism, not memos.** Role grants that make cross-schema queries fail in CI, and network policies that make cross-database connections impossible, turn a discipline problem into an enforced invariant.
- **Cross-service queries are composition (live, fresh, fan-out cost) or read models (precomputed, fast, lagged).** Tilt toward read models as fan-out and QPS grow; keep composition for narrow, freshness-critical, low-volume reads. Measure read p99, fan-out factor, and read-model lag.
- **Duplication is a design choice, not a smell; the bytes are a rounding error and the real cost is keeping copies in sync via events.** Denormalize identity and small attributes freely; reference, don't copy, large blobs.
- **A wanted cross-service transaction or a five-way JOIN is usually your boundaries talking.** Before building elaborate distributed plumbing, ask whether the two things that must be atomic actually belong in one service — sometimes the right fix is to redraw the seam, not to write a fancier saga.
- **Split the data last and least.** It is the most expensive, least reversible step in the whole transition. The modular monolith with disciplined schema ownership gives you most of the benefit at a fraction of the cost — and it is the right answer more often than you think.

## Further reading

- *Microservices Patterns*, Chris Richardson — the canonical treatment of database-per-service, API composition, the saga pattern, and CQRS read models, with running code. The chapters on managing transactions and querying across services are the direct deep-dive for this post.
- *Building Microservices* (2nd ed.), Sam Newman — the decomposition and data-ownership chapters, plus the most honest published discussion of *how hard* splitting a database actually is and the strangler-fig techniques to do it incrementally.
- Martin Fowler, "Integration Database vs Application Database" and "Database per Service" (martinfowler.com / microservices.io) — the original naming of the integration-database anti-pattern and the pattern that replaces it.
- *Designing Data-Intensive Applications*, Martin Kleppmann — for the mechanisms underneath: replication, partitioning, and the consistency models you now live in once your data is distributed.
- This series and the database series, for the mechanisms this post forward- and cross-links: [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), [the transactional outbox and reliable event publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing), [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice), [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith), [saga pattern (database)](/blog/software-development/database/saga-pattern-distributed-transactions), [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual), and [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).

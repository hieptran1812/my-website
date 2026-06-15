---
title: "Monolith First and the Modular Monolith: The Architecture Most Teams Should Actually Ship"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Why you should start with a monolith, what a modular monolith really is, and how to build one so that the first service you ever extract costs hours instead of a quarter."
tags:
  [
    "microservices",
    "modular-monolith",
    "monolith-first",
    "software-architecture",
    "distributed-systems",
    "domain-driven-design",
    "backend",
    "bounded-context",
    "architecture-fitness",
    "migration",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/monolith-first-and-the-modular-monolith-1.webp"
---

A team I worked with spent the better part of a year building a microservices platform for a product that had not yet found a thousand paying customers. They had eleven services, a service mesh, a message bus, three databases, and a deploy pipeline so elaborate that shipping a one-line copy change to a button label required touching four repositories. They were proud of it. They had read all the right blog posts. And they were drowning: every feature crossed two or three service boundaries, every boundary needed a network call and an API contract and a retry policy, and every incident involved staring at four dashboards trying to work out which of eleven things had actually broken. The architecture that was supposed to let them move fast had made them slower than the scrappy monolith they replaced. The cruel irony is that the boundaries they had baked in — the ones that now cost a sprint to move — were *guesses*, made on a whiteboard, before they had a single real usage pattern to learn from.

This post is the argument for the road they should have taken, and the road most teams should take: **start with a monolith, and if you have any architectural ambition at all, make it a *modular* monolith** — a single deployable unit that is, on the inside, cleanly split into modules with explicit interfaces, separate schemas, and dependency rules the build itself enforces. The modular monolith is not a way-station you tolerate until you can afford the "real" architecture. For most companies it *is* the real architecture, the one they should run for years, possibly forever. And for the minority who genuinely outgrow it, it is the thing that makes the eventual split cheap, because the seams are already there.

![A side by side comparison showing a tangled big ball of mud monolith on the left and a clean modular monolith with separate schemas and explicit interfaces on the right](/imgs/blogs/monolith-first-and-the-modular-monolith-1.webp)

The figure above is the distinction the whole article turns on. On the left is the thing everyone is scared of when they hear "monolith" — a big ball of mud, one deploy unit with tangled imports, one shared schema everything joins across, and no boundaries at all so any code can reach into any other code. On the right is a modular monolith: still one deploy unit, still one process you can run on a laptop, but internally partitioned into modules that each own their schema, expose an explicit interface, and cannot reach into each other's internals because the build won't compile if they try. Both ship as one artifact. Only one of them is something you can grow. By the end of this article you will be able to build the right-hand one — lay out its modules, write the dependency rules that keep it honest, design the interfaces so that future extraction is a transport swap rather than a rewrite — and you will be able to read the signals that tell you when, if ever, it is time to break the first piece off.

This is the practitioner's layer. The mechanics of *why* a distributed transaction is hard, or *how* an event log gives you exactly-once, live in their own deep-dives, which I link to where they matter. Here the question is narrower and more immediate: what should you actually build on Monday, and how do you build it so that the choices you make today do not become the regrets you pay for in eighteen months?

## 1. Why "MonolithFirst" is the senior default

In 2015 Martin Fowler wrote a short, much-cited piece called "MonolithFirst." The argument is deceptively simple and, in my experience watching teams succeed and fail, almost always correct: **you should not start a new project with microservices, even if you are sure the system will be large enough to need them eventually.** Start with a monolith, learn the domain, and split later when you understand where the boundaries actually are.

The reasoning rests on one uncomfortable fact: at the start of a project, you do not know where the boundaries go. The whole value of microservices comes from cutting the system along the right seams — the seams where a change to one capability rarely forces a change to another, where the data naturally clusters, where one team can own a piece end to end. Those seams are not visible on day one. They are an *emergent* property of how the domain actually behaves once real users hit it, and you discover them by building the thing and watching where the friction lands. A senior who has decomposed a few monoliths will tell you the same thing every time: the boundaries you would have drawn at the start are not the boundaries you draw after a year of operating the system, and they are not even close.

So why is starting with microservices so costly when you guess wrong? Because of the cost-of-change curve, which I dig into properly in [Evolutionary Architecture: Designing for Change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change). A module boundary that is *wrong* inside a monolith is cheap to move — you re-slice the code, the compiler tells you everything that breaks, you run the tests, you ship. A *service* boundary that is wrong is expensive to move, because it has hardened into a network API, a separate database, a deploy pipeline, a team's ownership, and very likely a few contracts other teams now depend on. You have to coordinate across teams, migrate data across databases, version an API, and do it all without downtime. The same logical mistake — "I put the line in the wrong place" — costs an order of magnitude more to fix once it is a service. Microservices do not just defer the boundary decision; they *bake it into concrete* before you have the evidence to make it well.

There is a second-order reason that is even more decisive in practice: **microservices are an operational tax you pay continuously, whether or not you are getting value from them.** Every service needs a deployment story, a health check, a dashboard, an on-call runbook, a retry-and-timeout policy on every call into it, a way to trace a request across it, and a story for what happens when it is down mid-request. A monolith pays none of that. A method call does not time out, does not partially fail, does not need a circuit breaker, and shows up in a single stack trace. When you split into services *before* you have a reason to, you take on all of that cost and get none of the benefit, because the benefit — independent deployment, independent scaling, independent ownership — only matters once you have multiple teams, divergent scaling needs, or a blast-radius problem. With one team and one workload, you have just made your life harder for free. (We catalogue exactly when those benefits arrive in the sibling post [What Are Microservices and When Not to Use Them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them).)

Fowler's own nuance is worth keeping: there is a competing camp ("DesignFirst") that says if you already deeply understand the domain — because you have built this exact thing before — you can start with services and skip the monolith phase. That is real, and occasionally right. But it requires a level of certainty about boundaries that almost no greenfield team actually has, and the failure mode of being wrong is so much more expensive in the services case that the asymmetry should make you default to monolith-first unless you can articulate, specifically, why your boundaries are already known.

So "monolith first" is the senior default not because monoliths are good and services are bad, but because *deferring an expensive, hard-to-reverse decision until you have evidence* is good, and microservices force that decision early. The rest of this article is about doing monolith-first *well* — because a monolith built carelessly becomes the big ball of mud everyone is rightly afraid of, and a monolith built with discipline becomes the modular monolith that makes the future easy.

## 2. What a modular monolith actually is

The word "monolith" carries trauma. People hear it and picture a 400,000-line codebase where the shipping logic imports the billing logic which imports the user-profile logic which, three layers down, runs a SQL query that joins across the entire schema, and where changing the tax calculation somehow breaks the password-reset email. That is a real thing, and it is awful, but it is not the *monolith* that is awful — it is the *lack of internal structure*. The deploy topology (one unit) and the internal structure (modular or muddy) are independent choices, and people conflate them constantly.

A **modular monolith** decouples those two things. It is:

- **One deployable unit.** One build artifact, one process (or one fleet of identical processes behind a load balancer), one deploy. You run it on a laptop. There is no network between the modules. A call from one module to another is an in-process function call, measured in nanoseconds to microseconds, that cannot partially fail, time out, or get lost on the network.

- **Internally split into modules**, where each module is a **bounded context** — a chunk of the domain with a clear responsibility and a name a business person would recognize: catalog, orders, payments, inventory. (The discipline of finding those contexts is domain-driven design, which gets its own treatment in [Service Boundaries with Domain-Driven Design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design); here we just use the result.)

- **Communicating through explicit, narrow interfaces.** A module does not reach into another module's internals. It calls a small, deliberately designed public API — a port — and everything behind that port is private. If catalog wants a product's price, it does not run a SQL query against the payments tables; it calls `paymentsModule.getCharge(orderId)` or, better, payments does not even know about catalog.

- **Owning a separate schema per module.** Catalog's tables live in a `cat_` schema (or a `catalog` Postgres schema, or a separate logical database); payments' tables live in `pay_`. No module reads or writes another module's tables. There are **no cross-module database joins**. This is the single most important and most violated rule, and it is the one that makes everything else possible.

- **With dependency rules enforced by the build, not by good intentions.** A test fails the CI build if the orders module imports a class from inside the payments module instead of going through its public port. Discipline that depends on code review is discipline that erodes; discipline that fails the build is discipline that lasts.

The mental shift here is to treat module boundaries as **future service boundaries**. Everything you do to make a modular monolith clean is, not coincidentally, exactly what you would need to do to extract a service. Separate schemas mean a module's data can be physically moved to its own database without untangling joins. A narrow public interface means a module can be reached over the network without redesigning how callers talk to it. A build-enforced dependency graph means you know — provably, not hopefully — that nothing is secretly coupled in a way that will surprise you during extraction. The modular monolith is microservices' architecture *minus the network*. You get the boundary discipline now, in-process and cheap, and you add the network only when and if you need it.

That last point is what makes the modular monolith such an underrated choice. It is not "the simple thing you do before the real thing." It is "the architecture that has 90% of the structural benefits of microservices and almost none of their operational cost," and for most teams the remaining 10% never becomes worth the cost.

## 3. ShopFast: laying out the modules

Let me make this concrete with a running example I will carry through the whole article: **ShopFast**, a mid-sized e-commerce system. It has four obvious bounded contexts, and they map cleanly onto modules.

![A tree diagram of the ShopFast application as one deploy unit branching into catalog, orders, payments, and inventory modules, each with its own schema prefix and a public API](/imgs/blogs/monolith-first-and-the-modular-monolith-2.webp)

- **Catalog** owns products, descriptions, categories, prices-as-displayed, search. Its schema is `cat_`.
- **Orders** owns the shopping cart, checkout flow, order lifecycle, order history. Its schema is `ord_`. Orders is the *orchestrator* — checkout is the use case that touches everyone else.
- **Payments** owns charging a card, refunds, the integration with the payment service provider, payment records. Its schema is `pay_`.
- **Inventory** owns stock levels, reservations, replenishment, the warehouse-facing view. Its schema is `inv_`.

Each module exposes a small public API. Catalog exposes a `ProductLookup` port. Payments exposes a `PaymentGateway` port. Inventory exposes a `StockLedger` port. Orders, being the orchestrator, mostly consumes the others. These public APIs are the *only* surface other modules are allowed to touch. Everything else inside each module — the entities, the repositories, the database tables, the helper services — is private.

Here is what the folder layout looks like. I will show it in a Java/Spring-ish style because it makes the module boundary visually obvious, but the same structure works in Go (one package per module), Python (one top-level package per module with import-linter rules), TypeScript (one workspace per module), or any language with a notion of public versus private. The point is the *shape*, not the language.

```bash
shopfast/
├── build.gradle                      # one build, one artifact
├── shared-kernel/                    # truly shared types ONLY: Money, ids, errors
│   └── src/main/java/com/shopfast/shared/
│       ├── Money.java
│       ├── ProductId.java
│       └── OrderId.java
├── catalog/
│   └── src/main/java/com/shopfast/catalog/
│       ├── api/                       # PUBLIC: other modules import only this
│       │   ├── ProductLookup.java     # the port interface
│       │   └── ProductView.java       # the DTO it returns
│       └── internal/                  # PRIVATE: nobody outside catalog touches this
│           ├── domain/Product.java
│           ├── infra/ProductRepository.java
│           └── ProductLookupImpl.java
├── orders/
│   └── src/main/java/com/shopfast/orders/
│       ├── api/OrderService.java
│       └── internal/
│           ├── domain/Order.java
│           ├── CheckoutHandler.java   # orchestrates catalog + inventory + payments
│           └── infra/OrderRepository.java
├── payments/
│   └── src/main/java/com/shopfast/payments/
│       ├── api/
│       │   ├── PaymentGateway.java
│       │   └── ChargeResult.java
│       └── internal/...
└── inventory/
    └── src/main/java/com/shopfast/inventory/
        ├── api/
        │   ├── StockLedger.java
        │   └── Reservation.java
        └── internal/...
```

The convention that does the heavy lifting is the `api` versus `internal` split inside every module. The `api` package is the module's public face: interfaces and the data-transfer objects they exchange. The `internal` package is everything else, and *no code outside the module is allowed to import anything from another module's `internal` package*. That is the rule. In a moment we will make the build enforce it, but first notice how little is in `api`: a port interface and a couple of DTOs. The public surface of a module should be small and deliberately designed, because that surface is the thing you will least be able to change later — it is the module's contract, the modular-monolith equivalent of the public API in the cost-of-change curve. Keep it narrow now and you keep your options open later.

A note on the **shared kernel**. There is always a small set of types that genuinely belong to everyone — money, IDs, common error types, maybe a date-range value object. Those go in a tiny `shared-kernel` module that all modules may depend on. The discipline is to keep the kernel *small and stable*: it is a magnet for coupling, and the moment business logic creeps into it, you have a new big ball of mud forming in the basement. The rule I use is that the shared kernel may contain value objects and constants but no behavior that belongs to any single bounded context, and it should change roughly never. If you find yourself editing the shared kernel weekly, something is in there that should not be.

## 4. The dependency rules that keep it honest

Modules are only modules if the boundaries hold. The difference between a modular monolith and a big ball of mud is *entirely* in whether the dependency rules are actually enforced — and on a real team, over a real year, with real deadline pressure, "actually enforced" means "the build fails," not "we agreed in a design doc."

![A graph showing the orders module depending on the catalog, payments, and inventory ports and on a shared kernel, with no module importing another module's internals](/imgs/blogs/monolith-first-and-the-modular-monolith-3.webp)

The figure shows ShopFast's allowed dependency graph. Orders, the orchestrator, depends on the *ports* of catalog, payments, and inventory — never their internals. The leaf modules depend on the shared kernel and nothing else of each other. There are no cycles: orders calls catalog, but catalog never calls back into orders. This acyclic, ports-only shape is what makes the graph a true DAG and what makes each module independently extractable. A cycle between two modules means they are not really two modules — they are one module pretending, and you would have to extract them together.

The rules, stated precisely:

1. **A module may depend on another module's `api` package only — never its `internal`.**
2. **No cross-module database access.** A module reads and writes only its own schema. No joins across module schemas, no `SELECT` against another module's tables.
3. **No dependency cycles between modules.** The module graph is a DAG.
4. **The shared kernel may be depended on by anyone but may depend on nothing** (except the standard library).

To enforce rule 1 and rule 3 on the JVM, the standard tool is **ArchUnit** — a library that lets you write architecture rules as ordinary unit tests, so they run in CI and fail the build. Here is the test that protects ShopFast's boundaries:

```java
package com.shopfast.architecture;

import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.importer.ClassFileImporter;
import com.tngtech.archunit.library.dependencies.SlicesRuleDefinition;
import com.tngtech.archunit.lang.ArchRule;
import org.junit.jupiter.api.Test;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;

class ModuleBoundaryTest {

    private final JavaClasses classes =
        new ClassFileImporter().importPackages("com.shopfast");

    // Rule 1: nobody may import another module's internal package.
    @Test
    void modulesMustNotDependOnEachOthersInternals() {
        ArchRule rule = noClasses()
            .that().resideOutsideOfPackage("com.shopfast.payments..")
            .should().dependOnClassesThat()
            .resideInAPackage("com.shopfast.payments.internal..")
            .because("payments internals are private; call the PaymentGateway port");
        rule.check(classes);
    }

    // Rule 3: the module graph must be acyclic.
    @Test
    void modulesMustBeFreeOfCycles() {
        SlicesRuleDefinition.slices()
            .matching("com.shopfast.(*)..")   // one slice per module
            .should().beFreeOfCycles()
            .check(classes);
    }
}
```

That is the whole trick. The first test says no class outside the `payments` package may touch anything in `payments.internal`; you write one per module (or generalize it). The second test slices the codebase by module and asserts there are no cycles. The moment a developer, under deadline pressure, types `import com.shopfast.payments.internal.PaymentRecord` into the orders module to "just grab the charge amount real quick," the build goes red and the pull request cannot merge. The boundary defends itself. This is the modular-monolith analogue of an **architecture-fitness function**, the build-enforced architectural invariant I describe in [Evolutionary Architecture: Designing for Change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) — a test that fails when the system drifts away from its intended shape.

In Python the equivalent is **import-linter**, configured in `setup.cfg` or `pyproject.toml`:

```ini
[importlinter]
root_package = shopfast

[importlinter:contract:module-independence]
name = Modules talk only through public api packages
type = forbidden
source_modules =
    shopfast.orders.internal
    shopfast.catalog.internal
    shopfast.payments.internal
    shopfast.inventory.internal
forbidden_modules =
    shopfast.orders.internal
    shopfast.catalog.internal
    shopfast.payments.internal
    shopfast.inventory.internal

[importlinter:contract:layers]
name = Module layering
type = layers
layers =
    shopfast.orders
    shopfast.payments
    shopfast.catalog
    shopfast.inventory
    shopfast.shared
```

The `forbidden` contract says no module's internals may import another module's internals; the `layers` contract enforces the dependency direction (orders may depend on the lower layers but not vice versa). Run `lint-imports` in CI and the boundary is enforced exactly as on the JVM. Go gets this almost for free: put each module in its own package, keep the public port in the package's exported (capitalized) symbols and everything else unexported (lowercase), and `go build` plus a small `go vet`-style linter or an internal-package convention (`internal/` directories are compiler-enforced as private to their parent) does the rest. The tooling differs; the principle is identical: **the wrong import does not compile.**

Rule 2 — no cross-module database access — usually cannot be enforced by a static import check, because it is a runtime SQL concern. The enforcement here is structural: each module is given a data source (or a repository) scoped to its own schema, and modules simply have no handle on each other's tables. If catalog's repositories are wired with a connection whose search path or schema is `cat_` and it has no way to address `pay_` tables, the join is impossible to write. We will see the schema separation in section 6. The combination — static checks for code dependencies, structural scoping for data — is what keeps the whole thing honest.

## 5. Designing the interface to mirror a future RPC

Here is where craft separates a modular monolith you can extract from one that merely *looks* modular. The public port of a module should be designed as if it were already a remote call, even though today it is an in-process method call. If you do that, extraction later is a transport swap, not a redesign.

![A pipeline showing an orders call to the payment gateway port passing through an interface to an in-process implementation today and a future gRPC stub, where only the transport changes](/imgs/blogs/monolith-first-and-the-modular-monolith-5.webp)

The figure shows the path. Orders calls the `PaymentGateway` interface. Today that interface is satisfied by an in-process implementation — a plain object, a method call costing roughly a microsecond, that cannot fail in the ways a network call can. Tomorrow, if payments becomes a service, the *same interface* is satisfied by a gRPC client stub — a method call costing roughly five milliseconds that can time out, retry, and fail. The caller does not change. That is the payoff of designing the port like a network contract from the start.

What does "designing it like a network contract" mean in practice? A few specific disciplines:

**Pass data, not object graphs.** A remote call cannot send you a live `Order` entity with lazy-loaded relationships and a database session attached; it can only send a serialized snapshot. So design your ports to take and return plain data-transfer objects — value objects with primitive fields — not your internal domain entities. This forces you to think about *what data actually crosses the boundary*, which is exactly the question a network API forces, and it prevents the insidious coupling where the payments module ends up depending on the shape of the orders module's internal entity.

```java
package com.shopfast.payments.api;

// The PUBLIC port for the payments module. Note: it traffics in plain
// DTOs (ChargeRequest / ChargeResult), not internal domain entities.
// This signature would translate one-to-one into a protobuf RPC.
public interface PaymentGateway {

    ChargeResult charge(ChargeRequest request);

    RefundResult refund(RefundRequest request);
}
```

```java
package com.shopfast.payments.api;

import com.shopfast.shared.Money;
import com.shopfast.shared.OrderId;

// A flat, serializable request. Every field is a primitive or a
// shared-kernel value object. No references into the orders module.
public record ChargeRequest(
    OrderId orderId,
    Money amount,
    String currency,
    String paymentMethodToken,   // tokenized, never a raw card number
    String idempotencyKey        // designed in from day one (see below)
) {}
```

**Make every mutating call idempotent from day one.** A network call can be lost after it succeeded, so the caller retries, so the callee can receive the same "charge this card" request twice. The standard defense is an idempotency key: the caller generates a unique key per logical operation, the callee records it, and a repeat with the same key returns the original result instead of charging twice. In a monolith this seems like overkill — an in-process call does not get duplicated. But if you bake the idempotency key into the *interface* now, the day you put a network between caller and callee you are already safe, and you have not had to change the contract. (The full treatment of why at-least-once delivery needs this is in [Idempotency and Deduplication: Making At-Least-Once Safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe).) Designing for idempotency early is the single highest-leverage habit for cheap extraction.

**Return results, not exceptions, for expected failures.** A card decline is not an exceptional condition; it is an expected outcome. Across a network you cannot rely on exception propagation anyway. So model expected outcomes as values in the result type:

```java
package com.shopfast.payments.api;

public sealed interface ChargeResult {
    record Succeeded(String chargeId, java.time.Instant at) implements ChargeResult {}
    record Declined(String reason) implements ChargeResult {}
    record GatewayUnavailable(boolean retryable) implements ChargeResult {}
}
```

The caller — the orders module's checkout handler — then handles those outcomes explicitly, which is exactly the code it would need to write if payments were remote and could return a "gateway unavailable, retryable" response. Notice how this nudges you toward thinking about partial failure *before* you have a network: the `GatewayUnavailable(retryable)` case is meaningless in a pure in-process world, but designing it in now means the day extraction happens, the orchestration logic is already failure-aware.

Here is the orchestrator using all three ports through their public interfaces only:

```java
package com.shopfast.orders.internal;

import com.shopfast.catalog.api.ProductLookup;
import com.shopfast.inventory.api.StockLedger;
import com.shopfast.payments.api.PaymentGateway;
import com.shopfast.payments.api.ChargeResult;

// Orders is the orchestrator. It depends on the PORTS of the other
// three modules and nothing else. Wired by the DI container.
public class CheckoutHandler {

    private final ProductLookup catalog;
    private final StockLedger inventory;
    private final PaymentGateway payments;

    public CheckoutHandler(ProductLookup catalog,
                           StockLedger inventory,
                           PaymentGateway payments) {
        this.catalog = catalog;
        this.inventory = inventory;
        this.payments = payments;
    }

    public CheckoutResult checkout(Cart cart, String idempotencyKey) {
        var priced = catalog.priceCart(cart);            // in-process today
        var reservation = inventory.reserve(priced.lines());
        if (!reservation.ok()) {
            return CheckoutResult.outOfStock(reservation.shortfall());
        }
        var charge = payments.charge(new ChargeRequest(
            cart.orderId(), priced.total(), "USD",
            cart.paymentToken(), idempotencyKey));

        return switch (charge) {
            case ChargeResult.Succeeded s ->
                CheckoutResult.confirmed(s.chargeId());
            case ChargeResult.Declined d -> {
                inventory.release(reservation.id());      // compensate
                yield CheckoutResult.paymentDeclined(d.reason());
            }
            case ChargeResult.GatewayUnavailable g -> {
                inventory.release(reservation.id());
                yield CheckoutResult.tryAgainLater();
            }
        };
    }
}
```

Look at what this code already does that it would need to do as distributed services: it reserves inventory before charging, compensates (releases the reservation) when the charge fails, and handles the "gateway unavailable" case distinctly. That compensation logic is, in embryo, a **saga** — the pattern for keeping data consistent across services without a distributed transaction, which gets its full treatment in [The Saga Pattern: Distributed Transactions](/blog/software-development/database/saga-pattern-distributed-transactions). The beautiful thing is that you write it *in the monolith*, where you can also fall back to a real database transaction if you want, and where you can test the compensation paths trivially. Then when payments and inventory become separate services with separate databases and a true distributed transaction is off the table, your orchestration is already shaped correctly. You have practiced the distributed pattern in the safety of a single process.

#### Worked example: the cost of moving a boundary you got wrong

Let me put numbers on the central claim that a wrong boundary is cheap in a monolith and expensive in a service, because this is the entire reason monolith-first wins.

Suppose ShopFast originally lumped "payments" and "billing" (subscriptions, invoices, dunning) into a single module, then discovered after a year that they have completely different change rhythms and should be separate. In the **modular monolith**, fixing this means: split the package into `payments` and `billing`, move the relevant classes (the IDE does this mechanically), split the `pay_` schema into `pay_` and `bill_` (a migration that renames tables and moves a few foreign keys — a day of careful work on a copy plus a tested migration), update the two ArchUnit tests, and ship. Call it **one engineer, three to five days**, all of it reversible, all of it caught by the compiler and the test suite if you get a reference wrong. Total blast radius: zero customers, zero other teams, one deploy.

Now suppose the same mistake had been baked into a **microservices** layout: payments-and-billing was already a single *service* with its own database, its own deploy pipeline, and three other services calling its API. Splitting it now means: stand up a new billing service (repo, CI, dashboards, on-call, deploy pipeline — a week of platform work), carve the billing tables out of the shared database into a new one and migrate live data without downtime (dual-write, backfill, cutover — two to three weeks if you are disciplined, and this is where outages happen), version and migrate the API so the three callers can move from the old combined endpoint to the two new ones (a consumer-driven contract change coordinated across three teams — two to four weeks of cross-team scheduling), and run both old and new in parallel during the transition. Call it **two to three engineers across three teams for six to ten weeks**, with real risk of a data-migration incident, and several places where a mistake is visible to customers.

Same logical mistake. In the monolith it is a tidy refactor measured in days. In the services world it is a cross-team program measured in weeks-to-months with downtime risk. That ratio — roughly **3 days versus 8 weeks, call it 10× to 15×** — is the cost of guessing the boundary wrong, and it is precisely why you do your boundary-guessing *inside the monolith* where mistakes are cheap, and only promote a boundary to a service once you are confident it is right.

## 6. Separate schemas: the data discipline that makes or breaks it

Code boundaries are the easy part; engineers can be taught not to import the wrong package, and a linter backs them up. The boundary that *actually* determines whether your monolith is extractable is the **data** boundary, and it is the one teams violate without even noticing, because a cross-module join is so convenient and the database does not have an ArchUnit test.

![A layered stack diagram of a single module showing the public API on top, then application services, the private domain model, persistence adapters, and the module's own schema at the bottom](/imgs/blogs/monolith-first-and-the-modular-monolith-6.webp)

The figure shows the internal layering of one module — say payments. The public API sits on top and is the only thing other modules see. Below it, the application services and the private domain model. At the bottom, the persistence adapters and the module's *own* schema (`pay_` tables). Other modules can see only the top layer; the schema at the bottom is as private as the domain model. This is the rule that the database-per-service principle (the defining rule of microservices, covered in [Database Per Service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices)) borrows from — except here we get it without the operational cost of actually running separate databases, by using **separate schemas within one database**.

Concretely: each module's tables are namespaced. In Postgres you can use real schemas (`catalog.products`, `payments.charges`) or a table-name prefix convention (`cat_products`, `pay_charges`) — schemas are cleaner because you can grant per-schema permissions, but the prefix works in databases without schema support. The non-negotiable rules:

1. **A module owns its tables. No other module reads or writes them.**
2. **No foreign keys across module boundaries.** A foreign key from `ord_orders.payment_id` to `pay_charges.id` is a hard physical coupling that makes extraction painful, because you cannot move `pay_charges` to another database while a foreign key constraint binds it. Instead, orders stores the payment id as a plain value (a logical reference) and trusts the payments module to resolve it through the port.
3. **No cross-module joins.** If orders needs catalog data to render an order line, it either calls the catalog port and joins in application code, or it stores a *denormalized copy* of the data it needs at the time of the order (the product name and price as they were when the order was placed — which, for orders, is actually the *correct* behavior, because an order should reflect the price at purchase time, not today's price).

Each module manages its own migrations. The migration directory is partitioned by module, and each module's migrations only ever touch its own schema:

```sql
-- payments/db/migrations/V1__create_payments_schema.sql
-- Payments owns the pay_ schema. No other module's migrations touch it.
CREATE SCHEMA IF NOT EXISTS payments;

CREATE TABLE payments.charges (
    id              UUID PRIMARY KEY,
    order_id        UUID NOT NULL,        -- logical ref to orders; NO foreign key
    amount_cents    BIGINT NOT NULL,
    currency        CHAR(3) NOT NULL,
    status          TEXT NOT NULL,        -- succeeded | declined | refunded
    idempotency_key TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The idempotency key is unique so a retried charge is a no-op insert.
CREATE UNIQUE INDEX uq_charges_idempotency
    ON payments.charges (idempotency_key);

CREATE INDEX ix_charges_order ON payments.charges (order_id);
```

```sql
-- orders/db/migrations/V1__create_orders_schema.sql
-- Orders stores a PLAIN payment_id. No FK to payments.charges.
CREATE SCHEMA IF NOT EXISTS orders;

CREATE TABLE orders.orders (
    id           UUID PRIMARY KEY,
    customer_id  UUID NOT NULL,
    payment_id   UUID,                    -- logical ref; resolved via the port
    status       TEXT NOT NULL,
    total_cents  BIGINT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orders.order_lines (
    order_id     UUID NOT NULL REFERENCES orders.orders(id),  -- FK WITHIN orders is fine
    product_id   UUID NOT NULL,           -- logical ref to catalog
    product_name TEXT NOT NULL,           -- DENORMALIZED snapshot at purchase time
    unit_cents   BIGINT NOT NULL,         -- DENORMALIZED price at purchase time
    qty          INT NOT NULL
);
```

Notice the two denormalized columns in `order_lines`: `product_name` and `unit_cents`. A junior engineer's instinct is to normalize these away — "why store the product name on the order line when we can join to catalog?" The answer is twofold. First, *correctness*: an order line should record what the customer actually bought at the price they actually paid; if the product is later renamed or repriced, the order must not change. Second, *extractability*: the moment you join `ord_order_lines` to `cat_products`, you have welded orders and catalog together at the database level, and neither can be extracted without untangling that join. The denormalized copy is both more correct *and* keeps the boundary clean. This is a recurring theme — the discipline that makes extraction cheap is very often *also* better design on its own terms. Foreign keys within a module are fine (`order_lines` to `orders`); foreign keys across modules are the thing to refuse.

#### Worked example: the join that quietly welded two modules

A real failure mode I have watched: a team built ShopFast with clean code-level module boundaries — ArchUnit tests, separate packages, the works — and felt great about it. But in the reporting feature, an analyst-turned-engineer wrote a dashboard query that joined `orders.orders`, `payments.charges`, and `catalog.products` in a single SQL statement, because it was the obvious way to build the "revenue by product category" report. It passed code review (it was "just a query"), it was fast, it shipped. Eighteen months later, when the team tried to extract payments into its own service, they discovered *forty-three* such queries scattered across reporting, admin tooling, and a few "quick" feature endpoints, each one a cross-schema join binding payments' tables to the rest of the system.

The extraction estimate, which should have been two weeks, became a *two-month* program, almost all of it spent finding and rewriting those forty-three joins — replacing each one with either a port call plus an application-side join, or a denormalized read model, or a query against a CQRS-style projection (the read-model pattern from [Event Sourcing and CQRS with an Event Log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log)). The code boundaries had held perfectly; the *data* boundary had been quietly violated forty-three times, and the data boundary is the one that determines extractability. The lesson the team took away, and the one I would press on you: **the cross-module join is the single most dangerous thing in a modular monolith, and it is invisible to your import linter.** You need a separate discipline for it — schema-scoped database users so the join *cannot be written*, or a query review that specifically flags cross-schema joins, or both. Do not trust that clean code means clean data.

## 7. The trade-off matrix: three architectures, named costs

It is time to be explicit and side-by-side about the choice, because the whole argument for the modular monolith is a trade-off argument: it is not the simplest thing (the big ball of mud is simpler to *start*), and it is not the most scalable thing (microservices win on independent deploy and scale), but it dominates on the dimensions that matter for most teams most of the time.

![A decision matrix comparing big ball of mud, modular monolith, and microservices across boundary-change cost, deploy independence, operational complexity, refactor safety, and the cost of a wrong boundary](/imgs/blogs/monolith-first-and-the-modular-monolith-4.webp)

Read the figure column by column. The **big ball of mud** is operationally simple (one deploy, low ops cost) but catastrophic on everything structural: boundaries are so expensive to change they effectively never get changed, refactoring is unsafe because nothing is isolated, and a wrong boundary is catastrophic because by the time you notice, everything depends on everything. The **modular monolith** keeps the low operational complexity of a single deploy, *adds* the refactor safety and cheap boundary-changes of strong internal structure, and the only thing it gives up versus microservices is deploy independence — it is still one deploy unit, so you cannot ship the payments module without shipping the whole app. **Microservices** buy that deploy independence and per-service scaling, but pay for it with high operational complexity (every service is a full operational entity) and a high cost to move a boundary once it has hardened into a service.

Here is the same comparison as a table, with the costs named explicitly, because the kit's rule is right: never recommend anything without naming what it costs.

| Dimension | Big ball of mud | Modular monolith | Microservices |
| --- | --- | --- | --- |
| **Cost to change a boundary** | Very high — touches everything | Low — refactor, recompile, ship in hours | High — data migration + API versioning + cross-team coordination, weeks |
| **Deploy independence** | None | None — one deploy unit | Per team / per service |
| **Operational complexity** | Low (one process) | Low (one process) | High (N services, mesh, tracing, on-call) |
| **Refactor safety** | Unsafe — no isolation, hidden coupling | Safe — compiler + ArchUnit catch violations | Hard — refactoring across a network boundary is a migration |
| **Cost of a wrong boundary** | Catastrophic | Hours to days | Weeks to months |
| **Latency of internal calls** | ~1µs (in-process) | ~1µs (in-process) | ~1–10ms per hop (network) |
| **Failure model of internal calls** | Cannot partially fail | Cannot partially fail | Timeout, retry, partial failure, cascade |
| **Best when** | Never (it is an accident, not a choice) | Almost always to start; often forever | Many teams, divergent scaling, blast-radius isolation needed |
| **Team scale that fits** | 1 confused team | 1–3 teams comfortably; more with discipline | Many teams, one or two services each |

The big ball of mud is in the table only because it is the *default* you fall into if you do not actively maintain boundaries — it is what a monolith becomes through neglect. Nobody chooses it; you arrive at it. The real choice is between the modular monolith and microservices, and the matrix makes the senior recommendation visible: **start modular-monolith, because it dominates the big ball of mud on every structural axis and dominates microservices on operational cost, and only the deploy-independence and per-service-scaling columns favor microservices — and those columns only start to matter once you have multiple teams or divergent scaling needs.**

The decision framework for the broader microservices-versus-monolith question, with the full cost accounting, lives in [What Are Microservices and When Not to Use Them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them); this matrix is the narrower three-way comparison that includes the modular monolith as a first-class option, which most "monolith versus microservices" framings wrongly omit. The whole point of this post is that the middle column exists and is, for most teams, the right answer.

## 8. The signals that you have outgrown the monolith

Most companies never outgrow a well-built modular monolith. I want to say that clearly before listing the signals, because the signals are easy to misread as "we should go microservices" when they actually mean "we should fix our CI" or "we have an org problem, not an architecture problem." But for the teams that genuinely do outgrow it, the signals are concrete and measurable, and they cluster in two areas: the **org axis** and the **infrastructure axis**. Code size, notably, is *not* on the list — a 500,000-line modular monolith with clean boundaries is fine; a 50,000-line big ball of mud is not.

![A timeline of signals you have outgrown the monolith from deploy contention at two teams through growing CI time and blast radius to divergent scaling at five-plus teams and the decision to extract the first service](/imgs/blogs/monolith-first-and-the-modular-monolith-7.webp)

The timeline shows them roughly in the order they appear:

**Deploy contention between teams.** This is usually the first real signal. When you have one team, one deploy unit is great — you ship when you are ready. When you have three or four teams committing to the same monolith, they start to collide. Someone's feature is mid-review when another team needs to ship a hotfix. The merge queue backs up. A bad change from team A blocks team B's release. You build elaborate release-train schedules to coordinate. The pain here is *coupled deployment* — independent teams forced into a shared release cadence — and it is the single best reason to extract a service, because deploy independence is precisely what extraction buys.

**Build and test time crossing a threshold.** When CI grows from 8 minutes to 35 minutes, every engineer's day gets slower, and the feedback loop that makes a monolith pleasant to work in starts to break. *But be careful here*: most of the time, slow CI is a CI problem, not an architecture problem. Before you split the codebase to make the build faster, try test parallelization, build caching (Gradle/Bazel remote cache), running only the tests affected by a change, and splitting the test suite by module so a change to payments only runs payments' tests. These often take a 35-minute build back under 10 minutes for a fraction of the cost of extracting a service. Splitting the deploy unit to speed up the build is using a sledgehammer to fix a loose screw — and you inherit the network's failure modes as a side effect.

**Fault blast radius.** In a monolith, a memory leak in the reporting module can take down checkout, because they share a process. If one module's bad deploy or runaway query can bring down the whole system, and you have a part of the system (say, payments) where that blast radius is unacceptable, that is a real argument for isolating it into its own process. This is one of the strongest *non-org* reasons to extract: not "we want independent deploys" but "we cannot tolerate this module sharing fate with the rest."

**Divergent scaling needs.** When one module needs fundamentally different resources than the rest — the image-processing module wants GPUs, the search module wants 64 GB of RAM for its index, the rest of the app is happy on a small CPU box — running them all in one deploy unit means you scale the whole thing to the needs of the hungriest module, which is wasteful. If catalog search needs to scale to 10× the traffic of checkout, and they are in one process, you pay for checkout-scale infrastructure ten times over. Extracting the divergent module lets you scale it independently. (Note: this often shows up as a *cost* signal, which is measurable — see the worked example below.)

**Team count crossing the two-pizza line.** Amazon's "two-pizza team" rule — a team small enough to be fed by two pizzas, roughly 6–10 people — is really a statement about communication overhead and ownership. When your org grows past two or three two-pizza teams all working in one codebase, the coordination cost of the shared monolith starts to exceed the operational cost of separate services. This is Conway's Law in action: your architecture wants to mirror your communication structure, and a single monolith owned by eight teams is fighting that gradient. (The full org-and-architecture argument is its own post, [Conway's Law and Team Topologies for Microservices](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices).)

The crucial diagnostic question, which I will press hard in the stress-test section, is: **is the pain coming from the architecture, or from the org?** Deploy contention and team-count pain are *org* signals — they are about how humans coordinate, and extracting a service is a way of giving a team autonomy. CI time and blast radius are partly architecture signals but often have cheaper fixes. The mistake is to read an org problem as an architecture problem and "solve" it with a distributed system that makes the org problem worse by adding distributed-systems complexity on top.

#### Worked example: when divergent scaling makes extraction pay for itself

Numbers make the divergent-scaling signal concrete. Suppose ShopFast's catalog search gets 8,000 requests per second at peak (browsing is the bulk of traffic), while checkout gets 400 requests per second (most browsers do not buy). They live in one modular monolith, deployed as 20 identical pods, each pod sized to handle the combined load. Search is CPU-and-memory-hungry (it holds an in-memory index); checkout is light. Because they share a process, every pod carries the full search index — say 6 GB of RAM each — even though only the search code paths need it. Twenty pods × 6 GB = 120 GB of RAM provisioned, much of it to satisfy search's footprint replicated across pods that mostly serve checkout and order traffic.

Now extract catalog-search into its own service. Search runs on, say, 16 memory-heavy pods (6 GB each, 96 GB total) scaled to its 8,000 RPS. The rest of the monolith — orders, payments, inventory, and catalog *writes* — runs on 6 small pods (1.5 GB each, 9 GB total) scaled to its 400-RPS checkout load plus background work. Total RAM drops from **120 GB to roughly 105 GB**, and more importantly the *expensive* memory-heavy capacity (96 GB) now scales independently with browse traffic while the cheap capacity (9 GB) scales with checkout. On a cloud bill where memory-optimized instances cost meaningfully more than general-purpose ones, separating the scaling axes can cut the relevant slice of infrastructure cost by **30–50%**, and — this is the part that matters operationally — a traffic spike in browsing no longer forces you to over-provision the checkout path, and a checkout deploy no longer has to drag 6 GB of search index through a rolling restart.

That is a *real* microservices win, measurable in dollars and in deploy time, and it is the kind of signal that justifies extraction. Contrast it with "we have eleven services because the architecture diagram looked impressive" — that has no measurable win at all, only the operational tax. When you extract, extract *against a measurable signal*: a dollar figure, a p99 you cannot hit, a blast radius you cannot tolerate, a team you cannot unblock. If you cannot name the number, you are not ready to extract.

## 9. Extracting the first service cleanly

When the signals are real and the number is named, it is time to extract — and the whole point of having built a *modular* monolith is that this is now a measured, low-drama operation rather than a rewrite. The key insight is that **extraction is not "rewrite the module as a service." It is "move the module across a network boundary, changing as little code as possible."** If you built the module well, almost nothing inside it changes; what changes is the transport between the module and its callers.

![A before and after comparison showing the payments module with an in-process port and shared schema becoming a payments service with the same port over gRPC and its schema moved to its own database](/imgs/blogs/monolith-first-and-the-modular-monolith-8.webp)

The figure shows the payments extraction. *Before*: payments is a module with a `PaymentGateway` port satisfied in-process, its `pay_` schema in the shared database, deployed with the app. *After*: the *same* `PaymentGateway` port is now satisfied by a gRPC client; the `pay_` schema has moved to its own database; and payments deploys alone, which has the bonus of letting you isolate it for PCI compliance (a real reason payments is so often the first thing extracted — the regulatory blast-radius argument).

**Which module do you extract first?** For ShopFast, I would extract **payments**, and the reasoning generalizes into a checklist for picking the first extraction:

1. **It has the clearest, most stable boundary.** Payments has a narrow, well-understood contract (charge, refund) that rarely changes shape. You want your *first* extraction to be the lowest-risk one, so pick the module whose interface you are most confident is right. Do not learn distributed systems on your messiest, most-coupled module.
2. **It has a compelling independent reason to be separate.** Payments touches PCI-scoped data; isolating it shrinks your compliance surface (only the payments service is in PCI scope, not the whole monolith). It also has a stronger uptime/blast-radius requirement than reporting. There is a *real* benefit, not just "the diagram looks better."
3. **It has the fewest inbound dependencies.** Look at the dependency graph: payments is mostly *called by* orders and does not call back into the rest of the system. A leaf in the dependency graph (or close to it) is far easier to extract than a hub. Extracting orders — the orchestrator that calls everyone — would be the *hardest* first move, because it has the most edges to convert to network calls.
4. **It already owns its data cleanly.** Because we forbade cross-schema joins and cross-module foreign keys, `pay_` can physically move to its own database without untangling joins. If you skipped that discipline, *this* is where it bites: you cannot extract a module whose tables are joined to everyone else's.

The mechanical steps, assuming a well-built module:

```protobuf
// payments/proto/payments.proto
// The gRPC contract mirrors the PaymentGateway port one-to-one. The
// shape was designed for this from day one, so this is transcription,
// not redesign.
syntax = "proto3";
package shopfast.payments.v1;

service Payments {
  rpc Charge (ChargeRequest) returns (ChargeResult);
  rpc Refund (RefundRequest) returns (RefundResult);
}

message ChargeRequest {
  string order_id           = 1;
  int64  amount_cents       = 2;
  string currency           = 3;
  string payment_method_token = 4;
  string idempotency_key    = 5;   // already part of the interface
}

message ChargeResult {
  enum Status { SUCCEEDED = 0; DECLINED = 1; GATEWAY_UNAVAILABLE = 2; }
  Status status   = 1;
  string charge_id = 2;
  string reason    = 3;
  bool   retryable = 4;
}
```

Then, in the monolith, you swap the wiring. The orders module still depends on the `PaymentGateway` interface — *it does not change at all*. You simply register a different implementation: instead of the in-process `PaymentGatewayImpl`, you register a `PaymentGatewayGrpcClient` that implements the same interface by calling the new service.

```java
package com.shopfast.payments.client;

import com.shopfast.payments.api.PaymentGateway;
import com.shopfast.payments.api.ChargeRequest;
import com.shopfast.payments.api.ChargeResult;

// Implements the SAME public port the orders module already depends on.
// Orders code does not change; only the wiring (which impl is injected) does.
public class PaymentGatewayGrpcClient implements PaymentGateway {

    private final PaymentsGrpc.PaymentsBlockingStub stub;

    public PaymentGatewayGrpcClient(PaymentsGrpc.PaymentsBlockingStub stub) {
        this.stub = stub.withDeadlineAfter(800, java.util.concurrent.TimeUnit.MILLISECONDS);
    }

    @Override
    public ChargeResult charge(ChargeRequest req) {
        var grpcReq = toProto(req);                 // map DTO -> protobuf
        var resp = stub.charge(grpcReq);            // network call: can now time out
        return fromProto(resp);                     // map protobuf -> DTO
    }
    // ... refund, mapping helpers ...
}
```

Because the interface was designed like a network contract — flat DTOs, idempotency key, result types including a "gateway unavailable, retryable" case — the gRPC client is a thin mapping layer, not new logic. The caller's failure handling (the `switch` in `CheckoutHandler`) already covers the `GatewayUnavailable` case it now actually can hit. You add a timeout (the `withDeadlineAfter` above — 800 ms), and you will want to wrap it in the resilience layers (retry, circuit breaker, bulkhead) covered in [Resilience Patterns: Timeouts, Retries, Circuit Breakers, Bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads), because now the call really can fail in ways an in-process call never could. But the *business logic* is untouched.

The data move is the riskier half, and it follows the standard zero-downtime migration playbook: stand up the new payments database, set up change-data-capture or dual-writes from the old `pay_` schema to the new one so they stay in sync, backfill historical rows, verify the two are consistent, cut reads over, then cut writes over and stop the dual-write. The mechanics of doing this safely — the outbox pattern, CDC, dual-writes — are in [Change Data Capture and the Outbox Pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and the broader incremental-cutover strategy (route a growing slice of traffic to the new service behind a facade) is the strangler-fig pattern detailed in [Strangler Fig: Migrating a Monolith to Microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices). The point for *this* post is that the data move is *possible at all* only because you forbade cross-schema joins and cross-module foreign keys from day one. The discipline you maintained when it felt like overkill is exactly what makes today's extraction a two-week project instead of a two-month one.

![A graph of ShopFast after the first extraction showing the monolith still holding catalog, orders, and inventory modules in-process while payments runs as a separate gRPC service talking to an external payment provider](/imgs/blogs/monolith-first-and-the-modular-monolith-9.webp)

And here is the result, and the part most "go microservices" narratives miss: **you do not extract everything.** After the first extraction, ShopFast is one payments service plus a monolith still containing catalog, orders, and inventory as in-process modules. That is a perfectly good — often *ideal* — end state. You extracted the one module with a real reason to be separate, you left the rest in the cheap, fast, simple monolith, and you now have a *hybrid* that gets the benefit of services exactly where the benefit exists and pays the network tax nowhere else. If a second module later earns its extraction, you extract it then, the same way. You are not on a one-way road to a hundred services; you are extracting modules one at a time, each against a named signal, and stopping whenever the next extraction stops being worth it. For many teams, that stopping point is one or two services plus a healthy monolith, forever.

## 10. Case studies

The strongest evidence that the modular monolith is a serious architecture and not a beginner's compromise comes from companies operating at enormous scale who deliberately chose it — or who chose microservices, regretted it, and came back. I will keep the specifics order-of-magnitude and stick to what these teams have said publicly.

**Shopify's modular monolith.** Shopify runs one of the largest Ruby on Rails codebases in the world — millions of lines, thousands of engineers — and it is, deliberately, a *monolith*. Rather than break it into microservices, Shopify invested in **componentization**: restructuring the monolith internally into bounded components with enforced boundaries, building tooling (notably an open-source tool called Packwerk) that enforces which components may depend on which, and flagging boundary violations at build time — exactly the architecture-fitness-function discipline this post argues for, applied at a scale most teams will never reach. Shopify's public engineering writing makes the argument plainly: the problems people attribute to monoliths are really problems of *unclear boundaries*, and you can fix those boundaries inside the monolith without paying the distributed-systems tax. Their bet, at enormous scale, is the central bet of this article: a modular monolith with enforced boundaries beats a premature distributed system. The lesson: the modular monolith scales to thousands of engineers if you invest in boundary tooling — code size is not the thing that forces you out.

**Segment's reversal back to a monolith.** Segment is the canonical "we went microservices and it was a mistake" story, told in their own widely read engineering post. They had decomposed their data-routing pipeline into a large and growing fleet of microservices — one per destination integration, eventually over a hundred — and discovered that the operational cost had exploded: every service had its own dependencies, its own deploy, its own queue, and shared libraries had to be updated across all of them, so a single change rippled into a punishing amount of coordinated work. The microservices had not given them isolation; they had given them a hundred copies of the same operational burden. Segment *consolidated back into a monolith* (they called it "Centrifuge" in some of the rework) and reported large gains in operability and developer velocity. The lesson is not "microservices bad" — it is that splitting by a dimension that does not actually need independence (one service per destination, when destinations shared most of their logic and lifecycle) multiplies cost without buying isolation. They split on the wrong axis, and the fix was to come back. This is the cost of a wrong boundary, paid at the service level, exactly as the matrix predicts.

**GitHub and Basecamp's "majestic monolith."** Basecamp (and its creator, who coined the phrase "the majestic monolith") and GitHub both run large, successful products on substantially monolithic Rails codebases. The argument from this camp is cultural as much as technical: a single, well-organized codebase that a developer can hold in their head and run entirely on a laptop is a productivity superpower, and most teams reach for microservices to solve organizational problems that microservices do not actually solve — and that a disciplined monolith plus good internal structure solves more cheaply. GitHub has, over many years, extracted *some* pieces into services where there was a real reason (search, certain background-processing workloads), while keeping the core a monolith — which is precisely the *hybrid, extract-against-a-signal* end state this post recommends, not a wholesale migration. The lesson: extraction is selective and signal-driven, not a phase transition you make all at once.

**Amazon's two-pizza teams (the org side of the same coin).** Amazon is famous for service-oriented architecture, but the underrated part of the story is *why*: the architecture followed the org. Amazon organized into small, autonomous "two-pizza teams" that each own their service end to end, and the service boundaries exist to give those teams independence. The lesson for the monolith-first practitioner is the inverse: if you do *not* have many autonomous teams that need to deploy independently, you do not have Amazon's problem, and copying Amazon's architecture without Amazon's org gives you the costs without the benefits. The architecture earns its keep when it maps to a real team topology — which is the Conway's-Law argument, and the single best predictor of whether microservices will help you.

The through-line across all four: **the architecture that wins is the one that matches your actual constraints — your team count, your scaling axes, your blast-radius needs — not the one that matches the most impressive blog post.** Shopify and Basecamp stayed monolith because their constraints did not force a split. Segment went distributed, found the split bought no isolation, and reversed. Amazon split because its org genuinely needed autonomous teams. None of them chose by fashion; all of them chose by constraint.

## 11. Stress test: is the pain the architecture, or the org?

Now let me do what a senior does before recommending anything: try to break the recommendation. The thesis is "build a modular monolith, extract selectively against signals." Where does that fall apart, and is the failure really the architecture's fault?

**Stress: "Our monolith is genuinely painful. The architecture is the problem."** Most of the time, when a team says this, I find one of two things. Either the monolith is a *big ball of mud* — no enforced boundaries, cross-module joins everywhere, a change to billing breaks checkout — in which case the problem is the missing modularity, not the single deploy unit, and the fix is to *introduce* boundaries (Packwerk/ArchUnit, schema separation) *in place*, which is far cheaper than extracting services and gets you most of the benefit. Or the pain is *organizational*: too many teams contending for one deploy, ownership that is fuzzy so everything is everyone's and therefore no one's problem, a release process that is slow for reasons unrelated to the architecture. Extracting services *can* relieve org pain (by giving a team an autonomous deploy), but it does so at the cost of distributed-systems complexity, and you should first ask whether you can relieve the org pain more cheaply — clearer module ownership, a faster release process, team boundaries that match module boundaries inside the monolith.

**Stress: "What breaks at 10× traffic?"** A modular monolith scales *horizontally* perfectly well for stateless request handling — you run more identical copies behind a load balancer, exactly as you would with a service. The thing that does *not* scale is anything that must be a single instance (a background scheduler, an in-memory cache that needs to be coherent) and, critically, the *shared database*, which all copies hit. At 10× traffic the monolith's failure point is almost always the database, not the application tier — and *that* is true of microservices too unless each service has its own database. So 10× traffic is rarely an argument for extraction per se; it is an argument for the same database scaling work ([Database Partitioning and Sharding](/blog/software-development/database/database-partitioning-and-sharding), read replicas) you would do anyway. The exception is the divergent-scaling case from section 8, where one module's resource profile is so different that co-locating it is wasteful — *that* is a real extraction signal at scale.

**Stress: "What happens when a dependency is down?"** This is the question that flips entirely between architectures, and it is the modular monolith's quiet superpower. In a monolith, a "dependency" between modules is an in-process call: it cannot be "down." The catalog module is as available as the process it runs in. There is no network partition between orders and payments because there is no network. This is *why* the in-process call is so much simpler — you delete an entire category of failure (timeouts, partial failures, retries, circuit breakers, cascading failures) by not having a network. The moment you extract payments into a service, payments *can* be down relative to orders, and now you need timeouts, retries, circuit breakers, and graceful degradation ([Handling Partial Failures and Graceful Degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation)). The honest framing: extraction does not just add a network hop's latency; it adds an entire failure mode that did not exist before, and you must build the resilience machinery to survive it. That cost is invisible on the architecture diagram and very visible at 3am. It is a reason to extract *only when the benefit clearly exceeds it*, and to extract the smallest number of modules that gets you the benefit.

**Stress: "A service is deployed mid-request."** In a monolith, a deploy is atomic from a request's perspective — a request runs entirely against one version of the code (you drain old instances). Across services, request A might hit the new version of payments and the old version of orders simultaneously, so the two versions must be compatible — which is why microservices force you into rigorous API versioning and contract testing ([API Versioning and Consumer-Driven Contract Testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing)). The monolith sidesteps this entirely: all modules deploy together, always at the same version, so there is no version-skew problem between modules. This is another hidden cost of extraction and another reason the single deploy unit is genuinely simpler, not just simpler-looking.

The stress test does not break the recommendation; it sharpens it. The modular monolith is not painful *because* it is a monolith — it is painful when it is a big ball of mud (fix the boundaries in place) or when the org has outgrown a shared deploy (extract selectively, against a named signal, accepting the resilience and versioning costs that come with it). The architecture is rarely the problem. The boundaries — or the org — usually are. A senior diagnoses *which* before reaching for a distributed system, because the distributed system is the most expensive possible answer and the wrong answer to an org problem.

## 12. When to reach for this (and when not to)

Here is the decisive recommendation, stated plainly.

**Reach for the modular monolith — and start here essentially always — when:** you are building something new and do not yet know the domain's true boundaries (which is almost always); you have one to three teams; you want the option of future extraction without paying for it now; you value being able to run the whole system on a laptop, refactor across boundaries safely, and use real database transactions where they help. This is the default. If someone proposes microservices for a greenfield project with one team, the burden of proof is on them to name the specific, present benefit that justifies the operational tax — and "we will need it eventually" is not a present benefit; it is a reason to keep the boundaries clean so you can extract when "eventually" arrives.

**Stay in the modular monolith — possibly forever — when:** the signals from section 8 are not firing. No deploy contention you cannot manage with module ownership; CI time you can keep reasonable with caching and test-splitting; blast radius you can tolerate; scaling needs that are uniform enough to handle by running more identical copies; a team count under the two-or-three-two-pizza-teams threshold. Most companies live here their entire lives and are right to. The modular monolith is not a phase; for the majority it is the destination.

**Extract a service — selectively, one at a time — when:** a *specific* module fires a *measurable* signal. A blast-radius requirement you cannot meet in-process (payments, with its PCI scope and uptime needs). A divergent scaling profile with a dollar figure attached (the search-index example). An org boundary where a team genuinely needs to deploy independently and you have tried the cheaper fixes. Extract that module, leave the rest, and stop. Each extraction should be defensible with a number, not a vibe.

**Go full microservices from the start — rarely — when:** you have built this exact system before and *know* the boundaries with high confidence; you already have many autonomous teams on day one (e.g., a large company spinning up a new platform with dedicated teams); or you have a hard regulatory or multi-tenancy isolation requirement that mandates separate deployables from the outset. These are real but uncommon. If you are not in one of them, monolith-first is the lower-variance bet, and the modular monolith makes the bet nearly free to reverse.

The asymmetry is the whole point: **starting modular-monolith and extracting later is cheap and reversible; starting microservices and consolidating later is expensive and rare** (Segment did it, and it was a project). When two paths have different reversal costs, you take the one that is cheap to undo. That is just the cost-of-change curve applied to your single biggest architectural decision.

## 13. Key takeaways

- **Default to monolith-first.** You do not know the boundaries on day one, and microservices bake a guessed boundary into expensive concrete. Defer the hard, irreversible decision until you have evidence. (Junior: start simple. Senior: defer the costly decision until the cost of deferring exceeds the cost of being wrong.)
- **A monolith's reputation is about missing structure, not the deploy unit.** "One deploy unit" and "internally muddy" are independent choices. The modular monolith keeps the first and refuses the second.
- **Modules are future services. Build them that way.** Separate schema per module, narrow public ports, no cross-module joins, no cross-module foreign keys. Everything that makes a modular monolith clean is exactly what makes extraction cheap.
- **Enforce boundaries with the build, not with good intentions.** An ArchUnit or import-linter test that fails the build on an illegal import is the only discipline that survives a year of deadline pressure. A design doc is not enforcement.
- **The data boundary is the one that determines extractability, and it is invisible to your import linter.** The cross-schema join is the single most dangerous thing in a modular monolith. Scope database access per module so the bad join cannot be written.
- **Design ports like RPCs before they are RPCs:** flat DTOs, idempotency keys, result types that include "unavailable." Then extraction is a transport swap, not a rewrite, and your orchestration is already failure-aware.
- **Extract against a measurable signal, never a vibe.** Deploy contention, a dollar figure from divergent scaling, an intolerable blast radius, an org boundary you cannot otherwise unblock. If you cannot name the number, you are not ready.
- **Diagnose org-versus-architecture before reaching for a distributed system.** Most "the monolith is painful" complaints are a big ball of mud (fix boundaries in place) or an org problem (clarify ownership, speed up CI). A distributed system is the most expensive possible answer and the wrong answer to an org problem.
- **You do not have to extract everything.** A monolith plus one or two extracted services is often the ideal permanent end state. Stop extracting when the next extraction stops being worth its operational tax.
- **The senior move is matching architecture to constraints, not to fashion.** Shopify and Basecamp stayed monolith; Segment reversed; Amazon split — each by constraint, none by trend.

## 14. Further reading

- Martin Fowler, ["MonolithFirst"](https://martinfowler.com/bliki/MonolithFirst.html) and ["MicroservicePremium"](https://martinfowler.com/bliki/MicroservicePremium.html) — the foundational arguments for starting with a monolith and the threshold past which microservices earn their cost.
- Sam Newman, *Building Microservices* (2nd ed.) and *Monolith to Microservices* — the latter is the definitive practitioner's guide to incremental decomposition and is essentially a book-length version of "extract selectively, against a signal."
- Chris Richardson, *Microservices Patterns* — the pattern catalogue (database-per-service, saga, API composition) whose monolith-side seeds you plant inside the modular monolith.
- Shopify Engineering, "Deconstructing the Monolith" and the Packwerk project — a real, enormous-scale modular monolith with build-enforced component boundaries.
- Segment Engineering, "Goodbye Microservices: From 100s of Problem Children to 1 Superstar" — the canonical microservices-to-monolith reversal, and a clinic in splitting on the wrong axis.
- This series and its mechanism deep-dives: [What Are Microservices and When Not to Use Them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them), [Service Boundaries with Domain-Driven Design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design), [Strangler Fig: Migrating a Monolith to Microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices), [Database Per Service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), and the architecture-fitness-function discipline in [Evolutionary Architecture: Designing for Change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change).

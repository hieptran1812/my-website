---
title: "The Saga Pattern in Practice: Building a Transaction That Spans Services"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How to actually implement a business transaction across multiple services when a distributed ACID transaction is off the table — designing the steps and compensations, choosing choreography or orchestration, defeating the isolation anomalies, and shipping a hand-rolled orchestrator and a Temporal-style workflow you can copy."
tags:
  [
    "microservices",
    "saga",
    "distributed-transactions",
    "orchestration",
    "choreography",
    "distributed-systems",
    "software-architecture",
    "backend",
    "idempotency",
    "data-consistency",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/the-saga-pattern-in-practice-1.webp"
---

The ShopFast `placeOrder` endpoint used to be twelve lines of code. Reserve the stock, charge the card, create the shipment, return `200 OK`. It lived in one service, talked to one database, and the whole thing ran inside a single `BEGIN ... COMMIT`. If the card declined, the transaction rolled back and the reserved stock simply vanished from the uncommitted write — no stock was ever held, no money ever moved, nothing to clean up. The database did all the hard work, and the engineer who wrote it never had to think about partial failure because partial failure was impossible. Either all four things happened, or none of them did.

Then ShopFast split into services. Inventory got its own database, payment got its own (actually it got an external payment processor), and shipping got a third. The instant that happened, the twelve-line endpoint became a lie. There is no `BEGIN` that spans three databases and a third-party API. There is no `COMMIT` that atomically flips state in inventory, payment, and shipping at once. The moment you adopt [database-per-service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) — and you should, it is the rule that defines microservices — you give up the one tool that made `placeOrder` correct. So one afternoon a card declined *after* the inventory had already committed its reservation, and a unit of stock sat held forever for an order that never existed. Multiply by a few thousand orders a day and the warehouse showed zero available units of a product that had two hundred sitting on the shelf.

This post is about the pattern that fills the hole left by the missing transaction: the **saga**. A saga is a sequence of local transactions, one per service, where each step that can be undone has a paired **compensating transaction** that semantically reverses it. There is no global rollback; there is a carefully ordered chain forward and, when something fails, a carefully ordered chain backward. The companion [saga-pattern deep-dive in the database series](/blog/software-development/database/saga-pattern-distributed-transactions) derives the mechanism — why a multi-service operation cannot be one transaction, the formal step taxonomy, the isolation theory. This post is the practitioner's how-to: you will leave able to *build* the ShopFast place-order saga two different ways (a hand-rolled orchestrator state machine and a Temporal-style durable workflow), classify and order the steps so compensation is even possible, defeat the anomalies that the lack of isolation lets in, make every step idempotent, and instrument the whole thing so that when a saga gets stuck — and one will — you can find it before the customer does.

![A flow graph showing the ShopFast happy path from reserve inventory to charge payment to create shipment, with a branch where a declined payment runs the release inventory compensation and ends in a cancelled order](/imgs/blogs/the-saga-pattern-in-practice-1.webp)

## The problem, stated precisely: no 2PC across services

Let us be exact about what we lost, because the whole pattern is a workaround for one specific missing capability. A classic relational database gives you ACID: a transaction is **A**tomic (all-or-nothing), **C**onsistent (it moves the database from one valid state to another), **I**solated (concurrent transactions do not see each other's half-finished work), and **D**urable (once committed, it survives a crash). Atomicity and isolation are the two we care about here, and they are exactly the two we cannot get across services.

You might think: the database can do two-phase commit (2PC) across nodes, so why not across services? 2PC — the XA protocol in the SQL world — has a coordinator ask every participant to *prepare* (promise it can commit), and only if all say yes does it tell them all to *commit*. It does, in principle, give you atomicity across multiple databases. The reasons it is the wrong tool for microservices are worth stating plainly, because "just use 2PC" is a tempting wrong answer a senior must be able to shoot down in a design review:

- **It couples availability.** During the prepare phase, every participant holds locks and waits. If the coordinator crashes after some participants have prepared, those participants are *stuck* holding locks until the coordinator recovers — this is the infamous "in-doubt" or "blocking" state. One slow or dead participant freezes the others. Your payment processor (a third-party HTTP API) cannot participate in your XA transaction at all, so the whole premise collapses the moment a step is a non-transactional external call. We unpacked this availability-versus-consistency tension in [the CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc); 2PC sits firmly on the consistency-over-availability side, which is the wrong default for a customer-facing checkout.
- **It does not scale.** Holding cross-service locks for the duration of a multi-hop business operation — which might involve a slow external charge taking 200ms or a human approval taking minutes — destroys throughput. Locks are the enemy of concurrency.
- **It is not how the modern stack is built.** Most message brokers, NoSQL stores, and SaaS APIs simply do not speak XA. You would be designing your architecture around a protocol most of your components cannot join.

So we give up global atomicity and global isolation. The saga gives back a *weaker but usable* atomicity — "every step either runs or is compensated for, so the system converges to all-done or all-undone" — and gives back *nothing automatic* on isolation, which is why half of this post is about clawing back just enough isolation by hand. That trade is the entire pattern in one sentence.

## What a saga actually is: local transactions plus compensations

Here is the core idea, and it is genuinely simple once the framing clicks. Take the operation you wish were one transaction — reserve inventory, charge payment, create shipment — and split it into **a sequence of local transactions**, one per service. Each local transaction is fully ACID *within its own service and database*; ShopFast's inventory service can use a normal `BEGIN ... COMMIT` to atomically decrement available stock and write a reservation row. The saga is the glue that runs these local transactions in order.

The catch is what happens when step *N* fails after steps 1 through *N−1* have already committed. You cannot roll them back — they are committed, durable, visible. Instead, for every step that *can* be undone, you write a **compensating transaction**: another local transaction that semantically reverses the effect. `ReserveInventory` is paired with `ReleaseInventory`. `ChargePayment` is paired with `RefundPayment`. When step *N* fails, the saga runs the compensations for steps *N−1, N−2, … , 1* in reverse order, and the system ends up — eventually, not instantly — back in a coherent state.

The word **semantically** is doing enormous work in that paragraph, and it is the difference between a junior who has read about sagas and a senior who has operated one. A compensation is *not* a rollback. A rollback makes it as if the transaction never happened: no trace, no side effect, nothing. A compensation is a *new business fact* that counteracts a previous one. `RefundPayment` does not un-charge the card; it issues a refund, which is a separate ledger entry the customer will see on their statement. `ReleaseInventory` does not un-reserve; it adds a fresh "released N units" event. The customer may have received an email ("your order is confirmed!") that you now have to follow with a second email ("actually, we couldn't complete your order"). You cannot un-send the first email. The world has moved on, and your compensation is an apology, not a time machine.

That figure-1 graph above is the whole shape: a forward chain (reserve → charge → ship → confirmed) and a backward branch (charge fails → release inventory → cancelled). Notice the failure does not branch from every node identically — it branches from wherever you are, undoing only what actually committed. If `ChargePayment` fails, you compensate only `ReserveInventory`. If `CreateShipment` fails, you compensate `ChargePayment` (refund) *and* `ReserveInventory` (release). The compensation chain is always a suffix-reverse of the steps that succeeded.

### The three properties a saga gives you (and the one it doesn't)

- **Atomicity (weakened to eventual):** the saga converges to "all steps committed" or "all committed steps compensated." There is a window in between where it is partially done. That window is the price.
- **Consistency (per-service, plus eventual global):** each local transaction keeps its own database consistent; the global business invariant ("stock reserved iff order active") holds *eventually*, once the saga finishes. This is the eventual-consistency model we examine in [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and apply concretely in [data consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice).
- **Durability:** trivially, because each local transaction is durable in its own service.
- **Isolation: GONE.** This is the one you do not get. While a saga is mid-flight, other transactions *can* see its partial, uncommitted-from-a-business-view state — stock that is reserved but may be released, money that is charged but may be refunded. We will spend a whole section killing the anomalies this lets in.

## Designing a saga: pivot, compensatable, retriable

Before you write a single line of orchestrator code, you classify every step. This is the step a junior skips and a senior never does, because the *order* of the steps is not a free choice — it is determined by which steps can be undone. Richardson's *Microservices Patterns* gives the taxonomy, and it is the most useful three buckets in this whole area:

- **Compensatable transactions** — steps that can be semantically undone by a compensation. `ReserveInventory` (undo: release). `ApplyCoupon` (undo: un-apply). These can run early because if a later step fails, you can walk them back.
- **The pivot transaction** — the point of no return. It is the step that, once it commits, the saga is *going to complete forward*; there is no compensation for it, or compensating it is so costly/visible you treat it as irreversible. `ChargePayment` is the classic pivot: once the money has moved, you commit to fulfilling the order. (You *can* refund, but in many designs the charge is the pivot and everything after it is retried-to-success rather than rolled back — because refunding annoys customers and costs processor fees.) A saga has exactly one pivot.
- **Retriable transactions** — steps that come *after* the pivot and are guaranteed to eventually succeed if retried. `CreateShipment`, `SendReceipt`. They cannot fail the saga; they can only be slow. So you design them to be retried with backoff until they succeed, and they must be idempotent.

The design rule that falls out of this taxonomy is mechanical: **order all compensatable steps before the pivot, and all retriable steps after it.** Why? Because if anything fails *before or at* the pivot, you still can compensate everything that ran (it was all compensatable). And if anything fails *after* the pivot, you cannot compensate (you have passed the point of no return) — but that's fine, because everything after the pivot is retriable, so you drive forward to completion instead of backward. The pivot is the hinge between backward-recovery territory and forward-recovery territory.

![A vertical stack classifying saga steps as compensatable reserve inventory and apply coupon, then the pivot charge payment with no undo, then retriable create shipment and send receipt](/imgs/blogs/the-saga-pattern-in-practice-3.webp)

For ShopFast, the classification and ordering is:

1. `ReserveInventory` — compensatable (release). **Before pivot.**
2. `ApplyCoupon` / `ChargeLoyaltyHold` — compensatable. **Before pivot.**
3. `ChargePayment` — **pivot.** Once the money moves, we are completing this order.
4. `CreateShipment` — retriable. **After pivot.**
5. `SendReceiptEmail` — retriable. **After pivot.**

If your steps refuse to be ordered this way — say you have two irreversible steps, or a retriable step has a hard dependency on a compensatable step that runs after the pivot — that is the design telling you the saga is wrong, or that two of your services have a boundary in the wrong place. Reordering steps is a real design lever; sometimes the fix is to make a step compensatable (add a cancel API) so it can move before the pivot.

A subtlety worth internalizing: "the pivot" is not a property of a service or an operation in the abstract — it is a property of *this saga's risk tolerance*. `ChargePayment` is technically reversible (you can refund), so why call it the pivot rather than a compensatable step? Because in ShopFast's business, a charge-then-refund is *visible and costly* — the customer sees a charge flicker on their statement, you eat processor fees, and support gets a ticket. So we *choose* to treat the charge as the point of no return: once the money has moved, we commit to fulfilling rather than refunding, and we make everything after it retriable-to-success. A different business — say one where refunds are free and invisible (store credit, internal ledger) — might legitimately make charge a compensatable step and put the pivot somewhere else entirely, like `AllocateToWarehouse` where allocation triggers a physical pick that genuinely cannot be undone. The taxonomy is a design tool, not a fact you look up. The senior question is always "which step, once done, makes us decide to push forward no matter what?" — and the answer depends on what is *expensive to undo in your domain*, not on which API happens to have a delete endpoint.

There is also a practical constraint on *how many* compensatable steps you want before the pivot. Every compensatable step you add before the pivot is a step that, on a late failure, you must successfully undo — and each compensation is itself a network call that can fail. A saga with eight compensatable steps before the pivot has an eight-deep undo chain, and the probability that *all eight* compensations succeed is lower than for a two-deep chain. So while the taxonomy says "compensatable before pivot," the pragmatic refinement is "keep the pre-pivot chain short, and move the pivot as early as you safely can." A short pre-pivot chain means a short, reliable compensation path. This is one more reason to push the pivot early when the business allows it: it shrinks the window in which you owe the world an undo.

#### Worked example: why "charge first, reserve later" oversells

Suppose a junior orders the ShopFast steps as `ChargePayment → ReserveInventory → CreateShipment`, reasoning "get the money first, it's the important part." Now run a Black Friday spike: 200 units of a hot sneaker, 5,000 checkout attempts in the first minute. With charge-first, you charge thousands of cards *before* checking stock. The first 200 reservations succeed; reservations 201 through 5,000 fail because stock is exhausted. Now you must `RefundPayment` for 4,800 customers — 4,800 refunds, each costing a processor fee of roughly \$0.30 in non-refundable interchange on many gateways, so about \$1,440 in pure fees plus 4,800 confused customers watching a charge-then-refund flicker on their statements and flooding support.

Order it correctly — `ReserveInventory` (compensatable) *before* `ChargePayment` (pivot) — and the 4,800 that cannot get stock fail at the *reservation* step, before any money moves. Zero refunds, zero fees, zero charge-flicker. Same business logic, same services; the only difference is that the irreversible step (charge) comes after the cheaply-reversible one (reserve). That is the entire payoff of classifying steps, and it is worth real money.

## Two coordination styles: choreography vs orchestration

A saga is a *pattern*, not an implementation. The same five steps can be coordinated two fundamentally different ways, and this is the single biggest architectural decision you make when building one. It is the same choreography-versus-orchestration axis we dissected for general event flows in [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — here we apply it specifically to sagas, where the stakes are higher because failure handling is now in scope.

![A before and after comparison contrasting a choreography saga where events chain steps with no central state against an orchestration saga where a coordinator issues commands and holds saga state in one place](/imgs/blogs/the-saga-pattern-in-practice-2.webp)

**Choreography.** No central brain. Each service listens for events and reacts. The order service emits `OrderCreated`; the inventory service hears it, reserves stock, emits `InventoryReserved`; the payment service hears *that*, charges the card, emits `PaymentCharged`; the shipping service hears that and ships. Compensation is just more events: if payment fails, payment emits `PaymentFailed`, the inventory service hears it and releases stock, emits `InventoryReleased`, and the order service hears that and marks the order cancelled. The "saga" exists only as an emergent property of services reacting to each other.

The appeal is real: maximal decoupling, no single point that every team has to coordinate through, each service deployable independently. The cost is brutal for anything non-trivial: **there is no single place that knows the saga's state.** To answer "where is order #4471 stuck?" you have to reconstruct the flow from logs across every service. Adding a step means finding the right event to subscribe to and emitting a new one — and getting the failure-event wiring right *everywhere*. Worst of all, the failure logic is scattered: each service has to know "if I hear `PaymentFailed`, I undo my part," which means the rules of the saga are smeared across N codebases with no one owning the whole. Cyclic event dependencies — service A's event eventually triggers service A again — creep in unseen. Choreography sagas are fine at 2–3 steps with loosely-coupled teams; past that they become the distributed monolith we warn about in the [shared-data anti-patterns](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith) post, just hidden inside the event bus.

**Orchestration.** One service — the *orchestrator* or *coordinator* — owns the saga. It holds the saga's state explicitly (in a database row), issues commands to each service in turn ("Payment service: charge order #4471"), waits for the reply, and decides the next step. On failure it issues the compensation commands in reverse. The flow lives in one readable place; the saga's state is one queryable row; adding a step is a code change in one service. The cost is a new coupling point: the orchestrator knows about every service it drives, and it can swell into a god-service if you let business logic that belongs in the participant services leak into the coordinator. The discipline is: the orchestrator decides *sequencing and failure handling*; each service decides *how* to do its own step.

![A flow graph showing a saga orchestrator persisting state to a durable saga instance row and issuing commands to inventory, payment, and shipping services, with a declined payment driving a compensate path that undoes prior steps](/imgs/blogs/the-saga-pattern-in-practice-6.webp)

My strong default for any saga with more than about three steps, or any saga with real compensation logic, is **orchestration** — and increasingly, orchestration delegated to a durable workflow engine rather than hand-rolled, for reasons we will reach. The visibility you buy is worth the coupling you pay, every time you are paged at 3am and need to answer "where is it stuck?" in seconds instead of hours. The mechanism deep-dive makes the same call and explains the second-order reason: with orchestration, the orchestrator is the *only writer* of saga state, which makes the state machine far easier to reason about.

![A tree decision diagram that routes a cross-service operation to a local transaction if it fits one aggregate, to choreography for three or fewer steps with loose teams, and to a workflow engine for more steps or real compensation](/imgs/blogs/the-saga-pattern-in-practice-7.webp)

## The decision matrix: saga vs the alternatives

Before building, prove to yourself a saga is even the right tool. Here are the four honest options for a cross-service operation, scored on the dimensions that actually decide it. The matrix figure renders the same comparison; the table is the canonical reference.

![A decision matrix comparing no cross-service transaction, choreography saga, orchestration saga, and two-phase commit across coupling, failure visibility, cross-service isolation, availability, and build complexity](/imgs/blogs/the-saga-pattern-in-practice-4.webp)

| Dimension | No cross-svc txn | Choreography saga | Orchestration saga | 2PC / XA |
|---|---|---|---|---|
| **Cross-service coupling** | None | Loose (event bus) | Central hub | Tight (lock protocol) |
| **Failure visibility** | N/A | Scattered across N services | One place (state row) | At the coordinator |
| **Cross-service isolation** | None | None | None | Full ACID |
| **Availability under partial failure** | High | High | High | Low (blocking) |
| **Build / operate complexity** | Lowest | High (wiring, no central view) | Medium | High + rare tooling |
| **When it wins** | Op fits one aggregate | 2–3 steps, autonomous teams | >3 steps, need a single view | Same DB / true ACID need |

The two rows that decide most arguments are **isolation** and **availability**. If you genuinely need cross-service isolation — no other transaction may *ever* see partial state — a saga cannot give you that, and you must either keep the operation inside one aggregate (so it is one local transaction) or, in the rare case where the components actually support XA and you can tolerate blocking, use 2PC. If you need availability — the checkout must keep working when one downstream is briefly slow — the saga wins and 2PC loses, because 2PC blocks on the slowest participant.

The often-forgotten first column matters most: **the cheapest correct saga is no saga at all.** If you can redraw your service boundaries so the whole operation lives in one aggregate / one database, you keep a real ACID transaction and skip this entire post. We will return to this in the "when not to" section — it is the senior move that juniors miss.

## Building it #1: a hand-rolled orchestrator state machine

Let us build the ShopFast place-order saga as an explicit state machine. The orchestrator is itself a service with its own database table — the *saga log* — that records, durably, exactly where each saga is. The single most important property is that the saga's state must be persisted *before* and *after* every step, so that if the orchestrator crashes mid-saga it can recover by reading the row and resuming. A non-durable orchestrator (state only in memory) is the classic "orphaned saga" bug: the process dies, and the saga is stuck forever with stock held and no one to release it.

First, the saga state table. This is the heart of the whole thing.

```sql
-- The saga log: one row per running saga, durably persisted.
CREATE TABLE order_saga (
    saga_id        UUID PRIMARY KEY,
    order_id       UUID NOT NULL,
    state          TEXT NOT NULL,          -- STARTED, INVENTORY_RESERVED, PAID, ...
    direction      TEXT NOT NULL DEFAULT 'FORWARD', -- FORWARD | COMPENSATING
    payload        JSONB NOT NULL,         -- amounts, sku, qty, payment token, etc.
    completed_steps JSONB NOT NULL DEFAULT '[]', -- so we know what to compensate
    last_error     TEXT,
    deadline_at    TIMESTAMPTZ,            -- when this step is considered "stuck"
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX order_saga_stuck ON order_saga (deadline_at)
    WHERE state NOT IN ('COMPLETED', 'CANCELLED');
```

Now the orchestrator. I will write it in Go because the explicit-state, explicit-error style suits a state machine, and the structure transfers directly to any language. Each step is a `(action, compensation)` pair; the orchestrator runs actions forward, recording each completed step, and runs compensations backward on failure.

```go
package saga

import (
    "context"
    "errors"
    "time"
)

// A Step is one local transaction plus its semantic undo.
type Step struct {
    Name       string
    Action     func(ctx context.Context, s *OrderSaga) error
    Compensate func(ctx context.Context, s *OrderSaga) error // nil for the pivot
    Retriable  bool // steps after the pivot
}

// The ordered ShopFast place-order saga. Compensatable steps first,
// the pivot (ChargePayment) in the middle, retriable steps last.
func placeOrderSteps(inv InventoryClient, pay PaymentClient, ship ShippingClient) []Step {
    return []Step{
        {
            Name: "ReserveInventory",
            Action: func(ctx context.Context, s *OrderSaga) error {
                return inv.Reserve(ctx, s.OrderID, s.SKU, s.Qty)
            },
            Compensate: func(ctx context.Context, s *OrderSaga) error {
                return inv.Release(ctx, s.OrderID) // idempotent
            },
        },
        {
            Name: "ChargePayment", // PIVOT: no Compensate; after this we go forward.
            Action: func(ctx context.Context, s *OrderSaga) error {
                return pay.Charge(ctx, s.OrderID, s.Amount, s.PaymentToken)
            },
            Compensate: nil,
        },
        {
            Name:      "CreateShipment", // retriable
            Retriable: true,
            Action: func(ctx context.Context, s *OrderSaga) error {
                return ship.Create(ctx, s.OrderID, s.Address)
            },
        },
    }
}
```

And the driver. The forward loop runs each action, persisting state after each success. On a failure *before or at* the pivot, it flips to `COMPENSATING` and walks the completed steps in reverse. On a failure *after* the pivot, it retries with backoff because those steps are retriable and must complete.

```go
func (o *Orchestrator) Run(ctx context.Context, s *OrderSaga) error {
    steps := o.steps
    pivotIdx := indexOfPivot(steps) // the step with Compensate == nil

    // ---- forward ----
    for i := 0; i < len(steps); i++ {
        step := steps[i]
        err := step.Action(ctx, s)

        if err == nil {
            s.CompletedSteps = append(s.CompletedSteps, step.Name)
            s.State = step.Name + "_DONE"
            s.DeadlineAt = time.Now().Add(stepTimeout(step))
            o.store.Save(ctx, s) // DURABLE checkpoint after every step
            continue
        }

        // A step failed.
        if i > pivotIdx {
            // After the pivot: we cannot go back. Retry to completion.
            return o.retryForward(ctx, s, steps, i) // backoff + jitter
        }
        // Before or at the pivot: compensate everything that ran.
        s.Direction = "COMPENSATING"
        s.LastError = err.Error()
        o.store.Save(ctx, s)
        return o.compensate(ctx, s, steps, i-1)
    }

    s.State = "COMPLETED"
    o.store.Save(ctx, s)
    return nil
}

// Walk completed steps in REVERSE, running each compensation.
func (o *Orchestrator) compensate(ctx context.Context, s *OrderSaga, steps []Step, from int) error {
    for i := from; i >= 0; i-- {
        c := steps[i].Compensate
        if c == nil {
            continue // pivot has no compensation
        }
        if err := retry(ctx, 5, func() error { return c(ctx, s) }); err != nil {
            // Compensation itself failed after retries: this is a hard alert.
            s.State = "COMPENSATION_FAILED"
            o.store.Save(ctx, s)
            return errors.New("compensation failed, manual intervention: " + s.OrderID.String())
        }
    }
    s.State = "CANCELLED"
    o.store.Save(ctx, s)
    return nil
}
```

Three things in this code are load-bearing and easy to get wrong:

1. **`o.store.Save` after every step.** This is the durability that lets a crashed orchestrator resume. Without it, you have an in-memory state machine and an orphaned-saga bug waiting to happen.
2. **Compensation runs in reverse and retries hard (5 attempts).** A compensation that fails is far worse than a normal step that fails, because it means you are stuck mid-undo — money refunded but stock not released, say. We treat `COMPENSATION_FAILED` as a page-someone alert, not a log line.
3. **The pivot is `nil` compensation by construction**, and the code branches on `i > pivotIdx` to choose forward-retry vs backward-compensate. The taxonomy from the design section is literally encoded in the control flow.

### How recovery actually works, step by step

The piece that makes a hand-rolled orchestrator correct rather than merely plausible is recovery, and it is worth walking through the exact sequence because this is where most home-grown orchestrators have a latent bug that only fires under a crash. The orchestrator is itself a service with replicas; one of them owns a given saga at a time. On startup — and periodically thereafter — each replica runs a recovery scan:

```go
// Recovery: resume every saga that was mid-flight when a replica died.
func (o *Orchestrator) RecoverInflight(ctx context.Context) error {
    rows, _ := o.store.FindNonTerminal(ctx) // state NOT IN (COMPLETED, CANCELLED, COMPENSATION_FAILED)
    for _, s := range rows {
        // Claim the saga so two replicas don't resume it at once.
        if !o.store.TryClaim(ctx, s.SagaID, o.replicaID, 30*time.Second) {
            continue // another replica owns it
        }
        switch s.Direction {
        case "FORWARD":
            // Re-issue from the first step NOT in completed_steps. Idempotency
            // makes re-issuing an already-applied step a safe no-op.
            o.resumeForward(ctx, s)
        case "COMPENSATING":
            // Continue walking completed_steps in reverse from where we stopped.
            o.resumeCompensation(ctx, s)
        }
    }
    return nil
}
```

The subtle correctness argument is this. Suppose the orchestrator crashed *after* calling `pay.Charge` but *before* it managed to persist `PAID`. On recovery, the saga row still says `INVENTORY_RESERVED`, so `resumeForward` re-issues `ChargePayment`. Did we just double-charge? No — *only because* `ChargePayment` is idempotent on the order id and passes the idempotency key to the processor. The recovery logic re-issues a step it is not sure committed, and idempotency turns that uncertainty into safety. This is the single most important reason idempotency is not optional in a saga: the recovery path *depends* on it being safe to re-run a step whose outcome is unknown. An orchestrator without idempotent steps is correct only as long as it never crashes, which is to say, it is not correct.

The `TryClaim` with a lease (30 seconds here) is the other recovery subtlety. Two orchestrator replicas must never drive the same saga simultaneously, or they will both re-issue steps and race. A short lease on the saga row — claim it, renew while working, let it expire if you die — gives at-most-one-active-driver without a separate distributed lock service. If a replica claims a saga and then dies, the lease expires and another replica picks it up on the next scan. This is the same lease pattern you would use for any singleton-per-key work; it is just applied to "who is driving this saga right now."

One more recovery nuance that bites teams: **never delete the saga row when it completes.** Keep terminal rows (`COMPLETED`, `CANCELLED`) for an audit/debug window — at least long enough to investigate "why was this order cancelled?" weeks later. A common variant keeps the row and adds a `archived_at` so the hot table stays small while history survives. Deleting on completion saves a little storage and costs you every future investigation.

### The choreography version, for contrast

If you went choreography instead, there is no orchestrator and no central state row. Each service is an event handler plus a compensating handler. Here is the payment service's pair, in Python, consuming from the broker (this is the realistic shape — see [the anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) for the producer/broker/consumer plumbing this rides on):

```python
# Choreography: payment service reacts to InventoryReserved and emits the next event.
@on_event("InventoryReserved")
def handle_inventory_reserved(evt):
    order_id = evt["order_id"]
    # Idempotency: have we already processed THIS event?
    if processed_events.seen(evt["event_id"]):
        return
    try:
        charge_id = psp.charge(order_id, evt["amount"], evt["payment_token"],
                               idempotency_key=order_id)  # exactly-once at the PSP
        processed_events.record(evt["event_id"])
        publish("PaymentCharged", {"order_id": order_id, "charge_id": charge_id})
    except CardDeclined:
        # Compensation is just ANOTHER event the upstream services react to.
        publish("PaymentFailed", {"order_id": order_id, "reason": "card_declined"})

# The compensating handler lives in the INVENTORY service, not here.
# Inventory must know the saga's failure rules — that scattering is the cost.
```

```python
# Over in the inventory service: the compensating handler.
@on_event("PaymentFailed")
def handle_payment_failed(evt):
    if processed_events.seen(evt["event_id"]):
        return
    inventory.release(evt["order_id"])     # idempotent compensation
    processed_events.record(evt["event_id"])
    publish("InventoryReleased", {"order_id": evt["order_id"]})
```

Read those two handlers and you can already feel the problem: the saga's failure rule ("on `PaymentFailed`, release inventory") lives in the inventory service, the success rule lives in payment, and to understand the whole flow you must hold all the handlers in your head at once. Compare that to the orchestrator, where the entire flow is one readable `Run` function. That difference is why I default to orchestration for anything non-trivial.

There is a second, quieter problem with the choreography version that the code does not show: **nobody owns the saga's lifecycle.** In the orchestrator, the `Run` function is born when the saga starts and dies when it reaches a terminal state — there is a clear owner with a clear beginning and end, and a clear place to attach a timeout. In choreography, the "saga" is never instantiated as a thing; it is an emergent pattern of independent reactions. That means there is no object whose job is to notice "step 2 fired but step 3 never did, 30 seconds ago." Each service only knows about the events it personally subscribes to. To add a timeout to a choreography saga you have to invent a *separate* watchdog service that subscribes to the start event, starts a timer, subscribes to the end event, and alerts if the timer fires first — which is, ironically, a partial orchestrator bolted on after the fact. By the time you have built that watchdog, you have rebuilt the worst half of an orchestrator (the central coupling) without getting the best half (the single readable flow). This is the deepest reason choreography sagas do not age well: the visibility you eventually need forces you to grow an orchestrator anyway, and growing one late is harder than designing one up front.

## Backward recovery vs forward recovery

The orchestrator code above already encodes the two recovery strategies, but they deserve a name and a clear rule because choosing wrong is a common bug.

**Backward recovery** is compensation: undo the steps that ran, end in a cancelled/failed state. This is what you do when a failure happens *before or at the pivot* — you still can walk everything back. ShopFast: payment declines, so you release the inventory and cancel the order. The customer ends up where they started.

**Forward recovery** is retry-to-completion: the failure is *after the pivot*, so there is no going back; you retry the failed step (and subsequent retriable steps) until they succeed. ShopFast: payment already charged, but `CreateShipment` returned a 503 because the shipping service was briefly down. You do *not* refund the customer and cancel — you keep retrying `CreateShipment` with backoff until it succeeds, because that is the only correct end state once the money has moved. This is why every post-pivot step must be designed to *eventually succeed if retried* — if a post-pivot step can fail permanently, your saga has no valid terminal state and the design is broken.

![A timeline of five events showing reserve inventory succeeding, charge payment being declined, the saga marking itself compensating, release inventory returning the stock, and the order ending in a clean idempotent cancelled state](/imgs/blogs/the-saga-pattern-in-practice-5.webp)

#### Worked example: a payment failure, with timing

Let us trace the most common ShopFast failure end to end with real numbers, because "compensation happens" is too vague to debug at 3am. A customer places an order for 1 unit of SKU `SNK-42`, total \$129.00.

- **T+0ms** — Orchestrator persists `order_saga` row, state `STARTED`, then calls `inv.Reserve`. Inventory's local transaction decrements `available` by 1 and writes a reservation row in 4ms. Orchestrator records `ReserveInventory` in `completed_steps`, state `INVENTORY_RESERVED`, saves (2ms write). **1 unit now held.**
- **T+180ms** — Orchestrator calls `pay.Charge` against the external processor. The processor takes 175ms and returns `card_declined`. This failure is *at the pivot*, so we are in backward-recovery territory.
- **T+185ms** — Orchestrator sets `direction = COMPENSATING`, `last_error = card_declined`, saves the row. This save is the durability point: even if the orchestrator crashes right now, recovery sees `COMPENSATING` and knows to run compensations.
- **T+240ms** — Orchestrator walks `completed_steps` in reverse: `ReserveInventory` → run its compensation `inv.Release`. Inventory's local transaction increments `available` by 1 and marks the reservation released, in 4ms. The release is idempotent: if it runs twice (because the orchestrator crashed and resumed), the second call is a no-op.
- **T+250ms** — Orchestrator sets state `CANCELLED`, saves. The order ends cancelled, **1 unit returned to available**, no money moved (the charge never succeeded), and the customer gets a "we couldn't complete your order" message.

Total compensation latency: ~70ms after the failure. The key invariant the timeline proves: **only the step that actually committed (`ReserveInventory`) is compensated; the pivot that failed (`ChargePayment`) leaves nothing to undo.** The whole saga, including compensation, finished in a quarter-second, and the stock count is correct.

## The hard part: lost isolation and its anomalies

Now the genuinely tricky part, the part that separates a saga that works in the demo from one that works on Black Friday. Because a saga gives up isolation, *other transactions can observe its partial state*. The mechanism deep-dive enumerates these formally; here is what they look like in ShopFast and how you fix each in code.

**Dirty read / the oversold-stock anomaly.** Saga A reserves the last unit of `SNK-42` (stock now appears as 0 reserved-but-not-paid). Saga B starts before A finishes and, depending on how you wrote the read, might see the unit as available and reserve it too — now two orders are headed for one unit, and the loser only finds out at shipment time. This is the classic lost-update / dirty-read problem, and it is the anomaly you will hit first.

![A before and after comparison showing two concurrent sagas both reading stock equals one and both reserving it leading to an oversold double ship, versus a pending state lock where saga A holds the unit and saga B is rejected so only one ships](/imgs/blogs/the-saga-pattern-in-practice-8.webp)

The countermeasure is the **semantic lock**: a "pending" / "tentative" state that you set at the start of a saga and clear at the end, and that *other sagas respect*. It is a lock implemented in your own application state, not by the database. For ShopFast inventory, you do not just have `available`; you track *held* (reserved by an in-flight saga) versus *committed*. A reservation atomically moves a unit from available to held; another saga reading availability subtracts held, so it cannot grab a unit that is tentatively claimed. In SQL, the reservation is a conditional, atomic decrement:

```sql
-- Semantic lock: reserve only if truly available; held units block other sagas.
-- This whole statement is one atomic local transaction in the inventory service.
UPDATE inventory
SET available = available - 1,
    held      = held + 1
WHERE sku = 'SNK-42'
  AND available >= 1           -- the guard: cannot reserve what isn't there
RETURNING held;
-- If 0 rows updated, the unit is gone; the saga fails ReserveInventory cleanly.

-- A second saga running the SAME statement concurrently is serialized by the
-- row lock and sees available = 0, so it is rejected. No oversell.
```

The reservation row also carries a saga reference and a `status`:

```sql
CREATE TABLE reservation (
    order_id   UUID PRIMARY KEY,
    sku        TEXT NOT NULL,
    qty        INT  NOT NULL,
    status     TEXT NOT NULL,   -- PENDING | COMMITTED | RELEASED  (the semantic lock)
    saga_id    UUID NOT NULL,
    expires_at TIMESTAMPTZ      -- so a crashed saga's lock can be reclaimed
);
```

The `status = PENDING` is the semantic lock made explicit. A second saga that queries "is this unit shippable?" treats `PENDING` as unavailable. When the saga completes, status flips to `COMMITTED`; when it compensates, status flips to `RELEASED` and the unit returns to `available`. The `expires_at` matters: if the saga's orchestrator crashes and never releases, a sweeper reclaims expired pending reservations so the lock cannot leak forever.

**Other countermeasures, briefly** (the deep-dive covers the full menu):

- **Commutative updates.** Design compensations so order does not matter. `Release(+1)` and `Reserve(-1)` are commutative on a counter — applying them in any order yields the right total — so a delayed compensation cannot corrupt the count. Make your steps add/subtract deltas rather than set absolute values, and a lot of ordering anxiety disappears. The contrast that makes this vivid: a *non*-commutative compensation, like `SET balance = balance - 50` paired with a compensation `SET balance = original_balance`, breaks the instant a concurrent transaction also touched the balance, because restoring the *absolute* original value silently discards the concurrent change — a lost update. The same logic written as deltas (`balance = balance - 50`, compensate with `balance = balance + 50`) is immune, because the two deltas commute with anything else happening to the balance. Prefer relative deltas over absolute sets in any value a saga touches.
- **Reread value ("by value").** Before a step commits an irreversible effect, re-read the value it depends on and abort if it changed under you. ShopFast re-reads the price right before `ChargePayment`; if a concurrent price update changed it, abort rather than charge the stale amount. This is optimistic concurrency applied to a saga: you do not lock the price for the saga's whole duration (that would re-introduce the cross-service locking we are trying to avoid), you just check at the last moment that the assumption you started with still holds, and bail cleanly if it does not. The aborted saga compensates and the customer retries with the current price — annoying, but correct, and far better than charging a stale amount.
- **Pessimistic view / reordering.** Reorder steps so the window of exposed partial state is as short as possible — e.g., do the step that exposes risky partial state as late as you can before the pivot. If the dirty-read window is 50ms instead of 5s, the probability another saga observes the partial state drops by two orders of magnitude. You cannot eliminate the window without isolation you do not have, but you can shrink it until anomalies are rare enough to handle by exception rather than by design.

The honest framing of all these countermeasures: **you are reconstructing, by hand and per-field, the isolation the database used to give you for free.** That is real work, and it is the hidden cost of a saga that the happy-path tutorials skip. A senior pricing out a saga counts not just the orchestrator and the compensations but the isolation countermeasures for every field two concurrent sagas might fight over. If that count is large — many shared, contended fields — that is a signal the operation maybe should not have been split across services in the first place, which brings us back to the "keep it in one aggregate" escape hatch.

#### Worked example: two concurrent sagas race for the last unit

Concrete numbers. `SNK-42` has `available = 1, held = 0`. Two checkouts, saga A and saga B, arrive 3ms apart.

*Without the semantic lock* (naive `SELECT available; if >0 then reserve`): A reads `available = 1` at T+0. B reads `available = 1` at T+3ms (A has not committed its decrement yet). Both pass the `if > 0` check. Both write a reservation. `available` is now `-1` or the second write silently overwrites — either way **two orders claim one unit.** Both charge successfully (the cards are fine). At shipment, the warehouse has one sneaker and two paid orders. You now run an *unplanned* compensation: refund one customer, who is furious because they got a confirmation email and a charge for a product you cannot ship. This is the "oversold sneaker" incident in production, and it traces directly to the missing isolation.

*With the semantic lock* (the conditional `UPDATE ... WHERE available >= 1`): A's update acquires the row lock, sets `available = 0, held = 1`, commits in 4ms. B's identical update *blocks* on the row lock until A commits, then evaluates `WHERE available >= 1` against the now-current `available = 0`, matches zero rows, and returns. The inventory client sees "0 rows updated" and fails `ReserveInventory` for saga B *before any money moves*. Saga B ends `CANCELLED` at the reservation step; customer B gets "out of stock," no charge, no refund, no fury. The database's row lock did the serialization; the semantic lock turned that serialization into a clean business outcome. **One unit, one shipment, zero oversell.** The fix is two lines of SQL and a `status` column, and it is the highest-leverage thing in this whole post.

## Idempotency and exactly-once effects

Every step and every compensation in a saga **will** run more than once. The orchestrator retries on timeout; the broker redelivers; a crashed orchestrator resumes and re-issues a command it is not sure committed. If `ChargePayment` runs twice, you double-charge the customer. If `ReleaseInventory` runs twice, you return two units when one was held. The only defense is idempotency: each operation must be safe to apply repeatedly, producing the same effect as applying it once. This is so central that it gets its own posts — [idempotency and deduplication, making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) for the mechanism, and the sibling [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) for the service-topology view — but you cannot ship a saga without it, so here is the practitioner's version.

The pattern is an **idempotency key** carried on every step, plus a dedup record at the receiver. For ShopFast, the natural key is the saga/order id (one charge per order). The receiver records keys it has processed and short-circuits repeats:

```go
// Receiver side (e.g. inside the Payment service's Charge handler).
func (p *PaymentService) Charge(ctx context.Context, orderID uuid.UUID,
    amount Money, token string) (ChargeID, error) {

    // Idempotency: have we already charged THIS order?
    if existing, ok := p.charges.Lookup(orderID); ok {
        return existing.ChargeID, nil // return the SAME result, do not re-charge
    }

    // The external PSP also wants its own idempotency key — exactly-once at the edge.
    res, err := p.psp.Charge(stripe.ChargeParams{
        Amount:         amount.Cents,
        Source:         token,
        IdempotencyKey: orderID.String(), // Stripe/Adyen dedup on this
    })
    if err != nil {
        return "", err
    }
    // Record atomically with the business write so a crash can't lose the dedup.
    p.charges.Record(ctx, orderID, res.ID)
    return ChargeID(res.ID), nil
}
```

Two layers of idempotency here, both necessary:

1. **Your own dedup table** keyed on `orderID` — so a retried command returns the prior result instead of charging again.
2. **The processor's idempotency key** — every serious payment gateway (Stripe, Adyen, Braintree) accepts an `Idempotency-Key` header and guarantees one charge per key for 24h+. You *must* pass it; relying only on your own table is racy if your record write and the PSP call are not atomic. Passing the key makes the PSP itself the exactly-once boundary.

Compensations need the same treatment. `ReleaseInventory` keyed on `orderID` flips the reservation `status` to `RELEASED` only if it is currently `PENDING`; if it is already `RELEASED`, it is a no-op. That conditional flip *is* the idempotency:

```sql
-- Idempotent compensation: only release a PENDING reservation, exactly once.
UPDATE reservation
SET status = 'RELEASED'
WHERE order_id = $1 AND status = 'PENDING'
RETURNING qty;
-- 0 rows => already released (or never reserved) => no-op, return success.
```

The deeper guarantee — that the *event publishing* itself is atomic with the local transaction, so you never commit a charge but fail to tell the orchestrator — comes from the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing), explored for this series in [the transactional outbox and reliable event publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) and at the storage layer in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern). In a choreography saga especially, the outbox is non-negotiable: it is what guarantees the `PaymentCharged` event is published if and only if the charge committed.

## Building it #2: a Temporal-style workflow engine

Hand-rolling the orchestrator teaches you the mechanics, and for a 3-step saga it is fine. But notice everything the hand-rolled version made *you* responsible for: persisting state after every step, recovering after a crash, retrying with backoff, running compensations in reverse, timing out stuck steps, deduping retries. That is a lot of distributed-systems plumbing, and getting all of it right for every saga in your company is a losing battle. This is why production teams increasingly reach for a **durable workflow engine** — Temporal, Cadence (its predecessor), or a managed equivalent like AWS Step Functions or Netflix Conductor. The engine provides *durable execution*: your workflow code looks like a plain function calling steps in sequence, and the engine transparently persists every step's result, replays your function from history after a crash, and handles retries and timeouts for you.

The conceptual shift is large enough to be worth stating: with a workflow engine, **the state machine disappears into ordinary control flow.** You write `if/else` and `try/finally` and the engine makes them durable. Here is the ShopFast saga as a Temporal workflow in Go. Compare its readability to the hand-rolled state machine — the saga is just a function, and compensation is a deferred cleanup stack.

```go
// A Temporal workflow: durable execution makes the orchestrator's bookkeeping vanish.
func PlaceOrderWorkflow(ctx workflow.Context, order Order) error {
    ao := workflow.ActivityOptions{
        StartToCloseTimeout: 30 * time.Second,
        RetryPolicy: &temporal.RetryPolicy{   // the engine retries for you
            InitialInterval:    time.Second,
            BackoffCoefficient: 2.0,
            MaximumAttempts:    5,
        },
    }
    ctx = workflow.WithActivityOptions(ctx, ao)

    // Compensation stack: register undos as we go, run them on failure.
    var compensations []func()
    compensate := func() {
        // Run in REVERSE, each as its own retried activity.
        for i := len(compensations) - 1; i >= 0; i-- {
            compensations[i]()
        }
    }

    // Step 1: ReserveInventory (compensatable)
    if err := workflow.ExecuteActivity(ctx, ReserveInventory, order).Get(ctx, nil); err != nil {
        return err // nothing to compensate yet
    }
    compensations = append(compensations, func() {
        _ = workflow.ExecuteActivity(ctx, ReleaseInventory, order).Get(ctx, nil)
    })

    // Step 2: ChargePayment (PIVOT)
    if err := workflow.ExecuteActivity(ctx, ChargePayment, order).Get(ctx, nil); err != nil {
        compensate()           // backward recovery: undo the reservation
        return err
    }
    // Past the pivot: from here we go FORWARD only.

    // Step 3: CreateShipment (retriable — the engine retries to success)
    if err := workflow.ExecuteActivity(ctx, CreateShipment, order).Get(ctx, nil); err != nil {
        // Retriable: do NOT compensate the charge; the RetryPolicy already
        // retried 5x. Escalate for forward recovery instead of refunding.
        return fmt.Errorf("shipment stuck after retries, needs ops: %w", err)
    }
    return nil
}
```

Everything the hand-rolled orchestrator did manually — the durable saga row, the resume-after-crash, the per-step retry policy — the engine does for you. If the worker process running this workflow crashes at step 2, Temporal *replays* the workflow function from its event history on another worker, skipping the already-completed `ReserveInventory` (it returns the recorded result) and resuming exactly where it left off. You did not write a single line of recovery code. The activities themselves (`ReserveInventory`, `ChargePayment`) are plain functions that call your services and must still be idempotent — the engine guarantees the *workflow* is durable, but an activity can run more than once on retry, so the idempotency keys from the previous section still apply.

The trade-off, stated honestly because the matrix demands it: you now run and operate a workflow engine (a Temporal cluster is a real piece of infrastructure with its own database, or you pay for Temporal Cloud), and your saga logic is coupled to the engine's programming model. For a single 3-step saga, that is overkill — the hand-rolled version is less infrastructure. For a company with dozens of multi-step workflows (orders, refunds, onboarding, payouts), the engine pays for itself many times over because every team stops reinventing durable-state plumbing. The crossover is roughly: more than a couple of multi-step sagas, or any saga with non-trivial compensation, and a workflow engine is the senior call. (This is exactly the tree-diagram routing above.)

One caution about workflow engines that the marketing pages skip: durable execution is not magic, and it imposes a real constraint on how you write workflow code. Because the engine recovers a crashed workflow by *replaying the function from its recorded history*, your workflow function must be **deterministic** — it must produce the same sequence of activity calls every time it is replayed with the same history. That means no `time.Now()`, no random numbers, no direct network calls, no reading a config file *inside the workflow function* — anything non-deterministic must be done inside an *activity* (which is recorded) and not in the workflow body (which is replayed). A junior who calls `rand.Intn()` or `time.Now()` directly in a Temporal workflow ships a time bomb: it works in testing and then, on the first crash-and-replay in production, the replay diverges from history and the engine throws a non-determinism error mid-recovery. The engine gives you durability for free, but it bills you in a programming discipline you must actually learn. This is the genuine cost behind "the state machine disappears" — it does not disappear, it moves into the engine's runtime, and the price of admission is writing deterministic workflows.

#### Worked example: the cost crossover between hand-rolled and engine

Put numbers on "when does the engine pay for itself," because "it depends" is not an answer a senior gives in a planning meeting. Suppose hand-rolling a *single* robust saga — orchestrator service, durable state table, recovery scan, lease-based claiming, sweeper, compensation retries, alerting — is roughly two engineer-weeks to build and, critically, an ongoing tax: every subtle distributed-systems bug (a missed durability checkpoint, a non-idempotent compensation, a leaked lease) is a production incident that costs hours to diagnose. Call it 2 weeks build plus, conservatively, half a day a month of operate-and-fix per saga.

Now suppose your company has *eight* distinct multi-step business workflows over two years — orders, refunds, subscription changes, merchant onboarding, payouts, dispute handling, account closure, data export. Hand-rolling all eight is ~16 engineer-weeks of build, plus eight sagas each carrying that half-day-a-month tax — roughly 4 engineer-days a month of operations across the fleet, every month, forever, and eight separate places for the same class of bug to hide. Standing up a Temporal cluster (or buying Temporal Cloud) is a larger up-front cost — call it 4–6 engineer-weeks to operate the cluster, build the shared deployment, and train the teams — but then each *additional* workflow is days, not weeks, because the durable-execution plumbing is already solved once, centrally, and tested by the whole company's traffic. Past roughly the third or fourth non-trivial saga, the cumulative hand-rolled tax (build + monthly operate + scattered bug surface) exceeds the engine's fixed cost, and every saga after that is pure savings. That is the crossover: not "two sagas," not "ten," but the point where the *recurring* per-saga operations tax of N hand-rolled orchestrators overtakes the *one-time* fixed cost of running one engine — which, for most teams shipping several multi-step workflows, lands at three or four sagas.

## Optimization: making the saga production-grade

A correct saga can still be slow and brittle. Three optimizations matter, all measurable.

**1. Parallelize independent steps.** If two pre-pivot steps do not depend on each other, run them concurrently instead of in sequence. ShopFast's `ReserveInventory` and `ApplyLoyaltyHold` are independent — both must succeed before the pivot, but neither needs the other's result. Running them sequentially costs `t(reserve) + t(loyalty)`; running them in parallel costs `max(t(reserve), t(loyalty))`.

```go
// Parallel fan-out for independent pre-pivot steps, then join before the pivot.
future1 := workflow.ExecuteActivity(ctx, ReserveInventory, order)
future2 := workflow.ExecuteActivity(ctx, ApplyLoyaltyHold, order)

err1 := future1.Get(ctx, nil)
err2 := future2.Get(ctx, nil)
if err1 != nil || err2 != nil {
    compensate() // undo whichever of the two actually succeeded
    return errors.Join(err1, err2)
}
// Both done; proceed to the pivot.
```

The win is real but bounded: only *independent* steps parallelize, and you must compensate whichever of the parallel steps succeeded if its sibling failed (which is why the compensation stack registers each independently). If `ReserveInventory` takes 40ms and `ApplyLoyaltyHold` takes 60ms, sequential is 100ms, parallel is 60ms — a 40% latency cut on the pre-pivot phase for a 4-line change. Note the pivot and post-pivot steps usually *cannot* parallelize, since the pivot gates everything after it.

**2. Timeouts on every step.** A saga with no per-step timeout can hang forever waiting on a wedged downstream — the "stuck pivot with no timeout owner" failure. Every step gets a `StartToCloseTimeout` (Temporal) or a `deadline_at` (hand-rolled). Set it from the downstream's p99 plus headroom: if `ChargePayment`'s p99 is 175ms, a 2-second timeout is generous headroom that still catches a truly hung call. In choreography, *nobody owns the timeout* — there is no central clock watching the whole saga — which is another quiet argument for orchestration.

**3. Retry with backoff and jitter.** Retriable post-pivot steps (and compensations) retry on transient failure, but a naive immediate-retry loop hammers a struggling downstream and turns a blip into an outage. Exponential backoff with jitter spreads the retries: 1s, 2s, 4s, 8s, each ± a random jitter so a thousand sagas retrying do not synchronize into a thundering herd. This interacts with [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control): backoff *is* client-side backpressure on a struggling dependency.

#### Worked example: saga success-rate and stuck-saga alerting math

You cannot operate what you cannot measure. Define the metrics and the alert thresholds with numbers.

ShopFast runs ~50,000 place-order sagas/day, ~0.58/second average, peaking at ~5/second. Empirically, ~3% of sagas hit *backward recovery* (mostly card declines) — that is *normal and expected*, not an incident; those are clean cancellations. The numbers you alert on are different:

- **Stuck-saga rate.** A saga is "stuck" if it has been in a non-terminal state past its `deadline_at`. A sweeper job queries `WHERE deadline_at < now() AND state NOT IN ('COMPLETED','CANCELLED')` every 10 seconds. Baseline stuck rate should be near zero (transient blips that resolve on retry). Alert if the stuck rate exceeds **0.5% of in-flight sagas for 5 minutes** — at 5/sec that is ~15 sagas/minute, so a sustained handful of genuinely-stuck sagas trips it. Below that threshold, retries are handling it; above it, a downstream is actually down and a human should look.
- **Compensation-failure rate.** `COMPENSATION_FAILED` is a *paging* alert at **any** rate above zero sustained — a failed compensation means money/stock is in an inconsistent state needing manual repair. Even one per hour warrants investigation; a spike is a 3am page.
- **Saga p99 duration.** If the happy-path saga normally completes in 250ms p99 and that climbs to 2s, a downstream is degrading before it fully fails — a leading indicator. Alert on p99 > 4× baseline.

The arithmetic that makes "0.5% for 5 minutes" the right knob: too tight (e.g., alert on a single stuck saga) and you page on every transient blip and train the team to ignore alerts; too loose (e.g., 5%) and at 5/sec you would tolerate ~750 stuck sagas before noticing — 750 customers with held stock or pending charges, which is a visible outage. 0.5% sustained for 5 minutes is roughly "more stuck sagas than retries can clear, persisting longer than a transient blip," which is the actual definition of an incident.

![A vertical stack of observability layers showing the saga instance state row, a per-step stuck-after deadline, a sweeper job that finds overdue sagas, an alert on the stuck rate, and a tracing span that correlates by saga id](/imgs/blogs/the-saga-pattern-in-practice-9.webp)

## Observability: where is the saga stuck?

The first question in every saga incident is "where is order #4471 right now?" An orchestration saga answers it with a single `SELECT * FROM order_saga WHERE order_id = '4471'` — state, direction, completed steps, last error, all in one row. That queryability is, operationally, the single best reason to choose orchestration. A choreography saga forces you to grep N services' logs and reconstruct the flow, which is exactly the 3am nightmare from the event-driven post.

Beyond the state row, instrument three things, all of which depend on a saga correlation id propagated through every step:

- **A `saga_id` on every log line, span, and event.** Carry it in the trace context so a single [distributed trace](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) shows the whole saga — reserve → charge → ship — as one connected tree. When a step is slow, the trace shows exactly which one and how slow.
- **The sweeper for overdue sagas.** The `deadline_at` index from the schema lets a background job find every saga past its step deadline. This is the mechanism that converts "a saga silently hung" into "an alert fired." Without it, a stuck saga is invisible until a customer complains.
- **A saga state dashboard.** A simple count by `state` over time (`STARTED`, `INVENTORY_RESERVED`, `PAID`, `COMPENSATING`, `COMPLETED`, `CANCELLED`, `COMPENSATION_FAILED`) tells you at a glance whether sagas are flowing or piling up in a state. A growing `COMPENSATING` count means a downstream just started failing; a growing `INVENTORY_RESERVED` count that never advances means payment is stuck.

## Stress-testing the design

A saga that works in the demo is not a saga that works in production. Stress-test the design against the three failures that actually happen.

**Stress 1: the compensation itself fails.** Payment declines, the orchestrator runs `ReleaseInventory`, and *the inventory service is down* — the compensation throws. Now you are stuck mid-undo: money is fine (never charged), but a unit is held with no one to release it. The design must handle this: (a) compensations retry hard with backoff (5+ attempts in the code above), because the inventory service is probably just briefly down; (b) if retries exhaust, transition to `COMPENSATION_FAILED` and *page a human* — do not silently give up, because the system is now genuinely inconsistent; (c) the `expires_at` on the reservation is the backstop — even if the orchestrator never recovers, the sweeper reclaims the expired hold so the stock is not lost forever. The "compensation storm" failure — a non-idempotent compensation that releases stock multiple times on retry — is prevented by the idempotent conditional `UPDATE ... WHERE status = 'PENDING'`.

**Stress 2: the orchestrator crashes mid-saga.** The orchestrator dies after charging payment but before creating the shipment. With the hand-rolled version, recovery reads every `order_saga` row not in a terminal state and resumes: it sees `state = PAID`, `direction = FORWARD`, and re-issues `CreateShipment` (idempotent, so re-issuing is safe). With Temporal, the worker crash is invisible — the engine replays the workflow from history on another worker and resumes at step 3. The non-negotiable precondition for both: **state is persisted before and after every step**, so the post-crash truth is in durable storage, not the dead process's memory. A non-durable orchestrator turns this stress into a permanent orphaned saga.

**Stress 3: two concurrent sagas race.** Covered in the worked example above — the semantic lock (`UPDATE ... WHERE available >= 1`) serializes them at the row level so only one wins the last unit. Stress it further: ten sagas race for one unit. Nine block on the row lock, evaluate `available >= 1` against `available = 0` after the winner commits, match zero rows, and fail `ReserveInventory` cleanly. Ten sagas, one shipment, zero oversell — the database's row lock scales the serialization for free, no application-level distributed lock needed. (If your inventory were sharded across nodes you would need [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) to keep one SKU's stock on one shard so the row lock still works — a real consideration at scale.)

**Stress 4: a downstream is permanently down.** If `CreateShipment` (a retriable post-pivot step) cannot succeed after exhausting retries because the shipping service is *permanently* broken, the saga has no valid terminal state — it cannot go back (past the pivot) and cannot go forward (step keeps failing). This is the one stress a saga cannot fully absorb; the answer is escalation to a dead-letter / manual-intervention queue plus an alert, and treating the shipping service's availability as a hard dependency of the post-pivot phase. This connects to broader [partial-failure and graceful-degradation handling](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation): sometimes the right degraded behavior is "accept the order, fulfill the shipment asynchronously when shipping recovers," which is exactly what forward recovery with a durable retry queue gives you.

## Case studies

**Uber and Cadence/Temporal for trip and order workflows.** Uber built Cadence (open-sourced, later forked into Temporal by its original authors) precisely because hand-rolled orchestrators for their many long-running, multi-step workflows — trips, payments, driver onboarding — were each reinventing durable state, retries, and timeouts, and each getting some of it subtly wrong. The lesson generalizes: once you have more than a couple of non-trivial sagas, the *durable execution* engine is not gold-plating, it is the thing that stops every team from shipping its own buggy orchestrator. Uber's order and fulfillment flows are textbook orchestration sagas: a coordinator drives reserve-charge-fulfill with compensations, and the engine makes the coordinator durable.

**Travel booking — the canonical multi-irreversible-step saga.** Booking a trip is the example the literature reaches for because it stresses the pattern hard: reserve a flight, reserve a hotel, reserve a car, charge the card. Each reservation is *separately* expensive to compensate (cancellation fees, inventory held by third parties you do not control), and several steps touch external systems that do not support 2PC at all. Real travel platforms run these as orchestration sagas with explicit compensations (cancel-flight, cancel-hotel) and careful pivot placement — typically the *payment* is the pivot, and the reservations before it are held as tentative/pending (semantic locks against the suppliers) so that a failure cancels the holds cleanly. The hard lesson travel teaches: when *every* step has a costly compensation, ordering and the choice of pivot are where you win or lose money, and a step whose compensation depends on a third party's SLA is a step whose saga can get genuinely stuck — which is why these systems invest heavily in the stuck-saga sweeper and manual-intervention tooling.

**A compensation-gone-wrong lesson.** The most instructive real-world failures are not the sagas that compensate, but the ones where compensation was assumed to be a true rollback and wasn't. The recurring production incident — documented across many engineering blogs and post-mortems in this space — is the *double charge that idempotency stopped at the database*: a service made its database write idempotent (the charge row had a unique key) but forgot that the *external payment call* it made before the database write was not idempotent, so a retry hit the gateway twice and charged the customer twice even though only one row was written. The fix is the two-layer idempotency from earlier — pass the idempotency key to the *processor*, not just your own table — and it is the single most common saga bug in the wild. The broader lesson: a compensation is a business action with its own side effects, fees, and customer-visibility, and treating it as a free undo is how sagas hurt customers. Design the compensation as carefully as the forward step, because at some traffic level, it *will* run.

**AWS Step Functions and the state-machine-as-config approach.** Worth naming as a third implementation point on the spectrum between hand-rolled and full Temporal: AWS Step Functions models the saga as an explicit state machine defined in JSON (the Amazon States Language), where each state is a task (an action) and you wire `Catch` handlers to route failures to compensation states. It is more declarative than a Temporal workflow (you draw the state machine rather than writing imperative control flow) and fully managed (no cluster to operate), at the cost of being tied to AWS and being clunkier for complex branching. The lesson it teaches is that the saga's *structure* — states, transitions, compensation routes — is itself a first-class artifact worth making explicit and version-controlled, rather than burying it implicitly in event subscriptions. Whether that artifact is JSON (Step Functions), code (Temporal), or a hand-rolled table-driven state machine, the senior instinct is the same: make the saga a thing you can read in one place, not a behavior you reconstruct from logs.

The thread connecting all the case studies: every team that started with choreography for a multi-step money-moving flow eventually migrated to orchestration or a workflow engine, and every team that skipped idempotency on the *external* call eventually double-charged someone. These are not exotic failure modes you might hit; they are the failure modes you *will* hit, in roughly that order, as the saga's traffic grows. Building for them up front is cheaper than the incident that teaches them to you.

## When to reach for a saga (and when not to)

The senior move is to question the saga before building it. A saga is genuine complexity — a state machine, compensations, semantic locks, idempotency, a sweeper, alerting — and you should pay that cost only when you must.

**Reach for a saga when:**

- A single business operation genuinely spans multiple services with separate databases (the [database-per-service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) reality), and you need it to converge to all-done or all-undone.
- One or more steps is an external system (a payment processor) that cannot join a distributed transaction.
- You can tolerate eventual consistency for the window the saga is in flight, and you can tolerate (and counteract) the lack of isolation.
- The operation is too important to leave to "hope each step succeeds" — you need explicit compensation.

**Skip the saga when:**

- **The operation fits inside one aggregate / one service's database.** This is the most important "skip," and the one juniors miss most. If reserving stock and recording the order can live in *one* service with *one* transaction, do that — you get real ACID and zero saga complexity. The temptation to split into more services is often premature; keeping a tightly-coupled operation inside one boundary is frequently the *senior* call, not the junior one. Drawing the boundary so that what-changes-together-stays-together is the lesson of [service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design).
- **The operation is read-only** — there is nothing to compensate, so no saga is needed; a scatter-gather query suffices.
- **You genuinely need cross-service isolation** and cannot tolerate any partial-state visibility — then a saga is the wrong tool, and you must either co-locate the data (one aggregate) or accept 2PC's blocking cost.
- **The steps cannot be ordered to make compensation possible** (two truly irreversible steps with no pivot that works) — that is the design telling you the boundaries are wrong; fix the decomposition, do not force the saga.

Plainly: the cheapest correct saga is the one you didn't need because the operation lived in one aggregate. Reach for the pattern when the operation truly crosses services and you have accepted the consistency trade — not because microservices "should" use sagas.

## Key takeaways

- **A saga replaces the impossible distributed transaction with a chain of local transactions plus compensations.** No global rollback exists; you build the reverse chain by hand. (Junior: "I'll just use a transaction." Senior: "There is no transaction across databases; here's the compensation for each step.")
- **Classify every step as compensatable, pivot, or retriable, then order them compensatable-before-pivot-before-retriable.** The order is not a free choice; it is determined by reversibility. Charge-first oversells; reserve-first does not.
- **A compensation is a semantic undo, not a rollback.** Money refunded, email apologized-for, stock released — new business facts, not time travel. Design the compensation as carefully as the forward step.
- **Default to orchestration over choreography for any non-trivial saga.** The single queryable state row that answers "where is it stuck?" is worth the central-coupling cost, every page you take.
- **Sagas give up isolation; you claw it back with semantic locks (pending states), commutative updates, and reread-by-value.** The conditional `UPDATE ... WHERE available >= 1` plus a `PENDING` status is the highest-leverage two lines in the whole pattern — it stops the oversell.
- **Every step and compensation will run more than once; make them idempotent with an idempotency key, and pass that key to external systems too.** Idempotency that stops at your database double-charges through the payment gateway.
- **Persist saga state before and after every step.** A non-durable orchestrator becomes an orphaned saga the first time it crashes mid-flight. A durable workflow engine (Temporal/Cadence) makes this automatic.
- **Reach for a workflow engine once you have more than a couple of multi-step sagas** — durable execution stops every team from reinventing buggy state-persistence plumbing.
- **Instrument the saga: a state row, per-step deadlines, a sweeper for overdue sagas, and alerts on stuck-rate (>0.5% for 5m) and compensation-failure (any).** A saga you cannot see where it's stuck is a saga you cannot operate.
- **The cheapest correct saga is no saga: if the operation fits one aggregate, keep it there.** Question the boundary before you build the state machine.

## Further reading

- [Saga pattern: distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions) — the mechanism deep-dive: the formal step taxonomy, the isolation theory, and the full countermeasure menu this post applies.
- [Event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — the coordination-style axis in full, with the event-wiring and visibility trade-offs.
- [Database-per-service: the rule that defines microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) — why you lost the cross-service transaction in the first place.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) and [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) — the two-layer idempotency every saga step needs.
- [The transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) and [the transactional outbox and reliable event publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) — how to publish a saga's events atomically with its local transaction.
- [Consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) — the consistency model a saga lives in.
- Chris Richardson, *Microservices Patterns* — Chapter 4 ("Managing transactions with sagas") is the canonical written treatment of the step taxonomy and countermeasures.
- Sam Newman, *Building Microservices* (2nd ed.) — the chapter on workflow and the distributed-transaction discussion, with Newman's pragmatic "keep it in one service if you can" stance.
- The Temporal and (legacy) Cadence documentation — the durable-execution programming model behind the workflow-engine version of the saga.

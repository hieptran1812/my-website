---
title: "The Saga Pattern: Orchestration vs Choreography for Distributed Transactions"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn how to coordinate a business transaction across services with no distributed ACID lock by chaining local transactions and compensating in reverse, and how to choose between choreography and orchestration when you wire it up."
tags:
  [
    "message-queue",
    "saga-pattern",
    "distributed-transactions",
    "choreography",
    "orchestration",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "microservices",
    "compensation",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/saga-pattern-orchestration-vs-choreography-1.webp"
---

A customer clicks "Place order." In the time it takes the spinner to stop, your platform must reserve a unit of inventory, charge a credit card, create a shipment, and tell the warehouse to pick the package. Ten years ago all four of those rows lived in one database, and a single `BEGIN` / `COMMIT` made the whole thing atomic: either all four happened or none did, and no other transaction ever saw a half-placed order. Then you split the monolith into an inventory service, a payment service, a shipping service, and an order service, each with its own database, and that single `COMMIT` vanished. There is no transaction that spans four databases owned by four services on four different hosts. The guarantee you took for granted — all-or-nothing across the whole order — is simply gone, and nothing in the microservices toolbox hands it back to you for free.

This is the problem the **saga pattern** exists to solve. A saga is not a distributed transaction; it is a sequence of *local* transactions, each one committing in a single service's own database, stitched together by messages. Step one reserves inventory and commits; that commit emits a message that triggers step two, which charges the card and commits; that commit triggers step three, which ships. If everything succeeds, the business outcome is the same as the old atomic transaction. But when a step fails after earlier steps already committed — and they *have* committed, you cannot take that back — the saga runs **compensating transactions** in reverse: it refunds the charge, then releases the inventory reservation. Compensation is not a rollback. A rollback un-writes uncommitted changes; a compensation issues a *new* committed transaction that semantically undoes a previous one. The figure below shows the spine of the whole idea: a chain of local transactions, each one emitting the message that drives the next.

![A saga drawn as a chain of three local transactions where each service commits on its own and emits the message that triggers the next service in the sequence](/imgs/blogs/saga-pattern-orchestration-vs-choreography-1.webp)

There are two ways to wire the steps together, and the choice shapes your whole architecture. In **choreography**, there is no coordinator: each service subscribes to the events of the previous step and decides on its own what to do next, so the flow is decentralized and emergent. In **orchestration**, a central saga orchestrator — usually a persisted state machine — issues a command to each service in turn and records progress, so the flow is explicit and observable but lives in one component. Neither is universally right. By the end of this post you will be able to design an order saga end to end, write both the choreographed and the orchestrated version, reason precisely about what other transactions can see while a saga is mid-flight (sagas are *not* isolated), pick the right coordination style for a given flow, and size the idempotency and ordering machinery that any saga silently depends on. This post leans on three siblings in the series: [the transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) for how a step reliably publishes its event, [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) for why every saga step and compensation must be safe to run twice, and [event-driven architecture](/blog/software-development/message-queue/event-driven-architecture-events-commands-documents) for the events-versus-commands distinction that separates choreography from orchestration.

## 1. Why distributed ACID transactions do not scale

Start with the thing you are giving up, because the saga only makes sense once you accept that the alternative is genuinely unavailable. An ACID transaction across multiple services would need two properties at once: **atomicity** (all participants commit or all abort) and **isolation** (no other transaction sees a partial result). To get both across independent databases you need a coordinator and a protocol, and the textbook protocol is **two-phase commit (2PC)**, usually over the XA standard.

Two-phase commit works like this. A transaction coordinator contacts every participant and says "prepare": do everything except the final commit, write enough to durable storage that you *promise* you can commit if I later ask you to, and hold whatever locks are needed to keep that promise. Each participant does the work, takes the locks, and votes yes or no. Only if every participant votes yes does the coordinator send "commit" to all of them; if anyone votes no, it sends "abort" to all. On paper this gives you atomicity and isolation across four databases.

In practice 2PC is the wrong tool for a chain of microservices, and the reasons are not academic — they are operational and they bite in production.

The first problem is **locks held across the network**. In the prepare phase, every participant has taken locks on the rows it touched and must hold them until the coordinator decides. That decision involves at least one more network round trip to every participant. So the duration of the lock is not the duration of one local query; it is the duration of the slowest participant plus the coordinator's round trips plus any queueing. A 2-millisecond local update becomes a lock held for tens of milliseconds across a fleet, and that lock blocks every other transaction that wants the same rows. Under load this collapses throughput, because the hot rows — the popular product's inventory count, the shared counter — are exactly the rows everyone contends on, and now they are locked for an order of magnitude longer.

The second problem is the **blocking failure mode**. Suppose every participant has voted yes and is sitting in the prepared state, holding its locks, waiting for the commit message — and the coordinator crashes. The participants cannot safely commit (maybe someone voted no and they did not hear) and cannot safely abort (maybe everyone voted yes and the coordinator already told someone to commit). They are stuck holding locks until the coordinator recovers. This is not a rare edge case; coordinator failure during the commit window is the canonical 2PC disaster, and it can wedge a whole set of services. The figure below puts 2PC and the saga side by side: the blocking, lock-holding coordinator on the left versus the local-commit-plus-compensate model on the right.

![A side-by-side comparison of two-phase commit holding locks behind a single coordinator that blocks if it dies, against a saga that commits each step locally and compensates failures](/imgs/blogs/saga-pattern-orchestration-vs-choreography-2.webp)

The third problem is **availability coupling**. With 2PC the transaction can only commit if *every* participant is reachable and healthy at commit time. The availability of the whole operation is the product of the availabilities of all participants. Four services at 99.9% each give you roughly 99.6% combined just from the multiplication, before you count the coordinator. You took four services that could each degrade independently and chained their uptimes together, which is the opposite of why you split the monolith. This is the distributed-systems tradeoff space the [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) formalizes: insisting on strong cross-service consistency costs you availability and latency, and most order flows would rather stay available.

The fourth problem is **brokers and many services do not speak XA well**. Kafka does not participate in XA transactions with your database; neither do most modern data stores you would actually use across a microservices fleet. You would be building distributed locking around components that were never designed to be locked together. Even where XA is technically available, the operational burden — heuristic decisions, in-doubt transaction cleanup, coordinator high availability — is brutal.

So the industry made a deliberate trade. Instead of atomicity *and* isolation across services, the saga keeps **atomicity** at the granularity of each local transaction (each step is all-or-nothing inside one database) and gives up **isolation** across the saga as a whole. The saga as a whole becomes *eventually consistent*: it reaches a consistent end state (fully committed or fully compensated), but in between, other transactions can observe partial progress. The rest of this post is largely about living with that one concession safely.

### The availability arithmetic, with numbers

It is worth seeing the availability tradeoff as a number rather than a slogan, because the number is what convinces a skeptical architect. With 2PC, the operation commits only if every participant *and* the coordinator are available at commit time. Availability multiplies. Take four services and a coordinator at 99.95% each — a respectable single-component SLO. The combined availability for a 2PC commit is roughly 0.9995 raised to the fifth power, which is about 99.75%. That sounds fine until you translate it: 99.75% is about 22 hours of unavailability per year for the *combined* operation, even though each component was down for only about 4.4 hours. You manufactured 5x the downtime by chaining the components synchronously.

A saga inverts this. Each step only needs its *own* service available at the moment that step runs, and the steps are spread over time and decoupled by the broker. If the payment service is briefly down when the `InventoryReserved` event arrives, the event waits in the queue (or the orchestrator retries the command after a backoff), and the step runs when payment recovers — the saga does not abort, it *pauses*. The operation's availability is no longer the product of all participants at one instant; it degrades to "each step eventually finds its service up," which the broker's durability and retry buy you almost for free. This is the [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) tradeoff resolved in favor of availability and latency: you accept that the order is not instantaneously consistent across all four services so that a transient outage in one of them stalls a saga for seconds instead of failing the whole operation. For a business flow, "the order takes two extra seconds because payment was restarting" is almost always better than "the order failed because payment was restarting."

> A saga trades the isolation of a distributed ACID transaction for availability and low latency. You keep all-or-nothing inside each service and give up "no one sees the in-between" across services. Everything hard about sagas flows from that single trade.

## 2. The saga: local transactions plus compensation

A saga is defined by two sequences that mirror each other. Going forward, there is a sequence of local transactions T1, T2, ... Tn, each in one service, each committing independently, each (except the last) emitting a message that triggers the next. Going backward, there is a sequence of compensating transactions C1, C2, ... Cn-1, where Ci semantically undoes the effect of Ti. If the saga gets to Tn, it commits and you are done. If it fails at step Ti, it runs the compensations for the steps that already committed — Ci-1, then Ci-2, and so on down to C1 — in reverse order.

Concretely, for the order saga, the forward transactions are:

- **T1** — Inventory service: decrement available stock for the SKU by one, reserve it for this order. Commit.
- **T2** — Payment service: charge the customer's card for the order total. Commit.
- **T3** — Shipping service: create a shipment, hand it to the warehouse pick queue. Commit.

And the compensations are:

- **C1** — Inventory service: release the reservation, increment available stock back by one.
- **C2** — Payment service: refund the charge (or void the authorization if not yet captured).
- **C3** — usually unnecessary, because the last step is often the *pivot* (more on that in section 7). If shipping is the final step and it succeeds, the saga is done; if it fails, there is nothing after it to undo, so you compensate T2 and T1 only.

The key mental shift from a database transaction is that **each Ti has already committed and is visible to the world the instant it finishes**. When T1 commits, the inventory is genuinely reserved in the inventory database, and a query against that database right now will show one fewer unit available. There is no "prepared but not committed" limbo. That is what makes the local commit cheap and non-blocking — and it is exactly what makes compensation necessary, because you cannot un-commit T1; you can only run a new transaction C1 that puts the world back into an equivalent state.

Each forward step has three responsibilities, and getting all three right is the whole craft:

1. **Do the local work and commit it atomically** inside one database transaction.
2. **Reliably emit the message** that drives the next step — and "reliably" is doing heavy lifting here. If the service commits T1 but crashes before publishing the "inventory reserved" event, the saga stalls forever. This is the dual-write problem, and the answer is the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing): write the outgoing event into an outbox table *in the same local transaction* as the business change, and let a relay publish it afterward, so the event is as durable as the commit.
3. **Be idempotent**, because the message that triggers it may be delivered more than once.

Here is a forward step written with the outbox baked in, so the commit and the event are one atomic local transaction:

```python
# Payment service: T2 — charge the card and emit the next event atomically.
def handle_inventory_reserved(event):
    order_id = event["order_id"]
    # Idempotency: if we already processed this saga step, do nothing.
    if payments_repo.exists(order_id):
        return  # duplicate delivery — safe no-op

    with db.transaction() as tx:                       # one local ACID txn
        charge = gateway.charge(
            card_token=event["card_token"],
            amount_cents=event["total_cents"],
            idempotency_key=f"charge:{order_id}",       # gateway-level dedup
        )
        payments_repo.insert(tx, order_id, charge.id, status="captured")
        # Emit the next saga event INTO THE SAME TRANSACTION (outbox row):
        outbox.insert(tx, topic="payments", key=order_id, type="PaymentCaptured",
                      payload={"order_id": order_id, "charge_id": charge.id})
    # tx committed → both the charge record and the outbox event are durable.
```

Notice three things. The `payments_repo.exists` check at the top makes the step idempotent. The gateway call carries its own `idempotency_key`, so even the external charge is deduplicated by the payment processor. And the outbox insert is *inside* the same transaction, so the "PaymentCaptured" event is guaranteed to be published if and only if the charge was recorded. That last point is non-negotiable; without it, a crash between the commit and the publish either loses the event (the saga stalls) or, if you publish first, creates a phantom event (the saga advances on a charge that did not happen).

## 3. Compensating transactions are semantic, not ACID rollback

This is the section that separates engineers who have read about sagas from engineers who have run them. A compensation is **semantic**, not transactional, and the difference is not pedantic — it determines what you can promise the business.

An ACID rollback restores the exact prior state. If a transaction updated a row from 5 to 4 and then rolled back, the row is 5 again, byte for byte, and crucially *no other transaction ever saw the 4* because isolation hid it. A compensation cannot do either of those things. It runs after Ti already committed and became visible, so the world already reacted to Ti. C1 cannot restore the exact prior state; it can only move the world to a state that is *business-equivalent* to "T1 never happened."

Walk through what that means for the order saga. When you compensate the charge, you do not delete the charge — you issue a **refund**. The customer's statement now shows a charge and a matching refund. The money is back, the business outcome is equivalent to "never charged," but the *history* is different, and that difference is permanent and sometimes visible (the customer sees both lines; your finance team reconciles both). When you compensate the inventory reservation, you increment stock back — but in the interval between reserving and releasing, another customer may have seen "out of stock" and left. You cannot un-show them the out-of-stock page. The compensation restores the *count*, not the *counterfactual*.

Some effects **cannot be compensated at all**, and recognizing them is a design constraint, not a footnote. If a saga step sends a physical package, you cannot un-send it; the best you can do is initiate a return, which is a different, slower, messier process. If a step sends an email saying "your order shipped," you cannot un-send the email; you can only send a correction. Steps with irreversible side effects must be ordered carefully so they come *last*, after the point where the saga is committed to succeeding (the pivot — section 7). You never want to ship the package and *then* discover the card was declined.

This gives three categories of step, and every step you design falls into one:

| Step type | Property | Example | Design rule |
| --- | --- | --- | --- |
| Compensatable | Has a semantic inverse | Reserve inventory → release; charge → refund | Put these before the pivot |
| Pivot | The point of no return; once it commits the saga must complete | Capture payment, or the last reversible step | Exactly one per saga, placed deliberately |
| Retriable | Runs only after the pivot; must eventually succeed | Ship, send confirmation email, update analytics | Put these after the pivot; retry until they succeed |

The discipline is: arrange your steps so that all the **compensatable** ones come first, then the **pivot**, then the **retriable** ones. Before the pivot, any failure unwinds cleanly by compensating backward. After the pivot, you have decided to commit, so any failure in a retriable step is handled by *retrying forward* — never by compensating, because the steps after the pivot are exactly the ones you cannot undo. A failed shipment is retried, not refunded.

Two more properties make compensations harder than they look. First, **compensations must themselves be idempotent**, because the compensation message can be delivered more than once. Refunding twice would return the money twice. Second, compensations must handle the case where the forward step **partially happened**. If T2 timed out — you sent the charge request, never got a response, and do not know whether the card was charged — then C2 must be written to refund *if and only if* a charge exists, which it discovers by querying the gateway with the idempotency key rather than blindly issuing a refund. Idempotent, query-then-act compensations are the only safe kind.

```python
# Payment service: C2 — compensate the charge. Idempotent and query-first.
def compensate_charge(order_id):
    record = payments_repo.find(order_id)
    if record is None:
        return  # T2 never committed here; nothing to undo (or duplicate compensate)
    if record.status == "refunded":
        return  # already compensated — idempotent no-op
    # Ask the gateway with the SAME idempotency key T2 used, to learn the truth:
    charge = gateway.lookup(idempotency_key=f"charge:{order_id}")
    if charge and charge.status == "captured":
        gateway.refund(charge.id, idempotency_key=f"refund:{order_id}")
    payments_repo.mark_refunded(order_id)   # commit the new status locally
```

### Backward recovery versus forward recovery

There are two distinct strategies for handling a failed step, and conflating them is a common design mistake. **Backward recovery** is what we have been describing: a step failed, so unwind the committed steps by compensating in reverse, and abort the saga. **Forward recovery** is the opposite: a step failed, but the saga *must* complete (you are past the pivot, or the business cannot tolerate aborting), so you keep retrying the failed step — possibly with intervention — until it succeeds. The same saga uses both: backward recovery before the pivot, forward recovery after it.

The reason this matters is that the two strategies impose opposite requirements on a step. A step you might *backward-recover* must be **compensatable** — it needs a semantic inverse. A step you will *forward-recover* must be **retriable to eventual success** — it must not be the kind of operation that can fail permanently with no path forward. Placing a step on the wrong side of the pivot is a design bug. If you put a non-compensatable step (ship the package) before the pivot, a later failure cannot unwind it. If you put a step that can fail permanently (charge a card that might be declined for good) after the pivot, forward recovery loops forever on something that will never succeed. The pivot is precisely the dividing line: everything before it must be compensatable, everything after it must be forward-recoverable, and the pivot itself is the single step that converts uncertainty into commitment.

A subtle corollary: **compensations themselves should not be allowed to fail permanently**. A compensation runs during backward recovery, and if a compensation cannot complete, the saga is stuck half-unwound — money taken but not refunded, inventory reserved but not released. Compensations therefore must be designed as retriable forward-recovery operations: keep trying the refund until the gateway accepts it, alert a human if it exhausts retries, but never give up silently. The asymmetry is worth internalizing: forward steps before the pivot can fail (you compensate), but compensations cannot be allowed to fail (you retry them to death). A saga that compensates is only as reliable as its least reliable compensation.

## 4. Choreography: services react to events

The first way to wire the steps together uses no coordinator at all. In **choreography**, each service subscribes to the events that the previous steps publish and, when it sees an event it cares about, runs its local transaction and publishes its own event. The saga's flow is not written down anywhere as a single artifact; it *emerges* from the subscriptions. Think of it as a relay race where each runner knows only the person they take the baton from and the person they hand it to — no one is watching the whole track.

![A choreographed saga where order, inventory, payment, and shipping services each subscribe to events on a shared bus and emit their own next event with no central coordinator](/imgs/blogs/saga-pattern-orchestration-vs-choreography-3.webp)

For the order saga, choreography looks like this. The order service creates the order in a pending state, commits, and publishes `OrderCreated`. The inventory service subscribes to `OrderCreated`; on receipt it reserves stock, commits, and publishes `InventoryReserved`. The payment service subscribes to `InventoryReserved`; it charges the card, commits, and publishes `PaymentCaptured`. The shipping service subscribes to `PaymentCaptured`; it creates the shipment, commits, and publishes `OrderShipped`. The order service subscribes to `OrderShipped` and flips the order from pending to confirmed. Each event is a *fact about the past* — "this happened" — not a command telling someone what to do, and that framing is exactly the events-versus-commands distinction from [event-driven architecture](/blog/software-development/message-queue/event-driven-architecture-events-commands-documents): choreography is built from events, orchestration from commands.

Compensation in choreography also propagates through events. If the payment service fails to charge, it publishes a `PaymentFailed` event. The inventory service subscribes to `PaymentFailed` and runs its compensation — release the reservation — then publishes `InventoryReleased`. The order service subscribes to either `PaymentFailed` or `InventoryReleased` and marks the order as canceled. The failure path is just another set of events flowing backward through the same decentralized web.

```python
# Inventory service in a choreographed saga — two subscriptions.
@subscribe("orders.OrderCreated")
def on_order_created(evt):
    if inventory_repo.reservation_exists(evt["order_id"]):
        return                                   # idempotent: already reserved
    with db.transaction() as tx:
        ok = inventory_repo.try_reserve(tx, evt["sku"], qty=1, order_id=evt["order_id"])
        if ok:
            outbox.insert(tx, "inventory", evt["order_id"], "InventoryReserved",
                          {"order_id": evt["order_id"], "sku": evt["sku"]})
        else:
            outbox.insert(tx, "inventory", evt["order_id"], "InventoryRejected",
                          {"order_id": evt["order_id"], "reason": "out_of_stock"})

@subscribe("payments.PaymentFailed")             # compensation trigger
def on_payment_failed(evt):
    # Release the reservation this saga made earlier. Idempotent compensation.
    with db.transaction() as tx:
        released = inventory_repo.release_if_held(tx, evt["order_id"])
        if released:
            outbox.insert(tx, "inventory", evt["order_id"], "InventoryReleased",
                          {"order_id": evt["order_id"]})
```

The appeal of choreography is real. There is **no central component** to build, deploy, scale, or keep highly available — the saga has no single point of failure that lives outside the participating services. Services are **loosely coupled**: the payment service does not know the shipping service exists; it only knows the events it consumes and the events it produces. Adding a new *consumer* of an existing event is trivial — a fraud-check service that also wants to react to `InventoryReserved` just subscribes, and no existing service changes. For simple, short, mostly-linear sagas with two or three steps, choreography is often the right and lighter choice.

The cost is equally real and shows up as the system grows. The flow is **emergent and invisible**: no single file or service describes "this is what placing an order does." To understand the saga you must read every subscription across every service and reconstruct the graph in your head. When something stalls, *where* did it stall? You have to trace events across service boundaries to find the step that did not fire its next event. Worse, choreography invites **cyclic dependencies** and event ping-pong: service A's event triggers B, whose event triggers C, whose event triggers A, and now a small change to A's events can loop. And the failure topology gets dense — every service needs to know which upstream failures it must compensate for, which spreads compensation logic across the whole fleet. We will quantify these tradeoffs against orchestration in section 6.

There is also a subtle correctness trap specific to choreographed compensation: a service must be able to tell a *forward* event from a *compensating* one and route to the right handler, and it must compensate only the steps *it* performed for *this* saga. In the inventory code above, `on_payment_failed` calls `release_if_held`, which is conditional precisely because the inventory step may not have run for this order (the saga could have failed before reaching inventory), and a blind "release inventory" on a reservation that does not exist would either error or, worse, release stock that belongs to a *different* order. Every choreographed compensation handler must therefore be written defensively: identify the saga by a key, check whether this service has anything to undo, and undo only that. The decentralization that makes choreography lightweight is the same property that scatters this defensive logic across every service, where it must be kept consistent by convention rather than by a single state machine — which is exactly the maintenance burden that pushes growing systems toward orchestration.

## 5. Orchestration: a central saga coordinator

The second way to wire the steps puts a single component in charge. A **saga orchestrator** is a service (often a persisted state machine) that knows the entire flow. It issues a **command** to each service — "reserve inventory for order 123" — waits for the reply, records the result in its own durable saga state, and then decides the next command. The services no longer chain to each other; each one only talks to the orchestrator. The flow lives in one place, written down explicitly as states and transitions.

![An orchestrated saga where one orchestrator issues a command to the inventory, payment, and shipping services in turn and persists the saga state to track progress](/imgs/blogs/saga-pattern-orchestration-vs-choreography-4.webp)

The orchestrator is a state machine, and writing it out makes the explicitness obvious. The saga has states like `STARTED`, `INVENTORY_RESERVED`, `PAYMENT_CAPTURED`, `SHIPPED`, `COMPLETED`, plus failure states `COMPENSATING`, `ABORTED`. Each state defines what command to send next and what to do on success or failure. Crucially, the orchestrator **persists its state after every transition**, so if the orchestrator process crashes, it recovers by reading the saga's last recorded state and resuming from exactly there — the saga survives a coordinator crash, unlike 2PC.

```python
# Saga orchestrator: a persisted state machine for the order saga.
class OrderSaga:
    def on_command_reply(self, saga_id, reply):
        saga = saga_store.load(saga_id)          # durable saga state
        state = saga.state

        if state == "STARTED" and reply.type == "InventoryReserved":
            saga.advance("INVENTORY_RESERVED")
            saga_store.save(saga)                 # persist BEFORE next command
            self.send("payment", ChargeCard(saga_id, saga.total_cents))

        elif state == "INVENTORY_RESERVED" and reply.type == "PaymentCaptured":
            saga.advance("PAYMENT_CAPTURED")
            saga_store.save(saga)
            self.send("shipping", CreateShipment(saga_id, saga.address))

        elif state == "PAYMENT_CAPTURED" and reply.type == "ShipmentCreated":
            saga.advance("COMPLETED")
            saga_store.save(saga)

        # ---- failure → compensate backward, driven centrally ----
        elif reply.type in ("PaymentFailed", "OutOfStock", "ShipmentFailed"):
            saga.advance("COMPENSATING")
            saga_store.save(saga)
            self.run_compensations(saga)          # emit RefundCharge, ReleaseStock...

    def run_compensations(self, saga):
        if saga.reached("PAYMENT_CAPTURED"):
            self.send("payment", RefundCharge(saga.id))
        if saga.reached("INVENTORY_RESERVED"):
            self.send("inventory", ReleaseStock(saga.id))
        saga_store.save(saga.advance("ABORTED"))
```

The orchestrator can be hand-rolled like this, or built on a durable-execution framework — Temporal, AWS Step Functions, Camunda/Zeebe, Netflix Conductor — that handles the state persistence, retries, and timeouts for you. Those frameworks are popular precisely because the orchestrator's hard parts (durable state, exactly-once-ish command dispatch, timeout handling) are tedious and easy to get wrong by hand.

The advantages mirror choreography's weaknesses. The flow is **explicit**: one artifact, the state machine, *is* the saga; you can read it, draw it, and reason about it without spelunking through five services' subscriptions. It is **observable**: the orchestrator's saga state table tells you instantly where every in-flight saga is — `47 sagas stuck in PAYMENT_CAPTURED`, and you know payment-to-shipping is the problem. **Compensation logic is centralized**: the orchestrator decides what to compensate and in what order, so the services do not each need to know the failure topology — they just expose forward commands and compensation commands and obey. And **complex flows are tractable**: branches ("if order total over a threshold, require manual review"), parallel steps ("reserve inventory and run fraud check at the same time, wait for both"), and timeouts ("if shipping does not confirm in 24 hours, escalate") are natural to express in a state machine and awful to express as emergent events.

The cost is the central component itself. The orchestrator is a service you must build, deploy, scale, monitor, and make highly available; if it is down, no saga can make progress (mitigated by persisting state and recovering, but it is still a dependency every saga funnels through). It can become a **coupling magnet** and a god-service if you let business logic that belongs in the domain services leak into the orchestrator — the discipline is that the orchestrator owns *sequencing and compensation policy*, while the services own *the actual work*. Done well, the orchestrator is thin: it knows the order of steps and the compensation rules, nothing more.

### Timeouts and the orchestrator's recovery model

The orchestrator earns its keep on the two failure modes that choreography handles badly: **steps that never reply** and **its own crash**. Both are about durable state plus time.

Consider a step that never replies. The orchestrator sends `ChargeCard` to the payment service and waits. The reply never comes — maybe the payment service crashed after charging but before publishing its reply, maybe the reply was lost. A choreographed saga has no natural owner of this timeout; the chain simply stops, and nothing notices until a human does. The orchestrator, by contrast, sets a **timer** when it dispatches each command and persists the deadline alongside the saga state. When the deadline fires with no reply, the orchestrator has a policy: re-send the command (safe because the step is idempotent — a redelivered `ChargeCard` finds the existing charge and replies with the prior result), or, after N retries, declare the step failed and begin compensation. This is exactly why durable-execution frameworks exist: Temporal and Step Functions model each step as an activity with a timeout and a retry policy, and they persist the timer so it survives a worker restart. Hand-rolling it means a `deadline` column on the saga row and a sweeper job that periodically scans for sagas whose deadline has passed and nudges them.

Now consider the orchestrator's own crash. Because it persists state *before* dispatching each command, recovery is mechanical: on restart it scans the saga store for non-terminal sagas, and for each one it re-derives what to do from the recorded state. A saga recorded as `INVENTORY_RESERVED` with a pending `ChargeCard` command simply has that command re-sent; idempotency on the payment side makes the re-send harmless. A saga recorded as `COMPENSATING` resumes its compensations from where it left off. The critical invariant is **persist-then-act**: the orchestrator must durably record its intent *before* it sends the command that acts on that intent, never after. If it sends the command and then crashes before recording it, on recovery it cannot know the command was sent and may skip it; if it records intent first, the worst case is sending the command twice, which idempotency absorbs. This persist-then-act ordering is the same discipline as the [transactional outbox](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing): make the durable record and the side effect agree by ordering them so a crash only ever causes a safe duplicate, never a silent loss.

The payoff is the property 2PC could never give you: a saga that is mid-flight when the coordinator dies does not wedge anything. No locks are held — every committed step already released its local locks at commit time — so other transactions proceed unaffected, and the saga resumes when the orchestrator comes back. The orchestrator is a *bookkeeper of progress*, not a *holder of locks*, and that single difference is why it survives crashes gracefully where a 2PC coordinator turns a crash into a fleet-wide stall.

## 6. Choreography vs orchestration tradeoffs

Now put them head to head. The decision is not religious; it is a function of how many steps the saga has, how complex its branching is, and how badly you need to see what is happening. The matrix below scores the two styles on the four dimensions that actually drive the choice.

![A decision matrix scoring orchestration against choreography on coupling, visibility, added complexity, and failure handling](/imgs/blogs/saga-pattern-orchestration-vs-choreography-6.webp)

Let me make each row concrete rather than hand-wavy.

**Coupling.** Choreography couples services to *events*, not to each other — no service holds a reference to another service. That is genuinely looser. Orchestration couples every service to the *orchestrator*: the orchestrator must know all of them. But there is a subtlety people miss. Choreography's loose coupling at the *interface* level hides a tight coupling at the *flow* level: the order of steps is encoded implicitly in who-subscribes-to-what, smeared across services, so changing the order means changing several services in lockstep. Orchestration centralizes that flow coupling into one place where it is at least visible and changeable in one spot.

**Visibility.** This is orchestration's biggest win and it is not close. With an orchestrator you have a saga state store: one query tells you how many sagas are in each state, which are stuck, and how long they have been stuck. With choreography there is no such store unless you build a separate saga-tracking service or lean entirely on distributed tracing to stitch the event chain together after the fact. When a CEO asks "why are 3,000 orders stuck," the orchestration answer is a SQL query and the choreography answer is a forensic investigation.

**Added complexity.** Choreography adds *no new component* — that is its complexity win. But it adds *distributed* complexity: the saga logic is spread across N services, and understanding it requires holding all N in your head. Orchestration adds *one component* but concentrates the complexity there, where it is contained and inspectable. The rule of thumb: for 2–3 step linear sagas, choreography's "no new component" wins; for 4+ steps or any branching, orchestration's "contained complexity" wins, because emergent flows past three steps become genuinely hard to reason about.

**Failure handling.** This is where the difference is sharpest. In choreography, every service must independently know which upstream failures require it to compensate, so compensation logic is distributed and each new failure mode may touch several services. In orchestration, the orchestrator owns the compensation policy centrally: it knows exactly which steps committed and runs their compensations in reverse, and adding a new failure mode is a change in one place.

| Dimension | Choreography | Orchestration |
| --- | --- | --- |
| New component to operate | None | The orchestrator (build, deploy, HA) |
| Where flow logic lives | Smeared across N services | One state machine |
| Observability of in-flight sagas | Needs tracing or a side service | A query on the saga state store |
| Adding a new event *consumer* | Trivial (just subscribe) | Trivial (orchestrator may not even change) |
| Adding a new *step* in the middle | Touch several services | Touch the orchestrator |
| Compensation logic | Distributed across services | Centralized in the orchestrator |
| Risk of cyclic event dependencies | High | Low (orchestrator imposes a DAG) |
| Best fit | Short, linear, 2–3 steps | Long, branching, needs visibility |

A pragmatic real-world answer many teams land on is **hybrid**: orchestrate the *core* business saga (the order flow) where visibility and complex failure handling matter, and let *peripheral* reactions happen by choreography (analytics, recommendations, notifications subscribe to the events the orchestrated saga emits, and the orchestrator neither knows nor cares). You get the orchestrator's observability for the part that must not break and choreography's looseness for the parts that are fire-and-forget.

#### Worked example: adding a "fraud check" step in each style

A concrete change makes the tradeoff vivid. Product wants a new step: after the card is charged but before shipping, run a fraud check; if it flags the order, hold it for manual review instead of shipping.

In **choreography**, this step inserts into the event chain between `PaymentCaptured` and shipping. You must: (1) build the fraud service to subscribe to `PaymentCaptured`, run its check, and publish `FraudCleared` or `FraudFlagged`; (2) *change the shipping service* so it subscribes to `FraudCleared` instead of `PaymentCaptured` — because shipping must no longer fire directly off the charge; (3) add a handler somewhere for `FraudFlagged` to move the order to manual-review state; (4) make sure nothing else was relying on shipping reacting to `PaymentCaptured`. You touched the shipping service (which has nothing to do with fraud) because the chain's wiring lives in subscriptions. That is two services modified plus one new service, and a subtle re-wiring that is easy to get wrong and hard to test, because the "old" subscription must be removed atomically with the new one or you get double-shipping or no-shipping during the rollout.

In **orchestration**, you: (1) build the fraud service exposing a `RunFraudCheck` command and a reply; (2) add a `FRAUD_CHECK` state to the orchestrator between `PAYMENT_CAPTURED` and `SHIPPED`, with a branch — on `FraudCleared` go to shipping, on `FraudFlagged` go to a `MANUAL_REVIEW` state. The shipping service is *not touched*; it still exposes the same `CreateShipment` command and does not know a fraud step now precedes it. You changed one new service plus the orchestrator's state machine, in one place, with a clear branch you can read and test. This is the canonical reason teams pick orchestration as sagas grow: inserting and reordering steps is a local edit to one state machine, not a cross-service re-wiring exercise. The cost you pay for that, every day, is operating the orchestrator.

## 7. Sagas are not isolated: semantic locks and pivots

Here is the sharp edge that the marketing slides skip. A saga gives you atomicity (each step) and durability (each commit), but it explicitly **does not give you isolation**. While a saga is mid-flight — after T1 has committed but before the saga reaches its end — other transactions can see the partial state. This is not a bug; it is the price of giving up 2PC, and you must design for it consciously.

Concretely, after T1 reserves inventory and before T2 charges the card, the inventory database shows one fewer unit available. Any other transaction — another customer's order saga, an analytics query, an admin dashboard — sees that reservation. If T2 then fails and C1 releases the inventory, the unit reappears. In the window between, the system showed a reservation for an order that will never be placed. Worse anomalies are possible. The classic three are **dirty reads** (saga B reads data that saga A wrote and will later compensate away), **lost updates** (two sagas read the same value and both write, one clobbering the other), and **non-repeatable reads** (a saga reads a value, another saga changes it, the first saga reads again and gets a different answer mid-flight). These are exactly the isolation anomalies that database isolation levels exist to prevent — and a saga, by construction, runs at an isolation level weaker than read-committed across the whole flow. The deeper theory of what "consistent" even means across services is in [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual); a saga lands firmly on the eventual end.

You manage this with **countermeasures**, and the saga literature (the Garcia-Molina and Salem paper, and Chris Richardson's modern treatment) names them. Three are load-bearing.

**Semantic lock.** Mark the records a saga touches with a flag — a pending state — that other transactions can see and respect. When T1 reserves inventory, it does not just decrement a count; it writes a *reservation* record tied to the order, with a status. Other readers that understand the protocol treat reserved-but-not-confirmed stock as "claimed, outcome pending," not as "sold." When the saga completes, the lock is released (reservation confirmed) or compensated (reservation removed). The semantic lock is application-level — it is not a database row lock held across the network like 2PC; it is a *data value* that means "in progress," and other transactions choose how to interpret it. The cost is that you must teach every reader of that data what the pending state means, and you must handle two sagas contending for the same semantic lock (the second one fails fast or waits).

**Pivot transaction.** Designate one step as the **pivot**: the point of no return. Everything before the pivot is compensatable; the pivot itself either commits (and the saga is now guaranteed to complete) or aborts (and the saga is guaranteed to compensate); everything after the pivot is retriable and *will* eventually succeed. The pivot turns an uncertain saga into a certain one at a well-defined moment. In the order saga, capturing the payment is a natural pivot: before it, you can still back out cleanly (release inventory); at capture, you commit to the order; after it, shipping and confirmation are retriable steps that you keep retrying until they succeed, never compensating. The taxonomy figure below places forward, compensating, and pivot transactions in one hierarchy.

![A taxonomy tree of saga transactions splitting into forward, compensating, and pivot transactions with the pivot marked as the point of no return](/imgs/blogs/saga-pattern-orchestration-vs-choreography-7.webp)

**Commutative updates and reread.** Where you can, make updates *commutative* so order does not matter — incrementing and decrementing a stock count commute, so two concurrent reservations and releases land on the right total regardless of interleaving (as long as you never go negative, which the semantic lock guards). And where a saga must act on a value it read earlier, **reread and revalidate** at the moment of action rather than trusting the stale read, so a value another saga changed mid-flight does not cause a lost update. This is the same defensive discipline as optimistic concurrency control: read a version, and at write time fail if the version moved.

The practical upshot: a saga's lack of isolation is *manageable but not free*. You design the data so partial states are recognizable (semantic locks), you design the flow so there is a clear commit point (pivot), and you design the updates so concurrent sagas do not corrupt each other (commutativity and reread). Skip these and you will ship a saga that works in the demo and produces oversold inventory and double-shipped orders under real concurrency.

> Sagas are not isolated. Other transactions see your half-finished work. Semantic locks make that work *recognizable* as in-progress, the pivot makes the outcome *certain* at a defined moment, and commutative-plus-reread updates keep concurrent sagas from corrupting each other.

## 8. Idempotency and ordering requirements

Every saga silently depends on two guarantees from the messaging layer, and if you do not provide them, the saga is quietly broken in ways that only show up under load. The two are **idempotency** and **ordering**, and they are worth stating as hard requirements because skipping them is the most common way sagas fail in production.

**Idempotency is mandatory because delivery is at-least-once.** Real message brokers deliver at least once: a consumer may receive the same message twice because an ack was lost, a rebalance reprocessed an offset, or a relay republished an outbox row. That means every saga step *and every compensation* will, eventually, be invoked more than once for the same saga instance. If "charge the card" runs twice, you double-charge. If "release inventory" runs twice, you over-credit stock. The fix is the subject of [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe): give each step a natural idempotency key — the saga id plus the step name — and record, in the same local transaction as the work, that this step ran. On a duplicate, the recorded marker makes the step a no-op. The code in sections 2 and 3 already did this with the `exists` checks and the gateway idempotency keys; that was not optional defensive coding, it is the load-bearing safety property.

There is a subtle second-order requirement: the idempotency record and the business effect must commit **together**. If you charge the card and then, in a separate transaction, record "charged," a crash in between lets a redelivery charge again. The dedup marker must be written in the same atomic local transaction as the effect, which is exactly why the outbox and the idempotency table live in the service's own database next to the business rows.

**Ordering matters within a saga instance, and per-key ordering is what you need.** Consider the failure path: the saga emits `PaymentCaptured` and then, milliseconds later, `PaymentFailed` is impossible for the same instance — but `InventoryReserved` followed by a compensating `InventoryReleased` for the *same order* must be processed in that order, or the consumer applies the release before the reserve and the reservation leaks. You do not need *global* ordering across all sagas; you need ordering *within each saga instance*, which maps cleanly onto **per-key ordering** in the broker. Partition by saga id (or order id), and all events for one saga land in one partition and are delivered in order. This is the standard per-key ordering guarantee: one partition per key, one consumer per partition, in-order delivery. It is also why partitioning by a *random* key, or by anything other than the saga's identity, silently breaks a saga — spread one saga's events across partitions and the broker no longer promises they arrive in order, so a compensating event can overtake the forward event it is meant to undo. Key by the saga id, always.

The interaction of the two is where people get burned. Idempotency lets you process a duplicate safely, but if duplicates can also arrive *out of order* (which they can during a rebalance that replays from an earlier offset), your idempotency logic must be **monotonic**: it must not let an old, replayed event undo a newer state transition. The standard technique is to attach a **sequence number or version** to each saga state transition and have each consumer reject any event whose sequence is not greater than the last one it applied for that saga. That turns "process each event at least once" into "apply each transition exactly once, in order," which is what the saga actually needs.

```python
# Apply a saga event idempotently AND in order using a per-saga sequence number.
def apply_saga_event(evt):
    saga_id, seq = evt["saga_id"], evt["seq"]
    with db.transaction() as tx:
        cur = saga_store.lock_row(tx, saga_id)          # row lock for this saga only
        if seq <= cur.last_seq:
            return                                       # duplicate or out-of-order replay → drop
        apply_transition(tx, cur, evt)                   # the actual state change
        saga_store.set_last_seq(tx, saga_id, seq)        # advance the watermark, same txn
    # Commit: transition + watermark move together → exactly-once, in-order effect.
```

#### Worked example: sizing the idempotency and ordering machinery

Make it concrete with numbers. Say the order saga runs at peak **2,000 new orders per second**. Each saga emits roughly **6 events** end to end on the happy path (created, reserved, captured, shipped, confirmed, plus one internal), so the event rate is about **12,000 events/second**. You partition by order id. To keep per-partition throughput comfortable — say each consumer instance handles **1,000 events/second** with headroom — you need on the order of **12 partitions minimum**, and you would provision **24–32** to leave room for bursts and for the failure path, which roughly doubles event volume during an incident (every compensating event is an extra message). Because ordering is per-saga and a saga's events all share one order-id key, they all land in one partition, so 32 partitions gives you 32-way parallelism while preserving the in-order guarantee each saga needs.

Now the idempotency store. With at-least-once delivery and a measured duplicate rate of, say, **0.5%** under normal operation (spiking during rebalances), at 12,000 events/second you are doing 12,000 dedup checks/second, of which ~60/second are duplicates that the check turns into no-ops. The dedup table holds one row per (saga id, step) — 6 rows per saga, 12,000 new rows/second — so you need a retention/cleanup policy: keep dedup records only as long as a duplicate could plausibly arrive (say 7 days, matching broker retention) and then prune, or the table grows by ~7 billion rows a week. The sequence-number-per-saga technique adds one small integer column to the saga state row, no extra table. The lesson: sagas do not run on hope; they run on a correctly partitioned topic and a pruned idempotency store, and both must be sized for the *failure* path, not just the happy path, because compensation roughly doubles message volume exactly when the system is already stressed.

### Testing a saga: the failure matrix you cannot skip

Sagas have a testing problem that ordinary request/response code does not: the interesting behavior is in the *failure* paths, and there is one failure path per step, plus the compensation chain each triggers. A saga with N forward steps has N distinct "this step failed" scenarios, each of which must compensate the steps before it in the correct reverse order, and each compensation must itself be tested for idempotency and for the partial-failure case (the step half-happened). The number of paths grows fast, and teams that test only the happy path ship sagas that work in the demo and corrupt data in production.

The discipline is a **failure matrix**: enumerate, for each step, what happens when it fails, what happens when it times out (replies eventually, or never), and what happens when its compensation is delivered twice. For the order saga that is a small grid — reserve fails, charge fails, ship fails, each crossed with timeout and duplicate-compensation — and every cell is a test. The high-value tests are the nasty ones: charge *times out* (you do not know if the card was charged) and the compensation must query-then-act; the orchestrator *crashes mid-compensation* and must resume from its persisted `COMPENSATING` state without double-refunding; two sagas race for the *last unit* of inventory and exactly one must win. Property-based testing helps here — generate random interleavings of step successes, failures, and duplicate deliveries, and assert the invariant that the saga always reaches either fully-committed or fully-compensated, never a stuck middle. The invariant, not the path, is what you are verifying: *the saga has no permanent partial state*. If you can state that invariant and test it under adversarial interleaving, you have tested the saga; if you have only tested the happy path, you have tested nothing that matters.

## 9. A complete order saga with failure and compensation

Tie it all together with the full order saga, both happy path and failure path, as a single state machine. The grid below is the orchestrated saga's states: the forward path across the top (start, reserved, charged, shipped) and the compensate path that unwinds to aborted on a failure.

![The order saga as a state machine that advances through reserved, charged, and shipped on success and drops into a compensate path ending in aborted on failure](/imgs/blogs/saga-pattern-orchestration-vs-choreography-8.webp)

The happy path is the boring, good case, and most of your traffic lives here. The order service creates the order `PENDING` and starts the saga. Inventory reserves one unit (semantic lock: a reservation record, not a silent decrement). Payment captures the charge — this is the pivot. Shipping creates the shipment (retriable; if the warehouse API is down, retry, never compensate). The order service marks the order `CONFIRMED`. Every step committed locally, every step emitted its next event via the outbox, every step was idempotent. End state: consistent and complete.

The failure path is the interesting case and the reason sagas exist. Suppose shipping fails *before* it became retriable — say payment capture itself fails because the card is declined. Payment publishes `PaymentFailed`. Because capture was the pivot and it aborted, the saga is now committed to *unwinding*, not completing. Compensation runs in reverse: the only forward step that committed before payment was the inventory reservation, so C1 releases it, and the order service marks the order `CANCELED`. The figure below traces the reverse-order compensation: the forward steps committed, step three failed, and the compensations fire in the opposite order from how the forward steps ran.

![A timeline where the reserve and charge steps commit, the ship step fails, and the saga compensates the charge first and the reservation second, in reverse order](/imgs/blogs/saga-pattern-orchestration-vs-choreography-5.webp)

The reverse order is not arbitrary; it is required. Compensations run in the opposite order from the forward steps because later steps may depend on earlier ones. You refund the charge before releasing the inventory because the charge was made *after* the reservation, and unwinding a stack means popping the most recent first. If a compensation has its own dependency on the result of a later forward step, doing them out of order leaves a dangling reference. Reverse order is the safe default and the one the orchestrator enforces by walking its recorded state backward.

The happy-path-versus-failure contrast is the cleanest way to see the whole pattern at once, so here it is in one frame: on the left every step commits forward to a done saga; on the right a late failure forces the committed steps to be compensated in reverse to an aborted saga.

![A before-and-after view contrasting the saga happy path where every step commits forward with the failure path where committed steps are compensated in reverse](/imgs/blogs/saga-pattern-orchestration-vs-choreography-9.webp)

#### Worked example: trace an order saga where shipping fails

Walk a single order through the failure path with real values, because the abstract description hides the parts that bite. Order #8842: one unit of SKU `WIDGET-1`, total \$129.00, ship to an address the warehouse later rejects as undeliverable.

**T+0.00s — T1 reserve.** Inventory service receives `OrderCreated` for #8842. It writes a reservation row (`order_id=8842, sku=WIDGET-1, qty=1, status=RESERVED`) — a *semantic lock*, visible to other readers as claimed-pending — decrements available stock from 50 to 49, commits, and emits `InventoryReserved` via the outbox. Available stock is now genuinely 49 to the whole world.

**T+0.30s — T2 charge (the pivot).** Payment service receives `InventoryReserved`. It calls the gateway with `idempotency_key=charge:8842`, captures \$129.00, writes a payment record `status=captured`, commits, emits `PaymentCaptured`. This is the pivot — the saga is now committed to completing or fully compensating. The customer's card shows a \$129.00 charge.

**T+0.55s — T3 ship FAILS.** Shipping service receives `PaymentCaptured`, tries to create a shipment, and the warehouse API rejects the address as undeliverable — a hard, non-retriable failure (this is not a transient outage you retry through; the address is simply bad). Shipping emits `ShipmentFailed`. Now: capture was the pivot and it *succeeded*, so normally shipping would be retriable. But this failure is permanent, so the orchestrator's policy treats an unrecoverable post-pivot failure as a trigger to compensate the entire saga — a deliberate escalation, because there is no point retrying an undeliverable address forever.

**T+0.60s — C2 refund.** The orchestrator, in state `PAYMENT_CAPTURED`, receives `ShipmentFailed`, transitions to `COMPENSATING`, persists that state, and begins unwinding in reverse. The most recent committed forward step was the charge, so it sends `RefundCharge(8842)`. Payment service runs the idempotent, query-first compensation: it looks up `charge:8842`, confirms a \$129.00 capture exists, issues a refund with `idempotency_key=refund:8842`, marks the record `refunded`, commits, emits `ChargeRefunded`. The customer's statement now shows a \$129.00 charge *and* a \$129.00 refund — semantically equivalent to "never charged," but a different history.

**T+0.85s — C1 release.** Next in reverse is the reservation. The orchestrator sends `ReleaseStock(8842)`. Inventory service runs `release_if_held(8842)`: it finds the `RESERVED` row, deletes the reservation, increments available stock from 49 back to 50, commits, emits `InventoryReleased`. The semantic lock is gone; the unit is genuinely available again.

**T+0.90s — ABORTED.** The orchestrator records the saga as `ABORTED`. The order service receives the failure outcome and marks order #8842 `CANCELED`, perhaps queuing a "we couldn't ship to that address, please update it" email (itself a retriable post-decision step). End state: the customer is not charged, the inventory is not consumed, the order is cleanly canceled. No distributed lock was ever held; no service blocked on another; and at every instant the system was in a state it could explain — which is the entire point.

Notice what made this safe. The reservation was a *semantic lock*, so during the 0.9 seconds the saga was in flight, other sagas saw the unit as claimed-pending and did not also sell it. Every compensation was *idempotent and query-first*, so a duplicate `RefundCharge` would have found `status=refunded` and done nothing. The compensations ran in *reverse order* (refund before release). And the orchestrator's *persisted state* meant that if it had crashed at T+0.60s, it would have recovered into `COMPENSATING` and resumed the refund — the saga survives a coordinator crash that would have wedged a 2PC transaction. That recovery property, more than anything, is why sagas beat distributed transactions for long-running business flows.

## Case studies and war stories

**The oversold-inventory incident (missing semantic lock).** A retail team built a choreographed order saga where T1 reserved inventory by simply decrementing a count column, no reservation record. Under a flash sale, two sagas for the last unit both read "1 available," both decremented to 0, both proceeded to charge — and one customer got a unit that did not exist. The decrement *was* atomic per transaction, but there was no semantic lock that a second saga could see and respect, so both passed the "is there stock" gate before either committed. The fix was a conditional reservation (`UPDATE ... SET available = available - 1 WHERE available >= 1` returning rows affected, the commutative-update technique from section 7) plus a reservation row that the second saga's update would fail against. The lesson: a saga's lack of isolation is invisible until concurrency makes two sagas race on the same row, and the only defense is making the in-flight state recognizable and the update conditional.

**The stuck-saga forensics problem (choreography with no visibility).** A payments company ran a six-step choreographed saga across six services and one day found that some fraction of transfers were stalling — committed at step three, never reaching step four. With no orchestrator and no saga state store, there was no single place to ask "which sagas are stuck and where." Engineers spent a full day stitching together logs and traces across six services to discover that one service had silently stopped emitting its event after a deploy changed a topic name. They migrated the core flow to an orchestrator specifically so that "show me all sagas not in a terminal state, grouped by current state" became a single query. The lesson is the visibility row of the tradeoff matrix made painfully real: choreography's emergent flow is fine until you need to debug it at 3 a.m., and then the absence of a central state store costs you hours.

**The double-refund from a non-idempotent compensation.** A travel-booking saga compensated a failed booking by issuing a refund directly, with no idempotency key and no check for an existing refund. During a broker rebalance, the `RefundBooking` message was redelivered, and the compensation ran twice, refunding the customer twice — the company ate the difference. The fix was the query-first idempotent compensation pattern from section 3: look up the refund by a deterministic key before issuing it, and make the refund call itself carry an idempotency key so the payment processor deduplicates. The lesson: compensations are messages too, they are delivered at-least-once like everything else, and a compensation that is not idempotent is a money-losing bug waiting for a rebalance.

**The 2PC migration that did not scale.** A team tried to keep strong consistency across an order database and a Kafka publish using an XA-style distributed transaction wrapper. It worked in staging. In production, under load, prepared transactions piled up whenever the broker hiccupped, holding database locks, and a single slow broker partition would back up locks until the order database's connection pool exhausted and the whole order service fell over. They ripped out 2PC and adopted the [transactional outbox](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) plus a saga, trading cross-system atomicity for per-service atomicity plus compensation. Throughput went up and the cascading-lock outages stopped. The lesson is section 1 in the field: 2PC's lock-holding and blocking failure modes are not theoretical; they take down services under exactly the load you built the system to handle.

## When to reach for this (and when not to)

**Reach for a saga when** you have a business transaction that genuinely spans multiple services with separate databases, the steps each map to a clean local transaction, and you can define a semantic compensation for each step that needs to be undoable. Order placement, money transfers, multi-service provisioning, travel bookings (flight plus hotel plus car), and any "do these N things across N services, all or nothing" flow are textbook sagas. If the flow is long-running (seconds to days, with human steps or external callbacks in the middle), a saga — almost certainly orchestrated — is the right model, because no distributed lock could possibly be held that long.

**Choose choreography when** the saga is short (2–3 steps), mostly linear, the services are owned by different teams who prefer loose event coupling, and you do not anticipate complex branching or a strong need for centralized visibility. It is the lighter option and you should not reach for an orchestrator to sequence three steps.

**Choose orchestration when** the saga has four or more steps, has branches or parallel steps or timeouts, or when operational visibility into in-flight sagas matters (it usually does for anything touching money). Use a durable-execution framework (Temporal, Step Functions, Camunda) rather than hand-rolling the state persistence unless you have a strong reason not to. The orchestrator's observability pays for its operational cost the first time you need to answer "where are all the stuck orders" in one query.

**Do not reach for a saga when** the whole operation fits in a single service's single database — then just use a local ACID transaction; a saga there is pure overcomplication. Do not use a saga when you actually need *isolation* across the steps (no one may ever see partial state) — a saga cannot give you that, and if the business truly requires it, you need a different design (a single owning service, or accepting the latency of synchronous coordination). And do not build a saga for a step whose effect is irreversible and cannot be placed after a pivot — if the very first step ships a package and you cannot un-ship it, no compensation exists and the saga's core promise breaks; redesign so irreversible steps come last.

Finally, do not confuse a saga with *exactly-once magic*. A saga is built on at-least-once messaging plus idempotency plus compensation. It gives you a consistent *end state*, not a hidden distributed transaction. If you skip the idempotency and the semantic locks because the demo worked, production concurrency will find you.

A useful decision sequence when you are unsure: first ask whether the operation truly spans multiple services with separate data stores — if not, a local transaction beats a saga every time. If it does span services, ask whether every step has a sensible semantic compensation or can be placed after a pivot as a retriable step — if some step is irreversible *and* must run early, the saga model does not fit and you should redesign the flow before writing any code. If the saga is viable, count the steps and look for branching: two or three linear steps across teams that value loose coupling lean choreography; four or more steps, any branching, or money on the line lean orchestration on a durable-execution framework. And whichever style you pick, budget from day one for the three things that make a saga production-grade rather than demo-grade: idempotent steps and compensations keyed by saga id, semantic locks so concurrent sagas do not corrupt shared rows, and per-saga ordering so events apply in the sequence the saga intended. Those three are not optional polish; they are the difference between a saga that holds under load and one that quietly oversells inventory and double-charges cards the first time traffic spikes.

## Key takeaways

- A **saga** replaces an unavailable distributed ACID transaction with a sequence of **local transactions**, each committing in one service and emitting the message that triggers the next; on failure it runs **compensating transactions in reverse**.
- **Two-phase commit does not scale** for microservices: it holds locks across the network, blocks if the coordinator dies mid-commit, and multiplies your services' availabilities together. The saga trades cross-service isolation for availability and low latency.
- **Compensation is semantic, not a rollback.** You cannot un-commit a local transaction; you issue a new one that is business-equivalent to undoing it — refund the charge, release the reservation. Some effects (a shipped package) cannot be compensated, so order steps so irreversible ones come last.
- **Choreography** has no coordinator: services react to each other's events. It is loosely coupled and lightweight but the flow is emergent and hard to see. **Orchestration** uses a central state machine that issues commands and tracks progress: explicit and observable, at the cost of operating one more component.
- Pick by shape: **choreography for short linear sagas (2–3 steps); orchestration for long, branching, or visibility-critical sagas (4+ steps)**. A **hybrid** — orchestrate the core, choreograph the periphery — is often best.
- **Sagas are not isolated.** Other transactions see partial state. Defend with **semantic locks** (recognizable pending state), a **pivot transaction** (the point of no return), and **commutative + reread** updates so concurrent sagas do not corrupt each other.
- Every saga depends on **idempotency** (at-least-once delivery means every step and compensation runs more than once — dedup by saga-id-plus-step, committed in the same transaction) and **per-key ordering** (partition by saga id so each saga's events are delivered in order; add a sequence number to apply transitions exactly once and monotonically).
- Reliable step publishing requires the [transactional outbox](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing): write the next event in the same local transaction as the business change, so the saga never stalls on a lost event.
- Size the messaging for the **failure path**, not just the happy path: compensation roughly doubles event volume exactly when the system is already stressed.
- A saga gives you a consistent **end state**, not exactly-once magic. It is at-least-once messaging plus idempotency plus compensation, and it survives a coordinator crash that would wedge a 2PC transaction.

## Further reading

- [The transactional outbox pattern: atomic database writes and reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — how each saga step reliably emits its next event without the dual-write problem.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the dedup machinery every saga step and compensation requires.
- [Event-driven architecture: events, commands, and documents](/blog/software-development/message-queue/event-driven-architecture-events-commands-documents) — the events-versus-commands distinction that separates choreography from orchestration.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — how per-key ordering and partitioning underpin a saga's in-order event delivery.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — where a saga's eventual consistency sits in the spectrum.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the formal reason strong cross-service consistency costs availability and latency.
- Hector Garcia-Molina and Kenneth Salem, "Sagas" (1987) — the original paper that named the pattern.
- Chris Richardson, *Microservices Patterns* — the modern, practical treatment of sagas, semantic locks, and the orchestration-versus-choreography decision.

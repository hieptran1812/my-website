---
title: "The Saga Pattern: Distributed Transactions Without Distributed Transactions"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Why a business operation that spans multiple services cannot use a single ACID transaction, and how the saga pattern stitches local commits and compensating actions into something that behaves like one, with all the anomalies that come from giving up isolation."
tags:
  [
    "saga-pattern",
    "distributed-transactions",
    "microservices",
    "orchestration",
    "choreography",
    "compensating-transaction",
    "temporal",
    "event-driven",
    "distributed-systems",
    "databases",
    "idempotency",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/saga-pattern-distributed-transactions-1.webp"
---

The most expensive bug I ever helped diagnose was a single oversold pair of limited-edition sneakers. One pair. The dollar figure was trivial. What was not trivial was the three days four engineers spent reconstructing how a system that had a column literally named `quantity_available` had let two customers buy the last unit, charge both their cards, and ship neither — because the warehouse, by then, had exactly zero. The order service said both orders were valid. The payment service said both charges succeeded. The inventory service said the count went negative for a few hundred milliseconds and then a "correction job" clamped it to zero. Every individual service was behaving exactly as designed. No service had a bug. And yet the business had committed fraud against one of its own customers, twice.

This is the shape of almost every distributed-data incident I have been paged for: not a crash, not a corrupt row, but a *business invariant* — "you cannot sell stock you do not have," "you cannot ship an order you have not charged," "the money debited equals the money credited" — that no single service is responsible for enforcing, and that nothing enforced across the boundary between them. In a single relational database, this class of bug is impossible by construction. You wrap the order insert, the inventory decrement, and the payment record in one `BEGIN … COMMIT`, and the database's atomicity and isolation guarantees make the whole thing all-or-nothing and invisible-until-done. The moment you split those three writes across three services with three databases, that single transaction is gone, and you are left holding the question this entire article is about: **how do you make a multi-step business operation behave like a transaction when there is no transaction?**

The answer the industry converged on is the **saga pattern**, first described by Hector Garcia-Molina and Kenneth Salem in their 1987 SIGMOD paper ["Sagas"](https://dl.acm.org/doi/10.1145/38713.38742) — written, fascinatingly, not for microservices (which would not exist for two more decades) but for long-lived database transactions that held locks too long. A saga is a sequence of local transactions where each step has a **compensating transaction** that semantically undoes it. The diagram above is the mental model for the whole article: a saga is a chain of local commits, one per service, with no shared transaction spanning them; when a later step fails, you do not "roll back" — there is nothing to roll back, because each step already committed — instead you run the compensations of the earlier steps in reverse, walking the system back to a state that is *consistent at the business level* even though it was never *isolated* the way a database transaction would have been. That missing isolation is the hard part, and it is where most of this article lives.

## Why a multi-service operation cannot be one transaction

![A saga is a chain of local transactions, one local commit per service, with no shared transaction spanning them and the next step triggered by the previous step's event](/imgs/blogs/saga-pattern-distributed-transactions-1.webp)

Read figure 1 left to right. An order goes through four services in turn: the order service creates the order row and commits to *its* database; the inventory service decrements stock and commits to *its* database; the payment service charges the card and commits to *its* database; the shipping service creates a shipment and commits to *its* database. Five boxes, four independent commits, and — this is the entire point — **no transaction spans them.** Each `COMMIT` is final and visible the instant it returns. There is no outer `BEGIN` wrapping the four, because there is no single database that owns all four tables.

The obvious question is: why not? Distributed databases have solved cross-node transactions for decades. The answer is the **two-phase commit** protocol (2PC), and the reason sagas exist is that 2PC across heterogeneous services is, in Martin Kleppmann's framing in *Designing Data-Intensive Applications* Chapter 9, a technology that "provides safety but at the cost of operational headaches." Let me make that concrete rather than hand-wave it.

In 2PC, a coordinator runs a transaction in two rounds. In the **prepare** phase it asks every participant, "can you commit this? promise me you will be able to," and each participant does all the work, writes it durably, takes the locks, and replies "prepared" — but does *not* commit. In the **commit** phase, once every participant has promised, the coordinator tells them all to commit. The protocol is correct: if any participant cannot prepare, the coordinator aborts everyone. But look at what a participant must do between "prepared" and "commit": it must hold its locks, keep its transaction open, and **wait for the coordinator** — and it cannot unilaterally decide the outcome, because that would break atomicity. If the coordinator crashes after some participants have prepared but before it sends the commit decision, those participants are stuck **in doubt**: they hold locks, they block any conflicting transaction, and they cannot proceed until the coordinator comes back and tells them what was decided. This is not a theoretical edge case. It is the defining failure mode of 2PC, and it is why Kleppmann calls distributed transactions "the source of major outages."

The table below is the assumption-versus-reality framing that should make you reach for a saga the moment you hear "we'll just do a distributed transaction across the services."

| What people assume about 2PC | What actually happens in production |
| --- | --- |
| It is atomic, so it is safe | It is atomic *only if the coordinator never permanently dies*; coordinator failure leaves participants blocked in-doubt, holding locks indefinitely |
| Services can participate easily | Each participant needs an XA-compatible resource manager; most message brokers (Kafka included) and most HTTP services have no usable XA interface at all |
| Locks are held briefly | Locks are held across the *entire* prepare-to-commit window, which spans network round trips to every participant — orders of magnitude longer than a local transaction |
| It scales with more services | Latency and lock-hold time grow with the slowest participant; one slow service stalls everyone; availability is the *product* of all participants' availabilities |
| Recovery is automatic | In-doubt transactions often require a human to manually `COMMIT` or `ROLLBACK` from the transaction manager's recovery log after a coordinator crash |

That last row is the killer. A protocol whose recovery story includes "a human reads the coordinator's log at 3 a.m. and decides the fate of in-doubt transactions" is not a protocol you want on the hot path of every order. And the availability math is brutal: if each of four services is independently up 99.9% of the time, a 2PC across all four succeeds only when *all four* are simultaneously up, which is `0.999⁴ ≈ 99.6%` — you have made the combined operation *less* available than any individual service, by coupling their fates. This is the opposite of what microservices are for.

> The whole promise of splitting a monolith into services is that they fail independently. A distributed transaction couples their fates back together. You spent a year decoupling deployments and you just re-coupled commits.

So the database-per-service architecture — where each service owns its data and no other service touches that database directly — makes 2PC structurally unavailable. AWS's own [prescriptive guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/saga-orchestration.html) states it plainly: "In distributed systems that follow a database-per-service design pattern, the two-phase commit is not an option. This is because each transaction is distributed across various databases, and there is no single controller that can coordinate." The saga is what you reach for instead. If you want the full picture of the protocol the saga is replacing, I cover the mechanics and failure modes of 2PC in [two-phase commit and how it fails](/blog/software-development/database/two-phase-commit-and-how-it-fails); this article assumes you have decided not to use it and need something else.

## The saga: atomicity without isolation

Here is the precise definition from the original paper, because the modern microservices framing often loses the rigor. Garcia-Molina and Salem define a saga as a long-lived transaction that can be broken into a sequence of sub-transactions `T₁, T₂, …, Tₙ`, where each `Tᵢ` is an ordinary ACID transaction, and where **each `Tᵢ` has an associated compensating transaction `Cᵢ`** that semantically undoes the effects of `Tᵢ`. The database (or, today, the orchestrator) then guarantees one of two outcomes:

1. The whole sequence runs to completion: `T₁ T₂ … Tₙ`, or
2. A prefix runs and then is compensated: `T₁ T₂ … Tⱼ Cⱼ Cⱼ₋₁ … C₁` for some `j < n`.

That second outcome is the heart of it. If `Tⱼ₊₁` fails, you do not magically un-commit `T₁ … Tⱼ` — they are already durable. Instead you run their compensations in **reverse order**: `Cⱼ` first (undo the most recent committed step), then `Cⱼ₋₁`, and so on back to `C₁`. The reverse ordering matters because compensations, like the forward steps, can have dependencies — you release the stock reservation before you cancel the order it belonged to, not after.

Now state the guarantee precisely, and notice what is missing. A saga gives you **atomicity** in a weakened, *semantic* sense: either all the effects happen, or all the effects are undone, eventually. It does **not** give you **isolation**. This is not an implementation detail you can fix; it is mathematically inherent in the structure. Each `Tᵢ` commits independently and is *immediately visible to the rest of the world* the moment it commits. There is no version of the saga where the intermediate states are hidden, because hiding them would require holding the locks across all `n` steps — which is exactly the 2PC behavior you abandoned. The [dimosr writeup of the saga paper](https://dimosr.github.io/saga-transactions/) puts the formal property crisply: "a saga transaction provides guarantees around atomicity, but it does not provide guarantees around isolation."

The four-letter mnemonic that helps me remember which properties survive the transition is **ACD, not ACID**:

| ACID property | In a local transaction | In a saga |
| --- | --- | --- |
| **A** — Atomicity | All-or-nothing, instant | All-or-nothing *eventually*, via compensations |
| **C** — Consistency | Invariants hold at commit | Invariants hold *at the business level* if compensations are correct |
| **I** — Isolation | Concurrent txns can't see each other's intermediate state | **Gone.** Other sagas and reads *can* see intermediate state |
| **D** — Durability | Committed data survives crashes | Yes — each local commit is durable |

The "I" is the only one you sacrifice, but it is the one that causes the subtle, expensive, hard-to-reproduce bugs — the oversold sneaker. Most of the engineering effort in building a correct saga goes into clawing back *just enough* isolation, through application-level countermeasures, to keep the specific anomalies that matter to your business from happening. We will get to those countermeasures, but first you have to understand the two ways to wire the steps together, because that decision shapes everything else.

## 1. Orchestration versus choreography: where the brain lives

![Orchestration centralizes saga control flow in one coordinator that sends commands and drives compensation; choreography spreads it across services that emit and subscribe to events with no central brain](/imgs/blogs/saga-pattern-distributed-transactions-2.webp)

**The first architectural decision in any saga is where the coordination logic lives, and there are exactly two answers.** Chris Richardson, who catalogued the saga pattern for the microservices era on [microservices.io](https://microservices.io/patterns/data/saga.html), names them precisely:

- **Choreography**: "each local transaction publishes domain events that trigger local transactions in other services." There is no central coordinator. The order service commits and emits `OrderCreated`; the inventory service is subscribed to `OrderCreated`, reserves stock, and emits `StockReserved`; the payment service is subscribed to `StockReserved`, charges, and emits `Charged`; and so on. The saga's control flow is *implicit*, distributed across the services' event subscriptions.
- **Orchestration**: "an orchestrator (object) tells the participants what local transactions to execute." A single orchestrator service owns the saga's state machine. It sends `ReserveStock` to the inventory service, waits for the reply, then sends `ChargePayment` to the payment service, and so on. On failure it drives the compensation chain centrally.

Figure 2 is the before/after. The left column is orchestration: one coordinator owns the state machine, sends commands and awaits replies, drives compensation centrally, and — the cost, in amber — is a single point of failure and a service that knows about everyone. The right column is choreography: services emit and subscribe to events, there is no central brain and no single point of failure, but the flow is implicit and — the cost, in red — cyclic dependencies between services become genuinely hard to debug.

### When choreography is the right call

Choreography shines when the saga is short (two to four steps), the steps are loosely related, and you want maximum decoupling. Nobody owns the whole flow, so there is no service that becomes a god-object knowing about every other service. It is also naturally resilient: there is no orchestrator to crash. The price is that the business process exists only in the *emergent* behavior of the subscriptions — there is no single place to read to understand "what happens when an order is placed." You reconstruct the flow by grepping for who publishes and who subscribes to each event type.

The failure mode that bites teams hard is **cyclic dependencies**. Service A emits an event B reacts to, B emits one C reacts to, and six months later someone adds a subscription that makes C emit something A reacts to, and now you have a loop nobody designed and nobody can see. Richardson's guidance, and mine, is that choreography stops scaling around four or five participants — past that, the implicit flow becomes impossible to hold in your head, and the cost of *not* having a single place that describes the process exceeds the cost of the orchestrator.

Here is a choreographed payment step in Python. Notice there is no coordinator anywhere — the service just reacts to one event type and emits the next.

```python
# payment_service/consumer.py — a single participant in a CHOREOGRAPHED saga.
# It knows nothing about "the order saga"; it only knows: when stock is
# reserved, charge the card; emit success or failure for the next service.

import json
from dataclasses import dataclass

@dataclass
class StockReserved:
    order_id: str
    customer_id: str
    amount_cents: int

def handle_stock_reserved(event: StockReserved, broker, payments_db):
    # Idempotency guard: this consumer may see the same event more than once
    # (at-least-once delivery). The unique constraint makes the charge exactly-once.
    try:
        with payments_db.begin() as tx:
            tx.execute(
                "INSERT INTO charges (order_id, customer_id, amount_cents, status) "
                "VALUES (%s, %s, %s, 'PENDING')",
                (event.order_id, event.customer_id, event.amount_cents),
            )
    except UniqueViolation:
        # Already processed this order_id; do nothing, do not double-charge.
        return

    try:
        auth = charge_card(event.customer_id, event.amount_cents)  # external PSP call
    except CardDeclined:
        with payments_db.begin() as tx:
            tx.execute("UPDATE charges SET status='DECLINED' WHERE order_id=%s",
                       (event.order_id,))
        # Publish FAILURE — the inventory service is subscribed to this and will
        # run its own compensation (release the reservation). No orchestrator.
        broker.publish("payment.failed",
                       json.dumps({"order_id": event.order_id, "reason": "declined"}))
        return

    with payments_db.begin() as tx:
        tx.execute("UPDATE charges SET status='CAPTURED', psp_ref=%s WHERE order_id=%s",
                   (auth.ref, event.order_id))
    # Publish SUCCESS — the shipping service is subscribed to this.
    broker.publish("payment.captured",
                   json.dumps({"order_id": event.order_id, "psp_ref": auth.ref}))
```

The compensation in a choreographed saga is itself choreographed: the `payment.failed` event is what the inventory service subscribes to in order to know it must release the reservation. There is no central code that says "on failure, run these compensations." The undo is just another event-and-reaction.

#### Second-order gotcha: choreography has no built-in timeout owner

In orchestration, the orchestrator naturally owns timeouts — it sent `ChargePayment`, so it knows to give up after 30 seconds and start compensating. In choreography, *who* notices that the payment service never emitted anything? Nobody is waiting. The usual fix is a separate "saga timeout" mechanism — a scheduled job, or a process-manager pattern — which is the camel's nose of orchestration creeping back in. If you find yourself building a timeout-watcher for a choreographed saga, that is a strong signal you actually wanted orchestration.

### When orchestration is the right call

Orchestration wins for anything complex: many steps, conditional branches ("if the customer is a VIP, skip the fraud check"), parallel steps that must join, and especially **sagas where the compensation logic is non-trivial**. The orchestrator gives you a single, readable, testable definition of the entire business process — and, critically, a single place that owns failure handling. Richardson's own conclusion: "For complex sagas, orchestration is typically more effective." Every serious saga framework — Temporal, AWS Step Functions, Netflix Conductor, Camunda — is an orchestration engine, which tells you where the industry landed for real systems.

The cost is real and you should name it: the orchestrator knows about every participant, which is a form of coupling, and it is a component that can fail. Both are mitigable. The coupling is *command* coupling, not data coupling — the orchestrator sends commands and receives replies but never touches another service's database, so the services stay independently deployable. And the single-point-of-failure concern is solved by making the orchestrator itself durable and replicated, which is precisely what Temporal and Step Functions do. AWS notes that using Step Functions "mitigates the single point of failure issue, which is inherent in the implementation of the saga orchestration pattern," because the engine "has built-in fault tolerance and maintains service capacity across multiple Availability Zones."

| Dimension | Choreography | Orchestration |
| --- | --- | --- |
| Coordination logic | Implicit, spread across event subscriptions | Explicit, in one orchestrator/state machine |
| Single point of failure | None | The orchestrator (mitigate with durability + replication) |
| Where to read "the process" | Nowhere — reconstruct from pub/sub graph | One place — the state machine definition |
| Coupling | Loose (event coupling) | Command coupling (orchestrator knows participants) |
| Complex/conditional flows | Painful, cyclic deps emerge | Natural |
| Timeouts | No natural owner | Orchestrator owns them |
| Good up to | ~4 participants | Dozens of steps, branches, joins |
| Debugging | Distributed trace across services | Inspect orchestrator state + history |

My rule of thumb: **start with choreography only if the saga is genuinely small and likely to stay that way; reach for orchestration the moment there is a branch, a join, a timeout you care about, or a compensation chain longer than one step.** In practice that means most real sagas are orchestrated, and the rest of this article's code uses orchestration, because it is where the interesting failure handling lives.

## 2. The orchestrator as a state machine

![An orchestrator models the order saga as a state machine that advances on success replies and branches into a reverse compensation chain on any failure reply](/imgs/blogs/saga-pattern-distributed-transactions-7.webp)

**An orchestrator is, at its core, a durable state machine: each state means "I have sent command X and I am awaiting its reply," and each reply either advances the machine or routes it into compensation.** Figure 7 shows the order saga as exactly this. `START` transitions to `AWAIT order created`; on the order service's success reply, to `AWAIT stock reserved`; then `AWAIT payment charged` — which is the pivot, marked amber — then `AWAIT shipment created`, then the `CONFIRMED` terminal state. On *any* failure reply, the machine branches into the red compensation chain: `COMPENSATE release stock`, then `COMPENSATE cancel order`, then the `ABORTED` terminal state. The whole saga is captured in that one graph, and that is the single biggest reason to orchestrate: the process is *legible*.

The non-negotiable property of this state machine is that it must be **durable**. If the orchestrator process crashes between "sent `ChargePayment`" and "received the reply," it must, on restart, know it was in `AWAIT payment charged` and either re-await or re-issue. If the state lived only in memory, a crash would orphan the saga: stock reserved, money possibly charged, and nobody driving it to completion or compensation. So the orchestrator persists its current state and history after every transition. Here is a hand-rolled orchestrator that makes the durability explicit — this is the thing the frameworks build for you, shown unrolled so you can see the moving parts.

```python
# orchestrator.py — a durable, hand-rolled saga orchestrator state machine.
# State lives in a `saga_instances` table so a crash + restart resumes cleanly.
# Each step is (command, compensation). On failure we run compensations of
# completed steps in REVERSE order.

import enum, json, time
from dataclasses import dataclass
from typing import Callable, Optional

class Status(enum.Enum):
    RUNNING       = "RUNNING"
    COMPENSATING  = "COMPENSATING"
    CONFIRMED     = "CONFIRMED"   # terminal: forward path completed
    ABORTED       = "ABORTED"     # terminal: compensated back to start

@dataclass
class Step:
    name: str
    action: Callable[[dict], dict]        # the forward local transaction (a command)
    compensation: Optional[Callable[[dict], None]]  # None => retriable, no undo
    retriable: bool = False               # if True, retry forward instead of compensating

class Orchestrator:
    def __init__(self, db, steps: list[Step]):
        self.db = db
        self.steps = steps

    def _save(self, saga_id, status: Status, idx: int, ctx: dict):
        # Durable checkpoint after EVERY transition. This is what survives a crash.
        with self.db.begin() as tx:
            tx.execute(
                "INSERT INTO saga_instances (id, status, step_index, context, updated_at) "
                "VALUES (%s, %s, %s, %s, now()) "
                "ON CONFLICT (id) DO UPDATE SET status=%s, step_index=%s, "
                "context=%s, updated_at=now()",
                (saga_id, status.value, idx, json.dumps(ctx),
                 status.value, idx, json.dumps(ctx)),
            )

    def run(self, saga_id: str, ctx: dict):
        # Resume: load existing state if this saga already started (crash recovery).
        row = self.db.fetchone(
            "SELECT status, step_index, context FROM saga_instances WHERE id=%s",
            (saga_id,))
        if row:
            status, idx, ctx = Status(row.status), row.step_index, json.loads(row.context)
            if status in (Status.CONFIRMED, Status.ABORTED):
                return status                      # already finished; idempotent no-op
            if status is Status.COMPENSATING:
                return self._compensate(saga_id, idx, ctx)
        else:
            idx = 0
            self._save(saga_id, Status.RUNNING, idx, ctx)

        # Forward path.
        while idx < len(self.steps):
            step = self.steps[idx]
            try:
                result = self._invoke_forward(step, ctx)   # idempotent command call
                ctx.update(result)
                idx += 1
                self._save(saga_id, Status.RUNNING, idx, ctx)
            except StepFailed:
                if step.retriable:
                    # Forward recovery: this step MUST eventually succeed. Back off
                    # and retry; never compensate past a retriable step.
                    time.sleep(backoff(step.name))
                    continue
                # Backward recovery: compensate everything completed BEFORE this step.
                self._save(saga_id, Status.COMPENSATING, idx, ctx)
                return self._compensate(saga_id, idx, ctx)

        self._save(saga_id, Status.CONFIRMED, idx, ctx)
        return Status.CONFIRMED

    def _compensate(self, saga_id: str, failed_idx: int, ctx: dict):
        # Run C_j, C_{j-1}, ..., C_1 in REVERSE order over completed steps.
        for j in range(failed_idx - 1, -1, -1):
            comp = self.steps[j].compensation
            if comp is None:
                # No compensation => this was the pivot or a retriable step.
                # We must NOT be here past a pivot; reaching it is a design bug.
                raise SagaInvariantViolation(
                    f"step {self.steps[j].name} has no compensation but needs undo")
            self._invoke_compensation(comp, ctx)  # idempotent, retried until it succeeds
            self._save(saga_id, Status.COMPENSATING, j, ctx)
        self._save(saga_id, Status.ABORTED, 0, ctx)
        return Status.ABORTED
```

Two things in that code are doing the heavy lifting and deserve to be called out. First, `_save` is invoked after *every* transition, not just at the start and end — that is the durability that lets a crash resume. Second, `_compensate` walks the completed steps in reverse and refuses to proceed past a step with no compensation, which is how the code enforces the pivot rule we are about to discuss. The `_invoke_forward` and `_invoke_compensation` helpers (elided) are where idempotency lives; we will build a real idempotent step shortly.

### Second-order optimization: the orchestrator should be the only writer of saga state

A subtle bug I have seen twice: a "monitoring" job that "helpfully" updates the `saga_instances` table to mark stuck sagas as failed. Now two processes write saga state, and you have reintroduced a race — the orchestrator advances a step at the same moment the monitor marks it aborted, and the saga both compensates and continues. The orchestrator must be the *sole* writer of its own state machine. Monitors read; they do not write. If you need to force-fail a stuck saga, send the orchestrator a command and let *it* transition.

## 3. The canonical example, end to end

Let me ground all of this in the order saga that figure 1 sketched, because the abstract `Tᵢ`/`Cᵢ` notation hides where the real decisions are. Here are the four forward steps and their compensations, with the property that makes each one tricky.

| Step | Forward `Tᵢ` | Compensation `Cᵢ` | Why it's tricky |
| --- | --- | --- | --- |
| 1. Create order | Insert order row, `status=PENDING` | Set `status=CANCELLED` (a *semantic* undo, not a delete) | The order row must persist for audit; you cancel, not delete |
| 2. Reserve stock | Decrement `available`, increment `reserved` | Increment `available`, decrement `reserved` | Concurrent sagas read `available` — this is where the oversell happens |
| 3. Charge payment | Capture funds via PSP | Refund (or void the auth) | A refund is *not* a clean undo — fees, customer-visible, the PSP may decline the refund |
| 4. Create shipment | Allocate a shipment, print a label | Cancel the shipment | If the package already left the building, you cannot un-ship it |

Notice that **not one of these compensations is a true inverse.** Cancelling an order leaves a cancelled row, not a clean slate. A refund costs money and is visible to the customer in a way the charge-then-refund pair was not. And cancelling a shipment is only possible if it has not physically left. This is the universal truth about compensations: they are **semantically** equivalent to undoing, not **physically** equivalent. Garcia-Molina and Salem's word for this is exactly right — *compensating*, not *reversing*. You compensate for the effect; you rarely erase it.

Here is the orchestrator wired up with these four steps. The payment step is special — it is the pivot — and you can see how that is encoded.

```python
# order_saga.py — the canonical order saga, wired into the orchestrator above.

def create_order(ctx):
    order_id = order_service.create(ctx["customer_id"], ctx["items"])  # commits PENDING
    return {"order_id": order_id}

def cancel_order(ctx):
    order_service.cancel(ctx["order_id"])      # sets status=CANCELLED, idempotent

def reserve_stock(ctx):
    inventory_service.reserve(ctx["order_id"], ctx["items"])  # may raise OutOfStock
    return {"stock_reserved": True}

def release_stock(ctx):
    inventory_service.release(ctx["order_id"], ctx["items"])  # idempotent

def charge_payment(ctx):
    # PIVOT. Once this commits, the saga MUST roll forward — we will not refund
    # automatically as part of normal flow; a refund is a business decision, not
    # an automatic compensation. So: no compensation registered.
    ref = payment_service.capture(ctx["order_id"], ctx["amount_cents"])  # may decline
    return {"psp_ref": ref}

def create_shipment(ctx):
    # RETRIABLE. After payment, we are committed to fulfilling. If the shipping
    # service is down, we retry forever; we never undo the charge for a transient
    # shipping outage.
    label = shipping_service.create(ctx["order_id"])  # retry on transient failure
    return {"label": label}

ORDER_SAGA = [
    Step("create_order",   create_order,    cancel_order),
    Step("reserve_stock",  reserve_stock,   release_stock),
    Step("charge_payment", charge_payment,  compensation=None),               # pivot
    Step("create_shipment",create_shipment, compensation=None, retriable=True),# forward-only
]
```

Run this with a card that gets declined and you get exactly the reverse-order compensation from the paper: `charge_payment` raises, the orchestrator transitions to `COMPENSATING`, runs `release_stock` (undo step 2), then `cancel_order` (undo step 1), and ends `ABORTED`. The customer sees "your card was declined, no order was placed," and behind the scenes the stock is back on the shelf and the order row reads `CANCELLED`. No oversell, no phantom charge.

![On a step failure the saga does not abort cleanly like a transaction; it walks the already-committed steps backward in time, running each compensation in reverse order](/imgs/blogs/saga-pattern-distributed-transactions-3.webp)

Figure 3 puts that on a timeline, and it is worth dwelling on how *different* this is from a database rollback. In a local transaction, an abort is instantaneous and leaves no trace — the partial work simply never became visible. In a saga, the failure at `t2` (the card declined) happens *after* `T1` and `T2` already committed and were already visible to the world. There is no instantaneous abort; there is only a sequence of new forward operations — `C2` at `t3`, then `C1` at `t4` — that each *commit* and *each undo a previous commit*, until at `t5` the system reaches a state that is consistent at the business level. The compensations are themselves local transactions on the timeline, not magic; the difference between figure 1's happy path and figure 3's rollback is just *which* sequence of local commits runs.

## 4. Forward recovery, backward recovery, and the pivot

![Compensatable steps roll back, the pivot is the commit point with no undo, and retriable steps roll forward because their effects cannot be undone](/imgs/blogs/saga-pattern-distributed-transactions-5.webp)

**Every step in a saga is one of exactly three kinds, and which kind it is determines how the saga recovers when something downstream fails.** This taxonomy is Richardson's, and it is the single most useful mental model for designing a saga that does not have an impossible compensation lurking in it. Figure 5 is the matrix:

- **Compensatable transaction**: can be undone by a compensation `Cᵢ`. On a downstream failure, you roll *back* by running `Cᵢ`. Steps 1 and 2 of the order saga (create order, reserve stock) are compensatable.
- **Pivot transaction**: the point of no return. It either commits the saga (after which the saga *will* complete) or, if it fails, is the last thing that can abort it. A pivot has no compensation, because once it succeeds, you do not undo it automatically. Charging the card is the pivot: once the money is captured, the saga is committed to fulfilling the order.
- **Retriable transaction**: comes *after* the pivot, has no compensation, and is guaranteed to eventually succeed if you retry it enough. On failure you roll *forward* — keep retrying — never backward. Creating the shipment is retriable: after payment, a transient shipping outage must not undo the charge; you retry until the shipment is created.

The two recovery directions follow directly. **Backward recovery** is the paper's `Cⱼ … C₁`: undo the committed compensatable steps in reverse. It is only possible *before* the pivot. **Forward recovery** is "retry until success": it is the only option *after* the pivot, because there is nothing to roll back to. The pivot is the hinge between them.

### Why the pivot exists: some actions cannot be undone

![The pivot splits a saga into a compensatable zone where you roll back, the pivot commit point, and a retriable zone where you can only roll forward](/imgs/blogs/saga-pattern-distributed-transactions-6.webp)

Figure 6 makes the structure spatial. The blue zone on the left is compensatable: every step there has a `Cᵢ`, and any failure runs the undos and aborts. The red box in the middle is the pivot — "once T3 commits, rollback is impossible." The green zone on the right is retriable: every step there must be retried to success, and the saga never aborts once it is in that zone.

The reason the pivot is a *concept* and not just "the third step" is that some actions are genuinely irreversible, and you must structure the saga so that the irreversible actions come as late as possible and act as the commit point. The canonical irreversible action is **sending an email** (or an SMS, or a push notification). You cannot un-send "Your order has shipped!" If your saga sends that email at step 2 and then fails at step 4, you have lied to the customer and there is no compensation — there is no `C` that reaches into their inbox and deletes the message. The fix is to **order the saga so that user-visible, irreversible actions happen after the pivot**, in the retriable zone, where they only happen on the success path. You send "order confirmed" *after* the payment is captured, never before.

When you cannot reorder — when an irreversible action genuinely must happen early — your only tools are:

1. **A counter-action instead of an undo.** You cannot un-send the shipping email, but you can send a "we're sorry, your order was cancelled" email. The counter-action is itself a compensation; it just is not invisible. Document that the customer will receive two emails in the failure case, and decide whether that is acceptable. Often it is.
2. **A pessimistic guard before the irreversible action.** Do everything that *can* fail before the thing that cannot. If you must send a physical letter, validate the address, confirm payment, and reserve everything *first*, so that by the time you commit the irreversible step, the probability of needing to undo it is as close to zero as you can make it. This is the pivot principle applied at fine grain.
3. **Make the irreversible action the pivot.** If charging the card and sending a legally-required disclosure both must happen, and neither can be undone, then the *later* one is your pivot and the saga simply cannot abort once it reaches it. Design the rest of the saga to validate everything compensatable before that line.

> The pivot is not a step you pick; it is a property you design for. You arrange the saga so that everything reversible happens before the irreversible thing, and the irreversible thing becomes the commit point.

## 5. The hard part: a saga is not isolated

![Two concurrent sagas with no isolation produce a lost update because saga B reads the intermediate balance that saga A will later compensate away, overwriting B's debit](/imgs/blogs/saga-pattern-distributed-transactions-4.webp)

**This is the section that separates engineers who have read about sagas from engineers who have shipped one and been paged for it.** The lack of isolation is not a footnote; it is the dominant source of saga bugs, and it manifests as the exact anomalies you may know from database [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) — dirty reads, lost updates, non-repeatable reads — except now they happen at the *business* level, across services, where no database isolation level can save you because there is no single database.

Figure 4 walks through a lost update with two concurrent sagas operating on a shared account balance, starting at 500. Saga A debits 100, committing the balance to 400, and is now mid-flight (its later step has not run yet). Saga B reads 400 — a **dirty read** of A's uncommitted intermediate state — and debits 50, committing 350. Now A's pivot fails, so A compensates by restoring the balance it remembers, 500 — and in doing so **erases B's debit entirely.** The final balance is 500, but the correct value, given that B's debit was legitimate and A aborted, is 450. B's 50 is gone, silently. This is a **lost update**, and it happened precisely because A's intermediate state (balance = 400) was visible to B, which is exactly the isolation that a saga does not provide.

Here is the catalog of anomalies, the same family as database isolation anomalies but at the saga level:

| Anomaly | Saga-level manifestation | Concrete example |
| --- | --- | --- |
| **Dirty read** | One saga reads another's uncommitted intermediate state | B reads the balance A debited but hasn't finalized; B acts on a number that will be compensated away |
| **Lost update** | Two sagas read-modify-write the same data; one's update is overwritten | Two sagas both reserve the "last" unit because both read `available=1`; one compensation restores the count and erases the other's reservation |
| **Non-repeatable read** | A saga reads the same data twice across steps and gets different values | A long saga reads a price in step 1, a concurrent price change commits, step 4 charges the new price |

The oversell from the opening is the lost-update / dirty-read anomaly in its purest form: two order sagas both read `available = 1`, both decrement to 0, both charge. No database lock was violated — each decrement was a valid local transaction — but the *business invariant* "don't sell what you don't have" was violated because the two sagas were not isolated from each other.

## 6. Countermeasures: clawing back just enough isolation

Richardson catalogs a set of **countermeasures** — application-level design techniques that recover *just enough* isolation to prevent the specific anomalies that matter to your business. You do not get ACID isolation back; you get targeted defenses. Here are the ones that earn their keep, in roughly the order I reach for them.

### 6.1 Semantic lock — the one you almost always need

A **semantic lock** is an application-level lock represented as a *state flag* on the data, signaling "this is in-progress, do not act on it as if it were final." The cheapest and most effective version is a status column: `PENDING`, `CONFIRMED`, `CANCELLED`. While a saga holds the resource in `PENDING`, other sagas and reads can *see* that it is pending and refuse to treat it as committed. This is the single most important countermeasure, and you will need it for nearly every non-trivial saga.

In the order saga, the order row's `status` column *is* the semantic lock. The order is `PENDING` while the saga runs; only the final step flips it to `CONFIRMED`. Any other process — a "ship pending orders" job, a customer-facing "your orders" page — keys off `status` and does not treat a `PENDING` order as real. The compensation flips it to `CANCELLED`, which is itself a semantic lock for downstream readers ("this order will never be shipped").

```sql
-- Semantic lock via a status column. The inventory "reserved" count is a
-- semantic lock too: stock is held but not yet sold.
CREATE TABLE orders (
    id          UUID PRIMARY KEY,
    customer_id UUID NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PENDING'   -- PENDING | CONFIRMED | CANCELLED
                CHECK (status IN ('PENDING','CONFIRMED','CANCELLED')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A reader that must NOT act on in-flight orders filters them out:
SELECT * FROM orders WHERE status = 'CONFIRMED';  -- never sees PENDING/CANCELLED

-- The inventory reservation IS a semantic lock: 'reserved' stock is not 'available'
-- but is also not 'sold'. The oversell guard lives in this atomic conditional UPDATE.
UPDATE inventory
   SET available = available - 1,
       reserved  = reserved  + 1
 WHERE sku = '...'
   AND available >= 1          -- the guard: refuse to go negative, atomically
RETURNING available;            -- 0 rows affected => out of stock => fail the step
```

That `WHERE available >= 1` is doing enormous work: it makes the decrement *conditional and atomic within the inventory database*, so two concurrent reservations cannot both succeed when only one unit exists. This is the local-transaction isolation of the *inventory* database standing in for the missing saga-level isolation, applied exactly where the invariant lives. The semantic lock (the `reserved` count) plus the conditional update (the guard) is how you stop the oversell.

The cost of a semantic lock is that you have pushed the complexity onto every *reader*: everyone who queries the data must now understand the lock and handle the in-progress state. A reader that ignores `status` and treats a `PENDING` order as confirmed re-introduces the anomaly. Semantic locks also raise the possibility of deadlock between sagas (saga A holds order X pending and wants Y; saga B holds Y and wants X) — which you handle exactly as you would database deadlocks, with ordering and timeouts (see [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production)).

### 6.2 Commutative updates — make order not matter

If two operations on a resource **commute** — give the same result regardless of order — then the dirty-read/lost-update problem evaporates, because there is no "wrong order" to observe. The classic example is the account balance from figure 4. The lost update happened because the compensation *restored a remembered value* (`balance = 500`), which overwrites concurrent changes. If instead both the debit and its compensation are expressed as **relative deltas** — "subtract 100," "add 100 back" — then they commute with B's "subtract 50," and the final balance is correct no matter how the operations interleave:

```sql
-- NON-commutative (causes the lost update in figure 4): compensation overwrites.
UPDATE accounts SET balance = 500 WHERE id = ...;   -- restores a snapshot, erases B

-- COMMUTATIVE: express debit and its compensation as relative deltas.
UPDATE accounts SET balance = balance - 100 WHERE id = ...;  -- T: debit
UPDATE accounts SET balance = balance + 100 WHERE id = ...;  -- C: compensate
-- Now A's (-100, +100) and B's (-50) commute: final = 500 - 100 + 100 - 50 = 450. Correct.
```

Commutative updates are the cleanest countermeasure when they apply, because they require no locks and no reader awareness — the math just works out. They apply naturally to counters, balances, and any additive quantity. They do *not* apply when the operation is not additive (you cannot express "set the customer's tier to GOLD" as a commutative delta), which is why semantic locks remain the general-purpose tool.

### 6.3 Pessimistic view — reorder steps to limit exposure

The **pessimistic view** countermeasure reorders the saga's steps to minimize the business risk from the lack of isolation, by doing the things that *increase* risk as late as possible. Richardson's example: in a saga that both increases a customer's available credit and places an order, if the increase-credit step comes first, a concurrent saga could spend the increased credit on something else before the order completes, and a later compensation would leave the books wrong. Reorder so the order is placed first and the credit adjusted last, and the window of dangerous visibility shrinks. This is the pivot principle generalized: arrange the steps so that the most damaging intermediate states are the least likely to be observed.

### 6.4 Reread value ("by value") — detect and abort on concurrent change

The **reread value** countermeasure (Richardson also calls the routing variant "by value") has a step *reread* the data it depends on, immediately before acting, and abort or re-run if the data changed since it first read it. This is optimistic concurrency control — a compare-and-set — applied within a saga step. It is exactly the version-column pattern, and it composes beautifully with the outbox we will discuss next.

```sql
-- "Reread / by value": optimistic concurrency with a version column.
-- The saga step read version=7 earlier; it only commits if version is STILL 7.
UPDATE orders
   SET status = 'CONFIRMED', version = version + 1
 WHERE id = '...'
   AND version = 7;          -- 0 rows => concurrent modification => re-read & decide
-- This turns a silent lost update into a detectable conflict the saga can handle:
-- either retry against the new value, or abort and compensate.
```

The "by value" routing idea is a generalization: route the request differently based on the actual value of the data — for instance, a high-value order goes through an orchestrated saga with full compensation, while a low-value one uses a cheaper choreographed path. The countermeasure is "let the data's value drive the control flow so you only pay for isolation where the stakes justify it."

Here is the countermeasure decision table I keep in my head:

| Countermeasure | What it prevents | Cost | Reach for it when |
| --- | --- | --- | --- |
| **Semantic lock** (status flag) | Dirty reads, acting on in-progress state | Every reader must honor the flag; deadlock risk | Almost always — it's the baseline |
| **Commutative updates** (deltas) | Lost updates on additive data | Only works for additive quantities | Counters, balances, inventory counts |
| **Pessimistic view** (reorder) | Damage from observed intermediate state | Constrains step ordering | The risky step can be moved later |
| **Reread / by value** (version) | Lost updates via undetected concurrent change | Steps must carry & check versions | You can detect-and-retry instead of lock |
| **Pivot** (structural) | Compensating un-undoable actions | Forces forward-only after the pivot | There is an irreversible step |

The senior move is to use the *minimum* countermeasure that prevents the *specific* anomaly your business actually cares about. You do not need to defend against every theoretical interleaving; you need to defend against "we oversold a sneaker" and "we lost a customer's debit." Identify the invariants that matter, find the interleavings that violate them, and apply the cheapest countermeasure that closes each one.

## 7. Failure handling, idempotency, and exactly-once

Sagas live and die on **idempotency**, because the only delivery guarantee you can build cheaply in a distributed system is **at-least-once**: any command, event, or retry may be delivered more than once. AWS's saga guidance states the requirement directly: "Saga participants need to be idempotent to allow repeated execution in case of transient failures caused by unexpected crashes and orchestrator failures." If `charge_payment` runs twice because the orchestrator crashed after charging but before recording that it charged, you have double-charged a customer — unless the step is idempotent.

The standard construction is an **idempotency key**: a unique identifier for the *logical* operation (not the physical message), stored with a unique constraint so the second attempt is a no-op. Here is a fully idempotent payment step — the kind every saga participant needs.

```python
# idempotent_step.py — exactly-once EFFECT on top of at-least-once DELIVERY.
# The trick: a unique idempotency key + a results table. The first attempt does
# the work and records the result; every later attempt returns the recorded result.

def charge_payment_idempotent(ctx, db, psp):
    idem_key = f"charge:{ctx['order_id']}"   # logical id, stable across retries

    # 1. Fast path: did we already do this? Return the recorded result.
    row = db.fetchone(
        "SELECT result FROM idempotency_keys WHERE key = %s AND status = 'DONE'",
        (idem_key,))
    if row:
        return json.loads(row.result)        # no second charge

    # 2. Claim the key. If two attempts race, the unique constraint lets exactly
    #    one win; the loser falls through to step 1 on its retry.
    try:
        with db.begin() as tx:
            tx.execute(
                "INSERT INTO idempotency_keys (key, status) VALUES (%s, 'IN_PROGRESS')",
                (idem_key,))
    except UniqueViolation:
        # Another attempt is doing it (or finished). Re-read; if still in progress,
        # the caller will retry. Never proceed to charge here.
        row = db.fetchone(
            "SELECT result, status FROM idempotency_keys WHERE key = %s", (idem_key,))
        if row and row.status == 'DONE':
            return json.loads(row.result)
        raise StepInProgress(idem_key)

    # 3. We own the key. Do the side-effecting work. Pass idem_key to the PSP too,
    #    so even the EXTERNAL system dedupes if our process dies between charge and save.
    auth = psp.capture(order_id=ctx["order_id"],
                       amount_cents=ctx["amount_cents"],
                       idempotency_key=idem_key)   # Stripe, Adyen, etc. all support this

    # 4. Record the result so future attempts short-circuit.
    result = {"psp_ref": auth.ref, "amount_cents": ctx["amount_cents"]}
    with db.begin() as tx:
        tx.execute(
            "UPDATE idempotency_keys SET status='DONE', result=%s WHERE key=%s",
            (json.dumps(result), idem_key))
    return result
```

The subtle, load-bearing detail is step 3: passing `idem_key` to the *PSP*. There is still a window where your process charges the card, then crashes before writing `status='DONE'`. On retry, your local check finds the key `IN_PROGRESS`, and you would re-charge — *unless* the external system itself dedupes on the same key. Every serious payment API supports an idempotency key for exactly this reason. The general principle: **idempotency must extend to every external side effect, not just your own database.** A step is only as idempotent as its least idempotent side effect.

Compensations must be idempotent too — *especially* compensations, because the failure path is exactly where retries pile up. `release_stock` must be safe to run twice (releasing already-released stock is a no-op, enforced by the same status-flag check). A compensation that is not idempotent will, on a retry during compensation, double-release stock or double-refund, turning a clean rollback into a new corruption.

### Reliable event delivery: where the outbox comes in

For choreographed sagas, and for the command/reply channels of orchestrated ones, you need events to be published *reliably* — and this is precisely the problem the [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) post solves. The danger is the **dual write**: a saga step that commits its local transaction and *then* publishes an event as a separate operation can crash in between, leaving the database changed but the event never sent, which stalls the entire saga because the next service never hears about it.

The transactional outbox fixes this by writing the event into an `outbox` table *inside the same local transaction* as the business change, so the two commit atomically. A relay (or log-based CDC, à la Debezium) then publishes the outbox rows after commit, with at-least-once delivery — which is exactly why the downstream consumers must be idempotent, closing the loop. This connects directly to Kleppmann's *DDIA* Chapter 12, on derived data: the event stream is a *derivation* of the committed state, and as long as it is derived from a durable log of what actually committed, every downstream system — and every saga step waiting on it — eventually converges on the truth even through crashes and replays. The saga's correctness rests on this foundation: **reliable, ordered, at-least-once event delivery underneath, idempotent steps on top.**

```python
# A saga step that updates state AND emits its event ATOMICALLY via the outbox.
# No dual write: the UPDATE and the INSERT are one local transaction.
def reserve_stock_with_outbox(ctx, db):
    with db.begin() as tx:
        rows = tx.execute(
            "UPDATE inventory SET available = available - %s, reserved = reserved + %s "
            "WHERE sku = %s AND available >= %s RETURNING available",
            (ctx["qty"], ctx["qty"], ctx["sku"], ctx["qty"]))
        if not rows:
            raise OutOfStock(ctx["sku"])     # local txn rolls back; nothing emitted
        # Same transaction: enqueue the event that triggers the next saga step.
        tx.execute(
            "INSERT INTO outbox (aggregate_id, event_type, payload) VALUES (%s, %s, %s)",
            (ctx["order_id"], "StockReserved",
             json.dumps({"order_id": ctx["order_id"], "sku": ctx["sku"]})))
    # COMMIT. The relay/CDC publishes 'StockReserved' at-least-once after this.
```

## 8. Tooling: do not hand-roll the orchestrator in production

The hand-rolled orchestrator in section 2 exists to show you the moving parts. In production, you almost never want to maintain that yourself, because getting the durability, retries, timeouts, and history right is a multi-year systems-engineering effort that the saga frameworks have already done. Here is the landscape, with the actual tradeoffs.

| Tool | Model | State lives in | Compensation idiom | Best for |
| --- | --- | --- | --- | --- |
| **Temporal** / Cadence | Durable execution (code-as-workflow) | The workflow's own execution history (event-sourced) | `defer` / `try…catch…finally` in workflow code | Complex sagas where you want them expressed as ordinary code |
| **AWS Step Functions** | State machine (Amazon States Language, JSON) | The Step Functions service | Explicit `Catch` → compensating states | Serverless AWS shops; visual state machines |
| **Netflix Conductor** (Orkes) | JSON/code-defined workflows, central engine | Conductor's datastore | Failure-handling tasks in the workflow definition | Polyglot microservices, central workflow visibility |
| **Camunda 8 / Zeebe** | BPMN process models | Zeebe's append-only log (no central DB) | BPMN compensation events + handlers | Business-process-heavy domains; BPMN-literate teams |

### Temporal: durable execution makes the state machine disappear

[Temporal](https://temporal.io/blog/mastering-saga-patterns-for-distributed-transactions-in-microservices) takes the most radical and, to me, most pleasant approach: **durable execution**. You write the saga as ordinary code — a function that calls each step in sequence — and Temporal makes that function *durable*, meaning it survives process crashes by replaying its event history to reconstruct exactly where it was. The consequence for sagas is profound: you do not need a `saga_instances` table, because the workflow's execution state *is* the state, managed by Temporal. And compensation becomes a standard language construct — you register each compensation as you go and run them in a `finally`/`defer` on failure.

```python
# Temporal-style orchestration (Python SDK). No saga_instances table: the
# workflow's history IS the durable state. Compensations registered as we go.
from temporalio import workflow

@workflow.defn
class OrderSaga:
    @workflow.run
    async def run(self, order: dict) -> str:
        compensations = []   # LIFO stack of undos, run in reverse on failure
        try:
            order_id = await workflow.execute_activity(
                create_order, order, start_to_close_timeout=timedelta(seconds=30))
            compensations.append(lambda: workflow.execute_activity(
                cancel_order, order_id, start_to_close_timeout=timedelta(seconds=30)))

            await workflow.execute_activity(
                reserve_stock, order, start_to_close_timeout=timedelta(seconds=30))
            compensations.append(lambda: workflow.execute_activity(
                release_stock, order, start_to_close_timeout=timedelta(seconds=30)))

            # PIVOT: no compensation pushed after this point.
            await workflow.execute_activity(
                charge_payment, order, start_to_close_timeout=timedelta(seconds=60))

            # RETRIABLE: Temporal's retry policy rolls this forward automatically.
            await workflow.execute_activity(
                create_shipment, order,
                retry_policy=RetryPolicy(maximum_attempts=0))  # 0 = retry forever
            return "CONFIRMED"
        except Exception:
            # Backward recovery: run compensations in REVERSE registration order.
            for undo in reversed(compensations):
                await undo()   # each activity is idempotent + retried by Temporal
            raise   # marks the saga ABORTED
```

That `for undo in reversed(compensations)` is the paper's `Cⱼ … C₁` in five lines, and the durability that makes it crash-safe is entirely Temporal's responsibility. The retry policy on `create_shipment` is forward recovery made declarative. This is why durable-execution engines have largely won for new orchestrated sagas: the saga *reads like the business process*, and the hard distributed-systems parts are the engine's problem.

### AWS Step Functions: the state machine as JSON

Step Functions makes the state machine explicit and visual. You define forward states (`Place Order`, `Update Inventory`, `Make Payment`) and, via `Catch` clauses, route failures to compensating states (`Revert Inventory`, `Remove Order`). The AWS reference implementation wires it so that "if the workflow fails at the `Update Inventory` step, the orchestrator calls the `Revert Inventory` and `Remove Order` steps before returning a `Fail` state" — the reverse-order compensation, encoded as state transitions. The win is operational: Step Functions is a managed service with multi-AZ durability, so the orchestrator's single-point-of-failure concern is handled by AWS, and you get a visual execution history for free.

### Netflix Conductor and Camunda: central engines

[Netflix Conductor](https://github.com/Netflix/conductor) (now stewarded by Orkes) models workflows as explicit state machines composed of tasks, with workflow state persisted centrally — so "if a worker crashes or a task fails, Conductor retries the task based on predefined policies, using the persisted workflow state as the source of truth." It is a strong fit for polyglot shops that want one central place to define and observe long-running processes.

[Camunda 8 / Zeebe](https://camunda.com/blog/2025/06/how-a-bank-uses-compensation-events-camunda-8/) implements sagas through **BPMN compensation events**: you draw the process, link each activity to its undo task, and a compensation throw event "triggers the compensation within its scope and invokes all compensation handlers of completed activities" — BPMN's native expression of the reverse-order compensation chain. Zeebe is "distributed by design, has no central database, runs as peer-to-peer brokers," using a log-based, event-driven core (the same append-only-log idea as a write-ahead log) so the orchestrator itself is horizontally scalable. The fit is business-process-heavy domains where BPMN is already the lingua franca between engineers and analysts.

The meta-point: whatever you pick, the framework's job is to make the orchestrator's *state durable* and its *retries/timeouts/compensation* declarative. The saga semantics — local transactions, compensations, pivots, idempotency, semantic locks — are *yours* regardless of tool. No framework gives you back isolation; they give you a reliable place to put the control flow.

## 9. Saga versus 2PC: the tradeoff, stated honestly

![Sagas and two-phase commit make opposite tradeoffs: 2PC buys isolation at the cost of blocking and coupling, while a saga trades isolation away for availability and loose coupling](/imgs/blogs/saga-pattern-distributed-transactions-8.webp)

**A saga is not strictly better than 2PC; it makes the opposite tradeoff, and you should pick based on which property your operation cannot live without.** Figure 8 lays the two side by side across the dimensions that matter.

- **Atomicity**: 2PC gives true atomic commit — the intermediate state is never visible. A saga gives only *semantic* atomicity — the effects all happen or are all compensated, but the intermediate state *is* visible.
- **Isolation**: 2PC gives full ACID isolation. A saga gives *none* by default; you add semantic locks to get targeted isolation back.
- **Availability**: 2PC blocks on the coordinator and couples every participant's availability into a product. A saga stays available — each step is an independent local commit, and a slow service delays one saga, not the whole system.
- **Coupling**: 2PC is tight, lock-step coupling across heterogeneous resource managers. A saga is loose, async coupling via commands or events.
- **Operability**: 2PC needs an XA transaction manager and has a famously fragile recovery story (in-doubt transactions, manual intervention). A saga is application-level, testable, and recovers via retries and compensations you control.

The honest summary: **reach for 2PC when you genuinely need isolation and atomicity and can pay the availability/operability cost — and you are within a single trust/admin domain with XA-capable resources** (e.g., two databases under one DBA, briefly). Reach for a saga — which is to say, almost always, in a microservices architecture — **when you need availability and loose coupling and can engineer around the lack of isolation with countermeasures.** The microservices world chose sagas not because they are simpler (they are not — they push real complexity into your application) but because 2PC's availability and operability costs are intolerable across independently-owned services, exactly as Kleppmann's *DDIA* Chapter 9 argues when it explains why distributed transactions are avoided in practice.

## 10. A practical "design a saga" playbook

Here is the checklist I actually run when designing a saga, in order. It is the distillation of everything above into a procedure.

1. **List the steps as local transactions.** Each `Tᵢ` must be a single ACID transaction in one service's database. If a "step" touches two databases, it is two steps. Write them down as a sequence.
2. **Write each compensation `Cᵢ` — or prove it cannot exist.** For every step, write the compensation that semantically undoes it. The moment you hit a step with no compensation (send email, ship physical goods, capture funds you will not auto-refund), you have found a candidate **pivot**.
3. **Place the pivot.** Reorder the steps so that everything compensatable comes *before* the irreversible step, and the irreversible step is the commit point. Everything after the pivot must be **retriable** (forward-recoverable). If you cannot reorder so the pivot is late, your saga is fragile — rethink the decomposition.
4. **Classify every step** as compensatable, pivot, or retriable (figure 5). This tells you the recovery direction at each point: backward before the pivot, forward after.
5. **Identify the business invariants and the anomalies that threaten them.** "Don't oversell," "don't lose a debit," "don't double-charge." For each, find the concurrent interleaving that violates it (figure 4 is the template).
6. **Apply the minimum countermeasure per anomaly.** Semantic lock (status flag) as the baseline; commutative deltas for additive data; reorder (pessimistic view) where you can; version checks (reread/by value) where detect-and-retry beats locking. Do not over-engineer isolation you do not need.
7. **Make every step and every compensation idempotent.** Idempotency key + results table; extend the key to every external side effect (PSP, email provider). At-least-once delivery is the only delivery you get.
8. **Choose coordination.** Choreography only if ≤ ~4 loosely-coupled steps with no branches; orchestration otherwise. If you are unsure, orchestrate — it is the default for real systems.
9. **Make the coordination durable.** Pick a framework (Temporal, Step Functions, Conductor, Camunda) rather than hand-rolling the orchestrator's persistence, retries, and timeouts. Ensure event delivery is reliable via the transactional outbox + CDC.
10. **Instrument it.** Every saga instance gets a correlation id; emit a span per step and per compensation; alert on sagas stuck in a non-terminal state past a deadline. A saga you cannot observe is a saga you cannot debug, and they fail in subtle ways.

> The saga is not "distributed transactions made easy." It is "we accept that there is no distributed transaction, and we will engineer atomicity and just-enough isolation by hand." The playbook is how you do that without leaving a hole.

## Case studies from production

These are composites — the names are invented, the failures are real, drawn from incidents I have worked or reviewed. Each one is a specific way a saga goes wrong, and the lesson generalizes.

### 1. The oversold sneaker (the dirty read)

The symptom that opened this article: two customers bought the last unit. The wrong first hypothesis was a database bug — surely the count went negative because of a missing constraint. The actual root cause was that the order saga decremented inventory in a step that did a plain `UPDATE inventory SET available = available - 1` *without* the `available >= 1` guard, and the read of `available` happened in an earlier step, in a different service, with no semantic lock between them. Two sagas both read `available = 1`, both decremented, both charged. The fix was two lines: make the decrement a *conditional atomic* update (`WHERE available >= 1 RETURNING`), so the inventory database's own isolation enforces the invariant at the exact point it lives, and treat zero rows affected as an out-of-stock failure that aborts the saga. The lesson: **the guard must be atomic and in the same local transaction as the decrement**; a read-then-write split across steps is a lost update waiting to happen.

### 2. The double charge (idempotency that stopped at the database)

A payment saga step was idempotent against its own database — it had a `charges` table with a unique constraint on `order_id` — but the engineer assumed that protected against double charging. It did not. The orchestrator crashed *after* the PSP captured the funds but *before* the local `UPDATE charges SET status='CAPTURED'` committed. On restart, the step re-ran, saw no `CAPTURED` row, and called the PSP again. The PSP, having no idempotency key, happily charged the card a second time. The fix was to pass an idempotency key to the PSP so the *external* system deduped. The lesson, stated earlier and worth repeating: **a step is only as idempotent as its least idempotent side effect.** Your database constraint protects your database; it does nothing for the third party.

### 3. The un-cancellable shipment (a missing pivot)

An order saga reserved stock, created the shipment, *then* charged the card — in that order. When a card was declined at the last step, the compensation tried to cancel the shipment, but by then the warehouse's automation had already picked, packed, and handed the package to the carrier. There was no `C` for "un-ship." The team had built a compensation that was physically impossible to execute. The fix was a reorder: charge the card (the pivot) *before* creating the shipment, so the shipment lives entirely in the retriable zone and only happens on the success path. The lesson: **structure the saga so irreversible actions come after the pivot.** A compensation you cannot actually run is not a compensation.

### 4. The lost debit (non-commutative compensation)

A wallet service ran a saga that debited a balance and, on failure, "rolled back" by writing the *old balance* it had snapshotted. Exactly figure 4. A concurrent top-up committed in between, and the compensation's snapshot write erased it. The fix was to express both the debit and its compensation as relative deltas (`balance - amount`, `balance + amount`) so they commute with concurrent changes. The lesson: **a compensation that restores a remembered value is a lost-update generator; compensations on shared mutable quantities must be commutative deltas, not snapshot restores.**

### 5. The choreography cycle (the saga nobody designed)

A four-service choreographed saga grew, over a year, a fifth subscription that closed a loop: service A emitted an event E reacted to, and a new feature made E emit an event A reacted to under a condition that was *usually* false. One day a data anomaly made it true, and two services ping-ponged events at each other until the broker's retention filled. Nobody had designed this loop; it emerged from independent subscription changes. The fix was twofold: introduce an orchestrator to make the flow explicit and acyclic, and add a hop-count / saga-id guard so an event carrying a saga it had already visited was dropped. The lesson: **choreography's implicit flow makes cycles invisible until they fire; past a handful of participants, the lack of a single readable process definition is itself the bug.**

### 6. The orphaned saga (non-durable orchestrator state)

A hand-rolled orchestrator kept its state machine in memory and only wrote to the database at the start and the end. A deploy rolled the orchestrator pods mid-saga. Dozens of orders were left with stock reserved, payment charged, and no shipment — and nobody driving them, because the in-memory state died with the pods. Stock stayed reserved for hours until a human noticed `reserved` counts that did not match any active order. The fix was to checkpoint state after *every* transition (the `_save` in section 2's code) so a restarted orchestrator resumes, and ultimately to move to Temporal so durability was the engine's job. The lesson: **an orchestrator that does not persist after every transition will orphan sagas on every crash or deploy** — and deploys are far more frequent than crashes.

### 7. The compensation storm (non-idempotent undo)

During a regional outage, an orchestrator's connection to the inventory service flapped, so its `release_stock` compensation timed out and was retried many times. `release_stock` was *not* idempotent — it did `available = available + qty` unconditionally — so each retry added stock back again. By the time the dust settled, inventory counts were inflated by thousands of units, and the system happily oversold for a week before reconciliation caught it. The fix: make `release_stock` idempotent by guarding on the reservation's status (`WHERE reservation_status = 'RESERVED'`, flip to `'RELEASED'`), so a second release is a no-op. The lesson: **compensations are exactly where retries concentrate, so a non-idempotent compensation is more dangerous than a non-idempotent forward step.** The failure path must be the most robust part of the saga, not the least.

### 8. The non-repeatable price (a long saga and a concurrent change)

A subscription-upgrade saga read the plan price in step 1, did several slow steps (provisioning, notifying a partner), and charged the price in step 5. A price change committed between step 1 and step 5, and customers were charged the new price for a plan they had agreed to at the old price — a non-repeatable read across the saga's lifetime. The fix was to **capture the price into the saga's context at step 1 and charge the captured value**, never re-reading it, plus a version check so a fundamentally invalid plan aborted the saga. The lesson: **a long saga must snapshot the values it depends on into its own durable context at the moment of decision**, because the world will change underneath a saga that takes seconds or minutes, and re-reading mutable data mid-saga reintroduces non-repeatable reads.

### 9. The stuck pivot (timeout with no owner)

A choreographed saga's payment service hung — not failed, hung — and emitted neither success nor failure. Because choreography has no natural timeout owner, *nobody* noticed; the order sat `PENDING` with stock reserved indefinitely. The customer's card was eventually charged (the request had gone through; only the response was lost), but the saga never advanced because the `payment.captured` event was never emitted. The fix was a saga-timeout sweeper that, after a deadline, queried in-flight sagas and reconciled them against the payment service's actual state via an idempotent "get status by order id" call — and, longer term, a move to orchestration where the orchestrator owns the timeout. The lesson: **somebody must own the timeout for every step**; in choreography that owner does not exist by default, and building it is orchestration in disguise.

### 10. The compensation that needed a compensation

An orchestrator, while compensating a failed saga, hit a failure *in a compensation*: the refund step (a compensation for the charge) was declined by the PSP because the original charge had already been disputed by the customer and the funds were frozen. The orchestrator had no plan for "compensation failed," so it gave up, leaving the saga half-compensated: order cancelled, stock released, but money not returned. The fix was to make compensations *retriable with escalation* — retry the refund, and if it permanently fails, route to a human-review queue with full context rather than silently abandoning it. The lesson: **compensations can fail too, and a saga's design is incomplete until you have answered "what happens when a compensation fails?"** The answer is usually "retry, then escalate to a human" — never "give up silently."

## When to reach for a saga, and when not to

### Reach for a saga when

- A business operation must update data in **two or more services / databases** that you cannot put in one local transaction, and you have adopted database-per-service.
- You need the operation to stay **available** even when individual services are slow or briefly down — you cannot accept the all-participants-up coupling of 2PC.
- The steps can each be a **local ACID transaction**, and you can write a **semantic compensation** for the reversible ones (or structure the irreversible one as a late pivot).
- You can engineer around the lack of isolation: you know your business invariants and the **anomalies** that threaten them, and a **semantic lock** (or commutative update, or version check) closes each one.
- The operation is **long-running** — it spans seconds, minutes, or human approvals — where holding 2PC locks would be absurd and a durable orchestrator is the natural home.

### Skip the saga when

- The whole operation fits in **one database**. Then it is one transaction. Do not distribute a transaction that does not need distributing — a saga is strictly more complex and strictly less isolated than a `BEGIN…COMMIT`. This is the most common over-engineering I see.
- You genuinely **need full isolation** and the operation is short and within one administrative domain with XA-capable resources. Then 2PC, despite its costs, may be the honest choice — do not contort a saga to fake isolation it cannot provide.
- There is **no meaningful compensation** for the early steps and you cannot reorder to make the irreversible action a late pivot. A saga whose compensations are physically impossible is a saga that cannot recover — rethink the decomposition or the service boundaries.
- The "saga" is really a **single read-modify-write** that you are splitting across services for no reason. Co-locate the data; do not invent a distributed transaction to justify a service boundary that should not exist.
- You cannot make the steps **idempotent**. At-least-once delivery is non-negotiable in distributed systems; if a step's side effect cannot be made idempotent (and cannot be wrapped in an external idempotency key), a saga will eventually duplicate that effect.

The saga pattern is one of those ideas that is simple to state — local transactions plus compensations — and genuinely hard to get right, because all the difficulty is in the part the one-line summary omits: there is no isolation, and you have to engineer it back in, anomaly by anomaly, with semantic locks and commutative updates and carefully-placed pivots and relentlessly idempotent steps. Garcia-Molina and Salem saw the shape of it in 1987 for long-lived database transactions; the microservices era rediscovered that the same structure is the only sane way to make a business operation span services without coupling their commits. The frameworks — Temporal, Step Functions, Conductor, Camunda — have made the *durable orchestration* part a solved problem you can buy. The *semantics* part is still yours, and it always will be, because no engine can know which intermediate states your business can tolerate other people seeing. That judgment — which anomalies matter, and the cheapest countermeasure that closes each — is the actual skill of designing a saga, and it is worth more than any framework.

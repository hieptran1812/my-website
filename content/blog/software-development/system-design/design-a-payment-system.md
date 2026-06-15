---
title: "Design a Payment System: Ledgers, Idempotency, and Getting Money Right"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Design a payment system where correctness beats scale — a double-entry append-only ledger, idempotency on every mutating call, the outbox pattern for the dual-write problem, reconciliation against the processor, and how a senior reasons about money that must never be lost or duplicated."
tags:
  [
    "system-design",
    "payments",
    "ledger",
    "idempotency",
    "double-entry",
    "reconciliation",
    "architecture",
    "distributed-systems",
    "scalability",
    "consistency",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/design-a-payment-system-1.webp"
---

Most system design interviews reward scale. Design a feed for a billion users, a URL shortener doing a million writes a second, a chat system fanning out to ten thousand group members. The reflexes those problems train — shard early, cache aggressively, relax consistency until the latency number looks good — are exactly the reflexes that will get you fired if you bring them to a payment system. Here, the failure mode is not a slow page. It is a customer charged twice for a fifty-dollar order, or a merchant paid out twice for one sale, or a balance that says \$0 when the bank says \$10,000 and nobody can explain the gap. Payments is the one design where **correctness dominates scale**, where you would rather be slow and right than fast and wrong, and where "eventually consistent" is a phrase that ends careers when applied to a balance.

The hard part is not the throughput. A payment system at serious scale handles thousands of transactions per second, not millions — Stripe processes on the order of a few thousand TPS at peak, which is a rounding error compared to a social feed. The hard part is that every single one of those transactions moves real money through systems you do not control, over networks that drop packets, with retries you did not authorize, and you have to come out the other side able to **prove** — to an auditor, to a regulator, to an angry customer — exactly where every cent went and why. The whole design is organized around one question that a feed system never has to ask: *if this operation runs twice, or half-completes, or times out in an unknown state, does the books still balance?*

This post designs a payment system the way a senior would defend it in a review where the CFO is in the room. We will build a double-entry, append-only ledger and refuse to ever `UPDATE` a balance. We will put an idempotency key on every mutating call because the network *will* retry and we do not get a vote. We will confront the dual-write problem between our database and the external processor — the single nastiest bug in payments — and solve it with the outbox pattern plus reconciliation. We will demand strong, linearizable consistency for balances and explain exactly why eventual consistency is malpractice here. And we will stress-test the design against the three incidents that actually take down payment systems: a processor timeout where you do not know if the charge went through, a customer double-submit, and a network partition in the middle of moving money. Figure 1 is the spine of the whole thing — the payment state machine — and we will keep coming back to it.

![The payment state machine showing a charge moving from authorize through capture to settle with a separate terminal failed state](/imgs/blogs/design-a-payment-system-1.webp)

By the end you should be able to do five concrete things. Design a ledger schema you can defend to an auditor. Reason about idempotency well enough to explain what happens when two retries of the same charge arrive in the same millisecond. Diagnose and fix a dual-write inconsistency between your books and the processor. Run a reconciliation that finds drift before a customer does. And size the system — TPS, ledger growth, idempotency-store footprint — with back-of-the-envelope math a staff engineer would accept. The mechanism deep-dives this rests on — [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design), [consistency models](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects), and the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — already exist; this post is the architect's decision layer that wires them into a system that gets money right.

## 1. Framing and scope: what "payment system" even means

The first senior move is to refuse to design "a payment system" in the abstract, because the term covers three wildly different machines. A **payment gateway / charge service** takes a card and moves money from a customer to a merchant — this is the Stripe-charges surface, and it is what most people mean. A **wallet / ledger service** holds balances for users (think PayPal balance, a ride-hailing driver's earnings, an exchange's account) and moves money *between internal accounts* — no card touched, just debits and credits in your own books. A **payouts / disbursement service** moves money *out* to a bank account or card. Real systems combine all three, but the design pressures differ, and conflating them in a review signals you have not built one. We will design the core that all three share — the **ledger** — and then layer charging and payout on top, because the ledger is where correctness lives and everything else is plumbing around it.

Scope the functional requirements out loud, and keep them short. A user can fund their balance (pay in), spend it (move money to a merchant), and withdraw it (pay out). The system records every movement in an immutable ledger. Every balance is derivable from the ledger. Every external interaction with a card network or bank goes through a processor we do not control. That is the whole functional surface. Notice what is *not* here: no recommendations, no search, no feed ranking. Payments is functionally small and operationally enormous, which is the opposite of a feed.

Now the non-functional requirements, and this is where payments diverges hard from everything else in this series. **Correctness is the top-line SLO**, not latency and not availability. The system must never lose money, never create money, and never double-spend. Concretely: debits must always equal credits (the books balance to the cent, always), no charge may be applied twice for one intent, and no balance may go negative unless explicitly allowed (an overdraft is a product decision, not an accident). **Durability is absolute** — a committed money movement survives any single failure, full stop; we do not lose a confirmed transaction to a crashed node. **Auditability is a hard requirement** — for any balance at any past moment, we can produce the exact list of entries that produced it, because regulators and disputes demand it. **Consistency must be strong (linearizable) for balances** — when a debit commits, every subsequent read sees it, because a stale read here means a double-spend. Only *then* do we talk about latency (a charge should feel synchronous, p99 under a couple of seconds including the processor) and throughput (a few thousand TPS, which is modest).

The ranking matters because it dictates every trade-off downstream. When correctness and availability conflict — the [CAP theorem](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) says they will, during a partition — a payment system chooses **consistency and refuses the write**. A social feed chooses availability and shows a stale post. Getting this priority backwards is the single most common way to fail a payments design, so state it first and never waver: *we would rather reject a payment than process it wrong.*

## 2. Money representation: integers, never floats

Before any architecture, settle the most embarrassing way to lose money: floating-point. The number `0.1` cannot be represented exactly in IEEE 754 binary floating point. Add `0.1 + 0.2` in almost any language and you get `0.30000000000000004`. Multiply that error across millions of transactions and you have a ledger that does not balance, a reconciliation that never closes, and an auditor asking why \$0.000000004 went missing across the books. The rule is absolute and non-negotiable: **never represent money as a float. Ever.**

Represent money as an **integer count of the currency's minor unit** — cents for USD, pence for GBP, yen for JPY (which has no minor unit, so the integer *is* yen). \$49.99 is stored as the integer `4999`. A charge of \$1,000.00 is `100000`. Integers add, subtract, and compare exactly; there is no rounding error because there is no fraction. The only place rounding enters is when you *divide* (splitting a fee, computing a percentage), and there you make the rounding policy explicit and you account for the remainder so it does not vanish — the leftover cent goes somewhere on the books, never into the ether.

Currency is not optional metadata; it is part of the amount. An amount of `4999` is meaningless without knowing it is USD, because `4999` JPY is about \$33 while `4999` USD is \$49.99. Every monetary value in the system is a pair — `{amount: integer, currency: "USD"}` — and you **never add two amounts in different currencies** without an explicit conversion that itself produces a ledger entry. A multi-currency ledger keeps each currency's accounts separate and treats foreign-exchange as a movement between them, with the rate and the spread recorded as their own entries. The schema below makes the representation concrete.

```sql
-- Money is always integer minor units + an explicit currency.
-- amount_minor is BIGINT: cents, never a DECIMAL of dollars, never a float.
CREATE TYPE money AS (
    amount_minor  BIGINT,      -- 4999 means $49.99 (USD) or ¥4999 (JPY)
    currency      CHAR(3)      -- ISO 4217: 'USD', 'GBP', 'JPY'
);

-- A 64-bit signed integer holds up to ~9.2e18 minor units.
-- That is ~92 quadrillion dollars. You will not overflow it.
-- If you somehow run a national treasury, use NUMERIC(38,0). Still never float.
```

For languages, the same discipline applies: use a 64-bit integer or a fixed-point decimal type (`BigDecimal` in Java, `decimal.Decimal` in Python with a fixed context, `int64` of cents in Go), and forbid `float`/`double` for money in code review with a linter rule if you can. This is the cheapest correctness win in the entire system and the one most likely to be skipped by someone who has never been burned. Get it wrong and nothing downstream can save you, because every ledger entry is built on it.

## 3. High-level architecture: the components and how money flows

Before the deep dives, sketch the whole machine so the pieces have a home, because in a payment system the *boundaries* between components are where the correctness arguments live. Figure 3 is the high-level architecture: a payment API at the front, an append-only ledger as the system of record, an outbox written in the same transaction as the ledger, a settlement worker that drives the external processor, and a webhook handler that brings the processor's async confirmations back into the ledger. Every arrow in that diagram is a place money could be lost if the boundary is crossed carelessly, and the rest of this post is essentially a defense of each boundary.

![A high-level payment architecture where the API writes the ledger and an outbox in one transaction while a worker drives the external processor and webhooks confirm back into the ledger](/imgs/blogs/design-a-payment-system-3.webp)

Trace a charge through it. A request hits the **payment API**, carrying an idempotency key (section 6). The API, in **one local database transaction**, appends the ledger entries (debit + credit) *and* writes a row to the **outbox** table that says "a processor capture must happen." That single transaction is the atomicity boundary that makes the whole design safe — it either commits both the ledger and the intent-to-call-the-processor, or neither (section 7). The API returns quickly; the user sees "processing." Asynchronously, the **settlement worker** polls the outbox, calls the **external processor** (Stripe, a bank, a card network — something you do not control, drawn as external), and the processor eventually confirms via a **webhook** to a handler that verifies the signature, dedups the event, and writes the settlement back into the ledger. A **reconciliation job** (section 10) runs on a schedule outside this hot path, comparing the ledger against the processor's report to catch anything the live flow missed.

The component responsibilities are deliberately narrow, because narrow components are ones you can reason about. The **API** validates, claims the idempotency key, and writes ledger+outbox atomically — it does *not* call the processor synchronously, because you must never hold a database transaction open across a slow external call. The **ledger** is the single source of truth and is append-only; nothing mutates a balance. The **outbox** is the bridge between your atomic local world and the non-atomic external world. The **settlement worker** owns all processor communication and all the retry/timeout logic (section 9), so the ambiguity of external calls is contained in one place rather than smeared across the codebase. The **webhook handler** is the only inbound path from the processor and is idempotent by construction. And **reconciliation** is the auditor that trusts none of them and verifies everything against an external report. Keep these boundaries crisp and the correctness proofs stay local; blur them — call the processor from the API, mutate a balance in the worker — and the proofs fall apart.

Notice what is *not* in the architecture: no cache in front of balances on the write path (a stale cached balance is a double-spend, so balance reads that gate a spend go to the leader, not a cache), and no message broker strictly required (the outbox table *is* your durable queue; you can publish from it to Kafka if you want fan-out, per [queues and event streaming](/blog/software-development/system-design/queues-and-event-streaming-for-architects), but the table itself is enough). Payments architecture is conspicuously *simple* in its component count and conspicuously *careful* in its boundaries — the opposite of a feed, which has many components and relaxed boundaries. The simplicity is the point: fewer places for money to leak.

## 4. The double-entry ledger: append-only, immutable, derivable

Here is the heart of the design, and the one idea that, if you internalize nothing else, will keep you out of the worst trouble: **you never store a balance. You store entries, and you derive the balance.** A balance is not a fact you maintain; it is a *query* over an immutable log of movements. Figure 2 contrasts the two worlds — the naive mutable-balance design that overwrites a number in place and loses all history, against the append-only ledger that only ever inserts rows and computes the balance as a sum.

![A before and after comparison contrasting a mutable balance that is overwritten in place against an append-only ledger that appends immutable debit and credit rows from which the balance is derived](/imgs/blogs/design-a-payment-system-2.webp)

The accounting world solved this problem five hundred years ago with **double-entry bookkeeping**, and it is not a quaint tradition — it is a correctness invariant enforced by structure. Every money movement is recorded as *at least two* entries: a **debit** to one account and a **credit** to another, equal in magnitude. Money is never created or destroyed; it only *moves* between accounts, and every movement touches two sides. The invariant that falls out of this is the one you check on every transaction and in every reconciliation: **the sum of all debits equals the sum of all credits.** If that ever fails to hold, money has appeared or vanished, and you have a bug you must halt and fix, not paper over.

Think of accounts as buckets and a transaction as a transfer that takes from one bucket and puts into another, recorded as one debit and one matching credit. A user funding their wallet with \$50 is two entries: debit the "external funding source" account \$50, credit the user's wallet account \$50. The user spending \$30 at a merchant is debit the user's wallet \$30, credit the merchant's account \$30. The user's balance is just the sum of credits minus debits on their account — at this point, \$50 in minus \$30 out equals \$20. You did not `UPDATE balance = 20`. You appended entries, and \$20 is what the math says.

Why does append-only matter so much? Three reasons, each of which has burned a real company. **First, audit.** An immutable log means you can answer "what was this user's balance on March 3rd at 2pm and what entries produced it?" — you replay the entries up to that timestamp. An overwritten balance cannot answer that; the history is gone. **Second, concurrency safety.** `UPDATE balance = balance - 30` is a read-modify-write, and two of them racing can lose an update — both read \$50, both write \$20, and \$30 vanishes. An `INSERT` of a new entry never overwrites another `INSERT`; appends do not race in the destructive way updates do. **Third, correctness debugging.** When something is wrong, an append-only log lets you find *exactly which entry* was wrong and append a correcting entry. You **never delete or edit a ledger row** — a mistake is fixed by appending a reversing entry, the same way accountants do it. The ledger is a permanent record of what happened, including the mistakes and their corrections.

The schema makes the discipline structural. Entries are immutable (no `UPDATE`, no `DELETE` — enforce it with permissions and triggers if you can), grouped into transactions that must balance, and reference accounts.

```sql
CREATE TABLE accounts (
    id            BIGINT PRIMARY KEY,
    owner_id      BIGINT NOT NULL,          -- user, merchant, or a system account
    account_type  TEXT   NOT NULL,          -- 'user_wallet', 'merchant', 'external_funding'
    currency      CHAR(3) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    -- Note: NO balance column. The balance is derived, never stored here.
);

CREATE TABLE ledger_transactions (
    id            BIGINT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,            -- one transaction per intent (see section 6)
    kind          TEXT NOT NULL,            -- 'fund', 'spend', 'payout', 'reversal'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ledger_entries (
    id            BIGINT PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES ledger_transactions(id),
    account_id    BIGINT NOT NULL REFERENCES accounts(id),
    direction     TEXT   NOT NULL CHECK (direction IN ('debit','credit')),
    amount_minor  BIGINT NOT NULL CHECK (amount_minor > 0),
    currency      CHAR(3) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Entries are append-only: revoke UPDATE and DELETE from the app role.
-- Every ledger_transaction must have debits summing to credits, same currency.
```

Deriving a balance is a sum, and you make it correct and fast with the two techniques in the next section. But the conceptual model is fixed now and we will not revisit it: **the ledger is the source of truth, it is append-only, debits equal credits, and the balance is a derived quantity.** Everything else in the system exists to feed this ledger correctly and to verify it against the outside world.

One design decision worth making explicit before moving on is the **chart of accounts** — the set of account *types* your ledger uses — because getting it right is what makes the double-entry invariant actually catch bugs rather than just decorate them. A clean payment ledger has at least these account families: **user/customer accounts** (one per customer wallet), **merchant accounts** (one per merchant), **external accounts** that represent the boundary with the outside world (a single "external funding" account that money enters the system through, and an "external payout" account it leaves through), and **system accounts** for things like fees, reserves, and rounding. The external accounts are the trick that makes the books balance even though money genuinely crosses the system boundary: when \$100 enters from a customer's card, you do not create \$100 from nothing — you *debit the external-funding account* (which goes increasingly negative, representing "the outside world owes us this much, which we've taken in") and credit the user. The external account is the sink that absorbs the asymmetry, so that even at the boundary, debits equal credits. Without it, every pay-in would unbalance the books, and you would be tempted to "just add a balance," which is exactly the trap.

A second decision is whether to model a money movement as **two entries or many**. The minimal case is one debit and one credit, but real movements often touch three or more accounts at once — a \$50 charge that nets to the merchant might be: credit merchant \$48.55, credit fee account \$1.45, debit user \$50.00. Three entries, still balanced (one debit of \$50, two credits summing to \$50). The double-entry invariant generalizes cleanly: a *transaction* is balanced if its debits sum to its credits, regardless of how many entries it has. This is why the schema groups entries under a `ledger_transactions` row and enforces the balance per transaction, not per pair — a senior models the fee, the tax, and the principal as separate entries in one balanced transaction, so each shows up distinctly in the audit trail and in reconciliation, rather than smushing the fee into the principal and losing the ability to reconcile the fee independently.

#### Worked example: funding, spending, and reversing on the ledger

Walk a single user through three movements and watch the books stay balanced. The user **funds** \$100: transaction T1 appends two entries — debit `external_funding` 10000, credit `user_wallet` 10000. Debits (10000) equal credits (10000); the books balance. The user's wallet balance is now credits minus debits on their account: 10000 − 0 = \$100. The user **spends** \$30 at a merchant: transaction T2 appends debit `user_wallet` 3000, credit `merchant_acct` 3000. Books balance again. The user's wallet is now 10000 credited minus 3000 debited = \$70. Now the merchant **disputes** and we **reverse** the \$30: we do *not* delete T2. We append T3 — debit `merchant_acct` 3000, credit `user_wallet` 3000 — the exact mirror. The user's wallet is back to 10000 − 3000 + 3000 = \$100, the merchant nets to zero, and the audit trail shows *all three* transactions: the spend, and its explicit reversal, with timestamps. An auditor can see exactly what happened and when. That is what append-only buys you: the mistake and its correction both live in the record, forever, and at every step total debits equal total credits.

## 5. Deriving balances fast: snapshots and the read/write split

The obvious objection to "never store a balance, always sum the entries" is performance: summing a million entries on every balance read is absurd. The senior answer is **snapshots** (also called balance checkpoints): you periodically materialize the running balance as of a specific entry, and a balance read becomes "the latest snapshot plus the entries appended since." You get the audit guarantees of append-only *and* O(small) reads. This is exactly the pattern [event sourcing](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) uses — the event log is the truth, and snapshots are a derived optimization you can always rebuild by replaying.

A snapshot is itself an append-only row: `{account_id, as_of_entry_id, balance_minor, created_at}`. It is a *cache*, not a source of truth — if a snapshot is ever wrong, you discard it and recompute from entries, because the entries are the truth and the snapshot is just a fast path. To read a balance: take the latest snapshot for the account, then sum the entries with `id > as_of_entry_id`. If you snapshot every 1,000 entries, a balance read sums at most 1,000 rows on top of one snapshot lookup — fast and exact. You can snapshot more aggressively for hot accounts and lazily for cold ones.

```python
def get_balance(account_id: int) -> Money:
    snap = db.query_one(
        "SELECT balance_minor, as_of_entry_id FROM balance_snapshots "
        "WHERE account_id = %s ORDER BY as_of_entry_id DESC LIMIT 1",
        account_id)
    base = snap.balance_minor if snap else 0
    since = snap.as_of_entry_id if snap else 0

    delta = db.query_one(
        "SELECT COALESCE(SUM(CASE WHEN direction='credit' THEN amount_minor "
        "ELSE -amount_minor END), 0) AS d "
        "FROM ledger_entries WHERE account_id = %s AND id > %s",
        account_id, since).d
    return Money(base + delta, snap.currency)
```

This split — append-only writes, snapshot-accelerated reads — is the foundation of the optimization story we will return to in section 12. It also clarifies the consistency requirement: the *write* path (appending entries and the snapshot used to enforce "do not overdraw") must be linearizable, while a *reporting* read of a historical balance can tolerate replica lag. We will make that distinction precise with the decision tree in figure 8. For now, hold the shape: writes are strongly consistent and append-only; reads are accelerated by snapshots and can be relaxed when they are not gating a money movement.

## 6. Idempotency everywhere: the network retries and you do not get a vote

A payment API has one property that a feed API does not: **every mutating call must be idempotent, because every mutating call will be retried whether you like it or not.** A client's HTTP library retries on a connection reset. A load balancer retries on a 502. A user clicks "Pay" twice because the spinner hung. A mobile app resends a queued request when the network comes back. Each retry is correct behavior for the component doing it, and each one means your charge endpoint runs more than once. If running it twice charges the customer twice, you do not have a hypothetical bug — you have a bug scheduled for the next network hiccup under load. Figure 4 shows the flow that defuses this: an **idempotency key** the client generates once per intent and resends on every retry, so the first request does the work and every retry replays the stored result.

![An idempotency flow where the first request claims a key and performs the charge while every retry with the same key replays the stored response](/imgs/blogs/design-a-payment-system-4.webp)

The pattern, which Stripe popularized and most serious payment APIs now copy, works like this. The client generates a unique key per logical operation — a UUID is fine — and sends it in a header: `Idempotency-Key: ic_8f2a91`. The server, before doing any work, tries to **atomically claim** that key by inserting a row into an idempotency store. If the insert succeeds, this is the first time we have seen this key; do the work, store the response against the key, return it. If the insert *fails* with a unique-constraint violation, this key has been seen before; **do not redo the work** — look up the stored response and replay it. The retry gets the exact same answer as the original, the charge happens exactly once, and the customer is billed once. The mechanism in full — request fingerprinting to catch key reuse with a different body, TTL sizing, race handling — is the subject of the [idempotency deep-dive](/blog/software-development/system-design/idempotency-and-exactly-once-by-design); here we wire it into the payment flow.

The subtle part, and the question a good reviewer will ask, is: *what happens when two retries of the same key arrive in the same millisecond?* If both check "does this key exist?" before either has written, both see "no," and both do the work — a double charge. The fix is to never check-then-act; instead **make the claim atomic** by relying on a unique constraint at the database level. The `INSERT ... ON CONFLICT DO NOTHING` (or a unique index that throws) is the synchronization point — exactly one of the racing requests wins the insert and does the work; the loser gets a conflict and must wait for and replay the winner's stored result. The database's unique index, not application logic, is what serializes the race. This is why the `idempotency_key` column on `ledger_transactions` has a `UNIQUE` constraint: it is the lock.

```python
def charge(idempotency_key: str, request_body: dict) -> Response:
    # Atomic claim. The UNIQUE index is the synchronization point.
    claimed = db.execute(
        "INSERT INTO idempotency_keys (key, request_hash, status) "
        "VALUES (%s, %s, 'pending') ON CONFLICT (key) DO NOTHING",
        idempotency_key, sha256(request_body))

    if claimed.rowcount == 0:
        # Key already exists: this is a retry. Replay, never redo.
        row = db.query_one("SELECT request_hash, status, response "
                           "FROM idempotency_keys WHERE key = %s", idempotency_key)
        if row.request_hash != sha256(request_body):
            raise Conflict("idempotency key reused with a different body")
        if row.status == 'pending':
            # The original is still in flight. Do NOT start a second charge.
            raise Retry("original request in progress; retry shortly")
        return Response.from_stored(row.response)

    # We won the claim: do the work exactly once, inside one transaction
    # with the ledger write (section 7), then store the response.
    response = do_charge_and_write_ledger(request_body)
    db.execute("UPDATE idempotency_keys SET status='succeeded', response=%s "
               "WHERE key=%s", response.serialize(), idempotency_key)
    return response
```

Two refinements a senior adds. First, **fingerprint the request body** (`request_hash`) so that if a client reuses a key with a *different* payload — a real client bug — you reject it loudly rather than silently returning the wrong stored result. A key promises "this is the same intent"; a different body breaks that promise and you must surface it. Second, handle the **`pending` window**: between claiming the key and finishing the work, a retry might arrive. It must not start a second charge; it gets told "in progress, retry shortly," and it will eventually find the stored `succeeded` response. This is the engine of effectively-once, which the next two sections build the rest of.

## 7. The dual-write problem and the outbox pattern

Now the single nastiest bug in payments, the one that separates people who have operated a payment system from people who have only drawn one on a whiteboard. You have two systems that must agree: **your ledger database** and **the external processor** (Stripe, a card network, a bank). A charge means writing to both — record the entry in your ledger *and* tell the processor to move the money. The problem: **you cannot write to two independent systems atomically.** There is no transaction that spans your Postgres and Stripe's API. So whatever order you choose, a crash in between leaves them inconsistent, and inconsistent here means money. Figure 6 lays out the trap and the fix.

![A before and after comparison showing a naive dual write that can crash between the ledger write and the processor call versus a transactional outbox that commits the ledger and an outbox row in one transaction](/imgs/blogs/design-a-payment-system-6.webp)

Consider the two naive orderings and watch both fail. **Order A: call the processor first, then write the ledger.** The processor charges the card (\$50 leaves the customer), and then your process crashes before the ledger write commits. The money moved, but your books say nothing happened. The customer is out \$50 and your ledger cannot account for it — the worst kind of drift, because the customer noticed before you did. **Order B: write the ledger first, then call the processor.** Your ledger says the charge happened, you crash before calling the processor, the money never actually moves, but your books say it did — you will pay out a merchant for a charge that never settled. Both orderings can lose or invent money. There is no ordering of two non-atomic writes that is safe, and "just use a distributed transaction across both" is not available because you do not control the processor's transaction boundary.

The senior solution is the **transactional outbox**. The insight: you *can* atomically write two things that live in the *same* database. So instead of writing the ledger and then calling the processor, you write the ledger entry **and an outbox row** — a record that says "a processor call needs to happen" — in **one local transaction**. That transaction is atomic; either both the ledger entry and the outbox row commit, or neither does. A separate **settlement worker** then reads unprocessed outbox rows and calls the processor, marking each done when the processor confirms. Now a crash is recoverable: if you crash after committing the ledger+outbox but before calling the processor, the outbox row is still there, and the worker will retry it. The processor call becomes **at-least-once** (the worker retries until it succeeds), and because the charge endpoint is **idempotent** (section 6) and the processor accepts our idempotency key, at-least-once plus idempotent equals **effectively-once**. This is the same outbox the [transactional outbox deep-dive](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) covers as a publishing pattern; here it is solving the money dual-write.

```sql
BEGIN;
  -- 1. Append the ledger entries (debits = credits), still 'pending' settlement.
  INSERT INTO ledger_transactions (id, idempotency_key, kind) VALUES (...);
  INSERT INTO ledger_entries (transaction_id, account_id, direction, amount_minor, currency)
    VALUES (...), (...);   -- debit + credit, same currency

  -- 2. In the SAME transaction, record the work the worker must do.
  INSERT INTO outbox (id, transaction_id, action, processor_idempotency_key, status)
    VALUES (..., 'capture', 'pk_8f2a91', 'pending');
COMMIT;
-- After COMMIT: the settlement worker polls outbox WHERE status='pending',
-- calls the processor with processor_idempotency_key (so the processor dedups too),
-- and on confirmation marks the outbox row 'done' and the ledger txn 'settled'.
```

One detail that makes or breaks this: the **processor must also honor an idempotency key.** Because the worker retries at-least-once, the same capture might be sent to the processor twice (the first call succeeded but the worker crashed before recording it). If the processor dedups on the key we send, the second call is a safe replay and the money moves once. Every real processor — Stripe, Adyen, Braintree — supports idempotency keys precisely for this reason. You generate the key, store it on the outbox row, and send it on every retry of that capture. The dual-write problem dissolves into "idempotent at-least-once delivery over a transactional outbox," which is a problem we know how to solve.

## 8. The payment flow as a state machine

Tie the pieces together into the lifecycle of a single charge, because the flow is a **state machine** and treating it as one is how you avoid an entire class of bugs where you act on a payment that is not in a state that permits the action. Figure 1 (the cover) drew it; here we make the states precise. A card charge classically has three phases: **authorize** (the processor checks the card is valid and places a *hold* on the funds — money is reserved but not yet taken), **capture** (you tell the processor to actually take the held funds), and **settle** (the money clears from the card network to your account, which can take a day or more). Many flows do authorize-and-capture in one step for simplicity; you split them when you need to reserve funds now and charge later (a hotel hold, a marketplace that captures on ship).

The states a payment moves through in your system: `created` → `authorized` → `captured` → `settled` for the happy path, with `failed`, `declined`, and `reversed` as off-ramps. The crucial design rule: **only terminal states are safe to report as final to the user, and transitions are driven by events from the processor, not by optimism.** When you call the processor and it returns "authorized," you record `authorized` — you do *not* tell the user "payment complete," because settlement can still fail. A senior never lets the UI run ahead of the ledger. The state in your ledger is the truth; the UI reflects it, never leads it.

The settlement and final confirmation usually arrive **asynchronously** via a **webhook** from the processor. You ask the processor to capture; it accepts and returns "processing"; minutes or hours later it sends a webhook — `charge.settled` or `charge.failed` — to a URL you registered. Your webhook handler is the thing that moves the payment to a terminal state. This async confirmation is why the state machine matters: between "capture requested" and "settled," the payment is in a non-terminal state and you must handle a user refreshing, a duplicate webhook, and a webhook that never arrives (you reconcile, section 10). Webhooks are themselves an at-least-once channel — processors retry webhooks until you ack with a 200 — so your **webhook handler must be idempotent**: dedup on the processor's event id before applying the state transition, exactly as in section 6.

```python
def handle_webhook(event: ProcessorEvent):
    verify_signature(event)          # reject forgeries: webhooks are public endpoints
    # Idempotent: the processor retries webhooks; dedup on its event id.
    if db.seen_event(event.id):
        return Response(200)         # already applied; ack so it stops retrying
    db.record_event(event.id)

    txn = db.get_transaction(event.processor_idempotency_key)
    # Only legal transitions. Ignore events for already-terminal payments.
    if event.type == 'charge.settled' and txn.status == 'captured':
        append_settlement_entry(txn)     # ledger move, if any, on settle
        db.set_status(txn.id, 'settled')
    elif event.type == 'charge.failed' and txn.status in ('authorized','captured'):
        append_reversal_entries(txn)     # unwind the hold/capture on the ledger
        db.set_status(txn.id, 'failed')
    return Response(200)
```

The state machine plus idempotent webhooks plus the outbox is the complete happy-and-unhappy path for a single payment. What remains is verifying that this internal machine actually agrees with the outside world — which no amount of internal correctness can guarantee on its own.

## 9. The unknown-state problem: when the processor times out

This is the failure that defines payments, the one that separates the design from a toy. You call the processor to capture \$50. The connection times out. **You do not know whether the charge went through.** Maybe the request never reached the processor (money did not move). Maybe it reached the processor, the charge succeeded, and the *response* was lost on the way back (money moved, you do not know it). These two cases are indistinguishable from your side — the timeout looks identical — and they demand opposite actions: in the first you should retry, in the second you must not, because retrying would double-charge. The two-generals problem, in its purest, most expensive form.

You cannot prevent the ambiguity; you resolve it. The wrong move is to blindly retry (might double-charge) or blindly give up (might leave money in limbo). The senior move is **query-then-retry, made safe by idempotency.** First, because you sent an idempotency key, a blind retry is *already safe* — if the original succeeded, the processor recognizes the key and replays the original result instead of charging again; if the original never landed, the retry charges once. So the idempotency key alone downgrades the unknown-state problem from "catastrophic" to "safe to retry." That is the first reason the key is non-negotiable: it is what makes a timeout recoverable.

But you still want to *know* the truth, not just be safe, so you also **query**: ask the processor "what is the status of the charge with idempotency key `pk_8f2a91`?" The processor tells you authorized, settled, or not-found, and now you can set your ledger state correctly. The pattern in code is a bounded retry loop with idempotent calls and a status query as the tiebreaker. And critically: until you have a definitive answer, the payment stays in a **non-terminal** state (`pending_unknown`), the UI shows "processing," and the reconciliation job (next section) is the backstop that catches anything still ambiguous hours later. You never resolve an unknown to "succeeded" by guessing.

```python
def capture_with_unknown_state_handling(txn, key):
    for attempt in range(MAX_RETRIES):
        try:
            # Idempotent: a duplicate capture replays, never double-charges.
            return processor.capture(amount=txn.amount, idempotency_key=key)
        except Timeout:
            # We do not know if it went through. Query the truth.
            status = processor.get_status(idempotency_key=key)
            if status == 'succeeded':
                return status            # it landed; the timeout was on the response
            if status == 'not_found':
                continue                 # never landed; safe to retry the capture
            # still 'pending' on the processor side: back off and re-query
            sleep(backoff(attempt))
    # Exhausted retries without a definitive answer.
    db.set_status(txn.id, 'pending_unknown')   # NOT 'failed', NOT 'succeeded'
    alert("capture unresolved after retries; reconciliation will settle it")
```

The deeper lesson generalizes beyond payments: **for any external mutation, a timeout is not a failure — it is an unknown, and unknowns must be resolved by querying, made safe by idempotency, and never guessed.** A surprising number of production money bugs trace to code that treated a timeout as a failure, retried, and double-charged, or treated it as a success and never collected. Treat it as what it is: a question you have not answered yet.

## 10. Reconciliation: trust nothing, verify everything

Internal correctness is necessary but not sufficient, because your ledger is your *belief* about what happened and the processor's records are *their* belief, and beliefs drift. A webhook got lost. A timeout left a payment unresolved. A processor-side bug or a manual adjustment moved money you do not know about. The discipline that catches all of this is **reconciliation**: periodically (typically daily, on the processor's settlement report) you compare your ledger against the processor's records, find every discrepancy, classify it, and fix it. Figure 7 shows the shape — join your ledger against the processor report, and route every mismatch to a remediation path.

![A reconciliation graph that joins the internal ledger against the processor settlement report, matches by reference, and routes drift to an automated correction or a manual queue](/imgs/blogs/design-a-payment-system-7.webp)

The processor sends a **settlement report** — a file or API feed listing every transaction they processed for you in a period, with amounts, fees, and references. The reconciliation job loads it and joins it against your ledger on a stable reference (the idempotency key or the processor's transaction id). Every row falls into one of three buckets. **Matched**: present on both sides, same amount — no action, the overwhelming majority. **In your ledger but not the report** (or marked differently): you think you charged but the processor has no record, or it failed on their side — investigate and likely reverse the ledger entry. **In the report but not your ledger**: the processor moved money you have no record of — the dangerous case, often a lost webhook or a successful charge whose response you never saw — you append the missing ledger entry to match reality, because *the money actually moved* and your books must reflect it.

The output of reconciliation is a list of **breaks** — discrepancies — each routed to a fix. Small, well-understood breaks (a missing entry for a charge the report confirms) are auto-corrected by appending the reversing or reconciling entry, always as a new ledger transaction with a `kind='reconciliation'` so the audit trail shows the correction. Ambiguous or large breaks go to a **manual review queue** where a human (operations, finance) decides — because some drift indicates a bug you must fix at the source, not just patch in the ledger. The reconciliation job's success metric is **break rate**: what fraction of transactions reconcile cleanly, and how fast breaks are resolved. A healthy system reconciles >99.9% automatically and resolves the rest within a day; a rising break rate is the earliest signal that something upstream is broken.

```python
def reconcile(date):
    report = processor.fetch_settlement_report(date)   # their truth
    breaks = []
    for record in report:
        ours = db.find_transaction(record.reference)
        if ours is None:
            # Money moved on their side, no record on ours: append to match reality.
            breaks.append(Break('missing_internal', record, fix='append_entry'))
        elif ours.amount != record.amount:
            breaks.append(Break('amount_mismatch', record, ours, fix='manual'))
        # else: matched, no action
    # Also check the other direction: our 'settled' txns absent from the report.
    for txn in db.settled_transactions(date):
        if txn.reference not in report.references:
            breaks.append(Break('missing_external', txn, fix='manual'))

    for b in breaks:
        if b.fix == 'append_entry':
            append_reconciliation_entry(b)    # auto-correct, new ledger txn
        else:
            manual_review_queue.enqueue(b)    # human decides
    metrics.gauge('recon.break_rate', len(breaks) / max(len(report), 1))
    return breaks
```

Reconciliation is the **safety net under every other mechanism**. Idempotency, the outbox, the state machine, the unknown-state handling — all of them reduce drift, but none of them eliminate it, because the outside world is beyond your control. Reconciliation is the process that assumes drift *will* happen and guarantees you find it before a customer or an auditor does. A senior treats a payment system without reconciliation as fundamentally unfinished, no matter how clean the internal code looks.

There is a second, internal reconciliation that is just as important and easier to forget: the **ledger's own balance check**. Independently of the processor, you can verify your books are internally consistent by summing *all* debits and *all* credits across every account and confirming they are equal — because in double-entry they must be, by construction. Run this as a continuous invariant (a cheap query over the day's entries, or a running checksum) and alarm the instant it diverges, because a non-zero global imbalance means a bug appended a debit without its matching credit, and that is a five-alarm fire you want to catch within seconds, not at month-end close. The two reconciliations answer different questions: the internal one asks "do my own books balance?" (a code-correctness check), and the external one asks "do my books match the processor's reality?" (a world-correctness check). You need both, and a senior names both in the review because reviewers who have been burned will ask which one catches a given class of bug.

A practical note on **fees and timing**, because they are where naive reconciliation breaks. The processor's report shows the *net* settled amount — your \$50 charge arrives as \$48.55 after a \$1.45 processing fee — so your reconciliation must model the fee as its own ledger entry (debit a "processing fees" expense account \$1.45, so the gross \$50, the fee \$1.45, and the net \$48.55 all reconcile and the books still balance). And settlement is *delayed*: a charge captured Monday may settle Wednesday, so the report you reconcile against is for transactions that cleared in a window, not the ones you initiated that day. Reconcile on the processor's clearing date, account for the in-flight transactions explicitly, and your break rate stays near zero. Skip the fee modeling and every single transaction shows as a \$1.45 "mismatch," drowning the real breaks in noise — a classic rookie reconciliation that technically runs but is operationally useless.

#### Worked example: a lost webhook caught by reconciliation

Trace a real drift end to end. Monday 10:00, a customer pays \$50. The charge endpoint claims the idempotency key, appends the ledger entries (gross \$50, status `captured`), writes the outbox row, and commits — all good. The settlement worker calls the processor, which accepts and returns "processing." Tuesday, the processor settles the charge and fires a `charge.settled` webhook — but a deploy is rolling your webhook handler at that exact moment, the handler is briefly unavailable, the processor retries a few times and eventually gives up after its retry window. **The webhook is lost.** Your ledger still says `captured`, not `settled`; the money actually moved on the processor's side, but your books do not reflect the settlement. No customer complains yet, because they were charged correctly — but your `captured`-but-not-`settled` count is quietly wrong, and your available-to-payout figure is understated by \$50.

Tuesday night, **reconciliation runs**. It loads the processor's settlement report, which lists the \$50 charge (net \$48.55 after the \$1.45 fee) as settled. It joins on the idempotency key, finds your ledger transaction in `captured` state, and flags a `missing_external_settlement` break: the processor settled it, you did not record the settlement. Because this break class is well-understood and the report is authoritative (*the money really moved*), the job auto-corrects: it appends a reconciliation transaction — credit your settlement-receivable, debit the fee account \$1.45 — and moves the ledger transaction to `settled`. Wednesday morning, your books match reality to the cent, the payout figure is correct, and a human never had to touch it. The lesson: the lost webhook was inevitable (deploys happen, retry windows expire), and the *only* reason it did not become a real-money discrepancy is that reconciliation assumed it would happen and was watching for it. This is why "we have idempotent webhooks" is not a substitute for reconciliation — idempotent webhooks handle the *duplicate* webhook; reconciliation handles the *missing* one, and you need both.

## 11. Consistency: why balances must be linearizable

State the consistency requirement plainly: **balance-affecting operations must be linearizable** — strongly consistent — and there is no acceptable relaxation. The reason is the **overdraft / double-spend** check. To spend, you must verify the account has sufficient funds, then append the debit. If two concurrent spends both read a \$50 balance, both see "enough for \$40," and both append a \$40 debit, you have spent \$80 from a \$50 account — you created \$30 out of nothing. The check-then-act must be **serialized**: the second spend must see the first's debit before it decides. That is the definition of linearizability, and the full taxonomy of why eventual consistency cannot provide it is in the [consistency-models guide](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects). Figure 8 is the decision tree a senior uses to assign a consistency level per operation.

![A decision tree that routes money-moving operations to linearizable consistency and read-only reporting operations to lagging read replicas or snapshots](/imgs/blogs/design-a-payment-system-8.webp)

The decision tree is the key, because **not every operation needs linearizability — only the money-moving ones do**, and pretending otherwise leaves performance on the table. Walk the branches. *Does the operation move money (debit/credit)?* If yes, it needs linearizable consistency: route it to the single leader for the account's shard, do the balance check and the append in one serializable transaction, and accept the latency cost — correctness is the SLO. If the operation is *read-only* — showing transaction history, generating a monthly statement, a finance dashboard — it can run on a **read replica** with some lag, because a statement that is a few seconds stale is fine and a report does not gate a money movement. A **point-in-time snapshot read** (the balance as of yesterday) is likewise relaxable. The discipline: **strong consistency where money moves, relaxed reads everywhere else**, decided per operation, never blanket-applied.

How do you get linearizability in practice? The pragmatic answer for most payment systems is **single-leader per account (or per shard), with a serializable transaction for the spend.** Within one database leader, a `SERIALIZABLE` or `REPEATABLE READ` transaction that does `SELECT balance ... FOR UPDATE` then `INSERT entry` is linearizable for that account — the row lock (or the serializable conflict check) serializes concurrent spends. You do *not* need distributed consensus across the whole system; you need it *per account*, and an account lives on one shard with one leader. This is why we shard by account (section 12): each account's money movements are serialized on its home leader, and accounts are independent. When you genuinely need cross-shard atomicity (a transfer between two accounts on different shards), you reach for a [saga](/blog/software-development/database/saga-pattern-distributed-transactions) or two-phase commit — covered in the [consensus and coordination](/blog/software-development/system-design/consensus-and-coordination-in-distributed-systems) deep-dive — but the common case of "spend from one account" needs only single-leader linearizability, which is cheap.

```sql
-- Linearizable spend: balance check and debit serialized on the account's leader.
BEGIN ISOLATION LEVEL SERIALIZABLE;
  -- Derive the balance (snapshot + entries) and lock the account row.
  SELECT * FROM accounts WHERE id = :account_id FOR UPDATE;
  -- (compute balance = latest_snapshot + sum(entries since); app-side or a view)
  -- Refuse to overdraw: this check + the insert below are now serialized.
  -- If balance < :amount THEN ROLLBACK; raise insufficient_funds.
  INSERT INTO ledger_transactions (...) VALUES (...);
  INSERT INTO ledger_entries (...) VALUES (debit ...), (credit ...);
COMMIT;
-- Two concurrent spends on the same account cannot both pass the check:
-- the FOR UPDATE / serializable conflict serializes them. No double-spend.
```

The honest trade-off, stated for the review: linearizability costs latency and caps per-account write throughput (one account's spends serialize, so a single hot account is a bottleneck). For payments that is the *right* trade — a single user's account does not need 10,000 writes a second, and the few accounts that do (a large merchant's incoming credits) you handle with the sharded-counter / multiple-sub-account technique. You buy correctness with latency, and in payments that is always the right purchase.

## 12. Estimation, sharding, and the optimization angle

Now the back-of-the-envelope, because a senior sizes the system before optimizing it, and the numbers shape the design. Use the [estimation discipline](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) from earlier in the series. Suppose a mid-size payment platform: **1,000 transactions per second at peak** (a generous number — most are well under this), averaging maybe 200 TPS over a day. Each transaction writes at least 2 ledger entries (debit + credit), often more with fees, so call it ~3 entries per transaction. That is **3,000 entry-writes per second at peak**, ~600 sustained.

Ledger growth: at 200 TPS sustained × 3 entries × 86,400 seconds/day ≈ **52 million entries per day**. Each entry is maybe 100 bytes of columns plus index overhead, call it ~250 bytes stored, so ~13 GB/day, **~4.7 TB/year** of ledger before compression — and you *never delete* it, because it is the audit record. This is why the ledger is the storage driver, and why you plan for partitioning by time (old entries to cheaper storage) and never count on shrinking it. The idempotency store is smaller and *can* expire: at 1,000 TPS peak with a 24-hour TTL, you hold roughly 1,000 × 86,400 ≈ **86 million keys** at most, a few GB in something like Redis or a dedicated table — bounded because keys expire after the retry window closes.

Now the optimization story, and the senior framing is the opposite of a feed's: **the bottleneck is not read throughput, it is the linearizable write path, and the optimization is to make money movement scale horizontally without sacrificing per-account correctness.** Three levers, in order of impact. **First, shard by account.** Each account's writes serialize on its home shard's leader, but *different* accounts are independent, so N shards give you N parallel linearizable write streams. This is the [partitioning](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) lever, and it is what lets a correctness-first system still scale: you are not relaxing consistency, you are *partitioning* it so each partition stays strongly consistent and they run in parallel.

**Second, keep the ledger append-only for write throughput.** Appends do not contend the way updates do — there is no hot balance row that every write must lock, because there *is* no balance row. Writes to different accounts on the same shard are inserts that do not block each other. The append-only design is not just an audit feature; it is a *throughput* feature, because it removes the single-row update bottleneck that a mutable-balance design would create. **Third, async-settle.** The expensive, slow part of a charge is the external processor round-trip (hundreds of milliseconds to seconds). Do not hold a database transaction open across it. Commit the ledger+outbox locally (fast, milliseconds), return to the user quickly, and let the settlement worker do the slow processor call asynchronously. This decouples your write latency from the processor's latency — your p99 for "charge accepted" is your local commit time, not the processor's settle time.

How do you measure the win? **Per-account write latency p99** (the linearizable spend, target tens of milliseconds), **end-to-end charge-accepted p99** (local commit, target under a couple hundred ms excluding processor), **settlement lag** (time from accept to settled, dominated by the processor, monitored not optimized), and **reconciliation break rate** (the correctness SLO, target <0.1% auto-resolved). Notice none of these is "QPS served from cache" — in payments the metrics that matter are about correctness and the latency of the write path, not read scale. Figure 9 stacks the layers that produce effectively-once movement, which is the property all this optimization must preserve.

![A stack diagram showing client intent over at-least-once delivery over a dedup gate over an append-only ledger combining into effectively-once money movement](/imgs/blogs/design-a-payment-system-9.webp)

#### Worked example: sizing the spend path under a 10× spike

Take a 10× spike — a flash sale pushes peak from 1,000 to **10,000 TPS** for ten minutes. What breaks, and what holds? **Entry writes** go to ~30,000/second. If you have 50 account shards, each shard sees ~600 entry-writes/second on average — comfortably within a single Postgres leader's append capacity (tens of thousands of small inserts/second), *provided no single account is hot*. **The idempotency store** must claim 10,000 keys/second — a unique-index insert, trivial for Redis or a partitioned table. **The settlement workers** are the pressure point: each charge needs a processor call, and the processor itself rate-limits you (say 2,000 calls/second). At 10,000 TPS in, 2,000 settled/second out, the outbox *backs up* — and that is *fine, by design*. The ledger commits accepted (fast, local), users see "processing," and the outbox drains over the following minutes via [backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure). Settlement lag rises from seconds to minutes; **no money is lost, nothing double-charges, the books stay balanced.** The async-settle decoupling is exactly what turns a 10× spike from an outage into a temporary lag. The one real risk is a **single hot account** — a popular merchant receiving 5,000 of those 10,000 credits/second — whose writes serialize on one shard. You handle that by splitting the merchant into N sub-accounts (a sharded counter for credits) and summing them for the merchant's balance, restoring parallelism without losing the append-only audit trail.

### The cross-shard transfer: the one place single-leader is not enough

There is one operation that single-leader-per-account does not solve cleanly, and a sharp reviewer will reach for it: a **transfer between two accounts that live on different shards.** Within one account's shard, a spend is one local serializable transaction — easy. But "move \$30 from user A on shard 7 to merchant B on shard 12" touches two leaders, and there is no single local transaction that spans them. You are back to a distributed-transaction problem, the same shape as the DB-plus-processor dual-write, but now both sides are *yours*. Two honest options. **Option one: route both legs of a transfer through a single coordinating account/shard** — colocate related accounts so the common transfers stay local, which is a partitioning choice you make up front (put a user and the merchants they pay most onto the same shard where you can). **Option two: a two-phase saga** — debit A in one transaction (recording the transfer as `pending`), credit B in a second, and on failure of the second, append a compensating credit back to A. This is the [saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions), and it trades atomicity for availability: there is a window where A is debited and B is not yet credited, which is acceptable *only* because the ledger makes that window visible and recoverable (the `pending` transfer is a real ledger state, and the compensation is a real reversing entry). What you must *not* do is hide the window by mutating two balances and hoping; the whole point of the append-only ledger is that even the intermediate, mid-transfer state is recorded and reconcilable. For the rare case that genuinely needs atomic cross-shard commit, the coordination primitives — and their costs — live in the [consensus and coordination](/blog/software-development/system-design/consensus-and-coordination-in-distributed-systems) deep-dive; the senior default is to design the sharding so that transfers are local and reserve the saga for the unavoidable minority.

## 13. Trade-offs and rejected alternatives

A senior never recommends an approach without naming what it costs and what it rules out. Figure 5 is the decision matrix for the central choice — how to represent money movement — scoring each approach on correctness, auditability, read latency, and complexity.

![A decision matrix comparing mutable balance, append-only ledger, and event sourcing across correctness, auditability, read latency, and complexity](/imgs/blogs/design-a-payment-system-5.webp)

| Approach | You gain | You pay | When it wins |
| --- | --- | --- | --- |
| **Mutable balance (UPDATE)** | Trivial reads (balance is a column), lowest complexity | No audit trail, lost-update races, no point-in-time history | Never for real money; fine for non-monetary counters where loss is tolerable |
| **Append-only double-entry ledger** | Full audit trail, no destructive races, derivable history, balanced books | Reads need snapshots, more storage (never delete), more thinking | The default for any system that moves real money — this is what we built |
| **Event sourcing (full event log)** | Maximum flexibility, replay any view, time-travel | Highest complexity, eventual-consistency traps in projections | When you need many derived views and can invest in the machinery; overkill for a pure ledger |
| **Synchronous dual write (no outbox)** | Simplest to code (call DB then processor) | Crash between writes loses or invents money — fundamentally broken | Never; it only looks fine until the first crash mid-charge |
| **Transactional outbox + idempotent processor** | Atomic local commit, safe at-least-once external call, effectively-once | A worker, a poller, eventual settlement, more moving parts | The default for the DB-plus-external-processor dual-write — this is what we built |
| **Distributed transaction (2PC) across DB + processor** | True atomicity if it existed | You do not control the processor's transaction; not available; blocking coordinator | Never available for an external processor; reserve 2PC for systems you fully own |

The two rejected alternatives worth defending explicitly in a review. **"Why not just store the balance and update it in a transaction?"** Because you lose the audit trail (you cannot answer "what was this balance last Tuesday and why"), you invite lost-update races on the hot balance row, and the moment you need to correct a mistake you have no clean way to do it without rewriting history. The append-only ledger costs a snapshot mechanism and more storage; that cost is trivial next to the value of being able to *prove* every cent, which in payments is a legal requirement, not a nicety. **"Why not a distributed transaction across our DB and the processor?"** Because there is no such transaction — you do not control the processor's commit, the processor does not enlist in your XA coordinator, and even if it did, 2PC's blocking coordinator is a liability. The outbox gives you the achievable thing (atomic local commit + idempotent at-least-once external call) instead of the impossible thing (atomic write across two independent systems).

One more trade-off threaded through the whole design: **strong consistency versus availability during a partition.** We chose consistency — during a network partition that isolates an account's leader, we **refuse the write** rather than risk a double-spend on a stale replica. A feed system would choose availability. This is the [CAP / PACELC](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) decision, made deliberately and in the correctness-favoring direction, because in payments an unavailable charge is a retry and a wrong charge is a lawsuit.

## 14. Case studies: how real systems get money right

Three real-world patterns, each teaching a concrete lesson, with claims kept at the level of what these companies have publicly described.

**Stripe's idempotency keys.** Stripe was an early, influential popularizer of client-supplied idempotency keys on its API, and the design is essentially the one in section 6: the client sends an `Idempotency-Key` header, Stripe stores the first response keyed by it for a bounded window (publicly documented as on the order of 24 hours), and any retry with the same key replays the original result instead of creating a second charge. The lesson is that **idempotency is an API-contract feature, not an internal implementation detail** — Stripe exposes it so *your* retries are safe, which is what makes building a payment system on top of a processor tractable. They publicly write about request fingerprinting and the race-handling we covered; the takeaway for your own design is to make idempotency a first-class part of every mutating endpoint's contract, header and all.

**The double-entry ledger as the industry default.** Serious money-movement systems — exchanges, neobanks, large marketplaces — converge on an immutable, append-only, double-entry ledger, and several have written publicly about it; the pattern is old enough that purpose-built ledger databases (TigerBeetle is a public example designed specifically for double-entry financial transactions at high throughput) exist precisely because the append-only-debits-equal-credits model is so universal that it is worth a dedicated engine. The lesson: **you are not being clever by inventing a ledger; you are joining a five-hundred-year-old consensus**, and the systems that deviate (storing a mutable balance to "keep it simple") are the ones that end up with unexplained drift and an auditor's report. When you propose an append-only ledger in a review, you are proposing the boring, correct, industry-standard thing, and that is exactly the right move for money.

**The double-charge incident pattern.** The canonical payments post-mortem reads the same across companies even when the specifics differ: a processor call timed out, the code treated the timeout as a failure and retried *without* an idempotency key (or with idempotency missing on one path), the original charge had actually succeeded, and the customer was billed twice. The fix is always the same two-part remedy — **idempotency keys on every external mutation so retries are safe, and reconciliation against the processor's report to catch the doubles that slipped through before customers do.** The lesson a senior extracts is that the bug is rarely in the happy path; it is in the *timeout* path, the *retry* path, the *partial-failure* path — the paths that are hard to test and only fire under load. Which is precisely why this whole design is organized around those paths, not the happy one. (The forward-looking sibling post, [design a video streaming platform](/blog/software-development/system-design/design-a-video-streaming-platform), inverts these priorities — there, scale and availability dominate and a dropped frame is forgivable, which is a useful contrast for seeing how problem framing reshapes a design.)

## 15. Stress-testing the design: three incidents

Pose the three incidents that actually take payment systems down and walk the design through each, because a design you have not stress-tested is a hypothesis, not a plan.

**Incident 1 — the processor timeout (unknown state).** You capture \$50; the call times out; you do not know if it landed. Does the design hold? Yes: the idempotency key (section 6, 7) makes a retry safe — if it landed, the processor replays; if not, the retry charges once. The query-then-retry logic (section 9) resolves the truth where it can; where it cannot, the payment stays `pending_unknown` (never guessed to succeeded or failed), and reconciliation (section 10) settles it against the processor's report within the day. **No double charge, no lost charge, the books eventually balance to the cent.** The design holds because it was built for this exact case.

**Incident 2 — the customer double-submit.** A user clicks "Pay \$30" twice (or the mobile app resends). Two requests arrive, possibly in the same millisecond. Does the design hold? Yes: both carry the *same* idempotency key (generated once per intent on the client), so the atomic claim (the unique index, section 6) lets exactly one win and do the work; the other gets a conflict and replays the first's result. **One charge, one ledger transaction, one \$30 debit.** If the client erroneously generated two *different* keys for one intent — a client bug — the request fingerprint does not save you (different keys look like different intents), which is why the client-side discipline of "one key per intent" matters and why you also have reconciliation as the backstop. The design holds for the common case and degrades to "caught by recon" for the client-bug case.

**Incident 3 — a partition mid-transaction.** A network partition isolates the leader of an account's shard right as a spend is committing. Does the design hold? Here the CAP choice (section 13) pays off: the spend's balance-check-and-debit is a single serializable transaction on one leader, so it either commits fully (and survives, durably replicated to in-partition followers) or it does not commit at all (the client gets an error and retries, safely, with the same idempotency key). There is **no partial money movement** — the ledger entry and its idempotency claim commit atomically or not at all, because they are in one local transaction. The cost we accept: during the partition, spends on that account are **unavailable** (we refuse rather than risk a stale-replica double-spend), and that is the correct trade. When the partition heals, the account resumes; reconciliation confirms no drift occurred. The design holds by *refusing* to be available in the one situation where availability would mean incorrectness.

The thread through all three: **every incident resolves to "no money lost, no money invented, books balance, audit trail intact,"** sometimes at the cost of latency or temporary unavailability, never at the cost of correctness. That is what it means to design a payment system like a senior — you trade away the things you can afford to lose (a few hundred milliseconds, brief unavailability of one account) to protect the one thing you cannot (the integrity of the money).

## 16. When to reach for this design (and when not to)

Decisively: **reach for the full append-only ledger plus idempotency plus outbox plus reconciliation whenever real money moves and someone could be harmed by getting it wrong** — charges, wallets, payouts, exchange balances, anything an auditor or regulator will inspect, anything where a double-charge is a refund and a reconciliation nightmare. In that world, every mechanism in this post earns its complexity, and skipping any one of them is a future incident. The append-only ledger is non-negotiable; idempotency on mutating calls is non-negotiable; the outbox is the answer to the dual-write whenever an external processor is involved; reconciliation is the safety net you build *before* you need it, not after the first drift.

**Do not reach for the full machinery when there is no real money and no audit requirement.** A points/karma counter, a non-monetary "likes" count, an internal usage metric where an occasional lost increment is tolerable — these do not need a double-entry ledger, and forcing one on them is over-engineering that will slow you down for no correctness gain. The tell is the question "if this number is off by one, does anyone get hurt or does any law get broken?" If no, a simple counter (even a [sharded counter](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) for hot ones) is fine. If yes — if the answer involves money, refunds, regulators, or trust — you are in payment-system territory and the full design applies. And the in-between: a feature that touches money but is *backed by* a real payment provider (you are not the ledger of record, the provider is) can lean on the provider's ledger and idempotency and just keep your own reconciliation — you do not always have to be the system of record, but you always have to reconcile against whoever is.

One scope warning a senior states explicitly: **building your own ledger of record is a serious, regulated undertaking.** If your product can sit on top of a provider that *is* the system of record (Stripe holds the balance, you hold a reference), do that, and your job shrinks to idempotency + reconciliation against the provider. Build the full ledger yourself only when you genuinely must be the system of record — a wallet, an exchange, a neobank — and when you do, the design in this post is the floor, not the ceiling.

## 17. Key takeaways

- **Correctness dominates scale in payments.** The throughput is modest (thousands of TPS); the hard part is never losing or duplicating a cent. Rank correctness, durability, and auditability above latency, and state that ranking first in any review.
- **Never store a balance; derive it from an append-only, double-entry ledger.** Entries are immutable, debits always equal credits, and you fix mistakes by appending reversing entries — never by editing history. This buys audit, concurrency safety, and debuggability.
- **Represent money as integer minor units with an explicit currency.** Floats lose money to rounding; integers (cents) do not. This is the cheapest correctness win and the most-skipped one.
- **Put an idempotency key on every mutating call, and claim it atomically via a unique index.** The network retries and you do not get a vote; the unique constraint is the synchronization point that makes the same intent run exactly once.
- **Solve the DB-plus-processor dual-write with a transactional outbox.** You cannot write two independent systems atomically, so write the ledger and an outbox row in one local transaction and let an idempotent at-least-once worker do the external call. At-least-once + idempotent = effectively-once.
- **A timeout is an unknown, not a failure.** Resolve it with query-then-retry made safe by idempotency; never guess "succeeded" or "failed," and let a `pending_unknown` state plus reconciliation settle the rest.
- **Make balance-affecting operations linearizable; relax everything else.** Shard by account so each account's money movements serialize on one leader (correctness) while different accounts run in parallel (scale). Reporting reads can lag.
- **Reconcile against the processor on a schedule.** Internal correctness is your belief; reconciliation is how you verify it against reality and find drift before a customer does. A payment system without reconciliation is unfinished.
- **Optimize the write path, not read scale.** The bottleneck is the linearizable spend; the levers are shard-by-account, append-only (no hot row), and async-settle (decouple your latency from the processor's). Measure per-account write p99, charge-accepted p99, settlement lag, and break rate.

## 18. Further reading

- [Idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) — the full mechanism behind the idempotency-key flow this post wires into payments.
- [Consistency models, a practical guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) — why balances must be linearizable and what eventual consistency would cost here.
- [Transactional outbox pattern, reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — the dual-write solution, in depth.
- [Queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) — the delivery semantics and backpressure that let the settlement worker drain a spike safely.
- [Consensus and coordination in distributed systems](/blog/software-development/system-design/consensus-and-coordination-in-distributed-systems) — for the cross-shard transfer case where single-leader linearizability is not enough.
- [Saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions) — the alternative to 2PC when money must move across services that do not share a transaction.
- [Schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) — how to evolve a ledger schema and a payment API without breaking the immutability or the clients.
- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the sizing discipline used for the TPS and ledger-growth math.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — the consistency-over-availability choice, made deliberately.
- [Design a video streaming platform](/blog/software-development/system-design/design-a-video-streaming-platform) — the sibling case study where scale and availability invert the priorities, a useful contrast.

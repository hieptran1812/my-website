---
title: "Idempotency and Exactly-Once Effects Across Services"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Exactly-once delivery is a myth, but exactly-once effects are achievable and mandatory the moment you add retries and at-least-once messaging — here is the idempotency-key pattern, the dedup inbox, the concurrency races, and the subtle bugs, with the code to build all of it."
tags:
  [
    "microservices",
    "idempotency",
    "exactly-once",
    "deduplication",
    "distributed-systems",
    "software-architecture",
    "backend",
    "reliability",
    "saga",
    "resilience",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/idempotency-and-exactly-once-effects-across-services-1.webp"
---

The ShopFast support queue lit up on a Tuesday afternoon with a complaint that was both rare and infuriating: a customer had been charged twice for one order. Same order id, same amount, two separate `\$80` debits on their card three hundred milliseconds apart. The customer had clicked "Place order" exactly once. The order existed exactly once in the database. And yet the payment service had two charge records, the payment processor had two debits, and the customer — quite reasonably — wanted to know why a single click had cost them `\$160`. The on-call engineer pulled the logs and found something that looked, at first, completely impossible: the order service had sent the *same* charge request twice, and both had succeeded.

It was not impossible. It was inevitable. What had happened was the most ordinary thing in a distributed system. The order service called the payment service to charge the card. The payment service charged it successfully and started writing the `200 OK` back across the network. But the response packet was slow — a momentary blip on the wire, a garbage-collection pause, a network hiccup — and the order service's HTTP client gave up at its 200-millisecond timeout before the response arrived. From the order service's point of view, the call had *failed*: it never got a response, so as far as it knew, the charge had not happened. So it did what every well-built client does in the face of a failed call to a transient-looking error: it retried. It sent the charge again. And the payment service, having no memory that it had just charged this exact card for this exact order, charged it again. Two debits. One click. A furious customer.

This is the central, load-bearing problem of every distributed system that does anything important: **you can never tell the difference between "my request failed" and "my request succeeded but the response got lost."** From the caller's side, those two outcomes look *identical* — silence. And the moment you cannot tell them apart, you are forced to choose between two bad options: retry (and risk doing the thing twice) or give up (and risk not doing it at all). The entire discipline of this post exists to resolve that dilemma. By the end you will be able to explain why "exactly-once delivery" is a fairy tale that no network can deliver, why "exactly-once *effects*" is the achievable thing you actually want, and how to build it — the idempotency key, the dedup inbox, the unique-constraint race, the saga step that re-drives safely — correctly enough to run in production and survive a code review by someone who has been paged for a double-charge.

![A before and after comparison contrasting a blind retry that double-charges a customer against an idempotency key that makes the retry return the original stored result](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-1.webp)

We are deliberately at the practitioner's layer here. The *broker* mechanics — how a message system gives you at-least-once delivery, how Kafka's idempotent producer dedups on the wire, how a consumer group commits offsets — live in the message-queue deep-dives, and I will cross-link to them rather than re-derive them. My job is the service-design discipline: what you, the engineer building the payment service and the order service and the inventory consumer, actually have to decide, type, and operate so that a duplicate request, a redelivered event, or a re-run saga step does the right thing exactly once. That discipline is not optional. The instant you add a retry — and you must add retries, because the network is unreliable and a single dropped packet should not lose an order — you have signed up for duplicates, and the only question is whether you handle them deliberately or get paged for them at 3am.

## Why duplicates are not an edge case — they are the default

Let us be precise about *why* duplicates happen, because the precision is the whole insight, and because juniors tend to treat duplicates as a rare anomaly when they are in fact a structural certainty. There are four independent sources, and in any real system all four are firing constantly.

The first is **the retry after a timeout where the request actually succeeded** — ShopFast's exact bug. The caller sets a timeout (it must; an unbounded wait is how one slow dependency takes down a whole call graph). The timeout fires. The caller does not know whether the work happened. If it retries, and the work *had* happened, you get a duplicate. If it does not retry, and the work had *not* happened, you lose the operation. There is no third option that always wins, because the caller is missing the one piece of information — "did it succeed?" — that would let it choose correctly. This is not a bug in the retry logic. It is a fundamental limit of communicating over an unreliable channel.

The second is **at-least-once messaging**. Every production message broker that survives crashes — Kafka, RabbitMQ, SQS — delivers messages *at least* once, not *exactly* once. The reason is the same timeout problem one level down: after a consumer processes a message, it must acknowledge the broker so the message is not redelivered. If the consumer crashes after processing but before the ack lands, the broker — having never seen the ack — redelivers the message to another consumer, which processes it again. The broker is behaving correctly. It would rather deliver twice than risk delivering zero times, because a lost message is usually worse than a duplicate one. The deep treatment of why at-most-once, at-least-once, and exactly-once are the only three choices, and why the middle one is what you actually get, is in [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once); the consumer-side mechanics of making at-least-once safe are in [idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). My point here is the consequence: if you consume from any broker, you *will* see the same event twice, and your handler must be ready for it.

The third is **the client double-click** — or its programmatic cousin, a mobile app that fires the request, loses connectivity, and re-fires when the user taps again. This is duplicates entering from *outside* your system, before any of your retry logic runs. You do not control the client's retry behavior, so you cannot prevent the duplicate request; you can only make the duplicate harmless.

The fourth is **the saga step that re-runs**. A saga is a multi-step business transaction across services, coordinated by an orchestrator or by choreographed events (the full pattern is in [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice)). When a saga step fails or times out, the coordinator re-drives it — re-sends the "charge payment" command, re-runs the "reserve inventory" step. If that step is not idempotent, re-driving it does the work twice. A saga is essentially a machine for generating retries, which means a saga is essentially a machine for generating duplicates, which means **every saga step must be idempotent or the saga is broken**. This is not a nice-to-have. It is a precondition for the saga pattern to work at all.

Notice the common thread across all four: a duplicate is what happens when an operation is *attempted again* because the first attempt's outcome was unknown or unconfirmed. That is the entire game. And because retries, at-least-once delivery, double-clicks, and saga re-drives are all *good, necessary* behaviors — the alternative to each is losing data — you cannot make duplicates go away by removing their causes. You can only make the second, third, and fourth attempt produce the same result as the first. That property has a name, and it is the most important word in this post: **idempotency**.

## Exactly-once delivery is a myth; exactly-once effects is the goal

Here is the sentence that, once it clicks, fixes your mental model permanently: **exactly-once delivery is impossible, but exactly-once effects is achievable, and the second is the only one you ever actually wanted.**

Exactly-once *delivery* would mean: the message (or request) is delivered to the recipient exactly one time — never zero, never twice. This is impossible over an unreliable network, and the proof is a classic thought experiment called the **two generals problem**. Two generals on opposite hills must agree on a time to attack, communicating only by messengers who might be captured. General A sends "attack at dawn." Did it arrive? A does not know unless B sends an acknowledgment. Did the acknowledgment arrive? B does not know unless A acknowledges the acknowledgment. And so on forever — there is no finite number of messages after which both sides are *certain* the other received the last one. The sender can never be sure its message landed, which means it must either risk sending zero times (if it gives up) or risk sending more than once (if it retries on silence). You cannot build "exactly once" on top of a channel that loses messages, because the sender can never know whether to retry. Any system that claims "exactly-once delivery" is either lying or is quietly doing exactly-once *effects* under the hood and mislabeling it.

![A graph showing the server branching on an idempotency key lookup into a first-time path that charges and a replay path that returns the stored result](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-2.webp)

Exactly-once *effects* (sometimes called effectively-once) is a completely different and achievable goal. It says: the message or request may be *delivered* any number of times — once, twice, ten times — but the **observable effect on the world happens exactly once**. The card is charged once. The order is created once. The inventory is decremented once. The email is sent once. We give up on controlling how many times the request arrives — that is the two-generals problem and we cannot win it — and instead we make the *handler* recognize a repeat and decline to do the work twice. The figure above shows the shape of it: when a request arrives, the server looks up its idempotency key. On a miss (first time) it does the work and stores the result. On a hit (a replay) it skips the work entirely and returns the *stored* result. The customer's retry gets back the same `200 OK` and the same charge id it would have gotten the first time, never knowing — and not needing to know — that the response to its original attempt was lost.

This reframing is the single most valuable thing a junior can internalize about distributed systems. Stop trying to make the *network* deliver exactly once; you will fail, and you will waste weeks on two-phase commit schemes that do not scale. Instead, make your *operations* tolerate redelivery — make them idempotent — and let the network be as unreliable as it likes. The unreliability becomes a non-event, because the second delivery has no additional effect. Kafka's "exactly-once semantics" feature, which we will examine in the case studies, is precisely this: not magical exactly-once *delivery*, but a transactional mechanism that makes the *effect* of a consume-process-produce loop happen once even though the underlying delivery is at-least-once. Even the systems marketed as exactly-once are, under the hood, exactly-once *effects*.

## Idempotency, defined precisely — and which operations have it for free

An operation is **idempotent** if performing it many times has the same effect as performing it once. Formally: `f(f(x)) = f(x)`. Apply it once, apply it a hundred times — the resulting state is identical, and the result returned is identical. That is the property that turns "a duplicate arrived" from a disaster into a no-op.

Some operations are **naturally idempotent**, and recognizing them is the cheapest win in this entire field because they need *zero* extra machinery. The canonical examples:

- **`PUT` / "set to an absolute value"**: `PUT /orders/42/status = SHIPPED` sets the status to `SHIPPED`. Do it once, do it five times — the status is `SHIPPED`, full stop. There is no accumulation. This is why HTTP's spec *defines* `PUT` as idempotent and `POST` as not.
- **`DELETE` by id**: `DELETE /reservations/7` removes reservation 7. The first call removes it; the second finds it already gone and the post-condition ("reservation 7 does not exist") is still true. (You must return `204` or `404`-as-success on the second call rather than erroring, but the *state* is correct.)
- **Setting a flag, assigning an owner, writing a fixed value**: `user.email_verified = true`, `order.assigned_warehouse = "EU-1"`. Idempotent because they assign rather than accumulate.

And some operations are **not** naturally idempotent, and these are exactly the ones that bite you:

- **Increment / decrement**: `balance += 80`, `inventory_count -= 1`, `retry_count++`. Each call *accumulates*. Run it twice and you have added `\$160` or removed two units. This is the classic double-effect bug.
- **Append**: `INSERT INTO charges (...)`, "add a line item," "append to the audit log." Each call adds a row. Run it twice and you have two rows where there should be one.
- **Charge / capture / transfer money**: the most dangerous one, because the effect leaves your system entirely and lands on a customer's card. There is no undo button on a real-world side effect.
- **Send email / send SMS / push notification**: also irreversible and visible to a human. Sending the "your order shipped" email twice is annoying; sending the "your payment failed, please update your card" email twice when the payment actually succeeded is a support ticket.

![A tree classifying any write operation into naturally idempotent ones like PUT and DELETE versus effectful ones made safe by an idempotency key or a dedup store](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-8.webp)

The tree above is the decision you run on every write you design. First ask: is this operation naturally idempotent? If it is a full-replacement `PUT`, a `DELETE` by id, or a flag set, you are done — no extra storage, no key, nothing. If it has an external or accumulating effect (a charge, an append, an increment, an email), it is *not* free, and you must add a guard. Which guard depends on *who* is driving the operation: a client-driven write (an HTTP request from a browser, mobile app, or another service) wants an **idempotency key**; an event-driven write (a consumer pulling from a broker) wants a **dedup store** (an inbox). The rest of this post is the two guards and their sharp edges.

The good news hiding in this tree is that **a lot of non-idempotent operations can be made naturally idempotent by reshaping them**, and that is always cheaper than bolting on a key store. The trick is to convert *relative* operations into *absolute* ones. Instead of `balance += 80` (relative, not idempotent), compute the target and write `balance = 920` as a set (absolute, idempotent) — guarded by a version or condition so a stale write does not clobber a newer one. Instead of "append a charge row," use an *upsert* keyed by a deterministic id so the second insert collides with the first instead of adding a new row. We will make this concrete with code in a moment.

#### Worked example: how often does a no-key charge actually double-charge?

Engineers under-rate this bug because the timeout window feels narrow — "the response is only lost once in a while, what are the odds?" Let us put numbers on it the way the ShopFast on-call did during the post-mortem. The order-to-payment call has a client timeout of `200ms`. The payment service's p50 is `40ms` and its p99 is `220ms` — note that the p99 is *above* the timeout, which is itself a design smell, but it is common. That means roughly 1% of charge calls take longer than the `200ms` timeout. When a call exceeds the timeout, the order service has no idea whether the charge succeeded; in fact, the overwhelming majority of these slow-but-eventually-successful calls *did* charge the card — the work completed, only the response was late. So on that ~1% of calls, the order service retries, and without an idempotency key, the retry charges again.

ShopFast does `200` charges per second at peak. One percent of `200` is `2` charges per second that blow the timeout and get retried. Of those, suppose 80% actually succeeded server-side (the slow ones that did the work but were late) — that is `1.6` double-charges per second, `5,760` per hour at peak. Even if you assume only the truly-succeeded-then-retried fraction matters and round aggressively down, you are looking at **thousands of double-charges a day** on a system processing a few hundred orders a second. The "rare anomaly" is, at scale, a flood. And every single one is a chargeback, a support ticket, a refund, and a dent in trust. The fix — an idempotency key — costs you one indexed table and a few lines of code, and it takes the double-charge rate from "thousands a day" to **zero, structurally**, not "fewer."

## The idempotency-key pattern: the Stripe model

The idempotency key is the workhorse pattern for client-driven writes, and the canonical reference implementation is Stripe's. The idea is simple and the discipline is in the details. **The client generates a unique key for each logical operation and sends it with the request. The server records the key together with the result. If a request arrives with a key the server has already seen, the server does not re-run the operation — it returns the stored result.** The retry, the double-click, the saga re-drive all carry the *same* key (because they are the *same logical operation*), so they all collapse onto the one stored result.

Two non-obvious rules make or break this pattern, and skipping either is how people ship a key store that does not actually prevent double-charges:

**Rule 1: the client generates the key, and the key identifies the *operation*, not the *request*.** The order service generates one key — say a UUID — when it decides to charge for order 42, and it reuses that *same* key across every retry of that charge. If it generated a fresh key per retry, every retry would look like a brand-new operation and you would be back to double-charging. The key is "this particular charge of this particular order," stable across the network attempts that try to make it happen. A common, clean choice is to derive the key deterministically from the business operation — e.g. `charge:order-42:attempt-of-checkout-session-xyz` — so it is naturally the same on every retry without needing to persist a generated UUID across attempts.

**Rule 2: the server stores the key *and the result*, and the store of the key is in the same transaction as the side effect** — or, when the side effect is external (a PSP charge), the server records "key seen, charge in progress" *before* calling the PSP and the final result *after*, so a crash mid-charge is recoverable. We will dig into that crash case in the stress test; for now, the shape is: look up key, if present return stored response, else do the work and store `(key, response)` atomically.

Here is the server-side endpoint, with the unique constraint that makes it safe under concurrency:

```sql
-- The idempotency store. The UNIQUE constraint on idempotency_key
-- is the entire concurrency guarantee: two requests with the same
-- key cannot both insert.
CREATE TABLE idempotency_keys (
    idempotency_key  TEXT PRIMARY KEY,         -- client-supplied, unique
    request_hash     TEXT NOT NULL,            -- guard against key reuse
    status           TEXT NOT NULL,            -- 'in_progress' | 'completed'
    response_code    INT,                      -- stored result, filled on completion
    response_body    JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```python
# A charge endpoint that is safe to call any number of times with
# the same Idempotency-Key. The flow: claim the key, do the work,
# record the result. A replay short-circuits to the stored response.
def charge(request):
    key = request.headers["Idempotency-Key"]
    req_hash = sha256(canonical_json(request.body))

    # 1. Try to CLAIM the key. The unique PK means exactly one
    #    request can win this insert; concurrent dupes get an
    #    IntegrityError and fall to the replay branch.
    try:
        db.execute(
            "INSERT INTO idempotency_keys (idempotency_key, request_hash, status) "
            "VALUES (%s, %s, 'in_progress')",
            (key, req_hash),
        )
        db.commit()
    except IntegrityError:
        return _replay(key, req_hash)   # someone already has this key

    # 2. We own the key. Do the real, non-idempotent work exactly once.
    charge_id = psp.charge(amount=request.body["amount"],
                           card=request.body["card"],
                           # pass the SAME key to the PSP so even the
                           # PSP call is idempotent end-to-end:
                           idempotency_key=key)
    response = {"charge_id": charge_id, "status": "succeeded"}

    # 3. Record the result so future replays return it.
    db.execute(
        "UPDATE idempotency_keys SET status='completed', "
        "response_code=200, response_body=%s WHERE idempotency_key=%s",
        (Json(response), key),
    )
    db.commit()
    return 200, response


def _replay(key, req_hash):
    row = db.query_one(
        "SELECT request_hash, status, response_code, response_body "
        "FROM idempotency_keys WHERE idempotency_key=%s", (key,))

    # Key reuse guard: same key, DIFFERENT request body is a client
    # bug (they reused a key for a different operation). Reject it
    # loudly rather than returning the wrong stored result.
    if row.request_hash != req_hash:
        return 422, {"error": "idempotency_key_reused_with_different_body"}

    if row.status == "in_progress":
        # The original is still running. Tell the client to back off
        # rather than racing a second charge. 409 is the Stripe choice.
        return 409, {"error": "request_in_progress, retry shortly"}

    return row.response_code, row.response_body   # the stored result
```

This is the whole pattern in thirty lines, and every line is load-bearing. The `INSERT ... ` with a primary-key (or unique) constraint is the concurrency guard. The `request_hash` is the key-reuse guard. The `in_progress` / `completed` states handle the "original is still running" race. The PSP call passes the *same* key downstream so the idempotency is end-to-end, not just at our boundary — a critical detail people miss, because if your service is idempotent but the PSP you call is not, a retry of *your own* internal logic can still double-charge at the PSP.

The client side is trivially simple, which is the point — the burden lives in the server:

```python
# The client generates ONE key per logical operation and reuses it
# across every retry. Same key == same operation == one effect.
def place_charge(order_id, amount, card):
    key = f"charge:order-{order_id}"     # deterministic, stable across retries
    for attempt in range(3):
        try:
            resp = http.post("/v1/charges",
                             json={"amount": amount, "card": card},
                             headers={"Idempotency-Key": key},
                             timeout=0.2)
            return resp.json()
        except Timeout:
            # We DON'T know if it succeeded. Retry with the SAME key.
            # If the first attempt charged, the retry returns the
            # stored result; the card is debited exactly once.
            continue
    raise ChargeFailed(order_id)
```

The customer can double-click, the network can eat the response, the saga can re-drive — all three send `Idempotency-Key: charge:order-42`, and the card is charged exactly once. That is exactly-once *effects* delivered on top of an at-least-once world.

## Making a "charge" idempotent end to end

The `idempotency_key=key` parameter passed to `psp.charge` above deserves its own treatment, because it is the difference between an idempotency key that *looks* safe and one that *is* safe. Your idempotency table protects against duplicate requests arriving at *your* service. But your service then performs a side effect — the PSP charge — and that side effect is itself a network call that can time out and be retried *inside* your own handler. If the PSP is not idempotent, your retry of the PSP call double-charges even though your outer request was deduplicated. The chain is only as idempotent as its weakest link.

Real payment processors solve this the same way: they accept an `Idempotency-Key` header too. Stripe's charge API, Adyen's, Braintree's — all of them. So the correct design is to **propagate the same key down the entire call chain**: the client's key becomes your service's dedup key becomes the PSP's idempotency key. One key, one effect, end to end. Here is the PSP-style charge call made idempotent:

```go
// Charging the PSP. The key flows through from the original client
// request, so even if THIS call times out and we retry it, the PSP
// recognizes the key and returns the original charge instead of
// debiting the card again.
func (p *PSPClient) Charge(ctx context.Context, key string, amt Money, card Card) (ChargeID, error) {
    var lastErr error
    for attempt := 0; attempt < 3; attempt++ {
        req := ChargeRequest{Amount: amt, Card: card}
        // The SAME key on every attempt. The PSP dedups on it.
        resp, err := p.post(ctx, "/charges", req, WithHeader("Idempotency-Key", key))
        if err == nil {
            return resp.ChargeID, nil // first success OR replayed original
        }
        if !isRetryable(err) {
            return "", err // a hard decline is final; do not retry
        }
        lastErr = err
        time.Sleep(backoffWithJitter(attempt)) // 50ms, 100ms, 200ms + jitter
    }
    return "", fmt.Errorf("charge failed after retries: %w", lastErr)
}
```

Note the discipline that pairs with idempotency: **only retry retryable errors.** A `500`, a timeout, a connection reset — those are ambiguous (the work might have happened) and idempotency makes them safe to retry. A `402 card_declined` is *final* — the work definitively did not happen and never will with this card, so retrying just burns time and may trip fraud rules. Idempotency is what makes the retry of an *ambiguous* failure safe; it is not a license to retry *every* failure. The relationship between timeouts, retries, and the circuit breaker that stops you from retrying a dead dependency into the ground is the subject of [resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — idempotency is the property that makes those retries *safe to take* in the first place. Retries without idempotency are a double-charge generator; idempotency without bounded retries is a half-finished resilience story. You need both.

## Designing the key: scope, retention, and the reuse fingerprint

The idempotency key looks like a trivial string, but three design decisions around it separate a key store that quietly fails from one that holds up. A junior writes `key = uuid4()` and ships it; a senior thinks about scope, retention, and the reuse fingerprint, because each of those is a bug waiting to happen.

**Scope: what is "the same operation"?** A key must be unique to a logical operation and *stable across that operation's retries* — but it must *also* be distinct from a different operation that happens to look similar. If ShopFast's order service uses `key = "charge:order-42"`, that is correct for the *one* charge of order 42. But what if order 42 is legitimately charged twice — an initial authorization and a later capture, or a customer who adds an item and is charged again? Then `charge:order-42` is too coarse; the second legitimate charge collides with the first and gets the *stored* result, and the customer is under-charged. The key must scope to the *attempt*, not just the order: `charge:order-42:auth` and `charge:order-42:capture`, or include a monotonic charge-sequence number. The rule is **the key is unique to exactly the set of network attempts that should produce one effect, and no broader.** Too narrow (a fresh UUID per network retry) and you double-charge; too broad (one key for genuinely-distinct operations) and you under-charge. Both failure modes are silent. Get the scope right and write a comment explaining it, because the next engineer will not see why `:auth` and `:capture` are separate keys.

**Retention: how long do you remember a key?** You cannot keep keys forever; the table would grow without bound and the worked example showed how fast. So you set a retention window and evict. The window must be at least as long as the longest legitimate retry horizon — how late can a retry of the *same* operation plausibly arrive? For an interactive client, retries arrive within seconds; for a saga that may pause and re-drive, within minutes to hours; for a payment that a batch job reconciles, perhaps a day. Stripe picked 24 hours. The trap is setting the window *shorter* than your retry horizon: if the key is evicted before the last retry arrives, that retry is treated as new and re-runs the effect. So the window is a lower bound driven by your retry policy, not a free knob. And note the asymmetry with the dedup-store window from the consumer side: there the window is about how late a *broker* redelivers; here it is about how late your own *client or saga* retries. They are governed by different horizons and you size them independently.

**The reuse fingerprint: same key, different body.** A client bug — or a malicious client — can send the same key with a *different* request body. Maybe the developer hard-coded a key and forgot to vary it; maybe a key got reused across two genuinely different charges. If you blindly return the stored result, you charge for operation A but tell the client it was operation B — a silent correctness disaster. The guard is the `request_hash` in our endpoint: hash the canonical request body, store it with the key, and on a replay *compare* the incoming body's hash to the stored one. Match → return the stored result (a legitimate retry). Mismatch → reject with `422`, loudly, because the client has reused a key for a different operation and you must refuse rather than guess. This is why the endpoint above stored `request_hash` from the first request: it is the integrity check that makes the key store trustworthy under client bugs.

```python
# Canonicalize the request body before hashing so that semantically
# identical requests with different key ordering or whitespace produce
# the SAME fingerprint, and a genuinely different body produces a
# DIFFERENT one. Without canonicalization, a re-serialized retry can
# false-mismatch and get wrongly rejected.
import json, hashlib

def canonical_json(body: dict) -> str:
    # sort_keys + no extra whitespace == stable byte sequence
    return json.dumps(body, sort_keys=True, separators=(",", ":"))

def request_fingerprint(body: dict) -> str:
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()
```

The canonicalization matters more than it looks: if you hash the raw bytes, two retries that re-serialize the same logical body with different key ordering or whitespace produce different hashes, the reuse guard *false-positives*, and a legitimate retry gets rejected with `422`. So you canonicalize — sort keys, strip insignificant whitespace — before hashing, so the fingerprint depends on the *meaning* of the request, not its byte-level formatting. This is the kind of detail that works in every test (tests send identical bytes) and breaks in production (a different client library serializes differently). A senior writes the canonicalizer; a junior hashes the raw bytes and gets paged.

## The increment trap and the upsert fix

The most common non-idempotent operation hiding in an otherwise-careful service is the humble increment, so let us fix it concretely. Suppose the inventory service consumes `OrderPlaced` events and decrements stock. The naive handler:

```sql
-- NOT idempotent. A redelivered OrderPlaced decrements TWICE.
UPDATE inventory SET available = available - 1 WHERE sku = 'WIDGET-7';
```

At-least-once delivery guarantees this event arrives more than once eventually, and each redelivery removes another unit. Over a week of redeliveries your `available` count drifts below reality and you start overselling — or, after a compensating correction, underselling. The fix is to make the decrement idempotent by making it *conditional on having not already applied this specific event*:

![A before and after comparison showing a non-idempotent relative increment that double-counts on retry versus an idempotent upsert to an absolute value that is safe to replay](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-7.webp)

```sql
-- Idempotent. The reservation row's PK is the order id, so a
-- redelivery collides on INSERT and applies nothing. The decrement
-- happens at most once per order, regardless of redelivery count.
INSERT INTO reservations (order_id, sku, qty)
VALUES ('order-42', 'WIDGET-7', 1)
ON CONFLICT (order_id) DO NOTHING;

-- Then derive available from the reservation ledger, or decrement
-- only when the INSERT above actually inserted a row:
UPDATE inventory
SET available = available - 1
WHERE sku = 'WIDGET-7'
  AND NOT EXISTS (   -- guard: only if we just created the reservation
    SELECT 1 FROM reservations_applied WHERE order_id = 'order-42'
  );
```

The general principle, shown in the figure above, is **convert a relative mutation into an idempotent one by attaching a deterministic identity to it**. The reservation is keyed by `order_id`; the second event for `order-42` collides on the primary key and does nothing. You have turned "subtract one" (which accumulates) into "ensure exactly one reservation exists for this order" (which is idempotent). The same trick works for money: instead of `balance += 80`, write a `ledger_entries` row keyed by a deterministic entry id and derive the balance as the sum — a redelivery collides on the entry id and the balance is unaffected. This is, not coincidentally, exactly how double-entry accounting and event-sourced systems achieve correctness; the broader pattern is in [event sourcing and CQRS in microservices](/blog/software-development/microservices/event-sourcing-and-cqrs-in-microservices).

## The dedup inbox: idempotent consumers for at-least-once messaging

The idempotency key is the right tool when a *client* drives the operation and can carry a key. But a message *consumer* pulling from a broker is in a slightly different situation: the "key" is the message's own id (or a deterministic id derived from its content), and the consumer's job is to record which message ids it has already processed and drop redeliveries. This is the **inbox** pattern (the mirror image of the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) on the publishing side), and it is how you make a consumer idempotent against at-least-once delivery.

![A stack showing an at-least-once event passing through an inbox dedup check before any handler side effect runs, with a TTL window bounding the store](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-5.webp)

The shape, as the figure shows: a message arrives, the consumer checks whether its id is already in the inbox, and only if it is *not* does the handler run — and the inbox insert and the handler's state change commit in the *same local transaction*, so you can never run the handler without recording the id (or record the id without running the handler). The inbox row has a TTL so the table does not grow without bound.

```python
# An idempotent consumer. The inbox insert and the business write
# share ONE local DB transaction, so the message is processed
# exactly once even though the broker delivers it at-least-once.
def handle_order_placed(msg):
    msg_id = msg.headers["message-id"]      # broker-assigned, stable on redelivery
    with db.transaction() as tx:
        try:
            # Atomically record that we've seen this message.
            tx.execute(
                "INSERT INTO inbox (message_id, processed_at) VALUES (%s, now())",
                (msg_id,))
        except IntegrityError:
            # Already processed in a prior delivery. Drop it, ack the
            # broker, move on. This is the dedup.
            return ACK

        # Same transaction as the inbox insert: business side effect.
        tx.execute(
            "INSERT INTO reservations (order_id, sku, qty) "
            "VALUES (%s, %s, %s) ON CONFLICT (order_id) DO NOTHING",
            (msg.body["order_id"], msg.body["sku"], msg.body["qty"]))
    return ACK   # commit succeeded; safe to ack the broker
```

The ordering here is exactly right and the reason is subtle: because the inbox insert and the business write are in one transaction, either *both* commit or *neither* does. If the process crashes after the commit but before the broker ack, the broker redelivers, the inbox insert hits the unique constraint, and we ack without re-running the handler. If the process crashes *before* the commit, nothing was written, the broker redelivers, and we process cleanly. Either way: exactly-once effect. This is the consumer-side discipline that [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) calls "effectively-once" — at-least-once on the wire, exactly-once in the database.

#### Worked example: how big is a 7-day dedup store at 50k messages/second?

The inbox has a real cost, and a senior sizes it before shipping it. Suppose the analytics ingestion service consumes `50,000` messages per second and you want a dedup window of 7 days (long enough to cover the worst redelivery scenario — a consumer that was down over a long weekend, a broker that replayed a partition). How big is the store?

`50,000 msg/s × 86,400 s/day × 7 days = 30.24 billion message ids`. If each inbox row is just a 16-byte message id plus a timestamp and index overhead — call it `64 bytes` all-in with the B-tree index — that is `30.24e9 × 64 = ~1.9 TB`. Nearly two terabytes of dedup metadata for a week's window, growing forever until the TTL eviction keeps pace. And the *write* load is brutal: `50,000` inserts per second into a uniquely-indexed table, each one a synchronous round-trip on the hot path of every message. That index becomes the bottleneck of the whole consumer long before the business logic does.

This is the **dedup window versus storage trade-off** in raw numbers. A longer window catches more late redeliveries but costs more storage and slows lookups; a shorter window is cheap but a redelivery that arrives *after* the window evicted its id slips through as a duplicate. At `50k/s` you do not store seven days of ids in one Postgres table — you reach for the optimizations in the next section. But the worked number is the point: **dedup is not free, and at high throughput the dedup store, not the handler, is your scaling problem.** Size it deliberately. A 7-day window at 50k/s is ~2 TB; a 1-hour window is ~`11.5 GB`; a 5-minute window is under a gigabyte. Choose the smallest window that covers your realistic redelivery horizon, because every extra hour costs you ~`11.5 GB` and proportional write amplification.

## The decision matrix: which guard, and what it costs

You now have the full toolkit, so here is the decision laid out as a matrix — the explicit "what you gain, what you pay, when it wins" that this series demands. Never reach for a guard without naming its cost.

![A matrix comparing natural idempotency, idempotency keys, a dedup store, and no protection across correctness, storage cost, complexity, and best fit](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-3.webp)

| Strategy | Correctness | Storage cost | Complexity | Where it fits |
| --- | --- | --- | --- | --- |
| **Natural idempotency** (`PUT`, `DELETE`, set absolute, upsert) | Exactly-once, structurally | None | Lowest — reshape the operation | Any write you can express as "set to a value" or "ensure this row exists" |
| **Idempotency key** (client-supplied, server-stored) | Exactly-once within key retention | One row per operation (`key`→`result`) | Medium — endpoint logic, key-reuse and in-progress handling | Client-driven non-idempotent writes: charges, `POST`s, order creation |
| **Dedup store / inbox** (message-id keyed) | Exactly-once within the dedup window | One row per message; large at scale | High — TTL, partitioning, hot-path index | At-least-once event consumers, saga step handlers |
| **No protection** | Duplicates pass through | None | None — and that is the trap | Truly nothing: only acceptable if the op is *already* naturally idempotent |

Read the matrix as a priority order, top to bottom. **Prefer natural idempotency** — it is free and it cannot drift, because there is no separate store to evict from or fall out of sync with. If the operation cannot be reshaped to be naturally idempotent, and a *client* drives it, use an **idempotency key**. If a *broker* drives it, use a **dedup store**. "No protection" is in the table only to name it as the default-by-omission that you must consciously reject — the *absence* of a decision is itself a decision, and it is the one that double-charges customers.

The reason the cost columns matter: natural idempotency has zero ongoing storage and zero drift risk, which is why it wins whenever it fits. The idempotency key costs you one row per operation but those rows are bounded by your operation rate and you can retire them after a retention window (Stripe retains keys for 24 hours, on the reasoning that a legitimate retry of the *same* operation will not arrive a day late). The dedup store is the expensive one — one row per *message*, and at high message rates that is the multi-terabyte problem from the worked example, which is why it earns the "high complexity" rating and the optimization section below.

## Concurrency: two requests with the same key, racing

The single most under-tested part of an idempotency implementation is what happens when two requests with the *same* key arrive *simultaneously* — close enough that both check the store, both see "not present," and both proceed to do the work. A double-click that fires two requests 5ms apart; a client that retries before the original's response arrives; two saga workers that both pick up the same step. If your dedup is "read, then write" without atomicity, both reads miss, both writes succeed, and you have double-charged *despite having an idempotency key*. The key store gave you a false sense of safety.

![A graph showing two concurrent requests with the same key racing on a unique constraint, where one writer wins the insert and the loser reads the stored result](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-6.webp)

The fix is the unique constraint, and it is why the `idempotency_keys` table above made `idempotency_key` the primary key. As the figure shows, both requests attempt to `INSERT` the key. The database guarantees that **exactly one of them wins**; the other gets a unique-constraint violation. The winner does the work and stores the result. The loser catches the violation, falls into the replay branch, and either returns the stored result (if the winner has finished) or gets a `409 in_progress` (if the winner is still running). The atomicity is delegated to the database's unique index, which is precisely the kind of single-point linearizable operation databases are extremely good at. You do *not* try to coordinate this in application code with locks and reads — you let the unique constraint be the referee.

```sql
-- The race-safe claim. Whichever of the concurrent requests reaches
-- the index first wins; the rest get a unique violation and replay.
-- This is atomic at the storage layer; no app-level lock needed.
INSERT INTO idempotency_keys (idempotency_key, request_hash, status)
VALUES ('charge:order-42', 'a1b2...', 'in_progress')
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING idempotency_key;
-- If RETURNING gives a row, we won and must do the work.
-- If it gives no row, someone else owns the key; we replay.
```

If your store is Redis rather than a SQL database, the same atomicity comes from `SET key value NX` (set-if-not-exists) — `NX` is the unique constraint of the key-value world. The principle is invariant across stores: **the check-and-claim must be a single atomic operation, not a read followed by a separate write.** A read-then-write has a window between the read and the write where a concurrent request slips through, and at any real traffic level that window *will* be hit. This is the bug that passes every single-threaded test and fails the instant two users (or two retries) overlap.

#### Worked example: the race that the unique constraint catches

Walk it through with timing. Two requests carry `Idempotency-Key: charge:order-42`. Request A arrives at `T+0ms`; request B (a double-click) arrives at `T+3ms`. Both run the claim `INSERT`. The database serializes them on the unique index: A's insert lands at `T+1ms` and commits the `in_progress` row. B's insert reaches the index at `T+4ms`, finds the key already present, and gets a unique violation — `0` rows from the `ON CONFLICT DO NOTHING ... RETURNING`. B falls into replay. At `T+4ms`, A is still charging the PSP (charges take ~`40ms`), so B reads `status = 'in_progress'` and returns `409 retry shortly`. B's client backs off `100ms` and retries at `T+104ms`; by then A has finished and stored `status='completed', charge_id=ch_xyz`, so B's retry reads the *stored* result and returns the *same* `charge_id`. The card was charged once. Both requests returned success. The customer sees one charge and one confirmation. Without the unique constraint, both A and B would have seen "key not present," both would have charged, and the card would show `\$160`. The constraint is doing the entire job; the application code is just reacting to which side of it landed.

## Optimization: making the dedup store production-grade

For client idempotency keys the store is rarely the bottleneck — operation rates are bounded by humans and the row count is modest. For the *dedup inbox* on a high-throughput event consumer, the store *is* the bottleneck, as the `50k/s` worked example showed: ~2 TB for a 7-day window and 50,000 uniquely-indexed inserts per second on the hot path. Here is how a senior makes that affordable, with numbers.

![A stack showing a bloom-filter prefilter, partitioning by key, a unique index, and a TTL eviction layered above a naive store that keeps every key forever](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-9.webp)

**TTL eviction is the first and biggest lever**, and the figure above puts it at the foundation: the naive design keeps every key forever and grows unbounded; the production design evicts after a window. Choose the smallest window that covers your realistic redelivery horizon. Most brokers redeliver within seconds to minutes; a redelivery a *week* later only happens after a catastrophic operational event. If you drop the window from 7 days to 6 hours, the store shrinks from ~2 TB to `50,000 × 86,400 × 0.25 × 64 bytes ≈ 69 GB` — a 28× reduction — at the cost of letting through any duplicate that somehow arrives more than 6 hours late, which for most workloads is "never." Measure your actual redelivery-age distribution (the gap between a message's first and last delivery) and set the window at, say, the 99.99th percentile of it plus margin. Do not set it to "7 days because it felt safe."

**Partitioning spreads the write load** so no single index is the bottleneck. At `50k/s` a single Postgres table's unique index will saturate; shard the dedup store by a hash of the message id across N partitions (or N Redis nodes, or N database shards) and each handles `50k/N`. The mechanics of spreading load by key hash are in [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding); the consequence here is that dedup throughput scales horizontally because each key's check is independent — there is no cross-partition coordination, since a given message id always hashes to the same partition. Ten partitions take you from 50k inserts/s on one index to 5k each, comfortably within a single node's budget.

**A bloom-filter prefilter cuts the durable-store load** for the common case, which is that a message is *new* (a first delivery, not a redelivery). A bloom filter is a compact probabilistic set that answers "have I maybe seen this id?" with either "definitely no" or "maybe yes." If your redelivery rate is, say, 0.1% (one message in a thousand is a duplicate), then 99.9% of lookups are first-time, and the bloom filter answers "definitely no" for the vast majority — no durable-store read needed, you go straight to processing. Only on a bloom "maybe yes" do you pay for the authoritative durable check (the bloom filter has false positives but never false negatives, so a "no" is trustworthy). A few hundred megabytes of in-memory bloom filter can absorb 99%+ of the lookup traffic, turning the durable store from a hot-path dependency on every message into a cold-path confirmation on the rare suspected duplicate. The measurable win: durable-store reads drop from `50,000/s` to roughly the redelivery rate plus the false-positive rate, often a `50–100×` reduction.

Stack these and the numbers transform. The naive design — keep every id forever in one uniquely-indexed table — is ~2 TB, 50k synchronous indexed inserts/s, and a single hot index. The production design — 6-hour TTL, 10 partitions, bloom prefilter — is ~69 GB total spread across 10 nodes (~7 GB each), ~5k inserts/s per node, and durable reads only on suspected duplicates. That is the difference between "the dedup store is the limiting factor of the whole pipeline" and "the dedup store is a rounding error." How to *measure* the win: track the durable-store read rate (should collapse toward your redelivery rate), the p99 of the dedup check on the hot path (should drop to the bloom-filter latency, sub-millisecond), the store size against your TTL projection, and — the real correctness signal — the rate of duplicates that slipped through (count effects that fired twice; it should be zero, and if it is not, your window is too short).

## Where the key store lives: co-locate it with the side effect

A question that decides whether your idempotency layer is actually correct, and which almost nobody asks until it has bitten them: *where* do you store the key — and is that store in the same transactional domain as the side effect it guards? The answer is a hard rule with a sharp edge.

For a side effect that is a **local database write** — decrement inventory, insert a reservation, update an order — the key store *must* be in the *same database* as that write, so the "record the key" and "do the effect" can commit in *one transaction*. This is exactly what the dedup inbox does: the inbox insert and the business write share one `BEGIN ... COMMIT`. If you instead put the key in a *separate* store (a different database, or Redis) and the side effect in your main DB, you have re-created the dual-write problem from the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing): two writes to two systems with no shared transaction, and a crash between them leaves the key recorded but the effect not done (next retry skips it — lost effect) or the effect done but the key not recorded (next retry repeats it — duplicate effect). The whole point of idempotency is to make retries safe, and a non-atomic key store reintroduces the exact bug it was supposed to prevent. So: **local side effect → key store in the same DB, in the same transaction.** Full stop.

For a side effect that is **external** — a PSP charge, an email send, a third-party API call — you *cannot* enroll it in your local transaction (the PSP does not participate in your Postgres commit), so you fall back to the two-phase record from the stress test: write `in_progress` *before* the call, write `completed` *after*, and reconcile stale `in_progress` rows against the external system's own idempotency. Here the key store is still best kept in your local DB (so the `in_progress`/`completed` state transitions are durable and atomic with any *local* bookkeeping), and the external call's safety comes from propagating the same key downstream so the *external* system dedups it too.

This drives the **Redis-versus-SQL** decision for the key store. Redis (`SET key val NX EX <ttl>`) is gloriously fast and the `NX` flag gives you the atomic claim — but Redis is a *separate* store from your business DB, so it is only correct when the side effect is *external* (the charge is external anyway, so the key store does not need to be transactional with a local write) or when a small window of duplicate effect is genuinely acceptable. Redis also can *lose* keys on failover (it is not durable by default), which means a key claimed just before a failover can vanish and let a retry through — fine for "send a notification at most slightly-more-than-once," catastrophic for "charge a card once." For money, the key store goes in the durable, transactional database alongside the ledger; for low-stakes high-throughput dedup, Redis with a TTL is the right cheap tool. Match the store's durability and transactionality to the cost of a duplicate effect: irreversible and expensive (money, shipments) → durable SQL co-located with the write; cheap and tolerable (a cache warm, a metric) → Redis is fine.

## Idempotency in sagas: re-drive must be safe

A saga ties everything in this post together, because a saga is a sequence of steps across services where the coordinator re-drives any step whose outcome it could not confirm — which is to say, a saga is a structured, deliberate retry machine. If the saga orchestrator sends "charge payment" and the response times out, it cannot tell whether the charge happened (the two-generals problem again, now at the business-workflow level), so it re-sends. **Every saga step must therefore be idempotent, or re-driving it corrupts the business transaction.** This is not an optional refinement; it is the precondition that makes the saga pattern work at all, and it is why the [saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice) post and this one are two halves of the same coin.

Concretely, in ShopFast's checkout saga — reserve inventory, charge payment, create shipment — each command carries a saga-step idempotency key derived from the saga id and step name, e.g. `saga:order-42:charge`. The payment service stores that key exactly as it stores any client key. When the orchestrator re-drives the charge step, the same key arrives, the payment service recognizes it, and it returns the original charge result instead of charging again. The re-drive is a no-op effect-wise, which is exactly what you need: the orchestrator gets confirmation that the step completed (it did, originally), the saga advances, and the card was charged once.

```python
# A saga orchestrator step. The idempotency key is DETERMINISTIC
# from the saga id and step name, so re-driving the step sends the
# SAME key and the downstream service dedups it. Re-drive is safe.
async def execute_charge_step(saga):
    key = f"saga:{saga.id}:charge"          # stable across re-drives
    result = await payment_svc.charge(
        amount=saga.total,
        card=saga.card,
        idempotency_key=key,                # downstream dedups on this
    )
    saga.record_step("charge", result)      # advance the saga
    return result
```

The same applies to **compensations** — the "undo" steps a saga runs to roll back. If the saga decides to compensate the charge (refund it) and the refund step is re-driven, the refund must also be idempotent or you refund twice. So compensations carry their own deterministic keys: `saga:order-42:charge:compensate`. The whole saga, forward steps and compensations alike, is a lattice of idempotent operations keyed by saga-id-plus-step, which is what lets the orchestrator re-drive *anything* freely without fear of duplicate effects. This is also why event-driven *choreography* sagas — where services react to each other's events rather than a central orchestrator — lean even harder on idempotent consumers: every reacting service is consuming at-least-once events and must dedup them, as covered in [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration).

## Stress test: three ways idempotency fails, and how to survive each

A senior does not trust a pattern until they have tried to break it. Here are the three failure modes that take down a naively-built idempotency layer, posed as the on-call questions you should be able to answer instantly.

**Stress 1 — "the response was lost but the request succeeded; the client retries."** This is the headline case and the one the whole post is built around. The charge succeeds server-side; the response is dropped on the wire; the client times out and retries with the same key. *Survival:* the server's key lookup hits the `completed` row and returns the stored response. The card is charged once, the client gets a success, and the lost response becomes a non-event. This is the case that *works by design* if you built the pattern correctly — and the one that *double-charges* if you skipped the key or made the lookup non-atomic.

![A timeline showing a charge that succeeds at the processor, a lost response that triggers a timeout, a blind retry that charges again, and a customer dispute](/imgs/blogs/idempotency-and-exactly-once-effects-across-services-4.webp)

The timeline above is this exact failure in the *unprotected* case — the one ShopFast lived through. At `T+0` the order service posts the charge. At `T+40ms` the PSP debits `\$80` successfully. At `T+200ms` the client times out and the response is lost. At `T+250ms` the order service retries *blind* — no key — and at `T+290ms` the PSP debits `\$80` again. At `T+5m` the customer disputes the `\$160`. Every event on that timeline is correct behavior given the previous one; the bug is the *absence* of a key, which is why the retry at `T+250ms` was indistinguishable from a fresh charge. Add the key and the `T+290ms` event becomes "PSP recognizes key, returns original charge" — the timeline ends safely.

**Stress 2 — "the same key arrives on two concurrent requests."** Two requests, same key, overlapping in time, both checking the store at once. *Survival:* the unique constraint. Exactly one wins the claim insert; the other gets a violation and replays. The atomic check-and-claim — `INSERT ... ON CONFLICT` in SQL, `SET NX` in Redis — is the referee. *Failure mode if you got it wrong:* a read-then-write dedup lets both miss and both charge. The cure is to never separate the check from the claim; make it one atomic operation against a unique index.

**Stress 3 — "the side effect happened but the key write crashed."** The nastiest one. The server charged the PSP, then crashed before recording `completed` in the idempotency table. Now the key row says `in_progress` (from the initial claim) but the charge actually went through. A retry arrives. *Survival* requires that the claim happened *before* the side effect (so there is a record that something was attempted) and that recovery can *reconcile*: on seeing a stale `in_progress` row past some threshold, the server queries the PSP *by the idempotency key* — "did a charge with key `charge:order-42` succeed?" — and, because the PSP is also idempotent on that key, it returns the original charge, which the server then records as `completed`. The key flowing all the way to the PSP is what makes this recoverable. *Failure mode if you got it wrong:* if you stored the key only *after* the side effect (not before), a crash in the gap leaves *no* record that the charge was attempted, and a retry charges again — the classic "partial side effect before the key is stored" bug. The fix is the two-phase record: claim `in_progress` *before* the effect, finalize `completed` *after*, and reconcile stale `in_progress` rows against the downstream's own idempotency. This is the same structural dance as the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) — never have a side effect that you have no durable record of having attempted.

There is a fourth subtle bug worth naming even though it is not a "what breaks under load" scenario: **non-deterministic stored responses.** If your handler generates a timestamp, a random id, or reads "now," and you store *that* response, then a replay returns a response that disagrees with reality at replay time — or worse, if you *recompute* instead of returning the stored response, the recomputed value differs from the original. The rule: **store the exact response bytes and return them verbatim on replay.** Do not recompute. The stored response is the single source of truth for "what the first call returned," and a replay must be byte-identical to it, because the client cannot tell — and must not need to tell — that it got a replay rather than the original.

## Case studies

**Stripe's idempotency keys.** Stripe is the reference implementation that the whole industry copies, and for good reason: a payments API is the one place where a double-effect is unambiguously catastrophic. Stripe lets the client send an `Idempotency-Key` header (they recommend a V4 UUID) on any `POST`. Stripe stores the key and the resulting response, and a retry with the same key returns the *original* response — the same charge, the same id, the same status — rather than creating a new charge. Stripe retains keys for 24 hours, on the explicit reasoning that a legitimate retry of the *same logical operation* will arrive within that window; a request a day later with a recycled key is treated as new. They also guard against key reuse with a different request body (the `request_hash` check in our endpoint) and return a conflict if you reuse a key that is still mid-processing. Every detail in our endpoint above traces directly to Stripe's published design. The lesson: in a payments context, the idempotency key is not optional polish — it is the load-bearing safety mechanism, and Stripe built their API so it is *impossible* to make a safe retry without one.

**Kafka exactly-once for effects.** Kafka markets "exactly-once semantics" (EOS), and it is worth being precise about what that actually delivers, because it is exactly the "exactly-once *effects*, not delivery" distinction from the top of this post. Kafka's EOS combines two mechanisms: the **idempotent producer**, which tags each produced record with a producer id and sequence number so the broker dedups records that get resent after a network retry (preventing the producer from writing the same record twice), and **transactions**, which let a consume-process-produce loop atomically commit its consumer offset *and* its produced output together, so a redelivered input does not produce duplicate output. The underlying delivery is still at-least-once — Kafka does not magically make the network reliable — but the *effect* (the records that land in the output topic and the offset that advances) happens once. The internals are in [exactly-once in Kafka: idempotent producer and transactions](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions). The lesson for a service author: Kafka EOS only covers effects *inside* Kafka (topic-to-topic). The moment your handler does something *outside* Kafka — charges a card, writes to Postgres, sends an email — you are back to building your own idempotency, because Kafka's transaction cannot enroll an external side effect. EOS is not a substitute for an idempotent handler; it is a complement that handles the in-Kafka half.

**A double-charge incident, generalized.** Double-charge post-mortems are a genre, and they share a script that ShopFast's followed exactly. The pattern: a client (often a mobile app on a flaky network) fires a payment request; the request succeeds server-side but the response is lost to a timeout or a dropped connection; the client retries — sometimes automatically, sometimes because the user, seeing no confirmation, taps "pay" again; and the payment backend, lacking an idempotency key, treats the retry as a fresh charge. The frequency is proportional to how flaky the client network is and how aggressive the retry policy is, which is why mobile-heavy and emerging-market apps see it worst. The fix is always the same: thread an idempotency key from the client through the backend to the processor. The instructive part is the *detection* lag — because a double-charge produces no error (both charges "succeeded"), it is invisible to error dashboards and surfaces only through customer disputes and chargebacks, often weeks later, exactly like ShopFast's "rare and infuriating" Tuesday. The lesson: idempotency bugs are *silent and proportional to success*; you do not find them by watching error rates, you prevent them by design, and you detect residual ones by reconciling effects (count charges per order; alert if any order has more than one).

## When to reach for this (and when not to)

The honest recommendation is unusually blunt for this series, because the cost/benefit is unusually lopsided: **use idempotency everywhere you retry or consume at-least-once — which, in a microservices system, is essentially everywhere there is a write.** The reasoning is that the *cost* of idempotency is small and bounded (an indexed table, thirty lines of endpoint logic, a TTL job), while the *cost of omitting it* is unbounded and silent (double-charges, double-shipments, drifted inventory, all invisible until a customer or a reconciliation job finds them). When the downside of skipping a guard is "we corrupt money and don't notice for weeks," the guard is not optional.

That said, calibrate the *mechanism* to the operation, per the matrix:

- **Naturally idempotent already?** Do nothing extra. A `PUT` that sets a status, a `DELETE` by id, an upsert keyed by a business id — these are safe to retry as-is. Spending engineering effort adding a key store on top of an already-idempotent operation is gold-plating. Recognize the freebie and move on.
- **Client-driven non-idempotent write?** Idempotency key. Charges, order creation, any `POST` that creates or accumulates. The client supplies the key; the server stores key→result.
- **At-least-once event consumer or saga step?** Dedup inbox. The message id (or a deterministic content id) is the key; the inbox insert shares a transaction with the business write.
- **Truly read-only?** Nothing. A `GET` has no effect, so retrying it is free — idempotency is a property of *writes*. (Be careful, though: a "read" that lazily creates or counts something is not actually read-only.)

The one place to *not* over-engineer is the dedup *window* and *store* for low-throughput consumers. If a consumer processes 50 messages a second, do not build a partitioned bloom-filtered multi-terabyte dedup layer — a single indexed table with a few-hour TTL is plenty, and the optimizations in this post are for the `50k/s` case, not the `50/s` case. Match the mechanism's cost to the throughput. The matrix tells you *which* guard; your throughput tells you *how elaborate* that guard's storage needs to be.

## Key takeaways

1. **You can never tell "my request failed" from "my request succeeded but the response was lost."** That ambiguity forces a choice between retrying (risk doing it twice) and giving up (risk not doing it). Idempotency dissolves the dilemma by making the retry safe.
2. **Exactly-once *delivery* is impossible** (two generals); **exactly-once *effects* is achievable and is what you actually want.** Stop fighting the network; make your operations tolerate redelivery.
3. **Duplicates are the default, not an edge case** — retries after timeouts, at-least-once brokers, client double-clicks, and saga re-drives all generate them constantly. Handle them deliberately or get paged for them.
4. **Prefer natural idempotency** — reshape relative operations (`+= 80`) into absolute ones (set, or upsert keyed by a business id). It is free, it cannot drift, and it beats every bolt-on guard when it fits.
5. **The idempotency-key pattern** (client supplies a stable key, server stores key→result, replay returns the stored result) is the default for client-driven non-idempotent writes. Store the exact response and return it verbatim; never recompute.
6. **A unique constraint is the concurrency referee.** Make the check-and-claim a single atomic operation (`INSERT ... ON CONFLICT`, `SET NX`); a read-then-write dedup lets concurrent same-key requests both slip through.
7. **The dedup inbox makes consumers idempotent** by recording processed message ids in the same transaction as the business write — exactly-once effect on top of at-least-once delivery.
8. **Dedup storage is not free at scale**; size the window deliberately, then cut cost with TTL eviction, partitioning, and a bloom-filter prefilter. At `50k/s` the dedup store, not the handler, is your bottleneck.
9. **Every saga step and compensation must be idempotent** with a deterministic saga-step key, or re-driving the saga corrupts the transaction. A saga is a retry machine; idempotency is its precondition.
10. **Idempotency bugs are silent and proportional to success.** You prevent them by design and detect residuals by reconciling effects (count charges per order), never by watching error dashboards.

## Further reading

- [Delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — why at-least-once is what you actually get from any durable broker, and why exactly-once delivery is a myth at the wire level.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the broker-side and consumer-side dedup mechanics that complement the service-design discipline here.
- [Exactly-once in Kafka: idempotent producer and transactions](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) — how Kafka delivers exactly-once *effects* for topic-to-topic loops, and why it stops at external side effects.
- [The transactional outbox pattern for reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — the publishing-side mirror of the dedup inbox; never have an effect you cannot durably account for.
- [The saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice) — the multi-service transaction whose every step must be idempotent so re-driving is safe.
- [The transactional outbox and reliable event publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) — how the publisher records the fact atomically, the dual-write trap, and the relay.
- [Resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — the retry machinery that makes idempotency mandatory, and the breaker that bounds it.
- [Event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — why choreographed consumers lean hardest on idempotent handlers.
- [Handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) — what to do when, despite idempotency, a downstream effect cannot be confirmed at all.
- Sam Newman, *Building Microservices* (2nd ed.) — the resilience and inter-service-communication chapters on retries and idempotency.
- Chris Richardson, *Microservices Patterns* — the messaging, idempotent-consumer, and saga chapters.
- Stripe API documentation, "Idempotent Requests" — the reference design for client-supplied idempotency keys.

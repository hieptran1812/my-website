---
title: "Idempotency Keys, Safe Retries, and the Exactly-Once Illusion"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The definitive guide to making unsafe operations safe to retry — why there is no exactly-once over an unreliable network, how the Idempotency-Key header turns a double-charge risk into a guaranteed single charge, and the server-side mechanics that survive timeouts, concurrent duplicates, and key reuse."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "idempotency",
    "retries",
    "payments",
    "distributed-systems",
    "reliability",
    "exactly-once",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-1.png"
---

A payment integration I was on-call for double-charged a customer \$49.99 at 2 a.m. on a Saturday. Nobody wrote a bug. The code was "correct." What happened was this: the client `POST`ed a payment, our service charged the card, and then — somewhere between our server writing the charge and the bytes reaching the client — a load balancer recycled a connection and the response never arrived. The client's HTTP library did exactly what a well-behaved HTTP library does on a timeout: it retried. The second `POST` charged the card again. Two charges, one intended payment, one very unhappy customer, and one engineer (me) trying to explain to the customer-support lead why "the system worked as designed" and "we charged them twice" were both true sentences at the same time.

That incident is the whole reason this post exists. The uncomfortable truth underneath it is that **a client that sends a `POST` and times out cannot know whether it succeeded.** The request might have failed before it reached the server. It might have succeeded and the *response* got lost on the way back. From the client's side, those two cases are indistinguishable — the symptom is identical, a dead socket. If the client retries, it risks a double charge. If it does not retry, it risks a lost payment that the customer thinks went through. There is no third option the network hands you for free. This is not a flaw in your framework; it is a property of unreliable networks, and no amount of careful coding on the request path makes it go away.

What *does* make it go away — or rather, what makes it survivable — is a small, precise contract between client and server called an **idempotency key**. An idempotency key is a unique token (in practice a UUID) that the client generates once per *logical* operation and attaches to every physical attempt of that operation. The server remembers the key, and on any replay it returns the *stored* result instead of executing again. The customer is charged once no matter how many times the request crosses the wire. By the end of this post you will be able to design that mechanism end to end — the `Idempotency-Key` header, the storage record, the atomic insert that survives two concurrent duplicates racing each other, the body fingerprint that catches a key reused with different parameters, the TTL that lets you reclaim storage — and you will be able to explain, honestly, why "exactly-once" is a marketing word and "effectively-once" is the real engineering goal.

![a sequence showing a client posting a payment, the response timing out, the client retrying with the same idempotency key, and the server returning the cached 201 with no second charge](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-1.png)

This sits squarely on the series' spine: an API is a contract, and the contract has to say what a caller may safely assume when the network betrays them. We have already established the [HTTP method semantics that make `PUT` and `DELETE` idempotent and `POST` not](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete), and the [status codes that tell the truth about what happened](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx). This post is where we make the one method that is *not* naturally idempotent — `POST` — safe to retry anyway. We will use the running Payments & Orders API throughout: `POST /payments` with an `Idempotency-Key`, a timeout, and a safe retry that returns the cached `201`.

## 1. The core problem: networks are at-least-once

Start from the wire, not from the diagram. When a client sends an HTTP request, the request travels as packets across routers, load balancers, proxies, and finally your application server, which does some work and sends a response back along a (possibly different) path. Each hop can drop, delay, duplicate, or reorder packets. TCP papers over a lot of this for a single connection, but TCP cannot help you once the connection dies — and connections die constantly in production from idle timeouts, load-balancer recycling, deploys, GC pauses, and transient network partitions.

Here is the part that trips up even experienced engineers. Consider the four places a single request/response exchange can fail:

1. The request never reaches the server (dropped on the way in). **Nothing happened.**
2. The request reaches the server, but the server crashes before doing the work. **Nothing happened.**
3. The server does the work and crashes before responding. **The work happened.**
4. The server does the work and responds, but the response is lost on the way back. **The work happened.**

From the client's vantage point — a socket that timed out or reset — cases 1 and 2 look *exactly* like cases 3 and 4. The client observes "no response." It cannot distinguish "the charge never happened" from "the charge happened and I just didn't hear about it." This is the crux: **the absence of a response is not the absence of an effect.**

### Why retrying is mandatory, not optional

You might be tempted to say: fine, on a timeout, just don't retry. But that strategy loses real operations in cases 1 and 2, where nothing happened and a retry would have completed the user's intent. In a system that processes millions of requests, transient failures are not rare events — they are a steady background rate. If your client gives up on every timeout, you systematically lose a fraction of every operation, and for payments that means customers who think they paid but didn't, orders that vanish, and a reconciliation nightmare. Retrying is the *correct* behavior for a robust client; it is what every mature HTTP library, SDK, and gateway does by default.

So we are cornered. The network forces *at-least-once* delivery semantics on any client that wants to not lose operations: keep trying until you get an acknowledgement, which means a successful operation may be *delivered* — and therefore executed — more than once. The mirror image, *at-most-once*, is what you get if you never retry: you never duplicate, but you may lose operations on the first failure. There is no transport-level setting labeled "exactly-once" that you can switch on.

### The formal claim: there is no exactly-once over an unreliable network

Let me make this rigorous, because it is the load-bearing claim of the entire post. Model the channel as one that may drop any message with some probability, and where the sender cannot observe whether a message was delivered. Define the two failure modes precisely:

- **At-most-once**: every operation is executed zero or one times. Achievable by never retrying — but it admits loss (zero executions when one was intended).
- **At-least-once**: every operation is executed one or more times. Achievable by retrying until acknowledged — but it admits duplication (more than one execution).

The claim is that **no protocol over a lossy, acknowledgement-based channel can guarantee exactly-once *delivery*.** The informal proof is a symmetry argument. Suppose a sender transmits a message and is waiting for an ack. Two scenarios are observationally identical to the sender: (a) the message was lost, no ack will come; (b) the message arrived, was processed, and the *ack* was lost. In scenario (a) the sender must resend to avoid loss; in scenario (b) the sender must *not* resend to avoid duplication. But the sender cannot tell which scenario it is in — the observable state (no ack yet) is the same in both. Any decision rule the sender adopts ("resend after timeout $T$") will be wrong in one of the two scenarios for some choice of when the loss occurred. Therefore no sender-side strategy can simultaneously avoid both loss and duplication. Exactly-once *delivery* is impossible. This is the same result that underlies the [delivery-semantics discussion in the message-queue series](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — it is not specific to HTTP; it is a property of distributed messaging in general.

To make the symmetry concrete, fix the timeout at $T$ and ask what the sender does as the actual delivery time $t$ varies. If the message and its ack both arrive before $T$, all is well. But there is always a $t$ such that the ack arrives at $T + \epsilon$ — just after the sender gave up waiting. At that instant the sender's observable state (no ack seen) is *identical* to the state it would be in if the message had been lost entirely. The sender's policy must produce the same action in both cases because it cannot distinguish them, yet the *correct* action differs: resend in the loss case, do not resend in the late-ack case. No finite $T$ escapes this — pushing $T$ larger just moves the boundary, it never removes it. This is why the impossibility is fundamental and not an artifact of a too-short timeout.

What you *can* do is decouple **delivery** from **processing**. The network gives you at-least-once delivery. If the *processing* is **idempotent** — meaning executing it twice has the same effect as executing it once — then duplicated deliveries do no harm. At-least-once delivery plus idempotent processing equals what the industry calls **effectively-once** (sometimes "exactly-once semantics" or EOS, a phrase that smuggles in an asterisk). The operation is *executed* at-least-once at the transport level, but its *effect* is applied exactly once because the duplicate executions are recognized and suppressed. That asterisk — "semantics, not delivery" — is the whole game.

There is a deeper reason this works: idempotency turns a *temporal* problem (did this already happen?) into a *state* problem (does a record of this exist?). The network cannot tell the sender whether an effect happened, but the *server* can — it simply records that the effect happened, keyed by something stable the client controls. The client's job shrinks to "carry a stable identity across attempts"; the server's job is "check the record before acting." Neither side has to solve the impossible distinguishability problem, because the *key* makes the duplicate recognizable. This reframing — from "guarantee single delivery" to "recognize and absorb duplicates" — is the single most useful way to reason about reliable distributed messaging.

> **Idempotent (definition).** An operation is idempotent if performing it multiple times produces the same result as performing it once. `GET /payments/pay_123` is idempotent (reading twice changes nothing). `DELETE /orders/ord_9` is idempotent (deleting an already-deleted order leaves it deleted). `POST /payments` to *create* a charge is **not** idempotent by default — each call creates a new charge. The idempotency key is the mechanism we bolt on to make a non-idempotent `POST` behave idempotently.

The figure above traces the happy path of the solution: a `POST /payments` carrying key `k1` charges once and stores the result; the response is lost and the client times out; the client retries with the *same* key `k1`; the server recognizes `k1`, skips the second charge, and returns the cached `201`. One charge, one customer, two physical requests, one logical operation.

## 2. The Idempotency-Key header pattern

The pattern is disarmingly simple to state and surprisingly subtle to implement correctly. The client generates a unique key per *logical operation* and sends it in a request header. The server stores `key → result`. On any request carrying a key it has already seen and completed, the server returns the stored result without re-executing.

The header name has effectively standardized on `Idempotency-Key`. There is an [IETF draft, "The Idempotency-Key HTTP Header Field"](https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/), that documents the pattern that Stripe, PayPal, Adyen, and others converged on independently. It is not yet a ratified RFC, but the header name and semantics are de facto industry consensus, so use exactly that spelling.

Here is the first request the client sends. Note the key is a UUID the client generated locally — not something the server handed out:

```http
POST /v1/payments HTTP/1.1
Host: api.commerce.example
Authorization: Bearer <token>
Idempotency-Key: 5f3a9b1c-7d22-4e6a-9c0f-2b1e8d4a6f01
Content-Type: application/json

{
  "order_id": "ord_8842",
  "amount": 4999,
  "currency": "usd",
  "payment_method": "pm_card_visa"
}
```

A couple of details that matter and are easy to get wrong. The `amount` is `4999` — an integer count of the smallest currency unit (cents), not a float `49.99`. Representing money as a floating-point number is its own category of production incident; integer minor units sidestep binary-rounding bugs entirely. And the `Idempotency-Key` is a version-4 UUID: 122 bits of randomness, collision probability negligible for any realistic volume. The client must generate it *before* the first attempt and *reuse the same value across all retries of that attempt.* That is the entire client-side contract.

On the first request, the server does the real work and returns `201 Created`:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Idempotency-Key: 5f3a9b1c-7d22-4e6a-9c0f-2b1e8d4a6f01
Location: /v1/payments/pay_3kQ9

{
  "id": "pay_3kQ9",
  "order_id": "ord_8842",
  "amount": 4999,
  "currency": "usd",
  "status": "succeeded",
  "created": "2026-06-20T08:14:22Z"
}
```

Now the response gets lost in transit. The client times out and retries — *byte-for-byte the same request, including the same `Idempotency-Key`.* The server recognizes the key, finds the stored result, and returns it. Many implementations add a response header so the client can tell a replay from a fresh execution:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Idempotency-Key: 5f3a9b1c-7d22-4e6a-9c0f-2b1e8d4a6f01
Idempotent-Replayed: true
Location: /v1/payments/pay_3kQ9

{
  "id": "pay_3kQ9",
  "order_id": "ord_8842",
  "amount": 4999,
  "currency": "usd",
  "status": "succeeded",
  "created": "2026-06-20T08:14:22Z"
}
```

Same `201`, same `pay_3kQ9`, same `amount`. The card was charged once. The `Idempotent-Replayed: true` header is a courtesy — the client did not strictly need it, because the response is identical either way, which is the whole point — but it is useful for client-side logging and metrics.

### What "one logical operation" means

The single most common client-side mistake is generating a new key on every retry. If your retry loop calls `uuid4()` inside the loop, every attempt looks like a brand-new operation to the server, and you are right back to double-charging. The key must be generated **once**, *outside* the retry loop, bound to the user's *intent* to pay — not to the HTTP attempt. A correct client looks like this:

```python
import time
import uuid
import requests

def charge_order(order_id: str, amount: int) -> dict:
    # Generate the key ONCE, bound to this logical payment.
    idem_key = str(uuid.uuid4())
    body = {
        "order_id": order_id,
        "amount": amount,
        "currency": "usd",
        "payment_method": "pm_card_visa",
    }
    headers = {
        "Authorization": "Bearer <token>",
        "Idempotency-Key": idem_key,  # same value on every attempt below
        "Content-Type": "application/json",
    }

    for attempt in range(5):
        try:
            resp = requests.post(
                "https://api.commerce.example/v1/payments",
                json=body,
                headers=headers,
                timeout=10,
            )
            # Retry only on transport errors and 5xx; never on 4xx.
            if resp.status_code < 500:
                return resp.json()
        except requests.exceptions.RequestException:
            pass  # timeout / connection reset: retry with the SAME key
        backoff = min(2 ** attempt, 30)  # exponential, capped
        time.sleep(backoff)
    raise RuntimeError("payment failed after retries")
```

Notice `idem_key` is assigned before the loop and never reassigned. Notice the retry condition: retry on transport errors and `5xx` (server-side, possibly transient), never on `4xx` (the request itself is wrong — retrying it with the same key will only return the same `4xx`). And notice the exponential backoff with a cap, which we discuss in the retry section. This is the safe-retry shape; everything server-side exists to make this client behavior correct.

## 3. Server-side mechanics, in depth

The client contract is three lines. The server contract is where the engineering lives, because the server must be correct under three distinct adversaries at once: a sequential retry (same key, later), two concurrent duplicates (same key, simultaneously), and a key reused with a *different* body (same key, different request — almost always a client bug, possibly an attack). Let us build the server logic up from a single key lookup.

![a branching decision graph for an incoming key that forks into execute-and-store for a new key, replay-cached for a completed key, return 409 for a pending key, and return 422 for a key reused with a different body](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-3.png)

The figure shows the four-way fork. An incoming `POST` with a key and body triggers a lookup. The key is in one of four states, and each maps to a distinct response:

- **New key** → insert a `pending` record atomically, execute the operation once, store the `201` result, return it.
- **Completed key** (same body) → replay the stored `201` without re-executing.
- **Pending key** (a retry arrived while the first is still in flight) → return `409 Conflict`, or block briefly and then replay.
- **Reused key with a different body** → return `422 Unprocessable Entity` (or `409` in some designs), because the same key naming two different operations is a contract violation.

### Step one: insert the key atomically with a "pending" state

The most important and most subtly-wrong-able decision is *when* you write the key. The naive approach — execute the operation, then store the key — has a fatal gap: between executing and storing, a concurrent duplicate can slip in, see no key, and execute the operation a *second* time. The correct approach inverts the order: **insert the key in a `pending` state before doing any work**, using an atomic operation that fails if the key already exists. The unique constraint on the key column is your concurrency primitive.

In SQL, the table looks like this:

```sql
CREATE TABLE idempotency_keys (
    idempotency_key  TEXT        NOT NULL,
    account_id       TEXT        NOT NULL,
    endpoint         TEXT        NOT NULL,
    request_hash     TEXT        NOT NULL,   -- fingerprint of the body
    state            TEXT        NOT NULL,   -- 'pending' | 'completed'
    response_status  INT,                    -- e.g. 201, filled when completed
    response_body    JSONB,                  -- the stored representation
    locked_at        TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at       TIMESTAMPTZ NOT NULL,   -- created_at + TTL (e.g. 24h)
    PRIMARY KEY (account_id, endpoint, idempotency_key)
);
```

The composite primary key `(account_id, endpoint, idempotency_key)` is the scope — we will justify that scoping below. The atomic claim is an `INSERT ... ON CONFLICT DO NOTHING` (Postgres) that returns whether *this* caller won the insert:

```sql
INSERT INTO idempotency_keys
    (account_id, endpoint, idempotency_key, request_hash, state, expires_at)
VALUES
    ($1, $2, $3, $4, 'pending', now() + interval '24 hours')
ON CONFLICT (account_id, endpoint, idempotency_key) DO NOTHING
RETURNING idempotency_key;
```

If the `RETURNING` clause yields a row, *this* request won the race and is the elected executor — it proceeds to do the work. If it yields nothing, the key already exists, and this request must read the existing row and act on its state. There is no window in which two requests both believe they are the executor, because the database enforces the unique constraint atomically. This is the single most important line of the whole design.

### Step two: execute, then store the response

The elected executor performs the side effect — charge the card, create the payment row — and then updates the idempotency record to `completed`, storing the response status and body:

```sql
UPDATE idempotency_keys
SET state = 'completed',
    response_status = 201,
    response_body = $5,
    locked_at = NULL
WHERE account_id = $1 AND endpoint = $2 AND idempotency_key = $3;
```

There is a critical ordering and atomicity concern here that separates a correct implementation from a subtly broken one. Ideally the side effect (the charge) and the `completed` update happen in the same database transaction, so either both commit or neither does. But payments often call an *external* processor that is not part of your database transaction, which reintroduces the very at-least-once problem one layer down. The robust pattern is: (1) store enough to recover — record the in-flight attempt with the external processor's *own* idempotency token (more on AWS-style client tokens and Stripe's per-call keys in the case studies), (2) on a crash mid-flight, a recovery job or the next retry re-checks the external processor's status by that token rather than blindly re-charging. The principle is that **every layer that can be retried needs its own idempotency token**; you cannot make the whole chain effectively-once by securing only the outermost hop.

### Step three: what to do when a retry arrives mid-flight

Suppose the first request is still executing — `state = 'pending'`, the card charge is in flight — and a retry arrives. The retry's atomic insert fails (the row exists), it reads the row, and it sees `pending`. Now you have a design choice with two reasonable answers:

1. **Return `409 Conflict`** immediately, with a message like "a request with this idempotency key is already being processed." The client should wait and retry (which, by then, will usually find the key `completed` and get the cached result). This is the simplest correct behavior and what Stripe does — it surfaces `409` for a key whose original request is still in progress.
2. **Block briefly** (a short poll loop with a timeout) and then return the now-`completed` result. This is friendlier to the client but ties up a server connection while it waits, and you must cap the wait to avoid pile-ups.

I default to `409` with a `Retry-After` header and let the client's backoff loop handle the wait — it keeps server connections short and pushes the waiting to the client, which is where backoff already lives:

```http
HTTP/1.1 409 Conflict
Content-Type: application/problem+json
Retry-After: 2

{
  "type": "https://api.commerce.example/errors/idempotency-in-progress",
  "title": "Request already in progress",
  "status": 409,
  "detail": "A request with idempotency key 5f3a9b1c... is still being processed. Retry shortly.",
  "instance": "/v1/payments"
}
```

That error body is [RFC 9457 `problem+json`](https://www.rfc-editor.org/rfc/rfc9457.html), the machine-readable error envelope we cover in the [error-design post](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx). The `409` is honest: the request did not fail, it is simply not done yet, and the client may safely try again.

### Step four: reject a key reused with a different body — `422`

Here is the subtle abuse case. Suppose a client sends key `k1` with a `\$49.99` charge, gets it processed, and then — through a bug, a code path that recycles keys, or malice — sends key `k1` again but with a `\$5,000.00` charge. If you blindly replay the cached `201`, you silently ignore the new (larger) request, which is wrong. If you execute the new request under the old key, you have defeated the whole point. The correct answer is to **detect that the key is being reused with a different request body and reject it with `422 Unprocessable Entity`** (some APIs use `400` or `409`; `422` communicates "the request is well-formed but semantically conflicts with prior use of this key").

How do you detect "different body"? You store a **fingerprint** — a hash of the canonicalized request body — alongside the key, and compare it on every reuse. That is the next section.

### The whole handler, in one place

It helps to see the four branches as one piece of server code rather than four prose paragraphs. Here is a complete idempotency middleware for the payments endpoint, in Python with a SQL store. Read it as the executable form of the decision graph above — the comments mark each of the four states:

```python
def handle_payment(request, db):
    key = request.headers.get("Idempotency-Key")
    if key is None:
        return problem(400, "missing-idempotency-key",
                       "This endpoint requires an Idempotency-Key header.")

    account_id = request.auth.account_id
    endpoint = "POST /v1/payments"
    fp = fingerprint(request.json)

    # State 1 — try to become the executor with an atomic insert.
    won = db.execute(
        """
        INSERT INTO idempotency_keys
            (account_id, endpoint, idempotency_key, request_hash,
             state, locked_at, expires_at)
        VALUES (%s, %s, %s, %s, 'pending', now(), now() + interval '24 hours')
        ON CONFLICT (account_id, endpoint, idempotency_key) DO NOTHING
        RETURNING idempotency_key
        """,
        (account_id, endpoint, key, fp),
    ).fetchone()

    if won is not None:
        # We are the elected executor. Do the work exactly once.
        try:
            payment = charge_card(request.json)  # the side effect
            body = serialize(payment)
            db.execute(
                """
                UPDATE idempotency_keys
                SET state='completed', response_status=201,
                    response_body=%s, locked_at=NULL
                WHERE account_id=%s AND endpoint=%s AND idempotency_key=%s
                """,
                (body, account_id, endpoint, key),
            )
            return response(201, body, headers={"Idempotency-Key": key})
        except TransientError:
            # Do NOT store a transient failure. Release so a retry can re-attempt.
            db.execute(
                "DELETE FROM idempotency_keys "
                "WHERE account_id=%s AND endpoint=%s AND idempotency_key=%s",
                (account_id, endpoint, key),
            )
            raise

    # The key already exists — read the row and branch on its state.
    row = db.execute(
        "SELECT request_hash, state, response_status, response_body "
        "FROM idempotency_keys "
        "WHERE account_id=%s AND endpoint=%s AND idempotency_key=%s",
        (account_id, endpoint, key),
    ).fetchone()

    # State 4 — key reused with a DIFFERENT body.
    if row.request_hash != fp:
        return problem(422, "idempotency-key-reuse",
                       "This key was first used with a different request.")

    # State 3 — original request still in flight.
    if row.state == "pending":
        return problem(409, "idempotency-in-progress",
                       "A request with this key is still being processed.",
                       headers={"Retry-After": "2"})

    # State 2 — completed; replay the stored result verbatim.
    return response(row.response_status, row.response_body,
                    headers={"Idempotency-Key": key, "Idempotent-Replayed": "true"})
```

Every branch of the figure is one path through this function: the atomic insert decides executor vs. not; the fingerprint comparison guards against reuse; the `state` field distinguishes in-flight from completed. The `TransientError` handler is the subtle bit most implementations miss — on a transient failure we *delete* the `pending` record so the next retry genuinely re-attempts, rather than caching a failure (we return to this in the stress-test section). Notice too that the executor's `charge_card` and the `completed` update should sit in one transaction when the side effect is local; when it is an external processor call, you instead record the processor's own token before the call so a crash mid-flight is recoverable.

### Documenting the contract in OpenAPI

Because the idempotency key is part of the *contract*, it belongs in your spec, not buried in a wiki page. An [OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) fragment makes the header, its requirement, and the conflict responses discoverable to anyone generating a client:

```yaml
paths:
  /v1/payments:
    post:
      summary: Create a payment
      parameters:
        - name: Idempotency-Key
          in: header
          required: true
          description: >
            A client-generated UUID identifying this logical operation.
            Reuse the same value on every retry. Results are stored for 24h.
          schema:
            type: string
            format: uuid
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/PaymentCreate"
      responses:
        "201":
          description: Payment created (or replayed for a repeated key).
          headers:
            Idempotent-Replayed:
              description: Present and true when this response is a replay.
              schema: { type: boolean }
        "409":
          description: A request with this idempotency key is still in progress.
          content:
            application/problem+json:
              schema: { $ref: "#/components/schemas/Problem" }
        "422":
          description: The idempotency key was reused with a different request body.
          content:
            application/problem+json:
              schema: { $ref: "#/components/schemas/Problem" }
```

Marking the header `required: true` and documenting the `409`/`422` responses turns "be careful to send a key" from tribal knowledge into a contract that codegen and linters enforce. A client SDK generated from this spec will surface the header as a mandatory argument, which is exactly the nudge you want.

## 4. The fingerprint: detecting key reuse with different parameters

A **fingerprint** is a fixed-length hash (SHA-256 is standard and overkill-safe) of the request's meaningful content, computed at the moment you store the key. On any reuse of the key, you recompute the fingerprint of the incoming request and compare. Match means "legitimate retry, replay the cached result." Mismatch means "this key has been used for a *different* operation, reject with `422`."

What goes into the fingerprint is a deliberate decision. You want to hash the request *body* and any parameters that define the operation, but **not** volatile metadata that legitimately changes between attempts — timestamps, the `Authorization` token (which may be refreshed), tracing headers, the `User-Agent`. A reasonable fingerprint is a hash of the canonical JSON body, with keys sorted so that `{"a":1,"b":2}` and `{"b":2,"a":1}` hash identically:

```python
import hashlib
import json

def fingerprint(body: dict) -> str:
    # Canonicalize: sort keys, no insignificant whitespace, UTF-8.
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Canonicalization matters because two JSON documents can be byte-different but semantically identical (key order, whitespace), and you do not want a legitimate retry that re-serialized the body in a different key order to be flagged as a mismatch. Sorting keys and stripping insignificant whitespace gives you a stable hash for the same logical content.

On a reuse with a mismatched fingerprint, the response is a `422`:

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/problem+json
Idempotency-Key: 5f3a9b1c-7d22-4e6a-9c0f-2b1e8d4a6f01

{
  "type": "https://api.commerce.example/errors/idempotency-key-reuse",
  "title": "Idempotency key reused with a different request",
  "status": 422,
  "detail": "Key 5f3a9b1c... was first used with a different request body. Use a new key for a new operation.",
  "instance": "/v1/payments"
}
```

This protects the client from its own bugs. A client that accidentally reuses a key for a genuinely different payment gets a loud `422` instead of a silent wrong result, which is exactly what you want a contract to do — fail visibly at the boundary rather than corrupt data quietly downstream.

#### Worked example: a timed-out POST retried with the same key returns the cached result

Let me walk the full cycle with concrete bytes, because this is the canonical case the whole pattern exists for.

**T+0ms.** The client charges order `ord_8842` for `\$49.99`. It generates `Idempotency-Key: 5f3a9b1c-...` once and sends:

```http
POST /v1/payments HTTP/1.1
Idempotency-Key: 5f3a9b1c-7d22-4e6a-9c0f-2b1e8d4a6f01
Content-Type: application/json

{"order_id":"ord_8842","amount":4999,"currency":"usd","payment_method":"pm_card_visa"}
```

**T+5ms.** The server runs `INSERT ... ON CONFLICT DO NOTHING`. No prior row exists, so the insert succeeds and returns the key — this request is the executor. It writes `state='pending'`, `request_hash=sha256(canonical_body)`.

**T+120ms.** The server charges the card (one charge, \$49.99), creates `pay_3kQ9`, updates the record to `state='completed'`, `response_status=201`, `response_body={...}`.

**T+121ms.** The server sends `201 Created`. But a proxy recycles the connection and the response is dropped. **The charge happened. The client heard nothing.**

**T+10,121ms.** The client's `timeout=10` fires. It catches the connection error and retries — same body, **same key `5f3a9b1c-...`**.

**T+10,126ms.** The server runs the same `INSERT ... ON CONFLICT DO NOTHING`. This time the row exists, so the insert returns nothing. The server reads the row: `state='completed'`. It recomputes the fingerprint of the incoming body and compares to `request_hash` — they match (same body). The server returns the stored response verbatim:

```http
HTTP/1.1 201 Created
Idempotency-Key: 5f3a9b1c-7d22-4e6a-9c0f-2b1e8d4a6f01
Idempotent-Replayed: true
Location: /v1/payments/pay_3kQ9

{"id":"pay_3kQ9","order_id":"ord_8842","amount":4999,"currency":"usd","status":"succeeded","created":"2026-06-20T08:14:22Z"}
```

**Result:** one charge of \$49.99, one payment `pay_3kQ9`, two physical requests, one logical operation. The customer is charged exactly once. The `Idempotent-Replayed: true` header tells the client this was a replay, but the client did not need it to be correct — the response is identical either way, which is precisely the property that makes the retry safe.

![a before and after comparison showing a retry without a key producing two charges totaling ninety-nine dollars and ninety-eight cents versus a retry with the same key producing one charge of forty-nine dollars and ninety-nine cents](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-2.png)

The figure contrasts the two worlds. On the left, no key: the retry is a *second* operation, the card is charged twice, total \$99.98. On the right, with a key: the retry is the *same* operation replayed, the card is charged once, total \$49.99. The key is what collapses two physical requests into one logical operation.

## 5. Concurrency: two duplicates racing on the same key

The sequential retry is the easy case — the first request has finished before the second arrives. The hard case is two duplicates arriving *simultaneously*. This happens more than you would think: a client with an aggressive retry that fires before the timeout, a user double-clicking "Pay," a load balancer that fans a request to two backends, or a mobile app that resends on network change. Both requests carry the same key. Both reach the server within milliseconds of each other. If your design has any window where both can read "no key" and both proceed to execute, you double-charge — and now you cannot even blame the network, because both requests *did* reach you.

![a graph showing two concurrent requests with the same key both attempting an atomic insert where one wins and executes the charge while the other violates the unique constraint sees the pending key and waits to replay the stored result](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-6.png)

The atomic insert is what saves you, and the figure shows why. Request A and request B both arrive with key `k1`. Both attempt the `INSERT ... ON CONFLICT DO NOTHING`. The database serializes these: exactly one of them wins the insert (say A) and becomes the executor; the other (B) hits the unique constraint, its insert returns nothing, and it reads the row to find `state='pending'`. B does not execute. B either returns `409` immediately or waits for A to finish and then replays A's stored `201`. The unique constraint is doing the mutual exclusion for you — it is a distributed lock that you get for free from any database with a unique index.

#### Worked example: two concurrent duplicates racing on the same key

**T+0.000s.** Request A (key `k1`) and request B (key `k1`) arrive 0.4 ms apart on two different application servers behind the load balancer. Both are about to charge `\$49.99` for order `ord_8842`.

**T+0.001s.** A executes `INSERT INTO idempotency_keys (...) VALUES (..., 'pending', ...) ON CONFLICT DO NOTHING RETURNING idempotency_key`. The insert succeeds; `RETURNING` yields the key. **A is the executor.**

**T+0.0014s.** B executes the *identical* insert. The unique constraint on `(account_id, endpoint, idempotency_key)` is now violated; `ON CONFLICT DO NOTHING` makes the insert a no-op; `RETURNING` yields nothing. **B is not the executor.** B reads the existing row: `state='pending'`.

**T+0.002s.** B has a choice. Returning `409` immediately is correct; for this example assume B does a short bounded wait (poll every 50 ms, cap 2 s) for A to complete.

**T+0.120s.** A finishes the charge (one charge, \$49.99), creates `pay_3kQ9`, updates the row to `state='completed'`, stores the `201` body. A returns `201`.

**T+0.150s.** B's next poll reads `state='completed'`, fingerprint matches, and B returns the *same* stored `201` for `pay_3kQ9`.

**Result:** two simultaneous requests, one `INSERT` won, one charge of \$49.99, two identical `201` responses pointing at the same `pay_3kQ9`. No double charge, even though both requests reached the server. The unique constraint was the entire safety mechanism.

A note on the alternative locking strategies, because you will see them in the wild. Instead of `INSERT ... ON CONFLICT`, some implementations use an advisory lock keyed on a hash of the idempotency key (`SELECT pg_advisory_xact_lock(hashtext($key))`), or a `SELECT ... FOR UPDATE` on a pre-existing row, or a distributed lock in Redis (`SET key value NX PX ttl`). All of these are valid; they differ in where the lock lives and how it expires. The unique-constraint-on-insert approach has the nice property that the *record itself* is the lock — there is no separate lock to leak — and it works on any relational database without extra infrastructure. If you reach for Redis `SET NX`, be careful: a Redis lock that expires before the operation finishes can let a second executor in, which is the classic Redlock controversy; the database row's `state` field is more robust because it persists and is transactional.

## 6. Scoping, TTL, and storage

Two operational decisions remain: what scope a key is unique within, and how long you keep it.

### Scope a key to an account and an endpoint

A raw UUID is globally unique with overwhelming probability, so you *could* make the idempotency key globally unique across your whole API. But that is a mistake for two reasons. First, **isolation**: customer A's key should never collide with customer B's key, even if (through a bad client or a copy-pasted example) two different accounts happen to send the same UUID. Scoping the key to the account means A's key `k1` and B's key `k1` are different records, and one customer can never observe or interfere with another's idempotency state. Second, **clarity of meaning**: the same key on `POST /payments` and on `POST /refunds` represents two genuinely different operations; scoping by endpoint keeps them in separate namespaces. That is exactly why the table's primary key is the composite `(account_id, endpoint, idempotency_key)` rather than `idempotency_key` alone.

The practical rule: **a key is unique within `(account, endpoint)`.** Stripe scopes idempotency keys per account (per API key, effectively). This also has a security benefit — an attacker who guesses or steals another account's key still cannot replay or poison their idempotency records, because the lookup is scoped to the authenticated account, not the bare key.

### TTL: expire stored keys

You cannot store idempotency records forever — that table would grow without bound, and an idempotency record for a payment made two years ago is useless because no client is still retrying it. So every record gets a **TTL** (time-to-live) and is purged after it expires. The industry-standard window is **24 hours**, which is what Stripe uses: idempotency results are stored for 24 hours, after which the same key is treated as new.

Why 24 hours and not, say, an hour or a week? It is a balance. The window must comfortably exceed the longest realistic retry horizon — a client whose request failed should still get the cached result on any retry it will plausibly make, including a human noticing a failure and re-running a job the next morning. But it should not be so long that storage costs balloon or that a *legitimately new* operation is blocked because a user happened to reuse a UUID from yesterday. Twenty-four hours covers essentially all automated and human retry loops with margin, and after that the key is reclaimable. You implement expiry either with a `DELETE WHERE expires_at < now()` sweep job, or natively with a TTL index in a store that supports it (Redis `EXPIRE`, DynamoDB TTL, MongoDB TTL index).

There is a subtle edge here: what if a retry arrives *after* the TTL has expired and the record has been purged? Then the server sees a new key, executes again, and you have a duplicate. In practice this is acceptable because no well-behaved client retries a 24-hour-old request — but it is why you size the TTL generously relative to your retry policy, and why the TTL is part of the *contract* you document, not an implementation detail you hide.

![a layered stack of the stored idempotency record showing the key as a client-generated UUID, the scope of account plus endpoint, the SHA-256 fingerprint of the request body, the pending or completed state, the stored 201 response, and the 24-hour TTL](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-5.png)

The figure lays out everything one record holds: the **key** (client-generated UUID), the **scope** (account + endpoint), the **fingerprint** (SHA-256 of the canonical body), the **state** (`pending` or `completed`), the **stored response** (status + body), and the **TTL** (expires in 24 h, then reusable). Every field earns its place — drop any one and you lose the ability to either replay correctly, detect reuse, handle the race, or reclaim storage.

| Concern | Without the field | With the field |
|---|---|---|
| Replay a retry | Cannot return the original result | `response_status` + `response_body` replay verbatim |
| Detect different-body reuse | Silently replay the wrong result | `request_hash` mismatch → `422` |
| Survive concurrent duplicates | Both requests execute | unique key + `state='pending'` elects one executor |
| Isolate customers | A's key can hit B's record | `account_id` in the primary key |
| Reclaim storage | Table grows forever | `expires_at` + sweep / TTL index |

### A storage alternative: Redis with an atomic claim

The SQL table is the most robust store because the record is transactional and durable, but many teams already run Redis on the hot path and prefer to keep the idempotency check there. The Redis version uses `SET key value NX` — set-if-not-exists, the atomic claim — as the executor election, mirroring the SQL `ON CONFLICT`:

```python
def claim_or_read(redis, scope_key, fingerprint, ttl_seconds=86400):
    # Atomic claim: only one caller wins NX.
    won = redis.set(scope_key, f"pending:{fingerprint}",
                    nx=True, ex=ttl_seconds)
    if won:
        return "executor", None       # we won — do the work
    stored = redis.get(scope_key)     # someone else owns it
    state, stored_fp = stored.split(":", 1)
    if stored_fp.split("|", 1)[0] != fingerprint:
        return "mismatch", None        # different body → 422
    if state == "pending":
        return "in_progress", None     # → 409
    return "completed", stored         # replay
```

Two cautions specific to Redis. First, the `ex=ttl_seconds` TTL is *also* your stale-lock timeout, which couples two concerns that the SQL design keeps separate (`expires_at` for reuse vs. `locked_at` for the in-flight lock) — if you set a 24-hour TTL, a crashed executor parks the key for 24 hours unless you add a separate, shorter lock with its own expiry. Second, Redis is typically not part of the same transaction as your card charge, so the "store the response after the side effect" step is two separate writes; if the process dies between them you are back in the `pending`-recovery case. For payments specifically I prefer the SQL store precisely because the idempotency record and the payment row can commit atomically together; reach for Redis when the operation is lower-stakes or when the latency of the SQL round-trip genuinely matters and you have a recovery path for the gap.

### Idempotency from the receiving side: webhooks

So far the client sends the key and the server dedups. The mirror image happens with **webhooks**, where *your* service is the receiver and a third party (Stripe, GitHub, a payment processor) is the sender that retries. The sender cannot tell whether your endpoint processed the event, so it retries on any non-`2xx` or timeout — at-least-once again, now pointed at you. The same impossibility result means *you* must dedup, using the event's own ID as the idempotency key:

```http
POST /webhooks/payments HTTP/1.1
Content-Type: application/json
Webhook-Id: evt_9fK2m1
Webhook-Signature: v1,<signature>

{"type":"payment.succeeded","data":{"id":"pay_3kQ9","amount":4999}}
```

The `Webhook-Id` (Stripe calls its delivered event ID `evt_...`; the standardized header in the Standard Webhooks spec is `webhook-id`) is the dedup key. Before acting on a webhook, record its ID in a `processed_events` table with a unique constraint; if the insert conflicts, you have already handled this event and you return `200` *without* re-processing. The shape is identical to the inbound idempotency-key design — atomic insert on the ID, skip the side effect on conflict — because it is the same problem from the other end of the wire. The broker-side internals of how senders retry and order these are covered in the [message-queue series](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once); here the point is that being an idempotent *receiver* is just as much a part of the contract as being a server that honors a client's key.

## 7. Idempotency keys vs the idempotent HTTP methods

It is worth being precise about how the idempotency key relates to the *built-in* idempotency of certain HTTP methods, because conflating the two leads to either redundant machinery or dangerous gaps. We covered the method semantics in depth in the [methods and idempotency post](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete); here is the part that matters for retries.

[RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html) defines a method as **idempotent** if "the intended effect on the server of multiple identical requests with that method is the same as the effect for a single such request." By that definition:

- `GET`, `HEAD`, `OPTIONS` are **safe** (no side effects) and therefore trivially idempotent.
- `PUT` is idempotent: `PUT /orders/ord_9` with a full representation sets the resource to that state; doing it twice leaves the same state. The *target state is fixed by the request*, so repetition is harmless.
- `DELETE` is idempotent: deleting `/orders/ord_9` twice leaves it deleted (the second call may return `404`, but the *state* is the same).
- `POST` is **not** idempotent: it means "process this payload according to the resource's own semantics," which for a collection like `/payments` means "create a new payment." Two `POST`s create two payments. That is the whole problem.
- `PATCH` is **not guaranteed** idempotent — a JSON Merge Patch that sets a field is idempotent, but a JSON Patch with an `add` to an array or an increment is not.

So the idempotent methods are safe to retry *for free* — the server promises that repetition does not change the effect. The idempotency *key* is the mechanism you bolt onto the non-idempotent methods (`POST`, sometimes `PATCH`) to *manufacture* that same retry-safety. Put bluntly: **`PUT` and `DELETE` are idempotent by their semantics; `POST` becomes effectively idempotent only because you remember the key.**

![a matrix comparing GET, PUT and DELETE, POST create, and POST with a key across whether each is idempotent, safe to retry, and the reason why](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-7.png)

The matrix summarizes it. `GET` is idempotent because it has no side effect. `PUT`/`DELETE` are idempotent because the target state is fixed by the request. `POST` to create is *not* idempotent and *not* safe to retry — each call creates a new row. `POST` with a key *is* safe to retry, because the key keys the row: the second call finds the existing record and replays.

| Method | Idempotent by spec? | Safe to retry on timeout? | Mechanism |
|---|---|---|---|
| `GET` / `HEAD` | Yes (also safe) | Yes | No side effect |
| `PUT` | Yes | Yes | Target state fixed by request |
| `DELETE` | Yes | Yes | Repeated delete = deleted |
| `PATCH` | Sometimes | Only if the patch is idempotent | Depends on the patch operation |
| `POST` (create) | No | No — risks duplicate | Each call creates a new resource |
| `POST` + `Idempotency-Key` | Effectively yes | Yes | Server dedups on the stored key |

A practical corollary: **if you can model an operation as a `PUT` with a client-chosen ID, you may not need an idempotency key at all.** If the client generates the resource ID and does `PUT /payments/{client_chosen_id}`, then a retry is a `PUT` to the same URL with the same body — naturally idempotent, no separate header needed. This is sometimes called "client-generated IDs" and it is a clean alternative when your resource model allows the client to name the resource. The idempotency-key header is the more general tool because it works even when the server assigns the ID (the common case for payments, where you do not want clients minting `pay_*` identifiers), but knowing the `PUT`-with-client-ID alternative keeps you from reaching for the heavier machinery when a simpler shape would do.

## 8. The exactly-once illusion, named honestly

Let me return to the claim from Section 1 and make it land, because it is the conceptual core and the place where well-meaning system descriptions go wrong. You will see "exactly-once" promised by message brokers, payment APIs, and stream processors. It is almost always shorthand for "exactly-once *semantics*," which is *not* the same as exactly-once *delivery*, and the difference is exactly the difference between marketing and engineering.

What you actually get, and the only thing you can get over an unreliable network, is **at-least-once delivery plus idempotent processing**. The operation may be *delivered* (and therefore *attempted*) more than once. The dedup layer — the idempotency key and its stored record — *recognizes the duplicates and suppresses the duplicate effects*. The *effect* lands exactly once. So the honest name is **effectively-once**: at-least-once delivery, exactly-once effect.

![a matrix comparing at-most-once, at-least-once, and effectively-once across whether duplicates can occur, whether operations can be lost, and how each is achieved](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-4.png)

The matrix puts the three semantics side by side. **At-most-once** never duplicates but may lose operations — it is "send once and give up," wrong for payments. **At-least-once** never loses but may duplicate — it is "retry until acknowledged," and it is what the network gives you. **Effectively-once** is at-least-once delivery *plus* server-side dedup on the key: it never loses and never duplicates the *effect*. That last row is the only acceptable target for a money-moving endpoint, and the idempotency key is how you reach it.

| Semantic | Duplicates? | Lost operations? | How achieved | Fit for payments? |
|---|---|---|---|---|
| At-most-once | Never | Possible | Send once, never retry | No — loses charges |
| At-least-once | Possible | Never | Retry until acknowledged | No — double charges |
| Effectively-once | Effect once | Never | At-least-once + idempotent dedup | Yes — the target |

This connects directly to the [message-queue delivery-semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once), which proves the same impossibility for brokers and shows how Kafka's "exactly-once semantics" is the same trick: at-least-once delivery with producer idempotence and transactional dedup. The HTTP idempotency key and the Kafka producer ID are the same idea wearing different headers. Whenever someone says "exactly-once," your reflex should be to ask "exactly-once *delivery* or *semantics*?" — the answer is always semantics, and the mechanism is always a dedup key plus at-least-once retries.

### Why this matters beyond pedantry

The illusion is not harmless. If you *believe* the network gives you exactly-once delivery, you will not build the dedup layer, and you will double-charge in production — which is precisely the incident that opened this post. Naming the semantics honestly forces the right architecture: you assume at-least-once everywhere, you make every side-effecting operation idempotent (via a key, a client-chosen ID, or a naturally idempotent design), and you treat "I saw this exact effect already" as a first-class state in your system rather than an impossible one. The illusion costs you money; the honest model costs you one extra table.

### Observability: prove the dedup is working

A dedup layer that silently fails is worse than no dedup layer, because you stop watching for double charges precisely when you have started causing them. So instrument the idempotency path as a first-class signal. The metrics worth emitting on every request that carries a key:

- **Replay rate** — the fraction of keyed requests that hit a *completed* record and were replayed. A healthy value is small but nonzero (it is the rate at which clients retry after a lost response); a sudden spike often means a client's response handling broke or a network path degraded. A *zero* replay rate on a high-volume payments endpoint is suspicious — it may mean clients are generating a fresh key per attempt (defeating dedup) rather than that nothing ever fails.
- **In-progress conflict rate** (`409`) — how often a retry arrives while the original is still in flight. Rising values mean either aggressive client retries firing before the timeout, or your executor is slow enough that retries overlap it.
- **Reuse-mismatch rate** (`422`) — keys reused with a different body. This should be near zero; a nonzero value is a client bug worth a ticket, not background noise.
- **Stale-lock takeovers** — how often a `pending` record exceeded its lock timeout and a retry took over. Any sustained rate here means executors are dying mid-flight and you should look at the side-effecting dependency.

Tie these to your RED metrics and traces so a single correlation/request ID threads the original attempt and all its retries together. Then a double-charge — if one ever slips through — shows up as two *executions* under what should have been one key, which your alerting can catch directly. The point of effectively-once is that you can *verify* it: count distinct effects per logical key, and assert it never exceeds one.

## 9. Retries done right: the client side of the contract

The idempotency key makes retries *safe*; it does not make a retry *policy* for you. A bad retry policy with a good idempotency key still hurts — it just hurts in latency and load instead of in double charges. The client side of the contract has its own rules.

**Retry only on retryable failures.** Retry on transport errors (connection reset, timeout) and on `5xx` and `429`. Do **not** retry on `4xx` other than `429` — a `400`, `401`, `403`, `404`, `409` (other than in-progress), or `422` means the request is wrong in a way that retrying will not fix; you will just burn quota getting the same error. The `429` is special: it means "slow down," and you should honor the `Retry-After` header before retrying.

**Use exponential backoff with jitter.** A naive "retry immediately, then again, then again" turns a transient blip into a self-inflicted denial of service — every client retries in lockstep and hammers the recovering server (the "thundering herd"). Exponential backoff spaces retries out: wait $2^{n}$ seconds before attempt $n$, capped at some maximum. **Jitter** — adding randomness to the delay — de-synchronizes clients so they do not all retry at the same instant. A common form is "full jitter": sleep a random amount between 0 and the exponential cap. The [dead-letter-queue and backoff post in the message-queue series](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) goes deeper on backoff math; the takeaway here is that backoff and the idempotency key are complementary — the key makes the retry *correct*, backoff makes the retry *kind to the server*.

```python
import random
import time

def retry_with_backoff(send, max_attempts=5, base=0.5, cap=30):
    for attempt in range(max_attempts):
        result = send()  # returns (status, body) or raises on transport error
        status = result[0]
        if status < 500 and status != 429:
            return result  # success or a non-retryable 4xx — stop
        # retryable: 5xx or 429
        sleep_for = min(cap, base * (2 ** attempt))
        sleep_for = random.uniform(0, sleep_for)  # full jitter
        time.sleep(sleep_for)
    raise RuntimeError("exhausted retries")
```

**Bound the number of retries.** Infinite retries on a genuinely down dependency keep a doomed request alive forever, holding connections and confusing your metrics. Cap attempts (5 is a common default) and then surface a clear failure to the user — with the idempotency key logged, so that a later manual retry or a recovery job can reuse it and benefit from the dedup.

**Honor `Retry-After`.** When the server sends `429` or `503` with `Retry-After`, that header is the server telling you exactly how long to wait. Override your backoff with it. Ignoring `Retry-After` is how a client gets itself rate-limited harder.

## 10. When to reach for idempotency keys (and when not to)

Idempotency keys are not free. They add a storage table, a hot lookup on the request path, a fingerprint computation, and a contract clients must follow correctly. So apply them where the cost is justified and skip them where the operation is already safe.

![a decision tree classifying an incoming write into read-only needing no key, naturally idempotent PUT or DELETE needing no key, and a side-effecting POST that requires a key](/imgs/blogs/idempotency-keys-safe-retries-and-exactly-once-illusions-8.png)

The decision tree is short. Classify the incoming write:

- **Read-only (`GET`)** → no key needed. Retry freely; reads have no side effects.
- **Naturally idempotent (`PUT`, `DELETE`, idempotent `PATCH`)** → no key needed. The method's semantics already make repetition safe. Adding a key here is redundant machinery.
- **Side-effecting non-idempotent `POST`** (charge a card, place an order, send money, dispatch an email, mint a coupon) → **require a key.** This is where the double-charge lives and where the cost pays for itself.

**Reach for a key when:**

- The operation moves money, creates a financial obligation, or has a real-world side effect that is expensive or impossible to undo (a charge, a refund, a payout, an order, a shipment, a notification).
- The operation is exposed to clients on unreliable networks (mobile, third-party integrations, anything over the public internet) where timeouts and retries are routine.
- Duplicate execution is materially harmful — not just an extra DB row, but a second charge, a second email, a second order.

**Do not reach for a key when:**

- The operation is a `GET` or otherwise has no side effects. Retrying is already safe; a key adds nothing.
- The operation is naturally idempotent (`PUT` to a known ID, `DELETE`). The semantics already cover you; prefer the simpler shape.
- A duplicate is harmless or self-healing. If `POST`ing the same analytics event twice just over-counts a metric you already tolerate noise in, the machinery may not be worth it — though here you should weigh whether the noise is truly harmless.
- You can restructure the operation to be a `PUT` with a client-chosen ID. That gets you idempotency from HTTP semantics without the extra header and table — reach for the lighter tool first.

A common anti-pattern worth calling out: requiring an idempotency key on *every* endpoint, including `GET`s and `PUT`s. This is cargo-culting. It adds a hot-path lookup and a contract burden to operations that are already idempotent, fragments your storage with useless records, and trains clients to treat the key as boilerplate they paste blindly rather than a meaningful per-operation token. Require it precisely where it earns its keep: side-effecting, non-idempotent writes.

## 11. Case studies: how the big payment APIs actually do this

These designs are worth studying because they converged independently on nearly the same solution, which is strong evidence the solution is right. I will keep to what is publicly documented and accurate; where I am summarizing a pattern rather than quoting a guarantee, I say so.

### Stripe — the reference implementation

Stripe popularized the `Idempotency-Key` header for HTTP APIs, and its public documentation describes the pattern precisely. Clients pass an `Idempotency-Key` header on `POST` requests (and other mutating calls). Stripe **stores the result of the first request and replays it on any subsequent request with the same key**, so retries do not create duplicate charges. Per Stripe's docs, **results are saved for 24 hours** — after that the key is treated as new. Stripe scopes keys per account (effectively per API key). Crucially, Stripe also **detects key reuse with a different request body**: sending the same key with different parameters returns an error rather than silently replaying, which is the fingerprint behavior we built in Section 4. Stripe recommends clients generate keys with enough entropy to avoid collisions (a V4 UUID is the canonical choice). When a request with a given key is still in progress, Stripe surfaces a conflict so the client retries rather than racing. This is the design this entire post mirrors, because Stripe's design is the one the rest of the industry standardized around.

### PayPal and Adyen — the same pattern, different header lifetimes

PayPal's APIs use a `PayPal-Request-Id` header on create-style calls, which serves the identical purpose: a client-generated unique ID that makes the operation safe to retry without creating duplicates. The semantics are the same as Stripe's idempotency key — first request executes and is stored, retries with the same ID return the original result. The retention window and exact behaviors are documented per-endpoint, so consult PayPal's reference for the specific call you are integrating rather than assuming a single global window. Adyen similarly supports an `Idempotency-Key` header on its payment requests, with the same store-and-replay semantics to prevent duplicate payments on retries. The pattern is uniform across the major processors; the header *name* and the retention *window* are the main things that differ, which is exactly why you read each provider's docs rather than assuming.

### AWS — client request tokens

AWS does not use a single global `Idempotency-Key` header; instead, many AWS API actions take a parameter commonly called a **client token** (for example, `ClientToken` on EC2's `RunInstances`, or `clientRequestToken` on various services). The semantics are the idempotency pattern by another name: the client supplies a unique token, AWS deduplicates requests carrying the same token within a service-defined window, and a retry with the same token returns the original result instead of launching a second set of resources. The window and exact behavior vary by service and are documented per-action — some are minutes, not 24 hours — so the lesson is the *pattern* generalizes while the *parameters* are per-service. The fact that AWS, Stripe, PayPal, and Adyen all arrived at "client-supplied unique token + server-side dedup window" independently is about as strong a signal as you get in API design that this is the correct shape.

### The common thread

Across all four: a **client-generated** unique token, attached to the **logical operation** (not the HTTP attempt), **stored server-side** with a **bounded retention window**, **replayed** on retries, and (in the better implementations) **fingerprinted** to reject reuse with a different body. The differences are cosmetic — header name, window length, scope granularity. The shape is the contract.

| Provider | Token mechanism | Scope | Retention | Reuse-with-different-body |
|---|---|---|---|---|
| Stripe | `Idempotency-Key` header | Per account | 24 hours (documented) | Returns an error |
| PayPal | `PayPal-Request-Id` header | Per call type | Per-endpoint (see docs) | Per-endpoint behavior |
| Adyen | `Idempotency-Key` header | Per merchant account | Per-endpoint (see docs) | Store-and-replay |
| AWS | `ClientToken` / request-token param | Per service/action | Per-service window (often minutes) | Per-service behavior |

Read the table as "same idea, four dialects." If you are integrating, the two cells you must check per provider are *retention window* (how long does a retry still get the cached result?) and *reuse behavior* (does a key with a changed body error, or silently replay the old one?). Getting the retention window wrong is the subtle one: a job that retries a failed payment the *next morning* will be outside a minutes-long window and will execute fresh — a duplicate. Match your retry horizon to the provider's window, or carry your own outer idempotency layer that you control.

### The before→after, walked concretely

It is worth stating the consequence of *not* having this, end to end, because the cost is invisible until it is not. Picture the same `POST /payments` endpoint shipped twice — once without an idempotency key, once with — and a fleet of mobile clients on flaky networks.

**Before (no key).** A mobile client on a subway charges \$49.99. The train enters a tunnel as the response is in flight; the socket dies; the client retries. The second `POST` reaches a server that has no memory of the first, so it charges again. At a steady 0.5% timeout-after-success rate on, say, 200,000 daily payments, that is roughly 1,000 double charges a day — 1,000 refund tickets, 1,000 chargebacks-in-waiting, and a support team that cannot tell a "real" double charge from a legitimate two-purchase customer because there is no key to correlate the duplicates. The fix after the fact is worse than the prevention: you are now writing reconciliation scripts that scan for two charges to the same card for the same amount within N seconds — a heuristic that has both false positives and false negatives, on production money.

**After (with key).** The same subway, the same dead socket, the same retry. The retry carries the same `Idempotency-Key`; the server finds the `completed` record and replays the original `201`. Zero double charges. The 1,000 daily duplicates simply do not exist, and the support load they generated does not exist either. The cost was one table, one atomic insert on the request path, and a header in the contract. This is the entire trade: a small, bounded, *designed-in* cost versus an unbounded, *discovered-in-production* cost that lands on customers and finance, not just engineering.

This is the series' contract-and-consequences frame in one stroke: the key is part of the contract you publish, and the double-charge is the consequence that lands on a caller you will never meet when the contract is missing.

## 12. Stress-testing the design

A good design earns trust by surviving the questions you ask it before production does. Let me push on the corners.

**What if the response is too large to store?** Idempotency records hold the full response body, which is fine for a payment object but a problem for an endpoint that returns megabytes. The mitigation is to store a reference (the created resource's ID and status) rather than the full body, and on replay re-fetch the resource — at the cost of a read. For most create endpoints the response is small (a payment object is a few hundred bytes) and storing it inline is fine; reserve the reference approach for genuinely large responses.

**What if the operation succeeds but the `completed` update fails?** This is the dangerous gap: the card is charged, but you crash before flipping `state` to `completed`, so the record stays `pending`. A naive retry sees `pending`, returns `409` forever, and the customer is stuck. The robust answer is a **reconciliation path**: the `pending` record carries enough information (the external processor's own idempotency token) that a recovery job — or the next retry after a timeout on the `pending` lock — can *ask the processor whether the charge succeeded* by that token, and either finalize the record (if it did) or release it for a clean retry (if it did not). This is why "every retryable layer needs its own idempotency token" is the governing principle: the outer key dedups your API; the processor's token dedups the charge; together they make the whole chain recoverable. Never re-charge blindly on a `pending` record.

**What if a malicious client floods you with random keys?** Each new key creates a `pending` record, so a flood of distinct keys is a write-amplification attack on your idempotency table. Defend with the same tools you use for any abuse: rate-limit by account, cap the size of stored bodies, and let the TTL reclaim the records. Because keys are scoped per account, one abusive account cannot pollute another's namespace.

**What if two retries arrive, the first is `pending`, and you chose the "block and wait" strategy — and the first never completes (the executor died)?** Your wait must be **bounded** (a few seconds), and a `pending` record must have a **lock timeout**: if `locked_at` is older than, say, 30 seconds with no completion, the record is considered stale and a retry may take over the execution (after re-checking the external processor, per the reconciliation path above). Without a lock timeout, a dead executor parks the key forever.

**What if the client sends no idempotency key at all?** You have a design choice: reject with `400` (require the key on this endpoint — strict, safest for payments) or accept and execute without dedup (lenient, but then retries are unsafe). For money-moving endpoints I lean toward *requiring* the key and documenting it as mandatory, returning a `400` with a `problem+json` body that names the missing header, so a client cannot accidentally opt out of safety. For lower-stakes side effects, accepting without a key and just not deduping is a reasonable trade.

**What about idempotency across a long-running operation?** When the `POST` kicks off async work and returns `202 Accepted` with a status resource, the idempotency key dedups the *initiation*, not the eventual completion. A retry of the initiating `POST` with the same key returns the same `202` and the same status-resource URL — it does not start the job twice. The completion is then tracked through the status resource, which is the subject of the [long-running-operations post](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks). The key still does its job: it makes the *trigger* effectively-once.

#### Worked example: classifying responses for the idempotency record

One last subtlety: *which* responses do you store and replay, and which do you let the client retry fresh? The rule is to store a result once the operation has reached a **terminal, deterministic outcome** — success or a definitive business failure — and to *not* store transient failures.

- The charge succeeds → store `201` + body, replay it. ✓
- The charge is declined by the bank (a real, terminal business outcome) → store the `402 Payment Required` (or your declined-payment representation), replay it. A retry should see the same decline, not re-attempt the card. ✓
- The server hit a transient `503` (a dependency was briefly down) → do **not** store this as the final result; the operation did not reach a terminal outcome. Release or leave the record so a retry can genuinely re-attempt. ✗ (do not cache)
- The request was malformed (`400`/`422`) → arguably store nothing for idempotency purposes, since the operation never began; a corrected request needs a new key anyway.

This is where many naive implementations break: they cache *every* response, including transient `5xx`s, so a request that *would* have succeeded on retry is permanently poisoned to return the cached `503`. Store terminal outcomes; let transient failures be retried.

## 13. Key takeaways

- **The network is at-least-once.** A client that `POST`s and times out cannot know whether the operation succeeded; retrying risks duplication, not retrying risks loss. This is a property of unreliable networks, not a bug you can code away.
- **Exactly-once delivery is impossible; effectively-once is the goal.** What you actually build is at-least-once delivery plus idempotent processing. The dedup layer suppresses duplicate effects so the *effect* lands exactly once. When anyone says "exactly-once," they mean *semantics*, not delivery.
- **The idempotency key makes a non-idempotent `POST` safe to retry.** The client generates a UUID once per *logical* operation (never per attempt), sends it as `Idempotency-Key`, and the server stores `key → result` and replays on any seen key.
- **The atomic insert is the safety mechanism.** Insert the key in a `pending` state with a unique constraint *before* doing the work; the database elects exactly one executor and parks concurrent duplicates. This is what survives two requests racing on the same key.
- **Fingerprint the body to catch reuse.** Hash the canonical request body, store it with the key, and reject a key reused with a different body via `422` — fail loudly at the boundary instead of silently returning the wrong result.
- **Scope keys to `(account, endpoint)` and give them a TTL.** Per-account scoping isolates customers and blocks key-poisoning; a 24-hour TTL covers realistic retry horizons and lets you reclaim storage.
- **`PUT`/`DELETE` are idempotent for free; `POST` needs the key.** If you can model the operation as a `PUT` to a client-chosen ID, you may not need the header at all — reach for the lighter tool first.
- **Require keys where duplicates hurt; skip them where the operation is already safe.** Side-effecting, non-idempotent writes earn the machinery; reads and naturally idempotent writes do not.
- **Every retryable layer needs its own idempotency token.** Your API key dedups your endpoint; the payment processor's token dedups the charge. Securing only the outer hop leaves the inner one duplicating.
- **Store terminal outcomes, not transient failures.** Caching a transient `503` poisons a request that would have succeeded on retry.

## 14. Further reading

- [Stripe — Idempotent requests](https://docs.stripe.com/api/idempotent_requests) — the canonical implementation: `Idempotency-Key` header, 24-hour result storage, per-account scope, and rejection of key reuse with a different body.
- [IETF draft — The Idempotency-Key HTTP Header Field](https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/) — the standardization effort documenting the header name and semantics the industry converged on.
- [RFC 9110 — HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html) — the formal definitions of *safe* and *idempotent* methods that ground why `PUT`/`DELETE` are idempotent and `POST` is not.
- [RFC 9457 — Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457.html) — the `application/problem+json` error envelope used for the `409` and `422` responses here.
- [PayPal — Idempotency (PayPal-Request-Id)](https://developer.paypal.com/api/rest/reference/idempotency/) — the same pattern under a different header name.
- Within this series: the intro hub [what an API is — the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the sibling posts on [methods and idempotency](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete), [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx), and [long-running operations](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks); and the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- The broker-side mirror of this whole argument: [delivery semantics — at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the same impossibility result and the same dedup-plus-retry solution, for message queues.

---
title: "Idempotency and Exactly-Once by Design: Safe Retries in a Distributed World"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Design every mutating operation to be safe to run twice — the Stripe-style idempotency-key pattern, race-safe dedup stores, effectively-once consumers, and how a senior reasons about exactly-once effect when exactly-once delivery is impossible."
tags:
  [
    "system-design",
    "idempotency",
    "exactly-once",
    "retries",
    "deduplication",
    "distributed-transactions",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/idempotency-and-exactly-once-by-design-1.webp"
---

The network will retry your request, and you do not get a vote. A client's HTTP library retries on a connection reset. A load balancer retries on a 502. A service mesh retries on a deadline. A message broker redelivers on a missing ack. A human clicks "Pay" twice because the spinner hung. Every one of these is correct behavior for the component doing it — and every one of them means your mutating operation is going to run more than once. The only question a senior asks is whether running it twice costs the company money, sends a customer two emails, or decrements an inventory count into the negatives. If the answer is "yes," and you have not designed for it, you do not have a bug yet. You have a bug scheduled for the next time the network hiccups under load, which is to say you have a bug.

Here is the claim this entire post rests on: **exactly-once delivery is impossible, but exactly-once effect is achievable, and the gap between those two sentences is where good distributed systems are built.** You cannot guarantee that a message crosses the network exactly one time — the two-generals problem says so, and no amount of cleverness repeals it. What you *can* guarantee is that no matter how many times a request arrives, it changes the world exactly once. You get there not by trying to stop the retries, but by making the operation *idempotent*: safe to apply repeatedly, with the same end state as applying it once. Figure 1 shows the whole stakes in one picture — a naive retry that double-charges a customer fifty dollars, versus the same retry carrying an idempotency key that replays the first result and bills them once.

![Before and after comparison of a naive retry that charges a customer fifty dollars twice versus an idempotency key that replays the first charge result and bills only once](/imgs/blogs/idempotency-and-exactly-once-by-design-1.webp)

By the end of this post you should be able to do four concrete things. First, look at any mutating operation and classify it: naturally idempotent, or in need of an explicit dedup mechanism. Second, design a race-safe idempotency-key flow the way Stripe does — client-generated key, server-side dedup store, request fingerprint to catch key reuse, response replay, TTL — and defend the dedup-store schema in a review when someone asks "what happens when two retries arrive at the same millisecond?" Third, make an at-least-once message consumer *effectively-once* by deduping on a stable id before the side effect runs. Fourth, optimize the whole thing: keep dedup cheap, size the TTL, build a fast path for the common no-retry case, and survive a dedup-store outage without either double-charging or going down. The mechanism deep-dives for message-layer dedup live in the `message-queue/` folder — this post is the architect's decision layer that sits on top of them.

## 1. Why exactly-once delivery is impossible (and why that is fine)

Start with the impossibility, because every design that ignores it is fragile. Two parties communicate over an unreliable network. Party A sends a message and wants to know it arrived. The only way to know is an acknowledgement from B. But the ack travels over the same unreliable network, so it can be lost too. If A's ack times out, A cannot tell the difference between "B never got my message" and "B got it and processed it, but the ack vanished." A has exactly two choices, and both are wrong some of the time: retry (and risk B processing twice) or give up (and risk B never processing at all). This is the **two-generals problem**, and it has no solution. There is no protocol, no number of round trips, that lets both sides become certain. Add a third hop and you have made it worse, not better.

What this means in practice: any delivery channel can promise you *at-most-once* (send, never retry — you might lose it) or *at-least-once* (retry until acked — you might duplicate it), but never *exactly-once* at the delivery layer. The full taxonomy of these guarantees is worked out in the [delivery semantics deep-dive](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once); the one-line summary is that every real system that cares about not losing data picks at-least-once and then *deduplicates*. At-least-once plus dedup equals effectively-once. That is the recipe. There is no other recipe.

So when a vendor's slide says "exactly-once," read the fine print, because they mean one of two honest things. Either they mean exactly-once *effect* via dedup (which is real and good — this is what Kafka transactions give you, covered later), or they mean exactly-once delivery *within a closed system where they control both endpoints and the storage* (also real, also limited — it does not survive crossing into your code's side effects). What they cannot mean is "the email gets sent exactly once even though sending email is an external side effect over a network we do not control." Sending money and sending email are the two canonical operations that no delivery guarantee can make exactly-once for you. You have to make those idempotent yourself.

The senior reframe is liberating: stop trying to prevent duplicates. You will fail, and the effort spent failing makes the system more complex and more fragile. Instead, *assume* duplicates, make every mutating operation safe under duplication, and now retries become a feature. A system that is idempotent everywhere can retry aggressively, fail over freely, and replay its message log for recovery — all the things that make distributed systems resilient — precisely because none of those retries can corrupt state. Idempotency is not a defensive patch. It is the property that unlocks aggressive reliability.

It is worth being precise about where the duplicates come from, because the variety surprises people and each source is a place a non-idempotent system breaks. *Client retries* fire when an HTTP library or a mobile SDK retries on a connection reset or a 5xx — and the dangerous subset is the retry after a *timeout*, where the original request may have actually succeeded but the response was lost. *Proxy and load-balancer retries* fire when an L7 proxy retries an idempotent-looking request against another backend, sometimes retrying a request that the first backend was still processing. *Service-mesh retries* fire when a sidecar retries on a deadline, multiplying every retry by every hop in the mesh — a three-hop call with retries enabled at each hop can turn one user action into many backend executions. *Broker redelivery* fires when a message queue redelivers an unacknowledged message, which, as we will see, happens on every consumer deploy, not just on failures. *Human duplicates* fire when a user double-clicks or refreshes a form. And *recovery replays* fire when you intentionally reprocess a message log or re-run a batch job after a failure. Every one of these is a legitimate, even desirable, behavior — the mesh *should* retry, the broker *should* redeliver, the operator *should* be able to replay the log. The only thing that makes all of them safe is that the operation underneath is idempotent. Take that property away and every one of these helpful behaviors becomes a corruption vector.

## 2. What "idempotent" actually means

An operation is idempotent if applying it N times (N ≥ 1) leaves the system in the same state as applying it once. The mathematical definition is `f(f(x)) = f(x)`. The engineering definition has a subtlety the math hides: idempotency is about the *end state*, not the *return value* or the *side effects along the way*. "Set the account balance to \$100" is idempotent — run it ten times, the balance is \$100. "Add \$100 to the account balance" is not — run it ten times, you have added \$1,000. The first replaces; the second accumulates. Replacing operations are naturally idempotent; accumulating operations are not. Almost every idempotency decision reduces to spotting which of these two shapes your operation has.

HTTP bakes this distinction into its method semantics, and the spec is worth taking seriously because it tells you which verbs you get idempotency from for free. `GET`, `PUT`, and `DELETE` are defined as idempotent. `PUT /users/42 {name: "Ada"}` sets the resource to a value — run it repeatedly, same result. `DELETE /users/42` removes it — run it repeatedly, it stays removed (the second call is a no-op or a 404, but the *state* is identical). `GET` does not mutate at all, so it is idempotent trivially. `POST` is explicitly *not* idempotent: `POST /charges` is supposed to create a new charge each time, which is exactly the behavior you do *not* want under a retry. The API design choices that flow from this — when to model an operation as PUT versus POST, how to expose idempotency keys in a REST or gRPC contract — are part of the broader [API design trade-offs](/blog/software-development/system-design/api-design-rest-grpc-graphql), but the idempotency core is this section.

There is a second subtlety that trips up even experienced engineers: **idempotent is not the same as side-effect-free, and it is not the same as commutative.** A `DELETE` is idempotent but has a side effect (the resource is gone). Two idempotent operations are not necessarily safe to reorder — `PUT name=Ada` then `PUT name=Babbage` gives a different result than the reverse, even though each is individually idempotent. When people say "make it idempotent," they almost always mean "make the *retry* of this *specific* operation safe," not "make this operation reorderable with all others." Keep those separate in your head; conflating them leads to over-engineering. The job is narrow: a retry of the same logical request must not change the state a second time.

#### Worked example: classifying five operations

Walk five real operations through the replace-vs-accumulate test, because the classification is the whole decision and it is faster than people expect. **(1)** `PUT /cart/items {sku: "A1", qty: 3}` — sets quantity to 3. Replace. Naturally idempotent, no key needed; retry it freely. **(2)** `POST /cart/items {sku: "A1", qty: 1}` meaning "add one" — accumulates. Not idempotent; a retry adds a second unit. Needs a key, or remodel as a PUT of the absolute quantity. **(3)** `POST /charges {amount: 4999}` — creates a charge, moves \$49.99 of real money. Accumulates (each call is a new charge). Not idempotent; *must* have a key, because the side effect is irreversible money movement. **(4)** `POST /accounts/42/credits {amount: 1000}` meaning "add \$10 of credit" — accumulates. Not idempotent; needs a key. **(5)** `DELETE /sessions/current` — removes the session. Replace (toward "absent"). Idempotent; the second call is a harmless no-op. Three of five need explicit protection, and the three are exactly the ones that touch money or accumulate — that is not a coincidence, it is the pattern. The decision tree in figure 4 encodes this test formally.

## 3. The idempotency-key pattern, done right

When an operation is *not* naturally idempotent — a charge, a credit, a "send this email" — you make it idempotent by attaching an **idempotency key**: a unique token, generated by the client, that represents the *intent* of one specific request. The server uses the key to recognize a retry and serve back the original result instead of executing again. This is the pattern Stripe pioneered for its API and the one every serious payments and commerce system has since copied. Figure 2 shows the canonical flow: the request carries the key, the server checks a dedup store, executes only if the key is unseen, records the result, and replays the stored response on any retry.

![Pipeline showing an idempotency key flowing through a store check, a single execution that charges fifty dollars, and a recorded result with a twenty-four hour time to live](/imgs/blogs/idempotency-and-exactly-once-by-design-2.webp)

Get five things right and the pattern is bulletproof. Get any one of them wrong and it leaks.

**1. The client generates the key, not the server.** This is the part people get backwards. If the server generated the key and returned it, the client would have to make a request to *get* the key before making the request to *use* it — and the first request is exactly the one that can be lost. The key must exist *before* the first attempt so that the retry can carry the *same* key. So the client mints a UUID (or any collision-resistant token) at the moment the user expresses intent — clicks "Pay" — and reuses that identical key on every retry of that one logical action. One intent, one key, many possible attempts.

**2. The dedup store maps key → result.** The server keeps a record per key: the key itself, the status (in-flight / completed), the stored HTTP response (status code and body) once complete, and a timestamp. On a request, it looks up the key. Unseen key → this is a first attempt, proceed. Seen-and-completed key → this is a retry, replay the stored response, do *not* execute. Seen-and-in-flight key → a concurrent duplicate is mid-execution; return a "request in progress" signal (Stripe returns a 409) so the client backs off and retries.

**3. The request fingerprint catches key reuse with a different body.** Here is the subtle one that separates a real implementation from a toy. A client could reuse an idempotency key with a *different* request body — a bug, or worse, a malicious replay. If you blindly replay the stored response, you silently ignore the new (different) request, which hides a real error from the caller. So you store a *fingerprint* of the request — a hash of the relevant fields (amount, currency, destination) — alongside the key. On a retry, you compare the incoming fingerprint to the stored one. Match → genuine retry, replay safely. Mismatch → the client reused a key for a different operation; return a hard error (Stripe returns 400 "keys cannot be reused with different parameters"). The fingerprint turns the idempotency key from "trust me it's the same" into "prove it's the same."

**4. Response replay returns the *original* result byte-for-byte.** When a retry hits a completed key, you do not re-run anything and you do not construct a fresh response — you return the *exact* response the first attempt produced, including the original status code, body, and ideally the original resource id. The whole point is that the client cannot tell whether it was the first attempt or the thousandth. If the first attempt created charge `ch_abc` and returned 201, every retry returns 201 with `ch_abc`. This is also why you store the full response, not just "success" — the client needs the charge id back.

**5. The TTL bounds the storage.** You cannot keep idempotency records forever; they would grow without bound. So each record carries a time-to-live. Stripe keeps idempotency keys for 24 hours. The TTL must be *longer than the longest plausible retry window* — if a client could retry after 25 hours and you expired the key at 24, you would execute again and double-charge. For most APIs, 24 hours is comfortably longer than any sane client retry schedule (clients give up in seconds to minutes), so the TTL is about storage reclamation, not correctness, as long as it exceeds the retry window. We will size this precisely in the optimization section.

A worked trace makes the five rules concrete. A customer clicks "Pay \$50." The client mints `ic_8f2a` and sends `POST /charges {amount: 5000}` with `Idempotency-Key: ic_8f2a`. The server checks the store: absent. It atomically claims the key as in-flight with fingerprint `sha256(...amount:5000...)`, charges the card, gets back charge `ch_abc`, records `{status: completed, response: 201, body: {id: ch_abc}}`, and returns 201. The network drops the response on the way back. The client's HTTP library times out and *retries* with the same `ic_8f2a` and the same body. The server checks the store: present, fingerprint matches, status completed — it replays the stored 201 with `ch_abc` *without charging again*. The customer was charged once, the client got a clean 201 with the charge id, and neither side can tell a retry happened. Now take the variant where the client retried with a *different* body (`amount: 9999` — a client bug). Same key, but the fingerprint mismatches, so the server returns 400 "key reused with different parameters" instead of silently replaying the \$50 charge for a request that asked for \$99.99. Every one of the five rules did a specific job in that trace; remove any one and a specific thing breaks.

```python
# Idempotency middleware (FastAPI-style). The store is injected; see section 5
# for the race-safe atomic-claim version of store.claim().
import hashlib, json
from fastapi import Request, Response, HTTPException

def fingerprint(method: str, path: str, body: bytes) -> str:
    # Hash the parts of the request that define the operation's identity.
    return hashlib.sha256(method.encode() + b"|" + path.encode() + b"|" + body).hexdigest()

async def idempotency_middleware(request: Request, call_next):
    key = request.headers.get("Idempotency-Key")
    # Only mutating, non-natural operations need a key. GET/PUT/DELETE skip this.
    if request.method != "POST" or key is None:
        return await call_next(request)

    body = await request.body()
    fp = fingerprint(request.method, request.url.path, body)

    record = store.get(key)
    if record is not None:
        if record["fingerprint"] != fp:
            # Same key, different request: reject loudly, do not silently replay.
            raise HTTPException(status_code=400,
                detail="Idempotency-Key reused with different request parameters")
        if record["status"] == "completed":
            # Genuine retry of a finished request: replay the stored response verbatim.
            return Response(content=record["response_body"],
                            status_code=record["response_status"],
                            media_type="application/json")
        # status == "in_flight": a concurrent duplicate is still running.
        raise HTTPException(status_code=409, detail="A request with this key is in progress")

    # First time we have seen this key. Atomically claim it (see section 5).
    if not store.claim(key, fp, ttl_seconds=86_400):
        # Lost the claim race to a concurrent duplicate; treat as in-flight.
        raise HTTPException(status_code=409, detail="A request with this key is in progress")

    response = await call_next(request)
    resp_body = b"".join([chunk async for chunk in response.body_iterator])
    store.complete(key, status_code=response.status_code, body=resp_body)
    return Response(content=resp_body, status_code=response.status_code,
                    media_type="application/json")
```

## 4. Natural idempotency versus operations that need a key

Not every operation needs the machinery of section 3. The first move a senior makes is to ask whether the operation can be *made naturally idempotent* by remodeling it, because natural idempotency costs nothing — no store, no key, no extra latency — while an idempotency key costs a store hit on every request. Figure 3 lays the four approaches side by side so you can see what each one buys and what it charges.

![Matrix comparing natural idempotency, idempotency keys, dedup tables, and dedup on consume across safety, storage cost, added latency, and cross-service reach](/imgs/blogs/idempotency-and-exactly-once-by-design-3.webp)

The matrix reads like this. **Natural idempotency** (PUT, SET, DELETE, "set status to X") is free and strong, but only available when the operation *replaces* rather than accumulates. Reach for it first, always. **The idempotency key** is the general-purpose tool: strong safety, costs one dedup-store hop and one record of storage per request, and — crucially — propagates across services. **A dedup table** (the key pattern's storage realized as a relational table with a unique constraint) gives the same safety and lets you make the dedup *atomic with the side effect* in the same database transaction, which is the cleanest correctness story when the side effect is itself a database write. **Dedup on consume** is the messaging variant: the consumer records each processed message id before acting, achieving effectively-once for an at-least-once stream, but the dedup is per-consumer and does not compose across services for free.

The decision is mechanical, and figure 4 turns it into a tree you can walk in a design review. Is the operation mutating at all? If not, it is idempotent trivially. Does it *replace* state (PUT, SET)? Then it is naturally idempotent — no key. Does it *accumulate* (POST, increment, append, send)? Then you need an explicit key. And if that accumulating operation has an *external side effect* that cannot be rolled back — moving money, sending an email, calling a third-party API — then a bare key is not enough; you need dedup *plus a status check before the irreversible action*, because the dangerous window is between "decided to act" and "acted," and a crash there must not re-send.

![Decision tree mapping an operation from mutating to replacing or accumulating, then routing accumulating operations with external effects to dedup plus a status check](/imgs/blogs/idempotency-and-exactly-once-by-design-4.webp)

The remodeling trick deserves emphasis because it is the cheapest win available. Many operations that *look* like they accumulate can be redefined to replace. "Add item to cart" (accumulate) becomes "set cart line for SKU A1 to quantity 3" (replace) — now a retry is harmless and you never needed a key. "Increment the like count" (accumulate) becomes "record that user U liked post P" in a set, then the count is `SELECT COUNT(*)` — the set insert is naturally idempotent (inserting the same (U, P) twice is a no-op with a unique constraint). "Append to the audit log" can carry a client-supplied event id with a unique constraint, making the append a dedup-table insert. Before you build idempotency-key infrastructure, ask whether the operation can be turned into a replace or a unique-constrained insert. Often it can, and then idempotency is *structural* — it falls out of the data model and needs no runtime machinery at all.

## 5. The dedup store and the race that ruins everything

Now the hard part, the part that separates implementations that work in the demo from implementations that survive production. The idempotency-key flow has a race condition lurking in it, and if you implement it the obvious way — check the store, then do the work, then record the key — you have built a system that double-charges under exactly the condition you built it to prevent: a concurrent retry. Figure 5 shows the race and its fix.

![Graph showing two concurrent retries both reading an absent key, then an atomic claim electing one winner that executes once while the loser awaits the result, versus both executing and double charging](/imgs/blogs/idempotency-and-exactly-once-by-design-5.webp)

Here is the race in slow motion. Two requests, same idempotency key, arrive within a millisecond of each other (a client retried because the first response was slow, but the first one was not actually lost — both are now in flight). Request A checks the store: key absent. Request B checks the store: key *also* absent, because A has not recorded it yet. Both conclude "first attempt, proceed." Both charge the card. Two charges, \$100, the exact failure mode the whole pattern exists to prevent. The naive check-then-act sequence is not idempotent at all under concurrency — it just *looks* idempotent in single-threaded testing, which is why this bug ships so often.

The fix is to **claim the key atomically before doing the work**, using a single operation that both checks and records in one indivisible step. The winner of that atomic claim does the work; everyone else sees the key already claimed and waits or returns "in progress." There are two clean ways to do this, and you should know both.

**Atomic claim with a relational unique constraint.** Insert the key row with `status = 'in_flight'` *before* doing the work. A unique constraint on the key column means the second insert *fails* — and that failure is your signal that someone else won. The loser does not execute; it returns 409 or polls for the winner's result. This is the dedup-table approach, and it has a beautiful property when the side effect is a database write: you can do the claim and the side effect in *the same transaction*, so they commit or roll back together.

```sql
-- Dedup table: the unique constraint on idempotency_key is the entire safety mechanism.
CREATE TABLE idempotency_keys (
    idempotency_key   TEXT        PRIMARY KEY,           -- client-generated, unique
    request_fp        TEXT        NOT NULL,              -- sha256 of method|path|body
    status            TEXT        NOT NULL DEFAULT 'in_flight',  -- in_flight | completed
    response_status   INTEGER,                           -- replayed on retry
    response_body     JSONB,                             -- replayed on retry
    resource_id       TEXT,                              -- e.g. the charge id created
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at        TIMESTAMPTZ NOT NULL               -- created_at + TTL (24h)
);
CREATE INDEX idx_idem_expires ON idempotency_keys (expires_at);  -- for TTL sweeps

-- Atomic claim: only the FIRST inserter wins. Concurrent duplicates hit the
-- ON CONFLICT branch and change nothing, so they learn "someone else has this key".
INSERT INTO idempotency_keys (idempotency_key, request_fp, status, expires_at)
VALUES ($1, $2, 'in_flight', now() + interval '24 hours')
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING idempotency_key;
-- If RETURNING yields a row, you are the winner: do the work, then UPDATE to 'completed'.
-- If it yields nothing, you lost: read the existing row and replay or return 409.
```

**Atomic claim with Redis `SET NX`.** When you want the dedup store off your main database for latency, Redis gives you the same atomicity with `SET key value NX EX ttl` — set the key *only if not exists*, with a TTL, in one round trip. The command returns OK to exactly one caller and nil to the rest. The winner proceeds; the losers see nil and know the key is claimed.

```python
# Redis atomic claim. SET NX is a single atomic op: exactly one caller gets True.
def claim(self, key: str, fingerprint: str, ttl_seconds: int) -> bool:
    # Store the in-flight marker + fingerprint as one value, only if the key is new.
    won = self.redis.set(
        name=f"idem:{key}",
        value=json_dumps({"status": "in_flight", "fp": fingerprint}),
        nx=True,            # only set if it does not already exist -> the atomic claim
        ex=ttl_seconds,     # TTL so a crashed in-flight key cannot wedge forever
    )
    return won is True       # True = you won the claim; None = someone else holds it
```

There is one more failure to handle, and it is the reason the in-flight status exists. What if the winner *claims* the key, starts the charge, and then *crashes* before recording the result? The key is now stuck in `in_flight` forever, and every retry sees "in progress" and gets a 409 — the operation is permanently wedged. Two defenses, used together. First, the TTL on the in-flight record means a crashed claim eventually expires and a fresh attempt can proceed (this is why the Redis `EX` and the SQL `expires_at` are not optional). Second, and more robust, make the claim and the side effect *atomic* where possible — in the SQL approach, do the charge-side-effect insert and the key-row update in one transaction, so a crash rolls back the claim too, and the next retry genuinely starts fresh. The in-flight-then-crash window is the single nastiest corner of this pattern, and "TTL plus same-transaction commit" is how you close it. The atomicity of the dedup record with the side effect is precisely the guarantee the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) gives you when the side effect is *publishing a message*, which is why outbox and idempotency are two faces of the same coin.

## 6. Where to keep the keys: dedup-store design

The dedup store is on the hot path of every mutating request, so its design directly sets your write latency and your storage bill. The three realistic homes for it each buy something different.

**Same database as the side effect (a dedup table).** This is the strongest correctness story and the one I reach for first when the side effect is itself a row in that database. Because the key claim and the business write live in the same transaction, they are atomic for free — no two-store coordination, no window where one committed and the other did not. The cost is that every idempotent request now writes a row to a table that can grow large and hot, and you have to sweep expired rows. For a service whose side effects are database writes (orders, ledger entries, account changes), the dedup table is usually correct *and* simplest. It also gives you a durable audit trail of every request the service has seen.

**A dedicated key-value store (Redis).** When you want the dedup decision off your primary database — because the primary is precious and you do not want to add a write to its hot path, or because you want sub-millisecond claims — Redis is the standard answer. `SET NX EX` is atomic and fast, TTL is built in (no sweep job — Redis expires keys for you), and you can scale it independently. The cost is that the dedup is now in a *separate* store from the side effect, so they are no longer atomic; you have re-introduced the "claimed but did not complete" window and must lean on the in-flight TTL and a careful ordering (claim, then do work, then mark complete, then — only on a *completed* mark — is replay safe). Redis is the right call when latency matters more than transactional simplicity and the side effect is external to your database anyway (a third-party charge), where you could not have made it atomic regardless.

**A purpose-built store (DynamoDB with a conditional write, or similar).** At very large scale, teams use a managed KV store with a conditional `PutItem` (`attribute_not_exists(pk)`) as the atomic claim and a TTL attribute for expiry. Same shape as Redis, with managed durability and horizontal scale. The trade-off is operational simplicity versus the cost-per-request and the eventual-consistency wrinkles of the specific store.

The decision rule: **if the side effect is a write to a database, put the dedup table in that same database and make them atomic.** If the side effect is external (a payment processor, an email provider), you cannot get atomicity anyway, so optimize the dedup store for latency and durability — Redis or a managed KV — and protect the gap with an in-flight status plus a *post-hoc verification* (after a crash, ask the processor "did charge for key X go through?" before re-attempting). That verification step is the difference between "we might double-charge after a crash" and "we never double-charge," and it is the part naive implementations skip.

### Stress test: the dedup-store outage

Now the question a design reviewer will ask: what happens when the dedup store itself is *down*? This is the scenario that separates a design that has thought about failure from one that has not, because the dedup store sits on the hot path of every mutating request, and "it is always up" is not an assumption a senior makes. There are two wrong answers and one right one, and which is right depends entirely on the cost of the operation.

**Wrong answer one: fail open.** If the dedup store is unavailable, proceed *without* deduping — "better to maybe-double-charge than to be down." For a payments endpoint, this is catastrophic: an outage of your dedup store, which often coincides with the kind of load spike that triggers *more* retries, becomes a window where every retry double-charges. You have built a system whose duplicate-prevention disappears exactly when duplicates are most likely. Never fail open for money.

**Wrong answer two: fail closed, naively.** If the dedup store is down, reject all mutating requests. This is safe (no double-charges) but it means a dedup-store outage is a *total* outage of your write path — you have coupled your availability to the availability of the dedup store, turning a dependency into a single point of failure. For some systems that is an acceptable trade (correctness over availability for money); for others it is not.

**The right answer is to match the failure response to the operation's reversibility.** For irreversible operations (charge a card, send money), fail closed — reject the write rather than risk a duplicate, and treat the dedup store as a hard dependency you must keep highly available (replicate it, give it more nines than the service it protects). For reversible or low-harm operations (record an event, update a non-critical counter), you can fail open and reconcile later, because a duplicate is cheap to detect and undo after the fact. The architectural move is to *make the dedup store at least as available as the most-critical operation it guards*, which usually means replicating it (a Redis cluster with replicas and automatic failover, or a multi-AZ managed store) and load-testing the failover so that a single node loss does not take the write path down. And there is a subtle third defense for irreversible external operations regardless of the dedup store's state: the *external system's own idempotency*. If your card processor also accepts an idempotency key, then even a complete loss of your local dedup store cannot double-charge, because the processor dedups on the key you forward. That is why forwarding the key to every downstream that supports one is not redundancy — it is the layer that survives your own store failing.

### Stress test: the key collision

A second reviewer question: what if two *different* logical operations accidentally generate the *same* idempotency key? A collision means the second operation is treated as a retry of the first and replayed — it silently does not execute, and the caller gets back a response for a *different* operation. This is a correctness disaster precisely because it is silent. Two defenses make it a non-issue. First, **use a collision-resistant key**: a version-4 UUID (122 random bits) has a collision probability so low that across billions of keys you will never see one — do not let clients use sequential integers or short tokens as idempotency keys, because those collide. Second, **the request fingerprint catches the collision even if it happens**: if two different operations somehow shared a key, their request bodies differ, so the fingerprint mismatch fires and the server returns a 400 instead of silently replaying the wrong response. The fingerprint is the safety net under the key's randomness — belt and suspenders, and for money you want both.

## 7. Idempotent consumers: making at-least-once messaging effectively-once

Everything so far has been request/response. The same logic governs message consumers, and the stakes are arguably higher because at-least-once *redelivery* is not an edge case in messaging — it is the *normal* contract. A broker delivers a message, the consumer processes it, the consumer's ack is lost (crash, timeout, network blip), the broker redelivers. The consumer that is not idempotent now sends the email twice, charges the card twice, or double-counts the event. Figure 6 shows the broken consumer and the fix.

![Before and after comparison of an at-least-once consumer that sends an email twice on redelivery versus a dedup on consume consumer that records the message id first and sends once](/imgs/blogs/idempotency-and-exactly-once-by-design-6.webp)

The fix is **dedup on consume**: before performing the side effect, record that you have processed this message's stable id, and only act if it was not already recorded. "Stable id" is load-bearing. You need an identifier that is *the same across redeliveries of the same logical message*. The broker's delivery id will not do — many brokers assign a new delivery id on each redelivery. You want either the *producer-assigned message id* (Kafka offset within a partition, or a producer-set message key) or, better, a *business key* derived from the message content (the order id, the payment id) that means "this logical event." Deduping on a business key is more robust than deduping on a broker id because it survives the message being republished by a different producer or re-derived from a different source — it dedups on *what the message means*, not on *which copy* it is. The full mechanism for message-layer dedup, including how brokers assign ids and how producer sequencing works, is in the [idempotency and deduplication deep-dive](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe); here we focus on the consumer's decision.

The ordering matters as much as in the request case, and the same race applies. The wrong order is "do the work, then record the id" — crash in between and the redelivery re-does the work. The right order is "record the id atomically, *then* do the work" — but now you have the inverse risk: you recorded the id and then crashed before doing the work, so the redelivery sees the id, thinks it is done, and *skips* the work, losing it. The clean resolution is the same as section 5: make the id-recording and the side effect *atomic* when the side effect is a database write (one transaction: insert the processed-id row and write the business change together), and when the side effect is *external* (email, money), record the id with an in-flight status, perform the external action, then mark complete — and on recovery, verify with the external system whether the action actually happened before deciding to redo or skip.

#### Worked example: an effectively-once email consumer

Make an order-confirmation email consumer effectively-once, with numbers. The system processes 2,000 order events per minute at peak. The broker is at-least-once; measured redelivery rate is about 0.3% (6 redeliveries per minute), spiking to ~5% during a consumer deploy when in-flight messages get redelivered after the old pods drain. Sending a duplicate confirmation email is a customer-trust bug — "why did I get two receipts?" support tickets — so duplicates are not acceptable.

Design: a `processed_messages` table keyed on the order's *business key* `email:confirmation:{order_id}` (not the broker delivery id, so it survives redelivery and republish). The consumer, on each message, runs a transaction: `INSERT ... ON CONFLICT DO NOTHING` the business key, and if the insert returned a row (we are first), enqueue the email send and commit; if it conflicted (already processed), commit nothing and ack the message. The email send itself goes through the email provider's *own* idempotency key (most providers — SendGrid, Postmark, SES via a message-id — support one), set to the same `email:confirmation:{order_id}`, so even if our consumer logic somehow double-fires, the provider dedups. That is **defense in depth**: our dedup table catches the common redelivery; the provider's idempotency key catches the rare gap where we recorded-then-crashed and a manual replay re-fires. Storage cost: at 2,000/min with a 7-day retention (longer than any redelivery could plausibly lag), that is ~20M rows of (key, timestamp), roughly a few GB — trivially cheap, and the 7-day TTL sweep reclaims it. The redelivery spike during deploys, which would have sent ~100 duplicate emails per deploy, now sends zero.

```python
# Effectively-once email consumer. Dedup on a business key, atomic with enqueue.
def handle_order_event(msg, db, mailer):
    order_id = msg["order_id"]
    dedup_key = f"email:confirmation:{order_id}"   # business key, stable across redelivery

    with db.transaction() as tx:
        first_time = tx.execute(
            "INSERT INTO processed_messages (dedup_key, processed_at) "
            "VALUES (%s, now()) ON CONFLICT (dedup_key) DO NOTHING "
            "RETURNING dedup_key",
            (dedup_key,),
        ).fetchone()

        if first_time is None:
            return ack(msg)        # already processed: this is a redelivery, skip the email

        # We are the first to process this order. Enqueue the send inside the same tx
        # so the dedup row and the intent-to-send commit together (transactional outbox).
        tx.execute(
            "INSERT INTO email_outbox (idempotency_key, template, to_order, status) "
            "VALUES (%s, 'order_confirmation', %s, 'pending')",
            (dedup_key, order_id),
        )
    # Outbox relay sends with the provider's idempotency key == dedup_key (defense in depth).
    ack(msg)
```

## 8. Concurrent duplicates: the stress test

The single hardest scenario to reason about is two duplicates that are genuinely *concurrent* — not "retry after the first failed" but "two copies in flight at the same instant." This happens more than people expect: a double-click that fires two requests before either returns, a client retry triggered by a slow (not failed) first attempt, a load balancer that retries a request that was actually still processing. Figure 7 walks the timeline of two such requests through a correct implementation.

![Timeline of two concurrent requests with the same idempotency key arriving together, both checking the store and finding it absent, then one winning an atomic insert while the other returns a 409 and later replays the stored result](/imgs/blogs/idempotency-and-exactly-once-by-design-7.webp)

The timeline tells the whole story. At T+0ms requests A and B arrive carrying the same key. At T+1ms both check the store and both see the key absent — this is the dangerous moment, and a naive implementation would let both proceed. At T+2ms both attempt the atomic claim, but the claim is atomic, so exactly one wins: A inserts the key row, B's insert hits the unique constraint and fails. B, having lost the race, does *not* execute; it returns a 409 "in progress" (or, in a more polished design, it blocks-and-polls for A's result rather than bouncing the client). At T+40ms A finishes the charge and records the result, flipping the key to `completed`. At T+80ms B's retry comes back, finds the key now `completed`, and replays A's stored 200 response with A's charge id. The customer was charged once. Both clients got a successful, identical response. Neither can tell it was a race. That is the property you are buying.

Two design choices make the 409 path nicer. First, **block-and-poll instead of immediate 409**: the loser can wait up to a short bounded time (say 5 seconds) polling for the winner's result, and only return 409 if the winner is still not done — this hides the race from the client entirely in the common case where the winner finishes quickly. Second, **a bounded in-flight timeout**: if the loser polls and the winner *never* completes (the winner crashed), the in-flight record must expire so the next attempt can take over — otherwise the operation wedges. The block-and-poll plus in-flight-TTL combination turns concurrent duplicates from a client-visible error into a transparent slight delay, which is what a polished payments API actually does.

## 9. Idempotency across services and the distributed-transaction tie-in

A single service deduping its own writes is the easy case. The architect's case is **idempotency across a chain of services**, where a retried request at the edge must not cause duplicate effects three services deep. The key insight: the idempotency key must *propagate* down the call chain, and each service must derive a *deterministic* downstream key from the one it received, so that a retry of the whole chain dedups at every hop — not just the first. Figure 8 shows the layers the key passes through, and figure 9 shows it propagating across services.

![Stack of layers showing the client generating the key, the gateway passing it through, the service claiming it with a fingerprint check, and the dedup store deciding the exactly once effect with a twenty four hour time to live](/imgs/blogs/idempotency-and-exactly-once-by-design-8.webp)

The layer stack in figure 8 makes one point sharply: *the key is generated once, at the top, by the client, and is decided once, at the bottom, in the dedup store.* The gateway and intermediate services *carry* the key but do not make the dedup decision — the dedup store is the single authority. This matters because if two layers each tried to dedup independently with different stores and different keys, they could disagree, and a retry could be deduped at one layer but executed at another. One key, one deciding store per logical effect.

![Pipeline showing an idempotency key propagating from the API edge through an order service and payment service to an outbox event, with each service deriving a deterministic key from the inbound one](/imgs/blogs/idempotency-and-exactly-once-by-design-9.webp)

Now the cross-service propagation in figure 9. The API receives `Idempotency-Key: ic_8f2a` and dedups at the edge. It calls the order service, passing the key; the order service derives its *own* dedup key deterministically — `ord:ic_8f2a` — so that a retry of the API call produces the *same* order-service key and dedups there too. The order service calls the payment service, which derives `pay:ic_8f2a`. When the payment service publishes a "payment captured" event to its outbox, it uses the same derived key as the message id, so a downstream consumer dedups on it. The whole chain is idempotent end to end because *every key is a deterministic function of the original*. A retry of the top-level request regenerates every downstream key identically, and every hop recognizes the retry.

A note on *why deterministic derivation* and not just "forward the same key everywhere." If every service used the *identical* key, then a service that legitimately needs to make two distinct downstream calls — say the payment service both captures a charge and records a ledger entry — would collide its own two operations on one key, and the second would be deduped as a retry of the first. Prefixing the key per operation (`pay:` for the capture, `ledger:` for the entry, both derived from `ic_8f2a`) keeps each *logical* operation distinct while keeping each *deterministic* under a chain retry. The rule: one derived key per distinct side effect, every derivation a pure function of the inbound key plus the operation name, no randomness anywhere in the derivation (randomness would make the retry generate a different key and defeat the whole scheme). This is the part teams get subtly wrong — they either share one key and collide distinct operations, or they generate fresh keys per hop and lose the chain-retry dedup. Deterministic-per-operation derivation threads the needle.

This is where idempotency meets distributed transactions. You cannot wrap a charge across three services in a single ACID transaction — there is no shared transaction manager across service boundaries, and you would not want the coupling if there were. So you use a **saga**: a sequence of local transactions, each idempotent, with compensating actions to undo on failure. Idempotency is what makes a saga *safe to retry* — if a saga step times out, the orchestrator retries it, and because the step is idempotent (keyed on the saga id), the retry does not double-apply. The full saga mechanics, including orchestration versus choreography and compensation design, are in the [saga pattern deep-dive](/blog/software-development/database/saga-pattern-distributed-transactions); the connection to grasp here is that *idempotency is a precondition for sagas* — a saga whose steps are not idempotent cannot safely retry, and a saga that cannot retry is not fault-tolerant. There is a second, subtler tie: the *compensating* action of a saga must itself be idempotent, because a compensation can also be retried (the rollback's ack can be lost just like the forward step's). A non-idempotent compensation — "refund the charge" that refunds twice on a retry — is as dangerous as a non-idempotent forward step, and it is the one people forget because they focus on the happy path. Make both directions idempotent. Similarly, the [transactional outbox](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) gives you atomic "change state and publish event," and the event carries the idempotency key forward so consumers dedup — outbox and idempotency together are how you get reliable, exactly-once-effect propagation across an event-driven architecture, the kind of design covered in [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects).

## 10. Exactly-once in Kafka, at the architect level

Kafka advertises "exactly-once semantics," and it delivers — but it is worth understanding *exactly what* it delivers, because the phrase is precise and the boundaries matter for an architect deciding whether it solves your problem. Kafka's exactly-once is built from two mechanisms. The **idempotent producer** assigns each producer a producer id and a per-partition sequence number; the broker tracks the last sequence it accepted per producer-partition and rejects duplicates, so a producer retry (which would otherwise append the same record twice) is deduplicated *at the broker*. This gives you exactly-once *produce* within a session. **Transactions** extend this across multiple partitions and across the consume-process-produce cycle: a consumer reads, a processor transforms, a producer writes, and the read offset commit plus the output writes are committed *atomically* — either all happen or none do, and consumers reading the output with `read_committed` isolation never see the records of an aborted transaction.

```python
# Kafka transactional consume-process-produce: the heart of exactly-once *within Kafka*.
producer = KafkaProducer(transactional_id="payments-processor-1",
                         enable_idempotence=True)          # producer id + sequence numbers
producer.init_transactions()

for batch in consumer.poll():
    producer.begin_transaction()
    for record in batch:
        result = process(record)                            # your transform
        producer.send("payments-out", value=result)
    # Commit the *input offsets* inside the same transaction as the *output writes*.
    producer.send_offsets_to_transaction(consumer.position(), consumer.group_id)
    producer.commit_transaction()                           # atomic: offsets + outputs together
```

The architect's caveat, and it is the whole point: **Kafka's exactly-once holds *within the Kafka boundary*.** Consume-from-Kafka, process, produce-to-Kafka is exactly-once. The moment your processor reaches *outside* Kafka — writes to Postgres, charges a card, sends an email — that external side effect is *not* covered by the Kafka transaction, because Kafka cannot enroll your database or Stripe or SendGrid in its transaction protocol. So for stream processing that stays inside Kafka (a Kafka Streams topology aggregating events), Kafka's exactly-once is genuinely exactly-once and you can lean on it fully. For a consumer that touches the outside world, you are back to dedup-on-consume with a business key, exactly as section 7 describes — Kafka's exactly-once gets the message to you reliably, and *you* make the external side effect idempotent. The full mechanism, including how transactional markers and the transaction coordinator work, is in the [exactly-once in Kafka deep-dive](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions). The decision: use Kafka transactions when your processing stays in Kafka; use dedup-on-consume when it does not. Most real pipelines are the latter at the edges, so do not let "Kafka does exactly-once" lull you into skipping your own dedup at the points where the stream meets the world.

There is a cost to Kafka transactions worth naming so the trade-off is explicit. Transactions add latency and throughput overhead: each transaction has a begin/commit round trip to the transaction coordinator, the broker writes transaction markers into the log, and `read_committed` consumers must wait for the commit marker before they can read past an open transaction, which adds end-to-end latency proportional to your transaction duration. In practice, batching many records per transaction amortizes the overhead — a transaction per record is wasteful, a transaction per poll-batch of a few hundred records is the sweet spot. The senior framing: Kafka transactions are not free exactly-once, they are *bought* exactly-once, and the price is latency and a coordinator dependency. If your pipeline stays in Kafka and the price is acceptable, they are the cleanest tool. If your pipeline exits Kafka, you pay the price *and* still need dedup-on-consume at the boundary, which is the worst of both — so for boundary-crossing pipelines, often plain at-least-once delivery plus your own dedup-on-consume is simpler and cheaper than transactions you cannot fully use anyway.

## Trade-offs: the decision matrix

Every approach buys safety and pays somewhere, and a senior names the bill before recommending. Here is the consolidated decision matrix; figure 3 visualizes the same comparison.

| Approach | Safety | Storage cost | Added latency | Cross-service | When it wins |
| --- | --- | --- | --- | --- | --- |
| **Natural idempotency** (PUT/SET/DELETE) | Built-in, strong | None | None | Inherent | Operation *replaces* state; always try this first |
| **Idempotency key** | Strong | One record per request, TTL-bounded | +1 dedup-store hop (~1ms) | Propagates via key derivation | Accumulating ops, external side effects, public APIs |
| **Dedup table** (unique constraint) | Strong, atomic with DB side effect | Grows with TTL; needs sweep | +1 upsert in the same txn | Shared store, atomic | Side effect *is* a DB write; want atomicity + audit trail |
| **Dedup on consume** (business key) | Effectively-once | One row per message, TTL-bounded | +1 lookup/insert | Per-consumer; not free across services | At-least-once messaging; redelivery is the norm |
| **Kafka transactions** | Exactly-once *in Kafka* | Internal (offsets/markers) | Transaction overhead per batch | Only within Kafka | Consume-process-produce that stays inside Kafka |

The senior reading of this table: there is no single best approach; there is the *right* approach for the operation's shape. Replace-shaped operations get natural idempotency and pay nothing. Accumulate-shaped operations with database side effects get a dedup table and pay one upsert for atomicity. Accumulate-shaped operations with external side effects get an idempotency key plus a status check and pay a store hop plus a verification. Streaming that stays in Kafka gets Kafka transactions; streaming that exits Kafka gets dedup-on-consume at the exit. The mistake is picking one approach for everything — either over-paying (idempotency-key infrastructure for operations that could be naturally idempotent) or under-protecting (assuming Kafka's exactly-once covers your database write when it does not).

## Optimization: making dedup cheap and fast

Idempotency adds a store hit to every mutating request, and on a high-QPS system that hit is a real cost in latency, throughput, and dollars. The optimization work is to keep the protection while shrinking the tax. Four levers.

**1. Fast-path the common no-retry case.** The overwhelming majority of requests are *first* attempts that will never be retried — measured at most APIs, the retry rate is well under 1%. So optimize for the first attempt: a single atomic claim (the `INSERT ... ON CONFLICT` or `SET NX`) *is* the first-attempt path, and it is one round trip. You do not need a separate "check then claim" — the atomic claim both checks and claims, so the happy path is exactly one store operation, not two. Implementations that do a `GET` then a `SET` are paying two round trips on every request to optimize for the rare retry; collapse it to one atomic op and the common case costs one hop.

**2. Size the TTL deliberately.** The TTL must exceed the maximum retry window (or you risk re-execution) but should not vastly exceed it (or you pay to store dead keys). Measure your clients' actual retry behavior: most give up within seconds to a few minutes. A 24-hour TTL is comfortably safe and keeps storage bounded. The storage math is simple and worth doing: at 5,000 mutating requests/second with a 24-hour TTL, you hold `5,000 × 86,400 = 432M` keys at once. At ~200 bytes per key record (key, fingerprint, status, small response), that is ~86 GB resident. If that is too much, either shorten the TTL to the measured retry window (a 1-hour TTL cuts it to ~3.6 GB) or store only the key and fingerprint hot and the full response in cheaper storage. The TTL is a direct storage-cost knob; tune it with the measured retry distribution, not a guess.

**3. Watch for the hot key on the dedup store.** A single very popular key — say a webhook that fires for one mega-merchant, all carrying related keys, or a retry storm hammering one specific key — can hot-spot a single shard of the dedup store. The fix is the same as any hot-key problem: ensure the dedup store is sharded by the idempotency key (high-cardinality, naturally spreads), and for a genuine retry storm on one key, the block-and-poll design from section 8 *reduces* load because losers wait for the winner rather than each re-attempting the work. The idempotency store should never become the bottleneck; if it does, you have either a sharding problem (low-cardinality keys — fix the key generation) or a retry-storm problem (add backoff and the poll-for-winner path).

**4. Keep the dedup store *near* the side effect.** Every cross-store hop is latency. If the side effect is a database write, the dedup table in the *same* database costs you nothing extra in network hops (it is one more statement in a transaction you were already running) and buys atomicity. Moving the dedup to a remote Redis to "offload the database" can be a false economy if it adds a network round trip *and* re-introduces the atomicity gap. Profile before you separate them; co-located dedup is often both faster and more correct.

#### Worked example: dedup cost on a 5k QPS payments API

Put numbers on it. A payments API at peak handles 5,000 charge requests/second. Without idempotency, each charge is one Postgres transaction at, say, p99 4ms. Adding a dedup table in the same Postgres: the atomic claim is one extra `INSERT ... ON CONFLICT` *inside the existing transaction*, measured at +0.4ms p99 — so charge p99 goes from 4.0ms to 4.4ms, a 10% latency tax. Storage at 24h TTL: `5,000 × 86,400 × ~250 bytes ≈ 108 GB` resident, swept hourly; at a few cents per GB-month on SSD that is single-digit dollars per month — negligible against the revenue the charges represent. Now the *win*: the retry rate is 0.8%, so 40 requests/second are retries that, without dedup, would each be a duplicate charge. At an average \$30 charge, that is 40 × \$30 = \$1,200/second of *potential* double-charges prevented — refunds you do not issue, support tickets you do not field, chargebacks you do not eat. The dedup table costs +0.4ms and a few GB; it prevents on the order of a thousand dollars a second of duplicate-charge exposure. That is the trade a senior makes without hesitation: a 10% latency tax on the happy path to eliminate a recurring money-loss failure mode. The measurement that justifies it is the retry rate × charge value, and you should know both numbers for your own system.

## Operating idempotency in production: what to measure

A design that is correct on paper still needs to be *observable* in production, because the failure modes of idempotency are quiet — a double charge does not crash anything, it just bills someone twice and shows up in a support ticket a day later. So you instrument the dedup layer to surface the things that should not be happening. Four metrics earn their place on the dashboard.

**The dedup hit rate** — the fraction of requests that hit an already-seen key. In steady state this equals your retry rate, typically well under 1%. A *spike* in the dedup hit rate is a signal: a client is retry-storming (a bug, or a deploy that is replaying a backlog), or an upstream is misbehaving. A dedup hit rate that suddenly jumps to 20% tells you something is hammering you with duplicates *before* it turns into a load problem. Alert on the derivative, not the absolute.

**The fingerprint-mismatch rate** — how often a key arrives with a body that does not match the stored fingerprint. In a healthy system this is approximately zero. Any non-trivial fingerprint-mismatch rate means a client is reusing keys incorrectly — a real client bug you want to find and report back to the integrator, because it means their retries are *not* idempotent on their end. This metric catches integration bugs that are otherwise invisible.

**The in-flight-timeout rate** — how often an in-flight key expires without ever completing. Every one of these is a request that *claimed* a key, started work, and then crashed or hung before recording a result. A rising in-flight-timeout rate points at crashes or hangs in the side-effect path, and each one is a place where, without the external system's own idempotency, you might have a half-completed operation. This is your early warning for the nastiest corner of the pattern.

**The dedup-store latency (p50/p99)** — because the dedup store is on the hot path of every mutating request, its latency is *added* to every write. If the dedup store's p99 climbs, your write p99 climbs with it. Track it separately from the business operation so you can attribute latency correctly in a regression, and so a slow dedup store does not masquerade as a slow business logic.

Beyond metrics, two operational habits pay off. **Log the idempotency key on every request** (it is a natural correlation id — the same key ties together the original and all its retries across your logs and traces), and **keep the dedup records long enough to investigate** an incident — if a customer reports a double charge, the dedup table is your evidence of exactly what the server saw and decided. A dedup table with the request fingerprint and the stored response is a forensic record of every mutating request, which is worth keeping even past the correctness-required TTL for high-value operations like payments.

## Case studies

**Stripe's idempotency keys (the canonical implementation).** Stripe exposes an `Idempotency-Key` header on its mutating API endpoints, and the behavior is the reference design this post describes: client-generated key, server stores key → response for 24 hours, retries replay the stored response, and a key reused with *different* parameters returns a 400 rather than silently replaying. Stripe's own engineering writing documents the in-flight handling — a request still processing returns a 409 so the client backs off — and the fingerprint check that distinguishes a genuine retry from a key-reuse bug. The lesson for an architect: the pattern is not exotic; it is a header plus a dedup store plus five rules, and the rules (client-generated, fingerprint, in-flight, replay, TTL) are each there to close a specific failure. Copy all five, not three of them.

**The double-charge incident (a recurring pattern, not one company).** The most common production payments incident is a double charge from a *non-failed* retry: the first charge succeeded, but the response was slow or the connection dropped *after* the charge committed, so the client retried, and without an idempotency key the retry charged again. The customer sees two identical charges. The post-mortem is always the same: the charge endpoint was a `POST` with no idempotency key, the retry was the HTTP client's default behavior on a timeout, and the timeout fired *after* the charge had actually gone through. The fix is always the same: idempotency key, client-generated, dedup on the server. The deeper lesson is that the dangerous retry is not the one after a *failure* — it is the one after a *timeout on a request that actually succeeded*. That is the case naive implementations forget to test, because in a test the request either clearly succeeds or clearly fails; the "succeeded but the client does not know" case only shows up under real network conditions.

**The email-sent-twice bug (the consumer side).** An order-confirmation consumer reads from an at-least-once queue, sends the email, then acks. A deploy drains the consumer pods; in-flight messages whose acks had not landed get redelivered to the new pods; the new pods send the confirmation emails *again*. Customers get two receipts; support gets tickets; trust erodes. The root cause is a consumer that performs the side effect *before* recording that it processed the message, with no dedup on a stable id. The fix is dedup-on-consume keyed on the business key (`email:confirmation:{order_id}`), atomic with the enqueue, plus the email provider's own idempotency key as defense in depth — exactly the design in section 7. The lesson: at-least-once redelivery is *guaranteed* to happen on every deploy, not just on failures, so a non-idempotent consumer is not "mostly fine" — it double-fires on a schedule you control (your own deploys), which makes it both more frequent and more embarrassing than people assume.

**Kafka exactly-once at the stream boundary (the scoping lesson).** Teams adopt Kafka's exactly-once expecting it to cover their whole pipeline, then discover a duplicate charge because the processor that consumes from Kafka *also* calls a payment API, and the Kafka transaction never covered that external call. The system was exactly-once *up to* the Kafka boundary and at-least-once *past* it. The lesson is precise and worth internalizing: Kafka's exactly-once is real but *scoped to Kafka*, and the architect's job is to know exactly where the stream stops being internal and apply dedup-on-consume at that boundary. The bug is not in Kafka; it is in assuming Kafka's guarantee extends to side effects it cannot see.

## When to reach for this (and when not to)

**Reach for explicit idempotency** whenever a mutating operation is not naturally idempotent *and* a duplicate would cause real harm: money movement, external notifications, inventory changes, anything a user would notice happening twice. For public APIs, expose an idempotency key on every mutating endpoint as a matter of contract — clients *will* retry, and giving them a safe way to do it is table stakes for a serious API. For message consumers on an at-least-once channel (which is nearly all of them), dedup on a stable business key before any external side effect, always, because redelivery is the norm and not the exception.

**Reach for natural idempotency first**, before any of the machinery. If you can remodel the operation as a PUT/SET (replace) or a unique-constrained insert, do it — it is free, it has no runtime cost, and it has no race to get wrong. A large fraction of operations that *look* like they need idempotency keys can be turned into naturally idempotent operations by changing the data model, and that is always the better answer when available.

**Do not over-engineer** the read path: `GET` is already idempotent, so do not add dedup machinery to reads. Do not add idempotency keys to operations that are already naturally idempotent — it is pure cost for no benefit. Do not build a distributed cross-service idempotency framework when a single service's dedup table solves the actual problem; propagate keys across services only when a *retry of the whole chain* is a real failure mode you have to defend against. And do not reach for Kafka transactions for a pipeline whose side effects are external — they will not cover the external part, and you will need dedup-on-consume anyway, so the transaction overhead buys you nothing at the boundary that matters.

**The one place to never skip it**: anything that moves money or sends an irreversible external message. There is no "we'll add idempotency later" for a charge endpoint. The day the network hiccups under load is the day you double-charge a thousand customers, and you do not get to schedule that day. Build it in from the first version of any operation with an irreversible external side effect.

## Key takeaways

- **Exactly-once delivery is impossible; exactly-once effect is achievable.** Stop trying to prevent duplicates. Assume them, make every mutating operation safe under duplication, and retries become a feature instead of a hazard.
- **Classify every mutating operation as replace or accumulate.** Replace-shaped operations (PUT, SET, DELETE) are naturally idempotent and free. Accumulate-shaped operations (POST, increment, send) need explicit protection. The classification *is* the decision.
- **The idempotency-key pattern is five rules, and you need all five.** Client-generated key, dedup store mapping key to result, a request fingerprint to catch key reuse, verbatim response replay, and a TTL longer than the retry window. Three out of five leaks.
- **The race between record-key and do-work is the bug that ruins naive implementations.** Claim the key *atomically before* the work — a unique-constraint insert or `SET NX` — so concurrent duplicates elect exactly one winner. Check-then-act is not idempotent under concurrency.
- **Make the dedup atomic with the side effect when you can.** If the side effect is a database write, put the dedup table in the same database and commit them together; if it is external, use an in-flight status plus post-crash verification with the external system.
- **Dedup consumers on a stable business key, not a broker delivery id.** At-least-once redelivery is the norm, especially on every deploy. Record the business key before the side effect and dedup on what the message *means*, not which copy it is.
- **Idempotency keys must propagate across services as deterministic derivations.** A retry of the whole chain should dedup at every hop, which only works if each service derives its downstream key from the inbound one. Idempotency is the precondition that makes sagas safe to retry.
- **Kafka's exactly-once is scoped to Kafka.** It covers consume-process-produce inside the cluster; the moment you touch an external side effect, you are back to dedup-on-consume. Know exactly where the stream meets the world.
- **The optimization is one atomic store op on the happy path.** Collapse check-then-claim into a single atomic claim, size the TTL to the measured retry window, shard by the high-cardinality key, and keep the dedup store near the side effect. The tax is a sub-millisecond hop; the saving is every duplicate charge you never make.

## Further reading

- [Delivery semantics: at-most-once, at-least-once, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the taxonomy and why at-least-once-plus-dedup is the only real recipe for exactly-once effect.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the message-layer mechanism this post's consumer section sits on top of.
- [Exactly-once in Kafka: the idempotent producer and transactions](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) — how producer ids, sequence numbers, and the transaction coordinator implement Kafka's scoped exactly-once.
- [The transactional outbox pattern for reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — atomic "change state and publish event," the twin of idempotent dedup for the producer side.
- [The saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions) — why idempotent steps are a precondition for safely retriable sagas.
- [API design: REST, gRPC, GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) — where to expose idempotency keys in an API contract and how HTTP method semantics encode idempotency.
- [Queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) — the event-driven architectures where idempotent consumers and outbox propagation come together.
- Stripe API reference, "Idempotent requests" — the canonical public documentation of the client-generated-key, fingerprint, 24-hour-TTL design.

---
title: "Methods and Idempotency: GET, POST, PUT, PATCH, DELETE in Practice"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A practitioner's deep dive into HTTP method semantics and the idempotency property that decides what a caller may safely retry — with worked Payments and Orders examples, JSON Patch versus Merge Patch, and the idempotency-key pattern that makes a POST safe."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "idempotency",
    "http-methods",
    "patch",
    "json-patch",
    "rfc-9110",
    "retries",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-1.png"
---

A customer is checking out. Their phone is on a flaky train connection. They tap **Pay**, your mobile SDK fires `POST /payments` for `\$49.99`, and then — silence. The request left the device. The TCP connection stalled. Seven seconds later the SDK's timeout fires, and the SDK does the only thing it knows how to do: it retries. It sends the exact same `POST /payments` again.

Here is the part that ruins a Friday afternoon. The *first* request was not lost. It arrived at your server, your payment processor charged the card, the row was written, and the `201 Created` response started its journey back — and *that* is what got dropped on the train. From the server's point of view, the charge succeeded. From the client's point of view, nothing happened, so it tried again. Now there are two charges. The customer paid `\$99.98` for a `\$49.99` order, you get a chargeback, a support ticket, and a one-star review that says "double charged, never again."

Nothing in that story is a bug in the usual sense. Nobody wrote a wrong `if`. The mobile SDK did exactly what a robust client *should* do on an unreliable network — it retried. The server did exactly what a `POST` handler does — it created a resource. The whole failure lives in a single property of one HTTP method: `POST` is **not idempotent**, so retrying it is not safe, and nobody on either side encoded that fact into the contract. This post is about that property and the four siblings around it. By the end you will be able to look at any endpoint and answer two questions instantly: *what is the caller allowed to assume about this method, and can they retry it after a timeout without doing damage?* Those two questions are the whole game.

![A comparison matrix of GET POST PUT PATCH and DELETE across the safe, idempotent, cacheable, and has-body properties](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-1.png)

This is post B7 in the series, and it sits right on the spine: an API is a contract, not a function call. When you choose a method you are not picking a syntax — you are making a promise to a caller you will never meet about what is safe to repeat. We are going deeper than the [HTTP overview post](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers), which introduced safe and idempotent at altitude; here we live inside those two words, derive *why* they matter on a real network, and walk each of the five methods through the Payments and Orders example until the rules are reflexes. If you want the contract-and-product framing that anchors the whole series, start at [what an API actually is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); if you want the one-page checklist this all rolls up into, that is the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

## Two words that run everything: safe and idempotent

Before we touch a single endpoint we need two definitions absolutely nailed, because every rule in this post is derived from them. They come straight from the HTTP specification (RFC 9110, the current consolidation of HTTP semantics), and people misuse them constantly — usually by conflating them, or by assuming "idempotent" means "returns the same response," which is *not* what it means.

A method is **safe** if it has no observable side effect on the server's state — it is read-only from the resource's point of view. A safe request is a question, not a command. `GET`, `HEAD`, and `OPTIONS` are the safe methods. "No observable side effect" is doing real work in that sentence: a `GET` is allowed to write an access-log line, bump a hit counter, warm a cache, or record analytics, because none of those change the *resource* the caller asked about. What it may not do is mutate the thing the caller is reading in a way the caller would care about. If `GET /orders/ord_9` cancels the order, the method is no longer safe, and we will see in a moment why that is not a stylistic complaint but a correctness disaster.

A method is **idempotent** if making N identical requests has the same effect on server state as making it once. Read that precisely: it is about the *effect on the server*, not about the *response the client gets back*. `GET`, `HEAD`, `OPTIONS`, `PUT`, and `DELETE` are idempotent. `POST` and `PATCH` are not (PATCH "depends," and we will dig into exactly why). The classic point of confusion: a `DELETE` is idempotent even though the first call returns `204 No Content` and the second might return `404 Not Found`. Different *responses*, identical *effect* — after one DELETE the resource is gone; after a hundred DELETEs the resource is gone. The end state of the server is the same. That is idempotence. The status code is the client's view of the round-trip; idempotence is a statement about the server's state.

Notice these two properties are **independent axes**, not a single ranking. Every safe method is automatically idempotent — if you change nothing, then changing nothing N times is the same as changing nothing once — but the reverse is false. `PUT` and `DELETE` are idempotent and *not* safe; they change state, but they change it to a *fixed* end state regardless of repetition. So the four-way grid is real: safe-and-idempotent (`GET`), not-safe-but-idempotent (`PUT`, `DELETE`), and not-safe-not-idempotent (`POST`, `PATCH`). There is no "safe but not idempotent" cell — it cannot exist, by the definitions.

That gives us our first comparison table, the one you should be able to reproduce from memory after this post. Cacheability is the third column because it is downstream of safety, and "has a body" is the fourth because it trips people up.

| Method | Safe | Idempotent | Cacheable | Request body |
| --- | --- | --- | --- | --- |
| `GET` | Yes | Yes | Yes (the default) | No (ignored by spec) |
| `HEAD` | Yes | Yes | Yes | No |
| `OPTIONS` | Yes | Yes | No | No |
| `POST` | No | **No** | Only with explicit freshness | Yes |
| `PUT` | No | Yes | No | Yes |
| `PATCH` | No | **No** (in general) | No | Yes |
| `DELETE` | No | Yes | No | Usually not |

The two boldface "No"s in the idempotent column are where the money leaks. The figure above is exactly this table rendered as a decision matrix; keep it in your head as the rest of the post fills in the *why* behind each cell.

A word on the two quieter safe methods, because people forget they exist. `HEAD` is `GET` without the response body — same headers, same status, no payload. It is how a client asks "does this exist, what is its size, what is its `ETag`, has it changed?" without paying to download it; a smart client uses `HEAD` to check freshness before a heavy `GET`. `OPTIONS` asks "what can I do with this resource?" — it is the method behind CORS preflight in browsers, where the browser fires an automatic `OPTIONS` to learn whether a cross-origin `POST` is allowed before sending it. Both are safe and idempotent for the same reason `GET` is: they read, they do not write. If you implement `GET`, you almost get `HEAD` for free, and a well-behaved server should support it on every readable resource.

There is one more subtlety in the definition of safe worth stating plainly, because it is where the rule gets misapplied. "No observable side effect" does *not* mean "no side effect at all" — it means none the *caller* would object to. A `GET` that increments a view counter, lazily populates a cache, or logs the request is still safe, because the caller's *resource* is unchanged and the caller did not ask for and would not notice those internal effects. The line is crossed only when the side effect changes what the caller reads or would care about. A `GET /articles/42` that *publishes* the draft, or `GET /orders/ord_9` that *cancels* the order, is unsafe — the caller's resource changed as a direct result of a read. Keep that distinction sharp and the "GET that deletes" anti-pattern becomes obviously wrong rather than a judgment call.

#### The principle: why idempotence is a contract, not a convention

Here is the rule stated rigorously, because the rest of the post depends on it being airtight. Define the server state as $S$ and a request $R$ as a function that transforms state: applying it gives $S' = R(S)$. The method is **idempotent** if applying it again is a fixed point:

$$R(R(S)) = R(S)$$

For `PUT /orders/ord_9` with a full order body, this holds by construction: the first PUT sets the order to exactly the body you sent, and the second PUT sets it to *exactly the same* body, so the state after the second is identical to the state after the first. For `DELETE /orders/ord_9`, the first delete removes the row, and "remove an already-removed row" is a no-op, so $R(R(S)) = R(S)$ again. For `POST /payments`, the function *appends* a new charge each time, so $R(R(S)) \neq R(S)$ — each application moves the state. That single inequality is the double charge. The math is trivial; the consequence is a chargeback.

This is the difference between a convention and a contract. A convention is "we usually return JSON." A contract is "if you retry an idempotent request, the server's state is guaranteed identical to one successful call." A caller can build retry logic *on top of* that guarantee without asking you. That is the entire point of HTTP method semantics: they let a generic client — a browser, a CDN, a load balancer, a mobile SDK, a service mesh — reason about your endpoints without reading your docs. Break the semantics and you break every layer that trusted them.

## Why this matters: at-least-once networks and the lost response

To feel the weight of idempotence you have to internalize one fact about networks: **you cannot build at-most-once delivery on top of an unreliable channel without help.** This is not pessimism, it is a theorem about distributed systems, and it is the same reason message brokers offer at-least-once but rarely true exactly-once. (If you want that argument in full, the message-queue series has the canonical treatment of [at-most-once, at-least-once, and exactly-once delivery](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once); everything there about retries applies one-for-one to HTTP.)

The core problem is the **lost response**. When a client sends a request and the call times out, the client is in a state of fundamental ignorance. It does not know which of two worlds it is in:

1. The request never reached the server (the write did not happen).
2. The request reached the server, the write committed, and the *response* was lost on the way back (the write *did* happen).

From the client's side these two worlds are indistinguishable — in both cases it saw a timeout and no response body. And the client has to do *something*. If it gives up, then in world 1 it has silently dropped a legitimate operation. If it retries, then in world 2 it has duplicated the operation. There is no safe move that works in both worlds — *unless* the operation is idempotent, in which case retrying is correct in both worlds: world 1 executes it once, world 2 executes it again to the same fixed point. The ambiguity collapses. The retry is unconditionally safe.

So the derivation is clean and worth stating as a rule: **on an at-least-once network, an idempotent request is safe to retry blindly, and a non-idempotent request is not.** This is *the* reason the property exists. It is not a tidiness rule from a style guide. It is the load-bearing fact that lets clients, proxies, gateways, and SDKs implement automatic retries at all. Every HTTP retry library on earth — the ones built into AWS SDKs, gRPC, browser `fetch` with retry middleware, your service mesh's outlier-detection retries — encodes exactly this: retry the idempotent methods automatically, leave `POST` alone (or require a key).

It is worth pausing on *why* the network is at-least-once and not, say, exactly-once, because that is the assumption everything rests on. The fundamental obstacle is that acknowledgement is itself a message that can be lost. The sender transmits; the receiver acts and acknowledges; but the acknowledgement travels back over the same lossy channel and can vanish. The sender, seeing no ack, cannot distinguish "my request was lost" from "your ack was lost," so to make progress it must resend — and resending is what produces "at least once." You can push the duplicate-detection into the receiver (that is exactly what an idempotency key does), but you cannot make the channel itself deliver exactly once, because that would require the two endpoints to agree on whether a message arrived using only a medium that can drop the agreement. This is the same impossibility that the [delivery-semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) derives for message brokers; HTTP is just another instance. The practical upshot for an API designer is permanent: assume every mutating request can arrive more than once, and design so that arriving more than once is harmless.

There is also a quieter, second-order consequence of getting this wrong: the **retry storm**. Suppose your `POST /payments` is slow under load and starts timing out at, say, the client's 5-second deadline even though the server eventually completes in 7 seconds. Every client times out, every client retries, and now the server is doing *double* the work — the original requests that are still running plus all the retries — which makes it slower still, which causes more timeouts, which causes more retries. The system spirals into a feedback loop and falls over, and the post-mortem shows a load graph that doubled for no apparent reason. Idempotency keys defuse this on the correctness side (no double charges) but the load amplification is real regardless, which is why mature clients pair retries with exponential backoff and jitter rather than retrying immediately. The method's idempotence tells you *whether* it is safe to retry; backoff tells you *how* to retry without making the outage worse. They are complementary, and a `POST` on a money path needs both: a key for correctness, backoff for stability.

![A timeline contrasting a timed-out POST that is blindly retried and double-charges against an idempotent PUT retry that lands on the same state](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-2.png)

The timeline above is the train scenario laid out tick by tick: the `POST` commits, the response is lost, the blind retry charges a second time, and the parallel universe where the same operation is expressed as an idempotent `PUT` simply re-asserts the existing state. Same network failure, opposite outcomes, decided entirely by which method carries the operation.

#### Worked example: a timed-out POST retried unsafely versus with a key

Let me make the cost concrete with a full wire trace. First the unsafe version. The client sends a charge:

```http
POST /payments HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json

{
  "order_id": "ord_9f2a",
  "amount": 4999,
  "currency": "usd",
  "source": "tok_visa_demo"
}
```

The server charges the card, persists `pay_a1` with `amount: 4999`, and starts sending back:

```http
HTTP/1.1 201 Created
Location: /payments/pay_a1
Content-Type: application/json

{ "id": "pay_a1", "order_id": "ord_9f2a", "amount": 4999, "status": "succeeded" }
```

That `201` never arrives — the connection dropped. The client's HTTP library, configured (reasonably!) to retry on timeout, fires the *identical* request again. The server has no way to know it is a retry; the bytes are the same as a brand-new charge. It charges the card again and persists `pay_b2` with `amount: 4999`. The customer is now out `\$99.98`. The server logs look perfectly healthy: two successful `201`s, zero errors. The damage is invisible until the customer notices.

Now the safe version. The only change is one header — an **idempotency key**, a client-generated unique token (typically a UUID) that names *this specific operation* so the server can recognize a repeat:

```http
POST /payments HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Idempotency-Key: 7c1f4e0a-3b2d-4a6e-9f01-2c5d8e7a6b4f
Content-Type: application/json

{ "order_id": "ord_9f2a", "amount": 4999, "currency": "usd", "source": "tok_visa_demo" }
```

The first time the server sees this key, it executes the charge, stores `pay_a1` *and* records the key alongside the response it produced. The response is lost as before, the client retries with the *same* `Idempotency-Key`, and now the server's first action is to look the key up. It finds it, sees the charge already happened, and **replays the stored `201` with `pay_a1`** instead of charging again. One charge. `\$49.99`. The customer is fine.

That is the whole idempotency-key pattern in miniature, and it is the bridge that lets a non-idempotent `POST` behave like an idempotent operation *for retry purposes*. We will sketch the handler later in this post, and the series goes fully deep on it — including key expiry, the race where two retries arrive at once, and the "exactly-once is an illusion" caveat — in the dedicated post on [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions). For now, hold the shape: the network is at-least-once, so the operation must be idempotent *somehow* — either by method semantics (`PUT`/`DELETE`) or by a key (`POST`).

## GET: the method that must never mutate

`GET` is the workhorse. The overwhelming majority of API traffic is reads, and `GET` is the method for reads. It is safe and idempotent, and — crucially — it is the only method that is **cacheable by default**. A response to a `GET` can be stored by the client, by a CDN, by a forward or reverse proxy, by your gateway, and replayed for the next caller without ever touching your origin. That single property is why a well-designed read API can serve enormous traffic from cache and why a badly designed one melts the database.

The contract a caller gets to assume about `GET` is strong and they will assume it whether you honor it or not: *this request reads, it does not change anything, I can call it as many times as I like, I can cache the result, and I can retry it freely on failure.* Browsers prefetch `GET` links. CDNs cache `GET` responses. Crawlers walk every `GET` they can find. Antivirus scanners and link-preview bots fire `GET` on every URL in an email. All of them assume `GET` is safe. Which brings us to the most famous self-inflicted wound in API history.

#### The "GET that deletes" anti-pattern

Early in the web's life, plenty of admin tools shipped links like `GET /admin/articles/42/delete`. It was convenient — a link is a `GET`, and a link is easy to put in a table of articles next to each row. It worked in testing. Then the team installed a tool that **prefetched links to make navigation feel fast**, or a crawler indexed the admin panel, or an email scanner followed the links in a notification. The prefetcher did what prefetchers do: it issued a `GET` to every URL on the page — including every `.../delete` link. It deleted every article on the site. Nobody clicked anything. The classic 2006 incident — a company's content silently wiped by Google's web accelerator following "delete" links — is the canonical telling, but the pattern has repeated dozens of times since, because the lesson keeps having to be re-learned.

The failure is not "someone made a mistake." The failure is that a `GET` mutated state, which **violates the safety contract that the entire web is built to assume.** The prefetcher was correct; the API was lying. The fix is never "ask the prefetcher to stop" — you do not control the prefetchers of the world. The fix is to use a method whose contract matches the intent: deleting is a state change, so it is `DELETE` (or a `POST` to an action endpoint if you must), never `GET`.

```bash
# Wrong — a GET that mutates. Any prefetcher or crawler will trigger it.
curl https://api.shop.example/orders/ord_9/cancel

# Right — the intent is a state change, so the method is one that may change state.
curl -X POST https://api.shop.example/orders/ord_9/cancellation \
  -H "Authorization: Bearer <token>"
```

The rule that falls out: **if calling it twice from a browser, a crawler, or a prefetcher would be a problem, it must not be a `GET`.** Reads are `GET`; everything else picks from the methods below. And while you are at it, give your `GET` responses honest cache directives — `Cache-Control` and `ETag` — so callers and proxies can cache them safely. (HTTP caching mechanics are their own deep topic later in the series, and the honesty of the *status code* you return is covered in [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).) But the precondition for any of it is that `GET` genuinely does not mutate.

## POST: the non-idempotent create-and-act method

`POST` is the method you reach for when none of the others fit — and that is by design. The HTTP spec defines `POST` as "the target resource process the enclosed representation according to the resource's own semantics," which is a fancy way of saying *do whatever this endpoint does with this body.* It is the universal "perform an action" method. It is **not safe** (it changes state) and **not idempotent** (each call is a fresh action). That non-idempotence is not a flaw to be designed around — for many operations it is exactly correct. Creating a new order, appending a charge, sending an email, kicking off a job: these are operations where "do it twice" genuinely means "two of them happened," and the method's semantics honestly reflect that.

The two big jobs for `POST` are **create** and **non-idempotent actions**.

For **create** against a collection, the convention is precise and worth following to the letter. The client `POST`s a representation *without* an id to the collection URL; the server mints the id, creates the resource, and returns `201 Created` with a `Location` header pointing at the new resource's URL. Returning the created body too is a courtesy that saves the client a follow-up `GET`.

```http
POST /orders HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json

{
  "customer_id": "cus_3kd",
  "items": [
    { "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 },
    { "sku": "SKU-CAP-RED",   "qty": 1, "unit_price": 1499 }
  ],
  "currency": "usd"
}
```

```http
HTTP/1.1 201 Created
Location: /orders/ord_9f2a
Content-Type: application/json

{
  "id": "ord_9f2a",
  "customer_id": "cus_3kd",
  "status": "pending",
  "total": 5497,
  "currency": "usd",
  "created_at": "2026-06-20T10:14:03Z"
}
```

Three details that separate a good `POST` create from a sloppy one. First, the status is `201 Created`, not a bare `200 OK` — the code tells the truth that a new resource now exists. Second, the `Location` header is mandatory in spirit: it is how the client (and any HATEOAS-aware tooling) learns the URL of the thing it just made. Third — and this is the trap — **the response is not idempotent.** If the client never gets that `201` and retries, it creates `ord_9f2b`, a second identical order. The server cannot tell the retry from a genuine second order, because a genuine second order with the same items is a perfectly legal thing for a customer to do. This is the create-side version of the double charge.

For **non-idempotent actions** — refund this payment, send this email, retry this shipment — `POST` to an action sub-resource is the honest choice. You model the action as a resource being created: `POST /payments/pay_a1/refunds` creates a refund. The refund is itself a resource with an id, which conveniently makes it addressable and gives you somewhere to attach an idempotency key. The anti-pattern to avoid is jamming verbs into a `GET` or pretending an action is a `PUT` when it genuinely is not idempotent.

```http
POST /payments/pay_a1/refunds HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Idempotency-Key: 3f9c2a1e-7b4d-4e8a-bc02-1d6f5a9e3c7b
Content-Type: application/json

{ "amount": 4999, "reason": "customer_request" }
```

```http
HTTP/1.1 201 Created
Location: /payments/pay_a1/refunds/ref_x8
Content-Type: application/json

{ "id": "ref_x8", "payment_id": "pay_a1", "amount": 4999, "status": "pending" }
```

Modeling the refund as a created sub-resource pays off three ways: it is addressable (you can `GET /payments/pay_a1/refunds/ref_x8` to check its status later), it is countable (`GET /payments/pay_a1/refunds` lists every refund attempt against that payment), and it is the natural home for the idempotency key. Compare that to a verb-style `POST /payments/pay_a1/refund` that returns a bare `200` with no resource — you have nowhere to hang the refund's own state, no URL to poll, and you have quietly turned a resource-oriented API into an RPC call. Resource-shaped actions age better.

A note on **asynchronous actions**, because not every `POST` finishes inside the request. If the action kicks off real background work — generating an export, settling a batch, fanning out webhooks — the honest response is `202 Accepted`, not `201`, with a status resource the client can poll: `202` plus `Location: /jobs/job_44` and a body describing the job. `202` means "I have accepted this request for processing; it is not done yet." The idempotency key still applies — a retried `POST` should return the *same* job, not start a second one — and the full long-running-operation pattern (poll versus webhook, the status resource lifecycle) is its own deep topic later in the series. The point here is that the status code must tell the truth about *when* the work happened, and `POST` is comfortable returning `201` (done, created), `200` (done, here is the result), or `202` (accepted, check back) depending on the reality.

The honest summary of `POST`: it is the method that admits "this operation is a fresh action each time, and the network might make me run it more than once." When that admission is true, `POST` is right — and you protect it with an idempotency key. When the operation actually *has* a fixed end state, you should be using `PUT` or `DELETE` instead, and getting idempotence for free.

## PUT versus PATCH: replace versus modify

This is the pair people get wrong most often, and the confusion costs real correctness, so we are going to spend the most time here. The one-line version: **`PUT` replaces the entire resource with the representation you send; `PATCH` applies a partial modification.** From that single difference — full versus partial — falls everything else, including why `PUT` is idempotent and `PATCH` generally is not.

### PUT: full replacement, and why that makes it idempotent

`PUT /orders/ord_9` says: *make the resource at this URL be exactly this body.* It is a complete-representation write. Whatever the order was before, after a successful `PUT` it is precisely the document you sent — every field you omit is treated as *absent*, not *unchanged*. (This is the part beginners trip on: a `PUT` with only `{"status": "paid"}` does not mean "set status to paid and leave everything else." Strictly, it means "the entire order is now just a status field," which is almost never what you want. `PUT` is all-or-nothing.)

That all-or-nothing semantics is *exactly* what makes `PUT` idempotent. Replaying the same full representation lands the resource on the same state every time — the function is a fixed point because the target state is fully specified, not relative to the current state. Recall $R(R(S)) = R(S)$: a `PUT` sets $S$ to a constant document $D$ regardless of what $S$ was, so $R(R(S)) = D = R(S)$. There is no accumulation, no drift. That is the formal reason a generic client may retry a `PUT` blindly.

#### Worked example: a PUT full-replace

Suppose the order currently looks like this:

```json
{
  "id": "ord_9f2a",
  "customer_id": "cus_3kd",
  "status": "pending",
  "shipping_address": { "line1": "1 Old St", "city": "Hanoi", "country": "VN" },
  "items": [{ "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 }],
  "currency": "usd"
}
```

The client wants to change the shipping address *and* mark the order as confirmed. With `PUT` it must send the **complete** new representation — every field it wants the order to have, because PUT replaces:

```http
PUT /orders/ord_9f2a HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
If-Match: "v7"
Content-Type: application/json

{
  "customer_id": "cus_3kd",
  "status": "confirmed",
  "shipping_address": { "line1": "42 New Ave", "city": "Hanoi", "country": "VN" },
  "items": [{ "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 }],
  "currency": "usd"
}
```

```http
HTTP/1.1 200 OK
ETag: "v8"
Content-Type: application/json

{ "id": "ord_9f2a", "customer_id": "cus_3kd", "status": "confirmed",
  "shipping_address": { "line1": "42 New Ave", "city": "Hanoi", "country": "VN" },
  "items": [{ "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 }], "currency": "usd" }
```

Now retry that exact request — same body, same URL. The server sets the order to the same document again; the result is byte-identical. That is the idempotence guarantee in action, and it is why a client retry library will happily fire a `PUT` on timeout without a second thought. (Notice the `If-Match: "v7"` header and the returned `ETag: "v8"`: that is an *optimistic-concurrency* control to stop two writers clobbering each other — a `PUT` is idempotent, but two *different* PUTs racing is a separate problem, solved by conditional requests. It is orthogonal to idempotence and we will not belabor it here.)

The cost of `PUT` is that the client must hold and send the *whole* representation, including fields it does not care about. If the order has thirty fields and you want to change one, you send all thirty — and if you forget one, `PUT` deletes it. Sending a partial body to a strict `PUT` is a data-loss bug waiting to happen. That cost is precisely what `PATCH` exists to remove.

### PATCH: partial modification, and the two formats

`PATCH /orders/ord_9` says: *apply this set of changes to the resource.* You send only what you want to change, not the whole thing. That is a huge ergonomic win — no read-modify-write of the full document, no risk of stomping a field you did not load. But it comes with a subtlety that bites: **a `PATCH` body is not a document, it is a description of changes,** and there are two standardized, mutually incompatible ways to describe those changes. Picking the wrong one — or, worse, inventing a third — is where PATCH goes off the rails.

The RFC for the `PATCH` method itself (RFC 5789) deliberately does *not* define the body format. It only says: the body is a "patch document" whose media type tells the server how to interpret it. Two media types dominate.

**JSON Merge Patch (RFC 7396)**, media type `application/merge-patch+json`. The patch body is a partial JSON object that looks like the target. The server *merges* it into the existing resource: present keys overwrite, and — here is the special rule — a key set to `null` means *delete this field*. It is the intuitive, ergonomic format, and it is what most people mean when they say "PATCH."

```http
PATCH /orders/ord_9f2a HTTP/1.1
Host: api.shop.example
Content-Type: application/merge-patch+json

{ "status": "paid" }
```

That body says "set `status` to `paid`, leave everything else alone." Clean. To remove the `coupon_code` field you would send `{ "coupon_code": null }`. Two limitations follow directly from the merge semantics. First, you **cannot store a literal JSON `null` as a value** — because `null` is overloaded to mean "delete." Second, **arrays are replaced wholesale, not edited** — `{ "items": [...] }` replaces the entire items array; there is no way in Merge Patch to say "append one item" or "remove the second element." For a partial *object* update where you never need to set null and never need surgical array edits, Merge Patch is perfect, and it is what I reach for by default.

**JSON Patch (RFC 6902)**, media type `application/json-patch+json`. The patch body is an **ordered array of operations** — `add`, `remove`, `replace`, `move`, `copy`, `test` — each targeting a location with a JSON Pointer path. It is more verbose and less intuitive, but strictly more expressive: it can delete a field unambiguously, edit a single array element, move a value, and even *test* a precondition before applying.

```http
PATCH /orders/ord_9f2a HTTP/1.1
Host: api.shop.example
Content-Type: application/json-patch+json

[
  { "op": "test",    "path": "/status", "value": "confirmed" },
  { "op": "replace", "path": "/status", "value": "paid" },
  { "op": "remove",  "path": "/coupon_code" },
  { "op": "add",     "path": "/items/1", "value": { "sku": "SKU-CAP-RED", "qty": 1, "unit_price": 1499 } }
]
```

That body says, in order: *assert the status is currently `confirmed` (and fail the whole patch if not), then set it to `paid`, remove the coupon field, and insert a new item at index 1.* The `test` op gives you a built-in optimistic check. The `add` op into an array index is the surgical array edit Merge Patch cannot express. The price is verbosity and the cognitive load of JSON Pointer paths.

![A before-and-after figure contrasting JSON Merge Patch sending a sub-document against JSON Patch sending an ordered list of operations](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-3.png)

The figure above lines the two formats up side by side: Merge Patch states *what the resource should look like* (and overloads `null` for delete), while JSON Patch states *what operations to perform* (and can therefore delete, test, and edit arrays without ambiguity). Choose Merge Patch for everyday partial object updates; reach for JSON Patch when you need explicit deletes, array surgery, or a built-in precondition test.

Here is the third entry for our comparison-table requirement — the head-to-head on PUT, PATCH (both formats), and POST for the two jobs of writing data:

| Operation | `PUT` | `PATCH` (Merge, RFC 7396) | `PATCH` (JSON Patch, RFC 6902) | `POST` |
| --- | --- | --- | --- | --- |
| Semantics | Full replace | Merge a sub-document | Apply ordered ops | Create / act |
| Body size | Whole resource | Only changed fields | Only the ops | New representation |
| Idempotent | Yes, always | Usually (often) | Depends on the ops | No |
| Delete a field | Omit it | Set it to `null` | `remove` op | n/a |
| Edit one array element | Resend whole array | Resend whole array | `add` / `replace` at index | n/a |
| Precondition baked in | No (use `If-Match`) | No (use `If-Match`) | Yes (`test` op) | No |
| Good for | Idempotent full updates | Simple partial updates | Surgical / array edits | Creates and actions |

### Why naive PATCH can be non-idempotent

We said `PATCH` is *generally* non-idempotent and the table says "depends." This is the subtle, interview-favorite point, so let us make it crisp. Whether a `PATCH` is idempotent depends entirely on whether the *operations* it describes are relative or absolute.

A `replace` is absolute — it sets a field to a fixed value, so replaying it lands on the same state: idempotent. A Merge Patch of `{ "status": "paid" }` is absolute too — re-merging the same value is a fixed point. Those are idempotent in practice, which is why many real PATCH endpoints happen to be safe to retry. But a *relative* operation breaks it. Consider a JSON Patch that appends:

```json
[ { "op": "add", "path": "/items/-", "value": { "sku": "SKU-CAP-RED", "qty": 1 } } ]
```

The `/-` token means "append to the end of the array." Apply it once, you have one extra item. Apply it twice (a blind retry after a lost response), you have *two* extra items. $R(R(S)) \neq R(S)$. The patch is **not idempotent**, and a client that blindly retries it on a timeout will duplicate the item — the same double-charge pathology, now hiding inside a PATCH. The lesson: **the `PATCH` *method* makes no idempotence promise; the idempotence of any given PATCH is a property of its body.** So you cannot let a generic retry library treat `PATCH` like `PUT`. If you want a partial update that is safe to retry, either keep the operations absolute (replace, not append) or protect it with an idempotency key just like a `POST`.

The practical rule I follow: design PATCH endpoints to use absolute operations only, document that they are idempotent, and reject or normalize relative ops — or, if relative ops are genuinely needed, treat the endpoint as non-idempotent and demand an `Idempotency-Key`. Do not leave it ambiguous, because a caller will assume the spec's general guidance and get burned.

There is a sneakier non-idempotent PATCH worth naming because it hides even from careful reviewers: the **counter increment**. Take a Merge Patch that some teams reach for to "bump" a value, expressed through a custom convention like `{ "loyalty_points": { "increment": 100 } }`. That is a relative operation wearing a Merge-Patch costume — applying it twice adds 200 points, not 100. The fact that it *looks* like an absolute field set lulls everyone into thinking it is idempotent. The fix is the same: either make it absolute (`{ "loyalty_points": 1500 }`, the new total, which a retry sets to the same value) or accept that it is non-idempotent and require a key. The meta-lesson is that idempotence is a property of *semantics*, not *syntax* — you cannot tell whether a PATCH is idempotent by looking at whether it is "small" or "uses the Merge media type." You have to ask: does applying it twice land on the same state as applying it once? If the answer involves the word "add," "append," "increment," or "toggle," the answer is no.

A related foot-gun is the **toggle**. A `PATCH` that flips a boolean — `{ "op": "replace", "path": "/active", "value": "!active" }`, or any "invert the current value" semantics — is the worst case: not only non-idempotent, but *anti*-idempotent, because applying it twice returns to the *original* state, so a retry can leave the resource in exactly the wrong condition depending on whether the first call landed. Never model a toggle as a relative flip on the wire. Model it as an absolute set: the client sends the *desired* final value (`{ "active": true }`), and a retry harmlessly re-asserts it. This single habit — *always send the desired end state, never the delta* — is the cleanest way to keep PATCH idempotent without thinking hard about it.

## DELETE: idempotent, with a status-code debate

`DELETE /orders/ord_9` removes the resource. It is **not safe** (it changes state) but it **is idempotent** (the end state — "the resource is gone" — is the same after one call or a hundred). That idempotence is real and useful: a client whose `DELETE` times out can retry it freely, because the worst case is "delete an already-deleted thing," which is a no-op.

The interesting question is what status code the *second* DELETE returns, and there is a genuine, long-running debate here that you should resolve deliberately rather than by accident.

The **first** DELETE has an easy answer. If the delete is synchronous and there is nothing meaningful to return, `204 No Content` is the canonical choice — "I did it, there is no body." If you want to return a representation of what was deleted (a tombstone, a confirmation), use `200 OK` with a body. If the delete is asynchronous — it kicks off a background job — `202 Accepted` is honest, optionally with a status URL the client can poll. So the first-call choice is `204` versus `200` versus `202`, decided by whether you have a body and whether the work is sync or async.

The **second** DELETE — the retry, or a delete of something already gone — is where the debate lives. Two defensible positions:

- **Strict-semantics camp: return `404 Not Found`.** The resource does not exist, so a request to operate on it gets a `404`. This is technically pure: `404` describes the *current* state of the URL accurately. The catch: it makes the *response* to a retry differ from the first call, which can confuse a naive client that treats `404` as "my delete failed" and surfaces an error to the user, even though the resource is, in fact, gone — which is what they wanted.
- **Retry-friendly camp: return `204 No Content` (or `200`) for any delete of an absent resource.** The argument: the caller's *goal* — "ensure this resource does not exist" — is satisfied, so report success. This makes DELETE pleasant to retry: every call, first or fiftieth, returns the same `204`. The catch: you have slightly blurred the truth, since a `204` now means both "I just deleted it" and "it was already gone."

Neither is wrong. The HTTP spec permits both — idempotence is about the *effect* (the resource is gone either way), not about the *response code*. My own preference for most APIs is the retry-friendly `204`-on-second-delete, because it makes clients' lives easier and the alternative leaks an implementation detail (whether the row was present *this* time) that the caller usually does not care about. But if your audit or compliance story needs a `DELETE` of a non-existent id to be visibly distinct, `404` is fully defensible. The one thing you must not do is pick by accident — document it, and be consistent across the whole API.

![A timeline showing a DELETE that succeeds with a 204 then is retried after a lost response with the choice between a 404 and a 204 on the second call](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-4.png)

The timeline above walks the retry: first DELETE returns `204`, the response is lost, the client retries, and you reach the fork — `404` (strict) or `204` (retry-friendly). Both leave the same end state; only the response code differs, which is the whole point of "idempotent effect, possibly different response."

### Soft delete and the contract it changes

A wrinkle that touches both correctness and the contract: many systems do not *physically* delete rows. They **soft delete** — set a `deleted_at` timestamp or a `status: "archived"` flag — so the data survives for audit, undo, or referential integrity (you cannot orphan the payments that reference an order you just nuked). Soft delete is a fine implementation choice, but it changes what `DELETE` *means* on the wire, and you owe the caller honesty about it.

If `DELETE /orders/ord_9` soft-deletes, then a subsequent `GET /orders/ord_9` will *still find the row* in the database — and now you have a decision. Most APIs make the soft-deleted resource invisible through the public contract: `GET` returns `404` (or `410 Gone` if you want to signal "this used to exist and is permanently retired"), the resource drops out of collection listings, and the soft-delete flag is an internal detail the caller never sees. That preserves the illusion that `DELETE` removed it, which keeps the contract simple. The alternative — exposing `deleted_at` and letting callers see and un-delete archived resources — is a legitimate richer model, but then it is not really a delete anymore; it is a state transition, and you might honestly model it as a `PATCH` that sets `status: "archived"` rather than a `DELETE`. Pick the model that matches your domain, but keep `DELETE` meaning "the caller can no longer see this resource through the API," or you will surprise people.

## PUT-for-create versus POST-for-create: who assigns the id

We have a loose end from the `POST` section: `POST /orders` is not idempotent, so a create can be duplicated on retry. There is a second way to create a resource that *is* idempotent — `PUT` to the resource's own URL — and the difference between them comes down to a single question: **who picks the id, the server or the client?**

`POST /orders` is **server-assigned id**. The client sends a representation with no id; the server mints `ord_9f2a` and returns it in `Location`. The server controls the id space, which is convenient and avoids client-side id collisions, but the create is non-idempotent: a retry produces a *new* id, hence a duplicate resource.

`PUT /orders/ord_9f2a` is **client-assigned id**. The client picks the id up front — typically a UUID it generates locally — and PUTs the representation directly to that URL. If the resource does not exist, the server creates it and returns `201 Created`; if it already exists, the server replaces it and returns `200 OK`. Crucially, **this create is idempotent**: a retry PUTs to the *same* URL with the *same* body, and because the id was chosen by the client and is stable across retries, the second PUT lands on the already-created resource and replaces it with identical content. No duplicate. The client-chosen id *is* the deduplication token — it plays the same role an idempotency key plays for `POST`, baked into the URL.

![A before-and-after figure contrasting POST create with a server-assigned id against PUT create with a client-chosen id that makes the create idempotent](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-5.png)

The figure above shows both paths: `POST` mints the id (and so a retry forks a second resource), while `PUT`-create has the client carry the id (so a retry re-asserts the same one). That is the whole trade-off — server control of the id space versus a retry-safe create.

#### Worked example: making create retry-safe with a client-chosen id

The client generates a UUID and PUTs the order to that URL:

```http
PUT /orders/0c8a7d6e-2f4b-4c1a-9e3d-5a6b7c8d9e0f HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json

{
  "customer_id": "cus_3kd",
  "items": [{ "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 }],
  "currency": "usd"
}
```

```http
HTTP/1.1 201 Created
Location: /orders/0c8a7d6e-2f4b-4c1a-9e3d-5a6b7c8d9e0f
ETag: "v1"

{ "id": "0c8a7d6e-2f4b-4c1a-9e3d-5a6b7c8d9e0f", "status": "pending", "total": 3998, "currency": "usd" }
```

The response is lost; the client retries the *identical* PUT. The resource now exists, so the server replaces it with the same body and returns `200 OK` instead of `201`. One order, not two. The status differs (`201` then `200`) — which is fine, because idempotence is about *effect*, not response code, exactly as with DELETE. If you want to *prevent* the create from silently overwriting an existing resource (so a UUID collision is an error rather than a clobber), add `If-None-Match: *` to the PUT: the server creates only if the resource is absent and returns `412 Precondition Failed` if it already exists.

When do you pick which? Reach for `PUT`-create when the client can naturally generate a good id (a UUID, a content hash, a natural key like an email or an external reference) and you want creates to be retry-safe for free. Reach for `POST`-create when the server must own the id space — sequential ids, ids that encode shard or region, ids that must be unguessable and server-secret, or when client-chosen ids would let a caller probe or collide with others' resources. And note the escape hatch: a `POST`-create *can* be made idempotent too, just not by the method — you add an `Idempotency-Key` header, which is the next section.

## Making POST safe: the idempotency-key pattern

We have arrived back at the train. `POST /payments` must stay a `POST` — charging a card is genuinely a fresh action, the server owns the payment id, and we cannot pretend it is a `PUT`. So we make the *retry* safe without making the *method* idempotent, using the **idempotency key**: a client-generated unique token, carried in the `Idempotency-Key` header, that names one logical operation so the server can recognize and collapse retries of it.

The contract is: *for a given idempotency key, the server executes the operation at most once, and every subsequent request with the same key returns the result of that one execution.* The client's job is to generate one key per logical operation (one checkout = one key, generated *before* the first attempt and reused across all retries of that attempt) and send it on every retry. The server's job is to store the key with the operation's outcome and short-circuit repeats.

![A graph of the server handling a POST with an idempotency key by looking the key up and either executing once, returning a conflict for an in-flight request, or replaying the stored response](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-6.png)

The graph above is the server-side decision: look the key up; if it is new, execute the charge and store the `201`; if it is already completed, replay the stored `201` without charging; if it is currently in flight (a retry arrived while the first is still running), return `409 Conflict` and tell the client to back off and retry shortly. All three paths converge on "exactly one charge."

Here is a stripped-down handler so the pattern is concrete rather than hand-wavy. The two non-obvious parts are (1) keying the dedup store and (2) handling the in-flight race:

```python
from flask import request, jsonify, abort

def create_payment():
    key = request.headers.get("Idempotency-Key")
    if not key:
        abort(400, "Idempotency-Key header is required for this endpoint")

    body = request.get_json()
    # Bind the key to the request fingerprint so a reused key with a
    # DIFFERENT body is rejected, not silently served the old result.
    fingerprint = hash_request(body)

    record = store.get(key)
    if record:
        if record.fingerprint != fingerprint:
            abort(422, "Idempotency-Key reused with a different request body")
        if record.status == "in_progress":
            # A retry arrived while the first attempt is still running.
            abort(409, "A request with this key is in progress; retry shortly")
        # Completed: replay the stored response verbatim. No second charge.
        return record.response_body, record.response_status

    # First time we have seen this key. Claim it atomically, then execute.
    store.put(key, status="in_progress", fingerprint=fingerprint)
    charge = payment_gateway.charge(body["amount"], body["currency"], body["source"])
    response = {"id": charge.id, "status": "succeeded", "amount": body["amount"]}
    store.put(key, status="done", fingerprint=fingerprint,
              response_body=response, response_status=201)
    return jsonify(response), 201
```

Three design points worth calling out. First, **bind the key to a request fingerprint.** If a client reuses a key but sends a different body, that is a client bug, and the safe behavior is to reject it (`422`) rather than silently return the old result for a different operation. Second, **handle the in-flight state**, because two retries can race: the second must not start a second charge while the first is mid-flight, so you claim the key atomically (a unique constraint or a conditional write) and return `409` to the loser. Third, **keys expire** — you store them for a window (24 hours is a common choice) and then forget them, because a retry days later is almost certainly a different logical operation. This is a sketch; the full treatment of races, storage, TTLs, and why "exactly once" is ultimately an illusion you only approximate lives in the [idempotency-keys post](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).

That gives us the complete picture of how to make every method retry-safe, which is the whole purpose of caring about idempotence in the first place:

![A decision graph of whether a timed-out request is safe to retry, branching on whether the method is idempotent and whether a POST carries an idempotency key](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-7.png)

The decision graph above ties it together: when a request times out, ask "is the method idempotent?" — if yes (`GET`, `PUT`, `DELETE`), retry blindly; if no (`POST`, `PATCH`), retry only if there is an idempotency key (or, for PATCH, only if the operations are absolute). That single question, asked at every retry point in your system, is the practical payoff of this entire post.

## Putting it together: choosing the method by intent

Step back from the individual methods and the choice becomes a short lookup keyed on *what the caller is trying to do*. You almost never have to agonize. The intent dictates the method, and the method dictates the safe/idempotent guarantees for free.

![A layered stack mapping each caller intent to the matching method, from reading with GET to safe non-idempotent retries with POST plus a key](/imgs/blogs/methods-and-idempotency-get-post-put-patch-delete-8.png)

The stack above is the cheat sheet: read → `GET`; create with a server id or trigger an action → `POST`; replace a whole resource → `PUT`; change a field or two → `PATCH`; remove a resource → `DELETE`; and when a `POST` must be retry-safe, `POST` + an `Idempotency-Key`. Each layer carries the guarantee the intent needs. The discipline that keeps an API honest is simply *refusing to let convenience override intent* — never a `GET` that mutates because a link was handy, never a `PUT` that does a partial update because the client did not want to send the whole body, never a `POST` to "fix" something that genuinely has a fixed end state and should be a `PUT`.

#### Worked example: a realistic order lifecycle across all five methods

Walk the Payments and Orders spine through one customer's session, and watch each method land where its intent fits:

```http
# 1. Read the cart's pricing — a pure read.
GET /carts/cart_77 HTTP/1.1

# 2. Create the order — server assigns the id, returns it in Location. Not idempotent.
POST /orders HTTP/1.1
{ "customer_id": "cus_3kd", "items": [ ... ], "currency": "usd" }
# -> 201 Created, Location: /orders/ord_9f2a

# 3. Charge the card — a non-idempotent action, protected by a key so a retry is safe.
POST /payments HTTP/1.1
Idempotency-Key: 7c1f4e0a-3b2d-4a6e-9f01-2c5d8e7a6b4f
{ "order_id": "ord_9f2a", "amount": 5497, "currency": "usd", "source": "tok_visa_demo" }
# -> 201 Created, Location: /payments/pay_a1

# 4. Mark the order paid — a single-field change, absolute, so idempotent.
PATCH /orders/ord_9f2a HTTP/1.1
Content-Type: application/merge-patch+json
{ "status": "paid" }
# -> 200 OK

# 5. Customer edits the shipping address fully before fulfilment — full replace, idempotent.
PUT /orders/ord_9f2a HTTP/1.1
If-Match: "v9"
{ "customer_id": "cus_3kd", "status": "paid", "shipping_address": { ... }, "items": [ ... ], "currency": "usd" }
# -> 200 OK, ETag: "v10"

# 6. Customer cancels before shipping — remove the order, idempotent.
DELETE /orders/ord_9f2a HTTP/1.1
# -> 204 No Content   (and a retry also returns 204 under the retry-friendly policy)
```

Every choice is forced by intent. Step 2 is a `POST` because the server owns the order id; step 3 is a `POST` *with a key* because charging is non-idempotent but must survive a retry; step 4 is a Merge `PATCH` because it changes one field absolutely; step 5 is a `PUT` because the client is replacing the whole resource; step 6 is a `DELETE` because it removes. No method is doing a job that belongs to another. That is what "methods in practice" looks like when the contract is honest.

## Stress-testing the design: the questions a reviewer should ask

A contract is only as good as the failures it survives. When I review a method-and-idempotency design — mine or someone else's — I do not check whether it works on the happy path; everything works on the happy path. I throw the four nasty scenarios at it and watch what breaks. Walk through them on the Payments and Orders API and you will see how each method's properties either save you or sink you.

**What happens when the client retries on a timeout?** This is the headline scenario and the reason for the whole post. For a `GET`, nothing — read it again, get the same answer (modulo concurrent writes by others), no harm. For a `PUT /orders/ord_9` with a full body, nothing — re-assert the same document, same end state. For a `DELETE /orders/ord_9`, nothing — the resource is already gone, the retry is a no-op (you return `204` or `404` per your policy). For a `POST /payments` *without* a key — disaster, the double charge. For a `POST /payments` *with* a key — fine, the server replays the stored `201`. So the reviewer's question reduces to a checklist: *every non-idempotent endpoint on a costly path must carry an idempotency key, and every idempotent endpoint must genuinely be idempotent (no hidden append, no relative PATCH op masquerading as a replace).* If you cannot answer "yes" for every mutating endpoint, you have a latent double-effect bug waiting for the next flaky network.

**What happens when two writers race?** Idempotence does not solve concurrency — it solves *repetition*, which is a different problem. Two *different* PUTs to the same order from two clients will clobber each other: the last write wins, and the first writer's change is silently lost. That is the *lost update* problem, and the fix is not idempotence but **optimistic concurrency** via conditional requests — `If-Match: "v7"` against the resource's current `ETag`, so a stale write gets a `412 Precondition Failed` instead of overwriting. Notice the two properties are orthogonal: a `PUT` is idempotent (the *same* PUT repeated is safe) *and* race-prone (two *different* PUTs conflict). A good reviewer keeps them separate — "is this retry-safe?" is idempotence; "is this concurrent-write-safe?" is conditional requests. Confusing the two leads people to think an idempotency key protects against lost updates, which it does not.

**What happens when a `PATCH` is replayed?** This is where the "depends" in the table earns its keep. If the PATCH body is `{ "status": "paid" }` (Merge Patch, absolute), a replay is a no-op — idempotent. If the body is a JSON Patch with `{ "op": "add", "path": "/items/-", ... }` (relative append), a replay duplicates the item — *not* idempotent. So the reviewer asks: *does any PATCH endpoint accept relative operations?* If yes, that endpoint is non-idempotent and either must reject relative ops or must require an idempotency key, exactly like a `POST`. The dangerous middle ground is a PATCH endpoint that *usually* gets absolute bodies and *occasionally* gets a relative one — it will pass every test until the day a client sends an append and a retry doubles it.

**What happens when the operation has side effects beyond your database?** The hardest case. Suppose `POST /payments` charges an external card processor and *then* writes a row. The idempotency key protects against *your* duplicate row, but what about the external charge? If you call the processor before checking the key, a retry that races the first request could charge twice at the processor even while your key store dedupes your own row. The discipline: claim the key *first* (atomically, marking it in-flight), then make the external call, then store the result — and rely on the processor's *own* idempotency mechanism (most payment processors accept an idempotency token too) so the dedup is end-to-end, not just at your edge. The general principle is that **idempotency must be enforced at every layer that has a side effect**, not only at the API boundary; a key that dedups your write but not the downstream charge is a half-measure that fails under a race. (This composition problem — making a chain of services idempotent end-to-end — is the same one the message-queue series tackles for brokers in its [delivery-semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) discussion.)

#### Worked example: a JSON Patch versus a Merge Patch on the same order

To cement the format difference, here is the *exact same intent* — "mark the order paid, drop the coupon, and append a gift-wrap line item" — expressed both ways, so you can see precisely what each format can and cannot do. Start from this order:

```json
{
  "id": "ord_9f2a",
  "status": "confirmed",
  "coupon_code": "WELCOME10",
  "items": [{ "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 }],
  "currency": "usd"
}
```

With **JSON Merge Patch** you can mark paid and drop the coupon cleanly, but the gift-wrap append is the problem — Merge Patch cannot append to an array, only replace the whole array, so you are forced to resend the entire `items` list including the element you did not change:

```http
PATCH /orders/ord_9f2a HTTP/1.1
Content-Type: application/merge-patch+json

{
  "status": "paid",
  "coupon_code": null,
  "items": [
    { "sku": "SKU-TEE-BLK-M", "qty": 2, "unit_price": 1999 },
    { "sku": "SKU-GIFTWRAP", "qty": 1, "unit_price": 499 }
  ]
}
```

That works, but notice the cost: you had to *read the order first* to know the existing items so you could resend them, and if another writer added an item between your read and your patch, you just deleted it (the lost-update trap again). Now the **JSON Patch** version expresses the *append* surgically, no read-modify-write of the array required:

```http
PATCH /orders/ord_9f2a HTTP/1.1
Content-Type: application/json-patch+json

[
  { "op": "replace", "path": "/status",      "value": "paid" },
  { "op": "remove",  "path": "/coupon_code" },
  { "op": "add",     "path": "/items/-",     "value": { "sku": "SKU-GIFTWRAP", "qty": 1, "unit_price": 499 } }
]
```

The JSON Patch is *more expressive* and avoids the array read-modify-write — but the `add` to `/items/-` is a **relative** operation, so this PATCH is **not idempotent**: replay it and you get two gift-wrap lines. The Merge Patch version, because it sets the whole array absolutely, *is* idempotent — replay it and nothing changes. So the formats trade off in two dimensions at once: Merge Patch is idempotent here but forces a full-array resend; JSON Patch is surgical but here is non-idempotent and must be protected with a key if retried. There is no universally right answer — there is the answer that matches what *this* endpoint needs, which is exactly the kind of decision the contract should make explicit.

## Verifying the contract: how you prove the methods behave

A method's properties are claims, and claims need tests — otherwise "this endpoint is idempotent" is a comment, not a guarantee, and comments rot. Here is how I actually verify method semantics, in roughly increasing strength.

First, **the cheap reflex test**: for every idempotent endpoint, the test harness sends the request *twice* and asserts the server state is identical after the second call as after the first. For a `PUT`, send it, capture the resource, send it again, diff — they must match. For a `DELETE`, delete, then delete again, and assert the second response matches your declared policy (`204` or `404`) and the resource is still gone. For a `POST` with a key, send it, send it again with the *same* key, and assert exactly one resource was created and the second response is the replayed first response. These tests are trivial to write and they catch the most expensive class of bug — the hidden non-idempotence — before it reaches a customer.

Second, **the negative tests**: send a `POST` with a key twice but with *different bodies* and assert a `422` (key reused with mismatched parameters), not a silent replay of the wrong result. Send a relative-op `PATCH` twice without a key and assert the endpoint either rejects it or requires a key — if a relative PATCH silently doubles the array, your test should be the thing that screams, not the customer.

Third, **contract tests** that pin the *observable* behavior the caller depends on, so a refactor cannot quietly change a `201` to a `200` or drop the `Location` header. A consumer-driven contract (the Pact style) captures "when I `POST /orders`, I get a `201` with a `Location` header and an `id`" as an executable expectation that runs in both the consumer's and provider's CI. That is the difference between *documenting* the contract and *enforcing* it. The full machinery — consumer-driven contracts, schema diffs, breaking-change linters — is its own post in this series, but the seed idea applies here: the method's safe/idempotent guarantee is part of the contract, so test it like one.

```bash
# A minimal idempotency contract check you can run in CI against a test env.
KEY=$(uuidgen)
# First call: expect 201 and capture the created id.
ID1=$(curl -s -X POST https://api.test.example/payments \
  -H "Idempotency-Key: $KEY" -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"order_id":"ord_test","amount":4999,"currency":"usd","source":"tok_visa_demo"}' \
  | jq -r '.id')
# Second call with the SAME key: must replay, not create a new payment.
ID2=$(curl -s -X POST https://api.test.example/payments \
  -H "Idempotency-Key: $KEY" -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"order_id":"ord_test","amount":4999,"currency":"usd","source":"tok_visa_demo"}' \
  | jq -r '.id')
[ "$ID1" = "$ID2" ] && echo "PASS: one payment, replayed" || echo "FAIL: double charge"
```

That eight-line script is the cheapest insurance you will ever buy against the train scenario. Run it on every deploy. If `ID1 != ID2`, your idempotency is broken and you found out from CI instead of from a chargeback.

## Case studies: how the leaders actually do this

These patterns are not academic. The most heavily used APIs in the world live and die by them, and their public documentation is a free masterclass.

**Stripe and the `Idempotency-Key` header.** Stripe is the reference implementation of idempotency keys, and they popularized the exact pattern in this post. Their API lets you pass an `Idempotency-Key` header on `POST` requests (and other mutating calls), and they guarantee that a request with a previously seen key returns the *original* response rather than performing the operation again — so a retried charge does not double-bill. Their published guidance is precisely what we derived: generate one key per logical operation (typically a V4 UUID), reuse it across retries of *that* operation, and they retain keys for a bounded window (on the order of 24 hours) before forgetting them. They also store the key against the request so a key reused with mismatched parameters is an error. If you want a battle-tested reference for the key pattern, read Stripe's idempotency docs — it is the same shape as the handler sketch above, hardened for production.

**AWS and idempotent operations.** Many AWS APIs are deliberately designed to be idempotent so that the SDKs' built-in retry behavior (with exponential backoff and jitter) is safe by default. Several services expose an explicit client-side token to make a create idempotent — EC2's `RunInstances`, for instance, takes a client token so that retrying the launch does not start duplicate instances, and DynamoDB transactional writes accept a client request token to dedupe. The principle is the one we derived from the lost-response problem: because the network is at-least-once and the SDK *will* retry on a timeout, the safe design is to make the operation idempotent (by method or by token) rather than to hope the response never gets lost. AWS engineering writing on this is explicit that retries are a fact of distributed systems and idempotency is the discipline that makes them safe.

**RFC 9110, the source of truth.** Both of the above are implementations of what the HTTP specification mandates. RFC 9110 (HTTP Semantics, the 2022 consolidation that supersedes the old RFC 7231) is the document that *defines* safe and idempotent, lists which methods are which, and states the consequence directly: a client *may* automatically retry an idempotent request when it believes the request did not reach the server. The PATCH method comes from RFC 5789; the two patch body formats from RFC 6902 (JSON Patch) and RFC 7396 (JSON Merge Patch). When a teammate insists "DELETE should return 404 on the second call" or "PATCH is always idempotent," the resolution is not opinion — it is reading the spec together. (For the cross-cutting view of how method semantics interact with distributed-systems realities at platform scale, the system-design series covers [REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql); we go deeper on the wire contract here.)

## When to lean on each method (and when not to)

Principles are only useful when they tell you what *not* to do. Here is the decisive version.

- **Reach for `GET`** for every read, and *only* for reads. Do not let a convenient link tempt you into a mutating `GET` — prefetchers and crawlers will fire it. If calling it twice from a browser would be a problem, it is not a `GET`. And do give `GET` responses real cache directives so the cacheability you get for free is actually used.
- **Reach for `POST`** to create with a server-assigned id, and for genuinely non-idempotent actions (charge, refund, send, enqueue). **Do not** use `POST` for something that has a fixed end state — that should be a `PUT` or `DELETE`, and you are throwing away free idempotence. And do not ship a `POST /payments` without an idempotency key; an unkeyed `POST` on a money path is a double-charge waiting for a timeout.
- **Reach for `PUT`** when the client holds (or can construct) the full representation and the update is a replacement — and for retry-safe create when the client can choose the id. **Do not** use `PUT` for partial updates; a `PUT` with a partial body silently deletes the fields you omitted. That is a data-loss bug, not a shortcut.
- **Reach for `PATCH`** for partial updates of large resources where sending the whole body is wasteful. Pick Merge Patch (RFC 7396) for simple field changes; pick JSON Patch (RFC 6902) when you need explicit deletes, array-element edits, or a built-in `test` precondition. **Do not** invent a third, bespoke patch format — pick a standard media type and honor its semantics. And **do not** assume `PATCH` is idempotent: keep operations absolute (replace, not append) or protect the endpoint with an idempotency key.
- **Reach for `DELETE`** to remove a resource; rely on its idempotence to make retries safe. **Do not** decide the second-call status code by accident — choose `404` (strict) or `204` (retry-friendly) deliberately and document it. If you soft-delete internally, keep the public contract meaning "the caller can no longer see this," or you will surprise people.
- **Reach for an idempotency key** whenever a non-idempotent method (`POST`, sometimes `PATCH`) sits on a path where a duplicate would be costly — money, irreversible side effects, external calls. **Do not** bolt it onto methods that are already idempotent (`GET`/`PUT`/`DELETE`); there it is redundant ceremony.

The meta-rule: choose the method by **intent**, accept the safe/idempotent guarantees that intent implies, and add an idempotency key exactly where intent and idempotence disagree.

## Key takeaways

- **Safe** means no observable side effect (`GET`/`HEAD`/`OPTIONS`); **idempotent** means N identical calls have the same effect on server state as one (`GET`/`PUT`/`DELETE`, and *not* `POST`/`PATCH`). They are independent axes, and every safe method is idempotent but not vice versa.
- Idempotence is about the **server's end state**, not the response code — which is why `DELETE` is idempotent even when the second call returns `404`.
- On an at-least-once network the **lost response** is the core problem: a timeout leaves the client unable to tell "never happened" from "happened, response lost." An idempotent request is safe to retry blindly; a non-idempotent one is not. That is *why* the property matters.
- `GET` must never mutate — the "GET that deletes" anti-pattern is a violation of the safety contract the whole web assumes, and prefetchers will trigger it.
- `PUT` replaces the whole resource (idempotent because it sets a fixed end state); `PATCH` modifies partially. Use **Merge Patch (RFC 7396)** for simple field changes and **JSON Patch (RFC 6902)** for explicit deletes, array edits, and `test` preconditions.
- A `PATCH` is **only as idempotent as its operations** — absolute `replace` is idempotent, relative `add`/append is not. Never let a generic retry library treat `PATCH` like `PUT`.
- `PUT`-for-create (client-chosen id) is idempotent; `POST`-for-create (server-assigned id) is not — the id-owner decides. When a `POST` must be retry-safe, add an **`Idempotency-Key`** so the server executes once and replays the stored result.
- Choose the method by **intent**; the guarantees follow for free; add a key exactly where the operation is non-idempotent but a duplicate would hurt.

## Further reading

- [RFC 9110 — HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html), §9 (Methods): the authoritative definitions of safe and idempotent and the per-method semantics.
- [RFC 5789 — PATCH Method for HTTP](https://www.rfc-editor.org/rfc/rfc5789.html): defines the `PATCH` method and why the body format is media-type-defined.
- [RFC 6902 — JavaScript Object Notation (JSON) Patch](https://www.rfc-editor.org/rfc/rfc6902.html): the ordered-operations patch format.
- [RFC 7396 — JSON Merge Patch](https://www.rfc-editor.org/rfc/rfc7396.html): the merge-a-sub-document patch format and the `null`-means-delete rule.
- The [Stripe API idempotency documentation](https://docs.stripe.com/api/idempotent_requests): the reference implementation of the `Idempotency-Key` pattern.
- Within this series: the [HTTP semantics overview](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers), [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx), and the deep dive on [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).
- The contract-and-product framing in the [intro hub](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and the [API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- For why "exactly once" is fundamentally hard on a network, the message-queue treatment of [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

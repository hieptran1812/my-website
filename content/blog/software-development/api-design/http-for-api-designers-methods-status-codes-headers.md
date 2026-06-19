---
title: "HTTP for API Designers: Methods, Status Codes, and the Headers That Matter"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn the HTTP semantics that actually drive good REST design — safe vs idempotent methods, honest status codes, and the headers that decide retries, freshness, and content — so your contract behaves the way callers expect."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "status-codes",
    "idempotency",
    "rfc-9110",
    "headers",
    "content-negotiation",
    "conditional-requests",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/http-for-api-designers-methods-status-codes-headers-1.png"
---

A payments integration I once owned double-charged a customer \$249.00 on a Friday night. Nobody wrote bad code. The mobile client sent a `POST /payments`, the server charged the card and started writing the `201 Created` response — and then the customer's train went into a tunnel. The TCP connection dropped before the response line arrived. The client's HTTP library did exactly what a reasonable HTTP library does on a connection error: it retried. A second `POST /payments` reached the server, which had no idea it was a replay, and charged the card again. Two charges, one intent, one furious customer, one pager going off at 11 p.m.

The bug was not in any single line. The bug was a misunderstanding of HTTP semantics: someone treated `POST` as if it were safe to retry, when the whole point of `POST` is that it is *not*. Every rule the rest of this series leans on — how to choose a status code honestly, why `PUT` is replaceable but `POST` is not, when a client may retry, when the server may cache, what a conditional request buys you — comes from a single specification, **RFC 9110, "HTTP Semantics."** It is the rulebook your callers already assume you are following, whether or not you have read it. This post is the principle layer: by the end you will be able to read and write a raw HTTP exchange, classify any method as safe and idempotent (or neither), choose the right status code from each family without lying, and reach for the right header — `ETag`, `Retry-After`, `Idempotency-Key` — to make your contract behave. We will ground every claim in HTTP semantics and walk it through the running **Payments & Orders** example: a `POST /payments`, a `GET /orders/{id}`, and a conditional `GET` that ends in a `304`.

![a vertical stack showing an HTTP exchange split into a request line, request headers, request body, a status line, response headers, and response body](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-1.png)

This is the second post in the series. The first, [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), framed the spine we keep returning to: **an API is a contract and a product, not a function call.** You are designing for a caller you will never meet, on a timeline of years. HTTP is the layer where that contract becomes concrete bytes on a wire. Get the semantics right and everything above — resources, errors, pagination, versioning — has a firm floor to stand on. Get them wrong and you ship a contract that lies, and the lie surfaces as a double charge in a tunnel at 11 p.m.

## 1. The anatomy of a request and a response

Before we can talk about *which* method or *which* status code, we have to agree on the shape of the message. An HTTP exchange is two messages: a **request** the client sends, and a **response** the server sends back. Each message has the same three-part structure — a start line, zero or more headers, and an optional body — and the entire contract between client and server is expressed in those three parts.

Here is a real request, byte for byte, the way it travels (the blank line that separates headers from body is part of the protocol — it is how the parser knows the headers have ended):

```http
POST /payments HTTP/1.1
Host: api.shop.example
Content-Type: application/json
Accept: application/json
Authorization: Bearer eyJhbGciOiJSUzI1Ni␊...
Idempotency-Key: 7f3b0c1e-9a2d-4c6f-8e10-2b9d4a1c5e7a
Content-Length: 84

{
  "order_id": "ord_8h2k",
  "amount": 24900,
  "currency": "USD"
}
```

Read it top to bottom. The **request line** is `POST /payments HTTP/1.1`: a method (`POST`), a request target (the path `/payments`), and the protocol version. Then a block of **headers**, each a `Name: value` pair, carrying metadata *about* the request — who I am (`Authorization`), what I am sending (`Content-Type`), what I will accept back (`Accept`), and a client-chosen key for safe retries (`Idempotency-Key`). A blank line. Then the **body**, here a small JSON document describing the payment. Note the amount is `24900` — an integer count of minor units (cents), not `249.00`; money in floats is its own category of bug, and we will treat money as integers throughout.

The response has the mirror-image shape:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /payments/pay_5tq9
ETag: "v1-a1b2c3"
Date: Fri, 20 Jun 2026 23:14:07 GMT

{
  "id": "pay_5tq9",
  "order_id": "ord_8h2k",
  "amount": 24900,
  "currency": "USD",
  "status": "succeeded"
}
```

The **status line** `HTTP/1.1 201 Created` carries the version, a three-digit **status code** (`201`), and a human-readable reason phrase (`Created`) that no program should ever parse — the code is the contract; the phrase is a courtesy. Then response headers — `Location` tells the client where the new resource lives, `ETag` is an opaque version tag we will use for caching — a blank line, and the body, which is the **representation** of the resource that was just created.

Two terms we will use constantly, defined now. A **resource** is the thing the URL identifies — the payment `pay_5tq9`, the order `ord_8h2k`. A **representation** is a concrete serialized snapshot of that resource at a moment in time, in some format — the JSON body above is *a* representation of the payment. The same resource can have many representations (JSON, XML, a summary view, a full view); content negotiation, later in this post, is how the client and server agree which one to exchange. This distinction is the quiet backbone of REST: you `GET` a resource and the server hands you a representation; you `PUT` a representation and the server makes the resource match it.

That is the whole anatomy. Methods live in the request line. Status codes live in the status line. The headers that matter live in the header blocks of both. Everything else in this post is a deeper reading of those three slots.

A few details about the request line repay attention because they shape your design. The **request target** is usually a path-and-query (`/payments?status=succeeded`), but it can also be an absolute URI (used when talking to a proxy) or `*` (for an `OPTIONS *` that asks about the server as a whole). For an API you will almost always work with paths and queries, and the split between them is a design decision the [URI design post](/blog/software-development/api-design/choosing-uris-collections-sub-resources-path-vs-query) treats in full: the **path** identifies the resource (`/orders/ord_8h2k`), the **query** parameters refine or filter the representation (`?fields=status,total`). Putting an identifier in the query (`/orders?id=ord_8h2k`) or a filter in the path is a smell that surfaces later as inconsistent caching and confusing semantics.

Headers are **case-insensitive in name** (`Content-Type` and `content-type` are the same header) and may legitimately appear multiple times for list-valued headers like `Accept`. HTTP/2 lowercases all header names on the wire, which is another reminder that the *name* is a key, not a string your code should compare case-sensitively. The body's length is communicated either by `Content-Length` (a fixed byte count) or by `Transfer-Encoding: chunked` (streamed in pieces, length unknown up front) — relevant when you design streaming or very large uploads, but for ordinary JSON bodies `Content-Length` is what you will see.

Finally, note what is *not* in the body. Metadata about the message — who sent it, what format it is, how fresh it is, how to cache it, how to authenticate it — lives in **headers**, not in the JSON. This is a discipline worth internalizing early: an `ETag` belongs in a header, not as a `version` field smuggled into the body; `Authorization` belongs in a header, not as a `token` query parameter that ends up in access logs. The protocol gives you a designed place for each kind of metadata, and using it keeps your bodies about *the resource* and your headers about *the exchange*.

## 2. Safe and idempotent: two words that decide what a client may retry

The single most useful pair of definitions in HTTP — the pair my double-charge bug came from misunderstanding — is **safe** and **idempotent.** They are different properties. A method can be one, both, or neither, and the combination tells you and your callers exactly what is allowed to happen on a retry. RFC 9110 defines them precisely; let me restate them in plain operational terms.

A method is **safe** if invoking it is *read-only* from the client's perspective — it does not request a change to server state. `GET`, `HEAD`, and `OPTIONS` are safe. Safe does not mean *nothing* happens on the server (a `GET` might write a log line, increment a counter, or warm a cache); it means the *client* is not asking for, and is not responsible for, any state change. The practical payoff: a safe request can be issued speculatively, prefetched by a browser, retried freely, and cached, because the client is promised that asking again costs nothing semantically.

A method is **idempotent** if the *effect on server state* of making the request **once** is the same as the effect of making it **N identical times.** Idempotent is about the *end state*, not the *response*. `PUT` is idempotent: `PUT /orders/42` with a given body sets order 42 to that body; doing it five times leaves order 42 in exactly the same state as doing it once. `DELETE` is idempotent: `DELETE /orders/42` makes order 42 gone; deleting an already-gone order leaves it gone. The *response* may differ on the second call (you might get `404` the second time), but the *server state* is identical — and idempotency is defined over state, not over responses.

Here is the key consequence, stated as a rule and then justified:

> **The retry rule.** A client may automatically retry a request only if the method is idempotent (or the request carries a mechanism, like an idempotency key, that makes it effectively idempotent). It may retry freely and cache only if the method is also safe.

Why does this rule hold? Because retries exist to defeat *ambiguous failures* — the cases where the client cannot tell whether the request succeeded. A connection that drops mid-flight, a timeout, a `5xx`: in all of these the client sent the request but never got a clean answer, so it does not know if the server acted. If the method is idempotent, the client can resolve the ambiguity by simply *trying again* — because the second attempt, if the first had in fact succeeded, leaves the server in the same place. The math is exactly the definition: for an idempotent operation, $f(f(x)) = f(x)$, so applying it once-or-twice is indistinguishable in the end state. For a non-idempotent operation like `POST /payments`, $f(f(x)) \neq f(x)$ — the second call charges the card again — so a blind retry is unsafe. That is the entire reason my customer was charged twice: a non-idempotent method was retried as if it were idempotent.

It is worth being precise about *why* the ambiguity is unavoidable, because it is a fact about distributed systems, not a flaw you can engineer away. When a client sends a request and never receives a response, there are three indistinguishable possibilities: (1) the request never reached the server; (2) it reached the server, the server acted, but the *response* was lost on the way back; (3) the server is still processing and will respond after the client's timeout. From the client's side these three look *identical* — it has a sent request and a missing response, and no amount of waiting tells it which case it is in. This is the **two generals problem** in miniature: you cannot guarantee both parties agree on whether an action happened using an unreliable channel. Idempotency is the practical escape hatch: if the operation is idempotent, the client does not *need* to know which case it is in, because retrying is safe in all three. Case (1) the retry does the work; cases (2) and (3) the retry is a harmless no-op on the end state. The whole value of idempotency is that it lets a client make progress *without* resolving an unresolvable question.

There is a subtlety that trips people up: **idempotency is about the effect, not the response code.** Consider `DELETE /orders/ord_8h2k`. The first call deletes the order and returns `204 No Content`. The second call finds nothing to delete and might return `404 Not Found`. The *responses differ* — but `DELETE` is still idempotent, because the *server state* is identical after one call or two: the order is gone. A client author who insists "it's not idempotent because I got a different status code the second time" has confused the two things RFC 9110 is careful to separate. The definition is over state. (Some APIs choose to return `204` even for an already-deleted resource to make the responses uniform too; that is a stylistic choice, not a correctness requirement.)

The two safe-and-idempotent methods beyond `GET` deserve a mention, because they are underused. **`HEAD`** is `GET` without a body: same headers, same status, no payload. It is the cheap way to ask "does this exist?" or "what is the current `ETag`/`Content-Length`?" without transferring the resource. **`OPTIONS`** asks what a resource supports — it is the mechanism behind CORS preflight in browsers and a way to discover the `Allow` set for a URL. Both are safe and idempotent, both are read-only, and both are free for a client to issue speculatively. Designing them in (or at least not breaking them) costs little and gives clients cheap ways to probe your API.

![a matrix comparing GET, PUT, DELETE, POST, and PATCH across safe, idempotent, and cacheable columns](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-2.png)

The figure above lays out the canonical table; here it is in prose, with the *reason* for each cell, because the reasons are what you carry to a new situation:

| Method | Safe? | Idempotent? | Cacheable by default? | Why |
| --- | --- | --- | --- | --- |
| `GET` | yes | yes | yes | pure read; same request, same effect (none); responses can be stored |
| `HEAD` | yes | yes | yes | like `GET` with no body; used to check existence or headers cheaply |
| `OPTIONS` | yes | yes | no | asks what is allowed; reads capability, changes nothing |
| `PUT` | no | yes | no | full replace; setting a resource to a value N times = setting it once |
| `DELETE` | no | yes | no | removal; deleting N times leaves it deleted |
| `POST` | no | no | rarely | "process this"; default semantics may create a new resource each call |
| `PATCH` | no | no by default | no | partial modify; not idempotent *unless* the patch is written to be |

Two cells deserve a closer look because they are where people get burned.

`POST` is **neither** safe nor idempotent, and that is by design. `POST`'s semantics in RFC 9110 are deliberately open: it means "the target resource should *process* the enclosed representation according to its own semantics." That openness is why `POST` is the catch-all for "create a thing," "run a job," "submit a form," "charge a card." None of those are naturally repeatable. So the default assumption a well-behaved client *must* make is: a `POST` that fails ambiguously **cannot** be blindly retried. To make a create safely retryable you have to *add* idempotency on top — which is what the `Idempotency-Key` header does, and what §6 covers.

`PATCH` is **not idempotent by default**, which surprises people who expect it to behave like `PUT`. The reason is that `PATCH` carries a *set of changes to apply*, not a full target state, and whether applying that set twice is a no-op depends entirely on what the changes are. A patch that says `set status to "shipped"` is idempotent — apply it twice, still shipped. A patch that says `increment quantity by 1` is **not** — apply it twice and you have added 2. JSON Patch (RFC 6902) `test` operations and JSON Merge Patch (RFC 7396) are common ways to write idempotent patches, but the method itself promises nothing. So a client must treat `PATCH` as non-idempotent unless your documentation explicitly says otherwise for a given resource.

The contrast with `PUT` is the cleanest way to feel the difference. `PUT /orders/ord_8h2k` says "make this resource *be exactly this representation*" — you send the whole order, and the result is fully determined by what you sent, regardless of what was there before. That total replacement is what makes it idempotent: the end state is a function of the request alone, not of the prior state. `PATCH` says "*apply these modifications* to whatever is there" — the result depends on both the patch *and* the current state, which is exactly why "add 1" is path-dependent. So the design rule is: if you want a clean idempotent partial update, prefer a JSON Merge Patch that *sets* fields to absolute values (`{"status": "shipped"}`), and avoid relative operations (increment, append) unless the client carries an idempotency key or the resource supports an explicit version precondition. The two RFCs differ in a way worth knowing: **JSON Merge Patch (RFC 7396)** is a partial JSON document where present keys overwrite and `null` deletes — simple, but it cannot patch inside arrays cleanly; **JSON Patch (RFC 6902)** is an ordered list of operations (`add`, `remove`, `replace`, `test`) that can target deep paths and even assert preconditions with `test`, at the cost of more verbosity. The dedicated method post covers both in depth; here the lesson is that the *shape* of your patch decides its idempotency.

#### Worked example: a PATCH that is and is not idempotent

The same endpoint, two patch bodies, two different safety verdicts. First, an absolute set — idempotent:

```http
PATCH /orders/ord_8h2k HTTP/1.1
Content-Type: application/merge-patch+json

{ "status": "shipped" }
```

Send this twice and the order is `shipped` either way; a retry is safe. Now a relative operation — *not* idempotent:

```http
PATCH /orders/ord_8h2k HTTP/1.1
Content-Type: application/json-patch+json

[ { "op": "test", "path": "/version", "value": 7 },
  { "op": "replace", "path": "/discount_cents", "value": 500 } ]
```

The `test` operation here is the trick that *restores* idempotency to an otherwise risky patch: it asserts the resource is at `version 7` before applying, and the server returns `409 Conflict` (or `412`) if the version has moved. A blind retry that arrives after the first one succeeded will find `version` bumped past 7, fail the `test`, and *not* double-apply. Without that `test`, a patch like `{ "op": "add", "path": "/notes/-", "value": "called customer" }` (append to an array) would add the note twice on a retry — a duplicate the customer-service team would later puzzle over. The `test`-guarded patch is the JSON-Patch equivalent of the `If-Match` header we meet in §6: both make a write conditional on the version the client last saw.

#### Worked example: classifying a retry decision

A mobile client times out on each of these. For each, decide: may the library auto-retry?

- `GET /orders/ord_8h2k` — **yes, freely.** Safe and idempotent. Worst case the server does the read twice; the customer notices nothing.
- `PUT /orders/ord_8h2k` with a full order body — **yes.** Idempotent: replaying the same full representation re-sets the order to the same state. (Caveat: if a concurrent writer changed the order between attempts, you may clobber them — that is the lost-update problem, solved with `If-Match` in §6, not by the retry itself.)
- `DELETE /payments/pay_5tq9` — **yes.** Idempotent. The second attempt may return `404` or `410` instead of `204`, but the payment stays deleted, which is all the client wanted.
- `POST /payments` with no idempotency key — **no.** Neither safe nor idempotent. A blind retry is the double-charge bug. The library should *not* retry; or, better, the request should have carried an `Idempotency-Key` so the server can dedupe it.

This single table — internalized, not memorized — prevents an entire class of production incidents. When a teammate proposes "let's just have the gateway retry failed writes," the right reflex is: *only the idempotent ones, and only if you can prove it.*

## 3. The status code families: a contract about responsibility

A status code is the most compressed promise in your API. It is three digits, and the **first digit** alone tells the caller something binding: who is responsible for what happened, and whether retrying could possibly help. RFC 9110 defines five families, and a correct client branches on the family *before* it ever looks at the body.

![a grid of the five status code families showing 2xx success, 3xx redirect, 4xx client error, and 5xx server error with retry guidance](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-4.png)

| Family | Meaning | Who must act | Retry helps? | Canonical examples |
| --- | --- | --- | --- | --- |
| `1xx` | informational | nobody (interim) | n/a | `100 Continue`, `101 Switching Protocols` |
| `2xx` | success | nobody | no | `200`, `201`, `202`, `204` |
| `3xx` | redirection / not-modified | client follows | no | `301`, `304`, `307`, `308` |
| `4xx` | client error | the *client* | not until the request is fixed | `400`, `401`, `403`, `404`, `409`, `422`, `429` |
| `5xx` | server error | the *server* | yes, with backoff | `500`, `502`, `503`, `504` |

The deep value of the families is in that **"retry helps?"** column, and it is worth deriving. A `4xx` says *the request itself is wrong* — bad syntax, missing auth, a precondition that failed, a duplicate that conflicts. Sending the identical bytes again will produce the identical error; the only way out is for the client to *change the request*. A `5xx` says *the request was plausibly fine but the server failed to handle it* — a crashed dependency, a timeout, an overloaded pool. The same request might succeed once the server recovers, so a retry **with backoff** is the correct response. This is precisely why returning the wrong family is so destructive: classify a transient overload as `4xx` and clients give up when they should retry; classify a malformed request as `5xx` and clients hammer you forever with bytes that can never succeed.

Now the hard part — choosing the *specific* code honestly. Most of the gray areas are pairs where two codes look applicable, and the choice carries information the caller depends on.

### 2xx: which success?

- **`200 OK`** — the request succeeded and the body is the result. Use for a `GET` that returns a representation, a `PUT`/`PATCH` that returns the updated resource, a `POST` that processed something and has a result to return but did *not* create a new addressable resource.
- **`201 Created`** — a `POST` (or `PUT`) created a **new resource**, and you are returning *its* representation. You **must** include a `Location` header pointing at the new resource. The distinction from `200` is real information: `201` tells the client "there is now a new thing at this URL," which a `200` does not. In our example, `POST /payments` returning `201` with `Location: /payments/pay_5tq9` lets the client store and later `GET` that URL.
- **`202 Accepted`** — the request was accepted for **asynchronous** processing that has *not* completed. The body typically describes a status resource the client can poll. Use this when a `POST` kicks off a long-running job (a refund that takes seconds to settle, an export). It is an honest "I have your request, I have not finished, here is where to watch." Do not return `200` for work that is still running — that is a lie the client will believe.
- **`204 No Content`** — success, and there is deliberately **no body.** The canonical use is a `DELETE` that succeeded, or a `PUT` that updated and has nothing new to say. A `204` must not carry a body; clients are entitled to stop reading. The design tension worth naming: a `PUT` or `PATCH` *can* return `200` with the updated resource (saving the client a follow-up `GET`) or `204` with nothing (saving bandwidth). Returning the updated representation is usually the better developer experience — the client sees the canonical, server-normalized result of its write, including any fields the server filled in — so prefer `200` with the body unless the resource is large and the client rarely needs it back.

There is a quiet principle linking the `2xx` choices to the rest of this post: **the success code, like the method, is a promise about what the client now knows.** A `201` promises "a new resource exists at `Location`." A `202` promises "I have your request but the work is *not done* — watch this status resource." A `204` promises "done, nothing to show." A client that gets `202` and treats it like `201` will try to `GET` a resource whose final state does not exist yet, and conclude your API is flaky. So choosing `201` vs `202` is not stylistic — it is the difference between "go ahead and use the result" and "keep polling." Tell the truth about completion the same way you tell the truth about failure.

#### Worked example: 201 vs 200 vs 202 on `POST /payments`

The same endpoint can honestly return all three depending on what actually happened:

```http
POST /payments HTTP/1.1
Content-Type: application/json

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD" }
```

- The card cleared synchronously and a payment resource now exists → **`201 Created`** + `Location: /payments/pay_5tq9` + the payment body.
- The provider returns "pending, settling asynchronously"; you created a payment in `processing` state and the client must poll → **`202 Accepted`** + a body with a status URL like `/payments/pay_5tq9` and `status: "processing"`.
- The client sent the same `Idempotency-Key` as a previous successful call → you replay the **stored** result. Stripe's convention here is to return the original code (`201`) again; some APIs return `200` to signal "this is a replay." Either is defensible; pick one and document it. We will use replay-returns-201 to keep the contract uniform.

### 3xx: redirection and the quiet workhorse, 304

- **`301 Moved Permanently` / `308 Permanent Redirect`** — the resource lives at a new URL forever; update your links. `308` preserves the method and body on redirect, `301` historically did not (browsers turned `POST` into `GET`), which is why `308` exists.
- **`304 Not Modified`** — the most useful `3xx` for an API. It is the answer to a *conditional* `GET`: "you already have the current version, here are no bytes, reuse what you cached." It is how `ETag` saves bandwidth, and §6 walks it end to end. A `304` carries no body and is, in effect, a `2xx` that costs nothing.

### 4xx: the client got it wrong — but in which way?

This is where most "honest status code" decisions live, and the gray pairs matter because each carries a different instruction to the client.

- **`400 Bad Request` vs `422 Unprocessable Content`.** Use `400` when the request is *malformed* — broken JSON, a missing required field, a value of the wrong type — the server could not even parse it into something meaningful. Use `422` when the request *parsed fine* but is *semantically invalid* — well-formed JSON, all fields present and well-typed, but `amount` is negative, or `currency` is `"ZZZ"`, or the order is already paid. The distinction tells the client whether the problem is *structure* (`400`) or *meaning* (`422`). Plenty of APIs collapse both into `400`; that is acceptable, but if you can split them you give clients a sharper signal. Be consistent: do not return `400` for a negative amount on one endpoint and `422` on another.
- **`401 Unauthorized` vs `403 Forbidden`.** This pair is mis-chosen constantly. `401` means *I do not know who you are* — no credentials, or invalid/expired credentials. It is a statement about **authentication**, and per RFC 9110 it **must** accompany a `WWW-Authenticate` header telling the client how to authenticate. `403` means *I know exactly who you are, and you are not allowed to do this* — a statement about **authorization**. The retry instruction differs: on `401` the client should *get fresh credentials and try again*; on `403` re-authenticating is pointless — the identity is fine, the permission is not. Returning `401` when you mean `403` sends clients into a credential-refresh loop that never fixes anything.
- **`404 Not Found` vs `410 Gone`.** `404` means *I have no resource at this URL* — it may never have existed, or it may exist but I am choosing not to reveal that (a common privacy tactic: return `404` instead of `403` so you do not leak that a resource exists to someone not allowed to see it). `410` means *this resource existed and is permanently gone* — a deprecated endpoint after its sunset, a deleted account past its retention window. `410` is a stronger, kinder signal to integrators: stop calling this, it is never coming back, update your code. Use `410` deliberately when you have *intentionally* retired something.
- **`405 Method Not Allowed`** — the URL exists but not for this method (`DELETE /payments` when payments cannot be bulk-deleted). You **must** include an `Allow` header listing the methods that *are* permitted, e.g. `Allow: GET, POST`.
- **`409 Conflict`** — the request conflicts with the current state of the resource. The classic case: trying to create something that already exists, or a write that violates a uniqueness or state constraint (refunding an already-refunded payment, transitioning an order from `delivered` back to `pending`). `409` says "your request was understood and valid in form, but it cannot be applied to the resource as it stands right now."
- **`412 Precondition Failed`** — a conditional write whose `If-Match` (or `If-Unmodified-Since`) precondition did not hold. This is the lost-update guard: "you tried to update the version you last saw, but it has changed since." §6 covers it.
- **`415 Unsupported Media Type`** — the request body's `Content-Type` is one the server cannot process (you sent `text/xml`, the endpoint takes only `application/json`). Distinct from `406 Not Acceptable`, which is about the *response*: the server cannot produce any of the types the client's `Accept` header asked for.
- **`428 Precondition Required`** — the server *requires* a conditional request and the client did not send one. This is how you *force* clients to use `If-Match` on writes so they cannot accidentally clobber concurrent updates. Returning `428` says "resend this with an `If-Match` header."
- **`429 Too Many Requests`** — the client exceeded a rate limit or quota. It **should** carry a `Retry-After` header telling the client how long to wait. This is the single most important code for protecting your service, and §6 pairs it with the header that makes it actionable.

### 5xx: the server got it wrong — and which layer?

- **`500 Internal Server Error`** — a generic, unexpected failure in *your* code: an unhandled exception, a bug. It says "this is on us and we did not anticipate it." Never use `500` for something the client did wrong — that misclassifies a `4xx` as a `5xx` and invites infinite retries.
- **`502 Bad Gateway`** — *you* are a proxy/gateway and an *upstream* server gave you an invalid response. In our example, the API gateway got garbage from the payments service. It points the finger at a dependency, not at the edge.
- **`503 Service Unavailable`** — you are temporarily unable to handle the request: overloaded, in maintenance, a dependency is down and you are shedding load. `503` **should** carry `Retry-After`. It is the honest "I am up enough to answer, but I cannot do this right now — come back in N seconds." It is also the correct code to return when a circuit breaker is open.
- **`504 Gateway Timeout`** — you are a gateway and the upstream did not answer in time. Distinct from `502` (upstream answered, but badly) and `503` (you are choosing not to serve).

The `502`/`503`/`504` distinction is not pedantry: it tells the *caller* (and your own on-call) where to look. A spike in `502` means an upstream is returning garbage; a spike in `503` means *you* are shedding load; a spike in `504` means an upstream is slow. Collapsing them all into `500` throws that diagnostic signal away.

#### Worked example: the same failure, three honest codes

Take one concrete failure — the payments provider is having a bad day — and watch how the *correct* code depends on what your service actually observed:

- Your gateway forwarded the charge to the payments service, which returned a malformed, non-HTTP response (a half-written body, a connection reset after the status line). Your gateway cannot make sense of it → **`502 Bad Gateway`.** The client learns: not your fault, an upstream broke.
- Your service is healthy but the payments provider has a known outage, so your circuit breaker is open and you are deliberately refusing to forward charges → **`503 Service Unavailable`** with `Retry-After: 60`. The client learns: temporarily off, come back in a minute.
- Your gateway forwarded the charge and the payments service simply never answered within your 5-second deadline → **`504 Gateway Timeout`.** The client learns: an upstream is slow; the charge *may or may not* have happened (back to the ambiguous-failure problem — which is exactly why the original `POST` should have carried an `Idempotency-Key`).
- A bug in *your* charge-mapping code threw a null-pointer exception → **`500 Internal Server Error`.** The client learns: this one is on you.

Four codes, one underlying "the payment did not go through," but each tells the client and your on-call a *different* true thing. That is the whole point of choosing honestly: the status code is the cheapest, most universally understood diagnostic signal you will ever emit, and every consumer — client libraries, gateways, dashboards, SLO calculators — already knows how to read it.

A note on the gray-area pairs, because choosing between them is where design judgment shows. The pairs are not arbitrary; each split exists because the two codes carry *different instructions to the client*:

- `400` vs `422`: **fix the structure** vs **fix the meaning.** A `400` says your bytes did not parse; a `422` says they parsed but violate a business rule. A client that gets `400` checks its serializer; a client that gets `422` checks its values.
- `401` vs `403`: **get credentials** vs **stop trying.** A `401` triggers a token refresh; a `403` should *not*. Returning `401` for an authorization failure sends clients into a refresh loop that can never succeed and looks, in your logs, like a client that "can't authenticate" when in fact it authenticated fine and simply lacks permission.
- `404` vs `410`: **maybe later** vs **never again.** A `404` leaves the door open (the resource might appear, or you are hiding it); a `410` slams it (this is permanently gone, update your integration). `410` is the polite signal to integrators that a deprecated endpoint has truly retired — far better than a silent `404` they keep retrying.
- `409` vs `412`: **state conflict** vs **version conflict.** A `409` says the resource's current *state* forbids your request (it is already refunded). A `412` says your request was conditional on a version that has since changed. Both are "the resource is not where you thought," but `412` specifically points at optimistic concurrency.
- `415` vs `406`: **I can't read your body** vs **I can't write what you'll accept.** `415` is about the request's `Content-Type`; `406` is about the response and the client's `Accept`. They sit on opposite ends of the same negotiation.

## 4. Don't lie with 200: the before→after that breaks clients

The most common status-code sin is not picking `400` over `422`; it is returning **`200 OK` with an error inside the body.** It usually starts innocently — a framework returns `200` by default, someone adds an `{"ok": false, "error": "..."}` envelope, and "the client can just check `ok`." It scales into an outage.

![a before and after comparison contrasting a 200 response carrying an error body with a proper 4xx response carrying a problem json body](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-5.png)

Here is the **before** — the lying success:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "ok": false,
  "error": "amount must be positive",
  "code": "INVALID_AMOUNT"
}
```

And here is the **after** — an honest `422` carrying an RFC 9457 `problem+json` body (the standard machine-readable error format: a document with `type`, `title`, `status`, `detail`, and `instance`):

```http
HTTP/1.1 422 Unprocessable Content
Content-Type: application/problem+json

{
  "type": "https://api.shop.example/problems/invalid-amount",
  "title": "Amount must be positive",
  "status": 422,
  "detail": "The field 'amount' was -24900 but must be a positive integer count of cents.",
  "instance": "/payments",
  "field": "amount"
}
```

Now walk the **consequences** of the `200` lie, layer by layer, because the damage is distributed across systems that never see the body:

- **The HTTP client library** records the call as a success. Its built-in error handling — the part that would surface the problem to application code — never fires. The failure is invisible to the caller's `try/except`.
- **The gateway and CDN** see `200` and, if the endpoint is cacheable or the response carries cache headers, may *cache the error*. Now every client gets the cached "success" that is really a failure.
- **Retries and circuit breakers** key off the status family. A `5xx` would trigger backoff; a `4xx` would *not* trigger a pointless retry. A `200` does neither — so a transient failure dressed as `200` is silently swallowed, and a permanent one is never retried into eventual success even when it should not be.
- **Dashboards and SLOs** compute error rate from status families. An API that returns `200` for failures reports a *fictional* 100% success rate while customers are getting charged twice. You cannot alert on an error you have hidden in a `200`.
- **Every consumer** must now write bespoke code to parse `ok: false` out of each endpoint, and they will get it wrong — one team checks `ok`, another checks `error == null`, a third checks the HTTP status (which is always `200`) and concludes everything is fine.

The fix is not "add a better envelope." The fix is: **the status line is the contract; tell the truth in it.** Use `4xx` for client errors and `5xx` for server errors, and put the human- and machine-readable detail in a `problem+json` body *under the correct code*. The next post on [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx) goes deeper on the full taxonomy, and the dedicated error-design post covers `problem+json` in full; the rule to carry from here is simply: **never return `200` for a failure.**

## 5. Content negotiation: one resource, many representations

Recall the resource-vs-representation distinction from §1: a resource (the payment) can be serialized many ways. **Content negotiation** is the HTTP mechanism by which client and server agree on *which* representation to exchange, using a small set of headers. You do not need to support a dozen formats to care about this — even a JSON-only API uses these headers, and using them correctly is what makes your `415`/`406` errors honest.

There are two directions. The **request body's** format is declared by the sender with `Content-Type`. The **response body's** format is *requested* by the client with `Accept` and *declared* by the server with its own `Content-Type`.

```http
POST /payments HTTP/1.1
Content-Type: application/json
Accept: application/json

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD" }
```

`Content-Type: application/json` says "the body I am sending is JSON." If the server only accepts JSON and you send `Content-Type: application/xml`, the honest reply is **`415 Unsupported Media Type`** — the server understood the request line but cannot process the body's format. `Accept: application/json` says "send the response back as JSON." If the server can only produce JSON and the client sends `Accept: application/xml`, the honest reply is **`406 Not Acceptable`** — the server cannot produce anything in the set the client will accept. (In practice many JSON-only APIs ignore a restrictive `Accept` and return JSON anyway; that is a pragmatic choice, but `406` is the *correct* code if you want to be strict.)

Media types are where API *versioning* sometimes lives, too. A **vendor media type** like `application/vnd.shop.v2+json` lets a client ask for version 2 of your representation via `Accept`, keeping the URL stable. We cover the trade-offs of media-type versioning versus URI versioning in the versioning post; for now the point is that `Accept`/`Content-Type` are not boilerplate — they are the knobs of negotiation, and they decide whether your `415` and `406` codes mean anything.

A related header you will meet: `Content-Language`, `Content-Encoding` (for compression — `gzip`, `br`), and the `Accept-Encoding` the client uses to request compression. Compression is a real lever on payload size and tail latency; we treat it properly in the performance post, but know that the negotiation lives in the same family of headers.

There is one more negotiation header that becomes critical the moment caching enters the picture: **`Vary`.** When a response depends on a request header — for example, you return JSON to `Accept: application/json` and a compact summary to `Accept: application/vnd.shop.summary+json` — a cache that keyed only on the URL would serve the wrong representation to the next client. `Vary: Accept` tells caches "this response varies by the `Accept` header, so key the cache entry on it too." Forgetting `Vary` is a classic way to ship a bug where one client's compressed response gets handed to another client that cannot decompress it, or one user's view gets served to another. The rule: any header you negotiate on, you must list in `Vary`, or you will poison shared caches.

#### Worked example: an honest 415 on the wrong Content-Type

A client integrating in a hurry sends an order as form-encoded data to a JSON-only endpoint:

```http
POST /orders HTTP/1.1
Content-Type: application/x-www-form-urlencoded
Accept: application/json

order_id=ord_8h2k&total=24900&currency=USD
```

The endpoint only knows how to parse `application/json`. The wrong move is to *try* to parse the form anyway, or to return `400` (which suggests "fix your JSON" when there is no JSON). The honest move is:

```http
HTTP/1.1 415 Unsupported Media Type
Accept-Post: application/json
Content-Type: application/problem+json

{
  "type": "https://api.shop.example/problems/unsupported-media-type",
  "title": "Unsupported request body format",
  "status": 415,
  "detail": "This endpoint accepts only application/json. You sent application/x-www-form-urlencoded."
}
```

The `415` plus an `Accept-Post` header listing what the endpoint *does* accept turns a confusing failure into a self-documenting one — the client sees exactly what to change. This is content negotiation doing its job: it is not decoration on a JSON API, it is the mechanism that lets your `415` carry a real, actionable instruction instead of a generic shrug.

## 6. Conditional requests, ETags, and the headers that change the contract

Now the headers that, more than any others, change what your API *promises*. Three jobs: **caching/freshness** (don't resend bytes the client already has), **concurrency control** (don't let two writers silently clobber each other), and **retries** (tell the client when and how to try again). All three are header-driven, and all three trace straight to RFC 9110.

![a matrix of headers showing Accept, Content-Type, ETag with If-Match, Retry-After, and Idempotency-Key and what each one controls](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-8.png)

Here is the same set as a reference table — the headers that *change what the client may assume*, who sends each, and the status code each one is associated with when things do not line up:

| Header | Sent by | Job | Related code |
| --- | --- | --- | --- |
| `Accept` | client | which response representation it will take | `406` if none can be produced |
| `Content-Type` | either side | the format of *this* message's body | `415` if the server can't read it |
| `Authorization` | client | the bearer token / credentials | `401` if missing or invalid |
| `ETag` | server | opaque version tag of the representation | enables `304` / `412` |
| `If-None-Match` | client | "only send if changed from this tag" | `304` if unchanged |
| `If-Match` | client | "only write if still at this tag" | `412` if it moved |
| `Cache-Control` | server | freshness directives (`max-age`, `private`) | governs `304` reuse |
| `Location` | server | URL of the created / status resource | paired with `201` / `202` |
| `Retry-After` | server | seconds (or date) until a retry is sensible | paired with `429` / `503` |
| `Idempotency-Key` | client | dedup token making a `POST` retry-safe | replays the stored `201` |

Read down the "Related code" column and you can see the design symmetry: each of these headers exists to make a particular status code *actionable* rather than merely informative. `Retry-After` turns `429` from "go away" into "go away for 30 seconds"; `ETag` turns a re-`GET` into a free `304`; `If-Match` turns a blind overwrite into an explicit `412` the client can recover from. Headers and status codes are two halves of one contract.

### Conditional GET: the ETag and 304

An **ETag** (entity tag) is an opaque string the server attaches to a representation — a version fingerprint. It can be a hash of the body, a version number, anything the server can compare. The client stores it and, on the next `GET`, sends it back in `If-None-Match`. If the resource has not changed, the server replies `304 Not Modified` with **no body**, and the client reuses its cached copy. A **conditional request** is exactly this: a request the server only fully fulfills if a precondition (here, "the version differs from the one I have") holds.

![a timeline showing a first GET returning 200 with an ETag, then a conditional GET with If-None-Match returning 304 Not Modified with no body](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-6.png)

Walk the round trip for `GET /orders/ord_8h2k`:

```http
GET /orders/ord_8h2k HTTP/1.1
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
ETag: "v7-3f9a"
Cache-Control: private, max-age=30

{ "id": "ord_8h2k", "status": "shipped", "total": 24900, "currency": "USD" }
```

The client caches the body and the tag `"v7-3f9a"`. Thirty seconds later it wants to re-check freshness without paying for the body:

```http
GET /orders/ord_8h2k HTTP/1.1
Accept: application/json
If-None-Match: "v7-3f9a"
```

If the order has not changed, the server compares the tag and short-circuits:

```http
HTTP/1.1 304 Not Modified
ETag: "v7-3f9a"
Cache-Control: private, max-age=30
```

No body. The client reuses its cached representation. The savings are real and worth quantifying honestly: an order representation of, say, 600 bytes of JSON is tiny, but a list endpoint returning a few hundred items can be tens to hundreds of kilobytes, and a `304` collapses that to a header-only response of well under 200 bytes. Over a cold mobile link, transferring 100 KB is typically in the 300–800 ms range just in transfer time, while a `304` is a single small round trip — so on a frequently-polled, rarely-changing resource the bandwidth and latency win is large. The `Cache-Control` header is the companion: `max-age=30` says "you may treat this as fresh for 30 seconds without even asking," and `private` says "only the end client may cache this, not a shared proxy" — important for anything tied to a user.

It is worth deriving *why* this is even allowed — why a `GET` can be cached but a `POST` generally cannot. Caching is the act of *reusing a prior response in place of issuing the request*. That is only sound if issuing the request again would not have changed anything — which is exactly the **safe** property. A `GET` is safe, so skipping it (by serving a cached copy) costs the server nothing semantically; the client's view of the world is the same whether it asked or reused. A `POST` is not safe, so "reusing a prior response instead of sending the request" would mean *skipping a state change the client asked for* — which is obviously wrong. So cacheability falls straight out of safety, the same way retryability fell out of idempotency in §2. The two big efficiency wins of HTTP — caching and safe retries — are both *consequences* of the two properties we defined at the start, which is why those definitions are the foundation the whole protocol stands on. The freshness model layers on top: `Cache-Control: max-age` gives a window during which the client may skip the request entirely, and the `ETag`/`304` mechanism handles the moment after that window expires, letting the client *revalidate* cheaply rather than re-download. The caching post develops the full freshness-and-revalidation lifecycle; here the principle is that conditional requests are the bridge between "definitely fresh" and "definitely stale."

### Conditional PUT: If-Match and the lost-update problem

The same ETag, used with a *different* conditional header, solves a completely different problem: **two writers racing.** Suppose two support agents open order `ord_8h2k`, both see `status: "shipped"`, and both edit it. Without protection, the second save silently overwrites the first — the **lost update.** The fix is a conditional write with `If-Match`:

```http
PUT /orders/ord_8h2k HTTP/1.1
Content-Type: application/json
If-Match: "v7-3f9a"

{ "id": "ord_8h2k", "status": "delivered", "total": 24900, "currency": "USD" }
```

The server applies the write **only if** the current ETag is still `"v7-3f9a"`. If agent two already saved (bumping the tag to `"v8-1c2d"`), the precondition fails and the server returns:

```http
HTTP/1.1 412 Precondition Failed
Content-Type: application/problem+json

{
  "type": "https://api.shop.example/problems/version-conflict",
  "title": "The order was modified by someone else",
  "status": 412,
  "detail": "You based your update on version v7-3f9a, but the current version is v8-1c2d. Re-fetch and retry."
}
```

This is **optimistic concurrency control**, and it is one of the highest-leverage uses of HTTP semantics there is: it turns a silent data-corruption bug into an explicit, recoverable `412` the client can handle by re-fetching and merging. If you want to *require* it — to forbid blind writes entirely — return **`428 Precondition Required`** when a write arrives without an `If-Match`, forcing every client to participate. The caching post goes deeper on weak vs strong ETags and on `Last-Modified`/`If-Modified-Since` as the date-based alternative; the principle here is that the *same* version tag powers both "don't resend" (`If-None-Match` → `304`) and "don't clobber" (`If-Match` → `412`).

### The Idempotency-Key: making POST safe to retry

Back to the double charge. `POST /payments` is not idempotent, so a client cannot blindly retry it. The fix is an application-level convention HTTP supports cleanly: the **idempotency key** — a unique client-generated value (a UUID is typical) that the server uses to deduplicate retries.

![a timeline showing a POST with an idempotency key, the server charging once and storing the result, a network timeout, a client retry with the same key, and the server returning the cached 201](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-3.png)

The mechanism, end to end:

1. The client generates a key per *intent* (one key for "charge \$249.00 for order ord_8h2k") and sends it: `Idempotency-Key: 7f3b0c1e-...`.
2. On the first request, the server checks whether it has seen that key. It has not, so it performs the charge **once**, stores the key alongside the resulting response, and returns `201`.
3. The connection drops; the client never sees the `201`. Its library retries the *same* `POST` with the *same* key.
4. The server looks up the key, finds the stored result, and **replays it** — returning the original `201` and the same payment `pay_5tq9` — **without charging again.**

```http
POST /payments HTTP/1.1
Content-Type: application/json
Idempotency-Key: 7f3b0c1e-9a2d-4c6f-8e10-2b9d4a1c5e7a

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD" }
```

A minimal server-side handler sketch makes the contract concrete:

```python
def create_payment(request):
    key = request.headers.get("Idempotency-Key")
    if key is None:
        return problem(400, "Idempotency-Key header is required")

    # Atomically claim the key, or fetch a prior result for it.
    existing = idempotency_store.get(key)
    if existing is not None:
        # Replay the stored response; do NOT charge again.
        return cached_response(existing)  # original 201 + body

    # First time we've seen this key: do the work exactly once.
    payment = charge_card(request.body)            # the side effect
    response = created_201(payment, location=f"/payments/{payment.id}")
    idempotency_store.put(key, response, ttl="24h")  # remember the result
    return response
```

A few rules that keep this honest. The key must be scoped to *one logical request*, not reused across different intents, or the second real charge would be wrongly suppressed. The stored result needs a TTL (24 hours is a common window) so the store does not grow forever. And the "claim the key" step must be atomic — two simultaneous retries that both miss the cache must not both charge; a database unique constraint or an atomic `SET NX` is the usual guard. This is why HTTP semantics matter all the way down: the `Idempotency-Key` header is the *contract*, but delivering on it requires at-least-once thinking from the message-queue world. For the deeper delivery-guarantee story — why "exactly once" is an illusion you assemble from at-least-once delivery plus idempotent receivers — see [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

### Retry-After: telling the client when to come back

When you return `429 Too Many Requests` (rate limited) or `503 Service Unavailable` (overloaded), the kind and machine-actionable thing to do is tell the client *when* to retry, with **`Retry-After`** — either a number of seconds or an HTTP date:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/problem+json
Retry-After: 30

{
  "type": "https://api.shop.example/problems/rate-limited",
  "title": "Rate limit exceeded",
  "status": 429,
  "detail": "You may make 100 requests per minute. Retry after 30 seconds."
}
```

A well-behaved client reads `Retry-After: 30` and waits 30 seconds instead of hammering you. Without it, clients guess — usually badly, often with a tight retry loop that turns a brief overload into a sustained one (the **thundering herd**). The rate-limiting post derives the token-bucket math behind the limit itself (allow a request if $tokens \ge 1$, refilling at rate $r$ tokens per second); here the point is that `429` *with* `Retry-After` is a complete, actionable contract, and `429` *without* it leaves the client blind.

### Location: where the new thing lives

We have already used it: `201 Created` should carry `Location` pointing at the new resource, and `202 Accepted` often carries `Location` pointing at the *status resource* the client polls. It is the header that makes "I created something" actionable — the client now has the URL to `GET` it.

#### Worked example: the conditional poll on an async refund

Combine three of these headers in one realistic flow — refunding a payment asynchronously, then polling for completion without wasting bytes:

```http
POST /refunds HTTP/1.1
Content-Type: application/json
Idempotency-Key: c0ffee00-1234-...

{ "payment_id": "pay_5tq9", "amount": 24900 }
```

```http
HTTP/1.1 202 Accepted
Location: /refunds/rfnd_9921
Content-Type: application/json

{ "id": "rfnd_9921", "status": "processing", "amount": 24900 }
```

The client polls the status resource, using `If-None-Match` so each poll is cheap while nothing changes:

```http
GET /refunds/rfnd_9921 HTTP/1.1
If-None-Match: "p1-aaaa"
```

```http
HTTP/1.1 304 Not Modified
ETag: "p1-aaaa"
```

Three `304`s later the refund settles, the ETag changes, and the poll finally returns the new state:

```http
HTTP/1.1 200 OK
ETag: "p2-bbbb"
Content-Type: application/json

{ "id": "rfnd_9921", "status": "succeeded", "amount": 24900 }
```

Every header earned its place: `Idempotency-Key` made the refund safe to retry, `202` + `Location` gave the client somewhere to watch, and the ETag made the polling loop nearly free until the moment something actually changed. This is the long-running-operation pattern in miniature; the LRO post develops it further, but it is built entirely from the HTTP semantics in this post.

## 7. The full request path: where a status code is decided

A status code is not chosen in one place. A request passes through several hops — TLS termination, the gateway, content negotiation, authentication, authorization, rate limiting, and finally your handler — and **any hop can short-circuit the request with its own status.** Understanding this path is what lets you assign blame correctly when something returns a `4xx` or `5xx` you did not expect.

![a branching graph of a POST flowing through content negotiation and an auth check, where each stage can short circuit with a 415 or 401 before reaching the handler that returns 201](/imgs/blogs/http-for-api-designers-methods-status-codes-headers-7.png)

Trace `POST /payments` through the hops:

1. **Content negotiation.** The gateway or framework checks the body's `Content-Type`. Wrong type → **`415`**, and the handler never runs. The `Accept` header is checked against what the server can produce → **`406`** if there is no overlap.
2. **Authentication.** Is there a valid `Authorization` token? No token or an expired one → **`401`** with `WWW-Authenticate`. This is "who are you," and it is decided before any business logic.
3. **Authorization.** The identity is known — is it *allowed* to create a payment for this order? No → **`403`**. This is "may you," and it is distinct from authentication. (Privacy tactic: if you do not want to reveal the order exists to someone not allowed to see it, return **`404`** instead of `403`.)
4. **Rate limiting.** Over the quota? → **`429`** with `Retry-After`, before you spend any compute.
5. **The handler.** Now the business logic runs. Valid create → **`201`** + `Location`. Already-paid order → **`409`**. Negative amount → **`422`**. An unhandled bug here → **`500`**. A downstream payments provider that times out → **`502`** or **`504`** depending on whether it answered.

The reason this matters for *design* is that it tells you where each code legitimately originates, which keeps your error contract coherent: the gateway owns `415`/`406`/`401`/`429`, the handler owns `409`/`422`/`201`, and `5xx` codes split between your code (`500`) and your dependencies (`502`/`504`). When you later put a real gateway in front of this — see [the API gateway and BFF pattern](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) — these responsibilities map directly onto layers you can configure independently. And when you secure the auth hop with mutual TLS between services, [service-to-service security](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) is where that lives; here we only need to know it produces a `401`/`403` decision before the handler.

The **ordering** of the hops is itself a design decision with cost implications. Notice that the cheap, broad rejections come *first*: you check `Content-Type` and `Accept` (a header read) before you check auth (a token verification, possibly a network call to an introspection endpoint), and you check auth and rate limits before you run the handler (which may touch a database or a payments provider). This is least-cost-first: reject the obviously-wrong requests using the cheapest check that can reject them, so you never spend an expensive operation on a request that a cheap one would have killed. A request that fails `415` should never reach the database. A request that fails `429` should never reach the payments provider. Getting this ordering wrong — for example, running the handler *before* the rate-limit check — means an attacker can exhaust your most expensive resource with requests you were always going to reject, which is a denial-of-service waiting to happen. The pipeline is not just where codes come from; it is your first line of load defense.

There is a principle here worth stating explicitly, because it is the *why* behind the whole hop ordering: **a request's status should be decided by the earliest hop that can decide it correctly.** A malformed `Content-Type` is fully knowable at the edge, so it should be rejected at the edge with `415`; whether an order is already paid is only knowable in the handler, so `409` belongs there. Pushing a decision earlier than the information allows produces wrong codes (the gateway guessing at business state); pushing it later than necessary wastes resources (the handler re-validating a token the gateway already rejected). Match each code to the hop that genuinely owns the information, and the contract stays both correct and cheap.

## 8. HTTP/1.1 versus HTTP/2: the connection underneath

One paragraph of plumbing, because it changes the cost model of your design without changing the semantics. HTTP/1.1 sends one request-response per connection at a time; to do several in parallel a client opens *several* TCP connections (browsers cap at around six per host), and within one connection requests queue head-of-line — a slow response blocks the ones behind it. Keep-alive lets a connection be *reused* across requests so you do not pay the TCP and TLS handshake every time, which is the main 1.1-era optimization. HTTP/2 multiplexes many concurrent streams over a **single** connection, compresses headers (HPACK), and removes the application-level head-of-line blocking, so a chatty API with many small requests becomes far cheaper — there is one handshake, one connection, and requests no longer wait in line. The semantics in this entire post — methods, status codes, headers, idempotency, conditional requests — are **identical** across 1.1, 2, and 3; the version only changes how efficiently those messages travel. The design lesson: 2 lowers the cost of small, numerous requests, which slightly weakens the "always batch to reduce round trips" reflex — but it does not change *what* a `201` or an `ETag` means. Payload size and compression still dominate latency for large responses; that is the performance post's territory.

Two practical consequences are worth carrying away. First, **header bloat is cheaper on HTTP/2/3** thanks to HPACK/QPACK header compression — repeated headers across requests on the same connection cost almost nothing — so the old instinct to minimize header verbosity matters less, though it never hurts to keep them lean. Second, **connection reuse is the single biggest latency lever you do not control in your handler**: a fresh TLS handshake is one to two extra round trips before any bytes of your response flow, and on a cold mobile link a round trip can be 50–150 ms, so a request that opens a new connection can pay 100–300 ms of pure setup before your server even sees it. This is why keep-alive (1.1) and multiplexing (2/3) matter so much for API performance: amortizing that handshake across many requests is often a larger win than shrinking the payload. HTTP/3 takes it further by running over QUIC (UDP) and eliminating the TCP-level head-of-line blocking that remained in 2, which helps most on lossy networks. None of this is your API's contract — it is the transport — but it is the cost model your design lives inside, and it is why "make fewer, fatter calls" is sound advice on 1.1 and weaker advice on 2/3.

## 9. Case studies: how mature APIs use these semantics

These are real, public, documented practices — useful precisely because they show the semantics in this post operating at scale.

**Stripe's idempotency keys.** Stripe's API lets clients send an `Idempotency-Key` header on `POST` requests (creating charges, refunds, etc.). Stripe stores the first response for a key and replays it on retries within a retention window, so a network retry on a charge does not double-charge. This is the exact mechanism in §6, and it is one of the reasons Stripe's API is trusted for money movement: the contract makes the unsafe-by-default `POST` safe to retry. Stripe also versions its API by date and pins each account to a version, which is the evolution story the versioning posts develop.

**GitHub's REST and conditional requests.** GitHub's REST API returns `ETag` and `Last-Modified` on resources and honors `If-None-Match`/`If-Modified-Since`, returning `304 Not Modified` for unchanged resources — and historically, a `304` did not count against the rate limit, which directly rewards clients for caching politely. The consequence design is elegant: a polling integration that respects `ETag` can check a resource frequently almost for free, because the `304`s are cheap and unmetered, while a naive integration that re-downloads everything burns through its hourly quota and starts getting `403`/`429`. The protocol is shaped so that the *polite* client and the *efficient* client are the same client. GitHub also publishes deprecation practice using `Sunset`-style signaling so integrators get a migration window. Both are §6 semantics applied to a large public surface, and both show the recurring lesson: good HTTP design makes the behavior you want from clients also the behavior that is cheapest for them.

**RFC 9457, `problem+json`.** The standard error format we used in §4 and §6 — a media type `application/problem+json` with `type`, `title`, `status`, `detail`, `instance` — is published as RFC 9457 (which obsoleted RFC 7807). Adopting it means clients can parse your errors with a shared, documented schema instead of bespoke per-endpoint guessing. It is the canonical answer to "how do I return a machine-readable error under the correct status code."

**Google's API Improvement Proposals (AIP).** Google's public AIP guide codifies status-code and method usage across its APIs — for example, standard methods (`Get`, `List`, `Create`, `Update`, `Delete`) mapping to HTTP verbs, and a consistent mapping of error conditions to codes. It is a good reference for *consistency at organizational scale*, which is the governance post's subject. The point for now: the biggest API providers do not invent semantics — they apply RFC 9110 rigorously and document the conventions on top.

### A stress test: walking the contract through failure

The best way to know whether you have the semantics right is to *stress-test the design* against the situations that actually break APIs in production. Take our `POST /payments` and ask, one failure at a time, what the contract promises:

- **The client times out and retries.** If the request carried an `Idempotency-Key`, the server replays the stored `201` and charges once; the contract holds. If it did not, you have the double-charge bug from the opening — which is *why* the key is not optional on a money-moving `POST`. The semantics told us this would happen the moment we classified `POST` as non-idempotent.
- **Two clients race to pay the same order.** The first `POST /payments` for `ord_8h2k` succeeds with `201`; the second should hit a uniqueness constraint and return `409 Conflict` ("this order already has a payment"), not a second charge. The `409` is the honest signal: the request was well-formed but conflicts with current state.
- **The client sends a negative amount.** The body parses fine, so this is not `400`; it violates a business rule, so it is `422 Unprocessable Content` with a `problem+json` body naming the field. A client that branches on the family knows immediately this is its mistake to fix, not ours.
- **The payments provider is down.** Your circuit breaker opens and you return `503 Service Unavailable` with `Retry-After: 60`. The client backs off for a minute instead of hammering you, and your `503` rate (not your `500` rate) spikes on the dashboard, pointing on-call straight at the dependency.
- **A support tool and a webhook both update the order at once.** Without `If-Match`, the later write silently clobbers the earlier — a lost update nobody notices until the numbers are wrong. With `If-Match` (and `428` to *require* it), the second write gets `412 Precondition Failed`, re-fetches, and merges. The corruption becomes a recoverable error.
- **A poller hammers `GET /orders/ord_8h2k` every second.** Without conditional requests, you transfer the full body 86,400 times a day per poller. With `ETag` + `If-None-Match`, all but the handful of polls that catch a real change return a tiny `304`, and your egress bill and tail latency drop accordingly.

Every one of these failures was *predictable from the semantics in this post* — and every honest answer is a status code or header we have already defined. That is the payoff of grounding design in RFC 9110: the spec already tells you what each failure should look like on the wire, so you are not inventing a contract under pressure during an incident; you are applying one you decided on calmly in advance.

## 10. When to lean on these semantics (and when not to)

Every rule here is a trade-off, and a few deserve an explicit "don't."

- **Do** return the most specific honest status code your clients can act on. **Don't** over-engineer the rare ones if your clients cannot distinguish them: if no consumer branches differently on `400` vs `422`, collapsing to `400` consistently is better than splitting them inconsistently. Consistency beats precision when precision is unused.
- **Do** add `Idempotency-Key` support to any `POST` that has a side effect a client might retry — payments, refunds, anything that moves money or creates a unique thing. **Don't** add it to safe reads or naturally-idempotent `PUT`/`DELETE`; it is pure overhead there, because those are already retry-safe.
- **Do** use `ETag` + conditional requests on resources that are read far more often than they change (a product catalog, an order detail). **Don't** bother on a resource that changes on nearly every read, or one so small that the body is cheaper than the bookkeeping — you would spend version-tracking effort to save nothing.
- **Do** require `If-Match` (with `428`) on writes to resources that multiple actors edit concurrently — anything where a lost update corrupts data. **Don't** force it on single-writer, owner-only resources where no race is possible; you would just add friction.
- **Do** send `Retry-After` with every `429` and `503`. **Don't** ever return `200` for a failure, and **don't** return `5xx` for a client mistake — the first hides outages, the second invites infinite retries.
- **Don't** retry non-idempotent requests automatically without an idempotency key. This is the one with teeth: it is the double-charge bug, and "the gateway will just retry everything" is how you ship it.

## 11. Key takeaways

- **Safe and idempotent are different properties.** Safe = read-only (the client requests no change); idempotent = doing it N times equals doing it once (same end state). `GET` is both; `PUT`/`DELETE` are idempotent-not-safe; `POST` is neither; `PATCH` is neither by default.
- **The retry rule falls straight out of idempotency.** A client may auto-retry only idempotent methods, because $f(f(x)) = f(x)$ resolves the ambiguous-failure problem. Retrying a `POST` blindly is the double-charge bug.
- **A status code is a contract about responsibility.** The first digit alone tells the caller who must act and whether a retry can help: `4xx` = fix the request, `5xx` = retry with backoff. Branch on the family before parsing the body.
- **Choose the specific code honestly:** `201` (with `Location`) for a create, `202` for async, `204` for empty success; `400` (malformed) vs `422` (semantically invalid); `401` (who are you) vs `403` (you may not); `404` vs `410` (gone for good); `409` for conflict, `412`/`428` for preconditions, `415`/`406` for negotiation, `429` (with `Retry-After`) for limits.
- **Never lie with `200`.** Returning success for a failure blinds client libraries, gateways, retries, circuit breakers, and your own SLO dashboards. Put the error under the right `4xx`/`5xx` with a `problem+json` body.
- **The same ETag powers two contracts:** `If-None-Match` → `304` (don't resend bytes the client has) and `If-Match` → `412` (don't let two writers clobber each other). Conditional requests turn silent data loss into an explicit, recoverable error.
- **The `Idempotency-Key` header makes an unsafe `POST` safely retryable** — but only if the server stores results, scopes the key to one intent, and claims the key atomically. The header is the contract; at-least-once delivery is the reality underneath.
- **HTTP semantics are version-independent.** Methods, codes, and headers mean the same thing on HTTP/1.1, 2, and 3; the version only changes how cheaply the messages travel.

## 12. Further reading

- **RFC 9110 — HTTP Semantics.** The canonical source for everything in this post: methods, safe/idempotent definitions, status codes, conditional requests, and header semantics. Read §9 (methods), §15 (status codes), and §13 (conditional requests).
- **RFC 9457 — Problem Details for HTTP APIs.** The `application/problem+json` error format (`type`/`title`/`status`/`detail`/`instance`); obsoletes RFC 7807.
- **RFC 5789 — PATCH Method for HTTP.** Defines `PATCH` and its non-idempotent-by-default semantics; pair with RFC 6902 (JSON Patch) and RFC 7396 (JSON Merge Patch).
- **RFC 6585 — Additional HTTP Status Codes.** Defines `428 Precondition Required`, `429 Too Many Requests`, and the `Retry-After` usage pattern.
- **MDN Web Docs — HTTP reference.** Practical, example-rich pages for each method, status code, and header; the fastest lookup when you are at the keyboard.
- **Within this series:** the intro hub, [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the deeper method post, [methods and idempotency](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete); the deeper code post, [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx); and the capstone, [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- **For the layers below the wire:** the paradigm-at-scale view in [REST vs gRPC vs GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql), and delivery guarantees in [at-most / at-least / exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

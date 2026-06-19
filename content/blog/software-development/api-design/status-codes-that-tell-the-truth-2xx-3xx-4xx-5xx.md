---
title: "Status Codes That Tell the Truth: 2xx, 3xx, 4xx, 5xx and the Gray Areas"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The honest guide to choosing HTTP status codes — the five families and their contract meaning, the 4xx gray areas like 400 vs 422 and 401 vs 403, the 5xx pitfalls, and why lying with 200 quietly breaks retries, caching, and alerting."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "status-codes",
    "error-handling",
    "rfc-9110",
    "problem-json",
    "rate-limiting",
    "conditional-requests",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-1.png"
---

At 2 a.m. a refund job stopped paging anyone, and that was the bug. The job called `POST /refunds` in a loop, and a downstream balance check had started rejecting half the requests because the merchant's float had run dry. The right thing to happen was a flood of `402`/`409` errors, a spike on the error dashboard, and a page to the on-call engineer. None of that happened. The refund service had been written to "always return 200 so clients don't have to handle errors," and it dutifully returned `200 OK` with a body that said `{"success": false, "reason": "insufficient funds"}`. Every layer between the job and the service — the load balancer's health logic, the metrics that count non-2xx responses, the client's retry policy, the alerting rule that fires on `5xx` rate — saw a clean wall of `200`s and concluded everything was fine. The failures were real, but they were invisible, because the status code lied. By morning, thousands of refunds were silently stuck, and the only signal anyone had was a customer-support queue filling up.

The status code is not decoration on top of the "real" answer in the body. It *is* part of the answer — arguably the most important part, because it is the one field that every participant in the request reads. The client reads it to decide whether to retry, redirect, re-authenticate, or surface an error to a human. The CDN and the cache read it to decide whether the response is storable and for how long. The load balancer reads it to decide whether the backend is healthy. Your metrics pipeline reads it to draw the error-rate graph the on-call dashboard is built on. A status code is a single integer with an outsized contract: it tells everyone in the chain *what happened* and *what to do next*, and it does so before anyone parses a byte of the body. When you return the wrong one — especially when you return `200` for something that failed — you are not being lenient with your callers. You are breaking retries, poisoning caches, and going dark on your own alerting.

![a two-by-three grid mapping the five status families onto success, redirect, client error, and server error rows with the client action each implies](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-1.png)

This is the eighth post in the **Designing APIs That Last** series, and it sits squarely in the *correct, predictable contract* layer of our spine: an API is a contract and a product, not a function call. The first post, [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), framed that spine; the second, [HTTP for API designers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers), laid down the semantics — methods, idempotency, the headers that matter — that this post leans on. Here we go deep on one field: the status code. By the end you will be able to choose the right code from each of the five families without lying; navigate the genuinely hard gray areas (`400` vs `422`, `401` vs `403`, `404` vs `403`, `404` vs `410`, `409` vs `412`, `429` and its `Retry-After`); pick the right `5xx` without ever leaking a stack trace; and pair every error with a machine-readable `problem+json` body so the human *and* the program on the other end both know what to do. We will ground each choice in **RFC 9110, "HTTP Semantics"** (and its companions, RFC 6585 for `428`/`429`/`431` and RFC 9457 for `problem+json`), and walk it through the running **Payments & Orders** API. Let us start with the one rule that everything else hangs on.

## 1. The status code is a contract, not a label

Before we argue about which specific code to send, we have to be precise about *who* reads it and *what they assume*. The honest way to think about a status code is as a promise to several audiences at once, each of whom acts on it mechanically.

The **client** branches on the first digit before it does anything else. A well-written client treats `2xx` as "proceed," `3xx` as "look elsewhere," `4xx` as "I sent something wrong, retrying the exact same request will not help," and `5xx` as "the server failed, a retry with backoff might succeed." HTTP libraries bake this in: a `requests` call in Python raises on `4xx`/`5xx` only if you ask it to, but retry middleware (urllib3's `Retry`, Go's `net/http` with a retry wrapper, the AWS SDK's retry layer) is configured by status family — they retry `429` and `5xx`, and they do *not* retry a `400`. If you return `200` for a failure, you have told that retry layer "success, stop here," and the failure is swallowed.

The **intermediaries** — the CDN, the shared cache, the reverse proxy — also read the code, and they cache on it. RFC 9110 defines a set of status codes as *cacheable by default*: `200`, `203`, `204`, `206`, `300`, `301`, `404`, `405`, `410`, `414`, `451`, `501`. That list matters more than it looks. A `404` is cacheable by default; a `503` is not. So if you return `200` for "this resource does not exist," a CDN may store that `200` and serve it to the next caller, who now sees a bogus success for a resource that was never there. The status code is an input to the caching machinery whether or not you intended it to be.

The **operators** — you, your dashboards, your SLOs — read the code in aggregate. The single most common API SLO is "the fraction of requests that return a non-`5xx` response," sometimes split into availability (no `5xx`) and a separate handle on `4xx` rate (often a sign of a client bug or an abuse pattern). If your service returns `200` for server-side failures, your availability graph is a flat green line over a burning building. The dashboard is only as honest as the codes feeding it.

> **The principle.** A status code is the one field of the response that is read by *every* participant — client, cache, proxy, load balancer, metrics pipeline — and acted on *mechanically*, before the body is parsed. Its contract is fixed by RFC 9110, not by your application. Therefore the status code must be *true*: it must reflect what actually happened, classified into the family whose mechanical behavior is correct for this outcome. Lying with the status code does not simplify your clients; it silently disables retries, corrupts caches, and blinds your own alerting.

Two corollaries fall straight out of this principle, and they are the cardinal sins we spend the rest of the post avoiding. **First: never return `200` with an error body.** The `{"success": false}` pattern is the refund outage above; it defeats every reader of the status line. **Second: never return `5xx` for a client error.** A `5xx` tells the client "back off and retry, I might recover" and tells your SLO "availability dropped." If the real problem is that the client sent an invalid `currency`, a retry will fail identically forever, and you have manufactured a fake availability incident out of a `400`. The status code points the finger; point it at the right party.

There is a third, quieter sin: using only `200`, `400`, and `500`. Plenty of APIs collapse the whole space into "ok / your fault / our fault." It is not *lying*, exactly, but it throws away information the client could have used. A `409 Conflict` and a `422 Unprocessable Content` are both "your fault," but the client's *next move* is completely different — re-read the current state and merge, versus fix a field and resend. Flattening them to `400` forces the client to parse the body to recover what the code should have told it for free.

![a vertical stack of the 2xx codes 200, 201 with Location, 202 Accepted, 204 No Content, and 206 Partial Content with what each one promises](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-2.png)

## 2. The 2xx family: success has shades

"It worked" is not one outcome. The `2xx` family encodes *how* it worked, and the differences are load-bearing for the client. Here are the five you actually use in a Payments & Orders API.

**`200 OK`** is the workhorse: a request succeeded and the body is the result. A `GET /orders/42` returns `200` with the order. A `POST` that performs an action and returns a result body (not a newly created resource) can also be `200`. It says nothing more than "success, here is what you asked for."

**`201 Created`** is `200` plus a specific fact: *a new resource now exists*, and you can find it at the URL in the `Location` header. This is the correct code for a successful `POST /payments` that creates a payment. The `Location` header is the part people forget, and it is the part that makes `201` worth more than `200`. It hands the client the canonical URL of the thing it just made, so the client never has to guess or reconstruct it.

```http
POST /payments HTTP/1.1
Host: api.shop.example
Content-Type: application/json
Idempotency-Key: 7f3b0c1e-9a2d-4c6f-8e10-2b9d4a1c5e7a

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD" }
```

```http
HTTP/1.1 201 Created
Location: /payments/pay_5kq9
Content-Type: application/json
ETag: "v1"

{ "id": "pay_5kq9", "order_id": "ord_8h2k", "amount": 24900,
  "currency": "USD", "status": "succeeded" }
```

(Note the amount is `24900` — an integer count of minor units, cents, not `249.00`. Money in floats is its own category of bug; we keep amounts as integers throughout, and a charge worth \$249.00 is `24900` on the wire.)

**`202 Accepted`** is the honest code for asynchronous work. It says: *I have accepted your request, but I have not finished it, and I may not even have started.* This is the right code when `POST /refunds` enqueues a refund that a downstream processor will settle in seconds or minutes. You return `202` with a pointer to a status resource the client can poll, and crucially you do *not* return `201` — because nothing is `Created` yet, and lying with `201` would tell the client the refund is done when it is merely queued.

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
Location: /refunds/rf_77a2

{ "id": "rf_77a2", "status": "pending",
  "status_url": "/refunds/rf_77a2" }
```

The client then polls `GET /refunds/rf_77a2` until `status` becomes `succeeded` or `failed`. We go deep on this long-running-operation pattern — `202`, the status resource, polling versus webhooks — in the dedicated post later in the series; here the point is narrow: *async work is `202`, not `200` and not `201`*.

**`204 No Content`** means "success, and there is deliberately no body to return." It is the right answer to a `DELETE /payments/pay_5kq9/hold` that succeeds, or a `PUT` that updates a resource where the client does not need the representation echoed back. A `204` *must* have an empty body; a client that sees `204` will not even try to parse one. Do not return `200` with `{}` when you mean `204` — the `204` tells the client "stop, nothing here" without a wasted parse, and it is a stronger contract.

**`206 Partial Content`** answers a ranged request: the client sent a `Range` header (say, asking for bytes 0–1023 of a large export), and the server returns just that slice with a `Content-Range` header describing it. You rarely hit `206` in a JSON CRUD API, but it appears the moment you serve large downloadable artifacts — a CSV export of a million orders, a PDF invoice — and want resumable downloads. The contract is: `206` plus `Content-Range` means "here is the slice you asked for, not the whole thing." The reason it earns its own code rather than reusing `200`: a client resuming an interrupted 500 MB export download needs to know whether the server *honored* its `Range` request (`206`, append to what I already have) or *ignored* it and sent the whole file from byte zero (`200`, discard my partial copy). Collapsing both to `200` would force the client to inspect `Content-Range` to find out — the same "make the client parse what the code should tell it" smell we keep returning to.

A word on why `201` versus `200` is not pedantry. The difference is the `Location` header and what the client may *assume* from it. A `200` says "here is a result"; a `201` says "a new, addressable resource now exists, and its canonical URL is in `Location`." That distinction drives client behavior: a client that creates a payment with `201 + Location: /payments/pay_5kq9` can store that URL and `GET` it later, poll it, or hand it to another system, all without reconstructing the path from the response body's `id` field and a hard-coded template. When you return `200` for a create, you force every client to know your URL-construction rules — `/payments/{id}` — which is exactly the kind of out-of-band knowledge a self-describing contract is supposed to eliminate. The `Location` header *is* the contract clause that says "you do not need to know how my URLs are built." Return it.

The flip side: do not invent a `201` where nothing was created. A `POST /payments/pay_5kq9/capture` that captures an existing authorized payment does not *create* a resource — it transitions one — so the honest code is `200` (with the updated payment) and there is no `Location`, because the resource already had a URL. Reserve `201` for the case where a brand-new resource came into existence and now has an address it did not have a moment ago.

#### Worked example: classifying a payment-creation outcome

Suppose a single `POST /payments` can end four ways, and we want each to carry the truthful code. The card charges and the payment is created: `201 Created` with `Location: /payments/pay_5kq9`. The charge is routed to an async processor that has not settled yet: `202 Accepted` with a `status_url`. The same idempotency key was already used and the payment already exists: also `201` (or `200`), returning the *cached* original response, because the contract for an idempotent retry is "give me back what you gave me last time" — never a second charge, and never a `409`. The `currency` field is `"US"` instead of a valid ISO code: that is not a `2xx` at all — it is a `422`, which we get to below. One endpoint, four codes, each one honest about what the server did. A client reading only the status line already knows whether to follow a `Location`, start polling, or surface a validation error to the user.

## 3. The 3xx family: look somewhere else (and 304 is the quiet star)

`3xx` means "the resource you want is reachable, but not exactly the way you asked." Most `3xx` codes are about *redirection* — the URL moved — and a couple are about *conditional requests*. The distinctions that bite are which redirects preserve the HTTP method, and the role of `304`.

The redirect codes split on permanence and on method preservation:

| Code | Permanence | Method on redirect | Use it for |
| --- | --- | --- | --- |
| `301 Moved Permanently` | permanent | historically may switch to GET | a resource that moved for good; update your bookmarks |
| `302 Found` | temporary | historically may switch to GET | a temporary redirect; the old URL is still canonical |
| `307 Temporary Redirect` | temporary | **method preserved** | temporary redirect that must keep `POST` as `POST` |
| `308 Permanent Redirect` | permanent | **method preserved** | permanent move that must keep `POST` as `POST` |

The trap lives in the "method on redirect" column. Historically, many clients responded to a `301` or `302` on a `POST` by re-issuing the follow-up request as a `GET` — silently changing the method. That is fine for a browser following a form-post-then-redirect-to-a-page pattern, but it is *catastrophic* for an API: a `POST /payments` that gets `302`-redirected and silently downgraded to `GET /payments/new-location` will not charge anything, and the client will think it did. RFC 9110 introduced `307` and `308` precisely to remove the ambiguity: **`307` and `308` preserve the method and body.** Rule of thumb for an API: if you must redirect a non-idempotent request, use `307`/`308` so the method survives, and prefer not to redirect writes at all.

The other `3xx` worth its own paragraph is **`304 Not Modified`**, and it is the most useful `3xx` in any read-heavy API. It is the success side of a *conditional request*: the client sends a `GET` with an `If-None-Match` header carrying the `ETag` (entity tag — an opaque version identifier the server assigned) it already has, and if the resource has not changed, the server returns `304` with an *empty body*. The client keeps using its cached copy. No body is transferred; the round-trip is reduced to headers.

![a left-to-right timeline of a conditional GET where a first 200 hands back an ETag and a later If-None-Match request returns a 304 with no body](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-3.png)

```http
GET /orders/42 HTTP/1.1
Host: api.shop.example
If-None-Match: "W/v7"
```

```http
HTTP/1.1 304 Not Modified
ETag: "W/v7"
Cache-Control: max-age=60
```

The quantitative payoff is real. Suppose an order representation is roughly 4 KB of JSON, and a mobile client polls it every 30 seconds to keep a screen fresh. With plain `200`s, every poll transfers ~4 KB; over a cold mobile link that is meaningful battery and bandwidth, and the transfer time is dominated by the body. With conditional requests, an unchanged order returns a `304` that is a few hundred bytes of headers — call it a 10–20× reduction in transferred bytes per poll on the common case where nothing changed. The server still does the work of computing the current `ETag`, but it skips serializing and shipping the body. We go deep on `ETag` generation, `Cache-Control`, and invalidation in the caching post later in the series; the status-code point is that **`304` is part of the success path, and a client that handles it cuts its own data bill.**

A common mistake: returning `200` with the full body even when the client sent `If-None-Match` and nothing changed. That is not *wrong* in the sense of lying — the request did succeed — but it throws away the entire benefit of the conditional request. If your endpoint emits `ETag`s, it must also honor `If-None-Match` and answer `304` when appropriate.

#### Worked example: a 302 that silently downgrades a payment

This is the redirect trap from section 3, walked end to end, because it is the one `3xx` mistake that quietly *loses money*. Suppose you move the payments endpoint behind a new path and decide to redirect the old one. A junior engineer reaches for the most familiar redirect, `302 Found`, and configures the gateway so `POST /charge` redirects to `POST /payments`. The intent is "send the same request to the new path." Here is what actually happens on the wire with a client that follows redirects (most HTTP libraries do, by default):

```http
POST /charge HTTP/1.1
Host: api.shop.example
Content-Type: application/json
Idempotency-Key: 7f3b0c1e-9a2d-4c6f-8e10-2b9d4a1c5e7a

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD" }
```

```http
HTTP/1.1 302 Found
Location: /payments
```

Now the client follows the `Location`. Historically — and this is baked into a great many clients for browser-compatibility reasons — a client that receives a `301` or `302` on a `POST` re-issues the *follow-up* request as a **`GET`**, dropping the method *and the body*:

```http
GET /payments HTTP/1.1
Host: api.shop.example
```

The body is gone. The method is wrong. The new endpoint sees a bare `GET /payments` (a *list* request, most likely), returns `200` with a page of existing payments, and the client — which never inspected the redirect — concludes the charge succeeded. **Nothing was charged.** The customer's order sits unpaid, the client's logs show a clean `200`, and the bug is invisible until reconciliation flags the missing money days later. The fix is one digit: use **`307 Temporary Redirect`** (or `308` for a permanent move), which the spec defines to *preserve the method and body*. With a `307`, the follow-up request is `POST /payments` with the original JSON and the original `Idempotency-Key` intact, and the charge goes through exactly once. The lesson generalizes: **never redirect a non-idempotent request with `301`/`302`; use `307`/`308` if you must redirect at all, and prefer not to redirect writes.** The safest design avoids redirecting writes entirely — return the new path in documentation and let clients update, rather than depending on every client's redirect-following behavior matching your assumption.

## 4. The 4xx gray areas, part one: validation and identity

Now the hard part. The `4xx` family — "you, the client, sent something I cannot or will not process" — has the most codes and the most genuine ambiguity. Pick the wrong `4xx` and you do not break a cache, but you mislead the client about *what to do next*, which is almost as bad. Let us take the pairs one at a time, with Payments/Orders examples for each.

![a matrix of the five gray-area pairs four hundred versus four twenty-two and others, each row showing what it means and what the client should do](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-4.png)

### 400 vs 422: malformed versus valid-but-rejected

This is the most-argued pair in API design, and the honest distinction is clean once you state it. **`400 Bad Request`** means the server *could not understand the request* — the syntax is broken. The JSON did not parse, a required header is missing, a query parameter is unparseable. The request never got far enough to be evaluated against business rules. **`422 Unprocessable Content`** (defined in RFC 9110, originally from WebDAV) means the server *understood the request perfectly* — the JSON parsed, the shape is right — but the *content* violates a semantic rule. The `amount` is negative; the `currency` is a well-formed string but not a currency you support; the `order_id` references an order that is already fully paid.

The reason the distinction matters to the client: a `400` usually points at a bug in how the client *constructs* requests (it is sending malformed JSON), while a `422` points at the *data the user entered* (the amount is wrong). Different teams, different fixes. Consider these two failures of the same endpoint:

```http
POST /payments HTTP/1.1
Content-Type: application/json

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD"
```

That body is missing its closing brace. The server cannot parse it. The truthful answer is `400`:

```http
HTTP/1.1 400 Bad Request
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/malformed-json",
  "title": "Request body is not valid JSON",
  "status": 400,
  "detail": "Unexpected end of input at byte 71." }
```

Now the body parses fine, but the values are wrong:

```http
POST /payments HTTP/1.1
Content-Type: application/json

{ "order_id": "ord_8h2k", "amount": -500, "currency": "ZZ" }
```

```http
HTTP/1.1 422 Unprocessable Content
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/validation-failed",
  "title": "One or more fields failed validation",
  "status": 422,
  "detail": "See errors for per-field details.",
  "errors": [
    { "field": "amount", "code": "must_be_positive",
      "message": "amount must be greater than zero" },
    { "field": "currency", "code": "unsupported",
      "message": "ZZ is not a supported ISO 4217 currency" }
  ] }
```

Both are paired with a `problem+json` body (RFC 9457 — media type `application/problem+json`, with `type`, `title`, `status`, `detail`, and an optional `instance`). We cover the design of that envelope in depth in the dedicated [error-design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract); the rule we enforce here is simply that *every `4xx` and `5xx` carries a `problem+json` body* so the client gets a machine-readable reason, not just a number. The `errors` array above is a domain extension on top of the standard members, which RFC 9457 explicitly allows.

Is it ever fine to just use `400` for everything in the `4xx`-validation zone? Yes — pragmatically, many excellent APIs use `400` for both syntax and semantic errors and lean entirely on the `problem+json` body to tell the two apart. GitHub's REST API, for instance, returns `422` for validation failures, while plenty of others return `400`. The honest position: `422` is the more precise code and is worth using if your framework distinguishes parse failures from validation failures; but a `400` with a *good* `problem+json` body is defensible and never *lies*. What is not defensible is a `200` with the validation errors in the body — that is the cardinal sin again.

#### Worked example: a 422 validation failure with problem+json

Walk the full cycle. A merchant's checkout form submits a payment for an order, but the customer typed an amount that exceeds the order total, and the currency dropdown defaulted to a code you do not support. The request is well-formed JSON; nothing is malformed. The server parses it, runs it through validation, and two rules fail. The truthful response is `422 Unprocessable Content` — *not* `400` (the request was understood) and *certainly* not `200` (it failed). The body is `application/problem+json` with a per-field `errors` array so the form can highlight exactly the `amount` and `currency` fields. The client reads `422`, knows this is a *data* problem the user must fix (not a transient failure to retry, not an auth problem to re-login), and re-renders the form with the two field errors attached. The retry layer in the HTTP client does *nothing* — `422` is not in its retryable set, which is correct, because resending the identical bad data would fail identically. One status code drove the client's entire recovery path without it parsing a sentence of prose.

### 401 vs 403: unauthenticated versus unauthorized

This pair confuses people because both feel like "you can't do that." The distinction is about *which question failed*. **`401 Unauthorized`** is misnamed in the spec — it really means *unauthenticated*: "I do not know who you are." The credentials are missing, malformed, or expired. The fix is to authenticate (log in, refresh the token, send a valid `Authorization` header). RFC 9110 *requires* a `401` to include a `WWW-Authenticate` header telling the client how to authenticate:

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="api", error="invalid_token"
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/unauthenticated",
  "title": "Missing or expired credentials",
  "status": 401 }
```

**`403 Forbidden`** means *unauthorized*: "I know exactly who you are, and you are not allowed to do this." The credentials are valid; the *permissions* are not. A merchant authenticated as `acct_123` tries to refund a payment that belongs to `acct_999`. Authenticating harder will not help — the same identity will be forbidden again. So a `403` should *not* carry `WWW-Authenticate`, because there is no authentication challenge to answer.

```http
HTTP/1.1 403 Forbidden
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/forbidden",
  "title": "You do not have permission to refund this payment",
  "status": 403 }
```

The decision rule is a single question: *would presenting valid credentials fix this?* If yes, it is `401` (the client needs to authenticate). If no — the credentials are already valid and the *identity* lacks permission — it is `403`. We treat the design of scopes, roles, and resource-level permissions that produce these `403`s in the dedicated authorization post later in the series.

| Pair | First code | Second code | The question that decides |
| --- | --- | --- | --- |
| 400 vs 422 | `400` malformed syntax | `422` valid shape, bad data | Did the request *parse and match the schema*? |
| 401 vs 403 | `401` unauthenticated | `403` unauthorized | Would *valid credentials* fix it? |
| 404 vs 410 | `404` not found (maybe later) | `410` gone (permanently) | Is the resource *gone for good and known to be*? |

## 5. The 4xx gray areas, part two: hiding existence, conflicts, and preconditions

### 404 vs 403: when to hide that something exists

Here is a subtle one with a security edge. Suppose the merchant `acct_123` requests `GET /payments/pay_owned_by_someone_else`. The payment exists, but it belongs to a different account. The "correct" code is arguably `403 Forbidden` — you are authenticated, you are just not allowed. But returning `403` *confirms the resource exists*. An attacker probing IDs can walk the space and learn which payment IDs are real by watching for `403` (exists, forbidden) versus `404` (does not exist). That is an enumeration leak.

The defensive pattern is to return **`404 Not Found` to hide existence**: from the perspective of an unauthorized caller, a resource they may not see *does not exist*. This is exactly what GitHub does for private repositories — request a private repo you have no access to and you get a `404`, not a `403`, precisely so that the API does not leak the existence of private repositories to people who should not know about them. The trade-off is honesty toward *legitimate* callers: a user who genuinely lost access to something they used to see now gets a confusing `404`. The rule of thumb: **use `404` to hide existence when the resource ID space is guessable or the existence of the resource is itself sensitive; use `403` when the caller is clearly entitled to know the thing exists but not to perform the action** (e.g. a teammate who can see an order but cannot refund it). For a payments resource keyed by an opaque, unguessable ID, `403` is often fine; for sequentially numbered IDs, `404` is the safer default.

### 404 vs 410: not found versus gone

Both mean "you won't find it here," but they make a different promise about the *future*. **`404 Not Found`** means "no resource at this URL *right now*" — it might appear later, it might be a typo, the server is non-committal. **`410 Gone`** is a stronger, deliberate statement: "this resource *existed and has been permanently removed*; stop asking, and update any links you have." `410` is the honest code for a deprecated, sunset endpoint you have fully retired, or a resource you hard-deleted and will never resurrect. The difference matters to crawlers and caches: a `410` tells a well-behaved client to purge the URL from its index, whereas a `404` invites periodic retries. Recall too that *both* `404` and `410` are cacheable by default per RFC 9110 — another reason to mean them.

In our API: a `GET /payments/pay_neverexisted` (a bad ID) is `404`. A `GET /v1/legacy-checkout` for an endpoint you sunset and removed last quarter is `410 Gone` (paired, ideally, with a `problem+json` pointing at the replacement). We cover the full deprecation-and-sunset choreography — the `Deprecation` and `Sunset` headers, the migration window — in its own post.

### 405 Method Not Allowed and the Allow header

`405 Method Not Allowed` means the *resource exists* but does not support the *method* you used. A `DELETE /payments/pay_5kq9` on an API where payments cannot be deleted (only refunded) is a `405`, not a `404` (the payment is right there) and not a `403` (it is not a permission issue). RFC 9110 *requires* a `405` to include an **`Allow` header** listing the methods that *are* supported, so the client learns the correct verb from the response itself:

```http
HTTP/1.1 405 Method Not Allowed
Allow: GET, POST
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/method-not-allowed",
  "title": "DELETE is not supported on /payments/{id}",
  "status": 405 }
```

Omitting the `Allow` header on a `405` is a contract violation that leaves the client guessing. It is cheap to include and the spec mandates it.

### 409 Conflict: the state clash

**`409 Conflict`** means the request was well-formed and authorized, but it conflicts with the *current state* of the resource. The classic Payments examples: trying to refund a payment that is already fully refunded; trying to cancel an order that has already shipped; two writers racing to update the same order. The defining trait of a `409` versus a `422` is that a `422` is about the *request's data* being wrong in isolation, while a `409` is about the request being *fine on its own* but incompatible with the *server's current state* — which means the client's correct recovery is to *re-read the current state and decide what to do*, not to fix a field.

```http
HTTP/1.1 409 Conflict
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/already-refunded",
  "title": "Payment has already been fully refunded",
  "status": 409,
  "detail": "Payment pay_5kq9 was refunded in full on 2026-06-18." }
```

#### Worked example: a 409 conflict on concurrent update

Two operators open the same order in two browser tabs to edit the shipping address. Both load `GET /orders/42`, which returns the order with `ETag: "W/v7"`. Operator A saves first, sending a conditional `PUT /orders/42` with `If-Match: "W/v7"`. The server checks: current `ETag` is `W/v7`, it matches, so the write succeeds and the order advances to `ETag: "W/v8"`. Now Operator B, still holding the stale `W/v7`, saves their edit:

```http
PUT /orders/42 HTTP/1.1
If-Match: "W/v7"
Content-Type: application/json

{ "shipping_address": { "line1": "742 Evergreen Terrace", "city": "Springfield" } }
```

The server compares the `If-Match` precondition (`W/v7`) against the current `ETag` (`W/v8`). They do not match — the order changed under Operator B. The truthful response is **`412 Precondition Failed`** (more on `412` next), which tells Operator B's client "the version you based this edit on is stale; re-fetch." If instead the conflict were *semantic* rather than version-based — say Operator B tried to change the address on an order that has *already shipped* — the right code is **`409 Conflict`**: the request is incompatible with the order's state, and re-fetching reveals why. The distinction: `412` is "your `If-Match`/`If-Unmodified-Since` precondition failed on the version"; `409` is "your request conflicts with the resource's state for a domain reason." Both push the client to re-read; `412` is the version-precondition flavor, `409` the general state-conflict flavor. Returning `200` here — silently overwriting Operator A's change with B's — is the lost-update bug, and the status code is exactly the mechanism that prevents it.

### 412 Precondition Failed and 428 Precondition Required

We just met **`412 Precondition Failed`**: the client sent a precondition header (`If-Match`, `If-Unmodified-Since`, `If-None-Match` on a write) and it did not hold against the current state. It is the conditional-request enforcement code, and it is how you implement optimistic concurrency control: the client says "only apply this `PUT` if the resource is still at the version I read," and a `412` says "it isn't — re-read."

Its companion is **`428 Precondition Required`** (RFC 6585): the server *refuses to process a write at all* unless the client sends a precondition. This closes the lost-update hole proactively. If your `PUT /orders/{id}` requires an `If-Match`, a client that sends a bare `PUT` with no `If-Match` gets:

```http
HTTP/1.1 428 Precondition Required
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/precondition-required",
  "title": "This request requires an If-Match header",
  "status": 428,
  "detail": "Provide the ETag you last read so we can detect concurrent edits." }
```

`428` is the server saying "I will not let you blind-write; give me a precondition so I can protect you from clobbering someone else's change." It turns optimistic concurrency from a convention the client *might* follow into a contract the server *enforces*.

### 415 Unsupported Media Type

**`415 Unsupported Media Type`** means the request body is in a format the endpoint does not accept — the `Content-Type` is wrong. A client that `POST`s `Content-Type: application/xml` to an endpoint that only takes JSON gets a `415`. It is distinct from `406 Not Acceptable` (the server cannot produce a representation matching the client's `Accept` header — a *response*-format problem). `415` is about what you *sent*; `406` is about what you asked to *receive*. We cover content negotiation — `Accept`, `Content-Type`, vendor media types — in the dedicated negotiation post; here, `415` is just one more honest `4xx` that points the client at the specific fix (change your `Content-Type`).

### 402 Payment Required: the special case worth knowing

A payments API meets one code most others never touch: **`402 Payment Required`**. For years it was a reserved, "future use" code that almost nobody returned, and the spec still describes it loosely. In practice, the payments industry has converged on using it for the specific case of a *declined charge* — the request was well-formed, authenticated, and authorized, but the card issuer said no (insufficient funds, a fraud hold, an expired card). This is genuinely different from a `422`: the *data* was fine and the server *did* try to charge; an external authority refused. It is different from a `409`: there is no state conflict. And it is different from a `403`: the caller is fully permitted to attempt the charge; the *bank* declined it. Stripe, for example, uses `402` for card errors. The honest design returns `402` with a `problem+json` body whose `type` and a domain `decline_code` tell the client *why* the bank refused, so the UI can say "your card was declined — try another card" rather than a generic error:

```http
HTTP/1.1 402 Payment Required
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/card-declined",
  "title": "The card was declined by the issuer",
  "status": 402,
  "detail": "Issuer response: insufficient_funds.",
  "decline_code": "insufficient_funds" }
```

The subtlety: a declined card is *not* a server failure, so it is never a `5xx` — the system worked perfectly, the answer was just "no." And it is *not* automatically retryable in the way a `429` is; retrying the same card with the same balance will be declined identically. A `402` tells the client "this attempt failed for a reason the user must act on," which is exactly the recovery path a checkout flow needs. This is the value of a precise code in one sentence: the client's next action — *prompt for a different card* — is fully determined by the status line and one `decline_code`, with no guessing.

## 6. The 4xx gray areas, part three: 429 Too Many Requests and Retry-After

**`429 Too Many Requests`** (RFC 6585) is the one `4xx` that is explicitly *retryable* — but only after waiting. It means "you are correct, but you are sending too fast; slow down." It is a `4xx` (the client's behavior is the problem, the server is healthy) but unlike its siblings, retrying the identical request *will* eventually succeed once the client backs off. The contract that makes `429` usable is the **`Retry-After`** header, which tells the client *how long* to wait before retrying — either a number of seconds or an HTTP date:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30
RateLimit-Limit: 100
RateLimit-Remaining: 0
RateLimit-Reset: 30
Content-Type: application/problem+json

{ "type": "https://errors.shop.example/rate-limited",
  "title": "Rate limit exceeded",
  "status": 429,
  "detail": "You may send 100 requests per minute. Retry after 30 seconds." }
```

The math behind the limit is worth making explicit, because the `Retry-After` value should be *derivable* from it, not guessed. A common rate limiter is the **token bucket**: the bucket holds up to $B$ tokens, refills at rate $r$ tokens per second, and each request consumes one token. The server admits a request only if $\text{tokens} \ge 1$. When the bucket is empty, the time until the next token is available is $1/r$ seconds, and the time until the bucket is full again is $B/r$. So the honest `Retry-After` for a single retry is $\lceil 1/r \rceil$ — wait for one token. If the limit is 100 requests per minute, $r = 100/60 \approx 1.67$ tokens per second, so the next token arrives in about $0.6$ seconds; a server that wants the client to spread out might return a larger `Retry-After` (say, the time to the window reset). The point is that the number is *computable* from the limiter's parameters, and a good limiter exposes both the limit and the reset so clients can self-pace rather than hammer-and-back-off. We go deep on token-bucket versus sliding-window limiters, quota headers, and abuse protection in the [rate-limiting post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection).

#### Worked example: a 429 with Retry-After driving client backoff

A batch importer pushes 10,000 `POST /payments` calls as fast as it can. The merchant's plan allows 100 requests per minute. After the first ~100 requests drain the bucket, the 101st request gets `429` with `Retry-After: 30` (the limiter chose to send the client to the next window boundary rather than dribble single tokens). A *naive* client ignores `Retry-After` and retries immediately — it gets `429` again, and again, generating a storm of failed requests that does nothing but waste both sides' CPU and inflate the `4xx` graph. A *correct* client reads `Retry-After: 30`, sleeps 30 seconds, and resumes; the next batch of ~100 succeeds, and the importer settles into a steady 100/minute rhythm. The status code (`429`, not `503` and certainly not `200`) tells the client *this is your rate, not a server failure*; the `Retry-After` header tells it *exactly how long to wait*. Honor both and the importer finishes cleanly in ~100 minutes; ignore them and it never finishes while melting the rate-limit budget. Here is the client side, honoring the header:

```python
import time, requests

def post_with_backoff(session, url, body, idempotency_key):
    while True:
        resp = session.post(url, json=body,
                            headers={"Idempotency-Key": idempotency_key})
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "5"))
            time.sleep(wait)          # honor the server's pacing
            continue                  # same idempotency key -> safe replay
        return resp                   # 2xx or a non-retryable 4xx
```

Note the idempotency key carries across retries: because `429` is retryable, the client *must* replay the *same* logical request, and reusing the idempotency key guarantees the eventual success does not double-charge. That interplay — retryable status plus idempotency key — is why the two features are designed together; the idempotency-keys post in the series covers the server side.

## 7. The 5xx family: own your failures, and never leak a stack trace

`5xx` means "the request was fine; *I* failed." This is the family that drives availability SLOs, triggers retries, and — done wrong — leaks your internals to the world. Four codes carry almost all the weight.

![a branching graph showing a payments request entering a gateway and resolving to 500 own bug, 502 bad gateway, 503 unavailable, or 504 timeout depending on where it failed](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-7.png)

**`500 Internal Server Error`** is the catch-all for "an unexpected condition prevented me from fulfilling the request" — an unhandled exception, a null where there should not have been one, a bug. It is your fault, it is generic, and it should be *rare*. A spike of `500`s is a code defect, not a capacity problem.

**`502 Bad Gateway`** is specific to proxies and gateways: "I am a gateway, I forwarded your request to an upstream, and the upstream gave me an *invalid* response" (garbage, a connection reset, a malformed reply). In a fleet where your API gateway proxies to a payments microservice, a `502` means the gateway reached the service but got nonsense back — often the service crashed mid-response or returned an unparseable body.

**`503 Service Unavailable`** means "I am temporarily unable to handle the request" — overloaded, in maintenance, or shedding load on purpose. The defining feature of `503` is that it is *expected to be temporary*, and it is the other code (besides `429`) that takes a **`Retry-After`** header to tell the client when to come back. A load shedder under traffic spike returns `503 Retry-After: 5`. Importantly, `503` is *not* cacheable by default, which is correct — you do not want a CDN serving "we're down" after you have recovered.

**`504 Gateway Timeout`** is the gateway's "I forwarded your request upstream and the upstream *did not answer in time*." It is distinct from `502` (upstream answered, but with junk) and from `503` (the gateway itself is overloaded). A `504` on `POST /payments` is dangerous precisely because the client does not know whether the charge happened — the upstream might have completed the charge and just been too slow to respond. This is the exact scenario the idempotency-key design exists to make safe: the client retries with the same key, and the server returns the cached result of the charge that *did* go through, instead of charging again.

| Code | Who failed | Retryable? | `Retry-After`? | Cacheable by default? |
| --- | --- | --- | --- | --- |
| `500` | this server (bug) | maybe, with caution | no | no |
| `502` | upstream sent junk | yes, with backoff | no | no |
| `503` | this server, temporarily | yes — honor the header | **yes** | no |
| `504` | upstream too slow | yes — but beware double-effect | no | no |

#### Worked example: a 504 that may have charged the card

This is the most dangerous status code in a payments API, so it is worth walking slowly. A client sends `POST /payments`. The gateway forwards it to the payments service, which forwards it to the card processor. The card processor *does* charge the card — the money moves — but it is overloaded and takes 35 seconds to reply, longer than the gateway's 30-second upstream timeout. The gateway gives up waiting and returns `504 Gateway Timeout` to the client. From the client's seat, the request *failed*; from the bank's seat, the charge *succeeded*. This is the exactly-once illusion in its purest form: the network told the client one thing and the world is in another state.

A naive client treats `504` like any other failure and retries the `POST` with a *fresh* idempotency key (or no key at all). Now a second charge goes through, and the customer is billed twice for one order — the same double-charge from the introduction, arrived at by a different route. The correct design is two rules working together. First, the client must retry with the **same** idempotency key it used originally, because `504` is genuinely retryable (the upstream might have failed *before* charging, in which case the retry is the only way the charge ever happens). Second, the server must have stored the result of the first attempt against that key, so the retry — same key — returns the cached `201` of the charge that already went through instead of charging again:

```http
POST /payments HTTP/1.1
Idempotency-Key: 7f3b0c1e-9a2d-4c6f-8e10-2b9d4a1c5e7a
Content-Type: application/json

{ "order_id": "ord_8h2k", "amount": 24900, "currency": "USD" }
```

```http
HTTP/1.1 201 Created
Location: /payments/pay_5kq9
Idempotency-Replayed: true

{ "id": "pay_5kq9", "status": "succeeded", "amount": 24900 }
```

The `504` told the truth (*the gateway did time out*), the retry was legitimate (a timeout is retryable), and the idempotency key made the retry *safe* (one charge, not two). Three honest signals — the status code, the retry policy, the idempotency key — cooperating. Get any one wrong and you either lose the charge (gave up on a real `504`) or double it (retried with a new key). The full server-side mechanics of storing and replaying by idempotency key are the subject of the dedicated idempotency-keys post; the status-code lesson is that **`504` means "I do not know if it happened," and the only safe response is an idempotent retry.**

The cardinal sin of the `5xx` family is **leaking a stack trace**. A `500` that returns the raw exception, the file paths, the SQL query, the framework version, and the internal hostnames is a gift to an attacker (it maps your internals, your dependencies, your schema) and useless to a legitimate client (it cannot act on your Python traceback). The honest `500` is opaque to the outside and *correlated* to the inside:

```http
HTTP/1.1 500 Internal Server Error
Content-Type: application/problem+json
X-Request-Id: 9f2c1a44-7b3e-4d2a-9c1f-2e6b8a0d5c11

{ "type": "https://errors.shop.example/internal-error",
  "title": "An unexpected error occurred",
  "status": 500,
  "detail": "Reference 9f2c1a44 when contacting support.",
  "instance": "/payments" }
```

The body says nothing about *what* broke; it hands the client a **correlation ID** (here `X-Request-Id` and the `instance`/`detail` reference) that maps to a full stack trace *in your logs*, where it belongs. Support can look up `9f2c1a44` and see everything; the attacker on the wire learns nothing. This is the discipline: detailed internally, opaque externally, joined by an ID. We cover correlation IDs, RED metrics, and SLOs in the observability post later in the series.

A subtle `5xx` discipline: **map your dependencies' failures to the right code, and never reflect a dependency `4xx` as your `5xx` or vice versa.** If your payments service calls a card processor and the processor returns a `4xx` because *your* request to it was malformed, that is *your* bug — a `500` to your client (you sent the processor garbage), not a `4xx` (your client did nothing wrong). Conversely, if the processor is *down*, that is a `502`/`503`/`504` to your client (an upstream failure), not a `500` (it is not a bug in your code). The status code you emit must reflect *your* responsibility boundary, not blindly echo the layer below.

![a vertical stack from a got-an-error root branching into a client four-xx side and a server five-xx side, each splitting into two concrete code choices](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-6.png)

## 8. Putting it together: a decision procedure you can run in your head

You do not need to memorize forty codes. You need a short procedure that converges on the right one. Run it top to bottom:

1. **Did it succeed?** If yes → `2xx`. Created a resource? `201 + Location`. Async/queued? `202 + status_url`. Nothing to return? `204`. Otherwise `200`. Conditional `GET` and nothing changed? `304` (a `3xx`, but on the success path).
2. **Is the resource elsewhere?** If the URL moved → `3xx`. Must preserve the method on a write? `307`/`308`. Permanent move for a `GET`? `301`.
3. **Whose fault is the failure?** If the *client* sent something wrong → `4xx`. If the *server* failed → `5xx`. This is the single most important fork: it decides whether the client should fix-and-resend (`4xx`) or back-off-and-retry (`5xx`), and it decides whether your availability SLO takes a hit.
4. **Within `4xx`, ask the narrowing questions:**
   - Could the server even *parse* it? No → `400`. Yes, but the *data* is invalid → `422`.
   - Is the caller *authenticated*? No → `401 + WWW-Authenticate`. Yes, but not *permitted* → `403` (or `404` to hide existence).
   - Does the resource *exist*? No, and never will → `410`. No, or non-committal → `404`.
   - Wrong *method* on a real resource? → `405 + Allow`.
   - Conflicts with current *state*? → `409`. *Version* precondition failed? → `412`. Precondition *required and missing*? → `428`.
   - Wrong request *format*? → `415`.
   - Too *fast*? → `429 + Retry-After`.
5. **Within `5xx`, ask where it failed:** My own bug → `500` (opaque body + correlation ID). Upstream returned junk → `502`. I am overloaded/maintaining → `503 + Retry-After`. Upstream too slow → `504`.
6. **Always pair a `4xx`/`5xx` with a `problem+json` body** — `type`, `title`, `status`, `detail`, optional `instance`, plus any domain extensions like a per-field `errors` array.

![a four-row matrix mapping each status family to its meaning, the client action, and whether an automatic retry is safe](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-8.png)

That procedure, plus the rule "never lie," gets you the right code in nearly every case. The remaining judgment calls — `400` vs `422`, `403` vs `404` for hiding existence — are the ones where reasonable teams differ, and the deciding factor is *which lie hurts the client less*: a `400` that should have been a `422` still tells the truth about fault and family; a `200` that should have been either tells a lie that breaks retries and caches.

## 9. Consequences: a before→after of the lying-200 pattern

It is worth walking the exact failure chain of the cardinal sin, because the cost is rarely visible at design time and brutal at 2 a.m.

![a before and after comparison contrasting a 200 with ok false body against an honest 422 carrying a problem json envelope](/imgs/blogs/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx-5.png)

**Before — `200` with an error body.** The refund service returns `200 OK` with `{"success": false, "reason": "insufficient_funds"}` for failed refunds. Here is who that breaks, in order:

- **The retry layer.** The client's HTTP retry middleware sees `200` and stops. A transient failure that *would* have succeeded on retry (a momentary balance lag) never gets retried, because the status said "done."
- **The cache.** `200` is cacheable by default. A CDN or shared cache may store the failure and serve it to the next caller, so a *successful* refund attempt by a different caller sees a stale "insufficient funds."
- **The metrics.** The error-rate dashboard counts non-`2xx`. The flood of failures is a flat green line. The SLO says 100% availability while half of refunds fail.
- **The alerting.** The page that fires on `5xx` rate (or on `4xx` spikes) never fires. The first signal is the support queue.
- **The client code.** Every consumer must now write `if resp.status_code == 200 and not resp.json()["success"]:` — duplicating the failure-detection logic the status code was supposed to centralize, and every consumer that forgets the second clause silently treats failures as successes.

**After — honest codes.** The same service returns `422` for an unprocessable refund (bad data), `409` for an already-refunded payment (state conflict), `429` when the caller is too fast, and `503 + Retry-After` when the downstream balance service is briefly down. Now: the retry layer retries the `503` and the `429` (correct) and does not retry the `422`/`409` (correct); the cache does not store the `503`; the error-rate graph shows the real failure rate; the `5xx` alert fires when the balance service degrades; and the client branches on the status family with zero body-parsing for the common decision. The *exact same failures* are now visible, actionable, and correctly retried — purely by telling the truth in the status line.

The measured cost of the lie, framed honestly: in the refund outage, the time-to-detection went from what should have been *minutes* (a `5xx` alert) to *hours* (a human noticing the support queue). The number of silently-stuck refunds scaled with that delay. The fix was not a new system — it was deleting the "always 200" rule and letting the codes be honest, after which the existing, untouched alerting caught the next incident in under five minutes.

## 10. Verifying the contract: testing that your codes are honest

A status code is a contract, and a contract you do not test is a contract you do not have. The good news is that status codes are unusually cheap to verify, because the assertion is a single integer plus a media type — you do not need to inspect the whole body to know whether the code is right. Here is how to lock the contract down at four levels: unit/integration tests, a `curl` smoke test, contract tests, and runtime monitoring.

**Assert the code *and* the companion header in integration tests.** The single most common status-code regression is dropping the spec-required header — shipping a `405` with no `Allow`, a `401` with no `WWW-Authenticate`, a `429` with no `Retry-After`. Make those assertions explicit so a refactor cannot silently delete them:

```python
def test_method_not_allowed_includes_allow_header(client):
    resp = client.delete("/payments/pay_5kq9")  # DELETE not supported
    assert resp.status_code == 405
    assert "Allow" in resp.headers          # the spec requires it
    assert "GET" in resp.headers["Allow"]
    assert resp.headers["Content-Type"] == "application/problem+json"

def test_rate_limit_includes_retry_after(client, exhaust_bucket):
    resp = client.post("/payments", json=valid_body)
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1   # a usable wait

def test_validation_failure_is_422_not_400(client):
    resp = client.post("/payments", json={"amount": -500, "currency": "ZZ"})
    assert resp.status_code == 422        # parsed fine, data invalid -> 422
    body = resp.json()
    assert body["status"] == 422          # problem+json mirrors the code
    assert any(e["field"] == "amount" for e in body["errors"])
```

Notice the last assertion in the `422` test: `body["status"] == 422`. The `problem+json` envelope carries a `status` member that *duplicates* the HTTP status line. That redundancy is deliberate and worth testing — if the two ever disagree (a `400` status line with a `body["status"]` of `422`), some caller will trust the wrong one, and you have reintroduced ambiguity. Assert they match.

**Keep a one-screen `curl` smoke test of the honesty rules.** Before every release, hit the cardinal cases and eyeball the status lines. The `-i` flag prints the status line and headers; `-s` quiets the progress meter:

```bash
# A malformed body must be 400, not 200
curl -si -X POST https://api.shop.example/payments \
  -H 'Content-Type: application/json' \
  -d '{ "amount": 24900, "currency": "USD"' | head -1   # expect: HTTP/1.1 400

# Valid-but-invalid data must be 422, not 400 and not 200
curl -si -X POST https://api.shop.example/payments \
  -H 'Content-Type: application/json' \
  -d '{ "order_id": "ord_8h2k", "amount": -1, "currency": "ZZ" }' | head -1

# A missing token must be 401 with a challenge
curl -si https://api.shop.example/payments/pay_5kq9 | grep -E 'HTTP|WWW-Authenticate'
```

**Lock the code-per-condition mapping with a contract test.** Consumer-driven contract testing (Pact and friends) lets a consumer declare "when I send *this* request, I expect *this* status and *this* body shape," and the provider's CI fails if it ever stops honoring that. Status codes are the cleanest thing to put in a contract because they are exact: the consumer pins `429` with a `Retry-After` header, and the provider can never quietly downgrade it to a `200`-with-error-body without breaking the build. We cover consumer-driven contracts and schema diffs in depth in the contract-testing post later in the series; the relevant point here is that the *status code is a first-class clause* of that contract, not an afterthought.

**Monitor the code distribution in production.** Tests catch the codes you thought to assert; production traffic exercises the ones you did not. Emit a metric tagged by status code (or at least by family) on every response, and watch two things: a sudden rise in `5xx` (an availability incident, by definition) and a sudden rise in a *specific* `4xx` like `409` or `422` (often a client bug, a bad deploy on a consumer, or — for `429` — an abuse pattern or a client that stopped honoring `Retry-After`). The single most valuable alert an API can have is "the `5xx` rate crossed the SLO threshold," and it only works if your `5xx` codes are *honest* — which loops back to the whole point of this post. A service that returns `200` for failures has a flat `5xx` graph and a useless alert. Honest codes are what make the monitoring layer worth building.

One more verification habit, cheap and high-value: **diff your OpenAPI spec's documented responses against what the service actually returns.** If the spec for `POST /payments` lists `201`, `400`, `422`, `409`, and `429`, but the service in production also emits a `503` under load that the spec never mentions, that undocumented code is a surprise the client never coded for. A spec-first workflow (covered in the OpenAPI post in this series) makes the documented set the source of truth, and a response-validation middleware or a replayed-traffic test catches drift. The contract is only as honest as the gap between what you *say* you return and what you *do* return.

## 11. Case studies: how the careful APIs do it

It helps to anchor these rules in shipping public APIs and the specifications that govern them. The details below are accurate to the public documentation; where a behavior is a general pattern rather than a documented guarantee, I say so.

**Stripe** is the canonical careful payments API, and its status-code use is deliberate. Stripe returns `402 Payment Required` for card errors (a declined card), `400` for invalid request errors, `401` for authentication failures, `404` for missing resources, `409` for conflicts, and `429` for rate limiting — and it pairs each with a structured error object carrying a `type`, a machine-readable `code`, and a human `message`, which is the same separation-of-concerns as `problem+json`. Stripe also leans hard on idempotency keys so that a client retrying after a `5xx` or a network timeout does not double-charge — exactly the `504`-is-dangerous scenario from section 7. The lesson: a payments API treats the status code and the structured error body as *one contract*, designed together.

**GitHub** demonstrates two patterns worth copying. First, its REST API returns **`422 Unprocessable Entity`** for validation failures (well-formed requests with invalid field values), with an `errors` array describing each failing field — the precise `400`-vs-`422` discipline from section 4. Second, GitHub returns **`404 Not Found` instead of `403`** for private resources a caller cannot access, so the API does not leak the existence of private repositories — the hide-existence pattern from section 5. GitHub's APIs also send rate-limit headers (`x-ratelimit-limit`, `x-ratelimit-remaining`, `x-ratelimit-reset`) alongside `429`-class responses so clients can self-pace.

**The specifications themselves** are the bedrock and worth reading directly. **RFC 9110 ("HTTP Semantics")** is the authoritative definition of every status code, the safe/idempotent/cacheable classifications, and the required headers (`WWW-Authenticate` on `401`, `Allow` on `405`, `Retry-After` on `503`/`429`). **RFC 6585** added `428 Precondition Required`, `429 Too Many Requests`, `431 Request Header Fields Too Large`, and `511 Network Authentication Required` — the "additional status codes" the original spec lacked. **RFC 9457 ("Problem Details for HTTP APIs")** standardizes the `application/problem+json` error body — `type`, `title`, `status`, `detail`, `instance` — and explicitly permits domain extensions. Together these three are the rulebook; this post is a guided tour of them, not a replacement.

**Google's API Improvement Proposals (AIP)** and the **Zalando** and **Microsoft REST API guidelines** all codify the same discipline at organizational scale: a fixed mapping from error conditions to status codes, a standard error envelope, and a prohibition on `200`-with-an-error-body. When an org with hundreds of services agrees on these rules in a style guide, the payoff is that a client written against one service's error handling works against all of them — which is the whole point of a *consistent* contract. We cover org-wide governance and style guides in the governance post later in the series.

## 12. When to reach for which code (and when not to)

Decisive recommendations, since every choice is a trade-off and the failure modes are concrete:

- **Do** use the full family — `201`, `202`, `204`, `304`, `409`, `422`, `429`, `503` — when the distinction changes the *client's next action*. The whole value of a precise code is that the client recovers correctly without parsing prose.
- **Don't** collapse everything into `200`/`400`/`500`. You are not simplifying the client; you are forcing it to re-derive from the body what the code should have told it for free, and you are blinding your own retries, caches, and alerts.
- **Don't ever** return `200` with an error body. This is the cardinal sin. If you remember one rule from this post, it is this one. It breaks retries (they stop), caches (they store the failure), and alerting (it never fires).
- **Don't** return `5xx` for a client error. A `422` masquerading as a `500` manufactures a fake availability incident, triggers pointless retries that will fail identically, and points the on-call engineer at the wrong system.
- **Do** pair *every* `4xx` and `5xx` with a `problem+json` body. The status code says *what family of thing went wrong*; the body says *which specific thing and how to fix it*. You need both.
- **Don't** over-engineer the `2xx` distinctions where they do not matter. If your `POST` returns the created resource and the client never uses `Location`, a `200` versus `201` debate is low-stakes; spend the energy on getting `4xx` honest, where the client's recovery actually diverges.
- **Use `404` over `403` to hide existence** when the ID space is guessable or the resource's existence is sensitive (private repos, other tenants' data). **Use `403`** when the caller is plainly entitled to know the thing exists but not to act on it.
- **Use `422` over `400`** when your framework can distinguish a parse failure from a validation failure; both are acceptable with a good `problem+json` body, but `422` is the more precise truth.
- **Always send the spec-required companion header**: `WWW-Authenticate` on `401`, `Allow` on `405`, `Retry-After` on `503` and `429`. Omitting them is a contract violation that leaves the client guessing.
- **Don't** leak a stack trace, SQL, file paths, or internal hostnames in a `5xx`. Return an opaque body plus a correlation ID; put the detail in your logs.

## Key takeaways

- **The status code is part of the contract, read mechanically by the client, every proxy, every cache, the load balancer, and your metrics — before anyone parses the body.** It must be true.
- **The first digit is a promise about responsibility and retryability**: `2xx` proceed, `3xx` look elsewhere, `4xx` you-fix-and-resend, `5xx` server-failed-back-off-and-retry. Branch on it before reading the body.
- **`2xx` has shades**: `201 + Location` for a create, `202 + status_url` for async, `204` for an empty success, `206` for ranges, and `304` is the success side of a conditional request that saves a body transfer.
- **Know the `4xx` gray pairs cold**: `400` (malformed) vs `422` (valid-but-rejected); `401` (unauthenticated, `+WWW-Authenticate`) vs `403` (unauthorized); `404` vs `403` (hide existence by returning `404`); `404` (maybe later) vs `410` (gone for good); `409` (state conflict) vs `412` (version precondition failed); `428` (precondition required); `415` (wrong format); `429 + Retry-After` (too fast).
- **`429` and `503` are the retryable failures that carry `Retry-After`** — honor the header and self-pace; reusing an idempotency key across the retry keeps it safe.
- **Within `5xx`, the code names the location of the failure**: `500` your bug, `502` junk from upstream, `503 + Retry-After` overloaded, `504` upstream too slow. Never leak a stack trace — return an opaque body with a correlation ID.
- **The cardinal sins**: `200` with an error body; `5xx` for a client error; collapsing the world into `200`/`400`/`500`. Each silently disables a layer that was relying on the truth.
- **Pair every `4xx`/`5xx` with `problem+json`** so both the program and the human on the other end know exactly what happened and what to do next.

This is the contract layer of the spine, the same question the whole series keeps asking: *what does the caller get to assume, and can I change this later without breaking them?* An honest status code is the smallest, most-read clause of that contract. Get it right and the layers above — error bodies, pagination, versioning — sit on solid ground. The next posts build directly on this: the [error-design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) designs the `problem+json` body these codes carry, and the [rate-limiting post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection) goes deep on the `429`/`Retry-After` machinery. When you are ready to assemble all of this into one review pass, the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) puts the status-code checklist alongside everything else.

## Further reading

- **RFC 9110 — HTTP Semantics** — the authoritative definition of every status code, the safe/idempotent/cacheable classifications, and the headers each code requires (`WWW-Authenticate`, `Allow`, `Retry-After`).
- **RFC 6585 — Additional HTTP Status Codes** — defines `428 Precondition Required`, `429 Too Many Requests`, `431`, and `511`.
- **RFC 9457 — Problem Details for HTTP APIs** — the `application/problem+json` error body (`type`, `title`, `status`, `detail`, `instance`) and its extension rules.
- **RFC 7232 — Conditional Requests** — `ETag`, `If-None-Match`, `If-Match`, and how `304` and `412` are produced.
- **Stripe API reference — Errors** — a careful, production payments API's status-code and error-object design, and its idempotency-key story.
- **GitHub REST API documentation** — `422` validation errors, the `404`-for-private-resources hide-existence pattern, and rate-limit headers.
- **Google AIP, Zalando RESTful API Guidelines, and the Microsoft REST API Guidelines** — organization-scale codifications of consistent status-code and error-envelope use.
- Within this series: the [intro hub](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the [HTTP semantics post](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers), the [error-design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract), the [rate-limiting post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection), and the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

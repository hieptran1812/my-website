---
title: "The API Design Playbook: A Review Checklist From First Endpoint to v2"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The one-page operational checklist for the whole series: a six-phase lifecycle review you run on any API change, a first-endpoint quickstart, the road from v1 to v2 without a breaking version, and the top ten mistakes with the post that fixes each."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "versioning",
    "review-checklist",
    "developer-experience",
    "contract",
    "playbook",
    "governance",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-1.png"
---

This is the last post in the series, and it is the one you will actually come back to. Everything in the other thirty-nine posts was building toward a single artifact: a checklist you run before you approve any API change. Not a vibe, not a "looks good to me," but a sequence of concrete questions with right answers, each one tracing back to a rule we derived earlier and the wire-level consequence of getting it wrong.

Here is the failure this post exists to prevent. A team ships a payments endpoint on a Friday. It works in the demo. Then over the next eighteen months: a network retry on a flaky mobile connection double-charges a customer because `POST /payments` had no idempotency key. An export job pages forever because the offset pagination skipped rows when new orders were inserted under the cursor. A "small cleanup" renames `amount_cents` to `amount` and 500s every mobile client that hard-coded the old field. A stolen access token still works six months later because the JWT had no expiry. A partner who integrated in year one is stranded when v2 ships with no `Sunset` header and no migration window. Every one of these is a known mistake with a known fix. The tragedy is that nobody ran the checklist.

The spine of this whole series has been one sentence: **an API is a contract and a product, not a function call.** You are designing for a caller you will never meet, on a timeline measured in years, across versions you cannot recall once shipped. So you design for three things, in this order — (1) a **correct, predictable contract**, (2) **safe evolution** so you can change it without breaking anyone, and (3) **developer experience** so consuming it is a pleasure — and you choose the paradigm (REST, gRPC, GraphQL, events) by **force, not fashion**, then secure and operate it like the public surface it is. The figure below is the shape of the rest of this post: six phases, run in order, design first and developer experience last.

![A vertical stack of the six API lifecycle phases running from design the contract at the top down to ship the developer experience at the bottom](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-1.png)

By the end of this post you will have a printable, six-phase review checklist; a "first endpoint" quickstart for day one; a walked-through review of a real endpoint against the checklist; an evolution scenario that takes an API toward a change *without* a breaking v2; and a top-ten list of mistakes, each linked to the post that fixes it. If you read only one post in this series, read [the intro on what an API actually is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and then read this one. The other thirty-eight are the depth behind each checkbox.

## How to use this playbook

The checklist is organized as a **lifecycle**, not an alphabet of features. You do not check "do we have pagination?" in isolation; you check it at the phase where it matters — Phase 1, designing the contract. The phases run in dependency order: you cannot meaningfully secure an endpoint (Phase 4) whose resource model (Phase 1) is wrong, and you cannot ship great docs (Phase 6) for a contract that is still changing shape under you.

There are two ways to read what follows. If you are designing a **new** endpoint, start at the "first endpoint quickstart" below and then walk Phases 1 through 6 in order. If you are **reviewing a change** to an existing API, skip to the printable checklist near the end and run it top to bottom — each gate links back to the phase that explains the *why*. Either way, the recurring question you ask at every single item is the question this whole series asks: **what does the caller get to assume, and can I change this later without breaking them?**

Why a checklist at all, rather than trusting senior judgment? Because API mistakes have a particular, nasty shape: they are cheap to make and expensive to discover. The renamed field passes code review, passes the tests (which were written against the new shape), passes the staging smoke test (which uses the new client), and ships green — and then the *old* mobile app, the one a third of your users have not updated, starts 500ing on a code path nobody tested. The double-charge does not happen in the demo; it happens three weeks later when a customer on a train loses signal mid-request and the retry fires. None of these bugs are visible at the moment you make them. The only defense is a *structural* one: a list of the failure modes that experience has catalogued, run mechanically against every change, so the question "did we handle the retry?" gets *asked* even when nobody happened to remember it. A checklist is not a replacement for expertise — it is how expertise survives contact with a Friday afternoon and a deadline.

There is a second reason. An API has many authors over many years. The engineer who designs the first endpoint is rarely the one who adds the fortieth, and almost never the one who finally has to deprecate something. A shared checklist is how a *team* — not a person — holds a consistent contract. It encodes "this is how we do APIs here" so the surface stays guessable as the roster turns over. That is the same argument the governance post makes for a style guide, scaled down to a single pull request.

Below is the map of the entire series, organized by the three goals. Use it as a table of contents: every checklist item links to the post that goes deep on it.

![A tree diagram showing the API equals contract and product root branching into three goals and then into the eight tracks of posts](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-2.png)

A note on vocabulary, because the checklist uses terms precisely. A **resource** is a thing the API exposes (an order, a payment). A **representation** is one serialized form of it (the JSON body). A method is **safe** if it has no side effects the caller is responsible for (GET, HEAD). A method is **idempotent** if making the same call twice has the same effect as making it once (PUT, DELETE; and POST *only* when you add an idempotency key). A change is **backward compatible** if an old client still works against the new server; **forward compatible** if a new client's payload still works against the old server. A **tolerant reader** is a client that ignores fields it does not recognize, which is what makes additive change safe. If any of those are fuzzy, the linked posts define them properly the first time they appear.

## The first endpoint quickstart: get these right on day one

Before the six phases, here is the irreducible minimum — the things you must get right on the very first endpoint, because they are expensive to retrofit and cheap to do correctly from the start. If you do nothing else from this playbook, do these.

**Model a resource, not a function.** Name the URL after a noun the domain owns — `/orders`, `/payments` — not after a verb like `/createOrder`. A collection (`/orders`) and an item (`/orders/{id}`) are the two shapes you need first. The full reasoning is in [resource modeling: turning a domain into nouns and URIs](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris), and the URI mechanics — when to use a sub-resource versus a query parameter — are in [choosing URIs: collections, sub-resources, path vs query](/blog/software-development/api-design/choosing-uris-collections-sub-resources-path-vs-query).

**Use the right method and mean it.** GET reads, POST creates, PUT replaces, PATCH partially updates, DELETE removes. The contract a caller relies on is the *semantics*: GET is safe and cacheable, PUT and DELETE are idempotent and safe to retry. Get this wrong — a GET with side effects, a POST that should have been idempotent — and you have lied to every cache, proxy, and retry library between you and the client. The full method-by-method treatment is in [methods and idempotency: GET, POST, PUT, PATCH, DELETE](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete), built on the foundation in [HTTP for API designers: methods, status codes, headers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers).

**Return an honest status code.** A `201 Created` for a new resource, `200 OK` for a read, `204 No Content` for a delete, `400`/`422` for a bad request, `404` for a missing resource, `409` for a conflict, `500` for *your* fault. Never, ever return `200 OK` with an error inside the body — it defeats every client's error handling. The full taxonomy is in [status codes that tell the truth: 2xx, 3xx, 4xx, 5xx](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).

**Make POST retry-safe with an idempotency key.** Any non-idempotent mutation that costs money or creates a resource needs an `Idempotency-Key` header so a retry after a timeout returns the original result instead of doing the work twice. This is the single most important thing to add to a payments endpoint, and the reason is covered in depth in [idempotency keys: safe retries and the exactly-once illusion](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).

**Design the error body once.** Pick `application/problem+json` (RFC 9457) — a body with `type`, `title`, `status`, `detail`, and `instance` — and use it for *every* error from day one. Retrofitting a consistent error shape after clients have parsed your ad-hoc errors is a breaking change. See [error design: a machine-readable, human-friendly contract](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract).

**Bound your collections.** Never return an unbounded list. Pick a default and maximum page size on day one, because `GET /orders` over a 50-million-row table without a limit is a denial-of-service waiting to happen. Cursor or keyset pagination is the right default; the trade-offs are in [pagination: offset, cursor, and keyset trade-offs at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale).

Here is the first endpoint, done right, as a single wire transcript. This is the artifact the quickstart produces:

```http
POST /v1/payments HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 7f9c2a1e-3b6d-4f0a-9c11-2e8d4a5b6c7d

{
  "order_id": "ord_8Kdf02",
  "amount_cents": 4999,
  "currency": "USD"
}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /v1/payments/pay_19Xz7c
ETag: "a1b2c3"

{
  "id": "pay_19Xz7c",
  "object": "payment",
  "order_id": "ord_8Kdf02",
  "amount_cents": 4999,
  "currency": "USD",
  "status": "succeeded",
  "created_at": "2026-06-20T10:14:02Z"
}
```

Notice what is already correct: a bearer token for authentication, an idempotency key for safe retries, an honest `201` with a `Location` header pointing at the new resource, an `ETag` for caching, snake_case field names chosen consistently, an explicit `currency`, money expressed in integer minor units (`amount_cents`) to avoid floating-point rounding, and a `created_at` in ISO-8601 UTC. That single response embodies a dozen checklist items. The rest of this post explains each one and shows you how to spot the version that gets it wrong.

Two of those choices are worth dwelling on, because they are the ones teams most often skip and most often regret. **Money in integer minor units.** A `4999` instead of `49.99` is not pedantry — IEEE-754 floating point cannot represent `0.10` exactly, so a client that does `total += line_item.amount` over a few hundred floats accumulates rounding error, and in a financial API rounding error is a reconciliation incident. Sending integer cents (or the smallest unit of the currency, which is not always two decimal places — the Japanese yen has zero, the Tunisian dinar has three) plus an explicit `currency` code means the client never has to guess the scale. **The `Idempotency-Key` as a client-supplied UUID.** The key is generated by the *client*, once, before the first attempt, and reused on every retry of *that* logical operation. The server stores the key with the result of the first successful processing; a retry with the same key returns the stored result instead of re-running the work. That is the whole mechanism that turns an at-least-once network (which is the only kind of network you have) into an effectively-once *effect*. The full derivation — and why "exactly-once delivery" is an illusion while "exactly-once effect" is achievable — is in [idempotency keys: safe retries and the exactly-once illusion](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).

## Phase 1 — Design the contract

This is where most of the durability is won or lost, because the contract is the part you cannot change later without breaking someone. Everything here answers: *is the contract correct, predictable, and honest?*

### 1.1 Resource and RPC model

Start by deciding whether your unit of design is a **resource** (a noun you do CRUD on) or a **procedure** (a verb that does a thing). Most of a commerce platform is resources — orders, payments, refunds — but some operations are genuinely procedural ("capture this authorization," "refund this charge") and forcing them into pure REST produces awkward sub-resources. The decision is covered in [RPC vs REST: when a procedure beats a resource](/blog/software-development/api-design/rpc-vs-rest-when-a-procedure-beats-a-resource); for the resource side, [resource modeling](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) is the deep dive.

**Review check:** Is each URL a noun the domain owns? Are relationships modeled as sub-resources (`/orders/{id}/payments`) only where the child cannot exist without the parent? Is anything genuinely procedural modeled honestly as an action rather than tortured into a fake resource?

### 1.2 URIs

A URI is a long-lived public identifier. Once a client bookmarks `/v1/orders/{id}`, you own that shape forever. Decide collection-vs-item structure, where the line between a path segment and a query parameter falls (identity goes in the path, filters go in the query), and a consistent pluralization and casing convention. The mechanics are in [choosing URIs](/blog/software-development/api-design/choosing-uris-collections-sub-resources-path-vs-query).

**Review check:** Plural collection nouns? Identity in the path, filters in the query? No verbs in the path (except for honest actions)? Stable — no element that will need to change, like a tenant name that can be renamed?

### 1.3 Methods and idempotency

The method is a promise to every intermediary about what the call does. The rule worth memorizing, derived from HTTP semantics: **safe methods (GET, HEAD) have no caller-visible side effects; idempotent methods (GET, HEAD, PUT, DELETE) can be repeated with the same effect; POST is neither, which is exactly why it needs an idempotency key to become retry-safe.** Why is PUT idempotent and POST not? Because PUT *names the target* — `PUT /orders/123` sets order 123 to a known state, and doing it twice lands in the same state — whereas POST asks the server to *create something new each time*, so a blind retry creates two. This is in [methods and idempotency](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete).

**Review check:** Does each method match its semantics? Is every GET truly side-effect-free? Is every money-moving or resource-creating POST protected by an `Idempotency-Key`? Is PATCH using a defined format (JSON Merge Patch or JSON Patch) rather than ad-hoc partial bodies?

#### Worked example: a retry that does not double-charge

A client sends a payment, the server processes it successfully, but the `201` response is lost on the way back — a dropped connection on a flaky link. The client, having received nothing, retries. Here is the second request, byte-for-byte identical to the first, *including the same idempotency key*:

```http
POST /v1/payments HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 7f9c2a1e-3b6d-4f0a-9c11-2e8d4a5b6c7d

{ "order_id": "ord_8Kdf02", "amount_cents": 4999, "currency": "USD" }
```

The server looks up the key, finds it already processed, and returns the *stored* result — the same `pay_19Xz7c`, the same `201`, no second charge:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Idempotent-Replayed: true
Location: /v1/payments/pay_19Xz7c

{ "id": "pay_19Xz7c", "status": "succeeded", "amount_cents": 4999 }
```

The customer is charged once. Without the key, the second request would have created `pay_2A...` and charged `\$49.99` a second time — the exact incident that pages a payments team at 2 a.m. The reviewer's job is one question: *what does this endpoint do when the client retries on a timeout?* If the answer is "double the work," the box is unchecked.

### 1.4 Status codes

A status code is the first thing a client's error handling branches on, so it must tell the truth. The most common sin is the `200`-with-an-error-body, which forces every client to parse the body to discover failure and breaks retry logic that keys off the status line. Choose the most specific honest code: `409` for a conflict, `422` for a semantically invalid but well-formed body, `412` for a failed precondition, `429` for rate limiting, `404` versus `403` chosen deliberately (do you reveal the resource exists?). See [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).

**Review check:** Does the status code alone tell the client what happened? No `200` with `"error"` in the body? `4xx` for the caller's fault, `5xx` for yours, and never the reverse? `429` carries `Retry-After`?

### 1.5 Payloads and naming

Pick one casing (snake_case or camelCase) and one envelope policy (bare object vs `{ "data": ... }` wrapper) and apply them everywhere. Inconsistency here is the slow tax: every endpoint that breaks the pattern is a special case a client integrator has to remember. Express money in integer minor units, timestamps in ISO-8601 UTC, enums as documented strings. The full treatment is in [designing request and response bodies: shape and naming](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming).

**Review check:** Consistent casing across every field? One envelope convention? Money as integer minor units with explicit currency? Timestamps ISO-8601 UTC? No leaking internal column names or database IDs you do not want to commit to forever?

### 1.6 Errors

Standardize on `application/problem+json` so a client can branch on a stable, machine-readable `type` URI and show a human the `detail`. The error contract is as much a contract as the success contract — clients code against your error `type`s. See [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract).

**Review check:** Every error returns `problem+json` with `type`, `title`, `status`, `detail`? Validation errors enumerate the offending fields? Error `type`s are stable URIs, not free-text that changes between releases?

### 1.7 Pagination and filtering

Bound every collection. Default to cursor or keyset pagination, not offset, because offset pagination over a table being written to **skips and duplicates rows** — the window shifts under you. The cost argument is concrete: an offset query at page $n$ is $O(n)$ because the database must scan and discard the first $n \times \text{limit}$ rows, while a keyset query that seeks on an indexed key is $O(1)$ per page regardless of depth. Pagination is in [pagination: offset, cursor, and keyset](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale); filtering, sorting, and field selection are in [filtering, sorting, and sparse fieldsets without reinventing SQL](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql).

**Review check:** Default and max page size set? Cursor or keyset (not offset) for large or hot collections? Stable, deterministic ordering? Filters and sorts validated against an allowlist, not passed through to the database? Sparse fieldsets bounded?

#### Worked example: the export that skipped rows

A reporting client exports all orders, newest first, in pages of 100 using offset pagination: `GET /v1/orders?sort=-created_at&limit=100&offset=0`, then `offset=100`, and so on. Between fetching page 1 and page 2, 30 new orders are created. Now the database, sorting newest-first, has 30 fresh rows at the *top*. The client's `offset=100` skips the first 100 rows of the *new* ordering — but 30 of those are the brand-new orders, which means rows that were on page 1's boundary have been shoved down past the offset and are **silently skipped**, while some rows the client already saw are **fetched again**. The export is both incomplete and duplicated, and nothing errors — the bug is invisible until someone reconciles the report against the source.

Cursor pagination fixes this by paging from a *stable position in the data*, not a moving numeric offset. The first response returns an opaque cursor; the next request passes it back:

```http
GET /v1/orders?sort=-created_at&limit=100 HTTP/1.1

HTTP/1.1 200 OK
{
  "data": [ ... 100 orders ... ],
  "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNi0wNi0yMFQwOTowMFoiLCJpZCI6Im9yZF84S2RmMDIifQ"
}
```

```http
GET /v1/orders?sort=-created_at&limit=100&cursor=eyJjcmVhdGVkX2F0... HTTP/1.1
```

The cursor encodes the `(created_at, id)` of the last row seen, so the next page is "everything older than *this exact row*," which is stable no matter how many new orders arrive at the top. The cost argument seals it: the cursor query is a `WHERE (created_at, id) < (?, ?) ORDER BY ... LIMIT 100` that seeks straight to the position on an index — $O(\log n)$ to find the start, $O(1)$ in the page depth — whereas `OFFSET 1000000` forces the database to scan and discard a million rows first, $O(n)$ and getting slower every page. For a 50-million-row table the difference is milliseconds versus a query that times out. The reviewer's question: *what does this paginate over when the collection has 50 million rows and is being written to?*

### 1.8 Content negotiation and the rest of the contract surface

Decide your media types and whether you support content negotiation (`Accept`/`Content-Type`) — covered in [content negotiation: media types and representations](/blog/software-development/api-design/content-negotiation-media-types-and-representations). Decide honestly whether you want hypermedia links in responses, and skip them when no client will follow them — the pragmatic take is in [HATEOAS in the real world: hypermedia links and when to skip](/blog/software-development/api-design/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip). Calibrate how "RESTful" you actually need to be using [the Richardson maturity model and what RESTful means](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means). And design from the consumer's side throughout — least surprise, good defaults, discoverability — which is the whole argument of [designing for the caller: developer experience as a goal](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal). Finally, long-running operations get the `202 Accepted` + status-resource pattern from [long-running operations: async jobs, polling, and callbacks](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks).

A word on each of these in practice. Content negotiation lets one URL serve multiple representations — `Accept: application/json` versus a vendor media type like `application/vnd.example.v2+json` — which is also one of the versioning strategies you will weigh in Phase 3. Hypermedia (HATEOAS) means embedding links to related actions in the response so a client *discovers* what it can do next rather than hard-coding URLs; it pays off for workflow-heavy APIs with many states and is dead weight for a simple CRUD resource, which is exactly the "and when to skip" judgment the linked post makes. The Richardson maturity model gives you the vocabulary to say *where* on the REST spectrum you are sitting — level 2 (proper resources, methods, and status codes) is where the vast majority of good HTTP APIs live, and reaching for level 3 (hypermedia) should be a deliberate choice, not a checkbox you tick to feel pure. Long-running operations — a refund that takes minutes to settle, a bulk export — should return `202 Accepted` immediately with a `Location` pointing at a status resource the client polls, rather than holding a connection open for minutes and timing out behind every proxy on the path.

**Review check:** Media types declared? Hypermedia only where a client benefits? The endpoint sits at a maturity level you chose on purpose? Long-running work returns `202` with a pollable status resource rather than blocking?

#### Worked example: a full design review of one endpoint

Let us run Phase 1 on a real candidate. A team proposes this for adding a refund:

```http
POST /v1/orders/createRefund?order=ord_8Kdf02&amt=10.00 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "ok": false, "msg": "order already fully refunded" }
```

Now walk the checklist. **1.1/1.2 URI:** `createRefund` is a verb, and the data is in the query string instead of a body — fail. A refund is a resource; this should be `POST /v1/refunds` with the order reference in the body, or `POST /v1/orders/{id}/refunds`. **1.3 Method/idempotency:** it is a money-moving POST with no `Idempotency-Key`, so a retry after a timeout double-refunds — fail. **1.4 Status:** it returns `200 OK` for a *failure* — fail; an already-refunded order is a `409 Conflict` (or `422`). **1.5 Payload:** the amount is `10.00`, a float, inviting rounding errors — fail; use `amount_cents: 1000`. The keys `ok` and `msg` match no convention used elsewhere — fail. **1.6 Errors:** the error is an ad-hoc shape, not `problem+json` — fail.

Here is the same operation after the review:

```http
POST /v1/refunds HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 9b2e6f10-7c44-4d8a-bb02-1f3e5a7c9d40

{
  "payment_id": "pay_19Xz7c",
  "amount_cents": 1000,
  "reason": "requested_by_customer"
}
```

```http
HTTP/1.1 409 Conflict
Content-Type: application/problem+json

{
  "type": "https://api.example.com/problems/already-refunded",
  "title": "Order already fully refunded",
  "status": 409,
  "detail": "Payment pay_19Xz7c has no remaining refundable balance.",
  "instance": "/v1/refunds"
}
```

Every "fail" above is now a "pass." The contrast is the whole point of the checklist: the same business operation, reviewed, becomes a contract a client can trust and retry. The figure below is that contrast in one frame.

![A before and after comparison contrasting an unreviewed endpoint that returns 200 with an error body against a reviewed endpoint that returns honest status codes and uses idempotency keys](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-4.png)

## Phase 2 — Choose the paradigm by force

You have a correct contract shape; now decide *how* it is delivered. The mistake to avoid is choosing the paradigm by fashion ("everyone uses GraphQL now") instead of by the forces on your system. The decision framework is the whole of [choosing a paradigm: REST vs gRPC vs GraphQL by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force), and it composes with the distributed-systems view in the system-design post on [API design across REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql).

The honest short version of the forces:

- **REST over HTTP/JSON** is the default for a public, resource-shaped API consumed by many unknown clients. It gets you HTTP caching, ubiquitous tooling, and a low barrier to entry. The Richardson-model and HTTP foundations from Phase 1 are exactly the REST design surface.
- **gRPC with Protocol Buffers** is the choice for high-throughput, low-latency, internal service-to-service calls where you control both ends and want a strongly-typed `.proto` contract with code generation and streaming. The deep dive — `.proto` service definitions, the four streaming modes, deadlines, status codes — is [gRPC and Protocol Buffers: contracts, codegen, and streaming](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming).
- **GraphQL** is the choice when many different clients need different *shapes* of the same graph of data and over-fetching from REST is painful — at the cost of the N+1 trap, hard HTTP caching, and a heavier server. It is rarely worth it for simple CRUD with a single client. See [GraphQL: the query language, schema, and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap).
- **Events and async** — webhooks, pub/sub, AsyncAPI — are the choice when the producer should not wait for the consumer and delivery is the point, not request/response. See [event-driven and async APIs: webhooks, pub/sub, and AsyncAPI](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi), which leans on broker internals like [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).
- **Streaming over a single connection** — Server-Sent Events, WebSocket, or gRPC server-streaming — is for pushing a continuous flow to the client. The trade-offs (and backpressure) are in [streaming APIs: SSE, WebSockets, and server streaming](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming).

Here is the decision compressed to a table — the force that should drive each choice, and the price you pay for it:

| Paradigm | Reach for it when | The price you pay | Force that decides |
| --- | --- | --- | --- |
| REST / JSON | Public, resource-shaped, many unknown clients | Over/under-fetching; chatty for graphs | Caller diversity + HTTP caching |
| gRPC / protobuf | Internal, high-throughput, low-latency, you own both ends | No browser-native support; opaque on the wire | Latency budget + typed contract |
| GraphQL | Many clients need different shapes of one graph | N+1 trap; weak HTTP caching; heavier server | Payload flexibility |
| Events / webhooks | Producer should not wait on consumer | Delivery, ordering, replay are now your problem | Decoupling + push not pull |
| Streaming (SSE/WS) | Continuous push over one connection | Connection state; backpressure | A live flow, not request/response |

A real system mixes these by force: REST for the public commerce surface, gRPC between internal payment and ledger services, webhooks to notify partners of a `refund.succeeded` event, and SSE to push live order status to a dashboard. Each surface is justified on its own — that is choosing by force, not picking one paradigm and bending everything to fit it.

**Review check:** Is the paradigm chosen by a named force (caller diversity, latency budget, payload flexibility, push vs pull) rather than by fashion? If GraphQL, is the N+1 trap addressed with a dataloader? If gRPC, are deadlines set on every call? If webhooks, are they signed and retried? If you are mixing paradigms, is each surface justified on its own?

## Phase 3 — Make it safe to change

This is the phase that separates an API that lasts from one that calcifies. The governing rule, derived from the **robustness principle** (be conservative in what you send, liberal in what you accept) and the **tolerant reader** pattern: **adding an optional response field is safe; adding a required request field, removing or renaming any field, narrowing a type, or changing the meaning of a value is breaking.** Why? Because an old client *ignores* the new optional field it does not know about (it is a tolerant reader), but it *cannot supply* a new required field and it *will* break when a field it relied on disappears or changes meaning. This asymmetry is the entire foundation of evolving without versioning, and it is in [backward and forward compatibility: the rules of safe change](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change).

The compatibility rules drive four sub-checks:

**3.1 Versioning strategy.** Decide URI versioning (`/v1`), header/media-type versioning, or — the often-correct answer — *no new version at all* because additive change covers you. Versioning in the URI fragments your surface and tempts you into a big-bang v2; prefer additive change until a true break forces your hand. The strategies and their trade-offs are in [versioning strategies: URI, header, media type, and not versioning](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning).

**3.2 Deprecation and sunset.** When you *must* retire something, do it humanely: send the `Deprecation` header, set a `Sunset` date far enough out (commonly 6–18 months for a public API), communicate it, and track who is still calling the old surface. The mechanics are in [deprecation and sunset: retiring an API humanely](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely).

**3.3 Contract tests.** Catch breaking changes *before* they ship by running consumer-driven contract tests and a schema diff in CI — `buf breaking` for protobuf, `oasdiff` for OpenAPI, Pact for consumer-driven contracts. A breaking change should fail the build, not page you at 2 a.m. See [contract testing: consumer-driven contracts and schema diffs](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs), which connects to [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale).

**3.4 Schema evolution.** When a field genuinely must change, use the **expand/contract** (parallel-change) pattern: add the new field, dual-write both, migrate clients, then remove the old field after the sunset window. Never rename in place. The field-lifecycle mechanics — adding, removing, renaming safely — are in [schema evolution: adding, removing, renaming fields safely](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely), and the underlying database move is [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations).

**Review check:** Is this change additive (safe) or does it remove/rename/require (breaking)? If breaking, can expand/contract make it non-breaking instead? Does the schema diff pass in CI? Do consumer contract tests pass? If a deprecation is involved, is there a `Deprecation` header, a `Sunset` date, and a comms plan?

#### Worked example: taking the API toward a change without a breaking v2

Product wants to support multi-currency refunds and rename the confusingly-named `reason` field to `reason_code`. The naive plan is "ship v2." Let us evolve without one.

**Step 1 — classify each change.** Multi-currency is *additive*: add an optional `currency` field to the refund request and an optional `currency` to the response. Old clients omit it and get the default currency; new clients send it. This is backward *and* forward compatible — no break. The rename of `reason` to `reason_code` is *breaking* if done in place, because old clients send `reason` and would suddenly get a `422`.

**Step 2 — apply expand/contract to the rename.** Expand: the server now accepts *both* `reason` and `reason_code` on input and returns *both* on output (dual-write). Document `reason` as deprecated. Old clients keep working; new clients adopt `reason_code`.

```json
{
  "payment_id": "pay_19Xz7c",
  "amount_cents": 1000,
  "currency": "USD",
  "reason": "requested_by_customer",
  "reason_code": "requested_by_customer"
}
```

**Step 3 — drive migration with data, not guesswork.** Instrument which field clients send. When telemetry shows the old `reason` field has dropped below your threshold of traffic, send the `Deprecation` header on responses that still receive the old field and set a `Sunset` date.

```http
HTTP/1.1 200 OK
Content-Type: application/json
Deprecation: true
Sunset: Wed, 30 Sep 2026 00:00:00 GMT
Link: <https://api.example.com/changelog#reason-code>; rel="deprecation"
```

**Step 4 — contract.** After the sunset date, with telemetry confirming near-zero old-field traffic, remove `reason` from the input contract. The schema diff flags the removal; you approve it deliberately because the migration window has closed.

The discipline that makes all four steps safe is the contract test running in CI on every commit. A consumer-driven contract test records what each client *actually depends on* and fails the build if a change would break that expectation. It is a recorded interaction the provider must keep honoring — here is the shape of one such expectation, expressed as a Pact-style fragment:

```json
{
  "consumer": "mobile-app",
  "provider": "payments-api",
  "interactions": [
    {
      "description": "create a refund",
      "request": { "method": "POST", "path": "/v1/refunds" },
      "response": {
        "status": 201,
        "body": { "id": "matching:type", "reason_code": "matching:type" }
      }
    }
  ]
}
```

Because the mobile app's contract asserts it reads `reason_code`, the build fails the instant someone tries to remove `reason_code` — but it says *nothing* about `reason`, so removing the deprecated field after the sunset passes cleanly. The contract test is the machine that turns "we think everyone migrated" into "we can prove it." This is exactly the workflow in [contract testing](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs); a schema-diff linter like `oasdiff` complements it by catching breaking changes at the spec level before a single test runs.

The result: a multi-currency, cleanly-named refund API and **no v2**. The version number never changed, no client was stranded, and the only "breaking" step happened after everyone had migrated. This is the road most changes should take. The figure below is that road drawn out — years of additive change, with a true v2 only as the rare last resort.

![A timeline from a single honest endpoint on day one through additive changes over months and years to a forced breaking v2 with a sunset window only at the far right](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-7.png)

## Phase 4 — Secure it like the public surface it is

An API is reachable by anyone who can reach the network, so treat it as hostile-input territory by default. This phase answers: *who is calling, what may they do, how often, and is their input safe?* The five sub-checks map to the five security posts.

**4.1 Authentication — who are you?** Choose among API keys, sessions, JWTs, and mTLS by client type. The rule: bearer tokens (JWTs) for first-party and OAuth clients, mTLS for service-to-service in a fleet, API keys only for low-stakes server-to-server. Every token must have an expiry — a token that lives forever is a token that, once stolen, works forever. See [authentication: API keys, sessions, JWT, and mTLS](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls), and for the fleet case [service-to-service security: mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

**4.2 Authorization — what may you do?** Authentication tells you the caller's identity; authorization decides what that identity may touch. The most dangerous and most common API vulnerability is **broken object-level authorization (BOLA)**: the server checks that you are logged in but not that *this* order belongs to *you*, so `GET /v1/orders/{any_id}` leaks every customer's data. Every resource access must check ownership, not just authentication. Scopes, roles, and resource-level permissions are in [authorization: scopes, roles, and resource-level permissions](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions).

**4.3 OAuth 2.0 and OIDC.** For third-party access, use the right OAuth flow per client: authorization-code-with-PKCE for apps acting on behalf of a user, client-credentials for machine-to-machine. Do not invent your own delegation scheme. The designer's view of grant types, tokens, and OIDC is in [OAuth 2.0 and OpenID Connect for API designers](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers).

**4.4 Rate limiting and abuse protection.** Bound how often anyone can call you, or one buggy client (or one attacker) takes the API down for everyone. The token-bucket model is simple and provably fair: a bucket holds up to $B$ tokens, refills at $r$ tokens per second, and a request is allowed only if $tokens \ge 1$, consuming one. The long-run admitted rate is exactly $r$ with bursts up to $B$. Concretely, with $r = 10$ tokens/sec and $B = 20$: a client that has been idle has a full bucket and can fire 20 requests instantly (the burst), then is throttled to 10/sec as the bucket refills. Over any long window the average admitted rate cannot exceed $r$ — that is the *provable fairness* of the bucket, and why it beats a naive fixed-window counter that lets a client send $2 \times$ the limit by straddling the window boundary. When you reject, return `429 Too Many Requests` with a `Retry-After` and the standard rate-limit headers so the client backs off correctly:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 2
RateLimit-Limit: 10
RateLimit-Remaining: 0
RateLimit-Reset: 2
Content-Type: application/problem+json

{
  "type": "https://api.example.com/problems/rate-limited",
  "title": "Too many requests",
  "status": 429,
  "detail": "Rate limit of 10 requests/second exceeded. Retry after 2 seconds."
}
```

A well-behaved client (or a generated SDK) reads `Retry-After` and waits exactly that long before retrying, instead of hammering you in a tight loop and making the overload worse. See [rate limiting, quotas, and abuse protection](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection).

**4.5 Input validation and the OWASP API Top 10.** Validate every input against a schema, allowlist accepted fields to prevent **mass assignment** (a client setting `is_admin: true` on a user-update because you bound the whole object), encode output, and walk the OWASP API Security Top 10 as a checklist of its own. See [input validation, output encoding, and the OWASP API Top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10).

Here is what the secured request path looks like — gateway, auth, validation, handler, store — with every gate able to reject before the handler runs:

![A branching graph of a request flowing through gateway then auth then validation to the handler and store with each gate able to reject to a 401 403 422 or 429 path](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-3.png)

A rejection at any gate is honest and specific. An expired token is a `401`; a valid token without the right scope or without ownership of the resource is a `403`; a malformed body is a `422`; too many requests is a `429`. Here is the authorization check that prevents the BOLA leak — the single most important line of security code in the whole API:

```python
@app.get("/v1/orders/{order_id}")
def get_order(order_id: str, caller=Depends(authenticated)):
    order = orders.find(order_id)
    if order is None:
        raise problem(404, "not-found", "Order not found.")
    # Authorization, not just authentication: does THIS caller own it?
    if order.customer_id != caller.customer_id and "orders:read_all" not in caller.scopes:
        raise problem(403, "forbidden", "You do not have access to this order.")
    return order
```

**Review check:** Every token expires? Every resource access checks ownership (no BOLA)? The right OAuth flow per client type? Rate limits enforced with `429` + `Retry-After`? Inputs validated and field-allowlisted (no mass assignment)? The OWASP API Top 10 walked?

## Phase 5 — Operate it

A correct, evolvable, secure API still fails its users if it is slow, opaque, or falls over under load. This phase answers: *can you run it, see it, and keep it fast?*

**5.1 Gateway and BFF.** Put a gateway in front to centralize routing, authentication, and rate limiting, and consider a backend-for-frontend (BFF) when different client types (mobile, web, partner) need tailored, aggregated responses. The responsibilities and the BFF pattern are in [API gateways: routing, auth, rate limiting, and the BFF pattern](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern), which connects to the fleet view in [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) and [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing).

**5.2 Caching.** Use HTTP caching deliberately: `ETag` plus conditional requests (`If-None-Match` → `304 Not Modified`) to avoid re-sending unchanged bodies, `Cache-Control` to let CDNs and clients cache safely, and a real invalidation strategy so stale data does not linger. A `304` is a few hundred bytes instead of a full payload — the cheapest latency win there is. The conditional-request handshake looks like this: the first response carries an `ETag` (a hash or version of the body), and the client sends it back on the next request as `If-None-Match`. If nothing changed, the server returns an empty `304` and the client reuses its cached copy:

```http
GET /v1/orders/ord_8Kdf02 HTTP/1.1
If-None-Match: "a1b2c3"

HTTP/1.1 304 Not Modified
ETag: "a1b2c3"
Cache-Control: private, max-age=30
```

The same `ETag`, used with `If-Match` on a write, also gives you optimistic concurrency control: a `PUT` with a stale `ETag` gets a `412 Precondition Failed` instead of silently clobbering someone else's update — the lost-update problem solved with one header. See [caching: ETags, Cache-Control, conditional requests, invalidation](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation).

**5.3 Observability and SLOs.** You cannot operate what you cannot see. Emit the RED metrics (Rate, Errors, Duration) per endpoint, propagate a correlation/request ID through every hop for tracing, log structured events, and define SLOs (e.g., "99.9% of `GET /orders` under 300 ms") so you know when the contract is degrading before customers tell you. See [observability for APIs: logs, metrics, traces, and SLOs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos).

**5.4 Performance and tail latency.** The number your customers feel is not the average — it is the **tail**. If one endpoint fans out to 10 backend calls and each has a p99 of 50 ms, the probability that *all ten* come in under 50 ms is roughly $0.99^{10} \approx 0.90$, so the *combined* request misses its 50 ms target about 10% of the time — the tail of a fan-out is worse than any single dependency. Trim payloads, compress (gzip/br), reuse connections (keep-alive, HTTP/2), and batch where it helps. See [API performance: payload size, compression, and tail latency](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency), and for the serialization layer below the wire, the payload-cost discussion in [python-performance](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means).

**Review check:** Behind a gateway with central auth and rate limiting? Cacheable responses carry `ETag`/`Cache-Control` and support conditional requests? RED metrics, tracing, and a defined SLO per endpoint? Payloads bounded and compressed, connections reused, and the *tail* (p99) measured, not just the average?

## Phase 6 — Ship the developer experience

The contract can be perfect and still fail if nobody can figure out how to call it. The final phase answers: *is this a pleasure to consume?* This is where the API stops being a contract and becomes a product.

**6.1 Spec-first with OpenAPI.** Write the OpenAPI 3.1 spec *before* the implementation, mock from it so clients can integrate in parallel, and generate server stubs and validation from it so the spec and the code cannot drift. The spec is the single source of truth for the contract. See [OpenAPI and the spec-first workflow: design, mock, generate](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate).

**6.2 SDKs and docs.** Generate SDKs from the spec so callers get a typed client in their language, and write reference docs with real, copy-pasteable examples (and a changelog). A good SDK turns the entire auth-and-retry dance into one method call; a good doc shows the request *and* the response *and* the error. See [SDKs, code generation, and reference docs developers love](/blog/software-development/api-design/sdks-code-generation-and-reference-docs-developers-love).

**6.3 Governance and a style guide.** Across an org, consistency *is* developer experience — an integrator who learned your orders API should be able to guess your payments API. Encode the conventions from Phases 1–5 into a written style guide, enforce it with a linter (Spectral for OpenAPI), and run an API review board for the decisions a linter cannot make. See [API governance and style guides: consistency across an org](/blog/software-development/api-design/api-governance-and-style-guides-consistency-across-an-org). The DX-as-a-goal mindset that frames this whole phase is [designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal).

Here is a fragment of the spec-first artifact that anchors Phase 6 — the OpenAPI definition of the refund endpoint, which generates the mock, the SDK, and the docs:

```yaml
openapi: 3.1.0
info:
  title: Payments API
  version: "1.0.0"
paths:
  /v1/refunds:
    post:
      summary: Create a refund
      parameters:
        - name: Idempotency-Key
          in: header
          required: true
          schema: { type: string, format: uuid }
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [payment_id, amount_cents]
              properties:
                payment_id: { type: string }
                amount_cents: { type: integer, minimum: 1 }
                currency: { type: string, default: "USD" }
                reason_code: { type: string }
      responses:
        "201": { description: Refund created }
        "409":
          description: Conflict
          content:
            application/problem+json:
              schema: { $ref: "#/components/schemas/Problem" }
```

The payoff of spec-first is parallelism and trust. The moment the spec exists, a frontend team can generate a mock server and build against it while the backend is still being written; a partner can generate a typed SDK and start integrating; the reference docs and the validation middleware both derive from the same file, so they cannot disagree with the running code. Contrast the alternative — docs written by hand in a wiki — which drift the instant the first endpoint changes and quietly become a source of *false* contracts that integrators code against and then get burned by. The single most demoralizing developer experience is a doc that lies, and hand-maintained docs always eventually lie.

A worked detail on the GraphQL escape hatch, since it is the paradigm most likely to fail a Phase 6 review: if you chose GraphQL in Phase 2, the resolver for a list field will, by default, fire one database query *per item* to load its children — the N+1 trap (one query for the list, $N$ for the children). A reviewer who sees a resolver without a dataloader should block it, because at 100 items that is 101 round-trips. The dataloader batches the $N$ child loads into a single `WHERE id IN (...)` query, collapsing 101 round-trips to 2. That single pattern is the difference between a GraphQL API that is fast and one that melts under a realistic query, and it is why "is the N+1 trap handled?" is a literal checklist box, covered in [the GraphQL post](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap).

**Review check:** Is there an OpenAPI spec, and is it the source of truth (not hand-maintained docs that drift)? Are SDKs and docs generated from it, with real examples and a changelog? Does the change conform to the org style guide and pass the linter? Would a new integrator find this endpoint *unsurprising* given the rest of the surface?

## The six phases, side by side

Here is the whole playbook as one table — the phase, the headline question it answers, and the track of posts that supplies the depth. This is the table to keep open while you review.

| Phase | Headline question | What you check | Track / posts |
| --- | --- | --- | --- |
| 1 · Design the contract | Is the contract correct and honest? | Resource/RPC model, URIs, methods + idempotency, status codes, payloads + naming, errors, pagination + filtering, content negotiation | A, B, C |
| 2 · Choose the paradigm | Right delivery, by force? | REST vs gRPC vs GraphQL vs events; streaming; chosen by the forces on the system | E |
| 3 · Make it safe to change | Can I change this without breaking anyone? | Compatibility rules, versioning strategy, deprecation/sunset, contract tests, schema evolution | D |
| 4 · Secure it | Who may do what, how often, with what input? | AuthN, authZ + BOLA, OAuth/OIDC, rate limiting, input validation/OWASP | F |
| 5 · Operate it | Can I run, see, and keep it fast? | Gateway/BFF, caching, observability/SLOs, performance/tail latency | G |
| 6 · Ship the DX | Is it a pleasure to consume? | OpenAPI spec-first, SDKs + docs, governance/style guide | H + A5 |

And the same six phases as a single picture, with the headline question and the track for each:

![A matrix mapping each of the six lifecycle phases to its headline review question and the track of posts that answers it](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-5.png)

## The road to v2: how to evolve without a breaking version (and when you cannot)

The single most important operational belief in this series is that **most APIs never need a v2.** A v2 is not a milestone to celebrate; it is an admission that you ran out of additive moves. It costs you a forked surface, a migration project for every client, and the long tail of partners who never migrate and force you to run v1 forever anyway.

So the road to v2 is, in practice, a road that *avoids* v2 for as long as possible:

1. **Add, never change.** New capability ships as new optional fields, new optional query parameters, new endpoints, new enum values that old clients can tolerate. The robustness principle makes all of this non-breaking. ([Backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change).)
2. **Expand/contract for the rare in-place change.** When a field genuinely must change shape, dual-write old and new, migrate clients with telemetry, then contract after the sunset window. The rename never breaks anyone because both forms exist during the migration. ([Schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely).)
3. **Use feature flags and content negotiation for divergent behavior** before reaching for a version number. A client can opt into new behavior via a header or a media type without you minting `/v2`. ([Versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning), [content negotiation](/blog/software-development/api-design/content-negotiation-media-types-and-representations).)

**When is v2 truly unavoidable?** When you must make a change that no amount of additive design or expand/contract can hide from clients: a fundamental restructuring of a core resource's identity or relationships; a security-forced change that *must* break old behavior; a paradigm shift (REST → gRPC) for a surface; or an accumulation of breaking changes large enough that a clean cut is genuinely kinder to clients than a death by a thousand deprecations. When that day comes, do it humanely: ship v2 alongside v1, document the migration field-by-field, send the `Deprecation` and `Sunset` headers on v1, give a generous window (18 months is a reasonable public default), and *track* who is still on v1 so you can reach out to the stragglers. The humane retirement mechanics are the whole of [deprecation and sunset](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely).

The asymmetry to remember: a non-breaking change is a deploy; a breaking change is a *program* — comms, migration guides, dual-running infrastructure, and months of follow-up. Price that program in before you reach for the version number, and you will reach for it far less often.

There is a cultural failure mode worth naming here, because it is so common. A team treats "v2" as a chance to fix everything they dislike about v1 in one heroic rewrite — and then discovers that v1 has a partner who integrated three years ago, never had a reason to migrate, and now generates 0.3% of traffic that happens to be the most lucrative 0.3%. You cannot turn v1 off, so you run two full APIs forever, doubling your maintenance, your security surface, and your test matrix. The dual-running cost of a breaking version is not a one-time migration; it is a tax you pay every day until the last straggler leaves, and the last straggler may never leave. This is why the additive road is not merely "nicer" — it is *cheaper over the life of the API by a wide margin*. The version number is the most expensive token in your contract; spend it like it is.

## The top ten mistakes (and the post that fixes each)

These are the ten that recur across every codebase I have reviewed. Each is a one-line smell with the post that supplies the fix.

1. **`200 OK` with an error in the body.** The client cannot branch on the status line, and retry logic breaks. Fix: [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).
2. **A money-moving POST with no idempotency key.** A network retry double-charges. Fix: [idempotency keys: safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).
3. **Offset pagination over a hot table.** The export skips and duplicates rows as the window shifts. Fix: [pagination: offset, cursor, and keyset](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale).
4. **Renaming or removing a response field in place.** Every client that read the old field 500s. Fix: [schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely) and [backward/forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change).
5. **Adding a *required* request field to an existing endpoint.** Old clients get a `422` they cannot fix. Fix: make it optional with a default; see [designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming).
6. **Checking authentication but not ownership (BOLA).** Any logged-in caller reads any resource by ID. Fix: [authorization: scopes, roles, and resource-level permissions](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions) and the OWASP walk in [input validation and the OWASP API Top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10).
7. **A token with no expiry.** A leaked credential works forever. Fix: [authentication: API keys, sessions, JWT, and mTLS](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) and the right flow from [OAuth 2.0 and OIDC](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers).
8. **No rate limiting.** One buggy client takes the whole API down. Fix: [rate limiting, quotas, and abuse protection](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection); put it at [the gateway](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern).
9. **An unbounded, uncached, uncompressed payload.** The p99 blows out on mobile and the database melts on a fan-out. Fix: [API performance: payload size, compression, and tail latency](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) and [caching with ETags](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation).
10. **No spec, hand-maintained docs that drift, and no consistency across the org.** Integrators guess wrong and file tickets. Fix: [OpenAPI spec-first](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate), [SDKs and reference docs](/blog/software-development/api-design/sdks-code-generation-and-reference-docs-developers-love), and [governance and style guides](/blog/software-development/api-design/api-governance-and-style-guides-consistency-across-an-org).

The same ten as a table, with who each one breaks and the fix:

| Mistake | Who it breaks | The fix |
| --- | --- | --- |
| `200` with error body | Client error handling and retries | [Status codes](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx) |
| POST without idempotency key | The customer, twice | [Idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) |
| Offset pagination on a hot table | The export, silently | [Pagination](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale) |
| Rename/remove a field in place | Every existing client | [Schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely) |
| New required request field | Old clients (instant `422`) | [Body design](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming) |
| AuthN without ownership (BOLA) | Every customer's data | [Authorization](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions) |
| Token with no expiry | Everyone, on a leak | [Authentication](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) |
| No rate limiting | All callers, on one bad one | [Rate limiting](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection) |
| Fat, uncached payload | The mobile p99 | [Performance](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) |
| No spec, drifting docs | Every integrator | [OpenAPI spec-first](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate) |

![A matrix listing the most common API mistakes alongside who each one breaks and the specific post in the series that supplies the fix](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-6.png)

## Case studies: who does this well

These are real, public practices worth modeling — described at the level I am confident is accurate.

**Stripe — idempotency keys and dated versioning.** Stripe popularized the `Idempotency-Key` header on its payment-creation endpoints precisely to make retries safe on a money-moving POST, exactly the Phase 1 / Phase 3 pattern this playbook centers. Stripe also versions by *date* and pins each account to the version it integrated against, upgrading customers deliberately rather than forcing a v2 — a strong real-world example of "evolve without a breaking version."

**GitHub — REST and GraphQL side by side, with honest deprecation.** GitHub runs both a REST API and a GraphQL API, choosing each by force (REST for simple resource access, GraphQL for clients that need to shape a graph of data), and uses `Sunset`-style deprecation headers and long migration windows when retiring surface — the Phase 2 and Phase 3 patterns in production.

**Google AIP and the major style guides.** Google's API Improvement Proposals (AIP), along with the Microsoft REST API guidelines and the Zalando RESTful API guidelines, are written style guides that encode exactly the kind of cross-org consistency Phase 6 asks for. If you need a starting point for your own style guide, adopt and adapt one of these rather than writing from scratch.

**RFC 9457 (`problem+json`).** The error-design pattern in Phase 1 is not a local invention; it is a standard. RFC 9457 ("Problem Details for HTTP APIs") defines the `type`/`title`/`status`/`detail`/`instance` envelope, which means a client library can parse your errors with off-the-shelf tooling.

The common thread: each of these treats the API as a long-lived contract and a product with users, secures and operates it as a public surface, and evolves it additively. That is the entire thesis of the series, validated in the wild.

What is striking, looking across these examples, is how *unglamorous* the durable choices are. There is no clever trick that makes an API last; there is a refusal to skip the boring steps — the idempotency key, the dated version, the `Sunset` header, the standard error envelope, the written style guide. Longevity in APIs is not won by brilliance, it is won by consistency, and consistency is precisely what a checklist enforces. The teams whose APIs you can still integrate against ten years later are not smarter; they are the ones who ran the equivalent of this checklist on every change, for ten years, without getting bored of it.

## How to run the review in practice

A checklist that lives in a blog post helps nobody. To make it real, embed it where the change happens:

**Put it in the pull-request template.** The printable checklist below should be the body of `.github/PULL_REQUEST_TEMPLATE.md` (or your equivalent) for any repo that exposes an API. The author checks the boxes as they open the PR; the reviewer verifies them. A box the author cannot honestly check is a conversation that happens *before* merge, which is the entire point.

**Automate the boxes a machine can check.** Several gates do not need human judgment and should fail the build, not the review: the schema diff (`oasdiff` / `buf breaking`) for breaking changes, the spec linter (Spectral) for style-guide conformance, a test that asserts no endpoint returns `200` with an `error` field, a test that every collection endpoint enforces a max page size. Every gate you automate is a gate that cannot be skipped on a busy day. Reserve human review for the judgment calls a linter cannot make — is this resource model right, is this genuinely the correct paradigm, is this break truly unavoidable.

**Run a lightweight API review board for the irreversible decisions.** Not every PR needs a committee, but the decisions that are expensive to reverse — a new top-level resource, a new public version, a new authentication scheme, a deprecation — benefit from a second set of senior eyes before they ship. Keep it fast and advisory, not a bureaucratic gate; its job is to catch the contract mistakes that will cost a migration, not to slow down a field addition. This is the governance layer from Phase 6, applied with restraint.

**Make the checklist a living document.** Every time an incident traces back to an API mistake that the checklist did not catch, add a box. The list you ship with should grow from your own post-mortems, not stay frozen as whatever a blog post suggested. The ten mistakes above are the universal ones; your team will have its own.

The goal of all of this is not ceremony. It is to move the moment of discovery *earlier* — from the production incident, where a mistake costs a customer and a 2 a.m. page, to the pull request, where it costs a comment and a five-minute fix. That shift, repeated across thousands of changes over years, is the difference between an API that lasts and one that becomes the thing nobody wants to touch.

## When to reach for this playbook (and when not to)

Every rule here is a trade-off, and a checklist applied without judgment becomes bureaucracy. So, plainly:

- **Run the full six-phase review** on anything public, anything that moves money, anything with more than one consuming team, and anything you cannot easily change later. The cost of the review is trivial next to the cost of a breaking change to a partner.
- **Do not gold-plate an internal, single-consumer, short-lived endpoint.** A script that one cron job calls does not need OpenAPI governance and an SDK. Get the contract honest (Phase 1) and skip the ceremony of Phases 5–6.
- **Do not version in the URI when additive change covers you** — you will fragment the surface and tempt yourself into a v2 you do not need. Reach for a version number only when a true break is forced.
- **Do not adopt GraphQL for simple CRUD with one client.** You buy the N+1 trap and the caching problem for flexibility you will not use. Choose by force.
- **Do not add HATEOAS links no client will follow.** Hypermedia that nobody navigates is payload weight and maintenance for zero benefit.
- **Do not return `200` with an error body to "be friendly" to clients that mishandle status codes.** Fix the clients; do not corrupt the contract for everyone else.

The meta-rule: the checklist exists to make the *important* decisions deliberate, not to make *every* decision heavy. Spend the review budget where the contract is hard to change.

## The printable review checklist

This is the part to copy into your pull-request template or your API review doc. A reviewer runs it top to bottom and blocks the merge the moment one box cannot be checked. Each gate maps to a phase above; the figure after it shows the five gates a reviewer clears in order.

**Phase 1 — Contract**
- [ ] Each URL is a noun the domain owns; identity in the path, filters in the query.
- [ ] Each method matches its semantics; every GET is side-effect-free.
- [ ] Every money-moving or resource-creating POST accepts an `Idempotency-Key`.
- [ ] The status code alone tells the client what happened; no `200` with an error body.
- [ ] `4xx` for the caller's fault, `5xx` for ours; `429` carries `Retry-After`.
- [ ] Consistent casing, one envelope convention, money as integer minor units, ISO-8601 UTC timestamps.
- [ ] Every error is `application/problem+json` with a stable `type`.
- [ ] Every collection is bounded (default + max page size); cursor/keyset, not offset, for hot data; stable ordering.
- [ ] Filters and sorts are validated against an allowlist.

**Phase 2 — Paradigm**
- [ ] The paradigm (REST / gRPC / GraphQL / events / streaming) is chosen by a named force, not fashion.
- [ ] If GraphQL: the N+1 trap is handled with a dataloader. If gRPC: deadlines are set. If webhooks: signed and retried.

**Phase 3 — Evolution**
- [ ] This change is additive (safe). If not, expand/contract makes it non-breaking, or it is a deliberate, communicated break.
- [ ] The schema diff and consumer contract tests pass in CI.
- [ ] Any deprecation carries a `Deprecation` header, a `Sunset` date, and a comms plan.

**Phase 4 — Security**
- [ ] Every token expires; the right OAuth flow per client type.
- [ ] Every resource access checks ownership, not just authentication (no BOLA).
- [ ] Inputs are validated and field-allowlisted (no mass assignment); the OWASP API Top 10 is walked.
- [ ] Rate limits are enforced with `429` + `Retry-After`.

**Phase 5 — Operability**
- [ ] Behind a gateway with central auth and rate limiting.
- [ ] Cacheable responses carry `ETag`/`Cache-Control` and support conditional requests.
- [ ] RED metrics, tracing with a correlation ID, and a defined SLO per endpoint.
- [ ] Payloads bounded and compressed, connections reused, the p99 (tail) measured.

**Phase 6 — Developer experience**
- [ ] The OpenAPI spec is updated and is the source of truth.
- [ ] SDKs and docs are generated from the spec, with real examples and a changelog.
- [ ] The change passes the org style-guide linter and would be unsurprising to a new integrator.

![A vertical stack of the five review gates a reviewer clears in order from an honest contract at the top to updated spec and docs at the bottom](/imgs/blogs/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2-8.png)

If a reviewer can check every box, the change is safe to ship. If they cannot, the unchecked box names the exact post to go read. That is the whole point of the series: forty deep dives collapsed into one gate you can run in ten minutes.

## Key takeaways

- **An API is a contract and a product, not a function call.** Design for a caller you will never meet, on a timeline of years. Every decision: *what does the caller get to assume, and can I change this later without breaking them?*
- **Run the lifecycle in order:** correct contract → right paradigm → safe to change → secured → operated → great DX. You cannot secure a wrong model or document a moving contract.
- **The robustness principle is the engine of evolution:** additive change is safe because tolerant readers ignore what they do not recognize. Removing, renaming, or requiring is breaking. Use expand/contract for the rare in-place change.
- **Most APIs never need a v2.** A v2 is an admission you ran out of additive moves. Price the migration program before reaching for a version number, and you will reach for it far less often.
- **Honest status codes and idempotency keys are non-negotiable on day one.** Never `200` with an error body; never an unprotected money-moving POST.
- **Authorization is not authentication.** Check ownership on every resource access — BOLA is the most common and most damaging API vulnerability.
- **The tail is the number customers feel.** Measure p99, not the average, and remember a fan-out's tail is worse than any single dependency's.
- **Consistency is developer experience.** A written style guide enforced by a linter, with the OpenAPI spec as the source of truth, makes the whole surface guessable.
- **The checklist makes the important decisions deliberate** — not every decision heavy. Spend the review budget where the contract is hard to change.

## Further reading

The canonical sources behind the rules in this playbook:

- **RFC 9110 — HTTP Semantics** (the authority on methods, status codes, and safe/idempotent semantics).
- **RFC 9457 — Problem Details for HTTP APIs** (the `problem+json` error envelope).
- **RFC 5789 — PATCH**, **RFC 6902 — JSON Patch**, **RFC 7396 — JSON Merge Patch** (partial updates).
- **RFC 6749 — OAuth 2.0** and **RFC 7519 — JWT** (delegated authorization and bearer tokens).
- **The OpenAPI 3.1 Specification**, the **gRPC / Protocol Buffers** docs, and the **GraphQL Specification** (the three paradigm contracts).
- **Google AIP**, the **Zalando RESTful API Guidelines**, and the **Microsoft REST API Guidelines** (style guides to adopt rather than write from scratch).
- The **OWASP API Security Top 10** (the security checklist behind Phase 4).

Within this series, the two posts that bookend everything: the intro hub, [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), and the deeper distributed-systems and evolution-at-scale views in [API design across REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) and [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale). Every checklist item above links to the post that goes deep on it — this playbook is the index; those forty are the depth. Print the checklist, pin it to your PR template, and run it before you approve any API change. That is how an API lasts from the first endpoint to v2 and well beyond.

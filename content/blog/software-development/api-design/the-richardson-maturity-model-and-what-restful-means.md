---
title: "The Richardson Maturity Model and What RESTful Actually Means"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Cut through the cargo-cult: what REST actually is, how the Richardson maturity levels really stack up, and why Level 2 is where good APIs live and should stay."
tags:
  [
    "api-design",
    "api",
    "rest",
    "richardson-maturity-model",
    "hateoas",
    "http",
    "hypermedia",
    "rest-constraints",
    "uniform-interface",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-1.png"
---

A few years ago I sat in a design review where two engineers argued for forty minutes about whether an endpoint was "RESTful." One said it was not, because the response did not include hypermedia links. The other said it absolutely was, because it used `GET` for reads and `POST` for writes and returned JSON. Neither of them could state, precisely, what REST is — and that is not a knock on them. Most of us learned "REST" as a folk practice: nouns in the URL, verbs as HTTP methods, JSON in the body, and a vague sense that links would be nice but nobody actually does them. We inherited a vocabulary without its definitions.

That gap matters because the word does real work. Teams reject perfectly good designs for failing a purity test they cannot articulate, and they ship genuinely broken designs while calling them RESTful. The fix is not more dogma. It is two precise things: knowing what Roy Fielding actually specified when he named "Representational State Transfer," and knowing the **Richardson Maturity Model** — a four-level ladder that Leonard Richardson laid out and Martin Fowler popularized, which lets you say *exactly* how much of the web's machinery a given API is using. Once you have both, the forty-minute argument collapses into a one-sentence answer: "It's a clean Level 2 API; it doesn't do hypermedia, and for this use case it shouldn't."

By the end of this post you will be able to do four things. First, define REST honestly — the five constraints Fielding named, and what each one buys you. Second, place any API on the maturity ladder: Level 0 (one URI, one verb — RPC tunneled through HTTP, "the swamp of POX"), Level 1 (many resources), Level 2 (HTTP verbs and status codes used as designed), and Level 3 (hypermedia controls, the thing people mean by HATEOAS). Third, make the call most teams should make: **Level 2 captures the great majority of the value, and Level 3 is rarely worth it for typical machine-to-machine APIs** — with the precise reasons why, and the genuine cases where Level 3 earns its keep. Fourth, render the same operation at each level so the differences stop being abstract.

![a vertical stack of the four Richardson maturity levels from Level 0 RPC over POST at the bottom up to Level 3 hypermedia at the top, with each rung adding one ingredient](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-1.png)

This is the fourth post in **Designing APIs That Last**, and it sits right after [resource modeling](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) — once you have nouns and URIs, the maturity model is the next lens. Throughout we will use the series' running spine: a **Payments & Orders** API for a fictional commerce platform, with `/orders`, `/payments`, and `/refunds`. We will take one operation — refunding part of an order — and show it at Level 0, Level 2, and Level 3, so you can feel the difference in the wire, not just the slogans. And we will keep landing on the question this whole series keeps asking: *what does the caller get to assume, and can I change this later without breaking them?* That question, it turns out, is the real reason the maturity model matters at all.

## 1. The word "REST" has a definition, and it is not "JSON over HTTP"

Let me start with the thing almost everyone gets wrong, because it reframes everything after it. **REST is not a protocol, a format, or a set of URL conventions.** It is an *architectural style* — a named bundle of constraints — that Roy Fielding described in Chapter 5 of his 2000 doctoral dissertation, *Architectural Styles and the Design of Network-based Software Architectures*. Fielding was one of the principal authors of the HTTP specification, and the dissertation was, in part, an attempt to explain *why* the web scaled the way it did. REST is the name he gave to the design that made the web work.

That origin is worth holding onto, because it tells you what REST is *for*: it is the set of properties that let a hypermedia system grow to planet scale, survive decades of independent change on both the client and server sides, and let intermediaries (caches, proxies, gateways) sit in the middle without understanding the application. REST is an answer to the question, "how do you build a distributed system that nobody controls end-to-end and that has to keep working as every part of it changes underneath you?" When you remember that, the constraints stop feeling arbitrary.

Here is the term, defined plainly the first time, since the series promises that. A **resource** is any concept worth naming and addressing — an order, a payment, the collection of all refunds for an order. A **representation** is a concrete snapshot of a resource in some format at some moment — the JSON document you get back when you `GET` it, or the form a browser renders. REST is literally *Representational State Transfer*: clients and servers exchange **representations** of resources, and the client drives the application forward by transferring from one **state** to the next as it acts on those representations. The name is the whole idea compressed into three words.

Now the five constraints. Fielding derived REST by starting from the "null style" (no constraints at all) and adding constraints one at a time, each one buying a property. There are six in the dissertation, but one — *code-on-demand*, the ability to ship executable code like JavaScript to the client — is explicitly optional, so the working set for API designers is five.

- **Client-server.** Separate the user-interface concern from the data-storage concern. The client owns presentation; the server owns the resources and their state. They are allowed to evolve independently as long as the interface between them holds.
- **Stateless.** Each request from client to server must carry everything the server needs to understand it. The server keeps no client *session* state between requests. We will spend real time on this one because it is the most violated and the most consequential.
- **Cacheable.** A response must, implicitly or explicitly, label itself as cacheable or not, so that a client or an intermediary can reuse it and skip a round-trip.
- **Uniform interface.** This is the big one, the constraint that most distinguishes REST. Every resource is manipulated through the *same small, generic interface* — in HTTP's case, a fixed set of methods, a uniform way to name resources (URIs), self-describing messages, and hypermedia as the engine of application state. We will unpack each piece.
- **Layered system.** A client cannot tell whether it is talking to the origin server or to an intermediary. This is what lets you slot a CDN, an API gateway, a load balancer, or a security proxy into the path without the client knowing or caring.

Notice what is *not* on that list. JSON is not on it — REST is format-agnostic; XML, JSON, Protobuf, even HTML are all fine representations. "Use `GET` for reads" is not on it either, at least not by that name; it falls out of the uniform-interface constraint combined with HTTP's method semantics. And nowhere does it say "put nouns in your URLs and avoid verbs." That is good advice, and it follows from the constraints, but it is a consequence, not the definition. When someone says an API "isn't RESTful," the honest follow-up is: *which constraint does it violate, and what property are you therefore losing?* If they cannot answer, they are doing taste, not architecture.

There is one more thing Fielding was emphatic about, and it is the crux of the HATEOAS debate we will reach later. In a 2008 blog post he wrote — paraphrasing — that an API which does not let the client be driven by hypermedia is not, by his definition, REST; it is RPC. He meant it strictly. The uniform-interface constraint includes the sub-constraint "hypermedia as the engine of application state" (**HATEOAS**), and an API that omits it is, in Fielding's own usage, not entitled to the name. This is the deep irony of the field: by the definition of the man who coined the term, **almost nothing the industry calls "REST" is actually REST.** That is not a reason to panic. It is a reason to be precise — and the maturity model is the tool that lets us be.

## 2. The maturity model: a ruler, not a scoreboard

In 2008 Leonard Richardson gave a talk that organized the messy spectrum of "HTTP-based APIs" into four levels, each defined by which slice of the web's machinery it uses. Martin Fowler then wrote a widely-read article, *Richardson Maturity Model*, that made the four-level ladder canonical. The single most important thing to understand about it is what Fowler himself stressed: **the levels are a way to understand the techniques behind RESTful thinking, not a grading rubric where higher is always better.** Richardson's own framing was descriptive. We added the moral judgment afterward, and the moral judgment is where teams go wrong.

So let me give you the model as a *ruler* first, neutral, and reserve the opinions for after.

- **Level 0 — one URI, one verb.** The API exposes a single endpoint (or a tiny handful) and tunnels every operation through `POST`, dispatching on the body. The HTTP layer is a dumb transport; the real "method" lives in a field of the request. Fowler, borrowing from Ian Robinson, calls this **the Swamp of POX** — Plain Old XML (or JSON) over HTTP. SOAP web services and most JSON-RPC fit here. It is *Remote Procedure Call* dressed as a web request.
- **Level 1 — resources.** Instead of one endpoint, the API has many, each naming a distinct resource: `/orders/42`, `/payments/991`. You have *divided the problem into nouns*. But you still typically talk to each resource with a single verb (usually `POST`), and the meaning of the call still lives partly in the body. You've borrowed the web's *addressing* but not its *vocabulary*.
- **Level 2 — HTTP verbs and status codes.** Now you use HTTP's methods as designed: `GET` to read, `POST` to create, `PUT` to replace, `PATCH` to modify, `DELETE` to remove — and you return *honest status codes*: `201 Created` with a `Location`, `404 Not Found`, `409 Conflict`, `422 Unprocessable Content`, `429 Too Many Requests`. The protocol now carries the semantics. Caches understand your `GET`s; clients can safely retry your idempotent calls. **This is where the overwhelming majority of well-designed "REST" APIs live, and — I will argue — where most of them should stay.**
- **Level 3 — hypermedia controls (HATEOAS).** Responses include *links* that tell the client what it can do next and where. The client does not construct URLs from documentation; it discovers them by following the links the server hands it. This is the constraint Fielding insisted was non-negotiable for the name "REST." It is also, in practice, the rarest.

![a four-row matrix mapping each Richardson level to what it adds and what it costs, with Level 2 marked as low cost and most value and Level 3 marked as high cost and thin payoff](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-2.png)

The matrix above is the whole argument of this post in one figure. Read the right-hand column: the *cost* of each step. Going from Level 0 to Level 1 is cheap and buys you addressability. Level 1 to Level 2 is also cheap — it is mostly a matter of using the methods and codes you already had — and it buys you the entire toolbox of HTTP intermediaries, caching, and safe retries. Level 2 to Level 3 is where the curve bends: the cost (designing a link vocabulary, building clients that follow links, maintaining the contract for both) rises sharply, while the marginal benefit for the typical API falls. We will spend section 7 and 8 on exactly why.

One more framing before we descend the ladder. The levels are *additive* and roughly *cumulative*: a Level 3 API is also doing Levels 1 and 2; a Level 2 API has already done Level 1. You do not skip rungs. And the model is a *direction*, not a destination — knowing the ladder tells you which next ingredient is available, and lets you make a deliberate choice about whether to reach for it. The skill is not "always climb." The skill is "climb until the next rung costs more than it pays, then stop and say so out loud."

## 3. Level 0: the Swamp of POX, or RPC wearing an HTTP costume

Let me make Level 0 concrete with the running example. Suppose our commerce platform exposes a single endpoint. Every operation — create an order, fetch an order, issue a refund — is a `POST` to `/api`, and the body names the operation.

```http
POST /api HTTP/1.1
Host: shop.example.com
Content-Type: application/json

{
  "method": "refundOrder",
  "params": {
    "order_id": "ord_42",
    "amount_cents": 1500,
    "reason": "damaged_item"
  }
}
```

And here is the response — and this is the tell, the thing that marks it as Level 0:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "ok": false,
  "error": "refund_exceeds_charge",
  "message": "Refund amount 1500 exceeds remaining charge of 1200"
}
```

The request *failed* — the refund exceeded what was charged — and yet the HTTP status is `200 OK`. The status line is lying. The real outcome is buried in an `ok` field in the body. To anything that reads HTTP — a cache, a proxy, a monitoring tool, a client's generic retry library — this is a success. That single fact is the root of most Level 0 pain.

Why does this style even exist? Because it is the most natural thing in the world for someone who thinks in function calls. You have a function `refundOrder(order_id, amount, reason)`, and you want to call it over the network, so you serialize the name and arguments and send them. That is precisely **Remote Procedure Call** — RPC. There is nothing inherently wrong with RPC as a paradigm; gRPC and JSON-RPC are deliberate, high-quality RPC designs, and a [later post in this series](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) treats RPC on its own terms. The problem with Level 0 specifically is that it takes RPC and tunnels it through HTTP *while pretending to be a web API*, getting none of HTTP's benefits and all of its overhead.

Here is the principle, stated rigorously, that explains *why* Level 0 hurts: **HTTP is a uniform interface whose value comes entirely from intermediaries being able to understand a request without understanding the application.** A cache can serve a `GET /orders/42` because the *method* `GET` tells it the request is safe (no side effects) and the URI tells it *what* is being identified — without the cache knowing anything about orders. The moment you collapse every operation into `POST /api` and hide the verb in the body, you have destroyed exactly that property. The method is now always `POST` (so always "unsafe, do not cache, do not blindly retry"), and the URI is now always `/api` (so it identifies nothing in particular). Every intermediary in the path is rendered blind. You are paying for HTTP — the headers, the connection setup, the parsing — and getting, in return, a worse version of a raw socket.

Let me make the consequence concrete and personal, since this series promises before→after damage.

#### Worked example: the cache that served a stale refund

A team I worked alongside ran a Level 0 "API gateway" — one POST endpoint, dispatch by body. To cut latency, an SRE put a caching proxy in front of it and configured it, reasonably, to cache `POST` responses by hashing the request body for a short TTL (a documented, if aggressive, optimization for genuinely idempotent reads). It worked beautifully for read-shaped operations like `getOrder`. Then a customer requested a refund, the merchant *re-submitted the same refund request* a second time after a timeout, and the proxy — seeing an identical body it had cached from a *failed* attempt seconds earlier — returned the cached `200 OK` *without ever reaching the backend*. The refund silently never happened, but both the merchant and the backend believed it had, because the cached body said `"ok": true` from a coincidentally similar earlier call. (The body-hash collided across two operations that differed only in a field the cache key didn't include.)

The root cause was not the SRE's cache config. The root cause was that **at Level 0 there is no honest way to tell an intermediary "this is a non-idempotent write, do not cache or replay it."** The method that would have carried that signal — `POST` as a non-safe, non-idempotent operation on a *specific resource* — had been flattened into a single opaque `POST /api`. The fix the team eventually shipped was to climb to Level 2: refunds became `POST /orders/{id}/refunds`, reads became `GET`, and the proxy's behavior became *correct by default* because the HTTP semantics finally told the truth. The lesson is the whole reason the maturity model matters: the levels are not about elegance, they are about *which actors in the system can act correctly without reading your documentation.*

![a before and after comparison showing a Level 0 refund as a single POST to slash api returning a misleading 200 versus a Level 2 refund as POST to a refunds resource returning 201 Created with a Location header](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-3.png)

## 4. Level 1: resources — dividing the world into nouns

Climbing to Level 1 is structurally simple and it is the single highest-leverage move you can make: **stop having one God-endpoint and start naming resources.** Instead of `POST /api` with a `method` field, you have many addresses, each identifying one thing.

```http
POST /orders/ord_42/refunds HTTP/1.1
Host: shop.example.com
Content-Type: application/json

{
  "amount_cents": 1500,
  "reason": "damaged_item"
}
```

Notice what changed and what did not. *Changed:* the URI now identifies a specific resource — the refunds collection belonging to order `ord_42`. The operation's *target* is now in the address where intermediaries and humans can both see it. We have borrowed HTTP's **addressing**. *Did not change:* we are still using `POST` for everything, and we are probably still returning `200 OK` for failures and stuffing outcomes in the body. We borrowed the web's nouns but not its verbs or its status vocabulary.

[Resource modeling](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) is its own deep topic — how to turn a domain into the right nouns, how to handle relationships and sub-resources, when something is a resource versus a field. I will not re-derive it here. The maturity-model point is narrower and worth stating crisply: **Level 1 is the move from "one endpoint that does everything" to "many endpoints that each identify something."** It is the difference between a switchboard and an address book. Even on its own, it is a large improvement: your URLs become meaningful, your logs become readable (`POST /orders/ord_42/refunds` tells you what happened; `POST /api` tells you nothing), and your access-control and routing rules can finally key off the resource being touched.

Here is a subtle thing most people miss about Level 1 in isolation. Plenty of "REST" APIs are *actually* Level 1, not Level 2, and they don't know it — because they have lovely resource URIs but then `POST` to all of them, or use `GET` for a query but tunnel mutations through a single `POST /orders/{id}/actions` with an action name in the body. That is Level 1 with Level-2 cosmetics. The litmus test is mechanical: **does the HTTP *method* carry meaning, and does the *status code* tell the truth?** If the answer is no — if everything is `POST` and everything returns `200` — you are at Level 1 no matter how pretty the URLs are. Knowing that lets you diagnose your own API honestly instead of assuming the nouns earned you the full credit.

## 5. Level 2: where the web's machinery actually pays off

This is the level that matters most, so we will go slow and concrete. At Level 2 you do two things together: **use HTTP methods according to their defined semantics, and return status codes that tell the truth.** Both pieces are necessary; people often do one and forget the other.

### 5.1 Methods carry meaning: safe and idempotent

Two properties of HTTP methods, defined in RFC 9110 (the current HTTP semantics specification), do almost all the work. A method is **safe** if it is intended to be read-only — it does not change server state in a way the client is responsible for. `GET`, `HEAD`, and `OPTIONS` are safe. A method is **idempotent** if making the same request twice has the same effect on the server as making it once. `GET`, `HEAD`, `OPTIONS`, `PUT`, and `DELETE` are idempotent; `POST` and `PATCH` are not (in general). Safe implies idempotent, but not the reverse — `DELETE` is idempotent (deleting an already-deleted thing leaves it deleted) but not safe (it changes state).

Why does this distinction earn a place in your design — why is it *load-bearing* and not trivia? Because these two bits are the *contract the client gets to assume*, and they directly govern what the client and every intermediary may safely do. Let me make that rigorous with the one piece of reasoning that justifies the whole level.

> **The retry-safety theorem (informal).** Networks fail. A client that sends a request and then loses the connection before receiving a response cannot know whether the server processed the request. Its only recovery options are to retry or to give up. A retry is safe to perform *automatically* if and only if the method is **idempotent** — because then the duplicate has no additional effect. Therefore the idempotency bit of each method is precisely the signal that tells a generic client library whether it may transparently retry on a timeout.

That theorem is why `GET` and `PUT` and `DELETE` can be retried by an HTTP client, a service mesh, or a load balancer with no application knowledge, while `POST` cannot — and it is why creating a payment with a bare `POST` and no extra machinery is dangerous on a flaky network. (The fix, **idempotency keys**, is a whole post of its own — see [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions); the short version is that you attach a client-generated key so the *server* can dedupe a retried non-idempotent `POST`.) The point for *this* post is that Level 2 is what makes the idempotency contract *expressible at all*. At Level 0, every operation is `POST`, so the client can never know what is safe to retry; at Level 2, the method itself is the answer.

Here is the Payments/Orders mapping written out, because seeing the verbs against the resources is worth a thousand words of theory:

| Operation | Method + URI | Safe? | Idempotent? | Typical success |
| --- | --- | --- | --- | --- |
| List an order's refunds | `GET /orders/ord_42/refunds` | yes | yes | `200 OK` |
| Read one refund | `GET /refunds/rf_88` | yes | yes | `200 OK` |
| Create a refund | `POST /orders/ord_42/refunds` | no | no | `201 Created` + `Location` |
| Replace an order's shipping address | `PUT /orders/ord_42/shipping` | no | yes | `200 OK` |
| Partially update an order | `PATCH /orders/ord_42` | no | no | `200 OK` |
| Cancel (delete) an order | `DELETE /orders/ord_42` | no | yes | `204 No Content` |

Read the two middle columns top to bottom. They are the contract. A client looking at this table knows, without reading a word of prose, that it may retry the `GET`s and the `PUT` and the `DELETE` blindly, but must be careful with the `POST` and the `PATCH`. That is the uniform interface doing its job: *generic knowledge of HTTP substitutes for specific knowledge of your API.*

### 5.2 Status codes that tell the truth

The second half of Level 2 is returning honest status codes. The principle is the mirror image of the method principle: **the status code is the part of the response that intermediaries and generic clients read first, so it must carry the real outcome.** A `404` must mean "not found," a `409` must mean "you tried to do something that conflicts with the resource's current state," a `422` must mean "your request was well-formed but semantically invalid," a `429` must mean "slow down." When you put the real outcome in the status line, every actor in the path can do the right thing: a cache won't store a `404` as if it were a successful body, a client's retry logic will back off on a `429` and its `Retry-After`, a gateway can route a `503` to a circuit breaker. (Choosing the *right* code in the gray areas — 404 vs 403, 409 vs 422 — is genuinely subtle and gets its own treatment in [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).)

Now the same refund, done right, at Level 2:

```http
POST /orders/ord_42/refunds HTTP/1.1
Host: shop.example.com
Content-Type: application/json
Idempotency-Key: 8f1a-44e2-9c30

{
  "amount_cents": 1500,
  "reason": "damaged_item"
}
```

A successful create:

```http
HTTP/1.1 201 Created
Location: /refunds/rf_88
Content-Type: application/json

{
  "id": "rf_88",
  "order_id": "ord_42",
  "amount_cents": 1500,
  "status": "pending",
  "reason": "damaged_item",
  "created_at": "2026-06-20T10:14:00Z"
}
```

And the *failure* from section 3 — refunding more than was charged — now told honestly, using the `problem+json` error format from RFC 9457 (a standard envelope with `type`, `title`, `status`, `detail`, `instance`):

```http
HTTP/1.1 422 Unprocessable Content
Content-Type: application/problem+json

{
  "type": "https://shop.example.com/errors/refund-exceeds-charge",
  "title": "Refund exceeds remaining charge",
  "status": 422,
  "detail": "Refund amount 1500 exceeds remaining charge of 1200",
  "instance": "/orders/ord_42/refunds",
  "remaining_charge_cents": 1200
}
```

Compare this to the Level 0 version. The status line says `422`, so anything in the path knows it was a client-side semantic failure — not cacheable as a success, not worth a blind retry. The `Content-Type` is `application/problem+json`, so a generic error handler knows how to parse it. And the body is machine-readable *and* human-friendly. This is the entire payoff of Level 2 in one exchange: **the truth is in the protocol, where the whole world can read it.**

![a before and after comparison of error handling showing a Level 0 response returning 200 OK with ok false in the body causing proxies to cache the error, versus a Level 2 response returning 422 with problem json and a 429 with Retry-After](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-8.png)

### 5.3 The forces that make Level 2 the sweet spot

Why do I keep saying Level 2 is where most APIs *should* live, and not merely where they *do* live? Because the cost-to-value ratio is extraordinary, and you can reason about it directly.

The *cost* of reaching Level 2 from Level 1 is nearly zero in new code: you are choosing `GET` instead of `POST` for a read, returning `404` instead of `200`-with-`"ok":false`. There is no new infrastructure to build, no new client capability to require. You are using parts of HTTP that already exist in every server framework and every client library on earth.

The *value* is the entire ecosystem of HTTP intermediaries, and it compounds. Caching with `ETag` and conditional requests (a `GET` that returns `304 Not Modified` saves the whole body — see [caching with ETags](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) for the mechanics). CDNs that can cache safe responses at the edge. Service meshes and load balancers that retry idempotent calls on your behalf. Gateways that rate-limit and return a clean `429`. Monitoring that buckets your traffic by status class (`2xx`/`4xx`/`5xx`) for free, because the status codes are honest. *None of that requires the client to do anything REST-specific.* It just requires your API to use HTTP as designed. That is the whole bargain: at Level 2, the web's enormous shared infrastructure works *for* you, automatically, because you spoke its language.

Put a number on one slice of it. Suppose a product-listing endpoint returns a 40 KB JSON body and is hit 10 million times a day, and suppose 70% of those hits would get an unchanged body. With Level 2 conditional requests (`If-None-Match` against an `ETag`), 7 million of those become `304 Not Modified` responses carrying only headers — call it 300 bytes each instead of 40 KB. That is roughly $7{,}000{,}000 \times (40{,}000 - 300)$ bytes saved per day, about 278 GB of egress you no longer pay for or wait on, *purely because the API is at Level 2 and the cache can speak conditional requests.* The exact numbers depend on your traffic shape, but the structural point does not: Level 2 unlocks a class of optimization that is simply unavailable below it.

### 5.4 Self-descriptive messages and content negotiation

There is a third, quieter piece of Level 2 that the methods-and-status framing tends to skip, and it's worth a moment because it's the part the uniform-interface constraint cares about most: **self-descriptive messages.** A message is self-descriptive when everything needed to interpret it travels with it — the `Content-Type` says what format the body is in, the `Cache-Control` and `ETag` say how it may be cached, the status line says what happened. An intermediary, or a generic client, can act correctly on the message *without out-of-band knowledge of your application.* That property is what makes the layered system useful rather than merely possible.

The most underused expression of self-descriptiveness is **content negotiation** — the mechanism by which a client states what it can accept and the server responds in a matching form. A client sends `Accept: application/json` (or `Accept: text/csv`, or a versioned media type), and the server either honors it or returns `406 Not Acceptable`. Here's an order requested two ways from the same URI:

```http
GET /orders/ord_42 HTTP/1.1
Accept: application/json
```

```http
GET /orders/ord_42 HTTP/1.1
Accept: text/csv
```

The *resource* is the same — `ord_42` — but the *representation* differs by what the client asked for. This is exactly the resource-versus-representation distinction from section 1, now operational on the wire: one resource, many representations, negotiated per request. It's also a foundation for some versioning strategies (a vendor media type like `application/vnd.shop.v2+json` lets the URI stay stable while the representation evolves — content negotiation gets its own treatment later in the series). The maturity-model point is simply that content negotiation is *only available at Level 2 and up*, because it depends on the request and response being self-describing in the way the uniform interface requires. At Level 0, where everything is `POST /api` with one implicit format, there is nothing to negotiate.

Why does self-descriptiveness earn its place rather than being academic nicety? Because it is the precise property that lets the *layered system* constraint pay off. Recall the deal: the layered-system constraint says a client can't tell whether it's talking to the origin or an intermediary. That's worthless unless the intermediary can *do something* with the message — cache it, route it, transform it — and it can only do that if the message describes itself. Self-descriptiveness is the bridge between "you may insert a proxy" (a Level-0-and-up freedom) and "the proxy can actually help" (a Level-2 capability). Stack the three Level 2 ingredients together — honest methods, honest status, self-describing messages — and you have rebuilt, for your API, the exact set of properties that let the web itself scale.

### 5.5 Where teams get Level 2 wrong

Reaching Level 2 is cheap, but reaching it *correctly* trips a lot of teams, and the failure modes are worth naming because each one quietly forfeits a benefit you thought you'd bought.

- **Action-as-resource creep.** The most common Level 2 regression is the "action endpoint": `POST /orders/{id}/cancel`, `POST /orders/{id}/approve`, `POST /payments/{id}/capture`. This is Level 1 hiding inside Level 2 cosmetics — you've got resource URLs, but the *operation* lives in the path suffix, not the method, so `cancel` and `approve` are both opaque `POST`s that no intermediary can distinguish. Sometimes an action genuinely *is* a resource (a `capture` creates a capture record, so `POST /payments/{id}/captures` returning `201` is legitimate and good). The smell is when the suffix is a *verb* that maps cleanly to an HTTP method you're declining to use: `cancel` that should be `DELETE`, `replace-address` that should be `PUT`. The test: *can this be a standard method on a resource?* If yes, use the method; if it's truly a non-CRUD operation, modeling it as a created sub-resource is the honest Level 2 move.

- **`200` for everything, errors in the body.** We covered this as the Level 0 tell, but it sneaks into otherwise-Level-2 APIs through middleware that wraps every response in an envelope like `{ "status": "error", "data": null }` while keeping the HTTP status at `200`. The URLs and methods are fine; the status codes lie. Every benefit that depends on honest status — cache behavior, retry logic, monitoring buckets — silently breaks. The fix is to let the status line carry the truth and reserve the body for *detail*, not for the *outcome*.

- **`POST` where `PUT` belongs (and vice versa).** Teams sometimes `POST` to update an existing resource because "it's a write." But if the operation is a full replacement that's safe to repeat, `PUT` is correct and *idempotent* — and that idempotency is a contract the client can use to retry. Choosing `POST` there throws away retry-safety for no reason. The mirror mistake is `PUT` to create-or-update at a server-chosen URL; if the server picks the ID, that's a `POST` to the collection returning `201` + `Location`, because the client can't name the resource yet.

- **Misusing `204`, `202`, and `201`.** `201 Created` should carry a `Location` to the new resource — omitting it is a small but real Level 2 miss, because the `Location` header *is* the self-descriptive pointer to what you just made. `202 Accepted` means "I've taken your request but haven't finished" — the right code for a refund that will be processed asynchronously, paired with a status resource the client can poll (the long-running-operation pattern is its own post). `204 No Content` is the right success for a `DELETE` with nothing to return. Reaching for the wrong one isn't fatal, but it muddies the contract the caller reads.

- **Inconsistent error shapes.** A Level 2 API that returns honest status codes but a *different JSON error shape* per endpoint (one returns `{"error": "..."}`, another `{"message": "...", "code": 12}`, a third a bare string) forces every client to special-case its error handling. Standardizing on `problem+json` (RFC 9457) across the whole surface is the cheap fix — one parser handles every error, and the `type` field gives each error a stable, documentable identity.

None of these are exotic. They're the everyday ways a "REST API" lands at Level 1.5 while believing it's at Level 2, forfeiting exactly the intermediary-and-retry benefits that justified the climb. The diagnosis from section 11 catches all of them: *does the method carry meaning, does the status tell the truth, is the error shape uniform?* Three questions, and you know whether your Level 2 is real or cosmetic.

## 6. Back to first principles: REST's constraints, and which level honors which

We now have the maturity ladder. Let me tie it back to Fielding's constraints, because that connection is what turns "follow the levels" from a checklist into understanding. The constraints are the *why*; the levels are *how far up the why you've climbed.*

![a tree of Fielding's five REST constraints branching into interaction rules client-server stateless and cacheable and structure rules uniform interface and layered system](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-4.png)

The tree above splits the five constraints into two families — rules about how client and server *interact*, and rules about the system's *structure*. That split is mine, for teaching; Fielding lists them flat. What matters is mapping each to the level that delivers it.

- **Client-server** is satisfied the moment you have any HTTP API at all — even Level 0 separates the client's UI from the server's data. So this one is free; everybody clears it.
- **Layered system** is also essentially free at the HTTP level: because HTTP is the protocol, you *can* slot a proxy or gateway in the path. But — and this is the catch — the *benefit* of the layered system (caches, gateways acting intelligently) only materializes when the messages are self-describing, which is a Level 2 property. At Level 0, you have a layered system that the layers cannot use.
- **Cacheable** is the constraint Level 2 most directly delivers. Below Level 2, every response is an opaque `POST` result that no cache can safely store. At Level 2, `GET` responses with proper `Cache-Control` and `ETag` headers are cacheable, and you reap the savings we just computed.
- **Uniform interface** is the heart of the matter, and it has four sub-parts that map straight onto the levels. (1) *Identification of resources* — that's Level 1 (URIs name things). (2) *Manipulation through representations* and (3) *self-descriptive messages* — that's Level 2 (standard methods, honest status codes, media types). (4) *Hypermedia as the engine of application state* (HATEOAS) — that's Level 3. So the uniform interface is exactly the spine of the ladder: each rung satisfies one more piece of it.
- **Stateless** is orthogonal to the level and so often violated that it deserves its own subsection. You will see, next.

![a five-row matrix listing each REST constraint alongside what it forbids and what property it buys, from client-server buying independent evolution to layered system buying insertable proxies](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-5.png)

The matrix makes the trade explicit: each constraint *forbids* a freedom and *buys* a property. Statelessness forbids server-side session state and buys you the ability for any server in a pool to handle any request. The uniform interface forbids bespoke per-call verbs and buys you generic tooling. This is the deal REST offers — give up some local freedom, get back global scalability and evolvability. A Level 2 API has taken most of that deal; a Level 0 API has taken almost none of it.

### 6.1 Statelessness, the constraint everyone violates

**Stateless** means each request carries everything the server needs; the server keeps no client *session* between requests. The classic violation is storing a login session in server memory and handing the client a session ID that points at it. Now request N+1 only works if it lands on the same server that handled request N — you have made your servers *sticky*, and stickiness is the enemy of horizontal scaling, of rolling deploys (restarting a server drops its sessions), and of resilience (a server dying loses its users' state).

The principle, derived: **if the server holds per-client state between requests, then requests are no longer independent, and the property "any server can handle any request" — which is what lets you scale out, deploy without draining, and survive a node loss — is gone.** Statelessness is the price of that property. The fix is to push session state to the *token* the client carries (a self-contained bearer token like a signed JWT, which the server can validate without a lookup) or to a shared store the cluster reads, not to any individual server's memory.

A precise caveat, because people overreach here: stateless refers to *session* state, not *resource* state. The server obviously stores resources — orders, payments, refunds; that is its whole job. The constraint is that it must not require *remembering the client across requests* to interpret the next one. "Stateless" does not mean "stateless server"; it means "self-contained requests." Get that distinction right and you avoid both the violation (sticky sessions) and the overreaction (thinking you can't have a database).

#### Worked example: the deploy that logged everyone out

A team kept user shopping carts in server memory keyed by an in-memory session — a Level 0-ish habit that had nothing to do with their otherwise tidy Level 2 resource URLs (you can be Level 2 on the surface and still violate statelessness underneath). It worked fine on one box. They scaled to four boxes behind a load balancer and turned on sticky sessions to keep it working — and then every routine deploy, which restarts boxes one at a time, dropped a quarter of all carts on the floor, mid-checkout, every single deploy. The before→after fix was to make the cart an actual resource: `PUT /carts/{id}` to a shared store, the cart ID carried in the client's token. After that, any box could serve any request, deploys stopped logging people out, and — not coincidentally — the cart became a thing they could `GET`, cache, and reason about. Honoring the statelessness constraint didn't just fix the deploy; it *promoted the cart up the maturity ladder* as a side effect, because self-contained requests and addressable resources are two faces of the same discipline.

## 7. Level 3: hypermedia, HATEOAS, and what an affordance is

Now the rung everyone argues about. At Level 3 the server's responses include **hypermedia controls** — links (and sometimes forms) that tell the client *what it can do next and at which URL*. The term of art is **HATEOAS**: Hypermedia As The Engine Of Application State. Decoded, it means the *application state* — where the client is in its workflow, what transitions are available — is *driven by the hypermedia* the server sends, not by URLs the client built from documentation.

Define the load-bearing term: an **affordance** is a possibility for action that the interface advertises. The word comes from design theory — a door handle affords pulling, a flat plate affords pushing. In a hypermedia API, a link with a relation like `cancel` *affords* cancellation; its presence says "you may cancel this now," and its absence says "you may not." The server, by including or omitting affordances based on the resource's current state, tells the client what is possible without the client hard-coding any business rules.

Here is the refund's parent order at Level 3, with affordances:

```http
GET /orders/ord_42 HTTP/1.1
Host: shop.example.com
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_42",
  "status": "paid",
  "total_cents": 1200,
  "_links": {
    "self":    { "href": "/orders/ord_42" },
    "refunds": { "href": "/orders/ord_42/refunds" },
    "cancel":  { "href": "/orders/ord_42", "method": "DELETE" },
    "invoice": { "href": "/orders/ord_42/invoice", "type": "application/pdf" }
  }
}
```

(The `_links` shape here follows the HAL convention — Hypertext Application Language — one of several hypermedia formats; others include JSON:API, Siren, and Collection+JSON. The format is a detail; the idea is the same.) Crucially, those links are *state-dependent*. If the order were already `cancelled`, a well-built Level 3 server would omit the `cancel` link entirely. The client does not need to know the rule "you can only cancel a paid, unshipped order" — it just checks whether the `cancel` affordance is present. The business logic lives on the server, and the client follows.

![a hypermedia graph showing a GET on an order returning links labeled self pay cancel and invoice, with the client following the pay and cancel links without any hard-coded URL](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-6.png)

The graph above shows the discovery flow: the client `GET`s the order, reads the affordances, and follows the `pay` or `cancel` link by name. It never constructs `/orders/ord_42/refunds` from a string template in its own code; it reads the URL the server gave it. That is the genuine, real benefit of Level 3, and it is worth stating in its strongest form:

> **The Level 3 promise.** If clients only ever follow links the server hands them, then the server can *relocate, rename, or restructure its URLs at will* — move `/orders/{id}/invoice` to a different host, change the path entirely — and correctly-built clients keep working, because they were never hard-coding URLs in the first place. The link relations (`self`, `cancel`, `invoice`) become the stable contract; the URLs behind them become an implementation detail the server is free to change. This is the same decoupling that lets a website reorganize its URL structure without breaking the browsers that visit it.

That promise is real, it is elegant, and it is the reason Fielding insisted HATEOAS is the line between REST and RPC. So why am I, and most of the industry, going to tell you that you probably shouldn't bother for a typical machine-to-machine API? Because the promise rests on a premise — *clients only ever follow links* — that is almost never true in practice. That is the next section, and it is the honest heart of this post.

## 8. The honest argument: you probably want Level 2, and that's fine

I want to be careful and fair here, because the HATEOAS debate produces more heat than light. Let me state the case *for* Level 3 at full strength first, then the case against for the common scenario, then where the line actually falls.

**The strong case for Level 3.** The decoupling promise above is genuine. There are systems where it has paid off enormously — most obviously the World Wide Web itself, which is a Level 3 hypermedia system par excellence: your browser follows links and submits forms it discovers in HTML, never hard-coding the URLs of the sites it visits, and the entire web reorganizes its URLs constantly without breaking browsers. That is HATEOAS working at planetary scale, and it is the existence proof that the idea is sound. For a *long-lived, public, evolvable* API consumed by *many clients you do not control and cannot coordinate with*, the ability to evolve URLs without a breaking change is real and valuable.

**The case against, for the typical API.** Now the reasons it usually doesn't pay, stated precisely rather than as a sneer:

1. **Clients hard-code anyway.** This is the empirical killer. The Level 3 promise requires that clients follow links instead of building URLs. But the developer integrating with your API does not have a generic hypermedia agent; they have a deadline. They open your docs, see that a refund lives at `POST /orders/{id}/refunds`, and write exactly that string into their code — because it is faster, easier to debug, and obvious. The link in your `_links.refunds` goes unread. So you paid the full cost of designing, documenting, and serving the affordances, and got *none* of the decoupling benefit, because the clients defeated it on day one. You cannot get the benefit unilaterally; it requires discipline from clients you do not control.

2. **The tooling is thin.** Hypermedia's promise of generic, link-following clients was supposed to be delivered by generic client *libraries* that understand `_links` and navigate for you. Those libraries exist but are niche; the mainstream tooling — code-generated SDKs from an OpenAPI spec, the `requests` library, every HTTP client a working engineer reaches for — is built around *constructing* URLs, not following them. A generated SDK has a method `client.orders.refund(order_id, amount)` that builds the URL internally. That SDK is, by construction, hard-coding your URLs. The ecosystem optimized for Level 2, and ecosystems are sticky.

3. **The affordance contract is still a contract.** People sometimes claim Level 3 frees you from versioning. It does not. The *link relations* (`cancel`, `refunds`, `invoice`) and the *shapes behind them* are now your contract, and they are just as breakable as URLs — rename a relation, change what `cancel` does, and you've broken clients exactly as surely as renaming a path would. You've moved the contract, not eliminated it. (Compatibility and evolution are their own deep topic; see [the capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).)

4. **It costs real complexity on both sides.** The server must compute and serialize state-dependent affordances on every response (which links to include given this order's status), and document the relation vocabulary, and keep it stable. The client — *if* it actually wants the benefit — must be written as a state machine that reacts to affordances rather than a straight-line script. Both sides pay, and for a small set of cooperating clients the payment buys almost nothing.

![a decision tree asking whether to add HATEOAS, branching to human-driven clients like browsers and decade-long public APIs where it pays off versus machine-to-machine clients with few callers and hard-coded SDKs where it does not](/imgs/blogs/the-richardson-maturity-model-and-what-restful-means-7.png)

The tree above is the decision, distilled. Follow the branches. **HATEOAS pays when the client is human-driven** — a browser literally needs links because a person navigates by clicking them — **or when the API is a long-lived public surface with many uncontrolled clients and a genuine need to evolve URLs over years.** **It usually doesn't pay when the client is machine-to-machine, you control or can coordinate with the small set of callers, and the SDK is going to hard-code the URLs regardless.** That last branch describes the *vast majority* of internal and partner APIs. So the honest default, said plainly: **build a clean Level 2 API. Use proper resources, methods, and status codes. Add hypermedia links only where you have a concrete reason a real client will follow them — and if you can't name that reason, don't.**

I'll be even more concrete about the middle ground, because it's where good teams actually land. You do not have to choose all-or-nothing. **Pragmatic, partial hypermedia** is common and sensible: include a `next` link in paginated responses (cursors are genuinely opaque, so handing the client the next URL is *more* robust than asking it to construct one — this is hypermedia earning its keep in a narrow spot), include a `self` link, include a `Location` header on `201 Created` (that *is* a hypermedia control, and everyone uses it without calling it HATEOAS). These are Level-2-plus-a-little, and they're great. What you usually shouldn't do is build a *full* affordance vocabulary that encodes your entire state machine into links no client will follow.

### 8.1 When to reach for full Level 3 (and when not to)

Let me make this a decisive recommendation, the way the series demands, with the cases named.

**Reach for Level 3 when:**

- The client is a *browser or human-driven UI* that genuinely navigates by following links — this is the original use case and it is unambiguous.
- The API is *public, long-lived, and consumed by many clients you cannot coordinate with*, and you have a real, specific anticipated need to evolve URLs or available actions over time. The cost of a flag-day URL migration across thousands of uncontrolled integrators is high enough that the affordance discipline pays for itself.
- You are implementing or consuming a *standardized hypermedia type* where the ecosystem expects it — some standards (e.g., certain healthcare, OAuth metadata, or ACME-style protocols) are built around hypermedia, and following the standard is the right move.
- The *next action genuinely depends on opaque server state* the client shouldn't replicate — a workflow engine where which step is available next is server-computed business logic. Sending an affordance is cleaner than publishing the rule.

**Do not reach for Level 3 when:**

- You control the clients, or there are few of them and you can coordinate a change. Just version the contract when you must; it's cheaper than full hypermedia.
- The clients are SDKs generated from a spec — they will hard-code URLs by construction, so the affordances are dead weight.
- You're adding links "because REST means hypermedia" with no client that will follow them. That is cargo-cult; you are paying cost for a benefit you've structurally disabled.
- The API is a simple, stable CRUD surface for an internal service. The URLs aren't going to move; the decoupling buys nothing.

This is the same *force, not fashion* discipline this series applies to choosing REST vs gRPC vs GraphQL — see the broader treatment in [the system-design paradigm overview](/blog/software-development/system-design/api-design-rest-grpc-graphql) and [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale), which both go wider on when each paradigm earns its place. The maturity-model version of "force, not fashion" is: *climb to Level 2 always; climb to Level 3 only when a real client will follow the links.*

### 8.2 A back-of-envelope for the HATEOAS decision

Let me make the cost/benefit reasoning a little more rigorous, because "rarely worth it" deserves better than a hand-wave. Think about what each side of the trade actually buys and costs over the life of an API.

The *benefit* of full Level 3 is the ability to change URLs without a breaking change. So the value is proportional to: how often do you actually need to move a URL, multiplied by how expensive a URL migration would otherwise be, multiplied by the fraction of clients that would *correctly follow the new link* rather than break anyway. Write that as $V \approx f_{\text{move}} \times C_{\text{migration}} \times p_{\text{follow}}$. The killer is the last term. For a machine-to-machine API with SDK-generated or hand-coded clients, $p_{\text{follow}}$ is close to zero — those clients hard-code URLs by construction — so the whole product collapses regardless of how large the other two factors are. You can have a URL you'd love to move and a painful migration ahead of you, and HATEOAS still saves you nothing, because the clients that would break aren't following the links anyway.

The *cost* of full Level 3, by contrast, is paid *continuously*: every response computes and serializes affordances, every client that wants the benefit must be written as an affordance-following state machine, and the relation vocabulary becomes a contract you maintain and version forever. That cost doesn't depend on whether you ever move a URL — you pay it on request one and every request after.

So the decision is: a *continuous, certain* cost against a *contingent* benefit that's gated by a $p_{\text{follow}}$ you usually can't make large. For the web — human-driven, browser clients that genuinely follow links — $p_{\text{follow}} \approx 1$ and the benefit dominates. For your internal Orders service with three SDK consumers you control, $p_{\text{follow}} \approx 0$ and the cost dominates. That asymmetry, not aesthetic preference, is why the honest default is Level 2. You're not rejecting hypermedia because it's hard; you're rejecting it because, for your client population, you've structurally disabled the only thing it buys.

The corollary is the genuinely useful one: if you *can* raise $p_{\text{follow}}$ — by shipping a hypermedia-aware client yourself, or by targeting browsers, or by adopting a standard ecosystem that follows links — the math flips and Level 3 becomes worth it. The decision tree in the figure above is really just this inequality with the numbers replaced by branches.

## 9. The same operation at all three levels, side by side

Let me consolidate everything into the comparison the brief asks for: one operation — *refund part of an order* — rendered at Level 0, Level 2, and Level 3, so the differences are wire-level and undeniable.

#### Worked example: refunding \$15.00 of a \$12.00... wait, \$15.00 of a \$20.00 order, at L0 / L2 / L3

A customer was charged \$20.00 for order `ord_42` and we want to refund \$15.00 for a damaged item.

**At Level 0** — one endpoint, dispatch by body, status always `200`:

```http
POST /api HTTP/1.1
Content-Type: application/json

{ "method": "refundOrder", "params": { "order_id": "ord_42", "amount_cents": 1500 } }
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "ok": true, "refund_id": "rf_88", "status": "pending" }
```

The caller learns nothing from the status line; it must parse `ok` to know what happened. No intermediary can cache, retry, or route intelligently. A second identical send on a network retry might create a *second* refund, because nothing here is idempotent and nothing tells anyone so.

**At Level 2** — resource, proper verb, honest status, idempotency key:

```http
POST /orders/ord_42/refunds HTTP/1.1
Content-Type: application/json
Idempotency-Key: 8f1a-44e2-9c30

{ "amount_cents": 1500, "reason": "damaged_item" }
```

```http
HTTP/1.1 201 Created
Location: /refunds/rf_88
Content-Type: application/json

{ "id": "rf_88", "order_id": "ord_42", "amount_cents": 1500, "status": "pending" }
```

The `201` and `Location` tell every actor "a new resource was created, here it is." The `Idempotency-Key` lets the server dedupe a retried `POST` so the network retry that double-charged at Level 0 is now safe. A `GET /refunds/rf_88` is a cacheable read. This is the version 95% of good APIs should ship.

**At Level 3** — same as Level 2, plus the client discovered the URL by following a link instead of hard-coding it:

```http
GET /orders/ord_42 HTTP/1.1
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_42",
  "status": "paid",
  "amount_remaining_cents": 2000,
  "_links": {
    "self":    { "href": "/orders/ord_42" },
    "refunds": { "href": "/orders/ord_42/refunds", "method": "POST" }
  }
}
```

The client reads `_links.refunds.href`, then issues the *exact same* `POST` as the Level 2 example to that URL. The difference is purely *where the URL came from*: a server-supplied link versus a client-side template. Everything below — methods, status codes, idempotency — is identical Level 2 machinery. **Level 3 is Level 2 plus link-following; it changes how the client *discovers* the URL, not how the operation *works*.** That's the whole distinction, and seeing it on the wire is what makes the HATEOAS debate finally concrete: if your client isn't going to read that `_links` block, Level 3 added a field nobody uses.

Notice the honest detail in the Level 3 response: `amount_remaining_cents: 2000`. A truly affordance-driven server might *omit* the `refunds` link once `amount_remaining_cents` hits 0, telling the client "no more refunds possible" without the client knowing the rule. That is the genuine power — and also exactly the kind of server-side state computation that most teams decide isn't worth building when the client is just going to call `POST /orders/{id}/refunds` and handle a `422` if it's over the limit.

## 10. Case studies: how real APIs actually sit on the ladder

Let me ground all of this in named, real APIs — accurately, and only where I'm confident of the detail.

**Stripe — pragmatic Level 2, famously.** Stripe's API is widely studied as a model of API design, and it is squarely, deliberately Level 2. It uses resource URLs (`/v1/charges`, `/v1/refunds`), HTTP methods, and honest status codes. It is *not* a HATEOAS API — Stripe does not drive clients with hypermedia link vocabularies; clients use Stripe's documented endpoints and its excellent generated SDKs, which (by design) hard-code the URLs. Stripe is also the canonical real-world example of **idempotency keys**: you send an `Idempotency-Key` header on a `POST`, and Stripe guarantees a retried request with the same key won't create a duplicate charge — exactly the Level 2 retry-safety mechanism we discussed. And Stripe handles evolution through *dated API versions* pinned per account, not through hypermedia. The lesson: one of the most admired APIs in the industry is a deliberate, polished Level 2, and chose *not* to climb to Level 3.

**PayPal — explicitly hypermedia-flavored.** PayPal's REST APIs are notable for actually shipping HATEOAS links: many PayPal API responses include an `links` array with `rel`, `href`, and `method` describing the next available actions (for example, the link to capture or approve a payment). PayPal's developer documentation has historically described its use of HATEOAS explicitly. This is a real, large, public payments API that took the Level 3 step — a useful counterpoint to Stripe. Both are successful; they made different, defensible calls. The difference traces to exactly the decision tree above: a large, public, multi-step-workflow payments surface is one of the more plausible places for affordances to help.

**GitHub — Level 2 REST with hypermedia touches, plus a separate GraphQL API.** GitHub's REST API includes hypermedia-style links in many responses (URL fields like `url`, `commits_url`, and templated URLs pointing at related resources), so it leans further toward hypermedia than a bare Level 2 API — without being a full affordance-driven HATEOAS design. It's a good example of the *pragmatic partial hypermedia* middle ground: handy related-resource links, but clients still largely use documented endpoints. GitHub also famously offers a *separate GraphQL API* for cases where the REST shape is too chatty — a reminder that the maturity ladder is one axis among several, and that "more RESTful" is not the only direction to grow.

**The World Wide Web — the one true Level 3 system.** The best case study for Level 3 is the thing that inspired REST in the first place. Your browser is a generic hypermedia client: it fetches HTML, finds links and forms (affordances) inside it, and lets a human follow them — without ever hard-coding the URLs of the sites it visits. Sites reorganize their URL structures constantly and browsers keep working, because the contract is "follow the links," not "know the URLs." This is HATEOAS delivering its full promise at the largest scale in computing history — and it works *because* the client (the browser) genuinely follows links and the actor driving it (a human) genuinely needs to discover what's possible. That's the condition. When your machine-to-machine client doesn't meet it, you don't get the web's magic; you get an unused `_links` block.

These four together tell the real story: **Level 2 is the industry default for machine-to-machine APIs (Stripe, most of GitHub), partial hypermedia is a sensible middle (GitHub, paginated `next` links everywhere), full Level 3 is chosen deliberately for specific surfaces (PayPal) and is the native mode of human-driven hypermedia (the web).** Nobody who knows what they're doing climbs the ladder for its own sake.

## 11. Diagnosing your own API: a quick field guide

Before the takeaways, here's how to place an API on the ladder in under a minute, because the diagnosis is mechanical once you know what to look at.

- **Is there essentially one endpoint, with the operation named in the body, and does everything return `200`?** Level 0. (Tell: `POST /api`, `POST /rpc`, `POST /graphql` — yes, GraphQL is technically Level 0 on *this* ruler, single endpoint, dispatch by body; that's not an insult, GraphQL is a deliberate non-REST paradigm, but on Richardson's axis it sits at 0.)
- **Are there many resource URLs, but you `POST` to most of them and outcomes hide in the body?** Level 1. (Tell: pretty URLs, but `POST /orders/{id}/cancel` instead of `DELETE`, and `200` on failures.)
- **Do methods carry meaning (`GET` reads, `POST`/`PUT`/`PATCH`/`DELETE` write) and do status codes tell the truth (`201`, `404`, `409`, `422`, `429`)?** Level 2. This is the target.
- **Do responses include state-dependent links/affordances that clients are *expected and built* to follow?** Level 3 — and the operative word is *built to follow*. If the links are present but no client reads them, you've got Level 2 carrying decorative Level 3 cosmetics, which is cost without benefit. Either commit (build clients that follow) or drop the links.

The diagnosis matters because it tells you the *next available move* and its price. At Level 0, climbing to 1 and 2 is cheap and high-value — do it. At Level 2, climbing to 3 is expensive and situational — stop and justify it against the decision tree. Knowing where you are is how you stop arguing about purity and start arguing about cost and benefit, which is the only argument worth having.

## 12. Key takeaways

- **REST is a named set of constraints, not "JSON over HTTP."** The five that matter for API design are client-server, stateless, cacheable, uniform interface, and layered system. When someone says an API "isn't RESTful," ask *which constraint* it violates and *what property* is therefore lost; if they can't answer, it's taste, not architecture.
- **The Richardson Maturity Model is a ruler, not a scoreboard.** Level 0 is RPC-over-HTTP (the Swamp of POX), Level 1 adds resources, Level 2 adds HTTP verbs and honest status codes, Level 3 adds hypermedia. Higher is not automatically better; the right level is the one where the next rung costs more than it pays.
- **Level 2 captures the great majority of the value.** It's nearly free to reach from Level 1, and it unlocks the entire ecosystem of HTTP intermediaries — caching, conditional requests, safe retries on idempotent methods, gateways, honest monitoring — none of which requires the client to do anything REST-specific.
- **The method's safe/idempotent bits are the client's retry contract.** A generic client may auto-retry an idempotent request and must not auto-retry a non-idempotent one; Level 2 is what makes that contract expressible. For non-idempotent `POST`s, idempotency keys restore retry-safety at the application layer.
- **Status codes must tell the truth.** Returning `200` with `"ok": false` lies to every cache, proxy, and retry library in the path. Put the real outcome in the status line and use `problem+json` (RFC 9457) for machine-readable errors.
- **Statelessness buys horizontal scale.** Keep no per-client *session* state on individual servers; push it to the token or a shared store. ("Stateless" governs sessions, not your database — the server still owns resource state.)
- **HATEOAS (Level 3) is rarely worth it for typical machine-to-machine APIs, and that's fine.** Clients hard-code URLs anyway, the tooling is built for constructing URLs not following them, and the affordance vocabulary is still a breakable contract. The decoupling promise is real but requires client discipline you usually don't control.
- **Level 3 genuinely pays for human-driven clients (the web), long-lived public APIs with uncontrolled clients that must evolve URLs, and standardized hypermedia protocols.** Outside those, prefer pragmatic partial hypermedia: a `Location` header, a `next` link in pagination, a `self` link — the cheap, useful pieces — and skip the full affordance machine.
- **Diagnose before you argue.** One endpoint + body dispatch = 0; resources but `POST`-everything = 1; honest methods + status = 2; followed affordances = 3. Knowing where you are tells you the next move and its cost.
- **The whole point is the caller's contract.** Every rung is really about *what the caller and the intermediaries get to assume without reading your docs* — and that's the question this entire series keeps returning to: what can the caller assume, and can you change it later without breaking them?

## 13. Further reading

- **Roy T. Fielding, *Architectural Styles and the Design of Network-based Software Architectures* (2000)** — the doctoral dissertation that defines REST; Chapter 5 is the one to read. Also his 2008 blog post "REST APIs must be hypertext-driven," which states the strict HATEOAS position.
- **Martin Fowler, "Richardson Maturity Model" (martinfowler.com, 2010)** — the canonical write-up of the four levels, with the Swamp-of-POX framing; the direct companion to this post.
- **Leonard Richardson & Sam Ruby, *RESTful Web Services* (O'Reilly)** — the book behind the model; later expanded ideas appear in Richardson, Amundsen & Ruby, *RESTful Web APIs*, which goes deep on hypermedia and media types.
- **RFC 9110 — HTTP Semantics** — the authoritative source for safe and idempotent methods and status-code meanings; the foundation of everything at Level 2.
- **RFC 9457 — Problem Details for HTTP APIs** — the `application/problem+json` error format used in the Level 2 examples here.
- **Within this series:** the intro hub [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the sibling [resource modeling: turning a domain into nouns and URIs](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris); the sibling [HATEOAS in the real world: hypermedia links and when to skip](/blog/software-development/api-design/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip); and the capstone [the API design playbook: a review checklist from first endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- **Going wider / at scale:** [API design — REST vs gRPC vs GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) and [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) for the distributed-systems view of paradigm choice and contract evolution.

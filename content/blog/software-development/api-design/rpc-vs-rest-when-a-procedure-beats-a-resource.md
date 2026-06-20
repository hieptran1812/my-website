---
title: "RPC vs REST: When a Procedure Beats a Resource"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "An honest comparison of the two dominant API mindsets — REST modeling nouns you CRUD with uniform HTTP verbs versus RPC modeling verbs you call by name — with JSON-RPC 2.0 in depth, a gRPC and tRPC preview, and the Payments and Orders cancel action shown both ways."
tags:
  [
    "api-design",
    "api",
    "rest",
    "rpc",
    "json-rpc",
    "grpc",
    "http",
    "graphql",
    "design",
    "distributed-systems",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-1.png"
---

A few years ago I inherited a Payments service whose public API was, on paper, a clean REST design. `GET /orders/{id}`, `POST /orders`, `PATCH /orders/{id}` — textbook nouns and verbs. Then I went looking for how a client cancels an order, and the trail went cold. There was no `DELETE /orders/{id}` (deleting an order and cancelling one are not the same thing — a cancelled order still exists, still has a history, still shows up in reports). What I found instead was a `PATCH /orders/{id}` that accepted a body of `{"status": "cancelled"}`. A client would write the literal string `"cancelled"` into a status field and pray the server interpreted that as "run the cancellation workflow: release the inventory hold, reverse the authorization, fire the `order.cancelled` webhook, email the customer." It did not always interpret it that way. Some clients wrote `"canceled"` (one L) and got a silent no-op. Some set `"cancelled"` directly on an order that had already shipped, and the server happily flipped the field while the warehouse kept packing. The "resource" was lying. The real thing the caller wanted was not to edit a field — it was to *invoke an action*: `cancelOrder`. The API had bent a verb into a noun and the seam was leaking.

That tension — actions that don't map cleanly to "edit this resource" — is the oldest argument in API design, and it has a name on each side. **REST** models your system as **nouns**: resources (an order, a payment, a refund) that you create, read, update, and delete through a small, *uniform* set of HTTP verbs. **RPC** — Remote Procedure Call — models your system as **verbs**: named procedures (`cancelOrder`, `capturePayment`, `recalculatePricing`, `sendReceipt`) that you call by name, the way you'd call a function in code. Neither is "correct." They are two different mental models that buy you different things, and most real APIs you have ever used are quietly a blend of both. This post is the honest comparison: what each mindset is actually *for*, why REST's uniform interface buys you caching and intermediaries and a browser-friendly surface, why RPC buys you a natural fit for action-oriented domains, how to handle the action-doesn't-fit-a-noun problem without lying, and how the modern typed-RPC stacks (JSON-RPC, gRPC, tRPC) actually work on the wire. By the end you will be able to look at any operation and decide — deliberately, not by reflex — whether it should be a resource or a procedure, and you will know what you are trading away either direction.

![A two-column before and after comparison showing the same order cancellation modeled as a REST sub-resource POST that returns 201 Created versus an RPC call named cancelOrder posted to a single endpoint](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-1.png)

This is post E22 in the series, and it opens Track E — "Beyond REST." Everything before this assumed REST as the default; the next several posts (gRPC, GraphQL, events, streaming) are about the times REST is the wrong default. We start here because RPC is not an exotic alternative to REST — it is the *older* idea, the one REST was a reaction against, and understanding that reaction is the only way to choose well. If you want the foundational frame first, the series intro [what an API really is — a contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) sets up the "an API is a contract and a product, not a function call" thesis that this whole post is in tension with. (RPC's entire pitch is: *but what if it could feel like a function call?*) For the distributed-systems-scale view of the same trade-off, the system-design overview of [REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) is the OUT-link to read alongside this; we go deeper on the *mindset*, it goes deeper on the *operational* picture.

## 1. Two mental models: a noun you CRUD vs a verb you call

Start with the thing both styles are trying to do: let a client on one machine cause something to happen on another machine, over a network. That is the whole job. The two mindsets differ in what they make *primary*.

**RPC is the older and more obvious idea.** You already know how to make things happen in a program: you call a function. `cancel_order(order_id, reason)`. RPC asks, "what if calling a function on a remote machine looked exactly like calling one locally?" The procedure name is the star of the show. The client says, in effect, "run the procedure called `cancelOrder`, here are the arguments." The transport — how the bytes get there — is an implementation detail you'd rather not think about. This idea is decades old; Sun RPC (1980s), CORBA, Java RMI, XML-RPC, SOAP, and today's JSON-RPC and gRPC are all the same instinct: *make the network disappear behind a function call.*

**REST is a deliberate reaction to RPC**, articulated by Roy Fielding in his 2000 dissertation describing the architectural style of the Web itself. Fielding's argument was that the network does *not* disappear, and pretending it does (CORBA, RMI) produced brittle, tightly-coupled, un-cacheable systems. Instead of hiding the network, REST embraces it and copies what made the Web scale to billions of clients: a small set of *uniform* operations applied to *resources* identified by URLs. The noun is the star. You don't call `getUser(42)`; you `GET` the resource at `/users/42`. You don't call `deleteUser(42)`; you `DELETE` `/users/42`. The verbs are fixed and few — GET, POST, PUT, PATCH, DELETE — and their meaning is defined once, by the HTTP specification (RFC 9110), not per-endpoint by you.

It's worth dwelling on *why* "make the network disappear" is a leaky promise, because it's the deepest principled difference between the two camps. A local function call is fast (nanoseconds), reliable (it either runs or your process crashes), and has exactly two outcomes (returns or throws). A *remote* call is none of those: it's slow (milliseconds to seconds), unreliable (the network can drop the request, drop the response, or hang), and has *three* outcomes — success, failure, and **the unknown** (the dreaded timeout, where you cannot tell whether the server did the thing). This third outcome is the one that double-charges customers. The "8 fallacies of distributed computing" (the network is reliable, latency is zero, bandwidth is infinite, and so on) are a catalog of exactly the assumptions a too-transparent RPC layer tempts you to make. REST doesn't make remote calls reliable — nothing does — but by refusing to hide that they're remote (you *see* the HTTP method, the status code, the headers; you're forced to think about caching and retries and idempotency), it keeps the network's nature in your face. RPC's gift and its danger are the same thing: it makes the remote call *look local*, which is pleasant right up until a timeout reminds you it never was. Good modern RPC (gRPC with deadlines and explicit status codes; tRPC with typed errors) re-surfaces the network's reality on purpose; bad RPC pretends it away.

Let me define the load-bearing terms now, because the rest of the post leans on them.

- **Resource**: any thing worth naming — an order, a payment, a customer, a collection of refunds. In REST a resource has a stable identifier (a URL) and one or more *representations* (the JSON or other format you actually send over the wire).
- **Uniform interface**: REST's central constraint. The *same small verb set* means the same thing on *every* resource. `GET` is always a safe read; `DELETE` always removes. A client (or a proxy, or a CDN) that understands HTTP understands your API's verbs *without reading your docs*, because the verbs aren't yours — they're HTTP's.
- **Procedure / method (in RPC)**: a named action with arguments and a return value. The name is arbitrary and domain-specific — `cancelOrder`, `transfer`, `recalculatePricing`. There is no fixed vocabulary; you invent the verbs.
- **Endpoint**: a network address you send to. REST has *many* (one URL per resource). Classic RPC has *one* (everything posts to a single procedure-dispatch URL).

Here is the crux, and the whole post hangs on it. REST fixes the verbs and lets the nouns multiply: a small, universal verb set across an unbounded set of addressable resources. RPC fixes the address and lets the verbs multiply: one transport, an unbounded set of named procedures. That single inversion — *which thing is fixed and which is free* — is what produces every downstream difference: caching, discoverability, coupling, browser-friendliness, and the action-fit problem we'll spend the most time on.

### The same operation, both ways

Take cancelling an order on our Payments platform. In code, the domain logic is unambiguous: there is a function, `cancelOrder(orderId, reason)`, that runs a workflow. The question is only how to expose it on the wire.

The RPC framing keeps the function shape:

```http
POST /rpc HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer <token>

{
  "jsonrpc": "2.0",
  "method": "cancelOrder",
  "params": { "order_id": "ord_123", "reason": "customer_request" },
  "id": 1
}
```

The REST framing has to *find a noun* for the action. The cleanest answer is to model the cancellation itself as a resource — a thing that gets created:

```http
POST /orders/ord_123/cancellation HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer <token>

{ "reason": "customer_request" }
```

Read those two carefully. They do the *same thing* to the *same system*. But the RPC version names the verb (`cancelOrder`) and the REST version names a noun (`cancellation`) and lets the HTTP verb (`POST`, "create this") carry the action. The RPC version posts to one address (`/rpc`) and dispatches on the `method` field inside the body. The REST version posts to a *resource-specific* address (`/orders/ord_123/cancellation`) that a proxy can see, route, and reason about without parsing the body. Hold those two requests in your head; we'll return to them constantly.

## 2. What REST's uniform interface actually buys you

It's easy to wave at "REST is more standard" without saying what the standardization *purchases*. The uniform interface is not aesthetic. It buys three concrete, measurable things, and they all flow from one fact: **the meaning of a REST request lives in the method and the URL, where any intermediary can read it, not buried in the body.**

### Buy #1: HTTP caching, basically for free

Caching is the headline. The Web scales because a `GET` is, by HTTP's definition, *safe* (it doesn't change server state) and *cacheable*. A safe method is one the client may issue without intending to cause an effect; that property is what lets a cache, a CDN, or a browser store the response and serve it again without asking the origin. Because REST puts reads behind `GET` and uses URLs as cache keys, every layer between client and server can help.

Concretely: `GET /products/sku_88` with an `ETag` (an opaque version tag for the representation) lets the next client send `If-None-Match` and get back a `304 Not Modified` with an empty body when nothing changed — saving the entire payload transfer. A 200 KB product page over a cold mobile link is often 300–800 ms just in transfer; a 304 collapses that to a few bytes plus round-trip time. None of that requires the application to do anything clever. The cache infrastructure does it because it *understands* `GET` and `ETag`, which are HTTP's, not yours. (The full mechanics of `ETag`, `Cache-Control`, and conditional requests are their own post later in the series; the point here is only that REST *qualifies* for them.)

RPC gets none of this. Every JSON-RPC call is a `POST`, and `POST` is neither safe nor cacheable by default — a cache that tried to cache `POST /rpc` responses would be serving stale or wrong results, because two `POST`s to the same URL with different bodies mean different things. The cache can't even tell `getOrder` from `cancelOrder`; they're the same method and URL. So with RPC you either give up HTTP caching entirely or rebuild it yourself at the application layer.

### Buy #2: intermediaries and observability

Because the operation is visible in the method and URL, every box in the path can do its job by inspecting the request line alone. A gateway can rate-limit `POST /orders/*/cancellation` differently from `GET /products/*`. A WAF can apply stricter rules to writes. Access logs read like a story: `GET /orders/123`, `POST /orders/123/cancellation`, `GET /orders/123` — you can see what happened without decoding bodies. Routing, authorization, metrics, and tracing all key off the URL pattern.

With RPC, every line in the access log is `POST /rpc 200`. To know what actually happened you must parse each request body, extract the `method` field, and re-derive the semantics the URL would have given you for free. Tooling can do it, but you're now doing application-aware inspection at infrastructure layers, which is exactly the coupling REST was designed to avoid. This is why a service mesh or API gateway feels natural in front of REST and awkward in front of RPC.

### Buy #3: discoverability and a browser-friendly surface

Anyone can open `https://api.github.com/users/octocat` in a browser tab and read it. That's not a trick; it's the uniform interface plus URLs-as-identifiers. A new engineer can explore a REST API by curling URLs, guessing related ones (`/orders` → `/orders/123` → `/orders/123/refunds`), and learning the shape interactively. The structure is *legible*.

An RPC API is a list of method names you cannot guess and cannot poke at from a browser address bar (a browser tab issues a `GET`; RPC needs a `POST` with a body). You discover it by reading documentation or a generated client, not by exploring. That's not fatal — internal services live behind generated stubs and nobody opens them in a browser — but for a *public* API where unknown developers must self-serve, discoverability is a real product feature, and REST has it built in.

There's a subtler discoverability point that matters for *integration speed*. When a developer hits an unfamiliar REST API, their existing instincts transfer: they know `GET` is safe to try, they know a `404` means they got the URL wrong, they know `401` means auth, they know they can paginate a collection. None of that knowledge is API-specific — it's HTTP knowledge they already have, and your API inherits it for free. With an RPC API, the developer's HTTP instincts are useless (everything's `POST`, everything's `200`); they must learn *your* method catalog and *your* error-code scheme from scratch before they're productive. For an internal service consumed by a generated, typed client, that learning cost is paid once by the codegen and never by a human. For a public API consumed by thousands of strangers, you pay that cost thousands of times. This is the same "design for a caller you will never meet" principle from the [series intro](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems): the more strangers, the more REST's borrowed-from-HTTP familiarity is worth, and the less RPC's tight, learn-my-catalog coupling is tolerable.

![A four-row by two-column matrix comparing REST resources against RPC procedures across HTTP caching, action fit, coupling, and browser fit, with each cell marked as a strength, a cost, or a failure](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-2.png)

### The principle, stated rigorously

Here is the rule underneath all three, worth stating precisely because it's the only thing you need to remember.

> **A REST request's semantics are encoded in the method + URL; an RPC request's semantics are encoded in the body. Anything that can read the method and URL but cannot (or will not) parse the body can therefore help a REST request and cannot help an RPC request.**

Caches, CDNs, gateways, proxies, browsers, and logs all live at that boundary — they read the request line cheaply and the body expensively (or not at all). So everything they do for free — caching by URL, routing by path, rate-limiting by pattern, logging by line — is exactly the set of things REST buys and RPC forfeits. That's not a value judgment; it's a consequence of *where the meaning lives*. When you choose RPC, you are choosing to move the meaning into the body, and you should do it knowing you've made the intermediaries blind.

### Where RPC sits on the Richardson maturity model

There's a useful lens here from Leonard Richardson's maturity model, which grades how "RESTful" an HTTP API is across four levels (0–3). It's worth a paragraph because it pins RPC precisely. **Level 0** is "the swamp of POX (plain old XML)" — a single URI, a single HTTP method (almost always `POST`), and the actual operation buried in the body. That is *exactly* classic RPC over HTTP. SOAP, XML-RPC, and JSON-RPC all sit at Level 0 by this taxonomy: one endpoint, one method, the verb in the payload. **Level 1** introduces *resources* (many URIs, but still mostly one method). **Level 2** adds *HTTP verbs* used correctly (GET for reads, POST/PUT/PATCH/DELETE for writes) and *status codes* used honestly — this is where most production "REST" APIs actually live, and where the caching/routing/status benefits kick in. **Level 3** adds *hypermedia controls* (HATEOAS — links in responses that tell the client what it can do next), which few APIs reach and fewer need.

The point isn't that Level 0 is "bad" — it's that the model makes the trade *legible*. Climbing from 0 to 2 is precisely the process of moving meaning out of the body and into the method and URL, which is the same process of *qualifying for the Web's free infrastructure*. RPC deliberately stays at Level 0 and accepts the consequences in exchange for the procedure-call directness. So when someone says "JSON-RPC isn't RESTful," they're correct and it's not an insult — JSON-RPC isn't *trying* to be REST; it's at a different point on the same axis, on purpose. (The series covers this taxonomy in full in [the Richardson maturity model and what RESTful means](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means).)

## 3. What RPC buys you: a natural fit for actions

If REST is so well-supported by the whole Web stack, why does anyone reach for RPC? Because a large class of operations *are verbs*, and forcing them into nouns produces APIs that are awkward to design, awkward to call, and awkward to reason about. RPC's purchase is alignment: when the domain is a list of actions, an API that is a list of actions is the honest model.

Look at the operations a Payments service actually needs to expose:

- `transfer(from, to, amount)`
- `capturePayment(payment_id)`
- `refund(payment_id, amount)`
- `cancelOrder(order_id, reason)`
- `recalculatePricing(cart_id)`
- `sendReceipt(order_id, email)`
- `retryWebhook(delivery_id)`

Every one of those is a *verb with arguments and a result*. None of them is naturally "create/read/update/delete a thing." What resource does `recalculatePricing` operate on — does it create a "pricing-calculation" resource? What about `sendReceipt` — is the receipt a resource you `POST`, and if the send fails, did you create it or not? You *can* answer these questions (we will, in the next section), but notice that you're now doing *translation work*: taking a clear verb and inventing a noun to host it. RPC skips the translation. The method *is* the verb. `capturePayment` is just `capturePayment`. There is no impedance mismatch because there is no mismatch — the wire shape matches the domain shape matches the code shape.

This alignment compounds in three ways.

**It matches your code.** Your service almost certainly has a `PaymentService` class with a `capture()` method. With RPC, the API surface and the internal interface are nearly the same shape, so the mapping is trivial and codegen is natural (define the procedure once in an interface description, generate both client and server stubs). With REST you maintain a translation layer that maps HTTP verbs and URL paths onto method calls, and that layer is where a lot of subtle bugs live (the `PATCH {"status": "cancelled"}` disaster from the intro is exactly that translation layer leaking).

**It matches the caller's intent.** A client integrating your Payments API thinks in actions: "I want to capture this payment." Reaching for "I should `POST` to the captures sub-collection of the payment resource to create a capture" is a longer path to the same place, and it's a path that requires the caller to learn *your* resource model before they can do the obvious thing. RPC lets them call the verb they already have in mind.

**It's honest about non-CRUD operations.** Some actions genuinely have no resource. "Recalculate pricing and tell me the new total without saving anything" is a pure computation — a function — with no resource created, updated, or deleted. Modeling it as a resource is a fiction. RPC names it for what it is: a procedure.

### The trade you're accepting

None of this is free. By moving the verb into the body you forfeit everything section 2 listed — caching, intermediary help, browser-friendliness, the self-describing surface. You also accept a *tighter coupling*: client and server now share a contract of method names and argument shapes that isn't anchored to anything external (no universal verb vocabulary, no standard status semantics). When that contract is generated from a single schema and both sides are *your* services, that coupling is cheap and even desirable. When the caller is a stranger you'll never meet, it's a liability. Hold that thought — it's the whole "when to" recommendation at the end.

![A decision tree that routes an action to a sub-resource when it changes one resource state, to a controller resource when it is a nameable process, or to an honest RPC call when it fits no noun](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-3.png)

## 4. The action-doesn't-fit-a-noun problem (and the three honest fixes)

This is the section that earns the post. Every REST API hits operations that resist nouns, and how you handle them is the difference between a clean API and a leaky one. There are exactly three honest moves, plus one dishonest one to avoid. The decision tree above is the map; let's walk it.

### Fix #1: model the action's *result* as a sub-resource

The best move, when it works, is to notice that the action produces a *thing* and model that thing. Cancelling an order produces a *cancellation*. Refunding a payment produces a *refund*. Capturing a payment produces a *capture*. These are real entities — they have a timestamp, an actor, a reason, an outcome — so they deserve to be resources.

```http
POST /orders/ord_123/cancellation HTTP/1.1
Content-Type: application/json

{ "reason": "customer_request" }
```

```http
HTTP/1.1 201 Created
Location: /orders/ord_123/cancellation
Content-Type: application/json

{
  "id": "cnl_456",
  "order_id": "ord_123",
  "reason": "customer_request",
  "status": "completed",
  "created_at": "2026-06-20T10:15:00Z"
}
```

This is genuinely RESTful. `POST` means "create a subordinate resource," the `201 Created` plus `Location` tells the client where the new cancellation lives, and you can later `GET /orders/ord_123/cancellation` to see it. The verb `cancel` got expressed as "create a cancellation," and crucially that's not a fiction — there really is a cancellation record. This is the move to reach for first, and it's what we'll show as a full worked example against the RPC version below.

When does it work? When the action *creates state you'd want to read back*. Cancellations, refunds, captures, shipments, payment attempts — all leave a record. Model the record.

### Fix #2: a controller resource for processes

Sometimes the action is a *process* you can name but it isn't subordinate to one specific resource, or it doesn't fit "create a thing." The convention (Google's API design guide calls these "custom methods"; others call them "controller resources") is a noun-ified verb at the collection level:

```http
POST /carts/cart_99/recalculation HTTP/1.1
Content-Type: application/json

{ "coupon_code": "SUMMER20" }
```

Or, when even that feels forced, the colon-suffixed custom method that Google's AIP guidelines endorse precisely because it *admits the action doesn't fit CRUD* while keeping it under the resource's URL space:

```http
POST /carts/cart_99:recalculate HTTP/1.1
Content-Type: application/json

{ "coupon_code": "SUMMER20" }
```

The `:recalculate` suffix is a small, honest concession: it says "this is an action on this resource, not a CRUD operation," but it keeps the resource-oriented URL structure so caching/routing/logging still partially work (the path is still resource-anchored). This is the middle ground — REST-shaped URLs hosting verb-shaped operations.

Google's AIP guidance is worth internalizing here because it gives a *rule* for when to reach for a custom method rather than improvising. Their position: prefer standard methods (the CRUD verbs on resources) whenever you reasonably can, and use a custom method only when the operation genuinely doesn't map to one — and even then, keep it under the resource's URL with the `:` separator so the resource hierarchy stays intact. The colon (rather than a `/`) is deliberate: a slash would imply `recalculate` is a *sub-resource* you could `GET` or `DELETE`, which it isn't — it's an action, and the colon signals exactly that without lying about it being a noun. Examples from real Google APIs include `POST .../instances/{id}:start`, `:stop`, `:reset` — server actions that don't create a readable record and so don't deserve a sub-resource, but also shouldn't be smuggled into a field edit. The custom method is the disciplined name for "this is RPC, but I'm keeping it inside my REST URL space so my tooling still mostly works." It's the most pragmatic point on the whole spectrum and the one I reach for most in production.

### Fix #3: admit it's RPC

The third move is the one engineers resist most and shouldn't: *just call it RPC.* When an operation has no resource, creates no record, and isn't subordinate to anything — a pure computation, a side-effecting command with no readable result — stop torturing the noun. Expose it as a procedure.

```http
POST /rpc HTTP/1.1
Content-Type: application/json

{ "jsonrpc": "2.0", "method": "estimateShipping",
  "params": { "zip": "94103", "weight_g": 1200 }, "id": 7 }
```

`estimateShipping` creates nothing, reads nothing, changes nothing — it computes. There is no resource. Inventing `POST /shipping-estimates` to "create an estimate" you immediately discard is a lie that costs the next engineer a confused hour. Naming it RPC is the truthful design. The skill here is not "REST good, RPC bad"; it's recognizing *which* operations earn a resource and which don't, and being honest about the rest.

### The dishonest move: a verb inside a resource body

The thing to *never* do — the thing my inherited Payments service did — is fake it by stuffing a verb into a resource's fields:

```http
PATCH /orders/ord_123 HTTP/1.1
Content-Type: application/json

{ "status": "cancelled" }
```

This looks RESTful (`PATCH` a resource) but it's the worst of both worlds. It's secretly RPC (the real operation is "run the cancellation workflow") but disguised as a field edit, so:

- The client can set `status` to an *invalid transition* (cancel a shipped order) and the API has to validate domain rules inside a generic field update — easy to forget.
- There's no record of *why* or *who* cancelled (the `PATCH` just flips a field; the structured `reason`, `actor`, `timestamp` of a cancellation resource are gone).
- You can't distinguish "the customer cancelled" from "an admin force-cancelled" from "the fraud system auto-cancelled" — they're all the same field write.
- Idempotency and retries are murky: is `PATCH {"status":"cancelled"}` on an already-cancelled order a no-op, an error, or a re-run of the side effects (re-sending the email, re-reversing the auth)?

A `POST /orders/ord_123/cancellation` resource or an honest `cancelOrder` RPC both solve all of these; the disguised field-edit solves none of them. The lesson: if the operation is really a verb, *say so* — either as a sub-resource (Fix #1) or as RPC (Fix #3). Don't smuggle it into a noun's body.

#### Worked example: the cancel action as REST sub-resource vs RPC

Let's do the same operation both ways, end to end, with the full request/response and a retry, so the differences are concrete rather than abstract.

**As a REST sub-resource.** The client creates a cancellation. To make the retry safe (a cancellation has side effects — it must not run twice on a network retry), it sends an idempotency key:

```http
POST /orders/ord_123/cancellation HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Idempotency-Key: 7c2f1a9e-cancel-ord_123
Content-Type: application/json

{ "reason": "customer_request" }
```

```http
HTTP/1.1 201 Created
Location: /orders/ord_123/cancellation
Idempotency-Key: 7c2f1a9e-cancel-ord_123
Content-Type: application/json

{
  "id": "cnl_456",
  "order_id": "ord_123",
  "status": "completed",
  "reason": "customer_request",
  "created_at": "2026-06-20T10:15:00Z"
}
```

If the client times out and retries with the *same* `Idempotency-Key`, the server recognizes the key, does **not** re-run the workflow, and replays the original `201` with the same `cnl_456`. The status code carries meaning: `201` (created), or `409 Conflict` if the order already shipped and cannot be cancelled, or `404` if the order doesn't exist. A cache and gateway see a write to a resource-specific URL and route/log it accordingly. (Idempotency keys are their own deep dive in [idempotency keys, safe retries, and the exactly-once illusion](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions); here, note only that REST hangs the key on a standard header.)

**As an RPC call.** The same operation, named directly:

```http
POST /rpc HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "cancelOrder",
  "params": { "order_id": "ord_123", "reason": "customer_request" },
  "id": 1
}
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "result": {
    "cancellation_id": "cnl_456",
    "order_id": "ord_123",
    "status": "completed"
  },
  "id": 1
}
```

Notice three things. First, the transport status is `200 OK` *even though* the operation might have failed — RPC carries success/failure in the *body* (a `result` field vs an `error` field), not the HTTP status. If the order can't be cancelled, you get `200 OK` with an `error` object, not a `409`. Second, there's no `Location`, no resource URL, nothing a cache can key on — the call is opaque. Third, idempotency isn't standardized: there's no `Idempotency-Key` convention in JSON-RPC, so you either bake an idempotency token into `params` or rely on the `id` (and the spec doesn't promise the server uses `id` for dedup — it's only for correlation). The RPC version is *shorter to call and reason about as a function*, and *weaker on everything the HTTP stack would have given the REST version*. That's the trade in one screen.

## 5. JSON-RPC 2.0 in depth

If you're going to do RPC over HTTP without a heavy framework, JSON-RPC 2.0 is the lingua franca. It's a tiny, transport-agnostic specification (it works over HTTP, WebSocket, raw TCP, stdio) that standardizes exactly one thing: the envelope. Ethereum and Bitcoin nodes speak it, the Language Server Protocol that powers your editor's autocomplete speaks it, and many internal tools use it because it's trivial to implement. Let's take it apart.

### The request envelope

A JSON-RPC request is an object with up to four fields:

```json
{
  "jsonrpc": "2.0",
  "method": "capturePayment",
  "params": { "payment_id": "pay_789", "amount": 4999, "currency": "USD" },
  "id": 1
}
```

- **`jsonrpc`** — the literal string `"2.0"`. It pins the protocol version so the server knows which rules apply. Always present.
- **`method`** — the procedure name. This is the *verb*, the whole point. A string. By convention, names beginning `rpc.` are reserved for the protocol itself.
- **`params`** — the arguments. Either an *object* (by-name: `{"payment_id": "pay_789"}`) or an *array* (by-position: `["pay_789", 4999]`). By-name is far more maintainable — adding a param doesn't shift positions — so prefer objects. `params` is optional (a procedure may take no arguments).

The by-name versus by-position choice in `params` deserves a hard line, because it's a real source of breakage. With positional `params` like `["pay_789", 4999, "USD"]`, the contract is *the order*. The day you need to insert an `idempotency_token` between `payment_id` and `amount`, every existing client breaks silently — they're still sending the amount in slot 2, which now means something else, and there's no error, just a wrong charge. With named `params` like `{"payment_id": "pay_789", "amount": 4999, "currency": "USD"}`, you add `"idempotency_token": "..."` as a new key and nothing existing shifts; old clients omit it, new clients send it. This is the *exact same* additive-compatibility principle that governs REST response fields and protobuf field numbers — additive is safe, reordering is breaking — showing up in yet another form. The rule generalizes: never make *position* part of your contract when you can make *name* part of it instead. (Bitcoin Core's RPC, for historical reasons, uses positional params, which is precisely why its client libraries are finicky about argument order; Ethereum's JSON-RPC is also largely positional. Newer designs default to named.)
- **`id`** — a client-chosen correlation handle (a number or string). It is *not* the procedure's identity and not an idempotency key; it exists so that when responses come back — especially out of order, or batched — the client can match each response to the request that produced it. The server must echo it back unchanged.

Note the amount: I wrote `4999`, meaning \$49.99 expressed in minor units (cents) as an integer. Money in floating point is a classic API bug; representing a \$49.99 charge as the integer `4999` cents avoids it. That's not JSON-RPC-specific, but it's the kind of contract decision the envelope is agnostic about — JSON-RPC standardizes the *frame*, you still design the `params`.

![A vertical stack showing the four JSON-RPC envelope fields jsonrpc, method, params, and id, plus the notification case where omitting id means the server sends no reply](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-4.png)

### The response envelope

A response echoes the `id` and carries *either* `result` *or* `error` — never both:

```json
{
  "jsonrpc": "2.0",
  "result": { "capture_id": "cap_001", "status": "succeeded" },
  "id": 1
}
```

On failure, `result` is absent and `error` is present:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": { "field": "amount", "reason": "must be a positive integer" }
  },
  "id": 1
}
```

The `error` object is itself standardized: `code` (an integer), `message` (a short human string), and optional `data` (anything you like — the structured detail). The spec reserves a range of codes:

| Code | Meaning | Analogous HTTP |
| --- | --- | --- |
| `-32700` | Parse error (invalid JSON) | 400 |
| `-32600` | Invalid Request (not a valid envelope) | 400 |
| `-32601` | Method not found | 404 |
| `-32602` | Invalid params | 400 / 422 |
| `-32603` | Internal error | 500 |
| `-32000` to `-32099` | Reserved for server-defined errors | (your domain errors) |

That last row matters: your *application* errors (insufficient funds, order already shipped) go in the `-32000..-32099` server-error band, with the human-friendly specifics in `data`. Compare this to REST, where those would be distinct HTTP status codes (`402`, `409`) that intermediaries understand. With JSON-RPC, *every* response — success or domain error or crash — is HTTP `200 OK`, and the real outcome is the `error.code` in the body. A monitoring system counting HTTP 5xx will see zero errors even as every call fails. That's a real operational gotcha: you must build error observability at the application layer.

### Notifications: fire-and-forget

If a request omits the `id` field entirely, it's a **notification** — the server processes it but sends *no response at all*. This is for genuine fire-and-forget signals (`logEvent`, `heartbeat`) where the client doesn't need an acknowledgment. It's a small but real feature REST has no direct equivalent for (the closest is `202 Accepted`, which still returns a response).

```json
{ "jsonrpc": "2.0", "method": "logClientEvent", "params": { "event": "page_view" } }
```

No `id`, so no reply comes back. Use it sparingly — you lose all confirmation, including confirmation the call was even valid.

### Batch calls

This is JSON-RPC's nicest feature and the one with no clean REST analog: send an *array* of requests in one HTTP POST and get an array of responses back. It collapses N round-trips into one.

#### Worked example: a JSON-RPC batch round-trip

The client needs to capture a payment, fetch the order, and log an event — three operations. Instead of three HTTP requests (three round-trips, three TLS-amortized but still serial latencies), it batches:

```http
POST /rpc HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

[
  { "jsonrpc": "2.0", "method": "capturePayment",
    "params": { "payment_id": "pay_789", "amount": 4999 }, "id": 1 },
  { "jsonrpc": "2.0", "method": "getOrder",
    "params": { "order_id": "ord_123" }, "id": 2 },
  { "jsonrpc": "2.0", "method": "logClientEvent",
    "params": { "event": "checkout_complete" } }
]
```

The server processes all three and returns an array. Two critical details: (1) the third call is a *notification* (no `id`), so it produces **no entry** in the response array — only the two calls with ids get responses; (2) the responses **may come back in any order**, so the client *must* match by `id`, never by array position:

```json
[
  { "jsonrpc": "2.0", "result": { "order_id": "ord_123", "status": "paid" }, "id": 2 },
  { "jsonrpc": "2.0", "result": { "capture_id": "cap_001", "status": "succeeded" }, "id": 1 }
]
```

See how the response array is `[id 2, id 1]` — reordered — and the notification (`logClientEvent`) is simply absent. The `id` field is doing exactly its job: it's the only reliable way to know which result is the capture and which is the order. If the whole batch is malformed JSON, the server returns a *single* error object (not an array). The win is concrete: one round-trip instead of three. On a 100 ms RTT link, that's roughly 100 ms total instead of ~300 ms — a 3x latency cut for this interaction. The cost: the batch is one HTTP request, so a gateway can't rate-limit or cache the three operations independently, and a partial failure (capture succeeds, order-fetch 500s) returns a mixed array your client must inspect element by element.

![A left-to-right timeline of a JSON-RPC batch where the client builds three calls, posts them once, the server runs each, and the array reply may reorder so the client matches by id](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-5.png)

## 6. RPC over HTTP: one endpoint vs many URLs

Step back and look at the shape of traffic each style produces, because it explains the operational feel of working with each.

A REST API is a *map* of URLs. `GET /orders/123`, `POST /orders/123/cancellation`, `GET /products/88`, `DELETE /carts/99/items/3` — each operation is a distinct (method, path) pair, and the path is a meaningful, hierarchical address. A request's destination *is* its meaning.

A classic RPC API is a *funnel*. Everything is `POST /rpc` (or whatever the single dispatch endpoint is named). The destination is constant; the meaning is entirely in the `method` field inside the body. The server reads the body, looks up `method` in a dispatch table, and calls the matching procedure.

![A branching acyclic graph showing a client whose REST calls pass through a shared cache to per-resource URLs while its RPC calls funnel through one endpoint that dispatches on a body field](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-6.png)

That difference is why the two feel so different to operate, and it's worth being precise about what the funnel forfeits.

**You lose HTTP caching.** Covered already, but it bears the operational restatement: there is no per-operation cache key, because the URL is constant and `POST` isn't cacheable. A read-heavy RPC API can't lean on CDNs or browser caches; every read hits your origin. A read-heavy REST API can serve a large fraction of reads from caches it never wrote.

**You lose status-code semantics.** REST's status codes are a shared vocabulary: `404` means not-found *everywhere*, so a client's generic retry logic, a gateway's circuit breaker, and a dashboard's error rate all understand them without your docs. RPC's transport is almost always `200 OK`; the real outcome hides in the body. You can rebuild a status taxonomy in your error codes (and JSON-RPC does), but it's *yours*, not shared, so generic tooling can't reason about it. The concrete failure: a load balancer configured to take an instance out of rotation on a spike of HTTP 5xx will *never* trip for a JSON-RPC service returning `200 OK` with `error` bodies, even if every single call is failing. (For how REST's codes are meant to be chosen honestly, see [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).)

**You lose browser-friendliness.** A browser address bar issues a `GET`. You cannot explore an RPC API from one, cannot link to a specific resource, cannot bookmark a result. For an internal service called only by code, irrelevant. For a public API, a real loss of discoverability.

**You gain a clean dispatch model and tight code alignment.** The funnel is *easy to implement* — one route, one dispatch table, one place to add auth and validation. And because the wire is method-name + params, codegen from a single schema is natural: define the service once, generate client and server. That's exactly why typed RPC frameworks (next section) thrive internally.

Here is the trade in one table:

| Concern | REST (many URLs) | RPC (one endpoint) |
| --- | --- | --- |
| Cache key | URL + method, free | none (`POST` not cacheable) |
| Routing | by path pattern | by body `method` field |
| Status semantics | HTTP codes, shared vocabulary | `200 OK` + body error code |
| Browser exploration | native (`GET` a URL) | not possible |
| Implementation | route table per resource | one route + dispatch table |
| Code-to-wire mapping | translation layer | near-direct (codegen-friendly) |

### Latency, coupling, and versioning differences

Three more axes that decide real designs:

**Latency.** RPC isn't intrinsically faster over plain JSON-over-HTTP — a JSON-RPC call and a REST call carry similar bytes. The latency wins come from (a) *batching* (one round-trip for many ops, as we measured: ~100 ms vs ~300 ms for three calls), and (b) *binary, multiplexed* transports like gRPC's protobuf-over-HTTP/2, which shrink payloads and avoid head-of-line blocking. REST *can* lose latency to caching wins it gets and RPC doesn't — a cached `304` is the fastest response of all. So "which is faster" depends entirely on whether the workload is batch-y/internal (RPC wins) or read-heavy/cacheable (REST wins).

**Coupling.** REST clients couple to a *uniform* interface plus your resource shapes; a tolerant-reader client (one that ignores fields it doesn't recognize) survives most additive changes. RPC clients couple to your *method signatures* — names and param shapes — which is tighter. When both sides are yours and generated from one schema, tight coupling is fine and even safer (the compiler catches mismatches). When the client is a stranger, tight coupling means a renamed method or reordered positional param breaks them silently.

**Versioning.** REST has several version strategies (URI `/v2`, media-type, header) covered in [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning). RPC versioning is usually done by *adding* methods (`cancelOrderV2`) or evolving the schema additively — and gRPC/protobuf has *excellent* additive-evolution rules (add fields with new numbers, never reuse or renumber). The compatibility *rules* are the same underneath (additive is safe, removal/rename is breaking — see [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change)); the *mechanisms* differ.

#### Worked example: adding a field, REST vs RPC

Make versioning concrete. Suppose we need to add a `cancelled_by` field to the cancellation flow — who triggered the cancel (customer, admin, fraud-system). Classify the change both ways.

**REST sub-resource.** We add `cancelled_by` to the *response* of `POST /orders/{id}/cancellation`:

```json
{
  "id": "cnl_456",
  "order_id": "ord_123",
  "status": "completed",
  "reason": "customer_request",
  "cancelled_by": "customer",
  "created_at": "2026-06-20T10:15:00Z"
}
```

Adding an *optional response field* is **non-breaking** by the tolerant-reader principle: a well-behaved client ignores fields it doesn't recognize, so old clients keep working and new clients can read the new field. No version bump needed. If instead we made `cancelled_by` a *required request* field, that *would* be breaking — old clients omitting it would now get a `400` — and we'd need a version strategy or a default. The status-code vocabulary helps here too: a client that sends an unknown enum value for `cancelled_by` gets a `422 Unprocessable Content`, which any client's validation-error handling already understands.

**gRPC/RPC.** We add the field to the protobuf message with a *new field number*:

```protobuf
message CancelOrderResponse {
  string cancellation_id = 1;
  string status = 2;
  string cancelled_by = 3;   // new — never reuse a retired number
}
```

This is non-breaking for the same reason, but the *mechanism* is the field number, not the field name: old clients decoding the response simply skip the unknown field-number-3 bytes. Protobuf's additive-evolution discipline ("add with a new number, never renumber, never reuse a deleted number") makes this airtight at the binary level — there's no string-name matching to drift. The one trap unique to RPC: if you *rename* a method (`cancelOrder` → `cancelOrderV2`) you've created a hard fork, because the method name *is* the routing key. With REST, renaming a resource path is the equivalent break. So the rules rhyme — additive safe, rename/remove breaking — but you reason about *fields and method names* in RPC where you reason about *fields, paths, and status codes* in REST. The deeper treatment of the field lifecycle is in [schema evolution — adding, removing, and renaming fields safely](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely), and the contract-test machinery that catches an accidental break before it ships is in [contract testing and schema diffs](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs).

### Stress-testing the choice: what happens when…

A design decision is only as good as its behavior under the bad cases. Let's take a concrete decision — "we'll expose the cancel operation as an RPC `cancelOrder` because our domain is action-heavy and the callers are our own services" — and stress it the way a senior reviewer would, then ask whether REST would have answered differently.

**The client retries on a timeout.** The network drops the response but the server already ran the cancellation (released inventory, reversed the auth, sent the email). The client, seeing no reply, retries `cancelOrder` with the same `params`. With naive RPC, the workflow runs *twice* — a second email, a double inventory release, a confused customer. JSON-RPC's `id` does *not* save you: the spec says `id` is for *correlation*, not deduplication, and the server is free to ignore it for that purpose. So you must add an idempotency token into `params` and dedup on it server-side — you're rebuilding, at the application layer, the `Idempotency-Key` header that REST standardizes. *REST's answer*: the sub-resource `POST /orders/{id}/cancellation` carries an `Idempotency-Key` header, and the server returns the cached `201` on retry. Verdict: REST has the conventional answer; RPC works but you build the convention yourself. Either way, the *underlying* property — a cancel must be safe to retry — is non-negotiable; see [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).

**Two writers race.** Two services both call `cancelOrder("ord_123")` within milliseconds. Without a guard, both pass the "is it cancellable?" check, both run the workflow. This is *not* an RPC-vs-REST distinction — it's a concurrency problem both styles share — but the *tools* differ: REST can use a conditional request (`If-Match` against an `ETag`) so the second writer gets a `412 Precondition Failed` and backs off; RPC has no conditional-request convention, so you implement the version check inside the procedure (compare-and-set on an order version column). REST hands you a standard optimistic-concurrency mechanism; RPC makes you roll it.

**The result has to be readable later.** Six months on, support needs to see *who* cancelled `ord_123` and *why*. If we modeled it as RPC `cancelOrder` and stored nothing readable, there's no resource to `GET`. We'd have to add a *separate* `getCancellation` method. The REST sub-resource gave us `GET /orders/{id}/cancellation` for free — the act of creating a resource left a readable record. This is the strongest argument for Fix #1 (sub-resource) over Fix #3 (frank RPC): *if the action leaves state someone will want to read, model it as a resource.* The cancel action does; that's why, even in an action-heavy internal service, I'd lean toward the sub-resource here and reserve frank RPC for the truly record-less operations like `estimateShipping`.

**The caller turns out not to be internal.** The decision rested on "callers are our own services." A year later, a partner integration needs `cancelOrder`. Now the tight coupling bites: the partner pins to your exact method name and `params` shape, you can't rename without coordinating, and they can't explore the API from a browser or curl it casually. If you'd chosen the REST sub-resource, the partner would already have a self-describing, cacheable, status-code-honest surface. Verdict: the *audience assumption* is the load-bearing part of an RPC choice. When in doubt about whether a caller will ever be a stranger, the conservative move is REST — you can always add internal RPC fast paths later, but you can't easily un-tightly-couple a public RPC surface.

The pattern across all four: RPC is fine *when the assumptions hold* (internal, action-heavy, record-less, retry-handled in-band), and each failed assumption is a place where REST's standardized conventions would have caught you. That's the honest summary — not "REST wins," but "REST front-loads the conventions; RPC makes you supply them, which is cheap internally and expensive at the public edge."

## 7. Modern typed RPC: gRPC and tRPC (a preview)

JSON-RPC is RPC stripped to its envelope. The frameworks people reach for in 2026 add the thing JSON-RPC deliberately omits: a *typed contract* with code generation, so client and server can't drift. The next post is a full deep-dive on gRPC; here's the preview that situates it in the RPC-vs-REST story.

### gRPC: typed RPC at Google scale

gRPC is Google's open-source RPC framework, and it is RPC taken seriously. You define the service contract once, in a `.proto` file using Protocol Buffers (a binary serialization format), and gRPC generates strongly-typed client and server stubs in a dozen languages.

```protobuf
syntax = "proto3";
package payments.v1;

service OrderService {
  rpc CancelOrder(CancelOrderRequest) returns (CancelOrderResponse);
  rpc CapturePayment(CapturePaymentRequest) returns (CapturePaymentResponse);
}

message CancelOrderRequest {
  string order_id = 1;
  string reason = 2;
}

message CancelOrderResponse {
  string cancellation_id = 1;
  string status = 2;
}
```

Notice the field *numbers* (`= 1`, `= 2`). Protobuf encodes by number, not name, which is what gives it its bulletproof additive evolution: add a `string actor = 3;` and old clients that don't know field 3 simply skip it; the rule is *never reuse or renumber a field*. The verb is front and center (`rpc CancelOrder(...)`) — pure RPC — but now it's *typed*, so a client calling `CancelOrder` with a wrong field type doesn't compile. gRPC rides HTTP/2 (multiplexed, binary), so it's compact and low-latency, and it adds four call shapes: unary (one request, one response, like a normal function), server streaming, client streaming, and bidirectional streaming — RPC's answer to "what about a stream of results?" It's the default for *internal* service-to-service calls at scale precisely because the audience is your own services (tight coupling is fine), latency matters, and codegen eliminates the translation layer. (Full treatment: [gRPC and Protocol Buffers — contracts, codegen, and streaming](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming).)

What gRPC gives up is exactly REST's list: it's not browser-native (browsers can't speak raw gRPC; you need grpc-web or a gateway), not human-curl-able, not cacheable by HTTP intermediaries. The trade is sharp: maximal typing, performance, and codegen, at the cost of the open-Web friendliness.

### tRPC: typed RPC for the TypeScript monolith

tRPC is a very different animal that makes the same RPC bet. It's a TypeScript library for full-stack apps where *the same team owns the client and the server in one codebase*. You define procedures on the server as plain TypeScript functions, and the client calls them with *full type inference* — no schema file, no codegen step, no `.proto`. If you rename a procedure's input on the server, the client code stops type-checking *immediately*, in your editor.

```javascript
// server: define a procedure
const appRouter = router({
  cancelOrder: publicProcedure
    .input(z.object({ orderId: z.string(), reason: z.string() }))
    .mutation(({ input }) => orderService.cancel(input.orderId, input.reason)),
});

// client: call it like a local async function, fully typed
const result = await trpc.cancelOrder.mutate({ orderId: "ord_123", reason: "customer_request" });
```

That's RPC's original dream — "make the network look like a function call" — finally delivered cleanly, *because* the constraint that made classic RPC dangerous (a stranger on the other end) is removed: it's your own code on both sides. tRPC is *not* for public APIs (it's TypeScript-to-TypeScript; a Python client can't consume it without an OpenAPI bridge). It's the modern proof that RPC's coupling cost is only a cost when the caller is a stranger — when you own both ends, RPC's directness is pure upside.

| Style | Transport | Contract | Best fit |
| --- | --- | --- | --- |
| REST | HTTP verbs + URLs | OpenAPI (optional) | public CRUD, cacheable reads |
| JSON-RPC | one POST endpoint | JSON envelope | node tooling, simple internal RPC |
| gRPC | HTTP/2 + protobuf | `.proto`, strict typed, codegen | internal low-latency service-to-service |
| GraphQL | one POST + query language | SDL, strict typed | client-shaped reads, many clients |
| tRPC | HTTP (TS only) | inferred TS types | full-stack TypeScript monorepo |

![A four-row by three-column matrix mapping REST, JSON-RPC, gRPC, and GraphQL onto their transport, contract style, and the workload each fits best](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-7.png)

A word on GraphQL since it's in the table: GraphQL is *also* technically RPC-shaped (one `POST /graphql` endpoint, the operation in the body), but it adds a query language so the *client* shapes the response. It's the third leg of the "beyond REST" stool and gets its own post — [GraphQL, the query language, the schema, and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap). For now, slot it as "RPC's one-endpoint shape, plus client-driven field selection."

## 8. The hybrid reality: most REST APIs have RPC in them

Here's the honest truth the dogma rarely admits: **almost every "RESTful" API you've ever used has RPC-style action endpoints, and that's completely fine.** Pure REST — every operation a CRUD on a noun, full HATEOAS hypermedia — is rare in the wild because real domains have actions that don't fit. The mature move isn't to pick a team; it's to be REST by default and RPC where the action earns it.

Look at APIs you trust:

- **Stripe** is widely cited as a model REST API, and it *mostly* is — resources like `/v1/charges`, `/v1/customers`, `/v1/refunds`. But it also has action endpoints that are RPC in spirit: `POST /v1/payment_intents/{id}/capture`, `POST /v1/payment_intents/{id}/cancel`, `POST /v1/charges/{id}/capture`. Those are verbs (`capture`, `cancel`) appended to a resource path — exactly the controller-resource / custom-method pattern from section 4. Stripe didn't pretend `capture` was a field edit; they exposed it as an action under the resource. (Stripe also pioneered the `Idempotency-Key` header for safe retries, which is how their action endpoints stay retry-safe — a REST-native solution to RPC's idempotency gap.)
- **GitHub** has a resource-rich REST API *and* a GraphQL API, side by side, because different clients have different forces — the GraphQL endpoint exists precisely for clients that need to shape their own responses and avoid REST's over-fetching.
- **Slack's Web API** is, candidly, mostly RPC: methods like `chat.postMessage`, `conversations.create`, `users.info` — verb-ish method names, all called via `POST` (or `GET`) to `https://slack.com/api/<method>`. It's an action-oriented domain ("post a message," "invite a user") and Slack modeled it as actions. Nobody calls Slack's API "wrong"; they called it *honest about its shape*.

So the design instinct to cultivate is not purity but *fit per operation*. Model your stable entities as resources (REST gives you caching, discoverability, and a legible URL map for free). When you hit a true action, prefer a sub-resource (Fix #1), fall back to a controller resource or custom method (Fix #2), and use frank RPC only when nothing else is honest (Fix #3). A REST API with a handful of `:action` endpoints is not a failure of discipline — it's discipline applied operation by operation.

#### Worked example: classifying a Payments API's operations

Take our platform and sort every operation by the move it deserves — this is the exercise to run on your own API:

| Operation | Best move | Why |
| --- | --- | --- |
| Read an order | `GET /orders/{id}` | pure read, cacheable, resource exists |
| List orders | `GET /orders?status=paid` | collection read, cacheable |
| Create an order | `POST /orders` → `201` | creates a resource |
| Cancel an order | `POST /orders/{id}/cancellation` | action creates a record (Fix #1) |
| Capture a payment | `POST /payments/{id}/capture` | action on a resource, no clean record-noun (Fix #2) |
| Refund a payment | `POST /payments/{id}/refunds` → `201` | creates a refund resource (Fix #1) |
| Estimate shipping | RPC `estimateShipping` | pure computation, no resource (Fix #3) |
| Recalculate pricing | `POST /carts/{id}:recalculate` | nameable process, no record (Fix #2) |

Reads and creates are clean REST. Cancellations and refunds become sub-resources. Captures and recalculations take controller/custom-method form. Only the pure-computation `estimateShipping` is frank RPC. That's the hybrid done deliberately — each operation in the form that's honest about its shape, not forced into one paradigm.

One discipline keeps a hybrid from sliding into a mess: be *consistent about the seams*. If you adopt the `:verb` custom-method convention for one action, use it for all of them — don't have `POST /carts/{id}:recalculate` next to `POST /carts/{id}/recalculation` next to `PATCH /carts/{id} {"action":"recalc"}`. Three styles for the same shape of operation is the real failure, not the existence of action endpoints. Pick *one* way to express "an action on this resource" and one way to express "a record-less procedure," document them, and apply them uniformly. A reviewer should be able to predict, from your existing endpoints, how the next action will be modeled. That predictability is itself a developer-experience feature — the same least-surprise principle the series argues for in [designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal). A hybrid API isn't undisciplined; an *inconsistent* one is.

## 9. Case studies: RPC and REST in the wild

A few accurate, named references, because the principles land harder against real systems.

**JSON-RPC in Ethereum and Bitcoin nodes.** When your wallet or a block explorer talks to an Ethereum node, it speaks JSON-RPC 2.0. Methods like `eth_getBalance`, `eth_sendRawTransaction`, `eth_blockNumber` are posted to the node's single RPC endpoint. Bitcoin Core's RPC interface (`getblockchaininfo`, `sendrawtransaction`, `getrawtransaction`) is the same idea — it predates the 2.0 spec but is the same envelope shape. Why RPC and not REST here? The domain is *intrinsically action/query oriented* ("send this transaction," "give me the balance at this block"), the clients are programmatic (no browser exploration needed), and batching multiple queries in one call is genuinely useful for an indexer pulling many balances. It's a textbook fit: action-heavy domain, programmatic clients, RPC.

**The Language Server Protocol (LSP).** Your editor's autocomplete, go-to-definition, and rename features almost certainly run over JSON-RPC. The editor (client) and the language server communicate by JSON-RPC messages — `textDocument/completion`, `textDocument/definition` — often over stdio, not even HTTP. This is a perfect showcase of JSON-RPC's transport-agnosticism and its notification feature (the server pushes diagnostics to the editor as notifications with no `id`, fire-and-forget). REST would be absurd here; there are no resources, just a fast bidirectional stream of method calls between two processes you control. Pure RPC, by force.

**gRPC at Google (and across the industry).** Internally, Google's services talk to each other over Stubby (gRPC's internal predecessor) and gRPC. Thousands of microservices, latency-sensitive, all owned by the same company — exactly the conditions where RPC's tight coupling is a feature (codegen catches mismatches at build time) and where HTTP caching/browser-friendliness are irrelevant (no browser is calling an internal billing service). The same pattern shows up across the industry for east-west (service-to-service) traffic, while north-south (client-facing) traffic stays REST or GraphQL. This is the cleanest real-world expression of the rule: **RPC for internal service-to-service, REST/GraphQL for the public edge.** For the fleet-level view of those internal calls, the microservices series covers [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

**Stripe and Slack as the hybrid proof.** Already covered in section 8, but worth re-stating as a case study: Stripe is REST-with-action-endpoints (the disciplined hybrid), Slack is mostly-RPC (the action-domain honest about itself), and both are widely regarded as excellent, developer-loved APIs. They sit at different points on the same spectrum, each chosen by the forces of its domain. There is no contradiction — there's just fit.

## 10. When to reach for RPC (and when not to)

Time for the decisive part. Every choice is a trade; here's the recommendation, stated plainly with the *don'ts* spelled out.

![A decision tree that routes public unknown clients toward REST or a REST-plus-actions hybrid and internal owned services toward RPC or typed gRPC by audience and domain shape](/imgs/blogs/rpc-vs-rest-when-a-procedure-beats-a-resource-8.png)

**Reach for RPC when:**

- **The domain is action-heavy.** If your operations are mostly verbs that don't leave readable records (`recalculate`, `estimate`, `notify`, `sync`, `execute`), forcing them into resources is fiction. Name the verbs.
- **The clients are your own services.** Internal service-to-service traffic is the home turf of RPC, especially typed RPC (gRPC). Tight coupling is fine — you control both ends and a schema-driven codegen step catches drift at build time. Use gRPC for low-latency east-west traffic.
- **You own both ends in one language/codebase.** A full-stack TypeScript app is tRPC's sweet spot: the network truly can disappear behind a typed function call because there is no stranger on the other side.
- **Batching or streaming dominates.** If a single interaction needs many operations (JSON-RPC batch) or a stream of results (gRPC streaming), RPC's shapes fit where REST has to bolt them on.
- **Performance is the constraint and the path is internal.** Binary protobuf over multiplexed HTTP/2 beats JSON-over-HTTP/1.1 on payload size and connection efficiency for chatty internal traffic.

**Reach for REST when:**

- **The domain is resource/CRUD-shaped.** Stable entities you create, read, update, delete map cleanly to resources; REST hands you caching, discoverability, and a legible URL map for free. Don't pay RPC's costs to model nouns.
- **The API is public, for callers you'll never meet.** REST's loose coupling, shared status-code vocabulary, and browser-friendly self-description are *product features* for self-serve developers. A stranger should be able to curl a URL and learn your API.
- **Reads dominate and are cacheable.** A read-heavy public surface that can serve `304`s from a CDN saves real money and latency that an all-`POST` RPC API cannot capture.

**Do NOT:**

- **Don't fake an action as a field edit** (`PATCH {"status":"cancelled"}`). It's RPC in disguise, and it loses the record, the audit trail, the validation, and the retry semantics a real action endpoint has. Use a sub-resource or honest RPC.
- **Don't return `200 OK` with an error in the body for a REST API.** That's borrowing RPC's worst habit. Use the HTTP status code (`409`, `422`, `402`) so intermediaries, retries, and dashboards understand the outcome. (RPC's `200`-plus-error-body is a *known cost* of RPC, not a pattern to copy into REST.)
- **Don't choose RPC for a public API just because it matches your code.** The thing that makes RPC pleasant internally — tight coupling to your method signatures — is exactly what burns a public caller when you rename a method. Match your internal interface internally; design a contract for strangers externally.
- **Don't put every operation behind one `POST /rpc` just to avoid thinking about URLs.** You forfeit caching, routing, logging legibility, and browser exploration. If most of your operations are reads of stable entities, that's a steep price for laziness.
- **Don't treat the choice as all-or-nothing.** The hybrid (REST + action endpoints) is the default for a reason. Decide per operation.

## 11. Key takeaways

- **REST models nouns; RPC models verbs.** REST fixes a small uniform verb set and lets resources multiply; RPC fixes one endpoint and lets named procedures multiply. That inversion produces every downstream difference.
- **A REST request's meaning lives in the method + URL; an RPC request's meaning lives in the body.** Therefore caches, gateways, browsers, and logs — which read the request line cheaply — help REST and are blind to RPC. That single fact is the whole trade.
- **REST buys caching, intermediaries, discoverability, and a shared status vocabulary; RPC buys a natural fit to actions, code alignment, batching, and codegen-friendly tight coupling.** Neither is free.
- **When an action doesn't fit a noun, use one of three honest moves**: model its result as a sub-resource (`POST /orders/{id}/cancellation`), use a controller resource or custom method (`POST /carts/{id}:recalculate`), or admit it's RPC. Never fake it as a field edit.
- **JSON-RPC 2.0 is the minimal RPC envelope**: `jsonrpc`/`method`/`params`/`id`, with batching, notifications (no `id`), and standardized error objects. Every response is HTTP `200`, so build error observability at the application layer.
- **gRPC and tRPC are typed RPC**: gRPC for internal, low-latency, polyglot service-to-service traffic (`.proto` + codegen, field-number additive evolution); tRPC for full-stack TypeScript where you own both ends.
- **Most real "REST" APIs are hybrids** — Stripe's `:capture` endpoints, Slack's `chat.postMessage` — and that's correct design, not compromise. Be REST by default, RPC where the action earns it, deciding per operation.
- **Choose by force, not fashion**: RPC for internal action-heavy services and same-team full-stack code; REST for public resource CRUD and cacheable reads. The audience and the domain shape pick the style.

## 12. Further reading

- **JSON-RPC 2.0 Specification** — the canonical envelope spec (jsonrpc.org). Short, complete, and the source of truth for `method`/`params`/`id`, batching, notifications, and error codes.
- **Roy Fielding, "Architectural Styles and the Design of Network-based Software Architectures"** (2000 dissertation, chapter 5) — the original definition of REST and its constraints, including the uniform interface; the document that frames RPC as the thing REST reacts against.
- **The gRPC documentation and the Protocol Buffers language guide** (grpc.io, protobuf.dev) — service/`rpc` definitions, the four streaming modes, deadlines, and protobuf's field-number evolution rules.
- **Google AIP (API Improvement Proposals), especially the "custom methods" guidance** (aip.dev) — the disciplined way to host verb-shaped operations under resource URLs (`:recalculate`), the controller-resource pattern in section 4.
- **RFC 9110, HTTP Semantics** — the definition of safe and idempotent methods and the status-code vocabulary that REST leans on and RPC forfeits.
- **Within this series**: the intro hub, [what an API really is — a contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the resource-modeling deep dive, [turning a domain into nouns and URIs](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris); the next post, [gRPC and Protocol Buffers — contracts, codegen, and streaming](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming); the decision framework, [choosing a paradigm — REST vs gRPC vs GraphQL by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); and the capstone, [the API design playbook — a review checklist from first endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- **For the distributed-systems-scale view**, the system-design overview of [REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) — read it alongside this post; we own the mindset, it owns the operational picture.

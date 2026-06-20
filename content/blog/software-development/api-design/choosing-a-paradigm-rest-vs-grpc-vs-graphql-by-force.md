---
title: "Choosing a Paradigm: REST vs gRPC vs GraphQL by Force, Not Fashion"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Pick REST, gRPC, or GraphQL from the forces that actually decide — who the client is, the latency budget, caching, streaming, evolvability — with a one-pass decision framework and three worked Payments and Orders scenarios."
tags:
  [
    "api-design",
    "api",
    "rest",
    "grpc",
    "graphql",
    "protobuf",
    "bff",
    "http",
    "architecture",
    "system-design",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-1.png"
---

A staff engineer I worked with once told a new team, with a completely straight face, "We're doing GraphQL because that's what the good companies use." Six months later the team had a single mobile client, a flat catalog of products, and a `/products/{id}` shape that never changed. They had also lost HTTP caching entirely (every request was now a `POST` to `/graphql`, which a CDN cannot cache by URL), introduced the N+1 resolver trap into their hottest read path, and shipped a custom error convention because GraphQL returns `200 OK` with errors in the body and their on-call dashboards keyed on status codes. They had bought every hard problem GraphQL is famous for and earned exactly none of its benefits, because they never had the force that GraphQL exists to solve: many diverse clients aggregating across many services. They chose by fashion, and fashion sent them the bill.

This is the post where Track E of the series stops adding paradigms and starts choosing between them. We have covered [when a procedure beats a resource](/blog/software-development/api-design/rpc-vs-rest-when-a-procedure-beats-a-resource), the [gRPC and Protocol Buffers contract](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming), and [GraphQL's schema and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap). Each post made a case *for* its paradigm. None of them is the right answer everywhere — and that is the whole point. The right answer is the one the **forces** of your situation select, and the forces are knowable before you write a line of code. This post complements the broader distributed-systems view in [API design: REST, gRPC, GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql); we lean on it heavily and go deeper on the *decision itself*.

![a matrix scoring REST gRPC and GraphQL across browser reach internal latency caching diverse clients and streaming where each paradigm dominates a different force](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-1.png)

By the end you will have a framework you can run in one pass over any new API: a short list of forces, a mapping from each force to the paradigm it favors, and a decision tree that resolves on the first hard constraint. You will be able to defend a paradigm choice in a design review with reasons that survive a skeptical principal engineer — "we chose gRPC because this is an internal, polyglot, latency-bound service with bidirectional streaming," not "gRPC is faster." And because real platforms almost never run one paradigm, you will see how to **mix** them honestly: gRPC for the internal core, REST and GraphQL at the edge, events for the asynchronous seams. The running example, as always, is a **Payments and Orders** platform — `/orders`, `/payments`, `/refunds`, the webhooks and the SDK — because nothing exposes a paradigm choice like money that must not be double-charged and a mobile screen that must not page forever.

Let me define the three words before we weigh them, so an engineer who has only ever written `@app.route("/orders")` can follow every trade-off. **REST** (Representational State Transfer) shapes your API as *resources* — nouns with URIs like `/orders/123` — that you act on with HTTP methods (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`), exchanging *representations* (usually JSON) and leaning on HTTP's own semantics: status codes, caching headers, content negotiation. **gRPC** (gRPC Remote Procedure Calls) shapes your API as *procedures* — you call a method like `GetOrder(GetOrderRequest)` on a service, the call and its types are defined in a `.proto` file, and the bytes on the wire are compact binary Protocol Buffers carried over HTTP/2. **GraphQL** shapes your API as a single *graph schema* the client queries against: the client sends one query describing exactly the fields it wants, possibly spanning many underlying services, and the server resolves each field. Three different answers to the same question — *how does a caller ask my system to do something* — and the forces below decide which answer fits.

## The forces that actually decide

A paradigm is not a taste. It is a response to a set of forces, the same way a bridge design is a response to span, load, and wind. Name the forces and the choice mostly makes itself. Here are the ones that have actually moved my decisions, roughly in order of how often they turn out to be decisive.

**Who is the client?** This is the single most decisive force, and it is the first question I ask in any design review. A *browser* wants something it can cache, link to, and debug with the network tab. A *mobile app* wants the fewest round-trips over a high-latency radio link and a payload tailored to one screen. *Another service in your own fleet* wants the lowest possible latency, a strict typed contract, and no human ever looking at the bytes. A *partner* integrating server-to-server wants stability, documentation, and a contract that will not break their nightly batch job. A *public, anonymous* caller wants the lowest barrier to entry — a `curl` command that works. These are different customers with different needs, and the paradigm that delights one can punish another.

**How many clients, and how diverse are their data needs?** One client with one data shape is a different world from twenty clients each wanting a different slice. If a single mobile team consumes your API and they always want the same `Order` shape, REST's fixed representation is perfect and GraphQL is overhead. But when an Android app, an iOS app, a web dashboard, a smart-watch widget, and a partner's analytics pipeline all want *different* fields off the same underlying data — some want the full order with line items, some want just a status badge — the cost of maintaining a bespoke REST endpoint per client becomes the dominant force, and a query language the client shapes itself starts to pay for its complexity.

**The latency and throughput budget.** How tight is the time budget, and how many calls per second? An internal pricing call on the checkout path that runs millions of times a day and must return in single-digit milliseconds is a different animal from a partner's once-a-day reconciliation pull. Binary serialization, HTTP/2 multiplexing, and persistent connections matter enormously at the first scale and are irrelevant at the second. We will do the payload math below, because "faster" is a claim that deserves numbers.

**Streaming needs.** Does the interaction fit request-then-response, or does it need a flow of messages in one or both directions? A live order-status feed, a real-time fraud-scoring stream, a bidirectional chat — these are streams, and the paradigm's streaming story stops being a footnote and becomes a gate.

**Caching needs.** Can responses be cached, and if so, by whom? A product catalog or an order that rarely changes is enormously cheaper to serve if a CDN or the browser can cache it by URL with an `ETag` and a `304 Not Modified`. The moment your API is a single `POST` endpoint — as both gRPC and GraphQL are over the wire — you have opted out of HTTP's free, battle-tested caching layer and must rebuild it yourself at the application level.

**Team, tooling, and polyglot reality.** What languages do your services speak, and what does your team already know? A shop where Go, Java, Python, and Rust services all call each other gains a lot from gRPC's code generation: one `.proto` file produces typed stubs in every language, and nobody hand-writes a client. A shop with one language and a team that has shipped REST for a decade pays a real learning tax to adopt anything else, and that tax is a legitimate force — though, as we will see, never the *only* one.

**Debuggability and operability.** Can a human read the wire when something breaks at 3 a.m.? REST over JSON is debuggable with `curl` and a browser; you can paste a request into a chat and a teammate understands it. gRPC's binary frames need `grpcurl` and reflection; GraphQL needs you to reconstruct which resolver failed inside a `200`. This is a force that bites later, in production, which is exactly why it gets undervalued at design time.

**Public vs internal.** A public API is a contract with strangers on a timeline of years, optimized for reach, stability, and the lowest barrier to entry. An internal API is a contract among teams you can coordinate with, optimized for speed and tight types, and you can change it across a deploy. The blast radius of a breaking change is a different order of magnitude, and that difference reshapes the trade-offs.

**Evolvability needs.** How will this contract change, and how easily can you change it without breaking callers? Protobuf's field numbers and reserved-field discipline, REST's tolerant-reader additive-change rules, and GraphQL's field-deprecation and schema-stitching all give different evolution stories. We covered the rules in [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) and [schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely); here the question is which paradigm makes safe change *easiest* for your situation.

These nine forces are the input to the framework. The output is a paradigm. The figure above scores the three paradigms against five of the highest-signal forces; the rows are forces, the columns are paradigms, and the cells tell you where each one wins and where it pays a tax. Read it as a map of dominance: no column is green everywhere, which is the entire reason this is a *decision* and not a default.

One more thing about forces before we weigh them: they are not equally weighted, and they are not independent. *Who is the client* is the heaviest force because it fixes several others downstream — a browser client implies you care about caching, debuggability, and reach all at once, because the browser environment carries those needs with it. *Public vs internal* is the second-heaviest for the same reason: it sets the blast radius of every future change and therefore how much evolvability discipline you must buy. The lighter forces — team familiarity, tooling — matter, but they are *tie-breakers*: they decide between paradigms that the heavy forces left close, and they should never override a heavy force. A useful habit is to write the forces down for a specific API, mark each as *binding* (this force alone forces a paradigm), *strong* (it would tip a close call), or *weak* (a tie-breaker), and then notice that you almost always have one binding force. The binding force is the answer; everything else is confirmation.

It is also worth saying plainly what is *not* a force. The newness of a paradigm is not a force. The fact that a conference talk praised it is not a force. The fact that a competitor uses it is not a force unless you have evidence they share your force profile, which you usually do not. A force is a *property of your situation* — your client, your latency budget, your team — that a paradigm either serves or taxes. If a reason for a paradigm choice cannot be phrased as "because of this property of our situation," it is fashion, and it belongs nowhere near the decision.

## The principle: a paradigm is a bet on which forces dominate

Here is the rule, stated rigorously, that the rest of the post depends on:

> **A paradigm choice is correct when it optimizes for the forces that dominate your situation and pays its taxes in the forces that do not.** Every paradigm is excellent at some forces and weak at others — there is no free lunch — so "correct" means *aligned with the dominant forces*, not *best in the abstract*.

Why must this be true rather than merely sounding wise? Because the three paradigms make genuinely *opposed* design choices, and an optimization for one force is, mechanically, a de-optimization for another.

Consider caching versus client-shaped queries. HTTP caching works because a `GET` to a URL is *safe* and *cacheable*: the same URL yields a cacheable representation, and a CDN can key its cache on that URL. REST exploits this directly. GraphQL deliberately gives up URL-based caching — every query is a `POST` with the query in the body, so two different queries hit the same URL and a URL-keyed cache cannot tell them apart — *in exchange for* letting the client specify exactly the fields it wants in one round-trip. You cannot have both properties from the same mechanism: a cache that keys on a stable URL is incompatible with a body that varies per client. So the moment you value client-shaped queries, you have mechanically devalued free HTTP caching. That is not GraphQL being bad; it is a conservation law.

Consider human-readability versus wire efficiency. JSON is self-describing text: the field names travel with every message, which is exactly what makes `curl` output readable and exactly what makes the payload larger. Protobuf drops the field names and ships field *numbers* against a schema both sides already hold, which is exactly what makes it compact and exactly what makes the wire unreadable without the schema. You cannot ship the field names (readability) and not ship them (compactness) at once. Another conservation law.

Consider a third pair: strict typed contracts versus a zero-codegen barrier. gRPC's strictness comes from a schema both sides compile against, which is precisely what makes a field mismatch a compile error — and precisely what forces every caller, including a stranger trying you out, to run codegen before the first request. REST's zero-barrier `curl`-it-now reach comes from *not* requiring a compiled contract — which is precisely what makes a field mismatch a runtime surprise rather than a compile error. Strictness and zero-barrier are opposed: the mechanism that gives you one denies you the other. You pick the one whose force dominates — compile-time safety for an internal fleet, zero-barrier reach for a public surface — and you accept the cost of the other.

Because these trade-offs are conservation laws and not accidents, no future version of any paradigm will "fix" the weakness without surrendering the strength. That is why choosing by force is permanent advice and choosing by fashion is not: fashion changes, the conservation laws do not. The discipline, then, is to *measure the forces* and let the dominant one bind the choice. Let me make that concrete with the dominant-force-first decision tree.

![a decision tree starting from who is the client branching to public or browser to internal service to many diverse clients each leading to REST gRPC or GraphQL](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-2.png)

The tree starts where every honest API decision starts: *who is the client?* A public or browser-facing client routes you toward REST, because reach, cacheability, and the lowest barrier to entry dominate. A service-to-service caller inside your own fleet routes you toward gRPC, because latency, strict typing, and polyglot codegen dominate. Many diverse clients with conflicting data needs route you toward GraphQL — usually as a backend-for-frontend — because over-fetching and under-fetching pain dominates. The tree is not the whole story; the rest of this post adds the second and third forces that refine each branch. But the first question is always the client, because the client's environment fixes more constraints than any other single force.

## REST: the default for reach, caching, and a resource-shaped world

Make REST your default and reach for the others only when a force pushes you off it. This is not nostalgia; it is the recognition that REST aligns with the most common force profile in the most common kind of API: a public or browser-facing surface, over resources that look like nouns, where responses are cacheable and the contract must stay stable for strangers for years.

When REST is the force-driven answer, the forces that dominate are: the client is a browser, a partner, or a public anonymous caller; the data is resource-shaped (orders, payments, refunds are nouns you `GET`, `POST`, and `PATCH`); responses are cacheable; the barrier to entry must be low; and a human will read the wire. REST wins all of those by leaning on HTTP's own semantics rather than reinventing them.

Here is the wire for our running example — boring, debuggable, and exactly the point:

```http
GET /v1/orders/ord_8a3 HTTP/1.1
Host: api.commerce.example
Accept: application/json
Authorization: Bearer <token>
If-None-Match: "etag-7c2f"
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
ETag: "etag-7c2f"
Cache-Control: private, max-age=60

{
  "id": "ord_8a3",
  "status": "paid",
  "total": { "amount": 4999, "currency": "USD" },
  "line_items": [
    { "sku": "SKU-114", "qty": 2, "unit_price": 2499 }
  ]
}
```

Notice what you get for free. The `ETag` plus `If-None-Match` give you conditional requests: if the order has not changed, the server returns `304 Not Modified` with an empty body, and the caller pays only the headers. `Cache-Control` lets the browser and any CDN cache the order for 60 seconds. The status code carries the truth — `200`, `404`, `409`, `429` — so your on-call dashboards key on a number every HTTP tool already understands. A partner debugging an integration pastes that `curl` into a chat and a teammate reads it instantly. None of this is incidental; it is the dividend of building *with* HTTP rather than *on top of* it, and it is why the price of a \$49.99 order (the `4999` minor units above) is one trivially cacheable `GET` away.

REST's evolvability story is the tolerant-reader one we derived earlier: adding an optional response field is non-breaking because a well-behaved client ignores fields it does not recognize; removing a field or renaming one is breaking. That gives you a wide lane for additive change without versioning, which is exactly the lane a long-lived public API needs.

There is a subtler reason REST is the right default for the public edge, and it is worth making explicit because it is the force engineers undervalue most: **REST has the lowest cost of being wrong.** When you ship a REST endpoint and later discover a field should have been named differently or a status code should have been `409` instead of `400`, the additive-change rules give you a path to fix it without breaking anyone — add the new field, deprecate the old, keep both for a window. When you ship a public gRPC contract and discover a `.proto` mistake, every integrator must recompile; when you ship a public GraphQL schema and discover a type modeled wrong, you are deprecating fields on a schema strangers query in ways you cannot see. The public edge is where you have the *least* information about your callers and the *longest* timeline, which means it is where the cost of being wrong is highest — and REST minimizes that cost. Defaulting to REST at the edge is not conservatism for its own sake; it is choosing the paradigm that is cheapest to correct in the place where you are most likely to need to.

A common objection: "but JSON over HTTP/1.1 is slow." For a public API, mostly it does not matter, and where it does, the fix is rarely a paradigm change. The transfer of a 5 KB JSON order over a warm HTTP/2 connection with gzip is dominated by network round-trip time, not serialization; the levers that actually move a public API's latency are connection reuse (HTTP/2 or keep-alive), compression (`gzip`/`br`), a CDN in front of cacheable reads, and not over-fetching — all of which REST supports natively. Reaching for gRPC to speed up a *public* API is solving a serialization problem you probably do not have while creating a reach problem you definitely will. We cover the actual latency levers in [API performance: payload size, compression, and tail latency](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency); the framework point is that "JSON is slow" is almost never the binding force for a public surface.

#### Worked example: a public payments API

A fintech wants to expose a **public Payments API** so any developer in the world can charge a card and issue a refund. Run the forces.

- **Who is the client?** Strangers — anonymous developers, partner servers, browser-based checkout pages. This is the most public client there is.
- **How many, how diverse?** Effectively unbounded and unknown. You will never meet most of them, and you cannot coordinate a breaking change with them.
- **Caching?** Reads like "fetch this charge" or "list these refunds" are eminently cacheable; a charge object is immutable once settled.
- **Debuggability?** A developer's first experience must be a `curl` that works in ten seconds, copied from your docs.
- **Latency budget?** Real but generous — a card charge is a human-facing action measured in hundreds of milliseconds, not microseconds.
- **Evolvability?** Critical, and across years. You must add fields for new payment methods without breaking a partner who integrated three years ago.

Every dominant force points the same way. **Decision: REST over HTTP and JSON, with `problem+json` errors and idempotency keys.** This is not a coincidence — it is why essentially every successful public payments API in the industry is REST. The lowest barrier to entry (`curl`), the broadest reach (any HTTP client in any language with zero codegen), free caching, honest status codes, and a wide additive-change lane all line up. A gRPC public payments API would force every integrator to compile a `.proto` and run a binary protocol through their firewall; a GraphQL public payments API would hand strangers a query language they can use to over-fetch your database into the ground. Here is the charge, idempotent against a retry:

```http
POST /v1/charges HTTP/1.1
Host: api.payments.example
Content-Type: application/json
Authorization: Bearer <token>
Idempotency-Key: idem_9f2a-charge-114

{ "amount": 4999, "currency": "USD", "source": "tok_visa" }
```

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /v1/charges/ch_71d

{ "id": "ch_71d", "amount": 4999, "currency": "USD", "status": "succeeded" }
```

If the network drops and the client retries with the same `Idempotency-Key`, the server returns the cached `201` and the customer is charged once — the [idempotency-key contract](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) doing its job. REST did not make this harder; HTTP gave us the headers to make it work.

**When NOT to default to REST:** when the dominant force is internal microsecond latency at massive throughput (reach for gRPC), or when many heterogeneous clients are drowning in over- and under-fetching (reach for a GraphQL BFF). If neither force is present, stay on REST. The most common paradigm mistake I see is leaving REST for a problem REST did not have.

## gRPC: the internal, low-latency, polyglot, streaming choice

gRPC is the answer when the client is *another service in your own fleet* and the dominant forces are latency, throughput, strict typing, polyglot codegen, and streaming. Inside a data center, between services you own, where no human reads the wire and every microsecond on the checkout path is multiplied by millions of calls a day, gRPC's choices stop being trade-offs and start being pure wins — because the forces it taxes (browser reach, free HTTP caching, human readability) are forces you do not have inside the fleet.

The contract is the `.proto`, and the contract *is* the API — codegen produces a typed client and server stub in every language from this one file:

```protobuf
syntax = "proto3";
package commerce.risk.v1;

service RiskScorer {
  rpc Score(ScoreRequest) returns (ScoreResponse);
  rpc ScoreStream(stream ScoreRequest) returns (stream ScoreResponse);
}

message ScoreRequest {
  string order_id = 1;
  int64 amount_minor = 2;
  string currency = 3;
  string customer_id = 4;
}

message ScoreResponse {
  double risk_score = 1;     // 0.0 safe .. 1.0 fraud
  string decision = 2;       // "approve" | "review" | "deny"
}
```

Three things this buys that REST cannot match inside the fleet. First, **wire efficiency**: the `ScoreRequest` ships field numbers and binary values, not field names and quoted strings, so it is meaningfully smaller and faster to parse than the equivalent JSON. Second, **strict, generated contracts**: the Java caller, the Go caller, and the Python caller all import a stub generated from the same `.proto`; nobody hand-writes a client, nobody mistypes a field name, and a mismatch is a compile error rather than a `500` in production. Third, **native streaming**: `ScoreStream` is a bidirectional stream — the caller can push a flow of orders and receive a flow of scores over one HTTP/2 connection, which is exactly what a real-time fraud pipeline wants.

Let me do the payload math honestly, because "gRPC is faster" deserves a number and a setup, not a slogan. Take the `ScoreRequest`. As JSON it is roughly:

```json
{"order_id":"ord_8a3","amount_minor":4999,"currency":"USD","customer_id":"cus_22b"}
```

That is about 80 bytes, and most of it is field names and quotes — `"order_id":`, `"amount_minor":`, the braces, the commas. The same message in Protobuf drops every field name (replaced by a one-byte field-number-and-type tag) and encodes the integer as a varint, landing in the rough neighborhood of 30 bytes for this payload — call it a 2–3x reduction, and the exact ratio depends entirely on how name-heavy your fields are (treat these as illustrative, not benchmarked). On a single call that difference is invisible. But the latency story is not only bytes: gRPC rides HTTP/2 with persistent, multiplexed connections, so it skips the TCP and TLS handshake that a naive HTTP/1.1 client pays per request, and it parses binary instead of tokenizing JSON. At a few hundred bytes per call the serialization win is modest; the *connection-reuse and parsing* wins are what actually move the p99 when you are doing tens of thousands of internal calls per second on a latency-bound path. The honest summary: gRPC's speed advantage is real and grows with call volume and connection churn, and is close to irrelevant for a once-a-day partner pull.

#### Worked example: an internal risk-scoring service

Our Payments platform needs a **risk-scoring service** that the checkout path calls on every order before authorizing the charge. Run the forces.

- **Who is the client?** The Orders service and the Payments service — internal callers we own and deploy alongside.
- **How many, how diverse?** A handful of internal services, all wanting the same `ScoreResponse` shape. No diversity of data needs.
- **Latency and throughput?** This is on the synchronous checkout path. It runs on every order — easily tens of thousands of calls per second at peak — and every millisecond it adds is a millisecond on the customer's checkout. The budget is single-digit milliseconds.
- **Streaming?** Yes — for batch re-scoring and for a real-time fraud feed, a bidirectional stream is the natural shape.
- **Polyglot?** The Orders service is in Go, the risk model is served from Python. One contract, two languages, no hand-written clients.
- **Caching, browser reach, public access?** None. No browser ever calls this; no human reads the wire; it never leaves the fleet.

Every dominant force points to gRPC, and crucially, *none* of gRPC's taxes apply: there is no browser to deny caching to, no stranger to hand a `.proto` to, no human who needs to `curl` it. **Decision: gRPC with Protocol Buffers, unary for the per-order score and a bidirectional stream for the fraud feed, with a strict deadline.** Here is the Go caller with a deadline — a force gRPC handles cleanly that REST handles awkwardly:

```go
ctx, cancel := context.WithTimeout(ctx, 8*time.Millisecond)
defer cancel()

resp, err := riskClient.Score(ctx, &riskpb.ScoreRequest{
    OrderId:     "ord_8a3",
    AmountMinor: 4999,
    Currency:    "USD",
    CustomerId:  "cus_22b",
})
if status.Code(err) == codes.DeadlineExceeded {
    // budget blown — fail open or closed per policy, do not block checkout forever
    resp = &riskpb.ScoreResponse{Decision: "review"}
}
```

The 8-millisecond deadline is a first-class gRPC concept that propagates across the call; if the risk service is slow, checkout does not hang, it degrades by policy. That is the kind of control a latency-bound internal path needs and is exactly the force gRPC was built for.

The polyglot-codegen force deserves one more concrete beat, because it is the one that quietly saves the most engineering time in a real fleet. In a shop where the Orders service is Go, the risk model is served from Python, and a reconciliation job is in Java, the *same* `.proto` above generates a typed client and server in all three languages. The Go team calls `riskClient.Score(...)` with a generated `ScoreRequest` struct; the Python team implements `def Score(self, request, context)` against a generated base class; the Java team gets a `ScoreRequest.Builder`. Nobody hand-writes an HTTP client, nobody hand-parses JSON, nobody mistypes `"amount_minor"` as `"amountMinor"` and discovers it in production — because the field name never appears in anyone's code, only the generated accessor does. When the contract changes, you regenerate, and every language's compiler flags the callers that need updating *before* deploy. That compile-time-across-languages safety is something REST simply cannot offer without a separate code-generation pipeline bolted on (OpenAPI generators do exist, but they are generating from a spec you maintain by hand, not from the contract itself). For a polyglot fleet that changes contracts often, this force alone can be binding — it is not about speed at all, it is about never shipping a field-name typo across a language boundary again.

It is worth being honest about gRPC's costs too, so the choice is eyes-open. The binary wire means you cannot debug with `curl` — you need `grpcurl` and server reflection, and a teammate cannot paste a request into chat and read it. Browsers cannot speak gRPC natively, so a browser client needs gRPC-Web and a proxy. Load balancers must understand HTTP/2 and long-lived connections, or they will pin all of a client's calls to one backend. And the operational tooling — tracing, logging, rate limiting at a gateway — is more mature for HTTP/JSON than for gRPC in many stacks. None of these costs matter inside a fleet that has invested in the tooling and where no browser or human touches the wire; all of them matter the instant the API faces outward. That is precisely why gRPC's home is the internal core and why pushing it to the public edge trades its strengths for a pile of its weaknesses.

**When NOT to reach for gRPC:** when the client is a browser (gRPC needs a proxy like gRPC-Web and you lose the native debuggability and caching), when the API is public (you are asking strangers to adopt a binary protocol and codegen), or when the call volume is low and the latency budget is generous (you are paying gRPC's operational and debuggability tax for a speed win you cannot measure). "gRPC is faster so always use it" is the single most common wrong reason, and it ignores that *faster at what cost, for which client* is the actual question.

## GraphQL: many diverse clients aggregating across services

GraphQL is the answer when the dominant force is **many diverse clients aggregating data across many services**, and the pain you are solving is over-fetching (a client downloading fields it does not need) and under-fetching (a client making three round-trips to assemble one screen). When a single client always wants the same shape, GraphQL is overhead; when five clients each want a different slice of data that lives behind several services, GraphQL's client-specified query language starts to earn its complexity.

The mechanism is a single schema the client queries against. The client sends one query naming exactly the fields it wants, and the server resolves each field, possibly from a different backend service:

```graphql
type Order {
  id: ID!
  status: String!
  total: Money!
  payment: Payment
  lineItems: [LineItem!]!
}

type Query {
  order(id: ID!): Order
}
```

A mobile screen that needs an order's status, total, and payment method — but not the full line items — asks for exactly that in one round-trip:

```graphql
query OrderHeader {
  order(id: "ord_8a3") {
    status
    total { amount currency }
    payment { method last4 }
  }
}
```

The web dashboard, wanting the full order with every line item, sends a *different* query against the *same* schema. Neither client got a bespoke endpoint; neither over-fetched; neither made three calls. That is the over/under-fetching force resolved. The cost — the conservation law — is the **N+1 trap**: a naive resolver that fetches each order's payment with a separate backend call turns one query over 50 orders into 51 calls, which we dissected and batched with a dataloader in the [GraphQL post](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap). You also pay the caching tax (every query is a `POST`, so no URL-keyed CDN caching without persisted queries) and the error tax (`200 OK` with errors in the body, so your status-code dashboards need rework).

The honest framing: GraphQL is a *client convenience and aggregation* technology, not a speed technology. It does not make the backend faster; it makes the *client's* life easier by moving the query-shaping to the client and the aggregation to the server. That is enormously valuable when you have many clients and many services, and pure cost when you have one of each.

#### Worked example: a mobile super-app

Our company ships a **mobile super-app** — one app with a home feed, an orders tab, a payments tab, and a wallet, backed by Orders, Payments, Refunds, and Loyalty services. The iOS app, the Android app, and a web version each render slightly different screens. Run the forces.

- **Who is the client?** Mobile and web apps that *we* own — but several of them, each with its own screen layouts.
- **How many, how diverse?** Three first-party clients plus a steady churn of new screens, each wanting a different combination of fields from four backend services. This is the over/under-fetching force in its purest form.
- **Latency?** The dominant latency cost on mobile is *round-trips* over a high-latency radio link; collapsing a screen's five REST calls into one query is the biggest win available.
- **Aggregation?** The home screen needs the latest order (Orders), its payment status (Payments), any pending refund (Refunds), and the loyalty balance (Loyalty) — four services, one screen.
- **Caching?** Mostly per-user and short-lived; the free URL caching REST offers is less valuable here than the round-trip savings.
- **Public?** No — these are first-party clients we can coordinate with on a deploy.

The forces point to GraphQL, specifically as a **backend-for-frontend (BFF)** — a server tier owned by the client teams that exposes one GraphQL schema and fans out to the backend services. **Decision: a GraphQL BFF over the four gRPC services, with dataloaders to batch the fan-out and persisted queries to win back caching.** The single home-screen query:

```graphql
query HomeScreen($userId: ID!) {
  latestOrder(userId: $userId) {
    id
    status
    total { amount currency }
    payment { method last4 }
    pendingRefund { amount status }
  }
  loyalty(userId: $userId) { points tier }
}
```

One round-trip from the phone, four services fanned out and batched behind the BFF. A new screen that needs different fields ships *without a backend change* — the client just asks for different fields off the same schema. That is the diverse-clients force paying off, and it is why super-apps and content platforms with many screens reach for GraphQL.

**When NOT to reach for GraphQL:** when you have one client with one stable shape (you are buying the N+1, the caching loss, and the error-convention rework for nothing — use REST), when the API is public and anonymous (a query language hands strangers a tool to over-fetch and craft expensive queries; you must add query-cost limits, depth limits, and persisted-query allowlists just to be safe), or when the dominant force is raw service-to-service latency (use gRPC). GraphQL solves an *aggregation and client-diversity* problem; if you do not have that problem, it is a liability.

## When streaming is the force

Most of the forces so far assume a request-then-response interaction: the caller asks, the server answers, done. But a real platform has interactions that do not fit that shape — a live order-status feed, a real-time fraud-scoring pipeline, a price ticker, a notification channel — and when the interaction is a *flow* of messages rather than a single answer, the paradigm's streaming story stops being a footnote and becomes a binding force. The question is not just "which paradigm," but "which paradigm can carry the *direction* and *cardinality* of messages this interaction needs."

There are four message patterns, and only some paradigms serve each well. **Unary** is one request, one response — every paradigm does this; it is the default. **Server streaming** is one request, many responses — the server pushes a flow back, like a live order-status feed where the client subscribes once and receives every state change. **Client streaming** is many requests, one response — the client pushes a flow up, like a batch of orders uploaded for scoring with one summary back. **Bidirectional streaming** is many requests, many responses interleaved over one connection — a real-time fraud feed where orders flow up and scores flow down continuously, or a chat. The cardinality you need is a hard constraint, and here is how the paradigms answer it.

**gRPC has native, first-class streaming in all four modes**, because it rides HTTP/2, which multiplexes many logical streams over one connection. A `.proto` declares a streaming RPC with the `stream` keyword on the request side, the response side, or both — we saw `ScoreStream` do exactly that above. This is gRPC's structural advantage for real-time internal pipelines: the streaming is part of the contract, generated into the stubs, with flow control and backpressure handled by HTTP/2 itself. When the dominant force is bidirectional, low-latency, internal streaming, gRPC is not just an option — it is the obvious answer, and the other paradigms are working around their lack of it.

**REST's streaming story is Server-Sent Events (SSE)** — a long-lived `GET` where the server holds the connection open and pushes a stream of `text/event-stream` events. It is one-directional (server to client only), it is plain HTTP so it works through every proxy and firewall and is `curl`-able, and it is exactly right for a public live feed where the client only needs to *receive*:

```http
GET /v1/orders/ord_8a3/events HTTP/1.1
Host: api.commerce.example
Accept: text/event-stream
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

event: status_changed
data: {"order_id":"ord_8a3","status":"shipped"}

event: status_changed
data: {"order_id":"ord_8a3","status":"delivered"}
```

SSE is the honest REST answer for *server-to-client* public streaming: a browser can consume it with the built-in `EventSource` API, it reconnects automatically with a `Last-Event-ID`, and it stays inside HTTP's debuggable, proxy-friendly world. What SSE cannot do is carry a *client-to-server* flow — for that you need a WebSocket (a full bidirectional channel) or gRPC. The trade-off, again, is a conservation law: SSE keeps HTTP's reach and debuggability by giving up the upstream direction.

**GraphQL's streaming story is subscriptions** — a `subscription` operation where the client subscribes to an event and the server pushes updates, typically over a WebSocket. It is the right tool when you already have a GraphQL BFF and a client that wants to *subscribe* to a slice of the graph (live updates to exactly the fields a screen renders) the same way it queries them. It inherits the WebSocket transport's needs (a stateful connection, more complex scaling, harder caching) but gives the client the same field-shaped control over a live feed that it has over a one-shot query.

The forces map cleanly. **Bidirectional, low-latency, internal → gRPC bidi streaming.** **Server-to-client, public, browser-friendly → SSE over REST.** **Client wants field-shaped live updates and you already run a GraphQL BFF → subscriptions.** **Full bidirectional in a browser without gRPC → a WebSocket.** We go deeper on the transport mechanics — SSE vs WebSocket vs gRPC streaming, and backpressure — in [streaming APIs](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming); the framework point here is that streaming cardinality and direction can be the *binding* force, and when they are, they often override the who-is-the-client default. A public order-status feed that only pushes might be SSE over your REST API even though you would normally not stream from a public surface; an internal fraud pipeline that needs bidirectional flow is gRPC even before you finish listing the other forces.

#### Worked example: a live order-tracking feed

The mobile super-app wants a **live order-tracking screen** that updates the moment an order's status changes — placed, paid, shipped, delivered — without the user pulling to refresh. Run the forces.

- **Who is the client?** The first-party mobile app, which is already talking to the GraphQL BFF for its other screens.
- **Direction and cardinality?** Server-to-client only — the client subscribes and receives a flow of status updates; it does not push anything up the stream. Server streaming, not bidirectional.
- **Latency?** Updates should land within a second or two of the backend state change; this is a notification feed, not a microsecond path.
- **Reach?** First-party app only, behind the BFF; no anonymous public consumer.

Two reasonable answers, and the framework picks between them by *what the client already speaks*. Because the app already runs through a GraphQL BFF, a **GraphQL subscription** lets the live feed return the exact same order fields the static screen already renders, with no new transport convention for the client team to learn — the subscription is the binding consideration. **Decision: a GraphQL subscription over the BFF for the live order-tracking screen.** If the same feed had to be consumed by an *anonymous public* page — say a shareable "track my package" link with no app and no BFF — the forces flip to server-to-client over a public surface, and the answer becomes **SSE over the REST API**, because reach, browser-native consumption, and proxy-friendliness dominate. Same feature, different binding force, different paradigm — exactly the discipline this whole post is about.

## The honest nuance: you almost always mix

Here is the part the fashion debates miss entirely: the question is rarely "REST *or* gRPC *or* GraphQL." A real platform of any size runs **all three**, each where its forces dominate, and the architecture is *how you combine them*, not which one you anoint. The mistake is treating paradigm choice as a religion when it is a layering decision.

![a graph of a hybrid stack with browser mobile and partner clients hitting a REST edge and a GraphQL BFF that both call internal Orders and Payments gRPC services](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-4.png)

The figure shows the shape almost every mature platform converges on. **gRPC runs the internal core** — Orders, Payments, Risk, Refunds call each other over typed binary contracts because internal latency and polyglot codegen dominate there. **REST faces the public edge** — partners and browser-based checkout get a stable, cacheable, `curl`-able resource API because reach and stability dominate there. **A GraphQL BFF faces the first-party apps** — the mobile super-app gets one screen-shaped query because client-diversity and round-trip cost dominate there. The clients on the left choose their edge by *their* forces; the services on the right speak gRPC by *their* forces; and the edge tiers translate between them. The graph branches (three client types) and merges (both edges call the shared Orders service) and is acyclic — which is also exactly what a healthy dependency graph should be.

Two combination patterns are worth naming because they are the workhorses.

**A gateway over gRPC services.** You can expose a REST or GraphQL surface that is, underneath, a thin translation layer over gRPC backends. gRPC itself ships a *gRPC-JSON transcoder* (and frameworks like grpc-gateway generate one from your `.proto` plus HTTP annotations) so that one `.proto` can serve both a gRPC internal contract and a REST public contract from the same definition. That gives you the internal speed of gRPC and the external reach of REST without writing the contract twice. The [API gateway and BFF post](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern) covers the gateway's broader responsibilities — routing, auth, rate limiting — and this transcoding is one of them.

**The BFF pattern.** A backend-for-frontend is a thin server tier, owned by a client team, that exists to shape one client's experience. It can be GraphQL (the super-app case above) or it can be a REST BFF that aggregates a handful of gRPC calls into one screen-shaped JSON response. The point of the BFF is that the *aggregation and shaping* logic lives in a tier the client team controls, so they can iterate on screens without dragging the backend services through a change. The microservices series goes deep on this in [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend); here the takeaway is that the BFF is *how* you let GraphQL or REST live at the edge while gRPC lives in the core.

![a stack diagram of the backend for frontend layering with a mobile client on top a GraphQL BFF then a batched fan-out to Orders and Payments gRPC services over datastores](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-5.png)

The stack figure makes the BFF layering literal: the mobile client sends one screen request to the GraphQL BFF, the BFF fans out three batched gRPC calls in parallel, the Orders and Payments services hold their typed contracts, and the datastores sit behind the services. Each layer speaks the paradigm its forces demand, and the BFF is the seam where edge paradigm meets core paradigm. This is the single most important structural idea in the post: **paradigm choice is per-layer, not per-platform.**

And then there are **events**, the fourth paradigm we have mostly set aside. Not every interaction is request-then-response. When the Payments service finishes a charge, the Orders service does not need to poll — it needs to be *told*, asynchronously. That is a webhook or an event on a broker, the "fire and forget, notify me later" shape, and it is a peer of REST, gRPC, and GraphQL rather than a subordinate of them. We give it its own treatment in [event-driven and async APIs](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi), with the broker mechanics linked out to the message-queue series for [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once). The framework rule for events is simple: when the interaction is asynchronous and notify-shaped — "tell the partner their charge settled" — reach for webhooks or pub/sub, not a synchronous call the caller has to wait on.

## A side-by-side: paradigms across the forces

Here is the comparison table the framework rests on. Read each row as a force and each cell as how the paradigm handles it. The point is not that one column wins; it is that the *pattern* of wins and losses across the rows is what selects the paradigm for your force profile.

| Force | REST / HTTP+JSON | gRPC | GraphQL |
| --- | --- | --- | --- |
| Best client | Browser, partner, public | Internal service-to-service | Many diverse first-party apps |
| Wire format | JSON text, human-readable | Binary Protobuf, compact | JSON over a single POST |
| Latency profile | Good; JSON + HTTP/1.1 overhead | Best; binary + HTTP/2 reuse | Good; collapses client round-trips |
| Native streaming | SSE only (one direction) | Four modes incl. bidirectional | Subscriptions (server push) |
| HTTP caching | Free; `ETag`, `304`, CDN by URL | App-level only | Hard; POST defeats URL caching |
| Diverse data needs | Bespoke endpoint per shape | Rigid generated stub per RPC | Client-shaped, one schema |
| Polyglot codegen | OpenAPI-generated SDKs | First-class; one `.proto` → all langs | Schema-generated typed clients |
| Debuggability | High; `curl`, browser | Low; needs `grpcurl`, reflection | Medium; errors inside `200` |
| Public reach | Highest; zero barrier | Low; binary + codegen barrier | Medium; needs cost guards |
| Evolvability | Additive via tolerant reader | Field numbers, reserved fields | Field deprecation, schema growth |
| Reach for it when | Public, cacheable, resource-shaped | Internal, fast, polyglot, streaming | Many clients, aggregation pain |

If you take one thing from this table, take this: the rows that are *green for REST and red for the others* (HTTP caching, public reach, debuggability) are exactly the forces of a public API, and the rows that are *green for gRPC and red for the others* (latency, native streaming, polyglot codegen) are exactly the forces of an internal fleet. The forces cluster, and that clustering is why the client-type question at the top of the tree is so decisive.

## The wrong reasons, and the right ones underneath them

Most paradigm regret I have witnessed traces to a small set of wrong reasons, and the cure is the same every time: find the real force underneath the fashionable claim, and let the force answer.

![a before and after contrast showing GraphQL chosen because it is modern losing ETag caching and gaining N plus one against REST chosen from the forces keeping cacheable resources](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-3.png)

The before-and-after figure is the team from the opening, drawn out. *Before*, they chose GraphQL because it was modern, for a single client doing simple resource CRUD: they lost `ETag` caching (every call became a `POST`) and inherited the N+1 resolver tax, with no diverse-clients force to justify either cost. *After*, choosing from the forces, the same single CRUD client gets REST: a cacheable `GET /orders/123` that returns `304` when unchanged, and a contract a human can read. Same requirements, opposite cost profile, and the only difference was *why* they chose.

Here is the map of wrong reasons to the forces they are clumsily groping at:

| Wrong reason (fashion) | The real force underneath (the honest question) |
| --- | --- |
| "gRPC is faster, so always use it" | Is this internal, latency-bound, high-throughput? If yes, gRPC; if it is a public once-a-day pull, the speed win is unmeasurable. |
| "GraphQL is modern / what good companies use" | Do you have many diverse clients with over/under-fetching pain? If not, you are buying N+1 and caching loss for nothing. |
| "REST is old / legacy" | Do you need public reach, free caching, and a low barrier? Those needs are not old; they are permanent for a public API. |
| "Our staff engineer wants gRPC on the resume" | Resume-driven design optimizes the wrong objective. Optimize the contract, not the careers. |
| "Let's standardize on one paradigm everywhere" | Different layers have different forces. Mix by layer; do not anoint a religion. |
| "The team already knows REST" | A real tie-breaker, but only a tie-breaker — never strong enough to override a dominant force like internal microsecond latency. |

![a matrix mapping wrong fashion reasons like gRPC is faster and GraphQL is modern to the real force underneath each that gives the honest answer](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-6.png)

The matrix figure renders that same map as a grid: each fashion reason on the left, the force that should have driven the decision on the right. The pattern is that *every* fashion reason is a force wearing a costume. "gRPC is faster" is "this is internal and latency-bound" in disguise. "GraphQL is modern" is "I have many diverse clients" in disguise. Strip the costume and the force answers. The one row that is genuinely a legitimate input — "the team already knows it" — is marked as a tie-breaker, not a driver, because tooling and familiarity *are* real forces (debuggability, velocity, the operational cost of a paradigm nobody on-call understands) but they are weak forces. They break ties between paradigms that the dominant forces left close; they never override a dominant force. A team that "knows REST" but is shipping an internal, microsecond-latency, polyglot, streaming service should learn gRPC, not bend the service to their comfort.

The deepest wrong reason of all is *consistency for its own sake* — "we should use one paradigm everywhere so the platform is uniform." It sounds disciplined and it is the opposite. Uniformity across layers with different forces means at least one layer is using the wrong paradigm. The disciplined position is *uniformity of decision process* — every team runs the same force-driven framework — producing a platform that is REST at the edge, gRPC in the core, GraphQL at the BFF, and events at the seams, because that is what the forces selected at each layer.

## Migration paths and the cost of switching

Choosing a paradigm is not a one-way door, but the door is heavy, and the cost of switching is itself a force you must weigh up front. A paradigm leaks into your client SDKs, your monitoring, your team's muscle memory, and your callers' code — and the more public the API, the more the switching cost is dominated by *the callers you cannot change*. So the rule is: weigh the switching cost before you choose, because "we can always change it later" is true and expensive.

![a timeline showing a paradigm migration from REST internal with high p99 to defining a proto contract to dual-running gRPC beside REST to cutting traffic over to deprecating REST with a Sunset header](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-7.png)

The timeline shows the only safe way to switch an *internal* paradigm — say, moving the risk-scoring service from a REST internal contract to gRPC because the JSON tax was blowing the p99. You never flip it overnight. You *define the new contract first* (the `.proto`), *dual-run* the gRPC service beside the REST one in shadow mode so you can compare results on real traffic, *cut traffic over* gradually (10% → 50% → 100%) with a canary, and only then *deprecate* the old REST endpoint with a `Sunset` header and a migration window. For internal callers you own, this is a few weeks of coordinated work. The expand-and-contract discipline is the same one from [deprecation and sunset](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely): add the new path, migrate callers, then remove the old path — never remove before migrate.

For a *public* API the cost is an order of magnitude higher, because the callers are strangers you cannot deploy. Switching a public API's paradigm effectively means running both forever or stranding integrators. This is why public APIs are so conservative about paradigm and why "REST by default for public" is such durable advice: the switching cost is nearly prohibitive, so you want to be right the first time. The realistic public migration is not a paradigm switch at all — it is a *new major version alongside the old*, with both live for years, which is the [versioning](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) story, not a paradigm story.

The cost of switching is not only the wire — it leaks into four places that are easy to forget at decision time. **SDKs and client libraries**: a paradigm change means regenerating or rewriting every SDK you publish, and every code sample in your docs. **Observability**: REST dashboards key on status codes and URL paths; gRPC keys on its own status codes and method names; GraphQL hides errors inside `200` and operations inside one URL — each move forces a monitoring rebuild, and the team flies blind during the transition unless you plan for it. **On-call muscle memory**: the engineer paged at 3 a.m. debugs the paradigm they know; introduce a second paradigm and you have doubled the surface they must understand under stress, which is a real reliability cost that does not show up in a benchmark. **The callers' code and timelines**: even internal callers have their own release trains, and "migrate by Q3" collides with their roadmaps. Add these up and "we can always change it later" is true but routinely costs far more than the engineer who says it expects — which is the whole reason switching cost belongs *in* the original decision, weighted as a force, not deferred as a someday-problem.

The asymmetry is worth stating as a rule: **internal paradigm choices are reversible at moderate cost; public paradigm choices are effectively permanent.** That asymmetry should make you *bolder* internally (try gRPC for the latency-bound service; you can dual-run and roll back) and *more conservative* publicly (default to REST; the door barely opens once strangers walk through it). Many teams get this backwards — they agonize over the internal choice they could easily reverse and ship a public choice they will live with for a decade without running the forces. Spend your decision energy where the door is heaviest.

#### Worked example: classifying the switching cost before you commit

Suppose the mobile team is two years in on a REST API with per-screen endpoints and is drowning in over-fetching — they want to switch to GraphQL. Before committing, classify the switching cost by force:

- **Who must change?** Only first-party clients (iOS, Android, web) we own and ship. *Low blast radius.* If this were public, stop — the cost would be prohibitive.
- **Can we dual-run?** Yes — stand up the GraphQL BFF beside the REST endpoints, migrate one screen at a time, keep REST live until the last screen moves. *Incremental, reversible per screen.*
- **What new operational tax?** N+1 monitoring, query-cost limits, a new error convention on the dashboards, dataloader plumbing. *Real, ongoing, must be staffed.*
- **What is the payoff?** New screens ship without backend changes; round-trips collapse; over-fetching ends. *Matches the dominant force.*

**Decision: migrate, incrementally, behind a BFF, because the blast radius is low (first-party only), the migration is reversible per screen, and the over-fetching force genuinely dominates.** Contrast the same request on a *public* API: same over-fetching pain, but the callers are strangers, the dual-run is forever, and the operational tax now includes defending a query language against hostile strangers. **Decision there: do not switch — add sparse fieldsets to the REST API instead** (the `?fields=` approach from [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql)), which buys most of the over-fetching relief at a fraction of the switching cost. Same symptom, opposite decision, because the *who-must-change* force flipped.

## Case studies: how real platforms run all three

The pattern in the wild is overwhelmingly *polyglot paradigms*, chosen by layer. A few accurate, generally-stated examples — described at the level of public, well-known architecture, not internal specifics I cannot verify.

**GitHub** runs both a **REST API** and a **GraphQL API** as public products, side by side. This is not indecision; it is force-driven layering exposed to the outside. The REST API serves the broad, cacheable, resource-shaped, `curl`-able use cases with the lowest barrier to entry. The GraphQL API serves clients that need to aggregate across many related objects (a repo, its issues, their labels, their assignees) in one query without a cascade of REST round-trips — the over/under-fetching force. They also model deprecation carefully, using the `Sunset` header convention on REST and field-level deprecation on the GraphQL schema, which is exactly the evolvability discipline this series preaches. Two paradigms, two force profiles, one platform.

**Stripe** is the canonical **public REST** payments API, and it is REST for precisely the forces in our public-payments worked example: maximal reach (any HTTP client, any language, zero codegen), a low barrier to entry (a `curl` that charges a card), idempotency keys for safe retries, and a famously disciplined additive-evolution and dated-versioning story so a partner who integrated years ago is not broken by new fields. The whole industry of public payments converged on REST for the same force reasons; it is the strongest single confirmation that the public-API force profile selects REST.

**Google** publishes **gRPC** as its standard for high-performance, polyglot, internal and inter-service communication, and the open-source gRPC project came directly out of that internal practice. Google also publishes the **AIP (API Improvement Proposals)** guide, which prescribes a resource-oriented design that maps cleanly to *both* a REST/JSON surface and a gRPC surface from the same definition (via gRPC-JSON transcoding). That is the gateway-over-gRPC pattern as official guidance: define the resource contract once, serve gRPC internally and REST externally. It is the clearest existing endorsement of "mix by layer, generated from one contract."

**Netflix and many large mobile platforms** popularized the **BFF (backend-for-frontend)** pattern and, increasingly, a **GraphQL federation** layer at the edge, precisely because they have many client form factors (TVs, phones, web, consoles) each wanting a different data shape over many backend services — the diverse-clients force at extreme scale. The backends are typically internal RPC services; the edge is a client-shaped aggregation layer. Same architecture as our super-app worked example, at planetary scale.

The thread through all four: **none of them picked one paradigm.** Each picked, per layer, the paradigm whose forces dominated that layer. That is the framework, validated by the platforms most engineers point to as exemplary. If the best-known platforms run all three by force, "we must standardize on one" is not best practice — it is the absence of the framework.

## When to reach for each (and when not to)

Decisive recommendations, because a framework that will not commit is just a list of considerations. Run the dominant force; let it bind.

**Reach for REST when** the client is a browser, a partner, or the anonymous public; the data is resource-shaped; responses are cacheable; the barrier to entry must be low; and a human will read the wire. This is the default — start here unless a force pushes you off. **Do not reach for REST when** you have a confirmed internal microsecond-latency, high-throughput, streaming need (gRPC) or many heterogeneous clients drowning in over-fetching (a GraphQL BFF).

**Reach for gRPC when** the client is another service in your own fleet; latency and throughput dominate; you have a polyglot fleet that benefits from one `.proto` generating clients in every language; you need real streaming (especially bidirectional); and no human or browser reads the wire. **Do not reach for gRPC when** the client is a browser (you need a proxy and lose native caching and debuggability), the API is public (you are imposing a binary protocol and codegen on strangers), or the latency win is unmeasurable at your call volume. Never adopt gRPC on the slogan "it is faster" without the latency-bound, high-throughput, internal force to back it.

**Reach for GraphQL when** you have many diverse, first-party clients aggregating data across multiple services, and over/under-fetching is a real, recurring pain — almost always as a BFF over internal services. **Do not reach for GraphQL when** you have one client with one stable shape (use REST; you are buying N+1 and caching loss for nothing), the API is public and anonymous (you must add depth limits, query-cost analysis, and persisted-query allowlists to be safe — only do this with eyes open), or the dominant force is raw service-to-service latency (use gRPC).

**Reach for events / webhooks when** the interaction is asynchronous and notify-shaped — "tell the partner the charge settled," "let the Orders service know payment completed" — rather than a synchronous request the caller must wait on. **Do not** make the caller poll a synchronous endpoint for a state change that the server already knows; push it. The mechanics live in [event-driven and async APIs](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi).

**Reach for a mix — and you almost always should — when** your platform has more than one of the above force profiles, which any platform of size does. gRPC in the core, REST at the public edge, GraphQL at the first-party BFF, events at the seams. **Do not** flatten that into one paradigm for the sake of tidiness; uniformity across differing forces guarantees at least one layer is wrong.

![a decision tree resolving the paradigm in one pass from public client to REST or GraphQL and internal only to gRPC or events selected by the first binding constraint](/imgs/blogs/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force-8.png)

The final figure is the whole framework collapsed to one pass. Start at the top — *public client or internal only?* A public client splits to REST (one shape, cacheable, broad reach) or a GraphQL edge (diverse shapes, many clients). An internal-only client splits to gRPC (synchronous RPC, latency, streaming) or events (fire-and-forget, async notify). The key property is that the *first binding constraint* selects the paradigm: you do not need to weigh all nine forces equally, you walk them in order of decisiveness and stop at the first one that hard-constrains the choice. Public-and-anonymous binds you to REST before you ever ask about caching. Internal-and-microsecond-bound binds you to gRPC before you ask about polyglot. That is why the framework resolves fast in practice — the dominant force is usually obvious once you ask *who is the client* honestly.

## Key takeaways

- **Choose by force, not fashion.** A paradigm is a response to forces — who the client is, latency, caching, streaming, diversity, evolvability, public vs internal — not a taste or a trend. Name the forces and the choice mostly makes itself.
- **Who is the client is the most decisive force.** Browser or partner or public → REST; another service in your fleet → gRPC; many diverse first-party clients → GraphQL. Ask it first and ask it honestly.
- **The trade-offs are conservation laws, not bugs.** Free HTTP caching is incompatible with client-shaped queries; compact binary is incompatible with human-readable wire. No paradigm will ever "fix" its weakness without surrendering its strength, which is why force-driven choice is permanent advice.
- **REST is the default; leave it only under a force.** It wins reach, caching, debuggability, and the lowest barrier to entry — exactly the public-API force profile. The most common mistake is leaving REST for a problem REST did not have.
- **gRPC is for the internal fleet, not because it is "faster."** Its speed advantage is real on internal, high-throughput, latency-bound, polyglot, streaming paths and unmeasurable on a public once-a-day pull. "gRPC is faster so always use it" is the top wrong reason.
- **GraphQL solves client diversity and aggregation, not speed.** Use it — almost always as a BFF — when many clients aggregating across many services drown in over/under-fetching. One stable client with one shape gets the N+1 and the caching loss for nothing.
- **You almost always mix, by layer.** gRPC in the core, REST at the public edge, GraphQL at the first-party BFF, events at the seams. Paradigm choice is per-layer, not per-platform; the BFF is the seam where edge paradigm meets core paradigm.
- **Every wrong reason is a force in costume.** Strip the costume — "modern," "faster," "legacy," "on my resume," "let us standardize" — and the force underneath gives the honest answer. Tooling and familiarity are real but weak forces: tie-breakers, never drivers.
- **Weigh the switching cost up front.** Internal paradigm switches are a staged, dual-run, deprecate-with-a-Sunset migration of a few weeks; public switches are nearly prohibitive, which is why public APIs are conservative — be right the first time.

## Further reading

- **Within this series:** the intro hub [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and the capstone [the API design playbook: a review checklist from first endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2); the Track E siblings [RPC vs REST: when a procedure beats a resource](/blog/software-development/api-design/rpc-vs-rest-when-a-procedure-beats-a-resource), [gRPC and Protocol Buffers: contracts, codegen, and streaming](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming), [GraphQL: the query language, schema, and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap), [event-driven and async APIs](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi), and [API gateways: routing, auth, rate limiting, and the BFF pattern](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern).
- **The distributed-systems view (read alongside this post):** [API design: REST, gRPC, GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) and [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) — this post goes deeper on the decision; those go wider on the systems context.
- **The BFF in a service fleet:** [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), and for the event seams, [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).
- **Canonical specifications:** RFC 9110 (HTTP Semantics), RFC 9457 (Problem Details for HTTP APIs), the gRPC documentation and the Protocol Buffers language guide, the GraphQL specification, and the OpenAPI 3.1 specification.
- **Style guides and design guidance:** Google's API Improvement Proposals (AIP) for resource-oriented design that serves both REST and gRPC, the Zalando RESTful API guidelines, and the Microsoft REST API guidelines.

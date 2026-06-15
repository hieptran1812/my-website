---
title: "API Design: REST vs gRPC vs GraphQL, and the Contracts That Outlive Them"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a senior chooses REST, gRPC, or GraphQL deliberately, designs pagination, idempotency, errors, and versioning that survive years of change, and optimizes payloads and round-trips with real numbers."
tags:
  [
    "system-design",
    "api-design",
    "rest",
    "grpc",
    "graphql",
    "architecture",
    "distributed-systems",
    "scalability",
    "pagination",
    "idempotency",
    "versioning",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/api-design-rest-grpc-graphql-1.webp"
---

Here is the uncomfortable truth a senior engineer carries into every API design review: the code you write today is the cheapest thing in the room, and the API you draw on the whiteboard is the most expensive. You can rewrite the implementation behind an endpoint a dozen times over its life and nobody outside your team will ever notice. But the moment a second party — a mobile app you ship to the App Store, a partner integration, another team's service, a webhook a customer's CFO depends on — starts calling your API, you have handed them a veto over its shape. Every field name, every status code, every pagination parameter, every nullability decision becomes a contract you cannot unilaterally change. The implementation is soft clay. The contract is concrete the instant it sets.

That asymmetry is the whole reason API design deserves to be treated as a first-class architectural decision rather than something you bang out while building the feature. The decision of *which protocol* — REST, gRPC, or GraphQL — gets the headlines, but it is genuinely the easy part once you know who the consumer is. The hard parts are the ones nobody asks about in the interview and everybody pays for in production: how you paginate so deep pages don't melt your database, how you make a payment endpoint safe to retry, how you model errors so a client can branch on them three years from now, and how you evolve a schema without a coordinated flag-day migration across teams you don't control. This article is about all of it, told the way a staff engineer would defend the design in review.

![A decision matrix comparing REST, gRPC, and GraphQL across browser support, typing, streaming, payload size, caching, and learning curve](/imgs/blogs/api-design-rest-grpc-graphql-1.webp)

The matrix above is the map for the first half of this piece. No protocol wins every row — REST owns browser support and caching, gRPC owns typing and payload size, GraphQL owns the chatty-client problem — which is exactly why "which is best" is the wrong question. The right question is "which consumer am I serving, and what does that consumer make cheap?" By the end you will be able to make that call deliberately, then design the durable parts of the contract — pagination, idempotency, errors, versioning, evolution — so that the API you ship can outlive three rewrites of the service behind it.

## 1. The contract is the architecture

Start with a reframing that changes how you treat every endpoint. An API is not a feature. It is a **contract** — a promise about a request shape, a response shape, and a set of guarantees (this is idempotent, this is paginated this way, this error means that) that other people build load-bearing code against. The architectural weight of an API comes entirely from this: a contract is the hardest thing in a system to change, because changing it requires coordinating every party who depends on it, and you frequently cannot even enumerate those parties.

A senior internalizes a single rule from this: **design the contract for the change you cannot avoid, and protect it from the change you can.** You will change the database behind an endpoint. You will change the language. You will re-shard, re-cache, rewrite the service three times. None of that should be visible at the contract. What *will* be visible — what you must design for — is the contract's own evolution: new fields, deprecated fields, new versions, new error cases. The contract has to be a thing that grows additively, never a thing that breaks.

This is why API design belongs in the same mental bucket as choosing a partition key or a public identifier scheme — the genuinely irreversible decisions. The companion piece on [evolutionary architecture and designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) lays out the cost-of-change curve: a decision costs roughly 1× to reverse at design time, 10× once in code, 100× once in data, and 1000× once it has leaked into a public contract. APIs live at the far-right, most-expensive end of that curve. That is not a reason to over-design them; it is a reason to design the *evolution path* deliberately so that the expensive reversals are rarely needed.

Three properties separate a contract that ages well from one that becomes a millstone:

- **It is explicit.** The shape, the types, the nullability, the error cases, and the guarantees are written down — ideally in a machine-readable schema (OpenAPI, protobuf, GraphQL SDL) that generates clients and validates requests. Implicit contracts ("the field is usually present") rot, because no two consumers agree on the unwritten parts.
- **It is evolvable.** Adding a field, a new optional parameter, or a new endpoint does not break existing callers. This is a property you design in from the first version, not something you bolt on when you need version 2.
- **It is honest about failure.** Errors are part of the contract, not an afterthought. A client that can only handle the happy path is a client that will page someone at 3 a.m. when your error shape surprises it.

Everything else in this article is in service of those three properties. The protocol choice affects how you achieve them. The patterns — pagination, idempotency, versioning — are how you make a specific contract durable. Let's start with the choice, because it constrains everything downstream.

## 2. REST: resources, HTTP semantics, and the power of boring

REST is the default for a reason, and the reason is not inertia — it is that REST leans on a protocol the entire internet already understands. When you model your API as **resources** identified by URLs and manipulated with HTTP verbs, you inherit an enormous amount of infrastructure for free: caching proxies, CDNs, browser fetch, `curl`, load balancers that route on path, and a generation of engineers who already know what `GET /users/42` means.

The discipline of REST is that the **HTTP method carries the semantics**, and getting those semantics right is most of what separates a real REST API from "JSON over HTTP." The method tells every intermediary — not just your server, but every proxy and cache in between — what it is allowed to do with the request:

- `GET` is **safe** (no side effects) and **idempotent** (calling it N times equals calling it once). A cache can serve it; a proxy can retry it; a crawler can hit it freely.
- `PUT` and `DELETE` are idempotent but not safe: `PUT /users/42` with the same body twice leaves the same final state, so a client that times out can safely retry.
- `POST` is neither safe nor idempotent by default — `POST /charges` twice creates two charges. This is the single most important fact about REST safety, and §6 on idempotency keys is entirely about taming it.

REST's other load-bearing principle is **statelessness**: every request carries everything the server needs to process it (auth token, parameters), and the server holds no client session between requests. This is what makes REST horizontally scalable without ceremony — any request can go to any server, so you can put a fleet of identical stateless instances behind a load balancer and scale by adding boxes. The session state, if any, lives in a shared store (a token, a cache, a database), not in the process. If you want the full picture of how that fleet gets fronted, the companion piece on [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) covers the routing layer that statelessness unlocks.

A small, real OpenAPI fragment shows the explicit-contract property in action. This is the kind of thing you check into the repo and generate clients from:

```yaml
# openapi.yaml — a paginated, cacheable list endpoint
paths:
  /v1/orders:
    get:
      summary: List orders for the authenticated account
      parameters:
        - name: limit
          in: query
          schema: { type: integer, default: 25, maximum: 100 }
        - name: cursor
          in: query
          description: Opaque cursor from a previous page's next_cursor
          schema: { type: string }
      responses:
        "200":
          description: A page of orders
          headers:
            ETag:
              schema: { type: string }
            Cache-Control:
              schema: { type: string, example: "private, max-age=30" }
          content:
            application/json:
              schema:
                type: object
                required: [data, next_cursor]
                properties:
                  data:
                    type: array
                    items: { $ref: "#/components/schemas/Order" }
                  next_cursor:
                    type: string
                    nullable: true
```

Notice what this contract already commits to: cursor-based pagination (not offset), an `ETag` for caching, a `Cache-Control` policy, and a `next_cursor` that is explicitly nullable to signal the last page. Those choices are the durable part. The handler behind them can be Go today and Rust next year.

What about HATEOAS — hypermedia as the engine of application state, the part of Roy Fielding's REST thesis where responses embed links telling the client what it can do next? Be honest about it: almost nobody implements full HATEOAS, and almost nobody needs to. The idea is elegant — a client that follows links instead of constructing URLs is decoupled from your URL structure — but in practice clients hardcode URL templates, and the cost of full hypermedia rarely pays back. A senior takes the *useful* slice (include a `next_cursor` or a `next` link for pagination, include relevant resource URLs in responses) and skips the dogma. REST without HATEOAS is still REST, and it is what the overwhelming majority of production "REST" APIs actually are.

One more piece of REST discipline pays off precisely because it's about evolution: **content negotiation** and the status-code vocabulary. The `Accept` header lets a client say what representation it wants and the server respond accordingly — the seam where header-based versioning lives, and the place to add a new response format without disturbing the old one. And the status codes are themselves a contract: a `200` is not a `201` is not a `202`, and a senior uses them precisely — `201 Created` with a `Location` header for resource creation, `202 Accepted` for an async operation that hasn't finished, `204 No Content` for a successful delete, `409 Conflict` for a write that lost a race, `412 Precondition Failed` for a failed conditional update. Clients branch on these. Collapsing everything into `200` and `500` throws away a vocabulary the entire web already speaks, and forces every client to parse the body to learn what a bare status code would have told it for free. The conditional-request codes especially — `304 Not Modified` for a matched `ETag`, `412` for a failed `If-Match` — are how REST does optimistic concurrency and caching at the protocol layer, which is exactly the free infrastructure that makes REST cheap on read-heavy public endpoints.

REST's costs are real and worth naming so you don't recommend it blindly. JSON is verbose and untyped — a field can silently change from a number to a string and no compiler catches it, so a client deserializing into a typed struct discovers the mismatch at runtime in production rather than at build time. There is no built-in streaming beyond Server-Sent Events. The resource model fights you when the client needs data that doesn't map cleanly to one resource — the profile screen that needs a user plus posts plus likes plus authors becomes a waterfall of resource fetches, which is exactly the gap GraphQL was invented to fill. And the lack of an enforced schema means the contract lives partly in documentation and convention, so it drifts unless you discipline it with OpenAPI and contract tests. But for a public API consumed by browsers and third parties, REST's compatibility with the entire web — caches, CDNs, proxies, `curl`, every HTTP client ever written — is a moat the other two protocols cannot cross, and that moat is worth more than typing for an audience you don't control.

## 3. gRPC: HTTP/2, protobuf, and the internal fast lane

gRPC is what you reach for when the consumer is **another one of your own services** and you care about throughput, latency, and strong typing more than about browser reach. It is built on three pillars that together make it the dominant choice for east-west, service-to-service traffic inside a system.

The first pillar is **protobuf** — a binary, schema-first serialization format. You write the contract as a `.proto` file, and `protoc` generates strongly-typed client and server stubs in every language you use. The wire format is compact (no field names on the wire, just tag numbers and packed bytes) and the typing is enforced at compile time, so a field-type mismatch is a build error, not a 2 a.m. incident.

```protobuf
// order.proto — the contract IS this file
syntax = "proto3";
package orders.v1;

service OrderService {
  rpc GetOrder(GetOrderRequest) returns (Order);
  rpc ListOrders(ListOrdersRequest) returns (ListOrdersResponse);
  // Server-streaming: push order status updates as they happen
  rpc WatchOrder(GetOrderRequest) returns (stream OrderEvent);
}

message Order {
  string id = 1;
  string account_id = 2;
  int64 amount_cents = 3;
  string currency = 4;
  OrderStatus status = 5;
  // field 6 reserved for future use — see §8 on evolution
}

message ListOrdersRequest {
  int32 page_size = 1;      // server caps this
  string page_token = 2;    // opaque cursor
}

enum OrderStatus {
  ORDER_STATUS_UNSPECIFIED = 0;  // proto3 enums MUST have a zero default
  ORDER_STATUS_PENDING = 1;
  ORDER_STATUS_PAID = 2;
  ORDER_STATUS_REFUNDED = 3;
}
```

The second pillar is **HTTP/2**, which buys multiplexing (many concurrent requests over one TCP connection without head-of-line blocking at the HTTP layer), binary framing, header compression, and — crucially — **streaming**. gRPC supports four call shapes: unary (one request, one response), server-streaming (one request, a stream of responses — the `WatchOrder` above), client-streaming, and bidirectional streaming. That makes it a natural fit for things REST has to bolt on awkwardly: live status feeds, telemetry ingestion, chat.

The third pillar is **codegen**. Because the contract is a schema, you don't hand-write clients — you generate them. A service team publishes its `.proto`, every consumer regenerates a typed stub, and the compiler catches breaking changes at build time across the whole fleet. This is the property that makes gRPC scale to hundreds of services: the contract is enforced mechanically, not by code review and hope.

There's a fourth strength that doesn't get a pillar but matters enormously in a mesh: gRPC has **deadlines and cancellation** built into the protocol. A caller sets a deadline on a call, and that deadline *propagates* down the call chain — if `order-service` gives `pricing-service` 50 ms and `pricing-service` calls `tax-service`, the remaining budget flows with the call, and when the deadline expires the whole chain is cancelled cooperatively instead of doing work nobody is waiting for anymore. This is the protocol-level support for the timeout-and-budget discipline that prevents the classic cascading failure where a slow dependency ties up threads all the way up the stack. Doing the same thing over REST requires bolting deadline headers onto every service and hoping everyone honors them; in gRPC it's part of the contract and the generated code handles it. For a service mesh where one user request fans out across a dozen services, deadline propagation is the difference between a localized slowdown and a fleet-wide thread exhaustion.

The cost of gRPC is the flip side of its strengths. It is **not natively browser-friendly** — browsers can't speak raw gRPC's HTTP/2 framing from `fetch`, so browser clients need gRPC-Web plus a proxy (Envoy) that translates, which adds a hop and operational surface. The binary wire format is **not human-readable** — you can't `curl` it and eyeball the response, which makes debugging heavier (you need `grpcurl` and the schema). And HTTP caching by intermediaries is essentially off the table, because the semantics live in the protobuf payload, not in cacheable HTTP verbs and URLs. None of these matter for internal service-to-service traffic. All of them matter for a public API. That asymmetry is the entire decision.

#### Worked example: gRPC vs REST payload and round-trip math for a service mesh

Make the "gRPC is faster internally" claim concrete with numbers, because a senior never recommends a protocol without quantifying the win. Take a service mesh where an `order-service` calls a `pricing-service` to price a cart, and this call happens **40,000 times per second** at peak across the fleet.

The response is a small object: an order ID, an account ID, an amount, a currency, a status, and a timestamp. Serialized as JSON with field names, whitespace-free, it runs about **220 bytes**. The same object as protobuf — tag numbers instead of field names, varint-packed integers, the binary status enum — runs about **45 bytes**. That is roughly a **4.9× reduction** in payload size for this shape, and small objects are where protobuf wins most because the field-name overhead dominates.

Now the bandwidth math. At 40,000 QPS, the response stream alone is:

- JSON: 40,000 × 220 bytes ≈ **8.8 MB/s ≈ 70 Mbps** of egress for just the bodies.
- protobuf: 40,000 × 45 bytes ≈ **1.8 MB/s ≈ 14 Mbps**.

That is **~56 Mbps saved** on one call path, before counting request bodies and headers. Across a mesh with dozens of such call paths, the cumulative cross-AZ traffic reduction is real money — cross-availability-zone traffic is billed, and at cloud rates of roughly \$0.01–\$0.02 per GB, shaving 56 Mbps continuously is on the order of **\$150–\$300/month per call path** in transfer alone, multiplied across the mesh.

The round-trip win is subtler but bigger. Over HTTP/1.1 with REST, each of those 40,000 calls might contend for a connection from a pool, and a saturated pool serializes requests (head-of-line blocking at the connection level). Over HTTP/2, gRPC **multiplexes** all 40,000 logical streams over a small handful of long-lived connections, so there is no per-call connection setup and no pool starvation. In load tests of this exact shape, teams routinely measure gRPC shaving **2–5 ms off p99** versus pooled HTTP/1.1 JSON, and far more under connection-pool pressure where REST's tail blows out to tens of milliseconds while gRPC stays flat. The lesson is not "gRPC is always faster" — it is that for high-fan, small-payload, internal traffic, gRPC's payload and multiplexing wins compound into measurable latency and dollar savings, which is precisely the workload it was designed for.

## 4. GraphQL: client-driven queries and the round-trip collapse

GraphQL exists to solve a specific, painful problem that REST creates: the **chatty client**. Picture a mobile app rendering a profile screen. It needs the user, the user's last ten posts, the like count on each post, and the author of each comment. In REST, that is `GET /users/42`, then `GET /users/42/posts`, then a `GET` per post's likes, then a `GET` per comment's author — a waterfall of round-trips, each paying mobile-network latency (50–150 ms on a bad cell connection), and most of them over-fetching fields the screen never shows. GraphQL collapses that whole waterfall into **one request** where the client declares exactly the shape of data it wants, and the server returns exactly that shape — no more, no less.

```graphql
# One query, one round-trip — the client declares the exact shape it needs
query ProfileScreen {
  user(id: "42") {
    name
    avatarUrl
    posts(first: 10) {
      edges {
        node {
          title
          likeCount
          comments(first: 3) {
            edges { node { text author { name } } }
          }
        }
      }
    }
  }
}
```

The wins are genuine and explain GraphQL's adoption at companies with many client teams and fast-moving product surfaces. **One round-trip** instead of a waterfall is a transformative latency win for mobile. **No over-fetching** — the client gets only the fields it asks for, which matters when payload size is bandwidth on a metered connection. And **client autonomy** — a new screen that needs a new combination of existing fields requires zero backend changes, because the client just asks for a different shape. That last property is why product-heavy orgs love GraphQL: it decouples client iteration speed from backend deploy cycles.

But GraphQL moves the hard problems from the client to the server, and a senior has to see those problems clearly before recommending it, because they are exactly the ones that cause production incidents.

![A graph showing one GraphQL client query fanning out through a schema executor into user, posts, likes, and author resolvers with an N plus one risk](/imgs/blogs/api-design-rest-grpc-graphql-7.webp)

The figure shows the structural reality: one client query becomes a **tree of resolver calls** on the server. The `user` field calls a resolver, `posts` calls another, and crucially `comments` and `author` call resolvers *per parent row*. This is the **N+1 problem**, and it is GraphQL's defining failure mode. If 100 posts come back and each post's `author` field triggers its own database query, one innocent-looking client query becomes 1 + 100 = 101 database round-trips. The client saved a round-trip; the server traded it for a hundred. §7 is entirely about defusing this with dataloaders, because every GraphQL deployment hits it.

The second GraphQL-specific hazard is the **complexity attack**. Because the client controls the query shape, a malicious or careless client can request a deeply nested, exponentially expanding query — `posts { comments { author { posts { comments { ... } } } } }` — that forces the server to do enormous work for a tiny request. A public GraphQL endpoint without query-cost analysis, depth limiting, and rate limiting keyed on *query cost* (not request count) is a denial-of-service waiting to happen. This is the dark mirror of GraphQL's flexibility: the same power that lets a good client fetch exactly what it needs lets a bad client ask for everything at once.

The third cost is **caching**. REST gets HTTP caching for free because a `GET /users/42` is a stable, cacheable URL. GraphQL queries are almost always `POST` with a body, and every query shape is different, so the dumb HTTP-layer caching that REST enjoys simply doesn't apply. GraphQL caching has to move into the application — persisted queries, response caching keyed on the query plus variables, and per-field caching — all of which you build and operate yourself.

So GraphQL is not "better REST." It is a different set of trade-offs: you gain client-driven flexibility and round-trip collapse, and you pay with server-side N+1 risk, query-cost management, and the loss of free HTTP caching. It wins decisively for **flexible, fast-moving clients with many screens and teams** — which is why it shows up at Facebook (where it was born), GitHub, Shopify, and the like. It is overkill for a simple internal CRUD API and a poor fit for high-throughput service-to-service traffic where gRPC's typing and binary efficiency dominate.

## 5. The decision: matching protocol to consumer

Now put the three together into a decision you can defend in a review. The mistake juniors make is treating this as a popularity contest ("everyone's using GraphQL now"). The senior move is to make it a function of **who consumes the API and what that consumer makes cheap.**

![A decision tree routing from who calls the API to REST for public browsers, GraphQL for chatty mobile clients, and gRPC for internal high-throughput services](/imgs/blogs/api-design-rest-grpc-graphql-2.webp)

The tree above encodes the reasoning. The root question is never "which protocol is best" — it is "who calls this?" From there the branches are clean:

- **Public API, consumed by browsers and third parties → REST.** You need the web's caching, browser `fetch`, `curl`-ability, and the lowest possible integration cost for outside developers. Stripe, Twilio, GitHub's REST API — the entire public-API economy runs on REST because REST minimizes the consumer's cost.
- **Internal, high-throughput, service-to-service → gRPC.** You control both ends, you want strong typing and codegen to enforce contracts mechanically across a fleet, and you want protobuf's payload efficiency and HTTP/2 multiplexing for the latency and bandwidth wins quantified in §3. This is the east-west backbone of most modern service architectures.
- **Flexible client (especially mobile) with many screens and fast product iteration → GraphQL.** When the binding constraint is client-team velocity and round-trip latency on flaky networks, GraphQL's one-query-fetches-exactly-what-I-need model pays for its server-side complexity.

These are not mutually exclusive — large systems run all three. A common and entirely sane architecture is **GraphQL or REST at the edge** (public, client-facing), translating into **gRPC behind the gateway** (internal, service-to-service). The edge speaks the consumer's language; the interior speaks the efficient one. The BFF pattern in §10 is exactly this seam.

Here is the full trade-off matrix as a table, because a markdown table is sometimes the clearest form and the series demands an explicit one:

| Property | REST | gRPC | GraphQL |
|---|---|---|---|
| Native browser support | Yes | No (needs gRPC-Web + proxy) | Yes |
| Strong typing / schema | Optional (OpenAPI) | Enforced (protobuf) | Enforced (SDL) |
| Streaming | SSE only | Bidirectional native | Subscriptions (extra infra) |
| Payload size | JSON (verbose) | protobuf (compact) | JSON (verbose) |
| HTTP caching | Free (GET + URL) | Effectively none | Hard (POST, varied shapes) |
| Over/under-fetching | Common | Per-RPC shape | Client controls exactly |
| N+1 risk | Client-side waterfall | Low | Server-side, severe |
| Learning curve | Low | High | Medium-high |
| Best consumer | Public / browser / 3rd party | Internal service-to-service | Chatty multi-screen clients |

The decisive columns flip depending on the row that matters most for *your* consumer. If your binding constraint is "third parties must integrate in an afternoon," browser support and learning curve dominate and REST wins. If it is "this call happens 40k times a second between our own services," payload size and typing dominate and gRPC wins. If it is "our six mobile teams ship new screens weekly," client-controlled fetching dominates and GraphQL wins. There is no universal answer because there is no universal consumer.

## 6. Idempotency keys: making mutations safe to retry

Now we leave the protocol war and enter the territory that separates a toy API from a production one. The single most important property a mutating endpoint can have — and the one most APIs get wrong — is **idempotency**: the guarantee that submitting the same operation twice has the same effect as submitting it once.

Why does this matter so much? Because the network is unreliable, and the failure mode is invisible to the client. A client sends `POST /charges` to charge a customer \$50. The server creates the charge, debits the card, and starts sending the `200 OK` back — and then the connection drops before the response arrives. The client now faces an impossible question: *did the charge happen?* It got no response. If it retries, it risks charging \$100. If it doesn't, it risks the customer never being charged. Without idempotency, the client cannot safely do anything. This is not a rare edge case; at scale, with millions of requests, "the response got lost after the work was done" happens constantly.

The fix is the **idempotency key**: the client generates a unique key (a UUID) for the logical operation and sends it with the request. The server records "I have seen this key, and here is the result I returned." If the same key arrives again, the server returns the **stored original result** instead of performing the operation a second time.

```python
# Server-side idempotency: the key makes a POST safe to retry
import hashlib, json

def create_charge(request, db):
    key = request.headers.get("Idempotency-Key")
    if not key:
        return error(400, "Idempotency-Key header required")

    # Fingerprint the request body so a reused key with a DIFFERENT body is caught
    body_fingerprint = hashlib.sha256(request.raw_body).hexdigest()

    existing = db.idempotency_records.get(key)
    if existing:
        if existing["fingerprint"] != body_fingerprint:
            # Same key, different request = client bug. Refuse, do not double-charge.
            return error(422, "Idempotency-Key reused with a different request body")
        # Replay the stored response — no second charge happens
        return existing["status"], existing["response"]

    # First time we've seen this key: do the work inside a transaction
    with db.transaction():
        charge = perform_charge(request.body)            # the side effect
        db.idempotency_records.put(key, {
            "fingerprint": body_fingerprint,
            "status": 201,
            "response": charge.to_json(),
            "created_at": now(),
        })
    return 201, charge.to_json()
```

Two subtleties make this robust rather than naive. First, **fingerprint the request body** and reject a reused key with a *different* body — that combination means a client bug (key collision), and silently replaying the old response or silently charging again both hide the bug. Returning `422` surfaces it. Second, **store the record and perform the side effect in the same transaction** (or use the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) if the side effect crosses a boundary), so you never end up in the state where the charge happened but the key wasn't recorded — which would let a retry double-charge.

This is a deep topic with its own failure modes — exactly-once delivery, deduplication windows, what to do about a key that's mid-flight when the retry arrives — and it is the subject of the dedicated companion piece on [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) and the message-queue deep-dive on [idempotency and deduplication that makes at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). The architectural rule to carry out of here is blunt: **every mutating endpoint that a client might retry — every `POST` that creates or charges or sends — must accept an idempotency key, and the contract must say so.** Stripe pioneered this pattern in their public API precisely because payments cannot tolerate a lost-response double-charge, and it has become the industry standard for a reason.

## 7. The N+1 explosion and the dataloader fix

Return to GraphQL's defining failure mode, because it is the most common production incident a GraphQL deployment suffers and the fix is a clean, generalizable pattern. The N+1 problem isn't unique to GraphQL — any time you fetch a list and then fetch a related thing per item, you have it — but GraphQL makes it spectacularly easy to trigger, because the resolver tree fans out automatically and the person writing the client query has no idea how many database calls their innocent-looking query just spawned.

![A before and after comparison showing a naive resolver issuing one hundred and one database round-trips versus a dataloader collapsing them to two](/imgs/blogs/api-design-rest-grpc-graphql-4.webp)

The figure makes the explosion concrete. A query for 100 users, each with their posts, runs the `users` resolver once (1 query) and then the `posts` resolver **once per user** (100 queries) — 101 round-trips for what should be 2. At a database round-trip cost of even 1 ms, that's 100 ms of pure serialized latency added by a pattern the client never sees. At 2 ms per round-trip and a few thousand such queries per second, it is the kind of load that takes a database from comfortable to on-fire.

The fix is the **dataloader** pattern: instead of each resolver firing its own query immediately, resolvers *register the keys they need* during a single execution tick, and a batching function collects all the keys and issues **one batched query** (`WHERE id IN (...)`), then distributes the results back to the waiting resolvers. It also caches within a request, so asking for the same key twice costs one fetch.

```javascript
// DataLoader collapses N per-row fetches into one batched query per tick
const DataLoader = require("dataloader");

// The batch function receives ALL keys collected during one execution tick
const postsByUser = new DataLoader(async (userIds) => {
  // ONE query for every user's posts, instead of one query per user
  const rows = await db.query(
    "SELECT * FROM posts WHERE user_id = ANY($1)",
    [userIds]
  );
  // Re-group results back into the order the keys were requested
  const byUser = new Map(userIds.map((id) => [id, []]));
  for (const row of rows) byUser.get(row.user_id).push(row);
  return userIds.map((id) => byUser.get(id));
});

// In the resolver: just .load(key) — batching happens automatically
const resolvers = {
  User: {
    posts: (user) => postsByUser.load(user.id),
  },
};
```

The result, shown in the figure's "after" column: 1 query for the 100 users plus 1 batched `IN (...)` query for all their posts equals **2 round-trips** instead of 101 — a roughly **50× reduction in database calls** for this shape. The latency that was 100 ms of serialized round-trips collapses to two round-trips, a few milliseconds.

This is the **GitHub and Shopify lesson**. Both run enormous public GraphQL APIs, and both invested heavily in batching, query-cost analysis, and complexity limits precisely because the N+1 explosion and the complexity attack are not theoretical at their scale — they are daily operational realities. GitHub's GraphQL API assigns every query a computed *cost* and rate-limits on that cost rather than on request count, because one expensive query can do the work of a thousand cheap ones. The senior takeaway: **if you ship GraphQL, you ship dataloaders and query-cost limits on day one, not after the first incident.** GraphQL without batching is a database-melting machine with a friendly query language.

## 8. Pagination done right: offset is a trap

Pagination is the most underestimated decision in API design. It looks trivial — "just add `?page=2`" — and that triviality is exactly the trap, because the obvious approach, **offset pagination**, breaks in two distinct and painful ways at scale, and by the time it breaks the parameter is baked into a public contract you can't easily change.

![A before and after comparison showing offset pagination scanning a million rows at one thousand eight hundred milliseconds versus keyset seeking directly to the cursor at four milliseconds](/imgs/blogs/api-design-rest-grpc-graphql-3.webp)

The figure shows the first and worse failure: **offset pagination gets slower the deeper you page.** `SELECT * FROM orders ORDER BY created_at LIMIT 50 OFFSET 1000000` does not magically jump to row one million. The database scans and *discards* the first million rows to find row 1,000,001, then returns the next 50. Page 1 is instant; page 20,000 scans a million rows. The figure's numbers are representative of a real measurement: a deep offset query running ~1,800 ms while the equivalent keyset query runs ~4 ms — roughly a **450× difference**, growing linearly with depth. Most users never page that deep, but crawlers, exports, and "show me everything" jobs do, and they will find the cliff.

The second failure is **correctness**, and it is sneakier. Offset assumes the underlying list is stable between page fetches. It never is. If a new order is inserted at the top while a client is paging, every subsequent offset shifts by one: the client sees a duplicate row (the one pushed from page 1 to page 2) and misses one entirely (the one that moved off the boundary). At scale, with constant inserts, offset pagination *silently drops and duplicates rows*. For an export that's supposed to be complete, that's a data-integrity bug masquerading as a pagination choice.

The fix is **cursor (keyset) pagination**: instead of "skip N rows," you say "give me the rows *after this specific position*." The cursor encodes the sort key of the last row you saw, and the next query seeks directly to it via the index:

```sql
-- Keyset pagination: seek to the cursor, no scanning of skipped rows.
-- Cursor encodes (created_at, id) of the last row from the previous page.
SELECT id, account_id, amount_cents, created_at
FROM orders
WHERE account_id = $1
  AND (created_at, id) > ($2, $3)   -- the decoded cursor; tuple compare
ORDER BY created_at, id
LIMIT 50;
```

Because `(created_at, id)` is indexed, the database **seeks** to the cursor position in `O(log n)` and reads exactly 50 rows — no scanning of skipped rows, so page 20,000 costs the same as page 1. And because the cursor is a *position in the data*, not a count of skipped rows, inserts elsewhere don't shift it: no duplicates, no drops. The trade-off you pay is that cursor pagination gives up random access — you cannot jump to "page 4,732," only "next" and "previous." For the overwhelming majority of APIs (infinite-scroll feeds, exports, syncs) that is no loss at all, and it is why every serious API — Stripe, Slack, GitHub, Twitter — paginates with cursors.

The contract detail that makes this clean: return the cursor as an **opaque token** (`next_cursor: "eyJjcmVhdGVkX2F0Ij..."` — a base64-encoded, possibly signed blob), never as a raw `(timestamp, id)` the client can construct. Opacity means you can change the *internal* cursor encoding later — add a tiebreaker column, switch the sort — without breaking clients, because they only ever echo back the token you gave them. Exposing the raw sort key in the contract would re-create the irreversibility problem this whole article is about.

#### Worked example: designing one durable orders endpoint

Pull the durable patterns together by designing a single endpoint the way you'd present it in review: **a paginated, idempotent, versioned `orders` API** with the actual contract, so the abstract rules become a concrete artifact.

**Listing (read), with cursor pagination and caching:**

```
GET /v1/orders?limit=50&cursor=eyJjcmVhdGVkX2F0Ijo...   HTTP/1.1
Authorization: Bearer <token>
If-None-Match: "ord-page-7f3a9c"          # client's cached ETag
```

```
HTTP/1.1 200 OK
ETag: "ord-page-7f3a9c"
Cache-Control: private, max-age=30
Content-Type: application/json

{
  "data": [ { "id": "ord_8a2", "amount_cents": 4999, "currency": "usd",
              "status": "paid", "created_at": "2026-06-15T10:04:11Z" } ],
  "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNi0wNi0xNVQxMDowNDoxMVoi..."
}
```

If the client's `If-None-Match` matches, the server returns `304 Not Modified` with an empty body — the client serves its cache and you save the serialization and bandwidth entirely. The `next_cursor` is opaque; `null` would signal the last page.

**Creating (mutation), idempotent and versioned:**

```
POST /v1/orders   HTTP/1.1
Authorization: Bearer <token>
Idempotency-Key: a3f1c2e8-7b94-4d2a-9f3e-1c6b8a0d5e72
Content-Type: application/json

{ "account_id": "acct_19", "amount_cents": 4999, "currency": "usd" }
```

```
HTTP/1.1 201 Created
Location: /v1/orders/ord_8a2
Content-Type: application/json

{ "id": "ord_8a2", "status": "pending", ... }
```

The contract commits to: a **URI version** (`/v1/`), **cursor pagination** with an opaque token and explicit nullable `next_cursor`, **ETag/Cache-Control** for conditional caching, an **idempotency key** on the mutating create, a **`Location` header** pointing at the new resource, and **`amount_cents`** (integer minor units, never a float — money in floats is a bug). Every one of those is a durable choice that's cheap now and expensive to retrofit. Notice what is *not* in the contract: nothing about the database, the language, the caching layer, or how orders are stored. Those are free to change. That separation is the whole point.

## 9. Error modeling: failure is part of the contract

A senior judges an API as much by its error responses as its success responses, because the happy path is easy and the error path is where integrations break at 3 a.m. The rule is simple and routinely violated: **errors are part of the contract, and they must be both machine-parseable and human-debuggable.** A client needs to *branch* on the error programmatically (retry? show the user? alert?), and an on-call engineer needs to *understand* it from the response alone.

![A matrix comparing bare HTTP status, problem plus json, gRPC status with details, and GraphQL errors across machine-parseability, human detail, and partial success](/imgs/blogs/api-design-rest-grpc-graphql-9.webp)

The matrix above grades the four common approaches. A **bare HTTP status code** is the floor: `400` tells the client *something* was wrong with the request, but not *what*, so the client can't distinguish "missing field" from "rate limited" from "card declined" — all might be `4xx`. It's machine-parseable only coarsely and carries no human detail. That's not a contract; it's a shrug.

The REST best practice is **`application/problem+json`** (RFC 9457, formerly 7807), a standardized error body that carries a stable machine-readable `type`, a human-readable `title` and `detail`, and arbitrary extension fields:

```json
{
  "type": "https://api.example.com/errors/insufficient-funds",
  "title": "Insufficient funds",
  "status": 402,
  "detail": "The card has insufficient funds for a charge of $49.99.",
  "instance": "/v1/charges/ch_8a2",
  "balance_cents": 3200,
  "required_cents": 4999
}
```

The `type` URI is the part a client branches on — it is stable and documented, so a client can write `if error.type endsWith "insufficient-funds"` and handle it specifically. The `title` and `detail` are for humans. The extension fields (`balance_cents`, `required_cents`) carry the structured context that turns a debugging session into a glance. This is the shape that ages well, because adding a new error `type` is an additive change that never breaks an existing client.

**gRPC** has its own well-designed model: a small set of canonical **status codes** (`NOT_FOUND`, `ALREADY_EXISTS`, `RESOURCE_EXHAUSTED`, `FAILED_PRECONDITION`, ...) that map cleanly to retry semantics, plus a `details` field carrying rich typed protobuf error messages. The status code tells the client *how to react* (a `RESOURCE_EXHAUSTED` says back off and retry with delay; a `FAILED_PRECONDITION` says don't retry, fix the request), and the details carry the specifics. This retry-semantics-in-the-code design is one of gRPC's quiet strengths, and it pairs naturally with [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure), where the `RESOURCE_EXHAUSTED` / `429` signal is the contract between server and client about when to slow down.

**GraphQL** is the odd one out because it returns `200 OK` even on partial failure, with an `errors[]` array alongside whatever `data` could be resolved. This enables **partial success** — a query for ten fields where one resolver fails still returns the other nine plus an error for the failed one — which is genuinely useful for composite queries, but it means HTTP status is useless as a health signal and clients *must* inspect the `errors` array. It's a different contract, and clients have to be written for it.

The cross-cutting senior rules, regardless of protocol:

- **Use a stable, documented error code/type the client branches on.** Free-text messages change; codes must not.
- **Map status to retry semantics.** `429`/`RESOURCE_EXHAUSTED` and `503` are retryable with backoff; `400`/`422`/`FAILED_PRECONDITION` are not — retrying won't help.
- **Never leak internals.** A `500` returns a generic message and a correlation ID, not a stack trace or a SQL error. The correlation ID lets you find the real error in your logs without exposing it to the caller.
- **Include a correlation/request ID in every error.** It's the single most useful field for debugging a distributed system, and it costs nothing to add.

## 10. Gateways, BFFs, and the layers a request crosses

So far we've talked about the contract at the endpoint. Now zoom out to where the request actually flows, because in any real system there are **layers between the client and the service**, and a senior designs those layers deliberately rather than letting them accrete.

![A stack diagram showing an API call descending through client, CDN edge, API gateway, backend for frontend, service, and datastore layers with latency at each](/imgs/blogs/api-design-rest-grpc-graphql-8.webp)

The stack shows the layers and their latency budget. Two of them deserve real attention because they are where senior API design happens: the **API gateway** and the **BFF**.

The **API gateway** is the single front door for cross-cutting concerns that you do *not* want every service reimplementing: authentication and token validation, rate limiting, request routing, TLS termination, request/response logging, and basic transformation. Centralizing these is the win — auth logic lives in one place, rate-limit policy is enforced consistently, and the services behind the gateway can assume "if a request reached me, it's already authenticated and within rate limits." The gateway is also the natural seam where a public REST or GraphQL contract translates into internal gRPC, as in §5: the gateway speaks the consumer's protocol on the outside and the efficient protocol on the inside.

The **BFF (backend-for-frontend)** is a more specific and often-misunderstood pattern, so be precise about it. A BFF is an API layer dedicated to *one* client type — a mobile BFF, a web BFF — whose entire job is to **shape the backend's data for that specific client's needs.** Why does this exist? Because a mobile app and a web app have genuinely different needs from the same backend: mobile wants fewer round-trips and smaller payloads (bandwidth and battery), web can afford chattier calls and richer responses. A single generic API serving both ends up either too chatty for mobile or too coarse for web. The BFF lets each client have an API tailored to it, aggregating and reshaping the responses from the same underlying services.

![A pipeline showing a mobile client request crossing an API gateway and a backend for frontend that shapes the payload before fanning out to internal gRPC services](/imgs/blogs/api-design-rest-grpc-graphql-5.webp)

The pipeline shows it end to end: the mobile client makes **one request** to the gateway (auth, rate limit), which routes to the **mobile BFF**, which **fans out to three internal services over gRPC**, aggregates and trims the responses, and returns exactly the shape the mobile screen needs — one round-trip from the client's perspective, the fan-out hidden inside the data center where round-trips are cheap (sub-millisecond, on a fast internal network) rather than across a cell connection where they're tens of milliseconds.

This is where the protocols compose into a coherent architecture. The client-facing edge speaks REST or GraphQL (whatever the consumer makes cheap); the gateway handles cross-cutting concerns; the BFF aggregates for a specific client; and the interior speaks gRPC for the efficiency quantified in §3. Notice the BFF and GraphQL solve the *same* problem — collapsing client round-trips and shaping payloads — by different means: GraphQL puts the shaping power in the client's query language, while a BFF puts it in a client-specific backend the platform team controls. Which you choose is itself a trade-off: GraphQL gives client teams autonomy at the cost of server-side complexity; a BFF gives the platform team control at the cost of a service per client type.

## 11. Versioning and evolution: never break a contract

We've arrived at the hardest and most important part: how a contract **changes over time** without breaking the people who depend on it. This is where the cost curve from §1 bites hardest, because by definition you're changing the thing at the expensive end of it. The governing principle is absolute: **never break a published contract.** Not "break it carefully," not "break it with notice" — never. You evolve it additively, and when you genuinely must make an incompatible change, you do it through a window where both shapes coexist.

Start with **versioning strategy**, because it's the first decision. There are three common approaches:

- **URI versioning** (`/v1/orders`, `/v2/orders`): the version is in the path. It's the most visible, the easiest to route and cache, and the easiest for a developer to reason about — you can see the version in the URL. The cost is that a "version" tends to mean the *whole* API, so bumping `v1` to `v2` is a heavy, all-or-nothing event. This is what Stripe, GitHub, and most public APIs use, because visibility and routability win for public consumers.
- **Header versioning** (`Accept: application/vnd.example.v2+json` or a custom `API-Version` header): the version travels in a header, keeping URLs clean and version-free. It's more RESTful in theory (the URL identifies the resource, not the representation), but it's less visible, harder to test by hand, and some caches and proxies handle it less gracefully.
- **No global version — continuous evolution**: don't version the whole API at all; evolve every endpoint additively forever, and use the expand-contract pattern below for the rare incompatible change. This is the most operationally pleasant for both you and your consumers when you can pull it off, because nobody ever does a "migrate to v2" project.

But here is the senior insight that the versioning-strategy debate usually misses: **the version number is a fallback, not the primary tool.** The primary tool is **additive, backward-compatible evolution** — and if you do that well, you rarely need a new version at all. The vast majority of changes an API needs are additive: a new field in a response, a new optional parameter, a new endpoint, a new enum value. None of those need to break anyone *if your clients are written to tolerate them* — which means clients must **ignore unknown fields** and not choke on new enum values. This is forward/backward compatibility, and it's a property of both sides of the contract.

protobuf bakes these rules into the format, which is one of gRPC's underrated strengths for evolution:

- **Field numbers are the contract, not field names.** You can rename a field freely (names aren't on the wire); you must never reuse or change a field's *number*.
- **Adding a field is always safe.** Old clients ignore unknown field numbers; new clients reading old messages see the field's default. Additive change is the default mode.
- **Never reuse a removed field's number**, and mark it `reserved` so nobody accidentally does — reusing a number means an old client interprets new data as the old field, a silent and nasty data-corruption bug.

```protobuf
message Order {
  string id = 1;
  string account_id = 2;
  int64 amount_cents = 3;
  string currency = 4;
  OrderStatus status = 5;
  string idempotency_key = 7;   // ADDED in a later version — safe, old clients ignore it
  reserved 6;                    // a field we removed; number 6 is burned forever
  reserved "legacy_amount";      // and its old name, so nobody re-adds it
}
```

For the genuinely incompatible change — you must rename a field clients read, change a type, or split one endpoint into two — the technique is **expand-contract** (also called parallel-change), and it is the most important evolution pattern a senior knows.

![A before and after comparison showing a hard field rename breaking old clients versus an expand contract migration that adds, dual-writes, then removes](/imgs/blogs/api-design-rest-grpc-graphql-6.webp)

The figure contrasts the two paths. The **before** column is the rookie move: rename `amount` to `amount_cents` in place, ship it, and watch every old client that still reads `amount` get back `null` or a `400` — a coordinated outage where you can't deploy the server until every client is updated, which for a public API with third-party consumers is *never*. The **after** column is expand-contract in three phases:

1. **Expand**: add the new field (`amount_cents`) alongside the old one (`amount`). The response now carries *both*. Every existing client keeps working because `amount` is still there; new clients can start reading `amount_cents`.
2. **Migrate** (the patient middle): dual-write both fields, announce the deprecation, and wait — instrument which clients still read the old field, and give them a real window (months, for a public API) to migrate. You do not move to the next phase until the old field's read traffic has drained to zero or to an acceptable floor.
3. **Contract**: only after consumers have moved off the old field do you remove it. Now `amount` is gone, but nobody depended on it anymore, so nothing breaks.

The cost of expand-contract is that you carry *both* shapes for a while — extra code, extra fields, the discipline to actually finish the contract phase rather than leaving the old field forever (which is how APIs accumulate cruft). The benefit is that you never break a client and never need a flag-day migration across teams you don't control. This is the same technique the evolutionary-architecture deep-dive applies to database schema changes; it generalizes to *any* contract change, and an API is the contract where it matters most. For the data-layer version of this — keeping two representations consistent during the migration window — the [change data capture and outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) is the mechanism that lets the dual-write stay reliable.

## 12. Optimization: payload, round-trips, and caching with numbers

The series demands an explicit optimization lens, and API design has three clear levers, each with a measurable win. A senior doesn't say "make it faster" — they name the bottleneck, pick the lever, and measure the delta.

**Lever 1 — payload size.** The bottleneck here is bytes on the wire, which matters most for high-throughput internal traffic and metered mobile connections. The §3 worked example quantified it: protobuf vs JSON for a small object is roughly a 4.9× size reduction, turning 70 Mbps of egress into 14 Mbps at 40k QPS. For public JSON APIs where you can't switch to protobuf, the cheaper win is **compression** — gzip or Brotli on responses cuts JSON by 60–80% for repetitive structures, and it's a one-line config change. Measure it as bytes-per-response and total egress; the win is direct bandwidth dollars.

**Lever 2 — round-trips.** The bottleneck is latency × number of sequential calls, which dominates on high-latency links (mobile). Three tools, each measured in round-trips eliminated: **GraphQL or a BFF** collapses a waterfall of N REST calls into 1 (the §10 fan-out), turning N × 80 ms of mobile latency into 1 × 80 ms plus cheap internal fan-out. **HTTP/2 multiplexing** removes connection-setup and head-of-line blocking, so even when you make many calls they share connections — the §3 latency win. **GraphQL query batching** (and dataloaders within a query) collapses the server-side N+1 from §7, the 101-to-2 reduction. Measure it as round-trips per user action and end-to-end p99.

There's a fourth concern that sits alongside these and is genuinely part of the *API contract*, not just the implementation: **rate limiting**. A public API without rate limits is an API one careless client can take down, so the limits — and how you communicate them — are a contract decision. The senior practice is to expose the limit state in response headers (`RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset`) so a well-behaved client can pace itself, and to return `429 Too Many Requests` (or gRPC `RESOURCE_EXHAUSTED`) with a `Retry-After` header when a client exceeds the limit, so the client knows *when* to come back rather than retrying blindly and making the overload worse. For GraphQL the limit must be keyed on query *cost*, not request count, because one query can be a thousand times more expensive than another — the GitHub model from §15. Rate limiting and pagination are two sides of the same coin: pagination bounds how much one *response* can return, and rate limiting bounds how many *requests* one client can make. Together they're how an API protects itself from being loved to death, and both belong in the contract from version one. The full treatment of the server-side mechanics — token buckets, sliding windows, distributed counters, and the backpressure that propagates upstream when limits aren't enough — is the subject of the [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) deep dive; the API-layer point is that the *limits and their signaling* are a contract you design, not a knob you add later.

**Lever 3 — caching.** The bottleneck is recomputing and re-sending data that hasn't changed, and this is where REST's design pays off. **ETags** enable conditional requests: the server sends `ETag: "abc"`, the client sends `If-None-Match: "abc"` next time, and if nothing changed the server returns `304 Not Modified` with an empty body — no serialization, no body bytes, just the headers. For a read-heavy endpoint, this can eliminate the bandwidth and serialization cost of the *majority* of requests. **Cache-Control** lets shared caches and CDNs serve responses without hitting your origin at all, which for public, cacheable reads can offload 90%+ of traffic to the edge. The wins compound with everything in the [caching strategies deep-dive](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite); the API-layer point is that REST's URL-and-verb model makes this caching *free*, which is one of the strongest reasons to reach for it on read-heavy public endpoints — and one of GraphQL's real costs, since `POST`-with-a-body queries forfeit it.

#### Worked example: optimizing a chatty mobile profile screen

Make the levers concrete on the §4 scenario — the mobile profile screen — and watch the numbers move. Baseline: a REST mobile client on a cell connection with **80 ms round-trip latency** rendering a screen that needs the user, ten posts, like counts, and comment authors.

**Baseline (naive REST waterfall):** the client makes `GET /users/42` (80 ms), then `GET /users/42/posts` (80 ms), then per-post like and author calls. Even pipelining some, it's realistically **6 sequential round-trips** ≈ 6 × 80 ms = **480 ms** of network time before the screen can render, plus over-fetching (each REST resource returns every field, of which the screen uses a third). On a flaky connection this is the difference between "snappy" and "why is this app so slow."

**Optimization 1 — collapse round-trips with a BFF or GraphQL:** one request, one round-trip from the client. The fan-out to internal services happens inside the data center at sub-millisecond round-trips, so the client-visible network cost drops from 480 ms to **~80 ms + ~10 ms of internal fan-out ≈ 90 ms** — a **~5.3× latency reduction** on the dominant cost.

**Optimization 2 — kill the server-side N+1 with dataloaders:** the internal fan-out that now fetches 10 posts' authors goes from 1 + 10 = 11 database round-trips to 2 (§7's pattern). Internal round-trips are cheap, but at thousands of concurrent screens this is the difference between a comfortable database and an overloaded one — measured as database QPS, it's a ~5× reduction in query volume for this screen.

**Optimization 3 — shrink the payload and add caching:** the BFF returns only the fields the screen shows (no over-fetching), and tags the response with an `ETag`. On the next refresh, if nothing changed, the `If-None-Match` request gets a `304` — zero body bytes. For a screen users refresh constantly, that's the majority of refreshes served for the cost of a header exchange.

The compounded result: client-visible latency from ~480 ms to ~90 ms (5.3×), database query volume for the screen down ~5×, and most refreshes served from cache at near-zero payload cost. None of these required a faster network or a bigger database — they required designing the API around the actual bottleneck. That is the optimization mindset the series is after.

## 13. Streaming: SSE, WebSockets, and gRPC streams

Not every API is request/response. When the server needs to *push* data — live order status, a notification feed, a collaborative cursor, telemetry — you need streaming, and the protocol choice has a clean decision behind it.

- **Server-Sent Events (SSE)** is the simplest: a long-lived HTTP `GET` where the server streams text events to the client over one connection. It's one-directional (server→client only), runs over plain HTTP so it traverses proxies and works in browsers natively, and auto-reconnects. Reach for it when the client only needs to *receive* a stream — a live status feed, a progress bar, server-pushed notifications. It's underrated precisely because it's boring and works everywhere.
- **WebSockets** upgrade the connection to a full-duplex, bidirectional channel. Reach for it when *both* sides push — chat, collaborative editing, multiplayer, anything interactive. The cost is that it's a different protocol (not plain HTTP after the upgrade), so it bypasses HTTP caching and some proxies, and you manage the connection lifecycle yourself.
- **gRPC streaming** (server-, client-, and bidirectional-streaming) is the internal choice — typed, efficient, multiplexed over HTTP/2. Reach for it for service-to-service streams (telemetry ingestion, event subscriptions inside the mesh), the same way you reach for gRPC unary calls internally.

The senior rule: **default to request/response, and add streaming only when the data is genuinely push-driven.** Streaming connections are stateful — they pin a client to a server (or to a connection a load balancer must keep alive), which complicates the stateless-scaling story from §2 and the load-balancing layer. A long-lived connection is a resource that has to be drained on deploy, rebalanced when a server dies, and accounted for in capacity (each connection holds memory and a file descriptor). Don't reach for streaming because it's cool; reach for it because the data is a stream. And when you do stream events at scale, the durable backbone is usually a log, not a pile of WebSocket connections — the [anatomy of a message system with producers, brokers, and consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) covers the substrate that a streaming API often sits on top of.

## 14. Stress-testing the design: what breaks at 10×?

A design isn't finished until you've tried to break it, so apply the series' stress-test discipline to the three failure modes this article keeps circling. For each, ask: what's the trigger, what breaks, and what's the design response?

**Stress 1 — a breaking schema change ships.** The trigger: a developer renames a field in place to "clean up" the API, or removes a field "nobody uses." What breaks: every client still reading the old field gets `null` or an error; for a public API, that's an outage you can't roll forward out of because you don't control the clients. The design response is everything in §11 — additive evolution as the default, `reserved` field numbers in protobuf, and expand-contract for the rare genuine break. The organizational response is a **contract test** in CI that diffs the schema against the published version and *fails the build* on a breaking change, so a rename-in-place can't even merge. The contract is too expensive to protect with code review alone; you protect it mechanically.

**Stress 2 — an N+1 explosion under load.** The trigger: a client ships a new GraphQL query that nests one level deeper than anyone tested, and at peak traffic it fans out to thousands of per-row database queries. What breaks: the database connection pool exhausts, query latency climbs, and the cascade from there is the classic one — slow queries hold connections, connections run out, every request queues, and a single expensive query shape takes the whole API down. The design response is dataloaders on every resolver (§7) so the fan-out batches, plus **query-cost analysis and depth limiting** that rejects a too-expensive query at the gateway before it touches the database, plus rate-limiting on query *cost* not request count (the GitHub model). The stress test that catches it: a load test that fires the deepest legal query at peak QPS and watches database connection-pool saturation, not just average latency.

**Stress 3 — a chatty mobile client on a degraded network.** The trigger: a mobile client on a 3G connection with 300 ms latency and 1% packet loss hits an API that takes 8 sequential round-trips per screen. What breaks: 8 × 300 ms = 2.4 seconds of pure network time per screen, retries pile up on lost packets, and the app feels broken even though every individual service is healthy. The design response is the round-trip collapse from §10 and §12 — a BFF or GraphQL that turns 8 client round-trips into 1, moving the fan-out inside the data center where a round-trip is sub-millisecond. The stress test: measure p99 *user-action* latency (not endpoint latency) at the worst realistic network profile, because the bottleneck is round-trip count × latency, and that only shows up when you simulate the bad network.

The pattern across all three: the failure isn't in any single component, it's in the *contract and the interaction shape*. That's exactly why API design is architecture — the failure modes are architectural, and the fixes are design decisions made before the load arrives, not knobs you turn during the incident.

## 15. Case studies: three production lessons

Three real-world architectures crystallize the principles, each teaching a different durable lesson.

**Stripe and API versioning.** Stripe runs one of the most-integrated public APIs in the world, and their versioning approach is the gold standard for "never break a consumer." Each Stripe account is pinned to the API version it first integrated against, and Stripe maintains backward compatibility by transforming requests and responses *between* version shapes at the API boundary — an old integration keeps seeing the old shape forever, even as the underlying API evolves underneath it. New accounts get the latest version; existing ones upgrade deliberately when they're ready. The lesson: for a public API, **the consumer's stability is the product.** The engineering cost of maintaining version transformations is real, but it's far cheaper than breaking thousands of paying integrations. This is expand-contract (§11) operationalized at company scale, with the version pin as the mechanism that lets old and new shapes coexist indefinitely.

**GitHub, Shopify, and GraphQL at scale.** Both companies ship large public GraphQL APIs, and both learned the N+1 and complexity-attack lessons (§7) the hard way at a scale where they're not theoretical. GitHub's response is the canonical one: every GraphQL query is assigned a computed **cost** based on the connections it traverses and the nodes it could return, and the rate limit is enforced on accumulated *cost* per hour, not on request count — because one deeply nested query can do the work of a thousand shallow ones, and counting requests would let an expensive query slip through. Shopify similarly enforces query-cost limits and leans hard on batching. The lesson: **GraphQL's flexibility is a liability without cost controls.** The same query language that lets a good client fetch exactly what it needs lets a careless or malicious one ask for the world, and at public scale you *will* meet both. Ship the dataloaders and the cost analysis on day one.

**A gRPC service-mesh rollout.** The third lesson is about internal architecture: organizations that have migrated their internal service-to-service traffic from JSON-over-HTTP to gRPC consistently report the same pattern of wins — meaningful payload reduction (the protobuf-vs-JSON delta from §3), tighter latency tails from HTTP/2 multiplexing eliminating connection-pool contention, and — the underrated one — **fewer integration bugs because the contract is enforced by the compiler.** When the contract is a `.proto` and clients are generated, a breaking change is a build failure across every consumer, caught at compile time rather than discovered in production. The cost they also consistently report: gRPC is harder to debug (no `curl`, binary payloads), needs gRPC-Web and a proxy for any browser-facing edge, and has a steeper learning curve. The lesson: **gRPC's typing-and-codegen discipline pays for itself inside a service mesh, but its costs are exactly why you keep it off the public edge** and translate to REST or GraphQL at the gateway (§5, §10). Match the protocol to the consumer, and the mesh and the public API can each use the one that fits.

## 16. When to reach for each (and when not to)

The decisive recommendations, stated plainly enough to use in a review.

**Reach for REST when** the consumer is a browser, a third party, or any external integrator; when you want free HTTP caching on read-heavy endpoints; when low integration cost and `curl`-ability matter; when the API is the public face of a product. REST is the correct default for public APIs, and "boring" is a feature when third parties have to integrate. **Don't** reach for REST when the client needs many different data shapes from one round-trip (GraphQL fits better) or when it's internal high-throughput traffic where JSON's verbosity and lack of typing cost you (gRPC fits better).

**Reach for gRPC when** both ends are yours, traffic is high-throughput service-to-service, you want strong typing and codegen enforcing contracts across a fleet, you need bidirectional streaming, or payload size and latency tails matter at scale. It's the right backbone for an internal service mesh. **Don't** reach for gRPC for a public browser-facing API (the gRPC-Web proxy tax and the lack of HTTP caching aren't worth it) or for a small system where the codegen and operational overhead exceed the benefit.

**Reach for GraphQL when** you have flexible, fast-moving clients — especially mobile — with many screens and many client teams, where round-trip collapse and client-controlled fetching are the binding constraints, and you're willing to invest in dataloaders, query-cost limits, and application-layer caching. **Don't** reach for GraphQL for a simple CRUD API (REST is less machinery), for high-throughput internal traffic (gRPC is more efficient), or anywhere you can't commit to the N+1 and complexity-attack defenses, because GraphQL without those is a production incident waiting to happen.

And the patterns are not optional regardless of protocol: **cursor pagination** (offset breaks at scale, §8), **idempotency keys** on every retryable mutation (§6), **stable machine-parseable error codes** (§9), and **additive, expand-contract evolution** (§11) belong in *every* API contract you design, REST or gRPC or GraphQL alike. The protocol is the easy choice; the durable patterns are what make the contract outlive the implementation.

## Key takeaways

- **The contract is the architecture.** The implementation is soft clay; the API is concrete the instant a second party depends on it. Design the contract for evolution, because reversing it sits at the 1000× end of the cost-of-change curve.
- **Match the protocol to the consumer, not to fashion.** Public/browser → REST; internal high-throughput → gRPC; chatty multi-screen client → GraphQL. The decision is a function of who calls the API, and large systems run all three behind a gateway.
- **Offset pagination is a trap.** It slows linearly with depth (450× at a million rows in the worked numbers) and silently drops and duplicates rows under concurrent writes. Use opaque cursor (keyset) pagination from version one.
- **Every retryable mutation needs an idempotency key.** The network loses responses after the work is done; without idempotency a client cannot safely retry, and at scale that means double-charges. Fingerprint the body and store the result with the side effect in one transaction.
- **GraphQL moves the hard problems to the server.** Ship dataloaders (101 round-trips to 2) and query-cost limits on day one, or the N+1 explosion and complexity attack will find you in production, as they found GitHub and Shopify.
- **Errors are part of the contract.** Use stable, documented, machine-parseable codes (problem+json, gRPC status, GraphQL errors) that map to retry semantics, carry a correlation ID, and never leak internals.
- **Never break a published contract.** Evolve additively (protobuf field-number rules; clients that ignore unknown fields), and use expand-contract for the rare genuine break — add, dual-write, then remove only after consumers have moved.
- **Optimize the right lever, and measure it.** Payload (protobuf 4.9×, compression 60–80%), round-trips (BFF/GraphQL N→1, HTTP/2 multiplexing), and caching (ETags → 304, Cache-Control → edge offload) each have a distinct bottleneck and a measurable win.

## Further reading

- [Evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) — the cost-of-change curve and expand-contract pattern that this article applies to API contracts.
- [Idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) — the deep dive on making mutations safe to retry, beyond the key-based sketch here.
- [Rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) — the contract between server and client for shedding load, the partner to error modeling's `429`/`RESOURCE_EXHAUSTED`.
- [Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) — the routing layer that stateless REST unlocks and that streaming connections complicate.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — where ETags and Cache-Control fit in the broader caching picture.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the mechanism that keeps a dual-write reliable during an expand-contract migration window.
- [Anatomy of a message system: producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) — the streaming substrate a push-based API often sits on top of.
- [Transactional outbox pattern: reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — how to make a mutation's side effect and its record commit atomically.
- Official references worth bookmarking: the gRPC documentation and protobuf language guide (field-evolution rules), the GraphQL specification and the DataLoader project, RFC 9457 (problem+json), and Stripe's public API reference as a model of versioning and idempotency done right.

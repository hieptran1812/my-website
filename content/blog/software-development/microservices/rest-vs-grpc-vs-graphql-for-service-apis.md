---
title: "REST vs gRPC vs GraphQL for Service APIs: A Decision Guide"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A junior-friendly, senior-deep guide to choosing the synchronous API style between your services: what REST, gRPC, and GraphQL actually are, the decision framework, the real payload and latency numbers, the N+1 trap and its fix, and why the production answer is usually gRPC inside and REST or GraphQL at the edge."
tags:
  [
    "microservices",
    "rest",
    "grpc",
    "graphql",
    "api-design",
    "distributed-systems",
    "software-architecture",
    "backend",
    "protobuf",
    "http2",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-1.webp"
---

The ShopFast team had shipped four services and a mobile app, and the question that stalled the design review was not *what* to build but *how the pieces should talk*. The order service needed to ask inventory "do we have 3 of SKU-771 in the Berlin warehouse?" and ask payment "charge this card for €58.40." The mobile app needed to render an order-details screen — the order, its line items, and the shipping status — in one fast paint. Someone said "just use REST, it's what we know." Someone else said "gRPC is faster, Google uses it for everything." A third person had read a GraphQL tutorial over the weekend and wanted one endpoint that returns exactly the fields the app asks for. All three were partly right, and that is exactly why the room could not converge: they were comparing three tools as if one had to win every contest, when in reality each tool is the best answer to a *different question*.

This is the post that ends that argument with a framework instead of a preference. By the end you will be able to walk into that review and say, with numbers behind you, something like: "Order-to-inventory and order-to-payment are internal east-west calls in a tight latency budget, so gRPC — binary Protobuf over HTTP/2, contract-first, codegen for both Go and Java. The mobile app is a varied north-south consumer that over-fetches badly on REST, so we put a GraphQL layer at the edge. The public partner API stays plain REST because partners want to `curl` it and cache it. Here is what each choice costs us." That sentence is the destination. Figure 1 is the map.

![A topology diagram showing the ShopFast mobile app and browser talking to a gateway over GraphQL and REST while the gateway and internal services talk to each other over gRPC](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-1.webp)

We will not re-derive *why* a network call is dangerous — the previous post, [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), covered the fallacies, the availability arithmetic, and the reflex that every remote call can be slow and can be down. This post assumes you carry that reflex and asks the next question: *given that you have decided to make a synchronous request-response call between two services, which wire protocol and contract style should it use, and what does the wrong choice cost you in latency, dollars, debuggability, and 3am pages?* We will build each contender from first principles, run the same "get an order with its items and shipping" call through all three, put real numbers on the wire, walk into the two traps that bite teams hardest (REST over-fetching and the GraphQL N+1), and finish with the pattern almost every mature fleet converges on.

## The one question that splits the whole field: north-south or east-west?

Before any tool comparison, install one axis, because it predicts the answer most of the time. Traffic in a microservices system flows in two directions, and the industry borrowed networking words for them. **North-south** traffic crosses the boundary of your system: a mobile app, a single-page web app in a browser, or a third-party partner calling in from the public internet. **East-west** traffic stays inside your system: your order service calling your inventory service, both running in the same Kubernetes cluster, often in the same availability zone, behind the same firewall, owned by teams that sit near each other.

These two kinds of traffic have almost opposite constraints, and that is the single most clarifying fact in this entire post. North-south clients are *heterogeneous and untrusted*: you do not control the browser, you cannot force a partner to regenerate a client when you change a field, the network is the public internet with real latency and real packet loss, and you must assume the caller is hostile until proven otherwise. East-west clients are *homogeneous and trusted*: you wrote both ends, you can regenerate both clients in the same CI run when the contract changes, the network is a fast datacenter LAN, and the caller is your own code. A protocol optimized for one is frequently wrong for the other.

REST and GraphQL are *edge-shaped*: they ride plain HTTP, work in any browser, are human-readable, and tolerate loosely-coupled, slowly-evolving clients. gRPC is *internal-shaped*: it is binary, contract-first, fast, supports streaming, and assumes the caller is a generated client you control — but it does not run natively in a browser and is harder to poke at with `curl`. So when ShopFast's mobile app talks to the gateway, that is north-south, and REST or GraphQL fits. When the order service talks to inventory, that is east-west, and gRPC fits. Hold that axis in your head; everything below either confirms it or names the exceptions.

## What REST actually is (and what it is not)

REST gets used as a synonym for "JSON over HTTP," which is roughly true in practice and useful enough to run with, but it is worth being precise because the precision explains REST's real strengths. REST — Representational State Transfer, Roy Fielding's 2000 dissertation — is an *architectural style*, not a protocol. Its core idea is that you model your domain as **resources** identified by URLs, and you act on them with a small fixed set of HTTP verbs: `GET` to read, `POST` to create, `PUT`/`PATCH` to update, `DELETE` to remove. The server sends back a *representation* of the resource, almost always JSON. The interaction is *stateless* — every request carries everything the server needs, so any server instance can handle any request, which is what lets you scale REST horizontally by just adding boxes behind a load balancer.

It is worth pausing on the verbs, because juniors often treat them as decoration when they are actually a contract about *safety* and *idempotency* that the whole HTTP ecosystem relies on. `GET` is **safe** (it must not change state) and **idempotent** (calling it ten times is the same as calling it once), which is *why* it is cacheable and *why* a proxy or browser is allowed to retry it freely. `PUT` and `DELETE` are idempotent but not safe — putting the same resource twice lands in the same final state. `POST` is neither, which is why browsers warn before re-submitting a form. This is not pedantry: when your retry logic, your CDN, and your load balancer all make decisions based on the verb's declared semantics, choosing the wrong verb (a `GET` that mutates, a `POST` that should have been idempotent) is how you end up with double-charged customers and uncacheable reads. The verb is a promise, and the infrastructure trusts it.

That design buys REST its three genuine superpowers. First, **ubiquity**: every language, every framework, every browser, every proxy, every CDN, every API tool on earth speaks HTTP and JSON. You can call a REST endpoint from a shell with `curl`, read the response with your eyes, and cache it in a CDN you did not write — no client library, no codegen, no toolchain, nothing to install. That zero-friction reach is the single reason REST became the lingua franca of public APIs, and it is genuinely hard to overstate how much it matters when the consumer is someone you will never meet. Second, **HTTP caching for free**: because a `GET` is a safe, idempotent read of a resource at a URL, the entire HTTP caching machinery — `ETag`, `Cache-Control`, `If-None-Match`, 304 Not Modified, edge caches — works without you building anything. A correctly-tagged `GET /products/771` can be served from a CDN edge in single-digit milliseconds and never touch your service, and when the resource has not changed the server answers `304 Not Modified` with an empty body, saving even the bytes. Third, **debuggability**: the request and response are text a human can read, so when something breaks at 3am you can replay the exact call from a terminal with `curl -v` and see exactly what came back, headers and all — no binary decoder, no special tooling, just your eyes and a shell. That property is an *availability* feature, because the faster a half-asleep on-call engineer can see the actual bytes, the faster the incident ends.

Here is a small, idiomatic REST endpoint for ShopFast's order resource. Notice the verbs map to operations, the URL identifies the resource, and the response is a plain JSON representation:

```python
# order_service/rest_api.py  (FastAPI)
from fastapi import FastAPI, HTTPException, Response

app = FastAPI()

@app.get("/orders/{order_id}")
def get_order(order_id: str, response: Response):
    order = repo.find_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="order not found")
    # ETag enables HTTP caching: clients send If-None-Match and we
    # answer 304 when nothing changed, skipping the body entirely.
    response.headers["ETag"] = order.version_tag()
    response.headers["Cache-Control"] = "private, max-age=15"
    return {
        "id": order.id,
        "status": order.status,
        "total_cents": order.total_cents,
        "currency": order.currency,
        # ... and ~35 more fields the caller may or may not need
    }
```

Now the costs, because REST is not free. The first is **over-fetching and under-fetching**, the disease that GraphQL exists to cure. The order endpoint returns the *whole* order — every field — because the server cannot know which fields a given caller wants. The mobile order-summary screen needs three fields (`id`, `status`, `total_cents`) and gets forty. That is over-fetching: wasted bytes, wasted serialization, wasted bandwidth, multiplied across every request. The mirror problem is under-fetching: the screen also needs the line items and the shipping status, which live at *other* URLs, so the client makes three sequential calls (`GET /orders/42`, `GET /orders/42/items`, `GET /shipping/42`) — three round trips where one would do. The second cost is **no schema by default**. Plain REST has no machine-readable contract unless you bolt one on with OpenAPI/Swagger, so a client and server can silently disagree about a field's type until a runtime error proves it. The third is **chattiness for relational data**: anything graph-shaped — orders with items with products with reviews — tends to fan out into many round trips, the N+1 problem reappearing at the API layer.

## What gRPC actually is (and why Google's internal traffic runs on it)

gRPC is a contract-first, binary RPC framework that Google open-sourced in 2015, derived from their internal "Stubby" system that had been carrying essentially all of Google's east-west traffic for over a decade. The "g" is officially recursive and changes meaning per release; treat it as "Google RPC." Three design choices define it, and each one is a direct answer to a REST cost.

The first choice is **contract-first with Protocol Buffers**. You do not start by writing handler code; you start by writing a `.proto` file that declares your service, its methods, and the exact shape of every message. That file *is* the contract, and it is the source of truth for both ends. Here is ShopFast's inventory and order contract:

```protobuf
// shopfast/order/v1/order.proto
syntax = "proto3";
package shopfast.order.v1;

message GetOrderRequest {
  string order_id = 1;
}

message Money {
  string currency = 1;   // ISO-4217, e.g. "EUR"
  int64 amount_cents = 2;
}

message OrderItem {
  string sku = 1;
  int32 quantity = 2;
  Money unit_price = 3;
}

message Order {
  string id = 1;
  string status = 2;
  Money total = 3;
  repeated OrderItem items = 4;
  string shipping_status = 5;
}

service OrderService {
  // Unary: one request, one response.
  rpc GetOrder(GetOrderRequest) returns (Order);
  // Server streaming: subscribe to status changes for one order.
  rpc WatchOrderStatus(GetOrderRequest) returns (stream Order);
}
```

The numbered field tags (`= 1`, `= 2`) are the heart of Protobuf's wire format and its schema-evolution story. On the wire, Protobuf does not send field *names* — it sends the *tag number* plus a type, then the value. That is why the payload is small (no repeated `"shipping_status":` text in every message) and why evolution is safe: as long as you never reuse a tag number, you can add field 6 next year and old clients simply ignore the tag they do not recognize, while new clients reading old data get the field's default. This is the disciplined cousin of the schema-evolution problem that [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) treats in depth; Protobuf bakes a good chunk of the discipline into the format itself.

The second choice is **HTTP/2 as the transport**. This is where most of gRPC's speed comes from, and it is worth understanding because it explains why "gRPC is fast" is true for a specific, mechanical reason rather than as a vibe. Figure 3 lays out the layers gRPC stacks on top of each other.

![A layered stack diagram showing a gRPC method call sitting on Protobuf binary encoding, riding multiplexed HTTP/2 streams with HPACK header compression over a single reused TLS connection on a raw TCP socket](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-3.webp)

Under HTTP/1.1, a single TCP connection can carry one request at a time; if you want to send a second request while the first is in flight, you open a second connection, and browsers and clients keep pools of them. Worse, HTTP/1.1 has *head-of-line blocking*: response two cannot start until response one finishes on that connection. HTTP/2 fixes both. It **multiplexes** many concurrent streams over a *single* TCP connection, each stream an independent request-response that can interleave with the others, so one connection between order and inventory carries thousands of concurrent calls without head-of-line blocking at the HTTP layer. It also **compresses headers** with HPACK, so the repetitive metadata (the same `:authority`, `content-type`, auth token on every call) is not re-sent in full each time. The practical effect east-west: where an HTTP/1.1 client juggles a pool of connections and pays a TLS handshake when the pool grows, a gRPC client reuses *one* warm connection and pipes everything through it, which removes connection-setup cost from the hot path entirely.

The third choice is **code generation**. The `.proto` compiles, via `protoc` and language plugins, into typed client stubs and server interfaces in Go, Java, Python, C++, Rust, TypeScript, and more. You never hand-write serialization or HTTP plumbing; you call a typed method and get a typed result. Here is the generated client in use from ShopFast's gateway, calling the order service:

```go
// gateway/order_client.go
import (
    "context"
    "time"
    orderv1 "shopfast/gen/order/v1"
    "google.golang.org/grpc"
)

func fetchOrder(conn *grpc.ClientConn, orderID string) (*orderv1.Order, error) {
    client := orderv1.NewOrderServiceClient(conn)

    // ALWAYS pass a deadline. gRPC propagates it across the call chain,
    // so a 200ms budget here becomes the budget the order service sees.
    ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
    defer cancel()

    order, err := client.GetOrder(ctx, &orderv1.GetOrderRequest{OrderId: orderID})
    if err != nil {
        // err carries a gRPC status code: Unavailable, DeadlineExceeded, NotFound...
        return nil, err
    }
    return order, nil
}
```

Two things in that snippet are not incidental. The `ctx` with a deadline is the seam-showing design the previous post praised: gRPC *forces* you to think about timeouts because the idiom puts a context in your face, and it *propagates* that deadline down the chain so a 200ms budget at the gateway is enforced at every hop. If the gateway sets a 200ms deadline and has already spent 120ms when it calls the order service, the order service sees an 80ms budget, and the inventory service it calls sees whatever is left after that — the whole call tree shares one shrinking clock, and any hop that blows the budget gets a clean `DeadlineExceeded` rather than hanging a thread forever. That single feature prevents an entire class of cascading-timeout incidents that plain REST clients have to reinvent by hand.

The streaming method (`WatchOrderStatus(...) returns (stream Order)`) is the other thing REST cannot do cleanly — a real, typed, server-streaming (and bidirectional-capable) channel over the same connection, which is why gRPC is the natural choice for live order tracking, a telemetry firehose, or a chat-like flow. Consuming it is as simple as ranging over a stream; each `Recv()` yields the next typed `Order` until the server closes it:

```go
// gateway/watch_order.go — consume a server-streaming RPC.
func watchOrder(client orderv1.OrderServiceClient, orderID string) error {
    ctx := context.Background()  // no per-call deadline: this stream is long-lived
    stream, err := client.WatchOrderStatus(ctx, &orderv1.GetOrderRequest{OrderId: orderID})
    if err != nil {
        return err
    }
    for {
        order, err := stream.Recv()
        if err == io.EOF {
            return nil // server closed the stream cleanly
        }
        if err != nil {
            return err // transport error or status code
        }
        pushToClient(order.Status, order.ShippingStatus) // each update, as it happens
    }
}
```

To do the same thing with REST you would reach for Server-Sent Events or long-polling — both workable, both bolted on, neither typed, and neither sharing the connection multiplexing that lets gRPC run a hundred such streams plus a thousand unary calls over the *same* socket. That is the difference between streaming being a first-class verb in your contract and streaming being a pattern you assemble out of HTTP primitives.

### HTTP/1.1 versus HTTP/2: the mechanical reason gRPC is fast

It is worth dwelling on the transport because "gRPC is fast" confuses juniors who assume the binary payload alone explains it. A large share of the win is the *connection model*, and you can get part of it for REST too once you understand the mechanism. Take a single client that needs to issue six concurrent calls to one backend. Under **HTTP/1.1**, a connection carries exactly one request-response at a time, and responses must come back in request order on that connection (head-of-line blocking). So the client opens a *pool* of connections — browsers cap this at around six per host — and spreads the six calls across six sockets. Each new socket pays a TCP handshake (one round trip) and, if encrypted, a TLS handshake (one or two more round trips) before a single byte of your request flows. On a 1ms LAN that connection setup is a couple of milliseconds; across regions at 80ms RTT it is *hundreds* of milliseconds of pure setup before work begins, and the pool churns as connections idle out and get re-established.

Under **HTTP/2**, those same six calls travel as six independent *streams* multiplexed over a **single** TCP connection. There is no head-of-line blocking at the HTTP layer — stream three's response can arrive before stream one's — and there is exactly one TCP handshake and one TLS handshake, paid once when the connection is first established and then amortized across every subsequent call for the life of the connection. A gRPC client keeps that one connection warm and pipes thousands of calls through it; the connection-setup cost simply leaves the hot path. HTTP/2 also compresses headers with HPACK, so the repetitive metadata that REST re-sends in full on every request — the same `:authority`, `content-type`, `user-agent`, and bearer token — is sent once and then referenced by index, which matters more than people expect when payloads are small and headers are a large fraction of the bytes.

The practical upshot, and the lever to reach for first when optimizing east-west traffic, is **connection reuse**. The single biggest latency saving on a hot internal path is usually not switching JSON to Protobuf; it is stopping the per-call TCP+TLS handshake by keeping a warm multiplexed connection. You can capture most of that win for REST too by running it over HTTP/2 (modern frameworks and load balancers support it) and ensuring your HTTP client reuses connections (keep-alive on, a sane connection pool, no accidental client-per-request). gRPC simply makes HTTP/2 and connection reuse the *default and only* path, which is part of why it is fast without anyone having to remember to configure it. The flip side — that HTTP/2's single connection can hit a per-connection concurrent-stream cap under extreme load — is a real tuning concern we will return to in the optimization section.

So what does gRPC cost? Three real things. It is **not browser-native**: a browser cannot open the raw HTTP/2 frames gRPC needs, so you cannot call a gRPC service directly from JavaScript in a page — you need a gRPC-Web shim plus a translating proxy. It is **harder to debug**: the payload is binary, so you cannot just read it; you need tooling (`grpcurl`, reflection, or a Protobuf-aware proxy) to inspect a call. And it has a **higher learning curve and toolchain weight**: `protoc`, the plugins, the build wiring, the generated-code-in-your-repo question — all real friction a small team feels. Those costs are exactly why gRPC is glorious east-west and awkward north-south.

## What GraphQL actually is (and the problem it was born to solve)

GraphQL came out of Facebook in 2012 and was open-sourced in 2015, and it was born from a very specific pain: a mobile app over a slow phone network making a dozen REST round trips to render one screen, each one over-fetching fields the screen ignored. GraphQL's core move is to **invert who decides the response shape**. In REST the *server* decides what a `GET /orders/42` returns. In GraphQL the *client* decides, by sending a query that describes precisely the fields and nested relationships it wants, and the server returns exactly that — no more, no less — from a *single* endpoint, usually `POST /graphql`.

The mechanics are worth making concrete for a junior, because the word "resolver" hides a simple idea. You define two things. The first is a **schema**: a strongly-typed declaration of your domain as a graph of types and the fields each type has — `Order` has a `total` and a list of `items`, an `OrderItem` has a `product`, a `Product` has a `name`. The second is a set of **resolvers**: one small function per field that knows how to *fetch that field's value*. When a query arrives, the GraphQL engine walks the query tree the client sent and calls the matching resolver for each requested field, top down — resolve the `order`, then for that order resolve its `items`, then for each item resolve its `product`, then for each product resolve `name`. The engine assembles the returned values into the exact JSON shape the query described. Crucially, a resolver can fetch from *anywhere* — a database, a Redis cache, a downstream gRPC service, a third-party API — which is precisely what makes GraphQL a natural aggregation layer: each field can come from a different backend, and the client never knows or cares. The flip side, which the N+1 example will make painfully clear, is that "one resolver per field, called once per parent object" is also exactly how a single innocent-looking query fans out into a storm of backend calls. Here is ShopFast's GraphQL schema for the order-details screen:

```graphql
# gateway/schema.graphql
type Money { currency: String!  amountCents: Int! }

type Product { sku: ID!  name: String!  imageUrl: String }

type OrderItem { product: Product!  quantity: Int!  unitPrice: Money! }

type Shipping { status: String!  carrier: String  eta: String }

type Order {
  id: ID!
  status: String!
  total: Money!
  items: [OrderItem!]!     # nested relationship, resolved on demand
  shipping: Shipping!      # nested relationship, resolved on demand
}

type Query {
  order(id: ID!): Order
}
```

And here is the *same* "get an order with its items and shipping" request that took three REST round trips, now as one GraphQL query that asks for exactly the fields the mobile screen paints:

```graphql
query OrderScreen {
  order(id: "42") {
    id
    status
    total { amountCents currency }
    items {
      quantity
      product { name imageUrl }   # only name + image, not the whole product
    }
    shipping { status eta }
  }
}
```

One request, one round trip, exactly the fields the screen needs, with nested relationships resolved server-side. Figure 4 contrasts the two.

![A before and after comparison contrasting three over-fetching REST calls for an order screen against a single GraphQL query that returns exactly the requested fields in one round trip](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-4.webp)

This is genuinely transformative for one job: being the **aggregation layer** in front of many backends — the Backend-for-Frontend, the edge API that a varied set of clients hits. A mobile app, a web app, and a smart-TV app each want different slices of the same data; with REST you either build three custom endpoints or make all three over-fetch, but with GraphQL each client asks for its own slice from one schema. GraphQL is so good at this that it has become the default for public, client-facing APIs at companies whose product *is* a varied client experience. The gateway role here connects directly to the next post in this series, [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), where GraphQL is one of the headline BFF technologies.

GraphQL's costs are real and they are sharp. The headline one is the **N+1 resolver trap**, which we will turn into a worked example below — naive resolvers can quietly turn one query into dozens of database calls. The second is **caching is harder**: because everything is a `POST` to one URL with a query in the body, the free HTTP `GET`-caching machinery that REST enjoys does not apply. A CDN cannot key a cache on "this URL" because every client sends a different query body to the same `/graphql` URL, so you lose edge caching entirely and have to rebuild it yourself at the field or entity level — caching a resolved `Product` by SKU in Redis, for instance, rather than caching whole responses. That is more code, more cache-invalidation reasoning, and more ways to serve stale data than REST's nearly-free model. The third is **complexity and a foot-gun surface**: the server now executes arbitrary client-shaped queries, which means a careless or malicious client can ask for a pathologically deep or wide query that hammers your backends — a security and stability concern we will stress-test later. There is also a real *operational* cost: error handling is fiddly (GraphQL returns `200 OK` with an `errors` array even on partial failure, so naive monitoring that watches HTTP status codes goes blind), and field-level metrics require instrumenting resolvers rather than reading access logs. GraphQL trades server-side simplicity for client-side flexibility, and you pay that trade in resolver discipline, caching machinery, and query governance.

## The same call, three ways: a side-by-side reckoning

Let us make the comparison concrete by holding the *task* fixed — "get order 42 with its items and shipping status" — and varying only the API style, because that is the apples-to-apples view the ShopFast review needed. This is the discipline that cuts through tool-advocacy arguments: pin the exact operation, then describe how each style serves it, including the parts that are awkward. A protocol comparison done on abstract feature lists always favors whatever the author likes; a comparison done on one concrete call exposes the real ergonomics.

With **REST**, the mobile app issues three requests. `GET /orders/42` returns the order (forty fields, of which it keeps three). `GET /orders/42/items` returns the line items (each item a full product object, of which it keeps name and image). `GET /shipping/42` returns the shipping record. Three sequential round trips because item two depends on knowing the order exists; on a 60ms mobile RTT that is ~180ms of network time alone, and roughly 70% of the bytes returned are discarded. The upside: each call is cacheable, readable, and trivially debuggable.

With **gRPC**, you would not typically expose this to the mobile app directly (browser problem), but east-west the gateway calls `OrderService.GetOrder(order_id: "42")` and gets back a single `Order` message with items and shipping nested, in one round trip, as ~560 bytes of binary Protobuf over a warm HTTP/2 connection. If the order service itself needs inventory and shipping data from other services, *those* are gRPC calls too, and the deadline propagates through all of them. The upside: smallest payload, lowest latency, typed end to end. The downside: opaque on the wire, and no browser can make that call.

With **GraphQL**, the mobile app sends the single `OrderScreen` query above to `POST /graphql`. The gateway's resolvers fan out — possibly to the same gRPC services internally — and assemble exactly the requested shape into one JSON response, one round trip. The upside: exact fields, one trip, one schema for all clients. The downside: that fan-out is where N+1 lurks, and the response is not HTTP-cacheable for free.

The decision matrix in Figure 2 stacks the three across the properties that actually drive the choice. Read it as "no row is a tie-breaker by itself; the consumer and access shape decide which rows matter."

![A comparison matrix scoring REST, gRPC, and GraphQL across browser support, schema strength, streaming, payload size, HTTP caching, debuggability, and learning curve](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-2.webp)

Here is the same matrix as a table you can paste into a design doc, with the *why* spelled out:

| Property | REST / JSON | gRPC / Protobuf | GraphQL |
|---|---|---|---|
| Browser-native | Yes — any `fetch` works | No — needs gRPC-Web + proxy | Yes — `POST` from any client |
| Schema / contract | Optional (OpenAPI bolt-on) | Mandatory `.proto`, codegen | Mandatory typed schema |
| Wire format | Text JSON, verbose | Binary Protobuf, compact | Text JSON, verbose |
| Streaming | SSE / long-poll only | First-class uni + bidi streams | Subscriptions (extra setup) |
| Over/under-fetching | Both, badly | Fixed per method | Solved — client picks fields |
| HTTP caching | Free on `GET` (ETag/CDN) | None (custom) | Hard (`POST`, build it) |
| Debuggability | `curl`, readable | Binary, needs tooling | Explorer/introspection, body-only |
| Best traffic | North-south, public, cacheable | East-west, internal, low-latency | North-south aggregation / BFF |
| Main risk | Chatty, no contract | Not browser-native, opaque | N+1 resolvers, query abuse |

## Performance reality: putting numbers on the wire

"gRPC is faster" and "GraphQL solves over-fetching" are claims that deserve measurement, not faith, so this section pins them down. There are three places a synchronous API spends time and money, and a senior reasons about all three rather than fixating on whichever one a blog post happened to mention. The first is **payload size** — how many bytes cross the wire per call, which drives bandwidth, egress cost, and the time-on-wire portion of latency. The second is **serialization cost** — the CPU spent turning your in-memory objects into bytes and back, which is invisible until you are at scale and then becomes a top consumer of cores. The third is **transport overhead** — connection setup, head-of-line blocking, header bloat, and the round trips you pay before any real work happens. The worked examples below put concrete numbers on the first two and the HTTP/2 section above covered the third; together they explain the latency and cost deltas you will actually observe.

A word of honesty before the numbers: none of these wins is free or automatic. Protobuf's payload and CPU advantages assume you generate and reuse the codecs; GraphQL's over-fetching win assumes your resolvers are batched; REST's caching win assumes you set the headers correctly. The numbers below are what you get when the relevant discipline is in place, and the failure modes (the N+1, the cold connection, the unguarded query) are what you get when it is not.

#### Worked example: the order payload, JSON versus Protobuf, at 10k RPS

Take ShopFast's `Order` for order 42: id, status, a `Money` total, five `OrderItem`s (each with a SKU string, an int quantity, and a `Money` unit price), and a shipping status. Let us size it on the wire and price the serialization, because "Protobuf is smaller and faster" deserves numbers.

As pretty-printed JSON with descriptive field names — `"shipping_status"`, `"amount_cents"`, `"currency"` repeated in every nested object — this message lands at roughly **1,180 bytes**. Minified (no whitespace) it drops to about 720 bytes, but even minified, every one of those field-name strings is sent on every message, and the numbers are encoded as decimal text. As Protobuf, the same data is about **560 bytes**: field names become 1-byte tags, the integers use varint encoding, and there is no structural punctuation. That is roughly a **2.1× reduction** versus pretty JSON and about 1.3× versus minified. Turn on gzip and both shrink, but Protobuf still wins because there is simply less redundant text to compress, and Protobuf decode does not pay the cost of parsing and allocating strings for field names.

Now the CPU. Serializing this object to JSON in a typical server runtime costs on the order of **~3 microseconds**; Protobuf serialization of the same object costs roughly **~0.9 microseconds** — call it 3× cheaper, because Protobuf writes bytes directly while JSON formats numbers as text and escapes strings. Three microseconds sounds like nothing until you multiply by load. At **10,000 requests per second**, JSON serialization burns ~30 milliseconds of CPU *per second per core's worth of work* just on the encode side — about 3% of one core continuously, before you count decode on the receiver, before garbage collection of all those intermediate strings. Protobuf cuts that to ~9ms/s. Across a fleet doing millions of internal RPCs per second — which is exactly Google's situation, and the reason Stubby/gRPC exists — that delta is *racks of servers* and a real fraction of the latency budget.

Putting payload and CPU together, the measured end-to-end picture for this call at 10k RPS on a warm path looks like the numbers in Figure 7: JSON-over-HTTP/1.1 around **34ms p99**, gRPC-Protobuf-over-HTTP/2 around **19ms p99**, with the gap coming from smaller payloads, cheaper (de)serialization, no per-request connection setup, and no HTTP-layer head-of-line blocking. The numbers are order-of-magnitude illustrative — your hardware, payload, and network will shift them — but the *shape* is robust and reproducible: binary-over-multiplexed beats text-over-pooled for high-volume east-west traffic.

![A measured comparison matrix showing the same order payload as REST JSON versus gRPC Protobuf across payload bytes, serialization CPU, p99 latency at ten thousand requests per second, connections per host, and wire format](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-7.webp)

This is the optimization angle made concrete, and it is worth stating the lever order plainly, because juniors often reach for the wrong one first. The biggest east-west win is usually **connection reuse** — one warm HTTP/2 connection instead of a fresh TLS handshake per call removes 1–2 RTTs and a CPU-heavy handshake from the hot path. The second is **payload size** — Protobuf over JSON, plus only sending fields you need. The third is **compression** — gzip or Brotli on the wire, which helps JSON more than Protobuf precisely because JSON has more redundancy to squeeze. Measure each independently; do not assume the binary-format win alone explains a latency drop when half of it was really the connection you stopped re-establishing.

#### Worked example: the GraphQL N+1 that turned one query into 51 database hits

This is the trap that bites every team the first month they ship GraphQL, and it is worth walking through slowly because the fix is small but the failure is severe. ShopFast's "recent orders" screen runs this query:

```graphql
query RecentOrders {
  recentOrders(limit: 50) {   # returns 50 orders
    id
    total { amountCents }
    items {
      product { name }         # the resolver that bites us
    }
  }
}
```

The naive resolver wiring looks innocent. The `recentOrders` resolver runs **one** query: `SELECT * FROM orders ORDER BY created_at DESC LIMIT 50`. Then GraphQL resolves the `items` field for each of the 50 orders, and for each item it resolves the `product` field by calling `getProduct(sku)`, which runs `SELECT * FROM products WHERE sku = ?`. If each order has a few items pointing at, say, 50 distinct products across the page, you get **1 query for the orders plus 50 product queries = 51 database round trips** for one screen. Under load this is catastrophic: at 100 concurrent screen loads that is 5,100 product queries hitting the database, p99 climbs to ~480ms, and the database connection pool saturates. The query *looked* like one request; it became 51.

```typescript
// gateway/resolvers.ts  — THE NAIVE VERSION (N+1)
const resolvers = {
  Query: {
    recentOrders: (_, { limit }) => db.query(
      "SELECT * FROM orders ORDER BY created_at DESC LIMIT $1", [limit]),
  },
  OrderItem: {
    // Called once PER item. 50 items -> 50 separate product queries.
    product: (item) => db.query(
      "SELECT * FROM products WHERE sku = $1", [item.sku]),
  },
};
```

The fix is a **DataLoader** — a per-request batching-and-caching utility (the pattern is from Facebook's `dataloader` library, ported to every language). Instead of each `product` resolver firing its own query immediately, it hands its key to a loader. The loader *collects* all the keys requested within one tick of the event loop, then fires **one** batched query — `SELECT * FROM products WHERE sku = ANY($1)` — and distributes the rows back to the waiting resolvers. The 50 individual lookups collapse into **one** batched query; combined with the orders query that is **2 database hits** for the whole screen, and p99 drops from ~480ms to ~22ms. The loader also de-duplicates: if three items reference the same product, it is fetched once.

```typescript
// gateway/loaders.ts  — THE FIX (batched + cached per request)
import DataLoader from "dataloader";

export const makeProductLoader = () => new DataLoader(async (skus: string[]) => {
  // ONE query for ALL skus requested this tick.
  const rows = await db.query(
    "SELECT * FROM products WHERE sku = ANY($1)", [skus]);
  const bySku = new Map(rows.map((r) => [r.sku, r]));
  // Return results in the SAME ORDER as the input keys (DataLoader contract).
  return skus.map((sku) => bySku.get(sku) ?? null);
});

// resolvers.ts — product resolver now just defers to the loader.
const resolvers = {
  OrderItem: {
    product: (item, _args, ctx) => ctx.productLoader.load(item.sku),
  },
};
```

Figure 5 contrasts the two execution profiles. The lesson generalizes beyond GraphQL: any time a per-item resolver or per-item loop issues its own remote call, you have reinvented the N+1, whether the backend is a SQL database or a downstream gRPC service. The reflex is *batch at the boundary* — and notice that gRPC's `repeated` fields and batch RPCs are the contract-first version of the same idea, which is why an internal `BatchGetProducts(skus)` RPC is often the cleanest fix when the resolver fans out to a service rather than a table.

![A before and after comparison showing a naive GraphQL resolver firing fifty-one database queries for one screen versus a DataLoader batching them into two queries](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-5.webp)

#### Worked example: the fleet-scale dollar difference between JSON and Protobuf east-west

The per-call numbers feel academic until you scale them to a fleet, so let us do the back-of-the-envelope arithmetic that turns a protocol choice into a line item on the cloud bill — the kind of estimation [back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) teaches. Suppose ShopFast's internal fabric does **2 million internal RPCs per second** at peak across all services (a mid-size estate; large ones do far more), each carrying a payload comparable to our order message. We will compare an all-JSON-over-HTTP/1.1 internal fabric against an all-Protobuf-over-HTTP/2 one.

Start with **bandwidth**. At 1,180 bytes per JSON message versus 560 bytes per Protobuf message, the per-message saving is ~620 bytes. Across 2M RPS that is 2,000,000 × 620 bytes ≈ **1.24 GB/s** of traffic that simply does not exist on the Protobuf fabric. Much of that is intra-AZ and effectively free, but the slice that crosses availability-zone or region boundaries is billed at cloud egress rates (cross-AZ egress is commonly around \$0.01/GB, cross-region higher). If even 10% of those RPCs cross a billed boundary, that is ~124 MB/s of billed traffic eliminated, which at \$0.01/GB and round-the-clock operation works out to roughly **\$3,200/month** of egress that the binary format saves — for one mid-size fleet, on payload size alone, before counting that JSON's verbosity also inflates the *response* direction.

Now **CPU**, which is the bigger cost at this scale. At ~3μs to serialize a message as JSON versus ~0.9μs as Protobuf, the per-message saving is ~2.1μs of encode (and a similar saving on decode at the receiver). Across 2M RPS the encode side alone is 2,000,000 × 2.1μs ≈ **4.2 CPU-seconds per second** saved — i.e., ~4 full cores' worth of work eliminated continuously, and roughly double that once you count decode. Call it ~8 cores of steady-state CPU that the Protobuf fabric never has to provision, plus the garbage-collection pressure from all those transient JSON strings that you also avoid. On a fleet that is several large instances you do not have to run, every hour of every day. Add the connection-reuse win — no per-call TCP+TLS handshake on the warm HTTP/2 path — and you also shave a round trip and a handshake-CPU spike off the tail of every cold call.

None of these individual numbers is precise for *your* system; the payload, the cross-boundary fraction, the runtime, and the instance prices all move them. But the *structure* of the calculation is exactly how a senior justifies "we standardized on gRPC internally" to a finance-conscious org: at fleet scale, a 2× payload reduction and a 3× serialization-CPU reduction stop being micro-optimizations and start being headcount-sized line items. This is the same reasoning that drove Google to Stubby and that the case studies below echo.

## The decision framework: six questions that pick the tool

Pull the threads together into a procedure you can actually run in a design review. Figure 6 renders it as a decision tree; here is the reasoning behind each branch.

![A decision tree starting from the choice of API style, branching on whether traffic is internal east-west or external north-south, then on streaming needs and client variety, leading to gRPC, REST, or GraphQL recommendations](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-6.webp)

**1. Who consumes the contract — your own code or someone else's?** If both ends are services you own and ship together (east-west), you can use a contract-first binary protocol because you regenerate both clients in the same CI run. If a browser, a partner, or a long-lived external client consumes it (north-south), you need something ubiquitous and loosely-coupled. This question alone routes most decisions: east-west → lean gRPC; north-south → lean REST or GraphQL.

**2. What is the latency and throughput budget?** For a hot internal path doing tens of thousands of RPS where every millisecond and every CPU cycle counts — the order-to-inventory call in the checkout critical path — gRPC's smaller payloads, cheaper serialization, and connection reuse are worth real money. For a low-volume, human-paced call, the difference is noise and you should optimize for debuggability instead, which favors REST.

**3. Do you need streaming?** If you need to push a live feed — order status updates, a telemetry stream, a chat channel, a long-running job's progress — gRPC's first-class server-streaming and bidirectional streaming are the cleanest answer. REST forces you into Server-Sent Events or long-polling; GraphQL has subscriptions but they need a separate WebSocket transport and more setup. Streaming need → strong pull toward gRPC.

**4. How varied are the clients, and how badly do they over-fetch?** If one backend serves a mobile app, a web app, and a TV app that each want different field slices of the same data, GraphQL's client-specified queries eliminate the over-fetching and the proliferation of custom endpoints. This is the BFF/aggregation case, and it is GraphQL's home turf. If there is one client with a stable, simple access pattern, GraphQL's complexity is not worth it and REST is simpler.

**5. Do you need free HTTP caching and CDN-ability?** If the data is read-heavy, cacheable, and you want to serve it from a CDN edge without writing caching code, REST's `GET`-with-`ETag` model is a gift — a correctly-tagged public REST endpoint is the cheapest possible way to serve cacheable reads. gRPC and GraphQL both make you build caching yourself. Cacheable public reads → REST.

**6. What can your team operate at 3am?** Ergonomics are not a soft factor; they are an availability factor. A team that can `curl` an endpoint and read the response debugs incidents faster than one squinting at binary frames. gRPC's toolchain weight (`protoc`, plugins, codegen in the build) is real friction for a small team. Weigh the operational maturity you actually have, not the one you aspire to.

Run those six and the answer usually writes itself. The reason ShopFast lands on "gRPC inside, GraphQL/REST at the edge" is that questions 1–3 point one way for internal calls and questions 4–6 point the other way for the edge — which is not a contradiction, it is the *correct* heterogeneous answer.

## The pattern almost everyone converges on: gRPC inside, REST/GraphQL at the edge

Here is the punchline you can take to any design review, and it is the configuration mature fleets land on independently because the forces pushing toward it are structural, not fashionable. **Internal east-west traffic runs on gRPC**: services talk to each other with binary Protobuf over HTTP/2, contract-first, codegen for every language in the fleet, deadlines propagating through the call graph, streaming where needed. **External north-south traffic enters through an edge** — an API gateway or BFF — that speaks **REST and/or GraphQL** to the outside world and **gRPC inward** to the services.

This is not a compromise; it is using each tool where its constraints match. The edge gets the ubiquity, caching, and browser-friendliness of HTTP/JSON; the internal fabric gets the speed, typing, and streaming of gRPC; and the gateway is the *translation seam* where one becomes the other. The order service never has to be browser-callable because nothing outside the cluster ever calls it directly — the gateway does, over gRPC, and re-exposes a curated slice as REST or GraphQL. This also gives you a clean place to do auth-token exchange, rate limiting, and request shaping, which is exactly the gateway's job in [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) and the resilience layer in [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads).

The translation seam needs a real mechanism, and there are two common ones. You can run a proxy like **Envoy** with gRPC-JSON transcoding, which reads the same `.proto` (annotated with HTTP mappings) and exposes a REST/JSON façade that it transcodes to gRPC on the way in. Or you use **grpc-gateway**, a code generator that produces a reverse-proxy server translating a RESTful JSON API into gRPC. Either way, the `.proto` stays the single source of truth and the REST surface is generated from it. Figure 8 shows why this layer is not optional: a browser literally cannot speak raw gRPC, so *something* must transcode.

![A topology diagram showing a browser using gRPC-Web and a third party using REST both reaching an Envoy or gRPC-gateway proxy that transcodes their requests into native gRPC for the internal service alongside a mobile native gRPC client](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-8.webp)

```yaml
# envoy gRPC-JSON transcoder (excerpt) — REST/JSON in, gRPC out.
http_filters:
  - name: envoy.filters.http.grpc_json_transcoder
    typed_config:
      "@type": type.googleapis.com/envoy.extensions.filters.http.grpc_json_transcoder.v3.GrpcJsonTranscoder
      proto_descriptor: "/etc/envoy/order.pb"   # compiled from order.proto
      services: ["shopfast.order.v1.OrderService"]
      print_options: { add_whitespace: true, always_print_primitive_fields: true }
```

## Stress test: what breaks when the assumptions break

A design is only as good as its behavior under the conditions you did not plan for. Let us deliberately break each choice the way production breaks it, because the failure modes are where the real understanding lives — and this series insists you stress-test every design ([handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) goes deeper on the recovery side).

**What happens to gRPC when a browser or a third party needs the API?** This is the most common gRPC regret, and it shows up not at design time but six months later when the product team says "we need a web dashboard" or "a partner wants to integrate." Raw gRPC is dead on arrival in a browser — the browser's fetch/XHR machinery cannot produce the HTTP/2 frame patterns gRPC's protocol requires, and there is no way around that from page JavaScript. Your options are all friction: adopt **gRPC-Web** (a browser-compatible variant) which *still* requires a translating proxy (Envoy) in front because gRPC-Web is not wire-compatible with gRPC; or stand up a **REST/JSON façade** via transcoding; or hand the partner a generated client in their language and hope they want one. The lesson is preventive: if there is any realistic chance an external or browser client will need a service's API, do not expose gRPC as the *only* contract — put a gateway in front from day one, exactly as Figure 8 shows, so the transcoding seam exists before you need it. Retrofitting a gateway under live traffic is far more painful than building it in.

**What happens to GraphQL under a malicious or careless deep query?** This is GraphQL's signature stability and security risk, and it is severe because GraphQL hands the client the ability to *shape the server's workload*. Consider a recursive schema — an `Order` has `items`, an item has a `product`, a product has `relatedProducts`, each of which has `items`, and so on. A hostile client (or a buggy one) can submit a query nested twenty levels deep, or a query that requests a million-element list, and a naive server will dutifully try to resolve it, fanning out into a combinatorial explosion of resolver calls and backend queries that exhausts CPU, memory, and connection pools — a denial-of-service from a single small request. The query *body* might be 2KB; the *work* it triggers can be unbounded. This asymmetry — tiny request, enormous server work — is what makes it a security problem and not merely a performance footgun. With REST, a client that wants to hammer you has to send many requests, which your rate limiter sees and throttles; with an unguarded GraphQL endpoint, one request slips past the rate limiter and then detonates inside the resolver tree where the rate limiter cannot see it. The defense therefore has to live *inside* the GraphQL layer, analyzing the query's shape before execution, because by the time the work starts fanning out it is already too late to count requests.

The defenses are layered, and a public GraphQL endpoint that lacks them is a liability. Figure 9 stacks them. First, **query depth limiting**: reject any query nested beyond a fixed depth (say 10), which kills the recursive-explosion class outright. Second, **query cost analysis**: assign each field a cost, sum the cost of an incoming query *before* executing it, and reject anything over a budget (say 1,000 points) — this catches the "wide" attacks (a list of a million items) that depth limits miss. Third, **persisted queries**: in production, do not accept arbitrary query strings at all; register the exact set of queries your real clients use at build time, give each a hash, and let clients send only the hash. The server executes only allowlisted queries, which eliminates arbitrary-query attacks entirely and shrinks the request to a tiny hash. Fourth, **resolver timeouts and the DataLoader batching** from the worked example, so even an allowed query cannot run away. Disable introspection in production for public endpoints too, so attackers cannot trivially map your schema.

![A layered defense stack for a public GraphQL endpoint showing persisted-query allowlisting, max depth limits, a query-cost budget, and resolver timeouts with DataLoader wrapping the raw resolver layer](/imgs/blogs/rest-vs-grpc-vs-graphql-for-service-apis-9.webp)

```python
# graphql query-cost guard (excerpt) — reject before executing.
from graphql import validate, parse

MAX_DEPTH = 10
MAX_COST = 1000  # field-cost budget per query

def guard(query_str: str, schema):
    doc = parse(query_str)
    if query_depth(doc) > MAX_DEPTH:
        raise ValueError("query too deep (max 10)")
    if estimated_cost(doc, schema) > MAX_COST:
        raise ValueError("query too expensive (budget 1000)")
    return validate(schema, doc)  # standard schema validation last
```

**What happens to REST when the data is graph-shaped?** REST degrades into chattiness. A screen that needs an order, its items, each item's product, and each product's reviews becomes a cascade of round trips, and on a slow client network the latency stacks. The honest answer is that this is precisely the case GraphQL or a custom aggregation endpoint exists to solve — so when you find a REST API where clients routinely make 5+ chained calls to render one view, that is the signal to put a GraphQL BFF or a purpose-built composite endpoint in front of it, not to add a sixth call.

**What happens to gRPC during a rolling deploy with a contract change?** This is where Protobuf's discipline earns its keep, and where carelessness still bites. If you *add* a field with a new tag number, old and new code interoperate fine — old readers ignore the unknown tag, new readers see defaults from old writers. But if you *reuse* a tag number for a different field, or *change a field's type*, or *remove a required-by-convention field*, you get silent data corruption across the version skew window when both old and new pods are serving simultaneously. The rule, enforced by tooling like Buf's breaking-change detector, is: tag numbers are forever, only add, never repurpose — which is the contract-evolution discipline that [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) formalizes for every protocol.

## Optimization: making each style production-grade with measured wins

Each style has a different highest-leverage optimization, and naming them precisely is the difference between a 2× win and a rounding error.

For **gRPC east-west**, the biggest lever after connection reuse is **channel and connection pooling done right**. A single HTTP/2 connection multiplexes well, but most implementations cap concurrent streams per connection (commonly ~100), so a hot path doing 10k concurrent RPCs to one backend needs a small *pool* of connections (a handful) and a load balancer that spreads streams across them — get this wrong and you bottleneck on one connection's stream limit while CPU sits idle. The measurable win: spreading 10k concurrent calls across 4 connections instead of 1 can drop p99 from the hundreds of milliseconds (queued behind the stream cap) back to ~20ms. Second lever: enable **gzip compression** on large messages, but *not* on tiny ones — compression has a fixed CPU cost that loses money below ~1KB payloads, so set a compression threshold and measure both sides.

For **REST north-south**, the biggest lever is **caching the cacheable**. A read-heavy public endpoint with a correct `ETag` and `Cache-Control` can serve the overwhelming majority of requests from a CDN edge, never touching your origin. The measurable win is dramatic and cheap: if 80% of `GET /products/{id}` requests are repeat reads of popular products, a CDN with a 60-second TTL can absorb 80% of that traffic at ~5ms edge latency, cutting origin load by 5× and origin egress cost proportionally. The second lever is **pagination and field selection** — return 20 items with a cursor, not 10,000 items in one response, and offer a `?fields=id,status,total` parameter so clients can opt out of over-fetching without you building GraphQL.

For **GraphQL**, the optimization story is the N+1 fix (DataLoader, measured above: ~480ms → ~22ms p99) plus **response caching at the field or entity level**. Because the whole-response `GET` cache is unavailable, you cache *resolved entities* — cache the `Product` by SKU in Redis with a short TTL, and the DataLoader checks the cache before the database, so popular products are served from memory across all queries that reference them. The measurable win compounds with batching: a screen that hit the database 51 times naively, then 2 times with DataLoader, can hit it **0 times** when the products are warm in the entity cache. Cross-service caching is its own deep topic in [caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services).

The discipline across all three is the same and it is the series' refrain: **measure before and after, in p50/p99/p999 latency, throughput, CPU, and dollar cost, under realistic load.** A "faster" protocol that you cannot operate, or a cache you cannot invalidate correctly, is not an optimization. The numbers in this post — 1,180 vs 560 bytes, 34ms vs 19ms p99, 51 vs 2 queries, 480ms vs 22ms — are the *shape* of the wins; your job is to reproduce the shape on your own load and confirm the magnitude before you commit a design to it.

## Case studies

The history of these three tools is, conveniently, a set of large companies discovering the framework above the hard way and then writing it down. None of these organizations decided "gRPC inside, GraphQL at the edge" from a whiteboard; each arrived at its piece of the pattern by hitting the cost the framework predicts — Google by drowning in serialization CPU at fleet scale, GitHub and Netflix by drowning in either over-fetching or a proliferation of bespoke endpoints, and the gRPC-everywhere shops by hitting the browser wall. Read these as the empirical backing for the decision tree, not as trivia. Each one is accurate to order-of-magnitude; where a precise number is uncertain it is framed as such, because the lesson survives without the exact figure and inventing a precise number would only undermine it.

**Google's Stubby and the birth of gRPC.** Long before gRPC was public, Google's entire internal service fabric ran on **Stubby**, a binary RPC system over a multiplexed transport — essentially the internal ancestor of gRPC. The scale was the forcing function: Google's internal services exchange on the order of *tens of billions* of RPCs per second across the fleet, and at that volume the cost difference between text JSON and binary Protobuf is not a micro-optimization, it is the difference between a manageable serving footprint and an impossible one. The 2–3× payload reduction and 3× serialization-CPU reduction we measured for one order message, multiplied across that RPC volume, is *thousands of machines* and a large fraction of the latency budget for every product. gRPC is the open-sourced, standards-based (HTTP/2) version of that lesson, and the design choices — contract-first Protobuf, streaming, deadline propagation — are exactly the ones a fleet at that scale is forced to make. The takeaway for a junior: gRPC's east-west bias is not arbitrary; it is the residue of a decade of running the largest internal service mesh on the planet.

**GitHub's pivot to GraphQL for its public API.** GitHub ran a famous, mature, well-documented REST API (v3) for years, and in 2016 introduced a **GraphQL API (v4)** as the path forward for its public developer platform. The stated motivation maps precisely to the over/under-fetching framework: REST consumers were either making many round trips to assemble the data a page needed, or GitHub was building ever more specialized REST endpoints to avoid that, and neither scaled to the variety of integrations built on GitHub. GraphQL let integrators ask for exactly the fields they needed across the repository/issue/PR graph in one request, and gave GitHub a single typed schema to evolve. The lesson: when your API is *the product* and is consumed by a wildly varied set of external clients with different data needs, GraphQL's client-specified queries are a structural win over a proliferation of REST endpoints — this is the north-south, high-variety branch of the decision tree, chosen at scale.

**Netflix and the federated GraphQL edge.** Netflix has many client surfaces — TVs, phones, tablets, browsers, game consoles — each rendering a different experience over a large set of backend microservices. Netflix engineering has publicly described moving to a **federated GraphQL** architecture, where a single GraphQL schema at the edge is composed from many services that each own a slice of it, so client teams query one graph while backend teams own their subgraphs independently. This is the GraphQL-BFF pattern at organizational scale, and it is solving the exact problem the framework predicts GraphQL is for: many heterogeneous north-south clients, a deeply relational domain, and a desire to stop each client team from hand-assembling data across services. The lesson: GraphQL's value is highest at the edge of a large, multi-client, microservice estate — and federation is how you keep that single schema from becoming a monolith again.

**A company standardizing gRPC internally.** Many engineering organizations that adopt microservices eventually mandate gRPC for *all* internal service-to-service calls and forbid internal REST, because a single contract-first standard buys uniform codegen across a polyglot fleet (Go, Java, Python services all generated from the same `.proto`), uniform deadline propagation, uniform observability hooks, and a `protoc`-enforced contract discipline. Companies like **Square, Lyft, and Dropbox** have publicly discussed gRPC adoption for internal traffic at significant scale. The recurring lesson — and the recurring regret — is the browser/partner edge: teams that standardized gRPC *everywhere* and then needed a web or partner API discovered they had to retrofit a transcoding gateway, which is why the mature end-state is always "gRPC internal, translating gateway at the edge" rather than "gRPC everywhere." The framework in this post is, in effect, the distilled scar tissue from that retrofit.

## When to reach for each (and when not to)

Decisive recommendations, because the point of a framework is to commit.

**Reach for gRPC** when both ends are internal services you own and deploy together, when the path is latency- or throughput-sensitive, when you need streaming, when you have a polyglot fleet that benefits from uniform codegen, and when your team has (or will invest in) the `protoc`/Buf toolchain. This is the default for *internal east-west* traffic in a serious microservices fleet. **Do not reach for gRPC** as the *only* contract for anything a browser or third party will call — you will be forced to bolt on a transcoding gateway anyway, so put one there from the start. Do not reach for it for a tiny, low-volume internal call where the toolchain weight outweighs the marginal latency win; plain REST is fine and more debuggable.

**Reach for REST** when the consumer is external and heterogeneous, when the data is cacheable and you want free HTTP/CDN caching, when the access pattern is simple and resource-oriented, when partners want to `curl` it, and when debuggability and ubiquity matter more than raw speed. This is the default for *public, cacheable, resource-shaped north-south* APIs. **Do not reach for REST** when clients routinely chain 5+ calls to render one view (you have a chattiness problem GraphQL or a composite endpoint solves) or when you need streaming or strict typing across a fast internal path (gRPC wins).

**Reach for GraphQL** when one backend serves many varied clients (mobile, web, TV) that each want different field slices, when the domain is deeply relational and over-fetching is a real cost, and when an aggregation/BFF layer at the edge is the right place to compose data. This is the default for *high-variety north-south aggregation*. **Do not reach for GraphQL** for internal service-to-service calls (the flexibility is wasted and gRPC is faster and simpler there), for a single client with a stable simple access pattern (REST is less complex), or for a public endpoint you will not protect with depth limits, cost analysis, and persisted queries — an unguarded public GraphQL endpoint is a DoS waiting to happen.

And the meta-recommendation, the one that resolves the ShopFast review: **you do not have to pick one for the whole system.** The mature answer is heterogeneous by design — gRPC where the framework says gRPC, REST and GraphQL where the framework says edge. A senior is comfortable running all three because each is the *correct* answer to a different question, and the gateway is the seam that lets them coexist. This is consistent with the broader posture in [what are microservices and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them) and the build-quality bar in [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice): pick the cost you can afford for the job in front of you, name it explicitly, and move on.

## Key takeaways

- **The first question is direction, not tool.** North-south (external, heterogeneous, untrusted) wants REST or GraphQL; east-west (internal, homogeneous, trusted) wants gRPC. Get the direction right and the tool mostly falls out.
- **REST's superpowers are ubiquity, free HTTP caching, and debuggability; its taxes are over/under-fetching, chattiness on graph data, and no schema by default.** It is the right default for public, cacheable, resource-shaped APIs.
- **gRPC's superpowers are small binary payloads, cheap serialization, connection reuse, first-class streaming, and contract-first codegen; its taxes are no browser support, opaque debugging, and toolchain weight.** It is the right default for internal east-west traffic, which is why Google's fleet runs on its ancestor.
- **GraphQL's superpower is client-specified queries that kill over-fetching and collapse round trips at an aggregation edge; its taxes are the N+1 resolver trap, no free HTTP caching, and a query-abuse attack surface.** It is the right default for high-variety north-south BFF traffic.
- **The numbers are real and reproducible in shape:** Protobuf is ~2× smaller and ~3× cheaper to serialize than JSON for the same payload; binary-over-HTTP/2 cuts p99 meaningfully at 10k RPS; a DataLoader turns a 51-query N+1 into 2 queries and drops p99 from ~480ms to ~22ms.
- **Never expose gRPC as the only contract for anything browsers or partners might call** — put a transcoding gateway (Envoy / grpc-gateway) at the edge from day one rather than retrofitting one under live traffic.
- **Never ship a public GraphQL endpoint without depth limits, query-cost analysis, persisted queries, and resolver batching** — an unguarded one is a one-request DoS.
- **The mature end-state is heterogeneous: gRPC inside, REST and/or GraphQL at the edge, a gateway as the translation seam.** This is not a compromise; it is each tool used where its constraints fit.
- **Measure before and after — p50/p99/p999, throughput, CPU, dollars, under realistic load — before committing a design to a "faster" protocol.** The protocol you cannot operate at 3am is slower than the one you can.

## Further reading

- *Building Microservices* (2nd ed.), Sam Newman — the communication chapters cover synchronous vs asynchronous styles and the trade-offs framework this post applies.
- *Microservices Patterns*, Chris Richardson — the API and communication patterns, including the API gateway and the costs of synchronous coupling.
- The official **gRPC documentation** (grpc.io) and the **Protocol Buffers** language guide — the canonical source for `.proto` syntax, streaming, deadlines, and wire-format details.
- The **GraphQL specification** (spec.graphql.org) and the **Apollo / DataLoader** docs — the source for schema design, resolvers, and the N+1 batching pattern.
- Roy Fielding's dissertation, *Architectural Styles and the Design of Network-based Software Architectures* — what REST actually means, from the source.
- This series: [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) for the why-before-the-how; [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) for the edge translation seam; [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) for evolving these contracts safely; and [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) for when the right answer is no synchronous call at all.

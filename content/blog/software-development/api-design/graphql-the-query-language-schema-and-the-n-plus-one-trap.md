---
title: "GraphQL: The Query Language, the Schema, and the N+1 Trap"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The honest deep dive on GraphQL — how the client shapes the response in one request, the schema and resolvers that make it work, the N+1 trap that ambushes every naive list resolver, how DataLoader batches it from N+1 to 1+1, and the caching, cost-limiting, and authorization problems GraphQL hands you in return."
tags:
  [
    "api-design",
    "api",
    "graphql",
    "schema",
    "resolvers",
    "dataloader",
    "n-plus-one",
    "rest",
    "http",
    "developer-experience",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-1.png"
---

A mobile engineer once showed me a network trace from our commerce app's order-history screen. To paint one row — an order's total, the customer's name, and whether the payment had cleared — the app fired three requests in sequence: `GET /orders/9021`, then `GET /customers/42`, then `GET /payments?order_id=9021`. Each one had to *finish* before the next could start, because the customer id came back inside the order and the payment query needed the order id. On a strong office Wi-Fi connection that waterfall cost maybe 90 milliseconds. On a congested mobile link in a basement, each round-trip was 250–400 ms, and the three of them serialized into well over a second of staring at a spinner — for one row. The list screen had twenty rows. We were not slow because our database was slow. We were slow because the **shape of the API forced the client to make conversation when it wanted to make one statement.**

That trace is the cleanest argument for GraphQL I have ever seen. REST gave the mobile team a fixed menu: the `/orders` endpoint returns *the order representation the server decided on*, no more and no less. If that representation has fifty fields and the screen needs three, the client over-fetches forty-seven fields it throws away — wasted bytes on a metered connection. If the representation is lean and the screen needs the customer's name, the client under-fetches and must make a second round-trip to go get it — the mobile waterfall. **Over-fetching and under-fetching are the same disease seen from two sides: the server, not the client, decides the shape, and the server cannot know every screen in advance.** GraphQL's founding idea is to hand that decision to the caller. The client sends a query describing *exactly* the tree of fields it wants — orders, and for each order its customer's name and its payment's status — and the server returns precisely that tree, in one request.

![A before-and-after comparison contrasting a three-hop REST waterfall against a single GraphQL request that fetches orders with customer and payments together](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-1.png)

This is the honest deep dive. GraphQL is genuinely powerful for the problem it was built for — many heterogeneous clients hitting an aggregation layer over many backend services — and it is genuinely a poor fit for a simple single-client CRUD app, where it buys you the N+1 problem and the caching problem and the query-cost problem in exchange for flexibility you do not need. By the end of this post you will be able to read and write a GraphQL **schema** (the typed contract), trace how a query resolves through **resolvers** (the functions behind each field), *derive* the **N+1 problem** from first principles and fix it with **DataLoader** (the per-request batcher), and reason about the hard problems GraphQL hands back to you — caching with no URL to cache, depth and complexity limits so a malicious query cannot take the database down, partial errors, per-field authorization, and versioning by deprecation. We will use the running Payments & Orders API the whole way. This post lives in the same series as the contract-first framing in [what an API is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and the paradigm-choice decision in [choosing a paradigm by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); GraphQL is one paradigm you choose by force, never by fashion.

## What GraphQL actually is (and is not)

Let me kill three myths before the schema, because they cause more bad GraphQL than any technical mistake.

**GraphQL is not a database query language.** The "QL" misleads people into thinking it talks to a database the way SQL does. It does not. GraphQL has no idea what a table, a row, or a join is. It is a language for describing a *graph of typed fields* and a runtime for *resolving* those fields by calling whatever code you wire up — a SQL query, a REST call to another service, a cache lookup, a hardcoded constant. The graph in "GraphQL" is the type graph of your domain (an `Order` has a `Customer`, a `Customer` has many `Order`s), not a graph database. Your resolvers can be backed by Postgres, by three microservices, and by a Redis cache all at once, and the client never knows.

**GraphQL is not inherently faster than REST.** It can be slower. The same naive resolver that triggers the N+1 problem will issue *more* database queries than a well-designed REST endpoint with a hand-tuned join. GraphQL's win is on the *network*: it removes round-trips between the client and your server. It does nothing for the round-trips between your server and your data — those you must batch yourself. People who adopt GraphQL expecting magic performance and then watch their database melt under N+1 queries are the most common GraphQL casualties.

**GraphQL is not RESTful, and it does not try to be.** It is closer to a typed RPC (remote procedure call) over HTTP than to REST. There is one endpoint, almost always `POST /graphql`. There are no resource URIs, no HTTP verbs carrying semantics, no status codes telling you what happened to your data. A GraphQL request that asks for a field you are not allowed to see still returns `200 OK` — the failure is reported *inside the response body*, in an `errors` array. This single fact ripples into everything: caching, error handling, monitoring, and authorization all work differently because the HTTP layer is no longer carrying the meaning. We will pay that bill in full later in the post.

So what *is* it? GraphQL is a **specification** (the language plus the type system plus the execution algorithm) and an ecosystem of implementations. A GraphQL API is defined by a **schema** — a strongly typed contract that says exactly which queries are legal and what shape each one returns. The client writes a query against that schema; the server validates it against the schema before executing a line of resolver code; and the response is guaranteed to match the requested shape. The schema is the contract, exactly as this series insists every API is a contract. The difference from REST is *where* the contract lives: in REST it is spread across URIs, methods, status codes, and an OpenAPI document; in GraphQL it is concentrated in one schema document. That concentration is GraphQL's greatest strength and the root of most of its problems.

## The schema: a typed contract in SDL

The schema is written in **SDL** — the Schema Definition Language, GraphQL's human-readable type syntax. Let me build the Payments & Orders schema piece by piece, because every concept you need is right here.

```graphql
# Scalars are the leaf types — the concrete values, never further selectable.
# GraphQL ships five built-ins: Int, Float, String, Boolean, ID.
# You declare custom scalars for domain values the built-ins do not cover.
scalar DateTime

# An enum constrains a field to a fixed set of named values.
enum OrderStatus {
  OPEN
  PAID
  REFUNDED
  CANCELLED
}

enum PaymentStatus {
  PENDING
  SUCCEEDED
  FAILED
}

# An object type is a named bag of fields. Each field has a type.
# A trailing "!" means NON-NULL: the server promises this is never null.
# "[Payment!]!" means a non-null list of non-null Payments (no nulls inside, never the whole list null).
type Order {
  id: ID!
  status: OrderStatus!
  totalCents: Int!
  createdAt: DateTime!
  # A field can return another object type — this is the "graph" edge.
  customer: Customer!
  # A field can take ARGUMENTS, here to paginate the order's payments.
  payments(first: Int = 10, after: String): PaymentConnection!
}

type Customer {
  id: ID!
  email: String!
  name: String          # nullable: the server may not know the name yet
  orders(first: Int = 20, after: String): OrderConnection!
}

type Payment {
  id: ID!
  amountCents: Int!
  status: PaymentStatus!
  processedAt: DateTime
}
```

Read `totalCents: Int!` as "the field `totalCents` is a non-null integer." We store money as integer cents, never floats, because a `Float` in JSON cannot represent `\$0.10` exactly and you do not want rounding drift in a payments system. The `!` non-null marker is the single most important contract tool in SDL, and it cuts both ways. On a **response** field, `!` is a *promise the server makes to the client*: `id: ID!` says "I will always give you an id." On an **argument** or input field, `!` is a *requirement the server imposes on the client*: a non-null argument must be supplied. Getting null-ability right is the heart of GraphQL contract design, and we will see exactly why it is load-bearing when we get to error handling — a single null in a non-null field can blank out an entire branch of the response.

### The three root types

Every GraphQL schema has up to three special **root types** that are the entry points — the only places a client can start a request:

```graphql
# Query: the read root. Everything a client can FETCH hangs off here.
type Query {
  order(id: ID!): Order
  customer(id: ID!): Customer
  orders(first: Int = 20, after: String, status: OrderStatus): OrderConnection!
}

# Mutation: the write root. Everything that CHANGES state hangs off here.
type Mutation {
  createPayment(input: CreatePaymentInput!): CreatePaymentPayload!
  refundPayment(paymentId: ID!, amountCents: Int!): RefundPaymentPayload!
}

# Subscription: the realtime root. A long-lived stream of events.
type Subscription {
  paymentStatusChanged(orderId: ID!): Payment!
}
```

The third root, **`Subscription`**, is the realtime one, and it is worth a word because it changes the transport. A subscription is a *long-lived* operation: the client opens a persistent connection (almost always a WebSocket using the `graphql-ws` protocol, occasionally Server-Sent Events) and the server pushes a new payload every time the subscribed event fires — here, every time a payment's status changes for a given order. Unlike a query, a subscription does not resolve once and close; it stays open, streaming results until the client unsubscribes or the connection drops. That makes subscriptions operationally heavier than queries — you hold a connection per subscriber, you need a pub/sub backbone behind the resolver to fan events out, and you must handle reconnection and missed-event replay yourself. Many teams that think they want subscriptions are better served by plain webhooks or polling a `Query` field; reach for subscriptions only when sub-second realtime genuinely matters, and lean on a real broker (the delivery-guarantee mechanics live in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once)) rather than trying to make GraphQL itself durable.

The split is not cosmetic; it carries the same safe-versus-unsafe distinction this series has hammered on with HTTP methods (see [methods and idempotency](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete)). **`Query` fields must be side-effect-free** — they are the moral equivalent of `GET`, and GraphQL execution is allowed to run sibling query fields *in parallel and in any order* precisely because reads do not interfere. **`Mutation` fields are the only place state changes**, and the spec guarantees that *top-level mutation fields execute serially, left to right*, so two mutations in one request happen in a defined order. If you put a side effect in a query resolver you have broken a contract the client is entitled to rely on, and you have created a bug that will surface only when the engine reorders or parallelizes your reads. The lesson maps straight onto the safe/idempotent reasoning in [HTTP for API designers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers).

### Inputs, interfaces, and unions

Mutations take **input types**, which are a distinct kind from object types. Output objects can have fields with arguments and resolvers; input objects are pure data the client sends in. Keeping them separate prevents a whole class of mistakes:

```graphql
input CreatePaymentInput {
  orderId: ID!
  amountCents: Int!
  # An idempotency key so a client retry does not double-charge.
  idempotencyKey: String!
}

type CreatePaymentPayload {
  payment: Payment
  # We return a typed list of errors rather than throwing, so the
  # client can handle "card declined" without parsing a stack trace.
  userErrors: [UserError!]!
}

type UserError {
  field: String
  message: String!
  code: String!
}
```

Notice the `idempotencyKey` riding inside the input. GraphQL gives you no transport-level idempotency the way an HTTP `Idempotency-Key` header does, because everything is `POST /graphql`. You have to thread it through the schema yourself, and your `createPayment` resolver has to honor it — the same safe-retry discipline covered in [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions), just relocated from a header into an input field. This is the first of several places where GraphQL hands a job back to you that REST got for free from HTTP.

When a field can return *one of several types*, SDL gives you two tools. An **interface** is a contract a set of types share — common fields with possibly type-specific extras:

```graphql
interface Node {
  id: ID!
}

type Order implements Node {
  id: ID!
  # ...rest of Order
}

# A union groups unrelated types with NO shared fields — e.g. a
# search result that might be an Order, a Customer, or a Refund.
union SearchResult = Order | Customer | Refund
```

The `Node` interface is not an accident of my example — it is the cornerstone of the **Relay specification**, a conventions document (from the Relay GraphQL client) that standardizes object identification and pagination. Every object that implements `Node` is globally re-fetchable by a single opaque `id`, which is what lets a client cache and re-hydrate any object it has seen. We will return to Relay when we hit pagination and caching, because its `Connection` pattern is the disciplined answer to "how do I page a list without letting a client ask for ten million rows."

![A taxonomy tree rooting the Query type over Order and Customer object types that branch down into their concrete non-null scalar fields](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-2.png)

The tree above is worth pausing on. A schema *is* a connected type graph rooted at `Query`, where every path you can walk eventually bottoms out in a scalar — a leaf you cannot select into further. A client query is a *sub-tree* of this graph: it starts at a root field and selects a path down to the scalars it wants. The runtime's whole job is to walk that sub-tree and produce a JSON object with exactly that shape. Understanding the schema as a graph and a query as a sub-tree is the single most useful frame for everything that follows.

## Queries: the client picks the shape

A GraphQL **query** is a **selection set** — a nested set of curly braces naming the fields you want. Here is the query that replaces our three-hop mobile waterfall with one request:

```graphql
query OrderHistoryRow {
  order(id: "9021") {
    id
    totalCents
    status
    customer {
      name
    }
    payments(first: 1) {
      edges {
        node {
          status
          amountCents
        }
      }
    }
  }
}
```

The response mirrors the query's shape *exactly* — this isomorphism between request and response is GraphQL's signature property and the thing that makes it feel so good to consume:

```json
{
  "data": {
    "order": {
      "id": "9021",
      "totalCents": 4999,
      "status": "PAID",
      "customer": { "name": "Mai Tran" },
      "payments": {
        "edges": [
          { "node": { "status": "SUCCEEDED", "amountCents": 4999 } }
        ]
      }
    }
  }
}
```

The client asked for `totalCents`, `status`, the customer's `name`, and the latest payment's `status` and `amountCents` — and got precisely those, nested precisely as requested, no over-fetch and no second round-trip. The top-level `data` envelope is mandated by the spec; we will see its companion `errors` array shortly.

### Variables, aliases, fragments

Hardcoding `"9021"` into the query is wrong for a real client — you would have to send a new query string for every order, defeating caching and inviting injection. Instead you parameterize with **variables**, declared with a `$` sigil and sent as a separate JSON map:

```graphql
query OrderHistoryRow($orderId: ID!, $payCount: Int = 1) {
  order(id: $orderId) {
    totalCents
    status
    customer { name }
    payments(first: $payCount) {
      edges { node { status amountCents } }
    }
  }
}
```

```json
{ "orderId": "9021", "payCount": 1 }
```

The query string is now constant across requests — only the variables map changes. That constancy is what makes **persisted queries** possible later (you register the constant string once, then send only its hash). Variable types are checked against the schema: `$orderId: ID!` must be supplied because it is non-null, and passing a string where an `Int` is expected is rejected at validation time, before any resolver runs.

**Aliases** let you rename a field in the response or request the same field twice with different arguments:

```graphql
query TwoOrders {
  paid: order(id: "9021") { status }
  pending: order(id: "9022") { status }
}
```

Without aliases both fields would collide on the key `order` in the response; with them you get `data.paid` and `data.pending`. **Fragments** factor out a repeated selection set so you do not copy-paste field lists across queries — and they are how real clients keep their queries DRY as the app grows:

```graphql
fragment OrderRow on Order {
  id
  totalCents
  status
  customer { name }
}

query OrderList {
  orders(first: 20) {
    edges { node { ...OrderRow } }
  }
}
```

The fragment `OrderRow on Order` declares the type it applies to, so the engine can validate that every field in it actually exists on `Order`. This validation-before-execution is a recurring theme: because the schema is fully typed, an enormous class of "you asked for a field that does not exist" errors is caught statically, returned as a validation error, and never reaches your resolver code.

#### Worked example: one query replaces three REST round-trips

Let me make the win concrete and quantified, because "fewer round-trips" is easy to wave at and easy to over-claim.

**REST, the waterfall.** The mobile order-history row needs the order, its customer, and its latest payment. The customer id lives inside the order body and the payment is keyed by order id, so the three calls are *serially dependent* — each must complete before the next can issue:

```http
GET /orders/9021            -> 200, body includes customer_id: 42   (~250 ms RTT)
GET /customers/42           -> 200, body includes name              (~250 ms RTT, blocked)
GET /payments?order_id=9021 -> 200, latest payment                  (~250 ms RTT, blocked)
```

On a link with a 250 ms round-trip, the total wall-clock cost is roughly $3 \times 250 = 750$ ms, because the dependency chain forbids parallelism. The order body also carried perhaps forty fields the screen never showed — over-fetch on top of under-fetch. With twenty rows on screen, a naive client repeats this per row; even a smart client that bulk-fetches still pays the three serial hops once and re-shapes on the device.

**GraphQL, one request.** The single query above bundles all three needs into one `POST /graphql`. The wall-clock network cost is one round-trip: roughly $1 \times 250 = 250$ ms, a clean $3\times$ reduction in *network latency* for this screen, plus the payload carries only the four requested fields instead of forty. The generalization is the real point: if a screen needs data that REST exposes across $k$ serially dependent endpoints, REST pays $\approx k \cdot \text{RTT}$ and GraphQL pays $\approx 1 \cdot \text{RTT}$. As $k$ grows — and on a rich screen $k$ is easily five or six — the gap widens linearly. **This is the entire value proposition of GraphQL, stated honestly: it collapses client-to-server round-trips.** It does not, by itself, collapse server-to-database round-trips. That is the next section, and it is where naive GraphQL goes to die.

## Mutations: changing state through the schema

Reads are the easy half. Here is the create-payment mutation in full, with the idempotency key doing its job:

```graphql
mutation CreatePayment($input: CreatePaymentInput!) {
  createPayment(input: $input) {
    payment {
      id
      status
      amountCents
    }
    userErrors {
      field
      code
      message
    }
  }
}
```

```json
{
  "input": {
    "orderId": "9021",
    "amountCents": 4999,
    "idempotencyKey": "pay_req_8f3a2c"
  }
}
```

Two design decisions in that payload are worth defending. First, the mutation returns *both* a `payment` and a `userErrors` list, and on a business-logic failure like a declined card it returns `200 OK` with `payment: null` and a populated `userErrors` rather than throwing a top-level GraphQL error. This is the **payload-with-userErrors** pattern Shopify popularized, and the reasoning is that a declined card is an *expected* outcome the client must handle in the UI, not an *exceptional* one that belongs in the protocol-level `errors` array. Reserve the top-level `errors` for genuinely exceptional failures — a malformed query, an internal crash, an auth failure — and put recoverable business outcomes in a typed `userErrors` field. Your clients will thank you, because they can pattern-match on `code` instead of string-parsing a generic error.

Second, the `idempotencyKey` makes the mutation *safe to retry*. If the network times out after the charge succeeded but before the response arrived, the client resends the identical mutation with the same key, and the resolver — which recorded the key the first time — returns the *original* payment rather than charging again. Without it, a single timeout double-charges the customer, which is exactly the kind of consequence this series exists to prevent.

## Resolvers: every field is a function

Here is the mental shift that unlocks GraphQL's performance behavior. **Every field in the schema is backed by a resolver — a function that returns that field's value.** When a query arrives, the engine walks the selection set and calls the matching resolver for each requested field, passing down the *parent* object as the first argument. Scalar fields usually need no explicit resolver (the engine reads the property off the parent object — a "default resolver"), but every object-to-object edge — `order.customer`, `customer.orders` — is a function you write.

A resolver's signature is `(parent, args, context, info)`. `parent` is the value returned by the field one level up; `args` are this field's arguments; `context` is a per-request object you populate with the authenticated user, the database connection, and — crucially — your DataLoaders; `info` carries the query AST and field path. Here is the Payments & Orders root and edge resolvers in JavaScript:

```javascript
const resolvers = {
  Query: {
    // Root resolver: fetch one order by id from the database.
    order: (parent, args, context) =>
      context.db.orders.findById(args.id),

    orders: (parent, args, context) =>
      context.db.orders.findPage({
        first: args.first,
        after: args.after,
        status: args.status,
      }),
  },

  Order: {
    // EDGE resolver: given an Order (the parent), fetch its Customer.
    // This is the field that will trigger N+1 — watch it closely.
    customer: (order, args, context) =>
      context.db.customers.findById(order.customerId),

    payments: (order, args, context) =>
      context.db.payments.findPageByOrder(order.id, args),
  },

  Customer: {
    orders: (customer, args, context) =>
      context.db.orders.findPageByCustomer(customer.id, args),
  },
};
```

Read `Order.customer` carefully: it receives the *already-fetched* order as `order`, pulls `order.customerId`, and issues a database lookup to fetch that one customer. For a single order this is fine — one query for the order, one for its customer, two total. The trap is what happens when the query asks for a *list* of orders and the customer of each. The engine does not know your resolvers share a database; it just calls `Order.customer` once per order in the list, dutifully, in a loop. And there is the N+1.

![A branching resolution tree where the orders list resolver fans out to per-order objects whose customer and payments resolvers merge back into one response](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-3.png)

The resolution tree above shows the structure that causes the problem. The `Query.orders` resolver runs once and returns N order objects. Then for *each* of those N orders, the engine invokes the `Order.customer` resolver — N separate invocations, each issuing its own database query. The branches then merge back into a single response object. The fan-out from one parent resolver to N child-resolver invocations is structurally identical for any list-of-objects field, which is why N+1 is not a bug in one resolver but a *pattern* that ambushes every naive list resolver you will ever write.

### How the engine executes a query, step by step

It is worth slowing down on the execution algorithm itself, because everything about GraphQL's performance and error behavior falls out of it. The runtime processes a request in four distinct phases, and a malformed request can be rejected at any of the first three before a single line of your resolver code runs:

1. **Parse.** The query string is lexed and parsed into an abstract syntax tree. A syntax error — an unclosed brace, a stray character — is rejected here with a parse error in the `errors` array. No resolver runs.
2. **Validate.** The AST is checked against the schema: does every field exist on its type, are the argument types correct, are non-null variables supplied, are fragments applied to compatible types, does the query exceed your depth or complexity limits? A query that asks for `order.flavor` (no such field) is rejected here. This is the phase that makes GraphQL feel safe: an enormous class of errors is caught statically, against the typed schema, before execution. It is also where you hang your cost-limiting guards — depth and complexity analysis are validation rules.
3. **Execute.** The engine walks the validated selection set. For each requested field it calls the field's resolver with `(parent, args, context, info)`, passing the parent resolver's return value down. Sibling fields under one parent may run concurrently (for queries); top-level mutation fields run serially, left to right. The result of each resolver becomes the parent for that field's own sub-selection, and the recursion bottoms out at scalars.
4. **Complete / coerce.** Each resolved value is coerced to its declared type. A value returned for an `Int!` field that is `null` triggers the non-null bubbling described earlier; an enum value not in the enum's value set is a coercion error. The final `data` tree is assembled to mirror the query exactly.

The single most consequential fact in that list is that **execution calls one resolver per field per object.** That phrasing — *per object*, not *per field* — is the entire N+1 story in five words. A field selected on a list of one hundred objects invokes its resolver one hundred times. The engine has no idea that one hundred invocations of `Order.customer` could share a database query; it is faithfully honoring the resolver contract, one field on one object at a time. DataLoader works precisely because it sits *underneath* this per-object invocation, collecting the hundred independent asks and collapsing them before they reach the database. Once you internalize "one resolver call per field per object," you can predict the query count of any GraphQL request by eye, which is the skill that separates engineers who ship safe GraphQL from those who ship a database-melting endpoint.

### Default resolvers and the trivial-field cost

A subtlety that trips people up: not every field needs a resolver you write. When a resolver returns an object whose property names match the field names — say `Query.order` returns a row object with `id`, `status`, `totalCents` columns — the engine's **default resolver** simply reads `parent.fieldName` for each scalar. You write resolvers only for fields that need *work*: a database fetch, a computed value, a call to another service, a permission-filtered list. This is why `Order.customer` needs a resolver (it has to fetch a different row) but `Order.totalCents` does not (it is already on the order object). The practical upshot is that your N+1 risk lives entirely in the *non-trivial* edge resolvers — the ones that fetch related objects. Audit those, not the scalar leaves.

## The N+1 problem, derived from first principles

Let me state the problem precisely, because vague intuitions about "too many queries" do not help you fix it. Consider this entirely reasonable client query:

```graphql
query OrderFeed {
  orders(first: 50) {
    edges {
      node {
        id
        totalCents
        customer {
          name
        }
      }
    }
  }
}
```

The client wants fifty orders and, for each, the customer's name. Now trace the resolver calls with the naive code from the last section:

1. The engine calls `Query.orders` **once**. It runs one database query — `SELECT * FROM orders ... LIMIT 50` — and returns 50 order objects. That is **1 query**.
2. For the `customer` field on each of those 50 orders, the engine calls `Order.customer` **50 times**, once per order. Each call runs `SELECT * FROM customers WHERE id = ?` for that order's `customerId`. That is **50 queries**.

Total database queries: $1 + 50 = 51$. Generalize to a list of $N$ parents: you issue **1 query for the list plus N queries for the children, for $N + 1$ queries total.** That is the N+1 problem, and now you can see its anatomy exactly: a single list resolver produces $N$ parents, and a child resolver that fetches per-parent runs once per parent. The "+1" is the list query; the "N" is the per-child queries. There is nothing GraphQL-specific about the *cause* — the same anti-pattern appears in any ORM that lazy-loads a relationship inside a loop. GraphQL simply makes it *easy and invisible* to write, because the per-field resolver model hides the loop. You wrote one innocent `Order.customer` function and the engine turned it into a fifty-iteration database hammer.

The cost is not theoretical. Each of those 51 queries is a separate network round-trip to the database, a separate parse-plan-execute cycle, a separate connection-pool checkout. If a single primary-key lookup takes 2 ms, 51 of them serialized is over 100 ms of database time for a query that *should* take two lookups. Scale the page to 200 orders, or nest one level deeper so each order also fetches its payments (another N child queries), and you are issuing hundreds of queries per request. This is the mechanism by which teams "adopt GraphQL for performance" and then watch their p99 latency and their database CPU both go through the roof. The B-tree index lookups themselves are fast — that is covered in [how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) — but firing N of them when one batched `WHERE id IN (...)` would do is pure self-inflicted waste.

![A before-and-after figure showing a naive resolver issuing N+1 database queries beside a DataLoader version that batches the children into a single IN query for 1+1 total](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-4.png)

## DataLoader: batching N+1 into 1+1

The fix has a beautiful shape. The N customer queries all have the same form — `SELECT * FROM customers WHERE id = ?` — differing only in the id. If we could *collect* all the customer ids the resolvers want during one request, then issue a single `SELECT * FROM customers WHERE id IN (?, ?, ?, ...)`, we would replace N queries with one. That is exactly what **DataLoader** does. DataLoader is a tiny per-request utility (originally from Facebook, now a standard library in every GraphQL ecosystem) that does two things: **batching** and **caching**.

The clever part is *when* it batches. DataLoader exploits the event loop's tick model. When you call `loader.load(id)`, it does not fire a query — it records the id, returns a promise, and *waits*. All the `load()` calls that happen during the same tick of the event loop (which is exactly what happens when the engine invokes `Order.customer` across all 50 orders before yielding) get collected into one list. At the *end* of the tick, DataLoader fires your **batch function** once with the whole list of ids, you run one `WHERE id IN (...)` query, and DataLoader resolves each individual promise with its matching row. The per-resolver code does not change shape — each resolver still "asks for one customer" — but underneath, fifty asks become one query.

Here is the same `Order.customer` resolver, now batched. The only change at the call site is `findById` becomes `context.loaders.customer.load`:

```javascript
const DataLoader = require("dataloader");

// Build a FRESH set of loaders for EVERY request. This is critical:
// the per-request cache must not leak data between users.
function buildLoaders(db) {
  return {
    customer: new DataLoader(async (ids) => {
      // ids is the BATCH: e.g. [42, 17, 42, 88, 17, ...]
      const rows = await db.customers.findByIds(ids); // ONE query: WHERE id IN (...)
      const byId = new Map(rows.map((r) => [r.id, r]));
      // MUST return results in the SAME ORDER as the input ids,
      // one entry per id, or DataLoader resolves the wrong promises.
      return ids.map((id) => byId.get(id) || null);
    }),
  };
}

const resolvers = {
  Order: {
    // Looks like a single fetch; is actually batched across the request.
    customer: (order, args, context) =>
      context.loaders.customer.load(order.customerId),
  },
};
```

Two contracts inside the batch function are non-negotiable, and they are the most common DataLoader bugs. First, **the batch function must return an array the same length as the input ids, in the same order.** DataLoader maps `result[i]` to the promise for `ids[i]` positionally. If your `WHERE id IN (...)` returns rows in database order, you must re-sort them to match the input order — which is what the `Map` does above. Get this wrong and customer 42 gets resolved with customer 17's data, a silent data-corruption bug that authorization checks will not catch. Second, **you must construct loaders per request, not once at startup.** A loader caches every key it sees for its lifetime; a process-wide loader would serve stale data and, far worse, leak one user's customer record into another user's response. Per-request construction is why `context` is the home for loaders.

Now retrace the N+1 query with loaders in place:

1. `Query.orders` runs **once** — 1 query, 50 orders. (The "+1".)
2. The engine calls `Order.customer` 50 times in one tick. Each call does `loaders.customer.load(customerId)`, which *queues* the id and returns a promise. No query fires yet.
3. The tick ends. DataLoader flushes the queue — say it collected ids `[42, 17, 42, 88, ...]` — *deduplicates* them (the per-request cache means a repeated id like `42` is loaded once), and fires the batch function with the unique ids. **1 query**: `SELECT * FROM customers WHERE id IN (42, 17, 88, ...)`.
4. DataLoader resolves all 50 promises from that single result set.

Total: $1 + 1 = 2$ queries, **regardless of N.** We went from $N + 1$ to a constant $2$. That is the headline result: **DataLoader turns the per-list cost from $O(N)$ database queries into $O(1)$.** And the deduplication is a bonus — if the same customer placed twenty of the fifty orders, the naive version still ran fifty queries, but the loader fetches each distinct customer exactly once.

![A timeline of one DataLoader batch tick where individual load calls are queued and deduplicated before the tick ends and a single batched query resolves them all](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-6.png)

#### Worked example: tracing N+1 and fixing it to 1+1

Let me run the whole thing end to end with numbers, the way you would in a code review.

**Setup.** A merchant dashboard requests `orders(first: 100)` and, per order, `customer { name }` and `payments(first: 1) { ... }`. There are 100 orders, owned by 60 distinct customers (some merchants are repeat buyers). Assume each database round-trip is 2 ms.

**Naive resolvers.** Query count: 1 for the orders list, plus 100 for `Order.customer` (one per order, no dedup), plus 100 for `Order.payments` (one per order). Total $= 1 + 100 + 100 = 201$ queries. At 2 ms each, serialized inside the request, that is $201 \times 2 = 402$ ms of database time for one screen. The customer queries are especially wasteful: 40 of them re-fetch a customer already fetched, because the naive code has no cache.

**With DataLoader.** Wrap `Order.customer` in a customer loader and `Order.payments` in a payments-by-order loader. Now: 1 query for the list; 1 batched `WHERE id IN (...)` for customers — and because of dedup it fetches only the **60 distinct** customers, not 100; 1 batched query for payments across all 100 order ids. Total $= 1 + 1 + 1 = 3$ queries. At 2 ms each that is $3 \times 2 = 6$ ms of database time. We cut database round-trips from 201 to 3 — a $67\times$ reduction — and the result is *correct by construction* because the loader's per-request cache also handles the repeat customers for free. (These are illustrative figures with a stated 2 ms assumption, not a measured benchmark; the *shape* — $O(N)$ collapsing to $O(1)$ — is the durable result, and it holds whatever your real per-query latency is.)

This is why I tell teams that **DataLoader is not optional in production GraphQL — it is part of the contract you make with your database.** A GraphQL server without loaders on its object-to-object edges is a denial-of-service generator pointed at your own data tier. Treat "is this edge resolver batched?" as a mandatory review checklist item, exactly as you would treat "is this list endpoint paginated?" in REST.

## The problems GraphQL hands back to you

Here is where the honest part of the deep dive earns its keep. GraphQL solved the client's fetching problem by moving the shape decision to the caller — but in doing so it walked away from a pile of things HTTP and REST gave you for free. You have to rebuild every one of them. Pretending otherwise is how teams get burned.

### Caching: there is no URL to cache

In REST, `GET /orders/9021` is a cache key. A CDN, a reverse proxy, the browser's HTTP cache, and an `ETag`-driven conditional request all key on that URL and method (the full mechanism is in [caching with ETags](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation), once it ships). GraphQL sends everything as `POST /graphql` with the query in the body, and `POST` is not cacheable, and even if it were, *every distinct selection set is a distinct response* — there is no stable URL to key on. The free HTTP caching layer simply does not apply.

You get it back in three layers, none free:

- **Persisted queries / APQ.** You register each query string on the server (by its SHA-256 hash) and clients then send only the hash plus variables. This shrinks the request, lets you `GET` the query (a `GET` with a hash in the query string *is* cacheable by a CDN), and — as a security bonus — lets you *allowlist* exactly which queries are permitted. Apollo's **Automatic Persisted Queries (APQ)** negotiates this: the client tries the hash, the server says "unknown, send the full query," the client sends it once, and from then on the hash suffices.
- **Field-level / entity caching.** Because every object has a global `id` (the Relay `Node` pattern), client libraries like Apollo Client and Relay maintain a normalized cache keyed by `__typename` plus `id`. When two queries both fetch `Order:9021`, the second is served from the cache. This is powerful but lives *on the client*, not in your shared infrastructure.
- **Server-side response/resolver caching.** You can cache individual resolver results (e.g. a Redis cache inside the `Order.customer` batch function) or whole responses keyed by the normalized query plus variables. This is hand-built and you own the invalidation, which — as always — is the hard part.

The honest summary: **REST gives you HTTP caching for free and GraphQL makes you build a caching strategy.** For a read-heavy public API where most responses are identical and cacheable at the edge, this is a real point in REST's favor.

There is a deeper reason the loss stings, and it is worth naming so you weigh it correctly. HTTP caching is *layered* — it works at the browser, at a corporate proxy, at the CDN edge, and at a reverse proxy in front of your origin, all without your application knowing or caring, because they all key on the same URL and `Cache-Control`/`ETag` semantics defined by RFC 9110. A cache hit at the CDN edge never touches your servers at all; that is how a popular REST resource serves a million reads from one origin fetch. GraphQL's `POST /graphql` is opaque to every one of those layers — they see a `POST` with a body and pass it straight through. So the caching you rebuild is *not equivalent*: APQ recovers edge-cacheability only for the exact persisted queries you register, entity caching lives on each individual client and does not help a cold client, and server-side response caching protects your database but still spends a full request round-trip and your origin's CPU. You can get to a good place, but you will write and operate code to do what REST got from the protocol. If your traffic profile is "many clients reading the same popular public data," that is a strong, concrete reason to keep REST for those endpoints even in a GraphQL shop — and several large GraphQL adopters do exactly that, running REST for cacheable public reads and GraphQL for the personalized, client-shaped surfaces.

### Query cost and depth limiting: the malicious deep query

This one is a security issue, not a performance nicety. Because the client controls the query shape and your schema has cycles in its *type* graph (`Order.customer.orders.0.customer.orders...`), a client can write a legal query of arbitrary depth and breadth:

```graphql
query Bomb {
  orders(first: 100) {
    edges { node {
      customer { orders(first: 100) {
        edges { node {
          customer { orders(first: 100) {
            edges { node { customer { email } } }
          } }
        } }
      } }
    } }
  }
}
```

Walk the multiplication: 100 orders, each with a customer who has 100 orders, each with a customer who has 100 orders. That is $100 \times 100 \times 100 = 10^6$ leaf objects requested in *one* tiny request, and a deeper nest is $100^d$ for depth $d$ — exponential in the depth the attacker chooses. Even with DataLoader batching the database calls, the engine still has to *materialize* a million objects in memory and serialize them. One unauthenticated request can pin a CPU and exhaust memory. **A single GraphQL endpoint that accepts arbitrary queries is, by default, a denial-of-service vector.** You must defend it.

The defenses stack:

- **Depth limiting** rejects any query nested deeper than a fixed bound (say 10 levels). Cheap to compute by walking the AST; kills the recursive nesting above outright.
- **Complexity / cost analysis** assigns each field a cost (a scalar is 1; a list field multiplies by its requested `first`), sums the query's total cost, and rejects anything over a budget (say 1000 points). This is more precise than depth alone because it accounts for breadth, not just depth. The math is a simple recursion: a list field's cost is its `first` argument times the cost of one element, and an element's cost is the sum of its selected fields' costs. So `orders(first: 50) { id customer { name } }` costs roughly $50 \times (1 + 1) = 100$ points — fifty orders, each with two scalar-ish fields. Nest one more list and the multipliers compound: `orders(first: 50) { customer { orders(first: 50) { id } } }` costs about $50 \times (50 \times 1) = 2500$, over a 1000-point budget, and is rejected at validation. Depth alone would have waved that query through because it is only three levels deep; cost analysis catches it because it sees the $50 \times 50$ breadth explosion. That is why mature APIs meter by cost, not depth or request count.
- **Paginated connections** via the Relay cursor spec. Forbid unbounded list fields entirely: every list takes a `first` (or `last`) argument, you cap it (e.g. `first` may not exceed 100), and you return a `Connection` with `edges`, `cursor`, and `pageInfo`. This is the GraphQL incarnation of the cursor-pagination discipline from [pagination trade-offs at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale): a stable opaque cursor, a bounded page size, no `OFFSET` drift.
- **Timeouts and rate limits** as the backstop — kill any query that runs past 5 seconds, and meter requests per client.

```graphql
# The Relay Connection pattern: bounded, cursor-based, no unbounded lists.
type OrderConnection {
  edges: [OrderEdge!]!
  pageInfo: PageInfo!
  totalCount: Int
}

type OrderEdge {
  cursor: String!     # opaque, stable position marker
  node: Order!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

![A vertical stack of request guards layering persisted queries, depth limit, complexity budget, rate limit, and timeout above query execution](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-7.png)

### Error handling: partial data and an errors array

REST uses the HTTP status code to say what happened: `404` not found, `403` forbidden, `422` unprocessable. GraphQL returns `200 OK` for almost everything (an HTTP `4x`/`5xx` from a GraphQL endpoint usually means the *transport* failed — bad JSON, gateway down — not that your data operation failed). The data outcome lives in the body, which can carry **both** `data` and `errors` at once — *partial success*:

```json
{
  "data": {
    "order": {
      "id": "9021",
      "totalCents": 4999,
      "customer": null
    }
  },
  "errors": [
    {
      "message": "Not authorized to read customer",
      "path": ["order", "customer"],
      "extensions": { "code": "FORBIDDEN" }
    }
  ]
}
```

The order resolved fine; the `customer` sub-field failed authorization, so `customer` is `null` and the `errors` array carries a structured entry with a `path` pointing exactly at the failed field and an `extensions.code` for machine handling. This partial-data model is genuinely nicer than REST's all-or-nothing for composite responses — one bad field does not blank the whole screen. But it forces clients to check `errors` *even on a `200*, and it interacts dangerously with non-null. If `customer` had been typed `customer: Customer!` (non-null) and the resolver errored, the null cannot sit in a non-null field, so **the error bubbles up to the nearest nullable ancestor and nulls *that* entire branch.** A misplaced `!` can turn a single forbidden field into a wholesale blanked-out `order`. This is why null-ability is a contract decision, not a default — design your `!`s knowing that non-null means "if this fails, take its parent down with it."

### Authorization is per field, not per route

In REST you can often guard a whole route: a middleware on `/admin/*` checks a role and you are done. GraphQL has one route, so route-level authz is meaningless — a single query can touch admin fields and public fields together. **Authorization in GraphQL is per field.** The `Order.customer` resolver must itself check whether the current viewer (from `context.user`) may see that customer; `Mutation.refundPayment` must check refund permissions; a field exposing internal cost data must check an admin scope. You can centralize this with directives (`@auth(requires: ADMIN)`) or a policy layer in `context`, but the discipline is the same one in [authorization with scopes and roles](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions): every field is a potential leak, so every sensitive field gets a check. Forgetting one is the classic GraphQL data-exposure bug — a field you never meant to be reachable is reachable because the schema is one big connected graph and someone wrote a query path to it.

### Versioning by deprecation, not v2

GraphQL's official stance is *no versioning*. Because the client requests exactly the fields it wants, you evolve the schema **additively**: adding a new field, a new type, or a new optional argument never breaks an existing client, because that client never asked for the new thing. This is the tolerant-reader principle from [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) made structural. When a field must go away, you mark it `@deprecated` with a reason, leave it working, and watch usage drain via your field-level metrics before you ever remove it:

```graphql
type Order {
  totalCents: Int!
  # Renaming "total" to "totalCents": keep the old field, deprecate it,
  # remove it only after field-usage metrics show zero callers.
  total: Int @deprecated(reason: "Use totalCents; this assumed dollars.")
}
```

There is no `/v2/graphql`. There is one evolving schema, fields are born and deprecated and (eventually, carefully) removed, and tooling (`graphql-inspector`, `buf`-style schema diffs in CI) flags any *breaking* change — removing a field, making an optional argument required, narrowing a return type — before it merges. This is one place GraphQL is genuinely cleaner than REST's URI-versioning sprawl.

### The quirks: uploads and rate limiting

Two smaller gotchas round out the list. **File uploads** are not in the GraphQL spec — JSON has no binary type — so the community uses the `multipart/form-data`-based GraphQL multipart request spec, or, more sanely, side-steps GraphQL entirely and uploads to a plain `POST` endpoint or a pre-signed URL, then passes the resulting id into a mutation. **Rate limiting** by request count is meaningless when one request can be a thousand times more expensive than another, so mature GraphQL APIs (GitHub's, Shopify's) rate-limit by *computed query cost* — you get a budget of points per minute and each query spends points proportional to its complexity. That is the same complexity score you computed for DoS defense, now doing double duty as a fairness meter.

## Federation and schema stitching at a glance

One more piece, because it is where GraphQL's "many clients, many backends" sweet spot actually pays off. In a microservices fleet you do not want one giant monolithic schema owned by one team. **Schema stitching** (the older approach) and **federation** (the modern one, from Apollo) let independent services each own a *subgraph* — Orders owns the `Order` type, Users owns `Customer`, Payments owns `Payment` — and a **gateway** composes them into one unified schema that clients query as if it were monolithic. The gateway parses the client query, plans which subgraph resolves which field, fetches from each, and stitches the result back together — resolving cross-subgraph references (an `Order` referencing a `Customer` that lives in the Users subgraph) via entity-resolution calls between the gateway and the owning subgraph.

![A federation graph where a gateway query planner fans a single client query out to the orders, users, and payments subgraphs and composes one response](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-8.png)

This is the architecture that makes GraphQL worth its costs at scale: clients get one typed graph over a whole fleet, each backend team owns its slice, and nobody coordinates a monolith. It is also where the N+1 problem reappears in a *new* shape — the gateway-to-subgraph "fetch entities by id" calls are themselves a batching problem, solved with the same batched entity resolution. The deeper distributed-systems view of composing services lives in [the API gateway and BFF pattern](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) and the schema-at-scale view in [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale); here the point is just that federation is the *reason* large orgs reach for GraphQL, and a single-client CRUD app needs none of it.

## GraphQL vs REST: the honest comparison

Now we can put the whole trade-off in one table. There is no universal winner — there is a fit between a paradigm's strengths and the *forces* of your problem, which is the framing of [choosing a paradigm by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force).

![A matrix comparing GraphQL and REST across fetching, caching, versioning, cost control, and learning curve showing each wins on a different axis](/imgs/blogs/graphql-the-query-language-schema-and-the-n-plus-one-trap-5.png)

| Force | GraphQL | REST |
| --- | --- | --- |
| **Fetching** | Client shapes the response; no over- or under-fetch; one round-trip for a composite screen | Server fixes the shape; over-fetch common; under-fetch forces waterfalls |
| **Caching** | No URL to cache; you build APQ, entity, and resolver caches yourself | `GET` + URL + `ETag` give you free HTTP/CDN caching |
| **Versioning** | Additive evolution; `@deprecated` fields; no `/v2` | URI/header/media-type versions, or additive-only by discipline |
| **Cost control** | One endpoint accepts any query; you MUST add depth/complexity/timeout guards | Each route is bounded by hand; DoS surface is smaller by default |
| **Errors** | `200` + partial `data` + `errors` array; non-null can blank a branch | Status code carries the outcome; all-or-nothing per request |
| **Authorization** | Per field; one route means no route-level guard | Often per route; coarser and easier to reason about |
| **Tooling / DX** | Introspection, typed schema, generated clients, one strongly typed contract | OpenAPI, mature HTTP tooling, every proxy understands it |
| **Learning curve** | Schema + resolvers + loaders + cost limits — steeper | HTTP verbs + status codes — most engineers already know it |

And the more focused over/under-fetch comparison, the problem GraphQL was actually built to solve:

| Problem | REST symptom | How GraphQL solves it |
| --- | --- | --- |
| **Over-fetching** | `GET /orders/9021` returns 40 fields; the screen needs 3 | Client selects only the 3 fields; payload shrinks accordingly |
| **Under-fetching** | Need customer + payment too; fire 2 more serial requests | One query nests `customer` and `payments`; one round-trip |
| **Multiple clients** | Each client wants a different shape; server forks endpoints or over-serves | Each client sends its own selection set against one schema |
| **Evolving fields** | Removing a field breaks every client that reads it | Add new fields freely; `@deprecate` old ones; clients opt in |

## Introspection and the developer-experience payoff

One genuine GraphQL strength deserves its own section, because it is the reason engineers who use a well-built GraphQL API tend to love it: the schema is **introspectable** at runtime. A GraphQL server answers a special meta-query — `__schema` and `__type` — that returns the entire type system: every type, every field, every argument, every description, every deprecation reason. This is not an add-on; it is in the spec, and it means the contract is *self-describing and machine-readable by default*, with no separate OpenAPI document to keep in sync.

```graphql
query IntrospectOrder {
  __type(name: "Order") {
    name
    fields {
      name
      description
      type { name kind }
      isDeprecated
      deprecationReason
    }
  }
}
```

Everything good about GraphQL tooling flows from introspection. The in-browser **GraphiQL** and **Apollo Sandbox** explorers autocomplete your fields as you type and show inline docs, because they introspected the schema. **Code generators** turn the schema plus your queries into fully typed client code — a TypeScript type for every query's exact response shape — so a misnamed field is a *compile error in the client*, not a runtime surprise. This is a level of contract-enforced developer experience that REST reaches only with a maintained OpenAPI spec and codegen on top, and even then the OpenAPI document can drift from the implementation. In GraphQL the schema *is* the implementation's type system, so it cannot lie. When this series says an API is a product whose users are other engineers (see [designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal)), introspection-driven tooling is GraphQL's strongest product feature.

The cost to weigh: introspection on a public API can leak your entire schema to anyone, including fields you forgot to lock down. Production public GraphQL APIs commonly *disable* introspection for untrusted callers and rely on persisted-query allowlists, trading a little DX for a smaller attack surface — the same field-level-authorization discipline applied to the schema itself.

## A problem-solving narrative: shipping the merchant dashboard safely

Let me walk the reasoning end to end the way it actually happens in a design review, because the individual rules only matter when they collide on a real screen. The task: ship a merchant dashboard that lists a store's recent orders, each with the customer's name and the latest payment's status, and lets the merchant issue a refund inline. The merchant can have tens of thousands of orders.

**Step 1 — shape the query.** The screen needs orders, and per order a customer and a payment. In REST this is the three-hop waterfall from the opening. In GraphQL it is one query with a nested selection set, so we start there: `orders(first: 25) { edges { node { id totalCents status customer { name } payments(first: 1) { edges { node { status } } } } } }`. One round-trip, exact fields. Good.

**Step 2 — stress-test the database cost.** That query, naively resolved, is a textbook N+1: 1 query for the 25 orders, 25 for the customers, 25 for the payments — 51 queries for one screen, growing linearly if we raise the page size. We wrap `Order.customer` in a customer loader and `Order.payments` in a payments-by-order loader. Now it is 1 + 1 + 1 = 3 queries regardless of page size, and the customer loader's dedup means a merchant with many repeat buyers costs even less. We add "every object-to-object edge resolver is loader-backed" to the PR checklist so the next field someone adds does not silently reintroduce N+1.

**Step 3 — stress-test the attacker.** The endpoint accepts any query, and our type graph has cycles (`order.customer.orders.customer...`). An attacker can write a nested bomb that asks for $100^d$ objects. So we add a depth limit (max 10) and a complexity budget (cost ≤ 1000, where list fields multiply by their `first`), both enforced in the validation phase before execution. We forbid unbounded lists: every list field requires a `first` and we cap it at 100. We add a 5-second execution timeout as a backstop. Only now is the endpoint safe to expose.

**Step 4 — stress-test authorization.** One route means no route-level guard. The merchant must see *only their own* orders, and must not read another merchant's customer. So `Query.orders` filters by the authenticated merchant from `context.user`, and `Order.customer` checks that the viewer owns the order before returning the customer — a per-field check, because a malicious query could try to walk from an order it can see to a customer it cannot. We centralize the policy in `context` so every resolver calls the same `can(user, "read", resource)`.

**Step 5 — stress-test the refund mutation.** The merchant clicks "refund." A flaky network could fire the mutation twice. So `refundPayment` takes an idempotency key in its input and the resolver records it, returning the original refund on a retry instead of refunding twice — the same safe-retry discipline as a payment. We return a `userErrors` list so "amount exceeds original charge" is a typed, recoverable error the UI can show, not a top-level protocol error that blanks the screen.

**Step 6 — stress-test evolution.** Next quarter, product wants to rename `totalCents` because some old client assumed dollars. We do not ship `/v2/graphql`. We add the new field, mark the old one `@deprecated`, watch field-usage metrics until the old field's callers hit zero, and only then remove it — with a schema-diff linter in CI blocking any accidental breaking change in the meantime.

Six steps, and notice that *five of them are rebuilding things REST gave us for free* — cost bounding, route-level authz, retry safety, caching strategy, all reconstructed at the field and schema level. That is the trade, stated as a checklist: GraphQL buys the client a perfect query shape, and the price is that the server re-earns every protection HTTP used to carry. For this dashboard — one rich screen, an internal client, a database that would have N+1'd — the trade is worth it. For a single-page CRUD app over three tables, it is not, and the next-to-last section says so plainly.

## Case studies: who runs GraphQL and why

**Facebook (the origin).** GraphQL was built inside Facebook around 2012 and open-sourced in 2015, born from exactly the mobile-waterfall pain in this post's opening: the News Feed mobile app was making too many round-trips over slow connections, each over-fetching or under-fetching, and the team wanted the client to declare its data needs once. DataLoader came from the same effort, which tells you the N+1 problem was understood as inseparable from GraphQL from day one — they shipped the disease and the cure together.

**GitHub (REST and GraphQL side by side).** GitHub built its **GraphQL API v4** alongside its REST v3, an unusually candid public statement that GraphQL is a *complement*, not a replacement. GitHub's GraphQL API is a textbook example of cost-based rate limiting: instead of counting requests, it computes a point cost per query (roughly, the number of nodes a query could return) and gives you a budget per hour, which is the only sane way to meter an endpoint where one query can be a thousand times heavier than another. It also leans hard on the Relay `Connection` pattern for every list.

**Shopify.** Shopify runs one of the largest public GraphQL APIs, the backbone of its app and storefront ecosystem, where the "many heterogeneous third-party clients" force is maximal — thousands of apps each needing a different slice of the same commerce data. Shopify popularized the mutation **`userErrors`** pattern this post used (recoverable business errors as a typed field, not as top-level protocol errors) and meters with a *calculated query cost* leaky-bucket system. It is the canonical example of GraphQL fitting its sweet spot: one schema, an army of clients with divergent needs.

**Netflix (federation).** Netflix runs a large **federated** GraphQL architecture (its open-source DGS framework plus a federation gateway), composing a unified graph over many backend domain services so that its studio and content-engineering UIs can query across the whole fleet through one typed surface while each backend team owns its subgraph. This is the federation story from the last section in production — the scenario where GraphQL's costs are clearly worth paying.

(Each of these is stated at the level I can vouch for — the paradigm choices and the named patterns are accurate; I have deliberately not invented specific internal latency or traffic numbers.)

## When to reach for GraphQL, and when not to

This is the heart of the honest take, so I will be blunt.

**Reach for GraphQL when:**

- **You have many heterogeneous clients** — web, iOS, Android, a partner integration, an internal admin tool — each needing a *different shape* of the same data. One schema serves all of them without forking endpoints or over-serving. This is the strongest single signal.
- **You are building an aggregation layer / BFF** (backend-for-frontend) over multiple backend services, and clients want one round-trip that stitches data from several of them. GraphQL's resolver model is purpose-built for fan-out aggregation, and federation scales it across teams.
- **Your screens are graph-shaped and round-trip-sensitive** — rich mobile UIs over slow links where the REST waterfall genuinely hurts, as in the opening trace.
- **Schema evolvability matters and you want field-level usage analytics** to deprecate safely. GraphQL's per-field metrics tell you exactly who reads what before you remove anything.

**Do not reach for GraphQL when:**

- **You have a single client doing simple resource CRUD.** This is the big one. For a single web app over a handful of tables, REST gives you free HTTP caching, free route-level authz, free per-route cost bounding, and a paradigm every engineer already knows — and GraphQL hands you the N+1 problem, the caching problem, and the query-cost problem in exchange for flexibility you will not use. You will spend your first month rebuilding caching and DoS protection that REST gave you for nothing. **Do not buy GraphQL's costs to solve a problem you do not have.**
- **You need aggressive edge / CDN caching of public read responses.** No stable URL means no free CDN cache; if your traffic is mostly identical public reads, REST's `GET`-plus-`ETag` model wins outright.
- **You are exposing a small, stable, public API to untrusted callers and cannot invest in cost-limiting infrastructure.** An unguarded GraphQL endpoint is a DoS vector; if you cannot build depth/complexity/cost limits, do not ship a public arbitrary-query surface.
- **The operation is fundamentally an action, not a data graph** — "charge this card," "rotate this key." Those are RPC-shaped and often clearer as a plain endpoint or, in a typed service mesh, as gRPC (covered in [gRPC and Protocol Buffers](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming)). GraphQL mutations can do it, but you gain nothing over a focused endpoint.

The meta-rule, the one this whole series keeps returning to: **choose the paradigm by the force of your problem, not by what is fashionable.** GraphQL is a precise tool for a real and common problem — many clients aggregating over many backends — and a liability when applied to a problem it was not built for.

## Key takeaways

- **GraphQL hands the response-shape decision to the client.** One typed query gets exactly the fields requested in one round-trip, solving REST's over-fetching (too many fields) and under-fetching (too many round-trips) at once.
- **The schema is the contract.** SDL types, non-null `!`, enums, interfaces/unions, and the `Query`/`Mutation`/`Subscription` roots define exactly which queries are legal; the engine validates every query against it before a resolver runs.
- **Every field is a resolver.** A query resolves as a tree walk, calling one function per field. Object-to-object edges on list fields are where the trouble starts.
- **N+1 is structural, not a one-off bug.** A list of $N$ parents plus a per-parent child resolver issues $N + 1$ queries. Derive it, then expect it on every list edge.
- **DataLoader turns $N+1$ into $1+1$ ($O(N)$ into $O(1)$).** It batches all `load()` calls in one event-loop tick into a single `WHERE id IN (...)` and caches per request. Build loaders per request; return batch results in input order. Treat loaders as mandatory in production.
- **GraphQL gives back problems HTTP solved for you.** No URL to cache (build APQ + entity + resolver caching), an arbitrary-query DoS surface (add depth + complexity + timeout limits + bounded connections), partial errors in a `200` body (check `errors`; design your non-nulls), and per-field authorization (guard every sensitive field).
- **Version by deprecation, not by `/v2`.** Evolve the schema additively; `@deprecated` fields drain via usage metrics; a schema-diff linter catches breaking changes in CI.
- **Fit beats fashion.** Many clients and aggregation/BFF — yes. Single-client CRUD or edge-cached public reads — no; you will pay GraphQL's costs for flexibility you do not need.

## Further reading

- **The GraphQL Specification** (spec.graphql.org) — the authoritative definition of the language, type system, and the execution algorithm, including the serial-mutation and parallel-query guarantees.
- **DataLoader** (github.com/graphql/dataloader) — the source and docs for the batching-and-caching utility; read the batch-function contract (length and order) carefully.
- **The Relay GraphQL Server Specification** — the `Node` interface, global object identification, and the `Connection`/`edges`/`pageInfo` cursor-pagination pattern referenced throughout this post.
- **Apollo's docs on Persisted Queries (APQ) and Federation** — the modern playbook for caching, cost control, and composing a federated graph across services.
- **GitHub GraphQL API documentation** — a public, production example of cost-based rate limiting and Relay connections you can study against the live schema via introspection.
- Within this series: the contract framing in [what an API is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the paradigm decision in [choosing a paradigm by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); the pagination discipline in [pagination trade-offs at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale); query-param safety in [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql); and the full review checklist in [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- For the layers below the API surface: [how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) for why the batched `IN` lookup is cheap, and [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) for federated-schema evolution across an org.

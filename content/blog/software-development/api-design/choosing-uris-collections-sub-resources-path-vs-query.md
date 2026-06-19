---
title: "Choosing URIs: Collections, Sub-resources, and the Path Query Split"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn the craft of URI design — the anatomy of a URI, collections versus items versus singletons, when to nest a sub-resource versus flatten it with a query, and the one rule that keeps a surface predictable for years: identity in the path, everything that narrows a set in the query."
tags:
  [
    "api-design",
    "api",
    "rest",
    "uri-design",
    "url-design",
    "http",
    "resource-modeling",
    "naming",
    "caching",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-1.png"
---

A payments platform I helped review had an endpoint that looked harmless: `GET /customers/{customer_id}/orders/{order_id}/items/{item_id}/refunds`. Five path segments, two of them redundant, and a support engineer who needed to answer one question — *which refunds happened on Tuesday?* — discovered there was no way to ask it. You could not list refunds. You could only navigate to them, one customer and one order and one line item at a time, because somebody had decided early on that a refund "belongs to" an item, which belongs to an order, which belongs to a customer, and had encoded that entire ownership chain into the path. The data was all there. The URIs made it unreachable. When the finance team asked for a daily refunds report, the answer was a three-week project to add `GET /refunds?created_after=...` — a top-level collection that should have existed from day one.

That is what bad URI design costs you, and it almost never shows up the day you ship it. The endpoint works in the demo. It works for the one access pattern the first client needed. Then a second client shows up wanting to slice the data a different way, and the path — which you cannot change without breaking everyone — turns out to have baked exactly one access pattern into the address of every resource. A URI is not a label you stick on a handler. It is a long-lived identifier that clients will hard-code, that proxies will cache by, that logs will index by, that a partner integration will pin to for three years. You get to design it once.

![A layered diagram showing the anatomy of a URI split into scheme, host, path, and query, with the path carrying identity and the query carrying filtering](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-1.png)

This post is the practical craft of choosing URIs, and it sits on one central principle that I will state now and spend the rest of the post earning: **put identity and hierarchy in the path; put filtering, sorting, pagination, and projection in the query.** Everything else — when to nest a sub-resource versus flatten it, how to name collections, why deep nesting hurts, why an id in the query is a smell and a filter in the path is a worse one, why you should pick one canonical URL and redirect the rest — follows from that split. We will work the whole post through the series' running **Payments and Orders** example for a fictional commerce platform (`/orders`, `/payments`, `/refunds`, line items, and a `/me` singleton), so you can see each rule land on real wire.

By the end you will be able to look at any resource a product person describes and write its URI without hesitating: decide whether it is a collection, an item, or a singleton; decide whether a relationship nests or flattens; name it so the next endpoint is guessable; and know exactly which parts of the request go in the path and which go in the query — and why getting that split wrong quietly destroys your cacheability, your reporting, and your ability to evolve. This is post B6 in the series, and it builds directly on the resource model from [Resource Modeling: Turning a Domain Into Nouns and URIs](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) and the contract framing from [What Is an API: The Contract Between Systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems).

A few terms before we start, because this series defines its jargon the first time it appears. A **URI** (Uniform Resource Identifier) is the full address that names a resource — `https://api.shop.example/v1/orders/o_91`. A **URL** is a URI that also tells you how to *locate* the thing (the scheme and host); for HTTP APIs the two words are interchangeable and I will use them that way. A **resource** is any thing worth naming and addressing: a single order, the collection of all orders, the current authenticated user. A **collection** is a resource that *is* a set of other resources (`/orders`). An **item** (or *member*) is one element of a collection (`/orders/o_91`). A **singleton** is a resource of which there is exactly one in its context, with no id in the path (`/me`). A **sub-resource** is a resource addressed *underneath* another resource's path (`/orders/o_91/items`). And the **path** and **query** are the two halves of a URI we are about to dissect. Keep those straight and the rest is mechanical.

## 1. The anatomy of a URI: location, identity, and selection

Start with the structure, because the rules only make sense once you can see the parts. A URI as defined by RFC 3986 has a handful of components, and for an HTTP API exactly four of them matter day to day. Take this one apart:

```http
https://api.shop.example/v1/orders/o_91?fields=id,total&expand=customer#summary
```

- **Scheme** — `https`. For any API you ship in this decade this is `https`, full stop. Plain `http` exists only to issue a redirect to `https`. The scheme tells the client which protocol and, critically, that the connection is encrypted; it is not something you "design" so much as a non-negotiable default.
- **Authority (host and optional port)** — `api.shop.example`. This locates the *server*. You do design this: whether your API lives at `api.shop.example`, at `shop.example/api`, or at `api.shop.example/v1` is a real decision (a dedicated subdomain keeps the API surface, its TLS certificate, its caching policy, and its routing separate from your marketing site, which is why most serious APIs use one).
- **Path** — `/v1/orders/o_91`. This is the heart of the matter. The path **identifies** the resource. Every segment is hierarchical: `/v1` is the version namespace, `/orders` is the collection, `o_91` is the item within it. The path answers the question *which thing do you mean?*
- **Query** — `?fields=id,total&expand=customer`. A set of `key=value` pairs after the `?`, joined by `&`. The query **selects, filters, and shapes**. It answers *which slice, in what order, projected how?* It never changes *which* resource you are talking to — it modifies what you get back about it (or, for a collection, which members you get).
- **Fragment** — `#summary`. Everything after the `#`. This is the one part that is **never sent to the server.** The browser or client keeps it and uses it to scroll to or address a piece of the returned representation. For an API it is almost always irrelevant; you should not design behavior that depends on it, because your server will never see it.

That is the whole anatomy, and the figure above lays it out as a stack from the most fixed part (scheme) to the most client-local part (fragment). The two segments doing real design work are the **path** and the **query**, and the single most important sentence in this entire post is about how to divide labor between them. I will say it plainly and then defend it for the next ten thousand words.

> **Identity and hierarchy go in the path. Filtering, sorting, pagination, and projection go in the query.**

Why is the boundary drawn exactly there, and not somewhere else? It is not arbitrary; it falls out of what each part *is for* in the architecture of the web.

The path is the resource's **identity** — its primary key in the address space of your API. RFC 9110 (HTTP semantics) and every cache, proxy, gateway, and CDN between your client and your server treat the path-plus-query as the **cache key** for a safe request, but they treat the *path* as the routing key: a gateway matches a request to a backend by its path, a rate limiter buckets by path template, your logs and metrics aggregate by path template (`/orders/{id}`, not `/orders/o_91`). The path is the stable, structural, hierarchical name. Things that go in the path are things that define *which resource exists at this address forever*.

The query is the resource's **selection mechanism.** A query parameter modifies the request without changing the resource's identity. `GET /orders` and `GET /orders?status=paid` address the *same collection resource*; the second just asks for a filtered view of it. That is why filters, sorts, page cursors, and field projections live there: they are all ways of saying "the same resource, but narrowed/ordered/windowed/trimmed." They are *views*, and views belong in the query. As a useful rule of thumb that I will sharpen later: **if removing a piece of the request would change which resource you mean, it belongs in the path; if removing it would just give you more or differently-shaped data about the same resource, it belongs in the query.**

Two smells follow immediately, and you should be able to spot both on sight:

1. **An id in the query is a smell.** `GET /orders?id=o_91` puts identity — the single most path-shaped thing there is — into the selection half. It is the unmistakable fingerprint of an RPC-over-HTTP design (more on that in §6) where every endpoint is `GET /getThing?id=...`. The id is identity; identity belongs in the path: `GET /orders/o_91`.
2. **A filter in the path is a worse smell.** `GET /orders/paid` looks tidy until you realize `paid` is a *value of a field*, not a resource. Now `paid`, `pending`, `cancelled`, and every future status is a hard-coded path. You cannot combine filters (`paid` AND `created last week`). You cannot add a status without shipping a new path. And `/orders/paid` collides ambiguously with `/orders/{id}` — is `paid` an order id or a status? The filter belongs in the query: `GET /orders?status=paid`.

There is one more reason the boundary sits where it does, and it is the one that matters most over a multi-year timeline: **the path is far harder to change than the query.** A path segment is identity, and identity is what clients hard-code, what proxies cache by, what your reverse-proxy routes by, and what a partner pins their integration to. Change a path and you have broken a contract; you need a redirect, a deprecation window, and a migration plan (we get to all three later in this post and in the versioning posts). A query parameter, by contrast, is *additive by nature*: you can add a new optional filter (`?currency=usd`) tomorrow and no existing client notices, because a client that does not send it gets the old, unfiltered behavior. So the split is also a **rate-of-change** split: the slow-moving, contractual part of a request goes in the slow-moving part of the URI (the path), and the fast-moving, experimental, client-specific part goes in the part you can extend freely (the query). When you find yourself wanting to put something in the path that you suspect you will want to change later, that hesitation is information — it probably belongs in the query.

The rest of this post is, in a real sense, just consequences of this one split. Let us go build the resources.

## 2. Collections, items, and singletons: the three shapes a URI takes

Almost every resource you address is one of three shapes, and knowing which shape you are designing tells you the URI immediately.

**A collection** is a resource that is a set of members. Its URI is a **plural noun**: `/orders`, `/payments`, `/refunds`, `/customers`. You `GET` a collection to list its members (with filters and pagination in the query) and you `POST` to a collection to create a new member. The collection has its own identity — `/orders` is a real, addressable thing, not just a prefix — which is exactly why pagination and filtering attach to it cleanly.

**An item** (a member of a collection) is a single element, addressed by appending its identifier to the collection: `/orders/o_91`, `/payments/p_7`. You `GET` an item to read it, `PUT`/`PATCH` to modify it, `DELETE` to remove it. The identifier is part of the path because it is *identity*. The item URI is the canonical home of that one resource for as long as it exists.

**A singleton** is a resource of which there is exactly one *in its context*, so it has no id segment. The classic example is the current authenticated user: `/me` (or `/user`, or `/account`). There is precisely one "current user" per request — the one your bearer token identifies — so an id would be redundant and, worse, a security hazard (you do not want `/users/{id}` doing double duty as "me" and "anyone"). Singletons also show up as sub-resources: `/orders/o_91/shipping_address` is a singleton if an order has exactly one shipping address. You `GET` and `PUT` a singleton; you usually cannot `POST` to it (there is nothing to create — it already exists) and often cannot `DELETE` it.

Here is the trio on the wire, in our Payments and Orders API. List a collection, filtered:

```http
GET /v1/orders?status=paid&limit=25 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: private, max-age=0, must-revalidate

{
  "object": "list",
  "data": [
    { "id": "o_91", "status": "paid", "total": "49.99", "currency": "usd" },
    { "id": "o_88", "status": "paid", "total": "12.00", "currency": "usd" }
  ],
  "has_more": true,
  "next_cursor": "eyJpZCI6Im9fODgifQ"
}
```

Read one item:

```http
GET /v1/orders/o_91 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
ETag: "W/v3-7f1c"
Cache-Control: private, max-age=0, must-revalidate

{ "id": "o_91", "status": "paid", "total": "49.99", "currency": "usd", "customer_id": "c_42" }
```

Read the singleton:

```http
GET /v1/me HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "id": "c_42", "email": "buyer@example.com", "default_currency": "usd" }
```

Notice the consistency: a client who learns the `orders` collection already knows how `payments` and `refunds` behave, because the *shape* — plural collection, item by id, filters in the query — is uniform across every noun. That uniformity is the entire payoff, and it is worth saying again because people forget it under deadline: the goal of URI design is **predictability**. A caller should be able to *guess* the next endpoint correctly without reading documentation. The three shapes give them the grammar to do that.

A subtle point about the collection-versus-item boundary that catches people: **the collection and the item are different resources with different methods, even though they share a prefix.** `POST /orders` means "create a new order *in this collection*" — the collection decides the id, so the client does not supply one. `PUT /orders/o_91` means "create-or-replace *this specific* order" — the client already knows the id. That difference is why `POST` goes to the collection (the server mints the id and returns it in `Location`) while `PUT` goes to the item (the client names the resource). It is also why you almost never `DELETE /orders` (deleting an entire collection is rarely a thing you want to expose) but routinely `DELETE /orders/o_91`. The URI structure — collection here, item there — is what lets the *same* method mean the right thing at each level. Get the structure right and the method semantics fall out; get it muddled (an id-less "create" that the client supplies an id for, say) and you end up fighting HTTP.

One more shape worth naming because it confuses newcomers: a **collection can itself be a sub-resource.** `/orders/o_91/items` is a collection (of line items) that lives under an item (`o_91`). So "collection," "item," and "singleton" are not levels in the path — they are *roles* a resource plays, and a single path can alternate roles segment by segment: `/orders` (collection) → `o_91` (item) → `items` (collection) → `li_3` (item). Reading a URI well means reading it as that alternation, which is exactly what the tree in the next section shows.

#### Worked example: choosing the shape for "the cart"

Suppose product asks for "a shopping cart." Is it a collection, an item, or a singleton? Walk it. Does a customer have *many* carts at once? On most commerce platforms, no — there is one active cart per customer. That makes the cart a **singleton** of the customer: `GET /v1/me/cart`, `PUT /v1/me/cart`. But the *things in the cart* are many, so the cart's contents are a **collection** sub-resource: `GET /v1/me/cart/items`, `POST /v1/me/cart/items`, `DELETE /v1/me/cart/items/ci_3`. One decision — "how many?" — produced the whole URI family. If, instead, you ran a B2B platform where a buyer can have several named carts ("Q3 reorder," "samples"), the cart becomes a **collection**: `/v1/carts`, `/v1/carts/cart_5`, `/v1/carts/cart_5/items`. The shape is not a matter of taste; it is determined by the cardinality in the domain. Ask "how many of these exist in this context?" and the shape answers itself: one-in-context → singleton, many → collection with items.

## 3. Sub-resources and the hierarchy: nest when you own the lifetime

Now we get to the question that breaks more APIs than any other: **when do you nest a resource under another, and when do you give it a top-level home?** The deep-nesting disaster from the intro lives here.

![A tree diagram of the URI hierarchy showing orders and payments collections, a me singleton, items beneath an order, and refunds beneath a payment as owned sub-resources](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-2.png)

The figure shows a clean hierarchy: collections at the top (`/orders`, `/payments`), the `/me` singleton beside them, items one level down (`/orders/o_91`), and sub-resources one level below that (`/orders/o_91/items`, `/payments/p_7/refunds`). Look at the depth: nesting stops at **one level below the item.** That is not an accident. It is the single most useful rule of thumb in URI design.

**The one-level-of-nesting rule:** a URI should rarely go deeper than `/collection/{id}/sub-collection` (or `/collection/{id}/sub-collection/{sub-id}`). Past one level of nesting, you almost always want to flatten the deeper resource into a top-level collection and express the relationship with a query parameter or a link in the body.

Why one level, and why does deep nesting hurt so much? Three concrete reasons, each of which is a real production failure I have watched happen.

**First, deep paths encode exactly one access pattern and forbid all others.** The refunds disaster from the intro — `/customers/{c}/orders/{o}/items/{i}/refunds` — means the *only* way to reach a refund is to know its customer, its order, and its item. The finance team that wanted all of Tuesday's refunds had no path to ask. A top-level `/refunds` collection supports every access pattern: `/refunds?created_after=...`, `/refunds?payment_id=p_7`, `/refunds?status=failed`. Deep nesting is a premature commitment to one navigation route, made permanent by the fact that you cannot change a path without breaking clients.

**Second, deep paths carry redundant identity that can desynchronize.** In `/customers/c_42/orders/o_91/items/li_3`, the `c_42` is redundant — `o_91` already determines its customer. Now the server must either ignore `c_42` (in which case why is it in the path?) or *validate* that `o_91` really belongs to `c_42` and return a confusing error if a client passes a stale or wrong customer. You have invented a way for two parts of the same URI to disagree. Worse, if an order can ever be reassigned to a different customer, every cached URI for its items is now wrong.

**Third, deep paths are simply harder to use and to read in logs.** Five segments is a lot to construct correctly, and your metrics dashboard now has path templates five levels deep that are awkward to group.

So the rule is: **nest only when the child genuinely cannot exist without the parent — when the parent owns the child's lifetime — and even then, stop at one level.** Otherwise flatten.

![A before and after diagram contrasting a four-level deeply nested path against a flat top-level collection addressed with a query filter](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-3.png)

How do you tell "owns the lifetime" from "merely related"? Apply two tests, shown side by side in the next figure.

![A decision matrix comparing nesting as a sub-resource against flattening with a query across ownership of lifetime, whether the child is queried alone, and how many parents it has](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-7.png)

**Test 1 — does the child die with the parent?** A line item has no meaning without its order. Delete the order, the line items go with it. They are never created, listed, or referenced independently — you always have an order in hand when you touch them. That is genuine ownership, and it earns a sub-resource: `/orders/o_91/items`. A *payment*, by contrast, has an independent lifetime: it is audited on its own, it appears in financial reports, it can in principle be queried without any order context, and a refund or dispute references it directly. So a payment is a **top-level resource** (`/payments/p_7`) that *links* to its order via a field (`"order_id": "o_91"`), even though the order "has" a payment in plain English. "Has-a" in the domain does not mean "nested" in the URI.

**Test 2 — will anyone ever query this child on its own?** The refunds report is the tell. If a stakeholder will ever ask "show me all the X across the system, filtered some way," then X needs a top-level collection so the filter has a home. Refunds, payments, and orders all pass this test (finance wants cross-cutting reports). Line items fail it (nobody asks for "all line items across all orders" — and if they did, you would expose `/order-items?...` as a reporting view, not as the primary home). When in doubt, lean toward a top-level collection with a query, because **you can always add a convenience sub-resource later, but you cannot remove a path clients already depend on.**

Here is the nuance that confuses people: you can have **both**, as long as one is canonical. It is perfectly fine to expose `/orders/o_91/items` (the natural nested read for "give me this order's lines") *and* understand internally that line items are owned by the order. What you should not do is expose a *four-level* path. And for an independent entity like payments, the top-level `/payments/p_7` is canonical and `/orders/o_91/payment` (if it exists at all) is a convenience that should `301`-redirect or simply return the same representation.

Let me show the flatten-with-a-query pattern on the wire, because it is the workhorse. The order's line items as an owned sub-resource:

```http
GET /v1/orders/o_91/items HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "object": "list",
  "data": [
    { "id": "li_1", "sku": "BOOK-1", "qty": 2, "unit_price": "12.00" },
    { "id": "li_2", "sku": "MUG-7", "qty": 1, "unit_price": "8.99" }
  ]
}
```

And the same relationship, but for refunds — which we flatten to top-level so finance can slice them — expressed as a filter:

```http
GET /v1/refunds?payment_id=p_7&created_after=2026-06-15 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "object": "list",
  "data": [
    { "id": "re_3", "payment_id": "p_7", "amount": "10.00", "status": "succeeded" }
  ],
  "has_more": false
}
```

The first is navigation (I have an order, give me its lines). The second is query (I have a question about refunds across the system). Both are valid; the difference is whether the relationship lives in the **path** (ownership) or the **query** (a filter on an independent collection). For the deeper distributed-systems view of how these access patterns map onto a service fleet, the [system-design treatment of REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) is the companion read; here we are owning the URL surface itself.

## 4. The path/query split, sharpened: responsibilities and worked cases

We have stated the split. Now let us make it precise enough to apply mechanically, because the gray-area cases are where people slip.

![A matrix mapping identity, filtering, sorting, pagination, and projection onto the path column or the query column to show the division of responsibility](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-4.png)

The matrix is the whole rule in one picture. Read it row by row:

| Request concern | Belongs in | Why | Example |
| --- | --- | --- | --- |
| **Identity** (which resource) | **Path** | It defines the address; it is the cache/routing key | `/orders/o_91` |
| **Hierarchy / ownership** | **Path** | Structural containment is identity | `/orders/o_91/items` |
| **Filtering** (which members) | **Query** | A view over the same collection resource | `?status=paid&currency=usd` |
| **Sorting** (what order) | **Query** | Ordering is a view, not identity | `?sort=-created_at` |
| **Pagination** (which window) | **Query** | A window over the same set | `?cursor=eyJ&limit=50` |
| **Projection / sparse fields** (which fields) | **Query** | Shape of the representation, not identity | `?fields=id,total` |
| **Expansion** (inline related data) | **Query** | A view choice, not a different resource | `?expand=customer` |
| **Format / version negotiation** | **Header** (usually) | Content negotiation; sometimes query as a fallback | `Accept: application/json` |

This is the path/query split as a working tool. Identity and structural ownership go left, into the path. Everything that takes the *same* resource and narrows it, orders it, windows it, or trims it goes right, into the query. The detailed treatment of the right-hand column — how to design filters, sorts, and sparse fieldsets without accidentally reinventing SQL on your URL — is its own post: [Filtering, Sorting, and Sparse Fieldsets Without Reinventing SQL](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql). And the pagination row deserves its own deep dive too, because the choice of offset versus cursor versus keyset has real correctness consequences under writes: [Pagination: Offset, Cursor, and Keyset Tradeoffs at Scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale). Here I only want to establish *where they go*: in the query, always.

There is a clean test for any ambiguous case. Ask: **if I remove this part of the request, do I get a different resource, or the same resource shaped differently?**

- Remove `o_91` from `/orders/o_91?fields=id` and you get `/orders?fields=id` — a *different resource* (the whole collection instead of one order). So `o_91` is identity → path. ✓
- Remove `fields=id` from `/orders/o_91?fields=id` and you get `/orders/o_91` — the *same order*, just with all fields. So `fields` is a view → query. ✓
- Remove `status=paid` from `/orders?status=paid` and you get `/orders` — the *same collection*, unfiltered. So `status` is a filter → query. ✓

The test never fails, because it is just the definition of "identity" restated operationally.

#### Worked example: a filter that wants to be a path (and why you must resist)

A team building the orders API was asked to add "let clients fetch only their own orders." The first proposal was `/my-orders` — a whole new collection. The second was `/orders/mine`. Both are tempting and both are wrong, and the reason is instructive.

`/orders/mine` collides with `/orders/{id}`: the router cannot tell whether `mine` is an order id or a magic word, so you need a special case in the path matcher — fragile and surprising. `/my-orders` forks the surface: now there are two collections of orders, `/orders` and `/my-orders`, that drift apart (a filter added to one is forgotten on the other), and a client must learn both. The correct answer is that "mine" is a **filter**, and it belongs in the query — except that in this case the filter value comes from the *authenticated identity*, so the cleanest design is to scope it implicitly: `GET /orders` returns *the caller's* orders by default (because authorization already knows who they are), and an admin token can pass `GET /orders?customer_id=c_42` to scope to someone else. One collection, one set of filters, identity-driven scoping handled by authorization rather than by inventing a path. The path stayed a clean noun; the "which orders" question went where it belongs — into the selection layer, here driven by the token.

The stress test for this design: what happens when a client retries `GET /orders` on a flaky link? Because `GET` is **safe** (no side effects) and **idempotent** (same result every time), the retry is free — no risk of duplication, no special handling. That safety is *a consequence of putting the right things in the right place*: identity in the path keeps the URL stable enough to cache and retry, and filters in the query keep the resource the same across calls. Method semantics and idempotency get their own full treatment in the sibling post [Methods and Idempotency: GET, POST, PUT, PATCH, DELETE](/blog/software-development/api-design/methods-and-idempotency-get-post-put-patch-delete).

#### Worked example: where does a date-range "report" live?

Finance asks for a "daily refunds report" — all refunds between two dates, sorted by amount, with only the id, amount, and status fields. New engineers reach for a new endpoint: `GET /reports/refunds/2026-06-15`. Resist it, and walk the split instead. *Which resource* are we talking about? Refunds — that is `/refunds`, a collection we already have. *What are the dates?* A filter on that collection: `?created_after=2026-06-15&created_before=2026-06-16`. *The sort?* A view: `?sort=-amount`. *Only three fields?* Projection: `?fields=id,amount,status`. Put it together and the "report" is not a new resource at all — it is the refunds collection, sliced:

```http
GET /v1/refunds?created_after=2026-06-15&created_before=2026-06-16&sort=-amount&fields=id,amount,status&limit=100 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "object": "list",
  "data": [
    { "id": "re_9", "amount": "120.00", "status": "succeeded" },
    { "id": "re_3", "amount": "10.00", "status": "succeeded" }
  ],
  "has_more": false
}
```

No `/reports` namespace, no per-date path, no new handler — just the existing collection with four query parameters, each doing exactly the job §1 assigned it. Tomorrow finance wants weekly granularity, or to also see failed refunds, or to add the `currency` field; every one of those is *another query parameter on the same URL*, not another endpoint. That is the dividend of the split: a single, well-modeled collection absorbs an unbounded number of "reports" without growing the surface a client must learn. Compare the alternative — a `/reports/refunds/daily`, `/reports/refunds/weekly`, `/reports/refunds/failed` sprawl — and you can feel the path/query split doing real work.

## 5. Naming conventions: making the next endpoint guessable

The path/query split tells you *where* things go. Naming conventions tell you *how to spell them* so the surface is consistent. None of these rules is cosmic truth; the value is in **picking one and applying it everywhere**, because consistency is what makes the next endpoint guessable. Here is the set I recommend, with the reasoning.

**Use plural nouns for collections.** `/orders`, not `/order`. The collection *is* a set, so the plural reads correctly: `GET /orders` ("get the orders"), `GET /orders/o_91` ("get order o_91 from the orders"). Mixing singular and plural (`/order` here, `/payments` there) is the most common inconsistency I see, and it destroys guessability — the client can no longer predict whether the next collection is singular or plural. Pick plural and never deviate. (The rare exception is a true singleton like `/me`, which is singular *because there is exactly one*.)

**Use nouns, not verbs.** `/orders`, not `/getOrders` or `/createOrder`. The HTTP method is the verb; the path is the noun. We will spend the next section on why, because it is the difference between REST and RPC-in-disguise.

**Pick one word-separator convention and one casing.** For multi-word path segments, the two live options are `kebab-case` (`/order-items`, `/payment-methods`) and a single concatenated word where natural. Stripe uses `payment_methods`-style snake_case in field names but keeps paths lowercase; Google's AIP guidelines recommend `camelCase`-free, lowercase resource names. The web's overwhelming convention for *paths* is **lowercase with hyphens** (`/payment-methods`), reserving `snake_case` or `camelCase` for *JSON field names* in the body. The one rule that actually matters: be consistent, and prefer lowercase, because **paths are case-sensitive in the URI spec** (`/Orders` and `/orders` are technically different resources) and mixed case invites bugs where a client lowercases a path and gets a `404`.

**No trailing-slash inconsistency.** Decide whether `/orders` and `/orders/` are the same resource (they should be) and enforce it — either normalize by stripping the trailing slash, or `301`-redirect one form to the other. What you must not do is let `/orders` return data and `/orders/` return `404` (or, worse, a different result). A trailing-slash mismatch is a classic source of "works on my machine" client bugs and split cache entries.

**No file extensions in the path.** `/orders/o_91`, not `/orders/o_91.json`. The representation format is negotiated with the `Accept` header (`Accept: application/json`), not encoded in the URL. Putting `.json` in the path conflates the resource with one of its representations, makes content negotiation impossible, and means adding a CSV export later requires a whole new URL family (`.csv`) instead of a new `Accept` value. (Some APIs do offer a `?format=csv` query *fallback* for browsers that cannot set headers — that is acceptable as a fallback because format is a *view*, but the header is canonical.) The full treatment of content negotiation is the sibling post [Content Negotiation: Media Types and Representations](/blog/software-development/api-design/content-negotiation-media-types-and-representations).

**Use stable, opaque ids — not natural keys — in the path.** This one is subtle and important. An item's id is part of its permanent address, so it must be **stable** (never reused, never changed) and ideally **opaque** (carries no meaning a client can parse). Compare:

- `/orders/o_91` — opaque. The client treats `o_91` as a black box. If you later change how ids are generated, shard the database, or migrate systems, the *format* of the opaque id can change without any client noticing, because no client ever parsed it.
- `/orders/2026-001-acme` — a natural key encoding year, sequence, and customer. The moment a client *parses* this (and they will — somebody always writes a regex), the structure becomes part of your contract. Change the format and you break them. And natural keys collide and mutate: a customer renames from "acme" to "acme-corp," an order is reassigned, two systems both want sequence `001`.

The defensive design is an opaque, prefixed id: `o_` for orders, `p_` for payments, `re_` for refunds (Stripe popularized this `ch_`, `cus_`, `pi_` convention). The prefix lets *you* and your logs tell at a glance what kind of object an id refers to, while the random suffix stays opaque to clients. A good shape is a typed prefix plus an unguessable, collision-resistant suffix (a base62-encoded ULID or UUID). If you have $n$ existing ids and a suffix drawn from a space of size $N$, the probability of a collision on the next insert is roughly $n/N$; with a 128-bit space that number is so small you can treat opaque-random ids as collision-free in practice, which is exactly why you should not hand-roll short sequential ids that *will* collide across shards.

A small worked classification of names, applying all the rules at once:

| Bad | Why it's bad | Good |
| --- | --- | --- |
| `/getOrders` | verb in path | `GET /orders` |
| `/order/91` | singular collection, numeric natural key | `GET /orders/o_91` |
| `/orders/91.json` | file extension; numeric id | `GET /orders/o_91` + `Accept` |
| `/Orders/Paid` | mixed case; filter as path | `GET /orders?status=paid` |
| `/orders?id=o_91` | identity in query | `GET /orders/o_91` |
| `/customers/c_42/orders/o_91/items/li_3/refunds` | five levels; redundant ids | `GET /refunds?payment_id=p_7` |

Every "good" form is shorter, more predictable, and more cacheable than its "bad" twin, and every one is just the path/query split plus the naming rules applied honestly.

## 6. Verbs versus nouns: why the path is a thing, not an action

I have leaned on "use nouns, not verbs" twice now. It deserves its own section, because it is the rule people break first and regret longest, and because the *why* is more interesting than the rule.

![A before and after diagram contrasting verb-in-path endpoints and an id hidden in the query against noun collections acted on by uniform HTTP methods](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-5.png)

The before column is the RPC-over-HTTP smell: `POST /createOrder`, `GET /getOrderById?id=o_91`, `POST /cancelOrder`. Each endpoint is a named procedure. The after column is REST: `POST /orders`, `GET /orders/o_91`, `DELETE /orders/o_91`. The method *is* the verb. Here is the argument for why the after column wins, made rigorous rather than asserted.

Suppose your API exposes $N$ kinds of things (orders, payments, customers, refunds) and supports $M$ operations on each (create, read, update, delete, list). In the verb-in-path style, every (thing, operation) pair is a distinct, separately-named endpoint, so the surface a client must learn grows as $O(N \times M)$ — `createOrder`, `cancelOrder`, `getOrder`, `listOrders`, `createPayment`, `refundPayment`, and on and on, each with its own ad-hoc URL and shape. Nothing about `createOrder` lets a client predict `cancelOrder`; they must memorize all $N \times M$ names.

In the noun style, the client learns the $M$ methods **once** — they are HTTP's, not yours — and each new resource costs only its noun. The surface collapses from $O(N \times M)$ to $O(N + M)$: $N$ nouns plus $M$ verbs the client already knows. When you add `shipments` next quarter, a client who has used `orders` already knows `GET /shipments`, `POST /shipments`, `GET /shipments/{id}` without reading a line of new docs. That predictability is the entire payoff, and it is a measurable reduction in what a caller must hold in their head.

There is a second payoff that the verb style throws away, and it is the one that connects URI design to caching and retries: **the HTTP method carries machine-legible promises that a custom verb cannot.** A `GET` is *safe* (no side effects) and *idempotent* (repeatable), so every proxy, gateway, CDN, and browser prefetcher on the path knows it can cache the response and a client can retry it freely. When you write `POST /chargeCard`, the `POST` still correctly signals "not safe, do not retry blindly," but you have buried any read semantics behind a verb the infrastructure cannot interpret. When you instead write `GET /charges/ch_1`, the entire web between client and server *knows* the request is a cacheable read — for free, without parsing your custom verb. Nouns plus standard methods make your URLs legible to the whole stack; verbs in the path reduce HTTP to a dumb tunnel and forfeit caching, conditional requests, and safe retries.

**The honest exception: actions that are not nouns.** Sometimes an operation genuinely does not map to creating or updating a resource — "capture this authorized payment," "cancel this order," "resend this receipt." You have two principled options, and a third that is occasionally pragmatic:

1. **Model the action as a resource.** "Capture" creates a *capture*; "refund" creates a *refund*. `POST /payments/p_7/refunds` is a `POST` that creates a new refund sub-resource — pure REST, no verb in the path, and it gives you an addressable record of the action (`/refunds/re_3`) for free. This is the best option when the action produces a thing worth recording, which is most financial actions.
2. **Model the action as a state transition via the field that changes.** Cancelling an order is just setting `status` to `cancelled`: `PATCH /orders/o_91` with `{ "status": "cancelled" }`. No new path at all.
3. **The pragmatic custom action (`POST /orders/o_91:cancel` or `POST /orders/o_91/cancel`).** When the action truly has no resource and no clean field (a side-effecting "resend the webhook"), a verb sub-path is acceptable *as long as it is a `POST`* (it is not safe, so `POST` is honest) and *clearly subordinate to a noun*. Google's AIP guidelines formalize this as the **custom method** pattern with a colon (`:cancel`). Use it sparingly; every custom action you add is a small surrender of uniformity.

The discipline is: reach for option 1 or 2 first, and only fall to option 3 when the action genuinely resists being a noun or a field. "I have an action" is almost never a reason to abandon nouns — it is usually a missing noun (a *refund*, a *capture*, a *cancellation*) you have not named yet.

## 7. How the URI drives routing, caching, and retries

The path/query split is not only about human predictability. It is about what the *machines* between your client and your server can do, and this is where URI design pays for itself in latency and reliability.

![A branching diagram showing a request routed by its path through a gateway to either an edge cache on a safe GET or to the orders handler, both producing the response](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-6.png)

Follow a request through the stack. The client sends `GET /orders/o_91`. The gateway matches the *path* against its route table — `/orders/{id}` routes to the orders service — without ever looking at the body. A rate limiter buckets by the path *template* (`/orders/{id}`), not the concrete id, so one customer hammering `o_91` and another hitting `o_88` share a sensible quota policy. Because the method is `GET` (safe), an edge cache or CDN can serve a fresh-enough cached copy, or revalidate cheaply with the `ETag`. On a miss, the request reaches the handler, which extracts the identity *from the path* and looks up the resource. The response carries an `ETag` and `Cache-Control`, so the *next* request can be a conditional `GET` that returns `304 Not Modified` with an empty body. None of this works if you hid the identity in the query as `?id=o_91` and shoved a verb in the path; the gateway, the limiter, and the cache would all be staring at an opaque procedure call. This routing-and-caching view connects directly to the gateway responsibilities covered in [the API gateway and backend-for-frontend pattern](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend).

Here is the principle stated as a rule you can verify: **the cache key for a safe HTTP request is the method plus the full URI (path and query) plus the `Vary` headers.** Two consequences fall out immediately, and both are URI-design decisions.

**Consequence 1: query-parameter order and redundancy fragment your cache.** `GET /orders?status=paid&limit=25` and `GET /orders?limit=25&status=paid` are, to a naive cache, two *different* keys — same resource, two cache entries, half the hit rate. So is `GET /orders?status=paid` versus `GET /orders?status=paid&` (a stray trailing `&`). The defensive design is to **normalize query parameters** (sort them, drop empties, canonicalize defaults) before they reach the cache, and to document a canonical form. A subtler version: if `limit` defaults to 25, then `?limit=25` and the bare `/orders` should be the *same* cache entry — so normalize the default away. Each unnecessary distinction halves a cache's effectiveness, and a cache's whole job is to multiply your effective capacity.

**Consequence 2: a stable path makes a `GET` safely retryable.** Because `GET /orders/o_91` is safe and idempotent, a client whose request times out on a flaky mobile link can retry it with zero risk — no duplicate, no surprise. That safety is *load-bearing* for reliability: it is why a mobile SDK can transparently retry reads, why a CDN can prefetch, why a health check can poll. Put identity in the query and add a verb, and you have not changed the *semantics* (a `GET` is still safe) but you have made the URL harder to cache and you have signaled nothing useful to intermediaries.

#### Worked example: the conditional GET that saves a payload

A mobile client polls an order's status every few seconds while a payment processes. Naively, that is a `200 OK` with the full order body every poll — say a 1.2 KB JSON document. Over a slow link, 1.2 KB is small but not free; the round-trip and parse cost add up across thousands of devices. Design the URI right and you get conditional requests for free. First poll:

```http
GET /v1/orders/o_91 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
ETag: "W/v3-7f1c"
Cache-Control: private, max-age=0, must-revalidate
Content-Type: application/json

{ "id": "o_91", "status": "pending", "total": "49.99", "currency": "usd" }
```

The client stores the `ETag` and on the next poll sends it back:

```http
GET /v1/orders/o_91 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
If-None-Match: "W/v3-7f1c"
```

```http
HTTP/1.1 304 Not Modified
ETag: "W/v3-7f1c"
Cache-Control: private, max-age=0, must-revalidate
```

A `304` with an empty body — the server confirmed "nothing changed" in a handful of bytes instead of resending 1.2 KB. Multiply by every poll across every device and the saving is real. And this only works because the order has a **stable, canonical URI** (`/orders/o_91`) that the `ETag` is anchored to. The full caching mechanics are the subject of the later post [Caching: ETags, Cache-Control, Conditional Requests, Invalidation](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation); the point here is that *the URI design is the precondition for all of it.* If the database-side cost of these lookups matters to you, the [B-trees post on how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) covers why an opaque-id primary-key lookup is cheap.

## 8. Matrix parameters and other URI features you should avoid

The URI spec offers a few features beyond path and query, and almost all of them are traps. Knowing they exist — and why to skip them — is part of the craft.

**Matrix parameters** are the big one. RFC 3986 permits `;key=value` parameters attached to a *single path segment*: `/orders;status=paid/o_91`. The idea, floated by Tim Berners-Lee, was to attach parameters to a specific segment rather than the whole URI (useful, in theory, for parameterizing intermediate segments). In practice, **avoid matrix parameters entirely.** The reasons are decisive:

- **Tooling support is inconsistent.** Many HTTP clients, proxies, and frameworks do not parse `;`-parameters correctly; some treat the whole thing as one literal path segment. You are relying on a corner of the spec that the ecosystem never fully implemented.
- **They confuse the cache and the router.** A gateway routing by path now has to understand matrix syntax to match `/orders;status=paid`, and a cache may or may not normalize them. You have created exactly the routing-and-caching headaches the path/query split exists to avoid.
- **They solve a problem the query already solves.** Anything you would express with a matrix parameter — filtering a collection — the query string handles better, with universal tooling support and clean caching. `/orders?status=paid` beats `/orders;status=paid` on every axis.

The only honest niche for matrix parameters is parameterizing an *intermediate* segment in a multi-segment path (e.g. a map-tile service encoding projection per segment), and even there the modern answer is "use a query and flatten." For an API, never use them.

A few more URI features to handle with care:

- **Encoded slashes (`%2F`) in path segments.** If a resource's id can contain a slash, encoding it as `%2F` is technically legal but many servers and proxies reject or normalize it inconsistently for security reasons (path-traversal defenses). Prefer ids that never contain reserved characters — another argument for opaque, base62 ids.
- **Reserved and unsafe characters.** Keep path segments to unreserved characters (letters, digits, `-`, `.`, `_`, `~`). Anything else must be percent-encoded, and percent-encoding in a path is a frequent source of double-encoding bugs. Opaque ids sidestep the whole class.
- **Very long query strings.** Servers and proxies cap URL length (commonly 8 KB, sometimes as low as 2 KB at a CDN). A filter like `?ids=o_1,o_2,...,o_5000` will be silently truncated or rejected. When a "query" needs to be that big, it is a sign you want a `POST` with a body (a search endpoint) — but note the trade-off: a `POST` is not cacheable. Some APIs offer `POST /orders/search` precisely for queries too large or too sensitive for a URL; that is a deliberate, documented exception to "reads are GETs," not a default.
- **Sensitive data in the query.** The path and query end up in server logs, proxy logs, browser history, and `Referer` headers. Never put a secret, a token, or PII in the URL. Tokens go in the `Authorization` header; sensitive search criteria may justify a `POST` body.

There is a concrete failure I have watched bite teams on the encoded-slash point, and it is worth spelling out because it looks like a server bug when it is really a URI-design bug. A team used file paths as resource ids — `/files/reports%2F2026%2Fq3.pdf` — reasoning that the id "is" a path. The endpoint worked in local tests against the dev server, then `404`'d in production behind a reverse proxy that, by default, rejected encoded slashes as a path-traversal defense (a sensible default, since `%2F` is exactly how attackers try to sneak `../` past path checks). The "fix" was to flip a proxy flag to allow encoded slashes — which is to say, to *weaken a security control* to accommodate a bad id choice. The real fix was an opaque id (`/files/f_8ak2`) with the human-readable path kept as a *field in the body*, not in the URL. The lesson generalizes: anything you put in a path segment must survive every proxy, gateway, and normalizer between the client and your server, and the safest way to guarantee that is to keep path segments to opaque, unreserved-character ids and push everything human-readable into the body or the query.

The throughline: the path/query split is not just a guideline, it is the *only* part of the URI that the whole ecosystem agrees on and handles well. Stick to it, and avoid the spec's exotic corners.

## 9. Canonical URLs, aliases, and idempotent reads

A resource should have **one canonical URL** — one address that is *the* address — even when convenience or history gives it several. Multiple URLs for the same resource fragment your caches (each alias is a separate cache key), confuse clients about which to store, and split your analytics. Pick one and make the rest redirect.

![A timeline showing an alias request receiving a 301 redirect to the canonical URI, which then validates with an ETag and serves a 304 from one shared cache key](/imgs/blogs/choosing-uris-collections-sub-resources-path-vs-query-8.png)

The figure walks the lifecycle. A client hits an alias — say the old, awkward `/orders/?id=o_91`, or a convenience nested path `/customers/c_42/orders/o_91` — and the server responds `301 Moved Permanently` (or `308`, which preserves the method) with a `Location` header pointing at the canonical `/orders/o_91`. The client follows the redirect, hits the canonical URL, and from then on everything — the `ETag`, the cache entry, the conditional `GET` — converges on **one shared key.** Aliases that `301` to a canonical form let you offer convenient or legacy URLs without fragmenting your cache, because every cache entry, validation, and `304` lands on the same canonical address.

```http
GET /v1/customers/c_42/orders/o_91 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 301 Moved Permanently
Location: /v1/orders/o_91
Cache-Control: public, max-age=86400
```

The `301` itself is cacheable (`max-age=86400`), so well-behaved clients and proxies learn the canonical form and stop hitting the alias entirely. This is how you migrate an API from a bad URL shape to a good one *without a breaking change*: ship the new canonical URL, redirect the old one, give clients a `Deprecation` window, and let the redirects carry the traffic over. (Deprecation and `Sunset` headers are the subject of a later post; the relevant point here is that canonical URLs plus redirects are a *tool for safe URL evolution.*)

Where do you tell clients the canonical URL? Two good places, and you should use both:

1. **A `Location` header** on the `201 Created` response when a resource is born. When a client `POST`s to `/orders` and you create `o_91`, return `201 Created` with `Location: /v1/orders/o_91`. The client now knows the canonical address of the thing it just made.
2. **A `self` link in the body.** Many APIs include `"url": "/v1/orders/o_91"` or a `_links.self` field in every resource representation, so a client that received an order in a list always knows its canonical address without constructing it by hand. This is the lightweight, useful end of HATEOAS (the hypermedia approach where responses carry links to related actions); the heavier end gets its own honest treatment in [HATEOAS in the Real World](/blog/software-development/api-design/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip).

Here is the creation flow on the wire, with the canonical URL handed back in the `Location` header:

```http
POST /v1/orders HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: tok_example

{ "customer_id": "c_42", "items": [{ "sku": "BOOK-1", "qty": 2 }] }
```

```http
HTTP/1.1 201 Created
Location: /v1/orders/o_91
Content-Type: application/json

{ "id": "o_91", "status": "pending", "url": "/v1/orders/o_91", "total": "24.00", "currency": "usd" }
```

The `Idempotency-Key` there is a placeholder pointing forward to a later post — its job is to make the *create* safely retryable even though `POST` is not naturally idempotent, but that is a story for [Idempotency Keys: Safe Retries and Exactly-Once Illusions](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions). The URI-design point is the `Location` header: the server, not the client, declares the canonical address.

## 10. Case studies: how the careful APIs choose URIs

It helps to see the principles in the wild, so here are three widely-used APIs and the URI conventions they actually follow. I am keeping these to claims I am confident are accurate; where a detail is fuzzy I describe the pattern rather than invent specifics.

**Stripe** is the reference example for clean, opaque, prefixed ids and flat collections. Its resources are plural-noun collections — `/v1/charges`, `/v1/customers`, `/v1/payment_intents`, `/v1/refunds` — and items are addressed by typed opaque ids like `ch_...` (charge), `cus_...` (customer), `pi_...` (payment intent), `re_...` (refund). The prefix tells you the object type at a glance; the suffix is opaque, so Stripe can change id generation internally without breaking anyone. Stripe keeps nesting shallow: a refund is a *top-level* `/v1/refunds` collection even though it logically belongs to a charge, and you filter `GET /v1/refunds?charge=ch_...` — exactly the "flatten an independent entity, link with a query" pattern from §3. Filtering, pagination (cursor-based, with `starting_after`/`ending_before`), and field expansion all live in the query string, and identity lives in the path. It is a textbook application of the path/query split, and it is a large part of why Stripe's API has a reputation for being pleasant to consume.

**GitHub's REST API** leans into hierarchical paths that mirror real ownership: `/repos/{owner}/{repo}`, `/repos/{owner}/{repo}/issues`, `/repos/{owner}/{repo}/issues/{number}/comments`. Notice the nesting is justified — an issue genuinely belongs to a repo, a comment to an issue — and it still stays within a sensible depth, with filtering (`?state=open&labels=bug`), sorting (`?sort=created&direction=desc`), and pagination (`?per_page=100&page=2`, plus `Link` headers for next/prev) all in the query. GitHub also uses *both* a natural-ish key (the `{owner}/{repo}` pair and issue `{number}`) and shows the trade-off honestly: those keys are stable enough to be in paths because GitHub treats renames carefully (a renamed repo `301`-redirects from the old path — the canonical-URL-with-redirect pattern from §9 in production). GitHub additionally offers a GraphQL API for clients that need to fetch deep, related graphs in one round-trip rather than walking the nested REST paths — a reminder that URI design and paradigm choice interact, which the later post [Choosing a Paradigm: REST vs gRPC vs GraphQL by Force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force) takes up.

**Google's API Improvement Proposals (AIP)** are the most explicitly codified URI conventions. AIP defines resources as having a hierarchical **resource name** like `publishers/{publisher}/books/{book}` and a small set of standard methods (`List`, `Get`, `Create`, `Update`, `Delete`) that map onto HTTP methods, with **custom methods** expressed using the colon syntax (`:cancel`, `:batchGet`) for the genuine non-CRUD actions — the formalized version of §6's "pragmatic custom action." AIP is firm that collection ids are plural and resource names are hierarchical but bounded, and it draws the path/query split sharply: the resource name identifies; query parameters (and request fields) filter and page. If you want a single, opinionated, internally-consistent rulebook to adopt wholesale, AIP is the most complete one in public.

The common thread across all three: **plural-noun collections, opaque-or-careful identifiers in the path, shallow nesting justified by real ownership, and filtering/sorting/paging in the query.** None of them put an id in the query or a filter in the path on their core resources. The conventions in this post are not idiosyncratic — they are what the careful APIs converged on independently, because the forces (caching, predictability, evolvability) push everyone toward the same answers.

## 11. Stress-testing the design: what breaks at the edges

A design is only as good as its behavior at the edges. Let us push the URI design through the hard cases the way you would in a review.

**What happens when the collection has 50 million rows?** `GET /orders` cannot return 50 million orders. The URI design has already handled this: the collection's *identity* is `/orders` (stable, in the path), and *which window you get* is in the query (`?cursor=...&limit=50`). Because pagination lives in the query and not the path, you can change page size, switch from offset to cursor, or add a new filter without touching the resource's address or breaking a single cached URL. Had you (heaven forbid) encoded the page in the path as `/orders/page/3`, every page would be a separate, fragile resource, and reordering the underlying data would shuffle which orders live at `/orders/page/3` — a moving target masquerading as an identity.

**What happens when a client retries on a timeout?** For a `GET`, nothing bad: it is safe and idempotent, so the retry is free — and that safety is a direct consequence of putting identity in the path and views in the query, which keeps the URL a stable, cacheable, repeatable read. For a `POST /orders`, a naive retry could create a duplicate order; the fix is the `Idempotency-Key` header (a later post), but note that the *URI* did its part by giving the created resource a clean canonical home (`/orders/o_91`) the moment it existed.

**What happens when two clients want the same data sliced differently?** This is the refunds-report scenario, and it is the whole reason for the flatten-and-query rule. Because refunds are a top-level collection, one client can ask `GET /refunds?payment_id=p_7` (refunds for one payment) and another can ask `GET /refunds?created_after=2026-06-15&status=failed` (all failed refunds this period) against the *same* resource. The path stayed a clean noun; both questions went into the query where they belong. Deep-nest refunds under items and one of those questions becomes impossible without a migration.

**What happens when a resource needs to move?** Say payments were originally nested as `/orders/o_91/payment` and you realize they need to be top-level (`/payments/p_7`) for financial reporting. With canonical URLs and redirects (§9), you ship `/payments/p_7` as canonical, `301`-redirect `/orders/o_91/payment` to it, and let clients migrate over a `Deprecation` window — a non-breaking move. Had you no redirect discipline, you would face a hard cutover that strands every client still hitting the old path.

**What happens when the URL gets too long?** A client wants to filter by 5,000 ids. The query string blows past the 8 KB URL cap and gets truncated. The honest answer is a documented `POST /orders/search` with the id list in the body — trading cacheability for the ability to express a large query — clearly marked as the exception to "reads are GETs." The URI design did not fail; it told you where its boundary is.

Each of these is a case where the path/query split either *prevented* a problem (stable identity, cacheable reads, flexible filtering) or *told you exactly where its limits are* (long queries → `POST` search). That is what a good design does: it makes the common cases trivial and the hard cases legible.

## 12. Comparison tables: the decisions in one place

Two tables consolidate the decisions this post turns on. The first is the path-versus-query responsibility split — the central principle — as a reference card:

| Request concern | Path | Query | Header | Body | Notes |
| --- | --- | --- | --- | --- | --- |
| Identity (which resource) | ✓ | ✗ | ✗ | ✗ | id in query is an RPC smell |
| Hierarchy / ownership | ✓ | ✗ | ✗ | ✗ | one level deep, max |
| Filtering | ✗ | ✓ | ✗ | rarely | `POST /search` only when too long |
| Sorting | ✗ | ✓ | ✗ | ✗ | `?sort=-created_at` |
| Pagination | ✗ | ✓ | ✗ | ✗ | cursor/limit in query |
| Sparse fields / projection | ✗ | ✓ | ✗ | ✗ | `?fields=id,total` |
| Expansion of related data | ✗ | ✓ | ✗ | ✗ | `?expand=customer` |
| Format negotiation | fallback | fallback | ✓ | ✗ | `Accept` header is canonical |
| Authentication / tokens | ✗ | ✗ | ✓ | ✗ | never in the URL |
| New resource state (create/update) | ✗ | ✗ | ✗ | ✓ | the payload carries the data |

The second compares the two ways to model a relationship — the nest-versus-flatten decision from §3:

| Dimension | Nest as sub-resource | Flatten to top-level + query |
| --- | --- | --- |
| URI | `/orders/o_91/items` | `/order-items?order_id=o_91` |
| When it fits | child owned by parent; dies with it | child has independent lifetime |
| Queried alone? | no — only via the parent | yes — cross-cutting reports |
| Multiple parents? | no — exactly one owner | yes — references span entities |
| Path depth | one level below the item | always one level (flat) |
| Filter flexibility | only the parent relationship | any field, any combination |
| Cacheability | per-parent cache entries | one collection, normalized query |
| Risk | premature commitment to one access pattern | a little more verbose to navigate |
| Examples | line items, an order's shipments | payments, refunds, invoices |

Read together, the two tables are the operational core of this post: the first says *where each part of a request goes*, and the second says *whether a relationship lives in the path or the query.* If you internalize nothing else, internalize these.

## 13. When to reach for each pattern (and when not to)

Every rule here is a default with a cost, so here is the decisive guidance, including the cases where the "obvious" choice is wrong.

**Do** put identity and hierarchy in the path, and filtering/sorting/pagination/projection in the query. This is the default for essentially every resource-oriented HTTP API, and deviating from it should make you nervous.

**Do** nest a sub-resource when the child is owned by the parent and never queried alone (line items under an order). **Don't** nest past one level, and don't nest an entity that has an independent lifetime or that anyone will ever want to query across the system — flatten it to a top-level collection and link with a query. The cost of over-nesting (a permanently-committed access pattern, an unreachable reporting query) is far higher than the cost of a slightly more verbose navigation.

**Do** use plural-noun collections, opaque stable ids, lowercase paths, and a consistent separator. **Don't** put verbs in the path; the method is the verb. The narrow exception is a genuine custom action with no resource and no field to change — and even then, model it as a noun (`POST /payments/p_7/refunds`) or a colon-suffixed `POST` (`POST /orders/o_91:cancel`) before you reach for a bare `/cancelOrder`.

**Don't** put an id in the query (`?id=o_91`) — that is identity, and it belongs in the path. **Don't** put a filter value in the path (`/orders/paid`) — that is a view, it belongs in the query, and it collides with `/orders/{id}`.

**Don't** use matrix parameters (`;key=value`). The ecosystem never fully supported them and the query string does the job with universal tooling and clean caching.

**Don't** encode the format in the path (`.json`); negotiate it with `Accept`. **Don't** let `/orders` and `/orders/` diverge; normalize or redirect.

**Do** give every resource one canonical URL and `301`-redirect aliases to it, and hand the canonical address back in `Location` on `201 Created`. **Don't** ship two parallel collections for the same thing (`/orders` and `/my-orders`) — fold the difference into a filter or token-driven scoping.

**The meta-rule that overrides all the others: be consistent.** A surface that applies a slightly-imperfect rule uniformly is more usable than one that applies the perfect rule in three places and a different rule in the fourth, because consistency is what lets a caller predict the next endpoint. When you are unsure, copy the convention you already used; do not invent a new one.

When does *none* of this apply? If you are not building a resource-oriented HTTP API at all — if your surface is genuinely a set of procedures (a machine-learning inference call, a batch compute job, a streaming feed) — then RPC or gRPC may be the honest paradigm and you should not contort it into fake resources. The sibling post [RPC vs REST: When a Procedure Beats a Resource](/blog/software-development/api-design/rpc-vs-rest-when-a-procedure-beats-a-resource) and the [system-design paradigm overview](/blog/software-development/system-design/api-design-rest-grpc-graphql) cover that fork. URI design is the craft *within* the resource-oriented choice; if you have made a different choice, make it cleanly rather than half-REST.

## 14. Key takeaways

- **The central rule: identity and hierarchy go in the path; filtering, sorting, pagination, and projection go in the query.** Everything else in URI design is a corollary of this split.
- **The test for any ambiguous case:** if removing a piece of the request changes *which* resource you mean, it is identity → path; if it just changes the resource's *shape or slice*, it is a view → query.
- **Three resource shapes:** plural-noun collections (`/orders`), items by id (`/orders/o_91`), and id-less singletons (`/me`). The shape is set by cardinality — "how many in this context?" — not by taste.
- **Nest only when the parent owns the child's lifetime and nobody queries the child alone — and stop at one level.** Otherwise flatten to a top-level collection and link with a query. You can add a sub-resource later; you can't remove a path clients depend on.
- **Use nouns, not verbs;** the HTTP method is the verb. The payoff is an $O(N+M)$ surface a caller can guess, plus caching, conditional requests, and safe retries the whole web infrastructure understands for free.
- **An id in the query and a filter in the path are both smells** — the first hides identity, the second freezes a view into the address and collides with item ids.
- **Use opaque, stable, typed ids** (`o_91`, `re_3`) — never natural keys a client will parse and you can't change.
- **Skip matrix parameters, file extensions, sensitive data in URLs, and trailing-slash ambiguity;** stick to the path/query split the ecosystem actually supports.
- **Give every resource one canonical URL,** redirect aliases to it with `301`, and return it in `Location` on create — so caches, ETags, and clients all converge on one key.

## 15. Further reading

- **RFC 3986 — Uniform Resource Identifier (URI): Generic Syntax.** The definition of scheme, authority, path, query, and fragment; the source of truth for what a URI *is*.
- **RFC 9110 — HTTP Semantics.** Safe and idempotent methods, the meaning of a `GET`, conditional requests, and why the path/query split makes reads cacheable and retryable.
- **Google API Improvement Proposals (AIP), aip.dev.** The most complete public rulebook for resource names, standard methods, and custom (`:verb`) methods — adopt it wholesale if you want one consistent style guide.
- **Stripe API reference, stripe.com/docs/api.** A widely-cited example of opaque prefixed ids, flat collections, and filtering/pagination in the query — the path/query split done well.
- **Zalando and Microsoft REST API Guidelines.** Two public, opinionated style guides that codify plural collections, casing, pagination, and error conventions; good for resolving the "which convention?" debates with a citation.
- **Within this series:** the intro hub [What Is an API: The Contract Between Systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the foundational [Resource Modeling: Turning a Domain Into Nouns and URIs](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris); the query-side siblings [Filtering, Sorting, and Sparse Fieldsets Without Reinventing SQL](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql) and [Pagination: Offset, Cursor, and Keyset Tradeoffs at Scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale); and the capstone [The API Design Playbook: A Review Checklist From First Endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

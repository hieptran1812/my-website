---
title: "Resource Modeling: Turning a Domain Into Nouns, Relationships, and URIs"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Walk a real commerce domain — customers, orders, line items, payments, refunds, shipments — into a clean REST resource model, learn why nouns beat verbs, and learn the path-versus-query split that holds up for years."
tags:
  [
    "api-design",
    "api",
    "rest",
    "resource-modeling",
    "uri-design",
    "http",
    "domain-modeling",
    "data-modeling",
    "naming",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-1.png"
---

A team I worked with shipped a payments API in a hurry. The first endpoint anyone wrote was `POST /createOrder`. It worked, so the next person added `POST /cancelOrder`, then `GET /getOrderById`, then `POST /addLineItemToOrder`, and — because a product manager asked for it on a Friday — `POST /orderRefundProcessor`. By the time the API had thirty endpoints, no two of them agreed on anything. Some took the id in the path, some in the body, some in a query parameter. Some returned the object, some returned `{ "success": true }`, some returned the id as a string and some as a number. Every new integration started with a developer reading all thirty endpoint names trying to guess which verb did what. There was no way to look at one URI and predict the next one, because there was no model underneath — just a pile of remote procedure calls wearing an HTTP costume.

That API was not badly *implemented*. The handlers were fine. It was badly *modeled*. Nobody had ever sat down and asked the one question that turns a domain into an API: **what are the nouns?** Resource modeling is that question, taken seriously. It is the move where you look at a business — orders being placed, payments being charged, refunds being issued — and decide which things are first-class *resources* with stable identities and addresses, how those resources relate to one another, and what their URIs should be. Get this right and the rest of REST falls out almost mechanically: the methods, the status codes, the pagination, the caching, the versioning all have an obvious home. Get it wrong and you spend the next three years apologizing to client teams for a surface nobody can predict.

![A resource graph of the commerce domain showing a customer placing orders, an order containing line items and being paid by a payment, and the payment being reversed by a refund](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-1.png)

This post is the foundation the rest of this series builds on. We will introduce the **Payments and Orders** example that recurs in every later post — customers, orders, line items, payments, refunds, shipments — and turn it into a resource model and a URI tree you could hand to a client team tomorrow. We will cover identifying resources (the nouns) versus actions (the verbs) and *why* REST prefers nouns; the three shapes a resource takes (entity, collection, singleton); how to model one-to-many and many-to-many relationships as sub-resources versus top-level resources with links; the path-versus-query split that keeps identity separate from filtering; the naming conventions that make a surface predictable; what to do when an action genuinely does not fit a noun (the `POST /payments/{id}/refund` versus `POST /refunds` debate); how to choose resource granularity so the API is neither chatty nor bloated; and the single most important distinction in API design — that **the resource, its representation, and your database schema are three different things**, and the API is not your tables.

By the end you will be able to take any domain a product person describes in a meeting and walk it into a resource model that a stranger can navigate without documentation. That is not a small skill. It is the difference between an API that lasts and one you rewrite in eighteen months.

A note on terms before we start, because this series promises to define jargon the first time it shows up. A **resource** is any thing worth naming and addressing — an order, a customer, the collection of all orders. A **URI** (Uniform Resource Identifier) is the address of a resource: `/orders/o_91`. A **representation** is a concrete encoding of a resource's current state that travels on the wire — the JSON document you get back from a `GET`. **REST** (Representational State Transfer) is the architectural style that says: name your resources, address them with URIs, and act on them with a uniform set of methods, transferring representations back and forth. Everything below is in service of those four ideas.

## 1. Nouns and verbs: why REST is built on things, not actions

Start with the deepest principle, because everything else is a corollary of it. In REST, **the resources are the nouns of your domain and the HTTP methods are the verbs.** You do not invent a new verb for each operation. You name the thing, and you act on the thing with the small, fixed vocabulary HTTP already gives you: `GET` to read it, `POST` to create one, `PUT`/`PATCH` to change it, `DELETE` to remove it.

Why is this better than the procedure-call style, where every operation is its own named endpoint? The answer is the **uniform interface**, and it is worth deriving rather than asserting. Suppose your API has *N* kinds of things (orders, payments, customers, refunds) and you support *M* operations on each (create, read, update, delete, list). In the RPC style, every (thing, operation) pair is a distinct, separately-named endpoint, so the surface grows as $O(N \times M)$ — `createOrder`, `cancelOrder`, `getOrder`, `listOrders`, `createPayment`, `refundPayment`, and on and on, each with its own ad-hoc shape. A client must learn all $N \times M$ names because nothing about `createOrder` lets you predict `cancelOrder`.

In the resource style, you learn the *M* methods **once** — they are HTTP's, not yours — and then each new resource costs only its noun. The surface a client must memorize collapses from $O(N \times M)$ to $O(N + M)$: *N* nouns plus *M* verbs they already know. When you add `shipments` next quarter, a client who has used `orders` already knows `GET /shipments`, `POST /shipments`, `GET /shipments/{id}` without reading a single line of new documentation, because the *verbs are uniform across all nouns*. That predictability — the ability to guess the next endpoint correctly — is the entire payoff of REST. It is not aesthetic preference; it is a real reduction in what a caller must hold in their head.

There is a second, subtler payoff that the RPC style throws away: **the HTTP method carries a promise the verb-in-path style cannot.** HTTP defines methods as *safe* or *idempotent*, and those words have precise meanings a client can rely on. A method is **safe** when calling it has no side effects on the server — `GET` is safe, so a caller (or a proxy, or a browser prefetcher, or a search crawler) can issue it freely without changing anything. A method is **idempotent** when issuing it twice has the same effect on server state as issuing it once — `GET`, `PUT`, and `DELETE` are idempotent, so a client whose request times out on a flaky network can *safely retry* it. `POST` is neither safe nor idempotent, which is exactly why "create" lives there. When you name an endpoint `POST /chargeCard`, the method `POST` still tells an intermediary "not safe, do not retry blindly" — but you have buried that signal behind a verb that says nothing standard. When you instead write `GET /charges/ch_1`, every proxy, gateway, and CDN on the path *knows* the request is safe and cacheable without parsing your custom verb. The uniform interface is not just smaller for humans; it is *machine-legible* to the whole web infrastructure between your client and your server. An RPC-over-HTTP API forfeits all of that and reduces HTTP to a dumb transport tunnel.

This is why "use nouns" is a structural decision, not a style preference. It is the precondition for getting HTTP's caching, conditional requests, safe retries, and intermediary cooperation *for free*. Throw away the nouns and you throw away the reason to be on HTTP at all — at which point you would be better served by an explicit RPC framework like gRPC that at least does not pretend.

![A before and after diagram contrasting verb-in-path endpoints like POST slash createOrder against noun resources acted on by uniform HTTP methods](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-2.png)

Here is the same operation in both styles, on the wire. The RPC version:

```http
POST /createOrder HTTP/1.1
Host: api.shop.example
Content-Type: application/json

{ "customerId": "c_42", "lines": [{ "sku": "BOOK-1", "qty": 2 }] }
```

And the resource version:

```http
POST /orders HTTP/1.1
Host: api.shop.example
Content-Type: application/json

{ "customer_id": "c_42", "items": [{ "sku": "BOOK-1", "quantity": 2 }] }
```

```http
HTTP/1.1 201 Created
Location: /orders/o_91
Content-Type: application/json

{ "id": "o_91", "status": "pending", "customer_id": "c_42", "total": "39.98", "currency": "USD" }
```

The bodies are nearly identical. The difference is in the *envelope*: the resource version uses `POST` (the method) against `/orders` (the noun), returns `201 Created` (the standard "made a new thing" code), and puts the address of the thing it just made in the `Location` header. A client that has ever created any resource on this API knows, without being told, that the same shape works here. The RPC version teaches a client nothing reusable.

### The smell test: a verb in the path

The cleanest diagnostic for a modeling mistake is a **verb in the path**. `POST /createOrder`, `GET /fetchCustomers`, `POST /order/cancel` — the moment a path segment is a verb, you have skipped the modeling step and gone straight to writing a function call. The fix is almost always to find the *noun the verb is acting on* and move the verb into the HTTP method. `createOrder` becomes `POST /orders`. `fetchCustomers` becomes `GET /customers`. `order/cancel` is the only interesting case, and we will spend a whole section on it later, because "cancel" is a verb that does not obviously have a noun — and the answer reveals something deep about modeling.

This is not a hard rule for the sake of rules. We will see later that some operations genuinely resist being nouned, and forcing them produces worse APIs than a tasteful action endpoint would. But the *default* is nouns, and "I have a verb in my path" should always trigger the question "what is the noun, and which HTTP method is this verb?"

#### Worked example: turning a feature request into a resource

A product manager says: "We need a way for customers to save items for later, and a way to move a saved item into their cart." A junior instinct produces `POST /saveItemForLater` and `POST /moveToCart`. Two verbs, two new endpoints, nothing reusable.

Model it instead. What are the *things*? A customer has a **wishlist** (a collection of saved items) and a **cart** (a collection of cart items). "Save for later" means: create an entry in the wishlist. "Move to cart" means: delete it from the wishlist and create it in the cart. So:

```http
POST /customers/c_42/wishlist          # save for later (create a wishlist entry)
DELETE /customers/c_42/wishlist/w_3    # remove from wishlist
POST /customers/c_42/cart              # add to cart (create a cart entry)
```

"Move to cart" is two operations — a `DELETE` and a `POST` — which is honest, because moving an item really is two state changes. The model exposes that truth instead of hiding it behind a verb. And now the API has two new *resources* (`wishlist`, `cart`) that the client already knows how to read, page, and delete, because they obey the same uniform interface as everything else. One feature request, zero new verbs, two reusable nouns. That is the trade you are always trying to make.

It is worth lingering on *why* the noun version composes and the verb version does not, because it generalizes to every future feature. The `wishlist` and `cart` resources, once they exist, automatically support operations nobody explicitly asked for: a client can `GET /customers/c_42/wishlist` to render the saved-items screen, page it with `?limit=20`, count it, and delete entries — all for free, because those are the uniform methods acting on a collection. The `POST /saveItemForLater` verb, by contrast, supports *exactly one thing*: saving for later. The next time the product manager asks "can we show the customer how many items they have saved?", the verb API needs a *new* endpoint (`GET /wishlistCount`?), while the noun API already answered the question the moment the resource existed. Nouns accrue capability; verbs accrue endpoints. Over a few years that difference is the gap between an API with forty coherent resources and one with four hundred incoherent procedures.

## 2. Three shapes: entities, collections, and singletons

Once you have your nouns, each one shows up in your URI tree in one of three shapes. Knowing which is which keeps your paths consistent.

An **entity** (also called an item or instance) is one specific thing with an identity: the order `o_91`, the customer `c_42`. Its URI is the collection plus the identifier: `/orders/o_91`. You `GET` it to read it, `PATCH` it to change part of it, `DELETE` it to remove it.

A **collection** is the set of all entities of a kind: all orders, all customers. Its URI is the bare plural noun: `/orders`, `/customers`. You `GET` it to list (always paged — more on that below), and you `POST` to it to create a new member. A `POST` to a collection is the canonical "make me a new one of these" operation, and it returns `201 Created` with a `Location` header pointing at the new entity's URI.

A **singleton** is a resource of which there is exactly one *in a given context* — there is no collection and no id. The classic example is the current user: `GET /me` returns the authenticated caller's own account, and there is only ever one "me" per request. In our domain, an order's current totals might be exposed as a singleton sub-resource: `GET /orders/o_91/summary` returns the one summary of that one order. There is no `/orders/o_91/summary/{id}` because there is nothing to enumerate. Singletons are `GET` and sometimes `PUT`, rarely `POST` (you do not create a second "me").

The mistake to avoid is mixing the shapes. If `/orders` is a collection, then `/orders/o_91` must be an entity in that collection — not, say, a different concept entirely. And the collection noun should be **plural** (`/orders`, not `/order`), so that `/orders` reads as "the set of orders" and `/orders/o_91` reads as "the order o_91 within that set." This plural-collection convention is nearly universal across well-designed APIs, and consistency here is what lets a client predict that if `/orders/o_91` exists, then `/orders` lists them.

Here is the whole commerce domain rendered as a URI tree, which is the single most useful artifact you produce in resource modeling. It is the map a client navigates.

![A tree of the API URI hierarchy showing the v1 root branching into customers, orders, and payments collections, each with item URIs and sub-resources for line items and refunds](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-3.png)

Reading that tree top to bottom: the root `/v1` holds three top-level collections — `/customers`, `/orders`, `/payments`. Each collection holds entities — `/customers/c_42`, `/orders/o_91`, `/payments/p_7`. And some entities own *sub-resources* — `/orders/o_91/items` (the line items of that order), `/payments/p_7/refunds` (the refunds against that payment). The depth of nesting tells you about ownership, which is the subject of the next two sections.

A guideline on depth, since "how deeply may I nest?" is the most common follow-up question: **stop nesting at the point where the parent stops being necessary to identify the child.** `/orders/o_91/items/li_1` is fine — three levels — because you need the order to know which `li_1` you mean (item ids are only unique *within* an order). But `/orders/o_91/items/li_1/discounts/d_3` is a smell: by the time you are four levels deep, you are almost certainly modeling something that wants to be a top-level resource with a reference. A practical ceiling is two levels of nesting (`collection/item/sub-collection/item`); past that, promote. The reason is partly ergonomic — long URIs are hard to read and hard to build — and partly semantic: deep nesting usually means you have conflated "is related to" with "is owned by," and the deeper relationship is really a reference between two first-class things.

#### Worked example: is the shipping address an entity, a sub-resource, or a singleton?

Take a deceptively simple question: where does a customer's shipping address live? Run it through the three shapes. Is it an **entity** in a collection? Only if a customer can have *many* addresses they manage independently — and they can: home, work, a gift recipient. So addresses are a sub-collection of the customer: `GET /customers/c_42/addresses` lists them, `POST /customers/c_42/addresses` adds one, `GET /customers/c_42/addresses/addr_2` reads one. They are sub-resources because an address has no meaning outside the customer who owns it (the owned-child rule), and they are a *collection* because there are many.

But now consider the *default* shipping address — the one used unless the customer picks another. There is exactly one of those per customer, with no id to enumerate. That is a **singleton**: `GET /customers/c_42/default-address` returns the one default, and `PUT /customers/c_42/default-address` sets it (a `PUT`, because you are replacing the single value, and `PUT` is idempotent so setting it twice is harmless). The same underlying data — addresses — surfaces as both a collection (all of them) and a singleton (the chosen default), and each shape is correct for its access pattern. Recognizing which shape a noun takes in a given context is most of the skill; the URIs follow once you have named the shape.

## 3. Relationships: one-to-many and the sub-resource decision

Domains are not flat lists of independent things. An order *has* line items. A customer *places* many orders. A payment *can have* refunds. These relationships are the connective tissue of your model, and how you express them in URIs is the highest-leverage decision in this whole post, because it determines the shape of the entire tree.

There are two ways to expose a relationship between resource A and resource B:

1. **As a sub-resource** — B lives under A in the path: `/orders/o_91/items`. This says "the items *of* this order," and the order's identity is part of every item's address.
2. **As a top-level resource with a link** — B has its own home, and A *references* it: `/payments/p_7` exists at the top level, and the order body carries a `payment_id` (or a hypermedia link) pointing at it.

The principle that decides between them is **ownership of lifetime and addressability**. Ask two questions:

- **Does B exist independently of A?** If B has no meaning without its parent A — a line item is meaningless without the order it belongs to; it is never queried on its own, never shared between orders, and dies when the order dies — then B is an *owned child* and belongs as a **sub-resource** under A.
- **Is B addressed, queried, or audited on its own, possibly from multiple parents?** If B is a thing the business cares about independently — a payment is reconciled by finance, audited by itself, and one customer's payment is a first-class record regardless of which order triggered it — then B is a *first-class entity* and belongs at the **top level**, with parents linking to it.

![A before and after diagram contrasting line items modeled as an owned sub-resource under an order against a payment modeled as a standalone top-level resource the order links to](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-5.png)

In the commerce domain this gives a clear answer for every relationship:

- **Order → line items: sub-resource.** A line item ("2 copies of BOOK-1 at \$19.99") has no life outside its order. You never list "all line items across all orders" as a business operation. So: `/orders/o_91/items`, `/orders/o_91/items/li_1`. The order owns them.
- **Customer → orders: top-level with link.** An order is a first-class record. Finance audits orders, support looks up orders by id, an order outlives the session that created it. So orders live at `/orders`, and you express the relationship two ways at once: a `customer_id` field on the order body, and a *convenience filter* `GET /customers/c_42/orders` (which is just `GET /orders?customer_id=c_42` with a friendlier address). Note that the order is *not* `/customers/c_42/orders/o_91` as its canonical URI — its canonical address is `/orders/o_91`, because an order is owned by the business, not by the customer's URI namespace.
- **Order → payment: top-level with link.** A payment is audited and reconciled independently, so `/payments/p_7` is canonical and the order carries a `payment_id`.

Here is the contrast on the wire. The owned sub-resource — line items — reads naturally as part of the order:

```http
GET /orders/o_91/items HTTP/1.1
Host: api.shop.example
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [
    { "id": "li_1", "sku": "BOOK-1", "quantity": 2, "unit_price": "19.99", "line_total": "39.98" }
  ],
  "order_id": "o_91"
}
```

The linked top-level resource — the payment — is referenced by id from the order and fetched separately:

```http
GET /orders/o_91 HTTP/1.1
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "o_91",
  "status": "paid",
  "customer_id": "c_42",
  "payment_id": "p_7",
  "total": "39.98",
  "currency": "USD"
}
```

A client that wants payment detail follows the `payment_id`: `GET /payments/p_7`. We will see in the section on granularity that you can *embed* a linked resource inline to save a round-trip when callers always need it — but the canonical model keeps the payment addressable on its own.

### Comparison: sub-resource versus top-level-with-link

| Dimension | Sub-resource `/orders/o_91/items` | Top-level + link `/payments/p_7` |
|---|---|---|
| When to use | B is owned by A, no independent life | B is a first-class business entity |
| Canonical URI | Includes the parent's id | Independent of any parent |
| Lifecycle | Created/deleted with the parent | Has its own lifecycle and id |
| Queried alone? | Rarely or never | Yes — by finance, support, audit |
| Shared by many parents? | No | Possibly (a refund references a payment) |
| Deletion semantics | Cascades with parent | Independent; parent just loses the link |
| Example here | line items, an order's addresses | payments, refunds, shipments |

The failure mode of getting this wrong is concrete. Model a payment as a sub-resource `/orders/o_91/payment` and you have made finance's life miserable: their reconciliation job, which lists all payments in a settlement window, now has to walk every order to find its payment, because there is no `/payments` collection to page through. Conversely, model line items as a top-level `/items` collection and you have created a useless global list nobody queries, plus the awkward question of what `GET /items` even means across all orders. **Match the URI nesting to the ownership of lifetime, and the rest is easy.**

#### Worked example: why the canonical URI matters for a refund

This is the kind of decision that looks academic until it breaks something. Suppose you modeled refunds as a sub-resource of the order — `/orders/o_91/refunds/re_88` — because, after all, the refund "belongs to" the order in a loose sense. Now finance's nightly reconciliation job needs to settle every refund issued yesterday against the card processor's report. With a top-level `/refunds` collection, that job is one paged query: `GET /refunds?created_after=2026-06-19&created_before=2026-06-20`. With the sub-resource model, there *is no* global refunds collection — the only way to find yesterday's refunds is to enumerate every order, then enumerate each order's refunds, filtering by date. That is millions of requests to answer a question that should be one.

Worse, the sub-resource URI couples the refund's identity to the order's. A refund references a *payment*, and one payment can theoretically settle more than one order in some business models — so which order's namespace owns the refund? The question has no clean answer because the premise is wrong: the refund is not owned by the order, it is owned by the *business*, and its canonical address must therefore be top-level. The convenience view `GET /orders/o_91/refunds` can still exist (it is just `GET /refunds?order_id=o_91`), but the *canonical* URI — the one you put in the `Location` header, the one finance stores, the one a webhook references — is `/refunds/re_88`. **The canonical URI of a resource is its identity; choose it by who owns the resource, not by where it happens to be convenient to list it.**

Here is the relationship expressed three ways at once in a real order body, which is what a mature API actually returns — an id to follow, plus a self link, so a client can navigate without string-building URIs:

```http
GET /orders/o_91 HTTP/1.1
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "o_91",
  "status": "paid",
  "customer_id": "c_42",
  "payment_id": "p_7",
  "total": "39.98",
  "currency": "USD",
  "links": {
    "self": "/orders/o_91",
    "items": "/orders/o_91/items",
    "payment": "/payments/p_7",
    "customer": "/customers/c_42"
  }
}
```

The `links` object is a light touch of hypermedia: the server tells the client where the related resources live, so the client follows a URL instead of hard-coding `/payments/` + the id. Whether to include such links — and how far to take hypermedia — is its own debate (we cover [HATEOAS in the real world](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) later, including when those links are weight nobody uses). The point for *modeling* is narrower: the relationships in your domain graph become either nesting (sub-resources) or references (an id and optionally a link), and which one you pick follows directly from ownership of lifetime.

## 4. Many-to-many: model the relationship itself

One-to-many is the common case. Many-to-many is where modelers get sloppy, and it deserves its own treatment because the answer is often *a third resource you did not see coming.*

Consider tags on orders, or categories on products: an order can have many tags, and a tag can be on many orders. There are two clean ways to model this.

**The simple way — a sub-collection of references.** Expose the relationship as a sub-resource that holds *links to* the other side:

```http
PUT /orders/o_91/tags/priority HTTP/1.1
```

```http
HTTP/1.1 204 No Content
```

```http
DELETE /orders/o_91/tags/priority HTTP/1.1
```

```http
HTTP/1.1 204 No Content
```

Here `PUT /orders/o_91/tags/priority` means "ensure this order has the `priority` tag" — and crucially, `PUT` is **idempotent** (a method is idempotent when calling it twice has the same effect as calling it once), so a client can safely retry the tagging on a flaky network without worrying about adding the tag twice. The tag itself is a top-level resource (`/tags/priority`) the order merely references; the sub-collection `/orders/o_91/tags` lists the references.

**The richer way — promote the relationship to a resource.** Sometimes the *link itself* carries data. In our domain, a shipment is exactly this: it relates an order to a set of line items being shipped, and it carries its own data — a carrier, a tracking number, a status. That is not a bare reference; it is a thing. So a shipment is a first-class resource:

```http
POST /shipments HTTP/1.1
Content-Type: application/json

{ "order_id": "o_91", "item_ids": ["li_1"], "carrier": "UPS" }
```

```http
HTTP/1.1 201 Created
Location: /shipments/s_5
Content-Type: application/json

{ "id": "s_5", "order_id": "o_91", "item_ids": ["li_1"], "carrier": "UPS", "status": "label_created", "tracking": null }
```

The lesson generalizes: **when a many-to-many relationship has attributes of its own, the relationship is a resource.** An "enrollment" relates a student to a course but also carries a grade and an enrollment date — so it is a resource, not a join you hide. A "membership" relates a user to a team but carries a role — so it is a resource. The moment you find yourself wanting to attach a field to the *link* between two things, stop treating it as a link and give it a name, an id, and a URI. This is the same instinct that, in database design, turns an implicit join table into an explicit entity — but the API model is making the decision for the *caller's* sake, not the storage engine's.

There is a tell that helps you decide quickly. Ask: **if I delete the relationship, do I want to keep a record that it once existed?** A tag on an order is just a label — delete it and there is nothing to remember; the relationship is a bare reference, so `PUT`/`DELETE` on `/orders/o_91/tags/priority` is enough. A shipment, by contrast, you absolutely want to keep after the order is delivered — it has a tracking number, a delivery date, a carrier; it is part of the order's history. That residual value is the signal that the relationship deserves to be a full resource with a permanent id. Bare references are transient and cheap; relationship-resources are durable and addressable. Here is the contrast as a table you can apply to any many-to-many you meet:

| Question | Bare reference (sub-collection) | Relationship resource (top-level) |
|---|---|---|
| Does the link carry its own data? | No — just connects two things | Yes — carrier, role, grade, date |
| Keep a record after it is removed? | No — delete is forgetting | Yes — it is history |
| How to add / remove | `PUT` / `DELETE` on a sub-path | `POST` / `DELETE` on a collection |
| Idempotent to add? | Yes — `PUT` is idempotent | No — `POST` creates a new record |
| Example here | a tag on an order | a shipment, a membership |

The idempotency row is worth a second look because it is a real wire difference, not a detail. Adding a *bare reference* with `PUT /orders/o_91/tags/priority` is idempotent — call it ten times on a retry storm and the order has the `priority` tag exactly once. Creating a *relationship resource* with `POST /shipments` is **not** idempotent — call it ten times and you have created ten shipments, which is a real bug (ten labels printed, ten carrier pickups). That is precisely the situation where a client must send an `Idempotency-Key` header so the server can recognize the retry and return the *same* created shipment instead of making a new one. The shape of the relationship — reference versus resource — therefore changes the retry contract, which is why getting the model right is not separable from getting the wire behavior right. (Idempotency keys get a full post of their own later in the series; for now, note that the *need* for one falls directly out of the modeling choice.)

#### Worked example: refunds are not a verb, they are a resource

Here is the most important modeling decision in the payments domain, and it is the one teams get wrong most often. A customer wants a refund. The naive endpoint is `POST /payments/p_7/refund` — a verb (`refund`) acting on a payment. It works. So why is it wrong?

Walk the consequences. A refund is *not* a transient action; it is a financial record with a life of its own. It has an amount (you can partially refund). It has a status — a refund to a card is not instant; it goes `pending`, then `succeeded`, sometimes `failed`. It has an id finance needs to reference in a ledger. It can be disputed. It shows up on a statement days later. None of that fits "fire a verb and forget." If you model it as `POST /payments/p_7/refund`, the client gets back a `200 OK` and... then what? How do they check whether the refund actually settled? There is no URI to `GET`. How does finance list all refunds in a settlement window? There is no `/refunds` collection. You have hidden a first-class business object inside a verb.

Model it as a resource instead. Creating a refund is creating a member of the `/refunds` collection:

```http
POST /refunds HTTP/1.1
Content-Type: application/json
Idempotency-Key: rfnd-7f3a-2c91

{ "payment_id": "p_7", "amount": "19.99", "currency": "USD", "reason": "customer_request" }
```

```http
HTTP/1.1 201 Created
Location: /refunds/re_88
Content-Type: application/json

{
  "id": "re_88",
  "payment_id": "p_7",
  "amount": "19.99",
  "currency": "USD",
  "status": "pending",
  "created_at": "2026-06-20T14:03:11Z"
}
```

Now everything has a home. The client polls `GET /refunds/re_88` and watches `status` move `pending → succeeded`. Finance lists `GET /refunds?created_after=2026-06-01&status=succeeded`. The refund is auditable, addressable, and idempotent (note the `Idempotency-Key` header — a string the client picks so that a retried request creates the refund only once; we devote a whole post to it later). The verb endpoint could give you none of that. **When the action produces a thing the business will refer to later, the action is a resource.**

This does not mean *every* action becomes a resource. We tackle the genuine exceptions — the cancellations and state-transitions that resist nouning — in the next section.

## 5. When an action does not fit a noun

Be honest: not every operation is a clean CRUD on a noun. Some are state transitions ("publish this draft," "cancel this order," "approve this request"), and forcing them into pure resource creation produces contortions worse than the problem. This is where dogmatic REST goes wrong, and where good judgment earns its keep.

There are three tools, in order of preference.

**First choice — model the result as a resource (when there is one).** This is the refund case from the last section. If the action *produces* a durable thing — a refund, a shipment, an invoice — create that thing. `POST /refunds`, not `POST /payments/p_7/refund`. This is the default and you should reach for it first.

**Second choice — model the state as a field you change.** If the action just flips an entity between states, expose the state as a field and let the client `PATCH` it. "Cancel an order" is "set the order's status to `cancelled`":

```http
PATCH /orders/o_91 HTTP/1.1
Content-Type: application/merge-patch+json

{ "status": "cancelled" }
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "id": "o_91", "status": "cancelled", "cancelled_at": "2026-06-20T15:00:00Z" }
```

The `application/merge-patch+json` media type (RFC 7396) means "merge these fields into the resource," so the client sends only what changes. The server enforces the legal state machine — it rejects an illegal transition (you cannot cancel an already-shipped order) with a `409 Conflict` or `422 Unprocessable Content` and a clear error body. This keeps the order a single addressable resource whose *state* is data, rather than scattering its lifecycle across a fistful of verb endpoints.

**Third choice — a controller / action sub-resource (the honest escape hatch).** Sometimes the action neither produces a resource nor cleanly maps to a field. It is a genuine *operation* — "recalculate this order's tax," "resend this receipt email," "rotate this API key." These are procedures, and pretending otherwise is worse than admitting it. The accepted REST pattern is a **controller resource**: a verb as the final path segment on a specific entity, invoked with `POST` (because it is neither safe nor idempotent):

```http
POST /orders/o_91/recalculate HTTP/1.1
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "id": "o_91", "total": "43.18", "tax": "3.20" }
```

The rules for using this escape hatch responsibly: (1) it always hangs off a *specific resource* (`/orders/o_91/recalculate`, never a free-floating `/recalculate`); (2) it always uses `POST`; (3) you use it *only* when the first two tools genuinely do not fit, not because modeling is hard. If you find more than a handful of these across your whole API, you have under-modeled — go back and find the nouns you missed.

![A matrix comparing an action modeled as a verb endpoint against the same action modeled as a created state resource across whether it needs an id, has a lifecycle, and the resulting verdict](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-7.png)

### Comparison: action-as-verb versus action-as-resource

| Question about the action | Lean verb endpoint | Lean resource / field |
|---|---|---|
| Does it produce a durable thing? | No — transient effect | Yes — a refund, shipment, invoice |
| Does the result need its own id? | No | Yes — referenced later |
| Does it have a status that changes over time? | No — instant | Yes — pending, then settled |
| Does it just flip an entity's state? | Maybe — use `PATCH` a field | n/a |
| Is it audited or reconciled alone? | No | Yes |
| Verdict | controller sub-resource, `POST` | create the resource or `PATCH` the field |

The decision tree in practice: **(1) does the action create a thing? Make the thing. (2) Does it change an entity's state? PATCH the state field. (3) Neither? A `POST` controller sub-resource on the specific entity, used sparingly.** Run "refund," "cancel," and "recalculate tax" through that tree and you land on `POST /refunds`, `PATCH /orders/{id}`, and `POST /orders/{id}/recalculate` respectively — three different answers, each correct for its case. That is what modeling well looks like: not one rigid rule, but a principled order of preference.

## 6. The path/query split: identity versus narrowing

With the resources and relationships settled, the next decision is the cleanest of all once you see it: **identity goes in the path; everything that filters, sorts, or pages a set goes in the query string.** This single rule resolves a huge fraction of "where does this parameter go?" arguments.

The reasoning is semantic. A path identifies *which resource you mean*. `/orders/o_91` means exactly one order; `/orders` means the collection of all of them. The path is the resource's *name*. A query string, by contrast, does not change *which* resource — it operates *on* the collection you already named, narrowing it: `GET /orders?status=paid&sort=-created_at&limit=50` still asks about "the orders collection," but returns a filtered, sorted, paged *view* of it.

A test that always works: **could two different query strings legitimately point at the same underlying set of things?** `?status=paid` and `?status=pending` both query `/orders` — same resource, different views — so `status` is a query parameter. But `/orders/o_91` and `/orders/o_92` are *different resources*, so the id belongs in the path. Identity in the path; everything else in the query.

![A matrix showing identity belonging in the path while filtering, sorting, and pagination belong in the query string](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-4.png)

Three concrete responsibilities live in the query string, and our running example exercises all three:

**Filtering** narrows the set by attribute. `GET /orders?status=paid&customer_id=c_42` returns paid orders for one customer. Filters are *predicates*, and you should bound them — an unbounded free-form filter is how someone `?sort=`s your database into the ground, a failure mode we will revisit.

**Sorting** orders the result. The widely-used convention is a `sort` parameter with a leading `-` for descending: `GET /orders?sort=-created_at` (newest first), `GET /orders?sort=total` (cheapest first). Always define a *stable* default sort, because an unstable order breaks pagination.

**Pagination** windows the result, because returning all fifty million orders in one response is not an option. `GET /orders?limit=50&cursor=eyJpZCI6...` returns fifty orders plus a cursor to fetch the next page. (We have a full post on offset versus cursor versus keyset pagination later; here it is enough to know it lives in the query string and that page size must be *bounded* by the server so a client cannot ask for `?limit=10000000`.)

```http
GET /orders?status=paid&sort=-created_at&limit=2 HTTP/1.1
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [
    { "id": "o_93", "status": "paid", "total": "12.00", "created_at": "2026-06-20T13:00:00Z" },
    { "id": "o_91", "status": "paid", "total": "39.98", "created_at": "2026-06-20T12:00:00Z" }
  ],
  "next_cursor": "eyJpZCI6Im9fOTEifQ"
}
```

A subtle corollary: the *convenience* sub-resource we mentioned earlier, `GET /customers/c_42/orders`, is exactly equivalent to `GET /orders?customer_id=c_42`. The path version is friendlier to read and lets you scope authorization naturally ("this customer can see only their own orders"), but it is the *same query underneath*. There is nothing wrong with offering both, as long as they return the same shape. What you must not do is put a *filter* in the path as if it were identity — `/orders/paid` is a modeling bug, because `paid` is not the id of an order; it is a value of the `status` filter. That belongs in the query: `/orders?status=paid`.

**Stress-test the query side, because it is where the database gets hurt.** A query parameter is an invitation, and an unbounded invitation is a denial-of-service waiting to happen. Walk the failure: you ship `GET /orders?sort=...` allowing any field, and a well-meaning client sorts by a column with no index. Now every page request triggers a full-table sort over fifty million rows, the query takes eight seconds, connections pile up, and the database falls over — not because anyone was malicious, but because you let the *query string reach the database unbounded*. Three rules close the hole, and they are part of the model, not an afterthought: (1) **allowlist** the filterable and sortable fields — only fields you have indexed and intend to support, returning `400 Bad Request` for `?sort=ssn` or any unknown field; (2) **bound** the page size server-side — a request for `?limit=10000000` is silently capped at, say, 100, and the response says how many it actually returned; (3) **require a default stable sort** so pagination is deterministic. The model decision here is that *the query string is a constrained vocabulary you define*, not a passthrough to your `WHERE` and `ORDER BY` clauses. The fields a caller may filter and sort by are as much a part of your contract as the resources themselves — and like the resources, they are far easier to *add* later than to take away. Start strict.

## 7. Naming conventions: the consistency that makes a surface predictable

Resource modeling is half "what are the things" and half "what do we call them, consistently." The second half feels like bikeshedding until you have lived with an inconsistent surface, at which point you understand that *consistency is the feature*. A client should be able to guess the next URI from the last one. Here is the convention set that buys that, with the *why* attached so you can defend each one in review.

**Plural nouns for collections.** `/orders`, not `/order`. The collection is a set, so its name is plural, and the entity is a member: `/orders/o_91`. Picking plural-everywhere means a client never has to remember whether *this* resource is the special singular one. (Some style guides allow a singular for true singletons like `/me` — that is fine, because there is no collection there to be plural about.)

**No verbs in paths.** Covered above, but it is a naming rule too: path segments are nouns. `GET /orders`, never `GET /getOrders`. The method is the verb.

**Pick one casing and never deviate.** Paths are conventionally **lowercase with hyphens** for multi-word segments: `/shipping-addresses`, not `/shippingAddresses` or `/shipping_addresses`. JSON field names are conventionally either `snake_case` or `camelCase` — *pick one for the whole API* and enforce it with a linter. Mixing `customer_id` in one endpoint and `customerId` in another is the kind of inconsistency that makes a client write defensive code to handle both. This series uses `snake_case` in bodies and hyphenated lowercase in paths, consistently, because the worst choice is no choice.

**No file extensions in URIs.** `/orders/o_91`, never `/orders/o_91.json`. The *representation format* is negotiated with the `Accept` header (`Accept: application/json`), not baked into the URI. The URI names the resource; the format is a separate concern (content negotiation, which gets its own post). A `.json` extension hard-codes a format decision into the identity of the resource, which means you cannot serve the same resource as `application/json` and `text/csv` from one URI.

**No trailing slashes (and be consistent).** Decide whether `/orders` or `/orders/` is canonical — almost everyone picks no trailing slash — and `301` redirect the other form or reject it. Inconsistency here causes subtle cache misses and broken relative links.

**Stable, opaque identifiers.** The id in `/orders/o_91` should be *opaque* to the client — a string they store and echo back, not something they parse. Prefixing the type (`o_91`, `c_42`, `re_88`, like Stripe does with `cus_`, `ch_`, `re_`) is a small kindness: a human reading a log instantly knows `re_88` is a refund, and a client that accidentally passes a customer id where a refund id was expected gets a clean `404` instead of a confusing one. Do *not* leak your database's auto-increment integers as ids — they reveal your row counts, they make resources guessable (`/orders/1`, `/orders/2`, ...), and they tie your public surface to a storage detail you might want to change.

#### Worked example: a naming review that catches three bugs

A teammate opens a pull request adding shipment tracking. The proposed endpoints:

```http
GET  /order/{id}/getShipments
POST /createShipment.json
GET  /shipments?orderID=...
```

A naming review catches three problems before they ship and become forever:

1. `/order/{id}/getShipments` — singular collection (`order` should be `orders`) and a verb in the path (`getShipments`). Fix: `GET /orders/{id}/shipments`.
2. `/createShipment.json` — a verb (`create`) and a file extension. Fix: `POST /shipments`, with `Accept: application/json` for the format.
3. `?orderID=...` — inconsistent casing; the rest of the API uses `snake_case`, so this should be `?order_id=...`.

None of these are functional bugs — all three "work." But each one chips at the predictability of the surface, and once an inconsistency ships in a public API, you can almost never take it back without breaking a client. The review is the cheapest possible place to fix them. This is why mature API teams run a *style linter* (we cover Spectral and governance in a later post) — it catches the casing and pluralization slips automatically, so humans can focus on the genuinely hard modeling questions.

## 8. Resource, representation, and storage: the API is not your database

This is the section to internalize even if you forget everything else, because it is the distinction that separates an API that survives a re-platforming from one that shatters the first time you touch the database. **The resource, its representation, and your persistence model are three different things, and the API contract is bound to the first two — never the third.**

Define them sharply:

- The **resource** is the *concept* — "an order." It has a stable identity (`o_91`) and a meaning in the domain. It does not have a format; it is an abstraction.
- The **representation** is a concrete *encoding of the resource's state at a moment*, in a specific media type, that travels on the wire. The JSON document you get from `GET /orders/o_91` is a representation. The same resource could have an XML representation, a CSV representation, a compact mobile representation — all of the same underlying order. "Representational State Transfer" is named for this: you transfer *representations*, not the resource itself.
- The **persistence model** is how you *store* the resource's state — which is your private business. An order might be three normalized SQL tables (orders, line_items, addresses) joined at read time; it might be a document in MongoDB; it might be assembled from a write-ahead event log; it might be cached in Redis. The client neither knows nor cares.

![A stack diagram showing the order as a resource concept on top, its JSON representation on the wire below, and the service layer mapping to a three-table storage model at the bottom](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-6.png)

The contract — the promise you make to callers — is about the **representation**: these field names, these types, this shape. It must *not* be about the storage. The instant your JSON is a one-to-one mirror of your table columns, you have welded your public contract to your schema, and now you cannot refactor your database without breaking every client. That is a self-inflicted wound, and it is astonishingly common.

Here is the difference, concretely. Suppose orders are stored across three tables. A naive API leaks the join:

```json
{
  "order_row": { "ord_pk": 9914, "cust_fk": 4421, "st": 2 },
  "line_item_rows": [
    { "li_pk": 5, "ord_fk": 9914, "prod_fk": 88, "qty": 2, "cents": 1999 }
  ]
}
```

Every storage detail is visible: the primary keys (`ord_pk`), the foreign keys (`cust_fk`), the table-shaped nesting (`order_row`, `line_item_rows`), the cryptic status integer (`st: 2`), prices as integer cents named `cents`. A client must learn your schema to use your API. And the day you migrate `prod_fk` to a UUID, or split the status into its own table, or move to event sourcing, every client breaks.

The well-designed representation maps to the *domain*, not the tables:

```json
{
  "id": "o_91",
  "customer_id": "c_42",
  "status": "paid",
  "items": [
    { "id": "li_1", "sku": "BOOK-1", "quantity": 2, "unit_price": "19.99", "line_total": "39.98" }
  ],
  "total": "39.98",
  "currency": "USD",
  "created_at": "2026-06-20T12:00:00Z"
}
```

Opaque ids, a human-meaningful `status` string, prices as decimal strings with an explicit `currency`, names that describe the *domain* (`unit_price`, `line_total`) not the *columns*. A **service layer** (sometimes called a mapper, serializer, or presenter) sits between the storage and the wire and translates one into the other. That layer is your insurance policy: it lets you change the database freely as long as you can still *produce the same representation*. You could move orders from Postgres to DynamoDB tomorrow, and if the service layer still emits that JSON, no client notices.

### Comparison: representation versus storage

| Aspect | Representation (the contract) | Storage (your private business) |
|---|---|---|
| Audience | The caller you never meet | Your own engineers |
| Field names | Domain language (`unit_price`) | Schema language (`cents`, `prod_fk`) |
| Identifiers | Opaque, stable (`o_91`) | Auto-increment ints, internal keys |
| Status | Meaningful string (`paid`) | Whatever is efficient (an enum int) |
| Shape | What callers find natural | What is efficient to query |
| Changeability | Slow, governed, versioned | Free — refactor anytime |
| Coupling | Bound to the contract | Bound to nothing public |

There is a price-handling detail worth pausing on, because it is a classic storage-leak bug. Notice the representation uses `"unit_price": "19.99"` — a **string**, not the number `19.99`. Money is not a floating-point number; `0.1 + 0.2` is famously not `0.3` in IEEE-754 floats, and a payments API that returns `total: 39.980000000000004` will be the subject of an angry support ticket within a week. Internally you almost certainly store integer minor units (1999 cents) to avoid float math entirely. But you do not have to *leak* that choice — you can store cents and represent the value as a decimal string `"19.99"` plus a `currency`, giving the client an exact value with no float ambiguity and no need to know your storage uses cents. The representation serves the *caller*; the storage serves *correctness*; the service layer reconciles them. That is the whole pattern in one field.

The status field is the other instructive leak. Storage might encode an order's status as a small integer enum (`0 = draft, 1 = pending, 2 = paid, 3 = shipped, 4 = cancelled`) because integers are compact and index well. If you leak `"status": 2` to the wire, you have committed two sins at once. First, the client must learn your enum *mapping* to do anything useful, which is documentation debt forever. Second — and this is the trap — the integers are now part of your contract, so you can never reorder or renumber them. Insert a new `refunded` state between `paid` and `shipped` and either you break every client that hard-coded `3 == shipped`, or you append `refunded = 5` out of logical order and live with an enum that no longer reads sensibly. Represent the status as a *string* (`"status": "paid"`) and the storage integer is free to change underneath; the service layer maps `2 → "paid"`, and adding `"refunded"` is a backward-compatible additive change (a tolerant client ignores statuses it does not recognize — the robustness principle, which gets its own post). The string is for the caller; the integer is for the index; never confuse the two.

Here is the service-layer mapping made concrete — a tiny presenter that takes a storage row and produces the representation, which is the seam that keeps the contract independent of the schema:

```python
# The mapper is the ONLY place that knows about the storage shape.
STATUS_NAMES = {0: "draft", 1: "pending", 2: "paid", 3: "shipped", 4: "cancelled"}

def present_order(row, items):
    return {
        "id": f"o_{row['ord_pk']}",                  # opaque, type-prefixed id
        "customer_id": f"c_{row['cust_fk']}",        # never leak the raw FK
        "status": STATUS_NAMES[row["st"]],           # int -> domain string
        "items": [present_item(i) for i in items],
        "total": format_money(sum(i["cents"] for i in items)),  # cents -> "39.98"
        "currency": row["currency"],
        "created_at": row["created_at"].isoformat() + "Z",
    }
```

Every storage detail — the `ord_pk` primary key, the `st` integer, the `cents`, the table-shaped row — is consumed *inside* the mapper and never escapes. The day you migrate `ord_pk` to a UUID or move orders to a document store, you change `present_order` and *nothing on the wire moves*. That function is the physical embodiment of "the API is not your database." If you cannot point at the equivalent function in your own codebase, your contract is probably welded to your schema, and you will discover it the hard way during your first real migration. (For how to actually run that migration without downtime once you *do* need to change storage, the database series covers [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — the API mapper is what lets the storage change be invisible to callers in the first place.)

## 9. Granularity: chatty versus coarse

The last modeling dial is **granularity** — how much each resource contains, and therefore how many round-trips a real task takes. Get it too fine and the API is *chatty*: rendering one screen takes five sequential `GET`s, each paying a full network round-trip, which on a mobile connection (typically 50–200 ms per round-trip) turns into a half-second of dead time before anything renders. Get it too coarse and the API is *bloated*: every `GET /orders` drags down the customer's full profile, all line items, the shipment history, and the refund ledger, so a list view that needed three fields downloads a 200 KB payload it throws 95% of away — and a 200 KB JSON body over a cold mobile link is commonly 300–800 ms just in transfer.

![A decision tree for resource granularity branching on whether data is consumed piecemeal or together, leading to embedding children versus offering sparse fieldsets](/imgs/blogs/resource-modeling-turning-a-domain-into-nouns-and-uris-8.png)

The principle that breaks the tie is **shape the resource to how callers actually consume it, not to how you store it.** Two levers tune the dial without abandoning a clean model:

**Embedding (to fight chattiness).** When a child is *almost always* needed with its parent, include it inline rather than forcing a second request. Line items with an order are the canonical case: a client showing an order almost always shows its items, so embedding them in the order representation (as the `"items": [...]` array above) saves a round-trip per order with no real downside, because you never wanted the items without the order anyway. You can make embedding *optional* and explicit with a query parameter, so the caller chooses: `GET /orders/o_91?expand=items,payment` returns the order with its items and payment inlined, while the bare `GET /orders/o_91` returns just the order with ids to follow. (Stripe popularized exactly this with its `expand[]` parameter.) This gives chatty callers a way to coalesce round-trips without forcing the cost on callers who do not want it.

**Sparse fieldsets (to fight bloat).** When a resource is large but a caller needs only a slice, let them ask for just the fields they want: `GET /orders?fields=id,status,total`. A list view that needs three fields then transfers a tiny payload instead of the full order. This is the inverse lever — it lets a caller *shrink* a coarse resource on demand. (We give filtering, sorting, and sparse fieldsets a full post later; here the point is that they are the relief valve for over-coarse resources.)

#### Worked example: an order-list screen, three ways

A mobile "my orders" screen shows, per order: the id, the status, the total, and the date. The customer has 200 orders; the screen shows 20 at a time.

**Too fine.** `GET /orders` returns just ids: `[{"id":"o_91"}, ...]`. The client then fires 20 separate `GET /orders/{id}` calls to get each one's status and total. That is 21 round-trips for one screen. On a 100 ms mobile link, even with some parallelism, the screen stalls for well over a second, and the server eats 21× the request overhead. Chatty and slow.

**Too coarse.** `GET /orders` returns the *full* order each — embedded line items, the linked customer profile, shipment history, refund ledger — 20 fat objects totaling maybe 250 KB, of which the screen uses four fields per order. One round-trip, but a quarter-megabyte transferred to render a list, hammering the mobile user's data plan and p99. Bloated.

**Right-sized.** `GET /orders?fields=id,status,total,created_at&limit=20` returns 20 lean objects with exactly the four fields, a few KB total, in one round-trip. When the user *taps* an order, *then* you fetch the detail with its embeds: `GET /orders/o_91?expand=items,payment`. The list is cheap and the detail is complete, each shaped to its actual use. Same clean resource model underneath — `/orders` is still the collection, `/orders/o_91` still the entity — tuned with the two granularity levers. **The model does not change; the consumption pattern picks the levers.**

## 10. The full commerce domain, modeled end to end

Let us put it all together into the artifact this whole post has been building toward: the complete URI tree of the Payments and Orders API, which is the spine for the rest of the series. Each line is a deliberate modeling decision you can now defend.

```http
# Customers — first-class entities (audited, support looks them up)
GET    /customers                          # list (paged)
POST   /customers                          # create one
GET    /customers/c_42                     # read one
PATCH  /customers/c_42                     # partial update
GET    /customers/c_42/orders              # convenience filter = /orders?customer_id=c_42
GET    /me                                 # singleton: the authenticated customer

# Orders — first-class entities (a business record outliving the session)
GET    /orders                             # list, filter, sort, page in the query
POST   /orders                             # create one -> 201 + Location
GET    /orders/o_91                        # read one
PATCH  /orders/o_91                        # change state (status) or fields
DELETE /orders/o_91                        # cancel (soft) -> 204

# Line items — owned sub-resource of an order (no life of their own)
GET    /orders/o_91/items                  # list this order's items
POST   /orders/o_91/items                  # add an item
GET    /orders/o_91/items/li_1             # read one item
DELETE /orders/o_91/items/li_1             # remove an item
POST   /orders/o_91/recalculate            # controller: an honest action escape hatch

# Payments — first-class entities (finance reconciles them)
GET    /payments                           # list (settlement windows)
POST   /payments                           # charge -> 201 (Idempotency-Key!)
GET    /payments/p_7                        # read one

# Refunds — first-class entities (NOT a verb on payment)
GET    /refunds                            # list
POST   /refunds                            # create -> 201 (references payment_id)
GET    /refunds/re_88                      # read one (poll status)

# Shipments — the order/items relationship promoted to a resource (carries data)
GET    /shipments                          # list
POST   /shipments                          # create -> 201 (references order_id)
GET    /shipments/s_5                      # read one (poll tracking)
```

Read it as a set of decisions:

- **Customers, orders, payments, refunds, shipments are top-level** because each is a first-class business record — independently audited, addressed, and reconciled.
- **Line items are a sub-resource of orders** because they are owned: no order, no item.
- **Refunds are a resource, not a verb** because a refund is a durable financial object with a status, an amount, and an id finance references later.
- **Shipments are a resource** because the order/items relationship carries its own data (carrier, tracking, status) — the relationship *is* a thing.
- **Cancellation is a `PATCH` (or `DELETE`)** on the order, because it is a state transition on an existing entity, not a new thing.
- **Tax recalculation is a controller sub-resource** because it is a genuine procedure that neither creates a durable thing nor cleanly maps to a single field — the honest exception, used once.
- **Identity is in every path; filtering, sorting, and paging live only in query strings** on the collection endpoints.

A client team can navigate this tree without a manual. They can guess that if `/orders` lists orders, `/payments` lists payments. They can guess that creating anything is a `POST` to the plural collection that returns `201` and a `Location`. They can guess that an order's items live at `/orders/{id}/items`. *That guessability is the deliverable of resource modeling.* It is what "RESTful" buys you in practice, and it is why we will spend the next post on the [Richardson maturity model](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) — to make precise what "level of REST" this model sits at and what each rung is worth.

## Case studies: how real APIs model their domains

Theory is cheap; let us check the model against APIs millions of developers actually use. These are accurate as of their long-standing public design; specifics are framed generally where a detail might drift.

**Stripe — the gold standard for the payments domain.** Stripe's API is, not coincidentally, the closest real-world analog to our running example, and it makes exactly the modeling choices argued above. Resources are plural-collection nouns: `/v1/charges`, `/v1/customers`, `/v1/refunds`, `/v1/payment_intents`. Identifiers are *opaque and type-prefixed* — `cus_` for customers, `ch_` for charges, `re_` for refunds, `pi_` for payment intents — which is the kindness described in the naming section. Crucially, **a refund is a top-level resource** (`/v1/refunds`), created by `POST /v1/refunds` with a `charge` (or `payment_intent`) reference in the body — not a verb on the charge. A refund has its own id, its own status, and its own lifecycle, exactly as argued. Stripe also pioneered the optional-embed lever with its `expand[]` parameter — `GET /v1/charges/ch_1?expand[]=customer` inlines the linked customer — letting callers coalesce round-trips on demand. And idempotency keys (`Idempotency-Key` header) make `POST`s safe to retry, which is precisely why creating a refund or a charge can be modeled as a plain resource creation without fear of double-charging. Stripe is what this post looks like, shipped at scale.

**GitHub — sub-resources and the ownership tree.** GitHub's REST API models its domain as a clean ownership hierarchy: `/repos/{owner}/{repo}` is an entity, and the things *owned by* a repo nest under it as sub-resources — `/repos/{owner}/{repo}/issues`, `/repos/{owner}/{repo}/pulls`, `/repos/{owner}/{repo}/issues/{number}/comments`. An issue has no meaning outside its repo, so it nests; this is the owned-child rule in the wild. GitHub also demonstrates the limits of one paradigm — its REST API can require many round-trips to assemble a complex view, which is exactly the over/under-fetching pressure that led GitHub to *also* ship a GraphQL API for clients that need to fetch a precise graph in one request. That is the granularity trade-off playing out at the level of an entire paradigm, and we cover it in [the system-design view of REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql).

**Google's API design guide (AIP).** Google's public *API Improvement Proposals* codify much of this post into a house style across thousands of services. They mandate resource-oriented design: resources are nouns with a collection/resource hierarchy, a small set of standard methods (List, Get, Create, Update, Delete) maps onto the verbs, and — for the genuine exceptions — they define **custom methods** with a `:verb` suffix (e.g. `POST /v1/orders/o_91:recalculate`), which is their disciplined version of the controller sub-resource escape hatch. The fact that Google needed a formal, reviewed convention for "when an action does not fit a noun" tells you the problem in section 5 is real, common, and worth a rule rather than an improvisation.

The throughline across all three: **first-class business objects become top-level resources, owned children nest as sub-resources, ids are opaque, and the rare true-action gets a disciplined, clearly-marked escape hatch.** None of them leak their storage; all of them are guessable. The model in this post is not invented — it is the distilled practice of the APIs you already trust.

## When to reach for clean resource modeling (and when not to)

Resource modeling is the right default for almost any long-lived API, but it is a tool with edges. Be honest about them.

**Reach for it when:**

- You are designing a **public or long-lived API** that other teams or external developers will consume. The guessability and stability pay for themselves many times over across years and clients.
- Your domain is genuinely **noun-shaped** — entities with identities, relationships, and lifecycles (which most business domains are: orders, accounts, documents, devices).
- You want **HTTP's machinery for free** — caching by URI, conditional requests, the uniform method semantics, intermediaries that understand `GET` is safe. A clean resource model is what unlocks all of it.

**Do not force it when:**

- The operation is genuinely a **procedure, not a thing** — "calculate a route between two points," "translate this text," "run this report." Twisting `POST /routeCalculations` into existence to avoid `POST /routes:calculate` (or a clean RPC) is dogma serving itself. When the domain is verb-shaped, an RPC-style or action-oriented design is *more* honest, and we cover exactly when [a procedure beats a resource](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) is a legitimate design center, not a failure.
- You have **one client and one server, both internal, both deployed together**, and you will never have a third consumer. The full ceremony of opaque ids, careful URI trees, and a service-layer mapping is overhead you may not need at MVP scale. (But: "internal, one client" has a way of becoming "three teams depend on it" without anyone deciding it should — so model cleanly the moment a second consumer appears.)
- You are tempted to add **HATEOAS hypermedia links** that no client will ever follow, or **sub-resources for relationships nobody navigates**. Modeling serves the caller; a link or a resource the caller ignores is just weight. Model the relationships that are actually traversed.
- The data is fundamentally a **stream or a real-time feed**, not a collection of addressable things — a live price tick, a log tail. That is a job for SSE, WebSockets, or an event API, not a resource you `GET`.

The meta-rule: **model the domain as resources because it makes the contract predictable and changeable — and stop the moment the modeling stops serving the caller.** Every node in your URI tree should earn its place by being something a real client names, fetches, or navigates. If it does not, it is decoration, and decoration is the thing you cannot delete later without breaking someone.

## Key takeaways

1. **Resources are nouns; HTTP methods are the verbs.** A verb in your path (`/createOrder`) is a modeling mistake — find the noun and move the verb into the method. This collapses the surface a client must learn from $O(N \times M)$ to $O(N + M)$.
2. **Three shapes:** entity (`/orders/o_91`), collection (plural, `/orders`), singleton (`/me`). Collections are plural; entities are the collection plus an opaque id.
3. **Nest owned children as sub-resources** (`/orders/o_91/items` — a line item has no life of its own) and **promote first-class entities to the top level with links** (`/payments/p_7` — finance audits it independently). Match URI nesting to ownership of lifetime.
4. **A many-to-many relationship that carries its own data is a resource**, not a hidden join — a shipment, an enrollment, a membership. The link itself gets a name, an id, and a URI.
5. **When an action does not fit a noun:** first model the result as a resource (`POST /refunds`); else `PATCH` a state field (`status: cancelled`); else, sparingly, a `POST` controller sub-resource (`/orders/o_91/recalculate`) on the specific entity.
6. **Identity in the path; filtering, sorting, and pagination in the query.** `/orders/o_91` names which; `/orders?status=paid&sort=-created_at&limit=50` narrows the set.
7. **Consistency is the feature:** plural collections, lowercase-hyphen paths, one body casing enforced by a linter, no verbs, no file extensions, opaque type-prefixed ids.
8. **The resource, its representation, and your storage are three different things.** The contract binds to the representation, never the tables — a service layer maps between them so you can refactor the database without breaking a client. Money is a decimal string with a currency, never a float.
9. **Tune granularity with two levers, not by changing the model:** embed (`?expand=items`) to fight chattiness, sparse fieldsets (`?fields=id,status`) to fight bloat. Shape resources to how callers consume them.
10. **A well-modeled API is guessable.** A client should predict the next URI from the last. That guessability — not any single rule — is the deliverable, and it is what makes the contract last.

## Further reading

- **RFC 9110 — HTTP Semantics.** The authoritative definition of methods, status codes, and the meaning of "resource" and "representation" this whole post rests on.
- **RFC 7396 — JSON Merge Patch.** The media type behind `PATCH`ing a state field like `status: cancelled`.
- **Google API Improvement Proposals (AIP).** The resource-oriented design standard, including standard methods and the `:verb` custom-method pattern for genuine actions — the disciplined version of the escape hatch in section 5.
- **Stripe API Reference.** The clearest real-world payments resource model: plural collections, opaque type-prefixed ids, refunds as a resource, the `expand[]` embed lever, idempotency keys.
- **Zalando RESTful API Guidelines** and **Microsoft REST API Guidelines.** Two thorough, public, opinionated style guides that codify naming, pagination, and versioning conventions you can adopt wholesale.
- **"REST in Practice" (Webber, Parastatidis, Robinson).** A book-length, principled treatment of resources, representations, and hypermedia.
- Within this series: start at the hub, [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); go deeper on URIs in [choosing URIs: collections, sub-resources, path vs query](/blog/software-development/api-design/choosing-uris-collections-sub-resources-path-vs-query); place this model on the [Richardson maturity model](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means); and see the whole picture in [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- For the distributed-systems and scale view of paradigms and evolution, read [API design: REST vs gRPC vs GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) and [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale).

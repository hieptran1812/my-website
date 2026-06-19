---
title: "HATEOAS in the Real World: Hypermedia, Links, and When to Skip It"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The honest practitioner take on hypermedia: what HATEOAS actually promises, the few places links genuinely pay, and why most teams ship Level 2 and are right to."
tags:
  [
    "api-design",
    "api",
    "rest",
    "hateoas",
    "hypermedia",
    "hal",
    "json-api",
    "http",
    "pagination",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-1.png"
---

A few months into running the Payments API for a fictional commerce platform, I watched a partner integration break in the most avoidable way imaginable. We had shipped refunds at `POST /orders/{id}/refund`. A quarter later, refunds grew their own lifecycle — partial refunds, refund reversals, a refund that could itself be disputed — so we promoted them to a first-class resource at `POST /refunds` with an `order_id` in the body. We announced it. We left the old path alive behind a redirect. And still, three days after the cutover, a partner's nightly reconciliation job started throwing. Their code built the refund URL by hand: `base_url + "/orders/" + order.id + "/refund"`. When we eventually retired the old route, that string template pointed at nothing, and their job `404`ed every night until someone noticed the alerts.

Here is the thing that nagged at me afterward. There is a style of API design whose entire purpose is to prevent exactly this failure. If our order resource had *carried* the refund URL — `order._links.refund.href` — and the partner had *followed* that link instead of assembling their own, the move from `/orders/{id}/refund` to `/refunds` would have been invisible to them. The server would have changed one field in a response body; the client would have followed the new value; nothing would have broken. That style has an ugly acronym: **HATEOAS** — Hypermedia As The Engine Of Application State. It is the top rung of the [Richardson maturity ladder](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means), the "Level 3" that most teams point at and then quietly decline to build.

This post is the honest reckoning with that decision. I am going to define HATEOAS precisely, show you the real wire format in HAL, JSON:API, Siren, and the bare `Link` header, and walk the genuine payoff — a server that can move URLs and change available actions without breaking anyone, an order resource that advertises `pay`, `cancel`, and `refund` only when each is actually allowed. Then I am going to be equally honest about why, after twenty-five years of REST, almost nobody ships full HATEOAS: clients hard-code URLs anyway, SDK code generators bake paths into compiled artifacts, the client-side tooling to *follow* links is thin, and most teams ship a clean [Level 2 API](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) and are completely fine. Figure 1 shows the one idea that makes hypermedia click — an order whose available links change as it moves through its lifecycle — and we will return to that order again and again.

![A state diagram of an order moving from created to paid to refunded, where each state advertises a different set of action links such as pay, cancel, and refund](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-1.png)

By the end you will be able to do four concrete things. First, read and write the standard hypermedia formats and know what each one actually adds over plain JSON. Second, decide — with a real checklist, not a purity score — when links earn their cost and when they are pure ceremony. Third, build the one piece of hypermedia almost every API *should* adopt: `next`/`prev` links on paginated collections. And fourth, design a state-machine resource (our order) whose response tells the client exactly what it may do next, so the client stops hard-coding the rules of your domain. This sits squarely on the series' spine: an API is a contract you must be able to evolve without breaking a caller you will never meet — and hypermedia is one specific, double-edged tool for buying that evolvability.

## What HATEOAS actually is

Let me strip the acronym down to its working parts, because the name is genuinely terrible and the idea underneath is simple.

A **hypermedia control** is a piece of a response that tells the client about a *possible next interaction* — typically a link (a URL the client can follow, tagged with a **relation type**, or **rel**, that names what the link means: `self`, `next`, `cancel`, `refund`) and optionally the method, the expected fields, and the media type. **Hypermedia** is just the property of a representation that it carries these controls inline. (A **representation** is the concrete bytes you return for a resource — the JSON body of an order, say. The same order resource can have many representations: JSON, XML, a compact mobile shape.) And **HATEOAS** — "Hypermedia As The Engine Of Application State" — is the constraint that the *application's state transitions are driven by the hypermedia the server sends*. The client does not hard-code where to go next; it reads the links the server provided and follows one of them.

Contrast that with how nearly every API you have used actually works. The client has a base URL and a mental map of the route table: "to refund order 42, `POST` to `/orders/42/refund`." That map lives in the client's source code, in its generated SDK, in a partner's integration guide. The URLs are **out-of-band knowledge** — agreed once, baked in, never re-checked. HATEOAS says: stop doing that. The server is the authority on its own URLs and on what actions are currently possible. Let it tell you, in-band, in every response.

The canonical analogy is the part of the web you use without thinking. When you load a web page, your browser does not have a hard-coded table of every link on the site. The HTML *is* the hypermedia: every `<a href>` is a link the server chose to show you, every `<form action>` is an action with a method and fields. You click links; you submit forms; the server decides, page by page, what you can do next by which controls it renders. You can redesign an entire site's URL structure overnight and no browser breaks, because no browser ever hard-coded a URL — it followed the ones in the markup. The whole web is a working, planet-scale proof that HATEOAS *can* work. Fielding's complaint, famously, was that "REST APIs must be hypertext-driven" and that most so-called REST APIs are not REST at all because they violate exactly this constraint. He is right about the definition. Whether it matters for *your* JSON API is the real question, and we will get there.

### The two distinct promises

It helps to separate two things HATEOAS is claimed to deliver, because they have very different cost-benefit profiles and people conflate them constantly.

The first promise is **URL decoupling**: the client never constructs a URL, so the server can move, rename, and restructure its URLs at will. Move refunds from `/orders/{id}/refund` to `/refunds`? Just change the `href` in the link; the following client doesn't notice. This is the promise that would have saved my partner integration.

The second promise is **affordance discovery**, which is the more interesting one. An **affordance** is "an action that is currently possible." HATEOAS says the server should advertise, in each response, exactly which actions the client may take *right now*, given the resource's current state. An order in `created` state can be paid or canceled, so its response carries `pay` and `cancel` links. Once paid, those vanish and a `refund` link appears. Once refunded or canceled, no action links remain. The state machine of your domain — which transitions are legal from which states — stops being something every client must re-implement from your documentation, and becomes something the server *computes and ships*. The client just renders the buttons for the links that are present.

That second promise is where, in my experience, hypermedia is most defensible, so let me make it concrete before we touch a single format.

## The state-machine payoff, made concrete

Here is our order resource at Level 2 — clean, conventional, hypermedia-free. It is a perfectly good API response.

```json
{
  "id": "ord_8f2a",
  "status": "created",
  "amount": 4999,
  "currency": "USD",
  "customer_id": "cus_19bc",
  "created_at": "2026-06-20T10:15:00Z"
}
```

A client looking at this knows the order's `status` is `created`. But it does not know what it is *allowed* to do with a `created` order. To know that, the client must have read your documentation and encoded a rule: "if `status == created`, show the Pay and Cancel buttons; if `status == paid`, show Refund; if `status == canceled` or `refunded`, show nothing." That rule — your domain's state machine — now lives in every client. When you add a `pending_authorization` state, or decide that a `paid` order under \$1.00 cannot be refunded, every client must learn the new rule and re-ship. The contract leaked into the consumer.

Now here is the same order with HATEOAS, in the HAL format we will dissect in a moment:

```json
{
  "id": "ord_8f2a",
  "status": "created",
  "amount": 4999,
  "currency": "USD",
  "customer_id": "cus_19bc",
  "created_at": "2026-06-20T10:15:00Z",
  "_links": {
    "self":   { "href": "/orders/ord_8f2a" },
    "pay":    { "href": "/orders/ord_8f2a/payment" },
    "cancel": { "href": "/orders/ord_8f2a/cancellation" }
  }
}
```

The client no longer needs the rule. It renders an action for each link present: a Pay button because `pay` is there, a Cancel button because `cancel` is there. No Refund button, because there is no `refund` link — the server knows a `created` order cannot be refunded and simply did not offer it. The client's logic collapses to: "for each link in `_links`, show the corresponding control." The state machine lives on the server, where it belongs, computed fresh on every read.

Watch what happens after the order is paid:

```json
{
  "id": "ord_8f2a",
  "status": "paid",
  "amount": 4999,
  "currency": "USD",
  "paid_at": "2026-06-20T10:17:32Z",
  "_links": {
    "self":   { "href": "/orders/ord_8f2a" },
    "refund": { "href": "/refunds?order_id=ord_8f2a" }
  }
}
```

The `pay` and `cancel` links are gone — you cannot pay an order twice, and a paid order is canceled by *refunding* it, not by canceling it. A `refund` link has appeared. The client did not change. It re-fetched the order, saw a different set of links, and rendered different buttons. And notice the `refund` link points at `/refunds?order_id=...` — the *new* resource from my opening story. A hypermedia client following that link would have sailed through the migration without a line of code changing. That is the entire pitch, in two response bodies.

This is the genuinely good idea inside HATEOAS, and it is worth holding onto even if you reject the rest: **let state-dependent availability of actions be expressed as the presence or absence of links, so the client renders affordances instead of re-deriving your business rules.** Whether you call it HATEOAS or just "we put action links in the response" matters not at all.

### Why this is provably more evolvable

Let me make the evolvability claim rigorous rather than just asserting it, because "more decoupled" gets thrown around loosely.

Consider the set of facts a client must know to drive your API. At Level 2, that set is: (1) the base URL, (2) the URL *template* for every action — how to construct `/orders/{id}/refund` from an order id, (3) the state-transition rules — which actions are legal in which states. Call these the client's **assumptions**. Every assumption is a clause in the contract that, if the server changes it, breaks the client.

A hypermedia client following links holds far fewer assumptions. It knows (1) the base URL — the single **entry point** it must be told out-of-band — and (2) the *vocabulary of rels* — that a link tagged `refund` means "refund this order," that `next` means "the next page." It does not hold URL templates (it reads `href`), and it does not hold transition rules (it reads which links are present). The server has bought freedom along two whole axes: it can change any URL, and it can change which actions are available in which states, **without modifying the client's assumption set.** The only assumptions that remain — the entry URL and the rel vocabulary — are precisely the ones a hypermedia contract promises *not* to break.

That is the formal version of the payoff. The honest caveat, which we will hammer later: it only pays if clients actually follow links rather than hard-coding them anyway, and if the rel vocabulary really does stay stable while URLs churn. When those conditions fail — and they usually do — you have paid for decoupling you never collect.

![A before and after comparison showing a client that hard-codes URLs breaking on a route change versus a client that follows server links surviving the same change](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-2.png)

Figure 2 puts the two clients side by side. The brittle client templates `/orders/{id}/refund`, the server renames the route, and the client `404`s — my opening story, generalized. The resilient client reads `_links.refund.href`, the server updates that one field when it moves the URL, and the client follows the new value to a clean `200`. Same server change; opposite outcomes; the difference is entirely whether the client built the URL or followed it.

## The standard hypermedia formats

"Put links in the response" sounds simple until you have to decide *exactly where* and *in what shape*, and discover that the JSON object you are returning has no native concept of a link. There are several competing conventions for bolting hypermedia onto JSON, and they differ in how much they add. Let me show you the four you will actually encounter, with real bodies, then compare them in a table.

### HAL — Hypertext Application Language

HAL is the minimalist's choice and by far the most common hypermedia format in the wild. It adds exactly two reserved keys to your JSON object: `_links` and `_embedded`. Everything else is your normal resource fields. A HAL document looks like this:

```json
{
  "id": "ord_8f2a",
  "status": "paid",
  "amount": 4999,
  "currency": "USD",
  "_links": {
    "self":     { "href": "/orders/ord_8f2a" },
    "customer": { "href": "/customers/cus_19bc" },
    "refund":   { "href": "/refunds?order_id=ord_8f2a", "title": "Refund this order" }
  },
  "_embedded": {
    "items": [
      { "sku": "BOOK-001", "qty": 1, "price": 4999,
        "_links": { "self": { "href": "/products/BOOK-001" } } }
    ]
  }
}
```

`_links` is an object keyed by **rel** (the relation name). Each link is an object with at least an `href`, plus optional `title`, `type` (media type), `templated` (a boolean meaning the href is a [URI Template](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) with `{placeholders}` the client fills in), and `name` (to disambiguate multiple links of the same rel). `_embedded` lets you inline related resources to save round-trips — here, the order's line items are embedded rather than requiring a second fetch, and each embedded resource carries its own `_links`.

HAL's strength is its restraint. It adds links and embedding and *nothing else*. Its glaring weakness for our purposes: **HAL has no concept of an action.** A HAL link tells you a URL and a rel; it does not tell you the HTTP method, the body fields, or the content type to use. The convention is that you `GET` a link unless out-of-band knowledge says otherwise. So our `refund` link in HAL says "here is where refunds live" but not "`POST` a body with these fields." For state transitions that need a method and a payload, HAL leans on you to know the method already — which dilutes the affordance-discovery promise. HAL is registered as media type `application/hal+json`.

### JSON:API

JSON:API is a far more opinionated, batteries-included spec. It dictates the entire envelope: top-level `data`, `errors`, `meta`; resources as `{ type, id, attributes, relationships }`; and `links` objects at multiple levels. Its hypermedia model centers on **typed relationships** — first-class, addressable links between resources — and on standardized **pagination links**.

```json
{
  "data": {
    "type": "orders",
    "id": "ord_8f2a",
    "attributes": {
      "status": "paid",
      "amount": 4999,
      "currency": "USD"
    },
    "relationships": {
      "customer": {
        "links": {
          "self": "/orders/ord_8f2a/relationships/customer",
          "related": "/customers/cus_19bc"
        },
        "data": { "type": "customers", "id": "cus_19bc" }
      }
    },
    "links": {
      "self": "/orders/ord_8f2a"
    }
  }
}
```

The `relationships` object is JSON:API's signature: each related entity gets both a `self` link (the relationship itself, which you can `PATCH` to re-link) and a `related` link (the related resource). This is genuinely richer than HAL's flat `_links` — it models the *graph* of your domain, not just a bag of URLs. JSON:API also standardizes collection pagination: a collection response carries `links` with `first`, `last`, `prev`, and `next` at the top level, which is the pagination win we will champion later. Like HAL, though, core JSON:API has **no native action concept** — no method, no input fields for a state transition. (Extensions and conventions exist, but they are not in the base spec.) Its media type is `application/vnd.api+json`, and that `vnd.` vendor prefix is itself a [content-negotiation](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) signal.

### Siren

Siren is the format that takes affordance discovery seriously. Where HAL and JSON:API give you links, Siren gives you **entities, links, *and* actions** — and a Siren action is a complete, executable description of a state transition.

```json
{
  "class": ["order"],
  "properties": {
    "id": "ord_8f2a",
    "status": "paid",
    "amount": 4999,
    "currency": "USD"
  },
  "actions": [
    {
      "name": "refund-order",
      "title": "Refund this order",
      "method": "POST",
      "href": "/refunds",
      "type": "application/json",
      "fields": [
        { "name": "order_id", "type": "hidden", "value": "ord_8f2a" },
        { "name": "amount",   "type": "number", "title": "Refund amount in cents" },
        { "name": "reason",   "type": "text" }
      ]
    }
  ],
  "links": [
    { "rel": ["self"], "href": "/orders/ord_8f2a" }
  ]
}
```

Look at the `actions` array. The `refund-order` action carries everything a client needs to *execute* the refund without any out-of-band knowledge: the `method` (`POST`), the `href` (`/refunds`), the request `type` (`application/json`), and the `fields` — the input names, types, and even a pre-filled hidden `order_id`. A truly generic Siren client could render a form for this action — a number input for the amount, a text input for the reason — entirely from the response, with zero domain knowledge. This is the closest the JSON world gets to the HTML `<form>` that makes the browser a universal hypermedia client.

Siren is the most powerful of the three and the least adopted, for a reason that should give you pause: almost nobody writes a generic Siren client. Building a UI that dynamically renders forms from `actions` is a real engineering project, and most teams would rather hand-code the refund form against a documented endpoint. The richest hypermedia format is the least used precisely because its payoff requires a kind of client almost nobody builds. Hold that thought.

### The Link header (RFC 8288)

The fourth option puts links nowhere in the body at all. **RFC 8288 ("Web Linking")** standardizes the HTTP `Link` header, letting you attach typed links to any response — JSON, binary, a `204 No Content`, anything — without touching the payload format.

```http
HTTP/1.1 200 OK
Content-Type: application/json
Link: </orders/ord_8f2a>; rel="self",
      </customers/cus_19bc>; rel="customer",
      </refunds?order_id=ord_8f2a>; rel="refund"

{
  "id": "ord_8f2a",
  "status": "paid",
  "amount": 4999,
  "currency": "USD"
}
```

The syntax is `<URI>; rel="name"` entries, comma-separated, with optional parameters like `title`, `type`, and `hreflang`. The killer application of the `Link` header is **pagination**, where the IETF even registered the standard rels `next`, `prev`, `first`, and `last`. GitHub's REST API famously uses exactly this for its collections — your client reads the `Link` header to find the `next` page rather than constructing it. The `Link` header's virtue is that it is format-agnostic and leaves your JSON body untouched, which matters when the body shape is already nailed down or when the resource is not JSON at all. Its limit, like HAL's, is that a header link has no method or fields — it is a bare URL with a rel, suited to navigation (`self`, `next`, `customer`) far more than to actions.

![A comparison matrix of HAL, JSON:API, Siren, and the Link header across what each format adds for links, relationships, and actions](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-3.png)

Figure 3 lays the four formats against what they actually add. The pattern is clear: every format does links; JSON:API and Siren add typed relationships and embedding; only Siren ships true actions with method and fields; the `Link` header adds links while leaving the body completely free. Here is the same comparison as a table you can keep:

| Format | Links | Relationships / embedding | Actions (method + fields) | Body intrusion | Media type |
| --- | --- | --- | --- | --- | --- |
| **HAL** | `_links` keyed by rel | `_embedded` for inlined resources | None — `GET` implied | Adds 2 reserved keys | `application/hal+json` |
| **JSON:API** | `links` at every level | First-class typed `relationships` | None in core spec | Dictates entire envelope | `application/vnd.api+json` |
| **Siren** | `links` array, rel-tagged | `entities` (embedded + linked) | Full `actions`: method, href, type, fields | Dictates entire envelope | `application/vnd.siren+json` |
| **Link header** | `Link:` header, RFC 8288 | None | None — bare URL + rel | **Zero** — body untouched | any |

My practical read: if you adopt any hypermedia at all, **HAL for bodies and the `Link` header for pagination** cover the overwhelming majority of real needs with the least ceremony. Reach for JSON:API only if you want its whole opinionated envelope (and can live with its verbosity), and reach for Siren only if you are genuinely building a generic, action-rendering client — which you almost certainly are not.

### Hypermedia and content negotiation

One detail worth nailing down, because it trips people up: a hypermedia format is a **media type**, and choosing one is a [content-negotiation](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) decision, not just a body-shape decision. When you return HAL, the correct `Content-Type` is `application/hal+json`, not bare `application/json`. The `+json` suffix is a **structured syntax suffix**: it says "this is a JSON document, *and* it follows the HAL conventions on top of JSON." A generic JSON parser still reads it fine; a HAL-aware client knows to look for `_links`. The same goes for `application/vnd.api+json` (JSON:API) and `application/vnd.siren+json` (Siren). Those `vnd.` prefixes mark them as vendor/community media types registered for a specific convention.

This matters for evolution. A client can send `Accept: application/hal+json` to *ask* for the hypermedia representation, and `Accept: application/json` to ask for the plain one — and a server can serve both representations of the same resource depending on what the client negotiated. That gives you a graceful adoption path: ship plain JSON by default, and serve HAL only to clients that explicitly opt in by `Accept`. Old clients keep getting the shape they expect; new, link-following clients ask for and receive the richer representation. You did not break anyone, and you introduced hypermedia as an *additive*, negotiated capability rather than a flag-day change to every response.

The trap to avoid: returning HAL-shaped bodies (with `_links`) under a bare `Content-Type: application/json`. It works — JSON is JSON — but it lies about the contract. A client has no signal that `_links` carries meaning, and you have quietly coupled every consumer to a HAL-flavored shape without negotiating it. If you commit to HAL, label it `application/hal+json` and let `Accept` drive the choice. The media type *is* part of the contract; treating it as cosmetic is how you end up unable to add embedding later because someone parsed your "plain JSON" assuming a fixed key set.

### Emitting links on the server: the link is a function of state

The server side of hypermedia is less glamorous than the formats but it is where the affordance-discovery payoff is actually earned or lost. The core principle: **a link is not a static field; it is a function of the resource's current state (and often the caller's permissions).** You do not store `_links` in the database. You *compute* them on every read, from the order's `status` and the requesting principal's scopes. Here is a compact, honest version of what that looks like:

```python
def order_links(order, principal):
    links = {"self": {"href": f"/orders/{order.id}"}}
    if order.status == "created":
        links["pay"] = {"href": f"/orders/{order.id}/payment",
                        "title": "Pay for this order"}
        links["cancel"] = {"href": f"/orders/{order.id}/cancellation",
                           "title": "Cancel this order"}
    elif order.status == "paid":
        # only surface refund if the caller is allowed to issue one
        if principal.has_scope("refunds:write") and order.refundable():
            links["refund"] = {"href": f"/refunds?order_id={order.id}",
                               "title": "Refund this order"}
    # terminal states (canceled, refunded) get only self
    return links
```

Two things in that snippet carry the whole lesson. First, the action links derive from `order.status` — the single source of truth for the state machine now lives here, on the server, and a client gets the *current* answer on every fetch. Second, the `refund` link is gated on `principal.has_scope("refunds:write")` *and* `order.refundable()`. This is a quietly powerful property of state-dependent links: they can encode **authorization** as well as state. A read-only caller never sees a `refund` link, so it never even renders the button — the affordance reflects not just "is this order refundable" but "can *you* refund it." That is genuinely hard to express cleanly at Level 2, where every client must combine the order's status with its own knowledge of the caller's permissions to decide what to show. Here the server, which knows both, just omits the link.

The cost is equally honest: you now have a function whose output is a contract surface. `order_links` must be tested (does a `created` order really expose exactly `self`, `pay`, `cancel`?), versioned (renaming `refund` is breaking, as we will see), and kept in lockstep with the actual transition handlers (if the `refund` link appears but `POST /refunds` rejects the order, you have lied to the client). The link table and the handler table must agree, forever. That coupling is the real tax of affordance-discovery hypermedia — not the bytes on the wire, but the discipline of keeping "what I advertise" exactly equal to "what I'll accept."

#### Worked example: an order's state-dependent action links across its lifecycle

Let me walk the full lifecycle of one order as a sequence of real request/response pairs, so you can see the links appear and disappear as state changes. This is the affordance-discovery promise in motion.

The client creates an order. The order is `created`, so the response offers `pay` and `cancel`:

```http
POST /orders HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 5f3c-create-ord-8f2a

{ "customer_id": "cus_19bc", "items": [ { "sku": "BOOK-001", "qty": 1 } ] }
```

```http
HTTP/1.1 201 Created
Content-Type: application/hal+json
Location: /orders/ord_8f2a

{
  "id": "ord_8f2a",
  "status": "created",
  "amount": 4999,
  "currency": "USD",
  "_links": {
    "self":   { "href": "/orders/ord_8f2a" },
    "pay":    { "href": "/orders/ord_8f2a/payment", "title": "Pay for this order" },
    "cancel": { "href": "/orders/ord_8f2a/cancellation", "title": "Cancel this order" }
  }
}
```

The client renders a Pay button and a Cancel button — one per link present. The user clicks Pay. The client does **not** construct `/orders/ord_8f2a/payment`; it follows `_links.pay.href`:

```http
POST /orders/ord_8f2a/payment HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 5f3c-pay-ord-8f2a

{ "payment_method_id": "pm_4471" }
```

```http
HTTP/1.1 200 OK
Content-Type: application/hal+json

{
  "id": "ord_8f2a",
  "status": "paid",
  "amount": 4999,
  "currency": "USD",
  "paid_at": "2026-06-20T10:17:32Z",
  "_links": {
    "self":   { "href": "/orders/ord_8f2a" },
    "refund": { "href": "/refunds?order_id=ord_8f2a", "title": "Refund this order" }
  }
}
```

The order is now `paid`. The `pay` and `cancel` links are gone; a `refund` link has appeared. The client re-renders: no more Pay or Cancel button, a Refund button instead. The user, satisfied for now, does nothing. A week later they request a refund — the client follows `_links.refund.href`:

```http
POST /refunds?order_id=ord_8f2a HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 5f3c-refund-ord-8f2a

{ "amount": 4999, "reason": "customer_request" }
```

```http
HTTP/1.1 201 Created
Content-Type: application/hal+json
Location: /refunds/rfnd_22a1

{
  "id": "rfnd_22a1",
  "order_id": "ord_8f2a",
  "status": "succeeded",
  "amount": 4999,
  "_links": {
    "self":  { "href": "/refunds/rfnd_22a1" },
    "order": { "href": "/orders/ord_8f2a" }
  }
}
```

And if the client now re-fetches the order, it finds a terminal state with no action links at all:

```http
GET /orders/ord_8f2a HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/hal+json

{
  "id": "ord_8f2a",
  "status": "refunded",
  "amount": 4999,
  "currency": "USD",
  "_links": {
    "self": { "href": "/orders/ord_8f2a" }
  }
}
```

A `refunded` order is terminal. No `pay`, no `cancel`, no `refund` — only `self`. The client renders no action buttons, because there are none to render. Across that entire lifecycle the client held exactly two pieces of out-of-band knowledge: the entry URL and the meaning of the rels `pay`, `cancel`, and `refund`. It never built a URL, never encoded the rule "you can't refund a `created` order" or "you can't pay a `paid` order," and never re-shipped when we later added a `pending_authorization` state with its own links. That is the dream. Now let me explain why the dream is so rarely realized.

## The honest reality: why HATEOAS is rare

I have just spent a third of this post selling hypermedia. Now the confession: in roughly fifteen years of building and consuming HTTP APIs, I have shipped genuine, end-to-end HATEOAS — where real clients drive state by following links and ignore hard-coded URLs — exactly a handful of times, and full Siren-style action-rendering clients zero times. The industry as a whole is the same. The vast, overwhelming majority of successful, beloved, production HTTP APIs sit at Level 2 and do not lose sleep over it. This is not because everyone is lazy or ignorant of Fielding. It is because the theoretical payoff keeps failing to survive contact with how clients are actually built. Let me walk the reasons, because understanding *why* it fails is what lets you spot the rare case where it won't.

![A vertical stack showing the layers that erode the HATEOAS payoff, from clients hard-coding URLs through SDK codegen baking paths to thin tooling, ending at teams staying at Level 2](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-4.png)

Figure 4 stacks the reasons, each one peeling effort off the top. Here they are in detail.

**Clients hard-code URLs anyway.** This is the fatal one. You can ship every link beautifully, and the consuming developer will look at your docs, see that refunds live at `/refunds`, and write `client.post("/refunds", ...)`. Following a link is *extra work* for them — fetch the order, parse `_links`, find the right rel, extract the `href`, then make the call — versus the one obvious line of building the URL directly. Developers optimize for the code in front of them, not for your future URL migration. Worse, your reference docs and tutorials show the literal URLs, so you are *teaching* them to hard-code. The decoupling you built is only collected if the client cooperates, and absent strong incentive, clients don't. The link is there; nobody follows it; the URL is now load-bearing exactly as if you had no link at all — except you also can't move it, because *some* client somewhere hard-coded it.

**SDKs and codegen bake URLs in.** Most serious API consumers don't hand-write HTTP calls; they use an SDK — often one *you* generated from your [OpenAPI spec](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means). And code generators produce methods like `client.orders.refund(orderId)` whose body contains a *compiled-in* path template `"/orders/" + orderId + "/refund"`. The URL is now frozen inside a versioned, published artifact sitting in the partner's `node_modules` or Maven cache. To "follow a link," a generated SDK would have to do a runtime fetch-and-parse, which generators almost never do because the OpenAPI document describes static paths, not link-following. So the dominant consumption mode for APIs is structurally incapable of collecting the URL-decoupling benefit. You shipped `_links`; the SDK ignored them.

**The client tooling is thin.** On the server side, emitting links is easy. On the *client* side, there is no widely adopted, batteries-included "hypermedia client" for JSON the way the browser is for HTML. There is no equivalent of "the browser renders any `<form>`." A few libraries exist (HAL clients, Ketting, traverson), but they are niche, and using one means writing your integration in a link-traversal style that most teams find unfamiliar and verbose. Without ubiquitous client tooling, hypermedia is a server-side gesture into a void.

**The rel names become the new contract.** Here is the subtle trap that bites teams who *do* go hypermedia. You decoupled the client from URLs — but you coupled it to your **rel vocabulary**. The client now hard-codes "look for the link with rel `refund`." Rename that rel to `process-refund` and you have broken every link-following client exactly as surely as renaming a URL breaks a hard-coding one. You did not eliminate the contract; you *moved* it from URLs to rels. That can be a good trade (rels are smaller, more stable, more semantic than URLs) but it is not the free lunch the pitch implies — it is a different, smaller contract that you must still version and not break.

**Most teams ship Level 2 and are fine.** Add it all up and the rational default emerges: a clean Level 2 API — good resource modeling, correct methods, honest status codes, consistent payloads, well-designed errors, sensible pagination — captures the great majority of the value REST offers. Going to Level 3 adds real cost (richer response shapes, the rel vocabulary as a versioned contract, server logic to compute available links per state) for a benefit (URL decoupling, affordance discovery) that most clients structurally cannot or will not collect. For a typical machine-to-machine API with a known set of clients you can coordinate with, that is a bad trade, and skipping it is not a failure of craft. It is craft.

| Reason HATEOAS stays rare | Mechanism | Net effect on the payoff |
| --- | --- | --- |
| Clients hard-code URLs | Building the URL is one obvious line; following a link is five | URL-decoupling benefit never collected; you still can't move the URL |
| SDK / codegen bakes paths | Generated methods compile in `"/orders/{id}/refund"` | Dominant consumption mode ignores `_links` entirely |
| Thin client tooling | No ubiquitous JSON hypermedia client (no browser equivalent) | Link-following is unfamiliar, verbose, rarely adopted |
| Rels become the contract | Client hard-codes rel `refund` instead of the URL | Contract moved, not removed; rel renames still break clients |
| Level 2 is enough | Methods + codes + payloads + pagination capture most value | Going to L3 costs more than it returns for known clients |

### Stress-testing the skeptic's position

Let me argue against myself, because the skeptical case has real holes worth probing.

*"But you said the partner integration broke. HATEOAS would have prevented it. Isn't that a point for hypermedia?"* It is — but notice the precise conditions. It would have helped *only if that partner followed the link*. Given that they hand-built `base_url + "/orders/" + id + "/refund"`, would they have followed a link instead? Almost certainly not — the same instinct that built the URL by hand would have built it by hand whether or not a link was present. The fix that actually works for hard-coding clients is not hypermedia they won't use; it is **not breaking the URL** (keep the redirect forever, or version the change), plus a clear [deprecation and `Sunset` signal](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means). Hypermedia helps the disciplined; it does nothing for the undisciplined, and the undisciplined are the ones who break.

*"What about the affordance-discovery benefit — surely state-dependent links are worth it even if URL decoupling isn't?"* This is the strongest pro-hypermedia argument and I largely agree, which is why I'll champion a scoped version of it shortly. But even here: a client *can* simply read `status` and apply the rules. Yes, that duplicates your state machine into the client. For a UI you own, talking to a backend you own, that duplication is cheap and the coordination is trivial — you ship both together. The affordance-discovery win is real precisely when you *cannot* coordinate: many independent clients, or clients you'll never meet. For one client you control, it's ceremony.

*"The web proves HATEOAS works at planet scale. Why won't it work for my JSON API?"* Because the web has something your JSON API does not: a universal, ubiquitous, generic hypermedia client — the browser — driven by a human who reads the rendered links and decides what to click. The web's hypermedia is consumed by a human-in-the-loop through a client that renders *any* `<a>` and *any* `<form>` with zero per-site code. Your JSON API is consumed by *programs*, which need per-API code regardless, and for which "follow the link" is just more code than "call the URL." The web is a proof that hypermedia works *with a generic client and a human*; it is not a proof that it works for machine-to-machine JSON, where neither condition holds. That asymmetry is the whole reason the analogy oversells.

## Where hypermedia genuinely pays

So when *should* you reach for links? After all the skepticism, there are three clear cases where hypermedia earns its keep, and one of them is something almost every API should adopt. Let me name them precisely.

![A decision tree branching on consumer shape, showing hypermedia paying off for public many-client APIs, stateful workflows, and pagination, while a single internal client should skip full HATEOAS](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-5.png)

Figure 5 is the decision tree. The branch that matters is not "is this RESTful" but "what does the consumer look like." Three "pays" cases, two "skip" cases.

### Case 1: long-lived public APIs with many independent clients

This is the textbook win and it is genuine. If your API is public, will live for years, and is consumed by a large number of clients you cannot coordinate with — clients you will never meet, on release cycles you do not control, some of whom will never update — then your ability to evolve URLs *without* a flag-day migration is worth real money. You cannot call every consumer and ask them to change `/orders/{id}/refund` to `/refunds`. But if a meaningful fraction of them follow links, you can move the URL for those clients invisibly and only worry about the hard-coders. The cost of HATEOAS amortizes over years and thousands of clients; the alternative (every URL is permanently load-bearing, every change is a versioned migration) is genuinely more expensive at that scale. **PayPal** is the canonical real-world example: its REST APIs return HATEOAS `links` arrays — a payment response carries links with rels like `self`, `approve`, and `capture`, each with an `href` and an HTTP `method`, and PayPal's own documentation instructs integrators to follow these links rather than construct URLs. Whether every PayPal integrator actually does so is the perennial question, but the design intent is exactly the public-many-client case.

### Case 2: workflow and state-machine resources

This is the affordance-discovery win, scoped correctly. If a resource has a non-trivial state machine where the *available actions change with state* — an order that can be paid, then refunded, then disputed; a document that can be drafted, submitted, approved, published, retracted; a deployment that can be started, paused, resumed, rolled back — then expressing "what can I do now" as the presence of links is genuinely cleaner than every client re-implementing your transition rules. The win grows with (a) the complexity of the state machine, (b) how often the rules change, and (c) how many distinct clients must stay in sync with them. A five-state workflow consumed by three different front-ends and two partner integrations is exactly where shipping `actions`/`_links` per state saves you from broadcasting "the rules changed" to five teams every quarter. This is the case I'd defend hardest, and it does *not* require full HATEOAS across your whole API — just on the workflow resources where it pays.

### Case 3: paginated collections — the one everyone should adopt

Here is my single strongest recommendation in this entire post, and it is the least controversial: **put `next` and `prev` links on every paginated collection.** This is hypermedia, technically — it is Level 3 — but it is so universally useful and so cheap that essentially every well-designed API does it, including ones that are otherwise pure Level 2 and proud of it.

The reasoning is airtight. Pagination, done well, uses a **cursor** or **keyset** — an opaque token encoding "where the last page ended" — rather than a numeric `offset`, because [offset paging skips and duplicates rows when the underlying table is written to between requests](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale). But an opaque cursor is, by design, something the client *cannot construct* — it doesn't know your encoding, your sort key, or your tie-breaker. So the *only* sane way to hand a client the next page is to hand it the URL: `next: "/orders?cursor=eyJpZCI6Im9yZF84ZjJhIn0&limit=50"`. The client follows the `next` link until there is no `next` link, and the absence of `next` is the unambiguous end-of-collection signal. The client never parses, decodes, or constructs the cursor — which is exactly what you want, because the cursor is your private implementation detail and you must be free to change its encoding without breaking anyone.

Notice this hits both HATEOAS promises at once and dodges every objection. URL decoupling: the cursor format is hidden, so you can change it freely. Affordance discovery: the presence of `next` tells the client "there is more," its absence tells it "you're done." And the objections evaporate — no client *wants* to construct an opaque cursor, the rel vocabulary is the tiny stable set IETF already standardized (`next`, `prev`, `first`, `last`), and even codegen SDKs handle "follow the `next` link in a loop" easily because it's a well-known idiom. This is hypermedia that pays for itself on day one. If you adopt nothing else from this post, adopt this.

#### Worked example: paginating a collection by following next links

Let me make the pagination case fully concrete with a real exchange. The client wants every order for a customer. It does not loop over page numbers; it follows links.

The first request fetches the collection with a page size. The response carries the page of data plus pagination links:

```http
GET /orders?customer_id=cus_19bc&limit=50 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Accept: application/hal+json
```

```http
HTTP/1.1 200 OK
Content-Type: application/hal+json

{
  "count": 50,
  "_links": {
    "self": { "href": "/orders?customer_id=cus_19bc&limit=50" },
    "next": { "href": "/orders?customer_id=cus_19bc&limit=50&cursor=eyJpZCI6Im9yZF84ZjJhIn0" }
  },
  "_embedded": {
    "orders": [
      { "id": "ord_8f2a", "status": "paid", "amount": 4999,
        "_links": { "self": { "href": "/orders/ord_8f2a" } } }
    ]
  }
}
```

(I have shown one embedded order for brevity; in reality there would be fifty.) The client processes the fifty orders, sees a `next` link, and follows it — verbatim, without parsing that base64-looking cursor:

```http
GET /orders?customer_id=cus_19bc&limit=50&cursor=eyJpZCI6Im9yZF84ZjJhIn0 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Accept: application/hal+json
```

```http
HTTP/1.1 200 OK
Content-Type: application/hal+json

{
  "count": 50,
  "_links": {
    "self": { "href": "/orders?customer_id=cus_19bc&limit=50&cursor=eyJpZCI6Im9yZF84ZjJhIn0" },
    "next": { "href": "/orders?customer_id=cus_19bc&limit=50&cursor=eyJpZCI6Im9yZF85YzFkIn0" }
  },
  "_embedded": { "orders": [ /* fifty more */ ] }
}
```

The loop continues. Eventually a page comes back with no `next`:

```http
HTTP/1.1 200 OK
Content-Type: application/hal+json

{
  "count": 12,
  "_links": {
    "self": { "href": "/orders?customer_id=cus_19bc&limit=50&cursor=eyJpZCI6Im9yZF85YzFkIn0" }
  },
  "_embedded": { "orders": [ /* the final twelve */ ] }
}
```

No `next` link means stop. The client's entire pagination logic is the timeless idiom shown below — note it never touches the cursor:

```python
def fetch_all_orders(client, start_url):
    orders, url = [], start_url
    while url:
        page = client.get(url).json()
        orders.extend(page["_embedded"]["orders"])
        url = page["_links"].get("next", {}).get("href")  # None ends the loop
    return orders
```

That `page["_links"].get("next", ...)` is the whole pattern: follow `next` until it's gone. Now compare it to the brittle alternative the client would write against an offset API:

```python
# brittle: client constructs page URLs and guesses the end condition
def fetch_all_orders_offset(client, customer_id, limit=50):
    orders, offset = [], 0
    while True:
        page = client.get(
            f"/orders?customer_id={customer_id}&limit={limit}&offset={offset}"
        ).json()
        batch = page["orders"]
        orders.extend(batch)
        if len(batch) < limit:          # fragile end heuristic
            break
        offset += limit                 # and rows shifted under us meanwhile
    return orders
```

The offset version constructs URLs, hard-codes the page size into the offset arithmetic, guesses the end condition from a short page (which is wrong if a full final page exists), and — the real killer — [skips or double-reads rows when orders are inserted or deleted between requests](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale), because `offset=100` means a different set of rows after the table changes. The link-following version has none of these problems, and it asked the client for nothing but "follow `next`." If you want the database-engine reasoning for *why* a stable keyset beats an offset under concurrent writes, the [b-tree and index-only-scan internals](/blog/software-development/database/b-trees-how-database-indexes-work) live in the database series; here we own the wire contract, and the wire contract is: hand the client a `next` link, never a page number.

The exact same thing works with the `Link` header instead of HAL `_links`, which is how GitHub does it — the data stays a clean JSON array and the navigation lives in the header:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Link: </orders?customer_id=cus_19bc&limit=50&cursor=eyJpZCI6Im9yZF84ZjJhIn0>; rel="next",
      </orders?customer_id=cus_19bc&limit=50>; rel="first"

[ { "id": "ord_8f2a", "status": "paid", "amount": 4999 } ]
```

Either way, the contract is identical: the server hands out opaque next/prev URLs; the client follows them; the cursor format stays the server's private business.

## Designing a stable rel vocabulary

If you do ship links, the single most important design artifact is not the format — it is your **set of relation types**. Recall the trap from earlier: hypermedia decouples clients from URLs but couples them to rels. So the rel vocabulary is now a contract surface as load-bearing as your field names, and it deserves the same deliberate design. A few rules I follow.

**Prefer registered rels where they exist.** The IANA link-relations registry already defines a large set of standard rels — `self`, `next`, `prev`, `first`, `last`, `up`, `collection`, `item`, `edit`, `related`, and more — established by RFC 8288 and a host of other specs. If a registered rel fits, use it. A client that knows the standard rels already understands your `next` and `self` without reading your docs, and you avoid inventing yet another name for "the next page." For pagination specifically, the registered `next`/`prev`/`first`/`last` are exactly right; do not invent `nextPage` or `more`.

**For domain actions, namespace your custom rels or document them as extensions.** There is no registered rel for "refund this order" — that is your domain. The convention is either a short, stable name (`pay`, `cancel`, `refund`) documented in your API reference, or, if you want to be unambiguous and collision-proof, a URI rel like `https://api.shop.example/rels/refund`. The URI form is verbose but self-describing — a client can (in principle) dereference it for documentation. In practice most APIs use short string rels and document them; just treat that documentation as a versioned part of the contract, because it is.

**Name rels for the action, not the URL or the implementation.** A good rel says *what the client can do* (`cancel`, `approve`, `refund`), not *where* (`order-cancellation-endpoint`) or *how* (`post-to-refunds`). The whole point is that the client binds to the *meaning* and the server is free to change the *mechanism*. If you bake the mechanism into the rel name, you have re-coupled the two things you separated and gained nothing.

**Keep the vocabulary small and stable.** Every rel is a promise. The more rels you mint, the larger your contract and the more ways you can break a client by renaming or removing one. A handful of well-chosen rels per resource type is plenty. Resist the urge to express every conceivable navigation as a distinct rel; clients that don't follow a given link don't benefit from its existence, and clients that do are now coupled to it.

When you must change a rel — and you will — treat it exactly like a [field rename under the compatibility rules](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means): ship the old and new rel side by side (two links, same `href`) for a deprecation window, signal the change in your changelog and via `Deprecation`/`Sunset` where applicable, then remove the old rel only after the window closes. The rel vocabulary is not a free-form set of strings; it is a typed, versioned interface, and the teams that get burned by hypermedia are the ones who treated rels as casual labels they could rename at will.

### The SDK and codegen problem, in detail

I claimed earlier that SDKs and code generation are a structural reason HATEOAS rarely pays. Let me make that concrete, because it is the objection that most often goes unexamined.

Picture how a generated SDK is built. You write an [OpenAPI 3.1 document](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) describing your paths — `POST /refunds`, `GET /orders/{id}` — their parameters, and their response schemas. A generator reads that document and emits a typed client library: a `RefundsApi` class with a `create_refund(order_id, amount, reason)` method whose body assembles the request to `/refunds`. The path is a *compile-time constant* inside the generated method. The SDK ships to a package registry; the partner installs version `2.3.1`; the path `/refunds` is now frozen inside an artifact in their dependency tree.

Now ask: how would that generated SDK "follow a link"? To collect the URL-decoupling benefit, `create_refund` would have to first `GET` the order, parse `_links.refund.href`, and `POST` there — but OpenAPI describes *static paths*, not link-following workflows, so the generator has nothing to generate that behavior from. There are emerging conventions for describing links in OpenAPI (the `links` object lets you express that one operation's response can feed another operation's parameters), but they are thinly supported by generators and rarely produce link-*following* client code. The dominant codegen toolchain is, structurally, a path-baking machine. It reads paths and emits methods that call those paths. Hand it `_links` and it does nothing with them.

This is why I keep saying the URL-decoupling benefit is rarely collected: the most common, most professional way to consume an API — a generated, typed SDK — is the way *least* able to benefit from hypermedia. The exception, again, is pagination: "call this method in a loop, following the `next` link until it's absent" is a well-understood idiom that good SDKs *do* implement (often exposing an auto-paginating iterator), because the cursor is opaque and there is genuinely no other sane way to do it. So even SDK-first ecosystems collect the *pagination* slice of hypermedia while ignoring the rest — which is exactly the pragmatic middle path falling out of the tooling, whether or not anyone designed it that way.

## The pragmatic middle path

You do not have to choose between "no hypermedia, ever" and "full HATEOAS purism." The right answer for most teams is a deliberate middle: adopt links exactly where they pay, and skip the rest without guilt. Here is the recipe I actually recommend and ship.

**Always do pagination links.** Every collection endpoint returns `next` (and `prev` where it makes sense, `first`/`last` if cheap) as opaque, follow-me URLs — in HAL `_links`, JSON:API `links`, or an RFC 8288 `Link` header, whichever matches your body convention. This is non-negotiable good practice; it costs almost nothing and buys you cursor-format freedom plus a clean end-of-collection signal. Cost: trivial. Benefit: permanent.

**Do state-transition links on workflow resources.** For resources with a real, evolving state machine — our order, a subscription, a deployment, a multi-step approval — include action links that reflect the *current* state: `pay`/`cancel` on a created order, `refund` on a paid one, nothing on a terminal one. Use HAL `_links` (and accept that the client knows the methods from your docs) or, if you genuinely have a generic client, Siren `actions` with method and fields. This duplicates your state machine *out* of every client and into the server, which is the win. Apply it only to resources where the state machine is non-trivial enough to be worth it — a two-state on/off toggle doesn't need links.

**Skip full HATEOAS everywhere else.** For simple CRUD resources with a known set of clients — a `GET /customers/{id}` that returns a customer — do not festoon the response with links no client will follow. A `self` link is a reasonable, cheap convention (it tells a client the canonical URL of what it's holding) and I usually include it. Beyond that, on a plain CRUD resource consumed by clients you can coordinate with, links are ceremony. Don't add them to hit a maturity level. Add them where they solve a problem you actually have.

This middle path is, not coincidentally, what most of the best-regarded APIs do in practice. They are not Level 3 zealots and they are not Level 2 minimalists; they put links where links pay — pagination and a few stateful workflows — and leave the rest clean.

![A graph showing a hypermedia client fetching an order, branching on which links are present, and following the chosen link rather than constructing any URL itself](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-6.png)

Figure 6 shows what the client side of this middle path looks like for a workflow resource: fetch the order, branch on which links are present, follow the chosen one. The client is dumb in the best way — it has no idea what the state-transition rules are; it only knows how to render and follow whatever links arrived.

#### Worked example: classifying a hypermedia change as breaking or not

A practical question the middle path forces: if you ship links, *what changes to them break clients?* This matters because it tells you what you've actually promised. Let me classify a few changes the way I would in a [backward-compatibility](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) review. Assume HAL clients that follow links by rel.

Suppose v1 of a paid order returns:

```json
{
  "id": "ord_8f2a",
  "status": "paid",
  "_links": {
    "self":   { "href": "/orders/ord_8f2a" },
    "refund": { "href": "/orders/ord_8f2a/refund" }
  }
}
```

Change A — **move the refund URL** from `/orders/{id}/refund` to `/refunds?order_id=...`, keeping the rel `refund`. For a link-following client: **non-breaking.** It reads `_links.refund.href` and follows whatever's there. This is the headline payoff — the change that broke my hard-coding partner is invisible to a following client. For a hard-coding client: breaking, but that client opted out of the contract by ignoring the link.

Change B — **rename the rel** from `refund` to `process-refund`. For a link-following client: **breaking.** The client looks up `_links.refund` and finds nothing; the affordance has vanished from its perspective. The rel is part of your contract exactly as a field name is. Treat a rel rename like a field rename: it needs versioning or a transition window. (You could ship *both* `refund` and `process-refund` rels pointing at the same href for a deprecation period — the link equivalent of keeping a renamed field's old alias.)

Change C — **add a new rel**, say `dispute`, on paid orders. For existing clients: **non-breaking.** A tolerant client iterates the links it understands and ignores rels it doesn't, exactly as it ignores response fields it doesn't recognize (the [robustness principle](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means)). New clients that understand `dispute` render the new button; old ones don't. This is the additive, safe change — and it's why links-for-affordances composes nicely with normal API evolution.

Change D — **stop returning a link in a state where it used to appear** (e.g., you decide refunds over 90 days old are forbidden, so the `refund` link disappears for old paid orders). For a following client: **non-breaking and correct** — it simply won't render the Refund button, which is exactly the new rule. For a hard-coding client: it'll `POST` to the refund URL and get a `409 Conflict` or `422`, having never consulted the (absent) link. This is precisely the affordance-discovery win: the rule change propagates to following clients for free and only bites the ones who ignored the links.

The lesson: **once you ship links, your rel vocabulary and the URLs you put in `href` are part of your contract** — but they break along *different* lines than a Level 2 contract. URLs become safe to move (the whole point); rels become the thing you must not rename casually. Plan your hypermedia contract with the same compatibility discipline you'd apply to fields, and the `Sunset`/`Deprecation` toolkit applies to rels and links just as it does to endpoints.

## Discoverability: the API root as an entry point

There is one more genuine hypermedia payoff worth naming, because it is cheap and it is the purest expression of the idea: a **discoverable entry point**. A fully hypermedia-driven API has a single well-known URL — the root, `GET /` or `GET /api` — whose response is nothing but a directory of links to the top-level resources. A client is told *one* URL out-of-band, fetches it, and discovers everything else by following links from there. In HAL:

```http
GET / HTTP/1.1
Host: api.shop.example
Accept: application/hal+json
```

```http
HTTP/1.1 200 OK
Content-Type: application/hal+json

{
  "_links": {
    "self":      { "href": "/" },
    "orders":    { "href": "/orders", "title": "Orders collection" },
    "payments":  { "href": "/payments", "title": "Payments collection" },
    "refunds":   { "href": "/refunds", "title": "Refunds collection" },
    "customer":  { "href": "/customers/{id}", "templated": true }
  }
}
```

The promise is that a client hard-codes only `https://api.shop.example/` and learns the path to every collection from the root document. Move `/orders` to `/v2/orders`? Update the root link; following clients adapt. Add a new top-level resource? Add a link; tolerant clients ignore what they don't recognize and new clients discover it. This is HATEOAS at its most coherent — the whole API is a graph you traverse from a single seed URL.

The honest assessment is the same as everywhere else in this post: it is theoretically lovely and rarely fully used. Clients overwhelmingly bookmark the collection URLs directly from your docs rather than re-fetching the root on every run, because re-fetching the root to discover `/orders` on every call is an extra round-trip for a URL that hasn't changed in three years. Where the root document *does* earn its keep is as **documentation that can't drift**: it is a machine-readable, always-current list of what the API offers, useful for exploration, for generating a client's initial configuration, and as a health-and-capability check. I include a root link document on public APIs for that reason — not because clients traverse from it on every request, but because it is the one place a curious developer (or a tool) can ask "what can this API do?" and get an answer that is, by construction, never out of date. The `templated: true` flag on the `customer` link, by the way, signals a [URI Template](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) — the client fills in `{id}` itself — which is the one sanctioned place a hypermedia client *does* construct part of a URL, from a template the server provided.

## Case studies — accurate, named, honest

Let me ground all of this in real APIs, being careful to state only what I can stand behind.

**PayPal — designed-for HATEOAS links.** PayPal's REST APIs are the most prominent commercial example of intentional HATEOAS. Their responses include a `links` array where each entry has `href`, `rel`, and `method` — for example, an order or payment response carries rels such as `self`, `approve`, and `capture`, and PayPal's developer documentation explicitly advises integrators to use these returned links to determine the next call rather than hard-coding URLs. The design intent is the long-lived-public-API case: PayPal has enormous numbers of independent integrators it cannot coordinate with, so URL decoupling and explicit next-step links are worth the cost. The honest caveat is the universal one — many integrators still hard-code the documented paths, so PayPal collects the benefit only from the disciplined fraction. The design is sound; adoption is, as always, partial.

**GitHub — partial hypermedia, pagination via the `Link` header.** GitHub's REST API is a textbook example of the pragmatic middle path. It is *not* a full HATEOAS API — most of its resource bodies are plain JSON whose URLs you learn from docs and an SDK. But for **pagination**, GitHub uses the RFC 8288 `Link` header with standard rels `next`, `prev`, `first`, and `last`, and the recommended client behavior is exactly "follow the `next` link until it's gone." GitHub also sprinkles `*_url` fields into many resources (e.g., a repository object includes URLs for its issues, commits, and so on) — a lightweight, HAL-adjacent form of "here's where related things live" without committing to a full hypermedia format. GitHub is the proof that you can be a hugely successful, beloved API by adopting hypermedia *exactly where it pays* (pagination, related-resource URLs) and skipping the rest.

**The Web itself — the Level 3 existence proof.** HTML is the most successful hypermedia system ever built, and it is the reason HATEOAS is theoretically sound. Every page is a representation whose `<a href>` links and `<form action>` controls *are* the affordances; the browser follows them with zero per-site code; a human reads the rendered controls and decides what to do next. Entire sites restructure their URLs without breaking a single browser, because no browser hard-codes a URL — it follows the ones in the markup. This is the genuine, planet-scale demonstration that hypermedia-driven state works. The crucial, honest asymmetry — the one that explains why the web's success does *not* automatically transfer to your JSON API — is that the web has a universal generic client (the browser) and a human in the loop, while machine-to-machine JSON APIs have neither. The web proves the *mechanism*; it does not prove the *economics* for programmatic consumers.

**Stripe — the deliberate Level 2 counter-example.** It is worth naming an enormously successful API that pointedly does *not* do HATEOAS for actions. Stripe's API is famously well-designed and is essentially clean Level 2: resources have documented, stable URLs; you learn them from docs and the SDK; bodies are plain JSON without `_links` action affordances. Stripe does support cursor-based pagination (`has_more`, `starting_after`/`ending_before`), which is the same opaque-cursor idea even if delivered as fields rather than follow-me links. Stripe's success is strong evidence for the thesis of this post: a superb developer experience comes from consistency, great errors, idempotency, and stability — *not* from hypermedia. You do not need Level 3 to build a world-class API, and one of the world's best is proudly Level 2.

The pattern across all four: hypermedia is a tool with a narrow, real sweet spot (huge public surfaces, pagination), and you can build a fantastic API by using it exactly there and nowhere else — or, in Stripe's case, almost nowhere and being better for the focus.

![A matrix mapping API scenarios such as public many-client, stateful workflow, pagination, and single internal client to their decoupling gain, build cost, and final verdict](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-7.png)

Figure 7 — wait, that figure shows the hypermedia-client decision flow; the scenario matrix is Figure 8 below. Both encode the same judgment from different angles: the client's runtime behavior (follow whatever links arrived) and the designer's up-front call (ship links only where the consumer shape rewards it).

## When to reach for hypermedia (and when not to)

Time for the decisive section. Here is how I actually make the call, as a checklist you can run in a design review.

**Reach for hypermedia when:**

- **You have a paginated collection.** Always ship `next`/`prev` as opaque follow-me URLs. This is the universal yes. There is essentially no API for which this is the wrong call. Do it even if you do no other hypermedia.
- **A resource has a non-trivial, evolving state machine** and the available actions genuinely change with state. Ship state-dependent action links (HAL `_links`, or Siren `actions` if you truly have a generic client). The win scales with how complex the rules are and how many clients must track them.
- **Your API is public, long-lived, and consumed by many clients you cannot coordinate with.** The URL-decoupling benefit amortizes over years and a client population you can't herd. This is where full(er) HATEOAS — links on most resources — starts to clear its cost bar.
- **You want a cheap canonical-URL convention.** A `self` link on resources is low-cost and genuinely useful (it tells a client the authoritative URL of what it's holding, handy for caching, re-fetching, and logging). I include `self` almost everywhere; it's the one link with near-universal positive ROI.

**Skip full HATEOAS when:**

- **You have one client, or a small set of clients you control.** If you ship the front-end and the backend together, duplicating the state machine into the client is cheap and coordinating changes is trivial. Action links buy you decoupling you don't need. Read `status`, apply the rules, move on.
- **Your clients consume you through generated SDKs.** Codegen bakes URLs in and ignores `_links`. Shipping elaborate hypermedia into an SDK-dominated ecosystem is effort poured into a channel that structurally can't collect it. (Pagination links are the exception — follow-`next` is an SDK-friendly idiom.)
- **The resource is simple CRUD with stable URLs.** A `GET /customers/{id}` returning a plain customer does not need a forest of links no client will follow. Adding them to hit "Level 3" is the textbook over-engineering Fielding's critics rightly mock — links nobody follows are pure cost.
- **You'd be adding links purely to win a maturity-model argument.** "We should be Level 3" is not a reason; it's cargo cult. The only reasons are the concrete payoffs above. If you can't name which one applies, you don't need the links.

The meta-rule, true to this series' spine: design for the caller you actually have. If your callers are a coordinated handful behind an SDK, optimize for their ergonomics (great docs, stable URLs, idempotency, honest errors) and skip the hypermedia ceremony. If your callers are a teeming, uncoordinatable public on a multi-year horizon, the decoupling links buy is worth real money — collect it, at least for pagination and your stateful workflows. The maturity level is an output of that decision, never the input.

![A matrix mapping API scenarios to their decoupling gain, build cost, and verdict, showing pagination and workflows paying while a single internal client should stay at Level 2](/imgs/blogs/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip-8.png)

Figure 8 is the scenario-to-verdict matrix you can paste into a design doc: read each row across to its verdict. Pagination is a near-universal "ship links." Stateful workflows and large public surfaces earn their action links. A single internal client should stay at Level 2 and spend the saved effort on docs and stability instead.

## Key takeaways

- **HATEOAS means the server's response carries the links and actions that drive the client's next move**, so the client follows links instead of hard-coding URLs and re-deriving your state machine. It promises two distinct things — URL decoupling and affordance discovery — with very different payoffs.
- **The genuinely good idea is state-dependent action links**: an order advertises `pay`/`cancel` when created, `refund` when paid, nothing when terminal. The client renders one control per link present and never encodes your transition rules. Keep this idea even if you reject the label.
- **The four formats add escalating machinery**: HAL (`_links`, minimal), JSON:API (typed `relationships`, opinionated envelope), Siren (full `actions` with method and fields, barely adopted), and the RFC 8288 `Link` header (zero body intrusion, perfect for pagination). HAL plus `Link`-header pagination covers most real needs.
- **HATEOAS is rare for structural reasons, not laziness**: clients hard-code URLs anyway, codegen SDKs bake paths in, client-side link-following tooling is thin, and rels just become a new contract. Most teams ship clean Level 2 and are completely right to.
- **Pagination links are the one piece almost everyone should adopt.** Opaque `next`/`prev` URLs hide your cursor format (so you can change it freely) and give an unambiguous end-of-collection signal — and they dodge every objection because no client wants to build an opaque cursor by hand.
- **Once you ship links, your rels and `href` values are part of the contract** — but they break along different lines: URLs become safe to move (the payoff), while a rel rename breaks following clients exactly like a field rename. Apply the same compatibility discipline.
- **The web proves the mechanism, not the economics.** HTML works because of a universal generic client (the browser) and a human in the loop; machine-to-machine JSON has neither, which is why the analogy oversells for programmatic consumers.
- **Design for the caller you have.** Coordinated clients behind an SDK: skip the ceremony, win on docs and stability (see Stripe). Uncoordinatable public clients over years: collect the decoupling, at least for pagination and stateful workflows (see PayPal, GitHub). The maturity level is the output, never the input.

## Further reading

- **Roy T. Fielding, "Architectural Styles and the Design of Network-based Software Architectures"** (2000 dissertation, Chapter 5) — the original definition of REST and the hypermedia constraint, plus his later blog post "REST APIs must be hypertext-driven."
- **RFC 8288, "Web Linking"** — the standard for the HTTP `Link` header and registered link relation types (`next`, `prev`, `first`, `last`, `self`), the cheapest hypermedia you can ship.
- **The HAL specification** (`application/hal+json`, the JSON Hypertext Application Language internet-draft by Mike Kelly) — `_links`, `_embedded`, and the conventions for rels and templated links.
- **The JSON:API specification** (`jsonapi.org`) — the full opinionated envelope, typed `relationships`, and standardized pagination `links`.
- **The Siren specification** (`github.com/kevinswiber/siren`) — entities, links, and the `actions` model with method, href, type, and fields.
- **RFC 9110, "HTTP Semantics"** — the authoritative reference for methods, status codes, and the `Link`-adjacent header machinery hypermedia rides on.
- Within this series: the [intro hub on the API as a contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the [Richardson maturity model and what RESTful means](/blog/software-development/api-design/the-richardson-maturity-model-and-what-restful-means) (where Level 3 is defined), [resource modeling: turning a domain into nouns and URIs](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris), [pagination: offset, cursor, and keyset trade-offs at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale), and the [capstone API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

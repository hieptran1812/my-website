---
title: "Versioning Strategies: URI, Header, Media-Type — and Not Versioning"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A decision-first guide to API versioning: when you must version at all, how URI, header, media-type, and Stripe-style dated versions actually work on the wire, why not versioning is often the right answer, and what every live version really costs you."
tags:
  [
    "api-design",
    "api",
    "rest",
    "versioning",
    "http",
    "media-types",
    "compatibility",
    "stripe",
    "deprecation",
    "rfc-9110",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-1.png"
---

There is a moment in the life of every API that you remember for years. Ours came on a Tuesday. The Payments team had renamed a single field — `amount` became `amount_cents`, because half the callers were treating the old field as dollars and the other half as cents, and someone had been double-charged \$48.00 instead of \$0.48. The fix was correct. It was also a breaking change, and we shipped it to the live `/orders` endpoint at 2 p.m. By 2:04 p.m., the on-call channel was a wall of red. Every mobile client that read `order.amount` got `undefined`, rendered `\$NaN`, and a meaningful fraction of them crashed on the parse. We had thousands of installed apps in the field that we could not patch, pinned to a contract we had just violated. We rolled back in eleven minutes. Then we spent the next quarter building a versioning strategy, because we had learned the hard way that *an API is a contract you cannot recall once it ships*.

This post is the decision post: **how do you version an API when you actually have to?** That qualifier matters, because the first and most important answer is that most of the time you should not version at all — you should make the change *compatibly*. The companion to this post, [backward and forward compatibility: the rules of safe change](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change), establishes which changes are safe to make in place (add an optional response field, add a new endpoint, accept a new optional request parameter) and which ones break a caller (rename or remove a response field, add a *required* request field, tighten validation, change a type or an enum's meaning). Read that first if you can. Here we pick up exactly where it leaves off: you have a change that you *cannot* make compatibly, you have callers in the field you cannot patch, and now you must give old callers the old contract while new callers get the new one. That is what versioning is — and only that.

By the end of this post you will be able to choose a versioning scheme for a real API on purpose rather than by reflex. We will go through the four live strategies on the wire, with real requests and responses for the Payments & Orders API that runs through this whole series: **URI versioning** (`/v1/orders` vs `/v2/orders`), **header versioning** (a custom `Api-Version` header), **media-type versioning** (`Accept: application/vnd.example.order.v2+json`), and **date-based / rolling versions** (Stripe's pinned `2024-04-10` model with a server-side transformation chain). We will then make the strongest possible case for the fifth option — **not versioning at all** — the additive-only, tolerant-reader discipline that Google and most GraphQL APIs run on, and we will be honest about when it is and is not viable. We will look at granularity (do you version the whole API, one resource, or one field?), at how clients select a version and how servers route on it, at the default-version policy that quietly decides your whole migration's fate, and at the part nobody warns you about: the running cost of every version you keep alive — the v1 you can never quite kill.

![A four by four comparison matrix scoring URI, header, media-type, and no-versioning strategies across visibility, cache-friendliness, granularity, and live maintenance cost](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-1.png)

The figure above is the map for the whole post. No row wins every column. URI versioning is the most visible and the easiest to cache and document, but it couples the version to your entire surface. Header versioning gives clean URLs but is invisible and easy to miss. Media-type versioning is the most RESTfully correct and the most fine-grained, but the tooling around it is thin and most engineers find it obscure. The additive-only "no version" path has the lowest standing cost of all — one URL, one tree of code — but it demands a discipline that not every team can hold. We will earn every cell in that table.

## The premise: version only when you cannot change compatibly

Let me state the principle sharply, because everything else hangs on it.

> **Versioning is the mechanism of last resort for serving two incompatible contracts at once. You introduce a new version only when a change is breaking *and* you cannot absorb it with an additive, compatible change. If the change is compatible, you ship it in place and you do not version anything.**

The reason this is the right ordering is not aesthetic; it is a cost argument. Every version you create is a fork of your contract that you must keep alive, document, test, monitor, and eventually retire. If you reach for a new version every time the schema moves, you will accumulate versions the way a desk accumulates cables, and within two years you will be running v1 through v7 in production with a skeleton crew that is afraid to touch any of them. Compatible change has none of that cost: you add the new optional field, old clients ignore it, new clients use it, and there is exactly one contract and one code path. So the discipline is: *exhaust compatible change first, and only when you genuinely cannot make the change compatibly do you create a version.*

What counts as "cannot make it compatibly"? A short, honest list, drawn from the compatibility rules:

- **Renaming a response field** (`amount` → `amount_cents`). Old readers look up the old name and get nothing.
- **Removing a response field** that clients depend on. Same failure, sometimes worse, because they may have stored it.
- **Changing the type or units of a field** (a string `"49.99"` becoming an integer `4999`; dollars becoming cents). The bytes parse but the meaning is wrong, which is the most dangerous kind of break because it is silent.
- **Adding a required request field**, or making an optional one required. Old callers who do not send it now get a `400`.
- **Tightening validation or narrowing an enum's accepted values.** A request that was legal yesterday is rejected today.
- **Restructuring the response** (flattening a nested object, splitting one resource into two, changing a list into a map).

Notice that several of these *could* be avoided. The rename can become an *addition*: keep `amount` and add `amount_cents` alongside it, deprecate `amount`, and remove it much later behind a long sunset. The "removal" can become a long deprecation where the field stays but is documented as going away. We cover that expand-and-contract dance in [schema evolution: adding, removing, renaming fields safely](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change). The point for *this* post is that when you have walked that road and the change still cannot be made additively — when you must fundamentally reshape a response, or change the meaning of an existing field, or remove a field that you genuinely cannot afford to keep — *that* is the trigger for versioning, and not before.

So the rest of this post assumes you have crossed that line. You have a real breaking change. Now: which scheme?

### The same change, four ways

To compare schemes fairly, we will hold the change constant and express it in each one. The change is the one from our outage: in the Orders resource, the field `amount` (a float in dollars) becomes `amount_cents` (an integer in minor units), and we restructure the embedded `customer` from a flat string into an object. Here is a v1 Order body and the v2 body we want to ship:

```json
{
  "id": "ord_8f2a",
  "amount": 49.99,
  "currency": "USD",
  "customer": "cus_7Yh3",
  "status": "paid"
}
```

```json
{
  "id": "ord_8f2a",
  "amount_cents": 4999,
  "currency": "USD",
  "customer": { "id": "cus_7Yh3", "email": "buyer@example.com" },
  "status": "paid"
}
```

That is unambiguously breaking on two counts: the field rename plus unit change, and the type change of `customer` from string to object. Every scheme below has to give a v1 caller the first body and a v2 caller the second.

## URI versioning: the version in the path

URI versioning puts the version number directly in the URL path: `/v1/orders/{id}` and `/v2/orders/{id}` are two different URIs. This is by a wide margin the most common scheme on the public internet, and for good reasons that we will get to — but it is also the one that REST purists object to most loudly, so it is worth understanding both why people use it and why purists wince.

Here is the v1 request and response on the wire. (A *request/response pair on the wire* just means the raw bytes the client sends and the server returns — the method, path, headers, status code, and body — as opposed to the SDK call you write in your language.)

```http
GET /v1/orders/ord_8f2a HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Accept: application/json

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_8f2a",
  "amount": 49.99,
  "currency": "USD",
  "customer": "cus_7Yh3",
  "status": "paid"
}
```

And the v2 request — a *different URL* — returns the new shape:

```http
GET /v2/orders/ord_8f2a HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Accept: application/json

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_8f2a",
  "amount_cents": 4999,
  "currency": "USD",
  "customer": { "id": "cus_7Yh3", "email": "buyer@example.com" },
  "status": "paid"
}
```

The version is right there in the path, so a `curl` against either contract is trivial and self-documenting:

```bash
# v1 — the old contract, for clients we cannot patch
curl -H "Authorization: Bearer <token>" \
  https://api.example.com/v1/orders/ord_8f2a

# v2 — the new contract, for new integrations
curl -H "Authorization: Bearer <token>" \
  https://api.example.com/v2/orders/ord_8f2a
```

### Why URI versioning is so popular

The advantages of URI versioning are concrete and operational, which is exactly why it dominates despite the purist objections.

**It is maximally visible.** The version is the first thing anyone sees in a request, a log line, an access trace, a browser address bar, a Slack message, a bug report. When a partner emails "your orders endpoint is returning a weird shape," the first question — *which version?* — answers itself from the URL they paste. Visibility is not a small thing; a huge fraction of API support burden is figuring out what the client actually called, and URI versioning makes that free.

**It routes and caches trivially.** Because the version is part of the path, your gateway, load balancer, and CDN can route and cache on the path alone with no special configuration. `/v1/orders/ord_8f2a` and `/v2/orders/ord_8f2a` are simply two different cache keys; a shared HTTP cache or CDN distinguishes them automatically. With header or media-type versioning you have to teach every cache to vary on a header (more on that pain below), and a misconfigured cache that *ignores* the versioning header will happily serve a v2 body to a v1 client — a silent, nasty bug. URI versioning makes that class of bug structurally impossible.

**It is easy to document and to navigate.** You can publish `/v1` docs and `/v2` docs as separate trees, and a developer can hold one version in their head at a time. Hyperlinks between resources naturally carry the version forward, so a v2 response that links to a related resource links to *its* v2 URI.

**It is dead simple to implement and reason about.** Routing on a path prefix is the single most universally supported thing every web framework, gateway, and proxy does. There is no negotiation, no parsing of structured media types, no precedence rules to get wrong.

### Why purists object — and where they have a point

The objection, in the strict-REST view, is that a URI is supposed to identify a *resource*, not a *representation* of that resource. The order with id `ord_8f2a` is one resource; `application/vnd.example.order.v1+json` and `...v2+json` are two representations of it. Putting `v1`/`v2` in the path means you now have two URIs that refer to *the same underlying thing*, which breaks the idea that a URI is a stable, canonical identifier. If a client stored `/v1/orders/ord_8f2a` as a link and you sunset v1, that link is now dead even though the order still exists. We cover the resource-versus-representation distinction in depth in [content negotiation, media types, and representations](/blog/software-development/api-design/content-negotiation-media-types-and-representations); the short version is that media-type versioning treats the version as a representation choice (the "correct" framing) while URI versioning treats it as a separate resource (the pragmatic framing).

The practical, non-theological objections are sharper and worth taking seriously:

- **It couples the version to the *entire* surface.** When you cut `/v2`, you typically version *everything* — `/v2/orders`, `/v2/payments`, `/v2/refunds` — even the resources that did not change. A caller who only consumes `/payments`, which was untouched, still has to migrate their base path from `/v1` to `/v2` for no functional reason. The version is global by default, which is the coarsest possible granularity. (You *can* version per-resource paths like `/orders/v2`, but it is unusual and confuses routing.)
- **It fragments your URL space.** Every resource now exists at two (or more) paths. Bookmarks, stored links, and `Location` headers from old responses all point at version-specific URIs that will eventually rot.
- **It tempts you to bump the version for trivial reasons.** Because `/v2` is cheap to stand up, teams sometimes cut a whole new version for a change that could have been additive — which is exactly the cost trap we warned about.

The honest verdict: for a **public API with many unknown clients**, especially ones you cannot patch (mobile apps, third-party integrations, embedded devices), URI versioning's visibility and cache-simplicity usually outweigh its theoretical impurity. It is the default for a reason. Use it when the version naturally changes for the whole surface at once (a true v1→v2 platform jump) and when ops simplicity matters more than per-resource granularity.

![A before and after figure showing the same field rename expressed as a path change for URI versioning and as an Accept header change for media-type versioning](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-2.png)

#### Worked example: the rename, lived out in URI versioning

Walk the timeline of our `amount` → `amount_cents` change under URI versioning, because the *operational* story is where the trade-offs become real.

Day 0: `/v1/orders` is the only path, returning `amount`. You build `/v2/orders` as a parallel route returning `amount_cents` and the structured customer. Both run; they share the same database and most of the same code, diverging only at the serialization layer (the v1 serializer emits `amount` by dividing the stored cents by 100; the v2 serializer emits `amount_cents` directly). You announce v2, publish v2 docs, and update your SDK to call v2 by default for new integrations.

Day 1 to month 18: both versions are live. Your existing mobile apps keep calling `/v1` and keep working — the field they read is exactly where it always was. New web clients call `/v2`. This is the whole point: nobody is broken, and you got to make the change. The cost is that *every new feature* you add to Orders has to be considered for both serializers, and your tests run against both. A bug in the cents-to-dollars conversion in the v1 serializer is a real risk you have re-introduced by keeping v1 alive.

Month 18 to removal: you send `Deprecation` and `Sunset` headers on `/v1` responses (covered in [deprecation and sunset: retiring an API humanely](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely)), watch the v1 traffic decay, chase the last few stubborn callers, and finally return `410 Gone` on `/v1/orders`. If even one important partner never migrates, you do not get to remove v1 — and *that* is the version you can never kill. We come back to that cost at the end.

## Header versioning: the version in a custom header

Header versioning keeps a single, clean URL — `/orders/{id}` with no version in the path — and selects the version with a request header. The most common shape is a dedicated custom header:

```http
GET /orders/ord_8f2a HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Accept: application/json
Api-Version: 2

HTTP/1.1 200 OK
Content-Type: application/json
Vary: Api-Version

{
  "id": "ord_8f2a",
  "amount_cents": 4999,
  "currency": "USD",
  "customer": { "id": "cus_7Yh3", "email": "buyer@example.com" },
  "status": "paid"
}
```

A v1 client sends `Api-Version: 1` (or omits it and falls to the default — we will get to defaults) and receives the old shape from the same URL. The `Vary: Api-Version` response header is not decoration; it is mandatory and it is the crux of the trade-off, as we will see.

The version selector is sometimes a simple integer (`Api-Version: 2`), sometimes a date (`Api-Version: 2024-04-10`, which blurs into the date-based scheme below). Some APIs use a header named `Accept-Version` or a vendor-specific name. The mechanism is the same regardless of the spelling.

### What header versioning buys and costs

**The win is clean URLs.** A URI identifies a resource, full stop; the version is a per-request *concern*, expressed where concerns belong — in the headers. The same `/orders/ord_8f2a` link is valid forever, regardless of version. This is closer to the REST model than URI versioning and it keeps your URL space from fragmenting.

**It is per-request and easy to flip.** A client can change the header per call without rewriting URLs, which makes targeted testing easy: hit the same endpoint with `Api-Version: 1` and `Api-Version: 2` and diff the bodies. It also makes per-request granularity natural — a client could, in principle, pin different versions for different calls (though in practice you usually pin one version per client).

Now the costs, which are real:

- **It is invisible.** This is the headline drawback. The version is not in the URL, so it does not show up in a pasted link, a browser address bar, a casual `curl` someone types from memory, or most log formats unless you explicitly log that header. The renamed-field outage that opens this post is *more* likely under header versioning, because a developer copying a working request and "simplifying" it can easily drop the `Api-Version` header and silently fall to a different version. Visibility is a safety feature, and header versioning gives it up.
- **It complicates caching.** A shared cache (a CDN, a reverse proxy, the browser cache) keys on the URL by default. If `/orders/ord_8f2a` returns *different bytes* depending on `Api-Version`, the cache *must* be told to vary on that header — that is what `Vary: Api-Version` does. Miss it, and the cache will serve whatever it cached first to everyone: a v2 body to a v1 client, or vice versa. Even when you set `Vary` correctly, you fragment your cache (one entry per version) and many CDNs handle `Vary` on arbitrary custom headers poorly or ignore it entirely. With URI versioning none of this exists, because the version *is* the cache key.
- **Tooling and discoverability are weaker.** API explorers, browser-based testing, and "click the link" docs all work best with URLs. A header is one more thing a developer has to know to set, and one more thing to get wrong.

The honest verdict: header versioning is a reasonable choice when **clean, stable URLs matter to you** (for example, you publish resource links that you want to remain valid across versions) and when you control the clients well enough that the invisibility is not a footgun (internal services, an official SDK that always sets the header). For a sprawling public API consumed by people typing `curl` by hand, the invisibility is a liability. GitHub historically used a header (`X-GitHub-Api-Version`) for its REST API precisely because they wanted stable URLs and a controlled, SDK-mediated client base — more on that in the case studies.

### The caching trap, in detail — why `Vary` is not optional

The caching cost deserves more than a bullet, because it is the single most common way a header- or media-type-versioned API silently corrupts responses, and the failure is invisible until a customer reports nonsense. Let me make the mechanism concrete.

A shared HTTP cache — your CDN, a reverse proxy like Varnish or nginx, even the browser's own cache — stores responses keyed by *request method plus URL* by default. The premise of HTTP caching is that the same URL returns the same bytes, so the cache can answer a second request for that URL without bothering the origin. That premise is exactly what versioning-by-header breaks: `GET /orders/ord_8f2a` now returns *different bytes* depending on a header the cache is, by default, completely ignoring. So the first request that misses the cache — say a v2 client's — populates the cache entry for `/orders/ord_8f2a` with the v2 body. The next request for the same URL from a *v1* client is served that cached v2 body. The v1 client reads `amount_cents`, finds no `amount`, and you are right back in the opening outage — except this time it is intermittent, depends on cache warmth, and is nearly impossible to reproduce in a test environment with a cold cache.

The fix is the `Vary` response header, which tells every cache between you and the client: *the bytes for this URL depend on these request headers, so key on them too.* `Vary: Api-Version` makes the cache store and serve separate entries per version value. `Vary: Accept` does the same for media-type versioning. This is mandatory, not optional, and getting it wrong is a *data correctness* bug, not a performance one. Three things make it harder than it sounds:

- **You must set `Vary` on every cacheable response from the versioned endpoint**, not just some, and intermediaries only honor what they see — if your origin sets it but a misconfigured proxy strips it, you are exposed again.
- **`Vary` fragments the cache.** Each version value is a separate cache entry, so a resource served at three versions has three times the cache footprint and a third the hit rate per version. With URI versioning the "fragmentation" is the same but it is *explicit and intentional* — `/v1/...` and `/v2/...` are obviously distinct cache keys, and no header coordination is required.
- **Many CDNs handle `Vary` on arbitrary custom headers poorly.** Some normalize or drop unknown request headers before the cache key is computed; some only support `Vary` on a fixed allowlist (`Accept-Encoding`, `Accept-Language`). `Vary` on a custom `Api-Version` header may be silently ignored by an edge that you do not control — which means even a *correct* origin can be defeated by the network. `Vary: Accept` is more widely supported but is itself risky because proxies routinely rewrite `Accept`.

The quantitative point: caching is one of the cheapest latency wins you have. A response served from a nearby CDN edge can return in single-digit milliseconds versus the tens-to-hundreds of milliseconds of a full origin round-trip, and it offloads your origin entirely. URI versioning preserves that win for free because the version is in the cache key. Header and media-type versioning put it at risk, and the failure mode of getting it wrong is not "slower" — it is "wrong bytes to the wrong client." This is a real, recurring reason teams that started with header versioning migrate to URI or dated versions.

## Media-type versioning: the version in the Accept header

Media-type versioning is the strict-REST answer. The version is part of the *media type* the client asks for via the `Accept` header, using a vendor-specific media type (the `vnd.` prefix, registered for vendor-defined formats):

```http
GET /orders/ord_8f2a HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Accept: application/vnd.example.order.v2+json

HTTP/1.1 200 OK
Content-Type: application/vnd.example.order.v2+json
Vary: Accept

{
  "id": "ord_8f2a",
  "amount_cents": 4999,
  "currency": "USD",
  "customer": { "id": "cus_7Yh3", "email": "buyer@example.com" },
  "status": "paid"
}
```

A v1 client asks for `Accept: application/vnd.example.order.v1+json` and gets the old shape; the server echoes the chosen type in `Content-Type`. This is content negotiation used exactly as designed: the client states which *representation* it can handle, and the server picks the matching one or returns `406 Not Acceptable` if it cannot. The structure of the media type — `application/vnd.<vendor>.<resource>.<version>+json` — packs the vendor, the resource, the version, and the underlying syntax (`+json`) into one token.

### Why it is "correct" and why almost nobody loves it

This is the only scheme that does not lie about what a URI is. The order has one URI; `v1` and `v2` are two representations of it, negotiated through `Accept`, which is precisely what `Accept` is for. From a Roy-Fielding-grading-your-thesis standpoint, this is the right answer, and it has a real practical virtue:

- **Per-resource granularity falls out naturally.** Because the media type names the *resource* (`order`), you can version the Order representation independently of the Payment representation. A client can send `Accept: application/vnd.example.order.v2+json` while still using `vnd.example.payment.v1+json` for payments, in the same session, against the same base URLs. URI versioning forces a global `/v2`; media-type versioning is granular by construction.

But the costs are why you rarely see it in the wild:

- **It is obscure.** Most developers have never written a `vnd.` media type by hand and find the syntax baffling. The error mode — forgetting the exact media type and getting a `406`, or getting the *default* representation because they sent `Accept: application/json` — is confusing.
- **Tooling is thin.** OpenAPI can describe multiple media types, but code generators, mock servers, API explorers, and SDK toolchains handle vendor media types unevenly. Many HTTP clients and frameworks make it awkward to set a custom `Accept` and to branch server-side on a parsed media type with version segments.
- **Caching has the same `Vary: Accept` problem** as header versioning, and it is arguably worse because `Accept` is a header that proxies and CDNs already manipulate (they may normalize or strip media-type parameters), so relying on it for versioning is fragile across intermediaries.
- **It is even less visible than a custom header**, for the same reason — it lives in `Accept`, which nobody reads in a log.

One detail worth understanding precisely, because it is the source of the most confusing failure: the `+json` suffix. The media type `application/vnd.example.order.v2+json` declares that the *underlying syntax* is JSON (`+json`) but the *schema* is the vendor-defined, versioned `vnd.example.order.v2`. A naive client that sends `Accept: application/json` is asking for "any JSON" and will be matched by the server's *default* representation — which may or may not be the version it expects. So the failure is not always a clean `406 Not Acceptable`; sometimes it is a silent "you got the default version because you asked too vaguely," which is harder to notice than an outright rejection. A strict server can refuse `Accept: application/json` with a `406` and force clients to name the exact versioned media type, which is safer but even more hostile to casual use. This tension — be strict and reject vague requests, or be lenient and silently default them — is the same visibility-versus-convenience trade you see in every scheme, just expressed through content negotiation. The `406` flow, for completeness:

```http
GET /orders/ord_8f2a HTTP/1.1
Host: api.example.com
Accept: application/vnd.example.order.v9+json

HTTP/1.1 406 Not Acceptable
Content-Type: application/problem+json

{
  "type": "https://api.example.com/problems/unsupported-media-type",
  "title": "No matching representation",
  "status": 406,
  "detail": "Cannot produce 'application/vnd.example.order.v9+json'.",
  "supported": [
    "application/vnd.example.order.v1+json",
    "application/vnd.example.order.v2+json"
  ]
}
```

The honest verdict: media-type versioning is the right tool when you genuinely need **per-resource version granularity** and your clients are sophisticated (or fully SDK-mediated). GitHub used vendor media types (`application/vnd.github.v3+json`) for years and it worked because their clients were largely going through tooling that set the header for them. For most teams, the obscurity and tooling gaps outweigh the theoretical correctness, and they choose URI or dated versions instead.

#### Worked example: the same rename in all three schemes side by side

Here is the *identical* breaking change — `amount` → `amount_cents` plus the structured customer — expressed in URI, header, and media-type versioning, so you can see exactly what a client has to change.

A v2 request in **URI** versioning changes the *path*:

```http
GET /v2/orders/ord_8f2a HTTP/1.1
Host: api.example.com
Accept: application/json
```

A v2 request in **header** versioning changes a *header*, same path:

```http
GET /orders/ord_8f2a HTTP/1.1
Host: api.example.com
Accept: application/json
Api-Version: 2
```

A v2 request in **media-type** versioning changes the `Accept` *media type*, same path, no extra header:

```http
GET /orders/ord_8f2a HTTP/1.1
Host: api.example.com
Accept: application/vnd.example.order.v2+json
```

All three return the same v2 body. The difference is purely *where the version lives* — and that single choice cascades into how visible the version is, how the request is cached and routed, and how granular your versioning can be. The before-and-after figure above shows the URI and media-type forms of this exact change; the table later in the post scores all four schemes across these axes at once. This is the whole decision in one frame: the change is constant, only the carrier of the version differs.

## Date-based / rolling versions: the Stripe model

Now the scheme that the largest, most evolution-heavy APIs converge on, and the one worth understanding in real depth: **date-based versions with server-side request transformation**, popularized by Stripe.

The idea: instead of a small number of named major versions (`v1`, `v2`), each account (and each request, optionally) is pinned to a **version date**, like `2024-04-10`. That date is the snapshot of the API's behavior as of that day. When Stripe makes a breaking change, they assign it a new date; accounts created after that date get the new behavior, and existing accounts keep their pinned date and keep seeing the old behavior — *forever*, until they explicitly upgrade. A request can override the account default per-call with a header:

```http
POST /v1/charges HTTP/1.1
Host: api.stripe.com
Authorization: Bearer sk_test_YOUR_KEY_HERE
Stripe-Version: 2024-04-10
Content-Type: application/x-www-form-urlencoded

amount=4999&currency=usd&source=tok_visa
```

(Note that Stripe still has a `/v1` path prefix — that is a *frozen* major version that has not changed in many years; the *real* versioning happens through the date. This is a useful reminder that schemes compose: a frozen URI major plus a date-based minor is a common pairing.)

### The transformation chain — the part that makes this work at scale

Here is the engineering insight that makes dated versioning viable, and it is genuinely elegant. The server does **not** keep N copies of every endpoint, one per date. That would be unmaintainable across hundreds of breaking changes over a decade. Instead, the server keeps **one** current implementation, plus a chain of small, ordered **version transforms**, each of which knows how to translate between two adjacent versions for the specific thing that changed on that date.

When a request comes in pinned to an old date, the server runs it through the chain to *upgrade* the request to the current shape, executes the single current handler, and then runs the response *back down* the chain to *downgrade* it to the shape that the pinned date expects. Each transform is tiny — it touches only the one field or behavior that changed on its date — and the transforms are composable, so adding a new breaking change means writing one new transform at the head of the chain, not forking an endpoint.

![A graph showing a request pinned to an old date routed through a chain of version transforms up to the current handler and back down to the pinned response shape](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-4.png)

Concretely, take three dates relevant to our Orders resource:

- `2024-01-15`: `amount` is a float in dollars; `customer` is a string id.
- `2024-04-10`: `amount` renamed to `amount_cents` (integer).
- `2024-09-01`: `customer` expanded from a string id to an object.

The *current* handler (today) speaks only the `2024-09-01` shape: `amount_cents` and a structured customer. A client pinned to `2024-01-15` sends and expects the oldest shape. Here is the transform chain in pseudocode, written in the spirit of how Stripe describes it:

```python
# Each transform is small and touches ONE change.
# Transforms are ordered by date; the chain composes them.

class RenameAmountToCents:
    """Change introduced 2024-04-10."""
    def upgrade_request(self, body):
        if "amount" in body:
            body["amount_cents"] = round(body.pop("amount") * 100)
        return body

    def downgrade_response(self, body):
        if "amount_cents" in body:
            body["amount"] = body.pop("amount_cents") / 100
        return body

class ExpandCustomerObject:
    """Change introduced 2024-09-01."""
    def upgrade_request(self, body):
        if isinstance(body.get("customer"), str):
            body["customer"] = {"id": body["customer"]}
        return body

    def downgrade_response(self, body):
        cust = body.get("customer")
        if isinstance(cust, dict):
            body["customer"] = cust["id"]   # collapse back to the id string
        return body

# Ordered newest-last so we can walk forward to upgrade,
# backward to downgrade.
TRANSFORMS = [RenameAmountToCents(), ExpandCustomerObject()]

def handle(request, pinned_date):
    # Pick the transforms NEWER than the client's pinned date.
    active = [t for t in TRANSFORMS if t.date_after(pinned_date)]
    body = request.body
    for t in active:               # upgrade: oldest -> current
        body = t.upgrade_request(body)
    response = current_handler(body)   # ONE current implementation
    for t in reversed(active):     # downgrade: current -> pinned
        response = t.downgrade_response(response)
    return response
```

A request pinned to `2024-01-15` activates *both* transforms: its request body's `amount` is multiplied to `amount_cents` and its string `customer` is wrapped into an object on the way *in*; the current handler runs once; and on the way *out* the response's `amount_cents` is divided back to `amount` and the customer object is collapsed back to the id string. A request pinned to `2024-09-01` activates *zero* transforms — it already speaks the current shape and passes straight through. This is why the model scales: the cost of a new breaking change is one small transform, not a forked endpoint, and the current handler never has to know that old versions exist.

#### Worked example: a dated request through the transform chain

Trace a single `GET /v1/orders/ord_8f2a` from a client pinned to `2024-01-15`, step by step.

1. The request arrives. The server reads the account's pinned version (`2024-01-15`) — or a `Stripe-Version` header if the client overrode it per-request.
2. The router selects the active transforms: both `RenameAmountToCents` (after 2024-04-10) and `ExpandCustomerObject` (after 2024-09-01) are newer than the pinned date, so both are active. (For a `GET`, the request-body transforms are no-ops; they matter on writes, but the *response* transforms are what we care about here.)
3. The current handler loads the order and produces the *current* representation: `{"amount_cents": 4999, "customer": {"id": "cus_7Yh3", "email": "buyer@example.com"}, ...}`.
4. The response walks *back down* the chain. `ExpandCustomerObject.downgrade_response` collapses `customer` to the bare string `"cus_7Yh3"`. `RenameAmountToCents.downgrade_response` turns `amount_cents: 4999` into `amount: 49.99`.
5. The client receives exactly the `2024-01-15` shape it has always seen: `{"amount": 49.99, "customer": "cus_7Yh3", ...}`. It has no idea the data was ever shaped any other way.

The on-the-wire response, for a client pinned to the oldest date:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Stripe-Version: 2024-01-15

{
  "id": "ord_8f2a",
  "amount": 49.99,
  "currency": "USD",
  "customer": "cus_7Yh3",
  "status": "paid"
}
```

The cost of this approach is real but bounded: you must write and *test* every transform (a test that round-trips a body through `upgrade` then `downgrade` should return the original is a good invariant), and a long chain adds a little per-request CPU. But it is the only scheme that lets an API make *hundreds* of breaking changes over a decade while letting any given client pin to a single day and never migrate involuntarily. That is why it is the gold standard for large, long-lived APIs — and why it is overkill for an API with three endpoints and one internal client.

## Not versioning at all: additive-only and tolerant readers

Now the option I want you to take most seriously, because it is the one teams skip too fast: **do not version. Make every change additive, and require your clients to be tolerant readers.**

A *tolerant reader* (the term comes from the robustness principle — "be conservative in what you send, be liberal in what you accept") is a client that ignores fields it does not recognize and does not break when new ones appear. If every client is a tolerant reader, then **adding** to the response is always safe: new clients use the new field, old clients ignore it, and you never need a v2. The discipline on the *server* side is correspondingly strict: you only ever *add* — new optional response fields, new optional request parameters, new endpoints, new enum values that old clients can safely treat as "unknown." You never rename, never remove, never change a type, never make an optional thing required, never tighten validation. When you would have renamed `amount` to `amount_cents`, you instead *add* `amount_cents` and keep `amount` populated forever (or for a very long deprecation), and you document `amount` as the legacy field.

This is the model **Google's API design guide (AIP)** strongly favors for many services, and it is essentially the *only* model GraphQL offers in practice. GraphQL has no built-in URL/header versioning; the recommended evolution strategy is additive — add new fields and types, mark old ones `@deprecated`, and let clients (who already select exactly the fields they want) simply stop asking for the deprecated ones. Because a GraphQL client requests a precise field set, *removing* a field is the only breaking move, and additive growth plus deprecation covers almost everything.

![A decision tree for whether and how to version, branching from additive non-breaking change into ship-it-now versus expand-contract, and from breaking change into URI, dated, and media-type options](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-3.png)

### When not-versioning is viable — and when it is not

The decision tree above is the heart of the post: a change is either compatible (ship it; no version) or breaking, and only the breaking branch reaches the versioning schemes at all. Most changes live on the left branch. But the additive-only path has hard preconditions, and it is dishonest to pretend otherwise.

It is viable when:

- **You can enforce tolerant readers.** If you control the clients (internal services, your own SDK) you can guarantee they ignore unknown fields. If your data format and codegen ignore unknowns by default (Protobuf does; many JSON deserializers do *not* unless configured), you are most of the way there.
- **Your changes really are mostly additive.** New features tend to add fields and endpoints, which additive-only handles beautifully. If your domain forces frequent fundamental reshaping, additive-only will not save you.
- **You can tolerate carrying legacy fields.** The cost of additive-only is *response bloat and clutter*: you keep `amount` next to `amount_cents`, the old flat customer next to the new object, indefinitely. Over years this accretes cruft, and "remove a field" — the one thing additive-only cannot do safely — becomes the change you can never make, which is its own kind of un-killable v1.

It is *not* viable when:

- **You cannot make your clients tolerant.** A public API consumed by clients that do strict schema validation (rejecting unknown fields) cannot even add a field safely — adding `amount_cents` would break a client that validates against a closed schema. This is more common than people expect; a generated client with a strict deserializer will throw on an unrecognized field.
- **A change is genuinely incompatible and cannot be expressed additively.** Changing the *meaning* of an existing field, or being forced to remove one for legal/security reasons, cannot be done additively. Then you are back on the breaking branch and you must version.

The honest verdict: not-versioning is the *best* outcome when you can get it, because it has the lowest standing cost — one URL, one code tree, no parallel maintenance. Default to it. Reach for an explicit scheme only when a change is breaking and additive expression has failed.

#### Worked example: the same rename, done additively, with no version at all

It is worth showing the *exact* additive version of our running change, because the contrast with the versioned approaches is the whole argument. We want `amount` (dollars) to become `amount_cents` (integer minor units), and the flat `customer` string to become an object. Versioning gives a v1 caller one shape and a v2 caller another. Additive-only refuses to fork: it makes the *single* response carry both, forever.

Day 0, the v1 response, the only shape that exists:

```json
{
  "id": "ord_8f2a",
  "amount": 49.99,
  "currency": "USD",
  "customer": "cus_7Yh3",
  "status": "paid"
}
```

The additive change *adds* the new representations alongside the old, keeping every existing field exactly where it was:

```json
{
  "id": "ord_8f2a",
  "amount": 49.99,
  "amount_cents": 4999,
  "currency": "USD",
  "customer": "cus_7Yh3",
  "customer_detail": { "id": "cus_7Yh3", "email": "buyer@example.com" },
  "status": "paid"
}
```

Trace who is affected. An old client that reads `order.amount` and `order.customer` is *completely unaffected* — both fields are still present, still hold the same values, still mean the same thing. It silently ignores `amount_cents` and `customer_detail`, which it has never heard of, because it is a tolerant reader. A new client reads `amount_cents` and `customer_detail` and gets the modern shape. There is **one URL, one code path, one response, zero versions**, and *nobody broke*. That is the prize, and it is why additive-only is the default for the teams that can sustain it.

Now the catch, stated honestly. Notice what we could *not* do: we could not rename `amount` to `amount_cents` (we had to *keep* `amount`), we could not collapse the two `customer` fields into one (we had to *keep* the string), and we cannot remove either legacy field without breaking the old client. Over years, a popular resource accumulates these doublets — `amount` and `amount_cents`, `customer` and `customer_detail`, `created` and `created_at` — and the response gets cluttered and a little embarrassing. The deprecation tooling helps: you mark `amount` deprecated in the docs and OpenAPI spec, you may emit a `Deprecation` header, and you wait. But "wait" can mean forever if a client you cannot patch still reads the old field. The additive-only world trades the *parallel-version* tax for a *carried-legacy-field* tax — and which tax is cheaper depends entirely on how many of those doublets you accumulate and whether you can ever clear them. For most APIs the additive tax is far cheaper, which is why this is the recommended default. But it is a tax, not a free lunch, and pretending otherwise is how teams end up with hundred-field responses they are afraid to touch.

This is also the point where additive-only and dated versioning meet philosophically. Dated versioning lets you *present* a clean response per version (the transform downgrades to exactly the old shape, with no doublets visible to the old client) at the cost of a transform chain; additive-only keeps one literal response with all the doublets visible at the cost of clutter. Both avoid forking the handler. The dated model is "additive-only with a presentation layer that hides the legacy from new clients" — which is exactly why it scales to so many changes.

## Granularity: global vs per-resource vs per-field

A dimension that cuts across all the schemes is **how big a thing a single version covers** — your version's blast radius.

![A vertical stack of versioning granularity from global covering the whole API down through per-resource and per-field, with client pinning and default policy beneath](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-5.png)

- **Global versioning** versions the entire API at once: there is one `/v1` and one `/v2`, and *everything* moves together. This is what URI versioning pushes you toward, and it is the coarsest. The upside is conceptual simplicity (a client picks one version for the whole surface). The downside, as we saw, is that resources that did not change get dragged along anyway.
- **Per-resource versioning** lets each resource carry its own version: Orders can be at v2 while Payments stays at v1. Media-type versioning supports this naturally (the resource name is in the media type), and dated versioning supports it implicitly (a given date's behavior is per-feature, not per-whole-API). The upside is a much smaller blast radius — only callers of the changed resource have to think about it. The downside is more bookkeeping and the possibility of a confusing matrix of "which version of which resource."
- **Per-field versioning** is the finest grain: a single field changes and only that field's consumers are affected. This is what additive-only and dated transforms effectively achieve — the `amount_cents` transform touches *one field*, and a client pinned before that date is affected by *only* that one thing. GraphQL's `@deprecated` is per-field by design.

The trade-off is the usual one between coarse and fine. Coarse (global) is simple to reason about but forces unrelated migrations and tempts gratuitous version bumps. Fine (per-field) has the smallest blast radius and the least forced migration, but demands more sophisticated machinery (a transform chain, or a strict tolerant-reader discipline) and more bookkeeping. Most teams land in the middle — per-resource — or go all the way to per-field via the dated-transform model if they are large enough to justify the infrastructure.

## How clients select a version, and how servers route on it

A version scheme is only half the design. The other half is the *plumbing*: how a client tells you which version it wants, and how your server turns that into the right code path.

![A graph showing the server resolving a version from the path or a header or the Accept media type and falling back to a default before dispatching to the correct handler](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-8.png)

On the **client** side, selection follows the scheme: a path prefix (URI), a custom header (header), an `Accept` media type (media-type), or an account-level pin plus an optional per-request override (dated). The cleanest production pattern is the one Stripe uses: pin a version at the *account* or *integration* level (set once, stored server-side) so that every call is consistent without the client having to remember to set anything, with a per-request header available for testing and gradual migration. This gets you the safety of an explicit pin without the footgun of "the developer forgot the header on one of forty call sites."

On the **server** side, you want a *single* version-resolution step at the front of the request, before any business logic. The router reads the version from wherever the scheme puts it, falls back to the default if none is supplied, and then dispatches. A small framework-agnostic sketch:

```python
def resolve_version(request):
    # 1. URI prefix wins if present: /v2/orders
    if m := match_version_prefix(request.path):   # -> "2" or None
        return ("uri", m)
    # 2. Explicit header
    if v := request.headers.get("Api-Version"):
        return ("header", v)
    # 3. Versioned media type in Accept
    if v := parse_vendor_media_type(request.headers.get("Accept", "")):
        return ("media_type", v)
    # 4. Nothing supplied -> default policy (see next section)
    return ("default", DEFAULT_VERSION)

def dispatch(request):
    scheme, version = resolve_version(request)
    handler = ROUTES[(request.resource, version)]
    return handler(request)
```

Two things make this robust. First, resolve the version *once*, at the edge (often in the gateway — see [API gateways: routing, auth, rate limiting, and the BFF pattern](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) for where this sits in the request lifecycle), and pass it down as an explicit parameter; do not re-parse it in three places. Second, make the *unknown-version* path explicit: if a client asks for `Api-Version: 9` and you have no v9, return a clear `400` with a `problem+json` body listing the supported versions, not a `404` or a silent fallback to default. An honest error here saves hours of confused debugging:

```json
{
  "type": "https://api.example.com/problems/unsupported-version",
  "title": "Unsupported API version",
  "status": 400,
  "detail": "Requested version '9' is not supported.",
  "supported": ["1", "2"]
}
```

A subtle but important rule for the resolution order above: decide *one* canonical place the version comes from, and treat the others as overrides or errors, rather than letting three schemes coexist ambiguously. If a request arrives with `/v2` in the path *and* `Api-Version: 1` in a header, what wins? There is no universally right answer, but there *is* a universally right requirement: the behavior must be defined, documented, and consistent. The cleanest rule is "the most specific carrier wins, and a contradiction is a `400`" — but whatever you pick, write it down. An API that silently resolves contradictory version signals differently across endpoints is worse than one with a slightly awkward but predictable rule.

On the **SDK** side, the kindest thing you can do for callers is to make the version a *configuration value set once*, not something threaded through every call. A good SDK pins the version at client construction and attaches it automatically:

```javascript
// The version is set ONCE, at client construction.
// Every request the client makes carries it automatically.
const client = new ExampleClient({
  apiKey: "sk_test_YOUR_KEY_HERE",
  apiVersion: "2024-04-10",   // pin explicitly; do not float
});

// Callers never think about the version again.
const order = await client.orders.retrieve("ord_8f2a");
```

This collapses the whole "did I remember to set the header?" footgun into a single line of configuration, and it makes the version an explicit, reviewable part of the integration rather than an implicit default that drifts. It is also where you steer new integrations onto the modern contract: the SDK's default `apiVersion`, shipped in the latest SDK release, is the version most new callers will land on. Pinning explicitly (rather than letting the SDK float to "latest") is the safe default for the same reason server-side pinning is: a `npm update` of the SDK should never silently change which API contract your code talks to. The version is part of your contract; treat it like a dependency you upgrade deliberately, not a moving target.

## Default-version policy: pin vs latest

Here is a decision that looks tiny and turns out to govern the safety of your entire migration story: **what version does a client get when it does not specify one?** There are two policies, and they are opposites.

**Default to *latest*.** A client that sends no version gets whatever is newest. This is convenient for getting started ("just call `/orders`, you get the current shape") and keeps casual usage on the modern contract. But it is a *trap* for any client you cannot patch, because the day you ship a new breaking version, every unversioned client silently jumps to it — and breaks. This is the exact failure mode of our opening outage, generalized: "default to latest" means "every breaking change is a breaking change for everyone who did not explicitly opt out." It punishes the clients least equipped to keep up.

**Default to *pinned* (oldest stable, or the version in effect when the client started).** A client that sends no version gets a *fixed* version that never changes underneath it. New clients can opt into newer versions explicitly. This is the policy that makes long-lived APIs safe: a breaking change only affects a client when that client *chooses* to move to it. Stripe's account-pinning is exactly this — your integration is frozen at the date you built it until you deliberately upgrade. The cost is that brand-new integrations might land on an old default unless you steer them (so you set new accounts' pin to the current date at creation time).

> **The principle: an unversioned request must never silently change behavior. Default to a *pinned* version so that "doing nothing" means "keep working," not "ride the latest breaking change." The only safe default is the one that protects the client who is not paying attention.**

The right production setup, again following the large-API pattern: pin each account at the version current *when it was created*, default brand-new accounts to the newest version, and let any account upgrade explicitly. New integrations get the modern contract; old integrations never move involuntarily. That single policy does more for migration safety than any choice of URI-vs-header.

## Working the problem: choosing a scheme for the Payments API, then stress-testing it

Let me reason through the actual decision for our Payments & Orders API the way you would at a design review, then deliberately break the choice to see where it bends. This is the part that separates a scheme you picked from a scheme you *understand*.

**The setup.** We have a public API. Clients are a mix of: our own web app (we control it, deploy daily), an official SDK (we control it, but customers upgrade on their own schedule), and a long tail of third-party integrations and mobile apps we *cannot* patch and whose authors we mostly cannot reach. We change the API a few times a quarter; most changes are additive, but maybe twice a year we have a genuinely breaking one (a field re-meaning, a structural reshape, a removal forced by a compliance requirement). We run a CDN in front of read-heavy endpoints.

**The reasoning.** Start at the top of the decision tree: can we make most changes additively? Yes — most of our changes are new fields and endpoints, and we can make our own SDK a tolerant reader. So the *default* is no versioning: ship additive changes in place, carry the occasional legacy doublet, deprecate in docs. That handles the common case for free. The question is only what to do with the two-breaking-changes-a-year. For those, we have unpatchable public clients, we are read-heavy behind a CDN (so caching simplicity is worth real money and real correctness), and our breaking changes tend to be per-feature rather than whole-surface. The CDN correctness concern pushes us *away* from header and media-type versioning (the `Vary` trap is a data-corruption risk we do not want on our read path). The per-feature nature of our breaks pushes us *away* from coarse global URI bumps (we do not want to drag all of `/payments` into a new version because one Orders field changed). That combination — unpatchable clients, caching matters, changes are per-feature, many small breaks over time — is *exactly* the profile dated versioning was built for. So: a frozen `/v1` URI major (so the cache key is stable and the URL is honest), plus account-pinned date versions with a transform chain for the per-feature breaks, defaulting unversioned requests to a pinned date. We get cache-friendly URLs, per-field granularity, and no forced migration. The cost we accept is building and testing the transform chain.

Now stress-test that choice. A design that has not been attacked is just a guess.

- **What if a client retries on a timeout?** Versioning is orthogonal to idempotency, but the interaction bites: a retried write must hit the *same* version the original did, or a transform could reshape the retry differently and the idempotency key could match a body that no longer looks the same. Pin the version per-account (not per-request) so a retry naturally carries the same version, and key idempotency on the *post-transform* (current-shape) body so two retries at the same pinned date collapse identically. (See [idempotency keys: safe retries and the exactly-once illusion](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) for the retry contract itself.)
- **What if two transforms disagree about a field?** A long chain is composable only if each transform touches a *disjoint* concern. If two transforms both rewrite `amount`, their order matters and a bug in one corrupts the other. The discipline: one transform per breaking change, each touching the minimum surface, with a round-trip property test (`downgrade(upgrade(x)) == x` for the relevant shape) that runs in CI for every transform. This is the invariant that keeps the chain honest.
- **What if a partner pins to the oldest date for three years?** Then the oldest transforms in the chain run on every one of their requests forever, and you can never delete those transforms. This is the un-killable-v1 problem expressed in the dated model: the transform *code* is the liability, not a forked endpoint, which is cheaper — but it is still code you maintain and test indefinitely. Plan for it: contractually commit paid partners to a maximum version age, and keep the transforms small enough that carrying an old one is annoying rather than expensive.
- **What if the breaking change is whole-surface after all** — a true platform redesign, new auth model, new resource model? Then the per-feature dated model is the wrong grain, and you cut a real new URI major (`/v2`) and run the two majors side by side with their own sunset. Schemes compose: you can be additive within a major, dated for per-feature breaks within a major, and bump the URI major for a genuine platform jump. Use the coarse tool only for the coarse change.
- **What if the response is 10× bigger than planned** because additive doublets accumulated? Then you have hit the additive tax, and the fix is the dated presentation layer (downgrade hides legacy fields from clients that do not need them) plus an honest deprecation campaign to finally clear the oldest doublets — which you can only do once the last reader of the legacy field has moved, which loops you back to the un-killable-version problem. There is no escaping that every evolution strategy eventually meets the client who will not move; the strategies differ only in how cheaply they let you carry that client.

The decision survives the stress test, with its costs made explicit: build a transform chain, hold the one-concern-per-transform discipline, test the round-trip invariant, and accept that some old transforms live forever. That is a fair trade for a large public API. For a small internal API with one caller, *the same stress test would reject this design as massive over-engineering* — you would default to additive-only and never build the chain at all. The scheme is right only relative to the forces; name the forces, then choose.

## The cost of every live version

We have alluded to it throughout; now let us name it directly, because it is the consideration that should make you reluctant to version at all.

![A timeline of one version's life from shipping as the default through the period of two live versions, deprecation, a set sunset date, and finally removal with a 410 Gone](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-6.png)

Every version you keep alive is a standing liability with several distinct costs:

- **Maintenance.** Every bug fix, every security patch, every new feature has to be considered against *all* live versions. A fix to the v2 Orders serializer may need a parallel fix in the v1 serializer, or in the v1 transform. Two versions is roughly two code paths to keep correct; the cost grows with each version (sublinearly if you use transforms, near-linearly if you fork endpoints).
- **Testing.** Your test matrix multiplies by the number of versions. A change to Orders now needs verification against v1 *and* v2 shapes. Contract tests (covered in [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2)) run per-version. The transform-chain model adds the round-trip invariant test per transform.
- **Cognitive load and support.** Every engineer who touches Orders has to hold "which versions exist and how do they differ" in their head. Every support ticket starts with "which version are you on?" Documentation forks. Onboarding gets longer.
- **The version you can never kill.** This is the worst cost and the one that surprises teams. You announce a sunset, you send `Deprecation` and `Sunset` headers, you chase the laggards — and one important partner (a big customer, a regulator-facing integration, an embedded device with no update path) never migrates. Now you cannot remove v1 without breaking *them*, and the business will not let you break them. So v1 lives on, indefinitely, as dead weight you maintain forever. Plan for this from day one: the right move is to make migration as cheap as possible for clients (a clear changelog, a migration guide, an SDK that abstracts the version, generous sunset windows) and to set, contractually for paid partners, that old versions *will* be retired on a schedule. Hope is not a deprecation strategy.

The math is simple and unforgiving: the *steady-state* cost of your API is roughly proportional to the number of live versions you maintain, times the rate of change. Additive-only keeps that count at one. Dated transforms keep the *code* count at one (one handler) while supporting many logical versions, which is its whole appeal. URI/header/media-type versioning with forked handlers multiply the count directly. Choose accordingly: the cheapest version is the one you never had to create.

## Comparison tables

Two tables to crystallize the decision. The first scores the four explicit-versioning strategies plus not-versioning across the axes that matter operationally.

| Strategy | Where the version lives | Visibility | Caching | Granularity | Standing cost | Best when |
| --- | --- | --- | --- | --- | --- | --- |
| **URI** (`/v2/orders`) | Path | High — in the URL | Easy — path is the cache key | Whole surface (coarse) | Per version (forked) | Public API, unknown clients, true v1→v2 jump |
| **Header** (`Api-Version: 2`) | Custom header | Low — easy to miss | Needs `Vary`; fragile on CDNs | Per request | Per version (forked) | Clean URLs matter; controlled/SDK clients |
| **Media-type** (`vnd...v2+json`) | `Accept` media type | Low — obscure | Needs `Vary: Accept`; fragile | Per resource (fine) | Per resource type | Need per-resource grain; sophisticated clients |
| **Dated / rolling** (`2024-04-10`) | Account pin + header | Medium — logged date | Cache per version; well-understood | Per field (finest) | One handler + a transform chain | Large, long-lived API, many breaking changes |
| **No version** (additive) | Nowhere | None — invisible | Best — one URL | Per field | Lowest — one tree | Tolerant clients, mostly-additive change |

The second table is the one that should govern your *daily* decisions: for each kind of change, what does "version everything" cost you versus "stay additive"?

![A matrix comparing version-everything against additive-only across adding an optional field, renaming, removing, and tightening a validation rule](/imgs/blogs/versioning-strategies-uri-header-media-type-and-not-versioning-7.png)

| Change | Version everything | Additive-only | Verdict |
| --- | --- | --- | --- |
| Add optional response field | New version (overkill) | Just add it; tolerant readers ignore it | Additive wins, easily |
| Add optional request param | New version (overkill) | Just accept it; default when absent | Additive wins, easily |
| Rename a field | New version, clean | Add new name, keep old, deprecate old | Additive wins *if* you can carry the legacy field |
| Remove a field | New version forces the cut | Deprecate; you may never truly remove it | Versioning wins *if* removal is mandatory |
| Change a field's type/meaning | New version, safe | Cannot be done additively | Versioning required |
| Tighten validation | New version, safe | Can break existing valid requests | Versioning required |

The pattern is clear: additive-only handles the *common* changes (adding things) for free and is the right default, while explicit versioning earns its cost only for the genuinely incompatible changes (removing, retyping, re-meaning, tightening). Match the tool to the change.

## Case studies

A few real, accurately characterized examples — because these patterns were forged in production, not in a style guide.

**Stripe — dated versions with request transformation.** Stripe pins each account to a version *date* (for example `2024-04-10`), exposed and overridable via the `Stripe-Version` request header, with the major version frozen in the `/v1` path prefix. Their public engineering writing describes the model this post walks through: a single current implementation plus a chain of small version-change modules that upgrade incoming requests to the current shape and downgrade outgoing responses back to the client's pinned version. This is what lets Stripe ship breaking changes routinely while letting integrations stay pinned for years without forced migration. It is the gold standard for a large, long-lived, breaking-change-heavy API, and the reference design for "how do I support many versions without forking my endpoints N times."

**GitHub — media-type then header versioning.** GitHub's REST API historically encoded the version in a vendor media type (`Accept: application/vnd.github.v3+json`) — a real, prominent example of media-type versioning working at scale, made tolerable by clients that mostly went through GitHub's official libraries which set the header for you. GitHub later introduced a dated `X-GitHub-Api-Version` *header* for its REST API (date-stamped versions selected by a header), an evolution toward the dated model. GitHub also runs a *GraphQL* API alongside REST, where evolution is additive-plus-`@deprecated` rather than versioned — a clean illustration that the same company picks different evolution strategies for different paradigms by force, not fashion. (Always check GitHub's current docs for the exact header and date values; the *strategy* is the durable lesson.)

**Google / AIP — additive-first.** Google's public API Improvement Proposals (AIP) guidance leans hard toward backward-compatible, additive evolution within a major version: add fields, do not remove or repurpose them, and reserve a new major version for genuinely incompatible redesigns. Combined with Protobuf — whose wire format ignores unknown fields by design, making clients tolerant readers automatically — this makes additive-only the default and a new version the rare exception. It is the clearest large-scale endorsement of "don't version if you can avoid it."

**GraphQL — no versioning by design.** The GraphQL specification and the community's accepted practice provide *no* URL/header/media-type versioning mechanism. Because a client requests an exact set of fields, adding fields and types never affects existing clients, and the recommended evolution path is additive growth plus the `@deprecated` directive on fields and enum values being phased out. This is the purest expression of "tolerant readers + additive change = no versioning needed," and it is viable precisely because the client's field selection makes additive change inherently safe. We cover GraphQL's evolution model in more depth in the paradigm posts; the takeaway here is that the *right* answer to "how should I version?" is sometimes "you have designed yourself out of needing to."

For the distributed-systems view of how all this plays out across services and schema registries at scale — schema compatibility checks in CI, registry-enforced evolution rules, and the org-wide governance of breaking changes — see the system-design companion, [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale). This post owns the wire-level contract for a single API; that one owns the fleet-wide evolution machinery.

## When to reach for each (and when not to)

A decisive recommendation, because every choice here is a trade-off and the worst outcome is choosing by reflex.

- **First, try not to version.** If the change is additive and your clients are (or can be made) tolerant readers, ship it in place. Do *not* cut a `/v2` for a new optional field — you will fragment your surface for nothing. This is the right default for internal APIs, Protobuf/gRPC services, and GraphQL.
- **Reach for URI versioning** when you have a public API with many unknown, unpatchable clients and a genuine whole-surface v1→v2 jump, and when ops simplicity (trivial routing and caching) and visibility matter more than per-resource granularity. Do *not* use it if your changes are per-resource and you would be dragging unchanged resources into a new version for no reason.
- **Reach for header versioning** when clean, stable URLs are a hard requirement and you control your clients well (an SDK that always sets the header, internal services). Do *not* use it for a hand-`curl`-ed public API where the invisibility will cause developers to silently call the wrong version.
- **Reach for media-type versioning** only when you genuinely need per-resource granularity *and* your clients are sophisticated or fully SDK-mediated. Do *not* reach for it because it is "the RESTful way" if your team and tooling will fight the vendor media types the whole way — correctness you cannot operate is not correctness.
- **Reach for dated / rolling versions** when you are a large, long-lived API expecting many breaking changes over years and you can invest in the transform-chain infrastructure. Do *not* build a transform chain for an API with three endpoints and one internal caller; it is a sophisticated answer to a problem you do not have yet.
- **Whatever you choose, default to a *pinned* version, not latest**, and make migration cheap (changelog, migration guide, SDK that hides the version, generous sunset windows). Do *not* default unversioned requests to latest if you have clients you cannot patch — that turns every breaking change into an outage.

And a few flat "don'ts" that span all schemes: don't version for a change that could be additive; don't keep a version alive with no sunset plan (every version needs a planned death, even if a partner forces you to miss the date); don't default to latest for public APIs; don't mix three schemes in one API so callers cannot tell where the version lives; don't forget `Vary` on header/media-type schemes or your cache will betray you.

## Key takeaways

- **Version only when you must.** A change that can be made additively (new optional field, new endpoint) needs no version; reach for a scheme only when a change is genuinely breaking and cannot be expressed additively.
- **The four schemes differ mainly in *where the version lives* and what that costs.** URI (path) is visible and cache-trivial but couples the whole surface; header is clean-URL but invisible and needs `Vary`; media-type is RESTful and per-resource but obscure; dated is the gold standard for big APIs via a transform chain.
- **Not versioning is often the best answer.** Additive-only plus tolerant readers (Google/AIP, Protobuf, GraphQL) has the lowest standing cost — one URL, one code tree — and should be your default whenever you can enforce tolerant clients.
- **Dated versioning scales because of the transform chain.** One current handler plus small per-change transforms that upgrade requests and downgrade responses lets you support hundreds of logical versions without forking endpoints.
- **Granularity is a real axis.** Global (URI) is coarse and forces unrelated migrations; per-resource (media-type) and per-field (dated/GraphQL) shrink the blast radius at the cost of more machinery.
- **Default to a pinned version, never to latest.** An unversioned request must never silently ride a breaking change; pin clients at the version current when they started and let them upgrade deliberately.
- **Every live version is a standing cost** — maintenance, testing, support, and the un-killable v1 a partner refuses to leave. The cheapest version is the one you never had to create; plan each version's sunset before you ship it.

## Further reading

- [Backward and forward compatibility: the rules of safe change](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) — the prerequisite to this post: which changes are breaking and which are not, and how to make a change additively so you never have to version.
- [Deprecation and sunset: retiring an API humanely](/blog/software-development/api-design/deprecation-and-sunset-retiring-an-api-humanely) — the `Deprecation`/`Sunset` headers, migration comms, and how to actually retire a version (the back half of every versioning decision).
- [Content negotiation, media types, and representations](/blog/software-development/api-design/content-negotiation-media-types-and-representations) — the resource-vs-representation distinction that makes media-type versioning "correct," and how `Accept`/`Content-Type` negotiation works.
- [What is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) — the series hub: the API as a contract and a product you design for callers you will never meet.
- [The API design playbook: a review checklist from first endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) — the capstone checklist that ties versioning into the whole design-review process.
- [Schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) — the distributed-systems companion: schema registries, compatibility checks in CI, and org-wide evolution governance across a fleet of services.
- **Stripe API versioning** — Stripe's official API reference and engineering blog on dated versions and request transformation (the canonical real-world implementation of the dated model).
- **Google AIP (API Improvement Proposals)** and the **GraphQL specification** — the two clearest large-scale endorsements of additive, no-versioning evolution; read them together to see why "don't version" is a serious strategy, not a cop-out.

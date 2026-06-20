---
title: "Caching: ETags, Cache-Control, Conditional Requests, and Invalidation"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make your API faster and cheaper by never recomputing what hasn't changed — Cache-Control directives in depth, ETags and conditional requests that return a bodyless 304, If-Match optimistic concurrency, the Vary header, and the hard problem of invalidation, all on a Payments and Orders API."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "caching",
    "etag",
    "cache-control",
    "conditional-requests",
    "cdn",
    "rfc-9111",
    "performance",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-1.png"
---

A merchant dashboard on our Payments & Orders platform polls `GET /orders/ord_7Hk2` every thirty seconds to show a customer-service rep the live state of one order. The order was placed two days ago. It has not changed in those two days and it will not change again — it shipped, it was paid, it is done. And yet, every thirty seconds, twenty-four hours a day, our API serializes the same eighteen-kilobyte JSON document, gzips it, ships it across the wire, and the origin database executes the same five-table join to assemble a row that is byte-for-byte identical to the one it assembled thirty seconds ago. Multiply that by ten thousand reps with dashboards open, and the most expensive thing our API does all day is answer a question whose answer never changed.

That is the problem caching solves, and HTTP has solved it well for decades — so well that a correctly designed REST API gets most of the win for free, by setting a few response headers, while an RPC or GraphQL API that tunnels everything through `POST` has to reinvent the whole machinery by hand. The win comes in two flavors that are easy to confuse. The first is *freshness*: if the server promises a response is good for sixty seconds, a cache can serve it for sixty seconds without asking anyone — zero round-trips, zero origin work. The second is *validation*: when the promised time runs out, instead of re-downloading eighteen kilobytes, the client asks "is what I have still current?" and the server answers in a few bytes — `304 Not Modified`, no body — if nothing changed. Freshness avoids the trip entirely; validation makes the trip cheap.

![A vertical stack of the four caching layers from the browser cache down through the CDN edge, reverse proxy, and server cache to the slow origin database](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-1.png)

This post is the detailed, wire-level guide to making that happen. We will define every term from scratch — what a *validator* is, what a *conditional request* is, the genuinely crucial difference between `no-cache` and `no-store`, why `private` and `public` mean what they mean — and then go deep: `Cache-Control` directive by directive, `ETag` strong versus weak, the `If-None-Match` → `304` exchange byte by byte, `If-Match` for optimistic concurrency that prevents a lost update on a `PUT`, the `Vary` header that keeps a cache from serving an English page to a French client, and the genuinely hard problem of invalidation — why "there are only two hard things in computer science: cache invalidation and naming things" is a joke that stops being funny the first time a customer sees a stale balance. By the end you will be able to look at any `GET` endpoint and say precisely what its caching headers should be, why, and what breaks if you get them wrong. This is the operability layer of the series' spine — *what can the caller safely assume about how long this answer stays true, and can I change that answer later without serving someone a lie?* If you want the HTTP fundamentals underneath this, the companion post [HTTP for API designers: methods, status codes, and headers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers) lays the groundwork, and the broader frame lives in [What is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems).

## 1. Why cache at all: latency, load, cost, and bandwidth

Before we touch a single header, let us be precise about *why* we are doing this, because the four reasons pull in slightly different directions and the right caching design depends on which one is biting you.

**Latency.** A cache that lives closer to the client answers faster. A response served from the browser's own memory cache is effectively instantaneous — no network at all. A response served from a CDN edge node in the same city as the user might take five to twenty milliseconds; the same response from an origin three thousand kilometers and one database query away might take two hundred to four hundred milliseconds. For an interactive dashboard, that is the difference between "instant" and "laggy." Latency reduction is about *distance* and *work avoided*, and it is the reason caches are arranged in layers from the client outward.

**Load.** Every request a cache absorbs is a request the origin never sees. If our orders API gets one hundred requests per second for a particular hot order — say, a viral product everyone is tracking — and a CDN with a sixty-second freshness window absorbs all but one of them, the origin sees roughly one request per minute for that order instead of one hundred per second. That is a load reduction of about $100 \times 60 = 6000\times$ on that key. Load reduction protects the origin, and it is what lets a modest backend survive a traffic spike.

**Cost.** Load and cost are linked but not identical. Origin compute costs money — CPU seconds, database IOPS, the egress bandwidth out of your cloud region. CDN bandwidth is typically cheaper than origin egress, and CDN compute (serving a cached object) is nearly free compared to running your application code and a five-table join. If serving one uncached order response costs the origin, very roughly, \$0.0002 in compute plus \$0.00002 in egress, then absorbing 6000 of them at the edge per minute is real money at scale — and the database load you *don't* provision for is often the bigger saving than the line-item bandwidth bill.

**Bandwidth.** This is the one validation specifically targets. Even when a cache *can't* serve a response on its own — because the freshness window expired — it can often avoid re-transferring the body. An eighteen-kilobyte order document that hasn't changed can be confirmed current with a request and a `304 Not Modified` response that together total a few hundred bytes. If the resource is unchanged 95% of the time it is polled, then validation alone cuts the bytes-on-the-wire for those polls by roughly:

$$\text{savings} \approx p_{\text{unchanged}} \times \left(1 - \frac{\text{validation bytes}}{\text{full body bytes}}\right) = 0.95 \times \left(1 - \frac{300}{18000}\right) \approx 0.93$$

— about a 93% reduction in transferred bytes for the polling workload, with the origin still consulted on every poll but doing far less work per poll. That formula is approximate and the exact numbers depend on your payload, but the shape is the point: *validation gives you a big bandwidth win even when freshness gives you none.*

These four reasons map onto the four layers in the figure above. The browser cache is closest and serves a single user; the CDN edge is shared across all users near it; a reverse proxy like Varnish or nginx sits in front of your origin and shares cache across everyone; and a server-side cache (Redis, an in-process LRU) sits behind the application code, caching the *computed* result so even a cache miss at the edge doesn't always hit the database. A request descends through these layers and is answered by the first one that has a fresh copy. The art of HTTP caching is telling each layer, precisely and truthfully, what it is allowed to do.

It helps to put a number on how the layers compound, because the intuition that "a cache helps a bit" badly undersells what a *stack* of caches does. Suppose each layer has its own hit ratio — the fraction of requests it can answer without passing the request down. If the browser cache catches a fraction $h_b$ of requests, the CDN catches a fraction $h_c$ of what's left, the proxy catches $h_p$ of what's left after that, and the server cache catches $h_s$ of the remainder, then the fraction of original requests that actually reach the origin database is:

$$f_{\text{origin}} = (1 - h_b)(1 - h_c)(1 - h_p)(1 - h_s)$$

Plug in modest numbers — say each layer catches half of what it sees, $h = 0.5$ everywhere — and you get $f_{\text{origin}} = 0.5^4 = 0.0625$, meaning only about one request in sixteen ever touches the database. Push the CDN hit ratio up to $h_c = 0.9$ (entirely realistic for a read-heavy public endpoint) and the origin sees a small single-digit percentage of traffic. This multiplicative structure is *why* caching is the highest-leverage performance work on most read-heavy APIs: each layer doesn't add a fixed saving, it multiplies down the residual load, and the layer closest to the user (the browser) is both the cheapest and the first to fire. The corollary is also true and often forgotten — a single uncacheable response that slips through (a missing header, a personalized field that forces `private`) doesn't just lose its own caching, it loses the *product* of every downstream layer's benefit for that request.

One more piece of vocabulary that pays off later: the *hit ratio* itself is the lever you tune with `max-age`. Longer freshness windows raise the hit ratio (more requests fall inside the window) but raise the worst-case staleness too; shorter windows keep data fresher but send more requests down the stack. There is no universally right value — it's a per-resource decision driven by "how stale can this safely be?" — but the framing to carry forward is that `max-age` and hit ratio are two views of the same dial.

## 2. Cache-Control in depth: the directives that govern everything

`Cache-Control` is the response header (it can also appear on requests, but the response form is what governs caching) that tells every cache in the chain what they may do with this response. It is a comma-separated list of directives. Get these right and everything downstream behaves; get them wrong and you either cache nothing (slow, expensive) or cache something you shouldn't (a stale balance, one user's data served to another). Let us go through them.

`max-age=<seconds>` is the freshness lifetime. `Cache-Control: max-age=60` says "this response is fresh for sixty seconds from now; any cache may serve it without asking me, for sixty seconds." After that it is *stale*, which does not mean "delete it" — it means "ask me before you serve it again." Freshness is the zero-round-trip win.

`s-maxage=<seconds>` is `max-age` but *only for shared caches* — CDNs and reverse proxies, not the user's browser. This lets you say "browsers, keep this for five seconds; CDN, keep it for sixty" with `Cache-Control: max-age=5, s-maxage=60`. The shared-cache window can be longer because a CDN serving a slightly-stale order to a thousand users is usually fine, while you want each user's own browser to recheck sooner. `s-maxage` overrides `max-age` for shared caches and is the single most useful directive for tuning CDN behavior on an API.

`public` and `private` answer *who may store this*. `public` means any cache, including shared CDN and proxy caches, may store it. `private` means only a *private* cache — the end user's browser — may store it; shared caches must not. This is a correctness directive, not a performance one. A response that contains one user's order list (`GET /orders` scoped to the authenticated user) must be `private`, because if a shared CDN stored it under the URL `/orders` and then served it to the next user who asked for `/orders`, user B would see user A's orders. `private` is how you say "this is personalized; do not let it leak."

Now the two that everyone confuses, and the difference is genuinely crucial.

`no-cache` does *not* mean "do not cache." It means "you may store this, but you must revalidate with the origin before serving it from cache." A cache holding a `no-cache` response keeps the bytes; on the next request it makes a conditional request (the `If-None-Match` dance we cover below) and only serves its stored copy if the origin confirms it is still current. So `no-cache` is "store it, but never trust it without checking." It is perfect for resources that change unpredictably but are usually unchanged — you keep the body, and most checks come back `304`, so you save bandwidth without ever serving stale data.

`no-store` means "do not store this anywhere, in any cache, at all." No browser cache, no CDN, no proxy. The next request starts from nothing. This is for genuinely sensitive responses — a one-time payment confirmation token, a response containing full card data, anything that must never sit on disk in an intermediary. `no-store` is the heavy hammer. Reaching for it when you meant `no-cache` (or `private`) is a common and expensive mistake: you forfeit all caching, including the cheap validation win, for resources that would have been perfectly safe to store-and-revalidate.

| Directive | May a shared cache store it? | May a browser store it? | What happens on reuse |
| --- | --- | --- | --- |
| `public` | Yes | Yes | Served while fresh; revalidated when stale |
| `private` | No | Yes | Served while fresh by the browser only |
| `no-cache` | Yes (stores) | Yes (stores) | Must revalidate with origin *every* time before serving |
| `no-store` | No (never stores) | No (never stores) | Nothing stored; always a full fetch from origin |

`must-revalidate` tightens the stale rule. Normally a cache *may*, under some conditions (a network outage, an explicit allowance), serve a stale response rather than fail. `must-revalidate` forbids that: once stale, this response must be revalidated, and if the origin can't be reached, the cache must return a `504 Gateway Timeout` rather than serve stale. Use it where stale data is genuinely dangerous — an account balance, an inventory count you sell against. `proxy-revalidate` is the same rule but applies only to shared caches.

`stale-while-revalidate=<seconds>` (RFC 5861) is the directive that makes caching feel instant. It says "once this response goes stale, you may keep serving the stale copy for up to N more seconds *while you revalidate it in the background*." The user gets an instant answer (the stale copy), and the cache quietly refreshes from the origin so the *next* user gets fresh data. `Cache-Control: max-age=60, stale-while-revalidate=30` means "fresh for 60s, then for the next 30s serve the old copy instantly and refresh behind the scenes." It trades a tiny, bounded staleness for the elimination of the latency spike that every cache miss otherwise causes — and, as we'll see, it is also a primary defense against cache stampedes. Its sibling `stale-if-error=<seconds>` lets a cache serve stale content when the origin is returning errors, which is a cheap resilience win.

`immutable` is a promise: "this response will *never* change, so don't even bother revalidating it." It is meant for content at a versioned, content-addressed URL — `/static/app.a1b2c3.js`, where the hash in the filename guarantees the bytes never change. Setting `Cache-Control: max-age=31536000, immutable` tells the browser to serve it from cache for a year without ever sending a conditional request, even on a manual page reload. It is the right answer for fingerprinted static assets and almost never the right answer for an API resource that represents mutable state. Note the trade implied: `immutable` means you have moved invalidation *into the URL* — to change the content you change the URL, which is the versioned-URL strategy we revisit in the invalidation section.

![A matrix mapping Cache-Control directives like public, private, no-cache, no-store, and stale-while-revalidate to who may store and what happens on expiry](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-2.png)

A few directives round out the picture. `Expires: <HTTP-date>` is the older, absolute-time way to express freshness — `Expires: Wed, 18 Jun 2026 10:00:00 GMT` — and it predates `Cache-Control`. When both are present, `Cache-Control: max-age` *wins*, because relative time is immune to clock skew between the server and the cache (an absolute `Expires` is only as trustworthy as the two clocks agreeing). Emit `Cache-Control` as your primary control and treat `Expires` as a legacy fallback for ancient caches, if you bother with it at all. There is also `age` — a *response* header a shared cache adds to say "this has been sitting in cache for N seconds already," so a downstream cache can compute the *remaining* freshness correctly rather than starting the `max-age` clock over.

`Cache-Control` can also appear on the *request* side, sent by the client to influence caches. `Cache-Control: no-cache` on a request forces every cache to revalidate (this is what a browser's hard-reload does); `Cache-Control: max-age=0` similarly demands fresh-or-revalidated; and `Cache-Control: only-if-cached` asks for a cached response *or* a `504`, never a trip to the origin (useful for offline-first clients). You rarely set these as an API designer, but it's worth knowing that a client can override your freshness suggestion downward — it can always ask for *fresher* than you offered, never staler-than-allowed.

Here is the server side of all this — a small handler that emits a strong `ETag` and `Cache-Control` on a `GET` and short-circuits a conditional request to a `304`. The shape is the same in any framework:

```python
import hashlib
from flask import request, make_response, jsonify

def order_etag(order):
    # A strong validator: hash the canonical representation + a version stamp.
    # Any change to the order changes the bytes, hence the tag.
    raw = f"{order['id']}:{order['version']}:{order['updated_at']}".encode()
    return '"' + hashlib.sha256(raw).hexdigest()[:16] + '"'

@app.get("/orders/<order_id>")
def get_order(order_id):
    order = store.load(order_id)              # cheap key lookup, not the full join
    etag = order_etag(order)

    # Conditional request: if the client's stored tag still matches, send no body.
    inm = request.headers.get("If-None-Match")
    if inm and etag in [t.strip() for t in inm.split(",")]:
        resp = make_response("", 304)         # 304 Not Modified — empty body
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "private, max-age=30"
        resp.headers["Vary"] = "Accept, Authorization"
        return resp                            # we never built the 18 KB body

    body = render_order(order)                # the expensive serialization
    resp = make_response(jsonify(body), 200)
    resp.headers["ETag"] = etag
    resp.headers["Cache-Control"] = "private, max-age=30, stale-while-revalidate=60"
    resp.headers["Vary"] = "Accept, Authorization"
    return resp
```

The crucial efficiency detail is in the comments: the conditional branch computes the `ETag` from a *cheap* version stamp and a key lookup, and when it matches it returns `304` *before* calling `render_order` — so the expensive five-table join and JSON serialization never run. If you compute the ETag by hashing the fully-rendered body, you've already paid the cost you were trying to avoid; derive the tag from a version column or `updated_at` so the short-circuit is genuinely cheap.

For our Payments & Orders API, a sensible default policy looks like this. A settled, finished order — one in a terminal state — can be cached aggressively but should still revalidate because, however unlikely, a refund could change it: `Cache-Control: private, max-age=30, stale-while-revalidate=60`. A list of a user's orders is personalized, so it must be `private`. A payment confirmation that includes sensitive details is `Cache-Control: no-store`. A public product catalog entry that everyone sees identically is `Cache-Control: public, s-maxage=300, stale-while-revalidate=60` so the CDN absorbs the read traffic.

## 3. Freshness vs validation: served-without-asking vs revalidated

The single most important mental model in HTTP caching is the split between freshness and validation, so let us make it crisp. Every cached response is in one of two states. While it is *fresh* — within its `max-age`/`s-maxage` window — a cache serves it directly, with zero contact with the origin. The origin doesn't even know the request happened. That is the freshness win, and it is total: no round-trip, no body, no compute. Once the response goes *stale* — past its freshness lifetime — the cache may not serve it blindly. It must *validate*: send a conditional request to the origin asking "is my copy still good?" and either get back `304 Not Modified` (yes, reuse what you have) or `200 OK` with a fresh body (no, here's the new version).

| Property | Freshness | Validation |
| --- | --- | --- |
| Controlled by | `max-age`, `s-maxage`, `Expires` | `ETag` / `Last-Modified` validators |
| Round-trip to origin? | None — served locally | Yes — a conditional request |
| Origin work on a hit | Zero | Compare validators; build no body if `304` |
| Bytes transferred on a hit | Zero | A few hundred (headers only) |
| Latency on a hit | Near zero | One round-trip, small payload |
| Risk | Serving slightly-stale data | None — always confirms currency |
| Best when | Data tolerates a known staleness window | Data must be current but rarely changes |

The two compose. A well-designed response has *both*: a `max-age` so that frequent reads in a short window are served instantly from cache, *and* an `ETag` so that once the freshness window lapses, the recheck is a cheap `304` instead of a full re-download. You tune `max-age` to "how stale can this safely be?" and you add a validator so that the inevitable revalidation is nearly free. Setting one without the other leaves value on the table: a `max-age` with no validator means every revalidation is a full `200`; a validator with `max-age=0` (or `no-cache`) means you always pay one round-trip but always with a cheap body.

![A branching acyclic graph showing a GET request checked for freshness, served directly when fresh, and sent through a conditional request that returns either a bodyless 304 or a full 200 when stale](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-3.png)

The figure traces it: a `GET` arrives, the cache checks freshness, a fresh entry is served outright (zero origin bytes), and a stale entry triggers a conditional `GET` whose `If-None-Match` either confirms the copy (`304`, no body) or replaces it (`200`, new body). Notice this is acyclic and it branches at "fresh vs stale" and again at "unchanged vs changed" — that branching *is* the decision logic of every cache on earth.

## 4. Validators and conditional requests: ETag, If-None-Match, and the 304

A *validator* is a small token the server attaches to a response that uniquely identifies that exact version of the resource. A *conditional request* is a later request that carries the validator and says "only give me a body if the validator no longer matches." The two together are the validation half of caching.

The strong validator is the **`ETag`** — short for "entity tag." It is an opaque string the server computes for a representation; the client treats it as a black box and never tries to parse it. It might be a hash of the body, a version number, a database row version, or a timestamp encoded somehow — the client neither knows nor cares. What matters is the contract: *if two responses for the same URL have the same `ETag`, they are byte-for-byte identical; if the resource changes, the `ETag` changes.*

Here is the first half of the exchange — the server hands out an `ETag` on a normal `GET`:

```http
GET /orders/ord_7Hk2 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Accept: application/json

HTTP/1.1 200 OK
Content-Type: application/json
ETag: "a1b2c3"
Cache-Control: private, max-age=30
Vary: Accept, Authorization
Content-Length: 18044

{
  "id": "ord_7Hk2",
  "status": "shipped",
  "currency": "USD",
  "total": "149.90",
  "items": [
    { "sku": "WIDGET-1", "qty": 2, "unit_price": "49.95" },
    { "sku": "GADGET-9", "qty": 1, "unit_price": "50.00" }
  ],
  "updated_at": "2026-06-18T09:14:02Z"
}
```

The client stores the body *and* the `ETag` value `"a1b2c3"`. Thirty seconds later — past `max-age` — it wants the order again. Instead of a plain `GET`, it sends a *conditional* `GET` with `If-None-Match`, replaying the stored ETag:

```http
GET /orders/ord_7Hk2 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Accept: application/json
If-None-Match: "a1b2c3"

HTTP/1.1 304 Not Modified
ETag: "a1b2c3"
Cache-Control: private, max-age=30
Vary: Accept, Authorization
```

That `304 Not Modified` is the prize. It has **no body** — `Content-Length` is effectively zero, there is no JSON at all. The server computed the current `ETag`, saw it still equals `"a1b2c3"`, and answered "what you have is current; reuse it." The client serves its stored eighteen-kilobyte body and resets its freshness clock (the `304` carries a fresh `Cache-Control`, so the resource is fresh for another thirty seconds). We just confirmed currency and refreshed freshness while transferring a few hundred bytes of headers instead of eighteen thousand bytes of body.

`If-None-Match` reads literally: "give me the resource *if* its entity tag does *none* of my listed tags match" — i.e., send a body only if the resource has changed. When it *has* changed, the server ignores the precondition and returns a normal `200 OK` with the new body and a new `ETag`, and the client replaces what it stored.

![A left-to-right timeline of an ETag conditional request showing the first 200 with an ETag, the client storing it, a later If-None-Match request, the server comparing, and a bodyless 304 the client reuses](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-4.png)

#### Worked example: an ETag revalidation that saves the body

Concretely, walk the merchant dashboard from the intro through one full day with and without validation. The dashboard polls one finished order every thirty seconds — call it $2{,}880$ polls per day. The order is unchanged on every single poll.

**Without a validator.** Each poll is a plain `GET` answered with a full `200 OK`. The body is 18 KB (call it 18,000 bytes, ignoring gzip for round numbers). Daily bytes for this one dashboard on this one order:

$$2{,}880 \text{ polls} \times 18{,}000 \text{ bytes} \approx 51.8 \text{ MB/day}$$

and the origin builds the full document — runs the join, serializes — $2{,}880$ times.

**With an `ETag`.** The first poll of the window is a full `200` (18 KB); every subsequent poll within the freshness window is served from cache (0 bytes to origin), and each poll *after* freshness lapses is a conditional `GET` answered by a ~300-byte `304`. With `max-age=30` and a 30-second poll interval, essentially every poll lapses freshness and becomes a conditional request, so we get roughly one full `200` and $2{,}879$ conditional exchanges:

$$18{,}000 + 2{,}879 \times 300 \approx 0.88 \text{ MB/day}$$

That is about a **98% reduction** in bytes transferred for this workload — $51.8$ MB down to under $0.9$ MB — and the origin, while still consulted on each poll, short-circuits after the ETag comparison and never builds the 18 KB body or runs the serialization. Now multiply by ten thousand dashboards and the saving is the difference between a comfortable backend and a fire. *This is the entire reason validators exist.*

![A before-and-after comparison contrasting a client that refetches a full 200 response on every poll against one that uses an ETag and gets mostly bodyless 304 responses](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-5.png)

It is worth dwelling on *what kind* of saving this is, because it is easy to undersell. The freshness win (serving from cache within `max-age`) is the obvious one and it gets all the attention, but it has a ceiling: you can only be as stale as your `max-age`, and for a dashboard showing a *live* order you may not want to be stale at all. The validation win is what makes "always current, almost free" possible. Every single poll in the example *did* reach the origin and *did* confirm currency — the dashboard was never showing stale data — and yet the bytes-on-the-wire fell by 98% and the origin's per-poll work fell to an ETag comparison. You bought freshness *and* efficiency, which is normally a trade-off, by spending one small round-trip. When someone says "we can't cache this, it has to be live," the honest answer is usually "you can't use *freshness*, but you can absolutely use *validation*" — give it `Cache-Control: no-cache` (store-and-always-revalidate) plus an `ETag`, and every recheck is a cheap `304` while the data stays exactly current.

The shape of the saving also explains *which* workloads benefit most. Validation pays off in direct proportion to two things: how often the resource is re-read while unchanged, and how large the body is relative to the validation exchange. A small body that changes on every read gets nothing from an ETag (every request is a full `200` anyway, plus the overhead of computing the tag). A large body that is re-read constantly and changes rarely — exactly our order document — is the jackpot. Before you add validators everywhere, ask "is this re-read while unchanged, and is the body big?" If yes to both, an `ETag` is close to pure profit. If no to either, it's harmless but not transformative.

### Strong vs weak ETags

An `ETag` is *strong* by default and *weak* if prefixed with `W/`: `ETag: W/"a1b2c3"`. The distinction is about what "the same" means.

A **strong** `ETag` promises *byte-for-byte* identity. If two responses share a strong ETag, they are octet-for-octet the same. Strong validators are required for any use that depends on exact bytes — most importantly *range requests* (resuming a download from byte 5000 only makes sense if the bytes haven't shifted) and the `If-Match` optimistic-concurrency case below.

A **weak** `ETag` (`W/"..."`) promises only *semantic* equivalence: the responses mean the same thing, even if they differ in some trivial, non-meaningful way. The classic case is a response whose body is regenerated with a new timestamp comment or re-serialized with keys in a different order, or compressed differently — semantically identical, byte-different. A weak validator says "for caching purposes these are the same, so a `304` is correct," but you cannot safely do byte-range resumption against it. For most API caching, a weak ETag is *fine and often more honest*, because you usually care that the order is semantically unchanged, not that the JSON whitespace is identical. Reach for a strong ETag specifically when you need range requests or `If-Match` concurrency control.

| | Strong ETag `"a1b2c3"` | Weak ETag `W/"a1b2c3"` |
| --- | --- | --- |
| Promise | Byte-for-byte identical | Semantically equivalent |
| Range requests (resume) | Allowed | Not allowed |
| `If-Match` concurrency control | Allowed | Not allowed |
| `If-None-Match` caching | Allowed | Allowed |
| Cheap to compute? | Maybe (hash the body) | Often cheaper (a version/row stamp) |
| Best when | Exact bytes matter | Semantic identity is enough |

### Last-Modified and If-Modified-Since: the time-based validator

Before `ETag` there was time. The server can stamp a response with `Last-Modified: Wed, 18 Jun 2026 09:14:02 GMT`, and the client revalidates with `If-Modified-Since: <that date>`. The server compares the resource's modification time to the client's date and answers `304` if it hasn't been modified since, or a full `200` if it has.

```http
GET /orders/ord_7Hk2 HTTP/1.1
Host: api.example.com
If-Modified-Since: Wed, 18 Jun 2026 09:14:02 GMT

HTTP/1.1 304 Not Modified
Last-Modified: Wed, 18 Jun 2026 09:14:02 GMT
```

`Last-Modified` is simpler and cheaper when you already track a modification timestamp, but it has real limitations: it has only one-second resolution, so two changes within the same second are indistinguishable; it can't represent "changed back to a previous value" (the time always moves forward even when content reverts); and "modified" by wall-clock time isn't always the same as "the representation changed" (regenerating an unchanged document can bump its mtime). `ETag` has none of these problems because it identifies the *content*, not the *time*. When both are present, `ETag`/`If-None-Match` takes precedence over `Last-Modified`/`If-Modified-Since`. Use `Last-Modified` as a cheap fallback or alongside an ETag; use `ETag` as the primary validator.

| | `ETag` + `If-None-Match` | `Last-Modified` + `If-Modified-Since` |
| --- | --- | --- |
| Identifies | The exact content | The modification time |
| Resolution | Exact (any change → new tag) | One second |
| Detects revert to old value | Yes | No (time only moves forward) |
| Cost to produce | A hash or version stamp | A timestamp you likely have |
| Precedence when both sent | Wins | Used only if no ETag |
| Enables `If-Match` concurrency | Yes (strong) | No |

## 5. ETag + If-Match: optimistic concurrency control

The same validator that powers caching also solves a completely different problem: the *lost update*. Suppose two customer-service reps both open order `ord_7Hk2`, both decide to change the shipping address, and both save. Without protection, whoever saves last silently overwrites the other — the first rep's change vanishes with no error, no trace, no warning. This is the lost-update problem, and it is one of the most insidious bugs in any concurrent system because nothing ever looks broken; data just quietly disappears.

`If-Match` fixes it with *optimistic concurrency control* — "optimistic" because it assumes conflicts are rare and only checks for them at write time, rather than locking the row up front. The rule: a client that wants to update a resource must send the `ETag` it last read, in an `If-Match` header. The server applies the write *only if* the resource's current `ETag` still matches — meaning nobody else changed it in between. If it doesn't match, the resource has moved on, the client is working from a stale copy, and the server rejects the write with **`412 Precondition Failed`**. (`If-Match` requires a *strong* ETag, since it depends on exact identity.)

```http
PUT /orders/ord_7Hk2 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
If-Match: "a1b2c3"

{ "shipping_address": { "line1": "742 Evergreen Terrace", "city": "Springfield" } }

HTTP/1.1 200 OK
ETag: "d4e5f6"
Content-Type: application/json

{ "id": "ord_7Hk2", "status": "shipped", "shipping_address": { "line1": "742 Evergreen Terrace", "city": "Springfield" }, "updated_at": "2026-06-18T11:02:40Z" }
```

The write succeeded because the order's ETag was still `"a1b2c3"` when the server checked. Note the response carries a *new* `ETag`, `"d4e5f6"` — the resource changed, so its validator changed, and any cached copy with the old tag is now correctly considered stale.

#### Worked example: an If-Match update that returns 412 on a stale write

Now run the race. Rep A and Rep B both `GET /orders/ord_7Hk2` and both receive `ETag: "a1b2c3"`. Rep A submits a `PUT` with `If-Match: "a1b2c3"`; it succeeds and the order's ETag becomes `"d4e5f6"`. Seconds later Rep B — still holding the now-stale `"a1b2c3"` — submits *their* change:

```http
PUT /orders/ord_7Hk2 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
If-Match: "a1b2c3"

{ "shipping_address": { "line1": "1600 Pennsylvania Ave", "city": "Washington" } }

HTTP/1.1 412 Precondition Failed
Content-Type: application/problem+json

{
  "type": "https://api.example.com/problems/stale-write",
  "title": "Precondition Failed",
  "status": 412,
  "detail": "The order was modified by another request. Re-fetch and re-apply your change.",
  "current_etag": "d4e5f6"
}
```

Instead of silently clobbering Rep A's address change, the server returns `412 Precondition Failed` with a `problem+json` body (RFC 9457 — the machine-readable error format covered in [error design: a machine-readable, human-friendly contract](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract)). Rep B's client now knows the order moved underneath them. The correct recovery is to re-`GET` the order (picking up Rep A's change and the new ETag `"d4e5f6"`), show Rep B the current state, let them re-apply their edit on top, and `PUT` again with `If-Match: "d4e5f6"`. The lost update became a *visible, recoverable conflict* — exactly the behavior you want.

![A timeline of two writers racing on one order where the first PUT with If-Match succeeds and bumps the ETag, and the second PUT with the stale ETag is rejected with 412 before the loser refetches and retries](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-6.png)

There is a related precondition worth knowing: `If-Match: *` means "only if the resource exists at all" (any current representation), and `If-None-Match: *` means "only if it does *not* exist" — the latter is the canonical way to do a safe *create* that fails with `412` if someone else already created the resource at that URL, turning a `PUT` into a create-if-absent. And when a write *requires* a precondition but the client didn't send one, the right answer is `428 Precondition Required` — "you must use `If-Match` here" — which lets you enforce optimistic concurrency for everyone. The status-code semantics for `412` and `428` are covered in [status codes that tell the truth: 2xx, 3xx, 4xx, 5xx](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).

It is worth contrasting this with the alternatives, because optimistic concurrency is one of three ways to handle the lost-update problem and the trade-offs are real. The first alternative is *last-write-wins* — do nothing, let the second writer silently clobber the first. It's the default if you write no guard at all, and it's almost always wrong for anything a human cares about, precisely because it's silent: the data loss leaves no trace. The second is *pessimistic locking* — the client takes an explicit lock on the resource before editing, so no one else can change it until the lock is released. This prevents conflicts outright but is heavy: locks must be acquired, held, released, and (the hard part) *timed out* when a client takes the lock and then walks away from their desk, or you've wedged the resource indefinitely. Optimistic concurrency — the `If-Match` approach — is the middle path: no lock, no coordination, just a version check at write time that turns a conflict into a clean `412` the loser retries. It's optimal exactly when conflicts are *rare* (two reps editing the same order in the same ten seconds is unusual), which is the common case, and it degrades gracefully when they're not (the loser simply refetches and retries rather than blocking).

| Strategy | Conflict outcome | Coordination cost | Best when |
| --- | --- | --- | --- |
| Last-write-wins | Silent data loss | None | Conflicts truly don't matter (counters, telemetry) |
| Pessimistic lock | Blocked until released | High — acquire, hold, time out | Conflicts frequent and writes long |
| Optimistic (`If-Match`) | Visible `412`, retry | None until write time | Conflicts rare — the common case |

A practical note on *what* you put in the `ETag` for concurrency. You don't have to hash the body; a monotonic version number or a row-version column (`xmin` in Postgres, a `rowversion` in SQL Server, an integer `version` you bump on every write) is a perfect strong validator and is far cheaper to compute. The contract a client depends on — "the tag changes iff the resource changes" — is satisfied by any value with that property, and a bumped integer satisfies it trivially while a body hash also has to be *computed* on every read. The one rule: it must be a *strong* validator for `If-Match` to be legal, so don't use a weak `W/"..."` tag for concurrency control.

This is a beautiful piece of HTTP design: *the same `ETag` you emit for caching is the version token you check for concurrency.* One small response header does double duty — it makes reads cheap and it makes writes safe. And it composes with idempotency: an `If-Match` precondition makes a *write* safe against concurrent writers, while an `Idempotency-Key` makes the *same* write safe against being accidentally sent twice; a careful payments API uses both, because they defend against different failures.

## 6. The Vary header: caching per representation

Here is a subtle, dangerous bug. Our orders API supports content negotiation — it can return JSON or, for a legacy partner, XML, chosen by the `Accept` request header. A shared cache stores responses keyed by URL. If a JSON client requests `GET /orders/ord_7Hk2` and the cache stores the JSON response under the key `/orders/ord_7Hk2`, then an XML client requests the *same URL*, the cache will happily serve them the stored *JSON* — even though they asked for XML. The cache has no idea the response depends on a request header it never looked at.

The **`Vary`** header is the fix. It tells caches: "the correct response for this URL *varies* by these request headers, so include them in the cache key." `Vary: Accept` means "store and match a separate cached entry per distinct `Accept` value." Now the JSON entry and the XML entry are cached separately, and each client gets the representation they asked for.

```http
GET /orders/ord_7Hk2 HTTP/1.1
Accept: application/xml

HTTP/1.1 200 OK
Content-Type: application/xml
Vary: Accept, Accept-Encoding
ETag: "x9y8z7"
Cache-Control: public, max-age=60
```

The three you almost always need to consider:

- **`Vary: Accept`** — when the response format depends on content negotiation (JSON vs XML vs a vendor media type). Without it, a cache leaks the wrong representation.
- **`Vary: Accept-Encoding`** — when you serve gzip/brotli-compressed responses. A cache must not serve a brotli-compressed body to a client that only declared support for gzip. Almost every API that compresses needs this, and most CDNs handle it automatically — but you should still emit it.
- **`Vary: Authorization`** — the big one for authenticated APIs. If the response depends on *who* is asking (and a personalized response always does), the cache key must include the credential, or one user's data leaks to another. In practice, `Vary: Authorization` plus `Cache-Control: private` is the safe pairing for per-user responses: `private` keeps shared caches out entirely, and `Vary: Authorization` makes any cache that *does* store it key on the credential.

The failure mode of a *missing* `Vary` is exactly the kind of bug that ships to production and then leaks data: it works perfectly in testing (where every request looks the same), and breaks only when two clients with different `Accept` headers — or worse, different `Authorization` headers — hit the same cache. A missing `Vary: Authorization` on a `public`, cacheable per-user endpoint is a serious security incident waiting to happen: the CDN caches Alice's order list and serves it to Bob.

There is a cost to `Vary`, and it is worth understanding: every distinct value of a varied header multiplies the number of cache entries. `Vary: Accept` is usually fine because there are only a handful of `Accept` values in practice. But `Vary: User-Agent` is almost always a mistake — there are millions of distinct user-agent strings, so you'd shatter the cache into millions of near-duplicate entries and your hit ratio would collapse to near zero. Vary only on headers that genuinely change the response *and* have a small set of values. For per-user content, prefer `Cache-Control: private` (cache in the browser, where the cache key is naturally per-user) over a shared cache with `Vary: Authorization`.

## 7. What's cacheable: safe methods, auth, and the dangerous cases

Not everything should be cached, and the rules are mostly about HTTP method semantics. A *safe* method is one that doesn't change server state — `GET`, `HEAD`, `OPTIONS`. A method is *cacheable* if a response to it may be stored and reused. By default `GET` and `HEAD` responses are cacheable; `POST`, `PUT`, `PATCH`, and `DELETE` responses are not (a `POST` response *can* be cached if explicitly marked, but this is rare and subtle, so treat write-method responses as uncacheable in practice).

This is the deep reason REST gets caching for free and RPC/GraphQL don't, which we'll return to in the last section: REST puts reads on `GET` with the resource identity in the URL, so the whole HTTP caching machinery — keyed by method and URL — just works. The moment you tunnel a read through `POST` (as JSON-RPC and GraphQL typically do), you've told every cache "this might change state, don't touch it," and you've thrown away the URL as a cache key.

The dangerous cases are around authentication and personalization:

- **Never cache authenticated responses in a shared cache without care.** A response that depends on the caller's identity must be `private` (browser-only) or, if it must transit a shared cache, must `Vary: Authorization` *and* you must trust that cache completely. The default safe stance: authenticated, personalized responses are `Cache-Control: private, no-cache` (store in the browser, always revalidate) or `private, max-age=<small>` if a brief staleness is acceptable.
- **Never cache responses to write methods.** A cached `POST /payments` response served on a retry would be a disaster — though note this is exactly the problem [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) solve at the application layer, which is a *different* mechanism from HTTP caching: idempotency keys make a *retry* safe; caching makes a *re-read* cheap. Don't conflate them.
- **Be careful with responses that contain secrets.** Anything with a token, a full card number, or one-time credentials should be `no-store`. This is the one case where the heavy hammer is correct.
- **Public, non-personalized reads are the sweet spot.** A product catalog, a list of supported currencies, public reference data — these are identical for everyone, change rarely, and should be `public` with a healthy `s-maxage` so the CDN does almost all the work.

A useful default policy, written as a small decision: *if the response is the same for everyone and changes rarely, make it `public` with a CDN-friendly `s-maxage`; if it's per-user, make it `private`; if it's secret, make it `no-store`; and attach an `ETag` to everything in the first two categories so revalidation is cheap.*

## 8. Invalidation: the genuinely hard problem

Phil Karlton's line — "there are only two hard things in computer science: cache invalidation and naming things" — is funny right up until a customer refunds an order, refreshes the page, and still sees it marked "paid." That is cache invalidation failing, and it is hard for a fundamental reason: a cache is a *copy* of the truth, and the moment the truth changes, every copy everywhere is potentially wrong, and you may not know where all the copies are. You have a browser cache you don't control, CDN nodes in dozens of cities, a reverse proxy, an application cache — and the order just changed in the database. Which copies are now lies, and how do you stop serving them?

There are three fundamentally different strategies, and mature systems use all three for different things.

**TTL expiry (time-based).** You set a `max-age`/`s-maxage` and simply let entries go stale and revalidate (or expire) on their own. This is the cheap, set-and-forget approach: you never explicitly tell a cache anything changed; you just accept that a change may take up to the TTL to propagate. The trade-off is *staleness lag* — pick a 60-second TTL and a change can be invisible for up to 60 seconds. For most read-mostly data this is completely fine, and it is by far the most operationally robust strategy because there is nothing to coordinate. The whole game is choosing a TTL short enough that the staleness is acceptable and long enough that the hit ratio is high.

**Explicit purge (event-based).** When the truth changes, you actively tell the caches to drop the affected entries. On a refund, your application calls the CDN's purge API for `/orders/ord_7Hk2` (and any related URLs), and the edge nodes evict those entries so the next request goes to the origin and refetches. This gives near-instant freshness — the change is visible immediately — at the cost of *reach* and *coordination*: you have to know exactly which URLs are affected, you have to call every cache layer's purge mechanism, and a purge that misses a layer leaves a stale copy behind. Purge is powerful and necessary for "must be fresh now" data, but it is operationally heavier and a frequent source of "we purged the CDN but forgot the reverse proxy" bugs.

**Versioned URLs / cache-busting.** The cleverest strategy *avoids invalidation entirely* by putting a version in the URL. If the resource lives at `/static/app.a1b2c3.js` and you change it, you don't invalidate anything — you publish at a *new* URL, `/static/app.d4e5f6.js`, and update the references that point to it. The old URL still serves the old content (correctly — that *is* the old version), and nothing was ever stale because the URL and the content are bound together. This is why static-asset pipelines fingerprint filenames and serve them `immutable`: invalidation becomes a non-problem. The trade-off is that it only works where you control the references and the URL can change — perfect for assets and content-addressed data, not directly applicable to a stable API resource like `/orders/ord_7Hk2` whose URL must stay constant. (You *can* borrow the idea via an `ETag`, which is essentially a content version that lets validation, rather than URL rewriting, do the same job.)

![A matrix comparing TTL expiry, explicit purge, and versioned URLs across their freshness lag and their cost](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-7.png)

To make this concrete, walk a refund through a layered design. A customer refunds `ord_7Hk2`. The write lands in the database and bumps the order's version (so its `ETag` changes). Now the staleness question fires at every layer. The browser cache holds a `private, max-age=30` copy — it will revalidate within 30 seconds and, because the `ETag` changed, get a fresh `200`; worst-case staleness there is 30 seconds, which for a personalized view we've decided is acceptable. The CDN holds a `public` copy of any shared representation tagged `order-ord_7Hk2`; the refund handler issues a *soft purge* by that surrogate key, so the CDN marks its copy stale and refreshes once in the background — near-instant, no stampede. The reverse proxy, if it cached anything, is covered by the same purge call *if* we remembered to wire it up; if we didn't, its TTL is its safety net. The key design property is that *every* copy has a bounded staleness even when a purge misses a layer — because the TTL is always there underneath as the floor. That is the whole argument for layering: purge gives you speed where you wire it, TTL gives you a guarantee everywhere.

The honest engineering answer is to *layer* them: TTL as the safe default for everything (so a missed purge degrades to "stale for at most the TTL" rather than "stale forever"), explicit purge on top for data that must be fresh immediately, and versioned URLs for anything you control the address of. The most dangerous design is a long TTL with *no* purge and *no* validator on data that genuinely changes — that is the configuration where a refund stays invisible for an hour. If you take one rule from this section: never set a TTL longer than you are willing to serve stale data, because the TTL *is* your worst-case staleness when purge fails.

There is one more subtlety worth naming, because it bites teams who think purge is a silver bullet: invalidation is only as consistent as it is *complete and ordered*. If the refund write commits, the purge fires, and then a *late* read that started before the commit lands in the cache *after* the purge, you've re-cached the stale value and the purge accomplished nothing — a classic race. Defenses include purging *after* the commit is durable (not before), using the version-stamped `ETag` so even a re-cached entry revalidates correctly on the next read, and keeping TTLs short enough that the window of wrongness is bounded. This is why a validator under everything is so valuable: even when invalidation races lose, the `ETag` means the *next* conditional request still detects the change and corrects course. Purge optimizes the common case; the validator is the backstop that keeps a lost race from becoming permanent staleness.

## 9. CDN caching of API responses: surrogate keys and soft purge

CDNs were built for static files, but they are increasingly used to cache *API* responses at the edge — and the tooling for doing it well is worth knowing, because it turns the invalidation problem from "purge individual URLs" into something far more manageable.

**Surrogate keys (cache tags).** The problem with URL-based purge is that one logical change can affect many URLs. A refund on `ord_7Hk2` might need to invalidate `GET /orders/ord_7Hk2`, `GET /orders/ord_7Hk2/payments`, the customer's `GET /orders` list, and a `GET /customers/cus_99/summary`. Purging four URLs by hand is fragile. Surrogate keys solve this: when the origin responds, it tags the response with one or more keys via a header — Fastly calls it `Surrogate-Key`, Cloudflare calls them `Cache-Tag` (an Enterprise feature) — like `Surrogate-Key: order-ord_7Hk2 customer-cus_99`. The CDN remembers which cached entries carry which keys. Later, to invalidate *everything* touching that order, you issue a single purge by key — "purge `order-ord_7Hk2`" — and the CDN evicts every entry tagged with it, across all the URLs, in one call. This is the right abstraction for API caching: tag by the *domain entity*, purge by the entity, and you never have to enumerate URLs again.

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: public, s-maxage=300, stale-while-revalidate=60
Surrogate-Key: order-ord_7Hk2 customer-cus_99
ETag: "a1b2c3"
```

```bash
  # On a refund, the origin issues one purge-by-key to the CDN:
  curl -X POST "https://api.fastly.com/service/$SERVICE_ID/purge/order-ord_7Hk2" \
    -H "Fastly-Key: $FASTLY_TOKEN"
```

Note `Surrogate-Key` (and its companion `Surrogate-Control`, which sets CDN-only caching rules the CDN strips before forwarding to the browser) are *between origin and CDN* — they don't reach the end client. That separation is exactly what you want: the CDN can cache aggressively with `Surrogate-Control: max-age=300` while telling the browser something more conservative via the regular `Cache-Control`.

**Soft purge.** A normal ("hard") purge evicts an entry immediately, so the very next request is a guaranteed cache miss that goes to the origin — and if a thousand clients are watching that key, you've just created a stampede (next section). A *soft* purge instead marks the entry *stale* rather than deleting it. Combined with `stale-while-revalidate`, the CDN keeps serving the now-stale-but-still-present copy instantly while it refreshes once from the origin in the background. You get near-instant invalidation *without* the miss spike. Fastly's soft purge is the canonical example; the pattern is "mark stale, serve stale, revalidate once." For any high-traffic key, prefer soft purge over hard purge.

## 10. Cache stampede: the thundering herd at the origin

Here is a failure that catches teams off guard precisely because caching was working. A single very popular key — the catalog entry for a product that's trending, say — is cached at the edge with `max-age=60`. It's serving a thousand requests per second entirely from cache; the origin is idle and happy. Then the entry expires. In that instant, *all* the in-flight requests miss simultaneously, and a thousand requests per second stampede the origin at once, all trying to recompute the same expensive response. This is the **cache stampede** (or thundering herd, or dog-piling), and it can take down an origin that was comfortably handling the *cached* load, because it's suddenly hit with the full *uncached* load — and worse, with a thousand identical concurrent misses all doing the same work.

![A branching acyclic graph of a cache stampede where a hot key expires, a thousand concurrent misses either hammer the origin without coalescing or are coalesced into a single in-flight fetch while stale-while-revalidate serves the old copy](/imgs/blogs/caching-etags-cache-control-conditional-requests-invalidation-8.png)

There are two complementary defenses.

**Request coalescing (also called request collapsing or single-flight).** When many concurrent requests miss the same key, the cache lets *one* of them go to the origin and makes all the others *wait* for that single in-flight fetch, then serves them all from its result. A thousand simultaneous misses become *one* origin request. Most CDNs and reverse proxies do this for you — Varnish calls it "request coalescing," nginx has `proxy_cache_lock`, Fastly does it by default — and in your own application cache you implement it with a single-flight primitive (Go's `singleflight`, a per-key lock/promise) so concurrent callers for the same key share one computation. Coalescing is the structural fix: it makes the origin load on a popular-key miss independent of how many clients are watching.

**`stale-while-revalidate`.** The directive from §2 is also a stampede defense, and a graceful one. With `Cache-Control: max-age=60, stale-while-revalidate=30`, when the entry goes stale the cache doesn't make anyone wait for a miss at all — it serves the stale copy *instantly* to everyone and refreshes from the origin once in the background. The origin sees exactly one refresh request, not a herd, and no client experiences a latency spike. Combined with soft purge, this is the gold-standard pattern for a high-traffic key: clients always get an instant answer, and the origin only ever sees a trickle of single background refreshes.

A third, lighter technique is *jittered TTLs*: instead of setting `max-age=60` on every entry (so a batch populated together all expire together), add a small random spread — `60 ± 5` seconds — so expirations are smeared across time rather than synchronized into one cliff. It doesn't help a single hot key, but it prevents whole *populations* of keys from expiring in lockstep.

The principle here is worth stating plainly: *a cache miss on a hot key is a load amplifier.* Without coalescing, the origin load at the moment of expiry is proportional to the *request rate*, not to one request. Coalescing and `stale-while-revalidate` both break that amplification — coalescing by collapsing concurrent misses into one fetch, `stale-while-revalidate` by serving stale during the refresh so there's no synchronized miss at all.

## 11. The hard part for RPC and GraphQL: no cacheable URL

We have been quietly relying on something the whole time: REST's reads are `GET`s with the resource identity in the URL. That is precisely what makes HTTP caching *automatic* for REST — every cache in the world already keys on method and URL, so a well-behaved `GET /orders/ord_7Hk2` is cacheable at every layer with nothing but the right headers. This is one of REST's quietly enormous advantages, and it is the reason this entire post is mostly about *configuring* caching rather than *building* it.

RPC and GraphQL forfeit most of this, and it's worth being honest about why.

**RPC over `POST`.** JSON-RPC and most RPC styles send every call — reads included — as a `POST` to a single endpoint like `/rpc`, with the method name and arguments in the body. To an HTTP cache, that is an opaque, non-cacheable, possibly-state-changing write to one URL. The URL no longer identifies the resource, so it can't be a cache key, and the method (`POST`) says "don't cache." RPC reads get *zero* free HTTP caching. You either cache inside the application (your own keyed store, which you must invalidate yourself) or you don't cache reads at all. gRPC is in the same boat — it runs over HTTP/2 `POST` and gets no HTTP-layer response caching, so caching is an application concern.

**GraphQL.** GraphQL has the same root problem and then some. A GraphQL query is almost always a `POST` to a single `/graphql` endpoint with the query in the body, so again: one URL, `POST` method, no HTTP caching. Worse, two clients asking for *overlapping but different* field selections produce *different* responses from the *same* URL+method, so even if you forced it onto `GET`, the cache key would be wrong unless it incorporated the entire query string. GraphQL's answers to this are real but they are *reinventions* of what REST got for free:

- **Persisted queries.** The client registers a query once and thereafter sends only a short hash of it (Apollo's "Automatic Persisted Queries" do this). Because the hash is small and stable, you *can* now send the request as a `GET /graphql?extensions=...&variables=...` with the query identified by hash — which restores a cacheable URL, so CDNs can cache GraphQL *reads* after all. This is the standard way to make GraphQL cacheable at the edge, and it exists precisely because GraphQL threw away the free URL-based caching that REST has.
- **Client-side normalized caching.** Apollo Client and Relay maintain a normalized object cache keyed by entity ID on the client, so repeated fields are reused without refetching. This is genuinely good DX, but it is a *client* cache — it does nothing for shared CDN/proxy caching and nothing for the origin load from *new* clients.

The honest summary: REST trades some uniformity for getting the entire HTTP caching ecosystem for free; GraphQL trades that free caching for query flexibility and has to rebuild caching with persisted queries and client-side stores. Neither is wrong — it is the "choose the paradigm by force, not fashion" trade-off from [choosing a paradigm: REST vs gRPC vs GraphQL by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force). If cheap, ubiquitous, layered caching of reads is a top-three requirement, REST's URL-as-cache-key is a feature you give up reluctantly. The deeper details of how GraphQL resolves and where it pays its costs are in [GraphQL: the query language, schema, and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap).

## 12. Putting it together: a caching policy for Payments & Orders

Let us close the running example by writing the actual policy, endpoint by endpoint, so you can see the reasoning compose.

`GET /orders/{id}` (a single order, per-user, changes occasionally): `Cache-Control: private, max-age=30, stale-while-revalidate=60` plus a strong `ETag` and `Vary: Accept, Authorization`. Browser-only (it's personalized), fresh for 30s so a busy dashboard mostly reads from local cache, an ETag so the inevitable revalidation is a cheap `304`, and `stale-while-revalidate` so a refresh never blocks the UI. The same `ETag` guards `PUT`/`PATCH` with `If-Match`.

`GET /orders` (the user's order list, per-user): `Cache-Control: private, no-cache` plus an `ETag`. Lists change more often and a stale list is more confusing than a stale single order, so we revalidate every time — but the `ETag` keeps each revalidation a `304` when nothing changed.

`POST /payments`, `POST /refunds` (writes): `Cache-Control: no-store`. Never cache a write. Retry-safety is handled separately by an `Idempotency-Key`, not by caching.

`GET /payment-methods/{id}` containing sensitive data: `Cache-Control: no-store`. The heavy hammer, correctly, because the body has secrets.

`GET /catalog/products/{id}` (public, identical for everyone, read-heavy): `Cache-Control: public, s-maxage=300, stale-while-revalidate=60`, a `Surrogate-Key: product-{id}`, and `Vary: Accept-Encoding`. The CDN absorbs essentially all the traffic, a refund or price change triggers a soft purge by `product-{id}`, and the origin sees a trickle of background refreshes. This is the endpoint where caching earns its keep.

#### Worked example: a 304 and a 412 in one client session

Trace a single CS rep's session against `ord_7Hk2` to see both halves of the ETag story fire. They open the order: `GET` returns `200`, body, `ETag: "a1b2c3"`. The dashboard polls 30 seconds later with `If-None-Match: "a1b2c3"` and gets `304 Not Modified` — no body, a few hundred bytes, the order is confirmed unchanged. The rep edits the shipping address and `PUT`s with `If-Match: "a1b2c3"` — but in the meantime a refund landed, bumping the order to `ETag: "d4e5f6"`, so the write comes back `412 Precondition Failed`. The client refetches (`GET` → `200`, new body, `ETag: "d4e5f6"`), shows the rep the refund and asks them to re-confirm the address edit, and re-`PUT`s with `If-Match: "d4e5f6"`, which now succeeds with a fresh `ETag: "g7h8i9"`. One session, one `304` (cheap read), one `412` (safe write), zero lost updates, and a fraction of the bytes a naive client would have moved. *That is the whole post in one trace.*

## Case studies: caching and conditional requests in the wild

**GitHub's REST API: conditional requests and the rate-limit interplay.** GitHub's REST API attaches an `ETag` to most `GET` responses and actively encourages clients to send `If-None-Match`. The clever part is the interaction with rate limiting: GitHub's documentation states that a conditional request returning `304 Not Modified` *does not count against your primary rate limit*. So a well-behaved client that caches and revalidates with ETags can poll far more often than a naive client that refetches full bodies, because its `304`s are effectively free against the quota. This is a real, accurate example of caching being designed *into* the rate-limit contract — it rewards clients for being cache-friendly, which aligns the API consumer's interest (cheap polling) with the provider's (less load). GitHub also supports `Last-Modified`/`If-Modified-Since` as a fallback. The lesson: an ETag isn't just bandwidth — it can be quota, latency, and origin-load all at once. (Rate limiting itself is covered in [rate limiting, quotas, and abuse protection](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection).)

**Fastly and Cloudflare: surrogate keys / cache tags for API responses.** Fastly's `Surrogate-Key` and Cloudflare's `Cache-Tag` (an Enterprise feature) are the production mechanism behind "tag by entity, purge by entity" from §9. A typical pattern: an e-commerce origin tags each cached API/page response with the IDs of every entity it depends on (`Surrogate-Key: product-123 category-9 brand-acme`), and a single product update issues one purge-by-key that instantly evicts every cached response touching that product — listing pages, the product page, the category page — without the origin enumerating URLs. Fastly's *soft purge* (mark-stale-not-delete) combined with `stale-while-revalidate` is the well-documented way these CDNs invalidate hot API content without inducing a stampede. These are accurate, vendor-documented capabilities; the specifics of tag limits and pricing tiers vary, so check the current docs before you design around exact quotas.

**Stripe and the broader pattern.** Many payment and commerce APIs lean on the same building blocks this post describes — `ETag`s for conditional reads and version tokens, `no-store` on responses with sensitive material, and `Idempotency-Key` (a separate mechanism) for safe write retries. Rather than cite a specific Stripe header behavior I'm not certain is current, the accurate, general lesson is the division of labor: *caching headers make reads cheap; idempotency keys make writes safe; they are different tools and a mature payments API uses both.* When in doubt about a specific provider's current header semantics, read their reference docs — caching behavior is exactly the kind of detail that changes between API versions.

## When to reach for this (and when not to)

Caching is close to free upside on read-heavy, change-light data, but every directive is a trade-off and some choices are actively harmful. Be decisive:

- **Do** put an `ETag` on every cacheable `GET`. It costs little, makes revalidation nearly free, and doubles as your optimistic-concurrency version token. There is rarely a reason not to.
- **Do** use `private` for anything personalized and `public` + `s-maxage` for anything identical-for-everyone and read-heavy. Match the directive to *who the response is for*.
- **Don't** reach for `no-store` when you mean `private` or `no-cache`. `no-store` forfeits *all* caching, including the cheap validation win. Use it only for genuinely sensitive bodies.
- **Don't** cache personalized or authenticated responses in a *shared* cache without `Cache-Control: private` (or, if you must, a correct `Vary: Authorization`). A missing `Vary: Authorization` on a `public` per-user endpoint leaks one user's data to another — a security incident, not a perf bug.
- **Don't** set a long TTL on data that genuinely changes unless you also have a purge path (or a validator). The TTL is your worst-case staleness, and a refund invisible for an hour is a support escalation.
- **Don't** `Vary` on high-cardinality headers like `User-Agent` — you shatter the cache into millions of entries and the hit ratio collapses. Vary only on low-cardinality headers that truly change the response.
- **Don't** assume GraphQL or RPC gets HTTP caching for free. It doesn't. If cheap layered read-caching is a hard requirement, that is a real point in REST's favor; if you've already chosen GraphQL, budget for persisted queries.
- **Don't** hard-purge a hot key under load. Use soft purge plus `stale-while-revalidate`, or coalesce the misses, or you'll trade a stale entry for a stampede.

## Key takeaways

- **Freshness avoids the trip; validation makes the trip cheap.** A fresh response is served with zero origin contact; a stale one is revalidated, and an `ETag` makes that a bodyless `304`. Use both: `max-age` for the freshness window, an `ETag` so the revalidation is nearly free.
- **`no-cache` and `no-store` are not synonyms.** `no-cache` means "store it, but always revalidate before serving" — you keep the validation win. `no-store` means "store nothing, ever" — reserve it for secrets. Confusing them forfeits caching for free.
- **`public`/`private` is correctness, not just performance.** `private` keeps personalized responses out of shared caches; a missing `private` (or `Vary: Authorization`) on a per-user endpoint leaks one user's data to another.
- **The `ETag` does double duty.** `If-None-Match` gives you cheap `304` revalidation; `If-Match` gives you optimistic concurrency that turns a silent lost update into a visible `412 Precondition Failed` the client can recover from.
- **`Vary` keys the cache by the headers the response depends on.** Vary on `Accept`, `Accept-Encoding`, and (carefully) `Authorization`; never on high-cardinality headers like `User-Agent`.
- **Invalidation is layered: TTL, purge, versioned URLs.** TTL is cheap but lags, purge is instant but needs reach, versioned URLs sidestep the problem. Never set a TTL longer than you'll tolerate stale data when purge fails.
- **A cache miss on a hot key amplifies load.** Defend with request coalescing (one fetch serves all concurrent misses) and `stale-while-revalidate` (serve stale, refresh in the background); use soft purge, not hard purge, on hot keys.
- **REST gets caching for free; RPC and GraphQL must reinvent it.** A `GET` with the resource in the URL is cacheable everywhere. `POST`-tunneled reads aren't — GraphQL claws some of it back with persisted queries, but it's a rebuild of what REST had by default.

## Further reading

- **RFC 9111 — HTTP Caching.** The authoritative spec for `Cache-Control`, freshness, validation, and how caches must behave. The source of truth for everything in §2 and §3.
- **RFC 9110 — HTTP Semantics.** Defines conditional requests, the validator headers (`ETag`, `If-None-Match`, `If-Match`, `Last-Modified`, `If-Modified-Since`), strong vs weak validators, and `304`/`412`/`428` semantics.
- **RFC 5861 — HTTP Cache-Control Extensions for Stale Content.** Defines `stale-while-revalidate` and `stale-if-error`.
- **MDN Web Docs — HTTP caching and the `Cache-Control` header.** The most readable practical reference; excellent for the directive-by-directive behavior across real browsers and caches.
- **Fastly and Cloudflare documentation — surrogate keys / cache tags and soft purge.** Vendor docs for tag-based invalidation and stampede-safe purging of API responses at the edge.
- Within this series: the foundational [HTTP for API designers: methods, status codes, and headers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers), the companion [status codes that tell the truth: 2xx, 3xx, 4xx, 5xx](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx), the intro hub [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), and the capstone [the API design playbook: a review checklist from first endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- For the storage layer beneath a cacheable read — how the database finds a row fast enough that caching it is worth it — see [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work).

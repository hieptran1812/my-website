---
title: "Content Negotiation, Media Types, and Versioned Representations"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn the difference between a resource and its representation, how Accept and Content-Type let a client and server negotiate which bytes to exchange, how media types and vendor types are structured, and how a media type quietly becomes a versioning lever."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "content-negotiation",
    "media-types",
    "versioning",
    "representations",
    "rfc-9110",
    "accept-header",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/content-negotiation-media-types-and-representations-1.png"
---

The finance team filed a bug against the Orders API that, on paper, made no sense. "Your export endpoint is broken," it said. "The CSV has a column called `lineItems` with the literal text `object Object` in every row." I went looking for the export endpoint. There wasn't one. There was a `GET /orders/{id}` that returned JSON, and someone on the finance side had written a script that fetched the JSON, picked a few fields, and dumped them into a spreadsheet by hand. Their script flattened a nested array into a string, badly, and the result was the broken CSV they were now blaming on us. The deeper problem was not the script. It was that there was no honest way for finance to ask the API for the thing they actually wanted — a flat CSV row per order — so they had reverse-engineered one out of a representation that was never meant for them. We owned a `GET /orders/{id}` that spoke exactly one language, and every consumer who needed a different one was forced to translate.

That is the failure this post is about, and the cure is a single idea that the web has carried since the 1990s and that most API designers never name out loud: **a resource and its representation are not the same thing.** The order with ID `4821` — the customer, the line items, the amount, the status — is a *resource*. It is an abstract thing your system knows about. The bytes you send over the wire to describe that order are a *representation* of it, and there can be more than one: a JSON document for the mobile app, a CSV row for the finance spreadsheet, a PDF for the customer's receipt. They all describe the same order. They are different *views* of it, and the machinery that lets a client and a server agree on which view to exchange — without inventing a new endpoint for every format — is called **content negotiation.**

![a diagram showing one order resource fanning out to three representations as JSON for the app, CSV for finance, and PDF for the receipt, all selected by the client Accept header](/imgs/blogs/content-negotiation-media-types-and-representations-1.png)

This post is the layer of the contract where the client gets to say "this is the format I can read" and the server gets to honor it or honestly refuse. By the end you will be able to: explain why one URL can legitimately serve JSON, CSV, and PDF; read and write the `Accept` and `Content-Type` headers, including the `q`-value preference math that ranks competing offers; decompose any media type — `application/json`, `text/csv`, `application/vnd.acme.order+json` — into its grammar of type, subtype, structured suffix, and parameters; use a **vendor media type** as a quiet versioning lever before the dedicated versioning post drives it home; return `406 Not Acceptable` and `415 Unsupported Media Type` for the two different things they actually mean; set the `Vary` header so a CDN does not serve the CSV to a client that asked for JSON; and — most importantly — decide *when* multiple representations earn their keep and when a JSON-only API is exactly the right, boring, correct answer. We will run the whole thing through the **Payments & Orders** example: an order as JSON for the app, CSV for finance, and a versioned vendor type for partners who pin to a schema.

This post is the principle-and-mechanics layer that sits directly on top of [HTTP for API designers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers), where we first met `Accept` and `Content-Type` as two of the headers that matter; it leans forward into [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning), where media-type versioning gets weighed against URI and header versioning in full; and it sits alongside [designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming), which decides what goes *inside* the representation once you have negotiated which one to send. Everything ties back to the spine of [the whole series](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems): an API is a contract you must be able to evolve, and the resource-versus-representation split is one of the cleanest seams you have for evolving it without breaking anyone.

## 1. Resource versus representation: the distinction the web is built on

Let me define the two words precisely, because the entire post hinges on keeping them apart.

A **resource** is any thing your API names with a URL. `/orders/4821` is a resource. `/customers/77/payment-methods` is a resource. The resource is not its bytes; it is the *concept* — the order that exists in your database, in your domain, in the customer's account. If you delete the JSON file on disk and regenerate it tomorrow, the resource is still the same order. The URL is a stable name for a stable concept.

A **representation** is a concrete sequence of bytes that *describes* a resource at a moment in time, in a particular format, together with metadata that says what that format is. The same order resource has a JSON representation, a CSV representation, and a PDF representation. None of them *is* the order. Each is a rendering of it, chosen to suit a particular reader. RFC 9110, the HTTP semantics specification, is blunt about this: a target resource can have "one or more representations," and the job of a `GET` is to "transfer a current representation of the target resource." Not *the* representation — *a* representation. The plural is doing real work.

Once you internalize this, a lot of design questions answer themselves. Should the CSV export be a separate endpoint `/orders/4821/export.csv`, or the same `/orders/4821` with a different `Accept` header? The resource is the same order in both cases, so the *purest* REST answer is one URL with negotiated representations. (Whether that purity is worth the cost is a real question we will weigh honestly later — sometimes a distinct URL is the pragmatic winner. But know what you are trading.) Should adding a PDF receipt require a database change? No — the resource already exists; you are adding a new way to *render* it. Should two clients that ask for the same order in the same format get byte-identical bodies? Yes, ideally, because they are asking for the same representation of the same resource.

> **The principle.** A URL identifies a *resource*; the bytes on the wire are a *representation* of that resource; and content negotiation is the protocol by which the client states which representations it can consume and the server selects one. This is not an optimization or a convenience feature bolted onto HTTP. It is baked into the meaning of `GET`: "transfer a current representation." The headers `Accept`, `Accept-Language`, `Accept-Encoding`, and `Accept-Charset` are the client's side of the conversation; `Content-Type`, `Content-Language`, and `Content-Encoding` are the server's answer, the metadata that says "here is what I actually sent." The negotiation is *part of the contract*, not an afterthought.

Here is the smallest concrete example. A mobile app fetches an order to render a screen:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/json
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
Vary: Accept
ETag: "v3-9f1c"

{
  "id": "4821",
  "status": "paid",
  "currency": "USD",
  "amount": 24900,
  "line_items": [
    { "sku": "WIDGET-1", "qty": 2, "unit_price": 9950 },
    { "sku": "SHIP-STD", "qty": 1, "unit_price": 5000 }
  ]
}
```

Note that the amount is in integer minor units — `24900` cents, a \$249.00 order — because representing money as a floating-point dollar value is a classic way to lose a cent on rounding; that is a body-shape decision covered in [designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming), not a negotiation decision. What matters here is the `Content-Type: application/json` line and the `Vary: Accept` line — the server is telling the client (and any cache between them) "I selected the JSON representation, and the representation I serve from this URL *varies* depending on what you put in `Accept`."

Now finance fetches the *same resource* and asks for CSV instead:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: text/csv
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: text/csv; charset=utf-8
Vary: Accept
Content-Disposition: attachment; filename="order-4821.csv"

order_id,status,currency,amount_cents,line_item_count
4821,paid,USD,24900,2
```

Same URL. Same resource. Different bytes, because finance asked for a different *representation* of the same order. The CSV is not a lossy hand-translation of the JSON anymore; it is a first-class view the server produces deliberately, with the line items flattened to a count or to repeated rows in a way the server controls. The bug that started this post — `object Object` in a column — cannot happen, because the server, not a downstream script, decides how a nested array becomes flat tabular data.

This is the whole game. The rest of the post is the mechanics: how the client says what it wants, how the server chooses, what the media type strings actually mean, how to refuse honestly, and how to keep a cache from getting it wrong.

## 2. Proactive (server-driven) negotiation and the `Accept` header

There are two broad styles of content negotiation in HTTP, and the web overwhelmingly uses the first.

In **proactive negotiation** — also called server-driven negotiation — the client sends preference headers describing what it can handle, and the *server* picks the best representation it can produce and sends it. The client does not see a list of choices; it just gets the server's pick. This is the common case: you send `Accept: application/json` and you get JSON back. The negotiation is invisible because it usually succeeds on the first try.

In **reactive negotiation** — also called agent-driven negotiation — the server responds with a list of available representations (often a `300 Multiple Choices`) and lets the *client* choose with a second request. This is rare on real APIs and we will cover it briefly in its own section; for now, assume proactive.

The client's primary instrument is the `Accept` header. It is a comma-separated list of media types the client is willing to receive, each optionally annotated with a quality value — a `q`-value between `0` and `1` — that expresses *relative preference*.

```http
Accept: text/csv, application/json;q=0.8, */*;q=0.1
```

Read that as a ranked wishlist. The client most prefers `text/csv` (no explicit `q` means `q=1.0`, the maximum). Failing that, it will take `application/json` (`q=0.8`). As an absolute last resort it will take literally anything, `*/*`, but it really would rather not (`q=0.1`). The `q`-value is a preference *weight*, not a hard cutoff: it tells the server how to rank competing offers, not which ones are forbidden. A `q=0` is the one special case — it means "I refuse this type," explicitly removing it from consideration.

> **The principle, made rigorous.** Server-driven negotiation is a constrained maximization. The server has a set of representations it *can* produce, each with its own media type. The client supplies a preference function over media types via the `Accept` `q`-values. The server's job is to compute, over the intersection of "types I can produce" and "types the client will accept with $q > 0$," the representation that maximizes $q$, breaking ties by the *most specific* matching `Accept` entry (an exact `text/csv` beats a `text/*` wildcard beats a `*/*` wildcard, regardless of `q`, per RFC 9110's specificity rule) and then by server preference. If that intersection is empty — every type the client will accept is one the server cannot produce — the server has nothing to send and must say so with `406 Not Acceptable`. The math is trivial; the discipline is in computing it correctly and refusing honestly when it comes up empty.

![a timeline showing a client sending Accept, the server ranking offers by q-value, finding the best match, and returning either a 200 with the chosen Content-Type or a 406 Not Acceptable](/imgs/blogs/content-negotiation-media-types-and-representations-2.png)

Let me make the ranking concrete. Suppose finance's reporting tool sends:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: text/csv;q=1.0, application/json;q=0.5
```

The server can produce both `text/csv` and `application/json`. It computes: `text/csv` is acceptable at `q=1.0`, `application/json` at `q=0.5`. The maximum is `text/csv`. It serves CSV. Now suppose a different client sends only `Accept: application/json`. The server cannot match CSV (the client did not ask for it), but `application/json` is offered and acceptable, so it serves JSON. The same endpoint serves two different bodies to two different callers, each getting the most-preferred thing it can read, and neither client had to know that the other format even existed.

### The `q`-value math, worked precisely

The `q`-value grammar is finicky and worth pinning down, because servers and clients both get it subtly wrong. A `q`-value is a fixed-point number from `0` to `1` with at most three decimal places: `q=0.001` through `q=1.000`. Absence of `q` means `q=1.0`. The server ranks each *acceptable* representation by the `q` of the most specific `Accept` entry that matches its media type. "Most specific" has a strict order: a fully specified `type/subtype` (like `text/csv`) is more specific than a `type/*` wildcard (like `text/*`), which is more specific than the `*/*` wildcard.

Consider this header against a server that can produce `text/csv`, `text/html`, and `application/json`:

```http
Accept: text/*;q=0.5, text/csv;q=0.9, */*;q=0.1
```

For `text/csv`, two entries match: `text/csv` (specific, `q=0.9`) and `text/*` (wildcard, `q=0.5`) and `*/*` (`q=0.1`). The most specific match wins, so `text/csv` scores `0.9`. For `text/html`, the matches are `text/*` (`q=0.5`) and `*/*` (`q=0.1`); the most specific is `text/*`, so `text/html` scores `0.5`. For `application/json`, only `*/*` matches, so it scores `0.1`. The ranking is `text/csv` (`0.9`) > `text/html` (`0.5`) > `application/json` (`0.1`), and the server serves CSV. Note that the more-specific rule means `text/csv;q=0.9` beats `text/*;q=0.5` *even though both contain `text/csv`* — specificity is checked before the `q`-value comparison only when resolving which entry applies to a candidate, then `q` decides among candidates. Getting this right is what separates a negotiation layer that does what callers expect from one that surprises them.

## 3. The other preference dimensions: language, encoding, charset

`Accept` negotiates the *media type* — the format. But content negotiation has three more orthogonal axes, each with its own request header and its own response counterpart, and all four can negotiate at once.

`Accept-Language` negotiates **natural language**. A client can ask for French, falling back to English:

```http
Accept-Language: fr-FR, fr;q=0.9, en;q=0.5
```

If your error messages, product descriptions, or receipts are localized, the server picks the language the same way it picks the format — by `q`-ranked best match — and answers with `Content-Language: fr-FR`. For a machine-to-machine API this is often irrelevant (error *codes* should be language-neutral and machine-readable, with human prose as a secondary `detail` field). But the moment a representation is meant for a human — a PDF receipt, a localized product name — `Accept-Language` is the right lever, and it shares the exact same `q`-value and `Vary` mechanics as `Accept`.

`Accept-Encoding` negotiates **content coding**, almost always compression. This one earns its keep on nearly every API:

```http
Accept-Encoding: br, gzip, deflate
```

The client is saying "I can decompress Brotli, gzip, or deflate." The server picks one, compresses the body, and answers with `Content-Encoding: gzip` (or `br`). This is pure transfer-layer negotiation — the *representation* (the JSON document) is identical; only its on-the-wire byte encoding changes. A 200 KB JSON response can compress to roughly 20–40 KB with gzip on typical structured data with repeated keys, which on a cold mobile link is the difference between a snappy response and a sluggish one. Critically, `Content-Encoding` is *not* the same as `Transfer-Encoding`: `Content-Encoding` is a property of the representation (the client decompresses it and gets the canonical bytes), while `Transfer-Encoding` is a hop-by-hop framing concern. For API design you almost always want `Content-Encoding` negotiated via `Accept-Encoding`, and you almost always want to support gzip at minimum.

`Accept-Charset` negotiates the **character set**, but it is largely historical. RFC 9110 deprecates it; modern practice is to serve UTF-8 everywhere and carry the charset as a *parameter* on the `Content-Type` (`application/json; charset=utf-8`) rather than negotiate it. JSON in particular is defined to be UTF-8, so a `charset` parameter on `application/json` is technically redundant and some specs say to omit it. The lesson is that not every negotiation axis is worth using; charset negotiation is one you can almost always skip.

The important mental note: these four axes are *independent and simultaneous*. A single request can carry `Accept`, `Accept-Language`, and `Accept-Encoding`, and the server resolves each separately — format, language, compression — then sends one body with `Content-Type`, `Content-Language`, and `Content-Encoding` all set. This is exactly why the `Vary` header (Section 8) must list *every* request header the server used to choose, or a cache will serve the wrong combination to the wrong client.

#### Worked example: compression negotiation and the byte budget

It is worth a quick, honest quantification of *why* `Accept-Encoding` is the negotiation axis that nearly always pays off, because the gain is the kind of thing you can reason about with arithmetic rather than fold into hand-waving. Take a paginated `GET /orders?limit=50` that returns fifty order objects. JSON is verbose by design: every object repeats the same key names (`"id"`, `"status"`, `"currency"`, `"amount"`, `"line_items"`, and so on), so a list of fifty similar objects is dominated by repeated, highly compressible strings. A raw response in the neighborhood of 200 KB is realistic for fifty rich order objects. Gzip on structured JSON with repeated keys commonly achieves a 5×–10× reduction, so that 200 KB body lands around 20–40 KB on the wire after `Content-Encoding: gzip`; Brotli (`br`) typically shaves a bit more again on text. These are ranges, not a benchmark from a specific machine — actual ratios depend on the data's entropy — but the order of magnitude holds across almost any JSON API.

Now the latency arithmetic that makes it matter. On a cold mobile link with, say, an effective throughput of around 1 Mbps (roughly 125 KB/s) and meaningful round-trip latency, transferring 200 KB is on the order of 1.5–2 seconds of pure transfer time, while transferring 30 KB is on the order of a couple hundred milliseconds. That difference — better than a second, on every paginated fetch, repeated across a session — is the entire reason `Accept-Encoding: gzip` is essentially free money. The negotiation costs the client one header and the server a CPU-cheap compression pass; the payoff is a multiplicative cut in the bytes that dominate the response time. The trap, as always, is the cache: if you compress, you *must* add `Accept-Encoding` to `Vary` (Section 8), or a shared cache may hand a gzip body to a client that never said it could decompress one — turning a performance win into a corrupted-response bug. Compression negotiation is the one axis I would turn on by default; the rest you turn on when a consumer asks.

## 4. Media types in depth: the `type/subtype` grammar

We have been throwing around strings like `application/json` and `text/csv` and `application/vnd.acme.order+json`. These are not arbitrary labels. They are **media types** (the term "MIME type" is the older name; HTTP now calls them media types), and they follow a precise grammar defined in RFC 6838. Decomposing one tells a client everything it needs to know to parse the bytes before it reads a single one.

![a layered breakdown of a media type into top-level type, subtype, structured suffix, and parameter, using application slash vnd dot acme dot order plus json with a charset parameter](/imgs/blogs/content-negotiation-media-types-and-representations-3.png)

The shape is `type/subtype` optionally followed by `; parameter=value` pairs. Take it apart:

The **top-level type** is the coarse family: `application`, `text`, `image`, `audio`, `video`, `multipart`, `message`, `font`, `model`. For APIs you live almost entirely in `application` (structured data meant for a program) and occasionally `text` (human-readable line-oriented data like `text/csv` or `text/plain`). The distinction between `application/json` and a hypothetical `text/json` is historical and settled: JSON is `application/json` because it is structured data for a program, not free text for a human to read line by line.

The **subtype** is the specific format within the family: `json`, `xml`, `csv`, `pdf`, `octet-stream` (the generic "arbitrary bytes" type). The subtype is what actually tells the parser which library to reach for.

Subtypes live in registration **trees** that signal who owns the name:

- The **standards tree** has no prefix: `application/json`, `application/xml`, `text/html`. These are registered with IANA and are globally meaningful. You do not invent these.
- The **vendor tree** uses the `vnd.` prefix: `application/vnd.github+json`, `application/vnd.api+json`, `application/vnd.acme.order+json`. This is the tree *you* use for a media type that is specific to your product or organization. The `vnd.` says "this name means something to a particular vendor, not the whole world." This is the tree we will use as a versioning lever.
- The **personal/vanity tree** uses `prs.` and the **unregistered tree** uses `x.` (the older `x-` prefix on the *subtype* is discouraged by RFC 6838 for new types). You will rarely need these.

The **structured syntax suffix** is the `+something` at the end of a subtype: `+json`, `+xml`, `+cbor`. Defined in RFC 6839, the suffix declares the *underlying generic structure* of the format. `application/vnd.acme.order+json` says: "this is a vendor-specific schema (`vnd.acme.order`), and its bytes are encoded as JSON (`+json`)." The payoff is huge: a generic tool that knows nothing about your `order` schema can still see the `+json` suffix, reach for a JSON parser, and successfully parse the structure even if it cannot interpret the semantics. The suffix is the bridge between "this is a specific, possibly-versioned schema" and "any JSON-aware client can still read the bytes." It is the single most important feature for using vendor types without locking out generic tooling.

The **parameters** are `; key=value` pairs after the subtype. The two you will see most are `charset` (`application/json; charset=utf-8` — the character encoding) and the JSON-API-style `profile`. Parameters refine a type without changing its identity: `text/csv; charset=utf-8; header=present` is still `text/csv`, just with extra parse hints (here, that the first CSV row is a header line — a real `text/csv` parameter from RFC 7111). A subtle but important rule: parameters can affect equivalence. `text/html; charset=utf-8` and `text/html; charset=iso-8859-1` are *different* representations of potentially different bytes; a cache must treat them as distinct.

#### Worked example: parsing a vendor media type byte by byte

A partner integration sends this `Accept` header:

```http
Accept: application/vnd.acme.order.v2+json; charset=utf-8
```

Walk it the way the server does. Top-level type: `application` — structured data for a program. Subtype: `vnd.acme.order.v2`. The `vnd.` prefix says vendor tree, so this is *our* type, not a standards-tree type; `acme.order` names the producer and the schema; `.v2` is a version token *we* chose to bake into the subtype. Structured suffix: `+json` — whatever the `order.v2` schema is, its bytes are JSON, so any JSON parser can read the structure. Parameter: `charset=utf-8` — decode the bytes as UTF-8. The server reads this and concludes: "the partner wants version 2 of the order schema, serialized as UTF-8 JSON." It can produce that, so it responds:

```http
HTTP/1.1 200 OK
Content-Type: application/vnd.acme.order.v2+json; charset=utf-8
Vary: Accept
ETag: "v2-3a8e"

{
  "id": "4821",
  "status": "paid",
  "money": { "amount": 24900, "currency": "USD" },
  "items": [
    { "sku": "WIDGET-1", "quantity": 2 }
  ]
}
```

Notice the body shape differs from the plain `application/json` we served in Section 1 — `money` is now a nested object and the field is `items` not `line_items`. That is the point: the `v2` token in the media type told the server *which schema* to render, and the same resource (order `4821`) came back in a different representation. We will return to exactly this lever in Section 10.

## 5. `Content-Type`: the request side of the conversation

`Accept` is the client saying "here is what I can *receive*." Its mirror image is `Content-Type`, which appears on any message *that has a body* and declares what that body *is*. The direction matters, so be precise: when a client `POST`s or `PUT`s a body, the *client* sets `Content-Type` to describe its own request body; when the server responds with a body, the *server* sets `Content-Type` to describe the response. `Accept` is a wish; `Content-Type` is a fact about the bytes actually present.

Here is a `POST` that creates a payment. The client must declare the format of the body it is sending:

```http
POST /payments HTTP/1.1
Host: api.acme.com
Content-Type: application/json
Accept: application/json
Idempotency-Key: 4f9c2b10-3e7a-4b6e-9c2a-1d8f6e0a7b22
Authorization: Bearer <token>

{
  "order_id": "4821",
  "amount": 24900,
  "currency": "USD",
  "source": "card_tok_visa"
}
```

Two header roles in one request. `Content-Type: application/json` tells the server "parse my body as JSON." `Accept: application/json` tells the server "and please send the result back as JSON." They are independent: a client could perfectly well `POST` a JSON body and ask for a CSV confirmation back (`Content-Type: application/json`, `Accept: text/csv`), and the server would parse JSON in and emit CSV out. The request body format and the response body format are negotiated by *different* headers because they are *different decisions*.

The server's obligation is to *validate* the incoming `Content-Type`. If a client sends a body with a `Content-Type` the server cannot parse — say it `POST`s XML to an endpoint that only ingests JSON — the server must refuse with `415 Unsupported Media Type`. This is the request-side mirror of the `406` it would return for an `Accept` it cannot satisfy. Keep the two straight; they are the most-confused status codes in this whole area, and Section 7 nails the distinction down.

> **The consequence of ignoring `Content-Type`.** A server that ignores the incoming `Content-Type` and just tries `JSON.parse()` on whatever arrives is a server that will accept a form-encoded body, a YAML body, or random bytes and either crash with a `500` (a *server* error for a *client* mistake — a lie about whose fault it is) or, worse, silently misparse and write garbage. I have seen an endpoint that accepted `application/x-www-form-urlencoded` "by accident" because a permissive framework auto-parsed it, and a client that was *supposed* to send JSON shipped form-encoding for two years; when the framework was upgraded and tightened, every one of that client's requests started failing, and nobody knew why, because the contract had never actually said "JSON only" — it had just been quietly tolerant. Declaring and enforcing `Content-Type` is part of the contract being *honest* about what it accepts.

## 6. Reactive (agent-driven) negotiation, briefly

Proactive negotiation has the server choose. **Reactive negotiation** flips it: the server, instead of picking, responds with a description of the available representations and lets the *client* make a second request to fetch the one it wants. The canonical status for this is `300 Multiple Choices`, with a body (and optionally a `Link` header) listing the alternatives.

In theory this is elegant — the client has full information and the server does not have to guess. In practice you will almost never use it for an API, for three concrete reasons. First, it costs a round trip: the client asks, gets a menu, then asks again, doubling latency for what proactive negotiation does in one shot. Second, there is no widely-honored standard format for the `300` body, so every client would need bespoke logic to parse your particular menu — which defeats the point of using a standard mechanism. Third, the cases where it genuinely helps (a resource with many wildly different representations and a client that truly cannot express its preference up front) are vanishingly rare in machine-to-machine APIs.

There is one honest middle ground worth knowing: a `406 Not Acceptable` response *may* include a body that lists what the server *could* have offered, which is a degenerate, failure-time form of agent-driven negotiation — "I could not give you what you asked for, but here is what I have." That is genuinely useful for debugging and we will use it in Section 7. But the deliberate, success-path `300 Multiple Choices` dance is a corner of the spec you should know exists and then, for nearly every API, choose not to use. Proactive negotiation with `Accept` is the workhorse; reactive negotiation is the museum piece.

## 7. `406 Not Acceptable` versus `415 Unsupported Media Type`

These two status codes are about negotiation failures, they sit one digit apart, and they are confused constantly. Pin them down by asking *which header failed and in which direction*.

`406 Not Acceptable` is about the **response** and the `Accept` header. It means: "You told me, via `Accept`, the only formats you will accept, and I cannot produce any of them." The client asked for something the server cannot make. Example: a client sends `Accept: application/xml` to an endpoint that only emits JSON and CSV.

`415 Unsupported Media Type` is about the **request body** and the `Content-Type` header. It means: "You sent me a body in a format I cannot parse." The client *gave* the server something the server cannot read. Example: a client `POST`s a body with `Content-Type: application/xml` to an endpoint that only ingests JSON.

The mnemonic that has never failed me: **`406` is about what you can't *get*; `415` is about what you can't *send*.** `406` is the server saying "I can't *produce* that for you." `415` is the server saying "I can't *consume* that from you." One is about `Accept` (the response you want), one is about `Content-Type` (the request you sent).

![a flow diagram showing a request with Accept and Content-Type, where an unparseable body returns 415 and an Accept with no matching offer returns 406, otherwise a 200 with the chosen type and a Vary header](/imgs/blogs/content-negotiation-media-types-and-representations-6.png)

#### Worked example: an `Accept` negotiation returning JSON versus CSV, and a `406`

Three requests to the same `GET /orders/4821`, with the server able to produce `application/json` and `text/csv` only.

Request one — the mobile app wants JSON:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
Vary: Accept

{ "id": "4821", "status": "paid", "amount": 24900, "currency": "USD" }
```

Request two — finance wants CSV, and prefers it over JSON if both were available:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: text/csv, application/json;q=0.5
```

```http
HTTP/1.1 200 OK
Content-Type: text/csv; charset=utf-8
Vary: Accept

order_id,status,currency,amount_cents
4821,paid,USD,24900
```

Request three — a partner asks for XML, which the server cannot produce:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/xml
```

```http
HTTP/1.1 406 Not Acceptable
Content-Type: application/problem+json

{
  "type": "https://api.acme.com/problems/not-acceptable",
  "title": "No acceptable representation",
  "status": 406,
  "detail": "This resource can be served as application/json or text/csv. application/xml is not available.",
  "available": ["application/json", "text/csv"]
}
```

Three different outcomes from one URL, each *honest*: JSON to the app, CSV to finance, a `406` to the partner who asked for something that does not exist — and that `406` body lists `available` types so the partner can fix the request without filing a ticket. The error body itself is `application/problem+json`, the RFC 9457 problem-details format that gets its own treatment in the [error design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract); the point here is that even a negotiation *failure* should be machine-readable.

#### Worked example: a `415` on a bad request body

Now the request-body side. A client tries to create a payment but sends an XML body:

```http
POST /payments HTTP/1.1
Host: api.acme.com
Content-Type: application/xml
Accept: application/json

<payment><order>4821</order><amount>24900</amount></payment>
```

```http
HTTP/1.1 415 Unsupported Media Type
Accept-Post: application/json
Content-Type: application/problem+json

{
  "type": "https://api.acme.com/problems/unsupported-media-type",
  "title": "Unsupported request body format",
  "status": 415,
  "detail": "This endpoint accepts application/json request bodies. application/xml is not supported."
}
```

Notice the `Accept-Post` response header — a standard way to advertise which body media types a `POST` target accepts, the request-side analogue of how `406` advertises producible types. (`Accept-Patch` does the same for `PATCH`.) The client now knows *exactly* what to change: send JSON, not XML. No guessing, no `500`, no support ticket.

The contrast is worth stating once more because it is the single most common mistake in this whole area: a `406` is the server refusing to *produce* a representation the client demanded via `Accept`; a `415` is the server refusing to *consume* a body the client sent with the wrong `Content-Type`. Returning a `415` when you meant `406` (or a `400` when you meant either) sends the client debugging in the wrong direction.

## 8. The `Vary` header: why caches need to know you negotiated

Here is a failure that looks like black magic until you understand `Vary`. Your API serves both JSON and CSV from `GET /orders/4821`. A CDN sits in front of it for performance. The mobile app requests the order with `Accept: application/json`, the CDN has nothing cached, so it forwards to the origin, gets JSON back, and *caches the JSON keyed only by the URL*. Thirty seconds later, finance requests the *same URL* with `Accept: text/csv`. The CDN looks up its cache by URL, finds the JSON it stored for the app, and — having no reason to think otherwise — serves the JSON to finance with a `Content-Type: application/json` it never asked for. Finance's CSV importer chokes on a JSON document. The bug is intermittent, depends on who hit the URL first, and is nearly impossible to reproduce locally because there is no CDN in front of your dev box.

The fix is one response header: `Vary`.

> **The principle.** A cache stores responses keyed by the request URL. But when a response *depends on a request header* — when the same URL produces different bytes for different `Accept` values — the URL alone is not a sufficient cache key. The `Vary` header tells the cache: "the representation I just gave you varies by these request headers, so include them in your cache key." `Vary: Accept` means "cache this keyed by URL *and* the `Accept` header." Now the JSON-for-the-app and the CSV-for-finance are stored as two separate cache entries, and each client gets the right one. Omitting `Vary` on a negotiated response is not a style nit; it is a correctness bug that surfaces only behind a shared cache, which is exactly where it is hardest to catch.

The rule is mechanical and absolute: **`Vary` must list every request header your server used to select the representation.** If you negotiated on `Accept`, you need `Vary: Accept`. If you also negotiated language and compression, you need `Vary: Accept, Accept-Language, Accept-Encoding`. If you forget `Accept-Encoding`, a cache may serve a gzip-compressed body to a client that did not send `Accept-Encoding: gzip` and cannot decompress it — another invisible, intermittent failure.

```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Encoding: gzip
Content-Language: en
Vary: Accept, Accept-Encoding, Accept-Language
Cache-Control: private, max-age=60
ETag: "v3-9f1c"

<gzipped JSON bytes>
```

A few practical notes. `Vary: *` exists and means "this response varies on factors not captured by any header, so do not use a shared cache at all" — a sledgehammer that disables caching; reach for it only when you genuinely cannot enumerate the inputs. There is a real cost to `Vary`: every distinct value of a varied header is a separate cache entry, so a header with high cardinality (like a raw `User-Agent`, which has thousands of distinct values) can shatter your cache hit rate into uselessness — which is precisely why you should *never* `Vary` on `User-Agent` for negotiation. Negotiate on the purpose-built headers (`Accept`, `Accept-Language`, `Accept-Encoding`), which clients send with low, predictable cardinality. The deep mechanics of `ETag`, `Cache-Control`, and conditional requests get their own post in [caching with ETags and conditional requests](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation); here the single load-bearing rule is: **if you negotiated, you must `Vary`.**

## 9. Conditional requests are per-representation, not per-resource

There is a second, quieter way negotiation and caching interact, and getting it wrong produces the same class of "served the wrong bytes" bug as a missing `Vary`. It involves the `ETag` — the opaque validator a server attaches to a representation so a client can later ask "has this changed?" with a conditional request. The trap is treating an `ETag` as identifying the *resource* when it actually identifies a *representation*.

Recall the rule from [HTTP for API designers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers): a server stamps a response with `ETag: "v3-9f1c"`, the client caches that, and on its next request it sends `If-None-Match: "v3-9f1c"`. If the representation has not changed, the server replies `304 Not Modified` with no body, saving the transfer. That is the entire conditional-request machinery, and it is one of the highest-leverage performance features in HTTP — a `304` is a few bytes of headers instead of a full payload.

But here is the subtlety that content negotiation forces: **the `ETag` validates a specific representation, not the abstract resource.** The JSON representation of order `4821` and the CSV representation of the *same order* must carry *different* `ETag` values, because they are different bytes. If they shared an `ETag`, a client that cached the JSON under `"v3-9f1c"`, then asked for CSV with `If-None-Match: "v3-9f1c"`, could be told `304 Not Modified` — and would reuse the JSON bytes it had cached as if they were the CSV. That is the `Vary` cache-confusion bug wearing a conditional-request disguise.

> **The principle.** An `ETag` is a validator for the *selected representation*. When a resource has multiple negotiated representations, each must have its own distinct `ETag`, and the conditional-request match is only meaningful *within the same negotiated representation*. The `Vary` header and per-representation `ETag`s work together: `Vary: Accept` tells the cache to key by `Accept`, and the distinct `ETag` per representation makes the `If-None-Match` revalidation honest within each key. Drop either one and a shared cache can hand a client the wrong format on a `304`.

Walk it concretely. The app caches JSON:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/json
If-None-Match: "json-v3-9f1c"
```

```http
HTTP/1.1 304 Not Modified
ETag: "json-v3-9f1c"
Vary: Accept
```

The `304` is correct: same resource, same JSON representation, unchanged, so reuse the cached JSON. Now finance, which has never fetched the CSV, asks for it:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: text/csv
If-None-Match: "json-v3-9f1c"
```

```http
HTTP/1.1 200 OK
Content-Type: text/csv; charset=utf-8
ETag: "csv-v3-2b71"
Vary: Accept

order_id,status,currency,amount_cents
4821,paid,USD,24900
```

The server sees an `If-None-Match` whose `ETag` does not match the CSV representation's `ETag` (`"csv-v3-2b71"` ≠ `"json-v3-9f1c"`), so it correctly returns a full `200` with the CSV — *not* a `304`. The format of the validator string is opaque to the client (it is just a quoted token), but a server implementer typically derives it from both the resource version *and* the chosen media type, precisely so two representations of the same resource can never collide. The lesson generalizes: any time you negotiate, the cache key, the `Vary`, and the `ETag` all have to agree that a representation — not a resource — is the unit being cached and validated. We go deeper on conditional requests in the [caching post](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation); the negotiation-specific rule is the one above: **one `ETag` per representation, never one per resource.**

## 10. Vendor media types as a versioning lever

Now the idea this post has been building toward. Once you accept that a media type names *which schema* the bytes follow, the media type becomes a place to encode *which version of that schema*. This is **media-type versioning**, and it is one of three main strategies (URI versioning and header versioning are the other two), each weighed in full in the dedicated [versioning strategies post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning). Here I want to show *how it works through negotiation*, because it is the most elegant — and most misunderstood — use of everything we have built.

The lever: instead of versioning the *URL* (`/v1/orders/4821` vs `/v2/orders/4821`), keep one URL (`/orders/4821`) and version the *representation* via a vendor media type in the `Accept` header.

![a layered view of the vendor media type application slash vnd dot acme dot order dot v2 plus json, broken into the vnd tree, the producer and schema name, the version token, and the JSON suffix](/imgs/blogs/content-negotiation-media-types-and-representations-8.png)

A v1 client asks for v1; a v2 client asks for v2; both hit the same URL:

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/vnd.acme.order.v1+json
```

```http
HTTP/1.1 200 OK
Content-Type: application/vnd.acme.order.v1+json
Vary: Accept

{
  "id": "4821",
  "status": "paid",
  "currency": "USD",
  "amount": 24900,
  "line_items": [
    { "sku": "WIDGET-1", "qty": 2 }
  ]
}
```

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/vnd.acme.order.v2+json
```

```http
HTTP/1.1 200 OK
Content-Type: application/vnd.acme.order.v2+json
Vary: Accept

{
  "id": "4821",
  "status": "paid",
  "money": { "amount": 24900, "currency": "USD" },
  "items": [
    { "sku": "WIDGET-1", "quantity": 2 }
  ]
}
```

Same resource, same URL, two representations selected by `Accept`. The v2 schema renamed `line_items` to `items` and nested `currency`/`amount` under a `money` object — both *breaking* changes that would have shattered v1 clients if you had simply mutated the plain `application/json` body. By versioning the media type, the v1 client keeps asking for `vnd.acme.order.v1+json` and keeps getting the old shape forever, while the v2 client opts in to the new shape by changing one header. Crucially, the `+json` suffix means a generic, schema-agnostic tool can still parse *both* as JSON; the version lives in the *subtype*, not in a place that breaks JSON parsing.

#### Worked example: a vendor media type carrying a version, and the default

The subtle, important design decision is: **what does the server return when the client does not specify a version?** Two policies, with very different blast radii.

The dangerous policy is "default to latest." A client that sends plain `Accept: application/json` (or no `Accept` at all) gets whatever the newest schema is. This means the day you ship v2, every existing client that did not pin a version *silently flips to v2* and breaks on the renamed field. That is the renamed-field outage from this series' running scar, delivered automatically to everyone who was not paying attention.

The safe policy is "default to the *first* / *pinned* version, and require an explicit opt-in to anything newer." A client that sends plain `Accept: application/json` gets v1 — the oldest stable shape — forever. To get v2, a client must *deliberately* send `Accept: application/vnd.acme.order.v2+json`. New behavior is opt-in, never automatic. This is exactly what GitHub did when it moved off date-pinned versions: a request with no version header gets a documented default, and you opt into a specific version explicitly.

```http
GET /orders/4821 HTTP/1.1
Host: api.acme.com
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/vnd.acme.order.v1+json
Vary: Accept

{ "id": "4821", "status": "paid", "currency": "USD", "amount": 24900,
  "line_items": [ { "sku": "WIDGET-1", "qty": 2 } ] }
```

The client asked for generic `application/json`; the server answered with the *specific* `vnd.acme.order.v1+json` it actually served, pinning the unversioned client to v1 and telling it (via the response `Content-Type`) exactly which schema it got. Notice the asymmetry, which is the whole trick: the client may ask broadly, but the server answers *specifically*, so the unversioned caller is never silently upgraded.

Here is the trade-off table that previews the full versioning post — three places to put a version, and how each leaks.

| Strategy | Where the version lives | URL stays stable? | Main trade-off |
| --- | --- | --- | --- |
| URI versioning (`/v2/orders`) | In the path | No — the path changes | Brutally obvious and easy to route, but fragments the URL space; the same resource has two URLs, breaking caching and bookmarks |
| Header versioning (`API-Version: 2`) | In a custom request header | Yes — same path | Hidden from the URL, so easy to forget; non-standard header name; not as discoverable in logs |
| Media-type versioning (`vnd.acme.order.v2+json`) | In the `Accept` media type | Yes — same path | Purest REST (one resource, many representations), cache-friendly with `Vary`; but verbose, harder to test in a browser address bar, and surprising to engineers who have not seen it |

![a comparison matrix of URI versioning, header versioning, and vendor media-type versioning across where the version lives, whether the URL stays stable, and the main trade-off](/imgs/blogs/content-negotiation-media-types-and-representations-7.png)

The decisive point, which the versioning post defends at length, is that media-type versioning is *philosophically* the cleanest because it respects the resource-versus-representation split — one resource, one URL, many versioned representations — but it is *operationally* the hardest to live with: you cannot paste a versioned `Accept` into a browser address bar, junior engineers will not discover it, and your tooling has to teach every consumer the right header. Many excellent APIs reach for URI versioning precisely because it is impossible to use wrong by accident. Choose by your audience, not by purity. We will not settle that debate here; we are just showing that the *mechanism* falls straight out of content negotiation.

## 11. Negotiation strategies compared, and the case for query-param overrides

Server-driven `Accept` negotiation is the standards-blessed default, but it is not the only way to let a client pick a representation, and the alternatives are not all wrong. Here is the honest comparison.

| Strategy | How the client picks | Cache impact | Pros | Cons |
| --- | --- | --- | --- | --- |
| Server-driven `Accept` | `Accept` header, `q`-ranked | Needs `Vary: Accept` (correct but adds cache keys) | Standard, invisible to the URL, the "right" REST way; one URL per resource | Invisible in the URL/logs; can't paste into a browser; `Vary` mistakes cause cache bugs |
| Agent-driven (`300`) | Server lists choices, client re-requests | Two responses to cache | Client has full information; no server guessing | Extra round trip; no standard menu format; almost nobody uses it |
| Query-param override (`?format=csv`) | A query parameter on the URL | URL is the cache key (simple, correct by default) | Trivial to test in a browser; obvious in logs; no `Vary` foot-gun | Non-standard; conflates "which representation" with "which resource"; noisier URL space |
| Distinct URLs (`/orders/4821.csv`) | A different URL per format | URL is the cache key | Maximally explicit and discoverable; trivially cacheable | Fragments the surface; arguably violates the one-resource-one-URL ideal; combinatorial with versions |

![a matrix comparing server-driven Accept negotiation, agent-driven negotiation, query-param override, and distinct URLs across how each picks a representation, its cache impact, and its trade-off](/imgs/blogs/content-negotiation-media-types-and-representations-4.png)

I want to defend the query-param override, because the spec-purist instinct is to dismiss it and that instinct is sometimes wrong. A `?format=csv` (or a `.csv` URL suffix) has two genuine advantages that matter in the real world. First, it is *trivially debuggable*: anyone can paste `https://api.acme.com/orders/4821?format=csv` into a browser and see the CSV, with no header-editing tooling. For a public API whose consumers include analysts and not just engineers, that is worth a lot. Second, it makes the URL the complete cache key, sidestepping the entire class of `Vary` bugs — different formats are different URLs, so no shared cache can ever confuse them. The cost is that you have smeared format selection across both the URL and (if you also honor `Accept`) the header, which can produce ambiguity ("the URL says `.csv` but `Accept` says JSON — who wins?"). The clean resolution: if you offer a query/suffix override, let it *win* over `Accept` and document that loudly. GitHub, for instance, has historically supported both an `Accept`-header media type *and* a `.json`/`.diff`/`.patch` URL extension on some resources, with the extension taking precedence — pragmatism over purity.

The decision framework: use server-driven `Accept` negotiation as your default, because it is standard and keeps your URL space clean. Add a query-param/suffix override *on top* when a meaningful slice of your audience cannot or will not set headers (public APIs, exports an analyst triggers from a browser, file downloads a user clicks). Reach for fully distinct URLs only when a format is so different it is arguably a different *resource* (a paginated HTML report versus a raw data feed). Skip agent-driven `300` negotiation entirely unless you have a specific reason it is the only thing that works.

## 12. When multiple representations earn their keep — and when JSON-only is the right answer

This is the section where I talk you *out* of content negotiation as often as into it, because the most common content-negotiation mistake is building it before you need it.

> **The honest default.** Most APIs should serve JSON and only JSON, and that is completely fine. If every consumer of your API is a program that wants structured data, a single `application/json` representation is the simplest correct contract you can ship. You still set `Content-Type: application/json` on responses, you still validate the incoming `Content-Type` and return `415` for non-JSON bodies, you still set `Vary: Accept-Encoding` if you compress — but you do not build a CSV renderer, a PDF generator, or a vendor-type negotiation matrix that no client will ever exercise. Building multi-representation negotiation "for flexibility" before a second representation has an actual consumer is a textbook case of speculative generality: you pay the complexity now and the flexibility may never be used. A JSON-only API is not an unsophisticated API; it is an API that correctly scoped its representations to its consumers.

So when does a second representation genuinely earn its keep? Three patterns, each with a real consumer driving it.

**Exports and reporting.** When humans — finance, ops, analysts — need to pull data into spreadsheets, a `text/csv` (or `application/vnd.ms-excel`) representation is a real feature with a real user. This is the case that started this post: finance was *going* to get CSV one way or another; the only question was whether the server produced it correctly or a downstream script produced it badly. A negotiated CSV representation, with a `Content-Disposition: attachment; filename=...` so the browser downloads it as a file, turns a fragile hand-translation into a supported contract.

![a before and after contrast where one rigid JSON-only format forces finance to build a fragile side pipeline, versus negotiated representations where one URL serves JSON to the app and CSV to finance from a single source of truth](/imgs/blogs/content-negotiation-media-types-and-representations-5.png)

**File downloads and documents.** A PDF receipt, an invoice, a generated report — these are representations of a resource (the order, the invoice) that happen to be binary documents meant for a human to save or print. `application/pdf` with `Content-Disposition: attachment` is the right tool. The order resource does not change; you are adding a human-facing rendering of it.

**Public APIs with diverse consumers.** When you do not control your callers — a public developer platform, a partner ecosystem — you may genuinely have consumers who want JSON, consumers who want XML (legacy enterprise integrations are stubborn), and consumers who want a versioned vendor type so they can pin to a schema. Here, content negotiation is doing exactly its job: serving a heterogeneous audience from one resource model without forcing everyone onto the lowest common denominator.

And when *not* to: an internal microservice talking to three other internal services you also own — ship JSON, skip negotiation, and if you ever need a second format, add it then. A mobile-only backend — your one client wants JSON; a CSV renderer is dead code. A high-throughput service where every microsecond counts — negotiation logic is overhead you do not need if there is one format. The rule of thumb: **add a representation when a real, named consumer needs it, never on speculation.** You can always add `text/csv` later — it is a purely additive change (a new representation the server can produce), so it never breaks an existing JSON client. That additive-safety is itself a reason *not* to build it early: there is no penalty for waiting.

## 13. Case studies: how real APIs negotiate

Theory is cheap; let me ground this in APIs you can go inspect yourself. I will be careful to state only what is accurate and publicly documented, and to frame anything I am less sure of generally.

**GitHub: `application/vnd.github+json` and media-type versioning.** GitHub's REST API uses the vendor media type `application/vnd.github+json` and documents it as the recommended `Accept` value. Historically GitHub versioned its API through the media type — clients would send something like `application/vnd.github.v3+json` to pin to a major version, exactly the vendor-type-as-version-lever pattern from Section 10. GitHub also supported *custom media types* to request alternate representations of the same resource — for example asking for the raw, the HTML-rendered, or the text-matched form of a resource by varying the media type — which is the resource-versus-representation split applied at scale: one resource, several deliberately different renderings selected by `Accept`. (GitHub later layered a date-based `X-GitHub-Api-Version` header on top for finer-grained versioning; the relevant lesson for *this* post is that the media type was a real, production versioning and representation-selection lever for years.) The takeaway: a major public API ran for a long time on exactly the media-type negotiation this post describes.

**JSON:API: `application/vnd.api+json` and the parameter rules.** The JSON:API specification defines a vendor media type, `application/vnd.api+json`, and — notably — specifies strict rules about *parameters* on that media type. JSON:API servers must respond with `415 Unsupported Media Type` if a client sends the JSON:API media type with any *media-type parameters* in `Content-Type`, and must respond with `406 Not Acceptable` if every instance of the JSON:API media type in `Accept` is modified with parameters. This is a real, in-the-wild example of `406` and `415` being used for *exactly* the two purposes Section 7 describes — refusing to consume a malformed `Content-Type` (`415`) versus refusing to produce when the `Accept` cannot be honored (`406`) — and of media-type *parameters* being load-bearing in a negotiation. It is also a good cautionary tale: JSON:API's parameter strictness has tripped up many client libraries that naively appended `charset=utf-8`, which is precisely why parameter handling deserves care.

**HAL: `application/hal+json` as a hypermedia representation.** HAL (Hypertext Application Language) defines the media type `application/hal+json` (and `application/hal+xml`), a convention for embedding hypermedia links inside a JSON document via reserved `_links` and `_embedded` members. For this post, HAL is interesting as a *representation choice*: a server can offer the *same* resource as plain `application/json` (just the data) or as `application/hal+json` (the data plus navigation links), and a client selects via `Accept` whether it wants the hypermedia-enriched view. The `+json` suffix again does its job — a client that does not understand HAL's link conventions can still parse the body as ordinary JSON and ignore the `_links`. Whether you *should* serve HAL links is the subject of the [HATEOAS post](/blog/software-development/api-design/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip); here HAL is simply a clean example of two representations of one resource distinguished by media type.

**RFC 9457 problem details: `application/problem+json`.** Every error body in this post used `application/problem+json`, the media type RFC 9457 (which obsoleted RFC 7807) defines for machine-readable problem details. It is a registered standards-tree type with a `+json` suffix, so it negotiates and parses like any other JSON representation, but its media type signals to a client "this body is a structured error, not the success payload you asked for." That a *dedicated media type for errors* exists is itself the resource-versus-representation principle applied: the same `400`-class failure can be described by a structured representation the client can branch on, rather than a prose blob. The full error contract is the subject of the [error design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract).

The common thread across all four: the media type is not decoration. It is the load-bearing identifier that tells a client which schema, which version, which rendering, and even whether the body is a success or an error — all selected through the negotiation machinery we built.

## 14. A problem-solving narrative: designing the Orders export

Let me put the whole post to work on the actual problem that opened it. Finance needs order data in spreadsheets. We have a `GET /orders/{id}` returning JSON and a `GET /orders` collection returning paginated JSON. What do we ship?

**Step one: name the consumer and the need.** The consumer is the finance team; the need is "a flat, spreadsheet-friendly row per order, downloadable as a file." This is a real, named consumer — it passes the "earns its keep" test from Section 12. We are not speculating.

**Step two: resource or representation?** Is the CSV a new resource or a new representation of the existing orders? It is the *same orders* — same IDs, same data, same access controls — rendered as flat rows. So it is a *representation*, which means content negotiation, not a brand-new resource tree. We will serve `text/csv` from the existing `/orders` and `/orders/{id}` URLs.

**Step three: how does finance ask?** Two options. Server-driven: finance sends `Accept: text/csv`. Query-param override: finance hits `/orders?format=csv`. Finance's tool is partly a browser-driven analyst workflow and partly a scheduled script. The scheduled script can set `Accept`; the analyst clicking a link cannot easily. So we honor *both*: `Accept: text/csv` for the script, and `?format=csv` (winning over `Accept` when present) for the browser link, documented loudly. We set `Content-Disposition: attachment; filename="orders.csv"` so the browser downloads a file.

**Step four: set `Vary`.** Because the same URL now produces JSON or CSV depending on `Accept`, every negotiated response gets `Vary: Accept`. We confirm our CDN config honors `Vary` (some misconfigured CDNs strip it — that is a thing to *test*, per step seven).

**Step five: define the CSV representation deliberately.** The JSON has nested `line_items`. CSV is flat. We decide: the `/orders/{id}.csv` gives *one row per order* with a `line_item_count`, and a separate `/orders/{id}/line-items` with `Accept: text/csv` gives one row per line item. We do *not* let a downstream script flatten the nested array — the server owns the flattening, correctly, so `object Object` can never appear.

Now stress-test the design, the way the kit demands.

*What if finance asks for CSV on the paginated collection, `GET /orders?status=paid`, with 2 million matching orders?* A CSV of 2 million rows is a multi-hundred-megabyte body that will time out and blow memory. This is where representation negotiation collides with the *long-running export* problem: a synchronous `GET` is the wrong tool. The right answer is an asynchronous export job — `POST` an export request, get a `202 Accepted` with a status URL, poll, and download the finished CSV from a signed URL — which is the [long-running operations pattern](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks), not content negotiation. The lesson: negotiation chooses the *format*; it does not solve *volume*. Know the seam.

*What if a client sends `Accept: text/csv` to an endpoint where CSV makes no sense — say `POST /payments`?* You return `406 Not Acceptable`, because you cannot produce a CSV representation of a freshly created payment in a way the client could mean. Negotiation can fail, and failing honestly with `406` is correct.

*What if we later add a third representation, `application/pdf`, for receipts?* Purely additive. Existing JSON and CSV clients are unaffected — they never asked for PDF, so they never get it. This is the additive-safety property: a new representation is one of the safest changes you can make, which is exactly why you can defer it until a real consumer (the customer who wants a printable receipt) shows up.

*What if a cache in front of us ignores `Vary`?* Then JSON leaks to CSV clients intermittently — the black-magic bug from Section 8. We *verify* this in step seven rather than assume it.

**Step six: write it down in the contract.** The OpenAPI description lists each operation's `produces` set per content type, so `GET /orders/{id}` documents that it can return `application/json`, `text/csv`, and `application/vnd.acme.order.v2+json`, and the `responses` enumerate `200`, `406`, and `415`. The negotiation is now *discoverable* — a consumer reading the spec sees every representation on offer.

```yaml
paths:
  /orders/{orderId}:
    get:
      summary: Retrieve an order
      parameters:
        - name: orderId
          in: path
          required: true
          schema: { type: string }
        - name: format
          in: query
          required: false
          description: Optional override; wins over the Accept header when present.
          schema: { type: string, enum: [json, csv] }
      responses:
        "200":
          description: An order representation, negotiated by Accept.
          content:
            application/json: { schema: { $ref: "#/components/schemas/OrderV2" } }
            application/vnd.acme.order.v2+json: { schema: { $ref: "#/components/schemas/OrderV2" } }
            application/vnd.acme.order.v1+json: { schema: { $ref: "#/components/schemas/OrderV1" } }
            text/csv: { schema: { type: string } }
        "406": { description: No acceptable representation; see Accept header. }
        "415": { description: Unsupported request body media type. }
```

**Step seven: verify it.** Contract tests assert that `Accept: application/json` returns `application/json` and `Accept: text/csv` returns `text/csv`; that `Accept: application/xml` returns `406` with a problem body listing `available`; that `POST` with `Content-Type: application/xml` returns `415`; and — the easy one to forget — that the `200` carries `Vary: Accept`. A request-level test through a real shared cache (or a CDN staging tier) confirms the `Vary` is honored end-to-end, catching the cache-confusion bug *before* a customer does.

```bash
# JSON for the app
curl -sD - https://api.acme.com/orders/4821 \
  -H 'Accept: application/json' | grep -i '^content-type\|^vary'

# CSV for finance
curl -sD - https://api.acme.com/orders/4821 \
  -H 'Accept: text/csv' | grep -i '^content-type\|^vary'

# A format the server can't produce -> 406
curl -s -o /dev/null -w '%{http_code}\n' https://api.acme.com/orders/4821 \
  -H 'Accept: application/xml'

# A body format the server can't consume -> 415
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://api.acme.com/payments \
  -H 'Content-Type: application/xml' --data '<x/>'
```

That is the full arc — name the consumer, classify resource versus representation, choose the negotiation mechanism, set `Vary`, define the representation deliberately, document it, and verify it under a cache. Every decision traced back to the resource-versus-representation principle.

## 15. Implementing negotiation on the server, honestly

A short, practical note on doing this correctly in code, because the place negotiation goes wrong is the implementation, not the theory. Here is a minimal, honest negotiation handler in Python-flavored pseudocode that respects everything above:

```python
PRODUCIBLE = {"application/json", "text/csv", "application/vnd.acme.order.v2+json",
              "application/vnd.acme.order.v1+json"}

def get_order(request, order_id):
    order = store.load(order_id)            # the resource
    if order is None:
        return problem(404, "Order not found")

    # Query-param override wins over Accept, per our documented rule.
    override = request.query.get("format")
    if override == "csv":
        chosen = "text/csv"
    elif override == "json":
        chosen = "application/json"
    else:
        # Server-driven negotiation: rank Accept by q, intersect with PRODUCIBLE,
        # break ties by specificity then server preference, default to v1.
        chosen = best_match(request.headers.get("Accept", "*/*"),
                            offer=PRODUCIBLE, default="application/vnd.acme.order.v1+json")

    if chosen is None:                       # nothing the client accepts can be produced
        return problem(406, "No acceptable representation",
                       available=sorted(PRODUCIBLE))

    body, content_type = render(order, chosen)   # the representation
    headers = {"Content-Type": content_type, "Vary": "Accept"}
    if chosen == "text/csv":
        headers["Content-Disposition"] = 'attachment; filename="order-%s.csv"' % order_id
    return Response(200, body, headers)
```

Three things to copy from this and three traps to avoid. Copy: (1) the explicit `PRODUCIBLE` set, so negotiation is data-driven and a new representation is one line; (2) `Vary: Accept` set unconditionally on the negotiated response; (3) a `406` that *lists what is available*. Avoid: (1) defaulting to *latest* on a missing `Accept` — default to a *pinned* version (`v1`) so new schemas are opt-in; (2) returning `200` with an error body when negotiation fails — return the honest `406`; (3) hand-rolling the `q`-value parser if your framework or a well-tested library (`werkzeug`'s `parse_accept_header`, `accepts` middleware, content-negotiation libraries) already does it correctly, because the specificity-then-`q` rule is exactly the kind of thing people implement subtly wrong. Most mature web frameworks ship a correct negotiation primitive; use it, and reserve your custom logic for the policy (which versions, which default), not the parsing.

## 16. When to reach for this (and when not to)

A decisive recommendation section, because every choice here is a trade-off and the failure mode is over-building.

**Reach for content negotiation when:**

- You have a *named* consumer for a second representation — finance wants CSV, customers want PDF receipts, a partner wants XML or a pinned schema. The consumer must be real and present, not hypothetical.
- You are running a *public* API with a heterogeneous, uncontrolled audience. Negotiation lets you serve JSON, a vendor type, and legacy XML from one resource model.
- You want media-type *versioning* and you have weighed it (in the [versioning post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning)) against URI and header versioning and decided the one-URL-many-representations purity is worth the testability cost for your audience.
- You compress responses — then at minimum negotiate `Accept-Encoding` and set `Vary: Accept-Encoding`. This one is nearly always worth it.

**Do not reach for it when:**

- Your only consumers are programs that want JSON. Ship JSON, validate `Content-Type` (return `415` for non-JSON bodies), and stop. A JSON-only API is correct, not lazy.
- You are tempted to build CSV/PDF/XML renderers "for flexibility" with no consumer asking. That is speculative generality; adding a representation later is purely additive and breaks nobody, so *wait*.
- You would `Vary` on a high-cardinality header like `User-Agent` — that shreds your cache. Negotiate only on the purpose-built `Accept*` headers.
- You are reaching for agent-driven `300 Multiple Choices` — for nearly every API it costs a round trip and has no standard menu format; use server-driven `Accept` or a query-param override instead.
- You would return `200` with an error body when negotiation fails. Return `406` (can't produce) or `415` (can't consume) honestly, so gateways, retries, and clients can branch on the status line.
- A "representation" is so different it is arguably a different *resource* with different access rules and lifecycle. Then give it its own URL, do not contort one endpoint into serving both.

The meta-rule that subsumes all of these: **negotiation is a tool for serving one resource to multiple kinds of readers; if you have one kind of reader, you do not need the tool, and building it anyway is a cost with no payoff.**

## Key takeaways

- A **resource** is the thing a URL names; a **representation** is the bytes you send to describe it. One `/orders/4821` can have JSON, CSV, and PDF representations, all of the same order. This split is baked into the meaning of `GET` — "transfer *a* representation" — not bolted on.
- **Proactive (server-driven) negotiation** is the workhorse: the client sends `Accept` with `q`-value preferences, and the server serves the highest-ranked representation it can produce, breaking ties by media-type specificity. **Reactive (agent-driven) `300` negotiation** exists but is almost never the right choice for an API.
- A **media type** has a grammar — `type/subtype+suffix; params`. The `vnd.` tree marks a vendor-owned type, the `+json`/`+xml` **structured suffix** lets generic tools parse the bytes even if they do not know your schema, and **parameters** like `charset` refine without changing identity.
- **`406 Not Acceptable`** means "I can't *produce* what your `Accept` demands"; **`415 Unsupported Media Type`** means "I can't *consume* the `Content-Type` of your body." `406` is about what you can't get; `415` is about what you can't send. Failing honestly with the right code beats a `200`-with-error-body lie.
- A **vendor media type** is a versioning lever: `application/vnd.acme.order.v2+json` lets one URL serve v1 and v2, selected by `Accept`. Default a missing/generic `Accept` to a *pinned* version, never to *latest*, so new schemas are opt-in and no client is silently upgraded.
- **If you negotiated, you must `Vary`.** List every request header you used to select the representation (`Vary: Accept, Accept-Encoding, ...`) or a shared cache will serve the wrong representation to the wrong client. Never `Vary` on a high-cardinality header.
- **Most APIs should be JSON-only, and that is fine.** Add a second representation only when a real, named consumer needs it — exports, file downloads, public APIs. Because a new representation is purely additive, there is no penalty for waiting until the need is real.
- A **query-param/suffix override** (`?format=csv`, `.csv`) is a legitimate pragmatic complement to `Accept` for browser-driven and analyst audiences: trivially debuggable and immune to `Vary` bugs. If you offer it, document that it *wins* over `Accept`.

## Further reading

- [RFC 9110, "HTTP Semantics"](https://www.rfc-editor.org/rfc/rfc9110.html) — the authoritative source for content negotiation, the `Accept`/`Content-Type` headers, `q`-values, `406`/`415`, the `Vary` header, and the resource-versus-representation model. Sections on "Content Negotiation" and "Representations" are the canonical reference for this entire post.
- [RFC 6838, "Media Type Specifications and Registration Procedures"](https://www.rfc-editor.org/rfc/rfc6838.html) — the grammar of media types: top-level types, the standards/vendor (`vnd.`)/personal (`prs.`) trees, and registration rules.
- [RFC 6839, "Additional Media Type Structured Syntax Suffixes"](https://www.rfc-editor.org/rfc/rfc6839.html) — defines `+json`, `+xml`, and the other structured suffixes that let generic tooling parse vendor types.
- [RFC 9457, "Problem Details for HTTP APIs"](https://www.rfc-editor.org/rfc/rfc9457.html) — the `application/problem+json` media type used for every error body here (obsoletes RFC 7807).
- [The JSON:API specification](https://jsonapi.org/format/) — a real-world vendor media type (`application/vnd.api+json`) with strict parameter rules and explicit `406`/`415` semantics.
- [The HAL specification](https://stateless.group/hal_specification.html) — `application/hal+json`, a hypermedia representation distinguished from plain JSON purely by media type.
- Within this series: the [intro hub on the API as a contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); [HTTP for API designers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers) for where `Accept` and `Content-Type` first appear; [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) where media-type versioning is weighed in full; [designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming) for what goes inside the representation; and the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) that ties the whole contract together.

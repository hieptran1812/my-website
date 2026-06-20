---
title: "API Governance and Style Guides: Consistency Across a Large Org"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to make fifty teams' APIs feel like one company's API: a written style guide that codifies every design rule, a Spectral linter that enforces it in CI, a lightweight design-review board, an API catalog that finds shadow and zombie APIs, and a paved road that makes the compliant way the easy way."
tags:
  [
    "api-design",
    "api",
    "rest",
    "governance",
    "style-guide",
    "spectral",
    "openapi",
    "linting",
    "http",
    "developer-experience",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-1.png"
---

A developer on the partner-integrations team opens the docs for your Payments API. The list endpoint pages with a `cursor` query parameter, returns errors as `application/problem+json`, and uses `snake_case` field names like `created_at`. Two hours later the same developer opens the docs for your Orders API — built by a different team, in a different quarter, by people who never talked to the Payments team. Now pagination is `?page=3&pageSize=50`, errors come back as a bare `{ "error": "not found" }` body with a `200 OK` status, and the fields are `camelCase` like `createdAt`. The refunds service, built by yet a third team, paginates with `?offset=100&limit=25`, puts its auth token in a custom `X-Api-Token` header instead of `Authorization`, and returns money as a floating-point dollar amount, `19.99`, while Payments returns it as an integer count of cents, `1999`.

Each of these APIs is fine in isolation. Each team made a defensible local choice. But the *caller* — the one human being who has to integrate all three — pays a tax on every single one. They learn pagination three times. They write three different error parsers. They get burned once when `19.99` loses a cent to floating-point rounding and once more when they assume the Orders auth header works on Refunds. A developer who learned one of your APIs should already know the others; instead they have to learn each from scratch. Multiply that across fifty teams and a few hundred endpoints and the **integration tax compounds** into the single biggest drag on developer experience your platform has — and not one team is to blame, because the problem is not in any one API. The problem is that there is no *shared contract about how contracts are written*.

This is the governance problem, and it is the last problem this series tackles before the capstone, because it is the one that only appears at scale. Everything in Tracks A through G — HTTP semantics, resource modeling, pagination, error design, versioning, auth — was about getting *one* API right. Governance is about getting *fifty* APIs to agree, so that the whole platform feels like it was designed by one careful person on one long afternoon, even though it was built by hundreds of people over many years. By the end of this post you will be able to: write a style guide that codifies your house rules; turn that style guide into machine-enforced lint rules with Spectral that run in CI on every spec; stand up a design-review board that gates new APIs without becoming a bottleneck; build an API catalog that finds the shadow and zombie APIs nobody remembers; and choose a governance model — centralized, federated, or paved-road — that fits the size of your org. The thesis of the whole series holds here too: an API is a contract and a product. Governance is how you keep that contract consistent and that product coherent across an organization too big for everyone to know everyone.

![a side by side comparison showing five inconsistent team APIs with different pagination and error styles on the left and the same five APIs unified by one style guide on the right](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-1.png)

The figure above is the whole post in one image. On the left, three teams each speak a different dialect; on the right, the same three teams speak one language. Everything that follows is the machinery that gets you from left to right — and, crucially, *keeps* you there as new teams and new services arrive.

## 1. The problem at scale: the integration tax is real, and it compounds

Let us make the cost concrete, because "consistency is good" is the kind of platitude that loses every budget argument. The integration tax is not vague goodwill; it is measurable engineering time.

Consider a partner who must integrate three of your APIs. With a consistent platform, they learn the pagination model once, the error model once, the auth model once, and then every new endpoint is "the same thing with different nouns." Their time-to-first-successful-call on a *new* one of your APIs is minutes. With an inconsistent platform, every API is a fresh puzzle: they re-read the pagination docs, re-discover that this one returns errors in a body with a `200` status, re-learn that money is a float here and an integer there. Their time-to-first-call on each new API resets to hours.

Now scale the inconsistency itself. Suppose each of $N$ teams independently picks one of $k$ reasonable conventions for each of $d$ design decisions — pagination style, casing, error shape, auth header, date format, money representation. The probability that any two randomly chosen teams agree on *all* $d$ decisions is roughly $(1/k)^{d}$, which for even modest $k$ and $d$ is vanishingly small. With $k = 3$ and $d = 6$ that is $1/729$. The expected number of *distinct dialects* across your platform grows toward $N$ as the org grows: every team is, in practice, its own dialect. The caller does not experience an average; they experience the *union* of all dialects, because they integrate whichever APIs their use case happens to touch. Inconsistency is not additive — it is combinatorial.

The internal cost mirrors the external one. When the platform team wants to roll out a new cross-cutting capability — say, a standard rate-limit header, or request-correlation IDs for tracing, or a new field in every error body — they have to negotiate it with fifty teams who each parse and emit that field slightly differently. A change that *should* be one shared-library bump becomes fifty migration tickets. Consistency is not just a developer-experience nicety for external callers; it is the precondition for the platform team being able to *change anything at all* without a fifty-front war.

There is a security dimension too, which we will return to with the API catalog. Inconsistency breeds *unknown* surface. When every team rolls its own auth, some team rolls it wrong. When there is no registry of what exists, some endpoint is forgotten, never patched, and becomes the way in. The OWASP API Security Top 10 lists **Improper Inventory Management** (API9:2023) precisely because the APIs you have forgotten about are the ones that get exploited. Governance is, among other things, a security control.

There is one more cost that almost nobody budgets for: the cost of *correctness* bugs that inconsistency hides. When Payments returns money as the integer `1999` and Orders returns it as the float `19.99`, a downstream reconciliation job that sums charges across both APIs will silently get the wrong total — not crash, not error, just *wrong*, by a factor of one hundred on the Orders side. The bug does not announce itself; it shows up weeks later as a finance team asking why the ledger does not balance. Inconsistency in a *machine-readable* contract is more dangerous than inconsistency in a human-facing one, because the human at least gets confused and asks a question. The machine just computes the wrong answer with full confidence. A platform-wide rule that money is *always* an integer of minor units is not pedantry; it is the difference between a reconciliation that balances and one that does not.

Finally, consider the cost to the *teams themselves*, not just to callers. Every team that invents its own pagination has to *design*, *build*, *test*, and *maintain* that pagination — and get the hard parts right, like stable ordering under concurrent writes and bounded page sizes that protect the database. Fifty teams solving pagination is fifty implementations, of which maybe three are actually correct under load and the rest skip or duplicate rows when the table shifts. (This is exactly the offset-drift failure the [pagination post](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale) walks through.) The same is true of error serialization, of idempotency, of rate limiting. The integration tax is matched by a *re-implementation tax*: the org pays, fifty times over, to build the same wheels, most of them slightly out of round. Consistency is not just nicer for callers — it is *cheaper to build*, because the right implementation is written once and shared, a point we return to with the paved road.

### The principle: a platform is a product, and consistency is its UX

Here is the rule, stated rigorously enough to act on. **The unit of developer experience is not the endpoint; it is the platform.** A caller's cost to use your platform is not the cost of the *best* API on it, nor the *average* — it is the sum of the learning costs of every distinct convention they encounter. Formally, if a caller touches a set of APIs $S$ and $\text{conv}(S)$ is the number of distinct conventions across $S$ for the decisions that caller cares about, their integration cost is approximately proportional to $\text{conv}(S)$, not to $|S|$. The goal of governance is to drive $\text{conv}(S)$ toward $1$ regardless of how large $|S|$ grows.

That single sentence — drive the number of distinct conventions toward one — is what every tool in this post exists to serve. The style guide *defines* the one convention. The linter *enforces* it. The review board *catches* what the linter cannot. The catalog *finds* the APIs that escaped. The paved road *makes the one convention the easiest one to follow*. Keep that frame and the rest of the post organizes itself.

It is worth being precise about what consistency is *not*. Consistency is not uniformity for its own sake, and it is not a demand that every API be identical — a Payments API and a search API legitimately differ in their resource models, their performance profiles, and their access patterns. What governance standardizes is the *cross-cutting* dimensions: the dimensions that are not about the domain at all. Pagination is not a payments concept or an orders concept; it is a *collection* concept, and there is no good reason for two collections on the same platform to paginate differently. Error shape is not domain-specific; an error is an error. Auth is not domain-specific. Casing, date format, money representation, the correlation header — none of these encode anything about the business. They are pure *form*, and form is exactly what should be uniform so that *substance* — the actual nouns and verbs of each domain — is what a caller has to learn. Governance frees teams to differ where it matters (the domain) by forcing them to agree where it does not (the form). A team that resists governance as "stifling our design freedom" has usually confused the two: nobody is dictating their resource model; they are being asked to spell `created_at` the same way as everyone else.

## 2. The style guide: codifying Tracks A through G into rules

The foundation of governance is a written standard: a **style guide** that says, for every recurring API-design decision, "here is how we do it at this company." It is the single document that makes "consistent with the platform" a checkable claim instead of a matter of taste. You are not inventing new design wisdom here — you spent Tracks A through G learning the *why* behind each rule. The style guide is where you write down the *what*: the specific choice your org commits to, so that fifty teams make the same choice without each having to re-derive it.

![a vertical stack of style guide layers showing naming and casing, URI conventions, error format, pagination, versioning, auth, and data formats as ordered rules](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-2.png)

The stack above shows the layers a complete style guide covers. Let us walk them, with concrete example rules anchored in the Payments and Orders APIs, because a style guide written in generalities is a style guide nobody can apply.

### Naming and casing

Pick one casing and use it everywhere. The choice between `snake_case` and `camelCase` matters far less than the consistency of it — but you must *make* the choice, in writing, once.

> **Rule N-1.** All JSON property names use `snake_case`. Acronyms are lower-cased: `http_status`, not `HTTPStatus` or `httpStatus`.
>
> **Rule N-2.** Resource collections are plural nouns: `/payments`, `/orders`, `/refunds`. Never `/payment` for a collection, never a verb like `/createPayment`.
>
> **Rule N-3.** Timestamps end in `_at` and are RFC 3339 strings: `created_at`, `updated_at`, `captured_at`. Booleans read as predicates: `is_refundable`, not `refundable` or `refund_flag`.

The payoff is that a caller who has seen `created_at` on a Payment can *predict* the field name on an Order without reading the docs. Predictability is the entire point. Zalando's public RESTful API Guidelines, one of the most widely cited style guides in the industry, mandate `snake_case` for exactly this reason; Google's API Improvement Proposals (AIPs) standardize on `camelCase` in JSON. Both are correct; what matters is that within Zalando, or within Google, there is *one* answer.

A word on *how* to write these rules, because the format determines whether they get followed. Use RFC 2119 keywords — **MUST**, **SHOULD**, **MAY** — so that the strength of each rule is unambiguous: a MUST is enforced by the linter and blocks a merge; a SHOULD is a strong default that a reviewer may waive with a reason; a MAY is permission, not obligation. Number every rule so it can be cited: when a reviewer comments "violates N-1" on a pull request, the author can look up exactly what N-1 says, and the conversation is about the rule, not about the reviewer's taste. And give every rule a one-line *rationale* — not a paragraph, just the reason — so that a team who wants to deviate understands what they would be giving up. "Use `snake_case`" is a rule a team resents; "Use `snake_case` so a caller can predict field names across every API without reading the docs" is a rule a team understands. A rule with a reason is a rule that survives the first engineer who disagrees with it.

### URI conventions

> **Rule U-1.** Hierarchy is expressed through path nesting only where there is a true containment relationship: `/orders/{order_id}/line_items`. Cross-cutting filters go in the query string, not the path.
>
> **Rule U-2.** Resource identifiers in URIs are opaque strings. Never expose a raw auto-increment database ID; use a prefixed, typed identifier like `pay_3Kd92hf` for payments and `ord_8Fj01ka` for orders, so an ID is self-describing and un-guessable.

Rule U-2 carries a lot of weight. Prefixed IDs — popularized by Stripe — mean that when a caller pastes an ID into the wrong endpoint, the server can reject it immediately (`pay_…` sent to `/orders/{id}` is obviously wrong) and a human reading a log can tell at a glance what kind of object they are looking at. That is consistency paying a security and debuggability dividend.

### Error format

This is where inconsistency hurts most, because errors are what callers hit when they are already in trouble.

> **Rule E-1.** All error responses use `application/problem+json` as defined in RFC 9457, with at minimum `type`, `title`, `status`, and `detail`. The `type` is a stable URI; the `title` is a stable human-readable summary; the `detail` is request-specific.
>
> **Rule E-2.** The HTTP status code MUST match the error class. Never return `200 OK` with an error body. Validation failures are `422`; missing resources are `404`; auth failures are `401`; permission failures are `403`; conflicts are `409`.
>
> **Rule E-3.** Every error includes a `request_id` field that correlates to the value in the `X-Request-Id` response header, so a caller can quote one identifier when they open a support ticket.

A compliant error from any service on the platform looks like this on the wire:

```http
HTTP/1.1 422 Unprocessable Content
Content-Type: application/problem+json
X-Request-Id: req_7Hc02la9

{
  "type": "https://errors.example.com/payments/amount-too-small",
  "title": "Payment amount below the minimum",
  "status": 422,
  "detail": "The amount 50 is below the minimum chargeable amount of 100 (in cents).",
  "request_id": "req_7Hc02la9"
}
```

Because the *shape* is mandated, a caller can write one error-handling function that works against Payments, Orders, and Refunds alike. This is the [error-design contract](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) elevated from a per-API decision to an org-wide invariant.

### Pagination, versioning, status codes, auth, and formats

The remaining layers follow the same pattern — state the one choice, anchored in an example:

> **Rule P-1 (pagination).** All collection endpoints paginate with an opaque `cursor` parameter and a bounded `limit` (default 25, max 100). Offset pagination is prohibited for collections that can change under the reader, because rows shift and pages skip or repeat. The next cursor is returned in a `Link` header with `rel="next"`.
>
> **Rule V-1 (versioning).** New, additive changes ship without a version bump under the [tolerant-reader rule](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change). Breaking changes require a new version, selected via a `Api-Version` request header carrying a date, e.g. `Api-Version: 2026-06-20`. URI versioning is reserved for whole-platform major epochs only.
>
> **Rule S-1 (status codes).** `201 Created` for successful resource creation with a `Location` header; `202 Accepted` for accepted-but-async work; `204 No Content` for successful deletes. `429 Too Many Requests` always carries `Retry-After`.
>
> **Rule A-1 (auth).** All authenticated requests carry an OAuth 2.0 bearer token in the standard `Authorization: Bearer <token>` header. Custom auth headers are prohibited. Scopes are named `resource:action`, e.g. `payments:read`, `orders:write`.
>
> **Rule F-1 (formats).** Money is always an integer count of the currency's minor unit (cents for USD) plus an ISO 4217 `currency` field — never a floating-point major-unit amount. Dates and timestamps are RFC 3339 / ISO 8601 strings in UTC.

Rule F-1 is the one that prevents the floating-point cent loss from the intro. When a payment of \$19.99 is represented as the integer `1999` with `"currency": "USD"`, there is no rounding, no locale ambiguity, and no possibility that one team's `19.99` and another's `19.990` disagree. The rule is boring. Boring is the goal.

#### Worked example: a before-and-after of two teams unified by the style guide

Let us walk a concrete before-and-after, because the value of the style guide is only visible in the diff. Here is the Orders "list" endpoint *before* the style guide, as the Orders team originally shipped it:

```http
GET /orders?page=2&pageSize=50 HTTP/1.1
Host: orders.example.com
X-Api-Token: <token>

HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [
    { "orderId": 88213, "createdAt": "2026-06-18 14:02:11", "total": 19.99 }
  ],
  "page": 2,
  "pageSize": 50
}
```

And here is a *not found* on the same API, before:

```http
GET /orders/99999999 HTTP/1.1
Host: orders.example.com
X-Api-Token: <token>

HTTP/1.1 200 OK
Content-Type: application/json

{ "error": "order does not exist" }
```

Count the violations against the Payments-derived style guide: `pageSize`/`page` offset pagination (violates P-1), a custom `X-Api-Token` header (violates A-1), `camelCase` (violates N-1), a raw integer `orderId` (violates U-2), a space-separated non-RFC-3339 timestamp (violates F-1), a float `total` with no currency (violates F-1), and — worst of all — a `200 OK` on a missing resource with a bare error string (violates E-1 and E-2). Seven violations on two requests. A caller who learned the Payments API would be wrong about *every one* of these on Orders.

Now the *after*, once the Orders team adopts the style guide:

```http
GET /orders?cursor=b3JkXzhGajAxa2E&limit=50 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Api-Version: 2026-06-20

HTTP/1.1 200 OK
Content-Type: application/json
Link: <https://api.example.com/orders?cursor=b3JkXzlHazAy&limit=50>; rel="next"
X-Request-Id: req_a91kf02

{
  "data": [
    {
      "order_id": "ord_8Fj01ka",
      "created_at": "2026-06-18T14:02:11Z",
      "amount": 1999,
      "currency": "USD"
    }
  ]
}
```

And the not-found, after:

```http
GET /orders/ord_does_not_exist HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>

HTTP/1.1 404 Not Found
Content-Type: application/problem+json
X-Request-Id: req_a91kf03

{
  "type": "https://errors.example.com/common/not-found",
  "title": "Resource not found",
  "status": 404,
  "detail": "No order with id ord_does_not_exist.",
  "request_id": "req_a91kf03"
}
```

A caller who learned Payments now reads Orders *for free*. Same pagination, same auth, same casing, same ID shape, same money representation, same error envelope, same correlation header. The integration tax on the second API dropped to nearly zero. That is the entire return on the style guide, made visible in a diff.

The catch — and the reason the next section exists — is that a written style guide is a document, and documents do not enforce themselves. A wiki page titled "API Style Guide" that fifty busy teams are *asked* to follow is, empirically, a wiki page that fifty busy teams *do not read*. The rule has to become code.

## 3. Automated enforcement: a linter that reads every spec in CI

The single highest-leverage move in API governance is to turn the style guide from prose a human must remember into **rules as code** a machine checks automatically. The standard tool for this in the OpenAPI world is **Spectral** — an open-source JSON/YAML linter built for API specs. You write a *ruleset* (a YAML file of rules), point Spectral at every team's OpenAPI document, and it reports every violation. Wire it into CI and a non-compliant spec cannot merge. This is **shifting governance left**: the rule is checked at the moment of authorship, in the pull request, not discovered months later by an annoyed caller in production.

The principle behind shift-left is the same one that makes type-checkers and linters valuable everywhere: **the cost of fixing a defect grows with how late you catch it.** A naming violation caught by Spectral in a pull request costs the author thirty seconds to rename a field. The same violation caught after the API ships and a hundred callers have integrated it is a *breaking change* to fix — it now costs a deprecation cycle, a migration window, and a Sunset header. Catching it at lint time is cheaper by orders of magnitude, and — critically — it is caught *before any caller can depend on the mistake*. A linter that runs on every push is a wiki nobody can ignore.

![a branching pipeline showing an OpenAPI spec flowing into Spectral lint then a review gate then the catalog and ship, with a reject path branching off lint and review](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-3.png)

The figure shows where lint sits: the spec is the source of truth, Spectral lints it, compliant specs proceed to the review gate and then the catalog, and non-compliant specs branch immediately to a reject-and-fix path. The non-compliant spec never reaches a caller. Note that this is the [spec-first workflow](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate) doing double duty: the OpenAPI document you already write to mock and generate from is the *same* artifact governance lints. You get enforcement for free because the spec already exists.

### A Spectral ruleset that encodes the style guide

Here is a real Spectral ruleset that encodes several of the style-guide rules above. Spectral rules use a JSONPath-like `given` to select parts of the spec and a `then` with a built-in or custom function to assert something about them.

```yaml
# .spectral.yaml — the org style guide, as code
extends: ["spectral:oas"]   # start from the built-in OpenAPI ruleset

rules:
  # Rule N-2: collection paths must be plural nouns, lowercase
  paths-are-plural-lowercase-nouns:
    description: "Resource collections must be lowercase plural nouns (Rule N-2)."
    message: "Path segment '{{value}}' must be a lowercase plural noun, not a verb."
    severity: error
    given: "$.paths[*]~"        # the ~ selects the path KEY, not its value
    then:
      function: pattern
      functionOptions:
        # reject verbs and camelCase; require lowercase words and {params}
        match: "^(/[a-z_]+(/\\{[a-zA-Z_]+\\})?)+$"

  # Rule N-1: all schema property names must be snake_case
  properties-are-snake-case:
    description: "JSON property names must be snake_case (Rule N-1)."
    severity: error
    given: "$..properties.*~"
    then:
      function: casing
      functionOptions:
        type: snake

  # Rule E-1: every operation must declare a problem+json error response
  errors-use-problem-json:
    description: "Error responses must use application/problem+json (Rule E-1)."
    severity: error
    given: "$.paths[*][*].responses[?(@property >= '400')].content"
    then:
      field: "application/problem+json"
      function: truthy

  # Rule A-1: a custom auth header is forbidden; use Authorization
  no-custom-auth-header:
    description: "Use Authorization: Bearer, never a custom auth header (Rule A-1)."
    severity: error
    given: "$..parameters[?(@.in == 'header')].name"
    then:
      function: pattern
      functionOptions:
        notMatch: "(?i)^x-api-(key|token)$"
```

Running this against a compliant spec is silent. Running it against the *before* Orders spec from the worked example flags the `/orders` operation for offset pagination's absence of a cursor, the `pageSize` `camelCase` parameter, the `X-Api-Token` header, the `orderId`/`createdAt` `camelCase` properties, and the missing `problem+json` response. Five errors, each with a `message` that names the rule and points at the line. The author fixes them before the PR merges.

### A custom Spectral function for a rule the built-ins cannot express

The built-in functions (`pattern`, `casing`, `truthy`, `defined`, `enumeration`) cover most rules, but some org rules need custom logic. Spectral lets you write a JavaScript function and reference it. Here is a custom function that enforces **Rule F-1's money convention**: any schema property whose name suggests money (`amount`, `total`, `price`, `fee`) must be an integer, and the schema must also declare a sibling `currency` field.

```javascript
// functions/money-is-minor-units.js
export default function (targetVal, _opts, context) {
  // targetVal is a schema object (a "properties" map)
  const moneyNames = /^(amount|total|price|fee|subtotal)$/i;
  const results = [];
  for (const [name, schema] of Object.entries(targetVal || {})) {
    if (!moneyNames.test(name)) continue;

    if (schema.type !== "integer") {
      results.push({
        message: `Money field '${name}' must be an integer of minor units, not ${schema.type} (Rule F-1).`,
        path: [...context.path, name, "type"],
      });
    }
    if (!("currency" in targetVal)) {
      results.push({
        message: `Schema with money field '${name}' must also declare a 'currency' field (Rule F-1).`,
        path: [...context.path, name],
      });
    }
  }
  return results;
}
```

And the rule that wires it in:

```yaml
rules:
  money-is-minor-units:
    description: "Money must be integer minor units plus a currency (Rule F-1)."
    severity: error
    given: "$..properties"
    then:
      function: money-is-minor-units

functions: ["money-is-minor-units"]
functionsDir: "./functions"
```

#### Worked example: a custom Spectral rule failing a non-compliant spec in CI

Now let us watch this fail in CI, end to end, because "it works on my machine" is not governance. The Orders team opens a pull request that adds a `POST /orders` operation. Their OpenAPI spec contains this schema:

```yaml
components:
  schemas:
    Order:
      type: object
      properties:
        order_id:
          type: string
        total:
          type: number          # <-- a float, violating Rule F-1
          example: 19.99
        created_at:
          type: string
          format: date-time
```

Note the two problems: `total` is a `number` (float), and there is no `currency` property. The CI pipeline runs Spectral as a required check:

```bash
$ npx spectral lint openapi/orders.yaml --ruleset .spectral.yaml

openapi/orders.yaml
 24:13  error  money-is-minor-units  Money field 'total' must be an integer
                                     of minor units, not number (Rule F-1).
 24:13  error  money-is-minor-units  Schema with money field 'total' must
                                     also declare a 'currency' field (Rule F-1).

✖ 2 problems (2 errors, 0 warnings)
```

Spectral exits with a non-zero status code, the CI check turns red, and the merge button is disabled. The author cannot ship the float. They change `total` to `amount` with `type: integer` and add a `currency` field, re-run, and the check passes:

```yaml
components:
  schemas:
    Order:
      type: object
      properties:
        order_id:
          type: string
        amount:
          type: integer          # minor units
          example: 1999
        currency:
          type: string
          example: "USD"
        created_at:
          type: string
          format: date-time
```

```bash
$ npx spectral lint openapi/orders.yaml --ruleset .spectral.yaml
No results with a severity of 'error' found!
```

The float never reached production. The caller never saw `19.99`. No deprecation cycle was needed, because the mistake was caught before any contract was published. That is the entire economic argument for shift-left governance, demonstrated in one CI run: a defect that would have cost a multi-month migration cost thirty seconds in a pull request.

The crucial detail is *severity*. Spectral rules can be `error` (fails the build), `warn` (visible but non-blocking), `info`, or `hint`. A mature governance practice introduces new rules as `warn` first — so existing specs that violate them surface without breaking anyone's build — then promotes them to `error` once the backlog is cleared. This is how you add a rule to a hundred existing specs without declaring war on a hundred teams in one afternoon.

There is a second design decision in the ruleset itself: how to organize it so it stays maintainable as it grows to dozens or hundreds of rules. The practical pattern is to keep a single base ruleset — the org-wide style guide as code — and let teams `extends` it, exactly the way the example above extends `spectral:oas`. A team with a legitimate local need can layer additional, stricter rules on top, but cannot *weaken* the base rules, because the base ruleset is the contract. Distribute the base ruleset as a versioned package (an npm module, say) so that when the guild updates a rule, teams pick up the change by bumping a dependency — and so that you can see, from the dependency graph, which teams are running an old ruleset and missing the latest rules. The ruleset is itself an API the platform team ships to its own engineers, and it deserves the same versioning discipline as any other contract.

It also matters *where* Spectral runs. The strongest setup runs it in three places: in the editor (so an author sees a violation as they type, the cheapest possible feedback), in a pre-commit or pre-push hook (so the obvious mistakes never reach CI), and as a *required* status check in CI (so nothing merges without passing, regardless of whether the author ran the local checks). The CI check is the one that actually enforces — editor and hook checks are conveniences an author can skip — but the convenience checks are what make the enforcement feel helpful rather than punitive. By the time CI runs, a developer who used the local tools has already fixed everything, so CI is green and invisible. A developer who skipped them gets caught, but caught *kindly*, with the same machine-generated message pointing at the same line. The goal is never to surprise an author in CI with something they could have seen three steps earlier.

## 4. Manual review vs automated lint: what each is for

A natural question at this point: if Spectral catches violations automatically, why have humans review APIs at all? The answer is that lint and review catch *different* classes of problem, and a healthy governance practice uses both — but assigns each to what it is good at.

![a matrix comparing manual review against automated lint across cost per spec, coverage, and speed](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-7.png)

The matrix above makes the trade-off explicit. Automated lint wins decisively on **cost** (near-zero per spec once written), **coverage** (it checks *every* rule on *every* path of *every* spec, with no fatigue), and **speed** (seconds per push, no queue). What it cannot do is exercise *judgment*. Spectral can verify that you used `problem+json`; it cannot tell you that your error *taxonomy* is confusing, that you modeled a refund as a sub-resource of a payment when it should be a top-level resource, or that the API you are proposing duplicates one another team already shipped. Those are design questions, and design needs a designer.

| Concern | Automated lint (Spectral) | Manual review (humans) |
| --- | --- | --- |
| Cost per spec | Near zero after the ruleset is written | High — pulls senior engineers' time |
| Coverage | Total — every rule, every path, every spec | Partial — spot checks, fatigue sets in |
| Speed | Seconds, in CI on every push | Days, gated by reviewer availability |
| Catches | Mechanical rule violations (casing, shape, status, headers) | Design smells, modeling errors, duplication, naming taste |
| Consistency of verdict | Perfect — same input, same output | Variable — depends on the reviewer |
| Scales to | Thousands of specs trivially | Bounded by reviewer headcount |

The division of labor is therefore clear: **make the machine check everything mechanical, so that humans only review what requires judgment.** Every rule you can express in Spectral is a rule a human reviewer should *never* have to mention, because mentioning it is a waste of a senior engineer's scarce attention and a nitpick that demoralizes the author. A good design review never says "you used camelCase" — the linter already said that. It says "have you considered that this idempotency design double-charges on a concurrent retry?" That is the highest and best use of a reviewer, and the linter is what frees them to do it.

## 5. The API review board: a gate that does not become a bottleneck

For *new* APIs and for *breaking changes* to existing ones, you want a human design review before code is written — a review of the *spec*, not the implementation, because changing a spec is cheap and changing a shipped API is not. The mechanism is an **API review board** (sometimes called a design-review or API-council): a small group of experienced engineers who review proposed API designs against the style guide and against good taste.

The single greatest risk of a review board is that it becomes a **bottleneck** — a queue that every team must wait in, run by a few people who become a single point of failure for the whole org's velocity. A bottlenecked review board is worse than no board, because it teaches teams to *route around governance*: they ship the API without review, governance loses legitimacy, and you are back to fifty dialects plus resentment. So the entire design of the board must be oriented around *not* being a bottleneck.

![a timeline of a spec moving through the review gate from template to self-lint to office hours to checklist audit to approval or waiver](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-5.png)

The timeline above shows a review designed for throughput. Here are the techniques that keep it fast:

- **Templates and scaffolds.** Give teams an OpenAPI template that is already compliant — correct error responses, correct pagination parameters, correct auth scheme — so the *starting point* passes lint. Most of "review" is then just confirming the team filled in the nouns. The fastest review is the one that has nothing to fix.
- **Self-service lint before submission.** The team runs Spectral locally and in their own CI *before* they ever ask for review. By the time a human looks, every mechanical issue is already gone. The reviewer arrives at a clean spec and spends their time on design.
- **A written checklist.** The board reviews against a published checklist — the same one the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) provides — so review is consistent across reviewers and predictable for authors. A team can self-assess against the checklist before submitting and pre-empt most feedback.
- **Office hours, not a ticket queue.** Hold a regular open session where teams bring designs for synchronous discussion. A fifteen-minute conversation resolves what a multi-day async review thread cannot. Office hours also scale the board's *knowledge* — junior engineers attend, learn the reasoning, and become reviewers themselves.
- **Approve-by-default with exceptions.** This is the most important rule. The board's default answer is *yes*. A team may ship unless the board raises a concern within a fixed window (say, three business days). Silence is approval. This inverts the bottleneck: the burden is on the *board* to object in time, not on the *team* to wait for a green light. A team is never blocked by an overloaded reviewer; they are only blocked by an actual, articulated objection.
- **A waiver path.** When a team genuinely needs to deviate from the style guide — a legacy constraint, a third-party format they must match, a performance edge case — they file a written waiver that records *what* rule they are breaking and *why*. The board grants or denies it, and the waiver lives in the catalog next to the API. Deviations become *visible and intentional* instead of silent and forgotten. This is how you keep the style guide from being either tyrannical (no exceptions ever) or meaningless (everyone ignores it).

What should the board actually *check*? Only what the linter cannot: the resource model (are these the right nouns? is the hierarchy real containment or a forced nesting?), the [request and response shapes](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming) at the design level (is this envelope justified? is this field going to be a regret in a year?), idempotency and concurrency semantics, the [versioning and compatibility](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) plan, the auth and scope model, and — the one thing only a human with org-wide context can catch — *duplication*: "the Inventory team already exposes this; should you call them instead of building a parallel API?" That last check alone often pays for the entire board.

#### Worked example: a breaking-change proposal through the board

The Payments team wants to rename the `amount` field to `amount_minor` to make the minor-units convention explicit in the name. They open a design proposal. The mechanical check passes — `amount_minor` is valid `snake_case`, it is an integer, `currency` is present — so Spectral is green. But this is a *rename*, and the board's job is the judgment Spectral lacks.

In office hours, a reviewer asks the one question that matters: "How many callers read `amount` today?" The catalog says: forty-one internal services and two external partners. The reviewer points out that renaming a response field is a *breaking change* — every one of those callers that reads `amount` will get `null` and may 500. The decision the board reaches is not "no"; it is "yes, but as an *additive expand-then-contract*": ship `amount_minor` *alongside* `amount`, mark `amount` deprecated in the spec with a `deprecated: true` flag and a `Deprecation` header, give callers an eighteen-month migration window tracked in the catalog, and only then remove `amount`. The rename happens — without breaking a single caller. The board did not block the change; it shaped a breaking change into a safe one. That is the value a human reviewer adds on top of the linter, and it is exactly the kind of judgment the matrix in the previous section reserved for people.

## 6. The API catalog: you cannot govern what you cannot find

Here is a question that humbles every large org: *how many APIs do you have?* Almost no one can answer it. There is the official list, and then there is the long tail of APIs that some team stood up for an internal tool three years ago, documented nowhere, owned by people who have since left, and never decommissioned. You cannot apply a style guide to an API you do not know exists. You cannot patch a vulnerability in an API you forgot you were running. **Governance starts with an inventory.**

An **API catalog** (also called a service registry, API portal, or — when built on the popular open-source developer portal — a Backstage software catalog) is a single registry of every API your org runs: its name, its owning team, its versions, its lifecycle status (in-design, live, deprecated, retired), its OpenAPI spec, and a link to its docs. It is the answer to "what do we have, who owns it, and is it healthy?"

![a branching diagram showing gateway logs and the registered catalog reconciling into live versus known and surfacing shadow APIs zombie APIs and healthy APIs](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-6.png)

The figure shows the catalog's most valuable function beyond mere listing: **reconciliation**. You take the set of routes actually serving traffic (from your gateway logs) and compare it to the set of APIs *registered* in the catalog. The diff is where the danger lives:

- **Shadow APIs** — routes serving live traffic with *no* registered spec and no owner. Nobody designed them to the style guide, nobody reviewed them, and nobody is watching them. They are pure unmanaged risk.
- **Zombie APIs** — registered APIs that were supposed to be retired (an old `v1` long past its Sunset date) but are *still answering requests*. Every zombie is an un-patched, un-monitored attack surface that the org believes is dead.

This is precisely **OWASP API9:2023, Improper Inventory Management.** The category exists because the APIs you have lost track of are statistically the ones that get breached — the old version with the un-patched auth bug, the internal endpoint accidentally exposed to the internet, the deprecated service nobody is monitoring. The catalog turns "we think we retired that" into "the catalog says it served 4,000 requests yesterday — it is a zombie, kill it." A reconciliation report that surfaces shadow and zombie routes is one of the highest-value security artifacts a platform team can produce, and it falls out of governance for free.

The catalog also makes the review board and linter *complete*: the linter can only check specs it can find, and the board can only review APIs that get submitted. The catalog is what closes the loop — it tells you which APIs have a spec (and so can be linted), which have an owner (and so can be held accountable), and which are flying dark. A practical enforcement: the gateway refuses to route to any path not registered in the catalog with a current spec. Now "register your API" is not a polite request; it is the only way to get traffic. The path of least resistance becomes the compliant one — which is the theme of the next section.

What should each catalog entry actually record? At minimum: a unique API name, the owning team and an on-call contact, the lifecycle status, every live version, a pointer to the OpenAPI spec, a link to the generated docs, and the API's classification (internal-only, partner, or public) — because the classification determines how strict the review and the security bar should be. A richer catalog also records the API's dependencies (which other APIs it calls) and dependents (who calls it), which is what makes a deprecation safe: before you Sunset a version you query the catalog for everyone who depends on it and notify them by name, rather than turning it off and waiting for the pages. The catalog is, in effect, the org's *dependency graph of contracts* — and a dependency graph you can query is the difference between a humane deprecation and a fire.

A note on how to populate it, because a catalog that requires manual data entry is a catalog that goes stale the day after launch. The durable approach is to make the catalog *derive itself* from artifacts teams already maintain. The owning team comes from the code repository's ownership metadata. The spec comes from the repository (committed next to the code, as spec-first demands). The live versions come from the gateway. The lifecycle status comes from the spec's own `deprecated` flags and the team's deployment metadata. When the catalog is assembled from sources of truth teams already keep current for *other* reasons, it stays accurate without anyone doing data-entry chores — and accuracy is the only thing that makes a catalog trustworthy enough to act on.

## 7. Stress-testing the governance system

A design is only as good as how it holds up under pressure, and a governance *system* is itself a design — a contract between the platform and the teams. So let us stress-test it the way the rest of this series stress-tests an API: pose the hard cases and reason through what happens.

**What happens when a team genuinely needs to break a rule?** Say a team integrates with a legacy partner that demands `XML` and `camelCase` and a custom auth header, none of which the style guide allows. The wrong answer is to either force them to comply (they cannot — the partner dictates the format) or to let them silently deviate (now the catalog is lying about the platform's consistency). The right answer is the *waiver*: the team files a written exception that records the rule, the reason, the scope (this one external-facing adapter), and an expiry date. The deviation becomes visible in the catalog, the board signs off, and — crucially — the deviation is *contained* to the adapter. The team still exposes a *compliant* internal API and translates at the edge. Governance bends without breaking, and the exception is documented rather than discovered.

**What happens when the style guide itself is wrong?** Rules are not handed down from a mountain; they are written by fallible people and they age. Suppose the org standardized on offset pagination years ago, before the scaling problems were understood, and now a dozen teams have offset endpoints that skip rows under load. The governance system must have a path to *evolve the standard*, not just enforce the current one. That path is the same waiver and metrics machinery running in reverse: a rising count of waivers against a rule, or a cluster of incidents traceable to a rule, is the signal that the rule needs to change. The guild revises the style guide, the new rule lands in Spectral as a `warn`, teams migrate over a deprecation window, and the rule is promoted to `error`. A governance system with no path to change its own rules calcifies and gets routed around; the feedback loop *is* the system.

**What happens when there are a hundred existing non-compliant specs and you introduce a new rule?** This is the migration problem, and it is where most governance programs die. If you turn the new rule on at `error` severity, a hundred builds break at once, a hundred teams are furious, and governance is now the enemy. The disciplined approach has three phases. First, ship the rule at `warn` so it is visible everywhere but blocks nobody — every team can *see* how far off they are. Second, hold the line on *new* specs: the rule is `error` for files created after a cutoff date, so the backlog stops growing even before it shrinks. Third, burn down the backlog with the catalog as a worklist, team by team, until the backlog is empty, then flip the rule to `error` org-wide. The principle is the same shift-left economics: you never break a build for a mistake that was made before the rule existed.

**What happens when two teams want to own the same noun?** The Inventory team and the Catalog team both want `/products`. Spectral cannot catch this — both specs are individually valid. This is the duplication problem that *only* a human reviewer with org-wide context can catch, and it is the single most expensive mistake governance prevents, because two parallel `/products` APIs mean two sources of truth that will drift, two integrations for every caller, and an eventual painful merge. The review board's job here is not to enforce a rule but to *broker a decision*: one team owns the canonical resource, the other consumes it. The catalog records the ownership so the next team that reaches for `/products` finds it already exists.

**What happens when the review board is on vacation?** This is why *approve-by-default* is not a nicety but a correctness property of the system. If the system required an explicit human green light, then a board that is overloaded, on holiday, or simply slow becomes a hard dependency on every team's release. Under approve-by-default, the absence of an objection within the window *is* the approval — so the throughput of the org does not depend on the availability of a handful of reviewers. The board is a safety net that *catches* problems, not a turnstile that *gates* progress. The distinction is the whole difference between governance that scales and governance that throttles.

This stress test surfaces the design principle behind the entire system: **governance must fail open, not closed.** When the linter is down, builds should warn, not block. When the board is unavailable, the default is yes. When a rule is wrong, there is a path to change it. When a team must deviate, there is a documented exception. A governance system that fails *closed* — that blocks by default whenever any part of it is unavailable or uncertain — teaches the org to route around it, which destroys the consistency it was built to protect. A system that fails *open* keeps the org moving while still catching the cases it can, and earns the trust that makes teams *want* to participate.

## 8. Governance models: centralized, federated, and the paved road

How you *organize* governance depends on how big you are. There are three broad models, and they are not mutually exclusive — most mature orgs run a blend that shifts as they grow.

![a matrix comparing centralized, federated, and paved-road governance models across consistency, bottleneck risk, and the org size they scale to](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-4.png)

| Model | How it works | Pros | Cons | Scales to |
| --- | --- | --- | --- | --- |
| **Centralized standards team** | One dedicated team owns the style guide, reviews every API, and holds the keys | Maximum consistency; one clear owner; one voice | Becomes a bottleneck; the queue forms; teams route around it; the standards team loses touch with day-to-day reality | Tens of teams |
| **Federated guild** | A cross-team guild of API champions co-owns the standard; review is peer-to-peer; each team has a designated reviewer | No central bottleneck; standards stay grounded in practice; spreads expertise | Consistency drifts without strong shared tooling; needs active coordination; can fragment if the guild goes quiet | Dozens of teams |
| **Platform-team paved road** | A platform team *provides* compliance as infrastructure — templates, scaffolds, a shared gateway, shared libraries, the linter, the catalog — so the easy way is the compliant way | Consistency by construction; near-zero bottleneck; self-service; compliance is a *byproduct* of using the platform | High upfront platform investment; the paved road must genuinely be the easiest path or teams build their own | Hundreds of teams |

The arc most orgs travel is: start centralized (a few people care a lot and review everything), move to federated as the review load exceeds what a central team can handle (deputize a guild), and converge on the paved road as the org gets large enough that the only scalable enforcement is *making compliance the default*. The models compose — a paved-road org still has a small central team that *owns the road* and a guild that *evolves the standard*, but the bulk of "governance" happens automatically because the tools make the right way the easy way.

### The paved road / golden path: make the compliant way the easy way

The deepest idea in modern API governance is this: **the most effective enforcement is not a rule, a review, or a linter — it is making the compliant path the path of least resistance.** This is the **paved road** (Netflix's term) or **golden path** (Spotify's term): a curated, supported, well-documented default way to build an API at your company, such that a team who just "follows the happy path" ends up compliant *without trying*.

Concretely, the paved road is a kit of infrastructure:

- **A service template / scaffold.** `create-api my-service` generates a new service whose OpenAPI spec already has the correct error responses, pagination parameters, auth scheme, and money types — already passing the linter. The team starts compliant.
- **A shared API gateway.** Auth, rate limiting, the standard `X-Request-Id` correlation header, `429` + `Retry-After`, and request logging are handled *at the gateway*, so no team implements them (or implements them wrong). The [gateway](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern) is where cross-cutting consistency is *enforced by infrastructure* rather than requested by documentation.
- **Shared client and server libraries.** A shared error-serialization library that emits `problem+json`. A shared pagination helper that does cursors correctly. A shared money type that *cannot* be a float. When the right behavior lives in a library every team already imports, the wrong behavior requires *extra effort* — and engineers do not do extra work to be non-compliant.
- **Generated SDKs and docs.** As the [SDK and docs post](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate) covers, generating SDKs and reference docs from the linted spec means the *published* developer experience is consistent for free, because it is generated from the consistent spec.

The genius of the paved road is that it changes the incentive. Under a pure rule-and-review regime, compliance is *friction* — extra work the team does to satisfy governance, resented accordingly. Under a paved road, compliance is *convenience* — the team gets auth, rate limiting, pagination, errors, SDKs, and docs *for free* by staying on the road, and would have to do *more* work to leave it. You have aligned the team's self-interest with the org's consistency goal. That is the only kind of governance that survives contact with a large, busy organization.

## 9. Metrics, culture, and maturity: making governance stick

A governance program you cannot measure is a governance program you cannot defend at budget time and cannot improve. Track a small set of metrics that map directly to the goal of driving distinct conventions toward one.

- **Spec coverage.** The fraction of live APIs (from the catalog's reconciliation) that have a registered OpenAPI spec. This is your visibility metric: $\text{coverage} = \frac{\text{APIs with a spec}}{\text{all live APIs}}$. You cannot lint, review, or generate docs for the rest. Drive it toward 100%.
- **Lint pass rate.** The fraction of registered specs that pass the Spectral ruleset at `error` severity. This is your consistency metric. Track it over time as you promote rules from `warn` to `error`; a dip after a promotion is expected and should recover as teams fix the backlog.
- **Time-to-first-call.** The median time for a new developer to make their first successful authenticated call against a *new-to-them* API on the platform. This is your developer-experience metric, and it is the one that proves the integration tax went down. On a consistent platform with a paved road, it should be minutes and should *not* grow as you add APIs.
- **Waiver count and age.** How many active style-guide waivers exist, and how old they are. A growing pile of stale waivers means the style guide has drifted from reality and needs revision — waivers are a feedback signal, not just an escape hatch.
- **Review cycle time.** Median days from design submission to approval. If this climbs, the board is becoming a bottleneck; deploy more office hours, more templates, or more approve-by-default.

### Culture: carrots over sticks

The hardest part of governance is not technical; it is cultural. Governance imposed as a *stick* — a compliance team that says no, a review that blocks, a linter experienced as nagging — breeds exactly the resentment that drives teams to route around it. Governance offered as a *carrot* — a paved road that gives teams auth, docs, SDKs, and rate limiting for free; a linter that catches their bugs before a caller does; a review that makes their API better — gets *adopted voluntarily*, which is the only enforcement that scales.

The operating principle: **make compliance the path of least resistance, and make the benefits of compliance visible to the team that complies, not just to the org.** A team adopts the shared error library because it saves them from writing one, and consistency is the byproduct. A team submits to review because the reviewer caught a concurrency bug that would have paged them at 3 a.m., and consistency is the byproduct. Celebrate the well-designed API publicly. Make the platform team's success metric *adoption*, not *enforcement actions*. You are not the API police; you are the API platform, and your customers are your own engineers.

One concrete cultural practice ties this together: treat the style guide and the paved road as products with a *changelog* and a *feedback channel*, the same way you would treat any API you ship to external developers. When the guild adds or changes a rule, announce it the way you would announce an API change — with the rationale, the migration path, and the timeline. When a team hits friction on the paved road, that friction is a bug report against the platform, not a failure of the team, and it goes into the platform's backlog. The teams who consume your governance are exactly the kind of caller this whole series has been about: people you are designing a contract for, on a timeline of years, whose experience determines whether your contract is adopted or routed around. Govern your governance the way you would have it govern everything else, and the system becomes self-consistent — which is, after all, the entire point.

### Maturity stages

Governance is a journey, not a switch. A rough maturity model, so you know where you are and what is next:

| Stage | What it looks like | The next move |
| --- | --- | --- |
| **0 — Ad hoc** | No style guide; every team invents everything; no inventory | Write the style guide; start the catalog |
| **1 — Documented** | A written style guide exists on a wiki; review is informal | Encode the guide in Spectral; lint locally |
| **2 — Enforced** | Spectral runs in CI; a review board exists with a checklist | Build the paved road; templates and shared libraries |
| **3 — Paved** | Templates, gateway, and shared libraries make compliance the default; the catalog reconciles live traffic | Measure time-to-first-call; treat the platform as a product |
| **4 — Self-improving** | Metrics drive ruleset evolution; waivers feed back into the guide; governance is invisible because it is built in | Keep the road paved as the org and the standards both evolve |

![a tree showing how much governance to apply by org size branching into few teams, many teams, and org scale with a concrete approach under each](/imgs/blogs/api-governance-and-style-guides-consistency-across-an-org-8.png)

The tree above answers the practical question every team actually asks: *how much governance do we need?* The answer scales with the number of teams. A startup with three teams does not need a Spectral ruleset and a review board; it needs a one-page shared doc and the habit of reviewing each other's PRs. An org with many teams needs the linter in CI and a guild that owns the rules. An org at hundreds-of-teams scale needs the full paved road — a platform team, a catalog, scaffolds. Apply the level of governance that matches your size; the next section makes that recommendation decisive.

## 10. Case studies: how real orgs do this

These are accurate, public references — the standards and tools named here are real and worth reading in full.

- **Zalando's RESTful API Guidelines.** One of the most comprehensive and widely cited public API style guides. It mandates `snake_case` JSON, prescribes `problem+json` for errors, specifies pagination and versioning conventions, and is organized as numbered, individually citable rules ("MUST", "SHOULD", "MAY" in RFC 2119 style). It is the canonical example of "a style guide written so it can be enforced." Zalando also publishes a companion Spectral-style ruleset (Zally) to check specs against it.
- **Google's API Improvement Proposals (AIPs).** Google's design guidance is published as a numbered, searchable set of AIPs (aip.dev) covering resource naming, standard methods, pagination, long-running operations, errors, and versioning. AIPs are the design rules behind Google's vast API surface, and the format — small, numbered, individually addressable proposals — is itself a governance lesson: a style guide that is a single 200-page document is one nobody reads; a set of small numbered rules is one teams can cite in review.
- **Microsoft REST API Guidelines.** Microsoft's public guidelines codify their conventions for naming, errors, pagination, versioning, and long-running operations across Azure and other surfaces. Like Zalando's, they exist precisely because a company that large cannot achieve consistency through goodwill — only through a written, enforced standard.
- **Adidas API Guidelines.** Adidas publishes its own public REST API guidelines, a useful example of a non-tech-first enterprise treating API consistency as a first-class engineering concern — evidence that this is not just a FAANG luxury.
- **Spectral (Stoplight).** The open-source linter this post is built around. It ships a built-in OpenAPI ruleset, supports custom rules and custom JavaScript functions, integrates into CI, and is the de facto standard for "style guide as code." Orgs encode their Zalando-or-AIP-flavored rules in a `.spectral.yaml` and run it on every spec.
- **Backstage and the software catalog.** Backstage, the open-source developer portal originally from Spotify, popularized the **software catalog** — a registry of all services and APIs with owners, lifecycle, and docs — and the **golden path** idea: templates that scaffold a new, compliant service. It is the most common open-source foundation for the API-catalog and paved-road practices described here.

The pattern across all of them is identical: a *written* standard, organized as small citable rules, *enforced* by tooling, with a *catalog* of what exists and a *paved road* that makes the standard the default. Different companies, same machine.

## 11. When to reach for governance — and how much

Governance is a cost, and like every design decision in this series it is a trade-off. The failure modes are symmetric: too little governance and you drown in the integration tax; too much and you crush the velocity of small teams under process they do not need. Match the dose to the org.

**Reach for heavy governance when:** you have many teams (more than can fit in one room), external callers or partners who integrate multiple APIs, a regulatory or security obligation to know your full API surface, or a platform team whose job is to enable other teams. At that scale, the linter, the catalog, and the paved road pay for themselves many times over.

**Keep governance light when:** you are a small org or a single team. A startup with one product team does *not* need a Spectral ruleset, a review board, and a catalog — that is process theater that will slow them down for no benefit, because with one team there is only ever one dialect. They need a one-page shared doc and the habit of reviewing each other's PRs. Adding a review board to a three-team org *creates* the bottleneck it was meant to prevent.

**Specific "do nots":**

- **Do not start with a review board before you have a style guide.** Without a written standard, review is just one senior engineer's taste, applied inconsistently, experienced as arbitrary. Write the rules first.
- **Do not enforce a rule in Spectral at `error` severity on day one** across a hundred existing specs — you will break a hundred builds and lose the org's trust. Introduce rules as `warn`, clear the backlog, then promote.
- **Do not let the review board check things the linter can check.** A reviewer who comments "use snake_case" on a PR is wasting their judgment and nagging the author. Automate the mechanical; reserve humans for design.
- **Do not run governance as a stick.** A compliance team measured by enforcement actions, with no paved road to make compliance easy, will be routed around, and you will have the worst of both worlds: process *and* inconsistency.
- **Do not skip the catalog because "we know what we have."** You do not. Reconcile against gateway traffic and you will find shadow and zombie APIs you forgot existed — which is exactly why OWASP lists improper inventory management as a top-10 risk.
- **Do not write a 200-page monolithic style guide.** Nobody reads it, so nobody follows it. Small, numbered, individually citable rules — Zalando-and-AIP style — are the ones that actually get applied in review.

For service-fleet concerns that sit just below the governance layer — how the gateway routes to and discovers the services your catalog tracks — see [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing); governance decides *what* should exist and *how it should look*, and the fleet infrastructure decides *how requests reach it*.

## Key takeaways

- **The unit of developer experience is the platform, not the endpoint.** A caller pays a tax for every distinct convention they meet. Governance exists to drive the number of distinct conventions toward one, no matter how many APIs you have.
- **The style guide is the foundation.** Codify Tracks A through G — naming, URIs, errors, pagination, versioning, status codes, auth, money and date formats — into small, numbered, citable rules anchored in concrete examples.
- **Rules as code beat a wiki nobody reads.** Encode the style guide in a Spectral ruleset and run it in CI on every spec. Shift governance left: a violation caught in a pull request costs seconds; the same violation caught in production costs a deprecation cycle.
- **Lint and review do different jobs.** Make the machine check everything mechanical so humans only review what needs judgment — resource modeling, semantics, duplication, taste. A reviewer who nitpicks casing is a reviewer the linter has failed.
- **A review board must not become a bottleneck.** Templates, self-service lint, a published checklist, office hours, approve-by-default-with-exceptions, and a written waiver path keep the gate fast and keep teams from routing around it.
- **You cannot govern what you cannot find.** An API catalog that reconciles live gateway traffic against registered specs surfaces shadow and zombie APIs — directly addressing OWASP API9, improper inventory management.
- **Pick the governance model that fits your size:** centralized for tens of teams, federated for dozens, paved-road for hundreds. The models compose, and most orgs migrate along that arc as they grow.
- **The paved road is the strongest enforcement.** Templates, a shared gateway, and shared libraries that make the compliant way the easy way turn compliance from friction into convenience. Carrots over sticks: align the team's self-interest with the org's consistency.
- **Measure it.** Spec coverage, lint pass rate, and time-to-first-call tell you whether the integration tax is actually falling. Match the dose of governance to the org — light for a startup, heavy for a platform.

## Further reading

- [What is an API? The contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) — the series intro and the contract-and-product frame that governance enforces across an org.
- [The API design playbook: a review checklist from first endpoint to v2](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) — the capstone checklist your review board reviews against.
- [OpenAPI and the spec-first workflow](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate) — the spec is the artifact governance lints; spec-first makes enforcement free.
- [Designing request and response bodies: shape and naming](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming) — the body conventions your style guide standardizes.
- [Error design: a machine-readable, human-friendly contract](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) — the problem+json error envelope a style guide mandates platform-wide.
- [Versioning strategies: URI, header, media-type, and not versioning](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) — the versioning policy a style guide commits the whole org to.
- **Zalando RESTful API Guidelines** (opensource.zalando.com/restful-api-guidelines) — a comprehensive public style guide of numbered MUST/SHOULD/MAY rules, plus the Zally linter.
- **Google API Improvement Proposals** (aip.dev) — Google's design rules as small, numbered, individually citable proposals.
- **Microsoft REST API Guidelines** (github.com/microsoft/api-guidelines) — Microsoft's public conventions for naming, errors, pagination, and versioning.
- **Spectral** (github.com/stoplightio/spectral) — the open-source OpenAPI/JSON linter for encoding a style guide as enforceable rules with custom functions.
- **Backstage** (backstage.io) — the open-source developer portal behind the software-catalog and golden-path patterns.
- **OWASP API Security Top 10** — API9:2023 Improper Inventory Management, the risk an API catalog exists to mitigate.

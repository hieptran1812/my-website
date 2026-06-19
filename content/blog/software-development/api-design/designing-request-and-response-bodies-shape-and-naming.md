---
title: "Designing Request and Response Bodies: Shape, Naming, and Consistency"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn how to shape JSON payloads that survive years of clients: one casing convention, fields named so the next one is guessable, money as minor units instead of floats, IDs as strings, RFC 3339 timestamps, the null-versus-omitted-versus-empty trap, and when an envelope earns its keep."
tags:
  [
    "api-design",
    "api",
    "rest",
    "json",
    "payload-design",
    "naming-conventions",
    "consistency",
    "http",
    "rfc-3339",
    "envelopes",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-1.png"
---

A few years ago I watched a single character break a mobile app for two days. Someone on the backend team renamed a field from `total` to `totalAmount` in the order response, shipped it on a Friday, and went home. The web client used a tolerant JSON deserializer and shrugged. The iOS app used a strict, code-generated model and crashed every checkout screen on launch. The fix was trivial; the damage was not. Thousands of users could not pay, and the postmortem opened with a question that the team had never written down: *what, exactly, did we promise the response body would look like?*

That question is the whole subject of this post. The body — the JSON you send back, and the JSON you accept — is the part of your API that callers actually touch on every single request. They will parse it, store it, log it, and build screens out of it. They will write code that assumes `amount` is a number and `status` is a known string. And here is the uncomfortable truth: **the moment your body ships, its shape becomes a contract you cannot quietly change.** Add a field, fine. Rename one, remove one, change a type, or send `null` where a client expected an array — and somewhere a parser falls over, a screen goes blank, or a sum comes out a cent short.

![a before and after comparison showing a messy order body with mixed casing, float money, and integer status next to a clean body with consistent snake case, minor units, and a lowercase enum](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-1.png)

This post is about the craft of shaping those bodies so they last. We will work the whole way through with the series' running example — a fictional commerce platform with `/orders`, `/payments`, and `/refunds` — and by the end you will be able to: pick and enforce one casing convention; name booleans, timestamps, enums, money, and IDs so a client can *guess* the next field before reading the docs; decide when a bare object beats a `data` envelope and when the full `data`/`meta`/`links` wrapper earns its weight; encode absence (`null` vs omitted vs empty) without lying to the caller; and design a request body that is strict enough to protect you and tolerant enough to evolve. Most of these are not arbitrary style choices. As we go, I will derive *why* each one holds — why floats break money, why an integer ID will eventually betray you, why "always return an array, never null" is a real rule and not just a preference.

If you have only ever returned `jsonify(order.__dict__)` from a Flask route, you can follow all of this. We will define every term the first time it shows up. But we are going to go deep, because the body is where consistency either compounds into a pleasant, learnable API or rots into a field-by-field guessing game. This post sits in the heart of the series spine: an API is a **contract** and a **product**, not a function call. The body is the most-touched surface of that contract, so it is where consistency pays the largest dividend.

## 1. Consistency is the master rule

Before any specific naming convention, there is one rule that dominates all of them: **be consistent.** A consistent convention you slightly dislike beats an inconsistent mix of conventions you love. The reason is mechanical, not aesthetic.

Here is the principle, stated rigorously. An API surface is a set of field names and types spread across many resources. A caller learns your API by example: they read the `/orders` response, internalize the pattern, and then *predict* the shape of `/payments` and `/refunds` before they ever read those docs. Every prediction that holds is documentation the caller did not have to read. Every prediction that fails is a surprise — a bug, a support ticket, a slow afternoon spent diffing two responses to find out that one resource calls it `created_at` and another calls it `createdDate`. So consistency has a measurable payoff: it lowers the *information* a caller must absorb to use your API. If field naming is perfectly regular, learning one resource teaches you the rest. If it is irregular, every resource is a fresh memorization task.

Let me put a rough number on this, because design reviews go better with one. Suppose a client integrating with your API encounters `n` fields across all your resources. If naming is perfectly regular — every timestamp is `*_at`, every reference is `*_id`, every money field is minor-units-plus-currency — then once the client has learned the *patterns* (a constant, small number of rules), each new field costs them almost nothing to understand: they predict its name and type and are right. The learning curve flattens fast. If naming is irregular, every field is an independent fact to memorize, so the cost grows linearly with `n`, and worse, the client cannot trust *any* prediction, so they must read the docs for every single field defensively. The difference between "learn five rules, then predict everything" and "memorize a hundred independent facts" is the difference between an API a developer can hold in their head and one they must keep a reference tab open for. Consistency is not politeness; it is a reduction in the information a caller must carry.

Concretely, three things must be consistent:

1. **Casing.** Pick `snake_case` *or* `camelCase` and never mix them within a response, across responses, or between request and response. `snake_case` (lowercase words joined by underscores, like `order_id`) is the more common choice for JSON APIs and reads cleanly in URLs and logs. `camelCase` (first word lowercase, subsequent words capitalized, like `orderId`) matches JavaScript and JSON conventions in the browser. Both are fine. **A mix is never fine.** A body with `order_id` next to `totalAmount` next to `Status` tells every reader that nobody is minding the contract.

2. **Field names across resources.** If the timestamp a record was created is `created_at` on `/orders`, it is `created_at` on `/payments`, `/refunds`, and everything else. The currency code is always `currency`. The customer reference is always `customer_id`. A field that means the same thing must be *spelled* the same everywhere. The opposite — `created_at` here, `created` there, `creation_time` in a third place — forces the caller to maintain a translation table in their head.

3. **The same entity, always serialized the same way.** A `customer` object embedded in an order should have the same fields, in the same shape, as the `customer` you get from `GET /customers/{id}` (modulo deliberate trimming, which we will get to). If the embedded version drops `email` and renames `id` to `customer_id`, your caller now has two incompatible mental models of "customer" and code that cannot share a parser between them.

The figure below is a quick reference for the per-field naming rules we are about to derive. Keep it nearby; the rest of this section and the next two are essentially the justification for each row.

![a matrix of field naming rules with rows for boolean, timestamp, enum, money, and identifier, each showing a bad encoding beside the recommended good encoding](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-2.png)

### Why I lean toward snake_case (but it does not matter much)

You will read holy wars about this. They are not worth your time. The honest position: `snake_case` and `camelCase` are both readable, both widely tooled, both fine. I lean `snake_case` for HTTP/JSON APIs for three small reasons: it matches the casing most people already use in URL path and query segments (`?page_size=50`), it survives being lowercased in logs and SQL without ambiguity, and it visually distinguishes "this is a wire field" from "this is a JavaScript variable." Google's API Improvement Proposals (AIP) standardize on `snake_case` for field names in their `.proto` definitions, then let the JSON mapping present them as the proto's lowerCamelCase if a client prefers — one source of truth, two presentations. If your primary consumer is a browser SPA and your whole stack is JavaScript, `camelCase` will feel more native. Pick one. Write it in your style guide. Lint it in CI. Then stop thinking about it forever.

#### Worked example: a consistent versus inconsistent naming pass over two resources

Let me make the cost of inconsistency concrete by shaping the same two resources twice. First, the inconsistent version — the kind that accretes when three engineers add fields over a year without a style guide:

```json
{
  "OrderID": "ord_81000",
  "createdDate": "2026-06-18T09:30:00Z",
  "total_amt": 4999,
  "Status": 2,
  "customer": { "id": "cus_77", "Name": "A. Rivera", "email_addr": "a@example.com" },
  "isPaid": true
}
```

```json
{
  "payment_id": "pay_5500",
  "created": 1718703000,
  "amount": "49.99",
  "state": "SUCCEEDED",
  "order": "ord_81000",
  "refunded": false
}
```

Count the ways a client gets hurt. The order's identifier is `OrderID`; the payment's is `payment_id`. Created time is `createdDate` (an RFC 3339 string) on one, `created` (a Unix epoch integer) on the other. The order's money field is `total_amt` (an integer of cents); the payment's `amount` is a quoted decimal string `"49.99"`. Status is a magic integer `2` versus an uppercase string `"SUCCEEDED"`. The boolean is `isPaid` versus `refunded`. The customer is a nested object in one and a bare ID string in the other, and the customer's email is `email_addr`. A caller writing code against both resources cannot reuse a single field name, a single date parser, a single money handler, or a single status enum. Every resource is a new dialect.

Now the consistent version:

```json
{
  "id": "ord_81000",
  "object": "order",
  "created_at": "2026-06-18T09:30:00Z",
  "amount": 4999,
  "currency": "usd",
  "status": "paid",
  "customer_id": "cus_77",
  "is_paid": true
}
```

```json
{
  "id": "pay_5500",
  "object": "payment",
  "created_at": "2026-06-18T09:30:05Z",
  "amount": 4999,
  "currency": "usd",
  "status": "succeeded",
  "order_id": "ord_81000",
  "is_refunded": false
}
```

Now `created_at` is always an RFC 3339 string, `amount` is always an integer in minor units paired with `currency`, `status` is always a lowercase string, IDs are always prefixed strings, booleans always start with `is_`, and references are always `{thing}_id`. A client who learned `/orders` already knows how to parse `/payments`. That is the entire payoff of consistency: **learning one resource teaches you the next.** The two bodies are not identical — they should not be — but they are *predictable from each other*, which is the property that matters.

## 2. Naming fields well

Consistency tells you to pick a rule and hold it. This section is about which rules to pick. Most of these are conventions with real reasons behind them, and the reasons are worth understanding so you can defend them in a design review.

### Booleans: prefix with `is_` or `has_`

A boolean field reads best as a yes/no question the field name asks. `is_paid`, `is_refunded`, `has_shipped`, `is_default`. The prefix does two jobs. First, it signals the type: a reader sees `is_` and knows the value is `true` or `false`, not a status string or a count. Second, it forces you to phrase the field as a clear assertion, which surfaces ambiguity early. A field called `active` could be a boolean, a status, or a timestamp; `is_active` can only be a boolean. Avoid negative names — `is_not_deleted` is a parser for your reader's brain. Prefer the positive form (`is_active`) and let the client negate it if they must.

### Timestamps: RFC 3339, always, with an offset

This is the one I will fight you on, because timestamps cause more cross-team pain than almost any other field. Use **RFC 3339** — the internet profile of ISO 8601 — for every date and time, and store/transmit in UTC with an explicit offset:

```json
{
  "created_at": "2026-06-18T09:30:00Z",
  "updated_at": "2026-06-18T09:31:42.512Z",
  "captured_at": "2026-06-18T11:30:00+02:00"
}
```

The `Z` (or the explicit `+02:00`) is the load-bearing part. A timestamp without an offset — `2026-06-18T09:30:00` — is *ambiguous*: nine-thirty in whose time zone? The reader has to guess, and guesses about time produce the worst class of bug, the kind that only manifests for users in certain regions during certain months. By contrast, `Z` means UTC, unambiguously, forever. RFC 3339 is also lexicographically sortable: sort the strings and you sort the instants, which is a small daily convenience that adds up. Name timestamp fields with an `_at` suffix (`created_at`, `updated_at`, `expires_at`) so the type is visible in the name.

Do not send Unix epoch integers as your primary timestamp representation. Epochs (`1718703000`) are compact and unambiguous about the instant, but they are unreadable to a human debugging a log, easy to confuse between seconds and milliseconds (off by a factor of 1000, a real and common bug), and they throw away the offset information that an RFC 3339 string preserves. If a client genuinely needs an epoch, they can convert one; the wire format should favor the human-debuggable, self-describing string.

### Enums: lowercase strings, not magic integers

When a field has a small fixed set of values — an order's `status`, a payment's `status` — encode it as a **lowercase string**, not an integer. `"paid"`, not `2`. The reason is readability and evolvability. A response that says `"status": 2` is opaque: the reader must find a lookup table to learn that `2` means `paid`, `1` means `pending`, `3` means `cancelled`. A response that says `"status": "paid"` is self-documenting. More importantly, integer enums are fragile under change. If you ever reorder, insert, or deprecate a value, the integers shift and every cached or hard-coded mapping on the client side is now wrong, silently. String enums are stable: `"paid"` means `paid` regardless of how many statuses you add later.

There is a forward-compatibility subtlety worth stating: **clients should treat an unknown enum value gracefully.** If you add a new `status` of `"disputed"` next year, an old client that only knows four statuses should not crash on the fifth — it should fall back to a sensible default (show the raw string, treat it as "other"). That is the tolerant-reader principle applied to enums, and it is why adding an enum value is a *non-breaking* change only if clients were written tolerantly. Document the closed set, but design as if the set will grow.

### Money: minor units plus a currency, never a float

This is the rule I see violated most often and regret most painfully, so let me actually derive *why* floats break money rather than just asserting it.

A IEEE 754 double-precision float (the JSON `number` type in practice, what JavaScript and most parsers use) stores values in binary, as a sum of powers of two. Many decimal fractions have no exact finite binary representation, exactly as $1/3$ has no exact finite decimal representation. The decimal $0.1$ is one such value: in binary it is the repeating fraction $0.0001100110011\ldots$, which a 64-bit double must truncate. So $0.1$ is not stored as $0.1$; it is stored as the nearest representable double, which is very slightly off. The classic demonstration, true in essentially every language that uses IEEE 754:

```python
>>> 0.1 + 0.2
0.30000000000000004
>>> 0.1 + 0.2 == 0.3
False
```

Now imagine — and look at the figure below as you do — an order with three line items at \$0.10, \$0.20, and \$0.03. Summed as floats, the total can land a hair off from \$0.33, and when you round for display you may get \$0.32 or \$0.34 depending on the order of operations. Across millions of transactions, those fractions of a cent accumulate into real reconciliation breaks: your ledger disagrees with the payment processor by amounts that are individually invisible and collectively a finance team's nightmare. Money is not a measurement, where a tiny error is acceptable; money is a *count*, and counts must be exact.

![a before and after comparison contrasting money stored as a float that drifts on addition with money stored as integer minor units plus a currency code that adds exactly](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-4.png)

The fix is to represent money as an **integer of the currency's minor unit, paired with an explicit currency code**. \$49.99 becomes `{ "amount": 4999, "currency": "usd" }` — 4999 *cents*. Integers add, subtract, and compare exactly; there is no rounding because there is no fraction. The currency code (ISO 4217, lowercase by Stripe convention, uppercase in the standard — pick one and be consistent) is mandatory because `4999` is meaningless without it: 4999 of what? Cents? Yen (which has no minor unit, so `4999` would be 4,999 whole yen)? Dinar (three decimal places, so `4999` is 4.999 dinars)? The minor-unit count and the currency are a *pair*; neither stands alone.

#### Worked example: a money field done wrong with floats versus minor units

Here is the wrong way, the way that passes a demo and fails in production:

```json
{
  "id": "ord_81000",
  "currency": "usd",
  "line_items": [
    { "description": "API plan",   "price": 0.10 },
    { "description": "Add-on",      "price": 0.20 },
    { "description": "Support fee", "price": 0.03 }
  ],
  "total": 0.33
}
```

A client that sums `line_items[].price` to validate `total` runs `0.10 + 0.20 + 0.03`, gets `0.32999999999999996`, compares it to `0.33`, finds them unequal, and either logs a spurious integrity error or — worse — "corrects" the total. Multiply across an export of a million orders and you have a finance ticket that takes a week to root-cause.

Now the right way:

```json
{
  "id": "ord_81000",
  "currency": "usd",
  "amount": 33,
  "line_items": [
    { "description": "API plan",   "amount": 10 },
    { "description": "Add-on",      "amount": 20 },
    { "description": "Support fee", "amount": 3 }
  ]
}
```

The client sums `10 + 20 + 3`, gets exactly `33`, and it equals `amount`. No drift, ever, because there is no floating point in the arithmetic. Stripe, the canonical reference here, has represented amounts as integer minor units (`amount` in cents) since its first public API, precisely to make this class of bug impossible on the wire. If you must support currencies with non-standard minor units, document the multiplier per currency (USD: 100, JPY: 1, BHD: 1000) or carry an explicit `currency_minor_unit` field; do not paper over it with a float.

It is worth tracing the failure all the way to the finance team's desk, because that is where the abstract "float drift" becomes a concrete cost and the reason payment companies treat this as non-negotiable. Suppose your API returns order totals as floats, and a downstream service consumes those totals to build a daily settlement file that it sends to the payment processor. The processor computes its own totals as exact integers of cents (because *it* knows better). Each individual order is off by at most a fraction of a cent, far below the display precision, so nothing looks wrong in any UI. But the settlement file aggregates hundreds of thousands of orders, and the rounding errors do not all cancel — they have a slight bias depending on the arithmetic order, so the daily total drifts from the processor's by a few cents to a few dollars. Now the two ledgers disagree. A reconciliation job flags the mismatch. An engineer is paged. They spend a day proving that no money was actually lost — it is purely an artifact of float arithmetic — and the only real fix is to stop using floats for money, which means a breaking change to the API body that started it. The bug is cheap to prevent (use integers) and expensive to diagnose and unwind after the fact, which is the worst possible cost profile. That is why "money is minor units, never a float" is the one body rule I will not compromise on in a review.

Two honorable mentions for money. If you have a hard requirement to represent amounts larger than $2^{53}$ minor units (the largest integer a JSON `number` can hold exactly — about 90 trillion dollars in cents, so rarely a real constraint) or you want belt-and-suspenders safety against a sloppy parser silently treating your integer as a float, you can transmit the amount as a **decimal string** (`"amount": "4999"` or `"49.99"`) and parse it with an arbitrary-precision decimal type. This is what some financial APIs do. The cost is that the client must remember to use a decimal library and never `parseFloat` it. Integer minor units are simpler and sufficient for the overwhelming majority of commerce APIs; I reach for the string only when amounts can genuinely exceed safe-integer range.

### Identifiers: strings, even when they are numeric

Make every ID a **string** in the JSON body, even if your database stores it as a 64-bit integer. `"id": "81000"`, not `"id": 81000`. There are three reasons, and they compound.

First, **JSON numbers are doubles in practice.** A 64-bit database integer can exceed $2^{53}$, the largest integer a double represents exactly. A JavaScript client parsing `"id": 9007199254740993` gets back `9007199254740992` — the ID is silently corrupted by one, and now a client cannot fetch the resource it was just told about. Transmitting the ID as a string sidesteps the entire floating-point representation question; a string is a string.

Second, **IDs should be opaque.** A client should never do arithmetic on an ID, infer the next one, or assume it is sequential. Encoding it as a string discourages all of that. It also lets you change the *format* of new IDs without changing the field's type: today your IDs are numeric, tomorrow they are UUIDs or Stripe-style prefixed tokens (`ord_81000`, `pay_5500`), and the client code that treats `id` as an opaque string keeps working through the transition. The prefix is a quiet gift to anyone reading a log: `cus_77` is unmistakably a customer, `pay_5500` a payment, so a stray ID in an error message is self-identifying. (Use obvious placeholders in docs and examples — `cus_example`, not a real token — and never paste a live key or secret into a sample body.)

Third, **strings compose with everything.** They drop into URLs without parsing, into headers, into other JSON, into log lines, without anyone having to remember a numeric type. The cost of string IDs is essentially zero — a few bytes — and the failure mode they prevent (silent integer corruption in a JavaScript client) is one of the nastiest bugs to diagnose because the data *looks* fine until you compare it byte-for-byte.

There is a tempting counter-argument worth addressing: "but my IDs are auto-increment integers, and a string just wastes bytes and makes my database joins awkward." The database storage and the wire representation are *different concerns*. Store the ID however your database is happiest — a `BIGINT` primary key, a UUID column — and serialize it as a string at the API boundary. The serialization layer is exactly where the wire contract and the storage model are allowed to diverge, and the cost of the string conversion is a single `str()` call on the way out. What you are buying with that one call is immunity from the worst category of ID bug and the freedom to change your ID generation strategy later without a breaking change to every client. That is a trade worth making every time.

### A short note on field naming pitfalls

A few smaller naming traps that show up in reviews, each with a one-line fix:

- **Do not encode units in a value when you can encode them in the field name or a sibling.** `"timeout": 30` is ambiguous — thirty what? Name it `timeout_seconds`, or `timeout_ms`. The unit belongs in the contract, not in a client's assumption.
- **Do not abbreviate inconsistently.** `qty` here and `quantity` there, `desc` and `description`, `addr` and `address`. Pick the full word (clearer) or the abbreviation (shorter) and apply it uniformly. Mixed abbreviation is just inconsistency wearing a disguise.
- **Do not leak internal jargon into public field names.** If your team calls something a "widget" internally but customers call it a "line item," the field is `line_item`, not `widget`. The body speaks the consumer's language, not your codebase's.
- **Reserve a small set of meta-field names and use them everywhere.** `id`, `object` (the resource type), `created_at`, `updated_at`. When these are universal, a client can write generic code that works across every resource — log every object's `id` and `object`, sort any collection by `created_at` — and that generic code is a large DX win.

## 3. Arrays, nullability, and the cost of ambiguity

Two of the subtlest body-design decisions are about *absence* and *collections*. They look trivial and they are not, because they encode meaning that clients will read whether or not you intended it.

### Always return an array, never `null`, for collection fields

If a field holds a list — `line_items`, `refunds`, `tags` — and there are no items, return an **empty array** `[]`, never `null` and never an omitted field. The principle is the tolerant-reader/robustness idea applied to client code. A client that receives an array can write one code path: iterate it. Zero items means the loop body runs zero times — completely safe. But if the field is sometimes an array and sometimes `null`, the client must guard *every* access with a null check, and the first time they forget (and they will), they get a null-pointer or `TypeError` exception in production. You have pushed a footgun across the wire.

The same logic says: do not omit the array key entirely when it is empty, because now the client cannot tell "this resource has no refunds" from "this resource type does not have refunds" from "the server forgot to include them." An always-present `[]` is unambiguous: the field exists, and it is empty. The rule generalizes: **a field whose type is a collection should always be that collection type, possibly empty, never null and never absent.**

There is a quantitative way to see why this matters across a whole API. Suppose your response has `k` collection-typed fields, and each one independently might be `null` instead of `[]` because some code path forgot to default it. A client that wants to be safe must write `k` separate null guards, one per field, on every code path that touches the response. Miss any one of them and you have a latent crash. If instead the server *guarantees* every collection field is always an array, the client writes *zero* null guards — the guarantee moves the burden from `k` checks on every client to one rule on the server. That asymmetry — fix it once on the server versus guard it everywhere on every client forever — is the entire argument for server-side discipline on body shape, and it recurs throughout this post. The server is one place; the clients are many places, written by people you will never meet, who will not read your docs carefully. Every ambiguity you resolve on the server is an ambiguity those clients never have to handle.

### `null` versus omitted versus empty: three different statements

For scalar (non-collection) fields, there are three ways to express "nothing here," and they mean *different things*. Conflating them is a real source of bugs. The figure lays out the semantics I recommend.

![a matrix showing the three forms of absence with null meaning a known empty value, an omitted key meaning unknown or unchanged on a patch, and an empty array meaning present with zero items](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-5.png)

- **`null`** means *the value is known and there is no value.* `"shipped_at": null` says "this order has definitively not shipped yet; the field applies but is empty." It is a positive statement of absence.
- **An omitted key** means *the value is unknown, not applicable, or unchanged.* In a **response**, omitting a field can mean "we are not telling you this right now" (perhaps it is in a different representation or behind an `?expand=`). In a **request**, especially a `PATCH` with merge semantics (RFC 7396, JSON Merge Patch), an *omitted* key means "leave this field as it is," while an *explicit `null`* means "clear this field." That distinction is load-bearing: `PATCH {"discount": null}` clears the discount; `PATCH {}` leaves it untouched. If your API does not distinguish them, you cannot express "clear this field" at all.
- **An empty value** — `[]`, `""`, `{}` — means *the field is present and its value is genuinely empty.* An empty array has zero items; an empty string is a string of length zero (which is *not* the same as "no string," i.e. `null`).

The cost of ambiguity here is concrete. If your API uses `null` and "omitted" interchangeably, a client cannot reliably implement a PATCH that clears a field, cannot tell "unknown" from "empty," and ends up writing defensive code that guesses your intent. Decide your semantics, write them in the docs (one paragraph is enough), and hold to them. My default: in responses, prefer `null` over omission for fields that exist in the model but have no value (so the shape is stable and the client always sees the key), and reserve omission for fields that are genuinely conditional (an expandable relation that was not requested). In PATCH requests, treat omitted as "unchanged" and explicit `null` as "clear."

#### Worked example: a PATCH that must distinguish clear from leave-alone

A customer wants to remove the discount from a draft order but keep the coupon code. With JSON Merge Patch semantics:

```http
PATCH /orders/ord_81000 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/merge-patch+json

{
  "discount": null
}
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_81000",
  "object": "order",
  "discount": null,
  "coupon_code": "SPRING25",
  "amount": 4999,
  "currency": "usd",
  "status": "draft"
}
```

The `discount` was cleared because the client sent it explicitly as `null`; `coupon_code` was untouched because the client *omitted* it. Had the client sent `{}`, nothing would have changed. This only works because the API treats `null` and "omitted" as distinct — which is exactly why you must not let them blur together in your responses either, or clients will lose the ability to reason about your PATCH semantics by analogy.

## 4. The envelope debate: bare, `data`, or full

Now the question that launches a thousand design-review arguments: do you return the resource *bare*, or wrapped in an envelope? There are three positions, and — unusually for this post — there is no single right answer. The right answer depends on what the body needs to carry.

![a stack of envelope layers showing the data payload at the base with optional meta, links, and errors layers above it and a bare single resource alternative](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-3.png)

**Position 1: the bare object.** Return the resource as the top-level JSON object, no wrapper:

```json
{
  "id": "ord_81000",
  "object": "order",
  "amount": 4999,
  "currency": "usd",
  "status": "paid"
}
```

This is the simplest possible thing. For a single resource — `GET /orders/{id}` — it is hard to beat. The client gets exactly what they asked for with no unwrapping. The downside shows up for *collections* and for *metadata*: if you later need to return a total count, a next-page cursor, or a list of warnings, a bare object has nowhere to put them without either polluting the resource (jamming `total_count` into the resource itself, which is wrong because it is not a property of the resource) or making a breaking change to wrap it after the fact.

**Position 2: the `data` wrapper.** Put the payload under a `data` key:

```json
{
  "data": {
    "id": "ord_81000",
    "object": "order",
    "amount": 4999,
    "currency": "usd",
    "status": "paid"
  }
}
```

The single `data` key buys you one thing: **room to grow.** Today it is just `data`; tomorrow you add `meta` or `links` as sibling keys without touching the payload's shape. It also gives every response a uniform top-level structure, so a client SDK can unwrap `data` generically. The cost is one extra level of nesting that every client must peel, on every response, including single-resource fetches where there is no metadata to justify it.

**Position 3: the full envelope.** `data` plus `meta` plus `links`:

```json
{
  "data": [
    { "id": "ord_81000", "object": "order", "amount": 4999, "currency": "usd", "status": "paid" },
    { "id": "ord_81001", "object": "order", "amount": 1299, "currency": "usd", "status": "pending" }
  ],
  "meta": {
    "total_count": 1240,
    "page_size": 2
  },
  "links": {
    "next": "/orders?cursor=ZW5kXzgxMDAx&limit=2",
    "prev": null
  }
}
```

This is where the envelope genuinely earns its keep: a **collection with pagination**. The `data` array holds the items; `meta` holds counts and page info; `links` holds the cursors or URLs for the next and previous pages. (JSON:API, the formal specification, standardizes exactly this — top-level `data`, `meta`, `links`, and `errors` members. We will look at it as a case study.) Without a wrapper, you have nowhere to hang the pagination cursor, and pagination is the single most common reason an envelope is worth its cost. For the deeper mechanics of cursors versus offsets and why a stable key beats a moving window, see the sibling post on [pagination at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale).

Here is the comparison that I keep in my head when deciding, and the figure that summarizes it:

![a matrix comparing the bare object, data wrapper, and full envelope styles by what each is best for, what metadata it carries, and its cost](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-8.png)

| Style | Best for | Carries | Cost |
| --- | --- | --- | --- |
| Bare object | Single resource (`GET /orders/{id}`) | Just the resource's fields | No place for metadata; adding a wrapper later is breaking |
| `data` wrapper | Single resource where you want uniform structure | Resource under `data`, room for siblings | One extra nesting level on every response |
| Full envelope (`data`/`meta`/`links`) | Collections with pagination, warnings, navigation | Items + counts + cursors + per-field errors | Verbose; overkill for a single resource |

The pragmatic position that I have landed on after shipping several of these: **be honest about consistency versus minimalism, and decide once for the whole API.** The two defensible choices are (a) bare for single resources, full envelope for collections — pragmatic, minimal nesting where you do not need it, structure where you do; or (b) `data` wrapper *everywhere*, single resources included, so the top-level shape is perfectly uniform and an SDK can always do `response.data`. Stripe famously does neither in the strict sense — single objects are bare, lists are wrapped in `{ "object": "list", "data": [...], "has_more": true, "url": "..." }` — which is essentially choice (a) with a list-specific envelope. The one thing you must *not* do is be inconsistent: bare here, `data`-wrapped there, full envelope in a third place, with no rule. That forces clients to special-case every endpoint.

### When an envelope earns its keep (and when it does not)

| Situation | Envelope? | Why |
| --- | --- | --- |
| `GET /orders/{id}` (single resource) | No (bare) | Nothing to carry beside the resource; nesting is pure overhead |
| `GET /orders` (collection) | Yes (full) | Needs `total_count`, `next`/`prev` cursors — pagination metadata has no other home |
| `POST /orders` returning the created resource | No (bare) | One resource; the client wants it directly |
| A response with non-fatal warnings | Yes (`meta.warnings`) | Warnings are about the response, not properties of the resource |
| A list that will *never* paginate (e.g. the fixed set of supported currencies) | Optional | A bare array is fine; a tiny `data` wrapper is fine; just be consistent with your other lists |

The rule underneath all of this: **metadata about the response does not belong inside the resource.** A `total_count`, a `next` cursor, a `request_id`, a deprecation `warning` — none of those are properties of an order. Putting them inside the order's body is a category error that will confuse every client and force you to document which fields are "really" part of the resource. The envelope exists precisely to give response-level metadata a home that is clearly separate from the resource it accompanies. If you have such metadata, you need an envelope; if you do not, you do not.

## 5. Nesting depth: embed or reference

A response body is a tree, and the question of how deep that tree goes — whether an order *embeds* its customer object or merely *references* it by ID — is a real trade-off between payload size, round-trips, and coupling. There is no universal answer; there is a decision procedure, shown in the figure.

![a decision tree for whether to embed a related resource inline, embed it on demand via an expand parameter, reference it by id, or link to a separate paged collection, branching on whether the relation is small and always needed or large and unbounded](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-6.png)

The forces at play:

- **Embedding** the related resource inline (`"customer": { "id": "cus_77", "name": "...", "email": "..." }`) saves the client a round-trip — they have the customer right there. But it inflates the payload (every order now carries a full customer), it duplicates data (the same customer embedded in a hundred orders), and it couples the order's representation to the customer's: change the customer's shape and you change every order response that embeds it.
- **Referencing** by ID (`"customer_id": "cus_77"`) keeps the order small and decoupled, but forces a second request (`GET /customers/cus_77`) if the client needs customer details. That second request can be a problem if a client is rendering a list of a hundred orders and needs each customer — the classic N+1 round-trip pattern.

The decision procedure I use:

1. **If the relation is small, bounded, and almost always needed together with the parent — embed it.** A payment's `amount`/`currency` are not a relation, they are intrinsic; embed without thinking. A short, stable summary object (a customer's `id` and `name`) that nearly every consumer of the order wants — embed a *trimmed* version.
2. **If the relation is sometimes needed and sometimes not — make it expandable.** Reference by ID by default, and let the client opt into embedding with a query parameter: `GET /orders/ord_81000?expand=customer`. This is Stripe's expandable-objects pattern, and it is genuinely good design: the default payload stays lean, and the client that needs the nested data asks for it explicitly and pays for it explicitly. (How to design the `expand`/`fields` parameters safely is the subject of the sibling post on [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql).)
3. **If the relation is large, unbounded, or a collection in its own right — reference it as a sub-collection.** An order's `refunds` could be zero or fifty; do not embed an unbounded array of full refund objects in the order body. Either embed a bounded, trimmed list with a link to the full collection, or reference it via a `links`/`refunds_url` that the client pages independently.

A useful way to frame the embed-versus-reference choice is as a bet about the client's access pattern. If nearly every consumer of an order also needs the customer's name in the same screen, then embedding (or expanding by default) saves a round-trip that *would have happened anyway*, so the embed is free in latency terms and only costs payload bytes. If most consumers only need the `customer_id` to link to a separate screen, then embedding forces every one of them to download and parse a customer object they will throw away, so referencing is the right default and expansion serves the minority who need depth. You are not choosing once and forever; you are choosing a *default* that matches the common access pattern and providing an `expand` escape hatch for the rest. Getting the default right means watching how clients actually use the data, not guessing — which is one more reason to instrument your API and let real usage, not intuition, set the default depth.

The general principle: **keep nesting shallow by default, and make depth opt-in.** A body that is three or four levels deep by default is hard to read, expensive to transfer, and tightly coupled. A body that is one or two levels deep with explicit, client-requested expansion stays lean for the common case and rich for the case that needs it. As a rough rule, if a default response routinely exceeds a couple hundred kilobytes because of eagerly embedded relations, you are paying a real latency tax — a 200 KB JSON body over a cold mobile link is typically several hundred milliseconds of pure transfer before the client renders a thing. For the deeper view of how payload size drives tail latency, and how this composes with caching and compression, the API-performance posts in this series and the [system-design API paradigm post](/blog/software-development/system-design/api-design-rest-grpc-graphql) go further; here we only need the rule: shallow by default, deep on request.

#### Worked example: the same order, referenced versus expanded

Default response — customer referenced by ID, refunds linked:

```http
GET /orders/ord_81000 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_81000",
  "object": "order",
  "amount": 4999,
  "currency": "usd",
  "status": "paid",
  "customer_id": "cus_77",
  "refunds": [],
  "created_at": "2026-06-18T09:30:00Z"
}
```

Lean: one round-trip, a few hundred bytes, no coupling to the customer's full shape. Now the client that is rendering an order-detail screen and needs the customer's name asks for it explicitly:

```http
GET /orders/ord_81000?expand=customer HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_81000",
  "object": "order",
  "amount": 4999,
  "currency": "usd",
  "status": "paid",
  "customer": {
    "id": "cus_77",
    "object": "customer",
    "name": "A. Rivera",
    "email": "a@example.com"
  },
  "refunds": [],
  "created_at": "2026-06-18T09:30:00Z"
}
```

The client got the nested customer because it asked, and the field swapped from `customer_id` (a string reference) to `customer` (an embedded object). Note the small contract subtlety: the *unexpanded* form is `customer_id`, the *expanded* form replaces it with `customer`. Stripe handles this by always using the same key (`customer`) and making its value either a string ID or a full object depending on expansion — a polymorphic field. Either convention works; document which you chose, because a client must know whether to look for `customer` or `customer_id`. The point stands: the default stays small, depth is opt-in, and nobody pays for the customer object on requests that do not need it.

## 6. Request bodies: the tolerant-reader, strict-writer tension

So far we have mostly talked about *responses*. Request bodies — what you *accept* — have their own discipline, and it pulls in two directions at once.

The two principles, both true, in tension:

- **Be a tolerant reader.** This is the robustness principle (Postel's law) applied to your inputs: be liberal in what you accept. Specifically, *ignore fields you do not recognize* rather than rejecting the whole request. Why? Because it makes the API forward-compatible: a client built against next year's version of your SDK might send a field your current server does not know about, and the request should still succeed for the fields you do understand. Tolerance of unknown fields is what lets clients and servers deploy independently without lockstep coordination.
- **Be a strict writer.** Validate the fields you *do* care about ruthlessly. Required fields must be present; types must be correct; enums must be in the allowed set; money must be a non-negative integer; strings must be within length limits. Reject violations with a precise, machine-readable error — do not coerce, do not guess, do not silently accept a malformed amount.

The figure shows the path an incoming order body takes through a strict writer that is also a tolerant reader: parse, validate strictly, drop unknown keys (never blindly assign them), and branch to either a clean `201 Created` or a per-field `422`.

![a request flow graph showing a client order body parsed and strictly validated then branching to either dropping unknown keys and returning a 201 created or returning a 422 problem json with per field errors](/imgs/blogs/designing-request-and-response-bodies-shape-and-naming-7.png)

Why does tolerant reading matter so much that it earns a named principle? Because of how deploys actually happen in a distributed system. You do not get to upgrade every client and the server in one atomic instant. A new SDK version ships, some clients adopt it, others lag months behind, and meanwhile your server is rolling out independently. During those overlapping windows, a client running tomorrow's SDK may send a field your currently-deployed server does not recognize, and a client running last year's SDK may receive a field it does not recognize. If either side rejects what it does not understand, every deploy becomes a coordinated, lockstep event across teams that do not share a release schedule — which in practice means deploys stop happening, or they break things. Tolerant reading is what *decouples* the deploy schedules: each side ignores what it does not need, so each side can evolve on its own clock. That decoupling is the entire reason the robustness principle exists, and it is why "ignore unknown fields" is not laziness but a deliberate compatibility guarantee. (The full ruleset for what counts as a safe change lives in the [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) sibling; the body is where you cash the guarantee in.)

The "drop unknown keys" step deserves emphasis because it is also a **security** control. The opposite — blindly binding every field in the request body onto your internal model — is the **mass-assignment** vulnerability (number 3 on the OWASP API Security Top 10, "Broken Object Property Level Authorization"). If your order model has an internal `is_admin_override` or `account_balance` field and you naively do `Order(**request_body)`, an attacker can set fields they were never meant to touch by including them in the body. The defense is to *allow-list* the fields you accept — bind only the known, intended inputs — and ignore the rest. Tolerant reading and mass-assignment defense are the same mechanism: parse into a known schema, not into your raw model.

Here is a validation middleware that embodies both principles — strict on known fields, tolerant of unknown ones, allow-listed against mass assignment:

```python
ALLOWED_FIELDS = {"amount", "currency", "customer_id", "line_items"}

def validate_create_order(body: dict) -> tuple[dict, list[dict]]:
    errors = []
    # Tolerant reader + mass-assignment defense: keep only allow-listed keys.
    clean = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}

    # Strict writer: validate the fields we accept.
    amount = clean.get("amount")
    if not isinstance(amount, int) or amount < 0:
        errors.append({"field": "amount", "code": "invalid",
                       "detail": "amount must be a non-negative integer in minor units"})
    currency = clean.get("currency")
    if currency not in {"usd", "eur", "gbp"}:
        errors.append({"field": "currency", "code": "unsupported",
                       "detail": "currency must be one of usd, eur, gbp"})
    if "customer_id" not in clean:
        errors.append({"field": "customer_id", "code": "required",
                       "detail": "customer_id is required"})
    return clean, errors
```

When validation fails, return a `422 Unprocessable Entity` with a machine-readable body that lists *every* field error at once — not the first one, all of them — so the client can fix the form in one round-trip rather than playing whack-a-mole. The full error-shape design (RFC 9457 `problem+json`, error taxonomies, actionable messages) is the subject of the sibling post on [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract); here is the shape:

```http
POST /orders HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "amount": 49.99, "currency": "xyz" }
```

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/problem+json

{
  "type": "https://api.example.com/problems/validation",
  "title": "Your request body failed validation.",
  "status": 422,
  "errors": [
    { "field": "amount", "code": "invalid",
      "detail": "amount must be a non-negative integer in minor units" },
    { "field": "currency", "code": "unsupported",
      "detail": "currency must be one of usd, eur, gbp" },
    { "field": "customer_id", "code": "required",
      "detail": "customer_id is required" }
  ]
}
```

Note that the client sent `49.99` (a float) and `"xyz"` (an unknown currency) and omitted `customer_id`, and the server reported all three problems in one response. That is the strict-writer payoff: fail loudly, fail completely, fail in a way the client can act on programmatically.

### Request and response symmetry

A small but high-value consistency rule: where it makes sense, the request body and the response body for the same resource should **look like each other.** If you create an order with `{ "amount": 4999, "currency": "usd", "customer_id": "cus_77" }`, the order you get back should contain those same field names with those same shapes — plus the server-assigned fields (`id`, `created_at`, `status`). The benefit is that a client can round-trip: read a resource, modify a field, and write it back (for a `PUT`) using the same field names it read, without translating between an "input shape" and an "output shape." 

Symmetry also pays off in your *own* tooling. When the request and response share field names and shapes, a single schema definition (one OpenAPI component, one type in your codegen) can describe both the input and the output with a small annotation marking which fields are read-only. Generated SDKs become cleaner — one `Order` model with read-only fields flagged, rather than a separate `OrderInput` and `OrderOutput` that drift apart over time. The contract is easier to document, easier to test, and easier for a client to reason about, all because you resisted the temptation to invent a different shape for input than for output.

The asymmetry that is legitimate: the response carries **read-only, server-controlled fields** that the request must not (and cannot) set — `id`, `created_at`, `updated_at`, computed totals, `status` transitions the server owns. A strict writer simply ignores those if a client sends them (tolerant reading again) rather than erroring, but it never lets a client *set* them. The principle: the request is a *subset* of the response in field names, restricted to the client-writable fields. When a client sees a field in the response, it should be able to guess whether it can send that field in a request by asking "is this something the server computes, or something I provide?" Document the read-only fields explicitly so there is no guessing.

#### Worked example: a well-shaped create-order round trip

The request — only the client-writable fields, all using the conventions we have built up:

```http
POST /orders HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 8f1d2c3b-4a5e-6f70-8192-a3b4c5d6e7f8

{
  "amount": 4999,
  "currency": "usd",
  "customer_id": "cus_77",
  "line_items": [
    { "description": "Pro plan, monthly", "amount": 4999 }
  ]
}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /orders/ord_81000

{
  "id": "ord_81000",
  "object": "order",
  "amount": 4999,
  "currency": "usd",
  "customer_id": "cus_77",
  "line_items": [
    { "description": "Pro plan, monthly", "amount": 4999 }
  ],
  "status": "pending",
  "is_paid": false,
  "refunds": [],
  "created_at": "2026-06-18T09:30:00Z",
  "updated_at": "2026-06-18T09:30:00Z"
}
```

Read the response against every rule from this post and it holds: `snake_case` throughout; `amount` is an integer of cents paired with `currency`; `id` and `customer_id` are opaque prefixed strings; `status` is a lowercase enum string; `is_paid` is an `is_`-prefixed boolean; `created_at`/`updated_at` are RFC 3339 with a `Z` offset; `refunds` is an empty array, not `null`; the request fields are a clean subset of the response fields; and the server added the read-only `id`, `status`, `is_paid`, `refunds`, `created_at`, and `updated_at`. The `Idempotency-Key` header on the request makes the `POST` safe to retry — that mechanism is its own deep topic, covered in the sibling post on [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) — and the `Location` header points at the created resource. This is what a body that will last looks like.

## 7. Stress-testing the shape

Good design survives contact with the awkward cases. Let me walk a few that bodies routinely hit, and show how the conventions above hold up.

**What happens when you must add a field?** Adding an *optional* field to a *response* is non-breaking, *if* clients are tolerant readers (they ignore unknown fields). This is the single most important reason to insist on tolerant clients and to never depend on a fixed field set: it is what lets you evolve the body without a version bump. The order response can grow a `tax_amount` field next year and every well-behaved client keeps working. The flip side: adding a *required* field to a *request* **is** breaking, because existing clients do not send it and will now fail validation. So new request fields must be optional with a sensible default, or they force a new version. (The full compatibility ruleset is the sibling post on [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change); the body is where these rules bite hardest.)

**What happens when you must remove or rename a field?** Both are breaking changes to a response, full stop — the renamed-`total`-to-`totalAmount` story that opened this post. The safe path is expand-then-contract: add the new field alongside the old, populate both for a deprecation window, mark the old one deprecated in the docs and ideally with a response warning, and only remove it after the window closes and metrics show no client reads it. Never do a rename as a single atomic change to a live response shape. The deeper field-lifecycle mechanics live in the sibling post on [schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely).

**What happens when the same entity appears in two contexts?** An order's `customer` embedded inline and the standalone `GET /customers/{id}` should be the *same shape* (the embedded version may be trimmed, but the fields it does include must match name-for-name and type-for-type). If they diverge, a client that built a `Customer` model from one endpoint cannot reuse it on the other. This is the "same entity always serialized the same way" rule from section 1, and it is most often violated precisely at embedding boundaries, where it is tempting to hand-craft a slightly different shape.

**What happens when a client sends a field as the wrong type?** A strict writer rejects it with a `422` and a per-field error, never coerces it. The temptation to be "helpful" — accepting `"4999"` (a string) for `amount` and parsing it to an integer — is a trap: it makes the contract fuzzy, lets bugs through silently (a client that meant to send `49.99` and accidentally stringified a float gets a surprising result), and means two clients can send "the same" request in incompatible ways. Pick one accepted type per field and enforce it.

**What happens when a list grows to fifty million rows?** This is why you never embed an unbounded collection in a body and why collection responses carry a pagination envelope. An order embedding all its (bounded) line items is fine; a `GET /orders` that tried to return all fifty million orders in one `data` array would time out, blow memory on both ends, and produce a body no client could parse. Bounded embeds, paged collections — the shape must assume the collection is large even when today it is small.

## 8. Case studies: how real APIs shape their bodies

A few well-known APIs to ground these rules in practice. I am describing publicly documented, stable design choices; where I am unsure of a current detail I keep it general.

**Stripe — minor units and expandable objects.** Stripe's API has represented monetary amounts as integers in the currency's smallest unit (`amount` in cents for USD) since its earliest public versions, paired with a `currency` field. This is the canonical real-world example of the minor-units rule and the reason it has become near-universal in payments APIs: a payments company simply cannot afford float drift. Stripe also popularized **expandable objects**: a related resource (like `customer` on a charge) is returned as an ID string by default and as a full nested object when the client passes an `expand` parameter, keeping default payloads lean while letting clients opt into depth. Stripe objects also carry an `object` field (`"object": "charge"`, `"object": "list"`) that names the type inline, which makes a logged or stored body self-identifying — a small touch that pays off in debugging.

**Google AIP — field naming as a standard.** Google's API Improvement Proposals codify field naming rules across all of Google's APIs: `snake_case` field names, `_time` or `_at` suffixes for timestamps (using `google.protobuf.Timestamp`, which maps to RFC 3339 in JSON), required-versus-optional semantics, and consistent resource naming. The relevant lesson is not the specific suffix Google chose but that they *wrote it down once and enforce it org-wide*, so that thousands of Google APIs feel like one API. That organizational consistency — a client who learned one Google API can predict the next — is exactly the payoff section 1 argued for, achieved at scale through a shared style guide and linting.

**JSON:API — the formal envelope.** JSON:API is a specification (not just a convention) for the full envelope: top-level `data`, `errors`, `meta`, `links`, and `included` members, with strict rules about resource identification (`type` + `id`), relationships, and sparse fieldsets. Its strength is that it removes envelope bikeshedding entirely — there is one right shape, tooling exists for it, and a JSON:API client can talk to any JSON:API server. Its cost is verbosity and ceremony: for a small internal API with one client, the full JSON:API structure is more machinery than the problem needs. It is a good case study in *when the full envelope earns its keep* — large, multi-client, long-lived APIs where the consistency and tooling are worth the verbosity — and when it does not.

**GitHub — bare resources and consistent meta-fields.** GitHub's REST API returns single resources as bare objects and uses consistent meta-fields across them — every resource carries an `id`, timestamps as RFC 3339 strings (`created_at`, `updated_at`, `pushed_at`), and `*_url` fields that link to related resources rather than embedding them by default (a referencing strategy that keeps payloads lean and lets the client fetch related data on demand). Pagination is handled with `Link` headers carrying `rel="next"`/`rel="prev"` rather than a body envelope — a reminder that pagination metadata does not *have* to live in the body; it can live in headers, and GitHub chose that path. The durable lesson is the consistency of the meta-fields: once you know GitHub names timestamps `*_at` and links `*_url`, you can predict those fields on any GitHub resource, which is the section-1 payoff again, achieved on a very large public API.

**A note on accuracy.** These are stable, documented design choices, but APIs evolve; treat the specifics as illustrative of the *principle*, and check the current docs before depending on an exact field name or behavior. The durable lessons — minor units for money, opt-in depth, write the envelope rule down once — outlast any particular version.

## 9. When to reach for an envelope (and when not to)

Because the envelope question generates the most argument, here is the decisive guidance, stated plainly.

**Reach for the full `data`/`meta`/`links` envelope when:**

- You are returning a **collection that paginates.** This is the dominant case. The cursor or page links and the total count have no honest home except a response-level envelope.
- You are building a **large, multi-client, long-lived** API where org-wide structural uniformity (and the tooling that rides on it, like JSON:API) is worth the verbosity.
- You routinely return **response-level metadata** — warnings, deprecation notices, request IDs surfaced in the body, rate-limit echoes — that is genuinely not a property of the resource.

**Do not reach for an envelope (return the bare object) when:**

- You are returning a **single resource** with no accompanying metadata. Wrapping `GET /orders/{id}` in `data` buys nothing but a nesting level every client must peel.
- Your API is **small and has one client**, and the only "collections" are short, fixed lists that will never paginate. A bare array is honest and simple.
- You would be adding `meta` and `links` keys that are **always empty.** An envelope full of `null`s and `[]`s is ceremony, not structure. If the metadata slots are never used, you do not need the envelope.

And two things never to do, regardless: do not mix styles within one API (bare here, wrapped there, with no rule), and do not put response-level metadata *inside* the resource to avoid an envelope — that is the category error that makes clients unable to tell a resource's real fields from response bookkeeping.

## 10. Designing the order body from scratch: a narrative

To pull all of this together, let me walk the actual reasoning of shaping the `/orders` response body, the way it happens in a real design session, with the stress tests applied as we go. This is the problem-solving spine: pose the design, reason to a decision, then attack the decision.

**Start with the domain, not the database.** An order, in the business sense, has: an amount and currency; a customer; a status; a set of line items; some timestamps; and a relationship to payments and refunds. Notice I did not start from a database table. The body's job is to represent the *resource as the caller understands it*, not to mirror your schema. If your `orders` table has a `tenant_id` foreign key and a `legacy_pricing_engine_flag`, those are storage concerns, not contract fields. The first design move is to decide which fields are *part of the resource's public meaning* and which are internal. Only the public ones go in the body.

**Shape each field by its category, using the rules we built.** Amount and currency: minor units plus an ISO code, `"amount": 4999, "currency": "usd"` — exact arithmetic, no float. Customer: a relationship, so reference it by ID by default (`"customer_id": "cus_77"`) and make it expandable, because most order-list views do not need the full customer but the order-detail view does. Status: a lowercase string enum (`"paid"`), documented as a closed set but designed for clients to tolerate new values. Line items: a bounded array of objects, always present (`[]` if somehow empty), each item itself shaped by the same rules (its own `amount`, its own `description`). Timestamps: `created_at` and `updated_at`, RFC 3339, UTC with `Z`. Booleans: `is_paid` with the prefix. ID: an opaque prefixed string. Refunds: potentially unbounded, so *referenced* as a sub-collection or embedded as a bounded, trimmed list with a link — never an eager array of full refund objects.

**Now choose the envelope.** A single `GET /orders/{id}` returns the order bare — there is no response-level metadata to carry. `GET /orders` returns the full envelope, because it paginates and the cursor and total count need a home. We decide this once and apply it to every resource in the API, so `/payments` and `/refunds` follow the same rule and a client never has to ask "is this one wrapped?"

Now the stress tests — the part that turns a plausible design into a durable one:

- **A client retries `POST /orders` on a timeout.** The body design alone does not solve this; the `Idempotency-Key` header does. But the body must be *consistent* between the original and the replayed response — the same `id`, the same fields — so a client comparing them sees they got the same resource. (Mechanism in the [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) sibling.)
- **Two writers race on the same order.** The body's `updated_at` and an `ETag` let a client do a conditional update and detect a lost write; the body shape supports it by always carrying `updated_at`. This is where the discipline of "always include the timestamp" pays off — concurrency control leans on it.
- **The orders collection grows to fifty million rows.** Because `GET /orders` was designed as a paginated envelope from day one, nothing changes: the client pages with cursors. Had we returned a bare unbounded array, this case would have forced a breaking change to add an envelope.
- **Finance asks for a `tax_amount` next quarter.** It is an additive optional response field, so it is non-breaking — tolerant clients ignore it until they are updated, and the contract evolves without a version bump. This only works because every field already follows the minor-units rule, so `tax_amount` slots in as another integer-plus-`currency` without inventing a new shape.
- **A partner pins to today's body for three years.** Because nothing in the body is a magic integer enum or a float, and because IDs are opaque strings, almost everything we might want to change later (add a status value, change ID format, add fields) is non-breaking by construction. The body was designed to be *evolvable*, which is the second pillar of the series spine.

The lesson of the narrative: the per-field rules are not independent style choices. They compose into a body that is *consistent* (one set of conventions), *correct* (exact money, unambiguous time, safe IDs), and *evolvable* (additive change is non-breaking, no fragile encodings to migrate). Shape each field by its category, choose the envelope once, then stress-test against retries, races, scale, and change — and you have a body that earns the word "lasts."

## 11. Key takeaways

- **Consistency is the master rule.** One casing convention, the same field name for the same concept everywhere, the same entity serialized the same way. Learning one resource should teach the next. Write the rules down once and lint them.
- **Money is a count, not a measurement.** Represent it as an integer of the currency's minor unit paired with an explicit currency code (`{ "amount": 4999, "currency": "usd" }`). Floats drift because decimals like `0.1` have no exact binary form; that drift becomes real reconciliation breaks at scale.
- **IDs are opaque strings, even when numeric.** It sidesteps JSON's `2^53` integer limit (which silently corrupts large IDs in JavaScript), discourages clients from doing arithmetic on IDs, and lets the ID format change without changing the field's type.
- **Timestamps are RFC 3339 with an explicit offset, always.** `2026-06-18T09:30:00Z` is unambiguous, human-readable, and lexicographically sortable. A timestamp without an offset is a regional bug waiting to happen.
- **Booleans get an `is_`/`has_` prefix; enums are lowercase strings, not magic integers.** The prefix signals the type and forces a clear assertion; string enums are self-documenting and stable under change. Clients should tolerate unknown enum values.
- **Always return an array (possibly empty), never `null`, for collection fields.** And distinguish `null` (known empty), omitted (unknown or unchanged on a PATCH), and empty (`[]`/`""`) deliberately — conflating them is a real source of bugs.
- **An envelope earns its keep mainly for paginated collections.** Bare object for a single resource; full `data`/`meta`/`links` for a collection that needs cursors and counts. Decide once for the whole API and never mix styles. Response metadata never lives inside the resource.
- **Be a tolerant reader and a strict writer.** Ignore unknown request fields (forward compatibility and mass-assignment defense via allow-listing); validate the fields you accept ruthlessly and reject violations with a complete, per-field `422`. Keep request and response shapes symmetric, with the request a subset of the writable response fields.

The body is the most-touched surface of your API's contract, and shape and naming choices made casually on a Friday afternoon become promises you cannot quietly take back. Design them as if a client you will never meet will parse them for years — because that is exactly what will happen. For the bigger picture of the API as a versioned contract and product, start at the series hub, [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), and when you are ready to put all of this on a checklist, the capstone [API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) ties the whole series together.

## Further reading

- **RFC 8259 — The JavaScript Object Notation (JSON) Data Interchange Format.** The grammar for the format your bodies are written in, including the (deliberately loose) treatment of numbers that is exactly why money and large IDs need care.
- **RFC 3339 — Date and Time on the Internet: Timestamps.** The internet profile of ISO 8601; the reason `created_at` should always carry an explicit offset.
- **RFC 7396 — JSON Merge Patch.** The semantics behind "omitted means unchanged, explicit `null` means clear" for `PATCH` request bodies.
- **RFC 9457 — Problem Details for HTTP APIs.** The standard `application/problem+json` error shape your strict writer should return; the foundation of the sibling [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) post.
- **Google AIP (API Improvement Proposals).** Google's org-wide API style guide; the case study for writing field-naming and resource rules down once and enforcing them at scale.
- **JSON:API specification.** The formal `data`/`meta`/`links`/`errors` envelope; read it to decide when the full envelope's tooling and uniformity are worth its verbosity.
- **Stripe API reference.** A long-lived, widely-imitated example of minor-unit money, expandable objects, inline `object` type fields, and consistent list envelopes.
- Within this series: the hub [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the capstone [API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2), and the siblings on [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract), [pagination](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale), [filtering and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql), and [content negotiation](/blog/software-development/api-design/content-negotiation-media-types-and-representations).

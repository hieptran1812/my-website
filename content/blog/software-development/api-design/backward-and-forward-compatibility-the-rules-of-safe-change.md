---
title: "Backward and Forward Compatibility: The Rules of Safe Change"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The precise rules for changing an API without breaking the callers you will never meet — what backward and forward compatibility actually mean, why adding an optional field is free but renaming one is an outage, the tolerant-reader principle that makes additive change safe, and the expand-and-contract pattern that turns a breaking change into four safe steps."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "compatibility",
    "versioning",
    "evolution",
    "tolerant-reader",
    "payments",
    "backward-compatibility",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-1.png"
---

A field rename took down every mobile client we had for forty minutes, and the diff that caused it was four characters long. The Payments service had a response field named `amount`. A well-meaning engineer decided `total` read better — `total` is what you'd say out loud, after all — and renamed it. The schema looked tidier. The unit tests passed, because the unit tests had been updated to read `total` in the same commit. The change shipped at 2pm on a Tuesday.

By 2:03pm the on-call channel was on fire. Every iOS and Android build older than the one we'd shipped that morning — which is to say, every build on every customer's phone — was reading `response.amount`, getting `null`, and crashing on the unwrap. The web app, which auto-deploys, had already updated to `total` and was fine. So from the web team's perspective nothing was wrong. From the perspective of a person standing in a checkout line trying to pay, the app had simply stopped working. We rolled back, added `amount` back as an alias, and spent the next two days writing the postmortem that produced this post. The lesson was not "be careful." The lesson was that **renaming a field is a breaking change, and adding one is not, and the difference between those two facts is the single most important thing you can know about evolving an API.**

This is the second pillar of designing an API that lasts. The first pillar is getting the contract right — correct HTTP semantics, honest status codes, consistent shapes, idempotent retries. But a correct contract that you can never change is a trap, because requirements change, the business grows, and the schema you froze on launch day will be wrong within a year. The whole point of the [API-as-contract framing](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) is that you are designing for a caller you will never meet, on a timeline of years, across client versions you cannot recall once they ship. So you must be able to **change the contract without breaking the people who already depend on it.** That ability has precise rules. They are not folklore, they are not "use your judgment," and they are not "just version everything." They are derivable from one principle, and once you internalize the principle you can classify any proposed change in about ten seconds.

By the end of this post you will be able to: define backward and forward compatibility precisely and say which one a given change threatens; apply the tolerant-reader principle and derive *why* additive change is safe; classify any change as breaking or non-breaking using the request/response/validation taxonomy; recognize the enum-evolution trap and the reading-versus-writing asymmetry that catch experienced engineers; execute the expand-and-contract pattern to make a "breaking" change in safe steps; and verify compatibility mechanically instead of trusting a reviewer's memory. We will walk all of it on the Payments and Orders API, classifying real proposed changes one by one.

![A two-panel comparison where adding an optional response field lets an old client survive while renaming a field produces a null and a 500 for every client](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-1.png)

## 1. Two directions of compatibility, and why both matter

The word "compatible" is doing too much work. When someone says "is this change backward compatible?" they often mean "will it break anything?" — but that question is ambiguous until you say *which side changed and which side stayed old.* There are two distinct directions, and a safe rollout almost always needs both.

**Backward compatibility** means: a **new server works with old clients.** You deploy a new version of the Payments API. The clients out in the world — mobile apps, partner integrations, that one cron job a former employee wrote — have not changed. Backward compatibility is the promise that those unchanged clients keep working against your new code. This is the direction people think about most, because it is the direction that produces 2pm outages. You control the server; you do not control when a customer updates their phone.

**Forward compatibility** means: an **old server works with new clients.** This sounds backwards until you remember that deployments are not atomic. During a rolling deploy, half your server fleet is running the new code and half is still on the old code, and a single client's two requests in a row might hit each. Or a client team ships a new app that starts sending a field your server does not understand yet, because the client release got ahead of the server release. Forward compatibility is the promise that an *older* server tolerates input shaped for a *newer* contract — typically by ignoring fields it does not recognize rather than rejecting the whole request.

Here is the cleanest way to hold the two apart. Backward compatibility is about a **new producer and an old consumer**; forward compatibility is about an **old consumer and a new producer's output reaching it early**, or equivalently an **old server consuming a new client's output.** In an HTTP API the "producer" of the response is the server and the "consumer" is the client, but for the *request* it is reversed — the client produces and the server consumes. That symmetry is exactly why a single API has compatibility obligations in both directions, and why the reading-versus-writing asymmetry we discuss in section 6 exists.

![A matrix contrasting backward compatibility for a new server with old client against forward compatibility for an old server with a new client](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-2.png)

Why does forward compatibility matter if you control your own server? Three concrete reasons, all of which I have been paged for:

1. **Rolling deploys are not atomic.** You deploy server `v2` to a fleet of fifty instances over ten minutes. For those ten minutes, a load balancer sends some requests to `v1` and some to `v2`. If your client (or your own newer instances) start writing data shaped for `v2` and a `v1` instance reads it back, `v1` must tolerate the new shape. This is the classic *mixed-version window.* If you never had forward compatibility, every deploy would be a potential outage during its own rollout.

2. **Stored data outlives the code that wrote it.** A payment record written by `v2` with a new `risk_score` field will be read by `v1` after a rollback. If `v1` rejects unknown fields, your rollback — your *safety mechanism* — becomes the thing that breaks production. Forward compatibility is what makes rollback safe.

3. **Client releases get ahead of server releases.** Mobile app review can take days. A web deploy takes minutes. It is entirely normal for a client to ship an expectation before the server has caught up. If the client sends an optional `idempotency_scope` field and the old server simply ignores it, you have decoupled the two release trains. If the old server returns `400 Bad Request` on the unknown field, you have chained them together, and now your mobile and backend teams cannot ship independently.

So the rule is: **design for both directions at once.** Most additive changes are backward *and* forward compatible if both sides follow the tolerant-reader principle, which is the engine that makes all of this work and the subject of the next section.

### A precise way to state the goal

Let me make this concrete with a tiny bit of set notation, because it sharpens the intuition. Think of the response body as a set of fields. Call the old set $F_{old}$ and the new set $F_{new}$. A purely **additive** change to the response satisfies $F_{old} \subseteq F_{new}$ — every field the old client knew about is still present, and the new server merely added more. If that subset relation holds *and* the meaning and type of each field in $F_{old}$ is unchanged, then an old client reading a new response finds everything it was looking for and a few things it does not recognize. If the client ignores what it does not recognize — the tolerant-reader principle — the change is backward compatible by construction.

The breaking changes are exactly the ones that violate the subset relation or change a member's meaning: removing a field (so $F_{old} \not\subseteq F_{new}$), renaming a field (which is removal of the old name plus addition of a new one), or changing a field's type or semantics so that the old client reads it and gets something it cannot use. The whole taxonomy falls out of this one relation, which is why memorizing rules is unnecessary once you have the principle.

### The mixed-version window, quantified

It is worth dwelling on *why* the two directions both matter even for a service with no external clients, because engineers who own both ends often assume they are exempt and learn otherwise during a rollout. The reason is that **a deploy is not an instant — it is a window during which two versions of your code serve traffic simultaneously**, and during that window every request is a coin flip between old and new.

Suppose you roll a new version to a fleet of $N$ instances, replacing them one at a time, and a full deploy takes $T$ minutes. At the midpoint of the deploy, roughly half the fleet is new and half is old. If your service handles $\lambda$ requests per second, then over a deploy window of $T$ minutes the number of requests served by a *mixed* fleet is on the order of $\lambda \cdot 60T$. Concretely, a service at 500 requests/second with a ten-minute rolling deploy serves about $500 \times 600 = 300{,}000$ requests during the window where old and new instances coexist. If even a fraction of those involve a write by a new instance read by an old one — or a request shaped by a new client landing on an old instance — and the old code is *not* forward compatible, that fraction becomes errors. A 1% incompatibility rate over 300,000 requests is 3,000 failed requests, on every single deploy, forever, for as long as the incompatibility exists.

That number is the entire argument for forward compatibility on internal services. You are not protecting a phantom external client; you are protecting yourself during the ten minutes of every deploy when your own fleet disagrees with itself about the schema. The same arithmetic governs rollback: if `v2` wrote data with a new field and you roll back to `v1`, then `v1` reads `v2`-shaped data for as long as it takes to drain and re-migrate — and if `v1` is strict, your rollback *is* your outage. Forward compatibility is what makes "just roll back" a safe sentence instead of a second incident.

So the rule generalizes past public APIs: **the moment your producer and consumer are not the same process deployed atomically, you are in a distributed system with a mixed-version window, and you owe both compatibility directions.** Two services in a fleet, a client and a server, a writer and a reader of the same database — all the same. (The fleet-wide version of this argument, including schema registries and reader/writer schema resolution for data at rest, is in [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale).)

## 2. The tolerant reader and the robustness principle

In 1980 Jon Postel wrote, in the TCP specification (RFC 761), a sentence that has governed protocol evolution ever since: *"Be conservative in what you do, be liberal in what you accept from others."* This is the **robustness principle**, also called **Postel's law**. Applied to APIs, it becomes the **tolerant reader** pattern, a term popularized by Martin Fowler: when you parse a message, read the fields you need, validate the parts you depend on, and **ignore everything else** — do not reject a message just because it contains a field you have never seen.

The word "tolerant" describes the *reader*. A tolerant reader is one that does not break when the message grows. Let me make the consequence as sharp as possible, because this single design choice is the difference between additive change being free and additive change being an outage.

Consider a JSON deserializer. Many libraries default to one of two behaviors when they encounter a field not present in the target type:

- **Strict:** raise an error on unknown fields. (Jackson's `FAIL_ON_UNKNOWN_PROPERTIES`, Go's `DisallowUnknownFields`, Pydantic's `extra="forbid"`, a `JSONDecoder` configured to reject extras.)
- **Tolerant:** silently ignore unknown fields. (Most defaults: Jackson off, Go's standard `json.Unmarshal`, Pydantic v2 default `extra="ignore"`.)

If your clients use a strict reader, then the moment your server adds a single optional field to the response, *every strict client fails.* The server did the safe thing — it only added — and the client still broke. The break is the client's fault in a moral sense, but the outage is yours in an operational sense, because customers cannot pay. This is why the robustness principle is a *contract-wide* discipline: the server promises to only add, and the client promises to tolerate additions. Neither half is sufficient alone.

![A two-panel comparison where a strict reader rejects an added field and halts the rollout while a tolerant reader ignores it and ships clean](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-4.png)

### Deriving why additive change is safe

Now we can derive the central claim rigorously, not just assert it. The claim is: **adding an optional field is non-breaking; adding a required field or removing a field is breaking.** Here is the derivation, split by direction.

**Adding an optional response field is backward compatible.** Take an old client `C` that was written against response schema $F_{old}$. It reads some subset of fields $R \subseteq F_{old}$ that it actually uses. The new server produces $F_{new} = F_{old} \cup \{f\}$ where $f$ is the new field. Since $R \subseteq F_{old} \subseteq F_{new}$, every field `C` reads is still present with the same meaning. The only difference `C` observes is the presence of `f`, which it does not read. *If* `C` is a tolerant reader, it ignores `f` and behaves identically. Therefore the change does not break `C`. The "if it is tolerant" is load-bearing; that is why tolerance is a precondition, not a nicety.

**Adding an optional request param with a default is backward compatible.** An old client `C` sends a request omitting the new param `p`. The new server, seeing `p` absent, applies its documented default. The behavior `C` observes is identical to before, *provided the default reproduces the old behavior.* If the default changes behavior — say a new `expand` param defaults to including more data and that more data overflows a buffer somewhere — then it is not really additive; you have changed the default behavior, which is breaking. So the precise rule is: an optional request param is non-breaking only if its absence reproduces the pre-existing behavior exactly.

**Adding a *required* request field is breaking.** Here the asymmetry bites. An old client `C` does not send the new field `f` because it was written before `f` existed. The new server now requires `f` and rejects the request — `400` or `422` — when it is absent. `C` was sending perfectly valid requests yesterday and gets rejected today. Nothing the server can do to a *required* field saves the old client, because the old client has no way to know it should start sending `f`. This is breaking by construction, and it is the most common self-inflicted wound in API evolution.

**Removing or renaming a response field is breaking.** A client that reads field `f` finds it gone (removal) or under a new name (rename, which is removal of the old name). It reads `null`, or it throws on a missing key, depending on its reader. Either way, behavior changes. A rename is the worst case because it *looks* additive — you added `total`! — but it is simultaneously a removal of `amount`, and removals break readers.

This is the whole game. Every other rule in the taxonomy is a corollary. Let me now lay the taxonomy out completely, because the corollaries have sharp edges.

## 3. The change taxonomy: what breaks and what does not

The fastest way to classify a change is to ask three questions in order: *Which surface does it touch — request, response, or validation? Does it only add, or does it remove/rename/retype/require? Will an existing client still parse and still send valid input?* Sort the change into the tree and the answer falls out.

![A taxonomy tree splitting a proposed change into request, response, and validation surfaces with safe and breaking leaves under each](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-3.png)

Here is the canonical classification. I have grouped it by surface, with the *reason* attached to each row, because the reason is what lets you classify a change the taxonomy does not list explicitly.

| Change | Breaking? | Why |
| --- | --- | --- |
| Add an optional **response** field | No | Tolerant readers ignore it; $F_{old} \subseteq F_{new}$ |
| Add a new **endpoint** / resource | No | No existing caller routes to it |
| Add an optional **request** param with a back-compatible default | No | Absence reproduces old behavior |
| Add a new **enum value** to a response | Depends | Safe only if clients tolerate unknown values (see section 4) |
| Relax a **validation** rule (accept more input) | No | Everything that was valid is still valid |
| Make a response field **nullable** when it never was | Sometimes | Old clients may assume non-null; can NPE |
| Add a new **optional** header | No | Old clients ignore unknown headers |
| Remove a **response** field | Yes | Readers of it now get nothing |
| Rename a field (request or response) | Yes | Old name disappears = removal |
| Change a field's **type** (string → number, scalar → object) | Yes | Old client's parse/deserialize fails |
| Make an optional field **required** | Yes | Old senders omit it and get rejected |
| Add a new **required request** field | Yes | Old clients never send it |
| **Tighten** a validation rule (reject input that was valid) | Yes | Previously valid requests now `422` |
| Change a **default** value or default behavior | Yes | Old clients relying on the old default shift silently |
| Change a **status code** for the same condition | Yes | Clients branch on status codes |
| Change the **error shape** (envelope, field names) | Yes | Error-handling code parses the shape |
| Change **URL structure** / move a resource | Yes | Hard-coded paths `404` |
| Remove an **enum value** from a request you accept | Yes | Clients still send the removed value |
| Reduce a numeric **range** or **page-size** cap | Sometimes | Clients near the old limit start failing |

Read the *Why* column as the load-bearing part. The "Breaking?" column is a lookup; the "Why" column is the skill. If you understand *why* tightening validation breaks (some request that was valid yesterday is invalid today, and a client out there is sending exactly that request), you can classify the next weird change you have never seen — "we're going to start rejecting `currency` codes that aren't ISO 4217" — without consulting a table. (That one is breaking: somebody is sending `USDT` or a lowercase `usd` and it works today.)

A few rows deserve a closer look because they trip people up.

**"Relax a validation rule" is non-breaking, "tighten" is breaking.** This is the validation-direction asymmetry. Loosening means the set of accepted inputs *grows* — everything valid before is still valid, plus more. No existing caller sends input that was valid and is now rejected, because nothing got rejected. Tightening means the accepted set *shrinks* — some input that worked yesterday fails today, and by hypothesis some client is sending exactly that input or you would not have needed to tighten. So: you can always become *more* permissive safely; becoming *less* permissive is a breaking change wearing a "bug fix" costume.

**"Change a status code" is breaking even if the new code is more correct.** Suppose you were returning `200 OK` with an error body when a payment failed (a sin we will cover elsewhere), and you fix it to return `402 Payment Required`. That fix is breaking, because clients branch on the status code. A client that checked `if status == 200: assume success` now treats a real failure as a success, or vice versa. The *correct* code is the one you should have shipped on day one; changing it after clients depend on the wrong one is a breaking change, and you handle it like one (version, or a long deprecation). Correctness and compatibility are different axes.

**"Make a field nullable" is the sneaky one.** If a response field `customer_email` was always present and non-null, clients wrote `payment.customer_email.toLowerCase()` with no null check, because your contract implied it was always there. The day you allow guest checkouts and `customer_email` becomes `null`, those clients throw a null-pointer exception. You added nothing and removed nothing — you *widened the type* from `string` to `string | null` — and that widening is breaking for readers who assumed the narrow type. The fix is to treat nullability as part of the contract from the start, and to document fields that may be absent.

To see why the status-code rule bites so hard, walk the before→after concretely. Suppose the create-payment endpoint used to return `200 OK` even when the payment was declined, putting the real outcome in the body:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "id": "pay_4410", "status": "declined", "decline_reason": "insufficient_funds" }
```

A client written against this learned the wrong lesson: it treats `200` as success and reads `status` from the body. Plenty of clients, in practice, just check the status code and skip the body on a `2xx`. Those clients have been quietly recording declined payments as successful for as long as this contract has existed. Now you fix it, correctly, to return a proper status:

```http
HTTP/1.1 402 Payment Required
Content-Type: application/json

{
  "type": "https://errors.acme-commerce.com/payment-declined",
  "title": "Payment declined",
  "status": 402,
  "detail": "The card was declined: insufficient funds."
}
```

The new response is *more correct in every way* — a real status code, a `problem+json` body per RFC 9457. And it is still a **breaking change**, because a client that branched `if (res.status < 400) markOrderPaid()` now correctly fails to mark the order paid, which changes its behavior, and a client that branched `if (res.status === 200) showSuccess()` now falls into an error path it never tested. Both the buggy old behavior and the fixed new behavior are *observable contract*, and changing the observable contract breaks the clients pinned to the old observation. This is the most counterintuitive consequence in the whole taxonomy: **fixing a bug in your status codes is a breaking change**, because clients calcify around whatever you actually shipped, correct or not. You handle the fix exactly like any other breaking change — stage it behind a version or a long-announced deprecation — even though every instinct says a bug fix should be free. It is not free; it is breaking with a guilty conscience.

## 4. The enum-evolution trap

This one deserves its own section because it catches *good* engineers, and the failure mode is silent until it is loud. Adding a new value to an enum looks like the most additive change imaginable — you are not touching any existing value, just adding `crypto` to a `payment_method` field that already has `card`, `bank_transfer`, and `wallet`. And yet adding an enum value can break clients in a way that no amount of tolerant-reader discipline prevents, because the break is not in the *parser*, it is in the *control flow*.

Here is the trap. A client receives the `payment_method` field and switches on it exhaustively:

```javascript
switch (payment.payment_method) {
  case "card":          return renderCardIcon();
  case "bank_transfer": return renderBankIcon();
  case "wallet":        return renderWalletIcon();
  // no default case
}
```

When the server adds `crypto` and sends it down, this switch falls through to nothing. Best case, the icon is blank. Worst case, the language throws on a non-exhaustive match. In strongly typed languages with exhaustiveness checking — Rust's `match`, a TypeScript `switch` with `never` in the default, Swift's `switch` — an *old binary compiled before the enum value existed* may not even handle the new case gracefully, because the compiler proved exhaustiveness over the old set of variants. The client did everything right at compile time and still breaks at runtime when the server's set of values grows past what the client knew.

So the rule on enums is genuinely subtle, and it is a *shared* contract obligation:

- **For the server:** adding an enum value to a **response** is safe *only if clients are documented and built to tolerate unknown values.* You cannot just decide it is safe; you have to have established the contract that says "this field may contain values you do not recognize; handle the unknown case." If you established that contract, adding values is forever non-breaking. If you did not, your first new value is a breaking change.
- **For the client:** never switch exhaustively on an enum you receive over the wire without a default/fallback branch. Treat an open enum as "one of these known values, *or something I do not recognize yet*," and render the unknown case as a graceful default — a generic icon, the raw string, a "see web for details" link.

The Protobuf world has built this directly into the type system, which is instructive. We will look at it in the case studies, but the short version: a Protobuf `enum` that follows the convention always reserves `0` for an `UNSPECIFIED`/unknown value, and the wire format preserves unrecognized enum numbers rather than failing — so a new value sent to an old reader survives as "unknown" rather than crashing. The lesson generalizes far past Protobuf: **design enums to be open, and document them as open, before you ever need to add a value.**

#### Worked example: an enum addition that breaks an exhaustive client

Let me walk this end to end on Payments, because seeing the wire and the consequence together makes the rule stick.

The Orders API has been live for a year. Orders have a `fulfillment_status` enum: `pending`, `shipped`, `delivered`, `cancelled`. A mobile client renders a status badge:

```swift
switch order.fulfillmentStatus {
case .pending:   badge = .gray("Processing")
case .shipped:   badge = .blue("On the way")
case .delivered: badge = .green("Delivered")
case .cancelled: badge = .red("Cancelled")
}
```

Swift requires `switch` over an enum to be exhaustive. The client compiled fine because at build time those were the only four cases. Now the business adds buy-online-pickup-in-store, and the server introduces a new status, `ready_for_pickup`:

```http
GET /v1/orders/ord_8821 HTTP/1.1
Host: api.acme-commerce.com
Authorization: Bearer <token>

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "ord_8821",
  "fulfillment_status": "ready_for_pickup",
  "total": { "amount": 4999, "currency": "USD" }
}
```

The JSON parses fine — the field is still a string, the reader is tolerant. But the Swift app decodes `fulfillment_status` into its `FulfillmentStatus` enum, and decoding a value that is not one of the four known cases *fails the decode*, throwing before the `switch` ever runs. The order screen shows an error state. Every customer who used pickup sees a broken order. The server added a value and broke a client that did everything the compiler asked.

The fix, applied retroactively and painfully, was to decode the enum with a fallback: model it as `case unknown(String)` so any unrecognized value decodes to `.unknown("ready_for_pickup")` and renders a neutral "See details" badge. Had the contract said from day one "`fulfillment_status` is an open enum; render unknown values as a generic state," the new value would have been a non-event. The cost of not designing the enum as open was a second emergency deploy and a behind-the-scenes data fix to suppress pickup orders from old app versions until adoption caught up.

The takeaway: an enum addition is "non-breaking" only relative to a contract that explicitly promised open enums. Absent that promise, it is breaking. Write the promise down on day one.

You can encode the open-enum promise directly in the schema so codegen and reviewers both see it. In OpenAPI 3.1, document the field as a string with a *suggested* enumeration and an explicit note that the list is open, rather than as a closed `enum` that the generator will turn into an exhaustive client type:

```yaml
components:
  schemas:
    FulfillmentStatus:
      type: string
      description: >
        Open enum. Known values are listed below, but clients MUST tolerate
        values they do not recognize and render them as a neutral state.
        New values may be added at any time without a version change.
      # Listed as examples, not a closed `enum`, so generated clients
      # produce an open type with an unknown fallback rather than an
      # exhaustive match that breaks on a new value.
      examples: ["pending", "shipped", "delivered", "cancelled"]
```

Contrast that with a closed `enum: [pending, shipped, delivered, cancelled]`, which tells the code generator "these are *all* the values," producing exactly the exhaustive client type that breaks on `ready_for_pickup`. The schema choice propagates all the way to whether a generated SDK has a `default`/`unknown` arm. Documenting the field as open *and* shaping the schema so codegen honors it is how you make the open-enum promise real instead of aspirational.

## 5. Where compatibility is actually decided

It helps to see *where* in a single request these compatibility checks happen, because the two directions are not abstract — they correspond to two concrete parse steps. A request crosses two checkpoints: the server parses the request body (forward-compatibility checkpoint — does the old server tolerate the new client's input?), and the client parses the response body (backward-compatibility checkpoint — does the old client tolerate the new server's output?).

![A flow graph showing a request parsed at the server with a strict-versus-tolerant branch and the response parsed at the client with an ignore-extras-versus-rename-NPE branch](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-6.png)

Trace it. The client sends `POST /payments`. The server's deserializer runs first — this is where forward compatibility lives. If the client sent a field the server does not know (`idempotency_scope`, say) and the server's reader is strict, the request dies here with `400` before any business logic runs. If the reader is tolerant, the unknown field is ignored and the handler proceeds. Then the handler builds a response and sends it back. The client's deserializer runs — this is where backward compatibility lives. If the server added a field, a tolerant client ignores it and reads `200 OK`. If the server renamed or removed a field the client reads, the client gets a `null` or a missing key and may throw — the dreaded NPE that is, from the user's seat, a `500`-equivalent: the app is broken.

The reason this picture matters is that it tells you *where to put your defenses.* Forward compatibility is enforced by configuring your server's deserializer to ignore unknown fields. Backward compatibility is enforced by never removing or renaming fields your clients read, and by documenting your clients' obligation to tolerate additions. Two different surfaces, two different controls, both required.

### The wire, both ways

Here is forward compatibility in action — a client sending a field the server has never seen, and the server tolerating it:

```http
POST /v1/payments HTTP/1.1
Host: api.acme-commerce.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 7e1c9a3f-2b44-4d18-9f3a-6c2e8b1d04aa

{
  "amount": 4999,
  "currency": "USD",
  "order_id": "ord_8821",
  "fraud_signals": { "device_fingerprint": "abc123" }
}
```

Suppose `fraud_signals` is a field a *newer* client started sending but the server has not deployed support for yet (client release got ahead of server release). A tolerant server ignores `fraud_signals` entirely and processes the payment exactly as if it were absent:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /v1/payments/pay_4410

{
  "id": "pay_4410",
  "amount": 4999,
  "currency": "USD",
  "status": "succeeded"
}
```

No error. The client gets its payment; the new field is simply dropped on the floor until the server learns to use it. That is forward compatibility, and it is what lets the client and server teams ship on independent schedules.

Now backward compatibility — the server having added a field the old client does not know:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /v1/payments/pay_4410

{
  "id": "pay_4410",
  "amount": 4999,
  "currency": "USD",
  "status": "succeeded",
  "risk_score": 0.07,
  "network_fee": { "amount": 30, "currency": "USD" }
}
```

The old client reads `id`, `amount`, `currency`, `status`, and ignores `risk_score` and `network_fee`. It behaves identically to before. That is backward compatibility, and it is what lets you ship server features without coordinating a client release.

## 6. The reading-versus-writing asymmetry

We have brushed against this twice; now let me state it as a rule because it is the single most useful heuristic in the whole post. **Adding a required field to a request is breaking. Adding a field to a response is not.** Same action — "add a field" — opposite verdict. Why?

Because the client *writes* the request and *reads* the response, while the server *reads* the request and *writes* the response. Compatibility is always about the **reader** being able to cope with what the **writer** produced. When you add a required request field, you are demanding that the *writer* — the client — produce something new. But the old client cannot be retroactively taught to produce a field that did not exist when it was compiled. The writer is frozen; you cannot add an obligation to a frozen writer. So requiring new input from old clients is impossible without breaking them.

When you add a response field, you are adding to what the *writer* — the server — produces. The reader — the client — is the one that has to cope, and a tolerant reader copes with extra fields trivially (it ignores them). The writer can always produce *more*; the reader can always ignore *more.* That is the asymmetry: **you can always add to what you produce, but you can never add to what someone else must produce.**

This generalizes into a clean operating rule:

- **You may freely add to outputs** (response fields, new endpoints, new optional headers) — readers tolerate additions.
- **You may freely add *optional* inputs** (request params with safe defaults) — old writers omit them, and the reader fills the default.
- **You may never add *required* inputs** without breaking old writers — they cannot produce what they do not know about.
- **Symmetrically, removing from your output breaks your readers** (clients reading the response), and **removing what you accept as input breaks your writers** (clients that still send it).

If you remember nothing else, remember: *additions to outputs are free; additions to required inputs are breaking; removals break the reader of whatever you removed.*

#### Worked example: classify six proposed changes

The Payments team has six changes queued for the next sprint. The product lead wants to know which can ship freely and which need a versioning or expand/contract plan. Let me classify each one the way I would in a design review, stating the verdict and the *why* in one breath. This is the drill that makes the skill automatic.

![A matrix classifying six proposed Payments changes as non-breaking, breaking, or depends, each with the precise reason it holds](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-7.png)

**Change 1 — Add a `network_fee` object to the payment response.** *Non-breaking.* It is an addition to the output. Old clients read `amount`, `currency`, `status`; they ignore `network_fee`. $F_{old} \subseteq F_{new}$, types of existing fields unchanged. Ship it freely, no version bump, no coordination. This is the textbook free change.

**Change 2 — Add a required `statement_descriptor` field to the create-payment request.** *Breaking.* It is a new *required input.* Every old client that calls `POST /payments` omits `statement_descriptor` because it did not exist when they shipped, and the server now rejects them with `422`. The frozen writer cannot be taught to send it. To do this safely you either make it optional with a sensible default (then it is non-breaking) or you treat the requirement as a breaking change and stage it. Almost always the right move is: ship it *optional* now, watch adoption, and only consider requiring it in a future major version if you truly must.

**Change 3 — Rename `amount` to `total` in the response.** *Breaking.* A rename is a removal of `amount` plus an addition of `total`. The addition is free; the removal breaks every client that reads `amount` (which is the exact bug from the intro). Verdict: breaking. The right way to do this is expand/contract, which we walk in section 7.

**Change 4 — Add a `POST /v1/payments/{id}/disputes` endpoint.** *Non-breaking.* A brand-new resource. No existing caller routes to a URL that did not exist, so nothing that worked yesterday breaks today. New endpoints are always additive. Ship freely.

**Change 5 — Add `crypto` to the `payment_method` enum in the response.** *Depends.* If the contract documented `payment_method` as an open enum and clients render unknown values gracefully, it is non-breaking. If clients switch exhaustively (the section-4 trap), it breaks them. Verdict for *our* API, where we documented open enums from day one: non-breaking. Verdict for an API that never made that promise: breaking, handle with care.

**Change 6 — Start rejecting `currency` codes that are not valid ISO 4217.** *Breaking.* This *tightens* validation. Some client is sending a lowercase `usd` or a typo'd `US` that the server currently accepts (perhaps it uppercases and trims internally). The day you tighten, those requests `422`. Tightening always shrinks the accepted set, and by hypothesis someone is in the part you are cutting off. The safe path: log the offending requests for a deprecation window, contact the senders, *then* enforce — or enforce only on a new version.

Five seconds per change once the principle is internalized. Two of the six are free (`network_fee`, the new endpoint), one is conditional (`crypto`, on the open-enum promise), and three are breaking (required `statement_descriptor`, the rename, the validation tightening) — and the three breaking ones each have a safe path that turns them into a sequence of non-breaking steps. That sequence is the expand/contract pattern.

### Configuring the readers — the practical control

Because so much rides on readers tolerating unknown fields, it is worth showing exactly where you set this, since the default is a coin flip across ecosystems and you do not want to discover yours during an incident. On the *server* side — your forward-compatibility control — you want the request deserializer to ignore unknown fields. In a Python service using Pydantic:

```python
from pydantic import BaseModel, ConfigDict

class CreatePaymentRequest(BaseModel):
    # extra="ignore" is the default in Pydantic v2, but make it explicit
    # so a future config change can't silently flip you to strict.
    model_config = ConfigDict(extra="ignore")

    amount: int
    currency: str
    order_id: str
```

In a Go service, the danger is the *opposite* — the standard `json.Unmarshal` is tolerant by default, but a well-meaning engineer can flip it strict, which is how forward compatibility quietly dies:

```go
// Tolerant (correct for a request reader): standard Unmarshal ignores
// unknown fields, so a newer client's extra field is dropped harmlessly.
err := json.Unmarshal(body, &req)

// DANGEROUS for an API request reader: DisallowUnknownFields() makes the
// reader strict, so any field the client adds before the server knows it
// returns an error. Use this only for config files, never the wire.
dec := json.NewDecoder(r.Body)
dec.DisallowUnknownFields()
err := dec.Decode(&req)
```

The lesson: **forward compatibility is one configuration line away from being broken, in either direction.** Make the choice explicit in code and assert it in a test, because the default differs by library and the failure mode (strict reader rejecting a new client's field) does not show up until a client team gets ahead of you. On the client side, your backward-compatibility control is the same idea pointed the other way: decode the response with a tolerant reader so server-added fields are ignored, and never model an over-the-wire enum without an unknown fallback.

#### Worked example: forward compatibility carries a deploy across the mixed-version window

Here is the scenario the mixed-version arithmetic warned about, walked concretely. The Payments team is rolling out support for a new `idempotency_scope` request field that lets a client narrow the scope of an idempotency key. The *client* SDK ships first (it went through app review weeks ago and is already on phones); the *server* support is deploying now, across a fleet of forty instances over an eight-minute rolling deploy.

During those eight minutes, a single client's `POST /payments` might land on a server instance that already understands `idempotency_scope`, or on one that does not yet. The new client always sends the field:

```http
POST /v1/payments HTTP/1.1
Host: api.acme-commerce.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 7e1c9a3f-2b44-4d18-9f3a-6c2e8b1d04aa

{
  "amount": 4999,
  "currency": "USD",
  "order_id": "ord_8821",
  "idempotency_scope": "order"
}
```

An instance that lands on the *old* code has never heard of `idempotency_scope`. Because the request reader is tolerant (the Go `json.Unmarshal` default, the Pydantic `extra="ignore"` default), the old instance silently ignores `idempotency_scope` and falls back to its existing behavior — global idempotency-key scope. The payment still succeeds, with the *old* semantics. The new instance reads `idempotency_scope` and applies the narrower scope. Both return `201 Created`; no request `400`s during the window.

Is the brief behavioral difference acceptable? In this case yes, because both behaviors are *correct* — the old one is just less precise, and the window is eight minutes. The decoupling it buys is enormous: the client team shipped weeks ago without waiting for the server, and the server deploy did not need a synchronized client release. *That* is the payoff of forward compatibility — not that nothing changes, but that nothing *breaks* while the two sides are out of step. Now run the counterfactual: had the old instances been strict readers, every request from the new client hitting an old instance during those eight minutes would have returned `400 Unknown field: idempotency_scope`. At 500 requests/second with even a quarter of traffic from new clients hitting old instances at the deploy midpoint, that is thousands of failed payments during a routine deploy. The single line `extra="ignore"` is the difference.

## 7. Expand and contract: making a breaking change safely

Sometimes a change really is necessary and really is breaking — you genuinely need to rename `amount` to `total`, or split a `name` field into `first_name` and `last_name`, or change a type. You cannot wish the breaking-ness away. But you can almost always *decompose a single breaking change into a sequence of non-breaking steps.* This is the **expand and contract** pattern, also called **parallel change**, and it is the most important evolution technique you will ever learn after the tolerant-reader principle itself.

The shape is always the same. A direct breaking change is "remove the old thing and add the new thing in one step." Expand/contract splits that into: **expand** (add the new thing while keeping the old thing — non-breaking, additive), then a **migration window** (both exist, clients move at their own pace, you may dual-write to keep them in sync), then **contract** (remove the old thing — only after every client has stopped using it). The breaking step (the removal) happens last, after you have made it harmless by ensuring nobody depends on the old thing anymore.

![A five-step timeline of an expand-and-contract rename moving from add the new field through dual-write and client migration to removing the old field](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-5.png)

#### Worked example: rename a field safely in three steps

Let me do the `amount` → `total` rename — the change that took us down — the *right* way. The principle: never do the rename as one step. Do it as expand, migrate, contract.

**Step 1 — Expand (add `total`, keep `amount`).** Ship a server release where the response carries *both* fields, holding identical values:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "pay_4410",
  "amount": 4999,
  "total":  4999,
  "currency": "USD",
  "status": "succeeded"
}
```

This is a pure addition — `amount` is untouched, `total` is new. Old clients keep reading `amount`; nothing breaks. On the request side, if clients can *write* this field too (say it appears in a create or update body), accept both names and treat them as synonyms, preferring whichever is present. This is **dual-write** on the read path: the server writes both fields to the wire so that whichever a client reads, it gets the right value. Internally you may store one column and project both onto the wire, or store both and keep them in sync — the wire contract is what matters to the caller.

**Step 2 — Migrate (move clients to `total`).** Now the slow part, measured in weeks or months, not minutes. You announce the rename through your changelog and developer comms. You add a `Deprecation` header to responses (covered in the [deprecation and sunset post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning)) signaling that `amount` is on its way out. Crucially, you **instrument which clients still read `amount`.** You cannot directly see which response fields a client reads — they all come down the wire together — so the practical move is to deprecate the *old behavior* loudly, ship new SDK versions that read `total`, and track SDK/client-version adoption. For request fields you *can* measure directly: count how many incoming requests still send the old field name. You do not contract until that count is effectively zero (or covers only clients you have explicitly chosen to abandon).

**Step 3 — Contract (remove `amount`).** Only after migration is complete — every active client reads `total`, no incoming request sends `amount` — do you ship the removal:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "pay_4410",
  "total": 4999,
  "currency": "USD",
  "status": "succeeded"
}
```

This step *is* breaking in the abstract — you removed a field. But you have made it *harmless* because no live client depends on `amount` anymore. The breaking change happened on a contract that no longer had any dependents on the removed surface. That is the entire trick of expand/contract: **you do the breaking step last, after you have driven the dependency count to zero.**

The same shape handles every "breaking" change:

- **Split `name` into `first_name`/`last_name`:** add the two new fields alongside `name` (expand); migrate clients to read the new ones; populate `name` from the concatenation during the window; remove `name` (contract).
- **Change a type, e.g. `amount` from a number `4999` to a money object `{ "amount": 4999, "currency": "USD" }`:** add a *new field* with the new type (`amount_money`) rather than mutating `amount` in place; migrate; remove the old `amount`. Never change a field's type under the same name — that is unavoidably breaking; rename-into-a-new-type instead.
- **Tighten a validation rule:** during the window, *accept* the old-format input and *log* it as deprecated; new clients send the new format; once the old-format count hits zero, start rejecting.

Expand/contract is slower than just making the change. That is the point. You are trading wall-clock speed for the guarantee that no caller breaks, and on a public or widely-consumed API that trade is almost always correct. The narrow exception is an internal API with a single consumer you control and can deploy atomically — then a coordinated breaking change can be cheaper than the dual-write bookkeeping. We will draw that line precisely in the "when to reach for this" section.

### Stress-testing the pattern

A pattern is only worth trusting if it survives the awkward cases, so let me stress-test expand/contract against the questions a skeptical reviewer should ask.

*What happens when a client writes the old field and reads the new one during the window?* This is the dual-write requirement made concrete. If a client `PUT`s a payment with `amount` (old name) and then reads it back expecting `total` (new name), the server must, during the window, accept `amount` on write and project it onto both `amount` and `total` on read. Dual-write is bidirectional: accept either name on input, emit both on output. If you only dual-write the read path, a client mid-migration that writes the old name and reads the new one sees a stale or empty value. The window is exactly the period where you carry this redundancy; it ends when you can prove no client uses the old name in *either* direction.

*What happens when two writers race during a type change?* Suppose you are migrating `amount` (a bare integer) to `amount_money` (an object), and during the window writer A sends the old `amount: 4999` while writer B sends the new `amount_money: {amount: 4999, currency: "USD"}`. The server needs a single source of truth and a deterministic projection: store the canonical form (say, the money object), derive the legacy `amount` from it on read, and when a write arrives in the legacy form, *upconvert* it to the canonical form (attaching the account's default currency). The races resolve because every write funnels through one canonical representation; the two field names are just two views of it. Without a canonical form, the two writers can clobber each other's representation and you get drift.

*What happens when a client never migrates?* This is the hard one, and it is why the contract step is gated on *zero* consumers, not on a calendar date. If after your deprecation window one stubborn integration still reads `amount`, you have a decision, not a deadline: either extend the window (cheap — `amount` is a few bytes on the wire), or make a deliberate, communicated choice to break that one client (rare, and only after direct contact). What you must *not* do is contract on schedule while a known consumer still depends on the field — that is just the intro outage with a project plan attached. The expand/contract pattern does not give you permission to break clients; it gives you a way to drive the count of breakable clients to zero *before* you do the breaking step.

*What happens at 50 million stored records?* If the migration touches data at rest, the contract step may require a backfill, and a backfill over 50 million rows is its own project — batched, throttled, resumable, run off-peak. The wire-contract rules in this post are necessary but not sufficient for data migrations; pair them with the storage-side discipline in [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations), which covers the expand/contract pattern at the database layer (add column, dual-write, backfill, switch reads, drop column — the exact same shape, one layer down).

## 8. Verifying compatibility instead of remembering it

The intro outage happened because the only thing standing between a breaking change and production was a code reviewer's memory, and memory fails. The mature answer is to make compatibility a *mechanical gate* — something CI enforces, not something a human is supposed to notice. You stack several gates from cheapest to strongest, and a breaking change has to defeat all of them to reach a client.

![A layered stack of compatibility gates from schema diff and contract tests through old-SDK replay and canary to a version bump only when truly breaking](/imgs/blogs/backward-and-forward-compatibility-the-rules-of-safe-change-8.png)

**Schema diff / breaking-change linters.** If you have a machine-readable schema — OpenAPI for REST, a `.proto` for gRPC, an SDL for GraphQL — a diff tool can compare the new schema against the last shipped one and *classify* each change. For OpenAPI, [`oasdiff`](https://github.com/oasdiff/oasdiff) reports breaking versus non-breaking changes and can fail CI on a breaking one. For Protobuf, `buf breaking` compares against a baseline and rejects field-number reuse, type changes, and removals. This catches the *renamed `amount`* before the PR merges, because the diff sees `amount` removed and flags it. This is the single highest-leverage gate; if you adopt only one thing from this section, adopt schema diff in CI. (We go deep on this in the [contract-testing post](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs); here I just want to establish that it is a gate, not a courtesy.)

**Consumer-driven contract tests.** A schema diff knows what *could* break a generic client; a consumer-driven contract (CDC) test knows what *will* break a *specific* client, because the consumer publishes a contract describing exactly the requests it sends and the response fields it reads. With a tool like Pact, the consumer's contract is replayed against the provider in the provider's CI. If the provider removes a field the consumer's contract reads, the provider's build fails — *with the name of the consumer team in the error.* This is strictly more precise than a schema diff: it knows that removing `risk_score` is fine because no consumer reads it, while removing `amount` fails because the mobile contract reads it. Schema diff is your fast, broad net; CDC is your precise, consumer-aware one. Use both.

**Old-SDK replay.** A pragmatic, low-tech gate that catches things both of the above miss: keep the previous released SDK version in CI and run its integration tests against the new server build. If the old SDK was reading `amount` and you removed it, the old SDK's tests fail. This is the closest thing to "what an old client actually does" because it *is* an old client.

**Canary with error-rate watch.** Even after the static gates pass, deploy the change to a small slice of traffic (1–5%) and watch the `4xx` and `5xx` rates and the client-side crash telemetry. A compatibility break that slipped through static checks shows up as a spike in client errors on the canary fleet before it reaches everyone. Automatic rollback on an error-rate threshold turns a forty-minute outage into a two-minute blip on 1% of traffic.

**Deprecation headers before any removal.** When a removal *is* eventually warranted (the contract step of expand/contract), it is preceded by a `Deprecation` and `Sunset` header window so clients get machine-readable warning. The removal is never a surprise.

**Version bump — the last resort.** And only when a change is *genuinely, unavoidably* breaking and cannot be staged additively do you cut a new version (URI, header, or media-type — see the [versioning post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning)). Versioning is expensive: you now maintain two surfaces, two sets of tests, two deprecation timelines. It is the tool you reach for *after* additive change and expand/contract have failed you, not before.

The ordering matters: cheap-and-static first (schema diff catches most breaks in milliseconds, in the PR), then consumer-aware (CDC), then behavioral (old-SDK replay), then runtime (canary), and only then the heavyweight options (deprecation window, version bump). A change should have to defeat the whole stack to break a customer, and it almost never can.

## Case studies: how the careful APIs actually do this

Theory is cheap; let me ground the rules in how three well-known systems actually evolve, because they have made these exact trade-offs at scale and their practices are documented.

### Stripe: additive-only changes plus dated versions

Stripe's public API is one of the most-studied examples of long-lived compatibility, and their approach is a clean two-part strategy. First, the **vast majority of changes are additive and unversioned** — Stripe adds new fields, new endpoints, and new optional parameters without changing a version, relying on the expectation that clients tolerate new response fields. Their developer documentation has long stated that they consider adding new API resources, adding optional request parameters, adding new response properties, and reordering properties to be *backwards-compatible* changes that clients should expect at any time. That is the tolerant-reader contract written into the developer policy: *we will add to the response; you must not break when we do.*

Second, for the genuinely breaking changes that additive evolution cannot cover, Stripe historically used **dated API versions** — a version string like `2023-10-16` pinned per account (and overridable per request via a header). A breaking change ships behind a new dated version; existing accounts stay on their pinned version and see no change until they explicitly upgrade. The combination is the lesson: *make almost everything additive so you almost never need a version, and when you truly must break, pin the break behind an explicit version so no one is forced to migrate on your schedule.* The reason this works is precisely the asymmetry from section 6 — adding to responses is free, so the overwhelming majority of API evolution can be additive, and versioning is reserved for the rare unavoidable break.

### Protocol Buffers: field numbers and the never-reuse rule

Protobuf bakes compatibility into the encoding itself, and the mechanism is worth understanding because it makes the abstract rules physical. In a `.proto` message, every field has a **field number**, and the *number*, not the field name, is what gets written on the wire:

```protobuf
message Payment {
  string id = 1;
  int64  amount = 2;
  string currency = 3;
  string status = 4;
  double risk_score = 5;   // added later — new field number
}
```

Because the wire format keys on the number, you can **rename a field freely** (`amount` → `total`) without breaking the wire, since the number `2` is unchanged — the name only matters to the generated code, not the bytes. You can **add a field** by giving it a new, never-before-used number (`risk_score = 5`); old readers encounter field number `5`, do not recognize it, and (per Protobuf's rules) preserve it as an unknown field rather than failing. That is the tolerant-reader principle implemented at the byte level. And there is one inviolable rule that mirrors section 3's "never remove" guidance: **never reuse a field number.** If you delete a field, you mark its number `reserved` so it can never be assigned to a different field later — because a new field reusing an old number would be silently misread by any client still holding the old meaning of that number. Protobuf's `reserved` keyword is the type-system enforcement of "never remove, only retire." The general lesson for *any* API: give every field a stable identity, never recycle it, and let readers preserve what they do not understand.

The deeper view of why this matters across a whole organization is in the system-design treatment of [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale), which covers the storage and streaming side (Avro/Protobuf schema registries, reader/writer schema resolution) that sits below the wire contract we own here.

### The "never remove, only deprecate" practice

The most durable practice across careful APIs — Stripe, GitHub, Google's APIs, and most internal platforms that have been burned once — is deceptively simple: **you almost never remove anything; you deprecate it and leave it.** A deprecated field keeps working; it just carries a documented warning, a changelog entry, and often a `Deprecation`/`Sunset` header signaling an end-of-life date. Removal, *if* it ever happens, comes only after a long window and only for surfaces with no remaining consumers — the contract step of expand/contract.

Why such restraint? Because the cost asymmetry is brutal. Keeping a deprecated field costs you a few bytes on the wire and a line in the docs. Removing a field that one forgotten integration still reads costs you an outage, an emergency rollback, and a partner's trust. Google's API design guidance (the AIP/AEP guidelines) and GitHub's API deprecation practice (announce, provide a `Sunset` date and migration path, then retire) both encode the same instinct: the default is *additive and permanent*, removal is *rare, announced, and gated on zero consumers.* This is not laziness; it is the rational response to designing for callers you cannot see and cannot coordinate with. (When removal genuinely must happen on a public API, the humane mechanics — `Deprecation`/`Sunset` headers, comms, migration windows — are their own discipline.)

### GitHub: a versioned REST surface plus a typed GraphQL one

GitHub is a useful contrast to Stripe because it runs two public APIs with two different evolution stories. The REST API leans on the same additive-only-plus-deprecation discipline, and when GitHub must make a breaking change it uses a dated REST API version (a `X-GitHub-Api-Version` header) so that callers opt into the new contract rather than being forced onto it — the same "pin the break behind an explicit version" instinct as Stripe. GitHub also publishes deprecations through changelog posts and, where applicable, `Sunset`-style signaling, giving integrators a window and a migration path before anything is retired. The general lesson: even at GitHub's scale, breaking changes are the exception, gated behind an explicit version a client must request.

The GraphQL API tells the other half of the story. GraphQL's type system makes *additive* evolution natural — adding a field to a type is non-breaking because GraphQL clients ask for exactly the fields they want, so a client that never selected the new field never sees it. But GraphQL has its own breaking-change rules: removing a field or an enum value, or making a previously nullable argument required, breaks the queries that depend on them. GitHub's GraphQL schema therefore carries `@deprecated` directives with a reason and a removal note rather than removing fields outright — the schema *itself* documents what is on the way out. The takeaway across both APIs is the one constant of this whole post: **grow the contract by addition, signal removals long in advance, and never yank a field out from under a caller.** The tooling differs (dated REST versions, GraphQL `@deprecated` directives, Stripe's account-pinned versions, Protobuf's `reserved` numbers) but the discipline is identical.

## When to reach for this (and when not to)

Compatibility discipline is not free, so be honest about when to spend it. The three tools — additive-only, expand/contract, and explicit versioning — form a ladder of increasing cost, and you should always reach for the cheapest one that handles the change. The table makes the trade-offs explicit.

| Strategy | What it handles | Cost to you | Cost to clients | Reach for it when |
| --- | --- | --- | --- | --- |
| Additive-only + tolerant readers | New fields, new endpoints, new optional params | Near zero — ship and forget | Zero — old clients untouched | Always, as the default; it covers the large majority of changes |
| Expand / contract (parallel change) | Renames, type changes, field splits, validation tightening | Moderate — dual-write and tracking over weeks | Low — clients migrate at their own pace | A change is genuinely breaking but staging it additively is feasible |
| Explicit version (URI / header / media-type) | Fundamental reshaping; semantics that two names cannot bridge; security fixes that must reject now | High — two surfaces, two test suites, two sunsets | High — a forced migration on your schedule | A break cannot be staged additively at all; last resort |
| Coordinated atomic break | Anything | Low *only if* you truly own both ends | Zero *only if* there are no other consumers | Internal RPC, single consumer, atomic deploy — and you have verified that claim |

Read the rows top to bottom as escalating cost and prefer the highest row that works. Most teams reach for the bottom rows far too early — versioning a change that a new optional field would have covered, or cutting `/v2` when expand/contract would have migrated everyone invisibly. Every row you descend, you take on maintenance burden and impose migration cost on people you will never meet.

**Reach for strict additive-only evolution + tolerant readers when:**

- You have **clients you do not control** — public API, partner integrations, mobile apps where old versions linger for months. Here every breaking change is an outage waiting for a customer who has not updated. Additive-only is mandatory, not optional.
- You **cannot deploy clients and servers atomically** — which is almost always, the moment you have more than one deployable. The mixed-version window during a rollout demands forward compatibility even for purely internal services.
- The API is **stored**, replicated, or replayed — payments, event logs, anything where data written by one version is read by another later. Schema evolution rules apply to data at rest exactly as they apply to the wire.

**Reach for expand/contract when:**

- A change is genuinely breaking but necessary (rename, type change, split, validation tightening). Decompose it into expand → migrate → contract rather than shipping the break directly. This is the default tool for any unavoidable change to an existing surface.

**Reach for an explicit version (URI/header/media-type) only when:**

- A breaking change **cannot be staged additively** — a fundamental reshaping of a resource, a change in semantics that two field names cannot bridge, a security fix that must reject previously-valid input *now*. Versioning is the heavyweight escape hatch. Use it sparingly, because every version you ship is a surface you must maintain, test, and eventually sunset. (The [versioning strategies post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) covers how to choose among URI, header, and media-type versioning — and makes the case for not versioning at all when additive change covers you.)

**Do NOT bother with the full ceremony when:**

- You own **both ends and deploy them together** as a single unit — an internal RPC between two services in the same repo, released atomically by the same pipeline, with no other consumers. Here a coordinated breaking change can genuinely be cheaper than the dual-write bookkeeping of expand/contract. *But verify the "no other consumers" claim* — the second a third team starts calling your "internal" endpoint, you are back in the public-API regime and probably do not know it. The most dangerous internal API is the one everyone forgot was internal.
- The field is **brand new and unreleased.** If `total` shipped to nobody — it is on `main` but never deployed, or only ever lived behind a feature flag that no client saw — you can rename it freely. Compatibility obligations attach only to surfaces a real caller has observed. Before that, change whatever you like.
- You are **pre-launch with no users.** Before your first real consumer, the cheapest thing in the world is to fix the contract. Spend that freedom; it ends the day someone integrates.

The throughline: compatibility cost scales with the number and reach of your consumers and your *inability to coordinate* with them. Zero uncontrolled consumers, atomic deploy — break freely. Many uncontrolled consumers, independent deploys — additive-only, expand/contract, version as last resort.

### A note on Postel's law's dark side

I have sold you hard on the tolerant reader, so let me be honest about its cost, because the robustness principle has a well-documented downside. *Being too liberal in what you accept hides bugs.* If your server silently ignores every field it does not understand, then a client that sends `amout` (typo'd `amount`) gets *no error* — the server drops the misspelled field, applies a default, and processes a payment for the wrong amount or no amount. The client developer has no idea anything is wrong until a customer complains. Tolerance turned a clear `400 Unknown field: amout` into a silent, latent bug.

This is the tension at the heart of Postel's law, and modern protocol designers (notably the IETF in RFC 9413, "Maintaining Robust Protocols") have pushed back on unconditional liberality. The resolution is not "be strict" or "be liberal" but **be tolerant of the things that are designed to vary and strict about the things that are not.** Concretely: ignore *unknown* fields (they are how the contract grows — tolerance here buys evolvability), but *do* validate the fields you depend on (a missing required field, a wrong type, an out-of-range value should be a clear `422` with a [problem+json](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) body, not a silent default). And for *known* clients you control, a strict-reader mode in development that warns on unknown fields catches typos early, while production stays tolerant for evolvability. Tolerance is a tool for *growth*, not an excuse to skip validation. Liberal about additions; strict about the contract you actually depend on.

## Key takeaways

- **Backward compatibility = new server, old client. Forward compatibility = old server, new client.** You need both, because rolling deploys, rollbacks, and out-of-order client/server releases all create mixed-version windows. Name which direction a change threatens before you ship it.
- **The tolerant-reader principle is the engine of safe change.** Be conservative in what you send, liberal in what you accept: read the fields you need, ignore the ones you do not. This is a *contract-wide* discipline — the server promises to only add, the client promises to tolerate additions. Either half alone is insufficient.
- **Additive change is safe; removal, rename, retype, and requiring are breaking.** Derive it from $F_{old} \subseteq F_{new}$ with unchanged meanings: a tolerant reader copes with additions, but nothing copes with a field that vanished or changed type underneath it.
- **The reading-versus-writing asymmetry:** you can always add to what *you* produce (response fields, new endpoints), but you can never add to what *someone else* must produce — so adding a required request field is breaking while adding a response field is free.
- **Watch the enum-evolution trap.** Adding an enum value breaks clients that switch exhaustively. Design and *document* enums as open from day one, and render unknown values gracefully; Protobuf's reserved-`UNSPECIFIED` and never-reuse rules are the canonical implementation.
- **Tightening validation is breaking; relaxing it is not.** Loosening grows the accepted set (nothing that worked stops working); tightening shrinks it (something a client sends today fails tomorrow). Changing a status code or error shape is breaking even when the new one is more correct.
- **Expand and contract turns any breaking change into safe steps:** add the new thing while keeping the old (expand), let clients migrate at their own pace (dual-write to keep both in sync), then remove the old thing only after its dependency count hits zero (contract). The breaking step happens last, when it is harmless.
- **Verify compatibility mechanically, not from memory:** schema diff (`oasdiff`, `buf breaking`) in CI, consumer-driven contract tests, old-SDK replay, and a canary with error-rate watch — stacked cheapest-first. Reserve an explicit version bump for breaks that truly cannot be staged additively.
- **Default to additive-and-permanent; never remove, only deprecate.** Keeping a deprecated field costs bytes; removing one a forgotten client reads costs an outage. Removal is rare, announced, and gated on zero remaining consumers.

## Further reading

- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html) — the authoritative definition of methods, status codes, and headers your compatibility rules build on.
- [RFC 9413: Maintaining Robust Protocols](https://www.rfc-editor.org/rfc/rfc9413.html) — the IETF's modern, nuanced take on Postel's law and where unconditional tolerance hides bugs.
- [Protocol Buffers: Updating a Message Type](https://protobuf.dev/programming-guides/proto3/#updating) — the canonical field-number, never-reuse, and reserved rules, and why they make Protobuf forward and backward compatible by construction.
- [Stripe API versioning and upgrades](https://stripe.com/docs/upgrades) — the dated-version-plus-additive-change strategy in production at scale.
- [oasdiff — OpenAPI diff and breaking-change detection](https://github.com/oasdiff/oasdiff) and [buf breaking](https://buf.build/docs/breaking/overview) — the schema-diff gates referenced in section 8.
- Martin Fowler, [TolerantReader](https://martinfowler.com/bliki/TolerantReader.html) — the original write-up of the tolerant-reader pattern for evolving services.
- Within this series: the [API-as-contract intro](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the [versioning strategies post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning); the deep dive on [schema evolution — adding, removing, renaming fields safely](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely); [contract testing and schema diffs](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs); and the [API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- Out of series: the system-design view of [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale), which covers the storage and streaming side beneath the wire contract.

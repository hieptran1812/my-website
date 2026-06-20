---
title: "Schema Evolution: Adding, Removing, and Renaming Fields Without Breaking Clients"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The field-level playbook for changing a payload schema over years without a single client 500 — the field lifecycle, why you can always add an optional field but never silently remove or retype one, and the expand-contract recipes for adding, removing, renaming, retyping, and extending enums, with the Protobuf reserved rules and the database migration underneath."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "schema-evolution",
    "compatibility",
    "protobuf",
    "versioning",
    "deprecation",
    "payments",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-1.png"
---

A few years ago I watched a one-line change take down a payments integration on a Friday afternoon. The change was, by any reasonable code-review standard, trivial. Someone had decided that `status` was a vague name for the field on an order — it could mean the payment status, the fulfillment status, or the overall order status — and renamed it to `state`. The diff was four characters longer. The unit tests passed because the tests had been updated in the same commit. The deploy went green.

Within ninety seconds the on-call channel lit up. A partner's mobile app, compiled six months earlier and pinned to a release that millions of phones still ran, did `order.status.toUpperCase()` on every order in the wallet view. The field was gone. `order.status` was now `undefined`. `undefined.toUpperCase()` threw, the render crashed, and the wallet screen showed a white box on every device that had not updated. We had not changed the meaning of anything. We had not removed any capability. We had renamed one key, and we had silently broken a contract that thousands of clients depended on — a contract nobody had written down, but which every one of those clients had encoded by reading `order.status` and trusting it would be there tomorrow.

That is the whole problem of schema evolution in one story. An API's payload is a contract, and a payload schema is a contract you will keep changing for years, across versions of clients you can no longer recall and cannot force to upgrade. The rule the rename broke is the one rule this entire post is built around: **you can always add an optional field, but you can never silently remove or retype one.** Everything else — the recipes for adding, removing, renaming, retyping, extending an enum, restructuring — is machinery for obeying that rule while still getting to change your schema. This post is the concrete, field-level companion to [the rules of backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change): that post tells you *what* is and isn't breaking; this one is the hands-on playbook for *how* you actually make each kind of change land safely on the wire.

![A vertical stack showing the four stages a field passes through over its life, introduce then in use then deprecate then remove, with a danger arrow marking the shortcut that skips a stage and breaks a live client](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-1.png)

By the end you will have a recipe for every common schema change — add, remove, rename, retype, change an enum, change requiredness, restructure — that keeps every existing client working through the change. You will know why Protobuf is unusually forgiving and what its one unforgivable mistake is. You will see how the API schema is *not* the database schema, and how an expand-contract migration runs underneath the wire. We will do all of it on the running Payments and Orders API: we will rename `status` to `state`, retype `amount` from integer cents to a decimal string, and add a new enum value — each one step by step, with the actual request and response bodies at every step.

## The field lifecycle: introduce, in use, deprecate, remove

Before any recipe, the mental model. A field is not a thing you add once and delete once. A field has a *life*, and that life has four stages. The most expensive bugs in schema evolution come from treating a field as if it has two stages — it exists, then it doesn't — and flipping it from one to the other in a single deploy.

The four stages are: **introduce**, **in use**, **deprecate**, and **remove**. You *introduce* a field by adding it to responses (and accepting it on requests). It enters *in use* when clients start reading and writing it; from that moment it is load-bearing. You *deprecate* it by announcing it will go away, stopping documentation of it, but continuing to return it so that nothing breaks while clients migrate off. Only after telemetry shows that essentially no client reads it anymore do you *remove* it.

Figure 1 above shows these stages as a stack with one extra arrow: the shortcut from "in use" straight to "remove." That shortcut is the rename that took down the wallet view. It is every "let's just delete the field, nobody uses it" that turned out to be wrong because *somebody* used it and you had no way to know. The discipline of schema evolution is, almost entirely, the discipline of never taking that shortcut.

Why does the shortcut break things, mechanically? Because of the *tolerant reader* principle and its limits. A tolerant reader is a client written so that it ignores fields it does not recognize and does not fall over when an optional field is absent. The robustness principle — "be conservative in what you send, liberal in what you accept" — says clients *should* be tolerant readers. In practice, most clients are tolerant of *extra* fields (they ignore them) but *intolerant* of *missing* fields they depend on. `order.status.toUpperCase()` is not a tolerant read of `status`; it is a hard dependency on `status` existing and being a string. So the asymmetry at the heart of compatibility falls straight out of how real clients are written:

- **Adding a field** is safe because tolerant readers ignore what they don't know. A client written last year does not break because you sent it a new key this year.
- **Removing or retyping a field** is dangerous because a client that *reads* that field has a hard dependency on its presence and type, and you cannot enumerate every such client.

That asymmetry is not a style preference. It is a direct consequence of the fact that you do not control the clients and cannot force them to upgrade. You are designing for a caller you will never meet. The lifecycle is how you respect that.

### The one rule, stated precisely

Let me state the rule with the precision it deserves, because the precision is where the safety lives:

> On a **response**, you may **add** an optional field at any time; you may **never** remove a field, change its type, or change the meaning of its existing values, without first running it through deprecate-then-remove over a window long enough for clients to migrate.
>
> On a **request**, you may **add** an *optional* field at any time, and you may **relax** a field from required to optional; you may **never** add a *required* field or tighten an optional field to required, because old clients that don't send it will start failing validation.

Notice the symmetry breaks between request and response. On responses, *you* are the writer and the client is the reader, so the danger is removing something the reader expects. On requests, the *client* is the writer and *you* are the reader, so the danger is requiring something the writer doesn't send. Both reduce to the same root: a change is breaking when it invalidates an assumption a deployed peer is already making. We will hold this line through every recipe below.

## The operation-to-recipe map

There is a small, finite set of schema changes you will ever make to a field, and each one has exactly one safe recipe. It is worth memorizing the table, because once you have it, schema evolution stops being scary and becomes mechanical.

![A matrix mapping each schema operation, add remove rename retype and add enum value, across response and request columns to its single safe recipe, with adding an optional field marked safe everywhere](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-2.png)

| Operation | On response | On request | Safe recipe |
|---|---|---|---|
| **Add a field** | Safe (optional) | Safe (must be optional) | Ship it optional, with a documented default; never required on a request |
| **Remove a field** | Breaking | Safe to stop reading it | Deprecate, keep returning it, remove only after telemetry shows zero reads |
| **Rename a field** | Breaking in place | Breaking in place | Add the new name, dual-write/dual-read both, deprecate the old, remove the old |
| **Retype a field** | Breaking in place | Breaking in place | Add a *new* field of the new type alongside; never reinterpret the old one |
| **Add an enum value** | Client must tolerate it | Breaking if you require clients to send it | Document that unknown values may appear; clients must default-handle the unknown |
| **Remove an enum value** | Breaking | Breaking | Don't; reserve it; stop producing it but keep accepting/documenting it as legacy |
| **Optional → required (request)** | n/a | Breaking | Don't in place; add a new required field on a new version, or validate softly first |
| **Required → optional (request)** | n/a | Safe | Relax freely; old clients still send it, new clients may omit it |

The shape of this table is the whole game. Read down the "Safe recipe" column and you see the same two moves over and over: **add the new thing alongside the old thing, run both in parallel for a window, then retire the old thing once telemetry proves it's safe.** That pattern has a name borrowed from database migrations: **expand and contract**. You *expand* the schema so both the old and new representations are valid at once, you migrate everyone across during the overlap, and then you *contract* by removing the old representation. Every safe schema change is a small expand-contract.

The contrast that makes it concrete is in-place change versus expand-contract.

![A before and after diagram contrasting an in-place rename that drops the old field in one deploy and breaks every reader against an expand contract sequence that adds the new field keeps the old one then removes it after telemetry](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-3.png)

| | In-place change | Expand and contract |
|---|---|---|
| **Deploys** | One | At least two, often spread over months |
| **Old + new valid together?** | No — only one shape is valid at a time | Yes — both shapes valid during the overlap window |
| **Rollback** | Hard — old shape is already gone | Trivial during expand; the old field is still there |
| **Client coordination** | Forced — everyone must upgrade at once | Gradual — each client migrates on its own clock |
| **Failure mode** | Every reader of the changed field breaks instantly | A client that didn't migrate keeps working on the old field |
| **Who pays the cost** | Every consumer, on your schedule, all at once | You, the producer, who carries two representations for a window |

The trade is explicit: expand-contract costs *you*, the producer, the complexity and storage of carrying two representations for a while, and in exchange it costs your *consumers* nothing. In-place change costs you almost nothing and costs every consumer a forced, synchronized, error-prone migration — which, for a public API, means it costs you your reputation when it fails. For any API with consumers you don't control, expand-contract is not the careful option; it is the only option.

## Adding a field safely

Adding is the easy case, and it is easy for a reason worth understanding rather than just enjoying. Start with the running example. Here is an order in the current Payments and Orders API:

```http
GET /v1/orders/ord_8Hq2 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Accept: application/json
```

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z"
}
```

Now product wants to expose an estimated delivery date. We add it:

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z",
  "estimated_delivery": "2026-06-25"
}
```

A client compiled before this change parses the JSON, sees a key it doesn't have a field for, and — if it is a tolerant reader — ignores it. The wallet app from the intro never asked for `estimated_delivery`, so it never reads it, so it cannot break on it. That is the whole mechanism: **a reader cannot depend on a field it has never seen.** Adding to a response is safe because it grows the set of keys, and growth is invisible to anyone not looking for the new key.

There are three disciplines that turn "technically safe" into "safe in practice," and skipping any of them is how a safe add becomes a support ticket:

1. **Make it optional, with a default.** The new field must be allowed to be absent or null, and the client must have a sensible interpretation when it is. If `estimated_delivery` can be `null` for orders that haven't shipped, document that `null` means "not yet estimated," and pick that default deliberately. A field that is "added" but then required-on-read in practice is a hidden trap.
2. **Document it.** A field nobody knows about is a field nobody uses, which means you've shipped complexity for no value. Add it to the OpenAPI spec, the changelog, and the reference docs the moment it ships.
3. **Don't add it to *requests* as required.** Adding `estimated_delivery` to the *response* is free. Adding a required `idempotency_key` to a request would break every old client that doesn't send it. New request fields must be optional, full stop. (For the deeper why behind requests-vs-responses asymmetry, see [the compatibility rules post](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change).)

In an OpenAPI 3.1 schema, "optional with a documented default" looks like this — note `estimated_delivery` is simply not listed under `required`, and it carries a nullable type and a description:

```yaml
components:
  schemas:
    Order:
      type: object
      required: [id, status, amount, currency, created_at]
      properties:
        id: { type: string }
        status: { type: string, enum: [pending, paid, shipped, refunded] }
        amount: { type: integer, description: "Total in minor units (cents)." }
        currency: { type: string }
        created_at: { type: string, format: date-time }
        estimated_delivery:
          type: [string, "null"]
          format: date
          description: "Estimated delivery date; null until the order ships."
```

The contrast with the broken alternative is sharp. If you had instead made `estimated_delivery` **required** in the response schema and a client's strict JSON-schema validator rejected payloads missing required fields, you'd break every consumer the moment any order lacked an estimate. "Required on a response" sounds harmless — you're *promising more*, not less — but it is a forward-compatibility hazard for validating clients and it boxes *you* in, because now you can never legitimately omit the field. Default to optional and you keep your options open. Add it required only if you are certain it will *always* be present, and even then, ask whether the certainty is worth the rigidity.

#### Worked example: adding a field that a validating client tolerates

A partner runs a strict reader: it validates every response against a frozen JSON Schema and logs a warning on any unknown property, but does not reject. We add `estimated_delivery`. Here is exactly what happens on the wire and in the partner's code.

Before the change, the partner's schema declares `additionalProperties: true` (the tolerant-reader default). The response now contains the new key:

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z",
  "estimated_delivery": "2026-06-25"
}
```

The partner's validator sees a property not in its schema. Because `additionalProperties` is `true`, validation passes silently; the extra key is carried but unused. Zero errors, zero behavior change. Now flip one switch in the partner's config — `additionalProperties: false` — and the *same* response *fails* validation with `must NOT have additional properties`. Same wire bytes, opposite outcome. The lesson cuts both ways: **adding a field is safe against tolerant readers and dangerous against strict ones**, so the contract must *tell* clients to be tolerant. The single most useful sentence in an API's evolution policy is: "We may add new fields to responses at any time; your client must ignore fields it does not recognize." That sentence is what makes additive evolution work, and it is the one [Stripe and most large APIs publish explicitly](https://stripe.com/docs/upgrades). It moves the burden of tolerance onto the reader, where it belongs, because the writer cannot enumerate the readers.

## Removing a field safely

Removing is where expand-contract earns its keep, and it is the operation people get wrong most often because "delete the field" *feels* like the simplest possible change. It is the most dangerous. You cannot see who reads a response field. There is no compiler error, no failing test in your repo, no log line. The only signal that a removal is safe is *telemetry that the field is no longer read*, and you cannot get that telemetry after you've removed the field — you have to gather it while the field is still there.

So removal is a contract operation: you keep the promise, then you stop documenting the promise, then — only once you can prove no one is relying on it — you stop keeping it. The recipe:

1. **Decide the field is going away.** Suppose we are dropping `legacy_reference`, an old integration code we no longer populate meaningfully.
2. **Stop documenting it.** Remove it from the OpenAPI spec, the reference docs, the SDK's typed model. New clients written from the docs will never know it existed. Add a `Deprecation` header on responses and announce a `Sunset` date, the way [the deprecation-and-sunset post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) describes.
3. **Keep returning it.** This is the non-obvious step. You keep the field on the wire — possibly frozen to a static value or `null` — for the entire deprecation window. The field is alive in the payload but dead in the docs. Existing clients that still read it keep working.
4. **Watch the telemetry.** Instrument reads. For a field, "reads" is harder to measure than for an endpoint, but you have options: log which API versions/clients are still sending requests to the affected resources, expose the deprecated field only in a way you can correlate, or — most practically — survey large known consumers directly. For internal APIs, grep the monorepo. For public APIs, watch the access patterns of clients pinned to older SDK versions.
5. **Remove it.** Once reads are at zero (or as close as you can responsibly get) and the `Sunset` date has passed, drop the field from the payload.

Here is the field during deprecation — present, but pinned and annotated as gone-soon. The response carries headers that announce the plan:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Deprecation: true
Sunset: Wed, 31 Dec 2026 23:59:59 GMT
Link: <https://docs.shop.example/changelog/legacy-reference>; rel="deprecation"
```

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z",
  "legacy_reference": null
}
```

The `legacy_reference` key is still there. A client that does `order.legacy_reference` gets `null` instead of crashing on a missing property — and if that client treated the value as optional (which a well-written reader of a sometimes-null field does), it keeps working. We have moved the field from "alive and documented" to "alive and undocumented," which is the safe waystation between in-use and gone. After the sunset date, with telemetry quiet, the final payload simply omits it:

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z"
}
```

The consequences of skipping step 3 are exactly the wallet-app disaster. If you go from "documented" to "absent" in one deploy, every client that reads the field — and you cannot enumerate them — breaks at the same instant. The fix is not heroics; it is the window. The window is cheap (you carry one extra key for a few months) and the break is catastrophic (every reader, all at once, no rollback that doesn't re-break the new clients). That asymmetry is why the window always wins.

One honest caveat on "telemetry shows zero reads": for a public API you can rarely prove *zero*. What you can do is prove that no client you can identify is reading it, give a long and well-communicated sunset, and accept that a tiny long-tail of abandoned integrations may break — those are integrations whose owners ignored a year of deprecation notices, and at some point the cost of carrying a field forever exceeds the cost of that long tail. The judgment call is *how long* the window is, not *whether* there is one.

## Renaming a field: never in place

A rename is the operation that started this post, and it is worth being blunt: **there is no such thing as renaming a field in place.** What looks like a rename is actually two operations — adding a new field and removing an old one — and you must run each through its own safe recipe, overlapping. An in-place rename is a removal-without-a-window of the old name plus an introduction of the new name in the same instant, which means it carries all the danger of an in-place removal.

Let's rename `status` to `state` on the order, the way we *should* have the first time. Four steps, and I'll show the wire body at each one.

![A left to right timeline of a safe rename, add state then dual write status and state then deprecate status then clients migrate then watch reads fall to zero then remove status, spread across twenty six weeks](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-4.png)

#### Worked example: renaming `status` to `state` in four steps, with the wire body at each step

**Step 1 — Add `state`, keep `status` (expand).** We add the new field. We do *not* touch the old one. Both are now in every response, and crucially they hold the same value — the server *dual-writes* both from the same source of truth so they can never disagree.

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "state": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z"
}
```

At this point nothing has broken and nothing has migrated. Old clients read `status` and are happy. New clients can start reading `state`. The two fields are kept in lockstep by the serializer, which reads one internal value and writes it to both keys. This is the single most important property: **during the overlap, old and new must always agree**, so that a client reading either one gets a correct answer. If you let them drift, you've created a worse bug than the rename you were avoiding.

On the *request* side, if clients can also *set* the status (say on a `PATCH`), you dual-*read*: accept either `status` or `state` on input, and if both are present, define a deterministic precedence (new wins, or reject the conflict with a `422`). Here is a server-side handler sketch that dual-reads on input and dual-writes on output:

```python
def normalize_state(body: dict) -> str:
    # Dual-read: accept either key on the request.
    new = body.get("state")
    old = body.get("status")
    if new is not None and old is not None and new != old:
        raise ValidationError(
            field="state",
            detail="state and status disagree; send only one",
            status=422,
        )
    return new if new is not None else old

def serialize_order(order) -> dict:
    # Dual-write: emit both keys from one source of truth.
    return {
        "id": order.id,
        "status": order.state,   # old name, same value
        "state": order.state,    # new name, same value
        "amount": order.amount_minor,
        "currency": order.currency,
        "created_at": order.created_at.isoformat() + "Z",
    }
```

**Step 2 — Deprecate `status`.** Now we announce that `status` is going away. Stop documenting it; the OpenAPI spec lists only `state`. Add the `Deprecation`/`Sunset` headers and a changelog link. The field is still on the wire and still correct; it is simply marked for retirement.

```http
HTTP/1.1 200 OK
Content-Type: application/json
Deprecation: true
Sunset: Sat, 31 Oct 2026 23:59:59 GMT
Link: <https://docs.shop.example/changelog/status-to-state>; rel="deprecation"
```

```json
{
  "id": "ord_8Hq2",
  "status": "paid",
  "state": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z"
}
```

**Step 3 — Clients migrate.** Over the deprecation window — weeks to months for an internal API, often six to eighteen months for a public one — clients update to read `state`. You help them: update SDKs so the generated model exposes `state` and marks `status` `@deprecated`, write a migration note, and email known integrators. Each client migrates on its own clock. Nobody is forced to coordinate a synchronized cutover.

**Step 4 — Remove `status` (contract).** Once telemetry says reads of `status` have fallen to zero and the sunset date has passed, you drop it. The serializer stops emitting it; the contract is now clean.

```json
{
  "id": "ord_8Hq2",
  "state": "paid",
  "amount": 4999,
  "currency": "usd",
  "created_at": "2026-06-18T09:14:02Z"
}
```

That is a rename. Four steps, two of which are "wait," spread over a window long enough that no living client is surprised. Compare it to the four-character in-place diff that crashed the wallet view: the safe version is more work for *us*, the producer, and zero work for every consumer. The unsafe version was less work for us and a fire drill for everyone downstream. The whole craft of schema evolution is being willing to pay the producer-side cost so your consumers never pay the consumer-side cost.

A note on the SDK and generated code, because a rename ripples there. When you regenerate the SDK after step 1, the typed model gains a `state` property; after step 2, you mark `status` as deprecated (most codegen tools emit a `@Deprecated`/`@deprecated` annotation from the OpenAPI `deprecated: true` flag, which makes the user's compiler or linter nudge them); after step 4, the property disappears and the *major* SDK version bumps, because *for the SDK* removing a property is a breaking change even though the API handled it gracefully. The API contract and the SDK contract are related but not identical — the API can deprecate gradually while each SDK release is a discrete artifact with its own semver. Plan the SDK major bump to land at the contract's contract step, not before.

## Changing a type: the `amount` saga

Retyping is the most insidious schema change because it *looks* like it might be safe and almost never is. The temptation is strongest with money. Our orders carry `amount` as an integer number of minor units — cents — which is the correct way to represent money in JSON, because floating-point dollars lose pennies. So `4999` means \$49.99. Good design. But suppose we expand to a currency with three decimal places, or a partner insists on a decimal-string representation to feed a system that parses `"49.99"` directly. The instinct is to change `amount` from an integer to a string.

**Do not reinterpret the existing field.** If you change `amount` from `4999` (integer cents) to `"49.99"` (decimal string) in place, here is the catastrophe: every client that does integer math on `amount` — `amount / 100` to get dollars, `amount > 5000` to gate a fraud check, `sum(amount for o in orders)` to total a cart — now receives a string. In a loosely typed language, `"49.99" / 100` is `NaN` or `0.4999` depending on coercion rules, `"49.99" > 5000` is a string-vs-number comparison that silently returns nonsense, and the sum concatenates strings. There is no error. There is no 500. There is just *wrong money*, computed silently, in production, until someone reconciles the books and finds the numbers don't add up. A retype that changes the *interpretation* of an existing field is the worst kind of breaking change, because it breaks *correctness* without breaking *availability* — the system stays up and lies to you.

The safe recipe is the expand-contract you already know, applied to type: **add a new field of the new type alongside the old one; never reinterpret the old.**

**Step 1 — Add `amount_decimal` as a string, keep `amount` as integer cents.** Both present, both correct, both derived from the same internal money value.

```json
{
  "id": "ord_8Hq2",
  "amount": 4999,
  "amount_decimal": "49.99",
  "currency": "usd",
  "state": "paid"
}
```

A client doing integer math reads `amount` and is correct. A client wanting the decimal string reads `amount_decimal` and is correct. Neither one was reinterpreted; you *grew* the schema with a new key, which is the always-safe add.

**Step 2 — Deprecate `amount`** (if you truly intend to retire the integer form), document `amount_decimal` as canonical, set a sunset, dual-emit through the window.

**Step 3 — Migrate clients** to `amount_decimal` over the window.

**Step 4 — Remove `amount`** once telemetry is quiet.

In practice, money fields are so load-bearing that many teams *never* contract — they keep both `amount` (cents) and `amount_decimal` (string) forever, because the cost of carrying two representations of a number is tiny and the risk of removing a field that touches money is enormous. That is a legitimate outcome: **the contract step is optional.** Expand-contract guarantees you *can* remove safely; it does not require that you ever do. Sometimes the right answer is to expand and simply never contract.

The same recipe covers every retype: integer to string, string to structured object, scalar to array, a flat timestamp to a structured `{value, timezone}`. In every case: add a new field of the new type, run both in parallel, never reinterpret the bytes a client already knows how to read. If you find yourself writing "we'll just change the type and update the docs," stop — that sentence is the bug.

## Changing an enum

Enums hide a compatibility trap that catches even careful teams, and it is the inverse of everything above: with enums, **adding a value is the dangerous-feeling-but-usually-safe operation, and the real break is on the client side, not yours.**

Our order `state` enum is `pending`, `paid`, `shipped`, `refunded`. Product introduces partial refunds, so we add a value: `partially_refunded`. Adding an enum value to a *response* does not remove or retype anything — every old value still appears and means what it meant. So why is it a hazard? Because of how clients handle enums. A client that wrote:

```python
if state == "refunded":
    show_refund_badge()
elif state == "shipped":
    show_tracking()
else:
    show_nothing()   # silently swallows unknown states
```

...will quietly mishandle `partially_refunded` — best case it shows nothing, worst case a `match`/`switch` with an `assert_never` or an exhaustive-enum check *throws* on the unrecognized value and crashes. The break did not happen in your payload; it happened in a client that assumed it had seen every possible value forever.

So the recipe for adding an enum value is a *documentation and contract* discipline, not a payload trick:

1. **Tell clients, in the contract, that new enum values may appear.** This is the enum equivalent of "we may add fields." Publish: "The set of values for `state` may grow; your client must handle an unrecognized value gracefully (e.g., a default case) rather than crash."
2. **Add the value.** It is now a valid response value.
3. **On requests, be more careful.** If clients *send* an enum (say to filter `GET /orders?state=...`), a new value you accept is fine, but you must not *require* clients to send a new value — and removing a value clients might send is breaking.

In an OpenAPI schema you simply extend the `enum` list — but the meaningful change is the prose policy:

```yaml
state:
  type: string
  enum: [pending, paid, shipped, refunded, partially_refunded]
  description: >
    The order's lifecycle state. This set may grow over time; clients
    MUST treat any unrecognized value as an opaque string and handle it
    via a default branch rather than failing.
```

**Removing an enum value is breaking** — the same as removing a field. A client may have logic keyed on `refunded`, and a response producing it is part of the contract. The safe move when a value becomes obsolete is to *stop producing it* but keep *accepting and documenting* it as a legacy value, and to **reserve** it — never reassign that string to a new meaning. Re-using `"shipped"` to mean something new later would be a silent retype of the value's meaning, which is exactly the `amount`-saga bug in enum clothing. Reserve removed values the same way you reserve removed field names: write down that they are retired and never to be reused.

## Changing requiredness

Requiredness changes are asymmetric, and the asymmetry is the now-familiar request/response split, so it is short to state and easy to get right once you see it.

- **Optional → required on a *request* is breaking.** You are now demanding something old clients don't send. Every old client's request fails validation with a `422` (or `400`). Don't do it in place. If you genuinely need a field to be required — say a new compliance rule requires every payment to carry a `customer_reference` — your options are: (a) add it required on a *new version* of the endpoint, leaving the old version's looser contract intact; or (b) introduce it optional, soft-validate (log when it's missing, accept anyway), drive adoption to ~100% through outreach, *then* flip it to required once telemetry shows essentially every caller already sends it. Option (b) is the request-side expand-contract.
- **Required → optional on a *request* is safe.** You are relaxing a constraint. Old clients still send the field (no harm); new clients may omit it. Relax freely. The only care needed is defining the behavior when the field is omitted — pick and document a default.
- **On a *response*, "required" is a promise *you* make.** Making a response field required (always present) is a forward-compatibility hazard for the reasons in the adding section — it boxes you in. Relaxing a response field from always-present to sometimes-absent is *breaking* for any client that read it unconditionally, so treat it as a removal: deprecate-then-soften over a window, with a documented null/absent meaning.

| Change | Direction | Breaking? | Why |
|---|---|---|---|
| Optional → required | Request | **Yes** | Old clients don't send it; validation fails |
| Required → optional | Request | No | Constraint relaxed; old clients unaffected |
| Always-present → sometimes-absent | Response | **Yes** | Old readers that read it unconditionally break |
| Sometimes-absent → always-present | Response | No (but boxes you in) | Readers tolerant of absence still work; you lose flexibility |

The pattern, again: tightening a constraint on the side the *peer* controls is breaking; relaxing is safe. Requiredness is just constraint-tightening by another name.

#### Worked example: making a request field required without breaking old clients

A new regulation requires every payment to carry a `customer_reference` for audit. Today the field doesn't exist; tomorrow it must be present on every `POST /v1/payments`. The naive move — add it to the `required` list in the OpenAPI request schema and ship — breaks every existing integration on the next deploy, because the moment your validator runs, an old client's request looks like this:

```http
POST /v1/payments HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Content-Type: application/json

{ "order_id": "ord_8Hq2", "amount": 4999, "currency": "usd" }
```

...and your now-stricter validator answers with a `problem+json` rejection that the client has no idea how to fix, because it was compiled before the field existed:

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/problem+json

{
  "type": "https://docs.shop.example/errors/missing-field",
  "title": "Missing required field",
  "status": 422,
  "detail": "customer_reference is required",
  "instance": "/v1/payments"
}
```

Every old client now fails to create a payment. That is a complete outage for the un-migrated, on your schedule, with no rollback that doesn't undo the regulation you were trying to satisfy. The request-side expand-contract avoids it in three phases:

**Phase 1 — Introduce optional + soft-validate.** Add `customer_reference` to the request schema as *optional*. Accept requests with or without it. When it's missing, *log* the omission with the client identity and *accept the request anyway*. You now have a precise telemetry signal — exactly which callers don't yet send the field — without breaking anyone. The same un-migrated request above succeeds with a `201`, and you record a metric `payment.customer_reference.missing{client_id=...}`.

**Phase 2 — Drive adoption.** Update the SDK so the field appears (and is documented), email the callers your telemetry flagged, give them a deadline, and watch the missing-field metric fall. This is the migration window; it can run weeks to months. You are not blocking anyone — you are converting a hard break into a measured rollout.

**Phase 3 — Flip to required, once telemetry says it's safe.** When `payment.customer_reference.missing` has been at (or essentially at) zero for a full window, change the schema to mark the field `required`. Now the `422` above only fires for callers who ignored the entire migration — and you flip it *knowing* who they are, having given them every chance. The constraint is finally tightened, but the tightening landed on an empty set.

The contrast with the naive flip is the whole point: same end state (`customer_reference` required), but one version breaks everyone instantly and the other breaks no one who paid attention. The cost is carrying an optional-then-required field through a window and instrumenting one metric. That is the request-side mirror of the response-side deprecate-then-remove, and it generalizes to any constraint you need to tighten: introduce it loosely, measure conformance, and flip it only once the measurement says the flip is a no-op.

## Restructuring: flattening and nesting

Sometimes the change isn't one field but the *shape* — you want to flatten a nested object, or nest a set of flat fields into a sub-object, or split one resource into two. These are the changes most likely to genuinely need a new version, because they touch many fields at once and there is no single key whose addition or removal captures the change.

Suppose the order currently has a flat shipping address:

```json
{
  "id": "ord_8Hq2",
  "shipping_name": "A. Customer",
  "shipping_line1": "1 Market St",
  "shipping_city": "San Francisco",
  "shipping_postal": "94105"
}
```

...and you want to nest it under a `shipping_address` object, both because it's cleaner and because you're adding a `billing_address` and want symmetry:

```json
{
  "id": "ord_8Hq2",
  "shipping_address": {
    "name": "A. Customer",
    "line1": "1 Market St",
    "city": "San Francisco",
    "postal": "94105"
  }
}
```

You have two honest paths. The first is **additive restructure within one version**: add the new nested `shipping_address` object alongside the old flat fields, dual-write both (the serializer fills the nested object *and* the legacy flat keys from the same internal address), deprecate the flat fields, and contract after the window. This is just expand-contract applied to a cluster of fields at once. It works, and it is the right call when the restructure is localized.

The second path is **a new representation or version**, and you reach for it when the restructure is pervasive — many fields move, the resource splits, the semantics shift — such that dual-emitting both shapes makes the payload a confusing union of old and new. When carrying both shapes in one payload would itself be a worse contract than either shape alone, version it: serve the old shape at `/v1/orders` and the new shape at `/v2/orders` (or via a media-type/header version, per [the versioning strategies post](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning)), keep v1 alive through its own sunset, and let clients opt into v2. The decision tree:

![A decision tree starting from a needed schema change splitting into can it be additive leading to add optional or expand contract versus must it break leading to whole shape restructure or a new version side by side](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-8.png)

The honest rule of thumb: **stay additive (expand-contract within one version) as long as you can; cut a new version only when additive evolution would produce a payload that is worse than a clean break.** Versioning fragments your surface area — now you maintain two contracts, two sets of docs, two test suites — so it is a real cost, not a free escape hatch. Most field-level changes never need it. Restructures sometimes do.

## Protobuf-specific rules: why field numbers make it forgiving

Everything above was framed in JSON, where fields are identified by *name*. Protobuf — the schema language behind gRPC, covered in depth in [the gRPC and Protocol Buffers post that ships in this series](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs) — identifies fields by *number*, and that one decision makes Protobuf unusually forgiving about evolution, with exactly one rule you must never break.

Here is the order message:

```protobuf
syntax = "proto3";

message Order {
  string id = 1;
  string state = 2;
  int64 amount_minor = 3;
  string currency = 4;
}
```

On the wire, Protobuf does not send field *names*. It sends, for each field, a tag composed of the field *number* and a *wire type* (the low-level encoding: varint, 64-bit, length-delimited, etc.), followed by the value. The number `2` and the wire type, not the string `"state"`, is what identifies the field. This has profound consequences for evolution:

- **Renaming a field is free and non-breaking.** Because names aren't on the wire, renaming `state` to `status` in the `.proto` changes nothing about the bytes. Old and new code that share the same field *number* interoperate perfectly. (Your *generated code* changes — `order.getState()` becomes `order.getStatus()` — so it's a source-level change for *your* codebase, but it is not a wire break.) This is the opposite of JSON, where the name *is* the identity and a rename is the dangerous operation.
- **Adding a field is non-breaking** as long as you use a new, unused number. Old readers see an unknown field number and skip it (proto3 preserves unknown fields), exactly the tolerant-reader principle baked into the format.
- **Changing the *type* in a way that changes the wire type is breaking**, the same as JSON. `int64` to `string` changes the wire type; old bytes won't decode. But some type changes are wire-compatible (`int32`/`int64`/`uint32`/`bool` are all varints and interconvert with caveats) — the Protobuf language guide enumerates which.

And then the one unforgivable mistake, the rule that the format cannot protect you from:

> **Never reuse a field number.** When you remove a field, *reserve* its number (and ideally its name) so it can never be reassigned.

![A before and after diagram contrasting reusing Protobuf field number four for a new meaning which corrupts old bytes against reserving number four and the name status so any reuse becomes a compile error](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-6.png)

Why is reuse catastrophic? Because the number is the identity. Suppose field `2` was `state` (a string, length-delimited), you remove it, and later — forgetting it was ever used — you add a new field `2` called `ttl_seconds` as an `int32` (varint). Now an old client that still sends bytes tagged with number `2` and the length-delimited wire type hits your new code, which expects number `2` to be a varint. Best case: a decode error. Worst case — if the wire types happen to be compatible — the old `state` bytes are *silently decoded as* `ttl_seconds`, and you have a `ttl_seconds` of `1885434739` (the integer interpretation of the ASCII bytes of `"sta"`) with no error anywhere. That is the `amount`-saga corruption, at the binary level, and the format gives you no warning.

The fix is `reserved`, and it turns the silent corruption into a loud compile error:

#### Worked example: removing a Protobuf field with `reserved`

We retire `state` (field `2`). The wrong way is to just delete the line. The right way:

```protobuf
syntax = "proto3";

message Order {
  reserved 2;
  reserved "state";

  string id = 1;
  // field 2 is retired; its number and name are reserved forever
  int64 amount_minor = 3;
  string currency = 4;
}
```

The `reserved 2;` line tells the compiler that field number `2` is retired and may never be reassigned. `reserved "state";` does the same for the name, so nobody can later add a *different* field called `state`. Now if a teammate, six months later and unaware of the history, writes `int32 ttl_seconds = 2;`, the protoc compiler refuses to build it because `2` is reserved. The corruption that the format could not detect at runtime is caught at *compile* time, before a single byte ships. That is the entire purpose of `reserved`: it converts the one unforgivable evolution mistake from a silent production bug into a build failure.

The discipline generalizes: reserve removed field numbers *and* names, every time, no exceptions. The cost is two lines in a `.proto`. The bug it prevents is silent binary data corruption that may not surface until reconciliation. There is no clearer cost-benefit in all of schema evolution. The same logic extends to `enum` values in proto: removing an enum constant should be paired with `reserved` on its number and name, for exactly the same reason you reserve a JSON enum string — a future reuse would silently re-map old data to a new meaning.

It is also worth being precise about *which* JSON-side changes Protobuf makes free that JSON does not, and vice versa, because the two formats invert each other's risk profile. In JSON the field *name* is the identity, so a rename is the dangerous operation and a type change is dangerous; in Protobuf the field *number* is the identity, so a rename is free and a number-reuse is the catastrophe. A team that runs both a JSON/REST surface and a gRPC surface over the same domain has to hold both rule sets at once: the same internal "rename `status` to `state`" is a free `.proto` edit but a full four-step expand-contract on the REST side. The serializer/codegen layer is where the two reconcile — which is the natural segue into how generated code reacts to every one of these changes.

## What each change does to generated code and SDKs

A schema change does not stop at the wire. For any API that ships an SDK or relies on code generation, every add, remove, rename, and retype propagates into a *typed artifact* in the consumer's language — and the rules for evolving that artifact are subtly different from the rules for evolving the wire, because an SDK release is a discrete, versioned package, not a continuous contract. Getting this wrong is how a wire-safe change still breaks a consumer's *build*.

Walk the operations again, this time from the generated-code side:

- **Adding a response field.** On the wire, safe. In the SDK, also safe and purely additive: regenerate, and the typed model gains a new (optional) property. A consumer who upgrades the SDK sees a new accessor; a consumer who doesn't is unaffected. This is a *minor* SDK version bump.
- **Adding a required request field.** Wire-breaking (as we covered) — and in a *typed* SDK it is doubly visible, because the generated request constructor or builder now demands an argument that the consumer's call site doesn't pass, so it fails to *compile*. The compile failure is actually a feature here: it forces the consumer to confront the new requirement at build time rather than discovering it as a runtime `422`. But it means the SDK change is a *major* bump.
- **Removing a response field.** Wire-breaking for readers — and in the SDK, removing a property is a source-breaking change: any consumer code that referenced `order.legacyReference` no longer compiles. This is why you *deprecate* the SDK property (most codegen emits a `@Deprecated`/`@deprecated`/`[Obsolete]` annotation from the OpenAPI `deprecated: true` flag) for one or more minor releases *before* removing it in a major release. The annotation turns into a compiler warning at the consumer's call site — the SDK's equivalent of the `Deprecation` header.
- **Renaming a field.** On the wire you ran a four-step expand-contract. In the SDK, the cleanest path mirrors it: add the new property (`getState()`), keep and `@deprecated`-mark the old one (`getStatus()`) so both compile during the overlap, then drop the old property in a major release. The SDK major bump should be *timed to land at the contract's contract step* — when the wire actually removes `status` — so that the SDK and the wire retire the old name together. Bumping the SDK major early strands consumers who haven't migrated; bumping it late leaves a dead accessor returning the now-removed field as null.
- **Retyping a field.** Because the safe wire recipe is "add a new field of the new type," the SDK simply gains a new typed accessor (`getAmountDecimal(): string` alongside `getAmount(): number`). No existing accessor's type changes, so nothing breaks. This is yet another reason never to retype in place: an in-place retype changes a generated accessor's *return type*, which is a silent source break in statically typed languages — `int amount = order.getAmount()` stops compiling when the return type flips to `String`, and in dynamically typed SDKs it does the same silent-wrong-value damage as on the wire.
- **Adding an enum value.** The riskiest for codegen. A naive generator turns a closed `enum` into a closed type (a Java `enum`, a Rust `enum`, a TypeScript union). When the API adds `partially_refunded`, an old SDK that deserializes into that closed type may *throw on deserialization* of the unknown variant — the SDK breaks before the consumer's code even runs. The defenses: generate enums as *open* (an unknown-value fallback variant like `UNKNOWN_TO_SDK`, or deserialize the raw string and only map known values), and document that consumers must rebuild the SDK to gain new variants. This is why "we may add enum values" is not just a wire policy but an SDK-generation requirement: the generator must produce *tolerant* enum handling or the additive change becomes a deserialization crash.

The throughline: **the wire contract and the SDK contract evolve on different clocks and with different breakage rules, and the serializer/codegen layer is the seam.** A change that is additive on the wire can be source-breaking in the SDK (a removed property), and a change that is breaking on the wire can be caught helpfully at compile time in the SDK (a new required field). Plan SDK semver around the *consumer's build*, not just the wire: minor bumps for additive properties and deprecations, major bumps for removals and required-field additions, timed so the SDK's major lands with the wire's contract step. The SDK is the place where the abstract "tolerant reader" rule becomes a concrete codegen setting — and if your generator emits closed enums and strict models by default, your tolerant-reader policy is a lie no matter what your docs say.

## The API schema is not the database schema

A trap that quietly undermines all of the above: conflating the API contract with the database table. They are different schemas with different lifetimes, different audiences, and different change rules, and the serializer between them is what lets each evolve on its own clock.

The database schema serves *your* code — it can change as fast as you can ship a migration, and you control every reader (your own services). The API schema serves *callers you don't control*, on a timeline of years. If you let the API payload be a thin mirror of your table — `SELECT *` straight to JSON — then *every* database migration becomes an API change, and you've coupled your slowest-to-evolve contract (the public API) to your fastest-changing one (your tables). Rename a column for internal clarity and you've renamed an API field and crashed the wallet view. The serializer is the firewall that prevents this: it maps internal column names and types to external field names and types, so you can rename `status_code` to `state_enum` in the database while the API keeps emitting `state`, and vice versa.

This decoupling is what makes the two-sided expand-contract possible. When we renamed `status` to `state` on the wire, the database underneath ran its *own* expand-contract, on its *own* schedule, joined only at the serializer:

![A graph showing the client reading JSON from the API contract through a serializer that maps both ways down to a new database column and an old database column running in parallel before the old column is dropped after the API removes its field](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-5.png)

The figure shows the two migrations as parallel tracks. On the API side: add `state`, dual-write, deprecate `status`, remove `status`. On the database side: add a `state` column, backfill it from the old `status_code` column, dual-write both columns, drop the old column — the [zero-downtime schema migration](/blog/software-development/database/zero-downtime-schema-migrations) playbook, which goes deep on how to add a column, backfill without locking, and drop safely under live traffic. The crucial insight is that **these two tracks need not be synchronized.** The database can finish its migration and drop the old column long before the API removes the deprecated field — the serializer simply reads the new column and writes *both* JSON keys. Or the API can rename its field while the database column keeps its old name forever, if there's no reason to migrate the column. The serializer absorbs the mismatch.

So the rule is: **design the API schema deliberately, as its own artifact, not as a reflection of your tables.** Choose field names for the caller's clarity, choose types for the caller's safety (cents-as-integer regardless of how the database stores money), and let the serializer translate. The reward is that you can refactor your database freely without touching the contract, and evolve the contract without being held hostage by your storage. Two schemas, two expand-contracts, one serializer between them.

## Verifying that a change is actually safe

Recipes are necessary but not sufficient; the discipline is verifying, *before* you ship, that a change you *believe* is additive really is. Three mechanisms, in increasing order of strength.

**Schema diff / breaking-change linters.** A tool diffs the new schema against the old and flags breaking changes mechanically. For Protobuf, `buf breaking` checks a `.proto` against a baseline and fails CI on a removed field, a reused number, a changed type. For OpenAPI, `oasdiff` (and similar) classify each change as breaking or non-breaking. This is the cheapest, most automatable guard: it catches the "I thought this was safe" mistakes at PR time. Wire it into CI so that a breaking change requires an explicit, reviewed override.

```bash
# Fail CI if the new proto breaks wire compatibility with the committed baseline.
buf breaking --against '.git#branch=main'

# Classify OpenAPI changes; exit non-zero on a breaking change.
oasdiff breaking openapi-v1.yaml openapi-v2.yaml --fail-on ERR
```

**Contract tests / consumer-driven contracts.** A schema diff knows your schema changed; it doesn't know whether any *consumer* actually depended on the part that changed. Consumer-driven contract testing (Pact and similar) inverts this: each consumer publishes the subset of the contract it actually uses, and your provider build verifies it still satisfies every published consumer contract. Now "is this safe?" is answered by "do all real consumers' expectations still pass?" rather than by a blanket schema rule. This is the subject of [the contract-testing post](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs) — the point here is that it is the strongest *pre-deploy* signal that a removal or retype is safe, because it's grounded in what consumers genuinely read.

**Telemetry and canaries.** The last line is production reality. For removals, instrument field reads (as discussed) so "zero reads" is a measured fact, not a hope. For any change, canary it: route a small slice of traffic to the new schema, watch error rates and client-side crash telemetry, and roll back instantly if a deprecated-then-removed field turns out to still have readers. The order — lint at PR time, contract-test in CI, canary in production — is layered defense, each catching what the previous layer can't.

There is one more verification trick worth naming, because it catches the consequences a schema diff cannot reason about: **replay real traffic against the candidate.** Capture a sample of production requests (and the responses you sent), point a shadow copy of the new build at the captured requests, and diff the new responses against the old ones field by field. A removed field shows up as a key that vanished; a retype shows up as a value whose JSON type changed; a renamed field shows up as one key disappearing and another appearing. Crucially, replay diffing surfaces *unintended* changes too — the field you didn't realize your serializer change dropped, the timestamp format that shifted because you upgraded a library. It is the empirical complement to the schema diff: the linter checks what the schema *says*, replay checks what the service *actually emits*. For a payments API where a silent change to `amount` is a financial incident, the response-replay diff is the cheapest insurance you can buy.

A small consequence worth flagging on the *cost* side of additive evolution: every field you add — and especially every field you add-but-never-contract, like the permanent `amount` plus `amount_decimal` pair — grows the payload. One extra short string field per object is negligible, but a list endpoint returning 200 orders, each carrying three deprecated-forever fields you never removed, multiplies that bloat by 200. Over a cold mobile link, an extra few kilobytes of redundant fields is typically tens of milliseconds of transfer time you didn't need to spend, and it compounds across every request. This is not an argument against expand-contract; it is an argument *for* eventually running the contract step when you safely can, rather than letting "expand and never contract" become "accumulate dead fields forever." Carrying a deprecated money field forever is a defensible deliberate choice; carrying fifty of them because nobody ever cleaned up is just payload rot.

## Schema-registry compatibility modes

For event-driven and streaming APIs (Kafka topics, Avro/Protobuf messages), the compatibility rules above get *enforced* by a schema registry, and it's worth knowing the modes because they encode the exact same asymmetries in configuration. A schema registry (the Confluent Schema Registry is the canonical example) stores the schema for each topic and rejects a producer's attempt to register a new schema version that would violate the topic's configured compatibility mode.

![A matrix of schema registry compatibility modes, backward forward and full, showing whether adding an optional field and removing a field are allowed and which side readers or writers must upgrade first](/imgs/blogs/schema-evolution-adding-removing-renaming-fields-safely-7.png)

The modes map directly to the consumer/producer upgrade order:

| Mode | What it allows | Who upgrades first |
|---|---|---|
| **BACKWARD** | New schema can read data written by the *old* schema. Add optional fields, remove fields (with defaults). | **Consumers** upgrade first |
| **FORWARD** | Old schema can read data written by the *new* schema. Add fields, can't remove required ones. | **Producers** upgrade first |
| **FULL** | Both directions — the safe intersection. | Either order |
| **NONE** | No checks (don't, unless you have another guarantee). | — |

In Avro specifically, this is why **defaults are mandatory for safe evolution**: a field added with a default can be read by old consumers (they get the default), and a field removed can still be read in old data (the new reader uses the default for the now-absent field). Avro's whole compatibility story rests on the same idea as the JSON "optional with a default" rule — the default is what lets a reader cope with a field the writer didn't send, or vice versa. The registry just turns the rule into a gate: try to register an incompatible schema and the producer's `register()` call fails before a single message is published. It is the schema diff linter, promoted to a runtime guard on the message bus.

## Case studies

**Protobuf field numbers and `reserved`.** The Protobuf language guide is explicit and unambiguous: when you delete a field, reserve its number (and name) to prevent reuse, because reusing a number causes old and new clients to disagree about which field a number identifies — leading to "data corruption" and "privacy bugs," in the guide's own framing. The format's design — identity by number, not name — is *why* renaming is free and *why* number-reuse is the cardinal sin. This is the single most important accurate fact about Protobuf evolution: the wire forgives almost everything except reusing a number.

**Stripe's additive evolution.** Stripe is the reference example of evolving a public payments API without versioned URIs for the day-to-day. They treat adding new fields and new (well-handled) enum values as **non-breaking** and ship them continuously, and they tell integrators in their documentation that their code must tolerate new fields and unrecognized enum values without breaking. When they make a genuinely breaking change, they pin it to a dated API version, and existing integrations stay on their pinned version until they opt in. The practical takeaway is exactly this post's thesis: design so that the *overwhelming majority* of changes are additive (and therefore free), and reserve the heavy machinery of versioning for the rare genuine break.

**Avro and schema-registry compatibility modes.** The Confluent Schema Registry's compatibility modes (BACKWARD, FORWARD, FULL, and their transitive variants) operationalize the compatibility rules for event streams, and Avro's reliance on field defaults is the canonical demonstration that "optional with a default" is the universal safe-evolution primitive across formats — JSON, Avro, Protobuf all rest on it. The accurate point: the same asymmetry (add-with-default is safe, remove-required is not) shows up identically whether the contract is an HTTP response, an Avro record on a topic, or a Protobuf message, because it falls out of the reader/writer relationship, not the format.

## When to reach for this (and when not to)

Expand-contract is the default for any field-level schema change to a contract with consumers you don't control. But a few honest "when not to":

- **When the API has exactly one consumer that you control and deploy atomically with the producer** — say an internal BFF for a single mobile app you ship together — you can sometimes change a field in place, because there is no skew window: the producer and the single consumer deploy as one unit. Even then, the moment a *second* consumer appears, or deploys drift apart, you're back to expand-contract. Treat the atomic-deploy case as a rare exception, not a license.
- **When you can stay additive, do — don't reach for a new version.** Cutting a `/v2` to avoid an expand-contract is over-engineering: you've fragmented your surface, doubled your docs and tests, and stranded the partners who never migrate off `/v1`. Version only when additive evolution would produce a payload genuinely worse than a clean break (the pervasive restructure).
- **When the change is a true removal of a capability** — not a rename or a retype, but "this thing no longer exists" — and a large fraction of clients depend on it, sometimes the honest move is to *not* remove it, ever. Carrying a deprecated field forever is cheap; breaking a thousand integrations is not. The contract step of expand-contract is optional; "expand and never contract" is a valid permanent state.
- **When you're tempted to retype a money or identity field in place**, never. Add a new field of the new type. Money and IDs are the fields where a silent retype does the most invisible damage, and they are exactly the fields people are most tempted to "just fix the type on."

## Key takeaways

- **The one rule: you can always add an optional field; you can never silently remove or retype one.** Everything else is machinery for obeying it.
- **Every field has a lifecycle — introduce, in use, deprecate, remove** — and skipping the deprecate stage (the shortcut from in-use straight to remove) is the single most common way schema changes break clients.
- **Every change is an expand-contract**: add the new thing alongside the old, run both in parallel through a deprecation window, retire the old only after telemetry proves it's safe. The contract step is optional; "expand and never contract" is fine.
- **A rename is add-new + dual-write + deprecate-old + remove-old.** There is no in-place rename; an in-place rename is an instant removal-without-a-window and breaks every reader.
- **A retype is a new field of the new type, never a reinterpretation of the old.** The `amount` cents-to-string change becomes a new `amount_decimal` field — reinterpreting `amount` in place silently computes wrong money.
- **Adding an enum value is usually safe but requires a contract clause** telling clients to default-handle unknown values; removing a value is breaking — reserve it instead.
- **Optional→required on a request is breaking; required→optional is safe.** Tightening a constraint on the side the peer controls breaks; relaxing is safe.
- **Protobuf is forgiving because identity is the field number, not the name** — renames are free, but reusing a removed number silently corrupts data, so always `reserved` removed numbers and names.
- **The API schema is not the database schema.** Put a serializer between them so each runs its own expand-contract on its own clock and a column rename never crashes a client.
- **Verify before you ship**: schema-diff linters at PR time, consumer-driven contract tests in CI, telemetry and canaries in production.

This post is the field-level companion to the principles in [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change), the versioning trade-offs in [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning), and the verification machinery in [contract testing](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs). It all serves the spine of the series — an API is a contract and a product, not a function call — which starts at [what is an API](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and lands at [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2). The whole question every schema change must answer is the question the wallet-app rename failed to ask: *can I change this without breaking a caller I will never meet?*

## Further reading

- [Protocol Buffers Language Guide (proto3) — Updating a Message Type / Reserved Fields](https://protobuf.dev/programming-guides/proto3/) — the authoritative rules on field numbers, what's wire-compatible, and why you must reserve removed fields.
- [OpenAPI Specification 3.1](https://spec.openapis.org/oas/v3.1.0) — how `required`, `nullable`/type unions, `enum`, and `deprecated: true` are expressed; the basis for schema diffing.
- [Confluent Schema Registry — Schema Evolution and Compatibility](https://docs.confluent.io/platform/current/schema-registry/fundamentals/avro.html) — BACKWARD/FORWARD/FULL modes and Avro defaults as the safe-evolution primitive.
- [Stripe API — Versioning and Upgrades](https://stripe.com/docs/upgrades) — additive evolution in practice and the "your code must tolerate new fields and enum values" contract.
- [Expand and contract pattern (parallel change) — Martin Fowler](https://martinfowler.com/bliki/ParallelChange.html) — the original framing of expand-contract for safe schema change.
- [RFC 9457 — Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457) — the error envelope you return when a request violates a tightened constraint during a migration.
- Within this series: [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change), [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning), [contract testing](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs), and out to [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) for the database track underneath.

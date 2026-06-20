---
title: "Deprecation and Sunset: Retiring an API Humanely"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "You shipped a contract and now you must retire part of it without betraying the callers who depend on it — learn the Deprecation and Sunset headers, the human comms, usage telemetry, the brownout technique, and a 410 Gone that points the way home."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "deprecation",
    "sunset",
    "versioning",
    "rfc-8594",
    "rfc-9745",
    "developer-experience",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-1.png"
---

A staff engineer at a fictional commerce platform we will call Northwind once told me the most expensive line of code she ever wrote was a single deletion. She removed an endpoint, `/v1/charges`, that her telemetry said handled "almost no traffic." The deploy went out on a Friday. By Monday, a regional grocery chain's overnight settlement batch — which ran once a week, on Sunday night, and so had not shown up in her Friday dashboard — had failed silently, retried against a `404`, decided the charges had never been created, and re-submitted three days of transactions. Customers were double-charged. The grocery chain's finance team spent a week reconciling. Northwind's account team spent a quarter rebuilding trust. The endpoint really did handle almost no traffic. It just handled the wrong kind of "almost no."

That is the trap at the heart of this post. You designed a contract. You shipped it. Real systems you will never see now depend on the exact shape of it — a field name, a status code, a URL path. And the moment your product evolves, you will want to retire some of that surface: a field that no longer means what it used to, an endpoint that a better one replaces, an entire version you would love to stop maintaining. **Retirement is part of the contract, not an exception to it.** An API is a contract and a product, not a function call — and the way you take something away from your callers is as much a part of that product as the way you gave it to them.

This post is about doing that retirement *humanely*: with enough warning, enough signal, and enough measurement that nobody gets double-charged on a Monday morning. We will separate three words that get used loosely — **deprecation** (it still works; don't build anything new on it), **sunset** (the date it stops working), and **removal** (it's gone). We will wire up the machine-readable signals every modern HTTP client can read: the `Deprecation` header (RFC 9745), the `Sunset` header (RFC 8594), and `Link` relations that point at your migration guide. We will pair those with the human signals — changelog, email, dashboard banner, and direct outreach to your biggest consumers — because a header in a log nobody reads is not communication. We will measure usage before pulling the trigger, because you cannot sunset what you cannot count. We will size the migration window honestly. We will use the **brownout** technique to flush out the silent stragglers like that Sunday-night batch *before* they break in production. And when the day finally comes, we will return a `410 Gone` with a `problem+json` body that points the caller home, not a bare `404` that looks like a bug they introduced. The figure below is the whole arc on one line.

![a timeline showing an endpoint moving from generally available to deprecated to a sunset date to a brownout phase to a final 410 Gone removal](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-1.png)

Throughout, the running example is the same Payments and Orders API we have used across this series. Two concrete retirements anchor everything: we will **deprecate a `legacy_status` field** on the payment resource (it still works, but it lies a little, and we want callers off it), and we will **sunset `/v1/charges` in favor of `/v1/payment_intents`** (a whole endpoint, replaced by a better-shaped one). By the end you will be able to plan and execute a retirement that your callers experience as a courteous, well-lit off-ramp rather than a cliff in the dark. This is a sibling of [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning) and the natural consequence of [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change): you version so that you *can* change, and deprecation is how you actually retire what the new version replaces. For the platform-scale view of evolving a schema across many services, we lean on [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale); here we own the wire contract and the caller's experience of being asked to move.

## 1. Three words that are not synonyms: deprecation, sunset, removal

If you take one thing from this post, take the vocabulary. The single most common cause of a botched retirement is a team using "deprecated" to mean three different things on three different days, so that callers genuinely do not know whether they have a year to migrate or a week. Pin the words down.

**Deprecation** is a *statement of intent*, not a change in behavior. A deprecated endpoint or field **still works exactly as it always did.** Nothing about the response changes except that you have now told the world: *this is going away; do not build anything new on it; start planning your move.* Deprecation is reversible in principle — you can un-deprecate something if you change your mind — and it imposes no immediate cost on the caller. The word comes from the Latin *deprecari*, "to pray against"; you are not removing the thing, you are praying people stop using it. The status code is unchanged: a deprecated endpoint that returned `200 OK` yesterday returns `200 OK` today.

**Sunset** is a *date*. It is the moment the thing stops working. RFC 8594 — titled, simply, "The Sunset HTTP Header Field" — defines `Sunset` as an HTTP-date marking "the point in time after which the resource will no longer be available." Before the sunset date, the resource works. On or after it, the resource may return an error or vanish. A sunset is a *commitment*: once you publish a date, callers plan around it, and moving it earlier is a breach of trust as serious as removing something without notice. (Moving it *later* is fine and often kind.) Deprecation without a sunset date is a vibe; deprecation *with* a sunset date is a contract clause.

**Removal** is the *act*. The thing is gone. The correct HTTP status for a resource that used to exist and has been intentionally retired is **`410 Gone`**, not `404 Not Found`. The distinction is semantic and it matters enormously for debugging, which we will return to in detail: `404` means "I have no idea what you're asking for" (maybe you typed it wrong), while `410` means "this existed, it is intentionally gone, and it is not coming back." We will make our `410` carry a `problem+json` body that names the replacement.

These three stages are sequential and each is announced before it arrives. You do not sunset something that was never deprecated; you do not remove something whose sunset date has not passed. The table below is the reference you should keep next to your retirement runbook.

| Stage | Does it still work? | Status code | What the caller may assume | Primary signal |
| --- | --- | --- | --- | --- |
| **Deprecated** | Yes, fully and unchanged | `200`/`201`/etc. as before | "Safe to use today, but plan to move" | `Deprecation` header |
| **Sunset** | Yes, until the published date | `200` now, `410` after the date | "I must migrate before this date" | `Sunset` header (a date) |
| **Removed** | No | `410 Gone` (not `404`) | "Stop calling this; here is the replacement" | `410` + `problem+json` |

![a comparison matrix of the deprecated sunset and removed stages across whether each still works the status code returned what the caller may assume and the primary signal](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-2.png)

#### Worked example: classifying our two retirements

Apply the vocabulary to the running example so the terms land.

The `legacy_status` field on the payment resource is, today, **deprecated.** We are adding a clearer `status` field with a richer state machine (`requires_action`, `processing`, `succeeded`, `canceled`) and `legacy_status` (`pending`, `paid`, `failed`) only approximates it. We will tell callers: *`legacy_status` is deprecated; read `status` instead.* It still appears in every response, with the same values it always had. There may not even be a sunset date yet — a deprecated field with no sunset is a perfectly valid, stable state we can hold for a long time. The cost of carrying one extra field is small; the cost of removing it abruptly is not.

The `/v1/charges` endpoint is on a more aggressive track. We have shipped `/v1/payment_intents`, which handles the same job with a better-shaped contract (it models the multi-step authentication that `charges` could not). We will **deprecate** `/v1/charges` immediately (it still works), publish a **sunset** date eighteen months out, run a **brownout** in the final weeks, and then **remove** it — at which point it returns `410 Gone`. Same vocabulary, very different timelines, because a whole endpoint with thousands of integrators is a heavier thing to retire than one redundant field.

Notice what the vocabulary buys us: precision in communication. When we email callers, "deprecated" and "sunset on 2027-12-15" mean specific, distinct things, and a caller reading them knows exactly how much runway they have.

## 2. Why removal is a breaking change, and why it is sometimes the only honest move

It is worth grounding this in the compatibility rules from earlier in the series, because deprecation only makes sense against that backdrop. The robustness principle — be conservative in what you send, liberal in what you accept — and its corollary, the tolerant reader, tell us which changes are safe. Adding an *optional* response field is non-breaking: a well-written client ignores fields it does not recognize. Adding a *required request* field is breaking: existing callers who do not send it now fail. **Removing** a response field or an endpoint is unambiguously breaking, because some caller's code reads that field or calls that path, and after removal that code gets `undefined`, a `KeyError`, or a `404`.

So why remove anything at all, if it breaks callers by definition? Because the alternative — never removing anything — is its own slow disaster. Every field you keep is a field you must keep *working*, *documented*, *tested*, and *secured* forever. The `legacy_status` field is not free: it is a second source of truth that can drift from `status`, a thing every new engineer must learn is a trap, an attack surface, a row in the contract test, a line in the SDK. Multiply that across a decade of "we'll just keep it for compatibility" decisions and you get the API equivalent of a house where every room is full of boxes nobody will open and nobody can throw away. The honest move is not to never break callers; it is to **break them on a schedule they agreed to, with a path out, having measured that almost nobody is still standing where the floor is about to disappear.**

This reframes the entire problem. Deprecation is not the opposite of compatibility; it is the *mechanism by which you eventually stop paying for compatibility you no longer need.* You buy the right to remove something by giving notice, providing a migration path, and measuring the blast radius. The notice and the path and the measurement are the price. Skipping them is theft — you take the caller's working integration without paying for it — and the bill always arrives, usually as a Monday-morning page.

There is a quantitative way to feel this. Suppose you maintain an endpoint with $N$ active integrators and each one's migration to the replacement costs them, on average, $c$ engineer-hours. The total switching cost you are about to impose on the ecosystem is roughly $N \cdot c$. Your job in retirement is to (1) make $c$ as small as you honestly can, by shipping a good replacement and a clear migration guide, and (2) spread the work over a window long enough that no single team is forced to drop everything. If $N$ is three internal teams and $c$ is two hours, you can move in a sprint. If $N$ is four thousand external integrators and $c$ is forty hours each, you are committing the ecosystem to roughly 160,000 engineer-hours of work, and a six-week window is not a plan, it is an ambush. The window must be proportional to $N \cdot c$. We will make that concrete in section 8.

## 3. The machine-readable signals: Deprecation, Sunset, and Link

A caller cannot act on a deprecation they never learn about, and the most reliable way to reach a *machine* is to put the signal *in the response the machine already reads.* HTTP gives us three standardized headers for exactly this, and a fourth, older one worth knowing about.

### The `Deprecation` header (RFC 9745)

RFC 9745, "The Deprecation HTTP Response Header Field," standardizes a header that announces a resource is deprecated. It takes one of two forms. The simplest is the boolean form, which uses the structured-fields boolean syntax (`?1` for true):

```http
Deprecation: ?1
```

That says, plainly, "this resource is deprecated, as of now." The more useful form carries a *date* — the moment deprecation took effect — as a structured-fields Date (an `@` sign followed by a Unix timestamp in seconds):

```http
Deprecation: @1734220800
```

`@1734220800` is `2024-12-15T00:00:00Z`. A client library can parse that and know not just *that* the resource is deprecated but *since when*, which is useful for "deprecated more than 90 days ago, escalate" automation. Note the distinction the RFC is careful about: `Deprecation` marks the date deprecation *began*, while `Sunset` marks the date the resource will *stop working*. They are different dates and they answer different questions ("how long has this been deprecated?" versus "how long do I have left?").

### The `Sunset` header (RFC 8594)

RFC 8594 defines `Sunset` as an HTTP-date (the same date format used by `Date` and `Expires`) indicating when the resource is expected to become unavailable:

```http
Sunset: Wed, 15 Dec 2027 00:00:00 GMT
```

That is the commitment: after that instant, `/v1/charges` may return `410`. A monitoring system can watch for `Sunset` headers across all the APIs a company consumes and raise a ticket as the date approaches — which is exactly the kind of automation that turns "we forgot to migrate" into "Jira told us in October."

### The `Link` header (RFC 8288) with deprecation and sunset relations

A date and a boolean are not enough. The caller needs to know *where to go.* RFC 8288, "Web Linking," lets you attach typed links to a response via the `Link` header. There are registered relation types for exactly this: `deprecation` (a link to documentation describing the deprecation) and `sunset` (a link to documentation about the sunset). Use them to point at your migration guide:

```http
Link: <https://docs.northwind.example/migrations/charges-to-payment-intents>; rel="deprecation"; type="text/html",
      <https://docs.northwind.example/sunset-policy>; rel="sunset"; type="text/html"
```

Now a developer who notices the `Deprecation` header in their logs can follow the link to the exact guide that tells them what to do. The machine signal and the human documentation are connected by a clickable URL, which is the whole point of hypermedia.

### The older `Warning` header and custom headers

Before RFC 9745, teams hand-rolled this with the `Warning` header (RFC 7234) or custom headers like `X-API-Deprecated`. The `Warning` header used a numeric warn-code plus agent and text:

```http
Warning: 299 - "Deprecated API: /v1/charges is deprecated; migrate to /v1/payment_intents by 2027-12-15"
```

The `299` warn-code is "Miscellaneous persistent warning" — the generic, human-readable bucket. This still works and is widely understood, but `Warning` was *deprecated itself* (the irony is real) in the HTTP-caching revision and is being phased out. Prefer the standardized `Deprecation`/`Sunset`/`Link` trio for new work, and treat `Warning` and custom `X-` headers as legacy you may still emit alongside for clients that only know how to grep for them. The cost of emitting all of them is a few hundred bytes per response; the benefit is that you reach clients on whatever they happen to parse.

#### Worked example: a deprecated response carrying all the signals

Here is `GET /v1/charges/ch_1a2b3c` during the deprecation-and-sunset window. The body is unchanged — that is the whole promise of deprecation — but the headers now carry the full machine-readable story.

```http
GET /v1/charges/ch_1a2b3c HTTP/1.1
Host: api.northwind.example
Authorization: Bearer <token>
Accept: application/json
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
Deprecation: @1734220800
Sunset: Wed, 15 Dec 2027 00:00:00 GMT
Link: <https://docs.northwind.example/migrations/charges-to-payment-intents>; rel="deprecation",
      <https://docs.northwind.example/sunset-policy>; rel="sunset"
Cache-Control: no-store

{
  "id": "ch_1a2b3c",
  "object": "charge",
  "amount": 4999,
  "currency": "usd",
  "status": "succeeded",
  "legacy_status": "paid",
  "created": 1734220900
}
```

Read it the way a client would. The status is `200 OK`: nothing is broken, the charge data is all there. The `Deprecation` header says this resource has been deprecated since `2024-12-15`. The `Sunset` header says it will stop working on `2027-12-15` — almost three years of runway. The `Link` headers point at the migration guide and the sunset policy. A `\$49.99` charge (`amount` is in the smallest currency unit, so `4999` cents) is returned exactly as before. And notice `legacy_status: "paid"` sitting right next to `status: "succeeded"` — that field is *also* deprecated, but at the field level, which we will signal differently in section 6 because a header cannot point at one field inside a body. This single response is, by itself, a humane deprecation: the caller gets their data, learns the resource is going away, learns exactly when, and gets a link to the fix.

## 4. The human signals: changelog, email, banner, and direct outreach

Here is the uncomfortable truth about all those beautiful headers: **most callers will never see them until something breaks.** A header lives in a response that flows into a log that nobody reads until there is an incident. If your entire deprecation strategy is "we set the `Deprecation` header," you have technically informed your callers in the same way that burying a notice on page 94 of a terms-of-service document technically informs a user. The machine signal is necessary and it is not sufficient. You must *also* reach the humans who own the code, through channels they actually read, at moments when they can act.

There is a layered set of channels, and a humane retirement uses several of them at once, because each reaches a different audience at a different moment. The figure stacks them from the machine signal at the bottom up to the most personal, highest-effort channel at the top.

![a vertical stack of deprecation signal channels from the machine readable headers at the base up through the changelog email dashboard banner and direct outreach to top consumers](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-3.png)

**The changelog** is the public, permanent record. Every deprecation gets an entry with the date deprecated, the sunset date, the replacement, and a link to the migration guide. The changelog is what a new integrator reads before they build, so it is how you stop people from adopting something you are about to retire. If your changelog is an RSS or Atom feed, sophisticated consumers can subscribe and get a machine-readable notice — a nice bridge between human and machine channels.

**Email** is the push channel for known integrators. If callers authenticate with API keys tied to an account, you know who they are and you can email the technical contact directly: "On 2024-12-15 we deprecated `/v1/charges`. It will sunset on 2027-12-15. Here is the migration guide. Your account made 1.2 million calls to this endpoint last month." That last sentence — *the specific, personalized usage number* — is what turns an ignorable broadcast into an action item. Generic "we deprecated something" emails get filtered; "you specifically are calling this 1.2 million times a month and it is going away" emails get forwarded to the on-call.

**The dashboard banner** reaches the product owner who logs into your developer console but never reads email. A persistent, dismissible-but-recurring banner — "You are using 1 deprecated endpoint. Sunset: Dec 2027. View migration guide." — catches the human at the moment they are already thinking about your platform. Tie it to the same telemetry as the email so it only shows to accounts that actually call the deprecated surface.

**Direct outreach** is the white-glove channel for your biggest consumers. This is a human from your team — a developer advocate, a solutions engineer, an account manager — reaching out one-to-one to the top ten callers, offering to walk them through the migration, answer questions, and crucially *commit to a date together.* This is expensive and you cannot do it for four thousand integrators, which is exactly why you measure usage first: the top ten callers are often 80 to 95 percent of the traffic (a power-law distribution that holds for most APIs), so a handful of personal conversations de-risks the overwhelming majority of the volume. The grocery chain in the opening story is precisely the kind of consumer who should have gotten a phone call, not a header.

| Channel | Audience it reaches | When it fires | Action it drives |
| --- | --- | --- | --- |
| `Deprecation`/`Sunset` headers | Client *code* and tooling | Every response, continuously | Flags/logs every call to the old surface |
| Changelog + email | Integrators and tech contacts | At announce, with usage data | Triggers the migration to be planned |
| Dashboard banner | The product owner who logs in | On next console login | Gets the work assigned internally |
| Direct outreach | The top ~10 callers by volume | Early, one-to-one | Secures a committed migration date |
| Brownout `503`s | Silent stragglers you missed | In the final weeks | Surfaces hidden hard dependencies |

![a matrix mapping each signal channel to the audience it reaches when it fires and the action it drives from headers through to the brownout](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-5.png)

The principle threading all five rows: **redundancy across channels is not waste, it is reliability.** Any one channel reaches some fraction of your callers. The header reaches automated tooling. The email reaches the inbox-driven. The banner reaches the console-driven. Outreach reaches the relationship-driven. The brownout reaches the ones who ignored everything else. Stack them and the fraction who learn before it hurts approaches one. Rely on a single channel and you are guaranteed to surprise somebody — and the somebody you surprise is, by the universal law of these things, your most important customer running their most important batch on the one night your dashboard wasn't looking.

## 5. Measuring usage: you cannot sunset what you cannot count

Everything above assumes you *know* who calls the deprecated surface and how much. If you do not, you are flying blind, and the Northwind story is your future. **The first concrete step of any retirement is not announcing it — it is instrumenting it.** Before you deprecate anything, you must be able to answer, for the endpoint or field in question: how many calls per day, trending which way, from which API keys, from which client versions, and — the question that gets forgotten — on what *cadence*, because a weekly batch is invisible to a daily dashboard.

The telemetry you need is **per-version and per-endpoint**, attributed to the caller. At minimum, every request to a deprecation candidate should emit a structured log line or a metric with the dimensions you will slice on:

```json
{
  "ts": "2026-06-20T08:14:02Z",
  "endpoint": "/v1/charges",
  "method": "POST",
  "api_version": "v1",
  "account_id": "acct_grocerco",
  "client_user_agent": "northwind-python/2.3.1",
  "deprecated": true,
  "status": 201
}
```

Aggregate that and you get the picture that lets you act responsibly: a time series of calls per day to `/v1/charges`, broken down by `account_id`. From it you can build the email list (everyone with nonzero traffic), the outreach list (the top ten by volume), and — critically — the *go/no-go* signal for removal (has the curve actually reached near-zero, or is there still a stubborn floor of traffic from someone who never migrated?).

A few measurement traps worth naming, because each one has stranded a real removal:

- **The weekly-batch blind spot.** A dashboard showing "last 24 hours" will report zero traffic for an endpoint called only by a Sunday-night settlement job. Always look at usage over a window at least as long as your slowest caller's cadence — a full month, ideally a quarter — before declaring something unused. This is the exact mistake from the opening.
- **Attribution to a shared key.** If many of your customers funnel through a single integration partner using one API key, your "top caller" is the partner, and the long tail of *their* customers is invisible to you. You may need the partner's help to reach the real stragglers, which is another reason for direct outreach.
- **Health checks and synthetic traffic.** Your own monitoring may hit the endpoint, inflating "usage" with calls that do not represent a real dependency. Tag and exclude internal/synthetic traffic so you are measuring real callers.
- **Caching upstream of you.** If a CDN or gateway caches the endpoint's responses, your origin sees fewer calls than really happen. Measure at the edge where the real client requests land, not only at the origin.

The decision to actually *remove* is gated on this measurement, and it is a loop, not a single check. The flow below is the gate: you read the telemetry, and only if usage is at or near zero do you proceed; otherwise you extend the window, do more outreach, run a brownout to flush stragglers, and *re-measure* before you ever return a `410`.

![a directed acyclic flow showing telemetry feeding a usage check that branches to either extend the window or run a brownout then reverify near zero usage before the final removal returns a 410](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-6.png)

The principle is blunt and worth stating as a rule: **measurement is a precondition, not a nicety.** "I think nobody uses this" is not a basis for removal; "the curve has been at zero across a full quarter, and the last nonzero caller confirmed their migration last Tuesday" is. The cost of the instrumentation is a structured log line per request. The cost of skipping it is a double-charged grocery chain. For the metrics and tracing machinery that produces this telemetry in the first place, see [observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos); deprecation is one of the highest-value uses of the per-endpoint, per-version dimensions that post tells you to emit.

## 6. Deprecating a single field versus a whole endpoint

Headers are great at deprecating a *resource* (an endpoint), because a header naturally scopes to the whole response. They are awkward at deprecating one *field inside* a body, because there is no standard "this one field is deprecated" header that points at a JSON path. The `legacy_status` retirement is exactly this case, and it deserves its own treatment.

When you deprecate a field, the response still returns it (removing it now would be the breaking change you are trying to avoid). What changes is everything *around* it. You document the deprecation prominently: in the field's description in your OpenAPI spec, in the reference docs, in the changelog. In OpenAPI 3.1, a schema property carries a `deprecated: true` flag, and good documentation renderers (and SDK generators) surface it as a strikethrough or a warning badge:

```yaml
components:
  schemas:
    Payment:
      type: object
      properties:
        status:
          type: string
          enum: [requires_action, processing, succeeded, canceled]
          description: The current state of the payment.
        legacy_status:
          type: string
          enum: [pending, paid, failed]
          deprecated: true
          description: >-
            Deprecated. Coarse legacy status that approximates `status`.
            Read `status` instead. Scheduled for removal in API v2.
            See https://docs.northwind.example/migrations/legacy-status
```

Now the field is marked deprecated in the one place that generates the SDK, the docs, and the type definitions, so every downstream artifact inherits the warning. A developer who pulls the generated client gets `legacy_status` annotated as deprecated by their IDE — the warning reaches them in the editor, before they ship code that depends on it. That is the field-level equivalent of the `Deprecation` header: the signal lands where the developer is already working.

You can still emit a *response-level* hint when a caller's request *would only make sense if they care about the field* — for example, if they explicitly requested it via a sparse-fieldset parameter like `?fields=legacy_status`. In that narrow case a `Deprecation` header on the response is honest, because the request itself names the deprecated thing. For the general case, where `legacy_status` just rides along in every payment body, the field-level documentation flag plus the changelog is the right signal, and you reserve the response headers for whole-resource deprecation.

The other field-level subtlety is *behavioral* deprecation. Sometimes a field is not going away but its *meaning* is changing. `legacy_status` returns `paid` for a payment that, under the new model, is in the `succeeded` *or* the partially-refunded state — the old field cannot express the distinction. You cannot signal that with a header at all; it is purely a documentation-and-migration-guide concern. Spell out in the migration guide exactly how each old value maps to the new state machine, including the cases where the mapping is lossy, because a caller who blindly maps `paid → succeeded` will mis-handle the partial-refund case. The lossy-mapping table *is* the migration guide for a field.

#### Worked example: the field migration guide as a mapping table

This is the heart of the `legacy_status` migration guide — the explicit, honest mapping from the deprecated field to its replacement, including where it is lossy.

| `legacy_status` (deprecated) | `status` (read this) | Notes on the mapping |
| --- | --- | --- |
| `pending` | `requires_action` *or* `processing` | Lossy: old field could not distinguish "needs the customer to authenticate" from "we are working on it." |
| `paid` | `succeeded` | Clean, but also returned for fully-captured payments that were later partially refunded — check `amount_refunded`. |
| `failed` | `canceled` | The new model also has explicit decline reasons in `last_payment_error`; the old field had none. |

A caller reading this knows the safe migration is not a one-line rename. It is: read `status`, and for the cases where the old field was lossy (the `pending` split, the refund nuance), read the additional fields the new model exposes. The migration guide that hides the lossy cases is worse than no guide, because it invites a confident wrong migration. The one above is humane precisely because it admits where the old field lied.

## 7. The brownout: flushing out the stragglers before sunset

You have deprecated. You have a sunset date. You have emailed, banner-ed, and called your top ten. Your telemetry shows traffic dropping — but there is a stubborn floor. A trickle of calls keeps coming from accounts that never responded, never migrated, and never will until something forces them. These are the silent stragglers, and they are the Sunday-night batch waiting to break. How do you find them and force their hand *before* the hard cutover, while you can still help?

The answer is the **brownout**: in the days or weeks before the sunset date, you intermittently make the deprecated endpoint fail — returning `503 Service Unavailable` or `410 Gone` for a scheduled fraction of requests — and then restore it. The term borrows from electrical engineering, where a brownout is a partial, intentional voltage reduction short of a full blackout. The effect on your callers is precise and deliberate: a caller with a robust integration that has already migrated notices nothing (they are not calling the endpoint). A straggler who is still calling it gets an *intermittent* failure — enough to trip their alerts, fill their error logs, and surface the dependency to a human on their side, while the endpoint *still works most of the time* so it is not a full outage. You have converted a future hard break into a present, survivable warning that the straggler cannot ignore.

A brownout schedule escalates so that the signal grows as the date nears. A typical shape for `/v1/charges` in its final month:

```http
GET /v1/charges/ch_1a2b3c HTTP/1.1
Host: api.northwind.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/problem+json
Retry-After: 3600
Sunset: Wed, 15 Dec 2027 00:00:00 GMT
Link: <https://docs.northwind.example/migrations/charges-to-payment-intents>; rel="sunset"

{
  "type": "https://docs.northwind.example/errors/sunset-brownout",
  "title": "Scheduled deprecation brownout",
  "status": 503,
  "detail": "This endpoint is in a scheduled brownout ahead of its sunset on 2027-12-15. It will be removed permanently. Migrate to POST /v1/payment_intents.",
  "instance": "/v1/charges/ch_1a2b3c"
}
```

Two design choices matter here. First, the brownout response carries `Retry-After` and uses `503` rather than `410` *during the brownout phase* — `503` semantically means "temporarily unavailable," which is true (it works again in an hour), and `Retry-After` tells a well-behaved client to back off rather than hammer you. A naively retrying client will succeed on its next attempt, which is intentional: we are warning, not yet removing. Second, the body is already a `problem+json` that names the replacement, so even the brownout failure is *educational* — the error itself tells the straggler exactly what to do.

The brownout schedule should be published in advance (in the migration guide and the changelog) so it is not a surprise, and it should escalate predictably. A reasonable ramp for the last month before a sunset:

- **T-minus 4 weeks:** brown out 5 percent of requests, during business hours only, for one hour.
- **T-minus 2 weeks:** 25 percent, for a few hours.
- **T-minus 1 week:** a full 24-hour brownout (the "dress rehearsal" for removal).
- **T-minus 2 days:** another full-day brownout.
- **Sunset date:** permanent `410`.

The contrast with a hard cutover is stark, and it is the difference between surfacing a problem when you can fix it together and discovering it when it pages someone at 3 a.m.

![a before and after contrast of a hard cutover that breaks every straggler at once versus a staged brownout that returns intermittent 503s with retry after so callers fix the dependency early](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-7.png)

GitHub has used exactly this technique on its public API — running scheduled "brownouts" of deprecated functionality (for example, deprecated authentication methods and legacy endpoints), where the API is intermittently disabled for short, pre-announced windows ahead of a permanent removal, specifically to give integrators a forcing function that is loud but not yet fatal. It works because it exploits a fact about software organizations: a deprecation email goes to a queue; an intermittent production failure goes to an on-call. The brownout moves your deprecation from the first queue to the second one on the *straggler's* side, weeks before the deadline, while there is still time and still a human from your team available to help.

A caution: a brownout is a deliberate, *scheduled, communicated* partial outage, and it must be operated like one. Announce the exact schedule. Keep the windows short early and lengthen them gradually. Monitor *your own* error budget and the callers who light up so your outreach team can reach them in real time ("we see your account erroring against `/v1/charges` during today's brownout — can we help you migrate?"). A brownout that is not announced is just an outage, and a brownout you do not watch is a missed chance to make the call that saves the relationship. Done right, it is the single most effective straggler-flushing tool you have. For the retry-and-backoff behavior that makes a client survive a brownout gracefully — honoring `Retry-After`, exponential backoff with jitter — the broker-side reasoning transfers directly from [dead-letter queues and retries with exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff).

## 8. Sizing the migration window: how long is humane?

How long should the window between deprecation and sunset be? The honest answer is *it depends*, and the two things it depends on are exactly the two factors in the switching-cost argument from section 2: **how many consumers depend on you, and how costly their switch is.** The window must be proportional to $N \cdot c$ — the number of integrators times the per-integrator migration cost — because that product is the total work the ecosystem must absorb, and a window shorter than the time it takes the slowest reasonable team to do that work is an ambush.

The figure lays out the rough bands, which are not arbitrary; they reflect how the unknown-ness and the contractual weight of your callers scale.

![a grid relating the API surface from internal to partner to public against its consumer count and the corresponding humane migration window from weeks to many months](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-8.png)

**Internal APIs** — consumed by teams inside your own company, whom you can find on Slack, whose deploy schedules you know, and whom you can help directly — can move in **weeks.** You know every caller (they are the services in your own architecture), $c$ is low (your colleagues, who can prioritize the migration this sprint), and you can coordinate the cutover. Internal does not mean *no* process: you still announce, still measure, still avoid breaking someone's release. But the window is short because the unknowns are small. For the service-to-service coordination that makes this tractable in a fleet, see [the system-design view of schema evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale).

**Partner APIs** — consumed by a known set of external companies under contracts, with their own release cycles and their own change-management processes — need **three to six months,** sometimes more. You know who they are (so you can email and call), but $c$ is higher (their engineers must schedule the work against their own roadmap, possibly through a change-advisory board) and you do not control their priorities. The window must survive a partner who only does quarterly releases.

**Public APIs** — consumed by an unbounded, partly anonymous population of integrators you cannot enumerate — need the longest windows, conventionally **six to twenty-four months.** Here $N$ is large and partly unknown, $c$ varies wildly across callers, and you cannot reach the long tail directly. Major platforms publish this as policy. Google's general guidance, expressed in its API deprecation and versioning practices, is to give substantial notice and maintain deprecated functionality for a defined, generous period; for some products Google has committed publicly to a minimum of one year of support after deprecation. Stripe is the canonical example of the *long* end: it has famously kept old API versions working for many years — integrations built years earlier continue to function against the version they pinned to — because the cost to Stripe of carrying old versions is far less than the cost to its ecosystem of forced migrations. The trade-off is real and it is a *business* decision: a longer window costs you more maintenance but buys you ecosystem trust; a shorter window saves you maintenance but spends trust you may not get back.

| Factor | Pushes the window *shorter* | Pushes the window *longer* |
| --- | --- | --- |
| **Consumer count** ($N$) | Few, known callers (internal) | Many, partly anonymous (public) |
| **Switching cost** ($c$) | Trivial rename, drop-in replacement | Re-architecture, data model change |
| **Caller release cadence** | Continuous deployment | Quarterly or annual releases |
| **Reachability** | All callers contactable directly | Long anonymous tail |
| **Severity of breakage** | Read-only, easily detected | Silent data corruption (payments) | 
| **Your maintenance cost** | Endpoint is cheap to keep | Security/compliance burden to keep |

Two rules sharpen the bands. First, **the window is a floor, not a ceiling** — if your telemetry says callers are still on the old surface as the sunset approaches, you extend it (which is exactly the "extend the window" branch in the decision flow), because shipping a `410` to live traffic is choosing to break a paying customer to hit a self-imposed date. Second, **higher breakage severity demands a longer window and a louder signal.** Retiring a read-only `GET` that returns a `404`-able typo on failure is forgiving; retiring a payments-creation endpoint, where a botched migration double-charges customers, demands the conservative end of the range, mandatory direct outreach, and a brownout. The `/v1/charges` sunset is on the eighteen-month end of public precisely because it is money.

## 9. Graceful removal: the 410 Gone that points the way home

The sunset date has arrived. Telemetry is at zero. The brownout flushed the stragglers and your outreach team confirmed the last ones migrated. Now you remove the endpoint — and the single most important decision at this moment is *what status code the removed endpoint returns.* The answer is **`410 Gone`**, not `404 Not Found`, and the difference is the difference between a caller who knows what happened and a caller who files a support ticket asking why their working code suddenly broke.

Recall the semantics. `404 Not Found` means the server *has no current representation and gives no indication whether the condition is temporary or permanent* — colloquially, "I don't know what you're asking for." `410 Gone`, per RFC 9110, means the resource *"is no longer available and will not be available again"* and that *"this condition is expected to be considered permanent."* When a caller hits a removed endpoint and gets a `404`, their first hypothesis is "I have a typo, or a config error, or a routing bug" — they will burn hours debugging *their own* code before they suspect you removed something. A `410` says, unambiguously, "this is gone on purpose; stop looking for the bug in your code." And a `410` with a `problem+json` body goes one step further: it tells them *where to go.*

![a before and after contrast of a bare 404 not found with an empty body that looks like a caller bug versus a 410 gone carrying a problem json that names the replacement endpoint and migration guide](/imgs/blogs/deprecation-and-sunset-retiring-an-api-humanely-4.png)

The contrast above is the conceptual one — a bare `404` that looks like a caller's bug versus a signal-rich `410` that names the way home. The concrete wire is what makes it real, so here is the actual removal response, side by side with the lazy version teams ship by accident when an endpoint simply falls out of the router.

#### Worked example: a 410 Gone with a migration-guide problem+json

This is what `POST /v1/charges` returns on and after the sunset date, contrasted with the lazy version.

The lazy, harmful version — the endpoint is simply gone from the router, so the framework's default handler answers:

```http
POST /v1/charges HTTP/1.1
Host: api.northwind.example
Authorization: Bearer <token>
Content-Type: application/json

{"amount": 4999, "currency": "usd"}
```

```http
HTTP/1.1 404 Not Found
Content-Type: text/html
Content-Length: 0

```

That empty `404` is a cruelty. The caller's code, which worked for years, now gets nothing actionable. Their retry logic may interpret the `404` as "the charge was never created" and re-submit — the double-charge failure mode again. The humane version returns a deliberate `410` with a `problem+json` body (RFC 9457, the standard machine-readable error format) that names the replacement, the date it was removed, and the guide:

```http
POST /v1/charges HTTP/1.1
Host: api.northwind.example
Authorization: Bearer <token>
Content-Type: application/json

{"amount": 4999, "currency": "usd"}
```

```http
HTTP/1.1 410 Gone
Content-Type: application/problem+json
Link: <https://docs.northwind.example/migrations/charges-to-payment-intents>; rel="sunset"

{
  "type": "https://docs.northwind.example/errors/endpoint-removed",
  "title": "Endpoint removed",
  "status": 410,
  "detail": "POST /v1/charges was deprecated on 2024-12-15 and removed on its sunset date 2027-12-15. Use POST /v1/payment_intents, which supports the same flow plus customer authentication.",
  "instance": "/v1/charges",
  "removed_on": "2027-12-15",
  "replacement": "/v1/payment_intents",
  "migration_guide": "https://docs.northwind.example/migrations/charges-to-payment-intents"
}
```

Every member earns its place. `type` is a stable URI a client can route on — automated tooling can recognize `endpoint-removed` and surface it distinctly. `title` and `detail` are human-readable; `detail` tells the whole story in one sentence including *both* dates. `instance` echoes the path. The extension members (`removed_on`, `replacement`, `migration_guide`) are the breadcrumbs that turn the error into a self-service migration: a developer who hits this in their logs can follow `migration_guide` and fix the call without ever filing a ticket. The `Link` header carries the same destination for clients that read headers rather than bodies. This is what "graceful" means concretely — the removal still answers the caller's real question, which is not "why is this gone" but "what do I do now."

A practical operational note: keep the `410` handler in place *indefinitely* after removal. The cost of a tiny route that returns a static `410` is near zero, and it keeps paying off for years — every late straggler who finally hits it gets the migration guide instead of a mystery `404`. Removing the *endpoint logic* (the database calls, the business code, the maintenance burden) is the point of the sunset; removing the *helpful tombstone* is a small, gratuitous additional cruelty you have no reason to commit. Leave the gravestone with the forwarding address on it.

## 10. The other side of the wire: how a good client consumes these signals

So far we have stood on the *provider* side — the team retiring the surface. But the whole reason to emit standardized headers is so the *consumer* side can act on them automatically, and a humane retirement only works if at least some of your callers are good citizens about consuming the signals. It is worth crossing to the other side of the wire, because understanding what a well-behaved client does with your headers tells you why emitting them is worth the effort, and because you yourself are a consumer of other people's APIs and should run the same machinery against them.

A robust client does three things with deprecation signals, in increasing order of sophistication. The first and cheapest is to **log them.** Any HTTP client wrapper can inspect every response for a `Deprecation` or `Sunset` header and emit a structured warning the first time it sees one for a given endpoint:

```python
def check_deprecation(response, endpoint):
    if "Deprecation" in response.headers:
        sunset = response.headers.get("Sunset")
        link = response.headers.get("Link", "")
        log.warning(
            "Calling deprecated endpoint %s (sunset=%s). Migration: %s",
            endpoint, sunset or "not yet dated", link,
        )
```

That single function, wrapped around an SDK's transport layer, means the day a provider deprecates an endpoint you depend on, *your own logs* start telling you — without anyone on your team reading the provider's changelog. This is the entire payoff of standard headers over bespoke ones: the consumer can write this once and have it work against every well-behaved API they call, because the header name is the same everywhere.

The second, more proactive move is to **surface deprecations to a dashboard or alert** rather than burying them in logs. A client that increments a metric tagged by endpoint and sunset date — `api_deprecated_calls{endpoint="/v1/charges",sunset="2027-12-15"}` — lets a consuming team build an alert that fires as the sunset date approaches and they are *still calling the endpoint.* This is the consumer-side mirror of the provider's go/no-go telemetry from section 5, and it is what turns "we forgot to migrate and got broken" into "our own monitoring told us in October that we were nine weeks from a sunset on something we still call 4,000 times a day." The most disciplined platform teams treat an inbound `Sunset` header on a dependency the same way they treat a TLS certificate nearing expiry: a dated, trackable obligation with an owner and a deadline.

The third and most advanced behavior is to **honor the brownout gracefully.** When a provider returns a brownout `503` with `Retry-After`, a well-behaved client backs off for the indicated interval rather than retrying in a tight loop and turning the brownout into a self-inflicted denial of service. Crucially, a client that is *also* doing the first two things will, during the brownout, see both the intermittent `503` *and* the accumulated deprecation warnings in its logs, and a competent on-call engineer connects the two: "we are erroring against an endpoint that has been warning us it is deprecated for a year." That connection is exactly the realization the brownout is engineered to provoke. The provider's brownout and the consumer's logging are two halves of the same conversation; the brownout is the provider raising their voice precisely because the consumer was not listening to the quieter signals.

This consumer-side view also explains why the *quality* of your headers matters so much. If you emit a `Deprecation` header but no `Link`, a consumer's logger records "this is deprecated" with no idea where to go — you have generated anxiety without giving relief. If you date the `Sunset` header, a consumer can compute "days remaining" and prioritize accordingly; if you leave it boolean, they cannot. Every member of the signal you emit is a field some consumer's automation will key on, so emit the richest honest signal you can: dated `Deprecation`, dated `Sunset`, `Link` to the guide, and a `problem+json` on the eventual `410`. The reward is that the conscientious fraction of your callers — usually your most sophisticated and highest-volume integrators, the ones you least want to break — migrate themselves, on their own schedule, without ever opening a support ticket, because you gave their automation everything it needed to do the work for them.

#### Worked example: a client that fails the build on a soon-to-sunset dependency

The most aggressive (and, for a critical dependency, most responsible) consumer-side pattern is to fail CI when a dependency is within some threshold of its sunset date. A test that runs against the provider's API in a staging environment can read the `Sunset` header and assert there is enough runway left:

```python
def test_charges_endpoint_not_sunsetting_soon():
    resp = client.get("/v1/charges/ch_test")
    sunset = resp.headers.get("Sunset")
    if sunset:
        days_left = (parse_http_date(sunset) - now()).days
        assert days_left > 90, (
            f"/v1/charges sunsets in {days_left} days — migrate to "
            f"/v1/payment_intents before this test starts failing."
        )
```

This test passes silently for years, then starts failing ninety days before the sunset — turning the provider's far-off date into a *present, blocking, impossible-to-ignore* signal in the consumer's own pipeline. It is the consumer choosing to be brown-ed-out by their own CI rather than by the provider's production `503`. A platform team that wires this around its critical third-party dependencies essentially never gets surprised by a sunset, because they converted a header they might have ignored into a build break they cannot. As a provider, knowing some of your best callers do exactly this is another reason to keep your `Sunset` dates honest and stable: you are feeding a machine on the other side that will hold you to them.

## 11. Versioned deprecation: deprecate v1 when v2 is stable, not before

There is a sequencing rule that teams get wrong often enough to deserve its own section: **do not deprecate the old version until the new one is genuinely ready to carry the load.** Deprecation is a signal that says "move to the replacement." If the replacement is not stable, not feature-complete, or not battle-tested, you are telling callers to migrate *toward a moving target*, and you will either have to retract the deprecation (which destroys your credibility — a deprecation you walk back teaches callers to ignore the next one) or you will force a painful double migration as v2 itself churns under them.

The correct sequence for a version retirement is a pipeline of gates, and each gate must pass before the next:

1. **Ship v2 and stabilize it.** `/v1/payment_intents` exists, is documented, has an SDK, and — critically — has run in production long enough that you trust it. Real callers are using it successfully. Its own contract has settled (you are not still renaming fields in it).
2. **Achieve feature parity (or document the gaps).** Every legitimate use of `/v1/charges` must be expressible in `/v1/payment_intents`. If there is a capability gap, either close it or explicitly document the workaround *before* deprecating, because a caller who cannot accomplish their task in v2 cannot migrate, no matter how loud your signals.
3. **Write the migration guide.** The field-by-field, endpoint-by-endpoint mapping (like the `legacy_status` table in section 6, but for the whole endpoint) must exist before you tell anyone to migrate. "Deprecated, migration guide coming soon" is not a deprecation; it is a threat.
4. **Now deprecate v1.** Set the `Deprecation` header, publish the sunset date, send the comms. Only now, with a stable destination and a written route, is the instruction to migrate honest.

The failure mode of deprecating too early is concrete and common. A team ships v2 in beta, immediately slaps a `Deprecation` header on v1 to "encourage adoption," and then spends six months changing v2's response shape based on early feedback. Every caller who dutifully migrated early now gets broken by v2's churn — they took your deprecation seriously and were punished for it. The next time you deprecate something, those callers wait until the brownout, because experience taught them your "deprecated" does not mean "stable replacement available." Deprecation is a promise that the replacement is ready; spend that promise carelessly and it stops working.

This also clarifies *what* you deprecate. You deprecate a *coherent unit* that the new thing fully replaces — a whole version (`v1` → `v2`), a whole endpoint (`/v1/charges` → `/v1/payment_intents`), or a whole field (`legacy_status` → `status`). You do not deprecate half of a thing, leaving callers in a state where part of their integration is blessed and part is condemned with no clean way to be wholly on the supported path. The unit of deprecation matches the unit of migration: a caller should be able to do one well-defined piece of work and emerge fully on the supported surface. This connects directly to the versioning strategy you chose — if you version in the URI (`/v1`, `/v2`), the version is the natural unit of deprecation; the trade-offs of that choice are the whole subject of [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning).

## 12. Publish the deprecation policy up front: it is part of the contract

Here is the move that separates a platform from a pile of endpoints: **you publish your deprecation policy before you ever deprecate anything.** The policy is itself part of the contract — arguably the most important part, because it tells callers what they can *count on* about how you will treat them when, inevitably, you need to change. A caller integrating with your API is making a multi-year bet. The single most important thing they want to know is not "how good is the API today" but "how will you treat me when you need to break it." A published deprecation policy answers that question before they ask it, and that answer is a major factor in whether a serious integrator builds on you at all.

A good deprecation policy, published in your developer documentation, commits in writing to the things this post has walked through:

- **The minimum notice period** between deprecation and sunset, by API tier. For example: "We will provide at least 12 months' notice before sunsetting any stable public endpoint, and at least 90 days for beta endpoints." This is the number an integrator's architect needs to do their own risk assessment.
- **The signals you will emit.** "Deprecated resources return a `Deprecation` header and, once dated, a `Sunset` header, with a `Link` to the migration guide. We announce in the changelog and email account technical contacts."
- **The removal behavior.** "Removed endpoints return `410 Gone` with a `problem+json` body naming the replacement."
- **The brownout practice, if you use one.** "In the final weeks before sunset we run announced brownouts; the schedule is published in each migration guide."
- **What is and is not covered.** "This policy covers stable, generally-available endpoints. Beta and experimental endpoints, clearly marked as such, may change or be removed with shorter notice." (You need this escape hatch so you can iterate on new things without committing to a year of support for a beta you might scrap. The deal is: beta = move fast, GA = stability guaranteed.)

GitHub publishes such practices, including its use of brownouts and its `Sunset`-header conventions. Stripe's effective policy — visible in how long it keeps old versions alive — is part of why developers trust it with their revenue. Google's API design guidance (its AIP guidelines) treats deprecation and versioning as first-class, documented concerns rather than ad-hoc events. The common thread is that these are *promises made in advance*, not improvised under pressure. The policy turns every individual deprecation from a unilateral surprise into the predictable execution of a deal the caller already agreed to when they integrated. That predictability is, in the end, what "humane" reduces to: the caller is never *surprised*, because everything that happens was described in advance, signaled in multiple channels, measured before it was triggered, and routed toward a replacement when it arrived. Tie this back to the series' spine — an API is a contract and a product. The deprecation policy is the clause of the contract that governs the contract's own end, and a product that handles its own retirements gracefully is a product people trust enough to build their business on.

To bring the whole arc back to one place, walk the `/v1/charges` retirement end to end as a single timeline, because the steps only make sense as a sequence. Before anything is announced, the team instruments the endpoint and watches a full quarter of traffic, slicing by account so the weekly settlement batches show up. With the replacement `/v1/payment_intents` stable in production and the migration guide written, they flip the `Deprecation` header on, publish a sunset date eighteen months out in the `Sunset` header with a `Link` to the guide, write the changelog entry, and email every account with nonzero traffic a personalized note carrying that account's own call volume. The top ten callers — the ones who, between them, account for the overwhelming majority of the load — get a phone call and a committed migration date. Over the following year the traffic curve falls as integrators move, tracked on the same dashboard that justified the deprecation in the first place. In the final month, the announced brownout schedule ramps from a five-percent business-hours flicker to full-day dress rehearsals, and each brownout `503` carries `Retry-After` and a `problem+json` naming the replacement, so the last stragglers light up the outreach team's screen and get a helping hand while there is still time. On the sunset date, with telemetry at zero and the last caller's migration confirmed, the endpoint logic is deleted and the route is replaced by a permanent `410 Gone` whose `problem+json` body names `/v1/payment_intents`, the removal date, and the guide — a tombstone with a forwarding address that will quietly redirect any straggler who shows up years later. No customer is double-charged. No grocery chain spends a week reconciling. The retirement is, from the caller's seat, a well-lit off-ramp they were invited onto, reminded about, and gently herded down — which is the entire difference between a platform and a pile of endpoints that occasionally vanish overnight.

## Case studies: how the platforms that get retired-from for a living do it

A few accurate, named references, because the ideas above are easier to trust when you can see them in the wild.

**GitHub — `Sunset` headers and scheduled brownouts.** GitHub's REST API documents deprecation through changelog posts and, on deprecated endpoints and behaviors, emits `Sunset` headers indicating when the functionality will be removed. More distinctively, GitHub has used *brownouts* — pre-announced, temporary disablements of deprecated functionality (for example, when phasing out legacy authentication mechanisms and certain API behaviors) — precisely as a forcing function to surface integrations that had not migrated, well ahead of the permanent cutover. The brownout schedule is published in advance so that integrators see intermittent failures, check their logs, find the deprecation notice, and migrate while there is still time. This is the section-7 technique in production at a platform with an enormous, partly-anonymous integrator population.

**Stripe — long version lifetimes and pinned versions.** Stripe's approach to evolution is the long-window archetype. API changes that would break callers are released as new dated *versions*, and an account is pinned to the version it integrated against; old versions keep working for years. Stripe maintains backward compatibility aggressively, so an integration built against an old version continues to function without forced migration far longer than most APIs allow. The lesson for deprecation: Stripe rarely needs to *remove* things abruptly because its versioning strategy lets old and new coexist for a very long time, which dramatically lowers the human cost of evolution — the deprecation, when it comes, lands on callers who have had years of runway. (Stripe's well-known *idempotency-key* design is a separate but related reliability practice — see [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) — and it is exactly the mechanism that prevents the double-charge failure mode this post keeps invoking when a caller retries against a changed surface.)

**Google — deprecation and support windows as published policy.** Google's API design guidance treats deprecation and versioning as first-class, documented concerns: clients should be insulated from breaking changes by versioning, deprecation should be announced with substantial notice, and deprecated functionality should be maintained for a defined period. For several Google Cloud products, the deprecation policy commits publicly to a minimum support window — on the order of a year — after a deprecation is announced, so that customers can plan migrations against a contractual guarantee rather than a hope. The takeaway is the section-12 move: the deprecation *policy* is published up front and is itself part of what you are buying when you build on the platform.

**The standards themselves.** It is worth noting that the headers in this post are not vendor inventions but IETF standards: `Sunset` (RFC 8594), `Deprecation` (RFC 9745), and the `Link` relations via Web Linking (RFC 8288). When you emit them, you are speaking a vocabulary that monitoring tools, client libraries, and other engineers already understand — which is precisely the reason to prefer the standard headers over a bespoke `X-API-Going-Away` of your own design.

## When to reach for this (and when not to)

Deprecation-and-sunset is the right tool for retiring a *coherent, externally-depended-on* piece of contract. It is not always the right tool, and over-applying its full ceremony has costs. A few decisive calls:

- **Do** run the full ritual — headers, comms, telemetry, brownout, `410` — when retiring anything on a **public or partner** API, and especially anything touching **money, data integrity, or auth**, where a botched removal corrupts state. The `/v1/charges` sunset is the canonical case.
- **Don't** stretch a multi-month, multi-channel campaign over an **internal** endpoint that three teams you can Slack depend on. Announce it, measure it, help them move, cut over in a sprint. Ceremony proportional to blast radius; a year-long sunset for an internal RPC nobody outside the building can call is theater.
- **Don't** deprecate the old thing **before the new thing is stable** (section 11). If v2 is still churning, you are not ready to deprecate v1, no matter how much you want people off it. A retracted deprecation costs more credibility than a late one.
- **Do** prefer **not removing at all** when the cost of carrying the old surface is genuinely low and additive change covers you. A redundant *response* field that costs a few bytes and never lies can often just stay — the tolerant-reader principle means callers who do not read it are unharmed. Reserve removal for surfaces that impose real ongoing cost: a security liability, a second source of truth that drifts, a maintenance burden, a confusingly-wrong field like a `legacy_status` that misleads. Removal is a tool for reducing genuine debt, not for tidiness. If keeping it is cheap and harmless, keeping it may be the more humane choice than imposing $N \cdot c$ of migration work to satisfy your aesthetics.
- **Don't** return a bare `404` for a removed endpoint, ever. If you are going to remove something, the marginal cost of a `410` with a migration pointer over a `404` is one small route handler, and it saves every future straggler hours of misdirected debugging.
- **Don't** publish a sunset date you are not prepared to honor *or* extend honestly. A date you quietly slip teaches callers your dates are fiction; a date you blow through and enforce anyway breaks the callers who trusted you. Pick a date you can defend, and if telemetry says callers are still there, extend it loudly rather than enforce it silently.

## Key takeaways

- **Deprecation, sunset, and removal are three distinct stages, not synonyms.** Deprecation means "still works, don't build new on it." Sunset is the *date* it stops working. Removal is the act, and it returns `410 Gone`. Pin the words down or your callers won't know how much runway they have.
- **Emit the standard machine signals.** The `Deprecation` header (RFC 9745, boolean or a date), the `Sunset` header (RFC 8594, an HTTP-date), and `Link` relations (`rel="deprecation"`, `rel="sunset"`, RFC 8288) pointing at the migration guide. Add a `Warning`/custom header only as legacy belt-and-suspenders.
- **The machine signal is necessary but not sufficient.** Pair it with human channels — changelog, email with *personalized usage numbers*, dashboard banner, and direct outreach to your top callers — because the header lives in a log nobody reads until something breaks.
- **You cannot sunset what you cannot count.** Instrument per-version, per-endpoint, per-caller usage *before* deprecating. Watch over a window at least as long as your slowest caller's cadence — a weekly batch is invisible to a daily dashboard, and that blind spot is how grocery chains get double-charged.
- **Size the window to $N \cdot c$.** Internal: weeks. Partner: months. Public: 6–24 months. The window is a floor, not a ceiling — if callers remain as the date approaches, extend it; never ship a `410` to live paying traffic to hit a self-imposed deadline.
- **Use the brownout to flush stragglers.** Pre-announced, escalating, intermittent `503`/`410` in the final weeks turns a silent dependency into an on-call page on the *straggler's* side, weeks before the hard cutover, while you can still help.
- **Remove with a `410 Gone` carrying a `problem+json`** that names the replacement, the removal date, and the migration guide — never a bare `404` that looks like the caller's own bug. Leave the helpful tombstone up indefinitely.
- **Deprecate v1 only when v2 is stable**, feature-complete, and the migration guide is written. A retracted or premature deprecation teaches callers to ignore your next one.
- **Publish your deprecation policy up front.** It is part of the contract — the clause that governs how you treat callers when you must break them. Predictability is what "humane" reduces to.

## Further reading

- **RFC 8594 — The Sunset HTTP Header Field.** The standard for signaling when a resource will become unavailable. The normative definition of the `Sunset` header used throughout this post.
- **RFC 9745 — The Deprecation HTTP Response Header Field.** The standard `Deprecation` header (boolean and dated forms) for announcing a resource is deprecated.
- **RFC 8288 — Web Linking.** Defines the `Link` header and typed relations, including `deprecation` and `sunset`, used to point callers at the migration documentation.
- **RFC 9110 — HTTP Semantics.** The authoritative semantics of `410 Gone` versus `404 Not Found`, plus `Retry-After` and the status codes used during a brownout.
- **RFC 9457 — Problem Details for HTTP APIs.** The `application/problem+json` format used for the brownout `503` and the removal `410` bodies.
- **Google API Improvement Proposals (AIP) — deprecation and versioning.** Google's documented, first-class treatment of API versioning and deprecation as policy rather than ad-hoc events.
- Within this series: the intro hub [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2); the siblings [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning), [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change), and [observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos); and the platform-scale view in [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale).

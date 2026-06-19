---
title: "Error Design: A Machine-Readable, Human-Friendly Error Contract"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Errors are part of your API's contract and most of its developer experience — learn to design RFC 9457 problem+json bodies with a stable machine code, field-level validation, a trace id, an actionable message, no leaked internals, and an honest retryable-vs-terminal signal."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "error-handling",
    "problem-json",
    "rfc-9457",
    "validation",
    "rate-limiting",
    "developer-experience",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-1.png"
---

A few years ago I watched a payments integration take six weeks to ship that should have taken six days. The happy path — create an order, charge a card, return a receipt — was done by the end of the first week. The other five weeks were spent on failure. What does the client do when the card is declined? When the gateway times out? When the request is malformed? When we are rate-limited mid-batch? The vendor's API answered every one of those questions the same way: a bare HTTP `500` with an empty body, or worse, a `200 OK` wrapping `{"success": false, "message": "An error occurred. Please try again."}`. The integrating team ended up reverse-engineering failure modes by triggering them in a sandbox and grepping the response strings, building a fragile lookup table from English prose to behavior. The first time the vendor reworded one of those messages — "An error occurred" became "Something went wrong" — three retry paths silently broke, and a customer got double-charged because a declined-then-reworded error was misclassified as retryable.

That is the lesson this whole post is built on: **most of the time a developer spends integrating your API is spent handling failure, not the happy path.** Your error responses are not an afterthought bolted onto the "real" contract. They *are* the contract, for the part of the contract that matters most under pressure. A great error does two jobs at once, and you cannot skip either: it tells *code* exactly what to do (retry, fix the input, give up, ask for a different card), and it tells a *human* exactly what went wrong in language they can act on. An error that only a human can read forces machines to parse prose. An error that only a machine can read leaves the on-call engineer staring at `ERR_4471` with no idea what it means. The figure below is the entire argument in one frame — an opaque error versus an actionable one.

![a side by side comparison of an opaque 500 with an empty body against an actionable problem+json that carries a stable code, a detail message, a trace id, and a next step](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-1.png)

By the end of this post you will be able to design an error contract that holds up for years: a consistent **RFC 9457 problem+json** envelope across every endpoint; a *stable machine-readable code* that is separate from the HTTP status; field-level validation errors a form can map directly to inputs; a correlation id the caller can quote to your support team; messages that say what to *do*; a discipline that never leaks stack traces, SQL, or personal data; and an honest signal of whether an error is *retryable* or *terminal*. We will keep returning to the series' running example — a "Payments & Orders" API for a fictional commerce platform — and we will walk three failures end to end: a declined card, a validation failure, and a rate-limit error. This is part of the **["Designing APIs That Last"](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems)** series, and it composes directly with the post on **[status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx)** — the status code is the *headline*; the body is the *story*.

## 1. The principle: an error is a two-audience contract

Start from the spine of this whole series: an API is a contract you design for a caller you will never meet, on a timeline of years. When you return success, the caller's code reads the body and moves on. When you return an error, you have created a *decision point* in someone else's program — and you, the API author, are the only one who knows enough to tell them which branch to take. The principle follows directly:

> **An error response must carry enough structure for a program to decide what to do without parsing prose, and enough plain language for a human to understand what went wrong without reading your source code.**

Why must both be present? Consider who reads an error. Two distinct audiences touch every failure, and they read different fields:

- **The caller's program** runs in production, at 3 a.m., with no human watching. It needs a *stable token* it can branch on: `if error.code == "card_declined": prompt_for_new_card()`. It does not read English. It cannot tolerate the message being reworded next quarter.
- **The caller's developer** reads the error during integration and during incidents. They need a sentence that explains the failure and, critically, *what to do about it*. They also need an identifier they can paste into a support ticket so your team can find the exact request in your logs.

An error that serves only the program is hostile during debugging — `{"code": 4471}` tells the on-call engineer nothing. An error that serves only the human forces the program to do natural-language processing on `detail` strings, which is exactly the brittle reverse-engineering my opening story described. The robustness principle — *be conservative in what you send* — applies to errors with a vengeance: send a payload structured enough that a tolerant reader never has to guess.

There is a quantitative argument hiding here too. If the probability that a single integration call hits *some* error path is small — say a few percent across declines, validation, timeouts, and rate limits — the *number of distinct failure modes* a real integration must handle is large: declines alone fan out into insufficient funds, expired card, fraud hold, issuer-unreachable, and a dozen more. A developer integrating against your API must enumerate and branch on each one. If the only discriminator is a free-text `message`, the size of their branching logic is bounded below by the number of distinct strings you happen to emit, and that set changes every time a copywriter edits a message. A stable, enumerated set of machine codes turns an open-ended NLP problem into a closed `switch` statement. That is the difference between a six-day integration and a six-week one.

It is worth being precise about *why* the prose channel is so fragile, because it explains every design choice that follows. A human-readable message is, by intent, optimized for the human: it should be clear, polite, and — increasingly — localized, A/B-tested, and tone-adjusted by people who are not engineers and who have no idea anyone's code depends on the exact wording. That is healthy. Product and content teams *should* be free to improve error copy. The bug is not that messages change; the bug is that anyone built a machine dependency on something that was always going to change. So the design move is not "freeze the messages" (you cannot, and you should not want to). The design move is to give machines a *different* surface to depend on — one whose stability you can guarantee precisely because no human ever needs to edit it for clarity. The `code` exists so the `message` is free to change. Keep that inversion in mind; it is the spine of everything below.

A second framing helps before we get into the wire. An error is the API author *handing control back to the caller along with instructions*. In the success case you return data and the caller proceeds down their main path. In the failure case the caller's program forks, and the fork it takes is *your* call to make on their behalf, because you are the only party who knows what actually went wrong. A declined card and a malformed body and a rate limit and an internal crash demand four completely different responses from the caller — retry, fix-and-resend, wait-and-retry, escalate-to-a-human — and the only way the caller can take the right fork is if your error tells them which one. An error that does not distinguish these is not "less helpful"; it is *actively misleading*, because the caller will pick a fork, and without a signal they will pick the wrong one as often as the right one. Designing errors is designing the caller's failure-path control flow. Treat it with the seriousness you would treat designing your own.

### What a bad error costs, concretely

Let me make the consequence vivid before we build the good version. Here is the kind of error my opening vendor returned for a declined card:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "success": false, "message": "We could not process your payment at this time. Please try again later." }
```

Three separate lies are encoded here. First, the status is `200 OK` — so every HTTP client library, every cache, every gateway treats this as success; the caller must inspect the body to discover failure, which means error handling cannot be centralized. Second, "please try again later" *invites a retry* on a card decline, which will never succeed and may re-trigger fraud heuristics. Third, there is no machine code, so the caller's only branch key is that English sentence. When the vendor later changed "could not process your payment" to "payment was declined," the caller's substring match for "could not process" stopped firing, the decline fell through to the generic-retry branch, and the retry logic resubmitted the charge. The seesaw between a reworded message and a retry loop is how a wording change becomes a double charge. Every one of those problems is a *design* problem, and every one is fixable. We will fix them all.

It is worth dwelling on the `200`-with-a-false-body anti-pattern, because it is the most common and the most damaging, and it keeps reappearing under different names. The reason it is so destructive is that the HTTP status code is not just a number your code reads — it is a *protocol-level signal* that an enormous amount of infrastructure reads automatically and acts on without your involvement. A `200` tells a CDN it may cache the response. It tells a reverse proxy the upstream is healthy. It tells a client library's retry policy "do not retry — this succeeded." It tells a load balancer's health check "this instance is fine." It tells an observability dashboard's error-rate metric "no error here." When you return `200` with `{"success": false}`, you have lied to every one of those layers simultaneously. Your error rate looks flat while customers fail. Your cache may serve a stale failure to the next caller. Your retry middleware does the opposite of what the situation needs. The body says one thing and the envelope says another, and every piece of generic infrastructure trusts the envelope. The status code is the part of the error that the *machines you do not control* read, and getting it wrong corrupts a dozen systems at once. This is exactly the argument the **[status codes post](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx)** makes in full; here we simply inherit its conclusion — the status is the headline, and it must be true — and build the body on top of an honest status.

The opposite failure is rarer but worth naming: the API that returns the *correct* status code and then a completely opaque body. A bare `400` with `Content-Length: 0` is honest at the protocol level — caches and proxies and retry policies all behave correctly — but it tells the caller's developer nothing about *which* of the dozen things that can produce a `400` actually happened. They are reduced to guessing, or to trial-and-error against your sandbox, which is the slow integration all over again. Honest status, opaque body is better than dishonest status, helpful body — but the whole point of this post is that you do not have to choose. The status carries the coarse protocol signal for the machines that read envelopes; the `problem+json` body carries the fine-grained application signal for the caller's code and developer. Get both right and the error serves every audience at once.

## 2. RFC 9457 problem+json: a standard envelope for failure

You do not have to invent an error format. The IETF already did, and it is good. **RFC 9457, "Problem Details for HTTP APIs"** (published 2023, obsoleting the near-identical **RFC 7807** from 2016), defines a small, extensible JSON object for carrying error information, served with the media type `application/problem+json`. A *media type* is the label in the `Content-Type` header that tells the client how to interpret the bytes; `application/problem+json` says "this is JSON, and specifically it is a problem-details document." Using a registered media type means generic tooling — gateways, SDK generators, error-monitoring agents — can recognize an error body without bespoke configuration.

The standard defines five core members. Here is the canonical shape, served with the right status and content type:

```http
HTTP/1.1 402 Payment Required
Content-Type: application/problem+json
Content-Language: en

{
  "type": "https://errors.example.com/payments/card-declined",
  "title": "The card was declined",
  "status": 402,
  "detail": "The issuing bank declined this charge for insufficient funds.",
  "instance": "/payments/pay_7Q2mK"
}
```

Each member targets a specific consumer. This figure lays out the full set, including the extension members we will add in the next sections:

![a vertical stack of the problem+json members from type and title down through status detail and instance to the extension members code trace_id and a per field errors array](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-2.png)

Walk the five core members:

- **`type`** is a URI that *identifies the problem type*. It is the primary dispatch key in the standard: two responses with the same `type` are the same *kind* of problem. It does not have to be dereferenceable, but if it is, it should point to human-readable documentation for that error type. A client may safely use `type` as the stable identifier — it must not change for a given error kind. The default is `"about:blank"`, which means "no specific type; consult `status`."
- **`title`** is a short, human-readable summary of the problem type. It should *not* change from occurrence to occurrence ("The card was declined," not "The card ending 4242 was declined"). It is advisory for humans, not a branch key for machines.
- **`status`** is the HTTP status code, duplicated into the body. Why duplicate it? Because problem documents get logged, forwarded, and stored detached from their HTTP envelope; carrying the status inside the body keeps it self-describing. It must match the actual HTTP status line.
- **`detail`** is a human-readable explanation *specific to this occurrence*. This is where occurrence-specific facts live ("declined for insufficient funds" vs "declined as suspected fraud"). It is for humans; do not parse it.
- **`instance`** is a URI identifying the specific occurrence of the problem — typically the resource or request path that failed. It lets you correlate the error with a particular entity.

That is the whole core. It is deliberately small. The power comes from two things the spec explicitly blesses: **extension members** and the fact that the format is *self-describing and consistent*. Let me address an objection right away. The spec leans on `type` as the machine identifier, and `type` is a URI — long, awkward to write into a `switch` statement, and easy to typo. In practice, the most usable APIs add a short, flat extension member for the machine branch key. That is the next section, and it is the single highest-leverage decision in error design.

#### Worked example: a malformed JSON body

Before we get fancy, here is the simplest correct error — a request whose body is not even valid JSON. The right status is `400 Bad Request` (the syntax is wrong), and the body is a minimal problem document:

```http
POST /orders HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 6a1c9f3e-2b4d-4a7e-9f01-7c8b2d6e1a44

{ "items": [ { "sku": "SKU-1", "qty": }
```

```http
HTTP/1.1 400 Bad Request
Content-Type: application/problem+json

{
  "type": "https://errors.example.com/common/malformed-json",
  "title": "The request body is not valid JSON",
  "status": 400,
  "detail": "Parse error near line 1, column 38: expected a value after the colon.",
  "instance": "/orders",
  "code": "malformed_json",
  "trace_id": "req_8f3ad21c"
}
```

Notice that even the simplest error already carries a stable `code` and a `trace_id`. Those two extensions are not optional polish — they are the load-bearing members for a real integration, so let's build them properly.

## 3. A stable machine code, separate from the HTTP status

This is the most important rule in error design, so I will state it as a principle and then defend it:

> **Carry a stable, enumerated, application-level error `code` as a separate member, distinct from both the HTTP status and the human-readable `detail`. The `code` is the machine's branch key; nothing else is.**

Why not just use the HTTP status as the branch key? Because the HTTP status is *coarse*. A single status maps to many application-level causes. `402 Payment Required` (or `400`, depending on your taste) covers card declined, but "declined" itself fans out into insufficient funds, expired card, suspected fraud, do-not-honor, and issuer-unreachable — each of which the caller handles differently. Insufficient funds means "ask the customer to use a different card." Expired card means "ask them to update the expiry." Suspected fraud means "do not retry and do not reveal why." Issuer-unreachable means "this might succeed on retry." The HTTP status cannot distinguish these; a machine code can.

Why not parse `detail`? Because `detail` is human prose, and as we saw, it changes. The whole point of separating `code` from `detail` is that you can reword `detail` freely — translate it, A/B-test it, add occurrence-specific facts — without breaking a single integration, because no machine reads it. The `code` is a frozen part of the contract; `detail` is a mutable convenience. This separation is what lets the two audiences evolve independently.

Here is the declined-card error with a proper code namespace:

```http
HTTP/1.1 402 Payment Required
Content-Type: application/problem+json

{
  "type": "https://errors.example.com/payments/card-declined",
  "title": "The card was declined",
  "status": 402,
  "detail": "The issuing bank declined this charge for insufficient funds.",
  "instance": "/payments/pay_7Q2mK",
  "code": "card_declined",
  "decline_code": "insufficient_funds",
  "trace_id": "req_8f3ad21c",
  "documentation_url": "https://docs.example.com/errors/card-declined"
}
```

A few design choices worth defending. The `code` is `card_declined` — short, `snake_case`, and *categorical*. The finer-grained reason lives in a second member, `decline_code` (`insufficient_funds`), exactly mirroring how Stripe structures its decline errors. This two-level scheme is deliberate: clients that only care "did the charge fail?" branch on `code`; clients that want to tailor the customer message branch on `decline_code`. Both are stable enumerations you document and version carefully.

### Rules for designing the code namespace

- **Codes are an enumeration, documented and stable.** Treat the set of codes like an enum in a shared schema: adding a new code is a *forward-compatible* change (clients fall through to a default branch), but changing the meaning of an existing code or removing one is *breaking*. This is the same compatibility logic as the rest of your contract — see **[backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change)** for why adding is safe and removing is not.
- **`snake_case`, lowercase, no spaces, no punctuation.** A code is a token, not a sentence. `card_declined`, not `Card Declined` or `CARD-DECLINED`.
- **Namespace or categorize them.** Either prefix (`payments.card_declined`) or pair the flat `code` with the URI `type` that gives the namespace. Pick one scheme and apply it everywhere.
- **The code is independent of the status.** The same `code` can in principle appear with different statuses across contexts, and one status hosts many codes. Do not derive one from the other; carry both.

Here is how the caller actually consumes this — note that the *only* thing the program branches on is `code` and `decline_code`:

```javascript
async function charge(order) {
  const res = await fetch("https://api.example.com/payments", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json",
      "Idempotency-Key": order.idempotencyKey,
    },
    body: JSON.stringify({ order_id: order.id, amount: order.total, currency: "USD" }),
  });

  if (res.ok) return res.json();

  // Centralized failure handling, keyed on the stable code — never on `detail`.
  const problem = await res.json();
  switch (problem.code) {
    case "card_declined":
      if (problem.decline_code === "insufficient_funds") return promptForDifferentCard(problem);
      if (problem.decline_code === "expired_card") return promptToUpdateExpiry(problem);
      return promptForDifferentCard(problem);
    case "rate_limited":
      return scheduleRetry(res.headers.get("Retry-After"));
    case "validation_failed":
      return showFieldErrors(problem.errors);
    default:
      // Unknown code: log the trace_id and fail closed, do NOT blind-retry.
      logUnknownError(problem.trace_id, problem.code);
      return failGracefully(problem);
  }
}
```

The `default` branch is the tolerant-reader principle in action: a client written today must not crash when you add a new `code` tomorrow. It logs the `trace_id` and fails closed rather than guessing. That single discipline is why a well-designed code namespace can grow for years without breaking anyone.

### Why two levels of code, not one

A reasonable objection: if the `code` is the branch key, why introduce a second member (`decline_code`) at all — why not just have more codes, like `card_declined_insufficient_funds`? The answer is about *who needs which granularity*, and it generalizes well beyond payments. Most callers only care about the coarse outcome: "the charge failed, so do not mark the order paid." They branch on `code == "card_declined"` and stop. A *minority* of callers — the ones building a polished checkout — want to tailor the customer-facing message to the specific reason: "your card has expired" versus "insufficient funds" versus a generic "your bank declined this." Those callers branch on `decline_code`. By splitting the two, the common case stays a single short comparison, and the rich case is available without forcing every caller to enumerate twenty fused codes they do not care about. The coarse code is the *category*; the fine code is the *reason within the category*. This is the same instinct as a well-designed exception hierarchy: catch the base class when you do not care, catch the specific subclass when you do.

There is also a compatibility benefit to the split. Issuers and gateways invent new decline reasons constantly. If decline reasons were fused into the top-level `code`, every new reason would be a new top-level code, and the top-level enumeration — the one most clients switch on — would churn. By isolating the churn in `decline_code`, the stable top-level `code` set stays small and changes rarely, while the volatile `decline_code` set can grow freely because clients treat unknown decline codes as "some decline reason we do not specifically handle" and fall back to the generic message. You have localized the instability to the member designed to absorb it.

### Mapping internal failures to public codes is a deliberate act

One trap worth naming: do not let your *internal* error taxonomy leak directly into your *public* codes. Internally you might distinguish `GatewayConnectionTimeout`, `GatewayConnectionRefused`, and `GatewayTLSHandshakeError` — three different exceptions in three different code paths. To the caller, all three are the same thing: "we could not reach the payment network; this might succeed on retry," which is one public code, `gateway_unavailable`, with a `503` and a `Retry-After`. The mapping from internal exception to public code is a *design decision you make on purpose*, not a mechanical 1:1 projection of your class hierarchy. If you let internal classes become public codes, two bad things happen: you leak your internal architecture (every refactor that splits or merges an exception class is now a breaking API change), and you flood the caller with distinctions they cannot act on. The public code namespace is a *curated, caller-facing vocabulary*, designed from the caller's decision tree backward — "what are the genuinely different things a caller would do?" — and then each branch of that tree gets a code. Build the namespace from the caller's actions, not from your stack traces.

#### Worked example: a `card_declined` with a stable code and a trace id

Let me trace the full declined-card cycle, because it exercises every member we have built. The customer checks out; the platform calls our Payments API:

```http
POST /payments HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 6a1c9f3e-2b4d-4a7e-9f01-7c8b2d6e1a44

{ "order_id": "ord_5T1pX", "amount": 4999, "currency": "USD", "payment_method": "pm_card_visa" }
```

The issuing bank declines for insufficient funds. We return:

```http
HTTP/1.1 402 Payment Required
Content-Type: application/problem+json
Content-Language: en

{
  "type": "https://errors.example.com/payments/card-declined",
  "title": "The card was declined",
  "status": 402,
  "detail": "The issuing bank declined this charge for insufficient funds. Ask the customer for a different payment method.",
  "instance": "/payments/pay_7Q2mK",
  "code": "card_declined",
  "decline_code": "insufficient_funds",
  "trace_id": "req_8f3ad21c",
  "documentation_url": "https://docs.example.com/errors/card-declined"
}
```

Now trace what each consumer does with this single body. The **caller's code** reads `code: "card_declined"` and `decline_code: "insufficient_funds"`, and renders "Your card was declined due to insufficient funds — please try a different card." It does *not* retry, because (as we will see in section 8) `card_declined` is terminal. The **caller's developer**, debugging a customer complaint, reads `detail` and `documentation_url` and understands immediately. The **support engineer**, handed `req_8f3ad21c` by the customer, greps the logs for that exact request and finds the full server-side context — including the parts we deliberately *did not* put in the response. One body, three audiences, zero guesswork. That is the contract working.

## 4. Field-level validation errors: an array of per-field problems

Card declines are single-cause. Validation is the opposite: a request can be wrong in five places at once, and the worst possible thing your API can do is fail on the first one, make the client fix it, resubmit, and fail on the second. A form with five bad fields should get *five* problems back in one response so the user can fix them all at once.

The right status for a syntactically valid body that fails semantic validation is `422 Unprocessable Content` — the JSON parsed fine (so it is not a `400`), but the *content* violated your rules. (Some teams use `400` for everything; the **[status codes post](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx)** argues the `400`-vs-`422` line in depth. Whichever you pick, be consistent.) The body carries an extension member — conventionally `errors` — that is an *array* of per-field problem objects:

![a before and after contrast of a flat 422 with a single unparseable detail string versus a 422 carrying an errors array with one entry per failing field each with its own pointer and code](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-5.png)

#### Worked example: a 422 with field-level problems

The platform tries to create a payment with several bad fields at once: a negative amount, an unsupported currency, and a missing order reference.

```http
POST /payments HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "amount": -50, "currency": "XQZ", "payment_method": "pm_card_visa" }
```

We validate the *entire* body and return every problem at once:

```http
HTTP/1.1 422 Unprocessable Content
Content-Type: application/problem+json

{
  "type": "https://errors.example.com/common/validation-failed",
  "title": "The request body failed validation",
  "status": 422,
  "detail": "3 fields are invalid. See the errors array for per-field details.",
  "instance": "/payments",
  "code": "validation_failed",
  "trace_id": "req_b73e0a91",
  "errors": [
    {
      "field": "order_id",
      "pointer": "/order_id",
      "code": "required",
      "detail": "order_id is required."
    },
    {
      "field": "amount",
      "pointer": "/amount",
      "code": "out_of_range",
      "detail": "amount must be a positive integer of minor currency units (cents)."
    },
    {
      "field": "currency",
      "pointer": "/currency",
      "code": "unsupported_value",
      "detail": "currency XQZ is not supported. Use one of: USD, EUR, GBP, VND."
    }
  ]
}
```

There is real design discipline in this shape. Each entry carries its own machine `code` (`required`, `out_of_range`, `unsupported_value`) so a client can map each error to a UI behavior without parsing `detail`. The `pointer` is a **JSON Pointer (RFC 6901)** — the `/amount` syntax that unambiguously locates a field even when it is nested deep in the body (`/items/0/price` points at the price of the first item). I include both a friendly `field` name and the precise `pointer`; the `field` is convenient for flat bodies, the `pointer` is correct for nested ones. The top-level `detail` summarizes the count; the per-field `detail` explains each specific failure *and what valid input looks like* — note that the currency error lists the supported set. That "here is what's valid" detail is the difference between an error and a help message.

A web form consumes this trivially: iterate `errors`, look up each `pointer` to find the input element, and render the per-field `detail` next to it. No prose parsing, no round-trip-per-field. Here is the consumer:

```javascript
function showFieldErrors(problem) {
  // Clear prior errors, then attach each problem to its input by JSON Pointer.
  clearFieldErrors();
  for (const err of problem.errors) {
    const input = formElementForPointer(err.pointer); // "/amount" -> the amount <input>
    if (input) attachError(input, err.detail, err.code);
    else attachFormError(err.detail); // pointer has no widget: show at form level
  }
}
```

One subtlety worth calling out: a per-field error needs a stable per-field `code` *and* a stable `pointer`, and you should think of both as part of the contract. Renaming a field changes its `pointer`, which is a breaking change for any client that maps errors back to inputs — the same field-lifecycle discipline as the rest of the body, covered in **[designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming)**.

## 5. Correlation: a trace id the caller can quote

When a customer emails support saying "my payment failed," the single most valuable thing they can include is an identifier that lets your team find the exact request in your logs. So *give it to them in the error body.* Every error response should carry a correlation identifier — `trace_id`, `request_id`, call it what you like, but be consistent — that:

- is **unique per request** (or per trace, if you propagate a distributed trace id across services),
- is **logged server-side** with the full context of the failure,
- is **safe to expose** (it is an opaque token, not a sequential id that leaks volume, and it carries no PII),
- and is **echoed in a response header too** (commonly the gateway's `X-Request-Id` or a W3C `traceparent`), so it is present even on errors where the body is empty (a raw `502` from a proxy, say).

This is the bridge between the public error and your private observability. You deliberately keep the *sensitive* detail out of the response (next section) — but you log all of it server-side, keyed by `trace_id`. So when the support engineer is handed `req_8f3ad21c`, they pull up the full picture: the upstream gateway's exact decline reason, the internal exception, the user and order ids, the timing. The customer-facing body stayed clean; the server-side log stayed complete; the `trace_id` joined them. This is precisely the observability discipline covered for the whole API in **[observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos)** — the error body is where it meets the caller.

A practical detail: propagate the *inbound* request id into the error rather than minting a fresh one if a gateway or load balancer already assigned one. If your gateway sets `X-Request-Id` on the way in (most do — see **[the API gateway and BFF pattern](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend)**), reuse that value as `trace_id` so the caller's id, your gateway's id, and your application logs all agree. One id, end to end, is worth more than three different ids that you have to manually correlate during an incident.

Why does this matter so much in a distributed system? Because a single inbound API call rarely fails in one place. The payments request fans out to a fraud-scoring service, a ledger service, and a third-party gateway, and the failure that bubbles up to the caller may have originated three hops deep. Without a correlation id that *propagates across all those hops*, an on-call engineer handed "payment failed at 14:32" has to manually stitch together logs from four services by timestamp and hope no two requests overlapped — a miserable, error-prone exercise during an incident. With a single `trace_id` propagated as a header on every internal hop (this is exactly what W3C `traceparent` standardizes), the engineer runs one query — "show me everything tagged `req_8f3ad21c`" — and gets the entire causal chain across all four services, in order, in seconds. The error body's `trace_id` is the *customer-facing entry point* into that chain. The customer quotes it; support pastes it; the whole distributed trace lights up. This is the same correlation discipline the broader **[observability post](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos)** covers for healthy traffic; errors are simply where it earns its keep most visibly.

One caution on the id itself: it must be *opaque and non-enumerable*. Do not use a sequential integer (`error #18,442`), because a sequential id leaks your volume (an attacker watching the id climb can estimate your traffic) and invites enumeration. Use a random token or a UUID-derived value. And it must carry no information that is itself sensitive — a `trace_id` derived from a customer's email or card number would defeat the entire leak-safety discipline of the next section. The id is a *handle*, not a *fact*: it lets the right people look up the facts in a system that is properly access-controlled, while revealing nothing on its own.

#### Worked example: a 404 that is honest about a missing resource

To show the minimal-but-complete shape one more time on a different failure, here is a `GET` for a refund that does not exist:

```http
GET /refunds/rf_does_not_exist HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 404 Not Found
Content-Type: application/problem+json

{
  "type": "https://errors.example.com/common/not-found",
  "title": "The resource was not found",
  "status": 404,
  "detail": "No refund with id rf_does_not_exist exists, or you do not have access to it.",
  "instance": "/refunds/rf_does_not_exist",
  "code": "not_found",
  "trace_id": "req_5d20af71"
}
```

Two subtle decisions here. There is *no* `errors` array — a `GET` has no body to validate, so attaching an empty per-field array would be noise; match the shape's richness to the failure. And the `detail` deliberately fuses "does not exist" with "or you do not have access," because revealing the difference is an enumeration oracle: if a `404` reliably means "exists but you cannot see it" while a different response means "does not exist," an attacker can probe which ids are real. For resources where the existence itself is sensitive, returning `404` for both "missing" and "forbidden" is the correct, leak-safe choice — the security angle reaching even into the wording of a not-found message.

## 6. Actionable messages and never leaking internals

Two rules pull in opposite directions and must be balanced: **say what the caller should DO**, and **never leak internals**. The art is putting all the actionable signal in and all the dangerous detail out.

### Actionable: tell the caller their next step

A `detail` that says "an error occurred" is useless. A `detail` that says "The `currency` field must be one of USD, EUR, GBP, VND — you sent XQZ" tells the caller exactly how to succeed on the next try. The test for a good message: *can the caller act on it without contacting you?* Aim for that. Where the action is involved, link to docs — a `documentation_url` member (as in our declined-card body) that points to a page explaining the error type and its remedies. Stripe and Twilio both do this; an error code that links to a doc page is a hallmark of an API built for the people consuming it, which is the whole thesis of **[designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal)**.

The figure below scores the bad and good shapes across the five dimensions a complete error must satisfy:

![a matrix scoring a bare 500 against a structured problem+json across five columns machine code human message trace id next action and no leaked internals where the bad row fails every column and the good row passes every one](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-3.png)

| Dimension | Bad error (`200` + `{"success": false}`) | Good error (`problem+json`) |
| --- | --- | --- |
| Machine-readable | No stable code; status lies (`200`) | Stable `code` separate from status |
| Human-readable | Generic, often misleading | Specific `detail`, says what to do |
| Traceable | No id; support cannot find the request | `trace_id` echoed in body and header |
| Actionable | "Try again later" (often wrong) | Names the field, the valid set, a doc link |
| Leak-safe | Sometimes dumps a stack trace in debug | Internals logged server-side only |
| Consistent | Differs per endpoint | One envelope, every endpoint |

### Leak-safe: the security angle

An error response is an *information channel to an attacker*, and verbose errors have leaked more secrets than almost any other class of bug. The rule is blunt:

> **An error body must never contain a stack trace, a raw SQL query or database error, an internal hostname or file path, a library version, a secret, or any personal data beyond what the caller already supplied.**

Why each is dangerous:

- **Stack traces and raw exceptions** reveal your framework, library versions (so an attacker can look up known CVEs), and code structure. A `psycopg2.errors.UniqueViolation` dumped into a response tells the attacker your database, your driver, and your schema.
- **SQL in errors** (`ERROR: duplicate key value violates unique constraint "users_email_key"`) hands an attacker your table and column names and is a classic enumeration oracle — it confirms which emails are already registered.
- **Internal hostnames, IPs, and file paths** map your internal topology.
- **PII echoed back** (the full card number, a government id) widens the blast radius of any logging or interception. Echo only what is safe and non-identifying.

The discipline that makes this automatic is a **central error mapper**: one place where every raised exception is converted into a `problem+json` document, internals are stripped, a `trace_id` is stamped, and the full original error is logged server-side. No handler ever serializes a raw exception to the response. The figure shows the flow:

![a directed graph where a handler raises into either a store timeout or a gateway decline both flowing into a central error mapper that redacts SQL and PII before emitting a problem+json with a stable code to the client](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-4.png)

Here is that mapper as a small framework-agnostic middleware. The shape is the point, not the language:

```python
import logging, uuid
log = logging.getLogger("errors")

PUBLIC_PROBLEMS = {
    "card_declined":     (402, "https://errors.example.com/payments/card-declined",  "The card was declined"),
    "validation_failed": (422, "https://errors.example.com/common/validation-failed","The request body failed validation"),
    "rate_limited":      (429, "https://errors.example.com/common/rate-limited",      "Too many requests"),
    "not_found":         (404, "https://errors.example.com/common/not-found",         "The resource was not found"),
}

def to_problem(exc, request):
    trace_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:8]}"

    if isinstance(exc, AppError) and exc.code in PUBLIC_PROBLEMS:
        status, type_uri, title = PUBLIC_PROBLEMS[exc.code]
        body = {
            "type": type_uri, "title": title, "status": status,
            "detail": exc.safe_detail,           # author-controlled, leak-checked prose
            "instance": request.path,
            "code": exc.code,
            "trace_id": trace_id,
        }
        if exc.field_errors:                      # validation: attach the per-field array
            body["errors"] = exc.field_errors
        # Log the FULL context server-side, keyed by trace_id — never in the response.
        log.warning("handled error", extra={"trace_id": trace_id, "code": exc.code, "detail": exc.internal_detail})
        return status, "application/problem+json", body

    # Anything unexpected becomes an opaque 500 — NO stack trace, NO exception text.
    log.error("unhandled error", exc_info=exc, extra={"trace_id": trace_id})
    return 500, "application/problem+json", {
        "type": "https://errors.example.com/common/internal",
        "title": "An unexpected error occurred",
        "status": 500,
        "detail": "An unexpected error occurred on our side. Quote the trace_id to support.",
        "instance": request.path,
        "code": "internal_error",
        "trace_id": trace_id,
    }
```

Two things make this safe. First, the `500` branch is *opaque on purpose* — it says nothing about what failed internally, but it still carries a `trace_id` so the caller has something to quote and you have something to grep. Second, only `exc.safe_detail` (prose the engineer wrote and reviewed) reaches the wire; `exc.internal_detail` (which may contain the raw upstream error) goes only to the log. The unknown-exception path can *never* leak, because it does not serialize `exc` at all.

## 7. Consistency: one error shape, every endpoint

Here is a rule that sounds boring and is in fact one of the highest-leverage decisions you will make:

> **Every endpoint in your API returns errors in the same shape. A caller writes error handling once and it works everywhere.**

The cost of inconsistency compounds. If `/payments` returns `{"error": {"code": ...}}`, `/orders` returns `{"errors": [...]}` at the top level, and `/refunds` returns a bare `{"message": ...}`, then the caller cannot centralize error handling at all — they need a per-endpoint adapter. Worse, *new* endpoints that copy whichever neighbor the author happened to look at perpetuate the divergence. Inconsistency is not a one-time tax; it is a recurring tax on every integration and every new endpoint, forever.

Consistency is what makes the `default` branch in our section-3 client possible. Because *every* error — declined card, validation failure, rate limit, not-found, internal — is the same `problem+json` envelope with `code` and `trace_id`, the caller's centralized handler reads those two members regardless of which endpoint failed. The shape is the contract; the codes vary within it.

Enforce consistency mechanically, not by hoping. Three tactics:

- **A single error type and serializer.** As in the mapper above, there is exactly one function that turns any failure into a response. Handlers `raise AppError(code=...)`; they never build error bodies themselves. This makes the shape uniform by construction.
- **Document the envelope once in your OpenAPI spec** and `$ref` it from every operation's error responses, so the schema is literally shared. See **[OpenAPI and the spec-first workflow](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate)**.
- **Lint for it.** A spec linter (Spectral) can assert that every `4xx`/`5xx` response uses the shared `problem+json` schema and that the `Content-Type` is `application/problem+json`. A drift in the shape fails CI before it ships.

Here is the shared response in an OpenAPI 3.1 fragment so the schema is defined exactly once:

```yaml
components:
  schemas:
    Problem:
      type: object
      properties:
        type: { type: string, format: uri }
        title: { type: string }
        status: { type: integer }
        detail: { type: string }
        instance: { type: string, format: uri-reference }
        code: { type: string, description: "Stable machine-readable error code." }
        trace_id: { type: string }
        errors:
          type: array
          items:
            type: object
            properties:
              field: { type: string }
              pointer: { type: string }
              code: { type: string }
              detail: { type: string }
      required: [type, title, status, code]
  responses:
    Error:
      description: A problem+json error response.
      content:
        application/problem+json:
          schema: { $ref: "#/components/schemas/Problem" }
paths:
  /payments:
    post:
      responses:
        "201": { description: Payment created }
        "402": { $ref: "#/components/responses/Error" }
        "422": { $ref: "#/components/responses/Error" }
        "429": { $ref: "#/components/responses/Error" }
        "500": { $ref: "#/components/responses/Error" }
```

Now `/orders` and `/refunds` `$ref` the same `Error` response. The shape cannot drift, because there is only one of it.

### Verifying the error contract is as real as verifying the happy path

Here is the failure mode I see most often even on teams that have internalized everything above: they design a beautiful error contract, document it, and then *test only the happy path*. The error responses are exactly the part of the contract that gets exercised in production but skipped in CI, because writing a test that forces a card decline or a database timeout takes effort. So the error shapes rot — a refactor changes the `500` body, a new endpoint forgets the envelope, a `detail` quietly starts including the exception text in one code path — and nobody notices until a caller files a bug. Errors are a contract, and an untested contract is a wish.

Treat error responses as first-class test cases. Concretely:

- **Contract-test the error envelope, not just the success body.** A consumer-driven contract test (Pact, or a schema-validation test in CI) should assert that every documented error response actually validates against the shared `Problem` schema, that the `Content-Type` is `application/problem+json`, and that the `status` member matches the HTTP status line. This is the same consumer-driven-contract discipline covered in **[contract testing](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs)**, pointed at failures instead of successes.
- **Force every error path in a test.** Inject a declined card via a test payment method, send a deliberately malformed body, exhaust the rate limit, kill a dependency to trigger the `500` path — and assert the *exact* `code` and the *presence* (never the exact text) of `detail` and `trace_id`. Asserting on `code` and not on `detail` is the test mirror of the contract itself: the test, like the caller, must depend only on the stable member.
- **Lint the error catalog for leaks.** A simple CI check can scan every error-response fixture for forbidden substrings — `Traceback`, `SELECT `, `at com.`, a stack-frame pattern, internal hostnames — and fail the build if any appear. This turns "never leak internals" from a code-review hope into a mechanical gate.
- **Diff the code namespace across versions.** Treat the set of `code` values like a schema. A CI step that diffs the documented codes between the current branch and `main` can flag a *removed* or *redefined* code as a potential breaking change, the same way `oasdiff` or `buf breaking` guards your request/response shapes.

#### Worked example: an idempotent retry that returns the cached error

A subtle interaction that error design must get right: what happens when a caller retries a request that *already failed terminally*? Suppose the platform sends a payment with an `Idempotency-Key`, the card is declined, and then — because the network ate the response and the client never saw the `402` — the client retries with the *same* key. The correct behavior is that idempotency caches the *outcome*, success or failure, so the retry returns the identical declined response rather than attempting a second charge:

```http
POST /payments HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Idempotency-Key: 6a1c9f3e-2b4d-4a7e-9f01-7c8b2d6e1a44
Content-Type: application/json

{ "order_id": "ord_5T1pX", "amount": 4999, "currency": "USD", "payment_method": "pm_card_visa" }
```

```http
HTTP/1.1 402 Payment Required
Content-Type: application/problem+json
Idempotency-Replayed: true

{
  "type": "https://errors.example.com/payments/card-declined",
  "title": "The card was declined",
  "status": 402,
  "detail": "The issuing bank declined this charge for insufficient funds.",
  "instance": "/payments/pay_7Q2mK",
  "code": "card_declined",
  "decline_code": "insufficient_funds",
  "trace_id": "req_8f3ad21c"
}
```

The same `trace_id` comes back (it was cached with the outcome), and an `Idempotency-Replayed: true` header signals this was a replay, not a fresh attempt. The deep mechanics of idempotency caching — including the question of *which* errors should be cached versus recomputed (you cache terminal outcomes; you generally do *not* cache a transient `503`, so the caller can genuinely retry) — are the subject of **[idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions)**. The error-design lesson here: your error responses must participate in idempotency, because a retried terminal failure must return the *same* failure, not a new charge.

## 8. Retryable vs terminal: signaling whether to try again

This is where error design meets distributed-systems reality, and where the opening double-charge actually happened. Some errors are *transient* — the same request, retried later, can succeed. Others are *terminal* — retrying the identical request will *never* succeed, and retrying it may cause harm. The caller must be able to tell which is which *without guessing*, and the signal is partly the status code and partly an explicit header.

> **The HTTP status family encodes retryability: most `4xx` errors are terminal (the request is wrong; fix it and resend), while `429` and `503` are retryable (the server is temporarily unable; wait and retry). Make this explicit with `Retry-After` on retryable errors.**

Why does the status family carry this meaning? Because `4xx` means "the client did something wrong" — and resending the *same* wrong thing produces the same wrong result. `400`, `401`, `403`, `404`, `409`, `422` are terminal: a malformed body stays malformed, a missing permission stays missing, a declined card stays declined. The two `4xx` exceptions are `429 Too Many Requests` (you were right, just too fast) and `408 Request Timeout`. The `5xx` family means "the server failed," and `503 Service Unavailable` specifically means "temporarily unable, try later." This figure lays out the split:

![a matrix contrasting retryable errors like 429 and 503 that carry a Retry-After header against terminal errors like 400 402 and 422 that the client must fix before resending](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-6.png)

| | Retryable | Terminal |
| --- | --- | --- |
| Typical status | `429`, `503`, sometimes `502`/`504` | `400`, `401`, `403`, `404`, `409`, `422`, `402` |
| Meaning | Server temporarily can't; same request may succeed later | Request is wrong/forbidden; same request never succeeds |
| `Retry-After` | Present — tells the client how long to wait | Absent |
| Correct client action | Wait the stated time, then retry (with backoff) | Fix the request (or give up); do **not** blind-retry |
| Example `code` | `rate_limited`, `service_unavailable` | `card_declined`, `validation_failed`, `forbidden` |

Two warnings. First, `card_declined` is *terminal* even though it lives at `402` and feels transient — retrying the same card produces the same decline and may trip fraud rules. This is exactly the trap from the opening: a reworded decline that fell through to a retry branch double-charged a customer. The status and the `code` must both say "terminal." Second, a `500` is *ambiguous* — it might be transient (a flaky dependency) or permanent (a bug) — so a careful client retries `500`s *only* if the operation is idempotent (a `GET`, or a `POST` carrying an `Idempotency-Key`), and otherwise treats it as terminal. Idempotency is what makes a retry safe at all; that is its own post, **[idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions)**.

### Signaling the wait: 429, 503, and Retry-After

When you *do* want the caller to retry, tell them when. The `Retry-After` header (RFC 9110) carries either a number of seconds or an HTTP date. On a `429`, pair it with rate-limit headers so the client can self-pace:

#### Worked example: a rate-limit error with Retry-After

The platform fires a burst of payment creations and exceeds its rate limit. Picture a token bucket: the bucket holds a burst capacity and refills at $r$ tokens per second; the server allows a request when $\text{tokens} \ge 1$ and rejects with `429` otherwise. When the bucket empties, the next request gets:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/problem+json
Retry-After: 2
RateLimit-Limit: 100
RateLimit-Remaining: 0
RateLimit-Reset: 2

{
  "type": "https://errors.example.com/common/rate-limited",
  "title": "Too many requests",
  "status": 429,
  "detail": "You have exceeded 100 requests per minute. Retry after 2 seconds.",
  "instance": "/payments",
  "code": "rate_limited",
  "trace_id": "req_c40fb18a"
}
```

The client reads `Retry-After: 2`, waits two seconds (during which the bucket refills at rate $r$), and retries — crucially, *with the same `Idempotency-Key`* so the retry cannot create a duplicate payment if the original actually went through. This figure walks the round-trip:

![a left to right timeline showing a burst hitting the rate cap then a 429 with Retry-After of 2 seconds then the client waiting then the token bucket refilling then a retry with the same key then a single 201 created](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-7.png)

The full discipline (rate-limiting algorithms, quota headers, the standardized `RateLimit-*` family) is its own post, **[rate limiting, quotas, and abuse protection](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection)**. The error-design point is narrow and important: a retryable error must *say so*, and the most reliable way to say so is the status code plus `Retry-After`. A retryable error without a `Retry-After` forces every client to invent its own backoff, and the naive ones hammer you the instant they fail — a retry storm that turns a brief overload into a sustained outage.

The retry-storm dynamic is worth making concrete, because it is a feedback loop that turns a small problem into an outage. Suppose your service briefly degrades and starts returning `503` to one in ten requests. A well-behaved client reads `Retry-After`, waits, and adds jitter, so its retries spread out and barely add to your load. A naive client — one that retries immediately on any failure with no backoff and no `Retry-After` to read — does the opposite: the instant it sees a `503`, it retries, *adding* load to an already-degraded service, which produces *more* `503`s, which produce *more* immediate retries. The system has positive feedback. A blip becomes a death spiral. This is not hypothetical; it is the mechanism behind a large fraction of cascading outages. Your error design is one of the controls on that loop. By returning an honest `503` (so retry middleware engages correctly), a `Retry-After` (so clients wait rather than spin), and a stable `code` (so clients can distinguish "retry me" from "do not retry me"), you give every client the information it needs to *not* pile on. You cannot force clients to behave, but you can make the well-behaved path the easy, obvious one — and you can avoid the two anti-patterns that guarantee a storm: a retryable error with no `Retry-After`, and a terminal error (`card_declined`, `validation_failed`) that the client mistakes for retryable because you did not give it a clear signal.

There is a deeper point about *what the caller gets to assume*, which is the recurring question of this whole series. When you return a `429` with `Retry-After: 2`, you are making a promise: "wait two seconds and you will probably get through." When you return a `402 card_declined`, you are making the opposite promise: "this exact request will fail again — do not bother." When you return a `500`, you are honestly admitting "I do not know whether retrying helps." Each of these is a different *contract about the future*, and the caller builds their retry logic on top of the promise you make. Break the promise — say "retry later" on a card decline — and you have not just returned a confusing error; you have lied about the future and induced the caller to take an action (retry) that costs them money (a double charge) or costs you load (a storm). Honesty about retryability is not a nicety. It is a load-bearing part of the contract, and it is the single place where error design most directly touches the reliability of both your system and your caller's.

## 9. Internationalization of messages

If your API serves clients in multiple languages, the human-readable members (`title`, `detail`, per-field `detail`) may need translation — but the machine-readable members (`type`, `code`, `decline_code`, per-field `code`, `pointer`) must *never* be translated. This falls straight out of the two-audience principle: codes are for machines and are language-neutral by definition; prose is for humans and is the only thing that varies by locale.

The clean mechanism is HTTP content negotiation. The client signals its preference with `Accept-Language`; the server returns the matching language in `detail` and declares it with `Content-Language`:

```http
POST /payments HTTP/1.1
Host: api.example.com
Accept-Language: fr-FR, fr;q=0.9, en;q=0.5
Content-Type: application/json

{ "amount": -50, "currency": "USD" }
```

```http
HTTP/1.1 422 Unprocessable Content
Content-Type: application/problem+json
Content-Language: fr-FR

{
  "type": "https://errors.example.com/common/validation-failed",
  "title": "La validation du corps de la requete a echoue",
  "status": 422,
  "detail": "Le champ amount doit etre un entier positif.",
  "instance": "/payments",
  "code": "validation_failed",
  "trace_id": "req_2f9a1c44",
  "errors": [
    { "field": "amount", "pointer": "/amount", "code": "out_of_range",
      "detail": "amount doit etre un entier positif en unites mineures (centimes)." }
  ]
}
```

The `code` is still `validation_failed` and the per-field `code` is still `out_of_range` — a French-locale client and an English-locale client branch on *identical* tokens; only the prose changes. This is why separating `code` from `detail` is not merely tidy: it is what makes localization possible without forking your client logic per language. A useful corollary: because the `code` is the stable contract, many teams let the *client* own translation entirely — the server emits a stable `code` and the client maps it to a localized string from its own message catalog, which guarantees consistent terminology across the whole product and removes locale handling from the server's hot path.

## 10. Partial success: the batch-error problem

The single-result model — one request, one status, one outcome — breaks down for *batch* operations. If the platform submits ten payments in one request and three fail, what status do you return? A `200` lies about the three failures; a `4xx`/`5xx` lies about the seven successes. The honest answer is to report each item's outcome independently.

There are two good designs, and which you pick depends on whether the batch is *atomic* (all-or-nothing) or *independent* (each item stands alone):

- **Atomic batch:** the whole batch succeeds or the whole batch is rejected. If any item is invalid, return a single `422` for the request and use the `errors` array to point at the offending items (e.g. `pointer: "/items/2/amount"`). Nothing is applied. This is the right model when the items must be consistent with each other (a multi-leg transfer that must balance).
- **Independent batch:** each item is processed on its own and can succeed or fail independently. The cleanest expression is `207 Multi-Status` (from WebDAV, RFC 4918, widely reused for JSON batch APIs) with a per-item result list, each carrying its own status and — for failures — its own embedded `problem+json`.

#### Worked example: a batch with partial success

The platform submits three payments; one succeeds, one is declined, one is invalid:

```http
POST /payments/batch HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json

{ "items": [
  { "order_id": "ord_a", "amount": 4999, "currency": "USD", "payment_method": "pm_ok" },
  { "order_id": "ord_b", "amount": 2500, "currency": "USD", "payment_method": "pm_nsf" },
  { "order_id": "ord_c", "amount": -5,   "currency": "USD", "payment_method": "pm_ok" }
] }
```

```http
HTTP/1.1 207 Multi-Status
Content-Type: application/json

{
  "results": [
    { "index": 0, "status": 201, "payment_id": "pay_a1" },
    { "index": 1, "status": 402, "problem": {
        "type": "https://errors.example.com/payments/card-declined",
        "title": "The card was declined", "status": 402,
        "code": "card_declined", "decline_code": "insufficient_funds",
        "detail": "The issuing bank declined this charge for insufficient funds." } },
    { "index": 2, "status": 422, "problem": {
        "type": "https://errors.example.com/common/validation-failed",
        "title": "The request body failed validation", "status": 422,
        "code": "validation_failed",
        "errors": [ { "field": "amount", "pointer": "/amount", "code": "out_of_range",
                      "detail": "amount must be a positive integer." } ] } }
  ],
  "trace_id": "req_e91b7d20"
}
```

This figure shows the per-item breakdown:

![a matrix of three batch items where item zero returns a 201 created item one returns a 402 card_declined and item two returns a 422 validation error each with its own code and outcome](/imgs/blogs/error-design-a-machine-readable-human-friendly-contract-8.png)

The design discipline: each result carries its own `status` and, on failure, the *same* `problem+json` shape you use everywhere else — embedded as a `problem` member rather than served as the top-level body. The consistency rule from section 7 pays off again: the caller's per-item failure handler is the *same code* as its top-level failure handler, because the error shape is identical whether it arrives alone or nested in a batch. And every payment carries its own `order_id`, so the caller can map results back to inputs even though `207` returns them as a list — and combined with per-item idempotency keys, the caller can safely retry *only* the failed items.

One honest caveat: `207 Multi-Status` is less universally understood than `200`, and some clients and gateways treat any non-`2xx`-but-not-`200` status oddly. If your batch clients are diverse and you cannot rely on `207` being handled correctly, an alternative is to return `200 OK` *and* require callers to inspect per-item `status` fields — but then you must document loudly that the top-level `200` does **not** imply every item succeeded. The trade-off is honesty (`207`) versus universality (`200`); pick deliberately and document whichever you choose. What you must *not* do is the thing the opening vendor did — return a single `200` that silently hides failures with no per-item structure at all.

## 11. The error catalog: errors as a documented, owned vocabulary

Everything so far produces a *set* of error codes scattered across your handlers. The final discipline is to treat that set as a first-class, documented artifact — an **error catalog** — and to own it the way you own any other part of the public contract. The catalog is the single source of truth for every `code` your API can emit: for each code, it lists the HTTP status it pairs with, whether it is retryable, the canonical `title`, a description of the cause, the remedy a caller should apply, and the documentation URL the `documentation_url` member points at. It is the table a developer scans during integration to enumerate what can go wrong, and it is the table your own engineers consult before inventing a new code (so they reuse `not_found` instead of minting `resource_missing`, `entity_not_found`, and `does_not_exist` across three teams).

Why does this matter beyond tidiness? Because without a catalog, your error vocabulary grows by accretion — every engineer invents a code in the moment, and you end up with `card_declined`, `payment_declined`, and `declined_card` all meaning the same thing on three endpoints, which is the consistency failure of section 7 wearing a different hat. The catalog is what makes the namespace a *designed* vocabulary rather than an emergent pile. Practically, the catalog is just data — a YAML or JSON file checked into the repo — and it can drive everything downstream:

```yaml
errors:
  card_declined:
    status: 402
    retryable: false
    title: "The card was declined"
    cause: "The issuing bank declined the charge."
    remedy: "Ask the customer for a different payment method. Inspect decline_code for the specific reason."
    docs: "https://docs.example.com/errors/card-declined"
  validation_failed:
    status: 422
    retryable: false
    title: "The request body failed validation"
    cause: "One or more fields violated the schema or business rules."
    remedy: "Inspect the errors array; fix each field and resend."
    docs: "https://docs.example.com/errors/validation-failed"
  rate_limited:
    status: 429
    retryable: true
    title: "Too many requests"
    cause: "The client exceeded its rate limit."
    remedy: "Wait the duration in Retry-After, then retry with the same Idempotency-Key."
    docs: "https://docs.example.com/errors/rate-limited"
```

From this one file you can generate the reference-docs page (so the docs can never disagree with the implementation), the `PUBLIC_PROBLEMS` table the mapper uses (so the status and title are defined once), the per-code documentation pages the `documentation_url` links to, and the CI lint that asserts every emitted `code` exists in the catalog and every cataloged code is documented. This is the **[SDKs and reference docs](/blog/software-development/api-design/sdks-code-generation-and-reference-docs-developers-love)** idea applied to errors: when the catalog is the source of truth, the docs, the code, and the wire all stay in sync because they all derive from the same file. The payoff is the developer-experience win the whole series chases — a caller can read your error catalog and know, before writing a line, exactly what can fail, how to detect it, and what to do — which is precisely the goal of **[designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal)**.

A last operational note: the catalog also gives you a place to track *deprecation* of error codes. Just as response fields have a lifecycle, so do error codes — occasionally you want to retire a code or split one into two. Because the catalog is versioned and CI diffs it, you can mark a code `deprecated` with a sunset note, keep emitting it during the migration window, and communicate the change the same way you communicate any other contract change. Errors are part of the contract; they evolve under the same rules as the rest of it.

## 12. Case studies: how real APIs design errors

These are accurate as of writing; APIs evolve, so treat the *structure* as the lesson rather than any single field name.

**Stripe** is the canonical example of the two-level code design. A Stripe error object carries a `type` (a broad category such as `card_error`, `invalid_request_error`, `api_error`, `rate_limit_error`), a `code` (a stable machine string such as `card_declined`, `expired_card`, `incorrect_cvc`), and — specifically for declines — a `decline_code` (the issuer's reason, e.g. `insufficient_funds`, `lost_card`, `fraudulent`). They also return a human `message` and, for invalid-request errors, a `param` naming the offending field — the same `field`/`pointer` idea we built. The pattern to copy: a *categorical* code plus a *fine-grained* sub-code, both stable, with the human message kept separate and freely reworded. Stripe's docs page for each code is the `documentation_url` idea institutionalized.

**Twilio** assigns every error a numeric code (the well-known `Twilio error codes`, like `21211` for an invalid `To` phone number) and — importantly — each code resolves to a dedicated documentation page describing the cause and the fix. The lesson: a stable, enumerated, *documented* code namespace is a feature you maintain like any other part of the contract, and pairing each code with a doc page is a force-multiplier for developer experience.

**RFC 9457 (and its predecessor RFC 7807)** is the standard the whole post is built on. It is adopted across the industry — notably by Zalando's REST API guidelines (which mandate `problem+json` for all error responses) and by many government and enterprise API standards. The lesson: you do not need to invent an envelope; adopt the registered media type and extend it with your `code`, `trace_id`, and `errors` members rather than rolling a bespoke shape that no tooling recognizes. Frameworks increasingly ship `problem+json` support out of the box (Spring's `ProblemDetail`, ASP.NET Core's `ProblemDetails`), so the standard is also the path of least resistance.

**Google's API Improvement Proposals (AIP-193)** define a canonical error model with a numeric/enum `code`, a developer-facing `message`, and a `details` array for structured, typed error info (including localized messages and field violations). The lesson, again convergent: separate a stable machine code from a human message, and put structured per-field detail in a dedicated array — exactly the shape we built, arrived at independently by a different large API program.

The convergence across four independent large APIs is itself the strongest argument: stable machine `code` + freely-worded human `message` + structured per-field `details` + a documentation link + a request id is not one team's preference. It is what error design converges to once you take the two-audience principle seriously.

## 13. When to reach for this — and when not to

Error design has trade-offs like everything else. Be decisive about where the full machinery earns its keep.

**Reach for the full `problem+json` + stable-code + per-field-errors + trace-id design when:**

- You are building a **public or partner-facing API**, or any API with consumers you do not control. The richer the error contract, the cheaper every integration.
- Failures are **business-meaningful and varied** — payments, with their many decline reasons, are the textbook case. The more distinct failure modes, the more a stable `code` namespace pays off.
- You have **validation-heavy endpoints** where clients build forms against your schema. Field-level errors are transformative for those clients.
- You operate at a scale where **support volume** matters; a `trace_id` in every error slashes the time to resolve a ticket.

**Do not over-build when:**

- **Don't return `200` with an error body.** This is the cardinal sin — it breaks centralized error handling, caching, and alerting. Use the status code. (The opening vendor's bug in one line.)
- **Don't put the machine branch key in `detail`.** If a client has to substring-match your prose, you have not designed an error contract; you have designed a trap that springs the next time a message is reworded.
- **Don't add a per-field `errors` array to an endpoint with no body to validate.** A `GET /orders/{id}` that 404s needs `type`, `title`, `status`, `code`, `trace_id` — not an empty `errors: []`. Match the shape's richness to the failure's shape.
- **Don't leak internals to "help debugging."** A stack trace in production is a security hole, not a convenience. Log it server-side keyed by `trace_id`; expose nothing. Resist the "just in dev" exception — dev configs leak to prod.
- **Don't invent a bespoke envelope when `problem+json` exists.** Rolling your own shape costs you every piece of tooling that recognizes the registered media type, for no benefit.
- **Don't translate machine codes.** `code` and `pointer` are part of the contract and must be language-neutral; only the prose members are localized.
- **Don't blind-retry terminal errors, and don't omit `Retry-After` on retryable ones.** The two halves of the retry contract; skipping either causes double-charges or retry storms.

## 14. Key takeaways

- **An error is a two-audience contract.** It must tell a *program* what to do (a stable `code`) and a *human* what went wrong (a clear `detail`). Serving only one audience breaks the other.
- **Adopt RFC 9457 `problem+json`** with `Content-Type: application/problem+json`. Use the five core members — `type`, `title`, `status`, `detail`, `instance` — and extend it; do not invent a bespoke shape.
- **Carry a stable machine `code` separate from the HTTP status and from `detail`.** The status is coarse; `detail` is mutable prose; the `code` is the one frozen branch key. `card_declined` beats parsing a message every time.
- **Return field-level validation errors as an array**, one per failing field, each with its own `code` and JSON Pointer. Validate the whole body and report all failures at once.
- **Put a `trace_id` in every error**, echoed in a header too, logged server-side with full context. It is the bridge between the clean public error and your complete private logs.
- **Make messages actionable and never leak internals.** Say what to do and link to docs; keep stack traces, SQL, hostnames, and PII out of the body and in the log — enforce it with one central error mapper.
- **One error shape, every endpoint.** Consistency is what lets a caller write error handling once. Enforce it with a shared serializer, a `$ref`'d OpenAPI schema, and a linter.
- **Signal retryable vs terminal honestly** via the status family and `Retry-After`. `429`/`503` are retryable with a wait; most `4xx` (including `card_declined`) are terminal. Retry only idempotent operations on ambiguous `500`s.
- **For batches, report per-item outcomes** (`207` with embedded `problem+json` per item, or documented per-item statuses) so one bad row never hides behind a blanket status.

This is the part of the contract your callers live in most. Get it right and integration is a pleasure; get it wrong and you have signed up for years of double-charges, retry storms, and reverse-engineered lookup tables. For the whole picture — how errors compose with shapes, status codes, idempotency, versioning, and DX — see the **[API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2)**.

## Further reading

- **RFC 9457 — Problem Details for HTTP APIs** (2023): the canonical `problem+json` specification and the members described here. <https://www.rfc-editor.org/rfc/rfc9457>
- **RFC 7807 — Problem Details for HTTP APIs** (2016): the predecessor RFC 9457 obsoletes; useful for understanding the lineage. <https://www.rfc-editor.org/rfc/rfc7807>
- **RFC 9110 — HTTP Semantics**: the authoritative definitions of status codes and the `Retry-After` header. <https://www.rfc-editor.org/rfc/rfc9110>
- **RFC 6901 — JavaScript Object Notation (JSON) Pointer**: the `/field` syntax for locating values inside a JSON document, used in field-level errors. <https://www.rfc-editor.org/rfc/rfc6901>
- **Stripe API — Errors**: the two-level `code` / `decline_code` design and per-code documentation pages. <https://stripe.com/docs/api/errors>
- **Twilio — Error and Warning Dictionary**: an enumerated, documented numeric error-code namespace. <https://www.twilio.com/docs/api/errors>
- **Google AIP-193 — Errors**: Google's canonical structured error model (`code`, `message`, `details`). <https://google.aip.dev/193>
- **Within this series**: the **[intro hub](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems)**, **[status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx)**, **[designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming)**, and **[designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal)**.

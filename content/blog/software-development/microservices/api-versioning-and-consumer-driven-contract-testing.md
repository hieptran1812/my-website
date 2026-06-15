---
title: "API Versioning and Consumer-Driven Contract Testing: How Services Evolve Without Breaking Each Other"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The discipline that makes independent deployability actually safe: backward-compatibility rules, expand-and-contract migrations, the versioning strategies that work, and how consumer-driven contract testing catches a breaking change in CI seconds before it would have paged you at 3am."
tags:
  [
    "microservices",
    "api-versioning",
    "contract-testing",
    "pact",
    "schema-registry",
    "distributed-systems",
    "software-architecture",
    "backend",
    "protobuf",
    "backward-compatibility",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-1.webp"
---

The ShopFast order service had a field named `total`. It had been named `total` since the first commit, back when ShopFast was a monolith and `total` meant "the number you charge the card." Two years and eleven services later, `total` was ambiguous: it could mean the subtotal before tax, the subtotal after tax, the grand total including shipping, or the grand total after the loyalty discount, and different parts of the company had quietly assumed different answers. So a senior engineer on the order team did the responsible thing. They renamed it to `grand_total` and added a separate `subtotal` field, shipped a clean PR with good tests, watched the order service's own test suite go green, and deployed on a Tuesday afternoon.

By Tuesday evening the mobile checkout screen was showing every order total as `\$0.00`. The mobile BFF — a separate service, owned by a separate team, that the order team did not talk to often — had been reading the `total` field for eighteen months. When `total` disappeared from the JSON, the BFF's deserializer did what most deserializers do with a missing field: it filled in the zero value and moved on, silently. No exception, no alert, just a stream of orders that all appeared to cost nothing. Meanwhile the shipping service, which used `total` to decide whether an order qualified for free shipping, started giving everybody free shipping, because `0` is less than the threshold. Neither failure showed up in the order service's tests, because the order service's tests do not know the BFF or the shipping service exist. The break lived in the *gap between services* — exactly the place no single team's test suite looks.

This is the central, under-taught problem of microservices. The whole promise of the architecture is independent deployability: each team ships its own service on its own schedule without a coordinated release train. But independent deployability is only safe if a producer can change its API without breaking consumers it cannot deploy in lockstep — consumers it may not even know it has. You cannot atomically update all the services in a fleet; deploys roll out one service at a time, and for some window the old consumer is talking to the new producer or vice versa. If your API evolution discipline is "rename the field and pray," you have not built microservices. You have built a distributed monolith with a longer fuse, where every change is a coordination problem and every coordination problem eventually becomes an incident.

![A service topology diagram showing the ShopFast order service renaming a field and that change rippling out to the mobile BFF, the shipping service, and an unknown analytics consumer, with no atomic deploy leading to a runtime break found in production](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-1.webp)

This post is the safety net for everything in this track. By the end you will be able to: classify any proposed API change as safe or breaking and know *why*; run the expand-and-contract migration that turns even a breaking change into a sequence of safe ones; choose between URI, header, media-type, and compatibility-only versioning with eyes open to each one's cost; replace a slow, flaky end-to-end test suite with consumer-driven contract tests that catch the ShopFast break *in CI, in seconds, before deploy*; and configure a schema registry so your event streams evolve as safely as your synchronous APIs. We will keep returning to ShopFast — the order service that wants to rename `total` — and we will watch a Pact contract catch the exact break that took down checkout, in a pull request, with a red X next to it.

## Why evolving an API is genuinely hard in a fleet

Start with the constraint that makes everything else necessary: **you cannot deploy all your services at the same instant.** In a monolith, a method rename is atomic. The caller and the callee are compiled together, deployed together, and the moment the new binary is live, both halves of the change are live. There is no window in which the caller has the old expectation and the callee has the new behavior, because there is only one deploy unit. The compiler is your contract test, and it runs at build time.

In a fleet, none of that holds. The order service and the mobile BFF are separate deploy units with separate pipelines. When the order team deploys the renamed field, the BFF is still running the binary that expects the old field — and it will keep running that binary for minutes, hours, or until the BFF team happens to deploy next, which might be next week. For that entire window, a new producer is serving an old consumer. This is the *rolling-deploy window*, and it is not an edge case; it is the normal state of a continuously deployed system. There is always some pair of services where one is ahead of the other.

It gets worse, and the worse part is the part juniors miss. **A producer often does not know who its consumers are.** When you publish an HTTP API or emit an event, you publish it to a network, not to a known list. The mobile BFF reads your `total` field; so does the shipping service; so does a nightly analytics job someone wrote a year ago and forgot to document; so does a partner integration you have never met. The producer team's mental model is "my three known consumers," and reality is "my three known consumers plus an unknown number of others." Every breaking change is therefore a change with an unknown blast radius. You are not asking "will this break my consumer?" You are asking "will this break *all* the consumers, including the ones not on my list?" — and you cannot answer that by reading your own code.

Add the [eight fallacies of distributed computing](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) on top. The network reorders and delays things, so even within a single logical request, the version of the producer that served the first call may differ from the one that serves a retry. Serialization means the consumer reconstructs the producer's data from bytes, and reconstruction is where version skew bites: a renamed field is just *absent* in the bytes, and an absent field is indistinguishable, to most deserializers, from a field that was explicitly set to its default. The producer did not send an error. It sent a perfectly valid message that happened to be missing the one thing the consumer needed, and the consumer cheerfully filled in a zero.

So the problem statement is precise. **Evolve a producer's API such that, during an arbitrarily long window where some consumers run the old expectation and some run the new, nothing breaks — without knowing who all the consumers are.** Everything in this post is a tool for satisfying that statement. The compatibility rules tell you which changes are safe by construction. Versioning gives you an escape hatch when a change cannot be made safe. Expand-and-contract sequences a breaking change into safe steps. And contract testing replaces "I don't know my consumers" with "my consumers told me exactly what they expect, and CI checks it."

## The one rule: additive changes are safe, destructive changes break

Before any versioning strategy, before any tool, internalize the rule that prevents the majority of API-evolution incidents: **adding is safe, removing and changing are not.** A consumer reads the fields it knows about and ignores the rest. So if you *add* a new optional field, every existing consumer keeps working — it never asked for the new field, so it does not notice it. If you *remove* a field, every consumer that read it now gets nothing. If you *rename* a field, that is a remove plus an add, so it breaks every reader of the old name. If you *change a field's type* — `string` to `number`, a flat field to a nested object — you break every consumer that parses it the old way. If you *tighten* a constraint — make an optional field required, narrow an enum, shrink a string's max length — you break callers that were sending the now-illegal value.

![A before and after comparison contrasting a breaking change that renames the total field and removes a required field against an additive change that adds grand_total while keeping the old total field optional so old readers are unaffected](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-2.webp)

Write it out as a checklist, because in a code review this is the lens you apply to every diff. **Safe (backward-compatible) changes:** adding a new optional field to a response; adding a new optional field to a request (the server defaults it); adding a new endpoint or RPC method; adding a new enum value *that old consumers can tolerate* (more on the caveat below); making a required request field optional; relaxing a validation constraint. **Breaking changes:** removing or renaming any field; changing a field's type or semantic meaning; making an optional request field required; removing an enum value or endpoint; adding a *required* request field (old clients do not send it); changing the meaning of an existing field without changing its name (the silent killer — same `total`, now post-discount instead of pre-discount).

Two terms are worth defining because they keep the rule precise. **Backward compatibility** means a *new producer* works with *old consumers* — you upgraded the server and the existing clients are fine. **Forward compatibility** means an *old producer* works with *new consumers* — you upgraded a client to expect a field the server has not started sending yet, and the client tolerates its absence. Most of the time when people say "compatible" they mean backward, but forward compatibility matters enormously the moment you have a rolling deploy, because during the window the new consumer really is talking to an old producer. A field you add must be optional precisely so the new consumer survives talking to the old producer that does not yet emit it.

There is one beautiful corollary that the best teams lean on hard: **if you only ever make backward- and forward-compatible changes, you never need to version your API at all.** The "version 2" only exists because someone needs to make a breaking change. If you can avoid breaking changes — and the compatibility rules plus expand-and-contract let you avoid almost all of them — you can run a single, unversioned, ever-evolving API for years. This is the secret behind APIs like Stripe's that famously almost never break clients: not heroic versioning machinery, but a near-religious commitment to additive-only evolution, with the rare genuine break handled by a dated version pin.

### The tolerant reader: Postel's law made concrete

The rule above protects consumers *if* consumers behave well, and the way a consumer behaves well is by being a **tolerant reader**. The principle is Postel's law, from the early internet: "be conservative in what you send, be liberal in what you accept." A tolerant reader does not demand that the producer's payload match its expectations exactly. It binds only the fields it actually uses, ignores fields it does not recognize, and defaults fields that are missing instead of throwing. The opposite — a *strict reader* that fails to parse if the payload has an unexpected extra field, or that requires every documented field to be present — turns every additive producer change into a consumer break, which defeats the entire point of additive changes being safe.

![A vertical stack showing the tolerant reader layers from binding only used fields through ignoring unknown fields and defaulting missing ones down to loose deserialization, contrasted at the bottom with a strict schema parse that fails on drift](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-5.webp)

Concretely, here is the difference in a JSON consumer. A strict reader configures its deserializer to reject unknown properties, so when the producer *adds* a field — a safe change! — the consumer crashes. A tolerant reader ignores them.

```java
// STRICT reader — a safe additive producer change breaks this consumer.
// Jackson with FAIL_ON_UNKNOWN_PROPERTIES = true (often the default).
ObjectMapper strict = new ObjectMapper()
    .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, true);
Order o = strict.readValue(json, Order.class); // throws if producer added a field

// TOLERANT reader — survives the producer adding fields it does not know.
ObjectMapper tolerant = new ObjectMapper()
    .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
    .configure(DeserializationFeature.READ_UNKNOWN_ENUM_VALUES_AS_NULL, true);
Order o = tolerant.readValue(json, Order.class); // ignores unknown fields
```

The tolerant reader is not just defensive hygiene; it is the thing that makes the additive-change rule *true in practice*. If half your consumers are strict readers, then "adding a field is safe" is a lie for your fleet, and you are back to coordinating deploys. So part of the discipline is enforcing tolerant reading as a fleet-wide convention: default deserializers to ignore unknowns, never bind a field you do not use, and write consumers that default missing fields rather than assume presence. The enum caveat from earlier lives here too: adding an enum value is safe only if old consumers treat unknown enum values gracefully (map to a default, route to a fallback) rather than crashing on the value they have never seen. That is a tolerant-reader property, not a producer property.

## Field evolution in Protobuf and gRPC: the rules baked into the format

For internal east-west traffic many teams use [gRPC and Protobuf](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis), and Protobuf is interesting because its wire format was *designed* around evolution. A Protobuf message is encoded as a series of `(field number, wire type, value)` tuples. The field's *name* is not on the wire at all — only its *number*. This is the key to everything: renaming a field in your `.proto` is a no-op on the wire, because the number is unchanged, so a rename is — uniquely among formats — backward compatible in Protobuf. What you must never touch is the *number*.

The rules fall out of the encoding. **You may** add a new field with a fresh number (old code ignores unknown field numbers, the textbook additive change). **You may** rename a field freely (the name is local). **You must never** reuse a field number for a different field, change a field's number, or change a field's type in an incompatible way (`int32` to `string` reinterprets the bytes). And when you *delete* a field, you must **reserve** its number and name so nobody on the team can ever accidentally reuse them — because if a future engineer assigns number 4 to a new `string promo_code` while some old producer is still emitting number 4 as an `int64 discount_cents`, the consumer will decode garbage with no error. Here is the order message evolving safely across two versions:

```protobuf
// order.proto — VERSION 1
syntax = "proto3";
package shopfast.order.v1;

message Order {
  string order_id = 1;
  int64 total_cents = 2;          // the ambiguous "total"
  string currency = 3;
  int64 discount_cents = 4;       // we will remove this field later
  repeated LineItem items = 5;
}
```

```protobuf
// order.proto — VERSION 2, evolved SAFELY
syntax = "proto3";
package shopfast.order.v1;        // same package: no breaking version bump needed

message Order {
  string order_id = 1;
  int64 subtotal_cents = 2;       // RENAME of total_cents — same number 2, free on the wire
  string currency = 3;
  reserved 4;                     // discount_cents removed: number reserved forever
  reserved "discount_cents";      //   and its name reserved so it can never be reused
  repeated LineItem items = 5;
  int64 grand_total_cents = 6;    // NEW field, fresh number — old consumers ignore it
  string promo_code = 7;          // NEW field, fresh number
}
```

Notice what the `reserved` keyword buys you: it converts a footgun into a compile error. If anyone later writes `int64 something = 4;`, `protoc` refuses to compile. That single line is the difference between "we removed a field cleanly" and "we removed a field and a junior reused its number eight months later and we spent a day debugging silently corrupted orders." In `proto3`, also note that there is no `required` keyword — every field is effectively optional, which is a deliberate decision: `required` is unremovable (you can never relax it without breaking everyone), so Protobuf removed it from the language entirely. The Google API design guide is blunt about this: treat every field as optional, validate in code, and never mark anything required at the schema level.

For gRPC specifically, the same rules cover the methods, not just the messages: you may add new RPC methods to a service freely (old clients do not call them), but you must not remove or rename an existing method, change its request or response message types incompatibly, or change its streaming-ness (turning a unary call into a stream is a break). Because the package name (`shopfast.order.v1`) carries the major version, a genuinely incompatible redesign goes in a *new* package, `shopfast.order.v2`, and you run both server-side until consumers migrate — which is just expand-and-contract at the package level. We will get there.

## Versioning strategies, and the cost of each

Sometimes a change genuinely cannot be made compatible — you are restructuring a resource, splitting one endpoint into two, changing semantics that have no additive path. Now you need versioning, and there are four strategies people actually use. They are not equivalent; each one trades client clarity against operational cost, and the worst mistake is picking one by taste rather than by who consumes the API.

![A decision matrix comparing URI path versioning, header versioning, media-type versioning, and compatibility-only across client visibility, HTTP caching, routing ease, whether it avoids dual code paths, and what each is best for](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-3.webp)

**URI path versioning** puts the version in the path: `GET /v2/orders/123`. It is the most visible and the easiest to route — your gateway or load balancer can send `/v1/*` and `/v2/*` to different deployments, and a developer reading a log line or a curl command instantly knows which version they hit. It caches cleanly because the URL is the cache key. The cost is purist discomfort (a URL is supposed to identify a *resource*, and `/v1/orders/123` and `/v2/orders/123` are arguably the same resource in two representations) and, more practically, that it bumps the version for the *whole API surface* even when one endpoint changed. It is the right default for **public** APIs and APIs consumed by clients you do not control, precisely because its bluntness is honesty: GitHub, Twilio, and Stripe-style dated URLs all live here.

**Header versioning** passes the version in a custom or standard header: `Accept-Version: 2` or a vendor header. The URL stays clean and RESTfully stable, and you can version per-request without touching the path. The cost is that the version is *invisible* — it does not show up in a browser address bar, a basic log line, or a naive curl, which makes debugging and cache configuration harder (you must set `Vary: Accept-Version` or your cache will serve a v1 response to a v2 request). It suits **internal** APIs where the consumers are your own services and the loss of visibility is acceptable.

**Media-type (content negotiation) versioning** is the REST-purist answer: `Accept: application/vnd.shopfast.order.v2+json`. The version rides in the standard `Accept` header as part of the media type, so the URL identifies the resource and the header identifies the representation — theoretically the cleanest model. In practice it shares header versioning's invisibility problems and adds a steeper learning curve; few teams outside the hypermedia-REST community run it well. It is defensible for a sophisticated public API with a strong REST culture, and overkill almost everywhere else.

**Compatibility-only ("no versioning")** is the strategy the best teams reach for first: never make a breaking change, so you never need a version. Every change is additive, every consumer is a tolerant reader, and the API just evolves. There is no `/v2`, no header, no dual code path to maintain. The cost is *discipline* — you give up the ability to ever clean up a messy field by removing it, and you accumulate some deprecated-but-still-present fields. But that cost is almost always smaller than the cost of running two versions, and crucially it is the only strategy that does not double your maintenance surface. The other three all share a hidden tax that the matrix makes explicit: **two versions means two code paths to maintain, test, and keep secure, until you can kill the old one.** A `/v2` you never retire is a `/v1` you maintain forever.

The honest recommendation: default to compatibility-only, use expand-and-contract (next section) to handle the breaking changes that look unavoidable, and reach for explicit versioning only for the genuinely irreducible breaks on public APIs where you cannot control the clients. For internal services behind your own gateway, you should almost never see a `/v2`.

## Expand and contract: how to make a breaking change one safe step at a time

Here is the technique that lets you rename `total` to `grand_total` without breaking the BFF, the shipping service, or the analytics job you forgot about. It goes by several names — **expand and contract**, **parallel change**, or in database circles the **expand/migrate/contract** pattern — and it is the single most valuable migration recipe in distributed systems. The idea is to never have a moment where the old shape and the new shape do not coexist. You *expand* the API to support both old and new simultaneously, *migrate* every consumer to the new shape on its own schedule, then *contract* by removing the old shape only after you have proven nobody uses it.

![A six-event timeline of the expand-and-contract rollout starting with adding grand_total and emitting both fields, the provider shipping both, the BFF and shipping services migrating to the new field, telemetry confirming zero old readers, and finally removing the old total field](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-4.webp)

The crucial property is the *deploy ordering*. The expand step must ship and be fully rolled out **before** any consumer starts using the new field, and the contract step must ship **after** every consumer has stopped using the old field and you have telemetry to prove it. Get the ordering wrong — contract before consumers migrate — and you have just shipped the original breaking change with extra steps. Let me make this concrete with numbers.

#### Worked example: the ShopFast field rename as an expand-and-contract rollout

The order service (provider) has two known consumers — the mobile BFF and the shipping service — plus an unknown number of others. We want to rename `total` to `grand_total`. Here is the rollout, day by day, with the deploy ordering that keeps it safe the whole way.

**Day 0 — Expand the provider.** The order team changes the order service to emit *both* fields: `total` (the old name) and `grand_total` (the new name), set to the same value. The response now carries redundant data. No consumer changes. This is a purely additive change — `grand_total` is a new optional field — so it is safe by the rule above. Deploy it and let it roll out to 100% of order-service instances. Critically, until this is fully rolled out, no consumer may depend on `grand_total`, because during the rollout some order-service instances are still on the old binary and do not emit it. Forward compatibility says the consumer would tolerate its absence, but we are not even asking consumers to read it yet.

```python
# order_service/serializers.py — EXPAND step: emit both fields.
def serialize_order(order):
    grand_total = order.subtotal + order.tax + order.shipping - order.discount
    return {
        "order_id": order.id,
        "subtotal": order.subtotal,
        "grand_total": grand_total,   # NEW canonical field
        "total": grand_total,         # OLD field, kept identical during migration
        "currency": order.currency,
        "items": [serialize_item(i) for i in order.items],
    }
```

**Day 1 — Provider fully rolled out.** Every order-service instance now emits both fields. We have entered the *parallel* state: old consumers read `total` and are happy; new consumers can start reading `grand_total` and will be happy. This is the window where migration is safe in both directions.

**Day 3 — Migrate the BFF.** The BFF team changes the mobile BFF to read `grand_total` instead of `total`, and deploys on their own schedule. They do not need to coordinate with the order team — the order service already supports both. One consumer migrated.

**Day 7 — Migrate the shipping service.** The shipping team does the same: read `grand_total`, deploy. Two known consumers migrated.

**Day 14 — Verify with telemetry before you dare contract.** This is the step juniors skip and seniors never skip. The order team has been emitting a metric every time a request reads the old field — more on how below — and they query it: how many requests in the last 7 days read `total`? The answer needs to be **zero**, sustained, across a full business cycle (a week that includes the Monday batch jobs and the weekend traffic peak). If the number is `0`, the unknown consumers either never existed or have already moved. If it is `4,200/day and falling`, you wait, and you go find that consumer. Suppose telemetry shows it has been a flat `0` for 7 days.

**Day 21 — Contract.** Now, and only now, the order team removes the `total` field from the serializer (and in Protobuf, reserves its number). Deploy. The migration is complete. The whole thing took three weeks of calendar time and *zero* coordinated deploys — every team moved on its own schedule, and at no instant was any consumer broken.

Total breaking changes shipped to production: **zero.** Total incidents: **zero.** Compare that to the original "rename and deploy on Tuesday," which shipped one breaking change and caused one checkout outage. The expand-and-contract version is more steps and more calendar time, and it is *unambiguously cheaper* once you price in the incident.

### Telemetry on the old field: how Day 14 actually works

The Day-14 verification only works if you instrumented the deprecated field. The cheapest mechanism is a counter that increments whenever a request reads (or, for a response field, whenever a client *would have* read) the old field. For a response field you cannot directly observe the client's read, so you proxy it: emit a deprecation signal whenever a request comes from a client version known to read the old field, or — better — have consumers self-report. The most robust pattern is the **deprecation header** on the request side and a usage counter on any old request field. Here is the server emitting both a deprecation signal and a usage metric:

```python
# order_service/middleware.py — telemetry + deprecation signaling.
from prometheus_client import Counter

deprecated_field_reads = Counter(
    "shopfast_deprecated_field_reads_total",
    "Reads of a deprecated request field, by field and caller",
    ["field", "caller"],
)

def handle_request(req, resp):
    # If a client still SENDS the old request field, count it by caller.
    if "total" in req.json:
        caller = req.headers.get("X-Caller-Service", "unknown")
        deprecated_field_reads.labels(field="total", caller=caller).inc()

    # Tell every client that the old field is on the way out (RFC 8594-style).
    resp.headers["Deprecation"] = "true"
    resp.headers["Sunset"] = "Sat, 30 Aug 2026 23:59:59 GMT"
    resp.headers["Link"] = '</docs/migrations/grand-total>; rel="deprecation"'
```

Now Day 14 is a Grafana query: `sum(rate(shopfast_deprecated_field_reads_total{field="total"}[7d]))`. If it is zero, contract. If it is not, the `caller` label tells you exactly which service to go talk to — including, often, the "unknown" consumer you did not know about, now revealed by its `X-Caller-Service` header or its source IP. The `Deprecation` and `Sunset` headers (the latter standardized in RFC 8594) are the polite, machine-readable way to tell every client "this is going away on this date," so a consumer's own monitoring can flag that it is consuming a soon-to-be-removed field. This is deprecation as a *process with a deadline and evidence*, not a wiki page nobody reads.

## The testing problem: why end-to-end tests are the wrong safety net

So far we have rules and a migration recipe. But rules require discipline, and discipline fails — someone will eventually ship a breaking change by accident, exactly like the original ShopFast Tuesday. You need an automated gate that catches it *before* deploy. The obvious idea is the one most teams reach for first and regret: **end-to-end integration tests.** Spin up the order service, the BFF, the shipping service, the payment service, the database, the message broker — the whole fleet — in a test environment, run a checkout, and assert it works. If the order team's field rename breaks the BFF, the e2e suite goes red.

It does work, in the sense that it catches the bug. It is also, at fleet scale, a productivity disaster, and understanding *why* is the key to understanding contract testing. The problems compound:

**It is slow.** Standing up twelve services with their databases, seeding data, and running a realistic flow takes minutes — often 20 to 40 minutes for a meaningful suite. That latency sits on the critical path of every merge. A team that deploys twenty times a day cannot afford a 35-minute gate on each one; the math alone (20 × 35 = 700 minutes of serialized CI per day) forces them to either batch deploys — killing independent deployability — or run the suite less often, killing its value.

**It is flaky.** With twelve services, two databases, and a message broker all live in one environment, the number of things that can be transiently wrong is enormous: a service is slow to start, a port collides, a test's data leaks into another test, the broker drops a message under load. Flakiness in the single digits of percent sounds tolerable until you do the arithmetic: an 8%-flaky suite that gates every merge means roughly one in twelve merges fails for no real reason, the engineer reruns it, loses fifteen minutes, and — worst of all — *learns to ignore red builds*, which is how a real failure eventually sails through.

**Nobody owns it.** This is the deep problem. The e2e suite tests the interaction between the order service and the BFF, but it lives in… whose repo? When it breaks, whose pager fires? The order team says "the BFF assertion failed, that's the BFF team's test." The BFF team says "the order service changed, that's the order team's bug." The shared e2e suite is an orphan — a tragedy of the commons where the test catches integration bugs but no team feels responsible for keeping it green, so it rots. Within a year it is either disabled or permanently yellow.

**It scales quadratically with the fleet.** Each new service can interact with every other, so the surface a true e2e suite must cover grows like the number of service *pairs*, not the number of services. Twelve services is sixty-six possible pairs; thirty services is four hundred thirty-five. You cannot test the integration matrix of a real fleet end to end. You will test a thin, brittle slice of it and tell yourself it is coverage.

The deepest reason e2e is the wrong tool, though, is conceptual: **an e2e test couples the deploy of one service to the test environment of every other.** To know whether you can safely deploy the order service, you must spin up everyone else. That is the distributed monolith again, wearing a CI badge. You wanted to know one thing — "does my change still satisfy what my consumers expect?" — and the e2e suite answered it by reassembling the entire universe. There is a far more surgical way to ask exactly that question.

## Consumer-driven contract testing: the surgical safety net

The insight behind **consumer-driven contract testing (CDC)** is to test the *contract* between two services in isolation, without running both of them at once. The contract is the precise set of expectations a consumer has about a provider: "when I call `GET /orders/123`, I expect a 200 with a body that has a `grand_total` number field and an `order_id` string field." If we can capture that expectation as a machine-readable artifact, we can do two things separately: verify that the consumer's code actually relies only on what it claims (by running the consumer's tests against a *mock* provider that returns exactly the contracted shape), and verify that the real provider actually satisfies the contract (by replaying the contracted requests against the real provider and checking the responses match). Neither check needs both services live. Each runs in its own pipeline, in seconds.

![A flow diagram showing the mobile BFF and shipping service consumers each writing a pact, both publishing to a pact broker, the provider verifying by replaying the pacts, then a can-i-deploy gate deciding whether to deploy or block](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-6.webp)

It is called *consumer-driven* because the consumer writes the contract first, from its real needs. This flips the usual direction: instead of the provider publishing a spec and hoping consumers conform, each consumer declares "here is exactly what I use," and the provider is held to the union of what its consumers actually need — no more, no less. A field no consumer's contract mentions is a field the provider is free to remove. That is a profound capability: contracts make the *unknown consumer problem* tractable, because the broker becomes the registry of who depends on what.

The two dominant tools are **Pact** (polyglot, the de-facto standard, with a broker that stores contracts and gates deploys) and **Spring Cloud Contract** (JVM-centric, contracts written in Groovy/YAML, generates provider stubs). The mechanics differ but the shape is identical. Let me walk through the ShopFast example end to end in Pact.

### Step 1: the consumer writes its expectations as a test

The mobile BFF team writes a test in their own repo. It does not call the real order service. It runs against a Pact *mock server* that the test configures: "for this request, return this response." Running the test does two jobs — it verifies the BFF's own code handles the contracted response correctly, and it *generates a pact file* recording the interaction.

```javascript
// bff/test/order-service.pact.test.js — CONSUMER side (the mobile BFF).
const { PactV3, MatchersV3 } = require("@pact-foundation/pact");
const { like, integer, string } = MatchersV3;
const { getOrder } = require("../src/order-client");

const provider = new PactV3({
  consumer: "mobile-bff",
  provider: "order-service",
});

describe("mobile BFF -> order service", () => {
  it("reads grand_total from an order", () => {
    provider
      .given("order 123 exists")
      .uponReceiving("a request for order 123")
      .withRequest({ method: "GET", path: "/orders/123" })
      .willRespondWith({
        status: 200,
        headers: { "Content-Type": "application/json" },
        // We assert ONLY the fields the BFF actually uses. Note: NOT "total".
        body: like({
          order_id: string("123"),
          grand_total: integer(4299),
          currency: string("USD"),
        }),
      });

    return provider.executeTest(async (mockServer) => {
      const order = await getOrder(mockServer.url, "123");
      expect(order.grandTotal).toBe(4299); // BFF's own logic, verified
    });
  });
});
```

Two things make this powerful. First, the BFF asserts *only the fields it uses* — `order_id`, `grand_total`, `currency`. It deliberately does not mention `total`, because after migration it does not read `total`. The contract therefore says "the order team is free to remove `total`; I don't care." Second, the matchers (`like`, `integer`, `string`) match by *type and structure*, not exact value, so the contract is about the *shape* of the response, not the specific order. Running this test produces a pact file, a JSON document describing the interaction.

### Step 2: the consumer publishes the pact to the broker

CI for the BFF publishes the generated pact to the Pact Broker, tagged with the consumer's version and branch. The broker is now the source of truth for "what the mobile BFF expects from the order service."

```bash
# bff CI — publish the contract the BFF just generated.
pact-broker publish ./pacts \
  --consumer-app-version "$GIT_SHA" \
  --branch "$GIT_BRANCH" \
  --broker-base-url "$PACT_BROKER_URL" \
  --broker-token "$PACT_BROKER_TOKEN"
```

### Step 3: the provider verifies every consumer's contract

Now the order service's CI, in the order team's repo, runs *provider verification*. It pulls every pact published against `order-service` from the broker — the BFF's pact, the shipping service's pact, and any others — spins up only the real order service (no consumers, no other services), replays each contracted request, and checks the real responses satisfy the contracted shapes. This is where the breaking change gets caught.

```java
// order-service provider verification (Pact JVM + JUnit 5).
@Provider("order-service")
@PactBroker(url = "${PACT_BROKER_URL}", authentication = @PactBrokerAuth(token = "${PACT_BROKER_TOKEN}"))
class OrderServicePactVerificationTest {

  @BeforeEach
  void setTarget(PactVerificationContext context) {
    context.setTarget(new HttpTestTarget("localhost", port));
  }

  @State("order 123 exists")
  void order123Exists() {
    orderRepository.save(new Order("123", 4299, "USD")); // set up provider state
  }

  @TestTemplate
  @ExtendWith(PactVerificationInvocationContextProvider.class)
  void verifyPacts(PactVerificationContext context) {
    context.verifyInteraction(); // replays each consumer's contracted request
  }
}
```

Here is the payoff. Suppose the order team, in a PR, deletes `grand_total` (or renames it back, or removes `total` before the BFF migrated — pick your break). Provider verification pulls the BFF's pact, which expects `grand_total`, replays `GET /orders/123`, gets a response *without* `grand_total`, and fails: **"BFF contract violated: expected field grand_total, got none."** That failure happens in the order team's CI, on the PR, in seconds, against only the order service — no BFF deploy, no e2e environment. The exact break that took down checkout on Tuesday is now a red X next to a pull request, with a message naming the consumer it would have broken. The order team learns who depends on the field *before* they delete it.

### Step 4: the can-i-deploy gate

Verification proves the provider satisfies the *currently published* contracts. But there is a subtler race: the BFF might publish a *new* contract (expecting some new field) that the deployed order service does not yet satisfy. The **can-i-deploy** gate closes this. Before any service deploys, it asks the broker: "given the versions of all my consumers and providers, is it safe for *this* version of *me* to go to *this* environment?" The broker checks the full compatibility matrix of verified contracts and answers yes or no. This is the gate that makes independent deployability *provably* safe rather than hopefully safe.

```bash
# In ANY service's deploy pipeline, before promoting to production:
pact-broker can-i-deploy \
  --pacticipant "order-service" \
  --version "$GIT_SHA" \
  --to-environment production \
  --broker-base-url "$PACT_BROKER_URL" \
  --broker-token "$PACT_BROKER_TOKEN" \
  --retry-while-unknown 12 --retry-interval 10

# Exit code 0 = every consumer contract is verified against this version -> deploy.
# Exit code 1 = a consumer expects something this version does not provide -> BLOCK.
```

If `can-i-deploy` returns non-zero, the pipeline halts and the deploy never happens. The order service literally *cannot* ship a version that violates a published consumer contract, because the gate refuses. That is the difference between "we have a rule about not breaking consumers" and "the system mechanically prevents breaking consumers." The `--retry-while-unknown` flag handles the timing where a provider verification is still in flight — it waits rather than guessing.

There is one subtlety that trips up teams adopting Pact, and it is worth stating plainly because it is the difference between a contract suite that protects you and one that lulls you. A contract test only protects against breaking a field *if some consumer's pact mentions that field.* If the BFF's pact never asserted `total` — because the BFF author only wrote a pact for the *new* `grand_total` path — then deleting `total` will pass verification, even though some *other* consumer (the shipping service, the forgotten analytics job) still needs it. The protection is exactly as complete as your consumers' contracts are. This is why the broker's list of pacticipants matters operationally: part of adopting contract testing is making sure *every* real consumer has published a pact, so the union of contracts actually covers the provider's surface. A provider with three live consumers but only one published pact has a 33%-covered contract and a false sense of safety. The senior habit is to periodically reconcile the broker's known consumers against the provider's *actual* callers (from access logs or tracing) and chase down any caller without a pact — those are your unknown consumers, and they are precisely the ones that cause outages.

### Provider states and the data-setup problem

One more piece of the provider-verification machinery deserves attention because juniors get stuck on it. Each contracted interaction often depends on a *precondition* — "order 123 exists," "the user has an active subscription," "the inventory is zero." The consumer's pact records this as a **provider state** (the `.given("order 123 exists")` line in the consumer test). During verification, the provider must *set up* that state before replaying the request, which is what the `@State("order 123 exists")` handler does — it seeds the repository so the replayed `GET /orders/123` returns the expected shape. This keeps verification hermetic: the provider does not need a populated production database, just a handler per state that creates exactly the data the contract assumes. It is the same principle as a unit test's `setUp`, scoped to the contract. Getting these states right is most of the work of adopting Pact on the provider side, and the failure mode when they are wrong is a verification that fails for a *data* reason ("no order 123") rather than a real *contract* reason ("missing field") — so a senior reviews state handlers as carefully as the contracts themselves.

## Expand and contract at the data layer: the field migration underneath the API

The API rename you just sequenced safely usually sits on top of a *database* change, and the same expand-and-contract recipe applies one layer down — in fact it is where the pattern was born, as the database refactoring known as parallel change. If the order service stores `total` in a `total_cents` column and you want the canonical name to be `grand_total_cents`, you cannot just `ALTER TABLE ... RENAME COLUMN` in one migration, because during the rolling deploy the old binary reads `total_cents` while the new binary reads `grand_total_cents`, and a single column cannot be named both at once. So you expand the *schema* first, backfill, dual-write through the transition, then contract — exactly mirroring the API rollout.

```sql
-- EXPAND (migration 1): add the new column, nullable, no rename yet.
ALTER TABLE orders ADD COLUMN grand_total_cents BIGINT NULL;

-- BACKFILL (migration 2, batched to avoid a long lock on a big table):
UPDATE orders SET grand_total_cents = total_cents
WHERE grand_total_cents IS NULL AND id BETWEEN :lo AND :hi;  -- run in id ranges

-- During the transition the application DUAL-WRITES both columns on every write,
-- so neither the old binary (reads total_cents) nor the new (reads grand_total_cents)
-- ever sees a stale value. Only after the new binary is 100% rolled out AND the
-- backfill is complete do we stop writing the old column.

-- CONTRACT (migration 3, only after telemetry shows zero readers of total_cents):
ALTER TABLE orders DROP COLUMN total_cents;
```

The discipline is identical to the API case and the deploy ordering is just as load-bearing: expand the schema before any code reads the new column, dual-write during the window so both binaries see correct data, backfill in batches so you do not hold a table-long lock (a single `UPDATE` on a hundred-million-row table is an outage of its own), and drop the old column only after proving nothing reads it. The mechanics of schema migration in a distributed system — online DDL, backfill batching, the dual-write window — get their full treatment in the database series' [change-data-capture and outbox](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) and [partitioning](/blog/software-development/database/database-partitioning-and-sharding) deep-dives; what matters here is recognizing that the *same* expand/migrate/contract shape runs at the API layer and the storage layer, and a clean field rename coordinates both. The contract testing above gates the API surface; for the storage layer the gate is the migration's own pre-checks plus the backfill's idempotency, which connects to [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe).

## Deprecation and sunsetting: retiring an API on a deadline, with evidence

Once you have expanded and migrated, the *contract* step — actually removing the old thing — is the part organizations are worst at, because removing a field feels risky and the safe-feeling choice is to leave it forever. That is how an API accretes dozens of zombie fields that everyone is afraid to touch, each one a small tax on every reader and a small lie in the documentation. A senior treats deprecation as a *managed lifecycle with a deadline and evidence*, not a permanent purgatory.

The lifecycle has four stages, and each has a concrete artifact. **Announce:** publish the deprecation in the schema and to consumers — for HTTP, the `Deprecation: true` header and a `Sunset` date (RFC 8594) on every response that includes the field; for events, a `doc` annotation and a changelog entry; for gRPC, the `[deprecated = true]` option on the field. **Measure:** instrument usage so you know who still reads it (the counter from earlier, ideally labeled by caller). **Migrate:** give consumers a real window — measured in weeks for internal APIs, often months for public ones — and chase the laggards using the telemetry. **Remove:** contract only when usage hits zero and stays there across a full business cycle. The `Sunset` date is a commitment, not a suggestion; the value of publishing a date is that it converts an open-ended "someday" into a deadline that consumer teams can plan against, and that your own monitoring can alarm on ("you are still consuming a field that sunsets in 14 days").

#### Worked example: deciding when it is safe to remove a deprecated field

The order team announced `total`'s deprecation 60 days ago with a `Sunset` date of today, and they have the per-caller usage counter running. They pull the last 30 days of `shopfast_deprecated_field_reads_total{field="total"}` and break it down:

- **Day 1–30 ago:** the field saw ~120,000 reads/day from three callers: `mobile-bff` (~80k/day), `shipping-svc` (~38k/day), and `unknown` (~2k/day). That `unknown` bucket — requests with no `X-Caller-Service` header — is the forgotten consumer, and it is exactly why you measure.
- **By day 10 ago:** `mobile-bff` and `shipping-svc` both dropped to **0** (they migrated). But `unknown` was still doing ~2k/day. Removing the field now would have broken a consumer the team had never identified — the analytics job, as it turned out, which had hardcoded `total` and no caller header.
- **The team traced the source IP** of the `unknown` reads to the data-platform team's nightly Spark job, got them to migrate, and watched `unknown` fall to **0** three days ago.
- **Today:** all three callers have been flat at **0 reads** for 72 hours, across a weekend peak and a Monday batch run. Quantitatively: 0 of the ~120,000 prior daily reads remain. Now — and only now — contracting is safe.

The number that mattered was not "is the sunset date here?" (it was) but "what percentage of the prior traffic still reads the field?" The answer went from 100% to 1.7% (`unknown` only) to 0%, and the team waited for the *zero*, then verified it persisted across a full cycle. Had they removed on the calendar date alone, at the moment `unknown` was still 1.7%, they would have broken the analytics pipeline silently — the exact missing-contract outage from the stress test, just delayed by 60 days. Telemetry turned "we think everyone migrated" into "we can prove no one reads it," and that proof is the entire license to contract.

## GraphQL evolution: a different shape, the same rules

A quick but important detour, because [GraphQL](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis) handles versioning in a way that looks different but obeys the same additive/breaking law. GraphQL APIs famously do *not* version (`/v2` is essentially unheard of); instead they evolve the single schema additively and deprecate fields in place with the built-in `@deprecated` directive. The reason it works so cleanly is the same reason the tolerant-reader pattern works: a GraphQL client asks for *exactly the fields it wants*, so adding a field to a type is invisible to every existing query, and you can mark an old field deprecated while it still resolves.

```graphql
type Order {
  orderId: ID!
  subtotal: Int!
  grandTotal: Int!                    # added — invisible to existing queries
  total: Int! @deprecated(reason: "Renamed to grandTotal; removed after 2026-08-30")
  currency: String!
}
```

The deprecation is machine-readable — it shows up in introspection and in tools like GraphiQL — so client developers see the warning, and you can run a query against your gateway's *field-usage analytics* (most production GraphQL servers, including Apollo's, track which fields each client operation selects) to get the exact same telemetry the REST counter gave you: which clients still select `total`, and when it is safe to remove. So GraphQL's "no versioning" is not magic; it is compatibility-only plus field-level deprecation plus per-field usage telemetry — the three ideas from this post, packaged into the type system. The lesson generalizes: whether your API is REST, gRPC, or GraphQL, the same three primitives — additive evolution, in-place deprecation with a deadline, and per-field usage telemetry — are what let you evolve safely, and the surface syntax is just where each protocol puts them.

## The trade-off matrix: contract tests versus full e2e

Now place the two strategies side by side. Contract tests are not a strictly superior replacement for end-to-end tests in every dimension — they trade away one real thing — and a senior is honest about what.

![A decision matrix comparing consumer-driven contract tests against full end-to-end tests across run time, flakiness, whether all services must be up, what kind of bug each catches, and who owns the test](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-7.webp)

The thing contract tests trade away is **whole-flow integration coverage.** A contract test verifies the *shape* of the interaction between one consumer and one provider. It does not verify that the end-to-end checkout *behaves correctly* — that the order, payment, inventory, and shipping services compose into a working purchase with the right business outcome. A contract can be perfectly satisfied while the overall flow is wrong (every service returns the contracted shape but the saga rolls back for a subtle reason). So contract tests do not eliminate the need for *some* higher-level testing; they shrink it. The right shape is a **testing pyramid for distributed systems**: a wide base of fast unit tests, a solid layer of contract tests covering every consumer-provider edge, and a *thin* layer of end-to-end tests covering only the two or three business-critical happy paths (checkout completes, refund issues) — not the full integration matrix. The thin e2e layer catches composition bugs; the contract layer catches the API-shape breaks that used to be e2e's main job, but does it in seconds with a clear owner. This is the bridge to [testing microservices from unit to chaos](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos), which goes deep on the rest of the pyramid.

| Property | Consumer-driven contract test | Full end-to-end test |
| --- | --- | --- |
| Run time | Seconds per service | 20–40 minutes per run |
| Needs all services running | No — each side tested in isolation | Yes — the whole fleet plus datastores |
| Flakiness | Low (deterministic, no shared env) | High (shared env, timing, data leakage) |
| Catches API-shape breaks | Yes, with the consumer named | Yes, but slowly and flakily |
| Catches whole-flow business bugs | No — shape only | Yes — real composed behavior |
| Who owns the test | Each team owns its side | Orphaned — tragedy of the commons |
| Scales with fleet size | Linearly (one verification per provider) | Quadratically (the pair matrix) |
| Gates deploy safely | Yes — `can-i-deploy` | Only by spinning up everything |

The decisive read: contract tests are the *primary* gate for API compatibility — fast, owned, deterministic, deploy-gating — and a thin e2e layer is the *secondary* gate for whole-flow behavior. Replacing your fat, flaky, orphaned e2e suite with this pairing is one of the highest-leverage CI changes a microservices org can make.

## Optimization: replacing the slow e2e gate with contract tests

Let me put numbers on that change, because "faster" is not an argument until it is measured.

![A before and after comparison showing a shared end-to-end gate that spins up twelve services in thirty-five minutes with eight percent flakiness and a serialized merge queue, replaced by a contract gate that verifies pacts in forty seconds with zero flakiness and per-service can-i-deploy](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-9.webp)

#### Worked example: the CI-time win from killing the shared e2e gate

ShopFast's platform team measures the old gate. The shared end-to-end suite spins up 12 services plus 3 datastores and a Kafka cluster, takes **35 minutes** wall-clock per run, and is **8% flaky** (one rerun in roughly every 12 runs). Because the environment is shared and stateful, only one merge can be tested at a time, so the suite gates a **serialized merge queue** across all 12 teams. The fleet does about **240 merges per week** (20 per team). The cost:

- CI time on the gate: 240 merges × 35 min = **8,400 minutes/week** of wall-clock on the critical path, serialized.
- Flaky reruns: 8% of 240 ≈ 19 reruns/week × 35 min = **665 wasted minutes/week**, plus the engineer's context-switch each time.
- The serialized queue means the *effective* throughput cap is one merge per ~35 minutes during contention — at peak, teams wait in line, and a single team's flaky run blocks everyone.

Now the contract gate. Each service runs provider verification against its consumers' pacts in about **40 seconds**, in its *own* pipeline, in parallel with every other service — no shared environment, no queue. `can-i-deploy` adds a few seconds. The suite is deterministic, so flakiness drops to effectively **0%**. The new cost:

- CI time on the compatibility gate: 240 merges × ~45 sec ≈ **180 minutes/week**, and it is *parallel*, not serialized — wall-clock on any single merge is under a minute.
- Flaky reruns: ~**0**.
- No merge queue: every team merges independently, which is the entire point of microservices.

The headline: the compatibility gate's CI time drops from **8,400 to ~180 minutes/week** — a **~47× reduction** — and per-merge latency on that gate falls from 35 minutes to under one. The flaky-rerun tax of ~665 minutes/week plus the human cost of "ignore the red build" goes to zero. ShopFast still keeps a *thin* nightly e2e run of the three critical happy paths — maybe 8 minutes, off the merge critical path — so it has not lost composition coverage; it has just stopped paying e2e prices for API-shape coverage that contracts do better. The measurable wins to report up: median merge-to-deploy time, CI minutes billed, and merge-queue wait time, all of which collapse.

That is the optimization story with the kind of numbers a staff engineer can take to a planning meeting: not "contracts feel better," but "we cut the compatibility gate from 8,400 to 180 CI-minutes a week and deleted the merge queue."

## Schema registries: contract testing for events

Everything above assumed synchronous request/response APIs. But a large fraction of inter-service communication in a real fleet is asynchronous — [event-driven, choreography or orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — and events have the *same* evolution problem with an extra twist: there is no synchronous response to verify against, the producer and consumers are decoupled in time, and a message written today might be read by a consumer six months from now or replayed from the start of a [retained log](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers). The tool here is the **schema registry** (Confluent Schema Registry for Kafka being the canonical one), and its compatibility modes are exactly the additive/breaking rules, enforced at publish time.

A schema registry stores the schema for each event topic — typically in **Avro**, **Protobuf**, or **JSON Schema** — and assigns it a version. When a producer tries to register a *new* version of a schema, the registry checks it against the configured **compatibility mode** and *rejects* the registration if the change would break someone. This moves the safety check left, all the way to schema-evolution time, before a single bad message is produced.

![A decision tree for choosing a schema registry compatibility mode based on whether consumers or producers upgrade first, branching to BACKWARD for consumers-first, FORWARD for producers-first, FULL for either order, and flagging NONE as risky](/imgs/blogs/api-versioning-and-consumer-driven-contract-testing-8.webp)

The modes encode *who can upgrade first*, which is the only question that matters during a rolling deploy:

- **BACKWARD** (the Kafka default): a new schema can read data written with the *old* schema. This means you must **upgrade consumers first**, then producers. Safe changes: delete a field, add an optional field with a default. This is the right default because in most event systems you control consumers and want them ready before new-shaped data arrives.
- **FORWARD**: old schema can read data written with the *new* schema, so you **upgrade producers first**. Safe changes: add a field, delete an optional field. Use when producers must move ahead of consumers.
- **FULL**: both directions hold — new reads old *and* old reads new — so you can upgrade in **any order**. Only fully-compatible changes (add/remove optional fields with defaults) pass. The safest and most restrictive.
- **NONE**: no checks. Reserved for the rare case where you genuinely manage compatibility by hand, and a footgun otherwise.

There are also *transitive* variants (`BACKWARD_TRANSITIVE`, etc.) that check the new schema against *all* prior versions, not just the immediately previous one — essential if old consumers might still be reading data from several schema versions ago, or if you replay a topic from the beginning. Setting the mode is one config call:

```bash
# Set FULL_TRANSITIVE compatibility on the shopfast-orders-value subject so any
# upgrade order is safe AND new schemas stay compatible with every past version.
curl -X PUT http://schema-registry:8081/config/shopfast-orders-value \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d '{"compatibility": "FULL_TRANSITIVE"}'
```

And here is the order-event Avro schema evolving safely under BACKWARD compatibility — exactly the same rename, handled the schema-registry way:

```json
{
  "type": "record",
  "name": "OrderPlaced",
  "namespace": "com.shopfast.events",
  "fields": [
    { "name": "order_id", "type": "string" },
    { "name": "subtotal_cents", "type": "long" },
    { "name": "currency", "type": "string", "default": "USD" },
    {
      "name": "grand_total_cents",
      "type": ["null", "long"],
      "default": null,
      "doc": "Added v2. Optional with default so old readers and writers stay compatible."
    }
  ]
}
```

The `"default": null` is doing the load-bearing work: under BACKWARD compatibility a new field *must* have a default, so that a consumer on the new schema reading an old message (which lacks the field) gets the default instead of a parse error. That single rule is the additive-change principle, mechanically enforced by the registry at the moment a producer tries to register an incompatible schema — the producer's CI fails to register and the bad event never ships. This is contract testing for events: the registry is the broker, the compatibility mode is the gate, and "schema registration failed: incompatible change" is the can-i-deploy of the streaming world.

## Stress test: a provider ships a breaking change with no contract

Pose the failure the way it actually unfolds, so you can recognize it in the wild and feel the cost of skipping the safety net. The order team has *no* contract testing. They delete the `total` field — a clean PR, green local tests, a confident deploy. Trace the blast radius and, crucially, *when you find out*.

**T+0, deploy.** The new order-service binary rolls out. Its own unit and integration tests passed; they never referenced the BFF. The PR merged. There is no can-i-deploy gate to refuse it, because there are no contracts. The break is now live and *silent* — no error, no alert, because removing a field is not an error on the producer side. The producer is doing exactly what it was told.

**T+2 min, the first silent corruption.** The BFF starts receiving order responses without `total`. Its tolerant-enough deserializer fills `total` with the zero value. Mobile users see `\$0.00` order totals. No exception is thrown, so no error metric moves. The BFF's dashboards are green. The only signal is in the *data* — order totals that are wrong — and nobody is alerting on "totals look suspiciously round."

**T+8 min, the second-order failure.** The shipping service, reading the same now-missing field, computes free shipping for everyone because `0` is below the free-shipping threshold. Still no exception. The break has now spread to a service the order team has *never heard of*, doing financial damage (shipping cost ShopFast was supposed to charge) that will not appear on any error dashboard at all.

**T+25 min, a human notices.** A customer-support agent gets a ticket: "my app says my order is \$0." Then another. Support escalates. Note the detection mechanism: a *human customer*, not a test, not a metric, not an alert. The mean-time-to-*detect* is bounded below by how fast customers complain, which is the worst possible detector — slow, noisy, and reputation-damaging.

**T+40 min, the wrong root cause.** On-call for the BFF gets paged. The BFF looks healthy — no errors, normal latency — so they spend twenty minutes ruling out the BFF before someone thinks to diff the order service's response payload and spots the missing field. The *cause* (order-service deploy at T+0) and the *symptom* (BFF showing zeros) are in different services owned by different teams, so the investigation crosses an org boundary, which is where [debugging distributed systems](/blog/software-development/microservices/debugging-distributed-systems-in-production) gets genuinely hard.

**T+55 min, rollback.** The order team rolls back. Totals recover. The free-shipping orders already placed are a manual finance cleanup. Total time from break to recovery: nearly an hour, most of it spent *discovering that there was a break and where it lived.*

Now run the same change *with* contract testing. The order team's PR triggers provider verification. It pulls the BFF's pact (which expects `grand_total`/`total`) and the shipping service's pact, replays them, and the BFF and shipping contracts both fail in **CI, in seconds**, with the message "consumer mobile-bff expects field total." The PR gets a red X. The break never merges, never deploys, never reaches a customer. Mean-time-to-detect drops from ~25 minutes (a customer complaint) to ~40 seconds (a CI job), and mean-time-to-*identify-the-consumer* drops from ~35 minutes of cross-team investigation to *zero* — the failure message names the consumer. That gap — customer-detected hour-long outage versus CI-detected red build — is the entire value proposition of the discipline, made vivid.

## Case studies: contract testing and schema registries in the wild

A few real-world data points, kept to what is publicly documented and accurate.

**Pact's origin and the broker pattern (DiUS / realestate.com.au).** Pact began at the Australian consultancy DiUS and saw heavy early use at realestate.com.au, which had many teams shipping services independently. The motivating pain was precisely the one in this post: end-to-end integration environments that were slow, flaky, and a coordination bottleneck across teams. The consumer-driven model and the broker (with `can-i-deploy`) were designed to let each team verify compatibility *without* a shared environment, so deploys could stay independent. The durable lesson: the broker turns the unknown-consumer problem into a queryable registry — you can finally ask "who depends on this provider, and is my change safe for all of them?" and get a mechanical answer.

**Confluent Schema Registry and Kafka's compatibility modes.** Confluent built the Schema Registry to solve event-schema evolution at scale for Kafka users, and BACKWARD compatibility as the default is a deliberate, opinionated choice: in most streaming architectures you control consumers and want them upgraded before new-shaped data flows, and you must be able to read historical data on replay. The registry enforcing compatibility *at schema-registration time* — failing the producer's deploy if a change is breaking — is the same shift-left idea as `can-i-deploy`, applied to events. Teams that adopt it report the failure mode flipping from "a bad event poisons a downstream consumer in production, days later" to "a CI job rejects the incompatible schema, immediately." The schema-registry-saved-us story is mundane precisely because the registry turns a future production incident into a present, local build failure.

**The classic missing-contract outage shape.** The most common public post-mortem pattern in this space is not one famous incident but a recurring genre: a producer team ships what they believe is an internal-only change to a field or event, unaware that a downstream consumer — often an analytics pipeline, a data-warehouse loader, or a partner integration nobody on the producer team knew about — depended on the old shape. The change passes the producer's own tests (which do not know the consumer), deploys, and silently corrupts the consumer's data for hours until someone notices wrong numbers in a dashboard. The lesson every such write-up converges on is the lesson of this post: the producer's test suite cannot protect consumers it does not know about, and the only durable fix is to make consumers *declare* their expectations (via contracts or a schema registry) so the producer is mechanically prevented from violating them.

**Stripe and additive-only evolution.** Stripe is widely cited for a strong commitment to not breaking API clients, using dated version pins so existing integrations keep their old behavior while new integrations get the new one. The practitioner takeaway is not "build Stripe's versioning machinery" — most internal services should not — but the underlying discipline: prefer additive evolution so aggressively that genuine breaks become rare events handled by an explicit, dated opt-in, rather than a constant tax. For an internal fleet, the equivalent is compatibility-only plus expand-and-contract, with contract tests as the gate that keeps the discipline honest.

## When to reach for this (and when not to)

Be decisive, because not every team needs the full apparatus on day one.

**Always internalize the compatibility rules and the tolerant-reader convention.** These are free — they are habits, not tooling — and they prevent the majority of API-evolution incidents. Even a two-service system benefits. There is no scenario where "additive changes are safe, removing/renaming breaks, read tolerantly" is the wrong default.

**Use expand-and-contract for every breaking change, at any scale.** It is a recipe, not a tool, and the discipline of "never have a moment where old and new don't coexist, and verify with telemetry before contracting" pays for itself the first time it prevents an incident.

**Reach for consumer-driven contract testing when you have multiple teams deploying multiple services independently and you feel the pain of e2e tests** — slow CI, flaky shared environments, an orphaned integration suite, or fear every time a producer changes a field. That is the sweet spot. Below roughly three or four services owned by one team, the overhead of running a broker and writing pacts may exceed the value; the team can hold the contracts in their head and a small e2e suite suffices. The crossover is *organizational* (independent teams) more than it is about service count.

**Use a schema registry the moment you have more than a couple of consumers on an event stream**, or any stream you replay or retain long-term. Event schemas drift silently and dangerously because there is no synchronous response to catch the break; the registry is cheap insurance, and BACKWARD or FULL compatibility on important topics is a near-automatic yes.

**Do not** try to replace all integration testing with contracts — keep a thin e2e layer for the two or three business-critical flows, because contracts verify shape, not composed behavior. **Do not** add versioning (`/v2`) reflexively; default to compatibility-only and reserve explicit versions for irreducible breaks on APIs whose clients you do not control. **Do not** skip the Day-14 telemetry step in expand-and-contract; contracting without proving zero readers is just a delayed breaking change.

The honest summary: the rules and expand-and-contract are for everyone, contract testing earns its keep when independent teams create the unknown-consumer problem, and schema registries are the events-shaped version of the same idea. This whole discipline is what makes the [independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) that justifies microservices actually *safe* rather than merely *possible*. Without it, you have the cost of microservices and the coordination of a monolith.

## Key takeaways

- **You cannot deploy a fleet atomically, and a producer rarely knows all its consumers** — so safe API evolution is the precondition for independent deployability, not an optional polish.
- **Additive changes are safe; removing, renaming, or retyping a field breaks consumers.** Apply that one lens to every diff in review. The silent killer is changing a field's *meaning* without changing its name.
- **Be a tolerant reader (Postel's law): bind only the fields you use, ignore unknowns, default missing fields.** This is what makes "additive is safe" true across your fleet rather than just in theory.
- **In Protobuf, the field number is the contract — never reuse or change it, and `reserved` a deleted field forever.** `proto3` dropped `required` on purpose; treat everything as optional.
- **Default to compatibility-only ("no versioning"); use expand-and-contract for breaking changes; reserve `/v2` for irreducible breaks on clients you don't control.** A `/v2` you never retire is a `/v1` you maintain forever.
- **Expand-and-contract makes a breaking change safe: emit both shapes, migrate consumers on their own schedule, and contract only after telemetry proves zero readers of the old shape.** Get the deploy ordering wrong and you've just shipped the break with extra steps.
- **End-to-end tests are the wrong primary safety net** — slow, flaky, orphaned, and quadratic in fleet size. They couple your deploy to everyone else's environment.
- **Consumer-driven contract testing catches API-shape breaks in CI, in seconds, with the broken consumer named, and `can-i-deploy` mechanically blocks an incompatible deploy.** It replaces "we have a rule" with "the system enforces the rule."
- **Schema registries are contract testing for events:** the compatibility mode (BACKWARD/FORWARD/FULL) encodes who upgrades first, and the registry rejects a breaking schema at registration time.
- **The senior move is to measure the win:** the compatibility gate dropping from thousands of serialized CI-minutes a week to a parallel sub-minute check, with flakiness near zero and no merge queue, is the argument that gets contract testing funded.

## Further reading

- Sam Newman, *Building Microservices* (2nd ed., O'Reilly) — the chapters on breaking changes, expand-and-contract, and why end-to-end tests over the whole fleet are a trap.
- Chris Richardson, *Microservices Patterns* (Manning) — API patterns, contract testing, and the testing pyramid for services.
- The Pact documentation and the Pact Broker / `can-i-deploy` guides — the canonical reference for consumer-driven contract testing in practice.
- Confluent Schema Registry documentation — compatibility modes (BACKWARD/FORWARD/FULL and the transitive variants) and Avro/Protobuf schema evolution rules.
- Martin Fowler, "ConsumerDrivenContracts" and "TolerantReader" (martinfowler.com) — the foundational essays behind both ideas.
- RFC 8594 (the `Sunset` HTTP header) and the `Deprecation` header draft — the standardized, machine-readable way to signal a deprecation timeline.
- This series: [REST vs gRPC vs GraphQL for service APIs](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis), [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration), [the anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice), [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), and the upcoming [testing microservices from unit to chaos](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos) and [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability).

---
title: "Contract Testing: Consumer-Driven Contracts and Schema Diffs"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to catch a breaking API change before it ships — schema-diff linters that fail CI on a removed field, consumer-driven Pact tests the provider must verify, a broker's can-i-deploy gate, and where each one actually fits."
tags:
  [
    "api-design",
    "api",
    "contract-testing",
    "pact",
    "consumer-driven-contracts",
    "schema-diff",
    "ci-cd",
    "openapi",
    "protobuf",
    "testing",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-1.png"
---

The worst outage I ever helped clean up was caused by a green build. The payments team had a `GET /payments/{id}` endpoint that returned, among other fields, an `amount` field that was a plain integer count of cents. Someone on that team decided the field name was ambiguous — was it dollars? cents? — and renamed it to `amount_cents`. A sensible change. They updated their handlers, updated their own tests, and shipped. Their entire test suite was green. Their integration tests, which stood up the service against a real database and exercised the endpoint, were green. They had done everything a careful team is supposed to do, and they had done it well.

The checkout service — written by a different team, in a different repository, on a different deploy cadence — was reading `body["amount"]` to decide whether a payment had captured the right total before it released the order to fulfillment. The day the rename shipped, `body["amount"]` started coming back `undefined`. The comparison silently evaluated to false, the order was held, and a queue of held orders backed up behind a check that could never pass again. The payments team's dashboards were perfectly green the entire time. The break did not live in either team's code — it lived in the *space between them*, on the wire, in the shape of a JSON body that one team owned and the other team trusted. No test either team had ever written was looking at that space.

That gap is the whole subject of this post. **A contract break is, by definition, the one kind of bug that lives between two services and therefore inside neither one's test suite.** The provider's tests assert that the provider behaves the way the provider's authors expect. The consumer's tests, run against a mock the consumer's authors wrote, assert that the consumer behaves the way *they* expect. Both can be green while the real, deployed wire shape has drifted apart. The figure below is the entire argument in one frame: a build that is green right up until production, versus a contract gate that fails the pull request before the dangerous change can ever reach the other team.

![a side by side comparison showing a green provider build that breaks the consumer in production versus a contract gate that fails the pull request before merge](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-1.png)

By the end of this post you will be able to do something concrete: stand up an automated layer that catches a breaking change *before it ships*, not after. We will cover two complementary techniques and exactly when to reach for each. The first is **schema-diff linting** — point a tool at your new OpenAPI or Protobuf or GraphQL schema, have it compare against the last published one, and fail the build on any breaking difference. The second is **consumer-driven contract testing** with Pact — let each consumer write down precisely what it depends on, generate a machine-readable contract from that, and make the provider prove in its own CI that it can still satisfy every consumer before it deploys. This is the verification layer for everything else in this track of the series. We have spent several posts establishing [the rules of safe change](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) and [how to evolve a schema by adding, removing, and renaming fields safely](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely). Those posts tell you what *is* a breaking change. This post is how you stop a human from shipping one anyway.

This connects directly to the spine of the whole series, the question we return to in [every post on what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems): *what does the caller get to assume, and can I change this later without breaking them?* Contract testing is how you make that question answerable by a machine, on every commit, instead of answerable by a postmortem.

## Why unit tests and integration tests do not catch contract breaks

Let us be precise about what each existing layer of testing actually verifies, because the gap is not obvious until you name it exactly.

A **unit test** verifies a unit of code in isolation — a function, a class, a small cluster of objects — usually with its collaborators replaced by stubs or mocks. Unit tests are the fastest and most numerous tests you own; you run thousands of them in milliseconds. A unit test on the payments service might assert that, given an order row and a successful gateway response, the serializer produces a body with the right total. But here is the thing: that test is written *by the payments team*, and it asserts the shape the payments team intends. When they renamed `amount` to `amount_cents`, they updated the serializer and they updated the unit test in the same commit. The unit test went from asserting `amount` to asserting `amount_cents`, and it stayed green. The test moved *with* the change because the same person wrote both. A unit test can never catch a contract break, because the unit test's expectation is owned by the same team that is making the change.

An **integration test** verifies that several real components work together — typically a service plus its real database, message broker, or downstream dependency, stood up in a test environment. Integration tests are slower (seconds, sometimes tens of seconds) and you own fewer of them. The payments team almost certainly had an integration test that did `POST /payments` and then `GET /payments/{id}` against a real Postgres and asserted on the response. But again: that assertion was written by the payments team, and when they renamed the field they updated the assertion. The integration test is *within the provider's boundary*. It stands up the provider and the provider's dependencies. It does not stand up the *consumer*. It has no knowledge that a checkout service three repositories away reads `body["amount"]`. The provider's integration test verifies the provider against the provider's own idea of correctness.

The defect is structural, not a matter of diligence. Both teams were diligent. The problem is that the contract — the agreed-upon shape of the wire — is **owned by no single test suite**. The provider's tests live inside the provider's repo and encode the provider's intent. The consumer's tests live inside the consumer's repo and, critically, run against a *mock* of the provider that the consumer wrote. The consumer's mock said "`GET /payments/{id}` returns a body with an `amount` field," because that is what the consumer's authors observed when they wrote the integration. After the rename, the consumer's mock was a *lie* — it described a provider that no longer existed. The consumer's tests passed against a fictional provider. The provider's tests passed against a fictional consumer. Reality, which is the actual deployed pair of services exchanging real bytes, was tested by no one.

This is worth stating as a principle, because it tells you exactly where the new layer has to go:

> **A contract is a shared expectation held across a boundary that neither side independently controls. A test can only catch a regression in an expectation it actually encodes. Therefore no test that lives entirely on one side of the boundary — using the other side's behavior as a mock or a stub — can catch a drift in the shared expectation. Catching a contract break requires a test whose expectation is owned by one side and whose satisfaction is verified by the other.**

That last sentence is the entire design of consumer-driven contract testing, stated before we have even named the tool. The consumer *owns the expectation* (it writes down what it depends on). The provider *verifies the satisfaction* (it proves it still produces that shape). Neither can quietly change the expectation without the other noticing, because the expectation now lives in a shared, versioned artifact instead of inside two mocks that drift independently.

There is a sharper way to see why the mock is the villain of the story. Every consumer that talks to a real provider faces a dilemma when it writes its own tests: testing against the *real* provider is slow, requires that provider to be running, and couples the consumer's CI to the provider's availability — so consumers, sensibly, mock the provider. But the moment you mock, you have created a *second, independent copy* of the contract: the real provider's behavior, and the consumer's belief about that behavior baked into the mock. Those two copies have no mechanical link. Nothing forces them to stay equal. They were equal on the day the consumer's author copied a real response into the mock, and they have been drifting apart, silently, on every provider change since. A mock is a snapshot of a contract frozen at the moment it was written; the contract itself keeps moving. The entire value of contract testing is that it *re-establishes the mechanical link* — it makes the consumer's mock and the provider's real behavior provably equal again on every commit, by turning the mock into a *published artifact the provider is forced to verify against.*

Consider what this means for the *robustness principle* — "be conservative in what you send, liberal in what you accept" — that underlies all of API compatibility. The robustness principle is a promise the consumer makes: *I will tolerate fields I do not understand, and I will not require fields you marked optional.* But a promise is not a guarantee, and a mock is where the promise quietly breaks. A consumer that writes a mock returning exactly `{"id", "status", "amount"}` and then asserts on that exact shape has *secretly stopped being a tolerant reader* — its tests will now fail if the provider adds a field, even though adding an optional field is supposed to be safe. So the very act of mocking can convert a tolerant consumer into a brittle one without anyone noticing. A well-written contract test, with type-and-shape matchers rather than exact-value matchers, *encodes the tolerant-reader posture explicitly*: it says "I depend on these fields having these types, and I make no claim about anything else." That is not a side benefit. It is the contract test forcing the consumer to be honest about exactly how tolerant it really is.

### The end-to-end trap is real but it is not the answer

The instinctive fix, when you first feel this gap, is: "fine, let us stand up *both* services and test them against each other for real." That is an **end-to-end test** — deploy the whole stack, every service live, and drive a real request through all of it. End-to-end tests do catch contract breaks. They catch them *late*, *slowly*, and *expensively*, and they catch them entangled with twenty other things.

Consider what a true end-to-end suite costs. You must deploy every service in the request path, with their real (or realistic) databases, brokers, and external dependencies, into a shared environment. That environment is a tragedy of the commons — every team's broken branch breaks everyone's tests. A single end-to-end run takes minutes to tens of minutes, and the failures are flaky: a slow dependency, a stale cache, a test-data collision, a network blip in the test cluster. When the suite goes red, you cannot tell whether you broke a contract, whether someone else's deploy is mid-flight, or whether the test environment simply hiccuped. The signal-to-noise ratio is terrible, and so — predictably — teams start ignoring red end-to-end builds, which means the suite that was supposed to be your safety net becomes decoration.

And critically: an end-to-end test catches the break *after both services are deployed together*. By then the breaking change is already on `main`, already deployed to a shared environment, already blocking every other team that shares it. The whole point of catching a contract break is to catch it *on the pull request*, before merge, while the change is still cheap to fix and isolated to the author. Contract testing's signature move is that it gives you the cross-service guarantee of an end-to-end test, but it runs *inside one service's CI pipeline with the other side replaced by a recorded, verified artifact* — so it is fast, deterministic, and runs pre-merge.

## The testing pyramid for APIs, and where contract tests sit

The classic testing pyramid is a heuristic about *quantity and speed*: many fast unit tests at the base, fewer slower service and integration tests in the middle, very few slow end-to-end tests at the top, and a sliver of manual testing above that. The shape encodes a hard-won lesson: tests that are slow and broad are expensive to own, flaky to run, and painful to diagnose, so you want as few of them as you can get away with — and you get away with fewer of them precisely by pushing coverage *down* into faster, narrower layers.

![a vertical stack of the API testing pyramid placing contract tests at the boundary between fast service tests and slow brittle end to end tests](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-2.png)

Contract tests are the layer that lets you push *cross-service* coverage down out of the expensive end-to-end tier. Without them, the only way to verify "service A and service B still agree on the wire" is to deploy both and test them together — an end-to-end test. With them, you verify the *same agreement* at the speed of a unit test, in each service's own pipeline, with no shared environment. That is why contract tests sit in the middle of the pyramid, not the top: they catch a category of bug that *feels* like it needs the whole stack, using a recorded artifact instead of the whole stack.

Here is the precise division of labor, which is the heart of why you want all of these layers and not just one:

| Test layer | Scope | Owns the expectation? | Catches a cross-service shape break? | Typical speed |
|---|---|---|---|---|
| Unit | One function or class, collaborators mocked | Yes — same author as the code | No | Milliseconds |
| Service / component | One service in process, internals real | Yes — same author as the code | No | Tens of ms |
| Integration | One service + its real DB or broker | Yes — same author as the code | No (no consumer present) | Seconds |
| Contract | The wire boundary between two services | Split — consumer owns it, provider verifies it | **Yes, in CI, pre-merge** | Seconds |
| End to end | Whole request path, every service live | No single owner | Yes — but late, slow, flaky | Minutes |

The figure below renders the same comparison as a grid so you can see the one row that matters — the contract row is the only layer that catches a cross-service break *without* needing both sides live at once.

![a matrix comparing unit integration contract and end to end tests on logic bugs cross service breaks speed and whether both sides must be live](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-3.png)

Read down the "cross-service break" column. Unit, service, and integration tests all say *missed* — not because their authors are careless but because, structurally, those tests live on one side of the boundary and mock the other. The contract layer says *caught in CI* and *no — replay*: it catches the break and it does so without deploying both sides, because it replays a recorded contract instead. The end-to-end layer also catches the break, but it is the slow, brittle, both-sides-live row. The lesson is not "contract tests replace your other tests." It is "contract tests are the cheapest place to catch the *specific* bug that lives between services, and they let you keep your expensive end-to-end suite tiny."

## Schema-diff linting: fail the build on a breaking diff

We have two distinct families of contract testing, and I want to introduce the simpler one first because most teams should adopt it on day one regardless of anything else they do. It is **schema-diff** (also called breaking-change linting): you compare the *new* version of your machine-readable schema against the *last published* version and fail the build if the difference is breaking.

The premise is that you already have a machine-readable description of your API — an OpenAPI document for a REST/HTTP API, a set of `.proto` files for a gRPC/Protobuf API, or an SDL file for a GraphQL API. (If you do not yet treat the spec as the source of truth, the [spec-first OpenAPI workflow post](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate) is the prerequisite.) A schema-diff tool encodes the *compatibility rules* — the same rules we derived in the [backward and forward compatibility post](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) — and applies them mechanically. Removing a response field: breaking. Adding a required request field: breaking. Narrowing an enum, tightening a type, changing a field's wire type, removing an endpoint, making an optional parameter required: all breaking. Adding an *optional* response field, adding a *new* optional query parameter, adding a *new* endpoint: all non-breaking, by the tolerant-reader principle.

The tools that do this for the three major schema languages are:

- **`oasdiff`** for OpenAPI. It diffs two OpenAPI 3.x documents and can classify each change by severity, with a `breaking` subcommand that exits non-zero when it finds a breaking change.
- **`buf breaking`** for Protocol Buffers. Part of the Buf toolchain, it compares your `.proto` files against a baseline (a git ref, an image, or a module in the Buf Schema Registry) and fails on wire-incompatible or source-incompatible changes per a configurable rule set.
- **`graphql-inspector`** for GraphQL. Its `diff` command compares two SDL schemas and flags breaking, dangerous, and non-breaking changes; in CI it fails on breaking ones.

The principle that makes this work is worth pausing on, because it is the same robustness principle that runs through the whole compatibility story. A consumer that follows the tolerant-reader rule **ignores fields it does not recognize** and **does not require fields the contract marks optional**. Given that, the set of changes a provider can make safely is exactly: *add optional things, never remove or tighten existing things.* A schema-diff tool is nothing more than a mechanical checker of that asymmetry. Removing a response field is breaking because a tolerant reader is allowed to ignore *unknown* fields, but it is *not* obligated to tolerate the *disappearance* of a field it was told it could rely on. The asymmetry — additions safe, removals/tightenings unsafe — falls directly out of what the consumer is and is not allowed to assume.

#### Worked example: oasdiff catches a removed field and fails CI

Let us reconstruct the exact outage from the introduction, but this time with a schema-diff gate in the pipeline. The payments team's published OpenAPI document — call it the baseline, the last thing that actually shipped — describes the payment resource like this:

```yaml
# baseline: openapi.yaml as last published
components:
  schemas:
    Payment:
      type: object
      required: [id, status, amount, currency]
      properties:
        id:
          type: string
          example: pay_3Nf8a2
        status:
          type: string
          enum: [pending, captured, failed, refunded]
        amount:
          type: integer
          description: Total in the currency's minor unit
          example: 4999
        currency:
          type: string
          example: USD
        created_at:
          type: string
          format: date-time
```

The engineer making the rename edits the working copy of the spec — they remove `amount` and add `amount_cents`:

```yaml
# proposed: openapi.yaml in the pull request
components:
  schemas:
    Payment:
      type: object
      required: [id, status, amount_cents, currency]
      properties:
        id:
          type: string
          example: pay_3Nf8a2
        status:
          type: string
          enum: [pending, captured, failed, refunded]
        amount_cents:
          type: integer
          description: Total in the currency's minor unit
          example: 4999
        currency:
          type: string
          example: USD
        created_at:
          type: string
          format: date-time
```

In CI, before this can merge, a step fetches the baseline spec (from the `main` branch, or from a published artifact registry) and runs the diff:

```bash
# fetch the last-published spec, then diff the PR's spec against it
git show origin/main:openapi.yaml > /tmp/baseline.yaml

oasdiff breaking /tmp/baseline.yaml ./openapi.yaml \
  --fail-on ERR \
  --format text
```

The tool prints something like this and exits with a non-zero status code:

```bash
2 breaking changes: 2 error, 0 warning

error  [response-property-removed]
       in components/schemas/Payment
       removed the response property 'amount'

error  [request-property-became-required]
       in components/schemas/Payment
       the property 'amount_cents' became required in the response

exit status 1
```

Because `oasdiff breaking` exited `1`, the CI job fails, the pull request gets a red check, and branch protection refuses the merge. The renamed field never reaches `main`, the checkout service never sees `body["amount"]` go `undefined`, and the queue of held orders never forms. The fix the diff *pushes the engineer toward* is the expand-then-contract migration from the [schema-evolution post](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely): add `amount_cents` *alongside* `amount`, keep both populated for a deprecation window, and only remove `amount` after telemetry shows no consumer reads it. The diff turns an invisible decision ("is this safe?") into a visible, enforced one.

The timeline below traces those steps in order — the moment a diff turns a quiet, dangerous edit into a loud, blocked one.

![a timeline showing a pull request that removes a field flowing through a schema diff step that detects the breaking change and fails the build before merge](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-4.png)

#### Worked example: buf breaking on a Protobuf field renumber

The Protobuf story has an extra wrinkle that makes schema-diff even more important: in Protobuf, the *field number* is the wire identity, not the field name. A `.proto` like this:

```protobuf
syntax = "proto3";
package payments.v1;

message Payment {
  string id = 1;
  PaymentStatus status = 2;
  int64 amount = 3;        // minor units
  string currency = 4;
  google.protobuf.Timestamp created_at = 5;
}
```

Renaming `amount` to `amount_cents` while *keeping* field number `3` is, surprisingly, **wire-compatible** — the bytes on the wire carry tag `3`, not the name `amount`, so a deployed binary keeps decoding fine. But it is **source-incompatible**: any consumer that regenerates code and references the old generated `.getAmount()` accessor will fail to compile. Worse is the genuinely dangerous mistake — *reusing* a field number for a different type:

```protobuf
message Payment {
  string id = 1;
  PaymentStatus status = 2;
  // amount removed
  string currency = 3;     // DANGER: number 3 reused, was int64 amount
  string amount_cents = 4;
}
```

Now field number `3`, which used to be an `int64` amount, is a `string` currency. Old clients that still send or read tag `3` as an `int64` will silently misinterpret a string's bytes as an integer — a corruption that no exception ever surfaces. `buf breaking` exists precisely to make this impossible to merge:

```bash
buf breaking --against '.git#branch=main'
```

```bash
payments/v1/payment.proto:9:3:Field "3" with name "currency" on message
"Payment" changed type from "int64" to "string".
payments/v1/payment.proto:9:3:Previously present field "3" with name
"amount" on message "Payment" was deleted without reserving the number.

exit status 100
```

The fix the tool teaches is the Protobuf discipline: **never reuse a field number; mark removed numbers `reserved`.** The corrected message reserves `3` so it can never be silently recycled:

```protobuf
message Payment {
  string id = 1;
  PaymentStatus status = 2;
  reserved 3;                  // amount used to live here; never reuse
  reserved "amount";
  string currency = 4;
  string amount_cents = 5;     // new number, additive, safe
}
```

That `reserved 3;` is a contract written *in the schema itself*: it tells every future editor, and the diff tool, that number `3` is salted earth. This is the same expand-and-never-recycle idea as the OpenAPI case, expressed in Protobuf's wire-number vocabulary. The [system-design post on schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) goes deeper on running this discipline across hundreds of services and a shared schema registry; here we are focused on the per-pull-request gate.

#### Worked example: graphql-inspector blocking a removed field

GraphQL deserves its own walk-through because its compatibility model is subtly different and the tooling reflects it. A GraphQL client asks for *exactly the fields it wants*, so the schema is the menu and the operation is the order. Removing a field from the schema is breaking only for clients that ordered it — but the diff tool, working from the schema alone, must assume the worst. Suppose the payments schema is:

```graphql
type Payment {
  id: ID!
  status: PaymentStatus!
  amount: Int!
  currency: String!
  createdAt: String!
}

enum PaymentStatus {
  PENDING
  CAPTURED
  FAILED
  REFUNDED
}
```

and a pull request renames `amount` to `amountCents` and also drops the `REFUNDED` enum value. Running the inspector against the published baseline:

```bash
graphql-inspector diff origin/main:schema.graphql ./schema.graphql
```

prints a classified list and exits non-zero on the breaking ones:

```bash
Detected the following changes (3) between schemas:

✖  Field 'amount' was removed from object type 'Payment'
✖  Enum value 'REFUNDED' was removed from enum 'PaymentStatus'
✔  Field 'amountCents' was added to object type 'Payment'

Detected 2 breaking changes
Error: Breaking changes detected
exit status 1
```

The enum removal is the GraphQL-specific trap worth noticing: a client with a non-exhaustive handling of `PaymentStatus` will not crash on a *new* enum value the way a strict consumer might, but *removing* `REFUNDED` breaks any client that still sends it as an argument or pattern-matches on it. The inspector classifies the removal as breaking precisely because it cannot prove no client depends on it. The reconciliation with reality — *which* clients still order `amount` — is exactly what operation safelisting adds on top, and we return to that in the case studies.

#### Worked example: wiring the diff into a CI gate

A schema-diff tool only protects you if it actually runs and actually blocks the merge. Here is the whole gate as a GitHub Actions job, so you can see that the "fail the build" claim is concrete and not hand-waved:

```yaml
name: api-contract
on: [pull_request]

jobs:
  schema-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # need history to read the baseline

      - name: Fetch the last-published spec
        run: git show origin/main:openapi.yaml > /tmp/baseline.yaml

      - name: Install oasdiff
        run: |
          curl -sSfL https://raw.githubusercontent.com/oasdiff/oasdiff/main/install.sh \
            | sh -s -- -b /usr/local/bin

      - name: Fail on a breaking diff
        run: |
          oasdiff breaking /tmp/baseline.yaml ./openapi.yaml \
            --fail-on ERR \
            --format githubactions
```

The whole gate is the last step: `oasdiff breaking ... --fail-on ERR` exits non-zero on a breaking change, the step fails, the job fails, and a *required* status check (configured in branch protection) refuses the merge. The `--format githubactions` flag makes the tool emit annotations that surface the diff inline on the changed lines in the pull request, so the reviewer sees `removed the response property 'amount'` pinned to the exact line that removed it. Nothing about this is exotic — the entire breaking-change gate is one tool, one baseline, and one required check. The discipline is in *deciding to require it*, which is an organizational choice as much as a technical one.

One subtlety that bites teams: *what is the baseline?* Using `origin/main:openapi.yaml` compares against the spec on the main branch, which is correct if main always reflects what is deployed. But if you deploy from tags, or run multiple versions in production, the honest baseline is *the spec that is actually live*, which is why mature setups publish the spec to an artifact registry on every deploy and diff against the *published* artifact rather than against a branch. The compatibility question is never "is this different from main" — it is "is this different from what a real caller is talking to right now." Get the baseline wrong and the gate either misses real breaks (baseline too new) or screams about changes already safely shipped (baseline too old).

### Where schema-diff shines, and where it is blind

Schema-diff has one enormous virtue: it requires **nothing from your consumers**. You do not need to know who they are, you do not need their code, you do not need them to participate. You compare your spec against your own last-published spec, and the compatibility rules are universal. That makes schema-diff the *only* practical contract gate for a **public API with unknown consumers** — you cannot run the tests of consumers you have never met, but you can promise never to break the spec you published to them.

But schema-diff has a corresponding blindness, and it is important to name it honestly. **Schema-diff verifies compatibility with the *spec*, not with what any *actual consumer depends on*.** It will faithfully refuse to let you remove a field. But it has no idea whether removing that field would break *anyone*. It treats every documented field as equally load-bearing, even one that no living consumer has ever read. Conversely — and this is subtler — it cannot catch a break that is technically schema-compatible but semantically catastrophic. If you keep the `status` field but start returning a new enum value `disputed` that you forgot to document, a permissive schema-diff might wave it through, while a consumer with a strict `switch` over the known statuses falls into its `default` branch and mishandles the payment. Schema-diff knows the *declared* shape; it does not know the *real* expectations of *real* code. For that, you need the second family.

## Consumer-driven contracts with Pact, in depth

**Consumer-driven contract testing** flips the direction of authority. Instead of the provider publishing a spec and hoping consumers conform, *each consumer declares exactly what it depends on*, and the provider must prove it satisfies *the union of all consumer expectations.* The contract is "driven" by the consumer because the consumer is the one who writes down the expectation. The reference implementation of this idea is **Pact** — both a specification (the JSON format of a "pact" file) and a family of libraries across languages.

Here is the mechanism, end to end. The consumer writes a test that runs against a **Pact mock provider** — a tiny local HTTP server that Pact spins up. In that test the consumer says, in code: "*when I send this request, I expect this response.*" Pact records every such interaction. If the consumer's *own* code, exercised in that test, actually sends those requests and correctly handles those responses, Pact writes out a **pact file**: a JSON document listing every (request, expected-response) pair the consumer relies on. That pact file is the contract — and crucially, it contains *only the parts of the provider's API the consumer actually touches.* If the consumer reads only `id`, `status`, and `amount`, the pact says nothing about `currency` or `created_at`, because the consumer does not depend on them.

The pact file is then published to a **Pact Broker** — a server that stores pacts, versions them, and tracks which consumer version expects what. The provider's CI pipeline pulls the relevant pacts from the broker and runs **provider verification**: for each interaction in each pact, Pact replays the recorded request against the *real* provider (started up in the provider's test harness) and checks that the *real* response matches what the consumer expected. If a consumer expected `amount` and the provider now returns `amount_cents`, verification *fails in the provider's pipeline* — the provider learns, before merging, that this change would break a real consumer.

The figure below shows the full flow, including the branch that makes the whole thing safe to deploy: the broker not only stores pacts and feeds them to provider verification, it also answers the **can-i-deploy** question.

![a graph showing the consumer test generating a pact file that flows to a broker which feeds provider verification and a can i deploy gate before deployment](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-5.png)

#### Worked example: a Pact consumer test for the checkout service

Let us write the checkout service's expectation of the payments API. The checkout service depends on exactly three things from `GET /payments/{id}`: the `status`, the `amount` (in minor units), and that the response is `200` with `Content-Type: application/json`. Here is the consumer test, using the Pact JavaScript library:

```javascript
const { PactV3, MatchersV3 } = require("@pact-foundation/pact");
const { like, integer, regex } = MatchersV3;
const { getPayment } = require("../src/payments-client");

const provider = new PactV3({
  consumer: "checkout-service",
  provider: "payments-api",
  dir: "./pacts",
});

describe("checkout reading a payment", () => {
  it("reads status and amount from a captured payment", () => {
    provider
      .given("a captured payment pay_3Nf8a2 exists")
      .uponReceiving("a request for a captured payment")
      .withRequest({
        method: "GET",
        path: "/payments/pay_3Nf8a2",
        headers: { Accept: "application/json" },
      })
      .willRespondWith({
        status: 200,
        headers: { "Content-Type": "application/json" },
        body: {
          id: like("pay_3Nf8a2"),
          status: regex("pending|captured|failed|refunded", "captured"),
          amount: integer(4999),
        },
      });

    return provider.executeTest(async (mockServer) => {
      const payment = await getPayment(mockServer.url, "pay_3Nf8a2");
      expect(payment.status).toBe("captured");
      expect(payment.amount).toBe(4999);
    });
  });
});
```

Two things deserve attention. First, the **matchers** — `like`, `integer`, `regex` — are doing something important. The consumer does not assert that the amount is *exactly* `4999`; it asserts that the amount is *an integer* (and uses `4999` only as the example the mock returns). It does not assert the status is *exactly* `captured`; it asserts the status *matches the enum pattern*. This is the difference between a contract test and a snapshot test. A contract test pins the *shape and type* of what the consumer depends on, not the exact values, because the consumer's code depends on `amount` being an integer it can compare, not on it being literally `4999`. Over-tightening matchers to exact values is the single most common Pact mistake; it makes the contract brittle and floods the provider with verification failures over irrelevant value differences.

Second, the `given(...)` line — the **provider state**. The consumer says "I expect this *given that* a captured payment exists." The provider, during verification, will need to set up exactly that state before replaying the request. This decouples the contract from any shared database: the consumer describes the precondition in words, and the provider is responsible for arranging it.

Running this test produces a pact file, `./pacts/checkout-service-payments-api.json`, that looks roughly like this:

```json
{
  "consumer": { "name": "checkout-service" },
  "provider": { "name": "payments-api" },
  "interactions": [
    {
      "description": "a request for a captured payment",
      "providerStates": [
        { "name": "a captured payment pay_3Nf8a2 exists" }
      ],
      "request": {
        "method": "GET",
        "path": "/payments/pay_3Nf8a2",
        "headers": { "Accept": "application/json" }
      },
      "response": {
        "status": 200,
        "headers": { "Content-Type": "application/json" },
        "body": { "id": "pay_3Nf8a2", "status": "captured", "amount": 4999 },
        "matchingRules": {
          "body": {
            "$.id": { "matchers": [{ "match": "type" }] },
            "$.status": {
              "matchers": [{ "match": "regex",
                "regex": "pending|captured|failed|refunded" }]
            },
            "$.amount": { "matchers": [{ "match": "integer" }] }
          }
        }
      }
    }
  ]
}
```

Notice what is *not* in this pact: there is no mention of `currency`, no `created_at`. The checkout service does not read them, so they are not in its contract, so the provider is free to change them without ever failing checkout's verification. **The contract describes the consumer's real dependency surface, which is almost always far smaller than the full API.** This is the core economic argument for consumer-driven contracts over a blanket spec check: it tells the provider precisely how much freedom it has, consumer by consumer.

#### Worked example: provider verification fails on the rename

Now the payments team's CI runs provider verification. Using the Pact JVM verifier as a concrete example, the provider's test harness looks like this:

```javascript
const { Verifier } = require("@pact-foundation/pact");

describe("payments-api honors its consumers", () => {
  it("verifies pacts from the broker", () => {
    return new Verifier({
      provider: "payments-api",
      providerBaseUrl: "http://localhost:8080",
      pactBrokerUrl: process.env.PACT_BROKER_URL,
      pactBrokerToken: "Bearer <token>",
      consumerVersionSelectors: [{ mainBranch: true }, { deployed: true }],
      publishVerificationResult: true,
      providerVersion: process.env.GIT_SHA,
      stateHandlers: {
        "a captured payment pay_3Nf8a2 exists": async () => {
          await seedPayment({ id: "pay_3Nf8a2", status: "captured", amount: 4999 });
        },
      },
    }).verifyProvider();
  });
});
```

The `stateHandlers` block is how the provider satisfies the `given(...)` precondition: before replaying checkout's request, it seeds a captured payment. Then Pact replays `GET /payments/pay_3Nf8a2` against the *real, running* payments service. With the field renamed to `amount_cents`, the real response body is `{"id": "...", "status": "captured", "amount_cents": 4999}`. Verification compares that against checkout's expectation, which demanded an integer at `$.amount`, and fails:

```bash
Verifying a pact between checkout-service and payments-api
  a request for a captured payment
    returns a response which
      has status code 200 (OK)
      has a matching body (FAILED)

Failures:

1) Verifying a pact between checkout-service and payments-api -
   a request for a captured payment has a matching body
   $.amount -> Expected an integer but the field is missing

exit status 1
```

This is the moment the whole system pays off. The *payments team*, in the *payments pipeline*, on *their pull request*, learns that this change breaks the *checkout service*. They did not have to know checkout existed; they did not have to read checkout's code. The broker told them which consumers depend on them, the pact told them exactly what checkout reads, and verification told them precisely what they broke. The break that, in the introduction, surfaced as a queue of held orders and a frantic postmortem, now surfaces as a red check on a pull request with a one-line explanation: `$.amount -> Expected an integer but the field is missing`.

### Provider states: the precondition that keeps pacts decoupled

The `given(...)` / `stateHandlers` pair is the part of Pact that newcomers underestimate, and getting it right is what separates a contract test from a flaky integration test in disguise. The problem it solves: a consumer's expectation almost always depends on *the provider being in some state.* Checkout's expectation "GET this payment returns status `captured`" only makes sense if a captured payment with that id exists. If the consumer and provider shared a database fixture, you would be back to a brittle, coupled setup where a change to seed data breaks tests across repos.

Provider states break that coupling by making the precondition a *named string* in the pact rather than a shared fixture. The consumer writes `given("a captured payment pay_3Nf8a2 exists")` — that is just a label, an opaque token the consumer and provider agree on. The provider, during verification, has a `stateHandler` keyed on that exact string whose job is to *arrange* the state however the provider likes: insert a row, stub an internal gateway client, prime a cache. The consumer does not know or care *how* the provider sets up a captured payment; it only knows *that* one will exist. This is the same separation-of-concerns that makes the whole approach scale — the consumer owns *what it depends on semantically*, the provider owns *how to make that true.*

The failure mode to avoid is letting state handlers reach into shared infrastructure that other tests also touch. A state handler that truncates and re-seeds the whole payments table will fight every other interaction's handler and produce order-dependent flakiness. The discipline is: each handler should set up *only* what its interaction needs, idempotently, and clean up after itself, so that interactions can run in any order. When a state handler is hard to write, it is usually telling you something true — that the consumer is depending on a state that is awkward to construct, which often means the consumer is depending on too much.

### Matchers: why a contract test is not a snapshot test

It is worth dwelling on the matcher discipline because it is where most teams either get the durable value or quietly poison their suite. A *snapshot test* records an exact response and fails on any byte difference. That is the wrong tool for a contract: the provider will legitimately change values — a new payment id, a different timestamp, an additional optional field — and a snapshot would flag every one of those as a failure, training everyone to rubber-stamp the diffs until a real break slips through unnoticed.

A contract test instead pins *the part of the structure the consumer's code actually relies on*, expressed as types and patterns:

```javascript
willRespondWith({
  status: 200,
  headers: { "Content-Type": "application/json" },
  body: {
    id: like("pay_3Nf8a2"),                    // any string
    status: regex("pending|captured|failed|refunded", "captured"),
    amount: integer(4999),                     // any integer; 4999 is the mock value
    line_items: eachLike({                      // an array of at least one item
      sku: like("SKU-1"),
      quantity: integer(2),
    }),
  },
});
```

Read this as a precise statement of dependence: "checkout depends on `id` being *a string*, `status` being *one of these four values*, `amount` being *an integer*, and `line_items` being *an array whose elements have a `sku` string and an integer `quantity`*." The literal `4999` and `"pay_3Nf8a2"` are only the values the mock server returns so the consumer's code has something concrete to run against; they are *not* part of the contract. The `eachLike` matcher is especially important: it says "at least one element of this shape," which lets the provider return one item or a thousand without ever changing the contract, while still pinning the *element shape* the consumer iterates over. Get the matchers right and the pact fails *only* when the provider changes something checkout actually depends on — which is exactly the signal you want and nothing else.

### A problem-solving narrative: a new field, a race, and a removed field

Let us reason through the full lifecycle of one change, the way you would on a real platform, and stress-test the design at each step.

**The change.** Checkout wants to display *when* a payment captured, so it needs a new `captured_at` timestamp from the payments API. This is a coordinated change across two teams and two repos. How do we ship it without a break in either direction?

**Step one — the consumer leads.** Checkout adds an interaction to its pact expecting `captured_at` (a string in date-time format) and ships *the pact* — but not yet the code that reads it, or it ships the reading code behind a guard that tolerates the field's absence. The pact is published to the broker tagged with checkout's branch.

**Step two — stress-test the race.** Here is the danger: if checkout *deploys* its new version (which now expects `captured_at`) *before* the payments provider that produces it, checkout's expectation is unmet in production. Does anything catch this? Yes — `can-i-deploy`. When checkout's deploy pipeline asks "can I deploy version X to production," the broker checks whether the *currently deployed* payments version has been verified against checkout's new pact. It has not (payments has not shipped `captured_at` yet), so `can-i-deploy` exits non-zero and *blocks checkout's deploy.* The race is mechanically impossible to lose, because the gate refuses to let the consumer get ahead of the provider.

**Step three — the provider catches up.** The payments team adds `captured_at` to the response (an additive, non-breaking change — schema-diff is happy, it is a *new optional field*), and provider verification now passes checkout's new pact. The broker records that payments version Y satisfies checkout's new pact. Now checkout's `can-i-deploy` goes green, and checkout can deploy.

**Step four — the removal, much later.** A year on, the payments team wants to remove the long-deprecated `amount` field (everyone migrated to `amount_cents`). Is it safe? They do not have to guess. The broker's matrix shows *every consumer pact and whether it still references `amount`.* If no live consumer's verified pact reads `amount`, removal is safe, and provider verification will confirm it by passing every pact even after the field is gone. If even one consumer still reads it, verification fails on *that consumer's* pact, names the consumer, and the removal is blocked until that team migrates. The decision "is it safe to remove this field" has gone from an act of faith to a query against recorded fact.

This narrative is the whole thesis in motion: the consumer *leads* with a pact, the gate *prevents the race*, the provider *catches up* under an additive change, and the broker *records* enough fact to make removal a decision rather than a gamble. Every dangerous moment — the race, the removal — is caught by a mechanical gate before it reaches production.

### The broker and the can-i-deploy gate

The pact file and verification get you most of the value, but the **broker** adds the piece that makes contract testing safe in a world of independent deploys: it tracks *which consumer versions and which provider versions have been verified against each other*, and it answers the question every deploy pipeline actually needs to ask — **"is it safe to deploy *this* version into *this* environment, given what is already running there?"**

This matters because of a timing problem. Verification proves that provider version *P* satisfies consumer pact *C*. But your environments contain *deployed* versions, and those move independently. Suppose checkout is about to ship a *new* version that starts reading a brand-new field `captured_at`. Checkout's new pact now demands `captured_at`. If checkout deploys before the payments provider that adds `captured_at`, checkout breaks — not because anyone removed anything, but because the consumer raced ahead of the provider. The can-i-deploy gate prevents exactly this. Before deploying, each side asks the broker:

```bash
pact-broker can-i-deploy \
  --pacticipant checkout-service \
  --version "$GIT_SHA" \
  --to-environment production \
  --broker-base-url "$PACT_BROKER_URL" \
  --broker-token "Bearer <token>"
```

The broker checks: for the version of checkout you want to deploy, is there a *verified* result against the version of payments-api currently in production? If yes, it prints a compatibility matrix and exits `0`; if no — if checkout's new pact has not been verified against what is live — it exits non-zero and the deploy is blocked:

```bash
Computer says no ¯\_(ツ)_/¯

The matrix does not contain any verified results between
checkout-service version 4f3a9c1 and the version of payments-api
currently deployed to production (version 9b2e7d4).
```

This is the gate that turns a pile of green checkmarks into an actual deployment guarantee. It is the difference between "all our contract tests passed at some point" and "the exact pair of versions about to run together in production has been proven compatible." It is also why the can-i-deploy step appears as the branch target in the flow figure: the broker feeds both provider verification *and* the deploy gate, and only a green can-i-deploy lets a version through to production.

The reason this matters more than it first appears is that *verification and deployment are decoupled in time.* Provider verification answers "does provider version Y satisfy consumer pact C?" — a fact about a *pair of artifacts.* Deployment is a fact about *what is running in an environment right now.* In an org with dozens of services deploying many times a day, the set of versions live in production is a constantly shifting combination, and no single CI run ever sees the whole combination. The broker is the only component that holds the full picture: it knows, for every environment, which version of each service is deployed, and it knows, for every pair of versions, whether they have been verified compatible. `can-i-deploy` is a query over that picture. Without it, you have a thousand green checks and no statement about whether the *specific combination* you are about to create has ever been tested. With it, you have a single, decisive yes-or-no per deploy, grounded in recorded fact about the exact environment you are deploying into.

There is a deployment-versus-release nuance worth flagging for anyone who runs feature flags or blue-green. `can-i-deploy` reasons about what is *deployed*, but if you can have two versions of a service running side by side during a rollout, the honest question is "is my new version compatible with *every* version that might receive traffic during the transition," which the broker models by letting you record multiple deployed versions per environment. The principle stays the same — prove the combination before you create it — but the set of combinations to prove grows during a rollout, and the broker has to know about each one. This is the contract-testing analogue of the same overlap-window reasoning that governs [safe schema change](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change): during any transition, both the old and the new shape must be simultaneously satisfiable.

## Provider contracts and schema-as-contract: the other direction

Consumer-driven is one direction of authority; there is a respectable opposite school called **provider contracts** or **schema-as-contract**, and it is exactly the schema-diff model elevated to a philosophy. Here the *provider* is the source of truth: it publishes an OpenAPI document (or `.proto`, or SDL), and *consumers validate themselves against it.* The provider promises "this spec is my contract; I will evolve it only compatibly," and consumers can generate clients from it, mock against it, and run their own tests against a mock that is *derived from the published spec* rather than hand-written.

The key advantage over a hand-written consumer mock — the thing that bit us in the introduction — is that the mock is *generated from the provider's real published spec*, so it cannot silently drift from the provider's actual shape. When the provider publishes a new spec, the consumer regenerates its mock and immediately sees, in its own tests, anything that changed. Combined with the schema-diff gate on the provider side (which prevents the provider from publishing a *breaking* spec in the first place), you get a respectable two-sided guarantee without ever exchanging consumer code.

The trade-off, again, is the blindness we named earlier: provider-contract / schema-as-contract verifies against the *declared* spec, not against *what consumers actually depend on.* The provider gets no signal about which fields are load-bearing and which are dead. It must treat the entire published surface as sacred, which over years calcifies into a spec full of fields nobody dares remove because no one can prove they are unused.

### Bi-directional contracts: reconciling both

The most interesting recent development is the **bi-directional contract** approach, popularized by Pactflow. It tries to get the best of both directions while paying neither's full cost. Here is the move: the *provider* publishes its OpenAPI spec (the provider contract), and each *consumer* publishes its pact (the consumer contract) — but **the provider does not run the consumer's pacts against a live provider.** Instead, a tool *statically reconciles the two contracts*: it checks that every interaction in every consumer's pact is *described by, and compatible with,* the provider's published OpenAPI spec.

The figure and the table below place all three approaches side by side. The bi-directional approach answers the question "does the consumer depend on anything the provider's spec does not promise?" — purely by comparing two documents, with no need to stand up the provider against the consumer's pact.

![a matrix comparing consumer driven Pact schema diff linting and bi directional provider spec contracts across best fit consumer code requirement what they catch and deploy gate](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-6.png)

| Approach | Source of truth | Who participates | Strongest guarantee | Main cost |
|---|---|---|---|---|
| Consumer-driven (Pact) | The consumer's real expectations | Consumer writes pact; provider verifies it | The provider proves it satisfies real consumer code | Provider must run verification; needs provider states |
| Schema-diff linter | The provider's published spec | Provider only | The published spec never changes breakingly | Blind to who actually depends on what |
| Provider spec / schema-as-contract | The provider's published spec | Provider publishes; consumer self-validates | Consumer mock cannot drift from spec | Treats the whole surface as load-bearing |
| Bi-directional (Pactflow) | Provider spec reconciled with consumer pacts | Both publish; tool reconciles statically | Consumer depends on nothing the spec omits | Static check is weaker than live verification |

The honest caveat on bi-directional: because the reconciliation is *static* — comparing a pact against a spec, not against running code — it cannot catch a case where the provider's *implementation diverges from its own spec.* If the payments service's OpenAPI says `amount` is an integer but the implementation actually returns a string, classic Pact provider verification (which hits the real service) catches it, while bi-directional reconciliation does not, because it trusts the spec. So bi-directional buys you cheaper participation (the provider only needs to keep its spec honest and run schema-diff) at the price of trusting that the provider's spec matches its implementation — which is precisely why you *also* keep the schema-diff gate and ideally some provider tests that validate responses against the spec.

## Where each approach fits: a decision

The two families are not competitors; they cover different *consumer topologies*. The deciding question is almost embarrassingly simple: **do you know who your consumers are, and can you run their tests?**

![a decision tree branching on whether consumers are known internal services or unknown public callers and routing to Pact bi directional or a published spec with a diff linter](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-8.png)

For an **internal service mesh with known consumers** — a fleet of microservices in your own org, where you can see every consumer's repository and run its CI — **consumer-driven Pact is the strongest tool you have.** You know exactly who depends on you, the broker tells you which versions, and verification proves you satisfy their *real* expectations, not a hypothetical spec. The fact that each pact contains only what that consumer actually touches gives you the maximum freedom to evolve: you can remove `currency` from the payment resource the moment the broker shows no consumer's pact references it. This is the case where consumer-driven contracts earn their keep, and it is the case D20 is most about for the [Payments/Orders running example](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) — checkout (consumer) and payments (provider) are both inside your walls.

For a **public API with unknown consumers** — anyone on the internet with an API key, whose code you will never see — **Pact is the wrong tool**, and it is important to say so plainly because teams over-apply it. You cannot write pacts for consumers you have never met; you cannot run their tests; the broker has no idea they exist. For public APIs the right gate is **a published spec plus a schema-diff linter**: you promise to evolve the spec only compatibly, and you enforce that promise mechanically on every pull request. The contract with the public is the spec, and the diff is how you keep that promise. (This is also where the [deprecation and sunset post](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) and `Deprecation`/`Sunset` headers do the work that can-i-deploy does internally — you cannot block a public caller's deploy, so you give them a migration window instead.)

Most real organizations are *both* at once, and the mature answer is to run *both layers*: schema-diff on every API regardless (cheap, universal, no consumer participation needed), plus Pact on the internal service-to-service edges where you control both sides. Schema-diff is your floor — it catches the universal, structural breaks for everyone. Pact is your ceiling on the internal mesh — it catches the semantic, real-dependency breaks that schema-diff is blind to. The bi-directional approach is the bridge for the awkward middle: a handful of *named* external partners whose pacts you can collect but whose CI you cannot run.

Let me stress-test the public-API case, because the temptation to over-engineer it is strong. Imagine the payments API is now public, with ten thousand integrators you will never meet. Someone proposes Pact "to be safe." Walk through what that buys you. You can write pacts for the three internal consumers you happen to know about, and verification will faithfully protect those three. But the other 9,997 integrators have no pact, so verification says nothing about them — and worse, the green checkmarks create a *false* sense that the public surface is protected when it is not. The integrator most likely to break is precisely the one you have never heard of, doing something with your API you never anticipated. For that population, the only honest contract is *the published spec plus an ironclad promise to evolve it only additively*, enforced by schema-diff on every pull request. You cannot test code you do not have; you can only keep a promise about a document you published. That is why the decision tree's right branch routes public surfaces to a spec-plus-diff gate and not to Pact — not because Pact is bad, but because it is structurally inapplicable when the consumer set is open.

The flip side is equally worth saying: do not reach for schema-diff *alone* on a busy internal mesh and call it done. Schema-diff will dutifully refuse to remove `currency`, forever, even after the last consumer stopped reading it years ago — because it cannot see consumers, only the declared surface. Over time that calcifies your internal API into a museum of fields nobody dares touch, which is its own kind of debt: you lose the *evolvability* half of the series' spine. On the internal mesh, where you *can* see consumers, Pact's broker tells you when a field is genuinely dead and safe to remove, restoring the freedom to evolve that schema-diff alone takes away. The two tools are not redundant; each fixes the other's blind spot. Schema-diff stops you from breaking what is used; Pact tells you what is no longer used so you can clean it up. Run them together and you get both safety and evolvability — exactly the pairing the [compatibility post](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) argues you must hold simultaneously.

## Test the errors and edge cases, not just the happy path

There is one failure mode of contract testing that deserves its own section, because it quietly undoes most of the value: **pinning only the happy path.** The first pact anyone writes asserts the `200`/`201` success case, because that is the case the demo exercises. But recall the lesson from the [error-design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract): *the part of the contract that matters most under pressure is the failure path*, and the failure path is exactly the part nobody writes a contract for.

![a before and after comparison showing a contract that pins only the success response leaving the error body unprotected versus one that pins the error and edge case responses too](/imgs/blogs/contract-testing-consumer-driven-contracts-and-schema-diffs-7.png)

Walk it through with the running example. The checkout service does not only read successful payments; it reads *declined* ones. When a card is declined, the payments API returns `402` with a `problem+json` body carrying a machine-readable `code` of `card_declined`. The checkout service has a `switch` on that code that decides whether to prompt for a new card (`card_declined`), retry later (`processor_unavailable`), or abandon (`fraud_blocked`). If checkout's pact pins *only* the `201`, then the payments team is free to reshape the error body — rename `code` to `error_code`, change `card_declined` to `declined`, move the field into a nested object — and *every contract test stays green*, while checkout's `switch` falls into its default branch and mishandles a real declined payment. The break has merely moved from the happy path to the error path, where it is more dangerous because it surfaces only when something is already going wrong.

So a contract test that is worth running pins the cases the consumer's code actually *branches on*:

```javascript
it("handles a declined payment", () => {
  provider
    .given("payment pay_decl1 was declined by the card network")
    .uponReceiving("a request for a declined payment")
    .withRequest({ method: "GET", path: "/payments/pay_decl1",
                   headers: { Accept: "application/json" } })
    .willRespondWith({
      status: 402,
      headers: { "Content-Type": "application/problem+json" },
      body: {
        type: like("https://errors.example.com/card-declined"),
        title: like("The card was declined"),
        status: 402,
        code: regex("card_declined", "card_declined"),
      },
    });

  return provider.executeTest(async (mockServer) => {
    const result = await getPayment(mockServer.url, "pay_decl1");
    expect(result.declineReason).toBe("card_declined"); // checkout branches on this
  });
});
```

The rule of thumb is precise: **the contract should pin every response shape the consumer's control flow actually depends on, and no more.** If checkout branches on the `code`, pin the `code`. If checkout reads a `Retry-After` header on a `429` to schedule a retry, pin that header. If checkout treats every `4xx` the same way (just shows a generic error), then a single representative error pact may be enough. The pact mirrors the consumer's *real* decision surface — which is exactly why it is consumer-*driven*. The same discipline applies to schema-diff: make sure your error schemas (the `problem+json` types, the documented error `code` enum) are *in the spec* and therefore *under the diff gate*, so renaming an error code is flagged as breaking just as loudly as renaming a response field.

Fixtures and examples in your spec do double duty here. The `examples` you write into your OpenAPI document are not just documentation — they can be *executed* as tests. A tool like Schemathesis can read your OpenAPI examples (and generate property-based variations) and fire them at a running service, asserting that real responses validate against the declared schema. That turns your `examples` block from prose into an assertion: a fixture is a test that the provider keeps producing the shape it claims to.

## Case studies: how this works in the wild

**Pact and the broker in an internal mesh.** Pact began at the Australian consultancy DiUS and grew into a polyglot ecosystem (`pact-jvm`, `pact-js`, `pact-python`, `pact-go`, and more) precisely because the problem it solves — independent teams shipping services that must agree on a wire — is universal in a microservices org. The pattern that organizations converge on is the one described above: consumers publish pacts to a shared broker tagged with their git branch and version; providers verify against the pacts of consumers that are *on main* or *currently deployed*; and every deploy pipeline runs `can-i-deploy` against the target environment. The broker's *network graph* and *matrix* views become the org's living map of who-depends-on-whom, which is often the first time anyone has had an accurate, automatically-maintained dependency graph of their own services. The Pact documentation is explicit that this is a tool for *known* consumers and *not* a fit for public APIs — a discipline worth honoring.

**Buf and the schema registry for Protobuf.** The Buf project built `buf breaking` and the Buf Schema Registry around exactly the field-number hazard we walked through. In a large gRPC estate, `.proto` files are the contract and they are shared across many services and many generated-client languages. Buf's model is to register a module's schema as a versioned artifact, then run `buf breaking --against` that published baseline in every pull request, so a wire-incompatible or source-incompatible change is rejected before it can be tagged and consumed. This is provider-contract / schema-diff thinking applied to Protobuf, and it is the mechanism that lets a large organization let many teams edit shared `.proto` files without periodically corrupting each other's wire formats.

**Stripe's additive-only evolution.** Stripe is the canonical example of evolving a heavily-used public API for over a decade with minimal breakage, and the relevant lesson here is *what their constraints imply about testing.* Because their consumers are unknown (every developer on the internet), they cannot use consumer-driven contracts; their discipline is instead a strict *additive-only* evolution policy, pinned API versions per account, and an internal compatibility layer that translates old request/response shapes to new ones. The testing analogue of "additive only forever" is precisely a schema-diff gate that refuses any non-additive change to the published surface — the public-API branch of our decision tree. The point is not that Stripe runs `oasdiff` (their tooling is internal); it is that the *rule* a public API must enforce is exactly the rule a schema-diff linter mechanizes.

**GraphQL schema checks at scale.** Teams running large GraphQL APIs (the pattern Apollo's tooling and `graphql-inspector` are built for) face the same removal hazard with an extra twist: a GraphQL client requests *only the fields it uses*, so the provider can often see, from logged operations, exactly which fields are live. The mature setup combines a schema *diff* check on every pull request (fail on breaking SDL changes) with *operation safelisting* — comparing the proposed schema change against the set of queries clients have actually run, so removing a field nobody queries is allowed while removing a field someone queries is blocked. That is a fascinating hybrid: a schema-diff gate informed by *observed* consumer behavior, which is consumer-driven thinking arrived at from the opposite direction. (For the GraphQL fundamentals — SDL, resolvers, the N+1 trap — see the dedicated GraphQL post later in this series.)

**The broker's dependency matrix as a planning tool.** A subtle, underappreciated payoff of running Pact across a service mesh is that the broker becomes the org's authoritative, *automatically maintained* dependency graph. Most organizations' "service dependency diagrams" are hand-drawn, out of date the day they are committed, and wrong in exactly the ways that hurt during an incident. The broker's matrix is different: it is derived from the pacts services actually publish and verify, so it reflects *real, exercised* dependencies, not someone's memory of them. When the payments team plans to deprecate a field, they do not poll every team in a chat channel asking "do you use `currency`?" — they read the matrix and see precisely which consumer versions reference it. When an incident hits the payments API, the on-call engineer reads the matrix to know, immediately and accurately, which consumers are in the blast radius. The contract tests pay for themselves twice: once as a gate that blocks breaks, and again as a living map that makes planning and incident response faster. This is the same role the [system-design schema-evolution-at-scale post](/blog/software-development/system-design/schema-and-api-evolution-at-scale) assigns to a central schema registry — the difference is that the broker's map is built from *consumer-declared* dependencies rather than from the provider's published schema, so it captures who-actually-uses-what rather than what-is-merely-offered.

**A note on accuracy.** Each of these tools (Pact, Buf, oasdiff, graphql-inspector) is real and the workflows above match their documented design, but the *internal* contract-testing setups of large API companies are mostly not public in detail. Where I describe Stripe's additive-only discipline or a GraphQL operation-safelisting flow, I am describing the *publicly documented pattern* and the *constraint that forces it*, not claiming knowledge of any company's private CI. The reasoning — public consumers force schema-diff, known consumers enable Pact — holds regardless of which exact tool any given team runs.

## When to reach for contract testing, and when not to

Every technique here is a trade-off, and the failure modes are as instructive as the successes. Be decisive about when *not* to do this.

**Reach for schema-diff linting essentially always.** It is cheap (one CI step), requires nothing from consumers, and catches the universal structural breaks. There is almost no API — public or internal, REST or gRPC or GraphQL — that should not have a breaking-change gate on its schema. If you adopt only one thing from this post, adopt this.

**Reach for consumer-driven Pact when you have a known, internal set of consumers whose tests you can run.** The classic fit is a service mesh inside one organization. The payoff scales with the number of consumers and the rate of independent change: the more teams depend on you and the faster everyone ships, the more a per-pull-request "you would break checkout" signal is worth.

**Do not use Pact for a public API with unknown consumers.** This is the most common over-application. You cannot write pacts for callers you have never met; the broker has no record of them; verification has nothing to verify against. Reaching for Pact here gives you a false sense of coverage — you will faithfully verify the three internal consumers you know about while the ten thousand external ones remain entirely unprotected. For public APIs, the spec is the contract and schema-diff is the gate.

**Do not over-tighten your matchers.** A pact that asserts exact values instead of types and shapes turns into a brittle snapshot test that fails on every irrelevant change and trains the provider team to ignore verification failures. Pin the *shape and type* the consumer's code depends on; use `like`, `integer`, `regex`, and array-element matchers; reserve exact-value matching for the rare case where the literal value genuinely matters (an enum the consumer branches on).

**Do not pin only the happy path.** As argued above, the error and edge-case responses are exactly the ones the consumer branches on under pressure, and they are the ones nobody writes a contract for by default. An undefended error body is the most dangerous gap, because the break surfaces only when something is already going wrong.

**Do not let Pact lull you into skipping schema-diff, or vice versa.** They cover different blind spots. Pact knows what consumers depend on but only the consumers you ran; schema-diff knows the whole declared surface but not who depends on what. On an internal mesh, run both. Pact is your semantic, real-dependency net; schema-diff is your structural, universal floor.

**Do not treat contract tests as a substitute for any other layer.** Contract tests verify the *boundary shape*; they say nothing about whether your business logic is correct, your database queries are right, or your service stays up under load. Keep your unit, service, and integration tests. Contract tests let you shrink the *end-to-end* tier, not the rest of the pyramid.

## Key takeaways

- A contract break lives *between* two services and therefore inside neither one's test suite; provider and consumer tests can both be green while the real wire shape has drifted. Catching it requires a test whose *expectation* is owned by one side and whose *satisfaction* is verified by the other.
- End-to-end tests do catch contract breaks, but late, slowly, flakily, and after both sides are already deployed together. Contract tests give you the same cross-service guarantee at unit-test speed, pre-merge, with the other side replaced by a recorded, verified artifact.
- **Schema-diff linting** (`oasdiff` for OpenAPI, `buf breaking` for Protobuf, `graphql-inspector` for GraphQL) mechanizes the compatibility rules: it compares your new schema to the last published one and fails the build on any breaking diff. It needs nothing from consumers, so it is the right gate for *public* APIs — and it should be on essentially every API.
- In Protobuf the field *number* is the wire identity; never reuse a number, always `reserved` a removed one, and let `buf breaking` enforce it — reusing a number silently corrupts data with no exception.
- **Consumer-driven Pact** flips authority to the consumer: each consumer writes a pact describing only what it actually depends on, the provider verifies it satisfies every consumer's pact in CI, and the broker's `can-i-deploy` gate proves the *exact* pair of versions about to run together is compatible.
- A pact contains only the consumer's real dependency surface, which is far smaller than the full API — that is what tells the provider precisely how much freedom it has to evolve, field by field.
- Pin the *shape and type* the consumer branches on, not exact values; over-tight matchers create brittle snapshot tests that get ignored.
- Pin the error and edge-case responses, not just the happy path — the failure path is the part the consumer branches on under pressure and the part nobody writes a contract for by default.
- Choose by consumer topology: **known internal consumers → Pact**; **unknown public consumers → published spec + schema-diff**; **named external partners → bi-directional**. Most real orgs run schema-diff everywhere *and* Pact on the internal edges.

## Further reading

- Pact documentation — the consumer test, provider verification, the broker, and `can-i-deploy`: <https://docs.pact.io/>
- Ian Robinson and Martin Fowler, "Consumer-Driven Contracts: A Service Evolution Pattern" — the original articulation of the idea: <https://martinfowler.com/articles/consumerDrivenContracts.html>
- Buf documentation — `buf breaking`, the rule sets, and the Buf Schema Registry for Protobuf: <https://buf.build/docs/>
- `oasdiff` — the OpenAPI diff and breaking-change detector: <https://www.oasdiff.com/>
- `graphql-inspector` — schema diffing and breaking-change detection for GraphQL: <https://the-guild.dev/graphql/inspector>
- Pactflow on bi-directional contract testing — reconciling a provider OpenAPI spec with consumer pacts: <https://docs.pactflow.io/docs/bi-directional-contract-testing/>
- Within this series: the [intro hub on the API as a contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the rules of [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change); [schema evolution — adding, removing, renaming fields safely](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely); the [spec-first OpenAPI workflow](/blog/software-development/api-design/openapi-and-the-spec-first-workflow-design-mock-generate); and the [capstone API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- Going deeper / out of series: [schema and API evolution at scale](/blog/software-development/system-design/schema-and-api-evolution-at-scale) for running these gates across hundreds of services and a shared schema registry.

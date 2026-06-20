---
title: "OpenAPI and the Spec-First Workflow: Design, Mock, Generate"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Write one machine-readable OpenAPI 3.1 document, then let it generate your docs, a live mock server, client and server SDKs, request validation, and a CI gate that blocks breaking changes — the spec-first workflow that makes the rest of API design automatable."
tags:
  [
    "api-design",
    "api",
    "openapi",
    "spec-first",
    "json-schema",
    "rest",
    "http",
    "mocking",
    "codegen",
    "developer-experience",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-1.png"
---

A payments team I worked with shipped a new `POST /payments` endpoint on a Friday. The backend engineer had been careful: she annotated the handler, the framework auto-generated a Swagger page, and the mobile team grabbed it Monday morning to start integrating. By Wednesday the mobile build was broken. The annotation said the response field was `amount` (an integer of minor units, cents); the actual handler had been refactored on Tuesday to return `amount_minor`, and nobody re-ran the annotation step. The published spec was a lie, and three engineers spent an afternoon discovering it the hard way — by `curl`ing the real server and diffing the response against a document that claimed to describe it.

That is the failure mode at the center of this post. The spec — the machine-readable description of your API — is only useful if it is *true*. And there are exactly two ways to keep it true: generate it from the code and pray the annotations stay in sync, or write the spec first, treat it as the contract, and make the code conform to it. This is the **spec-first vs code-first** decision, and it is the most consequential workflow choice in the whole series. Get it right and a single file fans out into your docs, a mock server the frontend builds against before the backend exists, generated SDKs, live request validation, contract tests, and a CI gate that refuses to merge a breaking change. Get it wrong and you ship documents that describe an API you no longer run.

![a diagram showing a single OpenAPI document fanning out to docs, a mock server, generated SDKs, request validation, breaking-change diffing, and linting as derived artifacts](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-1.png)

By the end of this post you will be able to: read and write an OpenAPI 3.1 document from `openapi` down through `paths` and `components`; decide between spec-first and code-first with eyes open; stand up a mock server from a spec so your frontend never waits on the backend; generate a typed client and a server stub with one command; validate live traffic against the spec at your gateway; and wire a CI gate that catches breaking changes before they reach a caller. This is the post where the rest of the series — the resource model, the error envelope, the pagination rules, the versioning policy — stops being prose you have to remember and becomes a file a machine can enforce. As always in [this series](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the question we keep returning to is: *what does the caller get to assume, and can I change this later without breaking them?* A good spec is how you write that assumption down so a tool can hold you to it.

## What OpenAPI actually is

Let me define the term precisely, because it carries some history. **OpenAPI** is a standard, language-agnostic format for describing an HTTP API. It is a single document — usually YAML, sometimes JSON — that says: here are my endpoints, here are the methods on each, here are the parameters and request bodies they accept, here are the responses they return with their status codes and shapes, and here is how you authenticate. It describes the *wire contract* without saying a word about the language or framework that implements it. A Go service, a Python service, and a Node service that expose the same endpoints have the same OpenAPI document.

The format used to be called **Swagger**. In 2015 the Swagger specification was donated to the Linux Foundation and renamed **OpenAPI**; "Swagger" now refers to the *tooling* (Swagger UI, Swagger Editor, Swagger Codegen) built by SmartBear around the standard, not the standard itself. So when you hear "Swagger spec," people almost always mean OpenAPI. The current version is **OpenAPI 3.1**, released in 2021, and its single most important property is this: **3.1 is fully aligned with JSON Schema**. Earlier versions (2.0, 3.0) used a *subset* of JSON Schema with subtle incompatibilities — `nullable: true` instead of a type union, no `oneOf`/`anyOf` at the top of a schema, no `$schema`. In 3.1 a schema in your OpenAPI document *is* a JSON Schema document. That alignment is what lets the same type definitions flow between your HTTP API, your event contracts, and a standalone validator. We will come back to why that matters.

A few terms before we go further, defined the first time they appear:

- A **resource** is a thing the API exposes — a payment, an order, a refund. It usually maps to a noun and a URI.
- A **representation** is the concrete bytes you send over the wire for a resource — typically a JSON object describing a payment.
- A **media type** (or content type) is the label for those bytes, like `application/json`. The spec ties a body to its media type.
- A **schema** is a description of a representation's shape — its fields, their types, which are required. In OpenAPI 3.1, a schema is JSON Schema.
- A **`$ref`** is a reference — a pointer from one place in the document to a definition elsewhere, so you write a thing once and reuse it.
- **Codegen** (code generation) means a tool reads the spec and emits source code — a typed client, a server skeleton, model classes.
- A **mock server** is a fake server that reads the spec and returns example responses, so callers can integrate before the real backend is built.

It helps to be clear about what OpenAPI is *not*, because the boundary is where people get confused. OpenAPI describes the *shape* of the contract — endpoints, methods, types, status codes, auth schemes — but it does not describe *behavior*. It cannot say "this endpoint is idempotent," only that it accepts an `Idempotency-Key` header (the idempotency semantics live in your prose and your implementation). It cannot say "a `409` means the order is already paid," only that a `409` response exists with a given body shape (the meaning lives in the `description` and your docs). It is a description of the *interface*, not a specification of the *semantics* behind it. This matters because it sets the boundary of what the tooling can enforce: a validator can reject a malformed body, but it cannot reject a *semantically wrong* response that happens to have the right shape. The spec automates the shape contract; the behavior contract still rides on your design discipline, your tests, and your docs. Knowing this keeps you from over-trusting the spec — it is a powerful guardrail, not a proof of correctness.

OpenAPI is, in short, the machine-readable form of every *structural* contract decision the rest of this series teaches. The error envelope you designed (RFC 9457 `problem+json`), the cursor pagination you chose, the bearer-token auth — all of their shapes can be written down in this one document, and once they are, every tool downstream can read it instead of guessing.

## Spec-first vs code-first: the workflow that decides everything

There are two ways to end up with an OpenAPI document, and they are not symmetric.

**Code-first** (sometimes "design-second" or "annotation-driven"): you write the handlers, decorate them with annotations or attributes, and a framework plugin reads those annotations at build time to *emit* the spec. In FastAPI the spec falls out of your Pydantic models and type hints; in Spring you sprinkle `@Operation` and `@Schema`; in Express you add JSDoc comments and run a generator. The spec is a *byproduct* of the code.

**Spec-first** (also "design-first" or "contract-first"): you write the OpenAPI document by hand — or in a visual editor like Stoplight Studio — *before* you write any handler. The spec is reviewed, agreed, and frozen as the contract. Then the backend implements it and the frontend consumes it, both reading the same file. The code is a *consequence* of the spec.

The difference looks cosmetic until you trace the consequences. Here is the principle, stated as a rule you can defend:

> **A spec generated from code can only ever describe the code; a spec the code is generated from can describe the agreement.** Code-first means the spec follows the implementation, so the implementation is free to drift ahead of it — and *will*, because the path of least resistance during a refactor is to change the code and forget the annotation. Spec-first inverts the dependency: the implementation is checked against the spec, so the spec cannot silently fall behind. Drift becomes a *test failure* instead of a *Wednesday surprise*.

The argument is about the *direction of the dependency arrow*. Whatever is downstream drifts; whatever is upstream is the source of truth. In code-first, the spec is downstream of the code, so the spec drifts. In spec-first, the code is downstream of the spec, so the code is the thing you verify. You can make code-first safe — but only by adding the same CI gate spec-first gives you for free, namely a step that diffs the generated spec against a checked-in baseline and fails on unexpected change. Most teams never add it, which is why most code-first specs are quietly wrong.

![a matrix comparing spec-first and code-first across drift risk, parallelism, reviewability, and upfront effort](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-2.png)

The second consequence is **parallelism**. With spec-first, the moment the contract is agreed, the frontend team can spin up a mock from it and build their entire feature against fake-but-conformant responses, while the backend team implements the real thing — *in parallel*, against the same document. Neither blocks the other. With code-first, the spec does not exist until the backend handler exists, so the frontend waits. On a team where the frontend and backend are different people (which is most teams), spec-first can collapse a two-week serial dependency into one week of parallel work.

The third is **reviewability**. A 200-line YAML document is a thing a human can read in a pull request and reason about: *is this the resource shape we agreed? is the error envelope consistent with our other endpoints? did we forget the `idempotency-key` header?* A code-first spec is an emergent property of scattered annotations across dozens of files; you cannot review the contract as a single artifact because it does not exist as one until build time.

Here is the honest cost: spec-first asks you to do design work up front, before the dopamine of a running endpoint. For a throwaway internal tool with one consumer who sits next to you, that cost may not pay back, and code-first is fine. The recommendation, then, is sharp: **for any new public, partner, or cross-team contract API — anything where the caller is someone you will not be able to ping on Slack — design the spec first.** For a private endpoint with a single co-located consumer, code-first with a drift gate is a reasonable shortcut. We will return to this in the when-to section.

| Dimension | Spec-first | Code-first |
| --- | --- | --- |
| Source of truth | The hand-written spec | The handler code |
| Drift risk | Low — code is verified against spec | High — annotations lag refactors |
| Parallel frontend/backend | Yes — mock from spec on day one | No — frontend waits on backend |
| Reviewable as one artifact | Yes — one YAML in a PR | No — emergent from annotations |
| Upfront effort | Higher — design before code | Lower — annotate as you go |
| Best fit | Public, partner, cross-team APIs | Private, single co-located consumer |

### Making code-first safe — the drift gate

Before we leave the comparison, let me be fair to code-first, because dismissing it outright is wrong and a lot of excellent APIs are built that way. FastAPI, in particular, derives a genuinely good OpenAPI 3.1 document from your Python type hints and Pydantic models, and a disciplined team can run a perfectly safe code-first workflow. The catch is the one word in that sentence: *disciplined*. Code-first is safe if and only if you add the enforcement that spec-first gives you for free. Here is what that discipline looks like concretely.

The trick is to *snapshot* the generated spec and diff it on every build. You commit a `openapi.snapshot.yaml` to the repo. In CI, you regenerate the spec from the running code, and you fail the build if it differs from the snapshot without the snapshot being updated in the same pull request. That single step converts the invisible-drift failure mode into a loud, reviewable test failure — the same property spec-first has, reached from the other direction.

```bash
# In CI: regenerate the spec from the running app and diff it
# against the committed snapshot. Fail if they differ.
python -c "import json, app; print(json.dumps(app.app.openapi()))" \
  > /tmp/openapi.generated.json

# oasdiff can compare; or a plain diff if you normalize first
oasdiff diff openapi.snapshot.json /tmp/openapi.generated.json \
  --fail-on-diff
```

Now a refactor that renames `amount` to `amount_minor` in the code changes the generated spec, the diff fails, and the engineer must *consciously* update the snapshot — which surfaces in the PR as a contract change a reviewer can see and a downstream team can be warned about. The original Wednesday-surprise becomes a Tuesday-PR-comment. Pair this with oasdiff's *breaking-change* classification (not just "is it different" but "is it a breaking difference") and code-first inherits spec-first's safety net.

So the real dividing line is not spec-first versus code-first; it is **gated versus ungated.** An ungated workflow of either kind lets the spec lie. A gated spec-first workflow and a gated code-first workflow converge on the same guarantee: the published contract is true, and a change to it is visible and reviewed. The reason this post argues *spec-first* for new contract APIs is that spec-first makes the gate the *default* — the spec exists before the code, so there is nothing to drift from on day one — whereas code-first makes the gate something a tired engineer has to remember to set up. Defaults win. But if your framework's code-first story is excellent and your CI gate is real, you have built spec-first's enforcement by another route, and that is fine.

#### Worked example: classifying a change as breaking

Let me make the gate concrete with a change classification, because "breaking versus non-breaking" is the judgment the whole gate automates. Suppose the spec currently has this response field on `Payment`:

```yaml
status:
  type: string
  enum: [requires_capture, succeeded, failed]
```

Consider three candidate changes and how oasdiff classifies each, with the reasoning from the compatibility rules:

1. **Add a new enum value `disputed` to the response.** This is *non-breaking* — an old client that switches on the three known values will hit its default branch for `disputed`, which is the tolerant-reader behavior. oasdiff reports this as a warning, not an error. (Adding an enum value to a *request* the server validates *is* breaking in the other direction — the server now rejects a value it used to accept, or accepts one old clients cannot produce; direction matters, and oasdiff knows which side the schema is on.)
2. **Remove `failed` from the response enum.** This is *breaking* — a client that has code to handle a `failed` payment has now lost a documented value, and worse, if the server still emits it, the response no longer matches the spec at all. oasdiff flags this as an error and the gate fails.
3. **Rename `status` to `state`.** This is *breaking* — it removes a documented response field, which is the canonical break (the renamed field that 500'd the mobile client in the opening story). The gate fails, and you are forced to choose: a new version, or the expand-and-contract migration where you add `state` as a new field, keep `status` populated for a deprecation window, and remove it only after clients have moved.

The value here is not the classification itself — an experienced engineer knows these rules — but that the *machine* applies them on *every* change, including the ones a junior engineer makes at 6pm on a Friday. The compatibility rules from the [backward-and-forward compatibility post](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) stop being lore you hope everyone remembers and become a gate nobody can bypass.

## The anatomy of an OpenAPI document

Let us build one. We will describe a single endpoint from our running Payments/Orders platform: creating a payment. An OpenAPI 3.1 document has a small number of top-level keys, and almost everything interesting lives under two of them — `paths` and `components`.

![a layered stack showing the top-level structure of an OpenAPI document from openapi version through info, servers, paths, components, and security schemes](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-3.png)

Here are the top-level keys, from the outside in:

- **`openapi`** — the version of the *format*, e.g. `3.1.0`. Not your API version; the spec format's version. Tooling reads this first to know which rules apply.
- **`info`** — metadata about *your* API: its `title`, its `version` (this *is* your API's version, e.g. `2024-10-01` or `1.4.0`), a `description`, contact and license. The `info.version` is what you bump when the API changes.
- **`servers`** — the base URLs the API is served from, typically a production URL and a sandbox URL. A client or mock prepends these to each path.
- **`paths`** — the heart of the document: a map from URL path (`/payments`) to the operations on it (`post`, `get`), each with its parameters, request body, and responses.
- **`components`** — reusable building blocks: `schemas` (the type definitions), `parameters`, `responses`, `examples`, and `securitySchemes`. Everything here is referenced from `paths` with `$ref` so you define it once.
- **`security`** — which security schemes apply, globally or per-operation.

Let me show the skeleton, then fill in the payment operation.

```yaml
openapi: 3.1.0
info:
  title: Acme Payments API
  version: "2024-10-01"
  description: Create and manage payments and refunds for the Acme commerce platform.
  contact:
    name: Acme API Support
    url: https://developer.acme.example/support
servers:
  - url: https://api.acme.example/v1
    description: Production
  - url: https://sandbox.acme.example/v1
    description: Sandbox (test cards, no real money moves)
paths: {}        # operations go here
components: {}   # reusable schemas, responses, security go here
```

### Paths and operations

A `path` is a URL template; the operations under it are keyed by HTTP method. Each operation declares its `parameters` (path, query, and header inputs), its `requestBody`, and its `responses` keyed by status code. Here is `POST /payments`, the spine of this whole post:

```yaml
paths:
  /payments:
    post:
      operationId: createPayment
      summary: Create a payment
      description: >
        Charges a payment method for an order. Supply an Idempotency-Key
        header so a retried request never charges twice.
      tags: [Payments]
      parameters:
        - $ref: "#/components/parameters/IdempotencyKey"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/PaymentCreate"
            examples:
              card_charge:
                $ref: "#/components/examples/CardChargeRequest"
      responses:
        "201":
          description: Payment created.
          headers:
            Location:
              description: URL of the created payment.
              schema: { type: string, format: uri }
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Payment"
              examples:
                created:
                  $ref: "#/components/examples/PaymentCreated"
        "422":
          $ref: "#/components/responses/ValidationProblem"
        "409":
          $ref: "#/components/responses/IdempotencyConflict"
      security:
        - bearerAuth: [payments:write]
```

Notice how thin the operation is. It does not *define* the payment shape, the idempotency header, or the error responses — it *references* them. The operation reads like an index: here is the input, here is the success output, here are the failure outputs, here is the auth required. That is the spec-first habit: keep operations as references into a well-organized `components` block so a reviewer reads the contract, not a wall of inline definitions.

The `operationId` deserves a note. It is a stable, unique name for the operation — `createPayment` — and codegen uses it directly. Generate a client from this spec and you get a method called `createPayment(...)`. Pick these names with the same care you would pick public method names, because that is exactly what they become.

### Components and `$ref` reuse

Now the part that pays for itself. Define the payment schema once, under `components/schemas`, and reference it from anywhere — the request, the response, a webhook payload, another endpoint. Here is the schema block, including the security scheme and the reusable error response:

```yaml
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: An OAuth2 access token in the Authorization header.

  parameters:
    IdempotencyKey:
      name: Idempotency-Key
      in: header
      required: true
      description: A unique key so a retried request is not processed twice.
      schema: { type: string, format: uuid }

  schemas:
    Money:
      type: object
      required: [amount_minor, currency]
      properties:
        amount_minor:
          type: integer
          description: Amount in the currency's minor unit (cents for USD).
          minimum: 0
          examples: [4999]
        currency:
          type: string
          description: ISO 4217 currency code.
          pattern: "^[A-Z]{3}$"
          examples: ["USD"]

    PaymentCreate:
      type: object
      required: [order_id, amount, payment_method_id]
      properties:
        order_id:
          type: string
          format: uuid
        amount:
          $ref: "#/components/schemas/Money"
        payment_method_id:
          type: string
          description: An identifier for a stored card or wallet.
        capture:
          type: boolean
          default: true
          description: If false, authorize only; capture later.

    Payment:
      allOf:
        - $ref: "#/components/schemas/PaymentCreate"
        - type: object
          required: [id, status, created_at]
          properties:
            id:
              type: string
              format: uuid
            status:
              type: string
              enum: [requires_capture, succeeded, failed]
            created_at:
              type: string
              format: date-time
```

Look at what `$ref` bought us. `Money` is defined once and used inside `PaymentCreate`, which is itself reused inside `Payment` via `allOf` (which composes schemas — a `Payment` is everything a `PaymentCreate` is, plus server-assigned fields). When you decide next quarter that `Money` needs a `display` field, you edit one schema and every consumer of it — the request, the response, the refund, the webhook — picks it up. The alternative, copy-pasting the money shape into a dozen operations, guarantees they drift apart the first time someone updates one and misses the others. This is the same robustness logic that runs through the whole series: define the contract once, reference it everywhere, and a change is a single edit a reviewer can see.

The `examples` field (note: in 3.1, schema-level examples are an *array* under the keyword `examples`, distinct from the OpenAPI `example`/`examples` used at the media-type level) is not decoration. Examples are what the mock server returns, what the docs render in their "try it" panel, and what a developer copies into their first `curl`. An inaccurate example is worse than none — it teaches the caller a shape that does not exist. We will treat keeping examples accurate as a first-class maintenance task.

### Writing and validating the spec

You do not have to write OpenAPI in a bare text editor, though many people do — and you should not rely on getting the YAML right by eye, because a misplaced indent or a typo'd `$ref` produces a document that is *valid YAML* but *invalid OpenAPI*, and the failure can be silent until a tool chokes on it. There are three common authoring approaches:

- **Hand-edit the YAML** with an editor extension that validates OpenAPI as you type (the Swagger Editor, the Redocly or Spectral VS Code extensions). This is the most common spec-first workflow and the one that keeps the document reviewable as a single artifact, because the file is the thing you edit.
- **Use a visual editor** like Stoplight Studio, which lets you build the spec through a form-based UI and writes the YAML for you. This is valuable when non-engineers — product managers, technical writers — need to read or contribute to the contract.
- **Split the spec across multiple files** with `$ref`s that point at other files (`$ref: "./schemas/Payment.yaml"`), then *bundle* them into a single document for tools that want one file. Redocly's CLI and `swagger-cli` both bundle. Large specs almost always do this — a 5,000-line single file is unreviewable, but a `paths/` directory and a `schemas/` directory with one file each is navigable.

Whichever you choose, *validate the spec in CI as its own step*, independent of the code:

```bash
# Validate the document is well-formed OpenAPI 3.1 (not just valid YAML)
npx @redocly/cli lint openapi.yaml

# Bundle a multi-file spec into one document for downstream tools
npx @redocly/cli bundle openapi.yaml -o dist/openapi.bundled.yaml
```

The reason to validate the spec separately from the code is sequencing: in spec-first, the spec is merged and consumed (by the mock, by the frontend) *before* the backend exists, so the spec must be provably valid on its own. A broken `$ref` or a malformed schema would otherwise surface as a confusing mock-server crash days later. Catch it at the source, in the spec's own PR.

#### Worked example: a complete `POST /payments` fragment

Put the pieces together and you have a self-contained, valid OpenAPI 3.1 document for one endpoint. This is the artifact you would open a pull request with on day one of the project, before a single line of handler code exists:

```yaml
openapi: 3.1.0
info:
  title: Acme Payments API
  version: "2024-10-01"
servers:
  - url: https://api.acme.example/v1
paths:
  /payments:
    post:
      operationId: createPayment
      summary: Create a payment
      parameters:
        - name: Idempotency-Key
          in: header
          required: true
          schema: { type: string, format: uuid }
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: "#/components/schemas/PaymentCreate" }
            example:
              order_id: "9f1c0b2e-4d3a-4f5b-8c7d-1a2b3c4d5e6f"
              amount: { amount_minor: 4999, currency: "USD" }
              payment_method_id: "pm_card_visa"
              capture: true
      responses:
        "201":
          description: Payment created.
          headers:
            Location: { schema: { type: string, format: uri } }
          content:
            application/json:
              schema: { $ref: "#/components/schemas/Payment" }
              example:
                id: "pay_3Nk8xQ2eZvKYlo2C0abc1234"
                order_id: "9f1c0b2e-4d3a-4f5b-8c7d-1a2b3c4d5e6f"
                amount: { amount_minor: 4999, currency: "USD" }
                payment_method_id: "pm_card_visa"
                capture: true
                status: "succeeded"
                created_at: "2024-10-01T12:34:56Z"
        "422":
          $ref: "#/components/responses/ValidationProblem"
      security:
        - bearerAuth: [payments:write]
components:
  securitySchemes:
    bearerAuth: { type: http, scheme: bearer, bearerFormat: JWT }
  responses:
    ValidationProblem:
      description: The request body failed validation.
      content:
        application/problem+json:
          schema: { $ref: "#/components/schemas/Problem" }
          example:
            type: "https://errors.acme.example/validation"
            title: "Your request body is invalid."
            status: 422
            detail: "amount.amount_minor must be a non-negative integer."
            instance: "/payments"
  schemas:
    Money:
      type: object
      required: [amount_minor, currency]
      properties:
        amount_minor: { type: integer, minimum: 0 }
        currency: { type: string, pattern: "^[A-Z]{3}$" }
    PaymentCreate:
      type: object
      required: [order_id, amount, payment_method_id]
      properties:
        order_id: { type: string, format: uuid }
        amount: { $ref: "#/components/schemas/Money" }
        payment_method_id: { type: string }
        capture: { type: boolean, default: true }
    Payment:
      allOf:
        - $ref: "#/components/schemas/PaymentCreate"
        - type: object
          required: [id, status, created_at]
          properties:
            id: { type: string }
            status: { type: string, enum: [requires_capture, succeeded, failed] }
            created_at: { type: string, format: date-time }
    Problem:
      type: object
      required: [type, title, status]
      properties:
        type: { type: string, format: uri }
        title: { type: string }
        status: { type: integer }
        detail: { type: string }
        instance: { type: string }
```

That document is the contract. It says a payment is created with `POST /payments`, requires an idempotency key, charges \$49.99 (the `4999` minor units in USD), returns `201` with the created payment and a `Location` header, and returns a `problem+json` body on validation failure — and it ties auth to a `payments:write` scope. Everything that follows in this post is something a tool does *with this exact document* and nothing else.

## What the single spec unlocks

This is the payoff, and the reason spec-first is worth the upfront effort. From the one document above, you derive — without writing more code by hand — a whole toolchain. Let me organize it as a small taxonomy of what the spec generates, split into artifacts for humans and artifacts for machines.

![a tree grouping the artifacts generated from a spec into human-facing docs and machine-facing tools such as generators, mocks, and linters](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-6.png)

### Interactive documentation

Point a documentation renderer at the spec and you get reference docs that are *always* in sync with the contract, because they *are* the contract rendered. The common choices:

- **Swagger UI** — the classic interactive explorer. It renders every operation with a "Try it out" button that constructs a real request against your `servers` URL. Great for developers poking at a sandbox.
- **Redoc** (and Redocly's hosted tooling) — a clean, three-column reference doc optimized for reading, not poking. It renders nested schemas well, which matters for a payments API with deep request bodies.
- **Stoplight** — a platform combining a visual spec editor (Stoplight Studio), hosted docs, and a mock server, popular with teams that want non-engineers to read and even edit the spec.

The key property is that none of these are *written*; they are *rendered*. There is no separate docs repo to update when you add a field. You edit the spec, the docs regenerate. The class of bug where the docs say one thing and the API does another — the bug that broke the mobile team in the opening story — is structurally impossible when the docs are a projection of the same file the code conforms to.

The economics here are worth stating, because hand-written reference docs are where API teams quietly lose weeks. A reference page for a single endpoint describes the path, the method, every parameter with its type and constraints, the request body shape, every response code with its body shape, the auth required, and at least one example. That is the *same information* the spec already encodes. Maintaining both is maintaining two copies of the truth, and the docs copy is the one that rots — it is the one nobody's compiler checks. A team that writes reference docs by hand will, within a year, have docs that lie in a dozen small ways: a parameter renamed in the code but not the docs, a response field added but not documented, an example with a stale field. Rendering the docs from the spec deletes that entire category of work *and* that entire category of bug. The only docs you still write by hand are the *narrative* ones — the getting-started guide, the conceptual overview, the migration guide — which is exactly right, because those are the parts a machine cannot generate.

A practical detail that bites teams: the "Try it out" feature in Swagger UI makes a *real* request to your `servers` URL. That is wonderful against a sandbox and dangerous against production — you do not want a docs reader accidentally creating a real payment. The fix is to list a sandbox server first in `servers`, or to host the interactive docs only against the sandbox and the read-only Redoc reference against production. The spec supports both because `servers` is just a list; you choose which environment the "try it" button points at.

### Mock servers — build the frontend before the backend exists

This is the single most underrated artifact, and it is why spec-first unlocks parallelism. A mock server reads the spec and serves responses that conform to it — drawn from your `examples` if present, or generated to match the schema if not. The frontend team integrates against the mock on day one. **Prism** (from Stoplight) is the standard open-source tool here.

```bash
# Serve a mock of the Payments API straight from the spec
npx @stoplight/prism-cli mock openapi.yaml
# [CLI] …  Prism is listening on http://127.0.0.1:4010
```

Now the frontend hits `http://127.0.0.1:4010/payments` with a `POST` and gets back a `201` with a payment body that matches the schema — the `created` example, or a generated one. Crucially, Prism also runs in *validation mode*: send it a request that violates the spec (a missing `Idempotency-Key`, a non-integer `amount_minor`) and it returns the spec-defined `422`, so the frontend tests its error handling against the real contract too.

#### Worked example: mocking and calling the spec

Stand up the mock and exercise it end to end. First, run the mock; then call it as a client would:

```bash
# Terminal 1 — start the mock from the spec
npx @stoplight/prism-cli mock openapi.yaml --port 4010
```

```bash
# Terminal 2 — call the mock exactly as a real client would
curl -X POST http://127.0.0.1:4010/payments \
  -H "Authorization: Bearer sk_live_YOUR_KEY_HERE" \
  -H "Idempotency-Key: 0b8f2a1c-9d4e-4a6b-8c7d-1e2f3a4b5c6d" \
  -H "Content-Type: application/json" \
  -d '{
        "order_id": "9f1c0b2e-4d3a-4f5b-8c7d-1a2b3c4d5e6f",
        "amount": { "amount_minor": 4999, "currency": "USD" },
        "payment_method_id": "pm_card_visa",
        "capture": true
      }'
```

The mock answers with the spec's `201` example:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /payments/pay_3Nk8xQ2eZvKYlo2C0abc1234

{
  "id": "pay_3Nk8xQ2eZvKYlo2C0abc1234",
  "order_id": "9f1c0b2e-4d3a-4f5b-8c7d-1a2b3c4d5e6f",
  "amount": { "amount_minor": 4999, "currency": "USD" },
  "payment_method_id": "pm_card_visa",
  "capture": true,
  "status": "succeeded",
  "created_at": "2024-10-01T12:34:56Z"
}
```

No backend exists yet. The frontend has a working `201` to render a receipt against, a `422` to render an error against, and a `409` to handle an idempotency conflict — all defined by the contract, all available the moment the spec is merged. The backend team, in parallel, is implementing a handler that must produce exactly these shapes. When the real backend comes online, the frontend points its base URL from `127.0.0.1:4010` to the sandbox, and if the spec was the source of truth for both, nothing changes.

### Client and server code generation

The spec is a *type system over the wire*, so a generator can emit typed code in any language. **openapi-generator** (the open-source successor to Swagger Codegen) supports dozens of targets. Two directions:

- **Client SDKs** — emit a typed client library so callers do not hand-roll HTTP. The generated `createPayment(...)` method takes a typed `PaymentCreate`, sets the headers, and returns a typed `Payment`. Misspell a field and the compiler catches it, not production.
- **Server stubs** — emit a server skeleton with the routes, the request/response models, and a place to drop your business logic. The framework wiring (parsing, serialization, basic validation) is generated; you write only the handler body.

```bash
# Generate a typed TypeScript client from the spec
npx @openapitools/openapi-generator-cli generate \
  -i openapi.yaml \
  -g typescript-fetch \
  -o ./sdk-ts

# Generate a Python server stub (FastAPI) you fill in
npx @openapitools/openapi-generator-cli generate \
  -i openapi.yaml \
  -g python-fastapi \
  -o ./server-py
```

The generated TypeScript client gives the frontend something like:

```javascript
import { PaymentsApi, Configuration } from "./sdk-ts";

const api = new PaymentsApi(
  new Configuration({ basePath: "https://sandbox.acme.example/v1",
                      accessToken: "Bearer <token>" })
);

const payment = await api.createPayment({
  idempotencyKey: crypto.randomUUID(),
  paymentCreate: {
    orderId: "9f1c0b2e-4d3a-4f5b-8c7d-1a2b3c4d5e6f",
    amount: { amountMinor: 4999, currency: "USD" },
    paymentMethodId: "pm_card_visa",
  },
});
// payment.status is typed to "requires_capture" | "succeeded" | "failed"
```

The `payment.status` field is typed to exactly the three enum values from the spec. If the backend later adds a fourth status without a contract change, that is caught — either by codegen producing a new union (after a spec edit) or by the diff gate refusing the change. The SDK is downstream of the spec, which is exactly where you want it. We go deep on SDK ergonomics in the sibling post on [SDKs, code generation, and reference docs developers love](/blog/software-development/api-design/sdks-code-generation-and-reference-docs-developers-love); here the point is just that they fall out of the spec for free.

### Request and response validation at the gateway

Because the spec describes exactly what a valid request and response look like, you can validate *live traffic* against it. Put a validator at the gateway and every inbound request is checked against the spec before it reaches your handler; every outbound response is checked before it reaches the client. A request that violates the contract — missing required field, wrong type, unknown enum value — gets a spec-defined `422` at the edge, and your handler only ever sees valid input.

![a flow where a client request reaches the gateway, which validates against the spec and either forwards a valid request to the handler or returns a 422, then checks the response against the spec too](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-7.png)

This is a real defense-in-depth win: validation logic that would otherwise be hand-written (and inconsistently applied) per handler is centralized and *derived from the same spec the client was built against*. The Prism CLI does this in *proxy* mode; gateways like Kong and Envoy have OpenAPI validation plugins; and library middleware (`express-openapi-validator` for Node, `connexion` for Python) does it in-process. The principle: the spec is not just documentation, it is an *executable* description of validity, and you should run it against your traffic, not just print it.

Here is what in-process validation middleware looks like in Node. The middleware loads the spec once at startup, then validates every request against it before the handler runs:

```javascript
import express from "express";
import * as OpenApiValidator from "express-openapi-validator";

const app = express();
app.use(express.json());

app.use(
  OpenApiValidator.middleware({
    apiSpec: "./openapi.yaml",
    validateRequests: true,   // reject bad inputs before the handler
    validateResponses: true,  // catch the handler returning a bad shape
  })
);

// The handler only ever sees a request that matches PaymentCreate.
app.post("/payments", (req, res) => {
  // req.body is guaranteed to have order_id, amount, payment_method_id.
  const payment = createPayment(req.body);
  res.status(201).location(`/payments/${payment.id}`).json(payment);
});

// The validator throws a spec-shaped error; map it to problem+json.
app.use((err, req, res, next) => {
  res.status(err.status || 500).type("application/problem+json").json({
    type: "https://errors.acme.example/validation",
    title: err.message,
    status: err.status || 500,
    detail: err.errors?.map((e) => `${e.path} ${e.message}`).join("; "),
    instance: req.path,
  });
});
```

There is a subtle but important payoff in `validateResponses: true`. Request validation protects you from bad callers; *response* validation protects your callers from *you*. If a refactor makes the handler return a payment without the `created_at` field the spec promises, the response validator catches it in staging — before a client that depends on `created_at` ever sees the broken shape. That is the spec enforcing the contract in *both* directions, which is exactly what you want from a source of truth.

Now consider the gateway form, which is often preferable for a fleet because it validates uniformly across many services without each one wiring middleware. A Kong route with the OpenAPI validation plugin looks like this:

```yaml
# Kong declarative config: validate requests against the spec at the edge
services:
  - name: payments
    url: http://payments.internal:8080
    routes:
      - name: payments-route
        paths: ["/v1/payments"]
    plugins:
      - name: openapi-validation
        config:
          api_spec: "@openapi.yaml"
          validate_request_body: true
          validate_request_headers: true   # the Idempotency-Key check
          verbose_response: true           # return which field failed
```

Now a request missing the `Idempotency-Key` header never reaches the payments service — it is rejected at the gateway with a `422`, and the service is free to *assume* the header is present, because the edge guaranteed it. This is the same defense-in-depth layering the [API gateway post](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern) develops; here the new idea is that the gateway's validation rule is *not hand-written* — it is the spec, loaded as config. Change the spec, and the gateway's notion of "valid" changes with it. There is exactly one definition of a valid `POST /payments` in your whole system, and everything reads it.

The performance cost is real but modest and worth naming honestly. JSON Schema validation of a typical request body is microseconds to low milliseconds depending on body size and schema depth — a token amount against the tens-to-hundreds of milliseconds a payment handler spends in the database and the network. For a pathological case (a deeply nested body with thousands of array items validated against a recursive schema) it can matter, and the fix is to bound array sizes in the schema itself with `maxItems`, which both protects validation cost and is good API hygiene. The general rule: validating at the edge is cheap relative to the work the handler does, and the contract guarantee it buys is worth far more than the microseconds it costs.

### Breaking-change detection and contract testing

The spec is also a baseline you can *diff*. **oasdiff** compares two OpenAPI documents and classifies the differences as breaking or non-breaking — removing a response field, adding a required request field, narrowing an enum, all flagged as breaking; adding an optional response field, adding a new endpoint, flagged as safe. Wire it into CI and a pull request that breaks the contract fails before it merges:

```bash
# Fail CI if the PR's spec breaks the contract vs main
oasdiff breaking main/openapi.yaml HEAD/openapi.yaml --fail-on ERR
```

This is the same backward-compatibility discipline from the [contract testing post](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs), now automated against the spec itself. A schema diff catches the *shape* breaks; consumer-driven contract tests (Pact) catch the *semantic* expectations a specific consumer has. Together they form the gate that lets you change the API confidently — the second of the three goals in the series spine, *safe evolution*. We deliberately link out rather than re-derive: this post owns *the spec as the artifact you diff*; that post owns *the testing strategy around it*.

It is worth being precise about *what oasdiff knows that a plain `diff` does not*. A line-by-line text diff tells you the file changed; it has no idea whether a change is safe. oasdiff understands OpenAPI semantics: it knows that adding an optional response property is safe but adding a required request property is not; that widening an enum on a response is safe but narrowing it is not; that loosening a `minLength` is safe but tightening it can reject inputs that used to pass; that removing a `2xx` response or an entire operation breaks any client that depended on it. The `--fail-on ERR` flag fails CI only on the *breaking* class, letting safe additive changes through without ceremony — which is exactly the policy you want, because additive change should be frictionless and breaking change should require a deliberate decision.

The schema diff is necessary but not sufficient, and the gap is instructive. oasdiff can only see what the spec *says*; it cannot see what a specific consumer actually *depends on*. Suppose your spec marks `detail` on the `Problem` error as optional, but one mobile client has parsed `detail` into its error UI and will render an empty screen if it disappears. A schema diff that drops `detail` (it is optional, after all) passes — yet a real consumer breaks. That is the gap consumer-driven contract testing fills: the consumer publishes the subset of the contract it actually uses, and the provider's CI verifies it still satisfies *that* subset. A minimal consumer-side Pact looks like this:

```javascript
// Consumer (mobile app) declares what it needs from POST /payments
provider
  .uponReceiving("a successful payment creation")
  .withRequest({
    method: "POST",
    path: "/payments",
    headers: { "Idempotency-Key": like("0b8f2a1c-..."), },
    body: { order_id: like("9f1c0b2e-..."),
            amount: { amount_minor: like(4999), currency: like("USD") },
            payment_method_id: like("pm_card_visa") },
  })
  .willRespondWith({
    status: 201,
    body: { id: like("pay_..."), status: term({ matcher: "succeeded|failed", generate: "succeeded" }),
            detail: like("charged"), created_at: like("2024-10-01T12:34:56Z") },
  });
```

That Pact says, in effect, "this consumer relies on `id`, `status`, and `detail` being present." Run it against the provider and dropping `detail` now *fails*, even though the schema diff allowed it. The two tools are complementary: oasdiff guards the *published* contract; Pact guards the *used* contract. A serious API change pipeline runs both — schema diff to catch broad shape breaks early and cheaply, contract tests to catch the per-consumer semantic breaks the schema cannot express. And the broker's "can I deploy?" check ties them together: before you ship the provider, you ask whether every consumer's recorded contract still passes against the new version.

### Linting and governance

Finally, the spec is *lintable*. **Spectral** runs a rule set over the document and flags style violations — an operation missing a `description`, a path that is not kebab-case, an error response that is not `problem+json`, a `4xx` without an example. It is `eslint` for your API contract, and it is how an organization enforces consistency across hundreds of endpoints written by dozens of teams.

```bash
# Lint the spec against the org's style guide
npx @stoplight/spectral-cli lint openapi.yaml --ruleset .spectral.yaml
```

A ruleset is itself a small YAML file. Here is a fragment that enforces three real organizational conventions — every operation must have a description, every error response must use `problem+json`, and every operation must declare a security scheme:

```yaml
# .spectral.yaml — extends the built-in OpenAPI rules with org policy
extends: ["spectral:oas"]
rules:
  operation-description:
    description: Every operation needs a human description.
    given: "$.paths[*][get,post,put,patch,delete]"
    then: { field: description, function: truthy }
    severity: error
  error-uses-problem-json:
    description: 4xx and 5xx must use application/problem+json.
    given: "$.paths[*][*].responses[?(@property.match(/^[45]/))].content"
    then: { field: "application/problem+json", function: truthy }
    severity: error
  operation-has-security:
    description: Every operation must declare a security requirement.
    given: "$.paths[*][get,post,put,patch,delete]"
    then: { field: security, function: truthy }
    severity: warn
```

Run that against our payments spec and it will pass — every operation has a description, the `422` uses `problem+json`, and `createPayment` declares `bearerAuth`. Add a new endpoint that forgets the security requirement and Spectral flags it *in the design-review PR*, before any caller is exposed. This is the difference between a style guide that is a wiki page nobody reads and a style guide that is *executable* and runs on every change. The wiki page describes the convention; the ruleset enforces it.

That governance layer — the rules, the review board, the org-wide style guide — is the subject of the sibling post on [API governance and style guides](/blog/software-development/api-design/api-governance-and-style-guides-consistency-across-an-org). The spec is what makes governance *automatable*: you cannot lint a contract that lives in scattered annotations, but you can lint a single YAML file in every PR. The recurring theme across docs, mocks, codegen, validation, diffing, and linting is the same — *the spec is the input to a tool*, never a document a human keeps in sync by hand. That is the entire bet of spec-first: write the contract once, in a machine-readable form, and let machines do everything else.

## The spec-first workflow loop

Put the artifacts in order and you get a workflow — a loop the whole team runs every time the API changes. This is the practical shape of spec-first day to day.

![a timeline of the spec-first workflow showing design, review, mock, parallel build, validate, generate, and a CI gate as ordered stages](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-5.png)

1. **Design.** Write or edit the OpenAPI document. Add the new endpoint, the new field, the new error. This is where the contract decisions from the rest of the series get encoded.
2. **Review.** Open a pull request with *only the spec*. The team reviews the contract as a single artifact. Spectral runs and flags style issues; a human checks semantics. This is the cheapest possible place to catch a bad design — before any code exists.
3. **Mock.** Merge the spec, and a mock server updates automatically. The frontend now has a conformant fake to build against.
4. **Build in parallel.** Frontend and backend implement against the same contract. The backend writes the handler that produces the spec's shapes; the frontend writes the UI that consumes them. Neither blocks the other.
5. **Validate.** As the real backend comes online, validate its live traffic against the spec — in tests, in a staging gateway, in proxy mode. Any divergence between what the handler does and what the spec promises is caught here.
6. **Generate.** Regenerate the docs and the SDKs from the merged spec. Because they are derived, this is a build step, not a writing task.
7. **CI gate.** On every future change to the spec, run oasdiff and the contract tests. A breaking change fails CI and forces an explicit decision: is this a new version, or do we revise the change to be additive?

The loop's defining property is that the *contract is decided first and enforced last*. Design moves to the front, where changes are cheap; enforcement sits in CI, where it is automatic. The middle — build, validate, generate — is parallel and largely mechanical. Compare this to the code-first loop, where the contract is *discovered last* (when the spec is generated from finished code) and never enforced (no gate), and you see why the same number of steps produces a wildly different drift profile.

The most valuable step is the cheapest one: **review**. Catching a design problem in the spec PR — before any code exists — costs minutes; catching the same problem after the backend is built, the frontend has integrated, and a partner has shipped against it costs weeks and an apology email. This is the oldest result in software economics, that the cost of fixing a defect rises by roughly an order of magnitude at each stage it survives, applied to API contracts. Spec-first pulls the contract decision all the way to the front, into a small reviewable artifact, which is the stage where fixes are cheapest. A reviewer reading the spec PR can ask the questions that are expensive to answer later: *should `amount` be an object or two scalar fields? is `payment_method_id` the right name or should it match what the rest of our API calls a payment source? do we want `capture` as a boolean or a richer enum? is a missing idempotency key a `400` or a `428`?* Every one of those is a one-line edit in the spec and a multi-team migration after launch.

There is a second reason review is the high-leverage step: the spec PR is the natural place to apply the [resource-modeling](/blog/software-development/api-design/resource-modeling-turning-a-domain-into-nouns-and-uris) and [error-design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) judgment from earlier in the series. Those posts taught you *what* a good contract looks like; the spec PR is *where you check it*, on a single artifact, with the whole team looking. The spec is not just a machine input; it is the medium in which contract design happens collaboratively. That is a benefit code-first cannot offer at all, because in code-first there is nothing to review until the code is written, by which point the design has already been baked into handlers.

### A problem-solving narrative: adding a field without breaking anyone

Let me walk a realistic change through the loop, because the abstract steps hide the judgment. Suppose product wants payments to carry an optional `statement_descriptor` — the text that appears on a customer's bank statement. How does this flow through spec-first?

You start at **design**: add an optional `statement_descriptor` string to `PaymentCreate` and to `Payment`. Because it is *optional on the request* and *additive on the response*, this is a non-breaking change by the compatibility rules — old clients that do not send it still work, and old clients that ignore it on the way back still work (the tolerant-reader principle). You open the PR with just that schema edit.

At **review**, Spectral checks that the new field has a `description` and the team confirms the semantics — should it have a `maxLength`? (Yes; bank statements truncate, so cap it at 22 characters with `maxLength: 22` and document why.) At the **CI gate**, oasdiff inspects the change and reports it as non-breaking, so the PR is allowed to merge without a version bump. The **mock** updates; the frontend can now send the new field. The backend implements it. The SDKs **regenerate** with the new optional parameter. No caller breaks, and the entire decision — *is this safe to ship without a new version?* — was answered mechanically by oasdiff reading the spec, not by an engineer's memory of the compatibility rules.

Now stress-test it. What if product instead wanted to *rename* `payment_method_id` to `source_id`? oasdiff flags that as breaking (a removed required field plus a renamed one). The gate fails. You are now forced to choose: ship it as a new API version, or — the better move from the [schema evolution post](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) — add `source_id` as an optional alias, keep `payment_method_id` working, deprecate it with a `Deprecation` header, and remove it only after a sunset window. The spec made the breaking-ness *visible at PR time*, which is the whole point: drift and breakage stop being things you discover in production and become things CI tells you about before merge.

## Pitfalls and how to avoid them

Spec-first is not free of failure modes. Here are the ones that bite teams, with the antidote for each.

**Drift, if you go code-first without enforcement.** This is the original sin. If you generate the spec from annotations and never diff the generated spec against a baseline, the spec will fall behind the code, silently, exactly when a refactor is busiest. The fix is mechanical: even in a code-first shop, commit a snapshot of the generated spec and fail CI if a build produces a different one without an accompanying, reviewed update. That turns code-first's invisible drift into a visible test failure — it borrows spec-first's gate without adopting its full workflow.

**Over-using `$ref` into an unreadable web.** `$ref` reuse is a virtue until it becomes a maze. If `Payment` refs `Money` refs `CurrencyAmount` refs `DecimalString`, and each lives in a different file, a reader cannot hold the shape in their head, and tooling (especially older codegen) may stumble on deeply nested or circular refs. The discipline: reference *meaningful, named domain types* (a `Money`, a `Problem`, an `Address`), not micro-types that exist only to save three lines. A `$ref` should name a concept a reviewer recognizes, not a syntactic fragment.

**Examples that quietly go stale.** Examples are documentation *and* mock data *and* the first thing a developer copies — which makes a wrong example actively harmful. The antidote is to treat examples as testable artifacts: validate each `example` against its own schema in CI (Spectral can do this), and where possible, generate examples from real (scrubbed) traffic so they cannot describe a shape the server never returns. An example that the schema rejects is a CI failure, not a code-review nicety.

**Letting the spec lag the org's style.** A spec that passes JSON Schema validation can still be a mess — inconsistent naming, missing descriptions, ad-hoc error shapes. That is what Spectral and a governance ruleset are for, and the time to run them is in the design-review PR, not after a hundred endpoints have set a hundred precedents. Lint from the first endpoint.

**Treating the spec as write-only.** The worst outcome is a beautiful spec that nobody reads because the docs, the mock, and the SDKs were all hand-built separately. If you write the spec but do not *wire it to* the docs renderer, the mock server, and the generators, you have paid the cost of spec-first and gotten none of the benefit. The spec must be load-bearing — every downstream artifact must read it — or it is just more documentation to keep in sync.

**Forgetting that examples and the schema can disagree.** A subtle, common bug: the `example` you write for an operation does not actually satisfy the `schema` for that operation. You write an example with `amount: 4999` (a bare integer) while the schema says `amount` is a `Money` object — and now your mock serves a shape the schema forbids, your docs show a wrong example, and a developer copies it into a request that gets rejected. Because the example and the schema are *both* in the spec but nothing checks them against each other by default, they drift silently. The antidote is one CI rule: validate every `example` against its own schema (Spectral's `oas3-valid-media-example` rule does exactly this), so a mismatched example fails the build like any other contract error. Treat your examples with the same rigor as your schemas; they are part of the contract a developer actually experiences first.

One more practitioner note on the whole approach: the failure I have seen sink spec-first adoptions is not technical — it is organizational. A team writes the spec, wires the mock and the generators, and then, three sprints in, an engineer under deadline pressure edits the *handler* to add a field and skips the spec, "just this once." If there is no CI gate, that one shortcut reopens the drift door, and within a quarter the spec is back to being aspirational. Spec-first is a workflow, and a workflow holds only if the enforcement is mechanical and impossible to skip. The lesson from every successful adoption is the same: the gate is not optional. Make the spec the thing CI checks on every PR — validate it, lint it, diff it, run the contract tests — and the workflow defends itself. Leave the gate off and rely on discipline, and you have rebuilt code-first's failure mode with extra steps.

![a before-and-after contrast of code-first drift where a stale spec lies to clients against spec-first where a CI gate blocks breaking changes and clients integrate against the truth](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-4.png)

## JSON Schema as the shared core, and AsyncAPI for events

One more piece, and it is the structural reason OpenAPI 3.1 matters so much. Underneath OpenAPI's payload descriptions is **JSON Schema** — the standalone standard for describing the shape of JSON data. In 3.1, OpenAPI's schemas *are* JSON Schema, full stop, with all of `oneOf`, `anyOf`, `allOf`, `const`, `$schema`, and proper type unions (`type: [string, "null"]` instead of the old `nullable`). This is not a footnote; it is what lets your type definitions live in one place and be used everywhere.

Before the events, it is worth dwelling on *why the 3.1 alignment is a big deal* and not a versioning footnote, because the difference shows up in real schemas. In OpenAPI 3.0, you could not say "this field is a string or null" the natural JSON Schema way; you wrote `type: string` plus a non-standard `nullable: true`, and every tool had to special-case that extension. In 3.1 you write the standard JSON Schema union `type: [string, "null"]`, and any plain JSON Schema validator — not just OpenAPI-aware tooling — understands it. The same goes for `oneOf`/`anyOf`/`allOf` at the top of a schema, for `$schema`, for `const`, for `examples` as an array, and for `$ref`s that point *out* of the document at an external JSON Schema. The practical consequence: a schema you author for your API can be lifted, unchanged, into a standalone validator, a code generator, a database constraint, or another spec. It is no longer "OpenAPI's dialect of JSON Schema" — it is JSON Schema.

Consider the `Money` type from earlier. Because it is plain JSON Schema, the exact same definition can be: referenced by `POST /payments` in OpenAPI; referenced by the `payment.succeeded` message in AsyncAPI; compiled by a code generator into a `Money` class in five languages; and loaded by a standalone validator in a background job that double-checks ledger entries. One definition, every context. The alternative — a `Money` shape redefined in each of those places — is precisely how the `amount` versus `amount_minor` mismatch from the opening story happens, except now it happens across systems instead of within one.

Now the events. Your HTTP API is not your only contract. You also publish *events* — a `payment.succeeded` webhook, a Kafka topic the fulfillment service consumes. Those events have payloads, and those payloads have shapes, and you do not want to describe the same `Payment` object once in OpenAPI and again, by hand and inconsistently, for your events. **AsyncAPI** is the sister specification for event-driven and asynchronous APIs — channels instead of paths, messages instead of request bodies — and it describes its message payloads with *the same JSON Schema*. So a `Payment` schema can be authored once and referenced by both your OpenAPI document (for `POST /payments`) and your AsyncAPI document (for the `payment.succeeded` event). Here is the shape of an AsyncAPI channel that reuses the very same `Payment` schema file:

```yaml
asyncapi: 3.0.0
info:
  title: Acme Payments Events
  version: "2024-10-01"
channels:
  paymentSucceeded:
    address: payment.succeeded
    messages:
      paymentSucceeded:
        payload:
          $ref: "./schemas/Payment.json"   # the SAME schema OpenAPI uses
operations:
  onPaymentSucceeded:
    action: send
    channel:
      $ref: "#/channels/paymentSucceeded"
```

The `$ref: "./schemas/Payment.json"` points at the identical file the OpenAPI document references. Change the payment shape once and both the synchronous endpoint and the asynchronous event move together — there is no second copy to drift. The event side is the subject of the sibling post on [event-driven and async APIs with webhooks, pub/sub, and AsyncAPI](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi); the broker mechanics — delivery guarantees, ordering, dead-letter queues — live in the [message-queue series](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once). What unifies them is JSON Schema as the shared type core.

![a matrix showing OpenAPI, AsyncAPI, and JSON Schema and how the two API specs share JSON Schema as their common payload language for reuse across HTTP and events](/imgs/blogs/openapi-and-the-spec-first-workflow-design-mock-generate-8.png)

The principle to take away: **do not let your synchronous and asynchronous contracts describe the same domain object twice.** Author the domain types in JSON Schema, reference them from both OpenAPI and AsyncAPI, and a change to the `Payment` shape propagates to your endpoint, your webhook, and your SDKs from a single edit. That is the same single-source-of-truth logic that motivates spec-first in the first place, extended across the synchronous/asynchronous boundary.

| Spec | Describes | Payload language | Shares types with |
| --- | --- | --- | --- |
| OpenAPI 3.1 | HTTP APIs (paths, operations) | JSON Schema (fully aligned) | AsyncAPI, via shared `$ref` |
| AsyncAPI | Event APIs (channels, messages) | JSON Schema | OpenAPI, via shared `$ref` |
| JSON Schema | Any JSON document's shape | Is the language | Both, as the common core |

## Case studies

A few real-world anchors, kept to what is publicly verifiable.

**Stripe** publishes an OpenAPI specification for its API on GitHub (the `stripe/openapi` repository), generated from its internal API definitions, and uses it to drive its SDKs and documentation. Stripe is a useful reference because it is a payments API — our running example's real-world cousin — and because it is dated/versioned, which shows how a spec evolves alongside a versioning policy. The lesson is that even a company whose spec is *generated* invests heavily in keeping it accurate and load-bearing, because their entire SDK and docs surface depends on it.

**GitHub** publishes a comprehensive OpenAPI description of its REST API (the `github/rest-api-description` repository), which it uses to power its documentation and the Octokit SDKs. GitHub maintains both a "bundled" (refs resolved) and "dereferenced" form, a practical detail for anyone whose tooling struggles with `$ref` — sometimes you ship a flattened spec for compatibility. It is a good model of a large, long-lived public API treating its OpenAPI document as a first-class product artifact.

**openapi-generator** is the open-source generator (a community fork of Swagger Codegen) that emits clients and server stubs in dozens of languages from an OpenAPI document. It is the workhorse behind a lot of internal SDK pipelines — generate, then add a thin hand-written ergonomics layer on top. The honest caveat practitioners report: generated code quality varies by target language and you often customize the templates, so treat it as a strong starting point, not a finished SDK.

**Prism** (Stoplight) is the standard open-source mock-and-validation proxy for OpenAPI. In mock mode it serves spec-conformant responses; in proxy mode it sits in front of a real server and validates live traffic against the spec. It is the concrete tool behind the "build the frontend before the backend" claim, and it is the easiest single thing to adopt to *prove* the value of spec-first to a skeptical team — stand up a mock in one command and let the frontend start.

**Redocly / Redoc and Stoplight** are the two common documentation-and-platform stacks: Redoc renders clean reference docs from a spec (Redocly adds hosting, linting, and bundling on top), and Stoplight provides a visual editor plus docs plus mocks. Both exist to make the spec a *readable* product, not just a machine artifact — the developer-experience payoff we treat as a first-class goal in [designing for the caller](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal).

## When to reach for spec-first (and when not to)

Every workflow choice is a trade-off; here is the decisive version.

**Reach for spec-first when:**

- You are building a **public, partner, or cross-team API** — anyone you cannot ping on Slack to coordinate a change. The contract must be agreed and stable, and the spec is how you agree it.
- The **frontend and backend are different people** (or teams). Spec-first's parallelism is worth the upfront design every time the consumer and producer are separate.
- You will **generate SDKs or docs** as a product surface. If callers will use a generated client, the spec is load-bearing and must lead, not lag.
- You need **governance** — consistency across many endpoints and teams. You can only lint and review a spec that exists as a single artifact before code.

**Do not reach for spec-first when:**

- It is a **private endpoint with one co-located consumer** who can absorb a change instantly. Code-first with a drift gate (snapshot the generated spec, fail CI on unreviewed change) gives you most of the safety for less ceremony.
- The API is a **genuine throwaway** — a one-week internal tool that will be deleted. Designing a contract for a thing with no future is gold-plating.
- Your **framework's code-first story is excellent and your team is disciplined about the drift gate.** A FastAPI shop that snapshots its generated spec and diffs it in CI has, in practice, reconstructed spec-first's enforcement from the other direction; the workflows converge. The failure is code-first *without* the gate, not code-first as such.

The deeper rule: **the spec must be the single source of truth, and it must be load-bearing.** Whether you write it by hand (spec-first) or generate it from annotations (code-first), it is only worth having if the docs, the mock, the SDKs, and the CI gate all *read it* and if drift is a *failing test*. A spec nobody enforces and nothing consumes is a liability — a second copy of the truth that will diverge from the first. Pick the direction that lets your team keep the spec true, and wire every downstream artifact to it.

## Key takeaways

- **OpenAPI 3.1 is the machine-readable form of your wire contract** — a single, language-agnostic YAML/JSON document, fully aligned with JSON Schema, that every downstream tool reads instead of guessing.
- **Spec-first inverts the dependency arrow.** Code-first makes the spec a byproduct that drifts; spec-first makes the code a consequence the spec can hold to. Whatever is downstream drifts — keep the spec upstream.
- **One spec fans out into the whole toolchain**: interactive docs (Swagger UI / Redoc / Stoplight), a mock server (Prism), client and server codegen (openapi-generator), live request/response validation, breaking-change diffing (oasdiff), and linting (Spectral). You write one file and derive the rest.
- **The mock unlocks parallelism.** With a mock server from the spec, the frontend builds against a conformant fake on day one while the backend implements in parallel — collapsing a serial dependency.
- **Organize operations as references into `components`.** Define each domain type once (`Money`, `Payment`, `Problem`), reference it everywhere with `$ref`, and a change is a single reviewable edit.
- **Make drift a failing test.** Whether spec-first or code-first, gate every change with oasdiff for breaking-change detection and Spectral for style — the contract is enforced last, in CI, not remembered.
- **JSON Schema is the shared core.** Author domain types once in JSON Schema and reference them from both OpenAPI (HTTP) and AsyncAPI (events) so synchronous and asynchronous contracts never describe the same object twice.
- **Use spec-first for public, partner, and cross-team APIs; code-first-with-a-gate is fine for a private, co-located, single-consumer endpoint.** The non-negotiable is that the spec is load-bearing and true.

## Further reading

- [OpenAPI Specification 3.1.0](https://spec.openapis.org/oas/v3.1.0) — the authoritative format reference; read the `paths`, `components`, and schema sections.
- [JSON Schema](https://json-schema.org/) — the standard OpenAPI 3.1 schemas are aligned with; the shared type core for OpenAPI and AsyncAPI.
- [AsyncAPI Specification](https://www.asyncapi.com/docs/reference/specification/latest) — the sister spec for event-driven APIs, using JSON Schema for message payloads.
- [Spectral](https://docs.stoplight.io/docs/spectral) — the open-source linter for OpenAPI and AsyncAPI; the basis for automated governance.
- [openapi-generator](https://openapi-generator.tech/) — generate client SDKs and server stubs in dozens of languages from a spec.
- [Prism (Stoplight)](https://github.com/stoplightio/prism) — the open-source mock server and validation proxy used in this post's worked examples.
- [oasdiff](https://www.oasdiff.com/) — diff two OpenAPI documents and classify changes as breaking or non-breaking for a CI gate.
- Within this series: the intro hub [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2), and the siblings on [SDKs and reference docs](/blog/software-development/api-design/sdks-code-generation-and-reference-docs-developers-love), [API governance and style guides](/blog/software-development/api-design/api-governance-and-style-guides-consistency-across-an-org), and [contract testing](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs).

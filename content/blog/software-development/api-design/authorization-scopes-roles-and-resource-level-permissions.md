---
title: "Authorization: Scopes, Roles, and Resource-Level Permissions"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Authentication proves who is calling; authorization decides what they may do — learn the four layers (scopes, roles, object-level, field-level), why Broken Object-Level Authorization is the number-one API vulnerability, and how to enforce ownership on every single object access."
tags:
  [
    "api-design",
    "api",
    "authorization",
    "rbac",
    "abac",
    "bola",
    "owasp",
    "security",
    "oauth",
    "multi-tenancy",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-1.png"
---

A penetration tester once handed a payments team a single screenshot. It was a `200 OK` response containing a refund record — amount, card last-four, the merchant's settlement account. Nothing was technically broken. The token was valid. The endpoint worked exactly as written. The only problem was that the token belonged to a small coffee-shop merchant, and the refund record belonged to a national retailer in a completely different account. The coffee shop had simply changed one number in the URL — `GET /v1/refunds/re_88231` became `GET /v1/refunds/re_88232` — and the API cheerfully returned a stranger's money movements.

No password was cracked. No token was forged. No TLS was downgraded. The authentication layer — the part that proves *who* is calling — did its job perfectly. What failed was the **authorization** layer: the part that decides *what* a proven caller may do, and crucially, *which specific objects* they may touch. The service had checked that the request carried a real token with the `refunds:read` permission, and then it trusted the id in the path. It never asked the only question that mattered: *does this caller own refund `re_88232`?*

This is not an exotic failure. It is the single most common, most exploited, and most expensive class of API vulnerability in production today. The OWASP API Security Top 10 has ranked **Broken Object-Level Authorization** as the number-one risk for years running. The breaches you read about — the ones that leak millions of records through a public API — are overwhelmingly authorization failures, not authentication failures. Attackers rarely need to break in. They log in legitimately and then walk sideways into data that was never theirs, because the API forgot to check ownership on every object it returned.

This post is about getting that layer right. We will recap the line between authentication (authN) and authorization (authZ), then build authZ up in layers: coarse **scopes** on a token, **roles** mapped to permission sets (RBAC), **attribute- and relationship-based** rules (ABAC and ReBAC), and the layer almost everyone underbuilds — **object-level permission**, the per-record ownership check that stops BOLA. We will work the exact attack from the screenshot and the exact check that closes it, decide where each layer belongs (gateway versus service), weigh `403` against `404` for hiding existence, and compare hand-rolled checks against a policy engine like OPA or Cedar. Throughout, we stay on the series' running example — a Payments and Orders API for a fictional commerce platform — because authorization is where a clean contract meets the messy reality of *whose data is this, anyway?* The figure below sketches the destination: two independent gates, a scope gate and an object gate, that a request must clear in order, and where it dies if it fails either.

![A branching flow showing a bearer token passing a scope gate and then an object-ownership gate, with denial branches to a 403 for a missing scope and a 404 for the wrong tenant before reaching an allowed 200 response](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-1.png)

Tie this back to the series spine: an API is a contract and a product, and authorization is the clause of that contract that says *"you may read your orders, and only yours."* A caller gets to assume that the surface is safe — that pinning their integration to your endpoint will not, on some future Tuesday, leak their data to a competitor because you added a sibling endpoint and forgot the ownership check. Authorization is how you keep that promise across years and versions you cannot recall.

## AuthN versus authZ: you proved who, now decide what

The two words look alike and the abbreviations are one letter apart, which is exactly why teams conflate them and ship holes. They are different questions, answered at different layers, with different inputs and different failure modes.

- **Authentication (authN)** answers *"who are you, and can you prove it?"* The input is a credential — an API key, a session cookie, a JWT (JSON Web Token, a signed, self-describing token), a client certificate in mTLS. The output is an authenticated **principal**: a verified identity, usually a user id and/or a client/app id, plus some claims. The sibling post [authentication: api keys, sessions, jwt, and mtls](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) covers this layer in full; here we assume it has already succeeded.
- **Authorization (authZ)** answers *"given who you are, what may you do — and to which objects?"* The input is the authenticated principal plus the request: the action (read, refund, delete), the resource type (order, payment), and the specific target object (order `ord_99`). The output is a decision: **allow** or **deny**.

The order is fixed and it is one-directional. You cannot authorize before you authenticate — you have no principal to reason about. And authentication never implies authorization: a perfectly valid token is the *beginning* of the conversation, not the end. The catastrophic breaches almost always live in the gap, where a team built a solid authN layer, felt secure, and treated "has a valid token" as if it meant "may do this thing to this object."

Here is the mental model worth burning in. Authentication is the bouncer checking your ID at the door of the building. Authorization is every locked door *inside* the building. Getting past the bouncer does not unlock the CFO's office. An API that checks the token and then hands over any record whose id you can guess is a building with a strict bouncer and no interior locks at all.

> **The principle.** Authentication establishes the principal; authorization is a *function* over (principal, action, resource, object) → {allow, deny}. The function must be evaluated **on every request and for every object the request touches** — never inferred from a prior successful authentication, never cached as "this user is fine," never skipped because the token was valid. The default must be **deny**: if no rule explicitly grants the action on the object, the answer is no.

That last clause — **deny by default** — is the safety property that makes the whole system fail closed. We will return to it repeatedly, because the BOLA family of bugs is, at root, a place where the system fell open: it returned data because nothing said "stop," rather than withholding data because nothing said "go."

It is worth being precise about *why* the function must run on every request rather than once per session. The temptation, especially in code that grew from a session-cookie web app, is to compute "this user is an admin" at login, stash it, and trust it for the life of the session. Two things break that. First, **state changes underneath you**: a user demoted from `admin` to `viewer` mid-session, an order transferred to a different team, a token revoked, an org membership removed — all of these invalidate a decision made earlier, and a session-cached "yes" keeps saying yes to a caller who should now be denied. Second, and more fundamentally, **the object is not known at session time**. The decision "may you read `ord_99`" cannot possibly have been made at login, because `ord_99` was not part of the login. Authorization is *per request, per object*, full stop. You may cache cheap, slow-changing inputs — a user's role for a few seconds, a relationship lookup behind a short TTL — but the *decision* is recomputed every time, against the *current* state and the *specific* object in the request. This is the difference between caching the ingredients and caching the meal: the ingredients are reusable, the verdict is not.

A second subtlety hides in the word "object." A single request can touch *many* objects, and every one of them needs its own check. A request to `POST /v1/refunds` with body `{"payment_id": "pay_1042", "order_id": "ord_99"}` references two objects — a payment and an order — plus it creates a third. Authorizing the *action* (`refund:create`) is necessary but nowhere near sufficient: you must also verify the caller owns `pay_1042` *and* `ord_99`, and that the two belong together. A `GET /v1/orders/ord_99?expand=customer,payments` that inlines related resources must authorize each expanded resource, not just the top-level order — a classic place where BOLA sneaks back in through a relationship the top-level check never covered. The rule generalizes: **for every id that crosses the wire from the client, run an ownership check against the authenticated principal.** No exceptions for "internal" ids, "opaque" ids, or ids you "generated yourself" — if the client can send it back, the client can send back a different one.

## The four layers an API actually needs

People say "we use RBAC" as if authorization were one decision. In a real API it is at least four, stacked, each answering a strictly narrower question. A request must satisfy all of them; failing any one is a deny.

![A vertical stack of four authorization layers labeled scope for what the app may call, role for what the user may do, object for whether the caller may touch this specific resource, and field for which columns are visible, over a deny-by-default base](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-2.png)

1. **Scope — what the *app/token* is allowed.** Coarse, set at the token level (typically via OAuth consent). A token carries `payments:read` but not `payments:write`. This bounds the *blast radius of the token itself*, independent of who the user is. A leaked read-only token cannot issue refunds.
2. **Role — what *this user* may do.** RBAC maps a role (`admin`, `member`, `viewer`) to a set of permissions (`refund:create`, `order:read`). A `viewer` may read orders but never refund one, regardless of which token the app holds.
3. **Object — may the caller touch *this specific* resource.** The ownership/tenancy check. A user may have `order:read`, but only for orders in *their* organization. This is the layer that stops BOLA, and the layer most often missing.
4. **Field — which *fields/columns* of the object are visible or writable.** A support agent may read an order but not the customer's full card number; a partner may read `total` but not the internal `cost_basis`. Field-level (and action-level) authorization is the finest grain.

Each layer answers a question the layer above cannot. Scope cannot tell tenant A's order from tenant B's — it only knows "this app may read payments." Role cannot either — `viewer` is a role on *all* orders the user can see, not a fence around a specific record. Only the object layer knows ownership. And no amount of object-level correctness tells you whether to redact the card number — that is the field layer's job. Conflating these is how holes appear: a team builds scopes, calls it authorization, and ships an API where any `payments:read` token reads every tenant's payments.

The four layers also map cleanly to *where* they are best enforced, which we will develop in its own section: scopes and roles are coarse and contextless enough to push toward the edge (a gateway), while object and field decisions need domain knowledge — who owns what — and must live in the service. You **cannot** do a BOLA check at the gateway, because the gateway does not know that `ord_99` belongs to org `org_A`. Hold that thought.

## Scopes: bounding what a token may do

A **scope** is a permission attached to a *token*, not to a user. It is the OAuth mechanism for an app to request, and a user to consent to, a bounded slice of access. When a third-party analytics dashboard integrates with your Payments API, it should ask for `payments:read` and `orders:read` — and emphatically not `payments:write` or `refunds:create`. The user consents to exactly that, the authorization server mints a token whose `scope` claim lists those strings, and your API enforces them. The companion post [oauth2 and openid connect for api designers](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers) covers how those tokens are issued; here we focus on enforcing them.

Scopes are deliberately **coarse**. Good scope design is a small, stable, resource-oriented vocabulary — typically `resource:action`:

```http
GET /v1/payments/pay_1042 HTTP/1.1
Host: api.commerce.example
Authorization: Bearer <access_token>
Accept: application/json
```

A token whose decoded claims include:

```json
{
  "sub": "usr_7c2",
  "client_id": "cli_dashboard",
  "scope": "payments:read orders:read",
  "org": "org_A",
  "exp": 1771200000
}
```

passes a `payments:read` check on that endpoint. The enforcement is a string-set membership test:

```python
REQUIRED_SCOPES = {
    ("GET", "/v1/payments"): {"payments:read"},
    ("POST", "/v1/refunds"): {"refunds:write"},
    ("POST", "/v1/payments"): {"payments:write"},
}

def check_scope(method, route, token):
    needed = REQUIRED_SCOPES.get((method, route), set())
    granted = set(token["scope"].split())
    if not needed.issubset(granted):
        raise Forbidden(missing=needed - granted)  # -> 403
```

Note what scopes do *not* know. The `scope` claim says `payments:read`. It does not say *which* payments. It cannot — the token was minted before this request existed, with no knowledge of `pay_1042` or whose payment it is. This is the defining limitation, and the source of the most dangerous misunderstanding in API authorization, which we will hammer on in its own section: **a scope is permission to call an *endpoint class*, never permission to read a *specific object*.**

A few scope-design rules earned in production:

- **Keep the vocabulary small and resource-shaped.** `payments:read`, `payments:write`, `refunds:write`. Resist `payments:read_settlement_account_for_eu_merchants`. Fine-grained access belongs in object/field layers, not in an ever-growing scope list that no human can audit.
- **Separate read from write.** The single highest-value split. A read-only integration token that leaks cannot mutate state. Most third-party integrations need only `:read`.
- **Scopes constrain *down*, never up.** A token's scopes can only *reduce* what the underlying user could do. If a `viewer` user authorizes an app for `refunds:write`, the app still cannot refund — the role check downstream denies it. Scope ∩ role, not scope ∪ role.
- **Document the scope a caller will encounter at consent.** A scope the user agrees to should be human-readable on the consent screen ("Read your payments") and map one-to-one to what the API enforces. A scope no human can interpret is a scope users rubber-stamp, which defeats the entire point of asking. The scope vocabulary is part of your public contract, exactly like a resource name or a status code, and it cannot be renamed later without breaking every integration that requested it.

There is a related question that trips teams: **what scopes does a first-party client need?** Your own mobile app and web SPA are also OAuth clients, but they typically act for a logged-in user with the user's full authority, so they often request broad scopes (or a special first-party scope). That is fine — the scope layer is about bounding *the token*, and a first-party token legitimately needs the user's full surface. But broad scopes make the *object* and *role* layers do *all* the work of separating one user's data from another's. So the more powerful the token, the more critical the downstream layers. A common, dangerous error is to reason "this is our own app, it is trusted, we can skip the object check" — the app is trusted, but the app is also fully under attacker control once it runs on a customer's phone. The mobile binary is not a trust boundary; the server is.

#### Worked example: a scope check and an object check are different gates

Take a request from the analytics dashboard, acting for a user in org A, holding a token with `payments:read`. It calls `GET /v1/payments/pay_9001`, where `pay_9001` belongs to org B.

**Gate 1 — scope.** Required: `payments:read`. Granted: `payments:read orders:read`. `{"payments:read"} ⊆ {"payments:read","orders:read"}` → **pass**. The token is allowed to call the read-payments endpoint class.

**Gate 2 — object.** Does the principal (user in org A) own `pay_9001` (org B)? Look up the payment, compare `payment.org == "org_B"` against `principal.org == "org_A"` → **fail**. Deny.

The request had every right to *attempt* the read and no right to *this* result. If your code only had Gate 1, it returns tenant B's payment with a `200`. Those are two independent gates, and the second one is the one people forget. The whole rest of this post is, in a sense, an argument for Gate 2.

## Roles and RBAC: mapping users to permission sets

Where scopes bound the *token*, **Role-Based Access Control (RBAC)** bounds the *user*. The idea is old and good: do not grant permissions to individuals (which becomes an unauditable sprawl); grant permissions to a small set of **roles**, and assign users to roles. A role is just a named bundle of permissions.

In the commerce platform, an organization's members might have:

| Role | Can read orders | Can create order | Can refund | Can manage members |
| --- | --- | --- | --- | --- |
| `owner` | yes | yes | yes | yes |
| `admin` | yes | yes | yes | no |
| `member` | yes | yes | no | no |
| `viewer` | yes | no | no | no |

A `viewer` is the running example: they can *read* an order but must never *refund* one. The check is again a membership test, this time against the permission set the role grants:

```python
ROLE_PERMS = {
    "owner":  {"order:read", "order:create", "refund:create", "member:manage"},
    "admin":  {"order:read", "order:create", "refund:create"},
    "member": {"order:read", "order:create"},
    "viewer": {"order:read"},
}

def require(principal, permission):
    perms = ROLE_PERMS.get(principal.role, set())
    if permission not in perms:
        raise Forbidden(needed=permission)  # -> 403
```

A `viewer` calling `POST /v1/refunds` hits `require(principal, "refund:create")`, which is not in `{"order:read"}`, and is denied with a `403`. Clean.

RBAC's strength is its weakness: it is **coarse and contextless**. The model says "a `member` may create orders." It cannot natively express "a `member` may refund an order *they* created, within 24 hours, if it is under \$100, and only in the EU region." Every "*they*," every "*within*," every "*under*," every "*region*" is an attribute or relationship that pure RBAC cannot see. The classic anti-pattern is the **role explosion**: teams try to encode context by minting `member_eu`, `member_eu_small_refund`, `member_eu_small_refund_24h`, and within a year there are 400 roles, nobody can say what any of them grant, and the audit is hopeless.

When you feel that pressure — when a role's meaning starts to depend on *which object* or *what context* — you have outgrown pure RBAC and need attributes or relationships. But do not jump early: for the majority of APIs, a handful of stable roles plus a solid object-ownership check covers the real requirements. We will give a decision tree for exactly this.

A few RBAC refinements worth knowing before you reach for something heavier, because they buy a lot of the flexibility people think requires ABAC:

- **Permission-based checks, not role-based checks, in code.** Even though you *assign* roles, your handlers should ask `require(principal, "refund:create")`, not `if principal.role == "admin"`. The indirection means you can later split `admin` into two roles, or add a custom role, without touching every handler. Code that hardcodes role *names* is code that fights you the day product wants a new role. Ask about permissions; let roles be the bundles.
- **Role hierarchies.** `owner` ⊇ `admin` ⊇ `member` ⊇ `viewer` lets a senior role inherit a junior role's permissions, so you define each permission once at the lowest role that should have it. This collapses the permission table and prevents the bug where you grant `order:read` to `admin` but forget to also grant it to `owner`.
- **Scoped roles (role *per* org).** In multi-tenant systems a user is rarely just "an admin" — they are "an admin *of org A* and a viewer *of org B*." The role lives on the *(user, org)* membership, not on the user globally. This is the bridge to the object layer: the role check already has to know *which org's* role to apply, which means it already has the tenant in hand. Lean into that — a role that is inherently scoped to an org is half of the tenancy check.

These three together — permission checks, hierarchy, scoped roles — stretch RBAC much further than the naive "one global role string per user" model, and they are what most teams actually mean by "we use RBAC." The line where they stop being enough is sharp: the moment the answer depends on a *relationship to the specific object* ("the team that owns this order") or a *computed attribute of the request* ("under \$100, before the cutoff"), no amount of role engineering will express it cleanly, and you cross into ABAC/ReBAC territory.

#### Worked example: the viewer who can read but not refund

Concretely, the wire:

```http
POST /v1/refunds HTTP/1.1
Host: api.commerce.example
Authorization: Bearer <access_token>
Content-Type: application/json

{ "payment_id": "pay_1042", "amount": 4999, "currency": "usd" }
```

The principal resolves to `{ "sub": "usr_v1", "org": "org_A", "role": "viewer" }`. The handler runs `require(principal, "refund:create")`. `viewer`'s permission set is `{"order:read"}`; `"refund:create"` is absent. The response:

```http
HTTP/1.1 403 Forbidden
Content-Type: application/problem+json

{
  "type": "https://api.commerce.example/problems/insufficient-role",
  "title": "Insufficient permissions",
  "status": 403,
  "detail": "Role 'viewer' cannot create refunds. Required: refund:create.",
  "instance": "/v1/refunds"
}
```

Two things make this a *good* deny. First, it uses `403` and the RFC 9457 `problem+json` envelope (see the sibling [error design: a machine-readable, human-friendly contract](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract) and [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx)) — the client learns *which permission* was missing, which is safe to reveal here because the *role* is not secret. Second, it denies on **action**, before the handler ever touches `pay_1042`. A refund the user may not perform should be stopped at the role gate, not deep in the refund-processing code where a bug might let it slip through. Defense in depth means checking early *and* checking again at the object.

## ABAC and ReBAC: when attributes or relationships drive the decision

Beyond roles lie two richer models. Both exist to express the context RBAC cannot.

**Attribute-Based Access Control (ABAC)** makes the decision a function of *attributes* — of the principal (region, department, clearance), of the resource (owner, amount, classification, tenant), of the action, and of the environment (time of day, request IP, MFA-recency). A policy reads like a sentence:

> Allow `refund:create` if `principal.role in {admin, owner}` **and** `resource.org == principal.org` **and** `resource.amount <= principal.refund_limit` **and** `now - resource.created_at < 24h`.

ABAC is enormously flexible — almost anything you can compute from attributes becomes a rule. The cost is that policies can sprawl and become hard to reason about and test; a poorly governed ABAC system is as opaque as a 400-role RBAC system, just in a different shape.

**Relationship-Based Access Control (ReBAC)** makes the decision a function of *relationships between objects*, expressed as a graph. Instead of "does this user have attribute X," it asks "is there a path in the relationship graph from this user to this object via an allowed edge?" The canonical sentence:

> Allow `order:read` if the user **is a member of the team** that **owns** the order.

That is two hops: `user --member--> team --owns--> order`. ReBAC shines for the things RBAC and ABAC handle awkwardly — nested groups, folders that inherit permissions, per-object sharing ("share *this one* order with that external auditor"), org → team → project → resource hierarchies. This is the model behind Google's internal authorization system, **Zanzibar**, and its open-source descendants, which we will name in the case studies.

Here is the honest comparison. None is universally "best"; each buys flexibility with cost.

![A three-by-three comparison matrix of RBAC, ABAC, and ReBAC across the rows decision input, flexibility, and operating cost, showing RBAC as cheap but coarse and ReBAC as flexible but requiring a relation store](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-3.png)

| Dimension | RBAC | ABAC | ReBAC |
| --- | --- | --- | --- |
| Decision input | Role name only | Attributes of subject/resource/env | Relationships in an object graph |
| Best at | Stable, org-wide roles | Context: region, amount, time, MFA | Per-object sharing, nested groups, hierarchies |
| Flexibility | Low — fixed role sets | High — any computable attribute | High — any expressible relationship |
| Reasoning / audit | Easy ("what does `admin` grant?") | Harder (policies can interact) | Harder (graph reachability) |
| Operating cost | Cheap | Policy can explode; needs governance | Needs a relationship store and consistency model |
| Typical home | Small/medium APIs, internal tools | Regulated, multi-condition rules | Collaboration, multi-tenant SaaS with sharing |

The pragmatic truth: most teams run a **hybrid**. RBAC for the coarse "what may this role do" question, plus an object-level check (which is really a tiny, hardcoded slice of ABAC/ReBAC: "is `resource.org == principal.org`?"). You reach for a full ABAC or ReBAC engine when the object-level rules stop being one-liners — when sharing becomes per-object, groups nest, or conditions multiply. Do not start there.

It helps to see ABAC and ReBAC as answers to two different shapes of question, because the shape tells you which to use. **ABAC answers "does the *state* of these things permit it?"** — properties you can read off the principal, the resource, and the environment without traversing any graph. A refund cap, a region restriction, a business-hours window, a "requires recent MFA" rule: all of these are attribute comparisons. **ReBAC answers "is there a *path* between this principal and this object?"** — and the path itself is the permission. "The user is on the team that owns the order" is not a property of the user *or* the order in isolation; it is a fact about the *edges* between user, team, and order. When your authorization sentences keep saying "the X that owns / belongs to / is a member of / is shared with the Y," you are describing a graph, and a graph is what ReBAC stores and walks.

The two compose, and the best real systems use both: a ReBAC check to establish *that* the caller has *some* relationship to the object, plus an ABAC condition that *narrows* it (`is a member of the owning team` **and** `amount <= refund_limit`). Cedar and OpenFGA both support conditional relationships for exactly this. But composition is also where complexity compounds — a system that needs both relationship traversal *and* attribute conditions *and* role hierarchy is a system whose authorization is now a small program, and a small program needs the same care as any other: version control, tests, review, and a way to answer "why was this allowed?" after the fact. That governance burden is the real cost of the flexible models, far more than the runtime latency people worry about first.

#### Worked example: "is a member of the team that owns this order"

A ReBAC check for `GET /v1/orders/ord_99` by `usr_7c2`, expressed in the OpenFGA / Zanzibar relationship-tuple style. The stored tuples (facts) might be:

```json
[
  { "user": "user:usr_7c2", "relation": "member", "object": "team:fulfillment" },
  { "user": "team:fulfillment#member", "relation": "owner", "object": "order:ord_99" }
]
```

and the model declares that `order#viewer` is implied by `order#owner`, and `owner` can be a team's members. The check `is user:usr_7c2 a viewer of order:ord_99?` resolves by walking: `usr_7c2` is a `member` of `team:fulfillment`; `team:fulfillment#member` is `owner` of `order:ord_99`; `owner` implies `viewer` → **allow**. If `usr_7c2` were not on that team, no path exists → **deny**. The decision came entirely from relationships, never from a role string on the user.

## The crux: scopes are not object-level permissions

This is the single most important paragraph in the post, so it gets its own section and I will say it three ways.

A token with `payments:read` is permission to call the *read-payments endpoint*. It is **not** permission to read *any particular payment*. A scope (and equally, a role permission like `order:read`) authorizes an **action on a resource type**. It says nothing — can say nothing — about *which instances* of that type the caller owns. The instance is named in the request (`/payments/pay_9001`), which did not exist when the token was minted. So the moment your code reads an object identified in the request, you have a second, independent obligation: **verify the caller is authorized for that specific object.**

The three framings:

1. **Type versus instance.** Scope/role = "may do X to resource-type T." Object permission = "may do X to *this instance* of T." Passing the first never implies the second.
2. **Token-time versus request-time.** Scopes are decided at token-issuance time, with no knowledge of future object ids. Object ownership can only be decided at request time, after you look up the object and learn whose it is.
3. **Endpoint versus row.** The gateway authorizes the *endpoint*; only the service, after a database lookup, can authorize the *row*.

The matrix below lines up the three controls so the distinction is unmissable.

![A matrix comparing scope, role, and object permission across what each one controls, who sets it, and its granularity, showing object permission as the only per-record control](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-4.png)

| Control | What it controls | Who/where it is set | Granularity | Stops BOLA? |
| --- | --- | --- | --- | --- |
| Scope | What the *app/token* may call | OAuth consent → token claim | Coarse — per endpoint class | No |
| Role (RBAC) | What the *user* may do | Org admin assigns role | Medium — per action | No |
| Object permission | May touch *this specific* record | Ownership/relationship in data | Fine — per object | **Yes** |

Read the last column. Neither scope nor role stops a caller from reading another tenant's data, because neither knows about tenants at the instance level. Only the object-permission check does. When teams say "we have authorization" and mean "we check scopes and roles," they have built the first two columns and left the third — the one that actually stops the breach — empty.

## BOLA / IDOR: the number-one API vulnerability

**Broken Object-Level Authorization (BOLA)** — historically also called **Insecure Direct Object Reference (IDOR)** — is what happens when the third column is empty. The pattern: the API exposes an object by an id in the path or body (`/orders/{id}`, `/refunds/{id}`, `"payment_id": "..."`), the caller is authenticated and has the right scope/role for the *endpoint*, and the server returns the object **without checking that the caller owns it**. Change the id, read someone else's data. It tops the OWASP API Security Top 10 because it is everywhere, trivial to find (just increment an integer or fuzz an id), and devastating when the objects are payments, orders, refunds, or PII.

Here is the dangerous code, and it is dangerous precisely because it looks reasonable and *works in every demo*:

```python
@app.get("/v1/orders/<order_id>")
@require_scope("orders:read")          # gate 1: the token may read orders
def get_order(order_id):
    order = db.orders.find_one({"id": order_id})   # gate 2: MISSING
    if not order:
        return problem(404, "Order not found")
    return jsonify(order)               # returns ANY order, any tenant
```

Both reviewers nod: there is an auth decorator, there is a 404 for missing orders, the JSON looks clean. But `find_one({"id": order_id})` queries the *entire* table with no tenant filter. The `require_scope` decorator confirmed the token may read orders *in general*; nothing confirmed the caller may read *this* order. Increment the id and you walk the whole table.

![A before-and-after contrast where the before column trusts the path id and leaks tenant B's order with a 200, and the after column filters by the caller's tenant and returns a 404 for an order that is not theirs](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-5.png)

#### Worked example: reading another tenant's order, then the check that stops it

**The attack.** A user in org A holds a normal token, fully authenticated, with `orders:read` and a `member` role. Their own orders are `ord_500`–`ord_550`. They run:

```bash
for id in $(seq 600 650); do
  curl -s -H "Authorization: Bearer <access_token>" \
    https://api.commerce.example/v1/orders/ord_$id
done
```

With the vulnerable handler, `ord_600`–`ord_650` belong to org B, but the query never filters by org. Every request returns a `200` with another tenant's order — customer names, line items, totals, the lot. The attacker exfiltrates a competitor's order book by counting. There was no authentication failure anywhere: every token was valid, every scope was present. The system fell *open* because nothing said stop.

**The fix.** Scope the lookup to the caller's tenant — make ownership part of the query, not an afterthought:

```python
@app.get("/v1/orders/<order_id>")
@require_scope("orders:read")
def get_order(order_id):
    order = db.orders.find_one({
        "id": order_id,
        "org": g.principal.org,          # the object-level check, in the query
    })
    if order is None:
        return problem(404, "Order not found")   # not theirs == not found
    return jsonify(order)
```

Now `find_one({"id": "ord_600", "org": "org_A"})` returns nothing, because `ord_600` is in org B. The caller gets a `404`. The exact same code path handles "genuinely does not exist" and "exists but not yours," and from the attacker's side they are indistinguishable — which is exactly the point of returning `404` rather than `403` here, as we will justify shortly. The enumeration leaks nothing.

The discipline that prevents the whole class:

- **Check ownership on *every* object access.** Not just `GET` by id — also `PATCH`, `DELETE`, and any nested reference (a `payment_id` in a refund body, an `order_id` in a query filter). Every id that crosses the wire from the client is attacker-controlled and must be re-validated against ownership.
- **Never trust an id in the path or body.** The id tells you *what* the caller wants, not *whether* they may have it. Treat client-supplied ids as untrusted input, exactly like you treat client-supplied strings for injection.
- **Push ownership into the data-access layer.** The most robust fix is to make it *impossible* to query without the tenant filter — a repository that takes the principal and always injects `org = principal.org`, or row-level security in the database keyed off a session variable. If a developer cannot write an unscoped query, they cannot write a BOLA bug. (For how the database enforces this efficiently with indexes, see [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans).)
- **Test it.** Every object-returning endpoint deserves a "cross-tenant" test: authenticate as tenant A, request a tenant-B object, assert `404`. This is cheap and catches regressions when someone adds an endpoint and forgets the filter.

There is a sibling vulnerability worth naming: **Broken Object Property Level Authorization** (the merge of old "excessive data exposure" and "mass assignment"). That is the *field* layer failing — returning fields the caller should not see, or letting a `PATCH` write fields they should not set (e.g. flipping `"role": "admin"` on themselves, or `"status": "paid"` on an order). The fix is the same philosophy at field grain: allow-list the readable and writable fields per role, never bind a request body straight onto your model. The input-validation sibling [input validation, output encoding, and the owasp api top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10) goes deep on mass assignment.

**Why integer ids make BOLA worse.** The coffee-shop example incremented `re_88231` to `re_88232`. Sequential or guessable ids turn a missing-ownership bug into a *trivially scriptable* one: an attacker does not even need to discover valid ids, they just count. Switching to unguessable ids — UUIDs, or random strings like Stripe's `re_3PqL...` — does *not* fix BOLA (the bug is still that you do not check ownership), but it does raise the bar from "increment a number" to "guess a 128-bit value," which collapses the *casual* enumeration attack. Treat unguessable ids as defense in depth, never as the fix. The fix is always the ownership check; opaque ids just buy you time and reduce the blast radius of the day you forget it. A useful framing: **a guessable id with no ownership check is a public API to your whole database.**

**Where BOLA hides that people miss.** The naive `GET /orders/{id}` is the obvious case, but the bug recurs in places that feel different:

- **List endpoints that accept a filter id.** `GET /v1/refunds?order_id=ord_600` — if the list query does not also filter by tenant, the filter becomes a BOLA vector even though no id is in the *path*. Filter the list by tenant first, then apply the client filter.
- **Bulk and batch endpoints.** `POST /v1/orders:batchGet` with a list of ids must check ownership on *each* id, not just the first or a random sample. Bulk endpoints are a favorite attacker target precisely because the per-item check is easy to forget under the pressure of "make it fast."
- **Nested and expanded resources.** `GET /v1/orders/ord_99?expand=payments` — you checked `ord_99`, but did you check that the expanded payments are also the caller's? If a payment can be linked to an order across tenants by a data bug, the expand leaks it.
- **Webhook and export payloads.** An export job or a webhook fired *as* a tenant must serialize only that tenant's data. A background job that runs without a principal is a job with no tenant filter — the most dangerous kind, because it runs unattended.
- **GraphQL and other graph traversals.** A GraphQL query can walk from an object the caller owns to one they do not via a relationship edge. Each resolver must authorize the object it returns, not assume the parent's authorization carries down the graph. The N+1 trap and authorization-per-resolver are two reasons GraphQL authorization is hard; see the paradigm post for the query mechanics.

## Multi-tenancy: always filter by tenant

BOLA in a multi-tenant API has a specific, high-stakes shape: **tenant A must never see tenant B's data.** Multi-tenancy means many independent customers (organizations) share one deployment and one database, separated only by a `tenant`/`org` column and the discipline of your queries. That separation is *logical*, not physical — there is no firewall between org A's rows and org B's rows except your `WHERE` clause. Forget it once and you have cross-tenant data leakage, the single most reputation-ending bug a B2B platform can ship.

The non-negotiable rule: **every query that touches tenant-scoped data filters by the caller's tenant.** Not most queries. Every one. The tenant id comes from the *authenticated principal* (the `org` claim in the verified token), never from a request parameter — because anything the client sends, the client can change. A `?org=org_B` query param, or an `X-Org-Id` header the client controls, is not a tenant boundary; it is an invitation.

```python
# WRONG — tenant from a client-controlled input
org = request.args.get("org")           # attacker sets ?org=org_B
orders = db.orders.find({"org": org})

# RIGHT — tenant from the authenticated principal
org = g.principal.org                    # from the verified token, immutable here
orders = db.orders.find({"org": org})
```

The defense-in-depth ladder for tenancy, weakest to strongest:

1. **Filter in each query.** Add `org = principal.org` everywhere. Correct but fragile — relies on every developer remembering, forever.
2. **Centralize in a repository/data layer.** A `ScopedRepository(principal)` whose every method injects the org filter. Now a developer cannot accidentally write an unscoped query without going around the shared data layer on purpose.
3. **Database row-level security (RLS).** Set a session variable (`SET app.current_org = 'org_A'`) at connection checkout and let the database enforce a policy that filters every row. Even a raw query with a bug cannot cross tenants. This is the strongest practical defense for shared-schema multi-tenancy.
4. **Physical isolation.** Separate schema or database per tenant. Strongest isolation, highest operational cost; reserved for the largest or most sensitive tenants.

Most teams should be at level 2 minimum and reach for level 3 when the data is sensitive (payments certainly qualify). The dangerous `GET /orders/{id}` that forgets the tenant check is the textbook level-0 failure — no enforced filter at all.

A concrete row-level-security policy, for those who want the strongest practical isolation in a shared schema (PostgreSQL syntax):

```python
# Set up once per table
# ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
# CREATE POLICY tenant_isolation ON orders
#   USING (org = current_setting('app.current_org'));

# Then at connection checkout, bind the principal's org to the session:
def checkout_connection(principal):
    conn = pool.acquire()
    conn.execute("SET app.current_org = %s", (principal.org,))
    return conn
```

With this in place, *every* query against `orders` — even a hand-written one in a one-off script, even a buggy one a developer wrote at 2 a.m. — is automatically constrained to the caller's org by the database itself. The application can no longer leak across tenants by forgetting a `WHERE` clause, because the constraint does not live in the application. The cost is real (you must set the session variable reliably at checkout, and connection-pool reuse means a missed `SET` reuses the previous tenant's context — a subtle and dangerous bug), but for payment-grade data it is the isolation worth paying for. For how the database evaluates these filters efficiently, the index has to lead with the tenant column; see [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans).

## Where to enforce: gateway for coarse, service for object-level

A natural instinct is to centralize all authorization at the API gateway — one place to reason about, no per-service duplication. It is half right, and the wrong half is dangerous.

The gateway sees the request *before* any business logic: the token, the method, the path, the headers. That is exactly enough to enforce the **coarse** layers — validate the token's signature and expiry, check that the token has the scope the route requires, perhaps enforce a coarse role. It is *not* enough to enforce the **object** layer, because the gateway has not looked up `ord_99` and does not know it belongs to `org_B`. Object ownership is a fact in your data, discoverable only by a query the gateway does not (and should not) run. **You cannot do a BOLA check at the gateway.**

![A branching flow from a client through a gateway that checks the token and scope and can deny at the edge, into the orders service that performs the object and field checks before reaching a store whose rows are scoped to the tenant](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-6.png)

So the responsibilities split:

| Layer | Where | Why there |
| --- | --- | --- |
| Token validity (signature, expiry) | Gateway | Stateless, no domain knowledge needed; reject junk at the edge |
| Scope check | Gateway | The required scope is a property of the *route*, known without data |
| Coarse role | Gateway or service | Sometimes known from claims; often needs org-membership lookup → service |
| **Object ownership (BOLA)** | **Service / domain** | Needs a data lookup: whose object is this? |
| Field-level | Service / domain | Needs the object *and* the role to decide which fields |

A gateway config that enforces the coarse layer for the route:

```yaml
# Gateway route policy (Kong/Envoy-style)
routes:
  - path: /v1/refunds
    methods: [POST]
    auth:
      jwt:
        issuer: https://auth.commerce.example
        audiences: [https://api.commerce.example]
      require_scopes: [refunds:write]   # coarse: the token may create refunds
    # NOTE: ownership of the specific payment is checked IN the refunds service.
```

This rejects unauthenticated and under-scoped requests at the edge — cheap, fast, and it keeps junk traffic off your services. But the comment is the load-bearing part: *ownership is checked in the service.* The gateway is a coarse filter, not the authorization system. Treating "the gateway checks scopes" as "authorization is handled" is precisely how BOLA ships. Defense in depth: coarse at the edge, fine in the domain, and never assume the upstream did the downstream's job. For the gateway's broader role, see [the api gateway and backend for frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), and for the trust model between services, [service-to-service security: mtls and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

## 403 versus 404: hiding the existence of objects

When the object check fails — the caller is authenticated, correctly scoped, but does not own the resource — what status code do you return? The choice has a security dimension, not just a semantic one.

- **`403 Forbidden`** means "I know what you are asking for, it exists, and you may not have it." It *confirms the object exists.*
- **`404 Not Found`** means "there is nothing here for you." It *does not confirm existence.*

For an action a principal is simply not allowed to perform — a `viewer` trying to refund — `403` is correct and honest: the *capability* is denied, and revealing "you lack `refund:create`" leaks nothing sensitive. But for an **object the caller does not own**, returning `403` leaks the object's *existence*. An attacker enumerating `/orders/{id}` learns: `404` means "no such id," `403` means "real order, just not yours." Now they have a valid-id oracle — they can map your entire keyspace, learn how many orders a competitor has, and target follow-up attacks, all without ever reading a body.

The defensive convention, used by GitHub among others: **return `404` for objects the caller cannot access, as if they do not exist.** Same response for "no such order" and "an order that is not yours." The attacker learns nothing from enumeration. This is why, in the BOLA fix above, the tenant-scoped query returns `None` and we map that to `404` — "not yours" collapses into "not found" automatically, which is both simpler code and better security.

A practical rule of thumb:

| Situation | Code | Reasoning |
| --- | --- | --- |
| Wrong action, capability denied (`viewer` refunds) | `403` | The role is not secret; revealing the missing permission is safe and helpful |
| Object exists but not owned by caller | `404` | Hide existence — avoid the valid-id enumeration oracle |
| Missing/invalid token | `401` | Authentication, not authorization — tell them to authenticate |
| Valid token, missing scope for the endpoint | `403` | The endpoint exists; the token class is under-privileged |

The nuance: do not be dogmatic. Inside a tenant, where every member can see the *list* of objects anyway, a `403` on one of them leaks nothing and is more honest. The `404` discipline matters most at trust boundaries — across tenants, across orgs, on public endpoints — where existence itself is sensitive. The status-codes sibling [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx) covers the broader semantics; here the rule is: **across a tenant boundary, prefer `404` to avoid leaking existence.**

## Policy engines versus hand-rolled checks

So far the checks have been inline Python — `if`-statements and set membership scattered through handlers. That is fine, even ideal, for a small, stable surface. But as the rules grow — more roles, attributes, relationships, conditions — inline checks become a liability: the logic is duplicated across endpoints, drifts out of sync, is hard to test in isolation, and is impossible to audit ("show me every rule that grants refund access" requires grepping the whole codebase). The mature answer is to **separate the policy decision from the enforcement** with a dedicated policy engine.

The architecture has names from the access-control literature: the **Policy Enforcement Point (PEP)** is the spot in your service that *asks* the question and acts on the answer; the **Policy Decision Point (PDP)** is the engine that *evaluates* the rules and returns allow/deny. Your service builds a decision request (principal, action, resource, context) and the engine answers it, evaluating policy written in a dedicated language.

![A branching decision flow where a service acting as the enforcement point sends request context to a policy engine, which combines relationship data and returns an allow or a deny, both of which are written to an audit log](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-7.png)

Two prominent open ecosystems:

**Open Policy Agent (OPA)** — a general-purpose policy engine with the Rego policy language. The service sends a JSON input; OPA evaluates Rego rules and returns a decision. A refund policy in Rego:

```python
# Rego (OPA) — file: refunds.rego
package refunds

default allow = false

allow {
    input.action == "refund:create"
    input.principal.role == "admin"
    input.resource.org == input.principal.org   # tenant match
    input.resource.amount <= 10000              # under $100.00, in cents
}
```

Your service calls OPA with the request context and trusts the boolean back. The policy now lives in one auditable place, versioned and testable on its own, decoupled from handler code.

**Cedar** (the language behind AWS Verified Permissions) and **OpenFGA** / **SpiceDB** (Zanzibar-style ReBAC) occupy the relationship end. A Cedar policy reads close to natural language:

```python
// Cedar policy
permit (
    principal,
    action == Action::"refund:create",
    resource
)
when {
    principal.role == "admin" &&
    resource.org == principal.org &&
    resource.amount <= 10000
};
```

The trade-off is real and worth stating plainly:

| | Hand-rolled checks | Policy engine (OPA/Cedar/OpenFGA) |
| --- | --- | --- |
| Setup cost | Near zero | New component, language, deploy story |
| Best for | Small, stable rule sets | Large, evolving, cross-service policy |
| Auditability | Poor — logic scattered | Strong — policy is one artifact |
| Testability | Per-handler | Policy tested in isolation |
| Consistency across services | Drifts | One source of truth |
| Latency | Inline, nanoseconds | A call (mitigated by sidecar/embedded) |

A policy engine is not free — it is an operational component, a language to learn, and a decision-point to keep fast and available. The guidance: **start hand-rolled, move to an engine when the rules outgrow a few clear `if`-statements or must be shared across services.** Reaching for OPA on day one for a three-role CRUD API is the same over-engineering as reaching for ReBAC before you have a sharing requirement.

## Field-level and action-level authorization

The finest grain is *within* an object. Two callers may both be allowed to read order `ord_99`, but see different *fields*; both may act on it, but only one may take a specific *action*.

**Field-level** authorization redacts or restricts fields by role. A support agent reads an order to help a customer but must not see the full card number; a partner integration sees `total` but not the internal `cost_basis`. The implementation is a per-role field allow-list applied on the way out:

```python
VISIBLE_FIELDS = {
    "owner":   {"id", "total", "status", "items", "cost_basis", "card_last4"},
    "support": {"id", "total", "status", "items", "card_last4"},   # no cost_basis
    "partner": {"id", "total", "status"},                          # no items, no internals
}

def project(order, role):
    allowed = VISIBLE_FIELDS[role]
    return {k: v for k, v in order.items() if k in allowed}
```

Equally important is the *write* side: a `PATCH` must allow-list which fields each role may set. Binding a request body straight onto your model is the **mass-assignment** bug — a member `PATCH`es `{"status": "paid"}` or `{"role": "owner"}` and the naive code happily writes it. The fix is a writable-field allow-list per role, never a blind merge.

**Action-level** authorization gates specific operations beyond CRUD. `POST /v1/orders/ord_99/cancel` and `POST /v1/orders/ord_99/refund` are distinct actions, each with its own permission (`order:cancel`, `refund:create`), even though both are "writes to an order." This is where exposing actions as explicit sub-resources or RPC-style endpoints pays off: the permission maps one-to-one to the action, instead of a coarse `order:write` that conflates "edit the shipping address" with "issue a refund." Model the action, then authorize the action.

## Delegation, impersonation, and auditing

Real platforms need two patterns that complicate the simple "the principal is the user" story.

**Delegation** lets one principal act *on behalf of* another with their consent — the OAuth model where an app holds a token *for* a user and acts within the user's authority, bounded by scopes. The decision must consider both: the app's scopes *and* the user's role/ownership. The effective permission is the **intersection** — the app can do no more than the user could, and no more than the scopes allow. This is why a `payments:write`-scoped app acting for a `viewer` user still cannot refund: `viewer` lacks the role even though the token has the scope.

**Impersonation** (often "act-as" or "sudo") lets a privileged principal — a support engineer — perform actions *as* a customer to reproduce an issue. This is powerful and dangerous, and the rules are non-negotiable:

- It must be **explicit and bounded** — a deliberate "act as `usr_x`," not an ambient capability, ideally time-limited and re-authenticated.
- The audit log must record **both identities**: the *real* actor (the support engineer) and the *impersonated* subject. OAuth token exchange (RFC 8693) carries exactly this with `act` (actor) and `sub` (subject) claims. "Refund issued by `usr_customer`" is a lie if a support engineer did it while impersonating; "Refund issued by `usr_support` acting as `usr_customer`" is the truth, and the truth is what you need when something goes wrong.

Which brings us to **auditing**, the layer people remember only after an incident. *Every authorization decision that matters — especially every deny, every impersonation, every privileged action — should be logged*: who (real and effective), what action, which object, the decision, and why. An audit log answers the post-incident questions that determine whether you can scope a breach: *who accessed this order? was this refund authorized? did anyone read tenant B's data?* A policy engine helps here too — because decisions flow through one PDP, you get a natural, uniform audit stream instead of inconsistent log lines scattered across handlers (the audit edge in the policy-engine figure above). Authorization that is not audited is authorization you cannot prove after the fact, which in a payments system is barely authorization at all.

A reasonable audit record for a denied cross-tenant access:

```json
{
  "ts": "2026-06-20T14:03:11Z",
  "actor": "usr_7c2",
  "acting_as": null,
  "action": "order:read",
  "resource": "order:ord_600",
  "resource_org": "org_B",
  "actor_org": "org_A",
  "decision": "deny",
  "reason": "object_not_owned",
  "request_id": "req_a91f"
}
```

That single line — emitted on the deny path — is what lets you later prove the BOLA attempt was *blocked*, count how many ids the attacker probed, and decide whether to rate-limit or ban. Pair authorization with rate limiting (see the sibling [rate limiting, quotas, and abuse protection](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection)) so an enumeration sweep trips a limit before it finishes.

Three practical rules make audit logs actually useful rather than write-only noise:

- **Log denies, not just allows.** Allows are the normal case and high-volume; denies are the signal. A spike of `object_not_owned` denies from one actor across many ids is an enumeration attack in progress, and it is invisible if you only log successes.
- **Log the *reason*, machine-readable.** `"reason": "object_not_owned"` versus `"insufficient_role"` versus `"missing_scope"` lets you triage at a glance whether a deny wave is an attack, a misconfigured client, or a deploy that broke a permission. A bare "deny" tells you almost nothing.
- **Make the audit log tamper-evident and separate.** Authorization decisions are exactly what an attacker who *does* get in wants to erase. Ship them to an append-only store the application cannot rewrite. For payments, this is frequently a compliance requirement, not a nicety.

The deepest reason to centralize through a policy engine, ahead of latency or reuse, is this audit story: when every decision flows through one decision point, the audit stream is uniform, complete, and queryable, instead of a scatter of inconsistent log lines that each developer formatted their own way. You can answer "who could have read this order, and who did?" with a query rather than an archaeology project.

## Case studies: how the real systems do it

A few accurate, named references, because the patterns in this post are not theoretical.

**OWASP API Security Top 10 — BOLA at number one.** The OWASP Foundation publishes the API Security Top 10 (most recent edition 2023), and **Broken Object Level Authorization (API1)** sits at the top, with **Broken Object Property Level Authorization (API3)** and **Broken Function Level Authorization (API5)** also in the list. Three of the top risks are authorization failures. The consistent finding behind the list: the most damaging API breaches are not broken authentication — they are authenticated callers reaching objects they should not, because the per-object check was missing. If you read one external source from this post, read the OWASP API Security Top 10.

**Google Zanzibar and OpenFGA — ReBAC at scale.** Google published *"Zanzibar: Google's Consistent, Global Authorization System"* (USENIX ATC 2019), describing the relationship-tuple system that authorizes Drive, YouTube, Cloud, and more — handling trillions of relationship tuples and millions of authorization checks per second with a defined consistency model (the "zookie" for snapshot consistency). Zanzibar is the intellectual parent of the open-source ReBAC engines **OpenFGA** (a CNCF sandbox project, originally from Auth0/Okta) and **SpiceDB** (by AuthZed). When per-object sharing and nested groups are your core problem — collaboration, folders, multi-tenant SaaS with cross-org sharing — this is the lineage to study.

**OPA and Cedar — policy as code.** The **Open Policy Agent** (a graduated CNCF project) with its **Rego** language is the general-purpose policy engine widely used for both API authorization and infrastructure policy (Kubernetes admission control, Terraform checks). **Cedar** is the open-source policy language created by AWS and used by Amazon Verified Permissions; it is designed for fast, analyzable authorization with policies that read close to plain language. Both embody the PEP/PDP split: decisions move out of handler code into reviewable, testable, versioned policy.

**Auth0 / Okta and the platform RBAC model.** Commercial identity platforms (Auth0, Okta, and the cloud IAMs) ship the RBAC building blocks directly: roles, permissions, and scope-bearing tokens. The common practical pattern they encode is exactly the hybrid this post recommends — coarse scopes on the token plus roles for users — with object-level decisions left to (or pushed into a companion engine like OpenFGA at) the application. The lesson from the platforms: scopes and roles are *solved and standardized*; the object-level check is the part *your* domain still has to own, because only your data knows who owns what.

**GitHub and the `404`-for-private convention.** GitHub's API famously returns `404` rather than `403` for resources a caller is not permitted to see (private repositories, for instance), specifically so that the existence of a private resource is not leaked through status codes. It is the cleanest real-world example of the existence-hiding rule from the `403`-versus-`404` section.

## Verifying the authorization contract

Authorization is a contract clause like any other, which means it can be tested, and *must* be — because it fails silently. A missing scope check throws a loud `403`; a missing *ownership* check throws nothing at all. It returns a `200` with the wrong data, and your happy-path tests, which authenticate as the resource owner, pass green forever. The bug is invisible to exactly the tests most teams write. So you need tests written from the *attacker's* seat.

The single highest-value test is the **cross-tenant test**, and every object-returning endpoint deserves one:

```python
def test_cannot_read_other_tenants_order(client):
    # Arrange: an order owned by tenant B
    order = create_order(org="org_B")
    token_a = issue_token(org="org_A", role="admin", scopes=["orders:read"])

    # Act: tenant A, fully authenticated and scoped, requests B's order
    resp = client.get(f"/v1/orders/{order.id}",
                       headers={"Authorization": f"Bearer {token_a}"})

    # Assert: hidden as not-found, never leaked
    assert resp.status_code == 404
```

Note the deliberate setup: tenant A is an `admin` with the `orders:read` scope — the *strongest* legitimate read position — and still must be denied for an object that is not theirs. A weaker test (a `viewer` with no scope) would pass for the *wrong reason*: it would be denied at the role or scope gate, never exercising the object gate. To test the object gate you must give the caller everything *except* ownership.

The matrix of tests worth writing for each protected endpoint:

| Test | Principal | Expected | Gate exercised |
| --- | --- | --- | --- |
| Owner reads own object | tenant A, has scope | `200` | happy path |
| Cross-tenant read | tenant A, full scope, B's object | `404` | object/ownership |
| Wrong action for role | `viewer`, own object, refund | `403` | role |
| Missing scope | token without scope | `403` | scope |
| No token | anonymous | `401` | authN |
| Mutate a field above role | `member` sets `status` via PATCH | `403`/ignored | field-level |

Beyond per-endpoint tests, the structural defenses earn their keep:

- **A lint or test that fails on an unscoped query.** If your data layer is a `ScopedRepository`, a static check can flag any direct `db.orders.find` outside it. The goal is to make the unsafe path *unwritable*, then test that the safe path is taken.
- **Authorization fuzzing in CI.** Tools and scripts that take a valid session and replay every endpoint with another tenant's ids, asserting non-`200`. This catches the new endpoint a teammate added last week without the ownership check — the single most common regression.
- **Policy unit tests, when you use an engine.** OPA's Rego and Cedar both support testing the policy *in isolation*, with synthetic inputs, so you can assert "an `admin` may refund under \$100 but not over" without standing up the whole service. This is one of the strongest arguments for a policy engine: the decision becomes independently testable.
- **Audit-log assertions.** Assert that a denied cross-tenant attempt *emits the deny audit record*. The blocked attack is only useful evidence if it was actually logged.

The mindset shift is the point. Functional tests prove the API does what it should *for the right caller*. Authorization tests prove it does *nothing* for the wrong one. Both are the contract; only one of them is the one that keeps you off the front page.

## When to reach for each layer (and when not to)

Authorization is a place where over-engineering is as common as under-engineering. Decisive guidance.

![A decision tree for picking an authorization model, branching from stable roles to RBAC, from context mattering to ABAC, and from per-object sharing to ReBAC](/imgs/blogs/authorization-scopes-roles-and-resource-level-permissions-8.png)

- **Always do scopes and object-level checks.** These are not optional and not a maturity stage. Every token gets bounded scopes (at minimum, read versus write); every object access gets an ownership check. There is no API small enough to skip the object check — that is the BOLA hole.
- **Use RBAC for stable, role-shaped rules.** A handful of roles (`owner`/`admin`/`member`/`viewer`) covers the vast majority of APIs. Start here. It is cheap, auditable, and everyone understands it.
- **Reach for ABAC when context drives the decision** — region, amount thresholds, time windows, MFA-recency, data classification. The tell is a role whose meaning depends on the *request*, not just the user. Add attributes; do not mint `member_eu_small_refund`.
- **Reach for ReBAC when sharing is per-object** — collaboration, nested groups, folder/hierarchy inheritance, "share *this one* order with an external auditor." The tell is permission that follows a *relationship between objects*, not a role on a user. This is where a Zanzibar-style engine earns its complexity.
- **Reach for a policy engine when rules outgrow a few clear `if`-statements**, when policy must be shared across services, or when auditors need a single artifact to review. Not before.

And the **do nots**:

- **Do not treat scopes as object permissions.** The whole post. A scope authorizes an endpoint class, never a specific record.
- **Do not enforce object-level checks at the gateway.** It lacks the data. Coarse at the edge, fine in the service.
- **Do not start with a policy engine or ReBAC** for a three-role CRUD API. You buy operational complexity and a new language to debug, for rules a `dict` would express. Escalate on evidence, not on aspiration.
- **Do not trust any id from the client** — path, query, or body. Re-validate ownership every time.
- **Do not derive the tenant from request input.** The `org` comes from the verified token, never from `?org=` or a client header.
- **Do not return `403` across a tenant boundary** if it leaks the existence of objects an attacker could enumerate. Prefer `404`.
- **Do not skip the deny-path audit log.** The blocked attempt is the evidence you will want.

## Key takeaways

- **AuthN proves who; authZ decides what — and *to which object*.** Authentication is the start of the conversation, never the end. A valid token grants nothing by itself.
- **Authorization is four layers: scope, role, object, field.** A request must clear all four. Most teams build the first two and leave the object layer empty — which is exactly where the breach is.
- **Scopes and roles are *not* object-level permissions.** They authorize an action on a *type*; only the object check authorizes a *specific instance*. Passing the first never implies the second.
- **BOLA is the number-one API vulnerability, and it is preventable.** Check ownership on *every* object access; never trust a client-supplied id; push the tenant filter into the data layer so an unscoped query is impossible to write.
- **Always filter by tenant, from the verified principal.** Logical separation is one `WHERE` clause from a cross-tenant leak. Prefer a scoped repository or row-level security over per-query discipline.
- **Enforce coarse at the gateway, fine in the service.** You cannot do a BOLA check at the gateway — it does not know who owns the object.
- **Prefer `404` over `403` to hide existence across trust boundaries.** Map "not yours" to "not found" so enumeration leaks nothing.
- **Reach for ABAC/ReBAC/policy engines on evidence, not aspiration.** Start RBAC plus object checks; escalate when context or per-object sharing forces it.
- **Audit every decision that matters — and record both identities under impersonation.** Authorization you cannot prove after the fact is barely authorization.

This is the clause of the API contract that says *"your data is yours."* Get the object-level check right on every endpoint, default to deny, filter by tenant always, and you close the door that most breaches walk through. For the layer below this — proving identity in the first place — see [authentication: api keys, sessions, jwt, and mtls](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) and [oauth2 and openid connect for api designers](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers); for the layer beside it — keeping malformed and malicious input out — see [input validation, output encoding, and the owasp api top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10). And tie it all back to the series frame in [what is an api: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and the [api design playbook review checklist](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

## Further reading

- **OWASP API Security Top 10 (2023)** — the canonical risk list; read API1 (BOLA), API3 (Broken Object Property Level Authorization), and API5 (Broken Function Level Authorization).
- **"Zanzibar: Google's Consistent, Global Authorization System"** (USENIX ATC 2019) — the foundational ReBAC paper behind OpenFGA and SpiceDB.
- **Open Policy Agent (OPA) documentation and the Rego language** — the general-purpose policy engine and policy-as-code model.
- **Cedar policy language** (cedarpolicy.com) and **AWS Verified Permissions** — analyzable, readable authorization policies.
- **OpenFGA** and **SpiceDB** — open-source Zanzibar-style relationship-based access control engines.
- **RFC 6749 (OAuth 2.0)** and **RFC 8693 (OAuth Token Exchange)** — scopes, delegation, and the `act`/`sub` claims for impersonation.
- **NIST RBAC model (INCITS 359)** — the formal definition of role-based access control.
- Within this series: [authentication: api keys, sessions, jwt, and mtls](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls), [oauth2 and openid connect for api designers](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers), [input validation, output encoding, and the owasp api top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10), and out to [service-to-service security: mtls and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

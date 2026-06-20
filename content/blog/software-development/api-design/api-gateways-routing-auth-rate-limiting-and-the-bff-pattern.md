---
title: "API Gateways: Routing, Auth, Rate Limiting, and the BFF Pattern"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Design the edge layer that fronts your API — what a gateway is, the cross-cutting concerns it owns at the front door, where coarse auth stops and object-level authorization stays in the service, and how the Backend-for-Frontend pattern gives each client one tailored payload instead of one chatty shared API."
tags:
  [
    "api-design",
    "api",
    "rest",
    "api-gateway",
    "bff",
    "rate-limiting",
    "http",
    "auth",
    "microservices",
    "operability",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-1.png"
---

The incident started with a deploy that touched nothing but a single service. The payments team shipped a config change that accidentally dropped a database connection pool from 50 to 5. Payments got slow — calls that used to return in 40 ms now took 6 seconds. That part was their bug, and it was contained. But within ninety seconds, the *orders* page was down too. And the *account* page. And the *mobile app's home screen*, which has nothing to do with payments at all.

Why? Because every one of those clients talked to payments through the same front door, and that front door had no per-route timeout. A request to render the order list would call orders, then call payments to decorate each order with its charge status, and *wait*. Threads piled up at the edge holding open sockets to a service that wasn't answering. The edge ran out of worker threads. Now requests that never touched payments — the account page, the mobile home screen — couldn't get a worker either. One slow service, multiplied by a shared front door with no isolation, took down the whole product. The fix was not heroics; it was a two-second timeout and a circuit breaker on the payments route, plus the realization that the thing fronting our API was load-bearing infrastructure we had never actually designed.

That front door has a name: the **API gateway** — a single edge component that sits between your clients and your services and handles the concerns that every service would otherwise re-implement: terminating TLS, checking that a caller is who they claim to be, enforcing how many requests they may send, routing the request to the right upstream, and stamping a correlation ID on it so you can trace it end to end. The gateway exists for one reason above all: **cross-cutting concerns should be solved once, at the edge, not copy-pasted into every service.** Get it right and your services stay small and focused. Get it wrong and you either re-implement auth ten times (and get it subtly wrong in three of them) or you grow a "gateway of doom" that has swallowed your business logic and become the monolith you were trying to avoid.

This post is about designing that edge layer well. By the end you will be able to say, for any concern, whether it belongs at the gateway or in the service; write a gateway route that authenticates, rate-limits, and dispatches to the right upstream; explain why **coarse** authentication belongs at the edge but **object-level** authorization does not; and design a **Backend-for-Frontend (BFF)** — a tailored API per client that aggregates and reshapes downstream services so a mobile screen makes one call instead of three. We will keep returning to the series' running example, a **Payments & Orders** commerce platform with `/orders`, `/payments`, `/refunds`, and a mobile app, because the edge is where all of those contracts meet the real world. The frame that runs through this whole series holds here too: an API is a contract and a product, not a function call — and the gateway is the part of the product the caller hits *first*, so its behavior under load, under attack, and under partial failure is part of the contract whether you designed it or not.

![A request flowing through an API gateway that terminates TLS, authenticates the token, rate-limits, then routes to the orders or payments service with a separate deny path for rejected requests](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-1.png)

## 1. What a gateway is, and the problem it solves

Strip away the product names and an API gateway is a **reverse proxy with opinions**. A plain reverse proxy forwards a request to a backend. A gateway forwards it *after* running a pipeline of cross-cutting checks, and it does that for many backends behind one address. The word "gateway" is doing the work of "single front door for a fleet of services."

To see why you want one, picture the alternative. You have three services — orders, payments, shipments — each exposing HTTP. Each one needs to: terminate TLS, parse the `Authorization` header, validate the bearer token (a **bearer token** is a credential where merely possessing the string grants access — "bearer" because the bearer of the token is treated as the authorized party), enforce a rate limit, log a request ID, and emit metrics. If each service does this itself, you have the same fifty lines of auth middleware in three codebases, in possibly three languages, maintained by three teams. When you rotate the signing key for your tokens, you deploy three times. When a new service joins, it re-implements the lot — and the new team's version of "validate the token" forgets to check the `exp` (expiry) claim, so expired tokens work against the new service for six weeks until someone notices.

A gateway pulls all of that to one place. The services behind it can assume: *if a request reached me, it already came over TLS, the caller is authenticated, and it's within rate limits.* They get to be small. They get to focus on their domain. This is the same robustness logic that runs through the whole series — you reduce the number of places a rule lives so there are fewer places to get it wrong.

> **The principle.** A gateway is justified exactly when a concern is (a) **cross-cutting** — every service needs it — and (b) **uniform** — it can be applied without knowing the service's domain. TLS termination, token verification, coarse rate limiting, request-ID injection, and routing all qualify: none of them need to know what an "order" is. Anything that requires domain knowledge — *can this user see this particular order?* — fails test (b) and must not live at the gateway. That single test, applied honestly, tells you what belongs at the edge and what does not, and it is the whole reason "gateway" and "monolith" are not synonyms.

Here is the request lifecycle the gateway runs, in order. The order matters: you authenticate before you rate-limit (so you can rate-limit per *identity*, not just per IP), and you rate-limit before you route (so a flood never reaches an upstream at all).

![A vertical stack of gateway responsibilities from TLS termination at the bottom through authentication, rate limiting, routing, transformation, observability, and caching at the top](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-2.png)

Read from the bottom up, that stack is the gateway's job description: terminate TLS and pick up an HTTP/2 connection; verify the token (**coarse** auth only — more on that limit below); apply the rate limit and per-tenant quota; route by path to the right upstream; optionally validate and transform the request; inject a correlation ID and emit RED metrics (Rate, Errors, Duration); and serve from cache or split traffic for a canary on the way out. Every one of those is a thing you would otherwise build, badly, in each service.

A note on vocabulary you will see in vendor docs. The traffic the gateway handles — between clients outside your system and your services — is called **north-south** traffic. The traffic *between* your services — orders calling payments calling shipments — is **east-west** traffic. The gateway owns north-south. East-west is the job of a different tool, the service mesh, which we will get to in section 8. Conflating the two is the single most common architecture mistake here, so hold that distinction: **gateway = client-to-edge; mesh = service-to-service.**

The full responsibility list, with the test applied to each, is worth having in one place. Every row that's "yes, at the edge" passes both halves of the principle — cross-cutting *and* uniform — and every row that's "no, in the service" fails the uniformity half because it needs to know what an order or a payment *means*:

| Responsibility | At the gateway? | Why | Stays in the service? |
| --- | --- | --- | --- |
| TLS termination | Yes | Uniform, certificate management in one place | Re-encrypt internally, optional |
| Token verification (authN) | Yes | Same check for every route | No — already proven at edge |
| Coarse scope check (authZ) | Yes | A boolean from the token, no domain | May re-check defensively |
| Object-level authZ | **No** | Needs to know who owns the row | **Yes — the service owns the data** |
| Rate limiting / quotas | Yes | Uniform accounting per identity | Local limits only if needed |
| Routing / load balancing | Yes | The whole point of a front door | n/a |
| Request validation (shape) | Yes | Cheap first line, reject 400/415 early | Yes — owns its invariants |
| Protocol translation (REST↔gRPC) | Yes | Mechanical, no domain knowledge | n/a |
| Response caching | Yes | Uniform for safe, cacheable responses | Sets cache headers it understands |
| Observability injection | Yes | Sees every request, stamps one ID | Adds domain spans |
| Business logic / workflow | **No** | Needs domain knowledge | **Yes — always** |

Read that table as the gateway's charter. Anything in the "Yes" rows is fair to pull to the edge; anything in the two bold "No" rows is a step toward the gateway of doom we dissect in section 6. The boundary is not stylistic — it's the line between an edge that stays thin and fast and one that slowly absorbs your platform until it *is* your platform.

## 2. Routing and TLS termination: the boring, load-bearing core

The most basic thing a gateway does is **routing**: look at an incoming request and decide which upstream gets it. Almost always this is **path-based** — `/orders/*` goes to the orders service, `/payments/*` goes to payments — though you can also route on host (`api.shop.com` vs `internal.shop.com`), header, or HTTP method. The client sees one consistent surface, `https://api.shop.com/...`, and never learns that orders and payments are separate deployments on separate hosts. That indirection is a feature: you can split a service in two, move it to a new host, or shard it by tenant without any client changing a single URL.

Routing also means **load balancing** across the instances of an upstream — the gateway picks one of the N healthy payments pods and forwards to it, usually round-robin or least-connections, skipping any that fail health checks. (How the gateway *learns* which pods are healthy and where they live is service discovery, which belongs to the fleet layer — see [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) for the mechanics; here we only care that the gateway has a current list of healthy upstreams to choose from.)

How does the gateway *pick* a route? It evaluates an ordered list of matchers against the incoming request — path prefix, exact path, host, method, header — and the **first match wins** (or the most-specific match wins, depending on the gateway). This ordering is itself a source of subtle bugs: if a broad rule (`/v1/*`) sits above a specific one (`/v1/payments`), the specific route never fires and payments traffic silently lands on the wrong upstream. The discipline is to order routes from most-specific to least-specific, and to treat the routing table as configuration you test — a routing change that mis-orders two rules can reroute live traffic with no code change and no obvious error. A good gateway lets you assert "this request → that upstream" in CI so a config edit can't quietly redirect production.

The other boring-but-load-bearing job is **TLS termination**. TLS (the encryption under HTTPS) is computationally real, and you do not want to manage certificates and renewals in every service. The gateway terminates the client's TLS connection — it holds the public certificate, decrypts the request, and then forwards it to the upstream over the internal network (often re-encrypted with an internal cert, or over a mesh's mTLS). Terminating once means certificate rotation happens in one place, and it means the gateway can read the request (headers, path) to make routing and auth decisions — which it could not do if the bytes stayed encrypted end to end.

Here is a concrete route config. This is a declarative gateway (the shape is close to Kong's or an Envoy/Contour `HTTPRoute`); read it as "match this path, do these things, send it here":

```yaml
# Gateway route: client-to-edge dispatch for the commerce API
routes:
  - name: orders-route
    match:
      hosts: ["api.shop.com"]
      paths: ["/v1/orders"]
    plugins:
      - jwt-auth          # verify the bearer token's signature + exp
      - rate-limit:
          limit: 100
          window: 60s     # per authenticated client
          identifier: jwt_claim.sub
      - correlation-id:
          header: X-Request-Id
          generate: uuid
    upstream:
      service: orders-svc
      timeout_ms: 2000    # per-route timeout — the lesson from the incident
      retries: 1
      circuit_breaker:
        max_failures: 5
        open_for: 30s

  - name: payments-route
    match:
      hosts: ["api.shop.com"]
      paths: ["/v1/payments"]
    plugins:
      - jwt-auth
      - rate-limit:
          limit: 20         # payments is more sensitive — tighter limit
          window: 60s
          identifier: jwt_claim.sub
      - correlation-id:
          header: X-Request-Id
    upstream:
      service: payments-svc
      timeout_ms: 2000
      retries: 0            # NEVER blind-retry a non-idempotent charge
```

Two design decisions in that config are worth dwelling on, because they are exactly the kind of thing that bites you in production.

First, the **per-route timeout**. The payments route gets `timeout_ms: 2000`. Without it, the gateway's default might be 60 seconds or, worse, *unbounded*, and you reproduce the incident from the intro: one slow upstream holds edge workers hostage and starves every other route. A timeout converts "slow forever" into "fail fast with a 504," which the client can handle. The math is unforgiving: if a route has 200 worker threads and each request to a hung upstream holds a thread for 60 s, you can absorb only about 3 such requests per second before the pool is exhausted; at a 2 s timeout that ceiling is 100/s. The timeout is not a nicety — it is the difference between a contained outage and a cascading one.

Second, **`retries: 0` on payments**. Retrying a `GET /orders` on a timeout is fine — `GET` is **safe** (it changes nothing) and **idempotent** (doing it twice has the same effect as once). Retrying a `POST /payments` is a way to charge a customer twice. The gateway must *not* blindly retry non-idempotent requests. (The right fix is an idempotency key carried by the client and honored by the payments service — covered in the [idempotency keys post](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions); the gateway's job is simply to *not make it worse* by retrying for the client.) This is a place where the gateway must respect HTTP semantics it learned from the [HTTP for API designers post](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers): safe and idempotent methods may be retried; the rest may not.

## 3. Authentication at the edge — and exactly where it stops

Here is the line that trips up the most teams. The gateway should handle **authentication (authN)** — *who is this caller?* — but only the **coarse** part of **authorization (authZ)** — *are you allowed in general?* — and it must **not** handle **object-level** authorization — *may you see this specific order?*. Getting this boundary right is most of what separates a clean gateway from a gateway of doom.

**Authentication at the edge.** The gateway verifies the caller's credential once, for everyone. For a public API that's usually a **JWT** (JSON Web Token, RFC 7519) carried in the `Authorization: Bearer <token>` header, or an API key. A JWT is a signed, base64url-encoded blob with three dot-separated parts — a header, a payload of **claims** (a claim is a statement about the caller, like `sub` for subject/user-id, `exp` for expiry, `scope` for granted permissions), and a signature. The gateway's job is precise: check the signature against the issuer's public key, check `exp` hasn't passed, check `iss`/`aud` are expected, and then **forward the verified identity to the upstream** — usually by stripping the raw token and injecting trusted headers like `X-User-Id` and `X-Scopes` that the upstream can trust *because it trusts the gateway*. The deep treatment of token types and trade-offs lives in the [authentication post](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls); here the point is *where* it happens: once, at the edge.

```http
POST /v1/payments HTTP/1.1
Host: api.shop.com
Authorization: Bearer <token>
Idempotency-Key: 8f1c2a4e-key
Content-Type: application/json

{ "order_id": "ord_8821", "amount_cents": 4999, "currency": "USD" }
```

The gateway verifies that token, finds it valid with `scope: "payments:write orders:read"`, and forwards to the payments upstream with the raw `Authorization` header replaced by trusted internal headers:

```http
POST /payments HTTP/1.1
Host: payments-svc.internal
X-User-Id: usr_5567
X-Scopes: payments:write orders:read
X-Request-Id: 3b9d-corr-id
Idempotency-Key: 8f1c2a4e-key
Content-Type: application/json

{ "order_id": "ord_8821", "amount_cents": 4999, "currency": "USD" }
```

**Coarse authorization (scope) at the edge.** The gateway *can* also do a cheap, uniform authZ check: does this token carry the scope this route requires? Posting to `/v1/payments` requires the `payments:write` scope; if the token lacks it, the gateway returns `403` without ever bothering the payments service. This is **coarse** because it's a single boolean from the token — "this caller is, in general, allowed to write payments" — and it needs no domain knowledge. It's the same kind of uniform rule as rate limiting, so it qualifies for the edge by the principle from section 1.

**Where it stops: object-level authZ stays in the service.** The gateway must *not* answer "may user `usr_5567` see order `ord_8821`?" Answering that requires knowing who owns `ord_8821`, which is a *row in the orders database* — domain knowledge the gateway does not and should not have. If the gateway tried, it would have to query the orders database, which makes it a second orders service, which is how the gateway-of-doom is born. So the boundary is sharp: **the gateway proves you are a valid caller with the right general scope; the service proves you may touch this specific object.** The full treatment of resource-level permissions — why this is the source of the OWASP "Broken Object Level Authorization" vulnerability, the #1 API risk — is in the [authorization post](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions). The one rule to carry away: a missing object-level check in the *service* is a data-leak bug that a perfect gateway cannot save you from.

This split is exactly what the next figure draws — concern by concern, what belongs at the edge versus in the service.

![A matrix comparing five concerns showing token authentication coarse scope and rate limiting belong at the gateway while object-level authorization and business logic belong in the service](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-3.png)

The matrix encodes the principle as a table you can apply to any new concern. Token authN, a coarse scope check, and rate limiting are green at the edge because they're cross-cutting and uniform. Object-level authZ and business logic are red at the edge because they need domain knowledge — they go in the owning service. When someone proposes "let's just have the gateway check the order owner," point at the red cell.

#### Worked example: a route that authenticates, scope-checks, and rate-limits

Let's wire the whole edge pipeline for the `POST /v1/payments` route and trace one request through it. Here is the route's policy, written declaratively:

```yaml
- name: create-payment
  match: { paths: ["/v1/payments"], methods: ["POST"] }
  plugins:
    - jwt-auth:
        issuer: "https://auth.shop.com"
        algorithms: ["RS256"]      # asymmetric — gateway holds only the public key
        verify: ["exp", "iss", "aud"]
    - require-scope:
        scope: "payments:write"    # coarse authZ — a boolean from the token
        on_missing: 403
    - rate-limit:
        limit: 20
        window: 60s
        identifier: jwt_claim.sub  # per user, not per IP
        on_exceed: 429
  upstream: { service: payments-svc, timeout_ms: 2000, retries: 0 }
```

Now four requests arrive, and you can predict each outcome from the policy alone:

1. **Valid token, has `payments:write`, 5th request this minute.** Signature checks out, `exp` is in the future, scope present, under the limit. The gateway forwards to `payments-svc` with `X-User-Id` injected. The service then does the object-level check (does this user own `ord_8821`?) and processes the charge. **Result: 201 Created.**
2. **Token signature invalid (tampered).** AuthN fails at the edge. **Result: 401 Unauthorized**, payments-svc never sees the request.
3. **Valid token, but scope is `orders:read` only.** AuthN passes, coarse authZ fails. **Result: 403 Forbidden** at the edge — again, payments-svc is never touched.
4. **Valid token with the right scope, but it's the 21st request this minute.** Everything passes except the limit. **Result: 429 Too Many Requests** with a `Retry-After` header.

Notice that three of the four rejections happen *at the edge*, before payments-svc burns any CPU. That's the efficiency dividend of authenticating and limiting at the front door: bad traffic dies cheaply and far from your expensive, stateful services.

**A note on the trust boundary.** The pattern above — strip the raw token, inject `X-User-Id` and `X-Scopes` — only works if the services *trust* those headers, and they should trust them *only* because they trust the gateway. That trust is not free; it has to be enforced. The standard mistake is to leave the services reachable directly (bypassing the gateway), so an attacker who can reach `payments-svc.internal` can simply *set their own* `X-User-Id: usr_admin` header and walk in as anyone. The fix is a real network boundary: services accept traffic *only* from the gateway (network policy, a private subnet, or — better — mutual TLS between the gateway and each service so the service cryptographically verifies the caller is the gateway). The trusted-header pattern is a convenience riding on top of a hard network boundary; without the boundary, it's a vulnerability with a friendly name. (This is precisely where the service mesh's east-west mTLS earns its place — see section 8.)

**API keys versus JWTs at the edge.** Not every caller carries a JWT. Server-to-server and partner integrations often present a long-lived **API key** instead — a single opaque secret string, usually in an `X-API-Key` header or as a bearer credential. The gateway handles both, but they sit at different points on the same trade-off curve, and the choice has real consequences for what the gateway must do on every request:

| Property | API key | JWT (RS256) |
| --- | --- | --- |
| Verification cost | Lookup in a store (stateful) | Signature check with cached public key (stateless) |
| Carries claims/scopes | No — key maps to a row | Yes — scopes baked into the token |
| Revocation | Instant (delete the row) | Hard — valid until `exp` unless you track a denylist |
| Expiry | Long-lived (manual rotation) | Short-lived (minutes to hours) |
| Best fit | Server-to-server, partners | End-user / OAuth clients |
| Gateway dependency | Needs the key store online | Self-contained once keys are cached |

The decisive line: a JWT can be verified *offline* (the gateway holds the issuer's public key and checks the signature — no network call), which keeps the edge fast and resilient even if the identity provider blips; but it's hard to revoke before it expires, which is why JWTs are short-lived. An API key is trivially revoked (delete the row) but requires a store lookup on every request, making the key store a dependency on the hot path. Most platforms use both: JWTs for end-user traffic (issued by an OAuth/OIDC flow — see the [OAuth2 and OpenID Connect post](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers)), API keys for partners and machine-to-machine. The gateway's job is the same in both cases — verify the credential once, forward a trusted identity — but the *failure modes* differ, and you design for both.

## 4. Rate limiting and quotas at the front door

Rate limiting is the gateway's seatbelt: it caps how many requests a caller may make in a window, so one client — buggy or malicious — can't consume the capacity that belongs to everyone. The gateway is the natural home for it because the limit is uniform (it's the same accounting for every route) and because you want to reject excess traffic *before* it reaches a service.

The standard algorithm is the **token bucket**. It works like a bucket that holds up to $B$ tokens and refills at a steady rate of $r$ tokens per second, up to its cap. Each request must take one token. The rule is: $\text{allow if } tokens \ge 1$, then decrement; otherwise reject with `429`. The bucket gives you two knobs that map to real behavior: $r$ is the sustained rate you permit over the long run, and $B$ is the **burst** you tolerate — a client that's been quiet can spend a full bucket at once. If $r = 20/60$ tokens per second (20 per minute) and $B = 20$, a client can fire 20 requests instantly, then is metered to one every 3 seconds until the bucket refills. The full derivation, the sliding-window alternative, and distributed-counter problems are the subject of the dedicated [rate limiting post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection); here we focus on the *contract* the gateway must present when it limits you.

That contract is HTTP-shaped and non-negotiable: when you reject, you return **`429 Too Many Requests`**, and you tell the client *when to come back* with a **`Retry-After`** header (seconds, or an HTTP date). You should also surface the limit state on every response so a well-behaved client can pace itself instead of probing for the wall. The de-facto headers are `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset`:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 18
RateLimit-Limit: 20
RateLimit-Remaining: 0
RateLimit-Reset: 18
Content-Type: application/problem+json

{
  "type": "https://api.shop.com/problems/rate-limited",
  "title": "Rate limit exceeded",
  "status": 429,
  "detail": "You have exceeded 20 requests per minute on /v1/payments.",
  "instance": "/v1/payments"
}
```

Two things make that a good citizen of the series' contract philosophy. The body is a **`problem+json`** error (RFC 9457) — a machine-readable envelope with `type`, `title`, `status`, `detail`, `instance` — so a client can program against the `type` URI rather than scraping a free-text string (this error contract is its own [error-design post](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract)). And the `Retry-After` turns a wall into a guardrail: a client that respects it backs off cleanly; a client that ignores it just keeps getting `429`s and never makes progress, which is the correct outcome for an abuser.

A subtlety: **rate limit by authenticated identity, not by IP**, which is exactly why auth runs *before* the limiter in the pipeline. Limiting by IP punishes everyone behind a corporate NAT or a mobile carrier gateway (thousands of users, one IP) and is trivially defeated by an attacker rotating IPs. Limiting by the token's `sub` claim gives each *account* its own bucket, which is fair and hard to evade. The cost is that you must authenticate first — which you were doing anyway.

**The distributed-counter problem, briefly.** A single gateway instance can keep the token bucket in local memory. But you run *several* gateway instances behind a load balancer (you must — it's a SPOF otherwise). Now a client's requests are spread across, say, four instances, and if each keeps its own local bucket of 20, the client effectively gets $4 \times 20 = 80$ — your limit silently quadrupled. The fix is a *shared* counter: the instances increment a counter in a fast central store (Redis is the usual choice), so all four see the same bucket. The trade-off is a network round-trip to Redis on every request, which adds latency and makes Redis a dependency on the hot path. Common mitigations are local pre-checks with periodic reconciliation, or sharding the counter — but the principle to internalize is that **rate limiting across a fleet is a distributed-state problem, not a local one**, and a naive per-instance limit leaks your cap by the instance count. The full treatment is in the [rate limiting post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection); the gateway-design takeaway is to make limit state shared, or accept that your real limit is `configured × instances`.

**Quotas versus rate limits.** They're related but distinct, and a mature gateway does both. A *rate limit* caps short-term velocity (20 requests per *minute* — protecting capacity from bursts). A *quota* caps longer-term volume (100,000 requests per *month* — usually tied to a pricing tier or a fair-use policy). Rate limits protect your system *now*; quotas govern the *commercial* relationship over time. A free-tier partner might get 1,000 requests/day; a paid partner 1,000,000. The gateway enforces both: it returns `429` for the velocity limit and (often) `403` or `429` with a distinct `problem+json` `type` for an exhausted quota, so the client can tell "slow down" from "you're out of allowance until next month." Surfacing quota state (`RateLimit-*` for the short window, a separate quota header or a portal for the monthly bucket) is part of treating the API as a product its consumers can plan around.

> **The consequence, before→after.** *Before:* an internal batch job, newly deployed, has a bug that calls `GET /v1/orders` in a tight loop — 5,000 requests/second. With no edge limit, those requests reach the orders service, exhaust its database connection pool, and now *every* client gets slow orders responses. A single buggy caller becomes a platform-wide brownout. *After:* the gateway caps that token at 100/min. The batch job gets `429`s after its first 100 requests and (if written correctly) backs off; the orders service never sees the flood; every other client is unaffected. The limit didn't fix the buggy job — but it contained the blast radius to the one caller that misbehaved. That containment is the entire point of putting the limiter at the front door.

## 5. The other edge jobs: transform, validate, aggregate, observe, canary

Routing, auth, and rate limiting are the headline jobs, but a gateway earns its keep on a handful of smaller cross-cutting concerns too. Each one follows the same test — cross-cutting and uniform, so it belongs at the edge.

**Request/response transformation.** The gateway can rewrite requests and responses without touching the upstream: strip internal headers before they leak to clients (you do *not* want `X-Internal-Trace` or a stack frame reaching the public), inject the trusted identity headers we saw, add CORS headers, or rewrite a legacy `/orders` path to a new `/v2/orders` upstream during a migration. Keep transforms *mechanical* — header munging, path rewriting, field renaming. The moment a transform needs to know what an order *means*, it's business logic and it has escaped to the wrong layer.

**Request validation.** The gateway can reject obviously malformed requests at the edge — wrong `Content-Type`, a body that fails the OpenAPI schema, a missing required header — returning `400` or `415` before the service is bothered. This is a cheap first line of defense and shrinks the attack surface each service must guard. It does *not* replace validation in the service (the service still owns its invariants), but it stops garbage early.

**Protocol translation.** A common and genuinely valuable gateway job is translating between protocols: clients speak REST/JSON over HTTP/1.1, while internal services speak **gRPC** over HTTP/2 for efficiency. The gateway accepts the JSON request, translates it to a gRPC call, and translates the response back. This lets you get gRPC's efficiency internally without forcing every external client (especially browsers, which can't speak raw gRPC) to deal with it. Which protocol fits which boundary is the subject of the [paradigm-choice post](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); the gateway is simply the seam where the translation happens.

**Caching.** For safe, cacheable responses, the gateway can serve from a shared cache and honor conditional requests — returning `304 Not Modified` when the client's `ETag` still matches, saving both the upstream call and the bytes on the wire. This is most of what a CDN does, pulled to your edge. The full mechanics of `ETag`, `Cache-Control`, and invalidation are the [caching post](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation); at the gateway the win is that a cached, validated response never wakes the service at all.

**Observability injection.** This one is quietly the most valuable. The gateway stamps every request with a **correlation ID** (a unique `X-Request-Id`) if one isn't present, and propagates it to every upstream. Now a single request can be traced across orders → payments → shipments by grepping one ID. The gateway also emits **RED metrics** — Rate (requests/sec), Errors (non-2xx rate), Duration (latency percentiles) — per route, for free, because every request passes through it. When the payments p99 latency spikes, you see it at the edge before any customer files a ticket. (Designing those signals into the API surface is the [observability post](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos); the gateway is where they're cheapest to collect.)

**Canary and traffic splitting.** Because all traffic flows through it, the gateway can route, say, 5% of requests for a route to a new version of a service while 95% go to the stable one. If the canary's error rate climbs, you shift the split back to 0% — a deploy you can undo by changing one number, without redeploying. This is north-south progressive delivery, and the gateway is the only place that sees enough traffic to do it. The split can be by percentage (5% of all traffic), by header (only requests with `X-Canary: true`, so your own QA hits the new version), or by a sticky hash of the user id (so a given user consistently sees one version, avoiding a jarring flip mid-session). The config is just a weighted upstream:

```yaml
- name: orders-route
  match: { paths: ["/v1/orders"] }
  upstream:
    split:
      - { service: orders-svc-v1, weight: 95 }
      - { service: orders-svc-v2, weight: 5 }   # canary — dial up or to 0
```

Because the gateway *also* emits per-upstream RED metrics, you can watch `orders-svc-v2`'s error rate and p99 next to v1's and make the dial-up decision on data, not hope. Pair that with the per-route circuit breaker and a bad canary fails *cheaply*: it serves 5% of traffic, its errors are visible immediately, and you set its weight to 0 in one edit. This is the operational payoff of having a single, instrumented front door — progressive delivery and instant rollback are configuration, not a deploy pipeline.

What is *not* on this list, and never should be: **business logic, object-level authorization, and stateful workflows.** The next section is about what happens when those creep in.

## 6. The gateway of doom: what does NOT belong at the edge

The gateway is seductive. It sees every request, it's already there, and "just add it to the gateway" is always the path of least resistance. Resist it. The failure mode is well known enough to have a name: the **gateway of doom** (sometimes "the overambitious API gateway") — a gateway that has accreted business logic until it has become the distributed monolith you split your services to avoid.

The tell-tale signs, each a thing that has failed test (b) — *can this be done without domain knowledge?*:

- **Object-level authorization.** Already covered, but it's the most common breach: "the gateway checks if you own the order." Now the gateway needs the orders schema, and a schema change is a coordinated gateway-plus-service deploy.
- **Response aggregation that knows the domain.** A gateway calling three services and *computing* a derived field — "the order is `shippable` if payment is captured and inventory is reserved" — has business logic in it. (Aggregation that merely *concatenates* responses is fine and is the BFF's job — the difference is whether the gateway *understands* the data or just glues it. We'll draw that line carefully in section 7.)
- **Stateful workflow / orchestration.** A gateway that holds a multi-step saga ("create order, then charge, then if charge fails, cancel order") is now a workflow engine. That coordination belongs in a service or an orchestrator, not the edge.
- **Data transformation that requires meaning.** Recomputing currency, applying tax rules, formatting a customer-facing message — all domain logic.

Why is this bad, concretely? Three reasons, each a real operational cost:

1. **Coupling.** Every service's domain changes now require touching the gateway. The gateway becomes the bottleneck team that every change queues behind — the exact organizational pathology microservices were supposed to dissolve.
2. **A blast radius the size of your whole platform.** The gateway is shared by *every* client and *every* service. A bug in business logic you added to the gateway can take down everything at once — there's no bulkhead, because the gateway *is* the shared resource.
3. **It becomes untestable and un-ownable.** Business logic split between the edge and the service can't be reasoned about in either place alone. No team fully owns it. Bugs hide in the seam.

> **The principle, stated as a deletion test.** For any logic in your gateway, ask: *if I deleted this and forced the relevant service to do it, would the service be able to?* If yes — if the service has the data and the domain knowledge to do it — then it belongs in the service, and the gateway is just borrowing responsibility it will eventually be punished for. The only logic that *cannot* move into a single service is genuinely cross-cutting, uniform logic — auth, limits, routing, observability — which is exactly the legitimate gateway job list. The deletion test and the cross-cutting-and-uniform test are the same test from two directions, and together they draw a hard, defensible line around the edge.

The healthy design target is a **thin, fast, dumb edge**. The gateway should be the part of your system you understand completely on a bad night at 3 a.m. Every smart thing it learns to do is a thing that can break in a way no single team can fix.

#### Worked example: how a "small" gateway feature becomes the gateway of doom

This is worth walking because it never arrives as a bad decision; it arrives as a series of reasonable ones. Watch the slide.

*Week 1.* Product wants the order list to show a "ready to ship" badge. The rule is "payment captured AND inventory reserved." The orders service has the order; payments and inventory are separate services. Someone notices the gateway already routes all three, so they add a tiny response transform: the gateway calls payments and inventory, computes `shippable = payment.captured && inventory.reserved`, and stamps it onto the orders response. Ten lines. Ships in an afternoon. *The gateway now contains a business rule and calls two extra services on the orders path.*

*Week 6.* The shipping rule changes — now it also requires "no active fraud hold." The gateway must now call the fraud service too, and the boolean grows a third term. The change is a *gateway* deploy, coordinated with the fraud team. The orders team, who own the *meaning* of "shippable," can't ship the change themselves; they file a ticket against the gateway team. *The domain rule now lives in a place the domain team doesn't own.*

*Week 14.* The fraud service has a slow morning. Because the gateway calls it *synchronously on the orders path* to compute `shippable`, the *entire order list* is now slow — for every client, web and mobile — over a rule that's cosmetic. A cosmetic badge has put a fraud-service dependency on the critical path of the most-hit endpoint on the platform. *The blast radius of a non-critical feature is now platform-wide, because it lives in the shared edge.*

*Week 20.* A new engineer tries to understand why `GET /orders` is slow. The latency isn't in the orders service (it looks fine in isolation) and isn't obviously in the gateway config. It's in business logic *hidden in the edge*, split from the service that owns the concept. Nobody can reason about it from one codebase. *The logic is now un-ownable.*

The fix at every step was the same and would have cost an afternoon: the `shippable` rule belongs in the **orders service**, which already has the order and can call payments/inventory/fraud itself (or, better, consume their events and store a denormalized flag). Apply the deletion test at Week 1 — *could the orders service compute this?* Yes. Then it belongs there, and the gateway stays a thin router. The gateway of doom is not built; it accretes, one reasonable ten-line transform at a time, until you can't tell where your platform's logic lives. The discipline is to refuse the first ten lines.

## 7. The Backend-for-Frontend (BFF) pattern

Now to the pattern in the title that most engineers underuse. The problem the BFF solves is real and you have probably felt it: **a single, one-size-fits-all API serves every client badly.** Your web app, your mobile app, and your partner integrations have genuinely different needs — different fields, different aggregations, different latency budgets, different versioning cadences — and a shared API forced to satisfy all of them ends up satisfying none of them well.

Concretely, the mobile order-detail screen needs: the order summary, the payment status, and the shipment ETA. With a shared REST API, the mobile app must make three calls — `GET /orders/8821`, `GET /payments?order_id=8821`, `GET /shipments?order_id=8821` — and stitch the results together on the device. Each response is shaped for *general* use, so it's fat: the orders response carries 30 fields when the screen shows 6; the payments response includes a full audit trail the screen never displays. The mobile app over-fetches, makes three round-trips over a high-latency cellular link, and writes brittle stitching code that breaks whenever any of the three responses changes shape.

![A before-and-after comparing a one-size-fits-all API that forces three chatty round-trips against a mobile BFF that returns one tailored payload from a single call](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-4.png)

The **Backend-for-Frontend** flips this. You build **one API per client type** — a *web BFF*, a *mobile BFF* — and each one is owned by (or built for) that client's team. The mobile BFF exposes exactly the endpoints the mobile app needs, shaped exactly how the screens consume them. `GET /mobile/order-screen/8821` returns *one* payload with the order summary, payment status, and shipment ETA already merged and trimmed to the fields the screen renders. The BFF makes the three downstream calls — server-side, over the fast internal network, in parallel — and returns the result the client actually wants. The mobile app makes one call and renders it directly.

The crucial distinction from section 6: a BFF *does* aggregate, but it aggregates *for a known client*, and the aggregation is *reshaping*, not new business logic. The order's `shippable` rule still lives in the orders service; the BFF just fetches and arranges. The BFF is allowed to know "the mobile order screen needs these three things merged" because that is *its client's* domain — a BFF is coupled to its client *on purpose*. That's the inversion: the gateway must stay client-agnostic; a BFF is gloriously client-specific. They are not the same component, and you often have both — a shared gateway in front of several BFFs.

Here is the BFF aggregation drawn as a fan-out-then-merge:

![A graph of a mobile BFF fanning one client request out to the orders payments and shipments services in parallel then merging and reshaping the results into a single tailored response](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-5.png)

The shape is the whole idea: one request in, parallel fan-out to the services the screen needs, one merged-and-trimmed payload out. Let's make it concrete with code and numbers.

#### Worked example: a mobile BFF aggregating three downstream calls into one response

The mobile app calls one endpoint:

```http
GET /mobile/order-screen/ord_8821 HTTP/1.1
Host: bff.shop.com
Authorization: Bearer <token>
```

The mobile BFF resolves it by fanning out to three services *in parallel* and merging. In Node-flavored pseudocode, the parallelism is the point — `Promise.all` issues all three calls at once, so the total time is the *slowest* call, not the sum:

```javascript
// Mobile BFF: one screen endpoint, fan-out, merge, reshape
app.get("/mobile/order-screen/:id", async (req, res) => {
  const id = req.params.id;
  const headers = { "X-User-Id": req.userId, "X-Request-Id": req.correlationId };

  // Fan out in PARALLEL — total latency = max(call), not sum
  const [order, payment, shipment] = await Promise.all([
    fetchJson(`http://orders-svc/orders/${id}`, headers),
    fetchJson(`http://payments-svc/payments?order_id=${id}`, headers),
    fetchJson(`http://shipments-svc/shipments?order_id=${id}`, headers),
  ]);

  // Reshape: return ONLY the fields the mobile screen renders
  res.json({
    order_id: order.id,
    total: order.total_cents,
    currency: order.currency,
    status: order.status,
    payment_status: payment.status,        // "captured" | "pending" | "failed"
    eta: shipment.estimated_delivery,       // ISO date the screen shows
    tracking_url: shipment.tracking_url,
  });
});
```

And the one tailored response the client renders directly:

```json
{
  "order_id": "ord_8821",
  "total": 4999,
  "currency": "USD",
  "status": "confirmed",
  "payment_status": "captured",
  "eta": "2026-06-24",
  "tracking_url": "https://track.shop.com/t/ZX19"
}
```

Now the numbers — and I'll mark what's modeled versus measured. On a high-latency mobile link, a round-trip's *network* cost dominates: assume an approximate 120 ms round-trip time (RTT) over a typical 4G connection (this varies widely — 4G RTTs commonly land in the 50–200 ms range, so treat 120 ms as a representative figure, not a benchmark). The **before** path makes three *sequential* client round-trips (the app often needs the order before it knows what to fetch next, or simply fires them in series): roughly $3 \times 120\,\text{ms} = 360\,\text{ms}$ of network latency alone, plus three fat payloads (call it ~40 KB + ~20 KB + ~15 KB) that the app must parse and stitch.

The **after** path is one client round-trip — $\approx 120\,\text{ms}$ of mobile network latency — and the three downstream calls happen server-side on the fast internal network *in parallel*, adding maybe 15–30 ms (the max of three fast internal calls, not their sum). The returned payload is trimmed to ~8 KB. So the BFF turns roughly **360 ms + ~75 KB** into roughly **120 ms + ~8 KB** — a network-latency cut on the order of $3\times$ and a payload cut of roughly $9\times$. On a slow link, payload size matters too: ~75 KB versus ~8 KB at, say, 1 Mbps effective throughput is the difference between ~600 ms and ~64 ms of *transfer* time on top of the RTT. The exact figures depend on your network and payloads — but the *shape* of the win (fewer round-trips, smaller bytes, no client stitching) is structural and reliable, and it's why mobile teams reach for BFFs.

There's a second, quieter win that's easy to miss: **independent evolution**. With a shared API, a field the mobile screen needs to change is a change to an API that web and partners also consume — so it's a coordinated, slow, backward-compatible change negotiated across consumers. With a per-client BFF, the mobile team changes *its own* BFF on *its own* schedule without touching anyone else's contract. The general principle of safe change still applies *below* the BFF (the downstream services are still shared and must evolve compatibly — see [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change)), but the BFF gives each client a contract surface it controls. That decoupling — each client team owning its own edge contract — is often the real reason a mature org adopts BFFs, beyond the latency win.

The cost side, stated honestly: a BFF is **another deployable service to own, monitor, and secure**, and it's an extra hop. You're trading operational surface for client experience. There's also a duplication tax — three BFFs may each implement a similar "fetch order + payment + shipment" aggregation, and that logic can drift. Keep the *shared* bits (a client for each downstream service, common auth) in a library the BFFs reuse, and let only the *reshaping* differ per client. That trade is worth it when the client's needs genuinely diverge from a shared API and the client is latency- or bandwidth-sensitive (mobile, mostly). It's *not* worth it when you have one client, or when the shared API already returns close to what each client needs — then a BFF is just a layer that adds a hop and a deploy for no payoff. (Section 9 makes the when/when-not call sharp.)

## 8. Gateway vs BFF vs service mesh: drawing the lines

Three components keep getting confused because they all sit "near the network." They solve different problems, and a mature platform often runs all three. The cleanest way to separate them is by the *direction* of the traffic they govern and *whose* problem they solve.

- **API gateway** — governs **north-south** traffic (clients ↔ your edge). It's the single, *client-agnostic* front door: TLS, authN, rate limiting, routing, observability. One per platform.
- **BFF** — also north-south, but **per client**. It's *client-specific* aggregation and reshaping. You have several (web BFF, mobile BFF), each owned for its client. A BFF often sits *behind* the shared gateway.
- **Service mesh** — governs **east-west** traffic (service ↔ service). It's infrastructure (sidecar proxies like Envoy injected next to each service) that handles service-to-service mTLS, retries, timeouts, and traffic policy *between* your internal services. The application code doesn't even see it. (The mesh's mTLS and zero-trust internals are the [service-to-service security post](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) in the microservices series — link out, don't re-derive.)

![A matrix comparing the API gateway BFF and service mesh across traffic direction what they own whether they are per client and their authentication role](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-6.png)

The matrix makes the division of labor concrete. The gateway and BFF are both north-south but split on *per-client* (gateway: no; BFF: yes). The mesh is the odd one out — east-west, infrastructure, invisible to clients — and it's where service-to-service mTLS lives, *not* the gateway. A request's full journey shows all three cooperating: it arrives at the **gateway** (authN, limit, route) → hits a **BFF** (aggregate for this client) → the BFF calls downstream services where the **mesh** secures each hop with mTLS and applies retries/timeouts. The wider microservices treatment of how these fit a fleet is in [the gateway and BFF in a microservices fleet](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend); this post owns the *API-design* view — what the caller gets to assume at each layer.

Laid out as a side-by-side, the differences are sharp enough that the "which do I need?" question usually answers itself:

| Dimension | API gateway | BFF | Service mesh |
| --- | --- | --- | --- |
| Traffic direction | North-south | North-south | East-west |
| Per client? | No — one front door | Yes — one per client | No — infrastructure |
| Primary job | Edge cross-cutting concerns | Aggregate + reshape for a client | Service-to-service policy |
| Auth role | Verifies the token | Trusts the gateway's identity | mTLS service identity |
| Knows the domain? | No — must stay generic | Yes — its client's screens | No — pure transport policy |
| Visible to clients? | Yes | Yes | No — invisible sidecars |
| How many you run | One per platform | Several (web, mobile, partner) | One mesh, many sidecars |
| Failure if missing | Concerns re-implemented per service | Clients over-fetch and stitch | Each service rolls its own mTLS/retry |

The bottom row is the one that justifies each component's existence — it's the *consequence* of not having it. Skip the gateway and every service re-implements auth (inconsistently). Skip the BFF and every client over-fetches and writes brittle stitching code. Skip the mesh and every service rolls its own service-to-service security and retry logic (inconsistently, again). Each component exists because a specific concern, left to each service, gets done many times and gets done wrong.

Here is the decision boiled to a small tree you can run in your head:

![A decision tree starting from traffic direction that routes north-south single-client traffic to a shared gateway north-south multi-client traffic to a BFF and east-west traffic to a service mesh](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-8.png)

Start at the top: is this **north-south** (a client is calling) or **east-west** (a service is calling another service)? East-west → that's a mesh concern (mTLS, retries between services), not a gateway. North-south → do your clients need *the same shape* or *different shapes*? Same shape, one client → a shared gateway is enough. Different shapes (web vs mobile vs partner) → add a BFF per client behind the gateway. That single tree resolves most "do we need X?" arguments before they start.

A word on **GraphQL at the edge**, because it's often pitched as "a BFF you don't have to write." A GraphQL gateway lets each client request exactly the fields it wants in one query, which solves the over-fetching and round-trip problem the BFF solves — for read-heavy, field-divergent clients it can be a genuinely good fit. The cost is that you inherit GraphQL's hard parts at your edge: the N+1 resolver trap, query-complexity attacks (a malicious deeply-nested query that explodes into millions of resolver calls), and the loss of simple HTTP caching (everything is a `POST` to one endpoint). The trade-offs are exactly the subject of the [paradigm-choice post](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); the edge-design point is that GraphQL is *one way* to implement BFF-style flexibility, with its own operational tax — not a free lunch.

## 9. Failure modes: the gateway as a single point of failure

The uncomfortable truth about the gateway is structural: **everything goes through it, so when it fails, everything fails.** It is, by construction, a single point of failure (SPOF) for north-south traffic. You don't get to wish that away; you design around it. The intro incident was one face of this — a slow *upstream* taking down the edge — but the gateway itself failing is the bigger risk because its blast radius is the whole platform.

The defenses, roughly in order of importance:

**1. Run it highly available (HA).** Never one gateway instance. Run several behind a layer-4 load balancer (or your cloud's managed equivalent), across multiple availability zones, so a single instance or zone dying doesn't take you down. The gateway must be *stateless* (or share state in a fast store like Redis for rate-limit counters) so any instance can serve any request and you can scale horizontally.

**2. Per-route timeouts — always.** The lesson from the intro, formalized: every upstream call gets a bounded timeout, tuned per route (payments might be 2 s, a slow report-generation endpoint 10 s). An unbounded timeout is how a slow upstream becomes an edge-wide outage. The math from section 2 is the justification: worker threads are finite, and a hung upstream holds them until the timeout fires.

**3. Circuit breaking.** When an upstream fails repeatedly (say, 5 failures in a row), the gateway *opens the circuit* for that route — it stops calling the upstream and immediately returns `503 Service Unavailable` with a `Retry-After`, for a cooldown period (say 30 s). This does two things: it stops piling load onto a service that's already struggling (giving it room to recover), and it fails *fast* instead of making every caller wait for the timeout. After the cooldown, the breaker goes *half-open* — it lets one probe request through; if that succeeds, it closes and resumes normal traffic; if it fails, it re-opens. This is the standard breaker state machine, and it's what turns a degrading dependency into a contained, self-healing failure.

![A timeline of a gateway protecting a flaky payments upstream with a two-second timeout retries that fail fast a tripped circuit breaker a fast 503 and a half-open recovery probe](/imgs/blogs/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern-7.png)

The timeline reads as the breaker's life: a request comes in and starts the timer; the call to payments hangs and hits the 2 s timeout; failures accumulate until the breaker trips open for 30 s; during that window every caller gets an immediate `503 · Retry-After: 30` instead of a 2 s stall; then a half-open probe tests recovery and, on success, closes the breaker. The key property is that **the slow dependency is converted into a fast, bounded failure** — callers learn quickly that payments is down and can degrade gracefully (the mobile order screen can render without the payment status rather than spinning).

**4. Bulkheads.** Isolate routes so one upstream's failure can't consume the worker pool another route needs. Give payments its own connection pool and worker budget separate from orders; then a payments meltdown can't starve orders. (This is the direct fix for the intro incident — the failure was that there *were* no bulkheads.)

**5. Sensible fallbacks.** For non-critical, cacheable data, the gateway can serve a slightly-stale cached response when the upstream is down rather than failing. "Show the last-known shipment ETA with a staleness marker" beats "show an error." This is a product decision, not a default — apply it only where stale is better than nothing, and *never* for payment confirmation or anything where stale is dangerous.

> **The stress test.** Walk the gateway through its worst days. *What if an upstream is slow?* Per-route timeout converts it to a fast 504/503; the breaker trips so you stop hammering it. *What if an upstream is down entirely?* Breaker is open, callers get an immediate 503, bulkheads keep other routes alive, fallbacks serve stale where safe. *What if a client floods you?* The per-identity rate limit caps them with 429s; the flood never reaches the service. *What if the gateway itself loses an instance?* HA across zones means the LB routes around it; stateless instances mean no session is lost. *What if your token issuer (the IdP) is down so you can't verify tokens?* This is the nasty one — fail *closed* (reject, since you can't prove identity) for sensitive routes, but cache the issuer's public keys (JWKS) aggressively so a brief IdP outage doesn't break verification, since JWT signatures can be checked offline with a cached key. Design the answer to each of these *before* the incident, because during the incident the gateway is the one thing that's definitely on the critical path.

## 10. Case studies and the product landscape

A few accurate, named references to ground all this, plus the products you'll actually choose between.

**Netflix and the origin of the BFF.** The Backend-for-Frontend pattern was popularized by engineers at SoundCloud and notably described by Phil Calçado and Sam Newman, but Netflix is the canonical large-scale example of *why* it exists: Netflix runs on an enormous range of devices — TVs, game consoles, phones, browsers — each with wildly different screen sizes, network conditions, and capabilities. Forcing all of them through one generic API meant every device over-fetched and every API change risked breaking a device class. Netflix's answer was device-specific edge adaptation (their "edge API" / device-team-owned adaptation layers), which is the BFF idea at scale: let the team that owns the client own the API shape for that client. The lesson generalizes: **when clients diverge enough, give each one its own backend.**

**Kong** is an open-source gateway built on NGINX/OpenResty with a plugin architecture — auth, rate limiting, transformations, and observability are plugins you enable per route, much like the YAML configs above. It's a popular self-hosted choice when you want control and a plugin ecosystem.

**Apigee** (now part of Google Cloud) is a full **API management** platform — gateway plus developer portal, monetization, analytics, and lifecycle governance. The distinction matters: a *gateway* enforces edge policy; *API management* wraps that with the business of running an API program (publishing, keys for partners, usage analytics, billing). You reach for Apigee-class tooling when the API is a *product you sell*, not just an internal edge.

**AWS API Gateway** is a managed, serverless gateway that pairs naturally with Lambda — it handles routing, throttling (its rate-limit feature), authorizers (custom or Cognito/JWT), and request/response mapping templates, and you pay per request. The appeal is zero servers to run; the trade-off is provider lock-in and the per-request cost model at very high volume.

**Envoy** is the high-performance C++ proxy that powers a great deal of modern edge and mesh infrastructure. It's the data plane behind **Contour** (a Kubernetes ingress/gateway) and behind service meshes like Istio. Envoy is where "the gateway and the mesh share a data plane" becomes literal — the same proxy technology terminates north-south traffic at the edge and secures east-west traffic between services. **NGINX** remains the workhorse reverse proxy underneath many of these and is itself a perfectly capable gateway for simpler needs.

The honest summary: for an internal edge with a plugin ecosystem, Kong or an Envoy-based gateway; for serverless on AWS, API Gateway; for selling an API as a product, an API-management platform like Apigee; for the simplest cases, NGINX. None of them changes the *design* questions in this post — they're just where you *configure* the answers.

## 11. When to reach for a gateway and a BFF (and when not to)

Decisive recommendations, because every choice here is a trade-off and the wrong default is expensive.

**Use a gateway when** you have more than a couple of services behind a public or shared edge and the cross-cutting concerns (TLS, auth, rate limiting, observability) are real. At that point, *not* having a gateway means re-implementing those concerns per service and getting them inconsistent — that inconsistency is itself a security and reliability risk. A single front door is the right default for any non-trivial multi-service API.

**Don't bother with a heavyweight gateway when** you have a single service and a single client — a small app's own framework (a bit of auth middleware, a rate-limit decorator) is simpler than operating a separate gateway product, and the gateway would just be a hop that adds latency and an operational burden for no payoff. Add the gateway when you add the *second* service or the *first* external client, not before.

**Never put business logic, object-level authZ, or stateful workflows in the gateway** — that's the gateway of doom from section 6. The deletion test is your guard: if a service could do it, the service should.

**Build a BFF when** you have multiple, genuinely divergent clients (the classic web + mobile split), *and* at least one of them is latency- or bandwidth-sensitive (mobile), *and* the shared API forces them to over-fetch or make many round-trips. That's the precise condition where the BFF's cost (an extra service, an extra hop) is repaid in client experience — fewer round-trips, smaller payloads, no client-side stitching, and the freedom to evolve each client's API on its own cadence.

**Don't build a BFF when** you have one client, or when your clients' needs are similar enough that a shared API (or a few well-designed query params and sparse fieldsets) already serves them — then a BFF is a layer of indirection that adds a hop, a deploy, and an on-call rotation for no real gain. A BFF per client is also a *team-shaped* decision: it works best when the client team owns its BFF. If no team will own it, it becomes an orphaned middle layer that decays. And **don't let a BFF become a mini-monolith** — if your "mobile BFF" starts holding the order-shipping business rule, you've recreated the gateway-of-doom one layer down.

**Reach for the service mesh (not the gateway) when** the problem is *east-west*: you have many services calling each other and you want consistent service-to-service mTLS, retries, and timeouts without baking them into every service. The gateway is the wrong tool for that; it doesn't see east-west traffic. Use the right layer for the direction.

## 12. Key takeaways

- **A gateway is a single front door for cross-cutting, uniform concerns** — TLS, authN, coarse scope, rate limiting, routing, observability — so each service stops re-implementing them. The test for "does it belong at the edge?" is: *cross-cutting AND uniform (no domain knowledge needed)?*
- **Authenticate at the edge; authorize objects in the service.** The gateway verifies the token and a coarse scope; *may you see this specific order?* requires domain knowledge and stays in the owning service. A perfect gateway cannot save you from a missing object-level check.
- **Rate-limit by authenticated identity, not IP**, which is why auth runs before the limiter. Reject with `429` + `Retry-After` and a `problem+json` body so well-behaved clients back off cleanly.
- **Per-route timeouts and circuit breakers are not optional.** They convert a slow or dead upstream into a fast, bounded failure instead of an edge-wide cascade. The gateway is a SPOF — run it HA, stateless, across zones, with bulkheads.
- **Never put business logic, object-level authZ, or workflows in the gateway.** That's the gateway of doom — apply the deletion test: if a service could do it, the service should.
- **A BFF is one tailored API per client**, aggregating and reshaping downstream services for *that* client's screens. It trades an extra service and hop for fewer round-trips, smaller payloads, and per-client evolution — worth it for divergent, latency-sensitive clients, wasteful for a single one.
- **Gateway, BFF, and mesh are different tools by traffic direction**: gateway = north-south, client-agnostic; BFF = north-south, per-client; mesh = east-west, service-to-service mTLS. A mature platform runs all three.
- **The edge is part of the contract.** What the caller gets to assume — that a request is authenticated, limited, and will fail fast under partial failure — is decided here, and it holds across every version you ship.

## Further reading

- **RFC 9110 — HTTP Semantics**: the status codes (401/403/429/503), methods, and the safe/idempotent definitions the gateway must respect when routing and retrying.
- **RFC 9457 — Problem Details for HTTP APIs** (`application/problem+json`): the error envelope your `429`/`503` responses should use.
- **RFC 7519 — JSON Web Token (JWT)** and **RFC 6749 — OAuth 2.0**: the token formats and flows the gateway verifies at the edge.
- **OWASP API Security Top 10**: especially "Broken Object Level Authorization" (API1) — the canonical proof that object-level authZ must live in the service, not the gateway.
- **Sam Newman, *Building Microservices*** and the Netflix tech blog on edge/device adaptation: the origin and rationale of the BFF pattern.
- **Envoy, Kong, AWS API Gateway, and Apigee documentation**: the concrete configuration surfaces for the policies in this post.
- Within this series: start at the intro hub [what is an API — the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); pair this with [rate limiting, quotas, and abuse protection](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection), [authorization — scopes, roles, and resource-level permissions](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions), [caching with ETags and conditional requests](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation), and [choosing a paradigm — REST vs gRPC vs GraphQL by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); finish with the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- For the fleet view: [the API gateway and Backend-for-Frontend in microservices](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) and [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing).

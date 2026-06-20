---
title: "Authentication: API Keys, Sessions, JWT, and mTLS"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The deep dive on proving who is calling your API — API keys with prefixes and hashing-at-rest, server sessions and the CSRF trap, JWTs from header to signature with the alg-confusion pitfalls, refresh-token rotation, and mTLS for zero-trust service identity, with the wire formats, the trade-offs, and a clear choice per client type."
tags:
  [
    "api-design",
    "api",
    "authentication",
    "jwt",
    "mtls",
    "api-keys",
    "security",
    "oauth",
    "http",
    "rest",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-1.png"
---

A partner integration team emailed us at 2 a.m. Their nightly settlement job had stopped working — every call to `POST /v1/payments` was coming back `401 Unauthorized`. Nothing in our code had changed. Nothing in theirs had changed. What *had* changed was that, three weeks earlier, one of their engineers had pasted a config file with our API key into a public GitHub gist while debugging a CI failure. A scanner found it in minutes. By the time we noticed the anomalous traffic and revoked the key, someone had used it to enumerate every order in their account. The key had no expiry. It had been minted two years prior, scoped to *everything*, and copied into four different deployment pipelines. Rotating it meant a coordinated change across all four, in the middle of the night, while their settlement window was open.

That is what an authentication failure looks like in production: not a clever cryptographic break, but a long-lived secret that leaked, sat valid for weeks, and could not be revoked without a fire drill. Authentication is the part of your API where mistakes are *catastrophic* rather than merely annoying. A badly designed pagination cursor skips a few rows. A badly designed auth scheme hands your customers' data to whoever finds the credential first. This post is about getting that boundary right — about answering, for every single request, the question **"who is calling?"** before any handler runs.

That question — *who* — is **authentication** (authN). It is distinct from *what may you do* — **authorization** (authZ), which is the [next post in this series](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions). Authentication establishes an identity; authorization decides what that identity is allowed to touch. They are often jammed into one middleware and one mental bucket, but they fail differently and they are designed differently. This post owns the first one. We will define both clearly, then go deep on the four schemes you will actually choose between — **API keys, session cookies, JWTs (bearer tokens), and mTLS** — using our running **Payments & Orders** API as the spine: a partner backend authenticating with a scoped API key, a mobile app carrying a short-lived JWT plus a refresh token, and internal services proving identity to each other with mutual TLS.

![a layered diagram showing a request passing through TLS then credential extraction then identity verification, branching to a 401 rejection or to an attached principal that reaches the handler](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-1.png)

By the end you will be able to: pick the right scheme for each kind of caller; read and validate a JWT without trusting it blindly; store and transport tokens so they do not leak; design API keys with prefixes, hashing-at-rest, scoping, and rotation; and reason about the one trade-off that governs all of this — **stateful (revocable but it does not scale) versus stateless (it scales but you cannot revoke before expiry)**. We will keep coming back to the series' spine: *an API is a contract you design for a caller you will never meet*. Authentication is the clause of that contract that says **prove you are who you claim, on every request, or I will not talk to you.** If you want the wider frame first, start at the [intro hub on the API contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); to see how this clause fits the whole review, jump to the [capstone playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

## 1. AuthN versus authZ: identity before permission

Let me define both terms precisely, because conflating them is the root of more security bugs than any cryptographic subtlety.

**Authentication (authN)** is the act of proving an *identity*. The output of authentication is a **principal** — a verified statement of who is making this request. "This request is from user `usr_9f2a`." "This request is from the partner application `app_payments_recon`." "This request is from the internal service `orders-service`." Authentication does not say what that principal may do. It only says, with some level of confidence, *who it is*.

**Authorization (authZ)** is the act of deciding whether a known principal is *allowed* to perform a *specific action on a specific resource*. "May `usr_9f2a` read order `ord_55c1`?" "May `app_payments_recon` issue a refund?" Authorization takes the principal that authentication produced and checks it against scopes, roles, ownership, and policy.

The order matters and it is strict: **you authenticate first, then you authorize.** You cannot decide what a caller may do until you know who the caller is. In HTTP terms this maps cleanly onto status codes, and the codes are worth memorizing because clients branch on them:

- **`401 Unauthorized`** means *authentication* failed or is missing. The name is a historical misnomer — it really means "unauthenticated." The correct response carries a `WWW-Authenticate` header telling the client which scheme to use. A `401` says: "I do not know who you are; present a valid credential and try again."
- **`403 Forbidden`** means *authorization* failed. The server knows who you are — authentication succeeded — but this identity is not allowed to do this. A `403` says: "I know exactly who you are, and the answer is still no." Retrying with the same credential will not help.

Getting these two confused leaks information. Returning `403` when a resource does not exist tells an attacker the resource *does* exist (they just can't see it); many APIs deliberately return `404` instead to avoid that. Returning `401` when the real problem is a missing scope sends the client into a pointless re-authentication loop. The status code is part of the contract; it tells the caller *how to recover*, and recovering from "I don't know who you are" is a completely different action than recovering from "you're not allowed."

This post is entirely about producing that **principal** correctly and cheaply, on every request, for every client type. Everything downstream — scopes, roles, rate limits keyed by tenant, audit logs — depends on it being trustworthy. If authentication is sloppy, authorization is decoration: it is checking the permissions of an identity you cannot actually trust.

### The boundary principle

Here is the architectural rule that the rest of this post leans on. **Authentication belongs at the boundary, before routing, and it must be uniform.** A request arrives, the edge (a gateway, a middleware, a filter) extracts the credential, verifies it, and attaches a principal to the request context — and only *then* does the request reach a handler. The handler should never see an unauthenticated request; by the time application code runs, the question "who is calling?" is already answered and the answer is sitting in the request context.

Why at the boundary and not in each handler? Because authentication that is implemented per-handler is authentication that will be *forgotten* in some handler. The one endpoint where a developer pasted a route and forgot the `@require_auth` decorator is the one an attacker finds. Centralizing it at the edge means a new endpoint is authenticated *by default* — the developer has to actively opt a route *out* of auth (a health check, say), which is a visible, reviewable decision, rather than silently forgetting to opt *in*. This is the same robustness logic that runs through the whole series: make the safe thing the default and the unsafe thing loud.

## 2. The `Authorization` header: Bearer and Basic

Before we get to schemes, we need the envelope they travel in. HTTP defines a standard place to put credentials: the **`Authorization` request header** (RFC 9110, which subsumed the older RFC 7235). Its value is a *scheme name* followed by the credential. Two schemes matter for APIs.

**`Basic`** carries a username and password, base64-encoded. The format is `Authorization: Basic <base64(username:password)>`. Note that base64 is *encoding, not encryption* — it is trivially reversible. `Basic` only has any security at all over TLS, where the whole header is encrypted in transit. It is simple and widely supported, which is why API keys are often delivered through it (key as username, empty password). Stripe, famously, accepts its secret key as the `Basic` username so you can authenticate with nothing more than `curl -u sk_live_YOUR_KEY_HERE:`.

**`Bearer`** carries an opaque or structured *token*: `Authorization: Bearer <access_token>`. The word "bearer" is the entire security model and you must internalize it: a bearer token is like cash. **Whoever bears it, may use it.** There is no second factor, no proof that the holder is the legitimate owner. If the token leaks, the finder *is* you, as far as the server is concerned, until the token expires. RFC 6750 defines the Bearer scheme for OAuth 2.0, and it is the scheme JWTs ride on. This is exactly why the rest of this post obsesses over short lifetimes, careful storage, and TLS-only transport: a bearer token's security reduces almost entirely to *how hard it is to steal and how quickly it stops working if it is stolen.*

A few rules about the header that are part of the contract:

- **Always over TLS.** A credential sent over plaintext HTTP is a credential you have published. There is no acceptable exception for production.
- **Never in the URL.** A token in a query string (`?access_token=...`) ends up in server access logs, browser history, the `Referer` header sent to third parties, and proxy logs. This is one of the most common real-world leaks. The credential goes in the *header*, where it is not logged by default.
- **Never log the header.** Your request-logging middleware must redact `Authorization`. A token in your own logs is a token your on-call engineer, your log-aggregation vendor, and anyone with read access to logs now holds.

When authentication is missing or fails, respond `401` with a `WWW-Authenticate` header naming the scheme so the client knows what to send:

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="payments-api", error="invalid_token", error_description="The access token expired"
Content-Type: application/problem+json

{
  "type": "https://api.shop.example/problems/invalid-token",
  "title": "Authentication failed",
  "status": 401,
  "detail": "The access token expired at 2026-06-20T09:15:00Z. Refresh and retry."
}
```

That body is an RFC 9457 `problem+json` document — the machine-readable error envelope this series uses everywhere (see [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract)). Notice it does *not* echo the token back, and the `detail` tells the client *how to recover* (refresh and retry) without leaking anything sensitive.

## 3. API keys: the simple shared secret

An **API key** is the simplest credential: a single, long, random string that the server has seen before and recognizes. The client sends it, the server looks it up, and if it matches a known key the request is authenticated as whatever that key represents. There is no handshake, no expiry negotiation, no public-key math. That simplicity is exactly why API keys are everywhere — and exactly why they are dangerous if you treat them carelessly.

API keys are best for **server-to-server** traffic and for **identifying a calling application** (which app, which integration, which partner) rather than an individual end user. A nightly reconciliation job, a partner's backend pulling order data, a CI pipeline deploying a webhook config — these are machine callers with no browser, no user session, and a stable trust relationship. An API key fits them well: the partner stores the secret in their server's environment, sends it on every request, and you know which partner is calling.

Here is the key weakness, stated plainly: **a naive API key is a long-lived, single-factor, plaintext-equivalent secret with no built-in expiry.** Every property in that sentence is a risk. *Long-lived*: it works for years, so a leak from any point in those years is exploitable. *Single-factor*: possession is the whole game; there is no second check. *Plaintext-equivalent*: it is sent on every request, so it sits in transit, in client configs, in CI variables, in the partner's deployment manifests. *No expiry*: nothing makes it stop working on its own.

You cannot remove all of those weaknesses — an API key is a shared secret by definition — but you can mitigate every one of them with disciplined design. Here is the full checklist.

### Designing API keys that do not ruin your night

**Prefix the key by type.** Stripe popularized this and it is now best practice: encode the key's *kind* in a human-readable prefix. `sk_live_` for a secret key in live mode, `sk_test_` in test mode, `pk_` for a publishable key, `rk_` for a restricted key. The benefit is twofold. First, a developer who pastes `sk_live_...` into a frontend bundle can be *caught* — secret scanners (GitHub's, your own pre-commit hooks) match on the prefix and block the commit before it leaks. Second, when a key shows up in a log or a support ticket, the prefix tells you instantly what you are dealing with. Our Payments API mints keys like `pk_orders_live_...` (publishable, orders scope) and `sk_recon_live_...` (secret, reconciliation).

**Show only a last-4 (or last-6) after creation.** When a key is created, show the full value *exactly once* — the user copies it into their secret manager and you never display it again. In the dashboard, show only a masked form like `sk_recon_live_••••••••a3f9`. This last-4 is enough for a human to recognize "yes, that's the key in our prod config" without the dashboard itself becoming a place where the full secret can be read or leaked.

**Hash keys at rest — never store the raw key.** This is the single most important rule and the most often violated. Treat an API key exactly like a password: when it is created, store a *hash* of it, not the key itself. Then your database — and any backup, any dump, any leaked snapshot — contains hashes, not usable credentials. When a request arrives, you hash the presented key and compare to the stored hash. A leaked database of hashed keys is far less catastrophic than one of raw keys. (Use a fast cryptographic hash like SHA-256 here rather than a slow password hash like bcrypt: API keys are high-entropy random strings, not low-entropy human passwords, so they do not need the brute-force resistance that slows password hashing — and you are hashing on *every request*, so speed matters. The lookup problem is solved by indexing on the prefix or a short non-secret key id, then hashing to confirm.)

**Scope keys to least privilege.** A key should grant the *narrowest* set of permissions the integration needs. Stripe's "restricted keys" let you create a key that can read charges but not issue refunds. Our reconciliation partner gets a key scoped to `orders:read payments:read` and *nothing else*. When that key leaked in the opening story, the damage was bounded by its scope: the attacker could enumerate orders, which was bad, but they could not *issue refunds or move money*, which would have been catastrophic. Scoping is the difference between an incident and a disaster.

**Rotate keys, and make rotation a routine, not an emergency.** Support having *two* active keys at once so a partner can roll from the old to the new with zero downtime: mint the new key, deploy it everywhere, confirm traffic has moved, then revoke the old one. If rotation requires downtime, it will not happen on schedule, and a key that is never rotated is a key whose entire history of leaks is still live. Set an internal expectation (90 days, say) and tooling that makes it a one-click operation.

#### Worked example: an API key request and how the server validates it

Here is the partner's reconciliation job calling our Payments API with its scoped secret key. The key is delivered as the `Basic` username (Stripe-style), so it never appears in a URL:

```bash
curl https://api.shop.example/v1/payments?status=settled \
  -u sk_recon_live_YOUR_KEY_HERE: \
  -H "Accept: application/json"
```

The empty value after the colon is the (empty) password — `Basic` requires a `user:pass` pair, and we only use the user half. On the wire, `curl` turns that into:

```http
GET /v1/payments?status=settled HTTP/1.1
Host: api.shop.example
Authorization: Basic c2tfcmVjb25fbGl2ZV9ZT1VSX0tFWV9IRVJFOg==
Accept: application/json
```

Now the server side. The key is `sk_recon_live_<id>_<secret>`, where `<id>` is a short non-secret identifier we can index on and `<secret>` is the high-entropy part we hash. Validation looks like this:

```python
import hashlib
import hmac

def authenticate_api_key(presented: str) -> Principal | None:
    # presented = "sk_recon_live_a1b2c3_<the long secret part>"
    if not presented.startswith(("sk_", "pk_", "rk_")):
        return None  # not even shaped like our key -> 401

    try:
        prefix, mode, key_id, secret = parse_key(presented)
    except ValueError:
        return None

    record = db.api_keys.find_one(key_id=key_id, revoked=False)
    if record is None:
        return None  # unknown or revoked -> 401

    # hash the presented secret and compare in constant time
    presented_hash = hashlib.sha256(secret.encode()).hexdigest()
    if not hmac.compare_digest(presented_hash, record.secret_hash):
        return None  # wrong secret -> 401

    db.api_keys.touch_last_used(key_id)  # for the dashboard + anomaly detection
    return Principal(
        kind="application",
        app_id=record.app_id,
        scopes=record.scopes,          # e.g. {"orders:read", "payments:read"}
        key_id=key_id,                 # for audit logging, never the raw key
    )
```

Three details earn their place. We index on the *non-secret* `key_id` so the database lookup is fast and does not require hashing every row. We compare hashes with `hmac.compare_digest`, a **constant-time** comparison — a normal `==` short-circuits on the first differing byte, which leaks timing information an attacker can use to recover a secret byte by byte. And the returned `Principal` carries `scopes`, which the *authorization* layer (the next post) will check; authentication's job ends at "this is application `app_payments_recon`."

The opening incident, retold through this design: the partner's key leaked. Because the key was *scoped*, the blast radius was bounded to reads. Because we stored a *hash*, the leak came from the partner's config, not our database. Because we supported *two active keys*, we rotated without downtime once we noticed. Because the key had a *prefix*, our own anomaly detection flagged a `sk_recon_live_` key suddenly making calls from a new IP range in a new country. Every mitigation pulled its weight. The one thing we could not do was make the key *expire on its own* before the leak — and that limitation is precisely what pushes us toward tokens for any caller that can handle them.

## 4. Session cookies: stateful, browser-first

The next scheme is the oldest one on the web, and it is still the right answer for one specific caller: a browser talking to a first-party server-rendered or same-origin application.

A **session** works like this. The user logs in (username and password, an OAuth dance, whatever). The server creates a **session record** in its own store — a row in a database, an entry in Redis — containing the user id, when it was created, when it expires, and so on. The server then hands the browser a **session id**: an opaque, random, *meaningless* string, set in a cookie. On every subsequent request the browser automatically attaches that cookie, the server looks up the session record by id, and if it exists and is valid the request is authenticated as that user.

The crucial property is that the session id is **opaque and meaningless** — it carries no information, it is just a key into the server's session table. All the state lives *on the server*. This is what makes sessions **stateful**, and statefulness is both the feature and the limitation. The feature: revocation is trivial and instant. To log a user out everywhere, you delete the session record; the next request with that cookie finds nothing and gets a `401`. There is no waiting for a token to expire. The limitation: every authenticated request requires a lookup in shared session state, and that shared state has to be reachable by every server that handles the user's requests. That works beautifully for one application backed by one session store. It does *not* compose across services — a downstream `orders-service` cannot validate a cookie without calling back to the session store, which couples your whole fleet to it. **Sessions are for first-party browser apps, not for service-to-service calls.**

The cookie itself must be configured defensively. The flags are non-negotiable:

```http
Set-Cookie: session=8f14e45fceea167a... ; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=86400
```

- **`HttpOnly`** — JavaScript cannot read the cookie via `document.cookie`. This is your defense against cross-site scripting (XSS): even if an attacker injects script into your page, it cannot exfiltrate the session id. This is *the* reason sessions can be safer than tokens stored in JavaScript-readable storage, a point we will return to with JWTs.
- **`Secure`** — the cookie is sent only over HTTPS, never plaintext.
- **`SameSite`** — controls whether the cookie is attached to *cross-site* requests, which is your primary defense against CSRF.

### The CSRF trap and SameSite

Because the browser attaches the session cookie *automatically* on every request to your domain, a malicious third-party site can trick the browser into making a request to your API *with the user's cookie attached* — without the user intending it. This is **Cross-Site Request Forgery (CSRF)**. The classic example: the user is logged into our Payments dashboard, then visits `evil.example`, which contains a hidden form that auto-submits `POST https://api.shop.example/v1/refunds`. The browser dutifully attaches the session cookie, and unless we defend against it, our server sees a perfectly valid, authenticated refund request that the user never meant to make.

The reason CSRF is a *cookie* problem and not a *token* problem is exactly the automatic-attachment behavior: a `Bearer` token in an `Authorization` header is *not* sent automatically by the browser, so an attacker's site cannot make the browser include it. Cookies are convenient precisely because they are automatic, and they are vulnerable precisely because they are automatic.

The modern defense is **`SameSite`**:

- `SameSite=Strict` — the cookie is *never* sent on any cross-site request. Maximum safety, but it breaks legitimate cross-site navigation (a link from an email to your app arrives with no cookie, so the user looks logged out).
- `SameSite=Lax` — the cookie is sent on top-level *navigations* (clicking a link) but not on cross-site *subrequests* like a hidden form POST or an `<img>` tag. This blocks the classic CSRF POST while keeping links working. It is the sane default and is now the browser default when no flag is set.
- `SameSite=None` — the cookie is sent on all cross-site requests (and `Secure` is then mandatory). You need this only for genuine cross-site embedding, and it re-opens the CSRF door, so it must be paired with an explicit anti-CSRF token.

For state-changing requests you should *also* layer a **synchronizer token** or **double-submit cookie** pattern: the server issues a CSRF token that the legitimate frontend echoes in a header (`X-CSRF-Token`), which a cross-site attacker cannot read or forge. `SameSite=Lax` plus a CSRF token on mutating endpoints is the belt-and-suspenders standard.

The honest summary: sessions are excellent for the browser case — instant revocation, `HttpOnly` protection against XSS exfiltration — but they carry the CSRF tax and they do not travel across services. Keep them for first-party browser apps and reach for a token everywhere else.

![a matrix comparing API key, session cookie, JWT, and mTLS across whether they are stateful, whether they are revocable, what they are best for, and their main risk](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-2.png)

That matrix is the decision in one frame, and the column that drives everything is **Revocable**. Stateful schemes (sessions, and API keys whose row you can delete) revoke instantly. Stateless JWTs do not — and understanding *why* is the heart of the next two sections.

## 5. The stateful versus stateless trade-off

Step back and notice the axis that organizes all four schemes. It is **where the source of truth lives.**

In a **stateful** scheme, the credential is a *reference* — a pointer into server-side state. A session id points at a session record; an API key (with its database row) points at a key record. To validate, the server *looks up* the referenced state. To revoke, the server *deletes* the state. The cost is the lookup on every request and the need for that state to be reachable; the benefit is total, instant control — the moment you delete the record, the credential is dead.

In a **stateless** scheme, the credential is *self-contained* — it carries its own claims and a signature proving those claims came from you. A JWT *is* the user id, the expiry, the scopes, all baked in and signed. To validate, the server just *verifies the signature* and checks the expiry — no database lookup, no shared state, no network call. This is the killer property: any service holding the verification key can authenticate a request *locally*, which is exactly what you need across a fleet of microservices. The cost is the mirror image of the benefit: because the credential is self-contained and you do *not* consult shared state on each request, **you cannot revoke it before it expires.** You signed a statement saying "this is user `usr_9f2a`, valid until 09:15." Every service will believe that statement until 09:15, even if you fired `usr_9f2a` at 09:01, because nothing checks back with you.

This is *the* trade-off of API authentication, and it has a clean resolution that the whole industry has converged on:

> **Keep stateless tokens short-lived, and pair them with a stateful refresh token for revocation.** The access token is stateless and fast and expires in minutes, so the "cannot revoke" window is small. The refresh token is stateful — a row you can delete — so you *can* cut someone off: you revoke the refresh token, and within one short access-token lifetime they can no longer get a new one.

That sentence is the most important one in this post. It dissolves the dilemma: you get the scaling of stateless verification for the common path (every API call), and you get the revocability of stateful state for the rare path (logout, compromise, key rotation). The number that governs the residual risk is the access-token lifetime. With a 15-minute access token, a revoked user keeps access for *at most* 15 minutes after you pull their refresh token — not forever, the way the leaked API key did.

![a before and after comparison contrasting a long-lived API key that leaks for an unbounded window against a short-lived JWT plus refresh token whose leak window is only minutes](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-3.png)

Hold this trade-off in mind as we now open up the JWT, because every JWT design decision — lifetime, algorithm, storage — is really a decision about *managing the cost of not being able to revoke it.*

## 6. JWT in depth: the three parts

A **JWT (JSON Web Token)**, defined by RFC 7519 and pronounced "jot," is the dominant format for stateless bearer tokens. It is a compact, URL-safe, *self-contained* statement of claims, cryptographically signed so the recipient can verify it was issued by someone holding the signing key and has not been altered. Let me build it up from its three parts, because understanding the structure is what lets you reason about its pitfalls.

A JWT is three base64url-encoded segments joined by dots: `header.payload.signature`. To be explicit about what *not* to do: I will never paste a full real-looking token here, because a complete `eyJ...` three-part string is a secret-shaped string that scanners (and this repo's push protection) will block — and that is itself the lesson, that a JWT is a credential to be guarded. So we will look at each part decoded and separately.

![a layered diagram of a JWT showing the header part the payload of claims and the signature with verification recomputing the signature and rejecting a tampered token](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-4.png)

**Part 1 — the header.** A small JSON object describing how the token is signed. Decoded, it looks like:

```json
{
  "alg": "RS256",
  "typ": "JWT",
  "kid": "2026-06-signing-key-01"
}
```

`alg` names the signing algorithm. `typ` is the media type, `JWT`. `kid` (key id) tells the verifier *which* key to use, which is what makes key rotation possible — you can have several signing keys live at once and the `kid` says which one signed this token. The base64url encoding of this JSON is the first segment, something like `eyJhbGciOi...<base64url header>`.

**Part 2 — the payload (the claims).** The substance of the token: a JSON object of **claims**, where a claim is simply a statement about the subject. Decoded:

```json
{
  "iss": "https://auth.shop.example",
  "sub": "usr_9f2a8c",
  "aud": "https://api.shop.example",
  "exp": 1782205500,
  "iat": 1782204600,
  "jti": "tok_5c1d2e3f",
  "scope": "orders:read payments:write"
}
```

These are the **registered claims** every verifier must know:

- **`iss` (issuer)** — who minted this token. You verify it matches your trusted authorization server.
- **`sub` (subject)** — *who the token is about*, the principal. This is the identity. For our mobile user it is their user id.
- **`aud` (audience)** — who the token is *for*. You verify the `aud` is *you*. This stops a token minted for service A from being replayed against service B; B checks `aud` and rejects a token addressed to A. Skipping this check is a real and common vulnerability.
- **`exp` (expiration)** — a Unix timestamp after which the token is invalid. You verify `now < exp`. This is the lifetime that bounds the un-revocable window.
- **`iat` (issued at)** — when it was minted. Useful for "this token is suspiciously old" checks and for invalidating all tokens issued before a password change.
- **`jti` (JWT id)** — a unique id for this specific token. Lets you maintain a *deny-list* of individual revoked tokens if you need fine-grained revocation before expiry (a small, optional bit of statefulness).

The encoded payload is the second segment, `eyJzdWIiOi...<base64url payload>`. **Critically: this is encoding, not encryption.** Anyone holding the token can base64url-decode the payload and read every claim. Paste a token into a debugger and you see all of it. The signature guarantees *integrity* (it has not been changed) and *authenticity* (you signed it), **not confidentiality.** So: **never put a secret in a JWT payload.** No passwords, no full credit-card numbers, no private data. The payload is public-readable by anyone who holds the token, which is the whole point of the figure above.

**Part 3 — the signature.** This is what makes the token trustworthy. The issuer takes the encoded header and payload, joins them with a dot, and runs the algorithm named in `alg` over that string with the signing key. The result, base64url-encoded, is the third segment, the `<signature>`. To *verify*, the recipient recomputes the signature over the received `header.payload` with the appropriate key and checks it matches. If even one byte of the header or payload was altered — an attacker bumping their own `scope` from `orders:read` to `payments:write` — the recomputed signature will not match and the token is rejected. The signature is the guarantee that the claims you are reading are exactly the claims the issuer signed.

#### Worked example: decoding and validating a JWT

A mobile request arrives:

```http
GET /v1/orders/ord_55c1 HTTP/1.1
Host: api.shop.example
Authorization: Bearer eyJhbGciOi...<header>.eyJzdWIiOi...<payload>.<signature>
```

The server validates it. *Validation is not just "is the signature valid"* — that is the mistake juniors make. Full validation is a checklist, and skipping any item is a vulnerability:

```python
import jwt  # PyJWT

def validate_access_token(token: str) -> Principal:
    try:
        claims = jwt.decode(
            token,
            key=public_key_for_kid(token),   # by `kid` in the header
            algorithms=["RS256"],            # PIN the algorithm — see pitfalls below
            audience="https://api.shop.example",   # verify `aud` is us
            issuer="https://auth.shop.example",    # verify `iss` is trusted
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
        # `exp` is checked automatically by jwt.decode and raises if expired
    except jwt.ExpiredSignatureError:
        raise Unauthorized("token expired")     # -> 401, client should refresh
    except jwt.InvalidTokenError:
        raise Unauthorized("token invalid")     # -> 401

    if claims["jti"] in revoked_jti_denylist:   # optional fine-grained revoke
        raise Unauthorized("token revoked")

    return Principal(kind="user", user_id=claims["sub"], scopes=claims["scope"].split())
```

Read the checklist out loud: verify the **signature** with the right key; pin the **algorithm**; check **`exp`** (not expired); check **`aud`** (it is for us); check **`iss`** (we trust the minter); optionally check the **`jti`** deny-list. Only then do we trust `sub` as the principal. Notice there is *no database call* on the happy path — that is the stateless speed — except the optional deny-list check, which is the small bit of statefulness we pay for fine-grained revocation.

### HS256 versus RS256: the algorithm choice

The `alg` choice is not cosmetic; it determines *who can mint tokens you will trust.* There are two families.

**HS256** is **symmetric** (HMAC with SHA-256). There is *one secret*, and the *same* secret both signs and verifies. Anyone who can verify a token can also *forge* one, because verifying and signing use the identical key. This is fine when there is exactly one trust boundary — the issuer and the verifier are the same service, or share a single secret you tightly control. The moment you need a *second* service to verify tokens, you must hand it the signing secret, and now that service (and anyone who compromises it) can mint tokens impersonating anyone.

**RS256** is **asymmetric** (RSA signature with SHA-256). There is a *key pair*: a **private key** that *only the issuer* holds and uses to sign, and a **public key** that anyone can hold and use to *verify* but *not* to sign. This is the property you want at scale: your authorization server keeps the private key locked down and signs tokens; every downstream service (`orders-service`, `payments-service`, the gateway) holds only the *public* key and can verify tokens locally without being able to forge them. Public keys are published at a standard endpoint as a **JWKS** (JSON Web Key Set, typically at `/.well-known/jwks.json`), keyed by `kid`, so verifiers fetch and cache them and rotation just works.

![a matrix comparing HS256 symmetric signing against RS256 asymmetric signing across the key model who can sign who can verify and the best fit](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-5.png)

The rule that falls out: **use HS256 only inside a single trust boundary; use RS256 (or ES256) the moment more than one party verifies tokens** — which, in any real service architecture, is immediately. The asymmetric default means a compromised verifier cannot mint tokens, only verify them, which is a dramatically smaller blast radius.

| Property | HS256 (symmetric) | RS256 (asymmetric) |
| --- | --- | --- |
| Keys | one shared secret | private + public pair |
| Who can sign | anyone with the secret | only the issuer (private key) |
| Who can verify | needs the shared secret | anyone (public key) |
| Verifier compromise | can forge any token | can only verify, never forge |
| Key distribution | secret to every verifier (risky) | publish public key via JWKS |
| Best fit | single service / one boundary | federation, microservice fleet |

### The dangerous pitfalls: `alg:none` and key confusion

Two JWT vulnerabilities are infamous enough that you must know them by name, because both come from trusting the *token* to tell you how to verify it.

**The `alg:none` attack.** Early JWT libraries honored an algorithm value of `none`, meaning "this token is unsigned." An attacker would take a valid token, change the header to `{"alg":"none"}`, strip the signature, edit the payload to elevate their `scope`, and send it. A naive verifier read `alg` *from the attacker-controlled header* and, seeing `none`, performed *no signature check at all*, accepting the forged claims. The defense is absolute: **never let the token's header dictate whether you verify.** Pin the accepted algorithm in your verification call — `algorithms=["RS256"]`, as in the worked example — and reject anything else, especially `none`.

**The key-confusion (RS256→HS256) attack.** A subtler version. Suppose your server verifies RS256 tokens with the public key. An attacker takes the *public* key (which is, by design, public), and crafts a token with the header changed to `{"alg":"HS256"}`, signing it with the public key *as if it were an HMAC secret*. If your verifier naively reads `alg` from the header and sees `HS256`, it will use the public key as an HMAC key to verify — and the signature *matches*, because the attacker used that same public key to sign. The token is accepted. The root cause is identical: the verifier trusted the attacker-controlled `alg` and used the wrong key type. The defense is identical: **pin the algorithm**, so an `HS256` header is rejected outright by a server that only accepts `RS256`.

The meta-lesson: *the token is data from an untrusted party.* Its header tells you the issuer's *intent*, but your verifier must decide the *policy* (which algorithm, which key) independently, from configuration, not from the token. A token does not get to choose how it is checked.

There is a quantitative reason the stateless path is worth this care. Verifying an RS256 signature is a single public-key operation — on the order of tens of microseconds per token on a modern CPU — and it happens *locally*, with no I/O. A stateful session lookup, by contrast, is a network round-trip to a session store: if that store is in the same data center you might pay $1$–$2$ ms, but a cross-region lookup or a momentarily slow store can spike to tens of milliseconds, and *every* authenticated request pays it. Multiply by the request rate of a fleet — say a service handling $10{,}000$ requests per second — and the difference between a $50\,\mu s$ local verify and a $2$ ms remote lookup is the difference between negligible CPU and a session store that is now a $20{,}000$-lookups-per-second bottleneck and a single point of failure. The stateless token does not just *avoid* the lookup; it removes a shared dependency from the request path entirely. That is the architectural payoff that justifies tolerating the bounded revocation delay, and it is why the math points the same way as the security analysis: short-lived, locally-verifiable, asymmetric tokens.

One more property of the JWT to keep honest: **size**. A JWT is sent on *every* request, and a fat payload — dozens of claims, embedded permission lists, a serialized profile — inflates every request by hundreds of bytes or more. Over a slow mobile link, repeated on every call, that adds up. Keep the payload lean: identity (`sub`), the standard validation claims (`iss`, `aud`, `exp`, `iat`, `jti`), and a *compact* scope string — not a denormalized copy of the user's entire permission set. If authorization needs rich data, look it up by `sub` at the authorization layer rather than stuffing it into a token that rides every request. The token proves *who*; it should not try to be the whole user record.

### Where to store JWTs on the client

For machine clients (servers, mobile backends) storage is the platform's secret store and not interesting. For *web* clients it is a genuine, contested decision with a clear answer, and it is governed by two threats: **XSS** (an attacker runs JavaScript in your page and tries to steal the token) and **CSRF** (an attacker's site makes the browser send a request with your credential).

- **`localStorage` — no.** A token in `localStorage` is readable by *any* JavaScript on the page. One XSS bug, one compromised npm dependency, one malicious analytics script, and your access token is exfiltrated. The convenience is not worth it. This is the single most common JWT storage mistake; do not make it.
- **`HttpOnly` cookie — good, with the CSRF caveat.** Store the token in an `HttpOnly; Secure; SameSite` cookie. JavaScript cannot read it (XSS-resistant), and `SameSite=Lax`/`Strict` plus a CSRF token handles the cross-site risk. This is the most robust option for a browser app, and it converges with the session story from §4 — the difference is the cookie now carries a self-contained token instead of a session id.
- **In memory (a JavaScript variable) — acceptable for the access token.** Keep the short-lived access token in a variable, never persisted; it dies on page refresh, at which point you silently re-acquire one via the refresh token. It is not readable from storage, only from running code, which shrinks the XSS window. The trade-off is the re-acquire-on-refresh complexity.

The standard modern pattern for a browser SPA: **refresh token in an `HttpOnly` cookie, access token in memory.** The refresh token — the high-value, longer-lived credential — is locked away from JavaScript entirely; the access token — short-lived and quickly replaceable — lives in memory where a single XSS gets only a 15-minute credential, not a persistent one. Never the access token in `localStorage`, never the refresh token anywhere JavaScript can read it.

## 7. Refresh tokens and rotation

We have established the resolution to the stateful/stateless dilemma — short access tokens plus a stateful refresh token — so let us make the refresh mechanism concrete, because it has its own important security pattern.

A **refresh token** is a long-lived, *stateful* credential whose only job is to obtain new access tokens. The client logs in once and receives both: a short-lived access token (minutes) for calling the API, and a longer-lived refresh token (days or weeks) stored securely. The client uses the access token until it expires, then presents the refresh token to the auth server's token endpoint to get a fresh access token — *without* making the user log in again. Because the refresh token is stateful (a row in the auth server's database), revoking it is instant: delete the row, and the user can no longer mint new access tokens. Within one access-token lifetime, they are fully cut off.

The token endpoint exchange looks like this:

```http
POST /oauth/token HTTP/1.1
Host: auth.shop.example
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&refresh_token=rt_8c1d...&client_id=mobile-app
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store

{
  "access_token": "eyJhbGciOi...<header>.eyJzdWIiOi...<payload>.<signature>",
  "token_type": "Bearer",
  "expires_in": 900,
  "refresh_token": "rt_9d2e...",
  "scope": "orders:read payments:write"
}
```

Two things to notice. The response sets `Cache-Control: no-store` so no proxy or cache retains the tokens. And the response returns a *new* `refresh_token`, different from the one sent — that is **refresh-token rotation**, the security pattern worth its own paragraph.

![a timeline showing login issuing an access and refresh token then API calls then access expiry then a refresh that rotates the old refresh token and detects reuse to revoke the family](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-6.png)

**Refresh-token rotation** means every time a refresh token is used, the server issues a *new* refresh token and *invalidates the old one*. The chain of refresh tokens for one login forms a "family." The payoff is **reuse detection**. Suppose an attacker steals a refresh token. Two things can now happen. If the legitimate client refreshes first, the attacker's stolen token is already invalidated — useless. If the attacker refreshes first, the legitimate client's next refresh presents the *now-old, already-used* token — and the server, seeing a *previously rotated* refresh token being reused, knows something is wrong: a refresh token should never be presented twice. The correct response is to **revoke the entire token family**, forcing a fresh login. So rotation turns a stolen refresh token from a silent, persistent compromise into a *detectable event* that cuts off both the attacker and (temporarily) the legitimate user, who simply logs in again. This is the OAuth 2.0 best-current-practice recommendation for public clients (mobile apps and SPAs), and it is the reason the timeline above ends in "reuse detected → revoke the family."

#### Worked example: a user is compromised and you cut them off

Our mobile user `usr_9f2a` reports their phone stolen. Walk the revocation:

1. Support marks the user's sessions for revocation. Concretely, the auth server **deletes all refresh-token rows** for `usr_9f2a` (the stateful action).
2. The thief's phone still holds a *valid access token* — it works until `exp`. With a 15-minute lifetime, that is a worst case of 15 minutes of residual access.
3. When that access token expires, the app tries to refresh. The refresh token is gone from the server. The refresh fails with `401`. The thief is locked out.

The whole design is visible here. The *stateless* access token gave us speed on millions of normal calls and cost us a *bounded* 15-minute revocation delay. The *stateful* refresh token gave us a single, instant lever to pull. If we had used a long-lived stateless JWT with no refresh layer, we could not have cut the thief off at all until the token expired — which is the long-lived-API-key failure mode all over again, just dressed up as a JWT. **A long-lived JWT is the worst of both worlds: it has the un-revocability of stateless and none of the safety of a short lifetime.** Short access token, stateful refresh, rotation. That is the pattern.

## 8. mTLS: proving service identity with certificates

Everything so far has put the credential *in the request* — a header, a cookie. **Mutual TLS (mTLS)** moves identity down into the *transport layer itself*, and it is the strongest answer for **service-to-service** authentication and the foundation of **zero-trust** networking.

Recall ordinary TLS (the `https://` you use every day): the *server* presents a certificate proving its identity to the *client*, and the client verifies it against a trusted certificate authority (CA). That is *one-way* authentication — the client learns who the server is, but the server learns nothing verified about the client. **Mutual TLS adds the second direction:** the *client also* presents a certificate, and the server verifies *it* against a trusted CA. After the handshake, *both* parties have cryptographically proven their identities, before a single byte of HTTP is exchanged. The client's identity is the **subject of its certificate** — `CN=orders-service` or, more robustly, a SPIFFE identity like `spiffe://shop.example/ns/prod/sa/orders-service` in the certificate's subject alternative name.

Why is this the strong scheme? Because the credential is a **certificate with a private key the client never transmits.** Unlike a bearer token — which travels in every request and can be replayed by anyone who steals it — the client's private key *stays on the client* and is used only to prove, via the TLS handshake, that it holds the key matching the public certificate. There is no bearer secret crossing the wire to intercept. The identity is bound to *possession of a private key*, not to *knowledge of a string*. That is a categorically stronger guarantee, which is why service meshes use it as the default.

mTLS also gives you **fast revocation** at the PKI level and naturally short-lived credentials. In a modern mesh, service certificates are issued with *very* short lifetimes (often hours) and rotated automatically by the mesh's certificate authority, so a compromised cert self-expires quickly — the same "short-lived beats long-lived" principle, now at the certificate layer. Revocation can also use certificate revocation lists (CRLs) or OCSP, but in practice the short lifetimes do most of the work.

The trade-off is **operational complexity and key management.** You need a CA, a way to issue certificates to every workload, a way to rotate them before they expire, and a way to distribute the trust anchor. Doing this by hand is miserable; this is exactly the problem a **service mesh** (Istio, Linkerd) or a workload-identity framework (SPIFFE/SPIRE) solves — it issues, rotates, and validates short-lived service certificates transparently, so the application code does not even know mTLS is happening; the sidecar handles the handshake. That is why mTLS is rare for the *public* edge (you cannot easily issue certs to every third-party developer's laptop) and dominant for the *internal* fleet (you own every workload and can automate the PKI).

This series owns the *API contract*; the *fleet plumbing* of mTLS — the mesh, the sidecars, the CA hierarchy, the zero-trust posture — is covered in depth in [service-to-service security: mTLS and zero-trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust). Read that for how to operate it across a service fleet. Here, the design point is simply: **for internal service-to-service calls, mTLS gives you the strongest, most automatable identity, with no bearer secret on the wire — reach for it inside the mesh, not at the public edge.**

A gateway config that requires a client certificate and forwards the verified identity to upstreams (Envoy-style) makes the contract concrete:

```yaml
# Listener: require and verify a client certificate (mTLS)
transport_socket:
  name: envoy.transport_sockets.tls
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
    require_client_certificate: true        # <- this makes it MUTUAL TLS
    common_tls_context:
      validation_context:
        trusted_ca: { filename: /etc/certs/mesh-ca.pem }   # trust the mesh CA
# Then forward the verified identity to the upstream as a header:
route_config:
  request_headers_to_add:
    - header: { key: x-forwarded-client-identity, value: "%DOWNSTREAM_PEER_URI_SAN%" }
```

The upstream `orders-service` now receives a request whose caller identity was *proven at the transport layer* and forwarded as `x-forwarded-client-identity: spiffe://shop.example/ns/prod/sa/payments-service`. No token to validate, no secret to leak — the TLS handshake already did the authentication.

One subtlety to get right with this pattern: the upstream must *only* trust `x-forwarded-client-identity` when the request genuinely came through the mTLS-terminating proxy, and never when a client could set that header directly. If a caller could reach the upstream *bypassing* the proxy and simply set `x-forwarded-client-identity: spiffe://.../sa/admin-service`, they would impersonate any service by typing a header — the exact mistake of trusting client-supplied identity. In a mesh this is enforced by network policy (the workload only accepts connections from its own sidecar) and by the sidecar *stripping* any inbound copy of that header before re-adding the verified one. The principle is the same one running through the whole post: identity must come from something the caller *proved*, never from something the caller merely *claimed* in a header they control.

It is also worth being clear about what mTLS authenticates: the *workload*, not the *user*. An mTLS handshake proves "this connection is from `payments-service`," which is exactly right for service-to-service trust. But if `payments-service` is acting *on behalf of* an end user, that user's identity is a *separate* fact that must travel in the request — typically as a JWT forwarded from the edge. So in a real fleet the two schemes compose: mTLS proves the calling *service*, and a forwarded token carries the *user* the service is acting for. The downstream then knows both "which service called me" (transport identity) and "for which user" (token claim), and can authorize on either. This layering — transport identity plus propagated user identity — is the standard zero-trust posture, and it is why §8's "mTLS for services, JWT for users" is a composition rather than a competition.

## 9. Putting it together: a request through the boundary

We have four schemes; in a real system several of them coexist, because different callers reach the same API. The gateway is where they converge. It inspects the credential on each request, routes it to the right verification path, and either rejects with `401` or attaches a verified principal and forwards to the handler.

![a branching flow where a request reaches the gateway which classifies the credential and sends it down an API key lookup a JWT verify or a client certificate check each ending in a 401 rejection or a trusted handler](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-7.png)

Reading the flow: a request arrives carrying *some* credential. The gateway classifies it — a `Basic` header with an `sk_`/`pk_` value is an API key; a `Bearer` header is a JWT; a client certificate in the TLS handshake is mTLS. Each goes down its own verification path: hash-and-look-up for the key, verify-signature-and-claims for the JWT, validate-the-cert-chain for mTLS. Every path has exactly two exits: **`401`** if verification fails (no valid identity), or **the handler** with a trusted principal attached. Crucially the graph *branches by credential type and merges back* at the handler — and only an authenticated request ever reaches application code. Authorization happens *after* this, inside or just before the handler, on the principal this stage produced.

This is the boundary principle from §1 made real. The handler author writes `def get_order(principal, order_id)` and *trusts* `principal` completely, because no path reaches them without it. The auth complexity — four schemes, signature math, cert chains — is centralized at the edge, not scattered through business logic.

For our Payments & Orders platform, the three callers map onto the three paths exactly:

```http
# 1) Partner reconciliation backend — scoped API key (Basic)
GET /v1/payments?status=settled HTTP/1.1
Authorization: Basic <base64 of sk_recon_live_YOUR_KEY_HERE:>

# 2) Mobile app — short-lived JWT bearer (+ refresh token held securely)
GET /v1/orders/ord_55c1 HTTP/1.1
Authorization: Bearer eyJhbGciOi...<header>.eyJzdWIiOi...<payload>.<signature>

# 3) Internal orders-service calling payments-service — mTLS, no header secret
GET /internal/payments/pay_77f2 HTTP/1.1
# identity proven by the client certificate in the TLS handshake; no Authorization header
```

Three callers, three schemes, each chosen by *who the caller is and what they can safely hold* — which is precisely the decision the final section formalizes.

| | API key | Session cookie | JWT bearer | mTLS |
| --- | --- | --- | --- | --- |
| State model | stateful row | stateful session | stateless (self-contained) | stateless (cert) |
| Revoke before expiry | yes, delete row | yes, drop session | no (use short exp + refresh) | yes, short cert + CRL |
| Carries claims | no (look up) | no (look up) | yes (signed payload) | identity = cert subject |
| Travels across services | poorly | no | yes (local verify) | yes (mesh) |
| Best caller | partner/server backend | first-party browser | mobile / SPA | internal service |
| Primary risk | long-lived leak | CSRF | un-revocable + storage | key/PKI management |
| Transport | TLS, header | TLS, cookie | TLS, header | TLS handshake itself |

### Token transport: the leaks you cause yourself

Most authentication breaches are not cryptographic — they are *plumbing*. The credential was perfectly good; you just left it somewhere it could be read. The discipline of token *transport and handling* is as important as the scheme, and it is worth being explicit about every place a credential tends to escape.

**The URL is not a safe place for a credential — ever.** It is tempting, because `?access_token=...` is easy to test and works in a browser address bar. But a URL is the *least* private part of a request. It lands in server access logs (nearly every web server logs the full request line by default), in the browser's history and autocomplete, in the `Referer` header that the browser sends to *every third-party resource the page loads* (your analytics provider now has the token), in CDN and proxy logs, and in error-tracking screenshots. A query-string token is, for practical purposes, a published token. The credential belongs in the `Authorization` header, which is not logged by default and not forwarded in `Referer`. The one acceptable exception — a *single-use, short-expiry, signed* URL for a download link — proves the rule by how carefully it is constrained.

**Your own logs are an attacker's treasure.** The request-logging middleware that dumps headers for debugging is the same middleware that writes `Authorization: Bearer eyJ...` into a log line that flows to your log aggregator, gets indexed, replicated to a backup, and is readable by every engineer with log access and every vendor in your logging pipeline. Redact `Authorization`, `Cookie`, and `Set-Cookie` at the source — before the log line is written, not after. The same goes for error reports: an exception handler that serializes the full request into a crash report has just shipped the user's token to your error-tracking SaaS. Treat credentials as you would treat raw passwords in logs, because they are functionally equivalent.

**Timing and comparison leaks are real.** We used `hmac.compare_digest` in §3 for a reason. A naive `==` on secrets short-circuits at the first differing byte, and the time difference, though tiny, is *measurable* over enough requests, letting an attacker recover a secret one byte at a time. Any comparison of a presented secret against a stored one — API key hash, session id, CSRF token, signature — must be constant-time. It is one function call; there is no excuse to skip it.

**Caching can preserve what should be ephemeral.** Token-endpoint responses (§7) set `Cache-Control: no-store` so that no intermediary keeps a copy of freshly minted tokens. Authenticated responses in general should be careful with caching — a shared cache that stores a per-user response keyed only by URL can serve user A's data to user B. The `Vary: Authorization` header and `private`/`no-store` directives are part of the auth contract, not just the caching one (see [caching, ETags, and conditional requests](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation)).

**Clock skew breaks `exp` if you let it.** Token validation depends on comparing `now` to `exp` and `iat`. If the issuer's clock and the verifier's clock disagree by a few seconds, a freshly issued token can appear *not yet valid* (`iat` in the future) or a just-valid token can appear *expired*. Real validators allow a small **leeway** (typically 30–60 seconds) on these time comparisons to absorb skew, while keeping the window tight enough that it does not meaningfully extend a token's life. This is a small detail that causes maddening intermittent `401`s in production if forgotten, especially across data centers.

### A note on what authentication does *not* solve

It is worth drawing the boundary clearly, because authentication is often asked to carry weight it cannot bear. Authentication tells you *who* is calling, with some confidence. It does *not* tell you whether they *should* be calling this endpoint — that is authorization. It does *not* protect against a malformed or malicious *payload* from an authenticated caller — an authenticated request can still carry an injection attack or a mass-assignment exploit, which is why [input validation and the OWASP API Top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10) is its own discipline. It does *not* by itself stop an authenticated client from hammering you — that is rate limiting and quota, keyed *by* the authenticated principal. And it does *not* prove the request was *intended* — a perfectly authenticated cookie-bearing request can be a CSRF forgery, which is why §4 layered `SameSite` and CSRF tokens on top of the cookie.

The mental frame: authentication establishes a *trusted identity* and hands it to everything downstream. Everything downstream — authorization, validation, rate limiting, auditing — *consumes* that identity but solves a different problem. An API's security is the *composition* of all of these; authentication is the foundation they all stand on, which is exactly why a weak foundation makes the rest decorative. Get the identity right, attach it at the boundary, and the other layers have something real to work with.

## 10. Case studies: how this is actually done

These are accurate, named references — the patterns above are not theoretical.

**Stripe — prefixed, scoped, secret API keys.** Stripe's API uses exactly the prefix scheme described in §3: `sk_live_` and `sk_test_` for secret keys, `pk_` for publishable keys, `rk_` for restricted keys, with the live/test mode encoded in the prefix so a test key cannot move real money. Keys are passed as the HTTP `Basic` username (so `curl -u sk_live_YOUR_KEY_HERE:` works), shown in full only at creation, and displayed afterward as a masked last-4 in the dashboard. Restricted keys let you scope a key to specific resources and read/write levels — the least-privilege principle that bounded the blast radius in our opening story. Stripe also pairs this with its idempotency-key mechanism (covered in [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions)) so that authenticated retries are safe — authentication and safe retries are complementary clauses of the same contract.

**OAuth 2.0 / OIDC at scale — short JWTs and JWKS.** Large identity providers (Auth0, Okta, Google, Microsoft Entra) issue **RS256-signed JWT access tokens** with short lifetimes (commonly an hour or less) and publish their public keys at a JWKS endpoint (`/.well-known/jwks.json`) keyed by `kid`. Every resource server fetches and caches those public keys and verifies tokens *locally* — no callback to the IdP per request — which is the stateless-scaling property in production across thousands of services. OpenID Connect layers an *identity* token (the `id_token`, also a JWT) on top of OAuth's *access* token; the access token says what you may do, the ID token says who you are. The full grant types and the OIDC layer are the subject of [OAuth 2.0 and OpenID Connect for API designers](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers). Refresh-token rotation with reuse detection, described in §7, is the OAuth working group's current best-practice recommendation for public clients.

**Service meshes — mTLS by default.** Istio and Linkerd implement automatic mutual TLS between workloads: a sidecar proxy (or, for Linkerd, a micro-proxy) terminates and originates TLS, the control plane acts as a certificate authority issuing **short-lived** workload certificates (Istio's default is on the order of 24 hours, rotated automatically), and identities follow the SPIFFE format (`spiffe://...`). Application code is unchanged — it makes a plain HTTP call to a sidecar, and the sidecar handles the mutual handshake. This is zero-trust in practice: no service trusts the network, every service-to-service call is authenticated by certificate, and there is no bearer secret to leak. The operational details live in the [microservices mTLS post](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

**GitHub — moving off long-lived tokens.** GitHub's evolution is itself a case study in the lesson of this post. It moved from long-lived personal access tokens toward *fine-grained*, *scoped*, *expiring* tokens and short-lived installation tokens for GitHub Apps, and it added secret scanning that matches token prefixes to catch leaks in public repos — the same prefix-enables-scanning argument from §3. The arc of the industry is unmistakable: from long-lived, broadly-scoped, never-expiring secrets toward short-lived, narrowly-scoped, auto-rotating ones. That arc *is* this post's thesis.

## 11. When to reach for each (and when not to)

Authentication has no single right answer; it has a right answer *per caller*. Here is the decisive guidance.

![a decision tree splitting callers into human clients and machine clients then into browser session cookies mobile or SPA short-lived JWTs internal service mTLS and partner backend scoped API keys](/imgs/blogs/authentication-api-keys-sessions-jwt-and-mtls-8.png)

**Reach for an API key when** the caller is a *server-side machine* with a stable, long-term relationship — a partner backend, a CI pipeline, a webhook configurator — *and* you can give it the discipline keys require (prefix, hash-at-rest, scope, rotation). Keys are simple and they identify an *application*, which is often exactly what you want for B2B integrations.

**Do not use an API key when** the caller is a browser or a mobile app, or when you need *per-user* identity, or when the key would have to live somewhere readable (a frontend bundle, a public repo). A `pk_` publishable key in a frontend is fine *only* because it is deliberately low-privilege; a secret key in a browser is a breach waiting to happen. And never use a key when you need fast, fleet-wide revocation across services — a leaked long-lived key is the failure this whole post opened with.

**Reach for a session cookie when** the caller is a *first-party browser* app on your own domain. You get instant revocation and `HttpOnly` protection against XSS exfiltration. **Do not use sessions** for native mobile apps (no cookie jar in the way browsers have one), for any cross-service call (the session store does not travel), or for third-party API consumers. And if you do use cookies, you *must* handle CSRF (`SameSite=Lax` plus a token on mutating endpoints) — sessions without CSRF defense are a vulnerability, not a scheme.

**Reach for a JWT (short-lived, RS256, plus a rotating refresh token) when** the caller is a *mobile app or SPA*, or when *multiple services* must verify identity locally without a shared session store. This is the default for modern user-facing auth at scale. **Do not use a JWT** as a *long-lived* credential — that recreates the un-revocable-API-key problem with extra steps. Do not store the access token in `localStorage`. Do not use HS256 once more than one party verifies. Do not skip the `aud`/`iss`/`exp` checks or fail to pin the `alg`. And do not reach for JWTs when a simple stateful session would do — for a single first-party web app, a session is simpler and instantly revocable, and you do not need the stateless property at all. *Statelessness is a tool for scaling verification across services; if you do not have that problem, you are paying the revocation cost for nothing.*

**Reach for mTLS when** the caller is an *internal service* inside a fleet you control, especially one already running a service mesh. It is the strongest identity, with no bearer secret on the wire and automatable short-lived certs. **Do not use mTLS** at the *public* edge for arbitrary third-party developers — you cannot practically issue and rotate certs to every external client's environment, and the operational burden is enormous. mTLS shines where you own both ends.

The unifying principle across all four, and the security crux of this whole post: **default to short-lived, asymmetric, scoped credentials with a clear revocation path, and centralize verification at the boundary.** Long-lived, symmetric, broadly-scoped, hard-to-revoke secrets are the recurring shape of every authentication disaster. When you must use a long-lived secret (a server-to-server API key), compensate hard — hash it, scope it tightly, prefix it for scanning, rotate it routinely, monitor its use. When you can avoid one (a user-facing or service-to-service caller), avoid it — short JWTs with rotation, or mTLS.

## 12. Stress-testing the design

Pose the failures and walk the responses, because a scheme is only as good as its behavior under attack.

**The token leaks.** A short-lived access token leaks (logged by mistake, intercepted on a misconfigured proxy). Damage window: until `exp` — minutes. Compare the opening API key: until a human noticed and rotated — weeks. The lifetime *is* the blast radius, and that is the whole argument for keeping it small.

**The refresh token is stolen.** Rotation with reuse detection (§7) turns this from a silent persistent compromise into a *detected event*: the first reuse of a rotated token triggers family revocation, cutting off both parties. Without rotation, a stolen refresh token is a quiet, long-lived backdoor.

**A verifier service is compromised.** With HS256, the attacker now holds the signing secret and can forge tokens impersonating *anyone* across the whole trust boundary — game over. With RS256, the attacker holds only the *public* key and can verify but not forge. This single difference is why asymmetric is the default at scale: it makes a verifier compromise survivable.

**The signing key must be rotated.** The `kid` header makes this graceful: publish the new key in the JWKS alongside the old, start signing new tokens with the new `kid`, let old tokens (signed with the old `kid`) verify until they expire, then retire the old key. No flag day, no mass re-issuance — the same expand/contract discipline the series applies to [schema evolution](/blog/software-development/api-design/schema-evolution-adding-removing-renaming-fields-safely).

**A token is replayed against the wrong service.** The `aud` claim defends this: a token minted for `payments-service` carries `aud: payments-service`, and `orders-service` rejects it because the audience is not itself. Skip the `aud` check and a token for one service works against all of them — a real, common, and serious bug.

**An attacker forges the algorithm.** The `alg:none` and key-confusion attacks (§6) both die against a verifier that *pins* the algorithm in configuration and refuses to read it from the token. The token is untrusted data; it does not get to choose how it is checked.

**A million callers, all hitting a downstream service.** This is where stateless wins decisively. Each service verifies JWTs locally with a cached public key — no per-request database lookup, no session-store hot spot, no network call to an auth service. The same scenario with sessions would funnel every request through a shared session store, which becomes the bottleneck and the single point of failure. *This is the entire reason stateless tokens exist*, and the reason we tolerate their revocation cost: at fleet scale, local verification is the only thing that scales.

## Key takeaways

- **Authentication answers "who is calling"; authorization answers "what may they do."** Authenticate first, then authorize. `401` means unauthenticated (try a different credential); `403` means unauthorized (the credential is fine, the answer is still no). Do them at the boundary, before routing, so a new endpoint is authenticated by default.
- **A bearer token is cash: whoever holds it, uses it.** That makes lifetime, storage, and TLS-only transport the entire security model. Never put credentials in a URL or a log; always over TLS.
- **API keys are simple but long-lived — compensate hard.** Prefix them for scanning, show only a last-4, *hash them at rest* with a constant-time compare, scope to least privilege, and support two-key rotation. They identify *applications*, best for server-to-server.
- **Sessions are stateful, browser-first, and instantly revocable — but they carry the CSRF tax.** Use `HttpOnly; Secure; SameSite=Lax` plus a CSRF token on mutating endpoints, and only for first-party browser apps; they do not travel across services.
- **The core trade-off is stateful (revocable, does not scale) versus stateless (scales, not revocable before expiry).** Resolve it with *short-lived access tokens plus a stateful, rotating refresh token*: stateless speed on the common path, stateful revocation on the rare one.
- **A JWT is three signed parts — and the payload is readable, not secret.** Verify the signature *and* `exp`, `aud`, `iss`; pin the algorithm; never trust the token's header to choose how it is verified (that is the `alg:none` and key-confusion defense). Prefer RS256 over HS256 the moment more than one party verifies. Never store an access token in `localStorage`.
- **A long-lived JWT is the worst of both worlds** — un-revocable like stateless, dangerous like a long-lived key. Short access token, rotating refresh token, every time.
- **mTLS is the strongest service-to-service identity** — no bearer secret on the wire, identity bound to a private key, short auto-rotated certs in a mesh. Reach for it inside the fleet you control, not at the public edge.
- **Pick by caller:** browser → session cookie; mobile/SPA → short JWT + refresh; internal service → mTLS; partner backend → scoped API key. The security crux: default to short-lived, asymmetric, scoped credentials with a clear revocation path.

## Further reading

- **RFC 7519 — JSON Web Token (JWT).** The token format: header, payload, registered claims (`iss`, `sub`, `aud`, `exp`, `iat`, `jti`), and signing. The authoritative source for everything in §6.
- **RFC 6750 — The OAuth 2.0 Bearer Token Usage.** Defines the `Authorization: Bearer` scheme and the `WWW-Authenticate` error responses — the envelope for §2.
- **RFC 8705 — OAuth 2.0 Mutual-TLS Client Authentication.** Using mTLS for client authentication and certificate-bound access tokens — the spec behind §8.
- **RFC 9110 — HTTP Semantics.** The `Authorization` header, `401`, and the `WWW-Authenticate` mechanism in their modern, consolidated form.
- **OWASP API Security Top 10** and the **OWASP Authentication / JWT Cheat Sheets.** The practitioner's catalog of how authentication fails in the wild, including `alg:none`, key confusion, and token-storage mistakes — the source for the pitfalls and storage guidance.
- **OAuth 2.0 Security Best Current Practice (RFC 9700 / the BCP draft lineage).** The recommendation for refresh-token rotation with reuse detection and short-lived tokens for public clients — the basis for §7.
- Within this series: the [intro hub on the API contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the [capstone review playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2), the next post on [authorization, scopes, and resource-level permissions](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions), [OAuth 2.0 and OpenID Connect for API designers](/blog/software-development/api-design/oauth2-and-openid-connect-for-api-designers), and [input validation and the OWASP API Top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10).
- Out of series: [service-to-service security: mTLS and zero-trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) for operating mTLS across a fleet.

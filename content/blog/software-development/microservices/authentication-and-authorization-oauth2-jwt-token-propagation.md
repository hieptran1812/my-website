---
title: "Authentication and Authorization: OAuth2, JWT, and Token Propagation"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How end-user identity is established once at the edge and carried — verifiably and narrowly — through a whole call graph of services: OIDC and PKCE, JWT validation against JWKS, token exchange, the confused-deputy trap, and authorizing at every hop with OPA."
tags:
  [
    "microservices",
    "authentication",
    "authorization",
    "oauth2",
    "oidc",
    "jwt",
    "open-policy-agent",
    "zero-trust",
    "distributed-systems",
    "software-architecture",
    "backend",
    "security",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-1.webp"
---

A support engineer at ShopFast opened a ticket that should have been impossible. A customer claimed that a \$1,200 refund had been issued to a card that wasn't theirs, on an order they never placed. The fraud team pulled the audit log and found the refund had been triggered by a perfectly ordinary internal call: the order service had called the payment service's `POST /refunds` endpoint, and the payment service had happily executed it. No alarm fired, because every layer of the system behaved exactly as built. The gateway had authenticated *a* user at the edge. The order service had a valid token. The payment service trusted the order service because it was inside the cluster. Each component did its small job correctly, and the sum of those correct behaviors was a stranger draining another customer's money. The post-mortem named the culprit in two words that every senior engineer should be able to recognize on sight: **confused deputy**. The order service, holding a token far more powerful than the action required, acted on behalf of a user it never checked the rights of, and the payment service authorized the call based on *who was asking* (a trusted internal service) rather than *on whose behalf* and *whether that person was allowed*.

This is the failure mode that separates engineers who have only ever bolted a login screen onto a monolith from those who have secured a fleet of services. In a monolith, identity is easy: the user logs in, you put a `user_id` in the session, and every piece of code that needs to know who's calling reads the same in-process session object. There is exactly one trust boundary — the edge — and inside it everything is the same process, so "who is this and what may they do" is a local function call. The moment you split that monolith into a call graph of services talking over the network, you have shattered that single trust boundary into dozens. Now identity has to be *established* at the edge, *carried* across every network hop, *verified* by each service that receives it, and *narrowed* so that no single service holds more power than the task in front of it needs. Get any one of those wrong and you get the ShopFast refund: a system where each part is correct and the whole is exploitable.

This post is the north-south companion to the previous one. Where [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) answered "is this *service* who it claims to be?" — the east-west, machine-to-machine question — this post answers "is this *end user* who they claim to be, and may they do this?" — the north-south, human-to-system question. The two are complementary and you need both: mTLS proves the *channel* and the *caller service*; the token proves the *user* on whose behalf the call is made. A refund call can be over a perfectly authenticated mTLS connection between two legitimate services and still be fraudulent, because mTLS says nothing about whether the human behind the request was allowed to issue a refund. By the end of this post you will be able to do four concrete things. First, stand up end-user authentication at the edge with OAuth2 and OpenID Connect — the authorization-code-plus-PKCE flow for users and client-credentials for service-to-service — and understand exactly what each token (access, ID, refresh) is for. Second, validate a JWT *locally* in any service by checking its signature against the issuer's published public keys, along with its audience, expiry, and scope, and reason honestly about the stateless trade-off this buys and the revocation problem it costs. Third, propagate identity safely across a call graph — choosing between forwarding the token, exchanging it for a narrower one, and asserting a trusted internal identity — and avoid the confused deputy. Fourth, decide *where* to authorize (coarse at the gateway, fine-grained per service) and express policy as code with Open Policy Agent instead of scattering `if user.role == "admin"` checks across forty repositories.

![A stacked diagram showing the end to end identity layers from user sign in through gateway validation to per service authorization with the rule that every hop must verify rather than trust the network](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-1.webp)

The senior thesis fits in one sentence, and the figure above is its skeleton: **authenticate once at the edge, propagate a verifiable identity, authorize at every service, and keep tokens short-lived and narrowly scoped.** Every section that follows is an elaboration or a defense of that sentence. We will use ShopFast — a user logs in, browses, and checks out, touching the gateway, the order service, and the payment service — as the spine throughout, because the abstractions only become clear when you watch a concrete request travel through them. Let us start with the distinction a surprising number of engineers blur, the one that the refund incident turned on.

## Authentication versus authorization: two questions, never one

The single most clarifying habit you can build is to keep two questions strictly separate in your head, because they have different answers, different mechanisms, different failure modes, and live in different places in your architecture. **Authentication** (often abbreviated *authn*) answers *who are you?* It is the act of establishing identity — proving that the entity making a request is the principal it claims to be. **Authorization** (*authz*) answers *what may you do?* It is the act of deciding whether an already-identified principal is permitted to perform a specific action on a specific resource. The order matters: you authenticate first to know who someone is, then authorize to decide what that someone may do. You cannot authorize an unknown principal any more than a bouncer can apply a guest list to a person whose name they don't know.

The ShopFast refund incident was, at root, an authorization failure that hid behind a successful authentication. The user *was* authenticated — they were a real, logged-in customer. The system knew who they were. What it failed to do was authorize: it never asked "is *this* user allowed to issue *this* refund on *this* order?" The order service forwarded a powerful token, and the payment service checked only that the call came from a trusted internal source. Authentication succeeded and authorization was simply absent, and absent authorization defaults, dangerously, to "allow." This is why the two concepts must never collapse into one: an enormous class of breaches is "authenticated but not properly authorized," and if you think of security as a single "logged-in or not" gate you will build exactly that class of hole.

A junior engineer tends to over-invest in authentication — strong passwords, multi-factor, fancy login flows — and under-invest in authorization, treating "is the user logged in?" as the whole of access control. The senior knows the opposite is usually where the breaches live: authentication is largely a solved problem you should buy rather than build, while authorization is the part that is specific to *your* domain, changes constantly as your product grows, and is where almost every real-world access bug lives. A leaked password is bad; a missing `can this user refund this order` check is the bug that ships to production unnoticed because everything *works* — the happy path is fine, and only an adversary or a confused deputy finds the gap. Keep the two questions on separate index cards in your mind for the rest of this post: every mechanism we discuss is doing one or the other, and conflating them is how you build the next refund incident.

## OAuth2 and OIDC in plain terms: four roles and what they exchange

OAuth2 has a reputation for being baroque, and the reputation is half-earned: the specification has many flows, grant types, and extension RFCs, and the names ("resource owner," "bearer token," "implicit grant") are unhelpfully abstract. But the core is simple once you anchor it to four roles and one idea. The one idea is **delegated authorization**: OAuth2 was designed so a user can grant an application limited access to their resources *without giving that application their password*. The classic motivating example is "let this photo-printing site access my photos on the cloud storage provider" — you want to grant access without handing over your storage password. Everything in OAuth2 follows from making that delegation safe.

The four roles are the cast you must keep straight, and the figure below shows them in motion for the flow ShopFast actually uses. The **resource owner** is the human — the ShopFast customer who owns their orders and payment methods. The **client** is the application acting on the user's behalf — ShopFast's single-page web app or mobile app. (Confusingly, "client" here means the *application*, not the end user; the end user is the resource owner.) The **authorization server** (AS) is the system that authenticates the user and issues tokens — for ShopFast, a managed identity provider like Okta or Auth0, or a self-hosted Keycloak. The **resource server** (RS) is the API that holds the protected resources and accepts tokens — ShopFast's order and payment APIs. The dance between these four produces a **token** that the client presents to the resource server, and that token is the entire point: it is a bearer credential that says "the holder of this is authorized to do X on behalf of user Y until time Z."

![A graph showing the OpenID Connect authorization code flow with PKCE where the user authorizes a client which redirects to the authorization server gets a one time code and exchanges it with a verifier for access ID and refresh tokens used against the resource server](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-2.webp)

OAuth2 by itself is an *authorization* framework — it was built to answer "may this app access this resource?" and it deliberately says nothing standardized about *who the user is*. That gap is filled by **OpenID Connect (OIDC)**, a thin identity layer on top of OAuth2. OIDC adds one crucial artifact — the **ID token** — and a small set of standard claims and endpoints, turning OAuth2 from "delegated access" into "delegated access *plus* a standard way to learn who logged in." In practice, when an engineer says "we use OAuth" for login, they almost always mean OIDC, because login requires knowing the user's identity, which is exactly what OIDC standardizes. The distinction matters for one practical reason that trips people up constantly: **the ID token is for the client to learn who the user is; the access token is for calling APIs.** Mixing them up — sending the ID token to your API, or trying to read user profile fields out of the access token — is a common and subtle bug we'll return to.

### The flows: authorization-code + PKCE for users, client-credentials for services

OAuth2 defines several *grant types* (flows), and modern practice has narrowed the choice sharply, killing off the dangerous ones. For a **user logging in through a browser or mobile app**, the correct flow is **authorization code with PKCE** (Proof Key for Code Exchange, pronounced "pixy"). Here is the shape, and it maps directly onto the figure above. The client generates a random secret called the *code verifier*, hashes it into a *code challenge*, and redirects the user's browser to the authorization server with that challenge attached. The user authenticates directly with the AS — entering their password and MFA on the AS's own pages, so the client *never sees the password*. The AS, after the user consents, redirects back to the client with a short-lived, single-use **authorization code**. The client then makes a back-channel call to the AS's token endpoint, presenting the code *and* the original code verifier. The AS hashes the verifier, checks it matches the challenge it stored, and only then issues tokens. 

Why the extra PKCE dance? Because the authorization code travels through the browser's redirect — a place where it can leak (browser history, referrer headers, a malicious app registered for the same redirect URI on mobile). Without PKCE, an attacker who steals the code could redeem it for tokens. PKCE binds the code to the specific client instance that started the flow: only the holder of the original verifier can redeem the code, so a stolen code is useless. PKCE was originally designed for mobile and SPA clients that cannot keep a client secret (anything shipped to a user's device is extractable), but current guidance — OAuth 2.1 — recommends PKCE for *all* authorization-code flows, including confidential server-side clients. The older **implicit flow** (tokens returned directly in the redirect URL) and the **resource owner password credentials** grant (the app collects the user's password and sends it to the AS) are both deprecated and you should treat them as forbidden in new systems: the implicit flow leaks tokens in URLs, and the password grant defeats the entire purpose of OAuth by making the client handle passwords.

For **service-to-service** calls where there is no user — a nightly batch job, one backend calling another's API on its own behalf — the flow is **client credentials**. The calling service authenticates to the AS with its own credentials (a client ID and secret, or better, an mTLS client certificate or a workload identity) and receives an access token representing *the service itself*, not any user. There is no resource owner because no human is involved. This is the flow you use for the machine-to-machine plane, and it's where this post overlaps with the mTLS world: the service's identity for client-credentials can and should be its mTLS workload identity rather than a long-lived shared secret. Keep the two flows mentally separated: authorization-code-plus-PKCE produces a token *on behalf of a user*; client-credentials produces a token *representing a service*. Most confused-deputy bugs come from using a service's client-credentials token where a user token should have flowed, so that the service acts with its own broad privilege instead of the user's narrow one.

### The three tokens: access, refresh, and ID

The flow above produces up to three tokens, and a senior can recite what each is for without hesitation. The **access token** is the bearer credential you present to a resource server to call its API. It is short-lived by design — minutes, not days — and carries the user's identity (the `sub` claim), the audiences it's valid for (`aud`), what it's allowed to do (`scope`), and an expiry (`exp`). The **refresh token** is a longer-lived credential whose *only* job is to obtain new access tokens when the current one expires, without forcing the user to log in again. It is never sent to resource servers — only back to the authorization server's token endpoint — and because it is long-lived it must be stored carefully (an HTTP-only secure cookie, or platform secure storage on mobile). The **ID token** (an OIDC addition) is a JWT that tells the *client* who the user is — name, email, when they authenticated — so the SPA can render "Hi, Maria" without an extra API call. The ID token is for the client; it is *not* an API credential and should never be sent to a resource server as authorization.

The short-access-token-plus-long-refresh-token split is one of the most important security trade-offs in the whole design, and it exists to bound the damage from a leaked token. We'll quantify it precisely later, but the intuition is this: a leaked access token is dangerous only until it expires, so making it expire in minutes shrinks the window an attacker has; the refresh token, being more powerful, lives in fewer places (never in URLs, never sent to APIs) and can be revoked centrally. The whole pattern is "keep the credential that flows everywhere short-lived and weak; keep the credential that's powerful in one safe place and revocable." Hold that thought — it's the lever that makes the otherwise-scary stateless JWT model survivable.

## JWT: structure, claims, and local verification

The access token can technically be opaque — a random string the resource server must call the AS to interrogate — but in the microservices world it is overwhelmingly a **JWT (JSON Web Token)**, and understanding its structure is non-negotiable for anyone securing services. A JWT is three base64url-encoded parts joined by dots: `header.payload.signature`. The figure below lays them out, and the critical insight it encodes is that the first two parts are *not secret and not trusted on their own* — anyone can read them, and only the third part, the signature, makes them believable.

![A stacked diagram of a JWT showing header with algorithm and key id then payload of claims like subject audience expiry and scope then a signature signed by the authorization server key verified against JWKS public keys with mandatory checks of expiry audience and issuer before the token is trusted](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-3.webp)

The **header** declares the signing algorithm (`alg`, e.g. `RS256` — RSA with SHA-256) and a key ID (`kid`) identifying *which* of the issuer's keys signed this token. The **payload** is a JSON object of *claims* — statements about the token. The registered claims you must know: `sub` (subject — the user's stable ID), `iss` (issuer — the URL of the AS that minted it), `aud` (audience — which resource servers this token is *for*), `exp` (expiry, a Unix timestamp), `iat` (issued-at), and `nbf` (not-before). Beyond those, the token typically carries `scope` (space-separated permissions like `orders:read orders:write`) and custom claims your domain needs (`roles`, `tenant_id`). The **signature** is a cryptographic signature over the encoded header and payload, produced with the AS's private key. Here is a decoded example — note that the token itself is shown truncated per security practice, because a real one is three long base64url segments you should never paste into a blog:

```json
// The JWT as transmitted (truncated — never paste a real one):
//   eyJ...HEADER... . ...PAYLOAD... . ...SIGNATURE...
//
// Decoded header:
{ "alg": "RS256", "typ": "JWT", "kid": "shopfast-key-2026-01" }

// Decoded payload (the claims):
{
  "iss": "https://auth.shopfast.example",
  "sub": "user_8f3a91",
  "aud": ["orders-api", "payment-api"],
  "scope": "orders:read orders:write payment:charge",
  "roles": ["customer"],
  "iat": 1750000000,
  "exp": 1750000900            // 15 minutes after iat
}
```

The reason JWTs took over the microservices world is the property the figure calls out: **any service can verify a JWT locally, without a network call back to the authorization server.** The AS signs tokens with a *private* key and publishes the corresponding *public* keys at a well-known endpoint — the **JWKS** (JSON Web Key Set) URL, conventionally `https://auth.shopfast.example/.well-known/jwks.json`. Any resource server fetches that public key set once, caches it, and from then on can verify a token's signature with pure local computation: decode the header, find the public key matching the `kid`, verify the signature over `header.payload`, and if it checks out, the claims are trustworthy. No round-trip, no shared secret, no dependency on the AS being up for *every* request. In a system doing tens of thousands of requests per second across dozens of services, this stateless local verification is the difference between an auth server that's a quiet key-publisher and one that's a synchronous bottleneck on every single call. We'll put real latency numbers on that difference shortly, because it is the whole reason this model dominates.

### Verifying correctly: signature is necessary but not sufficient

The single most common JWT security bug is verifying the *signature* and stopping there, as if a valid signature meant a valid token. It does not. A correctly-signed token can still be the wrong token for *this* service, or expired, or issued by the wrong authority. Verification has at least four mandatory checks, and skipping any of them is a vulnerability. Here is JWT-validation middleware in Go that does it properly, the kind you'd put in every service:

```go
// jwtmw.go — validate a JWT locally against the AS's JWKS.
// Uses github.com/MicahParks/keyfunc for cached JWKS + golang-jwt.
package authmw

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
)

const (
	expectedIssuer   = "https://auth.shopfast.example"
	expectedAudience = "orders-api" // THIS service's audience, not a wildcard
)

// jwks is fetched once at startup and refreshed in the background.
func NewMiddleware(jwksURL string) (func(http.Handler) http.Handler, error) {
	jwks, err := keyfunc.NewDefault([]string{jwksURL}) // caches keys, auto-refreshes on rotation
	if err != nil {
		return nil, err
	}

	parser := jwt.NewParser(
		jwt.WithValidMethods([]string{"RS256"}), // pin the alg — never accept "none" or HS256
		jwt.WithIssuer(expectedIssuer),          // checks iss
		jwt.WithAudience(expectedAudience),      // checks aud — token must be FOR us
		jwt.WithExpirationRequired(),            // exp must be present
		jwt.WithLeeway(30*time.Second),          // small clock-skew tolerance
	)

	mw := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			if raw == "" {
				http.Error(w, "missing token", http.StatusUnauthorized)
				return
			}
			claims := jwt.MapClaims{}
			// Parse verifies SIGNATURE via jwks, plus iss/aud/exp via parser options.
			if _, err := parser.ParseWithClaims(raw, claims, jwks.Keyfunc); err != nil {
				http.Error(w, "invalid token", http.StatusUnauthorized)
				return
			}
			// Pass verified identity + scopes downstream via request context.
			ctx := withIdentity(r.Context(), claims["sub"].(string), scopesOf(claims))
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
	return mw, nil
}

var errNoScope = errors.New("scope not granted")
```

Four things in that code are load-bearing and worth dwelling on. **Pin the algorithm** (`WithValidMethods([]string{"RS256"})`): the most infamous JWT exploit is the `alg: none` attack, where an attacker strips the signature and sets the algorithm to `none`, and a naive library accepts an *unsigned* token as valid. A close cousin is the RS256-to-HS256 confusion attack, where an attacker re-signs the token with HS256 using the *public* key as the HMAC secret; if your library accepts HS256 with a key it thinks is for RS256, the forgery passes. Pinning the algorithm to exactly what you expect kills both. **Check `aud`** (`WithAudience("orders-api")`): a token minted for the orders API must not be accepted by the payment API — the audience is what stops a token for service A from working on service B, and we'll build a whole section on why that matters. **Require and check `exp`** so an expired token is rejected. And **never trust the header's `kid` blindly** — look it up in the JWKS you fetched from the real issuer; a forged token can claim any `kid`, but it can't produce a signature that verifies against the real public key.

### The stateless appeal and its dark side: revocation

Local JWT validation is gloriously fast and decoupled, and that is precisely why it has a sharp edge: **once a JWT is issued, you cannot easily un-issue it.** Because every service validates the token by itself, with no call to a central authority, there is no central place that gets consulted to ask "is this token still good?" If a user logs out, if you detect a token was stolen, if an admin revokes someone's access — none of those events reach the services that are happily validating the token locally until it expires on its own. The token remains cryptographically valid until its `exp`. This is the fundamental tension of stateless auth: the property that makes it fast (no central check) is exactly the property that makes revocation hard. You have four ways to deal with it, and each is a trade-off:

1. **Short TTLs plus refresh** — the default and the right starting point. Make access tokens expire in 5–15 minutes so a leaked token is valid only briefly, and use refresh tokens (revocable centrally at the AS) to mint new ones. Revocation becomes "wait at most one TTL." This is cheap and stateless, at the cost of more frequent token refreshes.
2. **A token blacklist / denylist** — keep a small, fast store (Redis) of revoked token IDs (`jti` claim), and have each service check membership before trusting a token. This gives near-instant revocation but reintroduces a central lookup on every request — partially defeating the stateless win — and the store becomes a hot dependency.
3. **Introspection** — instead of local validation, services call the AS's introspection endpoint (RFC 7662) to ask "is this token active?" This is essentially treating the JWT as opaque. You get instant revocation and central control at the cost of a network call per request (or per cache-TTL), reintroducing the AS as a synchronous dependency.
4. **A hybrid** — local validation for the common case, with a short-TTL cached introspection or denylist check for high-value operations only (issuing a refund, changing a password). You pay the central-check cost only where revocation latency actually matters.

The senior move is almost always **start with short TTLs and refresh**, because it keeps the stateless model and bounds the exposure window to something you can defend in a post-mortem, and add a denylist or selective introspection only for the genuinely high-stakes actions. We'll quantify the revocation window next, because "how long is a leaked token valid?" is a question you will be asked in an incident review, and the answer is a direct function of your TTL choice.

#### Worked example: the revocation window with a 5-minute versus a 1-hour TTL

A ShopFast user clicks "log out" at 14:00:00 on a shared library computer, walks away, and an attacker sits down at 14:00:30 — the browser session is gone, but suppose the attacker had captured the access token from an earlier shoulder-surf or a compromised browser extension. How long can the attacker use that captured access token? With stateless JWTs and no denylist, the answer is **until the token's `exp`, regardless of the logout**, because logout invalidates the *refresh* token and the session at the AS, but cannot reach out and kill an already-issued access token sitting in an attacker's hands.

With a **1-hour TTL**, if the token was issued at 13:20:00 it expires at 14:20:00, giving the attacker a window of up to 20 minutes after the 14:00 logout, and in the worst case (token issued at 13:59:59) a full 59 minutes and 59 seconds. With a **5-minute TTL**, a token issued at 13:56:00 expires at 14:01:00 — the attacker has at most one minute after logout, and the absolute worst case is 4 minutes and 59 seconds. The exposure shrinks by roughly 12×. The cost of buying that shrink is concrete: a 5-minute access TTL means the client silently refreshes its token roughly every 5 minutes instead of every hour — about 12 token-endpoint calls per active hour per user instead of one. For a system with 50,000 concurrently active users, that's the difference between ~50,000 refreshes/hour and ~600,000 refreshes/hour hitting the AS's token endpoint. That's the trade: the AS's token endpoint must handle 12× more refresh traffic to buy a 12× smaller leak window. For most systems that's an easy yes — the token endpoint is cheap and the leak-window reduction is worth far more than the extra refreshes — but it's a real capacity line item you size for, not a free lunch. For the genuinely sensitive actions (a password change, a large refund), layer a denylist check so logout is *instant* for those paths and you don't depend on the TTL at all.

## Token propagation: carrying identity through the call graph

So the gateway has authenticated the user and holds a valid JWT. The user clicks "Place Order," which means the order service must call the payment service, which must call the fraud-check service. **How does the user's identity travel from the gateway down through that chain?** This is the question that defines microservices auth, and getting it wrong is how the ShopFast refund happened. There are three strategies, and the decision matrix below lays them against the properties that matter — downstream privilege, blast radius if a token leaks, latency, implementation cost, and whether the channel underneath needs mTLS.

![A matrix comparing how to propagate identity across forward as is token exchange and trusted header against downstream privilege blast radius if leaked added latency implementation cost and whether mTLS is needed underneath](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-6.webp)

**Forward the token as-is** is the simplest: the gateway passes the user's JWT to the order service, which passes the *same* token to the payment service, and so on down the chain. Every service sees the original user token, validates it locally, and authorizes against it. The appeal is obvious — zero extra latency, trivial to implement (copy one header), and every service genuinely knows who the end user is. The danger is equally real: the token that the payment service receives is *the full-privilege user token*, scoped for everything the user can do across all services. If the payment service is compromised, or logs the token, or has a bug that forwards it somewhere it shouldn't, the attacker now holds a token that works against *every* service the user can reach, not just payment. The blast radius of a leak is the union of everything in the token's scope. This is the strategy ShopFast was using, and it's why the over-powerful token reached the payment service unchecked.

**Token exchange** (RFC 8693, the *OAuth 2.0 Token Exchange* extension) is the senior answer to that blast radius. Before the order service calls the payment service, it asks the authorization server to *exchange* the broad user token for a new, **narrower** token — one scoped to only `payment:charge`, with `aud: payment-api` only, and a short TTL — that still carries the user's identity (`sub`) but none of the user's other powers. The payment service now receives a token that can do exactly one thing on exactly one service on behalf of exactly that user. If the payment service is compromised, the stolen token is useless against the order service (wrong audience) and can't issue refunds (wrong scope). The cost is one extra call to the AS per exchange — a few milliseconds, mitigated by caching exchanged tokens for their (short) lifetime — and the operational weight of configuring exchange policies. Here's the exchange request the order service makes:

```bash
# Order service narrows the user token before calling payment (RFC 8693).
# The order svc authenticates to the AS with its OWN client credentials,
# and presents the user's token as the subject_token.
curl -s -X POST https://auth.shopfast.example/oauth2/token \
  --cert /etc/certs/order-svc.pem --key /etc/certs/order-svc.key \
  -d grant_type=urn:ietf:params:oauth:grant-type:token-exchange \
  -d client_id=order-service \
  -d client_secret='<client-secret>' \
  -d subject_token='<incoming-user-access-token>' \
  -d subject_token_type=urn:ietf:params:oauth:token-type:access_token \
  -d audience=payment-api \
  -d scope='payment:charge'        # narrowed: NOT payment:refund, NOT orders:*

# Response (token truncated per security practice):
# {
#   "access_token": "eyJ...HEADER... . ...PAYLOAD... . ...SIGNATURE...",
#   "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
#   "token_type": "Bearer",
#   "expires_in": 120          // 2-minute TTL — only needs to outlive the call
# }
```

**Trusted internal identity header** is the third option: the gateway validates the user token, then strips it and injects a simple internal header — say `X-ShopFast-User: user_8f3a91` and `X-ShopFast-Scopes: payment:charge` — that downstream services trust. This is fast (no token to re-validate, no exchange call) and simple, but it is *only* safe if the internal network guarantees that header could only have come from the gateway. That guarantee is exactly what mTLS and a zero-trust mesh provide — if any pod could set `X-ShopFast-User` and be believed, a single compromised service could impersonate any user. This is why the matrix marks mTLS as *mandatory* underneath the trusted-header strategy: the header is an *assertion*, and an assertion is only as trustworthy as the proof that it came from a component allowed to make it. This is the precise hand-off to the previous post: [mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) is what makes a trusted internal identity header safe, by cryptographically guaranteeing the caller is the gateway and not some lateral-movement attacker. Use trusted headers *only* behind a service mesh that enforces caller identity; never on a flat network.

The figure below makes the contrast between forwarding and exchanging concrete by showing the call graph for ShopFast's checkout with a scope-narrowing exchange in the middle, so you can see exactly where the user token stops and the narrow token begins.

![A graph showing token propagation where the gateway validates the user JWT and passes it to the order service which performs an RFC 8693 token exchange at the auth server to obtain a narrow payment scoped token before calling the payment service which queries an OPA sidecar for an authorization decision](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-9.webp)

### Why "validate at every hop" is not paranoia

A tempting shortcut, and the one that caused the refund incident, is to validate the token *once* at the gateway and then let the internal network be trusted — services inside the cluster believe each other, no further checks. The figure below contrasts this "trust the network" posture with "verify at every hop," and the difference is the whole argument for zero trust applied to user identity.

![A before and after diagram contrasting trusting the network where the edge checks once and internal calls are unauthenticated so one breach forges any user against verifying at every hop where the edge validates the JWT and each service re-checks audience and scope so a breach blast radius is bounded](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-4.webp)

The "trust the network" model has a single, catastrophic failure mode: it assumes the network perimeter is impenetrable, that nothing hostile is ever *inside*. That assumption fails the moment any one service is compromised — a vulnerable dependency, a leaked credential, a misconfigured pod — because once an attacker has a foothold *inside* the trusted zone, they can call any service as any user, since nothing inside checks. The blast radius of one breached service is the entire system. "Verify at every hop" — validate the token's signature, audience, and scope at *each* service — turns that single point of catastrophe into a contained incident: a compromised payment service still can't call the order service as an arbitrary user, because the order service checks the token's audience and finds it isn't `orders-api`. This is precisely the **zero-trust** principle from the previous post applied to *user* identity rather than *service* identity: never trust based on network location; always verify the credential at the point of use. The cost is a few microseconds of local JWT validation per hop — which, as the next worked example shows, is genuinely negligible compared to the safety it buys.

#### Worked example: local validation versus introspection at 10,000 rps

ShopFast's order service handles 10,000 requests per second at peak. Each request carries a user token that must be validated. Consider the two validation strategies and their latency and dependency profiles.

**Local JWT validation.** Verifying an RS256 signature is a single public-key operation. On a modern CPU, RSA-2048 signature *verification* (the cheap direction — verification is far cheaper than signing) takes on the order of 30–100 microseconds, and the surrounding base64-decode and claim checks add a few more. Call it ~0.1 ms of CPU per validation, fully local, no network. At 10,000 rps that's 10,000 verifications per second — roughly one CPU-core's worth of work if a verification is ~0.1 ms (10,000 × 0.1 ms = 1,000 ms = 1 core-second per wall-clock second), spread across the service's cores. There is *no* added p99 latency from a network round-trip and *no* dependency on the auth server being up: if the AS is down, already-issued tokens still validate fine because the JWKS public keys are cached locally. The auth server is a bottleneck for *issuing* tokens, never for *validating* them.

**Introspection on every request.** Now suppose instead each service calls the AS's introspection endpoint to validate every token. That's a network round-trip — even on a fast internal network, realistically 2–10 ms p50 and 20–50 ms p99 once you account for connection handling, the AS's own processing, and tail effects. At 10,000 rps, the order service alone generates 10,000 introspection calls per second *to the auth server*, and that's just one service — the payment, fraud, and shipping services would each add their own. The AS now has to handle the *sum* of every service's request rate as synchronous validation traffic, which for a modest fleet is easily hundreds of thousands of introspections per second, making the AS a single synchronous dependency on the critical path of every request in the system. If the AS slows down or falls over, *every* service's p99 degrades or every request fails. You can cache introspection results, but caching reintroduces exactly the revocation-staleness you were trying to avoid, so you've paid the latency-and-dependency cost to get back to roughly where local validation already was.

The numbers make the choice stark: local validation is ~0.1 ms and zero new dependencies; per-request introspection is ~2–10 ms p50 with a system-wide synchronous dependency on the AS. This is *why* the industry standardized on stateless JWT validation, and it's why the revocation problem is one we manage with short TTLs rather than solve by going back to per-request central checks. The 12× refresh-traffic cost from the earlier worked example is a tiny price next to turning your auth server into a synchronous bottleneck on every one of hundreds of thousands of requests per second.

## The confused deputy, concretely — and the fix

We can now name exactly what went wrong at ShopFast and fix it. The **confused deputy** is a class of privilege-escalation bug where a program with legitimate high privileges is tricked into misusing them on behalf of a less-privileged party. In our case the *deputy* is the payment service: it has the legitimate, high privilege to charge and refund cards. The *trick* is that it executed a refund on behalf of a user who had no refund right, because it authorized the call based on *who was asking* (the trusted order service) rather than *whether the end user was permitted*. The order service, in turn, forwarded a token far more powerful than the action needed. Two mistakes compounded: an over-broad token was propagated, and the receiving service authorized on caller identity instead of user rights.

![A before and after diagram contrasting a confused deputy where a user lacking refund rights has the order service forward an admin wide token and the payment service runs the refund without checking the user against the fix where token exchange narrows scope to payment charge only so the token lacks refund scope and the payment service authorization denies the refund](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-7.webp)

The figure shows both the bug and the fix. The fix has two independent layers, and you want *both* because defense in depth means no single check is the only thing standing between a user and a refund. The first layer is **token exchange to narrow scope** (the propagation fix from the last section): the order service exchanges the user's token for one scoped to exactly the operation it's delegating — and crucially, it can only request a *narrowing*, never a broadening. If the user's original token doesn't include a refund scope, the AS will refuse to mint an exchanged token with `payment:refund`. So even before the payment service does anything, the token it receives simply *cannot* authorize a refund. The second layer is **per-service authorization**: the payment service, on receiving the refund request, checks the *user's* rights against the *user's* identity in the token — not whether the caller is a trusted service. "Is `sub=user_8f3a91`, a `customer`, allowed to refund order `ord_55021`?" The answer is no, and the payment service denies the call regardless of how it got there. Either layer alone would have stopped the incident; both together make it a non-event. Here's the scope-and-audience check the payment service performs *before* any business logic:

```python
# payment_service/authz.py — scope + audience guard on the refund endpoint.
from fastapi import Request, HTTPException

REQUIRED_SCOPE = "payment:refund"
THIS_AUDIENCE  = "payment-api"

def require_scope(claims: dict, scope: str, audience: str) -> None:
    # 1) The token must be FOR this service. A token minted for orders-api
    #    must never authorize a payment action, even if forwarded here.
    aud = claims.get("aud", [])
    aud = aud if isinstance(aud, list) else [aud]
    if audience not in aud:
        raise HTTPException(403, "token audience does not include this service")

    # 2) The token must carry the specific scope for THIS action.
    granted = set(claims.get("scope", "").split())
    if scope not in granted:
        raise HTTPException(403, f"missing required scope: {scope}")

async def refund_endpoint(request: Request):
    claims = request.state.verified_claims      # set by JWT middleware
    require_scope(claims, REQUIRED_SCOPE, THIS_AUDIENCE)   # <-- the confused-deputy guard
    # Only now do we touch money. The check above is what the incident lacked.
    user_id = claims["sub"]
    ...  # business logic: verify the order belongs to user_id, then refund
```

Notice the audience check is doing double duty here, which brings us to the next foundational idea.

## Scopes and audiences: a token for A must not work on B

Two claims do the heavy lifting of *containing* a token's power, and confusing or omitting them is behind a large share of microservices auth bugs. **Audience (`aud`)** answers *which services is this token for?* **Scope** answers *what actions does it permit?* Together they bound a token to "these actions on these services," and enforcing both at every service is what makes propagation safe.

The audience check is the one juniors most often skip, and it's the most important for blast-radius control. Imagine ShopFast mints a single token with `aud: ["orders-api", "payment-api", "shipping-api"]` and full scope — a token that works everywhere. If the shipping service (perhaps the least-hardened, written by a junior team, with a vulnerable image-resizing dependency) is compromised and leaks its tokens, those tokens *also work against the payment service*, because the audience includes `payment-api`. The weakest service's compromise becomes the strongest service's breach. The fix is **audience segmentation**: mint tokens scoped to the narrowest audience the request actually needs, so a token the shipping service holds has `aud: shipping-api` *only* and is rejected outright by the payment service's validation (`WithAudience("payment-api")` fails). Token exchange is the mechanism that produces these narrow-audience tokens on the fly. The senior rule: **a token's audience should be the smallest set of services that genuinely must accept it**, ideally exactly one, and every service must reject tokens that don't name it in `aud`. This is why the JWT middleware earlier pinned `expectedAudience` to *this* service rather than accepting any audience — that one line is the difference between contained and system-wide blast radius.

Scopes do the orthogonal job of limiting *actions*. A token with `scope: orders:read` can read orders but not place or cancel them; one with `payment:charge` can charge but not refund. Scopes are *coarse* by design — they're about API surface ("can call the charge endpoint"), not about *which specific resources* ("can refund order 55021 but not 55022"). That finer, resource-level question is *authorization* proper, and scopes are not the right tool for it — you don't mint a scope per order. Keep scopes for coarse capability gating at the API boundary, and use real authorization logic (below) for resource-level decisions. The mistake is trying to encode everything in scopes, producing tokens with hundreds of fine-grained scopes that are unmaintainable and that leak your authorization model into your token format.

## Authorization models and where to enforce them

Scopes get you coarse capability gating. Real authorization — "may *this* principal do *this* action on *this* resource *right now*?" — needs a model and a place to enforce it. The two dominant models are RBAC and ABAC, and the right architecture usually combines them.

**RBAC (Role-Based Access Control)** assigns permissions to *roles* and roles to users. A ShopFast user is a `customer`; a support agent is a `support`; a finance team member is a `finance-admin`. Permissions attach to roles: `finance-admin` may refund any order; `support` may refund up to \$50; `customer` may refund nothing. RBAC is simple to reason about and audit ("who can refund? the finance-admin role; who has that role? these eight people"), which is why it's the workhorse of most systems. Its limitation is rigidity: real policies often depend on *attributes* of the request that roles can't capture — "a support agent may refund an order *only if the order is theirs to handle, the amount is under \$50, and it's within 30 days of purchase*." Encoding every such combination as a role produces a *role explosion* (`support-under-50-within-30-days`), which is unmaintainable.

**ABAC (Attribute-Based Access Control)** decides based on *attributes* of the subject, the resource, the action, and the environment, evaluated by a policy. "Allow refund if `subject.role == support AND resource.amount <= 50 AND now - resource.purchased_at <= 30 days AND resource.region == subject.region`." ABAC is far more expressive and avoids role explosion, at the cost of more complex policy and the need for the relevant attributes to be available at decision time. Most mature systems use **RBAC for the coarse decisions and ABAC for the fine ones** — roles gate the broad capability, attribute rules handle the contextual nuances.

The second, equally important question is *where* to enforce authorization. The matrix below compares the three placements — gateway-only, per-service ad hoc, and centralized policy with OPA — against the properties that matter.

![A matrix comparing where to enforce authorization across gateway only per service ad hoc and centralized OPA against stopping unauthenticated requests early enforcing fine grained rules resisting internal breach policy consistency and operational overhead](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-8.webp)

**Gateway-only** authorization is coarse and necessary but never sufficient. The gateway can cheaply reject unauthenticated requests and do broad scope checks ("this token has *some* orders scope, let it through to the orders service"), which sheds load early and keeps obviously-bad traffic out. But the gateway doesn't know domain specifics — it can't know whether *this* user owns *that* order — and if it's the *only* place authorization happens, you're back to the trust-the-network failure: anything past the gateway is unchecked. **Per-service ad hoc** authorization (each service has its own `if user.role == ...` checks) fixes the fine-grained gap and resists internal breach, but it has a different disease: the policy is scattered across forty repositories in five languages, it drifts (the refund rule in payment says \$50, the one duplicated in the support tool says \$100, and nobody knows which is right), and auditing "who can refund?" means reading forty codebases. **Centralized policy with OPA** keeps the fine-grained enforcement *at* each service but moves the policy *definition* to one place, as code.

The senior answer the matrix points to is the combination: **coarse authentication and broad authorization at the gateway** (shed bad traffic early, cheaply) **plus fine-grained authorization at every service** (the real domain decisions), **with the policy defined centrally** rather than scattered. That last part — policy as code — is what OPA gives you.

### Policy as code with Open Policy Agent

**Open Policy Agent (OPA)** decouples *policy decisions* from *policy enforcement*. You write authorization rules in a declarative language called **Rego**, ship them to a lightweight OPA engine that runs as a sidecar next to each service (or as a library), and your service code makes one call — "given this input, allow or deny?" — instead of embedding the rules. The policy lives in one versioned repository, is tested in CI like any code, and is the single source of truth for "who can do what," while enforcement still happens locally at each service (fast, no central decision-service bottleneck — OPA evaluates locally with policy bundles distributed to it). Here is the ShopFast refund policy in Rego — the rule that would have denied the incident's refund:

```rego
# refund.rego — ShopFast refund authorization policy (single source of truth).
package shopfast.payment.refund

import future.keywords.if
import future.keywords.in

default allow := false

# A customer may never refund — only request one. This rule alone blocks
# the confused-deputy incident: user_8f3a91 is a customer.
allow if {
	"finance-admin" in input.subject.roles
}

# A support agent may refund their own region's orders, under $50,
# within 30 days of purchase (ABAC: role + attribute conditions).
allow if {
	"support" in input.subject.roles
	input.resource.amount <= 50
	input.resource.region == input.subject.region
	time.now_ns() - input.resource.purchased_at_ns <= ((30 * 24) * 3600) * 1000000000
}

# Explain the denial for audit logs and 403 bodies (no silent failures).
reason := "customer role cannot issue refunds" if {
	"customer" in input.subject.roles
	not allow
}
```

And the service-side call into OPA — note how thin the enforcement code is, because the *logic* lives in Rego:

```python
# payment_service/refund.py — ask OPA, then act. No policy logic in app code.
import httpx

OPA_URL = "http://localhost:8181/v1/data/shopfast/payment/refund/allow"  # sidecar

async def authorize_refund(claims: dict, order: dict) -> None:
    decision_input = {
        "input": {
            "subject":  {"id": claims["sub"], "roles": claims.get("roles", []),
                          "region": claims.get("region")},
            "resource": {"id": order["id"], "amount": order["amount"],
                          "region": order["region"],
                          "purchased_at_ns": order["purchased_at_ns"]},
            "action":   "refund",
        }
    }
    resp = await httpx.AsyncClient().post(OPA_URL, json=decision_input, timeout=0.05)
    if not resp.json().get("result", False):     # OPA said deny
        raise PermissionError("refund not authorized by policy")
```

The win is structural. When the finance team changes the refund policy ("support agents can now refund up to \$75"), you edit *one* Rego file, test it, and roll it out to every OPA sidecar through bundle distribution — no code change in any service, no risk of the rule being right in payment and wrong in the support tool. The policy is auditable (one file answers "who can refund?"), testable (Rego has a unit-test framework), and consistent (one source, distributed everywhere). The cost the matrix names is real — running an OPA sidecar next to every service adds memory and a deploy artifact, and there's a learning curve to Rego — which is why you reach for OPA when your authorization rules are complex enough that scattering them across services has become a liability, not on day one with three roles and two rules.

## API keys and mTLS for partner and service authentication

Not every caller is a logged-in human with an OIDC session. Two other principals need handling, and conflating them with user auth is a common mistake. **Partner / third-party API access** — another company's system calling ShopFast's API — is typically handled with **API keys** (a long random string identifying the partner) or, better for higher-value integrations, OAuth2 client-credentials with the partner as a registered client. API keys are simple and ubiquitous but weak: they're long-lived bearer credentials with no built-in expiry, often pasted into config and leaked, and they identify the *partner* but carry no fine-grained scope unless you build that mapping yourself. Treat an API key as a *coarse* identifier, gate it at the gateway, rate-limit per key, rotate them, and never let an API key alone authorize a high-value action without further checks. For anything beyond low-stakes read access, prefer client-credentials with proper scopes and short-lived tokens over a raw API key.

**Service-to-service authentication** — ShopFast's own services proving their identity to each other — is the domain of the previous post: **mTLS** gives each service a cryptographic identity (a certificate) that the receiving service verifies, so the *caller service* is authenticated at the transport layer. The key mental model to keep straight is that mTLS and user tokens authenticate *different principals on the same request*: mTLS says "this call came from the legitimate order service over a verified channel," and the JWT says "this call is on behalf of user_8f3a91 who is allowed to do this." You need both because each answers a question the other can't. A call can be from a legitimate service (mTLS passes) but on behalf of a user not allowed to do the action (authz fails), or carry a valid user token (JWT passes) but arrive from a service that has no business making it (mTLS catches a service that shouldn't be calling). The client-credentials grant from earlier is how a service gets a *token* representing itself when it needs to call an API that expects a token rather than just a certificate — and the best practice is to bind that client-credentials token to the service's mTLS identity rather than a shared secret, so there's no long-lived secret to leak. The clean separation of concerns: mTLS authenticates the *channel and the calling workload*; the JWT authenticates and authorizes the *user*; OPA decides the *fine-grained permission*. For the full treatment of the service-identity half, see [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust).

## The full sequence: a login and a checkout, end to end

Let's put every piece together by tracing one real ShopFast request from login to a charged card, because the abstractions only cohere when you watch them in sequence. The timeline below is the canonical happy path, and every event in it corresponds to a mechanism we've built.

![A timeline of a login and token propagation sequence showing user login with code and PKCE then the authorization server issuing a JWT with a fifteen minute time to live then the gateway validating against JWKS then the order service re verifying the audience then a token exchange to narrow the scope and finally the payment service authorizing through OPA](/imgs/blogs/authentication-and-authorization-oauth2-jwt-token-propagation-5.webp)

At **T+0** the user clicks "Log in" and the SPA starts the authorization-code-plus-PKCE flow — generating a verifier, redirecting to the AS, where the user authenticates with password and MFA on the AS's own pages. At **T+120ms** (after consent and the back-channel code exchange) the AS issues the token set: a 15-minute access token with `aud: orders-api`, scope `orders:read orders:write payment:charge`, `sub: user_8f3a91`; a refresh token in an HTTP-only cookie; and an ID token the SPA uses to render the user's name. At **T+125ms** the user clicks "Place Order"; the request hits the gateway, which validates the access token's signature against the cached JWKS, checks `iss`, `aud`, and `exp`, and does a coarse scope check before routing to the order service. At **T+130ms** the order service receives the request *and re-validates the token itself* — verify-at-every-hop — confirming `aud` includes `orders-api` and the scope permits placing an order. At **T+135ms**, before calling payment, the order service performs an RFC 8693 **token exchange**, swapping the broad user token for a narrow one scoped to `payment:charge` with `aud: payment-api` only and a 2-minute TTL. At **T+140ms** the payment service receives that narrow token, validates it (audience and scope both check out for a *charge*), and queries its **OPA sidecar** for the fine-grained decision — "may user_8f3a91 charge this card for this order?" — which returns allow. The card is charged. The whole identity machinery added a handful of milliseconds and a few CPU-microseconds of signature verification per hop, and in exchange it produced a request where every service independently verified the user, no service held more power than its task, and the confused-deputy path was closed by construction.

Now stress-test it, because the happy path is the easy part and the senior earns their title on the failure paths.

## Stress-testing the design

**A token leaks — how long is it valid?** An access token is captured (logged by accident, sniffed on a compromised client, extracted from a browser). With our 15-minute TTL and stateless validation, the answer is *up to 15 minutes from issuance*, and logout does not shorten that for the already-issued access token (it kills the refresh token and AS session). The exposed actions are bounded by the token's scope and audience — if it's a narrow exchanged token, the attacker can `payment:charge` on `payment-api` only, not refund and not touch orders. The mitigations are the ones we built: short TTLs bound the time; narrow scope and audience bound the actions; and for high-value operations a denylist check on `jti` makes revocation instant. The honest senior framing: you *cannot* make a leaked stateless token instantly worthless without giving up statelessness, so you make it *short-lived and narrow* so that "leaked" means "minutes of one capability on one service" rather than "everything until expiry." The earlier worked example quantified this — 5-minute TTL shrinks the window ~12× versus an hour.

**A service forwards a too-powerful token.** This is the confused deputy, and we've closed it two ways: token exchange means the order service *can't* forward refund power it was never asked to delegate (the AS refuses to broaden scope on exchange), and per-service authorization means the payment service checks the *user's* rights via OPA regardless of what token arrived. The stress test here is "what if a developer forgets the exchange and forwards the raw token anyway?" — and the answer is that the second layer catches it: the payment service's OPA query still asks "may this *customer* refund?" and gets a no. Defense in depth means a single forgotten step is a degradation, not a breach. The way you *find* the forgotten exchange before an incident is to make the audience check strict everywhere and alert on tokens arriving with broader audience than the receiving service needs.

**The auth server is down — do all logins fail?** This is the question that reveals whether you understood the stateless model. *New logins* fail — you cannot mint a token without the AS, and you cannot refresh an expired token without it, so users with expired tokens are locked out and new users can't log in. But *existing valid tokens keep working*, because validation is local against cached JWKS and needs no AS round-trip. A user who logged in five minutes before the AS went down can keep using ShopFast until their token expires; only then are they stuck. This is a genuine and important resilience property: the AS being down degrades *authentication* (getting new identity) but not *authorization of already-issued identity*, so an AS outage is a slowly-worsening degradation (more and more users hit expiry) rather than an instant total outage. Contrast this with the introspection model, where an AS outage fails *every* request immediately because every request needs the AS — which is the architectural reason, beyond latency, that stateless local validation is the right default. To harden further: run the AS highly available (it's now a critical dependency for the login flow), cache JWKS aggressively with a long TTL so key fetches don't depend on AS availability, and consider lengthening access-token TTL slightly under known AS-maintenance windows to widen the grace period — a deliberate, temporary trade of revocation window for resilience.

**The JWKS key rotates mid-flight.** The AS rotates its signing key (good hygiene). Tokens signed with the old key are still valid until they expire, and tokens signed with the new key carry a new `kid`. A service that cached only the old key would reject the new tokens. The fix is built into good middleware (the `keyfunc` library above auto-refreshes JWKS on encountering an unknown `kid`): publish both old and new keys in the JWKS during an overlap window at least as long as the max token TTL, so every in-flight token can still be verified. Rotate keys, but overlap them — the same lesson as rotating any credential without downtime, covered for the general case in [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management).

## Optimization: making it production-grade

The auth layer touches *every* request, so its overhead and reliability are first-order concerns, not afterthoughts. Four optimizations matter, each with a measurable win.

**Local validation over introspection.** This is the biggest lever and we've quantified it: ~0.1 ms local CPU versus 2–10 ms p50 (and 20–50 ms p99) per-request network introspection, and — more importantly — zero new synchronous dependency versus turning the AS into a system-wide bottleneck handling the *sum* of every service's request rate. At 10,000 rps per service across a dozen services, that's the difference between an AS that publishes keys quietly and one that must survive hundreds of thousands of synchronous validations per second. Default to local validation; reach for introspection only on the rare high-value path where instant revocation justifies the cost.

**JWKS caching.** The public keys change only on rotation (days to months apart), so fetching them on every validation would be absurd — yet a naive integration does exactly that. Cache the JWKS in memory with a TTL of minutes to hours and refresh in the background, falling back to a fetch only on an unknown `kid`. The measurable win: a JWKS fetch is a network round-trip (~5–20 ms); caching turns 10,000 fetches/sec into roughly one fetch per cache-TTL (a handful per hour), removing the AS from the per-request critical path entirely. This is the single configuration that makes "the AS is down but existing tokens still validate" *true* — without JWKS caching, you'd hit the AS on every validation and lose the resilience property.

**Short TTL plus refresh, tuned.** The TTL is a dial between revocation window and refresh load. Five to fifteen minutes is the sweet spot for most systems: short enough that a leak is minutes not hours, long enough that refresh traffic is manageable (the earlier worked example: ~12 refreshes/active-hour/user at 5 minutes). Measure your AS token-endpoint capacity and your active-user count, compute the refresh rate, and pick the shortest TTL your token endpoint can sustain. For high-traffic systems, ensure refreshes are spread (a little jitter) rather than synchronized, or you get a thundering-herd of refreshes every TTL boundary.

**OPA as a sidecar, with local evaluation.** Authorization decisions, like validation, must not become a network bottleneck. Running OPA as a *sidecar* (same pod, localhost call) with policy *bundles* distributed to it means each authorization decision is a local evaluation — sub-millisecond — with no call to a central policy service on the request path. The central policy *service* distributes bundles asynchronously; the *decisions* are local. The measurable win over a centralized decision service: a localhost OPA query is ~0.5–2 ms versus a cross-service policy call at network-round-trip cost, and again no shared synchronous dependency. The pattern mirrors JWT validation exactly: centralize the *definition* (keys, policy), distribute it, and *evaluate locally* on the hot path. That repeated pattern — central truth, local enforcement — is the architectural heart of doing identity at microservice scale.

## Case studies

**Auth0 and Okta: identity as a bought, not built, capability.** The dominant production pattern for the authentication half of this post is *don't build the authorization server* — use a managed identity provider like Auth0 (now part of Okta) or Okta's Customer Identity Cloud, or self-host Keycloak if you need to. These platforms implement OIDC, the authorization-code-plus-PKCE flow, JWKS publication, token exchange, and refresh-token rotation correctly, including the dozens of subtle security details (the `alg: none` defense, audience handling, MFA, breached-password detection) that are easy to get wrong by hand. The lesson is the one this post's authn/authz split implies: authentication is a largely-solved, commodity capability you should buy, freeing your engineering effort for *authorization*, which is specific to your domain and where your real access-control logic lives. Teams that build their own AS almost always regret it — not because it's impossible, but because the maintenance burden of keeping a correct, secure, standards-compliant identity server is enormous relative to its differentiation value.

**Google's internal identity propagation (BeyondCorp and end-user context).** Google's BeyondCorp architecture is the canonical large-scale realization of zero trust for *user* access — access decisions are made per-request based on authenticated user and device identity rather than network location, which is exactly the "verify at every hop, never trust the network" principle of this post applied at enormous scale. Internally, Google's services propagate an *end-user context* through call chains so that a backend deep in the call graph can authorize against the originating user's identity rather than the calling service's, and decisions are made by checking that propagated identity at the point of use. The lesson that maps directly onto our design: at scale, identity is established at the edge and *carried as a verifiable assertion* through the call graph, and every service authorizes against the *end user*, not the immediate caller — the confused-deputy fix, institutionalized.

**Netflix's edge authentication and token handling.** Netflix has written publicly about evolving its edge authentication, including the deliberate choice to *terminate* externally-facing, long-lived, cookie-style credentials at the edge and propagate a different, internally-meaningful, shorter-lived token inwards — rather than letting the raw external credential flow through the internal call graph. This is precisely the trusted-edge-plus-narrow-internal-token pattern: the edge does the heavy authentication, then mints or exchanges an internal identity token that's appropriate for service-to-service propagation, keeping the powerful external credential from sprawling across the fleet. The lesson is the propagation discipline this post argues for: the credential the user holds and the credential that flows internally need not — and for blast-radius reasons *should* not — be the same token.

**The JWT-revocation lesson, as a genre.** Rather than a single named incident, the recurring industry lesson is the one our revocation section is built on: teams that adopted stateless JWTs with long TTLs and *no* revocation strategy discovered, during a credential-leak or compromised-account incident, that they had no way to forcibly log someone out before the token expired — a logout button that didn't actually end the attacker's access. The fix that the industry converged on is exactly our recommendation: short access-token TTLs (minutes) as the baseline revocation mechanism, refresh-token rotation and revocation at the AS for session control, and a `jti` denylist for the high-value paths where you need *instant* revocation. The lesson is to design the revocation story *before* the incident, not during it: "how do we kill an active session right now?" is a question your architecture must have an answer to on day one, and "wait up to one TTL" is only an acceptable answer if your TTL is small enough to survive a post-mortem.

## When to reach for this (and when not to)

The mechanisms in this post are not all equally necessary at every scale, and a senior calibrates the investment to the system. **Always** authenticate end users at the edge with OIDC and a managed identity provider, and **always** validate tokens (signature, audience, expiry) at the edge — this is table stakes the moment you have users, even with a single service. **Validate at every service** (verify-at-every-hop) the moment you have more than a couple of services and any internal call could be reached by a compromised peer — which is essentially always in a real microservices deployment; the only systems that can responsibly skip it are tiny ones where the entire fleet is one trust domain you'd lose entirely in any breach anyway.

**Reach for token exchange and narrow audiences** when you have services of differing sensitivity (a payment service next to a low-stakes recommendation service) and the blast radius of the powerful service receiving an over-broad token is unacceptable — which is most systems handling money or PII. For a handful of equally-trusted internal services, forwarding the token is a defensible simplification *if* every service still validates audience and scope; the exchange complexity earns its keep when you have a clear privilege gradient across services. **Reach for centralized OPA** when your authorization rules have outgrown a handful of role checks — when policy is genuinely complex (attribute-based, multi-tenant, regulatory), changes often, and the cost of it drifting across scattered codebases has bitten you or clearly will. For three roles and five rules, OPA is over-engineering: put the checks in code, keep them well-tested, and adopt OPA when the policy's complexity and rate-of-change make scattered checks a liability. **Reach for trusted internal headers** *only* when you have a service mesh enforcing mTLS-verified caller identity underneath — otherwise it's a foot-gun, and you should forward or exchange tokens instead.

And when does the whole apparatus *not* apply? In a true monolith, end-user identity is an in-process session and none of the propagation machinery exists — which is one more entry on the ledger of [why a monolith can be the right call](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them) when you don't yet have the scale or team structure that forces a split. The distributed identity problem is a *cost* you take on when you distribute the system; if you haven't distributed it, don't pay the cost. This connects to the broader fallacy that distributing a system is free — every network boundary you add is also a *trust* boundary you now have to authenticate and authorize across, a point worth internalizing alongside the [fundamentals and fallacies of inter-service communication](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies). The auth overhead is part of the price of microservices, and like all the other prices in this series, you pay it deliberately, not by accident.

## Key takeaways

1. **Keep authentication and authorization on separate index cards.** Authn is "who are you," authz is "what may you do." The dangerous breaches are "authenticated but not properly authorized" — buy authentication, invest your engineering in authorization, because that's where your domain-specific bugs live.
2. **Authenticate once at the edge with OIDC and authorization-code-plus-PKCE; use client-credentials for service-to-service.** PKCE for all browser/mobile flows; the implicit and password grants are dead. The access token calls APIs, the refresh token mints new access tokens, the ID token tells the client who logged in — never confuse their jobs.
3. **A JWT is verifiable locally, but a valid signature is not a valid token.** Always also check `aud` (it's for *this* service), `exp` (not expired), and `iss` (right issuer), and pin the algorithm to defend against `alg: none` and RS256-to-HS256 confusion. Verify against the issuer's cached JWKS public keys.
4. **Stateless validation buys speed and decoupling and costs revocation.** Local validation is ~0.1 ms with no AS dependency; per-request introspection is ~2–10 ms p50 and makes the AS a system-wide bottleneck. Manage revocation with short TTLs (5–15 min) plus refresh as the baseline, and a `jti` denylist for high-value paths — design the "kill a session now" story before the incident.
5. **Propagate identity deliberately: forward, exchange, or trusted-header.** Forwarding is simplest but leaks the widest privilege; token exchange (RFC 8693) narrows scope and audience so a leak is "minutes of one capability on one service"; trusted internal headers are fast but safe *only* behind mTLS that proves the caller is the gateway.
6. **Verify at every hop — never trust the network.** The cost is microseconds of local validation; the benefit is turning a single breached service from a system-wide impersonation engine into a contained incident. This is zero trust applied to user identity, the companion to mTLS for service identity.
7. **Bound a token with audience and scope.** A token for service A must be rejected by service B (`aud`), and a token's scope must permit only the actions needed. Mint the narrowest audience that works — ideally exactly one service — so the weakest service's compromise isn't the strongest service's breach.
8. **Authorize coarse at the gateway and fine-grained at every service, with policy as code.** RBAC for broad capability, ABAC for contextual rules; define policy centrally (OPA/Rego) and enforce it locally as a sidecar — central truth, local evaluation, the same pattern as JWKS-cached JWT validation. Reach for OPA when scattered checks have become a liability, not on day one.
9. **The confused deputy is the bug to recognize on sight.** A privileged service acting on a less-privileged user's behalf without checking the user's rights. Close it with both narrowing (token exchange) and per-service user-level authorization — defense in depth so one forgotten step degrades rather than breaches.
10. **An auth-server outage degrades login, not already-issued identity — if you validate locally and cache JWKS.** Existing tokens keep working until expiry; only new logins and refreshes fail. That resilience property is a *reason* to choose stateless local validation over introspection, beyond the latency.

## Further reading

- *OAuth 2.0 Authorization Framework* (RFC 6749) and *OAuth 2.1* (the consolidating draft) — the canonical grant-type definitions and the modern guidance that mandates PKCE and retires the implicit and password grants.
- *Proof Key for Code Exchange* (RFC 7636) and *OAuth 2.0 Token Exchange* (RFC 8693) — the PKCE and token-exchange mechanisms this post leans on for safe user login and scope narrowing.
- *OpenID Connect Core* — the identity layer on top of OAuth2: the ID token, standard claims, and the discovery and JWKS endpoints.
- *JSON Web Token* (RFC 7519), *JWS* (RFC 7515), and *JWK/JWKS* (RFC 7517) — JWT structure, signing, and the public-key-set format you validate against.
- Open Policy Agent documentation and the Rego language guide — policy-as-code, the sidecar deployment model, and bundle distribution for centralized-definition, local-evaluation authorization.
- Sam Newman, *Building Microservices* (2nd ed.), and Chris Richardson, *Microservices Patterns* — the chapters on security, the Access Token pattern, and securing a service fleet.
- Sibling posts in this series: [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) (the east-west, service-identity companion to this north-south, user-identity post), [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) (where edge authentication lives), [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management) (how the AS client secrets and signing keys are stored and rotated), and the [fundamentals and fallacies of inter-service communication](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) (why every network boundary is also a trust boundary).
- Looking ahead, the cost of all this per-request verification feeds into [performance and cost optimization in microservices](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices), where the auth overhead becomes one of the latency and CPU line items you tune.

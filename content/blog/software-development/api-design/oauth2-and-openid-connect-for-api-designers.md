---
title: "OAuth 2.0 and OpenID Connect for API Designers"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Delegated access without sharing the password — the roles, the right flow per client (Authorization Code plus PKCE, Client Credentials, device code), the difference between an access token and an ID token, and exactly how a resource server validates a bearer token."
tags:
  [
    "api-design",
    "api",
    "oauth2",
    "openid-connect",
    "oidc",
    "authentication",
    "authorization",
    "jwt",
    "pkce",
    "security",
    "http",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/oauth2-and-openid-connect-for-api-designers-1.png"
---

A few years ago a finance team integrated a third-party accounting tool with our commerce platform. The tool needed to read payments so it could reconcile the books each night. The fastest way to ship it — the way someone always suggests in a hurry — was to create a service account, hand the tool that account's username and password, and let it log in like a human. It worked on Friday. By the following Wednesday the tool's vendor had a breach, our service account's password was in a dump, and that password was a skeleton key: it could read payments, issue refunds, change webhook URLs, and rotate other credentials. We had given an outside company the equivalent of our front-door key when all they ever needed was to peek through one window.

OAuth 2.0 exists to make that mistake impossible to make. Its entire purpose is **delegated access without sharing the password**: a way to let application X act on a user's behalf at API Y using a token that is *scoped* (it can only read payments, not issue refunds), *revocable* (you can kill it in one click without changing anyone's password), and *audience-bound* (a token minted for your payments API cannot be replayed against your refunds API). OpenID Connect, or OIDC, sits on top of OAuth and answers a different question — not "what may this app do?" but "who is this user?" — by adding an **ID token**, a signed statement of identity. The two are constantly confused, and confusing them is how you end up using an access token as proof of who someone is, which is one of the most common and most dangerous mistakes in API security.

This post is for the person on either side of that integration: the engineer *designing* an API that needs to accept delegated access, and the engineer *consuming* one and trying to pick the right flow. We will use the same running spine as the rest of this series — a fictional commerce platform's **Payments and Orders API** — and follow two concrete jobs through it: a third-party accounting app that wants `payments:read` on behalf of a logged-in merchant, and an internal nightly batch job that needs to read payments with no user involved at all. By the end you will be able to choose the correct grant for any client, explain why Authorization Code plus PKCE replaced the old implicit flow, validate a token at your resource server check by check, and avoid the handful of mistakes that cause most real OAuth incidents.

![a two-column comparison contrasting password sharing where an app holds the full account with delegated OAuth where the app holds only a scoped revocable token](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-1.png)

This is the same lens the whole series uses: an API is a contract and a product, not a function call. OAuth is the part of the contract that says *who is allowed to make a call, on whose behalf, and to do what* — and it has to keep that promise across years, across versions, and across callers you will never meet. If you want the bird's-eye framing first, start at the [series intro on the API as a contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) and keep the [API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) open as the checklist this all rolls up into.

## 1. The problem OAuth actually solves

Before we name a single grant type or token, get the problem crisp, because every design decision in OAuth falls out of it.

You have three parties who do not fully trust each other. A **user** owns some data (their payments history on your platform). An **application** — maybe one you wrote, maybe a third party — wants to do something with that data on the user's behalf. And there is an **API** that holds the data and must decide, on every request, whether to serve it. The naive solution is for the user to give the application their password so the application can log in *as* them. This is sometimes called the "password anti-pattern," and it is catastrophic for reasons that have nothing to do with how good or bad the application is:

- **It is all-or-nothing.** A password authenticates you for *everything* the account can do. There is no way to say "this app may read payments but may never issue a refund." The accounting tool that only needs `payments:read` gets the power to drain accounts.
- **It cannot be revoked surgically.** The only way to cut off one application is to change the password — which simultaneously cuts off every *other* application and the user themselves. So nobody ever revokes, and dead integrations keep their access for years.
- **It spreads the credential.** Every application that holds the password is a place the password can leak. You have multiplied your attack surface by the number of integrations.
- **It defeats multi-factor authentication.** If the app logs in with a password, where does the second factor go? It either gets bypassed or the app has to phish the user's MFA, which is worse.

The fix is a layer of indirection. Instead of giving the application the password, the user **delegates** a narrow, named permission to the application, and the application receives a **token** that represents exactly that permission — nothing more. The token is scoped (it names what it can do), it is time-limited (it expires), it is revocable (killing it does not touch the password), and it is bound to the specific API it was minted for. The application never sees the password. The user can grant ten apps ten different slices of access and revoke any one of them independently.

That is the whole game. OAuth 2.0 is the standard, defined in [RFC 6749](https://www.rfc-editor.org/rfc/rfc6749), for how that delegation happens over HTTP: how the user proves who they are to a trusted server, how they consent to a specific delegation, and how the application gets and uses the resulting token. Everything else — the grant types, the redirects, the token formats — is mechanism in service of that one idea.

> **The principle.** A bearer token is a *capability*, not an *identity*. Possessing it grants exactly the access it names, the way a hotel key card opens exactly your room for exactly the nights you paid for, and nothing else. The security model is therefore: mint the narrowest capability that does the job, make it expire, and be able to revoke it. A shared password is the opposite of a capability — it is identity, and identity is total. The single most important mental shift in OAuth is to stop thinking "log the app in" and start thinking "mint a capability for the app."

This reframing is why OAuth pairs naturally with the rest of API security. [Authentication](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) is about proving who is calling at all; [authorization](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions) is about deciding what that caller may do; OAuth is the protocol that lets a *third party* obtain a scoped slice of a *user's* authorization without becoming that user. Keep those three separate in your head and most of the confusion evaporates.

## 2. The four roles, named precisely

OAuth's specification language is precise, and using it precisely will save you hours of confused debugging. There are four roles. In an OAuth conversation, every party is exactly one of these.

![a layered stack showing the resource owner, the client, the authorization server, the resource server, and the scoped revocable token that flows between them](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-2.png)

- **Resource owner.** The human (usually) who owns the protected data and can grant access to it. On our platform, this is the merchant whose payments the accounting app wants to read. The resource owner is the only party who can *consent*.
- **Client.** The application requesting access on the resource owner's behalf. "Client" here does not mean "browser" or "front end" — it means the *application* in the OAuth sense. The accounting tool is a client. Our own mobile app is a client. An internal batch job is a client. Clients come in two flavors that matter enormously, which we will get to: **confidential clients** that can keep a secret (a server-side app), and **public clients** that cannot (a mobile app or single-page app, where any "secret" shipped to the device is extractable).
- **Authorization server.** The server that authenticates the resource owner, obtains their consent, and issues tokens. This is the trusted core. It is the only thing that ever sees the user's password. It might be your own identity service, or a provider like Okta, Auth0, Microsoft Entra ID, Google, or Keycloak. It exposes endpoints like `/authorize` (where the user logs in and consents) and `/token` (where the client exchanges its proof for tokens).
- **Resource server.** The API that holds the protected data and accepts tokens. This is our Payments and Orders API. Its job on every request is to validate the presented token and serve or deny accordingly. As an API designer, **this is the role you most often own**, so we will spend serious time on it.

The separation between the **authorization server** (mints tokens, holds login) and the **resource server** (validates tokens, holds data) is the structural heart of OAuth. It means your API never handles passwords, never runs a login screen, and never decides *how* a user proved who they are — it only checks tokens. Login complexity (passwords, MFA, social login, passkeys, account recovery) lives in one place, behind one team, and every API in your fleet benefits without re-implementing any of it. This is the same "separate the concern, own one wire contract" instinct that runs through [service-to-service security in a fleet](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust); OAuth is how that concern is solved at the user-delegation layer.

One subtlety: in many setups the authorization server and resource server are run by the same organization (we run both for our platform). That is fine. The *roles* are still distinct even when the *operators* are the same, and keeping the code paths separate — token minting here, token validation there — is what keeps the system clean as it grows.

A second subtlety trips up newcomers: the word "client." In everyday web talk a "client" is a browser and a "server" is the back end. In OAuth, "client" means **the application that wants the token**, and that application can be running on a server. Our accounting integration is a *confidential client* even though it has no UI of its own beyond a settings page — what matters is that it runs on the vendor's servers and can hold a secret. Our merchant-facing mobile app is a *public client* because it runs on a phone in someone's pocket, where any secret compiled into the binary can be pulled out with a disassembler. This single distinction — *can the client keep a secret?* — drives the choice between confidential-client mechanics (a client secret in the token request) and public-client mechanics (PKCE instead of a secret). When you read the OAuth specs, mentally substitute "the application requesting access" every time you see "client" and the prose will snap into focus.

Where do you register a client? Every authorization server has a **client registration** step (a console UI, an API, or [RFC 7591](https://www.rfc-editor.org/rfc/rfc7591) dynamic registration). At registration you declare the client's type (confidential or public), its allowed grant types, its allowed scopes, and — critically — its exact `redirect_uri`s. Registration is where a lot of the security is configured: a public client should not be allowed the Client Credentials grant; a client that only ever reads payments should not be registered for `refunds:write`. Treat the client registry as part of your security posture, not as a checkbox during onboarding.

## 3. OAuth is authorization; OIDC adds authentication

Here is the distinction that, once it clicks, prevents the single most common security bug in this whole area.

**OAuth 2.0 is an authorization framework.** Its output, the **access token**, answers exactly one question: *is the bearer of this token allowed to perform this action on this resource?* It deliberately says **nothing reliable about who the user is**. In fact, in the OAuth spec the access token is opaque to the client by design — the client is not supposed to look inside it. The token is for the *resource server* to interpret, not for the client to learn the user's identity from.

That created a real gap. People wanted to use "Log in with Google" or "Log in with GitHub" — they wanted *authentication*, proof of identity. Early on they hacked OAuth into doing this: the app would get an access token, then call some `/userinfo`-ish endpoint with it to learn the user's identity. This worked but was fragile and inconsistent across providers, and it invited a nasty class of bugs where an access token leaked from one context could be used to spoof a login in another.

**OpenID Connect (OIDC)** is a thin, standardized identity layer built *on top of* OAuth 2.0, defined in the [OIDC Core spec](https://openid.net/specs/openid-connect-core-1_0.html). It adds one new artifact and a few rules. The new artifact is the **ID token**: a JSON Web Token (JWT) issued by the authorization server that *proves who the user is* to the *client*. A JWT — pronounced "jot," defined in [RFC 7519](https://www.rfc-editor.org/rfc/rfc7519) — is a compact, signed token with three base64url parts (a header, a set of claims, and a signature) so the recipient can verify it was issued by who it says and was not tampered with. The ID token's claims include `sub` (a stable subject identifier for the user), `iss` (who issued it), `aud` (which client it is for), `exp` (when it expires), `iat` (when it was issued), and a `nonce` you will see in a moment, plus optional profile claims like `email` or `name`.

The two tokens have two completely different jobs, and conflating them is the bug:

| | Access token | ID token |
|---|---|---|
| **Question it answers** | What may the bearer do? | Who is this user? |
| **Audience (`aud`)** | The API (resource server) | The client (the app) |
| **Who reads it** | The resource server | The client |
| **Format** | Opaque or JWT (your choice) | Always a JWT (per OIDC) |
| **Sent to the API?** | Yes — in `Authorization: Bearer` | No — never sent to the API as proof of identity |
| **Purpose if misused** | (authorization) | (authentication) |

**The bug to never write: do not use an access token as proof of identity.** An access token's audience is the *API*, not your app. If your app receives an access token and tries to read the user's identity from it — or worse, accepts an access token from a *client* as a login assertion — you have opened a confused-deputy hole. An attacker can take an access token that was legitimately issued for *some other app* and present it to *your* app to log in as that user, because the access token never promised it was meant for your app. The ID token exists precisely to close this: it carries `aud = your client`, and an `azp` (authorized party) and `nonce` you verify, so it cannot be replayed from another app's context. **The rule is mechanical: identity comes from the ID token, API access comes from the access token, and you never substitute one for the other.**

To make the difference tangible, here is what a decoded ID token's claims look like next to a decoded access token's claims for the *same* login. The ID token is for the accounting app (the client); the access token is for the payments API (the resource server). They name different audiences on purpose.

```json
{
  "_comment": "ID token claims  -- audience is the CLIENT, read by the app",
  "iss": "https://auth.commerce.example.com",
  "sub": "merchant-9d12",
  "aud": "acct-app-7f3a",
  "nonce": "n-0S6_WzA2Mj",
  "email": "owner@merchant.example",
  "iat": 1718896400,
  "exp": 1718900000
}
```

```json
{
  "_comment": "Access token claims  -- audience is the API, read by the resource server",
  "iss": "https://auth.commerce.example.com",
  "sub": "merchant-9d12",
  "aud": "https://api.commerce.example.com/payments",
  "scope": "payments:read",
  "iat": 1718896400,
  "exp": 1718897300
}
```

Same `sub` (same human), same `iss` (same authorization server), but different `aud` and different payloads. The ID token carries profile data (`email`) and the replay-binding `nonce`; the access token carries `scope` and a *much* shorter `exp` (about 15 minutes here versus the ID token's session-length lifetime). If your code ever reaches for `email` out of the *access* token, that is the smell: profile data belongs in the ID token, and the access token may not even contain it. And if your code ever forwards an *ID* token to the API, the API's `aud` check will (correctly) reject it because the ID token's `aud` names the client, not the API. The two checks fail closed in exactly the right direction when you keep the tokens in their lanes.

There is one more reason designers conflate the tokens, and it is worth naming so you can resist it: convenience. It is tempting, in a small first-party app, to skip the ID token entirely and "just decode the access token to get the user." It works in the demo. It fails the day a second client appears, or the day a security review asks "what stops App B's access token from logging into App A?" Build it right from the start; the ID token costs you one extra `scope=openid` and one extra verification, and it is the difference between a system that composes and one that has to be re-secured later.

## 4. The grant types: which flow for which client

OAuth defines several "grant types" — also called flows — which are the different choreographies by which a client obtains tokens. The choice is not aesthetic. It is dictated by two facts about the client: *can it keep a secret?* and *is there a human present to interact?*

![a decision matrix of grant types across client type user presence and the use case each one fits](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-4.png)

| Grant type | Client type | Human present? | Use it for | Status |
|---|---|---|---|---|
| **Authorization Code + PKCE** | web, mobile, SPA | Yes (interactive) | Apps acting for a user | **Recommended default** |
| **Client Credentials** | confidential backend | No (machine-to-machine) | Service-to-service, batch jobs | Recommended |
| **Refresh token** | any (with prior grant) | No (silent renewal) | Getting new access tokens | Recommended |
| **Device Authorization (device code)** | TV, CLI, IoT | Yes (on a second screen) | Input-limited devices | Recommended |
| **Implicit** | browser-only (legacy) | Yes | (none — superseded) | **Deprecated** |
| **Resource Owner Password (ROPC)** | first-party only (legacy) | Yes | (none — superseded) | **Deprecated** |

The next four sections take these in turn. The headline, which you can act on immediately: **for anything with a user, use Authorization Code plus PKCE; for machine-to-machine with no user, use Client Credentials.** Those two cover the vast majority of real systems. The [OAuth 2.0 Security Best Current Practice, RFC 9700](https://www.rfc-editor.org/rfc/rfc9700), is the authoritative source for this guidance, and it is the document to cite when someone wants to use a deprecated flow.

## 5. Authorization Code plus PKCE: the right flow for users

This is the flow you will use the most, so we will derive it carefully and then walk a full wire example.

Start with the problem it solves. A user is sitting in a browser (or a mobile app). The client wants a token to act on their behalf at the API. The user must authenticate at the **authorization server** — and crucially, the *client must never see the password*. So the client cannot just collect a password and forward it. Instead, it must hand the user *off* to the authorization server, let the user log in there, and get back some proof.

The first idea (the old **implicit** flow) was: redirect the user to the authorization server, let them log in and consent, and have the authorization server redirect *back* with the **access token right in the URL fragment**. The client's JavaScript reads the token from the URL. Simple — and badly broken. The access token ends up in browser history, in server logs if the fragment leaks, in referrer headers, and is exposed to any script on the page. There is no proof that the party redeeming the response is the same party that started the flow. Implicit is deprecated for exactly these reasons.

The **Authorization Code** flow fixes the leakage by adding a second step. The authorization server does not return a token in the redirect — it returns a short-lived, single-use **authorization code**. The client then makes a *back-channel* (server-to-server, not through the browser) `POST` to the `/token` endpoint, exchanging the code for tokens. Tokens never travel through the browser URL. For a *confidential* client, the `/token` call is authenticated with the client's secret, so a stolen code is useless without the secret.

But what about *public* clients — mobile apps and SPAs that *cannot* keep a secret? Any secret you bake into a downloadable app can be extracted. For them, a stolen authorization code is dangerous: there is no secret to gate the exchange. This is exactly the gap **PKCE** fills.

### PKCE: Proof Key for Code Exchange

PKCE (pronounced "pixie"), defined in [RFC 7636](https://www.rfc-editor.org/rfc/rfc7636), binds the token exchange to the same client instance that started the flow, *without* needing a pre-shared secret. It works with a one-time, per-request secret the client generates itself:

1. The client generates a high-entropy random string called the **`code_verifier`** (43–128 characters).
2. It computes a **`code_challenge`** = base64url(SHA-256(`code_verifier`)). (There is a `plain` method too, but always use `S256`.)
3. It starts the flow by sending the `code_challenge` (and `code_challenge_method=S256`) to `/authorize`. The authorization server *remembers* the challenge alongside the code it issues.
4. When the client exchanges the code at `/token`, it sends the original `code_verifier`.
5. The authorization server hashes the `code_verifier` and checks it equals the stored `code_challenge`. Only the client that generated the verifier can produce a matching one, so a stolen code is worthless to anyone else.

The elegance: the *challenge* travels through the browser (where it might leak) but it is a one-way hash — useless on its own. The *verifier* travels only on the back-channel `POST`. An attacker who intercepts the code from the redirect cannot redeem it because they do not have the verifier, and they cannot derive the verifier from the challenge because SHA-256 is one-way. PKCE was originally designed for mobile apps but is now the recommendation for **every** Authorization Code flow, including confidential web clients and SPAs — it costs almost nothing and closes the code-interception attack regardless of client type.

![a timeline of the Authorization Code plus PKCE dance from building the challenge through redirect login consent code and the token exchange to the first API call](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-3.png)

#### Worked example: the accounting app gets `payments:read` via Authorization Code plus PKCE

Our merchant, logged into the third-party accounting app, clicks "Connect your commerce account." The accounting app is a confidential web client, but it uses PKCE anyway (best practice). Here is the full wire exchange.

**Step 1 — the client generates PKCE values and builds the authorization request.** The `code_verifier` is a random string; the `code_challenge` is its SHA-256 hash, base64url-encoded. The `state` is a random, unguessable value the client stores to defend against CSRF.

```http
GET /authorize?response_type=code
  &client_id=acct-app-7f3a
  &redirect_uri=https%3A%2F%2Faccounting.example.com%2Fcallback
  &scope=openid%20payments%3Aread
  &state=xR3k9Lq2mZ
  &code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
  &code_challenge_method=S256 HTTP/1.1
Host: auth.commerce.example.com
```

Note `scope=openid payments:read`: `openid` asks for an ID token (OIDC), and `payments:read` asks for the API capability. The browser is redirected to this URL.

**Step 2 — the user authenticates and consents at the authorization server.** This happens entirely on the authorization server's domain. The accounting app never sees the merchant's password or MFA. The merchant sees a consent screen: *"Accounting App wants to read your payments. Allow?"* They click Allow.

**Step 3 — the authorization server redirects back with a code (not a token).**

```http
HTTP/1.1 302 Found
Location: https://accounting.example.com/callback?code=SplxlOBeZQQYbYS6WxSbIA&state=xR3k9Lq2mZ
```

The accounting app's callback handler **first checks that `state` equals the value it stored.** If it does not match, this response did not originate from a flow this client started — reject it. This is the CSRF defense, and skipping it is a real, exploited bug.

**Step 4 — the client exchanges the code for tokens on the back channel.** This is a direct server-to-server `POST`, never through the browser. The confidential client authenticates itself (here with HTTP Basic using its client ID and secret) *and* supplies the `code_verifier`.

```http
POST /token HTTP/1.1
Host: auth.commerce.example.com
Content-Type: application/x-www-form-urlencoded
Authorization: Basic <base64 of client_id:YOUR_CLIENT_SECRET>

grant_type=authorization_code
&code=SplxlOBeZQQYbYS6WxSbIA
&redirect_uri=https%3A%2F%2Faccounting.example.com%2Fcallback
&code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk
```

The authorization server validates: the code exists and is unredeemed; the `redirect_uri` matches the one from step 1 exactly; the client authentication is valid; and SHA-256(`code_verifier`) equals the stored `code_challenge`. All pass.

**Step 5 — tokens come back.** Tokens are shown truncated here on purpose; never paste a full three-part JWT into anything that a secret scanner watches.

```json
{
  "access_token": "eyJhbGci...<header>.eyJzdWIi...<claims>.<signature>",
  "token_type": "Bearer",
  "expires_in": 900,
  "refresh_token": "8xLOxBtZp8...<opaque-refresh>",
  "id_token": "eyJhbGci...<header>.eyJpc3Mi...<claims>.<signature>",
  "scope": "openid payments:read"
}
```

The accounting app now has three things: an **access token** (valid 900 seconds = 15 minutes) it will send to the Payments API; an **ID token** it reads to learn the merchant's `sub` and email; and a **refresh token** it stores securely to get new access tokens without dragging the merchant back through login.

**Step 6 — the first API call.** The access token rides in the `Authorization` header as a *bearer* token (bearer = "whoever holds it can use it," which is exactly why it must travel only over TLS and never be logged).

```http
GET /v1/payments?limit=50 HTTP/1.1
Host: api.commerce.example.com
Authorization: Bearer eyJhbGci...<header>.eyJzdWIi...<claims>.<signature>
Accept: application/json
```

That is the complete dance. Notice what never happened: the accounting app never saw the merchant's password, the merchant's MFA stayed on the authorization server, the tokens never went through a browser URL, and a code intercepted in step 3 is useless without the step-4 verifier.

#### Worked example: classifying a redirect_uri mistake as a security break

Suppose someone "improves" the authorization server to do *prefix* matching on `redirect_uri` instead of *exact* matching, so that `https://accounting.example.com/callback` would also match `https://accounting.example.com.attacker.test/callback`. This is a breaking change to the security contract even though no API field changed shape.

- **Before (exact match):** an attacker who tricks the authorization server into redirecting elsewhere fails — the URI does not match, the server refuses.
- **After (prefix/substring match):** the attacker registers a look-alike host, starts a flow, and the authorization server happily redirects the *code* to the attacker's domain. Combined with any weakness in PKCE or `state`, the attacker can complete the flow. The fix in RFC 9700 is unambiguous: **redirect URIs must be compared by exact string match.** Register the full callback URL and match it character for character.

## 6. Client Credentials: machine-to-machine with no user

Now the second of our two jobs: the internal nightly batch job that reads payments to build a reconciliation report. There is **no user** here. No human to redirect, no consent screen, no resource owner. The "client" is acting *as itself*, not on anyone's behalf.

This is the **Client Credentials** grant. The client is a confidential client — it runs on a server you control, so it can hold a secret. It authenticates directly to the `/token` endpoint with its own credentials and receives an access token scoped to what it is allowed to do. No `/authorize`, no redirect, no ID token (there is no user identity to assert).

#### Worked example: the batch job gets a token and makes an audience-validated call

**Step 1 — the batch job requests a token.** It presents its client ID and secret and asks for the scope it needs and the audience it intends to call. Requesting an explicit `audience` (or `resource`, per [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707)) is what lets the authorization server stamp the token with the right `aud`, so the token cannot be replayed at a different API.

```http
POST /token HTTP/1.1
Host: auth.commerce.example.com
Content-Type: application/x-www-form-urlencoded
Authorization: Basic <base64 of batch-recon:YOUR_CLIENT_SECRET>

grant_type=client_credentials
&scope=payments%3Aread
&audience=https%3A%2F%2Fapi.commerce.example.com%2Fpayments
```

**Step 2 — the token comes back (no ID token, no refresh token).** Client Credentials tokens are typically not refreshed — when one expires, the job just requests another, since it has the secret. So no refresh token is issued.

```json
{
  "access_token": "eyJhbGci...<header>.eyJhdWQi...<claims>.<signature>",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "payments:read"
}
```

Decoded, the access token's claims look like this (the part the resource server cares about):

```json
{
  "iss": "https://auth.commerce.example.com",
  "sub": "batch-recon",
  "aud": "https://api.commerce.example.com/payments",
  "scope": "payments:read",
  "exp": 1718900000,
  "iat": 1718896400
}
```

Notice `sub` is the *client itself* (`batch-recon`), not a user — that is the signature of a Client Credentials token. And `aud` names exactly the payments API.

**Step 3 — the batch job calls the API, and the API validates the audience.** When the request arrives at the *refunds* API by mistake (someone copies a URL), the refunds resource server checks `aud`, sees it names the *payments* API, and rejects with `403`. The token simply does not work anywhere but where it was minted to work. That is the `aud` binding earning its keep.

```http
GET /v1/payments?status=succeeded&limit=200 HTTP/1.1
Host: api.commerce.example.com
Authorization: Bearer eyJhbGci...<header>.eyJhdWQi...<claims>.<signature>
Accept: application/json
```

A note on choosing Client Credentials versus a plain long-lived [API key](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls): an API key is a static secret with no built-in expiry, scope negotiation, or central issuance. Client Credentials gives you short-lived tokens, per-call scope, central revocation at the authorization server, and the same `aud` binding as your user flows — one consistent token-validation path across machine and user traffic. For internal service-to-service traffic on a fleet you may layer this with [mTLS and zero-trust networking](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust); the two are complementary, mTLS proving the *channel* and the token proving the *capability*.

#### Worked example: an API key breach versus a Client Credentials breach

Walk the same incident through both designs to see the difference concretely. Say the reconciliation job's credential leaks into a log file that a contractor can read.

- **Before (static API key):** the key has no expiry, so it is valid until a human notices and rotates it — often weeks. It typically carries broad access because static keys are rarely scoped tightly. There is no `aud`, so if the key is accepted by several APIs (a common shortcut), it works at all of them. Detection is hard because legitimate and malicious use look identical. The blast radius is "everything the key can touch, indefinitely."
- **After (Client Credentials):** the *token* in the log decayed to useless in roughly an hour (its `exp`). To get fresh tokens an attacker also needs the *client secret*, which lives only in your secret manager, not in the log. Even with a live token, `aud` confines it to the payments API; the refunds API rejects it on the `aud` check. And because tokens are minted centrally, you can revoke the client at the authorization server and every API stops honoring its new tokens at once. The blast radius is "read payments, for under an hour, at one API." Same leak, a fraction of the damage — and that reduction is purely structural, bought by choosing the right grant.

## 7. Refresh tokens and the device flow, briefly

Two more flows round out the practical toolkit.

**The refresh token flow** solves a tension. You want access tokens to be *short-lived* (15 minutes is common) so that a leaked one expires fast. But you do not want to drag the user through a login every 15 minutes. The refresh token bridges this: it is a long-lived credential the client exchanges, silently, for a fresh access token when the old one expires.

```http
POST /token HTTP/1.1
Host: auth.commerce.example.com
Content-Type: application/x-www-form-urlencoded
Authorization: Basic <base64 of acct-app-7f3a:YOUR_CLIENT_SECRET>

grant_type=refresh_token
&refresh_token=8xLOxBtZp8...<opaque-refresh>
&scope=payments%3Aread
```

The response is a new access token (and, with **refresh token rotation**, a new refresh token while the old one is invalidated — so a stolen refresh token is detectable: if both the attacker and the legitimate client try to use the same rotated token, the authorization server sees the reuse and revokes the whole chain). Because the refresh token is the long-lived crown jewel, it must be stored securely — server-side for confidential clients, in secure platform storage (Keychain, Keystore) for mobile, and **never** in browser-accessible storage for an SPA. For SPAs, the modern guidance is to keep refresh tokens in an `HttpOnly` cookie via a lightweight [backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), so JavaScript never touches them at all.

**The device authorization grant** (device code flow, [RFC 8628](https://www.rfc-editor.org/rfc/rfc8628)) handles input-limited devices: a smart TV, a CLI tool, an IoT box — anything where typing a password into the device itself is impractical or impossible. The device asks the authorization server for a code, displays a short user code and a URL (*"Go to example.com/device and enter ABCD-1234"*), and then **polls** the `/token` endpoint while the user completes login and consent on their phone or laptop. When the user finishes, the next poll returns the tokens. Our platform's CLI for merchants uses exactly this: `commerce login` prints a code, you approve it in your browser, and the CLI receives a token without ever handling your password. This is also why the AWS, GitHub, and Stripe CLIs feel the way they do.

The device flow has one wrinkle worth knowing: the polling has a *cadence the server controls*. The initial device-code response includes an `interval` (e.g. 5 seconds), and while the user has not yet approved, the `/token` endpoint returns `400` with `error: authorization_pending`. If the client polls too fast, the server returns `error: slow_down` and the client must increase its interval. Respecting that backpressure is part of being a well-behaved client; hammering the endpoint will get you rate-limited. The first poll after approval returns the same token set as any other flow.

```http
POST /device_authorization HTTP/1.1
Host: auth.commerce.example.com
Content-Type: application/x-www-form-urlencoded

client_id=commerce-cli&scope=payments%3Aread
```

```json
{
  "device_code": "GmRhmhcxhwAzkoEqiMEg_DnyEysNkuNhszIySk9eS",
  "user_code": "ABCD-1234",
  "verification_uri": "https://example.com/device",
  "expires_in": 600,
  "interval": 5
}
```

## 8. Why Implicit and ROPC are deprecated

Two old flows you will still see in tutorials and legacy code should be treated as **do not use**. Knowing *why* matters, because you will be asked to justify removing them.

**Implicit flow** returned the access token directly in the redirect URL fragment, skipping the code exchange. As covered above, that puts the token in browser history, logs, and referrers, exposes it to any script on the page, and offers no proof that the redeeming party started the flow. It existed because older browsers could not make cross-origin back-channel calls (no CORS). That constraint is gone. The replacement is **Authorization Code plus PKCE**, which works fine for SPAs today and never exposes a token in a URL. [RFC 9700](https://www.rfc-editor.org/rfc/rfc9700) explicitly says: do not use the implicit grant; use Authorization Code with PKCE.

**Resource Owner Password Credentials (ROPC)** had the client collect the user's username and password directly and send them to the `/token` endpoint. This is the password anti-pattern wearing an OAuth costume — the entire point of OAuth was to *avoid* the client ever seeing the password, and ROPC reintroduces it. It also cannot support MFA, federated login, or consent. It was only ever meant as a migration crutch for legacy first-party apps, and it is being removed. If you see it, plan its replacement with Authorization Code plus PKCE (for user flows) or Client Credentials (if there is no real user).

To see why implicit's leakage is not theoretical, walk where the token travels. In a redirect like `https://app.example.com/cb#access_token=...&token_type=Bearer`, the fragment after `#` lands in several places you do not control: the browser's history (so the token survives in the back button and in synced history across the user's devices), any analytics or error-reporting script on the page (which can read `window.location` and ship it to a third party), and — if the app ever does a server round-trip that includes the URL — your own server logs. None of those is an "attack" in the dramatic sense; they are ordinary browser behavior quietly copying a live credential into places it should never be. Authorization Code plus PKCE closes all of them at once because the *token* never appears in a URL — only the short-lived, single-use *code* does, and that code is worthless without the verifier held privately by the client. That is the whole argument for the migration in one sentence: move the secret off the URL.

| Deprecated flow | Why it existed | Why it is dead | Use instead |
|---|---|---|---|
| **Implicit** | No CORS in old browsers; token in URL | Token leaks via URL/history/logs; no integrity binding | Auth Code + PKCE |
| **ROPC** | Migrate legacy first-party password apps | Reintroduces the password anti-pattern; no MFA/consent | Auth Code + PKCE, or Client Credentials |

## 9. The tokens: access, ID, refresh, scopes, and audience

We have met all three tokens in passing. Now the designer's-eye view, because *how you shape your tokens* is a contract decision your resource servers and clients will live with for years.

![a layered stack contrasting the access token sent to the API the ID token read by the client and the refresh token kept secret](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-5.png)

**The access token** is the capability sent to the API. As an API designer you choose its format:

- **Opaque** — a random, meaningless string. The resource server cannot read it locally; it must call the authorization server's **introspection** endpoint to learn what the token grants. Pro: revocation is instant (the authorization server is the source of truth on every call) and the token leaks no information if intercepted. Con: every API request incurs a network round-trip to introspect.
- **JWT** — a self-contained, signed token whose claims (`iss`, `aud`, `exp`, `scope`, `sub`, …) the resource server reads and verifies *locally* using the authorization server's public key. Pro: no per-request network call; the API validates offline. Con: it cannot reflect a revocation until it expires — a JWT revoked at 12:00 is still cryptographically valid until its `exp`.

| | Opaque access token | JWT access token |
|---|---|---|
| **Resource server reads it** | Via introspection call | Locally, by verifying the signature |
| **Per-request cost** | Network round-trip | None (offline verify) |
| **Revocation** | Instant (server is truth) | Only at expiry (stale until `exp`) |
| **Leaks info if intercepted?** | No (meaningless string) | Yes (claims are readable, just not forgeable) |
| **Best fit** | High-value or rare calls; tight revocation | High-volume APIs; keep `exp` short |

The standard compromise: **JWT access tokens with a short lifetime** (5–15 minutes). You get offline validation for throughput, and the short `exp` bounds the damage of a leak and the staleness of a revocation. Pair it with refresh-token revocation at the authorization server so that killing a session stops *new* access tokens within one access-token lifetime.

There is a small, useful piece of math behind the lifetime choice. With a JWT access token of lifetime $T$, the **worst-case revocation lag** — the time between revoking a session and the last valid access token expiring — is exactly $T$. So if you set $T = 15$ minutes, a compromised session can keep calling for up to 15 minutes after you hit "revoke," no matter how fast your revocation propagates, because nothing checks the authorization server mid-token. Shrinking $T$ shrinks that window linearly. But it also raises the rate of refresh calls: a client active for a duration $D$ makes roughly $D / T$ refreshes, so halving $T$ doubles the load on your `/token` endpoint. The whole opaque-versus-JWT debate is really a choice on this curve: opaque tokens (or per-call introspection) push the revocation lag toward zero at the cost of a network round-trip per request, while long-lived JWTs minimize round-trips at the cost of a long revocation lag. The 5–15 minute JWT is the elbow of that curve for most APIs — small enough that the lag is tolerable, large enough that refresh traffic stays modest. For an operation where a 15-minute lag is unacceptable (issuing refunds, changing payout bank details), drop to introspection or a 60-second token specifically on that path.

**The ID token** is always a JWT (OIDC requires it). It is for the *client*, carries `aud = the client`, and proves the user's identity. It is read once at login to establish a session and is **never sent to the API**. If you ever find yourself putting an ID token in an `Authorization: Bearer` header to an API, stop — you have crossed the wires.

**The refresh token** is the long-lived key to mint more access tokens. It is opaque, never sent to the resource server, and stored with the most care. Rotate it on every use.

**Scopes** are the named permissions the access token carries — `payments:read`, `payments:write`, `refunds:write`, `orders:read`. They are the vocabulary of delegation: the consent screen shows them to the user, the token contains the granted subset, and the resource server enforces them per endpoint. Scopes are *coarse-grained* by design — they say "may read payments," not "may read payment `pay_123`." Fine-grained, per-resource authorization (this user may see *this* payment) lives in your [authorization layer](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions), not in the scope. A clean design uses scopes for the *category* of access and your own authorization logic for the *instance*.

The division of labor is worth making explicit, because new designers often try to push everything into scopes and end up with hundreds of them. Think of two gates the request passes. The **scope gate** is broad and lives in the token: "this token may read payments at all." The **resource gate** is narrow and lives in your application logic: "this token's subject is merchant 9d12, and payment `pay_123` belongs to merchant 9d12, so allow." A token with `payments:read` still must not let merchant A read merchant B's payments — that is not a scope failure, it is an *object-level* authorization failure (OWASP calls this BOLA, broken object-level authorization, the number-one API risk). Scopes alone can never enforce ownership, because the scope does not know which row you are about to read. So: scopes decide *what kind* of thing you may touch; your authorization layer decides *which specific* instances. Get both, every time. A token that passes the scope gate and skips the resource gate is exactly how one customer ends up reading another's data with a perfectly valid token.

**The audience (`aud`)** is the claim that binds a token to a specific resource server. It is the answer to "a token for API A must not be replayable at API B." When the authorization server mints a token for the payments API, it stamps `aud = https://api.commerce.example.com/payments`. The refunds API, validating an inbound token, checks `aud` against its *own* identifier and rejects anything that does not match. Without this check, any valid token from your authorization server would work at *every* API behind it — a single leaked payments token would also drain refunds. **Validating `aud` is not optional.**

| | Access token | ID token | Refresh token |
|---|---|---|---|
| **Job** | Grant API access | Prove user identity | Get new access tokens |
| **Audience** | The API | The client | The authorization server |
| **Format** | Opaque or JWT | Always JWT | Opaque |
| **Lifetime** | Short (minutes) | Short (login session) | Long (days–months) |
| **Sent to the API?** | Yes (`Bearer`) | No | No |
| **Where stored** | Memory / short cache | Session after login | Most-secure storage, rotated |

## 10. Validating a token at the resource server

This is the section every API designer must internalize, because the resource server is the role you own and the place where security is actually enforced. **A token is only as good as the validation the API performs on it.** A perfectly minted token presented to an API that "just trusts" the `Authorization` header is no security at all.

![a branching validation flow at the resource server checking signature issuer audience expiry and scope before serving or rejecting](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-6.png)

For a **JWT access token**, the resource server validates, in order, and rejects on the first failure:

1. **Signature.** Verify the JWT's signature against the authorization server's public key. This proves the token was issued by your authorization server and not forged. The public keys are published at the authorization server's **JWKS** (JSON Web Key Set) endpoint — a URL listing the current signing keys, each with a key ID (`kid`). The JWT's header names which `kid` signed it. Critically, **never trust the `alg` header to mean "no signature"** — pin the expected algorithm (e.g. `RS256` or `ES256`) and reject `none`. The classic JWT exploit is sending `alg: none` to a naive verifier.
2. **Issuer (`iss`).** Check `iss` equals your trusted authorization server's exact identifier. A token from a *different* issuer — even a real, valid one — is not yours to honor.
3. **Audience (`aud`).** Check `aud` contains *this* resource server's identifier. This is the replay-prevention check from the last section. Skip it and you have the most common audit finding in OAuth deployments.
4. **Expiry (`exp`).** Check the token has not expired (`exp` in the future), and if present that `nbf` ("not before") is in the past, allowing a small clock-skew leeway (a minute or two).
5. **Scope.** Check the token's `scope` claim includes the scope this endpoint requires. `GET /v1/payments` requires `payments:read`; a token with only `orders:read` gets a `403 Forbidden`.

Get all five right and you may serve the request. Fail signature/iss/aud/exp and you return `401 Unauthorized` (the credential is missing or invalid). Pass authentication but fail scope and you return `403 Forbidden` (you are authenticated, but not allowed) — the [status-code distinction matters](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx) for the caller's debugging. A small detail that helps callers: on a `401`, send a `WWW-Authenticate: Bearer error="invalid_token", error_description="..."` header so the client knows *why* (expired? bad signature? wrong audience?) and whether refreshing will help. A bare `401` with no hint sends integrators on a guessing hunt.

#### Worked example: the consequence of skipping the audience check

This is worth walking as a concrete before→after because it is the single most common real omission, and the failure mode is non-obvious.

Suppose your platform runs three APIs behind one authorization server — payments, refunds, and orders — and a junior engineer ships the orders API with validation that checks signature, `iss`, and `exp`, but **not** `aud`. Every token your authorization server issues is signed by the same key and carries the same `iss`, so the orders API will accept *any* of them.

- **Before the bug (every API checks `aud`):** the accounting app's token, minted with `aud = .../payments`, works only at the payments API. Presenting it to orders yields `403`. Containment is automatic.
- **After the bug (orders skips `aud`):** the accounting app — or anyone who captures its `payments:read` token — can now call the orders API and read order data the merchant never consented to share, simply because the orders API forgot to check who the token was *for*. No alarm fires. The token is genuine, signed, unexpired, and from the right issuer; the only thing wrong is that it was never meant for orders. This is a **confused-deputy** vulnerability, and it is invisible in logs because every individual check that *was* performed passed. The fix is one line — assert `aud` equals the orders API's own identifier — but you only write that line if you know to. This is why "validate `aud`" sits at the top of every OAuth security checklist and why it leads the pitfalls list below.

Here is a compact, real validation middleware in Python (using a JWKS client) so the checks are concrete, not hand-waved:

```python
import jwt  # PyJWT
from jwt import PyJWKClient

JWKS_URL = "https://auth.commerce.example.com/.well-known/jwks.json"
ISSUER = "https://auth.commerce.example.com"
AUDIENCE = "https://api.commerce.example.com/payments"

jwks_client = PyJWKClient(JWKS_URL)  # caches keys, refreshes on new kid

def validate(bearer_token: str, required_scope: str) -> dict:
    # 1. fetch the signing key named by the token's kid (JWKS rotation handled here)
    signing_key = jwks_client.get_signing_key_from_jwt(bearer_token)
    # 2-4. verify signature with a PINNED alg, plus iss, aud, exp, with small leeway
    claims = jwt.decode(
        bearer_token,
        signing_key.key,
        algorithms=["RS256"],          # pin it; never accept "none"
        audience=AUDIENCE,             # raises if aud mismatches
        issuer=ISSUER,                 # raises if iss mismatches
        options={"require": ["exp", "iss", "aud"]},
        leeway=60,                     # 60s clock-skew tolerance
    )
    # 5. scope check
    granted = set(claims.get("scope", "").split())
    if required_scope not in granted:
        raise PermissionError(f"missing scope: {required_scope}")  # -> 403
    return claims
```

### JWKS and key rotation

Step 1 hides something important: **key rotation**. Authorization servers rotate their signing keys periodically (and urgently if a key is compromised). They publish the *current set* of valid keys at the JWKS endpoint, each tagged with a `kid`. When a token arrives signed with a new `kid` your cache has not seen, the JWKS client re-fetches the key set, picks up the new key, and caches it. This is why you fetch keys *by `kid` from the JWKS endpoint* rather than hard-coding a public key: rotation must not require a redeploy of every resource server. Cache the JWKS response (respecting its `Cache-Control`) so you are not hitting the endpoint on every request, but be ready to refresh on an unknown `kid`. A resource server that hard-codes one key will start returning `401` to *every* user the moment the authorization server rotates — a self-inflicted outage that has happened to many teams.

### Token introspection vs local JWT validation

If your access tokens are **opaque**, the resource server cannot do steps 1–5 locally — there are no readable claims. Instead it calls the authorization server's **introspection** endpoint ([RFC 7662](https://www.rfc-editor.org/rfc/rfc7662)) to ask "is this token active, and what does it grant?"

```http
POST /introspect HTTP/1.1
Host: auth.commerce.example.com
Content-Type: application/x-www-form-urlencoded
Authorization: Basic <base64 of api-payments:YOUR_CLIENT_SECRET>

token=8xLOpaqueAccessTokenValueHere
```

```json
{
  "active": true,
  "scope": "payments:read",
  "aud": "https://api.commerce.example.com/payments",
  "iss": "https://auth.commerce.example.com",
  "exp": 1718900000,
  "sub": "batch-recon"
}
```

![a comparison matrix of local JWT validation versus token introspection across latency revocation awareness and best fit](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-7.png)

The trade-off is exactly the opaque-vs-JWT trade-off from §9, now seen from the validation side: introspection is a network round-trip per request (or per cache window) but reflects revocation in real time; local JWT validation is offline and fast but blind to revocation until `exp`. Many production systems run a hybrid: local JWT validation for most traffic, plus introspection (or a revocation-list check) on the highest-value operations like issuing a refund. Choose by the cost of being wrong: for `payments:read`, a 10-minute stale window is usually fine; for `refunds:write`, you may want introspection or a very short `exp` with active revocation checking.

### Discovery: how a resource server learns where to look

You may have noticed two magic URLs in the validation code: `/.well-known/jwks.json` and, mentioned earlier, `/.well-known/openid-configuration`. These are **discovery** endpoints, and they are the reason a resource server does not need its issuer, JWKS URL, introspection URL, and supported algorithms hard-coded. OIDC standardizes a **discovery document** at `/.well-known/openid-configuration` that lists every endpoint and capability of the authorization server. A resource server (or client library) fetches it once at startup and configures itself from it.

```json
{
  "issuer": "https://auth.commerce.example.com",
  "authorization_endpoint": "https://auth.commerce.example.com/authorize",
  "token_endpoint": "https://auth.commerce.example.com/token",
  "introspection_endpoint": "https://auth.commerce.example.com/introspect",
  "jwks_uri": "https://auth.commerce.example.com/.well-known/jwks.json",
  "scopes_supported": ["openid", "payments:read", "payments:write", "refunds:write"],
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "client_credentials", "refresh_token"],
  "id_token_signing_alg_values_supported": ["RS256", "ES256"],
  "code_challenge_methods_supported": ["S256"]
}
```

Two practical wins fall out of discovery. First, **the `issuer` value here is the exact string your tokens' `iss` claim must equal** — there is your `iss` check, sourced from the horse's mouth. Second, **`jwks_uri` is where the signing keys live**, and because you read it from discovery you survive the authorization server moving its JWKS endpoint. The design lesson for the API you build: publish a discovery document and a JWKS endpoint, keep them current, and your resource servers and third-party clients configure themselves correctly without a single hard-coded URL. It is the same instinct as a self-describing [OpenAPI document](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) — make the contract machine-readable so consumers do not guess.

### The SPA problem: where do tokens live in a browser?

Single-page apps deserve their own paragraph because the question "where do I store the token?" has been answered wrong for years. A browser has no truly safe place for a long-lived secret. `localStorage` is readable by *any* JavaScript on the page, so one cross-site-scripting (XSS) bug exfiltrates every token. Putting a refresh token there is the worst version of this — a stolen refresh token is a renewable breach.

The current best-practice answer has two parts. For the *access token*, keep it in memory only (a JavaScript variable, gone on refresh), accept that the user silently re-authenticates on reload via the authorization server's session cookie, and never persist it. For the *refresh token*, do not let the SPA hold it at all: run a thin **backend-for-frontend** (BFF) — a small server-side companion to the SPA — that completes the Authorization Code plus PKCE exchange, stores the refresh token server-side, and hands the browser only an `HttpOnly`, `Secure`, `SameSite` session cookie. JavaScript can never read an `HttpOnly` cookie, so XSS cannot steal it. The browser talks to the BFF; the BFF talks to the API with the access token. This is the same [backend-for-frontend pattern](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) you would deploy in a fleet, here doing double duty as a token vault. If you take one thing from this subsection: **never put a refresh token where page JavaScript can reach it.**

## 11. The redirect, CSRF, and replay defenses

Three small parameters in the Authorization Code flow do disproportionate security work. Designers skip them at their peril.

**Exact-match `redirect_uri`.** The authorization server must only ever send the code to a *pre-registered, exactly-matched* redirect URI. Register the full callback URL during client setup, and match it character for character on every request — no wildcards, no prefix matching, no trailing-slash fuzziness. This stops an attacker from redirecting the authorization response (and the code) to a host they control. We classified this as a security break in §5's second worked example for a reason.

**`state` for CSRF.** The `state` parameter is a random, unguessable value the client generates, includes in the `/authorize` request, and verifies on the callback. It defends against cross-site request forgery: without it, an attacker could trick a logged-in user's browser into completing an OAuth flow the attacker started, binding the *attacker's* account to the *victim's* session (a "login CSRF" / session-fixation attack). The client must reject any callback whose `state` does not match what it stored. Treat `state` as mandatory, not optional.

**`nonce` for ID-token replay (OIDC).** The `nonce` is a random value the client sends in the `/authorize` request *and* expects to find inside the returned ID token. It binds the ID token to *this* authentication request, so a previously captured ID token cannot be replayed into a new login. Where `state` protects the *authorization request*, `nonce` protects the *ID token*. In OIDC flows that return an ID token, validate that the ID token's `nonce` equals the one you sent.

A compact way to remember the three: `redirect_uri` controls *where the code can go*, `state` proves *this response answers my request*, and `nonce` proves *this ID token answers my login*.

### Consent is part of the contract too

It is easy to treat the consent screen as UI chrome, but for an API designer it is a contract surface as real as any endpoint. The consent screen is where the *resource owner* sees, in human language, exactly what they are about to delegate — and the words come from your **scopes**. If your scope is named `payments` (vague) the user reads "this app wants your payments" and cannot tell read from write. If it is named `payments:read` the consent reads honestly: "this app wants to *read* your payments." Scope naming is therefore not just an enforcement detail; it is the text of the promise the user agrees to. Name scopes as `resource:action` and keep them granular enough that consent means something.

Two consent behaviors matter in practice. **Incremental authorization** lets a client request more scopes later, with a fresh consent, rather than demanding everything up front — so the accounting app asks for `payments:read` on day one and only prompts for `refunds:write` if and when the user enables a refund feature. This keeps the first-run consent small and honest, which raises conversion and lowers the temptation to overscope. **Consent revocation** is the user-facing mirror of token revocation: a "connected apps" page where the resource owner can see every app they have granted and revoke any of them, killing that client's tokens. Both Google and GitHub expose exactly such a page, and your platform should too — a delegation the user cannot *see* and *undo* is not really a delegation, it is just access. When you design the API, design the revocation story alongside it; the day a partner integration misbehaves, "click revoke" is the difference between a five-minute fix and an emergency password reset for every affected merchant.

## 12. Common pitfalls (and the one-line fixes)

After enough OAuth incident reviews, the same handful of mistakes recur. Each has a cheap fix; the expensive part is knowing to look.

![a two-column figure pairing common OAuth pitfalls like a missing audience check and overscoping with their direct fixes](/imgs/blogs/oauth2-and-openid-connect-for-api-designers-8.png)

- **Missing `aud` check.** The most common audit finding. Without it, any token from your authorization server works at *every* API. **Fix:** every resource server validates `aud` against its own identifier and rejects mismatches.
- **Token leakage.** Tokens in URLs (the implicit-flow sin), in logs, in error reports, in referrer headers, in browser-accessible storage for SPAs. **Fix:** tokens travel only in headers over TLS; scrub `Authorization` from logs; use Authorization Code (not implicit) so tokens never hit a URL; keep refresh tokens out of JavaScript reach.
- **Overscoping.** A client asks for `refunds:write` when it only ever reads payments, because "we might need it later." Now a leak is far more dangerous and the consent screen is scary. **Fix:** request the *least* scope that does the job — `payments:read` and nothing more for the accounting app. Add scopes when you actually need them.
- **Long-lived access tokens.** A 30-day access token means a leak is a 30-day breach. **Fix:** short access tokens (minutes), refresh tokens for continuity, rotation on refresh.
- **Using an access token as proof of identity.** Covered in §3 — it invites the confused-deputy/token-substitution attack. **Fix:** identity from the ID token's verified `sub`/`aud`/`nonce`; never infer the user from an access token.
- **Accepting `alg: none` or an unpinned algorithm.** A forged JWT walks right in. **Fix:** pin the expected `alg`; reject `none`.
- **Hard-coding the signing key.** A self-inflicted outage on the next key rotation. **Fix:** fetch keys by `kid` from the JWKS endpoint, cache with refresh.
- **Wildcard or prefix `redirect_uri` matching.** Lets an attacker capture the code. **Fix:** exact string match against a pre-registered URI.

## 13. Case studies: how the big providers do this

These are accurate, public design choices worth learning from.

**Google** runs a textbook OIDC-on-OAuth identity provider. "Sign in with Google" issues an ID token (a JWT) the relying party validates for identity, and access tokens scoped to specific Google APIs (Gmail, Drive, Calendar) for delegated access. Google's documentation explicitly distinguishes the ID token (who the user is, for your app) from access tokens (what your app may call), and exposes a standard JWKS endpoint for signature verification — the same pattern you would build for your own platform.

**GitHub** historically used a non-PKCE OAuth App model and later added **GitHub Apps** with fine-grained, per-installation permissions and short-lived tokens — a move toward least privilege and away from broad user-scoped tokens. GitHub Apps mint installation access tokens that expire in an hour and carry only the permissions the installation was granted, which is the Client-Credentials-style, scoped-and-short philosophy applied to a developer platform. The lesson: scope down and expire fast, even for first-party tooling.

**Okta, Auth0, Microsoft Entra ID, and Keycloak** are the authorization servers most teams adopt rather than build. They implement the OAuth/OIDC endpoints (`/authorize`, `/token`, `/introspect`, JWKS, `/.well-known/openid-configuration` for discovery), handle login/MFA/federation, and let you define scopes and audiences. The practical takeaway for an API designer: you rarely build the authorization server; you build the *resource server* and configure the *client*, so the validation logic in §10 is the code you actually own.

**PKCE adoption** is now near-universal as the default for Authorization Code flows. The OAuth Working Group's [Security BCP (RFC 9700)](https://www.rfc-editor.org/rfc/rfc9700) recommends PKCE for *all* clients (not just public ones) and recommends *against* the implicit and ROPC grants. The draft work toward "OAuth 2.1" consolidates exactly this guidance: Authorization Code with PKCE as the one user flow, implicit and ROPC removed. If you are designing today, design as if OAuth 2.1 is already the rule.

**FAPI for finance.** The **Financial-grade API (FAPI)** profile, from the OpenID Foundation, is a hardened OAuth/OIDC profile for high-value APIs like open banking. It tightens the screws: PKCE required, exact redirect matching, sender-constrained tokens (so a stolen bearer token cannot be used by a different sender — via mTLS-bound tokens or DPoP), stronger client authentication, and signed request objects. If you are designing a payments or banking API where a stolen bearer token means stolen money, FAPI is the profile to study — it is what the §12 pitfalls look like once they are *mandated away* by a standard.

One FAPI idea is worth pulling forward even for non-banking APIs: **sender-constrained tokens**. A plain bearer token is exactly that — *bearer* — so whoever holds it can use it, which is why leakage is so dangerous. Sender constraining binds the token to a key the legitimate client holds, so a stolen token is useless to a thief who lacks that key. The two mechanisms are **mTLS-bound tokens** ([RFC 8705](https://www.rfc-editor.org/rfc/rfc8705)), which tie the token to the client's TLS certificate, and **DPoP** ([RFC 9449](https://www.rfc-editor.org/rfc/rfc9449)), which has the client sign each request with a proof-of-possession key. Both turn "whoever holds it wins" into "whoever holds it *and* the key wins," which is a large security upgrade for high-value endpoints. You will not need this for `payments:read`, but it is exactly the kind of control you reach for on `refunds:write` or `payout:create` — and knowing it exists keeps you from concluding that bearer-token leakage is simply unavoidable.

**Stripe and the consistency lesson.** Stripe is worth naming for how seriously it treats API surfaces as long-lived contracts — versioned, idempotency-keyed, and carefully evolved. The transferable lesson for OAuth is the same discipline applied to tokens and scopes: name scopes consistently, expire tokens aggressively, document the exact validation a resource server performs, and never break the token contract a partner depends on without a deprecation path. The auth layer is not exempt from the "an API is a contract" rule that governs the rest of your surface; a token format or a scope name, once partners depend on it, is as load-bearing as any response field.

## 14. When to reach for each flow (and when not to)

A decisive recommendation, because every choice is a trade-off and "it depends" is not an answer your reviewers can act on.

**Use Authorization Code plus PKCE when** there is a human and the client acts on their behalf — web apps, mobile apps, single-page apps. This is your default for user-facing delegation. **Do not** use implicit instead "because it is simpler"; it is not simpler once you account for the leakage, and it is deprecated. **Do not** skip PKCE "because it is a confidential web client"; PKCE is now recommended for all clients and costs almost nothing.

**Use Client Credentials when** there is no user — service-to-service, cron jobs, internal batch work. **Do not** use it to impersonate a user (there is no `sub` user, no consent, no user-scoped data ownership); if a real user's data is involved on their behalf, you need a user flow. **Do not** reach for a static API key when you could use Client Credentials and get expiry, scope, central revocation, and `aud` binding for free.

**Use the device flow when** the device cannot reasonably show a browser or accept typed credentials — TVs, CLIs, IoT. **Do not** use it for ordinary web apps where a redirect works fine; the polling and second-screen UX is worse than a redirect when a redirect is available.

**Use refresh tokens when** you need long-lived access without re-prompting — almost always, for user flows. **Do not** issue them to a browser-resident SPA without an `HttpOnly`-cookie/BFF arrangement; a refresh token reachable by page JavaScript is a refresh token waiting to be exfiltrated.

**Do not** build your own authorization server unless identity is your actual product. The login/MFA/federation/recovery surface is enormous and adversarial. Adopt a provider, and spend your effort on the resource-server validation you own. And **do not** reach for OAuth at all for the trivial case of one first-party client calling one first-party API with no third party and no user delegation — a [session or a scoped API key](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls) may be the right-sized tool, with OAuth waiting for the day a third party shows up. Match the mechanism to the threat, the same way you would [rate-limit by the abuse you actually face](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection) rather than by reflex.

## 15. Key takeaways

- **OAuth solves delegation, not login.** Its job is to let an app act on a user's behalf with a *scoped, revocable, audience-bound token* instead of the user's password. Stop thinking "log the app in," start thinking "mint a capability."
- **OAuth is authorization; OIDC adds authentication.** The access token grants API access; the ID token proves who the user is. **Never use an access token as proof of identity** — that is the confused-deputy bug.
- **Authorization Code plus PKCE is the default user flow.** It keeps tokens out of URLs and binds the code exchange to the client that started it. PKCE is now recommended for *all* clients; implicit and ROPC are deprecated.
- **Client Credentials is the machine flow.** No user, no consent, no ID token, no refresh token — just a confidential client trading its secret for a short-lived, scoped, audience-stamped token.
- **Three tokens, three jobs.** Access (to the API, short-lived), ID (to the client, identity), refresh (to the authorization server, long-lived, rotated). Do not cross the wires.
- **Validate every token on every request.** Signature (pinned `alg`, JWKS by `kid`), `iss`, `aud`, `exp`, then scope. Missing the `aud` check is the most common and most dangerous omission.
- **Pick opaque vs JWT by the cost of being wrong.** JWT for throughput with short `exp`; introspection for real-time revocation on high-value calls; hybrids are common.
- **`redirect_uri` exact-match, `state`, and `nonce` are not optional.** They control where the code goes, prove the response answers your request, and bind the ID token to your login.

## 16. Further reading

- [RFC 6749 — The OAuth 2.0 Authorization Framework](https://www.rfc-editor.org/rfc/rfc6749) — the base specification: roles, grant types, endpoints.
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://www.rfc-editor.org/rfc/rfc7636) — the `code_verifier`/`code_challenge` mechanism.
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html) — the identity layer, the ID token, and `nonce`.
- [RFC 9700 — Best Current Practice for OAuth 2.0 Security](https://www.rfc-editor.org/rfc/rfc9700) — the authoritative "use this, not that" guidance (and OAuth 2.1 direction).
- [RFC 7519 — JSON Web Token (JWT)](https://www.rfc-editor.org/rfc/rfc7519) and [RFC 7662 — OAuth 2.0 Token Introspection](https://www.rfc-editor.org/rfc/rfc7662) — the token format and the introspection endpoint.
- [RFC 8628 — OAuth 2.0 Device Authorization Grant](https://www.rfc-editor.org/rfc/rfc8628) — the device/CLI flow.
- [The Financial-grade API (FAPI)](https://openid.net/wg/fapi/) — the hardened profile for high-value/banking APIs.
- Within this series: the [API-as-a-contract intro hub](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), the [authentication post on keys, sessions, JWT, and mTLS](/blog/software-development/api-design/authentication-api-keys-sessions-jwt-and-mtls), the [authorization post on scopes, roles, and resource-level permissions](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions), the [rate-limiting and abuse-protection post](/blog/software-development/api-design/rate-limiting-quotas-and-abuse-protection), and the [API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

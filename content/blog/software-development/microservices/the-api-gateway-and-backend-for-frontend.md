---
title: "The API Gateway and Backend-for-Frontend: Owning the Edge Without Building a New Monolith"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Why clients should never call twenty services directly, what a gateway actually does, how the Backend-for-Frontend pattern keeps the edge from rotting into a god-service, and the worked latency math that decides when a fan-out is worth it."
tags:
  [
    "microservices",
    "api-gateway",
    "backend-for-frontend",
    "distributed-systems",
    "software-architecture",
    "backend",
    "edge",
    "aggregation",
    "envoy",
    "resilience",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-api-gateway-and-backend-for-frontend-1.webp"
---

The ShopFast mobile team shipped a redesigned home screen, and three days later the on-call channel lit up with a complaint nobody on the backend had predicted: the app was burning through users' mobile data and draining their batteries. The screen itself looked fine. The problem was underneath. To paint a single "your orders" view — the order summary, the product thumbnails and titles for each item, the shipping status, and a couple of personalized recommendations — the app was making six separate HTTPS calls to six separate services. On a fast office Wi-Fi connection nobody noticed. On a real phone, on a real cellular network somewhere between two cell towers with a 150-millisecond round-trip time, those six calls — some of them serialized because the recommendation call needed the order IDs from the first call — turned a one-screen render into nearly a second of staring at spinners, six TLS handshakes, six rounds of radio wake-up, and six chances for one flaky service to make the whole screen look broken.

The backend team's first instinct was to blame the mobile team. The mobile team's first instinct was to blame the backend for not having "one endpoint that just gives me the screen." They were both right, and the argument they were having is one of the oldest and most consequential at the edge of any microservices system. When you split a monolith into twenty services, you do not just split the *code* — you split the *thing the client used to talk to*. The monolith was one address, one login, one place that assembled the whole page. Twenty services are twenty addresses, and unless you do something deliberate about it, you have just made the client's life dramatically worse: more round trips, more cross-cutting concerns to reimplement (authentication, retries, TLS) in every client, and a brittle dependency where the *internal* shape of your system is now leaking out into apps you cannot redeploy on demand.

The something-deliberate is a gateway at the edge, and the right shape for it is the subject of this post. By the end you should be able to do four concrete things: explain precisely what problem a gateway solves and which jobs belong in it versus in your services; design a Backend-for-Frontend so that each client gets a tailored entry point instead of one bloated god-gateway; write the aggregation code that fans out to several services concurrently, bounds every call with a deadline, and returns a useful partial response when one upstream is slow; and recognize the three ways a gateway quietly turns back into the monolith you just escaped — and stop it. We will run the whole thing on ShopFast, do the latency arithmetic that decides whether the fan-out is worth it, and finish with real case studies and a build-versus-buy matrix you can take into a design review on Monday.

![A before and after comparison contrasting a mobile client making six direct round trips to services with auth and TLS duplicated everywhere against a single gateway entry point that fans out internally and owns the cross-cutting concerns](/imgs/blogs/the-api-gateway-and-backend-for-frontend-1.webp)

This post sits at a specific layer of the series. The [fundamentals and fallacies post](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) taught the laws that govern *any* service-to-service call — latency multiplies, availability multiplies, the network is hostile. The [REST vs gRPC vs GraphQL post](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis) chose the *protocol* for those calls. This post is about the *first hop*: the boundary where the outside world meets your fleet. Get it wrong and you either expose twenty services to every client (chaos) or funnel everything through one team's god-service (a new monolith). Get it right and you have a thin, fast, observable edge that lets the services behind it stay small and the teams behind *them* stay autonomous.

## The problem: a client should not have to know your org chart

Start with the thing the monolith gave you for free and the microservices split silently took away: *a single, stable, client-shaped interface*. In the monolith, the browser or the app talked to one origin. Behind that origin, a request handler called whatever internal modules it needed — the order module, the catalog module, the shipping module — all as in-process function calls, assembled the response, and sent back exactly the JSON the screen wanted. The client never knew or cared how many modules were involved, where they lived, or how they talked to each other. The internal structure was *encapsulated*, and that encapsulation is the entire value of a coarse-grained interface.

Decompose into services and, if you do nothing, you have de-encapsulated. The client now has to know that orders live at `orders.shopfast.internal`, products at `catalog.shopfast.internal`, shipping at `shipping.shopfast.internal`, and so on. That is bad for at least four distinct reasons, and it is worth separating them because each one motivates a different gateway capability.

**Chattiness.** The most visible problem and the one that bit the ShopFast mobile team. A screen that needs data from six services becomes six round trips from the client. On a same-datacenter link a round trip is sub-millisecond, so six calls between *services* is cheap. But the client is not in your datacenter; the client is on a phone on a cellular network where one round trip is 50 to 200 milliseconds, dominated by radio latency, not by your servers. Six serialized round trips at 150ms is 900ms of pure waiting before a single pixel of useful content. The fix is to move the fan-out *inside* the datacenter, where round trips are cheap, and give the client one fat call instead of six skinny ones. That is request aggregation, and it is the single most valuable thing a client-facing gateway does.

**Duplicated cross-cutting concerns.** Every client that talks directly to services has to independently implement the same plumbing: validate the auth token (or attach it to every call), terminate or negotiate TLS, retry on transient failure, set timeouts, add tracing headers, handle rate-limit responses. If the web app, the iOS app, the Android app, and three partner integrations each call services directly, that plumbing is reimplemented six times, in different languages, with subtly different bugs, and a change to your auth scheme means six coordinated client releases — two of which (the mobile apps) you cannot force users to install. Pulling these concerns to a single edge means you implement them *once*, correctly, in a place you control and can deploy on your own schedule.

**Topology leakage.** When the client knows the address and shape of every internal service, that internal structure becomes a *de facto public contract*. You wanted to split the order service into an order-write service and an order-query service? Too bad — the mobile app from eight months ago has the old URL hard-coded. You wanted to rename a field, merge two services, change a service's protocol from REST to gRPC for efficiency? Every one of those is now a breaking change to clients you cannot redeploy. A gateway is an *indirection layer*: clients talk to the gateway's stable contract, the gateway talks to whatever the internal topology happens to be today, and you are free to refactor the inside without touching the outside.

**No single place for policy.** Where do you enforce that anonymous users can hit the catalog but not the orders endpoint? Where do you rate-limit a misbehaving client to protect the whole fleet? Where do you terminate TLS so internal services can speak plain HTTP/2 to each other? Where do you cache the responses that are safe to cache so the services behind never even see the request? With direct-to-service access there is no single place; with a gateway there is exactly one, and "exactly one place to enforce a policy" is worth a great deal when the policy is "do not let the world DDoS our payment service."

So the gateway is the answer to a real problem, not architecture astronautics. It is the thing that gives you back the coarse-grained, encapsulated, policy-enforced front door the monolith had — *without* giving back the monolith. The danger, which the whole back half of this post is about, is that it is extremely easy to give back the monolith by accident.

## What a gateway actually does (and what it must not)

Let me be precise about the job description, because "gateway" gets used loosely and half the bad gateways in the world are bad because someone put the wrong responsibilities in them. A gateway crossing from the public internet into your service fleet does a defined set of cross-cutting jobs, and a request passes through them roughly in this order.

![A vertical stack of the gateway's cross-cutting layers showing TLS termination at the top, then token validation, then rate limiting and load shedding, then routing and load balancing, then aggregation and caching, with tracing spanning the whole hop](/imgs/blogs/the-api-gateway-and-backend-for-frontend-3.webp)

**TLS termination.** The gateway holds the public certificate and terminates the encrypted client connection. Inside the trust boundary, services can speak plain HTTP/2 to each other (or mTLS if you run a mesh — more on that at the end). Terminating TLS once at the edge means you manage certificates in one place and your internal services do not each have to.

**Authentication and token validation.** The gateway validates the caller's credentials — typically a bearer token (a JWT) or a session — *before* the request is allowed to reach a service. This is the difference between authentication (who are you?) and authorization (are you allowed to do this?). A well-built edge does the cheap, universal part here: verify the token signature, check it is not expired, reject anonymous traffic to protected routes. The deep, resource-specific authorization ("can *this* user read *that* order?") generally belongs in the service that owns the order, because only it knows the ownership rules. The pattern of validating once at the edge and then propagating identity downstream is exactly what [authentication and authorization with OAuth2, JWT, and token propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation) covers in depth; here, just hold the line that the gateway does *coarse* authn and forwards a trustworthy identity inward.

**Rate limiting and load shedding.** The gateway is where you protect the fleet from too much traffic — a buggy client in a retry loop, a scraper, a genuine spike. It counts requests per client or per API key and rejects (429) or sheds (drops cheaply) traffic above the limit. This is the cheapest place to say "no," because saying no at the edge costs almost nothing, whereas letting the request through and saying no three services deep has already burned work. The mechanics — token buckets, sliding windows, what to shed first — are the subject of [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding).

**Routing.** The core job. The gateway maps an incoming request — by path, host, header, or method — to an upstream service, and load-balances across that service's healthy instances. `GET /api/orders/*` goes to the order service; `GET /api/products/*` goes to the catalog service. The gateway does not need to know *where* those instances are; it asks service discovery, which is the topic of [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing). The gateway is a *consumer* of discovery, not a replacement for it.

**Protocol translation.** The outside world speaks REST/JSON over HTTP/1.1 because that is what browsers and most clients are comfortable with. Your internal services may prefer gRPC over HTTP/2 for efficiency and strong typing. The gateway can translate: accept JSON from the client, call the service over gRPC, translate the protobuf response back to JSON. This lets you get the internal efficiency of gRPC without forcing it on browsers that cannot easily speak it — a trade-off explored fully in the [protocol comparison post](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis). The translation is mechanical when there is a schema to drive it: gRPC's own `grpc-gateway` and Envoy's gRPC-JSON transcoder both read the protobuf service definition and generate the REST↔gRPC mapping automatically, so the gateway is not hand-maintaining a translation layer that drifts from the service contract. The one trap is that translation has a cost — a JSON parse, a protobuf marshal, and back — so do not translate on a hot path where you could have spoken the upstream's protocol natively; translate at the *edge* where the client genuinely cannot speak gRPC, and let internal calls stay gRPC-to-gRPC.

**Request aggregation and composition.** The valuable, BFF-flavored job: fan out to several services concurrently and stitch their responses into one client-shaped payload. This is what kills the chattiness problem. It is also the most dangerous capability, because "stitch responses together" is one short step from "implement business logic," and business logic in the gateway is the road back to the monolith.

**Response caching.** For responses that are safe to cache — a product's name and image rarely change — the gateway can cache at the edge and serve repeat requests without touching the service at all. Done well this is a huge win: the catalog service might handle 5,000 requests per second of genuine reads, but if 95% are cache hits at the edge, only 250 reach the service.

**Observability.** Every request crosses the gateway, which makes the gateway the perfect place to assign a trace ID, emit a structured access log, and record per-route latency and error metrics. The gateway sees the whole north-south traffic and is your first dashboard when something at the edge is wrong, feeding into [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry).

Now the negative space — the jobs a gateway *must not* take on, because each one is a known path to ruin:

- **Business logic.** Pricing rules, eligibility checks, order state machines, "if the user is a Gold member apply this discount." This belongs in services. The moment business logic lives in the gateway, every team that owns that logic has to coordinate a gateway deploy to change it, and the gateway stops being a thin proxy and becomes the place where all the rules live — which is the literal definition of the monolith you were escaping.
- **Per-resource authorization.** As above: "can this user read this order?" depends on data only the order service has. The gateway should not be reaching into databases to make authorization decisions.
- **Data transformation that requires domain knowledge.** Reshaping field names is fine. Computing a derived business value (loyalty points, tax, recommended items) is a service's job.
- **Stateful workflow.** Multi-step sagas, long-running orchestration — these need durable state and belong in a dedicated orchestrator or in the services, not in a stateless edge proxy. (The [saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions) covers where that state should live.)

The rule of thumb that keeps you honest: a gateway should be *configurable*, not *programmable*. If changing a routing rule, a rate limit, or an aggregation shape is a config change, you are in good shape. If it requires writing and deploying meaningful logic, you are accreting a monolith. (The BFF, as we will see, deliberately *is* programmable — which is exactly why a BFF should be owned by the client team and scoped to one client, not shared by everyone.)

## ShopFast's mobile screen: the worked latency case

Let me make the chattiness problem concrete with numbers, because "fewer round trips" is hand-waving until you do the arithmetic. This is the case that started the whole post.

#### Worked example: six direct calls versus one BFF call on a mobile link

The ShopFast "your orders" screen needs five pieces of data: the user's recent orders (order service), the product details for each ordered item (catalog service), the shipping status for each order (shipping service), the user's loyalty balance (loyalty service), and personalized recommendations (recommendations service). Plus a token-refresh call the app makes on cold start. Call it six calls.

Assume a realistic mobile network: 150ms round-trip time (RTT), which is typical for 4G with some congestion. Assume each service responds in 40ms of actual server time once it has the request. And assume the client cannot fully parallelize: recommendations need the order IDs, so that call waits for the orders call to come back. Say three of the six calls are serialized in a dependency chain and three can go in parallel.

Direct-from-client timing: TLS connection setup is amortized if the app reuses connections, so ignore handshake cost and count round trips. The three parallel calls cost one RTT plus server time: 150 + 40 = 190ms. The three serialized calls cost three RTTs plus server time: 3 × 150 + 3 × 40 = 570ms. Because the parallel batch overlaps with the start of the serial chain, the screen is not ready until the longer chain finishes — roughly **570ms** before the screen can render, in the good case. Add one dropped packet (mobile networks drop packets), one TCP retransmit, and you are over a second. This is exactly the "spinners and battery drain" the team saw.

Now route through a mobile BFF. The client makes **one** call: `GET /mobile/orders-screen`. That is one RTT from the client: 150ms. Inside the datacenter, the BFF fans out to all five services. Same-DC RTT is ~0.5ms, so the network cost inside is negligible; the BFF's wall-clock time is dominated by the *slowest* service it must wait for. If it parallelizes the independent calls and only serializes the genuinely dependent one (recommendations after orders), the BFF's internal time is roughly the orders call (40ms) plus the recommendations call (40ms) = 80ms, with everything else overlapping. So the client sees one RTT plus the BFF's internal work: 150 + 80 = **230ms**.

That is **570ms down to 230ms — a 60% reduction in time-to-screen** — and the client made one connection instead of six, doing one TLS negotiation, one radio wake-up, and exposing itself to one chance of a transient failure instead of six. The fan-out that used to happen across the slow mobile link now happens across the fast datacenter fabric. *This is the entire economic argument for aggregation at the edge*, and it is why the BFF pattern exists.

![A branching graph showing one mobile app request entering the mobile BFF which fans out concurrently to the order service, the product service, and the shipping service, then merges their responses into a single screen payload returned to the app](/imgs/blogs/the-api-gateway-and-backend-for-frontend-2.webp)

Note what the BFF did *not* do: it did not invent any business logic. It called the same services the client would have called, in the same way, and returned the same data — it just did the calling from a place where calling is cheap, and shaped the result into one payload. That discipline — aggregate and reshape, never compute domain logic — is what keeps the BFF a BFF and not a monolith.

## The Backend-for-Frontend pattern: one edge per client, not one edge for all

Here is the trap that catches teams who have understood everything so far. They build *a* gateway, it aggregates beautifully for the mobile app, and then the web team shows up. The web app is a different beast: it runs on a fast wired connection where round trips are cheap, it has a big screen so it wants *rich* data (full product descriptions, high-res images, the entire order history, all the recommendations), and it can happily make several parallel calls because latency is low. The mobile app wants the *opposite*: a slim payload to save bytes and battery, only the fields that fit on a small screen, aggressively pre-aggregated to minimize round trips.

If both clients share one gateway, that gateway now has to serve two contradictory masters. You add a `?fields=minimal` flag for mobile and `?fields=full` for web. Then the partner-integration team needs a stable, versioned, slowly-changing contract that looks nothing like either. Now your one gateway has three modes, branching logic for each client, and three teams filing tickets against it. It has become a shared bottleneck — the literal opposite of what microservices promised — and worse, it has become a place where client-specific concerns from three different clients all tangle together. That is the *god-gateway* anti-pattern.

The Backend-for-Frontend pattern is the fix, and it is almost embarrassingly simple: **build one gateway per client type, owned by the team that owns that client.** A mobile BFF, owned by the mobile team, tuned for slim payloads and minimal round trips. A web BFF, owned by the web team, tuned for rich payloads and many parallel calls. A partner BFF, owned by the platform/integrations team, exposing a deliberately stable, versioned public contract. Each one is small, focused, and serves exactly one consumer.

![A branching graph showing a web app, a mobile app, and a partner API each calling its own dedicated BFF, where the web BFF returns a rich payload, the mobile BFF a slim payload, and the partner BFF a stable contract, all three drawing from the same shared order product and shipping services](/imgs/blogs/the-api-gateway-and-backend-for-frontend-5.webp)

The term "Backend for Frontend" was coined at SoundCloud around 2015 by Phil Calçado and colleagues, describing exactly this: instead of one general-purpose public API trying to please every client, give each frontend its own backend that exists to serve *it*. Netflix had independently arrived at a similar idea years earlier with their device-specific adapter layer — more on both in the case studies. The insight is the same in both: the friction of one team's client roadmap waiting on another team's API is so costly that it is worth running multiple tailored edges to eliminate it.

What makes a BFF different from a generic gateway, concretely:

- **It is owned by the client team.** This is the load-bearing rule. The mobile team can change the mobile BFF on its own deploy schedule, in lockstep with the mobile app, without filing a ticket against a platform team. That is the autonomy a generic shared gateway destroys.
- **It is allowed to contain client-specific presentation logic.** Reshaping responses for the mobile screen, choosing which fields to include, deciding what a useful partial response looks like for *this* client — that is presentation concern, it is client-specific, and it legitimately lives in the BFF. (Crucially, it is still presentation, not domain logic. The BFF decides *how to show* the loyalty balance; the loyalty service decides *what the balance is*.)
- **It is scoped to one client, so it stays small.** A BFF that serves only the mobile app never accumulates the contradictory requirements that bloat a shared gateway. When the mobile screen changes, the mobile BFF changes; nothing else does.
- **It can be a different technology per client.** The mobile team writes its BFF in whatever language it is fluent in; the web team uses the language its frontend tooling integrates with (often Node, so the web BFF and the React app share types). They do not have to agree.

The cost — and there is always a cost — is duplication and operational surface. You now run three edges instead of one, three deploy pipelines, three sets of dashboards. There is genuinely duplicated code: the mobile BFF and the web BFF both call the order service and both validate tokens. The senior judgment is *how much* to split: a tiny startup with one web app and one mobile app might start with two BFFs or even one shared gateway and split later. A company with a dozen client surfaces and a dozen teams will have many BFFs. The signal that you have split too little is "every change waits on the gateway team"; the signal that you have split too much is "we maintain eight near-identical BFFs and a bug fix has to be applied to all of them." We will put concrete numbers on this trade-off in the matrix.

A common refinement worth naming: keep the truly universal cross-cutting concerns (TLS termination, coarse authn, global rate limiting, the very first observability touch) in a *thin shared edge proxy* in front of all the BFFs, and keep the client-specific aggregation and shaping in the per-client BFFs behind it. This is the "edge gateway + BFFs" topology, and it gets you the best of both: shared plumbing implemented once, client logic owned by client teams. Many large systems converge on exactly this two-layer shape.

## How to build it: the aggregating BFF in code

Abstract patterns are worthless until you can see the code, so here is the heart of a mobile BFF — the concurrent fan-out with per-call timeouts and graceful partial degradation. I will show it in Go, because Go's concurrency primitives make the structure unusually clear, and then show the equivalent shape in TypeScript for teams whose BFF shares a language with their frontend.

The non-negotiable requirements for this handler, drawn straight from the fallacies post (every remote call can be slow and can be down):

1. Fan out to the independent services **concurrently**, not serially, so the wall-clock time is the slowest call, not the sum.
2. Give each call its **own deadline**. A slow shipping service must not be allowed to hold the whole screen hostage.
3. When a non-essential call fails or times out, **degrade gracefully** — return the screen without that piece, not a 500 for the whole thing.
4. Distinguish **essential** data (no orders means there is no screen) from **optional** data (no recommendations is fine).

```go
// MobileOrdersScreen aggregates the data for the mobile "your orders" screen.
// It fans out concurrently, bounds every upstream call with its own deadline,
// and returns a useful partial response when a non-essential upstream is slow.
func (b *BFF) MobileOrdersScreen(w http.ResponseWriter, r *http.Request) {
    userID := identityFromContext(r.Context()) // set by the auth filter, see below

    // The overall budget for this screen. The client gave us, in effect,
    // ~250ms before it would rather show a partial screen than keep spinning.
    ctx, cancel := context.WithTimeout(r.Context(), 250*time.Millisecond)
    defer cancel()

    var (
        orders  []Order
        ordErr  error
        recs    []Rec   // optional: screen is fine without it
        loyalty *Loyalty // optional
        wg      sync.WaitGroup
    )

    // Orders is ESSENTIAL: kick it off and wait, because recommendations
    // depend on the order IDs and the screen is meaningless without orders.
    octx, ocancel := context.WithTimeout(ctx, 120*time.Millisecond)
    defer ocancel()
    orders, ordErr = b.orderClient.RecentOrders(octx, userID)
    if ordErr != nil {
        // Essential data failed: there is no screen. Fail fast and clean.
        writeError(w, http.StatusBadGateway, "orders unavailable")
        return
    }

    // Now fan out the OPTIONAL enrichments concurrently. None of them
    // is allowed to fail the request; each gets its own short deadline.
    wg.Add(2)

    go func() {
        defer wg.Done()
        rctx, rcancel := context.WithTimeout(ctx, 80*time.Millisecond)
        defer rcancel()
        if got, err := b.recClient.ForOrders(rctx, userID, orderIDs(orders)); err == nil {
            recs = got
        } // on error/timeout: leave recs nil, screen renders without it
    }()

    go func() {
        defer wg.Done()
        lctx, lcancel := context.WithTimeout(ctx, 60*time.Millisecond)
        defer lcancel()
        if got, err := b.loyaltyClient.Balance(lctx, userID); err == nil {
            loyalty = got
        }
    }()

    // Enrich each order with product + shipping, concurrently, bounded.
    enriched := b.enrichOrders(ctx, orders) // fans out per-order under the same ctx

    wg.Wait()

    // Shape the client-specific payload. This is PRESENTATION logic, which
    // legitimately lives in the BFF. No business rules, no pricing math.
    resp := MobileScreen{
        Orders:          enriched,
        Recommendations: recs,            // may be empty: that is fine
        Loyalty:         loyalty,         // may be nil: the app hides the badge
        Degraded:        recs == nil || loyalty == nil, // tell the client
    }
    writeJSON(w, http.StatusOK, resp)
}
```

The structure deserves a slow read. The orders call is essential and runs first because something downstream depends on it; if it fails, we fail fast with a clean 502 rather than spending the rest of the budget on a screen that cannot exist. Everything optional fans out under the *parent* context, so the overall 250ms budget caps the whole operation even if an individual call's own timeout is longer. Each optional call's failure is swallowed into a `nil` result, and the response carries a `Degraded` flag so the client knows it got a partial view and can decide whether to silently hide the missing badge or show a subtle "couldn't load recommendations" hint. Critically, the only "logic" in the handler is *presentation* — which fields go in the payload, what a degraded screen looks like. There is no pricing, no eligibility, no order state machine. That line is the whole discipline.

Here is the same shape in TypeScript, the way a web team sharing types with its React frontend would write it, using `Promise.allSettled` so that one rejected promise does not reject the whole batch:

```typescript
// Web BFF: richer payload, fast wired link, more parallel calls.
async function ordersScreen(req: Request, res: Response) {
  const userId = req.identity.sub; // set by the auth middleware
  const budgetMs = 400; // web tolerates a longer budget than mobile

  // Essential first.
  let orders: Order[];
  try {
    orders = await withTimeout(orderClient.recentOrders(userId), 150);
  } catch {
    return res.status(502).json({ error: "orders unavailable" });
  }

  // Fan out the optional enrichments concurrently; allSettled never throws.
  const [recs, loyalty, banners] = await Promise.allSettled([
    withTimeout(recClient.forOrders(userId, orders.map(o => o.id)), 120),
    withTimeout(loyaltyClient.balance(userId), 80),
    withTimeout(cmsClient.homeBanners(), 100), // web-only, mobile never asks
  ]);

  res.json({
    orders: await enrichOrders(orders, budgetMs), // products + shipping
    recommendations: settledOr(recs, []),
    loyalty: settledOr(loyalty, null),
    banners: settledOr(banners, []),
    degraded: [recs, loyalty].some(r => r.status === "rejected"),
  });
}

const settledOr = <T>(r: PromiseSettledResult<T>, fallback: T): T =>
  r.status === "fulfilled" ? r.value : fallback;
```

Notice the web BFF and the mobile BFF call *almost* the same services but make different choices: the web BFF has a bigger budget (400ms — a wired link affords it), fetches a `banners` block the mobile screen never shows, and tolerates more parallel calls. Those differences are *exactly why they are two BFFs and not one endpoint with flags*. Each is simple because each serves one client.

## Routing and the cross-cutting filters: the config

Aggregation is the programmable part. The cross-cutting plumbing — routing, auth, rate limiting — is the *configurable* part, and for that you generally do not write code; you write config and let a battle-tested gateway enforce it. Here is a routing config in two popular flavors so the shape is unmistakable.

A Spring Cloud Gateway route (Java teams, often runs as the BFF itself since it is programmable):

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: orders-route
          uri: lb://order-service          # lb:// resolves via service discovery
          predicates:
            - Path=/api/orders/**
          filters:
            - StripPrefix=1                 # /api/orders/x -> /orders/x upstream
            - name: CircuitBreaker          # wrap the upstream in a breaker
              args:
                name: ordersCb
                fallbackUri: forward:/fallback/orders
            - name: Retry
              args:
                retries: 2
                methods: GET                # only retry idempotent reads
                backoff: { firstBackoff: 50ms, maxBackoff: 200ms }
        - id: catalog-route
          uri: lb://catalog-service
          predicates:
            - Path=/api/products/**
          filters:
            - StripPrefix=1
            - name: RequestRateLimiter      # per-route rate limit
              args:
                redis-rate-limiter.replenishRate: 200   # tokens/sec
                redis-rate-limiter.burstCapacity: 400
```

And the same idea declared for Kong (a popular standalone gateway built on Nginx/OpenResty), as declarative YAML you would apply via `deck` or the admin API:

```yaml
services:
  - name: order-service
    url: http://order-service.internal:8080
    routes:
      - name: orders
        paths: ["/api/orders"]
        strip_path: true
    plugins:
      - name: jwt                          # validate the bearer token
      - name: rate-limiting
        config: { minute: 6000, policy: redis }   # 100 rps per consumer
      - name: proxy-cache
        config: { response_code: [200], request_method: ["GET"], cache_ttl: 30 }
  - name: catalog-service
    url: http://catalog-service.internal:8080
    routes:
      - name: products
        paths: ["/api/products"]
    plugins:
      - name: proxy-cache
        config: { response_code: [200], request_method: ["GET"], cache_ttl: 300 }
```

Read the two side by side and the model is clear: a route is a *predicate* (match this path/host/method) plus an *upstream* (send it here, load-balanced) plus an ordered chain of *filters/plugins* (do these cross-cutting things on the way through). Adding auth to a route is adding a plugin. Adding a per-route rate limit is config. Adding edge caching with a TTL is config. None of it is code, which is exactly why this layer can be owned by a platform team and changed safely without a service deploy.

For teams running a service mesh, the same edge routing is expressed as Envoy configuration, since Envoy is the proxy underneath most meshes and the most common standalone L7 edge. Here is a trimmed Envoy route stub that routes by path and applies a per-route timeout — the kind of thing a mesh's ingress gateway generates for you:

```yaml
route_config:
  name: edge_routes
  virtual_hosts:
    - name: shopfast
      domains: ["api.shopfast.com"]
      routes:
        - match: { prefix: "/api/orders" }
          route:
            cluster: order_service
            timeout: 0.2s                # hard per-route upstream timeout
            retry_policy:
              retry_on: "5xx,reset,connect-failure"
              num_retries: 2
              per_try_timeout: 0.08s
        - match: { prefix: "/api/products" }
          route:
            cluster: catalog_service
            timeout: 0.3s
```

The auth filter is worth one more snippet because it is where "validate once at the edge, propagate inward" becomes concrete. A request arrives with an opaque or JWT bearer token; the gateway verifies it, and *then forwards a verified, internal identity header* to upstreams so they do not each re-verify against the identity provider. Here is that filter as middleware (the principle holds whether it is a Kong plugin, an Envoy `ext_authz` call, or hand-written BFF middleware):

```go
// authFilter validates the client's JWT once at the edge and injects a
// trusted internal identity header for upstream services to consume.
func authFilter(next http.Handler, keys jwks.KeySet) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        raw := bearerToken(r)
        claims, err := jwt.Verify(raw, keys) // signature + expiry + audience
        if err != nil {
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return // reject at the edge: the service never sees this request
        }
        // Strip any client-supplied identity header (never trust the client),
        // then inject our own, signed by the edge for downstream trust.
        r.Header.Del("X-User-Id")
        r.Header.Set("X-User-Id", claims.Subject)
        r.Header.Set("X-User-Scopes", strings.Join(claims.Scopes, " "))
        ctx := context.WithValue(r.Context(), identityKey, claims.Subject)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```

The two security subtleties here separate a junior implementation from a senior one. First, the filter *strips* any incoming `X-User-Id` before setting its own — otherwise a client could simply send `X-User-Id: admin` and impersonate anyone, because downstream services trust that header. Second, downstream services trust this header *only because* it arrives over the internal trust boundary; the moment that boundary is not enforced (no mTLS, a service reachable from outside the cluster), the whole scheme is forgeable. This is precisely why edge authn and internal [mTLS / zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) are complementary, not redundant — the gateway authenticates the *human*, the mesh authenticates the *service*.

## The decision: which edge shape, and build versus buy

Now the trade-off section the series demands, in two parts: which *topology* (direct, single gateway, BFF-per-client, gateway+mesh) and which *implementation* (build it yourself or buy a managed product). Take the topology first, because it is the more consequential choice and the one that has long-term org consequences.

![A decision matrix comparing direct-to-service, a single gateway, a BFF per client, and a gateway plus mesh across client payload fit, team autonomy, coupling to clients, ops cost, and single-point-of-failure risk](/imgs/blogs/the-api-gateway-and-backend-for-frontend-4.webp)

| Property | Direct to service | Single gateway | BFF per client | Gateway + mesh |
|---|---|---|---|---|
| **Client payload fit** | Poor — client assembles | Generic, one shape | Tailored per client | Tailored per client |
| **Team autonomy** | High but unsafe | Low — shared chokepoint | High — client team owns it | High |
| **Coupling to clients** | Tight — topology leaks | Loose | Loose | Loose |
| **Cross-cutting concerns** | Duplicated in every client | Once, at the edge | Per BFF (some dup) | Once at edge + once in mesh |
| **Ops cost** | Low (no edge) but high client cost | Medium | High — N edges to run | Highest — edge + mesh |
| **SPOF risk** | Spread across services | Single shared chokepoint | Per-client blast radius | Per-client + mesh control plane |
| **When it wins** | Internal-only, one trusted client | Small system, one client type | Multiple distinct clients + teams | Many services needing east-west policy too |

Read across the rows and the recommendations fall out. **Direct-to-service** is defensible only when the "client" is another internal trusted service in the same cluster (east-west traffic, which is the mesh's job, not the gateway's) or a tiny system with exactly one client you fully control. For anything public-facing, the topology leakage and duplicated plumbing make it a false economy.

**A single gateway** is the right starting point for a small system with one dominant client type. Do not build three BFFs on day one for an app that has only a web frontend; that is premature. The single gateway gets you encapsulation, one place for policy, and aggregation, at modest ops cost. Its failure mode is *growth*: the day you have multiple distinct clients and multiple teams, the single gateway becomes the chokepoint everyone waits on, and *that* is the day to split.

**BFF per client** is the right shape once you have multiple genuinely different clients (web, mobile, partner) and the teams to own them. You pay real duplication and run several edges, but you buy back team autonomy — the thing microservices exist for. The honest cost is the duplication: if you find yourself fixing the same bug in five BFFs, factor the shared plumbing into a thin edge proxy or a shared library, but resist re-merging the client-specific shaping.

**Gateway + mesh** is not really an alternative to the others; it is what happens when the system is large enough that you also need to govern *service-to-service* traffic (the east-west direction). The gateway still owns north-south (client→system); the mesh adds mTLS, retries, and observability for service→service. We will draw that boundary precisely in the last section. It is the highest-ops option and you reach for it because you have outgrown not having it, never because it is fashionable.

Now build versus buy.

![A decision tree branching first on buy versus self-run, where buy leads to managed services like AWS API Gateway and Apigee, and self-run branches into standalone proxies like Kong, in-mesh Envoy edges, and framework gateways like Spring Cloud Gateway](/imgs/blogs/the-api-gateway-and-backend-for-frontend-7.webp)

The options cluster into four buckets, and the right one depends mostly on how much operational surface you want to own and whether you need programmable aggregation:

- **Managed / fully hosted** — AWS API Gateway, Google Apigee, Azure API Management. You pay per request and per feature; you get auth, rate limiting, caching, and a console with near-zero ops burden. The trade-offs are cost at high volume (per-request pricing adds up fast at millions of requests), latency overhead (a managed gateway adds its own hop), and limited programmability (aggregation logic is awkward; these are great as a *policy gateway*, less so as a *BFF*). Reach for these when you want to ship fast and not run infrastructure, especially early.
- **Self-hosted standalone proxy** — Kong, Tyk, Traefik, NGINX. You run them yourself (containers, usually on Kubernetes); they are config-driven, high-performance, plugin-extensible, and cost only the compute. Great as the shared policy edge in front of BFFs. The cost is that *you* operate them — upgrades, scaling, the on-call when they fall over.
- **In-mesh edge** — Envoy as the mesh's ingress gateway (Istio's `istio-ingressgateway`, or Envoy Gateway). If you already run a mesh, your edge proxy is Envoy anyway, so using it as the north-south gateway too means one proxy technology to operate. The cost is the mesh's substantial complexity, which only pays off if you need east-west governance.
- **Framework / programmable gateway** — Spring Cloud Gateway, or just a plain web service (Go, Node, anything) that *is* your BFF. This is the natural home for the aggregation logic, because aggregation is code and a framework gateway lets you write it idiomatically. A BFF is almost always this bucket; the cross-cutting policy edge in front of it is one of the other three.

The pragmatic senior answer is usually a *combination*: a bought or self-hosted policy edge (managed gateway or Kong) handling TLS, coarse authn, and global rate limiting, with programmable BFFs (Spring Cloud Gateway, or a Node/Go service) behind it doing the client-specific aggregation. Buy the commodity plumbing, build the part that is specific to your product.

## Optimization: making the edge fast and cheap

The gateway sits on the critical path of *every single request* in the system, which makes it the place where small inefficiencies multiply enormously and small optimizations pay back enormously. Here is where the real wins are, with numbers.

**Response aggregation is itself the first optimization** — we did that math already: 570ms to 230ms by moving the fan-out from the slow client link to the fast datacenter fabric. But within the BFF, *how* you fan out matters. Serial fan-out (call orders, await, call products, await, call shipping, await) makes the BFF's time the *sum* of all calls. Concurrent fan-out makes it the *max*. For five 40ms calls, that is the difference between 200ms and 40ms inside the BFF — a 5× improvement from nothing but changing serial awaits to concurrent ones. This is the single most common BFF performance bug: a developer writes the calls top-to-bottom with `await` on each line, and the BFF is needlessly serial.

**Edge caching** is the highest-leverage win when you have it. Reference data — product names, images, descriptions, category trees — changes rarely and is read constantly. Caching it at the gateway with a short TTL means the catalog service never sees the repeat reads.

#### Worked example: edge cache hit ratio and capacity saved

Suppose the catalog service receives 5,000 product-detail reads per second across the fleet, and product data changes at most a few times a day per product. With a 60-second edge cache TTL on `GET /api/products/{id}`, and a realistic access pattern where the top products dominate (a Zipf-ish distribution), you typically see a cache hit ratio of 90–95%. At 95% hits, only **250 requests per second** reach the catalog service instead of 5,000 — a **20× reduction in load**. If each catalog instance handles 500 rps comfortably, you go from needing 10 instances to needing 1 (with headroom for failover, say 2). That is an 80% reduction in the catalog service's compute bill, paid for by a cache that costs the gateway a few milliseconds and a little memory. The caveat is staleness: a 60-second TTL means a price or availability change can be up to 60 seconds stale at the edge, so you cache the *stable* fields (name, image, description) and *never* cache the volatile ones (live inventory count, current price during a flash sale). Cache discipline — what is safe to cache and for how long — is the whole game, and the broader strategy is the subject of [caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services).

**Connection pooling and HTTP/2 to upstreams.** A naive gateway opens a fresh TCP (and possibly TLS) connection to an upstream for every request, paying a handshake each time. A production gateway maintains a *pool* of warm, reused connections to each upstream, ideally over HTTP/2 so many concurrent requests multiplex over a few connections. The win: eliminating per-request connection setup shaves the TLS/TCP handshake (often 1–2 RTTs, which inside a DC is small but at scale is meaningful CPU) and, more importantly, bounds the number of connections each upstream must accept. Without pooling, a gateway serving 10,000 rps to an upstream might open 10,000 connections; with HTTP/2 multiplexing it might need a few hundred. Connection count is a real resource limit on the upstream, and exhausting it is a classic cascading-failure trigger.

**Compression at the edge.** The gateway is the right place to gzip or Brotli-compress responses going to the client, because the client link is the slow, bandwidth-constrained one. A 200KB JSON order list compresses to perhaps 20KB — a 10× bandwidth saving on exactly the link where bandwidth costs the user money and time. Internal service-to-service responses generally do not need compression (the DC link is fast and CPU is the scarcer resource there); the asymmetry is the point.

#### Worked example: fan-out throughput, timeout budget, and the connection cost

Capacity planning at the edge is where the aggregation pattern shows its real price, so put numbers on it. The mobile BFF handles 2,000 client requests per second at peak. Each request fans out to, on average, four upstream calls (orders, then products, shipping, and recommendations). That means the BFF generates 2,000 × 4 = **8,000 upstream calls per second** — the fan-out *multiplies* internal traffic by the fan-out factor, which is the first thing capacity planners forget. The shipping service, which only sees one of those four calls per request, must therefore be sized for 2,000 rps, not the 2,000 the BFF receives; check each upstream against the *per-upstream* share, not the client rate.

Now the timeout budget. The client gives the screen 250ms before it would rather show a partial view. Out of that, reserve ~20ms for the client→BFF network hop and the BFF's own serialization and merge work, leaving ~230ms for upstream work. Because the essential orders call (120ms deadline) runs *before* the recommendations call it feeds (80ms deadline), those two are serial: 120 + 80 = 200ms in the worst case, which fits inside 230ms with 30ms of slack. The product and shipping calls run concurrently *within* that window, so they do not add to the budget as long as each is under the orders+recs chain. The lesson: **the BFF's overall deadline must be at least the longest serial chain through its dependency graph, plus its own overhead** — not the sum of all calls, and not the single slowest call, but the slowest *path*. Get this wrong and you either set the deadline too low (and degrade healthy requests) or too high (and let a slow upstream eat the whole budget).

Finally the connection cost. At 8,000 upstream calls per second spread across four services, naive one-connection-per-call would churn 8,000 TCP setups per second. With HTTP/2 multiplexing and a warm pool, the BFF holds perhaps 50–100 connections *total* to each upstream and multiplexes all the calls over them — a four-orders-of-magnitude reduction in connection churn. That is the difference between an upstream that spends its CPU on your business logic and one that spends it accepting and tearing down sockets. Connection pooling is not a micro-optimization at this scale; it is the thing that keeps the upstreams from falling over under their own accept loop.

**Measure the win the right way.** The metric that matters at the edge is *client-perceived* latency at the tail — p99 of time-to-first-useful-byte, measured at the gateway, broken down per route. Aggregating the wrong way can *hurt* tail latency even while improving the mean, which brings us straight to the stress test.

## Stress-testing the edge: when the gateway is the bottleneck

A design is only as good as its behavior under failure, so let me break this one deliberately. There are two failure modes that every edge engineer must have a plan for: *one upstream in the aggregation is slow*, and *the gateway itself is the chokepoint*.

#### The aggregation tail-latency trap

Here is the subtle math that catches teams. Aggregation means the BFF waits for *several* services and the screen is not ready until the *slowest* of them returns. If you wait for all of them, your p99 is governed by the *worst* p99 in the fan-out — and tail latencies *compound* across a fan-out.

Concretely: suppose the order, product, and shipping services each have a p99 of 100ms — meaning 1% of calls to each are slower than 100ms. The probability that *all three* calls in a single fan-out return under 100ms is roughly 0.99 × 0.99 × 0.99 ≈ 0.970. So the probability that *at least one* is slow is about 3% — meaning the *aggregated* call's p97, not its p99, is already 100ms. Fan out to ten services and the probability all ten are fast is 0.99^10 ≈ 0.90, so 10% of your aggregated requests hit somebody's tail. **The more services you aggregate, the worse your tail latency gets, because you are exposed to the union of everyone's tails.** This is why a god-gateway that aggregates fifteen services has terrible p99 even when every individual service looks healthy.

The defenses are exactly the ones in the code earlier, and now you can see *why* they matter:

- **Per-call deadlines** cap how bad the tail can get. If shipping's deadline is 60ms, then no matter how slow shipping gets, it cannot push the aggregated call past 60ms — it just returns nothing and the screen degrades. The deadline converts a *latency* failure into an *availability-of-one-field* failure, which is far less bad.
- **Partial responses** (the `Degraded` flag) mean a slow non-essential service costs you a missing badge, not a failed screen. This is *graceful degradation*, the subject of [handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation), and the edge is where it most often pays off.
- **Hedged requests** for the truly latency-critical, idempotent calls: send a second request after, say, the p95 of the first, and take whichever returns first. This trades a little extra load for a dramatically tighter tail. Use it sparingly — it amplifies load — but it is a real tool.

![A branching graph showing the mobile BFF with a 250ms deadline fanning out to a fast order service, a fast product service, and a slow shipping service that exceeds the deadline, where the merge step combines the ready results and the slow one is timed out into a 200 partial response with the shipping field set to null](/imgs/blogs/the-api-gateway-and-backend-for-frontend-8.webp)

#### The gateway-as-chokepoint failure

Now the scarier one: the gateway itself falls over. Because every request crosses it, the gateway is a *single point of failure* by construction. If it goes down, the *entire system* is unreachable from outside, no matter how healthy your services are. And it can be taken down not by its own bug but by a slow *upstream*, through the exact thread-exhaustion mechanism that opened the [fundamentals post](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies).

![A timeline of a gateway becoming the bottleneck, starting with the shipping service p99 climbing to two seconds, then BFF threads blocking on the fan-out, then the gateway connection pool filling, then all routes returning 503, then timeout caps shedding the shipping calls, and finally partial payloads restoring traffic](/imgs/blogs/the-api-gateway-and-backend-for-frontend-6.webp)

Walk the timeline. At T+0, the shipping service's p99 climbs from 60ms to 2 seconds — not down, just slow (a GC pause, a slow query, a noisy neighbor). At T+15s, the BFF, calling shipping *without a tight per-call deadline*, holds its request-handling threads (or goroutines, or event-loop slots) open waiting. At T+45s, every one of the BFF's worker threads is parked on a shipping call that will not return in time; the BFF has no capacity to handle *new* requests. At T+90s, the connection pool is exhausted and the BFF returns 503 to *everything* — including the catalog browse that has nothing to do with shipping. One slow non-essential upstream has taken down the entire edge. At T+3m, the deadline you (hopefully) configured finally caps the shipping calls, threads free up, and the BFF starts shedding the shipping work. At T+5m, the BFF is returning partial payloads (orders without shipping status) and traffic recovers.

The lesson is stark: **an unbounded aggregation turns one slow dependency into a total edge outage.** The defenses against the chokepoint, beyond the per-call deadlines already covered:

- **Bulkheading.** Isolate the resource pools so that calls to one upstream cannot starve calls to another. If shipping has its own bounded thread pool of, say, 50 workers, then a slow shipping service can park at most 50 workers, not the whole gateway. The catalog browse path uses a *different* pool and stays healthy. This is the bulkhead pattern from [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads).
- **Circuit breakers.** When shipping has been failing or timing out for a while, *stop calling it* for a cooldown period and immediately return the degraded response. This stops the gateway from wasting threads on a known-bad upstream and gives shipping room to recover instead of being hammered by retries.
- **Load shedding at the edge.** When the gateway itself is saturated, shed low-priority traffic *cheaply* at the front door — return 503 fast to non-critical requests so the critical ones get through. Shedding at the edge is the cheapest possible place to shed, which is why the gateway is the right place for it. Again, [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) goes deep on the policy.
- **Run the gateway redundantly.** Never one gateway instance. Run several behind a load balancer across availability zones, so the loss of one instance — or one whole AZ — does not take the edge down. The gateway being a SPOF *architecturally* does not mean it has to be a SPOF *operationally*; redundancy and health-checked load balancing turn the single logical edge into many physical replicas.

Put together, the stress test teaches the central operating rule of the edge: **the gateway must protect itself from its upstreams.** Every call out of the gateway is bounded, isolated, and breakable, so that no single misbehaving service can convert "shipping is slow" into "the whole company is offline."

## Case studies: how the real systems did it

Patterns earn their place when they survive contact with production. Three real-world stories, each teaching a distinct lesson.

**Netflix and device-specific adapters — the proto-BFF.** In the early 2010s, Netflix faced an extreme version of the chattiness-and-diversity problem: their service was consumed by a bewildering range of devices — game consoles, smart TVs, Blu-ray players, phones, browsers — each with wildly different capabilities, screen sizes, and network conditions. A single generic API could not serve all of them well; a TV wants a different payload than a phone. Netflix's answer (described in their engineering writing of the era, notably by Daniel Jacobson) was a layer of *device-specific adapter scripts* running at the edge: each device team wrote a small adapter (in Groovy, dynamically loaded) that aggregated the underlying Netflix services into exactly the shape *that device* needed. That is the BFF idea in everything but name, several years before the name existed: the consumer team owns a thin edge that tailors the platform's services to its client. The lesson Netflix's experience teaches is that **client diversity, not just client count, is what justifies per-client edges** — when your clients differ enough that one payload genuinely cannot serve them all, you tailor at the edge or you make every client miserable. (It also taught the *cost*: maintaining many adapters is real operational weight, and Netflix later invested heavily in tooling to manage that surface.)

**SoundCloud coins "BFF."** Around 2015, SoundCloud's engineers, in the middle of decomposing their monolith, hit the shared-gateway problem head-on: a single public API was serving both their own mobile apps and third-party developers, and the two had irreconcilable needs — the mobile apps wanted to evolve fast in lockstep with app releases, the public API needed to stay stable for outside developers. Phil Calçado and colleagues' resolution was to split the edge: a dedicated backend *for* the mobile frontend, owned by the mobile team and free to change with the app, separate from the stable public API. They named it "Backend for Frontend," and the name stuck because it captured the essential insight precisely — *the edge exists to serve a specific frontend, so let the frontend's team own it.* The lesson: **the strongest argument for a BFF is organizational, not technical.** It is about which team can change which thing without waiting on another team. The latency win is real but secondary; the autonomy win is what makes the duplication worth paying.

**The gateway-as-bottleneck post-mortem.** A pattern documented in many companies' incident write-ups (and a category of failure rather than one famous incident) is the day the shared gateway became the thing that takes everything down. The shape is always similar: a single, heavily-shared API gateway accretes responsibilities — auth, rate limiting, dozens of routes, some aggregation, maybe a little business logic that crept in "just this once." It becomes the most critical and most frequently-changed component in the system, owned by a platform team that every product team files tickets against. Then one of two things happens: either a *deploy* of the gateway (to add one team's route) introduces a bug that breaks *everyone*, because everyone shares it; or a *slow upstream* drains the shared gateway's pool and 503s the whole site, as in our timeline. The lesson is the negative space of this whole post: **a gateway's blast radius is everything behind it, so the more you centralize into it — logic, routes, ownership — the bigger the thing that goes down together.** This is the precise failure the BFF pattern and the "configurable, not programmable" rule exist to prevent. Keep the shared edge thin and dumb; push client-specific logic into per-client BFFs with smaller blast radii; never let business logic colonize the edge.

A fourth, briefer one worth naming: **Amazon's two-pizza teams and service ownership.** Amazon's organizational model — small teams each owning a service end-to-end, talking to each other only through hardened APIs — is the deep reason BFFs work there. When the team that owns the mobile experience also owns the mobile BFF, the API boundary and the org boundary line up (Conway's Law working *for* you instead of against you — the subject of the upcoming series post on [Conway's Law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices)). The lesson: an edge architecture that fights your org chart will lose. Design the edge so each piece has a clear single owner, and the BFF-per-client shape falls out naturally.

## Where the gateway sits versus a service mesh

One persistent confusion deserves its own section, because engineers new to this conflate the gateway and the service mesh, and they solve different problems. The distinction is *direction of traffic*.

![A before and after style comparison contrasting north-south gateway traffic from client to system handling authn rate limiting and aggregation against east-west mesh traffic between services handling mTLS retries and sidecars, showing the two govern different traffic directions](/imgs/blogs/the-api-gateway-and-backend-for-frontend-9.webp)

**North-south traffic** is traffic crossing the boundary of your system — clients calling in, your system calling out. This is the *gateway's* domain. The gateway is concerned with the things that matter at a trust boundary: terminating public TLS, authenticating external callers, rate-limiting untrusted clients, aggregating responses for client convenience, translating between public and internal protocols. North-south is about the *edge*.

**East-west traffic** is traffic *between* your services, all inside the trust boundary — the order service calling the inventory service, the BFF calling the catalog service. This is the *service mesh's* domain. The mesh (Istio, Linkerd) deploys a sidecar proxy next to every service and governs *every* internal hop: mutual TLS so services authenticate each *other*, retries and timeouts and circuit breaking applied uniformly without each service implementing them, fine-grained traffic shifting for canary deploys, and per-hop observability. East-west is about the *fabric between services*.

They are complementary, not competing. A large system often runs *both*: a gateway at the north-south edge and a mesh on the east-west fabric. They even share technology — the mesh's ingress gateway is usually Envoy, and you can use that same Envoy as your north-south edge gateway, which is the "in-mesh edge" build option from earlier. But you do not *need* a mesh to have a gateway. A small system can have a perfectly good gateway and no mesh at all, handling its limited east-west resilience with libraries inside each service. You reach for the mesh when the number of services and the need for *uniform* east-west policy (especially mTLS and consistent retries across a large fleet) outgrows what per-service libraries can maintain. That threshold, and the substantial operational cost of running a mesh, is exactly what [service mesh: Istio, Linkerd, and when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) takes head-on. For now, hold the clean mental division: gateway owns the door, mesh owns the hallways.

## When to reach for this (and when not to)

Decisive guidance, because the series promises it and because "use a gateway" is not automatically correct.

**Reach for a single gateway as soon as you have a public client and more than two or three services.** The encapsulation, the single policy point, and the ability to refactor internals without breaking clients are worth the modest cost almost immediately. The bar for "we need a gateway" is low and you will clear it fast. Even a thin one — just TLS termination, auth, and routing — earns its keep on day one.

**Reach for aggregation in the gateway when you have a chatty client on a slow link.** Mobile apps are the canonical case. If your client is making more than two or three round trips to paint one screen, and especially if any of them are serialized, aggregation at the edge is a large, cheap win. Do the latency math; it usually pays for itself.

**Reach for BFF-per-client when you have multiple genuinely different clients *and* the teams to own them.** The trigger is not client count alone — it is client *divergence* (a TV and a phone want different payloads) combined with *organizational* friction (the mobile team keeps waiting on the platform team). When both are true, splitting into per-client BFFs buys autonomy that is worth the duplication. When they are not — one web app, one small team — a single gateway is simpler and you should not split prematurely.

**Reach for gateway + mesh when east-west policy outgrows libraries.** Many services, a hard requirement for uniform mTLS, the need for consistent retry/timeout/canary behavior across the fleet without touching every service's code — that is when the mesh's cost pays off. Until then, a gateway plus per-service resilience libraries is lighter and easier to operate.

**Do not reach for a gateway when:** you have exactly one internal trusted client and no public exposure (the "client" is another service — that is the mesh's job). You are at the modular-monolith stage with no real service fan-out yet (the [monolith-first post](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) argues for staying there longer than you think). Or the gateway you are about to build would, in practice, be the place all your business logic lives — in which case you have not built a gateway, you have rebuilt the monolith with extra latency, and you should stop and reconsider your service boundaries instead.

**Above all, do not let the gateway become programmable for everyone.** The single most important "when not to" is: do not let business logic into the shared edge, and do not let one shared gateway accumulate every team's routes and rules until it is the thing that takes the whole company down on every deploy. If the edge needs to be programmable — and aggregation legitimately does — make it a *per-client BFF* with a small blast radius and a single owner, not a shared god-service. That single discipline is the difference between a gateway that enables microservices and one that quietly recreates the monolith you worked so hard to escape.

## Key takeaways

1. **A gateway exists to give clients back the coarse-grained, encapsulated front door the monolith had** — without the monolith. It solves chattiness, duplicated cross-cutting concerns, topology leakage, and the lack of a single policy point. If you have a public client and several services, you need one.
2. **The gateway's job list is fixed**: TLS termination, coarse authn, rate limiting and shedding, routing and load balancing, protocol translation, response aggregation, edge caching, observability. Its forbidden list is just as fixed: no business logic, no per-resource authorization, no domain transformation, no stateful workflow.
3. **The governing rule is "configurable, not programmable."** Routing, auth, rate limits, caching — config. The moment you are writing and deploying meaningful logic in the shared edge, you are accreting a monolith. The one deliberate exception is the BFF's aggregation, which is why a BFF must be small and single-owner.
4. **Aggregation's win is moving the fan-out from the slow client link to the fast datacenter fabric** — ShopFast went from 570ms to 230ms by replacing six mobile round trips with one. Always fan out *concurrently* (max latency, not sum) and always give each call its own deadline.
5. **Backend-for-Frontend means one tailored edge per client type, owned by that client's team.** Web BFF, mobile BFF, partner BFF — each small, each serving one client, each free to deploy on its own schedule. The argument is organizational autonomy first, latency second.
6. **Aggregation makes tail latency worse, because you are exposed to the union of every upstream's tail.** p99 across a ten-way fan-out is governed by everyone's p99. Per-call deadlines, partial responses, and (sparingly) hedged requests are how you bound it.
7. **The gateway is a single point of failure by construction**, and a slow upstream can drain its pool and 503 the whole system. Defend it with per-call deadlines, bulkheads, circuit breakers, edge load-shedding, and — non-negotiably — redundant instances across availability zones.
8. **Edge caching of stable reference data is the highest-leverage performance win** — a 95% hit ratio turns 5,000 rps into 250 rps, an 80% compute saving — but only for fields that are safe to be seconds stale.
9. **Gateway is north-south, mesh is east-west.** The gateway owns the door (client→system); the mesh owns the hallways (service→service). They complement each other; you do not need a mesh to have a gateway, and you reach for the mesh only when uniform east-west policy outgrows per-service libraries.
10. **Build-vs-buy is usually "both": buy the commodity policy edge, build the product-specific BFF.** A managed or self-hosted gateway (AWS API Gateway, Kong) for TLS, authn, and global rate limiting, with programmable per-client BFFs (Spring Cloud Gateway, Node, Go) behind it for aggregation.

## Further reading

- Sam Newman, *Building Microservices* (2nd ed., O'Reilly) — Chapter 5 on inter-process communication and the section on API gateways and BFFs is the canonical practitioner treatment.
- Chris Richardson, *Microservices Patterns* (Manning) — the API Gateway pattern and the BFF variant, with the aggregation and composition details.
- Phil Calçado, "The Back-end for Front-end Pattern (BFF)" — the original write-up from the SoundCloud experience that named the pattern.
- Sam Newman, "Backends For Frontends" (samnewman.io) — a clear, opinionated treatment of when to split the edge per client and when not to.
- Daniel Jacobson et al., Netflix Tech Blog — the writing on Netflix's API and device-specific edge adapters that prefigured the BFF.
- AWS API Gateway, Kong, and Envoy official documentation — for the concrete routing, plugin, and filter configuration referenced above.
- In this series: [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), [REST vs gRPC vs GraphQL for service APIs](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis), [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice), and forward to [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing), [service mesh: Istio, Linkerd, and when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one), [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding), and [authentication and authorization with OAuth2, JWT, and token propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation).

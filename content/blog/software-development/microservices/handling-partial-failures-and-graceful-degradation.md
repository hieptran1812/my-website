---
title: "Handling Partial Failures and Graceful Degradation"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "In a distributed system partial failure is not an edge case, it is the steady state, and this is the practitioner's craft of designing so that when some dependencies fail the system bends instead of breaks: dependency classification, fallbacks, fail-open versus fail-closed, static stability, and the UX of degradation."
tags:
  [
    "microservices",
    "graceful-degradation",
    "partial-failure",
    "resilience",
    "distributed-systems",
    "software-architecture",
    "backend",
    "fault-tolerance",
    "fallbacks",
    "static-stability",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/handling-partial-failures-and-graceful-degradation-1.webp"
---

At 9:14 on a Tuesday morning the ShopFast product page started returning blank 500s to about a third of shoppers, and the on-call engineer who got paged spent the first frantic ten minutes looking in entirely the wrong place. The order service was healthy. The payment gateway was healthy. The catalog database was serving reads in single-digit milliseconds. Every dashboard that mattered was green. What was actually broken was the *recommendations* service — the little "You might also like" carousel at the bottom of the page that nobody had ever considered important enough to put on a runbook. A bad deploy had pushed a model that took four seconds to score a request, the page's aggregator called recommendations synchronously with no timeout worth the name, the request threads piled up waiting, the thread pool drained, and a page that was *mostly* working — price correct, stock correct, add-to-cart fine — served a hard error because one decorative section at the bottom could not answer. The carousel took down the store.

That incident is the entire subject of this post compressed into one morning. Nobody wrote a bug. Every individual service did roughly what it was told. The failure lived in the *spaces between* the services — in the unexamined assumption that every dependency would always answer, that a call either returns a value or you might as well give up. In a monolith that assumption is almost true: an in-process function call to your recommendations module either returns or throws, and if it throws you catch it locally and the rest of the request proceeds. The moment recommendations became a separate service across a network, that function call grew three new failure modes nobody designed for, and the most dangerous one — *it is taking a long time and I do not yet know if it will ever answer* — has no equivalent in single-process code at all.

![A before and after comparison contrasting a no fallback path where a downed recommendations service blocks the page thread and returns a full page error against a degraded path where a short timeout and an empty fallback let the page render its core sections](/imgs/blogs/handling-partial-failures-and-graceful-degradation-1.webp)

The figure above is the lesson the on-call engineer learned that morning, drawn as two paths through the same incident. On the left is what actually happened: recommendations is down, the page thread blocks on it for thirty seconds, and the whole page returns a 500 — zero percent available even though three of four backends were perfectly fine. On the right is the page they shipped by lunchtime: the same recommendations outage, but now the call has a two-hundred-millisecond timeout and a fallback that returns an empty carousel, so the page renders all of its core sections and the only visible damage is a missing "You might also like" strip that most users never scroll to. Same failure, same blast, two completely different outcomes — and the only difference is that somebody had *designed for* the failure instead of hoping it would not happen. By the end of this post you will be able to take any cross-service page or workflow and answer, for every dependency it touches, the questions that separate a junior who hopes the dependencies stay up from a senior who knows exactly which ones are allowed to fail and what the system does when they do: which dependencies are critical and which are optional, whether each one fails open or fails closed, what the fallback actually returns, and how you find out it is broken in a test instead of at 9:14 on a Tuesday.

## Partial failure is the steady state, not the exception

Start with the arithmetic, because it reframes the whole problem and most juniors have never done it. Suppose every service in your fleet is genuinely excellent: each one is up 99.9 percent of the time, which is about eight hours and forty-five minutes of downtime a year — a number most teams would be proud of. Now ask: at any given instant, what is the probability that *all* of them are simultaneously healthy? If you have ten such services, the probability that all ten are up at once is 0.999 to the tenth power, which is about 99.0 percent. With fifty services it is 0.999 to the fiftieth, about 95.1 percent. With a hundred services — Monzo famously runs more than 1,500, but a hundred is an ordinary mid-size fleet — it is 0.999 to the hundredth, about 90.5 percent. Read that again: with a hundred 99.9-percent services, *roughly one time in ten, at any random moment, at least one of them is down.* Not once a year. Continuously, somewhere, something is degraded.

This is the fact a junior has to internalize before any technique makes sense: **in a distributed system of meaningful size, the all-healthy state is the rare state.** You will essentially never observe a moment when every service, every database replica, every cache node, and every network link is simultaneously perfect. The steady state of a large microservices system is "mostly working, with something somewhere broken right now." If your design only behaves correctly when *everything* is up, your design is wrong by construction, because the everything-up condition almost never holds. The question is never "what happens if a dependency fails" — something is always failing — it is "what happens to *this* request when *that* dependency is the one that is currently failing." Designing for partial failure is not defensive paranoia; it is designing for the normal case.

There is a second, sharper fact stacked on top of the arithmetic, and it is the one that makes distributed failure genuinely harder than local failure. When you call a function in your own process, it has exactly two outcomes: it returns a value, or it throws. When you call a service across a network, it has *three*: it succeeds, it fails with an error you receive, or — the killer — it *times out*, and you are left in a state of genuine uncertainty about whether the work happened. This is one of the [fallacies of distributed computing that the inter-service communication post lays out](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies): the network is not reliable, and "I sent a request and heard nothing back" tells you nothing about whether the receiver processed it. The payment might have gone through and the response got lost on the way back. The order might be half-written. The third outcome — *succeeded, failed, or timed-out-unknown* — is the thing that makes partial failure a design problem rather than a try/catch problem, and it is why a sibling like [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) exists at all: half of resilience is making the retry of an unknown-outcome call safe to perform.

It is worth sitting with that third outcome a moment longer, because juniors reach for a retry as the universal cure and a senior knows exactly when a retry is a loaded gun. If a read times out — "give me the product price" — a retry is harmless: reading twice does no damage, you just want the answer. But if a *write* times out — "charge this card," "decrement this stock" — the retry is dangerous, because the original might have *succeeded* and the response merely got lost on the way back, so retrying charges the card twice. The timed-out-unknown outcome forks on whether the operation is *idempotent*: a read or an idempotent write can be safely retried, a non-idempotent write cannot, and the only way to make the dangerous case safe is to attach an idempotency key so the receiver recognizes and de-duplicates the replay. This is the entire premise of the sibling [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) post, and the practical takeaway for degradation is sharp: your *retry-then-fall-back* policy is only safe if the retried call is idempotent, so you must know, per dependency, whether a timeout means "safe to try again" or "must reconcile, never blindly replay." A fallback that blindly retries a non-idempotent write is not resilience; it is a double-charge generator.

So the discipline of this post divides cleanly. First you must *detect* that a dependency is failing or slow — that is the job of timeouts, retries, and circuit breakers, which the sibling [resilience patterns post on timeouts, retries, circuit breakers, and bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) covers in mechanical detail and which I will lean on rather than re-derive. Detection is necessary but it is not the answer. A circuit breaker that opens and then throws an exception up to your page handler has *detected* the failure and *done nothing useful with it* — you have merely converted a slow failure into a fast one. The subject of this post is the second half: once you have detected the failure, what do you *do*? **Timeouts and breakers detect failure; graceful degradation is the response.** This post is about the response.

## Dependency classification: the decision you make before the incident

Here is the single most important idea in the entire post, and it is one you make on a whiteboard months before any outage: **not all dependencies are equal, and you must classify each one as critical or optional before you can degrade anything.** A critical dependency is one without which the core function of the request is meaningless or unsafe — for a checkout, the payment service is critical, because "checkout" without taking money is not checkout. An optional dependency is one that *enriches* the response but whose absence still leaves a correct, useful result — the recommendations carousel, the "customers also viewed" strip, the loyalty-points preview, the estimated-delivery-date widget. The defining test is brutal and clarifying: *if this dependency returned nothing at all, is the response still correct and useful?* If yes, it is optional and you must design a fallback. If no, it is critical and you must decide what failing safely looks like.

The cardinal sin — the one that caused the ShopFast morning — is treating an optional dependency as if it were critical *by accident*, simply by calling it the same way you call a critical one and never deciding otherwise. Nobody at ShopFast ever made a conscious decision that recommendations should be able to take down the store. They just wrote `recs = recommendationsClient.get(productId)` in the same blocking, un-fallback'd style they wrote `price = priceClient.get(productId)`, and the default behavior of "an unhandled failure propagates up and fails the request" silently promoted recommendations to the same tier as price. **The absence of a classification decision is itself a decision — and it is always the wrong one, because it makes every dependency critical.**

![A branching graph showing a product BFF aggregator fanning out to a critical price service, a critical inventory service, an optional reviews service, and an optional recommendations service, with the price and inventory edges marked must succeed and all four converging on a partially renderable page](/imgs/blogs/handling-partial-failures-and-graceful-degradation-2.webp)

The figure shows the ShopFast product page as a dependency graph, and the classification is drawn right into it. The product BFF — the [backend-for-frontend aggregator the API gateway post describes](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), whose whole job is composing one page from several backends — fans out to four services. Price and inventory are marked critical: their edges say *must succeed*, because a product page that shows the wrong price or claims an out-of-stock item is in stock is worse than no page at all. Reviews and recommendations are marked optional: their edges carry no such requirement, because a product page with a missing reviews section is a slightly poorer page, not a wrong one. The page node at the bottom is labeled "partial ok" — its contract with the user is that it renders the core as long as the critical deps answer, and it tolerates any subset of the optional deps being absent. That single picture, drawn before launch, is what would have saved the Tuesday morning.

Classification is not always binary, and a senior draws finer lines. A useful refinement is three tiers. **Critical for correctness**: getting it wrong corrupts data or loses money (payment, inventory decrement, the order state machine). **Critical for the core experience**: without it the page is useless but nothing is corrupted (the product's price and title — you genuinely cannot sell a product whose price you do not know). **Optional / enriching**: the page is fully functional without it (reviews, recs, recently-viewed). The distinction between the first two matters because the *failure modes* differ — for a correctness-critical dependency you often **fail closed** (refuse to proceed), while for a core-experience-critical dependency you might **degrade** (show the product but disable the buy button, or serve a stale price with a banner) rather than show a blank error. We will get to that decision in a moment, but the point stands: you cannot make any degradation decision until you have written down, per dependency, which tier it is in.

#### Worked example: the blast radius of one unclassified dependency

Let me put the cost of skipping classification in numbers, because it is larger than intuition suggests. ShopFast's product page is served at 4,000 requests per second at peak. The four backend calls each normally return in p99 40ms, and the page composes them concurrently, so a healthy page renders in about 50ms. Now recommendations degrades to a 4-second response — not down, just slow, which is the more common and more dangerous case. Because the aggregator calls it with no timeout, every product-page request now holds its handler thread for the full 4 seconds waiting on recs. The aggregator runs 200 worker threads per instance across 20 instances, so 4,000 threads total. At 4,000 requests per second each holding a thread for 4 seconds, the system needs 4,000 × 4 = 16,000 concurrent threads to keep up, but it has 4,000. Within one second the pools are full; within two the queues overflow and the aggregator starts rejecting *every* product-page request — including the 99.97 percent of the page's value (price, inventory, reviews) that has nothing to do with recommendations. One unclassified optional dependency at 1/100th of the page's importance just took the *entire* page from 4,000 successful requests per second to roughly zero. That is the blast radius of a missing classification: not "the recs carousel is empty" but "the store is down," and the multiplier between the two is the whole point of this post.

## Graceful degradation: what the fallback actually returns

Once a dependency is classified optional, "degrade it" is a verb that needs a concrete object: degrade it *to what*? This is where juniors wave their hands and seniors get specific, because the *quality* of a fallback is the difference between a page that looks intentional and a page that looks broken. There are five fallback values worth knowing by name, roughly in descending order of how good they are.

The best fallback is a **cached value** — the last good answer you got from the dependency, served stale. If the reviews service is down, serving the reviews you cached two minutes ago is nearly as good as live reviews and the user cannot tell the difference. The second best is a **sensible default** — a value that is generically reasonable when you have nothing better. If the personalized recommendations service is down, fall back to the *non*-personalized "top sellers in this category," which is a perfectly good carousel that happens to be the same for everyone. The third is an **empty-but-valid response** — return an empty list rather than an error, and let the section render as "no recommendations right now" or simply not render at all. The fourth is a **partial response** — return what you *do* have and omit what you do not, which for a composed page means rendering the sections whose backends answered and dropping the sections whose backends did not. The worst "fallback," and the one to avoid, is **propagating the error**, which is what no-fallback code does by default.

```python
# A fallback wrapper: try the live call, fall back to cache, then default, then empty.
# The key property: this function NEVER raises. Optional deps must not throw upward.
async def get_recommendations(product_id: str) -> list[Recommendation]:
    try:
        # Short timeout: an optional dep gets a tighter budget than a critical one.
        return await recs_client.get(product_id, timeout_ms=200)
    except (TimeoutError, ServiceError, CircuitOpenError):
        # 1. Best: last-known-good value from cache, even if stale.
        cached = await recs_cache.get(product_id)
        if cached is not None:
            metrics.increment("recs.fallback.cache")
            return cached
        # 2. Sensible default: non-personalized top sellers for the category.
        category = await catalog_cache.get_category(product_id)  # also cached, also safe
        if category is not None:
            metrics.increment("recs.fallback.topsellers")
            return TOP_SELLERS.get(category, [])
        # 3. Empty-but-valid: a real, renderable, harmless answer.
        metrics.increment("recs.fallback.empty")
        return []
```

Three things in that snippet are load-bearing and easy to get wrong. First, **the function never raises** — every exit path returns a valid list, because the contract of an optional dependency's accessor is "I always give you something renderable." The instant an optional dependency's accessor can throw, you have re-coupled it to the request's success and undone the classification. Second, **the timeout is short** — 200ms for an optional dependency, not the multi-second default an HTTP client ships with. An optional section is not worth making the user wait; if it cannot answer in 200ms it does not get to be in this page render. Third, **every fallback path emits a metric** — `recs.fallback.cache`, `recs.fallback.empty` — so that "we are serving degraded recs" is *visible*, because the most insidious failure mode of graceful degradation is that it works so quietly nobody notices the dependency has been down for three days. We will return to that observability point; it is why a forward-link to [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) belongs here.

### Partial responses: render what you have

The page-composition case deserves its own treatment because it is the most common and the most visible form of degradation. The product BFF's job is to call four backends and assemble one page. The naive version uses a single try/catch around all four calls, so any one failure fails the whole page. The graceful version isolates each call and assembles the page from whatever succeeded, treating the four calls as independent and the optional ones as best-effort.

```typescript
// Partial-response aggregation: each call degrades independently.
// Critical calls (price, inventory) reject the whole page; optional calls never do.
async function buildProductPage(productId: string): Promise<ProductPage> {
  const [priceR, invR, reviewsR, recsR] = await Promise.allSettled([
    priceClient.get(productId, { timeoutMs: 300 }),     // critical
    inventoryClient.get(productId, { timeoutMs: 300 }),  // critical
    reviewsClient.get(productId, { timeoutMs: 200 }),    // optional
    recsClient.get(productId, { timeoutMs: 200 }),       // optional
  ]);

  // Critical: if either rejected (after its own retry/breaker), the page cannot render.
  if (priceR.status === "rejected" || invR.status === "rejected") {
    throw new PageUnavailableError("core product data unavailable");
  }

  return {
    price: priceR.value,
    inventory: invR.value,
    // Optional: degrade to a fallback value, never propagate the rejection.
    reviews: reviewsR.status === "fulfilled" ? reviewsR.value : reviewsFallback(productId),
    recommendations: recsR.status === "fulfilled" ? recsR.value : [],
    // Tell the client what degraded, so the UI can present it honestly (see UX section).
    degraded: [
      reviewsR.status === "rejected" ? "reviews" : null,
      recsR.status === "rejected" ? "recommendations" : null,
    ].filter(Boolean),
  };
}
```

Two design choices in that code are the whole art. `Promise.allSettled` rather than `Promise.all` is deliberate: `Promise.all` rejects the moment *any* promise rejects, which re-couples all four; `allSettled` waits for every call to settle and hands you the outcome of each independently, which is exactly the partial-failure model the network actually has. And the `degraded` array that ships to the client is the bridge to honest UX — the page does not silently pretend everything is fine; it tells the frontend *which* sections are running on fallbacks so the UI can decide whether to hide them, show a "temporarily unavailable" note, or quietly render the cached version. The [API gateway and BFF post](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) covers why this aggregation logic belongs in the BFF layer and not smeared across the frontend; here the point is narrower — the aggregator is the natural place to enforce the critical/optional contract, because it is the one component that sees all the calls at once.

### Feature toggles: shedding the optional under stress

A subtler form of degradation is not reacting to a *failed* dependency but proactively shedding load to *prevent* failure. When a system is under stress — a traffic spike, a degraded database, a region failover — the cheapest way to protect the core is to stop doing the expensive optional work *before* it tips you over. This is where feature flags earn their keep as a resilience tool, not just a release tool.

```python
# A feature-flag-gated optional section. Under stress, an operator (or an automated
# rule) flips the flag and the expensive personalization call is skipped entirely —
# the page still renders, just without the personalized strip.
async def maybe_personalize(user_id: str, page: ProductPage) -> ProductPage:
    if not flags.enabled("personalization", default=True):
        # Flag is off: shed this feature. Cheap, instant, no dependency call at all.
        metrics.increment("personalization.shed")
        return page
    try:
        page.personalized = await personalize_client.score(user_id, timeout_ms=150)
    except (TimeoutError, CircuitOpenError):
        page.personalized = None  # optional, so we just skip it
    return page
```

The pattern is worth dwelling on because it inverts the usual reaction to overload. The instinct under load is to retry harder, scale up, throw more capacity at the wall — all of which take minutes and often make things worse by amplifying load (the cascading-retry failure mode the resilience-patterns post warns about). Flipping a feature flag takes *seconds* and *removes* load. A "degradation mode" flag — or a set of them, one per optional feature — gives you a runbook step that is faster than any autoscaler: "we are in trouble, shed tiers 3 and 4 now, investigate the root cause with the core path safe." The automated version of this is the [load-shedding the rate-limiting post covers](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding), where the system flips its own degradation flags based on a queue-depth or latency signal, but even the manual version is one of the highest-leverage resilience tools a team can build, because it converts "scramble for capacity" into "press the button."

## Fail open versus fail closed: the decision that defines safety

Now the hardest and most consequential decision in the post, the one that gets asked in every serious design review and every senior interview: when a dependency you *cannot* simply skip is unavailable, do you **fail open** (proceed as if it had said yes) or **fail closed** (refuse to proceed)? The terms come from physical security — a "fail-open" door unlocks when the power dies (safe for fire escape, bad for a vault), a "fail-closed" door locks (safe for the vault, a death trap in a fire). In software the analogy is exact: a fail-open dependency, when down, defaults to *permit*; a fail-closed dependency, when down, defaults to *deny*.

![A decision tree rooted at a dependency unavailable node branching on whether it is on the critical path, with the critical branch leading to fail closed or degrade and the optional branch leading to fail open or feature toggle](/imgs/blogs/handling-partial-failures-and-graceful-degradation-4.webp)

The decision tree in the figure encodes the rule, and the rule is simpler than it looks: **fail in the direction where being wrong is cheap.** Walk the two canonical examples. The *authentication* service is down: do you fail open (let everyone in without checking) or fail closed (deny everyone)? You fail *closed*, every time, because the cost of failing open is that an attacker walks straight into every account during your outage — a security catastrophe — while the cost of failing closed is that legitimate users see "login is temporarily unavailable, please try again," which is annoying but harmless. The recommendations service is down: fail open (show a default carousel or hide it) or fail closed (refuse to render the page)? You fail *open*, because the cost of failing open is a generic carousel and the cost of failing closed is the Tuesday-morning outage. The asymmetry of consequences picks the default; the dependency's tier mostly correlates with it but the *consequence asymmetry* is the real decider.

```python
# Fail CLOSED: authorization. If the authz service is unreachable, DENY.
# The "safe" default for a permission check is "no", because a wrong "yes" leaks data.
async def can_access(user_id: str, resource: str) -> bool:
    try:
        return await authz_client.check(user_id, resource, timeout_ms=100)
    except (TimeoutError, CircuitOpenError, ServiceError):
        metrics.increment("authz.fail_closed")
        log.warning("authz unavailable, failing closed", user=user_id, resource=resource)
        return False  # deny. A wrong deny is an annoyance; a wrong allow is a breach.

# Fail OPEN: a non-blocking fraud SCORE that only enriches a decision.
# If the fraud service is down, default to "low risk" and let the order proceed —
# you would rather take a few risky orders than block ALL orders during an outage.
async def fraud_score(order: Order) -> float:
    try:
        return await fraud_client.score(order, timeout_ms=150)
    except (TimeoutError, CircuitOpenError, ServiceError):
        metrics.increment("fraud.fail_open")
        return 0.0  # treat as low risk; review async later via the outbox/event trail.
```

Notice the fraud example is the genuinely hard one, and it is where the decision stops being obvious. Fraud scoring *feels* security-critical, so the reflex is to fail closed — block any order you cannot score. But think through the consequence asymmetry under an *outage*: if the fraud service is down for ten minutes and you fail closed, you block *one hundred percent* of orders for ten minutes, which is a direct, measurable, large revenue loss, in exchange for preventing the *tiny fraction* of those orders that would have been fraudulent. If you fail open and queue every un-scored order for asynchronous review (you have the order, you can score it five minutes later when fraud recovers and claw back or cancel the genuinely bad ones), you take a small, recoverable risk to avoid a large, unrecoverable loss. That is usually the right call — but *usually*, not always, and the senior move is to make it explicitly, with the fraud team and the finance team in the room, rather than letting a default exception handler decide it. The decision also is not all-or-nothing: a sophisticated system fails closed for *high*-value orders (block the \$5,000 order it cannot score) and fails open for *low*-value ones (let the \$12 order through and review it later), tuning the threshold to the loss it can stomach.

The genuinely dangerous mistake here is the *silent* fail-open — a permission check wrapped in a `try/except` that returns `True` on error because somebody was lazy. This is how real breaches happen: a downstream entitlement service times out, the catch block returns "allowed" so the page does not error, and for the duration of the outage every user can see every other user's data. **Any fail-open decision on anything touching security or correctness must be a deliberate, reviewed, logged choice — never the accidental behavior of an over-broad exception handler.** When in doubt on a security boundary, fail closed; you can always loosen it deliberately later, but you cannot un-leak the data an accidental fail-open exposed.

## Static stability: keep running on last-known-good

There is a deeper resilience principle that the best systems are built on, and it has a name from Amazon's engineering culture: **static stability**. The idea is that a system should keep working *as well as it currently is* even when the thing that would normally let it *change* is unavailable. Concretely: separate the *data plane* (the path that serves user requests) from the *control plane* (the path that updates configuration, scales the fleet, recomputes routing). When the control plane is down, a statically stable data plane does not fail — it simply keeps doing whatever it was last told to do, on the last data it has, until the control plane comes back. The system is "stable" because it does not *need* the control plane to be up in order to keep serving; the control plane only changes things.

The canonical Amazon framing is about availability zones. If your fleet is spread across three AZs and one fails, a *dynamically* stable system reacts by launching replacement capacity in the surviving zones — but launching capacity needs the control plane (the API that provisions instances), and control planes are themselves software that can be degraded *exactly when the data plane is under stress*. A *statically* stable system instead pre-provisions enough capacity in every zone to absorb a zone loss *without launching anything*: if you run at 66 percent utilization across three zones, losing one leaves the other two at 99 percent, which holds, and you needed no control-plane action at all to survive. You paid for the spare capacity up front so that survival does not depend on a successful reaction during the failure. Static stability is, at its heart, the decision to **pay in advance with redundancy so that you do not have to make a successful API call in the middle of an emergency.**

![A branching graph showing a product read request entering the catalog service read path, which checks a cache that tolerates five minutes of staleness while the catalog database is down, returning a stale two hundred response with a banner on a hit or an honest five oh three on a cold miss](/imgs/blogs/handling-partial-failures-and-graceful-degradation-8.webp)

The same principle scales down from AZs to a single service's dependency on its database, which is the version you will build most often. The figure shows ShopFast's catalog read path made statically stable. The catalog service reads product data, and normally it reads through a cache backed by the catalog database. When the *database* goes down — a failover, a bad migration, a connection-pool exhaustion — a naive read path returns errors because every cache miss has to hit a database that is not answering. A statically stable read path instead treats the cache as the *primary* read source during the outage: on a cache hit it serves the last-known-good product data, stale but correct-enough, with a quiet "prices last updated a few minutes ago" banner; only on a genuine cache *miss* — data it has never seen or that has aged out — does it return an honest 503. The control plane here is "the database that refreshes the cache"; the data plane is "serving reads from the cache." Decouple them, and the catalog stays *readable* through a full database outage, which for a read-heavy product catalog is most of the value.

```python
# Static stability: serve stale-on-error rather than failing on a dead backend.
# The cache TTL governs FRESHNESS; a separate, longer "stale-while-down" budget
# governs how long we keep serving last-known-good when the DB is unreachable.
FRESH_TTL = 60          # seconds: after this we'd LIKE to refresh
STALE_BUDGET = 3600     # seconds: but we'll serve stale up to an hour if the DB is down

async def get_product(product_id: str) -> Product:
    entry = await cache.get(product_id)  # entry carries value + stored_at timestamp
    age = now() - entry.stored_at if entry else None

    if entry and age < FRESH_TTL:
        return entry.value                       # fresh: serve it, done

    # Stale-or-missing: try to refresh from the DB, but tolerate the DB being down.
    try:
        fresh = await db.read_product(product_id, timeout_ms=200)
        await cache.set(product_id, fresh, stored_at=now())
        return fresh
    except (DatabaseError, TimeoutError):
        if entry and age < STALE_BUDGET:
            metrics.increment("catalog.served_stale")
            return entry.value.with_staleness_banner(age)  # last-known-good + honest UI
        raise ProductUnavailableError()           # cold miss + DB down: honest 503
```

That `stale-while-down` budget separate from the freshness TTL is the crux of the implementation, and it maps directly to the broader consistency story: serving stale data is *exactly* the eventual-consistency trade that [the data-consistency-in-practice post and the CAP/PACELC theorem](/blog/software-development/database/cap-theorem-and-pacelc) make precise. When the database is down you are choosing availability over consistency — you serve a possibly-stale price rather than no price — and that is the right choice for a product *catalog* (a slightly stale price is fine to *display*; you re-validate it at checkout against the authoritative price service). It would be the *wrong* choice for the authoritative price at the moment of charging a card, where you must read fresh or fail closed. Static stability is not "always serve stale"; it is "decide, per read, whether last-known-good is acceptable, and engineer the read path so that the acceptable cases survive the backend being down." The full treatment of *when* stale is safe lives in [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice); here the point is the mechanism — a stale-while-down cache budget — that turns the theory into a read path that survives an outage.

## A decision matrix: per-dependency policy at a glance

Everything so far converges on one artifact a senior produces and a junior should learn to read: a per-dependency policy matrix. For every external dependency a service has, you write down its tier, its failure policy (fail closed, fail open, degrade, or cache-fallback), and — crucially — *what that policy costs* when it fires. The cost column is what keeps the matrix honest, because every policy buys availability with *some* currency: correctness, freshness, revenue, or user trust. Naming the cost forces you to admit the trade rather than pretend the fallback is free.

![A matrix table mapping six dependencies including payment, auth, inventory, reviews, recommendations, and the catalog database to a chosen failure policy and the cost that policy pays in correctness or revenue or user experience](/imgs/blogs/handling-partial-failures-and-graceful-degradation-3.webp)

The figure renders ShopFast's matrix, and the same content as a table you can copy and adapt for your own services:

| Dependency | Tier | Failure policy | What it costs you |
|---|---|---|---|
| Payment gateway | Critical (correctness) | Fail closed: honest "try again" | A lost or delayed sale |
| Auth / authz | Critical (security) | Fail closed: deny | Locked-out legitimate users |
| Inventory (at checkout) | Critical (correctness) | Fail closed or read-only | Cannot complete checkout |
| Price (on product page) | Critical (core UX) | Degrade: serve stale + banner | Possibly stale displayed price |
| Catalog DB (reads) | Critical (core UX) | Static stability: stale cache | Slightly stale catalog |
| Reviews | Optional | Fail open: cached, then hide | Less social proof |
| Recommendations | Optional | Fail open: default, then empty | Weaker upsell |
| Fraud score | Critical-ish | Fail open + async review | Small recoverable fraud risk |
| Delivery-estimate widget | Optional | Fail open: hide | No ETA shown |

The discipline this table enforces is that **no dependency is left without a policy**, which is precisely the failure that caused the Tuesday morning. When a new dependency is added to a service, the design review asks one question — "what is its row in the matrix?" — and a dependency cannot ship until it has one. The matrix is also a *contract* that the implementation must match: if the matrix says recommendations fails open to empty, then the code must have a fallback that returns empty and a test that proves it, which brings us to how you verify any of this before production.

#### Worked example: the availability math of degrading the optional two

Let me make the central claim quantitative, because "degrading helps" is a slogan until you put numbers on it. ShopFast's product page hard-depends, in the naive design, on four services in series — price, inventory, reviews, recommendations — each up 99.9 percent of the time. Composing them with no fallbacks means the page is up only when *all four* are up, and independent availabilities multiply: 0.999⁴ = 0.99600, so the page is available 99.6 percent of the time. That sounds close to 99.9, but the gap is enormous in human terms: 99.9 percent is 8.8 hours of downtime a year; 99.6 percent is about 35 hours a year — *four times worse* than its best single dependency, purely from stacking. Every dependency you hard-add to a synchronous path *multiplies* your downtime; this is why a chain of "individually excellent" services composes into a mediocre page.

![A before and after comparison showing four hard dependencies in series multiplying to ninety nine point six percent page availability against a design with two critical dependencies and two optional dependencies on fallbacks that recovers the page to roughly ninety nine point eight percent](/imgs/blogs/handling-partial-failures-and-graceful-degradation-7.webp)

Now degrade the two optional dependencies. Reviews and recommendations are given fallbacks that *never fail the page* — when they are down, the page renders without them. The page's availability now depends only on the two genuinely critical services, price and inventory: 0.999² = 0.99800, or 99.8 percent. We have cut the annual downtime from 35 hours to about 17.5 hours by changing *no infrastructure at all* — we did not make any service more reliable; we simply stopped letting two unimportant services vote on whether the page renders. And the experienced availability is better still, because for most of those 17.5 "down" hours the page is not fully down — a single critical service being down might let us serve a degraded read-only or stale-price page rather than a hard error, pushing the *user-perceived* availability higher than the strict "all critical up" number. The lesson in one line: **availability is not something you only buy with redundancy; you also buy it by refusing to depend on things you do not truly need.** Degrading the optional dependencies was free and bought back three-quarters of the downtime the naive design had thrown away.

## Detecting failure: where degradation meets the resilience patterns

Graceful degradation is the *response* to failure, but a response needs a *trigger*, and the trigger is the detection machinery the [resilience patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) covers in depth. I will not re-derive timeouts, retries, circuit breakers, and bulkheads here, but I do want to draw the precise seam between detection and response, because juniors conflate them and the conflation produces systems that detect beautifully and respond uselessly.

![A vertical stack of degradation tiers from a tier zero core browse and checkout layer down through live inventory cached reviews and static recommendation fallbacks to a tier four personalization layer that is shed first under load](/imgs/blogs/handling-partial-failures-and-graceful-degradation-6.webp)

A **timeout** is the detector that turns "I don't know if it will ever answer" — the deadly third outcome — into a definite "it failed, move on." A 200ms timeout on the recommendations call is what *lets* the fallback run; without it the call hangs and the fallback never executes, which is precisely the Tuesday-morning bug. **The single highest-leverage line of code in graceful degradation is the timeout**, because it converts unbounded uncertainty into a bounded failure you can respond to. A **circuit breaker** is the detector that notices a dependency is *consistently* failing and stops calling it at all — once it is open, every call to recommendations immediately runs the fallback without even attempting the network call, which saves the 200ms wait and, more importantly, stops you from hammering a struggling service. The breaker's *open* state and the fallback are partners: the breaker decides *when* to stop trying, and the fallback decides *what to do instead*. A **bulkhead** — an isolated thread pool or concurrency limit per dependency — is what stops a slow recommendations call from consuming the threads that price and inventory need, so that even before any fallback fires, recs cannot starve the critical path. The stack in the figure shows these as degradation *tiers*: under load the system sheds the cheapest tiers first (personalization, then static recs, then cached reviews) and protects the core browse-and-checkout tier the longest. The detection patterns are the mechanism that moves the system down the tiers; the degradation policy is what each tier *does*.

The clean mental separation, which is worth stating outright: **detection answers "is it failing?" and degradation answers "so what do we do?"** A circuit breaker with no fallback has answered the first question and shrugged at the second — it has only made the failure *fast* instead of *slow*. The reason this post is separate from the resilience-patterns post is that having the detectors without designed responses is a half-built system, and most teams build the detectors first (they are libraries you import) and forget the responses (they are decisions you have to make). The responses are the harder, more valuable half.

#### Worked example: cache-fallback hit rate keeps the catalog up through a DB outage

Here is the static-stability claim made quantitative. ShopFast's catalog database goes down for a full 20 minutes during a botched failover. The catalog service serves 6,000 product reads per second, and its read path is the stale-while-down cache from earlier with a 60-second freshness TTL and a one-hour stale budget. Question: what fraction of those reads survive the outage? It depends entirely on the cache hit rate, and the cache hit rate depends on the *working-set* concentration. ShopFast's traffic is heavily skewed — the top 10,000 products (a few percent of the catalog) account for about 90 percent of reads, and all of those hot products are comfortably within the cache's stale budget because they are read constantly. So during the 20-minute outage, roughly 90 percent of the 6,000 reads per second — 5,400 per second — are served stale-but-valid from the cache, with a freshness banner, and the user barely notices. The remaining 10 percent are cold or aged-out long-tail products, and those get an honest 503. Net result: a *total* database outage degrades the catalog to *90 percent availability* rather than zero, purely from a warm cache and a stale budget. Push the cache to hold the top 50,000 products and the survival rate climbs past 97 percent. The number to internalize: **with skewed read traffic, a stale-tolerant cache converts a total backend outage into a minor degradation, and the conversion rate is exactly your hit rate on the hot working set.** That is static stability paying off in a single number.

## A timeline: an optional outage handled the right way

It helps to walk a real incident end to end, the *good* version, so the patterns connect into a sequence rather than sitting as isolated techniques. This is the Tuesday morning re-run after ShopFast shipped the fixes.

![A six event timeline tracing a recommendations outage from a p99 latency spike through per call timeouts and a circuit breaker opening to an empty fallback being served and the page staying fully available until the service recovers and the breaker half opens](/imgs/blogs/handling-partial-failures-and-graceful-degradation-5.webp)

Trace the figure. At T+0 the recommendations service degrades again — same bad model, p99 jumps to four seconds. At T+5s the 200ms per-call timeout starts tripping on every recommendations call; each one fails fast at 200ms instead of hanging for four seconds, and the fallback returns an empty carousel. The critical difference from the original incident is already won here: the page threads are released at 200ms, not held for 4 seconds, so the thread pool never drains and the blast radius is contained to the recs section. At T+8s the circuit breaker sees that recommendations has failed the last N calls and *opens*, so subsequent calls do not even attempt the network — they return the fallback instantly, saving the 200ms and sparing the struggling recs service from a thundering herd of doomed requests. At T+8s onward the fallback is served on every request: empty carousel, page otherwise intact. At T+30s the dashboards confirm the page is still serving at 100 percent of its core function — price, inventory, reviews all green, the only signal being a spike in the `recs.fallback.empty` metric and an alert that fires *not* because the page is down (it is not) but because *recs has been on fallback for 30 seconds*, which is the right thing to page on. At T+2m the recs team rolls back the bad model, the breaker moves to half-open, lets a trial request through, sees it succeed, and closes — recommendations come back and the carousel repopulates, with no human ever having touched the page service. Zero failed page requests across the entire incident. *That* is what "designed for partial failure" looks like in a timeline: the failure happened, it was detected, the response was automatic, the blast radius was the one carousel it was allowed to be, and the alert fired on the degradation rather than on the (nonexistent) outage.

## The UX of degradation: honest beats slick

Degradation is not purely a backend concern, because the user *experiences* the degraded state and you get to choose how it feels. This is a genuinely underrated skill — most engineers stop at "the backend returned a fallback" and never design what the user sees, which produces degraded states that are confusing or, worse, dishonest. There are a few distinct presentation strategies and they trade off honesty against smoothness.

![A matrix mapping five degradation presentation choices including hide section disable with tooltip show stale with badge honest retry message and optimistic UI to how the user feels and the risk each one carries](/imgs/blogs/handling-partial-failures-and-graceful-degradation-9.webp)

**Hiding** a degraded section is the smoothest — if recommendations are empty, just do not render the carousel at all, and the user perceives nothing missing because they never knew it was supposed to be there. This is the right choice for purely additive optional sections, and its only risk is the silent-degradation problem: it is so invisible that *you* might not notice either, which is why the metric matters. **Disabling with a tooltip** — graying out the "Buy now" button with "Checkout is temporarily unavailable, try again in a moment" — is right when the user is mid-task and you must not let them proceed; you tell them honestly that the action is down rather than letting them click into a 500. **Showing stale data with a badge** ("prices updated a few minutes ago") is the static-stability presentation, and its risk is the sharpest in the matrix: if the user *acts* on stale data — adds to cart at a stale price — you must re-validate at the point of action, or you have let the degradation cause a correctness bug. **An honest retry message** — "something went wrong, please try again" with a working retry button — respects the user's intelligence and is almost always better than a fake spinner that never resolves. And **optimistic UI** — showing the action as done before the backend confirms it — is the slickest but the riskiest, because when the backend later fails you have to *roll back* a thing the user already saw succeed, which is jarring ("your order was placed… actually it wasn't").

The governing principle across all of these is **honesty over the appearance of perfection**. A user who is told "reviews are temporarily unavailable, the rest of the page is fine" trusts you more than a user who sees a blank space and wonders if the site is broken, and *far* more than a user who sees a fake "loading reviews…" spinner that never finishes. The single worst degradation UX is the infinite spinner — it converts a clean degraded state into a perception of total brokenness, because the user cannot tell the difference between "this section is degraded" and "the whole site is hung." Bound every spinner with a timeout that resolves to *some* honest state. The second worst is the silent wrong answer — showing stale data as if it were live and letting the user act on it — because it is the only degradation that can corrupt the user's *own* decisions. Pick the presentation per section, write it into the matrix alongside the backend policy, and the degraded experience becomes something you designed rather than something the user discovers.

## Finding partial-failure bugs before production

Everything above is a *design*, and a design that is never tested under failure is a hypothesis. The brutal truth of partial failure is that the failure paths are the *least*-exercised code in your entire system — they run, by definition, only when something is broken, which is rarely, so they rot. The fallback you wrote eighteen months ago calls a cache method that was renamed twelve months ago; it has thrown a `NoMethodError` every time it fired since, but it has only fired during outages, when everyone was busy looking elsewhere, so nobody noticed that your "graceful degradation" actually crashes. **An untested fallback is not a fallback; it is a second bug waiting for the first bug to wake it up.**

The discipline that fixes this is *deliberately injecting* the failures in a controlled setting and verifying the system degrades as designed — chaos engineering and fault injection, which the [testing microservices from unit to chaos post](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos) covers as a practice. The cheapest version is a test that mocks the dependency to fail and asserts the fallback fires:

```python
# A fault-injection test: prove recommendations degrades to empty, not to a 500.
async def test_product_page_survives_recs_outage(client, fake_recs):
    fake_recs.fail_with(TimeoutError, every_call=True)   # recs is "down"
    resp = await client.get("/product/abc-123")
    assert resp.status == 200                              # page must NOT error
    assert resp.json["price"] is not None                  # critical data present
    assert resp.json["recommendations"] == []              # degraded to empty
    assert "recommendations" in resp.json["degraded"]      # honestly flagged
    assert metrics.value("recs.fallback.empty") >= 1       # fallback actually ran

async def test_checkout_fails_closed_when_payment_down(client, fake_payment):
    fake_payment.fail_with(ServiceError, every_call=True)  # payment is "down"
    resp = await client.post("/checkout", json=valid_order)
    assert resp.status == 503                              # fail CLOSED, honestly
    assert resp.json["message"] == "Payment is temporarily unavailable, please retry"
    assert await orders.count() == 0                       # NO half-order written
```

Those two tests encode the two halves of the policy: the optional dependency *must* degrade to a valid response (the first test fails if recs ever 500s the page), and the critical dependency *must* fail closed cleanly without leaving a half-written order (the second test fails if a payment outage corrupts state). Run them in CI and the fallbacks cannot silently rot, because a renamed cache method breaks the first test the day it is renamed, not eighteen months later in production.

The production-grade version goes further into *live* fault injection — running a "GameDay" where you deliberately kill the recommendations service in a staging or even a controlled production environment and watch the product page survive in real time, the way Netflix's Chaos Monkey randomly terminates instances to *force* the system to be resilient continuously rather than hoping it is. The forward link to [testing microservices from unit to chaos](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos) is where this practice lives in detail; the point here is the principle: **the only way to know your degradation works is to break the dependency on purpose and watch.** A team that has never deliberately killed its own optional services has, in practice, never tested its fallbacks, and a fallback that has never been tested is a fallback that does not work.

## Optimization: tuning degradation for production

Once the degradation paths exist and are tested, there is real optimization to be done, and it has numbers attached. The first knob is **cache TTLs for static stability**, and it is a direct trade-off between freshness and survivability. A short freshness TTL (say 30 seconds) keeps displayed data current but means a backend outage longer than 30 seconds starts producing misses for anything not re-requested; a long *stale budget* (say one hour, independent of the freshness TTL) extends how long you can survive a total backend outage at the cost of serving older data during one. The right setting is per-data-type: a product *title* can have an hour-long stale budget because titles change rarely, while a *price* shown for display might tolerate a few minutes of staleness but must be re-validated at checkout. Concretely, raising ShopFast's catalog stale budget from 5 minutes to 60 minutes converts a 20-minute database outage from "90 percent of reads survive for the first 5 minutes, then collapse" into "90 percent survive the entire 20 minutes" — a free reliability win paid for only in the rare case where a price was stale for up to an hour, which the checkout re-validation catches anyway.

The second knob is **prefetching and cache warming**, which raises the hit rate that, as the worked example showed, *is* your survival rate during an outage. If a cold cache survives a backend outage at 60 percent and a warm one at 90 percent, then proactively warming the cache with the hot working set — running a background job that re-reads the top 50,000 products every few minutes regardless of user traffic — directly buys outage survivability, not just steady-state latency. The cost is the background read load on the database during *normal* operation, which is exactly when you can afford it. This connects to the broader caching story that the cross-service [caching strategies post] (a later sibling) develops, but the resilience framing is specific: a cache you keep warm is a cache that can carry you through a backend outage, so cache-warming is a *reliability* investment, not only a *latency* one.

The third knob is **degradation tiers**, formalizing the stack figure into an ordered shed list. Rank every optional feature by (value to the user) ÷ (cost to serve), and shed the lowest-ratio features first under load. Personalized recommendations are expensive (an ML scoring call per request) and marginally valuable, so they shed first; cached reviews are cheap and moderately valuable, so they shed last among the optionals; the core browse-and-checkout path never sheds. Wiring this to an *automatic* trigger — when p99 latency on the core path exceeds, say, 500ms or when the request queue depth crosses a threshold, automatically flip the tier-4 and tier-3 flags — turns degradation into a self-protecting reflex that the [load-shedding post](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) develops fully. The measurable win is concrete: ShopFast found that shedding personalization under load cut the core path's p99 from 1,400ms back to 480ms during a spike, because the expensive scoring calls were no longer competing for the same thread pool and database connections as checkout. **Degradation is not only a failure response; under load it is a performance optimization, because the cheapest way to make the core path fast is to stop doing the expensive optional work.**

The fourth, quieter optimization is **alerting on degradation rate rather than on outages**, which is where degradation and observability meet. Once your system degrades gracefully, your *outage* alerts go quiet — the page is up, so the page-availability SLO is met — and the danger is that a dependency can now be down for *days* without anyone noticing, because the fallback hides it perfectly. The fix is to alert on the *fallback metrics*: page `recs.fallback.empty > 5%` of requests for more than five minutes, even though the page itself is healthy. This is precisely the golden-signals discipline that [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) develops — you measure and alert on the degradation rate as a first-class signal, because in a well-degraded system the degradation rate is the *only* early signal that a dependency is sick. A team that only alerts on full outages will, paradoxically, get *worse* at noticing problems the better its degradation gets.

## Stress-testing the design against three concrete failures

A design is only as good as the failures it survives, so let me run the ShopFast design against the three questions a review should always ask, out loud, with the numbers.

*"Recommendations is down — does the product page survive?"* Yes, and here is the chain. The aggregator calls recs with a 200ms timeout, so a slow or dead recs costs at most 200ms (and zero once the breaker opens). `Promise.allSettled` isolates the recs rejection from the other three calls. The recs accessor's fallback returns an empty list and never raises. The page renders price, inventory, and reviews; the carousel is absent or shows non-personalized top sellers. The `recs.fallback.empty` metric spikes, an alert fires on the *degradation rate*, and the page-availability SLO is untouched. The blast radius is exactly one carousel, which is what the classification promised. This is the failure that, undesigned, caused the Tuesday morning; designed, it is a non-event.

*"Auth is down — do we fail open or closed?"* Closed, deliberately. The `can_access` function returns `False` on any auth-service error, so during the outage users see "login is temporarily unavailable, please retry" rather than being silently let into accounts that are not theirs. The cost is real — legitimate users are locked out for the duration — but it is the *cheap* wrong: an annoyed user who retries in two minutes versus a breach that exposes every account. The senior refinement is to scope the blast radius: a *cached* valid session (a token that already authenticated minutes ago and is still within its short TTL) can keep working through the auth outage, because we are not re-checking an unknown user, we are honoring a check that *already succeeded* — that is static stability applied to auth, failing closed for *new* logins while serving last-known-good for *established* sessions.

*"The cache is cold during the outage — now what?"* This is the honest failure mode of static stability, and the design admits it rather than pretending. If the catalog database goes down *and* the cache is cold for a given product (a long-tail item nobody has read recently, or right after a deploy flushed the cache), the read path has nothing last-known-good to serve and returns an honest 503 for that product. The mitigation is the cache-warming optimization — keep the hot working set warm so the cold set is small — but the *design principle* is that static stability degrades the cold set honestly instead of hanging or guessing. The worst possible answer to a cold miss during a backend outage is to *block* waiting for a database that will not answer (re-creating the Tuesday morning at the data layer); the right answer is to fail that one read fast and cleanly, which is why the read path has a 200ms database timeout even on the refresh attempt. Cold-cache-plus-dead-backend is the one case the design *cannot* fully save, and a senior says so plainly rather than claiming a fallback that does not exist.

### Degradation in asynchronous and event-driven flows

Everything so far has been about *synchronous* request paths, but a large fraction of a microservices system is asynchronous — events flowing over a broker, consumers projecting them into read models — and degradation there looks different. The good news is that async communication is *already* a form of graceful degradation: when a downstream consumer is slow or down, the events queue in the broker rather than failing the producer, so the producer keeps working and the consumer catches up later. This is why moving an optional dependency from a synchronous call to an asynchronous event is itself a resilience technique — the recommendations service, instead of being called synchronously during the page render, can consume "product viewed" events and pre-compute recommendations into a cache the page reads locally, which means the page never waits on recs at all and a recs outage simply means the pre-computed cache goes stale. The trade is the eventual-consistency window the [data-consistency post](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) details, but for an optional, enriching feature, "slightly stale recommendations" is exactly the kind of staleness you are happy to accept.

The degradation question in async flows shifts from "what do I return when the call fails" to "what do I do when the *queue backs up*," which is the [backpressure and flow-control problem the message-queue post covers](/blog/software-development/message-queue/backpressure-and-flow-control). When a consumer cannot keep pace, the broker's queue grows, the projection lag climbs, and your "fresh" read models become stale — a *silent* degradation that the synchronous fallback metrics never see, because no synchronous call failed. The defense is the same in spirit: a per-projection lag metric with a budget and an alert, so that "the inventory read model is now 45 seconds behind" pages someone before a user oversells against a stale count. The principle generalizes across both worlds: **degradation is whatever keeps the core invariant safe while a part of the system is behind or absent, and in the async world the part that is behind is a queue rather than a call.**

## Case studies

These are real, and each isolates one lesson from the post.

**Netflix and the Hystrix fallback philosophy.** Netflix built and open-sourced Hystrix, a library that wrapped every cross-service call in a circuit breaker with a mandatory *fallback* — the API explicitly required you to define what the call returns when it fails, making the fallback a first-class, non-optional part of every dependency call rather than an afterthought. Netflix's resulting principle is the one this whole post argues: the homepage should *always* render something, even if many of its personalized rows fall back to non-personalized "popular on Netflix" defaults, because a homepage that renders generic rows is infinitely better than a homepage that errors. Hystrix itself is now in maintenance mode (the ecosystem moved toward Resilience4j and service-mesh-level resilience), but the design lesson outlived the library: **make the fallback a required parameter of the call, not an optional try/catch you might forget**, and the Tuesday-morning class of bug becomes structurally impossible because there is nowhere to forget the fallback.

**Amazon and static stability.** Amazon's engineering writing (notably the Builders' Library essays) made "static stability" a named, taught principle: design the data plane so it keeps working on last-known-good state when the control plane is unavailable, and pre-provision capacity so that surviving a failure requires *no* successful control-plane action during the emergency. The canonical example is an EC2-style fleet that survives an Availability Zone loss not by frantically launching new instances (which needs a control plane that may itself be degraded) but by running enough spare capacity in the surviving zones to absorb the load with no action at all. The lesson for an ordinary team: **the most reliable reaction to a failure is no reaction** — pay for redundancy and last-known-good caching up front so that survival does not depend on a successful API call in the middle of the worst moment, because that is exactly the moment your APIs are least likely to work.

**The "one optional service took down the homepage" post-mortem (a recurring real pattern).** This exact failure has been publicly documented across many companies under different names, and the shape is always identical: a non-critical feature — a feature-flag service, an experiment-assignment service, an avatar/profile-image service, a recommendations widget — is called *synchronously* with no timeout and no fallback from a high-traffic page, the optional service slows down or hits its own dependency's limit, the page's threads block waiting on it, the thread pool exhausts, and a page that was 99 percent fine returns hard errors to everyone. The recurring root cause is never the optional service's outage (optional services *should* be allowed to be down); it is that nobody had *classified* it as optional and given it a timeout and a fallback, so it was silently treated as critical. The recurring lesson, and the one I most want a junior to carry out of this post: **the dangerous dependency is not the one you know is fragile; it is the one you never thought about, because the ones you never thought about are the ones with no fallback.** Go find them before they find you, with a chaos test.

**Monzo's 1,500+ services and platform-level resilience.** Monzo runs an unusually large number of fine-grained services, and a system that fine-grained makes the partial-failure arithmetic from the opening *brutal* — with that many services, something is always down. Monzo's answer was to push resilience defaults down into the platform (standard timeouts, standard retry-with-backoff, standard circuit-breaking, and a service mesh enforcing them) so that no individual team had to remember to wrap every call, and a new service inherited sane partial-failure behavior by default. The lesson: at scale, **resilience cannot be a thing each team remembers to do per call; it has to be a default the platform enforces**, because the arithmetic guarantees that the one call a team forgot to protect is the one that will be hit by a partial failure this week.

## When to reach for this (and when not to)

Graceful degradation is not free, and a senior names the cost. Every fallback is *more code* — a second code path that must be written, tested, and maintained, and which (as the rot problem showed) is the least-exercised code you own. Every degradation tier is a *product decision* — somebody has to decide that a generic carousel is acceptable, which means involving product and design, not just engineering. And degradation *hides* problems, which is a benefit during an incident but a liability for detection, requiring the degradation-rate alerting that is itself more work. So when is it worth it?

**Always design degradation for any optional dependency on a high-traffic synchronous path.** This is non-negotiable; the cost of the fallback is trivial next to the cost of the Tuesday morning, and the classification-plus-fallback is the cheapest reliability investment you can make. **Always make a deliberate fail-open versus fail-closed decision for any critical dependency** — even if the decision is "fail closed with an honest message," it must be a decision, logged and reviewed, not an accidental exception handler. **Build static-stability caching for read-heavy paths** where a stale answer is acceptable and the backend can plausibly be down — product catalogs, configuration, user profiles. These are high-leverage and you should reach for them by default.

When is it *over*-engineering? For an internal admin tool used by ten people, an elaborate degradation tier system is wasted effort — let it 500 and show a clear error; the ten users will retry. For a hard correctness boundary — moving money, decrementing inventory at the point of sale — do *not* reach for "degrade to a guess"; the right behavior is to fail closed cleanly, and a clever fallback that lets a charge proceed without confirming it is a correctness bug dressed as resilience. And for a small system with two or three services where the partial-failure arithmetic is still benign, you can run leaner — the all-healthy state is common enough at that scale that elaborate degradation is premature. The rule of thumb: **degrade the optional aggressively, fail the critical safely, and do not invent a fallback for a thing whose only correct failure mode is to stop.** The judgment of *which dependencies even need a policy* is the senior skill; the matrix is where you record the judgment.

## Key takeaways

- **Partial failure is the steady state.** With a hundred 99.9-percent services, something is down roughly one moment in ten — design for "mostly working, something broken now," because the all-healthy state is rare.
- **The network gives you three outcomes, not two:** succeeded, failed, and *timed-out-unknown*. The third is what makes partial failure a design problem and why retries must be made idempotent.
- **Classify every dependency as critical or optional before any incident.** The absence of a classification is itself a decision, and it is always the wrong one because it silently makes every dependency critical.
- **Degrade the optional, never let it vote on the core.** A recommendations outage must not be able to take down checkout; the fallback (cache, default, empty, partial response) must never raise.
- **Fail in the direction where being wrong is cheap.** Auth fails closed (a wrong allow is a breach); recommendations fail open (a wrong deny is an outage). Make every fail-open on a security or correctness boundary a deliberate, logged, reviewed choice.
- **Static stability means surviving on last-known-good.** Decouple the data plane from the control plane, pre-provision redundancy, and serve stale-while-down so survival needs no successful API call mid-emergency.
- **Detection and response are different halves.** Timeouts and circuit breakers *detect*; degradation is the *response*. A breaker with no fallback only makes failure fast, not survivable.
- **An untested fallback is a second bug waiting.** Failure paths are your least-exercised code; inject the failure in CI and in GameDays, or your "graceful degradation" silently crashes the first time it fires.
- **Alert on the degradation rate, not just on outages.** The better your degradation, the quieter your outage alerts — and the longer a dependency can be silently down. Make the fallback rate a first-class signal.
- **Availability is also bought by refusing to depend.** Degrading two optional dependencies lifted the page from 99.6 to 99.8 percent with zero infrastructure change — the cheapest reliability you will ever buy.

## Further reading

- [Resilience patterns: timeouts, retries, circuit breakers, and bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — the detection machinery this post responds to.
- [Idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) — making the retry of a timed-out-unknown call safe.
- [Inter-service communication fundamentals and the fallacies of distributed computing](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) — why the network gives you three outcomes.
- [The API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) — where partial-response aggregation and the critical/optional contract live.
- [Data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) — when serving stale data is safe.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — why static stability is fundamentally a consistency-versus-availability trade.
- [Rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) — automating degradation tiers under load.
- [Testing microservices from unit to chaos](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos) — fault injection and GameDays to prove fallbacks work.
- [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) — alerting on the degradation rate as a first-class signal.
- Sam Newman, *Building Microservices* (2nd ed.) — chapters on resilience and the failure of distributed calls.
- Chris Richardson, *Microservices Patterns* — the reliability and observability patterns in practice.
- Amazon Builders' Library, "Static stability using Availability Zones" — the canonical static-stability essay.

---
title: "Graceful Degradation and Fallbacks: Serve a Worse Answer, Not an Error Page"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The difference between a partial outage and a total one is whether you designed your system to degrade instead of collapse. Map every feature's criticality, build fallback chains, serve stale, and measure the degraded mode you forgot you were running."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "graceful-degradation",
    "fallbacks",
    "resilience",
    "fault-tolerance",
    "serve-stale",
    "circuit-breakers",
    "feature-flags",
    "availability",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/graceful-degradation-and-fallbacks-1.png"
---

At 14:07 on a Tuesday, the reviews service fell over. A bad deploy, a connection-pool exhaustion, the usual. By 14:09 the on-call for the storefront team was paged — not because reviews were down, but because the entire product-detail page was returning HTTP 500. Every product. Every customer. The "Add to Cart" button, the price, the photos, the stock count — all of it gone, replaced by a generic error page, because the page template made a synchronous call to the reviews service and nobody had ever asked what happens when that call doesn't come back. A feature that maybe 4% of visitors scroll down to read had just taken down 100% of the storefront. Revenue went to zero for eleven minutes over a module almost nobody uses.

That outage was not a reviews-service failure. The reviews service failed all the time; that was expected. The outage was a *design* failure. Somewhere in the request path, a non-critical dependency had been wired into the critical path as a hard dependency, and when it died, it took everything down with it. The system was built to collapse. The fix wasn't to make reviews more reliable — that's a losing game, you will never make every dependency perfect. The fix was to make the *page* survive the reviews service being down. Render the page, skip the reviews module, log it, page someone, and keep selling. That is graceful degradation, and the gap between a system that degrades and one that collapses is one of the highest-leverage reliability investments you will ever make.

This is the central idea of this post, and the figure below is the whole thesis in one picture: the *same* reviews outage produces a total page failure on the left and a barely-noticeable partial page on the right, and the only difference between them is whether someone designed the failure behavior on purpose.

![Two-column comparison showing a reviews outage causing a full page 500 on the left versus a partial page that omits the reviews module on the right](/imgs/blogs/graceful-degradation-and-fallbacks-1.png)

This post sits in the *respond and engineer* part of the series loop — define reliability, measure it, spend the error budget, reduce toil, respond to incidents, learn from them, and engineer the fix. Graceful degradation is one of the most direct ways to *spend less budget*: an outage you absorb without user-facing errors barely touches your error budget at all. If you have not yet read the intro map, [reliability is a feature — the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) frames why we treat reliability as something we engineer and budget rather than wish for. By the end of this post you will be able to: classify every feature in your product by criticality tier; choose the right fallback strategy for each kind of read; build a fallback chain that tries fresh, then cache, then a default before it ever errors; spot the synchronous non-critical call that is silently a hard dependency; and — the part everyone forgets — measure when you are running degraded so a fallback never quietly masks a six-hour outage.

Let me define the words first, because the rest of the post leans on them. A **dependency** is any external thing your code calls to do its job: a database, another microservice, a third-party API, a cache. A **critical path** is the sequence of calls a request *must* complete to deliver the user's core value — for a store, that is browse, add to cart, pay. **Graceful degradation** means that when a dependency fails, the system serves a worse-but-still-useful response instead of an error. **Collapse** is the opposite: one failure cascades into a total outage. A **fallback** is the worse-but-useful response you serve — a cached value, a default, an omitted section. And **stale** data is data that is older than your normal freshness target but still close enough to be useful. Hold those, and we can go deep.

## 1. The degrade-don't-collapse principle

Here is the principle stated as bluntly as I can: **a non-critical dependency failing should never take down a critical path.** That is the whole game. Everything else in this post is mechanics for upholding that one rule.

Why does it matter so much? Because of how availability composes. If your critical path makes a synchronous call to a dependency, and that dependency is up 99.9% of the time, then — assuming independent failures — your critical path can be *at most* 99.9% available, because every time the dependency is down, your path is down too. Chain three such dependencies in series and your ceiling drops to $0.999^3 \approx 99.7\%$. Add a fourth at 99.5% and you are under 99.2%. Serial hard dependencies multiply their *failure* probabilities into your service, and they do it whether the dependency is critical or not. The math does not care that reviews are "just a nice-to-have"; if reviews are a hard dependency on the product page, reviews' downtime *is* the product page's downtime.

The way out is to change the relationship. A **soft dependency** is one whose failure your code is designed to tolerate: when it fails, you fall back, and your availability is no longer bounded by it. Turning a hard dependency into a soft one is the single most valuable thing this post teaches, because it removes that dependency from your availability product entirely. The reviews service can be down for an hour and your product-page SLO is untouched, because a down reviews service produces a *fully successful* product-page response that simply lacks reviews.

It helps to think about degradation as a *spectrum* of responses rather than a binary up/down. At the top of the spectrum is the full experience: everything fresh, personalized, complete. One step down is *slightly worse*: a cached value instead of a fresh one, a generic recommendation instead of a personalized one — the user might not even notice. Further down is *visibly reduced*: a module is missing, search returns basic results, a feature is greyed out — the user notices but the product still works. At the very bottom is *collapse*: an error page, a 500, nothing. The art of graceful degradation is engineering your system so that a dependency failure moves you *one or two steps down the spectrum* instead of all the way to the bottom. A collapse is a degradation that nobody designed. Every outage is, in this framing, a failure to have built the next rung down.

There is a deeper reason this matters beyond the availability arithmetic, and it has to do with *correlation*. Real dependency failures are not independent the way the multiplication assumes — they cluster. A bad deploy, a network partition, a cloud-provider zone event, a database failover all tend to knock out several things at once. When failures correlate, the naive availability product *understates* your risk, because two dependencies you assumed were independent fail together. Degradation is your hedge against exactly this: when you cannot predict *which* dependency will fail or *when* they will fail together, designing each one to be soft means that no matter which combination goes down, the critical path survives as long as the truly-critical dependencies hold. You are buying insurance against a failure distribution you cannot fully model. That is why "make each non-critical dependency soft" is more robust than "make each dependency more reliable" — the former protects you against the correlated, surprising failures; the latter only shifts the per-component numbers and leaves the coupling intact.

There is also a *psychological* reason degradation is undervalued, and it is worth naming because it explains why so many systems are built to collapse. Degradation behavior is invisible during normal operation. When everything is healthy, the fallback code never runs, the degraded-mode SLI reads zero, and the partial-response logic is dead weight that slows nobody down and helps nobody. It only earns its keep during an outage — a rare event that, by definition, you hope never happens. So it is perpetually the thing that gets deprioritized: "we'll add the fallback later, recs is reliable enough for now." And then later never comes, until the 14:07 page. The discipline of graceful degradation is, in large part, the discipline of investing in code paths that pay off only on your worst day — which is precisely the SRE mindset, where you spend calm time today to buy calm during the next incident.

#### Worked example: what one hard dependency costs

Suppose your product page calls four services synchronously: catalog (99.95%), pricing (99.95%), inventory (99.9%), and reviews (99.5% — it is flaky, it is third-party-ish, it is nobody's priority). If all four are hard dependencies, the page's availability ceiling is:

$$0.9995 \times 0.9995 \times 0.999 \times 0.995 \approx 0.9930$$

That is 99.30%, which is about **6.1 hours of downtime per month**. Now make reviews a *soft* dependency — when it fails, you render the page without reviews. Reviews drops out of the product entirely:

$$0.9995 \times 0.9995 \times 0.999 \approx 0.9980$$

That is 99.80%, about **1.4 hours per month**. You just bought back nearly 4.7 hours of monthly downtime — a 77% reduction in user-facing failure minutes — by changing one dependency's *relationship*, without making reviews one bit more reliable. That is the leverage. It is why degradation belongs in your reliability toolbox right next to SLOs and on-call. If you want the SLO arithmetic that turns "99.80%" into "1.4 hours" rigorously, the sibling post [setting SLOs that mean something](/blog/software-development/site-reliability-engineering/setting-slos-that-mean-something) has the nines-to-downtime table; the point here is that degradation is a lever you pull *on* that math.

The mirror principle from the *design* side of the house is failure-domain mapping and blast-radius control, which the sibling post [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) covers — degradation is the runtime behavior you choose once you have mapped which dependencies are which. At the architecture layer, the system-design write-up on [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) treats the same idea from the design-time angle; this post is the operations layer — how you actually run it, measure it, and keep the fallbacks working.

## 2. Map every feature's criticality — the tiers

You cannot design degradation behavior you have not classified. The first artifact every team needs is a **feature-criticality map**: a table that names each feature, assigns it a tier, and states what it does when its dependency fails. Without it, "is this dependency critical?" gets answered ad hoc, in code review, by whoever happens to look — which is how a 4%-of-traffic reviews module ends up a hard dependency.

The tiers I use are simple enough to keep in your head:

- **P0 — must never degrade to an error.** The core flow: authentication, checkout, payment, the action that *is* the product. If a P0 path returns an error, you are having an outage. P0 paths get the most engineering: redundancy, retries on idempotent steps, and a *fail-fast* behavior so that if they truly cannot serve, the user gets a clear, fast error and a retry rather than a 30-second hang.
- **P1 — degrades to a simpler version.** Important but not core: search, filtering, the recommendation-ranked listing. When the fancy path fails, P1 falls back to a basic version — search degrades from a smart ranked result to a plain keyword match; the personalized listing degrades to a default sort. The user notices, but the feature still works.
- **P2 — may vanish silently.** Genuinely optional: the recommendation widget, the "people also bought" carousel, the live-visitor counter, the embedded social feed, most analytics. When a P2 dependency fails, the right behavior is to hide the module and serve the page without it, ideally with the user never knowing.

The figure below is the shape of that map: feature on the rows, and for each, its criticality, its on-failure behavior, and how much of the error budget it is allowed to consume.

![A three-row matrix mapping checkout, search, and recommendations to their criticality tier, dependency-failure behavior, and error-budget impact](/imgs/blogs/graceful-degradation-and-fallbacks-2.png)

The governing rule that falls out of this map is what I call the **core-flow survival rule**: *the core flow must survive any single non-core dependency failing.* Walk your critical path and, for each dependency it touches, ask "if this is down, does the core flow still complete?" If the answer is no for any P1 or P2 dependency, you have found a hard dependency that should be soft. This is a five-minute exercise that has saved me more outages than any monitoring I have ever set up.

How do you actually assign the tiers without endless debate? The most reliable forcing question is not "how important is this feature?" — every team thinks every feature is important — but "**what happens to the user, and to revenue, if this is gone for an hour?**" That reframes the conversation around impact instead of pride. If the answer is "the user cannot complete the thing they came to do" or "we lose money directly," it is P0. If the answer is "the user has a worse time but can still accomplish their goal," it is P1. If the answer is "most users would not notice," it is P2. The hour-long-outage framing is deliberately concrete: it stops people from rating their pet feature P0 because it *feels* central, and it surfaces the genuinely critical paths, which are usually fewer than people expect. A typical e-commerce product has maybe three or four true P0 flows — browse, add-to-cart, checkout, login — and a long tail of P1 and P2 features that everyone *assumed* were load-bearing but are not.

The second discipline is to make the map **a living artifact, not a one-time spreadsheet**. Dependencies change. A feature that was P2 last year is now P1 because the business built a workflow on top of it. A new third-party integration got wired into checkout and nobody re-classified the checkout path. The criticality map should be reviewed every quarter, and — more importantly — it should be checked *automatically* against reality. A lightweight but powerful practice is to annotate each downstream call in code with its tier (a decorator, a header, a config lookup) and then have a linter or a periodic job that asserts "no call annotated P2 is on a synchronous P0 path." That turns the core-flow survival rule from a manual review into a guardrail that fails the build when someone wires a non-critical dependency into a critical path. It is the same move as the degradation-policy-as-data idea in section 4: the more your criticality decisions live in checkable artifacts rather than people's heads, the less likely the 14:07 page becomes.

A third subtlety: criticality is not a property of a *service*, it is a property of a *call site*. The same recommendations service might be P2 when it powers the "you might also like" carousel on a product page and P1 when it powers the entire personalized homepage feed. The same pricing service is P0 on the checkout page (you must charge the right amount) and arguably P1 on a marketing landing page (an approximate "from \$29" is fine). So the map's unit is the *(feature, dependency)* edge, not the dependency. This is why the degradation policy in section 4 is keyed by feature, not by service — the right failure behavior depends on *where* and *why* you are calling, not just *what* you are calling.

Here is a concrete criticality map for a storefront, the kind you would keep in your repo as a markdown table and review every quarter:

| Feature | Tier | Dependency | On dependency failure | Budget allowed |
|---|---|---|---|---|
| Login / session | P0 | Auth service, session store | Fail fast, clear retry, page on-call | Full SLO budget |
| Add to cart / checkout | P0 | Cart, payment, inventory | Retry idempotent steps, fail fast if truly down | Full SLO budget |
| Product price | P0 | Pricing service | Serve last-known-good from cache, never blank | Full SLO budget |
| Search | P1 | Search cluster | Degrade to basic keyword search, banner | Partial |
| Product reviews | P2 | Reviews service | Omit module, render rest of page | Near-zero |
| Recommendations | P2 | Recs ML service | Fallback chain → generic popular → hide | Near-zero |
| Live visitor count | P2 | Analytics stream | Hide widget silently | Near-zero |

Notice that even within P0 the behavior differs. Product price is P0 — a product page with no price is broken — but pricing rarely *needs* to be real-time fresh, so its P0 behavior is "serve the last-known-good price from cache, never show a blank price." Checkout is P0 and *cannot* serve a stale answer (you cannot fake a payment authorization), so its behavior is "fail fast with a retry." Same tier, opposite fallback, because the right behavior depends on whether the read tolerates staleness, which is the next thing to reason about.

## 3. The fallback strategies, concretely

There are five fallback strategies I reach for, and almost every real degradation is one of these or a composition of them. Let me take each in turn, with the config to back it.

### Serve stale: the last-known-good value

For most reads, **stale data beats no data.** A product price from four minutes ago, a config value from last hour, a leaderboard from this morning — these are all far more useful than an error. The pattern is **stale-while-revalidate**: serve the cached value immediately, and kick off an asynchronous refresh in the background. When the fresh source is healthy, the cache stays warm and nobody notices it is a cache. When the fresh source is *down*, the cache keeps serving the last value it has, and the user keeps getting a useful answer.

The trick that makes this work during an outage is the **deliberately-long fallback TTL.** A cache has two notions of age. The *fresh* TTL is how long a value is considered current — say 60 seconds; past that you revalidate. The *fallback* (or stale) TTL is how long you are willing to keep serving a value when revalidation *fails* — and you set that much longer, maybe an hour, deliberately, so that a one-hour source outage produces zero user-facing errors. The figure below contrasts the two postures during a six-minute source outage: strict freshness errors out, stale-while-revalidate keeps serving.

![Two-column comparison of strict-freshness caching erroring out during a source outage versus stale-while-revalidate continuing to serve the last value](/imgs/blogs/graceful-degradation-and-fallbacks-4.png)

Here is a "serve stale" cache config in the shape you would put in front of a service — this happens to be an Nginx-style proxy cache, but the same three directives exist in every serious cache (Varnish `grace`, a CDN's stale-while-revalidate header, a Caffeine `refreshAfterWrite` plus a long `expireAfterWrite`):

```nginx
proxy_cache_path /var/cache/recs keys_zone=recs:50m
                 inactive=2h max_size=2g;

location /api/recommendations {
    proxy_pass http://recs_backend;
    proxy_cache recs;

    # Fresh for 60s. Past that, revalidate in the background.
    proxy_cache_valid 200 60s;

    # The degradation contract: keep serving the cached value
    # if the backend is down, has errored, or times out --
    # for up to 1 hour past expiry. A 1h backend outage
    # produces zero user-facing errors on this route.
    proxy_cache_use_stale error timeout updating
                          http_500 http_502 http_503 http_504;
    proxy_cache_background_update on;
    proxy_cache_lock on;

    # Tag responses so we can MEASURE degraded serving (see section 8).
    add_header X-Cache-Status $upstream_cache_status;
}
```

The `proxy_cache_use_stale` line is the entire point: `error timeout updating http_500 ...` tells the cache "if the backend errors, times out, is being updated, or returns a 5xx, serve the stale copy instead of propagating the failure." The `X-Cache-Status` header lets your observability count how often you are serving `STALE` — which, as we will see, is how you avoid the silent-degradation trap.

The mental shift that makes serve-stale work is to stop treating a cache as a *performance* optimization and start treating it as an *availability* mechanism. Most teams configure a cache to make things faster — they set a TTL that balances freshness against load, and they think of a cache miss as "a bit slower this time." But a cache configured for availability is configured around the question "what do I serve when the source is down?" and the answer it gives is "the last good value I have, for a deliberately long time." Those are different configurations. A performance cache might expire after 60 seconds and then go straight to the source on every miss; if the source is down, every request errors. An availability cache expires *for freshness* after 60 seconds but holds the value *for fallback* for an hour, so a source outage is invisible. The same cache, two TTLs, completely different failure behavior. Whenever you put a cache in front of a read, ask yourself which one you built — because if you only thought about the fast path, you have a performance cache that will betray you during the outage.

#### Worked example: the deliberately-long fallback TTL

Concretely: your homepage shows a "deal of the day" pulled from a promotions service. The promotion changes at most once an hour, so a 5-minute fresh TTL is plenty — that bounds how stale the displayed deal can be under normal conditions. Now set the *fallback* TTL to 6 hours. Walk the numbers on a promotions-service outage:

- Source healthy: every 5 minutes the cache revalidates in the background; users always see a deal at most 5 minutes old. Source load is one request per cache key per 5 minutes regardless of traffic.
- Source goes down at 10:00: the cache keeps serving the 09:57 deal. Users see a deal that is now up to a few minutes stale, then a few minutes more. No errors. No page.
- Source still down at 13:00: the deal is now ~3 hours stale, but it is *a valid deal* — promotions change slowly, the 09:57 deal is almost certainly still live. Users are unaffected. The degraded-mode SLI has been firing a low-urgency ticket since 10:05, so someone is fixing the source on business hours.
- Source comes back at 13:30: the next background revalidation succeeds, the cache warms, freshness returns to 5 minutes. Total user-facing errors over a 3.5-hour source outage: **zero.**

The fallback TTL of 6 hours was a *deliberate availability decision*: "I am willing to show a stale deal for up to 6 hours rather than show an error, because a slightly-old deal is harmless and an error is not." Had the fallback TTL equalled the fresh TTL of 5 minutes (the performance-cache default), the same outage would have produced 3.5 hours of errors on the homepage. One config number, the difference between an outage and a non-event.

### Cached fallback: serve from cache on dependency failure

A close cousin: rather than a transparent proxy cache, you explicitly catch the dependency failure in your own code and read from a cache you control. This gives you more flexibility — you can apply a longer fallback TTL only on the *failure* path, log the degradation, and decide per-call whether stale is acceptable. We will see this in the fallback-chain code shortly.

### Partial responses: render what works, omit what doesn't

For composite pages — a product page made of price, photos, stock, reviews, recommendations — the strongest pattern is the **partial response**: render every section that succeeded and replace the broken section with nothing, a placeholder, or a "couldn't load" message scoped to that section only. The product page renders fully without the unavailable reviews module. This is the fix for the opening outage. The key engineering move is that each section is fetched independently and a failure in one is caught locally, so it cannot propagate to the page-level response.

The structural requirement is *fault isolation per section*. If you assemble the page by fetching all sections, then rendering, a single failed fetch that raises an exception will, by default, abort the whole render — and you are back to a 500. The fix is to make each section's fetch independently failable, so that one section's exception becomes that section's empty placeholder rather than the page's error. In code that looks like fetching each module behind its own try/except (or, in an async world, gathering with `return_exceptions=True` and substituting a fallback for each exception):

```python
async def render_product_page(product_id):
    # Fetch every section concurrently. Critical sections (price, stock)
    # are awaited and will fail the page if THEY fail. Optional sections
    # are isolated: their failure becomes an omission, not a page error.
    core = await fetch_core(product_id)          # P0: price, stock, photos
    if core is None:
        raise PageError(500)                     # core failed -> honest error

    async def safe(coro, default):
        try:
            return await asyncio.wait_for(coro, timeout=0.2)
        except Exception as e:
            log.warning("section degraded: %s", e)
            page_degraded.add(coro.__name__)     # feeds the degraded SLI
            return default

    reviews = await safe(fetch_reviews(product_id), default=None)   # omit
    recs, _ = await get_recommendations(product_id)                 # chain
    qa = await safe(fetch_qa(product_id), default=[])               # empty

    return ProductPage(core=core, reviews=reviews, recs=recs, qa=qa,
                       degraded=bool(page_degraded))
```

The line that does the work is the `safe()` wrapper: it converts any exception or timeout from an optional section into that section's default and records that the page was degraded. The `core` fetch is deliberately *not* wrapped in `safe()` — if price or stock fails, the page genuinely cannot serve and should return an honest error, because a product page with no price is not a degraded experience, it is a broken one. This is the criticality map (section 2) expressed in control flow: P0 sections propagate failure, P1/P2 sections absorb it.

### Default / static fallback: a sensible default when personalization is down

When a *personalization* dependency fails — the recs model, the per-user feed ranking — you do not have to show an error or a blank box. You show a **sensible default**: generic best-sellers instead of personalized recommendations, a default sort instead of a learned ranking, the editorial homepage instead of the for-you feed. The user gets a worse-but-reasonable experience. Sometimes the right default is to hide the module entirely; sometimes it is a static, pre-computed "popular this week" list that you refresh hourly and that costs nothing to serve.

The defining property of a good static default is that it is **independent of the failing dependency**. If your "generic popular" list is itself computed by the recs ML service, then when recs goes down your default goes down with it and you have no fallback at all — a depressingly common mistake. The default must live on a different failure path: pre-compute it on a schedule and store it in object storage or a CDN, bake a hard-coded list into the deploy artifact, or compute it from a different, more-reliable system. The whole point of a default is that it is *boring and reliable* — it does not need to be smart, it needs to be there when the smart thing is not. A good test: "if every dynamic service in my stack is down, can I still serve this default?" If the answer is no, your default is not actually a fallback.

There is a useful hierarchy of defaults, ordered by how much of your infrastructure they depend on. The most robust is a *static asset* — a JSON file of popular products on a CDN, regenerated hourly by a batch job; it survives even a total backend outage. Next is a *pre-computed cache value* with a long fallback TTL — depends on the cache being up but not on the source. Then a *computed-from-a-simpler-source* default — e.g., "show the category's top sellers from the catalog DB" depends on the catalog being up but not on recs. The further down this hierarchy your default sits, the more failures it can survive. For your most critical degradation paths, push the default as far up the hierarchy (toward static assets) as you can.

### The fallback chain: try fresh → cache → default → fail

Compose the above into a **fallback chain**, and you get a layered defense where each layer catches the failure of the one above it. Try the fresh source (with a tight timeout). If that fails, try the cache (last-known-good). If that misses, serve the static default. Only if *everything* fails do you finally degrade to hiding the module or returning an error. The figure below shows the chain as a branching flow: each layer has a "hit" exit to the served page and a "fail/miss" exit to the next layer down.

![A branching flow showing a recommendations request trying fresh, then cache, then a static default, with each layer falling through to the next on failure before a final miss hides the module](/imgs/blogs/graceful-degradation-and-fallbacks-3.png)

Here is the fallback chain as code — Python, but the structure is language-agnostic. Note the tight timeouts (we fail *fast* down the chain, we do not stack four 5-second timeouts) and the explicit `degraded` flag that we will use to measure degraded mode:

```python
import time
import logging

log = logging.getLogger("recs")

# Prometheus-style counter; increment by fallback tier served.
RECS_SERVED = Counter(
    "recs_served_total", "Recs responses by source", ["source"]
)

def get_recommendations(user_id, *, fresh_timeout=0.15):
    """Fallback chain: fresh -> cache -> static default -> hide.
    Returns (items, degraded). `degraded` drives the SLI in section 8.
    """
    # Layer 1: fresh, with a TIGHT timeout. We will not wait 5s for
    # a recommendation that is not coming -- the user is staring at
    # a spinner. Fail fast and fall through.
    try:
        items = recs_client.get(user_id, timeout=fresh_timeout)
        RECS_SERVED.labels(source="fresh").inc()
        cache.set(f"recs:{user_id}", items, fresh_ttl=300, fallback_ttl=3600)
        return items, False
    except (Timeout, ServiceError) as e:
        log.warning("recs fresh failed for %s: %s", user_id, e)

    # Layer 2: last-known-good from the cache (long fallback TTL).
    cached = cache.get_allow_stale(f"recs:{user_id}")
    if cached is not None:
        RECS_SERVED.labels(source="cache").inc()
        return cached, True   # degraded: stale, but real and personal

    # Layer 3: static default -- generic popular items, refreshed hourly,
    # never personal but always available.
    popular = cache.get("recs:popular_fallback")
    if popular is not None:
        RECS_SERVED.labels(source="default").inc()
        return popular, True  # degraded: generic, not personal

    # Layer 4: total miss. Hide the module rather than error the page.
    RECS_SERVED.labels(source="empty").inc()
    log.error("recs total fallback miss for %s", user_id)
    return [], True
```

Two things make this chain correct and not just clever. First, the timeout is tight (`fresh_timeout=0.15` — 150 ms) so we do not make the user wait for a recommendation that is not coming; we will return to *why* tight timeouts matter for non-critical reads in section 5. Second, every return path sets `degraded` honestly and increments a per-source counter, so we can measure exactly how often we are serving fresh versus cache versus default versus nothing. A fallback chain you cannot measure is a fallback chain you cannot trust.

## 4. Designing degradation per tier

Now compose the tiers (section 2) with the strategies (section 3). The design decision is not "do we have a fallback" — it is "which fallback, given this feature's tier and its tolerance for staleness." The matrix below maps three representative reads — product reviews, personalized recommendations, account balance — to a strategy, their staleness tolerance, and their on-failure behavior, and it shows that the *same word* ("fallback") means three completely different things depending on the read.

![A matrix mapping product reviews, personalized recommendations, and account balance to fallback strategy, acceptable staleness, and on-failure behavior](/imgs/blogs/graceful-degradation-and-fallbacks-8.png)

Reviews tolerate hours of staleness and are P2, so the strategy is a partial response — omit the module. Recommendations tolerate minutes of staleness and are P2, so the strategy is a fallback chain ending in generic-popular. Account balance tolerates *zero* staleness and is P0 — a wrong balance is worse than no balance, because it makes a financial decision on bad data — so the strategy is fail fast, no stale, show an error with a retry. This is the crucial nuance that distinguishes a thoughtful degradation design from a reckless one: **stale is not universally safe.** For most reads, stale data beats no data. For money, medical dosing, security tokens, and inventory-at-checkout, stale data can be actively dangerous, and the correct degradation is to fail fast and honestly.

The way to make this concrete in code is a small per-feature policy object — a config artifact your services read so that the degradation behavior is data, not scattered if-statements:

```yaml
# degradation-policy.yaml -- one entry per feature.
features:
  product_reviews:
    tier: P2
    strategy: partial_response       # omit the module
    stale_allowed: true
    fallback_ttl: 6h
    timeout_ms: 200
    on_total_failure: hide

  recommendations:
    tier: P2
    strategy: fallback_chain         # fresh -> cache -> default -> hide
    stale_allowed: true
    fallback_ttl: 1h
    timeout_ms: 150
    on_total_failure: generic_popular

  account_balance:
    tier: P0
    strategy: fail_fast              # never serve a wrong number
    stale_allowed: false             # money: stale is dangerous
    fallback_ttl: 0s
    timeout_ms: 800
    on_total_failure: error_with_retry
```

With policy as data, you can audit your degradation behavior in one place, you can review it without reading code, and — importantly — you can *test* it (section 7) by driving each feature's failure path from its policy. The `stale_allowed: false` flag on `account_balance` is the line that prevents an engineer from "helpfully" adding a stale-serving fallback to a financial read where it would cause real harm.

## 5. Fail soft for non-critical, fail fast for critical-but-down

There is a subtle but important asymmetry in degradation, and getting it wrong is one of the most common ways teams turn a partial outage into a worse one. The asymmetry is this:

- For a **non-critical** dependency, you want to **fail soft** — catch the failure, fall back, and keep going, ideally silently. The page renders without the widget.
- For a **critical** dependency that genuinely cannot serve, you want to **fail fast** — return a clear error *quickly* rather than make the user wait. Do not hang for 30 seconds on a checkout that is not going to complete; tell the user fast, let them retry.

The figure below stacks the spectrum: a P2 widget fails soft and silent with no wait; P1 search fails soft to a basic version with a banner; a P0 dependency that is slow fails *fast* under a second; and a P0 path that truly cannot serve returns a clear error and a retry.

![A four-layer stack showing P2 widgets failing soft and silent, P1 search degrading to basic with a banner, P0 dependencies failing fast under one second, and a P0 path that cannot serve returning a clear retryable error](/imgs/blogs/graceful-degradation-and-fallbacks-6.png)

Why does failing *fast* matter so much for the non-critical case too? Because the enemy of degradation is not the failure — it is the *latency* of the failure. A recommendation service that returns an error in 5 ms is easy to fall back from. A recommendation service that *hangs* for 30 seconds is a disaster, because every request waiting on it is holding a thread, a connection, and a slot in your concurrency budget. This is exactly where degradation ties into the resilience primitives. You want a **timeout** that is tight enough that a slow dependency converts to a fast failure you can fall back from, and you want a **circuit breaker** that, after enough failures, stops calling the broken dependency entirely and goes straight to the fallback — so you do not even pay the timeout while it is clearly down.

The arithmetic of why a slow non-critical call is so dangerous is worth making explicit. By Little's Law, the number of concurrent in-flight requests $L$ equals arrival rate $\lambda$ times average latency $W$: $L = \lambda W$. Suppose your service handles 500 requests per second and the recommendation call normally adds 20 ms ($W = 0.02$), so the recs call holds $L = 500 \times 0.02 = 10$ requests in flight at any moment — trivial. Now the recs service degrades and that call takes 10 seconds before timing out. Suddenly $L = 500 \times 10 = 5000$ requests in flight, all parked waiting on a dead dependency, exhausting your thread pool and connection limits. The non-critical call has consumed your entire concurrency budget and now *every* request — including the P0 checkout that does not even use recs — is failing because there are no threads left. **A slow non-critical dependency causes a critical outage through resource exhaustion, not through being critical.** This is the deepest reason tight timeouts and circuit breakers are not optional for soft dependencies.

The companion sibling on [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) covers how to choose those timeouts and avoid retry storms; the planned sibling on circuit breakers, bulkheads, and load shedding (`circuit-breakers-bulkheads-and-load-shedding`) covers the breaker and the bulkhead that isolate a failing dependency's resource pool so it cannot drain your whole service. At the architecture layer the system-design write-up on [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) explains the cascade mechanism in depth. The point to carry from *this* post is that degradation and those primitives are the same idea seen from two angles: the breaker decides *when* to stop calling, and the fallback decides *what to serve instead*.

#### Worked example: the timeout that saved checkout

A team I worked with had recommendations on the cart page, called synchronously with a 5-second timeout and three retries. During a recs outage, each request to the cart page made one fresh call (5 s), retried (5 s), retried again (5 s) — up to 15 seconds parked per request before falling back. At 300 cart-page requests per second, Little's Law gives $300 \times 15 = 4500$ requests in flight, which blew through the 2000-connection pool. The cart page — a P0 path — started returning errors because there were no connections left to talk to the *cart* service, all of them consumed waiting on *recommendations*. A P2 outage had become a P0 outage.

The fix was three lines of policy: timeout 150 ms, zero retries on this non-critical read, circuit breaker opens after 20 consecutive failures. Now during a recs outage, the breaker opens within a couple of seconds and every subsequent request goes straight to the cached/generic fallback in under a millisecond. In-flight requests on the recs call dropped from 4500 to effectively zero. The cart page stayed at 99.97% through the next recs outage. Before: a recs outage took down checkout for the duration. After: a recs outage was invisible to checkout. Same recs reliability — the dependency was just as flaky — but now it was *soft*.

## 6. The traps that turn degradation against you

Degradation is one of those practices where the naive version creates new and worse failure modes. Four traps, each of which I have personally been burned by.

### Trap 1: the hard dependency you didn't know was hard

This is the opening outage and the worked example above: a "non-critical" call that is synchronous and on the critical path. The synchronous non-critical call is the killer. Someone adds an analytics ping to the checkout flow — "just fire off an event when an order completes" — and writes it synchronously, in the request path, with the default 30-second client timeout. For months it is fine, because analytics is fast. Then analytics has a bad day and every checkout hangs for 30 seconds. The figure below shows exactly this: the supposedly-P2 analytics call sits synchronously inside the P0 checkout flow, and when it is slow it blocks the thread and times out the whole checkout — versus the fix, which is to make the call fire-and-forget onto an async queue so checkout never waits on it.

![A flow showing a synchronous analytics call inside the checkout path blocking and timing out the whole flow on the left, versus the same call made fire-and-forget onto an async queue so checkout completes on the right](/imgs/blogs/graceful-degradation-and-fallbacks-5.png)

The fix is a rule: **non-critical work on a critical path must be asynchronous, fire-and-forget, and tightly bounded.** Push the analytics event onto a queue and return; let a background worker deliver it. If it must be synchronous, give it a timeout measured in tens of milliseconds and a circuit breaker, and treat its failure as a no-op. The audit to find these is mechanical: trace every synchronous call on each P0 path and ask "is this dependency P0? If not, why is the P0 path waiting on it?" Every "it's just a quick call" you find is a future 3am page. For the async-delivery side — getting the event onto a queue reliably without blocking — the message-queue series covers delivery guarantees and backpressure; the relevant idea here is simply that fire-and-forget means *the request does not wait*.

### Trap 2: the fallback that's never tested

A fallback path is code that runs approximately never — only during an outage, which is exactly when you least want to discover it is broken. And fallbacks rot. The cache key format changes and the fallback reads garbage. The "generic popular" list points at a table that was deprecated. The partial-response template throws when reviews is `None` because someone refactored it assuming reviews is always a list. You will not catch any of this in normal operation, because the fallback path is cold. **An untested fallback is not a fallback — it is a second bug waiting for the first one.**

The only fix is to exercise the fallback *deliberately and regularly*. This is where degradation meets chaos engineering: you inject the dependency failure on purpose, in a controlled way, and verify the fallback fires correctly and the user still gets a useful response. A minimal chaos experiment using Chaos Mesh to fail the reviews service for the storefront and assert the product page still returns 200:

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: reviews-down-game-day
  namespace: storefront
spec:
  action: pod-failure
  mode: all
  duration: "5m"
  selector:
    namespaces: [storefront]
    labelSelectors:
      app: reviews-service
  # During this 5m window an automated probe asserts:
  #   - GET /product/123 returns 200 (NOT 500)
  #   - the response omits the reviews module
  #   - product_page degraded SLI > 0 and an alert fired
```

Run this as a scheduled **game day** — say monthly — in staging first, then in production during low traffic once you trust it. The assertion is the important part: it is not enough to fail the dependency; you must *verify the fallback behavior*, including that the degraded-mode SLI moved and the right (low-urgency) alert fired. The sibling posts on self-healing and chaos-style validation in this series go deeper on game-day mechanics; the point here is that a fallback you have not failed-over in the last quarter is a fallback you should assume is broken.

There is a cheaper, always-on version of this that complements the periodic game day: a **fallback smoke test in CI**. Every build, run an integration test that points the service at a deliberately-dead version of each soft dependency (a stub that returns 503, or a sleep that exceeds the timeout) and assert that the response is the degraded one you expect — the page renders, the module is omitted, the status code is 200, the `degraded` flag is set. This catches the most common fallback rot — a refactor that assumed the dependency is always present — at build time, before it ever ships. The game day catches the things CI cannot: real network behavior, real connection-pool exhaustion, the breaker actually tripping under real concurrency, the alert actually paging the actual on-call. You want both: CI for "the fallback code still works as written," game days for "the fallback works under real conditions." A team that has both can say, with evidence, "we know what happens when reviews goes down, because we make it go down on purpose every month and we tested the path on every build." That sentence is the difference between hoping you degrade and knowing you do.

One more discipline that pays off: when a fallback *does* fire in production for real, treat it as a small signal worth reviewing, not just a ticket to close. The fact that the fallback fired means a dependency failed — which is information about that dependency's reliability — and the fact that the fallback *worked* (or did not) is information about your degradation design. A monthly review of "which fallbacks fired, how often, and did they behave correctly" turns every real degradation into a free game day you did not have to schedule. It is the cheapest validation there is, because production ran the test for you; you just have to look.

### Trap 3: the silent degradation nobody notices

Here is a failure mode that *feels* like a success and is therefore the most insidious. Your fallback works perfectly: the source dies, the cache serves stale, the page renders, no errors, dashboards green, on-call sleeps. And it keeps serving stale for six hours because *nobody knows the source is down* — the fallback is doing its job so well that it has hidden the outage. Then a user notices the prices are from this morning, files a ticket, and you discover you have been silently degraded since dawn. The figure below is this exact timeline: the source dies at 09:00, the fallback serves stale at 09:01, dashboards stay green, prices go three hours stale by noon, and you only find out at 15:00 from a user report.

![A timeline showing a source dying at 09:00, the fallback serving stale at 09:01, dashboards staying green, prices going stale by noon, and a user reporting wrong prices at 15:00, ending with adding a degraded-mode SLI and page](/imgs/blogs/graceful-degradation-and-fallbacks-7.png)

The principle that resolves this is **heal-and-page in parallel**: a fallback should make the system *behave* fine for the user *and* tell the operators it is running degraded, at the same time. The fallback heals the user-facing symptom; the alert pages (at appropriate urgency) so someone fixes the actual source. You never want a fallback that *only* heals, because then degradation becomes invisible and permanent. We will build the degraded-mode SLI and the alert that does this in section 8 — it is the single most important piece of measurement in this whole post.

### Trap 4: the fallback that masks a real outage forever

A relative of trap 3 with a longer fuse. Your fallback TTL is "until the source recovers," with no upper bound. The source recovers... except it does not, because the team that owns it quietly decommissioned it three weeks ago and your traffic is the only thing still calling it. You are now serving a value frozen three weeks ago, forever, with no error and no alert, and you find out when the data is so stale it is wrong in a way customers complain about. The fix is two-fold: a **bounded fallback TTL** (after, say, 4 hours of staleness on a price, *stop* serving stale and start failing — at that point an error is more honest than a lie), and a **staleness SLI** that measures the *age* of what you are serving, not just whether you are serving. Serving a value is not the same as serving a *correct* value; measure the age.

## 7. Measuring degraded mode — the SLI you forgot

Everything in section 6 points at one missing measurement: you need an SLI that distinguishes "served fully" from "served degraded." This is the part that separates teams who *do* degradation from teams who *survive* it. (SLI = Service Level Indicator, the ratio of good events to total events that you measure; if that is new, the sibling [SLI, SLO, SLA — the three numbers that matter](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter) defines them, and [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) covers how to pick ones that track what users actually feel.)

The classic availability SLI is binary: a request either succeeded (good) or errored (bad). But degradation breaks that binary, because a *degraded* response is neither — it succeeded, but it was worse. So you instrument three states, not two: **full**, **degraded**, and **failed**. Your code already knows which it served — that is what the `degraded` flag in the fallback chain (section 3) was for. Now you emit it as a metric label.

```python
# In the request handler, after assembling the page:
PAGE_SERVED.labels(
    route="product_page",
    quality="full" if not any_module_degraded else "degraded",
).inc()

PAGE_FAILED = Counter("page_failed_total", "5xx responses", ["route"])
```

Now you can compute two distinct SLIs in PromQL. The **availability SLI** (did we serve anything useful, full or degraded?) and the **full-experience SLI** (did we serve the *good* version?):

```promql
# Availability SLI: fraction of requests served at all (full OR degraded),
# over a 30-day rolling window. THIS is what most users feel.
sum(rate(page_served_total{route="product_page"}[30d]))
/
(
  sum(rate(page_served_total{route="product_page"}[30d]))
  + sum(rate(page_failed_total{route="product_page"}[30d]))
)

# Full-experience SLI: of everything we served, how much was the GOOD
# version (not degraded)? This is the one that catches silent degradation.
sum(rate(page_served_total{route="product_page", quality="full"}[30d]))
/
sum(rate(page_served_total{route="product_page"}[30d]))
```

The first ratio is high even during a reviews outage — because a degraded page still counts as served. That is correct: from the user's perspective, the store works. The *second* ratio drops the moment you start serving degraded pages, and *that* is the signal that catches silent degradation. You set an SLO on availability (the user-facing promise) and you *alert* on the full-experience SLI dropping (the operator-facing signal that you are running degraded), and you have implemented heal-and-page in parallel as two separate numbers.

Why two SLIs rather than one? Because they answer two genuinely different questions, and conflating them is how teams either over-alert or under-alert. The availability SLI answers "are users getting a usable product?" — and its SLO is the promise you make and the budget you burn against. The full-experience SLI answers "are we running on fallbacks?" — and it is not a promise to users, it is a tripwire for operators. If you only had the availability SLI, you would never see the silent degradation: the store works, availability is green, and the six-hour stale-price outage is invisible. If you only had the full-experience SLI, you would page yourself every time a single P2 widget blinked, because *any* degradation would look like a problem even when users are completely fine. Splitting them lets you promise the right thing (availability) and watch the right thing (degradation), with the right urgency on each.

This also clarifies how degradation interacts with the error budget. A degraded response does not spend availability budget — it counts as a good event in the availability SLI. So an outage you fully absorb through degradation costs you *zero* availability budget, which is the whole reason degradation is such a powerful budget-preservation tool. But it is not *free*: it spends *full-experience* budget, and if you treat the full-experience SLI as a real (softer) SLO with its own budget, you create the right incentive — fallbacks are cheap but not infinitely cheap, so the team is still motivated to fix the underlying source rather than living on fallbacks forever. That second budget is what stops trap 4 (the fallback that masks an outage forever) from becoming a cultural default. There is a longer treatment of how to slice and burn budgets across multiple SLIs in [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability); the degradation-specific lesson is that "served" and "served well" deserve separate budgets.

A practical instrumentation note: emit the `quality` label at the level the *user* experiences, not at the level of each internal call. If you increment a degraded counter for every internal cache hit, your full-experience SLI becomes noise, because internal caching is normal and healthy. What you want to count as "degraded" is when the *user-facing response* is worse than the full experience — a missing module, a generic-instead-of-personal recommendation, a stale price past its fresh window. Aggregate the per-section degradation up to one per-response verdict (`degraded = any section degraded`) and label the response with that. The dashboard, then, reads as "what fraction of pages did we serve at full quality," which is exactly the human question.

Now the alert that resolves traps 3 and 4 — a low-urgency page when the degraded fraction crosses a threshold, paired with a separate staleness alert:

```yaml
groups:
- name: degradation
  rules:
  # The recording rule: fraction of product-page responses
  # served degraded, over 10 minutes.
  - record: product_page:degraded_fraction:rate10m
    expr: |
      sum(rate(page_served_total{route="product_page",quality="degraded"}[10m]))
      /
      sum(rate(page_served_total{route="product_page"}[10m]))

  # Heal-and-page: the fallback is healing users, but we are
  # running degraded -- page someone (low urgency) to fix the source.
  - alert: ProductPageRunningDegraded
    expr: product_page:degraded_fraction:rate10m > 0.05
    for: 5m
    labels:
      severity: ticket          # NOT a 3am page -- users are fine
    annotations:
      summary: "Product page serving {{ $value | humanizePercentage }} degraded"
      description: "A dependency is down; fallbacks are firing. Find and fix the source."
      runbook: "https://runbooks.example.com/product-page-degraded"

  # Trap 4: we are serving STALE that is too old to be honest.
  - alert: PricingFallbackTooStale
    expr: max(pricing_fallback_age_seconds) > 14400   # 4 hours
    for: 5m
    labels:
      severity: page            # stale prices for 4h is now real harm
    annotations:
      summary: "Pricing fallback is {{ $value | humanizeDuration }} stale"
      description: "Past the bounded fallback TTL. Stop trusting the cache; the source has been down too long."
```

Notice the severities are deliberately different. `ProductPageRunningDegraded` is a `ticket`, not a page — because the user experience is *fine*, you have time, you do not wake anyone at 3am for a degraded page that is doing its job. But `PricingFallbackTooStale` is a `page`, because four-hour-old prices have crossed from "useful fallback" into "wrong data that could cost money." The alerting philosophy here — page on user pain, ticket on operator signal — comes straight from the sibling [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf); degradation is a perfect example of why severity must match impact, not match "something is wrong."

#### Worked example: a recommendations outage, fully handled

Let me run the whole machine end to end on a real-shaped incident, because the pieces only click when you see them together.

It is 11:00. The recommendations ML service starts returning 503s — a bad model deploy. Here is what happens, second by second, with the SLIs:

1. **11:00:00** — Recs starts erroring. The fallback chain (section 3) catches the first failures: fresh fails (150 ms timeout), cache hits for users who were served in the last hour, generic-popular for the rest. `recs_served_total{source="cache"}` and `{source="default"}` climb; `{source="fresh"}` drops toward zero.
2. **11:00:02** — The circuit breaker trips after 20 consecutive fresh failures. Now requests skip the 150 ms fresh attempt entirely and go straight to cache/default in under a millisecond. In-flight requests on the recs call stay flat — no resource exhaustion, no cascade. Checkout (P0, does not use recs) is completely unaffected.
3. **11:00:05** — `product_page:degraded_fraction` crosses 5%. The `for: 5m` timer starts.
4. **11:05:00** — `ProductPageRunningDegraded` fires as a *ticket*. The on-call gets a low-urgency notification, not a 3am page. They see "product page serving 18% degraded, recs is the source," open the runbook, and start the rollback of the bad recs model.
5. **11:00–11:14** — Throughout, the **availability SLI** stays at 100.00% (every request served a useful page) while the **full-experience SLI** sits at about 82% (18% of pages showed generic instead of personalized recs). Users see slightly-worse recommendations. Nobody sees an error. Revenue is unaffected.
6. **11:14:00** — Recs model rolled back, fresh calls succeed, breaker closes, `degraded_fraction` returns to zero, the ticket auto-resolves.

The proof, stated as the before→after this series demands: **before** this design, a recs 503 storm took down the product page (the cart Little's-Law cascade) — call it 40 minutes of full product-page outage per recs incident, several times a year, each one eating deeply into the error budget. **After**: zero user-facing product-page errors during the recs outage, the full-experience SLI dipped to 82% for 14 minutes, and the error budget spent on this incident was *zero availability minutes* — it cost only a low-urgency ticket and a model rollback. The outage became a non-event. That is what "spend less budget by degrading" means in concrete numbers.

## 8. War story: the dependencies that took everyone down

Graceful degradation is best taught by the outages that happened because it was missing. A few real-shaped ones — some documented, some composites I will flag as illustrative.

**Facebook, March 2019.** A configuration change triggered a multi-hour partial outage across Facebook, Instagram, and WhatsApp. The relevant lesson for us is how *coupled* the failure was — a problem in a shared backend subsystem degraded user-facing products that, ideally, should have been able to serve a reduced experience independently. When core shared infrastructure is a hard dependency for everything, there is no "degrade" — everything collapses together. The defensive design is to make as much of the product as possible *soft*-dependent on shared infra, with cached and default behaviors, so a shared-infra wobble subtracts features rather than taking the whole product to an error page.

**The dependency-graph cascade (the general pattern, well documented across the industry).** The Google SRE Book devotes real attention to *cascading failures* and *addressing cascading failures* precisely because the most damaging outages are rarely a single component dying — they are a non-critical degradation that, through synchronous coupling and retry amplification, cascades into a total outage. The book's prescriptions are the ones in this post: tight timeouts, circuit breakers, load shedding, and *graceful degradation* — serving a reduced response under stress rather than collapsing. Google's own services are explicitly designed to shed and degrade: Search will return a result from a subset of its index if some shards are slow, rather than waiting for every shard. A partial answer fast beats a complete answer never.

**The synchronous-analytics outage (illustrative composite, but I have seen variants of it three times).** A retailer adds a real-time personalization/analytics call to the checkout button — synchronous, default timeout, no breaker. Black Friday arrives, the analytics vendor's endpoint slows under load, every checkout hangs for 30 seconds, the connection pool exhausts, and checkout — the one P0 path that *must not fail* — fails at the worst possible moment, all because of a P2 call that should have been fire-and-forget. The postmortem finding is always the same: "a non-critical dependency was synchronous on the critical path." This is trap 1, and it is the single most common degradation failure in production.

**Amazon's "everything degrades" culture (documented in their architecture talks).** The flip side — what good looks like. Amazon's retail pages are explicitly built so that each module (recommendations, "customers also bought," reviews, related items) fetches independently and a failure in any one renders that module empty rather than failing the page. The product page is a composition of soft dependencies on a small set of hard ones (price, availability, the buy box). This is partial-response degradation as a *cultural default*, applied across thousands of services, and it is a large part of why the retail site stays up through dependency outages that would 500 a more naively-coupled design.

The thread through all four: outages cascade through *synchronous, hard, unbounded* coupling, and they are absorbed by *asynchronous-or-bounded, soft, degradable* design. The anatomy of how these cascades unfold once they start is covered in the system-design post on [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads), and the caching-specific failure modes — the stampede, the stale-forever, the cache as a hidden hard dependency — are in [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite), which is worth reading alongside the serve-stale section above.

## 9. Stress-testing the design

A degradation design is only as good as its behavior in the cases you did not plan for. Let me stress-test the storefront design against the hard questions.

**What if the dependency is down for two hours, not two minutes?** This is the serve-stale-vs-bounded-TTL tension. For reviews (P2, hours of staleness fine), two hours is no problem — the module stays omitted, no harm. For recommendations, the cache fallback TTL is one hour, so after an hour you fall through to generic-popular, which has no staleness problem (it is refreshed hourly and is generic by design). For pricing, the bounded fallback TTL of four hours means you serve last-known-good prices for up to four hours, then the `PricingFallbackTooStale` page fires and you make a *business* decision: keep serving stale prices (and accept the risk) or start failing the price (and accept the error). The design forces that decision to be made consciously rather than defaulting to "serve stale forever."

**What if the on-call is asleep?** The `ProductPageRunningDegraded` alert is a `ticket`, not a page, precisely so it does *not* wake anyone — because users are fine. That is correct: degradation that is working as designed should not interrupt sleep. The thing that *will* page is `PricingFallbackTooStale` (real harm) or the upstream source's own availability alert (the actual outage). The degradation buys you the slack to handle the source outage during business hours instead of at 3am, which is itself a reliability win — see [designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call) for why protecting sleep is a first-order goal.

**What if two dependencies fail at once — reviews and recommendations?** The partial-response design composes: each module fails independently, so the product page renders with both modules omitted/defaulted. The full-experience SLI drops further (now two modules degraded), but availability holds. The danger to check for is *correlated* failure that hits a shared resource — e.g., if both reviews and recs share a cache cluster and the cache is what failed. That is why bulkheads matter: isolate the resource pools so one dependency's failure cannot starve another's fallback. The breaker-and-bulkhead sibling covers this isolation.

**What if the fallback itself is the bottleneck?** Suppose recs is down and *everyone* falls through to the generic-popular cache key — that single key now takes the full read load. This is a cache stampede risk. Mitigations: serve generic-popular from a replicated cache or even a static file/CDN (it is the same for all users, so cache it aggressively and globally), and use request coalescing (`proxy_cache_lock`) so a thousand concurrent misses become one backend fetch. The fallback path must be *more* robust than the primary, not less — it is the path that runs during an outage.

**What if the error budget is already spent?** Then degradation is your best friend, because a degraded response does not spend availability budget (it counts as served). When you are out of budget and in a freeze, shipping a degradation/fallback *is* the kind of reliability work the freeze is meant to prioritize. The error-budget mechanics — what a freeze is, when it triggers — are in [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability).

## 10. How to reach for this (and when not to)

Graceful degradation is high-leverage but it is not free, and applying it indiscriminately creates its own problems. Here is when to reach for each piece and when to leave it on the shelf.

**Do build a fallback when** the dependency is non-critical (P1/P2), the read tolerates some staleness or a sensible default exists, and the dependency is on a path that real users hit. This is the 80% case and the ROI is enormous — recall the 77% downtime reduction from making one dependency soft.

**Do not serve stale when** the data must be correct: money, security tokens, inventory at the moment of purchase, medical or safety data, anything where a wrong value causes real harm. For these, fail fast and honest. A stale balance is worse than an error. The `stale_allowed: false` policy flag exists to enforce this.

**Do not add a fallback you will not test.** An untested fallback is a liability — it gives false confidence and breaks silently. If you cannot commit to a recurring game day that exercises it, you are better off with a simple, well-understood fail-fast than an elaborate fallback chain that is secretly broken. Fewer, tested fallbacks beat many rotting ones.

**Do not over-engineer the tail.** Not every read needs a four-layer fallback chain. A P2 widget on a low-traffic admin page can simply fail-fast-and-hide; building serve-stale plus cache plus default plus degraded-SLI for it is toil with no payoff. Match the investment to the tier: P0 paths get the full treatment, P2 widgets get "hide it." Spending a week building a fallback chain for a module that 0.3% of internal users see is exactly the kind of misallocated reliability effort the error-budget framework is meant to prevent.

**Do not let degradation become permanent and invisible.** The single most important guardrail: every fallback must be *measured* (the degraded-mode SLI) and *bounded* (a fallback TTL with an alert). A fallback without a degraded-mode SLI is a future silent six-hour outage. If you build nothing else from this post, build the degraded-fraction recording rule and the ticket-severity alert.

**Reach for fail-fast over degradation when** the dependency is genuinely critical and there is no useful reduced response. Do not invent a fake fallback for a thing that has no honest worse-version — a payment authorization either happened or it did not. For these, the "graceful" behavior is a fast, clear error with a retry path, not a pretend success.

## Key takeaways

- **Degrade, don't collapse.** A non-critical dependency failing should subtract a feature, never take down a critical path. That one rule prevents most of the worst outages.
- **Soft beats reliable.** Making a dependency *soft* (tolerated on failure) removes it from your availability product entirely — a far bigger win than making it slightly more reliable. One dependency going from hard to soft bought back 77% of downtime in our worked example.
- **Map criticality before you design failure.** P0 must never error; P1 degrades to basic; P2 may vanish silently. Keep the map in your repo and review it quarterly. The core flow must survive any single non-core dependency.
- **Pick the fallback per read.** Serve stale and partial responses for stale-tolerant reads; fail fast with no stale for money and safety. Stale data beats no data — except where stale data is dangerous.
- **Build the chain: fresh → cache → default → fail.** Each layer catches the one above. Tight timeouts so you fail fast down the chain; a circuit breaker so you skip the dead layer entirely.
- **The synchronous non-critical call is the killer.** Make non-critical work on a critical path async/fire-and-forget/timeout-fast. A slow P2 call exhausts your concurrency budget (Little's Law) and causes a P0 outage.
- **Test your fallbacks or they are not fallbacks.** Exercise them with scheduled chaos game days and assert the user-facing behavior, or assume they are broken.
- **Measure degraded mode or it becomes invisible.** Emit full/degraded/failed; SLO on availability, alert (low urgency) on the full-experience SLI; bound the fallback TTL and page when stale crosses into harm. Heal the user and page the operator, in parallel.

## Further reading

- [Reliability is a feature — the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the define→measure→budget→respond→learn→engineer loop this post slots into.
- [Designing for failure: failure domains, blast radius, and the SPOF you didn't see](/blog/software-development/site-reliability-engineering/designing-for-failure) — how to map failure domains and shrink blast radius; degradation is the runtime behavior you choose once they are mapped.
- [Timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) — choosing the tight timeouts that convert slow dependencies into fast, fallback-able failures, without causing retry storms.
- [Choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) — picking the indicators that distinguish "served fully" from "served degraded" so you measure what users actually feel.
- The planned sibling on circuit breakers, bulkheads, and load shedding (`circuit-breakers-bulkheads-and-load-shedding`) — the breaker that stops calling a dead dependency and the bulkhead that isolates its resource pool so it cannot drain your service.
- [Cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) (system design) — the architecture-time view of how cascades unfold and the patterns that stop them.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) (system design) — the serve-stale failure modes: stampedes, stale-forever, and the cache as a hidden hard dependency.
- *Site Reliability Engineering* (Google), chapters on "Addressing Cascading Failures" and "Handling Overload" — the canonical treatment of graceful degradation, load shedding, and serving reduced responses under stress.

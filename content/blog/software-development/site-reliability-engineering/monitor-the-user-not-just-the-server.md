---
title: "Monitor the User, Not Just the Server: Black-Box, Synthetic, and Real-User Monitoring"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn why every dashboard can be green while the CEO emails that the site is down, and build the three vantage points — white-box, black-box synthetic, and real-user monitoring — that catch what your server never sees."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "monitoring",
    "observability",
    "synthetic-monitoring",
    "real-user-monitoring",
    "black-box-monitoring",
    "core-web-vitals",
    "sli",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/monitor-the-user-not-just-the-server-1.png"
---

It is 9:14 on a Tuesday morning and your phone lights up with a message from the CEO: "Site is down. Customers are complaining. What's going on?" You open the dashboards. Every panel is green. Request rate looks normal. The p99 latency — the latency that only one request in a hundred exceeds — is a calm 80 milliseconds. The error rate at the load balancer is flat zero. Your service has handled forty-one thousand successful requests in the last five minutes, every one of them a clean HTTP 200. By every number you collect, the system is perfectly healthy.

And yet the CEO is right. The site *is* down. Three different customers have sent screenshots of a blank checkout page. Support is fielding calls. Twitter has started. Somewhere between the user's browser and the counter that says "100% success," reality forked. You measured the server. The user lives somewhere else.

This is the single most expensive lesson in monitoring, and almost everyone learns it the hard way: **the only reliability that matters is the reliability the user experiences.** Your server's opinion of its own health is interesting, but it is not the truth. The truth lives in the user's browser, on the user's network, after the user's DNS resolver, behind the user's CDN edge, through the load balancer that your server cannot fully see. A metric collected inside your data center answers the question "is my server doing its job?" It does not answer the question that pays your salary: "can the user do the thing they came to do?" Those are different questions, and on the worst mornings of your career they will give different answers.

![A topology diagram showing six hops between the real user and the app server, with DNS and CDN branching off to a broken-page outcome while the server counter reads green.](/imgs/blogs/monitor-the-user-not-just-the-server-1.png)

By the end of this post you will be able to do something concrete: build a monitoring posture that measures reliability from where the user actually stands. You will know the difference between **white-box** monitoring (your server's internal view) and **black-box** monitoring (probing the system from outside as a user would), and exactly which failures each one is blind to. You will write a **synthetic** probe — a scripted check that logs in, searches, and checks out from five regions every minute — and a **real-user monitoring** (RUM) snippet that captures what actual browsers felt, down to the Core Web Vitals and the JavaScript errors your backend never sees. You will be able to draw, for any incident, the exact reason a dashboard was green while users were furious, and you will have the three complementary vantage points that make "green dashboard, angry users" a thing that happened to you once, not every quarter.

This sits squarely in the measure-it stage of the reliability loop this series keeps coming back to: define reliability (with [SLIs and SLOs](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain)), measure it honestly, spend the error budget, reduce toil, respond to incidents, and learn. If you have not read the [intro to the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), start there for the big picture; this post is the deep dive on the "measure it from the user's side" half of that loop. Reliability is a feature, and a feature you cannot measure from the user's seat is a feature you are only pretending to ship.

## 1. The core split: white-box versus black-box

Let me define the two halves of the world, because the rest of this post is a long argument about how to combine them.

**White-box monitoring** is everything you instrument from the inside. Your service exports a counter every time it handles a request. Your load balancer reports how many connections it accepted and how many 5xx responses it returned. Your database publishes query latency, connection-pool saturation, and replication lag. Your Kubernetes nodes report CPU, memory, and disk. The defining property of white-box monitoring is that it watches the system *from within the system* — you have privileged access to internal state, and you measure things the outside world can never see directly. This is the world of Prometheus counters, the four golden signals (latency, traffic, errors, saturation), the [RED method](/blog/software-development/site-reliability-engineering/metrics-and-time-series-done-right) (Rate, Errors, Duration) for request-driven services, and the USE method (Utilization, Saturation, Errors) for resources. White-box monitoring is fantastic at telling you *why* and *where* something is broken once you know it is broken.

**Black-box monitoring** is the opposite stance. You stand outside the system, where the user stands, and you ask a single blunt question: when I do what a user does, does it work? You make a real HTTP request to the real public URL. You wait for a response. You check that the status code is 200 and the body contains the words you expect ("Add to cart," "Your order is confirmed"). You measure how long it took, end to end, from outside. Black-box monitoring has no privileged access and no opinions about internal state. It knows nothing about your CPU or your connection pool. It knows exactly one thing, and it knows it the way the user knows it: did the request succeed from out here?

The reason this distinction is not academic is that the two stances disagree precisely when it matters most. Look back at figure 1. Between the user's browser and your app server's success counter there are at least six independent hops, and a failure at most of them is *invisible* to white-box monitoring by construction:

- **DNS** fails to resolve your domain. Your servers are fine; the request never arrives. White-box: green.
- **The CDN or its TLS certificate** breaks. The edge serves an error or refuses the handshake. Your origin sees less traffic, which looks like a quiet morning. White-box: green.
- **The load balancer is healthy but routing wrong** — pointing at a stale target group, or an old deployment. The LB reports 200s because *its* backends answer; they are just the wrong backends. White-box: green.
- **A JavaScript error in the page** crashes the render after a clean 200. The HTML and all the assets returned successfully. The browser then threw an exception and showed a blank screen. White-box: gloriously, perfectly green.
- **A third-party script** (payments, analytics, a chat widget) fails to load and blocks the critical path. Your origin returned everything it owns. White-box: green.
- **A whole region** your servers don't live in goes dark — a regional ISP, a regional DNS resolver, a CDN PoP. Users in that region are down; your origin in another region is happy. White-box: green.

Notice the pattern. Every one of these is a real outage from the user's point of view, and every one of them produces *no signal at all* in metrics collected inside the data center. The white-box view is not lying — it is faithfully reporting that the server did its job. The problem is that "the server did its job" and "the user got what they came for" are different claims, and white-box monitoring can only ever verify the first.

### Why this is a measurement-point problem, not a tooling problem

Here is the principle, stated as precisely as I can. An availability SLI is a ratio: good events over total events, over a window. The number you get depends entirely on **where you stand to count.** If you count at the app server, "total" means "requests that reached the app server" — which silently excludes every request that died before it got there. Your denominator is missing exactly the failures that hurt the most. You can have a flawless 100% measured at the app, and a catastrophic 70% measured at the user, and *both numbers are arithmetically correct.* They are answering different questions about different populations of requests.

This is why you cannot fix the "green dashboard, angry users" problem by buying a better white-box tool, adding more Prometheus exporters, or tuning your alert thresholds down. No amount of internal instrumentation moves the measurement point outward. The only way to measure what the user feels is to measure *where the user is.* You move the measurement point to the edge of your system (black-box synthetic probes hitting the public URL) and, ideally, all the way into the user's browser (real-user monitoring). That is the whole game.

```
White-box question: "Is my server doing its job?"   -> answered by internal metrics
Black-box question: "Does it work from outside?"    -> answered by synthetic probes
RUM question:       "What did real users feel?"      -> answered from the browser
```

I broke my own rule there and used a plain code block instead of a diagram, which is fine because it is three lines of plain text, not box-drawing. The point stands: three questions, three measurement points, and a white-box tool can only ever answer the first one.

## 2. The green-dashboard, angry-users failure catalog

Let me make this catalog concrete, because "it could fail anywhere" is not actionable. There is a finite, well-known list of failures that read green inside and red outside, and once you have seen the list you start spotting them in every incident review. They split cleanly into two families: failures *on the path* between the user and your server, and failures *inside* the user's own browser.

They split cleanly into two families, and the table at the end of this section maps each one to the vantage point that catches it. The taxonomy tree that organizes these failures into path-side and client-side families is figure 6, which we will reach under section 4. Let me walk the families now and you will recognize each one when its figure arrives.

**Path failures (between the user and your origin):**

- **DNS outage or misconfiguration.** Your domain stops resolving, or resolves to the wrong place after a bad change to a record. The 2016 Dyn DNS outage took down a huge slice of the consumer internet — Twitter, Spotify, Reddit, GitHub — and not one of those companies' origin servers had a problem. Their *names* stopped resolving. Every origin dashboard was green.
- **TLS certificate expiry.** A cert silently expires and browsers refuse the connection with a scary full-page warning. The origin is up and would happily serve traffic; the handshake never completes. This is the single most common "completely down but completely green" failure, and it is entirely preventable, which makes it the most embarrassing.
- **CDN or edge failure.** Your CDN provider has a bad day, or your edge config breaks, or a cache rule starts serving a 403 for your main bundle. Origin traffic *drops*, which on a white-box dashboard looks like a slow morning, not an outage.
- **Load balancer routing wrong.** The LB health checks pass — the targets it points at answer 200 — but it points at the wrong target group, an old deployment, or a region that should have been drained. Healthy, and serving the wrong thing.
- **A region your servers can't see.** A regional network partition, a regional resolver outage, a single CDN PoP failing. Users in `eu-west` are down; your `us-east` origin is thriving.

**Client-side failures (inside the browser, after a clean 200):**

- **A JavaScript error.** The HTML and assets all returned 200. Then a null-reference exception in your bundle crashed the render, and the user sees a white screen. Your server is blameless and your server is green.
- **A CDN/asset failure for one file.** The HTML loaded but the main JavaScript bundle or a critical CSS file 404'd or got a stale hash from the CDN. The page is half-built and unusable.
- **A third-party script.** Your payment provider's SDK, your analytics, your A/B framework — anything you load from someone else's domain — fails to load or throws, and it sits on the critical path. We will live through exactly this in the first worked example.
- **A bad deploy that only breaks in real browsers.** A polyfill you dropped, a browser API you assumed, a Content-Security-Policy header that blocks your own inline script. Works in your test environment, broken on the user's actual Safari.

Here is the table I keep pinned, because it is the fastest way to answer "would we have caught this?" Each row is a failure mode; each column is a vantage point and what it sees.

| Failure mode | White-box (server) | Black-box synthetic | Real-user RUM |
| --- | --- | --- | --- |
| DNS outage | Blind — origin green | Catches — probe can't resolve | Catches — load failures spike |
| TLS cert expiry | Blind — origin green | Catches — handshake fails | Catches — sessions drop to zero |
| CDN / asset 404 | Blind — origin served it | Catches — asset check fails | Catches — render breaks, CLS jumps |
| LB routing wrong | Mostly blind — targets 200 | Catches — wrong content returned | Catches — wrong page or errors |
| Client-side JS error | Blind — backend perfect | Catches — scripted step fails | Catches — `onerror` fires |
| Third-party script dead | Blind — origin perfect | Catches — checkout step fails | Catches — JS error rate spikes |
| Regional outage | Partly blind — one region | Catches — region probe red | Catches — that region's RUM dark |
| Backend 5xx surge | **Catches** — error rate up | Catches — probe gets 5xx | Catches — error pages |
| Slow database query | **Catches** — duration up | Catches — probe latency up | Catches — LCP/INP degrade |

![A matrix mapping four green-dashboard failure modes against three vantage points, showing the white-box column blind on most rows while the synthetic and real-user columns catch them.](/imgs/blogs/monitor-the-user-not-just-the-server-3.png)

Read the white-box column top to bottom. For the first seven rows — every single failure that fits the "green dashboard, angry users" pattern — white-box is blind or partly blind. It only earns its keep on the last two rows, where the failure actually happens inside the server. That is the whole case for black-box and RUM in one table: they exist to cover the rows where the inside view is dark.

## 3. White-box: necessary, not sufficient

Before I spend ten sections arguing for monitoring the user, let me be scrupulously fair to white-box monitoring, because the lesson is "white-box is not *sufficient*," not "white-box is useless." White-box monitoring is the only thing that tells you *why.* When the user-facing signal goes red, white-box is how you find the cause in minutes instead of hours: which service started erroring, which dependency's latency spiked, which node ran out of memory, which deploy correlates with the turn. Black-box tells you the patient has a fever; white-box tells you it's the appendix.

![A before-and-after contrast showing white-box metrics reading 100 percent success and fast latency on the left while real-user monitoring shows a 38 percent JavaScript error rate and zero checkouts on the right.](/imgs/blogs/monitor-the-user-not-just-the-server-2.png)

That contrast is the whole reason white-box is necessary but not sufficient: the left column is the server's own view reading perfect, the right column is the real user's view reading broken, and they are the same system in the same minute. Here is a perfectly good white-box availability SLI as a Prometheus recording rule. This is the kind of rule you absolutely should have — it is the foundation, not the enemy.

```yaml
# recording rules: server-side request SLIs (white-box)
groups:
  - name: checkout-slo.rules
    interval: 30s
    rules:
      # numerator: successful requests (non-5xx) to the checkout API
      - record: job:checkout_requests_good:rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout",code!~"5.."}[5m]))

      # denominator: all requests to the checkout API
      - record: job:checkout_requests_total:rate5m
        expr: |
          sum(rate(http_requests_total{job="checkout"}[5m]))

      # the SLI: good / total over the last 5 minutes
      - record: job:checkout_availability:ratio5m
        expr: |
          job:checkout_requests_good:rate5m
            /
          job:checkout_requests_total:rate5m
```

And the matching latency SLI, computed from a histogram with `histogram_quantile`:

```promql
# p99 checkout latency, server-side, last 5 minutes
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket{job="checkout"}[5m])) by (le)
)
```

These are good rules. They will catch a backend 5xx surge or a slow-query regression fast, and you should keep them. But re-read the `expr` of the denominator one more time: `sum(rate(http_requests_total{job="checkout"}[5m]))`. The population it counts is "requests that the `checkout` job actually received and counted." If DNS is down, those requests never arrive, so they never appear in `http_requests_total`, so the denominator quietly shrinks and the ratio stays pinned at a beautiful 100%. The metric is not broken. It is faithfully reporting the availability *of requests that reached the server.* It has no vocabulary for requests that died on the way.

This is the structural blind spot, and it is worth saying out loud because it surprises even senior engineers: **a server-side success-rate SLI cannot, even in principle, go below the success rate of requests that reached the server.** It is mathematically incapable of seeing the failures that happen before the server. You can make it more granular, slice it by endpoint, add burn-rate alerts on top — none of that moves the measurement point. The fix is not a better white-box metric. The fix is a *different vantage point.*

### The honest role of white-box

So keep white-box monitoring and lean on it hard — for diagnosis. Here is the division of labor I recommend, and it is the spine of the rest of this post: alert on user-facing signals (which come from black-box and RUM), and *diagnose* with white-box. When the synthetic checkout probe goes red, you don't sit and stare at it; you pivot immediately to your white-box dashboards to find which service, dependency, or deploy is the cause. The user-facing signal is the smoke detector. The internal metrics are the floor plan you use to find the fire. You need both, and you need to be crisp about which job each one has.

## 4. Black-box and synthetic monitoring: probing like a user

Now we move the measurement point outward. **Black-box monitoring** means probing the system from the outside, and the active flavor of it — scripted checks that run on a schedule against your real endpoints — is what people mean by **synthetic monitoring.** Synthetic, because it is a synthetic (robot) user, not a real one; monitoring, because it runs continuously and alerts when the journey breaks.

The simplest synthetic check is a single HTTP request to your public URL, asserting status and content. The Prometheus Blackbox Exporter does exactly this, and it is the cheapest user-shaped signal you can deploy. Here is a module config that does more than "is it 200" — it follows redirects, validates the TLS chain, requires the response body to actually contain your app's content, and (crucially) warns you *before* the certificate expires:

```yaml
# blackbox.yml — a content-and-cert-aware HTTP probe
modules:
  http_checkout_page:
    prober: http
    timeout: 10s
    http:
      valid_status_codes: [200]
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      method: GET
      follow_redirects: true
      fail_if_not_ssl: true              # must be served over TLS
      # the page is only "up" if it actually contains the app shell:
      fail_if_body_not_matches_regexp:
        - "Proceed to checkout"
      # treat a maintenance/error page as a failure even if it's 200:
      fail_if_body_matches_regexp:
        - "We'll be back soon"
      tls_config:
        insecure_skip_verify: false      # verify the cert chain for real
      preferred_ip_protocol: "ip4"
```

Point Prometheus at it and you get a probe SLI with three signals that white-box can never give you: `probe_success` (did the whole thing work from outside), `probe_http_status_code` (what the public URL really returned), and the one that prevents the most embarrassing outage on earth:

```promql
# Days until the serving certificate expires — alert at < 14 days.
# This single query would have prevented every TLS-expiry outage ever.
(probe_ssl_earliest_cert_expiry - time()) / 86400
```

The content assertion (`fail_if_body_not_matches_regexp`) is the part people skip and the part that earns its keep. A status-code-only check is fooled by every failure where the server returns 200 with the wrong body — the maintenance page, the LB pointing at a stale deploy, a soft-404. Asserting that the body contains "Proceed to checkout" means the probe verifies *the right page* arrived, not just *a* page.

### Beyond a single page: scripting the critical user journey

A single-page probe catches a lot, but real reliability lives in the **critical user journey** — the multi-step flow that is the actual reason your product exists. For an e-commerce site that is log in → search → add to cart → check out. For a SaaS app it might be log in → open the dashboard → run the report. Monitoring each component (auth is up, search is up, payments is up) is *not* the same as monitoring the journey, because the journey breaks at the *seams* between components, and no component-level metric watches the seam.

![A taxonomy tree splitting green-dashboard failures into path failures like DNS, TLS, and load-balancer routing versus client-side failures like JavaScript errors and dead third-party scripts.](/imgs/blogs/monitor-the-user-not-just-the-server-6.png)

That figure is the failure taxonomy from section 2 — path failures versus client-side failures — and it is exactly the menu a journey probe is designed to walk through. A scripted browser probe drives a *real headless browser* through the whole journey and fails if any step fails, which means it exercises the seams, runs the real JavaScript, loads the real third-party scripts, and sees the real page the way a user does. Here is a Playwright journey probe for the canonical checkout flow:

```javascript
// checkout-journey.spec.js — a synthetic critical-user-journey probe.
// Runs every 60s from each region; emits a metric per step.
const { test, expect } = require('@playwright/test');

test('checkout journey is healthy end to end', async ({ page }) => {
  // Step 1: the app shell loads at all (catches DNS/TLS/CDN/LB)
  await page.goto('https://shop.example.com/', { waitUntil: 'networkidle' });
  await expect(page.locator('header.site-nav')).toBeVisible();

  // Step 2: log in (catches auth-service and session-cookie breakage)
  await page.fill('#email', process.env.SYNTH_USER);
  await page.fill('#password', process.env.SYNTH_PASS);
  await page.click('button[type="submit"]');
  await expect(page.locator('.account-menu')).toBeVisible();

  // Step 3: search returns results (catches search-service and the seam)
  await page.fill('#q', 'wireless headphones');
  await page.press('#q', 'Enter');
  await expect(page.locator('.product-card').first()).toBeVisible();

  // Step 4: add to cart and reach checkout (catches the third-party
  // payment SDK — the step that broke us in the worked example below)
  await page.locator('.product-card').first().click();
  await page.click('button.add-to-cart');
  await page.click('a.go-to-checkout');
  // assert the PAYMENT WIDGET actually rendered, not just the page:
  await expect(page.locator('iframe.payment-frame')).toBeVisible({
    timeout: 8000,
  });

  // Step 5: no uncaught JS errors fired during the whole journey
  // (wired via page.on('pageerror', ...) in the harness setup)
});
```

Read what step 4 asserts: not "the checkout page returned 200," but "the payment widget's `iframe` actually rendered." That assertion is the difference between catching the worked-example incident in two minutes and reading about it on Twitter. The page can return a perfect 200 while the payment SDK silently fails to load; only an assertion on the *rendered, JavaScript-executed result* catches that.

If you prefer a protocol-level journey without a full browser (cheaper, faster, but it won't run JavaScript), k6 does the multi-step HTTP flow with assertions and can run in CI and on a schedule:

```javascript
// checkout-journey.js — a k6 protocol-level journey probe
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = { vus: 1, iterations: 1, thresholds: {
  checks: ['rate>0.99'],                 // any failed step trips the alert
}};

export default function () {
  // login -> get session token
  const login = http.post('https://shop.example.com/api/login', JSON.stringify({
    email: __ENV.SYNTH_USER, password: __ENV.SYNTH_PASS,
  }), { headers: { 'Content-Type': 'application/json' } });
  check(login, { 'login 200': (r) => r.status === 200 });
  const token = login.json('token');
  const auth = { headers: { Authorization: `Bearer ${token}` } };

  // search returns results
  const search = http.get('https://shop.example.com/api/search?q=headphones', auth);
  check(search, {
    'search 200': (r) => r.status === 200,
    'has results': (r) => r.json('results.length') > 0,
  });

  // add to cart and start checkout
  const cart = http.post('https://shop.example.com/api/cart', JSON.stringify({
    sku: search.json('results.0.sku'), qty: 1,
  }), auth);
  check(cart, { 'cart 200': (r) => r.status === 200 });

  const checkout = http.post('https://shop.example.com/api/checkout/start', '{}', auth);
  check(checkout, {
    'checkout 200': (r) => r.status === 200,
    'payment intent created': (r) => r.json('paymentIntentId') !== '',
  });
  sleep(1);
}
```

The trade-off between the two is real and worth naming. The Playwright browser probe runs your *actual front-end JavaScript* and so catches client-side failures (JS errors, third-party scripts, broken render) that the k6 protocol probe cannot. The k6 probe is an order of magnitude cheaper to run, catches every server-and-seam failure, and is trivial to run on a tight schedule from many places. A mature shop runs both: the cheap k6 probe every 30 seconds for fast backend detection, and the expensive browser probe every 2–5 minutes for the client-side coverage. The browser probe is your only synthetic defense against the JavaScript-error class of failures — do not skip it because it is more expensive.

## 5. The multi-region probe: catching what one vantage can't

A synthetic probe is only as good as the place it runs from. A probe that runs only inside your own cloud region shares fate with your service — if your region's network has a problem, the probe goes down *with* the thing it is supposed to be watching, and you learn nothing. Worse, a probe that runs from exactly one place can never distinguish "the whole site is down" from "the site is down *from where the probe stands.*" Regional CDN failures, regional DNS resolver outages, and regional network partitions all hide from a single-vantage probe.

![A topology diagram showing the same critical journey monitored two ways, as a scripted synthetic probe from five regions and as a real-user funnel, both converging on a single page alert.](/imgs/blogs/monitor-the-user-not-just-the-server-7.png)

The fix is to run the *same* journey probe from multiple geographic locations — ideally on infrastructure that does not share fate with your serving infrastructure (a different cloud, or a managed synthetic provider's global PoPs). Now a failure has a *signature*: if all five regions go red at once, the problem is central (origin, global DNS, a global config push). If only `eu-west` goes red, the problem is regional (an EU PoP, an EU resolver, a partition) — and your white-box dashboards, which live in `us-east`, will be serenely green the whole time, which is exactly the signature of a regional outage.

Here is a multi-region synthetic config in the style of Grafana Synthetic Monitoring / a checks-as-code provider. Five probe locations, the journey on a 60-second cadence, and an alert that fires when two or more regions agree:

```yaml
# synthetics.yaml — the same journey from five geographies
check:
  name: checkout-journey
  job: critical-journey
  target: https://shop.example.com/
  frequency: 60s              # run every minute
  timeout: 15s
  probes:                     # five fate-isolated vantage points
    - us-east
    - us-west
    - eu-west
    - ap-southeast
    - sa-east
  settings:
    browser:                  # full browser so JS/third-party is exercised
      script: ./checkout-journey.spec.js
  alerting:
    # fire only when >= 2 regions agree -> kills single-vantage flakes
    sensitivity: medium
    failed_executions_to_alert: 2
```

And the Prometheus alert that turns those probe results into a page, with the multi-region logic baked into PromQL. A central outage trips one rule; a regional outage trips another and tells you *which* region in the alert itself:

```yaml
# alerting rules: synthetic journey, region-aware
groups:
  - name: synthetic-journey.alerts
    rules:
      # central failure: most regions failing the journey at once
      - alert: CheckoutJourneyDownGlobally
        expr: |
          (count(probe_success{job="critical-journey"} == 0))
            /
          (count(probe_success{job="critical-journey"}))
          >= 0.6
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Checkout journey failing from {{ $value | humanizePercentage }} of regions"
          runbook: "https://runbooks.example.com/checkout-journey"

      # regional failure: a single region red while the rest are green
      - alert: CheckoutJourneyDownInRegion
        expr: |
          probe_success{job="critical-journey"} == 0
        for: 3m
        labels: { severity: page }
        annotations:
          summary: "Checkout journey down in region {{ $labels.probe }} (others healthy)"
          runbook: "https://runbooks.example.com/regional-outage"
```

The `for: 2m`/`for: 3m` clauses matter: a single failed run from a single region is often a transient network hiccup, not an outage. Requiring the condition to hold for two or three minutes — and requiring multiple regions to agree for the global alert — is how you get a synthetic probe that pages on real outages and stays quiet on noise. (This is the same symptom-based, low-false-positive discipline that the sibling post on [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) builds out in full; if that post isn't live yet, the one-line version is: page on user-facing symptoms that persist, never on a single sample of a single cause.)

#### Worked example: a TLS expiry caught in one minute

Let me make the value of the multi-region probe concrete with the most preventable outage there is. A wildcard TLS certificate for `*.example.com` is set to expire at 00:00 UTC. Nobody renewed it; the renewal automation silently failed three weeks ago and the warning email went to a distribution list nobody reads.

At 00:00 UTC the certificate expires. Here is what each vantage point sees, minute by minute:

- **White-box (origin, `us-east`):** nothing. The origin is up, the app is serving, internal health checks pass. The `http_requests_total` counter keeps ticking on the residual traffic from already-established connections. Availability SLI: 100%. **Blind.**
- **Synthetic probe, all five regions, at 00:01 UTC:** every region's probe fails the TLS handshake simultaneously. `probe_success` drops to 0 across the board. `probe_ssl_earliest_cert_expiry - time()` is now negative. The `CheckoutJourneyDownGlobally` alert needs the condition for 2 minutes, so at **00:03 UTC it pages.** The page literally says "checkout journey failing from 100% of regions" and the runbook link is right there.
- **RUM:** new real-user sessions collapse to near zero (browsers refuse the connection), which the RUM dashboard shows as a cliff in session volume — corroborating, but the synthetic probe got there first because it doesn't depend on real traffic arriving.

Detection time: about three minutes from expiry, and it would have been *zero* if anyone had wired the `< 14 days` cert-expiry alert from section 4, which is the real lesson. The expensive version of this outage — the one where you find out from Twitter forty minutes later — is the default if you have only white-box monitoring. The cheap version is a single Blackbox Exporter probe and one PromQL rule. The arithmetic of that trade is not close.

This is also the textbook case for *why one vantage point is not enough.* If your single probe had run inside `us-east` on the same cloud as your origin, and the cert was served by an internal-only load balancer that the probe reached over a private link with a different cert, the probe could have stayed green while the public edge was dead. Probing from outside, from multiple isolated geographies, against the *public* URL the user actually hits, is the only configuration that has the user's vantage point.

## 6. Real-user monitoring: the ground truth of what users felt

Synthetic monitoring is a robot doing what a user *would* do. **Real-user monitoring (RUM)** is the measurement of what real users *actually* did — passive telemetry collected from real browsers and apps, in the field, on real networks and real devices. If synthetic is the canary, RUM is the autopsy and the census combined: it tells you the true, ground-level distribution of what your users experienced, including all the failures that only happen on a four-year-old Android phone on a 3G connection in a building with bad Wi-Fi, which no synthetic probe in a data center will ever reproduce.

RUM answers questions synthetic cannot, because it samples the *real* population:

- **The real geographic, device, and network distribution.** Your synthetic probe runs on a fast machine on a fast network. Your users do not. RUM tells you the p75 page-load time on the actual devices your actual users hold, which is the only latency number that maps to real revenue.
- **Core Web Vitals, the user-perceived performance SLIs.** LCP (Largest Contentful Paint — when the main content appears), INP (Interaction to Next Paint — how snappy clicks feel), and CLS (Cumulative Layout Shift — how much the page jumps around while loading). These are measured in the real browser and are the closest thing the industry has to a standardized "how did it *feel*" metric.
- **JavaScript errors in the wild.** The `window.onerror` and unhandled-promise-rejection events that fire in real users' browsers — the exact client-side failures your backend never sees. A spike in RUM JS-error rate is often the *first and only* signal of a bad front-end deploy.
- **The funnel of the critical journey, as real users walk it.** What fraction of real sessions that started checkout actually completed it? That number is the truest possible SLI for "can users do the thing," because it is literally measuring users doing the thing.

Here is the RUM ingestion in two parts. First, the lightweight browser snippet that measures Core Web Vitals using Google's `web-vitals` library and ships them to your collector. Note that it also wires up a global error handler — capturing JS errors is half the reason RUM earns its place.

```javascript
// rum.js — load async; measures Core Web Vitals + captures JS errors.
import { onLCP, onINP, onCLS } from 'web-vitals';

const RUM_ENDPOINT = '/rum/collect';
const ctx = {
  // dimensions you'll slice SLIs by later:
  release: window.__APP_RELEASE__,    // catches "which deploy regressed"
  region:  window.__EDGE_REGION__,    // catches "which region is slow"
  route:   location.pathname,
};

function send(metric) {
  const body = JSON.stringify({ ...ctx, name: metric.name, value: metric.value,
    rating: metric.rating, id: metric.id, ts: Date.now() });
  // sendBeacon survives page unload — critical for not losing the last metric
  navigator.sendBeacon(RUM_ENDPOINT, body);
}

onLCP(send);   // Largest Contentful Paint  — perceived load speed
onINP(send);   // Interaction to Next Paint  — perceived responsiveness
onCLS(send);   // Cumulative Layout Shift    — visual stability

// the failures your backend never sees:
window.addEventListener('error', (e) => {
  navigator.sendBeacon(RUM_ENDPOINT, JSON.stringify({ ...ctx,
    name: 'js_error', message: String(e.message),
    source: e.filename, line: e.lineno, ts: Date.now() }));
});
window.addEventListener('unhandledrejection', (e) => {
  navigator.sendBeacon(RUM_ENDPOINT, JSON.stringify({ ...ctx,
    name: 'js_unhandled_rejection', message: String(e.reason), ts: Date.now() }));
});
```

Second, the RUM data becomes SLIs the same way server data does — a ratio of good events over total, over a window — except the events are real user experiences. Here is the Core Web Vitals "good LCP" SLI and the JS-error-rate SLI as PromQL, assuming your collector exports RUM events as Prometheus metrics (a common pattern; you can equally compute these in your data warehouse):

```promql
# RUM availability-of-experience SLI: fraction of real page views with
# a "good" LCP (<= 2.5s) over the last 28 days, sliced by route.
sum(rate(rum_lcp_total{rating="good"}[28d])) by (route)
  /
sum(rate(rum_lcp_total[28d])) by (route)
```

```promql
# RUM JS-error SLI: error events per 1000 sessions, by release.
# A spike right after a deploy is the canonical bad-front-end-deploy signal.
1000 *
sum(rate(rum_js_error_total[5m])) by (release)
  /
sum(rate(rum_sessions_total[5m])) by (release)
```

The `by (release)` slice is the secret weapon. When the JS-error rate jumps, grouping by release instantly tells you *which deploy* did it, which collapses the diagnosis from "something is wrong on the front end" to "release `2026.06.20-a` is wrong on the front end, roll it back." That is the kind of signal that turns a 40-minute incident into a 4-minute one.

A word on sampling and cost, because RUM is the one vantage with a per-user price tag. Every real session ships a handful of beacons, and at high traffic that volume — and the bill — adds up. The standard discipline is to *sample*: send Core Web Vitals from, say, 10% of sessions (plenty for a stable p75 at scale), but capture JavaScript errors from 100% of sessions, because a rare error on a rare device is exactly the signal you cannot afford to sample away. Two different sampling rates for two different jobs: performance distribution wants representative sampling, error detection wants completeness. Get that split wrong — sample errors at 10% — and you will miss the bad deploy that only breaks on the one browser version held by 3% of your users, which is precisely the failure RUM exists to catch. The other cost worth naming is privacy: RUM beacons must never carry personally identifiable data in the URL or payload, so scrub query strings and never log form contents. Done right, RUM costs a rounding error and buys you the only ground truth there is; done carelessly, it is a data-protection incident waiting to happen. Sample deliberately, scrub aggressively, and the trade is overwhelmingly worth it.

### The Core Web Vitals SLI table

Treat the Core Web Vitals as first-class user-experience SLIs, with thresholds you can actually defend because they come from large-scale field research on what users perceive as fast. Here is the table I put at the top of every RUM dashboard:

| Metric | What it measures | "Good" threshold (p75) | What a regression means |
| --- | --- | --- | --- |
| LCP | When the main content paints | ≤ 2.5 s | The page feels slow to load; users bounce |
| INP | How fast clicks/taps respond | ≤ 200 ms | The page feels janky and unresponsive |
| CLS | How much layout jumps while loading | ≤ 0.1 | Users misclick; the page feels broken |
| JS error rate | Uncaught errors per session | ≤ a low baseline you set | A bad deploy; features silently broken |
| Journey completion | Real sessions that finish checkout | ≥ your SLO | The actual revenue path is broken |

The "p75" matters: you measure Core Web Vitals at the 75th percentile of real sessions, not the average, because the average is dominated by your fast users on fast devices and hides the long, painful tail. An SLI is a promise about the experience of the *typical bad case*, not the lucky median. (This is the same percentile discipline the sibling on [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) argues for at length — measure the experience, at a percentile that includes the people who are suffering.)

## 7. The three vantages, stacked: why you need all three

We now have three measurement points, and the thesis of this whole post is that none of them is optional. They are complementary, not redundant, because each one answers a question the other two cannot answer from where it sits.

![A layered stack diagram showing the user journey on top, then the real-user-monitoring layer with Core Web Vitals, then the black-box synthetic layer with multi-region probes, then the white-box internal layer with RED and USE telemetry.](/imgs/blogs/monitor-the-user-not-just-the-server-4.png)

Read the stack top to bottom:

- **RUM (closest to the user)** tells you **what real users actually felt** — the ground truth, including every client-side failure and the real-device, real-network distribution. Its weakness: it depends on real traffic. A low-traffic path or a brand-new feature with no users yet produces no RUM signal, and an outage that *prevents* users from arriving (DNS down) shows up in RUM only as an *absence* of data, which is a weak and ambiguous signal.
- **Black-box synthetic (at the edge)** tells you **whether it is up from outside, right now, on a schedule, regardless of traffic.** It covers the low-traffic and zero-traffic paths RUM can't, it gives you a clean, controlled, comparable signal (the same journey, same steps, every minute), and it catches path failures (DNS, TLS, CDN, region) cleanly because it *initiates* the request from outside. Its weakness: it is a robot, so it only tests the journeys you scripted, and it doesn't see the real device/network diversity.
- **White-box (inside)** tells you **why and where**, with the privileged internal detail that turns a red user-facing signal into a root cause in minutes. Its weakness is the entire first half of this post: it is blind to everything between the user and the server.

The combination has a property none of the three has alone: **detection from the user's side (synthetic + RUM) plus diagnosis from the inside (white-box).** Synthetic catches the outage before real users do and covers the paths with no organic traffic. RUM confirms what real users felt and catches the client-side failures synthetic might miss. White-box, once you know there's a problem, tells you what to fix. Drop any one and you have a gap: drop synthetic and you can't detect low-traffic or path failures proactively; drop RUM and you're blind to the real client-side and real-device experience; drop white-box and you can detect outages but can't diagnose them.

Here is the decision table for *which vantage to reach for*, the one I'd put in a runbook:

| You want to... | Reach for | Why |
| --- | --- | --- |
| Page on a real user-facing outage | Synthetic journey + RUM | Measured from the user's side; catches path + client failures |
| Catch an outage on a zero-traffic path | Synthetic | RUM has no signal without real users |
| Catch a regional / DNS / TLS / CDN failure | Multi-region synthetic | Initiates from outside, from many geos |
| Catch a client-side JS error | RUM + browser synthetic | Backend never sees it; needs real JS execution |
| Know the true experience distribution | RUM | Real devices, real networks, real percentiles |
| Diagnose *why* a user-facing signal is red | White-box | Privileged internal detail: which service, dep, deploy |
| Measure server-side error/latency budget burn | White-box | The internal request stream, fast and high-resolution |

The discipline that falls out of this table is simple and I'll state it as the rule of the post: **alert on user-facing SLIs (synthetic + RUM), and diagnose with white-box.** Your pages should fire on "the user can't check out," which only the user-side vantages can see. Your investigation, the moment a page fires, pivots to the inside view. Get that division right and "green dashboard, angry users" stops happening, because the things you page on are, by construction, the things the user feels.

## 8. SLIs from the user's vantage point

Let me tie this back to the currency of the whole series: the SLI, and through it the SLO and the error budget. An SLI (Service Level Indicator) is a number that measures one dimension of user-perceived reliability — a ratio of good events over total events over a window. An SLO (Service Level Objective) is the target for that number (say, 99.9% of checkouts succeed). The **error budget** is `1 − SLO` — the amount of failure you're allowed to spend before you stop shipping risk and start spending engineering on reliability. Everything in this post is, in the end, an argument about *where to measure the SLI*, because the measurement point determines whether your SLO is a promise about your server or a promise about your user.

#### Worked example: the same month, two budgets

Suppose your true user-experienced success rate is 99.0%, but 0.9% of the failures happen *before* the request reaches your server (DNS, TLS, CDN, a dead third-party script) and only 0.1% happen inside the server (a 5xx). Your server-side SLI measures `good_server / total_server`. The path failures never reached the server, so they're in neither the numerator nor the denominator of the server-side metric. The server-side SLI reports `99.9%` — it sees only the 0.1% of failures that happened inside. Your user-side SLI, measured at the edge or in the browser, reports the real `99.0%`. The gap — a full 0.9 percentage points — is precisely the failures the measurement point excluded.

Now run that gap through the error budget. With a 99.9% SLO over a 30-day month, your budget is `0.1% × 30 days × 24 × 60 = 43.2` minutes of allowed downtime per month. The server-side SLI says you're exactly at budget — spending none of it, perfectly healthy. The user-side SLI says you're at 99.0%, which is `1.0% × 43,200 = 432` minutes of "downtime-equivalent" failure per month — **ten times over budget.** Same system, same month, same users. The only thing that changed is where you stood to count, and it changed your conclusion from "we're fine" to "we've blown the budget tenfold." This is not a rounding error. It is the difference between shipping with confidence and shipping into a wall. And note which decision each number drives: at "1.0× budget" you keep shipping features; at "10× budget" the error-budget policy freezes risky launches. The measurement point doesn't just change a number on a dashboard — it changes whether you ship tomorrow.

```python
# slo_vantage.py — what measurement point does to the error budget.
SLO = 0.999                          # 99.9% objective
WINDOW_MIN = 30 * 24 * 60            # 30-day window in minutes
budget_min = (1 - SLO) * WINDOW_MIN  # allowed failure-minutes / month

# true failure decomposition (illustrative):
server_side_fail = 0.001             # 0.1% — 5xx inside the server
path_fail        = 0.009             # 0.9% — DNS/TLS/CDN/third-party, pre-server

sli_server = 1 - server_side_fail    # what white-box reports
sli_user   = 1 - (server_side_fail + path_fail)  # what the user feels

def burned_minutes(sli):             # failure-minutes implied by an SLI
    return (1 - sli) * WINDOW_MIN

print(f"Monthly error budget: {budget_min:.1f} min")          # 43.2 min
print(f"White-box SLI {sli_server:.3%} -> burned "
      f"{burned_minutes(sli_server):.1f} min "
      f"({burned_minutes(sli_server)/budget_min:.1f}x budget)")  # 1.0x — 'fine'
print(f"User-side SLI {sli_user:.3%} -> burned "
      f"{burned_minutes(sli_user):.1f} min "
      f"({burned_minutes(sli_user)/budget_min:.1f}x budget)")    # 10.0x — blown
```

The takeaway for SLI design is unambiguous: **specify the measurement point in the SLI definition, and put it as close to the user as you can defend.** "Availability = fraction of checkout requests returning 2xx, *measured at the load balancer* over a 28-day rolling window" is a defensible SLI. "Availability = fraction of checkout requests returning 2xx, *measured at the app server*" is a comforting lie, because it excludes the failures you most need to count. Best of all, where you can, define the SLI from the *journey completion rate measured in RUM* — "fraction of real sessions that started checkout and completed it" — because that ratio literally counts users doing the thing, and there is no closer vantage than that.

## 9. Measuring the critical user journey end to end

I have been circling this idea, so let me make it the centerpiece. The most common and most expensive monitoring mistake after "white-box only" is **monitoring components instead of journeys.** You have a dashboard for the auth service, a dashboard for the search service, a dashboard for the cart service, a dashboard for payments. Each one is green. Each one is, individually, doing its job. And the user still can't check out, because the failure is at a *seam* — auth returns a token in a new format that cart doesn't parse, or search returns a SKU that the cart service rejects, or the payment SDK loads but its config points at a sandbox. No component dashboard watches the seam between two components, so no component dashboard catches a seam failure.

![A before-and-after contrast showing component monitoring with every service reading up while the journey is broken, versus a single journey SLI that drops to sixty-one percent and pages in ninety seconds.](/imgs/blogs/monitor-the-user-not-just-the-server-8.png)

The fix is to add a single SLI that measures the *whole journey end to end* — and you measure it twice, once as a synthetic probe and once as a RUM funnel, exactly as figure 7 showed. The synthetic version is the Playwright journey from section 4: it drives the full flow and fails if any step fails, giving you a clean, scheduled, traffic-independent signal. The RUM version is a funnel computed from real-user events:

```promql
# RUM journey-completion SLI: fraction of real sessions that started
# checkout and reached "order confirmed", over a 28-day window.
sum(rate(rum_funnel_total{step="order_confirmed"}[28d]))
  /
sum(rate(rum_funnel_total{step="checkout_started"}[28d]))
```

When that ratio drops, you know users are abandoning the journey at a higher rate than your SLO allows — and you can break it down by `step` to find *where* in the journey they're dropping, by `region` to find *if it's regional*, and by `release` to find *which deploy* started it. A component dashboard can't do any of that, because it never had the concept of "the journey" in the first place.

Here is the comparison that decides where to invest, because journey monitoring is more work to build and you should know what it buys:

| Approach | Catches | Misses | Cost |
| --- | --- | --- | --- |
| Component metrics only | Single-service failures, 5xx, latency | Seam failures, client-side, the journey | Low — you have these already |
| Synthetic journey probe | Path + seam + client failures, low-traffic paths | Real-device/network diversity | Medium — script + run the journey |
| RUM journey funnel | What real users felt, real distribution, where they drop | Zero-traffic paths, proactive pre-launch | Medium — RUM pipeline + funnel events |
| All three together | Essentially everything in this post's catalog | Very little | Higher, but it's the complete picture |

The honest recommendation: build the synthetic journey probe *first*, because it gives you proactive, traffic-independent detection of the journey breaking, which is the highest-leverage single thing you can add to a component-only setup. Add the RUM funnel *second*, for ground truth and client-side coverage. Keep the component metrics for diagnosis. That ordering buys you the most reliability per unit of effort.

## 10. War story: the checkout that died while every dashboard smiled

Let me tell the worked-example incident in full, because it is the archetype and you will live some version of it. The numbers are illustrative but the shape is exactly real — I have run this bridge more than once.

![A timeline showing a deploy that swapped a payment vendor, the vendor JavaScript returning 404 while the backend stayed green, the synthetic checkout probe and real-user error rate both spiking, and a rollback that restored the journey in fourteen minutes.](/imgs/blogs/monitor-the-user-not-just-the-server-5.png)

**09:02.** A routine deploy ships. Among the changes: the front end now loads the payment provider's JavaScript SDK from a new URL, because the team migrated to a new version of the vendor's library. The change passed code review, passed CI, passed the staging smoke test (which used the vendor's *sandbox* SDK URL, served from a different host that was fine).

**09:03.** The new production SDK URL is wrong — a typo in the path, `v3/sdk.js` instead of `v3/checkout-sdk.js`. In production, every browser that reaches the checkout page requests the payment SDK and gets a **404 from the vendor's CDN.** The payment `iframe` never renders. The checkout button does nothing. Every user who reaches checkout is dead in the water.

**The white-box view, the whole time:** perfect. The origin served the checkout *page* (HTML, CSS, the app bundle) with clean 200s. `http_requests_total` for the `checkout` job shows normal volume and zero 5xx. The server-side availability SLI is pinned at 100%. The p99 latency is a calm 80ms. The payment SDK is a *third-party* resource loaded directly by the browser from the *vendor's* domain — it never touches your origin, so your origin has no idea it 404'd. Every dashboard is green. This is the moment the CEO email lands.

**09:04.** The signals that *do* move:

- The **synthetic browser probe** (running every minute) executes its checkout journey. At step 4 it asserts `expect(page.locator('iframe.payment-frame')).toBeVisible()`. The `iframe` is not there, because the SDK 404'd, so the assertion fails. `probe_success` for the `critical-journey` job drops to 0 across all five regions.
- The **RUM JS-error rate** spikes. Every real browser that hits checkout logs a resource-load error for the failed SDK and, when the user clicks the inert button, an unhandled exception. The `rum_js_error_total{release="2026.06.20-a"}` metric jumps from a baseline of ~2 per 1000 sessions to **38% of checkout sessions.**

**09:06.** The `CheckoutJourneyDownGlobally` alert, which requires the failure to persist for 2 minutes, fires and pages the on-call. The page reads: *"Checkout journey failing from 100% of regions."* The on-call opens the runbook, sees the journey probe is failing specifically at step 4 (payment widget), and pivots to white-box — which shows the origin is healthy, immediately ruling out a backend cause and pointing at the front end. The RUM error breakdown by `release` names the culprit deploy in one click.

**09:16.** Rollback of release `2026.06.20-a` completes. The previous front end loads the old, correct SDK URL. The synthetic probe's step-4 assertion passes again. `probe_success` returns to 1 across all regions. The RUM JS-error rate falls back to baseline. **MTTR — mean time to recovery, measured from first failure to restored service — was 14 minutes**, almost all of it the rollback itself; detection was about 4 minutes.

**The postmortem fix** had three parts, and they map exactly onto this post's thesis:

1. **The synthetic checkout probe's step-4 assertion** (`iframe.payment-frame` visible) was the thing that detected the outage. It was added two quarters earlier *because of a previous payment incident.* Without it, the only signal would have been RUM, and RUM alerting was not yet wired — so the outage would have run until a human noticed the Twitter complaints. **Action: keep and expand the synthetic journey probe.**
2. **RUM JS-error alerting** was *measuring* the spike but not *paging* on it. Action: add an alert on `rum_js_error_total` rate by release, so a bad front-end deploy pages on its own, independent of the synthetic probe.
3. **The staging smoke test used the sandbox SDK URL**, so it could never have caught a production-URL typo. Action: make the pre-production synthetic probe hit a production-shaped config, and add a canary step that runs the synthetic journey against the new release on a small slice of real traffic before full rollout.

The lesson in one sentence: **the backend metrics were flawless and the site was down, and the only things that saw the truth were the two vantages that measure from the user's side.** Every postmortem of this shape ends with "add a user-side check," because that is the only kind of check that could have caught it.

### Stress-testing the design

A good design survives the question "what if it's worse?" Let me push on this one.

**What if two incidents overlap** — a real backend 5xx surge *and* a regional CDN failure at the same time? The vantages decompose cleanly: white-box shows the 5xx surge (it sees inside), the multi-region synthetic shows *which* regions are red (regional CDN), and RUM corroborates both with the real error distribution. Three vantages give you three orthogonal signals, which is exactly what you want when the picture is confusing — each one rules something in or out.

**What if the on-call is asleep and the page is missed?** Then your detection was fine but your escalation failed, which is an alerting-and-escalation problem, not a monitoring-vantage problem — escalate to a secondary after N minutes unacked. The point of fast, user-side detection is that the *clock starts at the right moment*; what you do with the page is the next post's problem. (See the sibling on alerting and escalation.)

**What if the synthetic probe itself is flaky** and pages on a transient network blip from one region? That's why the global alert requires ≥2 regions to agree and the condition to hold for 2 minutes, and the regional alert holds for 3. A flaky probe that cries wolf is worse than no probe, because people learn to ignore it — so the probe must be *more* reliable than the thing it watches, which is why it runs on fate-isolated infrastructure with multi-sample, multi-region confirmation before it pages.

**What if there's no organic traffic on the broken path** (a B2B feature used twice a day, a new launch)? Then RUM has no signal and synthetic is the *only* thing that can catch it — which is the entire reason synthetic exists alongside RUM. The robot user generates the signal that real users aren't generating yet.

**What if the budget is already spent?** Then this monitoring is what *tells you* it's spent — honestly, from the user's side — and the [error-budget policy](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) kicks in: freeze risky launches and spend engineering on reliability until the budget recovers. Measuring from the user's vantage is what makes that budget number trustworthy enough to make a freeze decision on.

## 11. War story: the 2016 DNS outage nobody's origin could see

The most instructive real outage for this post is the October 2016 attack on Dyn, a major managed DNS provider. A massive DDoS — driven by the Mirai botnet of compromised IoT devices — overwhelmed Dyn's DNS infrastructure. For hours, large parts of the internet were unreachable for users on the US East Coast and beyond: Twitter, Spotify, Reddit, GitHub, Netflix, and many more.

Here is the part that matters for monitoring. **Not one of those companies' origin servers had a problem.** Their data centers were up. Their applications were serving. Their white-box dashboards — request rate, error rate, latency, saturation — were, for the requests that reached them, perfectly healthy. The failure was entirely in the layer *between the user and the origin*: users' browsers could not resolve the domain names to IP addresses, so the requests never started. A purely white-box monitoring posture would have shown each of those companies a calm, green morning while their users sat staring at "server not found."

What *would* have caught it, fast, is exactly the posture this post argues for: a multi-region black-box synthetic probe hitting the public URL from outside, which would have failed to resolve the domain from affected regions and paged within minutes — with the regional signature (East Coast probes red, others green for some, varying by resolver) that pointed straight at DNS. And RUM would have shown new-session volume from affected regions collapsing to zero, corroborating that real users couldn't even arrive. The vantage point is everything: the failure lived in a layer that white-box monitoring, by its nature, cannot see, and only a measurement taken from the user's side could see it.

The general lesson generalizes far past DNS. **Any failure in a layer you don't operate — DNS, the CDN, a third-party script, the user's ISP, a cloud provider's edge — is invisible to monitoring that lives inside your operated layer.** The CDN provider's own dashboards were red that day; yours were green; the user's experience matched the CDN's dashboard, not yours. You inherit the reliability of every layer between you and the user, and the only way to measure that inherited reliability is to measure from *past* all those layers — from where the user stands. (For the architecture-side treatment of how these failures cascade across layers, the system-design post on [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) is the companion; this post is about *seeing* the failure, that one is about *containing* it.)

## 12. Putting it together: the monitoring posture, end to end

Let me assemble the whole posture into something you could actually stand up next quarter, in priority order, because you won't build it all at once and the order matters.

**First, the cheapest user-side signal: a content-aware, cert-aware black-box probe.** One Blackbox Exporter module against your public URL, asserting 200 *and* that the body contains your real content, *and* alerting on cert expiry at < 14 days. This is an afternoon of work and it eliminates the entire TLS-expiry and total-outage class of "green but down" failures. If you do nothing else from this post, do this.

**Second, the multi-region synthetic journey probe.** Script the critical user journey (login → search → checkout) as a Playwright browser probe, run it every minute from five fate-isolated regions, assert on the *rendered, JavaScript-executed* result (the payment `iframe`, not just the status code), and page when ≥2 regions agree for 2 minutes. This is the single highest-leverage detection mechanism for the failures this whole post catalogs, because it walks the seams and runs the real front end.

**Third, RUM with Core Web Vitals and JS-error capture.** Ship the `web-vitals` snippet, capture `onerror` and unhandled rejections, slice everything by `release` and `region`, and turn the JS-error rate and the journey-completion funnel into SLIs. Alert on the JS-error-rate spike by release. This gives you ground truth and the client-side coverage even your browser synthetic might miss, plus the real-device/network distribution no synthetic can reproduce.

**Fourth, keep your white-box for diagnosis.** Your RED/USE metrics, your recording rules, your per-service dashboards — none of that goes away. It stops being the thing you *page* on and becomes the thing you *diagnose* with the moment a user-side page fires.

The result, measured honestly: before this posture, "green dashboard, angry users" was an incident every quarter, detected by customers, with detection times measured in tens of minutes and MTTRs dominated by the time it took a human to even believe something was wrong. After: the user-facing outages that used to be invisible now page in 2–4 minutes from the user's side, the on-call pivots to white-box for a root cause in minutes, and the embarrassing "we found out from Twitter" incidents go to roughly zero. You don't get this by buying a tool. You get it by *moving the measurement point to where the user stands* — which is the one idea this entire post is about.

## How to reach for this (and when not to)

Every practice has a cost, and a field manual that only tells you to add monitoring is selling you toil. So here is the honest "when not to."

**Do** run a content-and-cert-aware black-box probe against every user-facing service. This one is nearly free and prevents the most embarrassing outages there are. There is no service important enough to skip it and no service so trivial that a cert expiry won't still page someone at 3am.

**Do** build a synthetic journey probe for your *critical* journeys — the two or three flows that are the actual reason your product exists (sign-up, the core action, checkout). Walk the journey, assert on the rendered result, run it multi-region.

**Do** deploy RUM if you have a real front end with real users, especially a JavaScript-heavy SPA where client-side failures are common and invisible to the backend.

**Don't** write a synthetic probe for every endpoint and every minor flow. Synthetic probes are code you maintain; a hundred flaky probes that nobody trusts are worse than five solid ones. Probe the critical journeys; let RUM and white-box cover the long tail.

**Don't** alert on raw Core Web Vitals from a single sample or a thin slice — they're noisy at low volume. Aggregate to p75 over a window, alert on sustained regressions, and slice by release so the alert is actionable. A CLS alert that fires on one unlucky session will get muted, and a muted alert is worse than no alert.

**Don't** put RUM on a low-traffic internal tool where ten people use it twice a day. There's no statistical signal, the privacy/overhead cost isn't worth it, and a single synthetic probe tells you everything RUM would, more cheaply.

**Don't** chase a fifth nine on a path users can't perceive. If your synthetic and RUM both say users are happy at 99.9%, the engineering to reach 99.999% on a server-side metric is spending real money to move a number the user will never feel. Measure from the user; if the user can't tell, stop. (This is the [error-budget](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) discipline: spend reliability effort where the budget — measured from the user — says it hurts.)

**Don't** treat synthetic as a load test or RUM as your only observability. Synthetic verifies the journey works; it does not verify the journey works *under load* (that's a load test) and it does not replace tracing for diagnosis. Each tool has one job; don't ask it to do three.

## Key takeaways

- **The only reliability that matters is what the user experiences.** Your server's opinion of its own health is a diagnosis tool, not the truth. Measure as close to the user as you can defend.
- **White-box monitoring is necessary but not sufficient.** It tells you *why* and *where* once you know something is broken, but it is structurally blind to every failure between the user and the server: DNS, TLS, CDN, LB routing, client-side JS, third-party scripts, regional outages.
- **The measurement point determines the number.** A server-side success-rate SLI cannot, even in principle, see failures that never reached the server. Specify the measurement point in the SLI and push it toward the user; the same month can read 99.9% at the server and 99.0% at the user — ten times over budget.
- **Black-box synthetic monitoring probes from outside, like a user**, catching outages before real users do and covering low-traffic paths with no organic signal. Assert on *content* and the *rendered result*, not just the status code, and run multi-region so regional and DNS failures have a signature.
- **Real-user monitoring is the ground truth of what users felt** — Core Web Vitals (LCP, INP, CLS) at p75, JS errors in the wild, the real device/network distribution, and the journey funnel. Slice by release to name the bad deploy in one click.
- **You need all three vantages.** Synthetic + RUM to *detect* from the user's side; white-box to *diagnose* from the inside. Alert on user-facing signals, diagnose with internal ones.
- **Monitor the journey, not just the components.** Failures live at the seams between green components. A single end-to-end journey SLI — as a synthetic probe and a RUM funnel — catches what no component dashboard can.
- **The cert-expiry alert and the content-aware probe are nearly free** and prevent the most embarrassing class of outages. Do them first, today.

## Further reading

- *Site Reliability Engineering* (the Google "SRE Book"), chapters on Monitoring Distributed Systems and Service Level Objectives — the canonical source on the four golden signals, white-box vs black-box, and SLIs as ratios.
- *The Site Reliability Workbook*, chapters on Implementing SLOs and Alerting on SLOs — the multi-window, multi-burn-rate alerting that pairs with the user-side SLIs here.
- Prometheus Blackbox Exporter documentation — the reference for content-aware, cert-aware HTTP/TLS/DNS probes.
- The `web-vitals` library and the Core Web Vitals field-data guidance (web.dev) — measuring LCP, INP, and CLS from real browsers at p75.
- OpenTelemetry documentation on browser instrumentation and the synthetic-monitoring guidance from your provider of choice (Grafana Synthetic Monitoring, Checkly, or similar) — checks-as-code for multi-region journeys.
- Within this series: the [intro to the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) for the whole reliability loop; [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) for the measurement-from-the-user discipline; the sibling posts on dashboards that tell the truth and alerting that doesn't cry wolf for turning these signals into trustworthy dashboards and pages; and [the error budget](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) for spending reliability effort where the user-measured budget says it hurts.
- For the architecture side of these failures, the system-design posts on [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) and [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — this post is about *seeing* the failure from the user's side; those are about *designing* the system not to have it.

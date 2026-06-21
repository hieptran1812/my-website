---
title: "Deploying Safely: Progressive Delivery and the Art of a Small Blast Radius"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make the riskiest thing you do every day safe — roll out a new version to 1 percent of traffic, watch the SLIs, and roll back automatically the instant they regress, with canary specs, feature-flag configs, and worked rollouts you can copy."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "progressive-delivery",
    "canary",
    "blue-green",
    "feature-flags",
    "rollback",
    "argo-rollouts",
    "error-budget",
    "deployment",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/deploying-safely-progressive-delivery-1.png"
---

It is 09:41 on a Wednesday — the safest-looking hour of the week. No traffic spike, no marketing push, no full moon. An engineer merges a one-line change to the checkout service: a slightly stricter validation on a field that, it turns out, was always permissive in production. CI is green. The deploy pipeline does its thing. Forty seconds later, every checkout in the fleet starts returning a 400 on a request shape that ten percent of real users send. The error rate goes from 0.1% to 9% across the entire service in the time it takes to refill a coffee. The burn-rate alert fires. Three engineers join the bridge. Somebody asks the question that decides how bad this gets: *can we get back to the last good version, and how fast?*

Here is the uncomfortable truth that sits underneath that morning: **the deploy is the single biggest cause of incidents you will ever face.** Not the cosmic-ray bit flip, not the once-a-decade data-center fire — the change you yourself shipped, on purpose, ten minutes ago. Study after study of postmortems lands in the same place: the large majority of outages are *self-inflicted by a change*. Which is oddly good news. It means the highest-leverage reliability investment available to you is not exotic. It is making the act of deploying safe. If most outages come from changes, then bounding the damage a change can do is the most reliability you can buy per engineer-hour.

That bounding has a name — **blast radius** — and a thesis you can put on a sticky note: *never expose a new version to 100% of traffic at once.* Roll it out progressively. Watch the service level indicators (SLIs) — the user-facing measurements of error rate and latency that tell you whether users are in pain. And roll back automatically the instant they regress, before a human even reads the alert. That whole discipline is called **progressive delivery**, and it is the subject of this post. The figure below is the mental model we will build out piece by piece: take a change that might carry a bug, and bound its damage in two dimensions — *traffic* (only 1% sees it) and *time* (bake long enough to catch slow problems) — so a regression becomes a 1%-of-users, four-minute footnote instead of a fleet-wide outage.

![Diagram showing a risky change bounded in traffic and in time so its blast radius shrinks to one percent of users for four minutes](/imgs/blogs/deploying-safely-progressive-delivery-1.png)

By the end you will be able to: explain *why* deploys cause incidents and what bounding the blast radius actually buys you; choose between rolling, blue-green, canary, and feature-flag strategies from first principles; write an Argo Rollouts or Flagger canary spec with an automated analysis template that compares the canary against a live baseline; wire automated rollback to an SLO regression so a bad deploy aborts itself; reason about bake time and why "deploy and go straight to 100%" defeats the entire point; and handle the one deploy you cannot simply roll back — the database migration — with expand/contract. This is the *running-it* layer. We are not designing the architecture (the [system-design treatment of reliability and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) covers that); we are operating the daily, dangerous act of shipping.

## 1. Why deploys cause incidents — and what blast radius means

Start with the causal chain, because the whole strategy falls out of it. A bug does not appear in a running, untouched system. Software that worked yesterday and was not touched works today too; entropy in production is real but slow. The fast, violent failures — the ones that page you — overwhelmingly arrive *attached to a change*: a code deploy, a config push, a schema migration, a feature flag flip, an infrastructure-as-code apply. The change is the carrier. The bug rides in on it.

So if a deploy is the vehicle for incidents, the question that determines severity is: **when the bad version goes live, how many users does it reach, and for how long?** That product — *users affected × duration* — is the blast radius. It is the only thing that separates "a blip nobody noticed" from "a Sev1 that made the news." And critically, blast radius is something you *control* with how you deploy. The bug is the same bug either way; what differs is exposure.

Define the term plainly the first time, since the whole series leans on it: **blast radius** is the scope of harm a failure can cause — what fraction of traffic, users, or systems is affected when something breaks. A deploy that flips everyone to the new version at once has, by construction, a blast radius of 100% the moment the bug activates. A deploy that sends 1% of traffic to the new version has a blast radius of 1% until you choose to widen it. Same code, same bug, two orders of magnitude difference in damage.

There are two independent dimensions you can bound, and the strategies in this post each attack one or both:

- **Bound it in traffic.** Don't let the new version serve everyone. Serve 1%, then 5%, then 25%. The canary and feature-flag strategies live here. A bug in the new version can only hurt the slice of traffic routed to it.
- **Bound it in time.** Don't commit to the new version until it has run long enough to reveal its problems. Some bugs are instant (a 400 on a bad validation); some are slow (a memory leak that OOMs after twenty minutes, a cache that fills and then thrashes). Bounding in time — **bake time** — catches the slow ones before you widen exposure.

The instant-cutover deploy bounds neither. It is the default in a lot of shops precisely because it is the simplest thing that works on a good day, and most days are good days. That is the trap: a strategy that is fine 95% of the time and catastrophic the other 5% is not a fine strategy, because the 5% is where all your error budget goes. The figure below puts the two worlds side by side — ship to 100% and a human discovers the bug after it has hit everyone, versus ship to 1% and an automated analysis catches the regression while it is still a rounding error.

![Side by side comparison of deploying to all traffic at once versus a progressive canary that limits exposure to one percent before catching the regression](/imgs/blogs/deploying-safely-progressive-delivery-2.png)

This ties directly to the [error budget — the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). An error budget is the amount of unreliability you are *allowed* to spend in a window — if your service level objective (SLO) is 99.9% availability over 30 days, your budget is 0.1% of requests, which we will price out shortly. A deploy that hits 100% of traffic with a 9% error rate burns budget at a furious pace. A canary that hits 1% with the same 9% error rate burns it at one-hundredth the rate, *and* aborts before most of even that 1% is served. Progressive delivery is, at root, a way to make deploys cheap in budget terms — so you can deploy often (which is its own reliability win, because small frequent changes are easier to reason about than big rare ones) without going broke.

#### Worked example: the cost of a 100% blast radius

Take a service doing 10,000 requests per second with a 99.9% availability SLO over a 30-day month. The budget is $0.1\%$ of requests. Over 30 days at 10,000 rps that is about $10{,}000 \times 86{,}400 \times 30 = 2.59 \times 10^{10}$ requests, so the budget is roughly $2.59 \times 10^{7}$ — about 26 million failed requests allowed for the month.

Now ship a bug that fails 9% of requests, instantly, to 100% of traffic. You are now failing $10{,}000 \times 0.09 = 900$ requests per second. Your *entire monthly budget* of ~26 million failures is consumed in $2.59 \times 10^{7} / 900 \approx 28{,}800$ seconds — **eight hours**. But you will not be down for eight hours; you will page and roll back. Say a human notices in 6 minutes and rolls back in 4 — ten minutes of full exposure. That is $900 \times 600 = 540{,}000$ failed requests, about 2% of your *whole month's* budget gone in one deploy.

Run the same bug through a 1% canary that auto-aborts after 4 minutes of bad data. Now you are failing $10{,}000 \times 0.01 \times 0.09 = 9$ requests per second, for 240 seconds: $9 \times 240 = 2{,}160$ failed requests. That is 0.008% of the monthly budget — **250× cheaper** for the identical bug. The bug did not get smaller. The blast radius did.

That ratio — two to three orders of magnitude less budget burned for the same defect — is the entire economic argument for progressive delivery. Everything else in this post is mechanism.

It helps to see what those budgets buy you in plain wall-clock terms, because "99.9%" stays vague until you translate it into minutes you are allowed to be down. The nines table is worth memorizing:

| SLO (availability) | Budget per 30 days | Budget per year |
|---|---|---|
| 99% (two nines) | 7.2 hours | ~3.65 days |
| 99.9% (three nines) | 43.2 minutes | ~8.76 hours |
| 99.95% | 21.6 minutes | ~4.38 hours |
| 99.99% (four nines) | 4.32 minutes | ~52.6 minutes |
| 99.999% (five nines) | 25.9 seconds | ~5.26 minutes |

Read that as a budget you *spend*, because that is exactly what an incident does. At 99.99% you have 4.32 minutes for the *entire month*. The §1 worked example's instant-cutover deploy burned ten minutes of full-rate errors in one event — at a 99.99% SLO, that single bad deploy did not just dent the budget, it *blew the whole month's budget more than twice over*, in one deploy, at 09:41 on a quiet Wednesday. This is why a four-nines service literally cannot afford instant cutovers: a *single* unprotected bad deploy a month exceeds the entire allowance. Progressive delivery is not a nicety at high nines; it is the only arithmetic that closes. And the higher your SLO, the more the canary's 100×-to-470× budget savings is the difference between meeting it and missing it. This is the same budget that gates *whether you are allowed to deploy at all* — when the budget is already spent, the right move per the [error-budget policy](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) is to freeze risky deploys and spend the next window's effort on reliability, not features.

## 2. The four strategies, and the one dimension each bounds

There are four deployment strategies worth knowing, and the cleanest way to hold them in your head is by *which dimension of blast radius each one bounds*. Two of them (rolling, blue-green) are really about *how you replace running instances*; two of them (canary, feature flags) are about *how much traffic or how many users see the new behavior*. They compose — real systems use a canary rollout that itself ships code behind feature flags.

### Rolling deploy — bound in-flight risk, not traffic exposure

A **rolling deploy** replaces the fleet in batches: take down a few old instances, bring up the same number of new ones, wait for them to pass health checks, repeat until the whole fleet is new. Kubernetes does this by default with a `Deployment` (`RollingUpdate` with `maxSurge`/`maxUnavailable`).

What does it bound? It bounds *in-flight risk* — at no point are you fully down, because you always keep a quorum serving. But here is the subtlety that trips people up: **a rolling deploy does not bound traffic exposure.** Once a batch of new instances is up and behind the load balancer, it serves *normal, randomly-routed traffic* — the same as any other instance. By the time the roll is halfway done, half your traffic is hitting the new (possibly broken) version. By the time it finishes, 100% is. Rolling limits *unavailability during the swap*; it does not limit *who gets the bug*. It is a deployment mechanism, not a blast-radius control. Treat it as the safe default *substrate* you run a canary on top of, not as your blast-radius strategy by itself.

### Blue-green — instant flip, full blast radius at the flip

**Blue-green** stands up the entire new version (call it *green*) alongside the entire running old version (*blue*), in parallel. You health-check green privately, then flip the router so 100% of traffic goes to green at once. Blue stays warm and idle.

The win is **rollback speed**. If green misbehaves, you flip the router back to blue — which is still running, still warm — in seconds. No re-deploy, no waiting for old images to pull and pods to schedule. The cost is twofold: at the moment of the flip your blast radius is 100% (every user moves to green simultaneously, so a green bug hits everyone instantly), and you are paying for *double the infrastructure* during the overlap. Blue-green is the right tool when rollback speed dominates and instant exposure is acceptable — stateful services where you can't easily split traffic, or where the flip is gated by enough pre-prod confidence that a regression is unlikely but you want a fast escape hatch if it happens.

### Canary — bound traffic exposure, the gold standard

A **canary** (named for the canary in the coal mine — a small sacrificial sample that warns you before the danger reaches everyone) routes a small percentage of live traffic to the new version while everyone else stays on the old one. You start at 1%, watch the new version's SLIs, and only widen — 5%, 25%, 50%, 100% — if it stays healthy at each step. If it regresses, you halt and roll back, and only the canary slice ever saw the bug.

Canary bounds *traffic exposure* directly and tunably, and it is the gold standard for blast-radius control because it is the only strategy where the bad version is *both* limited to a small audience *and* observed against a live control before being trusted. It costs a little — you run a small overlapping set of canary instances during the rollout — and it requires enough traffic that 1% is statistically meaningful (1% of 10 rps is 0.1 rps, which tells you nothing in five minutes; 1% of 10,000 rps is plenty).

### Feature flags — decouple deploy from release, the finest grain

A **feature flag** (also called a feature toggle) is a runtime conditional: the new code ships to production *dark*, wrapped in `if flag_enabled("new_checkout"):`, and does nothing until you turn the flag on. This **decouples deploy from release** — the most important idea in the whole post. *Deploying* is moving code to production; *releasing* is exposing behavior to users. Flags separate them: you can deploy the binary to 100% of servers (low-risk — the new path is dormant) and then *release* the feature progressively to 1% of users, 5%, 25%, by flipping a flag value, with no redeploy. And you can **kill it instantly** — flip the flag off — which is the finest-grained, fastest rollback there is, faster even than blue-green because there is no traffic to move. This is the **kill switch** that the [mitigate-first, diagnose-later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) discipline reaches for first during an incident: you do not need to know *why* the feature broke to turn it off.

The cost of flags is *flag debt* — every flag is a branch in your code and your test matrix, and stale flags that nobody removed become a swamp. Discipline: every flag has an owner and an expiry.

The matrix below is the one to screenshot. It compares all four across the three axes that actually decide a choice — blast radius, rollback speed, and extra cost.

![Matrix comparing rolling, blue-green, canary, and feature flags across blast radius, rollback speed, and extra infrastructure cost](/imgs/blogs/deploying-safely-progressive-delivery-3.png)

| Strategy | Blast radius | Rollback speed | Extra cost | Best when |
|---|---|---|---|---|
| **Rolling** | All traffic eventually (no traffic bound) | Re-roll old image, ~20 min | None beyond one extra batch | Safe default substrate; low-stakes services |
| **Blue-green** | 100% at the flip | Flip router back, ~30 sec | Double infra during overlap | Rollback speed matters; hard-to-split traffic |
| **Canary** | 1% → 5% → 25%, tunable | Auto-abort, < 5 min | Small overlapping canary set | High traffic; you want analysis before trust |
| **Feature flags** | Per-user %, finest grain | Kill flag, < 10 sec | Flag debt to manage | Decouple deploy from release; instant kill switch |

The right answer in a mature setup is usually *not one of these* — it is a **canary rollout of a binary that itself gates risky behavior behind feature flags**, running on top of a rolling deploy substrate. Each layer bounds a different dimension. We will assemble exactly that.

If you want a quick way to *choose* under pressure, the decision tree below routes you from a few constraints to a default. Can you toggle the behavior in code at runtime — that is, is the risk in a *feature* you can gate? Then flags give you the finest grain. Do you have enough traffic that a 1% slice is statistically meaningful within a sane bake window? Then a canary with analysis is the gold standard. Can you afford to run two full copies during an overlap and is rollback *speed* what you care about most? Then blue-green's flip earns its cost. And if none of those pull hard, rolling is the safe default substrate you build on. None of these are exclusive — the tree picks your *primary* control, and you layer the others underneath.

![Decision tree that routes from constraints like runtime toggles, traffic volume, and infrastructure budget to a primary deploy strategy](/imgs/blogs/deploying-safely-progressive-delivery-7.png)

The point of the tree is not to be dogmatic — it is to make the choice *explicit and defensible* rather than "whatever the last service did." A team that can articulate "we chose blue-green here because traffic is too sticky for a representative 1% canary and a payment outage costs more than the doubled infra" is making a reliability decision; a team that copy-pasted a `Deployment` and hopes for the best is rolling dice with a 100% blast radius.

## 3. Canary, concretely: an Argo Rollouts spec with automated analysis

Talk is cheap; here is the artifact. [Argo Rollouts](https://argoproj.github.io/rollouts/) is a Kubernetes controller that replaces the stock `Deployment` with a `Rollout` resource that understands canary stages, pauses, and — crucially — *analysis*. [Flagger](https://flagger.app/) is the close cousin that does the same on top of a service mesh. I will show Argo Rollouts because the spec reads cleanly, then a Flagger equivalent.

The `Rollout` below ships the checkout service through 1% → 5% → 25% → 50% → 100%, baking at each step, and runs an `AnalysisTemplate` that compares the canary's error rate and latency against the baseline. If the analysis fails, the rollout halts and rolls back on its own.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout
spec:
  replicas: 20
  strategy:
    canary:
      # Run the analysis continuously against the canary as it widens.
      analysis:
        templates:
          - templateName: checkout-canary-slo
        startingStep: 1          # begin analysis after the first weight step
        args:
          - name: service
            value: checkout
      steps:
        - setWeight: 1           # 1% of traffic to the new version
        - pause: { duration: 10m }   # BAKE: soak 10 minutes at 1%
        - setWeight: 5
        - pause: { duration: 15m }
        - setWeight: 25
        - pause: { duration: 30m }   # longer bake as exposure grows
        - setWeight: 50
        - pause: { duration: 30m }
        - setWeight: 100         # full promotion only after clean bakes
      canaryService: checkout-canary
      stableService: checkout-stable
      trafficRouting:
        nginx:
          stableIngress: checkout
  selector:
    matchLabels: { app: checkout }
  template:
    metadata:
      labels: { app: checkout }
    spec:
      containers:
        - name: checkout
          image: registry.internal/checkout:v2.4.0
          ports: [{ containerPort: 8080 }]
```

The `AnalysisTemplate` is where the reliability reasoning lives. It defines the SLIs to compare and the thresholds that constitute a regression. This one watches two of the [four golden signals](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — error rate and latency — measured *on the canary specifically*, via Prometheus, against an absolute success floor and a baseline-relative latency ceiling.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: checkout-canary-slo
spec:
  args:
    - name: service
  metrics:
    - name: success-rate
      interval: 1m              # re-evaluate every minute during the bake
      count: 5                  # need 5 readings before a verdict
      successCondition: result >= 0.99   # canary must serve >= 99% success
      failureLimit: 1           # one breach => fail the analysis => rollback
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(
              http_requests_total{app="{{args.service}}",
                                  role="canary", code!~"5.."}[2m]))
            /
            sum(rate(
              http_requests_total{app="{{args.service}}",
                                  role="canary"}[2m]))
    - name: latency-p99-vs-baseline
      interval: 1m
      count: 5
      # canary p99 must not exceed baseline p99 by more than 20%
      successCondition: result <= 1.20
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            histogram_quantile(0.99, sum(rate(
              http_request_duration_seconds_bucket{app="{{args.service}}",
                                                   role="canary"}[2m]))
              by (le))
            /
            histogram_quantile(0.99, sum(rate(
              http_request_duration_seconds_bucket{app="{{args.service}}",
                                                   role="baseline"}[2m]))
              by (le))
```

Two design decisions in there are worth dwelling on, because they are the difference between a canary that protects you and one that lulls you.

First, **`count: 5` with `interval: 1m`** means the analysis needs five one-minute readings before it renders a verdict — it will not promote on a single lucky sample, and it will not panic on a single noisy one. `failureLimit: 1` says a single *breach* fails the analysis; tune this against your noise floor (a flappy dependency might warrant `failureLimit: 2`). The tension is real: too sensitive and you abort good deploys on noise (a false abort costs you a rollback and a re-attempt — annoying but safe); too lax and you promote a regression (a false promote costs you an outage — dangerous). Bias toward sensitive; a wrongly-aborted deploy is cheap, a wrongly-promoted one is not.

Second — and this is the one people skip — **the latency metric is *relative to baseline*, not absolute.** `result <= 1.20` means "the canary's p99 is at most 20% worse than the *currently running* old version's p99." Why relative? Because absolute thresholds lie under load. If a traffic spike pushes everyone's p99 from 200ms to 400ms, an absolute `p99 < 250ms` check would fail your canary for a problem the canary did not cause. Comparing canary-against-baseline cancels out the ambient conditions — both versions see the same spike — so the analysis isolates *the effect of the new version*. This is the core idea behind **automated canary analysis (ACA)**, popularized by Netflix's [Kayenta](https://github.com/spinnaker/kayenta): you are not asking "is the canary good in absolute terms," you are asking "is the canary *different from* a control running right now, in a way that matters." That control is why you run `baseline` pods (fresh instances of the *old* version) rather than comparing against the long-running stable fleet — fresh-vs-fresh removes confounders like cache warmth and JVM JIT state.

The figure shows that control loop: deploy the canary, scrape both canary and baseline SLIs every minute, score the canary against the baseline, and branch — widen if it passes, halt and roll back if it fails.

![Graph of the automated canary analysis loop where canary and baseline metrics are scraped, scored against each other, and the rollout either widens or aborts](/imgs/blogs/deploying-safely-progressive-delivery-5.png)

A third decision hides in `startingStep` and the `count`/`interval` pair, and it is about *statistical confidence*. A canary at 1% of traffic is a small sample, and small samples are noisy. If your baseline error rate is 0.1% and you observe 0.3% on the canary over thirty seconds, is the new version three times worse, or did the canary just happen to catch a few of the same transient failures the baseline would have? At low request counts you genuinely cannot tell. This is why the analysis accumulates several readings (`count: 5`) before deciding, and why teams with serious canary practices use a proper statistical test — the **Mann-Whitney U test** is what Kayenta uses — to ask "are the canary's metric samples drawn from a *worse* distribution than the baseline's, at a chosen significance?" rather than a naive threshold comparison. The practical upshot for a smaller team without that machinery: make sure 1% of your traffic produces *enough events per evaluation window* that the ratio is stable. A handy floor is that you want at least a few hundred requests per window on the canary before you trust an error-rate comparison; below that, widen the first step (start at 5% instead of 1%) or lengthen the window, because a canary you cannot read is a canary that promotes on luck.

There is also a subtle ordering question: do you analyze *before* or *after* each weight increase? The safe pattern, encoded above, is **analyze continuously and pause at each weight** — you raise the weight, then *soak and analyze at that weight* before raising again. The anti-pattern is to raise the weight and immediately raise it again on a timer with no gate in between (a "timed rollout" with no analysis), which is just a slow instant-cutover wearing a canary costume. The pause-and-analyze gate is the load-bearing part; the weights are just the granularity.

For teams on a service mesh, **Flagger** expresses the same thing more tersely. Here is the heart of a Flagger `Canary`, which drives a stock Kubernetes `Deployment` through the same progressive analysis:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: checkout
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: checkout
  analysis:
    interval: 1m
    threshold: 1          # # of failed checks before rollback
    maxWeight: 50         # promote to 100% only after passing at 50%
    stepWeight: 5         # advance 5% per healthy interval
    metrics:
      - name: request-success-rate
        thresholdRange: { min: 99 }    # >= 99% success
        interval: 1m
      - name: request-duration
        thresholdRange: { max: 500 }   # p99 <= 500ms
        interval: 1m
    webhooks:
      - name: load-test           # optional: drive traffic so 1% is meaningful
        url: http://flagger-loadtester.test/
        metadata:
          cmd: "hey -z 1m -q 20 -c 2 http://checkout-canary/healthz"
```

Same skeleton, same reliability story: advance only on healthy intervals, roll back on a breach, never trust a single sample.

## 4. Bake time — why "deploy then go to 100%" defeats the canary

Here is the most common way to have a canary and get no protection from it: ship the canary, glance at the dashboard, see green, and immediately promote to 100%. You did all the work of building a canary and then *threw away its only real advantage* — time.

The reason is that **bugs have latencies of their own.** Some defects announce themselves on the first request (a 400, a null-pointer, a 500). Those a one-minute canary catches. But a whole class of the *worst* production bugs are **slow-burn** — they need the new version to run for minutes or longer before they bite:

- A **memory leak** that grows a few megabytes per minute and OOM-kills the pod after twenty.
- A **cache that fills** — the new version's cache key changed, so the cache starts cold, the hit rate climbs as it warms, and the latency cliff only appears once it is full and starts evicting hot entries.
- A **connection-pool exhaustion** that only manifests once enough long-lived connections accumulate.
- A **slow query** that is fine at 1% traffic but tips a database index into a bad plan once concurrency rises.
- A **GC pause** or a **thread-pool starvation** that needs sustained load to provoke.

If you promote to 100% in ninety seconds, none of these have had time to appear *in the canary*. So they appear *in production, at full scale, after you have committed.* You converted a canary into a slow, expensive instant-cutover. **Bake time** — deliberately *pausing* the rollout at each weight for long enough to let slow-burn problems surface — is what makes a canary a canary.

How long is long enough? It depends on the slowest failure mode you are guarding against. A rule of thumb worth internalizing: *the bake at a given stage should be at least as long as the time-to-symptom of your slowest realistic bug.* If your service has historically had memory leaks that OOM in fifteen minutes, no stage should bake less than fifteen. Many teams bake longer at *higher* weights (more traffic exercises more code paths and provokes saturation faster) but require a *minimum wall-clock soak* at every stage regardless. The timeline figure shows the staged bake — short at the bottom weights where you mostly want to catch instant failures cheaply, longer as exposure grows.

![Timeline of canary stages from one percent to one hundred percent, each baked for an increasing duration to surface slow-burn problems](/imgs/blogs/deploying-safely-progressive-delivery-4.png)

A useful way to set bake durations is to look at your *own* incident history and ask: of the deploy-caused incidents we have had, what was the distribution of time-from-deploy-to-symptom? If most of your deploy bugs announced themselves in under five minutes, a ten-minute first bake catches the bulk of them cheaply. But if you have a fat tail — a handful of incidents where the symptom appeared 30 or 60 minutes after the deploy (the classic leak or cache-fill profile) — then at least one stage of your rollout must bake longer than that tail, or you are systematically blind to your most expensive failure class. The bake schedule should be *derived from your failure data*, not copied from a blog post (including this one — the 10/15/30/30 numbers are illustrative defaults, not a law).

There is a subtlety about *which* stage to bake longest. Naively you might bake longest at 1%, to catch everything before widening. But many saturation problems are *load-dependent* — a connection-pool exhaustion or a database-plan flip needs real concurrency to provoke, and 1% of traffic may never reach the threshold that triggers it. So the standard practice is to bake *meaningfully* at every stage (enough to catch the instant and the moderately-slow failures) but reserve the *longest* soak for a middle stage like 25–50% where you have both enough traffic to provoke load-dependent bugs *and* still a bounded blast radius if one appears. The 1% stage is your cheap first filter; the 25% stage is your saturation gauntlet; 100% is the victory lap you only take after both passed.

There is a real tension here, and it is worth naming honestly. Bake time slows you down. A five-stage rollout that bakes 10/15/30/30 minutes plus promotion takes the better part of two hours. If you deploy fifty times a day, that is a lot of wall clock — and worse, if you have an *urgent* fix (you are mid-incident and the deploy *is* the mitigation), you do not want to wait two hours to roll out the patch that stops the bleeding. So mature pipelines support two modes: the **default progressive rollout** with full bakes for routine changes, and an **expedited path** for incident mitigations that shortens or skips bakes *deliberately and with a human's eyes on it* — because during an incident the calculus flips (the current state is already bad, so getting the fix out fast can beat soaking it). The mistake is not "sometimes we go fast." The mistake is *always* going fast, by default, with no analysis, and calling it a canary.

#### Worked example: the leak the bake caught

A new version of the recommendations service has a leak: it forgets to close a gRPC stream, leaking ~8 MB/minute per pod. At the pod's 1 GiB limit, a pod that starts at 300 MiB will OOM after about $(1024 - 300)/8 \approx 90$ minutes. The first canary stage is 5% of traffic, baked 30 minutes. After 30 minutes the canary pods are at ~540 MiB and climbing — *no errors yet*, but the analysis also watches a memory-headroom SLI: `container_memory_working_set_bytes / spec_memory_limit > 0.7` for 5 minutes triggers a `failureLimit` breach. At minute 30 the canary pods cross 70% headroom and the analysis fails. The rollout halts at 5% and rolls back. Total user impact: zero errors (the leak never reached OOM at 5%), and the slow defect was caught before it ever touched 95% of traffic.

Now run the same version through "deploy and immediately 100%." All 20 pods leak in lockstep; at minute 90 they begin OOM-killing *together*, restart cold (latency spike from cold caches), leak again, and you are in a rolling crash-loop across the whole fleet at peak hours. The bug was identical. The bake — and a memory SLI in the analysis — is the entire difference between a non-event and a Sev2.

## 5. Automated rollback on SLO regression — taking the human off the critical path

The slowest, most variable component of any incident response is the human. A great on-call engineer might notice a fired alert in two minutes; a tired one at 3am might take fifteen. The triage — "is this real? what changed? do I roll back?" — adds more. Every minute of that human latency is a minute the bad version keeps serving. The single biggest reduction in blast radius you can make, after limiting traffic, is to **take the human off the critical path of the rollback decision** for the cases where the decision is mechanical.

And it *is* mechanical, in the canary case. The question "is the canary's error rate or latency significantly worse than the baseline's?" is exactly the kind of thing a machine evaluates faster, more consistently, and at 3am better than a person. So you encode it: the analysis (from §3) runs continuously, and a **failed analysis automatically halts the rollout and rolls back** — no page-acknowledge, no human, no debate. The human gets a *notification after the fact* ("rollout of checkout v2.4.0 auto-aborted at the 5% stage: success-rate 0.94 < 0.99") which is an entirely different, calmer event than a live outage.

This connects straight to the error budget. You can express the rollback rule in **burn-rate** terms: a deploy that burns error budget faster than some multiple of the sustainable rate should abort itself. Recall **burn rate** is how fast you are consuming budget relative to the steady rate that would exactly exhaust it over the window: $\text{burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}$. For a 99.9% SLO ($1 - \text{SLO} = 0.001$), an observed 9% error rate is a burn rate of $0.09 / 0.001 = 90\times$ — you are spending budget ninety times faster than sustainable, which would exhaust a 30-day budget in $30 / 90 \approx 8$ hours. A canary observing that on its slice should not wait for a human; the math already said "abort."

Here is an Argo Rollouts analysis that aborts on a *burn-rate* breach, expressed directly in PromQL. This is the artifact that ties progressive delivery to the budget:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: checkout-burnrate-guard
spec:
  args:
    - name: service
  metrics:
    - name: budget-burn-rate
      interval: 30s
      count: 6
      # SLO = 99.9% => allowed error fraction = 0.001.
      # Abort if the canary burns budget faster than 14x sustainable.
      successCondition: result < 14
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            (
              sum(rate(http_requests_total{app="{{args.service}}",
                                           role="canary", code=~"5.."}[2m]))
              /
              sum(rate(http_requests_total{app="{{args.service}}",
                                           role="canary"}[2m]))
            ) / 0.001
```

A `result` of `14` means "burning 14× the sustainable rate," which would exhaust a month's budget in ~2 days — a threshold borrowed from the multi-window burn-rate alerting practice in the Google SRE Workbook. If the canary crosses it, the rollout aborts. You have made the *deploy* honor the *budget* automatically.

Stress-test the automation, because un-stress-tested automation is how you turn a small incident into a big one. *What if the canary and the baseline are both bad?* Then your code is probably not the cause — a shared dependency is down, or a poison input is hitting both versions — and rolling back will not help; it will just thrash while the real problem persists. The fix is to make the analysis *differential*: the failure condition should be "canary is significantly *worse than* baseline," not "canary is bad in absolute terms." When both are bad, the differential is ~1 (they are equally bad), the analysis does *not* fail the canary, and instead you should fire a *separate* symptom-based alert that pages a human — because this is exactly the kind of judgment call automation must not make alone. *What if Prometheus itself is down or stale during the rollout?* Then the analysis has no data — and a well-built analysis treats *missing data as inconclusive*, neither promoting nor aborting, and pauses for a human, because promoting blindly is the worst outcome and aborting on a monitoring hiccup is a needless rollback. *What if the rollback re-triggers the same bug* (the bug is in a config both versions read, or in a migration)? Then rollback is not a safe mitigation, and you must have caught that at design time — which is why the next guardrail matters.

There is a guardrail you must add, or automated rollback becomes its own outage: **the rollback target must be known-good and the rollback path must itself be tested.** Auto-rolling-back to a previous version that *also* fails (because the real problem is a sick dependency, not your code) just thrashes. So: (a) the analysis should distinguish "canary is worse than baseline" (your change is bad → roll back) from "canary and baseline are *both* bad" (something external is wrong → rolling back won't help, *do* page a human). (b) Cap the auto-rollback attempts; if a rollback does not restore health, escalate rather than loop. (c) Never auto-rollback across a one-way door — which brings us to migrations in §7.

#### Worked example: the deploy the budget aborted

The checkout team ships v2.4.0 at 14:00. It has the 400-on-valid-input bug — 8% of requests fail. The canary spec routes 1% of 8,000 rps = 80 rps to it. At 8% error that is ~6.4 failed rps *on the canary*. The burn-rate guard evaluates every 30s; observed canary error fraction ≈ 0.08, divided by 0.001 = burn rate 80×, far past the 14 threshold. After 6 readings (~3 minutes of confirmation to dodge noise), the analysis fails. Argo halts and rolls back at the 1% stage. Total exposure: ~3 minutes of 1% traffic at 8% error = $80 \times 0.08 \times 180 \approx 1{,}150$ failed requests. No page fired as an outage; the team got an after-the-fact "auto-aborted" notice and looked at the diff over coffee. Compare the §1 worked example's 540,000 failures from the same bug at 100%: **the automated canary turned a Sev2 into a non-event, 470× cheaper.**

## 6. Blue-green and the 30-second rollback

Canary is the gold standard for *catching* a bad deploy, but its rollback is "stop widening and shift the canary slice back" — fast, but it assumes you caught it during the rollout. What about the deploy that looked fine through every canary stage, promoted to 100%, and *then* revealed a problem an hour later (a slow leak the bake missed, a once-an-hour batch job that hits a new code path)? Now you are fully on the new version and need out, fast. This is where **blue-green's instant flip-back** earns its double-infrastructure cost.

The contrast is stark and worth a figure. With a rolling deploy, reversing means re-rolling the *old* image back across the fleet — pull the old image, schedule old pods, wait for health checks, batch by batch — which on a 20-pod service is easily 15–20 minutes. With blue-green, *blue is still running, still warm, still holding connections* — you flip the router back to it and you are healthy in the time it takes a config change to propagate, which is seconds.

![Side by side of reversing a rolling deploy in about twenty minutes versus flipping a blue-green router back to the warm old version in about thirty seconds](/imgs/blogs/deploying-safely-progressive-delivery-6.png)

#### Worked example: 30 seconds versus 20 minutes

The payments service runs blue-green. At 16:10 the team flips traffic to green (v3.0.0). At 16:12 the error rate on green jumps to 4% — a config that pointed at a stale secrets path, invisible in staging. The on-call runs the flip-back: `kubectl patch service payments -p '{"spec":{"selector":{"version":"blue"}}}'`. Routing converges in ~25 seconds; the SLI is clean by 16:13. **Total outage: ~70 seconds, exposure window ~3 minutes at 4% error.**

Had payments been on a stock rolling deploy with no warm old version, the on-call would have triggered a rollback that re-rolls v2.9.0 across 16 pods at `maxSurge: 25%`: roughly four batches, each waiting ~3–4 minutes for image pull plus readiness, ≈ 18 minutes of degraded service. Same bug, 15× longer outage, ~15× more budget burned. The 30 seconds is what you bought with the doubled infrastructure during the overlap window. Whether that is worth it is a budget question — for a payments path where every minute of downtime is real revenue and trust, it usually is; for an internal dashboard, it usually is not.

A second cost of blue-green that is easy to forget is **state**. The instant flip is clean for stateless request handlers, but the moment your service holds in-flight state — open WebSocket connections, in-progress multi-step transactions, sticky sessions — a flip is not free. Connections pinned to blue do not magically move to green; you either drain them gracefully (let blue finish serving its in-flight work while green takes new connections) or you sever them and force reconnects. Draining takes time and complicates the "instant" story; severing causes a visible blip for connected users. And shared mutable state — a cache, a database — is *shared between blue and green*, so a green version that writes a poisoned cache entry or a bad row has already harmed blue's users too, and flipping back does not un-poison the cache. Blue-green isolates the *compute*, not the *data*. This is why the cleanest blue-green deploys are of stateless services in front of a shared, backward-compatible data layer — which loops right back to expand/contract in §7.

The honest caveat on blue-green: the *flip itself* is a 100%-blast-radius event. Every user moves to green at once, so if green is broken in a way your pre-flip health checks didn't catch, *everyone* gets it for the seconds-to-minutes until you flip back. Blue-green optimizes *recovery time*, not *exposure*. That is exactly why the strongest setups *canary the flip*: route 1% to green, analyze, *then* flip the rest — combining canary's exposure bound with blue-green's instant recovery. The strategies compose; you rarely pick just one.

## 7. The deploy you cannot roll back: database migrations and backward compatibility

Everything above assumes rollback is *possible* — that flipping back to the old version restores a working system. There is one category where that assumption breaks, and it is the source of the scariest deploys in any company: **schema migrations.** If your deploy includes a database change — drop a column, rename it, add a NOT NULL constraint, change a type — then rolling back the *code* to the old version may leave it talking to a *new schema it doesn't understand*. You have walked through a **one-way door**: the deploy that cannot be reversed because the data layer changed underneath it.

The discipline that keeps the door two-way is **expand/contract** (also called *parallel-change*). The rule: **never change a schema in a way that is incompatible with the currently-running code.** You decompose every schema change into a sequence of individually-reversible steps, where at every step the database is compatible with *both* the old code and the new code. That keeps rollback a live option throughout the rollout. The figure shows the phases for adding-and-using a new column:

![Stack of expand and contract migration phases that keep the schema compatible with both old and new code so rollback stays possible](/imgs/blogs/deploying-safely-progressive-delivery-8.png)

Walk a concrete change — say you are splitting a `full_name` column into `first_name` and `last_name`:

1. **Expand.** Add the new columns `first_name`, `last_name` as *nullable*. The old code ignores them; the new code can use them. The schema is compatible with both. This step is reversible (drop the columns).
2. **Deploy code that writes both.** The new version writes `full_name` *and* the split columns on every update (dual-write), and backfills nothing yet. Both old and new code can run against this schema. Reversible — roll back to the old code and `full_name` is still authoritative.
3. **Backfill.** A background job populates `first_name`/`last_name` for existing rows, in *batches* (never one giant `UPDATE` that locks the table — see the [database series on zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) if your engine needs online-DDL tooling like `gh-ost` or `pt-online-schema-change`). Still reversible; you are only filling new columns.
4. **Deploy code that reads new only.** Now the new columns are fully populated, switch reads to them. *This* is the step that, once promoted, makes a rollback to "writes only full_name" lossy — so promote it only after the read path is proven, and keep dual-writes a while longer as insurance.
5. **Contract.** Once you are confident and no rollback target needs the old column, *drop* `full_name`. This is the one-way step, done *last*, deliberately, often days or weeks after the code change, well outside the deploy that introduced the feature.

The reliability principle is: **separate the irreversible step (contract) from the deploy, in time, and make every intermediate state mutually compatible.** Expand and contract are different deploys, often different days. A deploy that *both* adds a feature *and* drops a column is a one-way door welded shut — if the feature is bad you cannot roll back without losing the dropped data. Pull them apart.

This is the operational counterpart to the design-time reasoning in [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) and the [microservices treatment of deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags): the architecture decides what's *possible*; expand/contract is how you *operate* a schema change so progressive delivery still has an escape hatch.

| Migration step | Reversible? | Compatible with old code? | Compatible with new code? |
|---|---|---|---|
| Expand (add nullable cols) | Yes (drop cols) | Yes | Yes |
| Dual-write | Yes (roll back code) | Yes | Yes |
| Backfill | Yes | Yes | Yes |
| Read-new-only | Lossy after promote | No (old wrote full_name) | Yes |
| Contract (drop old col) | **No — one-way door** | No | Yes |

## 8. What to watch during a rollout — the canary's golden signals plus the business

You cannot roll back on a regression you are not measuring. So the analysis must watch the *right* signals, and "the right signals" is not "every metric you have" — it is a small set that reflects user pain plus a couple that catch the slow killers. The canon is the **four golden signals**, measured *on the canary versus the baseline*:

- **Latency** — and specifically *tail* latency (p99, p99.9), because the mean hides the users who are suffering. Compare canary p99 to baseline p99 as a ratio (§3).
- **Errors** — the success rate (or its complement). The most direct proxy for "is the new version serving requests correctly."
- **Traffic** — request rate. Mostly a sanity check during a canary (you control the split), but a *drop* in canary traffic can mean the new version is crashing before it can be routed to, or failing health checks.
- **Saturation** — how full the resource tanks are: memory headroom, CPU, connection pools, queue depth. This is the golden signal that catches the slow-burn leaks the others miss until too late. *Always* include a saturation SLI in a canary analysis.

Beyond the technical four, watch **business metrics** where you can, because some regressions are *correct from the server's point of view and catastrophic for the user*. A deploy that returns HTTP 200 with an empty cart, or silently drops a payment confirmation, sails past an error-rate check. A canary that watches *checkout-completion rate* or *add-to-cart rate* against baseline catches these. The principle: **the canary should watch the closest proxy to user value you can measure, not just the closest proxy to server health.** That ties back to the series' foundation — [monitor the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server).

And watch **error-budget burn** explicitly, as §5 showed — it is the single number that says "this deploy is too expensive, stop." Here is the PromQL you would graph on a rollout dashboard, side by side canary and baseline:

```promql
# Canary success rate (last 2m), compared visually to baseline below it.
sum(rate(http_requests_total{app="checkout",role="canary",code!~"5.."}[2m]))
  / sum(rate(http_requests_total{app="checkout",role="canary"}[2m]))

# Baseline success rate — the control.
sum(rate(http_requests_total{app="checkout",role="baseline",code!~"5.."}[2m]))
  / sum(rate(http_requests_total{app="checkout",role="baseline"}[2m]))

# Canary p99 latency in seconds.
histogram_quantile(0.99, sum(rate(
  http_request_duration_seconds_bucket{app="checkout",role="canary"}[2m]))
  by (le))

# Canary saturation: memory working set as a fraction of the limit.
max(container_memory_working_set_bytes{pod=~"checkout-canary.*"})
  / max(kube_pod_container_resource_limits{pod=~"checkout-canary.*",
                                           resource="memory"})
```

The dashboard you want during a rollout has the canary and baseline on the *same panel* for each signal, so the divergence is visually obvious, plus a single big "burn rate" stat and the rollout's current weight. If you have to hunt across five tabs to know whether the canary is healthy, the dashboard is wrong — see [dashboards that tell the truth](/blog/software-development/site-reliability-engineering/dashboards-that-tell-the-truth).

| Signal | What it catches | How to compare | Always in a canary? |
|---|---|---|---|
| Latency (p99) | Slowness, degraded UX | Canary/baseline ratio | Yes |
| Errors (success rate) | Broken responses | Canary vs baseline, absolute floor | Yes |
| Traffic | Crash-before-route, health-check fails | Canary rate is non-zero and stable | Sanity check |
| Saturation (mem/CPU/pool) | Slow-burn leaks, exhaustion | Canary headroom threshold | Yes — the slow-killer guard |
| Business (completion rate) | Correct-but-wrong responses | Canary vs baseline funnel | Where measurable |

## 9. Feature flags as the finest grain — and the kill switch

Canary bounds *which traffic* sees the new version. Feature flags bound *which behavior* is exposed and *to whom* — at the granularity of an individual user, a cohort, a region, or a percentage — and they do it *after* the code is already deployed everywhere. This is the finest-grained control there is, and it is also the fastest mitigation, which is why it deserves its own treatment.

A progressive feature-flag rollout config (this is [OpenFeature](https://openfeature.dev/)-style, mirrored by LaunchDarkly/Flagsmith/Unleash) reads like a series of widening gates:

```json
{
  "flag": "new-checkout-flow",
  "defaultVariant": "off",
  "variants": { "on": true, "off": false },
  "targeting": {
    "rules": [
      {
        "description": "internal users first - dogfood",
        "if": { "attribute": "email", "op": "ends_with", "value": "@us.example" },
        "serve": "on"
      },
      {
        "description": "progressive percentage rollout, sticky per user",
        "rollout": {
          "bucketBy": "userId",
          "stages": [
            { "variant": "on", "weight": 1 },
            { "variant": "on", "weight": 5 },
            { "variant": "on", "weight": 25 },
            { "variant": "on", "weight": 100 }
          ]
        }
      }
    ]
  },
  "killSwitch": {
    "if": { "attribute": "globalIncident", "op": "eq", "value": true },
    "serve": "off"
  }
}
```

Three things make this powerful. **First, `bucketBy: userId` makes the rollout *sticky*** — a given user consistently gets the same variant, so they don't flicker between old and new on every request (which would be a jarring, bug-report-generating experience). **Second, you dogfood to internal users before any external percentage** — your own staff hit the new path first and report problems before customers see them. **Third, the `killSwitch`** — a single global flag that, when flipped, serves `off` to everyone *regardless of the rollout stage*, instantly, with no redeploy. That is the mitigate-first kill switch: during an incident you do not diagnose, you flip the switch, the new behavior vanishes, and the SLI recovers. Then you diagnose calmly. (For the why-this-works reasoning, see [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later).)

The decoupling is the deep idea. With flags, a deploy stops being a scary event because *the deploy doesn't change behavior* — the binary ships dark. The risk moves to the *release* (the flag flip), which you control independently, can do gradually, can target precisely, and can reverse in milliseconds. You can deploy on a Friday afternoon (the binary is inert) and release on Monday morning (flip the flag, watch, widen). This is the cleanest possible separation of "ship the code" from "expose the feature," and it is why feature flags are the finest grain in the progressive-delivery toolkit.

The cost, again, is **flag debt.** Every flag is a permanent branch until removed, doubling the relevant code paths and the test surface. A flag that was supposed to be temporary and is still there a year later is a landmine — it represents a code path that may have rotted. Discipline: every flag has an owner, a creation date, an expiry, and a ticket to remove it once it is fully rolled out (100% for two weeks → delete the flag and the dead branch). Treat stale-flag cleanup as scheduled toil reduction, exactly like the [automation work that gets you off the pager](/blog/software-development/site-reliability-engineering/automating-yourself-out-of-the-pager).

## 10. The deploy-as-the-suspect rule — and how it changes incident response

There is a heuristic every seasoned on-call carries, and it is the operational payoff of everything in §1: **when something breaks, the last deploy is the prime suspect.** Because most incidents trace to a change, the fastest path to mitigation during an incident is usually *not* "diagnose the bug" but "reverse the most recent change and see if the symptom clears." This is **mitigate first, diagnose later** applied to deploys, and it is why a *fast, safe rollback* is the single most valuable capability a team can have.

The reasoning is a Bayesian one. Given that an incident is happening, the prior probability that a recent change caused it is high — far higher than any single other cause. So the expected-value-maximizing first action is to test that hypothesis cheaply. Rolling back is cheap (seconds to minutes if you built the tooling in this post) and *diagnostic* — if the symptom clears, you have both mitigated *and* confirmed the deploy was the culprit, in one move. This is **bisection** applied to time: the deploy is the boundary between "working" and "broken," so reversing it is the first cut. (The [debugging series on binary-search-your-bug-with-bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) generalizes this to `git bisect` across many commits; in production the most recent deploy is the first commit you bisect on.)

For this to be the *reliable* first move, the rollback has to be:

- **Fast** — seconds (flag kill switch, blue-green flip) to a few minutes (re-roll), not the 40 minutes a manual redeploy from scratch takes.
- **Safe** — rolling back must not itself cause harm. This is *exactly* why expand/contract (§7) matters: if a migration made the rollback lossy, "just roll back" is no longer a safe first move, and you have lost your best mitigation. Protecting the rollback path is protecting your incident response.
- **Obvious** — the on-call should know, without thinking, *how* to roll back this service (it is in the [runbook that survives 3am](/blog/software-development/site-reliability-engineering/runbooks-that-survive-3am)) and *what* the last deploy was (the deploy markers should be annotated on every dashboard so the correlation between "deploy at 09:41" and "errors started at 09:41" is visible at a glance).

The single cheapest investment that makes the deploy-as-suspect rule *work* is **deploy annotations on your dashboards**. Every deploy should emit an event — a Grafana annotation, a vertical line on every relevant panel — tagged with the version and timestamp. Then when the error rate ramps at 09:41, the on-call sees a deploy marker *at 09:41* and the hypothesis writes itself. Without markers, the team spends the first ten minutes of the incident *reconstructing* what changed — paging the release channel, scrolling deploy logs, asking "did anyone ship anything?" — which is ten minutes of pure latency added to the outage. The marker collapses that to a glance. It is a one-line CI step (`curl` the Grafana annotations API on every deploy) that pays for itself the first time it shortens an incident, and it is the difference between "the deploy is the prime suspect" being a *principle* and being an *actionable first step*.

The stress test: *what if rolling back doesn't fix it?* Then you have learned something valuable cheaply — the incident is *not* the deploy — and you move to the next hypothesis (a dependency, a traffic shift, a data issue) with the change ruled out. *What if two deploys went out close together?* Bisect: roll back the most recent, then the next, narrowing as you would in `git bisect`. *What if the deploy is hours old and the incident is now?* Lower the prior — a slow-burn problem (§4) might have a deploy from an hour ago as its cause, so still check it, but widen the suspect list to include time-triggered things (a cron, a cache expiry, a certificate). The rule is "deploy is the *prime* suspect," not "deploy is the *only* suspect."

## 11. War story: the canary that paid for itself, and the migration that didn't

Two stories — one a win that the discipline bought, one a self-inflicted wound from skipping it. Both are composites of widely-documented real patterns; the specific numbers are illustrative but the failure modes are textbook.

**The win — Netflix and automated canary analysis.** Netflix popularized automated canary analysis at a scale where manual review is impossible — thousands of deploys a day across hundreds of microservices. Their open-source tool **Kayenta** (part of Spinnaker) does exactly what §3 describes: for each deploy it spins up a *canary* and a *baseline* (a fresh instance of the *current* version, not the long-running fleet — to cancel out cache-warmth and JIT confounders), routes a slice of traffic to each, and statistically compares dozens of metrics. A deploy is promoted only if the canary is statistically indistinguishable-or-better than the baseline. The reliability payoff is that bad deploys are caught *by a machine, in minutes, on a tiny slice* — the human is informed, not on the critical path. The principle that made it work is the one this whole post turns on: *compare the new version against a live control, not against an absolute threshold, on a bounded slice, and let the comparison drive an automatic decision.* That is the gold standard, and it is reproducible with Argo Rollouts or Flagger on a far smaller budget.

**The self-inflicted wound — the irreversible migration.** A team needed to change a column's type — `user_id` from a 32-bit integer to a 64-bit one, because they were running out of IDs. They wrote one migration and one deploy: the migration altered the column type *and* the code started writing 64-bit IDs in the same release. It passed CI. It passed a canary — for forty minutes, because no existing row had an ID large enough to break the old readers yet. They promoted to 100%. Two hours later a downstream service that *still expected 32-bit IDs* (it had not been updated, an integration nobody remembered) started truncating IDs and corrupting joins. The fix should have been "roll back" — but rolling back the code did not help, because the *schema* was already 64-bit and the corrupted rows were already written. They had welded a one-way door: the deploy that changed the schema and the code together. Recovery took hours of manual data repair instead of a 30-second flip. The lesson is exactly §7: **expand and contract are separate deploys; never ship the irreversible schema change in the same release as the code that depends on it, and never assume a canary that passes has exercised the data conditions that will break you at scale.** A canary catches what it has *seen*; a slow-burn or scale-triggered condition it has not seen yet, it cannot catch — which is also why bake time and a saturation SLI matter.

The throughline of both: progressive delivery is not a magic shield. It is a set of *bounds* — on traffic, on time, on reversibility — and it protects you exactly to the extent that the bug manifests *within those bounds before you widen them.* Design the bounds (canary weights, bake times, expand/contract) so that the failure modes you actually have show up inside them.

## 12. How to reach for this — and when not to

Progressive delivery has real cost — pipeline complexity, longer rollout wall-clock, infra overlap, flag debt, the discipline of writing analysis templates. Spend it where blast radius is expensive; skip it where it isn't.

**Reach for the full apparatus (canary + analysis + auto-rollback + flags) when:**

- The service is **high-traffic and user-facing** — 1% is statistically meaningful and a 100% blast radius is genuinely costly (revenue, trust, an SLO with a tight budget).
- You **deploy often** — the more deploys, the more the per-deploy safety compounds, and the more you need it automated because humans can't review every rollout.
- The service is **stateless or near-stateless at the request level** so traffic-splitting is clean.

**Reach for blue-green (not canary) when:** rollback *speed* dominates over exposure *minimization*, traffic is hard to split meaningfully (low volume, or sticky sessions that make a 1% slice unrepresentative), and you can afford the doubled infra during the overlap. Payments and other "every second of downtime is real money" paths often justify it.

**Reach for feature flags regardless** of the deploy strategy — they are cheap to add per-feature, they decouple deploy from release, and the kill switch is the best mitigation tool you will have during an incident. The one caution is flag-debt discipline.

**Do NOT bother (or actively avoid) when:**

- It is a **low-traffic internal tool or batch job.** A 1% canary on a service doing 5 rps gives you 0.05 rps on the canary — you will not get a statistically meaningful read in any reasonable bake window, so the canary is theater. Use a simple rolling deploy and a manual rollback runbook. Don't engineer a fifth nine where there are no users to notice the fourth.
- The change is a **pure config flip that is already behind a flag** — the flag *is* your progressive control; you don't need a second canary on top.
- You **cannot meaningfully measure the SLI** the analysis would gate on. A canary with no good success/latency/saturation signal is a canary that promotes blindly — worse than honest, because it *looks* safe. Get the observability first (see [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain)), then canary on it.
- You are **mid-incident and the deploy is the mitigation.** Do not make an urgent fix wait two hours of bakes; use the expedited path with a human watching. The default-progressive rule is for *routine* changes.

And the meta-rule: **never auto-remediate a failure mode you do not understand.** Automated rollback is safe *because* the rollback target is known-good and the decision (canary worse than baseline) is mechanical and correct. Auto-rolling-back into a thrash loop, or auto-promoting because the analysis was too lax, are both ways to turn automation into an incident. Automate the *mechanical, well-understood* decisions; keep humans on the *judgment* ones. This is the same line the [self-healing systems and their traps](/blog/software-development/site-reliability-engineering/self-healing-systems-and-their-traps) post draws.

## Key takeaways

- **The deploy is the #1 cause of incidents.** Most outages are self-inflicted by a change, so making deploys safe is the highest-leverage reliability investment you can make.
- **Never expose a new version to 100% at once.** Bound the blast radius of every deploy in *traffic* (1% → 5% → 25%) and in *time* (bake each stage), so a bug hits a sliver of users briefly instead of everyone for as long as a human takes to react.
- **Pick the strategy by which dimension it bounds.** Rolling bounds in-flight risk only; blue-green bounds *recovery time* (30-second flip, double infra); canary bounds *traffic exposure* (the gold standard); feature flags bound *exposed behavior per user* (finest grain, instant kill switch). They compose — canary a flagged binary on a rolling substrate.
- **Compare the canary against a live baseline, not an absolute threshold.** A canary/baseline ratio cancels out ambient load, so the analysis isolates *the effect of your change* — the core idea behind Kayenta-style automated canary analysis.
- **Bake time is the point of a canary.** Promoting to 100% the instant a canary looks green throws away its only real advantage. Slow-burn bugs (leaks, cache fills, pool exhaustion) need minutes to appear; soak long enough to catch them, and always include a saturation SLI.
- **Automate rollback on SLO/burn-rate regression.** A canary burning budget past ~14× sustainable should abort itself — no human on the critical path — and notify after the fact. Take the slowest, most variable component (the human) off the mechanical decision.
- **Protect the rollback path.** A fast, safe rollback is your best first mitigation (the deploy is the prime suspect). Expand/contract migrations keep rollback possible by never welding a one-way door — separate the irreversible step (drop column) from the deploy, in time.
- **Watch the golden signals canary-vs-baseline plus business metrics and budget burn.** A correct-but-wrong response (200 with an empty cart) passes an error check; a completion-rate SLI catches it.
- **Right-size it.** Full canary apparatus for high-traffic user-facing services that deploy often; rolling + a runbook for low-traffic internal tools; flags everywhere; never auto-remediate what you don't understand.

## Further reading

- *Site Reliability Engineering* (Google), the chapters on release engineering and on canarying — the canonical treatment of bounding change risk.
- *The Site Reliability Workbook* (Google), "Canarying Releases" and the multi-window multi-burn-rate alerting chapter — the source of the burn-rate thresholds used here.
- [Argo Rollouts documentation](https://argoproj.github.io/rollouts/) — `Rollout`, `AnalysisTemplate`, and traffic-routing specs for Kubernetes canary and blue-green.
- [Flagger documentation](https://flagger.app/) — progressive delivery on a service mesh with built-in canary analysis.
- [Kayenta / Spinnaker automated canary analysis](https://github.com/spinnaker/kayenta) — the statistical canary-vs-baseline model at scale.
- [OpenFeature](https://openfeature.dev/) — a vendor-neutral feature-flagging standard for the progressive-release configs in §9.
- Within this series: [reliability is a feature — the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), [the error budget — the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability), [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later), and [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure). When the chaos-engineering post ships, pair this with *chaos engineering — breaking on purpose*, which game-days the rollback paths this post relies on.
- Out of series: [microservices deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) and [CI/CD and independent deployability](/blog/software-development/microservices/ci-cd-and-independent-deployability) for the build-and-pipeline layer; [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) for the architecture that makes a bad deploy survivable.

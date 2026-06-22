---
title: "Progressive Delivery in the Pipeline: Wiring Canary and Blue-Green Into Kubernetes"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A plain rolling update gates on readiness, not on whether the new version is actually good. This is the toolchain field guide to wiring metric-gated canary and instant-rollback blue-green into your Kubernetes pipeline with Argo Rollouts, AnalysisTemplates, and traffic routing."
tags:
  [
    "ci-cd",
    "devops",
    "kubernetes",
    "progressive-delivery",
    "canary",
    "blue-green",
    "argo-rollouts",
    "flagger",
    "traffic-routing",
    "feature-flags",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-1.png"
---

The pipeline was clean. The image was built once in CI, scanned, signed, pushed to the registry, and promoted to prod without a rebuild. The Kubernetes `Deployment` rolled out in ninety seconds, every new pod passed its readiness probe, `kubectl rollout status` reported `deployment "checkout" successfully rolled out`, and the deploy dashboard went green. Twelve minutes later, the on-call engineer's phone went off. The new version of the checkout service had a subtle bug: under real production traffic, with real cart shapes that the staging fixtures never produced, one code path threw on about 4% of requests. The readiness probe never caught it, because the readiness probe hit `/healthz`, and `/healthz` returned `200 OK` the whole time. The pods were *ready*. The version was *broken*. And because a rolling update shifts **all** traffic onto the new pods the moment they pass readiness, 100% of customers hit that 4% error rate for the full twelve minutes it took a human to wake up, look at the dashboard, and type `kubectl rollout undo`.

That gap — between "the pods are ready" and "the new version is actually good" — is the entire subject of this post. A readiness probe answers a narrow question: *can this pod accept traffic without immediately falling over?* It says nothing about whether the new code returns correct answers, whether its latency regressed, whether it leaks memory, whether it throws on the long-tail inputs that only production carries. A plain rolling update has exactly one gate, and that gate is readiness. There is no metric-based gate, no way to say "shift 5% of traffic, watch the error rate for five minutes, and back out automatically if it climbs." That capability is what **progressive delivery** adds: a *traffic* layer that exposes the new version to a controlled slice of users, and an *analysis* layer that watches real metrics and decides — automatically — whether to widen the rollout or abort it. Figure 1 shows the shape of the loop we are going to build.

![Diagram of the gated progressive delivery loop showing CI building the image, CD updating the Rollout, a small traffic weight shift to the canary, an AnalysisTemplate querying Prometheus to compare baseline against canary, and the rollout either promoting to full traffic or auto-aborting with a rollback](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-1.png)

Here is the important boundary for this post, and I want to state it up front so we do not wander. The *why* of progressive delivery — why bounding the blast radius of a deploy matters, why you gate a release on a Service Level Indicator, how an error budget tells you how much risk you can afford, why the deploy is the single most common cause of incidents — is reliability theory, and it is covered thoroughly in the SRE companion post, [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery), and the post on [the error budget as the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability). I am going to lean on that theory and not re-derive it. **This post is the toolchain layer: how you actually *implement* metric-gated canary and instant-rollback blue-green inside a Kubernetes pipeline.** The `Rollout` resource. The canary steps. The `AnalysisTemplate`. The traffic routing. The pipeline step that triggers the rollout and waits for it. By the end you will be able to take a service that ships with a naive rolling update and turn it into one whose bad versions are caught and rolled back automatically before most users ever see them. This is the *deploy* step of the commit-to-build-to-test-to-package-to-deploy-to-operate spine — engineered so that the *operate* step does not start with a page. For where it all fits together, the series intro is the [CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model), and the layer directly beneath this one is [rolling updates and zero-downtime deploys](/blog/software-development/ci-cd/rolling-updates-and-zero-downtime-deploys).

## 1. Why a plain rolling update is not enough

Let us be precise about what a Kubernetes rolling update does, because it does a *great* job at the thing it is designed for and we are going to keep using it as the primitive underneath everything else.

When you `kubectl apply` a new image tag to a `Deployment`, the controller creates a new `ReplicaSet` and begins a careful handoff: it brings up new pods, waits for each to pass its readiness probe, adds the ready ones to the `Service` endpoints, and scales the old `ReplicaSet` down — all while honoring `maxSurge` (how many extra pods it may spin up) and `maxUnavailable` (how many it may take down at once) so the total serving capacity never dips below the floor you set. Done right, this is genuinely zero-downtime: not a single request is dropped. That mechanism — the surge guard, the readiness gate, graceful shutdown — is the subject of the sibling post on [rolling updates](/blog/software-development/ci-cd/rolling-updates-and-zero-downtime-deploys), and you should get it right *first*, because progressive delivery sits on top of it. A canary that drops requests on every pod handoff is a canary that drops requests; it just drops fewer of them.

So what is missing? Two things, and they are the two things that matter most for a high-risk change.

**First: the only gate is readiness, and readiness is not quality.** The rolling update controller widens traffic the instant a pod is ready. "Ready" means "the readiness probe returned success." A readiness probe is a liveness-adjacent health check; it typically verifies the process is up, the port is open, maybe that a database connection can be established. It does not — *cannot*, without you building it — verify that the new version returns correct answers, that its p99 latency did not double, that it does not throw on a category of inputs that staging never exercised. The error in our opening story passed readiness for the entire incident, because the pod was perfectly capable of *accepting* requests; it just answered 4% of them wrong. Readiness is a necessary gate. It is nowhere near a sufficient one.

**Second: there is no metric-based abort.** Once the rolling update commits to shifting traffic, there is no built-in mechanism that says "watch the error rate of the new pods, and if it diverges from the old pods by more than X, stop and reverse." The closest native tool is `progressDeadlineSeconds`, which fails the rollout if pods never *become* ready — but our pods became ready fine. To abort a rolling update on a metric regression, a human has to be watching a dashboard, recognize the problem, and run `kubectl rollout undo`. That is a manual MTTR loop measured in minutes-to-tens-of-minutes, exactly when the new version is serving errors to everyone.

Put plainly: a rolling update answers "are the new pods healthy enough to receive traffic?" Progressive delivery answers "is the new *version* good enough to keep, judged by the same metrics that define our SLOs?" The second question is the one that prevents incidents. Figure 2 puts the two side by side.

![Before and after comparison showing a plain rolling update gating only on readiness and shifting all traffic so 100 percent of users see a 4 percent error rate, versus a gated canary that exposes 5 percent of traffic, runs a metric gate against baseline, and auto-aborts so only about 5 percent of users ever saw errors](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-2.png)

The arithmetic of that difference is the whole pitch, and it is worth making concrete. Suppose a bad version serves a 4% error rate, and suppose your detect-and-revert loop is the same in both cases — say it takes four minutes for analysis (or a human) to be confident there is a regression and trigger a rollback.

- **Rolling update:** all traffic is on the bad version for those four minutes. If you serve 10,000 requests per minute, that is $4 \text{ min} \times 10{,}000 \text{ req/min} \times 0.04 = 1{,}600$ failed requests.
- **Canary at 5%:** only 5% of traffic touches the bad version during the bake. That is $4 \text{ min} \times 10{,}000 \times 0.05 \times 0.04 = 80$ failed requests.

Same bug, same detection time, **20× fewer failed requests** — because the canary bounds the *exposure*, not just the *duration*. That ratio is exactly the blast-radius reduction the SRE post derives from error-budget math; here we are just going to *build the thing that delivers it*.

There is a second, more subtle win hiding in that arithmetic, and it is the one that actually changes how a team behaves. With a naive rolling update, the *only* way to make a deploy safer is to detect-and-revert faster — to shrink the four minutes. That is a hard, expensive problem: it means better alerting, faster on-call response, more dashboards to stare at. With a canary, you have a *second* lever that is almost free: shrink the *weight*. Going from a 5% canary to a 1% canary costs you nothing in infrastructure (assuming a request-level router), and it cuts the exposure by another 5×. The blast radius of a deploy is the product $w \times r \times t$ — the weight $w$ of traffic on the bad version, the error rate $r$ of the bug, and the time $t$ it is live. A rolling update pins $w = 1$ and forces you to fight only $t$. Progressive delivery hands you $w$ as a tunable knob, and $w$ is the cheapest of the three to turn. That is why teams that adopt canaries tend to *also* deploy more often: each deploy is cheaper to be wrong about, so the cost of shipping drops, and the deploy-frequency DORA metric climbs as a side effect of the change-fail metric improving. The two move together, which is the empirical pattern the Accelerate research keeps finding — the high performers are fast *and* safe, not fast *or* safe, because the mechanisms that make a deploy safe also make it cheap.

The honest counterpoint, which the matrix in section 6 makes precise: a smaller weight also means a *weaker signal*. At 1% of 10,000 req/min you have 100 req/min on the canary; to observe a 4% error rate with statistical confidence inside a five-minute bake, you need enough samples that the noise floor does not swamp the signal. So $w$ is not a free knob all the way down — there is a floor set by your traffic volume and your metric's variance. The right canary weight is the smallest one that still gives the analysis enough samples to decide. We will return to this when we tune `failureLimit` and the bake duration; for now the principle is the thing to hold onto: progressive delivery turns blast radius from a fixed cost into an engineering parameter you choose deliberately per service.

## 2. The Argo Rollouts model: a drop-in `Deployment` replacement

The two dominant tools for implementing progressive delivery on Kubernetes are **Argo Rollouts** and **Flagger**. We will lead with Argo Rollouts because its model is the most explicit — you write the steps yourself — and come back to Flagger as a contrast in a later section. Both are open-source, both are CNCF-adjacent ecosystem standards, and both do fundamentally the same job: take over the deploy and run it as a gated, metric-aware sequence instead of a blind shift.

Argo Rollouts introduces a custom resource, the `Rollout`, which is **a near-drop-in replacement for a `Deployment`**. The spec is almost identical — `replicas`, `selector`, `template` (your pod spec) are all the same — except instead of `strategy.rollingUpdate` you write `strategy.canary` or `strategy.blueGreen`. A controller (the Argo Rollouts controller, running in your cluster) watches `Rollout` resources the same way the built-in controller watches `Deployment`s, and executes your strategy. Critically, the migration is mechanical: you change `kind: Deployment` to `kind: Rollout`, add `apiVersion: argoproj.io/v1alpha1`, replace the strategy block, and your existing pod template, probes, and resource requests carry over unchanged.

Here is the skeleton of a canary `Rollout`. Read the `strategy.canary.steps` list as the heart of it — that list *is* the rollout plan.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout
  namespace: shop
spec:
  replicas: 10
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout:1.8.0   # set by CD; build-once promoted image
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
  strategy:
    canary:
      canaryService: checkout-canary     # Service routed to the new pods
      stableService: checkout-stable     # Service routed to the current pods
      trafficRouting:
        nginx:
          stableIngress: checkout
      steps:
        - setWeight: 5
        - pause: { duration: 5m }
        - analysis:
            templates:
              - templateName: error-rate-canary
            args:
              - name: canary-service
                value: checkout-canary
              - name: stable-service
                value: checkout-stable
        - setWeight: 25
        - pause: { duration: 10m }
        - setWeight: 50
        - analysis:
            templates:
              - templateName: error-rate-canary
        - setWeight: 100
```

Walk the `steps` from top to bottom. The controller executes them in order. `setWeight: 5` tells the traffic router to send 5% of requests to the canary pods (the new version) and 90%-plus to the stable pods (the current version). `pause: { duration: 5m }` holds at that weight for five minutes — the *bake*, the window during which a slow problem like a memory leak or a gradual latency creep has time to show up. The `analysis` step runs an `AnalysisTemplate` (we will write it next) that queries a metric provider and either passes or fails. If it passes, the controller proceeds to `setWeight: 25`; if it fails, the controller **aborts** — it sets the canary weight back to zero, routing all traffic to the stable version, and the rollout is marked degraded. Figure 3 draws the steps as a ladder you climb only when each gate passes.

![Diagram of the canary steps drawn as a ladder showing setWeight 5 percent with a 5 minute bake and analysis, then setWeight 25 percent with a pause, then setWeight 50 percent with an analysis run, then setWeight 100 percent promoting to stable, with a side rung showing that any failed gate triggers an auto-abort rollback](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-3.png)

A few details that bite people the first time:

- A `pause` with **no** `duration` pauses *indefinitely* until a human runs `kubectl argo rollouts promote checkout`. That is a deliberate manual gate — useful for the final step into prod when you want a human in the loop — but if you write `- pause: {}` by accident, your rollout will sit there forever and you will think it hung. Always be intentional about whether a pause is timed or manual.
- The weights are not magic; *something* has to enforce them. With `trafficRouting` configured (here, NGINX ingress), the weight is enforced by the ingress controller splitting requests by percentage. Without `trafficRouting`, Rollouts falls back to approximating the weight with the *replica count* — which is crude and quantized, as we will see in section 5.
- `revisionHistoryLimit` controls how many old `ReplicaSet`s are kept around so that a rollback is instant (the old pods may still exist, scaled to zero, ready to scale back up). Keep at least 2–3.

You drive and observe a `Rollout` with the `kubectl argo rollouts` plugin:

```bash
# Watch the rollout progress live, with the step ladder and analysis results
kubectl argo rollouts get rollout checkout --watch

# Promote past a manual (indefinite) pause
kubectl argo rollouts promote checkout

# Promote straight to 100%, skipping remaining steps (use sparingly)
kubectl argo rollouts promote checkout --full

# Abort manually (e.g. you saw something the analysis didn't)
kubectl argo rollouts abort checkout

# Retry after fixing an aborted rollout
kubectl argo rollouts retry rollout checkout
```

That `--watch` view is the one you will live in during a deploy: it shows each step, the current weight, whether the analysis is `Running`/`Successful`/`Failed`, and the canary-vs-stable pod counts. It is the difference between "deploying blind" and "deploying with the instrument panel lit."

## 3. The `AnalysisTemplate`: the automated gate

The `pause` steps buy time; the `analysis` steps are what make the rollout *intelligent*. An `AnalysisTemplate` is a reusable custom resource that defines one or more **metrics**, each with a query against a provider (Prometheus, Datadog, New Relic, CloudWatch, Wavefront, a Kubernetes Job, or a raw web request) and a **success condition**. The Rollouts controller runs the query on an interval, evaluates the condition, and tracks how many times it passed or failed. If failures exceed `failureLimit`, the analysis fails, and the rollout aborts.

The single most important pattern is **canary-vs-baseline comparison**. You do not assert "the canary's error rate is below 1%" — that absolute threshold breaks the moment your traffic shifts or a downstream dependency hiccups, paging you for something that is not the new version's fault. Instead you assert "the canary's error rate is not meaningfully *worse* than the stable version's error rate, measured over the same window." If both are at 0.2%, the canary passes. If the stable is at 0.2% and the canary is at 4%, the canary is clearly the problem and you abort. This relative comparison is what makes the gate robust to noise — it cancels out whatever is affecting *both* versions. (This is the same SLI-relative reasoning the SRE post develops; here we are encoding it in PromQL.)

Here is an `AnalysisTemplate` that compares the canary's HTTP 5xx error rate against the stable version's, using Prometheus:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: error-rate-canary
  namespace: shop
spec:
  args:
    - name: canary-service
    - name: stable-service
  metrics:
    - name: canary-error-rate
      interval: 60s          # re-run the query every minute
      count: 5               # run it 5 times across the bake window
      failureLimit: 2        # allow 2 bad readings before aborting
      successCondition: result <= 0.02   # canary 5xx ratio must be <= 2%
      failureCondition: result > 0.05    # > 5% is an immediate hard fail
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(
              http_requests_total{service="{{args.canary-service}}", code=~"5.."}[2m]
            )) /
            sum(rate(
              http_requests_total{service="{{args.canary-service}}"}[2m]
            ))
    - name: canary-vs-stable-latency
      interval: 60s
      count: 5
      failureLimit: 1
      # canary p95 must be within 1.2x of stable p95 (a relative regression gate)
      successCondition: result[0] <= result[1] * 1.2
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            label_replace(
              histogram_quantile(0.95, sum(rate(
                http_request_duration_seconds_bucket{service=~"{{args.canary-service}}|{{args.stable-service}}"}[2m]
              )) by (le, service)),
              "svc", "$1", "service", "(.*)"
            )
```

Read the fields carefully, because each one is a lever:

- **`successCondition` / `failureCondition`.** A measurement passes if `successCondition` is true. A `failureCondition` (optional) lets you express an *immediate* hard fail — "if it ever crosses 5%, do not wait for `failureLimit`, abort now." This catches catastrophic regressions fast while tolerating brief noise on the softer threshold.
- **`failureLimit`.** How many failed measurements you tolerate before the metric is judged failed overall. This is your noise tolerance. Set it to 0 and a single flaky scrape aborts your rollout. Set it to 10 and a real regression runs for ten minutes before you abort. Tune it to your metric's variance — this is the knob that decides whether your gate is jumpy or sleepy.
- **`interval` and `count`.** `interval` is how often the query runs; `count` is how many times total. Together they define the analysis window: `count: 5` at `interval: 60s` watches for five minutes. Pair this with the `pause` durations so the canary bakes for as long as you watch.
- **The relative latency metric** (`result[0] <= result[1] * 1.2`) is the canary-vs-baseline pattern in its purest form: two series come back (canary p95, stable p95), and the canary passes only if it is within 1.2× of stable. A downstream slowdown that hits both versions raises both numbers and the ratio holds — no false abort.

When this analysis fails, the rollout transitions to `Degraded`, the controller sets the canary weight to 0 (full traffic back to stable), and `kubectl argo rollouts get rollout` shows you exactly which metric failed and on which measurement. That is the automated gate: no human, no dashboard-staring, no `rollout undo` typed at 3 a.m. The error budget is protected by code. (The SRE post on the error budget explains *how much* regression you should be willing to tolerate before the gate trips — that budget is what sets your thresholds.)

### Choosing what to gate on: the three metric families

Not all metrics make good gates. The ones that do fall into three families, roughly in order of how directly they protect users:

| Family | Example query target | What it catches | Caveat |
|---|---|---|---|
| **Reliability** (error rate, success rate) | 5xx ratio, gRPC non-OK ratio | crashes, exceptions, dependency failures | misses "wrong answer, 200 OK" bugs |
| **Latency** (p50/p95/p99 duration) | request duration histogram quantile | slow code paths, lock contention, N+1 queries | tail latency is noisy at low volume |
| **Business / domain** (rate of a key event) | checkout-completed rate, login-success rate | logic bugs that the HTTP layer cannot see | needs custom instrumentation |

The strongest gates combine a reliability metric (cheap, universal) with at least one business metric (catches the bugs that return `200 OK` but do the wrong thing — like the silent-mischarge example from section 4). The reliability metric is your seatbelt; the business metric is what actually notices when the *feature* is broken rather than the *server*. If you only ever gate on 5xx rate, you are protected against crashes and blind to logic errors. Pick at least one metric from a family that would have caught *your* last bad deploy — that is the honest test of whether your gate is real.

A subtle point on the query window: the `[2m]` range in the PromQL above is a *sliding window* over the metric, while `interval`/`count` is the *evaluation cadence*. Make the range window a bit shorter than the bake so each measurement reflects recent canary behavior, not a smear that includes the moments before traffic shifted. A `[2m]` rate inside a 5-minute bake evaluated every 60s is a reasonable starting point; a `[10m]` rate inside a 5-minute bake would be reading mostly pre-canary data and would never see the regression in time.

### Which provider, and the analysis state machine

Argo Rollouts speaks to many metric providers — Prometheus, Datadog, New Relic, CloudWatch, Wavefront, Graphite, InfluxDB, a raw `web` (HTTP) request, a Kubernetes `Job`, and Apache SkyWalking. Use the one your SLIs already live in; do not stand up Prometheus *only* for analysis if your org runs Datadog. The query language changes, but the contract — return a value, evaluate `successCondition` — is identical. The `web` and `Job` providers are the escape hatches: a `web` provider hits any HTTP endpoint and parses the JSON response with a JSONPath, so you can gate on a metric service you wrote yourself; a `Job` provider runs an arbitrary container to completion and treats a zero exit code as success, which is how you run a full smoke-test suite as an analysis step.

The state machine each measurement walks is worth internalizing, because it explains every "why did my rollout not abort?" question: a measurement is `Successful` if `successCondition` is true, `Failed` if `failureCondition` is true (or `successCondition` is false and there is no separate failure condition), `Error` if the query itself errored or timed out, and `Inconclusive` if you marked it so. The analysis as a whole fails when `Failed` measurements exceed `failureLimit` *or* `Error` measurements exceed `inconclusiveLimit`. The trap people hit: they set a generous `failureLimit` to tolerate noise, forget that a Prometheus outage produces `Error` (not `Failed`) measurements, and discover their rollout sailed past a window where the metric provider was down and the gate was effectively blind. Decide explicitly whether "I cannot measure it" means "abort" (safe) or "proceed" (dangerous) — set `inconclusiveLimit` accordingly.

### Inline analysis, background analysis, and the experiment

Two refinements worth knowing. First, you can run analysis as a **background** check that runs concurrently with *all* steps rather than as a discrete step — set it at `strategy.canary.analysis` with `startingStep: 1` so the controller continuously watches metrics from the second step onward and aborts the moment they regress, regardless of which weight you are at. This is stricter than step-gated analysis and is the right default for high-risk services. Second, Argo Rollouts has an `Experiment` resource for running an *ephemeral* canary alongside production — useful for A/B-style measurement where you spin up a baseline and a canary pod pair, send synthetic or mirrored traffic, and compare, without touching the real rollout. For most teams, step-gated and background analysis cover the need; reach for `Experiment` only when you genuinely need an isolated comparison.

## 4. Worked example: a bad version caught by the canary gate

Let us run the whole thing end-to-end on the opening incident, but this time with the gate in place. This is the payoff — the same bug, caught and reverted automatically.

#### Worked example: the 4% error version, aborted at 5%

The setup: `checkout` serves **10,000 requests per minute**, runs **10 replicas**, and its stable version errors at a baseline **0.2%** (normal noise from flaky downstreams). The new image `checkout:1.8.0` has the bug from the intro — it throws on roughly **4%** of real-traffic requests. The `Rollout` is the canary spec from section 2 (steps `5% → 25% → 50% → 100%` with a 5-minute bake and the `error-rate-canary` analysis), and `trafficRouting` via NGINX so 5% means 5% of *requests*, not "1 of 10 pods."

Here is the timeline, which Figure 7 also lays out as an ordered sequence:

![Timeline of a canary that catches a bad version showing the deploy at weight 5 percent, analysis starting, the canary error rate measured at 4 percent against a baseline of 0.2 percent, the failure limit being breached, the rollout auto-aborting and rolling back, and the stable version being restored](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-7.png)

1. **t = 0:00.** CD applies the new image to the `Rollout` (via Argo CD reconciling Git — section 9). The controller scales up canary pods and sets the NGINX weight to 5%. Now ~500 req/min hit the canary, ~9,500 hit stable.
2. **t = 0:00 → 5:00.** The `pause: 5m` bakes. The `analysis` step's metrics run every 60 seconds (`interval: 60s`, `count: 5`).
3. **t = 1:00.** First measurement. The canary's 5xx ratio comes back at `0.04` (4%). `successCondition: result <= 0.02` is **false** → measurement 1 is a failure. `failureCondition: result > 0.05` is false (4% < 5%), so it is not an *immediate* hard fail. `failureLimit: 2` means we tolerate 2 failures before aborting.
4. **t = 2:00.** Second measurement: still ~4%. Failure 2.
5. **t = 3:00.** Third measurement: ~4%. Failure 3 > `failureLimit: 2` → the metric is judged **Failed** → the analysis fails → the rollout **aborts**. The controller sets the canary weight to 0; NGINX routes 100% back to stable. The canary pods are scaled down. `kubectl argo rollouts get rollout checkout` shows `Status: Degraded` and the failed `canary-error-rate` measurements.

How much damage did the bug do? It was live at 5% weight for about **3 minutes** before the abort (the failureLimit needed three bad readings a minute apart). So:

$$ \text{failed requests} = 3 \text{ min} \times 10{,}000 \text{ req/min} \times 0.05 \times 0.04 = 60 $$

**Sixty failed requests, over three minutes, seen by ~5% of users.** Compare the naive rolling update from the intro: 100% of users, for the ~12 minutes it took a human to react, at 4%:

$$ 12 \text{ min} \times 10{,}000 \times 1.00 \times 0.04 = 4{,}800 \text{ failed requests} $$

That is an **80× reduction in failed requests** (4,800 → 60), and — just as important — **zero human involvement and zero page**. The on-call engineer wakes up to a Slack message that says "checkout rollout aborted by canary analysis: error rate 4% vs baseline 0.2%," looks at it over coffee, and the prod fleet is already safely on the old version. That is the difference between progressive delivery as a slide-deck phrase and progressive delivery as a thing your pipeline actually does. The change-fail rate that *reaches users* drops, because the gate catches the bad change before it counts as a failure your customers experienced.

One honest caveat: the analysis can only catch what your metrics expose. If the bug is "checkout silently charges the wrong amount but returns 200 OK," your 5xx-rate query will never see it. The gate is exactly as good as the SLIs behind it. That is why the SRE work of choosing the *right* SLIs to gate on — covered in the [progressive delivery reliability post](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) — is the prerequisite, not an afterthought. Wire a canary onto a service with no meaningful SLI and you have built an elaborate `pause` that does nothing.

## 5. Traffic routing: how the weight is actually enforced

We have been writing `setWeight: 5` as if it just happens. It does not just happen — *something* has to make 5% of requests go to the canary and 95% to stable. There are two fundamentally different ways to enforce a weight, and the difference matters a lot. Figure 5 contrasts them.

![Diagram showing how traffic weight is enforced with the Rollout setting a 10 percent weight, one path approximating it crudely by replica ratio of 1 of 10 pods, and a precise path through a mesh or NGINX or ALB that splits requests by exact percentage to the canary service and the stable service, with Flagger shown as a mesh-driven alternative](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-5.png)

**Replica-count weighting (crude, no extra infra).** If you do *not* configure `trafficRouting`, Argo Rollouts approximates the weight by adjusting how many canary vs stable pods sit behind a single `Service`. To get "10%," it makes the canary roughly 10% of the pods, and Kubernetes' default round-robin Service load-balancing sends roughly 10% of connections there. The problems: it is **quantized** — with 10 replicas, your finest grain is 10% (1 pod); you cannot do a true 1% or 5% canary without scaling to 100+ pods. It is **approximate** — round-robin balances *connections*, not requests, and a single long-lived HTTP/2 connection can carry wildly disproportionate traffic. And it **couples weight to capacity** — to get 5% weight you might run fewer canary pods than you need to actually serve 5% of load. Replica weighting is fine for a coarse 50/50-ish canary on a low-stakes service; it is wrong for a precise small-percentage gate.

**Traffic-routing weighting (precise, needs a router).** When you configure `trafficRouting`, Rollouts manipulates a real traffic-splitting layer that splits by *percentage of requests*, decoupled from pod count. The router can be:

- **A service mesh** — Istio (via `VirtualService` weight), Linkerd (via the SMI `TrafficSplit` or the native API), Consul. The mesh's sidecar proxies split traffic at the request level, so 5% means 5% of requests regardless of how many pods back each version.
- **An ingress controller** — NGINX (via canary annotations on a second `Ingress`), Traefik, Contour, AWS ALB (via target-group weights), Google Cloud Load Balancer, Apache APISIX. The ingress/load balancer splits at the edge.
- **A combination** — Rollouts supports composing multiple providers.

With a router you get true 1%/5%/10% weights, fine-grained step ladders, and weight that is independent of how you scale. The cost is operational: you now run and depend on a mesh or a smart ingress. For a service that already lives behind Istio or an ALB, this is free leverage. For a service behind a dumb L4 load balancer, adding a mesh *just* for canary weighting is a real decision — sometimes the right call, often not, and we will come back to that "when not to" in section 11.

Here is the NGINX-ingress flavor of routing, which is one of the lightest to adopt (the canary `Ingress` is the standard NGINX canary annotation mechanism, which Rollouts drives automatically — you usually only declare the primary):

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: checkout
  namespace: shop
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
    - host: checkout.acme.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: checkout-stable     # Rollouts adds the canary Ingress at runtime
                port:
                  number: 80
```

And the Istio flavor, where you declare the `VirtualService` with two routes and let Rollouts rewrite the `weight` fields as the canary climbs:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: checkout
  namespace: shop
spec:
  hosts:
    - checkout.acme.com
  http:
    - route:
        - destination:
            host: checkout-stable
          weight: 100        # Rollouts rewrites these two weights per step
        - destination:
            host: checkout-canary
          weight: 0
```

The mental rule: **if you need a precise small-percentage canary, you need a request-level traffic router.** Replica weighting is the fallback when you do not have one, and it is honest about being coarse.

### Beyond percentage: header and mirror routing

A request-level router unlocks two routing modes that replica weighting simply cannot do, and both are worth knowing because they sidestep the low-traffic signal problem from section 1.

**Header-based canary.** Instead of (or before) shifting a *percentage*, you can route by request attribute — send only requests carrying `X-Canary: true`, or only requests from internal employees, or only a specific cohort, to the canary. Argo Rollouts supports this with `setHeaderRoute` (on supported routers like Istio and ALB); Flagger supports it via `match` conditions. This is how you do a true *internal dogfood* canary: ship the new version, route your own team's traffic to it via a header your VPN injects, exercise it deliberately, and only *then* open a percentage gate to real users. It is the deploy equivalent of "test in prod, on yourself, first."

**Traffic mirroring (shadowing).** The router sends a *copy* of live requests to the canary while the real response still comes from stable — the canary's response is discarded. The canary sees real production traffic shapes and load, but no user is exposed to its output. This is the safest way to vet a new version under realistic load before any real exposure: you watch the canary's error rate and latency on mirrored traffic, and you only begin the weighted rollout once the mirror looks clean. The catch is side effects — a mirrored write to a database or a mirrored charge to a payment processor happens *twice*, so mirroring is only safe for idempotent or read-only paths unless you carefully sandbox the canary's downstreams. Used correctly, it turns "we found out under real load" from a rollback into a no-op observation.

These modes compose with the percentage ladder: a mature pipeline might mirror first (zero exposure, real load), then header-route to internal users (tiny known exposure), then open `setWeight: 1` (real users, bounded), then climb. Each rung trades a little more exposure for a little more signal, and a request-level router is what makes the whole staircase possible.

## 6. Blue-green with Argo Rollouts: instant cutover, instant rollback

Canary is a *gradual* shift. Blue-green is an *instant* one. Both are progressive delivery; they bound risk differently, and the right choice depends on what you are deploying.

In a blue-green deploy, you run **two full copies** of the service: the current version (blue) serving all live traffic, and the new version (green) deployed alongside it but receiving *no* live traffic yet. You point a **preview** endpoint at green, run smoke tests and analysis against it, and when you are satisfied you **flip** the active endpoint from blue to green in one atomic move. All traffic moves at once. If green turns out to be bad, you flip the active endpoint *back* to blue — also in one atomic move, also instant. Blue stays running until you are confident, so the rollback is just a pointer change.

Argo Rollouts implements this with `strategy.blueGreen`, two Services, and an optional pre-promotion analysis:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout
  namespace: shop
spec:
  replicas: 10
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout:1.8.0
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
  strategy:
    blueGreen:
      activeService: checkout-active       # the live Service; flips on promotion
      previewService: checkout-preview     # points at green before the flip
      autoPromotionEnabled: false          # require a human (or analysis) to flip
      scaleDownDelaySeconds: 600            # keep blue alive 10 min after the flip
      prePromotionAnalysis:
        templates:
          - templateName: smoke-and-error-rate
        args:
          - name: target-service
            value: checkout-preview
      postPromotionAnalysis:
        templates:
          - templateName: error-rate-active
        args:
          - name: target-service
            value: checkout-active
```

The key fields:

- **`activeService` / `previewService`.** Two Kubernetes Services. `previewService` always points at the new (green) pods; `activeService` points at the current (blue) pods until promotion, then flips to green. Your users hit `activeService`; your smoke tests hit `previewService`.
- **`prePromotionAnalysis`.** Runs against the *preview* before the flip. This is where you run smoke tests and check error rate on green while it is only serving synthetic/preview traffic — catch the obvious failures before any user sees them.
- **`autoPromotionEnabled: false`.** Holds after pre-promotion analysis passes until a human runs `kubectl argo rollouts promote checkout` (or you set it `true` to flip automatically once analysis passes). For high-stakes flips, keep a human in the loop.
- **`scaleDownDelaySeconds`.** *The most important rollback knob.* After the flip to green, blue's pods stay running (not serving) for this many seconds. As long as blue is alive, rolling back is an instant flip of `activeService` back to blue. Set this to cover the window in which you expect to discover load-induced problems. Set it to 0 and blue is gone the moment you flip — your instant rollback evaporates.
- **`postPromotionAnalysis`.** Runs against the *active* service *after* the flip, watching real production traffic on green. If it fails within the `scaleDownDelaySeconds` window, blue is still there to flip back to.

The defining trade-off of blue-green is right there in the resource: **2× the resources** (you run blue and green at full capacity simultaneously) in exchange for **instant cutover and instant rollback**. And the cutover is all-or-nothing — when you flip, 100% of traffic moves at once. There is no 5% slice absorbing the first hit. Pre-promotion analysis mitigates this by vetting green before the flip, but pre-promotion traffic is not real production traffic; some problems only appear under real load. That is the blue-green blast-radius story: instant rollback, but the *flip itself* exposes everyone the moment it happens. The contrast with canary is summarized in Figure 4.

There is a hard prerequisite that the YAML does not show and that bites teams the first time: **the instant rollback is only instant if the data layer can absorb it.** When you flip back from green to blue, blue must still be able to serve correctly. If green ran a database migration that blue's code does not understand — dropped a column blue still reads, or changed an enum blue still writes — then flipping back lands you on a version that is now broken against the migrated schema, and your "instant rollback" has nowhere safe to go. The discipline that keeps the rollback truly instant is the **expand-and-contract** (a.k.a. parallel-change) migration pattern: every schema change is split so that the old and new code versions can both run against the *same* schema during the overlap, and the destructive part (the contract) only happens after the new version is fully promoted and the old one is retired. This is its own subject — the database series covers it in [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — but the rule for *this* post is short: a blue-green flip is only reversible if blue and green are backward- and forward-compatible at the data layer for the whole overlap window. Skip that discipline and you have built a one-way door that merely *looks* like it opens both ways.

The same compatibility rule applies to a canary — arguably more so, because in a canary the two versions co-exist serving real traffic for the *entire* rollout, not just a brief overlap. A canary running for 30 minutes means 30 minutes of both versions reading and writing the same data store. So whichever strategy you pick, the contract is the same: the versions must be compatible while they share traffic. The strategies differ in *how long* that overlap lasts (a brief blue-green flip vs a longer canary climb), which is one more input to the choice — if a long version co-existence is risky for your data model, a short blue-green overlap may be the safer shape even though it costs more.

![Matrix comparing rolling update, canary, and blue-green across blast radius, rollback speed, extra cost, and metric gate, showing rolling exposes all users with a slow reverse and only a readiness gate, canary bounds blast radius to a 5 percent slice with second-level abort and an analysis gate at low extra cost, and blue-green flips zero to 100 percent instantly with a second-level rollback but at double the replicas](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-4.png)

## 7. Worked example: blue-green instant rollback under load

Canary's worked example showed catching a bug *before* it spread. Blue-green's superpower is different: it is the fastest *reversal* when a problem only shows up after the cutover.

#### Worked example: green smoke-passes, then spikes p99 under real load, flipped back in seconds

The setup: `checkout` again, 10 replicas, blue-green strategy with `scaleDownDelaySeconds: 600`. The new version `checkout:1.9.0` is correct — it returns the right answers, no error-rate regression — but it has a performance bug: a missing database index that only matters at production cardinality. Staging's tiny dataset never triggers it. The pre-promotion smoke tests pass (they exercise correctness, not load). So:

1. **t = 0:00.** Green deploys to the preview service. `prePromotionAnalysis` runs smoke tests against `checkout-preview`: all pass (correctness is fine), error rate on preview traffic is nominal. Analysis succeeds.
2. **t = 0:03.** A human reviews, runs `kubectl argo rollouts promote checkout`. `activeService` flips from blue to green. **100% of production traffic is now on green.** Blue stays running (scale-down delayed 600s).
3. **t = 0:04.** Under real production query volume, the missing index means full table scans. p99 latency climbs from 180ms to 2.4s. Downstream timeouts begin. `postPromotionAnalysis` (and your alerting) catch it within ~60 seconds.
4. **t = 0:05.** Decision: roll back. Because blue is still alive (we are well inside the 600s window), the operator runs `kubectl argo rollouts undo checkout` (or, equivalently, the controller's abort flips `activeService` back to blue). The active Service's selector flips back to blue's pods. **Traffic is back on the healthy version in the time it takes Kubernetes to propagate a Service endpoint update — a few seconds.**

Now compare the *same* latency bug shipped via a rolling update. A rolling update has already *replaced* blue's pods with green's as it progressed; there is no full copy of blue sitting ready. To roll back you run `kubectl rollout undo`, which starts a *new* rolling update back to the old image — bring up old pods, wait for readiness, surge guard, drain new pods. For a 10-replica service with conservative surge settings, that reverse roll takes the same ~10–15 minutes the forward roll took. During those 10–15 minutes, your p99 is at 2.4s and customers are abandoning carts.

The numbers, then. Blue-green reversal: **traffic restored in ~5 seconds** once the decision is made. Rolling update reversal: **~15 minutes**. If you serve 10,000 req/min and the latency bug costs you (say) 3% of requests to timeout-driven abandonment, blue-green loses about $\frac{5}{60} \times 10{,}000 \times 0.03 \approx 25$ requests during the reversal; the rolling update loses $15 \times 10{,}000 \times 0.03 = 4{,}500$. The flip is **~180× faster to reverse**, which is the entire reason blue-green exists. You pay for it with the 2× resource bill while both colors are up. Figure 8 shows the forward flip and the reversal as two columns.

![Before and after diagram of a blue-green flip showing the forward path where green runs on the preview service, pre-promotion smoke tests pass, and the active service flips to green, then the reversal path where green latency spikes p99, the active service flips back to blue, and traffic is reverted in seconds](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-8.png)

This is exactly the case the strategy matrix points at: when the failure mode you fear is *fast to detect but slow to reverse with a rolling update*, and you can afford the spare capacity, blue-green's instant flip-back is the cleanest insurance you can buy. When the failure mode is *gradual and you want to limit who ever sees it*, canary's slice-based exposure wins. The two are not competitors; they are different tools for different blast-radius shapes.

## 8. Flagger: the mesh-driven alternative

Argo Rollouts makes you write the steps. **Flagger** takes the opposite stance: you declare your *goal* — the canary increments, the interval, the metric thresholds — and Flagger automates the loop on top of a service mesh or ingress, promoting or rolling back by itself. It was built by Weaveworks, is now a CNCF project, and integrates tightly with Flux for GitOps and with Istio/Linkerd/App Mesh/NGINX/Gloo/Contour for traffic.

The model: Flagger watches a *regular* Kubernetes `Deployment` (not a custom `Rollout`) plus a `Canary` custom resource that describes the analysis. When you push a new image to the `Deployment`, Flagger detects the change, creates a canary copy, and runs the automated canary — stepping the weight up by `stepWeight` every `interval`, checking the metrics each round, and either promoting (copying the new spec to the primary `Deployment`) or rolling back. Here is a `Canary` for contrast with the Argo `Rollout`:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: checkout
  namespace: shop
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: checkout              # a normal Deployment; Flagger manages the canary
  service:
    port: 80
    targetPort: 8080
  analysis:
    interval: 1m                # evaluate every minute
    threshold: 5                # roll back after 5 failed checks
    maxWeight: 50               # canary tops out at 50% before promotion
    stepWeight: 10              # +10% per successful interval
    metrics:
      - name: request-success-rate
        thresholdRange:
          min: 99               # success rate must stay >= 99%
        interval: 1m
      - name: request-duration
        thresholdRange:
          max: 500              # p99 latency must stay <= 500ms
        interval: 1m
    webhooks:
      - name: smoke-test
        type: pre-rollout       # run before any traffic shift
        url: http://flagger-loadtester.shop/
        timeout: 30s
        metadata:
          type: bash
          cmd: "curl -sf http://checkout-canary.shop/healthz"
      - name: load-test
        type: rollout           # generate traffic during analysis
        url: http://flagger-loadtester.shop/
        metadata:
          cmd: "hey -z 1m -q 10 -c 2 http://checkout-canary.shop/"
```

The contrasts that matter when you choose between them:

| | Argo Rollouts | Flagger |
|---|---|---|
| Resource you deploy | a `Rollout` CRD (replaces `Deployment`) | a normal `Deployment` + a `Canary` CRD |
| Control style | imperative — *you* write the steps | declarative — Flagger drives the loop |
| Traffic routing | mesh, ingress, ALB, or replica-count | mesh or ingress (no replica-count fallback) |
| Built-in load/smoke testing | external (you wire it) | built-in webhooks + load-tester |
| GitOps pairing | Argo CD (same ecosystem) | Flux (same ecosystem) |
| Blue-green support | first-class (`strategy.blueGreen`) | via mesh, less first-class |
| Best when | you want explicit step control and blue-green | you want a hands-off mesh-driven canary |

Neither is "better." If you live in the Argo ecosystem (Argo CD for GitOps) and want explicit, auditable step control plus first-class blue-green, Argo Rollouts is the natural fit. If you live in the Flux/mesh world and want a more hands-off, convention-driven canary with built-in load generation, Flagger is excellent. Pick the one that matches the rest of your platform; do not run both for the same service.

## 9. Putting it in the pipeline: CI builds, CD gates, the Rollout runs

A `Rollout` resource is not progressive delivery by itself — it is the *engine*. The pipeline is what feeds it. Here is the clean separation of concerns, and it follows the series' "build once, promote everywhere" spine exactly:

- **CI builds and promotes the image** — once. The exact bytes you tested are the bytes that deploy. The pipeline never rebuilds per environment. (This is the whole point of the [build-once, promote-everywhere artifact post](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning).)
- **CD updates the `Rollout`** — it changes the image tag in the `Rollout` spec to the promoted digest.
- **The `Rollout` + `AnalysisTemplate` run the gated canary automatically** — stepping, baking, analyzing, promoting or aborting, with no further pipeline involvement.
- **The pipeline waits** for the rollout to report `Healthy` (promoted) or `Degraded` (aborted), and fails the deploy job if it aborted.

There are two ways to drive the CD step: **push** (the pipeline runs `kubectl`/`argo rollouts` against the cluster) and **pull** (GitOps — the pipeline commits the new image tag to a Git repo, and Argo CD reconciles it into the cluster).

### The push flavor (CI directly drives the rollout)

A GitHub Actions job that updates the image and waits on the rollout, failing the job if the canary aborts:

```yaml
name: deploy-prod
on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: "Promoted image tag (digest-pinned)"
        required: true

permissions:
  id-token: write      # OIDC to the cloud — no long-lived kube creds in CI
  contents: read

jobs:
  rollout:
    runs-on: ubuntu-latest
    environment: production    # gated environment: required reviewers, etc.
    steps:
      - uses: actions/checkout@v4

      - name: Configure cluster access via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::1234:role/gha-deploy
          aws-region: us-east-1
      - run: aws eks update-kubeconfig --name prod --region us-east-1

      - name: Install argo rollouts plugin
        run: |
          curl -sSL -o /usr/local/bin/kubectl-argo-rollouts \
            https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
          chmod +x /usr/local/bin/kubectl-argo-rollouts

      - name: Set the new image and trigger the rollout
        run: |
          kubectl argo rollouts set image checkout \
            checkout=ghcr.io/acme/checkout:${{ inputs.image_tag }} \
            -n shop

      - name: Wait for the gated rollout (fails if the canary aborts)
        run: |
          # blocks until Healthy (exit 0) or Degraded/aborted (non-zero)
          kubectl argo rollouts status checkout -n shop --timeout 30m

      - name: Report outcome
        if: failure()
        run: echo "::error::Rollout aborted by canary analysis — prod stayed on the previous version."
```

The crucial step is the last meaningful one: `kubectl argo rollouts status ... --timeout 30m` **blocks** until the rollout finishes. If the canary analysis aborts, `status` exits non-zero, the job fails, and your pipeline surfaces a clear "rollout aborted" — *and the cluster is already safely back on the old version because the abort happened in-cluster, not in the pipeline.* The pipeline did not have to do the rollback; the `Rollout` controller did. The pipeline just observes and reports.

### The pull flavor (GitOps — the safer default for prod)

The push flavor hands prod cluster credentials to CI (here softened with OIDC so they are short-lived, but CI still reaches into the cluster). The GitOps flavor inverts the trust: the pipeline only ever commits to Git, and **Argo CD running *inside* the cluster pulls the change and applies it**. CI never holds prod credentials. This is the more secure model and the one I recommend for production — the full why is in the GitOps posts, including the planned sibling on progressive delivery meeting GitOps (`progressive-delivery-meets-gitops`) and the existing treatment of pull-based CD.

In GitOps, CD is a `git commit` that bumps the image tag in the `Rollout` manifest in the config repo:

```bash
# CI's deploy step: update the image tag in the GitOps repo, then commit.
# Argo CD watches this repo and reconciles the change into the cluster;
# Argo Rollouts then runs the gated canary inside the cluster.
yq -i '.spec.template.spec.containers[0].image =
  "ghcr.io/acme/checkout:1.9.0@sha256:abc123..."' \
  envs/prod/checkout-rollout.yaml

git add envs/prod/checkout-rollout.yaml
git commit -m "deploy(checkout): promote 1.9.0 to prod"
git push
```

And the Argo CD `Application` that watches the repo and keeps the cluster in sync:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: checkout-prod
  namespace: argocd
spec:
  project: shop
  source:
    repoURL: https://github.com/acme/gitops
    targetRevision: main
    path: envs/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: shop
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

The division of labor is the elegant part: **Argo CD reconciles the *desired state*** (it applies the `Rollout` with the new image to the cluster), and **Argo Rollouts runs the *delivery*** (it executes the gated canary that the `Rollout` describes). Argo CD's job ends once the `Rollout` object matches Git; Rollouts' job is to safely make the running pods match the `Rollout`'s strategy. They compose cleanly because they own different layers. Argo CD even has a Rollouts integration so the Argo CD UI shows the canary progressing — `Healthy` only reports once the rollout is fully promoted. For the reconcile-loop mechanics that make this safe, see the GitOps posts in this series and the planned `progressive-delivery-meets-gitops`.

### Smoke tests on the canary and preview

In both flavors, run smoke tests against the *canary* (or *preview*) endpoint before widening. With Argo Rollouts, an `analysis` step can reference an `AnalysisTemplate` whose provider is a Kubernetes `Job` that runs your smoke suite against `checkout-canary`; with Flagger, the `pre-rollout` webhook does it. The principle is the same: hit the new version directly, on its own Service, and confirm the basics before the traffic gate widens. A smoke test is cheap insurance against the dumb failures (a missing config map, a typo in an env var) that should never make it past 5%.

### Measuring the win honestly

It is easy to claim a canary "improved our change-fail rate" and impossible to defend the claim without measuring it. Here is how to measure it honestly, because a number you cannot defend is worse than no number.

The DORA **change-failure rate** is the fraction of deploys that result in a degraded service requiring remediation (a rollback, a hotfix, a patch). The subtle thing a canary does is change *what counts as a failed change that reached users*. A deploy that the canary caught and auto-aborted at 5% is, from the customer's perspective, a near-miss, not an incident — only a sliver of traffic saw the regression and the fleet self-healed. So you should track two related numbers, not one:

- **Deploys that reached full production and then needed remediation** — the strict change-fail rate. A working canary drives this *down*, because the bad ones get caught before full promotion.
- **Rollout abort rate** — the fraction of rollouts the analysis aborted. This number going *up* is not bad; it means the gate is doing its job. But if it spikes, it is also your early warning that *upstream* quality (tests, review, staging fidelity) is slipping — the canary is catching what those should have caught first.

To attribute the improvement, instrument the rollout outcomes. Argo Rollouts emits Prometheus metrics (`rollout_info`, phase counts) and Kubernetes events on abort and promotion; pipe those into the same dashboard as your incident tracker. Then you can say, with evidence, "in the quarter before canaries, 7 of 40 prod deploys caused a customer-visible incident (17.5% change-fail rate); in the quarter after, 1 of 120 deploys did, and the canary aborted 9 others before they reached full traffic." That is a defensible before→after: the change-fail rate that *reached users* fell from 17.5% to under 1%, deploy frequency tripled, and the nine aborts are the receipts proving the gate earned its keep. The dishonest version of this — "canaries made us 80× safer" with no instrumentation behind it — is exactly the kind of unfalsifiable claim that makes engineers distrust the practice. Measure the aborts; they are the proof.

## 10. Feature flags: the finer-grained complement

Canary and blue-green operate at the **deploy** granularity — they control which *version of the binary* serves which slice of traffic. **Feature flags** operate at a finer granularity: they let you deploy code with a new feature *dark* (the binary is in prod, but the code path is gated behind a runtime flag set to off), and then turn the feature on progressively — 1% of users, then 5%, then everyone — *without another deploy*. This is the deepest form of "decouple release from deploy," and it composes with progressive delivery rather than competing with it. (The dedicated treatment is the planned sibling `feature-flags-decoupling-deploy-from-release`; the fleet-level view is in the microservices post on [deployment strategies including feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags).)

The distinction is worth being crisp about, because mixing them up leads to bad architecture:

- A **canary** answers "is *version 1.9.0* good?" It shifts a slice of traffic to a whole new build and gates on metrics. Its unit is *the deployable artifact*.
- A **feature flag** answers "should *this specific feature* be on for *this specific user*?" The whole fleet runs the same build; the flag decides behavior at request time. Its unit is *a code path*.

You use them together: deploy `1.9.0` (which contains the new checkout flow behind a flag) via a canary so you know the *build* is healthy, then, once it is fully promoted, ramp the *feature flag* from 1% → 100% over days while watching business metrics. The canary protects you from "the build is broken"; the flag protects you from "the feature is bad for users" and lets you kill the feature in milliseconds (a flag flip is faster than any rollback, because nothing redeploys). A minimal flag check looks like this:

```python
# The feature ships dark in the binary; the flag controls exposure at runtime,
# decoupled from the deploy. Turning it off is instant — no rollback, no redeploy.
def checkout(user, cart):
    if flags.is_enabled("new-checkout-flow", user=user, default=False):
        return new_checkout(user, cart)   # ramped 1% -> 5% -> 100% via the flag UI
    return legacy_checkout(user, cart)
```

The trade-off feature flags add is **complexity and flag debt**: every flag is a branch in your code, an entry in a flag service (LaunchDarkly, Unleash, Flagsmith, or your own), and a thing someone must remember to clean up. A codebase littered with stale flags is its own kind of mess. So flags are the right tool for *user-facing* progressive rollout and for kill-switches on risky features; they are *not* a substitute for the build-level canary that protects you from a broken binary. Use both, at the layer each is good at.

## 11. War story and how to reach for this

### War story: Knight Capital, the deploy that lost \$440M in 45 minutes

The canonical cautionary tale of a deploy with no blast-radius control is Knight Capital, August 1, 2012. Knight deployed new trading software to its production servers — eight of them. The deploy was manual, server by server, and a technician missed one: seven servers got the new code, one kept the old. Worse, the new code repurposed an old, dormant feature flag (`Power Peg`) that the old code on the eighth server interpreted differently. When the market opened, the eighth server began firing erroneous orders into the live market at machine speed. There was no automated metric gate watching for an order-rate anomaly, no canary slice to contain the damage, no fast automated rollback. By the time humans understood what was happening and stopped it, roughly **45 minutes** had passed and Knight had taken a pre-tax loss of about **\$440 million** — enough to effectively end the company, which was acquired shortly after.

It is unfair to reduce a complex incident to one lesson, but the deploy-engineering lessons are stark and they are exactly this post's subject: (1) an inconsistent fleet — seven servers new, one old — is precisely what GitOps reconciliation and an atomic blue-green flip prevent (the cluster either matches desired state or it does not; you do not get a "six of eight" partial flip). (2) Reusing a dormant flag for new behavior is the feature-flag debt failure mode — old flags must be *removed*, not silently repurposed. (3) Most of all: there was no automated gate watching a business metric (order rate, fill anomalies) that could have aborted within seconds instead of 45 minutes. A canary `AnalysisTemplate` on the right SLI is the institutional memory that says "this rollout is producing anomalous output — stop." Knight had the deploy. It had no gate. The gate is the entire point.

### How to reach for canary vs blue-green vs neither

Every one of these mechanisms has a cost, and the engineering maturity is in knowing when *not* to use them. Figure 6 is the decision tree I actually use.

![Decision tree for choosing a deploy strategy starting from deploying a risky change, branching on whether you have a Service Level Indicator to gate on, leading to canary with an analysis gate, blue-green with an instant flip, or a feature flag flip per user when you do, and to a plain rolling update with readiness only when you do not](/imgs/blogs/progressive-delivery-in-the-pipeline-canary-and-blue-green-6.png)

Use **canary (metric-gated)** when: the change is risky, you have a meaningful SLI to gate on (error rate, latency, a business metric), and you want to limit how many users ever see a regression. This is the default for high-traffic, high-stakes services. It needs a request-level traffic router for precise small percentages.

Use **blue-green** when: you need instant cutover and instant rollback, the failure mode you fear is fast-to-detect-but-slow-to-reverse, you can afford 2× capacity during the overlap, and an all-at-once flip is acceptable (or you vet thoroughly with pre-promotion analysis). It is also the cleaner choice when the new version cannot safely run *concurrently* with the old at the data layer for long — a short blue-green overlap is easier to reason about than a long canary co-existence. (Schema compatibility during the overlap is its own discipline — see [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations).)

Use a **plain rolling update** when: the change is low-risk, you have no SLI worth gating on, traffic is low, or the service is internal and a brief regression is tolerable. *Most* deploys of *most* services are fine with a well-configured rolling update. Do not add a canary to everything.

**When NOT to do this at all:**

- **No SLI to gate on?** Do not build a canary. A canary with no real metric to analyze is a glorified `pause` that adds operational surface and false confidence. Get the SLI first. The reliability side of choosing SLIs is in the [SRE progressive delivery post](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery).
- **A 3-person startup on a PaaS?** Do not stand up Istio + Argo Rollouts + Prometheus to canary a single service that deploys twice a week. The operational cost of the mesh and the analysis stack dwarfs the risk you are bounding. Use your PaaS's built-in gradual rollout or a simple rolling update, and revisit when traffic and team size justify it.
- **Rolling update still drops requests?** Fix that *first*. A canary built on a broken rolling primitive still drops requests on every pod handoff. The [rolling updates post](/blog/software-development/ci-cd/rolling-updates-and-zero-downtime-deploys) is the prerequisite.
- **Can't afford 2× capacity?** Then blue-green is off the table for that service; use canary, which only needs the small surge for the canary slice.

### Stress-testing the design

Reasoning to a strategy is not enough; you have to ask what breaks it.

- **What if the canary metric is noisy?** Set `failureLimit` above the noise floor, use canary-vs-baseline *relative* comparisons (which cancel common-mode noise), and lengthen the bake so the analysis has enough samples. A jumpy gate that aborts on noise trains the team to ignore it — worse than no gate.
- **What if traffic is too low for the canary to have signal?** At 5% of 100 req/min, you get 5 req/min on the canary — not enough to measure a 4% error rate with confidence inside a 5-minute bake. For low-traffic services, raise the canary weight, lengthen the bake, or use blue-green with thorough pre-promotion smoke tests instead. Statistics do not care about your YAML.
- **What if two PRs merge and deploy at once?** A `Rollout` is for one version at a time; a second update mid-rollout *restarts* the rollout from the new image (or, in GitOps, the latest commit wins the reconcile). Serialize prod deploys — a deploy queue or a GitHub `environment` with a concurrency group — so you never have two canaries fighting over the same traffic split.
- **What if the registry is down mid-rollout?** New canary pods cannot pull the image and never become ready; `progressDeadlineSeconds` eventually fails the rollout, leaving stable serving 100%. The build-once principle helps here too — the *stable* image is already on the nodes' image cache, so the running version is unaffected.
- **What if the rollback also fails?** Blue-green's `scaleDownDelaySeconds` is your safety margin: as long as blue is alive, flip-back is a Service selector change, which is about as reliable an operation as Kubernetes has. If you set the delay to 0 and blue is gone, your "instant rollback" is now a full rolling update back — which is exactly the slow path you were trying to avoid. Keep the delay generous on high-stakes services.
- **What if the analysis provider (Prometheus) is down?** A `failureCondition` that fires on a query *error* (not just a bad value) will abort the rollout — which is the safe default ("if I can't measure it, don't widen it"). Decide this explicitly; an analysis that treats "no data" as "pass" will happily promote a blind rollout.

## 12. Key takeaways

- A plain rolling update gates only on **readiness**, and readiness is not quality — it cannot tell that the new version returns wrong answers, regressed latency, or throws on long-tail inputs. Progressive delivery adds a **traffic layer** (expose a slice) and an **analysis layer** (watch real metrics, abort automatically).
- **Argo Rollouts** replaces the `Deployment` with a `Rollout` CRD whose `strategy.canary.steps` interleave `setWeight` / `pause` / `analysis`. The migration is mechanical: same pod template, swap the strategy block.
- The `AnalysisTemplate` is the automated gate. The robust pattern is **canary-vs-baseline relative comparison**, not absolute thresholds — it cancels common-mode noise. Tune `failureLimit` to your metric's variance.
- **Traffic weight needs a request-level router** (mesh, ingress, ALB) to be precise at small percentages; replica-count weighting is crude and quantized. No router, no true 1% canary.
- **Blue-green** trades 2× resources for instant cutover and instant rollback — flip the active Service to green, flip it back to blue if green misbehaves. `scaleDownDelaySeconds` keeps blue alive so the rollback stays a pointer change, not a slow reverse roll.
- **Flagger** is the declarative, mesh-driven alternative — you state the goal and it drives the loop, with built-in load/smoke webhooks. Choose by ecosystem (Argo CD → Rollouts; Flux → Flagger); do not run both for one service.
- In the pipeline: **CI builds once, CD updates the `Rollout`, the Rollout runs the gated canary, the pipeline waits and reports**. The GitOps flavor (Argo CD reconciles, Argo Rollouts runs) keeps prod credentials out of CI and is the safer default.
- **Feature flags** are the finer-grained complement — they ramp a *feature* at runtime, decoupled from the deploy, and kill it in milliseconds. Use them *with* a canary (which protects the *build*), not instead of it.
- Know when *not* to: no SLI to gate on, a tiny team on a PaaS, a broken rolling primitive underneath, or no spare capacity for blue-green. Most deploys are fine with a well-tuned rolling update.
- The measured win is real: in the worked example, the same 4% bug caused **4,800 failed requests via a rolling update vs ~60 via a 5% canary** (80× fewer), with zero human involvement; the blue-green reversal beat a rolling reversal by **~180×** on time-to-restore.

## Further reading

- The reliability theory this post implements — why you gate a deploy on an SLI and bound its blast radius: [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery), and [the error budget, the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability).
- The primitive this builds on: [rolling updates and zero-downtime deploys](/blog/software-development/ci-cd/rolling-updates-and-zero-downtime-deploys), and the series map, [from commit to production, the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model).
- The fleet-level view of the same strategies: [deployment strategies, blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags). Schema safety during a version overlap: [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations).
- The Argo Rollouts documentation (the `Rollout` spec, `AnalysisTemplate` reference, and traffic-router integrations), the Flagger documentation (the `Canary` resource and metric templates), and the Argo CD documentation for the GitOps pairing.
- Coming in this series: progressive delivery meeting GitOps (how the reconcile loop and the rollout controller compose, planned as `progressive-delivery-meets-gitops`) and feature flags as the decoupling of deploy from release (planned as `feature-flags-decoupling-deploy-from-release`).

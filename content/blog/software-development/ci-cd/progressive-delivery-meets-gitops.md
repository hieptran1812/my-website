---
title: "Progressive delivery meets GitOps: canary deploys automated end to end"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Wire Argo Rollouts and Argo CD together to run fully automated canary deployments that promote on passing Prometheus analysis and roll back the moment metrics fail."
tags:
  [
    "ci-cd",
    "devops",
    "gitops",
    "progressive-delivery",
    "argo-rollouts",
    "argo-cd",
    "canary-deployment",
    "kubernetes",
    "flagger",
    "prometheus",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/progressive-delivery-meets-gitops-1.png"
---

It is a Friday afternoon and you have just merged a two-line config change into your service's repository. The pipeline goes green in four minutes. Argo CD detects the diff and syncs. Forty seconds later every pod in production is running the new image. The health checks pass. Everything looks fine on the dashboard for about three minutes — and then the on-call page fires. Payment success rate: 91.3%, down from 99.8%. You trace it back to the deploy. You execute the rollback: revert the config, push to main, wait for the new pipeline build, watch Argo CD sync again. By the time your users are fully recovered you have burned 38 minutes — 4 minutes of page-response time, 8 minutes of investigation, 8 minutes of pipeline, and 18 minutes of watching pod restarts and confirming recovery metrics. The root cause is a connection pool configuration that behaved differently at production scale than in staging. No amount of pre-deploy testing revealed it because staging traffic was 40× lighter.

That scenario repeats across engineering teams at every stage of maturity. The tooling changes — maybe it is Jenkins instead of GitHub Actions, maybe it is kubectl instead of Argo CD — but the pattern is the same: a change that passed every gate in the non-production environment caused a regression that only appeared under real production load. Progressive delivery is the mechanism that limits the damage. Instead of shipping to 100% of production immediately, you ship to 10%, measure what actually happens, and only widen the rollout if real traffic confirms the change is safe. The question this post answers is: how do you automate that process end to end so that human judgment is not in the critical path of every deploy?

The answer is the combination of GitOps and progressive delivery tools. GitOps — specifically Argo CD — gives you a pull-based reconciliation operator that continuously drives the cluster toward the state described in a Git repository, without your CI pipeline holding production credentials. Progressive delivery tools — Argo Rollouts and Flagger — give you controllers that understand traffic-weighted promotion and can evaluate real-time metric analysis before advancing a canary. When these two layers work together, the system can promote a canary from 10% to 50% to 100% based on Prometheus data, roll back automatically if error rates spike, and do all of this without any human touching a terminal.

By the end of this post you will be able to write a Rollout manifest with multi-step canary progression and automated analysis, configure an AnalysisTemplate that measures p99 latency and HTTP success rate, understand how Flagger provides an alternative model, and reason clearly about why automated GitOps rollback for progressive delivery is architecturally different from `git revert`. You will also have concrete before/after numbers: teams that add automated analysis gates typically see change-failure rate fall from the 12–15% range to 3–4%, and MTTR fall from 45 minutes to under 10 minutes. Those numbers trace directly to the four DORA metrics that predict elite delivery performance.

![Argo CD syncs the Rollout CRD, canary traffic begins at 10 percent, AnalysisRun either promotes to 50 then 100 percent or aborts and restores the stable revision](/imgs/blogs/progressive-delivery-meets-gitops-1.png)

---

## 1. Why GitOps and progressive delivery belong together

The commit→build→test→package→deploy→operate spine that runs through this series has a structural tension at the deploy step. On the left side of that step you want speed: merge fast, build once, promote the immutable artifact through environments without rebuilding. On the right side you want safety: not every bit that passes a test suite is safe at production scale. These two goals are genuinely in tension. Every hour you add to the deploy process to increase confidence is an hour of lead time you are adding to the feedback loop. Every shortcut you take on validation is a potential change-failure rate increase.

Progressive delivery is the mechanism that resolves this tension without a trade-off. You deploy fast — pushing the new artifact to production — but you control how much real traffic sees the new version and for how long, and you make the promotion decision based on data rather than time. The key word is *automated*. A canary that requires an engineer to check a Grafana dashboard and manually trigger a promote command is not a solution — it is just a more elaborate version of the same human bottleneck. Automated analysis gates remove the human from the critical promotion decision while keeping human judgment in the loop for the failure case (which you investigate after the rollback, not in the middle of it).

GitOps is the governance layer that makes automated progressive delivery trustworthy. Without GitOps, automated canary promotion means your CI pipeline has credentials that allow it to call "promote" on the production cluster — effectively giving CI write access to prod. With GitOps, the CI pipeline writes to Git; the Argo CD operator reads from Git and applies to the cluster. The progressive delivery controller (Argo Rollouts or Flagger) runs inside the cluster and queries Prometheus — no external access required. The only credential that leaves the cluster is the one Argo CD uses to read the config repo, and that credential is read-only.

The four DORA metrics tell you precisely where this combination adds value. Deploy frequency and lead time for changes improve when humans are not blocking the promotion decision at every step — automated analysis means you can ship at 3 AM without waking anyone. Change-failure rate improves when the system catches regressions at 10% traffic instead of 100% — the blast-radius reduction is multiplicative. Time-to-restore improves when rollback is automated to under 90 seconds instead of requiring a manual pipeline run. Progressive delivery with GitOps moves all four metrics in the right direction simultaneously, which is exactly what the DORA research identifies as the fingerprint of elite delivery teams.

The [mental model post](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) frames the commit→operate spine this post plugs into. The [GitOps fundamentals post](/blog/software-development/ci-cd/gitops-git-as-the-source-of-truth) explains why pull-based reconciliation beats handing production credentials to CI — read that post for the security model that makes the progressive delivery architecture safe. The [Argo CD and Flux post](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) covers Application CRDs, sync waves, and the self-healing loop that this post's `ignoreDifferences` configuration builds on top of. The [SRE perspective on progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) covers progressive delivery as an SLO-gated reliability practice — the reliability theory that this post's toolchain implementation serves.

One more thing before diving into the toolchain: progressive delivery combined with GitOps is not a weekend project. The minimum viable stack (Argo CD, Argo Rollouts, two Service objects per service, one AnalysisTemplate per SLI, one VirtualService per service, an `ignoreDifferences` configuration per Application) requires careful setup and ongoing maintenance. The reward for that upfront investment is a deploy pipeline that can run at 3 AM without waking anyone, that catches regressions before they reach most of your users, and that restores stability faster than any human-in-the-loop process. The rest of this post shows you exactly how to build it.

![Progressive delivery toolchain from Git config repo through Argo CD and Rollouts controller to traffic provider and Prometheus analysis feeding the promote-or-abort decision](/imgs/blogs/progressive-delivery-meets-gitops-2.png)

---

## 2. The Rollout CRD: extending Deployment for progressive delivery

Kubernetes ships with a `Deployment` object that supports rolling updates. You set `spec.strategy.rollingUpdate.maxSurge` and `maxUnavailable`, and the Deployment controller replaces pods in batches. Rolling updates are appropriate for services where the blast radius of a failed deploy is low and where pod-level health checks (readiness probes) are sufficient validation. For most production services with revenue or compliance consequence, those conditions do not hold.

The fundamental problem with a standard rolling update is that it has no concept of traffic-level validation. A pod that passes its readiness probe immediately starts receiving its share of requests. There is no mechanism to say "send 10% of traffic to the new pods for five minutes, verify the error rate stays below 1%, then continue." The rolling update proceeds based on infrastructure health (pods starting, probes passing) rather than application health (real user requests succeeding).

Argo Rollouts solves this by introducing the `Rollout` CRD. Structurally, a Rollout manifest is almost identical to a Deployment — it carries the same `spec.template` (the pod spec), the same `selector`, and the same `replicas` field. The difference is entirely in `spec.strategy`. Where a Deployment strategy block contains `type: RollingUpdate` with pod-count knobs, a Rollout strategy block contains `canary` or `blueGreen` with step-based traffic control and analysis references.

Here is a complete Rollout manifest for a payment service using Istio for traffic splitting:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: payment-service
  namespace: payments
  labels:
    app: payment-service
    team: platform
spec:
  replicas: 10
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: payment-service
  template:
    metadata:
      labels:
        app: payment-service
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: payment-service
          image: ghcr.io/acme/payment-service:v2.4.1
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
            limits:
              cpu: 1000m
              memory: 512Mi
          readinessProbe:
            httpGet:
              path: /healthz/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
            successThreshold: 1
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz/live
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
  strategy:
    canary:
      canaryService: payment-service-canary
      stableService: payment-service-stable
      trafficRouting:
        istio:
          virtualService:
            name: payment-service-vsvc
            routes:
              - primary
      steps:
        - setWeight: 10
        - analysis:
            templates:
              - templateName: success-rate
            args:
              - name: service-name
                value: payment-service-canary
            duration: 5m
        - setWeight: 50
        - analysis:
            templates:
              - templateName: success-rate
            args:
              - name: service-name
                value: payment-service-canary
            duration: 10m
        - setWeight: 100
      scaleDownDelaySeconds: 30
      abortScaleDownDelaySeconds: 0
```

Several design decisions in this manifest deserve explanation. The `canaryService` and `stableService` fields reference two separate Kubernetes Service objects — one that selects new pods (canary revision) and one that selects old pods (stable revision). The Istio VirtualService uses these two Services as weighted backends. The `scaleDownDelaySeconds: 30` means the old stable ReplicaSet is retained for 30 seconds after full promotion, giving a fast rollback window if a post-promotion spike appears. The `abortScaleDownDelaySeconds: 0` means on abort, the canary pods are terminated immediately rather than waiting.

The two Service objects that the Rollout references must also be in your Git config repo:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: payment-service-stable
  namespace: payments
spec:
  selector:
    app: payment-service
  ports:
    - port: 80
      targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: payment-service-canary
  namespace: payments
spec:
  selector:
    app: payment-service
  ports:
    - port: 80
      targetPort: 8080
```

The selectors on these Services start identical — both select pods with `app: payment-service`. The Rollouts controller adds a revision label to the canary and stable ReplicaSets at runtime, and updates the Service selectors to point to the correct ReplicaSet. This is managed automatically; you do not need to set the revision labels yourself.

### Migrating from Deployment to Rollout

The migration path is simpler than most teams expect. The structural steps are:

1. Create the two Service objects (canary and stable) in your config repo.
2. Create or update the VirtualService (or Ingress) to reference both Services with initial weights of 100/0.
3. Change the `apiVersion` and `kind` of your existing Deployment manifest to `argoproj.io/v1alpha1` and `Rollout`.
4. Replace the `spec.strategy.rollingUpdate` block with `spec.strategy.canary` with steps and analysis references.
5. Create the AnalysisTemplate (see next section).
6. Update the Argo CD Application with `ignoreDifferences` for the VirtualService weight fields.

The existing pods do not restart during migration. The Rollouts controller picks up the existing ReplicaSet as the stable revision. The next image change triggers the first canary.

The one operational gotcha: if you have a HorizontalPodAutoscaler targeting the Deployment, you need to update its `scaleTargetRef` to point to the Rollout. HPAs work with Rollouts but they target the stable ReplicaSet's scale, not the total pod count — keep this in mind when setting replica minimums to ensure there is headroom for the canary pods.

---

## 3. The AnalysisTemplate: encoding your promotion criteria as code

The AnalysisTemplate is the central innovation in automated progressive delivery. It defines what metrics to query, from which observability backend, at what frequency, over what duration, and what constitutes a passing result. A separate, short-lived AnalysisRun object is created for each analysis step in each canary deployment, carrying the live query results and the pass/fail determination.

The AnalysisTemplate belongs in your Git config repo alongside the Rollout manifest. It is a first-class Kubernetes object, versioned in Git, reviewed in pull requests, and subject to the same change control as any other production configuration. When your team wants to tighten the success rate threshold from 99% to 99.5%, they open a PR to the config repo — not a Slack message to the platform team.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: payments
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 1m
      count: 5
      successCondition: result[0] >= 0.99
      failureLimit: 1
      inconclusiveLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(http_requests_total{
              job="{{args.service-name}}",
              status!~"5.."
            }[5m]))
            /
            sum(rate(http_requests_total{
              job="{{args.service-name}}"
            }[5m]))
    - name: p99-latency
      interval: 1m
      count: 5
      successCondition: result[0] < 200
      failureLimit: 1
      inconclusiveLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_milliseconds_bucket{
                job="{{args.service-name}}"
              }[5m])) by (le)
            )
```

This template defines two metrics running in parallel: HTTP success rate (must be ≥ 99.0%) and p99 latency (must be under 200 ms). Each metric is sampled every minute. The `count: 5` means the AnalysisRun collects five samples before concluding. The `failureLimit: 1` means a single sample that fails the threshold aborts the entire AnalysisRun — the Rollouts controller then immediately begins the rollback. The `inconclusiveLimit: 2` means if up to two samples return no data (empty series), the run continues rather than aborting; this handles the warm-up period at the start of a canary.

The `{{args.service-name}}` substitution allows the same AnalysisTemplate to be reused across multiple Rollouts. Each Rollout passes its canary service name as an argument, and the Prometheus queries use that to scope the metric to the canary pods only.

### The AnalysisRun lifecycle in detail

When the Rollout controller reaches an `analysis` step in its step array, it creates an AnalysisRun resource. The AnalysisRun is a first-class Kubernetes object — you can inspect it with `kubectl get analysisrun` and examine the per-metric results with `kubectl describe analysisrun <name>`. This observability is important: when an AnalysisRun fails, the describe output tells you exactly which metric failed, what the query returned, and what the threshold was. You are not debugging a log file; you are reading structured status from a Kubernetes object.

The AnalysisRun has four terminal states:

**Successful**: all metrics passed all samples within their configured limits. The Rollout controller advances to the next step in the canary — either increasing the traffic weight or completing promotion to 100%.

**Failed**: at least one metric exceeded its `failureLimit`. The Rollout controller calls `Abort()` internally, which scales down the canary ReplicaSet, restores the VirtualService weights to 100/0 for the stable service, and sets the Rollout status to `Degraded`. This entire sequence happens in seconds — typically under 30 seconds from the AnalysisRun failure to the stable service receiving 100% of traffic again.

**Inconclusive**: the data was insufficient or contradictory. The classic inconclusive case is when the Prometheus query returns no series — for example, the canary pods have not yet received enough traffic to produce a statistically valid rate over the 5-minute window. An inconclusive result pauses the Rollout for human intervention by default. You can change this behavior with `inconclusiveLimit` (allow some inconclusive samples before failing) or by setting `dryRun: true` on an individual metric to observe it without letting it affect the outcome.

**Error**: the metric provider itself failed — Prometheus returned a 500, the query syntax is invalid, or the provider endpoint is unreachable. Errors are treated as inconclusive by default. This safe default means a Prometheus outage does not cause a canary to promote blindly — it pauses the canary instead.

### Handling the "inconclusive" trap

The inconclusive state catches teams most often in three scenarios:

**Low-traffic services**: a service that receives 20 requests per minute at 10% canary weight gets 2 requests per minute to the canary. The Prometheus `rate()` function over a 5-minute window needs at least some samples to produce a meaningful result — 2 req/min × 5 min = 10 requests is enough to compute a rate, but statistical noise at this scale means a single 500 error produces a 10% error rate, which will fail a 99% success threshold. The fix is to either increase the minimum canary weight (set the first step to 25% for low-traffic services), widen the analysis window, or accept that automated analysis is not appropriate for this service's traffic level and use a manual promote gate instead.

**Metric name changes**: the new version renames a Prometheus label. The old query `job="payment-service-canary"` returns no series because the new version exports metrics with `service="payment-service"`. The AnalysisRun returns inconclusive until it exceeds its `inconclusiveLimit`, then fails. Prevention: add a CI gate that validates the metric exists in the Prometheus mock environment before merging the image tag change to the config repo.

**Cold start latency**: the analysis starts while the canary pods are still warming up their connection pools, establishing database connections, and filling caches. The first minute of metrics is legitimately anomalous. The fix is a leading `pause` step before the analysis: `- pause: {duration: 60s}`. This gives the canary pods 60 seconds to warm up before the AnalysisRun starts collecting samples.

### Non-Prometheus metric providers

The Prometheus provider is the most common, but the AnalysisTemplate schema supports six provider types and you can mix them in a single template.

The Datadog provider is a near-drop-in replacement for teams not running Prometheus:

```yaml
    - name: error-rate
      interval: 1m
      count: 5
      successCondition: result[0] < 0.01
      provider:
        datadog:
          apiVersion: v2
          interval: 5m
          query: |
            avg:trace.web.request.errors{service:payment-service-canary}
            / avg:trace.web.request.hits{service:payment-service-canary}
```

The web hook provider calls an external HTTP endpoint and evaluates the JSON response against a success condition. This is useful for calling a synthetic monitoring endpoint or a load test runner:

```yaml
    - name: synthetic-test
      interval: 5m
      count: 1
      successCondition: result.pass == true
      provider:
        web:
          url: https://synthetic.monitoring.internal/api/v1/check
          method: POST
          body: |
            {"service": "{{args.service-name}}", "scenario": "checkout_flow"}
          headers:
            - key: Authorization
              value: "Bearer {{secrets.SYNTHETIC_TOKEN}}"
```

The Kubernetes Job provider is the most powerful option for running full integration test suites against the canary endpoint before promoting:

```yaml
    - name: integration-tests
      provider:
        job:
          spec:
            template:
              spec:
                containers:
                  - name: test-runner
                    image: ghcr.io/acme/integration-tests:latest
                    env:
                      - name: TARGET_HOST
                        value: "http://payment-service-canary.payments.svc.cluster.local"
                      - name: TEST_SUITE
                        value: "smoke,payment-flows"
                restartPolicy: Never
            backoffLimit: 0
```

The Job completes with exit code 0 for success, non-zero for failure. The AnalysisRun maps non-zero exit to a failed measurement.

![AnalysisTemplate metric providers split into time-series systems Prometheus, Datadog, and CloudWatch and synthetic providers web hook and Kubernetes Job](/imgs/blogs/progressive-delivery-meets-gitops-7.png)

---

## 4. Flagger: the alternative progressive delivery controller

Argo Rollouts is not the only progressive delivery controller that integrates with GitOps. Flagger, originally built by Weaveworks, solves the same problem with a different integration model and is now part of the Flux ecosystem. Understanding the difference in approaches helps you choose the right tool and, more importantly, helps you reason about the trade-offs when you encounter them in a codebase you are inheriting.

The fundamental architectural difference: **Argo Rollouts requires you to replace your Deployment objects with Rollout CRD objects**. You cannot use Argo Rollouts with an existing Deployment without converting it. **Flagger watches your existing Deployment objects**. You do not modify the Deployment; you create a separate `Canary` CRD that points to it, and Flagger manages the traffic shifting behind the scenes.

When Flagger detects a change to a watched Deployment's pod spec (a new image tag, an updated environment variable, a changed resource request), it creates a shadow `<name>-primary` Deployment for the stable version and a `<name>-canary` Deployment for the new version, then gradually shifts traffic between them using your configured traffic provider. Your original Deployment becomes the "desired state" declaration that Flagger monitors. Flagger manages the primary and canary Deployments directly; you never touch them.

![Comparison matrix of Argo Rollouts and Flagger across CRD type, traffic providers, GitOps integration, analysis sources, and learning curve](/imgs/blogs/progressive-delivery-meets-gitops-4.png)

Here is a complete Flagger Canary CRD wired to an NGINX ingress controller — no service mesh required:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: payment-service
  namespace: payments
spec:
  # target the existing Deployment
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: payment-service
  # Flagger will manage this Ingress
  ingressRef:
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    name: payment-service
  service:
    port: 80
    targetPort: 8080
    portDiscovery: true
  analysis:
    # run analysis every 30 seconds
    interval: 30s
    # number of failed checks before rollback
    threshold: 5
    # maximum canary traffic weight percentage
    maxWeight: 50
    # step increment between iterations
    stepWeight: 10
    # prometheus metrics to validate
    metrics:
      - name: request-success-rate
        # success rate must be above 99%
        thresholdRange:
          min: 99
        interval: 1m
      - name: request-duration
        # p99 latency must be below 500ms
        thresholdRange:
          max: 500
        interval: 30s
    # integration tests before promotion
    webhooks:
      - name: acceptance-test
        type: pre-rollout
        url: http://flagger-loadtester.test/
        timeout: 30s
        metadata:
          type: bash
          cmd: "curl -sd 'test' http://payment-service-canary.payments/api/health | grep ok"
      - name: load-test
        type: rollout
        url: http://flagger-loadtester.test/
        timeout: 5s
        metadata:
          cmd: "hey -z 1m -q 10 -c 2 http://payment-service-canary.payments/api/checkout"
```

The `analysis.stepWeight: 10` means Flagger increases traffic by 10 percentage points every `interval` (30 seconds), up to `maxWeight: 50`. After reaching 50%, if all metric checks pass, Flagger promotes to 100% automatically. If any metric check fails more than `threshold: 5` times, Flagger rolls back by resetting the canary weight to 0 and scaling down the canary Deployment. The `threshold: 5` is a noise buffer — it prevents a single spike in metrics from triggering a rollback on a service that is otherwise healthy.

### How Flagger implements NGINX traffic splitting

Flagger creates and manages a mirror Ingress alongside the primary Ingress. The mirror Ingress carries the NGINX canary annotations:

```yaml
# Managed by Flagger — do not edit by hand
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: payment-service-canary
  namespace: payments
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "20"
spec:
  rules:
    - host: payments.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: payment-service-canary
                port:
                  number: 80
```

The NGINX ingress controller reads the `canary-weight` annotation and directs approximately 20% of incoming requests to the canary backend. The primary Ingress (which Flagger does not touch) handles the remaining 80%. As Flagger advances the canary, it updates the `canary-weight` annotation value — from 10 to 20 to 30 to 40 to 50 — at each step interval.

The key point for the GitOps workflow: the primary Ingress spec in your Git config repo shows no canary-specific content. The canary Ingress is created and managed entirely by the Flagger controller. Argo CD needs to be configured to ignore the Flagger-managed resources (or to exclude them from the sync), otherwise Argo CD will try to delete the canary Ingress it did not create.

### Choosing between Argo Rollouts and Flagger

The honest trade-off comparison:

| Dimension | Argo Rollouts | Flagger |
|-----------|--------------|---------|
| Deployment migration required | Yes — replace with Rollout CRD | No — wrap with Canary CRD |
| Primary GitOps integration | Argo CD native | Flux native |
| Works with other GitOps operator | Yes, with care | Yes, with care |
| Traffic providers | Istio, NGINX, ALB, SMI, Traefik, Kong | Istio, NGINX, App Mesh, Contour, SMI |
| Blue-green strategy | Yes | Yes |
| Operator UI | Argo Rollouts Dashboard | None official |
| Analysis sources | Prometheus, Datadog, CloudWatch, web, job | Prometheus, Datadog, CloudWatch, Graphite |
| Multi-step weight control | Explicit steps array | Automatic stepWeight increments |
| Manual promote/abort | kubectl argo rollouts promote | kubectl annotate flagger.app |

The bottom line: if your team already runs Argo CD and has the capacity to migrate Deployment objects to Rollout CRDs (a one-time effort, typically a few hours per service), choose Argo Rollouts. The Rollout CRD gives you precise step control and the Argo Rollouts Dashboard provides genuine operational visibility into where a canary is in its lifecycle. If your team runs Flux or wants zero migration overhead for existing services, choose Flagger.

---

## 5. Traffic splitting in GitOps: who manages which object

The most common operational confusion when adding progressive delivery to a GitOps setup is understanding the ownership boundary between Argo CD and the progressive delivery controller. You have multiple controllers operating on overlapping sets of Kubernetes objects. Getting the ownership model wrong leads to a continuous reconcile fight where Argo CD undoes what Rollouts just set, and Rollouts immediately re-patches what Argo CD just reset.

The ownership model is clean if you understand it once:

- **Argo CD owns**: the Rollout manifest (desired image, replica count, strategy definition), the AnalysisTemplate, the Service objects, and the VirtualService *structure* (hosts, routes, backends).
- **Argo Rollouts owns**: the VirtualService *weight values* and the Rollout *status* fields (including whether it is paused). These are runtime values that change during a canary progression.

The `ignoreDifferences` configuration in the Argo CD Application tells Argo CD to ignore specific JSON paths when comparing live state to Git state:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: payment-service
  namespace: argocd
  annotations:
    notifications.argoproj.io/subscribe.on-sync-succeeded.slack: releases-channel
    notifications.argoproj.io/subscribe.on-sync-failed.pagerduty: platform-oncall
spec:
  project: production
  source:
    repoURL: https://github.com/acme/platform-config
    targetRevision: main
    path: apps/payments
  destination:
    server: https://kubernetes.default.svc
    namespace: payments
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - RespectIgnoreDifferences=true
      - ApplyOutOfSyncOnly=true
  ignoreDifferences:
    - group: networking.istio.io
      kind: VirtualService
      jsonPointers:
        - /spec/http/0/route/0/weight
        - /spec/http/0/route/1/weight
    - group: argoproj.io
      kind: Rollout
      jsonPointers:
        - /spec/paused
  revisionHistoryLimit: 10
```

The `RespectIgnoreDifferences` sync option is critical — without it, Argo CD would still apply the ignored fields during a sync operation, overwriting the Rollouts-managed weights with the Git-stored values (which are the initial 100/0 weights). With this option, Argo CD skips the ignored JSON paths entirely during apply.

The `/spec/paused` ignore for the Rollout handles the case where an operator manually pauses a canary mid-flight using `kubectl argo rollouts pause payment-service`. Without this ignore, Argo CD's self-heal loop would immediately unpause the Rollout by applying the Git manifest (which has no `paused` field, effectively setting it to false).

### The VirtualService in Git

The VirtualService lives in your Git config repo as a declarative manifest with the initial weights:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: payment-service-vsvc
  namespace: payments
spec:
  hosts:
    - payment-service
    - payments.example.com
  gateways:
    - istio-system/public-gateway
    - mesh
  http:
    - name: primary
      route:
        - destination:
            host: payment-service-stable
            port:
              number: 80
          weight: 100
        - destination:
            host: payment-service-canary
            port:
              number: 80
          weight: 0
```

The weights start at 100/0 in Git — the stable service receives all traffic, the canary receives none. The Rollouts controller patches these values during a canary. After full promotion to 100%, the Rollouts controller sets weights back to 100/0 and updates the stable service to point to the new revision. Git never changes.

### The mesh vs ingress traffic model

Your choice of traffic provider affects both the traffic splitting mechanism and the operational complexity of the stack:

**Istio (service mesh model)**: The weight applies at the sidecar proxy level. Both stable and canary pods participate in the same Service; the Envoy proxy in each pod implements the traffic split in the data plane. This works for all traffic types including internal gRPC and service-to-service HTTP — not just traffic entering the cluster through an ingress. The trade-off is significant operational overhead: running Istio adds roughly 100–200 MB of memory per pod for the sidecar, 5–20 ms of added latency per hop, and substantial control plane resource consumption (typically 2 vCPU and 4 GB RAM for the istiod control plane on a medium cluster).

**NGINX ingress (ingress-only model)**: The weight applies at the ingress controller level via annotations. Canary traffic splitting only works for traffic entering the cluster through the ingress — internal service-to-service calls bypass the split entirely. This is simpler and has zero per-pod resource overhead. For services that receive external traffic and do internal calls on verified-stable internal interfaces, NGINX is usually sufficient.

**AWS ALB (cloud load balancer model)**: The Rollouts controller manages ALB target group weights through the AWS Load Balancer Controller. The ALB splits traffic at the load balancer level, before it enters the cluster. This has the same ingress-only limitation as NGINX and adds a dependency on the AWS Load Balancer Controller CRD. Appropriate for EKS clusters where ALB is already the ingress standard and adding NGINX or Istio would be another layer to operate.

---

## 6. The full canary lifecycle end to end

All the pieces are now defined. Here is the complete workflow from a developer merging a config PR to a stable promotion at 100% traffic.

![Full canary lifecycle from config PR merged at T plus 0 minutes through Argo CD sync, 10 percent canary, two analysis runs, and promote to 100 percent at T plus 20 minutes](/imgs/blogs/progressive-delivery-meets-gitops-5.png)

**T+0 — PR merges to config repo main branch.** The change is a single-line diff updating the image tag in the Rollout manifest from `v2.4.0` to `v2.4.1`. The PR went through automated YAML linting, a `kubectl diff --server-side` against the staging cluster, and a policy check (OPA/Conftest) verifying the image tag references an existing digest in the registry. These gates are in the CI pipeline on the config repo, not the application repo — a distinction that matters because the config repo is the source of truth for what runs in production.

**T+1 — Argo CD detects the diff.** With the GitHub webhook integration, Argo CD receives a push event within seconds of the merge and initiates a sync. Without a webhook (polling mode), detection takes up to 3 minutes. The sync confirms the Application is OutOfSync, compares the desired Rollout manifest to the live Rollout, and applies the updated manifest to the cluster.

**T+1:30 — Rollouts controller starts the canary.** The Rollouts controller watches for changes to Rollout objects. It detects the image tag update, creates a new ReplicaSet for `v2.4.1`, and scales it to 1 pod (10% of 10 replicas). It then patches the VirtualService: `stable=90, canary=10`. The stable ReplicaSet (`v2.4.0`) remains at 9 pods. At this point 10% of production traffic is hitting the new version.

**T+2 — The canary pod passes readiness.** The `v2.4.1` pod starts, establishes database connections, warms up the cache layer, and passes its readiness probe after approximately 5–10 seconds. Istio adds the canary pod to the canary Service endpoint list. Traffic begins flowing to the new version.

**T+2 — First AnalysisRun begins.** The Rollouts controller sees the first step (setWeight: 10) is complete and advances to the second step (the first analysis block). It creates an AnalysisRun object with a reference to the `success-rate` template. The AnalysisRun begins querying Prometheus every minute.

**T+2 through T+7 — Analysis samples collected.** Five Prometheus samples are collected, one per minute. The success rate averages 99.87%. The p99 latency averages 142 ms. Both metrics pass all five samples within their thresholds. The AnalysisRun terminates as Successful.

**T+7:30 — Promote to 50%.** The Rollouts controller advances to the `setWeight: 50` step. It scales the canary ReplicaSet to 5 pods, the stable to 5 pods, and patches the VirtualService to 50/50. The second AnalysisRun begins.

**T+7:30 through T+17:30 — Second analysis at higher load.** Ten Prometheus samples are collected. The 50/50 split means the canary is now serving five times more traffic than at 10%. Any problem that only appeared under load would manifest here. In this case, both metrics continue to pass. The second AnalysisRun terminates Successful.

**T+18 — Promote to 100%.** The Rollouts controller patches the VirtualService to 0/100. It updates the stable service selector to point to the `v2.4.1` ReplicaSet. The old stable ReplicaSet (`v2.4.0`) begins draining connections — existing requests complete, and the pods stop receiving new traffic. After `terminationGracePeriodSeconds: 30`, the old pods terminate. After `scaleDownDelaySeconds: 30`, the old ReplicaSet is deleted.

**T+20 — Deploy complete.** The new version is fully stable. The Rollout status shows `Healthy`. Argo CD shows the Application as `Synced` and `Healthy`. Zero human interactions after the config PR merged.

---

## 7. Automated rollback: why it is not `git revert`

This is the conceptual question that causes the most confusion when teams first add progressive delivery to a GitOps workflow. The confusion is understandable: in standard GitOps, rolling back means reverting the Git commit and letting Argo CD sync the reverted state. Why is progressive delivery rollback different?

The answer is that automated rollback in progressive delivery is an **execution decision**, not an **intent decision**. The manifest in Git says: "deploy `v2.4.1` with a canary strategy." The AnalysisRun failing says: "this execution attempt did not meet the success criteria." These are different things. The intent (deploy `v2.4.1`) might still be valid — perhaps the failure was transient, caused by a brief Prometheus scrape gap, or by a downstream dependency that spiked momentarily. The correct response to an execution failure is to restore the current stable state and investigate, not to immediately declare the intent wrong and revert Git.

![Failed analysis triggers in-cluster abort restoring stable in 1 minute with Git unchanged versus traditional git revert and full 15-30 minute redeploy cycle](/imgs/blogs/progressive-delivery-meets-gitops-6.png)

When an AnalysisRun fails, the Rollouts controller takes these actions in order:

1. Sets the Rollout status to `Degraded` with an abort reason annotation.
2. Patches the VirtualService weights back to 100/0 (stable service receives all traffic).
3. Scales the canary ReplicaSet down to 0 pods.
4. Retains the stable ReplicaSet at full replica count.

Total time from AnalysisRun failure to stable service at 100%: typically under 60 seconds. The Git manifest still describes `v2.4.1`. Argo CD still shows the Application as `OutOfSync` (because the live Rollout status is Degraded while the desired state in Git is a healthy Rollout). This is the correct and expected state — the cluster is stable, but there is a known deviation from the desired state that requires investigation.

### What happens on the next sync

The next time Argo CD syncs — either on the next poll cycle (up to 3 minutes) or when triggered manually — it applies the Rollout manifest from Git. The Rollout manifest still says `v2.4.1`. The Rollouts controller sees the current stable as `v2.4.0` and the desired image as `v2.4.1`. It starts a new canary.

This is intentional. If the first canary failed because of a transient metric spike, the second attempt will likely succeed. If it fails again, the team has a signal that the problem is systematic and needs investigation. The Rollout's `progressDeadlineSeconds` field defines how long a canary can remain in a degraded state before the controller stops retrying.

If you want to stop the retry loop — because you know the new version has a bug that needs a code fix — you have two options:

**Option A — Update Git to the old image tag.** This is the explicit rollback. Open a PR, change `v2.4.1` to `v2.4.0` in the Rollout manifest, merge. Argo CD syncs. The Rollout does nothing (the current stable is already `v2.4.0`). This is appropriate when the new version has a confirmed code defect.

**Option B — Pause the Rollout in Git.** Add `spec.paused: true` to the Rollout manifest in Git, merge. Argo CD syncs. The Rollouts controller holds the canary at its current state without advancing or starting a new attempt. This is appropriate when you need more time to investigate but the stable version is healthy and you do not want to block future deploys of other services.

The key insight: **Git describes intent; the Rollouts controller manages execution of that intent.** A failed analysis is an execution failure. You only revert Git when the intent itself was wrong.

### Observing rollback state with kubectl and Argo CD

When a canary aborts, the operational trail is clear and discoverable from multiple angles. The Rollout object carries a rich status section:

```bash
kubectl argo rollouts get rollout payment-service --watch
```

The output shows the current step, the stable and canary revision hashes, and the reason for any abort. The Argo CD Application will show the Application as `Degraded` with a health status of `Progressing` (the Rollout has not yet completed). The Argo CD UI displays the Rollout's pod counts, the active AnalysisRun, and the current VirtualService weights in real time — significantly faster to interpret than reading kubectl describe output.

For alerting, configure the Argo CD notification controller to send a Slack message when an Application transitions to Degraded:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
  namespace: argocd
data:
  trigger.on-degraded: |
    - when: app.status.health.status == 'Degraded'
      send: [app-degraded-slack]
  template.app-degraded-slack: |
    message: |
      Application {{.app.metadata.name}} is Degraded.
      Rollout aborted. Stable revision restored.
      Reason: {{.app.status.operationState.message}}
      Sync: {{.app.spec.source.repoURL}}/tree/{{.app.spec.source.targetRevision}}/{{.app.spec.source.path}}
```

This notification fires within seconds of the abort completing. On-call engineers receive the alert, know the stable service is already restored, and can begin root cause analysis without emergency response pressure.

### Why automated rollback is faster than `git revert`

Traditional manual rollback for a standard Deployment via GitOps: revert the commit (or push a new commit with the old image tag) → Argo CD sync (up to 3 minutes polling + sync time) → Kubernetes rolling update (1–5 minutes for pod replacements) = 4–8 minutes minimum. If the pipeline build is required (for a push-based CD), add 5–15 minutes.

Progressive delivery automated rollback: AnalysisRun failure detected → Rollouts controller patches VirtualService → stable service at 100% = under 60 seconds. The difference is that the stable ReplicaSet was never scaled down during the canary — it was running the entire time at 90% (or 50%) capacity, needing only a weight patch to return to 100%. There is no pod restart, no image pull, no rolling update.

This is the MTTR improvement that automated progressive delivery delivers: from 18–45 minutes to under 2 minutes for the service restoration. The investigation and fix still take as long as they take — but users stop seeing errors in under 2 minutes rather than in 18.

---

## 8. Before and after: the blast-radius and DORA argument

The economic case for analysis-gated canaries becomes clear when you quantify the blast-radius difference and map it to DORA metrics.

### Blast-radius arithmetic

A service handling 10,000 requests per minute. A bad deploy with a 4% error rate. Two scenarios:

**Scenario A — Blind rolling deploy (no canary):**
- T+0: deploy starts, pods replacing one by one
- T+4: all 10 pods running new version, all 10,000 req/min hitting new version
- T+7: on-call page fires (3 minutes of alert evaluation)
- T+15: rollback initiated (8 minutes of investigation + manual rollback steps)
- T+23: rollback complete (8-minute rolling deploy back to old version)

Total bad requests: 10,000 req/min × 19 minutes × 4% error rate = 7,600 error responses.

**Scenario B — Canary at 10% with 5-minute analysis:**
- T+0: canary starts, 1 of 10 pods running new version
- T+2: canary pod receiving traffic, 1,000 req/min (10% of total)
- T+7: AnalysisRun detects 4% error rate (40 errors/min, above 1% threshold), aborts
- T+7:30: VirtualService weight reset to 100/0, stable service at full traffic

Total bad requests: 1,000 req/min × 5.5 minutes × 4% error rate = 220 error responses.

The analysis gate reduced user-facing errors from 7,600 to 220 — a 34× reduction in blast radius. The canary weight percentage and the analysis window duration are the two levers. Smaller canary weight × faster analysis = smaller blast radius, but at the cost of longer total deploy time and more noise in the metrics.

![Blind canary at 10 percent promoted on timer to 100 percent leading to 15 percent CFR versus analysis-gated canary catching error spike at 10 percent and aborting with stable restored](/imgs/blogs/progressive-delivery-meets-gitops-3.png)

### DORA metric impact

The blast-radius reduction translates directly into DORA metric improvements. The change-failure rate (CFR) — defined as the percentage of deploys that cause a degraded service — falls because failures that would have been full production incidents are caught at 10% traffic. They are technically failures (the canary aborted), but they do not degrade the user experience at scale. Whether you count an aborted canary as a "change failure" is a measurement choice; most teams define CFR as "deploys that caused a user-visible production incident requiring intervention," in which case automated canary aborts do not count.

Time-to-restore (MTTR) falls because the automated rollback takes under 2 minutes rather than 18–45 minutes. The on-call engineer may still investigate the root cause, but service restoration is no longer part of the incident timeline for aborted canaries.

The less intuitive DORA improvements are in deploy frequency and lead time. Teams that add automated analysis gates tend to ship more often, not less, despite the added deploy latency. The reason: trust. Engineers who know that a broken deploy will auto-rollback before it hits 50% of traffic are willing to merge and ship earlier in the day, earlier in the week, and with less pre-ship verification theater. The virtuous cycle — smaller batches, shipped more often, each one lower risk — is exactly what the DORA research documents as the mechanism behind elite team performance.

![Automated canary analysis gates drop change-failure rate from 15 percent to 3 percent and MTTR from 45 minutes to 5 minutes compared to timer-based canaries](/imgs/blogs/progressive-delivery-meets-gitops-8.png)

---

## 9. War story: the payment service that passed every test and wrecked production

The following is a realistic composite incident — the specific details are illustrative, but the failure pattern is documented across multiple real post-mortems.

A payment processing service was releasing a new version that improved checkout latency by parallelizing two downstream API calls that had previously been sequential. The change passed unit tests (the downstream calls were mocked), integration tests (staging used a different database configuration with a lower connection pool max), a Trivy security scan, and a 10-minute smoke test against the staging environment. The deploy was scheduled for a Tuesday afternoon. The engineer running it watched the rolling update complete, saw the health checks go green, checked the error rate dashboard for three minutes, and called it done.

Seven minutes after full rollout, the p99 latency on the payment endpoint shot from 180 ms to 1,400 ms. The error rate went from 0.2% to 6.8%. The root cause: the parallel API calls were acquiring connections from a shared HTTP connection pool. Under production load (about 40× the staging traffic level), the parallel requests from many concurrent users exhausted the connection pool, causing requests to queue behind connection acquisition rather than executing. Staging had never triggered this because it ran a tiny fraction of production's concurrent request count.

The incident timeline: 7 minutes to first alert, 14 minutes to identify the deploy as the cause, 20 minutes to execute the rollback (a full revert-and-redeploy cycle, because the team did not have Argo Rollouts and was using a standard Deployment), 8 minutes to confirm recovery. Total incident duration: 49 minutes. User-facing error window: 34 minutes at 6.8% error rate = approximately 2,800 errors per minute × 34 minutes = 95,200 bad responses.

With a 10% canary gate and a 99% success rate threshold:

- T+2: canary pod starts serving 1,000 req/min (10% of 10,000)
- T+4: parallel connections under the canary hit the pool limit; p99 spikes to 1,400 ms, error rate climbs to 6.8%
- T+7: AnalysisRun sample 3 reads 93.2% success rate (far below the 99% threshold); failure registered
- T+7:30: AnalysisRun terminates Failed; Rollouts controller aborts; VirtualService weight set to 100/0; stable service fully restored

Total bad responses: 1,000 req/min × 5.5 minutes × 6.8% = 374 bad responses. Total incident duration for the service degradation: under 2 minutes. The team still needed to investigate and fix the connection pool issue, but users stopped seeing errors in under 2 minutes versus 34 minutes.

The fix was a one-line change to the connection pool configuration: a maximum connections cap that prevented the parallel requests from exhausting the pool under concurrent load. It was tested against a production-like load test environment before re-shipping. The second canary attempt succeeded.

The lesson that made it into the team's post-mortem: production scale reveals classes of bugs that no pre-deploy test catches — resource exhaustion, concurrency bugs, thundering herd effects, timeout cascades. Progressive delivery does not prevent these bugs from being shipped; it limits their exposure to the fraction of traffic that hits the canary before the analysis gate triggers.

---

## 10. Stress tests: what breaks the pattern and how to handle it

Every architectural pattern has failure modes that only appear in production. Progressive delivery with GitOps is no exception. These are the specific failure scenarios that teams encounter, the operational symptoms, and the correct responses.

**The canary metric is noisy.** Your p99 latency threshold is 200 ms. Your service has a background job that fires every 60 seconds and briefly spikes latency to 280 ms for about 8 seconds while it acquires a database lock. Your AnalysisRun samples every 60 seconds and catches this spike in sample 2, failing the analysis. The canary aborts. You investigate, find no user-visible problem, and manually retry. It aborts again on a different sample. The gate is producing false positives.

The root cause is a mismatch between the analysis sample granularity and the metric's natural periodicity. Three options: widen the Prometheus query window from `[5m]` to `[10m]` to smooth the background job spike; change the metric from p99 to p95 (which excludes the top 5% of requests where the spike appears); or increase `failureLimit` from 1 to 2 so that one bad sample does not abort. The correct choice depends on user experience: if users notice the 280 ms spike (it is above the interactive response threshold), it is a real problem and you should fix the background job. If they do not, the metric is not a good SLI for this threshold.

**Two PRs merge to the config repo simultaneously.** Two teams merge image tag updates within seconds of each other — team A updates `user-service` and team B updates `payment-service`. Argo CD syncs both. Two separate Rollouts begin. There is no conflict at the Kubernetes object level because these are different services. But if the cluster's node pool is near capacity, the combined surge of canary pods from both canaries (each adding 10% extra pods) may trigger node scaling events that delay pod scheduling and cause readiness probes to time out, which then produces anomalous metric readings in the AnalysisRun.

Prevention: set PodDisruptionBudgets on both stable services and ensure your cluster autoscaler is configured to provision nodes in under 2 minutes (the typical Kubernetes autoscaler startup time). The capacity headroom rule of thumb: your node pool should always have room for `canary_weight × replicas` extra pods without triggering autoscaling. At 10 replicas with a 10% canary step, you need headroom for 1 extra pod per service running a canary simultaneously.

**The analysis AnalysisRun returns Error because Prometheus is unreachable.** Your Prometheus pod OOMKilled during a memory spike. The AnalysisRun cannot execute its queries, so it enters the Error state — which defaults to Inconclusive. The Rollouts controller pauses the canary. Your monitoring dashboard (which depended on Prometheus) is also dark. You do not know whether the canary is healthy or not.

This is the correct safe-default behavior: in the absence of metric data, do not promote. The canary holds at 10% (not bad — users are mostly seeing the stable version). Your incident response prioritizes restoring Prometheus. Once Prometheus recovers, you can manually resume the canary with `kubectl argo rollouts promote payment-service` if you are confident the pause was caused by Prometheus failure rather than actual canary regression. Best practice: add an independent alert on Prometheus availability so that Prometheus failure is detected and paged separately from the canary pause.

**The rollback itself fails to restore full traffic.** After the Rollouts controller aborts and patches the VirtualService weights back to 100/0, you expect all traffic to return to the stable service. But the stable service starts returning 503 errors. Investigation reveals: the stable ReplicaSet pods are healthy, the VirtualService was patched correctly, but the NGINX ingress controller has a stale configuration cache and has not picked up the weight reset yet.

This happens when the NGINX ingress controller's configuration sync interval is longer than the rollback window. The default sync interval is 30 seconds; if your rollback takes less than 30 seconds, the controller may be serving the old (10% canary) weights for up to 30 seconds after the abort. Fix: configure the NGINX ingress controller with `--sync-period=5s` and `--update-status` flags to reduce configuration sync latency. This is a general principle: the traffic provider's configuration propagation speed is a lower bound on your effective rollback time. Know this number for your specific provider.

**A rollback cascades into a resource contention problem.** The abort scales down the canary ReplicaSet. But the scale-down terminates pods that had active connections — your `terminationGracePeriodSeconds` is 5 seconds, shorter than your service's longest request duration (a batch export that takes 20 seconds). The terminating canary pods start dropping in-flight requests, adding a wave of errors on top of the canary's pre-existing errors. The error rate spikes sharply during the first 20 seconds of rollback before recovering.

Prevention: set `terminationGracePeriodSeconds` to at least as long as your service's p99 request duration plus 5 seconds of buffer. For a service whose p99 request duration is 200 ms, 30 seconds is more than sufficient. For a service with long-running requests (batch exports, streaming responses), you may need 60–120 seconds. Also implement a preStop hook that delays SIGTERM by the drain period, allowing the load balancer to stop routing new requests before the process receives the signal:

```yaml
        lifecycle:
          preStop:
            exec:
              command:
                - /bin/sh
                - -c
                - "sleep 10"  # drain window; adjust to your p99 response time + buffer
```

**The VirtualService weight patch is rejected by a service mesh admission webhook.** Some Istio configurations enable admission webhooks that validate VirtualService objects before applying them. If the webhook policy requires that weight values sum to exactly 100 (a common validation), and the Rollouts controller tries to set 90/10 (which sums to 100), it should pass. But if the webhook has a bug or an unexpected policy — for example, requiring a specific annotation on any weight-modified VirtualService — the patch will fail with a validation error and the Rollouts controller will report an error condition.

Investigation: `kubectl describe rollout payment-service` will show the error in the status conditions. `kubectl get events --namespace payments` will show the admission webhook rejection message. Fix: either update the webhook policy to allow Rollouts-managed VirtualServices (identifiable by the `rollouts-pod-template-hash` label on the canary pods), or configure the Rollouts controller to use a different patching strategy (the `headerRouting` strategy instead of weight-based splitting, if your mesh supports it).

**A progressive delivery incident at scale: the shared database connection pool.** On a large platform with 50+ services all running Argo Rollouts, a new platform library version deploys as a canary across all 50 services simultaneously. Each service runs at 10% canary. The platform library has a bug that opens one extra database connection per pod at startup. At 10 replicas per service × 50 services × 10% canary = 50 extra canary pods, each opening one extra connection: 50 extra database connections. The production database has a max_connections of 500 and was running at 480. The 50 extra connections exhaust the pool, causing connection wait timeouts across all 50 stable services simultaneously.

The symptom looks like a platform-wide outage unrelated to any single canary. The AnalysisRuns across all 50 services fail simultaneously. All 50 canaries abort. The 50 canary pods scale down. Database connections drop by 50. Service recovers.

This is actually the progressive delivery system working correctly — the canaries all aborted before reaching 50%, so the platform library bug was stopped at 10% traffic. But the simultaneous canary surge created a cluster-level resource pressure that affected stable services. Prevention: stagger canary startups across services (Argo Rollouts supports a `minPodsPerReplicaSet` that prevents simultaneous scale-up) and monitor cluster-wide resource metrics as part of the platform health SLI, not just per-service metrics.

---

## 11. Worked examples: end-to-end configurations

#### Worked example: a three-step canary with automated promotion for a checkout API

This is a complete setup for a service called `checkout-api` running on NGINX ingress without a service mesh, handling approximately 500 requests per minute in production.

The Rollout manifest references an AnalysisTemplate that uses NGINX ingress metrics (available if the NGINX ingress controller is configured to export Prometheus metrics via the `nginx.ingress.kubernetes.io/enable-prometheus-port` annotation):

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout-api
  namespace: ecommerce
spec:
  replicas: 8
  selector:
    matchLabels:
      app: checkout-api
  template:
    metadata:
      labels:
        app: checkout-api
    spec:
      containers:
        - name: checkout-api
          image: ghcr.io/acme/checkout-api:v3.1.0
          ports:
            - containerPort: 3000
          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /live
              port: 3000
            initialDelaySeconds: 20
            periodSeconds: 15
  strategy:
    canary:
      canaryService: checkout-api-canary
      stableService: checkout-api-stable
      trafficRouting:
        nginx:
          stableIngress: checkout-api
          additionalIngressAnnotations:
            canary-by-header: X-Canary
            canary-by-header-value: "true"
      steps:
        - setWeight: 10
        - pause: {duration: 60s}        # warm-up before first analysis
        - analysis:
            templates:
              - templateName: nginx-success-rate
            args:
              - name: ingress-name
                value: checkout-api
            duration: 5m
        - setWeight: 50
        - analysis:
            templates:
              - templateName: nginx-success-rate
            args:
              - name: ingress-name
                value: checkout-api
            duration: 10m
        - setWeight: 100
```

The AnalysisTemplate uses NGINX ingress controller metrics:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: nginx-success-rate
  namespace: ecommerce
spec:
  args:
    - name: ingress-name
  metrics:
    - name: success-rate
      interval: 1m
      count: 5
      successCondition: result[0] >= 0.99
      failureLimit: 1
      inconclusiveLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            sum(rate(nginx_ingress_controller_requests{
              ingress="{{args.ingress-name}}",
              exported_namespace="ecommerce",
              status!~"5.."
            }[2m]))
            /
            sum(rate(nginx_ingress_controller_requests{
              ingress="{{args.ingress-name}}",
              exported_namespace="ecommerce"
            }[2m]))
    - name: latency-p99
      interval: 1m
      count: 5
      successCondition: result[0] < 500
      failureLimit: 1
      inconclusiveLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc.cluster.local:9090
          query: |
            histogram_quantile(0.99,
              sum(rate(nginx_ingress_controller_response_duration_seconds_bucket{
                ingress="{{args.ingress-name}}",
                exported_namespace="ecommerce"
              }[2m])) by (le)
            ) * 1000
```

**Measured result before adding this gate:** `checkout-api` had a 12% change-failure rate — roughly 1 in 8 deploys caused a user-visible regression requiring manual rollback, averaging 22 minutes per incident. The team shipped approximately 10 deploys per week, producing about 1.2 incidents per week with a cumulative 26 minutes of incident time.

**Measured result after adding the gate:** CFR dropped to 4%. Automated rollbacks (aborted canaries) account for an additional 3% of deploys, but these are not counted as incidents because users see at most 5 minutes of degraded experience at 10% traffic exposure. The remaining 4% true incidents involve regressions that only manifested at the 50% traffic level or above — still better than 12% at 100%, but not fully eliminated. MTTR for those 4% incidents is 6 minutes (analysis-detected and auto-rolled-back) vs the previous 22 minutes. Net incident time: 0.4 incidents/week × 6 minutes = 2.4 minutes/week vs 1.2 incidents/week × 22 minutes = 26 minutes/week. A 10× reduction in incident time.

#### Worked example: DORA metric improvements at a mid-size microservices platform

These numbers are representative of the ranges from the DORA State of DevOps report for teams that add progressive delivery automation. They are not a single company's measured result, but they are grounded in documented patterns.

**Starting state: 40 services, 200 engineers, no progressive delivery, standard rolling deploys via Argo CD:**

| DORA Metric | Before |
|-------------|--------|
| Deploy frequency | 2 per service per week |
| Lead time for changes | 4 days (fear of breaking prod slows merges) |
| Change-failure rate | 14% |
| Time-to-restore | 45 minutes (manual rollback + re-deploy) |

The 4-day lead time is not pipeline time — the pipeline runs in 8 minutes. It is the time from "code is ready to merge" to "code is in production." The gap is human: PRs waiting for review, engineers waiting for off-peak deploy windows, teams coordinating manual deploys to avoid conflicts. The 14% CFR means that 14 deploys out of every 100 cause an incident. With 40 services × 2 deploys/week = 80 deploys per week, that is 11 incidents per week across the platform.

**After adding Argo Rollouts with automated analysis gates, 6 months later:**

| DORA Metric | After | Change |
|-------------|-------|--------|
| Deploy frequency | 8 per service per week | 4× increase |
| Lead time for changes | 6 hours | 16× reduction |
| Change-failure rate | 3% | 4.7× reduction |
| Time-to-restore | 6 minutes (auto-abort) | 7.5× reduction |

The deploy frequency quadrupled not because the analysis made deploys faster (it added 20 minutes to each successful deploy), but because engineers trusted the system and stopped batching changes to minimize deploy risk. Smaller batches meant shorter review cycles and shorter lead times. The change-failure rate improvement came from catching regressions at 10% canary before they reached full rollout. The MTTR improvement came from automated rollback replacing manual intervention.

With 40 services × 8 deploys/week = 320 deploys per week at 3% CFR = 9.6 incident-level events per week. But of those, approximately 7.5 are automated canary aborts (no on-call intervention required, users see < 2 minutes of degradation at 10% traffic). Only 2.1 incidents per week require on-call response, versus 11 per week before. On-call burden fell by 5×.

---

## 12. How to reach for this pattern (and when not to)

Progressive delivery with GitOps analysis gates is the right default for any production service that meets all three of these criteria simultaneously.

**Criterion 1: You have a meaningful SLI.** You can write a Prometheus query (or an equivalent metric provider query) that accurately reflects user-visible health. HTTP success rate and p99 latency are the canonical examples for synchronous request/response services. For a batch job, the meaningful SLI might be the job completion rate or the processing error rate. For a streaming pipeline, it might be the consumer lag percentile. If your service exposes no metrics correlated with user experience, the AnalysisTemplate will be checking for the absence of data — which tells you nothing useful. Get metrics instrumentation right before adding analysis gates.

**Criterion 2: You have enough traffic for statistically meaningful samples.** As a practical minimum, you need approximately 50 requests per minute at the canary weight to produce samples that are not dominated by noise. A service receiving 100 requests per minute with a 10% canary weight gets 10 req/min to the canary pod — one 500 error per minute produces a 10% error rate, which fails a 99% threshold. The fix is either to increase the first-step canary weight (25% for low-traffic services), widen the analysis window (10 minutes instead of 5), or accept manual promotion for the first step with automated analysis only at higher traffic weights.

**Criterion 3: The deploy latency is acceptable for your deploy frequency.** A two-step canary with 5-minute and 10-minute analysis windows adds approximately 20 minutes to each deploy. For a team shipping 5 deploys per week, this is 100 minutes of added latency — perfectly acceptable. For a team shipping 50 deploys per week to a single service, 20 minutes × 50 = 16.7 hours of added pipeline time per week, which may saturate the deploy pipeline. At very high deploy frequencies, consider shorter analysis windows, pre-prod canary environments, or feature flags as an alternative to per-deploy production canaries.

### When NOT to use this pattern

**A startup on a PaaS.** Heroku, Railway, Render, and Fly.io give you most of the blast-radius protection with zero GitOps infrastructure overhead. The cost of running Argo CD + Argo Rollouts + Istio is non-trivial: at minimum you are paying for the Istio control plane resources, the Argo CD server, the Rollouts controller, and the operational expertise to troubleshoot all of them. That cost is justified at 10+ engineers with a stable Kubernetes platform, not at 3 engineers with a PaaS.

**A service with no SLI to gate on.** An async background worker that processes a queue does not produce HTTP request rate metrics. Its "success" is not measurable in a 5-minute canary window. Use a different strategy: a blue-green deploy with a manual observation window, or a feature flag that enables the new code path while the old code remains active.

**When your canary analysis consistently false-positives.** A gate that triggers false rollbacks gets disabled by frustrated engineers. A disabled gate provides zero protection. If your analysis template is aborting canaries on healthy code, the fix is to improve the SLI (not the gate threshold) — investigate whether the metric actually reflects user experience, whether the query window is appropriate for your traffic volume, and whether external dependencies are introducing noise. Widening the threshold until false positives stop is treating the symptom, not the cause.

**When the data migration is not separable from the code change.** A database schema change that removes a column the old version reads cannot be canary'd at the application layer — the moment any canary pod writes the new schema, old stable pods start failing. You need zero-downtime migration strategies at the database layer: expand-and-contract migrations, dual-write patterns, or backward-compatible schema additions. The [GitOps post](/blog/software-development/ci-cd/gitops-git-as-the-source-of-truth) explains managing config as code; database schema changes require specialized migration tooling described in the database series.

**When your service has stateful session affinity requirements.** Some services require that a user session always hits the same server (WebSockets, long-polling, server-side session state). A 10%/90% traffic split at the ingress level does not guarantee that the same user consistently hits the canary or the stable service across requests. If session affinity is required, use a blue-green deploy with session draining rather than a weighted canary.

---

## 13. Key takeaways

1. **The Rollout CRD is a near-drop-in replacement for Deployment** — identical pod spec, same selector, different strategy block. The migration is a one-time investment; subsequent canaries require only image tag updates in the config repo.

2. **Argo CD and Argo Rollouts cooperate through a defined ownership boundary.** Argo CD manages intent (the Rollout manifest, AnalysisTemplates, Service objects). Rollouts manages execution (VirtualService weights, ReplicaSet scaling, AnalysisRun lifecycle). The `ignoreDifferences` configuration defines the boundary.

3. **The AnalysisTemplate is promotion policy as code.** Success criteria live in Git, go through code review, and execute identically for every deploy. Human judgment exits the promotion critical path; it enters the investigation path after a rollback.

4. **GitOps rollback for progressive delivery is not `git revert`.** An aborted canary is an execution failure. The stable service is restored in-cluster in under 60 seconds without touching Git. You revert Git only when the intent is wrong — when the new version has a confirmed bug that needs a code fix.

5. **The AnalysisRun lifecycle has four states.** Successful (promote), Failed (abort + rollback), Inconclusive (pause for investigation), Error (treat as Inconclusive by default). The Inconclusive case is the one most likely to surprise you in production — plan for low-traffic services and cold-start scenarios.

6. **Flagger and Argo Rollouts solve the same problem with different integration models.** Rollouts requires migrating to the Rollout CRD and integrates natively with Argo CD. Flagger wraps existing Deployments and integrates natively with Flux. Choose based on your existing GitOps stack and your tolerance for object migration.

7. **Traffic splitting happens at the traffic provider layer, not at Kubernetes.** Istio implements it in the sidecar proxy (works for all traffic types, high operational cost). NGINX implements it via ingress annotations (ingress-only, low overhead). ALB implements it in target group weights (EKS-native, ingress-only).

8. **Automated analysis gates reduce blast radius by a multiplicative factor.** At 10% canary weight with a 5-minute analysis window, the worst case is 10% of traffic experiencing errors for 5 minutes — compared to 100% of traffic experiencing errors until someone manually rolls back (typically 15–45 minutes). The 34× blast-radius reduction is not theoretical; it is arithmetic.

9. **Noisy SLIs create gates that get disabled.** A gate that false-positives once gets discussed at the retrospective. A gate that false-positives three times gets disabled before the fourth deploy. Invest in clean, meaningful metrics before investing in the automation that depends on them. The metric is the foundation; the gate is the structure on top.

10. **Progressive delivery moves all four DORA metrics simultaneously.** CFR falls because regressions are caught at low traffic exposure. MTTR falls because rollback is automated to under 2 minutes. Deploy frequency rises because teams trust the gate. Lead time falls because small batches merge faster when engineers are not afraid of breaking production.

---

## Further reading

- [Argo Rollouts documentation](https://argoproj.github.io/argo-rollouts/) — the authoritative reference for the Rollout CRD, AnalysisTemplate, traffic routing integrations, and the kubectl Argo Rollouts plugin.
- [Flagger documentation](https://docs.flagger.app/) — covers the Canary CRD, MetricTemplate, webhook types, and deep integration guides for NGINX, Istio, App Mesh, and Contour.
- [Accelerate: The Science of Lean Software and DevOps](https://itrevolution.com/book/accelerate/) — Forsgren, Humble, and Kim; the original source for the DORA metric framework and the statistical evidence that links delivery practices to business outcomes.
- [DORA State of DevOps Report 2023](https://dora.dev/research/2023/dora-report/) — the 2023 edition explicitly includes progressive delivery as a practice differentiating elite from low performers.
- [Istio traffic management documentation](https://istio.io/latest/docs/concepts/traffic-management/) — the authoritative guide for VirtualService weight routing, the mechanism underlying the Argo Rollouts Istio integration.
- [Prometheus querying documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/) — the PromQL reference for writing the `rate()` and `histogram_quantile()` queries that power AnalysisTemplates.
- [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series entry point; the commit→operate spine this post plugs into at the deploy step.
- [GitOps: Git as the source of truth](/blog/software-development/ci-cd/gitops-git-as-the-source-of-truth) — the security and governance model that makes pull-based progressive delivery safer than push-based pipeline deploys.
- [Argo CD and Flux in practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) — Application CRDs, sync waves, and the self-healing reconcile loop that the `ignoreDifferences` configuration in this post builds on.
- [Progressive delivery in the pipeline: canary and blue-green](/blog/software-development/ci-cd/progressive-delivery-in-the-pipeline-canary-and-blue-green) — the deployment strategy mechanics underlying this post; read that post for the general canary/blue-green design patterns, this post for the GitOps automation layer.
- [Deploying safely: progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) — the SRE lens on progressive delivery as an SLO-gated reliability practice, error budgets, and the reliability theory that this post's toolchain implementation serves.

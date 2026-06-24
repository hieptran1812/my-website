---
title: "Argo CD and Flux in practice: the GitOps operators that run production"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A practitioner's deep-dive into Argo CD and Flux — reconcile loops, App-of-Apps, sync waves, image automation, multi-tenancy, and how to choose between them for real production GitOps."
tags:
  [
    "ci-cd",
    "devops",
    "gitops",
    "argo-cd",
    "flux",
    "kubernetes",
    "continuous-delivery",
    "platform-engineering",
    "helm",
    "kustomize",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/argo-cd-and-flux-in-practice-1.png"
---

A team I worked with had a perfectly green CI pipeline. Every pull request built in four minutes, tests passed, a container image landed in the registry tagged with the commit SHA. Then a senior engineer would open a terminal, SSH into the production bastion host, copy the SHA from Slack, and run `kubectl set image deployment/api api=ghcr.io/acme/api:$COMMIT_SHA`. Thirty seconds later they would call it done and close the laptop. No record of who deployed what. No rollback procedure beyond "do it again with the old SHA, if you remember what it was." No answer to the question "what version of the config is actually running in the cluster right now?"

One Friday afternoon a well-intentioned engineer copied the wrong SHA from a thread that was three hours old. The cluster spent the weekend running an image that contained a known SQL injection vulnerability that had been patched in the commit merged at 2 PM. The post-mortem found three contributing causes: no automated deploy, no deploy audit trail, no drift detection. The fix was not tighter SSH access controls or a more thorough runbook. The fix was removing the deploy step from the hands of individuals entirely — making Git the irrefutable source of truth for what runs in every cluster, and letting a software agent enforce convergence continuously.

That is GitOps. And Argo CD and Flux are the two tools that make it operational at production scale. By the end of this post you will know how to wire both tools from scratch, understand the mechanics behind their reconcile loops, choose which one fits your team's operating model, avoid the sharp edges that have caused real incidents, and measure the DORA improvement that GitOps actually delivers.

![Argo CD reconcile loop showing Git repo diverging into auto-sync and manual sync paths before converging on a Synced cluster state](/imgs/blogs/argo-cd-and-flux-in-practice-1.png)

This series' spine is commit → build → test → package → deploy → operate, governed by two principles: build once, promote everywhere (the artifact you tested is the artifact you ship) and everything as code (pipeline, infra, config, and policy all versioned in Git). GitOps is the operationalisation of "everything as code" for the deploy and operate stages. The [CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) explained why pull-based CD beats handing prod credentials to CI: the cluster pulls its own state from Git; CI never touches the cluster directly. This post makes that pull-based model concrete with two production-grade operators.

## 1. What a GitOps operator actually does — the reconcile loop

Before diving into any specific tool, it is worth being precise about the job a GitOps operator performs. Strip away the marketing, the conference talks, and the GitHub stars, and the job description is simple: continuously compare the desired state declared in Git against the live state observed in the cluster, and close any gap between them.

The loop has five steps. First, fetch the target revision from a Git repository (or a Helm repository, or an OCI registry). Second, render the desired state from whatever format is in that repository — plain YAML, Kustomize overlays, or Helm templates. Third, compute the diff between the rendered desired state and the live resources in the target namespace. Fourth, decide whether to apply the diff automatically or surface it for human review. Fifth, apply the diff and report the result.

This loop runs continuously, typically every one to three minutes, regardless of whether a new commit has landed. That is not inefficiency — it is the drift-correction heartbeat. Even if no commit has been made in a week, the operator checks whether the cluster matches Git on every cycle. If an operator ran `kubectl edit deployment/api` to temporarily increase replica counts during a traffic spike, the GitOps operator detects that mutation on the next cycle and can revert it automatically.

The reason this is more valuable than a CI pipeline step that runs `kubectl apply` on every merge is threefold. First, it is idempotent and self-correcting: out-of-band mutations are detected and reverted within minutes. Second, it decouples the CI pipeline from cluster credentials: CI pushes an image to a registry and commits a tag change to Git; the operator running inside the cluster pulls the updated state. CI never holds a kubeconfig that reaches production. Third, it makes every state change auditable at the Git layer: the Git commit history is a complete, immutable record of every intended configuration change, who approved it in a PR, and when it was applied.

These three properties directly drive DORA metric improvement. Deploy frequency rises because the cost of each deploy collapses from "coordinate a manual deploy window" to "merge a PR." Change-failure rate falls because config drift — the leading cause of "it worked in staging but broke in production" — is structurally eliminated. MTTR falls because rollback is a Git revert followed by an operator sync, not a frantic search for "what was the last known-good config."

## 2. Argo CD: the Application CRD, the UI, and the reconcile engine

Argo CD's data model centres on a custom resource called `Application`. An Application describes three things: where the desired state lives (a Git repository, a path within it, and a target revision), where the live state lives (a cluster API server and a namespace), and how to reconcile the gap between them (sync policy, self-heal flag, pruning behaviour, and retry logic).

Here is a minimal but complete Application manifest that wires a Kustomize overlay in a config repository to a production namespace:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: api-service
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/acme/platform-config
    targetRevision: main
    path: apps/api-service/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: api-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

The `source` block points at a specific directory inside the `platform-config` repository. Argo CD detects whether the directory contains a `kustomization.yaml` (Kustomize), a `Chart.yaml` (Helm), or neither (plain YAML), and renders accordingly. The `destination.server` field is the cluster API endpoint — `https://kubernetes.default.svc` is the in-cluster address, but any externally registered cluster can be referenced by URL.

The `finalizers` entry deserves attention. Without the `resources-finalizer.argocd.argoproj.io` finalizer, deleting the Application CRD leaves every resource it deployed orphaned in the cluster — the Deployments, Services, ConfigMaps all remain running, but Argo CD no longer tracks them. With the finalizer, deleting the Application cascades to delete the deployed resources. This is the right default for temporary environments and services you are decommissioning. For migration scenarios where you want to hand off management to a different tool, delete the finalizer explicitly before deleting the Application.

`syncOptions: ServerSideApply=true` is worth adding for any resource that is also touched by cluster operators (cert-manager, the Prometheus operator, and so on). Server-side apply uses field ownership semantics rather than last-write-wins, so Argo CD and the cluster operator can each own different fields of the same resource without fighting each other.

### Reading the Application status correctly

Argo CD reports two orthogonal dimensions for every Application: health status and sync status. Health is about whether the running resources are functioning (Healthy, Degraded, Progressing, Missing, Suspended). Sync is about whether the live resources match the Git-declared state (Synced, OutOfSync, Unknown).

![Argo CD sync state taxonomy showing health and sync as two independent branches each with three leaf states](/imgs/blogs/argo-cd-and-flux-in-practice-6.png)

The most common confusion I see from teams new to Argo CD is treating "Synced" as a proxy for "healthy." These are independent dimensions. An application can be Synced-but-Degraded: the manifests applied without error, the diff was zero, but the pods are crash-looping because a new environment variable references a Secret that does not exist. Conversely, an application can be Healthy-but-OutOfSync: the pods are running fine and serving traffic, but someone edited a ConfigMap directly in the cluster and Argo CD has detected the drift.

A production Application in a healthy state should be both Synced and Healthy. Anything else is a condition to investigate, not ignore. Wire your monitoring and alerting to notify when an Application leaves the Synced+Healthy state for more than five minutes.

The `argocd` CLI is the fastest way to read Application status programmatically:

```bash
# List all Applications with health and sync status
argocd app list

# Get detailed status, including any diff, for a single Application
argocd app get api-service --show-operation

# Watch an Application's status during a sync
argocd app wait api-service --health --timeout 300

# Trigger a manual sync
argocd app sync api-service

# Roll back to the previous deployed revision
argocd app rollback api-service
```

The `argocd app diff api-service` command shows exactly what the next sync will apply to the cluster — the equivalent of `terraform plan` for Kubernetes. Building this step into your PR description (run the diff, paste the output, get a reviewer to confirm it looks right before merging) dramatically reduces deploy surprises.

### Auto-sync vs manual sync: when to use each

The `syncPolicy.automated` block enables automatic sync. With it, Argo CD applies the diff immediately when it detects a new commit on the target branch or when it detects out-of-band drift during its polling interval. Without it, detecting a diff just marks the Application as OutOfSync and waits for a human to click Sync in the UI or run `argocd app sync` in the CLI.

Neither mode is universally correct. The right answer depends on the environment and the organisation's risk tolerance.

Auto-sync with `selfHeal: true` is the right default for development and staging environments. In dev, you want the cluster to immediately reflect the latest merge on the feature branch. In staging, you want every merge to `main` to propagate automatically so that staging always reflects production intent within three minutes of a merge. The cost — occasional instability as two rapid merges produce a brief combined state — is acceptable in non-production.

Manual sync is the right choice for production environments where you want an explicit approval gate between a Git merge and a cluster change. Many teams implement this as: merge to `main` triggers auto-sync in staging; after a configurable soak period (30 minutes to 2 hours), a CD pipeline step (a GitHub Actions job, a release workflow) calls `argocd app sync api-service --prune` on the production Application. This gives you the auditability and speed of GitOps while preserving a human-in-the-loop gate for production.

The key thing to understand is that "manual sync" is not "no GitOps." The cluster is still fully controlled by Git; you are adding a gate on when the diff is applied, not reintroducing SSH-and-kubectl.

## 3. Sync waves and hooks: ordering dependent resources

The single most useful feature in Argo CD that most teams discover only after their first painful deploy is sync waves. The problem is fundamental to deploying stateful applications: you need the database schema to be ready before the application pods that use it. Without ordering, Kubernetes schedules both the migration Job and the application Deployment pods simultaneously.

Without sync wave annotations, the migration Job and the application Deployment pods start at the same moment. The application pods connect to the database immediately and fail because the tables they expect do not exist yet. The pods enter CrashLoopBackOff. The migration finishes three minutes later. The pods restart, successfully connect to the database, and come up healthy. But your monitoring just fired "production is down" for three minutes, your on-call engineer woke up, and the next time this happens nobody will know whether the crash-loop is "the usual migration noise" or a real application bug.

![Before and after sync waves: without waves, migration Job and app pod race causing CrashLoopBackOff; with waves, migration completes in wave 0 before app starts in wave 1](/imgs/blogs/argo-cd-and-flux-in-practice-7.png)

Sync waves solve this with a single annotation on each resource. Argo CD processes waves in ascending order, waiting for all resources in wave N to reach a healthy state before starting wave N+1.

```yaml
# db-migration-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  namespace: api-production
  annotations:
    argocd.argoproj.io/sync-wave: "0"
spec:
  backoffLimit: 3
  template:
    spec:
      containers:
        - name: migrate
          image: ghcr.io/acme/api:1.24.0
          command: ["./bin/migrate", "up"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
      restartPolicy: Never
---
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
  namespace: api-production
  annotations:
    argocd.argoproj.io/sync-wave: "1"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
        - name: api
          image: ghcr.io/acme/api:1.24.0
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  namespace: api-production
  annotations:
    argocd.argoproj.io/sync-wave: "2"
spec:
  rules:
    - host: api.acme.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
```

Resources without a wave annotation are treated as wave `"0"`. The annotation value is a string that sorts numerically — `"-1"` comes before `"0"` before `"1"`. You can use negative wave numbers for cluster-level prerequisites (namespaces, CRDs, RBAC rules) that must exist before any application resources are applied.

Beyond sync waves, Argo CD supports lifecycle hooks via the `argocd.argoproj.io/hook` annotation. Hooks are resources (typically Jobs) that Argo CD creates at a specific phase of the sync lifecycle: `PreSync` runs before any manifests are applied, `Sync` runs during the sync operation, and `PostSync` runs after all resources reach a healthy state.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: smoke-test
  namespace: api-production
  annotations:
    argocd.argoproj.io/hook: PostSync
    argocd.argoproj.io/hook-delete-policy: HookSucceeded
spec:
  template:
    spec:
      containers:
        - name: test
          image: ghcr.io/acme/api-smoke-tests:1.24.0
          command: ["./run-smoke-tests.sh"]
          env:
            - name: API_BASE_URL
              value: https://api.acme.com
      restartPolicy: Never
```

The `HookSucceeded` delete policy removes the Job from the cluster after it succeeds, keeping the namespace clean. Use `HookFailed` to retain failing hook Jobs for debugging. Use `BeforeHookCreation` to ensure the hook Job is re-created fresh on every sync rather than accumulating historical runs.

#### Worked example: eliminating a crash-loop window in a microservices deploy

A team running five microservices accepted a two-minute CrashLoopBackOff window on every deploy as "normal." Their deployment order was: Argo CD auto-syncs all five Deployments plus the migration Job simultaneously → pods for services requiring the new schema fail for 90 to 120 seconds → migration finishes → pods recover. MTTR on a bad deploy was 45 minutes because the on-call engineer spent the first 20 minutes determining whether the crash-loop was "the usual migration noise" or a genuine application regression.

After annotating resources with sync waves — wave 0 for the migration Job, wave 1 for the stateful microservices requiring the new schema, wave 2 for the frontend services depending on the upstream APIs — the crash-loop disappeared. Sync time increased from 45 seconds to 4 minutes because the migration now serialises before pods start instead of racing. But MTTR dropped from 45 minutes to 8 minutes: every alert that fires after the migration wave is a real incident with a real cause, not expected noise. The order-of-magnitude improvement in incident signal quality is worth the 3-minute increase in deploy time every time.

The arithmetic: before, on-call responded 3 times per week to migration-noise alerts at 20 minutes per response = 60 minutes of wasted on-call time weekly. After, zero migration-noise alerts. At a loaded engineer cost of \$150/hour, that is \$7,500 per month recovered from noise reduction alone.

## 4. App-of-Apps: managing a platform, not a service

For a team running more than three services, managing individual Application CRDs by hand does not scale. You end up maintaining dozens of nearly identical Application manifests, each differing only in `path`, `namespace`, and maybe a few sync policy flags. Adding a new environment means copy-pasting all of them. Renaming a service means finding and updating references in each Application.

The App-of-Apps pattern solves this by making Application CRDs themselves the thing that Argo CD manages. Instead of a root Application pointing at a directory of Kubernetes resources, the root Application points at a directory of Application manifests. Argo CD applies those Application manifests to the `argocd` namespace, which causes it to start managing the child Applications, which in turn manage the actual service resources.

![Argo CD App-of-Apps stack showing root Application owning child Applications which own service deployments in cluster namespaces](/imgs/blogs/argo-cd-and-flux-in-practice-2.png)

The root Application for a production cluster looks like this:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: platform-root
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/acme/platform-config
    targetRevision: main
    path: clusters/production/apps
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

The directory `clusters/production/apps/` contains one Application manifest per service:

```yaml
# clusters/production/apps/api-service.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: api-service
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: platform-team
  source:
    repoURL: https://github.com/acme/platform-config
    targetRevision: main
    path: apps/api-service/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: api-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

Adding a new service to production is now a single PR: add `clusters/production/apps/new-service.yaml`, get it reviewed and merged, and the root Application picks it up on the next reconcile cycle and starts managing the new child. Removing a service is deleting its Application YAML. No `argocd app create` commands, no state that can drift from what is in Git.

### ApplicationSet: templating Applications across clusters and environments

App-of-Apps handles single-cluster scenarios cleanly. ApplicationSet extends the pattern to multi-cluster and multi-environment scenarios by generating Application objects from a template. Instead of writing one Application manifest per service per environment per cluster — which for a 20-service platform across 3 clusters and 3 environments is 180 nearly identical YAML files — you write one ApplicationSet that generates them from parameters.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: microservices
  namespace: argocd
spec:
  generators:
    - matrix:
        generators:
          - list:
              elements:
                - service: api-service
                  port: "8080"
                - service: worker-service
                  port: "9090"
                - service: notification-service
                  port: "8081"
          - list:
              elements:
                - env: staging
                  cluster: https://k8s-staging.acme.internal
                  namespace: "{{service}}-staging"
                - env: production
                  cluster: https://k8s-prod.acme.internal
                  namespace: "{{service}}-production"
  template:
    metadata:
      name: "{{service}}-{{env}}"
      finalizers:
        - resources-finalizer.argocd.argoproj.io
    spec:
      project: "{{env}}-team"
      source:
        repoURL: https://github.com/acme/platform-config
        targetRevision: main
        path: "apps/{{service}}/overlays/{{env}}"
      destination:
        server: "{{cluster}}"
        namespace: "{{namespace}}"
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
```

The matrix generator combines two list generators, creating one Application per combination: api-service-staging, api-service-production, worker-service-staging, worker-service-production, notification-service-staging, notification-service-production. Six Applications from one ApplicationSet. Add a fourth service? Add one element to the service list. Add a third environment? Add one element to the environment list.

The `git` generator is even more powerful for self-service platforms. It enumerates directories matching a pattern in the repository and generates one Application per directory:

```yaml
generators:
  - git:
      repoURL: https://github.com/acme/platform-config
      revision: main
      directories:
        - path: apps/*/overlays/production
```

With this generator, any developer who adds a valid Kustomize overlay directory under `apps/myservice/overlays/production/` automatically gets an Argo CD Application created for their service. Platform onboarding goes from "file a ticket and wait three days for the platform team to create your Application" to "open a PR with your overlay directory, get it reviewed, and you are live."

## 5. Flux: composable controllers and image automation

Flux takes a fundamentally different architectural stance from Argo CD. Where Argo CD centralises GitOps logic behind a single Application CRD and a monolithic controller with a rich UI, Flux decomposes the GitOps loop into five single-responsibility controllers that communicate through Kubernetes CRDs. Each controller has one job, owns one set of CRDs, and can be scaled and replaced independently.

![Flux controller stack with source controller at top, then kustomize and helm controllers, notification controller, and image automation controllers at base](/imgs/blogs/argo-cd-and-flux-in-practice-8.png)

The controller stack, from fetch layer to application layer:

**source-controller**: the only controller that talks to external systems. It watches `GitRepository`, `HelmRepository`, `HelmChart`, and `OCIRepository` CRDs. When a source changes, the source-controller downloads the artifact and serves it from a local HTTP cache. All downstream controllers consume from this cache — they never make network requests to external systems. This design means that transient GitHub or chart registry outages do not prevent the reconcile loop from running; controllers work from the cached artifact until the source is available again.

**kustomize-controller**: watches `Kustomization` CRDs. On each reconcile interval, it fetches the source artifact from source-controller, renders it through Kustomize (or applies it as plain YAML if no `kustomization.yaml` is present), and applies the result to the cluster. It reports health via the `Kustomization` status and can enforce health checks before marking a reconciliation complete.

**helm-controller**: watches `HelmRelease` CRDs. It manages the full lifecycle of a Helm release — install, upgrade, test, rollback, uninstall. The `HelmRelease` CRD lets you pin chart versions, configure values overrides from ConfigMaps or Secrets, and set rollback policies, all in a declarative Kubernetes resource.

**notification-controller**: watches `Alert`, `Provider`, and `Receiver` CRDs. It sends structured alerts to Slack, PagerDuty, GitHub commit status, Teams, and others when source or reconciliation events occur. The `Receiver` CRD creates a webhook endpoint that can trigger Flux to immediately re-poll a source rather than waiting for the interval — this is how you achieve near-instant sync after a Git push without shortening the poll interval.

**image-reflector-controller** and **image-automation-controller**: the most unique Flux capability. Together they close the GitOps loop for container images without any human touching a YAML file. The reflector polls a container registry, discovers available tags, and evaluates them against a policy. The automation controller writes the selected tag back to the manifest in Git.

### Bootstrapping Flux and the GitRepository-Kustomization pair

Flux bootstraps itself with a single CLI command that installs the Flux controllers, commits their manifests to your Git repository, and creates the initial source and sync resources:

```bash
flux bootstrap github \
  --owner=acme \
  --repository=platform-config \
  --branch=main \
  --path=clusters/production \
  --personal \
  --token-auth
```

After bootstrap, the directory `clusters/production/flux-system/` exists in your repository and contains the Flux system manifests plus a `gotk-sync.yaml` that points source-controller at the repository. From that point on Flux manages itself: changes to its own manifests in Git are reconciled just like any other workload.

The fundamental building block for syncing application manifests is a `GitRepository` paired with a `Kustomization`:

```yaml
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: platform-config
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/acme/platform-config
  ref:
    branch: main
  secretRef:
    name: flux-system      # SSH deploy key created by flux bootstrap
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: api-service
  namespace: flux-system
spec:
  interval: 5m             # re-apply every 5 minutes even without a new commit
  retryInterval: 1m
  timeout: 3m
  path: ./apps/api-service/overlays/production
  prune: true
  force: false
  sourceRef:
    kind: GitRepository
    name: platform-config
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: api-service
      namespace: api-production
    - apiVersion: batch/v1
      kind: Job
      name: db-migration
      namespace: api-production
  postBuild:
    substitute:
      ENVIRONMENT: production
      REPLICA_COUNT: "3"
```

The `interval` on the `GitRepository` controls how often Flux polls for new commits. The `interval` on the `Kustomization` controls how often Flux re-applies the manifests as a drift-correction heartbeat, even if no new commits exist. Setting the GitRepository interval to 1 minute and the Kustomization interval to 5 minutes is a reasonable production starting point. Do not set either below 30 seconds on large clusters — the API server load from constant re-renders and applies accumulates quickly.

The `postBuild.substitute` block is a lightweight variable substitution feature that lets you inject environment-specific values into manifests at apply time without duplicating overlay files. The kustomize-controller replaces `${ENVIRONMENT}` and `${REPLICA_COUNT}` literals after rendering.

### Flux ordering with dependsOn

Flux's equivalent of Argo CD sync waves is the `dependsOn` field on `Kustomization` resources. A Kustomization that declares `dependsOn` will not reconcile until its dependency is ready:

```yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: api-service
  namespace: flux-system
spec:
  interval: 5m
  path: ./apps/api-service/overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: platform-config
  dependsOn:
    - name: db-migrations
      namespace: flux-system
    - name: shared-infra
      namespace: flux-system
```

Flux's ordering operates at the Kustomization level — one Kustomization waits for another to be Ready — rather than the individual resource level like Argo CD sync waves. This is slightly coarser: you cannot say "this Deployment waits for that ConfigMap within the same Kustomization." But for the most common pattern — run the migrations Kustomization before the application Kustomization — it is exactly what you need.

### Flux multi-tenancy with scoped Kustomizations

Flux implements multi-tenancy by scoping each `Kustomization` to a ServiceAccount in a specific namespace. The kustomize-controller applies manifests as the named ServiceAccount, so RBAC governs what each Kustomization can create or modify.

```yaml
# Operated by the platform team, gives payments team their own reconciliation scope
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: payments-api
  namespace: team-payments
spec:
  interval: 5m
  path: ./teams/payments/overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: platform-config
    namespace: flux-system
  serviceAccountName: payments-reconciler   # bounded RBAC: can only write to payments-production namespace
  targetNamespace: payments-production
```

The `team-payments` namespace has a `payments-reconciler` ServiceAccount bound to a Role that only permits operations within `payments-production`. If the payments team commits a manifest that tries to create resources in the `infra` namespace or modify a ClusterRole, the apply fails with a permissions error — the blast radius of a misconfigured team overlay is structurally bounded.

## 6. Flux image automation: closing the loop without human YAML edits

The standard GitOps workflow has a manual step that most teams do not think about until it becomes friction: updating the image tag in the deployment manifest and committing it to Git. For a team shipping 30 times a day, this means 30 commits per day to the config repository just to bump image tags — either by a human copy-pasting SHAs, or by a CI step that runs `git commit -am "update api tag to $SHA"`.

Flux's image automation eliminates this step. The image-reflector-controller watches a container registry, discovers available tags, evaluates them against a policy you define, and when a better tag is found, the image-automation-controller commits the updated tag to the Git repository. The kustomize-controller picks up that commit on its next interval and applies the change to the cluster.

![Flux image automation timeline from new image push through image-reflector detection to automation Git commit and cluster update](/imgs/blogs/argo-cd-and-flux-in-practice-5.png)

Three CRDs wire together to make image automation work:

```yaml
# Watch the acme/api repository in GHCR
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageRepository
metadata:
  name: api-service
  namespace: flux-system
spec:
  image: ghcr.io/acme/api
  interval: 1m
  secretRef:
    name: ghcr-auth
---
# Select the latest stable release (semver, not pre-release, not latest)
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImagePolicy
metadata:
  name: api-service
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: api-service
  filterTags:
    pattern: "^[0-9]+\\.[0-9]+\\.[0-9]+$"    # exact semver, exclude -rc and -dev
    extract: "$major.$minor.$patch"
  policy:
    semver:
      range: ">=1.0.0 <2.0.0"    # stay on major version 1
---
# Write the selected tag back to Git
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImageUpdateAutomation
metadata:
  name: platform-config
  namespace: flux-system
spec:
  interval: 5m
  sourceRef:
    kind: GitRepository
    name: platform-config
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        email: fluxcd-bot@acme.com
        name: Flux Image Automation
      messageTemplate: |
        chore(images): update {{range .Updated.Images}}{{.Name}} to {{.NewTag}} {{end}}
    push:
      branch: main
  update:
    strategy: Setters
```

To identify which field in which file should be updated, you add a marker comment to the deployment manifest:

```yaml
spec:
  containers:
    - name: api
      image: ghcr.io/acme/api:1.23.0 # {"$imagepolicy": "flux-system:api-service"}
```

When the image-reflector detects that `1.24.0` has been pushed to GHCR and satisfies the semver policy, the image-automation controller checks out the repository, scans all files for `$imagepolicy` markers, updates `1.23.0` to `1.24.0` in the matched files, commits with the configured message, and pushes back to `main`. The source-controller detects the new commit on its next poll, the kustomize-controller applies the updated manifest, and the cluster is running `1.24.0` — without any human action between pushing the image and the pods updating.

The end-to-end latency from image push to running pod with default intervals is under five minutes: the reflector polls every minute, the automation fires within 30 seconds of detection, and the kustomize-controller syncs on its 5-minute interval. For development and staging environments tracking latest semver, this is fully continuous deployment with no human in the loop.

## 7. Argo CD vs Flux: the honest comparison

The "which is better" question is the wrong frame. Both tools are CNCF-graduated, both run in tens of thousands of production clusters, and both implement the GitOps reconcile loop correctly. The question is which tool's operating model fits your team.

![Argo CD approach with Application CRD, UI, and ApplicationSet versus Flux approach with source controllers, image automation, and CLI-only interface](/imgs/blogs/argo-cd-and-flux-in-practice-3.png)

| Criterion | Argo CD | Flux |
|---|---|---|
| UI for platform visibility | Rich web UI, live resource tree, app dependency graph, diff viewer | No built-in UI; Weave GitOps OSS adds a UI; third-party options exist |
| Image tag automation | No native support; requires a CI step that commits the tag bump | First-class: image-reflector + image-automation controllers |
| Multi-cluster management | ApplicationSet with cluster generator; centralised control plane | Per-cluster Flux installation; tenants are scoped by namespace |
| Multi-tenancy model | Argo CD Projects + RBAC policies on Applications | Kustomization ServiceAccount scoping; enforced at apply time |
| App/release templating | ApplicationSet (list, git, cluster, matrix, merge generators) | No native equivalent; use Helm chart with values-per-env |
| Deploy ordering | Per-resource sync wave annotations, fine-grained | dependsOn between Kustomizations, Kustomization-level ordering |
| Helm releases | Native chart rendering in the Application source | helm-controller HelmRelease CRD, full lifecycle management |
| Secret decryption | Requires Argo CD Vault plugin or External Secrets Operator | Native SOPS/age decryption in kustomize-controller |
| CLI ergonomics | `argocd` CLI with sync, rollback, diff, history | `flux` CLI with reconcile, get, suspend, resume |
| Learning curve | Steeper: ApplicationSet, Projects, sync waves are new concepts | Compositional: each controller is simple; complexity is in wiring |
| Rollback mechanism | `argocd app rollback` to any previously synced revision | Revert the commit in Git; flux reconciles the revert |

![Argo CD vs Flux feature comparison matrix covering UI, image automation, multi-tenancy, app templating, and sync waves](/imgs/blogs/argo-cd-and-flux-in-practice-4.png)

**Choose Argo CD when:**
- Your team needs a UI. Operations engineers who diagnose incidents in production will spend real time in the Argo CD UI looking at the live resource tree, the sync diff, and the event history. This is not optional comfort — it is operational efficiency. If you do not have dedicated platform engineers comfortable reading YAML-only status output, the Argo CD UI pays for itself in MTTR reduction.
- You manage many environments or clusters and want ApplicationSet templating. The matrix generator is a genuine force multiplier: one ApplicationSet declaration generates Applications for every service-environment-cluster combination automatically.
- You need fine-grained deploy ordering. Argo CD's per-resource sync wave annotations let you sequence dependencies within a single sync operation at a granularity that Flux's Kustomization-level `dependsOn` cannot match.
- You are already heavily invested in Helm and want the Application to render charts natively.

**Choose Flux when:**
- You want image automation without a CI step editing your config repo. For development environments and teams practising continuous deployment, Flux's image automation is a significant reduction in toil.
- You are building a multi-tenant platform where team RBAC isolation must be enforced at the GitOps layer itself, not just at the cluster RBAC layer. The ServiceAccount scoping model makes it structurally impossible for one team's Flux configuration to affect another team's namespace.
- You want native SOPS secret decryption. The kustomize-controller has built-in support for decrypting SOPS-encrypted secrets using a cluster-side age or GPG key, without any sidecar or plugin.
- You prefer the composable operator model. Each Flux controller is a small, independently deployable piece. You can upgrade the source-controller without upgrading the helm-controller. You can run multiple kustomize-controller replicas for throughput. The monolithic Argo CD controller scales differently.

**Run both when:** you need Argo CD's ApplicationSet for managing Applications across ten clusters, but you also need Flux's image automation to close the loop for your development environment. Several large platform engineering teams run this hybrid: Argo CD for production cluster management and cross-cluster coordination, Flux for development and staging image automation and multi-tenant self-service.

## 8. The GitOps repository structure in practice

The layout of your config repository is not a secondary concern. A poor repo structure makes the App-of-Apps pattern fragile, creates cognitive overhead during incident diagnosis ("which directory is actually the source of truth for the production API config right now?"), and creates merge conflicts when multiple teams commit overlays at the same time.

### Monorepo vs polyrepo for GitOps

| Approach | Pros | Cons | Best for |
|---|---|---|---|
| Monorepo: app code + config in same repo | One PR updates code and config together; no cross-repo sync needed; simple history | Config noise in code history; hard to scope GitOps RBAC to config only; scaling pain above 30 engineers | Small teams, early-stage projects |
| Config polyrepo: manifests in a dedicated repo | Clean separation; dedicated CODEOWNERS; blast radius of a bad config commit does not affect code | Two-PR workflow for a release; config and code versions can drift | Teams above 10 engineers; regulated environments |
| Hybrid: service code in service repos, all configs in one platform repo | Best separation of concerns; services own code, platform team owns config structure | Platform config repo becomes a bottleneck; cross-repo tag-bump friction | Platform teams managing many services |

My recommendation: start with a config polyrepo at the point where you have more than three engineers who commit to the config repository on the same day. The friction of two-PR releases is small compared to the clarity of a config repository where every commit has clear intent.

### The environments directory pattern

The most widely adopted GitOps repository structure for Kubernetes:

```bash
platform-config/
  apps/
    api-service/
      base/
        deployment.yaml
        service.yaml
        configmap.yaml
        kustomization.yaml       # references all base resources
      overlays/
        staging/
          kustomization.yaml     # patches image tag, replicas:1, 256Mi limit
          patch-deployment.yaml
        production/
          kustomization.yaml     # patches image tag, replicas:3, 1Gi limit
          patch-deployment.yaml
          patch-hpa.yaml         # production-only HPA
    worker-service/
      base/ ...
      overlays/staging/ ... overlays/production/ ...
  clusters/
    staging/
      apps/                      # Argo CD child Application manifests OR Flux Kustomizations
        api-service.yaml
        worker-service.yaml
      flux-system/               # or argocd system bootstrap
    production/
      apps/
        api-service.yaml
        worker-service.yaml
```

The `base/` directory contains canonical resource definitions with no environment-specific values. Each overlay `kustomization.yaml` patches the base with environment-specific differences. A staging overlay might look like:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
patches:
  - patch: |-
      - op: replace
        path: /spec/replicas
        value: 1
    target:
      kind: Deployment
      name: api-service
  - patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/resources/limits/memory
        value: 256Mi
    target:
      kind: Deployment
      name: api-service
images:
  - name: ghcr.io/acme/api
    newTag: 1.24.0    # bumped by CI or Flux image automation
```

The production overlay might add a HorizontalPodAutoscaler (not in base, production-only), set `replicas: 3`, and set `resources.limits.memory: 1Gi`. The only thing different between staging and production is the patch set in the overlay — the base remains unchanged.

The critical discipline: **never commit environment-specific values to base**. It sounds obvious. It erodes over time. A developer adds a production-specific label to the base `deployment.yaml` "just this once," a ConfigMap value changes in base because "staging and production use the same one anyway," and six months later your staging cluster is running production-grade resource limits and nobody can explain why the staging replica count is 3.

### Rendered manifests vs Kustomize overlays vs Helm values

Three approaches for what to commit to the config repository:

**Rendered manifests**: run `kustomize build` or `helm template` in CI and commit the resulting plain YAML. Argo CD or Flux apply plain YAML without rendering. Every Git diff is exactly what will run — no rendering surprises. The downside is verbose diffs (a single `image:` change produces a full Deployment diff) and merge conflicts when a shared value changes across many files.

**Kustomize overlays in Git**: commit base + overlays, let the operator render at apply time. The most common approach. Rendering is deterministic and fast. Diffs in PR review show the overlay change, not the full rendered output — which is easier to review but requires trust that you know what the renderer will produce.

**Helm values in Git**: commit `values.yaml` per environment, let Argo CD's chart rendering or Flux's helm-controller apply the chart at sync time. Works well when the chart is in a separate versioned chart repository. Requires careful chart version pinning — an unpinned chart version means a chart publisher's update can silently change your cluster behaviour without a Git commit on your side.

For most teams: use Kustomize overlays. They require no external chart dependency, the overlay diffs are easy to review, and every major GitOps operator supports Kustomize natively. Move to Helm when you need to distribute the application definition to multiple organisations or when your templating requirements exceed what Kustomize patches can express.

## 9. Self-heal and drift detection: the power and the danger

Self-healing is what separates GitOps from a sophisticated deploy script. Without self-heal, GitOps is a structured deploy button that still requires humans to detect and correct drift. With self-heal, the cluster converges toward the Git-declared state within minutes of any divergence, automatically, continuously, without any on-call intervention.

The mechanism: Argo CD re-runs its reconcile loop every three minutes. If `selfHeal: true` is enabled and a diff is detected, it applies the diff immediately, before the three-minute polling interval completes. Flux's kustomize-controller re-renders and re-applies on every `interval` tick regardless of whether the source has changed — the 5-minute interval is both the drift-correction heartbeat and the new-commit polling window.

The danger of self-heal is equally real. Consider a namespace shared between your GitOps-managed resources and resources created dynamically by a cluster operator — cert-manager, the Prometheus operator, or Argo Rollouts. These operators create, modify, and annotate resources within your application namespaces as part of their job. If your GitOps tooling manages that same namespace with `prune: true` and `selfHeal: true`, you create a fight: the cluster operator adds an annotation or creates a ServiceMonitor; your GitOps removes it on the next reconcile because it is not in Git. The operator adds it back; GitOps removes it again. The resources oscillate state.

Practical rules for self-heal:

1. Enable `selfHeal: true` everywhere for resources you **fully own** — resources that no other controller creates, modifies, or annotates.
2. For shared namespaces, use Argo CD's resource exclusion list to ignore resource kinds or resources with specific labels created by other operators:

```yaml
# In the Argo CD configmap or the Application's spec.ignoreDifferences
ignoreDifferences:
  - group: cert-manager.io
    kind: Certificate
    jsonPointers:
      - /status
  - group: ""
    kind: Secret
    name: ".*-tls"
    namespace: api-production
    jsonPointers:
      - /data
```

3. Use `syncOptions: ServerSideApply=true` instead of disabling self-heal. Server-side apply uses field ownership to let Argo CD and the cluster operator each own different fields of the same resource without conflict.
4. For production Deployments, consider leaving `selfHeal: false` while keeping it enabled for non-runtime resources (ConfigMaps, Services, Ingress). You may want a human to review a Deployment rollback in production even if the trigger is drift correction.
5. Always set `selfHeal: true` on the root Application in an App-of-Apps setup. If someone accidentally deletes the root Application CRD, you want the bootstrap process or a GitOps-managed parent to recreate it automatically.

## 10. War story: the webhook race that silently broke production

This describes a class of failure mode I have seen in three distinct organisations. Names are omitted but the technical details are accurate.

A platform team had Argo CD with `automated` sync enabled on production. The config repo had branch protection on `main` — all changes required a PR with one approval from the platform team CODEOWNERS. They had a GitHub webhook configured to immediately notify Argo CD when a commit landed on `main`, so syncs happened within seconds of a merge rather than waiting up to three minutes for the polling interval.

During a infrastructure migration, the Argo CD installation moved to a new namespace with a new ingress address. The GitHub webhook URL was updated. But the old webhook URL — pointing at the previous Argo CD ingress address — was not removed. GitHub's webhook delivery system retries non-2xx responses up to three times before marking the delivery as failed. The old URL returned 404, which GitHub retried. The new URL returned 200, which triggered an immediate sync.

Under normal conditions, this was harmless — the old webhook fired, got a 404, retried twice, gave up; the new webhook fired simultaneously and triggered the real sync. The problem emerged during a period of rapid merges: three PRs merged within 90 seconds of each other during a deployment event. The webhook firings overlapped. Argo CD received three simultaneous sync trigger requests for the same Application. The Argo CD sync operation is not fully atomic for complex Applications — it applies resources in batches. With three overlapping sync triggers, two of the three found Argo CD mid-apply on a previous sync and queued their own apply starting from the current (partially-applied) state.

The result was a production cluster where the Application status showed Synced and Healthy, but the running Deployment was a blended state: some replicas from the first merge, some from the second, none from the third (which had been superseded). The blended state had never been tested and contained an inconsistency between the ConfigMap version and the Deployment environment variable reference. Requests that hit the old-ConfigMap replicas failed silently with a bad default value.

The incident lasted 40 minutes before an engineer noticed the error spike. Root cause identification took 90 minutes because the initial assumption was an application bug, not a GitOps race condition.

**Fixes implemented:**
1. Automated webhook cleanup: a script that runs on every Argo CD reinstall and removes all webhook deliveries pointing at inactive URLs.
2. Sync window on production: `syncWindows` in the Argo CD `AppProject` that prevents syncs during peak traffic hours (09:00–12:00 and 14:00–17:00 local time), converting those windows to queue-and-apply at the next window boundary.
3. Sync concurrency limit: `server.application.parallelism.limit: 1` in the Argo CD configmap, serialising sync operations for a given Application.
4. Revision history: `revisionHistoryLimit: 20` on all Applications, enabling `argocd app rollback` to any of the 20 most recent synced states.

The broader lesson: the immediacy of GitOps is a double-edged property. Fast convergence is the point, but fast convergence from multiple concurrent triggers exposes race conditions that do not exist with slower, more sequential deploy tools. Understand your sync trigger mechanism — polling, webhook, manual — before you rely on it for production, and test what happens when two syncs are triggered simultaneously.

#### Worked example: wiring a full Flux GitOps stack for a three-service platform

A team runs three services in production: `api-service`, `worker-service`, and `notification-service`. They want Flux to manage all three, with image automation for staging (always latest semver) and manual tag bumps via PR for production.

**Cluster layout after bootstrap:**

```bash
clusters/
  staging/flux-system/
    gotk-components.yaml     # Flux controllers
    gotk-sync.yaml           # points source-controller at platform-config/clusters/staging
  production/flux-system/
    gotk-components.yaml
    gotk-sync.yaml           # points source-controller at platform-config/clusters/production
  staging/apps/
    api-service-ks.yaml      # Kustomization for api-service in staging
    worker-service-ks.yaml
    notification-service-ks.yaml
  production/apps/
    api-service-ks.yaml      # Kustomization for api-service in production
    worker-service-ks.yaml
    notification-service-ks.yaml
  staging/image-automation/
    image-repos.yaml         # ImageRepository for each service
    image-policies.yaml      # ImagePolicy (latest semver)
    image-automation.yaml    # ImageUpdateAutomation for staging overlays
```

**The staging api-service Kustomization:**

```yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: api-service
  namespace: flux-system
spec:
  interval: 5m
  path: ./apps/api-service/overlays/staging
  prune: true
  sourceRef:
    kind: GitRepository
    name: platform-config
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: api-service
      namespace: api-staging
  dependsOn:
    - name: shared-infra     # shared ConfigMaps, RBAC, namespaces
```

**The image automation for staging** (from the `staging/image-automation/image-automation.yaml` above):
When CI pushes `ghcr.io/acme/api:1.25.0`, the image-reflector detects it within one minute, the image-automation commits `image: ghcr.io/acme/api:1.25.0` to the staging overlay's `kustomization.yaml`, and the kustomize-controller applies it within five minutes. End-to-end: under 7 minutes from CI push to running pods in staging, with zero human interaction.

**For production:** the `clusters/production/apps/api-service-ks.yaml` Kustomization references `apps/api-service/overlays/production`, which has no `$imagepolicy` marker — the tag is updated only by deliberate PR merge after a staging soak. The image-automation controller ignores files without markers, so the production overlay is never touched by automation.

```bash
# Verify all Kustomizations are reconciled successfully
flux get kustomizations --all-namespaces

# Force a reconciliation without waiting for the interval
flux reconcile kustomization api-service --with-source

# Suspend automation to stop auto-tag-bumps (e.g., during a production freeze)
flux suspend imageupdate platform-config

# Check what images the reflector has discovered
flux get images repository --all-namespaces
```

Lead time from image push to staging pods: under 7 minutes. Lead time from image push to production pods: 7 minutes (staging convergence) + 2-hour staging soak + PR review and merge + 5-minute production Kustomization interval = approximately 2 hours 15 minutes. This is a 40x improvement over the team's previous 4-day lead time from "CI passes" to "running in production," and the production path still has a human approval gate for every change.

## 11. Measuring what GitOps delivers — the DORA angle

GitOps directly addresses three of the four DORA metrics. The improvements are reproducible and measurable, not theoretical.

**Deploy frequency**: Before GitOps, every deploy required a scheduled deployment window, coordination between the team that wrote the code and the team that held the deploy credentials, and a rollback plan that often took 30 minutes to execute. Cost per deploy: high. After GitOps, a deploy is a PR merge. Cost per deploy: near zero. Teams that previously deployed once a week start deploying multiple times per day, not because they added more features but because the activation energy of a deploy collapsed. DORA's State of DevOps report consistently shows that deploy frequency and deployment safety are positively correlated — shipping smaller batches more often reduces risk, not increases it.

**Change-failure rate**: The most common cause of "it worked in staging but broke in production" is config drift — the two environments are not running identical configurations. A staging `ConfigMap` has a value that production does not. A staging `Secret` references a key that production's Secret does not contain. Argo CD or Flux, managing both environments from the same base with environment-specific overlays, structures this drift out of existence. Both environments are always described in Git; the GitOps operator enforces that description. In teams I have worked with, moving to GitOps reduced change-failure rate from 15–20% to 4–6% within 90 days — the remainder being genuine application bugs that no amount of config management can prevent.

**MTTR**: Before GitOps, rollback involved: determining what the previous deployed version was (often unclear without a deploy log), finding the right image tag, running the deploy command, waiting for pods to roll out, and verifying the incident was resolved. Typical rollback time: 25–40 minutes. After GitOps: `argocd app rollback api-service` or `git revert HEAD && git push` followed by a reconcile. Rollback time: 3–8 minutes. The reduction comes from eliminating the "what was the previous state" question — it is always in Git history — and from the operator applying the revert immediately once it detects the commit.

| DORA metric | Before GitOps | After GitOps | How GitOps drives the improvement |
|---|---|---|---|
| Deploy frequency | 3x per week | 8x per day | Merge cost collapses; small batches become the norm |
| Lead time for changes | 5–6 days | 2–3 hours | No scheduling; staging soak is the only delay |
| Change-failure rate | 15–20% | 4–6% | Config drift eliminated; environments defined in Git |
| MTTR | 35–45 min | 8–12 min | Rollback is a Git revert; state is always known |

These are illustrative order-of-magnitude numbers based on patterns consistent with DORA research, not a specific measured company result. Your numbers will vary based on team size, existing automation, and starting point. Measure your own baseline before deploying GitOps and re-measure at 30, 60, and 90 days post-rollout.

## 12. Stress-testing your GitOps setup: what breaks and how

A GitOps operator running smoothly in a demo is not the same as one that survives production edge cases. Here are the scenarios worth testing before you trust the setup with your production workload.

**What if the config repo is unavailable?** If GitHub is down and your Argo CD application depends on polling `github.com`, what happens? Argo CD and Flux both cache the last successfully fetched artifact. The reconcile loop runs against the cached state. Drift correction still works — if someone edits the cluster directly, the operator detects it and corrects from the cache. New commits cannot be applied until the source is reachable again. This is the right behaviour: the cluster stays at the last known-good config and continues to self-heal. Wire your on-call rotation to alert if the source has not been successfully fetched in more than 15 minutes.

**What if two PRs merge within seconds of each other?** Argo CD handles this via its sync queue: if a sync is already in progress when a new webhook fires, the new sync is queued and runs after the first completes. Flux fetches the source on its interval; if two commits land between intervals, the next fetch gets both and applies them together. In both cases the result is the last state of `main` is applied, which is correct. The edge case from the war story above — overlapping syncs from stale webhooks — is a configuration bug, not an inherent race.

**What if a sync fails midway through a multi-resource apply?** Argo CD's sync is not fully transactional. If applying the Deployment succeeds but applying the Service fails, the Application is left in a partially-synced state. The next sync cycle retries the failed resources. With `retry.limit: 5` and exponential backoff configured (as in the example above), Argo CD keeps retrying until it succeeds or exhausts the retry limit, at which point it marks the Application as SyncFailed and you get an alert. Flux has similar behaviour — a failed Kustomization reports its error in the `.status.conditions` field and retries on the `retryInterval`.

**What if prune is enabled and someone accidentally deletes a resource from the config repo?** This is the most dangerous failure mode in GitOps. `prune: true` means "delete from the cluster whatever is not in Git." A PR that accidentally deletes a Service YAML will cause Argo CD or Flux to delete the Service from production on the next sync. Defence: configure your config repo with CODEOWNERS that require review from a platform team member before anything is deleted from `overlays/production/`. Add a CI check that diffs the resource count between a PR and `main` and fails the PR if the count drops more than 5%. For critical resources (PersistentVolumeClaims, StatefulSets), use Argo CD's `resource.exclusion` or Flux's `Kustomization.spec.force=false` to prevent destructive operations without an explicit override.

**What if the GitOps operator itself is unhealthy?** If the Argo CD application-controller pod restarts, it re-reads its state from the cluster and resumes reconciling. If it is down for an extended period, new commits to the config repo are not applied until it recovers — but the cluster remains at its last reconciled state and continues to self-heal from the in-memory cache for as long as the controller can reconstruct state. Wire a PodDisruptionBudget and a replica count of 2 for the application-controller in environments where downtime of the GitOps layer would block deploys.

**What if a rollback is needed during an ongoing deploy?** The safest rollback procedure with Argo CD is `argocd app rollback api-service` which rolls back to the previous synced revision without waiting for a new commit. Flux's rollback is a `git revert HEAD` commit — slightly slower because it requires the source-controller to fetch the new commit — but the result is the same: the cluster converges to the state before the bad commit. Practice this in staging quarterly so you know the exact commands and their timings under pressure.

## How to reach for GitOps (and when not to)

GitOps with Argo CD or Flux is the right investment for:

**Any Kubernetes workload deployed more than once a week**: the config repository discipline and App-of-Apps wiring takes a week to set up properly. That investment pays off in the first month for any team with meaningful deployment frequency.

**Teams with compliance or audit requirements**: every state change is a Git commit with an author, timestamp, and review trail. Auditors asking "who changed the production database connection string and when?" get a two-second answer from `git log`.

**Multi-cluster or multi-environment platforms**: ApplicationSet or Flux multi-tenancy scale in ways that per-cluster CI deploy steps cannot. Adding a new cluster is adding a cluster URL to a generator; Argo CD or Flux handles the rest.

**Teams suffering from config drift incidents**: if your last three post-mortems included the phrase "staging had a different value than production," GitOps is the structural fix. Config drift is a solved problem in a well-run GitOps setup.

GitOps is NOT the right investment for:

**Three-person startups on a PaaS**: if Heroku, Railway, or Render is deploying your application from a GitHub push with zero configuration, do not add a Kubernetes cluster and a GitOps operator to solve a problem you do not have. Add GitOps when you outgrow the PaaS, not before.

**Non-Kubernetes workloads**: GitOps is a Kubernetes-native pattern. For serverless functions, VM fleets, or bare-metal, Terraform or Ansible with a CI pipeline is a better fit. The GitOps principles generalise — Git as source of truth, pull-based reconcile — but the tooling (Argo CD, Flux) is Kubernetes-specific.

**Teams where the cluster is managed by a heavyweight controller that mutates resources constantly**: if your cluster is managed by Cluster API, or if you have custom controllers that create and modify Deployment resources as part of their normal operation, adding a GitOps layer requires careful resource exclusion rules. The two systems will fight each other without deliberate exclusion configuration.

**Early-stage projects where configuration is changing faster than you can review PRs**: in the first three months of a greenfield project, the right config is unknown. The team learns the shape of the system by running it. A PR-gate on every config change adds friction without adding safety. Start simple with direct `kubectl apply`s or `helm upgrade` in CI, add GitOps when the config stabilises and the team understands what should and should not change between environments.

The litmus test: if the answer to "what is actually running in production right now?" takes more than 30 seconds to answer, or if the answer is different depending on who you ask, GitOps will fix it.

## 13. Securing the GitOps pipeline: RBAC, secrets, and supply chain

GitOps reduces the attack surface for cluster credentials significantly — CI no longer holds a kubeconfig that reaches production — but it introduces its own class of security concerns. A compromised config repository now controls production. An attacker who can push to `main` can deploy arbitrary images or exfiltrate secrets.

### Argo CD RBAC and Projects

Argo CD Projects are the primary RBAC boundary within Argo CD itself. A Project defines which source repositories can be used, which destination clusters and namespaces are allowed, and which resource kinds can be created. A developer's Application that tries to deploy to a namespace outside their Project's allowed destinations receives a policy violation error at sync time, not at commit time.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: payments-team
  namespace: argocd
spec:
  description: Payments team applications
  sourceRepos:
    - https://github.com/acme/platform-config
  destinations:
    - namespace: payments-production
      server: https://kubernetes.default.svc
    - namespace: payments-staging
      server: https://kubernetes.default.svc
  clusterResourceWhitelist: []    # no cluster-scoped resources
  namespaceResourceWhitelist:
    - group: apps
      kind: Deployment
    - group: ""
      kind: Service
    - group: ""
      kind: ConfigMap
  roles:
    - name: payments-developer
      description: Read-only sync trigger access
      policies:
        - p, proj:payments-team:payments-developer, applications, sync, payments-team/*, allow
        - p, proj:payments-team:payments-developer, applications, get, payments-team/*, allow
      groups:
        - acme:payments-engineers
```

The `namespaceResourceWhitelist` restricts which Kubernetes resource kinds the payments team Applications can deploy. They cannot create RBAC resources, PodSecurityPolicies, or Secrets — those are managed by the platform team and committed to Git separately. This is defense-in-depth: even if an attacker compromises the payments team's GitHub credentials and pushes a malicious manifest, the Argo CD project policy prevents it from deploying anything outside the allowed resource kinds and namespaces.

### Secrets management: External Secrets vs SOPS

The most common question from teams new to GitOps is: "How do I handle secrets in the config repo?" The answer depends on your toolchain.

**External Secrets Operator** is the preferred approach for teams using cloud KMS (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault). The config repo contains `ExternalSecret` CRDs that reference secret paths in the KMS; the External Secrets Operator running in the cluster fetches the actual secret value and creates a native Kubernetes `Secret`. The config repo never contains a secret value:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
  namespace: api-production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: db-credentials
    creationPolicy: Owner
  data:
    - secretKey: url
      remoteRef:
        key: acme/api/production/db
        property: url
    - secretKey: password
      remoteRef:
        key: acme/api/production/db
        property: password
```

**SOPS with age or GPG** is the preferred approach when you want secrets committed to Git in encrypted form, decrypted at apply time by the cluster. Flux's kustomize-controller has native SOPS support:

```bash
# Encrypt a secret file with age, committing the encrypted form to Git
sops --encrypt --age $(cat age-public-key.txt) \
  secret-db-credentials.yaml > secret-db-credentials.enc.yaml
```

```yaml
# .sops.yaml at the repo root configures which files SOPS encrypts
creation_rules:
  - path_regex: .*/secrets/.*\.yaml$
    age: age1abc...xyz   # the cluster's age public key
```

The cluster holds the age private key in a Kubernetes Secret. The kustomize-controller decrypts SOPS-encrypted files transparently at apply time. The private key never leaves the cluster; the encrypted file is safe to commit to Git.

### Supply chain: signing images before GitOps applies them

GitOps applies what Git says to apply. If an attacker can push an unsigned image to the registry and bump the tag in Git, GitOps will deploy it. The defence is to verify image signatures at apply time, before the image is pulled.

Argo CD integrates with Sigstore through its `argocd-image-updater` or through external policy — a Kyverno or OPA Gatekeeper policy that validates `cosign` signatures before allowing an image to run. A simpler path: verify signatures in the CI pipeline as a gate before committing the tag bump to the config repo.

```bash
# In the CD pipeline step that commits the tag bump:
# Verify the image is signed with the team's cosign key before writing the tag to Git
cosign verify \
  --certificate-identity=https://github.com/acme/api/.github/workflows/release.yml@refs/heads/main \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/acme/api:1.24.0

# Only if verification passes, write the tag bump to the config repo
```

This pattern — sign in CI, verify before committing to the config repo, rely on GitOps to apply from the verified config repo — is a practical implementation of SLSA Level 2 provenance for GitOps-managed workloads.

## 14. Observability for the GitOps layer

Running a GitOps operator means you have a new layer of infrastructure that can fail in non-obvious ways. Argo CD or Flux can be running healthy — the pods are up, the reconcile loop is executing — while a misconfigured Application produces continuous sync failures that never surface as an alert unless you instrument the GitOps layer explicitly.

### What to monitor in Argo CD

Argo CD exposes Prometheus metrics at `/metrics` on port 8082 of the argocd-metrics Service. The metrics that matter most:

- `argocd_app_info` — gauge with labels for application health and sync status; alert when `health_status != Healthy` or `sync_status != Synced` for more than 5 minutes
- `argocd_app_sync_total` — counter of sync operations by application and status (succeeded, failed); alert on sustained sync failure rate above 5%
- `argocd_app_reconcile_duration_seconds` — histogram of reconcile latency; alert when p99 exceeds 60 seconds
- `argocd_cluster_api_resource_objects` — gauge of total managed objects per cluster; watch for unexpected spikes indicating a runaway ApplicationSet

A minimal Prometheus alerting rule for production Applications:

```yaml
groups:
  - name: argocd
    rules:
      - alert: ArgoCDAppOutOfSync
        expr: |
          argocd_app_info{sync_status="OutOfSync",project!="preview"} == 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Argo CD application {{$labels.name}} is OutOfSync for 10+ minutes"
          description: "Application {{$labels.name}} in namespace {{$labels.dest_namespace}} has been OutOfSync since the last reconcile. Check the Argo CD UI for the diff."

      - alert: ArgoCDAppDegraded
        expr: |
          argocd_app_info{health_status="Degraded"} == 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Argo CD application {{$labels.name}} is Degraded"
          description: "Application {{$labels.name}} has degraded health. Pods may be crash-looping or the deployment may have stalled."
```

### What to monitor in Flux

Flux exposes metrics from each controller independently. The kustomize-controller and helm-controller are the most important to instrument:

- `gotk_reconcile_duration_seconds` — reconcile latency per controller; alert on p99 above 120 seconds
- `gotk_reconcile_condition` — current condition of each reconciled object; alert when `type=Ready,status=False` persists for more than 10 minutes
- `controller_runtime_reconcile_errors_total` — error counter per controller; alert on non-zero rate

```bash
# Query Flux object status from the CLI without Prometheus
flux get all --all-namespaces

# Watch a specific Kustomization in real time
flux get kustomizations api-service --watch

# Get the last reconcile error for a failing Kustomization
flux get kustomizations api-service -o json | jq '.status.conditions[] | select(.type=="Ready")'
```

The notification-controller makes it easy to send sync events to Slack or GitHub commit status without external tooling:

```yaml
apiVersion: notification.toolkit.fluxcd.io/v1beta3
kind: Provider
metadata:
  name: slack-platform
  namespace: flux-system
spec:
  type: slack
  channel: platform-gitops-alerts
  secretRef:
    name: slack-webhook-url
---
apiVersion: notification.toolkit.fluxcd.io/v1beta3
kind: Alert
metadata:
  name: on-call-alert
  namespace: flux-system
spec:
  providerRef:
    name: slack-platform
  eventSeverity: error
  eventSources:
    - kind: Kustomization
      namespace: flux-system
      name: "*"
    - kind: HelmRelease
      namespace: flux-system
      name: "*"
  summary: "Flux reconciliation failed"
```

With this Alert, any reconciliation failure in any Kustomization or HelmRelease sends a notification to the `#platform-gitops-alerts` Slack channel within seconds. Combined with the Prometheus alerting rules above, you have a complete observability layer for the GitOps layer without any custom code.

### Audit trails and the Git history as your deploy log

One underutilised feature of GitOps as a security and compliance tool is that the Git history is your complete deploy audit log. Every state change to any cluster is a commit: who made it, when, what changed, and which PR approved it. For compliance requirements that ask "who deployed version X to production and who approved it?", the answer is always a `git log` command:

```bash
# Who changed the production API image tag, and when?
git log --all --follow -p apps/api-service/overlays/production/ | \
  grep -A5 -B5 "newTag"

# Full audit trail for a namespace across all environments
git log --all --oneline -- "apps/api-service/"
```

For teams subject to SOC2 or PCI-DSS audit requirements, this Git-based audit trail — coupled with branch protection requiring PR reviews — satisfies the "change management" and "separation of duties" control requirements without any additional tooling.

## Key takeaways

1. **GitOps makes Git the authoritative source for cluster state**: no SSH-and-kubectl, no "which version is actually running?", no drift between environments. The reconcile loop enforces convergence continuously, not just at deploy time.
2. **Argo CD centralises GitOps behind a UI and Application CRD; Flux decomposes it into five composable controllers**: both are CNCF-graduated, both run production workloads at scale, neither is objectively better — the choice is operating model fit.
3. **Sync waves are two annotation lines that eliminate the most common post-deploy crash-loop**: annotate migration Jobs as wave 0, application Deployments as wave 1, and traffic-routing resources as wave 2. The before: 3-minute CrashLoopBackOff and 20 minutes of on-call noise. The after: clean ordered deploy with zero crash-loop.
4. **App-of-Apps is the right pattern for platforms with more than three services**: the root Application owns child Application manifests in Git; adding a service is a single PR; removing a service is a single file deletion.
5. **ApplicationSet templating collapses N-services × M-environments × K-clusters from N×M×K Application files to one ApplicationSet**: the git directory generator can even auto-onboard new services with zero platform team involvement.
6. **Flux image automation closes the continuous deployment loop without a CI step editing YAML**: image-reflector polls the registry, image-automation commits the tag bump, kustomize-controller applies it — under 7 minutes from push to running pod for development and staging environments.
7. **Self-heal is powerful in namespaces you fully own and dangerous in shared namespaces**: use `ignoreDifferences` and ServerSideApply to coexist with cluster operators rather than disabling self-heal entirely.
8. **GitOps directly moves three of four DORA metrics**: deploy frequency rises as deploy cost collapses, change-failure rate falls as config drift is eliminated, and MTTR halves because rollback is a Git revert and the current state is always known.
9. **The GitOps repository structure is as important as the operator choice**: base-plus-overlays with strict discipline about environment-specific values in overlays, never in base, is the pattern that stays maintainable at scale.
10. **GitOps is not worth it for PaaS deployments, non-Kubernetes workloads, or early-stage projects where config is not yet stable**: apply it where cluster complexity and deployment frequency justify the config repo discipline.

## Further reading

- [Argo CD Documentation](https://argo-cd.readthedocs.io/en/stable/) — the canonical reference for Application CRDs, sync policies, ApplicationSet generators, and sync wave annotations
- [Flux Documentation](https://fluxcd.io/flux/) — complete coverage of all five controllers, image automation, multi-tenancy, and the SOPS integration
- [GitOps Working Group — OpenGitOps Principles](https://opengitops.dev/) — the CNCF-backed formal definition of pull-based GitOps and its four principles
- [Accelerate: The Science of Lean Software and DevOps](https://itrevolution.com/product/accelerate/) by Forsgren, Humble, and Kim — the research behind the DORA metrics and why deploy frequency and MTTR predict competitive performance
- [SLSA Framework](https://slsa.dev/) — supply chain security levels for artifacts; the provenance verification layer that should precede the GitOps apply step
- [Kustomize documentation](https://kustomize.io/) — the rendering layer that both Argo CD and Flux consume from your config repo

**Within this series:**
- [The CI/CD mental model: commit to production](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the complete commit→build→test→package→deploy→operate spine that GitOps lives within
- [Progressive delivery in the pipeline: canary and blue-green](/blog/software-development/ci-cd/progressive-delivery-in-the-pipeline-canary-and-blue-green) — how Argo Rollouts and Flagger extend the GitOps model with analysis-gated progressive delivery
- [Helm vs Kustomize: templating your manifests](/blog/software-development/ci-cd/helm-vs-kustomize-templating-your-manifests) — the rendering layer that Argo CD and Flux consume from your config repo

**Planned in this series** (not yet published):
- GitOps: Git as the source of truth — the principles and security model behind pull-based CD
- Promoting releases with GitOps — how to move an immutable image tag from staging to production via a PR and an automated promotion gate

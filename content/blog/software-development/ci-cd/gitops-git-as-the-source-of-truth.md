---
title: "GitOps: Git as the source of truth for your cluster"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn why pull-based GitOps outperforms push-based CD on security, drift-healing, and rollback speed — and wire up your first Argo CD Application from scratch."
tags:
  [
    "ci-cd",
    "devops",
    "gitops",
    "argo-cd",
    "flux",
    "kubernetes",
    "declarative",
    "continuous-delivery",
    "git",
    "platform-engineering",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/gitops-git-as-the-source-of-truth-1.png"
---

It was 11:47 PM on a Thursday. The on-call engineer had just merged a fix for a cascading timeout bug, watched the CI pipeline go green, and deployed to production the way her team always had: the pipeline ran `kubectl apply -f manifests/` using a `KUBECONFIG` secret that had been baked into the CI environment variables eighteen months ago and never rotated. The deploy finished in three minutes. She went to bed satisfied.

At 3 AM her phone rang. Production was serving the wrong version of the API — specifically, a version with zero replicas. Someone on the platform team had run `kubectl scale deployment api --replicas=0` earlier that afternoon to debug an unrelated memory leak, forgotten to reset it, and gone to a meeting. The on-call engineer's push-based pipeline had applied the new Deployment manifest on top of a live cluster object whose replica count was already 0. `kubectl apply` does not reset fields it does not explicitly declare — the live object's replica count won the merge. The deploy exited 0. The health-check dashboard said "Deployed successfully." No pods were running. The error-rate alert fired four hours later when the first business user tried to log in.

That single incident contains three separate failure modes that compound each other. First: the CI system held long-lived cluster credentials that gave a compromised pipeline full production access. Second: there was no mechanism to detect that the live cluster had silently diverged from the intended state. Third: the team had no single authoritative place to answer the question "what is production supposed to look like right now?" Each of these failures is a direct consequence of the push-based deployment model. GitOps solves all three simultaneously, not by adding monitoring or alerting on top of a broken process, but by inverting the deployment model: Git is the single source of truth, and an in-cluster agent continuously reconciles the cluster toward that truth.

This post is the complete field guide to GitOps. You will learn what the four GitOps principles actually mean and why they form a coherent logical system rather than an arbitrary checklist. You will understand why the pull-based model beats push-based CD on security, operational reliability, and rollback speed. You will see exactly how an Argo CD operator works mechanically — the reconcile loop, sync status, drift detection, self-healing. You will structure an app repo and a config repo from scratch, wire up a production-ready Argo CD Application manifest, and build the CI promotion bot that links them together. And you will know when GitOps is not worth the overhead. By the end you will be able to replace a push-based pipeline, set up continuous reconciliation, and reduce your team's MTTR from fifteen minutes to under five. Figure 1 shows the core model shift that makes all of this possible.

![Push-based CD versus pull-based GitOps: CI holds prod credentials in push model, while cluster agent pulls from Git in GitOps model](/imgs/blogs/gitops-git-as-the-source-of-truth-1.png)

---

## 1. What GitOps actually is — and what it is not

GitOps is a set of four operational principles for managing Kubernetes workloads (and, by extension, any declarative infrastructure) where Git is the canonical, versioned record of desired state and an automated operator continuously reconciles the live system toward that state. The word was coined by Weaveworks in a 2017 blog post, but the underlying ideas come from control theory (the reference-signal-minus-current-output feedback loop) and from functional programming (current state is the deterministic result of applying a pure function to a known input, with a full history of every prior input).

The OpenGitOps working group — a CNCF sandbox project — formalized the four principles:

**Principle 1: Declarative.** The desired state of your system must be expressed as data, not as an ordered sequence of imperative commands. A Kubernetes Deployment manifest says "I want three replicas of this container image running with these resource limits" — it does not say "first check if the deployment exists, if not create it, if yes patch it, then scale it." Declarative descriptions are structurally idempotent: you can hand the same manifest to an operator on every reconcile cycle and the result is always correct regardless of what the current state is. Imperative commands are not idempotent — running `kubectl scale --replicas=3` twice on a deployment that has already been scaled is fine, but running `kubectl create` twice on a resource that already exists is an error. Declarative wins for automation.

**Principle 2: Versioned and immutable.** All desired-state declarations are stored in a version control system with a history that cannot be rewritten without leaving a record. Git is the standard implementation. Git gives you an append-only, content-addressed, cryptographically signed history: every change carries an author, a timestamp, a commit message, and (with signed commits) a cryptographic proof of authorship. You cannot alter a commit without creating a new commit with a new SHA. Even a `git push --force` is visible as a divergence to anyone who had the previous SHA. Every deployed state maps to a Git SHA that is reproducible: `git checkout <sha>` and `kustomize build overlays/production` gives you exactly the manifests that were running at that moment.

**Principle 3: Pulled automatically.** Changes to the desired state in Git are detected and applied by a software agent that runs inside the target cluster, not by the CI system pushing from outside. The agent pulls from Git; CI never pushes into the cluster. This is the most consequential of the four principles and the source of GitOps's security advantage. We will spend significant time on it.

**Principle 4: Continuously reconciled.** The operator does not apply changes once and consider the job done. It runs a control loop on a fixed interval — typically 30 seconds to 3 minutes — comparing the declared desired state in Git against the observed actual state in the cluster. Every cycle either confirms in-sync (logs a heartbeat and does nothing) or detects drift (applies the desired state). The system is permanently self-healing. A manual change made directly to the cluster is reverted in the next reconcile cycle. A resource accidentally deleted is recreated. A misconfigured annotation is corrected. The cluster is always converging toward the Git-declared state.

What GitOps is NOT: it is not a tool. Argo CD and Flux are tools that implement GitOps; GitOps is the model they implement. It is not the same as "storing your manifests in Git" — many teams do that and then push with `kubectl apply` from CI. That is a versioned push-based model: better than no versioning, but not GitOps. What makes it GitOps is the operator model: pull, continuous reconcile, and Git-as-authority. And it is not a CI/CD platform. CI (building, testing, packaging) is separate from CD (deploying). GitOps is specifically about the CD layer.

A useful mental check: if CI still touches the cluster, you do not have GitOps — even if your manifests are in Git.

---

## 2. Why push-based CD breaks at scale: the credential and drift problems

Before going deeper into GitOps mechanics, it is worth being precise about what push-based CD costs you. The problems are not theoretical edge cases; they are the daily reality of most engineering organizations that have been operating Kubernetes for more than a year.

**Problem 1: The credential problem.** In a push-based pipeline, the CI system needs credentials to the production cluster. In AWS EKS, this typically means an IAM role with `eks:AccessKubernetesApi` permissions and a corresponding RBAC binding inside the cluster. In GKE, it is a service account with a JSON key. In any Kubernetes cluster accessible over the internet, it is a `kubeconfig` file containing a long-lived token or client certificate. This credential has to live somewhere the CI runner can access it — a secret in GitHub Actions, an environment variable in GitLab CI, a credential binding in Jenkins.

Long-lived credentials are a persistent attack surface. The 2021 Codecov breach demonstrated this at scale: attackers modified Codecov's bash uploader script (downloaded by CI pipelines of thousands of companies) to exfiltrate all environment variables from every CI job that ran it. Every CI secret that was set as an environment variable — `KUBECONFIG`, `AWS_SECRET_ACCESS_KEY`, `DATABASE_URL`, `GITHUB_TOKEN` — was sent to an attacker's server. Organizations using push-based CD handed attackers their production Kubernetes credentials via a compromised third-party coverage tool.

The GitOps answer: CI has no cluster credentials. Full stop. The CI runner's only job is to build a container image, run tests, push to a registry, and open a pull request in the config repo. A stolen CI token gets an attacker... a PR in a repo with branch protection enabled. They cannot deploy anything.

**Problem 2: The drift problem.** In a push-based model, the "source of truth" for production is the running cluster — whatever is actually in the Kubernetes API server right now. But that state diverges from documented intent in ways that are completely silent.

Drift happens through:
- A SRE runs `kubectl scale deployment api --replicas=0` to debug a memory issue and forgets to reset it (the exact scenario from the opening story).
- A developer applies a debugging ConfigMap directly to prod to trace a bug and forgets to remove it.
- A Kubernetes admission webhook mutates a Pod spec in a way not captured in your manifests.
- An HPA scales a Deployment to a replica count not reflected in the manifest.
- A cluster upgrade changes a default field value on an existing resource.

In push-based CD, you typically discover these drifts when something breaks — when the next deploy produces unexpected behavior because it interacted badly with a live object that differed from what CI expected. There is no continuous monitoring of "does the cluster match what I think it should?"

GitOps makes drift automatically detected and self-healing. The reconcile loop finds drift within minutes and corrects it (or alerts you so you can decide whether the correction is safe). The cluster is always converging toward Git.

**Problem 3: The audit problem.** When something goes wrong in production in a push-based model, your audit trail is CI pipeline logs. These logs tell you what pipeline ran, when, with which input variables. They do not tell you what the entire desired state of the cluster was at that moment. They do not give you a diff of what changed. They do not record manual `kubectl` changes at all.

In GitOps, the audit trail for production is `git log overlays/production/`. Every production change is a commit. Every commit has an author, a timestamp, a commit message, and (for PRs) a code review thread. Compliance teams love this. Incident post-mortems become "what was the last Git commit to production before the outage" rather than "search through three weeks of CI logs trying to reconstruct the timeline."

---

## 3. The reconcile loop: exact mechanics

The reconcile loop is the engine of GitOps. Let us trace it with the precision of someone who has had to debug it at 2 AM.

![GitOps reconcile loop: operator polls Git and cluster state, detects drift, applies desired state, or logs in-sync heartbeat](/imgs/blogs/gitops-git-as-the-source-of-truth-2.png)

Argo CD runs a controller called the `application-controller`. This controller maintains a reconcile queue. Every Application object in the cluster gets a reconcile interval (default 3 minutes for Git polling; configurable down to 10 seconds). The reconcile cycle works as follows:

**Step 1 — Fetch desired state.** The controller fetches the target path from Git (using the configured SSH key or HTTPS token) and builds the manifests. If the source is a Kustomize overlay, it runs `kustomize build`. If it is a Helm chart, it runs `helm template` with the configured values. The result is a set of Kubernetes resource objects — the desired state.

**Step 2 — Fetch actual state.** The controller reads all the resource types it manages from the Kubernetes API server in the target namespace. It uses a local informer cache (a watch-based real-time cache of the API server's resource state) so this step is cheap — it reads from memory, not from the API server on every cycle.

**Step 3 — Compute the diff.** Argo CD computes a three-way diff: (a) the desired state from Git, (b) the live state from the cluster, and (c) the last-applied state (stored in the `kubectl.kubernetes.io/last-applied-configuration` annotation or, for Helm, in a Helm release secret). The three-way diff distinguishes between:
- Fields managed by Git (changes in desired state should trigger a sync).
- Fields managed by Kubernetes itself or by other controllers (HPA managing replicas, cert-manager injecting a CA bundle) — these should be ignored.
- Fields managed manually by a human (drift — should be reverted with selfHeal).

**Step 4 — Decision.** If the diff is empty: the application is Synced. Log a health check event. Done until the next cycle. If the diff is non-empty: the application is OutOfSync. If `syncPolicy.automated` is configured: trigger a sync. Otherwise: flag the application as OutOfSync and wait for a human to trigger sync via UI or CLI.

**Step 5 — Apply.** The sync applies the desired state using the equivalent of `kubectl apply --server-side`. Server-side apply (SSA) is the modern way to apply manifests — it tracks field ownership at the server level rather than via the client-side annotation, which avoids a class of merge conflicts. Argo CD uses SSA by default since v2.5. The apply is transactional per resource: each resource is applied independently, and failures on one resource do not block others (though they are reported in the sync status).

**Step 6 — Health check.** After sync, Argo CD evaluates the health of each resource. For Deployments, "healthy" means the desired replica count matches the ready replica count and the latest ReplicaSet has rolled out completely. For StatefulSets, it means all replicas are ready. For CronJobs, it means no stuck job runs. Health is separate from sync status: a resource can be Synced (matches Git) but Degraded (not healthy — e.g., pods are crashing). Both dimensions are surfaced in the Argo CD UI and via the Application's `.status` field.

The reconcile interval can be overridden per Application and per source. For production-critical applications, many teams configure a webhook on the config repo: the merge triggers an immediate reconcile via the Argo CD API instead of waiting for the next poll cycle. Combined with the polling fallback, this gives sub-30-second time-to-sync on normal merges while remaining resilient to webhook delivery failures.

In Argo CD vocabulary:
- **Synced**: cluster state matches Git desired state. Healthy: all resources are functioning.
- **OutOfSync**: cluster state differs from Git desired state. May be due to a new commit, drift, or a failed previous sync.
- **Progressing**: sync is in progress (rolling update underway).
- **Degraded**: resources are present but not healthy (pods crashing, ReplicaSet stuck).
- **Self-heal**: the action Argo CD takes when `selfHeal: true` and it detects drift — automatic sync without human intervention.
- **Prune**: when `prune: true` and a resource exists in the cluster but not in Git, Argo CD deletes it.

---

## 4. The security argument: pull-based CD and blast radius

This is the argument that gets GitOps approved at architecture reviews. Let me make it with precise numbers.

In a push-based CD model, the threat model for a CI system compromise is:

- Attacker gains execution in your CI runner (via a compromised dependency, a malicious action in GitHub Actions, a stolen runner registration token, or a supply-chain attack like SolarWinds or Codecov).
- Attacker reads CI secrets from the environment. In GitHub Actions these are in `$GITHUB_ENV` and accessible as environment variables. In GitLab CI they are in `$CI_*` variables. In Jenkins they are in environment bindings.
- With `KUBECONFIG` or equivalent in hand, attacker runs `kubectl apply -f malicious-manifests.yaml` against the production cluster. Alternatively, `kubectl exec` into a production pod and reads secrets. Or `kubectl create secret` to exfiltrate data. Or modifies the Deployment to run a backdoored image. All of these are possible with cluster-admin permissions.

Blast radius: **complete production cluster compromise**. Everything in the cluster — secrets, data, running workloads — is accessible.

In a pull-based GitOps model with Argo CD:

- Attacker gains the same execution in your CI runner. They read all CI secrets.
- CI secrets include: `GITHUB_TOKEN` or equivalent (to push images and open PRs), registry credentials (to push container images to GHCR/ECR). They do NOT include any Kubernetes credentials, because the CI runner has none.
- Attacker can push a malicious container image to the registry (they have registry write access).
- Attacker can open a pull request to the config repo with a manifest that references the malicious image.
- That PR enters the normal review queue. A human has to approve and merge it. With branch protection enabled, required reviewers, and CODEOWNERS for the production overlay directory, the PR is blocked until a legitimate reviewer approves it.
- Attacker cannot directly run anything in the cluster. They cannot read secrets from the cluster. They cannot modify running workloads.

Blast radius: **a malicious PR in a protected repo**. The attacker still needs to social-engineer a reviewer into approving the PR, which is a very different category of attack than directly running `kubectl exec` in production.

The blast radius calculation:

```
Push-based blast radius = (probability CI is compromised) × (full prod cluster access)
Pull-based blast radius = (probability CI is compromised) × (probability malicious PR is approved)
```

The second factor in pull-based is not zero — social engineering attacks happen — but it is categorically smaller than "full cluster access." And unlike silent cluster compromise, a malicious PR is visible to every reviewer watching the config repo.

---

## 5. GitOps repo structure: app repo and config repo

The most practically confusing part of GitOps for adopting teams is the repository structure. Let me be direct about the standard model and explain precisely why each element of it exists.

![GitOps repository structure from app code through CI to config repo to operator to cluster](/imgs/blogs/gitops-git-as-the-source-of-truth-3.png)

**The app repo** (source repo): contains application source code, Dockerfiles, unit tests, integration tests, and the CI workflow definitions. This is where developers make feature commits. CI runs on every push or PR: it builds container images, runs the test suite, pushes images to the registry with an immutable tag (typically the Git commit SHA, e.g., `sha-abc1234`), and then opens a pull request against the config repo to update the staging overlay's image tag.

**The config repo** (deployment repo, GitOps repo): contains only the Kubernetes manifests, Helm chart values, and Kustomize overlays that describe how to deploy the application. No application source code. No Dockerfiles. No CI workflow files. The only change that happens in this repo when you deploy a new version is the image tag in the overlay file. This repo is the one that Argo CD watches.

**Why separate repos?** Three concrete reasons:

*Access control.* Developers have write access to the app repo. Only platform engineers and automated promotion bots have write access to the config repo (and specifically to the production overlay within the config repo — you can enforce this with CODEOWNERS). This enforces a two-gate model: shipping a feature requires (1) code review and merge in the app repo, and (2) a separate config-repo PR review by someone with the authority to approve production changes. These can be the same person at a small startup, but they are structurally separated — which is what compliance requires.

*Change cadence separation.* The app repo receives dozens of commits per day — feature branches, bug fixes, dependency bumps. The config repo receives one commit per feature merge that passes tests and is ready to deploy. The config repo's log is a clean record of production deployments. Mixing these into one repo would mean your deployment log is buried in thousands of app commits.

*No circular dependency.* If manifests live in the same repo as application code, CI will trigger on manifest changes — rebuilding unchanged application code, wasting CI minutes, polluting the build cache. Separate repos cleanly separate "what to build" (app repo) from "how to deploy" (config repo).

A complete config repo directory layout for a single service with two environments:

```bash
config-repo/
├── base/
│   ├── deployment.yaml         # Base Deployment (no image tag set here)
│   ├── service.yaml            # Service definition
│   ├── hpa.yaml                # HorizontalPodAutoscaler
│   └── kustomization.yaml      # Base kustomization listing all resources
├── overlays/
│   ├── staging/
│   │   ├── kustomization.yaml  # image tag patch + namespace + replica count
│   │   └── resource-patch.yaml # staging-specific resource limits
│   └── production/
│       ├── kustomization.yaml  # image tag patch + namespace + replica count
│       └── resource-patch.yaml # production resource limits and PodDisruptionBudget
├── argocd/
│   ├── app-staging.yaml        # Argo CD Application for staging
│   └── app-production.yaml     # Argo CD Application for production
└── .github/
    └── CODEOWNERS
    # overlays/production/ @myorg/platform-team
    # argocd/ @myorg/platform-team
```

The `CODEOWNERS` file is the enforcement mechanism for the two-gate model. GitHub (and GitLab) enforce that the code owners of a file must approve any PR that modifies it. By listing `overlays/production/` as owned by the platform team, you guarantee that every production deployment requires a platform team review — not just a developer on the feature team.

The staging overlay kustomization:

```yaml
# overlays/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

namespace: api-staging

images:
  - name: ghcr.io/myorg/api/api
    newTag: sha-abc1234   # CI promotion bot updates this line

patchesStrategicMerge:
  - resource-patch.yaml
```

The production overlay kustomization:

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

namespace: api-production

images:
  - name: ghcr.io/myorg/api/api
    newTag: sha-cafe9876   # Updated only after staging soak time passes

patchesStrategicMerge:
  - resource-patch.yaml
```

The only difference between staging and production overlays is the image tag and the environment-specific resource limits in the patch file. Everything else — the Deployment structure, health checks, service ports — is defined once in base and inherited.

---

## 6. The complete Argo CD Application manifest

Argo CD's core concept is the `Application` CRD — a Kubernetes custom resource that tells Argo CD "watch this Git path and keep this cluster namespace in sync with it." Here is a complete, production-grade Application manifest with every important field annotated:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: api-production
  namespace: argocd
  labels:
    app.kubernetes.io/name: api
    app.kubernetes.io/component: backend
    env: production
  # The finalizer ensures that deleting this Application object
  # also deletes all the Kubernetes resources it manages.
  # Without this, deleting the Application leaves orphaned cluster objects.
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  # Project scopes RBAC for multi-tenant clusters.
  # The 'default' project allows any source repo and any destination cluster.
  # Production clusters should use a named project with restricted source repos.
  project: platform-production

  source:
    repoURL: https://github.com/myorg/config-repo.git
    # Pin to a branch, not a tag. Argo CD will always track HEAD of this branch.
    # Use a specific SHA for fully pinned deploys (rare; breaks auto-promotion).
    targetRevision: main
    path: overlays/production

  destination:
    # 'https://kubernetes.default.svc' means the same cluster Argo CD runs in.
    # For remote clusters, use the cluster API server URL registered with Argo CD.
    server: https://kubernetes.default.svc
    namespace: api-production

  syncPolicy:
    automated:
      # prune: true — delete resources that are in the cluster but removed from Git.
      # Without this, deleted manifests leave orphaned objects in the cluster forever.
      prune: true
      # selfHeal: true — auto-revert manual changes to the cluster.
      # This is what makes the cluster continuously self-healing.
      selfHeal: true
    syncOptions:
      # Create the namespace if it doesn't exist yet.
      - CreateNamespace=true
      # Use foreground deletion so child resources are deleted before parents.
      - PrunePropagationPolicy=foreground
      # Only apply resources that are OutOfSync, not the entire set every cycle.
      # This dramatically reduces API server load on large apps.
      - ApplyOutOfSyncOnly=true
      # Use server-side apply for better field ownership tracking.
      - ServerSideApply=true
    retry:
      # Retry failed syncs up to 5 times with exponential backoff.
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  ignoreDifferences:
    # HPA manages .spec.replicas at runtime. If we don't ignore this,
    # Argo CD will revert HPA's scaling on every reconcile cycle.
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas
    # cert-manager injects caBundle into webhook configurations.
    # Ignore to prevent spurious OutOfSync on managed webhooks.
    - group: admissionregistration.k8s.io
      kind: MutatingWebhookConfiguration
      jqPathExpressions:
        - .webhooks[].clientConfig.caBundle
```

Let me explain the three non-obvious fields in detail:

`syncPolicy.automated.prune: true` is the field that gives you "cluster state must exactly match Git state." Without it, resources you delete from Git are left running in the cluster. With it, deleting a manifest from Git triggers deletion of the corresponding cluster object on the next sync. This is the correct behavior for a GitOps system — Git is authoritative, not the cluster. But be cautious: enable prune only after you are confident your Git state is complete. A common mistake is enabling prune before migrating all manual resources to Git, which results in prune deleting things that were never in Git.

`ignoreDifferences` is the necessary escape valve for fields that legitimately diverge between Git and the cluster at runtime. The HPA example is the canonical one: you declare `replicas: 2` in your Deployment manifest as a baseline, but HPA scales replicas to 8 during peak traffic. Without `ignoreDifferences`, Argo CD will revert replicas to 2 on the next sync cycle — overriding the HPA's work. With the `/spec/replicas` pointer ignored, Argo CD manages everything about the Deployment except the replica count, which HPA owns.

`syncOptions.ApplyOutOfSyncOnly: true` is a critical performance optimization for applications with many resources. Without it, Argo CD sends every resource in the desired state to the API server on every sync — even resources that have not changed. With it, only the resources that are actually OutOfSync are applied. On an application with 100 resources where only 2 have changed, this reduces API server load by 98% per sync cycle.

---

## 7. The complete GitOps deploy flow: CI to Argo CD

Let me trace a complete deployment through the system so every hand-off is visible. The example is a Go HTTP API deploying a bug fix.

![Complete GitOps deploy lifecycle from code merge through CI image build to config PR to Argo CD sync to cluster update](/imgs/blogs/gitops-git-as-the-source-of-truth-6.png)

**Stage 1 — Code PR merged to app repo (main branch).** The GitHub Actions CI workflow triggers:

```yaml
# .github/workflows/ci-and-promote.yaml
name: CI — build, test, and promote

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/api

jobs:
  # -----------------------------------------------------------------------
  # Job 1: build, test, and push the container image
  # -----------------------------------------------------------------------
  build-and-push:
    name: Build and push container image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      # OIDC token for keyless cosign signing (no stored signing key)
      id-token: write

    outputs:
      image_tag: ${{ steps.meta.outputs.version }}
      image_digest: ${{ steps.push.outputs.digest }}

    steps:
      - name: Checkout source
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract image metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=sha-,format=short
            type=ref,event=branch

      - name: Build and push (with layer cache)
        id: push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          # GitHub Actions cache for BuildKit layers
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Install cosign
        uses: sigstore/cosign-installer@v3

      - name: Sign the image (keyless OIDC)
        run: |
          # Sign the image by digest, not tag (tags are mutable; digests are not)
          cosign sign --yes \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ steps.push.outputs.digest }}

  # -----------------------------------------------------------------------
  # Job 2: open a config-repo PR to update the staging image tag
  # NOTE: this job has NO Kubernetes credentials of any kind.
  #       It only needs: registry read (to validate the image exists)
  #       and config repo write (to open the PR).
  # -----------------------------------------------------------------------
  open-staging-pr:
    name: Promote to staging via config-repo PR
    needs: build-and-push
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - name: Checkout config repo
        uses: actions/checkout@v4
        with:
          repository: myorg/config-repo
          # Personal access token scoped to config-repo only.
          # No cluster credentials here or anywhere in this job.
          token: ${{ secrets.CONFIG_REPO_PAT }}
          path: config-repo

      - name: Install kustomize
        uses: imranismail/setup-kustomize@v2

      - name: Update staging image tag
        working-directory: config-repo/overlays/staging
        run: |
          kustomize edit set image \
            ghcr.io/myorg/api/api=ghcr.io/myorg/api/api:${{ needs.build-and-push.outputs.image_tag }}
          # Verify the edit happened
          grep -q "${{ needs.build-and-push.outputs.image_tag }}" kustomization.yaml

      - name: Open pull request against config repo
        uses: peter-evans/create-pull-request@v6
        with:
          path: config-repo
          token: ${{ secrets.CONFIG_REPO_PAT }}
          branch: deploy/api-staging-${{ needs.build-and-push.outputs.image_tag }}
          base: main
          title: "deploy(staging): api ${{ needs.build-and-push.outputs.image_tag }}"
          body: |
            ## Automated promotion to staging

            **Image**: `ghcr.io/myorg/api/api:${{ needs.build-and-push.outputs.image_tag }}`
            **Image digest**: `${{ needs.build-and-push.outputs.image_digest }}`
            **Source commit**: ${{ github.server_url }}/${{ github.repository }}/commit/${{ github.sha }}
            **CI run**: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}

            After staging validation passes, promote to production by updating
            `overlays/production/kustomization.yaml` with the same tag.
          commit-message: |
            chore(staging): update api image to ${{ needs.build-and-push.outputs.image_tag }}
```

Notice that `open-staging-pr` has `permissions: contents: read` only — the minimum needed to check out its own repo. It uses a PAT scoped only to the config repo. There is no `KUBECONFIG`, no `AWS_ROLE_ARN`, no `GOOGLE_CREDENTIALS`, no cluster access of any kind in this job.

**Stage 2 — Config PR reviewed and merged to staging.** A platform engineer or an automated acceptance-test bot reviews the staging PR. If smoke tests against staging pass (run as a GitHub Actions workflow that triggers on the config PR), the PR is merged. The staging overlay now has the new image tag at HEAD of the config repo's main branch.

**Stage 3 — Argo CD detects the diff.** Within three minutes of the merge (or immediately if a webhook is configured), Argo CD's `api-staging` Application reconciles. It builds the Kustomize overlay, sees that the image tag in the Deployment's Pod template has changed, and computes the diff. The Application moves to OutOfSync.

**Stage 4 — Argo CD syncs.** With `syncPolicy.automated.selfHeal: true`, Argo CD triggers an automatic sync. It applies the updated Deployment via server-side apply. Kubernetes starts a rolling update: new pods with the new image are created, their readiness probes pass, and old pods are terminated. Typically this takes 30–90 seconds for a Deployment with 2 replicas.

**Stage 5 — Cluster is Synced.** The next reconcile loop (within 3 minutes of sync completion) finds no diff. The `api-staging` Application is Synced and Healthy.

**Stage 6 — Production promotion.** After a staging soak period (30 minutes to several hours, depending on your risk tolerance and the nature of the change), the promotion is done manually: open a PR in the config repo updating `overlays/production/kustomization.yaml` with the same image tag. This PR requires approval from the platform team (enforced by CODEOWNERS). After review and merge, Argo CD's `api-production` Application reconciles and syncs. Done.

Total elapsed time from code merge to production pod: depends on soak time. The technical pipeline (code merge → new image in production) takes roughly 15–25 minutes (7–9 min CI build, 2 min PR merge, 3 min Argo CD detection, 1 min sync). The staging soak period is the variable. This is the correct trade-off: slow deliberate production promotion with fast technical execution.

---

## 8. Rollback: git revert beats every alternative

Rollback is where the GitOps model's advantages are most viscerally clear. Let me be specific about the comparison.

![Rollback comparison: push-CD requires re-running full pipeline versus GitOps rollback via git revert applied automatically in minutes](/imgs/blogs/gitops-git-as-the-source-of-truth-4.png)

**Push-based rollback procedure:**

1. Identify the last known-good image tag. This requires checking CI logs, looking at the deployment history in the cluster (`kubectl rollout history`), or looking at the container registry for the previous tag.
2. Trigger the CI pipeline with that tag. This might mean reverting the source code commit and letting CI re-run, or manually triggering a pipeline with a specific image tag parameter.
3. Wait for the full CI pipeline to run (build, test, package steps), even though you are deploying an image that already exists and was already tested. Most pipelines do not short-circuit for "this image tag already exists in the registry."
4. Deploy. In a push-based model, the pipeline pushes to the cluster.

Total time: 10–20 minutes for a typical CI pipeline, assuming the pipeline parameter exists, you can find the right tag under pressure, and no one broke the pipeline since the last deploy.

**GitOps rollback procedure:**

```bash
# Step 1: Find the previous production commit
cd config-repo
git log --oneline overlays/production/kustomization.yaml
# a3f2e1b chore(production): update api image to sha-deadbeef  ← broken
# 9c1a7d3 chore(production): update api image to sha-cafe9876  ← last good

# Step 2: Revert the broken commit
git revert a3f2e1b --no-edit

# Step 3: Push (triggers Argo CD webhook if configured)
git push origin main
```

Argo CD detects the new commit (within 3 minutes, or immediately via webhook), computes the diff (the image tag has changed back to `sha-cafe9876`), and syncs. Kubernetes starts a rolling update back to the previous image. The cluster is running the previous version within 4–5 minutes of the `git push`.

You can also trigger rollback directly through Argo CD without touching Git:

```bash
# List revision history for the application
argocd app history api-production
# ID    DATE                           REVISION
# 47    2026-06-22 14:30:00 +0000     main (a3f2e1b)   ← current (broken)
# 46    2026-06-22 13:15:00 +0000     main (9c1a7d3)   ← previous (good)

# Roll back to revision 46
argocd app rollback api-production 46 --prune

# Monitor the sync
argocd app wait api-production --sync --health --timeout 120
```

Note that `argocd app rollback` puts the Application into a "suspended" state — it pins the sync target to revision 46 and disables automated sync temporarily. After the incident is resolved, you resume with `argocd app set api-production --sync-policy automated` or by pushing a new commit that supersedes the bad one.

#### Worked example: rollback timing comparison

A four-person team migrated their payment processing microservice from push-based CI to GitOps over six weeks. Before-and-after measurements from their MTTR drill (deliberately breaking a deploy and measuring time to recovery):

| Step | Push-based CD | GitOps (Argo CD) | Reduction |
|------|--------------|------------------|-----------|
| Identify broken version | 4 min | 90 sec (git log) | 62% |
| Initiate rollback | 3 min (find old tag, trigger pipeline) | 60 sec (git revert + push) | 67% |
| Pipeline / sync execution | 14 min | 3.5 min | 75% |
| Verify cluster healthy | 2 min | 30 sec (argocd app wait) | 75% |
| **Total MTTR (rollback drill)** | **23 min** | **6 min** | **74% reduction** |
| Audit trail after rollback | "Pipeline #847 re-ran" | Git commit with reason | Qualitative win |

The DORA 2023 State of DevOps report defines elite performers as having MTTR under one hour. The industry median is between one and 24 hours. A 74% reduction in rollback execution time moves most mid-tier teams from the medium to the high performer band on that metric alone.

---

## 9. Drift: the silent enemy

Drift is the condition where the running state of your cluster diverges from any documented desired state. It is the natural outcome of the push-based model over time — not an edge case, but the inevitable end state of a cluster that receives both pipeline-pushed changes and manual interventions.

![Manual kubectl drift creation versus GitOps automatic drift detection and revert within one reconcile cycle](/imgs/blogs/gitops-git-as-the-source-of-truth-8.png)

Drift accumulates silently in push-based environments through several mechanisms:

**Debugging changes.** An engineer SSHes into a node or runs `kubectl exec` to debug a production issue. They modify a ConfigMap to enable verbose logging. They forget to revert it. The cluster now has verbose logging in production consuming disk space and degrading query performance. Nobody knows why.

**Emergency patches.** A critical security vulnerability is discovered. An engineer runs `kubectl set image deployment/api api=company/api:patched` to bypass the full CI pipeline and get the fix out in 10 minutes. The fix works. The next day, a different team member triggers a normal pipeline deploy which overwrites the patched image with the previous release image (which contains the vulnerability) because the pipeline YAML still references the old version. The vulnerability is reintroduced.

**Scale adjustments.** During a traffic spike, an engineer runs `kubectl scale deployment api --replicas=10`. The spike passes and they reduce to 6, intending to tune it further. They get pulled into another incident. The Deployment manifest still says `replicas: 2`. Three months later, a cost audit shows unexplained Kubernetes compute charges. The 6 replicas running in prod are not in any manifest.

**Operator mutations.** cert-manager injects `caBundle` into webhook configurations. A service mesh sidecar injector adds sidecar containers to pods. An admission webhook adds scheduling constraints. These mutations are correct and expected, but in a naive push-based model, the next `kubectl apply` may overwrite them if the manifest does not account for them.

In GitOps with `selfHeal: true` and `prune: true`, the first three scenarios are impossible to sustain. Any manual change to the cluster is detected by the next reconcile cycle and reverted. This is not a punitive mechanism — it is a protective one. The engineer who ran `kubectl scale` to handle a spike can do so knowing that (1) Argo CD will alert on the drift, (2) they have a 3-minute window before it is reverted, and (3) the correct long-term response is to update the HPA min/max in the config repo, not to leave a manual scale change in place.

For the fourth scenario — legitimate operator mutations — the `ignoreDifferences` field in the Application spec is the answer. List the fields that Kubernetes controllers legitimately manage and Argo CD will exclude them from drift detection.

---

## 10. GitOps and IaC: the complete "everything as code" stack

GitOps and IaC (Infrastructure as Code) are complementary practices from the same philosophical root: the desired state of your system should be expressed as code, version-controlled in Git, and applied deterministically by automation. Terraform manages the cloud infrastructure layer (VPCs, EKS clusters, IAM roles, RDS instances). GitOps manages the Kubernetes workload layer (Deployments, Services, ConfigMaps, Argo CD Applications). Together they give you a complete, layered model where every component of your production environment is declared in Git.

The bootstrap sequence for a new environment:

```bash
# ── Layer 1: Cloud infrastructure ──────────────────────────────────────────
# Terraform creates the EKS cluster, node groups, VPC, security groups,
# IAM roles for IRSA (IAM Roles for Service Accounts), and ECR registries.
cd infrastructure/
terraform init -backend-config=backends/production.hcl
terraform apply -var-file=envs/production.tfvars

# ── Layer 2: Cluster platform components ────────────────────────────────────
# Install Argo CD itself, cert-manager, external-dns, external-secrets-operator.
# This is a one-time bootstrap — after this, GitOps manages itself.
helm repo add argo https://argoproj.github.io/argo-helm
helm upgrade --install argocd argo/argo-cd \
  --namespace argocd --create-namespace \
  --version 7.3.4 \
  --values helm-values/argocd-production.yaml

# ── Layer 3: Applications via App of Apps ────────────────────────────────────
# The App of Apps Application points to a directory containing all the
# Application manifests. One apply, then GitOps manages everything.
kubectl apply -n argocd -f config-repo/argocd/app-of-apps-production.yaml
```

After those three commands, Argo CD manages every application in production. Adding a new microservice to production is a PR to the config repo adding a new Application manifest in `argocd/`. Removing a service is a PR removing its Application manifest (with `prune: true`, Argo CD also deletes all the cluster resources the Application managed). No imperative steps. No "kubectl apply this directory before running that script."

The App of Apps pattern is the Argo CD way to manage many Applications declaratively:

```yaml
# config-repo/argocd/app-of-apps-production.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: app-of-apps-production
  namespace: argocd
spec:
  project: platform-production
  source:
    repoURL: https://github.com/myorg/config-repo.git
    targetRevision: main
    # This directory contains Application manifests, one per microservice.
    # Argo CD creates each Application and then each Application manages its own resources.
    path: argocd/production
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

The `argocd/production/` directory contains `api.yaml`, `worker.yaml`, `frontend.yaml`, `postgres-operator.yaml`, etc. — one Application per microservice. Argo CD reconciles the app-of-apps Application, which creates/updates/deletes the child Applications, which in turn reconcile the actual workloads. The entire cluster's desired state is a tree of Git-declared objects.

For deeper context on the IaC layer, see [infrastructure-as-code-from-clickops-to-declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative). The GitOps layer described here is specifically the workload layer — the Kubernetes application deployments that run on top of the infrastructure Terraform manages.

---

## 11. GitOps principles: the taxonomy

![GitOps four principles organized by storage axis and change-mechanism axis showing declarative versioned pulled reconciled](/imgs/blogs/gitops-git-as-the-source-of-truth-7.png)

The four GitOps principles are not a random checklist — they form a coherent logical structure. The declarative and versioned principles govern *how desired state is stored*: as data not commands, in an immutable versioned store with full history. The pulled and reconciled principles govern *how changes to desired state are applied*: by an agent pulling from the store rather than being pushed to, continuously not one-shot.

This structure reveals why every "GitOps-adjacent" approach that violates one principle breaks the model:

- **Declarative violated** (imperative scripts in Git, e.g., `deploy.sh`): you have versioned history but not idempotent application. Argo CD cannot safely re-run a shell script. Scripts have side effects that depend on order and context. The operator model breaks.

- **Versioned violated** (using a mutable config store like Consul or a database instead of Git): you have a pull model but no audit trail and no rollback to a prior state. Git revert is impossible. Compliance requires history.

- **Pulled violated** (pushing from CI using a cluster credential, even if manifests are in Git): you have versioned manifests and possibly continuous polling (some push-based tools poll for changes), but CI still holds prod credentials. The security blast radius argument collapses. This is the "manifests in Git but CI pushes" anti-pattern.

- **Continuously reconciled violated** (syncing only on commit, not on drift): you have a versioned pull model but no self-healing. A manual change that diverges the cluster from Git persists until the next commit. Drift accumulates between deployments.

All four principles must hold simultaneously for the system to be GitOps. This is not pedantry — each violation reintroduces a specific failure mode that GitOps was designed to eliminate.

---

## 12. Comparison tables: the full trade-off picture

The matrix below summarizes the four operational dimensions where push and pull diverge most sharply. Every team should walk through this with their security and platform leads before adopting GitOps — understanding the trade-offs prevents the all-too-common mistake of treating GitOps as simply "Argo CD installed."

![Push versus pull CD compared across credentials, drift detection, audit trail, and rollback showing GitOps advantage on every operational dimension](/imgs/blogs/gitops-git-as-the-source-of-truth-5.png)

| Dimension | Push-based CD | Pull-based GitOps |
|-----------|--------------|------------------|
| CI credentials to prod | Required (KUBECONFIG or cloud IAM role) | Not required |
| Blast radius if CI is compromised | Full prod cluster access | Draft PR in protected repo |
| Drift detection | None; manual discovery only | Automatic; every reconcile interval |
| Drift correction | Manual kubectl intervention | Automatic (selfHeal); within 3 min |
| Audit trail for prod changes | CI pipeline logs (incomplete) | Git commit log (complete) |
| Rollback mechanism | Re-run pipeline with old tag | `git revert` plus operator syncs |
| Rollback execution time | 10–20 min | 3–5 min |
| MTTR impact | Baseline | ~70% reduction (illustrative) |
| Multi-cluster promotion | Separate pipeline per cluster | Multiple Argo CD Applications in one cluster |
| Operator on-call cost | None (no in-cluster agent) | Argo CD must itself be healthy and updated |
| Config repo discipline required | Low | High (PRs, CODEOWNERS, branch protection) |
| Compliance audit readiness | Medium (CI logs) | High (Git is the record) |

| Feature | Argo CD | Flux |
|---------|---------|------|
| UI | Rich web UI with sync status | CLI-first; Weave GitOps UI optional |
| Application CRD | `Application`, `AppProject` | `GitRepository`, `Kustomization`, `HelmRelease` |
| Helm support | Native (source type `helm`) | Via `HelmRelease` CRD |
| Kustomize support | Native | Native |
| Bootstrap | `argocd-install.yaml` applied manually | `flux bootstrap` CLI |
| Multi-tenancy | Argo CD Projects with RBAC | Namespace isolation + RBAC |
| Drift detection speed | Configurable; default 3 min | Configurable; default 1 min |
| Self-hosted controller | Yes (single ArgoCD server) | Yes (lightweight per-feature controllers) |
| CNCF graduation | Graduated (2022) | Graduated (2022) |
| Best for | Teams wanting a rich UI and App of Apps | Teams preferring minimal footprint and CLI-native workflows |

Both are excellent. Argo CD's UI is more approachable for organizations that want visibility into sync status without CLI proficiency. Flux is more "Kubernetes-native" — every concept is a CRD, the architecture is lightweight controllers rather than a monolithic server, and it composes more naturally with the Kubernetes controller pattern.

---

## 13. Worked examples: from scratch to running GitOps

#### Worked example: zero to GitOps in one afternoon

Team: 4 engineers. Service: a Go HTTP API. Starting state: push-based GitHub Actions pipeline with a `KUBECONFIG` secret baked in.

**Hour 1 — Install Argo CD and set up the config repo:**

```bash
# Install Argo CD (stable version pinned for reproducibility)
kubectl create namespace argocd
kubectl apply -n argocd -f \
  https://raw.githubusercontent.com/argoproj/argo-cd/v2.11.3/manifests/install.yaml

# Wait for Argo CD to be ready
kubectl rollout status deploy/argocd-server -n argocd --timeout=120s

# Get the initial admin password (delete the secret after changing the password)
kubectl get secret argocd-initial-admin-secret \
  -n argocd -o jsonpath='{.data.password}' | base64 -d; echo

# Port-forward the UI for initial setup
kubectl port-forward svc/argocd-server -n argocd 8080:443 &

# Log in via CLI
argocd login localhost:8080 \
  --username admin \
  --password "$(kubectl get secret argocd-initial-admin-secret \
    -n argocd -o jsonpath='{.data.password}' | base64 -d)" \
  --insecure
```

**Hour 2 — Initialize the config repo structure and push the first Application:**

```bash
# In the config-repo (separate repository):
mkdir -p base overlays/{staging,production} argocd

# Base Deployment (no image tag — set per overlay)
cat > base/deployment.yaml << 'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/api/api:placeholder  # overridden by overlay
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
YAML

cat > base/kustomization.yaml << 'YAML'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
YAML

# Staging overlay
cat > overlays/staging/kustomization.yaml << 'YAML'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: api-staging
images:
  - name: ghcr.io/myorg/api/api
    newTag: sha-abc1234
YAML

# Apply the first Application and let Argo CD take it from there
kubectl apply -n argocd -f argocd/app-staging.yaml

# Verify sync
argocd app wait api-staging --sync --health --timeout 120
argocd app get api-staging
```

**Metrics before and after (3-month follow-up):**

| Metric | Before GitOps (push-based) | After GitOps (3 months) | Change |
|--------|---------------------------|------------------------|--------|
| CI secrets with prod access | 1 (KUBECONFIG, 18 months old) | 0 | -100% |
| Drift incidents per month | ~4 (discovered after-the-fact) | 0 sustained (auto-healed) | -100% |
| Rollback drill execution time | 23 min | 6 min | -74% |
| Deploy frequency | 3/day | 9/day | +3× |
| Change-failure rate | 14% | 5% | -64% |
| MTTR (incidents, measured) | 26 min | 9 min | -65% |

These numbers are illustrative of the typical directional improvements teams report. The DORA research (Accelerate, 2023 State of DevOps Report) documents similar improvements for teams adopting pull-based deployment automation and declarative configuration.

#### Worked example: catching a config mistake before production

CI builds `sha-deadbeef` from a feature branch merge. The promotion bot opens a PR to the staging overlay. During automated staging smoke tests (a GitHub Actions workflow that triggers on config-repo PR creation), the test suite fails: the new image returns HTTP 500 on `GET /healthz`. The test uploads the failure log as a PR comment. The Argo CD staging Application has been blocked from syncing because the PR is not merged.

The developer looks at the smoke test output: a missing environment variable `DATABASE_MAX_CONNECTIONS` — the new code added a new required env var but the staging values file was not updated. The developer updates the app code default to make the env var optional (with a sensible default), rebuilds, and a new PR `sha-cafe9876` passes the smoke tests. The staging PR is updated to `sha-cafe9876`, merged, Argo CD syncs, staging is healthy.

Two hours later, the production overlay is updated to `sha-cafe9876` via a second PR. Production deploy: zero issues.

Without the config-repo PR model and the staging smoke test gate, `sha-deadbeef` would have been deployed directly to production via a push-based pipeline, causing a 500 error on the health endpoint until someone noticed. The config-repo PR as the production gate is what caught it.

---

## 14. War story: the hardcoded deploy key

In a fintech platform serving payment processing for mid-market retailers (illustrative composite of incidents I have encountered or consulted on), a push-based GitLab CI pipeline had been the deployment mechanism for two years. The pipeline used a `KUBE_CONFIG` CI variable that had been created by an engineer who left the company fourteen months prior. The credential was a cluster-admin `kubeconfig` for the production EKS cluster, base64-encoded and stored as a protected variable in GitLab.

The credential had never been rotated. The company's SOC2 audit that year had flagged "access credential rotation policy" as a finding but the remediation was tracked as a low-priority ticket.

Fourteen months after the original engineer left, during a security incident review, the CISO's team discovered that a CI job from 11 months earlier had run `env` to debug a failing step. GitLab's artifact system had stored the full job log, including the complete environment variable dump, as a downloadable artifact. The artifact had an expiry of 30 days, so it was long gone — but the audit of artifact access logs showed that the artifact had been accessed three times during its 30-day window by three different accounts, one of which was an IP address associated with a third-party contractor firm that had since completed their engagement.

The remediation: rotate all credentials (two days of work across multiple teams), audit every cluster action in CloudTrail and Kubernetes audit logs for 14 months (three weeks for two engineers), implement OIDC-based keyless authentication for the remaining push-based deploys, and migrate the rest to Argo CD. The total cost of the incident — engineer time, external security firm audit, accelerated compliance certification timeline — was well into six figures.

The GitOps model eliminates this entire attack surface. CI never holds cluster credentials, so there is nothing to accidentally log, nothing to expire, and nothing to rotate. The Argo CD agent uses a Kubernetes ServiceAccount inside the cluster, accessed via the cluster's own RBAC, never exposed to the public internet.

The SolarWinds attack of 2020 is the highest-profile case in this category. Attackers compromised SolarWinds' Orion build pipeline and injected malicious code (later called SUNBURST) into the legitimate Orion software update, which was cryptographically signed by SolarWinds' own certificate and distributed through their official update channel. Approximately 18,000 organizations installed the update, including multiple US government agencies. The CISA advisory documents the full attack chain. The relevant lesson for GitOps: build system compromise is a real, documented threat vector. GitOps does not prevent build system compromise (that is the domain of SLSA provenance attestations, Sigstore cosign signing, and reproducible builds). But GitOps does add a human-reviewed config-repo PR as a gate between a compromised build and a running production workload. The attacker still has to get their malicious image approved by a reviewer who is presumably not going to approve a manifest that references an obviously malicious image. This is defense-in-depth, not a silver bullet.

---

## 15. Stress-testing the GitOps model

Every model has failure modes. The honest engineer stress-tests their own architecture before an incident does it for them.

**What if the config repo is unreachable?** Argo CD caches the last-known desired state in its internal database (backed by a Kubernetes Secret or an etcd backend). If Git is unreachable, Argo CD continues serving the cached desired state — no new syncs happen, but the cluster stays at its last-synced state. The Application status shows "Unknown" on the sync status but the health status (cluster reality) continues to be accurate. Recovery is automatic when Git becomes reachable again.

**What if two config-repo PRs merge within seconds of each other?** Git serializes commits; the second PR merge produces a commit that includes both changes (assuming no merge conflicts). Argo CD sees the final state at HEAD and syncs to it. If Argo CD is mid-sync when the second commit arrives, it will complete the first sync and then immediately start a second sync for the new HEAD. No lost updates, no partial-application of changes.

**What if Argo CD itself crashes during a sync?** The sync operation in Argo CD is idempotent (server-side apply is idempotent). When Argo CD restarts, its controller reconciles the current cluster state against Git and either completes the sync or identifies that the cluster is already in the desired state. Partial syncs do not leave the cluster in a corrupted state — the resources that were applied remain applied; the ones not yet applied will be applied on the next reconcile.

**What if the image referenced in the config repo does not exist in the registry?** The Kubernetes node will attempt to pull the image and fail with `ImagePullBackOff`. The Pod will enter a crash loop. Argo CD will report the Deployment as Degraded. The sync will show as Progressing or Degraded, not Synced. The old pods (from the previous ReplicaSet) will continue running until the new pods are ready — which they will not be if the image does not exist. Net result: no serving disruption, but an alert on Degraded application status. Rollback: `git revert` in the config repo.

**What if the git revert itself conflicts?** If multiple changes have been made to the config repo since the broken commit and the revert cannot be computed cleanly, you will get a merge conflict. The correct resolution is: manually edit the kustomization.yaml to set the image tag back to the last known-good tag, commit, push. Do not try to resolve a complex revert conflict under incident pressure — just directly edit the image tag. That is the operationally correct action in a production incident.

**What if selfHeal reverts an emergency fix?** If you ran `kubectl set image` for an emergency and Argo CD reverted it within 3 minutes, the right response is: (1) temporarily disable selfHeal (`argocd app set api-production --sync-policy none`), (2) apply the emergency fix again, (3) open a PR to the config repo capturing the fix properly, (4) merge, re-enable automated sync. The correct discipline is: if it is urgent enough to bypass GitOps, it is urgent enough to also open the emergency PR immediately after.

**What if the cluster is destroyed and needs to be rebuilt from scratch?** This is where the GitOps model shines most brightly. Because every application and every application configuration is declared in the config repo, rebuilding the cluster is a three-step operation: (1) run Terraform to recreate the cluster infrastructure, (2) install Argo CD on the new cluster, (3) apply the App of Apps manifest. Within 5–10 minutes, Argo CD has recreated every application exactly as it was. No manual steps, no "I think we had this ConfigMap, let me check Slack", no tribal knowledge required.

Compare this to push-based CD disaster recovery: you need to find every service's deploy script, run them in the correct order, hope that all the manual changes from the past year are still captured somewhere, and debug why service B cannot find service A's endpoint because someone hardcoded a cluster-internal IP six months ago and nobody documented it. GitOps disaster recovery is boring. That is the goal.

**What if a secret is accidentally committed to the config repo?** This is a real risk and a discipline requirement. Never commit raw secrets to the config repo. Use Kubernetes External Secrets Operator (which reads from HashiCorp Vault, AWS Secrets Manager, or GCP Secret Manager and creates Kubernetes Secrets in the cluster), Sealed Secrets (encrypted secrets committed to Git, decryptable only by the controller in the cluster), or direct IRSA/Workload Identity (where pods get cloud credentials via ServiceAccount annotations without any secret object in Git). The config repo should contain references to secrets, not secret values. If a secret is accidentally committed, treat it as a security incident: rotate the secret immediately, rewrite history with `git filter-repo`, force-push, and notify anyone who had access to the repo.

**What about multi-cluster GitOps?** A single Argo CD instance can manage multiple clusters. You register each cluster with Argo CD:

```bash
# Register a remote cluster with Argo CD
argocd cluster add production-us-east-1 \
  --kubeconfig ~/.kube/config \
  --kube-context production-us-east-1

# Verify registered clusters
argocd cluster list
# SERVER                              NAME                    VERSION  STATUS
# https://kubernetes.default.svc      in-cluster (argocd)     1.29     Successful
# https://api.prod-us-east.example.com production-us-east-1   1.29     Successful
# https://api.prod-eu-west.example.com production-eu-west-1   1.29     Successful
```

Each Application manifest specifies which cluster it targets via the `spec.destination.server` field. The config repo can have overlays per cluster or per region. This is the standard pattern for multi-region GitOps: one central Argo CD instance (sometimes called a "management cluster" or "hub cluster") watches the config repo and pushes desired state to spoke clusters. The spoke clusters receive the sync but do not hold the config repo credentials — the hub does.

For hub-spoke GitOps at scale, the alternative is running an Argo CD instance or a Flux controller in each spoke cluster. This avoids the single point of failure in the hub model and reduces blast radius — a compromised hub Argo CD instance cannot control all clusters. The trade-off is operational complexity: N Argo CD instances to upgrade and monitor instead of one. Most teams start with hub-spoke and migrate to distributed controllers as they scale beyond 5–10 clusters.

---

## 16. When NOT to use GitOps

GitOps has real costs. Be direct about when those costs exceed the benefits.

**Three-person startups.** If you are deploying a single service and your whole team has cluster access, the overhead of maintaining a separate config repo, operating Argo CD, and enforcing PR-based promotion is real friction that does not pay for itself. Use a PaaS (Render, Railway, Fly.io) or a simple push-based pipeline until you have more than one team deploying to the cluster or a security requirement around prod credentials.

**No Kubernetes.** GitOps as described is fundamentally about Kubernetes and declarative manifests. It maps poorly to serverless functions, traditional VM-based deploys, or anything relying on imperative deployment scripts. For non-Kubernetes environments, the closest analog is GitOps-style IaC (Terraform Cloud, Atlantis) but that is a different practice.

**No config-repo discipline.** GitOps requires engineers to make config changes via PRs, not ad-hoc `kubectl` commands. If the team culture is "apply it quick and update the manifest later," GitOps creates friction and resentment without the corresponding safety. Either fix the culture first (which GitOps can help enforce, but the organizational will must exist) or do not adopt GitOps yet.

**Stateful data.** GitOps controls Kubernetes resources, not the data stored in those resources. Do not try to model your PostgreSQL schema as a Kustomize overlay. Use database migration tools (Flyway, Liquibase, Atlas) that are triggered as part of the deploy process. The [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) patterns compose with GitOps deploys via an init container or a pre-sync hook.

**Ultra-low-latency emergency deploys.** The config-repo PR review step adds latency to the deploy process — typically 5–30 minutes for a human review. If your incident response process requires deploying a hotfix in under two minutes, you need an emergency bypass path. Most mature GitOps setups have one: either a break-glass mechanism (a specific CI job that bypasses the PR requirement, with full audit logging) or a pre-approved emergency PR template. Design your process to account for this before an incident reveals the gap.

**Services without declarative Kubernetes resources.** Some services use CRDs (Custom Resource Definitions) from operators that have their own reconcile loops. GitOps works fine here — Argo CD can manage any Kubernetes resource including CRDs and CR instances. But some operators (particularly older database operators) have reconcile behaviors that conflict with Argo CD's reconcile. Test the interaction carefully before enabling `selfHeal: true` on operator-managed resources.

**The migration itself.** The migration from push-based to GitOps has a risk window: the period when some services are managed by the old pipeline and others are managed by Argo CD. During this window, you have two systems that can both modify the cluster, which creates conflict potential. Plan the migration as a big-bang per service (one Friday afternoon: cut the service over completely, disable the old pipeline deploy step, verify Argo CD is managing it), not a gradual overlap. The overlap state is the most dangerous.

---

## 17. Measuring GitOps impact: the DORA arithmetic

Before you can demonstrate the value of a GitOps migration, you need baselines. Here is the measurement framework and the arithmetic you need.

**Lead time for changes** is the time from code commit to that code running in production. In a push-based pipeline it decomposes as:

```
Lead time = CI build time + deploy pipeline time + human approval time
          = 8 min        + 12 min               + 4 hours (on average, PRs sitting unreviewed)
          = ~4 hours total on average
```

After GitOps, the deploy step is faster but you add config-repo PR time:

```
Lead time = CI build time + config-repo PR open + config-repo review + Argo CD sync
          = 8 min        + 2 min                + 30 min (staging soak) + 3 min
          = ~43 minutes total on average (for low-risk changes with fast review)
```

The lead time win comes primarily from eliminating the "deploy pipeline re-runs" and "waiting for someone to trigger the deploy." The config-repo PR review is structured and fast because reviewers know exactly what they are reviewing (an image tag change, not a full diff of application code).

**Deploy frequency** is the number of production deploys per time period. In push-based pipelines, deploys are often rate-limited by the human fear of pushing to production (because rollback is hard, the deploy process is opaque, and there is no auto-correction of drift). In GitOps, deploys are structurally safer — rollback is fast, drift is auto-corrected, the audit trail is clear — so teams become less risk-averse about shipping smaller changes more frequently. The DORA 2023 research shows that teams deploying more frequently have lower change-failure rates (smaller changes are safer), which creates a reinforcing loop.

**Change-failure rate** is the percentage of production changes that cause incidents. The GitOps model reduces this through two mechanisms: (1) the config-repo PR review catches configuration mistakes before production (the "catching a config mistake" worked example above), and (2) drift auto-correction means that the baseline cluster state is always predictable, reducing the "background noise" incidents caused by unexpected drift interacting badly with new deploys.

**Time to restore (MTTR)** is where the GitOps impact is most immediately measurable. Run a rollback drill before and after the migration. Time every step. The arithmetic:

```
Push-based MTTR components:
  - Detect incident: 5-15 min (depends on alerting)
  - Identify cause: 5-20 min (which deploy? which change?)
  - Initiate rollback: 3-5 min (find old tag, trigger pipeline)
  - Pipeline executes: 10-20 min
  - Verify recovery: 2-5 min
  Total: 25-65 min

GitOps MTTR components:
  - Detect incident: 5-15 min (same; alerting unchanged)
  - Identify cause: 2-5 min (git log overlays/production/ shows last change)
  - Initiate rollback: 1-2 min (git revert + push)
  - Argo CD syncs: 3-5 min
  - Verify recovery: 1-2 min (argocd app wait)
  Total: 12-29 min
```

The detection time is the same — GitOps does not change your alerting. Everything after detection is faster. The "identify cause" step is dramatically faster in GitOps because you have a clear, readable Git log for production changes. In push-based CD, identifying which deploy caused an incident often requires correlating CI pipeline logs, deployment timestamps, and cluster resource history — work that takes 10–20 minutes under incident pressure.

---

## 18. How GitOps fits the commit-to-production spine

GitOps sits at the deploy and operate stages of the commit → build → test → package → deploy → operate spine described in the [series introduction](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model):

- **Commit → build → test → package**: unchanged. CI still builds the image, runs tests, and pushes to the registry. The only change is what happens after: CI opens a config-repo PR instead of running `kubectl apply`.
- **Deploy**: GitOps takes over. The config-repo PR is the deploy gate. The Argo CD sync is the deploy execution. Readiness probes provide deploy verification.
- **Operate**: continuous reconciliation provides the "operate" layer. Drift is detected and healed. Rollback is `git revert`. The `argocd app list` and the config repo Git log are the operational dashboard.

The DORA metrics all move in the right direction after a GitOps migration:

- **Deploy frequency** increases because the path from code merge to a new version being deployable (config-repo PR open) is shorter and more predictable. Teams consistently report 2–4× higher deploy frequency after GitOps adoption.
- **Lead time for changes** decreases because the deploy execution (Argo CD sync) takes 3–5 minutes, not 10–20 for a pipeline re-run.
- **Change-failure rate** decreases because every production config change requires a reviewed PR (adding a second set of eyes) and drift that would have caused surprise failures is auto-corrected before it causes an incident.
- **Time to restore** drops sharply because rollback is a `git revert` and a 3–5 minute operator sync.

The [push-vs-pull-cd-and-who-holds-the-keys](/blog/software-development/ci-cd/push-vs-pull-cd-and-who-holds-the-keys) post dives deeper into the credential security model. The [argo-cd-and-flux-in-practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) post covers Argo CD Rollouts and Flagger for progressive delivery layered on top of this GitOps model. The [designing-for-failure](/blog/software-development/site-reliability-engineering/designing-for-failure) SRE post explains how continuous reconciliation composes with reliability engineering practices.

---

## Key takeaways

1. **Git is the authoritative desired state.** What is in the config repo is what should be in the cluster. What is in the cluster but not in Git is a bug, not a feature.
2. **The pull model eliminates prod credentials from CI.** A compromised CI pipeline cannot deploy to production — it can only open a PR. The blast radius of a supply-chain attack drops from "full cluster access" to "draft PR pending human review."
3. **Continuous reconciliation makes drift self-healing.** Manual `kubectl apply` changes are detected and reverted within one reconcile cycle (minutes). The cluster is always converging toward Git.
4. **Rollback is `git revert`.** No pipeline re-run, no manual `kubectl rollout undo`, no digging through CI logs to find the old image tag. The operator syncs automatically within 3–5 minutes.
5. **The config-repo PR is the production gate.** Every production change requires a reviewed merge. This is the second line of defense that push-based CD lacks entirely.
6. **App repo and config repo must be separate.** Access control, change cadence separation, and no circular CI dependencies all require the separation to be real, not just a directory within the same repo.
7. **`selfHeal: true` and `prune: true` are what make it real GitOps.** Without selfHeal, drift is detected but not corrected. Without prune, deleted manifests leave ghost resources. Both must be enabled for the four principles to hold.
8. **CODEOWNERS on the production overlay is the enforcement mechanism.** Add `overlays/production/ @myorg/platform-team` to the config repo and enable required reviewers. This is the branch protection equivalent for your production deployment gate.
9. **GitOps is not a tool — it is four principles.** Argo CD and Flux are implementations. The principles are: declarative, versioned, pulled, continuously reconciled. Violating any one of them reintroduces a specific failure mode.
10. **Measure before and after.** MTTR, deploy frequency, change-failure rate, and drift incidents all move measurably after a GitOps migration. Establish baselines before you start so you can demonstrate the value of the migration.

---

## Further reading

- [Weaveworks: "GitOps — Operations by Pull Request" (2017)](https://www.weave.works/blog/gitops-operations-by-pull-request) — the original post that named the pattern.
- [OpenGitOps: the four GitOps principles](https://opengitops.dev) — the formal CNCF working group specification.
- [Argo CD documentation](https://argo-cd.readthedocs.io) — Application spec, sync policies, ignoreDifferences, App of Apps pattern, RBAC.
- [Flux documentation](https://fluxcd.io/docs) — GitRepository, Kustomization, and HelmRelease CRDs; the controller architecture.
- [DORA State of DevOps 2023 report](https://dora.dev) — the research data on deploy frequency, lead time, change-failure rate, and MTTR for elite vs low performers.
- [Accelerate by Forsgren, Humble, Kim](https://itrevolution.com/product/accelerate/) — the research foundation for DORA metrics and the organizational practices that move them.
- [CISA SolarWinds advisory (AA20-352A)](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-352a) — documented supply-chain attack via compromised build pipeline.
- [from-commit-to-production-the-cicd-mental-model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series intro and the commit-to-production spine.
- [infrastructure-as-code-from-clickops-to-declarative](/blog/software-development/ci-cd/infrastructure-as-code-from-clickops-to-declarative) — Terraform and GitOps composing for the full "everything as code" stack.
- [argo-cd-and-flux-in-practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) — progressive delivery with Argo CD Rollouts and Flagger on top of the GitOps model.
- [designing-for-failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — SRE perspective on how continuous reconciliation supports reliability.

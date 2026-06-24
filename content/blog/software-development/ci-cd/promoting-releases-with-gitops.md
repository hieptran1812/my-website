---
title: "Promoting releases with GitOps: from dev to prod via PR"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how to promote container images across dev, staging, and prod environments using a config repo, Kustomize overlays, PR-gated approvals, and Argo CD ApplicationSets for preview environments."
tags:
  [
    "ci-cd",
    "devops",
    "gitops",
    "kubernetes",
    "argo-cd",
    "kustomize",
    "release-management",
    "continuous-delivery",
    "preview-environments",
    "flux",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/promoting-releases-with-gitops-1.png"
---

A few years ago I was on a platform team that "promoted" releases by logging into the staging Kubernetes cluster, running `kubectl set image deployment/api api=ghcr.io/org/api:v1.2.3`, watching the rollout, and — if nothing caught fire for twenty minutes — repeating the same command against prod. The entire process lived in a Slack thread and someone's terminal history. When an incident happened at 2 AM six weeks later, the post-mortem question "what changed?" took an hour to answer, because the answer was nowhere in Git.

That was a team of twelve engineers shipping a service used by sixty thousand customers. The deploy process was fine when the team had four engineers and one environment. At twelve, with three services, two clusters, and three on-call engineers, it was a liability. We had no audit trail, no approval gate, and no reproducible record of what was running where. Two different engineers had edited the prod Deployment YAML on the same day in separate terminal sessions, and the second write silently clobbered the first. Neither of them knew it had happened until the incident. The first change was a critical security patch.

The fix was not glamorous: we created a separate Git repository for Kubernetes manifests, structured it with Kustomize overlays per environment, wrote a two-hundred-line GitHub Actions workflow that opened a pull request whenever a new image passed CI, and configured Argo CD to sync from each overlay. From that point forward, every change to what was running in any environment was a commit with an author, a timestamp, and a diff. Promotions became PR reviews. The audit trail was free, because it was just Git.

This post teaches that pattern end to end. By the time you finish reading, you will be able to set up a config repo with a proper Kustomize base-plus-overlay structure, wire a CI step that automatically bumps image tags and opens promotion PRs, configure Argo CD to sync each environment from its overlay, understand the trade-offs between rendering manifests in Git versus keeping raw Kustomize, handle multi-cluster promotion with ApplicationSet, and spin up ephemeral preview environments for pull requests. All with copy-and-adapt YAML. The four DORA metrics — deploy frequency, lead time for changes, change-failure rate, MTTR — are woven throughout because the whole point of this machinery is to move those numbers in the right direction.

![Config repo promotion ladder showing one OCI image digest flowing through dev, staging, and prod overlays](/imgs/blogs/promoting-releases-with-gitops-1.png)

## 1. Why a separate config repo?

The first question every team asks when they hear "config repo" is why the Kubernetes manifests should live in a different repository from the application code. The answer has three parts: blast-radius, review ergonomics, and the build-once-promote-everywhere principle that sits at the center of this entire series.

**Blast-radius isolation.** The application repo is where developers work. It has hundreds of commits per week, feature branches, experimental work-in-progress, and incomplete changes that exist alongside finished work. If you keep your Kubernetes manifests in the same repository, a developer who accidentally merges a broken branch also deploys that branch to every environment that auto-syncs from main. The config repo is the deployment surface. Keeping it separate means a broken application commit does not automatically become a broken deployment unless someone explicitly promotes it into the config repo. The config repo acts as a deliberate buffer between "code that compiles" and "code that runs in production."

There is a quantitative way to think about this. If your application repo averages 40 commits per day across 10 engineers, and roughly 5% of those commits introduce a regression that a human reviewer might catch in a PR review but that CI misses, then you have two regressions per day in the application repo that would reach production if deployment happened automatically on merge. The config repo's PR gate intercepts those. At a 90% catch rate — a conservative estimate given that PRs include linked CI results and test reports — you stop one regression from reaching production every five business days. Over a year, that is roughly 50 prevented production incidents, each of which would have cost your team recovery time and your users reliability.

**Review ergonomics.** In the application repo, a pull request diff is about code logic. It spans dozens of files, test changes, and potentially hundreds of lines of new business logic. Adding a Kubernetes manifest change to that diff mixes deployment concerns with implementation concerns. Reviewers who care about the deploy configuration — the SRE, the platform engineer, the on-call lead — have to wade through application code to find it, and they often do not. Application developers reviewing the same PR usually do not have the context to assess whether a change to a resource limit or a replica count is safe for prod under peak load.

A config repo PR is about one thing: what changes in the running system. The diff is manifest YAML, image tags, replica counts, and resource limits. Every line in that PR has a direct production consequence. Reviewers treat it accordingly, and the review quality shows. A team that routes all manifest changes through the config repo PR gate typically catches 60–80% of the environment configuration issues that previously reached production, based on the pattern I have seen across multiple incident post-mortems.

**Build once, promote everywhere.** The core principle of safe continuous delivery is that you build a single artifact — one OCI image with one content-addressable digest — and promote that exact artifact through environments. You do not rebuild for staging. You do not rebuild for prod. The image that passed unit tests in CI is the image that runs in staging. The image that passed acceptance tests in staging is the image that runs in prod. The same `sha256:a1b2c3...` digest flows through every overlay.

The mathematical argument is straightforward. Let $P(\text{build-drift})$ be the probability that two builds from the same commit produce functionally different binaries due to non-determinism in the build environment (different compiler versions, different package resolution, different ambient state). For most Go and Java services this is very low — perhaps 0.1%. But for services with complex dependency graphs, native code, or OS-specific tooling, it can be as high as 2–5%. If you rebuild for each environment and you have three environments, your probability of a prod binary that differs from the one tested in CI is roughly $1 - (1 - P)^2$ for staging and $1 - (1 - P)^3$ for prod. At $P = 0.02$, that is a 3.9% chance that prod is running something that was never tested. Across 500 deployments a year, that is roughly 20 production deployments of untested code. The config repo structure, by pointing every environment at the same image digest, makes that probability structurally zero.

There is one legitimate exception worth naming: very small teams with a single service, no separate SRE function, and a deployment process that is trivially reversible. A solo developer on a side project who deploys to a single Kubernetes cluster does not need a separate config repo. The overhead of managing two repos, syncing them, writing the PR automation, and keeping the tooling up to date, is not worth it until you have at least three services, a staging environment meaningfully different from prod, and more than one person who needs to approve a deployment. At that point the config repo pays for itself within weeks. Cross-link: see the [GitOps: Git as the source of truth](/blog/software-development/ci-cd/gitops-git-as-the-source-of-truth) post for the deeper pull-based reconciliation model that the config repo enables.

## 2. The config repo structure: base and overlays

The structure that has worked most reliably in practice — and the one that Argo CD's own documentation recommends — is a `base/` directory containing common manifests plus an `environments/` directory containing one overlay per environment.

![Config repo tree showing root branching to base manifests and environments directory with dev, staging, and prod overlays](/imgs/blogs/promoting-releases-with-gitops-7.png)

Here is a complete, real-world example for a service called `api`. The structure is intentionally minimal — you can add more files as the service grows, but start with exactly this layout. The `base/` directory holds common manifests; `environments/` has one subdirectory per environment:

```bash
# Initialize the config repo directory structure
mkdir -p config-repo/base
mkdir -p config-repo/environments/{dev,staging,prod}

# base: shared templates
touch config-repo/base/kustomization.yaml
touch config-repo/base/deployment.yaml
touch config-repo/base/service.yaml
touch config-repo/base/hpa.yaml

# dev overlay
touch config-repo/environments/dev/kustomization.yaml
touch config-repo/environments/dev/patch-image.yaml

# staging overlay
touch config-repo/environments/staging/kustomization.yaml
touch config-repo/environments/staging/patch-image.yaml
touch config-repo/environments/staging/patch-replicas.yaml

# prod overlay
touch config-repo/environments/prod/kustomization.yaml
touch config-repo/environments/prod/patch-image.yaml
touch config-repo/environments/prod/patch-replicas.yaml
touch config-repo/environments/prod/patch-resources.yaml
```

The base `kustomization.yaml` names the common resources. Nothing environment-specific belongs here:

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - hpa.yaml
```

The base `deployment.yaml` uses a placeholder image tag. This placeholder is what CI automation replaces per environment. The placeholder name is explicit and uppercase so that an accidental slip — someone deploying from base rather than an overlay — produces an obviously broken image reference rather than a silently stale one:

```yaml
# base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  labels:
    app: api
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
          image: ghcr.io/org/api:PLACEHOLDER
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
```

The dev overlay pins the image tag for that environment and inherits everything else from base:

```yaml
# environments/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - ../../base
images:
  - name: ghcr.io/org/api
    newTag: "v1.14.2"
namespace: api-dev
commonLabels:
  environment: dev
```

The staging overlay adds a replica-count patch on top, because staging should mirror prod's replica count for accurate load testing:

```yaml
# environments/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - ../../base
images:
  - name: ghcr.io/org/api
    newTag: "v1.13.8"
namespace: api-staging
commonLabels:
  environment: staging
patches:
  - path: patch-replicas.yaml
```

```yaml
# environments/staging/patch-replicas.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 3
```

The prod overlay pins a different (older, validated) tag and increases the replicas and resource limits for production traffic:

```yaml
# environments/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - ../../base
images:
  - name: ghcr.io/org/api
    newTag: "v1.13.1"
namespace: api-prod
commonLabels:
  environment: prod
patches:
  - path: patch-replicas.yaml
  - path: patch-resources.yaml
```

```yaml
# environments/prod/patch-resources.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    spec:
      containers:
        - name: api
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
```

Notice that dev has `v1.14.2`, staging has `v1.13.8`, and prod has `v1.13.1`. Each environment is at a different point in the promotion pipeline, and that is exactly correct. The config repo is a living snapshot of what is deployed where, and a `git diff HEAD~1` tells you exactly what changed, when, and who approved it. You can answer "what is running in prod right now?" by reading `environments/prod/kustomization.yaml`. You can answer "what changed in prod last Tuesday?" by running `git log --all environments/prod/`. No API calls to a deployment system, no terminal session logs, no Slack thread archaeology.

The `HorizontalPodAutoscaler` in the base applies to all environments by default, but you may want to override it per environment:

```yaml
# base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

In production you would likely increase `maxReplicas` to 50 or 100 via a patch. In dev you might reduce it to `minReplicas: 1` to save cluster cost. Each of those changes lives in the appropriate overlay and is reviewed when it changes.

### Naming convention discipline

The directory names `dev`, `staging`, and `prod` are arbitrary — what matters is consistency. Some teams use `development`, `qa`, `production`. Some use region-specific names like `us-east-1-prod`, `eu-west-1-prod`. Whatever names you choose, make them match the Argo CD Application names, the namespace names, and the CI environment variables exactly. Inconsistency here is the leading cause of "I deployed to the wrong environment" incidents.

A practical convention that scales well is to use the environment name as a namespace prefix and a DNS subdomain prefix simultaneously. If the environment is `staging`, the namespace is `api-staging`, and the service is reachable at `api.staging.internal.example.com`. When the environment is `prod`, it is `api.prod.internal.example.com` (or simply `api.internal.example.com` with an alias). This uniformity means anyone on the team can predict the URL and namespace for any service in any environment, which reduces the cognitive load on-call and during incidents.

### What belongs in the config repo vs the application repo

A common confusion when starting with the config repo pattern: where does the Helm `values.yaml` file live? What about environment-specific feature flags? Database migration scripts?

The rule of thumb: anything that controls **what runs and how it is configured in the cluster** lives in the config repo. Anything that controls **how the application behaves at runtime** lives in the application repo (or in a secrets manager referenced by the cluster).

Concretely:
- Config repo: Kubernetes Deployment manifests, Service, Ingress, HPA, PodDisruptionBudget, resource requests/limits, replica counts, image tags, Kustomize overlays, Helm values for infrastructure components (ingress controller, cert-manager, metrics-server)
- Application repo: application source code, Dockerfile, unit tests, integration tests, Helm chart templates (if the app ships its own chart), feature flag defaults
- Secrets manager or External Secrets Operator: database connection strings, API keys, TLS certificates, OIDC client secrets — referenced by the cluster via SecretStore but never committed to either repo in plaintext

Database migration scripts occupy a gray zone. They relate to both the application (they implement the schema the application expects) and the cluster (they need to run in the cluster's context, with database credentials). The most common approach is to run migrations as a Kubernetes Job triggered by a sync wave in Argo CD, with the Job manifest in the config repo and the migration SQL in a ConfigMap or in the container image itself. Cross-link: for the zero-downtime migration pattern see the `software-development/database` series.

## 3. Promotion via pull request: the PR is the approval gate

The key insight of GitOps-based promotion is that a merge event IS a deployment event. When the staging-to-prod PR is merged, Argo CD sees the change in the prod overlay and reconciles the cluster to match. There is no separate deploy button, no separate ticket, no separate Slack message to the SRE. The approval gate is the PR review. The deployment artifact is the merge commit. The audit trail is the Git history.

![GitOps promotion flow branching at the test gate: failure sends alert while success opens the staging PR](/imgs/blogs/promoting-releases-with-gitops-2.png)

This means that a promotion workflow has four actors working in sequence:

**Actor 1: The CI system** (GitHub Actions, GitLab CI, etc.) builds the image, pushes it to the registry, then commits the new tag to the dev overlay and opens a PR to the staging overlay. The CI system is the only actor with write access to the config repo via a deploy key. It never has access to the production cluster directly.

**Actor 2: Argo CD** polls the config repo (or receives a webhook notification) every 3 minutes, compares the desired state in Git to the actual state in the cluster, and reconciles the difference. Argo CD has cluster access but no Git write access. CI has Git write access but no cluster access. This separation of privileges is what makes pull-based GitOps fundamentally more secure than push-based CI/CD.

**Actor 3: The automated test suite** runs against the dev and staging environments after Argo CD syncs, verifying that the new image passes smoke tests and acceptance tests. The test suite sends a pass/fail signal back to the CI system via an API call or a commit status update.

**Actor 4: A human reviewer** approves the staging-to-prod PR. The reviewer looks at the image tag diff, checks the test results linked in the PR description, and merges. For a one-line tag change this review takes 90 seconds. For a tag change that also bumps resource limits or changes a feature flag, it takes longer — and that extra review time is valuable.

### The DORA impact of PR-gated promotion

The DORA State of DevOps 2023 research classifies teams into four performance bands — elite, high, medium, low — based on deploy frequency, lead time, change-failure rate, and MTTR. The transition from medium to high performance correlates most strongly with two practices: continuous integration (small batches, trunk-based development) and the kind of deployment gate described here.

Before implementing GitOps PR-gated promotion on a representative mid-size team (20 engineers, three services, two clusters):
- Deploy frequency: 2 per week
- Lead time for changes: 3.5 days (code ready Monday, deploy window Thursday evening, deployment call Friday)
- Change-failure rate: 18% (20% of deploys require an emergency rollback or patch within 24 hours)
- MTTR: 75 minutes (manual rollback: find the previous image tag, run `kubectl set image`, wait for rollout)

After implementing GitOps PR-gated promotion:
- Deploy frequency: 9 per day
- Lead time for changes: 2.2 hours (8 min CI + 12 min dev + 30 min staging PR async + 90 min prod PR)
- Change-failure rate: 5%
- MTTR: 8 minutes (revert is a one-line PR bumping the tag back; Argo CD reconciles in 30 seconds)

The change-failure rate drop from 18% to 5% came almost entirely from the dev smoke test gate — broken changes that previously reached staging now stop at dev. The lead time drop from 3.5 days to 2.2 hours came from eliminating the coordination overhead of a weekly deploy window. The MTTR drop from 75 to 8 minutes came from making rollback a merge operation rather than a manual cluster operation.

These numbers are representative, not a specific company's internal benchmark. I am presenting them as order-of-magnitude illustration, consistent with the DORA research findings that elite performers have lead times under one day and change-failure rates under 5%.

## 4. CI automation: the bump-image-tag step

The most important piece of glue in the whole system is the CI step that writes the new image tag into the config repo. This step runs after the image is built and pushed. It clones the config repo, updates the tag in the dev overlay, commits, and pushes. On success, it opens a pull request to staging.

#### Worked example:

Here is the complete GitHub Actions workflow for the application repo. The `update-config-repo` job runs after `build-and-push` and handles both the dev auto-promotion and the staging PR creation:

```yaml
# .github/workflows/deploy.yml  (in the application repo)
name: Build, push, and promote

on:
  push:
    branches:
      - main

env:
  IMAGE: ghcr.io/${{ github.repository_owner }}/api
  CONFIG_REPO: git@github.com:org/config-repo.git
  CONFIG_REPO_HTTP: https://github.com/org/config-repo

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      image_tag: ${{ steps.tag.outputs.tag }}
      image_digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - name: Set image tag
        id: tag
        run: echo "tag=$(git rev-parse --short HEAD)" >> "$GITHUB_OUTPUT"

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          cache-from: type=gha
          cache-to: type=gha,mode=max
          tags: |
            ${{ env.IMAGE }}:${{ steps.tag.outputs.tag }}
            ${{ env.IMAGE }}:latest
          labels: |
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.created=${{ github.event.head_commit.timestamp }}

  update-config-repo:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Install dependencies
        run: sudo apt-get install -y git gh

      - name: Configure git
        run: |
          git config --global user.email "ci-bot@org.com"
          git config --global user.name "CI Promotion Bot"

      - name: Clone config repo (SSH)
        env:
          DEPLOY_KEY: ${{ secrets.CONFIG_REPO_DEPLOY_KEY }}
        run: |
          mkdir -p ~/.ssh
          echo "$DEPLOY_KEY" > ~/.ssh/id_rsa
          chmod 600 ~/.ssh/id_rsa
          ssh-keyscan github.com >> ~/.ssh/known_hosts
          git clone ${{ env.CONFIG_REPO }} /tmp/config-repo

      - name: Bump image tag in dev overlay (direct push)
        env:
          TAG: ${{ needs.build-and-push.outputs.image_tag }}
        run: |
          cd /tmp/config-repo
          # Replace the newTag line in the dev kustomization.yaml
          sed -i "s|newTag:.*|newTag: \"$TAG\"|" environments/dev/kustomization.yaml
          git add environments/dev/kustomization.yaml
          git commit -m "chore(dev): bump api to $TAG

          App commit: ${{ github.sha }}
          Workflow run: ${{ github.run_id }}
          Image digest: ${{ needs.build-and-push.outputs.image_digest }}"
          git push origin main

      - name: Open staging promotion PR
        env:
          GH_TOKEN: ${{ secrets.CONFIG_REPO_PAT }}
          TAG: ${{ needs.build-and-push.outputs.image_tag }}
        run: |
          cd /tmp/config-repo
          # Refresh to get the dev-bump commit
          git pull origin main
          # Create a branch for the staging bump
          git checkout -b promote/staging-$TAG
          sed -i "s|newTag:.*|newTag: \"$TAG\"|" environments/staging/kustomization.yaml
          git add environments/staging/kustomization.yaml
          git commit -m "chore(staging): promote api to $TAG"
          git push origin promote/staging-$TAG

          # Open the PR (|| true: silently succeeds if PR already exists for this tag)
          gh pr create \
            --repo org/config-repo \
            --title "Promote api $TAG to staging" \
            --body "## Promotion summary

          **Service:** api
          **Image:** ghcr.io/org/api:$TAG
          **Digest:** ${{ needs.build-and-push.outputs.image_digest }}
          **App commit:** ${{ github.sha }}
          **CI run:** ${{ github.run_id }}

          ## Pre-merge checklist
          - [ ] Dev smoke tests passed (see CI run link above)
          - [ ] No open P0/P1 incidents on dev environment
          - [ ] Image digest matches what was tested in dev
          - [ ] Reviewer has checked the image tag diff (should be one line)

          ## Rollback
          If staging fails after merge: open a revert PR bumping the tag back to the previous value." \
            --base main \
            --head promote/staging-$TAG \
            --label "promotion,staging,needs-review" || true
```

A few things worth noting in this workflow.

The dev bump goes directly to `main` via a commit — dev auto-deploys because it is low-risk and you want fast feedback on every merge. The staging bump goes through a PR branch because a human needs to approve it. These two operations use different credentials: the dev bump uses a deploy key (SSH write access to the config repo); the staging PR uses a GitHub PAT (HTTPS API access to create PRs). Neither credential is a cluster credential. Argo CD handles cluster access independently.

The `|| true` at the end of `gh pr create` prevents the workflow from failing if a PR for that tag already exists (for example, if the workflow re-ran after a transient failure). This is intentional defensive coding — idempotency is more important than strict error signaling here.

The PR body is a checklist, not just a title. A well-structured PR body is the difference between a rubber-stamp review and a genuine gate. When the reviewer opens the PR and sees a checklist with "Dev smoke tests passed: yes/no," they are prompted to actually check before approving.

### Security note: deploy key versus PAT

The `CONFIG_REPO_DEPLOY_KEY` is an SSH deploy key with write access to the config repo. The `CONFIG_REPO_PAT` is a GitHub Personal Access Token scoped to `repo` (for PR creation). Neither of these should be the same key — deploy keys get push access; PATs get PR-creation access. Separating them limits blast radius if one leaks. The deploy key can be rotated without affecting the PAT, and vice versa.

A more secure approach — especially for organizations that have adopted OIDC keyless federation — is to use GitHub's OIDC token to authenticate to the config repo without storing any long-lived secret at all. This requires setting up a GitHub App with installation access to the config repo and exchanging the OIDC token for an installation token at workflow runtime. The overhead is higher, but the security properties are significantly better for high-assurance environments. See [configuration and secrets in Kubernetes](/blog/software-development/ci-cd/configuration-and-secrets-in-kubernetes) for the OIDC federation pattern in detail.

## 5. Rendered manifests vs raw Kustomize in Git

Once you have the config repo structure working, you face a non-obvious design decision: should you commit the raw Kustomize overlays — the kustomization.yaml files plus patches — or should you commit the fully rendered, expanded YAML that `kustomize build` produces?

![Comparison of Kustomize overlays in Git versus fully rendered YAML in Git, showing reviewer experience trade-offs](/imgs/blogs/promoting-releases-with-gitops-3.png)

The difference is significant in practice:

| Attribute | Kustomize overlays in Git | Rendered YAML in Git |
|---|---|---|
| PR diff size | Small: only changed lines | Large: full Deployment YAML (~200 lines for a typical manifest) |
| Reviewer tooling required | Must run `kustomize build` to see full impact | No tooling: diff is self-explanatory in GitHub PR UI |
| Drift detection model | Argo CD handles rendering; source of truth is the kustomization | Git diff IS the ground truth; rendered YAML is the source of truth |
| Multi-layer patch clarity | Patches compose implicitly; easy to miss a missed precedence | Final state is fully explicit; no composition ambiguity |
| CI overhead | Minimal: commit 8–20 lines per promotion | Moderate: run `kustomize build` in CI, commit 200–500 lines |
| Accidental divergence risk | Low: base changes propagate automatically to all overlays | Medium: base changes require re-rendering all overlays |
| Policy gate feasibility | Requires rendering in CI to apply policies | Policies apply directly to committed YAML |
| Compliance-friendliness | Auditors must understand Kustomize | Auditors can read YAML |

The rule of thumb: use raw Kustomize overlays in Git for small teams that trust their reviewers to run `kustomize build` locally and that have fewer than ten services. Use rendered YAML in Git for larger teams, compliance-heavy environments, or any team that has been burned by a patch composition bug — a bug where two patches interact in an unexpected way and the final YAML is subtly wrong in a way the kustomization.yaml review does not reveal.

Rendered manifests are also the right choice when you are using `kubectl diff` in CI to gate PRs on actual diff content. If the CI pipeline runs `kubectl diff --server-side -f rendered/prod/all.yaml` and fails the PR when the diff touches RBAC resources, PodSecurityAdmission labels, or namespace-scoped resources that should not change, you have a precise policy gate that requires rendered content to function.

Here is how to add a rendering step to the config repo's own CI pipeline:

```yaml
# .github/workflows/render.yml  (in the config repo)
name: Render Kustomize manifests and validate

on:
  pull_request:
    branches: [main]

jobs:
  render-and-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # needed for git push to work

      - name: Install kustomize
        run: |
          curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
          sudo mv kustomize /usr/local/bin/

      - name: Install kubeconform
        run: |
          wget https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz
          tar xf kubeconform-linux-amd64.tar.gz
          sudo mv kubeconform /usr/local/bin/

      - name: Render all environments
        run: |
          for env in dev staging prod; do
            mkdir -p rendered/$env
            kustomize build environments/$env > rendered/$env/all.yaml
            echo "Rendered $env: $(wc -l < rendered/$env/all.yaml) lines"
          done

      - name: Validate rendered YAML against Kubernetes schemas
        run: |
          for env in dev staging prod; do
            kubeconform -strict -summary rendered/$env/all.yaml
          done

      - name: Commit rendered manifests to PR branch
        run: |
          git config user.email "ci@org.com"
          git config user.name "CI Bot"
          git add rendered/
          git diff --staged --quiet || git commit -m "chore: re-render ${{ github.event.pull_request.head.sha }}"
          git push || echo "Nothing to push"
```

This workflow renders every overlay on every config repo PR, validates the rendered YAML against the Kubernetes API schemas using `kubeconform`, and commits the rendered output to the PR branch. The PR diff then shows both the kustomization.yaml change and the resulting rendered manifest change side by side.

A word on merge conflicts: when two PRs both touch the same overlay, the second to merge will conflict on the rendered file. This is the desired behavior — you want the conflict to surface explicitly rather than silently letting one change overwrite the other. Raw Kustomize overlays do not have this problem because the rendered output is not committed. Rendered YAML trades CI overhead and occasional merge conflict resolution for explicit conflict detection and auditor-readable diffs.

## 6. Promoting across clusters: multi-cluster GitOps

Most teams start with a single Kubernetes cluster running multiple namespaces, one per environment. That works fine until production needs to be in a different region, a different cloud account, or a different network security zone from staging. At that point you need multi-cluster GitOps.

Argo CD handles multi-cluster promotion through two patterns: App-of-Apps and ApplicationSet.

**App-of-Apps** is the simpler pattern. You create one "parent" Argo CD Application that points to a directory of Application manifests. Each child Application points to an environment's overlay. The parent syncs the children; the children sync the clusters:

```yaml
# argocd/apps/api-dev.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: api-dev
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/config-repo.git
    targetRevision: HEAD
    path: environments/dev
  destination:
    server: https://kubernetes.default.svc
    namespace: api-dev
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

```yaml
# argocd/apps/api-prod.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: api-prod
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/config-repo.git
    targetRevision: HEAD
    path: environments/prod
  destination:
    server: https://prod-cluster.example.com
    namespace: api-prod
  syncPolicy:
    # No automated sync for prod — manual sync required after PR merge
    syncOptions:
      - CreateNamespace=true
```

**ApplicationSet** is the more powerful pattern. It generates multiple Applications from a template, driven by a list generator or a cluster generator. For multi-cluster promotion it is the right tool at scale:

```yaml
# argocd/applicationsets/api-environments.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: api-environments
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - env: dev
            cluster: https://kubernetes.default.svc
            namespace: api-dev
            autoSync: "true"
            selfHeal: "true"
          - env: staging
            cluster: https://staging-cluster.example.com
            namespace: api-staging
            autoSync: "true"
            selfHeal: "true"
          - env: prod
            cluster: https://prod-cluster.example.com
            namespace: api-prod
            autoSync: "false"
            selfHeal: "false"
  template:
    metadata:
      name: "api-{{env}}"
    spec:
      project: default
      source:
        repoURL: https://github.com/org/config-repo.git
        targetRevision: HEAD
        path: "environments/{{env}}"
      destination:
        server: "{{cluster}}"
        namespace: "{{namespace}}"
      syncPolicy:
        automated:
          prune: "{{autoSync}}"
          selfHeal: "{{selfHeal}}"
        syncOptions:
          - CreateNamespace=true
```

Notice that prod has `autoSync: "false"`. Argo CD detects drift when the prod overlay changes (after the prod PR merges), but does not automatically apply it. Someone must click "Sync" in the Argo CD UI or run `argocd app sync api-prod`. This is the deliberate friction point for production: merge declares intent, sync executes it. Teams that want zero human touch on prod sync can enable `autoSync: "true"` there too — but then the prod PR approval is the only gate, and there is no additional friction before pods restart.

For teams using Flux instead of Argo CD, the equivalent is a Flux `Kustomization` resource per environment, with a separate `GitRepository` source pointing to the config repo:

```yaml
# flux-system/kustomizations/api-prod.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: api-prod
  namespace: flux-system
spec:
  interval: 5m0s
  path: "./environments/prod"
  prune: true
  sourceRef:
    kind: GitRepository
    name: config-repo
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: api
      namespace: api-prod
  timeout: 3m0s
  wait: true
  retryInterval: 1m0s
```

### Namespace-per-environment versus cluster-per-environment

| Dimension | Namespace-per-env (same cluster) | Cluster-per-env |
|---|---|---|
| Cost | Low: share one control plane and node pool | High: 3 separate control planes, 3 node pools |
| Blast radius | Medium: cluster-level misconfigs affect all envs | Low: prod cluster is fully isolated from staging |
| Network isolation | Requires careful NetworkPolicy; default is allow | Built-in: separate VPCs/VNets, no cross-cluster routes |
| Compliance (PCI DSS, SOC 2 Type II) | Difficult: auditors often require cluster-level separation for cardholder data | Natural: prod network is physically isolated |
| Operational complexity | Low to moderate: one kubeconfig | High: multi-cluster kubeconfig, cross-cluster service discovery |
| Secret isolation | Requires namespace-scoped RBAC + NetworkPolicy | Structural: secrets in prod cluster cannot be read from staging |
| Recommended for | SaaS products, internal tooling, startups | Financial services, healthcare, regulated workloads, multi-region |

For most teams shipping SaaS products, namespace-per-environment with strict RBAC and NetworkPolicy is sufficient until a compliance audit demands cluster-level isolation. At that point the config repo structure does not change — you only update the `destination.server` field in your Application manifests.

## 7. Image tag update automation: who updates the tag?

"Who updates the image tag in the config repo?" is the question that trips up every team moving to GitOps for the first time. There are three answers, each with a different set of trade-offs.

![Comparison of manual image tag editing versus automated CI PR for image tag updates in the config repo](/imgs/blogs/promoting-releases-with-gitops-8.png)

**Option 1: CI commits directly to the config repo.** The application repo's CI workflow clones the config repo and commits the new tag. This is what the GitHub Actions example in section 4 demonstrates. The advantages: explicit, auditable, uses tooling the team already knows, and the commit message can include the exact CI run ID and image digest. The disadvantages: the CI system needs write access to the config repo (deploy key), which is a security surface to manage; simultaneous CI runs can conflict on the commit; and the config repo becomes tightly coupled to the application repo's CI workflow.

The conflict problem is real at scale. If your application repo runs 20 CI builds per hour and each build tries to write to the same `environments/dev/kustomization.yaml` file, you will see frequent conflict failures. The typical mitigation is a retry with rebase:

```bash
# In the CI workflow, replace the simple push with a retry-with-rebase
for i in 1 2 3 4 5; do
  git pull --rebase origin main && git push origin main && break
  echo "Push attempt $i failed, retrying..."
  sleep $((i * 2))
done
```

At more than 30 concurrent builds per hour, even this breaks down. Option 2 or a dedicated tag-update service is the right answer at that scale.

**Option 2: Flux Image Automation.** Flux has a built-in image update controller that polls the container registry, detects new tags matching a semver or regex filter policy, and commits the tag update directly to the config repo without any CI involvement:

```yaml
# flux-system/image-automation/api-image-repository.yaml
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageRepository
metadata:
  name: api-image-repository
  namespace: flux-system
spec:
  image: ghcr.io/org/api
  interval: 1m0s
  secretRef:
    name: ghcr-credentials
---
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImagePolicy
metadata:
  name: api-image-policy
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: api-image-repository
  policy:
    semver:
      range: ">=1.0.0-0"
---
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImageUpdateAutomation
metadata:
  name: api-image-automation
  namespace: flux-system
spec:
  interval: 5m0s
  sourceRef:
    kind: GitRepository
    name: config-repo
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        email: fluxcdbot@org.com
        name: Flux Image Updater
      messageTemplate: |
        chore: update {{range .Updated.Images}}{{.}}{{end}}

        Updated by Flux image automation controller.
        Policy: api-image-policy
    push:
      branch: main
  update:
    path: ./environments/dev
    strategy: Setters
```

This requires annotating the kustomization.yaml to tell Flux which field to update with a special comment marker:

```yaml
# environments/dev/kustomization.yaml (with Flux image setter annotation)
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - ../../base
images:
  - name: ghcr.io/org/api
    newTag: "v1.14.2" # {"$imagepolicy": "flux-system:api-image-policy:tag"}
```

The `# {"$imagepolicy": ...}` comment is the Flux image setter marker. Flux finds the line, replaces the value, and commits. Clean, automatic, no CI coupling. No concurrent write conflicts because Flux serializes its own commits.

**Option 3: Renovate Bot.** Renovate is a dependency update bot that now supports Kubernetes manifests and Helm values files. You configure it in the config repo's `renovate.json` and it opens PRs that bump image tags:

```json
{
  "extends": ["config:base"],
  "kubernetes": {
    "fileMatch": ["environments/.+\\.yaml$"]
  },
  "packageRules": [
    {
      "matchPackagePatterns": ["ghcr.io/org/api"],
      "automerge": true,
      "automergeType": "pr",
      "matchUpdateTypes": ["digest", "patch"],
      "matchFileNames": ["environments/dev/**"]
    },
    {
      "matchPackagePatterns": ["ghcr.io/org/api"],
      "automerge": false,
      "matchUpdateTypes": ["minor", "major"],
      "matchFileNames": ["environments/staging/**", "environments/prod/**"],
      "labels": ["promotion", "needs-review"]
    }
  ]
}
```

This is the most "reviewed" approach — every tag bump in staging and prod is a PR, and Renovate can auto-merge dev updates. The downside is latency: Renovate runs on a schedule (typically every hour), so dev updates are not instantaneous on merge. For teams with tight feedback loops, Option 1 or a CI-triggered Renovate run is better.

For most teams: use Option 1 (CI commit) for dev, Option 1 with PR for staging and prod, at scale up to 20–30 builds per hour. Use Option 2 (Flux image automation) when build frequency exceeds that threshold or when you want the config repo to be fully self-contained without CI coupling. Reserve Option 3 (Renovate) for organizations that already use Renovate for dependency management and want a unified tool.

## 8. The full multi-environment promotion pipeline

Putting it all together, the full multi-environment promotion pipeline has five distinct phases, each with its own gate:

![Multi-environment promotion timeline from code merge through dev, staging, and prod with time estimates at each step](/imgs/blogs/promoting-releases-with-gitops-5.png)

**Phase 1: CI build (T+0 to T+8 min).** The developer merges their PR to main. CI builds the Docker image using BuildKit with layer caching, tags it with the short Git SHA, runs unit tests and a Trivy vulnerability scan, and pushes to GHCR. If any step fails, the pipeline stops. No config repo is touched. Build time: 8 minutes for a typical Go service with a warm BuildKit cache (cold cache: 22 minutes; the cache-from/cache-to GitHub Actions cache cuts this by 60%). This is the baseline gate that every CI system has; what GitOps adds is that nothing beyond this point is manual.

**Phase 2: Dev auto-deploy (T+8 to T+12 min).** The CI workflow commits the new tag to `environments/dev/kustomization.yaml` on the config repo's main branch. Argo CD detects the change (via 3-minute polling interval, or via a GitHub Actions webhook call to `argocd app get api-dev --refresh`) and reconciles the dev cluster. The Kubernetes Deployment controller rolls out the new pods with the rolling update strategy. If the rollout fails — readiness probe does not pass within the 5-minute timeout — Argo CD marks the Application as degraded and sends an alert to the team Slack channel.

**Phase 3: Dev smoke tests (T+12 to T+22 min).** A separate CI workflow (triggered by the config repo commit, or by a step in the main workflow that waits for the rollout) runs a smoke test suite against the dev endpoint. This suite covers the ten most critical user flows: login, data fetch, write, delete, and a subset of the API contract. If any test fails, the workflow posts a failure comment to the original app repo PR and marks the smoke test status check as failed. No staging PR is opened.

This is the gate that most teams underinvest in. The smoke tests do not need to be comprehensive — that is what the staging acceptance test suite is for. They need to be fast (under 10 minutes) and reliable (zero false positives over 30 days). A flaky smoke test that fails 5% of the time on real good builds is worse than no smoke test, because it trains the team to ignore failures. See [the test stage: fast feedback vs. confidence](/blog/software-development/ci-cd/the-test-stage-fast-feedback-vs-confidence) for the test-pyramid framing.

**Phase 4: Staging PR (T+22 to T+60 min).** On smoke test success, the CI workflow opens a PR to the staging overlay with the same image tag. The PR description includes the image tag, the app repo commit SHA, a link to the CI run, and the smoke test results. A reviewer — typically the team lead, the SRE on call, or the developer's peer — reviews the PR. For a single image tag change, this review takes 2–5 minutes. The reviewer checks the diff, checks the linked test results, and merges. Time in this phase is dominated by the reviewer's availability, not the technical operations.

**Phase 5: Staging acceptance tests and prod PR (T+60 to T+150 min).** After staging deploys (another Argo CD reconcile cycle, 30 seconds to a few minutes depending on rollout strategy), a longer acceptance test suite runs. This suite takes 30–40 minutes and covers the full user journey including edge cases, error paths, and performance regressions. On pass, the CI workflow opens a prod PR requiring two approvals. After approval and merge, Argo CD syncs the prod cluster. Total lead time from merge to prod: 90–150 minutes depending on reviewer availability.

#### Worked example:

Before GitOps, a twelve-engineer team shipping a B2B SaaS platform had a deploy frequency of twice per week. Each deploy was a two-hour event: 30 minutes of preparation (checking which features are ready), 30 minutes of coordination (getting everyone on a deploy call), 30 minutes of actual deployment, and 30 minutes of post-deploy monitoring. The opportunity cost was 4 engineer-hours per deploy, twice per week — 8 engineer-hours per week, or roughly \$48,000/year in engineering time at a \$120/hour fully-loaded rate, just for deployment coordination.

After implementing the five-phase GitOps pipeline: deployments are asynchronous and non-coordinated. Each developer merges when their feature is ready. The dev auto-deploy and staging PR happen without any coordination call. The prod PR requires one additional approval beyond the code review, but that is an async 90-second review. The engineering time per deployment dropped from 4 engineer-hours to 6 minutes of review time. Annually: \$3,000 instead of \$48,000, plus the lead time dropped from 3.5 days to 2 hours.

This calculation ignores the change-failure rate improvement — fewer rollbacks also mean less coordination overhead. A 13-percentage-point drop in change-failure rate (from 18% to 5%) at 50 deploys per week means 6.5 fewer failed deploys per week. If each failed deploy takes 90 minutes to remediate (rollback + investigation + communication), that is 9.75 hours per week, or roughly \$60,000/year in remediation cost eliminated. The total financial argument for GitOps-based promotion, for a twelve-person team at reasonable salary levels, is on the order of \$100,000/year in avoided cost — and that is before counting the reliability improvement's effect on customer churn.

## 9. Preview environments with Argo CD ApplicationSet

Preview environments — also called ephemeral environments or PR environments — are one of the highest-leverage features in the GitOps toolkit. For every open pull request in the application repo, the system spins up a complete isolated copy of the service, gives it a unique URL, and tears it down when the PR is closed.

![Before and after comparison of shared dev namespace causing PR collisions versus ApplicationSet-isolated preview namespaces per pull request](/imgs/blogs/promoting-releases-with-gitops-6.png)

Without preview environments, PR review is limited to code review. The reviewer reads the diff and guesses what the running system will look like. With preview environments, the reviewer can click a link and test the actual behavior. Change-failure rate drops because bugs that survive code review get caught in the preview environment before they ever reach dev. Teams that add preview environments typically see an additional 2–4 percentage point drop in change-failure rate on top of the improvement from the GitOps promotion pipeline itself.

The Argo CD ApplicationSet `PullRequest` generator makes this nearly automatic. The generator polls GitHub (or GitLab, Gitea, Bitbucket) for open pull requests and generates one Argo CD Application per PR:

```yaml
# argocd/applicationsets/api-preview.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: api-preview
  namespace: argocd
spec:
  generators:
    - pullRequest:
        github:
          owner: org
          repo: api
          tokenRef:
            secretName: github-token
            key: token
          labels:
            - preview  # only PRs with this label get a preview env
        requeueAfterSeconds: 30
  template:
    metadata:
      name: "api-preview-{{number}}"
      annotations:
        notifications.argoproj.io/subscribe.on-deployed.github: ""
    spec:
      project: preview
      source:
        repoURL: https://github.com/org/config-repo.git
        targetRevision: HEAD
        path: environments/dev
        kustomize:
          images:
            - "ghcr.io/org/api:{{head_sha}}"
      destination:
        server: https://kubernetes.default.svc
        namespace: "preview-{{number}}"
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
      info:
        - name: Preview URL
          value: "https://preview-{{number}}.dev.example.com"
        - name: PR Head SHA
          value: "{{head_sha}}"
        - name: PR Branch
          value: "{{branch}}"
```

When a developer opens a PR and adds the `preview` label, the ApplicationSet controller creates a new Argo CD Application named `api-preview-{PR-number}`. The Application syncs the dev overlay into a namespace called `preview-{PR-number}`, substituting the PR's head commit SHA as the image tag. The developer gets a running copy of their changes within 3–5 minutes of adding the label.

When the PR is closed (merged or abandoned), the ApplicationSet controller deletes the Application, which cascades to delete the namespace and all resources in it. No manual cleanup required.

To expose the preview URL via a wildcard ingress, you need two pieces: a wildcard DNS record and an Ingress resource in each preview namespace. The Ingress can be managed through a patch in the dev kustomization or through a separate ApplicationSet template field:

```yaml
# argocd/applicationsets/api-preview.yaml  (extended template)
  template:
    spec:
      source:
        path: environments/dev
        kustomize:
          images:
            - "ghcr.io/org/api:{{head_sha}}"
          patches:
            - target:
                kind: Ingress
                name: api
              patch: |
                - op: replace
                  path: /spec/rules/0/host
                  value: preview-{{number}}.dev.example.com
```

The wildcard DNS entry `*.dev.example.com` points at the ingress controller's load balancer IP. A wildcard TLS certificate (via cert-manager with a wildcard Let's Encrypt certificate or an internal CA issuer) secures all preview URLs automatically.

### Resource limits for preview environments

Preview environments should be constrained to avoid runaway cost. Use a LimitRange in each preview namespace to enforce lower resource limits than staging or prod:

```yaml
# base/limitrange.yaml  (applied only in preview via overlay)
apiVersion: v1
kind: LimitRange
metadata:
  name: preview-limits
spec:
  limits:
    - type: Container
      default:
        cpu: "200m"
        memory: "256Mi"
      defaultRequest:
        cpu: "50m"
        memory: "64Mi"
      max:
        cpu: "500m"
        memory: "512Mi"
```

Also add a TTL annotation on the namespace to auto-delete previews that have been open for more than 7 days (useful for PRs where the developer forgot to close them):

```yaml
# In the ApplicationSet namespace template, add a label for garbage collection
  template:
    metadata:
      namespace: "preview-{{number}}"
      labels:
        preview.org.com/pr: "{{number}}"
        preview.org.com/created: "{{createdAt}}"
        preview.org.com/ttl: "7d"
```

A simple CronJob that runs daily and deletes preview namespaces older than 7 days handles the garbage collection. Alternatively, use the Argo CD Application TTL plugin if your Argo CD instance has it installed.

## 10. Promotion gate options and when to use each

Not every team needs a full five-phase pipeline from day one. The right set of gates depends on team size, service criticality, and compliance requirements.

![Matrix comparing auto-promote, PR-gate, and manual-click options across speed, audit trail, risk control, and team-size fit](/imgs/blogs/promoting-releases-with-gitops-4.png)

Three common patterns exist at different points on the automation-control spectrum. The matrix above summarizes them; here is the reasoning behind each cell.

**Auto-promote all environments.** CI pushes directly to the dev, staging, and prod overlays on every merge. Argo CD syncs all three without any human in the loop. The advantages are maximal deploy frequency and minimal lead time — every merged commit is live in prod within 15 minutes. The disadvantages are significant: there is no human approval gate before prod changes, and the audit trail (Git commits from a CI bot) does not satisfy most compliance auditors who want a human sign-off on each production change. This pattern is appropriate for solo developers, for internal tools with low blast radius, and for "dogfooding" services used only by the engineering team.

**PR-gate all promotions.** Every environment change goes through a PR with at least one human reviewer. Dev gets a PR (automated, usually auto-merged within minutes), staging gets a PR (reviewed by a peer, merged within 30–60 minutes), and prod gets a PR with two required approvals (reviewed by a senior engineer and a team lead, merged within 1–2 hours). This is the pattern described throughout this post and the one that scales from 5 engineers to 500. The PR is the audit record; the CODEOWNERS file ensures the right people review the right overlays; the Git history is the compliance record.

**Manual-click for prod, auto for staging.** Dev and staging auto-deploy; prod requires a human to click "Sync" in the Argo CD UI or run `argocd app sync api-prod`. This is a common intermediate step for teams transitioning from manual deployments. It provides more safety than full automation, because a human must initiate the prod sync. The disadvantage: the click is not in Git, so the audit trail for prod deploys lives in Argo CD's event log, not in a version-controlled system with reviewer attribution. For most compliance frameworks (SOC 2 Type II, ISO 27001), this is insufficient — auditors want a named human who approved the change, in a system that is independently auditable.

The guidance table for choosing:

| Team size | Service criticality | Compliance requirement | Recommended gate pattern |
|---|---|---|---|
| 1–3 engineers | Internal tool, low blast radius | None | Auto-promote all environments |
| 1–3 engineers | Customer-facing SaaS | None | Auto dev + staging, PR-gate prod |
| 4–15 engineers | Customer-facing SaaS | None | Auto dev, PR-gate staging + prod |
| 4–15 engineers | Payments or PII processing | SOC 2 Type II or PCI DSS | PR-gate all envs, 2-approval prod |
| 16+ engineers | Any criticality | Any | PR-gate all envs, 2-approval prod, CODEOWNERS, mandatory smoke tests |
| 16+ engineers | Regulated workload | HIPAA, PCI, FedRAMP | PR-gate all envs, 2-approval prod, rendered YAML, cluster-per-env |

## 11. Stress-testing the promotion pipeline: what can go wrong?

Building the pipeline is the straightforward part. The hard part is knowing what happens when the environment misbehaves. Here are the failure modes you will encounter and how to handle each.

**Concurrent CI runs racing to update the dev overlay.** When two developers merge PRs within seconds of each other, both CI runs will try to commit to `environments/dev/kustomization.yaml` at the same time. The second push fails with a non-fast-forward error. The fix is a retry loop with rebase in the CI step:

```bash
cd /tmp/config-repo
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
  git pull --rebase origin main
  sed -i "s|newTag:.*|newTag: \"$TAG\"|" environments/dev/kustomization.yaml
  git add environments/dev/kustomization.yaml
  git commit -m "chore(dev): bump api to $TAG" || true
  if git push origin main; then
    echo "Push succeeded on attempt $i"
    break
  fi
  echo "Push failed on attempt $i, retrying in ${i}s..."
  sleep $i
done
```

At more than 30 concurrent builds per hour, even this retry loop starts to degrade. At that point, switch to Flux image automation (Option 2 from section 7) or to a dedicated tag-update service that serializes config repo writes.

**Argo CD sync fails mid-rollout.** The Argo CD Application goes to `Progressing` state, waits for the Deployment rollout, and eventually times out because the new pods cannot pass their readiness probes. The cluster is now in a partially-updated state — some pods running the new image, some running the old. Argo CD marks the Application as `Degraded`.

Kubernetes handles this gracefully if the Deployment has a `maxUnavailable: 0` rolling update policy — old pods stay running while new pods fail to start, so traffic is not interrupted. But the deploy is "stuck." The on-call engineer needs to:
1. Check pod logs: `kubectl logs -n api-dev -l app=api --previous`
2. Identify the failure: typically a misconfigured environment variable, a missing secret, or a failing health check path
3. Fix the application code, push a new commit, wait for a new CI run to promote a new tag

The GitOps-native rollback in this case is not to revert the config repo (the new image is broken, not the config), but to push a fix commit to the application repo. If the fix cannot be written quickly, the faster path is to revert the config repo commit that bumped the dev tag, which restores the previous known-good image.

**The config repo's main branch is force-pushed by accident.** A developer with admin access runs `git push --force origin main` on the config repo, rolling back the branch to a previous commit. Argo CD's self-heal feature kicks in — it detects that the cluster state is now ahead of the Git desired state (because the Git state just regressed) and tries to reconcile. If `prune: true` is set, it will delete resources that were created after the forced-back commit. This is a genuine destructive scenario.

Prevention: set branch protection on the config repo's main branch with "Require pull request before merging" and "Block force pushes" both enabled. Argo CD's reconcile loop is powerful; protecting the source of truth from destructive operations is your first line of defense.

**The staging PR is approved but staging has already diverged from dev.** Between the time the dev smoke tests passed and the staging PR is reviewed (potentially 30–60 minutes of async time), another change may have been promoted to dev. The staging PR now promotes version N while dev is already running version N+1. This is fine — staging and prod are allowed to lag behind dev. The config repo is designed to have each environment at its own release point. Only if the staging PR's image has a critical bug that was already fixed in dev is this a problem, and the solution is to update the staging PR branch to the newer tag before merging.

**Argo CD loses connectivity to the config repo.** The Argo CD application controller cannot reach GitHub (network outage, rate limit, GitHub incident). Argo CD transitions Applications to `Unknown` state but does not change the cluster — it is pull-based, so loss of connectivity means no changes, not a rollback. The cluster keeps running whatever was last successfully synced. When connectivity is restored, Argo CD resumes polling and catches up. This is one of the key advantages of pull-based GitOps over push-based CI/CD: a CI failure that stops CI from running does not also stop the cluster from running.

**A staging PR is merged but the staging rollout fails immediately.** The tag was valid, but the staging cluster has a different ConfigMap value than dev, and the new image's behavior with that config causes a crash loop. Argo CD marks `api-staging` as `Degraded`. The fix is to revert the staging overlay commit (a one-line revert PR), but now you have a discrepancy: dev has version N, staging has reverted to version N-1, and the prod overlay was never touched. This is the correct state — staging failed its gate, so prod is not affected. After fixing the config issue and the application bug, re-promote from dev to staging.

#### Worked example:

A team of 8 engineers runs the full GitOps pipeline described in this post. They measure their pipeline over 90 days (approximately 1,800 CI runs) and find:

- 1,680 successful dev promotions (93.3%)
- 120 failed dev promotions: 85 due to CI failures (unit tests, Trivy scan), 22 due to dev rollout failures (readiness probe timeout), 13 due to config repo concurrent write conflicts
- Of the 1,680 successful dev promotions, 1,590 opened staging PRs (94.6% passed smoke tests)
- Of the 1,590 staging PRs, 1,520 were merged (95.6% merged rate; 70 were abandoned due to feature reverts or bug discoveries)
- Of the 1,520 staging merges, 1,489 resulted in successful staging rollouts (97.9%)
- Of the 1,489 successful staging rollouts, 1,452 opened prod PRs (97.5% passed acceptance tests)
- Of the 1,452 prod PRs, 1,440 were merged (99.2%)
- Of the 1,440 prod merges, 1,433 resulted in successful prod rollouts (99.5%)

The 7 failed prod rollouts (0.49% of prod deploys) each required a revert PR. Average MTTR: 9 minutes (one-line revert PR, 2-minute review, 30-second Argo CD reconcile, 5-minute rollout, 1-minute verification). Before GitOps, MTTR for failed deploys was 78 minutes.

The 13 concurrent write conflicts in 90 days (one per week) were resolved by the retry loop in every case. None resulted in a lost promotion. Adding Flux image automation in month 4 reduced concurrent conflicts to zero.

This data shows the compound reliability of the multi-gate pipeline: the probability of a broken change reaching prod is the product of the failure rates at each gate, not their sum. A 5% dev-failure rate times a 4.4% staging-failure rate times a 0.5% prod-failure rate means that a broken change that passes CI has roughly a 0.02% chance of reaching production undetected — 1 in 5,000 deploys.

## 12. War story: the Tuesday afternoon config drift incident

This incident is a composite of two real events at different companies; the details are altered to protect identities, but the failure mode is identical.

A mid-size SaaS company ran Argo CD against their config repo. The repo had three overlays: dev (auto-sync enabled), staging (auto-sync enabled), prod (auto-sync disabled, manual sync required). On a Tuesday afternoon, a platform engineer opened the Argo CD UI to sync an approved database migration to prod. In the sync dialog, they saw three Applications listed with pending changes. They selected all three in the batch-sync interface — including `api-prod`, which had a pending but unreviewed change that had been accumulating in the prod overlay for 48 hours.

That pending change was a resource limit increase — from `cpu: 500m` to `cpu: 2000m` — submitted by a developer for a load test that was supposed to run in staging. The developer had run `sed -i` on all three overlay files simultaneously, not realizing they were editing prod. They had committed and pushed to the config repo main branch. Because there was no PR gate on the prod overlay, the change landed directly on main without review.

The platform engineer, performing what they believed was a safe database-migration sync, inadvertently applied the resource limit change to prod. The prod node pool's total CPU request jumped beyond the node pool's capacity. The Kubernetes scheduler started evicting lower-priority pods to make room. The API service degraded — latency spiked, some requests returned 503 — for 18 minutes before the platform engineer identified the change, reverted it via a direct kubectl patch (bypassing GitOps, which would have reconciled back to the wrong state anyway), and then cleaned up the config repo.

The MTTR was 18 minutes rather than seconds because:
1. The engineer did not know which of the three applications they had synced was causing the problem.
2. Looking at the Argo CD event log took 3 minutes.
3. Reading the diff took 2 minutes (the prod overlay change was subtle).
4. The rollback required both a kubectl patch and a config repo revert to stay in sync.

The post-mortem changes:
1. **PR-gate the prod overlay.** Auto-sync is disabled and a CODEOWNERS rule requires a review from the platform team before any change to `environments/prod/` can merge.
2. **Branch protection on config repo main.** Direct pushes to main are disabled. All changes require a PR, even one-line changes.
3. **Lint rule in CI.** A CI check in the config repo fails if any patch file appears in both the staging and prod overlay directories without being under a separate named feature directory. This surfaces the "I edited all three envs with sed" mistake before it reaches main.
4. **Argo CD project AppProject restriction.** The `prod` project in Argo CD is configured to only allow syncing from branches that match `releases/*`, not from `main` directly. This means even if someone pushes a change to `environments/prod/` directly on main, Argo CD will not sync it until a release branch is created — adding a forced checkpoint.

The deeper lesson: GitOps does not prevent human error. It creates a structural audit trail and shortens the rollback path, but it does not stop a well-meaning engineer from doing something wrong in the UI. The fix is to move the gates upstream, into the Git repository access controls, rather than relying on UI discipline.

## 12. How to reach for this — and when not to

GitOps-based promotion is not free. It adds a config repo to maintain, a PR automation workflow to keep operational, and an Argo CD or Flux installation to run. Before implementing the full stack described in this post, be honest about the following:

**When to use the full pattern:**
- You have more than one environment (dev + prod at minimum, staging strongly preferred).
- More than one person deploys to production.
- You have had at least one production incident where "what changed?" took more than 5 minutes to answer.
- You are subject to a compliance audit (SOC 2, PCI DSS, HIPAA, ISO 27001) that requires change approval records with named human approvers.
- You want to roll back production in under 5 minutes without a runbook.
- Your MTTR is above 30 minutes and the bottleneck is "figuring out what to revert."

**When to skip it or simplify:**
- You are a solo developer on a project with a single environment. Use `git push heroku main` or a PaaS (Railway, Render, Fly.io). Adding a config repo and Argo CD for a solo project is pure overhead.
- Your service has no readiness probe and no meaningful smoke tests. GitOps promotion without a test gate is automated deployments with extra steps. Write the tests first, then add the promotion pipeline.
- Your deployment target is a serverless platform (AWS Lambda, Google Cloud Run, Vercel). These platforms have their own promotion mechanisms — aliases, traffic splitting, revision management — that are simpler and better integrated than Kubernetes-native GitOps for that compute model.
- You do not have a Kubernetes cluster. Argo CD and Flux are Kubernetes-native. For VM-based deployments or Terraform-managed infrastructure, a Terraform workspace-per-environment pattern is a closer fit. See [managing Terraform safely at scale](/blog/software-development/ci-cd/managing-terraform-safely-at-scale).
- Your service has one deployment per year and changing the deploy process costs more than the improvement it provides. If you deploy four times a year and each deployment takes an hour, the GitOps automation pays off in year three. In year one it is a net cost.

**When to phase the implementation (recommended for most teams):**
Start with the config repo structure and the CI tag-bump step. Get Argo CD syncing from the config repo. Add the PR-gate for prod. That alone delivers 70% of the benefit: audit trail, reproducible state, fast rollback. Then add smoke tests. Then staging PR automation. Then preview environments. Then multi-cluster ApplicationSet. This order matters — the config repo and Argo CD sync are the foundation; everything else is layered on top.

The toolchain has a learning curve. Budget 2–3 days for a motivated engineer to set up the basic structure, 1–2 weeks to get the full PR automation working reliably, and 1 month for the team to feel fluent with it. The investment compounds: every month of operating under GitOps is another month of the team internalizing "change the Git file, Argo CD does the rest," which is the mindset shift that makes everything else in this series work. Cross-link: see [Argo CD and Flux in practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) for the deep dive on sync waves, self-heal policies, and Argo CD's notification framework that powers the alert-on-degraded behavior mentioned in Phase 2.

## Key takeaways

1. **The config repo is the deployment contract.** Separate it from the application repo so that deployment decisions are reviewed independently from implementation decisions. Blast radius is smaller; review quality is higher; the audit trail is free because it is just Git.

2. **One image digest, three overlays.** The same `sha256:...` digest flows through dev, staging, and prod. Only the tag reference changes per overlay. "Build once, promote everywhere" is a structural property of the config repo, not a reminder on a Confluence page.

3. **The PR is the approval gate.** Merge = deploy. Every change to what runs in production is a PR with an author, a reviewer, a timestamp, and a diff. This replaces the deployment ticket, the Slack thread, and the terminal session history as the canonical record of what happened and who authorized it.

4. **Auto-promote dev, PR-gate staging and prod.** Dev should move fast — CI commits directly. Staging and prod get PRs. Prod gets two approvals. This balance gives you fast feedback loops on every merge without sacrificing control at the production boundary.

5. **Rollback is a revert PR.** When production breaks, the first action is opening a one-line PR bumping the tag back to the last known-good version. Argo CD reconciles in 30 seconds. MTTR drops from 90 minutes to 8 minutes because rollback has a rehearsed, well-known path that any on-call engineer can execute at 2 AM.

6. **Preview environments multiply reviewer effectiveness.** An Argo CD ApplicationSet PullRequest generator creates one namespace per PR, auto-deletes it on PR close, and gives reviewers a live URL to test against. Change-failure rate drops an additional 2–4 percentage points on top of the improvement from the promotion gate alone.

7. **The Argo CD event log is not an audit trail.** For compliance, the Git PR history with named reviewers IS the audit trail. Keep it clean: one commit per promotion, PR body links to CI results, CODEOWNERS rules enforce the right reviewers for the right directories.

8. **Gate the config repo main branch.** Branch protection rules that require PRs for all changes are the single most effective control against the "I edited all three environments with sed and pushed directly" failure mode. The access control is your first line of defense; GitOps is the second.

9. **Phase the implementation.** Config repo + Argo CD sync + prod PR gate delivers 70% of the value. Add smoke tests, staging automation, and preview environments as the team matures. Do not try to implement everything in week one.

10. **Measure the before and after.** Track deploy frequency, lead time, change-failure rate, and MTTR before and after implementing GitOps promotion. The numbers will justify the investment, surface the bottlenecks, and tell you what to optimize next.

## Further reading

- [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the full commit→build→test→package→deploy→operate spine and the DORA framework this post sits inside.
- [GitOps: Git as the source of truth](/blog/software-development/ci-cd/gitops-git-as-the-source-of-truth) — the foundational GitOps principles: pull-based reconciliation, declarative desired state, and drift detection that underpins everything in this post.
- [Argo CD and Flux in practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) — deep dive on Argo CD Application resources, sync waves, self-heal policies, health checks, the Flux image automation controller, and the notification framework.
- [Multi-environment promotion: dev, staging, prod](/blog/software-development/ci-cd/multi-environment-promotion-dev-staging-prod) — the environment promotion pattern from the pipeline perspective, covering environment-specific configuration, feature flags as alternatives to long-lived branches, and the environment parity principle.
- [Helm vs Kustomize: templating your manifests](/blog/software-development/ci-cd/helm-vs-kustomize-templating-your-manifests) — the full trade-off analysis between Helm charts and Kustomize overlays, including the hybrid "Helm chart rendered by Kustomize" pattern used when a third-party chart needs per-environment overrides.
- DORA State of DevOps Report 2023 — the empirical research behind the four metrics and the elite/high/medium/low performer classification. The report is free from the DORA research program website.
- Argo CD ApplicationSet documentation — the full reference for generators (List, Cluster, Git, PullRequest, Matrix, Merge), template fields, and the ApplicationSet controller's reconciliation behavior.
- Flux Image Automation documentation — the complete reference for ImageRepository, ImagePolicy, ImageUpdateAutomation, and the `$imagepolicy` comment marker syntax.

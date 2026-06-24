---
title: "Push vs pull CD and who holds the keys"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand the fundamental architectural choice between push-based and pull-based continuous delivery, why it controls your blast radius when CI is compromised, and how to migrate to a pull-based GitOps model without downtime."
tags:
  [
    "ci-cd",
    "devops",
    "gitops",
    "kubernetes",
    "argo-cd",
    "flux",
    "security",
    "continuous-delivery",
    "oidc",
    "deploy-strategy",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-1.png"
---

The deploy key was in the CI job environment variable, named `KUBE_CONFIG`, base64-encoded, and sitting in the secrets vault of a SaaS CI provider. It had been there for two years. Nobody thought about it much — it just worked. The pipeline would finish its tests, call `kubectl apply -f k8s/`, and the cluster would update. Clean, simple, fifteen lines of YAML.

Then the CI provider disclosed a breach. The attacker had read access to customer secrets for roughly eleven days. Within forty-eight hours, several teams were rotating credentials, auditing their clusters for unauthorized workloads, and filing incident reports. The deploy key gave whoever had it the same access as the release engineer running `kubectl` from a laptop. It was a kubeconfig for the production namespace. Not staging. Not a preview environment. Production.

That incident — and the dozens like it that never make the news — is the reason the architectural choice between push-based and pull-based continuous delivery is not a matter of taste. It is a security boundary decision. Where do your production credentials live? Who can read them? What happens when the system that holds them is compromised? How many steps does an attacker need to take between "CI is breached" and "production is breached"?

This post traces both architectures from first principles. You will walk away able to draw the credential flow for each model, explain why pull-based CD is structurally more secure even without OIDC federation, wire up a GitHub Actions push-CD pipeline and a Flux-based pull-CD pipeline for the same service, reason about when push-based is a perfectly reasonable choice, and migrate an existing push-based system to pull-based incrementally without a big-bang cutover. Figure 1 previews the core contrast we will spend the whole post unpacking.

![Push-based CD places the kubeconfig inside CI which must reach out to the cluster; pull-based CD inverts this so the in-cluster operator polls Git and the cluster never accepts inbound connections from CI runners](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-1.png)

---

## 1. The vocabulary problem: CI, CD, and the deploy loop

Before getting into architecture, a terminology reset is needed because the industry has used "CI/CD" as a single compound noun long enough that the boundary between them has blurred. That blurring causes the design confusion this post addresses.

**Continuous integration (CI)** is the practice of merging developer branches into a shared mainline frequently — ideally multiple times per day — and verifying each merge with an automated build and test suite. CI ends the moment the verified artifact — the image, the JAR, the binary — and its accompanying metadata (the Git tag, the image digest, the SBOM) are published to a durable store. CI does not care where the artifact ultimately runs. CI's job is to produce a trustworthy, immutable artifact. That's it.

**Continuous delivery (CD)** is the practice of ensuring that the verified artifact can be released to production at any time, automatically or with a one-click approval. CD begins where CI ends: something must detect the new artifact, decide whether to promote it, and apply it to the target environment. The "who" and "how" of that promotion is precisely what separates push-based from pull-based architectures.

**Continuous deployment** (also abbreviated CD, which creates the collision) goes one step further: every artifact that passes CI gates is automatically deployed to production without a human approval step. Not every team wants or needs continuous deployment; continuous delivery is the more conservative superset that most regulated or safety-critical teams implement.

The key insight is that CI and CD are **separate concerns governed by separate systems**. CI is triggered by a code push. CD is triggered by an artifact promotion event — which might be automatic or might require a human to merge a PR or approve a pipeline stage. Once you see them as separate, the architectural question becomes obvious: where does the CD system live, and what credentials does it need?

In push-based CD, the answer is: "CI is the CD system, and it needs credentials that reach production." In pull-based CD, the answer is: "A dedicated operator running in the cluster is the CD system, and CI does not hold production credentials at all."

That difference in who holds the credentials — and where those credentials live — is the entire argument of this post.

There is a third dimension worth naming before the architecture deep-dive: **observability of desired state**. In push-based CD, what is running in production is determined by the last successful CI deploy job. If that job failed halfway through, or if someone ran `kubectl edit` after it, the actual state of the cluster may differ from what the CI logs claim. There is no single place you can look to know with certainty what the cluster should be running. You must check the CI logs (which might be stale or rotated) and the live cluster state and hope they match.

In pull-based CD, the answer is always Git. The HEAD of the config repository main branch is what the cluster should be running. The operator continuously verifies this and alerts if it does not match. "What should be running in production?" becomes a `git log` question rather than a CI log archaeology exercise. This observability property is as important as the security property for large teams operating under production reliability requirements.

---

## 2. Push-based CD: the CI system holds the keys

In push-based CD, the CI system is responsible for both building the artifact and deploying it. After tests pass, the CI job authenticates to the cluster, the cloud provider API, or the deployment target and runs the deploy command directly. The deploy is part of the same pipeline run that compiled the code and ran the tests.

![Blast radius comparison showing that a compromised CI system in push-based CD grants direct kubeconfig access to production, while pull-based CD limits an attacker to Git write access that still requires PR review before any cluster change](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-2.png)

The mechanics are straightforward. A GitHub Actions workflow stores a kubeconfig in an Actions secret (`KUBE_CONFIG`). The deploy job decodes the secret, writes it to `~/.kube/config`, and runs `kubectl apply` or `helm upgrade --install`. The CI runner is an ephemeral VM somewhere in GitHub's infrastructure. The deploy finishes. The secret is gone from memory when the runner terminates — until the next run.

Here is a representative push-based deploy workflow:

```yaml
# .github/workflows/deploy.yml  — push-based CD
name: Build and Deploy

on:
  push:
    branches: [main]

env:
  IMAGE_TAG: ${{ github.sha }}
  REGISTRY: ghcr.io/myorg/myservice

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Log in to GHCR
        run: echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u $ --password-stdin
      - name: Build and push image
        run: |
          docker build -t $REGISTRY:$IMAGE_TAG .
          docker push $REGISTRY:$IMAGE_TAG

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Write kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config
          chmod 600 ~/.kube/config
      - name: Deploy to cluster
        run: |
          helm upgrade --install myservice ./charts/myservice \
            --namespace production \
            --set image.repository=$REGISTRY \
            --set image.tag=$IMAGE_TAG \
            --wait \
            --timeout 5m
      - name: Verify rollout
        run: |
          kubectl rollout status deployment/myservice -n production --timeout=3m
```

The `KUBE_CONFIG` secret contains a service account token (or, in the worst case, a cluster-admin kubeconfig) with write access to the production namespace. Every CI run touches this secret. Every repository collaborator with Actions write permission can trigger a workflow run. The CI provider's infrastructure holds the secret for as long as you keep it there — which in practice is years, because rotating it is annoying.

### 2.1 The blast radius of a push-based credential breach

The blast radius arithmetic here is stark. If `KUBE_CONFIG` is a cluster-admin credential for the production namespace, then compromising CI gives an attacker everything. They can:

- Read all Kubernetes Secrets in the namespace — which typically contain database passwords, API keys, TLS certificates, and every other credential the application needs.
- Deploy arbitrary workloads: a cryptominer, a reverse shell, a data exfiltration container, a persistent backdoor as a DaemonSet.
- Delete Deployments or alter replica counts to cause a denial of service.
- Exfiltrate environment variables from running containers by exec-ing into pods using `kubectl exec`.
- Replace your application containers with a malicious image by patching the Deployment spec.
- Read ConfigMaps that may contain connection strings or other sensitive configuration.

The blast radius is not bounded by what CI normally does with the credential. It is bounded by what the credential permits. A service account with `cluster-admin` permissions is a skeleton key to everything in the cluster. Even a more restricted service account — say, one with `get`, `list`, `create`, `update` on Deployments and Services in one namespace — still allows an attacker to replace running containers with arbitrary images in that namespace.

The key question to ask about any push-based setup is: what is the minimum IAM permission this kubeconfig actually needs, and is that what it actually has? In practice, the answer to the second part is usually "no, it has far more," because the service account was created once by someone who wanted it to just work, granted broad permissions, and was never revisited.

### 2.2 OIDC federation: a meaningful improvement, not a complete fix

OIDC (OpenID Connect) federation to cloud providers lets GitHub Actions authenticate to AWS, GCP, or Azure by exchanging a short-lived OIDC token for a cloud IAM credential. This eliminates the need to store long-lived secrets in the CI secrets vault. There is no static `KUBE_CONFIG` to steal from the vault, because no static credential exists.

```yaml
# GitHub Actions OIDC-to-EKS push-based deploy
# No static KUBE_CONFIG secret required
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # required for OIDC token exchange
      contents: read
    steps:
      - name: Configure AWS credentials via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-deploy
          aws-region: us-east-1
      - name: Update kubeconfig for EKS
        run: aws eks update-kubeconfig --name prod-cluster --region us-east-1
      - name: Deploy via Helm
        run: |
          helm upgrade --install myservice ./charts/myservice \
            --namespace production \
            --set image.tag=${{ github.sha }} \
            --wait --timeout 5m
```

The IAM trust policy for the role restricts which repositories can assume it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:myorg/myservice:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

OIDC federation eliminates the stored static secret, which is a real and significant improvement. There is no credential in the vault to steal if the vault is breached. The token expires quickly (typically within one hour). The IAM role can be scoped to specific repositories, branches, and environments via the OIDC trust policy.

But OIDC federation does not eliminate the push-based blast radius — it reduces the window during which a stolen credential is useful. The token is still vended to the CI runner during the deploy job. If the runner is compromised during that window (a malicious action dependency, a supply chain attack in a transitive package, a container escape in the shared CI infrastructure), the attacker still has a valid cloud credential with cluster write permissions for however long the token is valid. The token is ephemeral, but the window is real.

More critically, OIDC federation does not change the network topology: the CI system still needs to reach the Kubernetes API server to run `kubectl` or `helm`. Push-based CD with OIDC is a significant security improvement over static secrets. It is not equivalent to pull-based CD.

---

## 3. Pull-based CD: the cluster agent holds the keys it needs, and CI holds none

In pull-based CD, the CI system's job ends when the artifact lands in the registry and the desired state is expressed in Git. A separate agent — running inside the cluster as a set of Kubernetes controllers — continuously watches a Git repository and optionally a container registry. When it detects a difference between what Git says should be running and what is actually running, it reconciles: it applies the updated manifest, rolls out the new image, and reports the sync status back to Git or a notification channel.

![Pull-based CD trust chain showing five layers from developer push through CI and registry to in-cluster operator, with CI holding only Git write access and the cluster agent handling all deployment actions](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-4.png)

The credential map in pull-based CD looks fundamentally different. CI needs:
- Permission to push images to the registry. This is a registry push credential — a short-lived OIDC-vended ECR token, a GitHub `GITHUB_TOKEN` for GHCR, or a robot account in Harbor with push-only access to one repository.
- Permission to write to the config repository. This is a GitHub App installation token or a deploy key with `contents:write` on the config repository only.

The in-cluster operator needs:
- An image pull secret stored as a Kubernetes Secret in the cluster — needed for the cluster to pull images from the registry during pod scheduling.
- A read-only deploy key or GitHub App credential to read the config repository. This is stored as a Kubernetes Secret in the `flux-system` or `argocd` namespace. CI never sees it.
- RBAC permissions to create and update Deployments, Services, ConfigMaps, and other resources in the namespaces it manages. This is a Kubernetes Role/ClusterRole binding, granted to the operator's service account, living entirely within the cluster.

At no point does CI hold a credential that allows it to communicate with the cluster API server. The cluster API server never needs to be reachable from the CI runner's network at all. The trust boundary is Git.

### 3.1 Flux-based pull-CD: the full mechanics

Flux is a CNCF graduated project implementing the GitOps reconcile loop for Kubernetes. Its architecture is a set of controllers (source-controller, kustomize-controller, helm-controller, image-reflector-controller, image-automation-controller, notification-controller) that watch custom resources and reconcile cluster state to match Git state.

A minimal Flux bootstrap for the `order-processor` service:

```bash
# One-time: bootstrap Flux into the cluster (run by a cluster admin with kubectl access)
# After this, CI never needs kubectl access again
flux bootstrap github \
  --owner=myorg \
  --repository=fleet-config \
  --branch=main \
  --path=clusters/production \
  --personal
```

This command:
1. Creates the `flux-system` namespace and installs all Flux controllers.
2. Generates an SSH deploy key for the `fleet-config` repository.
3. Stores the private key as a Kubernetes Secret in the cluster.
4. Adds the public key as a deploy key to the GitHub repository.
5. Commits the Flux manifests to `fleet-config` so Flux manages itself.

The CI system never participated in this. The deploy key is generated on the cluster and stored in the cluster. CI has no visibility into it.

Now add the application:

```yaml
# clusters/production/order-processor-source.yaml  (in fleet-config repo)
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: order-processor-config
  namespace: flux-system
spec:
  interval: 3m
  ref:
    branch: main
  url: https://github.com/myorg/order-processor-config
  secretRef:
    name: order-processor-config-auth  # read-only deploy key, stored in cluster
---
# clusters/production/order-processor-kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: order-processor
  namespace: flux-system
spec:
  interval: 5m
  path: ./production
  prune: true
  sourceRef:
    kind: GitRepository
    name: order-processor-config
  targetNamespace: production
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: order-processor
      namespace: production
  timeout: 5m
  retryInterval: 2m
```

And the CI step that "triggers" the deploy — which is actually just a Git commit:

```bash
# .github/workflows/ci.yml — the ONLY deploy-related CI step in pull-based CD
# CI never touches the cluster

NEW_TAG="${GITHUB_SHA:0:8}"
IMAGE="ghcr.io/myorg/order-processor:${NEW_TAG}"

# Clone the config repo using a GitHub App token (contents:write on config repo only)
git clone "https://x-access-token:${CONFIG_REPO_TOKEN}@github.com/myorg/order-processor-config.git" config-repo
cd config-repo

# Kustomize edit: updates the image reference cleanly
kustomize edit set image "ghcr.io/myorg/order-processor=${IMAGE}"

git config user.email "ci@myorg.com"
git config user.name "CI Bot"
git add kustomization.yaml
git diff --cached --exit-code && echo "No changes" || git commit -m "ci: deploy order-processor ${NEW_TAG}"
git push
```

Flux detects the new commit within 3 minutes (its poll interval) and applies the updated Kustomization. The cluster reconciles to the new image. CI never established a TCP connection to the Kubernetes API server.

### 3.2 Argo CD Application: the full spec

Argo CD is the other dominant pull-based CD operator, with a richer UI, an ApplicationSet controller for fleet management, and native integration with Argo Rollouts for progressive delivery.

```yaml
# argocd/applications/order-processor.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: order-processor
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io  # cascade-delete resources on app deletion
spec:
  project: production-apps                     # Argo CD Project scoping which repos/clusters
  source:
    repoURL: https://github.com/myorg/order-processor-config
    targetRevision: main
    path: clusters/production
  destination:
    server: https://kubernetes.default.svc     # in-cluster API server (no external reach needed)
    namespace: production
  syncPolicy:
    automated:
      prune: true       # delete resources removed from Git
      selfHeal: true    # reapply if cluster state drifts from Git
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  revisionHistoryLimit: 10
```

With `automated.selfHeal: true`, Argo CD continuously monitors the cluster for drift. If someone runs `kubectl edit deployment order-processor` directly in production and changes a memory limit, Argo CD will detect the divergence within its default 3-minute check interval and revert it. This makes drift detection a first-class, automatic feature — not something you remember to check occasionally.

The Argo CD `ApplicationSet` controller extends this to fleet management. One manifest targets all clusters:

```yaml
# argocd/applicationsets/order-processor-all-clusters.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: order-processor
  namespace: argocd
spec:
  generators:
    - clusters:
        selector:
          matchLabels:
            environment: production   # targets all clusters labeled as production
  template:
    metadata:
      name: "order-processor-{{name}}"
    spec:
      project: production-apps
      source:
        repoURL: https://github.com/myorg/order-processor-config
        targetRevision: main
        path: "clusters/{{metadata.labels.region}}/production"
      destination:
        server: "{{server}}"
        namespace: production
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

A single ApplicationSet drives deployment across every cluster in the fleet. Each cluster's operator pulls from the appropriate path in the config repo. Adding a new cluster means adding a new entry in the cluster registry and adding a label — not modifying any CI pipeline.

---

## 4. The credential argument, made precise

The security difference between push-based and pull-based CD is not a matter of opinion. It follows from two provable structural properties.

### 4.1 Where does the cluster credential live, and who can access it?

In push-based CD, the cluster write credential is accessible to:
1. Every CI run (it is loaded from the secrets vault for every deploy job).
2. Every repository admin (they can update the secret value).
3. The CI provider's infrastructure (they hold the encrypted secret).
4. Any malicious action or dependency that runs inside the CI job and exfiltrates environment variables.
5. Any supply chain attacker who compromises the CI provider's backend (as happened with Codecov).

In pull-based CD, the cluster write credential (the operator's RBAC service account token) is accessible to:
1. Kubernetes cluster admins (who can read Secrets with appropriate RBAC).
2. The in-cluster operator (which uses the service account token to make API calls).
3. Nobody else. CI never sees it. The CI provider never sees it. Supply chain attacks against CI do not expose it.

The structural advantage of pull-based CD is that the threat model for the cluster credential is entirely within the cluster's own security boundary. To steal the cluster credential in a pull-based setup, an attacker must first compromise the cluster itself — at which point you have bigger problems than a stolen deploy credential.

### 4.2 The scope-of-damage argument in numbers

It helps to be specific about what "blast radius" means in concrete, measurable terms. Assume a Kubernetes cluster running twenty microservices with typical production data access: a PostgreSQL connection string, an S3 bucket with customer PII, a Stripe API key, a Slack webhook, and a service-to-service JWT signing key.

If `KUBE_CONFIG` is cluster-admin and is stolen, the attacker's access includes:
- All 20 services' environment variables via `kubectl exec` or pod spec inspection — that is every credential listed above, for every service.
- The ability to read every Kubernetes Secret in the cluster — potentially hundreds.
- The ability to schedule new workloads in any namespace — enabling persistent access via DaemonSet or CronJob.
- The ability to exfiltrate data at full network speed by running a pod in the same VPC as the database.
- The ability to destroy the cluster by deleting namespaces.

If a GitHub App token with `contents:write` on the config repository is stolen, the attacker can:
- Push commits to the config repository — which, with branch protection, requires PR review before merging to main.
- If branch protection is absent, push directly to main and trigger an operator sync — which runs with the operator's scoped RBAC, not cluster-admin permissions.
- The operator's RBAC is scoped to specific namespaces and resource types. An attacker who triggers an operator sync can deploy a malicious image — but only to the namespaces the operator manages, only by going through the operator's apply logic, and with every change recorded as a Git commit.

The difference in scope is roughly:

```
Push-based breach scope  = all cluster secrets + arbitrary workload scheduling + data exfiltration
Pull-based breach scope  = config repo write + operator-scoped namespace changes + full Git history trail
```

This is not a small difference. Pull-based CD does not make breaches impossible. It makes the consequences of a CI breach dramatically more bounded and more detectable.

### 4.3 The network topology argument

![Network topology contrast showing push-based CD requiring an inbound connection through the firewall to the kube-apiserver, while pull-based CD needs only outbound port 443 from the cluster to Git and the registry](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-5.png)

Push-based CD requires the CI runner to establish a TCP connection to the Kubernetes API server. For a private cluster — a VPC-only EKS cluster, a GKE private cluster, an on-premises cluster in a datacenter — this means one of the following must be true:

- The cluster's API endpoint is public. (Simplest to configure, worst for security. The API server is exposed to the internet and must rely entirely on authentication and authorization to prevent abuse.)
- The CI runner is self-hosted inside the same VPC. (Requires running and maintaining your own CI runner fleet, tying CI availability to cluster network availability.)
- A private link or VPN connects the CI provider's network to the cluster's VPC. (Adds VPN infrastructure complexity, latency, and a new failure mode.)

Pull-based CD requires none of this. The in-cluster operator makes only outbound HTTPS connections on port 443 — to GitHub, to the container registry, to whatever notification system you configure. The API server endpoint does not need to be reachable from any external network. For air-gapped environments, the operator can be configured to use an internal Git mirror and an internal registry. The cluster security posture is: inbound connections from outside = none required for deployments.

For teams operating in regulated industries — financial services, healthcare, government — this is often not a preference but a compliance requirement. Many security frameworks mandate that production system administrative interfaces must not be reachable from the public internet. Pull-based CD satisfies that requirement without a VPN workaround.

---

## 5. Separating CI from CD: why the boundary matters for DORA

The DORA research framework (the Accelerate book and the annual State of DevOps reports) measures four metrics that predict organizational software delivery performance: deploy frequency, lead time for changes, change-failure rate, and time-to-restore service. Of these, deploy frequency and lead time are directly affected by where you draw the CI/CD boundary and what you count.

![CI to CD handoff timeline showing commit through a six-minute CI run to image push and Git tag, then the separate CD loop where the operator detects the tag and syncs the cluster to serve traffic](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-6.png)

**Deploy frequency** in pull-based CD must be measured at the operator sync layer, not the CI layer. In push-based CD, CI run completion = deploy (assuming the deploy job succeeds). In pull-based CD, CI completion means only that a new image and a new tag exist. The deploy happens when the operator syncs. If you want DORA metrics that accurately reflect how often you ship to production, instrument the operator's sync events, not the CI completions.

**Lead time for changes** decomposes differently in each model. The full path from code commit to traffic in production:

```
Lead time = PR review time
          + CI build and test time      (code commit → image+tag published)
          + Operator poll/detect time   (tag in Git → operator detects change)
          + Sync and rollout time       (operator applies manifest → pods ready)
          + Traffic propagation delay   (kube-proxy / LB propagates to endpoints)
```

With Flux at a 3-minute poll interval, you add at most 3 minutes to every lead time. For a team achieving total lead times of 2 hours (which DORA defines as "high performance"), 3 minutes is noise. If you need sub-minute lead times, Flux's webhook receiver and Argo CD's `argocd app sync` can be triggered from CI to eliminate the poll delay entirely.

**Change-failure rate** is where pull-based CD shows an indirect but real benefit. When CI and CD are separate, the CD operator becomes the single source of truth for what is running in production. You can gate a sync on health checks and rollback automatically if they fail. You can integrate Argo Rollouts or Flagger for progressive delivery with SLO-gated rollout. The CD operator is independently rollbackable without touching the CI pipeline.

In push-based CD, rollback means triggering a new CI pipeline run with an older image tag — which takes the full CI build time, runs all tests again (which you probably do not want to wait for in an emergency), and is a manual process. In pull-based CD, rollback is `git revert` followed by an operator sync — which can complete in under a minute.

### 5.1 The DORA correlation with pull-based CD

The 2023 State of DevOps report does not explicitly separate push-based from pull-based CD teams. But it does show a strong correlation between GitOps adoption and elite DORA performance. Teams practicing GitOps (which implies pull-based CD) are significantly more likely to be in the "elite" performance tier for deploy frequency and lead time.

The mechanism is not mysterious: GitOps teams have a single source of truth (Git) for desired cluster state. Every change is a Git commit. Rollback is a revert. Drift is detected and auto-healed. These properties make it structurally easier to deploy frequently and recover quickly. The four DORA metrics are a downstream effect of the architectural properties this post describes.

---

## 5b. GitLab CI and the push-based deploy model

The arguments in this post apply equally to GitLab CI, Jenkins, CircleCI, and any other CI platform. GitLab CI's model for push-based CD stores the kubeconfig as a CI/CD variable in the project or group settings:

```yaml
# .gitlab-ci.yml — push-based CD for GitLab CI
stages:
  - build
  - deploy

variables:
  IMAGE_TAG: $CI_COMMIT_SHORT_SHA
  REGISTRY: registry.gitlab.com/myorg/myservice

build:
  stage: build
  script:
    - docker build -t $REGISTRY:$IMAGE_TAG .
    - docker push $REGISTRY:$IMAGE_TAG
  only:
    - main

deploy-production:
  stage: deploy
  environment:
    name: production
    url: https://myservice.example.com
  script:
    - echo "$KUBE_CONFIG" | base64 -d > /tmp/kubeconfig
    - export KUBECONFIG=/tmp/kubeconfig
    - helm upgrade --install myservice ./charts/myservice
        --namespace production
        --set image.tag=$IMAGE_TAG
        --wait
  only:
    - main
  when: on_success
```

GitLab also supports OIDC federation to cloud providers via CI/CD variables with `id_tokens`:

```yaml
deploy-production-oidc:
  stage: deploy
  id_tokens:
    AWS_OIDC_TOKEN:
      aud: sts.amazonaws.com
  script:
    - >
      aws sts assume-role-with-web-identity
      --role-arn arn:aws:iam::123456789012:role/gitlab-deploy
      --role-session-name gitlab-deploy
      --web-identity-token "$AWS_OIDC_TOKEN"
      --duration-seconds 3600
      > /tmp/aws-creds.json
    - export AWS_ACCESS_KEY_ID=$(jq -r .Credentials.AccessKeyId /tmp/aws-creds.json)
    - export AWS_SECRET_ACCESS_KEY=$(jq -r .Credentials.SecretAccessKey /tmp/aws-creds.json)
    - export AWS_SESSION_TOKEN=$(jq -r .Credentials.SessionToken /tmp/aws-creds.json)
    - aws eks update-kubeconfig --name prod-cluster --region us-east-1
    - helm upgrade --install myservice ./charts/myservice --namespace production --wait
  only:
    - main
```

The security analysis is identical to GitHub Actions: OIDC federation eliminates the static secret but does not eliminate the network topology requirement or the live-job compromise window. The migration path to pull-based CD follows the same steps regardless of CI platform — bootstrap the operator from a cluster admin's terminal, add the application CRD to the fleet config, change the CI step from "run kubectl" to "commit image tag to config repo."

## 6. Push-based CD in practice: when it is the right call

Pull-based CD is architecturally superior for Kubernetes-native production workloads. That does not mean push-based CD is always wrong. There are real contexts where it is the appropriate choice.

**PaaS and serverless targets.** When you deploy to Heroku, Cloud Run, Lambda, App Engine, or Fly.io, there is no Kubernetes cluster to install an operator into. The platform provides a CLI or API for deploying. Push-based CD is the native model for these platforms, and it is correct:

```yaml
# Deploy to Cloud Run from GitHub Actions — push-based is correct here
- name: Deploy to Cloud Run
  uses: google-github-actions/deploy-cloudrun@v2
  with:
    service: order-processor
    image: gcr.io/myproject/order-processor:${{ github.sha }}
    region: us-central1
    flags: '--min-instances=2 --max-instances=20 --cpu=1 --memory=512Mi'
```

**Early-stage products and small teams.** A three-person startup does not need a GitOps operator. The cognitive overhead of maintaining Flux or Argo CD, a separate config repository, and image tag update automation outweighs the security benefit when the team is small enough that everyone who can access CI is also a cluster admin. Use a PaaS or a simple push-based deploy until you have a team large enough that the blast radius argument applies in practice.

**Ephemeral and preview environments.** Feature-branch preview deployments, per-PR test environments, and ephemeral staging clusters are created and destroyed frequently. There is no long-lived cluster to run an operator in. Push-based CD is the natural model for these environments. The security argument does not apply when the environment is throwaway.

**Non-Kubernetes infrastructure.** Ansible playbooks, Terraform-managed VMs, serverless function deployments, database migrations — none of these have an obvious pull-based model. Terraform's `plan` / `apply` workflow is inherently push-based. You can add a GitOps-style PR gate with Atlantis, but the underlying mechanism is still push.

**When the cluster is a managed PaaS itself.** Amazon ECS, Azure Container Instances, and similar services provide managed compute but not a programmable Kubernetes API where you can install operators. Push-based CD to these targets is the correct approach.

The general rule: if your target is a long-lived Kubernetes cluster running production workloads shared by a team of five or more, pull-based CD is the right default. For everything else, evaluate on its merits.

---

## 7. Push vs pull CD: the full trade-off comparison

Before going further into worked examples, the full comparison across all dimensions:

![Push vs pull CD comparison matrix across credential exposure, network path, drift detection, audit trail, and rollback speed, showing pull-based winning on four of five dimensions](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-3.png)

| Dimension | Push-based CD | Pull-based CD |
|---|---|---|
| Cluster credential location | CI secrets vault or OIDC-vended token | Kubernetes Secret in cluster, CI never sees it |
| CI breach blast radius | Direct cluster write access | Git write (requires reviewed commit to reach cluster) |
| Network requirement | CI must reach kube-apiserver | Cluster makes outbound HTTPS only |
| Drift detection | None by default | Continuous, operator auto-heals divergence |
| Rollback mechanism | Re-trigger CI with old image tag | `git revert` + operator sync (under 1 minute) |
| Lead time | CI run duration | CI run + operator poll interval (typically +1-3 min) |
| Setup complexity | Very low (15 lines of CI YAML) | Medium (operator install + config repo + auth) |
| Audit trail | CI logs only | Git commits + operator events + CI logs (three sources) |
| Multi-cluster deploy | Parallel CI jobs per cluster | Single config repo, multiple targets from one operator |
| Air-gap friendly | No (CI must reach cluster API) | Yes (cluster reaches Git/registry via proxy) |
| Progressive delivery | Custom CI stages (manual) | Native (Argo Rollouts, Flagger) |
| Self-healing | None (you notice when monitoring alerts) | Automatic (operator detects and corrects drift) |
| Emergency change path | Modify CI or run kubectl manually | `git commit` the emergency change, operator syncs |
| Regulated audit compliance | CI logs (can be deleted, rotated) | Git history (append-only, signed commits possible) |

The most important row is "CI breach blast radius." This is not a hypothetical. Real CI provider breaches have happened and will continue to happen. The question is not whether your CI provider will be breached but whether, when they are, the attacker gets a path to your production workloads.

---

## 8. The hybrid transitional pattern

The most common real-world migration path is not "install Flux and remove all kubectl from CI tomorrow." It is a gradual transition that decouples CI from CD one service at a time, and an intermediate state where CI still builds and tags but the operator handles the actual cluster mutation. This is the hybrid pattern.

![Hybrid transitional CD pattern showing CI handling build, test, and image push plus writing the new tag to a config repo, while the pull-based operator handles all actual cluster changes](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-8.png)

The hybrid pattern works as follows:

1. CI builds, tests, and pushes the image to the registry — its legitimate job.
2. CI commits the new image tag to a config repository — the only "deployment" action CI takes.
3. A pull-based operator watches the config repository and applies the updated manifest to the cluster.
4. CI holds no cluster credentials.

This is structurally identical to pure pull-based CD from a security standpoint. CI holds only a Git write credential for the config repository. The operator handles everything that touches the cluster.

Some teams run the hybrid pattern indefinitely. Others evolve further by automating the image tag commit via Flux's `ImageUpdateAutomation` controller or Argo CD's image updater plugin — at which point CI does not even need write access to the config repository. The operator watches the registry directly and updates the config repo automatically when a new tag appears.

### 8.1 Atlantis and Terraform: a pull-influenced push model

Terraform infrastructure management is inherently push-based — running `terraform apply` is a write operation to cloud APIs. But the GitOps principle (Git as the source of truth for desired infrastructure state) applies equally to IaC.

Atlantis is a server that runs inside your infrastructure and gives Terraform a PR-gated workflow:

```yaml
# atlantis.yaml  — per-repository Atlantis configuration
version: 3
projects:
  - name: production-network
    dir: infra/network
    workspace: production
    terraform_version: 1.7.0
    autoplan:
      when_modified: ["*.tf", "*.tfvars", "**/*.tf"]
      enabled: true
    apply_requirements:
      - approved       # human approval required
      - mergeable      # all CI checks must pass
  - name: production-eks
    dir: infra/eks
    workspace: production
    terraform_version: 1.7.0
    autoplan:
      when_modified: ["*.tf", "modules/**/*.tf"]
    apply_requirements:
      - approved
      - mergeable
      - undiverged     # no unmerged commits from base branch
```

When a PR touching Terraform files is opened, Atlantis runs `terraform plan` and posts the output as a comment. A human reviews the plan and approves. Atlantis applies on merge. The Terraform credentials (AWS/GCP/Azure IAM role) live inside the Atlantis server, not in CI. CI's job is to lint and validate the Terraform; Atlantis's job is to plan and apply it.

This is not pull-based CD in the pure GitOps sense — the apply is triggered by a PR merge event, not by an agent polling for drift. But it achieves the same goal of removing deployment credentials from the CI system and gating applies on human review.

---

## 9. CD architecture taxonomy

Understanding where different tools fit in the CD architecture space helps you pick the right approach.

![CD architecture taxonomy tree showing the binary choice between push-based and pull-based CD, with specific tools under each branch including kubectl in CI and OIDC federation under push, and Argo CD, Flux, and hybrid under pull](/imgs/blogs/push-vs-pull-cd-and-who-holds-the-keys-7.png)

Every CD architecture traces back to one binary decision: does CI push changes to the cluster, or does the cluster agent pull changes from Git? The tool taxonomy follows naturally from that choice.

**Choosing between Argo CD and Flux:**

| Capability | Argo CD | Flux |
|---|---|---|
| UI / dashboard | Rich web UI built-in | Optional Weave GitOps UI (separate install) |
| Multi-tenant isolation | Projects + RBAC policies | Namespace scoping + Kustomize overlays |
| Helm support | First-class (source field in Application) | HelmRelease + HelmRepository CRDs |
| Kustomize support | First-class | First-class |
| Progressive delivery | Argo Rollouts (separate install required) | Flagger (separate install required) |
| Image update automation | Argo CD Image Updater (separate plugin) | ImageUpdateAutomation controller (built-in) |
| Notification routing | Notification engine (Slack, webhook, email) | Notification controller built-in |
| CNCF maturity level | Graduated | Graduated |
| Bootstrap model | CLI + Application CRDs | `flux bootstrap` + GitRepository CRDs |
| Controller memory footprint | ~800 MB–1 GB | ~200–400 MB |
| Multi-cluster management | ApplicationSet + Hub/Spoke model | Flux remote cluster via `kubeconfig` secret |
| GitOps spec compliance | Partial | Full (GitOps Toolkit conformance) |

For a single cluster, the choice between Argo CD and Flux is often a UI preference or an organizational familiarity question. Argo CD's web dashboard is significantly more polished and more useful for teams that want a visual overview of sync status. Flux's lighter footprint and native image automation make it attractive for teams with tighter resource constraints or strong CLI/GitOps-purist preferences.

The cross-link to [Argo CD and Flux in practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) covers day-two operations for both tools in depth — things like managing secrets for the operator itself, handling sync failures, debugging reconcile errors, and tuning sync intervals.

---

## 10. Worked examples

#### Worked example: migrating the order-processor from push-based to Flux pull-based CD

**Starting point**: the `order-processor` service deploys via a GitHub Actions workflow with a `KUBE_CONFIG` Actions secret. The kubeconfig is a service account with `cluster-admin` permissions in the production namespace. Lead time: approximately 8 minutes (6 minutes CI + 2 minutes kubectl apply). Blast radius if CI is compromised: full cluster access.

**Step 1: Establish the config repository.** Create `myorg/order-processor-config` with the production Kubernetes manifests moved out of the application repo. Structure:

```
order-processor-config/
  production/
    kustomization.yaml
    deployment.yaml
    service.yaml
    hpa.yaml
```

The `kustomization.yaml` references the specific image tag CI will update:

```yaml
# production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
resources:
  - deployment.yaml
  - service.yaml
  - hpa.yaml
images:
  - name: ghcr.io/myorg/order-processor
    newTag: "abc1234"   # CI will update this value on every deploy
```

**Step 2: Install Flux with cluster bootstrap.** Run once by a cluster admin with existing kubectl access:

```bash
# Cluster admin runs this once — CI never runs this
export GITHUB_TOKEN=<admin-personal-access-token>
flux bootstrap github \
  --owner=myorg \
  --repository=fleet-config \
  --branch=main \
  --path=clusters/production \
  --personal \
  --components-extra=image-reflector-controller,image-automation-controller
```

**Step 3: Create the Flux resources for order-processor.** Commit to `fleet-config`:

```yaml
# clusters/production/order-processor.yaml
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: order-processor
  namespace: flux-system
spec:
  interval: 3m
  ref:
    branch: main
  url: https://github.com/myorg/order-processor-config
  secretRef:
    name: order-processor-readonly-auth
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: order-processor
  namespace: flux-system
spec:
  interval: 5m
  path: ./production
  prune: true
  sourceRef:
    kind: GitRepository
    name: order-processor
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: order-processor
      namespace: production
  timeout: 5m
```

**Step 4: Update CI to write the config repo instead of calling kubectl:**

```yaml
# .github/workflows/ci.yml (post-migration) — no KUBE_CONFIG required
jobs:
  build-and-tag:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Build and push image
        run: |
          docker build -t ghcr.io/myorg/order-processor:${GITHUB_SHA:0:8} .
          docker push ghcr.io/myorg/order-processor:${GITHUB_SHA:0:8}
      - name: Update config repo image tag
        env:
          CONFIG_REPO_TOKEN: ${{ secrets.CONFIG_REPO_TOKEN }}
        run: |
          git clone \
            "https://x-access-token:${CONFIG_REPO_TOKEN}@github.com/myorg/order-processor-config.git" \
            config-repo
          cd config-repo
          kustomize edit set image \
            "ghcr.io/myorg/order-processor=ghcr.io/myorg/order-processor:${GITHUB_SHA:0:8}"
          git config user.email "ci-bot@myorg.com"
          git config user.name "CI Bot"
          git add production/kustomization.yaml
          git commit -m "ci: deploy order-processor ${GITHUB_SHA:0:8}" || echo "no changes"
          git push
```

**Step 5: Delete the `KUBE_CONFIG` secret from GitHub Actions settings.** Verify Flux is syncing first:

```bash
flux get kustomizations --watch
# Expected: order-processor  True  Applied revision main@sha1:...  True
```

**Before/after numbers:**

| Metric | Before (push-based) | After (pull-based) |
|---|---|---|
| CI credentials held | kubeconfig, cluster-admin on prod | GitHub App token, contents:write on one config repo |
| Blast radius if CI breached | Full production cluster access | Git write to config repo (PR review gate active) |
| Lead time | 8 min (CI + kubectl) | 11 min (CI + 3 min Flux poll) |
| Rollback time | 5 min (trigger CI with old tag) | Under 1 min (`git revert` + Flux sync) |
| Drift detection | None | Automatic every 5 minutes |
| Audit trail | CI logs (ephemeral) | Git commits + Flux events + CI logs (permanent) |

The 3-minute lead time increase is the price of the security gain. For the vast majority of teams, that is an excellent trade.

---

#### Worked example: comparing a breach scenario in both architectures

Suppose a malicious package is introduced into your Node.js build dependencies via a dependency confusion attack. The CI job installs the malicious package during `npm ci`. The package's install script runs and attempts to exfiltrate environment variables to an attacker-controlled endpoint.

**In push-based CD with `KUBE_CONFIG` secret:**

The install script finds and exfiltrates:
- `KUBE_CONFIG`: the base64-encoded kubeconfig. The attacker decodes it, configures kubectl locally, and now has full cluster write access.
- The attacker deploys a privileged DaemonSet to every node in the cluster, establishing a persistent backdoor.
- The attacker reads all Secrets in the production namespace: database passwords, API keys, TLS private keys.
- Blast radius: total production compromise.
- Time from breach to cluster access: minutes.
- Detection: only if someone notices unexpected workloads in the cluster.

**In push-based CD with OIDC federation:**

The install script cannot steal a static kubeconfig (there is none). But if the malicious package runs during the `deploy` job (after OIDC authentication has already happened), it may find:
- The `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` environment variables that were set by the OIDC credential exchange.
- These are valid for up to one hour.
- The attacker uses them to call `aws eks update-kubeconfig` and get cluster access.
- Blast radius: cluster access for up to one hour.
- Detection: AWS CloudTrail logs the API calls.

**In pull-based CD with Flux:**

The install script finds:
- `CONFIG_REPO_TOKEN`: the GitHub App token with `contents:write` on the config repository only.
- The attacker uses this token to push a commit to the config repository — but branch protection requires PR review and at least one approval.
- Without bypassing the review, the malicious commit never reaches main.
- If the attacker somehow bypasses review (a second account, a social engineering attack on a reviewer), the malicious commit is still visible in Git history, attributable, and reversible with `git revert`.
- Blast radius: Git write access, gated by branch protection.
- Detection: the malicious commit appears in the GitHub PR UI immediately, with the CI Bot author label.

The comparison illustrates why pull-based CD with branch protection is structurally more resilient, not just theoretically more secure.

---

## 10b. Multi-cluster fleet management in pull-based CD

One of the most compelling advantages of pull-based CD becomes clear when you operate more than one cluster. In push-based CD, deploying to three clusters (dev, staging, production) means three separate CI jobs, three sets of credentials, three kubeconfigs. Adding a fourth cluster means updating the CI pipeline in every application repository. A platform team managing thirty microservices across four clusters manages 120 credential slots in CI. Rotating any credential requires updating it in 30 repositories.

In pull-based CD with Argo CD's ApplicationSet controller, adding a new cluster means:
1. Registering the cluster with the Argo CD hub cluster.
2. Adding a label to the cluster registration (`environment: production`, `region: eu-west-1`).
3. Committing nothing else. The ApplicationSet generator picks up the new cluster automatically.

The fleet manifests show this clearly:

```yaml
# One ApplicationSet manages all production clusters
# Adding a new cluster is a cluster registration — no app repo changes
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: fleet-production
  namespace: argocd
spec:
  generators:
    - matrix:
        generators:
          - clusters:
              selector:
                matchLabels:
                  environment: production
          - list:
              elements:
                - service: order-processor
                  path: services/order-processor
                - service: payment-gateway
                  path: services/payment-gateway
                - service: inventory-api
                  path: services/inventory-api
  template:
    metadata:
      name: "{{service}}-{{name}}"
      labels:
        service: "{{service}}"
        cluster: "{{name}}"
    spec:
      project: production
      source:
        repoURL: https://github.com/myorg/fleet-config
        targetRevision: main
        path: "{{path}}/overlays/{{metadata.labels.region}}"
      destination:
        server: "{{server}}"
        namespace: "{{service}}"
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

This single ApplicationSet drives deployment of three services to every cluster labeled `environment: production`. A new cluster with that label gets all three services deployed automatically. A new service requires adding one entry to the `list` generator — not updating any CI pipeline or any CI credential.

The credential model is equally clean: each spoke cluster has its own Argo CD agent with its own RBAC service account. The hub cluster's Argo CD installation holds kubeconfigs for the spoke clusters — but these are cluster-to-cluster credentials (not CI-to-cluster), managed by cluster admins, and rotated via cluster operations, not CI pipeline changes. CI has zero credentials in this architecture.

### 10c. Image update automation: eliminating the CI-to-config-repo write step

In the basic hybrid pattern, CI still needs write access to the config repository to update the image tag. This is a much smaller blast radius than a kubeconfig, but some teams want to eliminate it entirely. Flux's `ImageUpdateAutomation` controller and Argo CD Image Updater provide this capability.

With Flux image automation, CI's only job is to push the image to the registry. Flux watches the registry directly, detects new tags, and commits the image tag update to the config repository itself:

```yaml
# The image reflector tells Flux which tags to watch
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageRepository
metadata:
  name: order-processor
  namespace: flux-system
spec:
  image: ghcr.io/myorg/order-processor
  interval: 1m
---
# The image policy selects which tag to use (latest by semver)
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImagePolicy
metadata:
  name: order-processor
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: order-processor
  filterTags:
    pattern: '^[0-9a-f]{8}$'   # match 8-char git sha tags
    extract: '$0'
  policy:
    alphabetical:
      order: asc
---
# The automation controller commits the selected tag back to Git
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImageUpdateAutomation
metadata:
  name: flux-system
  namespace: flux-system
spec:
  interval: 5m
  sourceRef:
    kind: GitRepository
    name: fleet-config
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        email: flux@myorg.com
        name: Flux Image Updater
      messageTemplate: "chore: update {{range .Updated.Images}}{{.}}{{end}}"
    push:
      branch: main
  update:
    path: ./services
    strategy: Setters
```

In the `kustomization.yaml`, a comment marks where Flux should update the image tag:

```yaml
# production/kustomization.yaml
images:
  - name: ghcr.io/myorg/order-processor
    newTag: abc12345  # {"$imagepolicy": "flux-system:order-processor:tag"}
```

With this setup, CI's credential surface is reduced to registry push access only. No Git write access anywhere in CI. The full pull-based trust chain: developer pushes code → CI builds and pushes image → Flux detects new image tag in registry → Flux commits tag update to config repo → Flux syncs cluster. CI touches neither the config repo nor the cluster.

### 10d. Progressive delivery integration: the operator as the gatekeeper

Pull-based CD operators integrate naturally with progressive delivery tools because both operate at the cluster level with a reconciliation model. Argo Rollouts (a Kubernetes controller that replaces Deployments with a richer `Rollout` CRD) can be driven entirely by Argo CD sync, with no CI involvement in the canary logic:

```yaml
# A Rollout replaces a Deployment in the config repo
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: order-processor
  namespace: production
spec:
  replicas: 10
  revisionHistoryLimit: 5
  selector:
    matchLabels:
      app: order-processor
  template:
    metadata:
      labels:
        app: order-processor
    spec:
      containers:
        - name: api
          image: ghcr.io/myorg/order-processor:abc12345
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
  strategy:
    canary:
      steps:
        - setWeight: 10          # 10% traffic to new version
        - pause: {duration: 5m}  # wait 5 minutes
        - analysis:
            templates:
              - templateName: order-processor-success-rate
        - setWeight: 50          # 50% if analysis passes
        - pause: {duration: 5m}
        - setWeight: 100         # full rollout if still healthy
      analysis:
        templates:
          - templateName: order-processor-success-rate
        startingStep: 2
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: order-processor-success-rate
  namespace: production
spec:
  metrics:
    - name: success-rate
      interval: 2m
      successCondition: result[0] >= 0.99
      failureLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{service="order-processor",status=~"2.."}[5m]))
            /
            sum(rate(http_requests_total{service="order-processor"}[5m]))
```

When CI commits a new image tag to the config repo, Argo CD syncs the Rollout spec. Argo Rollouts takes over — it shifts 10% of traffic to the new version, waits 5 minutes, queries Prometheus for the success rate, and promotes or aborts based on the result. CI has zero role in any of this after the image push. The entire progressive delivery logic lives in the cluster, driven by the operator.

This is the architecture the SRE cross-link to [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) describes from the reliability theory side: progressive delivery as an SLO-gated safety practice. Pull-based CD is the toolchain that makes it practical.

---

## 11. War story / real case

### The Codecov breach and why KUBE_CONFIG in CI is real risk

In April 2021, an attacker gained unauthorized access to Codecov's CI build infrastructure and modified the Codecov bash uploader script distributed to customers. The modification added code to exfiltrate environment variables from any CI environment running the script. The modified script ran as part of thousands of customers' CI jobs for approximately two months before discovery.

The exfiltration was indiscriminate: it captured any environment variable that existed in the CI job's environment. In teams using push-based CD, that environment routinely contained `KUBE_CONFIG`, `KUBE_TOKEN`, AWS credentials, service account keys, and similar deployment credentials.

Security researchers and affected teams confirmed that attackers had access to customer CI environment variables. Twilio, HashiCorp, Codecov's own customers in financial services and technology — all had to assume their CI secrets were compromised and rotate all credentials.

Teams running pull-based CD with no cluster credentials in CI were in a materially better position. Their CI environment might have contained a `CODECOV_TOKEN` or a `GITHUB_TOKEN` with repository write access. A compromised `GITHUB_TOKEN` is a serious incident — but it does not give an attacker a path to the production cluster without bypassing PR review and the pull-based operator's trust chain.

The Codecov incident is not unique. The build environment is the attack surface of modern software supply chain attacks. Any credential that exists in the CI environment during any step of any job is potentially exposed to supply chain attacks against:
- CI providers (platform-level breaches).
- GitHub Actions marketplace actions (action supply chain attacks).
- Build tool dependencies (dependency confusion, typosquatting, compromised packages).
- Base images used in CI job containers.

Pull-based CD does not eliminate these attack vectors. It eliminates the consequence of a successful attack against CI: instead of cluster access, the attacker gets Git write access with a human-review gate.

### The drift incident that nobody saw coming

A payments team ran push-based CD for a high-volume transaction processing service. On a Tuesday afternoon, the on-call engineer responded to an OOM alert in production. The Deployment was running with a memory limit of 256 MB that had been correct when the service was small but was too tight for the current traffic volume. The engineer ran:

```bash
kubectl patch deployment payments-api -n production \
  --patch '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"512Mi"}}}]}}}}'
```

The service stabilized. The alert cleared. The incident was closed. The kubectl command was in the engineer's shell history, never committed to Git. The Git manifests still showed `memory: 256Mi`.

Three weeks later, a CI pipeline ran the full deploy job for an unrelated feature — changing one logging configuration parameter. The deploy job ran `kubectl apply -f k8s/deployment.yaml`. The apply used the version of `deployment.yaml` in Git, which still specified `memory: 256Mi`. The apply succeeded. The OOM kills returned within 45 minutes during the next traffic spike. It took two hours to diagnose because nobody remembered the three-week-old kubectl patch.

In a pull-based CD system with `selfHeal: true`, the kubectl patch would have been detected within Argo CD's default 3-minute check interval and reverted before the team left the office. The system would have auto-healed back to the Git state, which would have triggered an immediate alert ("hey, your manual fix was reverted — commit it to Git if you want it to persist"). The discipline is enforced by the system, not by human memory of events from weeks ago.

### The Friday deploy key rotation that took three days

A platform team at a mid-sized technology company had been running push-based CD for four years. The production kubeconfig in CI was a service account token created when the cluster was first set up. The token had no expiry (Kubernetes service account tokens prior to v1.24 were eternal by default). A security audit recommended rotating all long-lived credentials as part of an annual review.

The rotation of the deploy kubeconfig involved:
1. Creating a new service account token.
2. Generating a new kubeconfig and base64-encoding it.
3. Updating the Actions secret in each of seventeen application repositories.
4. Triggering a deploy of each service to confirm the new credential worked.
5. Watching for any service that had been deployed with the old credential in the interim.
6. Revoking the old service account token only after all seventeen services confirmed.

The rotation took three full working days. During that time, the old token was still valid. The "rotation window" — the period between "we decided to rotate" and "all services are confirmed using the new token" — was the most dangerous phase of the exercise, because both the old and new credentials were valid simultaneously.

In a pull-based CD system, the cluster credential is a Kubernetes Secret in the `flux-system` namespace. Rotating it means:

```bash
# Generate a new deploy key
flux create secret git order-processor-readonly-auth \
  --url=https://github.com/myorg/order-processor-config \
  --namespace=flux-system

# Update the public key in GitHub (add new, remove old)
# Flux reconciles automatically — no per-service action needed
```

One operation. All seventeen services rotate simultaneously because they all use the same operator. The old credential can be revoked immediately once the new one is in place. No CI pipelines to retrigger, no seventeen-repository update tour, no three-day rotation window.

---

## 12. How to reach for this (and when not to)

Pull-based GitOps CD is the right default for long-lived Kubernetes production workloads with a team of five or more. It is not the right choice for every situation.

**Reach for pull-based CD when:**
- You are running Kubernetes and have a team large enough that not everyone who can merge code should necessarily have direct cluster write access.
- You operate multiple clusters that should track the same desired state with per-environment overlays.
- You are in a regulated industry where change management requires a full audit trail linking production changes to reviewed and approved commits.
- Your cluster is private and you do not want to open firewall rules for CI.
- You have been burned by drift — where what CI last deployed differs from what is actually running.
- Your rollback story is "re-trigger CI with the old tag" — which means it takes as long as a full CI run and runs all tests that you do not want to wait for during an incident.
- You want to gate progressive delivery (canary, blue-green, weighted rollout) on automated SLO metrics without custom CI scripting.

**Do not add pull-based CD when:**
- Your deployment target is a PaaS, serverless platform, or managed container service without a programmable Kubernetes API.
- Your team has two or three engineers. The setup and maintenance overhead of a GitOps operator plus a config repo split is real. Use a PaaS.
- You are building ephemeral preview environments. Push-based CD is correct for short-lived clusters.
- Your infrastructure is primarily VMs, bare metal, or non-Kubernetes targets. Ansible, Terraform, or platform-native deploy tools are more appropriate.
- You have not yet separated configuration from application code. Do that separation first; pull-based CD without a clear config repository boundary creates confusion about what Git is the source of truth for.

**The migration sequence for an existing push-based team:**

1. **Add OIDC federation first.** Eliminate static kubeconfigs from the CI vault immediately. This alone reduces the blast radius from "indefinite cluster access" to "temporary token, bounded window." This is a one-day change per cluster.

2. **Create a config repository.** Move production manifests out of the application repo. Establish the separation between "code that changes the application behavior" (application repo) and "config that controls what runs in production" (config repo). This is a structural change that takes a few days but pays dividends for years.

3. **Install the operator.** Bootstrap Flux or install Argo CD into the cluster. Point it at the config repository. At this point, you have both push-based CI deploy and pull-based operator running simultaneously. They should be idempotent (both applying the same manifests), so the overlap is safe.

4. **Update CI to write the tag commit instead of running kubectl.** Change the CI deploy step from `kubectl apply` to `kustomize edit set image` + git commit. Stop triggering kubectl from CI. The operator now does the actual apply.

5. **Remove the OIDC deploy role from CI.** Once the operator is confirmed working for a full sprint, remove the IAM role and the `id-token: write` permission from CI. The blast radius is now bounded by Git write access only.

6. **Enable branch protection on the config repository.** Require PR review for commits to main. Require signed commits if your threat model warrants it. This closes the remaining gap: without branch protection, a compromised CI token can still push directly to main without review.

**Common mistakes during migration:**
- Giving the Flux or Argo CD service account `cluster-admin`. Scope it to the namespaces it manages.
- Forgetting to enable `prune: true`. Without pruning, resources deleted from Git continue running in the cluster — the opposite of what GitOps promises.
- Not planning for the `selfHeal` behavior. Document the process for making emergency runtime changes that should persist. The answer is always: commit it to Git. The operator will not fight you if the Git state matches what you applied manually — it only reverts changes that are not in Git.
- Storing the operator's deploy key in Terraform state without encrypting the state backend. The state file becomes the new attack vector.
- Setting the sync interval too low (under 30 seconds) in a large cluster. This causes unnecessary API server load. 3-5 minutes is the correct default; use webhook triggers if you need faster syncs for interactive development.

---

## 12b. Stress-testing the pull-based model

Every architecture has failure modes. Pull-based CD's failure modes are different from push-based CD's, and you should know them before adopting it.

**What if Git is down?** If GitHub has an outage, Flux cannot poll the config repository. Existing cluster state continues running — no deployments are disrupted. But you cannot deploy new versions until Git recovers. This is generally acceptable: Git uptime (GitHub SLA: 99.9%) is better than the reliability of the CI-to-cluster network path in push-based CD. For catastrophic availability requirements, configure a Git mirror in a second provider or use a self-hosted Gitea instance as a local mirror.

**What if the registry is down during a rollout?** If the container registry is unavailable when the operator tries to apply a new image, the pods fail to pull the image and enter `ImagePullBackOff`. Argo CD or Flux marks the sync as degraded. The existing running pods (the old version) are not terminated — Kubernetes respects the rolling update strategy and only kills old pods once new ones are healthy. The rollout stalls but does not cause a full outage. Once the registry recovers, the next sync attempt succeeds.

**What if the operator itself crashes?** Kubernetes restarts the operator pods automatically. The reconcile loop resumes. Any changes that would have been applied during the downtime are applied when the operator comes back up. The cluster may run slightly out-of-sync with Git during the window, but existing workloads continue running. This is a key advantage of pull-based CD: the operator is a reconciler, not a gatekeeper. The cluster does not stop serving traffic because the operator is momentarily unavailable.

**What if two PRs merge to the config repo simultaneously?** Both commits are on the main branch, ordered by Git. The operator sees them in sequence. The second commit's state is what the operator ultimately converges to. No conflicts — Git's ordering ensures a single canonical desired state at any point in time. Push-based CD with two parallel CI runs applying different manifests to the cluster simultaneously can race; pull-based CD cannot, because there is only one desired state at a time (the HEAD of the config repo).

**What if someone pushes a bad manifest to the config repo?** The operator detects the bad manifest during apply (invalid YAML, a missing required field, a reference to a nonexistent resource). It marks the sync as failed and sends an alert via the notification controller. The previous version continues running. No partial deployment has occurred. The fix is to commit a corrected manifest to the config repo — which the operator picks up in the next poll interval. Compare this to push-based CD where a bad `kubectl apply` might partially apply some resources before failing, leaving the cluster in an inconsistent state.

**What if selfHeal reverts an emergency fix before you can commit it?** This is the most common operational surprise teams encounter when first adopting pull-based CD. The correct procedure is:
1. Commit the emergency fix to the config repo first (this takes 30 seconds).
2. The operator sees the commit and applies it.
3. If you need the fix live before the next poll interval, trigger an immediate sync: `flux reconcile kustomization order-processor --with-source`.

The golden rule: Git first, cluster second. Any change you make directly to the cluster that is not in Git will be reverted by the operator. This is a feature, not a bug — it is what makes the operator the reliable source of truth.

## 13. Key takeaways

1. **Push-based CD gives CI the keys to production.** Any system with a kubeconfig or IAM role that can write to a cluster is one compromised CI job away from a production breach. This is not a hypothetical — CI provider breaches and supply chain attacks against CI dependencies are a documented, recurring threat.

2. **Pull-based CD inverts the trust model.** The cluster agent pulls from Git; CI writes only to Git. A compromised CI system loses direct cluster access entirely. The attacker gets Git write access, which is still a serious incident but is gated by PR review and is visible in the Git history.

3. **OIDC federation is a meaningful improvement to push-based CD** — eliminating static credentials from the CI vault is valuable. It does not eliminate the network topology problem (CI must still reach the cluster API), and it does not protect against a live CI job being compromised during the deployment window.

4. **CI ends when the image and Git tag land in their stores.** CD is a separate concern. Conflating them forces the CI system to hold credentials it should not have and makes rollback dependent on re-running CI.

5. **Deploy frequency and lead time are measured differently in pull-based CD.** Instrument the operator sync events, not the CI completion events, for accurate DORA metric calculation.

6. **Pull-based CD provides drift detection for free.** The reconcile loop continuously compares desired state in Git with actual state in the cluster. Drift is detected and automatically healed. Push-based CD silently accumulates drift until the next CI-driven deploy overwrites it — which may cause incidents weeks after the original manual change.

7. **The network argument alone justifies pull-based CD in private clusters.** Production clusters should not expose their API servers to external CI networks. Pull-based CD makes this the default posture without requiring VPN workarounds.

8. **Rollback in pull-based CD is a `git revert`** — a 30-second operation, independent of CI, completable by any engineer with repository write access during an incident.

9. **The hybrid pattern is the correct migration strategy.** CI builds and tags; CI commits the new tag to a config repository; the operator syncs. This achieves the security properties of pull-based CD on day one of adopting the operator, without a big-bang migration.

10. **Branch protection on the config repository is the second line of defense.** Pull-based CD reduces the blast radius from cluster access to Git write; branch protection ensures exploiting that Git write credential still requires bypassing review.

---

## 14. Further reading

- **Accelerate: The Science of Lean Software and DevOps** — Forsgren, Humble, Kim. The original empirical basis for the DORA metrics and their correlation with organizational delivery performance. The chapter on continuous delivery practices directly supports the arguments in this post.
- **DORA State of DevOps Reports** (https://dora.dev) — annual benchmark data on deploy frequency, lead time, change-failure rate, and MTTR across thousands of engineering teams. The 2023 report specifically covers GitOps adoption and its correlation with elite DORA performance.
- **Flux documentation — GitOps Toolkit** (https://fluxcd.io/flux/concepts/) — the official concepts guide explaining source controllers, the reconcile loop, health assessments, and image update automation.
- **Argo CD documentation** (https://argo-cd.readthedocs.io) — the Application CRD reference, sync policies, health assessment framework, and ApplicationSet controller for fleet management.
- **SLSA supply chain security framework** (https://slsa.dev) — the provenance and build integrity model that pull-based CD enforces at the artifact level. SLSA levels 2 and 3 require that the build system cannot be influenced by the deployer — a property pull-based CD helps satisfy.
- **GitHub Actions OIDC documentation** (https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/about-security-hardening-with-openid-connect) — the official guide to OIDC federation from GitHub Actions to AWS, GCP, and Azure.
- **Within this series:** [From commit to production: the CI/CD mental model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series spine that frames the commit→build→test→package→deploy→operate loop where this architectural choice sits. [GitOps: Git as the source of truth](/blog/software-development/ci-cd/gitops-git-as-the-source-of-truth) — the principles behind treating Git as the system of record for desired cluster state and what GitOps conformance means in practice. [Argo CD and Flux in practice](/blog/software-development/ci-cd/argo-cd-and-flux-in-practice) — day-two operational details for running either operator in production: managing secrets, debugging sync failures, tuning performance, and operating a multi-cluster fleet.
- **SRE cross-link:** [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) — how the progressive delivery capabilities that pull-based CD enables (canary rollouts, traffic shifting, automated rollback) integrate with SLO-gated rollout gates and error budget consumption as reliability practices.

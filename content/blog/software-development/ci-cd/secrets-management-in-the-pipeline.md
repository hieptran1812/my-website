---
title: "Secrets management in the pipeline: From CI vars to OIDC keyless auth"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how to eliminate long-lived credentials from your CI/CD pipeline using GitHub Actions OIDC, HashiCorp Vault dynamic secrets, External Secrets Operator, and zero-downtime rotation so a leaked token expires in minutes, not months."
tags:
  [
    "ci-cd",
    "devops",
    "secrets-management",
    "hashicorp-vault",
    "oidc",
    "kubernetes",
    "security",
    "github-actions",
    "aws",
    "pipeline-security",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/secrets-management-in-the-pipeline-1.png"
---

Three years ago, an engineer on a platform team I know left an AWS access key in a `.env` file. Not intentionally — they were debugging a flaky integration test locally, committed a hotfix under pressure, and the key hitched a ride. The CI job turned green. The pull request was approved. Nobody noticed the `.env` file in the diff because everyone was focused on the hotfix itself. Six months later, a cryptomining operator noticed. They had been silently provisioning EC2 instances in `us-east-1` for two months before the AWS bill triggered a billing alert — the account had a monthly spend that had quintupled. The incident response playbook said "rotate the key." The problem: rotating the key broke three services, two deploy pipelines, and one database migration job that had been using the same credential for reasons nobody could explain, because whoever originally configured it had left the company.

Total blast radius: \$47k in fraudulent compute, eighteen hours of engineering time across three teams, two production incidents caused by the rotation itself, and one very uncomfortable retrospective where the CISO asked the engineering director to explain how a six-month-old secret exposure was discovered by a billing spike rather than a security scan.

The most frustrating part was not the money. It was this: the key had been in the git history for six months. Deleting the commit did nothing. The key was in the GitHub archive, in every developer's local clone, in every CI provider's repository cache, in the audit logs of every service that had authenticated with it. The moment a secret touches version control, rotation is not optional — it is the only path to safety, and every second of delay extends the exposure window.

This post is about preventing that outcome, and about building the architecture that makes "secret leaked" a minor recoverable incident rather than a six-month undetected breach. By the end, you will know how to wire GitHub Actions OIDC to AWS with no access key stored anywhere in CI, configure HashiCorp Vault to hand your application a database credential that expires in one hour and is automatically revoked, deploy the External Secrets Operator to sync cloud secrets into Kubernetes pods without manual copying, implement zero-downtime rotation that does not page your on-call engineer on a Friday, and use Sealed Secrets to commit encrypted blobs to a GitOps repository safely. Figure 1 shows the architecture of the gold-standard approach — OIDC keyless federation — which you will understand fully by the end of section 5.

Every pattern here maps directly to the commit→build→test→package→deploy→operate spine from [the CI/CD mental model post](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model). Secrets appear at every stage of that spine: at build time (pulling private packages), at package time (pushing to a private registry), at deploy time (authenticating to the cluster and cloud), and at runtime (database URLs, API keys, signing keys). Getting this right is not a security checkbox exercise — it is a delivery reliability investment. The four DORA metrics measure delivery performance; change-failure rate tracks directly how often secret-related incidents cause production failures.

![OIDC keyless federation flow showing CI job obtaining a GitHub-issued JWT that AWS STS evaluates against an IAM condition to issue short-lived temporary credentials or deny access](/imgs/blogs/secrets-management-in-the-pipeline-1.png)

## Why secrets in CI/CD have a different threat model

Secrets management in production applications is well-understood: use a dedicated secrets store, rotate regularly, audit every access event, follow least-privilege IAM. What makes the CI/CD pipeline a harder problem is the convergence of three characteristics that rarely coexist elsewhere in the stack.

**High frequency execution.** In a mature delivery organisation, hundreds or thousands of pipeline runs happen every day. A monorepo with 20 services triggering pipelines on every pull request generates thousands of secret-access events per day. A long-lived credential that is valid for six months and accessed five hundred times per day produces an enormous attack surface by sheer volume of exposure events. Any one of those events — a malicious dependency, a compromised runner, a log statement that accidentally prints an environment variable — can exfiltrate the credential.

**Ephemeral, shared execution environments.** GitHub-hosted runners are fresh VMs discarded after each job, but they share network boundaries and infrastructure with GitHub's systems. Self-hosted runners may be shared across teams or repositories. GitLab CI executors may handle jobs from dozens of projects on the same machine. Container-based CI environments may have side-channel vulnerabilities. A credential that exists as an environment variable in a CI job is accessible to every process that runs in that job — including every build dependency, every test framework, every shell script, every third-party action you pull in.

**The dev-to-prod bridge.** Your CI pipeline's deploy step holds credentials that reach your production database, your cloud account, your Kubernetes cluster. The same pipeline that a junior engineer can trigger by opening a pull request ultimately exercises prod-level permissions if you are not careful. An engineer can accidentally `echo $DATABASE_URL` in a debugging step. A test can print environment variables. A malicious or compromised third-party GitHub Action can exfiltrate everything in `env`. This is the threat model you are designing against: a path from "low-trust developer action" to "high-trust credential access" that must be tightly controlled.

The DORA change-failure rate metric is the delivery-performance signal to watch here. Teams that automate credential rotation, use dynamic secrets, and implement OIDC federation consistently report change-failure rates below 5%. Teams that manage credentials manually and reactively — rotating when something breaks or when someone notices — sit above 15%. The correlation is not coincidental: a manual rotation process is an error-prone process, and every failed rotation that breaks a service registers as a failed change.

There is also a compounding effect across the other DORA metrics. A team that spends two days on an emergency credential rotation (investigating the breach, identifying affected services, coordinating the rotation, debugging the services it breaks) has just added two days to its mean time to restore (MTTR) for that incident. A team that has automated rotation with zero-downtime procedures restores in minutes. Lead time for changes is also affected: if developers fear that touching a secret configuration will trigger an hour of debugging, they avoid those changes — accumulating technical debt and reducing deploy frequency. Secrets hygiene is not peripheral to DORA; it sits on the critical path of all four metrics.

## Rule zero: secrets never touch Git

This rule is so fundamental it sounds obvious. The reason to state it rigorously is that the most common cause of credential exposure in organisations of all sizes is a developer committing a `.env` file, a configuration file with a hardcoded password, an authentication token in a test fixture, or a comment with "temporary credentials — delete before merging" that gets forgotten in code review. Understanding *why* this is unrecoverable — not just that it is bad — is what makes the rule stick.

The human failure mode is almost always the same: time pressure combined with unfamiliarity with the tooling. A developer debugging a CI integration test on a Friday afternoon copies their local `.env` into the repository temporarily "to make the test work." Or they hardcode an API key in a test fixture because they cannot figure out how to inject it from the CI environment. Or they add `echo $API_KEY` to debug a pipeline failure and forget to remove it. These are not malicious acts — they are the predictable result of making the secure path harder than the insecure path. The goal of the architecture in this post is to make the secure path the easy path: OIDC federation is easier than managing static keys once it is set up; Vault Agent is easier than remembering to rotate passwords; ESO is easier than manually copying secrets into Kubernetes manifests.

A useful mental model: think of secrets as radioactive material. It is fine to work with them, but you want to minimise the time they spend outside a shielded container. The shielded container is the secrets manager. The moment a secret leaves the secrets manager and enters a file, an environment variable, or a log line, the clock is ticking on its safe handling. OIDC and dynamic secrets minimise that window to seconds. Static credentials in CI expand that window to months or years.

This also reframes how you should evaluate secrets-related engineering work. The question is not "is this credential secure enough?" — it is "how long does this credential exist outside a secrets manager, and what can an attacker do with it during that window?" A database password that is valid for one hour and auto-revoked is not "as secure" as a permanent password with a good rotation schedule — it is qualitatively different, because the attack window is bounded by physics (one hour) rather than by process compliance (someone remembering to rotate it). The architecture in this post is about moving from process-dependent security to architecture-enforced security.

Git's append-only object model means that every version of every file you have ever committed is permanently stored in the object database. When you delete a file and create a new commit, you are adding a new tree object that excludes the file. You are not modifying or removing the previous commit objects. Anyone who cloned the repository before your deletion still has the object in their local `.git/` directory. Anyone who clones after the deletion can check out the previous commit by its SHA and see the file in full. GitHub's infrastructure archives every push. CI providers cache repository state. Package registries may mirror source. Public repositories are routinely scraped and archived by security researchers, bots, and malicious actors within minutes of a push.

The mathematical certainty is this: if a secret was in a public repository for any length of time, you must treat it as fully compromised regardless of whether you delete it. For private repositories, the risk depends on who has access — but "private repository in an organisation with dozens of engineers, CI access, and contractor access" is a much broader exposure surface than most people assume.

**The only correct response to a secret committed to Git**: assume it is compromised, rotate it immediately (before attempting history cleanup), and then optionally attempt history rewriting with `git filter-branch` or the BFG Repo Cleaner. History rewriting invalidates all open pull requests, forces every contributor to re-clone or reset their local branches, and does not retroactively remove the secret from GitHub's event archive or anyone's already-pushed fork. It is damage control, not remediation. Rotation is the only remediation.

### Pre-commit hooks: the first gate

The right place to catch secrets is before they reach the remote. `gitleaks` runs as a pre-commit hook to scan staged content before each commit:

```bash
# Install gitleaks
brew install gitleaks

# Or via pre-commit framework:
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks
        name: Detect secrets with gitleaks
        stages: [commit]
```

Install the hooks across your team with `pre-commit install`. `gitleaks` ships with detection rules for over 150 credential patterns: AWS access keys, GitHub personal access tokens, Stripe secret keys, Slack webhooks, private RSA keys, JWT secrets, Google API keys, Twilio auth tokens, and more. The patterns match both the credential value format and common surrounding context like `api_key =` or `Authorization: Bearer`.

When `gitleaks` finds a match, it prints the filename, line number, rule that fired, and the first few characters of the matched value (truncated to reduce exposure), then blocks the commit. The developer sees the problem at the moment they are trying to commit — not after the push has propagated across mirrors and CI caches.

The pre-commit hook is necessary but not sufficient. Hooks can be bypassed with `git commit --no-verify`. New developers may not have hooks installed. A CI gate provides the authoritative, non-bypassable check.

### CI scanning gates

```yaml
# .github/workflows/secret-scan.yml
name: Secret scan

on:
  pull_request:
  push:
    branches: [main]

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # full history — catches secrets added then deleted in the same PR
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
```

The `fetch-depth: 0` is critical. A shallow clone with `fetch-depth: 1` would only scan the latest commit. A developer who accidentally commits an API key in commit A, then deletes it in commit B, would pass the scan on commit B. With full history, gitleaks scans every commit in the pull request diff and catches commit A.

For repositories that have accumulated history you need to retroactively audit, TruffleHog performs a deep scan with live credential verification:

```bash
# Install TruffleHog
brew install trufflesecurity/trufflehog/trufflehog

# Scan full git history, only report verified (still-live) credentials
trufflehog git file://. --only-verified --json 2>/dev/null | \
  jq '{file: .SourceMetadata.Data.Git.file, commit: .SourceMetadata.Data.Git.commit, detector: .DetectorName}'
```

The `--only-verified` flag reduces false positives dramatically by actually testing the credential against its provider's authentication endpoint. Only credentials that successfully authenticate are reported. This is the tool to run when you inherit a codebase and need to know your actual exposure right now.

![Git history is permanent: committing a secret then deleting it still exposes the credential in every clone and every historical checkout, making immediate rotation the only remediation](/imgs/blogs/secrets-management-in-the-pipeline-6.png)

GitGuardian operates at the organisation level: it monitors every push to every repository in real time, maintains a centralised database of detected secrets organised by repository and developer, and sends remediation instructions including direct links to the provider's credential rotation interface. At the team level, gitleaks-in-CI provides adequate coverage. At the enterprise level, GitGuardian-style organisation-wide monitoring covers the long tail of inherited repositories, newly onboarded teams, and the discovery phase of historical exposure.

A practical baseline for any team: gitleaks pre-commit hook on developer machines, gitleaks GitHub Actions workflow in CI scanning full history on every PR, and a weekly TruffleHog scan of the full repository history with verified-only output piped to a Slack channel.

## CI platform secrets: the floor, not the ceiling

GitHub Actions, GitLab CI, CircleCI, and every major CI platform offer encrypted secret storage. These are good starting points and dangerous ending points — teams that treat CI platform secrets as a complete secrets management solution are one manual rotation failure or one `pull_request_target` misconfiguration away from a bad day.

In GitHub Actions, secrets are stored and injected at three scopes: repository, environment, and organisation. The `${{ secrets.X }}` syntax injects the value as an environment variable into the job execution environment. GitHub masks the secret value in logs: if any log line contains the exact string of the secret value, GitHub replaces it with `***`. A deployment job using environment secrets:

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Deploy application
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          STRIPE_SECRET_KEY: ${{ secrets.STRIPE_SECRET_KEY }}
        run: ./scripts/deploy.sh

      - name: Run database migrations
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: ./scripts/migrate.sh
```

The `environment: production` line is critical and often overlooked. An environment in GitHub Actions is not just a label — it is a protection gate. You can configure an environment to require a manual approval from designated reviewers before the job starts, restrict which branches are allowed to deploy to the environment, limit the secrets available to only those scoped to that environment, and add deployment delay rules. This is the mechanism that prevents a feature branch from deploying to production with production credentials.

In GitLab CI, secrets are called "variables" and scope to project, group, or environment. The `$DATABASE_URL` syntax accesses them. GitLab supports marking variables as "masked" (hidden in job logs) and "protected" (available only on protected branches and tags). A production deploy job in GitLab:

```yaml
# .gitlab-ci.yml
deploy:production:
  stage: deploy
  environment:
    name: production
    url: https://app.example.com
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  script:
    - ./scripts/deploy.sh
  variables:
    DATABASE_URL: $DATABASE_URL_PROD   # protected variable, available only on main
```

The `rules:` block restricts the job to the `main` branch. The `$DATABASE_URL_PROD` variable is marked protected in the GitLab project settings, so it is unavailable on feature branches — even if someone deliberately triggers a feature-branch pipeline manually.

### The fork PR attack surface

Here is a vulnerability that catches teams repeatedly: if your repository is public and your workflow triggers on pull request events from forks, you must understand exactly which secrets are accessible in that context.

For standard `pull_request` events, GitHub Actions does **not** expose `secrets.*` to jobs triggered by pull requests from forks. This is intentional and correct: fork PRs run in the context of the fork, not the base repository, so they do not receive the base repository's secrets. A malicious contributor cannot open a PR and print `${{ secrets.AWS_ACCESS_KEY_ID }}`.

The trap is `pull_request_target`. This event type runs in the context of the base repository, with access to base-repository secrets. It was designed for workflows that need to post comments or status checks on fork PRs. If you use `pull_request_target` to run code from the PR (like `actions/checkout@v4` with the PR SHA), that code executes with base-repository secrets in scope. An attacker who opens a PR can inject commands into your workflow and exfiltrate your production credentials.

The rule: never use `pull_request_target` to check out and run code from the PR branch unless you have an explicit, audited security review of the workflow. If you need to post statuses on fork PRs, use a two-workflow pattern: `pull_request` to run tests and upload artifacts, then `workflow_run` to post results using base-repo secrets against the already-completed test run.

### Inherent limitations of CI platform secrets

No matter how carefully you configure environment protection rules and branch restrictions, CI platform secrets have structural limitations that become problems at scale:

**No rotation mechanism.** Rotating a CI platform secret requires a human to go into the settings UI or call the API, update the value, and coordinate across any pipelines that reference it. There is no scheduling, no automation, no verification that the rotation succeeded. Most teams rotate CI secrets reactively — when something breaks or when an engineer remembers — which means they often go years without rotation.

**Limited audit trail.** GitHub shows you when a repository secret was last updated. It does not show you which pipelines accessed it, from which branches, at what times, with what results. You cannot answer "has this secret been used in the last 30 days?" without external logging. AWS CloudTrail provides this for AWS API calls made with a static key, but not for the key's presence in the CI environment.

**Static, long-lived credentials.** A `DATABASE_URL` set in 2021 may still be active and unchanged in 2026. Every day it exists unchanged is another day of potential exposure. CI platform secrets have no built-in TTL, no automatic rotation, and no expiry.

**No Kubernetes-native path.** Getting a CI secret into a running Kubernetes pod requires either mounting it as an environment variable in a Deployment spec (which means it lives in a Kubernetes Secret that had to come from somewhere), or writing a pipeline step that calls `kubectl create secret` (which means the pipeline holds a kubeconfig with cluster-admin access). Neither is elegant. The External Secrets Operator pattern (covered in section 4) solves this properly.

The table below summarises where CI platform secrets sit relative to the other approaches this post covers:

| Approach | Blast radius if leaked | Rotation | Audit trail | Dynamic creds | K8s integration | Ops cost |
|---|---|---|---|---|---|---|
| CI platform secrets | Indefinite, full credential scope | Manual, ad hoc | Last-updated only | No | Manual copy | Minimal |
| AWS Secrets Manager | Indefinite, IAM-scoped | Automated Lambda | Full CloudTrail | No (static values) | External Secrets Op. | Low |
| HashiCorp Vault static | Indefinite, policy-scoped | Automated or manual | Full audit log | No | ESO or Agent | High |
| HashiCorp Vault dynamic | 1h TTL, auto-revoked | Automatic (leases) | Full lease log | Yes | Agent sidecar | High |
| OIDC keyless | 1h TTL, IAM-role-scoped | Not needed (keyless) | AWS CloudTrail | Yes (STS temp creds) | IRSA / Workload ID | Minimal |
| Sealed Secrets | Indefinite, but encrypted-in-Git | Manual | None (Git-native) | No | Native K8s Secret | Low |

## HashiCorp Vault: dynamic secrets that cannot leak long-term

HashiCorp Vault is the most complete secrets management platform available. Its architecture is built around a core insight: the most dangerous class of secret is one that is static and long-lived. Vault's answer is the **secrets engine** — a plugin that generates credentials on demand, attaches them to a lease, and automatically revokes them when the lease expires. The long-lived credential problem disappears by construction.

The database secrets engine is the most commonly used. When your application needs a PostgreSQL connection, it calls Vault. Vault creates a new PostgreSQL user with a unique, random password, returns the credentials to the caller, and starts a countdown. When the lease timer hits zero — typically after one to eight hours depending on your configuration — Vault executes a revocation statement that drops the PostgreSQL user. The application can request a renewal before the lease expires to extend it, but the cumulative maximum TTL prevents indefinite renewal.

![Vault dynamic database credential lifecycle showing the request, user creation with 1-hour lease, active use window, optional renewal, automatic expiry, and DB user deletion](/imgs/blogs/secrets-management-in-the-pipeline-5.png)

The operational consequence: if a dynamic credential is somehow leaked — intercepted in a network trace, accidentally logged, exfiltrated by a malicious dependency — the attacker has at most one hour to use it before Vault automatically deletes the database user. Compare that to the \$47k incident described in the introduction, where a static key had been available for six months.

### Vault authentication methods

Vault is secured by its auth methods — the mechanisms by which clients prove their identity before being granted a Vault token. The two most relevant to CI/CD are AppRole (for pipelines) and Kubernetes (for pods).

**AppRole** is designed for machine authentication. A pipeline is assigned a `role_id` (not a secret — analogous to a username) and can request short-lived, single-use `secret_id` tokens from Vault. The pipeline uses both to authenticate and receive a Vault token scoped to a policy.

The correct pattern for CI pipelines: the `role_id` is stored as a CI platform secret (it is not a secret in the security sense — it does not grant access alone). The `secret_id` is generated fresh at the start of each pipeline run, with a 10-minute TTL and a single-use constraint:

```bash
#!/bin/bash
# CI pipeline step: authenticate to Vault and fetch dynamic DB credentials
set -euo pipefail

VAULT_ADDR="https://vault.internal.example.com"

# Generate a fresh secret_id (valid for 10 minutes, one use only)
# This call uses a pre-existing Vault token stored as a CI secret (a separate, limited bootstrap credential)
SECRET_ID=$(vault write -f -field=secret_id \
  auth/approle/role/ci-pipeline/secret-id \
  -wrap-ttl=10m)

# Exchange role_id + secret_id for a Vault token
VAULT_TOKEN=$(vault write -field=token auth/approle/login \
  role_id="${VAULT_APPROLE_ROLE_ID}" \
  secret_id="${SECRET_ID}")

export VAULT_TOKEN

# Read a dynamic database credential (Vault creates a new DB user with 1h TTL)
DB_CREDS=$(vault read -format=json database/creds/app-readonly)
export DB_USER=$(echo "${DB_CREDS}" | jq -r .data.username)
export DB_PASSWORD=$(echo "${DB_CREDS}" | jq -r .data.password)

echo "Dynamic DB credential issued: ${DB_USER} (expires in 1h)"
echo "Running integration tests..."
./scripts/run-integration-tests.sh
```

The `secret_id` is valid for 10 minutes and can be used once. Even if it is exfiltrated from the CI log before it is consumed, the attacker has a 10-minute window and one use. After the pipeline consumes it, it is gone.

**Kubernetes auth method** is cleaner for workloads running inside a cluster. Pods present their Kubernetes service account JWT to Vault, which validates it against the cluster's API server to verify the pod's identity. No pre-shared secrets are needed. The Vault policy is bound to a specific service account in a specific namespace:

```hcl
# vault-policy-app-backend.hcl
# Allows reading dynamic DB credentials and static config
path "database/creds/app-readonly" {
  capabilities = ["read"]
}

path "kv/data/app/config" {
  capabilities = ["read"]
}

path "sys/leases/renew" {
  capabilities = ["update"]
}
```

```bash
# Configure the Kubernetes auth role in Vault
vault write auth/kubernetes/role/app-backend \
  bound_service_account_names=app-backend \
  bound_service_account_namespaces=production \
  policies=app-backend \
  ttl=1h
```

### Vault Agent injector: transparent secret injection

The Vault Agent injector is a Kubernetes mutating admission webhook. When a pod is annotated with `vault.hashicorp.com/agent-inject: "true"`, the webhook injects a Vault Agent init container and a sidecar container into the pod's spec. The init container authenticates to Vault, fetches the requested secrets, writes them to a shared `emptyDir` volume as files, and completes. The main application container starts after the init container succeeds — so secrets are guaranteed to be available before the application process starts. The sidecar container runs alongside the application and handles lease renewal, re-fetching secrets before they expire and rewriting the files.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-backend
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: app-backend
  template:
    metadata:
      labels:
        app: app-backend
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "app-backend"
        vault.hashicorp.com/agent-inject-secret-db: "database/creds/app-readonly"
        vault.hashicorp.com/agent-inject-template-db: |
          {{- with secret "database/creds/app-readonly" -}}
          postgres://{{ .Data.data.username }}:{{ .Data.data.password }}@postgres.production.svc.cluster.local:5432/appdb?sslmode=require
          {{- end }}
        vault.hashicorp.com/agent-inject-secret-config: "kv/data/app/config"
        vault.hashicorp.com/agent-inject-template-config: |
          {{- with secret "kv/data/app/config" -}}
          STRIPE_KEY={{ .Data.data.stripe_key }}
          SENDGRID_KEY={{ .Data.data.sendgrid_key }}
          {{- end }}
    spec:
      serviceAccountName: app-backend
      containers:
        - name: app
          image: registry.example.com/app-backend:2.14.1
          env:
            - name: VAULT_SECRETS_PATH
              value: /vault/secrets
          command: ["/bin/sh", "-c"]
          args:
            - |
              # Load secrets as environment variables from the Vault-written file
              export DATABASE_URL=$(cat /vault/secrets/db)
              source <(cat /vault/secrets/config)
              exec ./app-backend
          volumeMounts:
            - name: vault-secrets
              mountPath: /vault/secrets
              readOnly: true
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
```

The application reads its database URL from `/vault/secrets/db` at startup. The Vault Agent sidecar renews the lease before it expires and rewrites the file. Applications that read the file fresh at each connection attempt (rather than caching the value in a connection string at startup) automatically pick up rotated credentials without a restart. The sidecar's renewal attempts are visible in its container logs, giving you observability into the credential lifecycle.

## Cloud-native secrets: AWS Secrets Manager, GCP Secret Manager, Azure Key Vault

If you are not operating Vault — and for most teams under 50 engineers, the operational overhead of Vault is not justified — cloud-native secrets managers are the right default. They provide centralised storage, automatic rotation for supported service integrations, complete access audit logs via the cloud provider's audit trail, and fine-grained IAM-based access control. The tradeoff is that they store static credentials (no dynamic issuance), the rotation granularity is coarser, and you are tied to a specific cloud provider's API.

### AWS Secrets Manager

AWS Secrets Manager stores secrets as versioned JSON blobs. It supports Lambda-based rotation functions for RDS databases, Redshift, DocumentDB, and any custom service via a custom Lambda. Every `GetSecretValue` API call is logged in AWS CloudTrail with the caller's identity, timestamp, and the secret ARN — giving you the audit trail that CI platform secrets lack.

Fetching a secret in a pipeline step (assuming an OIDC-federated IAM role with `secretsmanager:GetSecretValue` permission scoped to the specific secret ARN):

```bash
#!/bin/bash
# Fetch database credentials from AWS Secrets Manager
# No AWS_ACCESS_KEY_ID needed — credentials come from OIDC-federated role (see section 5)
set -euo pipefail

SECRET_JSON=$(aws secretsmanager get-secret-value \
  --secret-id "prod/app/database" \
  --region us-east-1 \
  --query SecretString \
  --output text)

export DB_HOST=$(echo "${SECRET_JSON}" | jq -r .host)
export DB_USER=$(echo "${SECRET_JSON}" | jq -r .username)
export DB_PASS=$(echo "${SECRET_JSON}" | jq -r .password)
export DB_NAME=$(echo "${SECRET_JSON}" | jq -r .dbname)

echo "Database credentials fetched for host: ${DB_HOST}"
```

The IAM policy for the CI role granting access to this secret follows least-privilege — the `Resource` is the specific secret ARN, not a wildcard:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:prod/app/database-*"
    }
  ]
}
```

The trailing `-*` wildcard after the secret name is required because AWS appends a 6-character suffix to the ARN when a secret is created. This pattern is documented in the AWS security best practices for Secrets Manager.

### External Secrets Operator

The External Secrets Operator (ESO) solves the bridge problem: how do cloud-stored secrets get into Kubernetes pods without manual copying? ESO runs as a Kubernetes controller, watches `ExternalSecret` CRD objects, reads the referenced secret from any supported backend (AWS Secrets Manager, AWS Parameter Store, GCP Secret Manager, Azure Key Vault, HashiCorp Vault, IBM Secrets Manager, Doppler, and more), and creates or updates a standard Kubernetes `Secret` object with the fetched values. Your pods read the Kubernetes Secret through normal volume mounts or `envFrom` — they have no idea ESO exists.

The two core CRDs are `SecretStore` (defines the backend connection: which cloud, which region, how to authenticate) and `ExternalSecret` (defines which secrets to fetch from the backend and which Kubernetes Secret to write them into).

```yaml
# SecretStore: AWS Secrets Manager backend using IRSA (IAM Roles for Service Accounts)
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: production
```

```yaml
# ExternalSecret: pull prod/app/database and write into Kubernetes Secret "app-database"
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-database-secret
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: app-database
    creationPolicy: Owner
    template:
      engineVersion: v2
      data:
        DATABASE_URL: "postgres://{{ .username }}:{{ .password }}@{{ .host }}:5432/{{ .dbname }}"
        DB_PASSWORD: "{{ .password }}"
  dataFrom:
    - extract:
        key: prod/app/database
```

```yaml
# Deployment referencing the ESO-managed secret
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-backend
  namespace: production
  annotations:
    reloader.stakater.com/auto: "true"   # triggers rolling restart when Secret updates
spec:
  template:
    spec:
      containers:
        - name: app
          image: registry.example.com/app-backend:2.14.1
          envFrom:
            - secretRef:
                name: app-database   # managed by ESO — updated every 1h from Secrets Manager
```

The `refreshInterval: 1h` means ESO re-reads from AWS Secrets Manager every hour. When the secret is rotated (manually or by a Lambda rotation function), ESO picks up the new value on the next poll interval and updates the Kubernetes Secret object. The `reloader.stakater.com/auto: "true"` annotation on the Deployment instructs the Reloader controller to trigger a rolling restart when the Secret changes, ensuring pods always run with the current credential.

For time-critical rotation scenarios where a 1-hour poll delay is unacceptable, you can set `refreshInterval: 5m` or call the ESO API to trigger an immediate refresh after a rotation event.

![Secrets storage taxonomy showing static options including CI platform vars, cloud secrets manager, and Sealed Secrets, versus dynamic or keyless options including Vault dynamic secrets and OIDC keyless federation](/imgs/blogs/secrets-management-in-the-pipeline-7.png)

### GCP Secret Manager

ESO's `ClusterSecretStore` for GCP Secret Manager using Workload Identity Federation:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: gcp-secret-manager
spec:
  provider:
    gcpsm:
      projectID: my-gcp-project-id
      auth:
        workloadIdentity:
          clusterLocation: us-central1
          clusterName: prod-cluster
          clusterProjectID: my-gcp-project-id
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets
```

The GCP service account (`external-secrets-sa@my-gcp-project-id.iam.gserviceaccount.com`) is bound to the Kubernetes service account via Workload Identity, eliminating the need for a static GCP service account key file. The same OIDC-no-static-keys principle that applies to GitHub-to-AWS federation applies to GKE-to-GCP Secret Manager.

### Azure Key Vault

Azure uses Managed Identity or a Service Principal for ESO authentication. With a User-Assigned Managed Identity on the node pool:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: azure-kv-store
  namespace: production
spec:
  provider:
    azurekv:
      tenantId: "your-tenant-id"
      vaultUrl: "https://myproductvault.vault.azure.net"
      authType: ManagedIdentity
      identityId: "/subscriptions/.../resourceGroups/.../providers/Microsoft.ManagedIdentity/userAssignedIdentities/eso-identity"
```

The pattern is consistent across all three cloud providers: the `SecretStore` specifies provider-specific auth; the `ExternalSecret` specifies which secret keys to fetch; the resulting Kubernetes `Secret` is consumed normally. ESO is the primary reason you should not build a custom `initContainer` that calls `aws secretsmanager get-secret-value` — that pattern leaks cloud credentials into pod specs and does not handle rotation or refresh.

## OIDC keyless federation: the gold standard for CI-to-cloud authentication

The most impactful architectural decision in secrets management for CI/CD pipelines is recognising that many credentials in CI are for the purpose of authenticating to a cloud provider or deploying to a cloud platform. And for that purpose, you do not need a stored credential at all.

GitHub Actions, GitLab CI, CircleCI, and most major CI platforms are OpenID Connect (OIDC) identity providers. They issue signed JWT tokens for each running job, containing structured claims about the job's identity: which repository triggered it (`repository` claim), which branch or tag (`ref` claim), which workflow file (`workflow` claim), which environment (`environment` claim), which SHA is being built (`sha` claim). These claims are cryptographically signed by the platform's OIDC provider key, and the signatures are publicly verifiable.

AWS, GCP, and Azure can all be configured as OIDC relying parties: they accept these signed JWTs as authentication tokens, validate the signature against the CI platform's well-known JWKS endpoint, verify the claims against an IAM condition, and issue temporary cloud credentials (STS temporary credentials for AWS, short-lived Google identity tokens for GCP, Azure AD tokens for Azure). The exchange happens in real time, takes under a second, and produces credentials with a 1-hour TTL.

![Comparison of static long-lived AWS access key in CI where a leaked key grants full account access forever versus OIDC keyless federation where a 1-hour token has minimal scoped permissions](/imgs/blogs/secrets-management-in-the-pipeline-2.png)

The security consequence: your CI pipeline holds no stored cloud credential. There is nothing in GitHub's encrypted secret store, nothing in a `.env` file, nothing in a hardcoded workflow `env:` block. The only thing the pipeline does is request a JWT from GitHub's OIDC provider (which is authorised automatically by the `id-token: write` permission) and exchange it for temporary credentials via the cloud provider's STS endpoint. A malicious actor who intercepts the OIDC JWT has a race window of seconds before it is consumed. A malicious actor who intercepts the resulting temporary credentials has at most one hour against a scoped IAM role — not full account access, not indefinite validity.

### Full walkthrough: GitHub Actions OIDC to AWS

The AWS side requires three things: an IAM OIDC Identity Provider that trusts GitHub's token endpoint, an IAM role with a trust policy that validates the JWT claims, and an IAM policy attached to that role granting the specific permissions the pipeline needs.

```bash
# Step 1: Create the OIDC Identity Provider in AWS (one-time setup per AWS account)
# The thumbprint is for token.actions.githubusercontent.com - verify from AWS documentation
aws iam create-open-id-connect-provider \
  --url "https://token.actions.githubusercontent.com" \
  --client-id-list "sts.amazonaws.com" \
  --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1"
```

The IAM role trust policy is where the security boundary lives. The `Condition` block restricts which GitHub repositories, environments, and branches can assume the role:

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
          "token.actions.githubusercontent.com:sub": "repo:myorg/myrepo:environment:production"
        }
      }
    }
  ]
}
```

The `sub` claim format is: `repo:{owner}/{repo}:environment:{environment}` for environment-scoped workflows, or `repo:{owner}/{repo}:ref:refs/heads/{branch}` for branch-scoped access. A fork, a different repository, a different environment, or a different branch all produce a different `sub` claim and are denied by the `StringEquals` condition.

For staging, create a separate IAM role with its own `sub` condition scoped to the staging environment, with a policy that grants access only to staging resources. The production role's trust policy ensures that even if a staging pipeline is compromised, it cannot assume the production role.

```yaml
# GitHub Actions workflow: deploy to AWS with OIDC (no stored keys)
name: Deploy to production

on:
  push:
    branches: [main]

permissions:
  id-token: write    # Required: allows the workflow to request an OIDC JWT
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials via OIDC federation
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsProductionRole
          aws-region: us-east-1
          role-session-name: GitHubActions-${{ github.run_id }}-${{ github.run_attempt }}

      - name: Verify assumed role (optional sanity check)
        run: aws sts get-caller-identity

      - name: Push container image to ECR
        env:
          ECR_REGISTRY: 123456789012.dkr.ecr.us-east-1.amazonaws.com
          IMAGE_TAG: ${{ github.sha }}
        run: |
          aws ecr get-login-password --region us-east-1 | \
            docker login --username AWS --password-stdin ${ECR_REGISTRY}
          docker build -t ${ECR_REGISTRY}/app-backend:${IMAGE_TAG} .
          docker push ${ECR_REGISTRY}/app-backend:${IMAGE_TAG}

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster production \
            --service app-backend \
            --force-new-deployment

      - name: Invalidate CloudFront cache
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ vars.CLOUDFRONT_DIST_ID }} \
            --paths "/*"
```

There is no `AWS_ACCESS_KEY_ID` and no `AWS_SECRET_ACCESS_KEY` in this workflow. The `configure-aws-credentials` action requests an OIDC JWT from GitHub's token endpoint at `https://token.actions.githubusercontent.com`, sends it to AWS STS via `AssumeRoleWithWebIdentity`, and receives temporary credentials (access key, secret key, session token) with a 1-hour TTL. Those temporary credentials are exported into the job environment automatically. When the job finishes, the credentials are gone. When the 1-hour TTL expires, they are cryptographically invalid.

![Secrets management maturity layers from CI platform secrets at the base through cloud secrets manager and Vault dynamic secrets to OIDC keyless at the apex with each layer reducing blast radius and increasing security guarantees](/imgs/blogs/secrets-management-in-the-pipeline-3.png)

#### Worked example: migrating a fintech team from static keys to OIDC

A fintech startup I worked with had seven GitHub Actions workflows, each storing `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as repository secrets. The keys had the `AdministratorAccess` IAM policy (justified historically as "it was easier at the time"). They were rotated annually at best. The keys were shared across all seven workflows and all environments — there was no isolation between staging and production access.

The migration process:

1. Create three IAM roles (dev, staging, production) with `trust policy` conditions scoped to specific GitHub environment names.
2. Attach least-privilege IAM policies to each role: dev gets read-only ECR and ECS Describe; staging gets ECR push and ECS deploy to staging; production gets ECR push, ECS deploy to production, and CloudFront invalidation.
3. Create a GitHub OIDC Identity Provider in the AWS account (one-time, 30 seconds).
4. Update each workflow: remove the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from `secrets.` references, add `permissions: id-token: write`, switch to `configure-aws-credentials` with `role-to-assume`.
5. Delete the 14 repository secrets (7 key IDs + 7 secret keys) from GitHub.

Total migration time: 4 engineer-hours for the initial setup, 1 hour for testing, 30 minutes for cleanup.

Before: 14 stored secrets, one rotation per year (optimistically), `AdministratorAccess` for all workflows, no isolation between environments, blast radius of a compromised key = full AWS account access indefinitely.

After: 0 stored AWS credentials in GitHub, no rotation needed (TTL is 1 hour), least-privilege per environment, blast radius of a compromised pipeline = 1-hour window against one ECS service and ECR repository. In the 18 months after migration: zero credential-related security incidents, zero time spent on credential rotation coordination.

### GitLab CI OIDC to GCP

GitLab's equivalent uses `id_tokens` with an `aud` claim targeting the GCP Workload Identity Federation pool:

```yaml
# .gitlab-ci.yml
deploy:
  stage: deploy
  image: google/cloud-sdk:460.0.0-alpine
  id_tokens:
    GCP_OIDC_TOKEN:
      aud: https://iam.googleapis.com/projects/123456789/locations/global/workloadIdentityPools/gitlab-pool/providers/gitlab-provider
  script:
    - echo "${GCP_OIDC_TOKEN}" > /tmp/oidc_token.json
    - |
      gcloud iam workload-identity-pools create-cred-config \
        projects/123456789/locations/global/workloadIdentityPools/gitlab-pool/providers/gitlab-provider \
        --service-account=deploy-sa@my-project.iam.gserviceaccount.com \
        --output-file=/tmp/gcp_creds.json \
        --credential-source-file=/tmp/oidc_token.json
    - gcloud auth login --cred-file=/tmp/gcp_creds.json --quiet
    - gcloud run deploy app-backend --image gcr.io/my-project/app:${CI_COMMIT_SHA} --region us-central1
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

The GCP Workload Identity Federation pool is configured to trust GitLab's JWT issuer and to extract the `sub` claim (which includes the project path and branch) for use in IAM Conditions. The service account binding restricts which GitLab project and branch can impersonate the deploy service account.

## Sealed Secrets: GitOps-safe secret management

GitOps means "everything in Git, Git is the source of truth." This principle breaks down at the secrets layer: Kubernetes Secrets are base64-encoded, not encrypted. A base64-encoded value is reversible in milliseconds. Committing a Kubernetes Secret manifest to Git is functionally identical to committing the plaintext password to Git.

Sealed Secrets solves this problem by encrypting the Secret manifest with the cluster's public key. The resulting `SealedSecret` CRD object contains a ciphertext blob that only the in-cluster Sealed Secrets controller can decrypt — because the controller holds the private key. The `SealedSecret` is safe to commit to the GitOps repository. When the Argo CD or Flux reconciler applies the manifest to the cluster, the Sealed Secrets controller decrypts it and creates the corresponding `Secret` object.

```bash
# One-time setup: install the Sealed Secrets controller in the cluster
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm upgrade --install sealed-secrets sealed-secrets/sealed-secrets \
  --namespace kube-system \
  --version 2.15.3 \
  --set fullnameOverride=sealed-secrets-controller

# Fetch the cluster's public key (safe to store in the GitOps repo)
kubeseal --fetch-cert \
  --controller-name sealed-secrets-controller \
  --controller-namespace kube-system \
  --kubeconfig /path/to/kubeconfig \
  > gitops-repo/cluster-public-key.pem
```

To seal a secret, create the Kubernetes Secret manifest (never commit this), then pipe it through `kubeseal`:

```bash
# Create the plain Secret manifest (NEVER commit this file)
kubectl create secret generic app-database \
  --namespace production \
  --from-literal=DATABASE_URL="postgres://appuser:$(openssl rand -base64 32)@postgres.production.svc:5432/appdb" \
  --from-literal=DB_PASSWORD="$(openssl rand -base64 32)" \
  --dry-run=client \
  --output yaml \
  > /tmp/app-database-plain.yaml   # stays local, never pushed

# Seal using the cluster public key
kubeseal \
  --cert gitops-repo/cluster-public-key.pem \
  --format yaml \
  --namespace production \
  < /tmp/app-database-plain.yaml \
  > gitops-repo/manifests/sealed-secrets/app-database-sealed.yaml

# Now commit the sealed file (the ciphertext blob)
git add gitops-repo/manifests/sealed-secrets/app-database-sealed.yaml
git commit -m "Add sealed database secret for production"

# Clean up the plaintext file
rm /tmp/app-database-plain.yaml
```

The resulting `app-database-sealed.yaml` contains only the encrypted blob. An attacker with read access to the Git repository can see the encrypted blob but cannot decrypt it without the controller's private key.

### Disaster: losing the controller private key

The critical operational risk with Sealed Secrets is key loss. If the controller is deleted without backing up the private key — or if the `kube-system` namespace is wiped, or the etcd backup is lost — every `SealedSecret` in the cluster becomes permanently undecryptable. You cannot recover the plaintext from the Git-committed ciphertext without the private key. You must re-seal every secret from scratch using the original plaintext values, which you must have stored somewhere else.

Back up the controller key immediately after installation, and repeat after any controller key rotation:

```bash
# Back up the controller key to AWS Secrets Manager
# The controller generates a TLS keypair as a Kubernetes Secret on first install
CONTROLLER_KEY=$(kubectl get secret \
  --namespace kube-system \
  --selector sealedsecrets.bitnami.com/sealed-secrets-key=active \
  --output json)

aws secretsmanager create-secret \
  --name "k8s/sealed-secrets-controller-key/prod-cluster" \
  --secret-string "${CONTROLLER_KEY}" \
  --description "Sealed Secrets controller TLS keypair for prod-cluster"

echo "Controller key backed up. Verify: aws secretsmanager describe-secret --secret-id k8s/sealed-secrets-controller-key/prod-cluster"
```

Run this after installation and add a monthly verification step to your ops runbook confirming the backup is current and restorable.

Sealed Secrets is excellent for GitOps workflows where you want declarative secret management without an external secrets manager dependency. It is the right choice for air-gapped clusters, self-hosted infrastructure, and teams that prioritise Git as the single source of truth. If you are already heavily invested in AWS or GCP, ESO with OIDC federation to the cloud secrets manager is generally more powerful — it provides central rotation, a full audit trail, and the secrets manager as the authoritative store rather than a Git blob.

## Secret rotation: zero-downtime as a first-class engineering concern

Most teams treat rotation as an operational inconvenience that happens reactively (after a breach) or ceremonially (once a year). The teams that have lived through a "rotation broke prod" incident recognise rotation for what it is: a deployment event that must be engineered with the same care as a code deployment.

![Zero-downtime rotation with dual-read showing both old and new credentials valid during a 15-minute window versus hard cutover causing 30-second downtime and a P1 incident](/imgs/blogs/secrets-management-in-the-pipeline-8.png)

The fundamental principle: **never revoke an old credential before the new one is confirmed working in production**. This is the dual-read window. During the window, both the old and new credentials are valid. The application is rolled to use the new credential. Health checks confirm the new credential works. Only then is the old credential revoked.

### The rotation runbook (zero-downtime version)

For a database password rotation:

1. **Generate and store the new credential.** Create a new database user or update the password in AWS Secrets Manager without revoking the old one. Both `app_user_v1` and `app_user_v2` (or the old and new passwords) are now valid simultaneously. The database accepts connections from both.

2. **Propagate the new credential.** Update the Kubernetes Secret (either directly, or via an ESO refresh triggered by the updated Secrets Manager value). The Reloader controller detects the Secret change and begins a rolling restart of the Deployment. Old pods still use the old credential; new pods use the new one. During the rolling restart, both credentials are needed simultaneously — this is why the dual-read window must be active.

3. **Verify health.** Wait for the rolling restart to complete: `kubectl rollout status deployment/app-backend -n production`. Check application health metrics — error rate, response latency, database connection pool errors. Set a verification window of at least 5 minutes after the rollout completes. Use your monitoring system to confirm the new credential is working.

4. **Revoke the old credential.** Only after health is confirmed do you revoke the old credential (drop the old database user or mark the old Secrets Manager version inactive). No in-flight requests can now fail, because all pods have completed their rollout and are using the new credential.

### Kubernetes Secret hot-reload

Kubernetes mounts Secrets as files in pods via `volumeMount`. When a Secret object is updated, the kubelet on each node detects the change via its watch against the API server and rewrites the mounted files — typically within 60 seconds, without a pod restart. Applications that read their database URL from a file at each connection attempt (rather than once at startup) pick up the new credential transparently.

Most production applications do not work this way — they read configuration at startup and cache it in a connection pool. For these applications, a rolling restart is required. The Reloader controller automates this: it watches Secrets and ConfigMaps, and when either changes, it triggers a rolling restart of any Deployment with the `reloader.stakater.com/auto: "true"` annotation.

```bash
# Install Reloader
helm repo add stakater https://stakater.github.io/stakater-charts
helm upgrade --install reloader stakater/reloader \
  --namespace kube-system \
  --set reloader.watchGlobally=false  # only watch annotated deployments
```

With Reloader, the end-to-end rotation latency is: Secrets Manager updated → ESO poll interval (up to 1h) → Secret updated → Reloader detects change (within seconds) → rolling restart begins → rollout completes (10–60 seconds depending on pod count and readiness). For most production applications with hourly rotation, this is acceptable. For applications with stricter SLOs on credential freshness, use Vault Agent with active push and a short renewal interval.

#### Worked example: the rotation that broke prod — and how to fix it

A team I consulted ran a quarterly secret rotation on their PostgreSQL password as part of a security review. Their runbook was: (1) update the Secrets Manager value, (2) restart the pods. They executed at 14:00 on a Thursday (they had learned from the Friday problem, at least).

14:00 — Updated the Secrets Manager value with the new password.
14:03 — Triggered `kubectl rollout restart deployment/app-backend`. Rolling restart begins.
14:04 — 50% of pods now use the new password. Database accepts both old and new simultaneously. Healthy.
14:07 — An engineer noticed ESO's local cache still showed the old Secret value. They manually ran `kubectl annotate externalsecret app-database-secret force-sync=$(date +%s) -n production`, which triggered an immediate ESO refresh. ESO re-read Secrets Manager (which now showed the new password) and updated the Kubernetes Secret. Reloader detected the Secret change and triggered a second rolling restart.
14:08 — The first rolling restart (triggered at 14:03) was midway through. The second rolling restart (triggered at 14:07) started. Pods were being replaced twice — some had neither the old nor the new credential correctly loaded in the 10-second window between restarts.
14:08:30 — Error rate spiked to 40%. On-call paged. On-call engineer ran `kubectl rollout undo deployment/app-backend` — rolling back to the previous ReplicaSet, which had the old credential. But the old credential had just been revoked by the rotation Lambda at 14:08 (which ran on a timer, not waiting for health verification).

Total impact: 4 minutes of elevated errors, 22 minutes of incident response, one database connection pool exhaustion event, and a rollback that failed because the rolled-back pods also had the wrong credential.

Root causes: (1) no dual-read window — old password was revoked before rollout was verified; (2) manual intervention created a race condition between two concurrent rolling restarts; (3) the rotation runbook had no explicit health-gate step with a timeout.

The fix: replaced the manual runbook with an AWS Lambda rotation function that executes the four-phase protocol with explicit waits:
- Phase 1 (0 minutes): generate new password, create new DB user, update Secrets Manager secret version.
- Phase 2 (0 minutes): trigger ESO refresh via annotation. Wait for Kubernetes Secret to update (poll for up to 5 minutes).
- Phase 3 (5 minutes): wait for rolling restart to complete. Poll `kubectl rollout status`. Set 10-minute timeout.
- Health gate (15 minutes): check error rate in CloudWatch. If error rate > 0.5% for 3 consecutive minutes, emit an SNS alert and halt. Do not proceed to phase 4.
- Phase 4 (only if health gate passes): revoke old DB user and mark old Secrets Manager version inactive.

After implementing the automated rotation Lambda, they ran 18 monthly rotations without a single production incident.

## Comparison: choosing the right secrets approach

| Approach | When to use | When NOT to use | Ops burden |
|---|---|---|---|
| CI platform secrets | Non-prod config, low-sensitivity, quick start | Production credentials, database passwords | Minimal |
| Cloud secrets manager | Production app secrets, any cloud-deployed service | Air-gapped infra, multi-cloud without ESO | Low |
| Vault dynamic | Database credentials, any secret with a revocation need | Small teams, first secrets investment | High |
| OIDC keyless | Any CI-to-cloud authentication | On-prem without an OIDC-capable CI platform | Minimal |
| Sealed Secrets | GitOps, air-gapped, self-hosted clusters | Already using ESO with cloud SM | Low |

The maturity progression most teams should follow:

1. Immediately replace any stored cloud credential in CI with OIDC federation. This is free, takes half a day, and eliminates the single most common CI security incident class.
2. Move production application secrets from CI platform variables into a cloud secrets manager (AWS SM, GCP SM, Azure KV). Deploy ESO to sync them into Kubernetes. Set up rotation schedules.
3. For database credentials specifically, evaluate Vault dynamic secrets. The operational cost is high but the security gain is qualitatively different — a 1-hour TTL credential that auto-revokes is categorically safer than a rotated static password.
4. For GitOps workflows, add Sealed Secrets for any secrets that must be version-controlled alongside manifests.

![Secrets management matrix comparing CI platform secrets, cloud secrets manager, Vault dynamic, and OIDC keyless across rotation, audit trail, dynamic credentials, and Kubernetes native integration dimensions](/imgs/blogs/secrets-management-in-the-pipeline-4.png)

## War story: the Codecov supply-chain attack and what it means for secrets in CI

In April 2021, Codecov disclosed that an attacker had modified their official bash uploader script (`codecov.io/bash`) by inserting code that exfiltrated environment variables to an external server. The attack was active for approximately 2 months before discovery. Every CI pipeline that ran `curl -s https://codecov.io/bash | bash` during that period had potentially leaked every environment variable available in that pipeline run — including `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, Heroku API keys, Twilio auth tokens, and any other secrets stored as CI environment variables.

The blast radius was enormous, spanning thousands of organisations, because of two compounding failures: (1) pipelines were piping an untrusted, third-party script directly into `bash`, giving it full access to the execution environment including all environment variables; (2) CI pipelines had been granted broad cloud permissions (in many cases, AdministratorAccess) so a coverage reporting step had the same access as a deployment step.

The mitigations the Codecov incident illustrates:

**Pin third-party CI actions and scripts to specific commit SHAs, not floating version tags.** The tag `@v3` can be moved by the vendor to point to a different commit. The SHA `@ab928c847f33bb18a1ea1bfd27b77d18773b71f0` is immutable. Renovate and Dependabot support automated SHA pinning with periodic updates. The [supply chain security post](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier) covers this in depth, including SLSA provenance and cosign-based signature verification.

**Limit CI job permissions using the minimal set.** A coverage reporting step has no legitimate need for `AWS_ACCESS_KEY_ID`. Split your pipeline into jobs with separate permission scopes: a `test` job with no cloud credentials, a `scan` job with read-only artifact registry access, a `deploy` job with the OIDC-federated deploy role. Even if the test job is compromised, it cannot reach production.

**Use OIDC federation** so that even if a malicious script exfiltrates the OIDC JWT before it is consumed, the attacker has a race window of seconds. The JWT is exchanged once and then the temporary credentials are what the pipeline uses. The Codecov attack would still have exfiltrated the temporary credentials — but the 1-hour TTL and scoped IAM role limits the attacker's window and blast radius from "indefinite full account access" to "1-hour window against one ECS service."

A second war story: the 2020 SolarWinds compromise. The attacker gained access to the SolarWinds build pipeline and injected a malicious payload into the Orion software build — a credential compromise at the build-system level that affected the resulting artifact. The immediate lesson for secrets management: even a fully OIDC-federated pipeline is only as trustworthy as the execution environment. If an attacker can modify your build steps, they can intercept credentials at the moment they are used. This is why pipeline security — pinned dependencies, signed artifacts, SBOM attestation — is inseparable from secrets management. See [securing the pipeline itself](/blog/software-development/ci-cd/securing-the-pipeline-itself) for the complementary controls.

A third, more recent pattern: the "misconfigured Kubernetes RBAC exposes secrets" class of incident. A cluster with overly permissive RBAC (for example, a service account bound to `cluster-admin` or a role with `get` on `secrets/*` cluster-wide) means that any compromised pod can read every Kubernetes Secret in the cluster. The combination of ESO-synced secrets (which make Kubernetes Secrets the live copy of your AWS Secrets Manager values) and permissive RBAC is particularly dangerous: a compromised pod can call the Kubernetes API, list all secrets in all namespaces, and exfiltrate every application credential in the cluster. The mitigation: RBAC policies for `Secret` resources should follow least-privilege — a pod should only be able to read the secrets it is explicitly bound to. Audit your cluster's RBAC with `kubectl auth can-i get secrets --as system:serviceaccount:production:app-backend -n production` for each service account. The `rakkess` tool provides a full RBAC access matrix across all resource types. Secrets management tooling (Vault Agent, ESO) is only as secure as the cluster-level access controls around the resulting Kubernetes Secrets.

## Secrets across the full pipeline lifecycle

Secrets do not just appear in the deploy step. Understanding where credentials are needed at each stage of the commit→build→test→package→deploy→operate spine helps you apply the right tool at the right point.

**Commit stage:** gitleaks and TruffleHog scanning ensures no secrets enter the repository. No credentials are needed in the commit itself — but the CI trigger uses a `GITHUB_TOKEN` (automatically provided by GitHub, scoped to the specific repository and workflow run, automatically expired after the run). You do not manage this token; GitHub manages it for you. It follows the OIDC principle: ephemeral, scoped, automatically rotated.

**Build stage:** private package registries require authentication. A Python package from a private PyPI mirror, an npm package from a private registry, a Go module from a private GOPROXY — all need credentials. The right pattern is to store these as CI platform secrets (they are infrequently rotated and not prod-sensitive) or, for cloud-hosted registries like AWS CodeArtifact, to use OIDC-federated credentials to generate a temporary registry token at build time:

```bash
# Generate a temporary CodeArtifact token using OIDC-federated AWS credentials
CODEARTIFACT_TOKEN=$(aws codeartifact get-authorization-token \
  --domain myorg \
  --domain-owner 123456789012 \
  --query authorizationToken \
  --output text)

# Configure pip to use the token (valid for 12 hours)
pip config set global.index-url \
  "https://aws:${CODEARTIFACT_TOKEN}@myorg-123456789012.d.codeartifact.us-east-1.amazonaws.com/pypi/internal/simple/"
```

The token is valid for 12 hours and scoped to the specific CodeArtifact domain. No long-lived registry credential is stored.

**Package stage:** pushing a built container image to a private registry (ECR, GHCR, Artifact Registry, Harbor) requires authentication. With OIDC-federated AWS credentials, the ECR login token is generated at runtime from the IAM role. With GHCR, the automatically provided `GITHUB_TOKEN` is sufficient for pushes to the same organisation's registry. With Harbor or a self-hosted registry, a robot account credential stored as a CI platform secret is acceptable — it scopes to the registry, not to production infrastructure.

**Deploy stage:** this is where the highest-privilege credentials are needed: the Kubernetes cluster credentials (kubeconfig) and the cloud credentials to update services. This is the stage where OIDC federation and Vault AppRole are most critical. The deploy job should be the only job with access to production IAM roles, and its `environment:` protection gate should require manual approval for production deploys.

**Operate stage (runtime):** applications running in production need secrets for their downstream dependencies: database connections, message queue credentials, third-party API keys, signing keys. These are the secrets managed by Vault Agent, ESO, or Sealed Secrets — delivered as file mounts or Kubernetes Secret references, automatically rotated, never visible in deployment manifests or pod specs.

Understanding this stage-by-stage breakdown helps you apply least privilege precisely: build jobs get registry read credentials, package jobs get registry write credentials, test jobs get no credentials, deploy jobs get OIDC-federated cloud credentials, and pods get short-TTL database credentials from Vault. Each stage's blast radius is contained to exactly what it needs.

#### Worked example: a complete pipeline with secrets at every stage

Here is what a fully-realised secrets architecture looks like for a Python web application deploying to ECS:

```yaml
name: CI/CD pipeline with proper secrets management

on:
  push:
    branches: [main]
  pull_request:

permissions:
  id-token: write
  contents: read
  security-events: write

jobs:
  # Stage 1: Scan — no cloud credentials needed
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@ab928c847f33bb18a1ea1bfd27b77d18773b71f0  # pinned SHA
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # Stage 2: Build — needs private CodeArtifact registry (uses OIDC-federated AWS creds)
  build:
    runs-on: ubuntu-latest
    needs: [secret-scan]
    steps:
      - uses: actions/checkout@ab928c847f33bb18a1ea1bfd27b77d18773b71f0
      - name: Configure AWS credentials for CodeArtifact
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsReadOnlyRole
          aws-region: us-east-1
      - name: Login to CodeArtifact
        run: |
          CODEARTIFACT_TOKEN=$(aws codeartifact get-authorization-token \
            --domain myorg --domain-owner 123456789012 \
            --query authorizationToken --output text)
          pip install -r requirements.txt --index-url \
            "https://aws:${CODEARTIFACT_TOKEN}@myorg-123456789012.d.codeartifact.us-east-1.amazonaws.com/pypi/internal/simple/"
      - name: Build Docker image
        run: docker build -t app:${{ github.sha }} .

  # Stage 3: Test — no cloud credentials needed
  test:
    runs-on: ubuntu-latest
    needs: [build]
    steps:
      - uses: actions/checkout@ab928c847f33bb18a1ea1bfd27b77d18773b71f0
      - name: Run unit tests
        run: pytest tests/unit/
      - name: Run integration tests (using local Docker Compose, no prod creds)
        run: docker-compose -f docker-compose.test.yml up --abort-on-container-exit

  # Stage 4: Package — needs ECR push access (separate role with write permission)
  package:
    runs-on: ubuntu-latest
    needs: [test]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@ab928c847f33bb18a1ea1bfd27b77d18773b71f0
      - name: Configure AWS credentials for ECR push
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsECRPushRole
          aws-region: us-east-1
      - name: Push image to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS \
            --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
          docker tag app:${{ github.sha }} \
            123456789012.dkr.ecr.us-east-1.amazonaws.com/app:${{ github.sha }}
          docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/app:${{ github.sha }}

  # Stage 5: Deploy — production role, requires environment approval
  deploy:
    runs-on: ubuntu-latest
    needs: [package]
    environment: production   # requires reviewer approval
    steps:
      - name: Configure AWS credentials for production deploy
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsProductionRole
          aws-region: us-east-1
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster production \
            --service app-backend \
            --force-new-deployment
```

Notice the principle: four separate IAM roles (read-only, ECR-push, production-deploy), each scoped to the stage that needs it. The test job has no AWS credentials at all. The scan job has no AWS credentials. Even if a malicious dependency is introduced into the test stage, it cannot reach production.

## How to reach for each pattern — and when not to

**Always use OIDC for CI-to-cloud authentication.** There is no scenario where storing a static cloud provider credential in a CI platform's secret store is better than OIDC federation, for any major CI platform (GitHub Actions, GitLab CI, CircleCI) connecting to any major cloud (AWS, GCP, Azure). It is free, adds no infrastructure, and the migration takes half a day. Not using OIDC is a deliberate choice to accept a larger blast radius for zero benefit.

**Use cloud secrets manager (AWS SM / GCP SM / Azure KV) when you are already in that cloud.** The integration is native, the ops cost is low, and ESO provides Kubernetes integration without building custom tooling. The main limitation is static credentials — you are still managing rotation, just with better tooling. For \$5–15/month in AWS Secrets Manager costs, this is a strongly positive investment for any production service.

**Do not adopt HashiCorp Vault as your first secrets management step.** Vault requires a platform team to install, configure, HA-deploy, back up, restore-test, and maintain. A small team that jumps to Vault before their CI pipeline is stable and their cloud IAM is sorted is optimising prematurely. Vault's payoff — dynamic secrets, fine-grained policies, multi-cloud support — is real but requires investment to realise. Reach for Vault when you have a clear need for dynamic credentials (database passwords with automatic revocation) or when you are operating at a scale where the centralised, provider-agnostic secrets API justifies the overhead.

**Do not use Sealed Secrets if you already have ESO with cloud secrets manager.** Both solve the same problem (secrets in GitOps repos) with different trade-offs. Using both creates confusion about which is authoritative. Pick one based on your primary constraint: cloud-provider integration (ESO) vs portability and Git-centric workflow (Sealed Secrets).

**Do not invest in automated rotation tooling for OIDC-federated or Vault-dynamic credentials.** Rotation automation is a mitigation for the world where you must store long-lived credentials. The goal is to eliminate long-lived credentials. Spending engineering time automating rotation of a static key that you should be replacing with OIDC is optimising the wrong thing.

**Do not use `pull_request_target` for anything that runs PR code.** The asymmetry between `pull_request` (fork-safe, no secrets) and `pull_request_target` (base-repo context, secrets accessible) is a persistent source of CI security incidents. Review every use of `pull_request_target` in your workflows for code-injection paths.

**Do not rely on log masking as a security boundary.** GitHub's secret masking replaces exact string matches with `***` in logs. It does not mask variations, base64-encoded versions, or fragments. A secret value split across two log lines, or printed in a transformed form (JSON-escaped, URL-encoded), bypasses masking. Masking is a useful last line of defence, not a substitute for keeping secrets out of log-producing code paths.

**Never rotate manually on a Friday, and never skip the health gate.** The rotation runbook must include an automated health verification step with a timeout and an explicit rollback path. Any rotation that lacks a dual-read window and a health gate is a change-failure waiting to happen. If your rotation tooling cannot support a dual-read window (some legacy credential types do not support multiple valid credentials simultaneously), schedule a maintenance window and have a tested rollback plan.

## Key takeaways

- A secret committed to Git is permanently compromised: git history is immutable, and `git rm` does not remove previous commits. The only remediation is immediate rotation. Prevention requires gitleaks as a pre-commit hook and a CI gate scanning full history on every pull request.
- CI platform secrets are the floor, not the ceiling. They are static, lack rotation automation, offer limited audit, and are vulnerable to the `pull_request_target` attack on public repositories. Use them only for low-sensitivity, non-production configuration.
- The fork PR attack exploits `pull_request_target` to execute fork-PR code with base-repository secrets in scope. Never run untrusted PR code in a `pull_request_target` workflow context.
- OIDC keyless federation is the gold standard for any CI-to-cloud authentication. No stored credential, 1-hour TTL, IAM-condition-locked to a specific repo and environment, free to implement. Migrate every static cloud key in CI to OIDC first — before any other secrets management investment.
- HashiCorp Vault dynamic secrets eliminate long-lived database credentials: a credential created for a 1-hour lease and auto-revoked on expiry cannot be exploited weeks after it was issued. The Vault Agent injector makes dynamic credentials transparent to applications running in Kubernetes.
- The External Secrets Operator is the Kubernetes-native bridge to cloud secrets managers. Define a `SecretStore` for the backend, an `ExternalSecret` for each secret, and the controller handles sync, refresh, and update. Pair it with Reloader for automatic rolling restarts on secret changes.
- Sealed Secrets makes GitOps safe for secrets: commit an encrypted `SealedSecret` CRD, not plaintext base64. Back up the controller private key immediately to a separate secrets store — key loss makes all sealed secrets permanently undecryptable.
- Zero-downtime rotation requires a dual-read window: both old and new credentials valid simultaneously during the rolling restart window. Automate the rotation runbook with an explicit health gate between the propagation phase and the revocation phase.
- Every rotation runbook must have a tested rollback path that handles the case where the old credential has already been revoked.
- Secrets management and supply-chain security are inseparable: OIDC scoping limits blast radius when a compromised build dependency exfiltrates credentials; pinned action SHAs and signed artifacts prevent malicious code from reaching the execution environment in the first place.

## Further reading

- [GitHub Actions OIDC documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect) — canonical source for the JWT structure, available claims, and step-by-step provider configuration
- [AWS IAM: configure GitHub OIDC identity provider](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html) — the official thumbprint, trust policy, and `AssumeRoleWithWebIdentity` documentation
- [HashiCorp Vault dynamic database secrets tutorial](https://developer.hashicorp.com/vault/tutorials/db-credentials/database-secrets) — PostgreSQL + MySQL engine setup, lease configuration, and revocation
- [External Secrets Operator documentation](https://external-secrets.io/latest/) — SecretStore, ExternalSecret, ClusterSecretStore reference and all supported backend providers
- [Sealed Secrets by Bitnami Labs](https://github.com/bitnami-labs/sealed-secrets) — installation, kubeseal workflow, key rotation, and backup procedures
- [SLSA framework](https://slsa.dev) — how secrets management fits into the broader supply-chain provenance and trust model
- [CI/CD mental model: from commit to production](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series spine; where secrets appear across the commit→build→test→package→deploy→operate pipeline
- [Software supply chain security: the new frontier](/blog/software-development/ci-cd/software-supply-chain-security-the-new-frontier) — pinning action SHAs, SBOM attestation, cosign signing, and how the Codecov and SolarWinds attacks could have been limited
- [Securing the pipeline itself](/blog/software-development/ci-cd/securing-the-pipeline-itself) — complementary pipeline security controls: least-privilege job permissions, runner isolation, dependency pinning
- [Configuration and secrets in Kubernetes](/blog/software-development/ci-cd/configuration-and-secrets-in-kubernetes) — the runtime configuration model: ConfigMaps, Secrets, volume mounts, and the application-side pattern for consuming what this post delivers

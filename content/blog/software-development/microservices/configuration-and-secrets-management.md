---
title: "Configuration and Secrets Management: Your Other Deploy Surface"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How dozens of services get their config and credentials safely — the 12-factor split, real secret stores, short-lived rotated creds, validate-at-boot, and why a bad config push has the same blast radius as a bad deploy."
tags:
  [
    "microservices",
    "configuration",
    "secrets-management",
    "vault",
    "kubernetes",
    "external-secrets",
    "12-factor",
    "distributed-systems",
    "software-architecture",
    "backend",
    "security",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/configuration-and-secrets-management-1.webp"
---

At 14:32 on a Thursday, ShopFast's checkout stopped working across every region at once. Not slowly, not for a subset of users — every "Place Order" button in the world returned a spinner that never resolved. The on-call engineer pulled up the dashboards and saw something that made no sense: there had been no deploy. No new image, no code merge, no Kubernetes rollout of any service. The last deployment had shipped four hours earlier and had been green the whole time. Yet checkout was dead, the payment service was timing out, and the order service's thread pools were pinned at 100% utilization across all forty pods. The post-mortem found the cause in a single line of a YAML file that nobody thought of as code: someone had edited a shared `ConfigMap` to change a timeout from `30` to `30000`, intending milliseconds, and the value was read as seconds. Every service that read that config now waited thirty thousand seconds — eight hours — before giving up on a downstream call. There was no bad code anywhere. There was a bad *value*, and it propagated to the entire fleet in fifteen seconds, faster than any deploy pipeline could have, with none of the safeguards a deploy gets.

This is the uncomfortable lesson that separates engineers who have been paged for a config incident from those who haven't: **configuration is part of your deploy surface, and secrets are part of your attack surface, and both deserve the exact same rigor you give to code.** A junior engineer thinks of config as "just settings" — strings you tweak in a file, less serious than real code. That instinct is how a one-character typo takes down a hundred-service fleet in under a minute, and how a credential committed "just for testing" ends up scraped from a public repo and used to drain a cloud account. Config and secrets are quietly two of the most dangerous things in a microservices system precisely because they don't *feel* dangerous. They have no compiler, often no review, frequently no rollout safety, and a blast radius that can exceed any single deploy because one shared value can hit every service at once.

By the end of this post you will be able to do four concrete things. First, split config from secrets correctly and route each to the right store — non-secret settings to ConfigMaps and a config service, credentials and keys to a real secret store like HashiCorp Vault or a cloud Secrets Manager. Second, wire short-lived, automatically-rotated database credentials into a fleet of services through the external-secrets operator, so a leaked credential expires in an hour instead of living for eighteen months. Third, treat config as code: validate it at boot so a bad value fails one pod fast instead of crashing forty, version it, and roll it out staged like a canary. Fourth, rotate a secret across forty services with zero downtime, and reason about exactly how fast you could rotate if a key leaked right now. The figure below is the map of the whole article — the layers that sit between your immutable image and the running process.

![A stacked diagram showing immutable code at the base then non-secret config then secrets then a validate-at-boot gate then the running process with per-environment behavior](/imgs/blogs/configuration-and-secrets-management-1.webp)

This post closes Track 6 of the series — the deployment and infrastructure track. We have [containerized our services with good Docker practices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices), learned [the Kubernetes essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) that schedule and heal them, and shipped them safely with [blue-green, canary, and feature-flag deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags). The thread running through all of those is *change management*: how you alter what's running in production without breaking it. Config and secrets are the change surface that the deploy pipeline doesn't always cover. The same image runs in dev, staging, and prod — what makes it behave differently in each is config, and what lets it authenticate to its database and payment processor is secrets. Get those two things right and your fleet is portable, auditable, and recoverable. Get them wrong and you have built a system where the most fragile and most attacked components are also the least-governed. This post also sets up the security track that follows: [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) and [authentication and authorization with OAuth2, JWT, and token propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation), both of which lean heavily on the secret-handling foundations we lay here.

## The 12-factor principle: config lives in the environment

Start with the foundational rule, because everything else is a consequence of it. The Twelve-Factor App methodology — written by the Heroku team and still the clearest articulation of how a deployable service should behave — devotes its third factor to config, and it makes one sharp claim: **strict separation of config from code.** The test it offers is brutal and useful: *could you open-source your codebase right now, this instant, without leaking any credentials?* If the answer is no — if there's a database password in a settings file, an API key in a constant somewhere — then your config is not properly separated from your code, and you have a problem whether or not you ever open-source anything.

The principle rests on a definition. Config, in the twelve-factor sense, is **everything that varies between deploys** — between your dev laptop, the staging cluster, and production. The database connection string differs. The payment processor's API endpoint points at a sandbox in staging and the real thing in prod. Log verbosity is high in dev and low in prod. Feature flags are on in one environment and off in another. None of these belong in the code, because the code is the *same* across all three environments. This is the "one codebase, many deploys" idea: a single immutable artifact — for us, one Docker image — runs everywhere, and what makes it behave differently is the config injected at deploy time. We covered why that one immutable artifact matters in the [containerizing post](/blog/software-development/microservices/containerizing-microservices-docker-best-practices); config is the other half of that contract. The image is the constant; config is the variable; never bake a variable into a constant.

Why "in the environment" specifically? The twelve-factor answer is that environment variables are the lowest-common-denominator config mechanism — every language, every OS, every container runtime supports them, with no library and no parsing. They are also trivially easy to change per-deploy without touching code, and impossible to accidentally commit if your code never references a literal value. In a Kubernetes world we've extended "the environment" to include ConfigMaps and mounted files and config services, but the spirit is identical: config is an *input* to the running process, supplied from outside, not a hard-coded part of it.

Here is the anti-pattern that this principle exists to kill, the thing a junior writes and a senior reviews out:

```python
# config.py — the WRONG way. This is config baked into code.
DATABASE_URL = "postgres://shopfast:hunter2@prod-db.internal:5432/orders"
PAYMENT_API_KEY = "sk_live_REDACTED_EXAMPLE_NOT_A_REAL_KEY"
LOG_LEVEL = "INFO"
FEATURE_NEW_CHECKOUT = False
```

Four lines, four distinct sins. `DATABASE_URL` is config (varies per environment) *and* contains a secret (the password `hunter2`). `PAYMENT_API_KEY` is a live secret sitting in the codebase. `LOG_LEVEL` and `FEATURE_NEW_CHECKOUT` are non-secret config that should vary per deploy but are now hard-coded. To change any of them you must edit code, review it, build a new image, and deploy — which means a config change has all the *latency* of a code change but, crucially, not the *safety*, because nobody treats this file like real logic. And the moment this repo is cloned to a laptop or a CI runner with broad read access, the live payment key is everywhere. The right version reads every one of these from the environment, never holds a default for a secret, and fails loudly if a required value is missing — we'll build exactly that, with validation, later in this post.

### The first split that organizes everything: config versus secrets

The single most useful distinction in this entire topic is the one between **config** and **secrets**, because it determines which store a value goes into, who can read it, how it's encrypted, whether it rotates, and what happens if it leaks. Both are "things injected from the environment," but they could not be more different in their handling.

**Config** is non-sensitive operational settings. The database *host and port* (not the password). The log level. The number of worker threads. A feature-flag default. The URL of the payment processor's API. The cache TTL. If one of these leaked, you would shrug — knowing that ShopFast's order service logs at `INFO` and talks to `pricing-svc:8080` is not a security event. Config can live in version control (it's often *good* that it does, for audit and rollback), it can be read by anyone on the team, and it changes for operational reasons.

**Secrets** are credentials, keys, and tokens — anything that grants *access*. The database password. The payment processor's `sk_live_` API key. The JWT signing key. The TLS private key. An OAuth client secret. If one of these leaks, it is a security incident, full stop, because possession of the secret *is* the access. Secrets must never live in version control, must be encrypted at rest and in transit, must be readable only by the specific service that needs them, should rotate regularly, and demand an audit trail of who and what accessed them when.

The reason this split matters operationally is that the *handling* is completely different, and conflating them gives you the worst of both. Put a secret in a ConfigMap and you've now got an unencrypted credential that's readable by anyone with cluster read access and probably checked into a Git repo as YAML — a leaked secret waiting to happen. Put non-secret config in Vault and you've buried a value that everyone needs to read behind an access-control system designed for credentials, paying complexity and latency for nothing. The discipline is simple to state: **non-secret config goes in config stores; secrets go in secret stores; the boundary is "would I care if this leaked?"** The figure that opened this post stacks them as separate layers precisely because they are separate concerns that happen to converge inside the same process.

There is a gray zone, and seniors get it right by erring toward the secret store. A database *connection string* contains both — host and port (config) and username and password (secret). The correct pattern is to split it: deliver the host and port and database name as non-secret config, and the password as a secret, and assemble the connection string inside the application at boot. We'll write exactly that. The rule of thumb for the gray zone: **if a string contains a credential, the whole string is treated as a secret unless you've split the credential out.**

A few more gray-zone calls worth pre-deciding so you're not debating them mid-incident. An internal API endpoint URL is config — it's not a secret unless the URL itself embeds a token (some webhook URLs do; if so, it's a secret). A tenant ID or account ID is usually config, not a secret — it identifies but doesn't authenticate. A bucket name is config; the credential to write to the bucket is a secret. An OAuth *client ID* is config (it's public by design); the OAuth *client secret* is, as the name says, a secret. The discriminating question is always the same one: **does possessing this value grant access, or merely describe the system?** Access-granting values are secrets and go in the secret store with encryption, rotation, and audit; descriptive values are config and go in the config store where the team can read and version them freely. When you genuinely can't decide, treat it as a secret — the cost of over-protecting a non-secret is a little extra complexity; the cost of under-protecting a real secret is a breach.

## Where non-secret config lives, and the trade-offs

Even once you've decided a value is non-secret config, you have a genuine choice about *how* it reaches the process, and the options trade off against each other in ways that bite if you pick blindly. There are four practical delivery channels, and the figure below lays them out against the properties that actually matter when you're operating a fleet.

![A decision matrix comparing env vars, ConfigMap, config service, and baked file across change-without-redeploy, audit history, per-environment ease, and operational complexity](/imgs/blogs/configuration-and-secrets-management-2.webp)

**Environment variables** are the twelve-factor default and the simplest thing that works. The process reads `os.environ["LOG_LEVEL"]` and that's it — no library, no parsing, supported everywhere. Their weakness is that they are fixed for the life of the process: to change an env var you must restart the process, which in Kubernetes means a rolling deploy of the Deployment. They also have weak audit (you have to look at the Deployment spec's history to see what they were) and they leak easily into logs and crash dumps if you're not careful — which is exactly why secrets-in-env-vars is a debated practice we'll return to. For genuinely static, non-secret settings that change only at deploy time, env vars are perfect.

**ConfigMaps** are Kubernetes' native config object: a key-value (or whole-file) blob, stored in the cluster, that you mount into pods as env vars or as files. Their big advantage over raw env vars is that they live as a distinct, versionable object — you keep the ConfigMap YAML in Git, you get a Git history of every change, and Kubernetes tracks the object's resource version. They are the right default for Kubernetes-hosted services. Their catch is subtle and bites people: **changing a ConfigMap does not automatically restart the pods that consume it.** If you mount the ConfigMap as env vars, a change has *no effect at all* until the pods restart — and nothing triggers that restart automatically. If you mount it as files, Kubernetes eventually updates the file in the pod (with a propagation delay of up to a minute or two), but your application has to actively re-read the file to notice. We'll deal with this gotcha directly.

**A config service** — a dedicated runtime config system like a feature-flag platform (LaunchDarkly, Unleach, Flagsmith), Spring Cloud Config, AWS AppConfig, or a homegrown service backed by a database — is the only option in the matrix that changes config *without a redeploy or restart*. The application subscribes to config and gets pushed updates, or polls for them, and applies them live. This is enormously powerful for things you genuinely need to change at runtime: feature flags, kill switches, rate-limit thresholds you tune during an incident. It is also the most operationally complex (it's another service to run and make highly available, and now your service has a new runtime dependency) and the most *dangerous*, because a live config change has no deploy pipeline in front of it — it takes effect everywhere, instantly, with whatever safety you bothered to build into the config service itself. We'll spend a section on dynamic config and its risks.

**A baked file** — config compiled into the image at build time — is the anti-pattern from the twelve-factor section, included in the matrix to be explicit about why it loses: any change requires a rebuild and a redeploy, and the same image can no longer run in multiple environments because the config is welded into it. The only legitimate use is for config that is genuinely identical across all environments and changes only when the code does — at which point it's arguably not "config" in the twelve-factor sense at all.

The practical recommendation for a microservices fleet: **ConfigMaps as the default delivery for per-environment non-secret config** (versioned in Git, mounted as env vars for things that change only at deploy, or as files for things you want to reload), plus **a config service for the small set of values you truly need to flip at runtime** (feature flags and operational kill switches). Env vars are how the ConfigMap reaches the process; the ConfigMap is how you version it.

### A ConfigMap and env injection, concretely

Here is ShopFast's order service getting its non-secret config from a ConfigMap, mounted as environment variables. This is the bread-and-butter pattern.

```yaml
# configmap.yaml — non-secret config, versioned in Git, one per environment
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-svc-config
  namespace: shopfast-prod
data:
  LOG_LEVEL: "INFO"
  DB_HOST: "orders-db.shopfast-prod.svc.cluster.local"
  DB_PORT: "5432"
  DB_NAME: "orders"
  PAYMENT_API_URL: "https://api.stripe.com/v1"
  DOWNSTREAM_TIMEOUT_MS: "800"     # milliseconds — units in the key name, lesson learned
  WORKER_CONCURRENCY: "16"
```

```yaml
# deployment.yaml (excerpt) — inject the ConfigMap as env vars
spec:
  template:
    spec:
      containers:
        - name: order-svc
          image: registry.shopfast.io/order-svc:v2.4.1
          envFrom:
            - configMapRef:
                name: order-svc-config
          # secrets come from a different source — see the ExternalSecret below
```

Two deliberate choices here pay off later. First, the timeout key is named `DOWNSTREAM_TIMEOUT_MS` with the unit in the name — this is a direct response to the units bug that opened the post. Ambiguous-unit config values are a recurring outage cause; encode the unit in the key so a reviewer sees `30000` next to `_MS` and flinches. Second, there is *no secret in this ConfigMap*. The DB host, port, and name are here; the DB password is conspicuously absent, because it comes from the secret store. The application assembles the connection string at boot from the non-secret parts plus the secret part.

## Secrets: the leaked-.env disaster and what a real store buys you

Now the harder half. The first rule of secrets is the one most often broken in the most expensive way: **never, ever commit a secret to version control.** The `.env` file disaster is so common it has a genre of its own. An engineer creates a `.env` file with the real production database password and Stripe key to test something locally, the file isn't in `.gitignore` because nobody added it, `git add -A` sweeps it in, and now the live credentials are in the repo's history *forever* — and Git history is not erased by deleting the file in a later commit; the secret sits in the object database until you rewrite history, which you almost never do. If that repo is public, automated scrapers find `sk_live_` and `AKIA` prefixes within *minutes* — there are bots whose entire job is to scan new public commits for credential patterns and use them before you notice. If it's private, every contributor, every CI runner, and every laptop that ever cloned it now holds your production keys.

The fix has two parts: stop secrets from reaching the repo (prevention), and put real secrets in a real store (the positive pattern). For prevention, you need two cheap, mandatory controls. A `.gitignore` that excludes the obvious offenders, and a pre-commit hook that *scans* for credential patterns so a secret can't sneak in even if someone bypasses `.gitignore`:

```gitignore
# .gitignore — keep secrets and local env out of the repo
.env
.env.*
*.pem
*.key
secrets/
**/credentials.json
**/serviceaccount*.json
```

```yaml
# .pre-commit-config.yaml — scan every commit for secrets before it lands
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
```

The pre-commit hook runs on every developer's machine before a commit is created, and you also run the same scanner in CI as a hard gate so the protection doesn't depend on every developer having installed the hook. `gitleaks` and `detect-secrets` both ship with rules for common credential shapes (AWS keys, Stripe keys, private-key headers, high-entropy strings). They are not perfect — they miss novel formats and produce false positives — but they catch the overwhelming majority of accidental commits, and a false positive that makes you confirm "yes, this random-looking string really isn't a secret" is a cheap price. The senior framing: **prevention is layered, not single-point.** `.gitignore` is the first layer, the local pre-commit hook the second, the CI scan the third, and a periodic full-history scan the fourth. Any one layer fails sometimes; the stack rarely does.

But not committing secrets only tells you where they *shouldn't* go. Where *should* they go? Into a purpose-built secret store, and the choice of store is a real decision with real trade-offs — which is the next figure.

![A decision matrix comparing Kubernetes Secret, Sealed Secret, cloud Secrets Manager, and Vault across encryption at rest, dynamic short-lived support, automatic rotation, audit logging, and operational cost](/imgs/blogs/configuration-and-secrets-management-3.webp)

**Kubernetes Secrets** are the built-in option, and they are better than nothing but weaker than most people assume. A Kubernetes `Secret` is, by default, just base64-encoded — *not encrypted* — data stored in `etcd`. Base64 is encoding, not encryption; anyone who can read the Secret object reads the value. You can turn on encryption-at-rest for `etcd` (and you should), but even then the Secret is decrypted and readable by anyone with `get secrets` RBAC permission in the namespace, there is no rotation, and there is no audit log of which pod read which secret when. Worse, the obvious way to manage a Secret — write its YAML and commit it to Git — puts the base64'd secret straight into the repo, recreating the very disaster we just spent a section preventing. Plain Kubernetes Secrets, committed to Git, are a trap dressed as a feature.

**Sealed Secrets** (the Bitnami project) solve the "can't commit the Secret to Git" problem elegantly. You encrypt a Secret with a public key whose private half lives only in the cluster, producing a `SealedSecret` that is *safe to commit* — only the in-cluster controller can decrypt it back into a real Secret. This gives you GitOps for secrets: the encrypted blob lives in your repo, versioned and reviewable, and the cluster decrypts it on apply. The catch is that sealed secrets are still *static* — there's no rotation, no dynamic issuance, no audit of access; you've solved the storage-and-delivery problem but not the lifecycle problem. They're a great fit for a smaller team that wants GitOps and doesn't yet need rotation or short-lived creds.

**Cloud Secrets Managers** (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault) and **KMS** (Key Management Service for the encryption keys themselves) are the managed-service answer. They encrypt at rest with a managed KMS key, they log every access in the cloud audit trail (CloudTrail and friends), and AWS Secrets Manager in particular can *automatically rotate* certain secrets (notably RDS database credentials) on a schedule with a managed Lambda. You pay a per-secret monthly fee and per-API-call cost, and you tie yourself to the cloud provider, but for most teams already on a cloud this is the pragmatic sweet spot: real encryption, real audit, managed rotation for the common cases, and no Vault cluster to operate.

**HashiCorp Vault** is the most powerful and the most operationally demanding. Its standout capability is **dynamic secrets**: instead of storing a static database password, Vault is configured with admin access to your database and *generates a fresh, unique, short-lived credential on demand* when a service asks for one — say, a Postgres user valid for one hour, after which Vault automatically revokes it. This flips the entire security model: there is no long-lived database password to leak, because the credential a service holds expires within the hour, and every credential is unique per service (so a leak is traceable to exactly one consumer). Vault also does encryption-as-a-service, detailed audit logging, and fine-grained policy. The cost is that Vault is a serious distributed system you must run, secure, unseal, and keep highly available — if Vault is down and a service needs a fresh credential, that service can't start. The figure below makes the dynamic-secrets argument vivid: it's the difference between a leak that lasts months and a leak that lasts an hour.

![A before-and-after comparison contrasting a static key checked into git that is never rotated for eighteen months with Vault dynamic credentials issued at boot with a one-hour TTL and auto-rotation](/imgs/blogs/configuration-and-secrets-management-5.webp)

#### Worked example: a leaked long-lived key versus short-lived rotated creds

Make the exposure window concrete. Suppose ShopFast's payment service authenticates to the database with a static password that was set once during initial setup and never rotated — a depressingly common state. An attacker obtains that password: maybe it leaked through a log line, maybe through a misconfigured backup, maybe through a former employee's laptop. With a static, never-rotated credential, **the exposure window is "from the moment it leaked until someone notices and rotates it."** In practice, with no rotation discipline and no access audit, that's often *months* — industry breach reports routinely cite mean-time-to-detect for credential compromise in the range of 200-plus days. Call it 6 months of full database access, undetected.

Now run the same leak against Vault dynamic credentials. The payment service got its database credential from Vault at boot, with a TTL of 1 hour, and Vault automatically rotates it. The attacker obtains the credential — but it was *already* going to be revoked within the hour. **The exposure window is at most 60 minutes**, and in expectation about 30 minutes (the leak lands somewhere in the credential's lifetime). The math is stark: 6 months versus 30 minutes is a factor of roughly 8,600× less exposure time. And there's a second-order win — because every service gets a *unique* dynamic credential, Vault's audit log shows exactly which credential was used for the attacker's queries, so you can identify which service's credential leaked, which you cannot do at all when forty services share one static password. The lesson seniors internalize: **rotation isn't hygiene theater; it bounds the blast radius of every leak you don't yet know about.** The credential you never rotated is the one already in someone else's hands.

## Injecting secrets: env vars, mounted files, and the sidecar agent

A secret store holds the secret; the service still has to *receive* it. There are three injection mechanisms, and the choice has real consequences for both security and the ability to rotate without restarting.

**Environment variables** are the simplest: the secret arrives as `os.environ["DB_PASSWORD"]`. This is twelve-factor-idiomatic and works everywhere, but it has two well-known weaknesses for *secrets* specifically. First, environment variables are easy to leak: they show up in crash dumps, in `/proc/<pid>/environ` (readable by anything with access to the process), in error-reporting tools that capture the environment, and in child processes that inherit the parent's environment (so if your app shells out to a subprocess, the subprocess sees your DB password). Second — and this is the operational killer — **env vars are fixed at process start**, so a secret delivered as an env var *cannot be rotated without restarting the process*. If your rotation strategy is "issue a new credential," an env-var secret means "restart every pod," which is a much bigger hammer.

**Mounted files** deliver the secret as a file inside the pod (e.g. `/var/run/secrets/db/password`). The application reads the file at the path. This is meaningfully safer than env vars — files don't leak into crash dumps or child-process environments the same way, and you can set restrictive file permissions — and critically, **a mounted file can be updated in place**, which means a secret can be *rotated without restarting the pod* if the application re-reads the file when it changes. This is the foundation of zero-downtime rotation. The catch is that the application has to be written to re-read the file (or watch it for changes) rather than reading it once at boot and caching it forever.

**Sidecar / agent injection** puts a small companion process in the pod (the Vault Agent, or a CSI driver) that authenticates to the secret store, fetches secrets, keeps them refreshed, and writes them to a shared in-memory volume that the application reads. The application itself never talks to Vault — it just reads a file the agent maintains. This is the most powerful pattern for dynamic secrets: the agent handles authentication, renewal, and rotation transparently, and can even template config files and signal the app to reload. The cost is an extra container per pod (more memory, more moving parts) and the operational complexity of running the agent.

The recommendation maps to your maturity. **Mounted files are the right default for any secret you want to rotate**, because in-place update plus re-read gives you zero-downtime rotation without a sidecar. **A sidecar agent is worth it when you've adopted dynamic short-lived secrets at scale**, because something has to continuously renew the lease and you don't want that logic in every application. **Env vars are acceptable only for secrets that genuinely never need runtime rotation** — and even then, mounted files are usually better.

There's one more reason file-based injection wins for secrets that doesn't show up until you're debugging a production incident: **observability hygiene.** When a secret is an environment variable, it shows up everywhere the environment is captured — in your APM tool's "process attributes" panel, in a Sentry or Rollbar error report that snapshots the environment, in the output of a `kubectl describe pod` if anyone ever mistakenly templated it into the spec, and in any log line that dumps `os.environ` for debugging. Every one of those is a place a secret can quietly land in a system that *isn't* designed to protect secrets — your log aggregator, your error tracker, your screen during a screen-share. A file at `/var/run/secrets/...` is read deliberately by the code that needs it and shows up in none of those incidental captures. The difference matters because the most common way secrets leak isn't a dramatic breach — it's a secret accidentally written to a log that's retained for ninety days and indexed by a search tool half the company can query. File-based injection narrows the surface where a secret can accidentally appear, and that narrowing is worth as much as the rotation benefit. The figure below shows the full injection flow with the external-secrets operator, and note how config and secrets travel separate paths that converge only inside the pod.

![A graph showing a git repo and ConfigMap delivering settings directly to the pod while the external-secrets operator pulls from Vault, syncs a short-lived Kubernetes Secret, and that flows into the same pod as a mounted file](/imgs/blogs/configuration-and-secrets-management-4.webp)

### The external-secrets operator: bridging Vault and Kubernetes

There is a tension between "Kubernetes-native secret delivery" (apps consume Kubernetes `Secret` objects, which is the path of least resistance) and "the secret should actually live in Vault or a cloud Secrets Manager" (which is where the real encryption, rotation, and audit are). The **External Secrets Operator (ESO)** resolves this tension and has become the standard glue. You declare an `ExternalSecret` — a custom resource that says "fetch *this* secret from *that* external store and materialize it as a Kubernetes Secret named *this*" — and the operator continuously syncs the value from the source of truth into a Kubernetes Secret that your pods consume normally. Your applications keep consuming plain Kubernetes Secrets (simple); your *actual* secrets live in Vault or AWS Secrets Manager (secure, rotated, audited); the operator keeps them in sync.

Here is ShopFast's `ExternalSecret` pulling the database password and the payment processor's API key from Vault:

```yaml
# external-secret.yaml — fetch from Vault, materialize as a k8s Secret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: order-svc-secrets
  namespace: shopfast-prod
spec:
  refreshInterval: 1m            # re-pull from Vault every minute -> picks up rotation
  secretStoreRef:
    name: vault-backend          # points at the Vault SecretStore config
    kind: ClusterSecretStore
  target:
    name: order-svc-secret       # the k8s Secret the operator creates/updates
    creationPolicy: Owner
  data:
    - secretKey: DB_PASSWORD     # key inside the resulting k8s Secret
      remoteRef:
        key: shopfast/order-svc/db   # path in Vault
        property: password
    - secretKey: PAYMENT_API_KEY
      remoteRef:
        key: shopfast/order-svc/payment
        property: api_key
```

```yaml
# deployment.yaml (excerpt) — mount the synced Secret as a FILE, not env, so it can rotate
spec:
  template:
    spec:
      containers:
        - name: order-svc
          envFrom:
            - configMapRef:
                name: order-svc-config     # non-secret config
          volumeMounts:
            - name: secrets
              mountPath: /var/run/secrets/order-svc
              readOnly: true
      volumes:
        - name: secrets
          secret:
            secretName: order-svc-secret    # the ESO-managed Secret
```

The `refreshInterval: 1m` is the load-bearing line for rotation: every minute, ESO re-pulls from Vault, and if Vault has rotated the credential, ESO updates the Kubernetes Secret, Kubernetes propagates the updated file into the mounted volume, and a reload-aware application picks it up — all without a restart. Note also that the secret is mounted as a *file* (`volumeMounts`), not via `envFrom`, precisely so rotation can happen in place. Config comes from the ConfigMap via `envFrom`; secrets come from the file mount; the two paths converge in the pod exactly as the figure shows.

For teams not on Vault, the same `ExternalSecret` shape works against AWS Secrets Manager, GCP Secret Manager, or Azure Key Vault by swapping the `secretStoreRef` — the operator abstracts the backend, which is part of why it's become the default. And for teams that want GitOps without an external store, here's the Sealed Secret alternative, where the encrypted blob lives safely in Git:

```yaml
# sealed-secret.yaml — safe to commit; only the in-cluster controller can decrypt it
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: order-svc-secret
  namespace: shopfast-prod
spec:
  encryptedData:
    DB_PASSWORD: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...   # RSA-encrypted, public-key sealed
    PAYMENT_API_KEY: AgCXr8eF2kLmNp9qZ0aB1cD2eF3g...
  template:
    metadata:
      name: order-svc-secret
      namespace: shopfast-prod
```

The `encryptedData` values are encrypted with the cluster's public sealing key, so committing this file to Git leaks nothing — the private key never leaves the cluster. The trade-off, as the secret-store matrix showed, is that this is static: you get safe storage and GitOps, but not rotation or dynamic issuance. It's the right choice when you want secrets in Git and don't yet need Vault's lifecycle features.

## Config is code: validate at boot, version, and roll back

Return to the incident that opened this post. A timeout value of `30000` seconds reached forty services in fifteen seconds and took down checkout, and there was no bad code — only a bad value. The deep lesson is this: **a bad config push has the same blast radius as a bad deploy, often larger, because one shared config value can hit every service at once and config frequently skips the safety rails a deploy gets.** A code deploy goes through review, CI, a build, and a staged rollout. A config change, in too many shops, is an edit to a YAML file applied straight to prod. If config has the blast radius of a deploy, it must earn the *rigor* of a deploy. That rigor has three parts: validate it, version it, and roll it out staged.

**Validate at boot — fail fast on missing or invalid config.** This is the single highest-leverage practice in the whole post. The idea is that a service must *refuse to start* if its config is missing or invalid, rather than starting and failing mysteriously later. A service that boots with a missing database password and then throws cryptic errors on the first request is far worse than a service that prints `FATAL: DB_PASSWORD not set` and exits immediately — the first wastes an hour of debugging during an incident; the second tells you exactly what's wrong before any traffic hits it. We touched on validate-at-boot in the [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice) post as part of what makes a service production-ready; here it's the load-bearing defense against bad config. Here's ShopFast's order service validating its config at boot, in Go:

```go
// config.go — load, validate, and fail fast. No service starts with bad config.
package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	LogLevel       string
	DBHost         string
	DBPort         int
	DBName         string
	DBPassword     string // from the mounted secret file, not env
	TimeoutMS      int
	WorkerConc     int
}

func Load() (*Config, error) {
	c := &Config{
		LogLevel: getEnv("LOG_LEVEL", "INFO"),
		DBHost:   os.Getenv("DB_HOST"),
		DBName:   os.Getenv("DB_NAME"),
	}

	// Required, no default — fail loudly if missing.
	if c.DBHost == "" {
		return nil, fmt.Errorf("DB_HOST is required and not set")
	}
	if c.DBName == "" {
		return nil, fmt.Errorf("DB_NAME is required and not set")
	}

	// Secret comes from the mounted file, never from env.
	pw, err := os.ReadFile("/var/run/secrets/order-svc/DB_PASSWORD")
	if err != nil {
		return nil, fmt.Errorf("reading DB_PASSWORD secret file: %w", err)
	}
	c.DBPassword = string(pw)

	// Parse AND range-check numerics. This is what catches the "30000" bug.
	c.DBPort, err = parsePort(os.Getenv("DB_PORT"))
	if err != nil {
		return nil, fmt.Errorf("DB_PORT: %w", err)
	}
	c.TimeoutMS, err = parseRange("DOWNSTREAM_TIMEOUT_MS", 50, 10000) // 50ms..10s sane
	if err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}
	c.WorkerConc, err = parseRange("WORKER_CONCURRENCY", 1, 256)
	if err != nil {
		return nil, err
	}
	return c, nil
}

func parseRange(key string, min, max int) (int, error) {
	v, err := strconv.Atoi(os.Getenv(key))
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer: %w", key, err)
	}
	if v < min || v > max {
		// 30000 is OUTSIDE 50..10000 -> this boots-fails instead of hanging.
		return 0, fmt.Errorf("%s=%d out of allowed range [%d,%d]", key, v, min, max)
	}
	return v, nil
}
```

The crucial line is in `parseRange`: a `DOWNSTREAM_TIMEOUT_MS` of `30000` is *out of the allowed range* `[50, 10000]`, so the service refuses to boot and prints exactly why. Had this validation existed on that Thursday, the first pod to read the bad ConfigMap would have crash-looped immediately with a clear error — and crucially, because Kubernetes does a *rolling* update, only one pod would have updated at a time, the new pod would fail its readiness check, the rollout would halt, and the other thirty-nine pods would keep serving on the old, good config. The bad value never reaches the fleet. **Validation at boot converts a fleet-wide silent outage into a single failed pod and a halted rollout.** That is the entire argument, and it's worth more than any other single line in this post.

**Range-check, don't just type-check.** Notice the validation does more than confirm `TIMEOUT_MS` is an integer — it confirms the integer is *sane*. `30000` parses fine as an integer; it's only the range check that catches it. Bad config is rarely the wrong *type*; it's usually a plausible-looking value that's out of bounds. Validate against the bounds your service can actually tolerate. The discipline generalizes: a worker-concurrency of `0` is a valid integer that means "do no work," a connection-pool size of `100000` is a valid integer that means "exhaust the database's connection limit," and a cache TTL of `-1` is a valid integer that means something nobody intended. Each of these passes a type check and fails a range check, and each has caused a real outage somewhere. The boundaries you encode are a compact specification of what your service can survive — write them down in the validation function, and the validation function becomes the place a new engineer learns the service's operating envelope.

**Cross-field validation catches the subtle ones.** Some config is only invalid in combination. A read timeout that's longer than the overall request deadline is internally inconsistent — the read will never get to time out before the request is abandoned. A retry budget of 3 attempts with a per-attempt timeout of 1 second under a request deadline of 2 seconds can't possibly complete all its retries. These cross-field checks belong in the same boot-time validation: after parsing each field, assert the *relationships* between them hold, and fail fast if they don't. The most insidious config bugs are the ones where every individual value looks reasonable and only their interaction is broken — and those are exactly the ones a per-field type check waves through.

**Version config and roll it back.** Because ConfigMaps live in Git, every change is a commit you can revert, and because the rollout is a Kubernetes operation, you can roll back the config the same way you roll back a deploy — `kubectl rollout undo`, or revert the Git commit and let your GitOps controller reconcile. This is why ConfigMaps-in-Git beats ad-hoc `kubectl edit`: an `edit` leaves no reviewable history and can't be cleanly reverted, while a committed ConfigMap is auditable and revertible. **Treat a config change as a pull request**, with review, exactly like code. The reviewer who saw `DOWNSTREAM_TIMEOUT_MS: "30000"` in a diff next to a `_MS` key name would have caught the bug before merge.

**Test config changes.** You test code; test config too. A cheap, high-value test is a CI job that loads each environment's config through the *same validation function* the service uses at boot, and fails the build if validation fails. That catches the bad value in CI, before it ever reaches a cluster:

```bash
# ci-validate-config.sh — run the service's own validation against each env's config
set -euo pipefail
for env in dev staging prod; do
  echo "Validating config for ${env}..."
  # Render the env's ConfigMap into env vars, then run the service in --validate-only mode
  # which calls config.Load() and exits 0 if valid, non-zero with the error if not.
  kubectl get configmap order-svc-config -n "shopfast-${env}" -o json \
    | jq -r '.data | to_entries[] | "\(.key)=\(.value)"' > "/tmp/${env}.env"
  env $(cat "/tmp/${env}.env" | xargs) ./order-svc --validate-only \
    || { echo "FAIL: ${env} config is invalid"; exit 1; }
done
echo "All environments' config validates."
```

The figure below contrasts the two worlds — an unvalidated push to the whole fleet versus a validated, staged rollout — and the difference in blast radius is the whole point.

![A before-and-after comparison showing an unvalidated config push to all forty services at once with a fleet-wide blast radius versus a validated and staged rollout that canaries one pod first and caps the blast radius at one pod](/imgs/blogs/configuration-and-secrets-management-7.webp)

#### Worked example: the blast radius of a bad global config push

Put numbers on the opening incident so the blast-radius argument is concrete. ShopFast runs 40 services, and the bad `DOWNSTREAM_TIMEOUT_MS` lived in a *shared* ConfigMap consumed by all of them. The ConfigMap is mounted as env vars, so the change took effect as pods restarted on the rolling update that the config-change tooling triggered. Timeline of the unvalidated push: the change merges at T+0; the GitOps controller syncs it to all 40 services' namespaces within about 15 seconds; rolling restarts begin, and within 40 seconds enough pods have read the bad value that checkout threads start blocking on the 30,000-second timeout; by T+90s the order service's thread pools are saturated fleet-wide and checkout is fully down; the team spots it, identifies config as the cause (no deploy happened, so it takes a few minutes to even suspect config), reverts the commit at T+4m, and the fleet recovers by T+11m. **Eleven minutes of total checkout outage across 40 services from one typo, with peak revenue loss.** At ShopFast's scale of, say, 200 orders per minute at a \$60 average order value, eleven minutes of dead checkout is roughly 2,200 lost orders and on the order of \$130,000 of delayed-or-lost revenue, plus the reputational hit of a public outage. The figure below is that timeline.

![A six-event timeline showing a bad config push from the merged typo through fleet sync, rolling restart, thread-pool saturation, rollback start, and recovery at eleven minutes](/imgs/blogs/configuration-and-secrets-management-6.webp)

Now the validated, staged version. Validation-at-boot is on, so the first pod to read `30000` crash-loops with `DOWNSTREAM_TIMEOUT_MS=30000 out of allowed range [50,10000]`. The rolling update's first new pod never passes readiness, so the rollout *halts automatically* — Kubernetes won't proceed to the second pod while the first is unhealthy. **Blast radius: one pod, which is already replaced by its old-config sibling. Zero customer-facing impact.** And because we also staged the rollout (config change applied to a canary namespace first, then promoted), even the crash-loop is contained to the canary before any production pod sees it. The numbers: 11 minutes and \$130k versus zero. The same typo, the same human error — the difference is entirely in whether config got the rigor of code. This is the senior point made arithmetic: validation and staging are not bureaucracy; they are a six-figure insurance policy that costs a few lines of YAML and a boot-time check.

## Dynamic config and feature flags: power and peril

Some config genuinely must change at runtime, without a redeploy — and this is where a config service earns its keep, and also where the sharpest foot-guns live. A **feature flag** is the canonical example: a boolean (or richer rule) you flip to turn a feature on or off for some or all users, instantly, without shipping code. ShopFast wraps its new checkout flow in a flag so it can be enabled gradually and disabled instantly if something goes wrong. We discussed feature flags as a *deployment strategy* in the [blue-green, canary, and feature-flags post](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) — the ability to decouple "deploy the code" from "turn on the feature" is a superpower for safe releases. Here we look at the *configuration* side: how a service reads dynamic config safely, and what makes it dangerous.

Reading a feature flag at runtime looks like this — the value is fetched from the config service (often cached locally with a short TTL and refreshed in the background) rather than read once at boot:

```python
# checkout.py — read a dynamic flag at request time, with a safe default
from feature_flags import flags  # SDK that streams updates from the config service

def handle_checkout(user, cart):
    # Evaluated at REQUEST time, not boot time — flips take effect live.
    # The SDK keeps a local cache and a safe default if the config service is unreachable.
    if flags.is_enabled("new_checkout_flow", user=user, default=False):
        return new_checkout(user, cart)
    return legacy_checkout(user, cart)
```

Two safety properties are baked in here and both matter. First, the flag is evaluated *per request*, so flipping it in the config service takes effect on the very next request — that's the whole point of dynamic config. Second, and critically, there is a `default=False`: **if the config service is unreachable, the code falls back to a safe default rather than failing.** A dynamic-config dependency that can take your service down when it's unavailable has turned a convenience into a new single point of failure. The SDK should cache the last-known-good value and degrade gracefully — never block a request waiting on the config service, never crash if it's down. This is just [graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) applied to config.

Now the peril, which is the flip side of the power. **A live config change has no deploy pipeline in front of it.** When you flip a flag, there is no CI, no review (usually), no staged rollout unless the config service provides one — the change takes effect everywhere, instantly. That is exactly as dangerous as it sounds: an engineer flipping a flag during an incident, fat-fingering a percentage rollout from 5% to 50%, or enabling a feature that isn't actually ready, can cause an outage with *less* friction than a code deploy would have required. The mitigations are: **audit every live change** (who flipped what, when, and why — an unaudited live config change is a debugging nightmare during the incident it caused), **stage flag rollouts** (1% of users, then 10%, then 50%, then 100% — the config service should support this), **default to safe**, and **make flags expire** so you don't accumulate thousands of stale flags whose combinations no one understands. Treat the feature-flag platform as part of your production change surface, with the same access controls and audit you'd want on `kubectl apply`. The most dangerous config change is the one that felt too easy to make.

## Per-environment config and the drift problem

A fleet runs in multiple environments — dev, staging, prod, maybe several prod regions — and each needs its own config: different database hosts, different external endpoints, different scaling parameters. The natural way to manage this is a base config plus per-environment overlays, which is exactly what Kustomize (built into `kubectl`) or Helm values files give you:

```yaml
# base/configmap.yaml — shared defaults
data:
  LOG_LEVEL: "INFO"
  WORKER_CONCURRENCY: "16"
  DOWNSTREAM_TIMEOUT_MS: "800"
---
# overlays/prod/configmap-patch.yaml — prod-specific overrides
data:
  LOG_LEVEL: "WARN"              # quieter in prod
  WORKER_CONCURRENCY: "64"       # bigger in prod
  DB_HOST: "orders-db.shopfast-prod.svc.cluster.local"
```

The danger this structure fights is **config drift**: over time, environments diverge in ways nobody intended, usually because someone made a quick `kubectl edit` directly in prod during an incident and never reflected it back into Git. Now prod's actual config differs from what the repo says, staging differs from prod, and the next time you test a change in staging it passes — because staging has different config than prod — and then it breaks in prod. Drift is insidious because it's invisible until it bites, and it's the reason "it worked in staging" is the most dangerous sentence in distributed systems.

The defenses are GitOps discipline and active drift detection. **GitOps discipline** means the repo is the *only* way config changes — no `kubectl edit` in prod, ever; the GitOps controller (Argo CD, Flux) continuously reconciles the cluster to match the repo and will *revert* any out-of-band change, which both prevents drift and surfaces it (Argo CD shows an "OutOfSync" status the moment reality diverges from Git). **Drift detection** can also be a periodic job that diffs each environment's effective config against the others and against the repo, flagging unexpected divergence. The senior framing: **config drift is the entropy of a distributed system — it accumulates unless something actively pushes back.** GitOps reconciliation is that something. Make the repo the source of truth and the cluster a *projection* of it, never the other way around. We discussed designing systems for change in the [evolutionary architecture](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) post; drift control is the operational discipline that keeps that evolution coherent across environments.

## Rotation without downtime

Rotation is the practice everyone agrees with and few do well, and it has a subtle correctness requirement that, missed, turns rotation itself into an outage. The naive approach — "generate a new password, update it everywhere, done" — breaks because there is a window during which some services have the old credential and some have the new one, and if you revoke the old one before everyone has the new one, the laggards fail to authenticate. **Zero-downtime rotation requires an overlap period where both the old and new credentials are valid simultaneously.**

The correct sequence, which Vault's dynamic-secrets and AWS Secrets Manager's rotation both implement, is: (1) create the new credential while keeping the old one valid — the database now accepts *both*; (2) propagate the new credential to all consumers (via ESO's sync and the mounted-file update); (3) wait until you're confident every consumer has the new one and is using it; (4) only *then* revoke the old credential. During the overlap, any service still on the old credential keeps working, and any service that's moved to the new one also works — nobody fails. The figure below walks through this for ShopFast rotating its database password across forty services.

![A six-event timeline showing zero-downtime secret rotation where a new password is issued while both old and new are valid, then synced to pods, reloaded from the mounted file, adopted fleet-wide, and only then is the old password revoked](/imgs/blogs/configuration-and-secrets-management-8.webp)

The mounted-file injection from earlier is what makes step (2) restart-free. Because the secret is a file, not an env var, ESO updates the file in place when Vault rotates, and a reload-aware application picks up the new value without restarting. Here's that reload-on-rotation pattern — watch the file, reconnect with the new credential, and keep serving:

```go
// secret_reload.go — watch the mounted secret file and reconnect on rotation
package secrets

import (
	"os"
	"sync"
	"time"
)

type DBCreds struct {
	mu       sync.RWMutex
	password string
	path     string
}

func (d *DBCreds) Get() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.password
}

// Watch re-reads the mounted secret file periodically; on change, swaps the
// in-memory credential and triggers a pool reconnect. No process restart needed.
func (d *DBCreds) Watch(onChange func(newPassword string)) {
	ticker := time.NewTicker(30 * time.Second)
	for range ticker.C {
		b, err := os.ReadFile(d.path)
		if err != nil {
			continue // keep last-known-good; never crash on a transient read error
		}
		next := string(b)
		d.mu.RLock()
		changed := next != d.password
		d.mu.RUnlock()
		if changed {
			d.mu.Lock()
			d.password = next
			d.mu.Unlock()
			onChange(next) // e.g. rebuild the DB connection pool with new creds
		}
	}
}
```

The application establishes new connections with the new credential while existing connections (authenticated with the still-valid old credential) drain naturally; once the overlap window closes and the old credential is revoked, only new-credential connections remain. No request fails. The `continue` on a read error is deliberate: a transient failure to read the file must never crash the service or drop the last-known-good credential — degrade to the old value, retry next tick.

#### Worked example: a secret leaks — how fast can you rotate across 40 services?

This is the stress-test every team should run as a tabletop exercise, because the answer reveals whether your secret management is real or theater. Scenario: at 09:00 you discover that ShopFast's database password has leaked. The question is not "is it bad" (it is) — the question is **how many minutes until that credential is dead everywhere.**

Run the clock for a *mature* setup with Vault dynamic secrets, ESO, and mounted-file reload. You trigger an immediate rotation in Vault (T+0). Vault issues the new credential, keeping the old valid (overlap begins). ESO's `refreshInterval` is 1 minute, so within 60 seconds every namespace's Kubernetes Secret is updated (T+1m). Kubernetes propagates the updated file into the mounted volumes — the file-update propagation delay is typically under a minute (T+2m). The reload watcher's 30-second tick picks up the change and reconnects pools (T+2.5m). You confirm via the audit log that all 40 services are now authenticating with the new credential (T+4m). You revoke the old credential in Vault (T+5m). **Total time from discovery to the leaked credential being dead everywhere: about 5 minutes, with zero failed requests.** And note the leaked credential was *already* on a one-hour TTL, so even if you'd done nothing, it would have died within the hour.

Now run the same clock for an *immature* setup: a single static password, stored in a Kubernetes Secret committed to Git, injected as an env var into all 40 services. To rotate, you must: pick a new password, update it in the database, update the Git-committed Secret (and remember the old one is now in Git history forever), and — because it's an env var — **restart every pod of every service**, since env-var secrets can't be updated in place. Restarting 40 services means 40 rolling deploys, each taking a few minutes, with the additional hazard that during the rollout some pods have the old password and some have the new one, and if you changed the database password to *only* accept the new one, the old-password pods now fail. To avoid that you need the overlap discipline you didn't build. Realistically this is **30 to 60 minutes of careful, error-prone, request-dropping work**, and the leaked credential is fully live the entire time because there's no TTL. The contrast — 5 minutes hands-off versus an hour of risky manual rotation — is the entire case for investing in dynamic secrets and file-based injection *before* you need them. The senior takeaway: **the time to design your rotation story is not during the breach.** Run this tabletop exercise now, measure your real number, and if it's "we've never rotated and we're not sure we can," that's your most urgent infrastructure project.

## Stress-testing the design

Good distributed-systems thinking means asking what breaks under stress. Three scenarios specific to config and secrets, reasoned through.

**A bad config value crashes every pod on boot.** Suppose validation-at-boot is working *too* well, or a genuinely required value goes missing — and now a config change makes *every* pod fail to start, including healthy ones cycling through normal restarts. With validation, the rolling update halts at the first failed pod (good — blast radius one). But consider the nastier variant: the bad config is applied, *and then* an unrelated event (a node failure, an autoscale event) forces healthy pods to restart — they now read the bad config and fail too, because nothing protected the *already-running* pods from the bad config that was sitting in the ConfigMap waiting. This is why validate-at-boot must be paired with staged rollout: the bad config should never have reached the live ConfigMap that a restarting pod reads. Stage config to a canary, verify the canary boots, *then* promote. And keep the last-known-good config one revert away, because `kubectl rollout undo` on the config is your fastest recovery.

**Config drift between staging and prod.** A change tests clean in staging and breaks in prod. The root cause is almost always that staging's config differs from prod's in a way that masked the bug — a smaller timeout, a different feature-flag state, a different external endpoint. The defense is to make staging config as prod-like as possible (same shapes, same flag states for the thing under test) and to run the change through prod's *actual* config in the CI validation job, not just staging's. When drift is the suspect, the GitOps controller's diff between desired (Git) and actual (cluster) state is the first thing to check — an "OutOfSync" prod is drift you can see.

**The secret store is down when a service needs to start.** If you've adopted Vault dynamic secrets and Vault is unreachable when a pod boots, the pod can't get its database credential and can't start — Vault has become a hard dependency of every cold start. This is real and it's the price of dynamic secrets. The mitigations: run Vault highly available (it's a quorum system; lose a node, stay up), cache credentials with a TTL long enough to ride out a short Vault outage (the running pods keep working on their existing leases), and use the sidecar-agent pattern so credential renewal is decoupled from the application's request path. The trade-off is explicit and worth stating plainly: **dynamic secrets shrink your leak window but add a runtime dependency on the secret store.** A static sealed-secret has no such dependency (the secret is local) but never rotates. Which risk you prefer — a runtime dependency on a highly-available Vault, or a never-rotated static secret — is a real architectural decision, and the right answer depends on your threat model and your operational maturity.

## When to reach for which store

A decisive recommendation, because the trade-off matrices are useless without a verdict. The figure below is the decision tree — sensitivity decides the store before convenience does.

![A decision tree showing a new config value branching on whether it is non-secret, which goes to a ConfigMap or config service, or sensitive, which then branches on whether it needs rotation to choose between cloud Secrets Manager or sealed secrets for static values and Vault for dynamic short-lived values](/imgs/blogs/configuration-and-secrets-management-9.webp)

**For non-secret config:** ConfigMaps versioned in Git, by default, with per-environment overlays via Kustomize or Helm. Add a config service (feature-flag platform) only for the specific values you must change at runtime. Never bake config into the image.

**For secrets, scaled to your maturity:**

- **Just starting, small team, want GitOps:** Sealed Secrets. Encrypted blobs in Git, decrypted only in-cluster, no external dependency. Accept that you won't rotate automatically and plan a manual rotation cadence.
- **Already on a cloud, want managed encryption, audit, and rotation for common cases:** the cloud Secrets Manager (AWS/GCP/Azure), fronted by the External Secrets Operator so your apps consume normal Kubernetes Secrets. This is the pragmatic sweet spot for most teams.
- **Need dynamic short-lived credentials, multi-cloud, or the strongest audit and policy:** HashiCorp Vault, with the external-secrets operator or the Vault Agent sidecar. Accept the operational burden of running Vault HA. The payoff is the 5-minute rotation and the one-hour leak window from the worked examples.

**For everyone, regardless of store:** never commit a secret (gitignore plus pre-commit scan plus CI gate), split config from secrets, validate config at boot, version config in Git, stage config rollouts, and have a tested rotation runbook. Those practices are store-independent and they're where most of the safety actually comes from. The store is the foundation; the discipline is the building.

## Case studies

**The leaked-credential genre (Codecov, and the broader pattern).** Real-world secret leaks through CI and source repos are common enough to be a category rather than a single incident. The 2021 Codecov breach is a clean illustration of the blast radius: attackers modified Codecov's Bash uploader script, which ran in thousands of customers' CI pipelines, and exfiltrated *environment variables* from those pipelines — which is to say, the secrets that customers had injected as env vars into CI. Because so many teams put live credentials into CI environment variables, a single compromised tool harvested a wide field of customer secrets, and affected organizations then had to rotate everything those pipelines could touch. The lessons are direct and they're the spine of this post: secrets in CI environment variables are a high-value, broadly-exposed target; the static, long-lived credential is the one that hurts when (not if) it leaks; and the only real mitigation that scales is short-lived credentials that expire before an attacker can use them widely, plus the assumption that *any* secret your CI can read may eventually be exfiltrated. The thousands of accidental-secret-commits that automated scanners find in public repos every day are the same lesson at smaller scale: prevention layers (gitignore, pre-commit, CI scan) catch most, rotation bounds the rest.

**Vault dynamic secrets in production.** HashiCorp Vault's dynamic-secrets capability — generating short-lived, unique-per-consumer database and cloud credentials on demand and auto-revoking them — has been adopted by many large engineering organizations precisely to eliminate the long-lived static credential as a class of risk. The pattern that matters for our purposes: a service authenticates to Vault (often via its Kubernetes service account identity), Vault checks policy and issues a fresh database credential with a short TTL, the service uses it, and Vault revokes it automatically when the lease expires. There is no static database password anywhere for an attacker to steal, every credential is traceable to exactly one consumer for forensics, and rotation is continuous and invisible. The cost — the operational weight of running Vault HA and unsealing it — is real, which is why the recommendation is to grow into Vault when the leak-window reduction is worth the operational investment, not to start there on day one. The trade-off is the same one the case study teaches: you exchange a *storage* risk (a static secret sitting somewhere) for an *availability* dependency (Vault must be up), and dynamic secrets are worth it when your threat model makes the storage risk the bigger one.

**A global-config-push outage.** The shape of ShopFast's opening incident is drawn from a real and recurring class of production outage: a single bad value in a centrally-managed, fleet-wide configuration propagates faster than any human or pipeline can react, and takes down a wide swath of services that share that config. Public post-mortems across the industry repeatedly cite this pattern — a config change, not a code change, as the trigger; a shared config object as the amplifier; and the *absence* of validation and staged rollout for config as the reason a typo became an outage. The lesson the industry has converged on is exactly the one this post argues: config deserves the same change-management rigor as code, because it has the same (often larger) blast radius. Validate it at boot so a bad value fails one instance fast; roll it out staged like a canary so a bad value is caught before it's global; version it so you can revert in seconds; and treat the shared, fleet-wide config object with the respect its blast radius demands. The cheapest possible insurance — a range check at boot and a one-pod canary — would have prevented the most expensive config outages on record.

## Key takeaways

1. **Config is part of your deploy surface; secrets are part of your attack surface.** Both deserve the rigor you give code. The junior treats them as "just settings"; the senior treats a config change as a deploy with a blast radius.
2. **Strictly separate config from code, and config from secrets.** One immutable image, many deploys; config in the environment; non-secret config to ConfigMaps and a config service; credentials to a real secret store. The boundary is "would I care if this leaked?"
3. **Never commit a secret — layer the prevention.** `.gitignore`, a local pre-commit scanner, a CI scan gate, and periodic full-history scans. Any one layer fails sometimes; the stack rarely does. A secret in Git history is there forever.
4. **Pick the secret store by maturity:** sealed secrets for GitOps without rotation, cloud Secrets Manager for managed encryption and audit, Vault for dynamic short-lived credentials. Front them with the external-secrets operator so apps consume normal Kubernetes Secrets.
5. **Short-lived rotated credentials shrink the leak window from months to an hour.** Dynamic secrets exchange a storage risk for an availability dependency on the secret store — a trade-off worth making when leaks are your bigger threat. The credential you never rotated is the one already in someone else's hands.
6. **Mount secrets as files, not env vars, when you want to rotate.** Files update in place; env vars are frozen at process start. File-based injection plus a reload watcher gives you zero-downtime rotation without a restart.
7. **Validate config at boot — range-check, not just type-check.** A service must refuse to start on invalid config. This single practice converts a fleet-wide silent outage into one failed pod and a halted rollout. It's the highest-leverage defense in this post.
8. **Stage config rollouts and version them in Git.** A bad config push has the blast radius of a bad deploy; give it the same staged rollout, review, and one-command rollback. Canary one pod before you go global.
9. **Dynamic config is powerful and dangerous — audit every live change.** Feature flags flip with no pipeline in front of them; default to safe, stage flag rollouts, audit who changed what, and make flags expire. The most dangerous change is the one that felt too easy.
10. **Fight config drift with GitOps reconciliation.** Make the repo the source of truth and the cluster a projection of it; never `kubectl edit` prod. Drift is the entropy of a distributed system — something must actively push back, or "it worked in staging" becomes your next outage.

## Further reading

- *The Twelve-Factor App* (Adam Wiggins / Heroku) — Factor III "Config" is the canonical statement of config-in-the-environment and the open-source test.
- Sam Newman, *Building Microservices* (2nd ed.) — chapters on configuration and on the security of a service fleet.
- Chris Richardson, *Microservices Patterns* — the Externalized Configuration pattern and how services consume config.
- HashiCorp Vault documentation — the dynamic secrets engines (database, AWS, PKI) and the Kubernetes auth method are the parts that matter for a fleet.
- External Secrets Operator documentation — `ExternalSecret` and `ClusterSecretStore` for bridging Vault and cloud Secrets Managers into Kubernetes.
- Bitnami Sealed Secrets — the GitOps-friendly static-secret option when you don't yet need Vault.
- Sibling posts in this series: [containerizing microservices with Docker best practices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices), [Kubernetes for microservices, the essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials), [deployment strategies: blue-green, canary, and feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags), [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice), and the forward-looking [service-to-service security with mTLS and zero trust](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) and [authentication and authorization: OAuth2, JWT, and token propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation).
- For the change-over-time discipline that config drift control supports, see [evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change), and for the graceful-degradation mindset behind safe dynamic-config defaults, [handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation).

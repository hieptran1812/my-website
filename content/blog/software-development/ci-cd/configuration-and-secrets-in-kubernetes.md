---
title: "Configuration and Secrets in Kubernetes: Inject, Don't Bake"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Keep your image environment-agnostic, inject config at deploy time, and handle secrets so they never touch Git in plaintext — so the same digest runs cleanly in dev, staging, and prod."
tags:
  [
    "ci-cd",
    "devops",
    "kubernetes",
    "secrets",
    "configmap",
    "gitops",
    "sealed-secrets",
    "external-secrets",
    "twelve-factor",
    "supply-chain",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/configuration-and-secrets-in-kubernetes-1.png"
---

A team I worked with had a deploy ritual that everyone hated but nobody questioned. To ship to production, the CI pipeline would run `docker build` *again* — a fresh build, against the same commit it had already built and tested twenty minutes earlier in staging — but this time with `--build-arg ENV=prod`. The Dockerfile baked the production database URL, the log level, and a fistful of API endpoints straight into the image at build time. The reasoning, years earlier, had been "the prod image should *be* the prod image." It sounded responsible. It was a quiet disaster.

The disaster surfaced on a Tuesday. The staging build was green, fully tested, beautiful. The production build — same commit, same Dockerfile, a different `--build-arg` — pulled a slightly newer base image because someone had pushed a `python:3.11` tag upstream in the intervening eighteen minutes. The prod image now had a different OpenSSL than the one staging tested against, a TLS handshake against the payment provider broke in a way that only triggered under prod's certificate chain, and checkout went down for ninety minutes during the evening peak. The bytes you tested were not the bytes you shipped, because you built the prod bytes separately. That is the whole failure in one sentence.

And there was a second, slower-burning problem in that same repo. To get the production database password into the cluster, someone had committed a Kubernetes `Secret` manifest — a real one, with the password base64-encoded in the `data:` field — into the GitOps repository. They genuinely believed base64 was a form of encryption. It is not. It is `base64 -d`-away from plaintext, it sat in Git history forever, and when the repo was later mirrored to a contractor's read-only fork, the production database credential went with it. Nobody noticed for four months.

This post is about fixing both of those at the root. By the end you will be able to: keep your container image environment-agnostic so one digest runs everywhere (the foundation of "build once, promote everywhere"); inject per-environment config with ConfigMaps as either env vars or mounted files, and know which to reach for and why; understand exactly why a Kubernetes `Secret` is *not* encrypted and what that means for Git; and stand up real, GitOps-compatible secret handling with Sealed Secrets, the External Secrets Operator, the CSI secrets driver, and etcd encryption-at-rest — including rotation that does not require a commit. We will tie all of it back to the series' spine, commit → build → test → package → deploy → operate, and to the two principles that govern it: build once, promote everywhere, and everything as code. If you want the map of the whole journey first, read [the CI/CD overview that frames this series](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model); for why the *artifact* must be immutable, read [build once, promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning), which this post is the operational other half of.

![Diagram contrasting an image with configuration baked in requiring three separate builds against a single immutable image with configuration injected at deploy time](/imgs/blogs/configuration-and-secrets-in-kubernetes-1.png)

## 1. The principle: config lives outside the artifact

Let me start with the rule and then earn it, because the rule is the load-bearing idea for everything else: **anything that varies between environments belongs in the environment, not the build.**

That is the heart of the config factor in the [Twelve-Factor App](https://12factor.net/config), and it is the operational consequence of build-once-promote-everywhere. Twelve-factor draws the line cleanly: config is "everything that is likely to vary between deploys (staging, production, developer environments, and so on)." Concretely, in a typical web service, config is the things in this list:

- Database and cache connection URLs (`postgres://db.prod.internal:5432/app`)
- External service endpoints (the payment gateway base URL, the auth provider issuer)
- Feature flags and rollout toggles (`ENABLE_NEW_CHECKOUT=true` in staging, `false` in prod for now)
- Log level and verbosity (`debug` in dev, `info` in staging, `warn` in prod)
- Resource tuning that differs by environment (connection pool size, worker count)
- Credentials and secrets (which deserve their own treatment — sections 5 onward)

The application *code* is the same across all environments. The *config* is what makes a deploy be "dev" or "prod." Put plainly: the same compiled artifact behaves differently purely because of values handed to it at startup — that is the whole of twelve-factor config. (There is a figure two paragraphs down that shows exactly where those values enter.)

### Why baking config in breaks build-once

Here is the chain of reasoning, made provable. Suppose you have $N$ environments and you bake config into the image. Then you need $N$ distinct images, one per environment, because each carries different baked values. But each of those $N$ images is a *separate build*. A separate build is a separate opportunity for drift: a base image that moved, a dependency that resolved to a newer patch, a transient registry that served a slightly different layer, a build cache that was warm for one and cold for the next. The number of "did the bytes change?" risk events scales with the number of builds, which scales with $N$.

Now do it the other way. Build *one* image, content-addressed by its digest `sha256:…`, and inject config at deploy time. The number of builds is $1$, independent of $N$. The image you tested in staging is — byte for byte, digest for digest — the image you run in prod. The drift surface for "the artifact changed between environments" collapses to zero, because there is exactly one artifact.

> The arithmetic: with baked config, your "untested-bytes-in-prod" risk is proportional to the number of environments, because each environment is its own build. With injected config, that risk is a constant — exactly one build, promoted unchanged. Going from $N$ builds to $1$ build is the single biggest reduction in deploy variance you can make for free.

That is why this post is the operational completion of [build once, promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning). Immutable artifacts are necessary but not sufficient. An immutable artifact with config baked in is still pinned to one environment — it is immutably *wrong* for the other two. The artifact has to be immutable *and* environment-agnostic. Config injection is what makes it agnostic.

![Layered stack diagram showing the immutable image at the base with ConfigMaps, Secret objects, and external secret sources entering from outside and binding into the running container](/imgs/blogs/configuration-and-secrets-in-kubernetes-2.png)

The figure above is the layering to hold in your head for the rest of this post. At the bottom is the one immutable image — no config, no secrets, just code and its runtime. Above it sit the per-environment ingredients: a ConfigMap for non-secret config, a Secret object for credentials (which, crucially, should be *materialized* in the cluster rather than committed in Git — we will spend most of this post on that distinction), and an external source of truth like HashiCorp Vault or a cloud key manager for the most sensitive values. At deploy time those bind into the Pod as environment variables and mounted volumes, and the same image becomes a dev, staging, or prod workload depending purely on what got bound. Kubernetes gives you the primitives to do this; the rest of the post is about using them correctly, because getting secrets wrong is where delivery setups quietly leak credentials.

## 2. ConfigMaps: the non-secret config primitive

A ConfigMap is a Kubernetes object that holds non-secret configuration as key-value pairs. That is the whole definition. It is deliberately boring, which is good — boring is what you want for the thing that varies thirty times a day. Here is one for a hypothetical `checkout` service:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: checkout-config
  namespace: prod
data:
  LOG_LEVEL: "warn"
  DATABASE_HOST: "db.prod.internal"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "checkout"
  PAYMENT_GATEWAY_URL: "https://api.payments.example.com"
  ENABLE_NEW_CHECKOUT: "false"
  WORKER_CONCURRENCY: "8"
  # A whole structured file can live here too:
  app-settings.yaml: |
    cache:
      ttl_seconds: 300
      max_entries: 50000
    timeouts:
      upstream_ms: 2000
      total_ms: 5000
```

Two things to notice. First, simple flat keys (`LOG_LEVEL`, `DATABASE_HOST`) sit alongside a whole embedded YAML document (`app-settings.yaml`). A ConfigMap can hold both — flat values for env vars and entire file blobs for mounting. Second, there is nothing secret here. `DATABASE_HOST` is not a secret; the *password* is. Keep that line bright: hostnames, ports, URLs, flags, and tuning go in a ConfigMap; passwords, tokens, and keys do not.

You inject a ConfigMap into a Pod two ways, and the choice matters more than people expect.

### Injection as environment variables

The simplest form. You map ConfigMap keys to env vars in the container spec:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
  namespace: prod
spec:
  replicas: 3
  selector:
    matchLabels: { app: checkout }
  template:
    metadata:
      labels: { app: checkout }
    spec:
      containers:
        - name: checkout
          image: ghcr.io/example/checkout@sha256:9f2c...  # digest-pinned, env-agnostic
          envFrom:
            - configMapRef:
                name: checkout-config        # pulls every key as an env var
          env:
            - name: LOG_LEVEL                  # or cherry-pick individual keys
              valueFrom:
                configMapKeyRef:
                  name: checkout-config
                  key: LOG_LEVEL
          ports:
            - containerPort: 8080
```

`envFrom` pulls every key in the ConfigMap into the container's environment. `valueFrom.configMapKeyRef` cherry-picks one key. Both are common; `envFrom` is convenient but indiscriminate, and it will happily inject that embedded `app-settings.yaml` blob as a giant, useless env var, so most teams use explicit `env` entries for clarity.

Env vars are the right reach for the twelve-factor purist and for small, flat config. They are universally understood — every language reads `os.environ` / `process.env` / `System.getenv`. But they carry three real costs:

1. **They are static at process start.** A container reads its environment exactly once, when the process spawns. Change the ConfigMap and the running Pod keeps the old values until it is restarted. There is no hot-reload of an env var.
2. **They leak.** The full environment of a process is visible at `/proc/<pid>/environ` to anything in the same Pod or with node access, it gets captured in crash dumps and stack-trace reporters that helpfully serialize the environment, and it tends to end up in logs the first time someone debug-prints `process.env`. For a database *host* that is fine. This is exactly why secrets-as-env-vars is a sharper risk than config-as-env-vars.
3. **They are flat.** A key-value string is awkward for genuinely structured config. You end up either with `DATABASE__POOL__MAX=20`-style flattening conventions or with a single env var holding a JSON blob, both of which are uncomfortable.

### Injection as a mounted volume (config-as-a-volume)

The other pattern mounts the ConfigMap as files inside the container:

```yaml
spec:
  template:
    spec:
      containers:
        - name: checkout
          image: ghcr.io/example/checkout@sha256:9f2c...
          volumeMounts:
            - name: config
              mountPath: /etc/checkout      # files appear here
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: checkout-config
            items:
              - key: app-settings.yaml      # mount just this key as a file
                path: app-settings.yaml
            defaultMode: 0444               # read-only for everyone
```

Now `/etc/checkout/app-settings.yaml` is a file the app reads at startup *and can re-read later*. This unlocks the property env vars cannot give you: **hot reload**. The kubelet syncs ConfigMap changes into the mounted volume (on a sync interval, typically tens of seconds), and a watching application — or a sidecar that signals `SIGHUP` — can pick up the new config without a restart. The volume form also handles structured and large config natively (it is a file, so it can be a 200-line YAML), and the file's permissions are explicit (`0444` here, and for secrets you would tighten to `0400`).

The trade-off table:

| Concern | Env vars (`envFrom` / `valueFrom`) | Mounted volume |
| --- | --- | --- |
| Simplicity | Highest — every language reads env | One extra mount + a file read |
| Reload | Static at start; needs a Pod restart | Hot-reload possible (kubelet syncs the file) |
| Leak surface | High — `/proc/<pid>/environ`, crash dumps, logs | Lower — a file with explicit perms (`0400`) |
| Structured config | Awkward (flatten or JSON-in-a-var) | Native (mount a whole YAML/JSON file) |
| Large config | Bad fit (env size limits, ugliness) | Designed for it |
| Best for | Small, flat, restart-on-change config | Structured, large, or hot-reloadable config |

![Matrix comparing environment variables and mounted files across reload behavior, leak surface, and structured-config support](/imgs/blogs/configuration-and-secrets-in-kubernetes-3.png)

As the matrix above lays out, the decision is not "one is better." It is: reach for env vars when config is small, flat, and a restart-on-change is acceptable (the common case — most config changes *want* a deliberate rollout anyway); reach for a mounted volume when you have structured config, large config, or a genuine hot-reload requirement. A useful default for most services is env vars for the handful of simple knobs plus a mounted file for any structured settings document. We will come back to the "config change needs a restart" wrinkle in section 9, because it is subtle and it bites people.

### A note on the `immutable` ConfigMap and the rollout signal

There is a subtlety in how a changed ConfigMap interacts with a Deployment that is worth making explicit, because it is the seam between "I edited config" and "the change took effect." A vanilla `kubectl apply` of a changed ConfigMap does *not*, by itself, restart any Pods. The Deployment's Pod template did not change, so the Deployment controller sees no reason to roll. Your Pods keep running with the values they read at start. This is exactly the static-at-start problem, and it is why so many "I changed the config but nothing happened" tickets exist.

There are three ways teams force the change to take effect, and naming them now sets up section 10:

```yaml
# Option A — name the ConfigMap with a content hash so a change
# rolls the Deployment. Kustomize's configMapGenerator does this for you:
#   configmap name becomes checkout-config-<hash>, the Deployment ref
#   updates, the template changes, and a rolling update happens.
configMapGenerator:
  - name: checkout-config
    literals: [LOG_LEVEL=warn]
# Option B — set a checksum annotation on the Pod template (Helm pattern):
#   annotations:
#     checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
# Option C — mark the ConfigMap immutable so the kubelet stops watching it
#   (faster, less API load) and you roll a NEW ConfigMap name on change:
immutable: true
```

The `immutable: true` field deserves a word. Marking a ConfigMap (or Secret) immutable tells the kubelet it will never change, so the kubelet stops watching it for updates — which reduces load on the API server at scale (thousands of Pods each watching their ConfigMaps is real traffic) and protects against accidental edits. The trade is that you can no longer edit it in place: to change it you create a *new* ConfigMap (with a new name, typically content-hashed) and point the Deployment at it, which rolls the Pods. That is a feature for production config — it forces every change through a fresh, named, rolling deploy, which is observable and revertible. The Kustomize `configMapGenerator` (option A) gives you content-hashed names automatically and pairs beautifully with immutability: every config change produces a new hash-suffixed ConfigMap, the Deployment's reference updates, and a clean rolling update carries the new config — no separate reloader needed.

#### Worked example: the env-var-only service that couldn't change a flag

A team ran their feature flags as env vars from a ConfigMap. Marketing wanted to flip `ENABLE_PROMO_BANNER` from `false` to `true` for a campaign at noon. An engineer edited the ConfigMap at 11:58. Noon came. Nothing happened. The banner did not appear, because the running Pods had read `ENABLE_PROMO_BANNER=false` at startup hours earlier and an env var does not change under a running process. The fix that day was a `kubectl rollout restart deployment/web`, which cycled the Pods and picked up the new value — but that rollout took four minutes and briefly halved capacity during a traffic spike. Had the flag been a key in a *mounted* config file with a watching reload, the flip would have taken effect in seconds with no restart. The cost difference: a four-minute capacity dip during peak vs. an instant, free change. That is the env-var "static at start" tax, paid at the worst possible moment.

## 3. Secrets: why base64 is not encryption

Now the hard part, and the part everyone gets wrong at least once. A Kubernetes `Secret` looks like a more secure ConfigMap. It has a `data:` field, it shows up redacted in `kubectl get`, the docs talk about it in a hushed tone. The trap is this: **the `data:` field of a Secret is base64-*encoded*, not encrypted.** Base64 is a reversible transport encoding — it exists so binary data survives a YAML/JSON text field, nothing more. Here is a Secret, and here is how trivially it decodes:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: checkout-db
  namespace: prod
type: Opaque
data:
  # This is NOT encrypted. It is base64. Anyone can decode it.
  DATABASE_PASSWORD: c3VwZXItc2VjcmV0LXBhc3N3b3Jk
```

```bash
echo "c3VwZXItc2VjcmV0LXBhc3N3b3Jk" | base64 -d
# super-secret-password
```

One command. No key, no crack, no effort. The base64 is there so the password (which might contain newlines or binary bytes) survives the YAML field — it provides exactly zero confidentiality. This is the number-one secrets misconception in Kubernetes, and it has a specific, expensive consequence:

> **The cardinal rule: never commit a plaintext Secret to Git.** A Secret manifest in Git is a plaintext credential leak. Not "a slightly risky thing." A leak. Base64 does not change that.

It is worth dwelling on *why* a Git commit specifically is so bad, because "I'll just remove it later" is the instinct and it is wrong. Three reasons:

- **Git history is forever.** Deleting the file in a later commit does not remove it from history. `git log -p`, `git show <old-sha>`, or a clone of any branch that ever contained it still has the value. Truly purging it requires history rewriting (`git filter-repo`) across every clone, fork, and mirror — which in practice means you can't, so you must rotate the credential instead.
- **The blast radius is everyone with read access — now and forever.** Every current collaborator, every future hire who clones the repo, every CI system with a checkout, every backup, every fork. A secret in a repo is a secret shared with the union of everyone who will *ever* touch that repo.
- **Public exposure is one misconfiguration away.** A private repo made public by accident, a fork pushed to a personal account, a mirror to a contractor — and now the credential is on the open internet, where automated scanners find exposed secrets within *minutes*.

![Flow diagram tracing how a base64 Secret committed to Git becomes a decodable credential preserved in history forever and detected by scanners](/imgs/blogs/configuration-and-secrets-in-kubernetes-4.png)

The figure traces the leak path: a Secret YAML with a base64 password gets committed and pushed; from there it forks two ways. One branch is that `base64 -d` decodes it in a single command. The other is that Git history retains it permanently even if you `rm` the file. Both converge on the same place: a credential breach where the only real remediation is to rotate everything the credential touched. A scanner like gitleaks or trufflehog may catch it on push — which is the good outcome, because it means you rotate proactively rather than reactively. We will set those scanners up in section 8.

### What a Secret *does* give you (and what it doesn't)

A Secret is not useless — it is the right *destination* for a credential inside the cluster. Compared to a ConfigMap, a Secret:

- Is stored separately and can be RBAC-restricted independently (you can grant a service account read on ConfigMaps but not Secrets).
- Is mounted into `tmpfs` (memory-backed) rather than written to disk on the node when used as a volume.
- Is redacted in `kubectl get -o yaml` output by default and excluded from some logging.
- Can have its etcd storage *encrypted at rest* if you configure it (next section).

So the Secret *object* is fine and correct — the destination. The problem is never the object; it is **how the value gets into the object**. The entire art of secrets management in Kubernetes is: get the real value into a Secret object in the cluster without the plaintext ever passing through Git or a developer's clipboard. Sections 6 and 7 are the two production-grade ways to do that. But first, the cluster-side hardening that every one of those patterns assumes.

## 4. Hardening the cluster: encryption at rest and RBAC

Even if no secret ever touches Git, by default a Secret's value sits in etcd — the cluster's key-value store — in plaintext (well, base64, which we have established is plaintext). Anyone who can read etcd, or who walks off with an etcd backup, reads your secrets. Two cluster-level controls close this.

### Encryption at rest (etcd encryption)

You configure the API server with an `EncryptionConfiguration` so that `Secret` resources are encrypted before they are written to etcd:

```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      # Prefer a KMS provider (envelope encryption with a cloud/HSM key):
      - kms:
          apiVersion: v2
          name: cloud-kms
          endpoint: unix:///var/run/kmsplugin/socket.sock
          timeout: 3s
      # Fallback so existing unencrypted secrets can still be read during rollout:
      - identity: {}
```

The API server is launched with `--encryption-provider-config=/etc/kubernetes/enc/enc.yaml`. With the `kms` v2 provider, Kubernetes does *envelope encryption*: it generates a data-encryption key (DEK) to encrypt the Secret, then encrypts the DEK with a key-encryption key (KEK) held in a cloud KMS or HSM that never leaves it. The plaintext secret is never written to etcd. (You can also use a local `aescbc` provider with a key in a file, but then the key sits on the control-plane node — KMS is strongly preferred because the master key lives in dedicated hardware.)

Two operational notes that trip people up. First, ordering matters: the *first* provider in the list is used to encrypt new writes; `identity: {}` (no encryption) must come *after* a real provider, only so the API server can still decrypt secrets written before you enabled encryption. Second, enabling encryption does not retroactively encrypt existing secrets — you have to re-write them: `kubectl get secrets --all-namespaces -o json | kubectl replace -f -` forces every Secret through the new encryption provider.

### RBAC on secrets

Encryption at rest protects the *storage*. RBAC protects the *access*. The principle is least privilege: a workload's service account should be able to read only the secrets it needs, and humans should rarely have blanket secret-read.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: checkout-secret-reader
  namespace: prod
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["checkout-db"]   # this ONE secret, not all of them
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: checkout-secret-reader
  namespace: prod
subjects:
  - kind: ServiceAccount
    name: checkout
    namespace: prod
roleRef:
  kind: Role
  name: checkout-secret-reader
  apiGroup: rbac.authorization.k8s.io
```

The `resourceNames` field is the part most teams skip and shouldn't: it scopes the grant to a single named Secret rather than "all secrets in the namespace." A common audit finding is a service account with `get`/`list` on `secrets` cluster-wide — which means a single compromised pod can exfiltrate every credential in the cluster. Scope it down. And remember that `list` and `watch` on secrets are nearly as powerful as `get` on a specific one, because a list returns the values; grant them sparingly.

### Don't forget the pipeline's own credentials

There is a secret most teams overlook precisely because it is not *in* the cluster: the credential the pipeline uses to talk to the cluster and the cloud. The old way was a long-lived service-account token or a cloud access key, pasted into the CI system as a secret variable — and that static key is a fat, permanent target (recall the Codecov story in section 8, where exactly these CI-resident credentials were harvested). The modern fix is **OIDC keyless federation**: the CI provider issues a short-lived, signed identity token for the specific workflow run, and the cloud (or cluster) is configured to *trust* that issuer for a scoped role — so no static key exists to leak.

```yaml
# GitHub Actions assuming a scoped AWS role via OIDC — no static keys in the repo.
permissions:
  id-token: write          # let the runner request an OIDC token
  contents: read
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/deploy-prod
          aws-region: eu-west-1
          # No aws-access-key-id / aws-secret-access-key here.
          # The runner exchanges its OIDC token for short-lived STS creds.
```

The trust policy on the AWS side restricts the role to a specific repo, branch, and even environment (`token.actions.githubusercontent.com:sub` must equal `repo:org/repo:environment:prod`), so a token minted for a feature branch cannot assume the prod role. The credentials live for minutes and never touch Git or the CI secret store. This is the same principle as ESO's "the operator authenticates *as* the cluster" — identity over static keys — applied to the pipeline's edge. It belongs to the broader pipeline-secrets topic, which the planned [secrets management in the pipeline](/blog/software-development/ci-cd/secrets-management-in-the-pipeline) post in this series covers end to end; here the point is that the cluster's own secrets discipline is undermined if the *pipeline* that deploys to it holds a static god-key.

These three controls — encryption at rest, tight RBAC, and OIDC instead of static keys at the pipeline edge — are the *floor*. They make the cluster a safe place to *hold* a secret and a safe thing to deploy to. They do nothing about how the secret value gets into the cluster, which is still the Git problem. So now: the two real patterns.

## 5. Sealed Secrets: encrypt so the blob is safe to commit

The first GitOps-native pattern keeps the source of truth in Git but makes the committed thing safe. [Bitnami Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) works on a beautifully simple asymmetric-crypto idea: a controller in the cluster holds a private key; it publishes the matching public key; you encrypt your secret *with the public key* on your laptop, producing a `SealedSecret` custom resource. Only the controller's private key can decrypt it. The encrypted blob is therefore safe to commit to Git — anyone can read it, nobody but the in-cluster controller can decrypt it.

You use the `kubeseal` CLI:

```bash
# 1. Create a normal Secret manifest locally (do NOT commit this).
kubectl create secret generic checkout-db \
  --namespace prod \
  --from-literal=DATABASE_PASSWORD='super-secret-password' \
  --dry-run=client -o yaml > /tmp/secret.yaml

# 2. Seal it with the cluster's public key -> a SealedSecret you CAN commit.
kubeseal --controller-namespace kube-system \
  --format yaml < /tmp/secret.yaml > checkout-db-sealed.yaml

# 3. Shred the plaintext; commit only the sealed version.
shred -u /tmp/secret.yaml
git add checkout-db-sealed.yaml && git commit -m "add sealed checkout-db secret"
```

The committed `checkout-db-sealed.yaml` looks like this — the value is an opaque ciphertext, not base64:

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: checkout-db
  namespace: prod
spec:
  encryptedData:
    DATABASE_PASSWORD: AgBy3i4OJSWK+PiTy...   # RSA-OAEP + AES-GCM, decryptable only in-cluster
  template:
    metadata:
      name: checkout-db
      namespace: prod
    type: Opaque
```

When your GitOps tool (Argo CD or Flux) applies that `SealedSecret`, the in-cluster controller decrypts `encryptedData` and creates a normal `Secret` named `checkout-db` that your Deployment consumes exactly as it would any Secret. The flow:

![Diagram of the Sealed Secrets workflow from local plaintext through kubeseal encryption to a committable blob decrypted only by the in-cluster controller](/imgs/blogs/configuration-and-secrets-in-kubernetes-5.png)

The figure walks the path: the plaintext exists only locally; `kubeseal` encrypts it with the cluster public key into a `SealedSecret` blob that is safe to commit; that blob is pushed and reviewed in the GitOps repo like any other manifest; the controller, holding the private key in-cluster, decrypts it; and the result is a normal Secret object mounted into the Pod. The plaintext value never enters Git — only the sealed ciphertext does.

A few sharp edges to know before you adopt it:

- **The sealing is scoped.** By default a `SealedSecret` is encrypted for a specific namespace and name (`strict` scope). You cannot move it to another namespace or rename it without re-sealing — which is a feature (it stops a sealed secret being reused elsewhere) but surprises people. There are looser `namespace-wide` and `cluster-wide` scopes if you genuinely need portability.
- **The controller's private key is the crown jewel.** If you lose it, every `SealedSecret` becomes undecryptable and you must re-seal everything with a new key. So you back up the sealing key (it is itself a Secret in `kube-system`) — securely, *not* in the same Git repo. The controller also rotates its key periodically and keeps old keys to decrypt old blobs.
- **Rotation of the *secret value* still means a `kubeseal` + commit.** Changing the password means re-sealing and pushing a new commit. That is fine for GitOps purity, but it does mean the secret's rotation cadence is coupled to your Git workflow. If you want rotation to be a pure operation in a vault with no commit, that is the next pattern's job.

Sealed Secrets is the right reach when you want the source of truth to *be* Git (full GitOps, the repo is the complete declarative state of the cluster), you do not already run a secrets manager, and your rotation cadence is comfortable being a commit. It is the lowest-infrastructure way to do secrets-in-GitOps safely.

## 6. External Secrets Operator: the value never enters Git at all

The second pattern moves the source of truth *out* of Git entirely. The credential lives in a dedicated secrets manager — HashiCorp Vault, AWS Secrets Manager, Google Secret Manager, Azure Key Vault — and the [External Secrets Operator](https://external-secrets.io/) (ESO) syncs it into a Kubernetes Secret at runtime. Git holds only a *reference* — a pointer that says "the value for `DATABASE_PASSWORD` lives at this path in Vault" — and never the value itself.

You declare two custom resources. A `SecretStore` (or `ClusterSecretStore`) says how to authenticate to the backend:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: prod
spec:
  provider:
    vault:
      server: "https://vault.internal:8200"
      path: "secret"
      version: "v2"
      auth:
        # The operator authenticates AS the cluster, using the Pod's
        # Kubernetes service-account token -> Vault. No static creds in Git.
        kubernetes:
          mountPath: "kubernetes"
          role: "checkout"
          serviceAccountRef:
            name: "checkout"
```

And an `ExternalSecret` says which keys to pull and what Kubernetes Secret to materialize:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: checkout-db
  namespace: prod
spec:
  refreshInterval: 1h               # re-sync from Vault every hour
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: checkout-db               # the k8s Secret ESO will create/update
    creationPolicy: Owner
  data:
    - secretKey: DATABASE_PASSWORD   # key in the resulting k8s Secret
      remoteRef:
        key: prod/checkout/db        # path in Vault
        property: password           # field at that path
```

Both of these are safe to commit to Git, because neither contains the value — only the *location* of the value (`prod/checkout/db`) and how to authenticate (the service account). The operator, running in-cluster, authenticates to Vault using the Pod's Kubernetes service-account token (so there is no static Vault token sitting around either), fetches the real value, and creates/updates the `checkout-db` Secret. Your Deployment consumes `checkout-db` like any other Secret.

![Diagram showing the External Secrets Operator syncing a value from Vault into a Kubernetes Secret while Git holds only a reference and rotation happens in Vault](/imgs/blogs/configuration-and-secrets-in-kubernetes-6.png)

As the figure shows, the value's only home is Vault (the source of truth). The `SecretStore` carries auth and endpoint; the `ExternalSecret` references a key but no value; the operator syncs on its refresh interval, pulling the live value from Vault into a freshly created Kubernetes Secret. The payoff sits at the bottom: rotation is a pure Vault operation — change the value in Vault and the operator re-syncs on its next interval, with no commit, no PR, no Git involvement at all. This is the property Sealed Secrets cannot give you.

### The CSI Secrets Store driver: mount straight from the vault

A close cousin worth knowing is the [Secrets Store CSI Driver](https://secrets-store-csi-driver.sigs.k8s.io/), which mounts secrets from an external store *directly into the Pod as files* via a CSI volume — optionally also syncing them into a Kubernetes Secret. Instead of an `ExternalSecret`, you write a `SecretProviderClass` and reference it as a volume:

```yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: checkout-vault
  namespace: prod
spec:
  provider: vault
  parameters:
    roleName: "checkout"
    vaultAddress: "https://vault.internal:8200"
    objects: |
      - objectName: "DATABASE_PASSWORD"
        secretPath: "secret/data/prod/checkout/db"
        secretKey: "password"
---
# In the Pod spec:
#   volumes:
#   - name: secrets
#     csi:
#       driver: secrets-store.csi.x-k8s.io
#       readOnly: true
#       volumeAttributes:
#         secretProviderClass: "checkout-vault"
```

The distinction from ESO: the CSI driver mounts the value as a file inside the Pod at request time and can avoid creating a long-lived Kubernetes Secret object at all (which means there is no Secret in etcd for an attacker to read — the value lives only in the Pod's `tmpfs` while it runs). ESO materializes a standard Secret object, which is more compatible with everything that already consumes Secrets (e.g. `envFrom`) but does leave the value in etcd (encrypted at rest, if you did section 4). Many teams run both: ESO for broad compatibility, CSI for the most sensitive workloads where you want the value to never become a persistent Secret object.

### Secret rotation, properly

Rotation is where these patterns earn their keep. The rule to hold onto: **rotate in the source of truth; let the sync re-materialize; make the app reload.** With ESO or CSI, you change the value in Vault (or AWS Secrets Manager rotates it automatically on a schedule, which the managed services do natively). The operator picks up the new value on its `refreshInterval` and updates the Kubernetes Secret. But there is one more link in the chain that bites people: **the application still has the old value loaded.** A Secret mounted as env vars will *not* see the new value until the Pod restarts (same static-at-start problem as ConfigMaps). A Secret mounted as a volume *will* get the new file, but the app has to re-read it. The complete rotation therefore needs one of:

- A volume mount plus an app that re-reads the secret file (a watching connection-pool, for instance, that re-authenticates on the next failed connection).
- A controller like [Stakater Reloader](https://github.com/stakater/Reloader) that watches Secrets/ConfigMaps and triggers a `rollout restart` of the consuming Deployment when they change — turning "the secret rotated" into "a fresh, zero-downtime rolling deploy with the new value."

A standard production setup is ESO syncing from Vault on a one-hour interval, Reloader watching the resulting Secret, and a rolling update kicking off automatically when the value changes — so a rotation in Vault becomes a hands-off rolling redeploy with the new credential, no commit and no human in the loop.

## 7. Choosing a pattern: a decision you can defend

Put the three real options side by side. The choice turns on one question first — *where should the source of truth live?* — and then on rotation cadence and existing infrastructure.

| Dimension | Sealed Secrets | External Secrets Operator | CSI Secrets Store |
| --- | --- | --- | --- |
| Source of truth | Git (encrypted blob) | External vault | External vault |
| In Git you commit | Encrypted ciphertext | A reference only | A reference only |
| Value in etcd? | Yes (decrypted Secret) | Yes (synced Secret) | Optional — can stay only in Pod tmpfs |
| Rotation | `kubeseal` + commit | Pure vault op, auto re-sync | Pure vault op, re-mount |
| Infra needed | One controller, no vault | A secrets manager + operator | A secrets manager + driver |
| GitOps fit | Excellent — repo is full state | Excellent — refs are declarative | Good — class is declarative |
| Best when | No vault yet; Git is truth | You already run Vault/cloud SM | Most sensitive; avoid etcd copy |

![Decision tree for choosing a Kubernetes secrets pattern based on whether the source of truth is Git or an external vault](/imgs/blogs/configuration-and-secrets-in-kubernetes-7.png)

The decision tree above splits first on source of truth. If you want truth to live in Git (full GitOps, no separate secrets manager to operate), the branch leads to Sealed Secrets — or to [SOPS](https://github.com/getsops/sops) with `age` or a cloud KMS key, an alternative that encrypts the secret *file* in place so the encrypted file is committable and a tool like Flux's `kustomize-controller` or Argo CD plugin decrypts it on apply. (SOPS and Sealed Secrets occupy the same "encrypted-in-Git" niche; SOPS encrypts files with KMS/age and is decoupled from any controller, Sealed Secrets uses its own controller. Pick one.) If you want truth in a dedicated vault — which you do once you have a real secrets manager, compliance requirements, or a need for automatic rotation — the branch leads to ESO (broad compatibility) or the CSI driver (most sensitive, keep the value out of etcd).

My default recommendation, stated plainly: **a small-to-mid team doing GitOps with no existing vault should start with Sealed Secrets** — it is the least infrastructure and it is correct. **A team that already operates Vault or is all-in on a cloud secrets manager should use the External Secrets Operator** — the source of truth belongs in the system built for it, and rotation-without-a-commit is worth a lot. Reach for the CSI driver specifically for the workloads where you do not want the value to ever persist as a Secret object. Do not run all three for the sake of it; pick the one that matches where your secrets *should* live.

## 8. The secret-in-the-manifest mistake: detection and the war stories

The patterns above prevent the leak. You also need a net that catches the mistake when someone bypasses them — because someone will. The net is automated secret scanning, run as a pre-commit hook *and* in CI, so a plaintext secret is blocked before it lands and caught if it slips through.

The three tools you will hear named:

- **gitleaks** — fast, configurable regex + entropy scanner; great as a pre-commit hook and a CI gate. Scans the working tree *and* full history.
- **trufflehog** — entropy- and verifier-based; notably it can *verify* a found credential (actually try it against the provider) so you know whether a hit is live or a false positive.
- **git-secrets** (AWS) — a focused pre-commit hook that blocks commits matching credential patterns (originally AWS keys).

A CI gate with gitleaks is a few lines:

```yaml
# .github/workflows/secret-scan.yml
name: secret-scan
on: [push, pull_request]
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # scan full history, not just the tip
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
```

And the pre-commit hook, which is where you *want* to catch it (before it is ever in history):

```bash
# Install the hook once per clone:
gitleaks protect --staged --verbose
# Or wire it via the pre-commit framework in .pre-commit-config.yaml:
#   - repo: https://github.com/gitleaks/gitleaks
#     rev: v8.18.0
#     hooks: [{ id: gitleaks }]
```

The pre-commit hook stops the leak before it exists; the CI gate is the backstop for clones that skipped the hook. You want both, because the pre-commit hook is advisory (a developer can `--no-verify` past it) and the CI gate is mandatory (it fails the PR). For the version-control mechanics of pre-commit hooks and history rewriting, the [version control](/blog/software-development/version-control/) posts go deeper; here the point is that scanning is a *delivery* control — it belongs in the pipeline.

### War story: when the secret was already public

These are not hypotheticals. The supply-chain era is littered with leaked-credential incidents, and the pattern is always the same: a secret that should never have been reachable became reachable.

The **Codecov bash-uploader incident (2021)** is the canonical delivery-pipeline lesson. Attackers altered Codecov's bash uploader script, which thousands of CI pipelines piped straight into `bash`. The modified script exfiltrated *environment variables* from the CI runner — and CI environments are stuffed with secrets injected as env vars: cloud keys, registry tokens, signing keys. Because so many pipelines passed secrets as plain env vars, a single compromised script harvested credentials across a huge number of organizations, several of which were then breached downstream. The lesson lands directly on section 2's warning: secrets-as-env-vars have a wide leak surface, and a CI runner is a high-value place for that surface to be exposed. Minimize secrets in the CI environment, prefer short-lived OIDC tokens over static keys, and never pipe a remote script straight into a shell with your secrets in scope.

The **dependency-confusion and `event-stream`-style supply-chain attacks** make the adjacent point: the credential does not have to leak from *your* repo to hurt you — a malicious dependency or a compromised build step running in *your* pipeline with *your* secrets in its environment can exfiltrate them just the same. This is why the supply-chain posts in this series (signing, SBOMs, provenance) and the secrets posts compose: signing tells you the artifact is the one you built, and tight secret scoping limits what a compromised build can steal.

And the everyday version, which is the one that will actually happen to you: an engineer in a hurry commits a `Secret` with a real base64 password to "get it working," intends to fix it later, and a scanner (or worse, an attacker) finds it. The remediation is never "remove the file." Because of Git history's permanence, the only safe remediation is to **rotate the credential immediately** — change it at the source so the leaked copy is worthless — and *then* clean up the repo. Treat any plaintext secret that touched Git as compromised, full stop. Rotate first, scrub second.

#### Worked example: the cost of the leaked password, and what the fix saved

Walk the leaked-database-password incident from the intro as a number, because the arithmetic is what convinces a skeptical team to adopt the patterns. A `Secret` YAML with a base64 production database password was committed to a GitOps repo. Base64 is not encryption, so the value was one `base64 -d` from plaintext. The repo was later mirrored to a contractor's read-only fork. The credential sat exposed for four months before discovery.

The remediation cost, honestly tallied: rotate the production database password (a coordinated change touching every service that connects, done under a maintenance window to avoid a thundering-herd of reconnects) — roughly half a day of senior-engineer time plus a 20-minute coordinated rollout; audit four months of database access logs for anomalous use of the credential — a day; rewrite Git history across the repo, every clone, and the contractor's mirror, then confirm the purge — most of a day and a fair amount of nagging; and the soft cost of a security review and a write-up. Call it three engineer-days and a maintenance window, *if* nothing was actually exploited. Had the credential been used by an attacker, the cost is unbounded.

Now the fix, priced out. Moving that one secret to the External Secrets Operator: the value goes into Vault once (`vault kv put secret/prod/checkout/db password=…`, two minutes), an `ExternalSecret` and `SecretStore` are committed (referencing the Vault path, never the value — fifteen minutes), and rotation thereafter is `vault kv put …` plus a one-hour auto-resync — *zero* commits, *zero* history risk, ever. The standing prevention (a gitleaks pre-commit hook plus a CI gate) is the few lines from the next section, added once. The trade is stark: a one-time setup measured in minutes per secret versus a recurring three-engineer-day-and-a-maintenance-window remediation each time someone fat-fingers a commit — and the recurring cost only *averages* to three days because it assumes you got lucky and nobody exploited it. That asymmetry is the entire business case for never letting a plaintext secret touch Git.

## 9. Config drift: when reality diverges from Git

There is a failure mode that has nothing to do with leaks and everything to do with discipline, and it quietly breaks the "same image, different config, predictable environment" promise. It is **config drift**: the live state of a ConfigMap (or Secret) in the cluster no longer matches what is declared in Git.

It usually starts innocently. Production is misbehaving, someone needs to bump `WORKER_CONCURRENCY` from 8 to 16 *right now*, and they run:

```bash
kubectl edit configmap checkout-config -n prod   # bumps WORKER_CONCURRENCY to 16
```

The fire is out. But now the cluster says 16 and Git says 8. Three things can happen next, all bad:

1. **A GitOps controller reverts it.** If Argo CD or Flux is watching that namespace with self-heal enabled, it sees the live state diverge from the declared state and *reconciles it back to 8* — undoing the fix, possibly re-igniting the fire, usually at the worst time and to everyone's confusion ("why did concurrency drop back to 8?!").
2. **Nothing reverts it, and the environments diverge.** Without self-heal, the change just sits there, undocumented, until the next person who reads Git "knows" concurrency is 8 and reasons from a false picture. Now prod behaves differently from what the repo says, and the next deploy from Git silently resets it.
3. **The next sync clobbers it unpredictably.** Depending on timing, the next `kubectl apply` or sync overwrites the manual change — or doesn't — and you have a race between humans and the controller.

This is the same drift problem that IaC fights at the infrastructure layer: manual `clickops` changes that aren't in code create state nobody can reproduce. The fix is identical and it is a discipline, not a tool: **config is code, and the only way to change it is a commit.** Want concurrency at 16? Edit the ConfigMap *in Git*, open a PR, merge it, and let the GitOps controller apply it. The cluster becomes a *projection* of Git, never a place you edit directly. If you genuinely need a break-glass manual change during an incident, make it — but the very next step is to backfill it into Git so reality and declaration reconverge, and so the postmortem has a record. This is exactly where this post meets GitOps and IaC: config-as-code in Git is the cure for config drift, just as infrastructure-as-code is the cure for infrastructure drift.

In an Argo CD setup the self-heal behaviour is an explicit setting, and knowing it is the difference between a controller that quietly undoes your fixes and one that pages you to fix them in Git:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: checkout-prod
  namespace: argocd
spec:
  source:
    repoURL: https://github.com/example/deploy.git
    path: overlays/prod
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: prod
  syncPolicy:
    automated:
      prune: true        # delete resources removed from Git
      selfHeal: true     # revert any manual cluster change back to Git's state
    syncOptions:
      - ApplyOutOfSyncOnly=true
```

With `selfHeal: true`, that `kubectl edit` bumping concurrency to 16 is reverted to 8 within the next reconcile — which is the *correct* behaviour, because it enforces that Git is the truth, but it means your only durable way to set 16 is to commit it. With self-heal off, Argo CD instead shows the Application as `OutOfSync` (a visible, alertable signal) and waits for a human to either sync or reconcile the difference into Git. Either way, the drift is *surfaced* rather than silent. The anti-pattern is a cluster with no GitOps controller at all, where a manual edit simply persists, invisible, until the next `kubectl apply` from a CI job overwrites it at a random time. The reliability framing of "the deploy reverted my fix" as an incident-response concern lives in the SRE posts; the *delivery* lesson here is that config must round-trip through Git so the cluster never holds state the repo doesn't know about.

### Stress-testing the design

A good way to trust a design is to break it on paper. Run the config-and-secrets setup through the failure modes:

- **What if two PRs change the same ConfigMap at once?** Git resolves it — the merge conflict surfaces in the PR, a human reconciles, and the GitOps controller applies the single merged result. This is precisely the benefit of config-as-code: concurrent config changes are mediated by the same review and merge machinery as code, not by two engineers racing `kubectl edit`.
- **What if the secrets backend (Vault) is down when a Pod starts?** With ESO, the *Kubernetes Secret already exists* from the last successful sync, so a Pod that mounts it starts fine even while Vault is unreachable — the operator just can't *refresh* until Vault returns. With the CSI driver mounting directly at Pod start, a Vault outage can block new Pods from starting (the mount fails), which is a sharper coupling — a reason to weigh ESO's cached Secret against CSI's no-etcd-copy for availability-critical workloads.
- **What if a secret rotates while a Pod is mid-request?** A volume-mounted secret updates the file, but in-flight connections keep the old credential until they reconnect; a graceful design re-authenticates on the next connection from the pool. An env-var secret does not change at all until a restart — which is why rotation needs the reloader or volume-mount path from section 10.
- **What if someone bypasses the pre-commit hook with `--no-verify`?** The CI gate (gitleaks on `pull_request`) is the mandatory backstop and fails the PR. Defence in depth is exactly two layers — advisory local, mandatory CI — so neither single bypass is fatal.
- **What if the Sealed Secrets controller's private key is lost?** Every `SealedSecret` becomes undecryptable. The mitigation is a secure, out-of-band backup of the sealing key (not in the same repo), which is why the ESO model — where the value lives in a vault built for durability and backup — is often the lower-risk choice once you have a vault to run.

### Per-environment config without per-environment images

The last piece of "same image, different environment" is *how* you express the per-environment differences as code. You do not want three hand-maintained copies of every manifest; you want one base plus per-environment overlays. Two tools own this space — Kustomize and Helm — and the [Helm vs Kustomize post](/blog/software-development/ci-cd/helm-vs-kustomize-templating-your-manifests) (planned in this series) goes deep on the trade-off. The sketch you need here:

With **Kustomize overlays**, a `base/` holds the environment-agnostic manifests and each environment is an overlay that patches only what differs:

```yaml
# base/kustomization.yaml
resources:
  - deployment.yaml
  - service.yaml
configMapGenerator:
  - name: checkout-config
    literals:
      - LOG_LEVEL=info
---
# overlays/prod/kustomization.yaml
resources:
  - ../../base
images:
  - name: ghcr.io/example/checkout
    digest: sha256:9f2c...        # the SAME digest base used; pinned here
configMapGenerator:
  - name: checkout-config
    behavior: merge
    literals:
      - LOG_LEVEL=warn            # prod overrides only what differs
      - DATABASE_HOST=db.prod.internal
```

With **Helm**, the chart is parameterized by a `values.yaml` and you supply a per-environment values file:

```yaml
# values-prod.yaml  ->  helm upgrade --install checkout ./chart -f values-prod.yaml
image:
  repository: ghcr.io/example/checkout
  digest: sha256:9f2c...          # same digest across all environments
config:
  logLevel: warn
  databaseHost: db.prod.internal
  enableNewCheckout: false
externalSecret:
  vaultPath: prod/checkout/db     # ref only, never the value
```

Either way, the crucial invariant is visible: the **image digest is identical across environments**, and only the *config values* (and the secret *references*) change per overlay/values file. That is build-once-promote-everywhere expressed as code. (When promoting, you flow the same digest through dev → staging → prod, changing only which values/overlay applies — see [build once, promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) for how the digest is pinned and tracked.)

#### Worked example: one image, three environments

Here is the proof that build-once-promote-everywhere holds, made concrete. The CI pipeline builds the `checkout` service once at commit `a1b2c3d`, producing image `ghcr.io/example/checkout@sha256:9f2c…`. That single digest is then deployed to three environments, each with its own ConfigMap and its own secret reference, and *nothing else changes*:

| | dev | staging | prod |
| --- | --- | --- | --- |
| Image digest | `sha256:9f2c…` | `sha256:9f2c…` | `sha256:9f2c…` (identical) |
| `LOG_LEVEL` | `debug` | `info` | `warn` |
| `DATABASE_HOST` | `db.dev.internal` | `db.stg.internal` | `db.prod.internal` |
| `ENABLE_NEW_CHECKOUT` | `true` | `true` | `false` |
| `WORKER_CONCURRENCY` | `2` | `4` | `8` |
| DB password source | Vault `dev/checkout/db` | Vault `stg/checkout/db` | Vault `prod/checkout/db` |
| Rebuild needed? | no | no | no |

![Grid showing one identical image digest promoted to dev, staging, and prod with only the per-environment config and secret reference changing](/imgs/blogs/configuration-and-secrets-in-kubernetes-8.png)

The grid above visualizes it: one digest at the top, promoted left to right with no rebuild, binding a different per-environment config in dev, staging, and prod. The numbers tell the whole story. The thing tested in staging — `sha256:9f2c…` — is the *exact* thing that runs in prod, because it is the same digest. Dev runs verbose logging with the new-checkout flag on for development; prod runs quiet logging with the flag off and a higher worker count. The database password for each environment is fetched from its own Vault path by ESO, never committed anywhere. No environment required its own build, so there is no rebuild drift. This is the entire thesis of the post made measurable: **environment-agnostic image plus per-environment injected config equals build-once-promote-everywhere, proven by the identical digest across the row.**

## 10. The config-change-needs-a-restart problem

I want to give this its own short section because it is the operational gotcha that ties config, secrets, and deploys together, and it is the thing people forget until it surprises them in production.

The core fact, restated for both config and secrets: **a value injected as an environment variable is read exactly once, at process start.** Change the ConfigMap or Secret, and a running Pod keeps the old value indefinitely. There is no signal, no reload, no warning — the Pod is simply running stale config, and it will keep doing so until it is restarted for some unrelated reason, at which point your "change" suddenly takes effect long after you made it (which is its own confusing incident).

You have three honest options, and you should choose deliberately per value:

1. **Restart on change (the explicit rollout).** Treat a config change like a code change: it triggers a rolling deploy. `kubectl rollout restart deployment/checkout` cycles the Pods so they re-read the env. This is clean and observable — the change is a deploy, with all the safety of a rolling update — but it costs a rollout and is wrong for things that must change *instantly* (a kill-switch flag).
2. **Mount as a volume and reload.** As covered in section 2, a ConfigMap or Secret mounted as a file gets updated in place by the kubelet, and a watching app can re-read it without a restart. This is the path to instant change, at the cost of writing reload logic (or running a sidecar that signals the app).
3. **A reloader controller.** Run [Stakater Reloader](https://github.com/stakater/Reloader), annotate the Deployment, and any change to a referenced ConfigMap or Secret automatically triggers a rolling restart. This automates option 1 — you get "the config changed, so the workload was redeployed with it" for free, which is the right default for env-var-based config that should change on a deploy cadence anyway:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
  namespace: prod
  annotations:
    # Reloader restarts this Deployment whenever a referenced
    # ConfigMap or Secret changes.
    reloader.stakater.com/auto: "true"
```

The decision rule: config that should change on a *deploy cadence* (most of it) → let it ride a rolling restart, automated by Reloader. Config that must change *instantly* and without a restart (a circuit-breaker flag, a feature kill-switch) → mount it as a volume with a watching reload, or better, put genuinely-instant toggles in a dedicated feature-flag service rather than a ConfigMap. Choosing the wrong one is how you get the noon-banner incident from section 2's worked example.

## 11. Measuring whether you actually got this right

A practice you cannot measure is a practice you cannot trust, so here is how to put honest numbers on "our config and secrets are handled correctly." None of these require fancy tooling; they require deciding what good looks like and then counting.

**Builds per release (the build-once metric).** Count how many `docker build` invocations occur between a commit and that commit reaching production. The target is exactly one. A team that rebuilds per environment will measure three or four; after moving config out of the image, they measure one. The before→after on a real migration looks like: build invocations per release 3 → 1, and — the consequence that actually matters — environment-specific image bugs (a prod-only failure that staging never saw because staging ran different bytes) dropping from "a couple per quarter" to zero, because there are no longer different bytes. State it honestly: the second number is the one you care about, and it is observable in your incident log.

**Plaintext-secret commits caught vs. leaked.** Once gitleaks runs as a pre-commit hook and a CI gate, you can count attempted secret commits blocked (the hook's output) and secrets that reached `main` anyway (the CI gate's failures, which should be zero, plus periodic full-history scans). A healthy program shows the pre-commit hook catching the occasional fat-finger (proving the net works) and zero secrets reaching a protected branch. The leading indicator is "scans run on every PR: yes"; the lagging indicator is "secrets found in history this quarter: 0."

**Time-to-rotate.** How long, end to end, from "we must rotate this credential" to "every workload is running on the new value"? With a committed Secret it is hours-to-days (edit, PR, deploy, plus history cleanup if it leaked). With ESO and a reloader it is the refresh interval plus a rolling restart — minutes, hands-off. Measuring this matters because rotation speed is your blast-radius limiter when a credential *is* compromised: the faster you rotate, the smaller the window an attacker has. A defensible before→after: time-to-rotate a database password 1 day → 15 minutes, and the rotation going from "a risky manual coordination" to "a Vault write plus an automatic resync."

**Config-drift incidents.** Count the times per quarter that production behaved unexpectedly and the root cause was "the live ConfigMap/Secret didn't match Git." Before GitOps with config-as-code, this is a recurring, hard-to-diagnose class of incident (someone `kubectl edit`-ed something months ago and nobody remembers). After, with self-heal or `OutOfSync` alerting, it trends to zero because drift is either prevented or surfaced. This one is worth tracking explicitly in postmortems with a tag, because it is invisible until you name it.

#### Worked example: lead time for a config change, decomposed

Decompose the lead time for the most common change a team makes — flipping a log level or a feature flag — under three setups, because the decomposition shows where the time goes and which practice removes it. Say the change itself (deciding the new value) takes 2 minutes in all three.

- **Baked into the image (the anti-pattern).** Edit the Dockerfile, rebuild, re-test, re-package, re-deploy: build 6 min + test 8 min + push 2 min + rolling deploy 3 min = 19 min of pipeline on top of the 2-minute change = **21 minutes**, and it ships *new bytes* you must re-trust.
- **Env-var ConfigMap, no reloader.** Edit the ConfigMap in Git (2 min change + 3 min PR + merge), GitOps syncs (1 min), then you remember it won't take effect and run a manual `rollout restart` (3 min) = **9 minutes**, and you had to *remember* the restart (the noon-banner trap).
- **Content-hashed ConfigMap via Kustomize, or a reloader.** Edit the ConfigMap in Git (2 min change + 3 min PR + merge), GitOps syncs, the hash changes, a rolling update carries the new config automatically (3 min) = **8 minutes**, with no separate step to forget and *zero* new image bytes to re-trust.

The arithmetic makes the point quantitative: moving config out of the image roughly halves the lead time for a config change (21 → 9 min) by deleting the build/test/package work, and content-hashing or a reloader removes the easy-to-forget manual restart on top. The deeper win is in the column you can't see in the minutes — the baked path ships untested bytes every time, while the injected paths ship the same trusted digest and change only data. Lead time for changes is one of the four DORA metrics; config injection improves it directly for the most frequent change class there is.

## 12. How to reach for this (and when not to)

Every practice here has a cost, and the most useful thing I can do is tell you when *not* to pay it.

**Always do this, even on day one, even tiny:** keep config out of the image (env vars from a ConfigMap is enough), and never commit a plaintext secret to Git. These two are free and non-negotiable. The first preserves build-once-promote-everywhere; the second prevents the leak that you cannot take back. A solo developer on a hobby project should still do both — `kubectl create secret` from a local value (never committed) is fine at that scale.

**Add a secret-scanning pre-commit hook and CI gate early.** It is a few lines and it catches the mistake that hurts the most. Cheap insurance; add it before you have anything worth leaking, because by the time you do, the habits are set.

**Reach for Sealed Secrets when** you adopt GitOps and want the repo to be the full declarative state, you have no existing secrets manager, and your rotation cadence is fine being a commit. It is the lowest-infrastructure correct answer.

**Reach for the External Secrets Operator (or CSI driver) when** you already run Vault or a cloud secrets manager, you have compliance requirements, or you need automatic rotation without commits. The source of truth belongs in the system built for secrets.

**Turn on etcd encryption-at-rest and tight RBAC** as soon as you have a real cluster with real secrets and more than a couple of operators. Before that — a single-node dev cluster you own end to end — it is lower priority than the Git hygiene above.

**When NOT to:**

- **Don't stand up Vault and ESO for a three-person startup on a managed PaaS.** If your platform (Render, Fly, a managed Kubernetes with a built-in secrets integration, a cloud Run service) already injects secrets from a managed store, use that. A self-operated Vault is a serious piece of infrastructure with its own availability and unsealing concerns; do not take it on until your secret count and compliance needs justify the operational weight.
- **Don't put genuinely instant feature flags in a ConfigMap.** ConfigMaps change on a sync/restart cadence. If you need a kill-switch that flips in milliseconds for a percentage of traffic, that is a feature-flag platform's job, not a ConfigMap's.
- **Don't run three secrets patterns at once "to be safe."** Pick the one that matches where your source of truth lives. Running Sealed Secrets *and* ESO *and* CSI multiplies the failure modes and the confusion about which value is authoritative.
- **Don't reach for hot-reload mounts everywhere.** Most config *should* change on a deploy cadence — a rolling restart is observable and safe. Hot-reload is for the specific values that genuinely cannot wait for a rollout. Reaching for it by default adds reload logic you then have to debug.

Worth noting where this meets reliability: for *stateful* workloads — databases, queues, anything with disk and identity — config and secret changes carry extra weight because a careless rolling restart can disrupt quorum or connections. The SRE post on [running stateful systems reliably](/blog/software-development/site-reliability-engineering/running-stateful-systems-reliably) covers the operational care those workloads need; the config-injection mechanics here still apply, but *how* you roll a change through them is a reliability question that post owns.

## 13. Key takeaways

- **Config that varies between environments belongs in the environment, not the build.** Database URLs, feature flags, log levels, endpoints — inject them at deploy time so one immutable image runs everywhere. Baking them in forces a rebuild per environment, which reintroduces the drift that build-once-promote-everywhere exists to kill.
- **ConfigMaps hold non-secret config; inject them as env vars (simple, static-at-start, leak-prone) or as mounted files (hot-reloadable, lower leak surface, good for structured config).** Choose per value, not per dogma.
- **A Kubernetes Secret is base64-encoded, not encrypted.** `base64 -d` is the whole "decryption." A Secret in Git is a plaintext leak, and Git history is forever — so the cardinal rule is never commit a plaintext secret.
- **The Secret object is the right destination; the art is getting the value in without it touching Git.** Sealed Secrets encrypt the blob so it is safe to commit (truth stays in Git); the External Secrets Operator and the CSI driver keep the value in a vault and put only a reference in Git (truth lives outside Git, rotation is a vault operation).
- **Harden the cluster: encryption-at-rest for etcd and least-privilege RBAC scoped to named secrets.** Encryption protects the storage; RBAC protects the access; both are the floor under any secrets pattern.
- **Scan for committed secrets with gitleaks, trufflehog, or git-secrets — as a pre-commit hook and a CI gate.** And if a secret ever touches Git, rotate it first, scrub second; removing the file does not undo the leak.
- **Config drift — a manual `kubectl edit` that isn't in Git — breaks the same-config-same-environment promise.** GitOps either reverts it or the environments silently diverge. Config is code: change it with a commit.
- **Express per-environment differences as Kustomize overlays or Helm values, with the image digest identical across all environments.** That identical digest *is* the proof that build-once-promote-everywhere holds.
- **Env-var values are read once at start.** Decide deliberately how each value changes: a rolling restart (automated by a reloader) for deploy-cadence config, a volume-mount reload for the rare instant-change value.

## Further reading

- [The Twelve-Factor App — Config](https://12factor.net/config) — the canonical statement of "config in the environment," the principle this post operationalizes for Kubernetes.
- [Kubernetes docs: ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/) and [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) — the primitives, including the explicit warning that Secret data is only base64-encoded.
- [Kubernetes docs: Encrypting Secret Data at Rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/) — the `EncryptionConfiguration` and KMS provider setup from section 4.
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets), [External Secrets Operator](https://external-secrets.io/), and the [Secrets Store CSI Driver](https://secrets-store-csi-driver.sigs.k8s.io/) — the three production secrets patterns, with `kubeseal`, `ExternalSecret`, and `SecretProviderClass` references.
- [gitleaks](https://github.com/gitleaks/gitleaks) and [trufflehog](https://github.com/trufflesecurity/trufflehog) — secret-scanning for the pre-commit hook and CI gate in section 8.
- Within this series: [the CI/CD overview that frames this series](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) (the map), [build once, promote everywhere](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) (why the artifact is immutable — this post is its operational other half), and the planned siblings on the Kubernetes objects that matter for delivery, Helm vs Kustomize templating, and secrets management in the pipeline.
- Out of series: [running stateful systems reliably](/blog/software-development/site-reliability-engineering/running-stateful-systems-reliably) (SRE) for the extra care config and secret changes need on stateful workloads.

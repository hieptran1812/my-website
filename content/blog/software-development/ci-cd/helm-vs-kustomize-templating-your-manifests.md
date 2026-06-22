---
title: "Helm vs Kustomize: Templating Your Manifests Without the Mess"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "One app, three environments, and a strong wish to never maintain three hand-copied YAML files that quietly drift. This post is the practical face-off between Kustomize's patch-an-overlay model and Helm's template-and-package model: when each fits, the worked artifacts for both, the over-templated-chart trap that hides bugs, and why you always render the manifest in CI before you deploy it."
tags:
  [
    "ci-cd",
    "devops",
    "kubernetes",
    "helm",
    "kustomize",
    "gitops",
    "manifests",
    "configuration",
    "templating",
    "everything-as-code",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/helm-vs-kustomize-templating-your-manifests-1.png"
---

The incident that taught me to take this post seriously started with a one-line config fix. A service was timing out against a downstream API, so we bumped a client timeout from 5 seconds to 15. The fix was obvious, the change was reviewed in two minutes, and it went out that afternoon. Staging got better. Production did not. Two days later the same pages were firing, the same timeouts, and a junior engineer spent an hour convinced the fix had not deployed at all before someone finally diffed the two manifests by hand and found the truth: we had three copies of the Deployment YAML, one per environment, and the fix had been pasted into `dev.yaml` and `staging.yaml` and never into `prod.yaml`. The production file had drifted. It had been drifting for months. Nobody could have told you what was different between the three, because the only way to know was to read 220 lines of YAML three times and compare them line by line, and nobody ever does that.

That is the problem this whole post is about, and it is almost embarrassingly common. You have one application. You run it in dev, staging, and production. The environments are not identical — prod runs five replicas where dev runs one, prod has bigger CPU and memory requests, prod points at a different image tag and a different database URL and a real secret instead of a fake one. The temptation, every single time, is to copy the manifest once per environment and edit the few lines that differ. It works on day one. By day ninety you have three files that share maybe ninety percent of their content, drift in the other ten percent, and no human alive can tell you what the real difference is supposed to be. A fix lands in two of three. A new field gets added to prod and forgotten in staging, so staging stops being a faithful rehearsal of prod, which is the entire reason staging exists. The duplication is not a style problem. It is a correctness problem and an incident generator.

![A side by side comparison showing three hand-copied manifests drifting apart where a fix reaches two of three environments versus one shared base with thin per-environment variation that cannot drift](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-1.png)

There are two dominant answers to this in the Kubernetes world, and they take opposite philosophical positions. **Kustomize** says: keep one base set of manifests and describe each environment as a small *patch* on top of it — no templating language at all, just YAML transforming YAML. **Helm** says: turn your manifests into *templates* with placeholders like `{{ .Values.replicaCount }}`, supply a file of values per environment, and wrap the whole thing in a versioned package you can install, upgrade, and roll back like software. Both solve the duplication problem. They solve it so differently that picking the wrong one for your situation creates a *new* mess — an over-templated Helm chart where every line is a knob nobody understands, or a Kustomize tree twenty overlays deep that is somehow harder to follow than the three files you started with. By the end of this post you will be able to write a real Kustomize base with environment overlays, write the equivalent Helm chart with values files, render both to see the actual YAML they produce, know which to reach for and when to combine them, and — the load-bearing habit — never deploy a template you have not rendered and read first.

This post sits on the series spine — `commit → build → test → package → deploy → operate` — at the **package** and **deploy** steps. The image is already built, scanned, and pinned by digest from earlier in the pipeline; now we are deciding how the *manifest* that places that image on the cluster varies per environment without duplication. It leans hard on two of the series' governing principles: **"build once, promote everywhere"** (the same image digest flows through all three environments — only the surrounding config varies) and **"everything as code"** (the base, the overlays, the chart, the values are all versioned in Git and reviewed). For the foundation, see the intro map, [from commit to production: the CI/CD model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model). For the objects we are templating, see [Kubernetes for delivery: the objects that matter](/blog/software-development/ci-cd/kubernetes-for-delivery-the-objects-that-matter), and for the config and secrets these tools manage, [configuration and secrets in Kubernetes](/blog/software-development/ci-cd/configuration-and-secrets-in-kubernetes).

## 1. The problem stated precisely: per-environment variation without duplication

Before reaching for a tool, it pays to name the problem exactly, because the shape of the problem tells you which properties the solution needs. You have a single application. You deploy it to N environments — let us say three: dev, staging, prod. Across those environments, a small, bounded set of things changes:

- **Replica count.** dev: 1. staging: 2. prod: 5.
- **Resource requests and limits.** dev runs lean to save money. prod gets real CPU and memory.
- **The image tag or digest.** Ideally the *same* digest is promoted through all three (build once, promote everywhere), but at any given moment dev might be ahead of prod.
- **Configuration.** Log level (debug in dev, info in prod), feature flags, the URL of the downstream API, the database connection string, timeouts, sampling rates.
- **Secrets.** Different credentials per environment, never the same database password in dev and prod.
- **Environment-specific resources.** prod might have a `HorizontalPodAutoscaler` and a `PodDisruptionBudget` that dev does not bother with; dev might have a debug sidecar that prod must never run.

Everything *else* — the container ports, the labels, the readiness and liveness probe paths, the volume mounts, the overall shape of the Deployment and Service — is identical across environments. If you tallied the lines, you would find that something like ninety percent of the YAML is shared and ten percent varies. The naive approach copies all of it three times and asks humans to keep the ninety percent in sync forever. That is the **manual-copy anti-pattern**, and it fails in a specific, predictable way.

The failure mode is *drift*. Drift is when two things that are supposed to be the same gradually become different without anyone deciding they should be. With three copies of a manifest, every edit is a chance to drift: you change one file and forget the others, or you change all three but make a typo in one, or someone adds a field to prod under deadline pressure and never backports it to staging. The insidious part is that drift is *silent*. Nothing errors. Each manifest is individually valid YAML that the cluster happily accepts. The system keeps running. You only discover the drift when it bites — when the fix that worked in staging mysteriously does not work in prod, and you finally diff the files and find they were never the same to begin with.

It is worth being precise about why duplication is *mathematically* a drift generator rather than just an aesthetic complaint, because the arithmetic makes the case more strongly than any appeal to taste. Suppose the shared portion of your manifest is $S$ lines and you maintain it across $E$ environments by hand. Every edit to a shared field is an opportunity to touch $E$ files and get all $E$ right; if the per-edit probability of correctly updating any single copy is $p$, the probability the edit lands correctly *everywhere* is $p^{E}$. With three environments and a generous $p = 0.95$ per file, you get a consistent result only about $0.95^{3} \approx 0.86$ of the time — meaning roughly one shared edit in seven leaves at least one environment drifted. Over a quarter of, say, forty shared edits, that is on the order of five or six drift events introduced silently, each waiting to surface as the "it worked in staging but not prod" incident. Now factor the *base-plus-overlay* model into the same arithmetic: a shared edit touches *one* file (the base), so the consistency probability is just $p$, not $p^{E}$ — drift on the shared portion drops from "almost guaranteed over a quarter" to "as reliable as a single edit." That collapse from $p^{E}$ to $p$ is the entire quantitative argument for templating, and it gets *worse* for the copy-paste approach as you add environments: a fourth or fifth environment makes $p^{E}$ fall off a cliff while the overlay model stays flat at $p$.

So the property we actually need is **a single source of truth for the shared parts, with the per-environment variation expressed explicitly and minimally.** We want to write the ninety percent *once*, and write only the ten-percent delta per environment, in a form small enough that a reviewer can look at the prod overlay and immediately see *exactly* how prod differs from the base — five replicas, bigger resources, this digest, this config — with nothing hidden. That is the bar. Both Kustomize and Helm clear it. They just take opposite routes there, and the route has consequences for readability, for packaging, for rollback, and for how badly things can go wrong when you over-use them. Let us take Kustomize first, because its model is the simpler of the two to hold in your head.

## 2. Kustomize: patch a base with overlays, no templating language

Kustomize's central idea is almost austere in its simplicity: **your manifests are plain YAML, and you transform plain YAML into other plain YAML.** There is no templating language. There are no placeholders, no `{{ }}`, no loops, no conditionals embedded in your files. What you write is valid Kubernetes YAML that you could apply directly; what Kustomize produces is also valid Kubernetes YAML. In between, it applies a series of *transformations* — patches, prefixes, label injections, image swaps — described declaratively in a file called `kustomization.yaml`. The thing you ship is the output of `kustomize build`, and because it is pure YAML in and pure YAML out, you can always run that command and read exactly what you are about to deploy.

The structure is a `base/` directory plus one `overlays/<env>/` directory per environment. The base holds the common manifests and a `kustomization.yaml` listing them. Each overlay has its own `kustomization.yaml` that points at the base and layers on the environment's specific changes.

![A flow diagram showing one Kustomize base feeding three environment overlays for dev staging and prod that each get rendered by kustomize build into a final manifest you apply](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-2.png)

Here is a real base. The Deployment is ordinary Kubernetes YAML — note that it has *no* environment-specific values baked in; it carries sensible defaults that the overlays will override.

```yaml
# base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-api
  labels:
    app: orders-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: orders-api
  template:
    metadata:
      labels:
        app: orders-api
    spec:
      containers:
        - name: orders-api
          image: ghcr.io/acme/orders-api  # tag set per overlay
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 250m
              memory: 256Mi
          envFrom:
            - configMapRef:
                name: orders-api-config
```

The base also has a `Service` and a base `kustomization.yaml` that ties the files together and applies labels and annotations every environment should carry:

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
commonLabels:
  app.kubernetes.io/name: orders-api
  app.kubernetes.io/part-of: commerce-platform
commonAnnotations:
  team: payments
```

`commonLabels` and `commonAnnotations` are *transformers*: Kustomize injects those labels onto every resource and into the selectors, so you do not repeat them on each object. (A note on versions: newer Kustomize prefers `labels:` with `includeSelectors: true` over the older `commonLabels`, which always touches selectors — for clarity I use `commonLabels` here, but be aware of the distinction in production.) Now the prod overlay, which is where the variation lives and where the readability payoff shows up:

```yaml
# overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: orders-prod
namePrefix: prod-
resources:
  - ../../base
patches:
  - path: replicas-and-resources.yaml
    target:
      kind: Deployment
      name: orders-api
images:
  - name: ghcr.io/acme/orders-api
    newTag: "1.8.3"
    digest: sha256:9c4e0b...   # pin by digest; promote the tested artifact
configMapGenerator:
  - name: orders-api-config
    literals:
      - LOG_LEVEL=info
      - DOWNSTREAM_TIMEOUT_SECONDS=15
      - DOWNSTREAM_URL=https://api.internal.acme.com
```

Read that overlay top to bottom and you know *everything* that makes prod prod: it lives in the `orders-prod` namespace, every object gets a `prod-` name prefix so you can never confuse it with staging, the image is pinned to a specific tag *and* digest, and the config carries the production log level, timeout, and downstream URL. The actual replica and resource changes live in the patch file it references:

```yaml
# overlays/prod/replicas-and-resources.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-api
spec:
  replicas: 5
  template:
    spec:
      containers:
        - name: orders-api
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
```

That is a **strategic-merge patch**: a partial Deployment that Kustomize merges into the base by matching on the structure. You only write the fields you are changing. Kustomize is smart about lists — it matches the container named `orders-api` by name and overrides just its `resources`, leaving the ports and probes from the base intact. (When the merge semantics get awkward — say you want to *remove* a field, or surgically change one element of a list by index — you switch that one change to a **JSON 6902 patch**, which uses explicit `op: replace`/`op: remove` operations against a JSON pointer path. Reach for JSON patches only when strategic-merge cannot express the change; they are more precise but less readable.)

The dev and staging overlays are the same shape, smaller deltas:

```yaml
# overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: orders-dev
namePrefix: dev-
resources:
  - ../../base
images:
  - name: ghcr.io/acme/orders-api
    newTag: "1.9.0-rc2"   # dev runs ahead
configMapGenerator:
  - name: orders-api-config
    literals:
      - LOG_LEVEL=debug
      - DOWNSTREAM_TIMEOUT_SECONDS=5
      - DOWNSTREAM_URL=https://api.staging.acme.com
```

dev does not even patch replicas or resources — it inherits `replicas: 1` and the lean resources straight from the base. The entire dev delta is *fifteen lines*. That is the whole point: the shared two hundred lines exist exactly once, in the base, and each environment's difference is small enough to take in at a glance.

### The configMapGenerator and the content-hash trick

The `configMapGenerator` deserves a closer look because it solves a problem most people hit the hard way. Kubernetes ConfigMaps are mutable — you can change one in place. But Deployments do *not* restart their pods when a referenced ConfigMap changes; the pods keep running with the old config until something else triggers a rollout. So you change a config value, apply it, and nothing happens. The new value sits in the ConfigMap, unused, until the next unrelated deploy picks it up — at which point a config change you made days ago suddenly takes effect and nobody connects the dots.

Kustomize's `configMapGenerator` fixes this by appending a **content hash** to the generated ConfigMap's name — `orders-api-config-7m2k9d4f8c` — and rewriting every reference to that name throughout your manifests. When the config content changes, the hash changes, the name changes, and because the Deployment now references a *different* ConfigMap name, the pod template changes, which triggers a rollout automatically. Change the config, get a rollout, see the change take effect immediately — exactly the behavior you wanted. The same applies to `secretGenerator` for Secrets. This is one of those small mechanisms that quietly prevents a whole class of "I changed the config but nothing happened" confusion. For the deeper treatment of how config and secrets flow through the cluster, see [configuration and secrets in Kubernetes](/blog/software-development/ci-cd/configuration-and-secrets-in-kubernetes).

### Rendering: what you see is what you ship

Because Kustomize is pure YAML transformation, the verification step is trivial and you should make it a reflex:

```bash
# See exactly what prod will get — no surprises, no hidden templating
kustomize build overlays/prod

# Or, since Kustomize is built into kubectl:
kubectl kustomize overlays/prod

# Apply it directly with -k:
kubectl apply -k overlays/prod
```

`kustomize build overlays/prod` prints the final, fully-merged manifest to stdout — the actual Deployment with 5 replicas, the `prod-orders-api` name, the prod config baked into a content-hashed ConfigMap, the image pinned to the digest. There is no gap between what the file says and what gets applied, because the rendering is mechanical. The `image transformer` (`images:` block) is worth calling out specifically: it sets the tag and digest per environment without touching the base, which is exactly how you wire "build once, promote everywhere" into Kustomize — the base names the image, each overlay pins the *digest* it promotes. We will return to this in the promotion section.

## 3. Helm: templating plus packaging

Helm starts from the opposite premise. Where Kustomize refuses to introduce a templating language, Helm *is* a templating language wrapped around a package manager. Your manifests become **templates** — files full of Go-template directives that get filled in from a values structure at render time. The collection of templates plus a default values file plus some metadata is a **chart**, and a chart is a versioned, redistributable artifact. Helm installs charts into a cluster as **releases**, tracks the history of every release, and can roll one back to a previous revision with a single command. That last capability — versioned packaging with rollback — is the thing Kustomize simply does not have, and it is why Helm dominates the distribution of third-party software on Kubernetes.

![A layered stack diagram of a Helm chart showing templated manifests and helper snippets over a default values file overridden by a per-environment values file packaged as a versioned release with history](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-3.png)

Here is the same `orders-api` as a Helm chart. First the directory layout:

```bash
orders-api/
  Chart.yaml            # name, version, appVersion
  values.yaml           # default values (the knobs)
  values-prod.yaml      # per-environment override
  values-staging.yaml
  templates/
    deployment.yaml     # templated
    service.yaml
    configmap.yaml
    _helpers.tpl        # reusable template snippets
```

The `Chart.yaml` carries the chart's own version (the package version) separately from the app version:

```yaml
# Chart.yaml
apiVersion: v2
name: orders-api
description: The orders API service
type: application
version: 0.4.2        # chart version — bump when the templates change
appVersion: "1.8.3"   # the app/image version this chart ships by default
```

The default `values.yaml` is the *surface* of the chart — the set of knobs an operator is allowed to turn:

```yaml
# values.yaml — defaults
replicaCount: 1
image:
  repository: ghcr.io/acme/orders-api
  tag: "1.8.3"
  digest: ""   # set per-env to pin
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 250m
    memory: 256Mi
config:
  logLevel: info
  downstreamTimeoutSeconds: 5
  downstreamUrl: https://api.staging.acme.com
```

The templated Deployment substitutes those values:

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "orders-api.fullname" . }}
  labels:
    {{- include "orders-api.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "orders-api.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "orders-api.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: orders-api
          image: "{{ .Values.image.repository }}{{ if .Values.image.digest }}@{{ .Values.image.digest }}{{ else }}:{{ .Values.image.tag }}{{ end }}"
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          envFrom:
            - configMapRef:
                name: {{ include "orders-api.fullname" . }}-config
```

A few things are now visible that did not exist in the Kustomize version. The `{{ include "orders-api.labels" . }}` calls reference helper templates defined in `_helpers.tpl` — reusable snippets that keep the label block consistent across every object:

```yaml
# templates/_helpers.tpl
{{- define "orders-api.fullname" -}}
{{ .Release.Name }}-orders-api
{{- end -}}

{{- define "orders-api.labels" -}}
app.kubernetes.io/name: orders-api
app.kubernetes.io/part-of: commerce-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: orders-api-{{ .Chart.Version }}
{{- end -}}

{{- define "orders-api.selectorLabels" -}}
app.kubernetes.io/name: orders-api
{{- end -}}
```

The image line shows a Go-template *conditional*: if `.Values.image.digest` is set, reference the image by `@sha256:...`; otherwise fall back to `:tag`. This is the kind of logic Helm makes easy and Kustomize makes awkward, and it is genuinely useful — but it is also the first crack through which the over-templating disease enters, because once you *can* put an `if` in your YAML, the temptation to put an `if` on *everything* is real.

The per-environment values file is small, just like the Kustomize overlay:

```yaml
# values-prod.yaml
replicaCount: 5
image:
  digest: sha256:9c4e0b...   # pin the promoted artifact
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: "1"
    memory: 1Gi
config:
  logLevel: info
  downstreamTimeoutSeconds: 15
  downstreamUrl: https://api.internal.acme.com
```

And you install or upgrade by layering values files, with the rightmost winning:

```bash
# Render to stdout to SEE the YAML before deploying — do this every time
helm template orders-api ./orders-api -f values-prod.yaml

# Install or upgrade idempotently (the standard production invocation)
helm upgrade --install orders-api ./orders-api \
  --namespace orders-prod --create-namespace \
  -f values-prod.yaml

# Inspect release history
helm history orders-api -n orders-prod

# Roll back to the previous revision in one command
helm rollback orders-api 0 -n orders-prod
```

### Helm as a package manager — the part Kustomize cannot do

Templating is half of Helm. The other half, and the half that actually justifies its existence for many teams, is **packaging**. A chart is a versioned artifact you can `helm package` into a `.tgz`, push to a chart repository (or, increasingly, store as an OCI artifact in the same registry that holds your images), and `helm install` from anywhere. This is how essentially every piece of third-party Kubernetes software is distributed: ingress controllers, databases, monitoring stacks, cert managers — you `helm install ingress-nginx ingress-nginx/ingress-nginx -f my-values.yaml` and you get a tested, versioned package with a defined value surface you are meant to customize. You did not write that chart. You do not want to fork and patch hundreds of its files. You want to set a dozen values and install it. Helm's value surface is exactly the right interface for that.

Charts also support **dependencies** — a chart can declare subcharts in `Chart.yaml`, so an umbrella chart for your platform can pull in the database chart, the cache chart, and the message-queue chart, each with its own pinned version. And every `helm upgrade` creates a new **revision** with the full rendered manifest stored as release state, so `helm rollback` is a real, atomic operation: it re-applies the exact manifests from a previous revision. Kustomize has no concept of a release or a revision; it renders YAML and hands it to `kubectl`, and rolling back means `git revert` and re-apply, which works but is not the same first-class operation. If your need is "ship this app to other people, with versioning and rollback," Helm is the answer and it is not close. The danger — and there is always a danger — is templating the *whole world*, which we will dissect shortly.

### Values precedence: the rule you will trip on once

There is one more piece of Helm mechanics that deserves explicit treatment because it surprises people at exactly the wrong moment, during a production deploy: **how Helm decides which value wins when several sources set the same key.** A real deploy often layers values from multiple places — the chart's `values.yaml`, a base `values-common.yaml`, an environment `values-prod.yaml`, and sometimes a last-minute `--set replicaCount=8` on the command line. When two of them set `replicaCount`, which one applies? The rule is a strict precedence order, lowest to highest:

```bash
# Lowest precedence first; each later source overrides the earlier
# 1. The chart's own values.yaml (defaults)
# 2. A parent chart's values for a subchart
# 3. Each -f / --values file, in the order given (later files win)
# 4. --set / --set-string flags on the command line (highest)
helm upgrade --install orders-api ./orders-api \
  -f values-common.yaml \
  -f values-prod.yaml \
  --set image.digest=sha256:9c4e0b...
```

The two clauses that bite are "later `-f` files win" and "`--set` beats everything." If you put `values-prod.yaml` *before* `values-common.yaml` on the command line, the common file overrides your prod settings — the opposite of what you meant — and nothing errors; you just deploy the wrong replica count. And a stray `--set` in a CI script silently overrides the values file a reviewer carefully approved, which means the rendered diff the reviewer saw is *not* what deploys. The defensive rule mirrors everything else in this post: render with the *exact* invocation you will deploy with, including every `-f` in the same order and every `--set`, and diff *that*. A subtle gotcha within the gotcha: `--set` parses `a.b.c=v` into nested structure and treats commas as list separators, so a value containing a comma (a connection string, an annotation) needs `--set-string` or careful escaping, or it silently fragments. Prefer values files over `--set` for anything non-trivial precisely so the value is a reviewable line in Git rather than an argument buried in a pipeline script. Kustomize, having no merge-from-the-command-line mechanism, simply does not have this class of surprise — what is in the overlay files is what renders, full stop — which is one more small point on its readability side of the ledger.

## 4. Head to head: the trade-offs that actually decide it

With both models on the table, the comparison sharpens into a few axes that genuinely determine the choice. The honest summary is that the two tools win on *opposite* axes, which is exactly why mature teams often run both.

![A comparison matrix of Helm versus Kustomize across templating logic packaging and rollback readability of source and in-house environment variation showing each tool winning opposite axes](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-4.png)

| Axis | Helm | Kustomize |
| --- | --- | --- |
| **Templating language** | Full Go templates: conditionals, loops, helpers, functions. Powerful — but YAML-inside-strings is hard to read and debug, and whitespace control (`nindent`, `-` trimming) is its own headache. | None. Pure YAML transformation. No logic, which means no logic *bugs*, but heavy conditional or loop-shaped variation is awkward to express. |
| **Readability of source** | A template file is not valid YAML on its own; you cannot eyeball the output without rendering. Whitespace hell is real. | What you see is YAML. A base and overlay both read as normal manifests. Lowest cognitive load to review. |
| **Packaging & redistribution** | First-class: versioned charts, chart repos / OCI artifacts, subchart dependencies. This is how you *ship* a third-party app. | No packaging. It transforms manifests you already have; it does not distribute them. |
| **Rollback** | First-class: `helm rollback` re-applies a stored prior revision atomically. Release history is built in. | No release concept. Rollback = `git revert` + re-apply (works via GitOps, but not a built-in operation). |
| **In-house env variation** | Heavier: you carry a full chart and value surface even for a simple service. | Lightest: a 15-line overlay expresses dev. Ideal for "my own service, three environments." |
| **Learning curve** | Steeper: Go templates, scoping (`.`), `_helpers.tpl`, value precedence, whitespace functions. | Gentler: it is just patches on YAML, with a handful of transformers. |
| **Built into kubectl** | No — separate binary, though widely installed. | Yes — `kubectl apply -k` and `kubectl kustomize`. |

The two rows that matter most in practice are *packaging/rollback* and *readability*. If your problem is "I am distributing software for others to install, or I need versioned releases with one-command rollback," Helm wins on the axis that matters and the readability cost is worth paying. If your problem is "this is my own service and I just need it to differ a bit across dev, staging, and prod," Kustomize wins on the axis that matters — its source stays readable, reviewable plain YAML — and the lack of packaging is irrelevant because you are not packaging anything; you are deploying your own code through your own pipeline.

A crucial point that the table cannot capture: **the two can be combined.** Kustomize can post-process the output of Helm. You can `helm template` a third-party chart to get rendered YAML, then run that through a Kustomize overlay to apply your organization's labels, namespace, and a couple of local patches the chart's value surface does not expose. The pattern looks like this:

```bash
# Render a third-party chart, then patch it with your local overlay
helm template ingress-nginx ingress-nginx/ingress-nginx \
  -f base-values.yaml > rendered/ingress-nginx.yaml

# overlays/prod/kustomization.yaml references rendered/ingress-nginx.yaml
# plus your org's commonLabels and a small patch:
kustomize build overlays/prod | kubectl apply -f -
```

Argo CD even supports this natively with its Helm-then-Kustomize rendering option. This combination lets you use Helm for what it is good at (consuming a packaged third-party app) and Kustomize for what *it* is good at (applying your own readable, reviewable local variation) without forking the upstream chart. It is a genuinely common production pattern and worth knowing exists.

#### Worked example: the same app in three environments, both ways

Let me make the comparison concrete with the running `orders-api`, and count what each approach costs you. The requirement is identical in both: dev runs 1 replica with lean resources, debug logging, a 5-second downstream timeout and an `rc` image; staging runs 2 replicas; prod runs 5 replicas with `cpu: 500m`/`memory: 512Mi` requests, info logging, a 15-second timeout, the production downstream URL, and the image pinned to `sha256:9c4e0b...`.

**Kustomize tally.** One base directory of around 200 lines (Deployment, Service, base `kustomization.yaml`). Three overlay directories: dev is a 15-line `kustomization.yaml` (it inherits replicas and resources from the base, so no patch file at all); staging is about 20 lines with a one-field replica patch; prod is a 25-line `kustomization.yaml` plus a 16-line patch file for replicas and resources. Total environment-specific code: roughly **75 lines** across three overlays. To review the prod delta, you read the 25-line overlay and the 16-line patch — 41 lines — and you have seen *everything* prod does differently. The rendered output is one `kustomize build overlays/prod` away and is plain YAML.

**Helm tally.** One chart: `Chart.yaml` (~6 lines), `values.yaml` (~18 lines of defaults), and `templates/` (Deployment ~30 lines, Service ~12, ConfigMap ~10, `_helpers.tpl` ~18) — call it about 95 lines, but they are *templated* lines, not directly readable YAML. Three values files: dev ~12 lines, staging ~3 lines, prod ~14 lines — roughly **29 lines** of environment-specific values, slightly less than Kustomize. *But* to understand what prod actually produces, reading the 14-line `values-prod.yaml` is not enough; you also have to mentally execute the templates (does that image conditional fire? what does `nindent 12` do to the resources block?), or — the honest answer — you run `helm template -f values-prod.yaml` and read the rendered output, because the values file alone does not tell you the result.

So the per-environment *delta* is comparable in size (Helm a touch smaller), but the **readability of the source differs sharply**: the Kustomize source is the thing you deploy; the Helm source is a program that *generates* the thing you deploy, and you cannot trust your eyes on it — you must render. That is the trade. For three environments of your own service, Kustomize's "what you see is what you ship" usually wins. The moment you need to *distribute* `orders-api` as a chart other teams install, or you want `helm rollback orders-api 3`, the Helm side's packaging tips the balance back. Same app, same result, different cost structure. Render both before you believe either — which brings us to the anti-patterns, because both tools have a failure mode that comes from using too much of their power.

## 5. The mechanics that bite: Go-template whitespace and Kustomize merge

Before the anti-patterns, it is worth spending real time on the day-to-day friction each tool generates, because that friction is where most engineers actually lose hours — and knowing the traps up front turns a frustrating afternoon into a five-minute fix. The two tools are frustrating in completely different ways, and the kind of frustration tells you something about the kind of mess each one tends toward.

Helm's friction is **whitespace and scope.** Go templates were not designed for YAML — they are a general text-templating engine, and YAML is whitespace-significant in a way that text templating is fundamentally hostile to. The result is a small grammar of trimming and indentation operators you must internalize or your renders will be subtly broken. The `{{-` and `-}}` markers trim whitespace to the left and right of a directive, which you need so that a `{{- if ... }}` on its own line does not leave a blank line in the output. The `nindent N` function indents a multi-line block by N spaces *and* prepends a newline; `indent N` does the same without the leading newline. Get the count wrong by two spaces and you produce YAML where a field lands under the wrong parent — and YAML will often *accept* it as valid, just with the wrong structure, which is exactly the silent-failure mode that makes these bugs expensive. Consider the resources block:

```yaml
# Correct: nindent 12 puts the resources map under the container's level
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

If you wrote `nindent 10` there, the `requests`/`limits` map would render two spaces too shallow, landing as a sibling of `resources:` rather than its child — invalid, and the error message Helm gives you (`error converting YAML to JSON`) points at a line number that is *after* the real problem, because the parser only notices once the structure is irrecoverable. The fix is always to render and look (`helm template ... | less`), find the misindented block in the *output*, and walk back to the `nindent` that produced it. There is no shortcut; you learn to read rendered YAML and reverse-engineer the template.

The scope rule is the other classic trip. Inside a `range` (Helm's loop) or a `with` block, the `.` (the root context) is *rebound* to the current item, so `.Values` no longer resolves — you have to reach back to the global context with `$`. This bites everyone once:

```yaml
# Inside a range, `.` is the list item; reach the root values with $
          env:
            {{- range .Values.extraEnv }}
            - name: {{ .name }}
              value: {{ .value | quote }}
            {{- end }}
            # WRONG inside the range: {{ .Values.config.logLevel }} — `.` is the item now
            # RIGHT: {{ $.Values.config.logLevel }}
```

These are not deep concepts, but they are sharp edges, and a chart accumulates dozens of them. Every one is a place a bug can hide, and none of them is caught by anything except rendering the output and reading it. That is the recurring theme: **Helm's power is real, and its tax is that the source is not the thing you deploy, so you cannot trust your eyes on the source.**

Kustomize's friction is different and, honestly, milder — but it has its own surprises, almost all of them about **how the strategic-merge patch decides what to merge.** The merge is structural: for maps it merges keys; for lists it depends on a *merge key*. Lists of containers merge by the `name` field, which is why patching one container by name works cleanly. But not every list has a merge key Kustomize knows about, and when it does not, the patch *replaces* the whole list rather than merging into it. The classic surprise is `env:` inside a container — environment variables are a list, and a naive strategic-merge patch that adds one env var can, depending on the resource and Kustomize version, replace the *entire* env list, silently dropping the others. The defensive move is to know which lists merge by key and which do not, and to reach for a JSON 6902 patch when you need surgical control:

```yaml
# A JSON 6902 patch: surgically append one env var without touching the rest
# overlays/prod/add-env.yaml is referenced under patches: with target:
- op: add
  path: /spec/template/spec/containers/0/env/-
  value:
    name: FEATURE_NEW_CHECKOUT
    value: "true"
```

The `-` at the end of the path means "append to the list," and the explicit `op: add` against a precise JSON pointer leaves everything else untouched. JSON patches are uglier to read than strategic-merge patches, but they are *unambiguous*, and that unambiguity is exactly what you want for the handful of changes where strategic-merge's cleverness works against you. The general rule: strategic-merge for the common case (it is readable and merges sensibly), JSON 6902 for the surgical case (removing a field, editing one list element by index, appending without replacing).

| Friction | Helm | Kustomize |
| --- | --- | --- |
| **Most common bug** | Wrong `nindent` count → misindented YAML, accepted-but-wrong structure | Strategic-merge replacing a whole list instead of merging it |
| **The trap** | Scope rebinding inside `range`/`with` — `.Values` stops resolving, need `$` | A patch with no merge key silently drops the rest of the list |
| **The fix** | Render and read the output; reverse-engineer from the misindented line | Switch that one change to a JSON 6902 patch for surgical control |
| **Error quality** | Points at a line *after* the real problem; cryptic `converting YAML to JSON` | Usually clear, but a silently-dropped list element errors only at runtime |
| **Underlying cause** | Go templates are text tooling forced onto whitespace-significant YAML | Strategic-merge semantics vary by field and Kustomize version |

Notice the asymmetry the table exposes. Helm's bugs are about *generating* YAML and are caught only by rendering. Kustomize's bugs are about *merging* YAML and are usually caught by rendering too — but its render is always plain YAML you can read directly, whereas Helm's render is the only window you have into what your template even does. Both tools, then, converge on the same survival rule, which is the whole point of the next two sections: **render the output, and review the output, every time.** Now, with the day-to-day friction understood, the larger structural failure modes.

## 6. The over-templated chart and the over-engineered overlay

Every powerful tool has a way of being over-used, and these two have opposite ones. Helm's is the **over-templated chart**: a chart so thoroughly parameterized that every line is a `{{ }}` expression, every value is a knob, and nobody — including the person who wrote it — can tell you what YAML actually comes out without rendering it and squinting. Kustomize's is the **twenty-overlays-deep tree**: a base that is patched by a regional overlay that is patched by an environment overlay that is patched by a tenant overlay, until tracing where a single value comes from requires opening six files.

![A side by side comparison of an over-templated 600 line chart where a bug hides in template logic versus a thin values surface with sane defaults that renders in CI and stays reviewable](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-5.png)

The over-templated chart is the more common and more dangerous of the two, so let me describe how it happens and why it bites. It starts innocently. Someone wants the chart to be flexible, so they make the replica count a value. Reasonable. Then the resources. Fine. Then someone needs to optionally add a sidecar, so a `{{ if .Values.sidecar.enabled }}` appears. Then a loop over a list of extra environment variables. Then a conditional that picks one of three probe configurations. Then helpers calling helpers. A year later the `deployment.yaml` template is 600 lines, ninety percent of it is template directives, and the actual Kubernetes objects are buried under control flow. The chart has become a small, untyped, untested *program* that emits YAML — and like any program, it has bugs. The difference is that this program has no compiler, no type checker, and runs only at deploy time.

Here is what an over-templated fragment looks like, and the bug hiding in it:

```yaml
# templates/deployment.yaml — over-templated; a bug is hiding here
{{- if .Values.autoscaling.enabled }}
{{- if not .Values.replicaCount }}
  replicas: {{ .Values.autoscaling.minReplicas | default 2 }}
{{- end }}
{{- else }}
  replicas: {{ .Values.replicaCount | default 1 }}
{{- end }}
```

Read that carefully. The intent is "if autoscaling is on, let the HPA manage replicas; otherwise use `replicaCount`." But the inner condition is `if not .Values.replicaCount` — and in Go templates, the integer `0` is falsy, and so is an *unset* value, but a value of `5` is truthy. The author meant to check whether autoscaling should set a floor, but the logic is tangled: when autoscaling is enabled *and* `replicaCount` happens to be set to a non-zero number, the `replicas` field is omitted entirely, which is correct for HPA — but when someone sets `autoscaling.enabled: true` and forgets to remove their `replicaCount: 1` override, the chart silently produces a Deployment with no `replicas` field, the HPA's `minReplicas` defaults somewhere it should not, and prod comes up with the wrong floor. Nobody catches it in review because nobody can read the rendered output of 600 lines of this in their head. The bug ships. This is not hypothetical; logic bugs in chart templates that only manifest for specific value combinations are one of the most common ways a Helm-based deploy goes wrong, precisely because the template is code that nobody tests like code.

The rescue is not "abandon Helm." It is **shrink the value surface to what actually varies, and render the chart in CI so the template's bugs surface as a diff before they reach a cluster.** The fix for the fragment above is to delete the cleverness:

```yaml
# templates/deployment.yaml — thin and honest
{{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
{{- end }}
```

One conditional, no nested negation, no `default` magic: if autoscaling is off, set replicas from a value; if it is on, omit the field and let the HPA own it. And `replicaCount` is simply *required* in the values when autoscaling is off — a `fail` guard in a helper makes that explicit rather than silently defaulting. The broader discipline is a **thin values surface**: expose the dozen things that genuinely differ per environment (replicas, resources, image, the handful of real config values) and *hard-code everything else in the template as a plain, readable line.* A reviewer reading your `values-prod.yaml` should see the entire story of how prod differs. If your value surface has eighty knobs, nobody knows which twelve matter, and the chart is back to being unreadable.

The Kustomize over-engineering failure is the mirror image: too many layers of overlay, each making the next harder to reason about. Kustomize *does* support composing overlays (an overlay can have a base that is itself an overlay), and that composition is genuinely useful for, say, a `base` → `region-overlay` → `cluster-overlay` hierarchy in a large multi-cluster fleet. But applied to a three-environment single service, a deep overlay tree is pure overhead — you now open four files to answer "what is the prod replica count," when a flat base-plus-three-overlays would have answered it in one. The rule for both tools is the same: **match the structure's depth to the problem's actual depth.** Three environments of one service is a shallow problem; solve it shallowly. A 200-cluster fleet with regional and per-tenant variation is a deep problem; the extra structure earns its keep there and only there.

#### Worked example: rescuing a 600-line chart and the change-fail-rate payoff

A team I worked with had inherited a chart exactly like the one above: 600 lines across three template files, roughly 70 distinct values, and a wiki page nobody trusted that tried to document them. Their change-failure rate on config changes — deploys that had to be rolled back or hotfixed — was running near **22%**. Better than one config deploy in five went wrong, and the post-incident reviews kept landing on the same root cause: "the rendered manifest was not what we expected." The template did something subtle with a value combination nobody had in their head.

The rescue took two moves. First, **render-in-CI**: every pull request that touched the chart or any values file ran `helm template` for all three environments and posted the rendered diff into the PR. Reviewers stopped approving template changes and started approving *YAML* changes — they could see that bumping the timeout value produced exactly one changed line in the rendered ConfigMap and nothing else, or that an innocent-looking helper edit had silently changed the labels on every object. Second, **value-surface reduction**: they audited which of the 70 values were ever set to anything other than the default in any environment. The answer was *eleven*. The other 59 knobs were dead weight — flexibility for variation that did not exist. They deleted them, hard-coding the values into the templates as plain lines, and the templates shrank from 600 lines to about 180, most of it now ordinary readable YAML.

The measured result over the following quarter: config change-failure rate fell from about **22% to roughly 6%**. The arithmetic of *why* is simple. If you deploy config changes 40 times a quarter and 22% fail, that is about 9 incidents; at 6% it is about 2 to 3 — you removed something like six rollbacks-and-hotfixes a quarter, each of which cost an hour of an engineer's attention and a chunk of error budget. The render-in-CI step caught a class of bug *before* deploy that previously could only be caught *in* prod, and the smaller surface meant fewer value combinations existed to go wrong in the first place. Note the honesty caveat: 22% and 6% are this team's numbers for *config-only* changes, not their overall change-fail rate, and your mileage depends on how template-heavy your chart was to begin with. The mechanism, though, is general — render before you ship, and expose only what varies. For why change-failure rate is one of the four metrics worth steering by, see the [intro map](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model).

## 7. Render for review: see the YAML, not the template

The single habit that prevents the most pain with both tools is also the simplest: **never deploy a manifest you have not rendered and read.** The whole risk that templating and patching introduce is the *gap* between the source you edit and the manifest the cluster receives. With three hand-copied files there was no gap — what you saw was what you applied — but with templates and overlays the source is a *generator*, and you cannot trust your eyes on a generator. So you close the gap by always running the generator and reviewing its output.

![A flow diagram showing a pull request change rendered by helm template and kustomize build then diffed against live state so reviewers approve the rendered YAML and template bugs are blocked before deploy](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-6.png)

Operationally this means two things. First, **make rendering a CI step** that runs on every PR touching manifests, and post the rendered diff as a comment or a check. Second, **review the rendered output, not the template change.** A reviewer who approves a `+1`/`-1` diff in a values file is not actually reviewing the deploy; they are trusting that the template does what they assume. A reviewer who sees the rendered diff — "this PR changes the prod ConfigMap's `DOWNSTREAM_TIMEOUT_SECONDS` from 5 to 15 and changes nothing else" — is reviewing the *actual change to the cluster*.

Here is a GitHub Actions job that renders both flavors and posts the diff. It renders the current PR's manifests and compares against the manifests rendered from the base branch, so the diff shown is precisely "what this PR changes about the deployed YAML":

```yaml
# .github/workflows/render-manifests.yaml
name: render-and-diff-manifests
on:
  pull_request:
    paths:
      - "deploy/**"
      - "charts/**"

jobs:
  render-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install tools
        run: |
          curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
          # kustomize ships in kubectl, but install standalone for `kustomize build`
          curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

      - name: Render PR head (Kustomize prod)
        run: kustomize build deploy/overlays/prod > /tmp/head-prod.yaml

      - name: Render base branch (Kustomize prod)
        run: |
          git worktree add /tmp/base "${{ github.event.pull_request.base.sha }}"
          (cd /tmp/base && kustomize build deploy/overlays/prod) > /tmp/base-prod.yaml

      - name: Render Helm chart for prod
        run: |
          helm template orders-api charts/orders-api \
            -f charts/orders-api/values-prod.yaml > /tmp/head-helm-prod.yaml

      - name: Post rendered diff
        run: |
          echo '### Rendered prod manifest diff' >> "$GITHUB_STEP_SUMMARY"
          echo '```diff' >> "$GITHUB_STEP_SUMMARY"
          diff -u /tmp/base-prod.yaml /tmp/head-prod.yaml >> "$GITHUB_STEP_SUMMARY" || true
          echo '```' >> "$GITHUB_STEP_SUMMARY"
```

You can go further and gate the merge on the diff being *intentional* — for instance, fail the job if the rendered prod manifest changes the image to a tag that is not a promoted digest, or if it changes a field a policy says must not change. That is policy-as-code territory (OPA/Conftest run against the rendered YAML), and it composes naturally: render first, then run policy on the concrete output rather than on the template. The principle holds across the series — the rendered manifest is the artifact you actually reason about, and it is also exactly what a GitOps controller wants. In a GitOps setup, the rendered manifests are committed (or rendered by the controller) and the controller reconciles the cluster to match them; reviewing the rendered diff in the PR *is* reviewing what GitOps will apply. For how a controller takes these rendered manifests and continuously reconciles the cluster to them, see the planned post on Argo CD and Flux in practice (`argo-cd-and-flux-in-practice`), which closes the loop from "rendered YAML in Git" to "live state in the cluster."

A small but important corollary: **render with the real values, including the per-environment ones.** It is common to render only the defaults and feel reassured, then discover that `values-prod.yaml` flips a conditional that changes the structure entirely. Render each environment's combination. The CI job above does prod; in practice you render dev, staging, and prod and diff each, because a change that is harmless in dev can be the one that breaks prod precisely because prod is the environment with the unusual values.

One practical caveat keeps the render-diff habit from becoming noise that reviewers learn to ignore: **normalize the output before you diff it.** A raw `helm template` or `kustomize build` can emit objects in a non-deterministic order, or include the content-hash suffix on a generated ConfigMap that changes on every config edit, so a one-line config change can produce a diff that *looks* like it touched twenty things. The fix is to pipe the rendered YAML through a normalizer — sort documents by kind and name, and either strip the hash suffix or accept that it changes — so the diff a reviewer sees is the *semantic* change and nothing else. A reviewer who trusts the diff reviews it carefully; a reviewer who has learned the diff is always noisy rubber-stamps it, and you have spent the CI minutes for nothing. The goal is a diff so clean that "this PR changes the prod timeout from 5 to 15 seconds and nothing else" is *literally* what the diff shows. When you hit that, code review of a deploy becomes as trustworthy as code review of the application code itself, which is the entire promise of "everything as code" — the manifest is reviewed like source because, rendered, it *is* readable source.

## 8. Versioning and per-environment image injection: wiring in build-once-promote-everywhere

The series' first principle is **build once, promote everywhere**: you build one image, test *that exact image*, and then move *that exact artifact* — identified by its immutable digest — through dev, staging, and prod. You never rebuild per environment, because a rebuild is a new artifact you have not tested, and "the artifact you tested is the artifact you ship" is the entire safety argument. Both Helm and Kustomize are where that principle meets the manifest: the image digest is the one value that must flow through the environments unchanged, while replicas and config vary around it.

![A timeline showing one image built once and identified by digest being promoted unchanged through dev staging and prod with only replica count varying so what was tested is what ships](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-8.png)

In Kustomize, the image transformer does this cleanly. The base names the image with no tag; each overlay pins the digest it is promoting:

```yaml
# Promotion is a one-line edit per overlay — pin the digest, not a moving tag
images:
  - name: ghcr.io/acme/orders-api
    digest: sha256:9c4e0b...   # the exact artifact that passed staging
```

Promoting from staging to prod is then a single, reviewable change: copy the digest that staging is running into the prod overlay's `images:` block, open a PR, look at the rendered diff (it should show *only* the image digest changing), merge. The digest is immutable, so there is no ambiguity about what `1.8.3` means six weeks from now when someone re-pushed that tag — the digest is the artifact, permanently. This is why pinning by digest rather than by mutable tag matters: a tag can move; a digest cannot. For the full treatment of why digests beat tags and how artifacts get versioned and promoted, see [build once, promote everywhere: artifacts and versioning](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) and [image registries, tagging, and promotion](/blog/software-development/ci-cd/image-registries-tagging-and-promotion).

In Helm, the same promotion is a values change, and the chart's image template should be written to take a digest:

```yaml
# values-prod.yaml — promote by setting the digest
image:
  digest: sha256:9c4e0b...   # overrides the default tag in the template
```

The CI pipeline can write this value automatically. After the image passes the staging gate, a promotion job updates the prod values (or prod overlay) with the digest and opens the PR:

```bash
# A promotion step in CI: pin the staging-tested digest into the prod overlay
DIGEST=$(crane digest ghcr.io/acme/orders-api:1.8.3)
yq -i ".images[0].digest = \"${DIGEST}\"" deploy/overlays/prod/kustomization.yaml
git commit -am "promote orders-api ${DIGEST} to prod" && gh pr create --fill
```

Now the chart version (or the overlay) and the image digest are *both* versioned in Git, and the promotion is an auditable commit. There is a subtlety worth pausing on: the **chart version** (`Chart.yaml`'s `version`) and the **app version** (the image) version independently. Bump the chart version when the *templates* change — a new field, a new probe, a structural change to the manifest. Bump the app version (and pin the digest) when the *image* changes. Keeping them separate means you can tell whether a deploy changed how the app is *shaped* (chart change) or what *code* is running (image change), which is exactly the distinction you want when you are staring at a release that broke and asking "what actually changed?" The promotion thread runs through the planned post on multi-environment promotion (`multi-environment-promotion-dev-staging-prod`), which covers the gates between dev, staging, and prod in full; here the point is narrower — whichever tool you pick, make the per-environment image a *pinned digest you promote*, never a tag you rebuild.

#### Worked example: lead-time decomposition for a promotion

Consider what a promotion actually costs in time, because it shows where templating helps. Suppose promoting `orders-api` from staging to prod involves: (a) deciding which digest to promote, (b) producing the prod manifest, (c) reviewing it, (d) applying it. With three hand-copied files, step (b) means hand-editing `prod.yaml` to point at the new image and *hoping* you did not also accidentally change something else, and step (c) means reviewing a full 220-line file to be sure. Call that 15 minutes of careful work plus the risk of a manual error. With a Kustomize overlay, step (b) is a one-line digest change generated by the CI promotion job in seconds, and step (c) is reviewing a rendered diff that shows exactly one line changed — call it 2 minutes. The promotion's human lead time drops from ~15 minutes to ~2, and — more importantly — the *variance* drops, because the mechanical render removes the "did I fat-finger another field?" risk that occasionally turned a 15-minute promotion into a 2-hour incident. If you promote 30 times a month, that is roughly 6.5 hours of engineer time saved a month, and one or two avoided "promotion changed something it should not have" incidents. The figure is illustrative, but the shape is real: templating turns promotion from a careful manual edit into a mechanical, reviewable, low-variance operation, which is precisely what you want at the riskiest step in the pipeline.

## 9. War story: the chart that fought back, and a drift that hid for months

Two real-flavored stories, because the failure modes here are worth seeing in the wild.

The first is the over-templated chart that hid a production-only bug. A platform team built an internal "golden chart" meant to deploy any of their fifty microservices — one chart to rule them all, with a value surface big enough to express every service's needs. By design it had to be flexible, so it grew conditionals: optional sidecars, optional init containers, three flavors of probe config selected by a value, a loop building env vars from a map, volume mounts assembled conditionally. It worked. Forty-nine services deployed through it fine. The fiftieth set a particular combination — `affinity` rules *and* a custom `securityContext` *and* an init container — that no other service used together, and a whitespace bug in the template (a missing `nindent` on the init-container block) produced a manifest where the init container's fields landed at the wrong indentation level and were silently dropped by the YAML parser. The Deployment applied cleanly. The init container that was supposed to run a database migration before the app started simply *was not there*. The app booted against an un-migrated schema and started throwing 500s. It took three hours and a `helm template | diff` against the previous release to find that the rendered manifest was missing a block the values clearly requested. The lesson the team took: render every service's manifest in CI and diff against the last good render, *especially* for the services with unusual value combinations, because the bug lived in a combination only one service exercised — exactly the case a defaults-only render would never catch.

There is a broader, public-record version of this same failure that is worth naming, because it shows the stakes scale with the blast radius. The most famous deploy disaster in finance, the Knight Capital incident of 2012, was at its core a *deployment-consistency* failure: a new piece of code was rolled out to eight servers, but one of the eight did not get the new deploy and kept running stale code that reused a flag whose meaning had changed. The fleet was supposed to be identical; one node had drifted from the other seven. In forty-five minutes the inconsistency generated millions of erroneous orders and roughly \$440M in losses, and the firm did not survive it. The mechanism — *the nodes were supposed to be the same and one was not* — is exactly the drift this whole post is about, just at a scale where it ends a company instead of paging an engineer. The lesson generalizes directly to manifests: when "what is deployed" is reconstructed by hand or applied unevenly, you are one missed copy away from a fleet that disagrees with itself, and the disagreement is invisible until it is catastrophic. A single rendered source of truth, applied identically everywhere, is not a tidiness preference; it is the control that makes "all the nodes are running the same thing" a property you can *guarantee* rather than hope for.

The second story is a GitOps drift that the manual-copy anti-pattern caused and Kustomize would have prevented. A team had a staging and a prod manifest, hand-maintained, and used a GitOps controller to apply each to its cluster. Someone, debugging a prod issue at 2am, ran `kubectl edit deployment` in prod to bump a memory limit and get the pods to stop OOM-killing. It worked, the incident closed, everyone went back to bed — and nobody updated the Git manifest. For a few weeks the GitOps controller and the live cluster disagreed: the controller's source said the old limit, the cluster had the new one. If the controller had self-heal on, it would have reverted the 2am fix at the next reconcile and re-triggered the OOMs; if self-heal was off, the drift just sat there until the next deploy, which re-applied the old limit and brought the OOMs back at the worst possible moment. The root cause was that the prod manifest was a *hand-maintained copy* — the 2am operator had no easy, reviewable way to encode "prod needs more memory" because doing it right meant editing a 220-line file under incident pressure, so they reached for `kubectl edit` instead. With a Kustomize overlay, the fix is a one-line patch to `overlays/prod/replicas-and-resources.yaml`, a 2-minute PR, and the rendered diff shows exactly the memory bump — encodable *correctly* even at 2am. Drift is often a symptom of the *correct* path being too expensive under pressure; cheapen the correct path and the drift stops happening. For how GitOps controllers detect and heal drift, and why pull-based reconciliation is the safer model, see the SRE treatment in [self-healing systems and their traps](/blog/software-development/site-reliability-engineering/self-healing-systems-and-their-traps) and the planned Argo CD post.

## 10. How to reach for this (and when not to)

The decision is genuinely not "Helm vs Kustomize" in the abstract; it is "what am I trying to do with these manifests," and the answer falls out of two questions.

![A decision tree asking what you are deploying that routes shipping to others or needing rollback toward Helm your own service env variation toward Kustomize and a third party chart with local patches toward combining both](/imgs/blogs/helm-vs-kustomize-templating-your-manifests-7.png)

**Reach for Helm when** you are distributing an application for others to install (you are *publishing* a chart), or you genuinely need its packaging and rollback features — versioned releases, `helm rollback`, subchart dependencies, a defined value surface as the customization interface. Installing third-party software (ingress controllers, databases, observability stacks) is almost always Helm, because that is how those things are shipped and you do not want to maintain a fork. If "give me a one-command, atomic rollback to revision 3" is a requirement, Helm gives it to you natively.

**Reach for Kustomize when** the manifests are your own service and you mostly need environment variation without duplication. Its source stays as readable plain YAML, it is built into `kubectl`, it has no templating language to learn or to bug out, and a 15-line overlay expresses an environment. For "my service, three environments, deployed through my own GitOps pipeline," Kustomize is the lighter, more legible choice, and the lack of packaging does not cost you anything because you are not packaging.

**Combine them when** you consume a third-party Helm chart but need local patches its value surface does not expose — `helm template` the chart, then Kustomize-overlay your org's labels, namespace, and small patches on top. Many mature platform teams run *both* deliberately: Helm for the third-party world, Kustomize for in-house services, and the combination for the seam between them.

Now the honest "when not to," because every practice has a cost:

- **Do not template at all for a single environment.** If you deploy one service to one environment, three hand-written manifests are *fine* — there is nothing to keep in sync, no drift to prevent. Templating earns its keep when variation exists. Introducing Helm to deploy one Deployment to one cluster is pure ceremony.
- **Do not build a 70-knob mega-chart.** A value surface should expose what *varies*, not everything that *could* vary. If a value is the same in every environment, hard-code it in the template as a plain readable line. The over-templated chart is worse than the copy-paste it replaced, because at least the copies were readable YAML.
- **Do not stack overlays twenty deep** for a problem that is shallow. Three environments of one service is base-plus-three-flat-overlays. Save the deep overlay hierarchy for the genuinely deep problem (a large multi-cluster, multi-region, multi-tenant fleet) where the layers map to real structure.
- **Do not skip the render-in-CI step**, whichever tool you pick. The entire risk of both tools is the gap between source and output; closing it costs one CI job and pays for itself the first time it catches a whitespace bug before prod.
- **Do not let `kubectl edit` become the prod fix path.** If the correct path (a patch + PR + rendered diff) is more expensive than `kubectl edit`, people will reach for `kubectl edit`, and you will get drift. Make the correct path cheap — a one-line overlay patch — and the drift stops.

## 11. Key takeaways

- **The real problem is per-environment variation without duplication.** Three hand-copied manifests do not have a style problem; they have a *drift* problem that silently generates incidents when a fix lands in two of three environments.
- **Kustomize patches a base with overlays — no templating language.** What you write is YAML; what `kustomize build` emits is YAML; the rendering is mechanical, so what you see is what you ship. Lightest and most readable for your own service across environments.
- **Helm templates with a Go engine and packages the result.** Its real edge is packaging: versioned charts, chart repos, subchart dependencies, and a first-class `helm rollback`. That is how you *ship* a third-party app and why you reach for it there.
- **The two win on opposite axes**, so combining them (`helm template | kustomize`) is a legitimate, common pattern, not a hack — Helm for the third-party world, Kustomize for the local patch.
- **The over-templated chart is the dominant Helm failure.** When every line is a `{{ }}` knob, the chart becomes an untested program that emits YAML, and config bugs hide in template logic. Shrink the value surface to what actually varies; hard-code the rest.
- **Always render before you deploy, and review the rendered output, not the template.** Make `helm template` / `kustomize build` a CI step that posts the rendered diff; reviewers should approve concrete YAML changes, not opaque template changes.
- **Pin the image by digest and promote it unchanged.** Wire build-once-promote-everywhere into the manifest via the image transformer (Kustomize) or a digest value (Helm); promotion becomes a one-line, reviewable change with a clean rendered diff.
- **Match the structure's depth to the problem's depth.** Three environments of one service is shallow — solve it shallowly. Deep overlay trees and giant value surfaces are overhead unless a genuinely deep, multi-cluster problem earns them.

## Further reading

- [From commit to production: the CI/CD model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series intro and the `commit → build → test → package → deploy → operate` spine these tools sit on.
- [Kubernetes for delivery: the objects that matter](/blog/software-development/ci-cd/kubernetes-for-delivery-the-objects-that-matter) — the Deployment, Service, ConfigMap, and Secret you are templating here.
- [Configuration and secrets in Kubernetes](/blog/software-development/ci-cd/configuration-and-secrets-in-kubernetes) — how the config and secrets these tools generate flow through the cluster, and why the configMapGenerator content-hash trick matters.
- [Build once, promote everywhere: artifacts and versioning](/blog/software-development/ci-cd/build-once-promote-everywhere-artifacts-and-versioning) and [image registries, tagging, and promotion](/blog/software-development/ci-cd/image-registries-tagging-and-promotion) — why you pin by digest and promote one artifact rather than rebuild per environment.
- Multi-environment promotion across dev, staging, and prod (`multi-environment-promotion-dev-staging-prod`, planned) and Argo CD and Flux in practice (`argo-cd-and-flux-in-practice`, planned) — how the rendered manifests get promoted through gates and reconciled onto the cluster by a GitOps controller.
- The official [Helm documentation](https://helm.sh/docs/) (charts, values, the template guide, and `helm rollback`) and the [Kustomize documentation](https://kubectl.docs.kubernetes.io/references/kustomize/) (transformers, generators, and patches) — both worth reading directly; the template/transformer reference pages repay the time.
- [The Twelve-Factor App](https://12factor.net/) — factor III (config in the environment) is the principle underneath all of this per-environment variation.
